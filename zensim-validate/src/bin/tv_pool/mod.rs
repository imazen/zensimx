//! `--tv-pool`: a within-ladder hinge over an external, label-free feature pool, in public builds.
//!
//! The pool's rows are raw feature vectors of the declared read set (`--derived-inputs`); they pass through the
//! same derived-input transform as every training row and are standardized with the run's scaler. Each pair names a
//! worse and a better row and a weight class; the hinge, its schedule and its RNG stream are the E36 pool's
//! (`mlp_train/e36_tv.rs`), with the class weights, `--tv-apply-every`, `--tv-batch` and `--tv-margin` taken from
//! the manifest and command line instead of E36's registered constants. No label or teacher value is read.
use arrow::array::{Array, Float32Array, ListArray, UInt8Array, UInt64Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Deserialize;
use std::{
    collections::HashSet,
    fs::File,
    path::{Path, PathBuf},
};
use zensim_validate::{
    derived_inputs::Fx1Declaration,
    mlp_train::{CompactRows, TvRegularizer},
    train_manifest::sha256_file,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Bound {
    path: PathBuf,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    features: Bound,
    pairs: Bound,
    /// Raw feature ids of each row's values, in order: the declaration's read set.
    feature_ids: Vec<usize>,
    /// Weight of each pair class (index = the pairs table's `class`).
    class_weights: Vec<f64>,
    feature_rows: usize,
    pair_rows: usize,
    #[serde(default)]
    provenance: serde_json::Value,
}

fn bound(b: &Bound) -> Result<(), String> {
    if sha256_file(&b.path).map_err(|e| e.to_string())? != b.sha256 {
        return Err(format!(
            "{}: digest differs from the manifest",
            b.path.display()
        ));
    }
    Ok(())
}

fn reader(
    path: &Path,
    columns: &[&str],
) -> Result<parquet::arrow::arrow_reader::ParquetRecordBatchReader, String> {
    let b = ParquetRecordBatchReaderBuilder::try_new(File::open(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let names: Vec<_> = b
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    if names != columns {
        return Err(format!(
            "{}: columns must be exactly {columns:?}",
            path.display()
        ));
    }
    b.with_batch_size(1024).build().map_err(|e| e.to_string())
}

/// Load and derive the pool. `weight`, `apply_every`, `batch` and `margin` are the run's `--tv-*` values.
pub(super) fn load(
    path: &Path,
    derived_path: &Path,
    raw_width: usize,
    weight: f64,
    apply_every: usize,
    batch: usize,
    margin: f64,
) -> Result<TvRegularizer, String> {
    let m: Manifest = serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if m.schema != "tv-pool-v1" {
        return Err("tv pool manifest schema must be tv-pool-v1".into());
    }
    bound(&m.features)?;
    bound(&m.pairs)?;
    if m.class_weights.is_empty()
        || m.class_weights.len() > 4
        || m.class_weights.iter().any(|w| !w.is_finite() || *w <= 0.0)
    {
        return Err("tv pool class weights: one to four finite positive values".into());
    }
    if !(weight > 0.0 && apply_every > 0 && batch > 0 && margin.is_finite() && margin >= 0.0) {
        return Err("--tv-pool needs --tv-weight > 0, --tv-apply-every > 0, --tv-batch > 0, --tv-margin >= 0".into());
    }
    let (derived, _) = Fx1Declaration::load(derived_path)?;
    if m.feature_ids != derived.load_ids() {
        return Err("tv pool feature ids differ from the --derived-inputs read set".into());
    }
    let kept: Vec<u32> = m.feature_ids.iter().map(|&x| x as u32).collect();
    let mut features = Vec::with_capacity(m.feature_rows);
    for batch in reader(&m.features.path, &["row_idx", "features"])? {
        let b = batch.map_err(|e| e.to_string())?;
        if b.columns().iter().any(|a| a.null_count() != 0) {
            return Err("tv pool null feature column".into());
        }
        let ids = b
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or("row_idx must be u64")?;
        let values = b
            .column(1)
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or("features must be a list")?;
        for i in 0..b.num_rows() {
            if ids.value(i) as usize != features.len() {
                return Err("tv pool rows must be numbered 0.. in order".into());
            }
            let v = values.value(i);
            let v = v
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or("features must be f32")?;
            if v.null_count() != 0
                || v.len() != kept.len()
                || !v.values().iter().all(|x| x.is_finite())
            {
                return Err("tv pool row lacks a finite value for every read-set id".into());
            }
            let mut compact = CompactRows {
                data: v.values().to_vec(),
                n_rows: 1,
                n_features: raw_width,
                kept: kept.clone(),
                derived: None,
            };
            derived.extend_compact(&mut compact, raw_width)?;
            if !compact.data.iter().all(|x| x.is_finite()) {
                return Err("tv pool derived input not finite".into());
            }
            if compact.kept.iter().map(|&x| x as usize).collect::<Vec<_>>()
                != derived.kept_columns(raw_width)
            {
                return Err("tv pool derived column order".into());
            }
            features.push(compact.data.iter().map(|&x| f64::from(x)).collect());
        }
    }
    let mut pairs = Vec::with_capacity(m.pair_rows);
    let mut classes = Vec::with_capacity(m.pair_rows);
    let mut seen = HashSet::new();
    for batch in reader(&m.pairs.path, &["worse_idx", "better_idx", "class"])? {
        let b = batch.map_err(|e| e.to_string())?;
        if b.columns().iter().any(|a| a.null_count() != 0) {
            return Err("tv pool null pair column".into());
        }
        let lo = b
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or("worse_idx must be u64")?;
        let hi = b
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or("better_idx must be u64")?;
        let class = b
            .column(2)
            .as_any()
            .downcast_ref::<UInt8Array>()
            .ok_or("class must be u8")?;
        for i in 0..b.num_rows() {
            let (l, h, c) = (lo.value(i) as usize, hi.value(i) as usize, class.value(i));
            if l >= features.len()
                || h >= features.len()
                || l == h
                || c as usize >= m.class_weights.len()
                || !seen.insert((l.min(h), l.max(h)))
            {
                return Err("tv pool pair out of range, self, duplicate or unknown class".into());
            }
            pairs.push((l, h));
            classes.push(c);
        }
    }
    if features.len() != m.feature_rows || pairs.len() != m.pair_rows || pairs.is_empty() {
        return Err("tv pool row or pair count differs from the manifest".into());
    }
    let mut band_weights = [0.0; 4];
    band_weights[..m.class_weights.len()].copy_from_slice(&m.class_weights);
    Ok(TvRegularizer {
        pairs,
        features,
        weight,
        apply_every,
        batch,
        band_id: Some(classes),
        band_weights: Some(band_weights),
        margin,
        pool: true,
    })
}

/// What `zentrain.repro` records about the pool.
pub(super) fn receipt(
    path: &Path,
    pool: &TvRegularizer,
    sample_seed: u64,
) -> Result<serde_json::Value, String> {
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    Ok(serde_json::json!({"schema":"tv-pool-effective-v1",
        "manifest_sha256":sha256_file(path).map_err(|e|e.to_string())?,"manifest":manifest,
        "lambda":pool.weight,"margin":pool.margin,"apply_every":pool.apply_every,"batch":pool.batch,
        "class_weights":pool.band_weights,"feature_rows":pool.features.len(),"pairs":pool.pairs.len(),
        "tv_rng_seed":sample_seed^0xe360_7476_726e_6700_u64,
        "hinge":"max(0, ladder_sign*(raw(better)-raw(worse)) + margin), scaled lambda*class_weight/(batch*mean class weight)"}))
}
