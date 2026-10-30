//! Strict feature-only E36 ingress. No human group/target construction.
use arrow::array::{
    Array, Float32Array, ListArray, StringArray, UInt8Array, UInt32Array, UInt64Array,
};
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
    protocol_build: String,
    labels_read: bool,
    fits: usize,
    complete: bool,
    split: String,
    zero_dropped_pairs: bool,
    mandatory_matrix_complete: bool,
    required_support: Bound,
    references: Bound,
    source_budget: Bound,
    #[serde(default)]
    source_budget_amendment: Option<Bound>,
    cohort: Bound,
    runtime: Bound,
    features: Bound,
    pairs: Bound,
    edge_proof: Bound,
    derived: Bound,
    feature_ids: Vec<usize>,
    raw_width: usize,
    feature_rows: usize,
    pair_rows: usize,
    source_budget_sha256: String,
    generation_runtime_sha256: String,
}
fn bound(b: &Bound) -> Result<(), String> {
    super::group_admission::safe(&b.path)?;
    if sha256_file(&b.path).map_err(|e| e.to_string())? != b.sha256 {
        return Err("E36 bound file digest mismatch".into());
    }
    Ok(())
}

fn source_budget_identity(
    reference: &str,
    budget: &str,
    runtime: &str,
    amendment: Option<&str>,
) -> bool {
    match amendment {
        None => reference == budget,
        // The registered 4096-only 16-to-6 cut preserves the immutable
        // generation runtime/reference index. Its exact complete budgets and
        // authorization are retained; this is not an arbitrary hash waiver.
        Some(amendment) => {
            reference == "324762fbf74b72fd6ebf3a53524015af07c0a175ca53b2f4ea9f63d6be11d7ba"
                && budget == "f9172e051b2e91d6ce9da336df439b251ec5e2f1c452ebbbe729981b2db82b1e"
                && runtime == "35238d8ae522471267469e9f36fb3e133004d259ca4d37bb1dcf92296a6f4f8c"
                && amendment == "48019e0e09d54137e0fe35ce06243db3d9f0c703d3fcf1a95054833c39213dc9"
        }
    }
}

pub(super) fn fit_budget(
    epochs: usize,
    pairs: usize,
    contract: Option<&str>,
    smoke: Option<&str>,
    argv: Option<&str>,
) -> bool {
    if smoke.is_some_and(|value| !value.is_empty()) || contract == Some("local-smoke") {
        epochs == 2
            && pairs == 128
            && contract == Some("local-smoke")
            && smoke == Some("2:128")
            && argv.is_some_and(digest)
    } else {
        epochs == 120 && pairs == 50_000
    }
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
        return Err(
            "E36 unexpected columns; labels and teacher scalars forbidden in ingress".into(),
        );
    }
    b.with_batch_size(256).build().map_err(|e| e.to_string())
}
fn digest(x: &str) -> bool {
    x.len() == 64
        && x.bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

fn json(p: &Path) -> Result<serde_json::Value, String> {
    serde_json::from_slice(&std::fs::read(p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

pub(super) fn load(
    path: &Path,
    lambda: f64,
    expected_derived: &Path,
    raw_width: usize,
) -> Result<TvRegularizer, String> {
    if !cfg!(e36_tv) {
        return Err("E36 requires its explicit private build".into());
    }
    super::group_admission::safe(path)?;
    let m: Manifest = serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if m.schema != "e36-tv-feature-only-v1"
        || m.protocol_build != "e36_tv_v1"
        || m.labels_read
        || m.fits != 0
        || !m.complete
        || m.split != "fit"
        || !m.zero_dropped_pairs
        || !m.mandatory_matrix_complete
        || ![0., 1., 10.].contains(&lambda)
        || !digest(&m.source_budget_sha256)
        || !digest(&m.generation_runtime_sha256)
    {
        return Err("E36 manifest is not a completed registered fit-only feature pool".into());
    }
    for b in [
        &m.features,
        &m.pairs,
        &m.edge_proof,
        &m.derived,
        &m.required_support,
        &m.references,
        &m.source_budget,
        &m.cohort,
        &m.runtime,
    ] {
        bound(b)?;
    }
    let references = json(&m.references.path)?;
    let budget = json(&m.source_budget.path)?;
    let cohort = json(&m.cohort.path)?;
    let runtime = json(&m.runtime.path)?;
    if let Some(amendment) = &m.source_budget_amendment {
        bound(amendment)?;
    }
    if m.source_budget.sha256 != m.source_budget_sha256
        || m.runtime.sha256 != m.generation_runtime_sha256
        || !source_budget_identity(
            references["source_budget_sha256"].as_str().unwrap_or(""),
            &m.source_budget.sha256,
            &m.runtime.sha256,
            m.source_budget_amendment
                .as_ref()
                .map(|value| value.sha256.as_str()),
        )
        || budget["selection_sha256"] != m.cohort.sha256
        || cohort["labels_read"] != false
        || cohort["internal_overlap"]["admitted"] != true
        || runtime["fits"] != 0
        || runtime["references_sha256"] != m.references.sha256
    {
        return Err("E36 source/split/runtime binding failed".into());
    }
    let cohort_rows = cohort["rows"].as_array().ok_or("E36 cohort rows")?;
    let mut allowed = HashSet::new();
    let mut excluded = HashSet::new();
    let mut reference_sizes = std::collections::HashMap::new();
    for r in references["rows"]
        .as_array()
        .ok_or("E36 normalized reference index")?
    {
        let key = r["sha256"].as_str().ok_or("E36 reference SHA")?;
        let admitted = cohort_rows.iter().any(|c| {
            c["origin"] == r["origin"]
                && c["family"] == r["family"]
                && c["source_sha256"] == r["source_sha256"]
                && c["split"] == "fit"
                && r["split"] == "fit"
        });
        if admitted {
            allowed.insert(key.to_owned());
            reference_sizes.insert(
                key.to_owned(),
                r["size"].as_u64().ok_or("E36 reference size")?,
            );
        } else {
            excluded.insert(key.to_owned());
        }
    }
    if allowed.is_empty() || !allowed.is_disjoint(&excluded) {
        return Err("E36 ambiguous/empty fit reference admission".into());
    }
    if sha256_file(expected_derived).map_err(|e| e.to_string())? != m.derived.sha256
        || m.derived.sha256 != "a97a5fa63937ed8fade3c56fd7470765a031d1d309afc36e631f371a1796e648"
        || raw_width != m.raw_width
        || raw_width != 1853
    {
        return Err("E36 derived/raw layout differs from base recipe".into());
    }
    let support: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&m.required_support.path).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if support["schema"] != "e36-required-support-v1"
        || support["admitted"] != true
        || support["zero_support_required_strata"] != 0
        || support["features_sha256"] != m.features.sha256
        || support["pairs_sha256"] != m.pairs.sha256
        || support["edge_proof_sha256"] != m.edge_proof.sha256
        || support["source_budget_sha256"] != m.source_budget.sha256
        || support["references_sha256"] != m.references.sha256
        || support["cohort_sha256"] != m.cohort.sha256
        || support["runtime_sha256"] != m.runtime.sha256
        || support["derived_sha256"] != m.derived.sha256
        || support["feature_rows"] != m.feature_rows
        || support["pair_rows"] != m.pair_rows
        || support["teachers"] != serde_json::json!(["butteraugli_pnorm3", "cvvdp@standard_fhd"])
        || support["teacher_margins"] != serde_json::json!([0.01, 0.01])
    {
        return Err("E36 required consensus support is incomplete".into());
    }
    let matrix_binding: Bound =
        serde_json::from_value(support["whole_matrix"].clone()).map_err(|e| e.to_string())?;
    bound(&matrix_binding)?;
    let matrix = json(&matrix_binding.path)?;
    if matrix["schema"] != "e36-whole-matrix-completion-v1"
        || matrix["complete"] != true
        || matrix["runtime_sha256"] != m.runtime.sha256
        || matrix["source_budget_sha256"] != m.source_budget.sha256
        || !["fallback", "full"].contains(&matrix["matrix"].as_str().unwrap_or(""))
    {
        return Err("E36 source-budget matrix completion binding failed".into());
    }
    let source_receipts: HashSet<_> = matrix["outputs"]
        .as_array()
        .ok_or("E36 source outputs required")?
        .iter()
        .map(|v| v.as_str().ok_or("E36 source receipt digest"))
        .collect::<Result<_, _>>()?;
    let strata = support["required_strata"]
        .as_object()
        .ok_or("E36 required strata required")?;
    if source_receipts.is_empty()
        || source_receipts.iter().any(|s| !digest(s))
        || strata.is_empty()
        || strata.values().any(|s| {
            s["selected_references"].as_u64().is_none_or(|v| v == 0)
                || s["unique_preferences"].as_u64().is_none_or(|v| v == 0)
        })
    {
        return Err("E36 required matrix/support is empty or has a zero-support stratum".into());
    }
    let (derived, _) = Fx1Declaration::load(&m.derived.path)?;
    if m.feature_ids != derived.load_ids()
        || derived.direct.len() != 410
        || derived.products.len() != 410
    {
        return Err("E36 must consume the exact diff+frag-820 declaration".into());
    }
    let mut features = Vec::new();
    let mut row_keys = Vec::new();
    let mut keys = HashSet::new();
    for batch in reader(
        &m.features.path,
        &["row_idx", "reference_sha256", "encoded_sha256", "features"],
    )? {
        let b = batch.map_err(|e| e.to_string())?;
        if b.columns().iter().any(|a| a.null_count() != 0) {
            return Err("E36 null feature column".into());
        }
        let ids = b
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or("E36 row index type")?;
        let refs = b
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("E36 reference key type")?;
        let encoded = b
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("E36 encoded key type")?;
        let values = b
            .column(3)
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or("E36 features must be f32 lists")?;
        for i in 0..b.num_rows() {
            if usize::try_from(ids.value(i)).map_err(|e| e.to_string())? != features.len()
                || !digest(refs.value(i))
                || !allowed.contains(refs.value(i))
                || !digest(encoded.value(i))
                || !keys.insert((refs.value(i).to_owned(), encoded.value(i).to_owned()))
            {
                return Err("E36 noncontiguous or duplicate physical feature key".into());
            }
            row_keys.push((refs.value(i).to_owned(), encoded.value(i).to_owned()));
            let v = values.value(i);
            let v = v
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or("E36 feature dtype")?;
            if v.null_count() != 0
                || v.len() != m.feature_ids.len()
                || !v.values().iter().all(|x| x.is_finite())
            {
                return Err("E36 missing/nonfinite consumed features".into());
            }
            let mut compact = CompactRows {
                data: v.values().to_vec(),
                n_rows: 1,
                n_features: raw_width,
                kept: m.feature_ids.iter().map(|&x| x as u32).collect(),
                derived: None,
            };
            derived.extend_compact(&mut compact, raw_width)?;
            if !compact.data.iter().all(|x| x.is_finite()) {
                return Err("E36 nonfinite fx1 product".into());
            }
            if compact.kept.iter().map(|&x| x as usize).collect::<Vec<_>>()
                != derived.kept_columns(raw_width)
            {
                return Err("E36 transformed feature order".into());
            }
            features.push(compact.data.iter().map(|&x| f64::from(x)).collect());
        }
    }
    let mut proofs = reader(
        &m.edge_proof.path,
        &[
            "worse_idx",
            "better_idx",
            "reference_sha256",
            "codec",
            "size",
            "mode_key",
            "q_low",
            "q_high",
            "worse_encoded_sha256",
            "better_encoded_sha256",
            "consensus_receipt_sha256",
        ],
    )?;
    let mut pairs = Vec::new();
    let mut classes = Vec::new();
    let mut seen = HashSet::new();
    let mut used = HashSet::new();
    for batch in reader(
        &m.pairs.path,
        &["worse_idx", "better_idx", "pair_weight_class", "q_band"],
    )? {
        let b = batch.map_err(|e| e.to_string())?;
        if b.columns().iter().any(|a| a.null_count() != 0) {
            return Err("E36 null preference column".into());
        }
        let proof = proofs
            .next()
            .ok_or("E36 missing edge proof batch")?
            .map_err(|e| e.to_string())?;
        if proof.num_rows() != b.num_rows() || proof.columns().iter().any(|a| a.null_count() != 0) {
            return Err("E36 edge proof shape/null".into());
        }
        let integers = |i| {
            proof
                .column(i)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or("E36 proof ordinal type")
        };
        let strings = |i| {
            proof
                .column(i)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or("E36 proof metadata type")
        };
        let plo = integers(0)?;
        let phi = integers(1)?;
        let pref = strings(2)?;
        let codec = strings(3)?;
        let mode = strings(5)?;
        let worse_sha = strings(8)?;
        let better_sha = strings(9)?;
        let consensus = strings(10)?;
        let size = proof
            .column(4)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or("E36 proof size type")?;
        let qlo = proof
            .column(6)
            .as_any()
            .downcast_ref::<UInt8Array>()
            .ok_or("E36 proof q type")?;
        let qhi = proof
            .column(7)
            .as_any()
            .downcast_ref::<UInt8Array>()
            .ok_or("E36 proof q type")?;
        let lo = b
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or("E36 worse type")?;
        let hi = b
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or("E36 better type")?;
        let weight = b
            .column(2)
            .as_any()
            .downcast_ref::<UInt8Array>()
            .ok_or("E36 weight class type")?;
        let band = b
            .column(3)
            .as_any()
            .downcast_ref::<UInt8Array>()
            .ok_or("E36 q band type")?;
        for i in 0..b.num_rows() {
            let (lo, hi) = (
                usize::try_from(lo.value(i)).map_err(|e| e.to_string())?,
                usize::try_from(hi.value(i)).map_err(|e| e.to_string())?,
            );
            if lo >= features.len()
                || hi >= features.len()
                || lo == hi
                || weight.value(i) > 1
                || band.value(i) > 2
                || !seen.insert((lo.min(hi), lo.max(hi)))
            {
                return Err("E36 malformed/duplicate/out-of-range pair; no drops allowed".into());
            }
            let q: Vec<u8> = (0..=20).chain((22..=100).step_by(2)).collect();
            let a = q
                .iter()
                .position(|&v| v == qlo.value(i))
                .ok_or("E36 proof unregistered q")?;
            let b = q
                .iter()
                .position(|&v| v == qhi.value(i))
                .ok_or("E36 proof unregistered q")?;
            let c = codec.value(i);
            if plo.value(i) != lo as u64
                || phi.value(i) != hi as u64
                || row_keys[lo].0 != pref.value(i)
                || row_keys[hi].0 != pref.value(i)
                || row_keys[lo].1 != worse_sha.value(i)
                || row_keys[hi].1 != better_sha.value(i)
                || !["jpeg", "rav1e", "svt", "webp", "jxl"].contains(&c)
                || weight.value(i) != u8::from(!["jpeg", "rav1e"].contains(&c))
                || ![
                    32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024,
                    1280, 1536, 2048, 4096,
                ]
                .contains(&size.value(i))
                || reference_sizes.get(pref.value(i)) != Some(&u64::from(size.value(i)))
                || serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(mode.value(i))
                    .is_err()
                || !digest(consensus.value(i))
                || !source_receipts.contains(consensus.value(i))
                || b <= a
                || ![1, 2, 4, 8, 16].contains(&(b - a))
                || band.value(i)
                    != if qlo.value(i) <= 20 {
                        0
                    } else if qlo.value(i) <= 60 {
                        1
                    } else {
                        2
                    }
            {
                return Err(
                    "E36 edge proof reference/codec/mode/size/lag/weight/band binding failed"
                        .into(),
                );
            }
            pairs.push((lo, hi));
            classes.push(weight.value(i));
            used.insert(lo);
            used.insert(hi);
        }
    }
    if proofs.next().is_some() {
        return Err("E36 extra edge proof rows".into());
    }
    if features.len() != m.feature_rows
        || pairs.len() != m.pair_rows
        || pairs.is_empty()
        || used.len() != features.len()
    {
        return Err("E36 exact feature/pair retention failed".into());
    }
    Ok(TvRegularizer {
        features,
        pairs,
        weight: lambda,
        apply_every: 50,
        batch: 32,
        band_id: Some(classes),
        band_weights: Some([2., 1., 0., 0.]),
        margin: 0.05,
    })
}

pub(super) fn check_contract(manifest: &Path, path: &Path, expected: &str) -> Result<(), String> {
    super::group_admission::safe(path)?;
    if !digest(expected) || sha256_file(path).map_err(|e| e.to_string())? != expected {
        return Err("E36 contract hash differs from frozen fit argv".into());
    }
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if value["schema"] != "e36-tv-locked-contract-v1"
        || value["manifest_sha256"] != sha256_file(manifest).map_err(|e| e.to_string())?
        || value["protocol_build"] != "e36_tv_v1"
        || value["fits_released"] != false
    {
        return Err("E36 contract does not bind the registered pre-fit pool".into());
    }
    let semantic_binding: Bound =
        serde_json::from_value(value["semantic_export"].clone()).map_err(|e| e.to_string())?;
    let matrix_binding: Bound =
        serde_json::from_value(value["matrix_admission"].clone()).map_err(|e| e.to_string())?;
    bound(&semantic_binding)?;
    bound(&matrix_binding)?;
    let semantic = json(&semantic_binding.path)?;
    let m = json(manifest)?;
    if semantic["schema"] != "e36-semantic-export-admission-v1"
        || semantic["admitted"] != true
        || semantic["split"] != "fit"
        || semantic["labels_read"] != false
        || semantic["fits"] != 0
        || semantic["features"] != m["features"]
        || semantic["pairs"] != m["pairs"]
        || semantic["edge_proof"] != m["edge_proof"]
        || semantic["support"] != m["required_support"]
        || semantic["feature_rows"] != m["feature_rows"]
        || semantic["pair_rows"] != m["pair_rows"]
    {
        return Err("E36 successful native semantic export admission required".into());
    }
    let whole_binding: Bound =
        serde_json::from_value(semantic["whole_matrix"].clone()).map_err(|e| e.to_string())?;
    bound(&whole_binding)?;
    let whole = json(&whole_binding.path)?;
    if whole["admission"]["path"] != serde_json::json!(matrix_binding.path)
        || whole["admission"]["sha256"] != matrix_binding.sha256
    {
        return Err("E36 semantic export and deadline-qualified matrix differ".into());
    }
    let matrix = json(&matrix_binding.path)?;
    if matrix["schema"] != "e36-qualified-matrix-admission-v1"
        || matrix["complete"] != true
        || matrix["matrix"] != whole["matrix"]
        || matrix["outputs"] != whole["outputs"]
    {
        return Err("E36 deadline-qualified matrix admission required".into());
    }
    Ok(())
}

pub(super) fn receipt(
    path: &Path,
    pool: &TvRegularizer,
    sample_seed: u64,
    contract_sha: &str,
) -> Result<serde_json::Value, String> {
    use sha2::Digest;
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let source = [
        include_str!("mod.rs"),
        include_str!("../zensim_mlp_train.rs"),
        include_str!("../../mlp_train/mod.rs"),
        include_str!("../../mlp_train/e36_tv.rs"),
        include_str!("../../../build.rs"),
    ]
    .join("\n");
    let source_sha = sha2::Sha256::digest(source.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    let classes = pool.band_id.as_ref().ok_or("E36 classes missing")?;
    let mean = classes
        .iter()
        .map(|&c| if c == 0 { 2. } else { 1. })
        .sum::<f64>()
        / classes.len() as f64;
    Ok(
        serde_json::json!({"schema":"e36-effective-tv-v1","protocol_build":"e36_tv_v1","cfg_e36_tv":cfg!(e36_tv),
        "manifest_sha256":sha256_file(path).map_err(|e|e.to_string())?,"contract_sha256":contract_sha,"bound_inputs":manifest,
        "binary_sha256":sha256_file(&std::env::current_exe().map_err(|e|e.to_string())?).map_err(|e|e.to_string())?,
        "embedded_protocol_source_sha256":source_sha,"lambda":pool.weight,"raw_margin":0.05,"every_primary_pairs":50,
        "output_polarity":"score","raw_definition":"pin-g","improvement":"raw(better)-raw(worse)",
        "epoch_local_counter":true,"batch":32,"mean_pool_weight":mean,"pair_weight_class_values":[2,1],
        "tv_rng_seed":sample_seed^0xe360_7476_726e_6700_u64,"normalization":"lambda*w/(32*mean_pool_weight)",
        "optimizer":"pre-update gradient addition; one existing primary Adam boundary; no TV-only Adam"}),
    )
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    #[test]
    fn original_budget_and_exact_registered_amendment_are_distinct() {
        let original = "324762fbf74b72fd6ebf3a53524015af07c0a175ca53b2f4ea9f63d6be11d7ba";
        let effective = "f9172e051b2e91d6ce9da336df439b251ec5e2f1c452ebbbe729981b2db82b1e";
        let runtime = "35238d8ae522471267469e9f36fb3e133004d259ca4d37bb1dcf92296a6f4f8c";
        let amendment = "48019e0e09d54137e0fe35ce06243db3d9f0c703d3fcf1a95054833c39213dc9";
        assert!(source_budget_identity(original, original, runtime, None));
        assert!(!source_budget_identity(original, effective, runtime, None));
        assert!(source_budget_identity(
            original,
            effective,
            runtime,
            Some(amendment)
        ));
        for (r, b, t, a) in [
            (effective, effective, runtime, amendment),
            (original, original, runtime, amendment),
            (original, effective, original, amendment),
            (original, effective, runtime, effective),
        ] {
            assert!(!source_budget_identity(r, b, t, Some(a)));
        }
    }
    #[test]
    fn diagnostic_needs_native_classification_exact_budget_and_argv() {
        let argv = "a".repeat(64);
        assert!(fit_budget(120, 50000, None, None, None));
        assert!(!fit_budget(2, 128, None, None, None));
        assert!(fit_budget(
            2,
            128,
            Some("local-smoke"),
            Some("2:128"),
            Some(&argv)
        ));
        for (epochs, pairs, contract, marker, hash) in [
            (120, 50000, "local-smoke", "2:128", argv.as_str()),
            (2, 128, "registered-fit", "2:128", argv.as_str()),
            (2, 128, "local-smoke", "", argv.as_str()),
            (2, 129, "local-smoke", "2:128", argv.as_str()),
            (3, 128, "local-smoke", "2:128", argv.as_str()),
            (2, 128, "local-smoke", "2:128", "invalid"),
        ] {
            assert!(!fit_budget(
                epochs,
                pairs,
                Some(contract),
                Some(marker),
                Some(hash)
            ));
        }
    }
}
