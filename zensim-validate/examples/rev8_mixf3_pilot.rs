//! Matched Rev7 vs Rev8 Alg S4 MixF3 pilot oracle, extractor, and trainer.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use zenpredict::{MetadataType, Model, Predictor, WeightDtype, WeightStorage};
use zenpredict_bake::{BakeLayer, BakeMetadataEntry, BakeRequest, append_metadata_utf8, bake};
use zensim::rev8::{IntensityTarget, RgbRows, Variant, extract_fx2};
use zensim_validate::dial_spline::{fit_identity_pinned_knots, fit_spline_knots, spline_payload};
use zensim_validate::mlp_train::{
    FeatureRows, GroupLossMode, MlpHyperparams, PairSampling, TrainingGroup, ValidationPolicy,
    certify_monotone_bake, train_mlp, validate_training_capabilities,
};
use zensim_validate::output_calibration_spline::{apply as apply_spline, parse_payload};
use zensim_validate::panel::ValAggregate;

const SPLINE_KEY: &str = "zentrain.output_calibration_spline";

#[derive(Parser)]
#[command(name = "rev8_mixf3_pilot")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Emit Rev8 Alg S4 Nits255/Nits500 JSON for a pair of images or synthetic recipe.
    OracleJson {
        #[arg(long, num_args = 2, value_names = ["REF", "DIST"])]
        images: Option<Vec<PathBuf>>,
        #[arg(long, value_name = "WxH:KIND")]
        synth: Option<String>,
    },
    /// Train one MixF3 base or continuation checkpoint and evaluate all requested splits.
    Train { job_json: PathBuf },
    /// Score pre-extracted 1825-col Rev7 feature rows with a shipped BakeScorer binary.
    ScoreRaw1825 {
        bake_bin: PathBuf,
        raw1825_f32_bin: PathBuf,
        n_rows: usize,
        out_json: PathBuf,
    },
}

fn check_safe_path(path: &Path) -> Result<(), String> {
    let candidates = [
        path.to_path_buf(),
        path.canonicalize().unwrap_or_else(|_| path.to_path_buf()),
    ];
    for p in candidates {
        let lower_full = p.to_string_lossy().to_ascii_lowercase();
        for bad in ["cid22_validation_set.csv", "public_test", "pairs_test"] {
            if lower_full.contains(bad) {
                return Err(format!("refusing protected path {path:?} (matched {bad})"));
            }
        }
        for comp in p.components() {
            let s = comp.as_os_str().to_string_lossy().to_ascii_lowercase();
            if s.contains("_sealed")
                || s.contains("holdout")
                || s.contains("terminal")
                || s.starts_with("labels__")
                || s.contains("cid22val")
            {
                return Err(format!(
                    "refusing protected component {s:?} in path {path:?}"
                ));
            }
        }
    }
    Ok(())
}

fn decode_rgb8(path: &Path) -> Result<(Vec<[u8; 3]>, usize, usize), String> {
    check_safe_path(path)?;
    let img = image::ImageReader::open(path)
        .map_err(|e| format!("open {path:?}: {e}"))?
        .with_guessed_format()
        .map_err(|e| format!("guess format {path:?}: {e}"))?
        .decode()
        .map_err(|e| format!("decode {path:?}: {e}"))?
        .into_rgb8();
    let (w, h) = img.dimensions();
    let rgb: Vec<[u8; 3]> = img.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
    Ok((rgb, w as usize, h as usize))
}

type SynthPair = (Vec<[u8; 3]>, Vec<[u8; 3]>, usize, usize);

fn synth_pair(spec: &str) -> Result<SynthPair, String> {
    let (dims, kind) = spec
        .split_once(':')
        .ok_or_else(|| format!("expected WxH:KIND, got {spec:?}"))?;
    let (ws, hs) = dims
        .split_once('x')
        .ok_or_else(|| format!("expected WxH:KIND, got {spec:?}"))?;
    let w: usize = ws.parse().map_err(|_| format!("bad width in {spec:?}"))?;
    let h: usize = hs.parse().map_err(|_| format!("bad height in {spec:?}"))?;
    if w == 0 || h == 0 {
        return Err("dimensions must be positive".into());
    }
    let mut a = Vec::with_capacity(w * h);
    let mut b = Vec::with_capacity(w * h);
    let mut state =
        (w as u32).wrapping_mul(73856093) ^ (h as u32).wrapping_mul(19349663) ^ 0x9e37_79b9;
    let mut next_u8 = || -> u8 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state >> 24) as u8
    };
    for y in 0..h {
        for x in 0..w {
            let n0 = next_u8();
            let base = [
                (((x * 255) / w.max(1)) as u8).saturating_add(n0 & 15),
                (((y * 255) / h.max(1)) as u8).saturating_add((x * 7) as u8 & 15),
                (((x + y) * 19) as u8).wrapping_add(n0 >> 3),
            ];
            let (pa, pb) = match kind {
                "identity" => (base, base),
                "uniform_shift" => (
                    base,
                    [
                        base[0].saturating_add(9),
                        base[1].saturating_add(7),
                        base[2].saturating_add(11),
                    ],
                ),
                "checker_gradient" => {
                    let cb = if (x + y) % 2 == 0 { 220u8 } else { 36u8 };
                    let pa = [cb, base[1], ((x * 13 + y * 5) as u8)];
                    let pb = [
                        (pa[0] / 12) * 12,
                        pa[1].saturating_sub(if (x / 8 + y / 8) % 2 == 0 { 6 } else { 0 }),
                        pa[2].saturating_add(4),
                    ];
                    (pa, pb)
                }
                "asymmetric_blobs" => {
                    let pa = [
                        if (x / 4 + y / 4) % 2 == 0 { 210 } else { 40 },
                        ((x * 11 + y * 17) as u8),
                        ((y * 29) as u8),
                    ];
                    let in_blob = x >= w / 4 && x < (3 * w) / 4 && y >= h / 4 && y < (3 * h) / 4;
                    let pb = if in_blob {
                        [
                            pa[0].saturating_sub(35),
                            pa[1].saturating_add(24),
                            pa[2].saturating_sub(28),
                        ]
                    } else {
                        [pa[0] / 8 * 8, pa[1], pa[2]]
                    };
                    (pa, pb)
                }
                other => return Err(format!("unknown synth recipe {other:?}")),
            };
            a.push(pa);
            b.push(pb);
        }
    }
    Ok((a, b, w, h))
}

#[derive(Serialize)]
struct OracleTargetOut {
    features: Vec<Vec<Vec<f64>>>,
    scale_fragility: Vec<Vec<f64>>,
    fx2_480: Vec<f64>,
    swap_exact_bits: bool,
}

#[derive(Serialize)]
struct OracleOut {
    width: usize,
    height: usize,
    nits255: OracleTargetOut,
    nits500: OracleTargetOut,
}

fn eval_oracle_target(
    va: RgbRows<'_>,
    vb: RgbRows<'_>,
    target: IntensityTarget,
) -> OracleTargetOut {
    let ab = extract_fx2(va, vb, Variant::AlgS4, target, false).expect("matching dims");
    let ba = extract_fx2(vb, va, Variant::AlgS4, target, false).expect("matching dims");
    let fx_ab = ab.alg_s4_fx2_inputs();
    let fx_ba = ba.alg_s4_fx2_inputs();
    let swap_exact_bits = fx_ab
        .iter()
        .zip(&fx_ba)
        .all(|(x, y)| x.to_bits() == y.to_bits())
        && ab.scale_fragility[..4]
            .iter()
            .flatten()
            .zip(ba.scale_fragility[..4].iter().flatten())
            .all(|(x, y)| x.to_bits() == y.to_bits());
    let features = (0..4)
        .map(|s| (0..3).map(|c| ab.features[s][c][..20].to_vec()).collect())
        .collect();
    let scale_fragility = (0..4).map(|s| ab.scale_fragility[s].to_vec()).collect();
    OracleTargetOut {
        features,
        scale_fragility,
        fx2_480: fx_ab.to_vec(),
        swap_exact_bits,
    }
}

fn cmd_oracle_json(images: Option<Vec<PathBuf>>, synth: Option<String>) -> Result<(), String> {
    let (a, b, w, h) = match (images, synth) {
        (Some(paths), None) if paths.len() == 2 => {
            let (ra, wa, ha) = decode_rgb8(&paths[0])?;
            let (rb, wb, hb) = decode_rgb8(&paths[1])?;
            if (wa, ha) != (wb, hb) {
                return Err(format!("dimension mismatch: {wa}x{ha} vs {wb}x{hb}"));
            }
            (ra, rb, wa, ha)
        }
        (None, Some(spec)) => synth_pair(&spec)?,
        _ => return Err("specify exactly one of --images REF DIST or --synth WxH:KIND".into()),
    };
    let va = RgbRows::new(&a, w, h, w).ok_or("invalid RgbRows for ref")?;
    let vb = RgbRows::new(&b, w, h, w).ok_or("invalid RgbRows for dist")?;
    let out = OracleOut {
        width: w,
        height: h,
        nits255: eval_oracle_target(va, vb, IntensityTarget::Nits255),
        nits500: eval_oracle_target(va, vb, IntensityTarget::Nits500),
    };
    println!(
        "{}",
        serde_json::to_string(&out).map_err(|e| e.to_string())?
    );
    Ok(())
}

fn read_f32_bin(path: &Path, expected_len: usize) -> Result<Vec<f32>, String> {
    check_safe_path(path)?;
    let mut f = File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let mut raw = Vec::with_capacity(expected_len * 4);
    f.read_to_end(&mut raw)
        .map_err(|e| format!("read {path:?}: {e}"))?;
    if raw.len() != expected_len * 4 {
        return Err(format!(
            "{path:?}: expected {} bytes ({expected_len} f32s), got {}",
            expected_len * 4,
            raw.len()
        ));
    }
    let out = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect();
    Ok(out)
}

fn read_f64_bin(path: &Path, expected_len: usize) -> Result<Vec<f64>, String> {
    check_safe_path(path)?;
    let mut f = File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let mut raw = Vec::with_capacity(expected_len * 8);
    f.read_to_end(&mut raw)
        .map_err(|e| format!("read {path:?}: {e}"))?;
    if raw.len() != expected_len * 8 {
        return Err(format!(
            "{path:?}: expected {} bytes ({expected_len} f64s), got {}",
            expected_len * 8,
            raw.len()
        ));
    }
    let out = raw
        .as_chunks::<8>()
        .0
        .iter()
        .map(|&c| f64::from_le_bytes(c))
        .collect();
    Ok(out)
}

fn read_u32_bin(path: &Path, expected_len: usize) -> Result<Vec<u32>, String> {
    check_safe_path(path)?;
    let mut f = File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let mut raw = Vec::with_capacity(expected_len * 4);
    f.read_to_end(&mut raw)
        .map_err(|e| format!("read {path:?}: {e}"))?;
    if raw.len() != expected_len * 4 {
        return Err(format!(
            "{path:?}: expected {} bytes ({expected_len} u32s), got {}",
            expected_len * 4,
            raw.len()
        ));
    }
    let out = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| u32::from_le_bytes(c))
        .collect();
    Ok(out)
}

#[derive(Deserialize)]
struct JobGroup {
    name: String,
    features_f32_bin: PathBuf,
    n_rows: usize,
    scores_f64_bin: PathBuf,
    ref_ids_u32_bin: Option<PathBuf>,
    train_weight: f64,
    validation_weight: f64,
    loss_mode: String,
}

#[derive(Deserialize)]
struct AnchorSpec {
    features_f32_bin: PathBuf,
    n_rows: usize,
    scores_f64_bin: PathBuf,
}

#[derive(Deserialize)]
struct EvalSplitSpec {
    name: String,
    features_f32_bin: PathBuf,
    n_rows: usize,
}

#[derive(Deserialize)]
struct TrainJob {
    out_json: PathBuf,
    out_ckpt: PathBuf,
    out_cal_bake: Option<PathBuf>,
    ckpt_dir: PathBuf,
    n_features: usize,
    init_seed: u64,
    sample_seed: u64,
    n_epochs: usize,
    pairs_per_epoch: usize,
    initial_lr: f64,
    constant_lr: bool,
    init_from: Option<PathBuf>,
    groups: Vec<JobGroup>,
    spline_anchor: AnchorSpec,
    eval_splits: Vec<EvalSplitSpec>,
}

#[derive(Serialize)]
struct CertOut {
    ramps: usize,
    free_ramps: usize,
    free_amplitude: f64,
    max_pairs: usize,
}

#[derive(Serialize)]
struct EvalOut {
    name: String,
    n_rows: usize,
    preds_raw: Vec<f64>,
    preds_cal: Vec<f64>,
}

#[derive(Serialize)]
struct TrainOut {
    n_features: usize,
    selected_epoch: usize,
    spline_kind: String,
    spline_xs: Vec<f64>,
    spline_ys: Vec<f64>,
    raw_identity: f64,
    cal_identity: f64,
    cert: CertOut,
    active_hidden_w2_neg: usize,
    active_signed_units: usize,
    w1_col_l1: Vec<f64>,
    evals: Vec<EvalOut>,
}

struct OwnedGroupData {
    name: String,
    features: Vec<f64>,
    scores: Vec<f64>,
    ref_ids: Option<Vec<u32>>,
    train_weight: f64,
    validation_weight: f64,
    loss_mode: GroupLossMode,
    n_rows: usize,
}

fn ensure_fresh_ckpt_dir(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {dir:?}: {e}"))?;
    let entries = std::fs::read_dir(dir).map_err(|e| format!("readdir {dir:?}: {e}"))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("readdir {dir:?}: {e}"))?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("ckpt_epoch") && name.ends_with(".bin") {
            return Err(format!(
                "stale checkpoint {name} already present in {:?}; refusing to reuse dirty dir",
                dir
            ));
        }
    }
    Ok(())
}

fn cleanup_intermediate_ckpts(dir: &Path) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("ckpt_epoch") && name.ends_with(".bin") {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
}

fn bake_calibrated_model(
    model: &Model,
    repro_json: &str,
    spline_bytes: &[u8],
) -> Result<Vec<u8>, String> {
    let l0 = model.layer(0);
    let l1 = model.layer(1);
    let (w0, w1) = match (&l0.weights, &l1.weights) {
        (WeightStorage::F32(w0), WeightStorage::F32(w1)) => (w0, w1),
        _ => return Err("expected F32 layers in selected checkpoint".into()),
    };
    let layers = [
        BakeLayer {
            in_dim: l0.in_dim,
            out_dim: l0.out_dim,
            activation: l0.activation,
            dtype: WeightDtype::F32,
            weights: w0,
            biases: l0.biases,
        },
        BakeLayer {
            in_dim: l1.in_dim,
            out_dim: l1.out_dim,
            activation: l1.activation,
            dtype: WeightDtype::F32,
            weights: w1,
            biases: l1.biases,
        },
    ];
    let md = [
        BakeMetadataEntry {
            key: "zentrain.repro",
            kind: MetadataType::Utf8,
            value: repro_json.as_bytes(),
        },
        BakeMetadataEntry {
            key: SPLINE_KEY,
            kind: MetadataType::Bytes,
            value: spline_bytes,
        },
    ];
    bake(&BakeRequest {
        schema_hash: model.schema_hash(),
        flags: 0,
        scaler_mean: model.scaler_mean(),
        scaler_scale: model.scaler_scale(),
        layers: &layers,
        feature_bounds: &[],
        metadata: &md,
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
        feature_order: None,
        output_order: None,
        compressed: true,
        hu_permutations: None,
    })
    .map_err(|e| format!("bake calibrated model: {e:?}"))
}

fn predict_raw_rows(
    model: &Model,
    path: &Path,
    n_rows: usize,
    n_features: usize,
) -> Result<Vec<f64>, String> {
    let flat = read_f32_bin(path, n_rows * n_features)?;
    let mut pred = Predictor::new(model);
    let mut out = Vec::with_capacity(n_rows);
    for row in flat.chunks_exact(n_features) {
        let y = pred
            .predict(row)
            .map_err(|e| format!("Predictor::predict: {e}"))?;
        out.push(f64::from(y[0]));
    }
    Ok(out)
}

fn cmd_train(job_json: &Path) -> Result<(), String> {
    check_safe_path(job_json)?;
    let raw = std::fs::read(job_json).map_err(|e| format!("read {job_json:?}: {e}"))?;
    let job: TrainJob =
        serde_json::from_slice(&raw).map_err(|e| format!("parse {job_json:?}: {e}"))?;
    if job.n_epochs == 0 {
        return Err("n_epochs must be >= 1".into());
    }
    if let Some(p) = &job.init_from {
        check_safe_path(p)?;
    }
    check_safe_path(&job.out_json)?;
    check_safe_path(&job.out_ckpt)?;
    check_safe_path(&job.ckpt_dir)?;
    ensure_fresh_ckpt_dir(&job.ckpt_dir)?;

    let mut owned_groups = Vec::with_capacity(job.groups.len());
    for g in &job.groups {
        let feat_f32 = read_f32_bin(&g.features_f32_bin, g.n_rows * job.n_features)?;
        let features: Vec<f64> = feat_f32.into_iter().map(f64::from).collect();
        let scores = read_f64_bin(&g.scores_f64_bin, g.n_rows)?;
        let ref_ids = match &g.ref_ids_u32_bin {
            Some(p) => Some(read_u32_bin(p, g.n_rows)?),
            None => None,
        };
        let loss_mode = match g.loss_mode.as_str() {
            "rank" => GroupLossMode::Rank,
            "mse" => GroupLossMode::Mse,
            "both" => GroupLossMode::Both,
            other => return Err(format!("unknown loss_mode {other:?}")),
        };
        owned_groups.push(OwnedGroupData {
            name: g.name.clone(),
            features,
            scores,
            ref_ids,
            train_weight: g.train_weight,
            validation_weight: g.validation_weight,
            loss_mode,
            n_rows: g.n_rows,
        });
    }

    let mut hp = MlpHyperparams {
        init_from: job.init_from.clone(),
        constant_lr: job.constant_lr,
        nonneg_distance: true,
        nonneg_pin: 100.0,
        nonneg_thresholds: true,
        monotone_cbc: true,
        max_pairs: 32,
        monotone_ramps: 0,
        free_ramps: 0,
        monotone_product_pairs: Vec::new(),
        monotone_product_scales: Vec::new(),
        n_hidden: 128,
        n_hidden_layers: 1,
        n_epochs: job.n_epochs,
        pairs_per_epoch: job.pairs_per_epoch,
        initial_lr: job.initial_lr,
        leaky_alpha: 0.0,
        seed: job.init_seed,
        init_seed: Some(job.init_seed),
        sample_seed: Some(job.sample_seed),
        log_every: 1,
        dump_checkpoints_every: 1,
        dump_checkpoints_dir: Some(job.ckpt_dir.clone()),
        l2_lambda: 1e-5,
        early_stop_patience: 0,
        validation_policy: ValidationPolicy::Mean,
        val_aggregate: ValAggregate::GeomeanSPP,
        pair_sampling: PairSampling::Uniform,
        out_dtype: WeightDtype::F32,
        minibatch_size: 1,
        parallel_batch: false,
        norm_in_norm_weight: 0.0,
        mse_weight: 1.0,
        ..MlpHyperparams::default()
    };

    let has_abs = owned_groups
        .iter()
        .any(|g| matches!(g.loss_mode, GroupLossMode::Mse | GroupLossMode::Both));
    hp = validate_training_capabilities(&hp, has_abs, 0);

    let mut tg: Vec<TrainingGroup<'_>> = owned_groups
        .iter_mut()
        .map(|g| TrainingGroup {
            name: g.name.clone(),
            human_scores: &g.scores,
            features: FeatureRows::Releasable {
                n_rows: g.n_rows,
                n_features: job.n_features,
                data: &mut g.features,
            },
            metric_sigmas: None,
            train_weight: g.train_weight,
            validation_weight: g.validation_weight,
            ref_ids: g.ref_ids.as_deref(),
            loss_mode: g.loss_mode,
        })
        .collect();

    let mut log = Vec::new();
    let _best_bytes = train_mlp(&mut tg, job.n_features, &hp, &mut log);

    let last_epoch = job.n_epochs - 1;
    let last_ckpt_path = job.ckpt_dir.join(format!("ckpt_epoch{last_epoch:03}.bin"));
    let ckpt_bytes = std::fs::read(&last_ckpt_path)
        .map_err(|e| format!("read final checkpoint {last_ckpt_path:?}: {e}"))?;
    cleanup_intermediate_ckpts(&job.ckpt_dir);

    let repro_json = serde_json::json!({
        "argv": [
            "--nonneg-distance",
            "--nonneg-thresholds",
            "--monotone-cbc",
            "--hidden", "128",
            "--max-pairs", "32",
            "--nonneg-pin", "100.0"
        ],
        "selected_epoch": last_epoch,
        "init_seed": job.init_seed,
        "sample_seed": job.sample_seed,
        "n_features": job.n_features,
    })
    .to_string();

    let stamped_bytes = append_metadata_utf8(&ckpt_bytes, "zentrain.repro", &repro_json)
        .map_err(|e| format!("append zentrain.repro: {e}"))?;
    if let Some(parent) = job.out_ckpt.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir {parent:?}: {e}"))?;
    }
    std::fs::write(&job.out_ckpt, &stamped_bytes)
        .map_err(|e| format!("write {:?}: {e}", job.out_ckpt))?;

    let model =
        Model::from_bytes(&stamped_bytes).map_err(|e| format!("parse stamped checkpoint: {e}"))?;
    let cert = certify_monotone_bake(&model, 100.0f32)
        .map_err(|e| format!("certify_monotone_bake failed: {e}"))?;

    let anchor_preds = predict_raw_rows(
        &model,
        &job.spline_anchor.features_f32_bin,
        job.spline_anchor.n_rows,
        job.n_features,
    )?;
    let anchor_tgt = read_f64_bin(&job.spline_anchor.scores_f64_bin, job.spline_anchor.n_rows)?;

    let (spline_kind, spline_xs, spline_ys) =
        match fit_identity_pinned_knots(&anchor_preds, &anchor_tgt, 18, 100.0, 2.0) {
            Ok((xs, ys)) => ("identity_pinned".to_string(), xs, ys),
            Err(reason) => {
                eprintln!(
                    "[train] fit_identity_pinned_knots fallback ({reason}); using fit_spline_knots"
                );
                let (xs, ys) = fit_spline_knots(&anchor_preds, &anchor_tgt, 18, true);
                ("fit_spline_knots_neg_tail".to_string(), xs, ys)
            }
        };
    let sp_bytes = spline_payload(&spline_xs, &spline_ys);
    let sp = parse_payload(&sp_bytes).ok_or("failed to parse fitted spline payload")?;

    if let Some(cal_path) = &job.out_cal_bake {
        if let Some(parent) = cal_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("mkdir {parent:?}: {e}"))?;
        }
        let cal_bytes = bake_calibrated_model(&model, &repro_json, &sp_bytes)?;
        std::fs::write(cal_path, &cal_bytes).map_err(|e| format!("write {cal_path:?}: {e}"))?;
    }

    let mut pred = Predictor::new(&model);
    let zeros = vec![0.0f32; job.n_features];
    let raw_identity = f64::from(
        pred.predict(&zeros)
            .map_err(|e| format!("identity predict: {e}"))?[0],
    );
    let cal_identity = apply_spline(raw_identity, &sp);

    let l0 = model.layer(0);
    let l1 = model.layer(1);
    let (w1, w2) = match (&l0.weights, &l1.weights) {
        (WeightStorage::F32(w0), WeightStorage::F32(w1)) => (w0, w1),
        _ => return Err("expected F32 weights for diagnostics".into()),
    };
    let n_hidden = l0.out_dim;
    let active_hidden_w2_neg = w2.iter().filter(|&&w| w < 0.0).count();
    let active_signed_units = (0..n_hidden)
        .filter(|&j| (0..job.n_features).any(|k| w1[k * n_hidden + j] < 0.0))
        .count();
    let w1_col_l1: Vec<f64> = (0..job.n_features)
        .map(|k| {
            (0..n_hidden)
                .map(|j| f64::from(w1[k * n_hidden + j].abs()))
                .sum()
        })
        .collect();

    let mut evals = Vec::with_capacity(job.eval_splits.len());
    for sp_spec in &job.eval_splits {
        let preds_raw = predict_raw_rows(
            &model,
            &sp_spec.features_f32_bin,
            sp_spec.n_rows,
            job.n_features,
        )?;
        let preds_cal: Vec<f64> = preds_raw.iter().map(|&r| apply_spline(r, &sp)).collect();
        evals.push(EvalOut {
            name: sp_spec.name.clone(),
            n_rows: sp_spec.n_rows,
            preds_raw,
            preds_cal,
        });
    }

    let out = TrainOut {
        n_features: job.n_features,
        selected_epoch: last_epoch,
        spline_kind,
        spline_xs,
        spline_ys,
        raw_identity,
        cal_identity,
        cert: CertOut {
            ramps: cert.ramps,
            free_ramps: cert.free_ramps,
            free_amplitude: cert.free_amplitude,
            max_pairs: cert.max_pairs,
        },
        active_hidden_w2_neg,
        active_signed_units,
        w1_col_l1,
        evals,
    };
    if let Some(parent) = job.out_json.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir {parent:?}: {e}"))?;
    }
    std::fs::write(
        &job.out_json,
        serde_json::to_string(&out).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("write {:?}: {e}", job.out_json))?;
    Ok(())
}

fn cmd_score_raw1825(
    bake_bin: &Path,
    raw1825_f32_bin: &Path,
    n_rows: usize,
    out_json: &Path,
) -> Result<(), String> {
    if std::env::var("ZENSIM_FORMULA_REV").as_deref() != Ok("7") {
        return Err("score-raw1825 requires ZENSIM_FORMULA_REV=7".into());
    }
    check_safe_path(bake_bin)?;
    check_safe_path(raw1825_f32_bin)?;
    check_safe_path(out_json)?;
    let bake_bytes = std::fs::read(bake_bin).map_err(|e| format!("read {bake_bin:?}: {e}"))?;
    let model = Model::from_bytes(&bake_bytes).map_err(|e| format!("parse bake: {e:?}"))?;
    let params = zensim_validate::bake_runtime::post_mode_params("extrapolate")?;
    let mut scorer = zensim::BakeScorer::new(&model)
        .map_err(|e| format!("BakeScorer::new: {e}"))?
        .with_parallel(false)
        .with_score_disposition(&params)
        .map_err(|e| format!("with_score_disposition: {e}"))?;
    let flat = read_f32_bin(raw1825_f32_bin, n_rows * 1825)?;
    let mut scores = Vec::with_capacity(n_rows);
    let mut row_f64 = vec![0.0_f64; 1825];
    for chunk in flat.as_chunks::<1825>().0 {
        for (dst, &src) in row_f64.iter_mut().zip(chunk) {
            *dst = f64::from(src);
        }
        let s = scorer
            .score_features(&row_f64, 256, 256, None)
            .map_err(|e| format!("score_features: {e}"))?;
        scores.push(s);
    }
    if let Some(parent) = out_json.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir {parent:?}: {e}"))?;
    }
    let payload = serde_json::json!({ "scores": scores });
    std::fs::write(
        out_json,
        serde_json::to_string(&payload).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("write {out_json:?}: {e}"))?;
    Ok(())
}

fn main() {
    zensim_validate::tier_cap::apply_from_env();
    let cli = Cli::parse();
    let res = match cli.cmd {
        Cmd::OracleJson { images, synth } => cmd_oracle_json(images, synth),
        Cmd::Train { job_json } => cmd_train(&job_json),
        Cmd::ScoreRaw1825 {
            bake_bin,
            raw1825_f32_bin,
            n_rows,
            out_json,
        } => cmd_score_raw1825(&bake_bin, &raw1825_f32_bin, n_rows, &out_json),
    };
    if let Err(e) = res {
        eprintln!("FATAL: {e}");
        std::process::exit(1);
    }
}
