// Caller-selected engineering packet. Only the explicit instrument cfg compiles
// this module; ordinary all-feature tests never open these external pixels.
use super::*;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};

#[path = "steerfix_precision.rs"]
mod precision;

mod stats {
    include!("../../examples/support/coherence_stats.rs");
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Objective {
    Served,
    PreFloor,
    SmoothFloor,
}

// The head/pin/winsor and feature transforms still execute in the native owner.
// Diagnostic calibration is detached only on this test-owned scorer, restored
// even on an error; every shipping scorer has Objective::Served.
pub(super) fn diagnostic_forward(
    scorer: &mut BakeScorer<'_>,
    features: &[f64],
    w: u32,
    h: u32,
    codec: Option<&str>,
) -> Result<f64, ZensimError> {
    assert!(scorer.members.is_empty() && scorer.disposition.is_none() && codec.is_none());
    let objective = scorer.diagnostic_objective;
    let metadata = Arc::clone(&scorer.metadata);
    scorer.diagnostic_objective = Objective::Served;
    let md = Arc::make_mut(&mut scorer.metadata);
    md.output_spline = None;
    md.per_codec_calibration = None;
    let raw = scorer.score_features(features, w, h, codec);
    scorer.metadata = metadata;
    scorer.diagnostic_objective = objective;
    let raw = raw?;
    Ok(match objective {
        Objective::Served => unreachable!(),
        Objective::PreFloor => raw,
        Objective::SmoothFloor => {
            let spline = scorer
                .metadata
                .output_spline
                .as_ref()
                .expect("smooth-floor needs spline");
            let span = spline.ys[spline.ys.len() - 1] - spline.ys[0];
            assert!(span > 0.0 && spline.derivs[0] > 0.0);
            let floor = spline.ys[0] - span;
            let linear = spline.ys[0] + spline.derivs[0] * (raw - spline.xs[0]);
            if raw <= spline.xs[0] && linear < floor {
                // C1 at the old floor crossing, strictly increasing beneath it;
                // span is the declared dial range, never fit to these cases.
                floor + span * ((linear - floor) / span).asinh()
            } else {
                crate::score_math::pchip_eval_capped(raw, &spline.xs, &spline.ys, &spline.derivs)
            }
        }
    })
}

#[derive(Deserialize)]
struct Job {
    cases: Vec<Case>,
    output: String,
    engine: bool,
    #[serde(default)]
    floor_recovery: bool,
    #[serde(default)]
    linear_probe: bool,
    // Explicit caller opt-in to the hash-bound native repair rows captured in
    // part 4. The current score owner and map still execute for every case.
    #[serde(default)]
    repair_feature_cache: Option<String>,
    // The process formula revision the packet runs at: 5 for every registered packet. The by-construction
    // revision-7 models (benchmarks/byconstruction_2026-10-22.md) declare 7; a repair cache must carry the same
    // revision, so they run without one.
    #[serde(default = "revision_five")]
    formula_revision: u64,
}
fn revision_five() -> u64 {
    5
}
#[derive(Deserialize)]
struct Case {
    key: String,
    reference: String,
    distorted: String,
    reference_sha256: String,
    distorted_sha256: String,
    reference_pixels_sha256: String,
    distorted_pixels_sha256: String,
    model: String,
    model_sha256: String,
    objective: Objective,
    block: usize,
}
fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn image(path: &str, expected: &str) -> (Vec<[u8; 3]>, usize, usize) {
    let bytes = std::fs::read(path).expect("admitted engineering pixel file");
    assert_eq!(sha(&bytes), expected);
    let img = image::load_from_memory(&bytes).unwrap().to_rgb8();
    (
        img.pixels().map(|p| p.0).collect(),
        img.width() as usize,
        img.height() as usize,
    )
}
fn owner(model: &crate::mlp::Model, objective: Objective) -> BakeScorer<'_> {
    let mut s = BakeScorer::new(model).unwrap().with_parallel(false);
    s.diagnostic_objective = objective;
    s
}

#[test]
fn engineering_packet() {
    let _token = archmage::testing::lock_token_testing();
    let path =
        std::env::var("STEERFIX_PACKET").expect("recipe selects an explicit engineering packet");
    let bytes = std::fs::read(&path).unwrap();
    let job: Job = serde_json::from_slice(&bytes).unwrap();
    assert!(matches!(job.formula_revision, 5 | 7));
    assert_eq!(
        std::env::var("ZENSIM_FORMULA_REV").as_deref(),
        Ok(job.formula_revision.to_string().as_str())
    );
    assert!(
        !std::path::Path::new(&job.output).exists(),
        "fresh output required"
    );
    assert_eq!(
        job.engine,
        std::env::var("ZENSIM_NEIGHBOUR_EXACT").as_deref() == Ok("1")
    );
    let cache = job.repair_feature_cache.as_ref().map(|path| {
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(value["schema"], "e34-native-repair-features-v1");
        assert_eq!(value["formula_revision"], job.formula_revision);
        let producer = std::fs::read(value["producer_model"].as_str().unwrap()).unwrap();
        assert_eq!(sha(&producer), value["producer_sha256"].as_str().unwrap());
        let model = crate::mlp::Model::from_bytes(&producer).unwrap();
        let reads: std::collections::BTreeSet<_> = owner(&model, Objective::Served)
            .consumed_feature_ids()
            .unwrap()
            .into_iter()
            .collect();
        (value, reads)
    });
    let mut summary = Vec::new();
    for (case_index, case) in job.cases.into_iter().enumerate() {
        let blob = std::fs::read(&case.model).unwrap();
        assert_eq!(sha(&blob), case.model_sha256);
        let model = crate::mlp::Model::from_bytes(&blob).unwrap();
        let (r, w, h) = image(&case.reference, &case.reference_sha256);
        let (d, dw, dh) = image(&case.distorted, &case.distorted_sha256);
        assert_eq!((w, h), (dw, dh));
        assert_eq!(sha(bytemuck::cast_slice(&r)), case.reference_pixels_sha256);
        assert_eq!(sha(bytemuck::cast_slice(&d)), case.distorted_pixels_sha256);
        let rs = crate::RgbSlice::new(&r, w, h);
        let ds = crate::RgbSlice::new(&d, w, h);
        let mut steering = owner(&model, case.objective);
        let mut worker = steering.prepare_steering(&rs, 1).unwrap();
        let mapped = worker.compute(&ds, None).unwrap();
        let mut pixel = owner(&model, case.objective);
        let base = pixel.compute(&rs, &ds, None).unwrap();
        assert_eq!(mapped.result().score().to_bits(), base.score().to_bits());
        assert_eq!(mapped.result().features(), base.features());
        let cached_rows = cache.as_ref().map(|(cache, available)| {
            let row = &cache["rows"][case_index];
            assert_eq!(row["key"], case.key);
            assert_eq!(row["block"], case.block);
            assert_eq!(row["reference_sha256"], case.reference_sha256);
            assert_eq!(row["distorted_sha256"], case.distorted_sha256);
            assert_eq!(row["width"], w);
            assert_eq!(row["height"], h);
            let bytes = std::fs::read(row["feature_file"].as_str().unwrap()).unwrap();
            assert_eq!(sha(&bytes), row["feature_sha256"].as_str().unwrap());
            let width = row["feature_width"].as_u64().unwrap() as usize;
            let n = 1 + w.div_ceil(case.block) * h.div_ceil(case.block);
            assert_eq!(bytes.len(), n * width * 8);
            let rows: Vec<f64> = bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| f64::from_le_bytes(*b))
                .collect();
            assert!(rows.iter().all(|x| x.is_finite()));
            for id in pixel.consumed_feature_ids().unwrap() {
                assert!(
                    available.contains(&id),
                    "cache does not cover consumed f{id}"
                );
                assert_eq!(
                    rows[usize::from(id)].to_bits(),
                    base.features()[usize::from(id)].to_bits()
                );
            }
            assert_eq!(
                pixel
                    .score_features(&rows[..width], w as u32, h as u32, None)
                    .unwrap()
                    .to_bits(),
                base.score().to_bits()
            );
            (rows, width)
        });
        let f = cached_rows
            .as_ref()
            .map_or(base.features(), |(rows, width)| &rows[..*width]);
        let s = mapped.sensitivities();
        let coefficients = job
            .linear_probe
            .then(|| linear_coefficients(&model, f.len()));
        let mut negative_rows = Vec::new();
        if let Some(c) = &coefficients {
            let ids = crate::feature_layout::declared_ids(&model).unwrap();
            if ids.iter_slots().any(|id| f[id] < 0.0) {
                negative_rows.push(json!({"row":"base"}));
            }
            assert_eq!(c.len(), s.len());
        }
        // Independent feature probes on the same served/diagnostic forward.
        let mut feature = owner(&model, case.objective);
        let mut sensitivity_objective = case.objective;
        if job.floor_recovery && case.objective == Objective::Served {
            // Independent expectation: probe the raw head, test the declared
            // spline's lower-tail crossing and check the served probes are
            // all zero. Do not call the production recovery helper here.
            let served_gradient = feature
                .score_features_fd_gradient(f, w as u32, h as u32, None)
                .unwrap();
            let mut raw = owner(&model, Objective::PreFloor);
            let output = raw.score_features(f, w as u32, h as u32, None).unwrap();
            if let Some(spline) = feature.metadata.output_spline.as_ref() {
                let span = spline.ys[spline.ys.len() - 1] - spline.ys[0];
                let linear = spline.ys[0] + spline.derivs[0] * (output - spline.xs[0]);
                if served_gradient.iter().all(|&s| s == 0.0)
                    && span > 0.0
                    && spline.derivs[0] > 0.0
                    && output < spline.xs[0]
                    && linear < spline.ys[0] - span
                {
                    feature = raw;
                    sensitivity_objective = Objective::PreFloor;
                }
            }
        }
        // The instrument's linear gain is the exact derivative used by the
        // encoder-facing map, including the registered lower-floor recovery.
        // A separate owner/buffer checks the complete sensitivity wiring.
        assert_eq!(
            s,
            feature
                .score_features_fd_gradient(f, w as u32, h as u32, None)
                .unwrap(),
            "instrument and served map must use the same exact sensitivity"
        );
        let mut served = owner(&model, Objective::Served);
        let served_base = served.compute(&rs, &ds, None).unwrap();
        let mut altered = d.clone();
        let mut blocks = Vec::new();
        let mut true_gain = Vec::new();
        let mut true_linear = Vec::new();
        let mut predicted = Vec::new();
        let mut served_bits = Vec::new();
        let mut bad = 0usize;
        let mut compared = 0usize;
        let mut max_abs = 0f64;
        let mut max_tol_ratio = 0f64;
        let mut coarse_actual = Vec::new();
        let mut coarse_local = Vec::new();
        let mut repair_index = 0;
        for y0 in (0..h).step_by(case.block) {
            for x0 in (0..w).step_by(case.block) {
                let (x1, y1) = ((x0 + case.block).min(w), (y0 + case.block).min(h));
                for y in y0..y1 {
                    let range = y * w + x0..y * w + x1;
                    altered[range.clone()].copy_from_slice(&r[range]);
                }
                let input = crate::RgbSlice::new(&altered, w, h);
                repair_index += 1;
                let (repair_features, repair_score, complete_score) =
                    if let Some((rows, width)) = &cached_rows {
                        let f = &rows[repair_index * width..(repair_index + 1) * width];
                        (
                            f.to_vec(),
                            pixel.score_features(f, w as u32, h as u32, None).unwrap(),
                            served.score_features(f, w as u32, h as u32, None).unwrap(),
                        )
                    } else {
                        let repaired = pixel.compute(&rs, &input, None).unwrap();
                        let complete = served.compute(&rs, &input, None).unwrap();
                        assert_eq!(repaired.features(), complete.features());
                        (
                            repaired.features().to_vec(),
                            repaired.score(),
                            complete.score(),
                        )
                    };
                if coefficients.is_some() {
                    let ids = crate::feature_layout::declared_ids(&model).unwrap();
                    if ids
                        .iter_slots()
                        .any(|id| repair_features.as_slice()[id] < 0.0)
                    {
                        negative_rows.push(json!({"row":"repair", "bounds":[x0,y0,x1,y1]}));
                    }
                }
                let df: Vec<_> = repair_features
                    .as_slice()
                    .iter()
                    .zip(f)
                    .map(|(a, b)| a - b)
                    .collect();
                let linear: f64 = df.iter().zip(s).map(|(d, s)| d * s).sum();
                let gain = mapped.refinement_gain(x0, y0, x1, y1);
                let delta = repair_score - base.score();
                true_gain.push(delta);
                true_linear.push(linear);
                predicted.push(gain);
                served_bits.push(complete_score.to_bits());
                let mut ca = 0.0;
                let mut cl = 0.0;
                if job.engine {
                    let snap = mapped
                        .neighbour_exact
                        .as_ref()
                        .expect("snapshot must capture admitted case");
                    let deltas = snap
                        .deltas((x0, y0, x1, y1), &crate::local_refine::Candidate::Reference)
                        .unwrap();
                    for (id, local) in deltas {
                        let actual = df[id];
                        let error = (local - actual).abs();
                        // Existing local_refine golden contract, unmodified.
                        let tol = 4.0
                            * f64::from(f32::EPSILON)
                            * f[id]
                                .abs()
                                .max(repair_features.as_slice()[id].abs())
                                .max(1e-20)
                            + 1e-6 * actual.abs();
                        compared += 1;
                        bad += usize::from(error > tol);
                        max_abs = max_abs.max(error);
                        max_tol_ratio = max_tol_ratio.max(error / tol.max(1e-30));
                        ca += s[id] * actual;
                        cl += s[id] * local;
                    }
                    coarse_actual.push(ca);
                    coarse_local.push(cl);
                }
                blocks.push(json!({"bounds":[x0,y0,x1,y1],"score_delta":delta,"linearized_gain":linear,"refinement_gain":gain,
                    "served_score_delta":complete_score-served_base.score(),"coarse_full_linear":ca,"coarse_local_linear":cl}));
                for y in y0..y1 {
                    let range = y * w + x0..y * w + x1;
                    altered[range.clone()].copy_from_slice(&d[range]);
                }
            }
        }
        let m2 = stats::spearman(&true_linear, &true_gain);
        let m3f = stats::spearman(&predicted, &true_gain);
        let report = json!({"key":case.key,"objective":format!("{:?}",case.objective),"model":case.model_sha256,
            "linear_probe":coefficients.as_ref().map(|c| json!({
                "coefficient_convention":"raw = pin - dot(c,x); compare s to -c",
                "max_abs_s_plus_c":s.iter().zip(c).map(|(s,c)|(s+c).abs()).fold(0.0_f64,f64::max),
                "s":s,"c":c,"negative_input_rows":negative_rows})),
            "sensitivity_objective":format!("{:?}",sensitivity_objective),
            "base_score":base.score(),"served_base_bits":served_base.score().to_bits(),"served_repair_bits":served_bits,
            "m2":m2,"m3f":m3f,"pass":m2>=0.99 && m3f>=0.70,"engine":job.engine,"blocks":blocks,
            "engine_feature_comparisons":compared,"engine_feature_disagreements":bad,"engine_max_abs_error":max_abs,
            "engine_max_tolerance_ratio":max_tol_ratio,"coarse_gain_rank":if job.engine {Some(stats::spearman(&coarse_local,&coarse_actual))} else {None},
            "map_vs_true_linear_rank":stats::spearman(&predicted,&true_linear),"sensitivity_shared_owner_exact":true,"sensitivity_method":"exact-gradient-d12","repair_feature_cache":job.repair_feature_cache});
        eprintln!(
            "{} {:?} M2={m2:.12} M3f={m3f:.12} engine_bad={bad}/{compared}",
            case.key, case.objective
        );
        summary.push(report);
        // Progress is persisted after each completed case; failures cannot hide it.
        std::fs::write(
            format!("{}.partial", job.output),
            serde_json::to_vec(&summary).unwrap(),
        )
        .unwrap();
    }
    std::fs::write(
        &job.output,
        serde_json::to_vec(&json!({"packet_sha256":sha(&bytes),"formula_revision":job.formula_revision,"rows":summary}))
            .unwrap(),
    )
    .unwrap();
}

// Test-owned inspection of the loaded, quantized L bake. No inferred weights,
// finite-difference coefficients, or changes to the serving implementation.
fn linear_coefficients(model: &crate::mlp::Model, width: usize) -> Vec<f64> {
    use zenpredict::{Activation, WeightStorage};
    fn weights(storage: WeightStorage<'_>) -> Vec<f64> {
        match storage {
            WeightStorage::F32(w) => w.iter().map(|&w| f64::from(w)).collect(),
            WeightStorage::F16(w) => w
                .iter()
                .map(|&w| f64::from(zenpredict::f16_bits_to_f32(w)))
                .collect(),
            WeightStorage::I8 { .. } => panic!("L probe supports f32/f16 only"),
        }
    }
    assert_eq!(model.n_layers(), 2);
    let first = model.layer(0);
    let last = model.layer(1);
    assert_eq!((first.out_dim, last.in_dim, last.out_dim), (1, 1, 1));
    assert_eq!(first.activation, Activation::Relu);
    assert_eq!(last.activation, Activation::Identity);
    assert_eq!(first.biases, &[0.0]);
    assert!(model.scaler_mean().iter().all(|&x| x == 0.0));
    let w1 = weights(first.weights);
    let w2 = weights(last.weights)[0];
    assert!(w1.iter().all(|&w| w >= 0.0) && w2 <= 0.0);
    let ids = crate::feature_layout::declared_ids(model).unwrap();
    assert_eq!(ids.len(), w1.len());
    let mut c = vec![0.0; width];
    for ((id, w), &scale) in ids.iter_slots().zip(w1).zip(model.scaler_scale()) {
        assert!(scale > 0.0);
        c[id] = -w2 * w / f64::from(scale);
    }
    c
}

#[test]
fn e34_linear_fixture() {
    let path = std::env::var("E34_LINEAR_FIXTURE").expect("explicit fixture recipe");
    let recipe: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let output = recipe["output"].as_str().unwrap();
    assert!(!std::path::Path::new(output).exists());
    let blob = zenpredict_bake::bake_from_json_str(&recipe["model"].to_string()).unwrap();
    let model = crate::mlp::Model::from_bytes(&blob).unwrap();
    let coefficients = linear_coefficients(
        &model,
        crate::feature_layout::declared_ids(&model)
            .unwrap()
            .iter_slots()
            .max()
            .unwrap()
            + 1,
    );
    assert!(coefficients.iter().any(|&c| c > 0.0));
    std::fs::write(output, blob).unwrap();
}

#[test]
fn e34_linear_model_contract() {
    let path = std::env::var("E34_LINEAR_MODEL_CONTRACT").expect("explicit model-contract job");
    let job: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let blob = std::fs::read(job["model"].as_str().unwrap()).unwrap();
    assert_eq!(sha(&blob), job["model_sha256"].as_str().unwrap());
    let model = crate::mlp::Model::from_bytes(&blob).unwrap();
    let width = crate::feature_layout::declared_ids(&model)
        .unwrap()
        .iter_slots()
        .max()
        .unwrap()
        + 1;
    let c = linear_coefficients(&model, width);
    assert!(matches!(
        model.layer(0).weights,
        zenpredict::WeightStorage::F16(_)
    ));
    assert!(matches!(
        model.layer(1).weights,
        zenpredict::WeightStorage::F32(_)
    ));
    assert!(
        model
            .scaler_scale()
            .iter()
            .all(|&s| s >= 1e-8 && s.is_finite())
    );
    assert!(
        model
            .metadata()
            .get("zentrain.feature_transforms")
            .is_none()
    );
    assert!(model.metadata().get("zensim.derived_inputs").is_none());
    let zeros = vec![0.0; width];
    let raw = owner(&model, Objective::PreFloor)
        .score_features(&zeros, 256, 256, None)
        .unwrap();
    let served = owner(&model, Objective::Served)
        .score_features(&zeros, 256, 256, None)
        .unwrap();
    assert_eq!(raw.to_bits(), 100.0_f64.to_bits());
    assert_eq!(served.to_bits(), 100.0_f64.to_bits());
    let report = json!({"model_sha256":sha(&blob),"input_width":model.layer(0).in_dim,
        "hidden_width":1,"output_width":1,"first_dtype":"f16","last_dtype":"f32",
        "nonzero_coefficients":c.iter().filter(|&&x| x>0.0).count(),
        "coefficient_min":c.iter().copied().fold(f64::INFINITY,f64::min),
        "coefficient_max":c.iter().copied().fold(f64::NEG_INFINITY,f64::max),
        "scaler_mean_zero":true,"scaler_scale_min":model.scaler_scale().iter().copied().fold(f32::INFINITY,f32::min),
        "hidden_bias_zero":true,"raw_zero_bits":raw.to_bits(),"served_zero_bits":served.to_bits(),
        "feature_transforms":false,"derived_inputs":false});
    use std::io::Write;
    std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(job["output"].as_str().unwrap())
        .unwrap()
        .write_all(&serde_json::to_vec_pretty(&report).unwrap())
        .unwrap();
}
