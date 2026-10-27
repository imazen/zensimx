//! Label-free probe of explicitly named packed models. No corpus or verdict reader.
//! Run with `ZENSIM_FORMULA_REV=5 serve_custom_bake --prodqual MODEL...`.
//! JSON records every dispatch permutation and every seed, including failures;
//! the caller compares the recorded bits and evaluates identity bands. Synthetic
//! ladder monotonicity is an observation, never an admission threshold here.

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use serde_json::json;
use zensim::research::{self, Request};
use zensim::{BakeScorer, RgbSlice};

const GEOMETRIES: [(usize, usize); 4] = [(64, 64), (97, 63), (131, 65), (255, 129)];
const DROPPED_BITS: [u32; 7] = [0, 1, 2, 3, 4, 5, 6];

// Same deterministic texture used by featcanon_rev5_parity, without a corpus.
fn reference(w: usize, h: usize) -> Vec<[u8; 3]> {
    (0..h)
        .flat_map(|y| {
            (0..w).map(move |x| {
                let g = ((x * 7 + y * 13) % 251) as u8;
                let t = (((x * 31) ^ (y * 17)) % 67) as u8;
                let s = g.wrapping_add(t);
                [s, g.wrapping_add(t / 2), 255 - s]
            })
        })
        .collect()
}

pub(super) fn run(paths: &[String]) {
    run_revision(paths, "5");
}

/// Explicit Rev7 probe; the original Rev5 entry retains its revision contract.
pub(super) fn run_rev7(paths: &[String]) {
    run_revision(paths, "7");
}

fn run_revision(paths: &[String], revision: &str) {
    assert!(
        !paths.is_empty(),
        "--prodqual requires explicit model paths"
    );
    assert_eq!(std::env::var("ZENSIM_FORMULA_REV").as_deref(), Ok(revision));
    let _guard = archmage::testing::lock_token_testing();
    let mut seeds = Vec::new();
    for (seed, path) in paths.iter().enumerate() {
        eprintln!("prodqual: seed {seed} model {path}");
        let bytes = std::fs::read(path).expect("read explicitly named model");
        let model = zenpredict::Model::from_bytes(&bytes).expect("parse packed model");
        assert_eq!(
            model
                .metadata()
                .get_utf8("zentrain.formula_revision")
                .unwrap(),
            revision
        );
        let request = Request::for_bake_bytes(&bytes).expect("declared extraction plan");
        let mut permutations = Vec::new();
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            eprintln!("prodqual: seed {seed} permutation {}", perm.label);
            let mut scorer = BakeScorer::new(&model)
                .expect("public scorer")
                .with_parallel(false);
            let mut rows = Vec::new();
            for (w, h) in GEOMETRIES {
                let pixels = reference(w, h);
                let source = RgbSlice::new(&pixels, w, h);
                let mut steering = scorer
                    .prepare_steering(&source, 8)
                    .expect("prepare steering");
                for dropped in DROPPED_BITS {
                    let mask = (255u32 << dropped) as u8;
                    let distorted: Vec<[u8; 3]> =
                        pixels.iter().map(|p| p.map(|c| c & mask)).collect();
                    let dest = RgbSlice::new(&distorted, w, h);
                    let cached = steering
                        .compute(&dest, None)
                        .expect("cached steering score");
                    // A separate scorer avoids aliasing the session's mutable borrow.
                    let mut direct = BakeScorer::new(&model).unwrap().with_parallel(false);
                    let served = direct
                        .compute(&source, &dest, None)
                        .expect("public pixel score");
                    let extracted =
                        research::extract(&request, &source, &dest).expect("canonical extractor");
                    let feature_score = direct
                        .score_features(extracted.values(), w as u32, h as u32, None)
                        .expect("feature score");
                    let identity_aware = direct
                        .score_features_with_identity(
                            extracted.values(),
                            w as u32,
                            h as u32,
                            None,
                            pixels == distorted,
                        )
                        .expect("identity-aware feature score");
                    let read_bits: Vec<u64> = request
                        .want()
                        .iter_slots()
                        .map(|id| extracted.values()[id].to_bits())
                        .collect();
                    let mismatches: Vec<usize> = request
                        .want()
                        .iter_slots()
                        .filter(|&id| {
                            served.features()[id].to_bits() != extracted.values()[id].to_bits()
                        })
                        .collect();
                    let cached_mismatches: Vec<usize> = request
                        .want()
                        .iter_slots()
                        .filter(|&id| {
                            cached.result().features()[id].to_bits()
                                != extracted.values()[id].to_bits()
                        })
                        .collect();
                    let scores = [
                        served.score(),
                        cached.result().score(),
                        feature_score,
                        identity_aware,
                    ];
                    rows.push(json!({
                        "width":w, "height":h, "dropped_bits":dropped,
                        "pixel_score":served.score(), "pixel_bits":served.score().to_bits(),
                        "cached_bits":cached.result().score().to_bits(),
                        "feature_score":feature_score, "feature_bits":feature_score.to_bits(),
                        "identity_aware_bits":identity_aware.to_bits(),
                        "finite":scores.iter().all(|s| s.is_finite()),
                        "feature_mismatches":mismatches,
                        "cached_feature_mismatches":cached_mismatches, "read_bits":read_bits,
                        "density_cells":cached.attribution().density().len(),
                        "sensitivity_bits":cached.sensitivities().iter().map(|x|x.to_bits()).collect::<Vec<_>>(),
                    }));
                }
            }
            permutations.push(json!({"label":perm.label, "rows":rows}));
        });
        seeds.push(
            json!({"seed":seed, "model":path, "caller_width":model.caller_input_width(),
            "declared_reads":request.want().iter_slots().count(),
            "permutations_run":report.permutations_run, "permutations":permutations}),
        );
    }
    println!(
        "{}",
        json!({"schema":2, "formula_revision":revision.parse::<u32>().unwrap(),
        "input":"synthetic integer texture; cumulative low-bit truncation",
        "geometries":GEOMETRIES, "dropped_bits":DROPPED_BITS, "seeds":seeds})
    );
}

// D12 measures the same prepared-map owner before and after the gradient change.
// Explicit model packet and output; synthetic texture only, no corpus discovery.
pub(super) fn timing(packet: &str, output: &str) {
    use std::io::Write;
    use std::time::Instant;
    assert_eq!(std::env::var("ZENSIM_FORMULA_REV").as_deref(), Ok("5"));
    #[cfg(not(target_arch = "x86_64"))]
    panic!("paired timing is registered for x86 v3");
    #[cfg(target_arch = "x86_64")]
    {
        archmage::X64V4xToken::dangerously_disable_token_process_wide(true).unwrap();
        archmage::X64V4Token::dangerously_disable_token_process_wide(true).unwrap();
    }
    let packet: serde_json::Value =
        serde_json::from_slice(&std::fs::read(packet).unwrap()).unwrap();
    let mut out = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output)
        .unwrap();
    for m in packet["models"].as_array().unwrap() {
        let bytes = std::fs::read(m["path"].as_str().unwrap()).unwrap();
        use sha2::{Digest, Sha256};
        let digest: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(digest, m["sha256"].as_str().unwrap());
        let model = zenpredict::Model::from_bytes(&bytes).unwrap();
        for side in [64, 256, 1024, 4096] {
            let pixels = reference(side, side);
            let distorted: Vec<_> = pixels.iter().map(|p| p.map(|x| x & 248)).collect();
            let src = RgbSlice::new(&pixels, side, side);
            let dst = RgbSlice::new(&distorted, side, side);
            let mut scorer = BakeScorer::new(&model).unwrap().with_parallel(false);
            let t = Instant::now();
            let mut worker = scorer.prepare_steering(&src, 8).unwrap();
            let prepare_seconds = t.elapsed().as_secs_f64();
            let mut samples = Vec::new();
            for repeat in 0..3 {
                let t = Instant::now();
                let result = worker.compute(&dst, None).unwrap();
                let seconds = t.elapsed().as_secs_f64();
                let feature_bits: Vec<_> = result
                    .result()
                    .features()
                    .iter()
                    .flat_map(|x| x.to_bits().to_le_bytes())
                    .collect();
                let feature_sha: String = Sha256::digest(&feature_bits)
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect();
                samples.push(json!({"repeat":repeat,"seconds":seconds,"score_bits":result.result().score().to_bits(),"features_sha256":feature_sha,"gain":std::hint::black_box(result.refinement_gain(0,0,32,32))}));
            }
            let record = json!({"model":m["name"],"model_sha256":digest,"side":side,"tier":"v3","bin":8,"prepare_seconds":prepare_seconds,"samples":samples});
            writeln!(out, "{record}").unwrap();
            out.flush().unwrap();
            eprintln!("D12 map timing {} {side} complete", m["name"]);
        }
    }
}
