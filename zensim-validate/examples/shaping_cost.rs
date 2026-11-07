//! Native vector-scoring instrument for SHAPING_INFERENCE_COST_PROTOCOL.md.
//! The caller owns scientific model admission and supplies pinned reference bits.
#[path = "../src/bin/bake_dial_refit/linear_basis.rs"]
mod linear_basis;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use zensim::{BakeScorer, corruption_head::CorruptionHead};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Arm {
    id: String,
    kind: Kind,
    model: PathBuf,
    sha256: String,
    expected_score_bits: Vec<u64>,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Kind {
    Bake,
    Tree,
    ShapedAdditive,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Workload {
    schema: String,
    admission_sha256: String,
    raw_ids: Vec<usize>,
    feature_bits: Vec<Vec<u32>>,
    arms: Vec<Arm>,
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn rows(workload: &Workload) -> Result<Vec<Vec<f64>>> {
    ensure!(workload.schema == "shaping-cost-v1", "unknown workload");
    ensure!(
        workload.admission_sha256.len() == 64
            && workload
                .admission_sha256
                .bytes()
                .all(|b| b.is_ascii_hexdigit()),
        "explicit scientific admission hash required"
    );
    ensure!(!workload.raw_ids.is_empty(), "empty feature layout");
    let unique: BTreeSet<_> = workload.raw_ids.iter().copied().collect();
    ensure!(unique.len() == workload.raw_ids.len(), "duplicate raw ID");
    ensure!(
        unique.last().copied().unwrap() < 1853,
        "raw ID outside revision 7"
    );
    ensure!(
        workload.feature_bits.len() == 40,
        "exactly forty rows required"
    );
    ensure!(!workload.arms.is_empty(), "no model arms");
    let names: BTreeSet<_> = workload.arms.iter().map(|a| &a.id).collect();
    ensure!(names.len() == workload.arms.len(), "duplicate model arm");
    for arm in &workload.arms {
        ensure!(!arm.id.is_empty(), "empty model identity");
        ensure!(
            arm.expected_score_bits.len() == 40,
            "reference coverage differs"
        );
        ensure!(
            arm.expected_score_bits
                .iter()
                .all(|b| f64::from_bits(*b).is_finite()),
            "nonfinite reference"
        );
    }
    workload
        .feature_bits
        .iter()
        .map(|row| {
            ensure!(row.len() == workload.raw_ids.len(), "row width differs");
            let mut dense = vec![0.0; 1853];
            for (&id, &bits) in workload.raw_ids.iter().zip(row) {
                let value = f32::from_bits(bits);
                ensure!(value.is_finite(), "nonfinite raw feature");
                dense[id] = f64::from(value);
            }
            Ok(dense)
        })
        .collect()
}

type Forward = Box<dyn FnMut(&[f64]) -> f64 + Send>;

fn forward(arm: &Arm, raw_ids: &[usize]) -> Result<Forward> {
    let bytes = std::fs::read(&arm.model)?;
    ensure!(
        digest(&bytes) == arm.sha256,
        "model hash differs: {}",
        arm.id
    );
    match arm.kind {
        Kind::Bake => {
            // A process owns one fixed set of models for its entire lifetime.
            let model = Box::leak(Box::new(
                zenpredict::Model::from_bytes(&bytes)
                    .map_err(|e| anyhow::anyhow!("model parse: {e:?}"))?,
            ));
            let mut scorer = BakeScorer::new(model).map_err(|e| anyhow::anyhow!("{e}"))?;
            // These research exports have a qualified vector contract. The
            // pixel-serving planner is a separate, stronger admission gate.
            let ids = zensim::declared_feature_ids(model)
                .context("explicit dense vector feature IDs required")?;
            ensure!(
                ids.iter().all(|id| raw_ids.contains(&usize::from(*id))),
                "workload omits a consumed input"
            );
            Ok(Box::new(move |row| {
                scorer
                    .score_features(row, 0, 0, None)
                    .expect("qualified forward")
            }))
        }
        Kind::ShapedAdditive => {
            let model = Box::leak(Box::new(
                zenpredict::Model::from_bytes(&bytes)
                    .map_err(|e| anyhow::anyhow!("model parse: {e:?}"))?,
            ));
            let text = model
                .metadata()
                .get_utf8("zentrain.research_basis")
                .map_err(|e| anyhow::anyhow!("basis metadata: {e:?}"))?;
            let basis: linear_basis::Spec = serde_json::from_str(text)?;
            basis.validate(raw_ids.len()).map_err(anyhow::Error::msg)?;
            ensure!(
                basis.width() == model.n_inputs() && model.n_outputs() == 1,
                "additive model shape differs"
            );
            ensure!(
                basis.calibration_ids.is_empty() && basis.calibration_targets.is_empty(),
                "unexpected additive calibration"
            );
            let mut predictor = zenpredict::Predictor::new(model);
            let ids = raw_ids.to_vec();
            let mut raw = vec![0.0f32; ids.len()];
            let mut shaped = vec![0.0f32; basis.width()];
            Ok(Box::new(move |row| {
                for (value, &id) in raw.iter_mut().zip(&ids) {
                    *value = row[id] as f32;
                }
                basis
                    .apply(&raw, &mut shaped)
                    .expect("qualified additive basis");
                f64::from(
                    predictor
                        .predict(&shaped)
                        .expect("qualified additive forward")[0],
                )
            }))
        }
        Kind::Tree => {
            let head = CorruptionHead::from_bytes(&bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
            ensure!(
                head.caller_input_width() == raw_ids.len(),
                "opaque tree width differs"
            );
            ensure!(
                head.declared_feature_ids()
                    .iter()
                    .copied()
                    .map(usize::from)
                    .eq(0..raw_ids.len()),
                "opaque tree IDs differ"
            );
            let ids = raw_ids.to_vec();
            let mut gathered = vec![0.0; ids.len()];
            Ok(Box::new(move |row| {
                for (value, &id) in gathered.iter_mut().zip(&ids) {
                    *value = row[id];
                }
                100.0
                    * head
                        .decision_function(&gathered)
                        .expect("qualified tree forward")
            }))
        }
    }
}

fn check(arm: &Arm, forward: &mut Forward, rows: &[Vec<f64>]) -> Result<()> {
    ensure!(
        digest(&std::fs::read(&arm.model)?) == arm.sha256,
        "model changed during execution"
    );
    for (index, (row, expected)) in rows.iter().zip(&arm.expected_score_bits).enumerate() {
        let actual = forward(row);
        ensure!(
            actual.is_finite() && actual.to_bits() == *expected,
            "{} row {index}: score bits {:016x} differ from {expected:016x}",
            arm.id,
            actual.to_bits()
        );
    }
    Ok(())
}

fn write(path: &Path, value: &serde_json::Value) -> Result<()> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(&serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

#[cfg(target_arch = "x86_64")]
fn dispatch() -> &'static str {
    use archmage::{SimdToken, X64V3Token, X64V4Token, X64V4xToken};
    if X64V4xToken::summon().is_some() {
        "auto-v4x"
    } else if X64V4Token::summon().is_some() {
        "auto-v4"
    } else if X64V3Token::summon().is_some() {
        "auto-v3"
    } else {
        "auto-scalar"
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn dispatch() -> &'static str {
    "automatic-non-x86"
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 4,
        "usage: shaping_cost qualify|bench WORKLOAD.json NEW_OUTPUT_DIR"
    );
    ensure!(
        matches!(args[1].as_str(), "qualify" | "bench"),
        "unknown mode"
    );
    let bytes = std::fs::read(&args[2])?;
    let mut workload: Workload = serde_json::from_slice(&bytes)?;
    for arm in &mut workload.arms {
        if arm.model.is_relative() {
            arm.model = Path::new(&args[2])
                .parent()
                .context("workload parent")?
                .join(&arm.model);
        }
    }
    let rows = Arc::new(rows(&workload)?);
    if args[1] == "bench" {
        ensure!(
            workload.arms.len() == 16,
            "all sixteen finalist models required"
        );
    }
    let output = Path::new(&args[3]);
    std::fs::create_dir(output).context("preserve existing output")?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let mut owners = Vec::new();
    for arm in &workload.arms {
        let mut owner =
            forward(arm, &workload.raw_ids).with_context(|| format!("arm {}", arm.id))?;
        check(arm, &mut owner, &rows)?;
        owners.push(Arc::new(Mutex::new(owner)));
    }
    let identity = json!({"schema":"shaping-cost-instrument-v1", "workload_sha256":digest(&bytes),
        "admission_sha256":workload.admission_sha256,
        "binary_sha256":digest(&std::fs::read(std::env::current_exe()?)?),
        "threads":1,"dispatch_ceiling":dispatch(),"architecture":std::env::consts::ARCH,"rows_per_call":40,"mode":args[1],"preflight_exact":true,
        "scope":"native vector scoring including per-row gathering and learned transforms; excludes model load and image extraction"});
    write(&output.join("IDENTITY.json"), &identity)?;
    let mut flags = Vec::new();
    let mut unreliable = false;
    let mut batches = 0;
    if args[1] == "bench" {
        while flags.iter().filter(|v| **v == Some(true)).count() < 32
            && flags.len() < 64
            && !unreliable
        {
            let needed =
                (32 - flags.iter().filter(|v| **v == Some(true)).count()).min(64 - flags.len());
            let result = zenbench::run(|suite| {
                suite.compare("shaping_native_40_rows", |group| {
                    group
                        .config()
                        .min_rounds(needed)
                        .max_rounds(needed)
                        .auto_rounds(false)
                        .warmup_time(if batches == 0 {
                            Duration::from_millis(100)
                        } else {
                            Duration::ZERO
                        })
                        .max_wall_time(Duration::from_secs(900));
                    for (arm, owner) in workload.arms.iter().zip(&owners) {
                        let owner = Arc::clone(owner);
                        let rows = Arc::clone(&rows);
                        group.bench(arm.id.clone(), move |b| {
                            let mut forward = owner.lock().unwrap();
                            b.iter(|| {
                                for row in rows.iter() {
                                    zenbench::black_box(forward(row));
                                }
                            });
                        });
                    }
                });
            });
            // Preserve raw evidence even when coverage or reliability refuses it.
            result.save(output.join(format!("zenbench-{batches:03}.json")))?;
            ensure!(result.comparisons.len() == 1, "comparison coverage differs");
            let comparison = &result.comparisons[0];
            let actual: BTreeSet<_> = comparison.benchmarks.iter().map(|b| &b.name).collect();
            let expected: BTreeSet<_> = workload.arms.iter().map(|a| &a.id).collect();
            ensure!(
                actual == expected && comparison.benchmarks.len() == expected.len(),
                "timed arm coverage differs"
            );
            let samples = &comparison.samples;
            for sample in samples {
                let order: BTreeSet<_> = sample.execution_order.iter().copied().collect();
                ensure!(
                    sample.iterations > 0
                        && sample.elapsed_ns.len() == expected.len()
                        && sample.compensated_ns.len() == expected.len()
                        && sample.execution_order.len() == expected.len()
                        && order.iter().copied().eq(0..expected.len()),
                    "incomplete paired round"
                );
            }
            ensure!(samples.len() <= needed, "unexpected round count");
            flags.extend(samples.iter().map(|s| s.gate_clean));
            unreliable |= result.unreliable;
            batches += 1;
            if samples.len() < needed {
                break;
            }
        }
    }
    for (arm, owner) in workload.arms.iter().zip(&owners) {
        check(arm, &mut owner.lock().unwrap(), &rows)?;
    }
    ensure!(
        digest(&std::fs::read(&args[2])?) == digest(&bytes),
        "workload changed"
    );
    let retained: Vec<_> = flags
        .iter()
        .enumerate()
        .filter_map(|(i, v)| (*v == Some(true)).then_some(i))
        .take(32)
        .collect();
    let complete = args[1] == "qualify" || (retained.len() == 32 && !unreliable);
    write(
        &output.join("RESULT.json"),
        &json!({"preflight_exact":true,"postflight_exact":true,
        "complete":complete,"timing_measured":args[1]=="bench","gate_clean":flags,
        "retained_round_indices":retained,"batches":batches,"unreliable":unreliable}),
    )?;
    ensure!(
        complete,
        "incomplete timing retained; retry in a new directory"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Workload {
        Workload {
            schema: "shaping-cost-v1".into(),
            admission_sha256: "a".repeat(64),
            raw_ids: vec![3, 0],
            feature_bits: vec![vec![1.5f32.to_bits(), (-0.0f32).to_bits()]; 40],
            arms: vec![Arm {
                id: "test".into(),
                kind: Kind::Bake,
                model: "missing".into(),
                sha256: "b".repeat(64),
                expected_score_bits: vec![100.0f64.to_bits(); 40],
            }],
        }
    }
    #[test]
    fn rows_preserve_feature_order_and_bits() {
        let rows = rows(&fixture()).unwrap();
        assert_eq!(rows[0][3], 1.5);
        assert_eq!(rows[0][0].to_bits(), (-0.0f64).to_bits());
        assert_eq!(rows[0][2], 0.0);
    }
    #[test]
    fn malformed_inputs_refuse_before_model_loading() {
        let mut w = fixture();
        w.raw_ids[0] = 0;
        assert!(rows(&w).is_err());
        let mut w = fixture();
        w.feature_bits.pop();
        assert!(rows(&w).is_err());
        let mut w = fixture();
        w.feature_bits[0][0] = f32::NAN.to_bits();
        assert!(rows(&w).is_err());
        let mut w = fixture();
        w.arms[0].expected_score_bits.pop();
        assert!(rows(&w).is_err());
        let mut w = fixture();
        w.arms[0].expected_score_bits[0] = f64::INFINITY.to_bits();
        assert!(rows(&w).is_err());
    }
}
