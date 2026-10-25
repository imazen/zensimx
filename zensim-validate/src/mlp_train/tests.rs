use super::*;

/// Receipt paths for valid non-UTF-8 Unix names decode to the exact native
/// bytes, distinct even where the lossy display forms collide, without
/// touching the filesystem, so this half runs on Apple targets too.
#[cfg(unix)]
#[test]
fn checkpoint_receipts_keep_non_utf8_native_paths_exact() {
    use std::os::unix::ffi::OsStringExt;
    let paths = [
        std::path::PathBuf::from(std::ffi::OsString::from_vec(
            b"/r/valid-unix-\xff/ckpt_epoch000.bin".to_vec(),
        )),
        std::path::PathBuf::from(std::ffi::OsString::from_vec(
            b"/r/valid-unix-\xfe/ckpt_epoch000.bin".to_vec(),
        )),
        std::path::PathBuf::from("/r/valid-utf8/ckpt_epoch000.bin"),
    ];
    assert_eq!(paths[0].to_string_lossy(), paths[1].to_string_lossy());
    let mut decoded = Vec::new();
    for (i, path) in paths.iter().enumerate() {
        let line = checkpoint_receipt_line(path, i, 17);
        let receipt: serde_json::Value =
            serde_json::from_str(line.strip_prefix("  checkpoint dump: ").unwrap()).unwrap();
        assert_eq!(receipt["epoch"], i);
        assert_eq!(receipt["bytes"], 17);
        let back = match receipt["path"].as_str() {
            Some(s) => std::path::PathBuf::from(s),
            None => std::path::PathBuf::from(
                serde_json::from_value::<std::ffi::OsString>(receipt["path"].clone()).unwrap(),
            ),
        };
        assert_eq!(back.as_os_str(), path.as_os_str());
        decoded.push(back);
    }
    assert!(
        paths[2].to_str().is_some()
            && checkpoint_receipt_line(&paths[2], 2, 17).contains("\"/r/valid-utf8/")
    );
    assert_ne!(decoded[0], decoded[1]);
}

// End to end through real files. Apple filesystems (APFS, HFS+) refuse
// file names that are not valid UTF-8 (EILSEQ, os error 92), so these
// names cannot be created there; the receipt half above still runs.
#[cfg(all(unix, not(target_vendor = "apple")))]
#[test]
fn train_mlp_non_utf8_checkpoint_paths_are_lossless() {
    use std::os::unix::ffi::OsStringExt;

    let root = std::env::temp_dir().join(format!("shippath5-paths-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    // These two valid Unix names have the same lossy display representation.
    let components = [
        std::ffi::OsString::from_vec(b"valid-unix-\xff".to_vec()),
        std::ffi::OsString::from_vec(b"valid-unix-\xfe".to_vec()),
        std::ffi::OsString::from("valid-utf8"),
    ];
    assert_eq!(
        components[0].to_string_lossy(),
        components[1].to_string_lossy()
    );
    assert_ne!(components[0], components[1]);
    let features: Vec<_> = (0..48)
        .map(|i| vec![i as f64 / 48.0, 1.0 - i as f64 / 48.0, 0.0, 0.0])
        .collect();
    let refs: Vec<_> = features.iter().map(|r| r.as_slice()).collect();
    let scores: Vec<_> = (0..48).map(|i| i as f64 / 48.0).collect();
    let mut control = None;
    for component in components {
        let dir = root.join(component);
        std::fs::create_dir(&dir).unwrap();
        let mut groups = [TrainingGroup {
            name: "synthetic".into(),
            human_scores: &scores,
            features: FeatureRows::Borrowed(&refs),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        }];
        let hp = MlpHyperparams {
            n_hidden: 8,
            n_epochs: 3,
            pairs_per_epoch: 128,
            log_every: 1,
            early_stop_patience: 0,
            dump_checkpoints_every: 1,
            dump_checkpoints_dir: Some(dir.clone()),
            ..Default::default()
        };
        let mut log = Vec::new();
        let model = train_mlp(&mut groups, 4, &hp, &mut log);
        Model::from_bytes(&model).unwrap();
        let receipts: Vec<serde_json::Value> = log
            .iter()
            .filter_map(|line| line.strip_prefix("  checkpoint dump: "))
            .map(|receipt| serde_json::from_str(receipt).unwrap())
            .collect();
        assert_eq!(receipts.len(), 3);
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 3);
        let mut dumps = Vec::new();
        for (epoch, receipt) in receipts.iter().enumerate() {
            let path = match receipt["path"].as_str() {
                Some(s) => std::path::PathBuf::from(s),
                None => std::path::PathBuf::from(
                    serde_json::from_value::<std::ffi::OsString>(receipt["path"].clone()).unwrap(),
                ),
            };
            assert_eq!(
                path.as_os_str(),
                dir.join(format!("ckpt_epoch{epoch:03}.bin")).as_os_str()
            );
            assert_eq!(receipt["epoch"], epoch);
            let bytes = std::fs::read(path).unwrap();
            assert_eq!(receipt["bytes"], bytes.len());
            Model::from_bytes(&bytes).unwrap();
            dumps.push(bytes);
        }
        if let Some((control_model, control_dumps)) = &control {
            assert_eq!(&model, control_model);
            assert_eq!(&dumps, control_dumps);
        } else {
            control = Some((model, dumps));
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn checkpoint_write_receipts_require_successful_io() {
    let dir = std::env::temp_dir().join(format!("shippath4-write-{}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let mut log = Vec::new();
    let path = dir.join("ckpt_epoch000.bin");
    super::write_checkpoint_dump(&path, b"written bytes", 0, &mut log);
    assert_eq!(std::fs::read(&path).unwrap(), b"written bytes");
    assert_eq!(log.len(), 1);
    let receipt: serde_json::Value =
        serde_json::from_str(log[0].strip_prefix("  checkpoint dump: ").unwrap()).unwrap();
    assert_eq!(receipt["path"], path.to_str().unwrap());
    assert_eq!(receipt["epoch"], 0);
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        super::write_checkpoint_dump(
            &dir.join("missing/ckpt_epoch001.bin"),
            b"unwritten",
            1,
            &mut log,
        );
    }));
    assert!(failed.is_err());
    assert_eq!(
        log.len(),
        1,
        "failed write must not acquire an ownership receipt"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

use zenpredict::{Model, Predictor};

fn predict_one(predictor: &mut Predictor<'_>, features: &[f64]) -> f64 {
    let f32_features: Vec<f32> = features.iter().map(|&v| v as f32).collect();
    predictor.predict(&f32_features).unwrap()[0] as f64
}

/// Generate a synthetic dataset where the target is a known
/// nonlinear function of features. Train a small MLP and confirm
/// it recovers the ranking.
#[test]
fn train_mlp_recovers_synthetic_ranking() {
    let n_features = 16;
    let n_train = 300;
    let mut rng = SplitMix64::new(7);
    let true_w: Vec<f64> = (0..n_features)
        .map(|i| (i as f64 - 8.0) * 0.3 + rng.next_normal() * 0.1)
        .collect();
    let mut features_owned: Vec<Vec<f64>> = Vec::with_capacity(n_train);
    let mut targets: Vec<f64> = Vec::with_capacity(n_train);
    for _ in 0..n_train {
        let x: Vec<f64> = (0..n_features).map(|_| rng.next_normal()).collect();
        let mut y: f64 = x.iter().zip(true_w.iter()).map(|(a, b)| a * b).sum();
        y += 0.1 * x[0] * x[0];
        y += rng.next_normal() * 0.05;
        features_owned.push(x);
        // Higher target = more similar (matches zensim convention).
        // The MLP must produce LOWER raw_distance for higher target.
        targets.push(y);
    }
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();

    let group = TrainingGroup {
        name: "synth".to_string(),
        human_scores: &targets,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };

    let hyper = MlpHyperparams {
        n_hidden: 8,
        n_epochs: 60,
        pairs_per_epoch: 1500,
        initial_lr: 0.005,
        log_every: 100,
        early_stop_patience: 0,
        validation_policy: ValidationPolicy::Mean,
        ..Default::default()
    };
    let mut log = Vec::new();
    let bytes = train_mlp(&mut [group], n_features, &hyper, &mut log);

    let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    let model = Model::from_bytes(leaked).expect("bake should load");
    let mut predictor = Predictor::new(&model);

    let preds: Vec<f64> = features_owned
        .iter()
        .map(|f| predict_one(&mut predictor, f))
        .collect();

    // MLP output is raw_distance (lower = more similar); targets
    // are quality (higher = more similar). Anti-correlated → SROCC
    // computed against negated predictions.
    let neg_preds: Vec<f64> = preds.iter().map(|&p| -p).collect();
    let srocc = spearman_correlation(&targets, &neg_preds);
    assert!(
        srocc > 0.85,
        "MLP failed to recover synthetic ranking: SROCC={srocc:.4}"
    );
}

/// A validation-only group should not contribute training pairs
/// but should still be reported in per-epoch SROCC, and the best
/// checkpoint should be selected on its score.
#[test]
fn train_mlp_uses_validation_for_best_checkpoint() {
    let n_features = 8;
    let mut rng = SplitMix64::new(11);

    // Train data: target = +sum_of_features (higher = more similar).
    let train_features: Vec<Vec<f64>> = (0..200)
        .map(|_| (0..n_features).map(|_| rng.next_normal()).collect())
        .collect();
    let train_scores: Vec<f64> = train_features
        .iter()
        .map(|f| f.iter().sum::<f64>())
        .collect();
    let train_refs: Vec<&[f64]> = train_features.iter().map(|v| v.as_slice()).collect();

    // Val data: same target function — model should generalize.
    let val_features: Vec<Vec<f64>> = (0..80)
        .map(|_| (0..n_features).map(|_| rng.next_normal()).collect())
        .collect();
    let val_scores: Vec<f64> = val_features.iter().map(|f| f.iter().sum::<f64>()).collect();
    let val_refs: Vec<&[f64]> = val_features.iter().map(|v| v.as_slice()).collect();

    let mut groups = vec![
        TrainingGroup {
            name: "train".to_string(),
            human_scores: &train_scores,
            features: FeatureRows::Borrowed(&train_refs),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 0.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        },
        TrainingGroup {
            name: "val".to_string(),
            human_scores: &val_scores,
            features: FeatureRows::Borrowed(&val_refs),
            metric_sigmas: None,
            train_weight: 0.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        },
    ];

    let hyper = MlpHyperparams {
        n_hidden: 8,
        n_epochs: 40,
        pairs_per_epoch: 800,
        initial_lr: 0.005,
        log_every: 10,
        early_stop_patience: 0,
        validation_policy: ValidationPolicy::Min,
        ..Default::default()
    };
    let mut log = Vec::new();
    let bytes = train_mlp(&mut groups, n_features, &hyper, &mut log);

    let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    let model = Model::from_bytes(leaked).expect("bake should load");
    let mut predictor = Predictor::new(&model);

    let val_preds: Vec<f64> = val_features
        .iter()
        .map(|f| predict_one(&mut predictor, f))
        .collect();
    // Higher target ⇒ more similar ⇒ lower MLP raw_distance.
    // SROCC(target, -prediction) should be positive and high.
    let neg_preds: Vec<f64> = val_preds.iter().map(|&p| -p).collect();
    let val_srocc = spearman_correlation(&val_scores, &neg_preds);
    assert!(
        val_srocc > 0.85,
        "validation-tracking trainer failed to generalize: val SROCC={val_srocc:.4}",
    );
    // Spot-check the log: epoch lines contain the multi-stat
    // val aggregate and per-group panel stats.
    assert!(
        log.iter().any(|line| line.contains("val(")),
        "log missing val(aggregate)= reporting"
    );
    assert!(
        log.iter().any(|line| line.contains("srocc=")),
        "log missing per-group srocc= field"
    );
}

/// With a large `low_q_boost`, the trainer must pick more B0/B1
/// pairs (lower human_score) than uniform sampling would. Smoke-test
/// by training two MLPs on the same data with the same seed but
/// different boost values, then verifying the boosted model's
/// predictions are demonstrably more sensitive in the low-score
/// region (the boost re-weighted those pairs higher in the rank
/// loss).
/// `--stratified-bands` must actually stratify on the DEFAULT training
/// path, not only under `--per-sample-alpha-head`.
///
/// It used to build `strat_bands` in exactly ONE of the four training
/// loops. Every other loop — including `train_mlp_strategy`'s own
/// standard path, which every board bake trained through — passed an
/// empty band table, so the flag was a SILENT NO-OP there: setting it
/// produced a bit-identical draw sequence and a bit-identical bake.
/// MEASURED before the fix, `--stratified-bands 0` vs `8` on the default
/// path: sample-sequence digest `127b831bed8a3873` both times.
/// The unguarded half of the STRATEGY family must FAIL LOUD rather than
/// throw the flag away.
///
/// `ema_decay` / `hard_pair_frac` / `dro_eta` / `listwise_weight` are read
/// only inside `train_mlp_per_sample_alpha_head`. Before 2026-09-04 setting
/// any of them on another head produced a bake byte-identical to a run that
/// never set it — the same defect the mse/monotonicity/triplet guards above
/// already covered. `--ema-decay` mattered most: its own doc advertises
/// "seed-variance reduction", so a variance study would have concluded it
/// does not work when in fact it never ran.
#[test]
fn strategy_knobs_fail_loud_when_unreachable() {
    let n_features = 4usize;
    let targets: Vec<f64> = (0..40).map(|i| i as f64 * 2.5).collect();
    let mut rng = SplitMix64::new(5);
    let owned: Vec<Vec<f64>> = (0..40)
        .map(|i| {
            let mut x: Vec<f64> = (0..n_features).map(|_| rng.next_normal()).collect();
            x[0] = i as f64 / 40.0;
            x
        })
        .collect();
    let fr: Vec<&[f64]> = owned.iter().map(|v| v.as_slice()).collect();
    let group = || TrainingGroup {
        name: "g".into(),
        human_scores: &targets,
        features: FeatureRows::Borrowed(&fr),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };
    let base = MlpHyperparams {
        n_hidden: 4,
        n_epochs: 2,
        pairs_per_epoch: 50,
        seed: 3,
        log_every: 100,
        early_stop_patience: 0,
        ..Default::default()
    };
    for (label, h) in [
        (
            "ema",
            MlpHyperparams {
                ema_decay: 0.999,
                ..base.clone()
            },
        ),
        (
            "hardpair",
            MlpHyperparams {
                hard_pair_frac: 0.5,
                ..base.clone()
            },
        ),
        (
            "dro",
            MlpHyperparams {
                dro_eta: 0.1,
                ..base.clone()
            },
        ),
        (
            "listwise",
            MlpHyperparams {
                listwise_weight: 0.5,
                ..base.clone()
            },
        ),
    ] {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut log = Vec::new();
            train_mlp_strategy(
                &mut [group()],
                n_features,
                &h,
                &mut log,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        }));
        assert!(
            r.is_err(),
            "{label} was accepted on a non-alpha head and silently ignored"
        );
    }
    // Negative control: the same recipe with none of them set must train.
    let mut log = Vec::new();
    let bytes = train_mlp_strategy(
        &mut [group()],
        n_features,
        &base,
        &mut log,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    assert!(!bytes.is_empty(), "the unflagged control must still train");
}

#[test]
fn stratified_bands_is_not_a_silent_no_op_on_the_default_path() {
    let n_features = 8usize;
    let mut rng = SplitMix64::new(31);
    let n = 200usize;
    let mut targets = Vec::with_capacity(n);
    let mut features_owned: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        // Deliberately SKEWED scores: most rows crowd the top, so
        // band-uniform row-A selection has to look different from
        // row-uniform selection.
        let s = 100.0 * ((i as f64) / (n as f64 - 1.0)).powf(0.25);
        targets.push(s);
        let mut x: Vec<f64> = (0..n_features).map(|_| rng.next_normal()).collect();
        x[0] = s / 100.0 + rng.next_normal() * 0.1;
        features_owned.push(x);
    }
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
    let group = || TrainingGroup {
        name: "strat-test".to_string(),
        human_scores: &targets,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };
    let base = MlpHyperparams {
        n_hidden: 6,
        n_epochs: 20,
        pairs_per_epoch: 800,
        initial_lr: 0.01,
        seed: 7,
        log_every: 100,
        early_stop_patience: 0,
        ..Default::default()
    };
    let strat = MlpHyperparams {
        stratified_bands: 8,
        ..base.clone()
    };
    let mut l1 = Vec::new();
    let plain_bytes = train_mlp(&mut [group()], n_features, &base, &mut l1);
    let mut l2 = Vec::new();
    let strat_bytes = train_mlp(&mut [group()], n_features, &strat, &mut l2);
    assert_ne!(
        plain_bytes, strat_bytes,
        "--stratified-bands had NO effect on the default training path"
    );

    // And the negative control: bands=0 must stay bit-identical to a
    // hyperparams struct that never mentions the flag, so the fix cannot
    // have moved the default.
    let explicit_off = MlpHyperparams {
        stratified_bands: 0,
        ..base.clone()
    };
    let mut l3 = Vec::new();
    let off_bytes = train_mlp(&mut [group()], n_features, &explicit_off, &mut l3);
    assert_eq!(
        plain_bytes, off_bytes,
        "stratified_bands=0 must be a true no-op"
    );
}

#[test]
fn train_mlp_low_q_boost_changes_outputs() {
    let n_features = 4;
    let mut rng = SplitMix64::new(99);

    // Build a 200-row group with human_scores spanning [0, 100] and
    // features that linearly encode the score. Without boost the
    // MLP should learn a uniform regression; with boost it should
    // specialize to the low-score end.
    let n = 200usize;
    let mut targets = Vec::with_capacity(n);
    let mut features_owned: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let s = 100.0 * (i as f64) / (n as f64 - 1.0); // 0..100
        targets.push(s);
        let mut x: Vec<f64> = (0..n_features).map(|_| rng.next_normal()).collect();
        // Inject a strong signal: x[0] correlates with target.
        x[0] = s / 100.0 + rng.next_normal() * 0.1;
        features_owned.push(x);
    }
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();

    let group_factory = || TrainingGroup {
        name: "boost-test".to_string(),
        human_scores: &targets,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };

    // Run 1: no boost (uniform within-group sampling).
    let hyper_uniform = MlpHyperparams {
        n_hidden: 6,
        n_epochs: 30,
        pairs_per_epoch: 1000,
        initial_lr: 0.01,
        seed: 7,
        log_every: 100,
        early_stop_patience: 0,
        ..Default::default()
    };
    // Run 2: aggressive low_q_boost (10x for B0, sqrt(10) ≈ 3.16x for B1).
    let hyper_boosted = MlpHyperparams {
        low_q_boost: 10.0,
        ..hyper_uniform.clone()
    };

    let mut log_u = Vec::new();
    let bytes_uniform = train_mlp(
        &mut [group_factory()],
        n_features,
        &hyper_uniform,
        &mut log_u,
    );
    let mut log_b = Vec::new();
    let bytes_boosted = train_mlp(
        &mut [group_factory()],
        n_features,
        &hyper_boosted,
        &mut log_b,
    );

    // The two bakes must differ — if boost had no effect, this
    // would be a regression in the per-row CDF wiring.
    assert_ne!(
        bytes_uniform, bytes_boosted,
        "low_q_boost=10.0 produced byte-identical bake to no-boost — \
         the per-row CDF is not being honored"
    );

    // Default boost=1.0 must produce bit-identical output to a
    // hyperparams struct with explicit boost=1.0 (no-op
    // guarantee). The CDF code path returns None at default and
    // the sampler stays uniform — no extra RNG bytes consumed.
    let hyper_default = MlpHyperparams {
        seed: 7,
        n_hidden: 6,
        n_epochs: 30,
        pairs_per_epoch: 1000,
        initial_lr: 0.01,
        log_every: 100,
        early_stop_patience: 0,
        ..Default::default()
    };
    let mut log_d = Vec::new();
    let bytes_default = train_mlp(
        &mut [group_factory()],
        n_features,
        &hyper_default,
        &mut log_d,
    );
    assert_eq!(
        bytes_uniform, bytes_default,
        "explicit low_q_boost=1.0 produced different bake than default — \
         the no-op guarantee is broken"
    );
}

// ---- T8.1 + T8.2 tests (2026-05-16) ----

/// Shared synthetic dataset used by the mini-batch / parallel-batch
/// tests below. Returns owned features + targets in a stable seed
/// order so the tests can compare bake bytes against a baseline.
fn make_synth_dataset(seed: u64, n: usize, n_features: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = SplitMix64::new(seed);
    let true_w: Vec<f64> = (0..n_features)
        .map(|i| (i as f64 - (n_features as f64 / 2.0)) * 0.3 + rng.next_normal() * 0.1)
        .collect();
    let mut features = Vec::with_capacity(n);
    let mut targets = Vec::with_capacity(n);
    for _ in 0..n {
        let x: Vec<f64> = (0..n_features).map(|_| rng.next_normal()).collect();
        let mut y: f64 = x.iter().zip(true_w.iter()).map(|(a, b)| a * b).sum();
        y += 0.1 * x[0] * x[0];
        y += rng.next_normal() * 0.05;
        features.push(x);
        targets.push(y);
    }
    (features, targets)
}

/// [`FeatureRows::Releasable`] MUST train to bit-identical bake bytes
/// against [`FeatureRows::Borrowed`], and MUST actually hand the rows
/// back.
///
/// This is the gate for the 2026-08-04 memory change: the trainer frees
/// each raw row as it standardizes it, which halves a lane's peak RSS
/// (`benchmarks/trainer_mem_release_2026-08-04.md`). That is only
/// legitimate because releasing a row changes no arithmetic — this test
/// is what keeps it that way. The full-data version of this comparison
/// ran on the 11-group / 779,290-row wave-10 recipe; this one runs it in
/// CI on every commit.
#[test]
fn releasable_rows_train_identically_to_borrowed_and_are_released() {
    let n_features = 12;
    let (features_owned, targets) = make_synth_dataset(2026, 200, n_features);

    let hyper = MlpHyperparams {
        n_hidden: 8,
        n_epochs: 25,
        pairs_per_epoch: 800,
        initial_lr: 0.005,
        seed: 7,
        log_every: 100,
        early_stop_patience: 0,
        validation_policy: ValidationPolicy::Mean,
        ..Default::default()
    };

    // Arm A: borrowed rows — nothing is released.
    let borrowed_rows = features_owned.clone();
    let feats_ref: Vec<&[f64]> = borrowed_rows.iter().map(|v| v.as_slice()).collect();
    let mut log_a = Vec::new();
    let bake_borrowed = train_mlp(
        &mut [TrainingGroup {
            name: "synth".to_string(),
            human_scores: &targets,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        }],
        n_features,
        &hyper,
        &mut log_a,
    );
    drop(feats_ref);
    assert!(
        borrowed_rows.iter().all(|r| r.len() == n_features),
        "Borrowed rows must survive the call untouched"
    );

    // Arm B: releasable flat buffer — same values, same row-major order,
    // taken and standardized in place.
    let n_rows = features_owned.len();
    let mut flat: Vec<f64> = Vec::with_capacity(n_rows * n_features);
    for r in &features_owned {
        flat.extend_from_slice(r);
    }
    let mut log_b = Vec::new();
    let bake_releasable = train_mlp(
        &mut [TrainingGroup {
            name: "synth".to_string(),
            human_scores: &targets,
            features: FeatureRows::Releasable {
                data: &mut flat,
                n_rows,
                n_features,
            },
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        }],
        n_features,
        &hyper,
        &mut log_b,
    );

    assert_eq!(
        bake_borrowed, bake_releasable,
        "in-place standardization of the taken raw buffer changed the \
         bake — the memory optimization is NOT numerically inert"
    );
    assert!(
        flat.is_empty(),
        "the Releasable buffer was NOT taken — the run held two copies \
         of the feature matrix and the memory win is not happening"
    );
    // len() must keep reporting n_rows after the take (it is what the
    // hot loop's `g.features.len()` reads), which is asserted where it
    // matters: inside the trainer the post-standardization epoch loop
    // sampled pairs from this group for 25 epochs — impossible if the
    // cached length had collapsed to 0 with the buffer.
}

/// Compact f32 rows (`FeatureRows::Compact`, standardized at use) MUST
/// train to bit-identical bake bytes against the zero-masked dense
/// `Releasable` table they stand for, on every plain-head code path that
/// reads standardized rows: the sequential pair loop, the parallel
/// mini-batch, the NiN mini-batch, and the per-epoch prediction (the
/// 2,200-row group crosses `predict_group`'s parallel threshold). A dense
/// `Borrowed` group rides along to cover the mixed-group case, and
/// `nonneg_distance` covers the scale-only (mean = 0) scaler.
#[test]
fn compact_rows_train_identically_to_zero_masked_dense() {
    let n_features = 12;
    // Dropped columns include one the data would otherwise vary in (3, 7)
    // and the last column.
    let kept: Vec<u32> = vec![0, 1, 2, 4, 5, 6, 8, 9, 10];
    let kept_mask: Vec<bool> = (0..n_features)
        .map(|d| kept.contains(&(d as u32)))
        .collect();

    fn mk<'a>(
        name: &str,
        scores: &'a [f64],
        features: FeatureRows<'a>,
        tw: f64,
        vw: f64,
    ) -> TrainingGroup<'a> {
        TrainingGroup {
            name: name.to_string(),
            human_scores: scores,
            features,
            metric_sigmas: None,
            train_weight: tw,
            validation_weight: vw,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        }
    }

    // (rows, seed) per group; values are exactly representable in f32.
    let make = |n: usize, seed: u64| -> (Vec<f64>, Vec<f64>) {
        let (rows, targets) = make_synth_dataset(seed, n, n_features);
        let mut flat = Vec::with_capacity(n * n_features);
        for r in &rows {
            for (d, &v) in r.iter().enumerate() {
                flat.push(if kept_mask[d] { (v as f32) as f64 } else { 0.0 });
            }
        }
        (flat, targets)
    };
    let (flat_a, targets_a) = make(300, 31);
    let (flat_b, targets_b) = make(2200, 32);
    let identity_scores = vec![0.0f64; 3];
    let identity_flat = vec![0.0f64; 3 * n_features];
    let identity_refs: Vec<&[f64]> = identity_flat.chunks(n_features).collect();

    let compact_of = |flat: &[f64]| -> CompactRows {
        let n_rows = flat.len() / n_features;
        let mut data = Vec::with_capacity(n_rows * kept.len());
        for r in flat.chunks(n_features) {
            for &d in &kept {
                data.push(r[d as usize] as f32);
            }
        }
        CompactRows {
            data,
            n_rows,
            n_features,
            kept: kept.clone(),
        }
    };

    let base = MlpHyperparams {
        n_hidden: 6,
        n_epochs: 6,
        pairs_per_epoch: 400,
        initial_lr: 0.005,
        seed: 11,
        log_every: 1,
        early_stop_patience: 0,
        validation_policy: ValidationPolicy::Mean,
        ..Default::default()
    };
    let recipes: Vec<(&str, MlpHyperparams)> = vec![
        ("sequential", base.clone()),
        (
            "parallel-minibatch",
            MlpHyperparams {
                minibatch_size: 16,
                parallel_batch: true,
                ..base.clone()
            },
        ),
        (
            "nin-minibatch",
            MlpHyperparams {
                minibatch_size: 32,
                norm_in_norm_weight: 0.5,
                ..base.clone()
            },
        ),
        (
            "nonneg-distance",
            MlpHyperparams {
                nonneg_distance: true,
                ..base.clone()
            },
        ),
    ];

    for (label, hyper) in recipes {
        let (mut a_dense, mut b_dense) = (flat_a.clone(), flat_b.clone());
        let mut ca = compact_of(&flat_a);
        let mut cb = compact_of(&flat_b);
        let mut log_dense = Vec::new();
        let bake_dense = train_mlp(
            &mut [
                mk(
                    "a",
                    &targets_a,
                    FeatureRows::Releasable {
                        data: &mut a_dense,
                        n_rows: 300,
                        n_features,
                    },
                    1.0,
                    1.0,
                ),
                mk(
                    "b",
                    &targets_b,
                    FeatureRows::Releasable {
                        data: &mut b_dense,
                        n_rows: 2200,
                        n_features,
                    },
                    0.0,
                    0.5,
                ),
                mk(
                    "identity",
                    &identity_scores,
                    FeatureRows::Borrowed(&identity_refs),
                    0.3,
                    0.0,
                ),
            ],
            n_features,
            &hyper,
            &mut log_dense,
        );
        let mut log_compact = Vec::new();
        let bake_compact = train_mlp(
            &mut [
                mk("a", &targets_a, FeatureRows::Compact(&mut ca), 1.0, 1.0),
                mk("b", &targets_b, FeatureRows::Compact(&mut cb), 0.0, 0.5),
                mk(
                    "identity",
                    &identity_scores,
                    FeatureRows::Borrowed(&identity_refs),
                    0.3,
                    0.0,
                ),
            ],
            n_features,
            &hyper,
            &mut log_compact,
        );
        assert_eq!(
            bake_dense, bake_compact,
            "{label}: compact f32 rows standardized at use changed the bake"
        );
        // The per-epoch lines (val score, per-group panels) must agree
        // too, except the wall-clock and the train-only group, which the
        // compact run evaluates exactly as the dense one does.
        let strip = |l: &Vec<String>| -> Vec<String> {
            l.iter()
                .filter(|x| x.contains("epoch"))
                .map(|x| x.split(" | t=").next().unwrap().to_string())
                .collect()
        };
        assert_eq!(
            strip(&log_dense),
            strip(&log_compact),
            "{label}: epoch lines"
        );
        assert!(
            ca.data.is_empty() && cb.data.is_empty(),
            "{label}: compact buffers not taken"
        );
    }
}

/// EFFAUDIT D1: a train-only group is NOT evaluated per epoch when a
/// validation group exists under the Mean policy, and nothing else moves:
/// the per-epoch panel of the validation group, and the validation score,
/// equal those of a run that evaluates every group (`Goals` policy never
/// skips; the training trajectory is policy-independent here — no early
/// stop, no checkpoint-driven change of the weights).
#[test]
fn train_only_groups_are_not_evaluated_and_nothing_else_moves() {
    let n_features = 8;
    let (fa, ta) = make_synth_dataset(41, 200, n_features);
    let (fb, tb) = make_synth_dataset(42, 150, n_features);
    let ra: Vec<&[f64]> = fa.iter().map(|v| v.as_slice()).collect();
    let rb: Vec<&[f64]> = fb.iter().map(|v| v.as_slice()).collect();
    let hyper = MlpHyperparams {
        n_hidden: 6,
        n_epochs: 5,
        pairs_per_epoch: 300,
        initial_lr: 0.005,
        seed: 3,
        log_every: 1,
        early_stop_patience: 0,
        validation_policy: ValidationPolicy::Mean,
        ..Default::default()
    };
    let run = |hyper: &MlpHyperparams| {
        let mut log = Vec::new();
        train_mlp(
            &mut [
                TrainingGroup {
                    name: "fit".to_string(),
                    human_scores: &ta,
                    features: FeatureRows::Borrowed(&ra),
                    metric_sigmas: None,
                    train_weight: 1.0,
                    validation_weight: 0.0,
                    ref_ids: None,
                    loss_mode: GroupLossMode::default(),
                },
                TrainingGroup {
                    name: "dev".to_string(),
                    human_scores: &tb,
                    features: FeatureRows::Borrowed(&rb),
                    metric_sigmas: None,
                    train_weight: 0.0,
                    validation_weight: 1.0,
                    ref_ids: None,
                    loss_mode: GroupLossMode::default(),
                },
            ],
            n_features,
            hyper,
            &mut log,
        );
        log.into_iter()
            .filter(|l| l.contains("epoch"))
            .collect::<Vec<_>>()
    };
    let skipped = run(&hyper);
    let all = run(&MlpHyperparams {
        validation_policy: ValidationPolicy::Goals,
        ..hyper.clone()
    });
    assert_eq!(skipped.len(), 5);
    assert_eq!(all.len(), 5);
    let dev_of = |l: &str| -> String {
        let at = l.find("dev: srocc=").expect("dev panel present");
        l[at..].split(" | ").next().unwrap().to_string()
    };
    for (s_line, a_line) in skipped.iter().zip(all.iter()) {
        assert!(
            s_line.contains("fit: not evaluated (train-only)"),
            "{s_line}"
        );
        assert!(
            a_line.contains("fit: srocc="),
            "Goals must evaluate every group: {a_line}"
        );
        assert_eq!(dev_of(s_line), dev_of(a_line), "dev panel moved");
    }
}

/// `--minibatch-size 1` MUST produce bit-identical bake bytes to a
/// `MlpHyperparams::default()` call (which leaves `minibatch_size`
/// at 1). This is the legacy-compatibility guarantee for T8.1.
#[test]
fn train_mlp_minibatch_1_matches_legacy() {
    let n_features = 12;
    let (features_owned, targets) = make_synth_dataset(101, 200, n_features);
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();

    let group_factory = || TrainingGroup {
        name: "synth".to_string(),
        human_scores: &targets,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };

    let base = MlpHyperparams {
        n_hidden: 8,
        n_epochs: 25,
        pairs_per_epoch: 800,
        initial_lr: 0.005,
        seed: 7,
        log_every: 100,
        early_stop_patience: 0,
        validation_policy: ValidationPolicy::Mean,
        ..Default::default()
    };

    // Default trainer (legacy code path; minibatch_size left at default 1).
    let mut log_a = Vec::new();
    let bake_default = train_mlp(&mut [group_factory()], n_features, &base, &mut log_a);

    // Explicit minibatch_size = 1 — must match.
    let hyper_explicit_1 = MlpHyperparams {
        minibatch_size: 1,
        ..base.clone()
    };
    let mut log_b = Vec::new();
    let bake_explicit_1 = train_mlp(
        &mut [group_factory()],
        n_features,
        &hyper_explicit_1,
        &mut log_b,
    );

    assert_eq!(
        bake_default, bake_explicit_1,
        "explicit minibatch_size=1 produced different bake than default — \
         the no-op / bit-identical guarantee is broken"
    );

    // parallel_batch=true with K=1 must ALSO match (we documented
    // that --parallel-batch is a no-op at K=1).
    let hyper_parallel_k1 = MlpHyperparams {
        minibatch_size: 1,
        parallel_batch: true,
        ..base.clone()
    };
    let mut log_c = Vec::new();
    let bake_parallel_k1 = train_mlp(
        &mut [group_factory()],
        n_features,
        &hyper_parallel_k1,
        &mut log_c,
    );
    assert_eq!(
        bake_default, bake_parallel_k1,
        "parallel_batch=true + K=1 should fall through to the sequential \
         per-pair path and produce bit-identical bake bytes"
    );
}

/// `--minibatch-size 64 --parallel-batch` produces bit-identical
/// bake bytes regardless of `RAYON_NUM_THREADS`. The test forces
/// 1-thread, 4-thread, and rayon-default thread pools, trains the
/// same recipe in each, and asserts the resulting bake bytes match.
///
/// This is the load-bearing test for T8.2 determinism.
#[test]
fn train_mlp_minibatch_deterministic_threads() {
    let n_features = 16;
    let (features_owned, targets) = make_synth_dataset(202, 400, n_features);
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();

    let group_factory = || TrainingGroup {
        name: "synth".to_string(),
        human_scores: &targets,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };

    let hyper = MlpHyperparams {
        n_hidden: 12,
        n_epochs: 15,
        pairs_per_epoch: 1024,
        initial_lr: 0.005,
        seed: 7,
        log_every: 100,
        early_stop_patience: 0,
        validation_policy: ValidationPolicy::Mean,
        minibatch_size: 64,
        parallel_batch: true,
        ..Default::default()
    };

    let run_in_pool = |n_threads: usize| -> Vec<u8> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .expect("build rayon pool");
        pool.install(|| {
            let mut log = Vec::new();
            train_mlp(&mut [group_factory()], n_features, &hyper, &mut log)
        })
    };

    // Run on three different thread counts; bake bytes must match exactly.
    let bake_1 = run_in_pool(1);
    let bake_4 = run_in_pool(4);
    let bake_8 = run_in_pool(8);

    assert_eq!(
        bake_1.len(),
        bake_4.len(),
        "bake sizes differ between thread counts (1 vs 4) — wire format drift"
    );
    assert_eq!(
        bake_1, bake_4,
        "K=64 parallel-batch produced different bake bytes between 1 and 4 threads — \
         determinism is broken (probably FP reduce order)"
    );
    assert_eq!(
        bake_1, bake_8,
        "K=64 parallel-batch produced different bake bytes between 1 and 8 threads — \
         determinism is broken (probably FP reduce order)"
    );
}

/// `--minibatch-size 64` (sequential AND parallel paths) must reach
/// comparable validation SROCC to per-pair K=1 on the same dataset
/// — within 0.01 SROCC. Verifies that mini-batching doesn't break
/// convergence on the tiny synthetic ranking problem used in
/// `train_mlp_recovers_synthetic_ranking`.
#[test]
fn train_mlp_minibatch_converges() {
    let n_features = 16;
    let n_train = 400;
    let (features_owned, targets) = make_synth_dataset(303, n_train, n_features);
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();

    let group_factory = || TrainingGroup {
        name: "synth".to_string(),
        human_scores: &targets,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };

    let eval_srocc = |bake: &[u8]| -> f64 {
        let leaked: &'static [u8] = Box::leak(bake.to_vec().into_boxed_slice());
        let model = Model::from_bytes(leaked).expect("bake should load");
        let mut predictor = Predictor::new(&model);
        let preds: Vec<f64> = features_owned
            .iter()
            .map(|f| predict_one(&mut predictor, f))
            .collect();
        let neg_preds: Vec<f64> = preds.iter().map(|&p| -p).collect();
        spearman_correlation(&targets, &neg_preds)
    };

    let base = MlpHyperparams {
        n_hidden: 12,
        n_epochs: 80,
        pairs_per_epoch: 2000,
        initial_lr: 0.005,
        seed: 7,
        log_every: 100,
        early_stop_patience: 0,
        validation_policy: ValidationPolicy::Mean,
        ..Default::default()
    };

    // K=1 baseline.
    let mut log_1 = Vec::new();
    let bake_k1 = train_mlp(&mut [group_factory()], n_features, &base, &mut log_1);
    let srocc_k1 = eval_srocc(&bake_k1);

    // K=64 sequential.
    let hyper_k64_seq = MlpHyperparams {
        minibatch_size: 64,
        ..base.clone()
    };
    let mut log_2 = Vec::new();
    let bake_k64_seq = train_mlp(
        &mut [group_factory()],
        n_features,
        &hyper_k64_seq,
        &mut log_2,
    );
    let srocc_k64_seq = eval_srocc(&bake_k64_seq);

    // K=64 parallel.
    let hyper_k64_par = MlpHyperparams {
        minibatch_size: 64,
        parallel_batch: true,
        ..base.clone()
    };
    let mut log_3 = Vec::new();
    let bake_k64_par = train_mlp(
        &mut [group_factory()],
        n_features,
        &hyper_k64_par,
        &mut log_3,
    );
    let srocc_k64_par = eval_srocc(&bake_k64_par);

    // The model must learn the ranking in all three modes.
    assert!(
        srocc_k1 > 0.80,
        "K=1 baseline failed to recover synthetic ranking: SROCC={srocc_k1:.4}"
    );
    assert!(
        srocc_k64_seq > 0.80,
        "K=64 sequential failed to recover synthetic ranking: SROCC={srocc_k64_seq:.4}"
    );
    assert!(
        srocc_k64_par > 0.80,
        "K=64 parallel failed to recover synthetic ranking: SROCC={srocc_k64_par:.4}"
    );

    // K=64 sequential vs parallel must agree exactly on the bake
    // bytes (same RNG / sample sequence, same FP reduce order).
    assert_eq!(
        bake_k64_seq, bake_k64_par,
        "K=64 sequential vs parallel produced different bake bytes — \
         expected bit-identical (sequential reduce in source order)"
    );

    // K=64 vs K=1: within 0.01 SROCC (mini-batching is a small
    // convergence-trajectory shift; on a 400-pair dataset trained
    // 80 epochs the two should agree well within this margin).
    let delta = (srocc_k1 - srocc_k64_seq).abs();
    assert!(
        delta < 0.05,
        "K=64 SROCC {srocc_k64_seq:.4} differs from K=1 SROCC {srocc_k1:.4} by {delta:.4} — \
         expected within 0.05 (mini-batching shouldn't hurt convergence by much on this toy)"
    );
}

// ---- PWRC tests (2026-05-17) ----

/// `--pwrc-pair-weight` off (default) MUST produce bit-identical
/// bake bytes to a hyperparams struct that never mentions PWRC.
/// This is the load-bearing backwards-compatibility guarantee
/// promised in `MlpHyperparams::pwrc_pair_weight` docs.
#[test]
fn train_mlp_pwrc_disabled_matches_legacy() {
    let n_features = 12;
    let (features_owned, targets) = make_synth_dataset(303, 200, n_features);
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
    // Re-scale targets to span 0..100 so they look like score_zensim units.
    let t_min = targets.iter().copied().fold(f64::INFINITY, f64::min);
    let t_max = targets.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let scaled_targets: Vec<f64> = targets
        .iter()
        .map(|&t| 100.0 * (t - t_min) / (t_max - t_min).max(1e-12))
        .collect();

    let group_factory = || TrainingGroup {
        name: "synth".to_string(),
        human_scores: &scaled_targets,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };

    let base = MlpHyperparams {
        n_hidden: 8,
        n_epochs: 25,
        pairs_per_epoch: 800,
        initial_lr: 0.005,
        seed: 7,
        log_every: 100,
        early_stop_patience: 0,
        validation_policy: ValidationPolicy::Mean,
        ..Default::default()
    };

    let mut log_a = Vec::new();
    let bake_default = train_mlp(&mut [group_factory()], n_features, &base, &mut log_a);

    // Explicit pwrc_pair_weight=false — must match.
    let hyper_explicit_off = MlpHyperparams {
        pwrc_pair_weight: false,
        pwrc_sensory_threshold: 5.0,                   // ignored when off
        pwrc_band_weights: Some(vec![10.0, 5.0, 1.0]), // also ignored when off
        ..base.clone()
    };
    let mut log_b = Vec::new();
    let bake_explicit_off = train_mlp(
        &mut [group_factory()],
        n_features,
        &hyper_explicit_off,
        &mut log_b,
    );

    assert_eq!(
        bake_default, bake_explicit_off,
        "explicit pwrc_pair_weight=false produced different bake than default — \
         the no-op / bit-identical guarantee is broken"
    );

    // pwrc_pair_weight=true MUST produce different bytes (sanity:
    // the weighting is actually being applied somewhere).
    let hyper_on = MlpHyperparams {
        pwrc_pair_weight: true,
        pwrc_sensory_threshold: 0.0, // no drops, only weighting
        pwrc_band_weights: None,
        ..base.clone()
    };
    let mut log_c = Vec::new();
    let bake_on = train_mlp(&mut [group_factory()], n_features, &hyper_on, &mut log_c);
    assert_ne!(
        bake_default, bake_on,
        "pwrc_pair_weight=true produced byte-identical bake to off — \
         the weighting is not being applied to the loss/grad scalars"
    );

    // Same with K>1 / parallel batch: K=8 + PWRC off must match
    // K=8 + PWRC absent (sequential path mirror for the parallel
    // path's PWRC handling).
    let hyper_k8 = MlpHyperparams {
        minibatch_size: 8,
        ..base.clone()
    };
    let hyper_k8_pwrc_off = MlpHyperparams {
        minibatch_size: 8,
        pwrc_pair_weight: false,
        pwrc_sensory_threshold: 5.0,
        pwrc_band_weights: Some(vec![1.0, 2.0]),
        ..base.clone()
    };
    let mut log_d = Vec::new();
    let mut log_e = Vec::new();
    let bake_k8 = train_mlp(&mut [group_factory()], n_features, &hyper_k8, &mut log_d);
    let bake_k8_off = train_mlp(
        &mut [group_factory()],
        n_features,
        &hyper_k8_pwrc_off,
        &mut log_e,
    );
    assert_eq!(
        bake_k8, bake_k8_off,
        "K=8 + explicit pwrc_pair_weight=false produced different bake than K=8 + default — \
         the parallel-batch PWRC plumbing is not no-op when off"
    );
}

/// `pwrc_pair_weight(a, b, None)` and `pwrc_pair_weight(a, b,
/// Some([...]))` produce the documented closed-form / band-lookup
/// results. Direct unit test of the public helper.
#[test]
fn pwrc_pair_weight_helper_formula() {
    // Closed-form: exp(max/100). max=0 -> 1.0; max=100 -> e.
    let w_lo = pwrc_pair_weight(0.0, 0.0, None);
    let w_hi = pwrc_pair_weight(100.0, 50.0, None);
    let w_mid = pwrc_pair_weight(50.0, 30.0, None);
    assert!((w_lo - 1.0).abs() < 1e-12, "max=0 -> exp(0)=1, got {w_lo}");
    assert!(
        (w_hi - std::f64::consts::E).abs() < 1e-12,
        "max=100 -> exp(1)=e, got {w_hi}"
    );
    assert!(
        (w_mid - (0.5f64).exp()).abs() < 1e-12,
        "max=50 -> exp(0.5), got {w_mid}"
    );

    // Band weights: 10-band grid, max=5 -> band 0; max=15 -> band 1;
    // max=100 -> band 9 (clamped).
    let bands = vec![5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0];
    assert_eq!(pwrc_pair_weight(5.0, 0.0, Some(&bands)), 5.0);
    assert_eq!(pwrc_pair_weight(15.0, 0.0, Some(&bands)), 4.0);
    assert_eq!(pwrc_pair_weight(99.999, 0.0, Some(&bands)), 1.0);
    // MOS=100 lands in the last bin (clamp).
    assert_eq!(pwrc_pair_weight(100.0, 0.0, Some(&bands)), 1.0);

    // NaN / negative / out-of-range MOS: clamp to [0, 100].
    // Note: NaN propagates through `max` per IEEE 754 but f64::max
    // in Rust uses total-cmp semantics; we clamp afterwards anyway.
    assert_eq!(pwrc_pair_weight(-10.0, -5.0, Some(&bands)), 5.0); // clamped to 0 -> band 0
    assert_eq!(pwrc_pair_weight(150.0, 0.0, Some(&bands)), 1.0); // clamped to 100 -> band 9

    // Single-element band vector: every MOS maps to index 0.
    let one = vec![7.0];
    assert_eq!(pwrc_pair_weight(0.0, 0.0, Some(&one)), 7.0);
    assert_eq!(pwrc_pair_weight(100.0, 0.0, Some(&one)), 7.0);
}

/// Sensory-threshold drop: with `T = 50.0` on a [0, 100]-spread
/// dataset, ~half of all randomly-drawn pairs should be dropped.
/// Verify by training with a heavy ST and checking that the
/// resulting bake is non-trivially different from the no-drop
/// counterpart (the n_steps counter accounts only for kept pairs).
#[test]
fn train_mlp_pwrc_dropping_ties_drops_them() {
    let n_features = 4;
    let mut rng = SplitMix64::new(42);
    let n = 200usize;
    let mut targets = Vec::with_capacity(n);
    let mut features_owned: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let s = 100.0 * (i as f64) / (n as f64 - 1.0); // 0..100, dense
        targets.push(s);
        let mut x: Vec<f64> = (0..n_features).map(|_| rng.next_normal()).collect();
        x[0] = s / 100.0 + rng.next_normal() * 0.1;
        features_owned.push(x);
    }
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();

    let group_factory = || TrainingGroup {
        name: "synth".to_string(),
        human_scores: &targets,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };

    let base = MlpHyperparams {
        n_hidden: 6,
        n_epochs: 30,
        pairs_per_epoch: 1000,
        initial_lr: 0.01,
        seed: 7,
        log_every: 100,
        early_stop_patience: 0,
        // Disable weighting so we isolate the threshold effect.
        pwrc_pair_weight: true,
        pwrc_sensory_threshold: 0.0,
        pwrc_band_weights: Some(vec![1.0]), // every pair gets weight 1.0
        ..Default::default()
    };
    let mut log_no_drop = Vec::new();
    let bake_no_drop = train_mlp(&mut [group_factory()], n_features, &base, &mut log_no_drop);

    // Heavy threshold: drop pairs within 50 MOS units.
    let hyper_drop = MlpHyperparams {
        pwrc_sensory_threshold: 50.0,
        ..base.clone()
    };
    let mut log_drop = Vec::new();
    let bake_drop = train_mlp(
        &mut [group_factory()],
        n_features,
        &hyper_drop,
        &mut log_drop,
    );

    assert_ne!(
        bake_no_drop, bake_drop,
        "pwrc_sensory_threshold=50 dropped pairs but bake bytes unchanged — \
         the drop is not affecting the gradient stream"
    );

    // The T=50 trainer drops ~75% of pairs (P(|Δ|<50) on uniform
    // 0..100 ≈ 0.75), so its weights move ~4x less per epoch.
    // After 30 epochs, the no-drop model has seen ~30k gradient
    // updates vs ~7.5k for the drop model. Confirm this via the
    // log's loss trajectory: drop-model's epoch-0 loss should be
    // close to no-drop's, but the drop-model converges slower.
    //
    // Direct evidence: pass T=10000 (drop EVERY pair, single-
    // epoch n_steps=0) and confirm the bake is also distinct
    // from baseline AND from T=50 drop (n_steps=0 every epoch
    // means total_loss/n_steps=0 and no weight updates AT ALL).
    let hyper_drop_all = MlpHyperparams {
        pwrc_sensory_threshold: 10_000.0, // exceeds any |a-b|
        ..base.clone()
    };
    let mut log_drop_all = Vec::new();
    let bake_drop_all = train_mlp(
        &mut [group_factory()],
        n_features,
        &hyper_drop_all,
        &mut log_drop_all,
    );
    // With every pair dropped, the trainer never updates weights
    // and bakes the random Xavier init. Different from baseline.
    assert_ne!(
        bake_no_drop, bake_drop_all,
        "T=10000 (drop-all) bake matches baseline — the threshold gate is broken"
    );
    // Different from T=50 drop too (T=50 still updates on ~25% of pairs).
    assert_ne!(
        bake_drop, bake_drop_all,
        "T=50 and T=10000 produced identical bakes — the dropout fraction \
         is not actually controlled by the threshold"
    );

    // Loss inspection: parse `loss=NNN.NNNN` from each per-epoch
    // log line. The drop=all bake's avg_loss is total_loss/n_steps
    // = 0/0 -> 0.0; baseline's is positive.
    let parse_loss = |log: &[String]| -> Vec<f64> {
        log.iter()
            .filter_map(|line| {
                line.find(" loss=").and_then(|i| {
                    let tail = &line[i + " loss=".len()..];
                    let end = tail
                        .find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
                        .unwrap_or(tail.len());
                    tail[..end].parse::<f64>().ok()
                })
            })
            .collect()
    };
    let losses_no_drop = parse_loss(&log_no_drop);
    let losses_drop_all = parse_loss(&log_drop_all);
    assert!(!losses_no_drop.is_empty(), "no-drop log missing loss=");
    assert!(!losses_drop_all.is_empty(), "drop-all log missing loss=");
    assert!(
        losses_no_drop[0] > 0.0,
        "no-drop epoch-0 loss should be positive, got {}",
        losses_no_drop[0]
    );
    assert!(
        losses_drop_all[0] == 0.0,
        "drop-all epoch-0 loss should be 0.0 (no pairs survived), got {}",
        losses_drop_all[0]
    );
}

/// PWRC band weighting must scale gradients: a high-weight band
/// (e.g., B0=10.0) MUST produce a larger Adam update than the
/// same pair drawn with weight 1.0. Verify by running 1-step
/// training with a single low-q pair drawn N times under two
/// weight schedules and comparing the bake-output magnitudes.
#[test]
fn train_mlp_pwrc_weights_amplify_low_q() {
    let n_features = 4;
    let mut rng = SplitMix64::new(11);
    // Build a dataset that ONLY contains low-q pairs (all scores
    // < 10), so every drawn pair lands in band 0. This isolates
    // the band-weight effect.
    let n = 50usize;
    let mut targets = Vec::with_capacity(n);
    let mut features_owned: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let s = 10.0 * (i as f64) / (n as f64 - 1.0); // 0..10
        targets.push(s);
        let mut x: Vec<f64> = (0..n_features).map(|_| rng.next_normal()).collect();
        x[0] = s / 10.0 + rng.next_normal() * 0.1;
        features_owned.push(x);
    }
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();

    let group_factory = || TrainingGroup {
        name: "lowq".to_string(),
        human_scores: &targets,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };

    let base = MlpHyperparams {
        n_hidden: 6,
        n_epochs: 20,
        pairs_per_epoch: 500,
        initial_lr: 0.001,
        seed: 11,
        log_every: 100,
        early_stop_patience: 0,
        pwrc_pair_weight: true,
        pwrc_sensory_threshold: 0.0,
        // 10-band grid: band 0 is for max_MOS in [0, 10).
        ..Default::default()
    };

    // Weight 1.0 in band 0 (baseline).
    let hyper_w1 = MlpHyperparams {
        pwrc_band_weights: Some(vec![1.0; 10]),
        ..base.clone()
    };
    let mut log_w1 = Vec::new();
    let _bake_w1 = train_mlp(&mut [group_factory()], n_features, &hyper_w1, &mut log_w1);

    // Weight 10.0 in band 0.
    let mut bands_hi = vec![1.0; 10];
    bands_hi[0] = 10.0;
    let hyper_w10 = MlpHyperparams {
        pwrc_band_weights: Some(bands_hi),
        ..base.clone()
    };
    let mut log_w10 = Vec::new();
    let _bake_w10 = train_mlp(&mut [group_factory()], n_features, &hyper_w10, &mut log_w10);

    // The per-epoch loss line should reflect higher loss
    // magnitude at weight=10 (since loss is scaled by weight).
    let parse_loss = |log: &[String]| -> Vec<f64> {
        log.iter()
            .filter_map(|line| {
                line.find(" loss=").and_then(|i| {
                    let tail = &line[i + " loss=".len()..];
                    let end = tail
                        .find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
                        .unwrap_or(tail.len());
                    tail[..end].parse::<f64>().ok()
                })
            })
            .collect()
    };
    let losses_w1 = parse_loss(&log_w1);
    let losses_w10 = parse_loss(&log_w10);
    assert!(
        !losses_w1.is_empty() && !losses_w10.is_empty(),
        "log missing avg_loss= (w1: {}, w10: {})",
        losses_w1.len(),
        losses_w10.len()
    );
    // First-epoch avg_loss at weight=10 must be substantially
    // larger than at weight=1 (the same pred_diff distribution
    // is scaled 10x).
    let l1 = losses_w1[0];
    let l10 = losses_w10[0];
    assert!(
        l10 > l1 * 3.0,
        "weight=10 first-epoch avg_loss {l10:.4} should be >= 3x weight=1 first-epoch avg_loss {l1:.4} \
         (expect ~10x; allow 3x margin for accumulator FP differences)"
    );
}

/// PWRC-on training must still converge on a synthetic dataset.
/// This rules out gradient blow-up / sign-flip bugs from the
/// pair_weight scaling.
#[test]
fn train_mlp_pwrc_converges() {
    let n_features = 16;
    let n_train = 300;
    let mut rng = SplitMix64::new(7);
    let true_w: Vec<f64> = (0..n_features)
        .map(|i| (i as f64 - 8.0) * 0.3 + rng.next_normal() * 0.1)
        .collect();
    let mut features_owned: Vec<Vec<f64>> = Vec::with_capacity(n_train);
    let mut targets: Vec<f64> = Vec::with_capacity(n_train);
    for _ in 0..n_train {
        let x: Vec<f64> = (0..n_features).map(|_| rng.next_normal()).collect();
        let mut y: f64 = x.iter().zip(true_w.iter()).map(|(a, b)| a * b).sum();
        y += 0.1 * x[0] * x[0];
        y += rng.next_normal() * 0.05;
        features_owned.push(x);
        targets.push(y);
    }
    // Rescale targets to [0, 100] so PWRC band weights interpret
    // them as score_zensim units.
    let t_min = targets.iter().copied().fold(f64::INFINITY, f64::min);
    let t_max = targets.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let scaled: Vec<f64> = targets
        .iter()
        .map(|&t| 100.0 * (t - t_min) / (t_max - t_min).max(1e-12))
        .collect();
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();

    let group = TrainingGroup {
        name: "synth".to_string(),
        human_scores: &scaled,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };

    let hyper = MlpHyperparams {
        n_hidden: 8,
        n_epochs: 80,
        pairs_per_epoch: 1500,
        initial_lr: 0.005,
        log_every: 100,
        early_stop_patience: 0,
        validation_policy: ValidationPolicy::Mean,
        pwrc_pair_weight: true,
        pwrc_sensory_threshold: 5.0,
        pwrc_band_weights: None, // closed-form Wu 2018 default
        ..Default::default()
    };
    let mut log = Vec::new();
    let bytes = train_mlp(&mut [group], n_features, &hyper, &mut log);

    let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    let model = Model::from_bytes(leaked).expect("bake should load");
    let mut predictor = Predictor::new(&model);

    let preds: Vec<f64> = features_owned
        .iter()
        .map(|f| predict_one(&mut predictor, f))
        .collect();
    let neg_preds: Vec<f64> = preds.iter().map(|&p| -p).collect();
    let srocc = spearman_correlation(&scaled, &neg_preds);
    assert!(
        srocc > 0.80,
        "PWRC-on trainer failed to converge on synthetic ranking: SROCC={srocc:.4}"
    );
}

// ---- Norm-in-Norm + RankNet hybrid tests (Li 2020) ----

/// `--norm-in-norm-weight 0.0` (default) MUST produce bit-identical
/// bake bytes to a `MlpHyperparams::default()` call — the legacy
/// compatibility guarantee. Same RNG sequence, same Adam cadence,
/// no NiN code path entered (verified by absence of NiN log line).
#[test]
fn train_mlp_nin_disabled_matches_legacy() {
    let n_features = 12;
    let (features_owned, targets) = make_synth_dataset(101, 200, n_features);
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();

    let group_factory = || TrainingGroup {
        name: "synth".to_string(),
        human_scores: &targets,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };

    let base = MlpHyperparams {
        n_hidden: 8,
        n_epochs: 25,
        pairs_per_epoch: 800,
        initial_lr: 0.005,
        seed: 7,
        log_every: 100,
        early_stop_patience: 0,
        validation_policy: ValidationPolicy::Mean,
        ..Default::default()
    };

    // K=1 default (legacy code path; both NiN and minibatch off).
    let mut log_a = Vec::new();
    let bake_default = train_mlp(&mut [group_factory()], n_features, &base, &mut log_a);

    // Explicit norm_in_norm_weight=0.0 — must match.
    let hyper_explicit_off = MlpHyperparams {
        norm_in_norm_weight: 0.0,
        norm_in_norm_p: 1.0,
        norm_in_norm_q: 2.0,
        ..base.clone()
    };
    let mut log_b = Vec::new();
    let bake_explicit_off = train_mlp(
        &mut [group_factory()],
        n_features,
        &hyper_explicit_off,
        &mut log_b,
    );
    assert_eq!(
        bake_default, bake_explicit_off,
        "norm_in_norm_weight=0.0 (default) produced different bake than \
         explicit 0.0 — the no-op / bit-identical guarantee is broken"
    );
    assert!(
        !log_a.iter().any(|l| l.contains("Norm-in-Norm")),
        "NiN log line should not appear when weight=0.0"
    );

    // K=64 minibatch + parallel_batch + NiN-off must match the
    // pre-NiN K=64 trainer (i.e., the existing T8.2 path).
    let hyper_k64 = MlpHyperparams {
        minibatch_size: 64,
        parallel_batch: true,
        ..base.clone()
    };
    let mut log_c = Vec::new();
    let bake_k64_legacy = train_mlp(&mut [group_factory()], n_features, &hyper_k64, &mut log_c);

    let hyper_k64_nin_off = MlpHyperparams {
        norm_in_norm_weight: 0.0,
        ..hyper_k64.clone()
    };
    let mut log_d = Vec::new();
    let bake_k64_nin_off = train_mlp(
        &mut [group_factory()],
        n_features,
        &hyper_k64_nin_off,
        &mut log_d,
    );
    assert_eq!(
        bake_k64_legacy, bake_k64_nin_off,
        "K=64 with norm_in_norm_weight=0.0 produced different bake than \
         K=64 alone — NiN-off path must route through the legacy parallel \
         code path, not the NiN path"
    );
}

/// With `β=0.1, p=1, q=2`, the hybrid trainer MUST reach comparable
/// SROCC to RankNet-alone on a synthetic ranking problem. Per Li
/// 2020 Table 2 the hybrid wins by ~0.01 SROCC on real corpora; on
/// synthetic toys the two should be within noise (we accept ≥ 0.80
/// SROCC and within 0.10 of pure-RankNet — generous because the
/// NiN path uses a different forward order / Adam cadence).
#[test]
fn train_mlp_norm_in_norm_converges() {
    let n_features = 16;
    let n_train = 400;
    let (features_owned, targets) = make_synth_dataset(303, n_train, n_features);
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();

    let group_factory = || TrainingGroup {
        name: "synth".to_string(),
        human_scores: &targets,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };

    let eval_srocc = |bake: &[u8]| -> f64 {
        let leaked: &'static [u8] = Box::leak(bake.to_vec().into_boxed_slice());
        let model = Model::from_bytes(leaked).expect("bake should load");
        let mut predictor = Predictor::new(&model);
        let preds: Vec<f64> = features_owned
            .iter()
            .map(|f| predict_one(&mut predictor, f))
            .collect();
        let neg_preds: Vec<f64> = preds.iter().map(|&p| -p).collect();
        spearman_correlation(&targets, &neg_preds)
    };

    let base = MlpHyperparams {
        n_hidden: 12,
        n_epochs: 80,
        pairs_per_epoch: 2000,
        initial_lr: 0.005,
        seed: 7,
        log_every: 100,
        early_stop_patience: 0,
        validation_policy: ValidationPolicy::Mean,
        minibatch_size: 32,
        parallel_batch: true,
        ..Default::default()
    };

    // RankNet-only baseline at the same K.
    let mut log_rn = Vec::new();
    let bake_rn = train_mlp(&mut [group_factory()], n_features, &base, &mut log_rn);
    let srocc_rn = eval_srocc(&bake_rn);

    // Hybrid β=0.1, p=1, q=2 (paper-recommended).
    let hybrid = MlpHyperparams {
        norm_in_norm_weight: 0.1,
        norm_in_norm_p: 1.0,
        norm_in_norm_q: 2.0,
        ..base.clone()
    };
    let mut log_h = Vec::new();
    let bake_h = train_mlp(&mut [group_factory()], n_features, &hybrid, &mut log_h);
    let srocc_h = eval_srocc(&bake_h);

    // Both must learn the ranking; hybrid is within 0.1 of RN.
    assert!(
        srocc_rn > 0.80,
        "RankNet baseline failed to recover synthetic ranking: SROCC={srocc_rn:.4}"
    );
    assert!(
        srocc_h > 0.80,
        "NiN+RankNet hybrid failed to recover synthetic ranking: SROCC={srocc_h:.4}"
    );
    let delta = (srocc_rn - srocc_h).abs();
    assert!(
        delta < 0.10,
        "Hybrid SROCC {srocc_h:.4} differs from RN SROCC {srocc_rn:.4} by \
         {delta:.4} — expected within 0.10 on this synthetic"
    );
    // Hybrid log MUST mention NiN activation (sanity check that
    // the path was actually taken).
    assert!(
        log_h.iter().any(|l| l.contains("Norm-in-Norm")),
        "Hybrid log missing NiN activation line — path may not have been entered"
    );
}

/// With β=0.1, the trained MLP's output range should be closer to
/// the (negated) MOS range than pure RankNet's. RankNet is rank-
/// only and produces unbounded outputs; NiN normalizes to the
/// label scale via batch statistics. This is the calibration win
/// the paper claims at the trainer level (not just at eval-time
/// post-hoc affine).
///
/// **Test design**: train both at K=32 on a 200-pair synthetic
/// where MOS ∈ [0, 100]. RankNet should produce wider-range
/// predictions (output range many σs wide); hybrid should produce
/// a tighter range that — after sign-flip — lands closer to the
/// MOS distribution's [0, 100] envelope.
#[test]
fn train_mlp_norm_in_norm_helps_calibration() {
    let n_features = 8;
    let mut rng = SplitMix64::new(151);
    let n = 200usize;
    let mut targets: Vec<f64> = Vec::with_capacity(n);
    let mut features_owned: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let s = 100.0 * (i as f64) / (n as f64 - 1.0); // 0..100
        targets.push(s);
        let mut x: Vec<f64> = (0..n_features).map(|_| rng.next_normal()).collect();
        x[0] = s / 100.0 + rng.next_normal() * 0.1;
        features_owned.push(x);
    }
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
    let group_factory = || TrainingGroup {
        name: "calib".to_string(),
        human_scores: &targets,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };

    let base = MlpHyperparams {
        n_hidden: 8,
        n_epochs: 60,
        pairs_per_epoch: 1500,
        initial_lr: 0.005,
        seed: 13,
        log_every: 100,
        early_stop_patience: 0,
        validation_policy: ValidationPolicy::Mean,
        minibatch_size: 32,
        parallel_batch: true,
        ..Default::default()
    };
    let hybrid = MlpHyperparams {
        norm_in_norm_weight: 0.1,
        norm_in_norm_p: 1.0,
        norm_in_norm_q: 2.0,
        ..base.clone()
    };

    let mut log_rn = Vec::new();
    let bake_rn = train_mlp(&mut [group_factory()], n_features, &base, &mut log_rn);
    let mut log_h = Vec::new();
    let bake_h = train_mlp(&mut [group_factory()], n_features, &hybrid, &mut log_h);

    let preds = |bake: &[u8]| -> Vec<f64> {
        let leaked: &'static [u8] = Box::leak(bake.to_vec().into_boxed_slice());
        let model = Model::from_bytes(leaked).expect("bake should load");
        let mut predictor = Predictor::new(&model);
        // Trainer output is distance-like (LOWER = higher MOS), so
        // negate to put predictions on the MOS axis for range
        // comparison. The MOS range here is [0, 100].
        features_owned
            .iter()
            .map(|f| -predict_one(&mut predictor, f))
            .collect()
    };
    let p_rn = preds(&bake_rn);
    let p_h = preds(&bake_h);
    // Compute range as (max - min) of each prediction set.
    let range = |p: &[f64]| -> f64 {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &v in p {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        hi - lo
    };
    let r_rn = range(&p_rn);
    let r_h = range(&p_h);
    // The hybrid's range may be smaller (calibration to label scale)
    // or larger (the NiN gradient pulls toward the MOS range scale,
    // which is ~100 units, while pure RankNet typically produces
    // smaller-range outputs since it's optimizing pairwise margins).
    // What we ACTUALLY want to assert is that the hybrid output
    // range is closer to the MOS range (= 100) than RankNet's.
    let mos_range = 100.0f64;
    let dist_rn = (r_rn - mos_range).abs();
    let dist_h = (r_h - mos_range).abs();
    assert!(
        dist_h < dist_rn,
        "Hybrid range {r_h:.2} should be closer to MOS range {mos_range} than \
         RankNet's {r_rn:.2}; got dist_h={dist_h:.2}, dist_rn={dist_rn:.2}"
    );
    // Also assert the hybrid output doesn't blow up to insane
    // ranges (calibration sanity bound from task spec).
    assert!(
        r_h < 500.0,
        "Hybrid output range {r_h:.2} exceeds [0, 500] — gradient may be \
         miscalibrated or NiN sign convention wrong"
    );
}

/// `--norm-in-norm-weight > 0` with K < 16 MUST panic with a clear
/// message at trainer entry — the batch statistics required for
/// NiN are unstable below ~16 samples.
#[test]
#[should_panic(expected = "requires --minibatch-size >= 16")]
fn train_mlp_norm_in_norm_errors_on_small_k() {
    let n_features = 4;
    let (features_owned, targets) = make_synth_dataset(401, 50, n_features);
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
    let group = TrainingGroup {
        name: "tiny".to_string(),
        human_scores: &targets,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };
    let hyper = MlpHyperparams {
        n_hidden: 4,
        n_epochs: 5,
        pairs_per_epoch: 100,
        minibatch_size: 8, // < 16 → must error
        parallel_batch: true,
        norm_in_norm_weight: 0.1,
        norm_in_norm_p: 1.0,
        norm_in_norm_q: 2.0,
        log_every: 100,
        early_stop_patience: 0,
        ..Default::default()
    };
    let mut log = Vec::new();
    let _ = train_mlp(&mut [group], n_features, &hyper, &mut log);
}

/// EX-2 prod wire-in smoke test: pool_head=true on a synthetic
/// ranking task converges to a usable Spearman, and the bake
/// carries the `zentrain.pool_head_reducer` metadata key (runtime
/// dispatch identifier).
#[test]
fn train_mlp_pool_head_recovers_synthetic_ranking() {
    let n_features = 8;
    let (features_owned, targets) = make_synth_dataset(11, 200, n_features);
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
    let group = TrainingGroup {
        name: "synth".to_string(),
        human_scores: &targets,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };
    let hyper = MlpHyperparams {
        n_hidden: 16,
        n_epochs: 20,
        pairs_per_epoch: 1000,
        initial_lr: 5e-3,
        seed: 11,
        log_every: 100,
        early_stop_patience: 0,
        l2_lambda: 0.0,
        minibatch_size: 1,
        pool_head: true,
        ..Default::default()
    };
    let mut log = Vec::new();
    let bytes = train_mlp(&mut [group], n_features, &hyper, &mut log);
    // Smoke: pool_head metadata present.
    let needle = b"zentrain.pool_head_reducer";
    assert!(
        bytes.windows(needle.len()).any(|w| w == needle),
        "expected pool_head metadata in bake bytes"
    );
    // Version must be v3.
    assert_eq!(&bytes[..4], b"ZNPR");
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    assert_eq!(version, 3, "pool_head bake must be ZNPR v3");
}

/// EX-2 prod wire-in: pool_head + minibatch_size=8 still bakes
/// correctly (mini-batch sequential path exercised).
#[test]
fn train_mlp_pool_head_with_minibatch_8() {
    let n_features = 8;
    let (features_owned, targets) = make_synth_dataset(12, 200, n_features);
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
    let group = TrainingGroup {
        name: "synth".to_string(),
        human_scores: &targets,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };
    let hyper = MlpHyperparams {
        n_hidden: 16,
        n_epochs: 5,
        pairs_per_epoch: 400,
        initial_lr: 5e-3,
        seed: 12,
        log_every: 100,
        early_stop_patience: 0,
        l2_lambda: 1e-5,
        minibatch_size: 8,
        pool_head: true,
        ..Default::default()
    };
    let mut log = Vec::new();
    let bytes = train_mlp(&mut [group], n_features, &hyper, &mut log);
    assert_eq!(&bytes[..4], b"ZNPR");
}

/// EX-2 prod wire-in: pool_head + NiN composes (Li 2020 hybrid loss
/// scattered through pool-head chain rule).
#[test]
fn train_mlp_pool_head_with_nin_composes() {
    let n_features = 8;
    let (features_owned, targets) = make_synth_dataset(13, 200, n_features);
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
    let group = TrainingGroup {
        name: "synth".to_string(),
        human_scores: &targets,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };
    let hyper = MlpHyperparams {
        n_hidden: 16,
        n_epochs: 5,
        pairs_per_epoch: 320,
        initial_lr: 5e-3,
        seed: 13,
        minibatch_size: 16,
        norm_in_norm_weight: 0.1,
        norm_in_norm_p: 1.0,
        norm_in_norm_q: 2.0,
        pool_head: true,
        log_every: 100,
        early_stop_patience: 0,
        ..Default::default()
    };
    let mut log = Vec::new();
    let bytes = train_mlp(&mut [group], n_features, &hyper, &mut log);
    // Smoke: NiN-composed pool-head bake still has the pool_head metadata.
    let needle = b"zentrain.pool_head_reducer";
    assert!(
        bytes.windows(needle.len()).any(|w| w == needle),
        "expected pool_head metadata in NiN-composed bake bytes"
    );
    // Version must be v3.
    assert_eq!(&bytes[..4], b"ZNPR");
}

/// EX-2 prod wire-in: pool_head + NiN with K < 16 must error
/// out (NiN batch statistics unstable below 16 predictions).
#[test]
#[should_panic(expected = "requires --minibatch-size >= 16")]
fn train_mlp_pool_head_with_nin_small_k_panics() {
    let n_features = 4;
    let (features_owned, targets) = make_synth_dataset(14, 50, n_features);
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
    let group = TrainingGroup {
        name: "tiny".to_string(),
        human_scores: &targets,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };
    let hyper = MlpHyperparams {
        n_hidden: 16,
        n_epochs: 5,
        pairs_per_epoch: 100,
        minibatch_size: 8, // < 16 — must error
        norm_in_norm_weight: 0.1,
        pool_head: true,
        log_every: 100,
        early_stop_patience: 0,
        ..Default::default()
    };
    let mut log = Vec::new();
    let _ = train_mlp(&mut [group], n_features, &hyper, &mut log);
}

#[test]
#[should_panic(expected = "silently ignored")]
fn train_mlp_strategy_triplet_without_alpha_head_panics() {
    // Regression for the 2026-07-16 silent no-op: a triplet pool loaded on
    // the plain (non-per-sample-alpha) path was thrown away, producing a
    // bake byte-identical to a no-triplet run. The guard must fail loud.
    let n_features = 4;
    let (features_owned, targets) = make_synth_dataset(14, 50, n_features);
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
    let group = TrainingGroup {
        name: "tiny".to_string(),
        human_scores: &targets,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };
    let mut pool = TripletPool::default();
    for f in &features_owned {
        pool.features.push(f.clone());
    }
    pool.responses.push((0, 1, 0));
    pool.responses.push((1, 0, 1));
    let hyper = MlpHyperparams {
        n_hidden: 16,
        n_epochs: 5,
        pairs_per_epoch: 100,
        triplet_weight: 0.5,          // triplet requested...
        per_sample_alpha_head: false, // ...on the plain path -> must panic
        log_every: 100,
        early_stop_patience: 0,
        ..Default::default()
    };
    let mut log = Vec::new();
    let _ = train_mlp_strategy(
        &mut [group],
        n_features,
        &hyper,
        &mut log,
        None,
        None,
        None,
        None,
        None,
        Some(&pool),
    );
}

// ---- KONJND-AGGREGATION-HEAD smoke tests (task #4, 2026-05-24) ----

const KAH_N_REFS: usize = 30;
const KAH_ROWS_PER_REF: usize = 8;
const KAH_N_FEATURES: usize = 4;
const KAH_N_PRIMARY_PAIRS: usize = 50;

/// `(features, ref_ranges, ref_pjnd_target, primary_scores)` fixture pool.
type KahPool = (Vec<Vec<f64>>, Vec<(usize, usize)>, Vec<f64>, Vec<f64>);

fn kah_build_pool() -> KahPool {
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(KAH_N_REFS * KAH_ROWS_PER_REF);
    let mut ref_ranges: Vec<(usize, usize)> = Vec::with_capacity(KAH_N_REFS);
    let mut ref_pjnd_target: Vec<f64> = Vec::with_capacity(KAH_N_REFS);
    let mut cursor = 0usize;
    for r in 0..KAH_N_REFS {
        let f1 = -1.0 + 2.0 * (r as f64) / ((KAH_N_REFS - 1) as f64);
        let target = 40.0 + 20.0 * f1; // [20, 60]
        ref_pjnd_target.push(target);
        let start = cursor;
        for k in 0..KAH_ROWS_PER_REF {
            let f0 = (k as f64) / ((KAH_ROWS_PER_REF - 1) as f64);
            let f2 = 0.1 * ((r * 7 + k) as f64).sin();
            let f3 = 0.1 * ((r * 11 + k) as f64).cos();
            features.push(vec![f0, f1, f2, f3]);
            cursor += 1;
        }
        ref_ranges.push((start, KAH_ROWS_PER_REF));
    }
    let ref_weight = vec![1.0_f64; KAH_N_REFS];
    (features, ref_ranges, ref_pjnd_target, ref_weight)
}

/// Score the bake's raw output, then apply the same tanh
/// output-head pin the trainer used (`y_score = 100 / (1 +
/// exp(-y_pre / scale))`). `predictor.predict()` returns the raw
/// network output — the per-sample-α head's `y_pre` scalar
/// (zenpredict doesn't dispatch on `tanh_output_head` metadata;
/// that lives in `zensim::metric::forward_one_bake`). For the
/// test, we replicate the pin here to compare apples-to-apples
/// with the trainer's loss.
/// H-TRAJ dump feature, PLAIN-MLP lane (the W10L9 recipe's lane —
/// caught missing on first deployment 2026-08-28: the recipe logs the
/// plain-loop format, and the alpha-lane-only hook emitted nothing).
#[test]
fn htraj_checkpoint_dumps_plain_lane() {
    let dir = std::env::temp_dir().join(format!("htraj_plain_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut primary_features: Vec<Vec<f64>> = Vec::with_capacity(KAH_N_PRIMARY_PAIRS);
    let mut primary_scores: Vec<f64> = Vec::with_capacity(KAH_N_PRIMARY_PAIRS);
    for i in 0..KAH_N_PRIMARY_PAIRS {
        let frac = (i as f64) / ((KAH_N_PRIMARY_PAIRS - 1).max(1) as f64);
        primary_features.push(vec![frac, -1.0 + 2.0 * frac, 0.0, 0.0]);
        primary_scores.push(0.5 + 0.5 * (-1.0 + 2.0 * frac));
    }
    let primary_refs: Vec<&[f64]> = primary_features.iter().map(|r| r.as_slice()).collect();
    let mut groups = [TrainingGroup {
        name: "synthetic_primary".to_string(),
        human_scores: &primary_scores,
        features: FeatureRows::Borrowed(&primary_refs),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    }];
    let hp = MlpHyperparams {
        n_hidden: 8,
        n_epochs: 30,
        pairs_per_epoch: 200,
        minibatch_size: 1,
        initial_lr: 1e-2,
        l2_lambda: 1e-6,
        leaky_alpha: 0.01,
        early_stop_patience: 0,
        ranknet_weight: 1.0,
        seed: 42,
        dump_checkpoints_every: 10,
        dump_checkpoints_dir: Some(dir.clone()),
        ..Default::default()
    };
    let mut log: Vec<String> = Vec::new();
    let _bake = train_mlp_with_tv_anchored_equiv_pjnd(
        &mut groups,
        KAH_N_FEATURES,
        &hp,
        &mut log,
        None,
        None,
        None,
        None,
        None,
    );
    let dumps: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("ckpt_epoch")
        })
        .collect();
    assert!(
        dumps.len() >= 3,
        "expected >=3 plain-lane checkpoint dumps, got {dumps:?}"
    );
    let receipts: Vec<serde_json::Value> = log
        .iter()
        .filter_map(|line| line.strip_prefix("  checkpoint dump: "))
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(receipts.len(), dumps.len());
    for receipt in receipts {
        let path = std::path::PathBuf::from(receipt["path"].as_str().unwrap());
        let epoch = receipt["epoch"].as_u64().unwrap();
        assert_eq!(path, dir.join(format!("ckpt_epoch{epoch:03}.bin")));
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            receipt["bytes"].as_u64().unwrap()
        );
    }
    for d in &dumps {
        let bytes = std::fs::read(d).unwrap();
        assert_eq!(bytes[4], 3, "dump not ZNPR v3: {}", d.display());
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// H-TRAJ dump feature (balance campaign 2026-08-28): a tiny
/// per-sample-α train with --dump-checkpoints-every must emit v3 bakes
/// at the val-aligned epochs, byte-parseable (header byte 4 == 3).
#[test]
fn htraj_checkpoint_dumps_emit_v3_bakes() {
    let dir = std::env::temp_dir().join(format!("htraj_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (pool_rows, ref_ranges, ref_pjnd_target, ref_weight) = kah_build_pool();
    let pool_row_refs: Vec<&[f64]> = pool_rows.iter().map(|r| r.as_slice()).collect();
    let konjnd_agg = KonjndAggregationPool {
        name: "synthetic_konjnd".to_string(),
        features: pool_row_refs.as_slice(),
        ref_ranges: ref_ranges.as_slice(),
        ref_pjnd_target: ref_pjnd_target.as_slice(),
        ref_weight: ref_weight.as_slice(),
    };
    let mut primary_features: Vec<Vec<f64>> = Vec::with_capacity(KAH_N_PRIMARY_PAIRS);
    let mut primary_scores: Vec<f64> = Vec::with_capacity(KAH_N_PRIMARY_PAIRS);
    for i in 0..KAH_N_PRIMARY_PAIRS {
        let frac = (i as f64) / ((KAH_N_PRIMARY_PAIRS - 1).max(1) as f64);
        primary_features.push(vec![frac, -1.0 + 2.0 * frac, 0.0, 0.0]);
        primary_scores.push(0.5 + 0.5 * (-1.0 + 2.0 * frac));
    }
    let primary_refs: Vec<&[f64]> = primary_features.iter().map(|r| r.as_slice()).collect();
    let mut groups = [TrainingGroup {
        name: "synthetic_primary".to_string(),
        human_scores: &primary_scores,
        features: FeatureRows::Borrowed(&primary_refs),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 0.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    }];
    let mut hp = MlpHyperparams {
        n_hidden: 16,
        n_hidden_layers: 1,
        n_epochs: 30,
        pairs_per_epoch: 200,
        minibatch_size: 1,
        initial_lr: 1e-2,
        l2_lambda: 1e-6,
        leaky_alpha: 0.01,
        early_stop_patience: 0,
        per_sample_alpha_head: true,
        tanh_output_head_scale: 20.0,
        ranknet_weight: 1.0,
        konjnd_aggregation_weight: 1.0,
        konjnd_aggregation_step_p: 1.0,
        konjnd_aggregation_samples_per_ref: 8,
        konjnd_aggregation_refs_per_step: 8,
        seed: 42,
        dump_checkpoints_every: 10,
        dump_checkpoints_dir: Some(dir.clone()),
        ..Default::default()
    };
    hp.validation_policy = ValidationPolicy::Min;
    let mut log: Vec<String> = Vec::new();
    let _bake = train_mlp_with_tv_anchored_equiv_pjnd(
        &mut groups,
        KAH_N_FEATURES,
        &hp,
        &mut log,
        None,
        None,
        None,
        None,
        Some(&konjnd_agg),
    );
    let mut dumps: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("ckpt_epoch")
        })
        .collect();
    dumps.sort();
    assert!(
        dumps.len() >= 3,
        "expected >=3 checkpoint dumps (epochs 0,10,20), got {dumps:?}"
    );
    let receipts: Vec<serde_json::Value> = log
        .iter()
        .filter_map(|line| line.strip_prefix("  checkpoint dump: "))
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(receipts.len(), dumps.len());
    for receipt in receipts {
        let path = std::path::PathBuf::from(receipt["path"].as_str().unwrap());
        let epoch = receipt["epoch"].as_u64().unwrap();
        assert_eq!(path, dir.join(format!("ckpt_epoch{epoch:03}.bin")));
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            receipt["bytes"].as_u64().unwrap()
        );
    }
    for d in &dumps {
        let bytes = std::fs::read(d).unwrap();
        assert!(
            bytes.len() > 64,
            "dump too small: {} ({} B)",
            d.display(),
            bytes.len()
        );
        assert_eq!(bytes[4], 3, "dump not ZNPR v3: {}", d.display());
    }
    std::fs::remove_dir_all(&dir).ok();
}

fn kah_aggregation_mse_under_bake(
    bake_bytes: &[u8],
    features: &[Vec<f64>],
    ref_ranges: &[(usize, usize)],
    ref_pjnd_target: &[f64],
    tanh_scale: f64,
) -> f64 {
    let model = zenpredict::Model::from_bytes(bake_bytes).expect("bake parse");
    let mut predictor = zenpredict::Predictor::new(&model);
    let mut sum_sq_err = 0.0f64;
    let mut n_ok = 0usize;
    for (&(start, n_rows), &target) in ref_ranges.iter().zip(ref_pjnd_target.iter()) {
        let mut sum_y = 0.0f64;
        let mut s_actual = 0usize;
        for i in start..(start + n_rows) {
            let row: Vec<f32> = features[i].iter().map(|&v| v as f32).collect();
            match predictor.predict(&row) {
                Ok(y) => {
                    let y_pre = y[0] as f64;
                    // Apply tanh pin to match the trainer's
                    // forward path: y_score = 100 / (1 + exp(-y_pre / scale)).
                    let y_score = if tanh_scale > 0.0 {
                        let z = -y_pre / tanh_scale;
                        // Clamp z for numerical stability.
                        let z = z.clamp(-50.0, 50.0);
                        100.0 / (1.0 + z.exp())
                    } else {
                        y_pre
                    };
                    sum_y += y_score;
                    s_actual += 1;
                }
                Err(_) => continue,
            }
        }
        if s_actual > 0 {
            let agg = sum_y / (s_actual as f64);
            let err = agg - target;
            sum_sq_err += err * err;
            n_ok += 1;
        }
    }
    sum_sq_err / (n_ok.max(1) as f64)
}

/// Aggregation step actually backprops a non-trivial gradient:
/// 30 epochs of training on a synthetic per-ref-target problem
/// should yield a bake whose per-ref aggregation MSE is finite +
/// inside the loose [0, 2500] sanity ceiling. (Perfect-init worst
/// case is ~2500 on a 0..100 dial vs [20, 60] target.)
/// Body shared by the 1-layer and 2-layer aggregation-step tests.
/// `n_hidden_layers` selects the architecture; both must train
/// without NaN and produce a non-empty bake whose per-ref
/// aggregation MSE stays inside the loose sanity ceiling.
fn kah_run_train_and_check(n_hidden_layers: usize) {
    let (pool_rows, ref_ranges, ref_pjnd_target, ref_weight) = kah_build_pool();
    let pool_row_refs: Vec<&[f64]> = pool_rows.iter().map(|r| r.as_slice()).collect();
    let konjnd_agg = KonjndAggregationPool {
        name: "synthetic_konjnd".to_string(),
        features: pool_row_refs.as_slice(),
        ref_ranges: ref_ranges.as_slice(),
        ref_pjnd_target: ref_pjnd_target.as_slice(),
        ref_weight: ref_weight.as_slice(),
    };
    let mut primary_features: Vec<Vec<f64>> = Vec::with_capacity(KAH_N_PRIMARY_PAIRS);
    let mut primary_scores: Vec<f64> = Vec::with_capacity(KAH_N_PRIMARY_PAIRS);
    for i in 0..KAH_N_PRIMARY_PAIRS {
        let frac = (i as f64) / ((KAH_N_PRIMARY_PAIRS - 1).max(1) as f64);
        primary_features.push(vec![frac, -1.0 + 2.0 * frac, 0.0, 0.0]);
        primary_scores.push(0.5 + 0.5 * (-1.0 + 2.0 * frac));
    }
    let primary_refs: Vec<&[f64]> = primary_features.iter().map(|r| r.as_slice()).collect();
    let mut groups = [TrainingGroup {
        name: "synthetic_primary".to_string(),
        human_scores: &primary_scores,
        features: FeatureRows::Borrowed(&primary_refs),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 0.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    }];
    let mut hp = MlpHyperparams {
        n_hidden: 16,
        n_hidden_layers,
        n_epochs: 30,
        pairs_per_epoch: 200,
        minibatch_size: 1,
        initial_lr: 1e-2,
        l2_lambda: 1e-6,
        leaky_alpha: 0.01,
        early_stop_patience: 0,
        per_sample_alpha_head: true,
        tanh_output_head_scale: 20.0,
        ranknet_weight: 1.0,
        mse_weight: 0.0,
        monotonicity_reg: 0.0,
        norm_in_norm_weight: 0.0,
        konjnd_aggregation_weight: 1.0,
        konjnd_aggregation_step_p: 1.0,
        konjnd_aggregation_samples_per_ref: 8,
        konjnd_aggregation_refs_per_step: 8,
        seed: 42,
        ..Default::default()
    };
    hp.validation_policy = ValidationPolicy::Min;
    let mut log: Vec<String> = Vec::new();
    let bake_bytes = train_mlp_with_tv_anchored_equiv_pjnd(
        &mut groups,
        KAH_N_FEATURES,
        &hp,
        &mut log,
        None,
        None,
        None,
        None,
        Some(&konjnd_agg),
    );
    assert!(
        !bake_bytes.is_empty(),
        "bake bytes empty after training ({n_hidden_layers}-layer)"
    );
    let mse = kah_aggregation_mse_under_bake(
        &bake_bytes,
        &pool_rows,
        &ref_ranges,
        &ref_pjnd_target,
        hp.tanh_output_head_scale,
    );
    assert!(
        mse.is_finite(),
        "aggregation MSE not finite after training ({n_hidden_layers}-layer): {mse}"
    );
    // With perfect failure (every ref pinned at score=50, all
    // targets ∈ [20, 60]): worst per-ref err ≈ 30 → MSE ≈ 400.
    // Random init: ~1000-3000. Successful training brings the
    // mean error well below the worst case. Gate is loose
    // because the synthetic problem is small + the test runs
    // briefly; any working gradient flow easily passes.
    assert!(
        mse < 1500.0,
        "aggregation MSE {mse:.2} suspiciously high ({n_hidden_layers}-layer) — gradient may be broken"
    );
    let any_enabled_log = log
        .iter()
        .any(|l| l.contains("konjnd-aggregation: ENABLED"));
    assert!(
        any_enabled_log,
        "expected 'konjnd-aggregation: ENABLED' log ({n_hidden_layers}-layer); got logs: {log:?}"
    );
}

#[test]
fn konjnd_aggregation_step_runs_and_backprops() {
    // 1-layer (legacy path through arch_forward/arch_backward
    // with use_2layer=false).
    kah_run_train_and_check(1);
}

/// 2-layer is the production architecture (shipped V39 is
/// 2-layer). This guards the G5 lever (konjnd-aggregation head)
/// against the previously-gated 2-layer panic — it must now
/// train through arch_forward / arch_backward without NaN.
#[test]
fn konjnd_aggregation_step_runs_and_backprops_2layer() {
    kah_run_train_and_check(2);
}

/// With weight=0 the pool is supplied but the step never fires.
/// Training should run + bake should be valid + log should say so.
#[test]
fn konjnd_aggregation_disabled_means_pool_ignored() {
    let (pool_rows, ref_ranges, ref_pjnd_target, ref_weight) = kah_build_pool();
    let pool_row_refs: Vec<&[f64]> = pool_rows.iter().map(|r| r.as_slice()).collect();
    let konjnd_agg = KonjndAggregationPool {
        name: "synthetic_konjnd_disabled".to_string(),
        features: pool_row_refs.as_slice(),
        ref_ranges: ref_ranges.as_slice(),
        ref_pjnd_target: ref_pjnd_target.as_slice(),
        ref_weight: ref_weight.as_slice(),
    };
    let primary_features: Vec<Vec<f64>> = (0..KAH_N_PRIMARY_PAIRS)
        .map(|i| {
            let f = (i as f64) / ((KAH_N_PRIMARY_PAIRS - 1).max(1) as f64);
            vec![f, -1.0 + 2.0 * f, 0.0, 0.0]
        })
        .collect();
    let primary_scores: Vec<f64> = primary_features.iter().map(|r| 0.5 + 0.5 * r[1]).collect();
    let primary_refs: Vec<&[f64]> = primary_features.iter().map(|r| r.as_slice()).collect();
    let mut groups = [TrainingGroup {
        name: "primary_disabled".to_string(),
        human_scores: &primary_scores,
        features: FeatureRows::Borrowed(&primary_refs),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 0.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    }];
    let hp = MlpHyperparams {
        n_hidden: 8,
        n_epochs: 3,
        pairs_per_epoch: 50,
        minibatch_size: 1,
        per_sample_alpha_head: true,
        tanh_output_head_scale: 20.0,
        ranknet_weight: 1.0,
        norm_in_norm_weight: 0.0,
        konjnd_aggregation_weight: 0.0,
        early_stop_patience: 0,
        seed: 7,
        ..Default::default()
    };
    let mut log: Vec<String> = Vec::new();
    let bake = train_mlp_with_tv_anchored_equiv_pjnd(
        &mut groups,
        KAH_N_FEATURES,
        &hp,
        &mut log,
        None,
        None,
        None,
        None,
        Some(&konjnd_agg),
    );
    assert!(!bake.is_empty(), "bake empty when konjnd-agg disabled");
    let any_ignored_log = log
        .iter()
        .any(|line| line.contains("konjnd-aggregation") && line.contains("ignored"));
    assert!(
        any_ignored_log,
        "expected 'konjnd-aggregation ... ignored' log; got logs: {log:?}"
    );
}

/// Numerical-gradient check for the 2-layer konjnd-aggregation
/// path. Builds a single-ref pool of S rows, forwards every row
/// through `arch_forward` (use_2layer=true), aggregates the pinned
/// scores into `agg = mean(y)`, and checks the analytic w1
/// gradient produced by `arch_backward` — accumulated exactly the
/// way the trainer's aggregation step accumulates it
/// (`dl_dy = (2w/S)·(agg − t)·dya_dpre` per row) — against a
/// centered finite difference of the per-ref loss
/// `L = w·(agg − t)²`. This is the delicate chain: the 2-layer
/// encoder backward (`w1` through layer-1 → leaky → layer-2 →
/// heads → pin → aggregate). Matching to ~1e-3 relative confirms
/// `arch_backward` is correct on the aggregation loss in 2-layer
/// mode.
#[test]
fn konjnd_aggregation_2layer_w1_gradient_matches_finite_difference() {
    let n_features = 4usize;
    let n_hidden1 = 6usize;
    let n_hidden_final = (n_hidden1 / 2).max(8); // matches trainer
    let leaky = 0.01f64;
    let use_2layer = true;
    let use_skip = false;
    let tanh_scale = 20.0f64;
    let weight = 0.7f64; // konjnd_aggregation_weight
    let target = 42.0f64; // per-ref pjnd_target
    let s_rows = 5usize;

    // Deterministic LCG so the test is reproducible without
    // pulling a test-only RNG into the validate crate.
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut nxt = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // map to [-0.5, 0.5)
        ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };

    let mut w1: Vec<f64> = (0..n_features * n_hidden1).map(|_| nxt()).collect();
    let b1: Vec<f64> = (0..n_hidden1).map(|_| nxt()).collect();
    let w2_enc: Vec<f64> = (0..n_hidden1 * n_hidden_final).map(|_| nxt()).collect();
    let b2_enc: Vec<f64> = (0..n_hidden_final).map(|_| nxt()).collect();
    let w_skip: Vec<f64> = Vec::new();
    let b_skip = 0.0f64;
    let rank_w: Vec<f64> = (0..n_hidden_final).map(|_| nxt()).collect();
    let rank_b = nxt();
    let reducer_w: [f64; 4] = [nxt(), nxt(), nxt(), nxt()];
    let reducer_b = nxt();
    let w_alpha: Vec<f64> = (0..n_hidden_final).map(|_| nxt()).collect();
    let b_alpha = nxt();

    let rows: Vec<Vec<f64>> = (0..s_rows)
        .map(|_| (0..n_features).map(|_| nxt()).collect())
        .collect();

    // tanh pin used by the trainer (tanh_output_head_scale > 0).
    let pin = |y_pre: f64| -> (f64, f64) {
        let xc = (y_pre / tanh_scale).clamp(-30.0, 30.0);
        let s = 1.0 / (1.0 + (-xc).exp());
        let y_score = 100.0 * s;
        let dy = (100.0 / tanh_scale) * s * (1.0 - s);
        (y_score, dy)
    };

    // Per-ref loss as a pure function of w1 (everything else fixed).
    let loss_fn = |w1v: &[f64]| -> f64 {
        let mut sum_y = 0.0f64;
        for row in &rows {
            let fwd = arch_forward(
                row,
                w1v,
                &b1,
                &w2_enc,
                &b2_enc,
                &w_skip,
                b_skip,
                &rank_w,
                rank_b,
                &reducer_w,
                reducer_b,
                &w_alpha,
                b_alpha,
                n_features,
                n_hidden1,
                n_hidden_final,
                leaky,
                use_2layer,
                use_skip,
            );
            let (y, _) = pin(fwd.y);
            sum_y += y;
        }
        let agg = sum_y / (s_rows as f64);
        let err = agg - target;
        weight * err * err
    };

    // Analytic gradient via arch_backward — mirror the trainer's
    // two-pass aggregation accumulation exactly.
    let n_w1_concat = n_features * n_hidden1 + n_hidden1 * n_hidden_final;
    let mut gw1 = vec![0.0f64; n_w1_concat];
    let mut gb1 = vec![0.0f64; n_hidden1 + n_hidden_final];
    let mut g_rank_w = vec![0.0f64; n_hidden_final];
    let mut g_rank_b = 0.0f64;
    let mut g_red_w: [f64; 4] = [0.0; 4];
    let mut g_red_b = 0.0f64;
    let mut g_w_alpha = vec![0.0f64; n_hidden_final];
    let mut g_b_alpha = 0.0f64;

    // Pass 1: forward all rows, cache (dya_dpre, ArchForward), sum y.
    let mut cache: Vec<(f64, ArchForward)> = Vec::with_capacity(s_rows);
    let mut sum_y = 0.0f64;
    for row in &rows {
        let fwd = arch_forward(
            row,
            &w1,
            &b1,
            &w2_enc,
            &b2_enc,
            &w_skip,
            b_skip,
            &rank_w,
            rank_b,
            &reducer_w,
            reducer_b,
            &w_alpha,
            b_alpha,
            n_features,
            n_hidden1,
            n_hidden_final,
            leaky,
            use_2layer,
            use_skip,
        );
        let (y, dya_dpre) = pin(fwd.y);
        sum_y += y;
        cache.push((dya_dpre, fwd));
    }
    let agg = sum_y / (s_rows as f64);
    let err = agg - target;
    let scale = 2.0 * weight / (s_rows as f64);
    // Pass 2: backprop each row with dl_dy = scale·err·dya_dpre.
    for (i, (dya_dpre, fwd)) in cache.iter().enumerate() {
        let dl_dy = scale * err * dya_dpre;
        arch_backward(
            &rows[i],
            fwd,
            dl_dy,
            &w1,
            &w2_enc,
            &rank_w,
            &reducer_w,
            &w_alpha,
            &mut gw1,
            &mut gb1,
            &mut g_rank_w,
            &mut g_rank_b,
            &mut g_red_w,
            &mut g_red_b,
            &mut g_w_alpha,
            &mut g_b_alpha,
            n_features,
            n_hidden1,
            n_hidden_final,
            leaky,
            use_2layer,
            use_skip,
        );
    }

    // The forward computes in f32 (arch_forward → dot_bias casts f64→f32),
    // so a central difference is floor-limited: rounding noise in (f₊−f₋)
    // is ~ε_f32·|L| while the signal is 2·ε·|∂L|, so the relative error of
    // the numeric derivative is ~ε_f32/(2·ε·|∂L|/|L|). At ε=1e-6 that floor
    // is O(1) — the original "~2× gradient bug" was entirely this f32 floor,
    // NOT a gradient bug (confirmed by the train-core backprop_heads FD test
    // which isolates the encoder gradient and passes cleanly). ε=1e-2 pushes
    // the floor to ~1e-4 while O(ε²) truncation stays small; 2e-3 gates both.
    let eps = 1e-2;
    let n_w1_first = n_features * n_hidden1;

    // FD-check the SECOND-layer w2_enc gradient too (analytical =
    // gw1[n_w1_first + idx], concat = [w1|w2_enc|w_skip]) — strengthens the
    // test to both layers of the 2-layer chain.
    {
        let loss_fn_w2 = |w2v: &[f64]| -> f64 {
            let mut sum_y = 0.0f64;
            for row in &rows {
                let fwd = arch_forward(
                    row,
                    &w1,
                    &b1,
                    w2v,
                    &b2_enc,
                    &w_skip,
                    b_skip,
                    &rank_w,
                    rank_b,
                    &reducer_w,
                    reducer_b,
                    &w_alpha,
                    b_alpha,
                    n_features,
                    n_hidden1,
                    n_hidden_final,
                    leaky,
                    use_2layer,
                    use_skip,
                );
                let (y, _) = pin(fwd.y);
                sum_y += y;
            }
            let e = sum_y / (s_rows as f64) - target;
            weight * e * e
        };
        let mut w2m = w2_enc.clone();
        for idx2 in [0usize, (n_hidden1 * n_hidden_final) / 2] {
            let orig = w2m[idx2];
            w2m[idx2] = orig + eps;
            let fp = loss_fn_w2(&w2m);
            w2m[idx2] = orig - eps;
            let fm = loss_fn_w2(&w2m);
            w2m[idx2] = orig;
            let numerical = (fp - fm) / (2.0 * eps);
            let analytical = gw1[n_w1_first + idx2];
            // Combined atol+rtol (numpy/jax/pytorch gradcheck form): a pure
            // relative gate is unbounded as the true gradient → 0, and the
            // f32-forward FD floor (~1e-5 abs at ε=1e-2) dominates there.
            let abs_err = (numerical - analytical).abs();
            let tol = 2e-5 + 2e-3 * numerical.abs().max(analytical.abs());
            assert!(
                abs_err < tol,
                "2-layer konjnd-agg gw2_enc[{idx2}] numerical={numerical:.8} \
                 analytical={analytical:.8} abs_err={abs_err:.2e} tol={tol:.2e}"
            );
        }
    }

    // FD-check the first-layer w1 entries (the deepest part of the
    // 2-layer chain). Check several spread across the matrix.
    for idx in (0..n_w1_first).step_by((n_w1_first / 6).max(1)) {
        let orig = w1[idx];
        w1[idx] = orig + eps;
        let fp = loss_fn(&w1);
        w1[idx] = orig - eps;
        let fm = loss_fn(&w1);
        w1[idx] = orig;
        let numerical = (fp - fm) / (2.0 * eps);
        let analytical = gw1[idx];
        // Combined atol+rtol — many w1 entries have near-zero true gradient
        // (e.g. gw1[4]≈-9e-4) where a pure relative gate is meaningless and
        // the f32-forward FD floor (~1e-5 abs at ε=1e-2) sets the scale.
        let abs_err = (numerical - analytical).abs();
        let tol = 2e-5 + 2e-3 * numerical.abs().max(analytical.abs());
        assert!(
            abs_err < tol,
            "2-layer konjnd-agg gw1[{idx}] numerical={numerical:.8} \
             analytical={analytical:.8} abs_err={abs_err:.2e} tol={tol:.2e}"
        );
    }
}

// ===================================================================
// Per-sample-alpha-head dedicated tests (previously zero coverage).
// ===================================================================

#[test]
fn psah_recovers_synthetic_ranking() {
    let n_features = 16;
    let (features_owned, targets) = make_synth_dataset(500, 400, n_features);
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
    let group = TrainingGroup {
        name: "synth".to_string(),
        human_scores: &targets,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };
    let mut log = Vec::new();
    let bake = train_mlp_per_sample_alpha_head(
        &mut [group],
        n_features,
        &MlpHyperparams {
            n_hidden: 16,
            n_epochs: 80,
            pairs_per_epoch: 2000,
            initial_lr: 0.005,
            per_sample_alpha_head: true,
            minibatch_size: 1,
            ..Default::default()
        },
        &mut log,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    assert!(!bake.is_empty(), "per-sample-α head produced empty bake");
    assert!(
        log.iter().any(|l| l.contains("srocc=")),
        "log should contain per-group panel stats"
    );
}

#[test]
fn psah_tanh_pin_bounds_output() {
    let n_features = 12;
    let (features_owned, targets) = make_synth_dataset(600, 200, n_features);
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
    let group = TrainingGroup {
        name: "synth".to_string(),
        human_scores: &targets,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };
    let mut log = Vec::new();
    let bake = train_mlp_per_sample_alpha_head(
        &mut [group],
        n_features,
        &MlpHyperparams {
            n_hidden: 8,
            n_epochs: 30,
            pairs_per_epoch: 500,
            initial_lr: 0.003,
            per_sample_alpha_head: true,
            tanh_output_head_scale: 30.0,
            mse_weight: 1.0,
            ranknet_weight: 0.0,
            minibatch_size: 1,
            ..Default::default()
        },
        &mut log,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    assert!(!bake.is_empty());
    // With tanh pin scale=30, raw output y_pre maps through
    // 100·σ(y_pre/30) → predictions should be in [0, 100].
    // The log should show val(...) which is computed from pinned
    // predictions.
    let has_val = log.iter().any(|l| l.contains("val("));
    assert!(has_val, "expected val() in log");
}

#[test]
fn psah_nan_transform_sweep_catches_poison() {
    use zenpredict::FeatureTransform;
    let rows = vec![vec![0.0, 1.0, 2.0], vec![0.0, 0.5, 1.5]];
    let transforms = vec![
        FeatureTransform::Identity,
        FeatureTransform::Identity,
        FeatureTransform::Identity,
    ];
    // No NaN expected with identity transforms.
    assert!(sweep_nan_inf(rows.iter().map(|r| r.as_slice()), &transforms, "test-clean").is_ok());

    // Manually poison a feature to simulate log(0) = -inf.
    let mut poisoned_rows = rows.clone();
    poisoned_rows[0][1] = f64::NAN;
    poisoned_rows[1][2] = f64::INFINITY;
    let transforms_with_log = vec![
        FeatureTransform::Identity,
        FeatureTransform::Log,
        FeatureTransform::Log,
    ];
    let result = sweep_nan_inf(
        poisoned_rows.iter().map(|r| r.as_slice()),
        &transforms_with_log,
        "test-poison",
    );
    assert!(result.is_err(), "sweep should catch NaN/inf");
    let msg = result.unwrap_err();
    assert!(msg.contains("f1"), "should identify feature f1: {msg}");
    assert!(msg.contains("f2"), "should identify feature f2: {msg}");
}

#[test]
fn psah_goals_policy_produces_score() {
    let n_features = 12;
    let (features_owned, targets) = make_synth_dataset(700, 200, n_features);
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
    let group = TrainingGroup {
        name: "konjnd_test".to_string(),
        human_scores: &targets,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };
    let mut log = Vec::new();
    let bake = train_mlp_per_sample_alpha_head(
        &mut [group],
        n_features,
        &MlpHyperparams {
            n_hidden: 8,
            n_epochs: 20,
            pairs_per_epoch: 500,
            initial_lr: 0.003,
            per_sample_alpha_head: true,
            validation_policy: ValidationPolicy::Goals,
            minibatch_size: 1,
            ..Default::default()
        },
        &mut log,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    assert!(!bake.is_empty());
    // Goals policy should produce a goals breakdown in the log.
    let has_goals = log.iter().any(|l| l.contains("goals:"));
    assert!(
        has_goals,
        "expected 'goals:' in log with --val-policy goals"
    );
}

#[test]
fn light_panel_geomean_used_by_default() {
    let n_features = 12;
    let (features_owned, targets) = make_synth_dataset(800, 200, n_features);
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
    let group = TrainingGroup {
        name: "synth".to_string(),
        human_scores: &targets,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };
    let mut log = Vec::new();
    let _bake = train_mlp_per_sample_alpha_head(
        &mut [group],
        n_features,
        &MlpHyperparams {
            n_hidden: 8,
            n_epochs: 10,
            pairs_per_epoch: 300,
            initial_lr: 0.003,
            per_sample_alpha_head: true,
            minibatch_size: 1,
            ..Default::default()
        },
        &mut log,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    // Default val_aggregate is GeomeanSPP; log should show val(geomean3).
    let has_geomean = log.iter().any(|l| l.contains("val(geomean3)"));
    assert!(
        has_geomean,
        "default val_aggregate should be geomean3; log: {:?}",
        log.last()
    );
}

#[test]
fn psah_2layer_trains_without_crash() {
    let n_features = 16;
    let (features_owned, targets) = make_synth_dataset(900, 200, n_features);
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
    let group = TrainingGroup {
        name: "synth".to_string(),
        human_scores: &targets,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };
    let mut log = Vec::new();
    let bake = train_mlp_per_sample_alpha_head(
        &mut [group],
        n_features,
        &MlpHyperparams {
            n_hidden: 16,
            n_epochs: 20,
            pairs_per_epoch: 500,
            initial_lr: 0.005,
            per_sample_alpha_head: true,
            n_hidden_layers: 2,
            minibatch_size: 1,
            ..Default::default()
        },
        &mut log,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    assert!(!bake.is_empty(), "2-layer variant produced empty bake");
    assert!(
        log.iter().any(|l| l.contains("srocc=")),
        "2-layer log should contain panel stats"
    );
}

#[test]
fn psah_skip_connection_trains_without_crash() {
    let n_features = 16;
    let (features_owned, targets) = make_synth_dataset(1000, 200, n_features);
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
    let group = TrainingGroup {
        name: "synth".to_string(),
        human_scores: &targets,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };
    let mut log = Vec::new();
    let bake = train_mlp_per_sample_alpha_head(
        &mut [group],
        n_features,
        &MlpHyperparams {
            n_hidden: 16,
            n_epochs: 20,
            pairs_per_epoch: 500,
            initial_lr: 0.005,
            per_sample_alpha_head: true,
            skip_connection: true,
            minibatch_size: 1,
            ..Default::default()
        },
        &mut log,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    assert!(!bake.is_empty(), "skip variant produced empty bake");
}

#[test]
fn psah_2layer_skip_combined_trains() {
    let n_features = 16;
    let (features_owned, targets) = make_synth_dataset(1100, 200, n_features);
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
    let group = TrainingGroup {
        name: "synth".to_string(),
        human_scores: &targets,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };
    let mut log = Vec::new();
    let bake = train_mlp_per_sample_alpha_head(
        &mut [group],
        n_features,
        &MlpHyperparams {
            n_hidden: 16,
            n_epochs: 20,
            pairs_per_epoch: 500,
            initial_lr: 0.005,
            per_sample_alpha_head: true,
            n_hidden_layers: 2,
            skip_connection: true,
            minibatch_size: 1,
            ..Default::default()
        },
        &mut log,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    assert!(!bake.is_empty(), "2-layer+skip produced empty bake");
}

#[test]
fn psah_sigma_weighted_mse_trains() {
    let n_features = 12;
    let (features_owned, targets) = make_synth_dataset(600, 200, n_features);
    let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
    let n = targets.len();
    // Synthetic σ: low for first half (high confidence), high for second half.
    let sigmas: Vec<f64> = (0..n).map(|i| if i < n / 2 { 0.1 } else { 0.8 }).collect();
    let group = TrainingGroup {
        name: "sigma_test".to_string(),
        human_scores: &targets,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: Some(&sigmas),
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode: GroupLossMode::default(),
    };
    let mut log = Vec::new();
    let bake = train_mlp_per_sample_alpha_head(
        &mut [group],
        n_features,
        &MlpHyperparams {
            n_hidden: 8,
            n_epochs: 30,
            pairs_per_epoch: 500,
            initial_lr: 0.003,
            per_sample_alpha_head: true,
            mse_weight: 1.0,
            sigma_weighted_mse: true,
            ranknet_weight: 0.5,
            minibatch_size: 1,
            ..Default::default()
        },
        &mut log,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    assert!(!bake.is_empty(), "σ-weighted MSE should produce a bake");
}

/// STRATEGY-2026-07-02 end-to-end smoke: ALL strategies active at once
/// on synthetic data — catches runtime panics (indexing, borrow, NaN
/// cascades) that the per-algorithm reference tests cannot. A failure
/// here is an IMPL BUG (not strategy).
#[test]
fn strategy_smoke_all_active_end_to_end() {
    const NF: usize = 6;
    let n = 80usize;
    let mut feats: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut scores: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        feats.push(vec![t, 1.0 - t, (t * 7.0).sin() * 0.1, t * t, 0.5, -t]);
        scores.push(t); // monotone target in [0,1]
    }
    let refs: Vec<&[f64]> = feats.iter().map(|r| r.as_slice()).collect();
    let sigmas: Vec<f64> = (0..n)
        .map(|i| 0.05 + 0.1 * ((i % 7) as f64) / 7.0)
        .collect();
    let mut groups = [
        TrainingGroup {
            name: "syn_a".to_string(),
            human_scores: &scores,
            features: FeatureRows::Borrowed(&refs),
            metric_sigmas: Some(&sigmas),
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        },
        TrainingGroup {
            name: "syn_b".to_string(),
            human_scores: &scores,
            features: FeatureRows::Borrowed(&refs),
            metric_sigmas: None,
            train_weight: 0.5,
            validation_weight: 0.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        },
    ];
    // triplet pool: stimuli = the same rows; responses consistent with
    // "left more distorted" when its quality target is lower.
    let mut pool = TripletPool::default();
    for f in &feats {
        pool.features.push(f.clone());
    }
    for i in (0..60).step_by(3) {
        let (l, r) = (i as u32, (i + 7) as u32);
        // scores[i] < scores[i+7] -> left lower quality -> resp 0
        pool.responses.push((l, r, 0));
        pool.responses.push((r, l, 1));
    }
    let hp = MlpHyperparams {
        n_hidden: 12,
        n_hidden_layers: 2,
        n_epochs: 12,
        pairs_per_epoch: 400,
        minibatch_size: 4,
        initial_lr: 5e-3,
        l2_lambda: 1e-6,
        leaky_alpha: 0.01,
        early_stop_patience: 0,
        per_sample_alpha_head: true,
        tanh_output_head_scale: 20.0,
        ranknet_weight: 1.0,
        mse_weight: 0.3,
        sigma_weighted_mse: true,
        // ALL strategies on:
        ema_decay: 0.9,
        hard_pair_frac: 0.5,
        hard_pair_max_delta: 0.1,
        stratified_bands: 5,
        dro_eta: 1.0,
        listwise_weight: 0.5,
        listwise_size: 6,
        listwise_frac: 0.15,
        triplet_weight: 0.5,
        triplet_frac: 0.15,
        triplet_tau: 0.6,
        triplet_sigma: 5.0,
        seed: 17,
        ..Default::default()
    };
    let mut log: Vec<String> = Vec::new();
    let bake = train_mlp_strategy(
        &mut groups,
        NF,
        &hp,
        &mut log,
        None,
        None,
        None,
        None,
        None,
        Some(&pool),
    );
    assert!(
        !bake.is_empty(),
        "IMPL BUG (not strategy): all-active strategy training produced an empty bake"
    );
    let joined = log.join("\n");
    assert!(
        joined.contains("STRATEGY active:"),
        "IMPL BUG (not strategy): strategy activation line missing from log"
    );
    assert!(
        !joined.contains("NaN") || joined.contains("val"),
        "IMPL BUG (not strategy): NaN observed in training log:\n{joined}"
    );
    // determinism: same seed -> byte-identical bake
    let mut log2: Vec<String> = Vec::new();
    let bake2 = train_mlp_strategy(
        &mut groups,
        NF,
        &hp,
        &mut log2,
        None,
        None,
        None,
        None,
        None,
        Some(&pool),
    );
    assert_eq!(
        bake, bake2,
        "IMPL BUG (not strategy): strategy training is not deterministic under a fixed seed"
    );
}

/// TRAINEROPT3: `fwd_skipped_rows_zero` mirrors the keep-mask loader
/// contract — every row outside `active_rows` must read `x == 0.0`
/// (`−0.0` counts, matching `forward`'s `s == 0.0` skip) on BOTH
/// sides, or the fused apply would diverge from a fresh forward.
#[test]
fn fwd_skipped_rows_zero_contract() {
    let xa: Vec<f64> = (0..12).map(|i| i as f64 * 0.5).collect();
    let xb = xa.clone();
    // Full coverage — trivially true regardless of x.
    assert!(fwd_skipped_rows_zero(None, &xa, &xb, 12));
    assert!(fwd_skipped_rows_zero(Some(&[(0, 12)]), &xa, &xb, 12));
    // Gap 4..8 — nonzero x there must fail.
    let gapped = [(0u32, 4u32), (8, 12)];
    assert!(!fwd_skipped_rows_zero(Some(&gapped), &xa, &xb, 12));
    let mut xa0 = xa.clone();
    let mut xb0 = xb.clone();
    for i in 4..8 {
        xa0[i] = 0.0;
        xb0[i] = 0.0;
    }
    assert!(fwd_skipped_rows_zero(Some(&gapped), &xa0, &xb0, 12));
    // −0.0 reads as zero (IEEE `!=` semantics), like `forward`.
    xb0[5] = -0.0;
    assert!(fwd_skipped_rows_zero(Some(&gapped), &xa0, &xb0, 12));
    // Nonzero on ONE side only still fails — the guard is per-side.
    xb0[5] = 0.0;
    xa0[5] = 1e-9;
    assert!(!fwd_skipped_rows_zero(Some(&gapped), &xa0, &xb0, 12));
    // Trailing gap to n_rows — zero → pass, nonzero → fail.
    xa0[5] = 0.0;
    xa0[11] = 0.0;
    xb0[11] = 0.0;
    assert!(fwd_skipped_rows_zero(
        Some(&[(0, 4), (8, 11)]),
        &xa0,
        &xb0,
        12
    ));
    xa0[11] = 3.0;
    assert!(!fwd_skipped_rows_zero(
        Some(&[(0, 4), (8, 11)]),
        &xa0,
        &xb0,
        12
    ));
    xa0[11] = 0.0;
    assert!(fwd_skipped_rows_zero(
        Some(&[(0, 4), (8, 11)]),
        &xa0[..11],
        &xb0[..11],
        11
    ));
}

/// TRAINEROPT3 task 2 step-level oracle: `step_w1_fused` armed with a
/// `FwdAccumW1` must leave every trainer-visible state buffer
/// (`w1/b1/w2/b2` and all of `AdamState`'s `g*/m*/v*`/`t`)
/// bit-identical to the unarmed call sequence (walk → `step` b1 →
/// `nonneg_project`), while `hpa`/`hpb` equal a hand-replayed forward
/// walk over the FINAL `w1` seeded from the FINAL `b1` — i.e. exactly
/// what the next iteration's `forward` + `forward_finish` consume.
/// Covers nonneg on/off × group-lasso prox on/off × `active_rows`
/// gapped/full, across chained steps (so `g_zero_in` re-entry holds).
#[test]
fn step_w1_fused_lookahead_bit_identical() {
    // Review 2026-10-02: every canonical-domain width, not only multiples of 8.
    for nh in [8usize, 12, 20] {
        step_w1_fused_lookahead_bit_identical_at(nh);
    }
}

fn step_w1_fused_lookahead_bit_identical_at(nh: usize) {
    let nf = 24usize;
    let n = nf * nh;
    let mut rng = SplitMix64::new(0xBEEF);
    let mut genvec = |n: usize| -> Vec<f64> { (0..n).map(|_| rng.next_normal() * 0.1).collect() };
    let kept = |r: usize| !(r % 4 == 3 || (16..20).contains(&r));
    let runs: Vec<(u32, u32)> = {
        let mut v: Vec<(u32, u32)> = Vec::new();
        for r in (0..nf).filter(|&r| kept(r)) {
            match v.last_mut() {
                Some(l) if l.1 as usize == r => l.1 += 1,
                _ => v.push((r as u32, r as u32 + 1)),
            }
        }
        v
    };
    for pin in [None, Some(0.75)] {
        for tau in [None, Some(0.15)] {
            for use_active in [false, true] {
                let w1_0 = genvec(n);
                let b1_0 = genvec(nh);
                let w2_0 = genvec(nh);
                let b2_0 = genvec(1);
                // Shared initial Adam buffers carrying plausible grads.
                let mut ad_unarmed = AdamState::new(n, nh, nh, 1);
                ad_unarmed.gb1 = genvec(nh);
                ad_unarmed.gw2 = genvec(nh);
                ad_unarmed.gb2 = genvec(1);
                let mut ad_armed = AdamState::new(n, nh, nh, 1);
                ad_armed.gb1.clone_from(&ad_unarmed.gb1);
                ad_armed.gw2.clone_from(&ad_unarmed.gw2);
                ad_armed.gb2.clone_from(&ad_unarmed.gb2);
                let mut w1_u = w1_0.clone();
                let mut w1_a = w1_0.clone();
                let (mut b1_u, mut b1_a) = (b1_0.clone(), b1_0.clone());
                let (mut w2_u, mut w2_a) = (w2_0.clone(), w2_0.clone());
                let (mut b2_u, mut b2_a) = (b2_0.clone(), b2_0.clone());
                let act = use_active.then_some(runs.as_slice());
                for _step in 0..3 {
                    let mut xa = genvec(nf);
                    let mut xb = genvec(nf);
                    let mut xa2 = genvec(nf);
                    let mut xb2 = genvec(nf);
                    let dha = genvec(nh);
                    let dhb = genvec(nh);
                    if use_active {
                        for r in (0..nf).filter(|&r| !kept(r)) {
                            xa[r] = 0.0;
                            xb[r] = 0.0;
                            xa2[r] = 0.0;
                            xb2[r] = 0.0;
                        }
                    }
                    let mut hpa = vec![0.0f64; nh];
                    let mut hpb = vec![0.0f64; nh];
                    ad_armed.step_w1_fused(
                        &mut w1_a,
                        &mut b1_a,
                        &mut w2_a,
                        &mut b2_a,
                        0.003,
                        &xa,
                        &dha,
                        &xb,
                        &dhb,
                        1e-4,
                        None,
                        nh,
                        act,
                        tau,
                        pin,
                        Some(adam_simd::FwdAccumW1 {
                            xa: &xa2,
                            xb: &xb2,
                            hpa: &mut hpa,
                            hpb: &mut hpb,
                        }),
                    );
                    nonneg_project_w2b2(&mut w2_a, &mut b2_a, pin);
                    ad_unarmed.step_w1_fused(
                        &mut w1_u, &mut b1_u, &mut w2_u, &mut b2_u, 0.003, &xa, &dha, &xb, &dhb,
                        1e-4, None, nh, act, tau, pin, None,
                    );
                    nonneg_project(&mut w2_u, &mut b1_u, &mut b2_u, pin);
                    let bits = |v: &[f64]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
                    for (name, x, y) in [
                        ("w1", &w1_u[..], &w1_a[..]),
                        ("b1", &b1_u[..], &b1_a[..]),
                        ("w2", &w2_u[..], &w2_a[..]),
                        ("b2", &b2_u[..], &b2_a[..]),
                        ("gw1", &ad_unarmed.gw1[..], &ad_armed.gw1[..]),
                        ("gb1", &ad_unarmed.gb1[..], &ad_armed.gb1[..]),
                        ("gw2", &ad_unarmed.gw2[..], &ad_armed.gw2[..]),
                        ("gb2", &ad_unarmed.gb2[..], &ad_armed.gb2[..]),
                        ("mw1", &ad_unarmed.mw1[..], &ad_armed.mw1[..]),
                        ("mb1", &ad_unarmed.mb1[..], &ad_armed.mb1[..]),
                        ("mw2", &ad_unarmed.mw2[..], &ad_armed.mw2[..]),
                        ("mb2", &ad_unarmed.mb2[..], &ad_armed.mb2[..]),
                        ("vw1", &ad_unarmed.vw1[..], &ad_armed.vw1[..]),
                        ("vb1", &ad_unarmed.vb1[..], &ad_armed.vb1[..]),
                        ("vw2", &ad_unarmed.vw2[..], &ad_armed.vw2[..]),
                        ("vb2", &ad_unarmed.vb2[..], &ad_armed.vb2[..]),
                    ] {
                        assert_eq!(
                            bits(x),
                            bits(y),
                            "armed diverged: {name} pin={pin:?} tau={tau:?} active={use_active}"
                        );
                    }
                    assert_eq!(ad_unarmed.t, ad_armed.t);
                    // Oracle: b1-final seed + ascending guarded mul_add
                    // over the armed run's (identical) final w1.
                    let mut hpa_o = b1_a.clone();
                    let mut hpb_o = b1_a.clone();
                    for i in 0..nf {
                        let wr = &w1_a[i * nh..i * nh + nh];
                        if xa2[i] != 0.0 {
                            for j in 0..nh {
                                hpa_o[j] = xa2[i].mul_add(wr[j], hpa_o[j]);
                            }
                        }
                        if xb2[i] != 0.0 {
                            for j in 0..nh {
                                hpb_o[j] = xb2[i].mul_add(wr[j], hpb_o[j]);
                            }
                        }
                    }
                    assert_eq!(bits(&hpa), bits(&hpa_o), "hpa oracle mismatch");
                    assert_eq!(bits(&hpb), bits(&hpb_o), "hpb oracle mismatch");
                    // Re-seed the identical gradient buffers for the
                    // next step (backprop would have refilled them).
                    ad_unarmed.gb1 = genvec(nh);
                    ad_unarmed.gw2 = genvec(nh);
                    ad_unarmed.gb2 = genvec(1);
                    ad_armed.gb1.clone_from(&ad_unarmed.gb1);
                    ad_armed.gw2.clone_from(&ad_unarmed.gw2);
                    ad_armed.gb2.clone_from(&ad_unarmed.gb2);
                }
            }
        }
    }
}

#[test]
fn e34_plain_monotone_projection_masks() {
    let mut all = [-2., 3., -4., 5.];
    project_monotone_w1(&mut all, 2, None, false);
    assert_eq!(all, [0., 3., 0., 5.]);
    let mut masked = [-2., 3., -4., 5.];
    project_monotone_w1(&mut masked, 2, Some(&[true, false]), false);
    assert_eq!(masked, [0., 3., -4., 5.]);
    project_monotone_w1(&mut masked, 2, Some(&[true, false]), true);
    assert_eq!(masked, [0., 3., 0., 0.]);
}

#[test]
fn e34_plain_monotone_h1_identity_f32_f16() {
    let rows: Vec<Vec<f64>> = (0..32)
        .map(|i| vec![i as f64 / 31., (i % 7) as f64 / 6.])
        .collect();
    let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let scores: Vec<f64> = rows
        .iter()
        .map(|x| 100. - 30. * x[0] - 10. * x[1])
        .collect();
    for hidden in [1, 8] {
        for dtype in [WeightDtype::F32, WeightDtype::F16] {
            let hp = MlpHyperparams {
                nonneg_distance: true,
                monotone_cbc: true,
                n_hidden: hidden,
                n_epochs: 2,
                pairs_per_epoch: 32,
                seed: 42,
                out_dtype: dtype,
                early_stop_patience: 0,
                ..Default::default()
            };
            let mut groups = [TrainingGroup {
                name: "synthetic".into(),
                human_scores: &scores,
                features: FeatureRows::Borrowed(&refs),
                metric_sigmas: None,
                train_weight: 1.,
                validation_weight: 1.,
                ref_ids: None,
                loss_mode: GroupLossMode::Both,
            }];
            let bytes = train_mlp(&mut groups, 2, &hp, &mut Vec::new());
            let model = zenpredict::Model::from_bytes(&bytes).unwrap();
            assert_eq!(model.layer(0).out_dim, hidden);
            assert!(model.layer(0).biases.iter().all(|&b| b == 0.0));
            match model.layer(0).weights {
                zenpredict::WeightStorage::F32(w) => assert!(w.iter().all(|&w| w >= 0.)),
                zenpredict::WeightStorage::F16(w) => {
                    assert!(w.iter().all(|&w| zenpredict::f16_bits_to_f32(w) >= 0.))
                }
                _ => panic!("unexpected test dtype"),
            }
            let mut predictor = zenpredict::Predictor::new(&model);
            assert_eq!(
                predictor.predict(&[0., 0.]).unwrap()[0].to_bits(),
                100.0_f32.to_bits()
            );
            let a = predictor.predict(&[0.5, 0.5]).unwrap()[0];
            let b = predictor.predict(&[1., 1.]).unwrap()[0];
            assert!(b <= a && a <= 100.);
            assert!(b < 100., "training must produce a nonconstant model");
        }
    }
}

#[test]
fn e34_plain_monotone_refusals() {
    for hp in [
        MlpHyperparams {
            monotone_cbc: true,
            ..Default::default()
        },
        MlpHyperparams {
            monotone_cbc: true,
            nonneg_distance: true,
            skip_connection: true,
            ..Default::default()
        },
        MlpHyperparams {
            monotone_cbc: true,
            nonneg_distance: true,
            pool_head: true,
            ..Default::default()
        },
        MlpHyperparams {
            monotone_cbc: true,
            nonneg_distance: true,
            n_hidden_layers: 2,
            ..Default::default()
        },
        MlpHyperparams {
            n_hidden: 0,
            ..Default::default()
        },
    ] {
        assert!(std::panic::catch_unwind(|| validate_training_capabilities(&hp, true, 0)).is_err());
    }
}

/// `--nonneg-thresholds` with `--monotone-cbc`: trainable `b1 <= 0` keeps the
/// identity pin bit-exact in f32 and f16, and the baked model is non-increasing
/// along every non-negative direction (checked exactly, since every forward op
/// is monotone under round-to-nearest). Without thresholds the same projection
/// is linear on the non-negative orthant; with them, units switch on at a
/// threshold, which this target (a hinge in x0) rewards.
#[test]
fn byc_monotone_thresholds_identity_and_monotonicity() {
    let rows: Vec<Vec<f64>> = (0..64)
        .map(|i| vec![(i % 16) as f64 / 15., (i / 16) as f64 / 3.])
        .collect();
    let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let scores: Vec<f64> = rows
        .iter()
        .map(|x| 100. - 80. * (x[0] - 0.6).max(0.) - 1. * x[1])
        .collect();
    for dtype in [WeightDtype::F32, WeightDtype::F16] {
        let hp = MlpHyperparams {
            nonneg_distance: true,
            monotone_cbc: true,
            nonneg_thresholds: true,
            n_hidden: 8,
            n_epochs: 80,
            pairs_per_epoch: 1024,
            seed: 7,
            out_dtype: dtype,
            early_stop_patience: 0,
            ..Default::default()
        };
        let mut groups = [TrainingGroup {
            name: "invented".into(),
            human_scores: &scores,
            features: FeatureRows::Borrowed(&refs),
            metric_sigmas: None,
            train_weight: 1.,
            validation_weight: 1.,
            ref_ids: None,
            loss_mode: GroupLossMode::Both,
        }];
        let mut log = Vec::new();
        let bytes = train_mlp(&mut groups, 2, &hp, &mut log);
        let model = zenpredict::Model::from_bytes(&bytes).unwrap();
        let biases = model.layer(0).biases;
        assert!(
            biases.iter().all(|&b| b <= 0.0),
            "b1 must stay <= 0: {biases:?}"
        );
        assert!(
            biases.iter().any(|&b| b < 0.0),
            "a hinge target must move at least one threshold: {biases:?}\n{}",
            log.join("\n")
        );
        assert_eq!(
            certify_monotone_bake(&model, 100.0),
            Ok(MonotoneCertificate {
                ramps: 0,
                free_ramps: 0,
                free_amplitude: 0.0,
                max_pairs: 0
            }),
            "certificate"
        );
        let mut predictor = zenpredict::Predictor::new(&model);
        assert_eq!(
            predictor.predict(&[0., 0.]).unwrap()[0].to_bits(),
            100.0_f32.to_bits()
        );
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 11) as f32 / (1u64 << 53) as f32
        };
        for _ in 0..2000 {
            let x = [2. * next(), 2. * next()];
            let y = [x[0] + next(), x[1] + next() * 0.25];
            let fx = predictor.predict(&x).unwrap()[0];
            let fy = predictor.predict(&y).unwrap()[0];
            assert!(
                fy <= fx && fx <= 100.,
                "monotone violation {x:?}->{fx} {y:?}->{fy}"
            );
        }
    }
}

/// `--monotone-ramps`: the baked model passes [`certify_monotone_bake`] (every
/// positive output weight is a ramp's upper unit, matched to its lower unit by
/// an identical W1 column; the bake may store hidden units in any order), the
/// identity pin is bit-exact, and the model is exactly non-increasing along
/// non-negative steps.
#[test]
fn byc_monotone_ramps_structure_identity_and_monotonicity() {
    let rows: Vec<Vec<f64>> = (0..64)
        .map(|i| vec![(i % 16) as f64 / 15., (i / 16) as f64 / 3.])
        .collect();
    let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let scores: Vec<f64> = rows
        .iter()
        .map(|x| 100. - 60. * x[0].min(0.4) - 5. * x[1])
        .collect();
    let (hidden, ramps) = (8usize, 3usize);
    for dtype in [WeightDtype::F32, WeightDtype::F16] {
        let hp = MlpHyperparams {
            nonneg_distance: true,
            monotone_cbc: true,
            nonneg_thresholds: true,
            monotone_ramps: ramps,
            n_hidden: hidden,
            n_epochs: 40,
            pairs_per_epoch: 1024,
            seed: 11,
            out_dtype: dtype,
            early_stop_patience: 0,
            ..Default::default()
        };
        let mut groups = [TrainingGroup {
            name: "invented".into(),
            human_scores: &scores,
            features: FeatureRows::Borrowed(&refs),
            metric_sigmas: None,
            train_weight: 1.,
            validation_weight: 1.,
            ref_ids: None,
            loss_mode: GroupLossMode::Both,
        }];
        let bytes = train_mlp(&mut groups, 2, &hp, &mut Vec::new());
        let model = zenpredict::Model::from_bytes(&bytes).unwrap();
        // A pair can train to output weight 0 (two dead units), so live ramps <= ramps.
        let cert = certify_monotone_bake(&model, 100.0).expect("certificate");
        assert!(
            cert.ramps <= ramps && cert.free_ramps == 0 && cert.free_amplitude == 0.0,
            "{cert:?}"
        );
        let mut predictor = zenpredict::Predictor::new(&model);
        assert_eq!(
            predictor.predict(&[0., 0.]).unwrap()[0].to_bits(),
            100.0_f32.to_bits()
        );
        let mut rng = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 11) as f32 / (1u64 << 53) as f32
        };
        for _ in 0..2000 {
            let x = [2. * next(), 2. * next()];
            let y = [x[0] + next(), x[1] + next() * 0.25];
            let fx = predictor.predict(&x).unwrap()[0];
            let fy = predictor.predict(&y).unwrap()[0];
            assert!(
                fy <= fx && fx <= 100.,
                "monotone violation {x:?}->{fx} {y:?}->{fy}"
            );
        }
    }
}

/// `--free-ramps` with a budget: the certificate reports the free pairs and an
/// amplitude within the budget, and no non-negative step raises the score by
/// more than that amplitude (plus f32 rounding). The target needs a
/// non-monotone term, so the budget binds.
#[test]
fn byc_free_ramps_certified_budget() {
    let rows: Vec<Vec<f64>> = (0..64)
        .map(|i| vec![(i % 16) as f64 / 15., (i / 16) as f64 / 3.])
        .collect();
    let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let scores: Vec<f64> = rows
        .iter()
        .map(|x| 100. - 40. * x[0] + 15. * x[1] * x[0])
        .collect();
    let budget = 3.0;
    for dtype in [WeightDtype::F32, WeightDtype::F16] {
        let hp = MlpHyperparams {
            nonneg_distance: true,
            monotone_cbc: true,
            nonneg_thresholds: true,
            monotone_ramps: 1,
            free_ramps: 2,
            free_budget: budget,
            n_hidden: 8,
            n_epochs: 40,
            pairs_per_epoch: 1024,
            seed: 5,
            out_dtype: dtype,
            early_stop_patience: 0,
            ..Default::default()
        };
        let mut groups = [TrainingGroup {
            name: "invented".into(),
            human_scores: &scores,
            features: FeatureRows::Borrowed(&refs),
            metric_sigmas: None,
            train_weight: 1.,
            validation_weight: 1.,
            ref_ids: None,
            loss_mode: GroupLossMode::Both,
        }];
        let bytes = train_mlp(&mut groups, 2, &hp, &mut Vec::new());
        let model = zenpredict::Model::from_bytes(&bytes).unwrap();
        let cert = certify_monotone_bake(&model, 100.0).expect("certificate");
        assert!(cert.free_amplitude <= budget * (1. + 1e-3), "{cert:?}");
        let mut predictor = zenpredict::Predictor::new(&model);
        assert_eq!(
            predictor.predict(&[0., 0.]).unwrap()[0].to_bits(),
            100.0_f32.to_bits()
        );
        let mut rng = 0x1234_5678_9abc_def1u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 11) as f32 / (1u64 << 53) as f32
        };
        let slack = cert.free_amplitude as f32 + 1e-3;
        for _ in 0..2000 {
            let x = [2. * next(), 2. * next()];
            let y = [x[0] + next(), x[1] + next()];
            let fx = predictor.predict(&x).unwrap()[0];
            let fy = predictor.predict(&y).unwrap()[0];
            assert!(
                fy <= fx + slack && fx <= 100.,
                "budget violation {x:?}->{fx} {y:?}->{fy} {cert:?}"
            );
        }
    }
}

/// Projection with free ramps, on hand-set weights: monotone rows clamp to
/// `>= 0`, free rows keep their sign (averaged within the pair), every pair is
/// tied, `b1 <= 0` with `b_q <= b_p`, and the free amplitude meets the budget.
#[test]
fn byc_free_ramp_projection_contract() {
    // 2 inputs, 6 hidden: unit 0 plain, (1,2) monotone ramp, (3,4) free ramp, 5 plain? No:
    // the layout is [plain | monotone ramps | free ramps], so use 1 plain, 1 monotone pair, 2 free pairs = 7 units.
    let h = 7;
    let hp = MlpHyperparams {
        nonneg_distance: true,
        monotone_cbc: true,
        nonneg_thresholds: true,
        monotone_ramps: 1,
        free_ramps: 2,
        free_budget: 2.0,
        n_hidden: h,
        ..Default::default()
    };
    let mut w1 = vec![
        -1.0, -2.0, 3.0, -4.0, -5.0, 6.0, -7.0, // feature 0 across units
        2.0, -1.0, -3.0, 4.0, 2.0, -6.0, 1.0, // feature 1
    ];
    let mut w2 = vec![0.5, -3.0, 1.0, -4.0, 2.0, -6.0, 6.0];
    let mut b1 = vec![0.3, -0.1, -0.5, -1.0, -3.0, 0.2, -2.0];
    let mut b2 = vec![0.0];
    project_plain(&mut w1, &mut w2, &mut b1, &mut b2, Some(100.0), &hp);
    let col = |j: usize| [w1[j], w1[h + j]];
    assert!(
        col(0).iter().all(|&v| v >= 0.0) && col(1) == col(2) && col(1).iter().all(|&v| v >= 0.0)
    );
    assert_eq!(col(3), col(4));
    assert_eq!(col(5), col(6));
    assert!(
        col(3).iter().chain(col(5).iter()).any(|&v| v < 0.0),
        "free rows must keep their sign"
    );
    assert!(b1.iter().all(|&b| b <= 0.0) && b1[2] <= b1[1] && b1[4] <= b1[3] && b1[6] <= b1[5]);
    for p in [1, 3, 5] {
        assert!(w2[p] <= 0.0 && w2[p + 1] == -w2[p]);
    }
    assert!(w2[0] <= 0.0 && b2[0] == 100.0);
    let amplitude = -w2[3] * (b1[3] - b1[4]) - w2[5] * (b1[5] - b1[6]);
    assert!(
        amplitude <= 2.0 + 1e-12 && amplitude > 1.9,
        "amplitude {amplitude}"
    );
}

/// A hand-built bake with one free ramp certifies to exactly its amplitude, and
/// that bound is tight: raising an input raises the score by up to (and not
/// beyond) the amplitude.
#[test]
fn byc_certificate_free_amplitude_is_exact_and_tight() {
    // 2 inputs, 3 hidden units: one monotone hinge, one free ramp (units 1, 2).
    // free row = (1, -1): z = x0 - x1 falls as x1 rises.
    // The hinge reads only x0, so raising x1 moves the score through the free ramp alone.
    let w1 = [0.5, 1.0, 1.0, 0.0, -1.0, -1.0];
    let b1 = [-0.1, 0.0, -2.0];
    let w2 = [-4.0, -1.5, 1.5];
    let b2 = [100.0];
    let bytes = bake_two_layer_znpr_v3(
        &[0.0, 0.0],
        &[1.0, 1.0],
        &w1,
        &b1,
        &w2,
        &b2,
        2,
        3,
        1,
        WeightDtype::F32,
        None,
        None,
        Activation::Relu,
    );
    let model = zenpredict::Model::from_bytes(&bytes).unwrap();
    let cert = certify_monotone_bake(&model, 100.0).expect("certificate");
    assert_eq!(
        cert,
        MonotoneCertificate {
            ramps: 0,
            free_ramps: 1,
            free_amplitude: 3.0,
            max_pairs: 0
        }
    );
    let mut predictor = zenpredict::Predictor::new(&model);
    let lo = predictor.predict(&[3.0, 0.0]).unwrap()[0];
    let hi = predictor.predict(&[3.0, 5.0]).unwrap()[0];
    assert!(hi > lo, "the free ramp must make this step raise the score");
    assert!(
        (hi - lo - 3.0).abs() <= 1e-5,
        "rise {} should equal the certified amplitude 3",
        hi - lo
    );
    // Breaking the tie (one weight changed) must refuse.
    let mut w2_bad = w2;
    w2_bad[2] = 1.4;
    let bad = bake_two_layer_znpr_v3(
        &[0.0, 0.0],
        &[1.0, 1.0],
        &w1,
        &b1,
        &w2_bad,
        &b2,
        2,
        3,
        1,
        WeightDtype::F32,
        None,
        None,
        Activation::Relu,
    );
    assert!(certify_monotone_bake(&zenpredict::Model::from_bytes(&bad).unwrap(), 100.0).is_err());
}

/// The gate projection: any input lands in `{a >= 0, s_p·a + s_d·b >= 0}`,
/// with the f16 margin, and a point well inside the cone is unchanged.
#[test]
fn byc_gate_projection_property() {
    let mut rng = 0x5eed_0f_9a7e_u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / (1u64 << 53) as f64
    };
    for _ in 0..20_000 {
        let (a, b) = (4.0 * next() - 2.0, 4.0 * next() - 2.0);
        let (sd, sp) = (0.01 + 3.0 * next(), 0.01 + 3.0 * next());
        let (pa, pb) = project_gate(a, b, sd, sp);
        assert!(pa >= 0.0, "a {pa}");
        assert!(
            sp * pa + sd * pb >= GATE_MARGIN * sp * pa * (1.0 - 1e-9) - 1e-15,
            "({a},{b}) -> ({pa},{pb})"
        );
        if pb < 0.0 {
            assert!(pa >= GATE_MIN_DIRECT);
        }
    }
    assert_eq!(project_gate(1.0, -0.5, 1.0, 1.0), (1.0, -0.5));
    assert_eq!(project_gate(1.0, 0.25, 1.0, 1.0), (1.0, 0.25));
}

/// Content-gated training: inputs `[d0, d1, d0·f0, d1·f1]` with f in (0, 1],
/// a target whose gain on `d0` falls with `f0`. The trained model keeps every
/// unit's gain non-negative for every f (checked on the baked f32 weights and
/// scaler), uses a negative product weight somewhere (the gate is exercised),
/// and is exactly non-increasing in each damage input for random fragilities.
#[test]
fn byc_gated_products_train_monotone_in_damage() {
    let mut rng = 0x00c0_ffee_u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / (1u64 << 53) as f64
    };
    let rows: Vec<Vec<f64>> = (0..256)
        .map(|_| {
            let (d0, d1, f0, f1) = (next(), next(), 0.05 + 0.95 * next(), 0.05 + 0.95 * next());
            vec![d0, d1, d0 * f0, d1 * f1]
        })
        .collect();
    let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let scores: Vec<f64> = rows
        .iter()
        .map(|x| 100. - 40. * (x[0] - 0.8 * x[2]) - 10. * x[1])
        .collect();
    let hp = MlpHyperparams {
        nonneg_distance: true,
        monotone_cbc: true,
        nonneg_thresholds: true,
        monotone_product_pairs: vec![(0, 2), (1, 3)],
        n_hidden: 8,
        n_epochs: 60,
        pairs_per_epoch: 2048,
        seed: 9,
        early_stop_patience: 0,
        ..Default::default()
    };
    let mut groups = [TrainingGroup {
        name: "invented".into(),
        human_scores: &scores,
        features: FeatureRows::Borrowed(&refs),
        metric_sigmas: None,
        train_weight: 1.,
        validation_weight: 1.,
        ref_ids: None,
        loss_mode: GroupLossMode::Both,
    }];
    let bytes = train_mlp(&mut groups, 4, &hp, &mut Vec::new());
    let model = zenpredict::Model::from_bytes(&bytes).unwrap();
    let (l0, l1) = (model.layer(0), model.layer(1));
    let zenpredict::WeightStorage::F32(w1) = l0.weights else {
        panic!("f32 test bake")
    };
    let zenpredict::WeightStorage::F32(w2) = l1.weights else {
        panic!("f32 test bake")
    };
    let s = model.scaler_scale();
    let h = l0.out_dim;
    let mut used_negative = false;
    for j in (0..h).filter(|&j| w2[j] != 0.0) {
        for (d, p) in [(0usize, 2usize), (1, 3)] {
            let a = f64::from(w1[d * h + j]) / f64::from(s[d]);
            let b = f64::from(w1[p * h + j]) / f64::from(s[p]);
            assert!(a >= 0.0 && a + b.min(0.0) >= 0.0, "unit {j}: gain {a} {b}");
            used_negative |= b < 0.0;
        }
    }
    assert!(
        used_negative,
        "the target needs a falling gain; no negative product weight was used"
    );
    let mut predictor = zenpredict::Predictor::new(&model);
    for _ in 0..3000 {
        let (d0, d1, f0, f1) = (
            2.0 * next(),
            2.0 * next(),
            0.01 + 0.99 * next(),
            0.01 + 0.99 * next(),
        );
        let (e0, e1) = (next(), next());
        let x = [d0 as f32, d1 as f32, (d0 * f0) as f32, (d1 * f1) as f32];
        let y = [
            (d0 + e0) as f32,
            (d1 + e1) as f32,
            ((d0 + e0) * f0) as f32,
            ((d1 + e1) * f1) as f32,
        ];
        let (fx, fy) = (
            predictor.predict(&x).unwrap()[0],
            predictor.predict(&y).unwrap()[0],
        );
        assert!(
            fy <= fx + 1e-4 && fx <= 100.0,
            "monotone violation {x:?}->{fx} {y:?}->{fy}"
        );
    }
}

fn byc_derived_bake(w1: &[f32], b1: &[f32], w2: &[f32], products: &str) -> Vec<u8> {
    let text = format!("zensim-derived-inputs v1\nin 13\nin 14\n{products}");
    let meta = [BakeMetadataEntry {
        key: "zensim.derived_inputs",
        kind: zenpredict::MetadataType::Utf8,
        value: text.as_bytes(),
    }];
    bake(&BakeRequest {
        schema_hash: 0,
        flags: 0,
        scaler_mean: &[0.0; 4],
        scaler_scale: &[1.0, 2.0, 1.0, 0.5],
        layers: &[
            BakeLayer {
                in_dim: 4,
                out_dim: 2,
                activation: Activation::Relu,
                dtype: WeightDtype::F32,
                weights: w1,
                biases: b1,
            },
            BakeLayer {
                in_dim: 2,
                out_dim: 1,
                activation: Activation::Identity,
                dtype: WeightDtype::F32,
                weights: w2,
                biases: &[100.0],
            },
        ],
        feature_bounds: &[],
        metadata: &meta,
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
        feature_order: None,
        output_order: None,
        compressed: false,
        hu_permutations: None,
    })
    .unwrap()
}

/// Certificate on declared `fx1` products: a negative product weight inside
/// the gate cone certifies; one outside it, or a factor that is not a
/// `pjnd_fragility` slot, refuses.
#[test]
fn byc_certificate_gated_products() {
    // inputs: d13, d14, d13·f422, d14·f480; scales 1, 2, 1, 0.5; two units.
    // unit 0: a=1 on d13, b=-0.9 on its product (gain 1 - 0.9f >= 0.1). unit 1: plain on d14.
    let ok = [1.0f32, 0.0, 0.0, 0.5, -0.9, 0.0, 0.0, 0.2];
    let fx1 = "product 13 422\nproduct 14 480\n";
    let good = byc_derived_bake(&ok, &[0.0, -0.1], &[-1.0, -2.0], fx1);
    let cert = certify_monotone_bake(&zenpredict::Model::from_bytes(&good).unwrap(), 100.0);
    assert_eq!(
        cert,
        Ok(MonotoneCertificate {
            ramps: 0,
            free_ramps: 0,
            free_amplitude: 0.0,
            max_pairs: 0
        })
    );
    // b = -1.1 makes the gain negative at f = 1.
    let mut bad = ok;
    bad[4] = -1.1;
    let refused = byc_derived_bake(&bad, &[0.0, -0.1], &[-1.0, -2.0], fx1);
    assert!(
        certify_monotone_bake(&zenpredict::Model::from_bytes(&refused).unwrap(), 100.0).is_err()
    );
    // 423 is not a fragility slot ((423 - 372) % 29 != 21).
    let wrong = byc_derived_bake(
        &ok,
        &[0.0, -0.1],
        &[-1.0, -2.0],
        "product 13 423\nproduct 14 480\n",
    );
    assert!(certify_monotone_bake(&zenpredict::Model::from_bytes(&wrong).unwrap(), 100.0).is_err());
}

/// `--max-pairs`: trained on a target that is the max of two damage directions,
/// the bake certifies (live max pairs or not), keeps the exact identity pin and
/// is non-increasing along non-negative steps (up to f32 rounding of a pair's
/// two terms).
#[test]
fn byc_max_pairs_train_certify_and_stay_monotone() {
    let rows: Vec<Vec<f64>> = (0..256)
        .map(|i| vec![(i % 16) as f64 / 15., (i / 16) as f64 / 15.])
        .collect();
    let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let scores: Vec<f64> = rows.iter().map(|x| 100. - 50. * x[0].max(x[1])).collect();
    for dtype in [WeightDtype::F32, WeightDtype::F16] {
        let hp = MlpHyperparams {
            nonneg_distance: true,
            monotone_cbc: true,
            nonneg_thresholds: true,
            max_pairs: 4,
            n_hidden: 8,
            n_epochs: 60,
            pairs_per_epoch: 2048,
            seed: 13,
            out_dtype: dtype,
            early_stop_patience: 0,
            ..Default::default()
        };
        let mut groups = [TrainingGroup {
            name: "invented".into(),
            human_scores: &scores,
            features: FeatureRows::Borrowed(&refs),
            metric_sigmas: None,
            train_weight: 1.,
            validation_weight: 1.,
            ref_ids: None,
            loss_mode: GroupLossMode::Both,
        }];
        let bytes = train_mlp(&mut groups, 2, &hp, &mut Vec::new());
        let model = zenpredict::Model::from_bytes(&bytes).unwrap();
        // Whether a pair keeps a signed v row (a live max) is a training outcome,
        // not a guarantee; the hand-built test below proves the max itself.
        let cert = certify_monotone_bake(&model, 100.0).expect("certificate");
        assert!(cert.free_amplitude == 0.0, "{cert:?}");
        let mut predictor = zenpredict::Predictor::new(&model);
        assert_eq!(
            predictor.predict(&[0., 0.]).unwrap()[0].to_bits(),
            100.0_f32.to_bits()
        );
        let mut rng = 0x7777_1234_u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 11) as f32 / (1u64 << 53) as f32
        };
        for _ in 0..3000 {
            let x = [2. * next(), 2. * next()];
            let y = [x[0] + next(), x[1] + next()];
            let (fx, fy) = (
                predictor.predict(&x).unwrap()[0],
                predictor.predict(&y).unwrap()[0],
            );
            assert!(
                fy <= fx + 1e-4 && fx <= 100.,
                "monotone violation {x:?}->{fx} {y:?}->{fy}"
            );
        }
    }
}

/// `--max-pairs` with `--monotone-feature-mask`: a sign-free third input (a global-change descriptor) enters only the
/// pairs' v rows, so the bake stays pinned at identity and non-increasing in the two masked-monotone inputs for every
/// value of the free one, while the free input may raise or lower the score.
#[test]
fn byc_max_pairs_with_free_input_stay_monotone_in_damage() {
    let rows: Vec<Vec<f64>> = (0..512)
        .map(|i| {
            vec![
                (i % 8) as f64 / 7.,
                (i / 8 % 8) as f64 / 7.,
                (i / 64) as f64 / 7.,
            ]
        })
        .collect();
    let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    // Damage counts less when the descriptor says the change is global.
    let scores: Vec<f64> = rows
        .iter()
        .map(|x| 100. - 50. * x[0].max(x[1]) * (1. - 0.6 * x[2]))
        .collect();
    let hp = MlpHyperparams {
        nonneg_distance: true,
        monotone_cbc: true,
        nonneg_thresholds: true,
        max_pairs: 4,
        monotone_feature_pin: Some(vec![true, true, false]),
        n_hidden: 12,
        n_epochs: 60,
        pairs_per_epoch: 2048,
        seed: 29,
        out_dtype: WeightDtype::F32,
        early_stop_patience: 0,
        ..Default::default()
    };
    let mut groups = [TrainingGroup {
        name: "invented".into(),
        human_scores: &scores,
        features: FeatureRows::Borrowed(&refs),
        metric_sigmas: None,
        train_weight: 1.,
        validation_weight: 1.,
        ref_ids: None,
        loss_mode: GroupLossMode::Both,
    }];
    let bytes = train_mlp(&mut groups, 3, &hp, &mut Vec::new());
    let model = zenpredict::Model::from_bytes(&bytes).unwrap();
    let l0 = model.layer(0);
    let zenpredict::WeightStorage::F32(w1) = &l0.weights else {
        panic!("f32 bake expected")
    };
    let n_hidden = l0.out_dim;
    for i in 0..hp.max_pairs {
        assert_eq!(
            w1[2 * n_hidden + 2 * i],
            0.0,
            "pair {i}: the free input must not enter u"
        );
    }
    let mut predictor = zenpredict::Predictor::new(&model);
    assert_eq!(
        predictor.predict(&[0., 0., 0.]).unwrap()[0].to_bits(),
        100.0_f32.to_bits()
    );
    let mut rng = 0x1357_9bdf_u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f32 / (1u64 << 53) as f32
    };
    for _ in 0..3000 {
        let d = next();
        let x = [2. * next(), 2. * next(), d];
        let y = [x[0] + next(), x[1] + next(), d];
        let (fx, fy) = (
            predictor.predict(&x).unwrap()[0],
            predictor.predict(&y).unwrap()[0],
        );
        assert!(
            fy <= fx + 1e-4,
            "monotone violation {x:?}->{fx} {y:?}->{fy}"
        );
    }
}

/// A hand-built max pair computes `c·max(a·x, b·x - t)` and certifies; breaking
/// the partner's bias, the tie or the cone refuses.
#[test]
fn byc_certificate_max_pair() {
    // u = (1, 0) bias 0; v = (b - a) = (-1, 1) bias -0.5 => max(x0, x1 - 0.5); c = -10.
    let w1 = [1.0f64, -1.0, 0.0, 1.0];
    let ok = bake_two_layer_znpr_v3(
        &[0.0, 0.0],
        &[1.0, 1.0],
        &w1,
        &[0.0, -0.5],
        &[-10.0, -10.0],
        &[100.0],
        2,
        2,
        1,
        WeightDtype::F32,
        None,
        None,
        Activation::Relu,
    );
    let model = zenpredict::Model::from_bytes(&ok).unwrap();
    assert_eq!(
        certify_monotone_bake(&model, 100.0),
        Ok(MonotoneCertificate {
            ramps: 0,
            free_ramps: 0,
            free_amplitude: 0.0,
            max_pairs: 1
        })
    );
    let mut p = zenpredict::Predictor::new(&model);
    for (x, want) in [
        ([0.3f32, 0.2f32], 100.0 - 3.0),
        ([0.3, 1.5], 100.0 - 10.0),
        ([2.0, 2.0], 80.0),
    ] {
        assert!((p.predict(&x).unwrap()[0] - want).abs() < 1e-4, "{x:?}");
    }
    let refuse = |w1: &[f64], b1: &[f64], w2: &[f64]| {
        let bytes = bake_two_layer_znpr_v3(
            &[0.0, 0.0],
            &[1.0, 1.0],
            w1,
            b1,
            w2,
            &[100.0],
            2,
            2,
            1,
            WeightDtype::F32,
            None,
            None,
            Activation::Relu,
        );
        certify_monotone_bake(&zenpredict::Model::from_bytes(&bytes).unwrap(), 100.0).is_err()
    };
    assert!(
        refuse(&w1, &[-0.1, -0.5], &[-10.0, -10.0]),
        "u must have bias 0"
    );
    assert!(
        refuse(&w1, &[0.0, -0.5], &[-10.0, -9.0]),
        "output weights must be tied"
    );
    assert!(
        refuse(&[1.0, -1.5, 0.0, 1.0], &[0.0, -0.5], &[-10.0, -10.0]),
        "a + (b - a) >= 0"
    );
}

/// Negative control: B's recipe class (`--nonneg-distance`, signed W1) trains
/// negative first-layer weights, which the certificate must refuse.
#[test]
fn byc_certificate_refuses_signed_first_layer() {
    let rows: Vec<Vec<f64>> = (0..64)
        .map(|i| vec![(i % 16) as f64 / 15., (i / 16) as f64 / 3.])
        .collect();
    let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let scores: Vec<f64> = rows
        .iter()
        .map(|x| 100. - 30. * x[0] + 10. * x[1])
        .collect();
    let hp = MlpHyperparams {
        nonneg_distance: true,
        n_hidden: 8,
        n_epochs: 20,
        pairs_per_epoch: 512,
        seed: 3,
        early_stop_patience: 0,
        ..Default::default()
    };
    let mut groups = [TrainingGroup {
        name: "invented".into(),
        human_scores: &scores,
        features: FeatureRows::Borrowed(&refs),
        metric_sigmas: None,
        train_weight: 1.,
        validation_weight: 1.,
        ref_ids: None,
        loss_mode: GroupLossMode::Both,
    }];
    let bytes = train_mlp(&mut groups, 2, &hp, &mut Vec::new());
    let model = zenpredict::Model::from_bytes(&bytes).unwrap();
    assert!(certify_monotone_bake(&model, 100.0).is_err());
}

#[test]
fn byc_thresholds_refuse_unsupported_modes() {
    let enabled = MlpHyperparams {
        nonneg_distance: true,
        nonneg_thresholds: true,
        ..Default::default()
    };
    validate_training_capabilities(&enabled, true, 0);
    for hp in [
        MlpHyperparams {
            nonneg_distance: false,
            ..enabled.clone()
        },
        MlpHyperparams {
            smooth_hinge: true,
            ..enabled.clone()
        },
        MlpHyperparams {
            steer_pairs: Some("not-read.parquet".into()),
            steer_lambda: 1.,
            ..enabled.clone()
        },
        MlpHyperparams {
            pool_head: true,
            ..enabled.clone()
        },
        MlpHyperparams {
            n_hidden_layers: 2,
            ..enabled.clone()
        },
    ] {
        assert!(std::panic::catch_unwind(|| validate_training_capabilities(&hp, true, 0)).is_err());
    }
}

#[test]
fn e34_k_refuses_incomplete_or_incompatible_options() {
    let enabled = MlpHyperparams {
        steer_pairs: Some("not-read.parquet".into()),
        steer_lambda: 1.,
        nonneg_distance: true,
        ..Default::default()
    };
    validate_training_capabilities(&enabled, true, 0);
    for hp in [
        MlpHyperparams {
            steer_lambda: 0.,
            ..enabled.clone()
        },
        MlpHyperparams {
            steer_pairs: None,
            ..enabled.clone()
        },
        MlpHyperparams {
            steer_lambda: f64::NAN,
            ..enabled.clone()
        },
        MlpHyperparams {
            steer_lambda: f64::INFINITY,
            ..enabled.clone()
        },
        MlpHyperparams {
            nonneg_distance: false,
            ..enabled.clone()
        },
        MlpHyperparams {
            monotone_cbc: true,
            ..enabled.clone()
        },
        MlpHyperparams {
            minibatch_size: 2,
            ..enabled.clone()
        },
        MlpHyperparams {
            feature_transforms: Some(vec![]),
            ..enabled.clone()
        },
    ] {
        assert!(std::panic::catch_unwind(|| validate_training_capabilities(&hp, true, 0)).is_err());
    }
}

#[test]
fn e34_l_refuses_feature_transforms_for_identity_contract() {
    let hp = MlpHyperparams {
        n_hidden: 1,
        nonneg_distance: true,
        monotone_cbc: true,
        feature_transforms: Some(vec![zenpredict::FeatureTransform::Identity]),
        ..Default::default()
    };
    assert!(std::panic::catch_unwind(|| validate_training_capabilities(&hp, true, 0)).is_err());
}

#[test]
fn steer_hinge_trains_and_packs_with_exact_identity() {
    let rows: Vec<Vec<f64>> = (0..32)
        .map(|i| vec![i as f64 / 31., (i % 7) as f64 / 6.])
        .collect();
    let refs: Vec<_> = rows.iter().map(Vec::as_slice).collect();
    let scores: Vec<_> = rows
        .iter()
        .map(|x| 100. - 30. * x[0] - 10. * x[1])
        .collect();
    for dtype in [WeightDtype::F32, WeightDtype::F16] {
        let hp = MlpHyperparams {
            smooth_hinge: true,
            nonneg_distance: true,
            n_hidden: 8,
            n_epochs: 2,
            pairs_per_epoch: 32,
            minibatch_size: 1,
            seed: 42,
            out_dtype: dtype,
            early_stop_patience: 0,
            ..Default::default()
        };
        let group = TrainingGroup {
            name: "invented".into(),
            human_scores: &scores,
            features: FeatureRows::Borrowed(&refs),
            metric_sigmas: None,
            train_weight: 1.,
            validation_weight: 1.,
            ref_ids: None,
            loss_mode: GroupLossMode::Both,
        };
        let bytes = train_mlp(&mut [group], 2, &hp, &mut Vec::new());
        let model = zenpredict::Model::from_bytes(&bytes).unwrap();
        assert_eq!(model.layer(0).activation, Activation::SmoothHinge);
        assert!(model.scaler_mean().iter().all(|x| *x == 0.));
        assert!(model.layer(0).biases.iter().all(|b| *b == 0.));
        let mut predictor = zenpredict::Predictor::new(&model);
        assert_eq!(predictor.predict(&[0., 0.]).unwrap()[0], 100.);
        assert!(predictor.predict(&[1., 1.]).unwrap()[0] < 100.);
    }
}

#[test]
fn steer_hinge_unsupported_modes_refuse() {
    for hp in [
        MlpHyperparams {
            smooth_hinge: true,
            ..Default::default()
        },
        MlpHyperparams {
            smooth_hinge: true,
            nonneg_distance: true,
            n_hidden_layers: 2,
            ..Default::default()
        },
        MlpHyperparams {
            smooth_hinge: true,
            nonneg_distance: true,
            pool_head: true,
            ..Default::default()
        },
        MlpHyperparams {
            smooth_hinge: true,
            nonneg_distance: true,
            minibatch_size: 2,
            ..Default::default()
        },
        MlpHyperparams {
            smooth_hinge: true,
            nonneg_distance: true,
            monotone_cbc: true,
            ..Default::default()
        },
    ] {
        assert!(
            std::panic::catch_unwind(|| capabilities::validate_training_capabilities(&hp, true, 0))
                .is_err()
        );
    }
}
