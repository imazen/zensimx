//! The model's raw output has ONE sign convention per run, and every loss term
//! must speak it.
//!
//! Before 2026-09-06 the trainer derived that convention at exactly one of its
//! polarity-sensitive sites (`train_mlp_strategy`'s sequential RankNet term) and
//! nowhere else. The pool head, the hybrid head, the per-sample-α head and both
//! plain-path mini-batch helpers each recomputed a bare `signum(mos_a − mos_b)`;
//! the TV within-ladder hinge and the α head's monotonicity hinge each hard-coded
//! an *opposite* assumption in prose. On the α-head path the result is a model
//! whose ordering is intact and whose sign is backwards — the fastclass2 campaign
//! measured its 2-layer α-head arm as the campaign's single best CID22 *ordering*,
//! `|−0.8921|` against the plain path's `+0.8863`, arriving as a negative number
//! that `bake_dial_refit pack` then could not spline at all (the output
//! calibration spline is monotone increasing by construction).
//!
//! These tests pin the invariant that closes it: **whatever convention a run
//! declares, every path obeys it.** Each arm trains a tiny synthetic corpus whose
//! target is a known monotone function of the features, then reads the trained
//! bake back through the same runtime `bake_verdict` scores with, and asserts the
//! SIGN of `corr(raw_output, quality)`.
//!
//! Four of the five arms that compile at the parent commit fail there. (This
//! file imports `OutputPolarity`, which the parent does not have, so the control
//! was run as a stripped variant with that import and `output_polarity_owner_
//! maps_conventions_to_signs` removed.)

use zenpredict::Model;
use zensim::BakeScorer;
use zensim_validate::mlp_train::{
    FeatureRows, GroupLossMode, MlpHyperparams, OutputPolarity, TrainingGroup, TvRegularizer,
    ValidationPolicy, train_mlp_strategy,
};

/// Deterministic synthetic corpus: `quality = w · x + 0.1·x0² + noise`, so a
/// model with any capacity at all can order it. Returns `(features, quality)`.
fn synthetic(n_features: usize, n_rows: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
    // A tiny self-contained SplitMix64 — the trainer's RNG is crate-internal and
    // this corpus only has to be deterministic, not identical to any run's.
    let mut state = seed;
    let mut next = move || -> f64 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // uniform in [0,1) then shifted to roughly N(0,1)-ish spread
        let u = (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        (u - 0.5) * 3.0
    };
    let w: Vec<f64> = (0..n_features)
        .map(|i| (i as f64 - n_features as f64 / 2.0) * 0.3)
        .collect();
    let mut feats = Vec::with_capacity(n_rows);
    let mut quality = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        let x: Vec<f64> = (0..n_features).map(|_| next()).collect();
        let mut y: f64 = x.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        y += 0.1 * x[0] * x[0];
        y += next() * 0.02;
        feats.push(x);
        quality.push(y);
    }
    // Rescale quality into [0, 1] so an absolute term has a sane target range.
    let lo = quality.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = quality.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    for q in quality.iter_mut() {
        *q = (*q - lo) / (hi - lo);
    }
    (feats, quality)
}

/// SROCC through THE owner (`zenstats`, re-exported as `zensim_validate::panel`).
///
/// The first draft hand-rolled a midrank Spearman here. That is the eleventh
/// private copy in this crate and exactly what the repo's NO-DUPLICATE-
/// IMPLEMENTATIONS rule names — "a second **Rust** site is just as much a
/// duplicate as a Python one" — in a lane whose whole framing is single
/// ownership. Caught by review 2026-09-06.
fn spearman(a: &[f64], b: &[f64]) -> f64 {
    // `compute_panel().srocc` is `.abs()` by the panel's documented
    // polarity-tolerant convention — useless here, since the SIGN is the whole
    // measurement. `zenstats::panel::spearman` is the signed owner.
    zensim_validate::panel::spearman(a, b)
}

/// Train one arm and return `corr(raw_output, quality)` read back through the
/// SAME runtime `bake_verdict` scores with — heads included.
fn raw_output_correlation(hyper: MlpHyperparams, loss_mode: GroupLossMode) -> f64 {
    let n_features = 12;
    let (feats, quality) = synthetic(n_features, 240, 20260906);
    let feats_ref: Vec<&[f64]> = feats.iter().map(|v| v.as_slice()).collect();
    let mut groups = [TrainingGroup {
        name: "synth".to_string(),
        human_scores: &quality,
        features: FeatureRows::Borrowed(&feats_ref),
        metric_sigmas: None,
        train_weight: 1.0,
        validation_weight: 1.0,
        ref_ids: None,
        loss_mode,
    }];
    let mut log = Vec::new();
    let bytes = train_mlp_strategy(
        &mut groups,
        n_features,
        &hyper,
        &mut log,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    let model = Model::from_bytes(leaked).expect("bake loads");
    let mut scorer = BakeScorer::new(&model).unwrap();
    let preds: Vec<f64> = feats
        .iter()
        .map(|row| scorer.score_features(row, 0, 0, None).unwrap())
        .collect();
    spearman(&preds, &quality)
}

fn base_hyper() -> MlpHyperparams {
    MlpHyperparams {
        n_hidden: 12,
        n_epochs: 60,
        pairs_per_epoch: 1200,
        initial_lr: 0.005,
        log_every: 1000,
        early_stop_patience: 0,
        validation_policy: ValidationPolicy::Mean,
        ..Default::default()
    }
}

/// The owner's decision table, independent of any training.
#[test]
fn output_polarity_owner_maps_conventions_to_signs() {
    assert_eq!(OutputPolarity::Distance.rank_target_sign(), 1.0);
    assert_eq!(OutputPolarity::Score.rank_target_sign(), -1.0);
    assert_eq!(OutputPolarity::Distance.ladder_sign(), 1.0);
    assert_eq!(OutputPolarity::Score.ladder_sign(), -1.0);
    assert!(OutputPolarity::Distance.label().contains("LOWER"));
    assert!(OutputPolarity::Score.label().contains("HIGHER"));
}

/// The auxiliary pair constrains an axis absent from the main training rows.
/// Swapping its endpoints must reverse the served preference under BOTH output
/// conventions. This exercises the actual plain-path hinge and exporter, not
/// just the polarity multiplier or a change in bake bytes.
#[test]
fn plain_tv_pair_order_controls_the_served_preference_in_both_polarities() {
    let main_rows: [&[f64]; 2] = [&[-1.0, 0.0], &[1.0, 0.0]];
    let targets = [0.0, 1.0];
    let endpoints = vec![vec![0.0, -1.0], vec![0.0, 1.0]];
    for loss_mode in [GroupLossMode::Rank, GroupLossMode::Both] {
        for reverse in [false, true] {
            let hyper = MlpHyperparams {
                n_hidden: 12,
                n_epochs: 20,
                pairs_per_epoch: 400,
                initial_lr: 0.01,
                mse_weight: if loss_mode == GroupLossMode::Both {
                    1.0
                } else {
                    0.0
                },
                early_stop_patience: 0,
                seed: 7101,
                ..Default::default()
            };
            let mut groups = [TrainingGroup {
                name: "synthetic-main-axis".into(),
                human_scores: &targets,
                features: FeatureRows::Borrowed(&main_rows),
                metric_sigmas: None,
                train_weight: 1.0,
                validation_weight: 0.0,
                ref_ids: None,
                loss_mode,
            }];
            let polarity = OutputPolarity::for_groups(&groups, &hyper);
            let tv = TvRegularizer {
                pairs: vec![if reverse { (1, 0) } else { (0, 1) }],
                features: endpoints.clone(),
                weight: 0.5,
                apply_every: 1,
                batch: 1,
                band_id: None,
                band_weights: None,
                margin: 0.5,
                pool: false,
                level: None,
            };
            let bytes = train_mlp_strategy(
                &mut groups,
                2,
                &hyper,
                &mut Vec::new(),
                Some(&tv),
                None,
                None,
                None,
                None,
                None,
            );
            let model = Model::from_bytes(&bytes).expect("trained bake loads");
            let mut scorer = BakeScorer::new(&model).unwrap();
            let low = scorer.score_features(&endpoints[0], 0, 0, None).unwrap();
            let high = scorer.score_features(&endpoints[1], 0, 0, None).unwrap();
            let oriented_gap =
                -polarity.ladder_sign() * (high - low) * if reverse { -1.0 } else { 1.0 };
            assert!(
                oriented_gap > 0.1,
                "{loss_mode:?}, reverse={reverse}: low={low}, high={high}, gap={oriented_gap}"
            );
        }
    }
}

/// Rank-only recipes are DISTANCE-shaped on every path. This is the legacy
/// convention every board bake trained under, and the arm that proves the fix
/// did not move it.
#[test]
fn rank_only_recipes_are_distance_shaped_on_every_path() {
    // (label, hyperparams)
    let arms: Vec<(&str, MlpHyperparams)> = vec![
        ("plain K=1", base_hyper()),
        (
            "plain K=32 (run_parallel_minibatch)",
            MlpHyperparams {
                minibatch_size: 32,
                ..base_hyper()
            },
        ),
        (
            "α head depth 1",
            MlpHyperparams {
                per_sample_alpha_head: true,
                ..base_hyper()
            },
        ),
        (
            "α head depth 2",
            MlpHyperparams {
                per_sample_alpha_head: true,
                n_hidden_layers: 2,
                ..base_hyper()
            },
        ),
        (
            "hybrid head",
            MlpHyperparams {
                hybrid_head: true,
                ..base_hyper()
            },
        ),
    ];
    for (label, hyper) in arms {
        let r = raw_output_correlation(hyper, GroupLossMode::Rank);
        assert!(
            r < -0.5,
            "{label}: a rank-only recipe is DISTANCE-shaped, so the raw output must \
             be NEGATIVELY correlated with quality; got SROCC {r:.4}"
        );
    }
}

/// An absolute term makes the run SCORE-shaped, and the rank term must follow.
///
/// The α-head arms are the ones that fail on the parent commit: that path never
/// applied `rank_target_sign`, so its RankNet term stayed DISTANCE-shaped while
/// its MSE term regressed the same output onto a score-unit target. Depth 2 is
/// where the campaign measured it (`raw CID22 −0.8921`).
#[test]
fn absolute_term_makes_every_path_score_shaped() {
    let arms: Vec<(&str, MlpHyperparams, GroupLossMode)> = vec![
        (
            "plain K=1 :both",
            MlpHyperparams {
                mse_weight: 1.0,
                ..base_hyper()
            },
            GroupLossMode::Both,
        ),
        (
            "α head depth 1 --mse-weight",
            MlpHyperparams {
                per_sample_alpha_head: true,
                mse_weight: 1.0,
                ..base_hyper()
            },
            GroupLossMode::Rank,
        ),
        (
            "α head depth 2 --mse-weight",
            MlpHyperparams {
                per_sample_alpha_head: true,
                n_hidden_layers: 2,
                mse_weight: 1.0,
                ..base_hyper()
            },
            GroupLossMode::Rank,
        ),
    ];
    for (label, hyper, mode) in arms {
        let r = raw_output_correlation(hyper, mode);
        assert!(
            r > 0.5,
            "{label}: an absolute term makes the run SCORE-shaped, so the raw output \
             must be POSITIVELY correlated with quality; got SROCC {r:.4}. A large \
             NEGATIVE value here is the defect this test exists for: the ordering is \
             learned and the sign is backwards."
        );
    }
}

/// The brief's requested proof, stated in the codebase's own convention: the α
/// head must carry the SAME polarity as the plain path at the same settings, at
/// both depths.
#[test]
fn alpha_head_raw_output_polarity_matches_plain_path_at_depth_1_and_2() {
    for (label, mse_weight, mode, want_positive) in [
        ("rank-only", 0.0, GroupLossMode::Rank, false),
        ("absolute", 1.0, GroupLossMode::Both, true),
    ] {
        let plain = raw_output_correlation(
            MlpHyperparams {
                mse_weight,
                ..base_hyper()
            },
            mode,
        );
        for depth in [1usize, 2usize] {
            let alpha = raw_output_correlation(
                MlpHyperparams {
                    per_sample_alpha_head: true,
                    n_hidden_layers: depth,
                    mse_weight,
                    ..base_hyper()
                },
                mode,
            );
            assert_eq!(
                plain > 0.0,
                alpha > 0.0,
                "{label}: α head at depth {depth} disagrees with the plain path about \
                 which way the raw output points — plain {plain:.4}, α {alpha:.4}"
            );
            assert_eq!(
                alpha > 0.0,
                want_positive,
                "{label}: α head at depth {depth} carries the wrong convention \
                 (SROCC {alpha:.4})"
            );
        }
    }
}

/// `run_parallel_minibatch` and `run_minibatch_with_nin` implement RankNet only.
/// Routing an absolute term through them silently discarded it; refuse instead.
#[test]
#[should_panic(expected = "absolute (mse/both) term is active")]
fn absolute_term_with_parallel_minibatch_is_refused() {
    raw_output_correlation(
        MlpHyperparams {
            mse_weight: 1.0,
            minibatch_size: 32,
            ..base_hyper()
        },
        GroupLossMode::Both,
    );
}

#[test]
#[should_panic(expected = "absolute (mse/both) term is active")]
fn absolute_term_with_norm_in_norm_is_refused() {
    raw_output_correlation(
        MlpHyperparams {
            mse_weight: 1.0,
            minibatch_size: 16,
            norm_in_norm_weight: 0.1,
            ..base_hyper()
        },
        GroupLossMode::Both,
    );
}
