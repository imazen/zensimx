//! Exact ties for the separately admitted neutral leg. Legacy groups retain
//! their inherited rank-target behavior. No target magnitude cutoff is used.
use super::{GroupLossMode, MlpHyperparams, PairSampling, TrainingGroup};

pub(super) const VOTE_GROUP: &str = "subjective_vote_bapps";

// This private protocol is admitted separately by the CLI. Keep unsupported
// trainer routes fail-closed, rather than accidentally applying hard RankNet.
pub(super) fn validate_votes(
    groups: &[TrainingGroup<'_>],
    hp: &MlpHyperparams,
    has_pair_lists: bool,
) {
    for group in groups
        .iter()
        .filter(|g| g.name.starts_with("subjective_vote_"))
    {
        assert_eq!(group.name, VOTE_GROUP, "unregistered vote group");
        assert!(
            !hp.per_sample_alpha_head
                && !hp.hybrid_head
                && !hp.pool_head
                && !(hp.parallel_batch && hp.minibatch_size > 1)
                && hp.norm_in_norm_weight == 0.
                && hp.pooled_legs.is_empty()
                && !hp.pwrc_pair_weight
                && hp.hard_pair_frac == 0.
                && hp.stratified_bands == 0
                && hp.pair_sampling == PairSampling::Uniform
                && hp.dro_eta == 0.
                && !has_pair_lists,
            "native votes require the registered plain sequential uniform trainer"
        );
        assert_eq!(group.loss_mode, GroupLossMode::Rank);
        assert_eq!(
            group.validation_weight, 0.,
            "vote probabilities are fit-only"
        );
        let refs = group
            .ref_ids
            .expect("native votes require pair-local references");
        assert_eq!(refs.len(), group.human_scores.len());
        let mut pairs = std::collections::BTreeMap::<u32, Vec<f64>>::new();
        for (&reference, &p) in refs.iter().zip(group.human_scores) {
            assert!(matches!(p, 0. | 0.5 | 1.), "two-vote probability required");
            pairs.entry(reference).or_default().push(p);
        }
        assert!(!pairs.is_empty(), "native votes require comparisons");
        assert!(
            pairs.values().all(|p| p.len() == 2 && p[0] + p[1] == 1.),
            "native votes require two complementary sides per comparison"
        );
    }
}

/// Normalized binomial NLL and derivative with respect to raw (y_b-y_a).
/// p is the observed share preferring a. The existing polarity owner maps
/// raw scores to distance differences; score-shaped models require sign -1.
pub(super) fn vote_loss(p: f64, raw_diff: f64, rank_sign: f64) -> (f64, f64) {
    let d = rank_sign * raw_diff;
    let (loss, probability) = if d >= 0. {
        let e = (-d).exp();
        ((1. - p) * d + e.ln_1p(), 1. / (1. + e))
    } else {
        let e = d.exp();
        (-p * d + e.ln_1p(), e / (1. + e))
    };
    (loss, rank_sign * (probability - p))
}

pub(super) fn tied(group: &TrainingGroup<'_>, a: f64, b: f64, legacy_target: f64) -> bool {
    if group.name == VOTE_GROUP {
        false // A one-to-one disagreement still supplies a likelihood gradient.
    } else if group.name == "subjective_rank" || group.name.starts_with("subjective_rank_") {
        assert_eq!(group.loss_mode, GroupLossMode::Rank);
        assert!(
            group.ref_ids.is_some(),
            "subjective rank requires source pairing"
        );
        a == b // Includes both signs of zero; signum() does not detect ties.
    } else {
        legacy_target == 0.
    }
}

#[cfg(test)]
mod tests {
    use super::vote_loss;

    #[test]
    fn native_vote_loss_matches_two_bernoulli_observations_and_derivative() {
        for p in [0., 0.5, 1.] {
            for raw in [-4., -0.1, 0., 0.1, 4.] {
                for sign in [-1., 1.] {
                    let d: f64 = sign * raw;
                    let probability = 1. / (1. + (-d).exp());
                    let expected = -(p * probability.ln() + (1. - p) * (1. - probability).ln());
                    let (loss, gradient) = vote_loss(p, raw, sign);
                    assert!((loss - expected).abs() < 2e-14);
                    let numerical = (vote_loss(p, raw + 1e-5, sign).0
                        - vote_loss(p, raw - 1e-5, sign).0)
                        / 2e-5;
                    assert!((gradient - numerical).abs() < 1e-9);
                    let swapped = vote_loss(1. - p, -raw, sign);
                    assert!((loss - swapped.0).abs() < 1e-14);
                    assert!((gradient + swapped.1).abs() < 1e-14);
                }
            }
        }
    }

    #[test]
    fn disagreement_pulls_predictions_together_and_extreme_logits_stay_finite() {
        for sign in [-1., 1.] {
            for raw in [-1000., -2., 0., 2., 1000.] {
                for p in [0., 0.5, 1.] {
                    let (loss, gradient) = vote_loss(p, raw, sign);
                    assert!(loss.is_finite() && gradient.is_finite());
                    if p == 0.5 && raw != 0. {
                        assert!(raw * gradient > 0.);
                    }
                }
            }
        }
        assert_eq!(vote_loss(0.5, 0., -1.), (2_f64.ln(), -0.));
    }
}
