//! Exact ties for the separately admitted neutral leg. Legacy groups retain
//! their inherited rank-target behavior. No target magnitude cutoff is used.
use super::{GroupLossMode, TrainingGroup};

pub(super) fn tied(group: &TrainingGroup<'_>, a: f64, b: f64, legacy_target: f64) -> bool {
    if group.name == "subjective_rank" || group.name.starts_with("subjective_rank_") {
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
