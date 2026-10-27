//! Private MlicOrder protocol. Admission binds codecs, keys and matching model;
//! this owner preserves the base draw's group, reference and collision outcome.
use std::collections::{BTreeMap, BTreeSet};

use super::{Draw, SplitMix64, Stratum, coprime_step};
use crate::mlp_train::{GroupLossMode, MlpHyperparams, PairSampling, TrainingGroup};

pub(crate) const GROUP: &str = "human_codec_strata";

pub(crate) fn validate(
    groups: &[TrainingGroup<'_>],
    hp: &MlpHyperparams,
    lists: &BTreeMap<String, Vec<[usize; 2]>>,
) {
    for group in groups.iter().filter(|g| g.name == GROUP) {
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
                && hp.dro_eta == 0.,
            "matched codecs require the registered plain sequential uniform trainer"
        );
        assert_eq!(group.loss_mode, GroupLossMode::Rank);
        assert_eq!(group.validation_weight, 0.);
        assert!(group.train_weight > 0. && group.train_weight.is_finite());
        let refs = group.ref_ids.expect("matched codecs require references");
        assert_eq!(refs.len(), group.human_scores.len());
        assert!(
            lists.contains_key(GROUP),
            "matched codecs require an admitted list"
        );
        assert!(
            !lists[GROUP].is_empty(),
            "matched codecs require eligible pairs"
        );
    }
}

struct Schedule {
    pairs: Vec<[usize; 2]>,
    walk: Stratum,
    visits: u64,
}

pub(crate) struct Plan {
    refs: Vec<u32>,
    schedules: BTreeMap<u32, Schedule>,
}

impl Plan {
    pub(crate) fn new(refs: &[u32], pairs: &[[usize; 2]], sample_seed: u64) -> Self {
        assert!(!pairs.is_empty(), "matched codecs require eligible pairs");
        let mut by_ref = BTreeMap::<u32, Vec<[usize; 2]>>::new();
        let mut seen = BTreeSet::new();
        for &[a, b] in pairs {
            assert!(
                a < refs.len() && b < refs.len() && a != b,
                "invalid matched endpoints"
            );
            assert_eq!(refs[a], refs[b], "cross-reference matched pair");
            assert!(seen.insert([a.min(b), a.max(b)]), "duplicate matched pair");
            by_ref.entry(refs[a]).or_default().push([a, b]);
        }
        let schedules = by_ref
            .into_iter()
            .map(|(reference, pairs)| {
                // A private stream: never draw from the trainer's sampler RNG.
                let mut seed =
                    SplitMix64::new(sample_seed ^ 0x6d63_6f64_6563_0000 ^ u64::from(reference));
                let walk = Stratum {
                    train_pos: 0,
                    rows: (0..pairs.len()).collect(),
                    step: coprime_step(pairs.len(), seed.next_u64()),
                    offset: (seed.next_u64() % pairs.len() as u64) as usize,
                };
                (
                    reference,
                    Schedule {
                        pairs,
                        walk,
                        visits: 0,
                    },
                )
            })
            .collect();
        Self {
            refs: refs.to_vec(),
            schedules,
        }
    }

    fn apply(&mut self, draw: Draw) -> Draw {
        let Draw::Pair { train_pos, ia, ib } = draw else {
            return draw; // Collisions never advance an eligible-reference visit.
        };
        assert_eq!(self.refs[ia], self.refs[ib], "base draw crossed references");
        let Some(schedule) = self.schedules.get_mut(&self.refs[ia]) else {
            return draw;
        };
        let visit = schedule.visits;
        schedule.visits = visit.checked_add(1).expect("matched visit overflow");
        if visit % 4 != 0 {
            return draw;
        }
        let replacement = visit / 4;
        let [mut ia, mut ib] = schedule.pairs[schedule.walk.row_at(replacement)];
        // Reverse each pair on its next cycle, including even-length lists.
        if (replacement / schedule.pairs.len() as u64) % 2 != 0 {
            std::mem::swap(&mut ia, &mut ib);
        }
        Draw::Pair { train_pos, ia, ib }
    }
}

pub(crate) fn draw(draw: Draw, plans: &mut [Option<Plan>]) -> Draw {
    let Draw::Pair { train_pos, .. } = draw else {
        return draw;
    };
    plans[train_pos].as_mut().map_or(draw, |p| p.apply(draw))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlp_train::sampling::{PairDrawCtx, RefBuckets, draw_pair, sample_stream_seed};

    #[test]
    fn preserves_actual_base_rng_groups_references_and_collisions() {
        let refs = [0, 0, 0, 0, 1, 1, 2, 2, 2];
        let buckets = [RefBuckets::build(&refs), RefBuckets::build(&refs)];
        let context = PairDrawCtx {
            cdf: &[0.7, 1.],
            row_counts: &[9, 9],
            per_row_cdfs: &[None, None],
            ref_buckets: &buckets,
            strat_bands: &[],
            plan: None,
            draw_index: 0,
        };
        for seed in [0, 17103, 510011] {
            let mut baseline = SplitMix64::new(sample_stream_seed(seed));
            let mut candidate = SplitMix64::new(sample_stream_seed(seed));
            let mut plans = [
                Some(Plan::new(&refs, &[[0, 2], [1, 3], [6, 8]], seed)),
                None,
            ];
            let mut visits = BTreeMap::<u32, usize>::new();
            let mut replacements = 0;
            let mut collisions = 0;
            for _ in 0..100_000 {
                let expected = draw_pair(&context, &mut baseline);
                let actual = draw(draw_pair(&context, &mut candidate), &mut plans);
                if let Draw::Pair {
                    train_pos: 0, ia, ..
                } = expected
                    && refs[ia] != 1
                {
                    let Draw::Pair {
                        train_pos,
                        ia: a,
                        ib: b,
                    } = actual
                    else {
                        panic!("lost pair")
                    };
                    assert_eq!(train_pos, 0);
                    assert_eq!((refs[a], refs[b]), (refs[ia], refs[ia]));
                    let visit = visits.entry(refs[ia]).or_default();
                    if *visit % 4 == 0 {
                        assert!([[0, 2], [1, 3], [6, 8]].contains(&[a.min(b), a.max(b)]));
                        replacements += 1;
                    } else {
                        assert_eq!(actual, expected);
                    }
                    *visit += 1;
                } else {
                    assert_eq!(actual, expected);
                    if matches!(expected, Draw::SameRow { .. }) {
                        collisions += 1;
                    }
                }
            }
            assert!(replacements > 1000 && collisions > 1000);
            assert_eq!(candidate.next_u64(), baseline.next_u64());
        }
    }

    #[test]
    fn cycles_every_match_in_both_orientations_without_counting_collisions() {
        let pairs = [[0, 3], [1, 4], [2, 5], [0, 5]];
        let mut plan = Plan::new(&[7; 6], &pairs, 17103);
        let base = Draw::Pair {
            train_pos: 0,
            ia: 1,
            ib: 2,
        };
        let collision = Draw::SameRow {
            train_pos: 0,
            row: 0,
        };
        let mut selected = Vec::new();
        for visit in 0..32 {
            assert_eq!(plan.apply(collision), collision);
            let actual = plan.apply(base);
            if visit % 4 == 0 {
                let Draw::Pair { ia, ib, .. } = actual else {
                    panic!("lost pair")
                };
                selected.push([ia, ib]);
            } else {
                assert_eq!(actual, base);
            }
        }
        assert_eq!(
            selected[..4].iter().copied().collect::<BTreeSet<_>>(),
            pairs.into_iter().collect()
        );
        for i in 0..4 {
            assert_eq!(selected[i], [selected[i + 4][1], selected[i + 4][0]]);
        }
    }

    #[test]
    fn refuses_invalid_or_duplicate_endpoints() {
        for pairs in [
            vec![[0, 0]],
            vec![[0, 8]],
            vec![[0, 2]],
            vec![[0, 1], [1, 0]],
        ] {
            assert!(std::panic::catch_unwind(|| Plan::new(&[0, 0, 1, 1], &pairs, 0)).is_err());
        }
    }
}
