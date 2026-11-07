//! Private controls for the registered monotone-head training study.
//! Pair generation remains owned by sampling::draw_pair. Ordering is a permutation.
use super::{MlpHyperparams, SplitMix64, TrainingGroup, sampling};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(super) enum Order {
    Random,
    Easy,
    Hard,
    Family,
    TeacherWarm,
    WithinPopulationEasy,
    WithinPopulationHard,
    WithinPopulationFamily,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(super) enum Schedule {
    Cosine,
    Constant,
    Consolidate,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(super) enum Balance {
    None,
    Family,
    SourceFamily,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Group {
    pub name: String,
    pub families: Vec<u32>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Config {
    pub schema: String,
    pub init_scale: f64,
    pub bias_scale: f64,
    pub dropout: f64,
    pub dropout_seed: u64,
    pub adamw: f64,
    pub schedule: Schedule,
    pub order: Order,
    pub teacher_mass: f64,
    #[serde(default, skip_serializing_if = "is_false")]
    pub human_only: bool,
    pub balance: Balance,
    pub groups: Vec<Group>,
}
fn is_false(value: &bool) -> bool {
    !*value
}
impl Config {
    fn validate_teacher_count(&self, count: usize) {
        if self.human_only {
            assert_eq!(count, 0, "human-only controls forbid teacher groups");
            assert_eq!(
                self.teacher_mass, 1.,
                "human-only controls cannot reweight a teacher"
            );
            assert!(
                !matches!(self.order, Order::TeacherWarm),
                "human-only controls cannot warm up a teacher"
            );
        } else {
            assert_eq!(count, 1);
        }
    }
    pub(super) fn validate(&self, hp: &MlpHyperparams) {
        assert_eq!(self.schema, "monotone-training-v1");
        assert!(self.init_scale.is_finite() && self.init_scale > 0.);
        assert!(self.bias_scale.is_finite() && self.bias_scale >= 0.);
        assert!((0.0..0.5).contains(&self.dropout));
        assert!((0.0..=0.01).contains(&self.adamw));
        assert!((0.0..=4.0).contains(&self.teacher_mass));
        assert!(
            self.adamw == 0. || hp.l2_lambda == 0.,
            "AdamW and coupled L2 must be separate arms"
        );
        assert!(hp.pair_sampling == super::PairSampling::Uniform && hp.stratified_bands == 0);
        assert!(hp.ema_decay == 0. && hp.pooled_legs.is_empty() && hp.hard_pair_frac == 0.);
    }
    pub(super) fn lr(&self, epoch: usize, hp: &MlpHyperparams) -> f64 {
        match self.schedule {
            Schedule::Constant => hp.initial_lr,
            Schedule::Cosine => {
                hp.initial_lr
                    * 0.5
                    * (1. + (std::f64::consts::PI * (epoch % 50) as f64 / 50.).cos())
            }
            Schedule::Consolidate => {
                if epoch * 4 >= hp.n_epochs * 3 {
                    hp.initial_lr * 0.1
                } else {
                    hp.initial_lr
                }
            }
        }
    }
}

pub(super) fn dropout_mask(p: f64, width: usize, rng: &mut SplitMix64) -> Option<Vec<f64>> {
    if p == 0. {
        return None;
    } // No RNG draws and no arithmetic changes in p=0 replay.
    Some(
        (0..width)
            .map(|_| {
                if rng.next_f64_unit() < p {
                    0.
                } else {
                    1. / (1. - p)
                }
            })
            .collect(),
    )
}
pub(super) fn decay(weights: &mut [f64], lr: f64, lambda: f64) {
    if lambda == 0. {
        return;
    }
    assert!((0.0..1.0).contains(&(lr * lambda)));
    for value in weights {
        *value *= 1. - lr * lambda;
    }
}

type Key = (usize, u32, u32, u32);
pub(super) struct Events {
    pub draws: Vec<sampling::Draw>,
    weights: BTreeMap<Key, f64>,
    families: Vec<Vec<u32>>,
    references: Vec<Vec<u32>>,
    balance: Balance,
}
fn key(
    draw: sampling::Draw,
    families: &[Vec<u32>],
    refs: &[Vec<u32>],
    balance: Balance,
) -> Option<Key> {
    if let sampling::Draw::Pair {
        train_pos: g,
        ia,
        ib,
    } = draw
    {
        let (a, b) = (families[g][ia], families[g][ib]);
        Some((
            g,
            if balance == Balance::SourceFamily {
                refs[g][ia]
            } else {
                0
            },
            a.min(b),
            a.max(b),
        ))
    } else {
        None
    }
}
fn digest(draws: &[sampling::Draw]) -> String {
    let mut digest = sampling::SampleSequenceDigest::new();
    for &draw in draws {
        digest.push(draw);
    }
    format!("{digest:?}")
}
fn code(draw: sampling::Draw) -> (usize, usize, usize, usize) {
    match draw {
        sampling::Draw::GroupTooSmall => (0, 0, 0, 0),
        sampling::Draw::SameRow { train_pos, row } => (1, train_pos, row, 0),
        sampling::Draw::Pair { train_pos, ia, ib } => (2, train_pos, ia, ib),
    }
}
fn permute(
    draws: &mut [sampling::Draw],
    order: Order,
    difficulty: &[f64],
    families: &[Vec<u32>],
    teacher: &[bool],
) {
    if order == Order::Random {
        return;
    }
    let prefix = draws.len() * 3 / 4;
    // Every fifth prefix event stays on a random replay stream; the final 25%
    // is untouched. Reinsert replay after each four curriculum events.
    let original = draws[..prefix].to_vec();
    if matches!(
        order,
        Order::WithinPopulationEasy | Order::WithinPopulationHard | Order::WithinPopulationFamily
    ) {
        let mut slots = BTreeMap::<usize, Vec<usize>>::new();
        for (i, &draw) in original.iter().enumerate() {
            if i % 5 == 4 {
                continue;
            }
            if let sampling::Draw::Pair { train_pos, .. } = draw
                && !teacher[train_pos]
            {
                slots.entry(train_pos).or_default().push(i);
            }
        }
        for positions in slots.values() {
            let mut indices = positions.clone();
            indices.sort_by(|&a, &b| match order {
                Order::WithinPopulationEasy => difficulty[b].total_cmp(&difficulty[a]),
                Order::WithinPopulationHard => difficulty[a].total_cmp(&difficulty[b]),
                Order::WithinPopulationFamily => {
                    let family = |i: usize| match original[i] {
                        sampling::Draw::Pair { train_pos, ia, ib } => (
                            families[train_pos][ia].min(families[train_pos][ib]),
                            families[train_pos][ia].max(families[train_pos][ib]),
                        ),
                        _ => unreachable!(),
                    };
                    family(a).cmp(&family(b))
                }
                _ => unreachable!(),
            });
            for (&destination, &source) in positions.iter().zip(&indices) {
                draws[destination] = original[source];
            }
        }
        // Check the actual realized stream, not just a small schedule fixture.
        for (i, &old) in original.iter().enumerate() {
            match old {
                sampling::Draw::Pair { train_pos, .. } if !teacher[train_pos] && i % 5 != 4 => {
                    assert!(
                        matches!(draws[i], sampling::Draw::Pair {train_pos: g, ..} if g == train_pos)
                    );
                }
                _ => assert_eq!(draws[i], old, "protected order slot changed"),
            }
        }
        println!(
            "TRAINING_ORDER_ISOLATION {}",
            serde_json::json!({"population_slots_equal":true,"teacher_events_fixed":true,
                "skips_fixed":true,"prefix_replay_fixed":true,
                "checked_prefix_events":prefix,"eligible_human_events":slots.values().map(Vec::len).sum::<usize>(),
                "changed_events":original.iter().zip(&draws[..prefix]).filter(|(a,b)|a!=b).count()})
        );
        return;
    }
    let mut indices = (0..prefix).filter(|i| i % 5 != 4).collect::<Vec<_>>();
    indices.sort_by(|&a, &b| match order {
        Order::Easy => difficulty[b].total_cmp(&difficulty[a]),
        Order::Hard => difficulty[a].total_cmp(&difficulty[b]),
        Order::Family => {
            let family = |i: usize| match original[i] {
                sampling::Draw::Pair {
                    train_pos: g,
                    ia,
                    ib,
                } => (
                    g,
                    families[g][ia].min(families[g][ib]),
                    families[g][ia].max(families[g][ib]),
                ),
                _ => (usize::MAX, 0, 0),
            };
            family(a).cmp(&family(b))
        }
        Order::TeacherWarm => {
            let is_teacher = |i: usize| match original[i] {
                sampling::Draw::Pair { train_pos, .. }
                | sampling::Draw::SameRow { train_pos, .. } => teacher[train_pos],
                _ => false,
            };
            is_teacher(b).cmp(&is_teacher(a))
        }
        Order::Random
        | Order::WithinPopulationEasy
        | Order::WithinPopulationHard
        | Order::WithinPopulationFamily => unreachable!(),
    });
    let mut curriculum = indices.into_iter();
    for i in 0..prefix {
        draws[i] = original[if i % 5 == 4 {
            i
        } else {
            curriculum.next().unwrap()
        }];
    }
}
impl Events {
    pub(super) fn new(
        config: &Config,
        hp: &MlpHyperparams,
        groups: &[TrainingGroup<'_>],
        train: &[usize],
        ctx: &sampling::PairDrawCtx<'_>,
        rng: &mut SplitMix64,
    ) -> Self {
        assert_eq!(config.groups.len(), train.len());
        let mut families = Vec::new();
        let mut references = Vec::new();
        let mut teacher = Vec::new();
        for (entry, &gi) in config.groups.iter().zip(train) {
            let group = &groups[gi];
            assert_eq!(entry.name, group.name);
            assert_eq!(entry.families.len(), group.human_scores.len());
            families.push(entry.families.clone());
            references.push(group.ref_ids.expect("study requires source IDs").to_vec());
            teacher.push(group.name == "cid22");
        }
        config.validate_teacher_count(teacher.iter().filter(|&&v| v).count());
        // Keep scaler membership fixed, including for teacher_mass=0. Only the
        // proposal CDF changes; no held-out rows or labels enter this operation.
        let mut cdf = ctx.cdf.to_vec();
        if config.teacher_mass != 1. {
            let mut total = 0.;
            for (i, &gi) in train.iter().enumerate() {
                total +=
                    groups[gi].train_weight * if teacher[i] { config.teacher_mass } else { 1. };
                cdf[i] = total;
            }
            for x in &mut cdf {
                *x /= total;
            }
        }
        let count = hp
            .n_epochs
            .checked_mul(hp.pairs_per_epoch)
            .expect("study draw count overflow");
        let mut draws = (0..count)
            .map(|i| {
                sampling::draw_pair(
                    &sampling::PairDrawCtx {
                        cdf: &cdf,
                        draw_index: i as u64,
                        ..*ctx
                    },
                    rng,
                )
            })
            .collect::<Vec<_>>();
        let original_digest = digest(&draws);
        let tail_digest = digest(&draws[count * 3 / 4..]);
        let mut difficulty = vec![0.5; count];
        for (position, &gi) in train.iter().enumerate() {
            if teacher[position] {
                continue;
            }
            let mut margins = draws
                .iter()
                .enumerate()
                .filter_map(|(i, d)| match *d {
                    sampling::Draw::Pair { train_pos, ia, ib } if train_pos == position => Some((
                        (groups[gi].human_scores[ia] - groups[gi].human_scores[ib]).abs(),
                        i,
                    )),
                    _ => None,
                })
                .collect::<Vec<_>>();
            margins.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut start = 0;
            while start < margins.len() {
                let mut end = start + 1;
                while end < margins.len() && margins[end].0 == margins[start].0 {
                    end += 1;
                }
                let rank = (start + end) as f64 / (2. * margins.len() as f64);
                for &(_, i) in &margins[start..end] {
                    difficulty[i] = rank;
                }
                start = end;
            }
        }
        let mut before = draws.iter().map(|&d| code(d)).collect::<Vec<_>>();
        before.sort_unstable();
        permute(&mut draws, config.order, &difficulty, &families, &teacher);
        let mut after = draws.iter().map(|&d| code(d)).collect::<Vec<_>>();
        after.sort_unstable();
        assert_eq!(before, after, "curriculum changed the event multiset");
        assert_eq!(
            tail_digest,
            digest(&draws[count * 3 / 4..]),
            "curriculum changed consolidation tail"
        );
        let mut counts = BTreeMap::<Key, usize>::new();
        for &d in &draws {
            if let Some(k) = key(d, &families, &references, config.balance) {
                *counts.entry(k).or_default() += 1;
            }
        }
        let mut weights = BTreeMap::new();
        let mut mass = Vec::new();
        for (g, &gi) in train.iter().enumerate() {
            let cells = counts.iter().filter(|(k, _)| k.0 == g).collect::<Vec<_>>();
            let total: usize = cells.iter().map(|(_, v)| **v).sum();
            for &(&k, &n) in &cells {
                weights.insert(
                    k,
                    if config.balance == Balance::None {
                        1.
                    } else {
                        0.2 + 0.8 * total as f64 / (cells.len() as f64 * n as f64)
                    },
                );
            }
            mass.push(serde_json::json!({"group":groups[gi].name,"pair_events":total,"family_source_cells":cells.len()}));
        }
        println!(
            "TRAINING_EVENTS {}",
            serde_json::json!({"events":count,"original_digest":original_digest,"ordered_digest":digest(&draws),"tail_digest":tail_digest,"multiset_equal":true,"tail_events":count-count*3/4,"prefix_replay_stride":5,"mass":mass,"min_pair_weight":weights.values().copied().reduce(f64::min),"max_pair_weight":weights.values().copied().reduce(f64::max)})
        );
        Self {
            draws,
            weights,
            families,
            references,
            balance: config.balance,
        }
    }
    pub(super) fn weight(&self, draw: sampling::Draw) -> f64 {
        if self.balance == Balance::None {
            1.
        } else {
            self.weights[&key(draw, &self.families, &self.references, self.balance).unwrap()]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn human_only_is_explicit_and_preserves_legacy_serialization() {
        let legacy = serde_json::json!({"schema":"monotone-training-v1",
            "init_scale":1.,"bias_scale":1.,"dropout":0.,"dropout_seed":0,
            "adamw":0.,"schedule":"cosine","order":"random","teacher_mass":1.,
            "balance":"none","groups":[]});
        let config: Config = serde_json::from_value(legacy.clone()).unwrap();
        assert!(!config.human_only);
        assert_eq!(serde_json::to_value(&config).unwrap(), legacy);
        config.validate_teacher_count(1);
        assert!(std::panic::catch_unwind(|| config.validate_teacher_count(0)).is_err());
        assert!(std::panic::catch_unwind(|| config.validate_teacher_count(2)).is_err());
        let mut human = config.clone();
        human.human_only = true;
        human.validate_teacher_count(0);
        assert!(std::panic::catch_unwind(|| human.validate_teacher_count(1)).is_err());
        human.teacher_mass = 0.;
        assert!(std::panic::catch_unwind(|| human.validate_teacher_count(0)).is_err());
        human.teacher_mass = 1.;
        human.order = Order::TeacherWarm;
        assert!(std::panic::catch_unwind(|| human.validate_teacher_count(0)).is_err());
    }
    #[test]
    fn within_population_orders_keep_teacher_skips_and_every_population_slot() {
        let events = (0..600)
            .map(|i| {
                if i % 31 == 0 {
                    sampling::Draw::GroupTooSmall
                } else if i % 17 == 0 {
                    sampling::Draw::SameRow {
                        train_pos: i % 3,
                        row: i,
                    }
                } else {
                    sampling::Draw::Pair {
                        train_pos: i % 3,
                        ia: i,
                        ib: i + 1,
                    }
                }
            })
            .collect::<Vec<_>>();
        let difficulty = (0..600)
            .map(|i| ((i * 137) % 601) as f64)
            .collect::<Vec<_>>();
        let families = vec![(0..601).map(|i| (i * 13 % 23) as u32).collect::<Vec<_>>(); 3];
        for order in [
            Order::WithinPopulationEasy,
            Order::WithinPopulationHard,
            Order::WithinPopulationFamily,
        ] {
            let mut changed = events.clone();
            permute(
                &mut changed,
                order,
                &difficulty,
                &families,
                &[false, false, true],
            );
            let mut old_codes = events.iter().map(|&d| code(d)).collect::<Vec<_>>();
            let mut new_codes = changed.iter().map(|&d| code(d)).collect::<Vec<_>>();
            old_codes.sort();
            new_codes.sort();
            assert_eq!(old_codes, new_codes);
            assert_eq!(events[450..], changed[450..]);
            let mut by_population = [Vec::new(), Vec::new()];
            for (i, &old) in events[..450].iter().enumerate() {
                match old {
                    sampling::Draw::Pair { train_pos, .. } if train_pos < 2 && i % 5 != 4 => {
                        let sampling::Draw::Pair {
                            train_pos: g,
                            ia,
                            ib,
                        } = changed[i]
                        else {
                            panic!("pair slot lost")
                        };
                        assert_eq!(train_pos, g);
                        by_population[g].push((
                            difficulty[ia],
                            families[g][ia].min(families[g][ib]),
                            families[g][ia].max(families[g][ib]),
                        ));
                    }
                    _ => assert_eq!(old, changed[i]),
                }
            }
            for g in 0..2 {
                assert!(events.iter().zip(&changed).any(
                    |(&a, &b)| matches!(a,sampling::Draw::Pair {train_pos,..} if train_pos==g)
                        && a != b
                ));
                assert!(by_population[g].windows(2).all(|p| match order {
                    Order::WithinPopulationEasy => p[0].0 >= p[1].0,
                    Order::WithinPopulationHard => p[0].0 <= p[1].0,
                    Order::WithinPopulationFamily => (p[0].1, p[0].2) <= (p[1].1, p[1].2),
                    _ => unreachable!(),
                }));
            }
        }
    }
    #[test]
    fn curricula_preserve_multiset_tail_and_replay() {
        let events = (0..100)
            .map(|i| sampling::Draw::Pair {
                train_pos: i % 2,
                ia: i % 7,
                ib: (i + 1) % 7,
            })
            .collect::<Vec<_>>();
        for order in [Order::Easy, Order::Hard, Order::Family, Order::TeacherWarm] {
            let mut changed = events.clone();
            permute(
                &mut changed,
                order,
                &(0..100).map(|i| i as f64 / 100.).collect::<Vec<_>>(),
                &vec![vec![0, 0, 1, 1, 2, 2, 3]; 2],
                &[false, true],
            );
            let mut a = events.iter().map(|&d| code(d)).collect::<Vec<_>>();
            let mut b = changed.iter().map(|&d| code(d)).collect::<Vec<_>>();
            a.sort();
            b.sort();
            assert_eq!(a, b);
            assert_eq!(events[75..], changed[75..]);
            for i in (4..75).step_by(5) {
                assert_eq!(events[i], changed[i]);
            }
        }
    }
    #[test]
    fn dropout_zero_consumes_no_rng_and_adamw_is_decoupled() {
        let mut rng = SplitMix64::new(7);
        let mut reference = SplitMix64::new(7);
        assert!(dropout_mask(0., 32, &mut rng).is_none());
        assert_eq!(rng.next_u64(), reference.next_u64());
        let mut w = [-2., 0., 3.];
        decay(&mut w, 0.1, 0.01);
        assert_eq!(w, [-1.998, 0., 2.997]);
    }
}
