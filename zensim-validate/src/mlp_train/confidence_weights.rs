//! Private, hash-bound MlicConfidence admission; absent leaves the trainer unchanged.
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

use super::contrast_precision::{self, Normalizer, Point};
use super::{
    GroupLossMode, MlpHyperparams, OutputPolarity, PairSampling, TrainingGroup, subjective_rank,
};

#[derive(Deserialize)]
struct Binding {
    rows: usize,
    scores_refs_sha256: String,
    points: Vec<Option<[f64; 2]>>,
}
#[derive(Deserialize)]
struct Group {
    name: String,
    train_weight: f64,
    validation_weight: f64,
    mode: String,
    binding: Option<Binding>,
}
#[derive(Deserialize)]
struct Manifest {
    schema: String,
    candidate: String,
    root_sha256: String,
    init_seed: u64,
    sample_seed: u64,
    epochs: usize,
    pairs_per_epoch: usize,
    selected_epoch: usize,
    n_features: usize,
    hidden: usize,
    #[serde(rename = "sigma_L")]
    sigma: f64,
    item_floor: f64,
    input_sha256: BTreeMap<String, String>,
    groups: Vec<Group>,
    tie_policy: String,
    pair_law: String,
    absolute_loss_changed: bool,
    group_masses_changed: bool,
}
/// A registered 60-epoch continuation of a full confidence fit (AIC-share
/// frontier stage B): the unchanged full-fit manifest, the starting checkpoint
/// and one appended, unweighted AIC public_train group. The subjective
/// authority binds the protocol, population and AIC weight independently.
#[derive(Deserialize)]
struct Continuation {
    schema: String,
    base_manifest: String,
    base_manifest_sha256: String,
    checkpoint: String,
    checkpoint_sha256: String,
    epochs: usize,
    pairs_per_epoch: usize,
    added_group: Group,
}
const CONTINUATION_EPOCHS: usize = 60;

/// The continuation's (epochs, pairs per epoch), after binding its starting
/// checkpoint bytes and its one unweighted AIC group.
fn continuation_budget(c: &Continuation, base: &Manifest, hp: &MlpHyperparams) -> (usize, usize) {
    assert_eq!(c.schema, "mlic-contrast-confidence-continuation-v1");
    assert_eq!(
        (c.epochs, c.pairs_per_epoch),
        (CONTINUATION_EPOCHS, base.pairs_per_epoch)
    );
    let start = hp
        .init_from
        .as_deref()
        .expect("confidence continuation requires its checkpoint");
    assert_eq!(start, Path::new(&c.checkpoint));
    assert_eq!(
        crate::train_manifest::sha256_file(start).expect("continuation checkpoint"),
        c.checkpoint_sha256,
        "confidence continuation checkpoint changed"
    );
    let g = &c.added_group;
    assert!(
        g.name == "aic_public_train"
            && g.mode == "rank"
            && g.validation_weight == 0.
            && g.train_weight.is_finite()
            && g.train_weight > 0.
            && g.binding.is_none(),
        "unregistered confidence continuation group"
    );
    (c.epochs, c.pairs_per_epoch)
}
struct WeightedGroup {
    points: Vec<Option<Point>>,
    refs: Vec<u32>,
    normalizers: BTreeMap<u32, Normalizer>,
}
pub(super) struct Plan {
    groups: Vec<Option<WeightedGroup>>,
    sigma: f64,
}

fn digest(data: &[u8]) -> String {
    crate::train_manifest::hex_lower(&Sha256::digest(data))
}
fn row_digest(scores: &[f64], refs: &[u32]) -> String {
    assert_eq!(scores.len(), refs.len());
    let mut hash = Sha256::new();
    for (&score, &reference) in scores.iter().zip(refs) {
        assert!(score.is_finite());
        hash.update(score.to_le_bytes());
        hash.update(reference.to_le_bytes());
    }
    crate::train_manifest::hex_lower(&hash.finalize())
}

impl Plan {
    pub(super) fn from_env(
        groups: &[TrainingGroup<'_>],
        width: usize,
        hp: &MlpHyperparams,
        has_pair_lists: bool,
        log: &mut Vec<String>,
    ) -> Option<Self> {
        let path = std::env::var_os("ZENSIM_RESEARCH_CONFIDENCE_MANIFEST");
        let pin = std::env::var_os("ZENSIM_RESEARCH_CONFIDENCE_SHA256");
        let receipt = std::env::var_os("ZENSIM_RESEARCH_CONFIDENCE_RECEIPT");
        if path.is_none() && pin.is_none() && receipt.is_none() {
            return None;
        }
        assert_eq!(
            usize::BITS,
            64,
            "registered sampler uses 64-bit modulo words"
        );
        let bytes = std::fs::read(path.expect("confidence manifest required"))
            .expect("read confidence manifest");
        let expected = pin.expect("independent confidence manifest hash required");
        assert_eq!(
            digest(&bytes),
            expected.to_str().expect("UTF8 manifest hash")
        );
        let value: serde_json::Value =
            serde_json::from_slice(&bytes).expect("confidence manifest JSON");
        let (manifest, continuation): (Manifest, Option<Continuation>) =
            if value["schema"] == "mlic-contrast-confidence-continuation-v1" {
                let c: Continuation =
                    serde_json::from_value(value).expect("confidence continuation schema");
                let base = std::fs::read(&c.base_manifest).expect("read base confidence manifest");
                assert_eq!(
                    digest(&base),
                    c.base_manifest_sha256,
                    "base confidence manifest changed"
                );
                let base = serde_json::from_slice(&base).expect("confidence manifest schema");
                (base, Some(c))
            } else {
                let manifest = serde_json::from_value(value).expect("confidence manifest schema");
                (manifest, None)
            };
        assert!(
            !has_pair_lists
                && !hp.per_sample_alpha_head
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
            "confidence requires registered sequential uniform trainer"
        );
        assert_eq!(manifest.schema, "mlic-contrast-confidence-v1");
        assert_eq!(manifest.candidate, "MlicConfidence");
        assert_eq!(
            manifest.root_sha256,
            "a7fd4af0c2bb5a8981be636e5c1cb1a51db57496b7c60d7bf6da01d71dd16952"
        );
        assert_eq!(
            (
                manifest.epochs,
                manifest.pairs_per_epoch,
                manifest.selected_epoch
            ),
            (120, 50000, 119)
        );
        let budget = match &continuation {
            None => (manifest.epochs, manifest.pairs_per_epoch),
            Some(c) => continuation_budget(c, &manifest, hp),
        };
        assert_eq!(
            (hp.n_epochs, hp.pairs_per_epoch, hp.n_hidden, width),
            (budget.0, budget.1, manifest.hidden, manifest.n_features)
        );
        assert_eq!(hp.init_seed.unwrap_or(hp.seed), manifest.init_seed);
        assert_eq!(hp.sample_seed.unwrap_or(hp.seed), manifest.sample_seed);
        assert_eq!(manifest.item_floor, 0.05);
        assert_eq!(manifest.tie_policy, "existing-native-owner");
        assert_eq!(manifest.pair_law, "within-ref-u64-modulo");
        assert!(!manifest.absolute_loss_changed && !manifest.group_masses_changed);
        assert!(manifest.sigma.is_finite() && manifest.sigma > 0.);
        for (name, pin) in &manifest.input_sha256 {
            assert_eq!(
                crate::train_manifest::sha256_file(Path::new(name))
                    .expect("confidence input missing"),
                *pin,
                "confidence input changed: {name}"
            );
        }
        let added = continuation.as_ref().map(|c| &c.added_group);
        let (plan, audit) = Self::bind(&manifest, added, groups, hp);
        let path = receipt.expect("fresh confidence normalizer receipt required");
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(Path::new(&path))
            .expect("fresh confidence normalizer receipt");
        let mut record = serde_json::json!({
            "schema": "mlic-native-contrast-normalizers-v1", "manifest_sha256": digest(&bytes),
            "sigma_L": manifest.sigma, "groups": audit, "sampler_changed": false,
            "absolute_loss_changed": false
        });
        if let Some(c) = &continuation {
            record["continuation"] = serde_json::json!({
                "base_manifest_sha256": c.base_manifest_sha256,
                "checkpoint_sha256": c.checkpoint_sha256, "epochs": c.epochs,
                "added_group": c.added_group.name, "added_group_weighted": false});
        }
        serde_json::to_writer_pretty(file, &record).expect("write native normalizer receipt");
        log.push(format!(
            "confidence manifest {} admitted; native normalizers recorded",
            digest(&bytes)
        ));
        Some(plan)
    }

    fn bind(
        manifest: &Manifest,
        added: Option<&Group>,
        groups: &[TrainingGroup<'_>],
        hp: &MlpHyperparams,
    ) -> (Self, Vec<serde_json::Value>) {
        let declarations: Vec<&Group> = manifest.groups.iter().chain(added).collect();
        assert_eq!(
            groups.len(),
            declarations.len(),
            "complete original group inventory required"
        );
        let sign = OutputPolarity::for_groups(groups, hp).rank_target_sign();
        let mut result = Vec::new();
        let mut audit = Vec::new();
        for (group, declaration) in groups.iter().zip(declarations) {
            assert_eq!(group.name, declaration.name);
            assert_eq!(
                group.train_weight.to_bits(),
                declaration.train_weight.to_bits()
            );
            assert_eq!(
                group.validation_weight.to_bits(),
                declaration.validation_weight.to_bits()
            );
            let tokens: Vec<_> = declaration.mode.split(',').collect();
            let loss = if tokens.contains(&"both") {
                GroupLossMode::Both
            } else if tokens.contains(&"mse") {
                GroupLossMode::Mse
            } else {
                GroupLossMode::Rank
            };
            assert_eq!(group.loss_mode, loss);
            assert_eq!(group.ref_ids.is_some(), tokens.contains(&"withinref"));
            let Some(binding) = &declaration.binding else {
                audit.push(serde_json::json!({"group": group.name, "baseline": true}));
                result.push(None);
                continue;
            };
            assert!(
                matches!(
                    group.name.as_str(),
                    "human"
                        | "subjective_rank_live_md"
                        | "subjective_rank_colourlab_rating_variability_2022"
                ),
                "unregistered confidence group"
            );
            assert_eq!(group.loss_mode, GroupLossMode::Rank);
            assert!(group.train_weight > 0. && group.validation_weight == 0.);
            let refs = group
                .ref_ids
                .expect("confidence requires original references");
            assert_eq!(binding.rows, group.human_scores.len());
            assert_eq!(binding.points.len(), binding.rows);
            assert_eq!(
                row_digest(group.human_scores, refs),
                binding.scores_refs_sha256
            );
            let points: Vec<_> = binding
                .points
                .iter()
                .map(|p| p.map(|[x, se]| Point { x, se }))
                .collect();
            let mut buckets = BTreeMap::<u32, Vec<usize>>::new();
            for (i, &reference) in refs.iter().enumerate() {
                buckets.entry(reference).or_default().push(i);
            }
            let mut normalizers = BTreeMap::new();
            let mut reference_audit = Vec::new();
            for (reference, indices) in buckets {
                let covered = indices.iter().filter(|&&i| points[i].is_some()).count();
                assert!(
                    covered == 0 || covered == indices.len(),
                    "partially covered reference"
                );
                if covered == 0 {
                    continue;
                }
                let local: Vec<_> = indices.iter().map(|&i| points[i].unwrap()).collect();
                let norm = contrast_precision::normalize(&local, manifest.sigma, 64, |a, b| {
                    let (a, b) = (
                        group.human_scores[indices[a]],
                        group.human_scores[indices[b]],
                    );
                    !subjective_rank::tied(group, a, b, sign * (a - b).signum())
                });
                reference_audit.push(serde_json::json!({"reference": reference, "rows": indices.len(),
                    "mean_precision": norm.as_ref().map(|n| n.mean_precision),
                    "legal_ordered_pairs": norm.as_ref().map_or(0, |n| n.legal_ordered_pairs),
                    "legal_draw_multiplicity": norm.as_ref().map(|n| n.legal_draw_multiplicity.to_string())}));
                if let Some(norm) = norm {
                    normalizers.insert(reference, norm);
                }
            }
            audit.push(serde_json::json!({"group": group.name, "baseline": false, "references": reference_audit}));
            result.push(Some(WeightedGroup {
                points,
                refs: refs.to_vec(),
                normalizers,
            }));
        }
        (
            Self {
                groups: result,
                sigma: manifest.sigma,
            },
            audit,
        )
    }

    pub(super) fn weight(&self, group: usize, a: usize, b: usize) -> f64 {
        let Some(group) = &self.groups[group] else {
            return 1.;
        };
        assert_eq!(group.refs[a], group.refs[b]);
        match (group.points[a], group.points[b]) {
            (None, None) => 1.,
            (Some(a_point), Some(b_point)) => group
                .normalizers
                .get(&group.refs[a])
                .expect("active pair requires a legal reference normalizer")
                .weight(a_point, b_point, self.sigma),
            _ => panic!("partial confidence coverage"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlp_train::FeatureRows;

    fn manifest(group: &TrainingGroup<'_>) -> Manifest {
        let refs = group.ref_ids.unwrap();
        serde_json::from_value(serde_json::json!({
            "schema":"mlic-contrast-confidence-v1", "candidate":"MlicConfidence",
            "root_sha256":"a7fd4af0c2bb5a8981be636e5c1cb1a51db57496b7c60d7bf6da01d71dd16952",
            "init_seed":1101, "sample_seed":101, "epochs":120, "pairs_per_epoch":50000,
            "selected_epoch":119, "n_features":1, "hidden":128, "sigma_L":0.5, "item_floor":0.05,
            "input_sha256":{}, "tie_policy":"existing-native-owner", "pair_law":"within-ref-u64-modulo",
            "absolute_loss_changed":false, "group_masses_changed":false,
            "groups":[{"name":group.name, "train_weight":1., "validation_weight":0.,
                "mode":"withinref,rank", "binding":{"rows":5,
                    "scores_refs_sha256":row_digest(group.human_scores,refs),
                    "points":[[0.,0.],[0.4,0.1],[2.,0.3],null,null]}}]
        })).unwrap()
    }

    #[test]
    fn loaded_rows_use_native_ties_and_keep_baseline_references() {
        let scores = [1., 1., 2., 3., 4.];
        let refs = [0, 0, 0, 1, 1];
        let features: [&[f64]; 5] = [&[0.]; 5];
        for name in ["human", "subjective_rank_live_md"] {
            let group = TrainingGroup {
                name: name.into(),
                human_scores: &scores,
                features: FeatureRows::Borrowed(&features),
                metric_sigmas: None,
                train_weight: 1.,
                validation_weight: 0.,
                ref_ids: Some(&refs),
                loss_mode: GroupLossMode::Rank,
            };
            let declaration = manifest(&group);
            let hp = MlpHyperparams::default();
            let (plan, audit) = Plan::bind(&declaration, None, &[group], &hp);
            assert_eq!(
                audit[0]["references"][0]["legal_ordered_pairs"],
                if name == "human" { 6 } else { 4 }
            );
            assert_eq!(plan.weight(0, 3, 4), 1.);
            assert_eq!(plan.weight(0, 0, 2), plan.weight(0, 2, 0));
            let mut sum = 0.;
            let mut count = 0;
            for a in 0..3 {
                for b in 0..3 {
                    if a != b && (name == "human" || scores[a] != scores[b]) {
                        sum += plan.weight(0, a, b);
                        count += 1;
                    }
                }
            }
            assert!((sum / count as f64 - 1.).abs() < 1e-12);
        }
    }

    #[test]
    fn loaded_order_mass_and_partial_reference_changes_refuse() {
        let scores = [1., 1., 2., 3., 4.];
        let refs = [0, 0, 0, 1, 1];
        let features: [&[f64]; 5] = [&[0.]; 5];
        let group = TrainingGroup {
            name: "human".into(),
            human_scores: &scores,
            features: FeatureRows::Borrowed(&features),
            metric_sigmas: None,
            train_weight: 1.,
            validation_weight: 0.,
            ref_ids: Some(&refs),
            loss_mode: GroupLossMode::Rank,
        };
        for mutation in 0..3 {
            let mut d = manifest(&group);
            match mutation {
                0 => d.groups[0].binding.as_mut().unwrap().points[0] = None,
                1 => d.groups[0].binding.as_mut().unwrap().scores_refs_sha256 = "different".into(),
                _ => d.groups[0].train_weight = 2.,
            }
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    Plan::bind(
                        &d,
                        None,
                        std::slice::from_ref(&group),
                        &MlpHyperparams::default(),
                    )
                }))
                .is_err()
            );
        }
    }

    fn aic_declaration(weight: f64) -> Group {
        serde_json::from_value(
            serde_json::json!({"name":"aic_public_train", "train_weight":weight,
            "validation_weight":0., "mode":"rank", "binding":null}),
        )
        .unwrap()
    }

    #[test]
    fn continuation_appends_one_unweighted_aic_group() {
        static REFS: [u32; 5] = [0, 0, 0, 1, 1];
        let scores = [1., 1., 2., 3., 4.];
        let features: [&[f64]; 5] = [&[0.]; 5];
        let group = |name: &str, weight: f64, refs: Option<&'static [u32]>| TrainingGroup {
            name: name.into(),
            human_scores: &scores,
            features: FeatureRows::Borrowed(&features),
            metric_sigmas: None,
            train_weight: weight,
            validation_weight: 0.,
            ref_ids: refs,
            loss_mode: GroupLossMode::Rank,
        };
        let human = group("human", 1., Some(&REFS));
        let declaration = manifest(&human);
        let aic = aic_declaration(256.);
        let hp = MlpHyperparams::default();
        let groups = [human, group("aic_public_train", 256., None)];
        let (plan, audit) = Plan::bind(&declaration, Some(&aic), &groups, &hp);
        assert_eq!(
            audit[1],
            serde_json::json!({"group":"aic_public_train", "baseline":true})
        );
        assert_eq!(plan.weight(1, 0, 4), 1.);
        // The human group keeps its full-fit weighting.
        let (alone, _) = Plan::bind(&declaration, None, &groups[..1], &hp);
        assert_eq!(plan.weight(0, 0, 2), alone.weight(0, 0, 2));
        for (groups, added) in [
            (&groups[..1], Some(&aic)),
            (&groups[..], None),
            (&groups[..], Some(&aic_declaration(64.))),
        ] {
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    Plan::bind(&declaration, added, groups, &hp)
                }))
                .is_err()
            );
        }
    }

    #[test]
    fn continuation_budget_binds_checkpoint_epochs_and_unweighted_group() {
        let scores = [1., 1., 2., 3., 4.];
        let refs = [0, 0, 0, 1, 1];
        let features: [&[f64]; 5] = [&[0.]; 5];
        let human = TrainingGroup {
            name: "human".into(),
            human_scores: &scores,
            features: FeatureRows::Borrowed(&features),
            metric_sigmas: None,
            train_weight: 1.,
            validation_weight: 0.,
            ref_ids: Some(&refs),
            loss_mode: GroupLossMode::Rank,
        };
        let base = manifest(&human);
        let dir = std::path::PathBuf::from(std::env::var_os("HOME").unwrap())
            .join("tmp")
            .join(format!("confidence-continuation-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let start = dir.join("start.bin");
        std::fs::write(&start, b"model").unwrap();
        let other = dir.join("other.bin");
        std::fs::write(&other, b"other").unwrap();
        let continuation = |value: serde_json::Value| -> Continuation {
            let mut c = serde_json::json!({"schema":"mlic-contrast-confidence-continuation-v1",
                "base_manifest":"unused", "base_manifest_sha256":"unused",
                "checkpoint":start, "checkpoint_sha256":digest(b"model"), "epochs":60,
                "pairs_per_epoch":50000, "added_group":serde_json::to_value(
                    serde_json::json!({"name":"aic_public_train", "train_weight":256.,
                        "validation_weight":0., "mode":"rank", "binding":null})).unwrap()});
            for (k, v) in value.as_object().unwrap() {
                c[k] = v.clone();
            }
            serde_json::from_value(c).unwrap()
        };
        let hp = MlpHyperparams {
            n_epochs: 60,
            init_from: Some(start.clone()),
            ..MlpHyperparams::default()
        };
        assert_eq!(
            continuation_budget(&continuation(serde_json::json!({})), &base, &hp),
            (60, 50000)
        );
        let refused = |c: Continuation, hp: &MlpHyperparams| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                continuation_budget(&c, &base, hp)
            }))
            .is_err()
        };
        for change in [
            serde_json::json!({"epochs":59}),
            serde_json::json!({"pairs_per_epoch":40000}),
            serde_json::json!({"schema":"mlic-contrast-confidence-v1"}),
            serde_json::json!({"checkpoint_sha256":digest(b"other")}),
            serde_json::json!({"checkpoint":other}),
            serde_json::json!({"added_group":{"name":"aic", "train_weight":256.,
                "validation_weight":0., "mode":"rank", "binding":null}}),
            serde_json::json!({"added_group":{"name":"aic_public_train", "train_weight":256.,
                "validation_weight":0., "mode":"withinref,rank", "binding":null}}),
            serde_json::json!({"added_group":{"name":"aic_public_train", "train_weight":256.,
                "validation_weight":1., "mode":"rank", "binding":null}}),
            serde_json::json!({"added_group":{"name":"aic_public_train", "train_weight":256.,
                "validation_weight":0., "mode":"rank", "binding":{"rows":5,
                    "scores_refs_sha256":"x", "points":[null, null, null, null, null]}}}),
        ] {
            assert!(refused(continuation(change.clone()), &hp), "{change}");
        }
        let unbound = MlpHyperparams {
            n_epochs: 60,
            ..MlpHyperparams::default()
        };
        assert!(refused(continuation(serde_json::json!({})), &unbound));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn native_equal_precision_training_parity() {
        use std::process::Command;
        let scores = [1., 1., 2., 3., 4.];
        let refs = [0, 0, 0, 1, 1];
        let features: [&[f64]; 5] = [
            &[0.1, 0.8],
            &[0.2, 0.7],
            &[0.4, 0.5],
            &[0.6, 0.3],
            &[0.9, 0.1],
        ];
        let mut groups = [
            TrainingGroup {
                name: "human".into(),
                human_scores: &scores,
                features: FeatureRows::Borrowed(&features),
                metric_sigmas: None,
                train_weight: 0.8,
                validation_weight: 0.,
                ref_ids: Some(&refs),
                loss_mode: GroupLossMode::Rank,
            },
            TrainingGroup {
                name: "absolute_control".into(),
                human_scores: &scores,
                features: FeatureRows::Borrowed(&features),
                metric_sigmas: None,
                train_weight: 0.2,
                validation_weight: 1.,
                ref_ids: None,
                loss_mode: GroupLossMode::Mse,
            },
        ];
        if let Some(child) = std::env::var_os("ZENSIM_CONFIDENCE_PARITY_CHILD") {
            let child = std::path::PathBuf::from(child);
            let hp = MlpHyperparams {
                n_hidden: 8,
                n_epochs: 120,
                pairs_per_epoch: 50000,
                init_seed: Some(1101),
                sample_seed: Some(101),
                minibatch_size: 1,
                parallel_batch: false,
                log_every: 1,
                early_stop_patience: 0,
                dump_checkpoints_every: 1,
                dump_checkpoints_dir: Some(child.clone()),
                ..MlpHyperparams::default()
            };
            let mut log = Vec::new();
            let bytes = crate::mlp_train::train_mlp_strategy_pairs(
                &mut groups,
                2,
                &hp,
                &mut log,
                None,
                None,
                None,
                None,
                None,
                None,
                &BTreeMap::new(),
            );
            std::fs::write(child.join("result.bin"), bytes).unwrap();
            std::fs::write(child.join("train.json"), serde_json::to_vec(&log).unwrap()).unwrap();
        } else {
            let root = std::path::PathBuf::from(std::env::var_os("HOME").unwrap())
                .join("tmp")
                .join(format!("confidence-native-parity-{}", std::process::id()));
            std::fs::create_dir(&root).unwrap();
            let base = serde_json::json!({
                "schema":"mlic-contrast-confidence-v1", "candidate":"MlicConfidence",
                "root_sha256":"a7fd4af0c2bb5a8981be636e5c1cb1a51db57496b7c60d7bf6da01d71dd16952",
                "init_seed":1101, "sample_seed":101, "epochs":120, "pairs_per_epoch":50000,
                "selected_epoch":119, "n_features":2, "hidden":8, "sigma_L":0.5, "item_floor":0.05,
                "input_sha256":{}, "tie_policy":"existing-native-owner", "pair_law":"within-ref-u64-modulo",
                "absolute_loss_changed":false, "group_masses_changed":false,
                "groups":[{"name":"human", "train_weight":0.8, "validation_weight":0.,
                    "mode":"withinref,rank", "binding":{"rows":5,
                        "scores_refs_sha256":row_digest(&scores,&refs),
                        "points":[[0.,0.1],[0.,0.1],[0.,0.1],null,null]}},
                    {"name":"absolute_control","train_weight":0.2,"validation_weight":1.,"mode":"mse","binding":null}]
            });
            for arm in ["baseline", "equal", "unequal"] {
                let dir = root.join(arm);
                std::fs::create_dir(&dir).unwrap();
                let mut command = Command::new(std::env::current_exe().unwrap());
                command.args(["--exact", "mlp_train::confidence_weights::tests::native_equal_precision_training_parity", "--nocapture"])
                    .env("ZENSIM_CONFIDENCE_PARITY_CHILD", &dir)
                    .env("RAYON_NUM_THREADS", "1").env("ZENSIM_MAX_TIER", "v3")
                    .env_remove("ZENSIM_RESEARCH_CONFIDENCE_MANIFEST")
                    .env_remove("ZENSIM_RESEARCH_CONFIDENCE_SHA256")
                    .env_remove("ZENSIM_RESEARCH_CONFIDENCE_RECEIPT");
                if arm != "baseline" {
                    let mut value = base.clone();
                    if arm == "unequal" {
                        value["groups"][0]["binding"]["points"][2] = serde_json::json!([2., 0.3]);
                    }
                    let bytes = serde_json::to_vec(&value).unwrap();
                    let path = dir.join("manifest.json");
                    std::fs::write(&path, &bytes).unwrap();
                    command
                        .env("ZENSIM_RESEARCH_CONFIDENCE_MANIFEST", path)
                        .env("ZENSIM_RESEARCH_CONFIDENCE_SHA256", digest(&bytes))
                        .env(
                            "ZENSIM_RESEARCH_CONFIDENCE_RECEIPT",
                            dir.join("receipt.json"),
                        );
                }
                let log = std::fs::File::create(dir.join("child.log")).unwrap();
                command.stdout(log.try_clone().unwrap()).stderr(log);
                assert!(
                    command.status().unwrap().success(),
                    "native child failed: {}",
                    dir.display()
                );
            }
            for epoch in 0..120 {
                let file = format!("ckpt_epoch{epoch:03}.bin");
                assert_eq!(
                    std::fs::read(root.join("baseline").join(&file)).unwrap(),
                    std::fs::read(root.join("equal").join(&file)).unwrap(),
                    "epoch {epoch}"
                );
            }
            let epoch_logs = |arm: &str| -> Vec<String> {
                let log: Vec<String> = serde_json::from_slice(
                    &std::fs::read(root.join(arm).join("train.json")).unwrap(),
                )
                .unwrap();
                log.into_iter()
                    .filter(|line| line.starts_with("  epoch "))
                    .map(|line| line.split(" | t=").next().unwrap().to_owned())
                    .collect()
            };
            let baseline_logs = epoch_logs("baseline");
            assert_eq!(baseline_logs.len(), 120);
            assert_eq!(baseline_logs, epoch_logs("equal"));
            assert_ne!(baseline_logs, epoch_logs("unequal"));
            let baseline = std::fs::read(root.join("baseline/result.bin")).unwrap();
            assert_eq!(
                baseline,
                std::fs::read(root.join("equal/result.bin")).unwrap()
            );
            assert_ne!(
                std::fs::read(root.join("baseline/ckpt_epoch119.bin")).unwrap(),
                std::fs::read(root.join("unequal/ckpt_epoch119.bin")).unwrap()
            );
            assert!(root.join("equal/receipt.json").is_file());
            assert!(root.join("unequal/receipt.json").is_file());
            eprintln!("native confidence parity artifacts: {}", root.display());
        }
    }
}
