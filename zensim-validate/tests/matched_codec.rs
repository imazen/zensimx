//! Invented targets exercise the registered private schedule through training.
use std::collections::BTreeMap;
use zenpredict::{Model, Predictor};
use zensim_validate::mlp_train::{
    FeatureRows, GroupLossMode, MlpHyperparams, TrainingGroup, train_mlp_strategy_pairs,
};

fn fit(
    matched: bool,
    pairs: Vec<[usize; 2]>,
    mutate: impl FnOnce(&mut MlpHyperparams),
) -> Vec<f32> {
    let features = [[0., 1., 0.], [1., 0., 0.], [0., 0., 1.], [1., 1., 0.]];
    let rows: Vec<&[f64]> = features.iter().map(|r| r.as_slice()).collect();
    let name = if matched {
        "human_codec_strata"
    } else {
        "human"
    };
    let mut groups = [
        TrainingGroup {
            name: name.into(),
            human_scores: &[0., 1., 2., 3.],
            features: FeatureRows::Borrowed(&rows),
            metric_sigmas: None,
            train_weight: 1.,
            validation_weight: 0.,
            ref_ids: Some(&[0, 0, 0, 0]),
            loss_mode: GroupLossMode::Rank,
        },
        TrainingGroup {
            name: "absolute_control".into(),
            human_scores: &[0., 1., 2., 3.],
            features: FeatureRows::Borrowed(&rows),
            metric_sigmas: None,
            train_weight: 0.,
            validation_weight: 1.,
            ref_ids: None,
            loss_mode: GroupLossMode::Both,
        },
    ];
    let mut hp = MlpHyperparams {
        n_hidden: 8,
        n_epochs: 1,
        pairs_per_epoch: 2000,
        seed: 1101,
        sample_seed: Some(17103),
        initial_lr: 0.01,
        l2_lambda: 0.,
        early_stop_patience: 0,
        parallel_batch: false,
        ..Default::default()
    };
    mutate(&mut hp);
    let lists = if matched {
        BTreeMap::from([(name.into(), pairs)])
    } else {
        BTreeMap::new()
    };
    let bake = train_mlp_strategy_pairs(
        &mut groups,
        3,
        &hp,
        &mut Vec::new(),
        None,
        None,
        None,
        None,
        None,
        None,
        &lists,
    );
    let model = Model::from_bytes(&bake).unwrap();
    let mut predictor = Predictor::new(&model);
    features
        .iter()
        .map(|row| predictor.predict(&row.map(|v| v as f32)).unwrap()[0])
        .collect()
}

#[test]
fn admitted_matches_reach_training_and_reverse_orientation_is_symmetric() {
    let baseline = fit(false, vec![], |_| {});
    let matched = fit(true, vec![[0, 1]], |_| {});
    let reversed = fit(true, vec![[1, 0]], |_| {});
    assert_ne!(
        baseline, matched,
        "the schedule must reach the real trainer"
    );
    assert_eq!(
        matched, reversed,
        "RankNet must preserve reversed-pair updates"
    );
}

#[test]
#[should_panic(expected = "matched codecs require eligible pairs")]
fn matched_group_refuses_empty_list() {
    fit(true, vec![], |_| {});
}

#[test]
#[should_panic(expected = "plain sequential uniform trainer")]
fn matched_group_refuses_alternate_head_before_dispatch() {
    fit(true, vec![[0, 1]], |hp| hp.pool_head = true);
}

#[test]
#[should_panic(expected = "plain sequential uniform trainer")]
fn matched_group_refuses_hard_pair_mining() {
    fit(true, vec![[0, 1]], |hp| hp.hard_pair_frac = 0.5);
}
