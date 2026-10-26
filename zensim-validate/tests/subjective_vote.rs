//! Invented two-vote responses exercise the actual sequential CPU trainer.
use zenpredict::{Model, Predictor};
use zensim_validate::mlp_train::{
    FeatureRows, GroupLossMode, MlpHyperparams, TrainingGroup, train_mlp,
};

fn fit(
    targets: &[f64],
    refs: &[u32],
    epochs: usize,
    mutate: impl FnOnce(&mut MlpHyperparams),
) -> Vec<f32> {
    let features = [[0., 1., 0., 0.], [1., 1., 1., 0.]];
    let rows: Vec<&[f64]> = features.iter().map(|r| r.as_slice()).collect();
    let mut groups = [
        TrainingGroup {
            name: "subjective_vote_bapps".into(),
            human_scores: targets,
            features: FeatureRows::Borrowed(&rows),
            metric_sigmas: None,
            train_weight: 1.,
            validation_weight: 0.,
            ref_ids: Some(refs),
            loss_mode: GroupLossMode::Rank,
        },
        TrainingGroup {
            name: "absolute_control".into(),
            human_scores: &[1., 0.],
            features: FeatureRows::Borrowed(&rows),
            metric_sigmas: None,
            train_weight: 0.,
            validation_weight: 1.,
            ref_ids: Some(&[0, 0]),
            loss_mode: GroupLossMode::Both,
        },
    ];
    let mut hp = MlpHyperparams {
        n_hidden: 8,
        n_epochs: epochs,
        pairs_per_epoch: 2000,
        seed: 1101,
        sample_seed: Some(101),
        initial_lr: 0.01,
        l2_lambda: 0.,
        log_every: 1,
        early_stop_patience: 0,
        parallel_batch: false,
        ..Default::default()
    };
    mutate(&mut hp);
    let bake = train_mlp(&mut groups, 4, &hp, &mut Vec::new());
    let model = Model::from_bytes(&bake).unwrap();
    let mut predictor = Predictor::new(&model);
    features
        .iter()
        .map(|row| predictor.predict(&row.map(|v| v as f32)).unwrap()[0])
        .collect()
}

#[test]
fn native_disagreement_updates_toward_equal_predictions() {
    let initial = fit(&[0.5, 0.5], &[0, 0], 0, |_| {});
    let trained = fit(&[0.5, 0.5], &[0, 0], 1, |_| {});
    let before = (initial[0] - initial[1]).abs();
    let after = (trained[0] - trained[1]).abs();
    assert!(before > 0.01, "fixture must begin with unequal predictions");
    assert!(
        after < before * 0.1,
        "disagreement was skipped: {before} -> {after}"
    );
}

#[test]
fn native_vote_preferences_follow_score_polarity_in_both_directions() {
    let a = fit(&[1., 0.], &[0, 0], 1, |_| {});
    let b = fit(&[0., 1.], &[0, 0], 1, |_| {});
    assert!(a[0] > a[1], "score must rise for the preferred side");
    assert!(b[0] < b[1], "reversing votes must reverse preference");
}

#[test]
#[should_panic(expected = "plain sequential uniform trainer")]
fn votes_refuse_unimplemented_parallel_loss() {
    fit(&[1., 0.], &[0, 0], 0, |hp| {
        hp.parallel_batch = true;
        hp.minibatch_size = 2;
    });
}

#[test]
#[should_panic(expected = "plain sequential uniform trainer")]
fn votes_refuse_unimplemented_pool_loss() {
    fit(&[1., 0.], &[0, 0], 0, |hp| hp.pool_head = true);
}

#[test]
#[should_panic(expected = "two complementary sides")]
fn votes_refuse_cross_comparison_pairing() {
    fit(&[1., 0.], &[0, 1], 0, |_| {});
}

#[test]
#[should_panic(expected = "two-vote probability")]
fn votes_refuse_non_native_probability() {
    fit(&[0.75, 0.25], &[0, 0], 0, |_| {});
}
