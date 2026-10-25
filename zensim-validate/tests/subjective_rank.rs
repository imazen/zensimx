//! Invented targets only. Exercise the actual plain CPU trainer and sampler.
use zenpredict::{Model, Predictor};
use zensim_validate::mlp_train::{
    FeatureRows, GroupLossMode, MlpHyperparams, TrainingGroup, train_mlp,
};

fn fit(
    name: &str,
    targets: &[f64],
    mse: f64,
    epochs: usize,
    refs: &[u32],
) -> (Vec<u8>, Vec<String>) {
    let features: Vec<_> = (0..targets.len())
        .map(|i| vec![i as f64 / targets.len() as f64, 1., (i % 3) as f64, 0.])
        .collect();
    let rows: Vec<_> = features.iter().map(Vec::as_slice).collect();
    // A zero-mass absolute leg fixes SCORE polarity just as the inherited
    // teacher legs do, while isolating every measured update to the rank leg.
    let mut groups = [
        TrainingGroup {
            name: name.into(),
            human_scores: targets,
            features: FeatureRows::Borrowed(&rows),
            metric_sigmas: None,
            train_weight: 8.,
            validation_weight: 1.,
            ref_ids: Some(refs),
            loss_mode: GroupLossMode::Rank,
        },
        TrainingGroup {
            name: "inherited_absolute".into(),
            human_scores: targets,
            features: FeatureRows::Borrowed(&rows),
            metric_sigmas: None,
            train_weight: 0.,
            validation_weight: 0.,
            ref_ids: None,
            loss_mode: GroupLossMode::Both,
        },
    ];
    let hp = MlpHyperparams {
        n_hidden: 8,
        n_epochs: epochs,
        pairs_per_epoch: 2000,
        seed: 1101,
        sample_seed: Some(101),
        mse_weight: mse,
        initial_lr: 0.01,
        l2_lambda: 0.,
        log_every: 1,
        early_stop_patience: 0,
        parallel_batch: false,
        ..Default::default()
    };
    let mut log = Vec::new();
    let bake = train_mlp(&mut groups, 4, &hp, &mut log);
    (bake, log)
}
fn predictions(bake: &[u8], n: usize) -> Vec<f32> {
    let model = Model::from_bytes(bake).unwrap();
    let mut predictor = Predictor::new(&model);
    (0..n)
        .map(|i| {
            predictor
                .predict(&[i as f32 / n as f32, 1., (i % 3) as f32, 0.])
                .unwrap()[0]
        })
        .collect()
}
#[test]
fn subjective_exact_ties_have_zero_gradient_and_are_counted() {
    let targets = [-1., -1., -10., -10.];
    let refs = [0, 0, 1, 1];
    let (initial, _) = fit("subjective_rank", &targets, 1., 0, &refs);
    let (trained, log) = fit("subjective_rank", &targets, 1., 1, &refs);
    assert_eq!(predictions(&initial, 4), predictions(&trained, 4));
    let line = log
        .iter()
        .find(|l| l.contains("exact_target_ties="))
        .unwrap();
    let count: usize = line
        .split("exact_target_ties=")
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    assert!(count > 0);
    let (legacy, _) = fit("inherited", &targets, 1., 1, &refs);
    assert_ne!(
        predictions(&legacy, 4),
        predictions(&trained, 4),
        "negative control must detect inherited tie gradients"
    );
}
#[test]
fn subjective_rank_ignores_mse_and_preserves_non_tied_control() {
    let targets: Vec<_> = (0..16).map(|i| -(i as f64)).collect();
    let refs = vec![0; 16];
    let (a, _) = fit("subjective_rank", &targets, 0., 3, &refs);
    let (b, _) = fit("subjective_rank", &targets, 1., 3, &refs);
    let (legacy, _) = fit("inherited", &targets, 1., 3, &refs);
    assert_eq!(predictions(&a, 16), predictions(&b, 16));
    assert_eq!(predictions(&b, 16), predictions(&legacy, 16));
    assert!(
        predictions(&b, 16)[0] > predictions(&b, 16)[15],
        "100-quality distance must rise as invented JND rises"
    );
}
#[test]
fn subjective_pairing_never_uses_between_source_magnitude() {
    let refs = [0, 0, 1, 1];
    let a = [-1., -2., -10., -11.];
    let b = [-1001., -1002., -10., -11.];
    let (first, _) = fit("subjective_rank", &a, 1., 3, &refs);
    let (shifted, _) = fit("subjective_rank", &b, 1., 3, &refs);
    assert_eq!(predictions(&first, 4), predictions(&shifted, 4));
}

#[test]
fn dataset_pair_preferences_use_only_pair_order_and_skip_exact_ties() {
    let refs = [0, 0, 1, 1];
    let targets = [1., 0., 0., 1.];
    let shifted = [1001., 1000., 500., 501.];
    // This helper returns the best validation epoch. One epoch isolates loss
    // equivalence from the native target magnitudes' effect on that monitor;
    // production dataset cells select their fixed final epoch instead.
    let (hard, _) = fit("subjective_rank_fixture", &targets, 1., 1, &refs);
    let (offset, _) = fit("subjective_rank_fixture", &shifted, 1., 1, &refs);
    let (legacy, _) = fit("subjective_rank", &targets, 1., 1, &refs);
    assert_eq!(predictions(&hard, 4), predictions(&offset, 4));
    assert_eq!(predictions(&hard, 4), predictions(&legacy, 4));
    let ties = [0., 0., -10., -10.];
    let (initial, _) = fit("subjective_rank_fixture", &ties, 1., 0, &refs);
    let (trained, _) = fit("subjective_rank_fixture", &ties, 1., 1, &refs);
    assert_eq!(predictions(&initial, 4), predictions(&trained, 4));
}
