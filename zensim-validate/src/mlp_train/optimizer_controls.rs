//! Private hash-bound controls; no sampling or model-family substitution.
use super::MlpHyperparams;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(super) enum OutputUpdate {
    #[default]
    Projected,
    NegativeAbsInit,
    NegativeSoftplus,
}
pub(super) fn softplus(x: f64) -> f64 {
    x.max(0.) + (-x.abs()).exp().ln_1p()
}
pub(super) fn softplus_slope(x: f64) -> f64 {
    if x >= 0. {
        1. / (1. + (-x).exp())
    } else {
        let e = x.exp();
        e / (1. + e)
    }
}
pub(super) fn inverse_softplus(x: f64) -> f64 {
    if x > 30. {
        x + (-(-x).exp()).ln_1p()
    } else {
        x.exp_m1().ln()
    }
}

pub(super) fn check_admission(hp: &MlpHyperparams) {
    if std::env::var_os("ZENSIM_PLAIN_OPTIMIZER_MANIFEST").is_some()
        || std::env::var_os("ZENSIM_PLAIN_OPTIMIZER_SHA256").is_some()
        || diagnostics_enabled()
    {
        assert!(
            super::shared_depth::enabled(hp),
            "optimizer controls and diagnostics require shared-depth CPU owner"
        );
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Schedule {
    Cosine,
    Constant,
    Consolidate,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Config {
    schema: String,
    pub init_scale: f64,
    pub schedule: Schedule,
    #[serde(default)]
    pub output_update: OutputUpdate,
}
impl Config {
    pub(super) fn load() -> Option<Self> {
        match (
            std::env::var_os("ZENSIM_PLAIN_OPTIMIZER_MANIFEST"),
            std::env::var("ZENSIM_PLAIN_OPTIMIZER_SHA256"),
        ) {
            (None, Err(_)) => None,
            (Some(path), Ok(hash)) => {
                assert_eq!(
                    crate::train_manifest::sha256_file(std::path::Path::new(&path)).unwrap(),
                    hash
                );
                let config: Self = serde_json::from_slice(&std::fs::read(path).unwrap())
                    .expect("plain optimizer manifest");
                assert_eq!(config.schema, "plain-shared-optimizer-v1");
                assert!(config.init_scale.is_finite() && config.init_scale > 0.);
                Some(config)
            }
            _ => panic!("plain optimizer controls require manifest and independent hash"),
        }
    }
    pub(super) fn lr(&self, epoch: usize, hp: &MlpHyperparams) -> f64 {
        match self.schedule {
            Schedule::Cosine => {
                hp.initial_lr
                    * 0.5
                    * (1. + (std::f64::consts::PI * (epoch % 50) as f64 / 50.).cos())
            }
            Schedule::Constant => hp.initial_lr,
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
pub(super) fn diagnostics_enabled() -> bool {
    match std::env::var("ZENSIM_OPTIMIZER_DIAGNOSTICS") {
        Err(std::env::VarError::NotPresent) => false,
        Ok(value) if value == "1" => true,
        _ => panic!("ZENSIM_OPTIMIZER_DIAGNOSTICS must be absent or 1"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn negative_softplus_effective_gradient_and_inverse_are_consistent() {
        for q in [-40., -5., -1., 0., 5., 40.] {
            let h = 1e-5;
            let numerical = -(softplus(q + h) - softplus(q - h)) / (2. * h);
            assert!((numerical + softplus_slope(q)).abs() < 1e-9);
            assert!((inverse_softplus(softplus(q)) - q).abs() < 1e-12);
        }
    }
    #[test]
    fn consolidation_boundary_and_cosine_prefix_are_exact() {
        let mut hp = MlpHyperparams {
            initial_lr: 0.001,
            n_epochs: 240,
            ..Default::default()
        };
        let mut c = Config {
            schema: "plain-shared-optimizer-v1".into(),
            init_scale: 1.,
            schedule: Schedule::Consolidate,
            output_update: OutputUpdate::Projected,
        };
        assert_eq!(c.lr(179, &hp), hp.initial_lr);
        assert_eq!(c.lr(180, &hp), hp.initial_lr * 0.1);
        c.schedule = Schedule::Cosine;
        let long: Vec<_> = (0..120).map(|i| c.lr(i, &hp)).collect();
        hp.n_epochs = 120;
        assert_eq!(long, (0..120).map(|i| c.lr(i, &hp)).collect::<Vec<_>>());
    }
    #[test]
    fn real_plain_trainer_neutral_replay_and_rescue() {
        const CHILD: &str = "ZENSIM_PLAIN_CONTROL_TEST_CHILD";
        if let Some(out) = std::env::var_os(CHILD) {
            use crate::mlp_train::{FeatureRows, GroupLossMode, TrainingGroup};
            let x: Vec<Vec<f64>> = (0..32)
                .map(|i| vec![(i % 8) as f64 / 8., (i / 8) as f64 / 4.])
                .collect();
            let y: Vec<f64> = x.iter().map(|v| 100. - 4. * v[0] - 2. * v[1]).collect();
            let refs: Vec<&[f64]> = x.iter().map(Vec::as_slice).collect();
            let hp = MlpHyperparams {
                n_hidden_layers: 2,
                n_hidden: 4,
                nonneg_distance: true,
                monotone_cbc: true,
                nonneg_thresholds: true,
                max_pairs: 1,
                n_epochs: 4,
                pairs_per_epoch: 256,
                log_every: 1,
                mse_weight: 0.1,
                ..Default::default()
            };
            let group = TrainingGroup {
                name: "control_fixture".into(),
                features: FeatureRows::Borrowed(&refs),
                human_scores: &y,
                metric_sigmas: None,
                train_weight: 1.,
                validation_weight: 1.,
                ref_ids: None,
                loss_mode: GroupLossMode::Both,
            };
            let bytes = super::super::train_mlp(&mut [group], 2, &hp, &mut Vec::new());
            let model = zenpredict::Model::from_bytes(&bytes).unwrap();
            assert_eq!(model.layer(0).activation, zenpredict::Activation::Relu);
            std::fs::write(out, bytes).unwrap();
        } else {
            let root = std::env::temp_dir().join(format!(
                "zensim-plain-control-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir(&root).unwrap();
            for mode in [
                "base",
                "diagnostics",
                "neutral",
                "rescue",
                "abs_init",
                "softplus",
            ] {
                let mut command = std::process::Command::new(std::env::current_exe().unwrap());
                command.args(["--exact","mlp_train::optimizer_controls::tests::real_plain_trainer_neutral_replay_and_rescue","--nocapture"])
                    .env(CHILD,root.join(format!("{mode}.bin"))).env_remove("ZENSIM_PLAIN_OPTIMIZER_MANIFEST")
                    .env_remove("ZENSIM_PLAIN_OPTIMIZER_SHA256").env_remove("ZENSIM_OPTIMIZER_DIAGNOSTICS");
                if mode != "base" {
                    command.env("ZENSIM_OPTIMIZER_DIAGNOSTICS", "1");
                }
                if matches!(mode, "neutral" | "rescue" | "abs_init" | "softplus") {
                    let path = root.join(format!("{mode}.json"));
                    let config = serde_json::json!({"schema":"plain-shared-optimizer-v1","init_scale":if mode=="rescue" {0.5} else {1.},
                        "schedule":if mode=="rescue" {"consolidate"} else {"cosine"},
                        "output_update":match mode {"abs_init"=>"negative_abs_init","softplus"=>"negative_softplus",_=>"projected"}});
                    std::fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();
                    command
                        .env(
                            "ZENSIM_PLAIN_OPTIMIZER_SHA256",
                            crate::train_manifest::sha256_file(&path).unwrap(),
                        )
                        .env("ZENSIM_PLAIN_OPTIMIZER_MANIFEST", path);
                }
                let output = command.output().unwrap();
                std::fs::write(root.join(format!("{mode}.stdout.log")), &output.stdout).unwrap();
                std::fs::write(root.join(format!("{mode}.stderr.log")), &output.stderr).unwrap();
                assert!(
                    output.status.success(),
                    "{}",
                    String::from_utf8_lossy(&output.stderr)
                );
                if mode != "base" {
                    assert!(
                        String::from_utf8_lossy(&output.stdout).contains("TRAINING_DIAGNOSTICS")
                    );
                }
            }
            let base = std::fs::read(root.join("base.bin")).unwrap();
            assert_eq!(base, std::fs::read(root.join("diagnostics.bin")).unwrap());
            assert_eq!(base, std::fs::read(root.join("neutral.bin")).unwrap());
            assert_ne!(base, std::fs::read(root.join("rescue.bin")).unwrap());
            assert_ne!(base, std::fs::read(root.join("abs_init.bin")).unwrap());
            assert_ne!(
                std::fs::read(root.join("abs_init.bin")).unwrap(),
                std::fs::read(root.join("softplus.bin")).unwrap()
            );
        }
    }
}
