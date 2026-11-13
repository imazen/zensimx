//! Bounded, hash-bound human-row RankNet weights. Sampling stays unchanged.
use super::{MlpHyperparams, TrainingGroup};
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Group {
    name: String,
    scores_sha256: String,
    weights: Vec<f64>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    groups: Vec<Group>,
}
pub(super) struct Plan(Vec<Vec<f64>>);
impl Plan {
    pub(super) fn load(groups: &[TrainingGroup<'_>], hp: &MlpHyperparams) -> Option<Self> {
        let (path, pin) = match (
            std::env::var_os("ZENSIM_HARD_CASE_MANIFEST"),
            std::env::var("ZENSIM_HARD_CASE_SHA256"),
        ) {
            (None, Err(std::env::VarError::NotPresent)) => return None,
            (Some(path), Ok(pin)) => (path, pin),
            _ => panic!("hard-case weights require manifest and independent hash"),
        };
        assert!(
            super::shared_depth::enabled(hp)
                && hp.minibatch_size == 1
                && hp.norm_in_norm_weight == 0.,
            "hard-case weights require shared-depth sequential RankNet"
        );
        assert_eq!(
            crate::train_manifest::sha256_file(std::path::Path::new(&path)).unwrap(),
            pin
        );
        let m: Manifest = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(m.schema, "human-hard-case-rank-weights-v1");
        assert_eq!(m.groups.len(), groups.len());
        let mut weights = Vec::new();
        for (g, binding) in groups.iter().zip(m.groups) {
            assert_eq!(g.name, binding.name);
            assert_eq!(g.human_scores.len(), binding.weights.len());
            let mut h = Sha256::new();
            for &s in g.human_scores {
                assert!(s.is_finite());
                h.update(s.to_le_bytes());
            }
            assert_eq!(
                crate::train_manifest::hex_lower(&h.finalize()),
                binding.scores_sha256
            );
            assert!(
                binding
                    .weights
                    .iter()
                    .all(|w| w.is_finite() && (0.5..=2.).contains(w))
            );
            if !g.name.starts_with("human_") || g.loss_mode.has_mse() {
                assert!(
                    binding.weights.iter().all(|w| *w == 1.),
                    "only human rank-only groups may be reweighted"
                );
            }
            weights.push(binding.weights);
        }
        Some(Self(weights))
    }
    pub(super) fn pair(&self, group: usize, a: usize, b: usize) -> f64 {
        (self.0[group][a] + self.0[group][b]) * 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_pair_weights_preserve_equal_weights_and_swap_symmetry() {
        let p = Plan(vec![vec![1., 1., 0.5, 2.]]);
        assert_eq!(p.pair(0, 0, 1), 1.);
        assert_eq!(p.pair(0, 2, 3), 1.25);
        assert_eq!(p.pair(0, 2, 3), p.pair(0, 3, 2));
    }
    #[test]
    fn native_neutral_weights_preserve_bytes_and_hard_weights_change_fit() {
        const CHILD: &str = "ZENSIM_HARD_CASE_TEST_CHILD";
        if let Some(out) = std::env::var_os(CHILD) {
            use super::super::{FeatureRows, GroupLossMode};
            let x: Vec<Vec<f64>> = (0..32)
                .map(|i| vec![(i % 8) as f64 / 8., (i / 8) as f64 / 4.])
                .collect();
            let y: Vec<f64> = x.iter().map(|v| 100. - 4. * v[0] - 2. * v[1]).collect();
            let refs: Vec<&[f64]> = x.iter().map(Vec::as_slice).collect();
            let group = TrainingGroup {
                name: "human_fixture".into(),
                features: FeatureRows::Borrowed(&refs),
                human_scores: &y,
                metric_sigmas: None,
                train_weight: 1.,
                validation_weight: 1.,
                ref_ids: None,
                loss_mode: GroupLossMode::Rank,
            };
            let hp = MlpHyperparams {
                n_hidden_layers: 2,
                n_hidden: 4,
                nonneg_distance: true,
                monotone_cbc: true,
                nonneg_thresholds: true,
                max_pairs: 1,
                n_epochs: 3,
                pairs_per_epoch: 256,
                log_every: 1,
                mse_weight: 0.,
                ..Default::default()
            };
            let bytes = super::super::train_mlp(&mut [group], 2, &hp, &mut Vec::new());
            std::fs::write(out, bytes).unwrap();
            return;
        }
        let root = std::env::temp_dir().join(format!("zensim-hard-case-{}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let mut h = Sha256::new();
        for i in 0..32 {
            let y = 100. - 4. * ((i % 8) as f64 / 8.) - 2. * ((i / 8) as f64 / 4.);
            h.update(y.to_le_bytes());
        }
        let score_pin = crate::train_manifest::hex_lower(&h.finalize());
        for mode in ["base", "neutral", "hard", "bad_binding"] {
            let path = root.join(format!("{mode}.json"));
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command.args(["--exact","mlp_train::hard_case_weights::tests::native_neutral_weights_preserve_bytes_and_hard_weights_change_fit","--nocapture"])
                .env(CHILD,root.join(format!("{mode}.bin"))).env_remove("ZENSIM_HARD_CASE_MANIFEST").env_remove("ZENSIM_HARD_CASE_SHA256");
            if mode != "base" {
                let weights: Vec<f64> = (0..32)
                    .map(|i| if mode == "hard" && i < 8 { 2. } else { 1. })
                    .collect();
                std::fs::write(&path,serde_json::to_vec(&serde_json::json!({"schema":"human-hard-case-rank-weights-v1",
                    "groups":[{"name":"human_fixture","scores_sha256":if mode=="bad_binding" {"bad"} else {&score_pin},"weights":weights}]})).unwrap()).unwrap();
                command.env("ZENSIM_HARD_CASE_MANIFEST", &path).env(
                    "ZENSIM_HARD_CASE_SHA256",
                    crate::train_manifest::sha256_file(&path).unwrap(),
                );
            }
            let result = command.output().unwrap();
            std::fs::write(root.join(format!("{mode}.stderr.log")), &result.stderr).unwrap();
            assert_eq!(
                result.status.success(),
                mode != "bad_binding",
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
        }
        let base = std::fs::read(root.join("base.bin")).unwrap();
        assert_eq!(base, std::fs::read(root.join("neutral.bin")).unwrap());
        assert_ne!(base, std::fs::read(root.join("hard.bin")).unwrap());
    }
}
