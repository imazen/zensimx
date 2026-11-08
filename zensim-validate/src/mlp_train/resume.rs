//! Full precision state for the constrained shared-depth CPU owner.
use super::{
    AdamState, MlpHyperparams, StdFeatures, TrainingGroup, input_shaping, resume_store,
    shared_depth,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::{Duration, Instant};

#[derive(Serialize, Deserialize)]
pub(super) struct State {
    pub identity: String,
    pub epoch: usize,
    pub pair: usize,
    pub weights: [Vec<f64>; 4],
    pub adam: AdamState,
    pub depth: Option<shared_depth::Head>,
    pub shapes: Option<input_shaping::State>,
    pub rng: u64,
    pub loss: f64,
    pub steps: u64,
    pub ties: u64,
    pub pending_steps: u64,
    pub best_score_bits: u64,
    pub best_bake: Option<Vec<u8>>,
    pub stale_epochs: usize,
    pub checkpoint_receipts: Vec<String>,
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::{fs, process::Command};
    #[test]
    fn real_trainer_mid_epoch_and_boundary_resume_are_bit_exact() {
        const CHILD: &str = "ZENSIM_RESUME_FIXTURE_TEST_CHILD";
        if let Some(out) = std::env::var_os(CHILD) {
            use crate::mlp_train::{CompactRows, FeatureRows, GroupLossMode};
            let wide = std::env::var("ZENSIM_RESUME_FIXTURE_KIND").as_deref() == Ok("wide");
            let x = (0..32)
                .map(|i| {
                    if wide {
                        (0..2345)
                            .map(|j| ((i * 7 + j * 3) % 31) as f64 / 31.)
                            .collect()
                    } else {
                        vec![(i % 8) as f64 / 8., (i / 8) as f64 / 4.]
                    }
                })
                .collect::<Vec<_>>();
            let rows = x.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let y = x
                .iter()
                .map(|x| 100. - 40. * x[0] - 10. * x[1])
                .collect::<Vec<_>>();
            let mut hp = MlpHyperparams {
                n_hidden_layers: 2,
                n_hidden: 4,
                nonneg_distance: true,
                monotone_cbc: true,
                nonneg_thresholds: true,
                max_pairs: 0,
                init_seed: Some(111),
                sample_seed: Some(222),
                early_stop_patience: 0,
                dump_checkpoints_every: 1,
                dump_checkpoints_dir: Some(
                    std::path::PathBuf::from(&out)
                        .parent()
                        .unwrap()
                        .join("checkpoints"),
                ),
                n_epochs: 3,
                pairs_per_epoch: 100,
                log_every: 1,
                mse_weight: 0.1,
                ..Default::default()
            };
            if std::env::var("ZENSIM_RESUME_FIXTURE_KIND").as_deref() == Ok("changed") {
                hp.initial_lr *= 2.;
            }
            if wide {
                hp.n_hidden = 128;
                hp.max_pairs = 32;
                hp.n_epochs = 2;
                hp.pairs_per_epoch = 8;
            }
            let shaping = matches!(
                std::env::var("ZENSIM_RESUME_FIXTURE_KIND").as_deref(),
                Ok("frozen" | "joint")
            );
            let mut compact = CompactRows {
                data: x.iter().flat_map(|v| [v[0] as f32, v[1] as f32]).collect(),
                n_rows: 32,
                n_features: 720,
                kept: vec![13, 422],
                derived: None,
            };
            if shaping {
                crate::derived_inputs::Fx1Declaration {
                    schema: crate::derived_inputs::FX1_SCHEMA.into(),
                    direct: vec![13],
                    products: vec![[13, 422]],
                    unpaired: vec![],
                }
                .extend_compact_lazy(&mut compact, 720)
                .unwrap();
                let mut mask = vec![false; 721];
                mask[13] = true;
                mask[720] = true;
                *super::super::INPUT_KEEP_MASK.lock().unwrap() = Some(mask);
            }
            let group = TrainingGroup {
                name: "resume_fixture".into(),
                human_scores: &y,
                features: if shaping {
                    FeatureRows::Compact(&mut compact)
                } else {
                    FeatureRows::Borrowed(&rows)
                },
                metric_sigmas: None,
                train_weight: 1.,
                validation_weight: 1.,
                ref_ids: None,
                loss_mode: GroupLossMode::Both,
            };
            fs::create_dir_all(hp.dump_checkpoints_dir.as_ref().unwrap()).unwrap();
            let mut training_log = Vec::new();
            let bytes = crate::mlp_train::train_mlp(
                &mut [group],
                if shaping {
                    721
                } else if wide {
                    2345
                } else {
                    2
                },
                &hp,
                &mut training_log,
            );
            let receipts = training_log
                .iter()
                .filter(|line| line.starts_with("  checkpoint dump: "))
                .collect::<Vec<_>>();
            fs::write(
                std::path::Path::new(&out).with_extension("receipts.json"),
                serde_json::to_vec(&receipts).unwrap(),
            )
            .unwrap();
            fs::write(out, bytes).unwrap();
            return;
        }
        let root = std::env::temp_dir().join(format!("zensim-full-resume-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        for kind in ["plain", "softplus", "frozen", "joint"] {
            let dir = root.join(kind);
            fs::create_dir(&dir).unwrap();
            let manifest = dir.join("head.json");
            fs::write(&manifest,serde_json::to_vec(&serde_json::json!({"schema":"monotone-head-v1","family":"softplus",
                "widths":[4],"first_width":4,"max_pairs":0,"init_seed":111,"sample_seed":222,"epochs":3,"pairs_per_epoch":100,
                "analytic":{"temperature":0.73,"learn_temperature":true,"bias":0.37}})).unwrap()).unwrap();
            let shape_manifest = dir.join("shapes.json");
            fs::write(&shape_manifest,serde_json::to_vec(&serde_json::json!({"schema":"zensim-input-shaping-v1","trainable":kind=="joint",
                "terms":[{"column":13,"damage":13,"reference":null,"separate":false,"knots":[0.25,1.0],"slopes":[0.8,1.2,1.4],"gate":[0.0,0.5,1.0]},
                {"column":720,"damage":13,"reference":422,"separate":true,"knots":[0.25,1.0],"slopes":[1.0,1.2,1.4],"gate":[0.0,0.4,1.0]}]})).unwrap()).unwrap();
            let invoke = |mode: &str, resume: bool, crash: Option<&str>, epoch: bool| {
                let mut command = Command::new(std::env::current_exe().unwrap());
                command.args(["--exact","mlp_train::resume::tests::real_trainer_mid_epoch_and_boundary_resume_are_bit_exact","--nocapture"])
                    .env(CHILD,dir.join(format!("{mode}.bin"))).env("ZENSIM_RESUME_FIXTURE_KIND",kind)
                    .env("ZENSIM_TRAIN_TEST_EVERY_PAIRS","37").env("RAYON_NUM_THREADS","1");
                if mode != "disabled" {
                    command.env(
                        "ZENSIM_TRAIN_SNAPSHOT_ROOT",
                        dir.join(if mode == "uninterrupted" {
                            "base-store"
                        } else if epoch {
                            "epoch-store"
                        } else {
                            "pair-store"
                        }),
                    );
                }
                if resume {
                    command.env("ZENSIM_TRAIN_RESUME", "1");
                }
                if let Some(pair) = crash {
                    command.env(
                        if epoch {
                            "ZENSIM_TRAIN_TEST_CRASH_EPOCH"
                        } else {
                            "ZENSIM_TRAIN_TEST_CRASH_PAIR"
                        },
                        pair,
                    );
                }
                if kind == "softplus" {
                    command
                        .env("ZENSIM_MONOTONE_RESEARCH_MANIFEST", &manifest)
                        .env(
                            "ZENSIM_MONOTONE_RESEARCH_SHA256",
                            crate::train_manifest::sha256_file(&manifest).unwrap(),
                        );
                }
                if matches!(kind, "frozen" | "joint") {
                    command
                        .env("ZENSIM_INPUT_SHAPING_MANIFEST", &shape_manifest)
                        .env(
                            "ZENSIM_INPUT_SHAPING_SHA256",
                            crate::train_manifest::sha256_file(&shape_manifest).unwrap(),
                        );
                }
                let output = command.output().unwrap();
                fs::write(
                    dir.join(format!(
                        "{mode}-{}.stdout",
                        if resume { "resume" } else { "first" }
                    )),
                    &output.stdout,
                )
                .unwrap();
                fs::write(
                    dir.join(format!(
                        "{mode}-{}.stderr",
                        if resume { "resume" } else { "first" }
                    )),
                    &output.stderr,
                )
                .unwrap();
                assert_eq!(
                    output.status.code(),
                    Some(if crash.is_some() { 86 } else { 0 }),
                    "{}",
                    String::from_utf8_lossy(&output.stderr)
                );
            };
            invoke("disabled", false, None, false);
            invoke("uninterrupted", false, None, false);
            let expected = fs::read(dir.join("disabled.bin")).unwrap();
            assert_eq!(expected, fs::read(dir.join("uninterrupted.bin")).unwrap());
            for (mode, epoch) in [("pair", false), ("epoch", true)] {
                invoke(mode, false, Some(if epoch { "1" } else { "37" }), epoch);
                invoke(mode, true, None, epoch);
                assert_eq!(
                    expected,
                    fs::read(dir.join(format!("{mode}.bin"))).unwrap(),
                    "{kind}/{mode}"
                );
                assert_eq!(
                    fs::read(dir.join("uninterrupted.receipts.json")).unwrap(),
                    fs::read(dir.join(format!("{mode}.receipts.json"))).unwrap(),
                    "checkpoint receipt prefix {kind}/{mode}"
                );
                let latest = |store: &str| {
                    let run = fs::read_dir(dir.join(store))
                        .unwrap()
                        .map(Result::unwrap)
                        .find(|e| e.file_type().unwrap().is_dir())
                        .unwrap()
                        .path();
                    let mut bytes = (0..2)
                        .map(|i| fs::read(run.join(format!("state-{i}.snap"))).unwrap())
                        .collect::<Vec<_>>();
                    bytes.sort_by_key(|b| u64::from_le_bytes(b[..8].try_into().unwrap()));
                    bytes.pop().unwrap()[40..].to_vec()
                };
                assert_eq!(
                    latest("base-store"),
                    latest(if epoch { "epoch-store" } else { "pair-store" }),
                    "full state {kind}/{mode}"
                );
            }
        }
    }

    #[test]
    fn full_width_snapshot_footprint_stays_inside_reserved_bytes() {
        let root =
            std::env::temp_dir().join(format!("zensim-full-width-resume-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let output=Command::new(std::env::current_exe().unwrap())
            .args(["--exact","mlp_train::resume::tests::real_trainer_mid_epoch_and_boundary_resume_are_bit_exact","--nocapture"])
            .env("ZENSIM_RESUME_FIXTURE_TEST_CHILD",root.join("final.bin"))
            .env("ZENSIM_RESUME_FIXTURE_KIND","wide")
            .env("ZENSIM_TRAIN_SNAPSHOT_ROOT",root.join("store"))
            .env("RAYON_NUM_THREADS","1").output().unwrap();
        fs::write(root.join("stdout.log"), &output.stdout).unwrap();
        fs::write(root.join("stderr.log"), &output.stderr).unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let run = fs::read_dir(root.join("store"))
            .unwrap()
            .map(Result::unwrap)
            .find(|e| e.file_type().unwrap().is_dir())
            .unwrap()
            .path();
        let bytes = fs::read_dir(&run)
            .unwrap()
            .map(|e| e.unwrap().metadata().unwrap().len())
            .sum::<u64>();
        assert!(bytes <= 64 * 1024 * 1024);
        let states = fs::read_dir(run)
            .unwrap()
            .map(Result::unwrap)
            .filter(|e| e.path().extension().is_some_and(|e| e == "snap"))
            .map(|e| e.metadata().unwrap().len())
            .collect::<Vec<_>>();
        assert_eq!(states.len(), 2);
        assert!(states.iter().sum::<u64>() + states.iter().max().unwrap() <= 64 * 1024 * 1024);
        fs::write(root.join("FOOTPRINT.json"),serde_json::to_vec(&serde_json::json!({"features":2345,"hidden":128,"retained_bytes":bytes,"snapshot_bytes":states,"per_run_ceiling_bytes":64*1024*1024})).unwrap()).unwrap();
    }
}

pub(super) struct Session {
    pub identity: String,
    store: resume_store::Store,
    last: Instant,
    pub loaded: Option<State>,
}

impl Session {
    pub(super) fn open(
        hp: &MlpHyperparams,
        groups: &[TrainingGroup<'_>],
        features: &StdFeatures,
        mean: &[f64],
        scale: &[f64],
    ) -> Option<Self> {
        let root = std::env::var_os("ZENSIM_TRAIN_SNAPSHOT_ROOT")?;
        let mut hash = Sha256::new();
        hash.update(b"zensim-full-state-v1\0");
        hash.update(crate::train_manifest::sha256_file(&std::env::current_exe().unwrap()).unwrap());
        hash.update(format!(
            "{hp:?}{mean:?}{scale:?}{:?}{:?}{:?}",
            super::input_keep_mask(),
            super::l2_feature_mult(),
            super::group_l1_lambda()
        ));
        // Hash effective model inputs, targets and pair membership, including
        // validation rows. Compact feature tables expand one scratch row only.
        let mut scratch = Vec::new();
        for (gi, g) in groups.iter().enumerate() {
            hash.update(format!(
                "{} {:?} {:?} {:?} {:?} {:?}",
                g.name,
                g.loss_mode,
                g.train_weight,
                g.validation_weight,
                g.ref_ids,
                g.metric_sigmas
            ));
            for value in g.human_scores {
                hash.update(value.to_bits().to_le_bytes());
            }
            for row in 0..g.human_scores.len() {
                for value in features.row(gi, row, &mut scratch) {
                    hash.update(value.to_bits().to_le_bytes());
                }
            }
        }
        let mut environment = std::env::vars()
            .filter(|(k, _)| {
                k.starts_with("ZENSIM_")
                    && !k.starts_with("ZENSIM_TRAIN_")
                    && !k.ends_with("_TEST_CHILD")
            })
            .collect::<Vec<_>>();
        environment.sort();
        hash.update(format!("{environment:?}"));
        let identity = hash
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        if std::env::var("ZENSIM_TRAIN_RESUME").as_deref() == Ok("1") {
            assert!(
                std::path::Path::new(&root).join(&identity).is_dir(),
                "no matching snapshot identity; refusing restart before storage admission"
            );
        }
        let store = resume_store::Store::open(std::path::Path::new(&root), &identity)
            .expect("snapshot storage admission");
        let loaded: Option<State> = store.load().expect("intact full-state snapshot");
        if std::env::var("ZENSIM_TRAIN_RESUME").as_deref() == Ok("1") {
            assert!(
                loaded.as_ref().is_some_and(|s| s.identity == identity),
                "no matching full-state snapshot; refusing a restart"
            );
        } else {
            assert!(
                loaded.is_none(),
                "existing state requires explicit ZENSIM_TRAIN_RESUME=1"
            );
        }
        Some(Self {
            identity,
            store,
            last: Instant::now(),
            loaded,
        })
    }
    pub(super) fn due(&self, _pair: usize) -> bool {
        #[cfg(test)]
        if let Ok(n) = std::env::var("ZENSIM_TRAIN_TEST_EVERY_PAIRS") {
            return _pair > 0 && _pair.is_multiple_of(n.parse::<usize>().unwrap());
        }
        self.last.elapsed() >= Duration::from_secs(60)
    }
    pub(super) fn save(&mut self, state: &State) {
        assert_eq!(state.identity, self.identity);
        self.store
            .save(state)
            .expect("bounded atomic snapshot write");
        self.last = Instant::now();
        #[cfg(test)]
        if std::env::var("ZENSIM_TRAIN_TEST_CRASH_EPOCH")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            == Some(state.epoch)
            && state.pair == 0
        {
            std::process::exit(86);
        }
        #[cfg(test)]
        if std::env::var("ZENSIM_TRAIN_TEST_CRASH_PAIR")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            == Some(state.pair)
            && state.pair > 0
        {
            std::process::exit(86);
        }
    }
}
