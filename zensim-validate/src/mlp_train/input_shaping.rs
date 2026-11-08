//! Hash-bound research shapes over raw compact fields, before standardization.
//! Parameters use the runtime's f32 wire arithmetic; Adam retains f64 state.
use super::{MlpHyperparams, StdFeatures, StdGroup};
use serde::{Deserialize, Serialize};
// Shared private wire implementation; the trainer does not call all runtime
// input-sensitivity helpers, but must use exactly its forward arithmetic.
#[allow(dead_code)]
#[path = "../../../zensim/src/derived_inputs/shaping.rs"]
mod wire;

pub(crate) const KEY: &str = "zentrain.input_shaping";

pub(super) fn check_admission(hp: &MlpHyperparams) {
    if std::env::var_os("ZENSIM_INPUT_SHAPING_MANIFEST").is_some()
        || std::env::var_os("ZENSIM_INPUT_SHAPING_SHA256").is_some()
    {
        assert!(
            super::shared_depth::enabled(hp),
            "input shaping requires shared-depth CPU owner"
        );
        assert_eq!(hp.ema_decay, 0., "input shaping does not support EMA");
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Term {
    pub column: usize,
    pub damage: u32,
    pub reference: Option<u32>,
    pub separate: bool,
    pub knots: [f32; 2],
    pub slopes: [f32; 3],
    pub gate: [f32; 3],
}
impl Term {
    fn shape(&self) -> wire::DamageShape {
        let [a, b] = self.knots;
        let [s0, s1, s2] = self.slopes;
        wire::DamageShape([a, b, s0, s1, s2].map(f32::to_bits))
    }
    fn gate(&self) -> wire::ReferenceGate {
        wire::ReferenceGate(self.gate.map(f32::to_bits))
    }
    pub(crate) fn value(&self, damage: f32, reference: f32) -> f32 {
        if self.separate {
            self.shape().eval(damage) * self.gate().eval(reference)
        } else {
            self.shape().eval(if self.reference.is_some() {
                damage * reference
            } else {
                damage
            })
        }
    }
    /// Derivatives of the real-valued shape at quantized wire parameters.
    /// The product input is quantized by the same f32 multiply as inference.
    fn derivatives(&self, damage: f32, reference: f32) -> [f64; 6] {
        let x = f64::from(if !self.separate && self.reference.is_some() {
            damage * reference
        } else {
            damage
        });
        let [a, b] = self.knots.map(f64::from);
        let basis = [x.min(a), (x - a).max(0.).min(b - a), (x - b).max(0.)];
        let mut out = [0.; 6];
        let factor = if self.separate {
            self.gate().value_slope(f64::from(reference)).0
        } else {
            1.
        };
        for j in 0..3 {
            out[j] = basis[j] * factor;
        }
        if self.separate {
            let f = f64::from(reference).clamp(0., 1.);
            let gate_basis = if f < 0.5 {
                [1. - 2. * f, 2. * f, 0.]
            } else {
                [0., 2. - 2. * f, 2. * f - 1.]
            };
            let shaped = self.shape().value_slope(f64::from(damage)).0;
            for j in 0..3 {
                out[j + 3] = shaped * gate_basis[j];
            }
        }
        out
    }
    pub(crate) fn declaration(&self) -> String {
        let shape = self
            .knots
            .iter()
            .chain(&self.slopes)
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" ");
        match self.reference {
            None => format!("shape {} {shape}", self.damage),
            Some(f) if self.separate => format!(
                "separate {} {f} {shape} {}",
                self.damage,
                self.gate
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
            Some(f) => format!("shape-product {} {f} {shape}", self.damage),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Config {
    pub schema: String,
    pub trainable: bool,
    pub terms: Vec<Term>,
}
impl Config {
    pub(crate) fn validate(&self) {
        assert_eq!(self.schema, "zensim-input-shaping-v1");
        assert!(!self.terms.is_empty(), "shaping needs terms");
        let mut seen = std::collections::BTreeSet::new();
        for t in &self.terms {
            assert!(seen.insert(t.column), "duplicate shaped column");
            assert!(
                !t.separate || t.reference.is_some(),
                "separate gate needs reference"
            );
            assert!(
                t.knots
                    .iter()
                    .chain(&t.slopes)
                    .chain(&t.gate)
                    .all(|v| v.is_finite() && *v >= 0.)
            );
            assert!(t.knots[0] > 0. && t.knots[1] > t.knots[0]);
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub(super) struct State {
    pub config: Config,
    positions: Vec<(usize, Option<usize>)>,
    params: Vec<f64>,
    gradients: Vec<f64>,
    m: Vec<f64>,
    v: Vec<f64>,
    steps: i32,
    initial_manifest_sha256: Option<String>,
}
impl State {
    pub(super) fn load(hp: &MlpHyperparams, features: &StdFeatures) -> Option<Self> {
        let path = std::env::var_os("ZENSIM_INPUT_SHAPING_MANIFEST");
        let hash = std::env::var_os("ZENSIM_INPUT_SHAPING_SHA256");
        let (path, hash) = match (path, hash) {
            (None, None) => return None,
            (Some(p), Some(h)) => (p, h),
            _ => panic!("shaping requires manifest and independent hash"),
        };
        assert!(
            super::shared_depth::enabled(hp),
            "shaping requires shared-depth CPU owner"
        );
        assert_eq!(hp.ema_decay, 0., "shaping does not support EMA");
        assert_eq!(
            crate::train_manifest::sha256_file(std::path::Path::new(&path)).unwrap(),
            hash.to_str().unwrap()
        );
        let config: Config =
            serde_json::from_slice(&std::fs::read(path).unwrap()).expect("shaping manifest");
        let mut state = Self::new(config, features);
        state.initial_manifest_sha256 = Some(hash.to_str().unwrap().to_owned());
        Some(state)
    }
    fn new(config: Config, features: &StdFeatures) -> Self {
        config.validate();
        let l = features
            .lazy
            .as_ref()
            .expect("shaping requires compact raw fields");
        let positions: Vec<_> = config
            .terms
            .iter()
            .map(|t| {
                assert!(t.column < features.n_features && l.kept.contains(&(t.column as u32)));
                assert_eq!(
                    l.mean[t.column], 0.,
                    "shaping requires scale-only standardization"
                );
                let locate = |id| {
                    l.stored_ids
                        .iter()
                        .position(|x| *x == id)
                        .expect("missing raw shaping field")
                };
                let damage = locate(t.damage);
                let reference = t.reference.map(locate);
                let source =
                    &l.sources[l.kept.iter().position(|x| *x as usize == t.column).unwrap()];
                match (source, reference) {
                    (super::ColumnSource::Stored(p), None) => assert_eq!(*p as usize, damage),
                    (super::ColumnSource::Product(a, b), Some(f)) => {
                        assert_eq!((*a as usize, *b as usize), (damage, f))
                    }
                    _ => panic!("shape does not match logical input column"),
                }
                (damage, reference)
            })
            .collect();
        for group in &features.groups {
            let StdGroup::Lazy { raw } = group else {
                panic!("shaping requires all groups compact")
            };
            for row in raw.chunks_exact(l.stored_width) {
                for (t, &(d, f)) in config.terms.iter().zip(&positions) {
                    assert!(row[d].is_finite() && row[d] >= 0., "invalid damage domain");
                    if let Some(f) = f {
                        assert!(
                            row[f].is_finite() && row[f] >= 0.,
                            "invalid reference domain"
                        );
                        if t.separate {
                            assert!(row[f] <= 1., "separate fragility must be in [0,1]");
                        }
                    }
                }
            }
        }
        let params: Vec<f64> = config
            .terms
            .iter()
            .flat_map(|t| t.slopes.into_iter().chain(t.gate).map(f64::from))
            .collect();
        let n = params.len();
        Self {
            config,
            positions,
            params,
            gradients: vec![0.; n],
            m: vec![0.; n],
            v: vec![0.; n],
            steps: 0,
            initial_manifest_sha256: None,
        }
    }
    fn raw<'a>(&self, features: &'a StdFeatures, group: usize, row: usize) -> &'a [f32] {
        let l = features.lazy.as_ref().unwrap();
        let StdGroup::Lazy { raw } = &features.groups[group] else {
            unreachable!()
        };
        &raw[row * l.stored_width..(row + 1) * l.stored_width]
    }
    pub(super) fn row<'a>(
        &self,
        features: &StdFeatures,
        group: usize,
        row: usize,
        scratch: &'a mut Vec<f64>,
    ) -> &'a [f64] {
        // The admitted lazy path always fills scratch, preserving unshaped columns.
        features.row(group, row, scratch);
        let l = features.lazy.as_ref().unwrap();
        let raw = self.raw(features, group, row);
        for (t, &(d, f)) in self.config.terms.iter().zip(&self.positions) {
            scratch[t.column] =
                f64::from(t.value(raw[d], f.map_or(1., |f| raw[f]))) / l.denom[t.column];
        }
        scratch
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn step(
        &mut self,
        features: &StdFeatures,
        group: usize,
        ia: usize,
        ib: usize,
        weights: &[f64],
        da: &[f64],
        db: &[f64],
        lr: f64,
    ) {
        if !self.config.trainable {
            return;
        }
        let a = self.raw(features, group, ia);
        let b = self.raw(features, group, ib);
        let nh = da.len();
        let l = features.lazy.as_ref().unwrap();
        for (j, (t, &(d, f))) in self.config.terms.iter().zip(&self.positions).enumerate() {
            let w = &weights[t.column * nh..(t.column + 1) * nh];
            let ga = w.iter().zip(da).map(|(w, g)| w * g).sum::<f64>() / l.denom[t.column];
            let gb = w.iter().zip(db).map(|(w, g)| w * g).sum::<f64>() / l.denom[t.column];
            let pa = t.derivatives(a[d], f.map_or(1., |f| a[f]));
            let pb = t.derivatives(b[d], f.map_or(1., |f| b[f]));
            for k in 0..6 {
                self.gradients[j * 6 + k] = ga * pa[k] + gb * pb[k];
            }
        }
        assert!(self.gradients.iter().all(|x| x.is_finite()));
        self.steps += 1;
        crate::adam_simd::adam_update(&mut crate::adam_simd::AdamUpdateArgs {
            w: &mut self.params,
            g: &mut self.gradients,
            m: &mut self.m,
            v: &mut self.v,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            bc1: 1. - 0.9_f64.powi(self.steps),
            bc2: 1. - 0.999_f64.powi(self.steps),
            lr,
        });
        assert!(
            self.m.iter().chain(&self.v).all(|x| x.is_finite()),
            "nonfinite shaping Adam state"
        );
        for (j, t) in self.config.terms.iter_mut().enumerate() {
            for k in 0..6 {
                let p = &mut self.params[j * 6 + k];
                assert!(
                    p.is_finite() && (*p as f32).is_finite(),
                    "nonfinite shaping update"
                );
                *p = p.max(0.);
                if k < 3 {
                    t.slopes[k] = *p as f32;
                } else {
                    t.gate[k - 3] = *p as f32;
                }
            }
        }
    }
    pub(super) fn stamp(&self, bytes: &[u8]) -> Vec<u8> {
        let bytes = zenpredict_bake::append_metadata_utf8(
            bytes,
            KEY,
            &serde_json::to_string(&self.config).unwrap(),
        )
        .unwrap();
        match &self.initial_manifest_sha256 {
            Some(hash) => zenpredict_bake::append_metadata_utf8(
                &bytes,
                "zentrain.input_shaping_initial_sha256",
                hash,
            )
            .unwrap(),
            None => bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn term(column: usize, reference: Option<u32>, separate: bool) -> Term {
        Term {
            column,
            damage: 13,
            reference,
            separate,
            knots: [0.25, 1.],
            slopes: [1., 1., 1.],
            gate: [0., 0.5, 1.],
        }
    }
    fn fixture() -> StdFeatures {
        StdFeatures {
            groups: vec![StdGroup::Lazy {
                raw: vec![0.6, 0.4, 0.3, 0.8],
            }],
            n_features: 721,
            lazy: Some(super::super::LazyStd {
                stored_width: 2,
                stored_ids: vec![13, 422],
                kept: vec![13, 720],
                sources: vec![
                    super::super::ColumnSource::Stored(0),
                    super::super::ColumnSource::Product(0, 1),
                ],
                mean: vec![0.; 721],
                denom: vec![1.; 721],
                template: vec![0.; 721],
            }),
        }
    }
    #[test]
    fn threshold_quantiles_use_shaped_train_rows_and_preserve_identity() {
        use super::super::{FeatureRows, GroupLossMode, TrainingGroup, init_thresholds};

        let mut features = fixture();
        features.groups.push(StdGroup::Lazy {
            // Validation rows must never affect TRAIN initialization.
            raw: vec![100., 0.2, 200., 0.9],
        });
        let scaler = features.lazy.as_mut().unwrap();
        scaler.denom[13] = 2.;
        scaler.denom[720] = 4.;
        let config = Config {
            schema: "zensim-input-shaping-v1".into(),
            trainable: false,
            terms: vec![term(13, None, false), term(720, Some(422), true)],
        };
        let identity = State::new(config.clone(), &features);
        let mut shaped_config = config;
        shaped_config.terms[0].slopes = [0., 3., 0.5];
        shaped_config.terms[1].slopes = [2., 0.5, 1.];
        shaped_config.terms[1].gate = [0.2, 0.7, 1.];
        let shaped = State::new(shaped_config, &features);

        let scores = [0., 1.];
        let dummy_row = [0.; 721];
        let rows = [&dummy_row[..], &dummy_row[..]];
        let groups: Vec<_> = (0..2)
            .map(|i| TrainingGroup {
                name: format!("group-{i}"),
                human_scores: &scores,
                features: FeatureRows::Borrowed(&rows),
                metric_sigmas: None,
                train_weight: 1.,
                validation_weight: 1.,
                ref_ids: None,
                loss_mode: GroupLossMode::default(),
            })
            .collect();
        let mut w1 = vec![0.; 721 * 2];
        w1[13 * 2] = 1.;
        w1[720 * 2 + 1] = 1.;
        let materialize = |state: Option<&State>| StdFeatures {
            groups: (0..2)
                .map(|g| {
                    let mut dense = Vec::new();
                    let mut scratch = Vec::new();
                    for r in 0..2 {
                        let x = match state {
                            Some(s) => s.row(&features, g, r, &mut scratch),
                            None => features.row(g, r, &mut scratch),
                        };
                        dense.extend_from_slice(x);
                    }
                    StdGroup::Dense(dense)
                })
                .collect(),
            n_features: 721,
            lazy: None,
        };
        let original_dense = materialize(None);
        let mut shaped_dense = materialize(Some(&shaped));
        for seed in [0, 42, 800] {
            let initialize = |f: &StdFeatures, s: Option<&State>| {
                let mut biases = [0.; 2];
                assert_eq!(
                    init_thresholds(&w1, &mut biases, f, s, &groups, &[0], 2, seed),
                    2
                );
                biases.map(f64::to_bits)
            };
            let baseline = initialize(&features, None);
            assert_eq!(baseline, initialize(&original_dense, None));
            assert_eq!(baseline, initialize(&features, Some(&identity)));
            let actual = initialize(&features, Some(&shaped));
            assert_ne!(
                actual, baseline,
                "fixture must expose unshaped initialization"
            );
            assert_eq!(actual, initialize(&shaped_dense, None));
            let StdGroup::Dense(validation) = &mut shaped_dense.groups[1] else {
                unreachable!()
            };
            validation.fill(1e20);
            assert_eq!(actual, initialize(&shaped_dense, None));
        }
    }
    #[test]
    fn native_identity_frozen_state_and_joint_update() {
        let features = fixture();
        let config = Config {
            schema: "zensim-input-shaping-v1".into(),
            trainable: false,
            terms: vec![term(13, None, false), term(720, Some(422), true)],
        };
        let mut state = State::new(config, &features);
        let mut base = Vec::new();
        let mut shaped = Vec::new();
        assert_eq!(
            features.row(0, 0, &mut base),
            state.row(&features, 0, 0, &mut shaped)
        );
        let before = serde_json::to_string(&state.config).unwrap();
        let mut w = vec![0.; 721];
        w[13] = 1.;
        w[720] = 1.;
        state.step(&features, 0, 0, 1, &w, &[1.], &[1.], 0.01);
        assert_eq!(before, serde_json::to_string(&state.config).unwrap());
        state.config.trainable = true;
        let initial = state.row(&features, 0, 0, &mut shaped)[720];
        for _ in 0..20 {
            state.step(&features, 0, 0, 1, &w, &[1.], &[1.], 0.01);
        }
        assert!(state.row(&features, 0, 0, &mut shaped)[720] < initial);
        assert!(
            state
                .config
                .terms
                .iter()
                .all(|t| t.slopes.iter().chain(&t.gate).all(|v| *v >= 0.))
        );
        for t in &state.config.terms {
            assert_eq!(t.value(0., 0.7), 0.);
        }
    }
    #[test]
    fn manifest_reaches_training_validation_and_every_checkpoint() {
        const CHILD: &str = "ZENSIM_SHAPING_TRAINER_TEST_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let root = std::env::temp_dir().join(format!(
                "zensim-input-shaping-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir(&root).unwrap();
            for trainable in [false, true] {
                let dir = root.join(if trainable { "joint" } else { "frozen" });
                std::fs::create_dir(&dir).unwrap();
                let c = Config {
                    schema: "zensim-input-shaping-v1".into(),
                    trainable,
                    terms: vec![term(13, None, false), term(720, Some(422), true)],
                };
                let path = dir.join("manifest.json");
                std::fs::write(&path, serde_json::to_vec(&c).unwrap()).unwrap();
                let hash = crate::train_manifest::sha256_file(&path).unwrap();
                let output=std::process::Command::new(std::env::current_exe().unwrap())
                    .args(["--exact","mlp_train::input_shaping::tests::manifest_reaches_training_validation_and_every_checkpoint","--nocapture"])
                    .env(CHILD,&dir).env("ZENSIM_INPUT_SHAPING_MANIFEST",&path).env("ZENSIM_INPUT_SHAPING_SHA256",hash)
                    .env_remove("ZENSIM_MONOTONE_RESEARCH_MANIFEST").env_remove("ZENSIM_MONOTONE_RESEARCH_SHA256")
                    .output().unwrap();
                std::fs::write(dir.join("stdout.log"), &output.stdout).unwrap();
                std::fs::write(dir.join("stderr.log"), &output.stderr).unwrap();
                assert!(
                    output.status.success(),
                    "{}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
        } else {
            use crate::mlp_train::{CompactRows, FeatureRows, GroupLossMode, TrainingGroup};
            let dir = std::path::PathBuf::from(std::env::var_os(CHILD).unwrap());
            let initial: Config =
                serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
            let declaration = crate::derived_inputs::Fx1Declaration {
                schema: crate::derived_inputs::FX1_SCHEMA.into(),
                direct: vec![13],
                products: vec![[13, 422]],
                unpaired: vec![],
            };
            let mut compact = CompactRows {
                data: (0..32)
                    .flat_map(|i| [(i % 8) as f32 / 8., (i / 8) as f32 / 4.])
                    .collect(),
                n_rows: 32,
                n_features: 720,
                kept: vec![13, 422],
                derived: None,
            };
            let y = compact
                .data
                .chunks_exact(2)
                .map(|v| 100. - 40. * f64::from(v[0]) - 20. * f64::from(v[0] * v[1]))
                .collect::<Vec<_>>();
            declaration.extend_compact_lazy(&mut compact, 720).unwrap();
            let mut mask = vec![false; 721];
            mask[13] = true;
            mask[720] = true;
            *super::super::INPUT_KEEP_MASK.lock().unwrap() = Some(mask);

            let hp = MlpHyperparams {
                n_hidden_layers: 2,
                n_hidden: 4,
                nonneg_distance: true,
                monotone_cbc: true,
                nonneg_thresholds: true,
                max_pairs: 0,
                init_seed: Some(111),
                sample_seed: Some(222),
                n_epochs: 3,
                pairs_per_epoch: 50,
                log_every: 1,
                mse_weight: 0.1,
                dump_checkpoints_every: 1,
                dump_checkpoints_dir: Some(dir.clone()),
                ..Default::default()
            };
            let group = TrainingGroup {
                name: "shaping_fixture".into(),
                features: FeatureRows::Compact(&mut compact),
                human_scores: &y,
                metric_sigmas: None,
                train_weight: 1.,
                validation_weight: 1.,
                ref_ids: None,
                loss_mode: GroupLossMode::Both,
            };
            let mut log = Vec::new();
            let bytes = crate::mlp_train::train_mlp(&mut [group], 721, &hp, &mut log);
            std::fs::write(dir.join("training.log"), log.join("\n")).unwrap();
            let mut changed = false;
            for bytes in
                std::iter::once(bytes).chain((0..3).map(|epoch| {
                    std::fs::read(dir.join(format!("ckpt_epoch{epoch:03}.bin"))).unwrap()
                }))
            {
                let model = zenpredict::Model::from_bytes(&bytes).unwrap();
                let state: Config =
                    serde_json::from_str(model.metadata().get_utf8(KEY).unwrap()).unwrap();
                changed |= serde_json::to_string(&state.terms).unwrap()
                    != serde_json::to_string(&initial.terms).unwrap();
                let stamped =
                    zenpredict_bake::append_metadata_utf8(&bytes, "zentrain.formula_revision", "5")
                        .unwrap();
                assert!(
                    zensim::BakeScorer::new(&zenpredict::Model::from_bytes(&stamped).unwrap())
                        .is_err()
                );
                declaration.compact_bake(&stamped, 720).unwrap();
            }
            assert_eq!(changed, initial.trainable);
        }
    }

    #[test]
    fn validation_prediction_uses_current_shape_parameters() {
        let features = fixture();
        let mut t = term(720, Some(422), true);
        t.slopes = [2., 2., 2.];
        t.gate = [1., 1., 1.];
        let state = State::new(
            Config {
                schema: "zensim-input-shaping-v1".into(),
                trainable: true,
                terms: vec![t],
            },
            &features,
        );
        let hp = MlpHyperparams {
            n_hidden: 4,
            max_pairs: 0,
            ..Default::default()
        };
        let mut head = super::super::shared_depth::Head::new(
            &hp,
            false,
            &mut super::super::SplitMix64::new(9),
        );
        head.mid.fill(1.);
        head.bias.fill(0.);
        head.output.fill(-1.);
        head.pin.fill(100.);
        let mut w = vec![0.; 721 * 4];
        w[13 * 4..14 * 4].fill(1.);
        w[720 * 4..721 * 4].fill(1.);
        let predictions =
            head.predict(&features, Some(&state), 0, 2, &w, &[0.; 4], &[0.; 4], &[0.]);
        let unshaped = head.predict(&features, None, 0, 2, &w, &[0.; 4], &[0.; 4], &[0.]);
        for row in 0..2 {
            let mut scratch = Vec::new();
            let x = state.row(&features, 0, row, &mut scratch);
            assert!((predictions[row] - (100. - 32. * (x[13] + x[720]))).abs() < 1e-12);
            assert_ne!(predictions[row], unshaped[row]);
        }
    }

    #[test]
    fn parameter_gradients_match_wire_central_differences() {
        for (reference, separate) in [(None, false), (Some(422), false), (Some(422), true)] {
            let mut t = term(13, reference, separate);
            t.slopes = [0.7, 1.3, 0.4];
            t.gate = [0.2, 0.8, 0.6];
            for (d, f) in [(0.1, 0.2), (0.6, 0.7), (2., 0.4)] {
                let analytical = t.derivatives(d, f);
                for k in 0..6 {
                    let mut plus = t.clone();
                    let mut minus = t.clone();
                    if k < 3 {
                        plus.slopes[k] += 0.001;
                        minus.slopes[k] -= 0.001;
                    } else {
                        plus.gate[k - 3] += 0.001;
                        minus.gate[k - 3] -= 0.001;
                    }
                    let numerical = f64::from(plus.value(d, f) - minus.value(d, f)) / 0.002;
                    assert!(
                        (analytical[k] - numerical).abs() < 0.0004,
                        "mode {reference:?}/{separate} param {k}: {} vs {numerical}",
                        analytical[k]
                    );
                }
            }
        }
    }
    #[test]
    fn trained_shape_checkpoint_compacts_and_serves_identically() {
        let features = fixture();
        let mut state = State::new(
            Config {
                schema: "zensim-input-shaping-v1".into(),
                trainable: true,
                terms: vec![term(13, None, false), term(720, Some(422), true)],
            },
            &features,
        );
        let mut weights = vec![0.; 721];
        weights[13] = 1.;
        weights[720] = 2.;
        for _ in 0..8 {
            state.step(&features, 0, 0, 1, &weights, &[-1.], &[0.5], 0.01);
        }
        let spec = serde_json::json!({"schema_hash":1,"scaler_mean":vec![0.;721],"scaler_scale":vec![1.;721],
            "metadata":[{"key":"zentrain.formula_revision","type":"utf8","text":"5"}],
            "layers":[{"in_dim":721,"out_dim":1,"activation":"relu","dtype":"f32","weights":weights,"biases":[0.]},
                {"in_dim":1,"out_dim":1,"activation":"identity","dtype":"f32","weights":[-1.],"biases":[100.]}]});
        let bytes = zenpredict_bake::bake_from_json_str(&spec.to_string()).unwrap();
        let stamped = state.stamp(&bytes);
        let declaration = crate::derived_inputs::Fx1Declaration {
            schema: crate::derived_inputs::FX1_SCHEMA.into(),
            direct: vec![13],
            products: vec![[13, 422]],
            unpaired: vec![],
        };
        let compact = declaration.compact_bake(&stamped, 720).unwrap();
        let model = zenpredict::Model::from_bytes(&compact).unwrap();
        assert!(
            model
                .metadata()
                .get_utf8(crate::derived_inputs::DERIVED_INPUTS_KEY)
                .unwrap()
                .starts_with("zensim-derived-inputs research-shaping-v1\n")
        );
        let mut scorer = zensim::BakeScorer::new(&model).unwrap();
        let mut row = vec![0.; 720];
        row[422] = 0.7;
        assert_eq!(scorer.score_features(&row, 96, 96, None).unwrap(), 100.);
        row[13] = 0.6;
        row[422] = 0.4;
        let mut scratch = Vec::new();
        let shaped = state.row(&features, 0, 0, &mut scratch);
        let original = zenpredict::Model::from_bytes(&bytes).unwrap();
        let mut predictor = zenpredict::Predictor::new(&original);
        let expected = predictor
            .predict(&shaped.iter().map(|x| *x as f32).collect::<Vec<_>>())
            .unwrap()[0];
        assert_eq!(
            scorer.score_features(&row, 96, 96, None).unwrap(),
            f64::from(expected)
        );
    }
}
