//! R14 T2: tied-softness soft-clip knees, optimized in log space.
use super::{MlpHyperparams, StdFeatures};
use std::sync::Arc;
use zenpredict::FeatureTransform;

/// Pre-transform caller values for T2; labels are never stored here.
#[derive(Clone)]
pub struct KneePlan {
    columns: Vec<usize>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    intervals: Vec<[f64; 2]>,
    groups: Vec<Vec<f32>>,
    train: Vec<bool>,
    fixed_intervals: Vec<usize>,
}

impl std::fmt::Debug for KneePlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KneePlan")
            .field("columns", &self.columns)
            .field("fixed_intervals", &self.fixed_intervals)
            .field(
                "group_rows",
                &self
                    .groups
                    .iter()
                    .map(|g| g.len() / self.columns.len())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl KneePlan {
    /// Capture selected raw columns before the ordinary transform pass. The
    /// statistics are the frozen TRAIN records produced by the R14 screen.
    pub fn capture(
        transforms: &[FeatureTransform],
        params: &[Vec<f32>],
        statistics: &serde_json::Value,
        groups: &[(&[f64], bool)],
    ) -> Result<Self, String> {
        super::transforms::validate(transforms, Some(params))?;
        let records = statistics
            .as_array()
            .ok_or("T2 statistics must be an array")?;
        let mut columns = Vec::new();
        let mut lower = Vec::new();
        let mut upper = Vec::new();
        let mut intervals = Vec::new();
        let mut fixed_intervals = Vec::new();
        let nf = transforms.len();
        let mut seen = vec![false; nf];
        for r in records {
            let c = r["trainer_column"]
                .as_u64()
                .ok_or("T2 missing trainer column")? as usize;
            if c >= nf {
                return Err("T2 column outside trainer width".into());
            }
            if seen[c] {
                return Err("T2 duplicate statistics column".into());
            }
            seen[c] = true;
            if transforms[c] != FeatureTransform::SoftClip {
                continue;
            }
            if params[c][0].to_bits() != params[c][1].to_bits() {
                return Err("T2 requires soft=knee".into());
            }
            let lo = r["p05"].as_f64().ok_or("T2 missing p05")?;
            let hi = r["p99"].as_f64().ok_or("T2 missing p99")?;
            if !lo.is_finite() || !hi.is_finite() || hi <= lo.max(0.) {
                fixed_intervals.push(c);
                continue;
            }
            let k = f64::from(params[c][0]);
            if k < lo.max(0.) || k > hi {
                return Err("T2 initial knee outside TRAIN bounds".into());
            }
            columns.push(c);
            lower.push(if lo > 0. { lo.ln() } else { f64::NEG_INFINITY });
            upper.push(hi.ln());
            intervals.push([lo.max(0.), hi]);
        }
        if columns.is_empty() {
            return Err("T2 has no eligible soft-clip knees".into());
        }
        if transforms
            .iter()
            .enumerate()
            .any(|(c, t)| *t == FeatureTransform::SoftClip && !seen[c])
        {
            return Err("T2 missing soft-clip TRAIN statistics".into());
        }
        let mut raw = Vec::new();
        let mut train = Vec::new();
        for &(flat, active) in groups {
            if !flat.len().is_multiple_of(nf) {
                return Err("T2 raw group width differs".into());
            }
            let values: Vec<f32> = flat
                .chunks_exact(nf)
                .flat_map(|row| columns.iter().map(move |&c| row[c] as f32))
                .collect();
            if values.iter().any(|v| !v.is_finite()) {
                return Err("T2 nonfinite raw caller value".into());
            }
            raw.push(values);
            train.push(active);
        }
        if !train.iter().any(|v| *v) {
            return Err("T2 requires TRAIN rows".into());
        }
        Ok(Self {
            columns,
            lower,
            upper,
            intervals,
            groups: raw,
            train,
            fixed_intervals,
        })
    }
}

pub(super) struct KneeState {
    plan: Arc<KneePlan>,
    pub(super) params: Vec<Vec<f32>>,
    log_knee: Vec<f64>,
    m: Vec<f64>,
    v: Vec<f64>,
    gradients: Vec<f64>,
    steps: i32,
    mean: Vec<f64>,
    scale: Vec<f64>,
}

impl KneeState {
    pub(super) fn new(
        hp: &MlpHyperparams,
        mean: &[f64],
        scale: &[f64],
        rows: &[(usize, bool)],
    ) -> Option<Self> {
        let plan = hp.r14_trainable_knees.as_ref()?.clone();
        assert_eq!(rows.len(), plan.groups.len(), "T2 group inventory differs");
        for ((n, active), (g, train)) in rows.iter().zip(plan.groups.iter().zip(&plan.train)) {
            assert_eq!(*n * plan.columns.len(), g.len(), "T2 raw row count differs");
            assert_eq!(active, train, "T2 TRAIN role differs");
        }
        let params = hp
            .feature_transform_params
            .clone()
            .expect("T2 parameter table");
        let log_knee = plan
            .columns
            .iter()
            .map(|&c| f64::from(params[c][0]).ln())
            .collect();
        let n = plan.columns.len();
        Some(Self {
            plan,
            params,
            log_knee,
            m: vec![0.; n],
            v: vec![0.; n],
            gradients: vec![0.; n],
            steps: 0,
            mean: mean.to_vec(),
            scale: scale.to_vec(),
        })
    }

    fn raw(&self, group: usize, row: usize, slot: usize) -> f64 {
        f64::from(self.plan.groups[group][row * self.plan.columns.len() + slot])
    }

    pub(super) fn row<'a>(
        &self,
        features: &StdFeatures,
        group: usize,
        row: usize,
        scratch: &'a mut Vec<f64>,
    ) -> &'a [f64] {
        let base = features.row_cow(group, row);
        scratch.clear();
        scratch.extend_from_slice(&base);
        for (j, &c) in self.plan.columns.iter().enumerate() {
            let y = FeatureTransform::SoftClip
                .apply_with_params(self.raw(group, row, j) as f32, &self.params[c]);
            assert!(y.is_finite(), "T2 nonfinite transformed caller value");
            scratch[c] = (f64::from(y) - self.mean[c]) / self.scale[c].max(1e-12);
        }
        scratch
    }

    /// For t(x;k,k)=sign(x)*k*(1+ln(abs(x)/k)) in the tail,
    /// dt/dlog(k)=sign(x)*k*ln(abs(x)/k). It is zero in the identity core.
    fn derivative(raw: f64, knee: f64) -> f64 {
        if raw.abs() <= knee {
            0.
        } else {
            raw.signum() * knee * (raw.abs() / knee).ln()
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn step(
        &mut self,
        group: usize,
        ia: usize,
        ib: usize,
        w1: &[f64],
        dh_a: &[f64],
        dh_b: &[f64],
        lr: f64,
    ) {
        self.steps += 1;
        let bc1 = 1. - 0.9_f64.powi(self.steps);
        let bc2 = 1. - 0.999_f64.powi(self.steps);
        let nh = dh_a.len();
        assert_eq!(nh, dh_b.len());
        for (j, &c) in self.plan.columns.iter().enumerate() {
            let row = &w1[c * nh..(c + 1) * nh];
            let da: f64 = row.iter().zip(dh_a).map(|(w, g)| w * g).sum();
            let db: f64 = row.iter().zip(dh_b).map(|(w, g)| w * g).sum();
            let k = f64::from(self.params[c][0]);
            let gradient = (da * Self::derivative(self.raw(group, ia, j), k)
                + db * Self::derivative(self.raw(group, ib, j), k))
                / self.scale[c].max(1e-12);
            assert!(gradient.is_finite(), "T2 nonfinite knee gradient");
            self.gradients[j] = gradient;
        }
        crate::adam_simd::adam_update(&mut crate::adam_simd::AdamUpdateArgs {
            w: &mut self.log_knee,
            g: &mut self.gradients,
            m: &mut self.m,
            v: &mut self.v,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            bc1,
            bc2,
            lr,
        });
        for (j, &c) in self.plan.columns.iter().enumerate() {
            assert!(
                self.m[j].is_finite() && self.v[j].is_finite(),
                "T2 nonfinite Adam state"
            );
            let proposal = self.log_knee[j].clamp(self.plan.lower[j], self.plan.upper[j]);
            let mut k = proposal.exp() as f32;
            assert!(
                proposal.is_finite() && k.is_finite() && k > 0.,
                "T2 invalid projected knee"
            );
            // Quantize inward at a nonrepresentable percentile boundary.
            // Nearest-f32 rounding can otherwise put a valid log projection
            // outside its TRAIN interval and make continuation reject it.
            let [lo, hi] = self.plan.intervals[j];
            if f64::from(k) < lo {
                k = k.next_up();
            } else if f64::from(k) > hi {
                k = k.next_down();
            }
            assert!(
                k.is_finite() && k > 0. && f64::from(k) >= lo && f64::from(k) <= hi,
                "T2 TRAIN interval has no representable projected knee"
            );
            self.log_knee[j] = proposal;
            self.params[c][0] = k;
            self.params[c][1] = k;
        }
    }

    /// Refit post-transform TRAIN scales for serialization, then compensate
    /// encoder rows by their positive scale ratio. The real-valued network is
    /// unchanged; the bake has the usual f64-to-f32 parameter rounding.
    pub(super) fn bake_scales_weights(&self, w1: &[f64], nh: usize) -> (Vec<f64>, Vec<f64>) {
        let n = self.plan.columns.len();
        let mut count = 0usize;
        let mut mean = vec![0.; n];
        for (gi, raw) in self.plan.groups.iter().enumerate() {
            if !self.plan.train[gi] {
                continue;
            }
            for row in raw.chunks_exact(n) {
                for (j, &c) in self.plan.columns.iter().enumerate() {
                    let value =
                        FeatureTransform::SoftClip.apply_with_params(row[j], &self.params[c]);
                    assert!(value.is_finite(), "T2 nonfinite post-transform TRAIN value");
                    mean[j] += f64::from(value);
                }
                count += 1;
            }
        }
        assert!(count > 0);
        for m in &mut mean {
            *m /= count as f64;
        }
        let mut var = vec![0.; n];
        for (gi, raw) in self.plan.groups.iter().enumerate() {
            if !self.plan.train[gi] {
                continue;
            }
            for row in raw.chunks_exact(n) {
                for (j, &c) in self.plan.columns.iter().enumerate() {
                    let y = f64::from(
                        FeatureTransform::SoftClip.apply_with_params(row[j], &self.params[c]),
                    );
                    var[j] += (y - mean[j]).powi(2);
                }
            }
        }
        let mut scales = self.scale.clone();
        let mut weights = w1.to_vec();
        for (j, &c) in self.plan.columns.iter().enumerate() {
            assert_eq!(self.mean[c], 0., "T2 requires scale-only standardization");
            scales[c] = (var[j] / count as f64).sqrt().max(1e-8);
            assert!(
                scales[c].is_finite(),
                "T2 nonfinite post-transform TRAIN scale"
            );
            let ratio = scales[c] / self.scale[c].max(1e-12);
            for w in &mut weights[c * nh..(c + 1) * nh] {
                *w *= ratio;
            }
        }
        (scales, weights)
    }
}

impl StdFeatures {
    pub(super) fn row_with_knees<'a>(
        &'a self,
        knees: Option<&KneeState>,
        group: usize,
        row: usize,
        scratch: &'a mut Vec<f64>,
    ) -> &'a [f64] {
        match knees {
            Some(k) => k.row(self, group, row, scratch),
            None => self.row(group, row, scratch),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn predict_with_knees(
        &self,
        knees: Option<&KneeState>,
        gi: usize,
        rows: usize,
        w1: &[f64],
        b1: &[f64],
        w2: &[f64],
        b2: &[f64],
        nh: usize,
        alpha: f64,
        smooth: bool,
    ) -> Vec<f64> {
        let Some(k) = knees else {
            return self.predict(gi, rows, w1, b1, w2, b2, nh, alpha, smooth);
        };
        let mut scratch = Vec::new();
        (0..rows)
            .map(|i| {
                let x = k.row(self, gi, i, &mut scratch);
                super::smooth_hinge::forward(x, w1, b1, w2, b2, self.n_features, nh, alpha, smooth)
                    .0
            })
            .collect()
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn checkpoint(
    knees: Option<&KneeState>,
    mean: &[f64],
    scale: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    nf: usize,
    nh: usize,
    no: usize,
    dtype: zenpredict::WeightDtype,
    transforms: Option<&[FeatureTransform]>,
    params: Option<&[Vec<f32>]>,
    activation: zenpredict::Activation,
) -> Vec<u8> {
    if let Some(k) = knees {
        let (scales, weights) = k.bake_scales_weights(w1, nh);
        super::bake_two_layer_znpr_v3(
            mean,
            &scales,
            &weights,
            b1,
            w2,
            b2,
            nf,
            nh,
            no,
            dtype,
            transforms,
            Some(&k.params),
            activation,
        )
    } else {
        super::bake_two_layer_znpr_v3(
            mean, scale, w1, b1, w2, b2, nf, nh, no, dtype, transforms, params, activation,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlp_train::{FeatureRows, GroupLossMode, TrainingGroup};

    #[test]
    fn log_knee_derivative_matches_independent_finite_differences() {
        let value = |x: f64, log_k: f64| {
            let k = log_k.exp();
            if x.abs() <= k {
                x
            } else {
                x.signum() * k * (1. + (x.abs() / k).ln())
            }
        };
        for k in [0.25_f64, 1., 4.] {
            for x in [-7_f64, -1., -0.25, 0., 0.25, 1., 7.] {
                let eps = 1e-5;
                let fd = (value(x, k.ln() + eps) - value(x, k.ln() - eps)) / (2. * eps);
                assert!(
                    (fd - KneeState::derivative(x, k)).abs() < 2e-5,
                    "x={x} k={k} fd={fd}"
                );
                let native = f64::from(
                    FeatureTransform::SoftClip.apply_with_params(x as f32, &[k as f32, k as f32]),
                );
                assert!((native - value(x, k.ln())).abs() < 1e-6);
            }
        }
    }

    fn state(lower: f64) -> KneeState {
        let raw = [4., 8.];
        let plan = KneePlan::capture(
            &[FeatureTransform::SoftClip],
            &[vec![0.5, 0.5]],
            &serde_json::json!([{"trainer_column":0,"p05":lower,"p99":0.75}]),
            &[(&raw, true)],
        )
        .unwrap();
        let hp = MlpHyperparams {
            r14_trainable_knees: Some(Arc::new(plan)),
            feature_transform_params: Some(vec![vec![0.5, 0.5]]),
            ..Default::default()
        };
        KneeState::new(&hp, &[0.], &[1.], &[(2, true)]).unwrap()
    }

    #[test]
    fn log_projection_honors_both_bounds_and_refuses_zero_without_an_epsilon() {
        let mut lower = state(0.25);
        lower.step(0, 0, 1, &[1.; 4], &[1.; 4], &[1.; 4], 1000.);
        assert_eq!(lower.params[0], vec![0.25, 0.25]);
        let mut upper = state(0.25);
        upper.step(0, 0, 1, &[1.; 4], &[-1.; 4], &[-1.; 4], 1000.);
        assert_eq!(upper.params[0], vec![0.75, 0.75]);
        let mut zero_lower = state(0.);
        assert_eq!(zero_lower.plan.lower[0], f64::NEG_INFINITY);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                zero_lower.step(0, 0, 1, &[1.; 4], &[1.; 4], &[1.; 4], 1000.);
            }))
            .is_err()
        );
    }

    #[test]
    fn projection_rounds_inward_at_nonrepresentable_percentile_boundaries() {
        let mut lower = state(0.25);
        let lo = f64::from(0.25_f32) + f64::from(0.25_f32.next_up() - 0.25) / 4.;
        Arc::make_mut(&mut lower.plan).lower[0] = lo.ln();
        Arc::make_mut(&mut lower.plan).intervals[0][0] = lo;
        lower.step(0, 0, 1, &[1.; 4], &[1.; 4], &[1.; 4], 1000.);
        assert_eq!(lower.params[0][0], 0.25_f32.next_up());
        assert!(f64::from(lower.params[0][0]) >= lo);

        let mut upper = state(0.25);
        let hi = f64::from(0.75_f32) - f64::from(0.75 - 0.75_f32.next_down()) / 4.;
        Arc::make_mut(&mut upper.plan).upper[0] = hi.ln();
        Arc::make_mut(&mut upper.plan).intervals[0][1] = hi;
        upper.step(0, 0, 1, &[1.; 4], &[-1.; 4], &[-1.; 4], 1000.);
        assert_eq!(upper.params[0][0], 0.75_f32.next_down());
        assert!(f64::from(upper.params[0][0]) <= hi);
    }

    #[test]
    fn checkpoint_scale_refit_preserves_the_learned_network_function() {
        let mut state = state(0.25);
        let w1 = [0.25, 0.5, 0.75, 1.];
        let w2 = [-1., -2., -3., -4.];
        state.step(0, 0, 1, &w1, &[1.; 4], &[1.; 4], 0.1);
        let (scales, weights) = state.bake_scales_weights(&w1, 4);
        let bytes = checkpoint(
            Some(&state),
            &[0.],
            &[1.],
            &w1,
            &[0.; 4],
            &w2,
            &[100.],
            1,
            4,
            1,
            zenpredict::WeightDtype::F32,
            Some(&[FeatureTransform::SoftClip]),
            Some(&[vec![0.5, 0.5]]),
            zenpredict::Activation::Relu,
        );
        let model = zenpredict::Model::from_bytes(&bytes).unwrap();
        assert_eq!(model.feature_transform_params().unwrap(), state.params);
        let mut predictor = zenpredict::Predictor::new(&model);
        for raw in [0_f32, 0.1, 4., 8.] {
            let y = f64::from(FeatureTransform::SoftClip.apply_with_params(raw, &state.params[0]));
            let before = 100. + w1.iter().zip(w2).map(|(w, o)| y * w * o).sum::<f64>();
            let after = 100.
                + weights
                    .iter()
                    .zip(w2)
                    .map(|(w, o)| y / scales[0] * w * o)
                    .sum::<f64>();
            assert!((after - before).abs() < 1e-12);
            // Four f32 hidden accumulators and the output reduction round on
            // serialization/inference, as for every native trainer checkpoint.
            assert!(
                (f64::from(predictor.predict_transformed(&[raw]).unwrap()[0]) - before).abs()
                    < 3e-5
            );
        }
    }

    #[test]
    fn native_training_learns_knees_and_exports_current_train_only_scaling() {
        let raw: Vec<f64> = [0., 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
            .into_iter()
            .flat_map(|x| [x, x * 0.3])
            .collect();
        let val_raw = [1000., 100., 2000., 400.];
        let transforms = vec![FeatureTransform::SoftClip, FeatureTransform::SoftSign];
        let params = vec![vec![0.5, 0.5], vec![1.]];
        let plan = KneePlan::capture(
            &transforms,
            &params,
            &serde_json::json!([{"trainer_column":0,"p05":0.1,"p99":2.},
                {"trainer_column":1,"p05":0.,"p99":400.}]),
            &[(&raw, true), (&val_raw, false)],
        )
        .unwrap();
        let shape = |flat: &[f64]| {
            flat.chunks_exact(2)
                .map(|row| {
                    (0..2)
                        .map(|c| {
                            f64::from(transforms[c].apply_with_params(row[c] as f32, &params[c]))
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let rows = shape(&raw);
        let val_rows = shape(&val_raw);
        let views: Vec<_> = rows.iter().map(Vec::as_slice).collect();
        let val_views: Vec<_> = val_rows.iter().map(Vec::as_slice).collect();
        let targets: Vec<_> = raw
            .chunks_exact(2)
            .map(|r| {
                100. - 10.
                    * f64::from(
                        FeatureTransform::SoftClip.apply_with_params(r[0] as f32, &[0.2, 0.2]),
                    )
            })
            .collect();
        let val_targets = [40., 30.];
        let mut groups = [
            TrainingGroup {
                name: "invented TRAIN".into(),
                features: FeatureRows::Borrowed(&views),
                human_scores: &targets,
                metric_sigmas: None,
                train_weight: 1.,
                validation_weight: 0.,
                ref_ids: None,
                loss_mode: GroupLossMode::Both,
            },
            TrainingGroup {
                name: "validation only".into(),
                features: FeatureRows::Borrowed(&val_views),
                human_scores: &val_targets,
                metric_sigmas: None,
                train_weight: 0.,
                validation_weight: 1.,
                ref_ids: None,
                loss_mode: GroupLossMode::Both,
            },
        ];
        let hp = MlpHyperparams {
            n_hidden: 4,
            max_pairs: 1,
            nonneg_distance: true,
            monotone_cbc: true,
            nonneg_thresholds: true,
            n_epochs: 3,
            pairs_per_epoch: 64,
            log_every: 1,
            initial_lr: 0.005,
            mse_weight: 0.1,
            early_stop_patience: 0,
            r14_transforms_registration: Some(super::super::transforms::REGISTRATION.into()),
            feature_transforms: Some(transforms),
            feature_transform_params: Some(params),
            r14_trainable_knees: Some(Arc::new(plan)),
            ..Default::default()
        };
        let bytes = super::super::train_mlp(&mut groups, 2, &hp, &mut Vec::new());
        let model = zenpredict::Model::from_bytes(&bytes).unwrap();
        let final_params = model.feature_transform_params().unwrap();
        let knee = final_params[0][0];
        assert!(knee != 0.5 && (0.1..=2.).contains(&knee), "knee={knee}");
        assert_eq!(final_params[0], vec![knee, knee]);
        assert_eq!(final_params[1], vec![1.]);
        let transformed: Vec<_> = raw
            .chunks_exact(2)
            .map(|r| {
                f64::from(
                    FeatureTransform::SoftClip.apply_with_params(r[0] as f32, &final_params[0]),
                )
            })
            .collect();
        let mean = transformed.iter().sum::<f64>() / transformed.len() as f64;
        let scale = (transformed.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / transformed.len() as f64)
            .sqrt();
        assert!((f64::from(model.scaler_scale()[0]) - scale).abs() < 1e-6);
        assert_eq!(model.scaler_mean(), &[0., 0.]);
        assert!(super::super::certify_monotone_bake(&model, 100.).is_ok());
        let mut predictor = zenpredict::Predictor::new(&model);
        assert_eq!(
            predictor.predict_transformed(&[0., 0.]).unwrap()[0].to_bits(),
            100_f32.to_bits()
        );
    }
}
