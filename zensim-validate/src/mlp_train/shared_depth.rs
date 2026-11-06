//! Second shared hidden layer for the constrained plain CPU owner.
//! Max-pair coefficients are tied in every intermediate column, so each
//! column sees the same monotone pair sum as the one-hidden-layer model.
#![allow(clippy::too_many_arguments)]
use super::{AdamState, MlpHyperparams, SplitMix64, StdFeatures, smooth_hinge};

pub(super) fn enabled(hp: &MlpHyperparams) -> bool {
    hp.n_hidden_layers == 2
        && hp.nonneg_distance
        && hp.monotone_cbc
        && hp.nonneg_thresholds
        && !hp.per_sample_alpha_head
        && !hp.pool_head
        && !hp.hybrid_head
}

pub(super) fn validate(hp: &MlpHyperparams) {
    assert_eq!(
        hp.out_dtype,
        zenpredict::WeightDtype::F32,
        "shared-depth research currently requires f32 baking"
    );
    assert!(
        hp.minibatch_size == 1
            && hp.norm_in_norm_weight == 0.
            && hp.init_from.is_none()
            && !hp.constant_lr
            && !hp.skip_connection
            && hp.monotone_ramps == 0
            && hp.free_ramps == 0
            && hp.monotone_product_pairs.is_empty()
            && hp.monotone_feature_pin.is_none()
            && hp.steer_pairs.is_none()
            && hp.pooled_legs.is_empty()
            && hp.r14_trainable_knees.is_none(),
        "shared depth requires constrained plain K=1 without auxiliary heads, ramps, transforms or continuation"
    );
    assert!(
        hp.feature_transforms.is_none() && hp.feature_transform_params.is_none(),
        "shared depth does not support input transforms"
    );
}

pub(super) struct Head {
    pub(super) mid: Vec<f64>,
    pub(super) bias: Vec<f64>,
    pub(super) output: Vec<f64>,
    pub(super) pin: Vec<f64>,
    pub(super) width: usize,
    pub(super) smooth: bool,
    adam: AdamState,
}

impl Head {
    pub(super) fn new(hp: &MlpHyperparams, smooth: bool, rng: &mut SplitMix64) -> Self {
        let width = (hp.n_hidden / 2).max(8);
        let mid = (0..hp.n_hidden * width)
            .map(|_| rng.next_normal() * (2. / (hp.n_hidden + width) as f64).sqrt())
            .collect::<Vec<_>>();
        let output = (0..width)
            .map(|_| rng.next_normal() * (2. / (width + 1) as f64).sqrt())
            .collect::<Vec<_>>();
        let mut head = Self {
            adam: AdamState::new(mid.len(), width, width, 1),
            mid,
            bias: vec![0.; width],
            output,
            pin: vec![hp.nonneg_pin],
            width,
            smooth,
        };
        head.project(hp);
        head
    }

    pub(super) fn project(&mut self, hp: &MlpHyperparams) {
        for weight in &mut self.mid {
            *weight = weight.max(0.);
        }
        for pair in 0..hp.max_pairs {
            for column in 0..self.width {
                let u = (2 * pair) * self.width + column;
                let v = u + self.width;
                let tied = (self.mid[u] + self.mid[v]) * 0.5;
                self.mid[u] = tied;
                self.mid[v] = tied;
            }
        }
        for bias in &mut self.bias {
            *bias = bias.min(0.);
        }
        for weight in &mut self.output {
            *weight = weight.min(0.);
        }
        self.pin[0] = hp.nonneg_pin;
    }

    pub(super) fn forward(&self, first: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
        smooth_hinge::forward(
            first,
            &self.mid,
            &self.bias,
            &self.output,
            &self.pin,
            first.len(),
            self.width,
            0.,
            self.smooth,
        )
    }

    // Return dL/d(first activation), before either layer's weights are updated.
    pub(super) fn backward(
        &mut self,
        first: &[f64],
        cache: &(f64, Vec<f64>, Vec<f64>),
        dy: f64,
    ) -> Vec<f64> {
        let mut pre_gradient = vec![0.; self.width];
        smooth_hinge::backprop_grad_head(
            &cache.1,
            &cache.2,
            dy,
            &mut pre_gradient,
            &mut self.adam.gb1,
            &self.output,
            &mut self.adam.gw2,
            &mut self.adam.gb2,
            self.width,
            0.,
            self.smooth,
        );
        let mut gradient = vec![0.; first.len()];
        for (i, &value) in first.iter().enumerate() {
            for (j, &g) in pre_gradient.iter().enumerate() {
                self.adam.gw1[i * self.width + j] += value * g;
                gradient[i] += self.mid[i * self.width + j] * g;
            }
        }
        gradient
    }

    pub(super) fn step(&mut self, hp: &MlpHyperparams, lr: f64) {
        if hp.l2_lambda > 0. {
            for (g, w) in self.adam.gw1.iter_mut().zip(&self.mid) {
                *g += hp.l2_lambda * w;
            }
            for (g, w) in self.adam.gw2.iter_mut().zip(&self.output) {
                *g += hp.l2_lambda * w;
            }
        }
        self.adam.step(
            &mut self.mid,
            &mut self.bias,
            &mut self.output,
            &mut self.pin,
            lr,
        );
        self.project(hp);
    }

    pub(super) fn predict(
        &self,
        features: &StdFeatures,
        gi: usize,
        rows: usize,
        w1: &[f64],
        b1: &[f64],
        scratch_w2: &[f64],
        scratch_b2: &[f64],
    ) -> Vec<f64> {
        let mut scratch = Vec::new();
        (0..rows)
            .map(|row| {
                let x = features.row(gi, row, &mut scratch);
                let (_, _, h) = super::forward(
                    x,
                    w1,
                    b1,
                    scratch_w2,
                    scratch_b2,
                    features.n_features,
                    b1.len(),
                    0.,
                );
                self.forward(&h).0
            })
            .collect()
    }
}

pub(super) fn first_backward(
    x: &[f64],
    pre: &[f64],
    h: &[f64],
    gradient: &[f64],
    adam: &mut AdamState,
) {
    // Reuse the canonical first-layer SIMD backprop. Its output slots are
    // projection scratch on this path, not model parameters; discard only
    // those scratch gradients. The real output gradients live in Head::adam.
    crate::simd_mlp::backprop_step(
        x,
        pre,
        h,
        1.,
        &mut adam.gw1,
        &mut adam.gb1,
        gradient,
        &mut adam.gw2,
        &mut adam.gb2,
        x.len(),
        h.len(),
        0.,
    );
    adam.gw2.fill(0.);
    adam.gb2.fill(0.);
}

#[allow(clippy::too_many_arguments)]
pub(super) fn checkpoint(
    depth: Option<&Head>,
    knees: Option<&super::r14_knees::KneeState>,
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
    transforms: Option<&[zenpredict::FeatureTransform]>,
    params: Option<&[Vec<f32>]>,
    activation: zenpredict::Activation,
) -> Vec<u8> {
    if depth.is_none() {
        return super::r14_knees::checkpoint(
            knees, mean, scale, w1, b1, w2, b2, nf, nh, no, dtype, transforms, params, activation,
        );
    }
    assert!(knees.is_none());
    super::bake_plain_znpr_v3(
        mean, scale, w1, b1, w2, b2, nf, nh, no, dtype, transforms, params, activation, depth,
    )
}

pub(super) fn first_grad_head(
    pre: &[f64],
    h: &[f64],
    gradient: &[f64],
    dpre: &mut [f64],
    adam: &mut AdamState,
) {
    crate::simd_mlp::backprop_grad_head(
        pre,
        h,
        1.,
        dpre,
        &mut adam.gb1,
        gradient,
        &mut adam.gw2,
        &mut adam.gb2,
        h.len(),
        0.,
    );
    adam.gw2.fill(0.);
    adam.gb2.fill(0.);
}

#[cfg(test)]
mod tests {
    use super::super::{FeatureRows, GroupLossMode, TrainingGroup};
    use super::*;

    fn hp() -> MlpHyperparams {
        MlpHyperparams {
            n_hidden_layers: 2,
            n_hidden: 4,
            nonneg_distance: true,
            monotone_cbc: true,
            nonneg_thresholds: true,
            max_pairs: 1,
            ..Default::default()
        }
    }

    fn score(head: &Head, x: &[f64], w: &[f64], b: &[f64]) -> f64 {
        let (_, _, h) =
            super::super::forward(x, w, b, &vec![0.; b.len()], &[0.], x.len(), b.len(), 0.);
        head.forward(&h).0
    }

    #[test]
    fn shared_depth_gradients_cover_every_matrix_and_bias() {
        for smooth in [false, true] {
            let mut head = Head::new(&hp(), smooth, &mut SplitMix64::new(51));
            head.mid.fill(0.08);
            head.bias.fill(-0.03);
            head.output.fill(-0.2);
            let x = [0.4, 0.7];
            let mut w = vec![0.7, -0.2, 0.3, 0.4, 0.2, 0.6, 0.5, -0.1];
            let mut b = vec![-0.1, -0.05, -0.1, -0.2];
            let (_, pre, h) = super::super::forward(&x, &w, &b, &[0.; 4], &[0.], 2, 4, 0.);
            let cache = head.forward(&h);
            let dh = head.backward(&h, &cache, 1.);
            let mut adam = AdamState::new(w.len(), b.len(), b.len(), 1);
            first_backward(&x, &pre, &h, &dh, &mut adam);
            let gradients = [
                adam.gw1.clone(),
                adam.gb1.clone(),
                head.adam.gw1.clone(),
                head.adam.gb1.clone(),
                head.adam.gw2.clone(),
                head.adam.gb2.clone(),
            ];
            for (which, gradient) in gradients.iter().enumerate() {
                for (i, expected) in gradient.iter().enumerate() {
                    let values = match which {
                        0 => &mut w,
                        1 => &mut b,
                        2 => &mut head.mid,
                        3 => &mut head.bias,
                        4 => &mut head.output,
                        _ => &mut head.pin,
                    };
                    let old = values[i];
                    values[i] = old + 1e-5;
                    let plus = score(&head, &x, &w, &b);
                    let values = match which {
                        0 => &mut w,
                        1 => &mut b,
                        2 => &mut head.mid,
                        3 => &mut head.bias,
                        4 => &mut head.output,
                        _ => &mut head.pin,
                    };
                    values[i] = old - 1e-5;
                    let minus = score(&head, &x, &w, &b);
                    let values = match which {
                        0 => &mut w,
                        1 => &mut b,
                        2 => &mut head.mid,
                        3 => &mut head.bias,
                        4 => &mut head.output,
                        _ => &mut head.pin,
                    };
                    values[i] = old;
                    let numeric = (plus - minus) / 2e-5;
                    assert!(
                        (numeric - expected).abs() < 1e-8,
                        "smooth={smooth} matrix={which} index={i} numeric={numeric} analytic={expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn shared_depth_projection_bake_identity_monotonicity_and_runtime_parity() {
        for smooth in [false, true] {
            let hp = hp();
            let mut rng = SplitMix64::new(55);
            let mut head = Head::new(&hp, smooth, &mut rng);
            let mut w = (0..8).map(|_| rng.next_normal()).collect::<Vec<_>>();
            let mut b = vec![-0.1; 4];
            let mut scratch_w2 = vec![-0.2; 4];
            let mut scratch_b2 = vec![100.];
            for round in 0..5 {
                // Exercise the projection after nontrivial optimizer updates.
                let (_, pre, h) =
                    super::super::forward(&[0.3, 0.7], &w, &b, &scratch_w2, &scratch_b2, 2, 4, 0.);
                let f = head.forward(&h);
                let dh = head.backward(&h, &f, if round % 2 == 0 { 1. } else { -1. });
                let mut adam = AdamState::new(8, 4, 4, 1);
                first_backward(&[0.3, 0.7], &pre, &h, &dh, &mut adam);
                adam.step(&mut w, &mut b, &mut scratch_w2, &mut scratch_b2, 0.01);
                head.step(&hp, 0.01);
                super::super::project_plain(
                    &mut w,
                    &mut scratch_w2,
                    &mut b,
                    &mut scratch_b2,
                    Some(100.),
                    &hp,
                );
                for j in 0..head.width {
                    assert_eq!(head.mid[j], head.mid[head.width + j]);
                }
                let bytes = checkpoint(
                    Some(&head),
                    None,
                    &[0.; 2],
                    &[1.; 2],
                    &w,
                    &b,
                    &scratch_w2,
                    &scratch_b2,
                    2,
                    4,
                    1,
                    zenpredict::WeightDtype::F32,
                    None,
                    None,
                    zenpredict::Activation::Relu,
                );
                let model = zenpredict::Model::from_bytes(&bytes).unwrap();
                assert_eq!(
                    model.layer(1).activation,
                    if smooth {
                        zenpredict::Activation::SmoothHinge
                    } else {
                        zenpredict::Activation::Relu
                    }
                );
                assert_eq!(model.layer(2).out_dim, 1);
                let mut predictor = zenpredict::Predictor::new(&model);
                assert_eq!(
                    predictor.predict(&[0., 0.]).unwrap()[0].to_bits(),
                    100f32.to_bits()
                );
                for i in 0..20 {
                    for j in 0..20 {
                        let x = [i as f32 / 8., j as f32 / 8.];
                        let y = predictor.predict(&x).unwrap()[0];
                        assert!(y <= 100.);
                        let expected = score(&head, &[f64::from(x[0]), f64::from(x[1])], &w, &b);
                        assert!(
                            (f64::from(y) - expected).abs()
                                < 8. * f64::from(f32::EPSILON) * expected.abs().max(1.),
                            "smooth={smooth} round={round} x={x:?} native={y} f64={expected}"
                        );
                        for k in 0..2 {
                            let mut greater = x;
                            greater[k] += 0.125;
                            assert!(predictor.predict(&greater).unwrap()[0] <= y + 1e-5);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn shared_depth_fused_first_update_matches_unfused() {
        let xa = [0.2, 0.7];
        let xb = [0.8, 0.3];
        let mut w = vec![0.4, -0.2, 0.3, 0.1, 0.2, 0.4, 0.1, 0.5];
        let mut b = vec![-0.1; 4];
        let mut out = vec![-0.2; 4];
        let mut pin = vec![100.];
        let (mut wf, mut bf, mut of, mut pf) = (w.clone(), b.clone(), out.clone(), pin.clone());
        let mut ordinary = AdamState::new(8, 4, 4, 1);
        let mut fused = AdamState::new(8, 4, 4, 1);
        let mut head = Head::new(&hp(), true, &mut SplitMix64::new(12));
        for _ in 0..5 {
            let (_, pa, ha) = super::super::forward(&xa, &w, &b, &out, &pin, 2, 4, 0.);
            let (_, pb, hb) = super::super::forward(&xb, &w, &b, &out, &pin, 2, 4, 0.);
            let fa = head.forward(&ha);
            let fb = head.forward(&hb);
            let ga = head.backward(&ha, &fa, 0.7);
            let gb = head.backward(&hb, &fb, -0.4);
            first_backward(&xa, &pa, &ha, &ga, &mut ordinary);
            first_backward(&xb, &pb, &hb, &gb, &mut ordinary);
            super::super::add_l2_grad_layer1(&mut ordinary.gw1, &w, 0.01, 4, None);
            ordinary.step(&mut w, &mut b, &mut out, &mut pin, 0.001);
            let mut da = vec![0.; 4];
            let mut db = vec![0.; 4];
            first_grad_head(&pa, &ha, &ga, &mut da, &mut fused);
            first_grad_head(&pb, &hb, &gb, &mut db, &mut fused);
            fused.step_w1_fused(
                &mut wf, &mut bf, &mut of, &mut pf, 0.001, &xa, &da, &xb, &db, 0.01, None, 4, None,
                None, None, None,
            );
            for (a, b) in w
                .iter()
                .chain(&b)
                .chain(&out)
                .chain(&pin)
                .zip(wf.iter().chain(&bf).chain(&of).chain(&pf))
            {
                assert!((a - b).abs() < 1e-12, "fused={b} unfused={a}");
            }
            head.step(&hp(), 0.001);
        }
    }

    #[test]
    fn shared_depth_trains_both_activations_and_refuses_unwired_modes() {
        let x: Vec<Vec<f64>> = (0..32)
            .map(|i| vec![(i % 8) as f64 / 8., (i / 8) as f64 / 4.])
            .collect();
        let y: Vec<f64> = x.iter().map(|v| 100. - 4. * v[0] - 2. * v[1]).collect();
        let refs: Vec<&[f64]> = x.iter().map(Vec::as_slice).collect();
        let mut bakes = Vec::new();
        for smooth in [false, true] {
            let hp = MlpHyperparams {
                smooth_hinge: smooth,
                n_epochs: 3,
                pairs_per_epoch: 50,
                log_every: 1,
                mse_weight: 0.1,
                ..hp()
            };
            let group = TrainingGroup {
                name: "depth_fixture".into(),
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
            assert_eq!(model.layer(1).out_dim, 8);
            assert_eq!(model.layer(2).out_dim, 1);
            let mut predictor = zenpredict::Predictor::new(&model);
            assert_eq!(predictor.predict(&[0., 0.]).unwrap()[0], 100.);
            assert!(predictor.predict(&[1., 1.]).unwrap()[0] < 100.);
            bakes.push(bytes);
        }
        assert_ne!(bakes[0], bakes[1]);
        for hp in [
            MlpHyperparams {
                minibatch_size: 2,
                ..hp()
            },
            MlpHyperparams {
                free_ramps: 1,
                ..hp()
            },
            MlpHyperparams {
                n_hidden_layers: 3,
                ..hp()
            },
            MlpHyperparams {
                skip_connection: true,
                ..hp()
            },
        ] {
            assert!(
                std::panic::catch_unwind(|| super::super::validate_training_capabilities(
                    &hp, true, 0
                ))
                .is_err()
            );
        }
    }
}
