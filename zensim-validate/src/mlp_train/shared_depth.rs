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

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct Head {
    pub(super) research: Option<super::monotone_research::Network>,
    pub(super) mid: Vec<f64>,
    pub(super) bias: Vec<f64>,
    pub(super) output: Vec<f64>,
    pub(super) pin: Vec<f64>,
    pub(super) width: usize,
    pub(super) smooth: bool,
    adam: AdamState,
    diagnostics_enabled: bool,
    diagnostics: super::monotone_research::Diagnostics,
    weight_diagnostics: WeightDiagnostics,
    #[serde(skip)]
    backward_pre_gradient: Vec<f64>,
    #[serde(default)]
    output_latent: Option<Vec<f64>>,
}

impl Head {
    pub(super) fn new(hp: &MlpHyperparams, smooth: bool, rng: &mut SplitMix64) -> Self {
        let width = (hp.n_hidden / 2).max(8);
        let mid = (0..hp.n_hidden * width)
            .map(|_| rng.next_normal() * (2. / (hp.n_hidden + width) as f64).sqrt())
            .collect::<Vec<_>>();
        let mut output = (0..width)
            .map(|_| rng.next_normal() * (2. / (width + 1) as f64).sqrt())
            .collect::<Vec<_>>();
        let update = super::optimizer_controls::Config::load()
            .map(|c| c.output_update)
            .unwrap_or_default();
        if update != super::optimizer_controls::OutputUpdate::Projected {
            for w in &mut output {
                *w = -w.abs().max(1e-8);
            }
        }
        let output_latent = (update == super::optimizer_controls::OutputUpdate::NegativeSoftplus)
            .then(|| {
                output
                    .iter()
                    .map(|w| super::optimizer_controls::inverse_softplus(-w))
                    .collect()
            });
        let mut head = Self {
            research: super::monotone_research::Network::from_env(hp, smooth, rng),
            diagnostics_enabled: super::optimizer_controls::diagnostics_enabled(),
            diagnostics: Default::default(),
            weight_diagnostics: Default::default(),
            backward_pre_gradient: Vec::new(),
            output_latent,
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
        if let Some(research) = &self.research {
            return research.forward(first);
        }
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

    pub(super) fn training_pair(
        &mut self,
        a: &[f64],
        b: &[f64],
    ) -> ((f64, Vec<f64>, Vec<f64>), (f64, Vec<f64>, Vec<f64>)) {
        if let Some(research) = &mut self.research {
            research.begin_pair();
            (research.training_forward(a), research.training_forward(b))
        } else {
            (self.forward(a), self.forward(b))
        }
    }

    // Return dL/d(first activation), before either layer's weights are updated.
    pub(super) fn backward(
        &mut self,
        first: &[f64],
        cache: &(f64, Vec<f64>, Vec<f64>),
        dy: f64,
    ) -> Vec<f64> {
        if let Some(research) = &mut self.research {
            return research.backward(first, cache, dy);
        }
        // Reuse pair-local scratch; checkpoint decoding leaves this empty.
        self.backward_pre_gradient.resize(self.width, 0.);
        self.backward_pre_gradient.fill(0.);
        smooth_hinge::backprop_grad_head(
            &cache.1,
            &cache.2,
            dy,
            &mut self.backward_pre_gradient,
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
            for (j, &g) in self.backward_pre_gradient.iter().enumerate() {
                self.adam.gw1[i * self.width + j] += value * g;
                gradient[i] += self.mid[i * self.width + j] * g;
            }
        }
        if self.diagnostics_enabled {
            let stats = &mut self.diagnostics;
            stats.backwards += 1;
            if stats.backwards.is_multiple_of(256) {
                stats.samples += 1;
                stats.first_values += first.len() as u64;
                stats.first_zeros += first.iter().filter(|&&x| x == 0.).count() as u64;
                stats.first_gradient_squared += gradient.iter().map(|v| v * v).sum::<f64>();
                stats.last_active.resize(cache.2.len(), 0);
                for (n, &h) in stats.last_active.iter_mut().zip(&cache.2) {
                    *n += u64::from(h != 0.);
                }
                stats.max_hidden = cache.2.iter().fold(stats.max_hidden, |m, h| m.max(h.abs()));
            }
        }
        gradient
    }

    pub(super) fn step(&mut self, hp: &MlpHyperparams, lr: f64) {
        if let Some(research) = &mut self.research {
            research.step(lr, hp.l2_lambda);
            return;
        }
        if hp.l2_lambda > 0. {
            for (g, w) in self.adam.gw1.iter_mut().zip(&self.mid) {
                *g += hp.l2_lambda * w;
            }
            for (g, w) in self.adam.gw2.iter_mut().zip(&self.output) {
                *g += hp.l2_lambda * w;
            }
        }
        if self.diagnostics_enabled {
            self.diagnostics.optimizer_steps += 1;
            self.diagnostics.output_gradient_squared +=
                self.adam.gw2.iter().map(|g| g * g).sum::<f64>();
        }
        if let Some(latent) = &self.output_latent {
            for (g, q) in self.adam.gw2.iter_mut().zip(latent) {
                *g *= -super::optimizer_controls::softplus_slope(*q);
            }
        }
        self.adam.step(
            &mut self.mid,
            &mut self.bias,
            self.output_latent.as_mut().unwrap_or(&mut self.output),
            &mut self.pin,
            lr,
        );
        if let Some(latent) = &self.output_latent {
            for (w, q) in self.output.iter_mut().zip(latent) {
                *w = -super::optimizer_controls::softplus(*q);
            }
        }
        if self.diagnostics_enabled {
            self.diagnostics.output_projection_hits +=
                self.output.iter().filter(|&&w| w > 0.).count() as u64;
        }
        self.project(hp);
    }

    pub(super) fn observe_first_weights(
        &mut self,
        xa: &[f64],
        pa: &[f64],
        ha: &[f64],
        ga: &[f64],
        xb: &[f64],
        pb: &[f64],
        hb: &[f64],
        gb: &[f64],
        weights: &[f64],
        l2: f64,
    ) {
        if self.diagnostics_enabled {
            self.weight_diagnostics
                .observe(xa, pa, ha, ga, xb, pb, hb, gb, weights, l2);
        }
    }

    pub(super) fn report_diagnostics(&mut self, epoch: usize) {
        if self.diagnostics_enabled {
            self.weight_diagnostics.report(epoch);
        }
        if let Some(research) = &mut self.research {
            research.report_diagnostics(epoch);
            return;
        }
        if !self.diagnostics_enabled {
            return;
        }
        let s = std::mem::take(&mut self.diagnostics);
        println!(
            "TRAINING_DIAGNOSTICS {}",
            serde_json::json!({
                "epoch":epoch,"backwards":s.backwards,"sample_stride":256,"sampled_responses":s.samples,
                "first_zero_activations":s.first_zeros,"first_activation_values":s.first_values,
                "last_unit_active_counts":s.last_active,"last_units":self.width,
                "mean_squared_first_activation_gradient":if s.samples>0 {Some(s.first_gradient_squared/s.samples as f64)} else {None},
                "max_last_activation_in_sample":s.max_hidden,"output_amplitudes":self.output,
                "output_zero_weights":self.output.iter().filter(|&&w|w==0.).count(),
                "output_projection_hits":s.output_projection_hits,"optimizer_steps":s.optimizer_steps,
                "mean_squared_regularized_output_gradient":if s.optimizer_steps>0 {Some(s.output_gradient_squared/s.optimizer_steps as f64)} else {None},
                "note":"sampled training responses; not a timing profile or proof of globally dead units"
            })
        );
    }

    pub(super) fn predict(
        &self,
        features: &StdFeatures,
        shapes: Option<&super::input_shaping::State>,
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
                let x = match shapes {
                    Some(s) => s.row(features, gi, row, &mut scratch),
                    None => features.row(gi, row, &mut scratch),
                };
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

// Observation-only storage: never share the optimizer's gradients or moments.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
struct WeightDiagnostics {
    scratch: Option<AdamState>,
    pairs: u64,
    samples: u64,
    data_squared: f64,
    regularized_squared: f64,
}

impl WeightDiagnostics {
    fn observe(
        &mut self,
        xa: &[f64],
        pa: &[f64],
        ha: &[f64],
        ga: &[f64],
        xb: &[f64],
        pb: &[f64],
        hb: &[f64],
        gb: &[f64],
        weights: &[f64],
        l2: f64,
    ) {
        self.pairs += 1;
        if !self.pairs.is_multiple_of(128) {
            return;
        }
        let adam = self
            .scratch
            .get_or_insert_with(|| AdamState::new(weights.len(), ha.len(), ha.len(), 1));
        assert_eq!(adam.gw1.len(), weights.len());
        adam.gw1.fill(0.);
        adam.gb1.fill(0.);
        first_backward(xa, pa, ha, ga, adam);
        first_backward(xb, pb, hb, gb, adam);
        self.data_squared += adam.gw1.iter().map(|g| g * g).sum::<f64>();
        let multipliers = super::l2_feature_mult();
        super::add_l2_grad_layer1(
            &mut adam.gw1,
            weights,
            l2,
            ha.len(),
            multipliers.as_ref().map(|v| v.as_slice()),
        );
        self.regularized_squared += adam.gw1.iter().map(|g| g * g).sum::<f64>();
        self.samples += 1;
    }

    fn report(&mut self, epoch: usize) {
        println!(
            "FIRST_WEIGHT_DIAGNOSTICS {}",
            serde_json::json!({
                "epoch":epoch,"pair_sample_stride":128,"pairs":self.pairs,"sampled_pairs":self.samples,
                "mean_squared_data_pair_gradient_norm":if self.samples>0 {Some(self.data_squared/self.samples as f64)} else {None},
                "mean_squared_regularized_pair_gradient_norm":if self.samples>0 {Some(self.regularized_squared/self.samples as f64)} else {None},
                "note":"First-layer weight pair-gradient squared norms, not per-parameter means. Canonical unfused backward plus coupled L2; excludes Adam moments, projection and decoupled decay."
            })
        );
        self.pairs = 0;
        self.samples = 0;
        self.data_squared = 0.;
        self.regularized_squared = 0.;
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
    shapes: Option<&super::input_shaping::State>,
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
        let bytes = super::r14_knees::checkpoint(
            knees, mean, scale, w1, b1, w2, b2, nf, nh, no, dtype, transforms, params, activation,
        );
        return match shapes {
            Some(s) => s.stamp(&bytes),
            None => bytes,
        };
    }
    assert!(knees.is_none());
    let bytes = super::bake_plain_znpr_v3(
        mean, scale, w1, b1, w2, b2, nf, nh, no, dtype, transforms, params, activation, depth,
    );
    match shapes {
        Some(s) => s.stamp(&bytes),
        None => bytes,
    }
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

    // Fixed-work allocation probes; run each test separately under heaptrack.
    #[inline(never)]
    fn scratch_allocation_workload(fresh: bool) {
        let mut hp = hp();
        hp.n_hidden = 128;
        hp.max_pairs = 32;
        let mut head = Head::new(&hp, false, &mut SplitMix64::new(12));
        let x = vec![0.5; 128];
        let cache = head.forward(&x);
        for _ in 0..8192 {
            if fresh {
                head.backward_pre_gradient = Vec::new();
            }
            std::hint::black_box(head.backward(std::hint::black_box(&x), &cache, 0.7));
        }
        assert!(std::hint::black_box(head.adam.gw1[0]).is_finite());
    }

    #[test]
    fn scratch_allocations_fresh() {
        scratch_allocation_workload(true);
    }

    #[test]
    fn scratch_allocations_reused() {
        scratch_allocation_workload(false);
    }

    #[test]
    fn backward_scratch_is_reset_and_not_checkpointed() {
        let hp = hp();
        let mut reused = Head::new(&hp, false, &mut SplitMix64::new(12));
        let mut restored: Head =
            serde_json::from_slice(&serde_json::to_vec(&reused).unwrap()).unwrap();
        reused.backward_pre_gradient = vec![91.; reused.width];
        assert!(restored.backward_pre_gradient.is_empty());
        for _ in 0..32 {
            let x = [0., 0.5, 1., 2.];
            let cache = reused.forward(&x);
            assert_eq!(
                reused.backward(&x, &cache, 0.7),
                restored.backward(&x, &cache, 0.7)
            );
            reused.step(&hp, 0.001);
            restored.step(&hp, 0.001);
            assert_eq!(
                serde_json::to_vec(&reused).unwrap(),
                serde_json::to_vec(&restored).unwrap()
            );
            // A fresh scratch allocation is the previous implementation path.
            restored.backward_pre_gradient = Vec::new();
        }
    }

    #[test]
    fn diagnostic_collection_does_not_change_updates() {
        let hp = hp();
        let mut a = Head::new(&hp, false, &mut SplitMix64::new(12));
        let mut b = Head::new(&hp, false, &mut SplitMix64::new(12));
        a.diagnostics_enabled = false;
        b.diagnostics_enabled = true;
        for _ in 0..512 {
            let x = [0., 0.5, 1., 2.];
            let ca = a.forward(&x);
            let cb = b.forward(&x);
            assert_eq!(ca, cb);
            assert_eq!(a.backward(&x, &ca, 0.7), b.backward(&x, &cb, 0.7));
            a.step(&hp, 0.001);
            b.step(&hp, 0.001);
            assert_eq!(a.mid, b.mid);
            assert_eq!(a.bias, b.bias);
            assert_eq!(a.output, b.output);
        }
        assert_eq!(b.diagnostics.samples, 2);
        assert_eq!(b.diagnostics.optimizer_steps, 512);
        assert!(b.diagnostics.output_gradient_squared.is_finite());
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
    fn first_weight_observer_matches_finite_difference_and_coupled_l2() {
        let xa = [0.2, 0.7];
        let xb = [0.8, 0.3];
        let weights = [0.4, -0.2, 0.3, 0.5];
        let bias = [-0.1, -0.15];
        let ga = [0.7, -0.4];
        let gb = [-0.2, 0.6];
        let activation = |x: &[f64], w: &[f64]| {
            let pre: Vec<f64> = (0..2)
                .map(|j| bias[j] + x[0] * w[j] + x[1] * w[2 + j])
                .collect();
            let h: Vec<f64> = pre.iter().map(|v| v.max(0.)).collect();
            (pre, h)
        };
        let objective = |w: &[f64]| {
            let (_, ha) = activation(&xa, w);
            let (_, hb) = activation(&xb, w);
            ha.iter().zip(ga).map(|(h, g)| h * g).sum::<f64>()
                + hb.iter().zip(gb).map(|(h, g)| h * g).sum::<f64>()
        };
        let (pa, ha) = activation(&xa, &weights);
        let (pb, hb) = activation(&xb, &weights);
        let mut observer = WeightDiagnostics::default();
        observer.observe(&xa, &pa, &ha, &ga, &xb, &pb, &hb, &gb, &weights, 0.);
        assert!(observer.scratch.is_none());
        observer.pairs = 127;
        observer.observe(&xa, &pa, &ha, &ga, &xb, &pb, &hb, &gb, &weights, 0.);
        let gradient = observer.scratch.as_ref().unwrap().gw1.clone();
        for i in 0..weights.len() {
            let mut plus = weights;
            let mut minus = weights;
            plus[i] += 1e-6;
            minus[i] -= 1e-6;
            let numerical = (objective(&plus) - objective(&minus)) / 2e-6;
            assert!((gradient[i] - numerical).abs() < 1e-9);
        }
        let mut regularized = gradient.clone();
        let mult = super::super::l2_feature_mult();
        super::super::add_l2_grad_layer1(
            &mut regularized,
            &weights,
            0.01,
            2,
            mult.as_ref().map(|v| v.as_slice()),
        );
        observer.pairs = 127;
        observer.observe(&xa, &pa, &ha, &ga, &xb, &pb, &hb, &gb, &weights, 0.01);
        assert_eq!(observer.scratch.as_ref().unwrap().gw1, regularized);
        assert_eq!(observer.samples, 2);
        let expected = gradient.iter().map(|g| g * g).sum::<f64>();
        assert_eq!(observer.data_squared, 2. * expected);
        assert_eq!(
            observer.regularized_squared,
            expected + regularized.iter().map(|g| g * g).sum::<f64>()
        );
        assert_eq!(observer.scratch.as_ref().unwrap().t, 0);
        assert!(
            observer
                .scratch
                .as_ref()
                .unwrap()
                .mw1
                .iter()
                .all(|&v| v == 0.)
        );
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
