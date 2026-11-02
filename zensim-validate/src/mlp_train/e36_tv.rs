//! External feature-pool hinge: the private E36 protocol and, in public builds, `--tv-pool`.
//! No optimizer calls: scheduled gradients join the existing primary Adam boundary. The E36 build
//! asserts its registered constants; a public build takes them from the pool's configuration.
use super::{AdamState, LevelPool, MlpHyperparams, SplitMix64, TvRegularizer, smooth_hinge};

pub(super) struct Pool<'a> {
    config: &'a TvRegularizer,
    mean_weight: f64,
    columns: Vec<usize>,
    mean: Vec<f64>,
    scale: Vec<f64>,
    /// `--level-pool` targets: the reference checkpoint's raw output on each level row, computed on first use.
    level_targets: std::cell::OnceCell<Vec<f64>>,
}

impl<'a> Pool<'a> {
    pub(super) fn new(config: &'a TvRegularizer, h: &MlpHyperparams) -> Self {
        assert!(h.nonneg_distance && h.minibatch_size == 1);
        assert!(!h.per_sample_alpha_head && !h.hybrid_head && !h.pool_head);
        if cfg!(e36_tv) {
            assert_eq!(config.band_weights, Some([2.0, 1.0, 0.0, 0.0]));
            assert_eq!(config.apply_every, 50);
            assert_eq!(config.batch, 32);
            assert_eq!(config.margin, 0.05);
            assert!([0.0, 1.0, 10.0].contains(&config.weight));
        }
        assert!(config.apply_every > 0 && config.batch > 0);
        assert!(!config.pairs.is_empty());
        let classes = config
            .band_id
            .as_ref()
            .expect("E36 pair weight classes required");
        let weights = config.band_weights.expect("pool class weights required");
        assert_eq!(classes.len(), config.pairs.len());
        assert!(classes.iter().all(|&c| weights[c as usize] > 0.0));
        assert!(
            config
                .pairs
                .iter()
                .all(|&(a, b)| a < config.features.len() && b < config.features.len() && a != b)
        );
        let mean_weight =
            classes.iter().map(|&c| weights[c as usize]).sum::<f64>() / classes.len() as f64;
        let mask = super::input_keep_mask().expect("E36 frozen input keep mask required");
        let columns: Vec<_> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &keep)| keep.then_some(i))
            .collect();
        assert!(
            config
                .features
                .iter()
                .all(|r| r.len() == columns.len() && r.iter().all(|v| v.is_finite()))
        );
        Self {
            config,
            mean_weight,
            columns,
            mean: Vec::new(),
            scale: Vec::new(),
            level_targets: std::cell::OnceCell::new(),
        }
    }

    pub(super) fn bind_scaler(&mut self, mean: &[f64], scale: &[f64]) {
        self.mean = self.columns.iter().map(|&i| mean[i]).collect();
        self.scale = self.columns.iter().map(|&i| scale[i].max(1e-12)).collect();
        assert!(self.mean.iter().chain(&self.scale).all(|v| v.is_finite()));
    }

    #[cfg(test)]
    fn standardize(&mut self, mean: &[f64], scale: &[f64]) -> Vec<f64> {
        self.bind_scaler(mean, scale);
        self.config
            .features
            .iter()
            .flat_map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(k, &v)| (v - self.mean[k]) / self.scale[k])
            })
            .collect()
    }

    pub(super) fn scheduled(&self, primary_steps: u64) -> bool {
        self.config.weight > 0.0
            && primary_steps > 0
            && primary_steps.is_multiple_of(self.config.apply_every as u64)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn accumulate(
        &self,
        rng: &mut SplitMix64,
        w1: &[f64],
        b1: &[f64],
        w2: &[f64],
        b2: &[f64],
        adam: &mut AdamState,
        nf: usize,
        nh: usize,
        h: &MlpHyperparams,
        polarity: f64,
    ) -> f64 {
        let mut loss = 0.0;
        let mut xlo = vec![0.0; nf];
        let mut xhi = vec![0.0; nf];
        assert_eq!(
            self.mean.len(),
            self.columns.len(),
            "E36 base scaler must be bound"
        );
        for _ in 0..self.config.batch {
            // Rejection sampling avoids modulo bias while retaining a TV-only
            // RNG stream; no primary sampler or scaler state is consumed.
            let n = self.config.pairs.len() as u64;
            let limit = u64::MAX - (u64::MAX % n);
            let draw = loop {
                let x = rng.next_u64();
                if x < limit {
                    break x;
                }
            };
            let i = (draw % n) as usize;
            let (lo, hi) = self.config.pairs[i];
            for (k, &i) in self.columns.iter().enumerate() {
                xlo[i] = (self.config.features[lo][k] - self.mean[k]) / self.scale[k];
                xhi[i] = (self.config.features[hi][k] - self.mean[k]) / self.scale[k];
                assert!(xlo[i].is_finite() && xhi[i].is_finite());
            }
            let (yl, hlp, hl, yh, hhp, hh) = smooth_hinge::forward_pair(
                &xlo,
                &xhi,
                w1,
                b1,
                w2,
                b2,
                nf,
                nh,
                h.leaky_alpha,
                h.smooth_hinge,
            );
            let violation = polarity * (yh - yl) + self.config.margin;
            let pair_weight = self.config.band_weights.unwrap()
                [self.config.band_id.as_ref().unwrap()[i] as usize];
            let scale =
                self.config.weight * pair_weight / (self.config.batch as f64 * self.mean_weight);
            loss += scale * violation.max(0.0);
            if violation > 0.0 {
                for (x, hp, hv, sign) in [(&xhi, &hhp, &hh, polarity), (&xlo, &hlp, &hl, -polarity)]
                {
                    smooth_hinge::backprop_step(
                        x,
                        hp,
                        hv,
                        scale * sign,
                        w1,
                        &mut adam.gw1,
                        &mut adam.gb1,
                        w2,
                        &mut adam.gw2,
                        &mut adam.gb2,
                        nf,
                        nh,
                        h.leaky_alpha,
                        h.smooth_hinge,
                    );
                }
            }
        }
        if let Some(level) = &self.config.level {
            loss += self.accumulate_level(level, rng, w1, b1, w2, b2, adam, nf, nh, h);
        }
        loss
    }

    /// The reference checkpoint's raw output on every level row, each row standardized with the reference's own
    /// scaler exactly as that model saw its inputs.
    fn level_reference_targets(&self, level: &LevelPool, nf: usize, nh: usize, h: &MlpHyperparams) -> Vec<f64> {
        let keep = super::input_keep_mask();
        let r = super::warm_start::InitialWeights::load(&level.reference, nf, h, keep.as_deref())
            .unwrap_or_else(|e| panic!("--level-pool reference: {e}"));
        assert_eq!(r.b1.len(), nh, "--level-pool reference hidden width");
        let mut x = vec![0.0; nf];
        level
            .features
            .iter()
            .map(|row| {
                for (k, &c) in self.columns.iter().enumerate() {
                    x[c] = (row[k] - r.scaler_mean[c]) / r.scaler_scale[c].max(1e-12);
                    assert!(x[c].is_finite());
                }
                let (y, ..) = smooth_hinge::forward_pair(
                    &x, &x, &r.w1, &r.b1, &r.w2, &r.b2, nf, nh, h.leaky_alpha, h.smooth_hinge,
                );
                assert!(y.is_finite());
                y
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn accumulate_level(
        &self,
        level: &LevelPool,
        rng: &mut SplitMix64,
        w1: &[f64],
        b1: &[f64],
        w2: &[f64],
        b2: &[f64],
        adam: &mut AdamState,
        nf: usize,
        nh: usize,
        h: &MlpHyperparams,
    ) -> f64 {
        let targets = self
            .level_targets
            .get_or_init(|| self.level_reference_targets(level, nf, nh, h));
        let n = level.features.len() as u64;
        let limit = u64::MAX - (u64::MAX % n);
        let scale = level.weight / level.batch as f64;
        let mut x = vec![0.0; nf];
        let mut loss = 0.0;
        for _ in 0..level.batch {
            let draw = loop {
                let v = rng.next_u64();
                if v < limit {
                    break v;
                }
            };
            let i = (draw % n) as usize;
            for (k, &c) in self.columns.iter().enumerate() {
                x[c] = (level.features[i][k] - self.mean[k]) / self.scale[k];
            }
            let (y, pre, hid, ..) =
                smooth_hinge::forward_pair(&x, &x, w1, b1, w2, b2, nf, nh, h.leaky_alpha, h.smooth_hinge);
            let r = y - targets[i];
            loss += scale * r * r;
            smooth_hinge::backprop_step(
                &x,
                &pre,
                &hid,
                2.0 * scale * r,
                w1,
                &mut adam.gw1,
                &mut adam.gb1,
                w2,
                &mut adam.gw2,
                &mut adam.gb2,
                nf,
                nh,
                h.leaky_alpha,
                h.smooth_hinge,
            );
        }
        loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn level_pool_gradient_matches_finite_difference() {
        // Invented rows and targets; the reference checkpoint is replaced by preset targets.
        let previous = super::super::INPUT_KEEP_MASK
            .lock()
            .unwrap()
            .replace(vec![true, false, true]);
        let h = MlpHyperparams {
            nonneg_distance: true,
            minibatch_size: 1,
            n_hidden: 4,
            ..Default::default()
        };
        let config = |weight: f64, level_weight: f64| TvRegularizer {
            pairs: vec![(0, 1), (0, 2)],
            features: vec![vec![0.2, 0.4], vec![0.3, 0.7], vec![0.6, 0.1]],
            weight,
            apply_every: 50,
            batch: 8,
            band_id: Some(vec![0, 1]),
            band_weights: Some([2., 1., 0., 0.]),
            margin: 0.05,
            pool: true,
            level: Some(LevelPool {
                features: vec![vec![0.5, 0.2], vec![0.1, 0.9], vec![0.7, 0.4], vec![0.3, 0.3]],
                reference: Vec::new(),
                weight: level_weight,
                batch: 16,
            }),
        };
        let mut w1: Vec<f64> = (0..12).map(|i| 0.05 + 0.03 * i as f64).collect();
        let mut b1 = vec![0.1, -0.2, 0.05, 0.0];
        let mut w2 = vec![-0.3, -0.1, -0.2, -0.05];
        let mut b2 = vec![0.4];
        let evaluate = |w1: &[f64], b1: &[f64], w2: &[f64], b2: &[f64], lw: f64| {
            let c = config(0.5, lw);
            let mut p = Pool::new(&c, &h);
            p.bind_scaler(&[0.1, 9., 0.2], &[2., 7., 4.]);
            p.level_targets.set(vec![0.2, -0.1, 0.35, 0.0]).unwrap();
            let mut a = AdamState::new(12, 4, 4, 1);
            let mut r = SplitMix64::new(36);
            let loss = p.accumulate(&mut r, w1, b1, w2, b2, &mut a, 3, 4, &h, 1.);
            assert_eq!(a.t, 0, "gradient admission must never call Adam");
            (loss, [a.gw1, a.gb1, a.gw2, a.gb2].concat())
        };
        let (l1, g1) = evaluate(&w1, &b1, &w2, &b2, 1.);
        let (l0, g0) = evaluate(&w1, &b1, &w2, &b2, 1e-300);
        assert!(l1 > l0, "the level term adds loss");
        assert!(g1.iter().zip(&g0).any(|(a, b)| a != b), "the level term adds gradient");
        let n = w1.len() + b1.len() + w2.len() + b2.len();
        fn nudge(j: usize, delta: f64, slices: [&mut Vec<f64>; 4]) {
            let mut k = j;
            for slice in slices {
                if k < slice.len() {
                    slice[k] += delta;
                    return;
                }
                k -= slice.len();
            }
        }
        for j in 0..n {
            nudge(j, 1e-6, [&mut w1, &mut b1, &mut w2, &mut b2]);
            let (lp, _) = evaluate(&w1, &b1, &w2, &b2, 1.);
            nudge(j, -2e-6, [&mut w1, &mut b1, &mut w2, &mut b2]);
            let (lm, _) = evaluate(&w1, &b1, &w2, &b2, 1.);
            nudge(j, 1e-6, [&mut w1, &mut b1, &mut w2, &mut b2]);
            assert!(((lp - lm) / 2e-6 - g1[j]).abs() < 1e-8, "parameter {j}: fd {} vs {}", (lp - lm) / 2e-6, g1[j]);
        }
        *super::super::INPUT_KEEP_MASK.lock().unwrap() = previous;
    }

    #[test]
    fn e36_preupdate_gradient_objective_and_inactive_control() {
        // Invented feature vectors only. No training group, optimizer call,
        // label load, bake or scaler fit occurs in this numerical admission.
        let previous = super::super::INPUT_KEEP_MASK
            .lock()
            .unwrap()
            .replace(vec![true, false, true]);
        let h = MlpHyperparams {
            nonneg_distance: true,
            minibatch_size: 1,
            n_hidden: 4,
            ..Default::default()
        };
        let config = |weight| TvRegularizer {
            pairs: vec![(0, 1), (0, 2)],
            features: vec![vec![0.2, 0.4], vec![0.3, 0.7], vec![0.6, 0.1]],
            weight,
            apply_every: 50,
            batch: 32,
            band_id: Some(vec![0, 1]),
            band_weights: Some([2., 1., 0., 0.]),
            margin: 0.05,
            pool: true,
            level: None,
        };
        let c = config(1.);
        let mut pool = Pool::new(&c, &h);
        let rows = pool.standardize(&[0.1, 9., 0.2], &[2., 7., 4.]);
        assert_eq!(
            rows,
            vec![
                0.05,
                0.05,
                0.09999999999999999,
                0.12499999999999999,
                0.25,
                -0.025
            ]
        );
        assert_eq!(pool.mean_weight, 1.5);
        assert!(!pool.scheduled(0));
        assert!(!pool.scheduled(49));
        assert!(pool.scheduled(50));
        assert!(!pool.scheduled(51));
        let mut w1 = vec![0.2; 12];
        let mut b1 = vec![0.3; 4];
        let mut w2 = vec![0.001; 4];
        let mut b2 = vec![0.];
        let evaluate = |w1: &[f64], b1: &[f64], w2: &[f64], b2: &[f64], lambda, polarity| {
            let c = config(lambda);
            let mut p = Pool::new(&c, &h);
            p.bind_scaler(&[0.1, 9., 0.2], &[2., 7., 4.]);
            let mut a = AdamState::new(12, 4, 4, 1);
            let mut r = SplitMix64::new(36);
            let loss = p.accumulate(&mut r, w1, b1, w2, b2, &mut a, 3, 4, &h, polarity);
            assert_eq!(a.t, 0, "gradient admission must never call Adam");
            (loss, [a.gw1, a.gb1, a.gw2, a.gb2].concat())
        };
        for polarity in [1., -1.] {
            let (_, g) = evaluate(&w1, &b1, &w2, &b2, 1., polarity);
            let (l10, g10) = evaluate(&w1, &b1, &w2, &b2, 10., polarity);
            let (l1, _) = evaluate(&w1, &b1, &w2, &b2, 1., polarity);
            assert!((l10 - 10. * l1).abs() < 1e-12);
            for (a, b) in g.iter().zip(g10) {
                assert!((10. * a - b).abs() < 1e-12);
            }
            for j in 0..g.len() {
                let mut perturb = |delta| {
                    let slices = [&mut w1, &mut b1, &mut w2, &mut b2];
                    let mut k = j;
                    for slice in slices {
                        if k < slice.len() {
                            slice[k] += delta;
                            break;
                        }
                        k -= slice.len();
                    }
                };
                perturb(1e-6);
                let (lp, _) = evaluate(&w1, &b1, &w2, &b2, 1., polarity);
                // End the temporary mutable borrow before evaluating again.
                let slices = [&mut w1, &mut b1, &mut w2, &mut b2];
                let mut k = j;
                for slice in slices {
                    if k < slice.len() {
                        slice[k] -= 2e-6;
                        break;
                    }
                    k -= slice.len();
                }
                let (lm, _) = evaluate(&w1, &b1, &w2, &b2, 1., polarity);
                let slices = [&mut w1, &mut b1, &mut w2, &mut b2];
                let mut k = j;
                for slice in slices {
                    if k < slice.len() {
                        slice[k] += 1e-6;
                        break;
                    }
                    k -= slice.len();
                }
                assert!(((lp - lm) / 2e-6 - g[j]).abs() < 1e-9, "parameter {j}");
            }
        }
        let c0 = config(0.);
        let p0 = Pool::new(&c0, &h);
        let mut tv_rng = SplitMix64::new(77);
        let primary = SplitMix64::new(88);
        let tv_before = tv_rng.state;
        let primary_before = primary.state;
        for step in 0..151 {
            if p0.scheduled(step) {
                let mut a = AdamState::new(12, 4, 4, 1);
                p0.accumulate(&mut tv_rng, &w1, &b1, &w2, &b2, &mut a, 3, 4, &h, 1.);
            }
        }
        assert_eq!(tv_rng.state, tv_before);
        assert_eq!(primary.state, primary_before);
        // Real primary gradient boundary, with the same forward/head owners
        // used by the fused branch, and no optimizer invocation.
        let xa = [0.1, 0., 0.3];
        let xb = [0.2, 0., 0.4];
        let (_, hap, ha, _, hbp, hb) = smooth_hinge::forward_pair(
            &xa,
            &xb,
            &w1,
            &b1,
            &w2,
            &b2,
            3,
            4,
            h.leaky_alpha,
            h.smooth_hinge,
        );
        let mut plain = AdamState::new(12, 4, 4, 1);
        let mut head = AdamState::new(12, 4, 4, 1);
        for (x, hp, hv, d) in [(&xa, &hap, &ha, 0.7), (&xb, &hbp, &hb, -0.2)] {
            smooth_hinge::backprop_step(
                x,
                hp,
                hv,
                d,
                &w1,
                &mut plain.gw1,
                &mut plain.gb1,
                &w2,
                &mut plain.gw2,
                &mut plain.gb2,
                3,
                4,
                h.leaky_alpha,
                h.smooth_hinge,
            );
        }
        let mut dha = vec![0.; 4];
        let mut dhb = vec![0.; 4];
        for (hp, hv, d, dh) in [(&hap, &ha, 0.7, &mut dha), (&hbp, &hb, -0.2, &mut dhb)] {
            smooth_hinge::backprop_grad_head(
                hp,
                hv,
                d,
                dh,
                &mut head.gb1,
                &w2,
                &mut head.gw2,
                &mut head.gb2,
                4,
                h.leaky_alpha,
                h.smooth_hinge,
            );
        }
        for i in 0..3 {
            for j in 0..4 {
                head.gw1[i * 4 + j] = xa[i] * dha[j] + xb[i] * dhb[j];
            }
        }
        let gradient = |a: &AdamState| [&a.gw1[..], &a.gb1, &a.gw2, &a.gb2].concat();
        let primary_gradient = gradient(&plain);
        for (a, b) in primary_gradient.iter().zip(gradient(&head)) {
            assert!((a - b).abs() < 1e-14);
        }
        let (_, tv_gradient) = evaluate(&w1, &b1, &w2, &b2, 1., 1.);
        pool.accumulate(
            &mut SplitMix64::new(36),
            &w1,
            &b1,
            &w2,
            &b2,
            &mut plain,
            3,
            4,
            &h,
            1.,
        );
        for ((p, t), combined) in primary_gradient
            .iter()
            .zip(tv_gradient)
            .zip(gradient(&plain))
        {
            assert!((p + t - combined).abs() < 1e-12);
        }
        assert_eq!(plain.t, 0);
        assert_eq!(head.t, 0);
        // Mixed active/inactive hinges keep the fixed full-pool denominator.
        let mut mixed = config(1.);
        mixed.features = vec![vec![0., 0.], vec![1., 0.], vec![-1., 0.]];
        let mut pm = Pool::new(&mixed, &h);
        let _rm = pm.standardize(&[0.; 3], &[1.; 3]);
        let mut active = 0;
        let mut expected_rng = SplitMix64::new(36);
        for _ in 0..32 {
            if expected_rng.next_u64() % 2 == 0 {
                active += 1;
            }
        }
        assert!(active > 0 && active < 32);
        let mut a = AdamState::new(12, 4, 4, 1);
        let mixed_loss = pm.accumulate(
            &mut SplitMix64::new(36),
            &[0.2; 12],
            &[0.3; 4],
            &[1.; 4],
            &[0.],
            &mut a,
            3,
            4,
            &h,
            1.,
        );
        assert!((mixed_loss - active as f64 * 2. * 0.85 / (32. * 1.5)).abs() < 1e-12);
        assert_eq!(a.t, 0);
        // Exercise the actual primary sampler, including collisions and
        // unusable groups; TV draws never enter its stream. Counters reset
        // at each epoch exactly as the plain owner does.
        use super::super::sampling::{Draw, PairDrawCtx, draw_pair};
        let cdf = [0.1, 1.];
        let counts = [1, 4];
        let cdfs = [None, None];
        let refs = [None, None];
        let bands = [];
        let mut ctx = PairDrawCtx {
            cdf: &cdf,
            row_counts: &counts,
            per_row_cdfs: &cdfs,
            ref_buckets: &refs,
            strat_bands: &bands,
            plan: None,
            draw_index: 0,
        };
        let mut control_rng = SplitMix64::new(88);
        let mut treatment_rng = SplitMix64::new(88);
        let mut independent = SplitMix64::new(77);
        let mut skipped = 0;
        let mut applications = 0;
        for epoch in 0..2 {
            let mut primary_steps = 0;
            for i in 0..300 {
                ctx.draw_index = epoch * 300 + i;
                let control = draw_pair(&ctx, &mut control_rng);
                let treatment = draw_pair(&ctx, &mut treatment_rng);
                assert_eq!(control, treatment);
                if !matches!(treatment, Draw::Pair { .. }) {
                    skipped += 1;
                    continue;
                }
                primary_steps += 1;
                if pool.scheduled(primary_steps) {
                    applications += 1;
                    let mut a = AdamState::new(12, 4, 4, 1);
                    pool.accumulate(&mut independent, &w1, &b1, &w2, &b2, &mut a, 3, 4, &h, 1.);
                    assert_eq!(a.t, 0);
                }
            }
            assert!(!pool.scheduled(0));
        }
        assert!(skipped > 0 && applications > 0);
        assert_eq!(control_rng.state, treatment_rng.state);
        *super::super::INPUT_KEEP_MASK.lock().unwrap() = previous;
    }
}
