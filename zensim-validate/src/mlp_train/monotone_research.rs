//! Private paper-derived heads. All exports use the existing ZNPR dense owner.
//! The manifest is opt-in; absence consumes no random draws and changes no math.
use super::{AdamState, MlpHyperparams, SplitMix64};
use serde::{Deserialize, Serialize};
use zenpredict::{Activation, WeightDtype};
use zenpredict_bake::{BakeLayer, BakeMetadataEntry, BakeRequest, bake};

const KNOTS: [f64; 6] = [0., 0.125, 0.5, 1., 2., 4.];
// A small strict margin protects the real-arithmetic bound when packing f32.
const NORM_LIMIT: f64 = 0.999;

pub(super) fn check_admission(hp: &MlpHyperparams) {
    if std::env::var_os("ZENSIM_MONOTONE_RESEARCH_MANIFEST").is_some()
        || std::env::var_os("ZENSIM_MONOTONE_RESEARCH_SHA256").is_some()
    {
        assert!(
            super::shared_depth::enabled(hp),
            "monotone research manifest requires constrained shared-depth CPU owner"
        );
        assert!(hp.ema_decay == 0., "research heads do not implement EMA");
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Family {
    Caps,
    Switch,
    Spline,
    Lipschitz,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Config {
    schema: String,
    family: Family,
    widths: Vec<usize>,
    init_seed: u64,
    sample_seed: u64,
    epochs: usize,
    pairs_per_epoch: usize,
    first_width: usize,
    max_pairs: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    training: Option<super::training_research::Config>,
}

struct Block {
    kind: Family,
    ni: usize,
    no: usize,
    w: Vec<f64>,
    b: Vec<f64>,
    extra: Vec<f64>,
    adam: AdamState,
}

impl Block {
    fn new(kind: Family, ni: usize, no: usize, rng: &mut SplitMix64) -> Self {
        let scale = (1. / ni as f64).sqrt();
        let w = (0..ni * no)
            .map(|_| {
                let value = rng.next_normal() * scale;
                if matches!(kind, Family::Caps | Family::Spline) {
                    value.abs()
                } else {
                    value
                }
            })
            .collect();
        let b = (0..no)
            .map(|_| match kind {
                Family::Caps => -rng.next_f64_unit(),
                Family::Switch | Family::Lipschitz => 2. * rng.next_f64_unit() - 1.,
                Family::Spline => 0.,
            })
            .collect();
        let extra = match kind {
            Family::Caps => (0..no).map(|_| 0.25 + 2. * rng.next_f64_unit()).collect(),
            Family::Spline => (0..no * KNOTS.len())
                .map(|_| 0.25 + rng.next_f64_unit())
                .collect(),
            Family::Lipschitz => (0..no).map(|_| rng.next_normal() * scale).collect(),
            Family::Switch => Vec::new(),
        };
        let mut result = Self {
            kind,
            ni,
            no,
            w,
            b,
            adam: AdamState::new(ni * no, no, extra.len(), 0),
            extra,
        };
        result.project();
        result
    }

    fn output_width(&self) -> usize {
        if self.kind == Family::Lipschitz {
            1
        } else {
            self.no
        }
    }

    fn pre(&self, x: &[f64], j: usize) -> f64 {
        x.iter()
            .enumerate()
            .fold(self.b[j], |z, (i, x)| z + self.w[i * self.no + j] * x)
    }

    fn switch_pre(&self, x: &[f64], j: usize) -> (f64, f64) {
        x.iter()
            .enumerate()
            .fold((self.b[j], self.b[j]), |(p, n), (i, x)| {
                let w = self.w[i * self.no + j];
                (p + w.max(0.) * x, n + w.min(0.) * x)
            })
    }

    fn spline(&self, z: f64, j: usize) -> (f64, f64) {
        let mut value = 0.;
        let mut derivative = 0.;
        for (k, &start) in KNOTS.iter().enumerate() {
            let end = KNOTS.get(k + 1).copied().unwrap_or(f64::INFINITY);
            let slope = self.extra[j * KNOTS.len() + k];
            value += slope * (z - start).max(0.).min(end - start);
            if z > start && z < end {
                derivative = slope;
            }
        }
        (value, derivative)
    }

    fn forward(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.ni);
        if self.kind == Family::Lipschitz {
            let value = (0..self.no).fold(x.iter().sum::<f64>(), |v, j| {
                v + self.extra[j] * (self.pre(x, j).max(0.) - self.b[j].max(0.))
            });
            return vec![value];
        }
        (0..self.no)
            .map(|j| match self.kind {
                Family::Caps => {
                    let z = self.pre(x, j);
                    if j % 2 == 0 {
                        z.max(0.)
                    } else {
                        z.max(0.) - (z - self.extra[j]).max(0.)
                    }
                }
                Family::Switch => {
                    let (p, n) = self.switch_pre(x, j);
                    p.max(0.) - n.max(0.)
                }
                Family::Spline => self.spline(self.pre(x, j), j).0,
                Family::Lipschitz => unreachable!(),
            })
            .collect()
    }

    fn backward(&mut self, x: &[f64], gradient: &[f64]) -> Vec<f64> {
        let mut dx = vec![0.; self.ni];
        if self.kind == Family::Lipschitz {
            dx.fill(gradient[0]);
            for j in 0..self.no {
                let z = self.pre(x, j);
                let dz = gradient[0] * self.extra[j] * f64::from(z > 0.);
                self.adam.gb1[j] +=
                    gradient[0] * self.extra[j] * (f64::from(z > 0.) - f64::from(self.b[j] > 0.));
                self.adam.gw2[j] += gradient[0] * (z.max(0.) - self.b[j].max(0.));
                for i in 0..self.ni {
                    self.adam.gw1[i * self.no + j] += dz * x[i];
                    dx[i] += dz * self.w[i * self.no + j];
                }
            }
            return dx;
        }
        for j in 0..self.no {
            let g = gradient[j];
            if self.kind == Family::Switch {
                let (p, n) = self.switch_pre(x, j);
                let sp = f64::from(p > 0.);
                let sn = f64::from(n > 0.);
                self.adam.gb1[j] += g * (sp - sn);
                for i in 0..self.ni {
                    let k = i * self.no + j;
                    let w = self.w[k];
                    let dw = if w > 0. {
                        sp
                    } else if w < 0. {
                        -sn
                    } else {
                        0.
                    };
                    self.adam.gw1[k] += g * x[i] * dw;
                    dx[i] += g * (sp * w.max(0.) - sn * w.min(0.));
                }
                continue;
            }
            let z = self.pre(x, j);
            let dz = if self.kind == Family::Caps {
                let top = f64::from(j % 2 == 1 && z > self.extra[j]);
                self.adam.gw2[j] += g * top;
                self.adam.gb1[j] += g * (f64::from(z > 0.) - top);
                g * (f64::from(z > 0.) - top)
            } else {
                for (k, &start) in KNOTS.iter().enumerate() {
                    let end = KNOTS.get(k + 1).copied().unwrap_or(f64::INFINITY);
                    self.adam.gw2[j * KNOTS.len() + k] += g * (z - start).max(0.).min(end - start);
                }
                g * self.spline(z, j).1
            };
            for i in 0..self.ni {
                self.adam.gw1[i * self.no + j] += dz * x[i];
                dx[i] += dz * self.w[i * self.no + j];
            }
        }
        dx
    }

    fn project(&mut self) {
        match self.kind {
            Family::Caps | Family::Spline => {
                self.w.iter_mut().for_each(|w| *w = w.max(0.));
                self.extra.iter_mut().for_each(|w| *w = w.max(1e-6));
                if self.kind == Family::Spline {
                    self.b.fill(0.);
                } else {
                    self.b.iter_mut().for_each(|b| *b = b.min(0.));
                }
            }
            Family::Lipschitz => {
                // Row-major x*W: induced l1->l1 norm is maximum absolute ROW sum.
                for row in self.w.chunks_exact_mut(self.no) {
                    let scale = (row.iter().map(|v| v.abs()).sum::<f64>() / NORM_LIMIT).max(1.);
                    row.iter_mut().for_each(|v| *v /= scale);
                }
                self.extra
                    .iter_mut()
                    .for_each(|v| *v = v.clamp(-NORM_LIMIT, NORM_LIMIT));
            }
            Family::Switch => {}
        }
        assert!(
            self.w
                .iter()
                .chain(&self.b)
                .chain(&self.extra)
                .all(|v| v.is_finite())
        );
    }

    fn step(&mut self, lr: f64, l2: f64) {
        for (g, w) in self.adam.gw1.iter_mut().zip(&self.w) {
            *g += l2 * w;
        }
        if matches!(self.kind, Family::Spline | Family::Lipschitz) {
            for (g, w) in self.adam.gw2.iter_mut().zip(&self.extra) {
                *g += l2 * w;
            }
        }
        self.adam
            .step(&mut self.w, &mut self.b, &mut self.extra, &mut [], lr);
        self.project();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(family: Family, deep: bool) -> Config {
        Config {
            schema: "monotone-head-v1".into(),
            family,
            widths: if deep { vec![4, 3] } else { vec![4] },
            first_width: 3,
            max_pairs: 1,
            init_seed: 1101,
            sample_seed: 101,
            epochs: 12,
            pairs_per_epoch: 5000,
            training: None,
        }
    }

    #[test]
    fn analytic_gradients_include_anchors_and_signed_switch_paths() {
        for family in [
            Family::Caps,
            Family::Switch,
            Family::Spline,
            Family::Lipschitz,
        ] {
            let mut net = Network::new(
                config(family, family == Family::Switch),
                100.,
                &mut SplitMix64::new(75),
            );
            let x = [0.61, 0.83, 0.37];
            let cache = net.forward(&x);
            let dx = net.backward(&x, &cache, 1.);
            let eps = 1e-5;
            for i in 0..x.len() {
                let mut a = x;
                a[i] += eps;
                let mut b = x;
                b[i] -= eps;
                let numeric = (net.forward(&a).0 - net.forward(&b).0) / (2. * eps);
                assert!(
                    (numeric - dx[i]).abs() < 1e-8,
                    "{family:?} input {i}: {numeric} vs {}",
                    dx[i]
                );
            }
            for block in 0..net.blocks.len() {
                for parameter in 0..3 {
                    if family == Family::Spline && parameter == 1 {
                        continue;
                    }
                    let gradient = match parameter {
                        0 => net.blocks[block].adam.gw1.clone(),
                        1 => net.blocks[block].adam.gb1.clone(),
                        _ => net.blocks[block].adam.gw2.clone(),
                    };
                    for (i, analytic) in gradient.iter().enumerate() {
                        let values = match parameter {
                            0 => &mut net.blocks[block].w,
                            1 => &mut net.blocks[block].b,
                            _ => &mut net.blocks[block].extra,
                        };
                        let old = values[i];
                        values[i] = old + eps;
                        let plus = net.forward(&x).0;
                        let values = match parameter {
                            0 => &mut net.blocks[block].w,
                            1 => &mut net.blocks[block].b,
                            _ => &mut net.blocks[block].extra,
                        };
                        values[i] = old - eps;
                        let minus = net.forward(&x).0;
                        let values = match parameter {
                            0 => &mut net.blocks[block].w,
                            1 => &mut net.blocks[block].b,
                            _ => &mut net.blocks[block].extra,
                        };
                        values[i] = old;
                        let numeric = (plus - minus) / (2. * eps);
                        assert!(
                            (numeric - analytic).abs() < 1e-8,
                            "{family:?} block={block} parameter={parameter}/{i}: numeric={numeric}, analytic={analytic}"
                        );
                    }
                }
            }
            for i in 0..net.output.len() {
                let old = net.output[i];
                net.output[i] = old + eps;
                let plus = net.forward(&x).0;
                net.output[i] = old - eps;
                let minus = net.forward(&x).0;
                net.output[i] = old;
                assert!(((plus - minus) / (2. * eps) - net.adam.gw1[i]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn nonconvex_controls_are_monotone_and_zero_anchored() {
        for kind in [
            Family::Caps,
            Family::Switch,
            Family::Spline,
            Family::Lipschitz,
        ] {
            let mut block = Block::new(kind, 1, 2, &mut SplitMix64::new(77));
            block.w.fill(1.);
            block.b.fill(0.);
            block.extra.fill(1.);
            let j = if kind == Family::Lipschitz { 0 } else { 1 };
            match kind {
                Family::Caps => {}
                Family::Switch => {
                    block.w.fill(-1.);
                    block.b.fill(1.);
                }
                Family::Spline => {
                    block.extra.fill(0.2);
                    for unit in 0..block.no {
                        block.extra[unit * KNOTS.len()] = 2.;
                    }
                }
                Family::Lipschitz => {
                    block.w.fill(0.4);
                    block.b.fill(-0.3);
                    block.extra.fill(-0.8);
                }
            }
            assert_eq!(block.forward(&[0.])[j], 0.);
            let a = block.forward(&[0.])[j];
            let b = block.forward(&[2.])[j];
            let mid = block.forward(&[1.])[j];
            assert!(
                mid > (a + b) * 0.5 + 1e-3,
                "{kind:?} did not break convexity"
            );
            let mut previous = 0.;
            for i in 0..1000 {
                let y = block.forward(&[i as f64 / 100.])[j];
                assert!(y >= previous - 1e-12);
                previous = y;
            }
        }
    }

    #[test]
    fn baked_native_identity_order_and_train_agreement_after_updates() {
        for kind in [
            Family::Caps,
            Family::Switch,
            Family::Spline,
            Family::Lipschitz,
        ] {
            let mut net = Network::new(
                config(kind, kind == Family::Switch),
                100.,
                &mut SplitMix64::new(91),
            );
            // First pair: ReLU(a*x)+ReLU((b-a)*x-t)=max(a*x,b*x-t).
            let w = [0.4, -0.2, 0.3, 0.2, 0.4, 0.1];
            let b = [0., -0.1, -0.2];
            let first = |x: [f64; 2]| {
                (0..3)
                    .map(|j| (b[j] + w[j] * x[0] + w[3 + j] * x[1]).max(0.))
                    .collect::<Vec<_>>()
            };
            for round in 0..4 {
                let x = first([0.7, 1.3]);
                let cache = net.forward(&x);
                net.backward(&x, &cache, if round % 2 == 0 { 0.4 } else { -0.7 });
                net.step(0.003, 1e-5);
                let bytes = net.bake(&[0.; 2], &[1.; 2], &w, &b, 2, 3);
                let model = zenpredict::Model::from_bytes(&bytes).unwrap();
                let mut predictor = zenpredict::Predictor::new(&model);
                for zero in [[0., 0.], [-0., 0.], [0., -0.]] {
                    assert_eq!(
                        predictor.predict(&zero).unwrap()[0].to_bits(),
                        100f32.to_bits()
                    );
                }
                for a in [0., 1e-8, 1e-6, 0.1, 0.5, 1., 4., 16., 1024.] {
                    for c in [0., 1e-8, 1e-6, 0.1, 0.5, 1., 4., 16., 1024.] {
                        let x = [a as f32, c as f32];
                        let y = predictor.predict(&x).unwrap()[0];
                        let expected = net.forward(&first([f64::from(x[0]), f64::from(x[1])])).0;
                        assert!(y.is_finite() && y <= 100., "{kind:?} score {y}");
                        // Score = pin - distance: near a score of zero the two
                        // large terms cancel. Bound f32 rounding by those terms,
                        // not by the small residual score alone.
                        let tolerance = 16.
                            * f64::from(f32::EPSILON)
                            * (net.pin.abs() + (net.pin - expected).abs()).max(1.);
                        assert!(
                            (f64::from(y) - expected).abs() <= tolerance,
                            "{kind:?} {x:?}: {y} vs {expected}"
                        );
                        for k in 0..2 {
                            let mut larger = x;
                            larger[k] += 0.125;
                            let z = predictor.predict(&larger).unwrap()[0];
                            assert!(z <= y + 1e-5, "{kind:?} monotone {x:?}: {y} to {z}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn dropout_uses_one_mask_for_pair_anchor_and_analytic_gradient() {
        for family in [
            Family::Caps,
            Family::Switch,
            Family::Spline,
            Family::Lipschitz,
        ] {
            let mut net = Network::new(config(family, false), 100., &mut SplitMix64::new(87));
            let width = net.output.len();
            for bits in 0..(1 << width) {
                net.mask = Some(
                    (0..width)
                        .map(|i| if bits & (1 << i) == 0 { 0. } else { 2. })
                        .collect(),
                );
                assert_eq!(net.training_forward(&[0., 0., 0.]).0, 100.);
                let x = [0.7, 0.4, 1.1];
                let cache = net.training_forward(&x);
                let gradient = net.backward(&x, &cache, 1.);
                for i in 0..x.len() {
                    let mut plus = x;
                    plus[i] += 1e-6;
                    let mut minus = x;
                    minus[i] -= 1e-6;
                    let numerical =
                        (net.training_forward(&plus).0 - net.training_forward(&minus).0) / 2e-6;
                    assert!(
                        (numerical - gradient[i]).abs() < 2e-7,
                        "{family:?} mask={bits}"
                    );
                    assert!(net.training_forward(&plus).0 <= cache.0 + 1e-12);
                }
                let eval = net.forward(&x).0;
                let saved = net.mask.take();
                assert_eq!(eval, net.forward(&x).0);
                net.mask = saved;
                net.adam.gw1.fill(0.);
            }
        }
    }

    #[test]
    fn native_fits_reduce_error_on_nonconvex_monotone_targets() {
        for family in [
            Family::Caps,
            Family::Switch,
            Family::Spline,
            Family::Lipschitz,
        ] {
            let mut cfg = config(family, false);
            cfg.widths = vec![16];
            cfg.first_width = 2;
            cfg.max_pairs = 0;
            let mut net = Network::new(cfg, 0., &mut SplitMix64::new(987));
            let target = |x: f64, y: f64| -(x.min(0.7) + 0.2 * x + 0.3 * y + (y - 1.).max(0.));
            let error = |net: &Network| {
                (0..121)
                    .map(|k| {
                        let x = (k / 11) as f64 / 5.;
                        let y = (k % 11) as f64 / 5.;
                        (net.forward(&[x, y]).0 - target(x, y)).powi(2)
                    })
                    .sum::<f64>()
                    / 121.
            };
            let before = error(&net);
            let mut rng = SplitMix64::new(101);
            for _ in 0..6000 {
                let x = 2. * rng.next_f64_unit();
                let y = 2. * rng.next_f64_unit();
                let cache = net.forward(&[x, y]);
                net.backward(&[x, y], &cache, cache.0 - target(x, y));
                net.step(0.003, 0.);
            }
            let after = error(&net);
            println!(
                "MONOTONE_TOY {{\"family\":\"{family:?}\",\"updates\":6000,\"initial_mse\":{before},\"final_mse\":{after}}}"
            );
            assert!(after < before * 0.1, "{family:?}: {before} to {after}");
        }
    }
}

#[derive(Default)]
struct Diagnostics {
    backwards: u64,
    samples: u64,
    first_zeros: u64,
    first_values: u64,
    last_active: Vec<u64>,
    first_gradient_squared: f64,
    max_hidden: f64,
}

pub(super) struct Network {
    config: Config,
    blocks: Vec<Block>,
    output: Vec<f64>,
    adam: AdamState,
    pin: f64,
    dropout_rng: SplitMix64,
    mask: Option<Vec<f64>>,
    diagnostics: Diagnostics,
}

impl Network {
    pub(super) fn from_env(
        hp: &MlpHyperparams,
        smooth: bool,
        rng: &mut SplitMix64,
    ) -> Option<Self> {
        let path = std::env::var_os("ZENSIM_MONOTONE_RESEARCH_MANIFEST");
        let hash = std::env::var_os("ZENSIM_MONOTONE_RESEARCH_SHA256");
        if path.is_none() && hash.is_none() {
            return None;
        }
        assert!(
            !smooth,
            "research manifest cannot silently replace SmoothHinge"
        );
        let path = path.expect("research manifest path required");
        let expected = hash.expect("independent research manifest hash required");
        let got = crate::train_manifest::sha256_file(std::path::Path::new(&path))
            .expect("research manifest hash");
        assert_eq!(got, expected.to_str().expect("manifest SHA256 UTF8"));
        let bytes = std::fs::read(path).expect("read research manifest");
        let config: Config =
            serde_json::from_slice(&bytes).expect("strict research manifest schema");
        assert_eq!(config.schema, "monotone-head-v1");
        assert_eq!(config.first_width, hp.n_hidden);
        assert_eq!(config.max_pairs, hp.max_pairs);
        assert_eq!(config.init_seed, hp.init_seed.unwrap_or(hp.seed));
        assert_eq!(config.sample_seed, hp.sample_seed.unwrap_or(hp.seed));
        assert_eq!(config.epochs, hp.n_epochs);
        assert_eq!(config.pairs_per_epoch, hp.pairs_per_epoch);
        if let Some(study) = &config.training {
            study.validate(hp);
        }
        Some(Self::new(config, hp.nonneg_pin, rng))
    }

    fn new(config: Config, pin: f64, rng: &mut SplitMix64) -> Self {
        assert!(config.max_pairs * 2 <= config.first_width);
        assert!((1..=2).contains(&config.widths.len()));
        assert!(config.widths.iter().all(|w| (1..=256).contains(w)));
        assert!(config.family == Family::Switch || config.widths.len() == 1);
        let mut ni = config.first_width - config.max_pairs;
        let mut blocks = Vec::new();
        for &width in &config.widths {
            let mut block = Block::new(config.family, ni, width, rng);
            if let Some(study) = &config.training {
                for w in &mut block.w {
                    *w *= study.init_scale;
                }
                for b in &mut block.b {
                    *b *= study.bias_scale;
                }
                block.project();
            }
            ni = block.output_width();
            blocks.push(block);
        }
        let output = (0..ni)
            .map(|_| rng.next_normal().abs() * (0.1 / ni as f64).sqrt())
            .collect();
        let dropout_rng = SplitMix64::new(config.training.as_ref().map_or(0, |c| c.dropout_seed));
        Self {
            config,
            dropout_rng,
            mask: None,
            diagnostics: Diagnostics::default(),
            blocks,
            output,
            adam: AdamState::new(ni, 0, 0, 0),
            pin,
        }
    }

    pub(super) fn training(&self) -> Option<&super::training_research::Config> {
        self.config.training.as_ref()
    }

    pub(super) fn init_first(&self, weights: &mut [f64], bias: &mut [f64]) {
        if let Some(study) = self.training() {
            for w in weights {
                *w *= study.init_scale;
            }
            for b in bias {
                *b *= study.bias_scale;
            }
        }
    }

    pub(super) fn begin_pair(&mut self) {
        self.mask = super::training_research::dropout_mask(
            self.config.training.as_ref().map_or(0., |c| c.dropout),
            self.output.len(),
            &mut self.dropout_rng,
        );
    }

    pub(super) fn training_forward(&self, first: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
        let mut cache = self.forward(first);
        if let Some(mask) = &self.mask {
            let last = &cache.2[cache.2.len() - self.output.len()..];
            cache.0 = last
                .iter()
                .zip(&self.output)
                .zip(mask)
                .fold(self.pin, |y, ((h, a), m)| y - h * a * m);
        }
        cache
    }

    fn basis(&self, first: &[f64]) -> Vec<f64> {
        let pairs = self.config.max_pairs;
        let mut basis = (0..pairs)
            .map(|i| first[2 * i] + first[2 * i + 1])
            .collect::<Vec<_>>();
        basis.extend_from_slice(&first[2 * pairs..]);
        basis
    }

    pub(super) fn forward(&self, first: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
        let mut current = self.basis(first);
        let mut cache = current.clone();
        for block in &self.blocks {
            current = block.forward(&current);
            cache.extend_from_slice(&current);
        }
        let score = current
            .iter()
            .zip(&self.output)
            .fold(self.pin, |y, (h, a)| y - h * a);
        (score, Vec::new(), cache)
    }

    pub(super) fn backward(
        &mut self,
        first: &[f64],
        cache: &(f64, Vec<f64>, Vec<f64>),
        dy: f64,
    ) -> Vec<f64> {
        let last = &cache.2[cache.2.len() - self.output.len()..];
        let mut gradient = self.output.iter().map(|a| -dy * a).collect::<Vec<_>>();
        for (j, (g, h)) in self.adam.gw1.iter_mut().zip(last).enumerate() {
            let m = self.mask.as_ref().map_or(1., |mask| mask[j]);
            *g -= dy * h * m;
            gradient[j] *= m;
        }
        let mut end = cache.2.len() - self.output.len();
        for block in self.blocks.iter_mut().rev() {
            let begin = end - block.ni;
            gradient = block.backward(&cache.2[begin..end], &gradient);
            end = begin;
        }
        assert_eq!(end, 0);
        let mut result = vec![0.; first.len()];
        let pairs = self.config.max_pairs;
        for i in 0..pairs {
            result[2 * i] = gradient[i];
            result[2 * i + 1] = gradient[i];
        }
        result[2 * pairs..].copy_from_slice(&gradient[pairs..]);
        if self.config.training.is_some() {
            let stats = &mut self.diagnostics;
            stats.backwards += 1;
            if stats.backwards.is_multiple_of(256) {
                stats.samples += 1;
                stats.first_values += first.len() as u64;
                stats.first_zeros += first.iter().filter(|&&x| x == 0.).count() as u64;
                stats.last_active.resize(last.len(), 0);
                for (count, &h) in stats.last_active.iter_mut().zip(last) {
                    *count += u64::from(h != 0.);
                }
                stats.first_gradient_squared += result.iter().map(|v| v * v).sum::<f64>();
                stats.max_hidden = last.iter().fold(stats.max_hidden, |m, h| m.max(h.abs()));
            }
        }
        result
    }

    pub(super) fn step(&mut self, lr: f64, l2: f64) {
        for block in &mut self.blocks {
            block.step(lr, l2);
        }
        for (g, w) in self.adam.gw1.iter_mut().zip(&self.output) {
            *g += l2 * w;
        }
        self.adam
            .step(&mut self.output, &mut [], &mut [], &mut [], lr);
        if let Some(study) = &self.config.training {
            super::training_research::decay(&mut self.output, lr, study.adamw);
            for block in &mut self.blocks {
                super::training_research::decay(&mut block.w, lr, study.adamw);
                if matches!(block.kind, Family::Spline | Family::Lipschitz) {
                    super::training_research::decay(&mut block.extra, lr, study.adamw);
                }
                block.project();
            }
        }
        self.output.iter_mut().for_each(|w| *w = w.max(0.));
        assert!(self.output.iter().all(|w| w.is_finite()));
    }

    pub(super) fn report_diagnostics(&mut self, epoch: usize) {
        if self.config.training.is_none() {
            return;
        }
        let stats = std::mem::take(&mut self.diagnostics);
        println!(
            "TRAINING_DIAGNOSTICS {}",
            serde_json::json!({
                "epoch": epoch, "backwards": stats.backwards, "sample_stride": 256,
                "sampled_responses": stats.samples, "first_zero_activations": stats.first_zeros,
                "first_activation_values": stats.first_values,
                "last_never_active_in_sample": stats.last_active.iter().filter(|&&n|n==0).count(),
                "last_units": self.output.len(), "last_unit_active_counts": stats.last_active,
                "max_last_activation_in_sample": stats.max_hidden,
                "mean_squared_first_activation_gradient": if stats.samples>0 {Some(stats.first_gradient_squared/stats.samples as f64)} else {None},
                "output_zero_weights": self.output.iter().filter(|&&w|w==0.).count(),
                "note": "training-response sample, not proof of globally dead units or a timing profile"
            })
        );
    }

    pub(super) fn bake(
        &self,
        mean: &[f64],
        scale: &[f64],
        w1: &[f64],
        b1: &[f64],
        nf: usize,
        nh: usize,
    ) -> Vec<u8> {
        assert!(mean.iter().all(|x| *x == 0.));
        let mut layers = vec![Dense::from_f64(nf, nh, Activation::Relu, w1, b1)];
        let pairs = self.config.max_pairs;
        let mut basis = Dense::zero(nh, nh - pairs, Activation::Identity);
        for i in 0..pairs {
            basis.set(2 * i, i, 1.);
            basis.set(2 * i + 1, i, 1.);
        }
        for i in 2 * pairs..nh {
            basis.set(i, i - pairs, 1.);
        }
        layers.push(basis);
        for block in &self.blocks {
            block.expand(&mut layers);
        }
        layers.push(Dense::from_f64(
            self.output.len(),
            1,
            Activation::Identity,
            &self.output.iter().map(|v| -v).collect::<Vec<_>>(),
            &[self.pin],
        ));
        let borrowed = layers.iter().map(Dense::borrow).collect::<Vec<_>>();
        let config = serde_json::to_vec(&self.config).expect("research config JSON");
        let metadata = [BakeMetadataEntry {
            key: "zentrain.monotone_research",
            kind: zenpredict::MetadataType::Utf8,
            value: &config,
        }];
        bake(&BakeRequest {
            schema_hash: 0,
            flags: 0,
            scaler_mean: &mean.iter().map(|v| *v as f32).collect::<Vec<_>>(),
            scaler_scale: &scale.iter().map(|v| *v as f32).collect::<Vec<_>>(),
            layers: &borrowed,
            feature_bounds: &[],
            metadata: &metadata,
            output_specs: &[],
            discrete_sets: &[],
            sparse_overrides: &[],
            feature_order: None,
            output_order: None,
            compressed: false,
            hu_permutations: None,
        })
        .expect("existing dense/ReLU research bake")
    }
}

struct Dense {
    ni: usize,
    no: usize,
    act: Activation,
    w: Vec<f32>,
    b: Vec<f32>,
}
impl Dense {
    fn zero(ni: usize, no: usize, act: Activation) -> Self {
        Self {
            ni,
            no,
            act,
            w: vec![0.; ni * no],
            b: vec![0.; no],
        }
    }
    fn from_f64(ni: usize, no: usize, act: Activation, w: &[f64], b: &[f64]) -> Self {
        Self {
            ni,
            no,
            act,
            w: w.iter().map(|v| *v as f32).collect(),
            b: b.iter().map(|v| *v as f32).collect(),
        }
    }
    fn set(&mut self, i: usize, j: usize, v: f64) {
        self.w[i * self.no + j] = v as f32;
    }
    fn borrow(&self) -> BakeLayer<'_> {
        BakeLayer {
            in_dim: self.ni,
            out_dim: self.no,
            activation: self.act,
            dtype: WeightDtype::F32,
            weights: &self.w,
            biases: &self.b,
        }
    }
}

impl Block {
    fn expand(&self, layers: &mut Vec<Dense>) {
        if self.kind == Family::Spline {
            let n = self.no * KNOTS.len();
            let mut hinges = Dense::zero(self.ni, n, Activation::Relu);
            let mut combine = Dense::zero(n, self.no, Activation::Identity);
            for j in 0..self.no {
                let mut previous = 0.;
                for (k, &knot) in KNOTS.iter().enumerate() {
                    let column = j * KNOTS.len() + k;
                    hinges.b[column] = -knot as f32;
                    for i in 0..self.ni {
                        hinges.set(i, column, self.w[i * self.no + j]);
                    }
                    let slope = self.extra[column];
                    combine.set(column, j, slope - previous);
                    previous = slope;
                }
            }
            layers.extend([hinges, combine]);
            return;
        }
        let pass = if self.kind == Family::Lipschitz {
            self.ni
        } else {
            0
        };
        let mut hinges = Dense::zero(self.ni, 2 * self.no + pass, Activation::Relu);
        let mut combine = Dense::zero(2 * self.no + pass, self.no + pass, Activation::Identity);
        for j in 0..self.no {
            let b = self.b[j];
            hinges.b[2 * j] = b as f32;
            hinges.b[2 * j + 1] = if self.kind == Family::Caps {
                (b - self.extra[j]) as f32
            } else {
                b as f32
            };
            combine.set(2 * j, j, 1.);
            if self.kind != Family::Caps || j % 2 == 1 {
                combine.set(2 * j + 1, j, -1.);
            }
            for i in 0..self.ni {
                let w = self.w[i * self.no + j];
                hinges.set(
                    i,
                    2 * j,
                    if self.kind == Family::Switch {
                        w.max(0.)
                    } else {
                        w
                    },
                );
                hinges.set(
                    i,
                    2 * j + 1,
                    match self.kind {
                        Family::Switch => w.min(0.),
                        Family::Caps => w,
                        _ => 0.,
                    },
                );
            }
        }
        for i in 0..pass {
            hinges.set(i, 2 * self.no + i, 1.);
            combine.set(2 * self.no + i, self.no + i, 1.);
        }
        layers.extend([hinges, combine]);
        if self.kind == Family::Lipschitz {
            let mut residual = Dense::zero(self.no + pass, 1, Activation::Identity);
            for j in 0..self.no {
                residual.set(j, 0, self.extra[j]);
            }
            for i in 0..pass {
                residual.set(self.no + i, 0, 1.);
            }
            layers.push(residual);
        }
    }
}
