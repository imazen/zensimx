//! Exact analytic research heads; retain the shared first-layer owner.
use super::{Activation, Block, Dense, Family};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Config {
    pub temperature: f64,
    pub learn_temperature: bool,
    pub bias: f64,
}
impl Config {
    pub(super) fn validate(&self, family: Family) {
        assert!(self.temperature.is_finite() && (-6.0..=6.0).contains(&self.temperature.ln()));
        assert!(self.bias.is_finite());
        assert!(family != Family::SiluMonotone || self.bias >= 0.);
    }
}

// Returns f(u), df/du. Exponents never have a positive argument.
fn unit(u: f64, softplus: bool) -> (f64, f64) {
    let e = (-u.abs()).exp();
    let sigmoid = if u >= 0. { 1. / (1. + e) } else { e / (1. + e) };
    if softplus {
        (u.max(0.) + e.ln_1p(), sigmoid)
    } else {
        (u * sigmoid, sigmoid + u * sigmoid * (1. - sigmoid))
    }
}

impl Block {
    pub(super) fn is_analytic(&self) -> bool {
        matches!(
            self.kind,
            Family::Softplus | Family::SiluMonotone | Family::SiluDiagnostic
        )
    }

    // z includes b. Returns value, d/dz (input/weight path), total d/db,
    // and d/dlog(tau). The anchor participates in BOTH parameter derivatives.
    pub(super) fn analytic_response(&self, z: f64, b: f64) -> (f64, f64, f64, f64) {
        let tau = self.extra[0].exp();
        let soft = self.kind == Family::Softplus;
        let u = z / tau;
        let (f, df) = unit(u, soft);
        if self.kind == Family::SiluDiagnostic {
            return (tau * f, df, df, tau * (f - u * df));
        }
        let v = b / tau;
        let (anchor, da) = unit(v, soft);
        (
            tau * (f - anchor),
            df,
            df - da,
            tau * ((f - u * df) - (anchor - v * da)),
        )
    }

    pub(super) fn expand_analytic(&self, layers: &mut Vec<Dense>) {
        let tau = self.extra[0].exp();
        let anchored = self.kind != Family::SiluDiagnostic;
        let width = self.no * if anchored { 2 } else { 1 };
        let act = if self.kind == Family::Softplus {
            Activation::Softplus
        } else {
            Activation::Silu
        };
        let mut responses = Dense::zero(self.ni, width, act);
        for j in 0..self.no {
            responses.b[j] = (self.b[j] / tau) as f32;
            for i in 0..self.ni {
                responses.set(i, j, self.w[i * self.no + j] / tau);
            }
            if anchored {
                responses.b[j + self.no] = responses.b[j];
            }
        }
        layers.push(responses);
        if anchored {
            // Unit coefficients make equal branch subtraction exact even under
            // FMA. Multiplying by arbitrary tau during subtraction would not.
            let mut difference = Dense::zero(width, self.no, Activation::Identity);
            for j in 0..self.no {
                difference.set(j, j, 1.);
                difference.set(j + self.no, j, -1.);
            }
            layers.push(difference);
        }
        let mut scale = Dense::zero(self.no, self.no, Activation::Identity);
        for j in 0..self.no {
            scale.set(j, j, tau);
        }
        layers.push(scale);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlp_train::SplitMix64;
    use zenpredict::{Model, Predictor};
    use zenpredict_bake::BakeRequest;

    #[test]
    fn all_parameter_gradients_include_both_anchor_terms() {
        for family in [
            Family::Softplus,
            Family::SiluMonotone,
            Family::SiluDiagnostic,
        ] {
            let mut block = Block::new(family, 2, 2, &mut SplitMix64::new(123));
            block.learn_temperature = true;
            block.extra[0] = 0.73_f64.ln();
            block.b.fill(0.37);
            let x = [0.23, 0.87];
            let g = [0.7, -0.3];
            let dx = block.backward(&x, &g);
            let loss =
                |b: &Block, x: &[f64]| b.forward(x).iter().zip(g).map(|(v, g)| v * g).sum::<f64>();
            let eps = 1e-5;
            for i in 0..2 {
                let mut plus = x;
                plus[i] += eps;
                let mut minus = x;
                minus[i] -= eps;
                assert!(
                    ((loss(&block, &plus) - loss(&block, &minus)) / (2. * eps) - dx[i]).abs()
                        < 1e-8
                );
            }
            for parameter in 0..3 {
                let gradients = match parameter {
                    0 => block.adam.gw1.clone(),
                    1 => block.adam.gb1.clone(),
                    _ => block.adam.gw2.clone(),
                };
                for (i, gradient) in gradients.iter().enumerate() {
                    let field = |b: &mut Block, value| match parameter {
                        0 => b.w[i] = value,
                        1 => b.b[i] = value,
                        _ => b.extra[i] = value,
                    };
                    let old = match parameter {
                        0 => block.w[i],
                        1 => block.b[i],
                        _ => block.extra[i],
                    };
                    field(&mut block, old + eps);
                    let plus = loss(&block, &x);
                    field(&mut block, old - eps);
                    let minus = loss(&block, &x);
                    field(&mut block, old);
                    assert!(
                        ((plus - minus) / (2. * eps) - gradient).abs() < 1e-8,
                        "{family:?} parameter={parameter}/{i}"
                    );
                }
            }
        }
    }

    #[test]
    fn exported_anchors_are_exact_at_arbitrary_temperature() {
        for family in [Family::Softplus, Family::SiluMonotone] {
            for temperature in [0.13_f64, 0.73, 1.0, 3.17] {
                let mut block = Block::new(family, 2, 2, &mut SplitMix64::new(42));
                block.extra[0] = temperature.ln();
                block.b.fill(0.37);
                let mut layers = Vec::new();
                block.expand(&mut layers);
                let borrowed = layers.iter().map(Dense::borrow).collect::<Vec<_>>();
                let bytes = BakeRequest::builder(0, 0, &[0., 0.], &[1., 1.], &borrowed)
                    .bake()
                    .unwrap();
                let model = Model::from_bytes(&bytes).unwrap();
                let mut p = Predictor::new(&model);
                for &v in p.predict(&[0., 0.]).unwrap() {
                    assert_eq!(v, 0.);
                }
                let mut last = [0.; 2];
                for n in 0..100 {
                    let x = [n as f32 * 0.031, 0.27];
                    let out = p.predict(&x).unwrap();
                    let reference = block.forward(&x.map(f64::from));
                    for j in 0..2 {
                        assert!(out[j] >= last[j]);
                        assert!((f64::from(out[j]) - reference[j]).abs() < 2e-6);
                        last[j] = out[j];
                    }
                }
            }
        }
    }
}
