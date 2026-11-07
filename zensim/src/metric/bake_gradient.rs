//! Exact reverse derivative of the loaded mathematical score graph.
//! Scalar f64 accumulation is deliberate: gradients have one arithmetic order
//! across inference dispatch tiers. The ordinary score forward is untouched.
use super::*;
use zenpredict::{Activation, FeatureTransform, WeightStorage};

type Node = usize;
struct Value {
    value: f64,
    parents: Vec<(Node, f64)>,
}
#[derive(Default)]
struct Tape {
    nodes: Vec<Value>,
}
impl Tape {
    fn node(&mut self, value: f64, parents: Vec<(Node, f64)>) -> Node {
        let id = self.nodes.len();
        self.nodes.push(Value { value, parents });
        id
    }
    fn constant(&mut self, x: f64) -> Node {
        self.node(x, vec![])
    }
    fn value(&self, x: Node) -> f64 {
        self.nodes[x].value
    }
    fn unary(&mut self, x: Node, value: f64, slope: f64) -> Node {
        self.node(value, vec![(x, slope)])
    }
    fn affine(&mut self, x: &[Node], weights: &[f64], bias: f64) -> Node {
        let value = x
            .iter()
            .zip(weights)
            .fold(bias, |v, (&x, &w)| v + self.value(x) * w);
        self.node(
            value,
            x.iter()
                .copied()
                .zip(weights.iter().copied())
                .filter(|(_, w)| *w != 0.)
                .collect(),
        )
    }
    fn product(&mut self, a: Node, b: Node) -> Node {
        self.node(
            self.value(a) * self.value(b),
            vec![(a, self.value(b)), (b, self.value(a))],
        )
    }
    fn clamp(&mut self, x: Node, lo: f64, hi: f64) -> Node {
        let v = self.value(x);
        self.unary(x, v.clamp(lo, hi), if v > lo && v < hi { 1. } else { 0. })
    }
    fn sigmoid(&mut self, x: Node, limit: f64) -> Node {
        let x = self.clamp(x, -limit, limit);
        let v = 1. / (1. + (-self.value(x)).det_exp(active_pow_form()));
        self.unary(x, v, v * (1. - v))
    }
    fn reverse(&self, output: Node, n: usize) -> Result<Vec<f64>, ZensimError> {
        let mut g = vec![0.; self.nodes.len()];
        g[output] = 1.;
        for i in (0..=output).rev() {
            if g[i] == 0. {
                continue;
            }
            for &(p, d) in &self.nodes[i].parents {
                g[p] += g[i] * d;
            }
        }
        g.truncate(n);
        if g.iter().any(|x| !x.is_finite()) {
            return Err(error("exact sensitivity is nonfinite at this input"));
        }
        Ok(g)
    }
}
fn error(reason: &'static str) -> ZensimError {
    ZensimError::ModelForwardFailed { reason }
}

// At a ReLU kink use the inactive branch; at clamp boundaries use zero.
// Singular root transforms at zero refuse when their derivative is active.
fn transform(
    t: &mut Tape,
    x: Node,
    tf: FeatureTransform,
    p: &[f32],
) -> Result<Vec<Node>, ZensimError> {
    use FeatureTransform::*;
    let v = t.value(x);
    let par = |i: usize, default: f64| p.get(i).map_or(default, |&v| f64::from(v));
    let root = |v: f64, e: f64| {
        (
            v.signum() * v.abs().powf(e),
            if v == 0. {
                if e < 1. {
                    f64::INFINITY
                } else if e == 1. {
                    1.
                } else {
                    0.
                }
            } else {
                e * v.abs().powf(e - 1.)
            },
        )
    };
    let (value, slope) = match tf {
        Identity => (v, 1.),
        Log => (v.ln(), 1. / v),
        Log1p => (v.ln_1p(), 1. / (1. + v)),
        SignedLog1p => (v.signum() * v.abs().ln_1p(), 1. / (1. + v.abs())),
        SignedSqrt => root(v, 0.5),
        SignedCbrt => root(v, 1. / 3.),
        SignedPow => root(v, if par(0, 0.5) > 0. { par(0, 0.5) } else { 1. }),
        ClipThenLog1p | ClipThenLog1pThenWinsor => {
            let eps = if tf == ClipThenLog1pThenWinsor && p.len() < 3 {
                0.
            } else {
                par(0, 0.)
            };
            let a = (v - eps).max(0.);
            let value = a.ln_1p();
            let slope = if v > eps { 1. / (1. + a) } else { 0. };
            if tf == ClipThenLog1pThenWinsor && p.len() >= 3 {
                (
                    value.clamp(par(1, f64::NEG_INFINITY), par(2, f64::INFINITY)),
                    if value > par(1, 0.) && value < par(2, 0.) {
                        slope
                    } else {
                        0.
                    },
                )
            } else {
                (value, slope)
            }
        }
        WinsorP99 | WinsorThenLog | WinsorThenLog1p | WinsorThenSignedCbrt => {
            let lo = par(0, f64::NEG_INFINITY);
            let hi = par(1, f64::INFINITY);
            let bounded = if p.len() >= 2 || tf == WinsorP99 {
                v.clamp(lo, hi)
            } else {
                v
            };
            let inside = if tf == WinsorP99 || p.len() >= 2 {
                v > lo && v < hi
            } else {
                // Stacked transforms ignore incomplete clamp parameters.
                true
            };
            let (value, slope) = match tf {
                WinsorThenLog => (bounded.ln(), 1. / bounded),
                WinsorThenLog1p => (bounded.ln_1p(), 1. / (1. + bounded)),
                WinsorThenSignedCbrt => root(bounded, 1. / 3.),
                _ => (bounded, 1.),
            };
            (value, if inside { slope } else { 0. })
        }
        SignedCbrtThenWinsor => {
            let (value, slope) = root(v, 1. / 3.);
            if p.len() < 2 {
                (value, slope)
            } else {
                (
                    value.clamp(par(0, 0.), par(1, 0.)),
                    if value > par(0, 0.) && value < par(1, 0.) {
                        slope
                    } else {
                        0.
                    },
                )
            }
        }
        QuantileBins => {
            if p.is_empty() {
                (v, 1.)
            } else {
                (
                    p.iter().filter(|&&e| v >= f64::from(e)).count() as f64 / p.len() as f64,
                    0.,
                )
            }
        }
        SoftSign => {
            let s = if par(0, 1.) > 0. { par(0, 1.) } else { 1. };
            (v / (s + v.abs()), s / (s + v.abs()).powi(2))
        }
        SoftClip => {
            let k = if par(0, 1.) > 0. { par(0, 1.) } else { 1. };
            let s = if par(1, 1.) > 0. { par(1, 1.) } else { 1. };
            if v.abs() <= k {
                (v, 1.)
            } else {
                (
                    v.signum() * (k + s * ((v.abs() - k) / s).ln_1p()),
                    s / (s + v.abs() - k),
                )
            }
        }
        YeoJohnson => {
            if p.is_empty() {
                (v, 1.)
            } else {
                let l = par(0, 1.);
                let (value, slope) = if v >= 0. {
                    if l.abs() < 1e-7 {
                        ((1. + v).ln(), 1. / (1. + v))
                    } else {
                        (((1. + v).powf(l) - 1.) / l, (1. + v).powf(l - 1.))
                    }
                } else if (l - 2.).abs() < 1e-7 {
                    (-(1. - v).ln(), 1. / (1. - v))
                } else {
                    (
                        -((1. - v).powf(2. - l) - 1.) / (2. - l),
                        (1. - v).powf(1. - l),
                    )
                };
                // Match the transform owner's intermediate f32 overflow guard,
                // including a power that overflows before division by lambda.
                let power = if v >= 0. && l.abs() >= 1e-7 {
                    (1_f32 + v as f32).powf(l as f32)
                } else if v < 0. && (l - 2.).abs() >= 1e-7 {
                    (1_f32 - v as f32).powf((2. - l) as f32)
                } else {
                    1.
                };
                if !power.is_finite() || !(value as f32).is_finite() {
                    (v, 1.)
                } else {
                    (value, slope)
                }
            }
        }
        Drop => return Ok(vec![]),
        Sinusoidal => {
            let mut out = Vec::new();
            for &f in p {
                let a = 2. * std::f64::consts::PI * f64::from(f);
                out.push(t.unary(x, (a * v).sin(), a * (a * v).cos()));
                out.push(t.unary(x, (a * v).cos(), -a * (a * v).sin()));
            }
            return Ok(out);
        }
        _ => return Err(error("unknown feature transform in exact sensitivity")),
    };
    Ok(vec![t.unary(x, value, slope)])
}

fn scaled_inputs(
    t: &mut Tape,
    s: &BakeScorer<'_>,
    roots: &[Node],
    dimensions: (u32, u32),
) -> Result<Vec<Node>, ZensimError> {
    // Identity inputs follow the forward owner's exact/prefix/+4 sizing.
    // Their layout may include size axes that have no feature root.
    let gathered: Vec<_> = if !s.layout.is_identity() || s.metadata.derived_inputs.is_some() {
        (0..s.layout.width())
            .map(|i| match s.layout.slot_at(i) {
                Some(id) => roots
                    .get(usize::from(id))
                    .copied()
                    .ok_or_else(|| error("exact sensitivity input shape mismatch")),
                None => Ok(t.constant(0.)),
            })
            .collect::<Result<_, _>>()?
    } else {
        Vec::new()
    };
    let mut inputs = if let Some(d) = s.metadata.derived_inputs.as_deref() {
        d.entries()
            .iter()
            .map(|e| match *e {
                crate::derived_inputs::DerivedInput::Direct(a) => gathered[a as usize],
                crate::derived_inputs::DerivedInput::Shaped(a, shape) => {
                    let node = gathered[a as usize];
                    let (value, slope) = shape.value_slope(t.value(node));
                    t.unary(node, value, slope)
                }
                crate::derived_inputs::DerivedInput::ShapedProduct(a, b, shape) => {
                    let node = t.product(gathered[a as usize], gathered[b as usize]);
                    let (value, slope) = shape.value_slope(t.value(node));
                    t.unary(node, value, slope)
                }
                crate::derived_inputs::DerivedInput::Separate(a, b, shape, gate) => {
                    let a = gathered[a as usize];
                    let b = gathered[b as usize];
                    let (value, slope) = shape.value_slope(t.value(a));
                    let a = t.unary(a, value, slope);
                    let (value, slope) = gate.value_slope(t.value(b));
                    let b = t.unary(b, value, slope);
                    t.product(a, b)
                }
                crate::derived_inputs::DerivedInput::Product(a, b) => {
                    t.product(gathered[a as usize], gathered[b as usize])
                }
            })
            .collect()
    } else {
        let values = if s.layout.is_identity() {
            roots
        } else {
            &gathered
        };
        let n = s.model.caller_input_width();
        let mut out = values[..values.len().min(n)].to_vec();
        if n == values.len() + 4 {
            let mut axes = Vec::new();
            append_mlp_size_axes(&mut axes, dimensions.0, dimensions.1);
            out.extend(axes.into_iter().map(|v| t.constant(v)));
        }
        if out.len() != n {
            return Err(error("exact sensitivity input shape mismatch"));
        }
        out
    };
    if let Some(tf) = s.model.feature_transforms() {
        let mut transformed = Vec::new();
        for (i, &x) in inputs.iter().enumerate() {
            let p = s
                .model
                .feature_transform_params()
                .map_or(&[][..], |p| p[i].as_slice());
            transformed.extend(transform(t, x, tf[i], p)?);
        }
        inputs = transformed;
    }
    if inputs.len() != s.model.n_inputs() {
        return Err(error("exact sensitivity expanded shape mismatch"));
    }
    for (i, x) in inputs.iter_mut().enumerate() {
        let scale = f64::from(s.model.scaler_scale()[i]);
        let scale = if scale == 0. { 1. } else { scale };
        *x = t.unary(
            *x,
            (t.value(*x) - f64::from(s.model.scaler_mean()[i])) / scale,
            1. / scale,
        );
    }
    Ok(inputs)
}

#[allow(clippy::too_many_arguments)] // Mirrors the two loaded pool-head parameter records.
fn pool(
    t: &mut Tape,
    h: &[Node],
    rank_w: &[f32],
    rank_b: f32,
    reducer: &[f32; 4],
    reducer_b: f32,
    p: f32,
    gate: Node,
) -> Node {
    let n = h.len() as f64;
    let mu = h.iter().map(|&x| t.value(x)).sum::<f64>() / n;
    let sigma = (h.iter().map(|&x| (t.value(x) - mu).powi(2)).sum::<f64>() / n).sqrt();
    let max = h
        .iter()
        .copied()
        .reduce(|a, b| if t.value(b) > t.value(a) { b } else { a })
        .unwrap();
    let p = f64::from(p);
    let moment = h
        .iter()
        .map(|&x| t.value(x).abs().det_powf(p, active_pow_form()))
        .sum::<f64>()
        / n;
    let norm = moment.det_powf(1. / p, active_pow_form());
    let alpha = t.value(gate);
    let rank = t.affine(
        h,
        &rank_w.iter().map(|&w| f64::from(w)).collect::<Vec<_>>(),
        f64::from(rank_b),
    );
    let weights: Vec<_> = h
        .iter()
        .map(|&x| {
            f64::from(reducer[0]) / n
                + if sigma > crate::score_math::POOL_STD_FLOOR {
                    f64::from(reducer[1]) * (t.value(x) - mu) / (n * sigma)
                } else {
                    0.
                }
                + if x == max { f64::from(reducer[2]) } else { 0. }
                + if moment > 0. && t.value(x) != 0. {
                    f64::from(reducer[3])
                        * moment.powf(1. / p - 1.)
                        * t.value(x).abs().powf(p - 1.)
                        * t.value(x).signum()
                        / n
                } else {
                    0.
                }
        })
        .collect();
    let pool_value = mu * f64::from(reducer[0])
        + sigma.max(crate::score_math::POOL_STD_FLOOR) * f64::from(reducer[1])
        + t.value(max) * f64::from(reducer[2])
        + norm * f64::from(reducer[3])
        + f64::from(reducer_b);
    let pooled = t.node(pool_value, h.iter().copied().zip(weights).collect());
    t.node(
        alpha * t.value(rank) + (1. - alpha) * pool_value,
        vec![
            (rank, alpha),
            (pooled, 1. - alpha),
            (gate, t.value(rank) - pool_value),
        ],
    )
}

fn primary(
    t: &mut Tape,
    s: &BakeScorer<'_>,
    roots: &[Node],
    dimensions: (u32, u32),
    codec: Option<&str>,
) -> Result<Node, ZensimError> {
    let mut input = scaled_inputs(t, s, roots, dimensions)?;
    let mut y = if let Some(mm) = s.metadata.minmax_head.as_deref() {
        for x in &mut input {
            *x = t.clamp(*x, -MINMAX_STD_CLAMP, MINMAX_STD_CLAMP);
        }
        let mut groups = Vec::new();
        for g in 0..mm.k {
            let mut pieces = Vec::new();
            for j in 0..mm.j {
                let i = g * mm.j + j;
                pieces.push(
                    t.affine(
                        &input,
                        &mm.w[i * mm.n..(i + 1) * mm.n]
                            .iter()
                            .map(|&x| f64::from(x))
                            .collect::<Vec<_>>(),
                        f64::from(mm.b[i]),
                    ),
                );
            }
            groups.push(
                pieces
                    .iter()
                    .copied()
                    .reduce(|a, b| if t.value(b) > t.value(a) { b } else { a })
                    .unwrap(),
            );
        }
        groups
            .iter()
            .copied()
            .reduce(|a, b| if t.value(b) < t.value(a) { b } else { a })
            .unwrap()
    } else {
        for layer in s.model.layers() {
            let mut out = Vec::new();
            for j in 0..layer.out_dim {
                let weights: Vec<_> = (0..layer.in_dim)
                    .map(|i| match layer.weights {
                        WeightStorage::F32(w) => f64::from(w[i * layer.out_dim + j]),
                        WeightStorage::F16(w) => {
                            f64::from(zenpredict::f16_bits_to_f32(w[i * layer.out_dim + j]))
                        }
                        WeightStorage::I8 { weights, scales } => {
                            f64::from(weights[i * layer.out_dim + j]) * f64::from(scales[j])
                        }
                    })
                    .collect();
                let x = t.affine(&input, &weights, f64::from(layer.biases[j]));
                let v = t.value(x);
                out.push(match layer.activation {
                    Activation::Identity => x,
                    Activation::SmoothHinge => t.unary(
                        x,
                        if v <= 0. {
                            0.
                        } else if v < 0.125 {
                            (4. * v) * v
                        } else {
                            v - 0.0625
                        },
                        if v <= 0. {
                            0.
                        } else if v < 0.125 {
                            8. * v
                        } else {
                            1.
                        },
                    ),
                    Activation::Relu => t.unary(x, v.max(0.), if v > 0. { 1. } else { 0. }),
                    Activation::LeakyRelu => t.unary(
                        x,
                        if v >= 0. { v } else { v * 0.01 },
                        if v >= 0. { 1. } else { 0.01 },
                    ),
                    Activation::Softplus | Activation::Silu => {
                        let e = (-v.abs()).exp();
                        let sigmoid = if v >= 0. { 1. / (1. + e) } else { e / (1. + e) };
                        if matches!(layer.activation, Activation::Softplus) {
                            t.unary(x, v.max(0.) + e.ln_1p(), sigmoid)
                        } else {
                            t.unary(x, v * sigmoid, sigmoid + v * sigmoid * (1. - sigmoid))
                        }
                    }
                    _ => return Err(error("unknown network activation in exact sensitivity")),
                });
            }
            input = out;
        }
        if let Some(m) = s.metadata.per_sample_alpha.as_deref() {
            let weights: Vec<_> = m.w_alpha.iter().map(|&w| f64::from(w)).collect();
            let logit = t.affine(&input, &weights, f64::from(m.b_alpha));
            let gate = t.sigmoid(logit, 20.);
            pool(
                t,
                &input,
                &m.rank_w,
                m.rank_b,
                &m.reducer_w,
                m.reducer_b,
                m.p_norm,
                gate,
            )
        } else if let Some(m) = s.metadata.hybrid_head.as_deref() {
            let logit = t.constant(f64::from(m.alpha_logit));
            let gate = t.sigmoid(logit, 20.);
            pool(
                t,
                &input,
                &m.rank_w,
                m.rank_b,
                &m.reducer_w,
                m.reducer_b,
                m.p_norm,
                gate,
            )
        } else {
            input[0]
        }
    };
    if let Some(scale) = s.metadata.tanh_pin_scale {
        let x = t.unary(y, t.value(y) / scale, 1. / scale);
        let v = t.sigmoid(x, 30.);
        y = t.unary(v, 100. * t.value(v), 100.);
    }
    #[cfg(all(test, steerfix_instrument))]
    match s.diagnostic_objective {
        steerfix_packet::Objective::PreFloor => return Ok(y),
        steerfix_packet::Objective::SmoothFloor => {
            let spline = s
                .metadata
                .output_spline
                .as_ref()
                .expect("smooth-floor spline");
            let span = spline.ys[spline.ys.len() - 1] - spline.ys[0];
            let floor = spline.ys[0] - span;
            let v = t.value(y);
            let linear = spline.ys[0] + spline.derivs[0] * (v - spline.xs[0]);
            if v <= spline.xs[0] && linear < floor {
                let z = (linear - floor) / span;
                return Ok(t.unary(
                    y,
                    floor + span * z.asinh(),
                    spline.derivs[0] / (1. + z * z).sqrt(),
                ));
            }
        }
        steerfix_packet::Objective::Served => {}
    }
    if let Some(spline) = s.metadata.output_spline.as_deref() {
        let v = t.value(y);
        y = t.unary(
            y,
            apply_output_calibration_spline(v, spline),
            spline_slope(v, spline),
        );
    }
    if let (Some(cal), Some(codec)) = (&s.metadata.per_codec_calibration, codec)
        && let Some((a, b)) = lookup_per_codec_affine(cal, codec)
    {
        y = t.unary(y, f64::from(a) + f64::from(b) * t.value(y), f64::from(b));
    }
    Ok(y)
}

fn score(
    t: &mut Tape,
    s: &BakeScorer<'_>,
    roots: &[Node],
    dimensions: (u32, u32),
    codec: Option<&str>,
) -> Result<Node, ZensimError> {
    let mut parts = Vec::new();
    let mut weights = Vec::new();
    let count = s.members.len() + 1;
    for (i, member) in std::iter::once(s).chain(s.members.iter()).enumerate() {
        let w = if let Some(weights) = &s.weights {
            weights[i]
        } else {
            1. / count as f64
        };
        if w != 0. {
            parts.push(if i == 0 {
                primary(t, s, roots, dimensions, codec)?
            } else {
                score(t, member, roots, dimensions, codec)?
            });
            weights.push(w);
        }
    }
    let mut y = t.affine(&parts, &weights, 0.);
    if let Some(p) = s.disposition {
        if !p.skip_score_mapping {
            let v = t.value(y);
            let a = p.score_mapping_a;
            let b = p.score_mapping_b;
            let slope = if v > 0. { -a * b * v.powf(b - 1.) } else { 0. };
            y = t.unary(y, distance_to_score_mapped(v, a, b), slope);
        }
        if !p.extrapolate_score {
            if p.soft_clamp_score {
                let x = t.unary(y, (t.value(y) - 50.) / 20., 1. / 20.);
                let v = 1. / (1. + (-t.value(x)).det_exp(active_pow_form()));
                let z = t.unary(x, v, v * (1. - v));
                y = t.unary(z, 100. * t.value(z), 100.);
            } else {
                y = t.clamp(y, 0., 100.);
            }
        }
    }
    #[cfg(feature = "corruption-head")]
    if let Some(head) = &s.corruption {
        let (h, threshold) = match head {
            Companion::Linear(h, threshold) => (score(t, h, roots, dimensions, codec)?, *threshold),
            Companion::Tree(h, threshold) => {
                let values: Vec<_> = roots
                    .iter()
                    .take(h.caller_input_width())
                    .map(|&x| t.value(x))
                    .collect();
                (
                    t.constant(
                        h.score_f64(&values)
                            .map_err(|_| error("corruption tree sensitivity forward failed"))?,
                    ),
                    *threshold,
                )
            }
        };
        if t.value(h) < threshold && t.value(h) < t.value(y) {
            y = h;
        }
    }
    Ok(y)
}

pub(super) fn gradient(
    s: &BakeScorer<'_>,
    features: &[f64],
    dimensions: (u32, u32),
    codec: Option<&str>,
) -> Result<Vec<f64>, ZensimError> {
    let mut t = Tape::default();
    let roots: Vec<_> = features.iter().map(|&x| t.constant(x)).collect();
    let y = score(&mut t, s, &roots, dimensions, codec)?;
    if !t.value(y).is_finite() {
        return Err(error("exact sensitivity forward is nonfinite"));
    }
    t.reverse(y, features.len())
}

fn spline_slope(x: f64, s: &OutputCalibrationSpline) -> f64 {
    let n = s.xs.len();
    if x <= s.xs[0] {
        return if s.ys[0] + s.derivs[0] * (x - s.xs[0]) > 2. * s.ys[0] - s.ys[n - 1] {
            s.derivs[0]
        } else {
            0.
        };
    }
    if x >= s.xs[n - 1] {
        return if s.ys[n - 1] + s.derivs[n - 1] * (x - s.xs[n - 1]) < 100. {
            s.derivs[n - 1]
        } else {
            0.
        };
    }
    if apply_output_calibration_spline(x, s) >= 100. {
        return 0.;
    }
    let i = s.xs.partition_point(|v| *v <= x) - 1;
    let h = s.xs[i + 1] - s.xs[i];
    let t = (x - s.xs[i]) / h;
    ((6. * t * t - 6. * t) * s.ys[i]
        + (3. * t * t - 4. * t + 1.) * h * s.derivs[i]
        + (-6. * t * t + 6. * t) * s.ys[i + 1]
        + (3. * t * t - 2. * t) * h * s.derivs[i + 1])
        / h
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn analytic_activation_sensitivity_matches_served_central_difference() {
        for activation in ["softplus", "silu"] {
            let spec = serde_json::json!({"schema_hash":1,"scaler_mean":[0.],"scaler_scale":[1.],
                "metadata":[{"key":"zentrain.feature_ids","type":"utf8","text":"13"}],
                "layers":[{"in_dim":1,"out_dim":1,"activation":activation,"dtype":"f32","weights":[1.],"biases":[-3.]},
                {"in_dim":1,"out_dim":1,"activation":"identity","dtype":"f32","weights":[-1.],"biases":[50.]}]});
            let bytes = zenpredict_bake::bake_from_json_str(&spec.to_string()).unwrap();
            let model = crate::mlp::Model::from_bytes(&bytes).unwrap();
            let mut scorer = BakeScorer::new(&model).unwrap();
            for value in [0.5, 2., 3., 4., 10.] {
                let mut row = vec![0.; 372];
                row[13] = value;
                let g = scorer
                    .score_features_fd_gradient(&row, 96, 96, None)
                    .unwrap()[13];
                let v = value - 3.;
                let sigmoid = 1. / (1. + (-v).exp());
                let expected = if activation == "softplus" {
                    -sigmoid
                } else {
                    -(sigmoid + v * sigmoid * (1. - sigmoid))
                };
                assert!((g - expected).abs() < 1e-12);
                row[13] = value + 0.002;
                let plus = scorer.score_features(&row, 96, 96, None).unwrap();
                row[13] = value - 0.002;
                let minus = scorer.score_features(&row, 96, 96, None).unwrap();
                assert!((g - (plus - minus) / 0.004).abs() < 0.002);
            }
        }
    }

    #[test]
    fn f32_f16_i8_storage_and_dispatch_have_exact_derivatives() {
        if !crate::ssim_form::run_in_own_process(
            "metric::bake::gradient::tests::f32_f16_i8_storage_and_dispatch_have_exact_derivatives",
            "D12_STORAGE_DISPATCH_OK",
            None,
        ) {
            return;
        }
        let _lock = archmage::testing::lock_token_testing();
        for dtype in ["f32", "f16", "i8"] {
            let spec = serde_json::json!({"schema_hash":1,"scaler_mean":[0.,0.],"scaler_scale":[2.,4.],
                "metadata":[{"key":"zentrain.feature_ids","type":"utf8","text":"13\n52"}],
                "layers":[{"in_dim":2,"out_dim":3,"activation":"relu","dtype":dtype,"weights":[1.,-1.,2.,0.5,2.,-1.],"biases":[1.,-1.,0.5]},
                {"in_dim":3,"out_dim":1,"activation":"identity","dtype":dtype,"weights":[-1.,-2.,-0.25],"biases":[100.]}]});
            let bytes = zenpredict_bake::bake_from_json_str(&spec.to_string()).unwrap();
            let model = crate::mlp::Model::from_bytes(&bytes).unwrap();
            let mut row = vec![0.; 372];
            row[13] = 3.;
            row[52] = 2.;
            let mut bits = None;
            let report = archmage::testing::for_each_token_permutation(
                archmage::testing::CompileTimePolicy::Warn,
                |_| {
                    let mut scorer = BakeScorer::new(&model).unwrap();
                    let before = scorer.score_features(&row, 96, 96, None).unwrap().to_bits();
                    let gradient = scorer
                        .score_features_fd_gradient(&row, 96, 96, None)
                        .unwrap();
                    assert_eq!(
                        before,
                        scorer.score_features(&row, 96, 96, None).unwrap().to_bits()
                    );
                    if dtype != "i8" {
                        assert_eq!(gradient[13], -0.75);
                        assert_eq!(gradient[52], -0.0625);
                    }
                    let current: Vec<_> = gradient.iter().map(|x| x.to_bits()).collect();
                    if let Some(bits) = &bits {
                        assert_eq!(&current, bits);
                    } else {
                        bits = Some(current);
                    }
                },
            );
            assert!(report.permutations_run > 0);
        }
        println!("D12_STORAGE_DISPATCH_OK");
    }
    #[test]
    fn transform_derivatives_match_independent_closed_forms_and_boundaries() {
        let cases = [
            (FeatureTransform::Log, 2., vec![], 0.5),
            (FeatureTransform::Log1p, 3., vec![], 0.25),
            (FeatureTransform::SignedLog1p, -3., vec![], 0.25),
            (FeatureTransform::SignedSqrt, -4., vec![], 0.25),
            (FeatureTransform::SignedCbrt, -8., vec![], 1. / 12.),
            (FeatureTransform::SoftSign, -3., vec![1.], 1. / 16.),
            (FeatureTransform::SoftClip, 3., vec![1., 2.], 0.5),
            (FeatureTransform::SignedPow, -4., vec![2.], 8.),
            (FeatureTransform::ClipThenLog1p, 4., vec![1.], 0.25),
            (FeatureTransform::ClipThenLog1p, 0., vec![1.], 0.),
            (FeatureTransform::WinsorP99, 4., vec![1., 3.], 0.),
            (FeatureTransform::WinsorThenLog, 2., vec![1., 3.], 0.5),
            (FeatureTransform::WinsorThenLog1p, 3., vec![1., 4.], 0.25),
            (
                FeatureTransform::WinsorThenSignedCbrt,
                8.,
                vec![1., 27.],
                1. / 12.,
            ),
            (
                FeatureTransform::SignedCbrtThenWinsor,
                8.,
                vec![1., 3.],
                1. / 12.,
            ),
            (
                FeatureTransform::ClipThenLog1pThenWinsor,
                4.,
                vec![1., 0., 2.],
                0.25,
            ),
            (FeatureTransform::QuantileBins, 2., vec![1., 3.], 0.),
            (FeatureTransform::YeoJohnson, 3., vec![0.], 0.25),
            (FeatureTransform::YeoJohnson, -3., vec![2.], 0.25),
            (FeatureTransform::YeoJohnson, 1e-6, vec![1e8], 1.),
        ];
        for (tf, x, p, expected) in cases {
            let mut t = Tape::default();
            let x = t.constant(x);
            let y = transform(&mut t, x, tf, &p).unwrap()[0];
            assert!(
                (t.reverse(y, 1).unwrap()[0] - expected).abs() < 1e-12,
                "{tf:?}"
            );
        }
        let mut t = Tape::default();
        let x = t.constant(0.25);
        let y = transform(&mut t, x, FeatureTransform::Sinusoidal, &[1.]).unwrap();
        assert_eq!(y.len(), 2);
        assert!((t.reverse(y[1], 1).unwrap()[0] + 2. * std::f64::consts::PI).abs() < 1e-12);
        assert!(
            transform(&mut t, x, FeatureTransform::Drop, &[])
                .unwrap()
                .is_empty()
        );
        let mut t = Tape::default();
        let x = t.constant(0.);
        let y = transform(&mut t, x, FeatureTransform::SignedSqrt, &[]).unwrap()[0];
        assert!(
            t.reverse(y, 1).is_err(),
            "a singular active derivative must not be fabricated"
        );
    }
    #[test]
    #[cfg(all(
        feature = "corruption-head",
        feature = "feature-regime-v2",
        feature = "custom-profiles"
    ))]
    fn floor_recovery_never_detaches_an_active_corruption_companion() {
        let mut payload = 2_u32.to_le_bytes().to_vec();
        for value in [-10_f32, -10., 100., 100.] {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        let hex: String = payload.iter().map(|x| format!("{x:02x}")).collect();
        let spec = serde_json::json!({"schema_hash":1,"scaler_mean":[0.],"scaler_scale":[1.],
            "metadata":[{"key":"zentrain.feature_ids","type":"utf8","text":"13"},
                {"key":"zentrain.formula_revision","type":"utf8","text":"5"},
                {"key":"zentrain.output_calibration_spline","type":"bytes","hex":hex}],
            "layers":[{"in_dim":1,"out_dim":1,"activation":"identity","dtype":"f32","weights":[-1000.],"biases":[-1000.]}]});
        let bytes = zenpredict_bake::bake_from_json_str(&spec.to_string()).unwrap();
        let base = crate::mlp::Model::from_bytes(&bytes).unwrap();
        let mut spec = spec;
        spec["metadata"].as_array_mut().unwrap().pop();
        spec["layers"][0]["weights"] = serde_json::json!([0.]);
        spec["layers"][0]["biases"] = serde_json::json!([-10000.]);
        let bytes = zenpredict_bake::bake_from_json_str(&spec.to_string()).unwrap();
        let head = crate::mlp::Model::from_bytes(&bytes).unwrap();
        let mut s = BakeScorer::new(&base)
            .unwrap()
            .with_linear_corruption_head(&head, 0.)
            .unwrap();
        let mut f = vec![0.; 720];
        f[13] = 1.;
        assert_eq!(s.score_features(&f, 96, 96, None).unwrap(), -10000.);
        assert!(
            s.score_features_fd_gradient(&f, 96, 96, None)
                .unwrap()
                .iter()
                .all(|&x| x == 0.)
        );
        assert!(
            s.prepared_steering_gradient(&f, 96, 96, None)
                .unwrap()
                .iter()
                .all(|&x| x == 0.)
        );
        assert_eq!(s.score_features(&f, 96, 96, None).unwrap(), -10000.);
    }
    #[test]
    fn per_sample_gate_chain_rule_and_spline_caps_are_exact() {
        let mut t = Tape::default();
        let h = [t.constant(1.), t.constant(3.)];
        let logit = t.affine(&h, &[1., -1.], 2.);
        let gate = t.sigmoid(logit, 20.);
        let y = pool(&mut t, &h, &[1., 1.], 0., &[1.; 4], 0., 2., gate);
        let g = t.reverse(y, 2).unwrap();
        let r = 5_f64.sqrt();
        assert!((g[0] + 1. / r).abs() < 1e-12);
        assert!((g[1] - (2. + 0.75 / r + r / 4.)).abs() < 1e-12);
        let s = OutputCalibrationSpline {
            xs: vec![0., 1., 2.],
            ys: vec![20., 50., 80.],
            derivs: vec![30.; 3],
        };
        for (x, expected) in [(-2., 0.), (-1., 30.), (0.5, 30.), (2.5, 30.), (3., 0.)] {
            assert_eq!(spline_slope(x, &s), expected);
        }
    }
    #[test]
    fn pool_heads_match_closed_form_on_nonuniform_hidden_values() {
        // h=[1,3]: mu=2, sigma=1, max=3, RMS=sqrt(5).
        // Constant alpha=1/2, rank=sum(h), pool=mu+sigma+max+RMS.
        let mut t = Tape::default();
        let h = [t.constant(1.), t.constant(3.)];
        let gate = t.constant(0.5);
        let y = pool(&mut t, &h, &[1., 1.], 0., &[1.; 4], 0., 2., gate);
        let g = t.reverse(y, 2).unwrap();
        assert!((g[0] - (0.5 + 0.25 / 5_f64.sqrt())).abs() < 1e-12);
        assert!((g[1] - (1.5 + 0.75 / 5_f64.sqrt())).abs() < 1e-12);
    }
}
