//! Private research input shapes. Parameters are f32 wire values, stored as
//! bits so declarations retain deterministic equality/order without NaNs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct DamageShape(pub(crate) [u32; 5]);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ReferenceGate(pub(crate) [u32; 3]);

fn values<const N: usize>(text: &[&str]) -> Result<[f32; N], &'static str> {
    if text.len() != N {
        return Err("shape: wrong parameter count");
    }
    let mut out = [0.; N];
    for (out, text) in out.iter_mut().zip(text) {
        *out = text.parse().map_err(|_| "shape: invalid parameter")?;
        if !out.is_finite() || *out < 0. {
            return Err("shape: parameters must be finite and nonnegative");
        }
    }
    Ok(out)
}

impl DamageShape {
    pub(crate) fn parse(text: &[&str]) -> Result<Self, &'static str> {
        let p = values::<5>(text)?;
        if !(p[0] > 0. && p[1] > p[0]) {
            return Err("shape: knots must be positive and increasing");
        }
        Ok(Self(p.map(f32::to_bits)))
    }
    pub(crate) fn parameters(self) -> [f32; 5] {
        self.0.map(f32::from_bits)
    }
    pub(crate) fn eval(self, x: f32) -> f32 {
        let [a, b, s0, s1, s2] = self.parameters();
        let x = if x < 0. { 0. } else { x };
        // Preserve exact identity initialization on the nonnegative domain.
        if s0 == 1. && s1 == 1. && s2 == 1. {
            return x;
        }
        let first = x.min(a);
        let second = (x - a).max(0.).min(b - a);
        let third = (x - b).max(0.);
        (first * s0 + second * s1) + third * s2
    }
    pub(crate) fn value_slope(self, x: f64) -> (f64, f64) {
        let [a, b, s0, s1, s2] = self.parameters().map(f64::from);
        if x <= 0. {
            return (0., 0.);
        }
        let y = x.min(a) * s0 + (x - a).max(0.).min(b - a) * s1 + (x - b).max(0.) * s2;
        (
            y,
            if x < a {
                s0
            } else if x < b {
                s1
            } else {
                s2
            },
        )
    }
}

impl ReferenceGate {
    pub(crate) fn parse(text: &[&str]) -> Result<Self, &'static str> {
        Ok(Self(values::<3>(text)?.map(f32::to_bits)))
    }
    pub(crate) fn parameters(self) -> [f32; 3] {
        self.0.map(f32::from_bits)
    }
    pub(crate) fn eval(self, x: f32) -> f32 {
        let [a, b, c] = self.parameters();
        if x <= 0. {
            return a;
        }
        if x >= 1. {
            return c;
        }
        let (lo, hi, t) = if x < 0.5 {
            (a, b, 2. * x)
        } else {
            (b, c, 2. * x - 1.)
        };
        (1. - t) * lo + t * hi
    }
    pub(crate) fn value_slope(self, x: f64) -> (f64, f64) {
        let [a, b, c] = self.parameters().map(f64::from);
        if x <= 0. {
            return (a, 0.);
        }
        if x >= 1. {
            return (c, 0.);
        }
        let (lo, hi, t) = if x < 0.5 {
            (a, b, 2. * x)
        } else {
            (b, c, 2. * x - 1.)
        };
        ((1. - t) * lo + t * hi, 2. * (hi - lo))
    }
}
