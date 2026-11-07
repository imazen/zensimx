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
    let mut out = [0.0_f32; N];
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
        if x.is_nan() {
            return x;
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identity_and_piecewise_slopes_cover_knots_and_extremes() {
        let identity = DamageShape::parse(&["0.3", "1.7", "1", "1", "1"]).unwrap();
        for x in [-0.0_f32, 0., f32::MIN_POSITIVE, 0.3, 1.7, 1e10, f32::MAX] {
            assert_eq!(identity.eval(x).to_bits(), x.to_bits());
        }
        let shape = DamageShape::parse(&["0.25", "1", "2", "0.5", "3"]).unwrap();
        let mut last = 0.;
        for i in 0..1000 {
            let x = i as f32 / 100.;
            let y = shape.eval(x);
            assert!(y >= last);
            last = y;
            let (reference, _) = shape.value_slope(f64::from(x));
            assert!(
                (f64::from(y) - reference).abs()
                    <= 4. * f64::from(f32::EPSILON) * reference.max(1.)
            );
        }
        for x in [0.1, 0.5, 2.0] {
            let (_, slope) = shape.value_slope(x);
            let delta = (shape.value_slope(x + 1e-6).0 - shape.value_slope(x - 1e-6).0) / 2e-6;
            assert!((delta - slope).abs() < 1e-8);
        }
    }
    #[test]
    fn reference_gate_nonnegative_does_not_claim_monotonicity() {
        let gate = ReferenceGate::parse(&["1", "0.2", "2"]).unwrap();
        assert!(gate.eval(0.25) < gate.eval(0.));
        assert!(gate.eval(0.75) > gate.eval(0.5));
        for i in -20..121 {
            assert!(gate.eval(i as f32 / 100.) >= 0.);
        }
        for x in [0.1, 0.7] {
            let (_, slope) = gate.value_slope(x);
            let numeric = (gate.value_slope(x + 1e-6).0 - gate.value_slope(x - 1e-6).0) / 2e-6;
            assert!((numeric - slope).abs() < 1e-8);
        }
        let identity = ReferenceGate::parse(&["0", "0.5", "1"]).unwrap();
        for i in 0..100 {
            let x = i as f32 / 100.;
            assert_eq!(identity.eval(x), x);
        }
    }
    #[test]
    fn invalid_parameters_are_refused() {
        for parameters in [
            ["0", "1", "1", "1", "1"],
            ["1", "1", "1", "1", "1"],
            ["1", "2", "-1", "1", "1"],
            ["1", "2", "NaN", "1", "1"],
            ["1", "inf", "1", "1", "1"],
        ] {
            assert!(DamageShape::parse(&parameters).is_err());
        }
        assert!(ReferenceGate::parse(&["1", "-0.1", "1"]).is_err());
        assert!(ReferenceGate::parse(&["1", "NaN", "1"]).is_err());
    }
}
