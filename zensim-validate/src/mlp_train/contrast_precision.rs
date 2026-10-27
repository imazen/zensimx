//! Registered contrast precision, conditional on the unchanged legal pair law.
//! This module consumes no RNG and never selects or replaces a training pair.

#[derive(Clone, Copy, Debug)]
pub(super) struct Point {
    pub x: f64,
    pub se: f64,
}

impl Point {
    fn validate(self) {
        assert!(self.x.is_finite() && self.x >= 0.);
        assert!(self.se.is_finite() && self.se >= 0.);
    }
}

pub(super) fn precision(a: Point, b: Point, sigma: f64) -> f64 {
    a.validate();
    b.validate();
    assert!(sigma.is_finite() && sigma >= 0.);
    let variance =
        a.se.max(0.05).powi(2) + b.se.max(0.05).powi(2) + sigma.powi(2) * (a.x - b.x).powi(2);
    assert!(variance.is_finite() && variance > 0.);
    variance.recip()
}

/// Exact integer multiplicity of one residue of a uniform unsigned word.
/// Production uses 64 bits; smaller spaces support exhaustive independent tests.
fn residue_count(index: usize, len: usize, bits: u32) -> u128 {
    assert!((1..=64).contains(&bits) && len > 0 && index < len);
    let space = 1_u128 << bits;
    assert!(len as u128 <= space);
    space / len as u128 + u128::from((index as u128) < space % len as u128)
}

#[derive(Debug)]
pub(super) struct Normalizer {
    pub mean_precision: f64,
    pub legal_ordered_pairs: usize,
    pub legal_draw_multiplicity: u128,
}

impl Normalizer {
    pub fn weight(&self, a: Point, b: Point, sigma: f64) -> f64 {
        precision(a, b, sigma) / self.mean_precision
    }
}

/// `legal` is supplied by the existing trainer's tie owner, not reimplemented.
/// Row positions must follow the original reference bucket's row order.
pub(super) fn normalize(
    points: &[Point],
    sigma: f64,
    word_bits: u32,
    legal: impl Fn(usize, usize) -> bool,
) -> Option<Normalizer> {
    assert!(sigma.is_finite() && sigma >= 0.);
    for point in points {
        point.validate();
    }
    let mut multiplicity = 0_u128;
    let mut weighted = 0.;
    let mut pairs = 0;
    let mut first: Option<f64> = None;
    let mut equal = true;
    for (a, &pa) in points.iter().enumerate() {
        for (b, &pb) in points.iter().enumerate() {
            if a == b || !legal(a, b) {
                continue;
            }
            let count = residue_count(a, points.len(), word_bits)
                * residue_count(b, points.len(), word_bits);
            multiplicity = multiplicity.checked_add(count).expect("pair mass overflow");
            let value = precision(pa, pb, sigma);
            if let Some(initial) = first {
                equal &= initial.to_bits() == value.to_bits();
            } else {
                first = Some(value);
            }
            weighted += count as f64 * value;
            pairs += 1;
        }
    }
    first.map(|initial| {
        // Preserve exact weight one for equal precision, including a single
        // legal unordered pair. Equal SE alone does not imply equal precision.
        let mean_precision = if equal {
            initial
        } else {
            weighted / multiplicity as f64
        };
        assert!(mean_precision.is_finite() && mean_precision > 0.);
        Normalizer {
            mean_precision,
            legal_ordered_pairs: pairs,
            legal_draw_multiplicity: multiplicity,
        }
    })
}
