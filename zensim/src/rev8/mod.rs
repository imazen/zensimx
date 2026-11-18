//! Experimental symmetric SDR scoring with an untrained head.

mod kernels;
mod stream;
#[cfg(test)]
mod tests;

use crate::score_math::pchip_eval_capped;

/// Geometry and feature recipe for the Rev8 prototype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// Four dyadic scales and algebraic symmetry.
    AlgS4,
    /// Five dyadic scales and algebraic symmetry.
    AlgS5,
    /// Five scales with softly oriented detail and artifact features.
    HybridS5,
}

impl Variant {
    const fn scales(self) -> usize {
        match self {
            Self::AlgS4 => 4,
            _ => 5,
        }
    }

    const fn hybrid(self) -> bool {
        matches!(self, Self::HybridS5)
    }
}

/// A borrowed sRGB8 image. Stride is in RGB pixels.
#[derive(Clone, Copy)]
pub struct RgbRows<'a> {
    pixels: &'a [[u8; 3]],
    width: usize,
    height: usize,
    stride: usize,
}

impl<'a> RgbRows<'a> {
    /// Validates nonempty dimensions and backing extent, including the last row.
    pub fn new(pixels: &'a [[u8; 3]], width: usize, height: usize, stride: usize) -> Option<Self> {
        let end = height
            .checked_sub(1)?
            .checked_mul(stride)?
            .checked_add(width)?;
        // Bound signed reflect coordinates and padded row extents.
        if width == 0 || stride < width || end > pixels.len() || end > isize::MAX as usize / 4 {
            return None;
        }
        Some(Self {
            pixels,
            width,
            height,
            stride,
        })
    }

    fn row(self, y: usize) -> &'a [[u8; 3]] {
        let start = y * self.stride;
        &self.pixels[start..start + self.width]
    }
}

/// Feature order within one (scale, XYB channel) group.
pub const FEATURE_NAMES: [&str; 24] = [
    "L1",
    "L2",
    "L4",
    "D21",
    "D42",
    "SD2",
    "SD4",
    "MAE",
    "MSE",
    "GMS",
    "edge_width",
    "BK",
    "RG",
    "Md",
    "Me",
    "Mf",
    "ART",
    "DET",
    "fragility",
    "SD8",
    "oriented_DET",
    "oriented_ART",
    "oriented_HL",
    "oriented_HG",
];
const COMMON: usize = 20;
const FEATURE_COUNT: usize = FEATURE_NAMES.len();
type Features = [f64; FEATURE_COUNT];

/// A synthetic 0–100 dial and the scalar features that feed it.
pub struct Result {
    /// Synthetic score with exact identity at 100.
    pub score: f64,
    /// Means indexed by scale, then X/Y/B, then [`FEATURE_NAMES`].
    /// Inactive scales and orientation slots are zero.
    pub features: [[Features; 3]; 5],
}

/// Compute a prototype score. Returns `None` for mismatched geometry.
/// With the `threads` feature, parallel calls use the caller's Rayon pool.
pub fn compute(a: RgbRows<'_>, b: RgbRows<'_>, variant: Variant, parallel: bool) -> Option<Result> {
    if (a.width, a.height) != (b.width, b.height) {
        return None;
    }
    let features = kernels::extract(a, b, variant, parallel);
    Some(Result {
        score: score(&features, variant),
        features,
    })
}

/// Name the runtime dispatch tier used by [`compute`].
pub fn tier() -> &'static str {
    kernels::tier()
}

fn score(features: &[[Features; 3]; 5], variant: Variant) -> f64 {
    // Synthetic maxout with two units of two affine pieces each.
    let mut units = [0.0; 4];
    for (s, scale) in features[..variant.scales()].iter().enumerate() {
        for (c, f) in scale.iter().enumerate() {
            let w = [0.85, 1.0, 0.65][c] * [1.0, 0.85, 0.7, 0.55, 0.4][s];
            units[0] += w * (f[0] + f[3] + 0.5 * f[13] + f[17]);
            units[1] += w * (f[5] + f[9] + f[14] + f[18]);
            units[2] += w * (f[7] + f[10] + f[11] + f[12] + f[15] + f[16]);
            units[3] += w * (f[4] + f[6] + f[8] + f[19] + f[20] + f[21] + f[22] + f[23]);
        }
    }
    let damage = units[0].max(units[1]) + units[2].max(units[3]);
    // Monotone PCHIP on synthetic knots (slopes from score_math::pchip_derivs).
    const X: [f64; 4] = [0.0, 0.125, 0.5, 1.5];
    const Y: [f64; 4] = [100.0, 90.0, 55.0, 0.0];
    const D: [f64; 4] = [
        -230.0 / 3.0,
        -6720.0 / 79.0,
        -101640.0 / 1411.0,
        -895.0 / 33.0,
    ];
    pchip_eval_capped(damage, &X, &Y, &D).clamp(0.0, 100.0)
}
