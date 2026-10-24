//! Color space conversion: linear RGB / sRGB → XYB
//!
//! XYB is the perceptual color space used by both ssimulacra2 and butteraugli.
//! X ≈ red-green opponent, Y ≈ luminance, B ≈ blue channel.
//! The cube root (LMS cone response) is the key nonlinearity.
//!
//! Two entry points for XYB conversion:
//! - [`srgb_to_positive_xyb_planar_into`]: sRGB u8 input (LUT-based linearization + SIMD)
//! - [`linear_to_positive_xyb_planar_into`]: linear f32 input (SIMD, skips LUT)
//!
//! RGBA compositing helpers produce linear f32 RGB output for all input formats,
//! ensuring identical XYB values regardless of input pixel format.

#[cfg(target_arch = "x86_64")]
use archmage::arcane;
use archmage::incant;
use archmage::magetypes;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use core::arch::x86_64::{__m512, __m512i};
use magetypes::simd::backends::F32x8Convert;
use magetypes::simd::backends::F32x16Convert;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::f32x8;
use magetypes::simd::generic::f32x8 as GenericF32x8;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::generic::f32x16;
use magetypes::simd::generic::f32x16 as GenericF32x16;

// Opsin absorbance matrix (from jpegli/ssimulacra2).
// `pub(crate)` so sibling feature modules (e.g. `xyb_lms_features`)
// can lift them without duplicating constants — single source of truth.
pub(crate) const K_M02: f32 = 0.078;
pub(crate) const K_M00: f32 = 0.30;
pub(crate) const K_M01: f32 = 1.0 - K_M02 - K_M00;
pub(crate) const K_M12: f32 = 0.078;
pub(crate) const K_M10: f32 = 0.23;
pub(crate) const K_M11: f32 = 1.0 - K_M12 - K_M10;
pub(crate) const K_M20: f32 = 0.243_422_69;
pub(crate) const K_M21: f32 = 0.204_767_45;
pub(crate) const K_M22: f32 = 1.0 - K_M20 - K_M21;
pub(crate) const K_B0: f32 = 0.003_793_073_4;

// ─── Gamut conversion matrices (linear light, row-major) ─────────────────
//
// Convert from wide-gamut linear RGB to sRGB linear RGB.
// Computed as: sRGB_from_XYZ × XYZ_from_<source>.
// All share the D65 whitepoint, so no chromatic adaptation is needed.

/// Display P3 linear → sRGB linear (3×3, row-major).
///
/// P3 and sRGB share D65 whitepoint; P3 has wider red/green primaries.
/// Matrix = M_srgb_from_xyz × M_xyz_from_p3.
#[rustfmt::skip]
const P3_TO_SRGB: [[f32; 3]; 3] = [
    [ 1.224_940_2, -0.224_940_2,  0.0        ],
    [-0.042_056_955, 1.042_056_9,  0.0        ],
    [-0.019_637_555, -0.078_636_04, 1.098_273_6],
];

/// BT.2020 linear → sRGB linear (3×3, row-major).
///
/// BT.2020 covers a much wider gamut than sRGB. Out-of-gamut colors
/// (negative sRGB values) are clamped to [0, 1].
/// Matrix = M_srgb_from_xyz × M_xyz_from_bt2020.
#[rustfmt::skip]
const BT2020_TO_SRGB: [[f32; 3]; 3] = [
    [ 1.660_491, -0.587_641_1, -0.072_849_9],
    [-0.124_550_5,  1.132_899_9, -0.008_349_4],
    [-0.018_151_0, -0.100_578_6,  1.118_729_6],
];

use crate::source::{ColorPrimaries, GamutMapping};

/// Apply a gamut conversion matrix to a linear RGB pixel.
///
/// Converts from the source color primaries to sRGB linear light.
/// For [`ColorPrimaries::Srgb`] this is a no-op. Out-of-gamut handling
/// follows `mapping` (issue #17): [`GamutMapping::Clip`] clamps to
/// \[0, 1\] (post-display-clamp semantics, the default);
/// [`GamutMapping::Preserve`] lets negative / >1 components flow into
/// XYB (the opsin stage's `max(0)` on the post-mix sum keeps the
/// cube-root domain valid), making codec gamut clipping detectable.
#[inline]
pub(crate) fn apply_gamut_matrix(
    rgb: &mut [f32; 3],
    primaries: ColorPrimaries,
    mapping: GamutMapping,
) {
    #[allow(unreachable_patterns)]
    let m = match primaries {
        ColorPrimaries::Srgb => return,
        ColorPrimaries::DisplayP3 => &P3_TO_SRGB,
        ColorPrimaries::Bt2020 => &BT2020_TO_SRGB,
        _ => return, // future variants: pass through unchanged
    };
    let [r, g, b] = *rgb;
    let out = [
        m[0][0] * r + m[0][1] * g + m[0][2] * b,
        m[1][0] * r + m[1][1] * g + m[1][2] * b,
        m[2][0] * r + m[2][1] * g + m[2][2] * b,
    ];
    #[allow(unreachable_patterns)]
    match mapping {
        GamutMapping::Clip => {
            rgb[0] = out[0].clamp(0.0, 1.0);
            rgb[1] = out[1].clamp(0.0, 1.0);
            rgb[2] = out[2].clamp(0.0, 1.0);
        }
        GamutMapping::Preserve => *rgb = out,
        // Future variants behave as the documented default (Clip).
        _ => {
            rgb[0] = out[0].clamp(0.0, 1.0);
            rgb[1] = out[1].clamp(0.0, 1.0);
            rgb[2] = out[2].clamp(0.0, 1.0);
        }
    }
}

/// Convert sRGB u8 to linear f32 via lookup table.
#[inline(always)]
pub(crate) fn srgb_u8_to_linear(v: u8) -> f32 {
    linear_srgb::default::srgb_u8_to_linear(v)
}

/// Convert sRGB u16 (0-65535) to linear f32.
#[inline]
pub(crate) fn srgb_u16_to_linear(v: u16) -> f32 {
    linear_srgb::default::srgb_u16_to_linear(v)
}

/// Fast cube root: bit manipulation + 2 Newton-Raphson iterations in f32.
/// Accurate to ~20 bits (sufficient for image quality metrics).
#[inline(always)]
pub(crate) fn cbrtf_fast(x: f32) -> f32 {
    // cbrt(0) = 0; the Halley iteration below would produce NaN for x=0
    // because t*numerator underflows below f32 min subnormal, yielding 0/0
    // in the second iteration.
    if x == 0.0 {
        return 0.0;
    }
    let mut t = cbrtf_initial(x);
    // Halley's method in f32 (each step roughly triples correct bits: 5→15→45)
    // Use mul_add to match SIMD Halley iterations: x.mul_add(2, r) and r.mul_add(2, x)
    // Division-first form t*(num/den) avoids underflow in t*num for tiny x.
    let mut r = t * t * t;
    t *= x.mul_add(2.0, r) / r.mul_add(2.0, x);
    r = t * t * t;
    t *= x.mul_add(2.0, r) / r.mul_add(2.0, x);
    t
}

/// Cube root initial estimate via bit manipulation (~5 bits accuracy).
/// Cheap integer-only operation; use as seed for Halley's refinement.
#[inline(always)]
fn cbrtf_initial(x: f32) -> f32 {
    const B1: u32 = 709_958_130;
    let ui = x.to_bits();
    let hx = (ui & 0x7FFF_FFFF) / 3 + B1;
    let ui_out = (ui & 0x8000_0000) | hx;
    f32::from_bits(ui_out)
}

/// Convert interleaved sRGB u8 to planar positive XYB, writing into pre-allocated buffers.
/// Each output slice must be at least `pixels.len()` long.
///
/// Used by the streaming pipeline in `streaming.rs` — avoids per-strip
/// allocations by reusing caller-owned XYB plane buffers.
pub fn srgb_to_positive_xyb_planar_into(
    pixels: &[[u8; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    incant!(
        srgb_to_positive_xyb_planar_inner(pixels, x_out, y_out, b_out),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}

/// [`srgb_to_positive_xyb_planar_into`] for a computation at `revision`:
/// the tier-dispatched production kernel through Rev3, the canonical body
/// ([`srgb_xyb_canon`]) at Rev4. The public form above stays production-only —
/// it is a bench export with no revision to honour.
pub(crate) fn srgb_to_positive_xyb_planar_into_at_revision(
    pixels: &[[u8; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    revision: crate::feature_defs::FormulaRevision,
) {
    let mode = crate::featcanon::mode(revision);
    if mode.active() {
        srgb_xyb_canon(pixels, x_out, y_out, b_out, mode, revision.cbrt_newton());
        return;
    }
    srgb_to_positive_xyb_planar_into(pixels, x_out, y_out, b_out);
}

/// Convert interleaved sRGB u8 to planar XYB (without positive shift).
#[allow(dead_code)]
pub fn srgb_to_xyb_planar(pixels: &[[u8; 3]]) -> [Vec<f32>; 3] {
    let n = pixels.len();
    let mut x_plane = vec![0.0f32; n];
    let mut y_plane = vec![0.0f32; n];
    let mut b_plane = vec![0.0f32; n];

    incant!(
        srgb_to_xyb_planar_inner(pixels, &mut x_plane, &mut y_plane, &mut b_plane),
        [v3, neon, wasm128, scalar]
    );

    [x_plane, y_plane, b_plane]
}

/// Convert interleaved sRGB u8 to planar XYB, writing into pre-allocated buffers.
#[allow(dead_code)]
pub fn srgb_to_xyb_planar_into(
    pixels: &[[u8; 3]],
    x_plane: &mut [f32],
    y_plane: &mut [f32],
    b_plane: &mut [f32],
) {
    incant!(
        srgb_to_xyb_planar_inner(pixels, x_plane, y_plane, b_plane),
        [v3, neon, wasm128, scalar]
    );
}

// Note: the standalone `make_positive_xyb` post-process function was
// removed in 0.3.0. The streaming pipeline now fuses the conversion +
// positive shift in `srgb_to_positive_xyb_planar_into` and
// `linear_to_positive_xyb_planar_into`, so the separate shift had no
// remaining callers.

// ============================================================================
// FEATCANON canonical + exact opsin conversion
//
// The chunk arithmetic of [`OpsinChunk`] written in inherent `f32` ops so that
// EVERY tier — and every pixel, including the remainder — gets the same bits
// by construction: fused `mul_add` matrix rows (the v3/v4/neon backends fuse;
// the scalar and wasm128 magetypes backends do not, which is one of the two
// known XYB tier divergences), `cbrt_midp_f32` lane-wise (bit-identical to the
// vector `cbrt_midp`: same seed, same unfused Halley steps), and the
// absorbance bias `-cbrtf_fast(K_B0)` exactly as the chunk path computes it.
//
// The remainder is always run through the chunk arithmetic (zero-padded),
// never the per-pixel `cbrtf_fast` form — so `n mod 8` no longer picks a
// different algorithm for the last pixels of a band on any tier.
//
// The `*_exact` sibling evaluates the same formula in f64 (real `f64::cbrt`)
// and rounds once at the f32 plane store — the featcanon "exact" oracle's
// convention: every arithmetic step exact, rounding only at storage.

/// The absorbance bias, `-cbrtf_fast(K_B0)`, as [`OpsinChunk::new`] builds it.
#[inline(always)]
fn absorbance_bias_f32() -> f32 {
    -cbrtf_fast(K_B0)
}

/// One pixel of canonical opsin absorbance mix + cube root + positive shift.
/// `cbrt7` selects revision 7's cube root ([`crate::det_math::cbrt_newton_f32`]).
#[inline(always)]
#[allow(clippy::manual_clamp)]
fn opsin_px_canon(ab: f32, r: f32, g: f32, b: f32, cbrt7: bool) -> (f32, f32, f32) {
    let m0 = K_M00
        .mul_add(r, K_M01.mul_add(g, K_M02.mul_add(b, K_B0)))
        .max(0.0);
    let m1 = K_M10
        .mul_add(r, K_M11.mul_add(g, K_M12.mul_add(b, K_B0)))
        .max(0.0);
    let m2 = K_M20
        .mul_add(r, K_M21.mul_add(g, K_M22.mul_add(b, K_B0)))
        .max(0.0);
    let root = |m: f32| {
        if cbrt7 {
            crate::det_math::cbrt_newton_f32(m)
        } else {
            magetypes::nostd_math::cbrt_midp_f32(m)
        }
    };
    let (t0, t1, t2) = (root(m0), root(m1), root(m2));
    let c0 = t0 + ab;
    let c1 = t1 + ab;
    let x = 0.5 * (c0 - c1);
    let y = 0.5 * (c0 + c1);
    (x.mul_add(14.0, 0.42), y + 0.01, (t2 - y) + 0.55)
}

/// Exact-mode sibling: same formula in f64 with the correctly-rounded(ish)
/// `f64::cbrt` — the real-arithmetic reference the candidates are measured
/// against.
#[cfg(feature = "oracle")]
#[inline(always)]
#[allow(clippy::manual_clamp)]
fn opsin_px_exact(ab: f64, r: f64, g: f64, b: f64, positive: bool) -> (f64, f64, f64) {
    let m0 = (K_M00 as f64)
        .mul_add(
            r,
            (K_M01 as f64).mul_add(g, (K_M02 as f64).mul_add(b, K_B0 as f64)),
        )
        .max(0.0);
    let m1 = (K_M10 as f64)
        .mul_add(
            r,
            (K_M11 as f64).mul_add(g, (K_M12 as f64).mul_add(b, K_B0 as f64)),
        )
        .max(0.0);
    let m2 = (K_M20 as f64)
        .mul_add(
            r,
            (K_M21 as f64).mul_add(g, (K_M22 as f64).mul_add(b, K_B0 as f64)),
        )
        .max(0.0);
    let t0 = m0.cbrt();
    let t1 = m1.cbrt();
    let t2 = m2.cbrt();
    let c0 = t0 + ab;
    let c1 = t1 + ab;
    let x = 0.5 * (c0 - c1);
    let y = 0.5 * (c0 + c1);
    if positive {
        (x.mul_add(14.0, 0.42), y + 0.01, (t2 - y) + 0.55)
    } else {
        (x, y, t2)
    }
}

/// Canonical sRGB8 → positive-XYB conversion driver. `mode` is the caller's
/// [`crate::featcanon::mode`]; with the `oracle` feature its exact arm
/// evaluates the same formula in f64 and rounds once at the plane store.
///
/// REV4VEC: full 16-pixel chunks run [`srgb_xyb_canon_vec`]'s lane-parallel
/// copy of the canonical per-element formula on the tiers with true
/// hardware FMA (x86 v4x/v4/v3, aarch64 NEON — every lane computes
/// [`opsin_px_canon`]'s op sequence bit-for-bit); every other tier and the
/// remainder run [`srgb_xyb_canon_body`], the same scalar body, so a pixel
/// gets identical bits wherever it sits.
#[allow(clippy::manual_clamp)]
pub(crate) fn srgb_xyb_canon(
    pixels: &[[u8; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    mode: crate::featcanon::Mode,
    cbrt7: bool,
) {
    #[cfg(feature = "oracle")]
    if mode.exact() {
        let n = pixels.len();
        let ab = -(K_B0 as f64).cbrt();
        for i in 0..n {
            let r = srgb_u8_to_linear(pixels[i][0]) as f64;
            let g = srgb_u8_to_linear(pixels[i][1]) as f64;
            let b = srgb_u8_to_linear(pixels[i][2]) as f64;
            let (x, y, bb) = opsin_px_exact(ab, r, g, b, true);
            x_out[i] = x as f32;
            y_out[i] = y as f32;
            b_out[i] = bb as f32;
        }
        return;
    }
    #[cfg(not(feature = "oracle"))]
    let _ = mode;
    incant!(
        srgb_xyb_canon_vec(pixels, x_out, y_out, b_out, cbrt7),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}

/// The per-element scalar canonical sRGB→XYB body — what the scalar and
/// wasm128 tier stubs run for every pixel, and what every fused tier runs
/// for the `n mod 16` remainder. The vector chunks must equal it
/// bit-for-bit; keep them in lockstep (REV4VEC).
fn srgb_xyb_canon_body(
    pixels: &[[u8; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    cbrt7: bool,
) {
    let ab = absorbance_bias_f32();
    for (i, &p) in pixels.iter().enumerate() {
        let (x, y, bb) = opsin_px_canon(
            ab,
            srgb_u8_to_linear(p[0]),
            srgb_u8_to_linear(p[1]),
            srgb_u8_to_linear(p[2]),
            cbrt7,
        );
        x_out[i] = x;
        y_out[i] = y;
        b_out[i] = bb;
    }
}

/// Canonical linear-f32 → XYB driver; `CLAMP` is the display-gamut clamp.
/// `mode` as for [`srgb_xyb_canon`]. REV4VEC as for [`srgb_xyb_canon`].
#[allow(clippy::manual_clamp)]
pub(crate) fn linear_xyb_canon<const CLAMP: bool>(
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    mode: crate::featcanon::Mode,
    cbrt7: bool,
) {
    #[cfg(feature = "oracle")]
    if mode.exact() {
        let n = pixels.len();
        let ab = -(K_B0 as f64).cbrt();
        for i in 0..n {
            let p = pixels[i];
            let (r, g, b) = if CLAMP {
                (
                    (p[0] as f64).max(0.0).min(1.0),
                    (p[1] as f64).max(0.0).min(1.0),
                    (p[2] as f64).max(0.0).min(1.0),
                )
            } else {
                (p[0] as f64, p[1] as f64, p[2] as f64)
            };
            let (x, y, bb) = opsin_px_exact(ab, r, g, b, true);
            x_out[i] = x as f32;
            y_out[i] = y as f32;
            b_out[i] = bb as f32;
        }
        return;
    }
    #[cfg(not(feature = "oracle"))]
    let _ = mode;
    incant!(
        linear_xyb_canon_vec::<CLAMP>(pixels, x_out, y_out, b_out, cbrt7),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}

/// One pixel of [`linear_xyb_canon`]'s scalar body — the optional display
/// clamp then [`opsin_px_canon`]. Shared by the fused tiers' remainder and
/// the scalar/wasm128 stubs (REV4VEC).
#[inline(always)]
#[allow(clippy::manual_clamp)]
fn linear_px_canon<const CLAMP: bool>(ab: f32, p: [f32; 3], cbrt7: bool) -> (f32, f32, f32) {
    let (r, g, b) = if CLAMP {
        (
            p[0].max(0.0).min(1.0),
            p[1].max(0.0).min(1.0),
            p[2].max(0.0).min(1.0),
        )
    } else {
        (p[0], p[1], p[2])
    };
    opsin_px_canon(ab, r, g, b, cbrt7)
}

/// The per-element scalar canonical linear→XYB body — as
/// [`srgb_xyb_canon_body`], but through [`linear_px_canon`].
fn linear_xyb_canon_body<const CLAMP: bool>(
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    cbrt7: bool,
) {
    let ab = absorbance_bias_f32();
    for (i, &p) in pixels.iter().enumerate() {
        let (x, y, bb) = linear_px_canon::<CLAMP>(ab, p, cbrt7);
        x_out[i] = x;
        y_out[i] = y;
        b_out[i] = bb;
    }
}

// ============================================================================
// REV4VEC lane-parallel canonical kernels
// ============================================================================
//
// The canonical per-element formulas are tier-independent by construction:
// every op is inherent-f32 IEEE — always-fused `mul_add`, `maxNum`/`minNum`,
// ties-even rounding, bit-level exponent reconstruction — so a 16-lane
// `f32x16` chunk on a true-FMA tier (x86 v4x/v4/v3, aarch64 NEON) computes
// the scalar canonical body bit-for-bit per lane. The magetypes scalar and
// wasm128 backends emulate `mul_add` as `a * b + c` (no hardware FMA), so
// those tiers keep the scalar canonical body for EVERY pixel: routing them
// through the generic vector would double-round every `mul_add`.
//
// [`CanonChunk`] is the [`OpsinChunk`] pattern widened to `f32x16` and
// extended with the PU21 canon constants; the `*_canon_vec` functions are
// the per-leaf `#[magetypes]` chunk drivers; the `*_vec_scalar` /
// `*_vec_wasm128` siblings are hand-written stubs that run the per-element
// scalar bodies (the `-scalar` in each tier list keeps `#[magetypes]` from
// emitting a non-canonical generic `_scalar` itself). Every driver runs the
// `n mod 16` remainder through the identical scalar body, so a pixel's bits
// never depend on where it sits in a band.

/// Splatted canonical constants (opsin + PU21) plus the arithmetic of one
/// 16-pixel chunk, in the generic magetypes `f32x16` vector. The single
/// definition every fused tier runs — on v3 and NEON `f32x16` decomposes
/// into two/four fused-FMA ops per lane, which is per-lane identical to one
/// fused `mul_add` each. Only x86_64 builds one: i686, wasm32 and aarch64 NEON run the scalar canonical body.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
struct CanonChunk<T: F32x16Convert> {
    token: T,
    /// Revision 7's cube root ([`crate::det_math::cbrt_newton_x16`]).
    cbrt7: bool,
    m: [[GenericF32x16<T>; 3]; 3],
    bias: GenericF32x16<T>,
    zero: GenericF32x16<T>,
    one: GenericF32x16<T>,
    ab: GenericF32x16<T>,
    half: GenericF32x16<T>,
    fourteen: GenericF32x16<T>,
    x_bias: GenericF32x16<T>,
    y_bias: GenericF32x16<T>,
    b_bias: GenericF32x16<T>,
    // PU21 canon (`pu21_encode_canon` + `/ PU_WHITE` + `PU_X_SCALE`).
    l_min: GenericF32x16<T>,
    l_max: GenericF32x16<T>,
    p: [GenericF32x16<T>; 7],
    pu_white: GenericF32x16<T>,
    pu_x_scale: GenericF32x16<T>,
}

#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
impl<T: F32x16Convert> CanonChunk<T> {
    #[inline(always)]
    fn new(token: T) -> Self {
        Self::new_at(token, false)
    }

    /// `cbrt7` selects revision 7's cube root in [`Self::opsin_positive`].
    #[inline(always)]
    fn new_at(token: T, cbrt7: bool) -> Self {
        let splat = |v: f32| GenericF32x16::splat_t(token, v);
        Self {
            token,
            cbrt7,
            m: [
                [splat(K_M00), splat(K_M01), splat(K_M02)],
                [splat(K_M10), splat(K_M11), splat(K_M12)],
                [splat(K_M20), splat(K_M21), splat(K_M22)],
            ],
            bias: splat(K_B0),
            zero: GenericF32x16::zero_t(token),
            one: splat(1.0),
            ab: splat(absorbance_bias_f32()),
            half: splat(0.5),
            fourteen: splat(14.0),
            x_bias: splat(0.42),
            y_bias: splat(0.01),
            b_bias: splat(0.55),
            l_min: splat(crate::pu21::PU21_L_MIN),
            l_max: splat(crate::pu21::PU21_L_MAX),
            p: [
                splat(crate::pu21::P[0]),
                splat(crate::pu21::P[1]),
                splat(crate::pu21::P[2]),
                splat(crate::pu21::P[3]),
                splat(crate::pu21::P[4]),
                splat(crate::pu21::P[5]),
                splat(crate::pu21::P[6]),
            ],
            pu_white: splat(PU_WHITE),
            pu_x_scale: splat(PU_X_SCALE),
        }
    }

    /// LUT-linearize 16 sRGB8 pixels and transpose to channel vectors —
    /// [`OpsinChunk::srgb_lanes`] at 16 lanes.
    #[inline(always)]
    fn srgb_lanes(
        &self,
        px: &[[u8; 3]; 16],
    ) -> (GenericF32x16<T>, GenericF32x16<T>, GenericF32x16<T>) {
        let mut r_arr = [0.0f32; 16];
        let mut g_arr = [0.0f32; 16];
        let mut b_arr = [0.0f32; 16];
        for i in 0..16 {
            let p = px[i];
            r_arr[i] = srgb_u8_to_linear(p[0]);
            g_arr[i] = srgb_u8_to_linear(p[1]);
            b_arr[i] = srgb_u8_to_linear(p[2]);
        }
        (
            GenericF32x16::from_array_t(self.token, r_arr),
            GenericF32x16::from_array_t(self.token, g_arr),
            GenericF32x16::from_array_t(self.token, b_arr),
        )
    }

    /// Gather 16 linear pixels to channel vectors, applying the `CLAMP`
    /// display-gamut clamp exactly as [`linear_px_canon`]
    /// (`.max(0).min(1)`).
    #[inline(always)]
    fn linear_lanes<const CLAMP: bool>(
        &self,
        px: &[[f32; 3]; 16],
    ) -> (GenericF32x16<T>, GenericF32x16<T>, GenericF32x16<T>) {
        let mut r_arr = [0.0f32; 16];
        let mut g_arr = [0.0f32; 16];
        let mut b_arr = [0.0f32; 16];
        for i in 0..16 {
            let p = px[i];
            r_arr[i] = p[0];
            g_arr[i] = p[1];
            b_arr[i] = p[2];
        }
        let r = GenericF32x16::from_array_t(self.token, r_arr);
        let g = GenericF32x16::from_array_t(self.token, g_arr);
        let b = GenericF32x16::from_array_t(self.token, b_arr);
        if CLAMP {
            (
                r.max(self.zero).min(self.one),
                g.max(self.zero).min(self.one),
                b.max(self.zero).min(self.one),
            )
        } else {
            (r, g, b)
        }
    }

    /// The canonical opsin absorbance mix — [`opsin_px_canon`]'s fused
    /// `mul_add` rows floored at zero — lane-parallel.
    #[inline(always)]
    fn mixed(
        &self,
        r: GenericF32x16<T>,
        g: GenericF32x16<T>,
        b: GenericF32x16<T>,
    ) -> [GenericF32x16<T>; 3] {
        let [m0, m1, m2] = &self.m;
        [
            m0[0]
                .mul_add(r, m0[1].mul_add(g, m0[2].mul_add(b, self.bias)))
                .max(self.zero),
            m1[0]
                .mul_add(r, m1[1].mul_add(g, m1[2].mul_add(b, self.bias)))
                .max(self.zero),
            m2[0]
                .mul_add(r, m2[1].mul_add(g, m2[2].mul_add(b, self.bias)))
                .max(self.zero),
        ]
    }

    /// [`opsin_px_canon`] lane-parallel: the mix, `cbrt_midp` (bit-identical
    /// to `cbrt_midp_f32`), and the positive-XYB shift.
    #[inline(always)]
    fn opsin_positive(
        &self,
        r: GenericF32x16<T>,
        g: GenericF32x16<T>,
        b: GenericF32x16<T>,
    ) -> (GenericF32x16<T>, GenericF32x16<T>, GenericF32x16<T>) {
        let [m0, m1, m2] = self.mixed(r, g, b);
        let (t0, t1, t2) = if self.cbrt7 {
            (
                // `mixed` ends in `max(0)`: the inputs are `+0` or positive.
                crate::det_math::cbrt_newton_nonneg_x16(self.token, m0),
                crate::det_math::cbrt_newton_nonneg_x16(self.token, m1),
                crate::det_math::cbrt_newton_nonneg_x16(self.token, m2),
            )
        } else {
            (m0.cbrt_midp(), m1.cbrt_midp(), m2.cbrt_midp())
        };
        let c0 = t0 + self.ab;
        let c1 = t1 + self.ab;
        let x = self.half * (c0 - c1);
        let y = self.half * (c0 + c1);
        (
            x.mul_add(self.fourteen, self.x_bias),
            y + self.y_bias,
            (t2 - y) + self.b_bias,
        )
    }

    /// [`crate::pu21::pu21_encode_canon`] lane-parallel: identical clamp
    /// and midp formula chain through the generated `f32x16`
    /// transcendentals, which are the same op sequence as the `det_math`
    /// scalar bodies on a fused tier.
    #[inline(always)]
    #[allow(clippy::manual_clamp)] // must match the scalar .max().min() lanes
    fn pu_encode(&self, v: GenericF32x16<T>) -> GenericF32x16<T> {
        let [p0, p1, p2, p3, p4, p5, p6] = self.p;
        let y = v.max(self.l_min).min(self.l_max);
        let yp = (p3 * y.log2_midp()).exp2_midp();
        let inner = (p0 + p1 * yp) / (self.one + p2 * yp);
        (p6 * ((p4 * inner.log2_midp()).exp2_midp() - p5)).max(self.zero)
    }

    /// [`pu_xyb_pixel_canon`] lane-parallel: the mix, canonical PU21 per
    /// channel divided by `PU_WHITE`, and the positive-XYB shift.
    #[inline(always)]
    fn pu_xyb(
        &self,
        r: GenericF32x16<T>,
        g: GenericF32x16<T>,
        b: GenericF32x16<T>,
    ) -> (GenericF32x16<T>, GenericF32x16<T>, GenericF32x16<T>) {
        let [m0, m1, m2] = self.mixed(r, g, b);
        let c0 = self.pu_encode(m0) / self.pu_white;
        let c1 = self.pu_encode(m1) / self.pu_white;
        let c2 = self.pu_encode(m2) / self.pu_white;
        let x = self.half * (c0 - c1);
        let y = self.half * (c0 + c1);
        (
            x.mul_add(self.pu_x_scale, self.x_bias),
            y + self.y_bias,
            (c2 - y) + self.b_bias,
        )
    }
}

/// Lane-parallel canonical sRGB8 → positive-XYB: [`srgb_xyb_canon_body`]'s
/// per-element formula through [`CanonChunk`], full 16-pixel chunks only;
/// the `n mod 16` remainder runs the scalar body on every tier.
/// `-scalar`: `#[magetypes]` would otherwise emit a generic `_scalar`
/// variant whose `mul_add` is unfused — not canonical. v4x is
/// [`srgb_xyb_canon_vec_v4x`]: the same arithmetic with the table held in
/// registers.
#[magetypes(define(f32x16), v4, v3, -scalar)]
fn srgb_xyb_canon_vec(
    token: Token,
    pixels: &[[u8; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    cbrt7: bool,
) {
    let k = CanonChunk::new_at(token, cbrt7);
    let n = pixels.len();
    let chunks = n / 16;
    for c in 0..chunks {
        let base = c * 16;
        // FIXED-SIZE ARRAY PATTERN: one range check at the boundary.
        let px: &[[u8; 3]; 16] = pixels[base..base + 16]
            .try_into()
            .expect("16 pixels per chunk");
        let (r, g, b) = k.srgb_lanes(px);
        let (x, y, bb) = k.opsin_positive(r, g, b);
        x_out[base..base + 16].copy_from_slice(&x.to_array());
        y_out[base..base + 16].copy_from_slice(&y.to_array());
        b_out[base..base + 16].copy_from_slice(&bb.to_array());
    }
    let ab = absorbance_bias_f32();
    for i in (chunks * 16)..n {
        let p = pixels[i];
        let (x, y, bb) = opsin_px_canon(
            ab,
            srgb_u8_to_linear(p[0]),
            srgb_u8_to_linear(p[1]),
            srgb_u8_to_linear(p[2]),
            cbrt7,
        );
        x_out[i] = x;
        y_out[i] = y;
        b_out[i] = bb;
    }
}

/// [`srgb_xyb_canon_vec`] on v4x, with the sRGB8 → linear table held in
/// sixteen registers ([`srgb_lanes_v4x`]) instead of three 16-lane gathers
/// per chunk (most of this kernel's time on Zen 5). The table values, the
/// opsin arithmetic ([`CanonChunk::opsin_positive`]) and the scalar
/// remainder are the generated variant's, so every output is bit-identical.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
fn srgb_xyb_canon_vec_v4x(
    token: archmage::X64V4xToken,
    pixels: &[[u8; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    cbrt7: bool,
) {
    let k = CanonChunk::new_at(token, cbrt7);
    let lut = srgb_lut_regs_v4x();
    let lut_bytes = srgb_lut_byte_planes_v4x();
    let n = pixels.len();
    // Sixty-four pixels at a time through the byte-plane table
    // ([`srgb64_channel_v4x`]), then sixteen through the register table.
    let groups = n / 64;
    for g in 0..groups {
        let base = g * 64;
        let px: &[[u8; 3]; 64] = pixels[base..base + 64]
            .try_into()
            .expect("64 pixels per group");
        let v = load_rgb64_v4x(px);
        let r = srgb64_channel_v4x::<0>(&lut_bytes, &v);
        let gr = srgb64_channel_v4x::<1>(&lut_bytes, &v);
        let b = srgb64_channel_v4x::<2>(&lut_bytes, &v);
        for m in 0..4 {
            let (x, y, bb) = k.opsin_positive(
                GenericF32x16::<archmage::X64V4xToken>::from_raw_t(token, r[m]),
                GenericF32x16::<archmage::X64V4xToken>::from_raw_t(token, gr[m]),
                GenericF32x16::<archmage::X64V4xToken>::from_raw_t(token, b[m]),
            );
            let o = base + 16 * m;
            x_out[o..o + 16].copy_from_slice(&x.to_array());
            y_out[o..o + 16].copy_from_slice(&y.to_array());
            b_out[o..o + 16].copy_from_slice(&bb.to_array());
        }
    }
    let chunks = n / 16;
    for c in groups * 4..chunks {
        let base = c * 16;
        let px: &[[u8; 3]; 16] = pixels[base..base + 16]
            .try_into()
            .expect("16 pixels per chunk");
        let (r, g, b) = srgb_lanes_v4x(&lut, px);
        let (x, y, bb) = k.opsin_positive(
            GenericF32x16::<archmage::X64V4xToken>::from_raw_t(token, r),
            GenericF32x16::<archmage::X64V4xToken>::from_raw_t(token, g),
            GenericF32x16::<archmage::X64V4xToken>::from_raw_t(token, b),
        );
        x_out[base..base + 16].copy_from_slice(&x.to_array());
        y_out[base..base + 16].copy_from_slice(&y.to_array());
        b_out[base..base + 16].copy_from_slice(&bb.to_array());
    }
    let ab = absorbance_bias_f32();
    for i in (chunks * 16)..n {
        let p = pixels[i];
        let (x, y, bb) = opsin_px_canon(
            ab,
            srgb_u8_to_linear(p[0]),
            srgb_u8_to_linear(p[1]),
            srgb_u8_to_linear(p[2]),
            cbrt7,
        );
        x_out[i] = x;
        y_out[i] = y;
        b_out[i] = bb;
    }
}

/// Byte positions that deinterleave 16 packed RGB8 pixels: R to bytes
/// 0..16, G to 16..32, B to 32..48.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
const DEINTERLEAVE_RGB16: [u8; 64] = {
    let mut p = [0u8; 64];
    let mut k = 0;
    while k < 16 {
        p[k] = (3 * k) as u8;
        p[16 + k] = (3 * k + 1) as u8;
        p[32 + k] = (3 * k + 2) as u8;
        k += 1;
    }
    p
};

/// Byte-permute tables that deinterleave 64 packed RGB8 pixels (192 bytes in
/// three registers) for channel `c`: byte `p = 16L + 4m + j` of the result is
/// the channel byte of pixel `16m + 4L + j`, the order in which the byte-plane
/// reassembly of [`srgb64_channel_v4x`] yields pixels `16m..16m + 16` in its
/// register `m`. `.0` permutes registers 0–1, `.1` registers 1–2, and `.2`
/// marks the bytes taken from `.1`.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
const fn deinterleave_rgb64(c: usize) -> ([u8; 64], [u8; 64], u64) {
    let mut lo = [0u8; 64];
    let mut hi = [0u8; 64];
    let mut mask = 0u64;
    let mut p = 0;
    while p < 64 {
        let q = 16 * ((p % 16) / 4) + 4 * (p / 16) + p % 4;
        let s = 3 * q + c;
        if s < 128 {
            lo[p] = s as u8;
        } else {
            hi[p] = (s - 64) as u8;
            mask |= 1u64 << p;
        }
        p += 1;
    }
    (lo, hi, mask)
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
const DEINTERLEAVE_RGB64: [([u8; 64], [u8; 64], u64); 3] = [
    deinterleave_rgb64(0),
    deinterleave_rgb64(1),
    deinterleave_rgb64(2),
];

/// The sRGB8 → linear table ([`srgb_u8_to_linear`]) as four byte planes
/// (byte `k` of each entry's little-endian bits), each four registers of 64
/// entries.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::rite(v4x, import_intrinsics)]
fn srgb_lut_byte_planes_v4x() -> [[__m512i; 4]; 4] {
    let lut: [f32; 256] = std::array::from_fn(|i| srgb_u8_to_linear(i as u8));
    std::array::from_fn(|k| {
        let plane: [u8; 256] = std::array::from_fn(|i| lut[i].to_bits().to_le_bytes()[k]);
        std::array::from_fn(|q| {
            let b: &[u8; 64] = plane[64 * q..64 * q + 64].try_into().unwrap();
            _mm512_loadu_si512(b)
        })
    })
}

/// 64 packed RGB8 pixels as three registers of 64 bytes.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::rite(v4x, import_intrinsics)]
fn load_rgb64_v4x(px: &[[u8; 3]; 64]) -> [__m512i; 3] {
    let bytes = px.as_flattened();
    std::array::from_fn(|q| {
        let b: &[u8; 64] = bytes[64 * q..64 * q + 64].try_into().unwrap();
        _mm512_loadu_si512(b)
    })
}

/// Channel `C` of 64 packed RGB8 pixels (`v`, three registers) to linear,
/// as four registers of sixteen pixels in order: two byte permutes and a
/// blend deinterleave the channel ([`DEINTERLEAVE_RGB64`]); each of the four
/// table byte planes is two 128-entry byte permutes blended on index bit 7;
/// two unpack levels reassemble each entry's four bytes. The entries are
/// copied byte for byte, so every value equals the scalar lookup.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::rite(v4x, import_intrinsics)]
fn srgb64_channel_v4x<const C: usize>(lut: &[[__m512i; 4]; 4], v: &[__m512i; 3]) -> [__m512; 4] {
    let (lo, hi, mask) = &DEINTERLEAVE_RGB64[C];
    let a = _mm512_permutex2var_epi8(v[0], _mm512_loadu_si512(lo), v[1]);
    let b = _mm512_permutex2var_epi8(v[1], _mm512_loadu_si512(hi), v[2]);
    let idx = _mm512_mask_blend_epi8(*mask, a, b);
    let upper = _mm512_movepi8_mask(idx);
    let planes: [__m512i; 4] = std::array::from_fn(|k| {
        let low = _mm512_permutex2var_epi8(lut[k][0], idx, lut[k][1]);
        let high = _mm512_permutex2var_epi8(lut[k][2], idx, lut[k][3]);
        _mm512_mask_blend_epi8(upper, low, high)
    });
    let lo01 = _mm512_unpacklo_epi8(planes[0], planes[1]);
    let hi01 = _mm512_unpackhi_epi8(planes[0], planes[1]);
    let lo23 = _mm512_unpacklo_epi8(planes[2], planes[3]);
    let hi23 = _mm512_unpackhi_epi8(planes[2], planes[3]);
    [
        _mm512_castsi512_ps(_mm512_unpacklo_epi16(lo01, lo23)),
        _mm512_castsi512_ps(_mm512_unpackhi_epi16(lo01, lo23)),
        _mm512_castsi512_ps(_mm512_unpacklo_epi16(hi01, hi23)),
        _mm512_castsi512_ps(_mm512_unpackhi_epi16(hi01, hi23)),
    ]
}

/// The sRGB8 → linear table ([`srgb_u8_to_linear`]) as sixteen registers of
/// sixteen entries.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::rite(v4x, import_intrinsics)]
fn srgb_lut_regs_v4x() -> [__m512; 16] {
    let lut: [f32; 256] = std::array::from_fn(|i| srgb_u8_to_linear(i as u8));
    std::array::from_fn(|k| _mm512_loadu_ps(lut[k * 16..k * 16 + 16].try_into().unwrap()))
}

/// One channel's sixteen table lookups, `idx` in `0..256` per 32-bit lane:
/// eight two-register permutes each serve 32 entries (index bits 0–4) and a
/// blend tree on bits 5–7 picks the group. The entries are copied, not
/// computed, so the result equals the scalar lookup bit for bit.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::rite(v4x, import_intrinsics)]
fn lut256_v4x(lut: &[__m512; 16], idx: __m512i) -> __m512 {
    let g: [__m512; 8] =
        std::array::from_fn(|k| _mm512_permutex2var_ps(lut[2 * k], idx, lut[2 * k + 1]));
    let b5 = _mm512_test_epi32_mask(idx, _mm512_set1_epi32(32));
    let b6 = _mm512_test_epi32_mask(idx, _mm512_set1_epi32(64));
    let b7 = _mm512_test_epi32_mask(idx, _mm512_set1_epi32(128));
    let s01 = _mm512_mask_blend_ps(b5, g[0], g[1]);
    let s23 = _mm512_mask_blend_ps(b5, g[2], g[3]);
    let s45 = _mm512_mask_blend_ps(b5, g[4], g[5]);
    let s67 = _mm512_mask_blend_ps(b5, g[6], g[7]);
    let s03 = _mm512_mask_blend_ps(b6, s01, s23);
    let s47 = _mm512_mask_blend_ps(b6, s45, s67);
    _mm512_mask_blend_ps(b7, s03, s47)
}

/// Sixteen packed RGB8 pixels to linear channel vectors: one byte permute
/// deinterleaves the 48 bytes, then [`lut256_v4x`] per channel.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::rite(v4x, import_intrinsics)]
fn srgb_lanes_v4x(lut: &[__m512; 16], px: &[[u8; 3]; 16]) -> (__m512, __m512, __m512) {
    let bytes = px.as_flattened();
    let lo: &[u8; 32] = bytes[..32].try_into().unwrap();
    let hi: &[u8; 16] = bytes[32..48].try_into().unwrap();
    let v = _mm512_inserti32x4::<2>(
        _mm512_castsi256_si512(_mm256_loadu_si256(lo)),
        _mm_loadu_si128(hi),
    );
    let d = _mm512_permutexvar_epi8(_mm512_loadu_si512(&DEINTERLEAVE_RGB16), v);
    let r = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(d));
    let g = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32::<1>(d));
    let b = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32::<2>(d));
    (lut256_v4x(lut, r), lut256_v4x(lut, g), lut256_v4x(lut, b))
}

/// The scalar tier's canonical sRGB→XYB: the per-element scalar body
/// (magetypes' generic `mul_add` is unfused on the scalar backend — the
/// scalar body is the only canonical form).
fn srgb_xyb_canon_vec_scalar(
    _token: archmage::ScalarToken,
    pixels: &[[u8; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    cbrt7: bool,
) {
    srgb_xyb_canon_body(pixels, x_out, y_out, b_out, cbrt7);
}

/// wasm128 has no hardware FMA — same scalar canonical body as the scalar
/// tier.
#[cfg(target_arch = "wasm32")]
fn srgb_xyb_canon_vec_wasm128(
    _token: archmage::Wasm128Token,
    pixels: &[[u8; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    cbrt7: bool,
) {
    srgb_xyb_canon_body(pixels, x_out, y_out, b_out, cbrt7);
}

/// NEON runs the scalar canonical body (2026-10-03): its vector `max`/`min` propagate NaN, while the canon clamps with
/// `f32::max`/`min`, which drop it, so a lane-parallel NEON chunk differed from the scalar canon on NaN inputs (CI aarch64).
#[cfg(target_arch = "aarch64")]
fn srgb_xyb_canon_vec_neon(
    _token: archmage::NeonToken,
    pixels: &[[u8; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    cbrt7: bool,
) {
    srgb_xyb_canon_body(pixels, x_out, y_out, b_out, cbrt7);
}

/// [`linear_xyb_canon`] lane-parallel — as [`srgb_xyb_canon_vec`], with
/// [`CanonChunk::linear_lanes`] for the input side.
#[magetypes(define(f32x16), v4x, v4, v3, -scalar)]
fn linear_xyb_canon_vec<const CLAMP: bool>(
    token: Token,
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    cbrt7: bool,
) {
    let k = CanonChunk::new_at(token, cbrt7);
    let n = pixels.len();
    let chunks = n / 16;
    for c in 0..chunks {
        let base = c * 16;
        let px: &[[f32; 3]; 16] = pixels[base..base + 16]
            .try_into()
            .expect("16 pixels per chunk");
        let (r, g, b) = k.linear_lanes::<CLAMP>(px);
        let (x, y, bb) = k.opsin_positive(r, g, b);
        x_out[base..base + 16].copy_from_slice(&x.to_array());
        y_out[base..base + 16].copy_from_slice(&y.to_array());
        b_out[base..base + 16].copy_from_slice(&bb.to_array());
    }
    let ab = absorbance_bias_f32();
    for i in (chunks * 16)..n {
        let (x, y, bb) = linear_px_canon::<CLAMP>(ab, pixels[i], cbrt7);
        x_out[i] = x;
        y_out[i] = y;
        b_out[i] = bb;
    }
}

fn linear_xyb_canon_vec_scalar<const CLAMP: bool>(
    _token: archmage::ScalarToken,
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    cbrt7: bool,
) {
    linear_xyb_canon_body::<CLAMP>(pixels, x_out, y_out, b_out, cbrt7);
}

#[cfg(target_arch = "wasm32")]
fn linear_xyb_canon_vec_wasm128<const CLAMP: bool>(
    _token: archmage::Wasm128Token,
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    cbrt7: bool,
) {
    linear_xyb_canon_body::<CLAMP>(pixels, x_out, y_out, b_out, cbrt7);
}

/// NEON runs the scalar canonical body (2026-10-03): its vector `max`/`min` propagate NaN, while the canon clamps with
/// `f32::max`/`min`, which drop it, so a lane-parallel NEON chunk differed from the scalar canon on NaN inputs (CI aarch64).
#[cfg(target_arch = "aarch64")]
fn linear_xyb_canon_vec_neon<const CLAMP: bool>(
    _token: archmage::NeonToken,
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    cbrt7: bool,
) {
    linear_xyb_canon_body::<CLAMP>(pixels, x_out, y_out, b_out, cbrt7);
}

// --- SIMD implementations ---

/// AVX-512 fused sRGB → XYB + make_positive: 16 pixels at a time.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn srgb_to_positive_xyb_planar_inner_v4(
    token: archmage::X64V4Token,
    pixels: &[[u8; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    let absorbance_bias = -cbrtf_fast(K_B0);

    let m00 = f32x16::splat_t(token, K_M00);
    let m01 = f32x16::splat_t(token, K_M01);
    let m02 = f32x16::splat_t(token, K_M02);
    let m10 = f32x16::splat_t(token, K_M10);
    let m11 = f32x16::splat_t(token, K_M11);
    let m12 = f32x16::splat_t(token, K_M12);
    let m20 = f32x16::splat_t(token, K_M20);
    let m21 = f32x16::splat_t(token, K_M21);
    let m22 = f32x16::splat_t(token, K_M22);
    let bias = f32x16::splat_t(token, K_B0);
    let zero = f32x16::zero_t(token);
    let ab = f32x16::splat_t(token, absorbance_bias);
    let half = f32x16::splat_t(token, 0.5);
    let fourteen = f32x16::splat_t(token, 14.0);
    let x_bias = f32x16::splat_t(token, 0.42);
    let y_bias = f32x16::splat_t(token, 0.01);
    let b_bias = f32x16::splat_t(token, 0.55);

    let n = pixels.len();
    let chunks = n / 16;

    for chunk in 0..chunks {
        let base = chunk * 16;

        let mut r_arr = [0.0f32; 16];
        let mut g_arr = [0.0f32; 16];
        let mut b_arr = [0.0f32; 16];
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance
        // Optimization"): ONE range check at the boundary, zero
        // interior. Indexing the `pixels` SLICE inside the loop
        // left 16 bounds checks per 16-pixel iteration in the
        // emitted code. Same loads, same order — bit-exact.
        let px: &[[u8; 3]; 16] = pixels[base..base + 16]
            .try_into()
            .expect("16 pixels per chunk");
        for i in 0..16 {
            let p = px[i];
            r_arr[i] = srgb_u8_to_linear(p[0]);
            g_arr[i] = srgb_u8_to_linear(p[1]);
            b_arr[i] = srgb_u8_to_linear(p[2]);
        }

        let r = f32x16::from_array_t(token, r_arr);
        let g = f32x16::from_array_t(token, g_arr);
        let b = f32x16::from_array_t(token, b_arr);

        let mixed0 = m00
            .mul_add(r, m01.mul_add(g, m02.mul_add(b, bias)))
            .max(zero);
        let mixed1 = m10
            .mul_add(r, m11.mul_add(g, m12.mul_add(b, bias)))
            .max(zero);
        let mixed2 = m20
            .mul_add(r, m21.mul_add(g, m22.mul_add(b, bias)))
            .max(zero);

        // SIMD cube root (~15 bits, single Halley iteration; safe because
        // mixed >= K_B0 ≈ 0.0038 — no zeros, denormals, NaN, or infinities).
        let t0 = mixed0.cbrt_midp();
        let t1 = mixed1.cbrt_midp();
        let t2 = mixed2.cbrt_midp();

        let c0 = t0 + ab;
        let c1 = t1 + ab;

        let x = half * (c0 - c1);
        let y = half * (c0 + c1);

        let x_pos = x.mul_add(fourteen, x_bias);
        let y_pos = y + y_bias;
        let b_pos = (t2 - y) + b_bias;

        x_out[base..base + 16].copy_from_slice(&x_pos.to_array());
        y_out[base..base + 16].copy_from_slice(&y_pos.to_array());
        b_out[base..base + 16].copy_from_slice(&b_pos.to_array());
    }

    // Remainder with AVX2 (f32x8)
    let v3 = token.v3();
    let absorbance_bias_neg = absorbance_bias;
    let ab8 = f32x8::splat_t(v3, absorbance_bias);
    let half8 = f32x8::splat_t(v3, 0.5);
    let zero8 = f32x8::zero_t(v3);
    let m00_8 = f32x8::splat_t(v3, K_M00);
    let m01_8 = f32x8::splat_t(v3, K_M01);
    let m02_8 = f32x8::splat_t(v3, K_M02);
    let m10_8 = f32x8::splat_t(v3, K_M10);
    let m11_8 = f32x8::splat_t(v3, K_M11);
    let m12_8 = f32x8::splat_t(v3, K_M12);
    let m20_8 = f32x8::splat_t(v3, K_M20);
    let m21_8 = f32x8::splat_t(v3, K_M21);
    let m22_8 = f32x8::splat_t(v3, K_M22);
    let bias8 = f32x8::splat_t(v3, K_B0);
    let fourteen8 = f32x8::splat_t(v3, 14.0);
    let x_bias8 = f32x8::splat_t(v3, 0.42);
    let y_bias8 = f32x8::splat_t(v3, 0.01);
    let b_bias8 = f32x8::splat_t(v3, 0.55);

    let rem_start = chunks * 16;
    let rem_chunks = (n - rem_start) / 8;
    for chunk in 0..rem_chunks {
        let base = rem_start + chunk * 8;
        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance
        // Optimization"): ONE range check at the boundary, zero
        // interior. Indexing the `pixels` SLICE inside the loop
        // left 8 bounds checks per 8-pixel iteration in the
        // emitted code. Same loads, same order — bit-exact.
        let px: &[[u8; 3]; 8] = pixels[base..base + 8]
            .try_into()
            .expect("8 pixels per chunk");
        for i in 0..8 {
            let p = px[i];
            r_arr[i] = srgb_u8_to_linear(p[0]);
            g_arr[i] = srgb_u8_to_linear(p[1]);
            b_arr[i] = srgb_u8_to_linear(p[2]);
        }
        let r = f32x8::from_array_t(v3, r_arr);
        let g = f32x8::from_array_t(v3, g_arr);
        let b = f32x8::from_array_t(v3, b_arr);

        let mixed0 = m00_8
            .mul_add(r, m01_8.mul_add(g, m02_8.mul_add(b, bias8)))
            .max(zero8);
        let mixed1 = m10_8
            .mul_add(r, m11_8.mul_add(g, m12_8.mul_add(b, bias8)))
            .max(zero8);
        let mixed2 = m20_8
            .mul_add(r, m21_8.mul_add(g, m22_8.mul_add(b, bias8)))
            .max(zero8);

        let t0 = mixed0.cbrt_midp();
        let t1 = mixed1.cbrt_midp();
        let t2 = mixed2.cbrt_midp();

        let c0 = t0 + ab8;
        let c1 = t1 + ab8;
        let x = half8 * (c0 - c1);
        let y = half8 * (c0 + c1);
        let x_pos = x.mul_add(fourteen8, x_bias8);
        let y_pos = y + y_bias8;
        let b_pos = (t2 - y) + b_bias8;

        x_out[base..base + 8].copy_from_slice(&x_pos.to_array());
        y_out[base..base + 8].copy_from_slice(&y_pos.to_array());
        b_out[base..base + 8].copy_from_slice(&b_pos.to_array());
    }

    // Scalar remainder
    for i in (rem_start + rem_chunks * 8)..n {
        let p = pixels[i];
        let r = srgb_u8_to_linear(p[0]);
        let g = srgb_u8_to_linear(p[1]);
        let b = srgb_u8_to_linear(p[2]);

        let mixed0 = K_M00
            .mul_add(r, K_M01.mul_add(g, K_M02.mul_add(b, K_B0)))
            .max(0.0);
        let mixed1 = K_M10
            .mul_add(r, K_M11.mul_add(g, K_M12.mul_add(b, K_B0)))
            .max(0.0);
        let mixed2 = K_M20
            .mul_add(r, K_M21.mul_add(g, K_M22.mul_add(b, K_B0)))
            .max(0.0);

        let c0 = cbrtf_fast(mixed0) + absorbance_bias_neg;
        let c1 = cbrtf_fast(mixed1) + absorbance_bias_neg;
        let c2 = cbrtf_fast(mixed2);

        let x = 0.5 * (c0 - c1);
        let y = 0.5 * (c0 + c1);

        x_out[i] = x.mul_add(14.0, 0.42);
        y_out[i] = y + 0.01;
        b_out[i] = (c2 - y) + 0.55;
    }
}
#[cfg(target_arch = "x86_64")]
#[arcane]
fn srgb_to_positive_xyb_planar_inner_v4x(
    token: archmage::X64V4xToken,
    pixels: &[[u8; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    let absorbance_bias = -cbrtf_fast(K_B0);

    let m00 = f32x16::splat_t(token, K_M00);
    let m01 = f32x16::splat_t(token, K_M01);
    let m02 = f32x16::splat_t(token, K_M02);
    let m10 = f32x16::splat_t(token, K_M10);
    let m11 = f32x16::splat_t(token, K_M11);
    let m12 = f32x16::splat_t(token, K_M12);
    let m20 = f32x16::splat_t(token, K_M20);
    let m21 = f32x16::splat_t(token, K_M21);
    let m22 = f32x16::splat_t(token, K_M22);
    let bias = f32x16::splat_t(token, K_B0);
    let zero = f32x16::zero_t(token);
    let ab = f32x16::splat_t(token, absorbance_bias);
    let half = f32x16::splat_t(token, 0.5);
    let fourteen = f32x16::splat_t(token, 14.0);
    let x_bias = f32x16::splat_t(token, 0.42);
    let y_bias = f32x16::splat_t(token, 0.01);
    let b_bias = f32x16::splat_t(token, 0.55);

    let n = pixels.len();
    let chunks = n / 16;

    for chunk in 0..chunks {
        let base = chunk * 16;

        let mut r_arr = [0.0f32; 16];
        let mut g_arr = [0.0f32; 16];
        let mut b_arr = [0.0f32; 16];
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance
        // Optimization"): ONE range check at the boundary, zero
        // interior. Indexing the `pixels` SLICE inside the loop
        // left 16 bounds checks per 16-pixel iteration in the
        // emitted code. Same loads, same order — bit-exact.
        let px: &[[u8; 3]; 16] = pixels[base..base + 16]
            .try_into()
            .expect("16 pixels per chunk");
        for i in 0..16 {
            let p = px[i];
            r_arr[i] = srgb_u8_to_linear(p[0]);
            g_arr[i] = srgb_u8_to_linear(p[1]);
            b_arr[i] = srgb_u8_to_linear(p[2]);
        }

        let r = f32x16::from_array_t(token, r_arr);
        let g = f32x16::from_array_t(token, g_arr);
        let b = f32x16::from_array_t(token, b_arr);

        let mixed0 = m00
            .mul_add(r, m01.mul_add(g, m02.mul_add(b, bias)))
            .max(zero);
        let mixed1 = m10
            .mul_add(r, m11.mul_add(g, m12.mul_add(b, bias)))
            .max(zero);
        let mixed2 = m20
            .mul_add(r, m21.mul_add(g, m22.mul_add(b, bias)))
            .max(zero);

        // SIMD cube root (~15 bits, single Halley iteration; safe because
        // mixed >= K_B0 ≈ 0.0038 — no zeros, denormals, NaN, or infinities).
        let t0 = mixed0.cbrt_midp();
        let t1 = mixed1.cbrt_midp();
        let t2 = mixed2.cbrt_midp();

        let c0 = t0 + ab;
        let c1 = t1 + ab;

        let x = half * (c0 - c1);
        let y = half * (c0 + c1);

        let x_pos = x.mul_add(fourteen, x_bias);
        let y_pos = y + y_bias;
        let b_pos = (t2 - y) + b_bias;

        x_out[base..base + 16].copy_from_slice(&x_pos.to_array());
        y_out[base..base + 16].copy_from_slice(&y_pos.to_array());
        b_out[base..base + 16].copy_from_slice(&b_pos.to_array());
    }

    // Remainder with AVX2 (f32x8)
    let v3 = token.v3();
    let absorbance_bias_neg = absorbance_bias;
    let ab8 = f32x8::splat_t(v3, absorbance_bias);
    let half8 = f32x8::splat_t(v3, 0.5);
    let zero8 = f32x8::zero_t(v3);
    let m00_8 = f32x8::splat_t(v3, K_M00);
    let m01_8 = f32x8::splat_t(v3, K_M01);
    let m02_8 = f32x8::splat_t(v3, K_M02);
    let m10_8 = f32x8::splat_t(v3, K_M10);
    let m11_8 = f32x8::splat_t(v3, K_M11);
    let m12_8 = f32x8::splat_t(v3, K_M12);
    let m20_8 = f32x8::splat_t(v3, K_M20);
    let m21_8 = f32x8::splat_t(v3, K_M21);
    let m22_8 = f32x8::splat_t(v3, K_M22);
    let bias8 = f32x8::splat_t(v3, K_B0);
    let fourteen8 = f32x8::splat_t(v3, 14.0);
    let x_bias8 = f32x8::splat_t(v3, 0.42);
    let y_bias8 = f32x8::splat_t(v3, 0.01);
    let b_bias8 = f32x8::splat_t(v3, 0.55);

    let rem_start = chunks * 16;
    let rem_chunks = (n - rem_start) / 8;
    for chunk in 0..rem_chunks {
        let base = rem_start + chunk * 8;
        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance
        // Optimization"): ONE range check at the boundary, zero
        // interior. Indexing the `pixels` SLICE inside the loop
        // left 8 bounds checks per 8-pixel iteration in the
        // emitted code. Same loads, same order — bit-exact.
        let px: &[[u8; 3]; 8] = pixels[base..base + 8]
            .try_into()
            .expect("8 pixels per chunk");
        for i in 0..8 {
            let p = px[i];
            r_arr[i] = srgb_u8_to_linear(p[0]);
            g_arr[i] = srgb_u8_to_linear(p[1]);
            b_arr[i] = srgb_u8_to_linear(p[2]);
        }
        let r = f32x8::from_array_t(v3, r_arr);
        let g = f32x8::from_array_t(v3, g_arr);
        let b = f32x8::from_array_t(v3, b_arr);

        let mixed0 = m00_8
            .mul_add(r, m01_8.mul_add(g, m02_8.mul_add(b, bias8)))
            .max(zero8);
        let mixed1 = m10_8
            .mul_add(r, m11_8.mul_add(g, m12_8.mul_add(b, bias8)))
            .max(zero8);
        let mixed2 = m20_8
            .mul_add(r, m21_8.mul_add(g, m22_8.mul_add(b, bias8)))
            .max(zero8);

        let t0 = mixed0.cbrt_midp();
        let t1 = mixed1.cbrt_midp();
        let t2 = mixed2.cbrt_midp();

        let c0 = t0 + ab8;
        let c1 = t1 + ab8;
        let x = half8 * (c0 - c1);
        let y = half8 * (c0 + c1);
        let x_pos = x.mul_add(fourteen8, x_bias8);
        let y_pos = y + y_bias8;
        let b_pos = (t2 - y) + b_bias8;

        x_out[base..base + 8].copy_from_slice(&x_pos.to_array());
        y_out[base..base + 8].copy_from_slice(&y_pos.to_array());
        b_out[base..base + 8].copy_from_slice(&b_pos.to_array());
    }

    // Scalar remainder
    for i in (rem_start + rem_chunks * 8)..n {
        let p = pixels[i];
        let r = srgb_u8_to_linear(p[0]);
        let g = srgb_u8_to_linear(p[1]);
        let b = srgb_u8_to_linear(p[2]);

        let mixed0 = K_M00
            .mul_add(r, K_M01.mul_add(g, K_M02.mul_add(b, K_B0)))
            .max(0.0);
        let mixed1 = K_M10
            .mul_add(r, K_M11.mul_add(g, K_M12.mul_add(b, K_B0)))
            .max(0.0);
        let mixed2 = K_M20
            .mul_add(r, K_M21.mul_add(g, K_M22.mul_add(b, K_B0)))
            .max(0.0);

        let c0 = cbrtf_fast(mixed0) + absorbance_bias_neg;
        let c1 = cbrtf_fast(mixed1) + absorbance_bias_neg;
        let c2 = cbrtf_fast(mixed2);

        let x = 0.5 * (c0 - c1);
        let y = 0.5 * (c0 + c1);

        x_out[i] = x.mul_add(14.0, 0.42);
        y_out[i] = y + 0.01;
        b_out[i] = (c2 - y) + 0.55;
    }
}

/// Fused sRGB → XYB + make_positive in one pass with vectorized Halley iterations.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn srgb_to_positive_xyb_planar_inner_v3(
    token: archmage::X64V3Token,
    pixels: &[[u8; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    let absorbance_bias = -cbrtf_fast(K_B0);

    let m00 = f32x8::splat_t(token, K_M00);
    let m01 = f32x8::splat_t(token, K_M01);
    let m02 = f32x8::splat_t(token, K_M02);
    let m10 = f32x8::splat_t(token, K_M10);
    let m11 = f32x8::splat_t(token, K_M11);
    let m12 = f32x8::splat_t(token, K_M12);
    let m20 = f32x8::splat_t(token, K_M20);
    let m21 = f32x8::splat_t(token, K_M21);
    let m22 = f32x8::splat_t(token, K_M22);
    let bias = f32x8::splat_t(token, K_B0);
    let zero = f32x8::zero_t(token);
    let ab = f32x8::splat_t(token, absorbance_bias);
    let half = f32x8::splat_t(token, 0.5);
    // Positive-shift constants
    let fourteen = f32x8::splat_t(token, 14.0);
    let x_bias = f32x8::splat_t(token, 0.42);
    let y_bias = f32x8::splat_t(token, 0.01);
    let b_bias = f32x8::splat_t(token, 0.55);

    let n = pixels.len();
    let chunks = n / 8;

    for chunk in 0..chunks {
        let base = chunk * 8;

        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance
        // Optimization"): ONE range check at the boundary, zero
        // interior. Indexing the `pixels` SLICE inside the loop
        // left 8 bounds checks per 8-pixel iteration in the
        // emitted code. Same loads, same order — bit-exact.
        let px: &[[u8; 3]; 8] = pixels[base..base + 8]
            .try_into()
            .expect("8 pixels per chunk");
        for i in 0..8 {
            let p = px[i];
            r_arr[i] = srgb_u8_to_linear(p[0]);
            g_arr[i] = srgb_u8_to_linear(p[1]);
            b_arr[i] = srgb_u8_to_linear(p[2]);
        }

        let r = f32x8::from_array_t(token, r_arr);
        let g = f32x8::from_array_t(token, g_arr);
        let b = f32x8::from_array_t(token, b_arr);

        let mixed0 = m00
            .mul_add(r, m01.mul_add(g, m02.mul_add(b, bias)))
            .max(zero);
        let mixed1 = m10
            .mul_add(r, m11.mul_add(g, m12.mul_add(b, bias)))
            .max(zero);
        let mixed2 = m20
            .mul_add(r, m21.mul_add(g, m22.mul_add(b, bias)))
            .max(zero);

        let t0 = mixed0.cbrt_midp();
        let t1 = mixed1.cbrt_midp();
        let t2 = mixed2.cbrt_midp();

        let c0 = t0 + ab;
        let c1 = t1 + ab;

        let x = half * (c0 - c1);
        let y = half * (c0 + c1);

        // Fused make_positive: X*14+0.42, Y+0.01, (B-Y)+0.55
        let x_pos = x.mul_add(fourteen, x_bias);
        let y_pos = y + y_bias;
        let b_pos = (t2 - y) + b_bias;

        x_out[base..base + 8].copy_from_slice(&x_pos.to_array());
        y_out[base..base + 8].copy_from_slice(&y_pos.to_array());
        b_out[base..base + 8].copy_from_slice(&b_pos.to_array());
    }

    // Scalar remainder
    let absorbance_bias_neg = absorbance_bias;
    for i in (chunks * 8)..n {
        let p = pixels[i];
        let r = srgb_u8_to_linear(p[0]);
        let g = srgb_u8_to_linear(p[1]);
        let b = srgb_u8_to_linear(p[2]);

        let mixed0 = K_M00
            .mul_add(r, K_M01.mul_add(g, K_M02.mul_add(b, K_B0)))
            .max(0.0);
        let mixed1 = K_M10
            .mul_add(r, K_M11.mul_add(g, K_M12.mul_add(b, K_B0)))
            .max(0.0);
        let mixed2 = K_M20
            .mul_add(r, K_M21.mul_add(g, K_M22.mul_add(b, K_B0)))
            .max(0.0);

        let c0 = cbrtf_fast(mixed0) + absorbance_bias_neg;
        let c1 = cbrtf_fast(mixed1) + absorbance_bias_neg;
        let c2 = cbrtf_fast(mixed2);

        let x = 0.5 * (c0 - c1);
        let y = 0.5 * (c0 + c1);

        x_out[i] = x.mul_add(14.0, 0.42);
        y_out[i] = y + 0.01;
        b_out[i] = (c2 - y) + 0.55;
    }
}

/// How a generic tier treats the last `n mod 8` pixels of a band.
///
/// `false` (neon, wasm128): the per-pixel form with `cbrtf_fast`, as before.
/// `true` (scalar): zero-pad them into one more 8-pixel array and run the chunk
/// arithmetic, so a pixel gets the same bits wherever it sits in a band. On the
/// scalar tier magetypes' `mul_add` is unfused and `cbrt_midp` differs from
/// `cbrtf_fast`, so without this a colour's bits depended on its position
/// (CLAUDE.md Known Bugs, 2026-09-25; user decision 2026-09-25 to change i686).
trait PaddedTail {
    const PAD_TAIL: bool;
}
impl PaddedTail for archmage::ScalarToken {
    const PAD_TAIL: bool = true;
}
#[cfg(target_arch = "x86_64")]
impl PaddedTail for archmage::X64V3Token {
    const PAD_TAIL: bool = false;
}
#[cfg(target_arch = "aarch64")]
impl PaddedTail for archmage::NeonToken {
    const PAD_TAIL: bool = false;
}
#[cfg(target_arch = "wasm32")]
impl PaddedTail for archmage::Wasm128Token {
    const PAD_TAIL: bool = false;
}

/// Splatted opsin constants plus the arithmetic of ONE 8-pixel chunk, in the generic
/// magetypes vector types. This is the single definition every generic tier runs
/// (neon, wasm128 and scalar): the chunk loops and the scalar tier's padded remainder
/// all call it, so a pixel gets the same bits wherever it sits in a band.
struct OpsinChunk<T: F32x8Convert> {
    token: T,
    m: [[GenericF32x8<T>; 3]; 3],
    bias: GenericF32x8<T>,
    zero: GenericF32x8<T>,
    one: GenericF32x8<T>,
    ab: GenericF32x8<T>,
    half: GenericF32x8<T>,
    fourteen: GenericF32x8<T>,
    x_bias: GenericF32x8<T>,
    y_bias: GenericF32x8<T>,
    b_bias: GenericF32x8<T>,
}

impl<T: F32x8Convert> OpsinChunk<T> {
    #[inline(always)]
    fn new(token: T) -> Self {
        let absorbance_bias = -cbrtf_fast(K_B0);
        let splat = |v: f32| GenericF32x8::splat_t(token, v);
        Self {
            token,
            m: [
                [splat(K_M00), splat(K_M01), splat(K_M02)],
                [splat(K_M10), splat(K_M11), splat(K_M12)],
                [splat(K_M20), splat(K_M21), splat(K_M22)],
            ],
            bias: splat(K_B0),
            zero: GenericF32x8::zero_t(token),
            one: splat(1.0),
            ab: splat(absorbance_bias),
            half: splat(0.5),
            fourteen: splat(14.0),
            x_bias: splat(0.42),
            y_bias: splat(0.01),
            b_bias: splat(0.55),
        }
    }

    /// LUT-linearize 8 sRGB8 pixels and transpose to one vector per channel.
    #[inline(always)]
    fn srgb_lanes(&self, px: &[[u8; 3]; 8]) -> (GenericF32x8<T>, GenericF32x8<T>, GenericF32x8<T>) {
        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];
        for i in 0..8 {
            let p = px[i];
            r_arr[i] = srgb_u8_to_linear(p[0]);
            g_arr[i] = srgb_u8_to_linear(p[1]);
            b_arr[i] = srgb_u8_to_linear(p[2]);
        }
        (
            GenericF32x8::from_array_t(self.token, r_arr),
            GenericF32x8::from_array_t(self.token, g_arr),
            GenericF32x8::from_array_t(self.token, b_arr),
        )
    }

    /// Opsin absorbance matrix (fused multiply-add chains, floored at zero), then the
    /// vector cube root `cbrt_midp`.
    #[inline(always)]
    fn cube_roots(
        &self,
        r: GenericF32x8<T>,
        g: GenericF32x8<T>,
        b: GenericF32x8<T>,
    ) -> [GenericF32x8<T>; 3] {
        let [m0, m1, m2] = &self.m;
        let mixed0 = m0[0]
            .mul_add(r, m0[1].mul_add(g, m0[2].mul_add(b, self.bias)))
            .max(self.zero);
        let mixed1 = m1[0]
            .mul_add(r, m1[1].mul_add(g, m1[2].mul_add(b, self.bias)))
            .max(self.zero);
        let mixed2 = m2[0]
            .mul_add(r, m2[1].mul_add(g, m2[2].mul_add(b, self.bias)))
            .max(self.zero);
        [mixed0.cbrt_midp(), mixed1.cbrt_midp(), mixed2.cbrt_midp()]
    }

    /// X = 0.5*(c0-c1), Y = 0.5*(c0+c1), B = t2.
    #[inline(always)]
    fn plain(
        &self,
        t: [GenericF32x8<T>; 3],
    ) -> (GenericF32x8<T>, GenericF32x8<T>, GenericF32x8<T>) {
        let c0 = t[0] + self.ab;
        let c1 = t[1] + self.ab;
        (self.half * (c0 - c1), self.half * (c0 + c1), t[2])
    }

    /// `plain` fused with make_positive: X*14+0.42, Y+0.01, (B-Y)+0.55.
    #[inline(always)]
    fn positive(
        &self,
        t: [GenericF32x8<T>; 3],
    ) -> (GenericF32x8<T>, GenericF32x8<T>, GenericF32x8<T>) {
        let (x, y, _) = self.plain(t);
        let x_pos = x.mul_add(self.fourteen, self.x_bias);
        let y_pos = y + self.y_bias;
        let b_pos = (t[2] - y) + self.b_bias;
        (x_pos, y_pos, b_pos)
    }
}

/// sRGB → positive XYB, shared body of the generic tiers.
///
/// The remainder follows [`PaddedTail::PAD_TAIL`]: the per-pixel `cbrtf_fast` form on
/// neon and wasm128 (unchanged), one zero-padded chunk on the scalar tier.
#[inline(always)]
fn srgb_positive_xyb_generic<T: F32x8Convert + PaddedTail>(
    token: T,
    pixels: &[[u8; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    /// One 8-pixel chunk.
    #[inline(always)]
    fn chunk<T: F32x8Convert>(
        k: &OpsinChunk<T>,
        px: &[[u8; 3]; 8],
    ) -> ([f32; 8], [f32; 8], [f32; 8]) {
        let (r, g, b) = k.srgb_lanes(px);
        let (x_pos, y_pos, b_pos) = k.positive(k.cube_roots(r, g, b));
        (x_pos.to_array(), y_pos.to_array(), b_pos.to_array())
    }

    let k = OpsinChunk::new(token);
    let n = pixels.len();
    let chunks = n / 8;

    for chunk_idx in 0..chunks {
        let base = chunk_idx * 8;
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance Optimization"): ONE
        // range check at the boundary, zero interior. Indexing the `pixels` SLICE
        // inside the loop left 8 bounds checks per 8-pixel iteration.
        let px: &[[u8; 3]; 8] = pixels[base..base + 8]
            .try_into()
            .expect("8 pixels per chunk");
        let (xs, ys, bs) = chunk(&k, px);
        x_out[base..base + 8].copy_from_slice(&xs);
        y_out[base..base + 8].copy_from_slice(&ys);
        b_out[base..base + 8].copy_from_slice(&bs);
    }

    let done = chunks * 8;
    if T::PAD_TAIL {
        let rem = n - done;
        if rem != 0 {
            let mut padded = [[0u8; 3]; 8];
            padded[..rem].copy_from_slice(&pixels[done..]);
            let (xs, ys, bs) = chunk(&k, &padded);
            x_out[done..n].copy_from_slice(&xs[..rem]);
            y_out[done..n].copy_from_slice(&ys[..rem]);
            b_out[done..n].copy_from_slice(&bs[..rem]);
        }
    } else {
        let absorbance_bias = -cbrtf_fast(K_B0);
        // Scalar remainder
        for i in done..n {
            let p = pixels[i];
            let r = srgb_u8_to_linear(p[0]);
            let g = srgb_u8_to_linear(p[1]);
            let b = srgb_u8_to_linear(p[2]);

            let mixed0 = K_M00
                .mul_add(r, K_M01.mul_add(g, K_M02.mul_add(b, K_B0)))
                .max(0.0);
            let mixed1 = K_M10
                .mul_add(r, K_M11.mul_add(g, K_M12.mul_add(b, K_B0)))
                .max(0.0);
            let mixed2 = K_M20
                .mul_add(r, K_M21.mul_add(g, K_M22.mul_add(b, K_B0)))
                .max(0.0);

            let c0 = cbrtf_fast(mixed0) + absorbance_bias;
            let c1 = cbrtf_fast(mixed1) + absorbance_bias;
            let c2 = cbrtf_fast(mixed2);

            let x = 0.5 * (c0 - c1);
            let y = 0.5 * (c0 + c1);

            x_out[i] = x.mul_add(14.0, 0.42);
            y_out[i] = y + 0.01;
            b_out[i] = (c2 - y) + 0.55;
        }
    }
}

/// Generic fused sRGB → XYB + make_positive with vectorized Halley iterations.
#[magetypes(neon, wasm128, scalar)]
fn srgb_to_positive_xyb_planar_inner(
    token: Token,
    pixels: &[[u8; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    srgb_positive_xyb_generic(token, pixels, x_out, y_out, b_out);
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn srgb_to_xyb_planar_inner_v3(
    token: archmage::X64V3Token,
    pixels: &[[u8; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    let absorbance_bias = -cbrtf_fast(K_B0);

    let m00 = f32x8::splat_t(token, K_M00);
    let m01 = f32x8::splat_t(token, K_M01);
    let m02 = f32x8::splat_t(token, K_M02);
    let m10 = f32x8::splat_t(token, K_M10);
    let m11 = f32x8::splat_t(token, K_M11);
    let m12 = f32x8::splat_t(token, K_M12);
    let m20 = f32x8::splat_t(token, K_M20);
    let m21 = f32x8::splat_t(token, K_M21);
    let m22 = f32x8::splat_t(token, K_M22);
    let bias = f32x8::splat_t(token, K_B0);
    let zero = f32x8::zero_t(token);
    let ab = f32x8::splat_t(token, absorbance_bias);
    let half = f32x8::splat_t(token, 0.5);

    let n = pixels.len();
    let chunks = n / 8;

    for chunk in 0..chunks {
        let base = chunk * 8;

        // Load 8 pixels, linearize via LUT, transpose to SoA
        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance
        // Optimization"): ONE range check at the boundary, zero
        // interior. Same loads in the same order — bit-exact.
        let px: &[[u8; 3]; 8] = pixels[base..base + 8]
            .try_into()
            .expect("8 pixels per chunk");
        for i in 0..8 {
            let p = px[i];
            r_arr[i] = srgb_u8_to_linear(p[0]);
            g_arr[i] = srgb_u8_to_linear(p[1]);
            b_arr[i] = srgb_u8_to_linear(p[2]);
        }

        let r = f32x8::from_array_t(token, r_arr);
        let g = f32x8::from_array_t(token, g_arr);
        let b = f32x8::from_array_t(token, b_arr);

        // Opsin absorbance matrix multiply with FMA
        let mixed0 = m00.mul_add(r, m01.mul_add(g, m02.mul_add(b, bias)));
        let mixed1 = m10.mul_add(r, m11.mul_add(g, m12.mul_add(b, bias)));
        let mixed2 = m20.mul_add(r, m21.mul_add(g, m22.mul_add(b, bias)));

        let mixed0 = mixed0.max(zero);
        let mixed1 = mixed1.max(zero);
        let mixed2 = mixed2.max(zero);

        let c0 = mixed0.cbrt_midp() + ab;
        let c1 = mixed1.cbrt_midp() + ab;
        let c2 = mixed2.cbrt_midp();

        // XYB transform: X = 0.5*(c0-c1), Y = 0.5*(c0+c1), B = c2
        let x = half * (c0 - c1);
        let y = half * (c0 + c1);

        // Store directly to planar output
        let x_arr = x.to_array();
        let y_arr = y.to_array();
        let b_arr_out = c2.to_array();
        x_out[base..base + 8].copy_from_slice(&x_arr);
        y_out[base..base + 8].copy_from_slice(&y_arr);
        b_out[base..base + 8].copy_from_slice(&b_arr_out);
    }

    // Scalar remainder
    for i in (chunks * 8)..n {
        let p = pixels[i];
        let r = srgb_u8_to_linear(p[0]);
        let g = srgb_u8_to_linear(p[1]);
        let b = srgb_u8_to_linear(p[2]);

        let mut mixed0 = K_M00.mul_add(r, K_M01.mul_add(g, K_M02.mul_add(b, K_B0)));
        let mut mixed1 = K_M10.mul_add(r, K_M11.mul_add(g, K_M12.mul_add(b, K_B0)));
        let mut mixed2 = K_M20.mul_add(r, K_M21.mul_add(g, K_M22.mul_add(b, K_B0)));

        mixed0 = mixed0.max(0.0);
        mixed1 = mixed1.max(0.0);
        mixed2 = mixed2.max(0.0);

        let bias_neg = -cbrtf_fast(K_B0);
        mixed0 = cbrtf_fast(mixed0) + bias_neg;
        mixed1 = cbrtf_fast(mixed1) + bias_neg;
        mixed2 = cbrtf_fast(mixed2);

        x_out[i] = 0.5 * (mixed0 - mixed1);
        y_out[i] = 0.5 * (mixed0 + mixed1);
        b_out[i] = mixed2;
    }
}

/// sRGB → XYB (no positive shift), shared body of the generic tiers.
/// The remainder follows [`PaddedTail::PAD_TAIL`].
#[inline(always)]
fn srgb_xyb_generic<T: F32x8Convert + PaddedTail>(
    token: T,
    pixels: &[[u8; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    /// One 8-pixel chunk: LUT-linearize, transpose to SoA, opsin + cube root,
    /// X = 0.5*(c0-c1), Y = 0.5*(c0+c1), B = c2.
    #[inline(always)]
    fn chunk<T: F32x8Convert>(
        k: &OpsinChunk<T>,
        px: &[[u8; 3]; 8],
    ) -> ([f32; 8], [f32; 8], [f32; 8]) {
        let (r, g, b) = k.srgb_lanes(px);
        let (x, y, c2) = k.plain(k.cube_roots(r, g, b));
        (x.to_array(), y.to_array(), c2.to_array())
    }

    let k = OpsinChunk::new(token);
    let n = pixels.len();
    let chunks = n / 8;

    for chunk_idx in 0..chunks {
        let base = chunk_idx * 8;
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance Optimization"): ONE
        // range check at the boundary, zero interior.
        let px: &[[u8; 3]; 8] = pixels[base..base + 8]
            .try_into()
            .expect("8 pixels per chunk");
        let (xs, ys, bs) = chunk(&k, px);
        x_out[base..base + 8].copy_from_slice(&xs);
        y_out[base..base + 8].copy_from_slice(&ys);
        b_out[base..base + 8].copy_from_slice(&bs);
    }

    let done = chunks * 8;
    if T::PAD_TAIL {
        let rem = n - done;
        if rem != 0 {
            let mut padded = [[0u8; 3]; 8];
            padded[..rem].copy_from_slice(&pixels[done..]);
            let (xs, ys, bs) = chunk(&k, &padded);
            x_out[done..n].copy_from_slice(&xs[..rem]);
            y_out[done..n].copy_from_slice(&ys[..rem]);
            b_out[done..n].copy_from_slice(&bs[..rem]);
        }
    } else {
        let absorbance_bias = -cbrtf_fast(K_B0);
        // Scalar remainder
        for i in done..n {
            let p = pixels[i];
            let r = srgb_u8_to_linear(p[0]);
            let g = srgb_u8_to_linear(p[1]);
            let b = srgb_u8_to_linear(p[2]);

            let mut mixed0 = K_M00.mul_add(r, K_M01.mul_add(g, K_M02.mul_add(b, K_B0)));
            let mut mixed1 = K_M10.mul_add(r, K_M11.mul_add(g, K_M12.mul_add(b, K_B0)));
            let mut mixed2 = K_M20.mul_add(r, K_M21.mul_add(g, K_M22.mul_add(b, K_B0)));

            mixed0 = mixed0.max(0.0);
            mixed1 = mixed1.max(0.0);
            mixed2 = mixed2.max(0.0);

            mixed0 = cbrtf_fast(mixed0) + absorbance_bias;
            mixed1 = cbrtf_fast(mixed1) + absorbance_bias;
            mixed2 = cbrtf_fast(mixed2);

            x_out[i] = 0.5 * (mixed0 - mixed1);
            y_out[i] = 0.5 * (mixed0 + mixed1);
            b_out[i] = mixed2;
        }
    }
}

/// Generic sRGB → XYB (without positive shift) with scalar cube root.
#[magetypes(neon, wasm128, scalar)]
fn srgb_to_xyb_planar_inner(
    token: Token,
    pixels: &[[u8; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    srgb_xyb_generic(token, pixels, x_out, y_out, b_out);
}

// ---------------------------------------------------------------------------
// Linear RGB → positive XYB conversion
// ---------------------------------------------------------------------------

/// Convert interleaved linear f32 RGB to planar positive XYB.
///
/// Input: `&[[f32; 3]]` — linear-light RGB values (typically in [0.0, 1.0]).
/// Output: 3 planes (X, Y, B) each of length `pixels.len()`, already positive-shifted.
///
/// This is the same opsin matrix + cube root + positive shift as the sRGB u8 path,
/// but skips the sRGB LUT linearization step. Results are identical for the same
/// linear RGB values (within floating-point precision).
///
/// `revision` is the computation's formula revision: Rev4 runs the canonical
/// body ([`linear_xyb_canon`]), every earlier revision the tier-dispatched
/// kernels below.
pub fn linear_to_positive_xyb_planar_into(
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    revision: crate::feature_defs::FormulaRevision,
) {
    let mode = crate::featcanon::mode(revision);
    if mode.active() {
        linear_xyb_canon::<true>(pixels, x_out, y_out, b_out, mode, revision.cbrt_newton());
        return;
    }
    incant!(
        linear_to_positive_xyb_planar_inner(pixels, x_out, y_out, b_out),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}

/// AVX-512 path for linear f32 → positive XYB (16 pixels at a time).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn linear_to_positive_xyb_planar_inner_v4(
    token: archmage::X64V4Token,
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    let absorbance_bias = -cbrtf_fast(K_B0);

    let m00 = f32x16::splat_t(token, K_M00);
    let m01 = f32x16::splat_t(token, K_M01);
    let m02 = f32x16::splat_t(token, K_M02);
    let m10 = f32x16::splat_t(token, K_M10);
    let m11 = f32x16::splat_t(token, K_M11);
    let m12 = f32x16::splat_t(token, K_M12);
    let m20 = f32x16::splat_t(token, K_M20);
    let m21 = f32x16::splat_t(token, K_M21);
    let m22 = f32x16::splat_t(token, K_M22);
    let bias = f32x16::splat_t(token, K_B0);
    let zero = f32x16::zero_t(token);
    let ab = f32x16::splat_t(token, absorbance_bias);
    let half = f32x16::splat_t(token, 0.5);
    let fourteen = f32x16::splat_t(token, 14.0);
    let x_bias = f32x16::splat_t(token, 0.42);
    let y_bias = f32x16::splat_t(token, 0.01);
    let b_bias = f32x16::splat_t(token, 0.55);

    let one = f32x16::splat_t(token, 1.0);

    let n = pixels.len();
    let chunks = n / 16;

    for chunk in 0..chunks {
        let base = chunk * 16;

        let mut r_arr = [0.0f32; 16];
        let mut g_arr = [0.0f32; 16];
        let mut b_arr = [0.0f32; 16];
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance
        // Optimization"): ONE range check at the boundary, zero
        // interior. Same loads in the same order — bit-exact.
        let px: &[[f32; 3]; 16] = pixels[base..base + 16]
            .try_into()
            .expect("16 pixels per chunk");
        for i in 0..16 {
            let p = px[i];
            r_arr[i] = p[0];
            g_arr[i] = p[1];
            b_arr[i] = p[2];
        }

        // Clamp to display gamut [0, 1]
        let r = f32x16::from_array_t(token, r_arr).max(zero).min(one);
        let g = f32x16::from_array_t(token, g_arr).max(zero).min(one);
        let b = f32x16::from_array_t(token, b_arr).max(zero).min(one);

        let mixed0 = m00
            .mul_add(r, m01.mul_add(g, m02.mul_add(b, bias)))
            .max(zero);
        let mixed1 = m10
            .mul_add(r, m11.mul_add(g, m12.mul_add(b, bias)))
            .max(zero);
        let mixed2 = m20
            .mul_add(r, m21.mul_add(g, m22.mul_add(b, bias)))
            .max(zero);

        let t0 = mixed0.cbrt_midp();
        let t1 = mixed1.cbrt_midp();
        let t2 = mixed2.cbrt_midp();

        let c0 = t0 + ab;
        let c1 = t1 + ab;

        let x = half * (c0 - c1);
        let y = half * (c0 + c1);

        let x_pos = x.mul_add(fourteen, x_bias);
        let y_pos = y + y_bias;
        let b_pos = (t2 - y) + b_bias;

        x_out[base..base + 16].copy_from_slice(&x_pos.to_array());
        y_out[base..base + 16].copy_from_slice(&y_pos.to_array());
        b_out[base..base + 16].copy_from_slice(&b_pos.to_array());
    }

    // AVX2 remainder
    let v3 = token.v3();
    let ab8 = f32x8::splat_t(v3, absorbance_bias);
    let half8 = f32x8::splat_t(v3, 0.5);
    let zero8 = f32x8::zero_t(v3);
    let m00_8 = f32x8::splat_t(v3, K_M00);
    let m01_8 = f32x8::splat_t(v3, K_M01);
    let m02_8 = f32x8::splat_t(v3, K_M02);
    let m10_8 = f32x8::splat_t(v3, K_M10);
    let m11_8 = f32x8::splat_t(v3, K_M11);
    let m12_8 = f32x8::splat_t(v3, K_M12);
    let m20_8 = f32x8::splat_t(v3, K_M20);
    let m21_8 = f32x8::splat_t(v3, K_M21);
    let m22_8 = f32x8::splat_t(v3, K_M22);
    let bias8 = f32x8::splat_t(v3, K_B0);
    let fourteen8 = f32x8::splat_t(v3, 14.0);
    let x_bias8 = f32x8::splat_t(v3, 0.42);
    let y_bias8 = f32x8::splat_t(v3, 0.01);
    let b_bias8 = f32x8::splat_t(v3, 0.55);

    let one8 = f32x8::splat_t(v3, 1.0);

    let rem_start = chunks * 16;
    let rem_chunks = (n - rem_start) / 8;
    for chunk in 0..rem_chunks {
        let base = rem_start + chunk * 8;
        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance
        // Optimization"): ONE range check at the boundary, zero
        // interior. Same loads in the same order — bit-exact.
        let px: &[[f32; 3]; 8] = pixels[base..base + 8]
            .try_into()
            .expect("8 pixels per chunk");
        for i in 0..8 {
            let p = px[i];
            r_arr[i] = p[0];
            g_arr[i] = p[1];
            b_arr[i] = p[2];
        }
        // Clamp to display gamut [0, 1]
        let r = f32x8::from_array_t(v3, r_arr).max(zero8).min(one8);
        let g = f32x8::from_array_t(v3, g_arr).max(zero8).min(one8);
        let b = f32x8::from_array_t(v3, b_arr).max(zero8).min(one8);

        let mixed0 = m00_8
            .mul_add(r, m01_8.mul_add(g, m02_8.mul_add(b, bias8)))
            .max(zero8);
        let mixed1 = m10_8
            .mul_add(r, m11_8.mul_add(g, m12_8.mul_add(b, bias8)))
            .max(zero8);
        let mixed2 = m20_8
            .mul_add(r, m21_8.mul_add(g, m22_8.mul_add(b, bias8)))
            .max(zero8);

        let t0 = mixed0.cbrt_midp();
        let t1 = mixed1.cbrt_midp();
        let t2 = mixed2.cbrt_midp();

        let c0 = t0 + ab8;
        let c1 = t1 + ab8;
        let x = half8 * (c0 - c1);
        let y = half8 * (c0 + c1);
        let x_pos = x.mul_add(fourteen8, x_bias8);
        let y_pos = y + y_bias8;
        let b_pos = (t2 - y) + b_bias8;

        x_out[base..base + 8].copy_from_slice(&x_pos.to_array());
        y_out[base..base + 8].copy_from_slice(&y_pos.to_array());
        b_out[base..base + 8].copy_from_slice(&b_pos.to_array());
    }

    // Scalar remainder
    let absorbance_bias_neg = absorbance_bias;
    for i in (rem_start + rem_chunks * 8)..n {
        let p = pixels[i];
        let r = p[0].clamp(0.0, 1.0);
        let g = p[1].clamp(0.0, 1.0);
        let b = p[2].clamp(0.0, 1.0);

        let mixed0 = K_M00
            .mul_add(r, K_M01.mul_add(g, K_M02.mul_add(b, K_B0)))
            .max(0.0);
        let mixed1 = K_M10
            .mul_add(r, K_M11.mul_add(g, K_M12.mul_add(b, K_B0)))
            .max(0.0);
        let mixed2 = K_M20
            .mul_add(r, K_M21.mul_add(g, K_M22.mul_add(b, K_B0)))
            .max(0.0);

        let c0 = cbrtf_fast(mixed0) + absorbance_bias_neg;
        let c1 = cbrtf_fast(mixed1) + absorbance_bias_neg;
        let c2 = cbrtf_fast(mixed2);

        let x = 0.5 * (c0 - c1);
        let y = 0.5 * (c0 + c1);

        x_out[i] = x.mul_add(14.0, 0.42);
        y_out[i] = y + 0.01;
        b_out[i] = (c2 - y) + 0.55;
    }
}
#[cfg(target_arch = "x86_64")]
#[arcane]
fn linear_to_positive_xyb_planar_inner_v4x(
    token: archmage::X64V4xToken,
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    let absorbance_bias = -cbrtf_fast(K_B0);

    let m00 = f32x16::splat_t(token, K_M00);
    let m01 = f32x16::splat_t(token, K_M01);
    let m02 = f32x16::splat_t(token, K_M02);
    let m10 = f32x16::splat_t(token, K_M10);
    let m11 = f32x16::splat_t(token, K_M11);
    let m12 = f32x16::splat_t(token, K_M12);
    let m20 = f32x16::splat_t(token, K_M20);
    let m21 = f32x16::splat_t(token, K_M21);
    let m22 = f32x16::splat_t(token, K_M22);
    let bias = f32x16::splat_t(token, K_B0);
    let zero = f32x16::zero_t(token);
    let ab = f32x16::splat_t(token, absorbance_bias);
    let half = f32x16::splat_t(token, 0.5);
    let fourteen = f32x16::splat_t(token, 14.0);
    let x_bias = f32x16::splat_t(token, 0.42);
    let y_bias = f32x16::splat_t(token, 0.01);
    let b_bias = f32x16::splat_t(token, 0.55);

    let one = f32x16::splat_t(token, 1.0);

    let n = pixels.len();
    let chunks = n / 16;

    for chunk in 0..chunks {
        let base = chunk * 16;

        let mut r_arr = [0.0f32; 16];
        let mut g_arr = [0.0f32; 16];
        let mut b_arr = [0.0f32; 16];
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance
        // Optimization"): ONE range check at the boundary, zero
        // interior. Same loads in the same order — bit-exact.
        let px: &[[f32; 3]; 16] = pixels[base..base + 16]
            .try_into()
            .expect("16 pixels per chunk");
        for i in 0..16 {
            let p = px[i];
            r_arr[i] = p[0];
            g_arr[i] = p[1];
            b_arr[i] = p[2];
        }

        // Clamp to display gamut [0, 1]
        let r = f32x16::from_array_t(token, r_arr).max(zero).min(one);
        let g = f32x16::from_array_t(token, g_arr).max(zero).min(one);
        let b = f32x16::from_array_t(token, b_arr).max(zero).min(one);

        let mixed0 = m00
            .mul_add(r, m01.mul_add(g, m02.mul_add(b, bias)))
            .max(zero);
        let mixed1 = m10
            .mul_add(r, m11.mul_add(g, m12.mul_add(b, bias)))
            .max(zero);
        let mixed2 = m20
            .mul_add(r, m21.mul_add(g, m22.mul_add(b, bias)))
            .max(zero);

        let t0 = mixed0.cbrt_midp();
        let t1 = mixed1.cbrt_midp();
        let t2 = mixed2.cbrt_midp();

        let c0 = t0 + ab;
        let c1 = t1 + ab;

        let x = half * (c0 - c1);
        let y = half * (c0 + c1);

        let x_pos = x.mul_add(fourteen, x_bias);
        let y_pos = y + y_bias;
        let b_pos = (t2 - y) + b_bias;

        x_out[base..base + 16].copy_from_slice(&x_pos.to_array());
        y_out[base..base + 16].copy_from_slice(&y_pos.to_array());
        b_out[base..base + 16].copy_from_slice(&b_pos.to_array());
    }

    // AVX2 remainder
    let v3 = token.v3();
    let ab8 = f32x8::splat_t(v3, absorbance_bias);
    let half8 = f32x8::splat_t(v3, 0.5);
    let zero8 = f32x8::zero_t(v3);
    let m00_8 = f32x8::splat_t(v3, K_M00);
    let m01_8 = f32x8::splat_t(v3, K_M01);
    let m02_8 = f32x8::splat_t(v3, K_M02);
    let m10_8 = f32x8::splat_t(v3, K_M10);
    let m11_8 = f32x8::splat_t(v3, K_M11);
    let m12_8 = f32x8::splat_t(v3, K_M12);
    let m20_8 = f32x8::splat_t(v3, K_M20);
    let m21_8 = f32x8::splat_t(v3, K_M21);
    let m22_8 = f32x8::splat_t(v3, K_M22);
    let bias8 = f32x8::splat_t(v3, K_B0);
    let fourteen8 = f32x8::splat_t(v3, 14.0);
    let x_bias8 = f32x8::splat_t(v3, 0.42);
    let y_bias8 = f32x8::splat_t(v3, 0.01);
    let b_bias8 = f32x8::splat_t(v3, 0.55);

    let one8 = f32x8::splat_t(v3, 1.0);

    let rem_start = chunks * 16;
    let rem_chunks = (n - rem_start) / 8;
    for chunk in 0..rem_chunks {
        let base = rem_start + chunk * 8;
        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance
        // Optimization"): ONE range check at the boundary, zero
        // interior. Same loads in the same order — bit-exact.
        let px: &[[f32; 3]; 8] = pixels[base..base + 8]
            .try_into()
            .expect("8 pixels per chunk");
        for i in 0..8 {
            let p = px[i];
            r_arr[i] = p[0];
            g_arr[i] = p[1];
            b_arr[i] = p[2];
        }
        // Clamp to display gamut [0, 1]
        let r = f32x8::from_array_t(v3, r_arr).max(zero8).min(one8);
        let g = f32x8::from_array_t(v3, g_arr).max(zero8).min(one8);
        let b = f32x8::from_array_t(v3, b_arr).max(zero8).min(one8);

        let mixed0 = m00_8
            .mul_add(r, m01_8.mul_add(g, m02_8.mul_add(b, bias8)))
            .max(zero8);
        let mixed1 = m10_8
            .mul_add(r, m11_8.mul_add(g, m12_8.mul_add(b, bias8)))
            .max(zero8);
        let mixed2 = m20_8
            .mul_add(r, m21_8.mul_add(g, m22_8.mul_add(b, bias8)))
            .max(zero8);

        let t0 = mixed0.cbrt_midp();
        let t1 = mixed1.cbrt_midp();
        let t2 = mixed2.cbrt_midp();

        let c0 = t0 + ab8;
        let c1 = t1 + ab8;
        let x = half8 * (c0 - c1);
        let y = half8 * (c0 + c1);
        let x_pos = x.mul_add(fourteen8, x_bias8);
        let y_pos = y + y_bias8;
        let b_pos = (t2 - y) + b_bias8;

        x_out[base..base + 8].copy_from_slice(&x_pos.to_array());
        y_out[base..base + 8].copy_from_slice(&y_pos.to_array());
        b_out[base..base + 8].copy_from_slice(&b_pos.to_array());
    }

    // Scalar remainder
    let absorbance_bias_neg = absorbance_bias;
    for i in (rem_start + rem_chunks * 8)..n {
        let p = pixels[i];
        let r = p[0].clamp(0.0, 1.0);
        let g = p[1].clamp(0.0, 1.0);
        let b = p[2].clamp(0.0, 1.0);

        let mixed0 = K_M00
            .mul_add(r, K_M01.mul_add(g, K_M02.mul_add(b, K_B0)))
            .max(0.0);
        let mixed1 = K_M10
            .mul_add(r, K_M11.mul_add(g, K_M12.mul_add(b, K_B0)))
            .max(0.0);
        let mixed2 = K_M20
            .mul_add(r, K_M21.mul_add(g, K_M22.mul_add(b, K_B0)))
            .max(0.0);

        let c0 = cbrtf_fast(mixed0) + absorbance_bias_neg;
        let c1 = cbrtf_fast(mixed1) + absorbance_bias_neg;
        let c2 = cbrtf_fast(mixed2);

        let x = 0.5 * (c0 - c1);
        let y = 0.5 * (c0 + c1);

        x_out[i] = x.mul_add(14.0, 0.42);
        y_out[i] = y + 0.01;
        b_out[i] = (c2 - y) + 0.55;
    }
}

/// AVX2 path for linear f32 → positive XYB (8 pixels at a time).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn linear_to_positive_xyb_planar_inner_v3(
    token: archmage::X64V3Token,
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    let absorbance_bias = -cbrtf_fast(K_B0);

    let m00 = f32x8::splat_t(token, K_M00);
    let m01 = f32x8::splat_t(token, K_M01);
    let m02 = f32x8::splat_t(token, K_M02);
    let m10 = f32x8::splat_t(token, K_M10);
    let m11 = f32x8::splat_t(token, K_M11);
    let m12 = f32x8::splat_t(token, K_M12);
    let m20 = f32x8::splat_t(token, K_M20);
    let m21 = f32x8::splat_t(token, K_M21);
    let m22 = f32x8::splat_t(token, K_M22);
    let bias = f32x8::splat_t(token, K_B0);
    let zero = f32x8::zero_t(token);
    let one = f32x8::splat_t(token, 1.0);
    let ab = f32x8::splat_t(token, absorbance_bias);
    let half = f32x8::splat_t(token, 0.5);
    let fourteen = f32x8::splat_t(token, 14.0);
    let x_bias = f32x8::splat_t(token, 0.42);
    let y_bias = f32x8::splat_t(token, 0.01);
    let b_bias = f32x8::splat_t(token, 0.55);

    let n = pixels.len();
    let chunks = n / 8;

    for chunk in 0..chunks {
        let base = chunk * 8;

        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance
        // Optimization"): ONE range check at the boundary, zero
        // interior. Same loads in the same order — bit-exact.
        let px: &[[f32; 3]; 8] = pixels[base..base + 8]
            .try_into()
            .expect("8 pixels per chunk");
        for i in 0..8 {
            let p = px[i];
            r_arr[i] = p[0];
            g_arr[i] = p[1];
            b_arr[i] = p[2];
        }

        // Clamp to display gamut: out-of-range values from lossy reconstruction
        // aren't visible on a real display, so measuring them would overcount error.
        let r = f32x8::from_array_t(token, r_arr).max(zero).min(one);
        let g = f32x8::from_array_t(token, g_arr).max(zero).min(one);
        let b = f32x8::from_array_t(token, b_arr).max(zero).min(one);

        let mixed0 = m00
            .mul_add(r, m01.mul_add(g, m02.mul_add(b, bias)))
            .max(zero);
        let mixed1 = m10
            .mul_add(r, m11.mul_add(g, m12.mul_add(b, bias)))
            .max(zero);
        let mixed2 = m20
            .mul_add(r, m21.mul_add(g, m22.mul_add(b, bias)))
            .max(zero);

        let t0 = mixed0.cbrt_midp();
        let t1 = mixed1.cbrt_midp();
        let t2 = mixed2.cbrt_midp();

        let c0 = t0 + ab;
        let c1 = t1 + ab;

        let x = half * (c0 - c1);
        let y = half * (c0 + c1);

        let x_pos = x.mul_add(fourteen, x_bias);
        let y_pos = y + y_bias;
        let b_pos = (t2 - y) + b_bias;

        x_out[base..base + 8].copy_from_slice(&x_pos.to_array());
        y_out[base..base + 8].copy_from_slice(&y_pos.to_array());
        b_out[base..base + 8].copy_from_slice(&b_pos.to_array());
    }

    // Scalar remainder
    let absorbance_bias_neg = absorbance_bias;
    for i in (chunks * 8)..n {
        let p = pixels[i];
        let r = p[0].clamp(0.0, 1.0);
        let g = p[1].clamp(0.0, 1.0);
        let b = p[2].clamp(0.0, 1.0);

        let mixed0 = K_M00
            .mul_add(r, K_M01.mul_add(g, K_M02.mul_add(b, K_B0)))
            .max(0.0);
        let mixed1 = K_M10
            .mul_add(r, K_M11.mul_add(g, K_M12.mul_add(b, K_B0)))
            .max(0.0);
        let mixed2 = K_M20
            .mul_add(r, K_M21.mul_add(g, K_M22.mul_add(b, K_B0)))
            .max(0.0);

        let c0 = cbrtf_fast(mixed0) + absorbance_bias_neg;
        let c1 = cbrtf_fast(mixed1) + absorbance_bias_neg;
        let c2 = cbrtf_fast(mixed2);

        let x = 0.5 * (c0 - c1);
        let y = 0.5 * (c0 + c1);

        x_out[i] = x.mul_add(14.0, 0.42);
        y_out[i] = y + 0.01;
        b_out[i] = (c2 - y) + 0.55;
    }
}

/// Linear f32 → positive XYB, shared body of the generic tiers. One 8-pixel chunk goes
/// through [`OpsinChunk`]; see [`PaddedTail`] for the remainder. `CLAMP` is the display-gamut
/// clamp of the input to `[0, 1]`: on for [`linear_to_positive_xyb_planar_into`], off for the
/// scalar tier of the unclamped `GamutMapping::Preserve` converter. One arithmetic definition
/// either way.
#[inline(always)]
fn linear_positive_xyb_generic<T: F32x8Convert + PaddedTail, const CLAMP: bool>(
    token: T,
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    /// One 8-pixel chunk. Clamp to display gamut (`CLAMP`): out-of-range values from lossy
    /// reconstruction aren't visible on a real display, so measuring them would
    /// overcount error.
    #[inline(always)]
    fn chunk<T: F32x8Convert, const CLAMP: bool>(
        k: &OpsinChunk<T>,
        px: &[[f32; 3]; 8],
    ) -> ([f32; 8], [f32; 8], [f32; 8]) {
        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];
        for i in 0..8 {
            let p = px[i];
            r_arr[i] = p[0];
            g_arr[i] = p[1];
            b_arr[i] = p[2];
        }
        let load = |arr: [f32; 8]| {
            let v = GenericF32x8::from_array_t(k.token, arr);
            if CLAMP { v.max(k.zero).min(k.one) } else { v }
        };
        let (r, g, b) = (load(r_arr), load(g_arr), load(b_arr));
        let (x_pos, y_pos, b_pos) = k.positive(k.cube_roots(r, g, b));
        (x_pos.to_array(), y_pos.to_array(), b_pos.to_array())
    }

    let k = OpsinChunk::new(token);
    let n = pixels.len();
    let chunks = n / 8;

    for chunk_idx in 0..chunks {
        let base = chunk_idx * 8;
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance Optimization"): ONE
        // range check at the boundary, zero interior.
        let px: &[[f32; 3]; 8] = pixels[base..base + 8]
            .try_into()
            .expect("8 pixels per chunk");
        let (xs, ys, bs) = chunk::<T, CLAMP>(&k, px);
        x_out[base..base + 8].copy_from_slice(&xs);
        y_out[base..base + 8].copy_from_slice(&ys);
        b_out[base..base + 8].copy_from_slice(&bs);
    }

    let done = chunks * 8;
    if T::PAD_TAIL {
        let rem = n - done;
        if rem != 0 {
            let mut padded = [[0.0f32; 3]; 8];
            padded[..rem].copy_from_slice(&pixels[done..]);
            let (xs, ys, bs) = chunk::<T, CLAMP>(&k, &padded);
            x_out[done..n].copy_from_slice(&xs[..rem]);
            y_out[done..n].copy_from_slice(&ys[..rem]);
            b_out[done..n].copy_from_slice(&bs[..rem]);
        }
    } else {
        let absorbance_bias = -cbrtf_fast(K_B0);
        // Scalar remainder
        for i in done..n {
            let p = pixels[i];
            let [r, g, b] = if CLAMP {
                [
                    p[0].clamp(0.0, 1.0),
                    p[1].clamp(0.0, 1.0),
                    p[2].clamp(0.0, 1.0),
                ]
            } else {
                p
            };

            let mixed0 = K_M00
                .mul_add(r, K_M01.mul_add(g, K_M02.mul_add(b, K_B0)))
                .max(0.0);
            let mixed1 = K_M10
                .mul_add(r, K_M11.mul_add(g, K_M12.mul_add(b, K_B0)))
                .max(0.0);
            let mixed2 = K_M20
                .mul_add(r, K_M21.mul_add(g, K_M22.mul_add(b, K_B0)))
                .max(0.0);

            let c0 = cbrtf_fast(mixed0) + absorbance_bias;
            let c1 = cbrtf_fast(mixed1) + absorbance_bias;
            let c2 = cbrtf_fast(mixed2);

            let x = 0.5 * (c0 - c1);
            let y = 0.5 * (c0 + c1);

            x_out[i] = x.mul_add(14.0, 0.42);
            y_out[i] = y + 0.01;
            b_out[i] = (c2 - y) + 0.55;
        }
    }
}

/// Generic linear f32 to positive XYB with gamut clamping and Halley iterations.
#[magetypes(neon, wasm128, scalar)]
fn linear_to_positive_xyb_planar_inner(
    token: Token,
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    linear_positive_xyb_generic::<_, true>(token, pixels, x_out, y_out, b_out);
}

/// Unclamped sibling of [`linear_to_positive_xyb_planar_into`],
/// used ONLY by the [`GamutMapping::Preserve`](crate::source::GamutMapping)
/// conversion path (issue #17).
///
/// Input components may be negative or > 1 (out-of-sRGB-gamut linear light
/// after a wide-gamut primaries matrix) — the clamped kernels would clip
/// them back to the gamut boundary, re-masking exactly the difference
/// `Preserve` exists to expose. The opsin mix's `max(0)` keeps the
/// cube-root domain valid for arbitrary finite input; inputs must be
/// finite (u8-linearized wide-gamut rows always are; `LinearF32Rgba`
/// callers own their float hygiene in this opt-in mode).
///
/// Follows the same tail policy as its clamped sibling ([`PaddedTail`]):
/// - **scalar tier**: the [`OpsinChunk`] arithmetic without the input clamp — full
///   8-pixel chunks plus a zero-padded remainder chunk
///   ([`linear_positive_xyb_generic`] with `CLAMP = false`), so for in-`[0,1]` inputs the
///   output is bit-identical to the clamped entry at every position;
/// - **every other tier** (v3/v4/v4x, neon, wasm128): the per-pixel form below for every
///   pixel — same `mul_add` chains, same [`cbrtf_fast`], same biases — which is
///   bit-identical to the clamped kernels' per-pixel remainder for in-`[0,1]` inputs.
///
/// Both locked by `unclamped_matches_clamped_scalar_for_in_gamut` and
/// `scalar_tier_unclamped_matches_clamped_at_every_position` below.
///
/// `revision` as for [`linear_to_positive_xyb_planar_into`].
pub(crate) fn linear_to_positive_xyb_planar_into_unclamped(
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    revision: crate::feature_defs::FormulaRevision,
) {
    let mode = crate::featcanon::mode(revision);
    if mode.active() {
        linear_xyb_canon::<false>(pixels, x_out, y_out, b_out, mode, revision.cbrt_newton());
        return;
    }
    incant!(
        linear_positive_unclamped_inner(pixels, x_out, y_out, b_out),
        [v3, neon, wasm128, scalar]
    );
}

#[magetypes(v3, neon, wasm128, scalar)]
fn linear_positive_unclamped_inner(
    token: Token,
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    linear_positive_unclamped_generic(token, pixels, x_out, y_out, b_out);
}

/// Tail policy of the unclamped converter: chunk arithmetic where the tier pads its tail,
/// the per-pixel form everywhere else.
#[inline(always)]
fn linear_positive_unclamped_generic<T: F32x8Convert + PaddedTail>(
    token: T,
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    if T::PAD_TAIL {
        linear_positive_xyb_generic::<_, false>(token, pixels, x_out, y_out, b_out);
    } else {
        linear_positive_unclamped_per_pixel(pixels, x_out, y_out, b_out);
    }
}

/// The per-pixel unclamped form (see [`linear_to_positive_xyb_planar_into_unclamped`]).
#[inline(always)]
fn linear_positive_unclamped_per_pixel(
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    let absorbance_bias = -cbrtf_fast(K_B0);
    for (i, p) in pixels.iter().enumerate() {
        let [r, g, b] = *p;

        let mixed0 = K_M00
            .mul_add(r, K_M01.mul_add(g, K_M02.mul_add(b, K_B0)))
            .max(0.0);
        let mixed1 = K_M10
            .mul_add(r, K_M11.mul_add(g, K_M12.mul_add(b, K_B0)))
            .max(0.0);
        let mixed2 = K_M20
            .mul_add(r, K_M21.mul_add(g, K_M22.mul_add(b, K_B0)))
            .max(0.0);

        let c0 = cbrtf_fast(mixed0) + absorbance_bias;
        let c1 = cbrtf_fast(mixed1) + absorbance_bias;
        let c2 = cbrtf_fast(mixed2);

        let x = 0.5 * (c0 - c1);
        let y = 0.5 * (c0 + c1);

        x_out[i] = x.mul_add(14.0, 0.42);
        y_out[i] = y + 0.01;
        b_out[i] = (c2 - y) + 0.55;
    }
}

/// HDR PU-XYB conversion (scalar): **absolute-luminance** linear RGB (cd/m²,
/// NOT clamped to `[0,1]`) → PU-encoded XYB-like planes.
///
/// The HDR-path analog of [`linear_to_positive_xyb_planar_into`]: identical
/// opsin mix + opponent structure, but each opsin-mixed channel goes through
/// **PU21** (normalized so 100 cd/m² → ~1.0, matching the cube-root white
/// point) instead of the cube root. The cube-root-domain absorbance centering
/// (`-cbrt(K_B0)`) is dropped — in PU space `K_B0` clamps below `PU21_L_MIN`,
/// so its contribution is ~0. See `docs/HDR_PLAN.md` §2b for the design
/// rationale and the validation gate. Scalar only (HDR is a minority path;
/// SIMD is a later optimization).
/// PU21(100 cd/m²) ≈ 256.3 — normalizes a 100-nit reference white to ~1.0,
/// the range the cube-root XYB white point sits in, so the downstream
/// opponent biases + feature kernels stay in calibration.
const PU_WHITE: f32 = 256.3;
/// Opponent X amplification in PU space. The cube-root path uses 14× to make
/// the tiny red-green cube-root difference visible; PU-space opsin
/// differences are already large, and HDR validation favors
/// luminance-dominant weighting (see docs/HDR_PLAN.md §2b).
const PU_X_SCALE: f32 = 4.0;

/// Scalar per-pixel PU-XYB (also the SIMD kernels' tail path and the
/// parity-test reference).
#[inline]
fn pu_xyb_pixel(p: [f32; 3]) -> (f32, f32, f32) {
    use crate::pu21::pu21_encode;
    let mixed0 = K_M00
        .mul_add(p[0], K_M01.mul_add(p[1], K_M02.mul_add(p[2], K_B0)))
        .max(0.0);
    let mixed1 = K_M10
        .mul_add(p[0], K_M11.mul_add(p[1], K_M12.mul_add(p[2], K_B0)))
        .max(0.0);
    let mixed2 = K_M20
        .mul_add(p[0], K_M21.mul_add(p[1], K_M22.mul_add(p[2], K_B0)))
        .max(0.0);
    let c0 = pu21_encode(mixed0) / PU_WHITE;
    let c1 = pu21_encode(mixed1) / PU_WHITE;
    let c2 = pu21_encode(mixed2) / PU_WHITE;
    let x = 0.5 * (c0 - c1);
    let y = 0.5 * (c0 + c1);
    (x.mul_add(PU_X_SCALE, 0.42), y + 0.01, (c2 - y) + 0.55)
}

/// Canonical per-pixel PU-XYB (REV4SERVE): [`pu_xyb_pixel`]'s structure with
/// the PU21 encode replaced by [`crate::pu21::pu21_encode_canon`] — the
/// production SIMD `pu` closure's midp formula chain replicated scalar-side
/// with always-fused `mul_add`. One body, every pixel, every tier.
#[inline(always)]
fn pu_xyb_pixel_canon(p: [f32; 3]) -> (f32, f32, f32) {
    let mixed0 = K_M00
        .mul_add(p[0], K_M01.mul_add(p[1], K_M02.mul_add(p[2], K_B0)))
        .max(0.0);
    let mixed1 = K_M10
        .mul_add(p[0], K_M11.mul_add(p[1], K_M12.mul_add(p[2], K_B0)))
        .max(0.0);
    let mixed2 = K_M20
        .mul_add(p[0], K_M21.mul_add(p[1], K_M22.mul_add(p[2], K_B0)))
        .max(0.0);
    let c0 = crate::pu21::pu21_encode_canon(mixed0) / PU_WHITE;
    let c1 = crate::pu21::pu21_encode_canon(mixed1) / PU_WHITE;
    let c2 = crate::pu21::pu21_encode_canon(mixed2) / PU_WHITE;
    let x = 0.5 * (c0 - c1);
    let y = 0.5 * (c0 + c1);
    (x.mul_add(PU_X_SCALE, 0.42), y + 0.01, (c2 - y) + 0.55)
}

/// Exact-mode sibling of [`pu_xyb_pixel_canon`]: the banding_glare formula
/// in f64 (real `f64::powf`), rounding once at the f32 store — the
/// featcanon "exact" oracle's convention, as [`opsin_px_exact`].
#[cfg(feature = "oracle")]
#[inline(always)]
#[allow(clippy::manual_clamp)] // NaN semantics must match the SIMD .max().min() lanes
fn pu_xyb_px_exact(p: [f32; 3]) -> (f64, f64, f64) {
    let pu = |v: f64| -> f64 {
        let y = v
            .max(crate::pu21::PU21_L_MIN as f64)
            .min(crate::pu21::PU21_L_MAX as f64);
        let yp = y.powf(crate::pu21::P[3] as f64);
        let inner = ((crate::pu21::P[0] as f64) + (crate::pu21::P[1] as f64) * yp)
            / (1.0 + (crate::pu21::P[2] as f64) * yp);
        ((crate::pu21::P[6] as f64)
            * (inner.powf(crate::pu21::P[4] as f64) - (crate::pu21::P[5] as f64)))
            .max(0.0)
    };
    let m0 = (K_M00 as f64)
        .mul_add(
            p[0] as f64,
            (K_M01 as f64).mul_add(
                p[1] as f64,
                (K_M02 as f64).mul_add(p[2] as f64, K_B0 as f64),
            ),
        )
        .max(0.0);
    let m1 = (K_M10 as f64)
        .mul_add(
            p[0] as f64,
            (K_M11 as f64).mul_add(
                p[1] as f64,
                (K_M12 as f64).mul_add(p[2] as f64, K_B0 as f64),
            ),
        )
        .max(0.0);
    let m2 = (K_M20 as f64)
        .mul_add(
            p[0] as f64,
            (K_M21 as f64).mul_add(
                p[1] as f64,
                (K_M22 as f64).mul_add(p[2] as f64, K_B0 as f64),
            ),
        )
        .max(0.0);
    let c0 = pu(m0) / (PU_WHITE as f64);
    let c1 = pu(m1) / (PU_WHITE as f64);
    let c2 = pu(m2) / (PU_WHITE as f64);
    let x = 0.5 * (c0 - c1);
    let y = 0.5 * (c0 + c1);
    (
        x.mul_add(PU_X_SCALE as f64, 0.42),
        y + 0.01,
        (c2 - y) + 0.55,
    )
}

/// Canonical linear-absolute-nits → PU-XYB driver; `mode` as for
/// [`srgb_xyb_canon`]. REV4VEC: full 16-pixel chunks run
/// [`pu_xyb_canon_vec`] on the fused-FMA tiers; the scalar canonical body
/// runs the remainder and every pixel on the non-fused tiers.
pub(crate) fn pu_xyb_canon(
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    mode: crate::featcanon::Mode,
) {
    #[cfg(feature = "oracle")]
    if mode.exact() {
        for i in 0..pixels.len() {
            let (x, y, bb) = pu_xyb_px_exact(pixels[i]);
            x_out[i] = x as f32;
            y_out[i] = y as f32;
            b_out[i] = bb as f32;
        }
        return;
    }
    #[cfg(not(feature = "oracle"))]
    let _ = mode;
    incant!(
        pu_xyb_canon_vec(pixels, x_out, y_out, b_out),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}

/// The per-element scalar canonical PU-XYB body — the fused tiers'
/// remainder and the non-fused tiers' whole input, as
/// [`srgb_xyb_canon_body`].
fn pu_xyb_canon_body(pixels: &[[f32; 3]], x_out: &mut [f32], y_out: &mut [f32], b_out: &mut [f32]) {
    for (i, p) in pixels.iter().enumerate() {
        let (x, y, bb) = pu_xyb_pixel_canon(*p);
        x_out[i] = x;
        y_out[i] = y;
        b_out[i] = bb;
    }
}

/// [`pu_xyb_pixel_canon`] lane-parallel over [`CanonChunk::pu_xyb`] — the
/// canonical opsin mix, canonical PU21, `/ PU_WHITE`, positive shift — for
/// full 16-pixel chunks; the scalar body covers the remainder.
/// `-scalar` as for [`srgb_xyb_canon_vec`].
#[magetypes(define(f32x16), v4x, v4, v3, -scalar)]
fn pu_xyb_canon_vec(
    token: Token,
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    let k = CanonChunk::new(token);
    let n = pixels.len();
    let chunks = n / 16;
    for c in 0..chunks {
        let base = c * 16;
        let px: &[[f32; 3]; 16] = pixels[base..base + 16]
            .try_into()
            .expect("16 pixels per chunk");
        let (r, g, b) = k.linear_lanes::<false>(px);
        let (x, y, bb) = k.pu_xyb(r, g, b);
        x_out[base..base + 16].copy_from_slice(&x.to_array());
        y_out[base..base + 16].copy_from_slice(&y.to_array());
        b_out[base..base + 16].copy_from_slice(&bb.to_array());
    }
    for i in (chunks * 16)..n {
        let (x, y, bb) = pu_xyb_pixel_canon(pixels[i]);
        x_out[i] = x;
        y_out[i] = y;
        b_out[i] = bb;
    }
}

fn pu_xyb_canon_vec_scalar(
    _token: archmage::ScalarToken,
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    pu_xyb_canon_body(pixels, x_out, y_out, b_out);
}

#[cfg(target_arch = "wasm32")]
fn pu_xyb_canon_vec_wasm128(
    _token: archmage::Wasm128Token,
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    pu_xyb_canon_body(pixels, x_out, y_out, b_out);
}

/// NEON runs the scalar canonical body (2026-10-03): its vector `max`/`min` propagate NaN, while the canon clamps with
/// `f32::max`/`min`, which drop it, so a lane-parallel NEON chunk differed from the scalar canon on NaN inputs (CI aarch64).
#[cfg(target_arch = "aarch64")]
fn pu_xyb_canon_vec_neon(
    _token: archmage::NeonToken,
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    pu_xyb_canon_body(pixels, x_out, y_out, b_out);
}

/// [`linear_to_pu_xyb_planar_into`] at an explicit formula revision:
/// the tier-dispatched production kernel through Rev3, the canonical body
/// ([`pu_xyb_canon`]) at Rev4 — the same convention
/// [`srgb_to_positive_xyb_planar_into_at_revision`] uses.
pub(crate) fn linear_to_pu_xyb_planar_into_at_revision(
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
    revision: crate::feature_defs::FormulaRevision,
) {
    let mode = crate::featcanon::mode(revision);
    if mode.active() {
        pu_xyb_canon(pixels, x_out, y_out, b_out, mode);
        return;
    }
    linear_to_pu_xyb_planar_into(pixels, x_out, y_out, b_out);
}

/// Absolute-luminance linear RGB → positive PU-XYB planes, SIMD-dispatched
/// (8 px/iter via the generic magetypes tiers; `x^p = exp2_midp_precise(p·log2_midp_precise(x))`
/// — the midp_precise transcendentals hold the scalar↔SIMD divergence to
/// ≤ 2e-3 per channel; `simd_matches_scalar_within_band` pins it).
pub(crate) fn linear_to_pu_xyb_planar_into(
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    incant!(
        pu_xyb_rows_inner(pixels, x_out, y_out, b_out),
        [v3, neon, wasm128, scalar]
    );
}

#[magetypes(v3, neon, wasm128, scalar)]
fn pu_xyb_rows_inner(
    token: Token,
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    // PU21 banding_glare parameters — single source: `pu21::P`.
    const P0: f32 = crate::pu21::P[0];
    const P1: f32 = crate::pu21::P[1];
    const P2: f32 = crate::pu21::P[2];
    const P3: f32 = crate::pu21::P[3];
    const P4: f32 = crate::pu21::P[4];
    const P5: f32 = crate::pu21::P[5];
    const P6: f32 = crate::pu21::P[6];
    let l_min = f32x8::splat_t(token, crate::pu21::PU21_L_MIN);
    let l_max = f32x8::splat_t(token, crate::pu21::PU21_L_MAX);
    let zero = f32x8::splat_t(token, 0.0);
    let one = f32x8::splat_t(token, 1.0);
    let p0 = f32x8::splat_t(token, P0);
    let p1 = f32x8::splat_t(token, P1);
    let p2 = f32x8::splat_t(token, P2);
    let p3 = f32x8::splat_t(token, P3);
    let p4 = f32x8::splat_t(token, P4);
    let p5 = f32x8::splat_t(token, P5);
    let p6 = f32x8::splat_t(token, P6);
    let inv_white = f32x8::splat_t(token, 1.0 / PU_WHITE);
    let pu = |v: f32x8| -> f32x8 {
        let y = v.max(l_min).min(l_max);
        let yp = (p3 * y.log2_midp_precise()).exp2_midp_precise();
        let inner = (p0 + p1 * yp) / (one + p2 * yp);
        (p6 * ((p4 * inner.log2_midp_precise()).exp2_midp_precise() - p5)).max(zero)
    };

    let n = pixels.len();
    let chunks = n / 8;
    let mut r = [0.0f32; 8];
    let mut g = [0.0f32; 8];
    let mut b = [0.0f32; 8];
    for c in 0..chunks {
        let base = c * 8;
        for (j, px) in pixels[base..base + 8].iter().enumerate() {
            r[j] = px[0];
            g[j] = px[1];
            b[j] = px[2];
        }
        let vr = f32x8::from_array_t(token, r);
        let vg = f32x8::from_array_t(token, g);
        let vb = f32x8::from_array_t(token, b);
        let kb0 = f32x8::splat_t(token, K_B0);
        let m0 = (f32x8::splat_t(token, K_M00) * vr
            + f32x8::splat_t(token, K_M01) * vg
            + f32x8::splat_t(token, K_M02) * vb
            + kb0)
            .max(zero);
        let m1 = (f32x8::splat_t(token, K_M10) * vr
            + f32x8::splat_t(token, K_M11) * vg
            + f32x8::splat_t(token, K_M12) * vb
            + kb0)
            .max(zero);
        let m2 = (f32x8::splat_t(token, K_M20) * vr
            + f32x8::splat_t(token, K_M21) * vg
            + f32x8::splat_t(token, K_M22) * vb
            + kb0)
            .max(zero);
        let c0 = pu(m0) * inv_white;
        let c1 = pu(m1) * inv_white;
        let c2 = pu(m2) * inv_white;
        let half = f32x8::splat_t(token, 0.5);
        let x = half * (c0 - c1);
        let y = half * (c0 + c1);
        (x * f32x8::splat_t(token, PU_X_SCALE) + f32x8::splat_t(token, 0.42))
            .store((&mut x_out[base..base + 8]).try_into().unwrap());
        (y + f32x8::splat_t(token, 0.01)).store((&mut y_out[base..base + 8]).try_into().unwrap());
        ((c2 - y) + f32x8::splat_t(token, 0.55))
            .store((&mut b_out[base..base + 8]).try_into().unwrap());
    }
    for i in (chunks * 8)..n {
        let (x, y, bb) = pu_xyb_pixel(pixels[i]);
        x_out[i] = x;
        y_out[i] = y;
        b_out[i] = bb;
    }
}

// ---------------------------------------------------------------------------
// RGBA/BGRA compositing helpers — all produce linear f32 RGB output
// ---------------------------------------------------------------------------

/// Precomputed deterministic noise table in linear light. 4096 values in
/// \[0.2, 0.8\], indexed by `(x ^ hash(y)) % 4096`.
///
/// Generated from integer-only hashing (wrapping multiply + xorshift) for
/// bit-identical results across all platforms. The final normalization
/// (u16 → f32 multiply + add) is exact in f32.
///
/// 4096 entries (16 KiB) fits in L1 cache and produces visually uniform
/// noise with no banding or diagonal artifacts.
const ALPHA_BG_TABLE: [f32; 4096] = {
    let mut table = [0.0f32; 4096];
    let mut i = 0u32;
    while i < 4096 {
        let mut h = i.wrapping_mul(2654435761);
        h ^= h >> 16;
        h = h.wrapping_mul(0x45d9f3b);
        h ^= h >> 16;
        table[i as usize] = 0.2 + (h & 0xFFFF) as f32 * (0.6 / 65535.0);
        i += 1;
    }
    table
};

/// Deterministic noise background value in linear light for the given pixel.
///
/// XORs the x coordinate with a hashed y to break row correlation, then
/// indexes into [`ALPHA_BG_TABLE`]. Cost: one wrapping multiply, one XOR,
/// one table lookup (always in L1).
#[inline(always)]
fn alpha_background_linear(x: usize, y: usize) -> f32 {
    let yh = (y as u32).wrapping_mul(2654435761);
    ALPHA_BG_TABLE[((x as u32) ^ yh) as usize & 0xFFF]
}

/// Composite sRGB u8 RGBA over a deterministic noise background, producing
/// linear f32 RGB.
///
/// Linearizes both foreground and background, then alpha-blends in linear space.
/// The noise background avoids the structured-pattern amplification that a
/// checkerboard causes in the multi-scale SSIM metric.
///
/// Uses straight alpha: `out = src * a + bg * (1-a)`.
pub(crate) fn composite_srgb8_rgba_to_linear(row: &[[u8; 4]], y: usize, out: &mut [[f32; 3]]) {
    for (x, &[r, g, b, a]) in row.iter().enumerate() {
        if a == 255 {
            out[x] = [
                srgb_u8_to_linear(r),
                srgb_u8_to_linear(g),
                srgb_u8_to_linear(b),
            ];
        } else if a == 0 {
            let bg = alpha_background_linear(x, y);
            out[x] = [bg, bg, bg];
        } else {
            let alpha = a as f32 * (1.0 / 255.0);
            let inv = 1.0 - alpha;
            let bg = alpha_background_linear(x, y);
            let rl = srgb_u8_to_linear(r);
            let gl = srgb_u8_to_linear(g);
            let bl = srgb_u8_to_linear(b);
            out[x] = [
                rl.mul_add(alpha, bg * inv),
                gl.mul_add(alpha, bg * inv),
                bl.mul_add(alpha, bg * inv),
            ];
        }
    }
}

/// Composite sRGB u8 BGRA over a deterministic noise background, producing
/// linear f32 RGB.
///
/// Swizzles B↔R during linearization. Alpha blending in linear space.
pub(crate) fn composite_srgb8_bgra_to_linear(row: &[[u8; 4]], y: usize, out: &mut [[f32; 3]]) {
    for (x, &[b, g, r, a]) in row.iter().enumerate() {
        if a == 255 {
            out[x] = [
                srgb_u8_to_linear(r),
                srgb_u8_to_linear(g),
                srgb_u8_to_linear(b),
            ];
        } else if a == 0 {
            let bg = alpha_background_linear(x, y);
            out[x] = [bg, bg, bg];
        } else {
            let alpha = a as f32 * (1.0 / 255.0);
            let inv = 1.0 - alpha;
            let bg = alpha_background_linear(x, y);
            let rl = srgb_u8_to_linear(r);
            let gl = srgb_u8_to_linear(g);
            let bl = srgb_u8_to_linear(b);
            out[x] = [
                rl.mul_add(alpha, bg * inv),
                gl.mul_add(alpha, bg * inv),
                bl.mul_add(alpha, bg * inv),
            ];
        }
    }
}

/// Composite linear f32 RGBA over a noise background, producing linear f32 RGB.
pub(crate) fn composite_linear_f32_rgba(row: &[[f32; 4]], y: usize, out: &mut [[f32; 3]]) {
    for (x, &[r, g, b, a]) in row.iter().enumerate() {
        if a >= 1.0 {
            out[x] = [r, g, b];
        } else if a <= 0.0 {
            let bg = alpha_background_linear(x, y);
            out[x] = [bg, bg, bg];
        } else {
            let inv = 1.0 - a;
            let bg = alpha_background_linear(x, y);
            out[x] = [
                r.mul_add(a, bg * inv),
                g.mul_add(a, bg * inv),
                b.mul_add(a, bg * inv),
            ];
        }
    }
}

/// Composite sRGB u16 RGBA over a noise background, producing linear f32 RGB.
///
/// u16 values 0-65535 are linearized via `srgb_u16_to_linear()`, then alpha-blended
/// in linear space against the noise background.
pub(crate) fn composite_srgb16_rgba_to_linear(
    row: &[u8],
    width: usize,
    y: usize,
    out: &mut [[f32; 3]],
) {
    for (x, out_pixel) in out.iter_mut().enumerate().take(width) {
        let off = x * 8; // 4 channels × 2 bytes
        let r = u16::from_ne_bytes([row[off], row[off + 1]]);
        let g = u16::from_ne_bytes([row[off + 2], row[off + 3]]);
        let b = u16::from_ne_bytes([row[off + 4], row[off + 5]]);
        let a = u16::from_ne_bytes([row[off + 6], row[off + 7]]);
        if a == 65535 {
            *out_pixel = [
                srgb_u16_to_linear(r),
                srgb_u16_to_linear(g),
                srgb_u16_to_linear(b),
            ];
        } else if a == 0 {
            let bg = alpha_background_linear(x, y);
            *out_pixel = [bg, bg, bg];
        } else {
            let alpha = a as f32 / 65535.0;
            let inv = 1.0 - alpha;
            let bg = alpha_background_linear(x, y);
            let rl = srgb_u16_to_linear(r);
            let gl = srgb_u16_to_linear(g);
            let bl = srgb_u16_to_linear(b);
            *out_pixel = [
                rl.mul_add(alpha, bg * inv),
                gl.mul_add(alpha, bg * inv),
                bl.mul_add(alpha, bg * inv),
            ];
        }
    }
}

/// Test helper: true when `incant!` dispatch of the bulk XYB conversions lands on the scalar
/// tier on this host (no vector token available, or all disabled by a test).
#[cfg(test)]
#[cfg(target_arch = "x86_64")]
pub(crate) fn dispatches_scalar() -> bool {
    use archmage::SimdToken as _;
    archmage::X64V3Token::summon().is_none()
}
#[cfg(test)]
#[cfg(target_arch = "aarch64")]
pub(crate) fn dispatches_scalar() -> bool {
    use archmage::SimdToken as _;
    archmage::NeonToken::summon().is_none()
}
#[cfg(test)]
#[cfg(target_arch = "wasm32")]
pub(crate) fn dispatches_scalar() -> bool {
    use archmage::SimdToken as _;
    archmage::Wasm128Token::summon().is_none()
}
#[cfg(test)]
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
)))]
pub(crate) fn dispatches_scalar() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::ColorPrimaries;

    /// Gated-mirror parity (issue #17): the unclamped scalar converter
    /// must be BIT-identical to the clamped entry's scalar remainder for
    /// in-gamut input. n = 7 (< the 8-wide SIMD chunk) forces the clamped
    /// entry onto its scalar remainder, where the clamp is an arithmetic
    /// no-op on in-[0,1] values — any divergence means the mirror drifted.
    #[test]
    fn unclamped_matches_clamped_scalar_for_in_gamut() {
        let pixels: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.25, 0.5, 0.75],
            [0.999, 0.001, 0.5],
            [0.1, 0.9, 0.3],
            [0.66, 0.33, 0.0],
            [0.5, 0.5, 0.5],
        ];
        let n = pixels.len();
        // Production (Rev1–3) and canonical (Rev4) arithmetic alike.
        for rev in [
            crate::feature_defs::FormulaRevision::Rev3,
            crate::feature_defs::FormulaRevision::Rev4,
        ] {
            let (mut xc, mut yc, mut bc) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
            let (mut xu, mut yu, mut bu) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
            linear_to_positive_xyb_planar_into(&pixels, &mut xc, &mut yc, &mut bc, rev);
            linear_to_positive_xyb_planar_into_unclamped(&pixels, &mut xu, &mut yu, &mut bu, rev);
            for i in 0..n {
                assert_eq!(
                    xc[i].to_bits(),
                    xu[i].to_bits(),
                    "{rev:?}: X differs at px {i}"
                );
                assert_eq!(
                    yc[i].to_bits(),
                    yu[i].to_bits(),
                    "{rev:?}: Y differs at px {i}"
                );
                assert_eq!(
                    bc[i].to_bits(),
                    bu[i].to_bits(),
                    "{rev:?}: B differs at px {i}"
                );
            }
        }
    }

    /// featcanon D1: the opsin leaf follows the revision it is GIVEN, in any
    /// process. Rev4 is the canonical body; Rev1–3 is the production dispatch.
    /// Run in a Rev3 and a Rev4 process, so the process switch provably does
    /// not reach the leaf.
    fn opsin_leaf_follows_the_passed_revision() {
        use crate::feature_defs::FormulaRevision;
        // 7 pixels: all remainder, where production (x86: `cbrtf_fast`) and
        // canonical (`cbrt_midp_f32`) differ for most colours.
        let px: Vec<[u8; 3]> = (0..7u8)
            .map(|i| [i.wrapping_mul(37), 255 - i * 29, i * 11 + 3])
            .collect();
        let n = px.len();
        let conv = |rev: Option<FormulaRevision>| {
            let (mut x, mut y, mut b) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
            match rev {
                Some(r) => {
                    srgb_to_positive_xyb_planar_into_at_revision(&px, &mut x, &mut y, &mut b, r)
                }
                None => srgb_to_positive_xyb_planar_into(&px, &mut x, &mut y, &mut b),
            }
            [x, y, b].map(|p| p.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
        };
        let canon = {
            let (mut x, mut y, mut b) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
            srgb_xyb_canon(
                &px,
                &mut x,
                &mut y,
                &mut b,
                crate::featcanon::Mode::Canon64,
                false,
            );
            [x, y, b].map(|p| p.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
        };
        let production = conv(None);
        for rev in [
            FormulaRevision::Rev1,
            FormulaRevision::Rev2,
            FormulaRevision::Rev3,
        ] {
            assert_eq!(
                conv(Some(rev)),
                production,
                "{rev:?} must run the production kernel"
            );
        }
        assert_eq!(
            conv(Some(FormulaRevision::Rev4)),
            canon,
            "Rev4 must run the canonical body"
        );
        assert_eq!(
            conv(Some(FormulaRevision::Rev5)),
            canon,
            "Rev5 must run the canonical body"
        );
        println!(
            "OPSIN-LEAF-RAN process={:?} production_differs_from_canonical={}",
            crate::ssim_form::active_revision(),
            production != canon
        );
    }

    #[test]
    fn opsin_leaf_follows_the_passed_revision_in_a_rev3_process() {
        if !crate::ssim_form::run_at_revision(
            "3",
            "color::tests::opsin_leaf_follows_the_passed_revision_in_a_rev3_process",
            "OPSIN-LEAF-RAN",
        ) {
            return;
        }
        opsin_leaf_follows_the_passed_revision();
    }

    #[test]
    fn opsin_leaf_follows_the_passed_revision_in_a_rev4_process() {
        if !crate::ssim_form::run_at_revision(
            "4",
            "color::tests::opsin_leaf_follows_the_passed_revision_in_a_rev4_process",
            "OPSIN-LEAF-RAN",
        ) {
            return;
        }
        opsin_leaf_follows_the_passed_revision();
    }

    /// Scalar tier: the unclamped converter runs the clamped converter's chunk arithmetic
    /// (2026-09-25), so on in-gamut input the two agree bit for bit at EVERY position of a
    /// band, chunks as well as remainder. Calls the scalar variants directly, so it runs on
    /// every host. Bands of 1..=40 pixels cover every remainder and up to 5 full chunks.
    #[test]
    fn scalar_tier_unclamped_matches_clamped_at_every_position() {
        use archmage::SimdToken as _;
        let token = archmage::ScalarToken::summon().expect("scalar token is infallible");
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        };
        // In-gamut: every component in [0, 1], including the exact endpoints.
        let mut pixels: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.5]];
        pixels.extend((0..500).map(|_| [next(), next(), next()]));
        for n in 1..=40usize {
            for start in 0..(pixels.len() - n).min(16) {
                let band = &pixels[start..start + n];
                let (mut xc, mut yc, mut bc) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
                let (mut xu, mut yu, mut bu) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
                linear_to_positive_xyb_planar_inner_scalar(token, band, &mut xc, &mut yc, &mut bc);
                linear_positive_unclamped_inner_scalar(token, band, &mut xu, &mut yu, &mut bu);
                for i in 0..n {
                    assert_eq!(
                        (xc[i].to_bits(), yc[i].to_bits(), bc[i].to_bits()),
                        (xu[i].to_bits(), yu[i].to_bits(), bu[i].to_bits()),
                        "n={n} start={start}: unclamped differs from clamped at px {i}"
                    );
                }
            }
        }
    }

    /// Scalar tier: a pixel gets the same bits in a full 8-pixel chunk and in the
    /// remainder (2026-09-25 fix). Calls the scalar variants directly, so it runs on
    /// every host, not only where dispatch lands on scalar. `mul_add` is unfused on
    /// this tier; before the fix the remainder's fused `cbrtf_fast` form disagreed with
    /// the chunk on ~85 % of colours.
    #[test]
    fn scalar_tier_remainder_matches_full_chunk_arithmetic() {
        use archmage::SimdToken as _;
        let token = archmage::ScalarToken::summon().expect("scalar token is infallible");

        // Deterministic pixel sets: an sRGB8 sample, and linear values including
        // out-of-range ones (the linear converter clamps).
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut srgb: Vec<[u8; 3]> = vec![[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 0, 255]];
        srgb.extend((0..3000).map(|_| {
            let v = next();
            [v as u8, (v >> 8) as u8, (v >> 16) as u8]
        }));
        let lin: Vec<[f32; 3]> = (0..3000)
            .map(|_| {
                let f = |v: u64| (v & 0xFF_FFFF) as f32 / (1u64 << 24) as f32 * 1.5 - 0.25;
                let v = next();
                [f(v), f(v >> 20), f(v.rotate_left(7))]
            })
            .collect();

        // Each pixel alone in a full chunk is the reference; every other position and
        // band length must reproduce it exactly.
        type Convert<'a, P> = &'a dyn Fn(&[P], &mut [f32], &mut [f32], &mut [f32]);
        fn check<P: Copy>(what: &str, px: &[P], convert: Convert<'_, P>) {
            let one = |p: P| {
                let band = [p; 8];
                let (mut x, mut y, mut b) = ([0f32; 8], [0f32; 8], [0f32; 8]);
                convert(&band, &mut x, &mut y, &mut b);
                for l in 1..8 {
                    assert_eq!(
                        (x[0].to_bits(), y[0].to_bits(), b[0].to_bits()),
                        (x[l].to_bits(), y[l].to_bits(), b[l].to_bits()),
                        "{what}: lanes of one full chunk disagree"
                    );
                }
                (x[0].to_bits(), y[0].to_bits(), b[0].to_bits())
            };
            let reference: Vec<_> = px.iter().map(|&p| one(p)).collect();
            for n in 1..=20usize {
                for start in 0..px.len().saturating_sub(n).min(64) {
                    let band = &px[start..start + n];
                    let (mut x, mut y, mut b) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
                    convert(band, &mut x, &mut y, &mut b);
                    for i in 0..n {
                        assert_eq!(
                            (x[i].to_bits(), y[i].to_bits(), b[i].to_bits()),
                            reference[start + i],
                            "{what}: n={n} band start={start} lane {i} differs from the full-chunk value"
                        );
                    }
                }
            }
        }

        check("srgb_to_positive_xyb", &srgb, &|p, x, y, b| {
            srgb_to_positive_xyb_planar_inner_scalar(token, p, x, y, b)
        });
        check("srgb_to_xyb", &srgb, &|p, x, y, b| {
            srgb_to_xyb_planar_inner_scalar(token, p, x, y, b)
        });
        check("linear_to_positive_xyb", &lin, &|p, x, y, b| {
            linear_to_positive_xyb_planar_inner_scalar(token, p, x, y, b)
        });
    }

    /// rev4canon C1: at [`FormulaRevision::Rev4`] every pixel's XYB is
    /// position-independent — identical bits in a full SIMD chunk and in the
    /// row tail, on EVERY dispatch tier, at band widths not divisible by the
    /// lane count. The canonical body ([`srgb_xyb_canon`]/[`linear_xyb_canon`])
    /// is one scalar per-pixel form, so this holds by construction; the gate
    /// exists so no later change can route Rev4's opsin leaf back to a
    /// tier-dispatched kernel whose remainder runs different arithmetic (the
    /// 2026-09-25 chunk-vs-remainder defect).
    #[test]
    fn rev4_xyb_body_tail_bits_identical_across_tiers() {
        // The permutation disables SIMD tokens PROCESS-WIDE, and the archmage
        // lock only serialises its own holders, so in this shared lib test
        // binary it would race every tier-sensitive test. The test therefore
        // re-executes itself as a child process (the repo's self-re-exec
        // pattern, e.g. `metric::bake`) and the permutation runs there alone.
        const CHILD: &str = "ZENSIM_XYB_TIER_CHILD";
        const SENTINEL: &str = "REV4-XYB-BODY-TAIL permutations=";
        if std::env::var_os(CHILD).is_none() {
            let exe = std::env::current_exe().expect("test binary path");
            let t0 = std::time::Instant::now();
            let out = std::process::Command::new(exe)
                .args([
                    "color::tests::rev4_xyb_body_tail_bits_identical_across_tiers",
                    "--exact",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD, "1")
                .output()
                .expect("re-exec the test binary");
            let (so, se) = (
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr),
            );
            assert!(out.status.success(), "child failed\n{so}\n{se}");
            assert!(so.contains(SENTINEL), "the child never ran its body\n{so}");
            println!(
                "rev4 XYB body==tail child: {:.2?}; {}",
                t0.elapsed(),
                so.lines().find(|l| l.contains(SENTINEL)).unwrap_or("")
            );
            return;
        }
        let _tier_lock = archmage::testing::lock_token_testing();
        use crate::feature_defs::FormulaRevision::Rev4;
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut srgb: Vec<[u8; 3]> = vec![[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 0, 255]];
        srgb.extend((0..1500).map(|_| {
            let v = next();
            [v as u8, (v >> 8) as u8, (v >> 16) as u8]
        }));
        let lin: Vec<[f32; 3]> = (0..1500)
            .map(|_| {
                let f = |v: u64| (v & 0xFF_FFFF) as f32 / (1u64 << 24) as f32 * 1.5 - 0.25;
                let v = next();
                [f(v), f(v >> 20), f(v.rotate_left(7))]
            })
            .collect();

        type Convert<'a, P> = &'a dyn Fn(&[P], &mut [f32], &mut [f32], &mut [f32]);
        fn check<P: Copy>(what: &str, px: &[P], convert: Convert<'_, P>) {
            let one = |p: P| {
                let band = [p; 8];
                let (mut x, mut y, mut b) = ([0f32; 8], [0f32; 8], [0f32; 8]);
                convert(&band, &mut x, &mut y, &mut b);
                for l in 1..8 {
                    assert_eq!(
                        (x[0].to_bits(), y[0].to_bits(), b[0].to_bits()),
                        (x[l].to_bits(), y[l].to_bits(), b[l].to_bits()),
                        "{what}: lanes of one full chunk disagree"
                    );
                }
                (x[0].to_bits(), y[0].to_bits(), b[0].to_bits())
            };
            let reference: Vec<_> = px.iter().map(|&p| one(p)).collect();
            // widths with a nonempty remainder under both the 8- and 16-wide
            // chunkings the SIMD kernels used.
            for n in [9usize, 13, 17, 23, 29, 31, 33, 37, 41] {
                for start in 0..px.len().saturating_sub(n).min(48) {
                    let band = &px[start..start + n];
                    let (mut x, mut y, mut b) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
                    convert(band, &mut x, &mut y, &mut b);
                    for i in 0..n {
                        assert_eq!(
                            (x[i].to_bits(), y[i].to_bits(), b[i].to_bits()),
                            reference[start + i],
                            "{what}: n={n} start={start} pos {i}: tail/body bits differ"
                        );
                    }
                }
            }
        }

        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            check("srgb_to_positive_xyb@rev4", &srgb, &|p, x, y, b| {
                srgb_to_positive_xyb_planar_into_at_revision(p, x, y, b, Rev4)
            });
            check("linear_to_positive_xyb@rev4", &lin, &|p, x, y, b| {
                linear_to_positive_xyb_planar_into(p, x, y, b, Rev4)
            });
            check(
                "linear_to_positive_xyb_unclamped@rev4",
                &lin,
                &|p, x, y, b| linear_to_positive_xyb_planar_into_unclamped(p, x, y, b, Rev4),
            );
            eprintln!("rev4 XYB body==tail: permutation {} clean", perm.label);
        });
        assert!(report.permutations_run >= 1);
        println!(
            "REV4-XYB-BODY-TAIL permutations={} all position-identical",
            report.permutations_run
        );
    }

    /// The unclamped converter handles out-of-gamut input without NaN/inf:
    /// the opsin mix's `max(0)` keeps the cube-root domain valid.
    #[test]
    fn unclamped_out_of_gamut_stays_finite() {
        let pixels: Vec<[f32; 3]> = vec![
            [1.66, -0.125, -0.018], // BT.2020 red through the matrix
            [-0.51, 1.04, -0.31],
            [2.0, -1.0, 3.0],
        ];
        let n = pixels.len();
        for rev in [
            crate::feature_defs::FormulaRevision::Rev3,
            crate::feature_defs::FormulaRevision::Rev4,
        ] {
            let (mut x, mut y, mut b) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
            linear_to_positive_xyb_planar_into_unclamped(&pixels, &mut x, &mut y, &mut b, rev);
            for i in 0..n {
                assert!(
                    x[i].is_finite() && y[i].is_finite() && b[i].is_finite(),
                    "{rev:?}"
                );
            }
        }
    }

    /// Verify P3→sRGB matrix: sRGB white (1,1,1) should stay (1,1,1).
    #[test]
    fn p3_to_srgb_preserves_white() {
        let mut rgb = [1.0f32, 1.0, 1.0];
        apply_gamut_matrix(&mut rgb, ColorPrimaries::DisplayP3, GamutMapping::Clip);
        for (c, &val) in rgb.iter().enumerate() {
            assert!(
                (val - 1.0).abs() < 1e-4,
                "P3 white channel {c}: expected 1.0, got {val}",
            );
        }
    }

    /// Verify BT.2020→sRGB matrix: sRGB white (1,1,1) should stay (1,1,1).
    #[test]
    fn bt2020_to_srgb_preserves_white() {
        let mut rgb = [1.0f32, 1.0, 1.0];
        apply_gamut_matrix(&mut rgb, ColorPrimaries::Bt2020, GamutMapping::Clip);
        for (c, &val) in rgb.iter().enumerate() {
            assert!(
                (val - 1.0).abs() < 1e-4,
                "BT.2020 white channel {c}: expected 1.0, got {val}",
            );
        }
    }

    /// P3 red primary (1,0,0) in linear P3 → sRGB linear should clamp:
    /// R > 1.0 → clamped to 1.0, G/B negative → clamped to 0.0.
    #[test]
    fn p3_red_clamps_to_srgb_gamut() {
        let mut rgb = [1.0f32, 0.0, 0.0];
        apply_gamut_matrix(&mut rgb, ColorPrimaries::DisplayP3, GamutMapping::Clip);
        assert_eq!(rgb[0], 1.0, "R should be clamped to 1.0");
        assert_eq!(rgb[1], 0.0, "G should be clamped to 0.0");
        assert_eq!(rgb[2], 0.0, "B should be clamped to 0.0");
    }

    /// Srgb primaries should be a no-op.
    #[test]
    fn srgb_is_noop() {
        let mut rgb = [0.5f32, 0.3, 0.8];
        let original = rgb;
        apply_gamut_matrix(&mut rgb, ColorPrimaries::Srgb, GamutMapping::Clip);
        assert_eq!(rgb, original);
    }

    /// P3 grey (0.5, 0.5, 0.5) should stay approximately (0.5, 0.5, 0.5)
    /// since the matrices share D65 whitepoint.
    #[test]
    fn p3_grey_stays_grey() {
        let mut rgb = [0.5f32, 0.5, 0.5];
        apply_gamut_matrix(&mut rgb, ColorPrimaries::DisplayP3, GamutMapping::Clip);
        for (c, &val) in rgb.iter().enumerate() {
            assert!(
                (val - 0.5).abs() < 1e-3,
                "P3 grey channel {c}: expected ~0.5, got {val}",
            );
        }
    }

    /// BT.2020 red (1,0,0) should clamp more aggressively than P3 red.
    #[test]
    fn bt2020_red_clamps_to_srgb_gamut() {
        let mut rgb = [1.0f32, 0.0, 0.0];
        apply_gamut_matrix(&mut rgb, ColorPrimaries::Bt2020, GamutMapping::Clip);
        assert_eq!(rgb[0], 1.0, "R should be clamped to 1.0");
        assert_eq!(rgb[1], 0.0, "G should be clamped to 0.0");
        assert_eq!(rgb[2], 0.0, "B should be clamped to 0.0");
    }

    /// Verify matrix rows sum to ~1.0 (whitepoint preservation).
    #[test]
    fn matrix_rows_sum_to_one() {
        for (name, m) in [("P3", P3_TO_SRGB), ("BT.2020", BT2020_TO_SRGB)] {
            for (row_idx, row) in m.iter().enumerate() {
                let sum: f32 = row.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-4,
                    "{name} row {row_idx} sum: {sum} (expected ~1.0)"
                );
            }
        }
    }
}

/// Bench-only hooks (`benches/pu21_bench.rs`) — not public API.
#[doc(hidden)]
pub fn bench_pu_xyb_dispatch(
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    linear_to_pu_xyb_planar_into(pixels, x_out, y_out, b_out);
}

/// Bench-only scalar reference — see [`bench_pu_xyb_dispatch`].
#[doc(hidden)]
pub fn bench_pu_xyb_scalar(
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    for (i, p) in pixels.iter().enumerate() {
        let (x, y, b) = pu_xyb_pixel(*p);
        x_out[i] = x;
        y_out[i] = y;
        b_out[i] = b;
    }
}

#[cfg(test)]
mod pu_simd_parity_tests {
    use super::*;

    /// SIMD (incant-dispatched) vs scalar per-pixel PU-XYB: the lowp
    /// transcendentals must stay within a tight band of the scalar powf
    /// path across the full luminance range (incl. HDR extremes), so the
    /// dispatched conversion cannot shift scores between machines.
    #[test]
    fn simd_matches_scalar_within_band() {
        // 1031 px (not a multiple of 8 — exercises the tail) spanning
        // 0.001..12000 nits log-spaced with chroma variation.
        let n = 1031;
        let pixels: Vec<[f32; 3]> = (0..n)
            .map(|i| {
                let t = i as f32 / (n - 1) as f32;
                let y = 0.001 * (12_000.0f32 / 0.001).powf(t);
                [y * 1.2, y, y * 0.7]
            })
            .collect();
        let mut xs = vec![0.0f32; n];
        let mut ys = vec![0.0f32; n];
        let mut bs = vec![0.0f32; n];
        linear_to_pu_xyb_planar_into(&pixels, &mut xs, &mut ys, &mut bs);
        let mut max_d = 0.0f32;
        for (i, p) in pixels.iter().enumerate() {
            let (x, y, b) = pu_xyb_pixel(*p);
            for (got, want) in [(xs[i], x), (ys[i], y), (bs[i], b)] {
                let d = (got - want).abs();
                if d > max_d {
                    max_d = d;
                }
            }
        }
        // Outputs live in ~[0, 2.5]; 2e-3 abs keeps score impact negligible
        // while leaving room for the midp_precise transcendental error budget
        // compounding through two pow chains.
        assert!(max_d <= 2e-3, "SIMD vs scalar max |delta| = {max_d}");
    }
}

#[cfg(test)]
mod rev4vec_tests {
    //! REV4VEC bit-exactness gate: each compiled `*_canon_vec` tier variant
    //! must reproduce the scalar canonical body **bit for bit** — the
    //! variants are invoked directly with their own summoned token, so no
    //! global dispatch state is touched and the tests run under the normal
    //! parallel test harness.
    use super::*;
    use archmage::SimdToken as _;

    /// Deterministic xorshift64 — reproducible leaf inputs, no new deps.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        /// A raw f32 bit pattern — hits ±0, denormals, ±inf, NaN, and every
        /// finite range as a side effect.
        fn f32_bits(&mut self) -> f32 {
            f32::from_bits(self.next() as u32)
        }
    }

    /// Every f32 special class the gate calls out, plus the canonical
    /// clamp/opsin bounds, as literal pixels (each repeated into all three
    /// channels and rotated across them).
    fn special_linear_pixels() -> Vec<[f32; 3]> {
        let vals: Vec<f32> = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::EPSILON,
            f32::from_bits(1),           // smallest denormal
            f32::from_bits(0x007F_FFFF), // largest denormal
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::from_bits(0x7FC0_0001), // a different NaN payload
            crate::pu21::PU21_L_MIN,
            crate::pu21::PU21_L_MAX,
            crate::pu21::PU21_L_MIN * 0.5,
            crate::pu21::PU21_L_MAX * 2.0,
            K_B0,
            10000.0,
            1e-30,
            1e30,
            -1e30,
            0.5,
            0.25,
        ];
        let mut px = Vec::new();
        for (i, &a) in vals.iter().enumerate() {
            for (j, &b) in vals.iter().enumerate().take(vals.len()).skip(i % 3).take(3) {
                px.push([a, b, vals[(i + j) % vals.len()]]);
            }
            px.push([a, a, a]);
        }
        px
    }

    /// `(pixels, x_out, y_out, b_out)` converter under test.
    type PlanarFn<'a, P> = dyn FnMut(&[P], &mut [f32], &mut [f32], &mut [f32]) + 'a;

    fn assert_planes_eq(name: &str, x: &[f32], y: &[f32], b: &[f32], e: [&[f32]; 3]) {
        for i in 0..e[0].len() {
            assert_eq!(
                x[i].to_bits(),
                e[0][i].to_bits(),
                "{name}: X differs px {i}"
            );
            assert_eq!(
                y[i].to_bits(),
                e[1][i].to_bits(),
                "{name}: Y differs px {i}"
            );
            assert_eq!(
                b[i].to_bits(),
                e[2][i].to_bits(),
                "{name}: B differs px {i}"
            );
        }
    }

    /// sRGB8 → XYB: deterministic pixel sweep (all channel values mixed,
    /// n = 4159 = 64 groups of 64 (v4x's byte-plane table) + 3 chunks of 16
    /// + 15 remainder).
    #[test]
    fn srgb_xyb_canon_vec_matches_scalar_body_on_every_compiled_tier() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let n = 4159;
        let px: Vec<[u8; 3]> = (0..n)
            .map(|i| {
                if i < 256 * 16 {
                    // Dense coverage: every u8 value in every channel slot.
                    [(i & 0xFF) as u8, ((i >> 8) & 0xFF) as u8, (i % 256) as u8]
                } else {
                    [
                        rng.next() as u8,
                        (rng.next() >> 8) as u8,
                        (rng.next() >> 16) as u8,
                    ]
                }
            })
            .collect();
        // Both cube roots: `cbrt_midp` (revisions 4-6) and revision 7's
        // `cbrt_newton`.
        for cbrt7 in [false, true] {
            let (mut xe, mut ye, mut be) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
            srgb_xyb_canon_body(&px, &mut xe, &mut ye, &mut be, cbrt7);

            let run = |name: &str, f: &mut PlanarFn<'_, [u8; 3]>| {
                let (mut x, mut y, mut b) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
                f(&px, &mut x, &mut y, &mut b);
                assert_planes_eq(
                    &format!("{name} cbrt7={cbrt7}"),
                    &x,
                    &y,
                    &b,
                    [&xe, &ye, &be],
                );
            };

            run("scalar", &mut |p, x, y, b| {
                srgb_xyb_canon_vec_scalar(
                    archmage::ScalarToken::summon().unwrap(),
                    p,
                    x,
                    y,
                    b,
                    cbrt7,
                )
            });
            #[cfg(target_arch = "x86_64")]
            {
                if let Some(t) = archmage::X64V3Token::summon() {
                    run("v3", &mut |p, x, y, b| {
                        srgb_xyb_canon_vec_v3(t, p, x, y, b, cbrt7)
                    });
                }
                #[cfg(feature = "avx512")]
                {
                    if let Some(t) = archmage::X64V4Token::summon() {
                        run("v4", &mut |p, x, y, b| {
                            srgb_xyb_canon_vec_v4(t, p, x, y, b, cbrt7)
                        });
                    }
                    if let Some(t) = archmage::X64V4xToken::summon() {
                        run("v4x", &mut |p, x, y, b| {
                            srgb_xyb_canon_vec_v4x(t, p, x, y, b, cbrt7)
                        });
                    }
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                if let Some(t) = archmage::NeonToken::summon() {
                    run("neon", &mut |p, x, y, b| {
                        srgb_xyb_canon_vec_neon(t, p, x, y, b, cbrt7)
                    });
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                if let Some(t) = archmage::Wasm128Token::summon() {
                    run("wasm128", &mut |p, x, y, b| {
                        srgb_xyb_canon_vec_wasm128(t, p, x, y, b, cbrt7)
                    });
                }
            }
        }
    }

    /// linear f32 → XYB, both clamp variants: 10^7 random bit patterns
    /// (all special classes reached) + the explicit special-class pixels,
    /// n ≡ 15 (mod 16) so the remainder path is covered too.
    #[test]
    fn linear_xyb_canon_vec_matches_scalar_body_on_every_compiled_tier() {
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        let mut px: Vec<[f32; 3]> = special_linear_pixels();
        let target = px.len() + 10_000_000;
        while px.len() < target {
            px.push([rng.f32_bits(), rng.f32_bits(), rng.f32_bits()]);
        }
        while px.len() % 16 != 15 {
            px.push([rng.f32_bits(), rng.f32_bits(), rng.f32_bits()]);
        }
        let n = px.len();

        for (clamp, cbrt7) in [(true, false), (false, false), (true, true), (false, true)] {
            let (mut xe, mut ye, mut be) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
            if clamp {
                linear_xyb_canon_body::<true>(&px, &mut xe, &mut ye, &mut be, cbrt7);
            } else {
                linear_xyb_canon_body::<false>(&px, &mut xe, &mut ye, &mut be, cbrt7);
            }
            let tag = if clamp { "clamped" } else { "unclamped" };
            let run = |name: &str, f: &mut PlanarFn<'_, [f32; 3]>| {
                let (mut x, mut y, mut b) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
                f(&px, &mut x, &mut y, &mut b);
                for i in 0..n {
                    assert_eq!(
                        (x[i].to_bits(), y[i].to_bits(), b[i].to_bits()),
                        (xe[i].to_bits(), ye[i].to_bits(), be[i].to_bits()),
                        "{tag}/{name} cbrt7={cbrt7}: differs at px {i}"
                    );
                }
            };
            run("scalar", &mut |p, x, y, b| {
                if clamp {
                    linear_xyb_canon_vec_scalar::<true>(
                        archmage::ScalarToken::summon().unwrap(),
                        p,
                        x,
                        y,
                        b,
                        cbrt7,
                    )
                } else {
                    linear_xyb_canon_vec_scalar::<false>(
                        archmage::ScalarToken::summon().unwrap(),
                        p,
                        x,
                        y,
                        b,
                        cbrt7,
                    )
                }
            });
            #[cfg(target_arch = "x86_64")]
            {
                if let Some(t) = archmage::X64V3Token::summon() {
                    run("v3", &mut |p, x, y, b| {
                        if clamp {
                            linear_xyb_canon_vec_v3::<true>(t, p, x, y, b, cbrt7)
                        } else {
                            linear_xyb_canon_vec_v3::<false>(t, p, x, y, b, cbrt7)
                        }
                    });
                }
                #[cfg(feature = "avx512")]
                {
                    if let Some(t) = archmage::X64V4Token::summon() {
                        run("v4", &mut |p, x, y, b| {
                            if clamp {
                                linear_xyb_canon_vec_v4::<true>(t, p, x, y, b, cbrt7)
                            } else {
                                linear_xyb_canon_vec_v4::<false>(t, p, x, y, b, cbrt7)
                            }
                        });
                    }
                    if let Some(t) = archmage::X64V4xToken::summon() {
                        run("v4x", &mut |p, x, y, b| {
                            if clamp {
                                linear_xyb_canon_vec_v4x::<true>(t, p, x, y, b, cbrt7)
                            } else {
                                linear_xyb_canon_vec_v4x::<false>(t, p, x, y, b, cbrt7)
                            }
                        });
                    }
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                if let Some(t) = archmage::NeonToken::summon() {
                    run("neon", &mut |p, x, y, b| {
                        if clamp {
                            linear_xyb_canon_vec_neon::<true>(t, p, x, y, b, cbrt7)
                        } else {
                            linear_xyb_canon_vec_neon::<false>(t, p, x, y, b, cbrt7)
                        }
                    });
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                if let Some(t) = archmage::Wasm128Token::summon() {
                    run("wasm128", &mut |p, x, y, b| {
                        if clamp {
                            linear_xyb_canon_vec_wasm128::<true>(t, p, x, y, b, cbrt7)
                        } else {
                            linear_xyb_canon_vec_wasm128::<false>(t, p, x, y, b, cbrt7)
                        }
                    });
                }
            }
        }
    }

    /// PU-XYB: same input space as the unclamped linear leaf, plus the
    /// log-spaced nits ramp the PU tests use.
    #[test]
    fn pu_xyb_canon_vec_matches_scalar_body_on_every_compiled_tier() {
        let mut rng = Rng(0xB529_7A4D_E969_A4F1);
        let mut px: Vec<[f32; 3]> = special_linear_pixels();
        for i in 0..1031usize {
            let t = i as f32 / 1030.0;
            let y = 0.001f32 * (12_000.0f32 / 0.001).powf(t);
            px.push([y * 1.2, y, y * 0.7]);
        }
        let target = px.len() + 10_000_000;
        while px.len() < target {
            px.push([rng.f32_bits(), rng.f32_bits(), rng.f32_bits()]);
        }
        while px.len() % 16 != 15 {
            px.push([rng.f32_bits(), rng.f32_bits(), rng.f32_bits()]);
        }
        let n = px.len();

        let (mut xe, mut ye, mut be) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
        pu_xyb_canon_body(&px, &mut xe, &mut ye, &mut be);
        let run = |name: &str, f: &mut PlanarFn<'_, [f32; 3]>| {
            let (mut x, mut y, mut b) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
            f(&px, &mut x, &mut y, &mut b);
            for i in 0..n {
                assert_eq!(
                    (x[i].to_bits(), y[i].to_bits(), b[i].to_bits()),
                    (xe[i].to_bits(), ye[i].to_bits(), be[i].to_bits()),
                    "{name}: differs at px {i} ({:?})",
                    px[i]
                );
            }
        };
        run("scalar", &mut |p, x, y, b| {
            pu_xyb_canon_vec_scalar(archmage::ScalarToken::summon().unwrap(), p, x, y, b)
        });
        #[cfg(target_arch = "x86_64")]
        {
            if let Some(t) = archmage::X64V3Token::summon() {
                run("v3", &mut |p, x, y, b| pu_xyb_canon_vec_v3(t, p, x, y, b));
            }
            #[cfg(feature = "avx512")]
            {
                if let Some(t) = archmage::X64V4Token::summon() {
                    run("v4", &mut |p, x, y, b| pu_xyb_canon_vec_v4(t, p, x, y, b));
                }
                if let Some(t) = archmage::X64V4xToken::summon() {
                    run("v4x", &mut |p, x, y, b| pu_xyb_canon_vec_v4x(t, p, x, y, b));
                }
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if let Some(t) = archmage::NeonToken::summon() {
                run("neon", &mut |p, x, y, b| {
                    pu_xyb_canon_vec_neon(t, p, x, y, b)
                });
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            if let Some(t) = archmage::Wasm128Token::summon() {
                run("wasm128", &mut |p, x, y, b| {
                    pu_xyb_canon_vec_wasm128(t, p, x, y, b)
                });
            }
        }
    }
}
