//! THE per-pixel SSIM dissimilarity — one owner for a form that was
//! hand-copied at 36 kernel sites.
//!
//! # Why this module exists
//!
//! Every SSIM-derived slot in the crate (`basic`'s `ssim_*`, the `masked`
//! block, the `IW` block — 3 signals x 2 blocks x 4 scales x 3 channels, plus
//! the basic block's own) is built from ONE per-pixel expression:
//!
//! ```text
//! num_m   = 1 - (mu1 - mu2)^2          // luminance
//! num_s   = 2*cov + C2                 // structure numerator
//! denom_s = var1 + var2 + C2           // structure denominator
//! d_raw   = 1 - num_m * num_s / denom_s
//! ```
//!
//! That expression was written out by hand **36 times** — 10 in
//! [`crate::fused`] and 26 in [`crate::simd_ops`], across four SIMD tiers and
//! their scalar tails, in vector and scalar spellings. `C2` itself had three
//! separate declarations. A form with 36 owners cannot be revised, which is
//! exactly the position the defect audit found the crate in: F4 (the one live
//! arithmetic defect) is a property of this expression, and there was nowhere
//! to fix it.
//!
//! Everything here is `#[inline(always)]` and generic over the backend trait,
//! following [`crate::fused`]'s `raw_moments_accumulate*` precedent. `nm` over
//! a release build shows zero `ssim_dissim_*` symbols surviving.
//!
//! # F4, and why the fix needs a form rather than a clamp
//!
//! `num_s/denom_s` is bounded in `[-1, 1]` by construction: `|2*cov| <= var1 +
//! var2` (Cauchy-Schwarz), so the structure term cannot run away. `num_m` is
//! `1 - D^2` for a mean difference `D`, which is **unbounded below**, so a
//! large local mean difference makes `d_raw` a large positive number. MEASURED
//! (`benchmarks/ssim_moment_explosion_2026-07-16.md`, 2,322,579 rows):
//! `f313 = iw_ssim_4th s0 ch2` reaches **5,814,302** against a photographic
//! p99.9 of **0.48**.
//!
//! The weights are NOT the amplifier, and the numbers on record prove it: the
//! `masked` weight is `1/(1 + k*a)`, bounded in `(0, 1]`, while the `IW`
//! weight is `1 + k*a` and unbounded — yet `f241` (masked, 5,797,029) and
//! `f313` (IW, 5,814,302) agree to **0.3 %**. A bounded weight cannot produce
//! 5.8e6 from a bounded `d_raw`, and an unbounded weight that mattered could
//! not land within 0.3 % of a bounded one. Both are ~1 there because the
//! pathology lives in flat regions where activity is ~0. The amplifier is
//! `num_m`.
//!
//! # Provenance: this is ssimulacra2's form, faithfully inherited
//!
//! `zensim/src/lib.rs` describes the no-`C1` luminance term as "ssimulacra2's
//! variant". VERIFIED against our own SSIMULACRA2 implementation rather than
//! assumed: `fast-ssim2`'s `simd_ops.rs`, `lib.rs` and `strip.rs` all compute
//! `num_m = mu_diff.mul_add(-mu_diff, 1.0)` with `C2 = 0.0009` and no `C1`.
//! So F4 is **inherited from the algorithm**, not a zensim coding slip, and
//! any bounded form here is a deliberate, measured DEVIATION from that
//! lineage — taken for a learned metric that must feed monotone linear heads,
//! and recorded as such.
//!
//! (`fast-ssim2` carries the same unbounded term. It is a different repo and
//! is NOT touched from here; the observation is reported, not acted on.)

use crate::feature_defs::FormulaRevision;
use archmage::autoversion;
use magetypes::simd::backends::{F32x8Backend, F32x16Backend};
use magetypes::simd::generic::{f32x8 as GenericF32x8, f32x16};

/// Scratch for the exact f64 second-pass kernel. Row-ring storage is
/// proportional to width and blur radius, independent of image height.
///
/// NOT on the served path any more: revision 3 is served by the FUSED
/// direct-error moments inside the existing H/V pass (`blur::fused_blur_h_ssim`
/// accumulates `Σ(a-b)²` in place of `Σab`, and `fused::fused_vblur_ssim_inner`
/// forms the dissimilarity from it). This kernel is the exactness REFERENCE the
/// bounded-error tests measure against — see
/// `benchmarks/stable_ssim_kernel_2026-09-08.md`.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Default)]
pub(crate) struct StableSsimScratch {
    rows: Vec<[f64; 4]>,
    vertical: Vec<[f64; 4]>,
}

/// Form SSIM from pairwise error moments, retaining precision near identity.
/// Inputs and output are contiguous planar arrays with stride `width`.
/// All moments use f64, including source products and sliding-window updates;
/// outputs round once to f32. No covariance subtraction enters the numerator.
/// The spatial kernel is one reflect-101 box with the specified radius.
#[cfg_attr(not(test), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
#[autoversion]
pub(crate) fn stable_ssim_plane(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    radius: usize,
    form: SsimLumaForm,
    out: &mut [f32],
    scratch: &mut StableSsimScratch,
) {
    assert!(width > 0 && height > 0);
    let n = width.checked_mul(height).expect("plane extent");
    assert!(src.len() >= n && dst.len() >= n && out.len() >= n);
    let diameter = radius
        .checked_mul(2)
        .and_then(|x| x.checked_add(1))
        .expect("blur diameter");
    scratch.rows.resize(
        width.checked_mul(diameter).expect("row ring extent"),
        [0.0; 4],
    );
    scratch.vertical.resize(width, [0.0; 4]);
    scratch.vertical.fill([0.0; 4]);
    let mirror = |i: isize, len: usize| {
        if i >= 0 && (i as usize) < len {
            i as usize
        } else {
            crate::metric::reflect_index(i.unsigned_abs(), len)
        }
    };
    let rad = isize::try_from(radius).expect("signed radius");
    let horizontal = |y: usize, output: &mut [[f64; 4]]| {
        let at = |x: usize| {
            let a = src[y * width + x] as f64;
            let b = dst[y * width + x] as f64;
            let e = a - b;
            [a, b, a * a + b * b, e * e]
        };
        let mut sums = [0.0; 4];
        for dx in -rad..=rad {
            let m = at(mirror(dx, width));
            for k in 0..4 {
                sums[k] += m[k];
            }
        }
        for (x, cell) in output.iter_mut().enumerate() {
            *cell = sums;
            let add = at(mirror(x as isize + rad + 1, width));
            let rem = at(mirror(x as isize - rad, width));
            // Exact no-op windows should not acquire recurrence drift.
            if add != rem {
                for k in 0..4 {
                    sums[k] = (sums[k] + add[k]) - rem[k];
                }
            }
        }
    };
    for row in 0..diameter {
        let y = mirror(row as isize - rad, height);
        let ring = &mut scratch.rows[row * width..(row + 1) * width];
        horizontal(y, ring);
        for (sum, add) in scratch.vertical.iter_mut().zip(ring) {
            for k in 0..4 {
                sum[k] += add[k];
            }
        }
    }
    let inv_n = 1.0 / (diameter as f64 * diameter as f64);
    let mut head = 0;
    for y in 0..height {
        for (value, m) in out[y * width..(y + 1) * width]
            .iter_mut()
            .zip(&scratch.vertical)
        {
            let a = m[0] * inv_n;
            let b = m[1] * inv_n;
            let mean_error2 = (a - b) * (a - b);
            let error_variance = (m[3] * inv_n - mean_error2).max(0.0);
            let variance_sum = (m[2] * inv_n - a * a - b * b).max(0.0);
            let luma_loss = match form {
                SsimLumaForm::Ssim2Legacy => mean_error2,
                SsimLumaForm::Clamp => mean_error2.min(1.0),
                SsimLumaForm::Lorentz => mean_error2 / (1.0 + mean_error2),
                SsimLumaForm::SsimLumaC1 => mean_error2 / (a * a + b * b + C_SSIM_LUMA as f64),
            };
            *value = (luma_loss + (1.0 - luma_loss) * error_variance / (variance_sum + C2 as f64))
                .max(0.0) as f32;
        }
        if y + 1 == height {
            break;
        }
        let ring = &mut scratch.rows[head * width..(head + 1) * width];
        for (sum, rem) in scratch.vertical.iter_mut().zip(ring.iter()) {
            for k in 0..4 {
                sum[k] -= rem[k];
            }
        }
        horizontal(mirror(y as isize + rad + 1, height), ring);
        for (sum, add) in scratch.vertical.iter_mut().zip(ring.iter()) {
            for k in 0..4 {
                sum[k] += add[k];
            }
        }
        head = (head + 1) % diameter;
    }
}

/// **Ablation mirror of [`stable_ssim_plane`], for separating the two halves
/// of the correction.**
///
/// The shipped kernel changed TWO things at once: it accumulates in f64, and
/// it forms the error variance DIRECTLY from a `(a-b)^2` moment instead of
/// recovering it from `var1 + var2 - 2*cov`. Which of those actually restores
/// locality is a question the shipped kernel cannot answer, because it does
/// both. This mirrors its algorithm exactly with the two knobs independent.
///
/// `f32_accum` rounds after every arithmetic operation, which models f32
/// EXACTLY rather than approximately: for `+ - * /` on f32-representable
/// operands, computing in f64 and rounding once to f32 is correctly rounded,
/// because f64's 53 bits exceed the 2p+2 = 50 needed at p = 24.
///
/// `direct_error` false switches the fourth moment to `a*b` and the final
/// expression to the legacy `1 - num_m*(2*cov + C2)/(var1 + var2 + C2)`, i.e.
/// the shipped formulation.
///
/// A test asserts `(f64, direct)` is BIT-IDENTICAL to [`stable_ssim_plane`],
/// so this is a faithful mirror and not a second implementation drifting on
/// its own.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ablation_plane(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    radius: usize,
    form: SsimLumaForm,
    // Bit k set => moment k accumulates in f32. The four moments have very
    // different magnitudes — `a`, `b` and `a^2+b^2` are order 0.5 while
    // `(a-b)^2` is order 1e-6 on near-lossless content — so their f32 drift
    // differs by orders of magnitude and they need not share a width.
    f32_mask: u8,
    direct_error: bool,
    // Re-seed the recurrence exactly every `reset` positions (0 = never).
    // The "periodic stability reset": it does NOT make the sum a function of
    // its own window, but it bounds how far a change propagates — to one tile
    // instead of the whole row/column.
    reset: usize,
) -> Vec<f32> {
    let rk = |k: usize, x: f64| {
        if f32_mask & (1 << k) != 0 {
            x as f32 as f64
        } else {
            x
        }
    };
    // Non-moment arithmetic (the final expression) follows moment 0.
    let r = |x: f64| rk(0, x);
    let diameter = radius * 2 + 1;
    let rad = radius as isize;
    let mirror = |i: isize, len: usize| {
        if i >= 0 && (i as usize) < len {
            i as usize
        } else {
            crate::metric::reflect_index(i.unsigned_abs(), len)
        }
    };
    let at = |x: usize, y: usize| {
        let a = src[y * width + x] as f64;
        let b = dst[y * width + x] as f64;
        let fourth = if direct_error {
            let e = rk(3, a - b);
            rk(3, e * e)
        } else {
            rk(3, a * b)
        };
        [a, b, rk(2, rk(2, a * a) + rk(2, b * b)), fourth]
    };
    let horizontal = |y: usize, output: &mut [[f64; 4]]| {
        let mut sums = [0.0f64; 4];
        for dx in -rad..=rad {
            let m = at(mirror(dx, width), y);
            for k in 0..4 {
                sums[k] = rk(k, sums[k] + m[k]);
            }
        }
        for (x, cell) in output.iter_mut().enumerate() {
            if reset != 0 && x != 0 && x % reset == 0 {
                // Exact re-seed: recompute this window from its own samples.
                sums = [0.0; 4];
                for dx in -rad..=rad {
                    let m = at(mirror(x as isize + dx, width), y);
                    for k in 0..4 {
                        sums[k] = r(sums[k] + m[k]);
                    }
                }
            }
            *cell = sums;
            let add = at(mirror(x as isize + rad + 1, width), y);
            let rem = at(mirror(x as isize - rad, width), y);
            if add != rem {
                for k in 0..4 {
                    sums[k] = rk(k, rk(k, sums[k] + add[k]) - rem[k]);
                }
            }
        }
    };
    let mut rows = vec![[0.0f64; 4]; width * diameter];
    let mut vertical = vec![[0.0f64; 4]; width];
    for row in 0..diameter {
        let y = mirror(row as isize - rad, height);
        let ring = &mut rows[row * width..(row + 1) * width];
        horizontal(y, ring);
        for (sum, add) in vertical.iter_mut().zip(ring) {
            for k in 0..4 {
                sum[k] = rk(k, sum[k] + add[k]);
            }
        }
    }
    let mut out = vec![0.0f32; width * height];
    let inv_n = r(1.0 / (diameter as f64 * diameter as f64));
    let mut head = 0;
    for y in 0..height {
        for (value, m) in out[y * width..(y + 1) * width].iter_mut().zip(&vertical) {
            let a = r(m[0] * inv_n);
            let b = r(m[1] * inv_n);
            let mean_error2 = r(r(a - b) * r(a - b));
            let variance_sum = r(r(r(m[2] * inv_n) - r(a * a)) - r(b * b)).max(0.0);
            let luma_loss = match form {
                SsimLumaForm::Ssim2Legacy => mean_error2,
                SsimLumaForm::Clamp => mean_error2.min(1.0),
                SsimLumaForm::Lorentz => r(mean_error2 / r(1.0 + mean_error2)),
                SsimLumaForm::SsimLumaC1 => {
                    r(mean_error2 / r(r(r(a * a) + r(b * b)) + C_SSIM_LUMA as f64))
                }
            };
            *value = if direct_error {
                let error_variance = r(r(m[3] * inv_n) - mean_error2).max(0.0);
                r(luma_loss
                    + r(r(1.0 - luma_loss) * r(error_variance / r(variance_sum + C2 as f64))))
                .max(0.0) as f32
            } else {
                // The SHIPPED formulation: covariance recovered by subtraction.
                let cov = r(r(m[3] * inv_n) - r(a * b));
                r(1.0
                    - r(r(1.0 - luma_loss)
                        * r(r(r(2.0 * cov) + C2 as f64) / r(variance_sum + C2 as f64))))
                .max(0.0) as f32
            };
        }
        if y + 1 == height {
            break;
        }
        if reset != 0 && (y + 1) % reset == 0 {
            // Exact vertical re-seed from the ring's own rows.
            for (x, sum) in vertical.iter_mut().enumerate() {
                *sum = [0.0; 4];
                for row in 0..diameter {
                    for k in 0..4 {
                        sum[k] = rk(k, sum[k] + rows[row * width + x][k]);
                    }
                }
            }
        }
        let ring = &mut rows[head * width..(head + 1) * width];
        for (sum, rem) in vertical.iter_mut().zip(ring.iter()) {
            for k in 0..4 {
                sum[k] = rk(k, sum[k] - rem[k]);
            }
        }
        horizontal(mirror(y as isize + rad + 1, height), ring);
        for (sum, add) in vertical.iter_mut().zip(ring.iter()) {
            for k in 0..4 {
                sum[k] = rk(k, sum[k] + add[k]);
            }
        }
        head = (head + 1) % diameter;
    }
    out
}

/// **Tiled (van Herk / Gil-Werman) mirror of the moment pass.**
///
/// The sliding recurrence's locality failure is that the window sum at `x`
/// depends on every sample the running sum has passed over, not on the window.
/// Resetting the recurrence periodically bounds that, but does not remove it —
/// a change still perturbs the rest of its own tile.
///
/// This decomposition removes it outright. With tiles of exactly the window
/// diameter `D`, and per-tile `prefix` / `suffix` running sums, the window at
/// `x` is `suffix[x] + prefix[x + D - 1]`: the suffix reads only `x ..= tile
/// end`, the prefix only `next tile start ..= x + D - 1`, and both ranges lie
/// INSIDE the window. So the sum is a deterministic function of exactly the
/// window's own samples, in an order fixed by the window's offset against the
/// tile grid. Identical window contents therefore give a bit-identical sum at
/// ANY precision — locality stops being a numerical property and becomes a
/// structural one.
///
/// It is also cheaper where it counts: the serial dependency chain shrinks
/// from the whole row to `D`, and tiles are independent, so the prefix/suffix
/// passes vectorize across tiles instead of being latency-bound.
///
/// Same knobs as [`ablation_plane`], so the 2x2 can be re-run against this
/// structure.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ablation_plane_tiled(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    radius: usize,
    form: SsimLumaForm,
    f32_accum: bool,
    direct_error: bool,
) -> Vec<f32> {
    let r = |x: f64| if f32_accum { x as f32 as f64 } else { x };
    let d = radius * 2 + 1;
    let rad = radius as isize;
    let mirror = |i: isize, len: usize| {
        if i >= 0 && (i as usize) < len {
            i as usize
        } else {
            crate::metric::reflect_index(i.unsigned_abs(), len)
        }
    };
    let moment = |x: usize, y: usize| {
        let a = src[y * width + x] as f64;
        let b = dst[y * width + x] as f64;
        let fourth = if direct_error {
            let e = r(a - b);
            r(e * e)
        } else {
            r(a * b)
        };
        [a, b, r(r(a * a) + r(b * b)), fourth]
    };

    /// One van Herk pass over `n` outputs whose window is `padded[i ..= i+d-1]`.
    fn van_herk(padded: &[f64], n: usize, d: usize, r: &dyn Fn(f64) -> f64) -> Vec<f64> {
        let len = padded.len();
        let mut prefix = vec![0.0f64; len];
        let mut suffix = vec![0.0f64; len];
        let mut i = 0;
        while i < len {
            let end = (i + d).min(len);
            // prefix: left-to-right within the tile
            let mut acc = 0.0;
            for j in i..end {
                acc = r(acc + padded[j]);
                prefix[j] = acc;
            }
            // suffix: right-to-left within the tile
            acc = 0.0;
            for j in (i..end).rev() {
                acc = r(acc + padded[j]);
                suffix[j] = acc;
            }
            i = end;
        }
        (0..n)
            .map(|x| {
                let last = x + d - 1;
                if x % d == 0 {
                    suffix[x]
                } else {
                    r(suffix[x] + prefix[last])
                }
            })
            .collect()
    }

    // Horizontal: one padded row per moment, reflect-101 at the plane edge.
    let mut hrow = vec![0.0f64; width * height * 4];
    let mut padded = vec![0.0f64; width + 2 * radius];
    for y in 0..height {
        for k in 0..4 {
            for (i, cell) in padded.iter_mut().enumerate() {
                let x = mirror(i as isize - rad, width);
                *cell = moment(x, y)[k];
            }
            let out = van_herk(&padded, width, d, &r);
            for (x, v) in out.into_iter().enumerate() {
                hrow[(y * width + x) * 4 + k] = v;
            }
        }
    }

    // Vertical: same decomposition down each column of the row sums.
    let mut vsum = vec![0.0f64; width * height * 4];
    let mut col = vec![0.0f64; height + 2 * radius];
    for x in 0..width {
        for k in 0..4 {
            for (i, cell) in col.iter_mut().enumerate() {
                let y = mirror(i as isize - rad, height);
                *cell = hrow[(y * width + x) * 4 + k];
            }
            let out = van_herk(&col, height, d, &r);
            for (y, v) in out.into_iter().enumerate() {
                vsum[(y * width + x) * 4 + k] = v;
            }
        }
    }

    let inv_n = r(1.0 / (d as f64 * d as f64));
    let mut out = vec![0.0f32; width * height];
    for (i, value) in out.iter_mut().enumerate() {
        let m = &vsum[i * 4..i * 4 + 4];
        let a = r(m[0] * inv_n);
        let b = r(m[1] * inv_n);
        let mean_error2 = r(r(a - b) * r(a - b));
        let variance_sum = r(r(r(m[2] * inv_n) - r(a * a)) - r(b * b)).max(0.0);
        let luma_loss = match form {
            SsimLumaForm::Ssim2Legacy => mean_error2,
            SsimLumaForm::Clamp => mean_error2.min(1.0),
            SsimLumaForm::Lorentz => r(mean_error2 / r(1.0 + mean_error2)),
            SsimLumaForm::SsimLumaC1 => {
                r(mean_error2 / r(r(r(a * a) + r(b * b)) + C_SSIM_LUMA as f64))
            }
        };
        *value = if direct_error {
            let error_variance = r(r(m[3] * inv_n) - mean_error2).max(0.0);
            r(luma_loss + r(r(1.0 - luma_loss) * r(error_variance / r(variance_sum + C2 as f64))))
                .max(0.0) as f32
        } else {
            let cov = r(r(m[3] * inv_n) - r(a * b));
            r(1.0
                - r(r(1.0 - luma_loss)
                    * r(r(r(2.0 * cov) + C2 as f64) / r(variance_sum + C2 as f64))))
            .max(0.0) as f32
        };
    }
    out
}

/// Scratch for [`stable_ssim_plane_tiled`]. Still O(width x radius): two
/// tiles of raw row sums plus a derived suffix and prefix tile, all
/// `diameter` rows tall, independent of image height.
#[derive(Default)]
pub(crate) struct TiledSsimScratch {
    /// Raw H-summed rows for the current and next row-tile, rolling.
    hraw: Vec<f64>,
    /// Backward cumulative over the current row-tile.
    suf: Vec<f64>,
    /// Forward cumulative over the next row-tile.
    pre: Vec<f64>,
    /// The horizontal pass's three live tiles (previous suffix, current
    /// prefix, current suffix) — about 1 KB, so the decomposition stays in L1
    /// and the padded row is never materialised.
    hpad: Vec<f64>,
}

/// Form the SSIM dissimilarity with the moment window computed by a two-level
/// (van Herk / Gil-Werman) decomposition instead of a sliding recurrence.
///
/// Same output contract as [`stable_ssim_plane`] and the same four moments,
/// but every window sum reads ONLY its own samples — see
/// `benchmarks/stable_ssim_kernel_2026-09-08.md`. Locality is therefore a
/// property of the traversal rather than of the accumulator width, and the
/// serial dependency shrinks from the row/column length to `diameter`.
///
/// MEASURED and NOT SELECTED: exactly local at any precision, but ~1.8x the
/// sliding kernel's time. Kept because it is the only construction that makes
/// locality precision-independent, and because that is what an f32 variant
/// would need. See `benchmarks/stable_ssim_kernel_2026-09-08.md`.
#[cfg_attr(not(test), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
#[autoversion]
pub(crate) fn stable_ssim_plane_tiled(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    radius: usize,
    form: SsimLumaForm,
    out: &mut [f32],
    scratch: &mut TiledSsimScratch,
) {
    assert!(width > 0 && height > 0);
    let n = width.checked_mul(height).expect("plane extent");
    assert!(src.len() >= n && dst.len() >= n && out.len() >= n);
    let d = radius * 2 + 1;
    let rad = radius as isize;
    let padded = width + 2 * radius;
    let row_stride = width * 4;

    // Three tiles live at once (previous suffix, current prefix, current
    // suffix) — about 1 KB, not the ~200 KB three row-sized arrays cost.
    scratch.hpad.resize(4 * d * 4, 0.0);
    scratch.hraw.resize(2 * d * row_stride, 0.0);
    scratch.suf.resize(d * row_stride, 0.0);
    scratch.pre.resize(d * row_stride, 0.0);

    let mirror = |i: isize, len: usize| {
        if i >= 0 && (i as usize) < len {
            i as usize
        } else {
            crate::metric::reflect_index(i.unsigned_abs(), len)
        }
    };

    // One horizontal van Herk pass, with TILE-LOCAL storage.
    //
    // Two earlier shapes were measured and rejected. The first materialised
    // prefix and suffix over the whole padded row, four separate traversals
    // (one per moment), with `x % d` — a real division, d = 11 — in the inner
    // loop: 3.44x the sliding recurrence. Making it tile-local in access but
    // still row-sized in storage got to 1.88x. The row-sized arrays are the
    // remaining cost: three of them at 4 moments is ~200 KB per row at
    // width 2048, which does not fit L1.
    //
    // Lagging the combine by one tile means only THREE TILES are ever live —
    // the previous tile's suffix, and the current tile's prefix and suffix.
    // That is `3 * d * 4` f64, about 1 KB, so the whole decomposition happens
    // in L1 and the padded row is never materialised at all.
    let moments = |i: usize, y: usize| {
        // Interior indices need no mirroring; only the first and last `radius`
        // padded positions do, so the branch leaves the hot path.
        let x = if i >= radius && i < radius + width {
            i - radius
        } else {
            mirror(i as isize - rad, width)
        };
        let a = src[y * width + x] as f64;
        let b = dst[y * width + x] as f64;
        let e = a - b;
        [a, b, a * a + b * b, e * e]
    };
    let horizontal = |y: usize, dest: &mut [f64], tiles_buf: &mut [f64]| {
        let (mom, rest) = tiles_buf.split_at_mut(d * 4);
        let (pre, rest) = rest.split_at_mut(d * 4);
        let (suf_a, suf_b) = rest.split_at_mut(d * 4);
        let mut prev_is_a = false;
        let mut t = 0;
        while t < padded {
            let end = (t + d).min(padded);
            let (suf_cur, suf_prev) = if prev_is_a {
                (&mut *suf_b, &*suf_a)
            } else {
                (&mut *suf_a, &*suf_b)
            };
            // Moments ONCE per position into the tile buffer. Computing them
            // separately in the prefix and suffix loops was measured slightly
            // slower than materialising the whole padded row: it doubles both
            // the source loads and the moment arithmetic.
            for j in t..end {
                let q = moments(j, y);
                mom[(j - t) * 4..(j - t) * 4 + 4].copy_from_slice(&q);
            }
            let n_t = end - t;
            // Fixed-size chunks so the four moments are one register quad per
            // step rather than four bounds-checked indexings.
            let (mq, _) = mom[..n_t * 4].as_chunks::<4>();
            let (pq, _) = pre[..n_t * 4].as_chunks_mut::<4>();
            let mut acc = [0.0f64; 4];
            for (q, o) in mq.iter().zip(pq.iter_mut()) {
                for k in 0..4 {
                    acc[k] += q[k];
                }
                o.copy_from_slice(&acc);
            }
            let (sq, _) = suf_cur[..n_t * 4].as_chunks_mut::<4>();
            acc = [0.0; 4];
            for (q, o) in mq.iter().zip(sq.iter_mut()).rev() {
                for k in 0..4 {
                    acc[k] += q[k];
                }
                o.copy_from_slice(&acc);
            }
            if t > 0 {
                combine(t - d, width, d, dest, pre, suf_prev);
            }
            prev_is_a = !prev_is_a;
            t = end;
        }
        // The final tile contributes only its aligned position: for x past a
        // tile start, `x + d - 1` would leave the padded row, and
        // `x < width = padded - (d - 1)` forbids that.
        let last = padded
            - if padded.is_multiple_of(d) {
                d
            } else {
                padded % d
            };
        let suf_last = if prev_is_a { &*suf_a } else { &*suf_b };
        if last < width {
            dest[last * 4..last * 4 + 4].copy_from_slice(&suf_last[0..4]);
        }
    };

    // One tile's worth of `out[x] = suf[x] + pre[x + d - 1]`, with the
    // tile-aligned position (where the window IS the tile) hoisted out.
    // Both operands are TILE-LOCAL: `suf` is the previous tile's suffix and
    // `pre` this tile's prefix, so the indices are offsets within a tile.
    // A nested `fn` rather than a closure so it can be used above its
    // definition and cannot capture anything by accident.
    fn combine(start: usize, width: usize, d: usize, dest: &mut [f64], pre: &[f64], suf: &[f64]) {
        if start >= width {
            return;
        }
        dest[start * 4..start * 4 + 4].copy_from_slice(&suf[0..4]);
        for x in (start + 1)..(start + d).min(width) {
            let i = x - start;
            let sq = &suf[i * 4..i * 4 + 4];
            let pq = &pre[(i - 1) * 4..(i - 1) * 4 + 4];
            let o = &mut dest[x * 4..x * 4 + 4];
            for k in 0..4 {
                o[k] = sq[k] + pq[k];
            }
        }
    }

    let inv_n = 1.0 / (d as f64 * d as f64);
    let tiles = height.div_ceil(d);
    // Padded row index p in 0..height + 2*radius; output row y reads p in
    // y..y+d-1, so the row-tile grid is over that padded space.

    // Split the scratch ONCE so the row builder can borrow its working arrays
    // while writing into the tile buffers.
    let TiledSsimScratch {
        hraw,
        suf,
        pre,
        hpad,
    } = scratch;

    let fill = |tile: usize, into: &mut [f64], hpad: &mut [f64]| {
        for i in 0..d {
            let p = tile * d + i;
            let y = mirror(p as isize - rad, height);
            let dest = &mut into[i * row_stride..(i + 1) * row_stride];
            horizontal(y, dest, hpad);
        }
    };

    let (raw_a, raw_b) = hraw.split_at_mut(d * row_stride);
    fill(0, raw_a, hpad);
    let mut cur_is_a = true;
    for t in 0..tiles {
        let (cur, next) = if cur_is_a {
            (&*raw_a, &mut *raw_b)
        } else {
            (&*raw_b, &mut *raw_a)
        };
        fill(t + 1, next, hpad);
        // suffix over the current tile, backwards
        for i in (0..d).rev() {
            let (dst_i, src_i) = (i * row_stride, (i + 1) * row_stride);
            for j in 0..row_stride {
                suf[dst_i + j] = cur[dst_i + j] + if i + 1 < d { suf[src_i + j] } else { 0.0 };
            }
        }
        // prefix over the next tile, forwards
        for i in 0..d {
            // `prev_i` is only read when `i > 0`; saturate so the debug-build
            // overflow check does not fire on the `i == 0` row (CI runs debug).
            let (dst_i, prev_i) = (i * row_stride, i.saturating_sub(1) * row_stride);
            for j in 0..row_stride {
                pre[dst_i + j] = next[dst_i + j] + if i > 0 { pre[prev_i + j] } else { 0.0 };
            }
        }
        for i in 0..d {
            let y = t * d + i;
            if y >= height {
                break;
            }
            let row = &mut out[y * width..(y + 1) * width];
            for (x, value) in row.iter_mut().enumerate() {
                let s = &suf[i * row_stride + x * 4..i * row_stride + x * 4 + 4];
                let m: [f64; 4] = if i == 0 {
                    [s[0], s[1], s[2], s[3]]
                } else {
                    let q = &pre[(i - 1) * row_stride + x * 4..(i - 1) * row_stride + x * 4 + 4];
                    [s[0] + q[0], s[1] + q[1], s[2] + q[2], s[3] + q[3]]
                };
                *value = finalize(&m, inv_n, form);
            }
        }
        cur_is_a = !cur_is_a;
    }
}

/// The per-pixel expression shared by the moment kernels: identical algebra to
/// [`stable_ssim_plane`]'s tail, factored out so the two traversals provably
/// differ only in HOW the window sums are formed.
#[inline(always)]
fn finalize(m: &[f64; 4], inv_n: f64, form: SsimLumaForm) -> f32 {
    let a = m[0] * inv_n;
    let b = m[1] * inv_n;
    let mean_error2 = (a - b) * (a - b);
    let error_variance = (m[3] * inv_n - mean_error2).max(0.0);
    let variance_sum = (m[2] * inv_n - a * a - b * b).max(0.0);
    let luma_loss = match form {
        SsimLumaForm::Ssim2Legacy => mean_error2,
        SsimLumaForm::Clamp => mean_error2.min(1.0),
        SsimLumaForm::Lorentz => mean_error2 / (1.0 + mean_error2),
        SsimLumaForm::SsimLumaC1 => mean_error2 / (a * a + b * b + C_SSIM_LUMA as f64),
    };
    (luma_loss + (1.0 - luma_loss) * error_variance / (variance_sum + C2 as f64)).max(0.0) as f32
}

/// Independent direct-window f64 reference, shared by the numerical
/// instrument and kernel tests. Centered moments deliberately avoid the
/// candidate's running-sum algorithm; raw algebra is a separate control.
/// Returns dissimilarities, maximum algebra disagreement, and direct local
/// means for the same instrument's edge-precision checks.
#[cfg(test)]
pub(crate) fn precision_reference(
    src: &[f32],
    dst: &[f32],
    w: usize,
    h: usize,
    radius: usize,
    form: SsimLumaForm,
) -> (Vec<f64>, f64, Vec<[f64; 2]>) {
    let mut result = Vec::with_capacity(w * h);
    let mut agreement = 0.0f64;
    let mut means = Vec::with_capacity(w * h);
    let mut samples = Vec::new();
    let radius = radius as isize;
    for y in 0..h {
        for x in 0..w {
            samples.clear();
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let xx = crate::metric::reflect_index((x as isize + dx).unsigned_abs(), w);
                    let yy = crate::metric::reflect_index((y as isize + dy).unsigned_abs(), h);
                    samples.push((src[yy * w + xx] as f64, dst[yy * w + xx] as f64));
                }
            }
            let count = samples.len() as f64;
            let m1 = samples.iter().map(|p| p.0).sum::<f64>() / count;
            let m2 = samples.iter().map(|p| p.1).sum::<f64>() / count;
            means.push([m1, m2]);
            let md = samples.iter().map(|p| p.0 - p.1).sum::<f64>() / count;
            let v1 = samples.iter().map(|p| (p.0 - m1).powi(2)).sum::<f64>() / count;
            let v2 = samples.iter().map(|p| (p.1 - m2).powi(2)).sum::<f64>() / count;
            let ve = samples
                .iter()
                .map(|p| (p.0 - p.1 - md).powi(2))
                .sum::<f64>()
                / count;
            let c2 = C2 as f64;
            let loss = match form {
                SsimLumaForm::Ssim2Legacy => md * md,
                SsimLumaForm::Clamp => (md * md).min(1.0),
                SsimLumaForm::Lorentz => md * md / (1.0 + md * md),
                SsimLumaForm::SsimLumaC1 => md * md / (m1 * m1 + m2 * m2 + C_SSIM_LUMA as f64),
            };
            let sd = loss + (1.0 - loss) * ve / (v1 + v2 + c2);
            let ssq = samples.iter().map(|p| p.0 * p.0 + p.1 * p.1).sum::<f64>() / count;
            let s12 = samples.iter().map(|p| p.0 * p.1).sum::<f64>() / count;
            let raw =
                1.0 - (1.0 - loss) * (2.0 * (s12 - m1 * m2) + c2) / (ssq - m1 * m1 - m2 * m2 + c2);
            agreement = agreement.max((sd - raw).abs());
            result.push(sd.max(0.0));
        }
    }
    (result, agreement, means)
}

/// SSIM structure/contrast stabiliser — ssimulacra2's value, and the ONE
/// declaration of it.
///
/// `0.0009 = 0.03^2 = (K2 * L)^2` at the textbook `K2 = 0.03`, `L = 1`. Before
/// this module it was declared three times (`fused.rs`, `simd_ops.rs`,
/// `feature_v2.rs`'s `C2_V2`), which is how [`C_SSIM_LUMA`] came to be missing
/// rather than merely omitted: there was no single place for it to be missing
/// FROM.
pub(crate) const C2: f32 = 0.0009;

/// SSIM luminance stabiliser — **derived, not chosen**, and used only by the
/// bounded forms.
///
/// Two independent derivations land on the same value:
///
/// 1. **From the family.** Every `bounded_sim(a, b, c) = (2ab+c)/(a^2+b^2+c)`
///    regularizer in this crate is `1e-4` — `C_EDGE`, `C_GMS`, `C_CONTRAST`,
///    `C_BV`. `bounded_sim` **is** the standard SSIM luminance term, already
///    present as a named, documented, shared primitive
///    (`feature_v2.rs`, "Bounded `(0, 1]`"). [`SsimLumaForm::SsimLumaC1`] is
///    that owner's form, applied to the one place in the crate that hand-rolls
///    an unbounded substitute for it.
/// 2. **From the constant already here.** [`C2`] is `(K2 * L)^2` at
///    `K2 = 0.03`, `L = 1`. The matching `C1 = (K1 * L)^2` at the textbook
///    `K1 = 0.01` is `1e-4`.
///
/// It regularises only as `mu1^2 + mu2^2 -> 0`; at photographic magnitudes it
/// is negligible, which is why the exact value is not load-bearing and the
/// agreement of the two derivations is what makes it defensible.
///
/// CORRECTION to `benchmarks/ssim_moment_explosion_2026-07-16.md` §7a, which
/// evaluated "C1(0.01)" — 100x this value. Its ordering conclusions are
/// unaffected; its photographic row moves from +0.0003 to +0.00004 here.
pub(crate) const C_SSIM_LUMA: f32 = 1e-4;

/// Which luminance term the per-pixel dissimilarity uses.
///
/// The revision axis is owned by [`crate::feature_defs`]; this enum is the
/// arithmetic that a revision SELECTS. Three bounded arms exist because the
/// choice between them is a measurement (`docs/PLAN_FEATURE_REV2_2026-09-05.md`
/// R6), not an argument — see each variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum SsimLumaForm {
    /// `1 - D^2`. ssimulacra2's form, shipped, **unbounded below** (F4).
    #[default]
    Ssim2Legacy,
    /// `(2*mu1*mu2 + C1) / (mu1^2 + mu2^2 + C1)` — the standard SSIM
    /// luminance term, i.e. `bounded_sim(mu1, mu2, C_SSIM_LUMA)`.
    ///
    /// Bounded `(0, 1]`, so `d_raw` lands in `[0, 2]` and the `.max(0)` floor
    /// the call sites apply becomes provably redundant. Preserves severity
    /// ORDER where a clamp would saturate. Its cost is that it re-introduces
    /// the mean-dependent (Weber) normalisation `lib.rs` says was dropped on
    /// purpose for a perceptually-uniform space.
    SsimLumaC1,
    /// `1 / (1 + D^2)`, i.e. `1 - saturate(D^2, 1)` in the crate's own
    /// `saturate(x, c) = x/(x+c)` idiom.
    ///
    /// Bounded `(0, 1]`, **no Weber normalisation** (so it keeps the
    /// perceptual-uniformity intent), and `c = 1` is the unique scale at which
    /// it agrees with [`Self::Ssim2Legacy`] to first order in `D^2` — so it
    /// introduces no new constant. Compresses the extreme tail harder than
    /// `SsimLumaC1`.
    Lorentz,
    /// `max(0, 1 - D^2)`. **Exact** for `D^2 <= 1` — the overwhelming majority
    /// of pixels — and flat above it, so it bounds without touching the
    /// photographic regime at all, at the cost of flattening rank among the
    /// worst pixels.
    Clamp,
}

impl SsimLumaForm {
    /// The form a registered revision selects.
    pub(crate) const fn for_revision(rev: FormulaRevision) -> Self {
        match rev {
            FormulaRevision::Rev1 => Self::Ssim2Legacy,
            FormulaRevision::Rev2
            | FormulaRevision::Rev3
            | FormulaRevision::Rev4
            | FormulaRevision::Rev5
            | FormulaRevision::Rev6
            | FormulaRevision::Rev7 => Self::REV2_LUMA,
        }
    }

    /// The bounded arm revision 2 ships — **`Clamp`, DECIDED BY MEASUREMENT**
    /// (`benchmarks/f4_arm_decision_2026-09-05.md`, R6).
    ///
    /// Named once so the R6 probe and the kernels cannot disagree about which
    /// arm "rev2" means.
    ///
    /// # Why `Clamp` and not [`Self::SsimLumaC1`]
    ///
    /// `docs/PLAN_FEATURE_REV2_2026-09-05.md` §1.4 registered a prior — *"if
    /// the probe cannot separate them, arm A (`SsimLumaC1`) ships"* — because
    /// it reuses `feature_v2`'s own `bounded_sim` owner and preserves severity
    /// ORDER among pathological pixels. **The probe DID separate them**, so
    /// the prior never fired. Four arms were extracted from ONE binary over
    /// 217,756 rows (the seven human eval corpora + the full 196,086-row
    /// safesyn training leg), fitted through the shipped Profile-D recipe at
    /// three slices x two solvers, and graded on rank, dial and cell deltas:
    ///
    /// * **F4's pathology occurs on NONE of it.** `Clamp` — which differs from
    ///   the shipped form only where `(mu1-mu2)^2 > 1` — moves **0 cells**, and
    ///   no slot anywhere reaches `|f| > 2` against the 5,814,302 on record
    ///   (which belongs to the bigcodec sweep, a population with no local
    ///   pixels).
    /// * **`Clamp` is therefore bit-identical to revision 1 on every row R6
    ///   fits or scores** — features, Gram, solve, spline and ZNPR bytes, all
    ///   six bakes sha-for-sha — so its rank delta is exactly 0 with a
    ///   degenerate CI.
    /// * `SsimLumaC1` moves **29.4 M** healthy cells (worst |delta| 0.771) and
    ///   `Lorentz` **24.0 M** (worst 0.0901), against a pre-registered 1e-4
    ///   bar, to buy at most `+0.0025` CID22 in one of six variants. Both fail
    ///   the healthy-cell gate; neither wins a rank majority in more than one
    ///   variant.
    ///
    /// So `Clamp` is the unique arm that is ONLY a fix: it changes the metric
    /// exactly where the metric was unbounded and nowhere else, which is what
    /// makes a rev2 flip cheap for every table whose content resembles those
    /// corpora.
    ///
    /// **Its known cost, recorded rather than glossed:** above `D^2 = 1` every
    /// pixel gets the same `num_m = 0`, so tail ORDER is flat there — the exact
    /// property §1.4's prior was protecting. `d` stays bounded in `[0, 2]`
    /// regardless. If a future population makes that order load-bearing,
    /// [`Self::Lorentz`] is the registered successor (bounded, no Weber term,
    /// monotone in `D^2`, and 4-6 orders of magnitude closer to revision 1 on
    /// healthy content than `SsimLumaC1`) — not `SsimLumaC1`.
    pub(crate) const REV2_LUMA: Self = Self::Clamp;

    /// Whether `d_raw` is bounded below by 0 by construction, making the
    /// call sites' `.max(0)` floor redundant rather than load-bearing.
    ///
    /// True for every arm whose `num_m` is bounded in `[0, 1]`: with
    /// `num_s/denom_s` in `[-1, 1]`, `d_raw = 1 - num_m*(num_s/denom_s)` lands
    /// in `[0, 2]`.
    pub(crate) const fn bounds_dissim(self) -> bool {
        !matches!(self, Self::Ssim2Legacy)
    }
}

/// **The revision this build ships.**
///
/// Changing this line is the era flip. Everything else — the registry entries,
/// the gates, the recalculation — hangs off it, and
/// `docs/PLAN_FEATURE_REV2_2026-09-05.md` R1 is the control that proves the
/// machinery around it is inert while it still reads `Rev1`.
pub(crate) const SHIPPED_REVISION: FormulaRevision = FormulaRevision::Rev1;

/// **The revision switch — one owner, read once per process.**
///
/// Modelled on [`crate::feature_v2`]'s `era2_dense_enabled`, deliberately and
/// for its stated reason: *"a switch that some call sites honour and others do
/// not is the same defect"*. Every SSIM kernel reads this ONCE, above its
/// loop, and passes the result down; no call site chooses its own form.
///
/// `ZENSIM_FORMULA_REV=1`, `=2` and `=3` pin revisions 1, 2 and 3, so a
/// research extraction can reproduce any era's semantics from one binary
/// (phase 3's G3.2). Anything else — including unset — is
/// [`SHIPPED_REVISION`]. The accepted values are the SAME BYTE LENGTH on
/// purpose: this repo has measured an environment block's size shifting a
/// binary's layout by ~10 % at 2304²
/// (`benchmarks/era2_perf_break_2026-08-31.md` §22.5), so an A/B that varies
/// the value must not vary the length.
pub(crate) fn active_revision() -> FormulaRevision {
    use std::sync::OnceLock;
    static REV: OnceLock<FormulaRevision> = OnceLock::new();
    *REV.get_or_init(|| match std::env::var("ZENSIM_FORMULA_REV").as_deref() {
        Ok("1") => FormulaRevision::Rev1,
        Ok("2") => FormulaRevision::Rev2,
        Ok("3") => FormulaRevision::Rev3,
        Ok("4") => FormulaRevision::Rev4,
        Ok("5") => FormulaRevision::Rev5,
        Ok("6") => FormulaRevision::Rev6,
        Ok("7") => FormulaRevision::Rev7,
        _ => SHIPPED_REVISION,
    })
}

/// Explicit research cross-revision replay keeps using the process kernels.
/// Normal model-bound requests never consult mutable thread/global overrides.
pub(crate) fn effective_revision(revision: FormulaRevision) -> FormulaRevision {
    #[cfg(feature = "cross-revision-diagnostic")]
    if cross_revision_diagnostic() {
        return active_revision();
    }
    revision
}

#[cfg(test)]
/// Run a SIMD-tier-toggling `#[test]` body alone in a child process.
///
/// `archmage::testing::for_each_token_permutation` disables tiers
/// process-wide. In the shared test process, every other test that compares
/// two extractions bit for bit without `lock_token_testing()` can then see a
/// tier change between, or inside, its calls (CI run 38104576912; the
/// class is in zensim's Known Bugs history). A toggler therefore owns its
/// own process. Returns `true` in that child (run the body); otherwise it
/// re-executes THIS test binary for exactly this one test, with
/// `ZENSIM_TIER_TOGGLER_CHILD` set to `test_path` and, when given,
/// `ZENSIM_FORMULA_REV` set to `rev`, and fails if the child fails or never
/// prints `sentinel`. Unlike [`run_at_revision`] it re-executes even when
/// the process is already at `rev`, so a revision child running many tests
/// (`rerun_tests_at_revision`) never toggles beside them.
pub(crate) fn run_in_own_process(test_path: &str, sentinel: &str, rev: Option<&str>) -> bool {
    const CHILD: &str = "ZENSIM_TIER_TOGGLER_CHILD";
    if std::env::var(CHILD).as_deref() == Ok(test_path) {
        return true;
    }
    let exe = std::env::current_exe().expect("test binary path");
    let mut cmd = std::process::Command::new(exe);
    cmd.args([test_path, "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD, test_path);
    if let Some(rev) = rev {
        cmd.env("ZENSIM_FORMULA_REV", rev);
    }
    let out = cmd.output().expect("re-exec the test binary");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{test_path} failed in its own process\n--- stdout ---\n{stdout}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains(sentinel),
        "{test_path} exited 0 in its own process but never reached its body \
         (sentinel {sentinel:?} absent)\n{stdout}"
    );
    false
}

#[cfg(test)]
/// Run one `#[test]` body under an explicit `ZENSIM_FORMULA_REV`.
///
/// `ssim_form::active_revision` is a `OnceLock`, so a revision cannot be
/// changed inside a running process and a revision-specific control has
/// to own its own process. Returns `true` when this process is ALREADY at
/// `rev` (run the body); otherwise it re-executes THIS test binary with
/// the variable set, running exactly this one test, and fails if the
/// child fails.
///
/// This is not a skip: the assertions always execute, once, in the
/// process that can see them. The parent proves the child really ran the
/// body by requiring `sentinel` on its stdout — without that, a filter
/// that matched nothing would exit 0 and the control would pass
/// vacuously.
pub(crate) fn run_at_revision(rev: &str, test_path: &str, sentinel: &str) -> bool {
    if std::env::var("ZENSIM_FORMULA_REV").as_deref() == Ok(rev) {
        return true;
    }
    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args([test_path, "--exact", "--nocapture", "--test-threads=1"])
        .env("ZENSIM_FORMULA_REV", rev)
        .output()
        .expect("re-exec the test binary");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{test_path} failed at ZENSIM_FORMULA_REV={rev}\n--- stdout ---\n{stdout}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains(sentinel),
        "{test_path} exited 0 at ZENSIM_FORMULA_REV={rev} but never reached its body \
         (sentinel {sentinel:?} absent) — the control did not run\n{stdout}"
    );
    false
}

#[cfg(test)]
/// Re-run an EXISTING set of tests under a different `ZENSIM_FORMULA_REV`.
///
/// The companion of [`run_at_revision`] for controls whose body already
/// exists and is already exercised at the shipped revision: rather than
/// copying a 200-line parity test, run the same one again in a process
/// pinned to `rev`. `expect` is the number of tests the filter must match —
/// a filter that silently stops matching would otherwise turn the control
/// into a no-op that still exits 0.
pub(crate) fn rerun_tests_at_revision(rev: &str, filter: &str, expect: usize) {
    if std::env::var("ZENSIM_FORMULA_REV").as_deref() == Ok(rev) {
        return; // the child is running the real tests; nothing to re-spawn
    }
    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args([filter, "--nocapture", "--test-threads=1"])
        .env("ZENSIM_FORMULA_REV", rev)
        .output()
        .expect("re-exec the test binary");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "`{filter}` failed at ZENSIM_FORMULA_REV={rev}\n--- stdout ---\n{stdout}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let ran = stdout
        .lines()
        .find_map(|l| {
            l.strip_prefix("test result: ok. ")?
                .split(' ')
                .next()?
                .parse::<usize>()
                .ok()
        })
        .unwrap_or(0);
    assert_eq!(
        ran, expect,
        "`{filter}` matched {ran} tests at ZENSIM_FORMULA_REV={rev}, expected {expect} — \
         the control is not running what it claims\n{stdout}"
    );
}

/// Refuse, EXPLICITLY, any route the active revision does not serve.
///
/// [`FormulaRevision::Rev3`] replaces the v1 SSIM signal with
/// [`stable_ssim_plane`], whose spatial support is exactly ONE reflect-101
/// box of `blur_radius`. `blur_passes != 1` selects
/// `streaming::process_strip_channel`'s separate blur+reduce fallback, which
/// no shipped profile reaches and whose `passes * radius` halo does not
/// describe a single box. Rev3 therefore does not serve it.
///
/// This is the ONLY correct third option. Serving it from the legacy moments
/// would silently mix two arithmetic eras inside one feature vector, and
/// panicking would put an abort on a public path; so every fallible entry
/// that turns a profile into a [`crate::metric::ZensimConfig`] calls this and
/// returns the error to the caller.
///
/// Callers that only inspect a config (pool-mode derivation, fold planning)
/// do not call it: they run no pixel kernel, and the entry that eventually
/// does has already refused.
///
/// REV4SERVE: Rev4 crossings are refused via [`refuse_rev4_mix`]: a
/// request at one revision served from a process pinned to another would
/// compute the request's label with the process's gates. A pure-Rev4
/// process (`ZENSIM_FORMULA_REV=4`) serves Rev4 computations — every leaf
/// on the served walk is canonical there.
pub(crate) fn check_route(config: &crate::metric::ZensimConfig) -> Result<(), crate::ZensimError> {
    refuse_rev4_mix(config.revision())?;
    if config.revision() >= FormulaRevision::Rev3 && config.blur_passes != 1 {
        return Err(crate::ZensimError::ModelForwardFailed {
            reason: "formula revision 3 serves blur_passes == 1 only (its stable SSIM moments are one reflect-101 box); use revision 1 or 2 for multi-pass blur profiles",
        });
    }
    Ok(())
}

/// The reason [`refuse_rev4_mix`] gives.
#[cfg_attr(not(feature = "feature-regime-v2"), allow(dead_code))] // the v2 walk's guard
pub(crate) const REV4_MIX: &str = "formula revisions 4 and 5 cannot be mixed with another revision in one process: other formula gates follow ZENSIM_FORMULA_REV, so the request and the process must carry the same revision when either is revision 4 or later";

/// **Refuse a computation whose revision disagrees with the process revision
/// when either one is Rev4** (featcanon D1).
///
/// Canonical arithmetic follows the computation's own revision (every
/// canonical leaf derives it from the revision it receives), but other
/// formula gates — the dense kernel's `direct` flag, `RootForm`, `PowForm`,
/// `HfGainForm` — still read the process switch. So a Rev4 request in a Rev3
/// process would compute Rev3 semantics, and a Rev3 request in a Rev4 process
/// would mix Rev4 gates into it. Both are refused rather than labelled with a
/// revision they did not compute. Mixing among Rev1–Rev3 is governed by the
/// pre-existing per-route checks and is unchanged.
#[cfg_attr(not(feature = "feature-regime-v2"), allow(dead_code))] // the v2 walk's guard
pub(crate) fn refuse_rev4_mix(requested: FormulaRevision) -> Result<(), crate::ZensimError> {
    let process = active_revision();
    if requested != process
        && (requested >= FormulaRevision::Rev4 || process >= FormulaRevision::Rev4)
    {
        return Err(crate::ZensimError::ModelLoadFailed { reason: REV4_MIX });
    }
    Ok(())
}

/// **Refuse the 256-row strips entries at Rev4** (REV4SERVE).
///
/// `compute_streaming_strips` / `compute_with_ref_streaming_strips` merge
/// per-strip `ScaleAccumulators` — an epsilon-equivalent but NOT
/// bit-identical summation tree relative to the fold walk's 128-row
/// kernel strips (their own doc: "within f64 machine epsilon"). At Rev4
/// the contract is bit-for-bit vs `research::extract`, so the strips
/// entries refuse rather than serve off-canon bits. Below Rev4 the
/// epsilon-equivalence contract is unchanged.
pub(crate) fn refuse_rev4_strips(requested: FormulaRevision) -> Result<(), crate::ZensimError> {
    if effective_revision(requested) >= FormulaRevision::Rev4 {
        return Err(crate::ZensimError::ModelLoadFailed {
            reason: "the 256-row strips walk merges per-strip accumulators — epsilon-equivalent, not bit-canonical: at formula revision 4 use the buffered or fold entries",
        });
    }
    Ok(())
}

/// Warn, once per process, that a pinned research revision is producing a
/// score from a built-in profile's revision-1 coefficients.
///
/// Silent when nothing is pinned, which is every shipping path — the check is
/// one `OnceLock` read and one comparison against a constant.
///
/// This is NOT a refusal. The same entry that scores also emits the features a
/// research extraction exists to collect, so refusing would break extraction to
/// protect a number the caller may not be reading. Making it visible is the
/// most that can be done here; the enforceable contract is `BakeScorer`'s,
/// where a bake declares its own revision.
#[inline]
pub(crate) fn warn_pinned_revision_scoring_once() {
    if active_revision() == SHIPPED_REVISION {
        return;
    }
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        eprintln!(
            "zensim: ZENSIM_FORMULA_REV pins {:?} pixels, but a built-in \
             profile's bake is revision-1 coefficients. Its FEATURES are the \
             pinned revision; its SCORE prices them with weights fit against \
             another extractor and is not a served score. Use BakeScorer with \
             a bake that declares its revision to get an enforced match.",
            active_revision()
        );
    });
}

/// Is the cross-revision DIAGNOSTIC bypass armed?
///
/// Two independent switches, both required: the `cross-revision-diagnostic`
/// cargo feature must be compiled in (so a product build does not contain the
/// bypass at all) and `ZENSIM_CROSS_REVISION_DIAGNOSTIC=1` must be set. See
/// that feature's declaration in `Cargo.toml` for what the resulting numbers
/// are and are not.
#[cfg(feature = "cross-revision-diagnostic")]
pub(crate) fn cross_revision_diagnostic() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ZENSIM_CROSS_REVISION_DIAGNOSTIC").as_deref() == Ok("1"))
}

/// Say so, once per process, on stderr. A cross-era number that reaches a
/// report without this line beside it in the log is not attributable.
#[cfg(feature = "cross-revision-diagnostic")]
pub(crate) fn warn_cross_revision_once(bake: FormulaRevision) {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        eprintln!(
            "zensim: CROSS-REVISION DIAGNOSTIC — serving a bake declared {bake:?} \
             with {:?} pixels ({:?} luminance form). The coefficients were fit \
             against a different extractor; these numbers are a measurement of \
             the extraction change, NOT a served score, and must not be \
             reported as model quality.",
            active_revision(),
            active_luma_form()
        );
    });
}

/// The luminance form the active revision selects.
///
/// Call this ONCE per kernel invocation, above the pixel loop — it reads a
/// `OnceLock`, which LLVM cannot hoist out of a loop for you.
#[inline]
pub(crate) fn active_luma_form() -> SsimLumaForm {
    use std::sync::OnceLock;
    static FORM: OnceLock<SsimLumaForm> = OnceLock::new();
    *FORM.get_or_init(|| {
        // `ZENSIM_SSIM_LUMA` is a MEASUREMENT override and nothing else. R6
        // has to fit the monotone linear class against all four luminance
        // arms on ONE set of pixels, from one binary, or the comparison is
        // confounded by a rebuild (this repo has measured a rebuild alone
        // moving a 2304^2 timing ~10 %). No shipping path sets it; the
        // revision is what a shipping path selects.
        match std::env::var("ZENSIM_SSIM_LUMA").as_deref() {
            Ok("ssim2") => SsimLumaForm::Ssim2Legacy,
            Ok("c1") => SsimLumaForm::SsimLumaC1,
            Ok("lorentz") => SsimLumaForm::Lorentz,
            Ok("clamp") => SsimLumaForm::Clamp,
            _ => SsimLumaForm::for_revision(active_revision()),
        }
    })
}

/// Hoisted splats + the selected form, built once per kernel invocation.
///
/// Bundling them keeps the per-site call short and keeps [`SsimLumaForm`]
/// loop-invariant, so the `match` is unswitched out of the inner loop after
/// inlining rather than evaluated per pixel.
#[derive(Clone, Copy)]
pub(crate) struct SsimSplats16<T: F32x16Backend + Copy> {
    form: SsimLumaForm,
    one: f32x16<T>,
    two: f32x16<T>,
    zero: f32x16<T>,
    c2: f32x16<T>,
    c1: f32x16<T>,
}

impl<T: F32x16Backend + Copy> SsimSplats16<T> {
    #[inline(always)]
    pub(crate) fn new(token: T, form: SsimLumaForm) -> Self {
        Self {
            form,
            one: f32x16::splat_t(token, 1.0),
            two: f32x16::splat_t(token, 2.0),
            zero: f32x16::zero_t(token),
            c2: f32x16::splat_t(token, C2),
            c1: f32x16::splat_t(token, C_SSIM_LUMA),
        }
    }

    /// `d_raw = 1 - num_m * num_s / denom_s`, BEFORE any weight multiply and
    /// before the call site's `.max(0)` floor — the two things that differ
    /// between the basic, masked and IW call sites.
    ///
    /// For [`SsimLumaForm::Ssim2Legacy`] the emitted operations are
    /// bit-identical to the hand-written form this replaced: same `mul_add`
    /// spellings, same order, same single divide.
    #[inline(always)]
    pub(crate) fn dissim(
        &self,
        m1: f32x16<T>,
        m2: f32x16<T>,
        ssq: f32x16<T>,
        s12: f32x16<T>,
    ) -> f32x16<T> {
        let num_s = self.two.mul_add((-m1).mul_add(m2, s12), self.c2);
        let denom_s = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)) + self.c2;
        // Every arm reduces to ONE divide by expressing the luminance term as
        // a (numerator, denominator) pair folded into the structure ratio.
        let (num_m, den_m) = match self.form {
            SsimLumaForm::Ssim2Legacy => {
                let mu_diff = m1 - m2;
                (mu_diff.mul_add(-mu_diff, self.one), self.one)
            }
            SsimLumaForm::SsimLumaC1 => (
                self.two.mul_add(m1 * m2, self.c1),
                m1.mul_add(m1, m2.mul_add(m2, self.c1)),
            ),
            SsimLumaForm::Lorentz => {
                let mu_diff = m1 - m2;
                (self.one, mu_diff.mul_add(mu_diff, self.one))
            }
            SsimLumaForm::Clamp => {
                let mu_diff = m1 - m2;
                (mu_diff.mul_add(-mu_diff, self.one).max(self.zero), self.one)
            }
        };
        // `den_m` is exactly 1.0 for the two arms that do not need it, and
        // `x * 1.0` is exact, so the legacy arm keeps its original rounding.
        self.one - (num_m * num_s) / (den_m * denom_s)
    }

    /// The DIRECT-ERROR form, for revision 3: the fourth moment is
    /// `Σ(a-b)²` (the H/V passes accumulate it in place of `Σab`), so the
    /// error variance is formed directly instead of recovered as
    /// `var1 + var2 - 2*cov`. Same algebra as the f64 reference `finalize`,
    /// in f32 with the crate's `mul_add` spellings. Bounded error, registered
    /// in `benchmarks/stable_ssim_kernel_2026-09-08.md`.
    #[cfg_attr(not(feature = "avx512"), allow(dead_code))] // 16-lane: only the v4x arcane bodies reach it
    #[inline(always)]
    pub(crate) fn direct(
        &self,
        m1: f32x16<T>,
        m2: f32x16<T>,
        ssq: f32x16<T>,
        err: f32x16<T>,
    ) -> f32x16<T> {
        let mu_diff = m1 - m2;
        let mean_error2 = mu_diff * mu_diff;
        let variance_sum = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)).max(self.zero);
        let error_variance = (err - mean_error2).max(self.zero);
        let luma_loss = match self.form {
            SsimLumaForm::Ssim2Legacy => mean_error2,
            SsimLumaForm::Clamp => mean_error2.min(self.one),
            SsimLumaForm::Lorentz => mean_error2 / (mean_error2 + self.one),
            SsimLumaForm::SsimLumaC1 => mean_error2 / m1.mul_add(m1, m2.mul_add(m2, self.c1)),
        };
        (self.one - luma_loss).mul_add(error_variance / (variance_sum + self.c2), luma_loss)
    }
}

/// 8-lane sibling of [`SsimSplats16`].
#[derive(Clone, Copy)]
pub(crate) struct SsimSplats8<T: F32x8Backend + Copy> {
    form: SsimLumaForm,
    one: GenericF32x8<T>,
    two: GenericF32x8<T>,
    zero: GenericF32x8<T>,
    c2: GenericF32x8<T>,
    c1: GenericF32x8<T>,
}

impl<T: F32x8Backend + Copy> SsimSplats8<T> {
    #[inline(always)]
    pub(crate) fn new(token: T, form: SsimLumaForm) -> Self {
        Self {
            form,
            one: GenericF32x8::splat_t(token, 1.0),
            two: GenericF32x8::splat_t(token, 2.0),
            zero: GenericF32x8::zero_t(token),
            c2: GenericF32x8::splat_t(token, C2),
            c1: GenericF32x8::splat_t(token, C_SSIM_LUMA),
        }
    }

    /// See [`SsimSplats16::dissim`].
    #[inline(always)]
    pub(crate) fn dissim(
        &self,
        m1: GenericF32x8<T>,
        m2: GenericF32x8<T>,
        ssq: GenericF32x8<T>,
        s12: GenericF32x8<T>,
    ) -> GenericF32x8<T> {
        let num_s = self.two.mul_add((-m1).mul_add(m2, s12), self.c2);
        let denom_s = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)) + self.c2;
        let (num_m, den_m) = match self.form {
            SsimLumaForm::Ssim2Legacy => {
                let mu_diff = m1 - m2;
                (mu_diff.mul_add(-mu_diff, self.one), self.one)
            }
            SsimLumaForm::SsimLumaC1 => (
                self.two.mul_add(m1 * m2, self.c1),
                m1.mul_add(m1, m2.mul_add(m2, self.c1)),
            ),
            SsimLumaForm::Lorentz => {
                let mu_diff = m1 - m2;
                (self.one, mu_diff.mul_add(mu_diff, self.one))
            }
            SsimLumaForm::Clamp => {
                let mu_diff = m1 - m2;
                (mu_diff.mul_add(-mu_diff, self.one).max(self.zero), self.one)
            }
        };
        self.one - (num_m * num_s) / (den_m * denom_s)
    }

    /// 8-lane sibling of [`SsimSplats16::direct`].
    #[inline(always)]
    pub(crate) fn direct(
        &self,
        m1: GenericF32x8<T>,
        m2: GenericF32x8<T>,
        ssq: GenericF32x8<T>,
        err: GenericF32x8<T>,
    ) -> GenericF32x8<T> {
        let mu_diff = m1 - m2;
        let mean_error2 = mu_diff * mu_diff;
        let variance_sum = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)).max(self.zero);
        let error_variance = (err - mean_error2).max(self.zero);
        let luma_loss = match self.form {
            SsimLumaForm::Ssim2Legacy => mean_error2,
            SsimLumaForm::Clamp => mean_error2.min(self.one),
            SsimLumaForm::Lorentz => mean_error2 / (mean_error2 + self.one),
            SsimLumaForm::SsimLumaC1 => mean_error2 / m1.mul_add(m1, m2.mul_add(m2, self.c1)),
        };
        (self.one - luma_loss).mul_add(error_variance / (variance_sum + self.c2), luma_loss)
    }
}

/// Scalar sibling — the `width % LANES` tails every vector kernel carries.
///
/// Same operations, same order, same `mul_add` spellings as the vector arms,
/// so a tail row and a vector row agree to the extent f32 FMA allows.
#[inline(always)]
pub(crate) fn ssim_dissim_raw_scalar(
    form: SsimLumaForm,
    m1: f32,
    m2: f32,
    ssq: f32,
    s12: f32,
) -> f32 {
    let num_s = 2.0f32.mul_add((-m1).mul_add(m2, s12), C2);
    let denom_s = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)) + C2;
    let (num_m, den_m) = match form {
        SsimLumaForm::Ssim2Legacy => {
            let mu_diff = m1 - m2;
            (mu_diff.mul_add(-mu_diff, 1.0f32), 1.0f32)
        }
        SsimLumaForm::SsimLumaC1 => (
            2.0f32.mul_add(m1 * m2, C_SSIM_LUMA),
            m1.mul_add(m1, m2.mul_add(m2, C_SSIM_LUMA)),
        ),
        SsimLumaForm::Lorentz => {
            let mu_diff = m1 - m2;
            (1.0f32, mu_diff.mul_add(mu_diff, 1.0f32))
        }
        SsimLumaForm::Clamp => {
            let mu_diff = m1 - m2;
            (mu_diff.mul_add(-mu_diff, 1.0f32).max(0.0f32), 1.0f32)
        }
    };
    let d = 1.0f32 - (num_m * num_s) / (den_m * denom_s);
    // The boundedness claim, checked rather than asserted in prose: a bounded
    // luminance term puts `d` in `[0, 2]`, which is what makes the call sites'
    // `.max(0)` floor redundant instead of load-bearing. The tolerance is for
    // f32 rounding at the `num_s/denom_s = 1` boundary, not for slack in the
    // claim. Debug-only: this is the innermost scalar tail.
    debug_assert!(
        !form.bounds_dissim() || (-1e-5..=2.0 + 1e-5).contains(&d),
        "bounded SSIM form {form:?} produced d = {d} outside [0, 2]"
    );
    d
}

/// One-call wrapper over [`SsimSplats16`] for the 16-lane kernels.
///
/// The splats are loop-invariant constants, so building them inside an
/// `#[inline(always)]` call is free — LLVM hoists them out of the loop
/// exactly as it did when they were hand-declared above it. The struct stays
/// public to this crate for any future site that wants to hoist explicitly.
#[inline(always)]
pub(crate) fn ssim_dissim16<T: F32x16Backend + Copy>(
    token: T,
    form: SsimLumaForm,
    m1: f32x16<T>,
    m2: f32x16<T>,
    ssq: f32x16<T>,
    s12: f32x16<T>,
) -> f32x16<T> {
    SsimSplats16::new(token, form).dissim(m1, m2, ssq, s12)
}

/// 8-lane sibling of [`ssim_dissim16`].
#[inline(always)]
pub(crate) fn ssim_dissim8<T: F32x8Backend + Copy>(
    token: T,
    form: SsimLumaForm,
    m1: GenericF32x8<T>,
    m2: GenericF32x8<T>,
    ssq: GenericF32x8<T>,
    s12: GenericF32x8<T>,
) -> GenericF32x8<T> {
    SsimSplats8::new(token, form).dissim(m1, m2, ssq, s12)
}

/// Scalar direct-error form — the tail sibling of [`SsimSplats16::direct`],
/// same spellings so vector and tail agree to their usual `mul_add` residual.
#[inline(always)]
pub(crate) fn ssim_direct_raw_scalar(
    form: SsimLumaForm,
    m1: f32,
    m2: f32,
    ssq: f32,
    err: f32,
) -> f32 {
    let mu_diff = m1 - m2;
    let mean_error2 = mu_diff * mu_diff;
    let variance_sum = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)).max(0.0f32);
    let error_variance = (err - mean_error2).max(0.0f32);
    let luma_loss = match form {
        SsimLumaForm::Ssim2Legacy => mean_error2,
        SsimLumaForm::Clamp => mean_error2.min(1.0f32),
        SsimLumaForm::Lorentz => mean_error2 / (mean_error2 + 1.0f32),
        SsimLumaForm::SsimLumaC1 => mean_error2 / m1.mul_add(m1, m2.mul_add(m2, C_SSIM_LUMA)),
    };
    (1.0f32 - luma_loss).mul_add(error_variance / (variance_sum + C2), luma_loss)
}

/// 16-lane direct-error form; see [`SsimSplats16::direct`].
#[cfg_attr(not(feature = "avx512"), allow(dead_code))] // only the v4x arcane bodies call this; they are not emitted without `avx512`
#[inline(always)]
pub(crate) fn ssim_direct16<T: F32x16Backend + Copy>(
    token: T,
    form: SsimLumaForm,
    m1: f32x16<T>,
    m2: f32x16<T>,
    ssq: f32x16<T>,
    err: f32x16<T>,
) -> f32x16<T> {
    SsimSplats16::new(token, form).direct(m1, m2, ssq, err)
}

/// 8-lane direct-error form; see [`SsimSplats8::direct`].
#[inline(always)]
pub(crate) fn ssim_direct8<T: F32x8Backend + Copy>(
    token: T,
    form: SsimLumaForm,
    m1: GenericF32x8<T>,
    m2: GenericF32x8<T>,
    ssq: GenericF32x8<T>,
    err: GenericF32x8<T>,
) -> GenericF32x8<T> {
    SsimSplats8::new(token, form).direct(m1, m2, ssq, err)
}

/// **FEATCANON exact oracle sibling** of [`ssim_dissim_raw_scalar`] /
/// [`ssim_direct_raw_scalar`]: the same formula evaluated in f64 (fused
/// `f64::mul_add` keeps the same product-sum groupings; constants widen
/// exactly). `direct` selects the revision-3 direct-error form. Used by the
/// `ZENSIM_FEATCANON=exact` extraction arm as the real-arithmetic reference.
/// Measurement only (`oracle` feature).
#[cfg(feature = "oracle")]
pub(crate) fn ssim_dissim_exact(
    form: SsimLumaForm,
    m1: f64,
    m2: f64,
    ssq: f64,
    s12: f64,
    direct: bool,
) -> f64 {
    if direct {
        let mu_diff = m1 - m2;
        let mean_error2 = mu_diff * mu_diff;
        let variance_sum = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)).max(0.0f64);
        let error_variance = (s12 - mean_error2).max(0.0f64);
        let luma_loss = match form {
            SsimLumaForm::Ssim2Legacy => mean_error2,
            SsimLumaForm::Clamp => mean_error2.min(1.0f64),
            SsimLumaForm::Lorentz => mean_error2 / (mean_error2 + 1.0f64),
            SsimLumaForm::SsimLumaC1 => {
                mean_error2 / m1.mul_add(m1, m2.mul_add(m2, C_SSIM_LUMA as f64))
            }
        };
        return (1.0f64 - luma_loss)
            .mul_add(error_variance / (variance_sum + C2 as f64), luma_loss);
    }
    let num_s = 2.0f64.mul_add((-m1).mul_add(m2, s12), C2 as f64);
    let denom_s = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)) + C2 as f64;
    let (num_m, den_m) = match form {
        SsimLumaForm::Ssim2Legacy => {
            let mu_diff = m1 - m2;
            (mu_diff.mul_add(-mu_diff, 1.0f64), 1.0f64)
        }
        SsimLumaForm::SsimLumaC1 => (
            2.0f64.mul_add(m1 * m2, C_SSIM_LUMA as f64),
            m1.mul_add(m1, m2.mul_add(m2, C_SSIM_LUMA as f64)),
        ),
        SsimLumaForm::Lorentz => {
            let mu_diff = m1 - m2;
            (1.0f64, mu_diff.mul_add(mu_diff, 1.0f64))
        }
        SsimLumaForm::Clamp => {
            let mu_diff = m1 - m2;
            (mu_diff.mul_add(-mu_diff, 1.0f64).max(0.0f64), 1.0f64)
        }
    };
    1.0f64 - (num_m * num_s) / (den_m * denom_s)
}

/// Synthetic derivative prerequisite for the actual Rev3 Clamp expression.
/// Partial order: distorted mean, mean square sum, mean squared error.
/// Reference mean is held fixed. Kinks select zero for the inactive branch;
/// these are mathematical partials, not derivatives of f32 rounding steps.
#[cfg(test)]
pub(crate) fn probe_rev3_ssim_partials(m1: f32, m2: f32, ssq: f32, err: f32) -> [f64; 3] {
    let form = SsimLumaForm::for_revision(crate::feature_defs::FormulaRevision::Rev3);
    assert_eq!(form, SsimLumaForm::Clamp);
    if ssim_direct_raw_scalar(form, m1, m2, ssq, err) <= 0.0 {
        return [0.0; 3];
    }
    let difference = m2 - m1;
    let q = difference * difference;
    let variance = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq));
    let error_variance = err - q;
    let denominator = f64::from(variance.max(0.0)) + f64::from(C2);
    let error = f64::from(error_variance.max(0.0));
    let luma = f64::from(q.min(1.0));
    let dl = if q < 1.0 {
        2.0 * f64::from(difference)
    } else {
        0.0
    };
    let de = if error_variance > 0.0 {
        -2.0 * f64::from(difference)
    } else {
        0.0
    };
    let dv = if variance > 0.0 {
        -2.0 * f64::from(m2)
    } else {
        0.0
    };
    let square = if variance > 0.0 {
        -(1.0 - luma) * error / denominator.powi(2)
    } else {
        0.0
    };
    let direct_error = if error_variance > 0.0 {
        (1.0 - luma) / denominator
    } else {
        0.0
    };
    [
        (1.0 - error / denominator) * dl
            + (1.0 - luma) * (de / denominator - error * dv / denominator.powi(2)),
        square,
        direct_error,
    ]
}

#[cfg(test)]
mod tests {
    #[test]
    fn rev3_ssim_partials_match_existing_scalar_function() {
        use super::*;
        let mut checks = 0;
        let mut max_normalized = 0.0_f64;
        for values in [
            [0.25, 0.35, 0.5, 0.06],
            [0.2, 1.7, 4.0, 3.0],
            [0.25, 0.35, 0.5, 0.005],
            [0.25, 0.35, 0.1, 0.06],
        ] {
            let partials = probe_rev3_ssim_partials(values[0], values[1], values[2], values[3]);
            for coordinate in 1..4 {
                for epsilon in [0.001, 0.0005] {
                    let mut plus = values;
                    let mut minus = values;
                    plus[coordinate] += epsilon;
                    minus[coordinate] -= epsilon;
                    let score = |v: [f32; 4]| {
                        f64::from(
                            ssim_direct_raw_scalar(SsimLumaForm::Clamp, v[0], v[1], v[2], v[3])
                                .max(0.0),
                        )
                    };
                    let actual = (score(plus) - score(minus)) / (2.0 * f64::from(epsilon));
                    let predicted = partials[coordinate - 1];
                    let error = (actual - predicted).abs();
                    let tolerance = 2e-5_f64.max(0.01 * actual.abs().max(predicted.abs()));
                    max_normalized = max_normalized.max(error / tolerance);
                    assert!(
                        error <= tolerance,
                        "moments={values:?} coordinate={coordinate} eps={epsilon}: predicted={predicted} actual={actual}"
                    );
                    checks += 1;
                }
            }
        }
        assert_eq!(probe_rev3_ssim_partials(0.25, 0.25, 0.5, 0.0), [0.0; 3]);
        eprintln!("SSIM_PARTIALS checks={checks} max_fraction_of_tolerance={max_normalized:.9e}");
    }

    #[test]
    fn stable_moments_match_direct_windows_and_analytic_controls() {
        let _ = stable_kernel_cases();
    }

    #[test]
    #[ignore = "changes process-wide SIMD dispatch; run this test alone"]
    fn stable_moments_are_exact_across_simd_tiers() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
        let mut baseline = None;
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            // The kernel AND the Rev3 pools it feeds: both are dispatched
            // code, both are new, and a tier that disagreed in the reducer
            // would move a pooled feature without moving a single per-pixel
            // signal. Snapshotting them together costs nothing and closes
            // that gap in the one test that may change global dispatch.
            let actual = (stable_kernel_cases(), stable_pool_cases());
            if let Some(expected) = &baseline {
                assert_eq!(&actual, expected, "SIMD permutation {}", perm.label);
            } else {
                baseline = Some(actual);
            }
        });
        assert!(report.permutations_run >= 1);
        println!(
            "{} SIMD permutations: exact stable moment outputs",
            report.permutations_run
        );
    }

    /// The three Rev3 retained-signal reducers over the same geometries the
    /// kernel cases use, snapshotted bitwise. `f64` sums are reduced per SIMD
    /// lane, so a tier with a different lane count reduces in a different
    /// order — these are recorded as bits, and any disagreement between tiers
    /// is therefore visible rather than absorbed by a tolerance.
    fn stable_pool_cases() -> Vec<[u64; 9]> {
        let mut out = Vec::new();
        for n in [1usize, 15, 16, 17, 31, 33, 64, 129, 512] {
            let signal: Vec<f32> = (0..n)
                .map(|i| ((i * 31 % 97) as f32 / 97.0) * 0.6)
                .collect();
            let activity: Vec<f32> = (0..n)
                .map(|i| ((i * 13 % 61) as f32 / 61.0) * 0.9)
                .collect();
            let ((m, m4, m2), (w, w4, w2)) =
                crate::simd_ops::ssim_signal_inline_both(&signal, &activity, 4.0, 4.0);
            let (om, om4, om2) = crate::simd_ops::ssim_signal_inline_mask(&signal, &activity, 4.0);
            let (oi, _, _) = crate::simd_ops::ssim_signal_iw_inline(&signal, &activity, 4.0);
            out.push([
                m.to_bits(),
                m4.to_bits(),
                m2.to_bits(),
                w.to_bits(),
                w4.to_bits(),
                w2.to_bits(),
                om.to_bits(),
                om4.to_bits() ^ om2.to_bits(),
                oi.to_bits(),
            ]);
        }
        out
    }

    /// The tiled kernel agrees with the sliding one to the registered
    /// acceptance, across the geometries and radii the kernel controls use.
    ///
    /// NOT bit-identical, and must not be: the two traversals sum the same
    /// samples in different orders. What is asserted is the registered bound
    /// (`2e-10 + 2e-6*|reference|`) against the SLIDING kernel, plus exact
    /// identity where the inputs are identical.
    #[test]
    fn tiled_kernel_matches_the_sliding_kernel_within_the_registered_bound() {
        let mut slide_scratch = StableSsimScratch::default();
        let mut tiled_scratch = TiledSsimScratch::default();
        let mut worst = 0.0f64;
        for (w, h) in [(1, 1), (3, 7), (17, 9), (64, 33), (129, 71), (192, 40)] {
            for radius in [0usize, 1, 5] {
                let source: Vec<f32> = (0..w * h)
                    .map(|i| 0.2 + ((i * 17 + i / w * 11) % 137) as f32 / 100.0)
                    .collect();
                let distorted: Vec<f32> = source
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        if i % 5 == 0 {
                            v + 0.0001
                        } else if i % 13 == 0 {
                            v * 0.7
                        } else {
                            v
                        }
                    })
                    .collect();
                for form in [
                    SsimLumaForm::Ssim2Legacy,
                    SsimLumaForm::Clamp,
                    SsimLumaForm::Lorentz,
                    SsimLumaForm::SsimLumaC1,
                ] {
                    let mut a = vec![0.0f32; w * h];
                    let mut b = vec![0.0f32; w * h];
                    stable_ssim_plane(
                        &source,
                        &distorted,
                        w,
                        h,
                        radius,
                        form,
                        &mut a,
                        &mut slide_scratch,
                    );
                    stable_ssim_plane_tiled(
                        &source,
                        &distorted,
                        w,
                        h,
                        radius,
                        form,
                        &mut b,
                        &mut tiled_scratch,
                    );
                    for (i, (&x, &y)) in a.iter().zip(&b).enumerate() {
                        assert!(y.is_finite(), "non-finite at {w}x{h} r{radius} idx {i}");
                        let (x, y) = (x as f64, y as f64);
                        let tol = 2e-10 + 2e-6 * x.abs();
                        assert!(
                            (x - y).abs() <= tol,
                            "{w}x{h} r{radius} {form:?} idx {i}: tiled {y} vs sliding {x}"
                        );
                        worst = worst.max((x - y).abs());
                    }
                    // Exact identity: an identical pair is exactly zero.
                    let mut z = vec![0.0f32; w * h];
                    stable_ssim_plane_tiled(
                        &source,
                        &source,
                        w,
                        h,
                        radius,
                        form,
                        &mut z,
                        &mut tiled_scratch,
                    );
                    assert!(
                        z.iter().all(|v| *v == 0.0),
                        "identical inputs must give exactly zero at {w}x{h} r{radius}"
                    );
                }
            }
        }
        println!("tiled vs sliding: worst |delta| {worst:.3e}");
    }

    /// Paired, interleaved timing of the two moment traversals.
    ///
    /// Both arms run in the SAME process and alternate every round, so
    /// thermal/turbo drift is shared rather than accumulated onto whichever
    /// ran second — the bias an isolated back-to-back comparison would bake
    /// in. Three planes per round, matching the kernel's registered timing
    /// receipt so the numbers are comparable to it.
    ///
    /// `#[ignore]`: a timing probe, not a gate. Run explicitly:
    /// `cargo test -p zensim --release --lib stable_kernel_traversal_ab -- --ignored --nocapture`
    #[test]
    #[ignore = "timing probe; run explicitly in release"]
    fn stable_kernel_traversal_ab() {
        let rounds: usize = std::env::var("ZEN_KAB_ROUNDS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30);
        for (w, h) in [(1024usize, 1024usize), (2048, 2048)] {
            let src: [Vec<f32>; 3] = core::array::from_fn(|c| {
                (0..w * h)
                    .map(|i| 0.2 + ((i * 17 + i / w * 11 + c * 41) % 137) as f32 / 100.0)
                    .collect()
            });
            let dst: [Vec<f32>; 3] = core::array::from_fn(|c| {
                src[c]
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| if i % 13 == 0 { v * 0.7 } else { v + 0.0001 })
                    .collect()
            });
            let mut out: [Vec<f32>; 3] = core::array::from_fn(|_| vec![0.0; w * h]);
            let mut slide = StableSsimScratch::default();
            let mut tiled = TiledSsimScratch::default();
            let (mut ts, mut tt) = (Vec::new(), Vec::new());
            for round in 0..rounds + 2 {
                // Alternate which arm goes first so neither owns the cold cache.
                for first_is_slide in [round % 2 == 0, round % 2 != 0] {
                    let start = std::time::Instant::now();
                    for c in 0..3 {
                        if first_is_slide {
                            stable_ssim_plane(
                                &src[c],
                                &dst[c],
                                w,
                                h,
                                5,
                                SsimLumaForm::Ssim2Legacy,
                                &mut out[c],
                                &mut slide,
                            );
                        } else {
                            stable_ssim_plane_tiled(
                                &src[c],
                                &dst[c],
                                w,
                                h,
                                5,
                                SsimLumaForm::Ssim2Legacy,
                                &mut out[c],
                                &mut tiled,
                            );
                        }
                    }
                    let ms = start.elapsed().as_secs_f64() * 1e3;
                    std::hint::black_box(&out);
                    if round >= 2 {
                        if first_is_slide {
                            ts.push(ms)
                        } else {
                            tt.push(ms)
                        }
                    }
                }
            }
            let stat = |v: &mut Vec<f64>| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                (v[0], v[v.len() / 2])
            };
            let (smin, smed) = stat(&mut ts);
            let (tmin, tmed) = stat(&mut tt);
            println!(
                "{w}x{h} n={}  sliding min {smin:.3} med {smed:.3} ms | tiled min {tmin:.3} med {tmed:.3} ms | tiled/sliding {:.3}x",
                ts.len(),
                tmed / smed
            );
        }
    }

    fn stable_kernel_cases() -> Vec<Vec<u32>> {
        use super::*;
        let mut snapshots = Vec::new();
        let mut scratch = StableSsimScratch::default();
        for (w, h) in [(1, 1), (3, 7), (17, 9), (64, 33), (129, 71)] {
            for radius in [0, 1, 5] {
                let source: Vec<f32> = (0..w * h)
                    .map(|i| 0.2 + ((i * 17 + i / w * 11) % 137) as f32 / 100.0)
                    .collect();
                let distorted: Vec<f32> = source
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        if i % 5 == 0 {
                            v + 0.0001
                        } else if i % 13 == 0 {
                            v * 0.7
                        } else {
                            v
                        }
                    })
                    .collect();
                for form in [
                    SsimLumaForm::Ssim2Legacy,
                    SsimLumaForm::Clamp,
                    SsimLumaForm::Lorentz,
                    SsimLumaForm::SsimLumaC1,
                ] {
                    let mut out = vec![f32::NAN; w * h + 1];
                    stable_ssim_plane(
                        &source,
                        &distorted,
                        w,
                        h,
                        radius,
                        form,
                        &mut out,
                        &mut scratch,
                    );
                    let (reference, _, _) =
                        precision_reference(&source, &distorted, w, h, radius, form);
                    for (&a, &b) in out.iter().zip(&reference) {
                        assert!(
                            a.is_finite() && (a as f64 - b).abs() <= 2e-10 + 2e-6 * b.abs(),
                            "{w}x{h}, r{radius}, {form:?}: {a} vs {b}"
                        );
                    }
                    assert!(out[w * h].is_nan(), "output overrun");
                    snapshots.push(out[..w * h].iter().map(|v| v.to_bits()).collect());
                    stable_ssim_plane(&source, &source, w, h, radius, form, &mut out, &mut scratch);
                    assert!(
                        out[..w * h].iter().all(|&v| v == 0.0),
                        "identity must be exact"
                    );
                }
                let source = vec![0.75f32; w * h];
                let distorted = vec![0.7501f32; w * h];
                let mut out = vec![0.0; w * h];
                stable_ssim_plane(
                    &source,
                    &distorted,
                    w,
                    h,
                    radius,
                    SsimLumaForm::Ssim2Legacy,
                    &mut out,
                    &mut scratch,
                );
                let expected = (source[0] as f64 - distorted[0] as f64).powi(2);
                assert!(
                    out.iter()
                        .all(|&v| (v as f64 - expected).abs() <= 2e-10 + 2e-6 * expected)
                );
                assert_eq!(scratch.rows.len(), w * (2 * radius + 1));
                assert_eq!(scratch.vertical.len(), w);
            }
        }
        snapshots
    }

    use super::*;

    /// A sweep that reaches both the healthy regime and F4's pathology.
    ///
    /// `ssq` is `E[s^2] + E[d^2]`, so `denom_s = ssq - mu1^2 - mu2^2 + C2` is
    /// the variance sum; the cases keep it positive, as the walk does.
    fn cases() -> Vec<(f32, f32, f32, f32)> {
        let mut v = Vec::new();
        for &(m1, m2) in &[
            (0.0f32, 0.0f32),
            (0.3, 0.3),
            (0.3, 0.31), // photographic
            (0.05, 0.9),
            (50.0, 60.0), // the 2026-07-16 analytic rows
            (100.0, 400.0),
            (50.0, 2450.0),
            (2450.0, 50.0),
        ] {
            for &var in &[0.0f32, 1e-6, 1e-3, 0.25, 10.0] {
                for &cov_frac in &[-1.0f32, -0.5, 0.0, 0.5, 1.0] {
                    let ssq = m1 * m1 + m2 * m2 + 2.0 * var;
                    let s12 = m1 * m2 + cov_frac * var;
                    v.push((m1, m2, ssq, s12));
                }
            }
        }
        v
    }

    /// The exact expression the 36 hand-copied sites carried, transcribed
    /// verbatim — the control for the extraction.
    fn legacy_scalar(m1: f32, m2: f32, ssq: f32, s12: f32) -> f32 {
        let mu_diff = m1 - m2;
        let num_m = mu_diff.mul_add(-mu_diff, 1.0f32);
        let num_s = 2.0f32.mul_add((-m1).mul_add(m2, s12), C2);
        let denom_s = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)) + C2;
        1.0f32 - (num_m * num_s) / denom_s
    }

    /// **The extraction gate.** `Ssim2Legacy` must reproduce the replaced
    /// expression BIT-for-BIT, not merely closely: this is what lets the
    /// 36-site rewrite claim to be inert.
    #[test]
    fn legacy_arm_is_bit_identical_to_the_expression_it_replaced() {
        for (m1, m2, ssq, s12) in cases() {
            let got = ssim_dissim_raw_scalar(SsimLumaForm::Ssim2Legacy, m1, m2, ssq, s12);
            let want = legacy_scalar(m1, m2, ssq, s12);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "legacy arm diverged at mu=({m1},{m2}) ssq={ssq} s12={s12}: {got} vs {want}"
            );
        }
    }

    /// **F4, stated as a failing property of the shipped form.** The legacy
    /// arm is unbounded; every other arm is not. If this ever stops failing
    /// for `Ssim2Legacy`, the pathology has been fixed somewhere else and
    /// this module's reason to exist has changed.
    #[test]
    fn only_the_legacy_arm_is_unbounded() {
        // mu1 = 50, mu2 = 2450 on a near-flat region: the configuration
        // `benchmarks/ssim_moment_explosion_2026-07-16.md` reproduces 5.8e6 from.
        let (m1, m2) = (50.0f32, 2450.0f32);
        let var = 1e-6f32;
        let ssq = m1 * m1 + m2 * m2 + 2.0 * var;
        let s12 = m1 * m2 + var;

        let legacy = ssim_dissim_raw_scalar(SsimLumaForm::Ssim2Legacy, m1, m2, ssq, s12);
        assert!(
            legacy > 1e6,
            "the F4 pathology should reproduce here, got {legacy}"
        );

        for form in [
            SsimLumaForm::SsimLumaC1,
            SsimLumaForm::Lorentz,
            SsimLumaForm::Clamp,
        ] {
            assert!(form.bounds_dissim(), "{form:?} claims not to bound");
            let d = ssim_dissim_raw_scalar(form, m1, m2, ssq, s12);
            assert!(
                (-1e-5..=2.0 + 1e-5).contains(&d),
                "{form:?} produced d = {d} outside [0, 2] on the pathological case"
            );
        }
    }

    /// Boundedness across the whole sweep, not just the headline case.
    #[test]
    fn bounded_arms_keep_d_in_zero_to_two_everywhere() {
        for form in [
            SsimLumaForm::SsimLumaC1,
            SsimLumaForm::Lorentz,
            SsimLumaForm::Clamp,
        ] {
            for (m1, m2, ssq, s12) in cases() {
                let d = ssim_dissim_raw_scalar(form, m1, m2, ssq, s12);
                assert!(
                    d.is_finite() && (-1e-4..=2.0 + 1e-4).contains(&d),
                    "{form:?} at mu=({m1},{m2}) ssq={ssq} s12={s12} gave {d}"
                );
            }
        }
    }

    /// **`Clamp` is exact where it matters.** It is defined to differ from the
    /// shipped form only where `(mu1-mu2)^2 > 1`, which is why it moved ZERO
    /// of 22,396 cells on the synthetic dump. Pinning that here means a future
    /// edit cannot quietly make it a general-purpose approximation.
    #[test]
    fn clamp_arm_is_bit_identical_to_legacy_below_the_knee() {
        for (m1, m2, ssq, s12) in cases() {
            if (m1 - m2) * (m1 - m2) > 1.0 {
                continue;
            }
            let c = ssim_dissim_raw_scalar(SsimLumaForm::Clamp, m1, m2, ssq, s12);
            let l = ssim_dissim_raw_scalar(SsimLumaForm::Ssim2Legacy, m1, m2, ssq, s12);
            assert_eq!(c.to_bits(), l.to_bits(), "clamp diverged below the knee");
        }
    }

    /// **The G-OWNER claim, checked.** `SsimLumaC1`'s luminance term is not
    /// merely *like* the crate's `bounded_sim` primitive — it IS it, and this
    /// compares the f32 kernel arm against the f64 owner rather than against a
    /// second transcription.
    ///
    /// `crate::feature_v2` only exists under `feature-regime-v2` — this test
    /// asserts parity with it, so it is unrunnable (not just untestable)
    /// without that feature.
    #[cfg(feature = "feature-regime-v2")]
    #[test]
    fn ssim_luma_arm_is_the_crate_s_own_bounded_sim() {
        for (m1, m2, ssq, s12) in cases() {
            let want_luma: f64 =
                crate::feature_v2::bounded_sim(m1 as f64, m2 as f64, C_SSIM_LUMA as f64);
            // Reconstruct the arm's luminance factor from its output:
            // d = 1 - luma * (num_s/denom_s).
            let num_s = 2.0f64 * ((s12 - m1 * m2) as f64) + C2 as f64;
            let denom_s = ((ssq - m1 * m1 - m2 * m2) as f64) + C2 as f64;
            let d = ssim_dissim_raw_scalar(SsimLumaForm::SsimLumaC1, m1, m2, ssq, s12) as f64;
            let want = 1.0 - want_luma * (num_s / denom_s);
            let scale = want.abs().max(1.0);
            assert!(
                (d - want).abs() <= 2e-3 * scale,
                "arm vs bounded_sim at mu=({m1},{m2}): {d} vs {want}"
            );
            assert!(
                want_luma > 0.0 && want_luma <= 1.0 + 1e-12,
                "bounded_sim escaped (0,1]: {want_luma}"
            );
        }
    }

    /// The constant is DERIVED, and both derivations are pinned so a future
    /// edit has to break an equation rather than a taste.
    ///
    /// The second derivation checks agreement with `crate::feature_v2`'s own
    /// regularizers, which only exist under `feature-regime-v2`.
    #[cfg(feature = "feature-regime-v2")]
    #[test]
    fn c1_is_derived_from_the_constants_already_present() {
        // 1. The SSIM relation to the C2 already in the kernel: C2 = (K2*L)^2
        //    at K2 = 0.03, L = 1 -> C1 = (K1*L)^2 at the textbook K1 = 0.01.
        let k2 = (C2 as f64).sqrt();
        assert!((k2 - 0.03).abs() < 1e-9, "C2 is not (0.03)^2: K2 = {k2}");
        // Tolerances are RELATIVE: both constants are f32, so an exact f64
        // comparison would only be testing f32's round-trip error (2.5e-12
        // here), not the derivation.
        let k1 = 0.01f64;
        let rel = |a: f64, b: f64| (a - b).abs() / b.abs();
        assert!(
            rel(C_SSIM_LUMA as f64, k1 * k1) < 1e-6,
            "C_SSIM_LUMA is not (0.01)^2: {C_SSIM_LUMA}"
        );
        // 2. The family's own bounded_sim regularizers.
        for c in [
            crate::feature_v2::C_EDGE,
            crate::feature_v2::C_GMS,
            crate::feature_v2::C_CONTRAST,
        ] {
            assert!(
                rel(c, C_SSIM_LUMA as f64) < 1e-6,
                "family regularizer {c} disagrees with C_SSIM_LUMA"
            );
        }
    }

    /// **G3.2** — pinning the shipped revision is the same as not pinning it.
    #[test]
    fn selecting_the_shipped_revision_is_a_no_op() {
        assert_eq!(
            SsimLumaForm::for_revision(SHIPPED_REVISION),
            SsimLumaForm::for_revision(active_revision()),
            "the active revision is not the shipped one (is ZENSIM_FORMULA_REV set?)"
        );
        assert!(
            !SsimLumaForm::Ssim2Legacy.bounds_dissim(),
            "the legacy arm must not claim to bound"
        );
    }
}
