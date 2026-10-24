//! Fused V-blur + feature extraction for streaming strips.
//!
//! Instead of 4 separate V-blur passes (writing to memory) followed by 7 reduction
//! passes (reading from memory), this module fuses everything into a single column-wise
//! pass. V-blurred values stay in registers and all features are computed inline.
//!
//! Memory pass reduction: ~40 passes → ~12 passes per channel (with fused H-blur).
#![allow(
    clippy::assign_op_pattern,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use crate::ssim_form::{
    ssim_direct_raw_scalar, ssim_direct8, ssim_direct16, ssim_dissim_raw_scalar, ssim_dissim8,
    ssim_dissim16,
};
#[cfg(target_arch = "x86_64")]
use archmage::arcane;
use archmage::incant;
use archmage::magetypes;
use magetypes::simd::backends::F32x8Backend;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::backends::F32x16Backend;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::f32x8;
use magetypes::simd::generic::f32x8 as GenericF32x8;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::generic::f32x16;

// ============================================================
// Free raw-moments accumulation — shared across every SIMD tier
// (`benchmarks/free_features_2026-09-01.md`, `benchmarks/profile_d_notax_2026-09-01.md`)
// ============================================================
//
// The four FREE raw moments (`Σs, Σd, Σs², Σd²` — `V1FreeExtras::RawMoments`,
// `StripChannelAccum::sum_s` etc.) are the same two-line accumulate + one
// conditional four-line finish at every call site, independent of lane
// width or which concrete SIMD backend produced the row's `s`/`d` values.
// Before this pair of helpers the sequence was hand-duplicated at 6 vector
// sites (`_v4`'s and `_v4x`'s native f32x16 main loops, `_v4`'s and `_v4x`'s
// f32x8 REMAINDER loops via `token.v3()`, `_v3`'s native f32x8 main loop,
// and the `#[magetypes(neon, wasm128, scalar)]`-generated function's f32x8
// main loop) plus 4 scalar-tail sites — the free-features doc's own count.
// Consolidating the two WIDTHS into one generic definition each removes the
// vector-site duplication (6 -> 2 source definitions, still 6 call sites)
// and is also the "no waste, no effort duplication" extension point a
// future `V1FreeExtras` variant (a "class C" slot — see the free-features
// doc §4) can reuse without re-deriving or re-copying this arithmetic a
// 7th/8th/9th time: add the new lane-accumulate step here once, call it
// from wherever the new class needs it.
//
// `#[inline(always)]`, not `#[rite]`: both helpers are GENERIC over a
// backend TRAIT (`T: F32x8Backend` / `T: F32x16Backend`), not a concrete
// token, so there is no single `#[target_feature]` string to attach at the
// definition site the way `#[rite]` needs for a concrete-token function —
// each monomorphized instantiation inherits its caller's already-established
// feature region purely through ordinary generic inlining, exactly like
// every other `T: F32x8Backend`-generic kernel in this codebase
// (`feature_v2.rs`'s `dense_block_kernel_generic`, `ssim_d_local_v`, etc.).
// `dense_block_kernel_generic`'s own doc comment already carries the
// MEASURED reason forcing the inline is mandatory here too: an un-inlined
// generic SIMD helper compiles to a call into a `core::arch` shim OUTSIDE
// the `#[target_feature]` region, measured as a 5.3x whole-extraction
// regression on that kernel. Verified inlined away on this refactor: `nm -C`
// on the compiled `ssim2_speed_bar` release binary shows the tier entry
// points present (`zensim::fused::__arcane_fused_vblur_ssim_inner_{v3,v4,v4x}`)
// and zero occurrences of `raw_moments_accumulate`/`raw_moments_finish`
// anywhere in the binary — no un-inlined call site survives.

/// Accumulate one row's contribution to the four free raw-moment lane sums.
/// Bit-identical to the hand-inlined `fm_s = fm_s + s; …` sequence it
/// replaces — same operations, same order, same intermediate rounding
/// (`s * s` before adding, not any fused/reassociated form).
/// Masked and IW pools of the v1 "372" extension, fused into the V sweep
/// (revision 3 only). The separate passes — `simd_ops::ssim_signal_inline_both`,
/// `edge_diff_channel_inline_both`, `build_inline_mse` — form exactly the same
/// per-pixel values from the same inputs; what differs is the ORDER the f64
/// chunk sums are added in (column-group-major here, row-major there), so the
/// pooled slots move at the last f64 bits and the families are registered as
/// a moved era. `act` is the V-blurred activity `blur(|src - H(src)|)`, from
/// the `h_act` plane the caller H-blurred; its V recurrence is the same
/// `sum + add - rem` the other four planes use.
#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct ExtPoolsWork {
    /// Any of it at all. When false every `h_act` read is skipped.
    pub on: bool,
    /// Masked (`1 / (1 + k * act)`) pools.
    pub mask: bool,
    /// IW (`1 + k * act`) pools.
    pub iw: bool,
    pub k_mask: f32,
    pub k_iw: f32,
}

// x86_64-only: the AVX-512 (`_v4x`) and AVX2 (`_v4`) V tiers are its only
// callers, and it uses the `f32x16` generic type which is imported only there.
// `dead_code` allow for the same reason as the sibling 16-lane helpers: a
// build without the `avx512` feature never dispatches the tiers that call it.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
#[inline(always)]
fn ext_accumulate16<T: F32x16Backend + Copy>(
    token: T,
    acc: &mut StripChannelAccum,
    sd: f32x16<T>,
    ed: f32x16<T>,
    pd: f32x16<T>,
    act: f32x16<T>,
    ext: ExtPoolsWork,
) {
    let one = f32x16::<T>::splat_t(token, 1.0);
    let zero = f32x16::<T>::zero_t(token);
    acc.act_sum += act.reduce_add() as f64;
    let d2 = pd * pd;
    if ext.mask {
        let w = one / f32x16::<T>::splat_t(token, ext.k_mask).mul_add(act, one);
        let da = (sd * w).max(zero);
        let d2a = da * da;
        let d4a = d2a * d2a;
        acc.masked_ssim_d += da.reduce_add() as f64;
        acc.masked_ssim_d2 += d2a.reduce_add() as f64;
        acc.masked_ssim_d4 += d4a.reduce_add() as f64;
        let e = ed * w;
        let a2 = e.max(zero) * e.max(zero);
        let dl2 = (-e).max(zero) * (-e).max(zero);
        acc.masked_art4 += (a2 * a2).reduce_add() as f64;
        acc.masked_det4 += (dl2 * dl2).reduce_add() as f64;
        acc.masked_mse += (d2 * w).reduce_add() as f64;
    }
    if ext.iw {
        let w = f32x16::<T>::splat_t(token, ext.k_iw).mul_add(act, one);
        let db = (sd * w).max(zero);
        let d2b = db * db;
        let d4b = d2b * d2b;
        acc.iw_ssim_d += db.reduce_add() as f64;
        acc.iw_ssim_d2 += d2b.reduce_add() as f64;
        acc.iw_ssim_d4 += d4b.reduce_add() as f64;
        let e = ed * w;
        let a2 = e.max(zero) * e.max(zero);
        let dl2 = (-e).max(zero) * (-e).max(zero);
        acc.iw_art4 += (a2 * a2).reduce_add() as f64;
        acc.iw_det4 += (dl2 * dl2).reduce_add() as f64;
        acc.iw_mse += (d2 * w).reduce_add() as f64;
    }
}

/// 8-lane sibling of [`ext_accumulate16`].
#[inline(always)]
fn ext_accumulate8<T: F32x8Backend + Copy>(
    token: T,
    acc: &mut StripChannelAccum,
    sd: GenericF32x8<T>,
    ed: GenericF32x8<T>,
    pd: GenericF32x8<T>,
    act: GenericF32x8<T>,
    ext: ExtPoolsWork,
) {
    let one = GenericF32x8::<T>::splat_t(token, 1.0);
    let zero = GenericF32x8::<T>::zero_t(token);
    acc.act_sum += act.reduce_add() as f64;
    let d2 = pd * pd;
    if ext.mask {
        let w = one / GenericF32x8::<T>::splat_t(token, ext.k_mask).mul_add(act, one);
        let da = (sd * w).max(zero);
        let d2a = da * da;
        let d4a = d2a * d2a;
        acc.masked_ssim_d += da.reduce_add() as f64;
        acc.masked_ssim_d2 += d2a.reduce_add() as f64;
        acc.masked_ssim_d4 += d4a.reduce_add() as f64;
        let e = ed * w;
        let a2 = e.max(zero) * e.max(zero);
        let dl2 = (-e).max(zero) * (-e).max(zero);
        acc.masked_art4 += (a2 * a2).reduce_add() as f64;
        acc.masked_det4 += (dl2 * dl2).reduce_add() as f64;
        acc.masked_mse += (d2 * w).reduce_add() as f64;
    }
    if ext.iw {
        let w = GenericF32x8::<T>::splat_t(token, ext.k_iw).mul_add(act, one);
        let db = (sd * w).max(zero);
        let d2b = db * db;
        let d4b = d2b * d2b;
        acc.iw_ssim_d += db.reduce_add() as f64;
        acc.iw_ssim_d2 += d2b.reduce_add() as f64;
        acc.iw_ssim_d4 += d4b.reduce_add() as f64;
        let e = ed * w;
        let a2 = e.max(zero) * e.max(zero);
        let dl2 = (-e).max(zero) * (-e).max(zero);
        acc.iw_art4 += (a2 * a2).reduce_add() as f64;
        acc.iw_det4 += (dl2 * dl2).reduce_add() as f64;
        acc.iw_mse += (d2 * w).reduce_add() as f64;
    }
}

/// Scalar sibling of [`ext_accumulate16`] for the ragged column tail.
#[inline(always)]
fn ext_accumulate_scalar(
    acc: &mut StripChannelAccum,
    sd: f32,
    ed: f32,
    pd: f32,
    act: f32,
    ext: ExtPoolsWork,
) {
    acc.act_sum += act as f64;
    let d2 = pd * pd;
    if ext.mask {
        let w = 1.0f32 / (1.0f32 + ext.k_mask * act);
        let da = (sd * w).max(0.0);
        let d2a = da * da;
        acc.masked_ssim_d += da as f64;
        acc.masked_ssim_d2 += d2a as f64;
        acc.masked_ssim_d4 += (d2a * d2a) as f64;
        let e = ed * w;
        let a2 = e.max(0.0) * e.max(0.0);
        let dl2 = (-e).max(0.0) * (-e).max(0.0);
        acc.masked_art4 += (a2 * a2) as f64;
        acc.masked_det4 += (dl2 * dl2) as f64;
        acc.masked_mse += (d2 * w) as f64;
    }
    if ext.iw {
        let w = 1.0f32 + ext.k_iw * act;
        let db = (sd * w).max(0.0);
        let d2b = db * db;
        acc.iw_ssim_d += db as f64;
        acc.iw_ssim_d2 += d2b as f64;
        acc.iw_ssim_d4 += (d2b * d2b) as f64;
        let e = ed * w;
        let a2 = e.max(0.0) * e.max(0.0);
        let dl2 = (-e).max(0.0) * (-e).max(0.0);
        acc.iw_art4 += (a2 * a2) as f64;
        acc.iw_det4 += (dl2 * dl2) as f64;
        acc.iw_mse += (d2 * w) as f64;
    }
}

#[inline(always)]
fn raw_moments_accumulate8<T: F32x8Backend + Copy>(
    fm_s: &mut GenericF32x8<T>,
    fm_d: &mut GenericF32x8<T>,
    fm_s2: &mut GenericF32x8<T>,
    fm_d2: &mut GenericF32x8<T>,
    fm_dd: &mut GenericF32x8<T>,
    fm_ds: &mut GenericF32x8<T>,
    s: GenericF32x8<T>,
    d: GenericF32x8<T>,
) {
    *fm_s = *fm_s + s;
    *fm_d = *fm_d + d;
    // Revision 1's two raw second moments, spelled so they stay BIT-identical:
    // `s * s` is still rounded before the add, and the add is still plain.
    *fm_s2 = *fm_s2 + s * s;
    *fm_d2 = *fm_d2 + d * d;
    // Revision 2's PAIRED second moment, `Σ(d−s)(d+s) = Σ(d²−s²)`, formed per
    // pixel so the cancellation happens on ONE pixel's small difference
    // instead of between two large totals. Accumulated unconditionally: it is
    // four cheap ops on registers already live, and an extra accumulator
    // cannot move revision 1's outputs, which is what keeps R1 byte-exact.
    let df = d - s;
    *fm_dd = *fm_dd + df * (d + s);
    // …and revision 2's paired FIRST difference. Both terms of
    // `gvar2 - gvar1` have to come from per-pixel differences, not from
    // differencing two large totals: MEASURED, fixing only the second moment
    // moved the paired disagreement 9.12 % -> 8.32 % and left `GLOBAL_CGAIN`
    // flat (68 -> 69 cells past the bar), because `(md - ms)` was still
    // `(sum_d - sum_s)/n` and carried the same `mean^2/gvar` amplification.
    *fm_ds = *fm_ds + df;
}

/// Reduce the four lane accumulators to scalars and add them into `acc`'s
/// running f64 sums — the band's-last-inner-row finish step. Bit-identical
/// to the hand-inlined `acc.sum_s += fm_s.reduce_add() as f64; …` it
/// replaces.
#[inline(always)]
fn raw_moments_finish8<T: F32x8Backend + Copy>(
    acc: &mut StripChannelAccum,
    fm_s: GenericF32x8<T>,
    fm_d: GenericF32x8<T>,
    fm_s2: GenericF32x8<T>,
    fm_d2: GenericF32x8<T>,
    fm_dd: GenericF32x8<T>,
    fm_ds: GenericF32x8<T>,
) {
    acc.sum_s += fm_s.reduce_add() as f64;
    acc.sum_d += fm_d.reduce_add() as f64;
    acc.sum_s2 += fm_s2.reduce_add() as f64;
    acc.sum_d2 += fm_d2.reduce_add() as f64;
    acc.sum_dd += fm_dd.reduce_add() as f64;
    acc.sum_ds += fm_ds.reduce_add() as f64;
}

/// 16-lane sibling of [`raw_moments_accumulate8`] — `_v4`'s and `_v4x`'s
/// native f32x16 main loops (x86-64 only, hence the arch gate matching
/// `f32x16`'s own import).
#[cfg(target_arch = "x86_64")]
// Reachable only through `incant!`'s AVX-512 tiers, which a build
// without the default `avx512` feature never dispatches — rustc then
// reports the 16-lane helpers as dead even though `_v4`/`_v4x` still
// compile and call them. Annotated rather than `cfg`-gated: gating
// them off breaks those bodies (measured — `cannot find function`).
#[allow(dead_code)]
#[inline(always)]
fn raw_moments_accumulate16<T: F32x16Backend + Copy>(
    fm_s: &mut f32x16<T>,
    fm_d: &mut f32x16<T>,
    fm_s2: &mut f32x16<T>,
    fm_d2: &mut f32x16<T>,
    fm_dd: &mut f32x16<T>,
    fm_ds: &mut f32x16<T>,
    s: f32x16<T>,
    d: f32x16<T>,
) {
    *fm_s = *fm_s + s;
    *fm_d = *fm_d + d;
    // Revision 1's two raw second moments, spelled so they stay BIT-identical:
    // `s * s` is still rounded before the add, and the add is still plain.
    *fm_s2 = *fm_s2 + s * s;
    *fm_d2 = *fm_d2 + d * d;
    // Revision 2's PAIRED second moment, `Σ(d−s)(d+s) = Σ(d²−s²)`, formed per
    // pixel so the cancellation happens on ONE pixel's small difference
    // instead of between two large totals. Accumulated unconditionally: it is
    // four cheap ops on registers already live, and an extra accumulator
    // cannot move revision 1's outputs, which is what keeps R1 byte-exact.
    let df = d - s;
    *fm_dd = *fm_dd + df * (d + s);
    // …and revision 2's paired FIRST difference. Both terms of
    // `gvar2 - gvar1` have to come from per-pixel differences, not from
    // differencing two large totals: MEASURED, fixing only the second moment
    // moved the paired disagreement 9.12 % -> 8.32 % and left `GLOBAL_CGAIN`
    // flat (68 -> 69 cells past the bar), because `(md - ms)` was still
    // `(sum_d - sum_s)/n` and carried the same `mean^2/gvar` amplification.
    *fm_ds = *fm_ds + df;
}

/// 16-lane sibling of [`raw_moments_finish8`].
#[cfg(target_arch = "x86_64")]
// Reachable only through `incant!`'s AVX-512 tiers, which a build
// without the default `avx512` feature never dispatches — rustc then
// reports the 16-lane helpers as dead even though `_v4`/`_v4x` still
// compile and call them. Annotated rather than `cfg`-gated: gating
// them off breaks those bodies (measured — `cannot find function`).
#[allow(dead_code)]
#[inline(always)]
fn raw_moments_finish16<T: F32x16Backend + Copy>(
    acc: &mut StripChannelAccum,
    fm_s: f32x16<T>,
    fm_d: f32x16<T>,
    fm_s2: f32x16<T>,
    fm_d2: f32x16<T>,
    fm_dd: f32x16<T>,
    fm_ds: f32x16<T>,
) {
    acc.sum_s += fm_s.reduce_add() as f64;
    acc.sum_d += fm_d.reduce_add() as f64;
    acc.sum_s2 += fm_s2.reduce_add() as f64;
    acc.sum_d2 += fm_d2.reduce_add() as f64;
    acc.sum_dd += fm_dd.reduce_add() as f64;
    acc.sum_ds += fm_ds.reduce_add() as f64;
}

/// Scalar sibling of [`raw_moments_accumulate8`] / [`raw_moments_accumulate16`]
/// — every tier's remainder-columns tail (width not a multiple of its
/// vector width) falls back to plain `f32`.
#[inline(always)]
fn raw_moments_accumulate_scalar(
    fm_s: &mut f32,
    fm_d: &mut f32,
    fm_s2: &mut f32,
    fm_d2: &mut f32,
    fm_dd: &mut f32,
    fm_ds: &mut f32,
    s: f32,
    d: f32,
) {
    *fm_s += s;
    *fm_d += d;
    *fm_s2 += s * s;
    *fm_d2 += d * d;
    let df = d - s;
    *fm_dd += df * (d + s);
    *fm_ds += df;
}

/// Scalar sibling of [`raw_moments_finish8`] / [`raw_moments_finish16`].
#[inline(always)]
fn raw_moments_finish_scalar(
    acc: &mut StripChannelAccum,
    fm_s: f32,
    fm_d: f32,
    fm_s2: f32,
    fm_d2: f32,
    fm_dd: f32,
    fm_ds: f32,
) {
    acc.sum_s += fm_s as f64;
    acc.sum_d += fm_d as f64;
    acc.sum_s2 += fm_s2 as f64;
    acc.sum_d2 += fm_d2 as f64;
    acc.sum_dd += fm_dd as f64;
    acc.sum_ds += fm_ds as f64;
}

// ============================================================
// Free BOUNDED-ERROR accumulation (class C) — shared across every SIMD tier
// (`benchmarks/free_features_classC_2026-09-04.md`)
// ============================================================
//
// The SECOND free tranche this kernel can carry, and the extension point the
// raw-moments note above was written for: `Σ mse_i` and the two luminance-
// binned weighted sums of the same `mse_i`, where
//
//     mse_i = sat((s − d)², C_MSE)          [`feature_v2::C_MSE`]
//     t     = sat(ref_Y,     C_LUM_T)       [`feature_v2::C_LUM_T`]
//
// are per-pixel functions of values ALREADY LIVE in this kernel's registers:
// `pd = s − d` is computed one line above for `acc.mse`, and for the Y
// channel the reference-luma plane IS this channel's `src` (the append
// kernel keeps `ref_y` a separate argument only so the weight is explicitly
// a function of the reference — `feature_v2::csfw_block_kernel_generic`'s
// own note). So no new plane, no new load, no new pass — the class-C
// definition.
//
// They finalize four families of 944 slots the v1-only walk otherwise leaves
// at their structural zeros:
//   * v2-348 `MSE`      (12 cells) — `clamp01(Σ mse_i / n)`
//   * append `LUM_DARK_ERR` / `LUM_MID_ERR` / `LUM_BRIGHT_ERR` (Y, 4 scales
//     = 12 slots) — the Bernstein-partition dark/bright weighted means, with
//     mid DERIVED at finalize exactly as `finish_append` derives it.
// No slot is renumbered and none is invented: these are existing 944-layout
// positions, filled by a cheaper route (the append-only feature-numbering
// discipline).
//
// Same `#[inline(always)]`-generic-over-a-backend-trait shape, and the same
// MEASURED reason for it, as the raw-moments pair above — see that comment;
// `#[rite]` cannot apply to a function generic over a backend TRAIT.

/// `feature_v2::C_MSE` as `f32`. Duplicated (not imported) because
/// `feature_v2` is behind the `feature-regime-v2` feature while this module
/// is unconditional; `feature_v2::tests::class_c_kernel_constants_match`
/// fails the build if the two ever disagree.
pub(crate) const C_MSE_F32: f32 = 0.01;
/// `feature_v2::C_LUM_T` as `f32`. Same duplication contract as
/// [`C_MSE_F32`].
pub(crate) const C_LUM_T_F32: f32 = 0.35;

/// Which FREE extra accumulators the fused v1 kernel carries this call.
///
/// Replaces the previous bare boolean `raw_moments` parameter: the class-C
/// tranche adds two more independently-requestable accumulator groups, and
/// one `Copy` struct keeps every kernel signature at ONE free-extras
/// parameter instead of three parallel bools. All-false is the pre-existing
/// instruction sequence, unchanged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FreeExtrasWork {
    /// Explicit per-request SSIM arithmetic; None retains research defaults.
    pub revision: Option<crate::feature_defs::FormulaRevision>,
    /// Skip peak and whole-plane HF reductions when no declared input reads them.
    pub local_only: bool,
    /// Rev5 plans may need HF ratios while omitting every peak.
    pub omit_peaks: bool,
    /// SSIM and MSE subset: omit artifact/detail reductions as well.
    pub omit_edges: bool,
    /// Σs, Σd, Σs², Σd² — the `V1FreeExtras::RawMoments` set.
    pub raw_moments: bool,
    /// Σ `mse_i` — the bounded per-pixel error. Every channel.
    pub bounded_err: bool,
    /// The dark/bright luminance-binned weighted sums of the SAME `mse_i`.
    /// Y channel only (`feature_v2::APPEND2_CHANNEL`), because that is the
    /// only channel whose `src` plane IS the reference luma the weight
    /// reads. Requires `bounded_err` (the weights multiply its `mse_i`).
    pub lum_bins: bool,
}

impl FreeExtrasWork {
    pub(crate) fn revision(self) -> crate::feature_defs::FormulaRevision {
        crate::ssim_form::effective_revision(
            self.revision
                .unwrap_or_else(crate::ssim_form::active_revision),
        )
    }
    fn luma_form(self) -> crate::ssim_form::SsimLumaForm {
        let revision = self.revision();
        if revision == crate::ssim_form::active_revision() {
            crate::ssim_form::active_luma_form()
        } else {
            crate::ssim_form::SsimLumaForm::for_revision(revision)
        }
    }
}

/// Accumulate one row's bounded per-pixel error into the lane sum and return
/// it, so the luminance bins can weight the SAME value rather than
/// recomputing it. `sat(x, c) = max(x,0)/(max(x,0)+c)` — the vector form of
/// `feature_v2::saturate`, written out here for the same
/// module-independence reason as [`C_MSE_F32`]. The `max(0)` is a no-op for
/// a real square and is kept only so a NaN input maps the same way the f64
/// definition maps it.
#[inline(always)]
fn bounded_err_accumulate8<T: F32x8Backend + Copy>(
    token: T,
    be_m: &mut GenericF32x8<T>,
    pd: GenericF32x8<T>,
) -> GenericF32x8<T> {
    let sq = (pd * pd).max(GenericF32x8::<T>::zero_t(token));
    let m = sq / (sq + GenericF32x8::<T>::splat_t(token, C_MSE_F32));
    *be_m = *be_m + m;
    m
}

/// Accumulate the dark/bright Bernstein-weighted numerator + denominator
/// lane sums for one row. `s` is the Y channel's source row (= the reference
/// luma); `m` is [`bounded_err_accumulate8`]'s per-pixel value.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn lum_bins_accumulate8<T: F32x8Backend + Copy>(
    token: T,
    wd_num: &mut GenericF32x8<T>,
    wd_den: &mut GenericF32x8<T>,
    wb_num: &mut GenericF32x8<T>,
    wb_den: &mut GenericF32x8<T>,
    s: GenericF32x8<T>,
    m: GenericF32x8<T>,
) {
    let ry = s.max(GenericF32x8::<T>::zero_t(token));
    let t = ry / (ry + GenericF32x8::<T>::splat_t(token, C_LUM_T_F32));
    let one_mt = GenericF32x8::<T>::splat_t(token, 1.0) - t;
    let wd = one_mt * one_mt;
    let wb = t * t;
    *wd_num = *wd_num + wd * m;
    *wd_den = *wd_den + wd;
    *wb_num = *wb_num + wb * m;
    *wb_den = *wb_den + wb;
}

/// Band's-last-inner-row finish for [`bounded_err_accumulate8`].
#[inline(always)]
fn bounded_err_finish8<T: F32x8Backend + Copy>(acc: &mut StripChannelAccum, be_m: GenericF32x8<T>) {
    acc.sum_msat += be_m.reduce_add() as f64;
}

/// Band's-last-inner-row finish for [`lum_bins_accumulate8`].
#[inline(always)]
fn lum_bins_finish8<T: F32x8Backend + Copy>(
    acc: &mut StripChannelAccum,
    wd_num: GenericF32x8<T>,
    wd_den: GenericF32x8<T>,
    wb_num: GenericF32x8<T>,
    wb_den: GenericF32x8<T>,
) {
    acc.lum_wd_num += wd_num.reduce_add() as f64;
    acc.lum_wd_den += wd_den.reduce_add() as f64;
    acc.lum_wb_num += wb_num.reduce_add() as f64;
    acc.lum_wb_den += wb_den.reduce_add() as f64;
}

/// 16-lane sibling of [`bounded_err_accumulate8`].
#[cfg(target_arch = "x86_64")]
// Reachable only through `incant!`'s AVX-512 tiers, which a build
// without the default `avx512` feature never dispatches — rustc then
// reports the 16-lane helpers as dead even though `_v4`/`_v4x` still
// compile and call them. Annotated rather than `cfg`-gated: gating
// them off breaks those bodies (measured — `cannot find function`).
#[allow(dead_code)]
#[inline(always)]
fn bounded_err_accumulate16<T: F32x16Backend + Copy>(
    token: T,
    be_m: &mut f32x16<T>,
    pd: f32x16<T>,
) -> f32x16<T> {
    let sq = (pd * pd).max(f32x16::<T>::zero_t(token));
    let m = sq / (sq + f32x16::<T>::splat_t(token, C_MSE_F32));
    *be_m = *be_m + m;
    m
}

/// 16-lane sibling of [`lum_bins_accumulate8`].
#[cfg(target_arch = "x86_64")]
// Reachable only through `incant!`'s AVX-512 tiers, which a build
// without the default `avx512` feature never dispatches — rustc then
// reports the 16-lane helpers as dead even though `_v4`/`_v4x` still
// compile and call them. Annotated rather than `cfg`-gated: gating
// them off breaks those bodies (measured — `cannot find function`).
#[allow(dead_code)]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn lum_bins_accumulate16<T: F32x16Backend + Copy>(
    token: T,
    wd_num: &mut f32x16<T>,
    wd_den: &mut f32x16<T>,
    wb_num: &mut f32x16<T>,
    wb_den: &mut f32x16<T>,
    s: f32x16<T>,
    m: f32x16<T>,
) {
    let ry = s.max(f32x16::<T>::zero_t(token));
    let t = ry / (ry + f32x16::<T>::splat_t(token, C_LUM_T_F32));
    let one_mt = f32x16::<T>::splat_t(token, 1.0) - t;
    let wd = one_mt * one_mt;
    let wb = t * t;
    *wd_num = *wd_num + wd * m;
    *wd_den = *wd_den + wd;
    *wb_num = *wb_num + wb * m;
    *wb_den = *wb_den + wb;
}

/// 16-lane sibling of [`bounded_err_finish8`].
#[cfg(target_arch = "x86_64")]
// Reachable only through `incant!`'s AVX-512 tiers, which a build
// without the default `avx512` feature never dispatches — rustc then
// reports the 16-lane helpers as dead even though `_v4`/`_v4x` still
// compile and call them. Annotated rather than `cfg`-gated: gating
// them off breaks those bodies (measured — `cannot find function`).
#[allow(dead_code)]
#[inline(always)]
fn bounded_err_finish16<T: F32x16Backend + Copy>(acc: &mut StripChannelAccum, be_m: f32x16<T>) {
    acc.sum_msat += be_m.reduce_add() as f64;
}

/// 16-lane sibling of [`lum_bins_finish8`].
#[cfg(target_arch = "x86_64")]
// Reachable only through `incant!`'s AVX-512 tiers, which a build
// without the default `avx512` feature never dispatches — rustc then
// reports the 16-lane helpers as dead even though `_v4`/`_v4x` still
// compile and call them. Annotated rather than `cfg`-gated: gating
// them off breaks those bodies (measured — `cannot find function`).
#[allow(dead_code)]
#[inline(always)]
fn lum_bins_finish16<T: F32x16Backend + Copy>(
    acc: &mut StripChannelAccum,
    wd_num: f32x16<T>,
    wd_den: f32x16<T>,
    wb_num: f32x16<T>,
    wb_den: f32x16<T>,
) {
    acc.lum_wd_num += wd_num.reduce_add() as f64;
    acc.lum_wd_den += wd_den.reduce_add() as f64;
    acc.lum_wb_num += wb_num.reduce_add() as f64;
    acc.lum_wb_den += wb_den.reduce_add() as f64;
}

/// Scalar sibling of [`bounded_err_accumulate8`] — the remainder-columns tail.
#[inline(always)]
fn bounded_err_accumulate_scalar(be_m: &mut f32, pd: f32) -> f32 {
    let sq = (pd * pd).max(0.0);
    let m = sq / (sq + C_MSE_F32);
    *be_m += m;
    m
}

/// Scalar sibling of [`lum_bins_accumulate8`].
#[inline(always)]
fn lum_bins_accumulate_scalar(
    wd_num: &mut f32,
    wd_den: &mut f32,
    wb_num: &mut f32,
    wb_den: &mut f32,
    s: f32,
    m: f32,
) {
    let ry = s.max(0.0);
    let t = ry / (ry + C_LUM_T_F32);
    let one_mt = 1.0 - t;
    let wd = one_mt * one_mt;
    let wb = t * t;
    *wd_num += wd * m;
    *wd_den += wd;
    *wb_num += wb * m;
    *wb_den += wb;
}

/// Scalar sibling of [`bounded_err_finish8`].
#[inline(always)]
fn bounded_err_finish_scalar(acc: &mut StripChannelAccum, be_m: f32) {
    acc.sum_msat += be_m as f64;
}

/// Scalar sibling of [`lum_bins_finish8`].
#[inline(always)]
fn lum_bins_finish_scalar(
    acc: &mut StripChannelAccum,
    wd_num: f32,
    wd_den: f32,
    wb_num: f32,
    wb_den: f32,
) {
    acc.lum_wd_num += wd_num as f64;
    acc.lum_wd_den += wd_den as f64;
    acc.lum_wb_num += wb_num as f64;
    acc.lum_wb_den += wb_den as f64;
}

/// Test-only handles on the class-C scalar integrands, so
/// `feature_v2::tests::class_c_integrands_match_the_f64_scalar_oracle` can
/// gate the arithmetic itself against the f64 `saturate` the append kernel
/// calls — rather than only observing it end-to-end through a whole walk.
/// The vector tiers are gated end-to-end instead (the geometry list in
/// `class_c_extras_match_the_944_walk` covers every tier's main loop AND
/// its 8-lane / scalar remainder).
#[cfg(all(test, feature = "feature-regime-v2"))] // callers live in feature_v2's tests
pub(crate) fn test_only_bounded_err_scalar(be_m: &mut f32, pd: f32) -> f32 {
    bounded_err_accumulate_scalar(be_m, pd)
}

/// Test-only handle on [`lum_bins_accumulate_scalar`]. See
/// [`test_only_bounded_err_scalar`].
#[cfg(all(test, feature = "feature-regime-v2"))] // callers live in feature_v2's tests
pub(crate) fn test_only_lum_bins_scalar(
    wd_num: &mut f32,
    wd_den: &mut f32,
    wb_num: &mut f32,
    wb_den: &mut f32,
    s: f32,
    m: f32,
) {
    lum_bins_accumulate_scalar(wd_num, wd_den, wb_num, wb_den, s, m)
}

/// Accumulated feature sums from a fused V-blur + feature extraction pass.
/// All values are raw sums (not yet divided by pixel count).
pub(crate) struct StripChannelAccum {
    pub ssim_d: f64,
    pub ssim_d4: f64,
    pub ssim_d2: f64,
    pub edge_art: f64,
    pub edge_art4: f64,
    pub edge_art2: f64,
    pub edge_det: f64,
    pub edge_det4: f64,
    pub edge_det2: f64,
    pub mse: f64,
    pub hf_sq_src: f64,
    pub hf_sq_dst: f64,
    pub hf_abs_src: f64,
    pub hf_abs_dst: f64,
    // Extended: L8 power pool and max
    pub ssim_d8: f64,
    pub edge_art8: f64,
    pub edge_det8: f64,
    pub ssim_max: f32,
    pub edge_art_max: f32,
    pub edge_det_max: f32,
    // --- Free raw moments (`raw_moments`; zero and untouched otherwise) ---
    // Σsrc, Σdst, Σsrc², Σdst² over the inner rows. Written ONLY when the
    // caller asks; every existing field's value and accumulation order is
    // unchanged whether it asks or not.
    pub sum_s: f64,
    pub sum_d: f64,
    pub sum_s2: f64,
    pub sum_d2: f64,
    /// Revision 2's paired second moment `Σ(d−s)(d+s) = Σ(d²−s²)`, formed
    /// per pixel. See [`raw_moments_accumulate16`] for why it exists.
    pub sum_dd: f64,
    /// Revision 2's paired first difference `Σ(d−s)`.
    pub sum_ds: f64,
    // --- Free bounded error (`FreeExtrasWork::bounded_err` / `lum_bins`;
    //     zero and untouched otherwise) ---
    /// Σ `sat((s−d)², C_MSE)` — the v2-348 `MSE` slot's numerator.
    pub sum_msat: f64,
    /// Σ `(1−t)²·mse_i` and Σ `(1−t)²` — `LUM_DARK_ERR`'s weighted mean.
    pub lum_wd_num: f64,
    pub lum_wd_den: f64,
    /// Σ `t²·mse_i` and Σ `t²` — `LUM_BRIGHT_ERR`'s weighted mean.
    /// (`LUM_MID_ERR` is DERIVED from these plus `sum_msat` and `n`, exactly
    /// as `feature_v2::finish_append` derives it.)
    pub lum_wb_num: f64,
    pub lum_wb_den: f64,
    // Fused v1-extension pools (`ExtPoolsWork`), revision 3 only.
    pub masked_ssim_d: f64,
    pub masked_ssim_d4: f64,
    pub masked_ssim_d2: f64,
    pub iw_ssim_d: f64,
    pub iw_ssim_d4: f64,
    pub iw_ssim_d2: f64,
    pub masked_art4: f64,
    pub masked_det4: f64,
    pub iw_art4: f64,
    pub iw_det4: f64,
    pub masked_mse: f64,
    pub iw_mse: f64,
    /// `sum activity` over the inner rows (the IW normaliser's numerator).
    pub act_sum: f64,
}

impl StripChannelAccum {
    pub fn zero() -> Self {
        Self {
            ssim_d: 0.0,
            ssim_d4: 0.0,
            ssim_d2: 0.0,
            edge_art: 0.0,
            edge_art4: 0.0,
            edge_art2: 0.0,
            edge_det: 0.0,
            edge_det4: 0.0,
            edge_det2: 0.0,
            mse: 0.0,
            hf_sq_src: 0.0,
            hf_sq_dst: 0.0,
            hf_abs_src: 0.0,
            hf_abs_dst: 0.0,
            ssim_d8: 0.0,
            edge_art8: 0.0,
            edge_det8: 0.0,
            ssim_max: 0.0,
            edge_art_max: 0.0,
            edge_det_max: 0.0,
            sum_s: 0.0,
            sum_dd: 0.0,
            sum_ds: 0.0,
            sum_d: 0.0,
            sum_s2: 0.0,
            sum_d2: 0.0,
            sum_msat: 0.0,
            lum_wd_num: 0.0,
            lum_wd_den: 0.0,
            lum_wb_num: 0.0,
            lum_wb_den: 0.0,
            masked_ssim_d: 0.0,
            masked_ssim_d4: 0.0,
            masked_ssim_d2: 0.0,
            iw_ssim_d: 0.0,
            iw_ssim_d4: 0.0,
            iw_ssim_d2: 0.0,
            masked_art4: 0.0,
            masked_det4: 0.0,
            iw_art4: 0.0,
            iw_det4: 0.0,
            masked_mse: 0.0,
            iw_mse: 0.0,
            act_sum: 0.0,
        }
    }
}

/// Fused V-blur + ALL feature extraction for SSIM channels.
///
/// Reads 6 inputs: 4 H-blurred planes (h_mu1, h_mu2, h_sigma_sq, h_sigma12) + raw src + dst.
/// Maintains 4 V-blur running sums per column group.
/// At each inner row, computes SSIM, edge, variance, texture, and MSE features
/// directly from register values — V-blur outputs never touch memory.
///
/// Returns accumulated feature sums for the inner rows of this strip.
pub(crate) fn fused_vblur_features_ssim(
    h_mu1: &[f32],
    h_mu2: &[f32],
    h_sigma_sq: &[f32],
    h_sigma12: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    sd_out: &mut [f32],
    store_sd: bool,
    // `ssq_out` / `s12_out`: the V-blurred `h_sigma_sq` / `h_sigma12` for the
    // inner rows — the exact planes `box_blur_v_from_copy` would produce from
    // the same inputs, taken straight out of the registers instead of a second
    // sweep. Used by the v1-pool replay's masked/IW SSIM slots (which is why
    // BIT-identity, not closeness, is required: `folded720_v1_pools_match_v1_path`
    // compares the pool slots to v1's 372 with `to_bits()`).
    //
    // Why they are bit-identical: each column's V-blur is an INDEPENDENT scalar
    // recurrence (`sum += src[add] - src[rem]`, then `sum * (1.0 / diam)`), so
    // lane width cannot change a value and only the index sequence can. Init
    // (`mirror_idx`) and `rem_idx` are written the same way in both kernels;
    // the ONE textual difference is the bottom-edge `add_idx` fold — this
    // kernel's `vblur_add_idx` takes `|2·(h−1) − add_raw|`, `box_blur_v`'s
    // takes `saturating_sub`. They differ only when `2·(h−1) < y + r + 1` for
    // some `y < h`, i.e. only when `h < r + 2` (= 7 at BLUR_RADIUS 5). The
    // folded walk reflect-pads to a 64px floor and runs 4 pyramid scales, so
    // the smallest plane it ever V-blurs is 8 rows and every band buffer is at
    // least that tall — the divergent branch is unreachable on this path.
    // (Not a general guarantee: a caller that V-blurs a <7-row plane must keep
    // using `box_blur_v_from_copy` if it needs v1 parity.)
    ssq_out: &mut [f32],
    s12_out: &mut [f32],
    store_sigma: bool,
    // `free`: which FREE extra accumulators to carry alongside the existing
    // sums — the four raw moments (Σs, Σd, Σs², Σd²) and/or the class-C
    // bounded-error family. See [`FreeExtrasWork`] and
    // `StripChannelAccum::sum_s` / `sum_msat`.
    free: FreeExtrasWork,
    ext: ExtPoolsWork,
    h_act: &[f32],
) -> StripChannelAccum {
    // Revision 3 is FUSED: the H pass already carries `Σ(a-b)²` in the
    // `h_sigma12` plane (see `blur::fused_blur_h_ssim`), so the V pass forms
    // the direct-error dissimilarity from the same four V-blurred planes it
    // always had — no second traversal. The exact f64 second-pass kernel this
    // replaced measured at 22% of the whole walk (perf, fold944_full@2048^2)
    // and is kept in `ssim_form` as the reference the bounded-error tests
    // measure against.
    let direct = free.revision() >= crate::feature_defs::FormulaRevision::Rev3;
    // Canonical arithmetic follows THIS computation's revision (featcanon D1).
    let mode = crate::featcanon::mode(free.revision());
    match mode {
        #[cfg(feature = "oracle")]
        crate::featcanon::Mode::Exact => {
            return fused_vblur_ssim_exact(
                h_mu1,
                h_mu2,
                h_sigma_sq,
                h_sigma12,
                src,
                dst,
                width,
                height,
                inner_start,
                inner_h,
                radius,
                mu1_out,
                mu2_out,
                store_mu,
                sd_out,
                store_sd,
                ssq_out,
                s12_out,
                store_sigma,
                free,
                direct,
                ext,
                h_act,
            );
        }
        #[cfg(feature = "oracle")]
        crate::featcanon::Mode::Canon32 => {
            return fused_vblur_ssim_canon::<crate::featcanon::LanesF32>(
                h_mu1,
                h_mu2,
                h_sigma_sq,
                h_sigma12,
                src,
                dst,
                width,
                height,
                inner_start,
                inner_h,
                radius,
                mu1_out,
                mu2_out,
                store_mu,
                sd_out,
                store_sd,
                ssq_out,
                s12_out,
                store_sigma,
                free,
                direct,
                ext,
                h_act,
                mode,
            );
        }
        crate::featcanon::Mode::Canon64 => {
            // rev4vec2: the Rec64 axis runs the chunked `f32x8`/`f64x8` body on
            // the fused-FMA tiers — bit-identical to the `LanesF64` scalar
            // canonical body. Rev5's `localwin` axis runs the per-output pair
            // tree body. Any OTHER blur axis (an oracle `BLUR` override)
            // keeps the scalar body, which honours it.
            let blur_axis = crate::featcanon::canon_blur_axis_for(mode, free.revision());
            if matches!(blur_axis, crate::featcanon::BlurMode::Local) {
                return incant!(
                    fused_vblur_ssim_canon64_vec::<crate::featcanon::Lanes16F64>(
                        h_mu1,
                        h_mu2,
                        h_sigma_sq,
                        h_sigma12,
                        src,
                        dst,
                        width,
                        height,
                        inner_start,
                        inner_h,
                        radius,
                        mu1_out,
                        mu2_out,
                        store_mu,
                        sd_out,
                        store_sd,
                        ssq_out,
                        s12_out,
                        store_sigma,
                        free,
                        direct,
                        ext,
                        h_act,
                        mode,
                    ),
                    [v4x, v4, v3, neon, wasm128, scalar]
                );
            }
            if matches!(blur_axis, crate::featcanon::BlurMode::Rec64) {
                return incant!(
                    fused_vblur_ssim_canon64_vec::<crate::featcanon::LanesF64>(
                        h_mu1,
                        h_mu2,
                        h_sigma_sq,
                        h_sigma12,
                        src,
                        dst,
                        width,
                        height,
                        inner_start,
                        inner_h,
                        radius,
                        mu1_out,
                        mu2_out,
                        store_mu,
                        sd_out,
                        store_sd,
                        ssq_out,
                        s12_out,
                        store_sigma,
                        free,
                        direct,
                        ext,
                        h_act,
                        mode,
                    ),
                    [v4x, v4, v3, neon, wasm128, scalar]
                );
            }
            return fused_vblur_ssim_canon::<crate::featcanon::LanesF64>(
                h_mu1,
                h_mu2,
                h_sigma_sq,
                h_sigma12,
                src,
                dst,
                width,
                height,
                inner_start,
                inner_h,
                radius,
                mu1_out,
                mu2_out,
                store_mu,
                sd_out,
                store_sd,
                ssq_out,
                s12_out,
                store_sigma,
                free,
                direct,
                ext,
                h_act,
                mode,
            );
        }
        #[cfg(feature = "oracle")]
        crate::featcanon::Mode::CanonNeum => {
            return fused_vblur_ssim_canon::<crate::featcanon::Neum64>(
                h_mu1,
                h_mu2,
                h_sigma_sq,
                h_sigma12,
                src,
                dst,
                width,
                height,
                inner_start,
                inner_h,
                radius,
                mu1_out,
                mu2_out,
                store_mu,
                sd_out,
                store_sd,
                ssq_out,
                s12_out,
                store_sigma,
                free,
                direct,
                ext,
                h_act,
                mode,
            );
        }
        crate::featcanon::Mode::Off => {}
    }
    incant!(
        fused_vblur_ssim_inner(
            h_mu1,
            h_mu2,
            h_sigma_sq,
            h_sigma12,
            src,
            dst,
            width,
            height,
            inner_start,
            inner_h,
            radius,
            mu1_out,
            mu2_out,
            store_mu,
            sd_out,
            store_sd,
            ssq_out,
            s12_out,
            store_sigma,
            free,
            direct,
            ext,
            h_act,
        ),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

/// Fused V-blur + feature extraction for edge-only channels (no SSIM).
///
/// Reads 4 inputs: 2 H-blurred planes (h_mu1, h_mu2) + raw src + dst.
/// Maintains 2 V-blur running sums per column group.
/// Computes edge, variance, texture, and MSE features inline.
///
/// `revision` is the computation's formula revision; it selects the canonical
/// body at Rev4 (featcanon D1) and nothing else. Revision 7 changes the edge
/// ratio's form, but no canonical revision takes this route.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fused_vblur_features_edge(
    h_mu1: &[f32],
    h_mu2: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    revision: crate::feature_defs::FormulaRevision,
) -> StripChannelAccum {
    // Canonical revisions route every channel through the fused SSIM kernels
    // (`streaming::active_channels`), so revision 7 never reaches this route;
    // its edge ratio here keeps the subtractive form.
    assert!(
        !revision.stable_forms(),
        "the edge-only route has no revision-7 form; canonical revisions use the SSIM route"
    );
    let mode = crate::featcanon::mode(crate::ssim_form::effective_revision(revision));
    match mode {
        #[cfg(feature = "oracle")]
        crate::featcanon::Mode::Exact => {
            return fused_vblur_edge_exact(
                h_mu1,
                h_mu2,
                src,
                dst,
                width,
                height,
                inner_start,
                inner_h,
                radius,
                mu1_out,
                mu2_out,
                store_mu,
            );
        }
        #[cfg(feature = "oracle")]
        crate::featcanon::Mode::Canon32 => {
            return fused_vblur_edge_canon::<crate::featcanon::LanesF32>(
                h_mu1,
                h_mu2,
                src,
                dst,
                width,
                height,
                inner_start,
                inner_h,
                radius,
                mu1_out,
                mu2_out,
                store_mu,
                mode,
            );
        }
        crate::featcanon::Mode::Canon64 => {
            // rev4vec2: Rec64 → the chunked f32x8/f64x8 body on the fused-FMA
            // tiers; any other (oracle) axis keeps the scalar canonical body.
            if matches!(
                crate::featcanon::canon_blur_axis(mode),
                crate::featcanon::BlurMode::Rec64
            ) {
                return incant!(
                    fused_vblur_edge_canon64_vec(
                        h_mu1,
                        h_mu2,
                        src,
                        dst,
                        width,
                        height,
                        inner_start,
                        inner_h,
                        radius,
                        mu1_out,
                        mu2_out,
                        store_mu,
                        mode,
                    ),
                    [v4x, v4, v3, neon, wasm128, scalar]
                );
            }
            return fused_vblur_edge_canon::<crate::featcanon::LanesF64>(
                h_mu1,
                h_mu2,
                src,
                dst,
                width,
                height,
                inner_start,
                inner_h,
                radius,
                mu1_out,
                mu2_out,
                store_mu,
                mode,
            );
        }
        #[cfg(feature = "oracle")]
        crate::featcanon::Mode::CanonNeum => {
            return fused_vblur_edge_canon::<crate::featcanon::Neum64>(
                h_mu1,
                h_mu2,
                src,
                dst,
                width,
                height,
                inner_start,
                inner_h,
                radius,
                mu1_out,
                mu2_out,
                store_mu,
                mode,
            );
        }
        crate::featcanon::Mode::Off => {}
    }
    incant!(
        fused_vblur_edge_inner(
            h_mu1,
            h_mu2,
            src,
            dst,
            width,
            height,
            inner_start,
            inner_h,
            radius,
            mu1_out,
            mu2_out,
            store_mu
        ),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

// ============================================================
// Helper: mirror-reflect row index for V-blur boundary handling
// ============================================================

#[inline(always)]
fn mirror_idx(i: usize, r: usize, height: usize) -> usize {
    if i <= r {
        (r - i).min(height - 1)
    } else {
        (i - r).min(height - 1)
    }
}

#[inline(always)]
fn vblur_add_idx(y: usize, r: usize, height: usize) -> usize {
    let add_raw = y + r + 1;
    if add_raw < height {
        add_raw
    } else {
        // Mirror-reflect: fold back from the boundary.
        // Use signed math to avoid underflow when add_raw >> height.
        let reflected = 2 * (height as isize - 1) - add_raw as isize;
        reflected.unsigned_abs().min(height - 1)
    }
}

#[inline(always)]
fn vblur_rem_idx(y: usize, r: usize, height: usize) -> usize {
    let rem_i = y as isize - r as isize;
    let idx = if rem_i < 0 {
        rem_i.unsigned_abs()
    } else {
        rem_i as usize
    };
    idx.min(height - 1)
}

// ============================================================
// AVX-512 implementations
// ============================================================

#[cfg(target_arch = "x86_64")]
#[arcane]
fn fused_vblur_ssim_inner_v4(
    token: archmage::X64V4Token,
    h_mu1: &[f32],
    h_mu2: &[f32],
    h_sigma_sq: &[f32],
    h_sigma12: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    sd_out: &mut [f32],
    store_sd: bool,
    ssq_out: &mut [f32],
    s12_out: &mut [f32],
    store_sigma: bool,
    // `free`: which FREE extra accumulators to carry alongside the existing
    // sums — the four raw moments (Σs, Σd, Σs², Σd²) and/or the class-C
    // bounded-error family. See [`FreeExtrasWork`] and
    // `StripChannelAccum::sum_s` / `sum_msat`.
    free: FreeExtrasWork,
    // Revision 3: form the dissimilarity with the direct-error moment that the
    // H pass put in `h_sigma12` (see `blur::fused_blur_h_ssim`).
    direct: bool,
    ext: ExtPoolsWork,
    h_act: &[f32],
) -> StripChannelAccum {
    let form = free.luma_form();
    let diam = 2 * radius + 1;
    let inv_v = f32x16::splat_t(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 16;

    // One slice construction per plane: every row access below is
    // `row_base + col_base` with `row_base = idx * width`, `idx <= height - 1`
    // (all index helpers clamp with `.min(height - 1)`), `col_base <= width - 16`
    // — so `base + 16 <= height * width` is statically provable and the
    // `[base..][..16]` / `[base..base + 16]` accesses compile without the
    // panic-edge recheck per load/store. Same trick as the canon64 bodies.
    let n_px = height * width;
    let h_mu1 = if h_mu1.len() >= n_px {
        &h_mu1[..n_px]
    } else {
        h_mu1
    };
    let h_mu2 = if h_mu2.len() >= n_px {
        &h_mu2[..n_px]
    } else {
        h_mu2
    };
    let h_sigma_sq = if h_sigma_sq.len() >= n_px {
        &h_sigma_sq[..n_px]
    } else {
        h_sigma_sq
    };
    let h_sigma12 = if h_sigma12.len() >= n_px {
        &h_sigma12[..n_px]
    } else {
        h_sigma12
    };
    let h_act = if h_act.len() >= n_px {
        &h_act[..n_px]
    } else {
        h_act
    };
    let src = if src.len() >= n_px { &src[..n_px] } else { src };
    let dst = if dst.len() >= n_px { &dst[..n_px] } else { dst };
    let mu1_out = if mu1_out.len() >= n_px {
        &mut mu1_out[..n_px]
    } else {
        mu1_out
    };
    let mu2_out = if mu2_out.len() >= n_px {
        &mut mu2_out[..n_px]
    } else {
        mu2_out
    };
    let sd_out = if sd_out.len() >= n_px {
        &mut sd_out[..n_px]
    } else {
        sd_out
    };
    let ssq_out = if ssq_out.len() >= n_px {
        &mut ssq_out[..n_px]
    } else {
        ssq_out
    };
    let s12_out = if s12_out.len() >= n_px {
        &mut s12_out[..n_px]
    } else {
        s12_out
    };

    // SSIM constants
    let one = f32x16::splat_t(token, 1.0);
    let zero = f32x16::zero_t(token);

    let mut acc = StripChannelAccum::zero();
    let inner_end = inner_start + inner_h;

    for cg in 0..col_groups {
        let col_base = cg * 16;

        // Initialize 4 running sums for this column group
        let mut sum_m1 = f32x16::zero_t(token);
        let mut sum_m2 = f32x16::zero_t(token);
        let mut sum_sq = f32x16::zero_t(token);
        let mut sum_s12 = f32x16::zero_t(token);
        let mut sum_act = f32x16::zero_t(token);
        // Free raw moments: one lane accumulator per column group, reduced
        // at the band's last inner row (see `raw_moments`). Dead code when
        // the caller did not ask.
        let (mut fm_s, mut fm_d, mut fm_s2, mut fm_d2) = (
            f32x16::zero_t(token),
            f32x16::zero_t(token),
            f32x16::zero_t(token),
            f32x16::zero_t(token),
        );
        // Revision 2\'s paired second moment (see `raw_moments_accumulate16`).
        let mut fm_dd = f32x16::zero_t(token);
        let mut fm_ds = f32x16::zero_t(token);
        // Free bounded-error lane accumulators (`free.bounded_err` /
        // `free.lum_bins`) — same band-batched shape and rationale as
        // `fm_*` above: vector-add every row, reduce ONCE at the band's
        // last inner row.
        let (mut be_m, mut be_wdn, mut be_wdd, mut be_wbn, mut be_wbd) = (
            f32x16::zero_t(token),
            f32x16::zero_t(token),
            f32x16::zero_t(token),
            f32x16::zero_t(token),
            f32x16::zero_t(token),
        );

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            let base = idx * width + col_base;
            let abase = idx * width + col_base;
            sum_m1 = sum_m1 + f32x16::from_array_t(token, h_mu1[base..][..16].try_into().unwrap());
            sum_m2 = sum_m2 + f32x16::from_array_t(token, h_mu2[base..][..16].try_into().unwrap());
            sum_sq =
                sum_sq + f32x16::from_array_t(token, h_sigma_sq[base..][..16].try_into().unwrap());
            sum_s12 =
                sum_s12 + f32x16::from_array_t(token, h_sigma12[base..][..16].try_into().unwrap());
            if ext.on {
                sum_act =
                    sum_act + f32x16::from_array_t(token, h_act[abase..][..16].try_into().unwrap());
            }
        }

        for y in 0..height {
            // Only accumulate features for inner rows
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;

                // V-blurred values (still in registers)
                let mu1 = sum_m1 * inv_v;
                let mu2 = sum_m2 * inv_v;
                let ssq = sum_sq * inv_v;
                let s12 = sum_s12 * inv_v;

                // Load raw pixel values
                let s = f32x16::from_array_t(token, src[base..][..16].try_into().unwrap());
                let d = f32x16::from_array_t(token, dst[base..][..16].try_into().unwrap());

                // === SSIM ===
                let sd = if direct {
                    ssim_direct16(token, form, mu1, mu2, ssq, s12).max(zero)
                } else {
                    (ssim_dissim16(token, form, mu1, mu2, ssq, s12)).max(zero)
                };
                let sd2 = sd * sd;
                let sd4 = sd2 * sd2;
                acc.ssim_d += sd.reduce_add() as f64;
                acc.ssim_d4 += sd4.reduce_add() as f64;
                acc.ssim_d2 += sd2.reduce_add() as f64;
                if !free.local_only {
                    acc.ssim_d8 += (sd4 * sd4).reduce_add() as f64;
                    acc.ssim_max = acc.ssim_max.max(sd.reduce_max());
                }
                if store_sd {
                    sd_out[base..base + 16].copy_from_slice(&sd.to_array());
                }
                if store_mu {
                    mu1_out[base..base + 16].copy_from_slice(&mu1.to_array());
                    mu2_out[base..base + 16].copy_from_slice(&mu2.to_array());
                }
                if store_sigma {
                    ssq_out[base..base + 16].copy_from_slice(&ssq.to_array());
                    s12_out[base..base + 16].copy_from_slice(&s12.to_array());
                }

                // === Edge ===
                let diff1 = (s - mu1).abs();
                let diff2 = (d - mu2).abs();
                let ed = (one + diff2) / (one + diff1) - one;
                let artifact = ed.max(zero);
                let detail_lost = (-ed).max(zero);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                if !free.omit_edges {
                    acc.edge_art += artifact.reduce_add() as f64;
                    acc.edge_art4 += a4.reduce_add() as f64;
                    acc.edge_art2 += a2.reduce_add() as f64;
                    acc.edge_det += detail_lost.reduce_add() as f64;
                    acc.edge_det4 += dl4.reduce_add() as f64;
                    acc.edge_det2 += dl2.reduce_add() as f64;
                    if !free.omit_edges {
                        acc.edge_art8 += (a4 * a4).reduce_add() as f64;
                        acc.edge_det8 += (dl4 * dl4).reduce_add() as f64;
                        acc.edge_art_max = acc.edge_art_max.max(artifact.reduce_max());
                        acc.edge_det_max = acc.edge_det_max.max(detail_lost.reduce_max());
                    }
                }

                // === HF energy (L2): (pixel - mu)² ===
                let vs = s - mu1;
                let vd = d - mu2;
                if !free.local_only {
                    acc.hf_sq_src += (vs * vs).reduce_add() as f64;
                    acc.hf_sq_dst += (vd * vd).reduce_add() as f64;
                }

                // === HF magnitude (L1): |pixel - mu| ===
                // diff1/diff2 already computed above
                if !free.local_only {
                    acc.hf_abs_src += diff1.reduce_add() as f64;
                    acc.hf_abs_dst += diff2.reduce_add() as f64;
                }

                // === MSE: (src - dst)² ===
                let pd = s - d;
                acc.mse += (pd * pd).reduce_add() as f64;
                if ext.on {
                    let act = sum_act * inv_v;
                    ext_accumulate16(token, &mut acc, sd, ed, pd, act, ext);
                }

                // === Free raw moments (`raw_moments`) ===
                // Plain sums of the raw pixels already in registers — no
                // new plane, no new load, no new pass. They finalize the
                // append block's GLOBAL_DMEAN / GLOBAL_CGAIN / GLOBAL_CLOSS
                // and append2's LUMA_MEAN_REF, which are the only 944 slots
                // whose value is a function of the RAW planes alone (see
                // `benchmarks/free_features_2026-09-01.md`). UNLIKE every
                // f64-reduce accumulator above it, this vector-adds across
                // rows (`fm_*`) with no per-row reduce_add, then reduces
                // ONCE at the band's last inner row — reduce_add is a
                // horizontal SIMD op, so 4 of them every row (one per
                // accumulator) is the shape this avoids; the doc above
                // prices the result. The vector sum is bounded to
                // `V1_BAND_ROWS` (32) rows of f32 before the f64 upgrade,
                // so the reorder costs negligible precision: worst |Δ| vs
                // the 944 append block is 5.35e-6 (was 4.62e-6 pre-batch),
                // ~2 orders below the module's 5e-4 tolerance
                // (`free_extras_match_the_944_append_block`).
                if free.raw_moments {
                    raw_moments_accumulate16(
                        &mut fm_s, &mut fm_d, &mut fm_s2, &mut fm_d2, &mut fm_dd, &mut fm_ds, s, d,
                    );
                    if y + 1 == inner_end {
                        raw_moments_finish16(&mut acc, fm_s, fm_d, fm_s2, fm_d2, fm_dd, fm_ds);
                    }
                }

                // === Free BOUNDED ERROR (`free.bounded_err` / `lum_bins`) ===
                // `pd` is the same register the `acc.mse` line above just
                // used; `s` is this channel's source row, which for the Y
                // channel IS the reference-luma plane the bins weight by.
                // No new plane, no new load, no new pass (class C — see the
                // helper block at the top of this module).
                if free.bounded_err {
                    let be_i = bounded_err_accumulate16(token, &mut be_m, pd);
                    if free.lum_bins {
                        lum_bins_accumulate16(
                            token,
                            &mut be_wdn,
                            &mut be_wdd,
                            &mut be_wbn,
                            &mut be_wbd,
                            s,
                            be_i,
                        );
                    }
                    if y + 1 == inner_end {
                        bounded_err_finish16(&mut acc, be_m);
                        if free.lum_bins {
                            lum_bins_finish16(&mut acc, be_wdn, be_wdd, be_wbn, be_wbd);
                        }
                    }
                }
            }

            // Slide V-blur window
            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            let aadd = add_idx * width + col_base;
            let arem = rem_idx * width + col_base;

            sum_m1 = sum_m1
                + f32x16::from_array_t(token, h_mu1[add_base..][..16].try_into().unwrap())
                - f32x16::from_array_t(token, h_mu1[rem_base..][..16].try_into().unwrap());
            sum_m2 = sum_m2
                + f32x16::from_array_t(token, h_mu2[add_base..][..16].try_into().unwrap())
                - f32x16::from_array_t(token, h_mu2[rem_base..][..16].try_into().unwrap());
            sum_sq = sum_sq
                + f32x16::from_array_t(token, h_sigma_sq[add_base..][..16].try_into().unwrap())
                - f32x16::from_array_t(token, h_sigma_sq[rem_base..][..16].try_into().unwrap());
            sum_s12 = sum_s12
                + f32x16::from_array_t(token, h_sigma12[add_base..][..16].try_into().unwrap())
                - f32x16::from_array_t(token, h_sigma12[rem_base..][..16].try_into().unwrap());
            if ext.on {
                sum_act = sum_act
                    + f32x16::from_array_t(token, h_act[aadd..][..16].try_into().unwrap())
                    - f32x16::from_array_t(token, h_act[arem..][..16].try_into().unwrap());
            }
        }
    }

    // Remainder columns with f32x8
    let col_base_8 = col_groups * 16;
    let v3 = token.v3();
    let inv_v8 = f32x8::splat_t(v3, 1.0 / diam as f32);
    let remaining_8groups = (width - col_base_8) / 8;

    let one8 = f32x8::splat_t(v3, 1.0);
    let zero8 = f32x8::zero_t(v3);

    for cg in 0..remaining_8groups {
        let col_base = col_base_8 + cg * 8;
        let mut sum_m1 = f32x8::zero_t(v3);
        let mut sum_m2 = f32x8::zero_t(v3);
        let mut sum_sq = f32x8::zero_t(v3);
        let mut sum_s12 = f32x8::zero_t(v3);
        let mut sum_act = f32x8::zero_t(v3);
        // Free raw moments: one lane accumulator per column group, reduced
        // at the band's last inner row (see `raw_moments`). Dead code when
        // the caller did not ask.
        let (mut fm_s, mut fm_d, mut fm_s2, mut fm_d2) = (
            f32x8::zero_t(v3),
            f32x8::zero_t(v3),
            f32x8::zero_t(v3),
            f32x8::zero_t(v3),
        );
        // Revision 2\'s paired second moment (see `raw_moments_accumulate16`).
        let mut fm_dd = f32x8::zero_t(v3);
        let mut fm_ds = f32x8::zero_t(v3);
        // Free bounded-error lane accumulators (`free.bounded_err` /
        // `free.lum_bins`) — same band-batched shape and rationale as
        // `fm_*` above: vector-add every row, reduce ONCE at the band's
        // last inner row.
        let (mut be_m, mut be_wdn, mut be_wdd, mut be_wbn, mut be_wbd) = (
            f32x8::zero_t(v3),
            f32x8::zero_t(v3),
            f32x8::zero_t(v3),
            f32x8::zero_t(v3),
            f32x8::zero_t(v3),
        );

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            let base = idx * width + col_base;
            let abase = idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array_t(v3, h_mu1[base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array_t(v3, h_mu2[base..][..8].try_into().unwrap());
            sum_sq = sum_sq + f32x8::from_array_t(v3, h_sigma_sq[base..][..8].try_into().unwrap());
            sum_s12 = sum_s12 + f32x8::from_array_t(v3, h_sigma12[base..][..8].try_into().unwrap());
            if ext.on {
                sum_act =
                    sum_act + f32x8::from_array_t(v3, h_act[abase..][..8].try_into().unwrap());
            }
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;
                let mu1 = sum_m1 * inv_v8;
                let mu2 = sum_m2 * inv_v8;
                let ssq = sum_sq * inv_v8;
                let s12 = sum_s12 * inv_v8;
                let s = f32x8::from_array_t(v3, src[base..][..8].try_into().unwrap());
                let d = f32x8::from_array_t(v3, dst[base..][..8].try_into().unwrap());

                // SSIM
                let sd = if direct {
                    ssim_direct8(v3, form, mu1, mu2, ssq, s12).max(zero8)
                } else {
                    (ssim_dissim8(v3, form, mu1, mu2, ssq, s12)).max(zero8)
                };
                let sd2 = sd * sd;
                let sd4 = sd2 * sd2;
                acc.ssim_d += sd.reduce_add() as f64;
                acc.ssim_d4 += sd4.reduce_add() as f64;
                acc.ssim_d2 += sd2.reduce_add() as f64;
                if !free.local_only {
                    acc.ssim_d8 += (sd4 * sd4).reduce_add() as f64;
                    acc.ssim_max = acc.ssim_max.max(sd.reduce_max());
                }
                if store_sd {
                    sd_out[base..base + 8].copy_from_slice(&sd.to_array());
                }
                if store_mu {
                    mu1_out[base..base + 8].copy_from_slice(&mu1.to_array());
                    mu2_out[base..base + 8].copy_from_slice(&mu2.to_array());
                }
                if store_sigma {
                    ssq_out[base..base + 8].copy_from_slice(&ssq.to_array());
                    s12_out[base..base + 8].copy_from_slice(&s12.to_array());
                }

                // Edge
                let diff1 = (s - mu1).abs();
                let diff2 = (d - mu2).abs();
                let ed = (one8 + diff2) / (one8 + diff1) - one8;
                let artifact = ed.max(zero8);
                let detail_lost = (-ed).max(zero8);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                if !free.omit_edges {
                    acc.edge_art += artifact.reduce_add() as f64;
                    acc.edge_art4 += a4.reduce_add() as f64;
                    acc.edge_art2 += a2.reduce_add() as f64;
                    acc.edge_det += detail_lost.reduce_add() as f64;
                    acc.edge_det4 += dl4.reduce_add() as f64;
                    acc.edge_det2 += dl2.reduce_add() as f64;
                    if !free.omit_edges {
                        acc.edge_art8 += (a4 * a4).reduce_add() as f64;
                        acc.edge_det8 += (dl4 * dl4).reduce_add() as f64;
                        acc.edge_art_max = acc.edge_art_max.max(artifact.reduce_max());
                        acc.edge_det_max = acc.edge_det_max.max(detail_lost.reduce_max());
                    }
                }

                // Variance
                let vs = s - mu1;
                let vd = d - mu2;
                if !free.local_only {
                    acc.hf_sq_src += (vs * vs).reduce_add() as f64;
                    acc.hf_sq_dst += (vd * vd).reduce_add() as f64;
                }

                // Texture
                if !free.local_only {
                    acc.hf_abs_src += diff1.reduce_add() as f64;
                    acc.hf_abs_dst += diff2.reduce_add() as f64;
                }

                // MSE
                let pd = s - d;
                acc.mse += (pd * pd).reduce_add() as f64;
                if ext.on {
                    let act = sum_act * inv_v8;
                    ext_accumulate8(v3, &mut acc, sd, ed, pd, act, ext);
                }

                // === Free raw moments (`raw_moments`) ===
                // Plain sums of the raw pixels already in registers — no
                // new plane, no new load, no new pass. They finalize the
                // append block's GLOBAL_DMEAN / GLOBAL_CGAIN / GLOBAL_CLOSS
                // and append2's LUMA_MEAN_REF, which are the only 944 slots
                // whose value is a function of the RAW planes alone (see
                // `benchmarks/free_features_2026-09-01.md`). UNLIKE every
                // f64-reduce accumulator above it, this vector-adds across
                // rows (`fm_*`) with no per-row reduce_add, then reduces
                // ONCE at the band's last inner row — reduce_add is a
                // horizontal SIMD op, so 4 of them every row (one per
                // accumulator) is the shape this avoids; the doc above
                // prices the result. The vector sum is bounded to
                // `V1_BAND_ROWS` (32) rows of f32 before the f64 upgrade,
                // so the reorder costs negligible precision: worst |Δ| vs
                // the 944 append block is 5.35e-6 (was 4.62e-6 pre-batch),
                // ~2 orders below the module's 5e-4 tolerance
                // (`free_extras_match_the_944_append_block`).
                if free.raw_moments {
                    raw_moments_accumulate8(
                        &mut fm_s, &mut fm_d, &mut fm_s2, &mut fm_d2, &mut fm_dd, &mut fm_ds, s, d,
                    );
                    if y + 1 == inner_end {
                        raw_moments_finish8(&mut acc, fm_s, fm_d, fm_s2, fm_d2, fm_dd, fm_ds);
                    }
                }

                // === Free BOUNDED ERROR (`free.bounded_err` / `lum_bins`) ===
                // `pd` is the same register the `acc.mse` line above just
                // used; `s` is this channel's source row, which for the Y
                // channel IS the reference-luma plane the bins weight by.
                // No new plane, no new load, no new pass (class C — see the
                // helper block at the top of this module).
                if free.bounded_err {
                    let be_i = bounded_err_accumulate8(v3, &mut be_m, pd);
                    if free.lum_bins {
                        lum_bins_accumulate8(
                            v3,
                            &mut be_wdn,
                            &mut be_wdd,
                            &mut be_wbn,
                            &mut be_wbd,
                            s,
                            be_i,
                        );
                    }
                    if y + 1 == inner_end {
                        bounded_err_finish8(&mut acc, be_m);
                        if free.lum_bins {
                            lum_bins_finish8(&mut acc, be_wdn, be_wdd, be_wbn, be_wbd);
                        }
                    }
                }
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            let aadd = add_idx * width + col_base;
            let arem = rem_idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array_t(v3, h_mu1[add_base..][..8].try_into().unwrap())
                - f32x8::from_array_t(v3, h_mu1[rem_base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array_t(v3, h_mu2[add_base..][..8].try_into().unwrap())
                - f32x8::from_array_t(v3, h_mu2[rem_base..][..8].try_into().unwrap());
            sum_sq = sum_sq
                + f32x8::from_array_t(v3, h_sigma_sq[add_base..][..8].try_into().unwrap())
                - f32x8::from_array_t(v3, h_sigma_sq[rem_base..][..8].try_into().unwrap());
            sum_s12 = sum_s12
                + f32x8::from_array_t(v3, h_sigma12[add_base..][..8].try_into().unwrap())
                - f32x8::from_array_t(v3, h_sigma12[rem_base..][..8].try_into().unwrap());
            if ext.on {
                sum_act = sum_act + f32x8::from_array_t(v3, h_act[aadd..][..8].try_into().unwrap())
                    - f32x8::from_array_t(v3, h_act[arem..][..8].try_into().unwrap());
            }
        }
    }

    // Scalar remainder
    let inv = 1.0 / diam as f32;
    for x in (col_base_8 + remaining_8groups * 8)..width {
        let mut sum_m1 = 0.0f32;
        let mut sum_m2 = 0.0f32;
        let mut sum_sq = 0.0f32;
        let mut sum_s12 = 0.0f32;
        let mut sum_act = 0.0f32;
        // Free raw moments: one lane accumulator per column group, reduced
        // at the band's last inner row (see `raw_moments`). Dead code when
        // the caller did not ask.
        let (mut fm_s, mut fm_d, mut fm_s2, mut fm_d2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        // Revision 2's paired second moment and first difference (see
        // `raw_moments_accumulate16`).
        let mut fm_dd = 0.0f32;
        let mut fm_ds = 0.0f32;
        // Free bounded-error lane accumulators (`free.bounded_err` /
        // `free.lum_bins`) — same band-batched shape and rationale as
        // `fm_*` above: vector-add every row, reduce ONCE at the band's
        // last inner row.
        let (mut be_m, mut be_wdn, mut be_wdd, mut be_wbn, mut be_wbd) =
            (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            sum_m1 += h_mu1[idx * width + x];
            sum_m2 += h_mu2[idx * width + x];
            sum_sq += h_sigma_sq[idx * width + x];
            sum_s12 += h_sigma12[idx * width + x];
            if ext.on {
                sum_act += h_act[idx * width + x];
            }
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let mu1 = sum_m1 * inv;
                let mu2 = sum_m2 * inv;
                let ssq = sum_sq * inv;
                let s12 = sum_s12 * inv;
                let sv = src[y * width + x];
                let dv = dst[y * width + x];

                // SSIM (f32 to match SIMD paths)
                let sd = if direct {
                    ssim_direct_raw_scalar(form, mu1, mu2, ssq, s12).max(0.0f32)
                } else {
                    (ssim_dissim_raw_scalar(form, mu1, mu2, ssq, s12)).max(0.0f32)
                };
                let sd2 = sd * sd;
                let sd4 = sd2 * sd2;
                acc.ssim_d += sd as f64;
                acc.ssim_d4 += sd4 as f64;
                acc.ssim_d2 += sd2 as f64;
                if !free.local_only {
                    acc.ssim_d8 += (sd4 * sd4) as f64;
                    acc.ssim_max = acc.ssim_max.max(sd);
                }
                if store_sd {
                    sd_out[y * width + x] = sd;
                }
                if store_mu {
                    mu1_out[y * width + x] = mu1;
                    mu2_out[y * width + x] = mu2;
                }
                if store_sigma {
                    ssq_out[y * width + x] = ssq;
                    s12_out[y * width + x] = s12;
                }

                // Edge (f32 to match SIMD paths)
                let diff1 = (sv - mu1).abs();
                let diff2 = (dv - mu2).abs();
                let ed = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;
                let artifact = ed.max(0.0f32);
                let detail_lost = (-ed).max(0.0f32);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                if !free.omit_edges {
                    acc.edge_art += artifact as f64;
                    acc.edge_art4 += a4 as f64;
                    acc.edge_art2 += a2 as f64;
                    acc.edge_det += detail_lost as f64;
                    acc.edge_det4 += dl4 as f64;
                    acc.edge_det2 += dl2 as f64;
                    if !free.omit_edges {
                        acc.edge_art8 += (a4 * a4) as f64;
                        acc.edge_det8 += (dl4 * dl4) as f64;
                        acc.edge_art_max = acc.edge_art_max.max(artifact);
                        acc.edge_det_max = acc.edge_det_max.max(detail_lost);
                    }
                }

                // Variance
                let vs = sv - mu1;
                let vd = dv - mu2;
                if !free.local_only {
                    acc.hf_sq_src += (vs * vs) as f64;
                    acc.hf_sq_dst += (vd * vd) as f64;
                }

                // Texture
                if !free.local_only {
                    acc.hf_abs_src += diff1 as f64;
                    acc.hf_abs_dst += diff2 as f64;
                }

                // MSE
                let pd = sv - dv;
                acc.mse += (pd * pd) as f64;
                if ext.on {
                    let act = sum_act * inv;
                    ext_accumulate_scalar(&mut acc, sd, ed, pd, act, ext);
                }

                // === Free raw moments (`raw_moments`) — scalar tail ===
                if free.raw_moments {
                    raw_moments_accumulate_scalar(
                        &mut fm_s, &mut fm_d, &mut fm_s2, &mut fm_d2, &mut fm_dd, &mut fm_ds, sv,
                        dv,
                    );
                    if y + 1 == inner_end {
                        raw_moments_finish_scalar(&mut acc, fm_s, fm_d, fm_s2, fm_d2, fm_dd, fm_ds);
                    }
                }

                // === Free BOUNDED ERROR (`free.bounded_err` / `lum_bins`) ===
                // `pd` is the same register the `acc.mse` line above just
                // used; `s` is this channel's source row, which for the Y
                // channel IS the reference-luma plane the bins weight by.
                // No new plane, no new load, no new pass (class C — see the
                // helper block at the top of this module).
                if free.bounded_err {
                    let be_i = bounded_err_accumulate_scalar(&mut be_m, pd);
                    if free.lum_bins {
                        lum_bins_accumulate_scalar(
                            &mut be_wdn,
                            &mut be_wdd,
                            &mut be_wbn,
                            &mut be_wbd,
                            sv,
                            be_i,
                        );
                    }
                    if y + 1 == inner_end {
                        bounded_err_finish_scalar(&mut acc, be_m);
                        if free.lum_bins {
                            lum_bins_finish_scalar(&mut acc, be_wdn, be_wdd, be_wbn, be_wbd);
                        }
                    }
                }
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            sum_m1 = sum_m1 + h_mu1[add_idx * width + x] - h_mu1[rem_idx * width + x];
            sum_m2 = sum_m2 + h_mu2[add_idx * width + x] - h_mu2[rem_idx * width + x];
            sum_sq = sum_sq + h_sigma_sq[add_idx * width + x] - h_sigma_sq[rem_idx * width + x];
            sum_s12 = sum_s12 + h_sigma12[add_idx * width + x] - h_sigma12[rem_idx * width + x];
            if ext.on {
                sum_act = sum_act + h_act[add_idx * width + x] - h_act[rem_idx * width + x];
            }
        }
    }

    acc
}
#[cfg(target_arch = "x86_64")]
#[arcane]
fn fused_vblur_ssim_inner_v4x(
    token: archmage::X64V4xToken,
    h_mu1: &[f32],
    h_mu2: &[f32],
    h_sigma_sq: &[f32],
    h_sigma12: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    sd_out: &mut [f32],
    store_sd: bool,
    ssq_out: &mut [f32],
    s12_out: &mut [f32],
    store_sigma: bool,
    // `free`: which FREE extra accumulators to carry alongside the existing
    // sums — the four raw moments (Σs, Σd, Σs², Σd²) and/or the class-C
    // bounded-error family. See [`FreeExtrasWork`] and
    // `StripChannelAccum::sum_s` / `sum_msat`.
    free: FreeExtrasWork,
    // Revision 3: form the dissimilarity with the direct-error moment that the
    // H pass put in `h_sigma12` (see `blur::fused_blur_h_ssim`).
    direct: bool,
    ext: ExtPoolsWork,
    h_act: &[f32],
) -> StripChannelAccum {
    let form = free.luma_form();
    let diam = 2 * radius + 1;
    let inv_v = f32x16::splat_t(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 16;

    // One slice construction per plane: every row access below is
    // `row_base + col_base` with `row_base = idx * width`, `idx <= height - 1`
    // (all index helpers clamp with `.min(height - 1)`), `col_base <= width - 16`
    // — so `base + 16 <= height * width` is statically provable and the
    // `[base..][..16]` / `[base..base + 16]` accesses compile without the
    // panic-edge recheck per load/store. Same trick as the canon64 bodies.
    let n_px = height * width;
    let h_mu1 = if h_mu1.len() >= n_px {
        &h_mu1[..n_px]
    } else {
        h_mu1
    };
    let h_mu2 = if h_mu2.len() >= n_px {
        &h_mu2[..n_px]
    } else {
        h_mu2
    };
    let h_sigma_sq = if h_sigma_sq.len() >= n_px {
        &h_sigma_sq[..n_px]
    } else {
        h_sigma_sq
    };
    let h_sigma12 = if h_sigma12.len() >= n_px {
        &h_sigma12[..n_px]
    } else {
        h_sigma12
    };
    let h_act = if h_act.len() >= n_px {
        &h_act[..n_px]
    } else {
        h_act
    };
    let src = if src.len() >= n_px { &src[..n_px] } else { src };
    let dst = if dst.len() >= n_px { &dst[..n_px] } else { dst };
    let mu1_out = if mu1_out.len() >= n_px {
        &mut mu1_out[..n_px]
    } else {
        mu1_out
    };
    let mu2_out = if mu2_out.len() >= n_px {
        &mut mu2_out[..n_px]
    } else {
        mu2_out
    };
    let sd_out = if sd_out.len() >= n_px {
        &mut sd_out[..n_px]
    } else {
        sd_out
    };
    let ssq_out = if ssq_out.len() >= n_px {
        &mut ssq_out[..n_px]
    } else {
        ssq_out
    };
    let s12_out = if s12_out.len() >= n_px {
        &mut s12_out[..n_px]
    } else {
        s12_out
    };

    // SSIM constants
    let one = f32x16::splat_t(token, 1.0);
    let zero = f32x16::zero_t(token);

    let mut acc = StripChannelAccum::zero();
    let inner_end = inner_start + inner_h;

    for cg in 0..col_groups {
        let col_base = cg * 16;

        // Initialize 4 running sums for this column group
        let mut sum_m1 = f32x16::zero_t(token);
        let mut sum_m2 = f32x16::zero_t(token);
        let mut sum_sq = f32x16::zero_t(token);
        let mut sum_s12 = f32x16::zero_t(token);
        let mut sum_act = f32x16::zero_t(token);
        // Free raw moments: one lane accumulator per column group, reduced
        // at the band's last inner row (see `raw_moments`). Dead code when
        // the caller did not ask.
        let (mut fm_s, mut fm_d, mut fm_s2, mut fm_d2) = (
            f32x16::zero_t(token),
            f32x16::zero_t(token),
            f32x16::zero_t(token),
            f32x16::zero_t(token),
        );
        // Revision 2\'s paired second moment (see `raw_moments_accumulate16`).
        let mut fm_dd = f32x16::zero_t(token);
        let mut fm_ds = f32x16::zero_t(token);
        // Free bounded-error lane accumulators (`free.bounded_err` /
        // `free.lum_bins`) — same band-batched shape and rationale as
        // `fm_*` above: vector-add every row, reduce ONCE at the band's
        // last inner row.
        let (mut be_m, mut be_wdn, mut be_wdd, mut be_wbn, mut be_wbd) = (
            f32x16::zero_t(token),
            f32x16::zero_t(token),
            f32x16::zero_t(token),
            f32x16::zero_t(token),
            f32x16::zero_t(token),
        );

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            let base = idx * width + col_base;
            let abase = idx * width + col_base;
            sum_m1 = sum_m1 + f32x16::from_array_t(token, h_mu1[base..][..16].try_into().unwrap());
            sum_m2 = sum_m2 + f32x16::from_array_t(token, h_mu2[base..][..16].try_into().unwrap());
            sum_sq =
                sum_sq + f32x16::from_array_t(token, h_sigma_sq[base..][..16].try_into().unwrap());
            sum_s12 =
                sum_s12 + f32x16::from_array_t(token, h_sigma12[base..][..16].try_into().unwrap());
            if ext.on {
                sum_act =
                    sum_act + f32x16::from_array_t(token, h_act[abase..][..16].try_into().unwrap());
            }
        }

        for y in 0..height {
            // Only accumulate features for inner rows
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;

                // V-blurred values (still in registers)
                let mu1 = sum_m1 * inv_v;
                let mu2 = sum_m2 * inv_v;
                let ssq = sum_sq * inv_v;
                let s12 = sum_s12 * inv_v;

                // Load raw pixel values
                let s = f32x16::from_array_t(token, src[base..][..16].try_into().unwrap());
                let d = f32x16::from_array_t(token, dst[base..][..16].try_into().unwrap());

                // === SSIM ===
                let sd = if direct {
                    ssim_direct16(token, form, mu1, mu2, ssq, s12).max(zero)
                } else {
                    (ssim_dissim16(token, form, mu1, mu2, ssq, s12)).max(zero)
                };
                let sd2 = sd * sd;
                let sd4 = sd2 * sd2;
                acc.ssim_d += sd.reduce_add() as f64;
                acc.ssim_d4 += sd4.reduce_add() as f64;
                acc.ssim_d2 += sd2.reduce_add() as f64;
                if !free.local_only {
                    acc.ssim_d8 += (sd4 * sd4).reduce_add() as f64;
                    acc.ssim_max = acc.ssim_max.max(sd.reduce_max());
                }
                if store_sd {
                    sd_out[base..base + 16].copy_from_slice(&sd.to_array());
                }
                if store_mu {
                    mu1_out[base..base + 16].copy_from_slice(&mu1.to_array());
                    mu2_out[base..base + 16].copy_from_slice(&mu2.to_array());
                }
                if store_sigma {
                    ssq_out[base..base + 16].copy_from_slice(&ssq.to_array());
                    s12_out[base..base + 16].copy_from_slice(&s12.to_array());
                }

                // === Edge ===
                let diff1 = (s - mu1).abs();
                let diff2 = (d - mu2).abs();
                let ed = (one + diff2) / (one + diff1) - one;
                let artifact = ed.max(zero);
                let detail_lost = (-ed).max(zero);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                if !free.omit_edges {
                    acc.edge_art += artifact.reduce_add() as f64;
                    acc.edge_art4 += a4.reduce_add() as f64;
                    acc.edge_art2 += a2.reduce_add() as f64;
                    acc.edge_det += detail_lost.reduce_add() as f64;
                    acc.edge_det4 += dl4.reduce_add() as f64;
                    acc.edge_det2 += dl2.reduce_add() as f64;
                    if !free.omit_edges {
                        acc.edge_art8 += (a4 * a4).reduce_add() as f64;
                        acc.edge_det8 += (dl4 * dl4).reduce_add() as f64;
                        acc.edge_art_max = acc.edge_art_max.max(artifact.reduce_max());
                        acc.edge_det_max = acc.edge_det_max.max(detail_lost.reduce_max());
                    }
                }

                // === HF energy (L2): (pixel - mu)² ===
                let vs = s - mu1;
                let vd = d - mu2;
                if !free.local_only {
                    acc.hf_sq_src += (vs * vs).reduce_add() as f64;
                    acc.hf_sq_dst += (vd * vd).reduce_add() as f64;
                }

                // === HF magnitude (L1): |pixel - mu| ===
                // diff1/diff2 already computed above
                if !free.local_only {
                    acc.hf_abs_src += diff1.reduce_add() as f64;
                    acc.hf_abs_dst += diff2.reduce_add() as f64;
                }

                // === MSE: (src - dst)² ===
                let pd = s - d;
                acc.mse += (pd * pd).reduce_add() as f64;
                if ext.on {
                    let act = sum_act * inv_v;
                    ext_accumulate16(token, &mut acc, sd, ed, pd, act, ext);
                }

                // === Free raw moments (`raw_moments`) ===
                // Plain sums of the raw pixels already in registers — no
                // new plane, no new load, no new pass. They finalize the
                // append block's GLOBAL_DMEAN / GLOBAL_CGAIN / GLOBAL_CLOSS
                // and append2's LUMA_MEAN_REF, which are the only 944 slots
                // whose value is a function of the RAW planes alone (see
                // `benchmarks/free_features_2026-09-01.md`). UNLIKE every
                // f64-reduce accumulator above it, this vector-adds across
                // rows (`fm_*`) with no per-row reduce_add, then reduces
                // ONCE at the band's last inner row — reduce_add is a
                // horizontal SIMD op, so 4 of them every row (one per
                // accumulator) is the shape this avoids; the doc above
                // prices the result. The vector sum is bounded to
                // `V1_BAND_ROWS` (32) rows of f32 before the f64 upgrade,
                // so the reorder costs negligible precision: worst |Δ| vs
                // the 944 append block is 5.35e-6 (was 4.62e-6 pre-batch),
                // ~2 orders below the module's 5e-4 tolerance
                // (`free_extras_match_the_944_append_block`).
                if free.raw_moments {
                    raw_moments_accumulate16(
                        &mut fm_s, &mut fm_d, &mut fm_s2, &mut fm_d2, &mut fm_dd, &mut fm_ds, s, d,
                    );
                    if y + 1 == inner_end {
                        raw_moments_finish16(&mut acc, fm_s, fm_d, fm_s2, fm_d2, fm_dd, fm_ds);
                    }
                }

                // === Free BOUNDED ERROR (`free.bounded_err` / `lum_bins`) ===
                // `pd` is the same register the `acc.mse` line above just
                // used; `s` is this channel's source row, which for the Y
                // channel IS the reference-luma plane the bins weight by.
                // No new plane, no new load, no new pass (class C — see the
                // helper block at the top of this module).
                if free.bounded_err {
                    let be_i = bounded_err_accumulate16(token, &mut be_m, pd);
                    if free.lum_bins {
                        lum_bins_accumulate16(
                            token,
                            &mut be_wdn,
                            &mut be_wdd,
                            &mut be_wbn,
                            &mut be_wbd,
                            s,
                            be_i,
                        );
                    }
                    if y + 1 == inner_end {
                        bounded_err_finish16(&mut acc, be_m);
                        if free.lum_bins {
                            lum_bins_finish16(&mut acc, be_wdn, be_wdd, be_wbn, be_wbd);
                        }
                    }
                }
            }

            // Slide V-blur window
            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            let aadd = add_idx * width + col_base;
            let arem = rem_idx * width + col_base;

            sum_m1 = sum_m1
                + f32x16::from_array_t(token, h_mu1[add_base..][..16].try_into().unwrap())
                - f32x16::from_array_t(token, h_mu1[rem_base..][..16].try_into().unwrap());
            sum_m2 = sum_m2
                + f32x16::from_array_t(token, h_mu2[add_base..][..16].try_into().unwrap())
                - f32x16::from_array_t(token, h_mu2[rem_base..][..16].try_into().unwrap());
            sum_sq = sum_sq
                + f32x16::from_array_t(token, h_sigma_sq[add_base..][..16].try_into().unwrap())
                - f32x16::from_array_t(token, h_sigma_sq[rem_base..][..16].try_into().unwrap());
            sum_s12 = sum_s12
                + f32x16::from_array_t(token, h_sigma12[add_base..][..16].try_into().unwrap())
                - f32x16::from_array_t(token, h_sigma12[rem_base..][..16].try_into().unwrap());
            if ext.on {
                sum_act = sum_act
                    + f32x16::from_array_t(token, h_act[aadd..][..16].try_into().unwrap())
                    - f32x16::from_array_t(token, h_act[arem..][..16].try_into().unwrap());
            }
        }
    }

    // Remainder columns with f32x8
    let col_base_8 = col_groups * 16;
    let v3 = token.v3();
    let inv_v8 = f32x8::splat_t(v3, 1.0 / diam as f32);
    let remaining_8groups = (width - col_base_8) / 8;

    let one8 = f32x8::splat_t(v3, 1.0);
    let zero8 = f32x8::zero_t(v3);

    for cg in 0..remaining_8groups {
        let col_base = col_base_8 + cg * 8;
        let mut sum_m1 = f32x8::zero_t(v3);
        let mut sum_m2 = f32x8::zero_t(v3);
        let mut sum_sq = f32x8::zero_t(v3);
        let mut sum_s12 = f32x8::zero_t(v3);
        let mut sum_act = f32x8::zero_t(v3);
        // Free raw moments: one lane accumulator per column group, reduced
        // at the band's last inner row (see `raw_moments`). Dead code when
        // the caller did not ask.
        let (mut fm_s, mut fm_d, mut fm_s2, mut fm_d2) = (
            f32x8::zero_t(v3),
            f32x8::zero_t(v3),
            f32x8::zero_t(v3),
            f32x8::zero_t(v3),
        );
        // Revision 2\'s paired second moment (see `raw_moments_accumulate16`).
        let mut fm_dd = f32x8::zero_t(v3);
        let mut fm_ds = f32x8::zero_t(v3);
        // Free bounded-error lane accumulators (`free.bounded_err` /
        // `free.lum_bins`) — same band-batched shape and rationale as
        // `fm_*` above: vector-add every row, reduce ONCE at the band's
        // last inner row.
        let (mut be_m, mut be_wdn, mut be_wdd, mut be_wbn, mut be_wbd) = (
            f32x8::zero_t(v3),
            f32x8::zero_t(v3),
            f32x8::zero_t(v3),
            f32x8::zero_t(v3),
            f32x8::zero_t(v3),
        );

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            let base = idx * width + col_base;
            let abase = idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array_t(v3, h_mu1[base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array_t(v3, h_mu2[base..][..8].try_into().unwrap());
            sum_sq = sum_sq + f32x8::from_array_t(v3, h_sigma_sq[base..][..8].try_into().unwrap());
            sum_s12 = sum_s12 + f32x8::from_array_t(v3, h_sigma12[base..][..8].try_into().unwrap());
            if ext.on {
                sum_act =
                    sum_act + f32x8::from_array_t(v3, h_act[abase..][..8].try_into().unwrap());
            }
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;
                let mu1 = sum_m1 * inv_v8;
                let mu2 = sum_m2 * inv_v8;
                let ssq = sum_sq * inv_v8;
                let s12 = sum_s12 * inv_v8;
                let s = f32x8::from_array_t(v3, src[base..][..8].try_into().unwrap());
                let d = f32x8::from_array_t(v3, dst[base..][..8].try_into().unwrap());

                // SSIM
                let sd = if direct {
                    ssim_direct8(v3, form, mu1, mu2, ssq, s12).max(zero8)
                } else {
                    (ssim_dissim8(v3, form, mu1, mu2, ssq, s12)).max(zero8)
                };
                let sd2 = sd * sd;
                let sd4 = sd2 * sd2;
                acc.ssim_d += sd.reduce_add() as f64;
                acc.ssim_d4 += sd4.reduce_add() as f64;
                acc.ssim_d2 += sd2.reduce_add() as f64;
                if !free.local_only {
                    acc.ssim_d8 += (sd4 * sd4).reduce_add() as f64;
                    acc.ssim_max = acc.ssim_max.max(sd.reduce_max());
                }
                if store_sd {
                    sd_out[base..base + 8].copy_from_slice(&sd.to_array());
                }
                if store_mu {
                    mu1_out[base..base + 8].copy_from_slice(&mu1.to_array());
                    mu2_out[base..base + 8].copy_from_slice(&mu2.to_array());
                }
                if store_sigma {
                    ssq_out[base..base + 8].copy_from_slice(&ssq.to_array());
                    s12_out[base..base + 8].copy_from_slice(&s12.to_array());
                }

                // Edge
                let diff1 = (s - mu1).abs();
                let diff2 = (d - mu2).abs();
                let ed = (one8 + diff2) / (one8 + diff1) - one8;
                let artifact = ed.max(zero8);
                let detail_lost = (-ed).max(zero8);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                if !free.omit_edges {
                    acc.edge_art += artifact.reduce_add() as f64;
                    acc.edge_art4 += a4.reduce_add() as f64;
                    acc.edge_art2 += a2.reduce_add() as f64;
                    acc.edge_det += detail_lost.reduce_add() as f64;
                    acc.edge_det4 += dl4.reduce_add() as f64;
                    acc.edge_det2 += dl2.reduce_add() as f64;
                    if !free.omit_edges {
                        acc.edge_art8 += (a4 * a4).reduce_add() as f64;
                        acc.edge_det8 += (dl4 * dl4).reduce_add() as f64;
                        acc.edge_art_max = acc.edge_art_max.max(artifact.reduce_max());
                        acc.edge_det_max = acc.edge_det_max.max(detail_lost.reduce_max());
                    }
                }

                // Variance
                let vs = s - mu1;
                let vd = d - mu2;
                if !free.local_only {
                    acc.hf_sq_src += (vs * vs).reduce_add() as f64;
                    acc.hf_sq_dst += (vd * vd).reduce_add() as f64;
                }

                // Texture
                if !free.local_only {
                    acc.hf_abs_src += diff1.reduce_add() as f64;
                    acc.hf_abs_dst += diff2.reduce_add() as f64;
                }

                // MSE
                let pd = s - d;
                acc.mse += (pd * pd).reduce_add() as f64;
                if ext.on {
                    let act = sum_act * inv_v8;
                    ext_accumulate8(v3, &mut acc, sd, ed, pd, act, ext);
                }

                // === Free raw moments (`raw_moments`) ===
                // Plain sums of the raw pixels already in registers — no
                // new plane, no new load, no new pass. They finalize the
                // append block's GLOBAL_DMEAN / GLOBAL_CGAIN / GLOBAL_CLOSS
                // and append2's LUMA_MEAN_REF, which are the only 944 slots
                // whose value is a function of the RAW planes alone (see
                // `benchmarks/free_features_2026-09-01.md`). UNLIKE every
                // f64-reduce accumulator above it, this vector-adds across
                // rows (`fm_*`) with no per-row reduce_add, then reduces
                // ONCE at the band's last inner row — reduce_add is a
                // horizontal SIMD op, so 4 of them every row (one per
                // accumulator) is the shape this avoids; the doc above
                // prices the result. The vector sum is bounded to
                // `V1_BAND_ROWS` (32) rows of f32 before the f64 upgrade,
                // so the reorder costs negligible precision: worst |Δ| vs
                // the 944 append block is 5.35e-6 (was 4.62e-6 pre-batch),
                // ~2 orders below the module's 5e-4 tolerance
                // (`free_extras_match_the_944_append_block`).
                if free.raw_moments {
                    raw_moments_accumulate8(
                        &mut fm_s, &mut fm_d, &mut fm_s2, &mut fm_d2, &mut fm_dd, &mut fm_ds, s, d,
                    );
                    if y + 1 == inner_end {
                        raw_moments_finish8(&mut acc, fm_s, fm_d, fm_s2, fm_d2, fm_dd, fm_ds);
                    }
                }

                // === Free BOUNDED ERROR (`free.bounded_err` / `lum_bins`) ===
                // `pd` is the same register the `acc.mse` line above just
                // used; `s` is this channel's source row, which for the Y
                // channel IS the reference-luma plane the bins weight by.
                // No new plane, no new load, no new pass (class C — see the
                // helper block at the top of this module).
                if free.bounded_err {
                    let be_i = bounded_err_accumulate8(v3, &mut be_m, pd);
                    if free.lum_bins {
                        lum_bins_accumulate8(
                            v3,
                            &mut be_wdn,
                            &mut be_wdd,
                            &mut be_wbn,
                            &mut be_wbd,
                            s,
                            be_i,
                        );
                    }
                    if y + 1 == inner_end {
                        bounded_err_finish8(&mut acc, be_m);
                        if free.lum_bins {
                            lum_bins_finish8(&mut acc, be_wdn, be_wdd, be_wbn, be_wbd);
                        }
                    }
                }
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            let aadd = add_idx * width + col_base;
            let arem = rem_idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array_t(v3, h_mu1[add_base..][..8].try_into().unwrap())
                - f32x8::from_array_t(v3, h_mu1[rem_base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array_t(v3, h_mu2[add_base..][..8].try_into().unwrap())
                - f32x8::from_array_t(v3, h_mu2[rem_base..][..8].try_into().unwrap());
            sum_sq = sum_sq
                + f32x8::from_array_t(v3, h_sigma_sq[add_base..][..8].try_into().unwrap())
                - f32x8::from_array_t(v3, h_sigma_sq[rem_base..][..8].try_into().unwrap());
            sum_s12 = sum_s12
                + f32x8::from_array_t(v3, h_sigma12[add_base..][..8].try_into().unwrap())
                - f32x8::from_array_t(v3, h_sigma12[rem_base..][..8].try_into().unwrap());
            if ext.on {
                sum_act = sum_act + f32x8::from_array_t(v3, h_act[aadd..][..8].try_into().unwrap())
                    - f32x8::from_array_t(v3, h_act[arem..][..8].try_into().unwrap());
            }
        }
    }

    // Scalar remainder
    let inv = 1.0 / diam as f32;
    for x in (col_base_8 + remaining_8groups * 8)..width {
        let mut sum_m1 = 0.0f32;
        let mut sum_m2 = 0.0f32;
        let mut sum_sq = 0.0f32;
        let mut sum_s12 = 0.0f32;
        let mut sum_act = 0.0f32;
        // Free raw moments: one lane accumulator per column group, reduced
        // at the band's last inner row (see `raw_moments`). Dead code when
        // the caller did not ask.
        let (mut fm_s, mut fm_d, mut fm_s2, mut fm_d2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        // Revision 2's paired second moment and first difference (see
        // `raw_moments_accumulate16`).
        let mut fm_dd = 0.0f32;
        let mut fm_ds = 0.0f32;
        // Free bounded-error lane accumulators (`free.bounded_err` /
        // `free.lum_bins`) — same band-batched shape and rationale as
        // `fm_*` above: vector-add every row, reduce ONCE at the band's
        // last inner row.
        let (mut be_m, mut be_wdn, mut be_wdd, mut be_wbn, mut be_wbd) =
            (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            sum_m1 += h_mu1[idx * width + x];
            sum_m2 += h_mu2[idx * width + x];
            sum_sq += h_sigma_sq[idx * width + x];
            sum_s12 += h_sigma12[idx * width + x];
            if ext.on {
                sum_act += h_act[idx * width + x];
            }
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let mu1 = sum_m1 * inv;
                let mu2 = sum_m2 * inv;
                let ssq = sum_sq * inv;
                let s12 = sum_s12 * inv;
                let sv = src[y * width + x];
                let dv = dst[y * width + x];

                // SSIM (f32 to match SIMD paths)
                let sd = if direct {
                    ssim_direct_raw_scalar(form, mu1, mu2, ssq, s12).max(0.0f32)
                } else {
                    (ssim_dissim_raw_scalar(form, mu1, mu2, ssq, s12)).max(0.0f32)
                };
                let sd2 = sd * sd;
                let sd4 = sd2 * sd2;
                acc.ssim_d += sd as f64;
                acc.ssim_d4 += sd4 as f64;
                acc.ssim_d2 += sd2 as f64;
                if !free.local_only {
                    acc.ssim_d8 += (sd4 * sd4) as f64;
                    acc.ssim_max = acc.ssim_max.max(sd);
                }
                if store_sd {
                    sd_out[y * width + x] = sd;
                }
                if store_mu {
                    mu1_out[y * width + x] = mu1;
                    mu2_out[y * width + x] = mu2;
                }
                if store_sigma {
                    ssq_out[y * width + x] = ssq;
                    s12_out[y * width + x] = s12;
                }

                // Edge (f32 to match SIMD paths)
                let diff1 = (sv - mu1).abs();
                let diff2 = (dv - mu2).abs();
                let ed = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;
                let artifact = ed.max(0.0f32);
                let detail_lost = (-ed).max(0.0f32);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                if !free.omit_edges {
                    acc.edge_art += artifact as f64;
                    acc.edge_art4 += a4 as f64;
                    acc.edge_art2 += a2 as f64;
                    acc.edge_det += detail_lost as f64;
                    acc.edge_det4 += dl4 as f64;
                    acc.edge_det2 += dl2 as f64;
                    if !free.omit_edges {
                        acc.edge_art8 += (a4 * a4) as f64;
                        acc.edge_det8 += (dl4 * dl4) as f64;
                        acc.edge_art_max = acc.edge_art_max.max(artifact);
                        acc.edge_det_max = acc.edge_det_max.max(detail_lost);
                    }
                }

                // Variance
                let vs = sv - mu1;
                let vd = dv - mu2;
                if !free.local_only {
                    acc.hf_sq_src += (vs * vs) as f64;
                    acc.hf_sq_dst += (vd * vd) as f64;
                }

                // Texture
                if !free.local_only {
                    acc.hf_abs_src += diff1 as f64;
                    acc.hf_abs_dst += diff2 as f64;
                }

                // MSE
                let pd = sv - dv;
                acc.mse += (pd * pd) as f64;
                if ext.on {
                    let act = sum_act * inv;
                    ext_accumulate_scalar(&mut acc, sd, ed, pd, act, ext);
                }

                // === Free raw moments (`raw_moments`) — scalar tail ===
                if free.raw_moments {
                    raw_moments_accumulate_scalar(
                        &mut fm_s, &mut fm_d, &mut fm_s2, &mut fm_d2, &mut fm_dd, &mut fm_ds, sv,
                        dv,
                    );
                    if y + 1 == inner_end {
                        raw_moments_finish_scalar(&mut acc, fm_s, fm_d, fm_s2, fm_d2, fm_dd, fm_ds);
                    }
                }

                // === Free BOUNDED ERROR (`free.bounded_err` / `lum_bins`) ===
                // `pd` is the same register the `acc.mse` line above just
                // used; `s` is this channel's source row, which for the Y
                // channel IS the reference-luma plane the bins weight by.
                // No new plane, no new load, no new pass (class C — see the
                // helper block at the top of this module).
                if free.bounded_err {
                    let be_i = bounded_err_accumulate_scalar(&mut be_m, pd);
                    if free.lum_bins {
                        lum_bins_accumulate_scalar(
                            &mut be_wdn,
                            &mut be_wdd,
                            &mut be_wbn,
                            &mut be_wbd,
                            sv,
                            be_i,
                        );
                    }
                    if y + 1 == inner_end {
                        bounded_err_finish_scalar(&mut acc, be_m);
                        if free.lum_bins {
                            lum_bins_finish_scalar(&mut acc, be_wdn, be_wdd, be_wbn, be_wbd);
                        }
                    }
                }
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            sum_m1 = sum_m1 + h_mu1[add_idx * width + x] - h_mu1[rem_idx * width + x];
            sum_m2 = sum_m2 + h_mu2[add_idx * width + x] - h_mu2[rem_idx * width + x];
            sum_sq = sum_sq + h_sigma_sq[add_idx * width + x] - h_sigma_sq[rem_idx * width + x];
            sum_s12 = sum_s12 + h_sigma12[add_idx * width + x] - h_sigma12[rem_idx * width + x];
            if ext.on {
                sum_act = sum_act + h_act[add_idx * width + x] - h_act[rem_idx * width + x];
            }
        }
    }

    acc
}

// ============================================================
// AVX2 implementations
// ============================================================

#[cfg(target_arch = "x86_64")]
#[arcane]
fn fused_vblur_ssim_inner_v3(
    token: archmage::X64V3Token,
    h_mu1: &[f32],
    h_mu2: &[f32],
    h_sigma_sq: &[f32],
    h_sigma12: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    sd_out: &mut [f32],
    store_sd: bool,
    ssq_out: &mut [f32],
    s12_out: &mut [f32],
    store_sigma: bool,
    // `free`: which FREE extra accumulators to carry alongside the existing
    // sums — the four raw moments (Σs, Σd, Σs², Σd²) and/or the class-C
    // bounded-error family. See [`FreeExtrasWork`] and
    // `StripChannelAccum::sum_s` / `sum_msat`.
    free: FreeExtrasWork,
    // Revision 3: form the dissimilarity with the direct-error moment that the
    // H pass put in `h_sigma12` (see `blur::fused_blur_h_ssim`).
    direct: bool,
    ext: ExtPoolsWork,
    h_act: &[f32],
) -> StripChannelAccum {
    let form = free.luma_form();
    let diam = 2 * radius + 1;
    let inv_v = f32x8::splat_t(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 8;
    // One slice construction per plane: every row access below is
    // `row_base + col_base` with `row_base = idx * width`, `idx <= height - 1`
    // (all index helpers clamp with `.min(height - 1)`), `col_base <= width - 8`
    // — so `base + 8 <= height * width` is statically provable and the
    // `[base..][..8]` / `[base..base + 8]` accesses compile without the
    // panic-edge recheck per load/store. Same trick as the canon64 bodies.
    let n_px = height * width;
    let h_mu1 = if h_mu1.len() >= n_px {
        &h_mu1[..n_px]
    } else {
        h_mu1
    };
    let h_mu2 = if h_mu2.len() >= n_px {
        &h_mu2[..n_px]
    } else {
        h_mu2
    };
    let h_sigma_sq = if h_sigma_sq.len() >= n_px {
        &h_sigma_sq[..n_px]
    } else {
        h_sigma_sq
    };
    let h_sigma12 = if h_sigma12.len() >= n_px {
        &h_sigma12[..n_px]
    } else {
        h_sigma12
    };
    let h_act = if h_act.len() >= n_px {
        &h_act[..n_px]
    } else {
        h_act
    };
    let src = if src.len() >= n_px { &src[..n_px] } else { src };
    let dst = if dst.len() >= n_px { &dst[..n_px] } else { dst };
    let mu1_out = if mu1_out.len() >= n_px {
        &mut mu1_out[..n_px]
    } else {
        mu1_out
    };
    let mu2_out = if mu2_out.len() >= n_px {
        &mut mu2_out[..n_px]
    } else {
        mu2_out
    };
    let sd_out = if sd_out.len() >= n_px {
        &mut sd_out[..n_px]
    } else {
        sd_out
    };
    let ssq_out = if ssq_out.len() >= n_px {
        &mut ssq_out[..n_px]
    } else {
        ssq_out
    };
    let s12_out = if s12_out.len() >= n_px {
        &mut s12_out[..n_px]
    } else {
        s12_out
    };

    let one = f32x8::splat_t(token, 1.0);
    let zero = f32x8::zero_t(token);

    let mut acc = StripChannelAccum::zero();
    let inner_end = inner_start + inner_h;

    for cg in 0..col_groups {
        let col_base = cg * 8;
        let mut sum_m1 = f32x8::zero_t(token);
        let mut sum_m2 = f32x8::zero_t(token);
        let mut sum_sq = f32x8::zero_t(token);
        let mut sum_s12 = f32x8::zero_t(token);
        let mut sum_act = f32x8::zero_t(token);
        // Free raw moments: one lane accumulator per column group, reduced
        // at the band's last inner row (see `raw_moments`). Dead code when
        // the caller did not ask.
        let (mut fm_s, mut fm_d, mut fm_s2, mut fm_d2) = (
            f32x8::zero_t(token),
            f32x8::zero_t(token),
            f32x8::zero_t(token),
            f32x8::zero_t(token),
        );
        // Revision 2\'s paired second moment (see `raw_moments_accumulate16`).
        let mut fm_dd = f32x8::zero_t(token);
        let mut fm_ds = f32x8::zero_t(token);
        // Free bounded-error lane accumulators (`free.bounded_err` /
        // `free.lum_bins`) — same band-batched shape and rationale as
        // `fm_*` above: vector-add every row, reduce ONCE at the band's
        // last inner row.
        let (mut be_m, mut be_wdn, mut be_wdd, mut be_wbn, mut be_wbd) = (
            f32x8::zero_t(token),
            f32x8::zero_t(token),
            f32x8::zero_t(token),
            f32x8::zero_t(token),
            f32x8::zero_t(token),
        );

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            let base = idx * width + col_base;
            let abase = idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array_t(token, h_mu1[base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array_t(token, h_mu2[base..][..8].try_into().unwrap());
            sum_sq =
                sum_sq + f32x8::from_array_t(token, h_sigma_sq[base..][..8].try_into().unwrap());
            sum_s12 =
                sum_s12 + f32x8::from_array_t(token, h_sigma12[base..][..8].try_into().unwrap());
            if ext.on {
                sum_act =
                    sum_act + f32x8::from_array_t(token, h_act[abase..][..8].try_into().unwrap());
            }
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;
                let mu1 = sum_m1 * inv_v;
                let mu2 = sum_m2 * inv_v;
                let ssq = sum_sq * inv_v;
                let s12 = sum_s12 * inv_v;
                let s = f32x8::from_array_t(token, src[base..][..8].try_into().unwrap());
                let d = f32x8::from_array_t(token, dst[base..][..8].try_into().unwrap());

                // SSIM
                let sd = if direct {
                    ssim_direct8(token, form, mu1, mu2, ssq, s12).max(zero)
                } else {
                    (ssim_dissim8(token, form, mu1, mu2, ssq, s12)).max(zero)
                };
                let sd2 = sd * sd;
                let sd4 = sd2 * sd2;
                acc.ssim_d += sd.reduce_add() as f64;
                acc.ssim_d4 += sd4.reduce_add() as f64;
                acc.ssim_d2 += sd2.reduce_add() as f64;
                if !free.local_only {
                    acc.ssim_d8 += (sd4 * sd4).reduce_add() as f64;
                    acc.ssim_max = acc.ssim_max.max(sd.reduce_max());
                }
                if store_sd {
                    sd_out[base..base + 8].copy_from_slice(&sd.to_array());
                }
                if store_mu {
                    mu1_out[base..base + 8].copy_from_slice(&mu1.to_array());
                    mu2_out[base..base + 8].copy_from_slice(&mu2.to_array());
                }
                if store_sigma {
                    ssq_out[base..base + 8].copy_from_slice(&ssq.to_array());
                    s12_out[base..base + 8].copy_from_slice(&s12.to_array());
                }

                // Edge
                let diff1 = (s - mu1).abs();
                let diff2 = (d - mu2).abs();
                let ed = (one + diff2) / (one + diff1) - one;
                let artifact = ed.max(zero);
                let detail_lost = (-ed).max(zero);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                if !free.omit_edges {
                    acc.edge_art += artifact.reduce_add() as f64;
                    acc.edge_art4 += a4.reduce_add() as f64;
                    acc.edge_art2 += a2.reduce_add() as f64;
                    acc.edge_det += detail_lost.reduce_add() as f64;
                    acc.edge_det4 += dl4.reduce_add() as f64;
                    acc.edge_det2 += dl2.reduce_add() as f64;
                    if !free.omit_edges {
                        acc.edge_art8 += (a4 * a4).reduce_add() as f64;
                        acc.edge_det8 += (dl4 * dl4).reduce_add() as f64;
                        acc.edge_art_max = acc.edge_art_max.max(artifact.reduce_max());
                        acc.edge_det_max = acc.edge_det_max.max(detail_lost.reduce_max());
                    }
                }

                // Variance
                let vs = s - mu1;
                let vd = d - mu2;
                if !free.local_only {
                    acc.hf_sq_src += (vs * vs).reduce_add() as f64;
                    acc.hf_sq_dst += (vd * vd).reduce_add() as f64;
                }

                // Texture
                if !free.local_only {
                    acc.hf_abs_src += diff1.reduce_add() as f64;
                    acc.hf_abs_dst += diff2.reduce_add() as f64;
                }

                // MSE
                let pd = s - d;
                acc.mse += (pd * pd).reduce_add() as f64;
                if ext.on {
                    let act = sum_act * inv_v;
                    ext_accumulate8(token, &mut acc, sd, ed, pd, act, ext);
                }

                // === Free raw moments (`raw_moments`) ===
                // Plain sums of the raw pixels already in registers — no
                // new plane, no new load, no new pass. They finalize the
                // append block's GLOBAL_DMEAN / GLOBAL_CGAIN / GLOBAL_CLOSS
                // and append2's LUMA_MEAN_REF, which are the only 944 slots
                // whose value is a function of the RAW planes alone (see
                // `benchmarks/free_features_2026-09-01.md`). UNLIKE every
                // f64-reduce accumulator above it, this vector-adds across
                // rows (`fm_*`) with no per-row reduce_add, then reduces
                // ONCE at the band's last inner row — reduce_add is a
                // horizontal SIMD op, so 4 of them every row (one per
                // accumulator) is the shape this avoids; the doc above
                // prices the result. The vector sum is bounded to
                // `V1_BAND_ROWS` (32) rows of f32 before the f64 upgrade,
                // so the reorder costs negligible precision: worst |Δ| vs
                // the 944 append block is 5.35e-6 (was 4.62e-6 pre-batch),
                // ~2 orders below the module's 5e-4 tolerance
                // (`free_extras_match_the_944_append_block`).
                if free.raw_moments {
                    raw_moments_accumulate8(
                        &mut fm_s, &mut fm_d, &mut fm_s2, &mut fm_d2, &mut fm_dd, &mut fm_ds, s, d,
                    );
                    if y + 1 == inner_end {
                        raw_moments_finish8(&mut acc, fm_s, fm_d, fm_s2, fm_d2, fm_dd, fm_ds);
                    }
                }

                // === Free BOUNDED ERROR (`free.bounded_err` / `lum_bins`) ===
                // `pd` is the same register the `acc.mse` line above just
                // used; `s` is this channel's source row, which for the Y
                // channel IS the reference-luma plane the bins weight by.
                // No new plane, no new load, no new pass (class C — see the
                // helper block at the top of this module).
                if free.bounded_err {
                    let be_i = bounded_err_accumulate8(token, &mut be_m, pd);
                    if free.lum_bins {
                        lum_bins_accumulate8(
                            token,
                            &mut be_wdn,
                            &mut be_wdd,
                            &mut be_wbn,
                            &mut be_wbd,
                            s,
                            be_i,
                        );
                    }
                    if y + 1 == inner_end {
                        bounded_err_finish8(&mut acc, be_m);
                        if free.lum_bins {
                            lum_bins_finish8(&mut acc, be_wdn, be_wdd, be_wbn, be_wbd);
                        }
                    }
                }
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            let aadd = add_idx * width + col_base;
            let arem = rem_idx * width + col_base;
            sum_m1 = sum_m1
                + f32x8::from_array_t(token, h_mu1[add_base..][..8].try_into().unwrap())
                - f32x8::from_array_t(token, h_mu1[rem_base..][..8].try_into().unwrap());
            sum_m2 = sum_m2
                + f32x8::from_array_t(token, h_mu2[add_base..][..8].try_into().unwrap())
                - f32x8::from_array_t(token, h_mu2[rem_base..][..8].try_into().unwrap());
            sum_sq = sum_sq
                + f32x8::from_array_t(token, h_sigma_sq[add_base..][..8].try_into().unwrap())
                - f32x8::from_array_t(token, h_sigma_sq[rem_base..][..8].try_into().unwrap());
            sum_s12 = sum_s12
                + f32x8::from_array_t(token, h_sigma12[add_base..][..8].try_into().unwrap())
                - f32x8::from_array_t(token, h_sigma12[rem_base..][..8].try_into().unwrap());
            if ext.on {
                sum_act = sum_act
                    + f32x8::from_array_t(token, h_act[aadd..][..8].try_into().unwrap())
                    - f32x8::from_array_t(token, h_act[arem..][..8].try_into().unwrap());
            }
        }
    }

    // Scalar remainder
    let inv = 1.0 / diam as f32;
    for x in (col_groups * 8)..width {
        let mut sum_m1 = 0.0f32;
        let mut sum_m2 = 0.0f32;
        let mut sum_sq = 0.0f32;
        let mut sum_s12 = 0.0f32;
        let mut sum_act = 0.0f32;
        // Free raw moments: one lane accumulator per column group, reduced
        // at the band's last inner row (see `raw_moments`). Dead code when
        // the caller did not ask.
        let (mut fm_s, mut fm_d, mut fm_s2, mut fm_d2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        // Revision 2's paired second moment and first difference (see
        // `raw_moments_accumulate16`).
        let mut fm_dd = 0.0f32;
        let mut fm_ds = 0.0f32;
        // Free bounded-error lane accumulators (`free.bounded_err` /
        // `free.lum_bins`) — same band-batched shape and rationale as
        // `fm_*` above: vector-add every row, reduce ONCE at the band's
        // last inner row.
        let (mut be_m, mut be_wdn, mut be_wdd, mut be_wbn, mut be_wbd) =
            (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            sum_m1 += h_mu1[idx * width + x];
            sum_m2 += h_mu2[idx * width + x];
            sum_sq += h_sigma_sq[idx * width + x];
            sum_s12 += h_sigma12[idx * width + x];
            if ext.on {
                sum_act += h_act[idx * width + x];
            }
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let mu1 = sum_m1 * inv;
                let mu2 = sum_m2 * inv;
                let ssq = sum_sq * inv;
                let s12 = sum_s12 * inv;
                let sv = src[y * width + x];
                let dv = dst[y * width + x];

                // SSIM
                let sd = if direct {
                    ssim_direct_raw_scalar(form, mu1, mu2, ssq, s12).max(0.0f32)
                } else {
                    (ssim_dissim_raw_scalar(form, mu1, mu2, ssq, s12)).max(0.0f32)
                };
                let sd2 = sd * sd;
                let sd4 = sd2 * sd2;
                acc.ssim_d += sd as f64;
                acc.ssim_d4 += sd4 as f64;
                acc.ssim_d2 += sd2 as f64;
                if !free.local_only {
                    acc.ssim_d8 += (sd4 * sd4) as f64;
                    acc.ssim_max = acc.ssim_max.max(sd);
                }
                if store_sd {
                    sd_out[y * width + x] = sd;
                }
                if store_mu {
                    mu1_out[y * width + x] = mu1;
                    mu2_out[y * width + x] = mu2;
                }
                if store_sigma {
                    ssq_out[y * width + x] = ssq;
                    s12_out[y * width + x] = s12;
                }

                // Edge
                let diff1 = (sv - mu1).abs();
                let diff2 = (dv - mu2).abs();
                let ed = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;
                let artifact = ed.max(0.0f32);
                let detail_lost = (-ed).max(0.0f32);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                if !free.omit_edges {
                    acc.edge_art += artifact as f64;
                    acc.edge_art4 += a4 as f64;
                    acc.edge_art2 += a2 as f64;
                    acc.edge_det += detail_lost as f64;
                    acc.edge_det4 += dl4 as f64;
                    acc.edge_det2 += dl2 as f64;
                    if !free.omit_edges {
                        acc.edge_art8 += (a4 * a4) as f64;
                        acc.edge_det8 += (dl4 * dl4) as f64;
                        acc.edge_art_max = acc.edge_art_max.max(artifact);
                        acc.edge_det_max = acc.edge_det_max.max(detail_lost);
                    }
                }

                // Variance
                let vs = sv - mu1;
                let vd = dv - mu2;
                if !free.local_only {
                    acc.hf_sq_src += (vs * vs) as f64;
                    acc.hf_sq_dst += (vd * vd) as f64;
                }

                // Texture
                if !free.local_only {
                    acc.hf_abs_src += diff1 as f64;
                    acc.hf_abs_dst += diff2 as f64;
                }

                // MSE
                let pd = sv - dv;
                acc.mse += (pd * pd) as f64;
                if ext.on {
                    let act = sum_act * inv;
                    ext_accumulate_scalar(&mut acc, sd, ed, pd, act, ext);
                }

                // === Free raw moments (`raw_moments`) — scalar tail ===
                if free.raw_moments {
                    raw_moments_accumulate_scalar(
                        &mut fm_s, &mut fm_d, &mut fm_s2, &mut fm_d2, &mut fm_dd, &mut fm_ds, sv,
                        dv,
                    );
                    if y + 1 == inner_end {
                        raw_moments_finish_scalar(&mut acc, fm_s, fm_d, fm_s2, fm_d2, fm_dd, fm_ds);
                    }
                }

                // === Free BOUNDED ERROR (`free.bounded_err` / `lum_bins`) ===
                // `pd` is the same register the `acc.mse` line above just
                // used; `s` is this channel's source row, which for the Y
                // channel IS the reference-luma plane the bins weight by.
                // No new plane, no new load, no new pass (class C — see the
                // helper block at the top of this module).
                if free.bounded_err {
                    let be_i = bounded_err_accumulate_scalar(&mut be_m, pd);
                    if free.lum_bins {
                        lum_bins_accumulate_scalar(
                            &mut be_wdn,
                            &mut be_wdd,
                            &mut be_wbn,
                            &mut be_wbd,
                            sv,
                            be_i,
                        );
                    }
                    if y + 1 == inner_end {
                        bounded_err_finish_scalar(&mut acc, be_m);
                        if free.lum_bins {
                            lum_bins_finish_scalar(&mut acc, be_wdn, be_wdd, be_wbn, be_wbd);
                        }
                    }
                }
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            sum_m1 = sum_m1 + h_mu1[add_idx * width + x] - h_mu1[rem_idx * width + x];
            sum_m2 = sum_m2 + h_mu2[add_idx * width + x] - h_mu2[rem_idx * width + x];
            sum_sq = sum_sq + h_sigma_sq[add_idx * width + x] - h_sigma_sq[rem_idx * width + x];
            sum_s12 = sum_s12 + h_sigma12[add_idx * width + x] - h_sigma12[rem_idx * width + x];
            if ext.on {
                sum_act = sum_act + h_act[add_idx * width + x] - h_act[rem_idx * width + x];
            }
        }
    }

    acc
}

// ============================================================
// Generic WASM128 + Scalar SSIM implementation (via magetypes)
// ============================================================

#[magetypes(neon, wasm128, scalar)]
fn fused_vblur_ssim_inner(
    token: Token,
    h_mu1: &[f32],
    h_mu2: &[f32],
    h_sigma_sq: &[f32],
    h_sigma12: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    sd_out: &mut [f32],
    store_sd: bool,
    ssq_out: &mut [f32],
    s12_out: &mut [f32],
    store_sigma: bool,
    // `free`: which FREE extra accumulators to carry alongside the existing
    // sums — the four raw moments (Σs, Σd, Σs², Σd²) and/or the class-C
    // bounded-error family. See [`FreeExtrasWork`] and
    // `StripChannelAccum::sum_s` / `sum_msat`.
    free: FreeExtrasWork,
    // Revision 3: form the dissimilarity with the direct-error moment that the
    // H pass put in `h_sigma12` (see `blur::fused_blur_h_ssim`).
    direct: bool,
    ext: ExtPoolsWork,
    h_act: &[f32],
) -> StripChannelAccum {
    let form = free.luma_form();
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    let diam = 2 * radius + 1;
    let inv_v = f32x8::splat_t(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 8;

    let one = f32x8::splat_t(token, 1.0);
    let zero = f32x8::zero_t(token);

    let mut acc = StripChannelAccum::zero();
    let inner_end = inner_start + inner_h;

    for cg in 0..col_groups {
        let col_base = cg * 8;
        let mut sum_m1_a = [0.0f32; 8];
        let mut sum_m2_a = [0.0f32; 8];
        let mut sum_sq_a = [0.0f32; 8];
        let mut sum_s12_a = [0.0f32; 8];
        let mut sum_act_a = [0.0f32; 8];
        // Free raw moments: one lane accumulator per column group, reduced
        // at the band's last inner row (see `raw_moments`). Dead code when
        // the caller did not ask.
        let (mut fm_s, mut fm_d, mut fm_s2, mut fm_d2) = (
            f32x8::zero_t(token),
            f32x8::zero_t(token),
            f32x8::zero_t(token),
            f32x8::zero_t(token),
        );
        // Revision 2\'s paired second moment (see `raw_moments_accumulate16`).
        let mut fm_dd = f32x8::zero_t(token);
        let mut fm_ds = f32x8::zero_t(token);
        // Free bounded-error lane accumulators (`free.bounded_err` /
        // `free.lum_bins`) — same band-batched shape and rationale as
        // `fm_*` above: vector-add every row, reduce ONCE at the band's
        // last inner row.
        let (mut be_m, mut be_wdn, mut be_wdd, mut be_wbn, mut be_wbd) = (
            f32x8::zero_t(token),
            f32x8::zero_t(token),
            f32x8::zero_t(token),
            f32x8::zero_t(token),
            f32x8::zero_t(token),
        );

        // Initialize running sums
        {
            let mut sm1 = f32x8::zero_t(token);
            let mut sm2 = f32x8::zero_t(token);
            let mut ssq = f32x8::zero_t(token);
            let mut ss12 = f32x8::zero_t(token);
            let mut sact = f32x8::zero_t(token);
            for i in 0..diam {
                let idx = mirror_idx(i, r, height);
                let base = idx * width + col_base;
                let abase = idx * width + col_base;
                sm1 = sm1 + f32x8::from_array_t(token, h_mu1[base..][..8].try_into().unwrap());
                sm2 = sm2 + f32x8::from_array_t(token, h_mu2[base..][..8].try_into().unwrap());
                ssq = ssq + f32x8::from_array_t(token, h_sigma_sq[base..][..8].try_into().unwrap());
                ss12 =
                    ss12 + f32x8::from_array_t(token, h_sigma12[base..][..8].try_into().unwrap());
                if ext.on {
                    sact =
                        sact + f32x8::from_array_t(token, h_act[abase..][..8].try_into().unwrap());
                }
            }
            sm1.store(&mut sum_m1_a);
            sm2.store(&mut sum_m2_a);
            ssq.store(&mut sum_sq_a);
            ss12.store(&mut sum_s12_a);
            sact.store(&mut sum_act_a);
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;
                let sum_m1 = f32x8::from_array_t(token, sum_m1_a);
                let sum_m2 = f32x8::from_array_t(token, sum_m2_a);
                let sum_sq = f32x8::from_array_t(token, sum_sq_a);
                let sum_s12 = f32x8::from_array_t(token, sum_s12_a);

                let mu1 = sum_m1 * inv_v;
                let mu2 = sum_m2 * inv_v;
                let ssq = sum_sq * inv_v;
                let s12 = sum_s12 * inv_v;
                let s = f32x8::from_array_t(token, src[base..][..8].try_into().unwrap());
                let d = f32x8::from_array_t(token, dst[base..][..8].try_into().unwrap());

                // SSIM
                let sd = if direct {
                    ssim_direct8(token, form, mu1, mu2, ssq, s12).max(zero)
                } else {
                    (ssim_dissim8(token, form, mu1, mu2, ssq, s12)).max(zero)
                };
                let sd2 = sd * sd;
                let sd4 = sd2 * sd2;
                acc.ssim_d += sd.reduce_add() as f64;
                acc.ssim_d4 += sd4.reduce_add() as f64;
                acc.ssim_d2 += sd2.reduce_add() as f64;
                if !free.local_only {
                    acc.ssim_d8 += (sd4 * sd4).reduce_add() as f64;
                    acc.ssim_max = acc.ssim_max.max(sd.reduce_max());
                }
                if store_sd {
                    sd.store((&mut sd_out[base..base + 8]).try_into().unwrap());
                }
                if store_mu {
                    mu1.store((&mut mu1_out[base..base + 8]).try_into().unwrap());
                    mu2.store((&mut mu2_out[base..base + 8]).try_into().unwrap());
                }
                if store_sigma {
                    ssq.store((&mut ssq_out[base..base + 8]).try_into().unwrap());
                    s12.store((&mut s12_out[base..base + 8]).try_into().unwrap());
                }

                // Edge
                let diff1 = (s - mu1).abs();
                let diff2 = (d - mu2).abs();
                let ed = (one + diff2) / (one + diff1) - one;
                let artifact = ed.max(zero);
                let detail_lost = (-ed).max(zero);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                if !free.omit_edges {
                    acc.edge_art += artifact.reduce_add() as f64;
                    acc.edge_art4 += a4.reduce_add() as f64;
                    acc.edge_art2 += a2.reduce_add() as f64;
                    acc.edge_det += detail_lost.reduce_add() as f64;
                    acc.edge_det4 += dl4.reduce_add() as f64;
                    acc.edge_det2 += dl2.reduce_add() as f64;
                    if !free.omit_edges {
                        acc.edge_art8 += (a4 * a4).reduce_add() as f64;
                        acc.edge_det8 += (dl4 * dl4).reduce_add() as f64;
                        acc.edge_art_max = acc.edge_art_max.max(artifact.reduce_max());
                        acc.edge_det_max = acc.edge_det_max.max(detail_lost.reduce_max());
                    }
                }

                // Variance
                let vs = s - mu1;
                let vd = d - mu2;
                if !free.local_only {
                    acc.hf_sq_src += (vs * vs).reduce_add() as f64;
                    acc.hf_sq_dst += (vd * vd).reduce_add() as f64;
                }

                // Texture
                if !free.local_only {
                    acc.hf_abs_src += diff1.reduce_add() as f64;
                    acc.hf_abs_dst += diff2.reduce_add() as f64;
                }

                // MSE
                let pd = s - d;
                acc.mse += (pd * pd).reduce_add() as f64;
                if ext.on {
                    let act = f32x8::from_array_t(token, sum_act_a) * inv_v;
                    ext_accumulate8(token, &mut acc, sd, ed, pd, act, ext);
                }

                // === Free raw moments (`raw_moments`) ===
                // Plain sums of the raw pixels already in registers — no
                // new plane, no new load, no new pass. They finalize the
                // append block's GLOBAL_DMEAN / GLOBAL_CGAIN / GLOBAL_CLOSS
                // and append2's LUMA_MEAN_REF, which are the only 944 slots
                // whose value is a function of the RAW planes alone (see
                // `benchmarks/free_features_2026-09-01.md`). UNLIKE every
                // f64-reduce accumulator above it, this vector-adds across
                // rows (`fm_*`) with no per-row reduce_add, then reduces
                // ONCE at the band's last inner row — reduce_add is a
                // horizontal SIMD op, so 4 of them every row (one per
                // accumulator) is the shape this avoids; the doc above
                // prices the result. The vector sum is bounded to
                // `V1_BAND_ROWS` (32) rows of f32 before the f64 upgrade,
                // so the reorder costs negligible precision: worst |Δ| vs
                // the 944 append block is 5.35e-6 (was 4.62e-6 pre-batch),
                // ~2 orders below the module's 5e-4 tolerance
                // (`free_extras_match_the_944_append_block`).
                if free.raw_moments {
                    raw_moments_accumulate8(
                        &mut fm_s, &mut fm_d, &mut fm_s2, &mut fm_d2, &mut fm_dd, &mut fm_ds, s, d,
                    );
                    if y + 1 == inner_end {
                        raw_moments_finish8(&mut acc, fm_s, fm_d, fm_s2, fm_d2, fm_dd, fm_ds);
                    }
                }

                // === Free BOUNDED ERROR (`free.bounded_err` / `lum_bins`) ===
                // `pd` is the same register the `acc.mse` line above just
                // used; `s` is this channel's source row, which for the Y
                // channel IS the reference-luma plane the bins weight by.
                // No new plane, no new load, no new pass (class C — see the
                // helper block at the top of this module).
                if free.bounded_err {
                    let be_i = bounded_err_accumulate8(token, &mut be_m, pd);
                    if free.lum_bins {
                        lum_bins_accumulate8(
                            token,
                            &mut be_wdn,
                            &mut be_wdd,
                            &mut be_wbn,
                            &mut be_wbd,
                            s,
                            be_i,
                        );
                    }
                    if y + 1 == inner_end {
                        bounded_err_finish8(&mut acc, be_m);
                        if free.lum_bins {
                            lum_bins_finish8(&mut acc, be_wdn, be_wdd, be_wbn, be_wbd);
                        }
                    }
                }
            }

            // Slide V-blur window
            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            let aadd = add_idx * width + col_base;
            let arem = rem_idx * width + col_base;
            let new_m1 = f32x8::from_array_t(token, sum_m1_a)
                + f32x8::from_array_t(token, h_mu1[add_base..][..8].try_into().unwrap())
                - f32x8::from_array_t(token, h_mu1[rem_base..][..8].try_into().unwrap());
            let new_m2 = f32x8::from_array_t(token, sum_m2_a)
                + f32x8::from_array_t(token, h_mu2[add_base..][..8].try_into().unwrap())
                - f32x8::from_array_t(token, h_mu2[rem_base..][..8].try_into().unwrap());
            let new_sq = f32x8::from_array_t(token, sum_sq_a)
                + f32x8::from_array_t(token, h_sigma_sq[add_base..][..8].try_into().unwrap())
                - f32x8::from_array_t(token, h_sigma_sq[rem_base..][..8].try_into().unwrap());
            let new_s12 = f32x8::from_array_t(token, sum_s12_a)
                + f32x8::from_array_t(token, h_sigma12[add_base..][..8].try_into().unwrap())
                - f32x8::from_array_t(token, h_sigma12[rem_base..][..8].try_into().unwrap());
            new_m1.store(&mut sum_m1_a);
            new_m2.store(&mut sum_m2_a);
            new_sq.store(&mut sum_sq_a);
            new_s12.store(&mut sum_s12_a);
            if ext.on {
                let new_act = f32x8::from_array_t(token, sum_act_a)
                    + f32x8::from_array_t(token, h_act[aadd..][..8].try_into().unwrap())
                    - f32x8::from_array_t(token, h_act[arem..][..8].try_into().unwrap());
                new_act.store(&mut sum_act_a);
            }
        }
    }

    // Scalar remainder
    let inv = 1.0 / diam as f32;
    for x in (col_groups * 8)..width {
        let mut sum_m1 = 0.0f32;
        let mut sum_m2 = 0.0f32;
        let mut sum_sq = 0.0f32;
        let mut sum_s12 = 0.0f32;
        let mut sum_act = 0.0f32;
        // Free raw moments: one lane accumulator per column group, reduced
        // at the band's last inner row (see `raw_moments`). Dead code when
        // the caller did not ask.
        let (mut fm_s, mut fm_d, mut fm_s2, mut fm_d2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        // Revision 2's paired second moment and first difference (see
        // `raw_moments_accumulate16`).
        let mut fm_dd = 0.0f32;
        let mut fm_ds = 0.0f32;
        // Free bounded-error lane accumulators (`free.bounded_err` /
        // `free.lum_bins`) — same band-batched shape and rationale as
        // `fm_*` above: vector-add every row, reduce ONCE at the band's
        // last inner row.
        let (mut be_m, mut be_wdn, mut be_wdd, mut be_wbn, mut be_wbd) =
            (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            sum_m1 += h_mu1[idx * width + x];
            sum_m2 += h_mu2[idx * width + x];
            sum_sq += h_sigma_sq[idx * width + x];
            sum_s12 += h_sigma12[idx * width + x];
            if ext.on {
                sum_act += h_act[idx * width + x];
            }
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let mu1 = sum_m1 * inv;
                let mu2 = sum_m2 * inv;
                let ssq = sum_sq * inv;
                let s12 = sum_s12 * inv;
                let sv = src[y * width + x];
                let dv = dst[y * width + x];

                // SSIM
                let sd = if direct {
                    ssim_direct_raw_scalar(form, mu1, mu2, ssq, s12).max(0.0f32)
                } else {
                    (ssim_dissim_raw_scalar(form, mu1, mu2, ssq, s12)).max(0.0f32)
                };
                let sd2 = sd * sd;
                let sd4 = sd2 * sd2;
                acc.ssim_d += sd as f64;
                acc.ssim_d4 += sd4 as f64;
                acc.ssim_d2 += sd2 as f64;
                if !free.local_only {
                    acc.ssim_d8 += (sd4 * sd4) as f64;
                    acc.ssim_max = acc.ssim_max.max(sd);
                }
                if store_sd {
                    sd_out[y * width + x] = sd;
                }
                if store_mu {
                    mu1_out[y * width + x] = mu1;
                    mu2_out[y * width + x] = mu2;
                }
                if store_sigma {
                    ssq_out[y * width + x] = ssq;
                    s12_out[y * width + x] = s12;
                }

                // Edge
                let diff1 = (sv - mu1).abs();
                let diff2 = (dv - mu2).abs();
                let ed = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;
                let artifact = ed.max(0.0f32);
                let detail_lost = (-ed).max(0.0f32);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                if !free.omit_edges {
                    acc.edge_art += artifact as f64;
                    acc.edge_art4 += a4 as f64;
                    acc.edge_art2 += a2 as f64;
                    acc.edge_det += detail_lost as f64;
                    acc.edge_det4 += dl4 as f64;
                    acc.edge_det2 += dl2 as f64;
                    if !free.omit_edges {
                        acc.edge_art8 += (a4 * a4) as f64;
                        acc.edge_det8 += (dl4 * dl4) as f64;
                        acc.edge_art_max = acc.edge_art_max.max(artifact);
                        acc.edge_det_max = acc.edge_det_max.max(detail_lost);
                    }
                }

                // Variance
                let vs = sv - mu1;
                let vd = dv - mu2;
                if !free.local_only {
                    acc.hf_sq_src += (vs * vs) as f64;
                    acc.hf_sq_dst += (vd * vd) as f64;
                }

                // Texture
                if !free.local_only {
                    acc.hf_abs_src += diff1 as f64;
                    acc.hf_abs_dst += diff2 as f64;
                }

                // MSE
                let pd = sv - dv;
                acc.mse += (pd * pd) as f64;
                if ext.on {
                    let act = sum_act * inv;
                    ext_accumulate_scalar(&mut acc, sd, ed, pd, act, ext);
                }

                // === Free raw moments (`raw_moments`) — scalar tail ===
                if free.raw_moments {
                    raw_moments_accumulate_scalar(
                        &mut fm_s, &mut fm_d, &mut fm_s2, &mut fm_d2, &mut fm_dd, &mut fm_ds, sv,
                        dv,
                    );
                    if y + 1 == inner_end {
                        raw_moments_finish_scalar(&mut acc, fm_s, fm_d, fm_s2, fm_d2, fm_dd, fm_ds);
                    }
                }

                // === Free BOUNDED ERROR (`free.bounded_err` / `lum_bins`) ===
                // `pd` is the same register the `acc.mse` line above just
                // used; `s` is this channel's source row, which for the Y
                // channel IS the reference-luma plane the bins weight by.
                // No new plane, no new load, no new pass (class C — see the
                // helper block at the top of this module).
                if free.bounded_err {
                    let be_i = bounded_err_accumulate_scalar(&mut be_m, pd);
                    if free.lum_bins {
                        lum_bins_accumulate_scalar(
                            &mut be_wdn,
                            &mut be_wdd,
                            &mut be_wbn,
                            &mut be_wbd,
                            sv,
                            be_i,
                        );
                    }
                    if y + 1 == inner_end {
                        bounded_err_finish_scalar(&mut acc, be_m);
                        if free.lum_bins {
                            lum_bins_finish_scalar(&mut acc, be_wdn, be_wdd, be_wbn, be_wbd);
                        }
                    }
                }
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            sum_m1 = sum_m1 + h_mu1[add_idx * width + x] - h_mu1[rem_idx * width + x];
            sum_m2 = sum_m2 + h_mu2[add_idx * width + x] - h_mu2[rem_idx * width + x];
            sum_sq = sum_sq + h_sigma_sq[add_idx * width + x] - h_sigma_sq[rem_idx * width + x];
            sum_s12 = sum_s12 + h_sigma12[add_idx * width + x] - h_sigma12[rem_idx * width + x];
            if ext.on {
                sum_act = sum_act + h_act[add_idx * width + x] - h_act[rem_idx * width + x];
            }
        }
    }

    acc
}

// ============================================================
// Edge-only fused V-blur (no SSIM, only 2 running sums)
// ============================================================

#[cfg(target_arch = "x86_64")]
#[arcane]
fn fused_vblur_edge_inner_v4(
    token: archmage::X64V4Token,
    h_mu1: &[f32],
    h_mu2: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
) -> StripChannelAccum {
    let diam = 2 * radius + 1;
    let inv_v = f32x16::splat_t(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 16;

    let one = f32x16::splat_t(token, 1.0);
    let zero = f32x16::zero_t(token);

    let mut acc = StripChannelAccum::zero();
    let inner_end = inner_start + inner_h;

    for cg in 0..col_groups {
        let col_base = cg * 16;
        let mut sum_m1 = f32x16::zero_t(token);
        let mut sum_m2 = f32x16::zero_t(token);

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            let base = idx * width + col_base;
            sum_m1 = sum_m1 + f32x16::from_array_t(token, h_mu1[base..][..16].try_into().unwrap());
            sum_m2 = sum_m2 + f32x16::from_array_t(token, h_mu2[base..][..16].try_into().unwrap());
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;
                let mu1 = sum_m1 * inv_v;
                let mu2 = sum_m2 * inv_v;
                let s = f32x16::from_array_t(token, src[base..][..16].try_into().unwrap());
                let d = f32x16::from_array_t(token, dst[base..][..16].try_into().unwrap());

                if store_mu {
                    mu1_out[base..base + 16].copy_from_slice(&mu1.to_array());
                    mu2_out[base..base + 16].copy_from_slice(&mu2.to_array());
                }

                // Edge
                let diff1 = (s - mu1).abs();
                let diff2 = (d - mu2).abs();
                let ed = (one + diff2) / (one + diff1) - one;
                let artifact = ed.max(zero);
                let detail_lost = (-ed).max(zero);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                acc.edge_art += artifact.reduce_add() as f64;
                acc.edge_art4 += a4.reduce_add() as f64;
                acc.edge_art2 += a2.reduce_add() as f64;
                acc.edge_det += detail_lost.reduce_add() as f64;
                acc.edge_det4 += dl4.reduce_add() as f64;
                acc.edge_det2 += dl2.reduce_add() as f64;
                acc.edge_art8 += (a4 * a4).reduce_add() as f64;
                acc.edge_det8 += (dl4 * dl4).reduce_add() as f64;
                acc.edge_art_max = acc.edge_art_max.max(artifact.reduce_max());
                acc.edge_det_max = acc.edge_det_max.max(detail_lost.reduce_max());

                // Variance
                let vs = s - mu1;
                let vd = d - mu2;
                acc.hf_sq_src += (vs * vs).reduce_add() as f64;
                acc.hf_sq_dst += (vd * vd).reduce_add() as f64;

                // Texture
                acc.hf_abs_src += diff1.reduce_add() as f64;
                acc.hf_abs_dst += diff2.reduce_add() as f64;

                // MSE
                let pd = s - d;
                acc.mse += (pd * pd).reduce_add() as f64;
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            sum_m1 = sum_m1
                + f32x16::from_array_t(token, h_mu1[add_base..][..16].try_into().unwrap())
                - f32x16::from_array_t(token, h_mu1[rem_base..][..16].try_into().unwrap());
            sum_m2 = sum_m2
                + f32x16::from_array_t(token, h_mu2[add_base..][..16].try_into().unwrap())
                - f32x16::from_array_t(token, h_mu2[rem_base..][..16].try_into().unwrap());
        }
    }

    // Remainder with f32x8
    let col_base_8 = col_groups * 16;
    let v3 = token.v3();
    let inv_v8 = f32x8::splat_t(v3, 1.0 / diam as f32);
    let remaining_8groups = (width - col_base_8) / 8;

    let one8 = f32x8::splat_t(v3, 1.0);
    let zero8 = f32x8::zero_t(v3);

    for cg in 0..remaining_8groups {
        let col_base = col_base_8 + cg * 8;
        let mut sum_m1 = f32x8::zero_t(v3);
        let mut sum_m2 = f32x8::zero_t(v3);

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            let base = idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array_t(v3, h_mu1[base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array_t(v3, h_mu2[base..][..8].try_into().unwrap());
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;
                let mu1 = sum_m1 * inv_v8;
                let mu2 = sum_m2 * inv_v8;
                let s = f32x8::from_array_t(v3, src[base..][..8].try_into().unwrap());
                let d = f32x8::from_array_t(v3, dst[base..][..8].try_into().unwrap());

                if store_mu {
                    mu1_out[base..base + 8].copy_from_slice(&mu1.to_array());
                    mu2_out[base..base + 8].copy_from_slice(&mu2.to_array());
                }

                let diff1 = (s - mu1).abs();
                let diff2 = (d - mu2).abs();
                let ed = (one8 + diff2) / (one8 + diff1) - one8;
                let artifact = ed.max(zero8);
                let detail_lost = (-ed).max(zero8);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                acc.edge_art += artifact.reduce_add() as f64;
                acc.edge_art4 += a4.reduce_add() as f64;
                acc.edge_art2 += a2.reduce_add() as f64;
                acc.edge_det += detail_lost.reduce_add() as f64;
                acc.edge_det4 += dl4.reduce_add() as f64;
                acc.edge_det2 += dl2.reduce_add() as f64;
                acc.edge_art8 += (a4 * a4).reduce_add() as f64;
                acc.edge_det8 += (dl4 * dl4).reduce_add() as f64;
                acc.edge_art_max = acc.edge_art_max.max(artifact.reduce_max());
                acc.edge_det_max = acc.edge_det_max.max(detail_lost.reduce_max());

                let vs = s - mu1;
                let vd = d - mu2;
                acc.hf_sq_src += (vs * vs).reduce_add() as f64;
                acc.hf_sq_dst += (vd * vd).reduce_add() as f64;
                acc.hf_abs_src += diff1.reduce_add() as f64;
                acc.hf_abs_dst += diff2.reduce_add() as f64;

                let pd = s - d;
                acc.mse += (pd * pd).reduce_add() as f64;
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array_t(v3, h_mu1[add_base..][..8].try_into().unwrap())
                - f32x8::from_array_t(v3, h_mu1[rem_base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array_t(v3, h_mu2[add_base..][..8].try_into().unwrap())
                - f32x8::from_array_t(v3, h_mu2[rem_base..][..8].try_into().unwrap());
        }
    }

    // Scalar remainder
    let inv = 1.0 / diam as f32;
    for x in (col_base_8 + remaining_8groups * 8)..width {
        let mut sum_m1 = 0.0f32;
        let mut sum_m2 = 0.0f32;

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            sum_m1 += h_mu1[idx * width + x];
            sum_m2 += h_mu2[idx * width + x];
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let mu1 = sum_m1 * inv;
                let mu2 = sum_m2 * inv;
                let sv = src[y * width + x];
                let dv = dst[y * width + x];

                if store_mu {
                    mu1_out[y * width + x] = mu1;
                    mu2_out[y * width + x] = mu2;
                }

                // Edge
                let diff1 = (sv - mu1).abs();
                let diff2 = (dv - mu2).abs();
                let ed = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;
                let artifact = ed.max(0.0f32);
                let detail_lost = (-ed).max(0.0f32);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                acc.edge_art += artifact as f64;
                acc.edge_art4 += a4 as f64;
                acc.edge_art2 += a2 as f64;
                acc.edge_det += detail_lost as f64;
                acc.edge_det4 += dl4 as f64;
                acc.edge_det2 += dl2 as f64;
                acc.edge_art8 += (a4 * a4) as f64;
                acc.edge_det8 += (dl4 * dl4) as f64;
                acc.edge_art_max = acc.edge_art_max.max(artifact);
                acc.edge_det_max = acc.edge_det_max.max(detail_lost);

                let vs = sv - mu1;
                let vd = dv - mu2;
                acc.hf_sq_src += (vs * vs) as f64;
                acc.hf_sq_dst += (vd * vd) as f64;
                acc.hf_abs_src += diff1 as f64;
                acc.hf_abs_dst += diff2 as f64;

                let pd = sv - dv;
                acc.mse += (pd * pd) as f64;
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            sum_m1 = sum_m1 + h_mu1[add_idx * width + x] - h_mu1[rem_idx * width + x];
            sum_m2 = sum_m2 + h_mu2[add_idx * width + x] - h_mu2[rem_idx * width + x];
        }
    }

    acc
}
#[cfg(target_arch = "x86_64")]
#[arcane]
fn fused_vblur_edge_inner_v4x(
    token: archmage::X64V4xToken,
    h_mu1: &[f32],
    h_mu2: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
) -> StripChannelAccum {
    let diam = 2 * radius + 1;
    let inv_v = f32x16::splat_t(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 16;

    let one = f32x16::splat_t(token, 1.0);
    let zero = f32x16::zero_t(token);

    let mut acc = StripChannelAccum::zero();
    let inner_end = inner_start + inner_h;

    for cg in 0..col_groups {
        let col_base = cg * 16;
        let mut sum_m1 = f32x16::zero_t(token);
        let mut sum_m2 = f32x16::zero_t(token);

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            let base = idx * width + col_base;
            sum_m1 = sum_m1 + f32x16::from_array_t(token, h_mu1[base..][..16].try_into().unwrap());
            sum_m2 = sum_m2 + f32x16::from_array_t(token, h_mu2[base..][..16].try_into().unwrap());
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;
                let mu1 = sum_m1 * inv_v;
                let mu2 = sum_m2 * inv_v;
                let s = f32x16::from_array_t(token, src[base..][..16].try_into().unwrap());
                let d = f32x16::from_array_t(token, dst[base..][..16].try_into().unwrap());

                if store_mu {
                    mu1_out[base..base + 16].copy_from_slice(&mu1.to_array());
                    mu2_out[base..base + 16].copy_from_slice(&mu2.to_array());
                }

                // Edge
                let diff1 = (s - mu1).abs();
                let diff2 = (d - mu2).abs();
                let ed = (one + diff2) / (one + diff1) - one;
                let artifact = ed.max(zero);
                let detail_lost = (-ed).max(zero);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                acc.edge_art += artifact.reduce_add() as f64;
                acc.edge_art4 += a4.reduce_add() as f64;
                acc.edge_art2 += a2.reduce_add() as f64;
                acc.edge_det += detail_lost.reduce_add() as f64;
                acc.edge_det4 += dl4.reduce_add() as f64;
                acc.edge_det2 += dl2.reduce_add() as f64;
                acc.edge_art8 += (a4 * a4).reduce_add() as f64;
                acc.edge_det8 += (dl4 * dl4).reduce_add() as f64;
                acc.edge_art_max = acc.edge_art_max.max(artifact.reduce_max());
                acc.edge_det_max = acc.edge_det_max.max(detail_lost.reduce_max());

                // Variance
                let vs = s - mu1;
                let vd = d - mu2;
                acc.hf_sq_src += (vs * vs).reduce_add() as f64;
                acc.hf_sq_dst += (vd * vd).reduce_add() as f64;

                // Texture
                acc.hf_abs_src += diff1.reduce_add() as f64;
                acc.hf_abs_dst += diff2.reduce_add() as f64;

                // MSE
                let pd = s - d;
                acc.mse += (pd * pd).reduce_add() as f64;
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            sum_m1 = sum_m1
                + f32x16::from_array_t(token, h_mu1[add_base..][..16].try_into().unwrap())
                - f32x16::from_array_t(token, h_mu1[rem_base..][..16].try_into().unwrap());
            sum_m2 = sum_m2
                + f32x16::from_array_t(token, h_mu2[add_base..][..16].try_into().unwrap())
                - f32x16::from_array_t(token, h_mu2[rem_base..][..16].try_into().unwrap());
        }
    }

    // Remainder with f32x8
    let col_base_8 = col_groups * 16;
    let v3 = token.v3();
    let inv_v8 = f32x8::splat_t(v3, 1.0 / diam as f32);
    let remaining_8groups = (width - col_base_8) / 8;

    let one8 = f32x8::splat_t(v3, 1.0);
    let zero8 = f32x8::zero_t(v3);

    for cg in 0..remaining_8groups {
        let col_base = col_base_8 + cg * 8;
        let mut sum_m1 = f32x8::zero_t(v3);
        let mut sum_m2 = f32x8::zero_t(v3);

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            let base = idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array_t(v3, h_mu1[base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array_t(v3, h_mu2[base..][..8].try_into().unwrap());
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;
                let mu1 = sum_m1 * inv_v8;
                let mu2 = sum_m2 * inv_v8;
                let s = f32x8::from_array_t(v3, src[base..][..8].try_into().unwrap());
                let d = f32x8::from_array_t(v3, dst[base..][..8].try_into().unwrap());

                if store_mu {
                    mu1_out[base..base + 8].copy_from_slice(&mu1.to_array());
                    mu2_out[base..base + 8].copy_from_slice(&mu2.to_array());
                }

                let diff1 = (s - mu1).abs();
                let diff2 = (d - mu2).abs();
                let ed = (one8 + diff2) / (one8 + diff1) - one8;
                let artifact = ed.max(zero8);
                let detail_lost = (-ed).max(zero8);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                acc.edge_art += artifact.reduce_add() as f64;
                acc.edge_art4 += a4.reduce_add() as f64;
                acc.edge_art2 += a2.reduce_add() as f64;
                acc.edge_det += detail_lost.reduce_add() as f64;
                acc.edge_det4 += dl4.reduce_add() as f64;
                acc.edge_det2 += dl2.reduce_add() as f64;
                acc.edge_art8 += (a4 * a4).reduce_add() as f64;
                acc.edge_det8 += (dl4 * dl4).reduce_add() as f64;
                acc.edge_art_max = acc.edge_art_max.max(artifact.reduce_max());
                acc.edge_det_max = acc.edge_det_max.max(detail_lost.reduce_max());

                let vs = s - mu1;
                let vd = d - mu2;
                acc.hf_sq_src += (vs * vs).reduce_add() as f64;
                acc.hf_sq_dst += (vd * vd).reduce_add() as f64;
                acc.hf_abs_src += diff1.reduce_add() as f64;
                acc.hf_abs_dst += diff2.reduce_add() as f64;

                let pd = s - d;
                acc.mse += (pd * pd).reduce_add() as f64;
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array_t(v3, h_mu1[add_base..][..8].try_into().unwrap())
                - f32x8::from_array_t(v3, h_mu1[rem_base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array_t(v3, h_mu2[add_base..][..8].try_into().unwrap())
                - f32x8::from_array_t(v3, h_mu2[rem_base..][..8].try_into().unwrap());
        }
    }

    // Scalar remainder
    let inv = 1.0 / diam as f32;
    for x in (col_base_8 + remaining_8groups * 8)..width {
        let mut sum_m1 = 0.0f32;
        let mut sum_m2 = 0.0f32;

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            sum_m1 += h_mu1[idx * width + x];
            sum_m2 += h_mu2[idx * width + x];
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let mu1 = sum_m1 * inv;
                let mu2 = sum_m2 * inv;
                let sv = src[y * width + x];
                let dv = dst[y * width + x];

                if store_mu {
                    mu1_out[y * width + x] = mu1;
                    mu2_out[y * width + x] = mu2;
                }

                // Edge
                let diff1 = (sv - mu1).abs();
                let diff2 = (dv - mu2).abs();
                let ed = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;
                let artifact = ed.max(0.0f32);
                let detail_lost = (-ed).max(0.0f32);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                acc.edge_art += artifact as f64;
                acc.edge_art4 += a4 as f64;
                acc.edge_art2 += a2 as f64;
                acc.edge_det += detail_lost as f64;
                acc.edge_det4 += dl4 as f64;
                acc.edge_det2 += dl2 as f64;
                acc.edge_art8 += (a4 * a4) as f64;
                acc.edge_det8 += (dl4 * dl4) as f64;
                acc.edge_art_max = acc.edge_art_max.max(artifact);
                acc.edge_det_max = acc.edge_det_max.max(detail_lost);

                let vs = sv - mu1;
                let vd = dv - mu2;
                acc.hf_sq_src += (vs * vs) as f64;
                acc.hf_sq_dst += (vd * vd) as f64;
                acc.hf_abs_src += diff1 as f64;
                acc.hf_abs_dst += diff2 as f64;

                let pd = sv - dv;
                acc.mse += (pd * pd) as f64;
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            sum_m1 = sum_m1 + h_mu1[add_idx * width + x] - h_mu1[rem_idx * width + x];
            sum_m2 = sum_m2 + h_mu2[add_idx * width + x] - h_mu2[rem_idx * width + x];
        }
    }

    acc
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn fused_vblur_edge_inner_v3(
    token: archmage::X64V3Token,
    h_mu1: &[f32],
    h_mu2: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
) -> StripChannelAccum {
    let diam = 2 * radius + 1;
    let inv_v = f32x8::splat_t(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 8;

    let one = f32x8::splat_t(token, 1.0);
    let zero = f32x8::zero_t(token);

    let mut acc = StripChannelAccum::zero();
    let inner_end = inner_start + inner_h;

    for cg in 0..col_groups {
        let col_base = cg * 8;
        let mut sum_m1 = f32x8::zero_t(token);
        let mut sum_m2 = f32x8::zero_t(token);

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            let base = idx * width + col_base;
            sum_m1 = sum_m1 + f32x8::from_array_t(token, h_mu1[base..][..8].try_into().unwrap());
            sum_m2 = sum_m2 + f32x8::from_array_t(token, h_mu2[base..][..8].try_into().unwrap());
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;
                let mu1 = sum_m1 * inv_v;
                let mu2 = sum_m2 * inv_v;
                let s = f32x8::from_array_t(token, src[base..][..8].try_into().unwrap());
                let d = f32x8::from_array_t(token, dst[base..][..8].try_into().unwrap());

                if store_mu {
                    mu1_out[base..base + 8].copy_from_slice(&mu1.to_array());
                    mu2_out[base..base + 8].copy_from_slice(&mu2.to_array());
                }

                let diff1 = (s - mu1).abs();
                let diff2 = (d - mu2).abs();
                let ed = (one + diff2) / (one + diff1) - one;
                let artifact = ed.max(zero);
                let detail_lost = (-ed).max(zero);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                acc.edge_art += artifact.reduce_add() as f64;
                acc.edge_art4 += a4.reduce_add() as f64;
                acc.edge_art2 += a2.reduce_add() as f64;
                acc.edge_det += detail_lost.reduce_add() as f64;
                acc.edge_det4 += dl4.reduce_add() as f64;
                acc.edge_det2 += dl2.reduce_add() as f64;
                acc.edge_art8 += (a4 * a4).reduce_add() as f64;
                acc.edge_det8 += (dl4 * dl4).reduce_add() as f64;
                acc.edge_art_max = acc.edge_art_max.max(artifact.reduce_max());
                acc.edge_det_max = acc.edge_det_max.max(detail_lost.reduce_max());

                let vs = s - mu1;
                let vd = d - mu2;
                acc.hf_sq_src += (vs * vs).reduce_add() as f64;
                acc.hf_sq_dst += (vd * vd).reduce_add() as f64;
                acc.hf_abs_src += diff1.reduce_add() as f64;
                acc.hf_abs_dst += diff2.reduce_add() as f64;

                let pd = s - d;
                acc.mse += (pd * pd).reduce_add() as f64;
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            sum_m1 = sum_m1
                + f32x8::from_array_t(token, h_mu1[add_base..][..8].try_into().unwrap())
                - f32x8::from_array_t(token, h_mu1[rem_base..][..8].try_into().unwrap());
            sum_m2 = sum_m2
                + f32x8::from_array_t(token, h_mu2[add_base..][..8].try_into().unwrap())
                - f32x8::from_array_t(token, h_mu2[rem_base..][..8].try_into().unwrap());
        }
    }

    // Scalar remainder
    let inv = 1.0 / diam as f32;
    for x in (col_groups * 8)..width {
        let mut sum_m1 = 0.0f32;
        let mut sum_m2 = 0.0f32;

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            sum_m1 += h_mu1[idx * width + x];
            sum_m2 += h_mu2[idx * width + x];
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let mu1 = sum_m1 * inv;
                let mu2 = sum_m2 * inv;
                let sv = src[y * width + x];
                let dv = dst[y * width + x];

                if store_mu {
                    mu1_out[y * width + x] = mu1;
                    mu2_out[y * width + x] = mu2;
                }

                // Edge
                let diff1 = (sv - mu1).abs();
                let diff2 = (dv - mu2).abs();
                let ed = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;
                let artifact = ed.max(0.0f32);
                let detail_lost = (-ed).max(0.0f32);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                acc.edge_art += artifact as f64;
                acc.edge_art4 += a4 as f64;
                acc.edge_art2 += a2 as f64;
                acc.edge_det += detail_lost as f64;
                acc.edge_det4 += dl4 as f64;
                acc.edge_det2 += dl2 as f64;
                acc.edge_art8 += (a4 * a4) as f64;
                acc.edge_det8 += (dl4 * dl4) as f64;
                acc.edge_art_max = acc.edge_art_max.max(artifact);
                acc.edge_det_max = acc.edge_det_max.max(detail_lost);

                let vs = sv - mu1;
                let vd = dv - mu2;
                acc.hf_sq_src += (vs * vs) as f64;
                acc.hf_sq_dst += (vd * vd) as f64;
                acc.hf_abs_src += diff1 as f64;
                acc.hf_abs_dst += diff2 as f64;

                let pd = sv - dv;
                acc.mse += (pd * pd) as f64;
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            sum_m1 = sum_m1 + h_mu1[add_idx * width + x] - h_mu1[rem_idx * width + x];
            sum_m2 = sum_m2 + h_mu2[add_idx * width + x] - h_mu2[rem_idx * width + x];
        }
    }

    acc
}

#[magetypes(neon, wasm128, scalar)]
fn fused_vblur_edge_inner(
    token: Token,
    h_mu1: &[f32],
    h_mu2: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
) -> StripChannelAccum {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    let diam = 2 * radius + 1;
    let inv_v = f32x8::splat_t(token, 1.0 / diam as f32);
    let r = radius;
    let col_groups = width / 8;

    let one = f32x8::splat_t(token, 1.0);
    let zero = f32x8::zero_t(token);

    let mut acc = StripChannelAccum::zero();
    let inner_end = inner_start + inner_h;

    for cg in 0..col_groups {
        let col_base = cg * 8;
        let mut sum_m1_a = [0.0f32; 8];
        let mut sum_m2_a = [0.0f32; 8];

        // Initialize running sums
        {
            let mut sm1 = f32x8::zero_t(token);
            let mut sm2 = f32x8::zero_t(token);
            for i in 0..diam {
                let idx = mirror_idx(i, r, height);
                let base = idx * width + col_base;
                sm1 = sm1 + f32x8::from_array_t(token, h_mu1[base..][..8].try_into().unwrap());
                sm2 = sm2 + f32x8::from_array_t(token, h_mu2[base..][..8].try_into().unwrap());
            }
            sm1.store(&mut sum_m1_a);
            sm2.store(&mut sum_m2_a);
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let base = y * width + col_base;
                let sum_m1 = f32x8::from_array_t(token, sum_m1_a);
                let sum_m2 = f32x8::from_array_t(token, sum_m2_a);

                let mu1 = sum_m1 * inv_v;
                let mu2 = sum_m2 * inv_v;
                let s = f32x8::from_array_t(token, src[base..][..8].try_into().unwrap());
                let d = f32x8::from_array_t(token, dst[base..][..8].try_into().unwrap());

                if store_mu {
                    mu1.store((&mut mu1_out[base..base + 8]).try_into().unwrap());
                    mu2.store((&mut mu2_out[base..base + 8]).try_into().unwrap());
                }

                let diff1 = (s - mu1).abs();
                let diff2 = (d - mu2).abs();
                let ed = (one + diff2) / (one + diff1) - one;
                let artifact = ed.max(zero);
                let detail_lost = (-ed).max(zero);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                acc.edge_art += artifact.reduce_add() as f64;
                acc.edge_art4 += a4.reduce_add() as f64;
                acc.edge_art2 += a2.reduce_add() as f64;
                acc.edge_det += detail_lost.reduce_add() as f64;
                acc.edge_det4 += dl4.reduce_add() as f64;
                acc.edge_det2 += dl2.reduce_add() as f64;
                acc.edge_art8 += (a4 * a4).reduce_add() as f64;
                acc.edge_det8 += (dl4 * dl4).reduce_add() as f64;
                acc.edge_art_max = acc.edge_art_max.max(artifact.reduce_max());
                acc.edge_det_max = acc.edge_det_max.max(detail_lost.reduce_max());

                let vs = s - mu1;
                let vd = d - mu2;
                acc.hf_sq_src += (vs * vs).reduce_add() as f64;
                acc.hf_sq_dst += (vd * vd).reduce_add() as f64;
                acc.hf_abs_src += diff1.reduce_add() as f64;
                acc.hf_abs_dst += diff2.reduce_add() as f64;

                let pd = s - d;
                acc.mse += (pd * pd).reduce_add() as f64;
            }

            // Slide V-blur window
            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            let add_base = add_idx * width + col_base;
            let rem_base = rem_idx * width + col_base;
            let new_m1 = f32x8::from_array_t(token, sum_m1_a)
                + f32x8::from_array_t(token, h_mu1[add_base..][..8].try_into().unwrap())
                - f32x8::from_array_t(token, h_mu1[rem_base..][..8].try_into().unwrap());
            let new_m2 = f32x8::from_array_t(token, sum_m2_a)
                + f32x8::from_array_t(token, h_mu2[add_base..][..8].try_into().unwrap())
                - f32x8::from_array_t(token, h_mu2[rem_base..][..8].try_into().unwrap());
            new_m1.store(&mut sum_m1_a);
            new_m2.store(&mut sum_m2_a);
        }
    }

    // Scalar remainder
    let inv = 1.0 / diam as f32;
    for x in (col_groups * 8)..width {
        let mut sum_m1 = 0.0f32;
        let mut sum_m2 = 0.0f32;

        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            sum_m1 += h_mu1[idx * width + x];
            sum_m2 += h_mu2[idx * width + x];
        }

        for y in 0..height {
            if y >= inner_start && y < inner_end {
                let mu1 = sum_m1 * inv;
                let mu2 = sum_m2 * inv;
                let sv = src[y * width + x];
                let dv = dst[y * width + x];

                if store_mu {
                    mu1_out[y * width + x] = mu1;
                    mu2_out[y * width + x] = mu2;
                }

                // Edge
                let diff1 = (sv - mu1).abs();
                let diff2 = (dv - mu2).abs();
                let ed = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;
                let artifact = ed.max(0.0f32);
                let detail_lost = (-ed).max(0.0f32);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                acc.edge_art += artifact as f64;
                acc.edge_art4 += a4 as f64;
                acc.edge_art2 += a2 as f64;
                acc.edge_det += detail_lost as f64;
                acc.edge_det4 += dl4 as f64;
                acc.edge_det2 += dl2 as f64;
                acc.edge_art8 += (a4 * a4) as f64;
                acc.edge_det8 += (dl4 * dl4) as f64;
                acc.edge_art_max = acc.edge_art_max.max(artifact);
                acc.edge_det_max = acc.edge_det_max.max(detail_lost);

                let vs = sv - mu1;
                let vd = dv - mu2;
                acc.hf_sq_src += (vs * vs) as f64;
                acc.hf_sq_dst += (vd * vd) as f64;
                acc.hf_abs_src += diff1 as f64;
                acc.hf_abs_dst += diff2 as f64;

                let pd = sv - dv;
                acc.mse += (pd * pd) as f64;
            }

            let add_idx = vblur_add_idx(y, r, height);
            let rem_idx = vblur_rem_idx(y, r, height);
            sum_m1 = sum_m1 + h_mu1[add_idx * width + x] - h_mu1[rem_idx * width + x];
            sum_m2 = sum_m2 + h_mu2[add_idx * width + x] - h_mu2[rem_idx * width + x];
        }
    }

    acc
}

// ============================================================================
// FEATCANON canonical bodies
//
// `fused_vblur_ssim_canon` / `fused_vblur_edge_canon` are the canonical-form
// rewrites of the fused V-blur kernels: identical element arithmetic on every
// tier (inherent `f32` ops — `mul_add` fused, no `reduce_add`), and every
// pooled sum accumulated through `featcanon::Pool` (fixed 8-virtual-lane f32
// partials + `era2_reduce8`, f64 lanes, or Neumaier — the measurement
// candidates (d)/(e)/(f)). The V-blur itself is a per-column recurrence
// (`sum += add − rem`), already position-stable; the pools' per-row
// `reduce_add` over a tier-width chunk was the divergence, and is replaced by
// the row-wide canonical lanes (lane = column mod 8), finished once per row.
//
// `fused_vblur_ssim_exact` is the exact-oracle arm: identical loop shape with
// every element evaluated in f64 (fused `f64::mul_add`) and Neumaier-compensated
// accumulation; plane stores round once at the f32 boundary.
// ============================================================================

/// Per-row canonical pool set for the SSIM fused kernel.
#[derive(Clone, Copy)]
struct VblurPools<P: Copy> {
    ssim_d: P,
    ssim_d4: P,
    ssim_d2: P,
    ssim_d8: P,
    edge_art: P,
    edge_art4: P,
    edge_art2: P,
    edge_art8: P,
    edge_det: P,
    edge_det4: P,
    edge_det2: P,
    edge_det8: P,
    hf_sq_src: P,
    hf_sq_dst: P,
    hf_abs_src: P,
    hf_abs_dst: P,
    mse: P,
    act_sum: P,
    masked_ssim_d: P,
    masked_ssim_d4: P,
    masked_ssim_d2: P,
    masked_art4: P,
    masked_det4: P,
    masked_mse: P,
    iw_ssim_d: P,
    iw_ssim_d4: P,
    iw_ssim_d2: P,
    iw_art4: P,
    iw_det4: P,
    iw_mse: P,
}

impl<P: crate::featcanon::Pool> VblurPools<P> {
    fn zero() -> Self {
        Self {
            ssim_d: P::zero(),
            ssim_d4: P::zero(),
            ssim_d2: P::zero(),
            ssim_d8: P::zero(),
            edge_art: P::zero(),
            edge_art4: P::zero(),
            edge_art2: P::zero(),
            edge_art8: P::zero(),
            edge_det: P::zero(),
            edge_det4: P::zero(),
            edge_det2: P::zero(),
            edge_det8: P::zero(),
            hf_sq_src: P::zero(),
            hf_sq_dst: P::zero(),
            hf_abs_src: P::zero(),
            hf_abs_dst: P::zero(),
            mse: P::zero(),
            act_sum: P::zero(),
            masked_ssim_d: P::zero(),
            masked_ssim_d4: P::zero(),
            masked_ssim_d2: P::zero(),
            masked_art4: P::zero(),
            masked_det4: P::zero(),
            masked_mse: P::zero(),
            iw_ssim_d: P::zero(),
            iw_ssim_d4: P::zero(),
            iw_ssim_d2: P::zero(),
            iw_art4: P::zero(),
            iw_det4: P::zero(),
            iw_mse: P::zero(),
        }
    }
    /// Add each pool's finished sum to `acc`, in the fixed order every body
    /// uses: once per row, or at revision 7 once per call.
    fn finish_into(&self, acc: &mut StripChannelAccum, free: FreeExtrasWork, ext: ExtPoolsWork) {
        acc.ssim_d += self.ssim_d.fin();
        acc.ssim_d4 += self.ssim_d4.fin();
        acc.ssim_d2 += self.ssim_d2.fin();
        acc.edge_art += self.edge_art.fin();
        acc.edge_art4 += self.edge_art4.fin();
        acc.edge_art2 += self.edge_art2.fin();
        acc.edge_det += self.edge_det.fin();
        acc.edge_det4 += self.edge_det4.fin();
        acc.edge_det2 += self.edge_det2.fin();
        acc.mse += self.mse.fin();
        if !free.local_only {
            acc.ssim_d8 += self.ssim_d8.fin();
            acc.edge_art8 += self.edge_art8.fin();
            acc.edge_det8 += self.edge_det8.fin();
            acc.hf_sq_src += self.hf_sq_src.fin();
            acc.hf_sq_dst += self.hf_sq_dst.fin();
            acc.hf_abs_src += self.hf_abs_src.fin();
            acc.hf_abs_dst += self.hf_abs_dst.fin();
        }
        if ext.on {
            acc.act_sum += self.act_sum.fin();
            if ext.mask {
                acc.masked_ssim_d += self.masked_ssim_d.fin();
                acc.masked_ssim_d4 += self.masked_ssim_d4.fin();
                acc.masked_ssim_d2 += self.masked_ssim_d2.fin();
                acc.masked_art4 += self.masked_art4.fin();
                acc.masked_det4 += self.masked_det4.fin();
                acc.masked_mse += self.masked_mse.fin();
            }
            if ext.iw {
                acc.iw_ssim_d += self.iw_ssim_d.fin();
                acc.iw_ssim_d4 += self.iw_ssim_d4.fin();
                acc.iw_ssim_d2 += self.iw_ssim_d2.fin();
                acc.iw_art4 += self.iw_art4.fin();
                acc.iw_det4 += self.iw_det4.fin();
                acc.iw_mse += self.iw_mse.fin();
            }
        }
    }
}

/// The band-lifetime accumulators (raw moments + bounded error + luma bins):
/// production vector-adds into per-column-group lanes and reduces once at the
/// band's last inner row; the canonical shape keeps that one-reduce-per-band
/// lifetime over canonical lanes.
#[derive(Clone, Copy)]
struct BandPools<P: Copy> {
    fm_s: P,
    fm_d: P,
    fm_s2: P,
    fm_d2: P,
    fm_dd: P,
    fm_ds: P,
    be_m: P,
    wd_num: P,
    wd_den: P,
    wb_num: P,
    wb_den: P,
}

impl VblurPools<crate::featcanon::Lanes16F32> {
    /// Revision 7: fold the f32 row lanes into `row` and clear them
    /// (`crate::featcanon::Lanes16F32::fold_into`). The `hf_*_dst` lanes hold
    /// the differences from the source (`(vd − vs)(vd + vs)`, `|vd| − |vs|`),
    /// so each source lane is added to both its own sum and the distorted
    /// sum, then the difference lane to the distorted sum: the ratio features
    /// then see the small difference summed directly, not recovered by
    /// subtracting two large sums.
    ///
    /// Only the pools the element step writes under `free` and `ext` are
    /// folded (the others' lanes stay zero, and so do their sums).
    fn fold_into<P: crate::featcanon::Pool>(
        &mut self,
        row: &mut VblurPools<P>,
        free: FreeExtrasWork,
        ext: ExtPoolsWork,
    ) {
        if !free.local_only {
            for (src, dst, rsrc, rdst) in [
                (
                    &mut self.hf_sq_src,
                    &mut self.hf_sq_dst,
                    &mut row.hf_sq_src,
                    &mut row.hf_sq_dst,
                ),
                (
                    &mut self.hf_abs_src,
                    &mut self.hf_abs_dst,
                    &mut row.hf_abs_src,
                    &mut row.hf_abs_dst,
                ),
            ] {
                for l in 0..16 {
                    rsrc.add(l, src.0[l]);
                    rdst.add(l, src.0[l]);
                    rdst.add(l, dst.0[l]);
                }
                *src = crate::featcanon::Lanes16F32::default();
                *dst = crate::featcanon::Lanes16F32::default();
            }
        }
        self.ssim_d.fold_into(&mut row.ssim_d);
        self.ssim_d4.fold_into(&mut row.ssim_d4);
        self.ssim_d2.fold_into(&mut row.ssim_d2);
        if !free.local_only && !free.omit_peaks {
            self.ssim_d8.fold_into(&mut row.ssim_d8);
        }
        if !free.omit_edges {
            self.edge_art.fold_into(&mut row.edge_art);
            self.edge_art4.fold_into(&mut row.edge_art4);
            self.edge_art2.fold_into(&mut row.edge_art2);
            self.edge_det.fold_into(&mut row.edge_det);
            self.edge_det4.fold_into(&mut row.edge_det4);
            self.edge_det2.fold_into(&mut row.edge_det2);
            if !free.omit_peaks {
                self.edge_art8.fold_into(&mut row.edge_art8);
                self.edge_det8.fold_into(&mut row.edge_det8);
            }
        }
        self.mse.fold_into(&mut row.mse);
        if ext.on {
            self.act_sum.fold_into(&mut row.act_sum);
            if ext.mask {
                self.masked_ssim_d.fold_into(&mut row.masked_ssim_d);
                self.masked_ssim_d4.fold_into(&mut row.masked_ssim_d4);
                self.masked_ssim_d2.fold_into(&mut row.masked_ssim_d2);
                self.masked_art4.fold_into(&mut row.masked_art4);
                self.masked_det4.fold_into(&mut row.masked_det4);
                self.masked_mse.fold_into(&mut row.masked_mse);
            }
            if ext.iw {
                self.iw_ssim_d.fold_into(&mut row.iw_ssim_d);
                self.iw_ssim_d4.fold_into(&mut row.iw_ssim_d4);
                self.iw_ssim_d2.fold_into(&mut row.iw_ssim_d2);
                self.iw_art4.fold_into(&mut row.iw_art4);
                self.iw_det4.fold_into(&mut row.iw_det4);
                self.iw_mse.fold_into(&mut row.iw_mse);
            }
        }
    }
}

impl<P: crate::featcanon::Pool> BandPools<P> {
    fn zero() -> Self {
        Self {
            fm_s: P::zero(),
            fm_d: P::zero(),
            fm_s2: P::zero(),
            fm_d2: P::zero(),
            fm_dd: P::zero(),
            fm_ds: P::zero(),
            be_m: P::zero(),
            wd_num: P::zero(),
            wd_den: P::zero(),
            wb_num: P::zero(),
            wb_den: P::zero(),
        }
    }
}

// ============================================================
// featacc blur axis: the V-blur window state behind the fused
// canon/exact bodies. `Rec` is the shipped per-column f32 sliding
// sums (bit-identical to production — a product build has only
// this variant and `blur_axis()` is a constant `Rec` there).
// `Rec64` runs the same recurrence in f64; `Fresh` keeps no state
// at all — `at*` re-sums the window per position in f64.
// ============================================================

/// The blurred planes a V window reads, plus its geometry.
struct VWinPlanes<'a> {
    m1: &'a [f32],
    m2: &'a [f32],
    sq: &'a [f32],
    s12: &'a [f32],
    act: &'a [f32],
    width: usize,
    height: usize,
    r: usize,
}

impl VWinPlanes<'_> {
    #[inline(always)]
    fn diam(&self) -> usize {
        2 * self.r + 1
    }
    #[inline(always)]
    fn inv32(&self) -> f32 {
        1.0 / self.diam() as f32
    }
    #[inline(always)]
    fn inv64(&self) -> f64 {
        1.0 / self.diam() as f64
    }
    /// `fresh` re-sum of `plane`'s column `x`, window centred at row `y`.
    /// Replays the sliding kernels' boundary map exactly (`tap_mirror`).
    #[cfg(feature = "oracle")]
    fn fresh(&self, plane: &[f32], x: usize, y: usize) -> f64 {
        if plane.is_empty() {
            return 0.0;
        }
        let mut s = 0.0f64;
        for k in -(self.r as isize)..=(self.r as isize) {
            s += plane[crate::featcanon::tap_mirror(y as isize + k, self.height) * self.width + x]
                as f64;
        }
        s * self.inv64()
    }
}

enum VWin {
    /// Production: per-column f32 sliding sums.
    Rec {
        m1: Vec<f32>,
        m2: Vec<f32>,
        sq: Vec<f32>,
        s12: Vec<f32>,
        act: Vec<f32>,
    },
    /// The same sliding recurrence in f64 — THE REV4 CANON's window state
    /// (rev4canon: `BlurMode::Rec64`).
    F64 {
        m1: Vec<f64>,
        m2: Vec<f64>,
        sq: Vec<f64>,
        s12: Vec<f64>,
        act: Vec<f64>,
    },
    /// No state — per-position f32 `localwin` pair tree — THE REV5 CANON's
    /// window (`BlurMode::Local`). Also the measurement arm for an explicit
    /// `ZENSIM_FEATCANON_BLUR=local`.
    Local,
    /// No state — per-position f64 re-summation.
    #[cfg(feature = "oracle")]
    Fresh,
}

impl VWin {
    /// Seed the initial window (row `y = 0`), `mirror_idx`-identical to the
    /// production init loop.
    fn new(blur: crate::featcanon::BlurMode, p: &VWinPlanes<'_>) -> Self {
        let seed32 = |plane: &[f32], out: &mut Vec<f32>| {
            if plane.is_empty() {
                return;
            }
            for i in 0..p.diam() {
                let idx = mirror_idx(i, p.r, p.height);
                let b = idx * p.width;
                for x in 0..p.width {
                    out[x] += plane[b + x];
                }
            }
        };
        match blur {
            crate::featcanon::BlurMode::Rec => {
                let mut m1 = vec![0.0; p.width];
                let mut m2 = vec![0.0; p.width];
                let mut sq = vec![0.0; p.width];
                let mut s12 = vec![0.0; p.width];
                let mut act = if p.act.is_empty() {
                    Vec::new()
                } else {
                    vec![0.0; p.width]
                };
                seed32(p.m1, &mut m1);
                seed32(p.m2, &mut m2);
                seed32(p.sq, &mut sq);
                seed32(p.s12, &mut s12);
                seed32(p.act, &mut act);
                Self::Rec {
                    m1,
                    m2,
                    sq,
                    s12,
                    act,
                }
            }
            crate::featcanon::BlurMode::Rec64 => {
                let seed64 = |plane: &[f32], out: &mut Vec<f64>| {
                    if plane.is_empty() {
                        return;
                    }
                    for i in 0..p.diam() {
                        let idx = mirror_idx(i, p.r, p.height);
                        let b = idx * p.width;
                        for x in 0..p.width {
                            out[x] += plane[b + x] as f64;
                        }
                    }
                };
                let mut m1 = vec![0.0; p.width];
                let mut m2 = vec![0.0; p.width];
                let mut sq = vec![0.0; p.width];
                let mut s12 = vec![0.0; p.width];
                let mut act = if p.act.is_empty() {
                    Vec::new()
                } else {
                    vec![0.0; p.width]
                };
                seed64(p.m1, &mut m1);
                seed64(p.m2, &mut m2);
                seed64(p.sq, &mut sq);
                seed64(p.s12, &mut s12);
                seed64(p.act, &mut act);
                Self::F64 {
                    m1,
                    m2,
                    sq,
                    s12,
                    act,
                }
            }
            crate::featcanon::BlurMode::Local => Self::Local,
            #[cfg(feature = "oracle")]
            crate::featcanon::BlurMode::Fresh => Self::Fresh,
        }
    }

    /// Advance the window from row `y` to `y + 1` (`vblur_add_idx` /
    /// `vblur_rem_idx` — the production index sequence).
    fn slide(&mut self, y: usize, p: &VWinPlanes<'_>) {
        let slide32 = |plane: &[f32], out: &mut [f32], ab: usize, rb: usize| {
            if plane.is_empty() {
                return;
            }
            for x in 0..p.width {
                out[x] = out[x] + plane[ab + x] - plane[rb + x];
            }
        };
        match self {
            Self::Rec {
                m1,
                m2,
                sq,
                s12,
                act,
            } => {
                let ab = vblur_add_idx(y, p.r, p.height) * p.width;
                let rb = vblur_rem_idx(y, p.r, p.height) * p.width;
                slide32(p.m1, m1, ab, rb);
                slide32(p.m2, m2, ab, rb);
                slide32(p.sq, sq, ab, rb);
                slide32(p.s12, s12, ab, rb);
                slide32(p.act, act, ab, rb);
            }
            Self::F64 {
                m1,
                m2,
                sq,
                s12,
                act,
            } => {
                let ab = vblur_add_idx(y, p.r, p.height) * p.width;
                let rb = vblur_rem_idx(y, p.r, p.height) * p.width;
                // Same empty-plane guard as `slide32` above: the edge
                // body's planes have no `sq`/`s12`/`act`, so absent planes
                // are skipped rather than indexed (their state vectors are
                // empty too — `sq`/`s12` stay `len == 0`).
                let slide64 = |plane: &[f32], out: &mut [f64]| {
                    if plane.is_empty() {
                        return;
                    }
                    for x in 0..p.width {
                        out[x] = out[x] + plane[ab + x] as f64 - plane[rb + x] as f64;
                    }
                };
                slide64(p.m1, m1);
                slide64(p.m2, m2);
                slide64(p.sq, sq);
                slide64(p.s12, s12);
                slide64(p.act, act);
            }
            Self::Local => {}
            #[cfg(feature = "oracle")]
            Self::Fresh => {}
        }
    }

    /// The `(mu1, mu2, ssq, s12, act)` plane values at `(x, y)` — f32, the
    /// canon bodies' element precision. `Rec` is the production `sum * inv`
    /// f32 multiply; `F64`/`Fresh` compute in f64 and narrow once at the
    /// same f32-store boundary production writes through.
    #[cfg_attr(not(feature = "oracle"), allow(unused_variables))]
    #[inline(always)]
    fn at32(&self, x: usize, y: usize, p: &VWinPlanes<'_>) -> (f32, f32, f32, f32, f32) {
        match self {
            Self::Rec {
                m1,
                m2,
                sq,
                s12,
                act,
            } => {
                let inv = p.inv32();
                let a = if act.is_empty() { 0.0 } else { act[x] * inv };
                (m1[x] * inv, m2[x] * inv, sq[x] * inv, s12[x] * inv, a)
            }
            Self::F64 {
                m1,
                m2,
                sq,
                s12,
                act,
            } => {
                let inv = p.inv64();
                // Absent planes read as `0.0` — the `act` convention,
                // extended to `sq`/`s12` (the edge body's planes leave all
                // three unpopulated).
                let g = |v: &[f64]| {
                    if v.is_empty() {
                        0.0
                    } else {
                        (v[x] * inv) as f32
                    }
                };
                (
                    (m1[x] * inv) as f32,
                    (m2[x] * inv) as f32,
                    g(sq),
                    g(s12),
                    g(act),
                )
            }
            Self::Local => {
                // Rev5 `localwin`: the row's own 11-tap f32 pair tree — no
                // recurrence, nothing carried between rows.
                let inv = p.inv32();
                let diam = p.diam();
                let g = |plane: &[f32]| -> f32 {
                    if plane.is_empty() {
                        return 0.0;
                    }
                    let mut t = [0.0f32; 33];
                    debug_assert!(diam <= 33);
                    for (k, t) in t.iter_mut().enumerate().take(diam) {
                        *t = plane[crate::featcanon::tap_mirror(
                            y as isize + k as isize - p.r as isize,
                            p.height,
                        ) * p.width
                            + x];
                    }
                    crate::blur::local_sum_taps(&t[..diam]) * inv
                };
                (g(p.m1), g(p.m2), g(p.sq), g(p.s12), g(p.act))
            }
            #[cfg(feature = "oracle")]
            Self::Fresh => (
                p.fresh(p.m1, x, y) as f32,
                p.fresh(p.m2, x, y) as f32,
                p.fresh(p.sq, x, y) as f32,
                p.fresh(p.s12, x, y) as f32,
                p.fresh(p.act, x, y) as f32,
            ),
        }
    }

    /// The `F64` variant's five per-column f64 window states — `None` on the
    /// other variants. The canon64 vector bodies are entered only under
    /// `BlurMode::Rec64` (the caller's axis gate), which constructs `F64`.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))] // only the x86_64 lane-parallel canon bodies call it
    #[inline(always)]
    fn f64_states(&self) -> Option<[&[f64]; 5]> {
        match self {
            Self::F64 {
                m1,
                m2,
                sq,
                s12,
                act,
            } => Some([m1, m2, sq, s12, act]),
            _ => None,
        }
    }
} // impl VWin

/// [`VWin::slide`]'s `F64` arm with each column's `state + add − rem`
/// evaluated eight columns at a time through `f64x8` — per-column op
/// sequence identical (the recurrence is per-column; SIMD lanes never
/// mix), `width % 8` tail scalar. Other variants fall through to `slide`;
/// the canon64 vector bodies invoke this only on `F64`.
///
/// This is a macro, not a method: the `f64x8` ops must compile INSIDE each
/// caller's `#[target_feature]` region — a trait-generic helper left
/// out-of-line compiles each op into a call to a `core::arch` shim that
/// cannot inline back (the measured dominant single-kernel cost of the Rev4
/// bake; see the note above `raw_moments_accumulate16`). `#[inline(always)]`
/// on the equivalent method was still emitted as an out-of-line call, so the
/// body expands at each call site instead. `f64x8` resolves to each
/// `#[magetypes]` caller's per-tier alias. The `chunks_exact` iterators keep
/// every load/store bounds-check-free without unsafe.
#[cfg(target_arch = "x86_64")] // only the x86_64 canon64 bodies invoke it
macro_rules! vwin_slide64x8 {
    ($win:expr, $token:expr, $y:expr, $p:expr) => {{
        match &mut $win {
            VWin::F64 {
                m1,
                m2,
                sq,
                s12,
                act,
            } => {
                let ab = vblur_add_idx($y, $p.r, $p.height) * $p.width;
                let rb = vblur_rem_idx($y, $p.r, $p.height) * $p.width;
                // One slice construction per row per plane; the chunk/tail
                // iterators then carry the bounds so no index is re-checked.
                for (out, plane) in [
                    (m1, $p.m1),
                    (m2, $p.m2),
                    (sq, $p.sq),
                    (s12, $p.s12),
                    (act, $p.act),
                ] {
                    if plane.is_empty() {
                        continue;
                    }
                    let add = &plane[ab..ab + $p.width];
                    let rem = &plane[rb..rb + $p.width];
                    let mut oc = out[..$p.width].chunks_exact_mut(8);
                    let mut ac = add.chunks_exact(8);
                    let mut rc = rem.chunks_exact(8);
                    for (o, (a8, r8)) in (&mut oc).zip((&mut ac).zip(&mut rc)) {
                        let a8: &[f32; 8] = a8.try_into().unwrap();
                        let r8: &[f32; 8] = r8.try_into().unwrap();
                        let cur = f64x8::load_t($token, (&*o).try_into().unwrap());
                        let a: [f64; 8] = std::array::from_fn(|i| a8[i] as f64);
                        let r: [f64; 8] = std::array::from_fn(|i| r8[i] as f64);
                        (cur + f64x8::from_array_t($token, a) - f64x8::from_array_t($token, r))
                            .store(o.try_into().unwrap());
                    }
                    for ((o, &a), &r) in oc
                        .into_remainder()
                        .iter_mut()
                        .zip(ac.remainder())
                        .zip(rc.remainder())
                    {
                        *o = *o + a as f64 - r as f64;
                    }
                }
            }
            _ => $win.slide($y, $p),
        }
    }};
}

impl VWin {
    /// f64 sibling of [`VWin::at32`] for the exact bodies. `Rec` widens the
    /// f32 production product (the plane value IS the f32 store); `F64` and
    /// `Fresh` keep the unrounded f64.
    #[cfg(feature = "oracle")]
    #[inline(always)]
    fn at64(&self, x: usize, y: usize, p: &VWinPlanes<'_>) -> (f64, f64, f64, f64, f64) {
        match self {
            Self::Rec {
                m1,
                m2,
                sq,
                s12,
                act,
            } => {
                let inv = p.inv32();
                let a = if act.is_empty() { 0.0 } else { act[x] * inv };
                (
                    (m1[x] * inv) as f64,
                    (m2[x] * inv) as f64,
                    (sq[x] * inv) as f64,
                    (s12[x] * inv) as f64,
                    a as f64,
                )
            }
            #[cfg(feature = "oracle")]
            Self::F64 {
                m1,
                m2,
                sq,
                s12,
                act,
            } => {
                let inv = p.inv64();
                let a = if act.is_empty() { 0.0 } else { act[x] * inv };
                (m1[x] * inv, m2[x] * inv, sq[x] * inv, s12[x] * inv, a)
            }
            Self::Local => {
                // The f32 pair-tree window value, widened (it IS the f32 store).
                let inv = p.inv32();
                let diam = p.diam();
                let g = |plane: &[f32]| -> f64 {
                    if plane.is_empty() {
                        return 0.0;
                    }
                    let mut t = [0.0f32; 33];
                    for (k, t) in t.iter_mut().enumerate().take(diam) {
                        *t = plane[crate::featcanon::tap_mirror(
                            y as isize + k as isize - p.r as isize,
                            p.height,
                        ) * p.width
                            + x];
                    }
                    (crate::blur::local_sum_taps(&t[..diam]) * inv) as f64
                };
                (g(p.m1), g(p.m2), g(p.sq), g(p.s12), g(p.act))
            }
            Self::Fresh => (
                p.fresh(p.m1, x, y),
                p.fresh(p.m2, x, y),
                p.fresh(p.sq, x, y),
                p.fresh(p.s12, x, y),
                p.fresh(p.act, x, y),
            ),
        }
    }
}

/// One column's contribution to the V-blur row/band pools — the canonical
/// per-element sequence shared by the `Pool`-generic scalar body
/// ([`fused_vblur_ssim_canon`], every `x`) and the chunked canon64 vector
/// body ([`fused_vblur_ssim_canon64_vec`]'s `width % 8` tail), so the tail
/// IS the canonical element step by construction.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn vblur_ssim_elem_canon<P: crate::featcanon::Pool>(
    mu1: f32,
    mu2: f32,
    ssq: f32,
    s12: f32,
    act: f32,
    s: f32,
    d: f32,
    x: usize,
    base: usize,
    form: crate::ssim_form::SsimLumaForm,
    direct: bool,
    // Revision 7: the v1 edge ratio as `(x − y)/(1 + y)`.
    stable7: bool,
    free: FreeExtrasWork,
    ext: ExtPoolsWork,
    store_mu: bool,
    store_sd: bool,
    store_sigma: bool,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    sd_out: &mut [f32],
    ssq_out: &mut [f32],
    s12_out: &mut [f32],
    row: &mut VblurPools<P>,
    // Revision 7: where the f32-lane pools add (`None`: `row`, f64).
    mut row32: Option<&mut VblurPools<crate::featcanon::Lanes16F32>>,
    band: &mut BandPools<P>,
    acc: &mut StripChannelAccum,
) {
    let lane = x & (P::LANES - 1);
    macro_rules! rowadd {
        ($f:ident, $v:expr) => {{
            let v = $v;
            match row32.as_deref_mut() {
                Some(r) => crate::featcanon::Pool::add(&mut r.$f, lane, v),
                None => row.$f.add(lane, v),
            }
        }};
    }
    let sd = if direct {
        ssim_direct_raw_scalar(form, mu1, mu2, ssq, s12).max(0.0f32)
    } else {
        ssim_dissim_raw_scalar(form, mu1, mu2, ssq, s12).max(0.0f32)
    };
    let sd2 = sd * sd;
    let sd4 = sd2 * sd2;
    rowadd!(ssim_d, sd);
    rowadd!(ssim_d4, sd4);
    rowadd!(ssim_d2, sd2);
    if !free.local_only && !free.omit_peaks {
        rowadd!(ssim_d8, sd4 * sd4);
        acc.ssim_max = acc.ssim_max.max(sd);
    }
    if store_sd {
        sd_out[base + x] = sd;
    }
    if store_mu {
        mu1_out[base + x] = mu1;
        mu2_out[base + x] = mu2;
    }
    if store_sigma {
        ssq_out[base + x] = ssq;
        s12_out[base + x] = s12;
    }

    // Edge
    let diff1 = (s - mu1).abs();
    let diff2 = (d - mu2).abs();
    let ed = if stable7 {
        (diff2 - diff1) / (1.0f32 + diff1)
    } else {
        (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32
    };
    let artifact = ed.max(0.0f32);
    let detail_lost = (-ed).max(0.0f32);
    let a2 = artifact * artifact;
    let dl2 = detail_lost * detail_lost;
    let a4 = a2 * a2;
    let dl4 = dl2 * dl2;
    if !free.omit_edges {
        rowadd!(edge_art, artifact);
        rowadd!(edge_art4, a4);
        rowadd!(edge_art2, a2);
        rowadd!(edge_det, detail_lost);
        rowadd!(edge_det4, dl4);
        rowadd!(edge_det2, dl2);
        if !free.omit_peaks {
            rowadd!(edge_art8, a4 * a4);
            rowadd!(edge_det8, dl4 * dl4);
            acc.edge_art_max = acc.edge_art_max.max(artifact);
            acc.edge_det_max = acc.edge_det_max.max(detail_lost);
        }
    }

    // Variance / texture
    let vs = s - mu1;
    let vd = d - mu2;
    if !free.local_only {
        match row32.as_deref_mut() {
            // Revision 7: the source sums and the differences
            // `(vd − vs)(vd + vs)` and `|vd| − |vs|` (exact subtractions
            // when the two are within a factor of 2) in the f32 lanes; the
            // fold adds both to the distorted sums (`VblurPools::fold_into`).
            Some(r) => {
                crate::featcanon::Pool::add(&mut r.hf_sq_src, lane, vs * vs);
                crate::featcanon::Pool::add(&mut r.hf_sq_dst, lane, (vd - vs) * (vd + vs));
                crate::featcanon::Pool::add(&mut r.hf_abs_src, lane, diff1);
                crate::featcanon::Pool::add(&mut r.hf_abs_dst, lane, diff2 - diff1);
            }
            None => {
                row.hf_sq_src.add(lane, vs * vs);
                row.hf_sq_dst.add(lane, vd * vd);
                row.hf_abs_src.add(lane, diff1);
                row.hf_abs_dst.add(lane, diff2);
            }
        }
    }

    // MSE + ext pools
    let pd = s - d;
    let d2 = pd * pd;
    rowadd!(mse, d2);
    if ext.on {
        rowadd!(act_sum, act);
        if ext.mask {
            let w = 1.0f32 / ext.k_mask.mul_add(act, 1.0f32);
            let da = (sd * w).max(0.0);
            let d2a = da * da;
            rowadd!(masked_ssim_d, da);
            rowadd!(masked_ssim_d2, d2a);
            rowadd!(masked_ssim_d4, d2a * d2a);
            let e = ed * w;
            let a2 = e.max(0.0) * e.max(0.0);
            let dl2 = (-e).max(0.0) * (-e).max(0.0);
            rowadd!(masked_art4, a2 * a2);
            rowadd!(masked_det4, dl2 * dl2);
            rowadd!(masked_mse, d2 * w);
        }
        if ext.iw {
            let w = ext.k_iw.mul_add(act, 1.0f32);
            let db = (sd * w).max(0.0);
            let d2b = db * db;
            rowadd!(iw_ssim_d, db);
            rowadd!(iw_ssim_d2, d2b);
            rowadd!(iw_ssim_d4, d2b * d2b);
            let e = ed * w;
            let a2 = e.max(0.0) * e.max(0.0);
            let dl2 = (-e).max(0.0) * (-e).max(0.0);
            rowadd!(iw_art4, a2 * a2);
            rowadd!(iw_det4, dl2 * dl2);
            rowadd!(iw_mse, d2 * w);
        }
    }

    // Free raw moments — band-lifetime canonical lanes.
    if free.raw_moments {
        band.fm_s.add(lane, s);
        band.fm_d.add(lane, d);
        band.fm_s2.add(lane, s * s);
        band.fm_d2.add(lane, d * d);
        let df = d - s;
        band.fm_dd.add(lane, df * (d + s));
        band.fm_ds.add(lane, df);
    }
    // Free bounded error + luminance bins.
    if free.bounded_err {
        let sqm = (pd * pd).max(0.0);
        let be_i = sqm / (sqm + C_MSE_F32);
        band.be_m.add(lane, be_i);
        if free.lum_bins {
            let ry = s.max(0.0);
            let t = ry / (ry + C_LUM_T_F32);
            let one_mt = 1.0 - t;
            let wd = one_mt * one_mt;
            let wb = t * t;
            band.wd_num.add(lane, wd * be_i);
            band.wd_den.add(lane, wd);
            band.wb_num.add(lane, wb * be_i);
            band.wb_den.add(lane, wb);
        }
    }
}

/// Canonical (candidate-arithmetic) fused V-blur + SSIM/edge/HF/MSE feature
/// accumulation. `P` selects the brief's accumulation candidate; the window
/// state follows the featacc [`crate::featcanon::blur_axis`].
#[allow(clippy::too_many_arguments)]
fn fused_vblur_ssim_canon<P: crate::featcanon::Pool>(
    h_mu1: &[f32],
    h_mu2: &[f32],
    h_sigma_sq: &[f32],
    h_sigma12: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    sd_out: &mut [f32],
    store_sd: bool,
    ssq_out: &mut [f32],
    s12_out: &mut [f32],
    store_sigma: bool,
    free: FreeExtrasWork,
    direct: bool,
    ext: ExtPoolsWork,
    h_act: &[f32],
    // The computation's canonical mode — selects the blur window's own
    // arithmetic (`canon_blur_axis`): `Rec64` under the c64 canon.
    mode: crate::featcanon::Mode,
) -> StripChannelAccum {
    let form = free.luma_form();
    // Revision 7: the v1 edge ratio without cancellation.
    let stable7 = free.revision().stable_forms();
    let r = radius;
    let inner_end = inner_start + inner_h;

    // rev4canon: the mode's OWN blur axis — `Rec64` under the c64 canon,
    // `Local` under this computation's Rev5 era, `Rec` only for the
    // superseded `c32` oracle reproduction.
    let planes = VWinPlanes {
        m1: h_mu1,
        m2: h_mu2,
        sq: h_sigma_sq,
        s12: h_sigma12,
        act: if ext.on { h_act } else { &[] },
        width,
        height,
        r,
    };
    let mut win = VWin::new(
        crate::featcanon::canon_blur_axis_for(mode, free.revision()),
        &planes,
    );

    let mut acc = StripChannelAccum::zero();
    let mut band = BandPools::<P>::zero();

    for y in 0..height {
        if y >= inner_start && y < inner_end {
            let base = y * width;
            let mut row = VblurPools::<P>::zero();
            for x in 0..width {
                let (mu1, mu2, ssq, s12, act) = win.at32(x, y, &planes);
                let s = src[base + x];
                let d = dst[base + x];
                vblur_ssim_elem_canon(
                    mu1,
                    mu2,
                    ssq,
                    s12,
                    act,
                    s,
                    d,
                    x,
                    base,
                    form,
                    direct,
                    stable7,
                    free,
                    ext,
                    store_mu,
                    store_sd,
                    store_sigma,
                    mu1_out,
                    mu2_out,
                    sd_out,
                    ssq_out,
                    s12_out,
                    &mut row,
                    None,
                    &mut band,
                    &mut acc,
                );
            }
            // One fixed-order finish per row per pool.
            acc.ssim_d += row.ssim_d.fin();
            acc.ssim_d4 += row.ssim_d4.fin();
            acc.ssim_d2 += row.ssim_d2.fin();
            acc.edge_art += row.edge_art.fin();
            acc.edge_art4 += row.edge_art4.fin();
            acc.edge_art2 += row.edge_art2.fin();
            acc.edge_det += row.edge_det.fin();
            acc.edge_det4 += row.edge_det4.fin();
            acc.edge_det2 += row.edge_det2.fin();
            acc.mse += row.mse.fin();
            if !free.local_only {
                acc.ssim_d8 += row.ssim_d8.fin();
                acc.edge_art8 += row.edge_art8.fin();
                acc.edge_det8 += row.edge_det8.fin();
                acc.hf_sq_src += row.hf_sq_src.fin();
                acc.hf_sq_dst += row.hf_sq_dst.fin();
                acc.hf_abs_src += row.hf_abs_src.fin();
                acc.hf_abs_dst += row.hf_abs_dst.fin();
            }
            if ext.on {
                acc.act_sum += row.act_sum.fin();
                if ext.mask {
                    acc.masked_ssim_d += row.masked_ssim_d.fin();
                    acc.masked_ssim_d4 += row.masked_ssim_d4.fin();
                    acc.masked_ssim_d2 += row.masked_ssim_d2.fin();
                    acc.masked_art4 += row.masked_art4.fin();
                    acc.masked_det4 += row.masked_det4.fin();
                    acc.masked_mse += row.masked_mse.fin();
                }
                if ext.iw {
                    acc.iw_ssim_d += row.iw_ssim_d.fin();
                    acc.iw_ssim_d4 += row.iw_ssim_d4.fin();
                    acc.iw_ssim_d2 += row.iw_ssim_d2.fin();
                    acc.iw_art4 += row.iw_art4.fin();
                    acc.iw_det4 += row.iw_det4.fin();
                    acc.iw_mse += row.iw_mse.fin();
                }
            }
            if y + 1 == inner_end {
                if free.raw_moments {
                    acc.sum_s += band.fm_s.fin();
                    acc.sum_d += band.fm_d.fin();
                    acc.sum_s2 += band.fm_s2.fin();
                    acc.sum_d2 += band.fm_d2.fin();
                    acc.sum_dd += band.fm_dd.fin();
                    acc.sum_ds += band.fm_ds.fin();
                }
                if free.bounded_err {
                    acc.sum_msat += band.be_m.fin();
                    if free.lum_bins {
                        acc.lum_wd_num += band.wd_num.fin();
                        acc.lum_wd_den += band.wd_den.fin();
                        acc.lum_wb_num += band.wb_num.fin();
                        acc.lum_wb_den += band.wb_den.fin();
                    }
                }
            }
        }

        // Slide V-blur window — same per-column recurrence as production.
        win.slide(y, &planes);
    }

    acc
}

/// **Rev5 `localwin` fused V-blur** — [`fused_vblur_ssim_canon`]'s Rev5 arm:
/// the V-blurred plane values at each inner row come from that row's own
/// 11-tap window through [`crate::blur::local_sum_taps`] — there is no `VWin`
/// state, so non-inner rows are skipped entirely (a recurrence would have to
/// visit them; a local window does not) and no rounding can travel between
/// rows. Elements and pools run the identical [`vblur_ssim_elem_canon`] step
/// the other canonical bodies run.
#[allow(clippy::too_many_arguments)]
fn fused_vblur_ssim_local<P: crate::featcanon::Pool>(
    h_mu1: &[f32],
    h_mu2: &[f32],
    h_sigma_sq: &[f32],
    h_sigma12: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    sd_out: &mut [f32],
    store_sd: bool,
    ssq_out: &mut [f32],
    s12_out: &mut [f32],
    store_sigma: bool,
    free: FreeExtrasWork,
    direct: bool,
    ext: ExtPoolsWork,
    h_act: &[f32],
    // Kept for signature parity with `fused_vblur_ssim_canon`; the axis was
    // already resolved to `Local` by the caller.
    _mode: crate::featcanon::Mode,
) -> StripChannelAccum {
    if radius > 0 {
        #[cfg(feature = "feature-regime-v2")]
        crate::fold_timing::work(crate::fold_timing::Work::VerticalPlane, 4);
    }
    if !free.local_only && !free.omit_peaks {
        #[cfg(feature = "feature-regime-v2")]
        crate::fold_timing::work(crate::fold_timing::Work::PeakBand, 1);
    }
    let form = free.luma_form();
    // Revision 7: the v1 edge ratio without cancellation.
    let stable7 = free.revision().stable_forms();
    // Revision 7's row lanes: the columns of complete eight-column chunks add
    // the f32-lane pools in `crate::featcanon::Lanes16F32`, folded every
    // `ROW_LANES_FOLD` columns and at the end of the chunks, as the vector
    // body does; the tail adds in f64.
    use crate::featcanon::ROW_LANES_FOLD;
    let rev7 = free.revision().row_lanes() && P::LANES == 16;
    let chunk_end = (width / 8) * 8;
    let r = radius;
    let inner_end = inner_start + inner_h;
    let planes = VWinPlanes {
        m1: h_mu1,
        m2: h_mu2,
        sq: h_sigma_sq,
        s12: h_sigma12,
        act: if ext.on { h_act } else { &[] },
        width,
        height,
        r,
    };
    let diam = planes.diam();
    let inv = planes.inv32();
    let has_act = !planes.act.is_empty();

    let mut acc = StripChannelAccum::zero();
    let mut band = BandPools::<P>::zero();
    let mut t_stack = [0.0f32; 33];
    let mut t_heap;
    let t = if diam <= 33 {
        &mut t_stack[..diam]
    } else {
        t_heap = vec![0.0; diam];
        &mut t_heap[..]
    };
    let mut rb_stack = [0usize; 33];
    let mut rb_heap;
    let rb = if diam <= 33 {
        &mut rb_stack[..diam]
    } else {
        rb_heap = vec![0; diam];
        &mut rb_heap[..]
    };

    // `localwin` rows: only the inner range produces outputs — nothing below
    // or above feeds a recurrence, so those rows are never visited.
    // Revision 7: the row pools' f64 lanes live for the whole call (each row's
    // f32 row lanes fold into them, and every fold clears the row lanes), and
    // each pool finishes once, after the last row.
    let mut call_row = VblurPools::<P>::zero();
    let mut row32 = VblurPools::<crate::featcanon::Lanes16F32>::zero();
    for y in inner_start..inner_end {
        for k in 0..diam {
            rb[k] =
                crate::featcanon::tap_mirror(y as isize + k as isize - r as isize, height) * width;
        }
        let base = y * width;
        let mut fresh = VblurPools::<P>::zero();
        let row = if rev7 { &mut call_row } else { &mut fresh };
        for x in 0..width {
            let mut win_at = |plane: &[f32]| -> f32 {
                if plane.is_empty() {
                    return 0.0;
                }
                // Rev5's shared vertical planes arrive here with radius zero.
                // The one-leaf tree and multiplication by one are the input
                // value itself; do not stage it through the general tap loop.
                if radius == 0 {
                    return plane[base + x];
                }
                for (k, t) in t.iter_mut().enumerate() {
                    *t = plane[rb[k] + x];
                }
                crate::blur::local_sum_taps(t) * inv
            };
            let mu1 = win_at(planes.m1);
            let mu2 = win_at(planes.m2);
            let ssq = win_at(planes.sq);
            let s12 = win_at(planes.s12);
            let act = if has_act { win_at(planes.act) } else { 0.0 };
            let s = src[base + x];
            let d = dst[base + x];
            vblur_ssim_elem_canon(
                mu1,
                mu2,
                ssq,
                s12,
                act,
                s,
                d,
                x,
                base,
                form,
                direct,
                stable7,
                free,
                ext,
                store_mu,
                store_sd,
                store_sigma,
                mu1_out,
                mu2_out,
                sd_out,
                ssq_out,
                s12_out,
                &mut *row,
                if rev7 && x < chunk_end {
                    Some(&mut row32)
                } else {
                    None
                },
                &mut band,
                &mut acc,
            );
            if rev7
                && (x + 1 == chunk_end || (x < chunk_end && (x + 1).is_multiple_of(ROW_LANES_FOLD)))
            {
                row32.fold_into(&mut *row, free, ext);
            }
        }
        // One fixed-order finish per pool: per row, or at revision 7 after
        // the last row.
        if !rev7 {
            row.finish_into(&mut acc, free, ext);
        }
        if y + 1 == inner_end {
            if rev7 {
                call_row.finish_into(&mut acc, free, ext);
            }
            if free.raw_moments {
                acc.sum_s += band.fm_s.fin();
                acc.sum_d += band.fm_d.fin();
                acc.sum_s2 += band.fm_s2.fin();
                acc.sum_d2 += band.fm_d2.fin();
                acc.sum_dd += band.fm_dd.fin();
                acc.sum_ds += band.fm_ds.fin();
            }
            if free.bounded_err {
                acc.sum_msat += band.be_m.fin();
                if free.lum_bins {
                    acc.lum_wd_num += band.wd_num.fin();
                    acc.lum_wd_den += band.wd_den.fin();
                    acc.lum_wb_num += band.wb_num.fin();
                    acc.lum_wb_den += band.wb_den.fin();
                }
            }
        }
    }

    acc
}

/// **canon64 vector body** — [`fused_vblur_ssim_canon`]'s `P = LanesF64` arm
/// over the `Rec64` window, with the ~30 pool element formulas evaluated
/// eight columns at a time through `f32x8`, the `(state * inv64) as f32`
/// window reads through `f64x8`, and the `width % 8` tail on
/// [`vblur_ssim_elem_canon`] — literally the scalar element step.
///
/// **Why this is bit-identical to the `LanesF64` scalar body.** Every
/// column is an independent computation: the Rec64 window slides per
/// column (`state + add − rem`, elementwise — `vwin_slide64x8!`), `at32`
/// is a per-element `* inv64` + single f64→f32 narrow, every element
/// formula is elementwise across columns, and pool adds are lane-local.
/// Column `x` maps to canonical lane `x & 7`, so an 8-aligned chunk's
/// `to_array()` lanes ARE the canonical lanes in order — each chunk add is
/// `add(l, v_l)` for `l = 0..8` with the same values in the same pool, in
/// the same per-pool order. No reduction is reassociated. The `mul_add`
/// spellings are `SsimSplats8`'s — the documented same-op-sequence sibling
/// of `ssim_dissim_raw_scalar`/`ssim_direct_raw_scalar` — fused FMA on
/// v3/v4/v4x/NEON.
///
/// Entered only when [`crate::featcanon::canon_blur_axis`] resolves `Rec64`
/// (the Canon64 arm's own axis; an oracle `ZENSIM_FEATCANON_BLUR` override
/// falls back to the scalar body, which honours the other variants).
/// `_scalar`/`_wasm128` keep the scalar canonical body: their `mul_add` is
/// unfused either way, so the scalar body is already their exact form.
#[magetypes(define(f32x8, f32x16, f64x8), v4x, v4, v3, -scalar)]
#[allow(clippy::too_many_arguments)]
fn fused_vblur_ssim_canon64_vec<P: crate::featcanon::Pool64>(
    token: Token,
    h_mu1: &[f32],
    h_mu2: &[f32],
    h_sigma_sq: &[f32],
    h_sigma12: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    sd_out: &mut [f32],
    store_sd: bool,
    ssq_out: &mut [f32],
    s12_out: &mut [f32],
    store_sigma: bool,
    free: FreeExtrasWork,
    direct: bool,
    ext: ExtPoolsWork,
    h_act: &[f32],
    mode: crate::featcanon::Mode,
) -> StripChannelAccum {
    let form = free.luma_form();
    // Revision 7: the v1 edge ratio without cancellation.
    let stable7 = free.revision().stable_forms();
    let r = radius;
    let inner_end = inner_start + inner_h;
    let planes = VWinPlanes {
        m1: h_mu1,
        m2: h_mu2,
        sq: h_sigma_sq,
        s12: h_sigma12,
        act: if ext.on { h_act } else { &[] },
        width,
        height,
        r,
    };
    let local = P::LANES == 16;
    if local {
        if radius > 0 {
            crate::fold_timing::work(crate::fold_timing::Work::VerticalPlane, 4);
        }
        if !free.local_only && !free.omit_peaks {
            crate::fold_timing::work(crate::fold_timing::Work::PeakBand, 1);
        }
    }
    let mut win = if local {
        VWin::Local
    } else {
        VWin::new(crate::featcanon::canon_blur_axis(mode), &planes)
    };
    debug_assert!(
        local || matches!(win, VWin::F64 { .. }),
        "canon64 vector body is entered only under the Rec64 axis"
    );
    let inv_v = f64x8::splat_t(token, planes.inv64());
    let splats = crate::ssim_form::SsimSplats8::new(token, form);
    let one = f32x8::splat_t(token, 1.0);
    let zero = f32x8::zero_t(token);
    let c_mse = f32x8::splat_t(token, C_MSE_F32);
    let c_lumt = f32x8::splat_t(token, C_LUM_T_F32);
    let km = f32x8::splat_t(token, ext.k_mask);
    let kiw = f32x8::splat_t(token, ext.k_iw);
    let mut acc = StripChannelAccum::zero();
    let mut band = BandPools::<P>::zero();
    let full = width / 8;
    // One slice construction per plane: every `base + x` / `x0 + 8` index
    // below is then statically `< height * width` or `<= full * 8`, so the
    // chunk loop and the scalar tail compile with no per-access bounds
    // checks (the tail still runs `vblur_ssim_elem_canon` verbatim).
    let src = &src[..height * width];
    let dst = &dst[..height * width];
    // `at32`'s F64 arm, one 8-column chunk: `(state[x] * inv64) as f32`
    // per lane — the narrow stays per-element inside the chunk.
    let narrow = |w: &[f64], x0: usize| -> f32x8 {
        let a = (f64x8::load_t(token, w[x0..x0 + 8].try_into().unwrap()) * inv_v).to_array();
        f32x8::from_array_t(token, std::array::from_fn(|i| a[i] as f32))
    };

    // Revision 7 (`rowlanes`): each f32-lane pool's chunk adds go to
    // `crate::featcanon::Lanes16F32` for the row, folded into the f64 lanes
    // every `ROW_LANES_FOLD` columns and at the end of the chunks
    // (`VblurPools::fold_into`); the tail adds in f64. Revisions 5 and 6 keep
    // the f64 per-chunk update.
    let rev7 = free.revision().row_lanes() && local;
    macro_rules! pool_add {
        ($tgt:ident, $tgt32:ident, $field:ident, $x0:expr, $v:expr) => {{
            let v: f32x8 = $v;
            if rev7 {
                let lanes = $tgt32.$field.chunk8($x0 & 15);
                (f32x8::load_t(token, lanes) + v).store(lanes);
            } else {
                let values = v.to_array();
                let lanes = $tgt.$field.chunk8($x0 & (P::LANES - 1));
                for l in 0..8 {
                    lanes[l] += values[l] as f64;
                }
            }
        }};
    }
    // Pools whose features difference or ratio large sums keep f64 lanes
    // (f32 row sums lose up to 2.4e-4 of those features to the
    // cancellation): the band raw moments, and through revision 6 the v1
    // variance, texture and contrast sums, which revision 7 accumulates in
    // the f32 lanes as differences instead (`VblurPools::fold_into`).
    macro_rules! pool_add64 {
        ($tgt:ident, $field:ident, $x0:expr, $v:expr) => {{
            let values = ($v).to_array();
            let lanes = $tgt.$field.chunk8($x0 & (P::LANES - 1));
            for l in 0..8 {
                lanes[l] += values[l] as f64;
            }
        }};
    }
    // Revision 7: the row pools' f64 lanes live for the whole call (each row's
    // f32 row lanes fold into them, and every fold clears the row lanes), and
    // each pool finishes once, after the last row.
    let mut call_row = VblurPools::<P>::zero();
    let mut row32 = VblurPools::<crate::featcanon::Lanes16F32>::zero();
    for y in if local {
        inner_start..inner_end
    } else {
        0..height
    } {
        if y >= inner_start && y < inner_end {
            let base = y * width;
            let mut fresh = VblurPools::<P>::zero();
            let row = if rev7 { &mut call_row } else { &mut fresh };
            let [wm1, wm2, wsq, ws12, wact] = win.f64_states().unwrap_or([&[]; 5]);
            // Provable-length slices: `x0 + 8 <= full * 8` is then literal,
            // so `narrow`'s window reads and the plane loads compile with
            // no `slice_index_fail` path. (`wact` stays conditionally read.)
            let wm1 = if local { wm1 } else { &wm1[..full * 8] };
            let wm2 = if local { wm2 } else { &wm2[..full * 8] };
            let wsq = if local { wsq } else { &wsq[..full * 8] };
            let ws12 = if local { ws12 } else { &ws12[..full * 8] };
            let wact = if wact.is_empty() {
                wact
            } else {
                &wact[..full * 8]
            };
            let srow = &src[base..base + full * 8];
            let drow = &dst[base..base + full * 8];

            // The vertical window's eleven mirrored tap rows, once per row:
            // each tap plane is cut to the chunks' exact span, so the loads
            // need no per-chunk offset arithmetic or range checks. Same taps,
            // same summation tree.
            let tap_base: [usize; 11] = if local && radius != 0 {
                std::array::from_fn(|k| {
                    crate::featcanon::tap_mirror(y as isize + k as isize - 5, height) * width
                })
            } else {
                [0; 11]
            };
            macro_rules! taps {
                ($plane:expr) => {{
                    let plane: &[f32] = $plane;
                    let t: [&[f32]; 11] = if !local || radius == 0 || plane.is_empty() {
                        [&[]; 11]
                    } else {
                        std::array::from_fn(|k| &plane[tap_base[k]..tap_base[k] + full * 8])
                    };
                    t
                }};
            }
            let (tm1, tm2, tsq, ts12, tact) = (
                taps!(planes.m1),
                taps!(planes.m2),
                taps!(planes.sq),
                taps!(planes.s12),
                taps!(planes.act),
            );
            // One chunk of `$w` columns (eight or sixteen) at column `x0`,
            // through vector type `$vt`: every term is element-wise, and the
            // pools add each lane's values in column order, so a sixteen-wide
            // step computes the same bits as its two eight-wide chunks.
            macro_rules! window8 {
                ($plane:expr, $state:expr, $taps:expr, $x0:expr) => {{
                    let x0: usize = $x0;
                    if !local {
                        narrow($state, x0)
                    } else if radius == 0 {
                        f32x8::load_t(token, $plane[base + x0..base + x0 + 8].try_into().unwrap())
                    } else {
                        debug_assert_eq!(radius, 5);
                        let tp: &[&[f32]; 11] = &$taps;
                        let t: [f32x8; 11] = std::array::from_fn(|k| {
                            f32x8::load_t(token, tp[k][x0..x0 + 8].try_into().unwrap())
                        });
                        ((((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7])))
                            + (t[8] + t[9])
                            + t[10])
                            * f32x8::splat_t(token, 1.0 / 11.0)
                    }
                }};
            }
            macro_rules! window16 {
                ($plane:expr, $state:expr, $taps:expr, $x0:expr) => {{
                    let x0: usize = $x0;
                    let _ = $state;
                    if radius == 0 {
                        f32x16::load_t(token, $plane[base + x0..base + x0 + 16].try_into().unwrap())
                    } else {
                        let tp: &[&[f32]; 11] = &$taps;
                        let t: [f32x16; 11] = std::array::from_fn(|k| {
                            f32x16::load_t(token, tp[k][x0..x0 + 16].try_into().unwrap())
                        });
                        ((((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7])))
                            + (t[8] + t[9])
                            + t[10])
                            * f32x16::splat_t(token, 1.0 / 11.0)
                    }
                }};
            }
            macro_rules! chunk {
                ($vt:ident, $w:literal, $splats:expr, $consts:expr, $add:ident, $add64:ident, $window:ident, $x0:expr) => {{
                    let x0: usize = $x0;
                    let (one, zero, c_mse, c_lumt, km, kiw) = $consts;
                    let m1v = $window!(planes.m1, wm1, tm1, x0);
                    let m2v = $window!(planes.m2, wm2, tm2, x0);
                    let ssqv = $window!(planes.sq, wsq, tsq, x0);
                    let s12v = $window!(planes.s12, ws12, ts12, x0);
                    let actv = if planes.act.is_empty() {
                        zero
                    } else {
                        $window!(planes.act, wact, tact, x0)
                    };
                    let sv = $vt::load_t(token, srow[x0..x0 + $w].try_into().unwrap());
                    let dv = $vt::load_t(token, drow[x0..x0 + $w].try_into().unwrap());

                    let sd = if direct {
                        $splats.direct(m1v, m2v, ssqv, s12v)
                    } else {
                        $splats.dissim(m1v, m2v, ssqv, s12v)
                    }
                    .max(zero);
                    let sd2 = sd * sd;
                    let sd4 = sd2 * sd2;
                    $add!(row, row32, ssim_d, x0, sd);
                    $add!(row, row32, ssim_d4, x0, sd4);
                    $add!(row, row32, ssim_d2, x0, sd2);
                    if !free.local_only && !free.omit_peaks {
                        $add!(row, row32, ssim_d8, x0, sd4 * sd4);
                        for v in sd.to_array() {
                            acc.ssim_max = acc.ssim_max.max(v);
                        }
                    }
                    if store_sd {
                        sd_out[base + x0..base + x0 + $w].copy_from_slice(&sd.to_array());
                    }
                    if store_mu {
                        mu1_out[base + x0..base + x0 + $w].copy_from_slice(&m1v.to_array());
                        mu2_out[base + x0..base + x0 + $w].copy_from_slice(&m2v.to_array());
                    }
                    if store_sigma {
                        ssq_out[base + x0..base + x0 + $w].copy_from_slice(&ssqv.to_array());
                        s12_out[base + x0..base + x0 + $w].copy_from_slice(&s12v.to_array());
                    }

                    // Edge
                    let diff1 = (sv - m1v).abs();
                    let diff2 = (dv - m2v).abs();
                    let ed = if stable7 {
                        (diff2 - diff1) / (one + diff1)
                    } else {
                        (one + diff2) / (one + diff1) - one
                    };
                    let artifact = ed.max(zero);
                    let detail_lost = (-ed).max(zero);
                    let a2 = artifact * artifact;
                    let dl2 = detail_lost * detail_lost;
                    let a4 = a2 * a2;
                    let dl4 = dl2 * dl2;
                    if !free.omit_edges {
                        $add!(row, row32, edge_art, x0, artifact);
                        $add!(row, row32, edge_art4, x0, a4);
                        $add!(row, row32, edge_art2, x0, a2);
                        $add!(row, row32, edge_det, x0, detail_lost);
                        $add!(row, row32, edge_det4, x0, dl4);
                        $add!(row, row32, edge_det2, x0, dl2);
                        if !free.omit_peaks {
                            $add!(row, row32, edge_art8, x0, a4 * a4);
                            $add!(row, row32, edge_det8, x0, dl4 * dl4);
                            for v in artifact.to_array() {
                                acc.edge_art_max = acc.edge_art_max.max(v);
                            }
                            for v in detail_lost.to_array() {
                                acc.edge_det_max = acc.edge_det_max.max(v);
                            }
                        }
                    }

                    // Variance / texture
                    let vs = sv - m1v;
                    let vd = dv - m2v;
                    if !free.local_only {
                        if rev7 {
                            // The difference form of the scalar step.
                            $add!(row, row32, hf_sq_src, x0, vs * vs);
                            $add!(row, row32, hf_sq_dst, x0, (vd - vs) * (vd + vs));
                            $add!(row, row32, hf_abs_src, x0, diff1);
                            $add!(row, row32, hf_abs_dst, x0, diff2 - diff1);
                        } else {
                            $add64!(row, hf_sq_src, x0, vs * vs);
                            $add64!(row, hf_sq_dst, x0, vd * vd);
                            $add64!(row, hf_abs_src, x0, diff1);
                            $add64!(row, hf_abs_dst, x0, diff2);
                        }
                    }

                    // MSE + ext pools
                    let pd = sv - dv;
                    let d2 = pd * pd;
                    $add!(row, row32, mse, x0, d2);
                    if ext.on {
                        $add!(row, row32, act_sum, x0, actv);
                        if ext.mask {
                            let w = one / km.mul_add(actv, one);
                            let da = (sd * w).max(zero);
                            let d2a = da * da;
                            $add!(row, row32, masked_ssim_d, x0, da);
                            $add!(row, row32, masked_ssim_d2, x0, d2a);
                            $add!(row, row32, masked_ssim_d4, x0, d2a * d2a);
                            let e = ed * w;
                            let ep = e.max(zero);
                            let en = (-e).max(zero);
                            let a2m = ep * ep;
                            let dl2m = en * en;
                            $add!(row, row32, masked_art4, x0, a2m * a2m);
                            $add!(row, row32, masked_det4, x0, dl2m * dl2m);
                            $add!(row, row32, masked_mse, x0, d2 * w);
                        }
                        if ext.iw {
                            let w = kiw.mul_add(actv, one);
                            let db = (sd * w).max(zero);
                            let d2b = db * db;
                            $add!(row, row32, iw_ssim_d, x0, db);
                            $add!(row, row32, iw_ssim_d2, x0, d2b);
                            $add!(row, row32, iw_ssim_d4, x0, d2b * d2b);
                            let e = ed * w;
                            let ep = e.max(zero);
                            let en = (-e).max(zero);
                            let a2i = ep * ep;
                            let dl2i = en * en;
                            $add!(row, row32, iw_art4, x0, a2i * a2i);
                            $add!(row, row32, iw_det4, x0, dl2i * dl2i);
                            $add!(row, row32, iw_mse, x0, d2 * w);
                        }
                    }

                    // Free raw moments — band-lifetime canonical lanes.
                    if free.raw_moments {
                        $add64!(band, fm_s, x0, sv);
                        $add64!(band, fm_d, x0, dv);
                        $add64!(band, fm_s2, x0, sv * sv);
                        $add64!(band, fm_d2, x0, dv * dv);
                        let df = dv - sv;
                        $add64!(band, fm_dd, x0, df * (dv + sv));
                        $add64!(band, fm_ds, x0, df);
                    }
                    // Free bounded error + luminance bins.
                    if free.bounded_err {
                        let sqm = (pd * pd).max(zero);
                        let be_i = sqm / (sqm + c_mse);
                        $add64!(band, be_m, x0, be_i);
                        if free.lum_bins {
                            let ry = sv.max(zero);
                            let t = ry / (ry + c_lumt);
                            let omt = one - t;
                            let wd = omt * omt;
                            let wb = t * t;
                            $add64!(band, wd_num, x0, wd * be_i);
                            $add64!(band, wd_den, x0, wd);
                            $add64!(band, wb_num, x0, wb * be_i);
                            $add64!(band, wb_den, x0, wb);
                        }
                    }
                    if rev7 && (x0 + $w).is_multiple_of(crate::featcanon::ROW_LANES_FOLD) {
                        row32.fold_into(&mut *row, free, ext);
                    }
                }};
            }
            // The configuration B and Tga run (radius 0, direct SSIM, edges and
            // HF on, no peaks, extensions, free extras or stores) has its own
            // loops at revision 7.
            let lean = rev7
                && radius == 0
                && direct
                && !free.local_only
                && free.omit_peaks
                && !free.omit_edges
                && !free.raw_moments
                && !free.bounded_err
                && !ext.on
                && !store_sd
                && !store_mu
                && !store_sigma;
            let mut c = 0usize;
            // Revision 7: sixteen columns per step on AVX-512 tiers.
            if rev7 && crate::feature_v2::steps_sixteen::<Token>() {
                let splats16 = crate::ssim_form::SsimSplats16::new(token, form);
                let consts16 = (
                    f32x16::splat_t(token, 1.0),
                    f32x16::zero_t(token),
                    f32x16::splat_t(token, C_MSE_F32),
                    f32x16::splat_t(token, C_LUM_T_F32),
                    f32x16::splat_t(token, ext.k_mask),
                    f32x16::splat_t(token, ext.k_iw),
                );
                macro_rules! pool_add16 {
                    ($tgt:ident, $tgt32:ident, $field:ident, $x0:expr, $v:expr) => {{
                        let v: f32x16 = $v;
                        if rev7 {
                            let lanes = &mut $tgt32.$field.0;
                            (f32x16::load_t(token, lanes) + v).store(lanes);
                        } else {
                            pool_add64_16!($tgt, $field, $x0, v);
                        }
                    }};
                }
                macro_rules! pool_add64_16 {
                    ($tgt:ident, $field:ident, $x0:expr, $v:expr) => {{
                        let values = ($v).to_array();
                        for (off, half) in [(0usize, &values[..8]), (8, &values[8..])] {
                            let lanes = $tgt.$field.chunk8(off);
                            for l in 0..8 {
                                lanes[l] += half[l] as f64;
                            }
                        }
                    }};
                }
                // The lean configuration's fourteen row pools stay in registers
                // between folds, and no per-chunk flag is tested. Each lane adds
                // the same values in the same column order as `chunk!`, so every
                // sum keeps its bits.
                if lean {
                    macro_rules! get {
                        ($f:ident) => {
                            f32x16::load_t(token, &row32.$f.0)
                        };
                    }
                    macro_rules! put {
                        ($f:ident, $v:expr) => {
                            ($v).store(&mut row32.$f.0)
                        };
                    }
                    let mut p = [
                        get!(ssim_d),
                        get!(ssim_d4),
                        get!(ssim_d2),
                        get!(edge_art),
                        get!(edge_art4),
                        get!(edge_art2),
                        get!(edge_det),
                        get!(edge_det4),
                        get!(edge_det2),
                        get!(hf_sq_src),
                        get!(hf_sq_dst),
                        get!(hf_abs_src),
                        get!(hf_abs_dst),
                        get!(mse),
                    ];
                    macro_rules! flush {
                        () => {
                            put!(ssim_d, p[0]);
                            put!(ssim_d4, p[1]);
                            put!(ssim_d2, p[2]);
                            put!(edge_art, p[3]);
                            put!(edge_art4, p[4]);
                            put!(edge_art2, p[5]);
                            put!(edge_det, p[6]);
                            put!(edge_det4, p[7]);
                            put!(edge_det2, p[8]);
                            put!(hf_sq_src, p[9]);
                            put!(hf_sq_dst, p[10]);
                            put!(hf_abs_src, p[11]);
                            put!(hf_abs_dst, p[12]);
                            put!(mse, p[13]);
                        };
                    }
                    let (one, zero) = (consts16.0, consts16.1);
                    let (m1p, m2p, sqp, s12p) = (planes.m1, planes.m2, planes.sq, planes.s12);
                    while c + 2 <= full {
                        let x0 = c * 8;
                        let at = base + x0;
                        let m1v = f32x16::load_t(token, m1p[at..at + 16].try_into().unwrap());
                        let m2v = f32x16::load_t(token, m2p[at..at + 16].try_into().unwrap());
                        let ssqv = f32x16::load_t(token, sqp[at..at + 16].try_into().unwrap());
                        let s12v = f32x16::load_t(token, s12p[at..at + 16].try_into().unwrap());
                        let sv = f32x16::load_t(token, srow[x0..x0 + 16].try_into().unwrap());
                        let dv = f32x16::load_t(token, drow[x0..x0 + 16].try_into().unwrap());
                        let sd = splats16.direct(m1v, m2v, ssqv, s12v).max(zero);
                        let sd2 = sd * sd;
                        let sd4 = sd2 * sd2;
                        p[0] = p[0] + sd;
                        p[1] = p[1] + sd4;
                        p[2] = p[2] + sd2;
                        let diff1 = (sv - m1v).abs();
                        let diff2 = (dv - m2v).abs();
                        let ed = if stable7 {
                            (diff2 - diff1) / (one + diff1)
                        } else {
                            (one + diff2) / (one + diff1) - one
                        };
                        let artifact = ed.max(zero);
                        let detail_lost = (-ed).max(zero);
                        let a2 = artifact * artifact;
                        let dl2 = detail_lost * detail_lost;
                        p[3] = p[3] + artifact;
                        p[4] = p[4] + a2 * a2;
                        p[5] = p[5] + a2;
                        p[6] = p[6] + detail_lost;
                        p[7] = p[7] + dl2 * dl2;
                        p[8] = p[8] + dl2;
                        let vs = sv - m1v;
                        let vd = dv - m2v;
                        p[9] = p[9] + vs * vs;
                        p[10] = p[10] + (vd - vs) * (vd + vs);
                        p[11] = p[11] + diff1;
                        p[12] = p[12] + (diff2 - diff1);
                        let pd = sv - dv;
                        p[13] = p[13] + pd * pd;
                        if (x0 + 16).is_multiple_of(crate::featcanon::ROW_LANES_FOLD) {
                            flush!();
                            row32.fold_into(&mut *row, free, ext);
                            p = [zero; 14];
                        }
                        c += 2;
                    }
                    flush!();
                }
                while c + 2 <= full {
                    chunk!(
                        f32x16,
                        16,
                        splats16,
                        consts16,
                        pool_add16,
                        pool_add64_16,
                        window16,
                        c * 8
                    );
                    c += 2;
                }
            }
            let consts8 = (one, zero, c_mse, c_lumt, km, kiw);
            if lean {
                // The lean configuration eight columns at a time (v3, and the
                // odd last chunk on AVX-512): `chunk!`'s pool adds without its
                // per-chunk flag tests.
                let (m1p, m2p, sqp, s12p) = (planes.m1, planes.m2, planes.sq, planes.s12);
                for c in c..full {
                    let x0 = c * 8;
                    let at = base + x0;
                    let m1v = f32x8::load_t(token, m1p[at..at + 8].try_into().unwrap());
                    let m2v = f32x8::load_t(token, m2p[at..at + 8].try_into().unwrap());
                    let ssqv = f32x8::load_t(token, sqp[at..at + 8].try_into().unwrap());
                    let s12v = f32x8::load_t(token, s12p[at..at + 8].try_into().unwrap());
                    let sv = f32x8::load_t(token, srow[x0..x0 + 8].try_into().unwrap());
                    let dv = f32x8::load_t(token, drow[x0..x0 + 8].try_into().unwrap());
                    macro_rules! add8 {
                        ($f:ident, $v:expr) => {{
                            let lanes = row32.$f.chunk8(x0 & 15);
                            (f32x8::load_t(token, lanes) + $v).store(lanes);
                        }};
                    }
                    let sd = splats.direct(m1v, m2v, ssqv, s12v).max(zero);
                    let sd2 = sd * sd;
                    let sd4 = sd2 * sd2;
                    add8!(ssim_d, sd);
                    add8!(ssim_d4, sd4);
                    add8!(ssim_d2, sd2);
                    let diff1 = (sv - m1v).abs();
                    let diff2 = (dv - m2v).abs();
                    let ed = if stable7 {
                        (diff2 - diff1) / (one + diff1)
                    } else {
                        (one + diff2) / (one + diff1) - one
                    };
                    let artifact = ed.max(zero);
                    let detail_lost = (-ed).max(zero);
                    let a2 = artifact * artifact;
                    let dl2 = detail_lost * detail_lost;
                    add8!(edge_art, artifact);
                    add8!(edge_art4, a2 * a2);
                    add8!(edge_art2, a2);
                    add8!(edge_det, detail_lost);
                    add8!(edge_det4, dl2 * dl2);
                    add8!(edge_det2, dl2);
                    let vs = sv - m1v;
                    let vd = dv - m2v;
                    add8!(hf_sq_src, vs * vs);
                    add8!(hf_sq_dst, (vd - vs) * (vd + vs));
                    add8!(hf_abs_src, diff1);
                    add8!(hf_abs_dst, diff2 - diff1);
                    let pd = sv - dv;
                    add8!(mse, pd * pd);
                    if (x0 + 8).is_multiple_of(crate::featcanon::ROW_LANES_FOLD) {
                        row32.fold_into(&mut *row, free, ext);
                    }
                }
            } else {
                for c in c..full {
                    chunk!(
                        f32x8,
                        8,
                        splats,
                        consts8,
                        pool_add,
                        pool_add64,
                        window8,
                        c * 8
                    );
                }
            }
            // The end of the chunks, unless the last chunk already folded.
            if rev7 && full > 0 && !(full * 8).is_multiple_of(crate::featcanon::ROW_LANES_FOLD) {
                row32.fold_into(&mut *row, free, ext);
            }
            // Scalar tail — the canonical element step verbatim.
            for x in full * 8..width {
                let (mu1, mu2, ssq, s12, act) = win.at32(x, y, &planes);
                let s = src[base + x];
                let d = dst[base + x];
                vblur_ssim_elem_canon(
                    mu1,
                    mu2,
                    ssq,
                    s12,
                    act,
                    s,
                    d,
                    x,
                    base,
                    form,
                    direct,
                    stable7,
                    free,
                    ext,
                    store_mu,
                    store_sd,
                    store_sigma,
                    mu1_out,
                    mu2_out,
                    sd_out,
                    ssq_out,
                    s12_out,
                    &mut *row,
                    None,
                    &mut band,
                    &mut acc,
                );
            }
            // One fixed-order finish per pool: per row, or at revision 7 after
            // the last row.
            if !rev7 {
                row.finish_into(&mut acc, free, ext);
            }
            if y + 1 == inner_end {
                if rev7 {
                    call_row.finish_into(&mut acc, free, ext);
                }
                if free.raw_moments {
                    acc.sum_s += band.fm_s.fin();
                    acc.sum_d += band.fm_d.fin();
                    acc.sum_s2 += band.fm_s2.fin();
                    acc.sum_d2 += band.fm_d2.fin();
                    acc.sum_dd += band.fm_dd.fin();
                    acc.sum_ds += band.fm_ds.fin();
                }
                if free.bounded_err {
                    acc.sum_msat += band.be_m.fin();
                    if free.lum_bins {
                        acc.lum_wd_num += band.wd_num.fin();
                        acc.lum_wd_den += band.wd_den.fin();
                        acc.lum_wb_num += band.wb_num.fin();
                        acc.lum_wb_den += band.wb_den.fin();
                    }
                }
            }
        }

        // Slide V-blur window — same per-column recurrence, eight columns
        // per f64x8 op.
        if !local {
            vwin_slide64x8!(win, token, y, &planes);
        }
    }

    acc
}

/// The scalar-tier siblings of [`fused_vblur_ssim_canon64_vec`]: scalar and
/// wasm128 keep the `LanesF64` canonical body — their `mul_add` is unfused,
/// so the scalar body is already their exact element arithmetic.
#[allow(clippy::too_many_arguments)]
fn fused_vblur_ssim_canon64_vec_scalar<P: crate::featcanon::Pool64>(
    _token: archmage::ScalarToken,
    h_mu1: &[f32],
    h_mu2: &[f32],
    h_sigma_sq: &[f32],
    h_sigma12: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    sd_out: &mut [f32],
    store_sd: bool,
    ssq_out: &mut [f32],
    s12_out: &mut [f32],
    store_sigma: bool,
    free: FreeExtrasWork,
    direct: bool,
    ext: ExtPoolsWork,
    h_act: &[f32],
    mode: crate::featcanon::Mode,
) -> StripChannelAccum {
    if P::LANES == 16 {
        return fused_vblur_ssim_local::<P>(
            h_mu1,
            h_mu2,
            h_sigma_sq,
            h_sigma12,
            src,
            dst,
            width,
            height,
            inner_start,
            inner_h,
            radius,
            mu1_out,
            mu2_out,
            store_mu,
            sd_out,
            store_sd,
            ssq_out,
            s12_out,
            store_sigma,
            free,
            direct,
            ext,
            h_act,
            mode,
        );
    }
    fused_vblur_ssim_canon::<P>(
        h_mu1,
        h_mu2,
        h_sigma_sq,
        h_sigma12,
        src,
        dst,
        width,
        height,
        inner_start,
        inner_h,
        radius,
        mu1_out,
        mu2_out,
        store_mu,
        sd_out,
        store_sd,
        ssq_out,
        s12_out,
        store_sigma,
        free,
        direct,
        ext,
        h_act,
        mode,
    )
}

/// Wasm128 sibling of [`fused_vblur_ssim_canon64_vec_scalar`].
#[allow(clippy::too_many_arguments, dead_code)]
fn fused_vblur_ssim_canon64_vec_wasm128<P: crate::featcanon::Pool64>(
    _token: archmage::Wasm128Token,
    h_mu1: &[f32],
    h_mu2: &[f32],
    h_sigma_sq: &[f32],
    h_sigma12: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    sd_out: &mut [f32],
    store_sd: bool,
    ssq_out: &mut [f32],
    s12_out: &mut [f32],
    store_sigma: bool,
    free: FreeExtrasWork,
    direct: bool,
    ext: ExtPoolsWork,
    h_act: &[f32],
    mode: crate::featcanon::Mode,
) -> StripChannelAccum {
    if P::LANES == 16 {
        return fused_vblur_ssim_local::<P>(
            h_mu1,
            h_mu2,
            h_sigma_sq,
            h_sigma12,
            src,
            dst,
            width,
            height,
            inner_start,
            inner_h,
            radius,
            mu1_out,
            mu2_out,
            store_mu,
            sd_out,
            store_sd,
            ssq_out,
            s12_out,
            store_sigma,
            free,
            direct,
            ext,
            h_act,
            mode,
        );
    }
    fused_vblur_ssim_canon::<P>(
        h_mu1,
        h_mu2,
        h_sigma_sq,
        h_sigma12,
        src,
        dst,
        width,
        height,
        inner_start,
        inner_h,
        radius,
        mu1_out,
        mu2_out,
        store_mu,
        sd_out,
        store_sd,
        ssq_out,
        s12_out,
        store_sigma,
        free,
        direct,
        ext,
        h_act,
        mode,
    )
}

/// NEON runs the scalar canonical body (2026-10-03): NEON vector `max`/`min` propagate NaN where the canon's `f32::max`/
/// `f64::max` drop it (the REV4VEC CI aarch64 failure), so only the x86_64 fused tiers run a lane-parallel body.
#[allow(clippy::too_many_arguments, dead_code)]
fn fused_vblur_ssim_canon64_vec_neon<P: crate::featcanon::Pool64>(
    _token: archmage::NeonToken,
    h_mu1: &[f32],
    h_mu2: &[f32],
    h_sigma_sq: &[f32],
    h_sigma12: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    sd_out: &mut [f32],
    store_sd: bool,
    ssq_out: &mut [f32],
    s12_out: &mut [f32],
    store_sigma: bool,
    free: FreeExtrasWork,
    direct: bool,
    ext: ExtPoolsWork,
    h_act: &[f32],
    mode: crate::featcanon::Mode,
) -> StripChannelAccum {
    if P::LANES == 16 {
        return fused_vblur_ssim_local::<P>(
            h_mu1,
            h_mu2,
            h_sigma_sq,
            h_sigma12,
            src,
            dst,
            width,
            height,
            inner_start,
            inner_h,
            radius,
            mu1_out,
            mu2_out,
            store_mu,
            sd_out,
            store_sd,
            ssq_out,
            s12_out,
            store_sigma,
            free,
            direct,
            ext,
            h_act,
            mode,
        );
    }
    fused_vblur_ssim_canon::<P>(
        h_mu1,
        h_mu2,
        h_sigma_sq,
        h_sigma12,
        src,
        dst,
        width,
        height,
        inner_start,
        inner_h,
        radius,
        mu1_out,
        mu2_out,
        store_mu,
        sd_out,
        store_sd,
        ssq_out,
        s12_out,
        store_sigma,
        free,
        direct,
        ext,
        h_act,
        mode,
    )
}

/// f64-exact sibling of [`fused_vblur_ssim_canon`]: every element formula in
/// f64 (fused `f64::mul_add` mirrors `ssim_direct_raw_scalar`'s order), every
/// sum Neumaier-compensated, planes rounded once at the f32 store.
/// Measurement only (`oracle` feature).
#[cfg(feature = "oracle")]
#[allow(clippy::too_many_arguments)]
fn fused_vblur_ssim_exact(
    h_mu1: &[f32],
    h_mu2: &[f32],
    h_sigma_sq: &[f32],
    h_sigma12: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    sd_out: &mut [f32],
    store_sd: bool,
    ssq_out: &mut [f32],
    s12_out: &mut [f32],
    store_sigma: bool,
    free: FreeExtrasWork,
    direct: bool,
    ext: ExtPoolsWork,
    h_act: &[f32],
) -> StripChannelAccum {
    use crate::featcanon::{Neum64, Pool as _};
    let form = free.luma_form();
    // Revision 7: the v1 edge ratio without cancellation.
    let stable7 = free.revision().stable_forms();
    let r = radius;
    let inner_end = inner_start + inner_h;

    // featacc blur axis: under `exact` `canon_blur_axis` defaults to `Fresh`;
    // `BLUR=rec` replays the shipped f32 sliding sums (each element then
    // carries the production blur's own drift — the isolation cell).
    let planes = VWinPlanes {
        m1: h_mu1,
        m2: h_mu2,
        sq: h_sigma_sq,
        s12: h_sigma12,
        act: if ext.on { h_act } else { &[] },
        width,
        height,
        r,
    };
    let mut win = VWin::new(
        crate::featcanon::canon_blur_axis(crate::featcanon::Mode::Exact),
        &planes,
    );

    let mut acc = StripChannelAccum::zero();
    let mut band = BandPools::<Neum64>::zero();

    for y in 0..height {
        if y >= inner_start && y < inner_end {
            let base = y * width;
            let mut row = VblurPools::<Neum64>::zero();
            for x in 0..width {
                let (mu1, mu2, ssq, s12, act) = win.at64(x, y, &planes);
                let s = src[base + x] as f64;
                let d = dst[base + x] as f64;

                let sd = crate::ssim_form::ssim_dissim_exact(form, mu1, mu2, ssq, s12, direct)
                    .max(0.0f64);
                let sd2 = sd * sd;
                let sd4 = sd2 * sd2;
                row.ssim_d.add64(x, sd);
                row.ssim_d4.add64(x, sd4);
                row.ssim_d2.add64(x, sd2);
                if !free.local_only && !free.omit_peaks {
                    row.ssim_d8.add64(x, sd4 * sd4);
                    acc.ssim_max = acc.ssim_max.max(sd as f32);
                }
                if store_sd {
                    sd_out[base + x] = sd as f32;
                }
                if store_mu {
                    mu1_out[base + x] = mu1 as f32;
                    mu2_out[base + x] = mu2 as f32;
                }
                if store_sigma {
                    ssq_out[base + x] = ssq as f32;
                    s12_out[base + x] = s12 as f32;
                }

                let diff1 = (s - mu1).abs();
                let diff2 = (d - mu2).abs();
                let ed = if stable7 {
                    (diff2 - diff1) / (1.0f64 + diff1)
                } else {
                    (1.0f64 + diff2) / (1.0f64 + diff1) - 1.0f64
                };
                let artifact = ed.max(0.0f64);
                let detail_lost = (-ed).max(0.0f64);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                if !free.omit_edges {
                    row.edge_art.add64(x, artifact);
                    row.edge_art4.add64(x, a4);
                    row.edge_art2.add64(x, a2);
                    row.edge_det.add64(x, detail_lost);
                    row.edge_det4.add64(x, dl4);
                    row.edge_det2.add64(x, dl2);
                    row.edge_art8.add64(x, a4 * a4);
                    row.edge_det8.add64(x, dl4 * dl4);
                    acc.edge_art_max = acc.edge_art_max.max(artifact as f32);
                    acc.edge_det_max = acc.edge_det_max.max(detail_lost as f32);
                }

                let vs = s - mu1;
                let vd = d - mu2;
                if !free.local_only {
                    row.hf_sq_src.add64(x, vs * vs);
                    row.hf_sq_dst.add64(x, vd * vd);
                    row.hf_abs_src.add64(x, diff1);
                    row.hf_abs_dst.add64(x, diff2);
                }

                let pd = s - d;
                let d2 = pd * pd;
                row.mse.add64(x, d2);
                if ext.on {
                    row.act_sum.add64(x, act);
                    if ext.mask {
                        let w = 1.0f64 / (ext.k_mask as f64).mul_add(act, 1.0f64);
                        let da = (sd * w).max(0.0);
                        let d2a = da * da;
                        row.masked_ssim_d.add64(x, da);
                        row.masked_ssim_d2.add64(x, d2a);
                        row.masked_ssim_d4.add64(x, d2a * d2a);
                        let e = ed * w;
                        let a2 = e.max(0.0) * e.max(0.0);
                        let dl2 = (-e).max(0.0) * (-e).max(0.0);
                        row.masked_art4.add64(x, a2 * a2);
                        row.masked_det4.add64(x, dl2 * dl2);
                        row.masked_mse.add64(x, d2 * w);
                    }
                    if ext.iw {
                        let w = (ext.k_iw as f64).mul_add(act, 1.0f64);
                        let db = (sd * w).max(0.0);
                        let d2b = db * db;
                        row.iw_ssim_d.add64(x, db);
                        row.iw_ssim_d2.add64(x, d2b);
                        row.iw_ssim_d4.add64(x, d2b * d2b);
                        let e = ed * w;
                        let a2 = e.max(0.0) * e.max(0.0);
                        let dl2 = (-e).max(0.0) * (-e).max(0.0);
                        row.iw_art4.add64(x, a2 * a2);
                        row.iw_det4.add64(x, dl2 * dl2);
                        row.iw_mse.add64(x, d2 * w);
                    }
                }

                if free.raw_moments {
                    band.fm_s.add64(x, s);
                    band.fm_d.add64(x, d);
                    band.fm_s2.add64(x, s * s);
                    band.fm_d2.add64(x, d * d);
                    let df = d - s;
                    band.fm_dd.add64(x, df * (d + s));
                    band.fm_ds.add64(x, df);
                }
                if free.bounded_err {
                    let sqm = (pd * pd).max(0.0);
                    let be_i = sqm / (sqm + C_MSE_F32 as f64);
                    band.be_m.add64(x, be_i);
                    if free.lum_bins {
                        let ry = s.max(0.0);
                        let t = ry / (ry + C_LUM_T_F32 as f64);
                        let one_mt = 1.0 - t;
                        let wd = one_mt * one_mt;
                        let wb = t * t;
                        band.wd_num.add64(x, wd * be_i);
                        band.wd_den.add64(x, wd);
                        band.wb_num.add64(x, wb * be_i);
                        band.wb_den.add64(x, wb);
                    }
                }
            }
            acc.ssim_d += row.ssim_d.fin();
            acc.ssim_d4 += row.ssim_d4.fin();
            acc.ssim_d2 += row.ssim_d2.fin();
            acc.edge_art += row.edge_art.fin();
            acc.edge_art4 += row.edge_art4.fin();
            acc.edge_art2 += row.edge_art2.fin();
            acc.edge_det += row.edge_det.fin();
            acc.edge_det4 += row.edge_det4.fin();
            acc.edge_det2 += row.edge_det2.fin();
            acc.mse += row.mse.fin();
            if !free.local_only {
                acc.ssim_d8 += row.ssim_d8.fin();
                acc.edge_art8 += row.edge_art8.fin();
                acc.edge_det8 += row.edge_det8.fin();
                acc.hf_sq_src += row.hf_sq_src.fin();
                acc.hf_sq_dst += row.hf_sq_dst.fin();
                acc.hf_abs_src += row.hf_abs_src.fin();
                acc.hf_abs_dst += row.hf_abs_dst.fin();
            }
            if ext.on {
                acc.act_sum += row.act_sum.fin();
                if ext.mask {
                    acc.masked_ssim_d += row.masked_ssim_d.fin();
                    acc.masked_ssim_d4 += row.masked_ssim_d4.fin();
                    acc.masked_ssim_d2 += row.masked_ssim_d2.fin();
                    acc.masked_art4 += row.masked_art4.fin();
                    acc.masked_det4 += row.masked_det4.fin();
                    acc.masked_mse += row.masked_mse.fin();
                }
                if ext.iw {
                    acc.iw_ssim_d += row.iw_ssim_d.fin();
                    acc.iw_ssim_d4 += row.iw_ssim_d4.fin();
                    acc.iw_ssim_d2 += row.iw_ssim_d2.fin();
                    acc.iw_art4 += row.iw_art4.fin();
                    acc.iw_det4 += row.iw_det4.fin();
                    acc.iw_mse += row.iw_mse.fin();
                }
            }
            if y + 1 == inner_end {
                if free.raw_moments {
                    acc.sum_s += band.fm_s.fin();
                    acc.sum_d += band.fm_d.fin();
                    acc.sum_s2 += band.fm_s2.fin();
                    acc.sum_d2 += band.fm_d2.fin();
                    acc.sum_dd += band.fm_dd.fin();
                    acc.sum_ds += band.fm_ds.fin();
                }
                if free.bounded_err {
                    acc.sum_msat += band.be_m.fin();
                    if free.lum_bins {
                        acc.lum_wd_num += band.wd_num.fin();
                        acc.lum_wd_den += band.wd_den.fin();
                        acc.lum_wb_num += band.wb_num.fin();
                        acc.lum_wb_den += band.wb_den.fin();
                    }
                }
            }
        }

        win.slide(y, &planes);
    }

    acc
}

/// Canonical (candidate) edge-only fused V-blur — same structure as the SSIM
/// body with the SSIM/ext arms absent, mirroring `fused_vblur_edge_inner`.
/// The `Pool`s are the struct so the canon64 vector body and the scalar
/// tail share one element step.
struct EdgeRowPools<P: Copy> {
    edge_art: P,
    edge_art4: P,
    edge_art2: P,
    edge_art8: P,
    edge_det: P,
    edge_det4: P,
    edge_det2: P,
    edge_det8: P,
    hf_sq_src: P,
    hf_sq_dst: P,
    hf_abs_src: P,
    hf_abs_dst: P,
    mse: P,
}

impl<P: crate::featcanon::Pool> EdgeRowPools<P> {
    fn zero() -> Self {
        Self {
            edge_art: P::zero(),
            edge_art4: P::zero(),
            edge_art2: P::zero(),
            edge_art8: P::zero(),
            edge_det: P::zero(),
            edge_det4: P::zero(),
            edge_det2: P::zero(),
            edge_det8: P::zero(),
            hf_sq_src: P::zero(),
            hf_sq_dst: P::zero(),
            hf_abs_src: P::zero(),
            hf_abs_dst: P::zero(),
            mse: P::zero(),
        }
    }
}

/// One column's contribution to the edge row pools — the canonical
/// per-element sequence shared by [`fused_vblur_edge_canon`] (every `x`)
/// and the canon64 vector body's `width % 8` tail.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn vblur_edge_elem_canon<P: crate::featcanon::Pool>(
    mu1: f32,
    mu2: f32,
    s: f32,
    d: f32,
    x: usize,
    base: usize,
    store_mu: bool,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    row: &mut EdgeRowPools<P>,
    acc: &mut StripChannelAccum,
) {
    let lane = x & 7;
    if store_mu {
        mu1_out[base + x] = mu1;
        mu2_out[base + x] = mu2;
    }
    let diff1 = (s - mu1).abs();
    let diff2 = (d - mu2).abs();
    let ed = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;
    let artifact = ed.max(0.0f32);
    let detail_lost = (-ed).max(0.0f32);
    let a2 = artifact * artifact;
    let dl2 = detail_lost * detail_lost;
    let a4 = a2 * a2;
    let dl4 = dl2 * dl2;
    row.edge_art.add(lane, artifact);
    row.edge_art4.add(lane, a4);
    row.edge_art2.add(lane, a2);
    row.edge_det.add(lane, detail_lost);
    row.edge_det4.add(lane, dl4);
    row.edge_det2.add(lane, dl2);
    row.edge_art8.add(lane, a4 * a4);
    row.edge_det8.add(lane, dl4 * dl4);
    acc.edge_art_max = acc.edge_art_max.max(artifact);
    acc.edge_det_max = acc.edge_det_max.max(detail_lost);
    let vs = s - mu1;
    let vd = d - mu2;
    row.hf_sq_src.add(lane, vs * vs);
    row.hf_sq_dst.add(lane, vd * vd);
    row.hf_abs_src.add(lane, diff1);
    row.hf_abs_dst.add(lane, diff2);
    let pd = s - d;
    row.mse.add(lane, pd * pd);
}

#[allow(clippy::too_many_arguments)]
fn fused_vblur_edge_canon<P: crate::featcanon::Pool>(
    h_mu1: &[f32],
    h_mu2: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    // The computation's canonical mode — selects the blur window's own
    // arithmetic (`canon_blur_axis`): `Rec64` under the c64 canon.
    mode: crate::featcanon::Mode,
) -> StripChannelAccum {
    let r = radius;
    let inner_end = inner_start + inner_h;

    // rev4canon: the mode's OWN blur axis — `Rec64` under the c64 canon,
    // `Rec` only for the superseded `c32` oracle reproduction.
    let planes = VWinPlanes {
        m1: h_mu1,
        m2: h_mu2,
        sq: &[],
        s12: &[],
        act: &[],
        width,
        height,
        r,
    };
    let mut win = VWin::new(crate::featcanon::canon_blur_axis(mode), &planes);

    let mut acc = StripChannelAccum::zero();
    for y in 0..height {
        if y >= inner_start && y < inner_end {
            let base = y * width;
            let mut row = EdgeRowPools::<P>::zero();
            for x in 0..width {
                let (mu1, mu2, _, _, _) = win.at32(x, y, &planes);
                let s = src[base + x];
                let d = dst[base + x];
                vblur_edge_elem_canon(
                    mu1, mu2, s, d, x, base, store_mu, mu1_out, mu2_out, &mut row, &mut acc,
                );
            }
            acc.edge_art += row.edge_art.fin();
            acc.edge_art4 += row.edge_art4.fin();
            acc.edge_art2 += row.edge_art2.fin();
            acc.edge_det += row.edge_det.fin();
            acc.edge_det4 += row.edge_det4.fin();
            acc.edge_det2 += row.edge_det2.fin();
            acc.edge_art8 += row.edge_art8.fin();
            acc.edge_det8 += row.edge_det8.fin();
            acc.hf_sq_src += row.hf_sq_src.fin();
            acc.hf_sq_dst += row.hf_sq_dst.fin();
            acc.hf_abs_src += row.hf_abs_src.fin();
            acc.hf_abs_dst += row.hf_abs_dst.fin();
            acc.mse += row.mse.fin();
        }
        win.slide(y, &planes);
    }
    acc
}

/// **canon64 vector body** — [`fused_vblur_edge_canon`]'s `P = LanesF64`
/// arm over the `Rec64` window, same bit-identical construction as
/// [`fused_vblur_ssim_canon64_vec`]: eight independent columns per
/// `f32x8`/`f64x8` op, chunk lanes = canonical lanes, `width % 8` tail on
/// [`vblur_edge_elem_canon`].
#[magetypes(define(f32x8, f64x8), v4x, v4, v3, -scalar)]
#[allow(clippy::too_many_arguments)]
fn fused_vblur_edge_canon64_vec(
    token: Token,
    h_mu1: &[f32],
    h_mu2: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    mode: crate::featcanon::Mode,
) -> StripChannelAccum {
    use crate::featcanon::{LanesF64, Pool as _};
    let r = radius;
    let inner_end = inner_start + inner_h;
    let planes = VWinPlanes {
        m1: h_mu1,
        m2: h_mu2,
        sq: &[],
        s12: &[],
        act: &[],
        width,
        height,
        r,
    };
    let mut win = VWin::new(crate::featcanon::canon_blur_axis(mode), &planes);
    debug_assert!(
        matches!(win, VWin::F64 { .. }),
        "canon64 vector body is entered only under the Rec64 axis"
    );
    let inv_v = f64x8::splat_t(token, planes.inv64());
    let one = f32x8::splat_t(token, 1.0);
    let zero = f32x8::zero_t(token);
    let mut acc = StripChannelAccum::zero();
    let full = width / 8;
    // Same pre-slicing as `fused_vblur_ssim_canon64_vec`: provable lengths,
    // no per-access `slice_index_fail` path in the chunk loop or tail.
    let src = &src[..height * width];
    let dst = &dst[..height * width];
    let narrow = |w: &[f64], x0: usize| -> f32x8 {
        let a = (f64x8::load_t(token, w[x0..x0 + 8].try_into().unwrap()) * inv_v).to_array();
        f32x8::from_array_t(token, std::array::from_fn(|i| a[i] as f32))
    };

    for y in 0..height {
        if y >= inner_start && y < inner_end {
            let base = y * width;
            let mut row = EdgeRowPools::<LanesF64>::zero();
            let [wm1, wm2, _, _, _] = win.f64_states().expect("Rec64");
            let wm1 = &wm1[..full * 8];
            let wm2 = &wm2[..full * 8];
            let srow = &src[base..base + full * 8];
            let drow = &dst[base..base + full * 8];
            for c in 0..full {
                let x0 = c * 8;
                let m1v = narrow(wm1, x0);
                let m2v = narrow(wm2, x0);
                if store_mu {
                    mu1_out[base + x0..base + x0 + 8].copy_from_slice(&m1v.to_array());
                    mu2_out[base + x0..base + x0 + 8].copy_from_slice(&m2v.to_array());
                }
                let sv = f32x8::load_t(token, srow[x0..x0 + 8].try_into().unwrap());
                let dv = f32x8::load_t(token, drow[x0..x0 + 8].try_into().unwrap());
                let diff1 = (sv - m1v).abs();
                let diff2 = (dv - m2v).abs();
                let ed = (one + diff2) / (one + diff1) - one;
                let artifact = ed.max(zero);
                let detail_lost = (-ed).max(zero);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                row.edge_art.add_chunk(&artifact.to_array());
                row.edge_art4.add_chunk(&a4.to_array());
                row.edge_art2.add_chunk(&a2.to_array());
                row.edge_det.add_chunk(&detail_lost.to_array());
                row.edge_det4.add_chunk(&dl4.to_array());
                row.edge_det2.add_chunk(&dl2.to_array());
                row.edge_art8.add_chunk(&(a4 * a4).to_array());
                row.edge_det8.add_chunk(&(dl4 * dl4).to_array());
                for v in artifact.to_array() {
                    acc.edge_art_max = acc.edge_art_max.max(v);
                }
                for v in detail_lost.to_array() {
                    acc.edge_det_max = acc.edge_det_max.max(v);
                }
                let vs = sv - m1v;
                let vd = dv - m2v;
                row.hf_sq_src.add_chunk(&(vs * vs).to_array());
                row.hf_sq_dst.add_chunk(&(vd * vd).to_array());
                row.hf_abs_src.add_chunk(&diff1.to_array());
                row.hf_abs_dst.add_chunk(&diff2.to_array());
                let pd = sv - dv;
                row.mse.add_chunk(&(pd * pd).to_array());
            }
            for x in full * 8..width {
                let (mu1, mu2, _, _, _) = win.at32(x, y, &planes);
                let s = src[base + x];
                let d = dst[base + x];
                vblur_edge_elem_canon(
                    mu1, mu2, s, d, x, base, store_mu, mu1_out, mu2_out, &mut row, &mut acc,
                );
            }
            acc.edge_art += row.edge_art.fin();
            acc.edge_art4 += row.edge_art4.fin();
            acc.edge_art2 += row.edge_art2.fin();
            acc.edge_det += row.edge_det.fin();
            acc.edge_det4 += row.edge_det4.fin();
            acc.edge_det2 += row.edge_det2.fin();
            acc.edge_art8 += row.edge_art8.fin();
            acc.edge_det8 += row.edge_det8.fin();
            acc.hf_sq_src += row.hf_sq_src.fin();
            acc.hf_sq_dst += row.hf_sq_dst.fin();
            acc.hf_abs_src += row.hf_abs_src.fin();
            acc.hf_abs_dst += row.hf_abs_dst.fin();
            acc.mse += row.mse.fin();
        }
        vwin_slide64x8!(win, token, y, &planes);
    }
    acc
}

/// Scalar-tier siblings of [`fused_vblur_edge_canon64_vec`]: scalar and
/// wasm128 keep the `LanesF64` canonical body — their `mul_add` is unfused.
#[allow(clippy::too_many_arguments)]
fn fused_vblur_edge_canon64_vec_scalar(
    _token: archmage::ScalarToken,
    h_mu1: &[f32],
    h_mu2: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    mode: crate::featcanon::Mode,
) -> StripChannelAccum {
    fused_vblur_edge_canon::<crate::featcanon::LanesF64>(
        h_mu1,
        h_mu2,
        src,
        dst,
        width,
        height,
        inner_start,
        inner_h,
        radius,
        mu1_out,
        mu2_out,
        store_mu,
        mode,
    )
}

/// Wasm128 sibling of [`fused_vblur_edge_canon64_vec_scalar`].
#[allow(clippy::too_many_arguments, dead_code)]
fn fused_vblur_edge_canon64_vec_wasm128(
    _token: archmage::Wasm128Token,
    h_mu1: &[f32],
    h_mu2: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    mode: crate::featcanon::Mode,
) -> StripChannelAccum {
    fused_vblur_edge_canon::<crate::featcanon::LanesF64>(
        h_mu1,
        h_mu2,
        src,
        dst,
        width,
        height,
        inner_start,
        inner_h,
        radius,
        mu1_out,
        mu2_out,
        store_mu,
        mode,
    )
}

/// NEON runs the scalar canonical body (2026-10-03): NEON vector `max`/`min` propagate NaN where the canon's `f32::max`/
/// `f64::max` drop it (the REV4VEC CI aarch64 failure), so only the x86_64 fused tiers run a lane-parallel body.
#[allow(clippy::too_many_arguments, dead_code)]
fn fused_vblur_edge_canon64_vec_neon(
    _token: archmage::NeonToken,
    h_mu1: &[f32],
    h_mu2: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
    mode: crate::featcanon::Mode,
) -> StripChannelAccum {
    fused_vblur_edge_canon::<crate::featcanon::LanesF64>(
        h_mu1,
        h_mu2,
        src,
        dst,
        width,
        height,
        inner_start,
        inner_h,
        radius,
        mu1_out,
        mu2_out,
        store_mu,
        mode,
    )
}

/// f64-exact sibling of [`fused_vblur_edge_canon`]. Measurement only
/// (`oracle` feature).
#[cfg(feature = "oracle")]
#[allow(clippy::too_many_arguments)]
fn fused_vblur_edge_exact(
    h_mu1: &[f32],
    h_mu2: &[f32],
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    inner_start: usize,
    inner_h: usize,
    radius: usize,
    mu1_out: &mut [f32],
    mu2_out: &mut [f32],
    store_mu: bool,
) -> StripChannelAccum {
    use crate::featcanon::{Neum64, Pool as _};
    let r = radius;
    let inner_end = inner_start + inner_h;

    // featacc blur axis (Fresh is the `exact` default).
    let planes = VWinPlanes {
        m1: h_mu1,
        m2: h_mu2,
        sq: &[],
        s12: &[],
        act: &[],
        width,
        height,
        r,
    };
    let mut win = VWin::new(
        crate::featcanon::canon_blur_axis(crate::featcanon::Mode::Exact),
        &planes,
    );

    let mut acc = StripChannelAccum::zero();
    for y in 0..height {
        if y >= inner_start && y < inner_end {
            let base = y * width;
            let mut edge_art = Neum64::zero();
            let mut edge_art4 = Neum64::zero();
            let mut edge_art2 = Neum64::zero();
            let mut edge_art8 = Neum64::zero();
            let mut edge_det = Neum64::zero();
            let mut edge_det4 = Neum64::zero();
            let mut edge_det2 = Neum64::zero();
            let mut edge_det8 = Neum64::zero();
            let mut hf_sq_src = Neum64::zero();
            let mut hf_sq_dst = Neum64::zero();
            let mut hf_abs_src = Neum64::zero();
            let mut hf_abs_dst = Neum64::zero();
            let mut mse = Neum64::zero();
            for x in 0..width {
                let (mu1, mu2, _, _, _) = win.at64(x, y, &planes);
                let s = src[base + x] as f64;
                let d = dst[base + x] as f64;
                if store_mu {
                    mu1_out[base + x] = mu1 as f32;
                    mu2_out[base + x] = mu2 as f32;
                }
                let diff1 = (s - mu1).abs();
                let diff2 = (d - mu2).abs();
                let ed = (1.0f64 + diff2) / (1.0f64 + diff1) - 1.0f64;
                let artifact = ed.max(0.0f64);
                let detail_lost = (-ed).max(0.0f64);
                let a2 = artifact * artifact;
                let dl2 = detail_lost * detail_lost;
                let a4 = a2 * a2;
                let dl4 = dl2 * dl2;
                edge_art.add64(x, artifact);
                edge_art4.add64(x, a4);
                edge_art2.add64(x, a2);
                edge_det.add64(x, detail_lost);
                edge_det4.add64(x, dl4);
                edge_det2.add64(x, dl2);
                edge_art8.add64(x, a4 * a4);
                edge_det8.add64(x, dl4 * dl4);
                acc.edge_art_max = acc.edge_art_max.max(artifact as f32);
                acc.edge_det_max = acc.edge_det_max.max(detail_lost as f32);
                let vs = s - mu1;
                let vd = d - mu2;
                hf_sq_src.add64(x, vs * vs);
                hf_sq_dst.add64(x, vd * vd);
                hf_abs_src.add64(x, diff1);
                hf_abs_dst.add64(x, diff2);
                let pd = s - d;
                mse.add64(x, pd * pd);
            }
            acc.edge_art += edge_art.fin();
            acc.edge_art4 += edge_art4.fin();
            acc.edge_art2 += edge_art2.fin();
            acc.edge_det += edge_det.fin();
            acc.edge_det4 += edge_det4.fin();
            acc.edge_det2 += edge_det2.fin();
            acc.edge_art8 += edge_art8.fin();
            acc.edge_det8 += edge_det8.fin();
            acc.hf_sq_src += hf_sq_src.fin();
            acc.hf_sq_dst += hf_sq_dst.fin();
            acc.hf_abs_src += hf_abs_src.fin();
            acc.hf_abs_dst += hf_abs_dst.fin();
            acc.mse += mse.fin();
        }
        win.slide(y, &planes);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fused extension (`ExtPoolsWork`) must form the SAME per-pixel
    /// values as the three separate passes it replaces; only the f64
    /// summation order may differ. Build the H planes, run the sweep with
    /// the extension on, then rebuild every pool the old way from the
    /// sweep's own stored `sd`/`mu` planes and the old activity chain, and
    /// demand agreement to f64 round-off.
    #[test]
    fn fused_extension_pools_match_the_separate_passes() {
        let (w, h, r) = (200usize, 70usize, 5usize);
        let n = w * h;
        let src: Vec<f32> = (0..n)
            .map(|i| ((i * 7919 + 13) % 977) as f32 / 977.0 * 0.6 + 0.1)
            .collect();
        let dst: Vec<f32> = src
            .iter()
            .enumerate()
            .map(|(i, &v)| (v + (((i * 31) % 17) as f32 - 8.0) * 0.004).clamp(0.0, 1.0))
            .collect();
        let (mut m1, mut m2, mut sq, mut pr) = (
            vec![0.0f32; n],
            vec![0.0f32; n],
            vec![0.0f32; n],
            vec![0.0f32; n],
        );
        crate::blur::fused_blur_h_ssim(&src, &dst, &mut m1, &mut m2, &mut sq, &mut pr, w, h, r);
        let mut act_raw = vec![0.0f32; n];
        crate::simd_ops::abs_diff_into(&src, &m1, &mut act_raw);
        let mut h_act = vec![0.0f32; n];
        crate::blur::box_blur_h(&act_raw, &mut h_act, w, h, r);
        let (inner_start, inner_h) = (8usize, 50usize);
        let (k_mask, k_iw) = (0.7f32, 0.35f32);
        let ext = ExtPoolsWork {
            on: true,
            mask: true,
            iw: true,
            k_mask,
            k_iw,
        };
        let (mut mu1_v, mut mu2_v, mut sd_v) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
        let acc = fused_vblur_features_ssim(
            &m1,
            &m2,
            &sq,
            &pr,
            &src,
            &dst,
            w,
            h,
            inner_start,
            inner_h,
            r,
            &mut mu1_v,
            &mut mu2_v,
            true,
            &mut sd_v,
            true,
            &mut [],
            &mut [],
            false,
            FreeExtrasWork::default(),
            ext,
            &h_act,
        );
        let mut activity = vec![0.0f32; n];
        let mut tmp = vec![0.0f32; n];
        crate::blur::box_blur_1pass_into(&act_raw, &mut activity, &mut tmp, w, h, r);
        let inner = inner_start * w..(inner_start + inner_h) * w;
        let ((sd_m, sd4_m, sd2_m), (sd_i, sd4_i, sd2_i)) = crate::simd_ops::ssim_signal_inline_both(
            &sd_v[inner.clone()],
            &activity[inner.clone()],
            k_mask,
            k_iw,
        );
        let ((art4_m, det4_m), (art4_i, det4_i)) = crate::simd_ops::edge_diff_channel_inline_both(
            &src[inner.clone()],
            &dst[inner.clone()],
            &mu1_v[inner.clone()],
            &mu2_v[inner.clone()],
            &activity[inner.clone()],
            k_mask,
            k_iw,
        );
        let (mse_m, mse_i) = crate::simd_ops::build_inline_mse(
            &activity[inner.clone()],
            k_mask,
            k_iw,
            &src[inner.clone()],
            &dst[inner.clone()],
        );
        let act_sum: f64 = activity[inner.clone()].iter().map(|&a| a as f64).sum();
        let pairs = [
            ("masked_ssim_d", acc.masked_ssim_d, sd_m),
            ("masked_ssim_d4", acc.masked_ssim_d4, sd4_m),
            ("masked_ssim_d2", acc.masked_ssim_d2, sd2_m),
            ("iw_ssim_d", acc.iw_ssim_d, sd_i),
            ("iw_ssim_d4", acc.iw_ssim_d4, sd4_i),
            ("iw_ssim_d2", acc.iw_ssim_d2, sd2_i),
            ("masked_art4", acc.masked_art4, art4_m),
            ("masked_det4", acc.masked_det4, det4_m),
            ("iw_art4", acc.iw_art4, art4_i),
            ("iw_det4", acc.iw_det4, det4_i),
            ("masked_mse", acc.masked_mse, mse_m),
            ("iw_mse", acc.iw_mse, mse_i),
            ("act_sum", acc.act_sum, act_sum),
        ];
        let mut worst = 0.0f64;
        for (name, fused, separate) in pairs {
            assert!(
                separate.is_finite() && separate != 0.0,
                "{name}: inert fixture ({separate})"
            );
            let rel = ((fused - separate) / separate).abs();
            worst = worst.max(rel);
            // The two orders differ at the f32 level, not only the f64 one:
            // both reduce 16-lane chunks in f32 before the f64 add, and at a
            // width that is not a multiple of 16 the separate pass's
            // row-major chunks straddle rows while the sweep's column groups
            // do not, so the chunks hold different pixels. Measured 1.7e-9 on
            // this 200-wide fixture; 1e-7 is a decade of margin over a 16-term
            // f32 partial sum, and a per-pixel arithmetic difference would be
            // orders of magnitude larger.
            assert!(
                rel <= 1e-7,
                "{name}: fused {fused:e} vs separate {separate:e} (rel {rel:e})"
            );
        }
        println!("FUSED-EXT-PARITY-RAN worst rel {worst:.3e} over 13 pools");
    }

    /// REV4VEC2 bit-exactness gate: the f64x8/f32x8 canon64 vblur bodies
    /// must reproduce the scalar `LanesF64` canonical body **bit for bit**
    /// — every `StripChannelAccum` field and every stored output plane.
    /// Each compiled tier variant is invoked directly on its summoned
    /// token; `mode = Canon64` pins the Rec64 window axis.
    mod rev4vec2 {
        use super::*;
        use crate::featcanon::LanesF64;
        use archmage::SimdToken as _;

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
            fn f32v(&mut self) -> f32 {
                if self.next().is_multiple_of(257) {
                    f32::from_bits(self.next() as u32)
                } else {
                    (self.next() % 4096) as f32 / 128.0 - 16.0
                }
            }
        }

        fn acc_bits(a: &StripChannelAccum) -> Vec<u64> {
            let f = [
                a.ssim_d,
                a.ssim_d4,
                a.ssim_d2,
                a.edge_art,
                a.edge_art4,
                a.edge_art2,
                a.edge_det,
                a.edge_det4,
                a.edge_det2,
                a.mse,
                a.hf_sq_src,
                a.hf_sq_dst,
                a.hf_abs_src,
                a.hf_abs_dst,
                a.ssim_d8,
                a.edge_art8,
                a.edge_det8,
                a.sum_s,
                a.sum_d,
                a.sum_s2,
                a.sum_d2,
                a.sum_dd,
                a.sum_ds,
                a.sum_msat,
                a.lum_wd_num,
                a.lum_wd_den,
                a.lum_wb_num,
                a.lum_wb_den,
                a.masked_ssim_d,
                a.masked_ssim_d4,
                a.masked_ssim_d2,
                a.iw_ssim_d,
                a.iw_ssim_d4,
                a.iw_ssim_d2,
                a.masked_art4,
                a.masked_det4,
                a.iw_art4,
                a.iw_det4,
                a.masked_mse,
                a.iw_mse,
                a.act_sum,
            ];
            let mut v: Vec<u64> = f.iter().map(|x| x.to_bits()).collect();
            v.push(a.ssim_max.to_bits() as u64);
            v.push(a.edge_art_max.to_bits() as u64);
            v.push(a.edge_det_max.to_bits() as u64);
            v
        }

        struct SsimCase {
            w: usize,
            h: usize,
            r: usize,
            inner_start: usize,
            inner_h: usize,
            store_mu: bool,
            store_sd: bool,
            store_sigma: bool,
            free: FreeExtrasWork,
            direct: bool,
            ext: ExtPoolsWork,
        }

        fn ssim_cases() -> Vec<SsimCase> {
            let free_all = FreeExtrasWork {
                revision: None,
                local_only: false,
                omit_peaks: false,
                omit_edges: false,
                raw_moments: true,
                bounded_err: true,
                lum_bins: true,
            };
            let free_none = FreeExtrasWork {
                revision: None,
                local_only: true,
                omit_peaks: false,
                omit_edges: true,
                raw_moments: false,
                bounded_err: false,
                lum_bins: false,
            };
            let free_mix = FreeExtrasWork {
                revision: None,
                local_only: false,
                omit_peaks: false,
                omit_edges: false,
                raw_moments: false,
                bounded_err: true,
                lum_bins: false,
            };
            let ext_off = ExtPoolsWork {
                on: false,
                mask: false,
                iw: false,
                k_mask: 0.0,
                k_iw: 0.0,
            };
            let ext_mask = ExtPoolsWork {
                on: true,
                mask: true,
                iw: false,
                k_mask: 0.5,
                k_iw: 0.25,
            };
            let ext_iw = ExtPoolsWork {
                on: true,
                mask: false,
                iw: true,
                k_mask: 0.5,
                k_iw: 0.25,
            };
            let ext_all = ExtPoolsWork {
                on: true,
                mask: true,
                iw: true,
                k_mask: 0.75,
                k_iw: 0.125,
            };
            vec![
                SsimCase {
                    w: 8,
                    h: 13,
                    r: 2,
                    inner_start: 0,
                    inner_h: 13,
                    store_mu: true,
                    store_sd: true,
                    store_sigma: true,
                    free: free_all,
                    direct: false,
                    ext: ext_all,
                },
                SsimCase {
                    w: 9,
                    h: 9,
                    r: 1,
                    inner_start: 0,
                    inner_h: 9,
                    store_mu: false,
                    store_sd: false,
                    store_sigma: false,
                    free: free_none,
                    direct: true,
                    ext: ext_off,
                },
                SsimCase {
                    w: 17,
                    h: 11,
                    r: 4,
                    inner_start: 1,
                    inner_h: 9,
                    store_mu: true,
                    store_sd: false,
                    store_sigma: true,
                    free: free_mix,
                    direct: false,
                    ext: ext_mask,
                },
                SsimCase {
                    w: 5,
                    h: 8,
                    r: 1,
                    inner_start: 2,
                    inner_h: 3,
                    store_mu: false,
                    store_sd: true,
                    store_sigma: false,
                    free: free_all,
                    direct: true,
                    ext: ext_iw,
                },
                SsimCase {
                    w: 16,
                    h: 6,
                    r: 0,
                    inner_start: 0,
                    inner_h: 6,
                    store_mu: true,
                    store_sd: true,
                    store_sigma: true,
                    free: free_mix,
                    direct: false,
                    ext: ext_iw,
                },
                SsimCase {
                    w: 24,
                    h: 4,
                    r: 3,
                    inner_start: 0,
                    inner_h: 2,
                    store_mu: false,
                    store_sd: false,
                    store_sigma: false,
                    free: free_none,
                    direct: false,
                    ext: ext_off,
                },
            ]
        }

        #[allow(clippy::too_many_arguments)]
        fn run_ssim_scalar(
            c: &SsimCase,
            planes: &[Vec<f32>; 6],
        ) -> (StripChannelAccum, [Vec<f32>; 5]) {
            let n = c.w * c.h;
            let mut out: [Vec<f32>; 5] = std::array::from_fn(|_| vec![f32::NAN; n]);
            let (o0, rest) = out.split_at_mut(1);
            let (o1, rest) = rest.split_at_mut(1);
            let (o2, rest) = rest.split_at_mut(1);
            let (o3, o4) = rest.split_at_mut(1);
            let acc = fused_vblur_ssim_canon::<LanesF64>(
                &planes[0],
                &planes[1],
                &planes[2],
                &planes[3],
                &planes[4],
                &planes[5],
                c.w,
                c.h,
                c.inner_start,
                c.inner_h,
                c.r,
                &mut o0[0],
                &mut o1[0],
                c.store_mu,
                &mut o2[0],
                c.store_sd,
                &mut o3[0],
                &mut o4[0],
                c.store_sigma,
                c.free,
                c.direct,
                c.ext,
                &planes[4],
                crate::featcanon::Mode::Canon64,
            );
            (acc, out)
        }

        #[allow(clippy::too_many_arguments)]
        fn run_ssim_vec(
            f: impl Fn(
                &[f32],
                &[f32],
                &[f32],
                &[f32],
                &[f32],
                &[f32],
                &mut [f32],
                &mut [f32],
                &mut [f32],
                &mut [f32],
                &mut [f32],
            ) -> StripChannelAccum,
            c: &SsimCase,
            planes: &[Vec<f32>; 6],
        ) -> (StripChannelAccum, [Vec<f32>; 5]) {
            let n = c.w * c.h;
            let mut out: [Vec<f32>; 5] = std::array::from_fn(|_| vec![f32::NAN; n]);
            let (o0, rest) = out.split_at_mut(1);
            let (o1, rest) = rest.split_at_mut(1);
            let (o2, rest) = rest.split_at_mut(1);
            let (o3, o4) = rest.split_at_mut(1);
            let acc = f(
                &planes[0], &planes[1], &planes[2], &planes[3], &planes[4], &planes[5], &mut o0[0],
                &mut o1[0], &mut o2[0], &mut o3[0], &mut o4[0],
            );
            (acc, out)
        }

        #[test]
        fn vblur_ssim_canon64_vec_matches_scalar_on_every_compiled_tier() {
            let mut rng = Rng(0xB529_7A4D_0C58_59A1);
            let mut compiled = 0usize;
            macro_rules! check_variant {
                ($name:literal, $call:expr) => {{
                    compiled += 1;
                    for c in ssim_cases() {
                        let n = c.w * c.h;
                        let planes: [Vec<f32>; 6] =
                            std::array::from_fn(|_| (0..n).map(|_| rng.f32v()).collect());
                        let (want_acc, want_out) = run_ssim_scalar(&c, &planes);
                        let call = &$call;
                        let (got_acc, got_out) = run_ssim_vec(
                            |m1, m2, sq, s12, src, dst, mo1, mo2, so, sqo, s12o| {
                                call(m1, m2, sq, s12, src, dst, mo1, mo2, so, sqo, s12o, &c)
                            },
                            &c,
                            &planes,
                        );
                        assert_eq!(
                            acc_bits(&got_acc),
                            acc_bits(&want_acc),
                            "{}: accum differs (w={} h={} r={} i0={} ih={} direct={})",
                            $name,
                            c.w,
                            c.h,
                            c.r,
                            c.inner_start,
                            c.inner_h,
                            c.direct
                        );
                        for (p, (g, e)) in got_out.iter().zip(&want_out).enumerate() {
                            for i in 0..n {
                                assert_eq!(
                                    g[i].to_bits(),
                                    e[i].to_bits(),
                                    "{}: out plane {} differs at {i} (w={} h={})",
                                    $name,
                                    p,
                                    c.w,
                                    c.h
                                );
                            }
                        }
                    }
                }};
            }
            type SsimFn = dyn Fn(
                &[f32],
                &[f32],
                &[f32],
                &[f32],
                &[f32],
                &[f32],
                &mut [f32],
                &mut [f32],
                &mut [f32],
                &mut [f32],
                &mut [f32],
                &SsimCase,
            ) -> StripChannelAccum;
            let scalar_t = archmage::ScalarToken::summon().expect("infallible");
            let scalar_f: Box<SsimFn> = Box::new(
                move |m1, m2, sq, s12, src, dst, mo1, mo2, so, sqo, s12o, c: &SsimCase| {
                    fused_vblur_ssim_canon64_vec_scalar::<crate::featcanon::LanesF64>(
                        scalar_t,
                        m1,
                        m2,
                        sq,
                        s12,
                        src,
                        dst,
                        c.w,
                        c.h,
                        c.inner_start,
                        c.inner_h,
                        c.r,
                        mo1,
                        mo2,
                        c.store_mu,
                        so,
                        c.store_sd,
                        sqo,
                        s12o,
                        c.store_sigma,
                        c.free,
                        c.direct,
                        c.ext,
                        src,
                        crate::featcanon::Mode::Canon64,
                    )
                },
            );
            check_variant!("scalar", scalar_f);
            #[cfg(target_arch = "x86_64")]
            if let Some(t) = archmage::X64V3Token::summon() {
                let f: Box<SsimFn> = Box::new(
                    move |m1, m2, sq, s12, src, dst, mo1, mo2, so, sqo, s12o, c: &SsimCase| {
                        fused_vblur_ssim_canon64_vec_v3::<crate::featcanon::LanesF64>(
                            t,
                            m1,
                            m2,
                            sq,
                            s12,
                            src,
                            dst,
                            c.w,
                            c.h,
                            c.inner_start,
                            c.inner_h,
                            c.r,
                            mo1,
                            mo2,
                            c.store_mu,
                            so,
                            c.store_sd,
                            sqo,
                            s12o,
                            c.store_sigma,
                            c.free,
                            c.direct,
                            c.ext,
                            src,
                            crate::featcanon::Mode::Canon64,
                        )
                    },
                );
                check_variant!("v3", f);
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            if let Some(t) = archmage::X64V4Token::summon() {
                let f: Box<SsimFn> = Box::new(
                    move |m1, m2, sq, s12, src, dst, mo1, mo2, so, sqo, s12o, c: &SsimCase| {
                        fused_vblur_ssim_canon64_vec_v4::<crate::featcanon::LanesF64>(
                            t,
                            m1,
                            m2,
                            sq,
                            s12,
                            src,
                            dst,
                            c.w,
                            c.h,
                            c.inner_start,
                            c.inner_h,
                            c.r,
                            mo1,
                            mo2,
                            c.store_mu,
                            so,
                            c.store_sd,
                            sqo,
                            s12o,
                            c.store_sigma,
                            c.free,
                            c.direct,
                            c.ext,
                            src,
                            crate::featcanon::Mode::Canon64,
                        )
                    },
                );
                check_variant!("v4", f);
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            if let Some(t) = archmage::X64V4xToken::summon() {
                let f: Box<SsimFn> = Box::new(
                    move |m1, m2, sq, s12, src, dst, mo1, mo2, so, sqo, s12o, c: &SsimCase| {
                        fused_vblur_ssim_canon64_vec_v4x::<crate::featcanon::LanesF64>(
                            t,
                            m1,
                            m2,
                            sq,
                            s12,
                            src,
                            dst,
                            c.w,
                            c.h,
                            c.inner_start,
                            c.inner_h,
                            c.r,
                            mo1,
                            mo2,
                            c.store_mu,
                            so,
                            c.store_sd,
                            sqo,
                            s12o,
                            c.store_sigma,
                            c.free,
                            c.direct,
                            c.ext,
                            src,
                            crate::featcanon::Mode::Canon64,
                        )
                    },
                );
                check_variant!("v4x", f);
            }
            assert!(compiled >= 1);
        }

        /// Revision 7's lean sixteen-column loop (the configuration B and Tga
        /// run: radius 0, direct SSIM, edges and HF on, no peaks, extensions,
        /// free extras or stores) against the generic chunk body: every
        /// compiled tier and the scalar body accumulate the same bits, over
        /// remainder columns, a single row and a width past the 4,096-column
        /// fold.
        #[test]
        fn vblur_ssim_rev7_lean_loop_matches_every_tier() {
            let mut rng = Rng(0x7E57_1EA2_5EED_0007);
            let free = FreeExtrasWork {
                revision: Some(crate::feature_defs::FormulaRevision::Rev7),
                local_only: false,
                omit_peaks: true,
                omit_edges: false,
                raw_moments: false,
                bounded_err: false,
                lum_bins: false,
            };
            let ext = ExtPoolsWork::default();
            type P = crate::featcanon::Lanes16F64;
            for (w, h, i0, ih) in [
                (16usize, 1usize, 0usize, 1usize),
                (37, 5, 1, 3),
                (64, 9, 0, 9),
                (255, 4, 1, 2),
                (4133, 3, 0, 3),
            ] {
                let n = w * h;
                let planes: [Vec<f32>; 6] = std::array::from_fn(|_| {
                    (0..n)
                        .map(|_| (rng.next() % 4096) as f32 / 4096.0)
                        .collect()
                });
                let [m1, m2, sq, s12, src, dst] = &planes;
                macro_rules! run {
                    ($f:path, $t:expr) => {{
                        let (mut a, mut b, mut c, mut d, mut e) =
                            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
                        acc_bits(&$f(
                            $t,
                            m1,
                            m2,
                            sq,
                            s12,
                            src,
                            dst,
                            w,
                            h,
                            i0,
                            ih,
                            0,
                            &mut a,
                            &mut b,
                            false,
                            &mut c,
                            false,
                            &mut d,
                            &mut e,
                            false,
                            free,
                            true,
                            ext,
                            &[],
                            crate::featcanon::Mode::Canon64,
                        ))
                    }};
                }
                let want = run!(
                    fused_vblur_ssim_canon64_vec_scalar::<P>,
                    archmage::ScalarToken::summon().expect("infallible")
                );
                assert!(want.iter().any(|&b| b != 0), "vacuous accumulators");
                #[cfg(target_arch = "x86_64")]
                if let Some(t) = archmage::X64V3Token::summon() {
                    assert_eq!(
                        run!(fused_vblur_ssim_canon64_vec_v3::<P>, t),
                        want,
                        "v3 {w}x{h}"
                    );
                }
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                if let Some(t) = archmage::X64V4Token::summon() {
                    assert_eq!(
                        run!(fused_vblur_ssim_canon64_vec_v4::<P>, t),
                        want,
                        "v4 {w}x{h}"
                    );
                }
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                if let Some(t) = archmage::X64V4xToken::summon() {
                    assert_eq!(
                        run!(fused_vblur_ssim_canon64_vec_v4x::<P>, t),
                        want,
                        "v4x {w}x{h}"
                    );
                }
            }
        }

        struct EdgeCase {
            w: usize,
            h: usize,
            r: usize,
            inner_start: usize,
            inner_h: usize,
            store_mu: bool,
        }

        #[test]
        fn vblur_edge_canon64_vec_matches_scalar_on_every_compiled_tier() {
            let cases = [
                EdgeCase {
                    w: 8,
                    h: 13,
                    r: 2,
                    inner_start: 0,
                    inner_h: 13,
                    store_mu: true,
                },
                EdgeCase {
                    w: 9,
                    h: 9,
                    r: 1,
                    inner_start: 0,
                    inner_h: 9,
                    store_mu: false,
                },
                EdgeCase {
                    w: 17,
                    h: 11,
                    r: 4,
                    inner_start: 1,
                    inner_h: 9,
                    store_mu: true,
                },
                EdgeCase {
                    w: 5,
                    h: 8,
                    r: 1,
                    inner_start: 2,
                    inner_h: 3,
                    store_mu: false,
                },
                EdgeCase {
                    w: 16,
                    h: 6,
                    r: 0,
                    inner_start: 0,
                    inner_h: 6,
                    store_mu: true,
                },
            ];
            let mut rng = Rng(0x1B2C_3D4E_5F60_7788);
            let mut compiled = 0usize;
            macro_rules! check_variant {
                ($name:literal, $call:expr) => {{
                    compiled += 1;
                    for c in &cases {
                        let n = c.w * c.h;
                        let m1: Vec<f32> = (0..n).map(|_| rng.f32v()).collect();
                        let m2: Vec<f32> = (0..n).map(|_| rng.f32v()).collect();
                        let src: Vec<f32> = (0..n).map(|_| rng.f32v()).collect();
                        let dst: Vec<f32> = (0..n).map(|_| rng.f32v()).collect();
                        let mut wo1 = vec![f32::NAN; n];
                        let mut wo2 = vec![f32::NAN; n];
                        let want = fused_vblur_edge_canon::<LanesF64>(
                            &m1,
                            &m2,
                            &src,
                            &dst,
                            c.w,
                            c.h,
                            c.inner_start,
                            c.inner_h,
                            c.r,
                            &mut wo1,
                            &mut wo2,
                            c.store_mu,
                            crate::featcanon::Mode::Canon64,
                        );
                        let mut go1 = vec![f32::NAN; n];
                        let mut go2 = vec![f32::NAN; n];
                        let got = $call(&m1, &m2, &src, &dst, &mut go1, &mut go2, c);
                        assert_eq!(
                            acc_bits(&got),
                            acc_bits(&want),
                            "{}: edge accum differs (w={} h={} r={} i0={} ih={})",
                            $name,
                            c.w,
                            c.h,
                            c.r,
                            c.inner_start,
                            c.inner_h
                        );
                        for i in 0..n {
                            assert_eq!(
                                go1[i].to_bits(),
                                wo1[i].to_bits(),
                                "{}: mu1 out {i}",
                                $name
                            );
                            assert_eq!(
                                go2[i].to_bits(),
                                wo2[i].to_bits(),
                                "{}: mu2 out {i}",
                                $name
                            );
                        }
                    }
                }};
            }
            check_variant!(
                "scalar",
                |m1: &[f32],
                 m2: &[f32],
                 src: &[f32],
                 dst: &[f32],
                 o1: &mut [f32],
                 o2: &mut [f32],
                 c: &EdgeCase| {
                    fused_vblur_edge_canon64_vec_scalar(
                        archmage::ScalarToken::summon().expect("infallible"),
                        m1,
                        m2,
                        src,
                        dst,
                        c.w,
                        c.h,
                        c.inner_start,
                        c.inner_h,
                        c.r,
                        o1,
                        o2,
                        c.store_mu,
                        crate::featcanon::Mode::Canon64,
                    )
                }
            );
            #[cfg(target_arch = "x86_64")]
            if let Some(t) = archmage::X64V3Token::summon() {
                check_variant!("v3", |m1: &[f32],
                                      m2: &[f32],
                                      src: &[f32],
                                      dst: &[f32],
                                      o1: &mut [f32],
                                      o2: &mut [f32],
                                      c: &EdgeCase| {
                    fused_vblur_edge_canon64_vec_v3(
                        t,
                        m1,
                        m2,
                        src,
                        dst,
                        c.w,
                        c.h,
                        c.inner_start,
                        c.inner_h,
                        c.r,
                        o1,
                        o2,
                        c.store_mu,
                        crate::featcanon::Mode::Canon64,
                    )
                });
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            if let Some(t) = archmage::X64V4Token::summon() {
                check_variant!("v4", |m1: &[f32],
                                      m2: &[f32],
                                      src: &[f32],
                                      dst: &[f32],
                                      o1: &mut [f32],
                                      o2: &mut [f32],
                                      c: &EdgeCase| {
                    fused_vblur_edge_canon64_vec_v4(
                        t,
                        m1,
                        m2,
                        src,
                        dst,
                        c.w,
                        c.h,
                        c.inner_start,
                        c.inner_h,
                        c.r,
                        o1,
                        o2,
                        c.store_mu,
                        crate::featcanon::Mode::Canon64,
                    )
                });
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            if let Some(t) = archmage::X64V4xToken::summon() {
                check_variant!("v4x", |m1: &[f32],
                                       m2: &[f32],
                                       src: &[f32],
                                       dst: &[f32],
                                       o1: &mut [f32],
                                       o2: &mut [f32],
                                       c: &EdgeCase| {
                    fused_vblur_edge_canon64_vec_v4x(
                        t,
                        m1,
                        m2,
                        src,
                        dst,
                        c.w,
                        c.h,
                        c.inner_start,
                        c.inner_h,
                        c.r,
                        o1,
                        o2,
                        c.store_mu,
                        crate::featcanon::Mode::Canon64,
                    )
                });
            }
            assert!(compiled >= 1);
        }
    }
}
