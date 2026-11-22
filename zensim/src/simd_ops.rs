//! SIMD-accelerated element-wise operations for SSIM computation.

use crate::ssim_form::{ssim_dissim_raw_scalar, ssim_dissim8, ssim_dissim16};
#[cfg(target_arch = "x86_64")]
use archmage::arcane;
use archmage::incant;
use archmage::magetypes;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::f32x8;
use magetypes::simd::generic::f32x8 as GenericF32x8;
use magetypes::simd::generic::f32x16;

/// Element-wise multiply: out[i] = a[i] * b[i]
pub fn mul_into(a: &[f32], b: &[f32], out: &mut [f32]) {
    incant!(mul_into_inner(a, b, out), [v4, v3, neon, wasm128, scalar]);
}

/// Element-wise: out[i] = a[i]*a[i] + b[i]*b[i] (sum of squares)
pub fn sq_sum_into(a: &[f32], b: &[f32], out: &mut [f32]) {
    incant!(
        sq_sum_into_inner(a, b, out),
        [v4, v3, neon, wasm128, scalar]
    );
}

/// Compute sum of squared differences: sum((a[i] - b[i])²)
pub fn sq_diff_sum(a: &[f32], b: &[f32]) -> f64 {
    // aarch64: use the SCALAR variant. On AArch64, NEON is baseline, so LLVM
    // autovectorises the scalar body anyway — the hand-written NEON variant is
    // competing with the autovectoriser, not with scalar code, and it LOSES.
    // Measured on Apple M4 Pro at 1024x1024 (zensim-bench/benches/stage_isolation.rs,
    // within-group A/B): NEON 0.351ms vs scalar 0.117ms — the NEON variant is 3x SLOWER.
    // Verified BIT-IDENTICAL to the NEON variant before switching (differing
    // lanes = 0), so this is a pure speed change with no effect on any score.
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if archmage::NeonToken::summon().is_some() {
            return sq_diff_sum_inner_scalar(
                archmage::ScalarToken::summon().expect("infallible"),
                a,
                b,
            );
        }
    }
    incant!(sq_diff_sum_inner(a, b), [v4, v3, neon, wasm128, scalar])
}

/// Compute sum of absolute differences: sum(|a[i] - b[i]|)
pub fn abs_diff_sum(a: &[f32], b: &[f32]) -> f64 {
    // aarch64: use the SCALAR variant. On AArch64, NEON is baseline, so LLVM
    // autovectorises the scalar body anyway — the hand-written NEON variant is
    // competing with the autovectoriser, not with scalar code, and it LOSES.
    // Measured on Apple M4 Pro at 1024x1024 (zensim-bench/benches/stage_isolation.rs,
    // within-group A/B): NEON 0.351ms vs scalar 0.087ms — the NEON variant is 4x SLOWER.
    // Verified BIT-IDENTICAL to the NEON variant before switching (differing
    // lanes = 0), so this is a pure speed change with no effect on any score.
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if archmage::NeonToken::summon().is_some() {
            return abs_diff_sum_inner_scalar(
                archmage::ScalarToken::summon().expect("infallible"),
                a,
                b,
            );
        }
    }
    incant!(abs_diff_sum_inner(a, b), [v4, v3, neon, wasm128, scalar])
}

/// Element-wise absolute difference: out[i] = |a[i] - b[i]|
///
/// **Deprecated for the streaming hot path (Phase 2 Lever 3, 2026-05-22)**
/// — the activity path uses [`crate::blur::box_blur_h_into_abs_diff`]
/// which fuses the blur with the abs-diff. Kept as a reference
/// implementation for the SIMD dispatch trio (v4/v3/scalar).
pub(crate) fn abs_diff_into(a: &[f32], b: &[f32], out: &mut [f32]) {
    incant!(
        abs_diff_into_inner(a, b, out),
        [v4, v3, neon, wasm128, scalar]
    );
}

/// `out[y*width + x] = |src[y*width + x] - mu[y*pitch + x]|` for `rows` rows:
/// the activity map's first step, reading the H-only blurred source straight
/// out of the (possibly pitched) plane `fused_blur_h_ssim` wrote, instead of
/// re-blurring `src` (`box_blur_h_into_abs_diff`). Bit-identical to that entry
/// (both sides equal one scalar recurrence — `blur`'s
/// `h_entries_are_bit_exact_at_a_degenerate_last_column_tile`; asserted
/// directly by `activity_from_the_fused_h_plane_is_bit_identical`).
pub(crate) fn abs_diff_rows_into(
    src: &[f32],
    mu: &[f32],
    out: &mut [f32],
    width: usize,
    pitch: usize,
    rows: usize,
) {
    debug_assert!(pitch >= width);
    for y in 0..rows {
        abs_diff_into(
            &src[y * width..y * width + width],
            &mu[y * pitch..y * pitch + width],
            &mut out[y * width..y * width + width],
        );
    }
}

/// Like ssim_channel but also computes 8th-power pool and max.
/// Returns (sum_d, sum_d4, sum_d2, sum_d8, max_d).
/// d8 = d4*d4 (one extra multiply per pixel). L8 = (sum_d8/N)^(1/8).
pub fn ssim_channel_extended(
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    sigma12: &[f32],
) -> (f64, f64, f64, f64, f32) {
    incant!(
        ssim_channel_extended_inner(mu1, mu2, sum_sq, sigma12),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// Like edge_diff_channel but also computes 8th-power pool and max for artifact/detail.
/// Returns (art_mean, art_4th, det_mean, det_4th, art_2nd, det_2nd, art_8th, det_8th, max_art, max_det).
pub fn edge_diff_channel_extended(
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f32, f32) {
    incant!(
        edge_diff_extended_inner(img1, img2, mu1, mu2),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// Like ssim_channel but weights each pixel distance by mask[i] before accumulation.
/// Returns (sum_d, sum_d4, sum_d2) — mean, 4th-power, and 2nd-power pools.
///
/// **Deprecated for the streaming hot path (Phase 2, 2026-05-22)** — the
/// mask-plane variant is superseded by [`ssim_channel_inline_mask`]
/// which derives the mask weight inline from activity. Kept as a
/// reference implementation for the SIMD dispatch trio and unit-test
/// coverage; crate-internal.
#[allow(dead_code)]
pub(crate) fn ssim_channel_masked(
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    sigma12: &[f32],
    mask: &[f32],
) -> (f64, f64, f64) {
    incant!(
        ssim_channel_masked_inner(mu1, mu2, sum_sq, sigma12, mask),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// Fused 2-mask variant of [`ssim_channel_masked`]. Computes the
/// unweighted SSIM distance once per pixel and accumulates with two
/// different masks into two separate `(sum_d, sum_d4, sum_d2)`
/// tuples in a single pass.
///
/// **Deprecated for the masked + IW combined path (2026-05-22)** —
/// the streaming hot path now uses
/// [`ssim_channel_masked_with_iw_inline`] which derives the IW weight
/// from the activity slice inline (`1 + k_iw · a`) instead of reading
/// a materialized `iw_weight` plane. Kept for API stability and unit
/// test coverage; mathematically equivalent.
///
/// Returns `((sum_d_a, sum_d4_a, sum_d2_a), (sum_d_b, sum_d4_b, sum_d2_b))`
/// where `_a` corresponds to `mask_a` and `_b` to `mask_b`.
#[inline]
#[allow(dead_code)]
pub fn ssim_channel_masked_2(
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    sigma12: &[f32],
    mask_a: &[f32],
    mask_b: &[f32],
) -> ((f64, f64, f64), (f64, f64, f64)) {
    incant!(
        ssim_channel_masked_2_inner(mu1, mu2, sum_sq, sigma12, mask_a, mask_b),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// Like edge_diff_channel but weights each pixel distance by mask[i].
/// Returns (art_mean, art_4th, det_mean, det_4th, art_2nd, det_2nd).
///
/// **Deprecated for the streaming hot path (Phase 2, 2026-05-22)** — see
/// [`edge_diff_channel_inline_mask`] for the activity-inline replacement.
/// Kept as a reference implementation for the SIMD dispatch trio and
/// unit-test coverage; crate-internal.
#[allow(dead_code)]
pub(crate) fn edge_diff_channel_masked(
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    mask: &[f32],
) -> (f64, f64, f64, f64, f64, f64) {
    incant!(
        edge_diff_masked_inner(img1, img2, mu1, mu2, mask),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// Fused 2-mask variant of [`edge_diff_channel_masked`]. Computes the
/// raw edge response (`(1 + |img2 - mu2|) / (1 + |img1 - mu1|) - 1`)
/// once per pixel and accumulates with two different masks in a single
/// pass.
///
/// **Deprecated for the masked + IW combined path (2026-05-22)** —
/// see [`edge_diff_channel_masked_with_iw_inline`] for the
/// activity-inline replacement. Kept for API stability and unit test
/// coverage.
///
/// Returns `((art4_a, det4_a), (art4_b, det4_b))` — only the 4th-power
/// pools are produced because the streaming caller only consumes
/// `art4`/`det4` from the masked/IW edge path (the mean and 2nd-pool
/// edge features come from the unmasked extended pass).
#[allow(dead_code)]
pub fn edge_diff_channel_masked_2_art4_det4(
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    mask_a: &[f32],
    mask_b: &[f32],
) -> ((f64, f64), (f64, f64)) {
    incant!(
        edge_diff_masked_2_art4_det4_inner(img1, img2, mu1, mu2, mask_a, mask_b),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// Fused weight-build + MSE-pool for the masked + IW combined path.
/// Given the blurred-activity buffer, builds **both** the masked weight
/// (`1 / (1 + k_mask * a)`) and the IW weight (`1 + k_iw * a`) into
/// `mask_out` / `iw_out`, and accumulates `Σ (src-dst)² · mask` for
/// **both** masks in the same pass.
///
/// **Deprecated for the streaming hot path (2026-05-22)** —
/// [`build_mask_and_iw_mse_inline`] does the same MSE accumulation
/// but folds the IW weight inline (never writes the iw plane),
/// eliminating one full plane write + the downstream read. Kept for
/// API stability and unit test coverage.
///
/// Returns `(masked_mse_sum, iw_mse_sum)`.
#[allow(dead_code)]
pub fn build_weights_and_mse(
    activity: &[f32],
    k_mask: f32,
    k_iw: f32,
    src: &[f32],
    dst: &[f32],
    mask_out: &mut [f32],
    iw_out: &mut [f32],
) -> (f64, f64) {
    incant!(
        build_weights_and_mse_inner(activity, k_mask, k_iw, src, dst, mask_out, iw_out),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// Build only the masked weight (`1 / (1 + k_mask * a)`) and accumulate
/// `Σ (src-dst)² · mask`. Used when only `extended_features` is on.
///
/// **Deprecated for the streaming hot path (Phase 2, 2026-05-22)** —
/// [`build_inline_mask_mse`] performs the same accumulation WITHOUT
/// writing the mask plane. Kept for API stability and unit-test coverage.
#[allow(dead_code)]
pub fn build_mask_weight_and_mse(
    activity: &[f32],
    k_mask: f32,
    src: &[f32],
    dst: &[f32],
    mask_out: &mut [f32],
) -> f64 {
    incant!(
        build_mask_weight_and_mse_inner(activity, k_mask, src, dst, mask_out),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// Build only the IW weight (`1 + k_iw * a`) and accumulate
/// `Σ (src-dst)² · w_iw`.
///
/// **Deprecated for the streaming hot path (2026-05-22)** —
/// [`build_iw_mse_only`] does the same MSE accumulation without
/// writing the iw plane at all (the plane was the only output to be
/// thrown away later). Kept for API stability and unit test coverage.
#[allow(dead_code)]
pub fn build_iw_weight_and_mse(
    activity: &[f32],
    k_iw: f32,
    src: &[f32],
    dst: &[f32],
    iw_out: &mut [f32],
) -> f64 {
    incant!(
        build_iw_weight_and_mse_inner(activity, k_iw, src, dst, iw_out),
        [v4, v3, neon, wasm128, scalar]
    )
}

// --- SIMD implementations ---

#[cfg(target_arch = "x86_64")]
#[arcane]
fn sq_sum_into_inner_v4(token: archmage::X64V4Token, a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let chunks = n / 16;
    for c in 0..chunks {
        let base = c * 16;
        let va = f32x16::from_array_t(token, a[base..][..16].try_into().unwrap());
        let vb = f32x16::from_array_t(token, b[base..][..16].try_into().unwrap());
        out[base..base + 16].copy_from_slice(&va.mul_add_relaxed(va, vb * vb).to_array());
    }
    let v3 = token.v3();
    let chunks8 = (n - chunks * 16) / 8;
    for c in 0..chunks8 {
        let base = chunks * 16 + c * 8;
        let va = f32x8::from_array_t(v3, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array_t(v3, b[base..][..8].try_into().unwrap());
        out[base..base + 8].copy_from_slice(&va.mul_add_relaxed(va, vb * vb).to_array());
    }
    for i in (chunks * 16 + chunks8 * 8)..n {
        out[i] = a[i].mul_add(a[i], b[i] * b[i]);
    }
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn sq_sum_into_inner_v3(token: archmage::X64V3Token, a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let chunks = n / 8;
    for c in 0..chunks {
        let base = c * 8;
        let va = f32x8::from_array_t(token, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array_t(token, b[base..][..8].try_into().unwrap());
        out[base..base + 8].copy_from_slice(&va.mul_add_relaxed(va, vb * vb).to_array());
    }
    for i in (chunks * 8)..n {
        out[i] = a[i].mul_add(a[i], b[i] * b[i]);
    }
}

#[magetypes(neon, wasm128, scalar)]
fn sq_sum_into_inner(token: Token, a: &[f32], b: &[f32], out: &mut [f32]) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let (a_chunks, a_tail) = a.as_chunks::<8>();
    let (b_chunks, _) = b.as_chunks::<8>();
    let (out_chunks, out_tail) = out.as_chunks_mut::<8>();
    for ((ac, bc), oc) in a_chunks.iter().zip(b_chunks).zip(out_chunks) {
        let va = f32x8::from_array_t(token, *ac);
        let vb = f32x8::from_array_t(token, *bc);
        va.mul_add_relaxed(va, vb * vb).store(oc);
    }
    for ((&a, &b), o) in a_tail
        .iter()
        .zip(b.iter().skip(a_chunks.len() * 8))
        .zip(out_tail)
    {
        *o = a.mul_add(a, b * b);
    }
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn sq_diff_sum_inner_v4(token: archmage::X64V4Token, a: &[f32], b: &[f32]) -> f64 {
    let n = a.len();
    let chunks = n / 16;
    let mut sum = 0.0f64;
    for c in 0..chunks {
        let base = c * 16;
        let va = f32x16::from_array_t(token, a[base..][..16].try_into().unwrap());
        let vb = f32x16::from_array_t(token, b[base..][..16].try_into().unwrap());
        let d = va - vb;
        sum += (d * d).reduce_add() as f64;
    }
    let v3 = token.v3();
    let chunks8 = (n - chunks * 16) / 8;
    for c in 0..chunks8 {
        let base = chunks * 16 + c * 8;
        let va = f32x8::from_array_t(v3, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array_t(v3, b[base..][..8].try_into().unwrap());
        let d = va - vb;
        sum += (d * d).reduce_add() as f64;
    }
    for i in (chunks * 16 + chunks8 * 8)..n {
        let d = a[i] - b[i];
        sum += (d * d) as f64;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn sq_diff_sum_inner_v3(token: archmage::X64V3Token, a: &[f32], b: &[f32]) -> f64 {
    let n = a.len();
    let chunks = n / 8;
    let mut sum = 0.0f64;
    for c in 0..chunks {
        let base = c * 8;
        let va = f32x8::from_array_t(token, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array_t(token, b[base..][..8].try_into().unwrap());
        let d = va - vb;
        sum += (d * d).reduce_add() as f64;
    }
    for i in (chunks * 8)..n {
        let d = a[i] - b[i];
        sum += (d * d) as f64;
    }
    sum
}

#[magetypes(neon, wasm128, scalar)]
fn sq_diff_sum_inner(token: Token, a: &[f32], b: &[f32]) -> f64 {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let (a_chunks, a_tail) = a.as_chunks::<8>();
    let (b_chunks, _) = b.as_chunks::<8>();
    let mut sum = 0.0f64;
    for (ac, bc) in a_chunks.iter().zip(b_chunks) {
        let va = f32x8::from_array_t(token, *ac);
        let vb = f32x8::from_array_t(token, *bc);
        let d = va - vb;
        sum += (d * d).reduce_add() as f64;
    }
    for (&a, &b) in a_tail.iter().zip(b.iter().skip(a_chunks.len() * 8)) {
        let d = a - b;
        sum += (d * d) as f64;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn abs_diff_sum_inner_v4(token: archmage::X64V4Token, a: &[f32], b: &[f32]) -> f64 {
    let n = a.len();
    let chunks = n / 16;
    let mut sum = 0.0f64;
    for c in 0..chunks {
        let base = c * 16;
        let va = f32x16::from_array_t(token, a[base..][..16].try_into().unwrap());
        let vb = f32x16::from_array_t(token, b[base..][..16].try_into().unwrap());
        let d = (va - vb).abs();
        sum += d.reduce_add() as f64;
    }
    let v3 = token.v3();
    let chunks8 = (n - chunks * 16) / 8;
    for c in 0..chunks8 {
        let base = chunks * 16 + c * 8;
        let va = f32x8::from_array_t(v3, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array_t(v3, b[base..][..8].try_into().unwrap());
        let d = (va - vb).abs();
        sum += d.reduce_add() as f64;
    }
    for i in (chunks * 16 + chunks8 * 8)..n {
        let d = (a[i] - b[i]).abs() as f64;
        sum += d;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn abs_diff_sum_inner_v3(token: archmage::X64V3Token, a: &[f32], b: &[f32]) -> f64 {
    let n = a.len();
    let chunks = n / 8;
    let mut sum = 0.0f64;
    for c in 0..chunks {
        let base = c * 8;
        let va = f32x8::from_array_t(token, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array_t(token, b[base..][..8].try_into().unwrap());
        let d = (va - vb).abs();
        sum += d.reduce_add() as f64;
    }
    for i in (chunks * 8)..n {
        let d = (a[i] - b[i]).abs() as f64;
        sum += d;
    }
    sum
}

#[magetypes(neon, wasm128, scalar)]
fn abs_diff_sum_inner(token: Token, a: &[f32], b: &[f32]) -> f64 {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let (a_chunks, a_tail) = a.as_chunks::<8>();
    let (b_chunks, _) = b.as_chunks::<8>();
    let mut sum = 0.0f64;
    for (ac, bc) in a_chunks.iter().zip(b_chunks) {
        let va = f32x8::from_array_t(token, *ac);
        let vb = f32x8::from_array_t(token, *bc);
        let d = (va - vb).abs();
        sum += d.reduce_add() as f64;
    }
    for (&a, &b) in a_tail.iter().zip(b.iter().skip(a_chunks.len() * 8)) {
        let d = (a - b).abs() as f64;
        sum += d;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn mul_into_inner_v4(token: archmage::X64V4Token, a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let chunks = n / 16;
    for c in 0..chunks {
        let base = c * 16;
        let va = f32x16::from_array_t(token, a[base..][..16].try_into().unwrap());
        let vb = f32x16::from_array_t(token, b[base..][..16].try_into().unwrap());
        out[base..base + 16].copy_from_slice(&(va * vb).to_array());
    }
    let v3 = token.v3();
    let chunks8 = (n - chunks * 16) / 8;
    for c in 0..chunks8 {
        let base = chunks * 16 + c * 8;
        let va = f32x8::from_array_t(v3, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array_t(v3, b[base..][..8].try_into().unwrap());
        out[base..base + 8].copy_from_slice(&(va * vb).to_array());
    }
    for i in (chunks * 16 + chunks8 * 8)..n {
        out[i] = a[i] * b[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn mul_into_inner_v3(token: archmage::X64V3Token, a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let chunks = n / 8;
    for c in 0..chunks {
        let base = c * 8;
        let va = f32x8::from_array_t(token, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array_t(token, b[base..][..8].try_into().unwrap());
        out[base..base + 8].copy_from_slice(&(va * vb).to_array());
    }
    for i in (chunks * 8)..n {
        out[i] = a[i] * b[i];
    }
}

#[magetypes(neon, wasm128, scalar)]
fn mul_into_inner(token: Token, a: &[f32], b: &[f32], out: &mut [f32]) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let (a_chunks, a_tail) = a.as_chunks::<8>();
    let (b_chunks, _) = b.as_chunks::<8>();
    let (out_chunks, out_tail) = out.as_chunks_mut::<8>();
    for ((ac, bc), oc) in a_chunks.iter().zip(b_chunks).zip(out_chunks) {
        let va = f32x8::from_array_t(token, *ac);
        let vb = f32x8::from_array_t(token, *bc);
        (va * vb).store(oc);
    }
    for ((&a, &b), o) in a_tail
        .iter()
        .zip(b.iter().skip(a_chunks.len() * 8))
        .zip(out_tail)
    {
        *o = a * b;
    }
}

// === abs_diff_into ===

#[cfg(target_arch = "x86_64")]
#[arcane]
fn abs_diff_into_inner_v4(token: archmage::X64V4Token, a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let chunks = n / 16;
    for c in 0..chunks {
        let base = c * 16;
        let va = f32x16::from_array_t(token, a[base..][..16].try_into().unwrap());
        let vb = f32x16::from_array_t(token, b[base..][..16].try_into().unwrap());
        out[base..base + 16].copy_from_slice(&(va - vb).abs().to_array());
    }
    let v3 = token.v3();
    let chunks8 = (n - chunks * 16) / 8;
    for c in 0..chunks8 {
        let base = chunks * 16 + c * 8;
        let va = f32x8::from_array_t(v3, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array_t(v3, b[base..][..8].try_into().unwrap());
        out[base..base + 8].copy_from_slice(&(va - vb).abs().to_array());
    }
    for i in (chunks * 16 + chunks8 * 8)..n {
        out[i] = (a[i] - b[i]).abs();
    }
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn abs_diff_into_inner_v3(token: archmage::X64V3Token, a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let chunks = n / 8;
    for c in 0..chunks {
        let base = c * 8;
        let va = f32x8::from_array_t(token, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array_t(token, b[base..][..8].try_into().unwrap());
        out[base..base + 8].copy_from_slice(&(va - vb).abs().to_array());
    }
    for i in (chunks * 8)..n {
        out[i] = (a[i] - b[i]).abs();
    }
}

#[magetypes(neon, wasm128, scalar)]
fn abs_diff_into_inner(token: Token, a: &[f32], b: &[f32], out: &mut [f32]) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let (a_chunks, a_tail) = a.as_chunks::<8>();
    let (b_chunks, _) = b.as_chunks::<8>();
    let (out_chunks, out_tail) = out.as_chunks_mut::<8>();
    for ((ac, bc), oc) in a_chunks.iter().zip(b_chunks).zip(out_chunks) {
        let va = f32x8::from_array_t(token, *ac);
        let vb = f32x8::from_array_t(token, *bc);
        (va - vb).abs().store(oc);
    }
    for ((&a, &b), o) in a_tail
        .iter()
        .zip(b.iter().skip(a_chunks.len() * 8))
        .zip(out_tail)
    {
        *o = (a - b).abs();
    }
}

// === ssim_channel_masked: returns (sum_d, sum_d4, sum_d2) ===

#[cfg(target_arch = "x86_64")]
#[arcane]
fn ssim_channel_masked_inner_v4(
    token: archmage::X64V4Token,
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
    mask: &[f32],
) -> (f64, f64, f64) {
    let form = crate::ssim_form::active_luma_form();
    let zero = f32x16::zero_t(token);

    let n = mu1.len();
    let chunks = n / 16;
    let mut sum_d = 0.0f64;
    let mut sum_d4 = 0.0f64;
    let mut sum_d2 = 0.0f64;

    for c in 0..chunks {
        let base = c * 16;
        let m1 = f32x16::from_array_t(token, mu1[base..][..16].try_into().unwrap());
        let m2 = f32x16::from_array_t(token, mu2[base..][..16].try_into().unwrap());
        let ssq = f32x16::from_array_t(token, sum_sq[base..][..16].try_into().unwrap());
        let s12v = f32x16::from_array_t(token, s12[base..][..16].try_into().unwrap());
        let mv = f32x16::from_array_t(token, mask[base..][..16].try_into().unwrap());

        let d = ((ssim_dissim16(token, form, m1, m2, ssq, s12v)) * mv).max(zero);
        let d2 = d * d;
        let d4 = d2 * d2;

        sum_d += d.reduce_add() as f64;
        sum_d2 += d2.reduce_add() as f64;
        sum_d4 += d4.reduce_add() as f64;
    }

    for i in (chunks * 16)..n {
        let d = ((ssim_dissim_raw_scalar(form, mu1[i], mu2[i], sum_sq[i], s12[i])) * mask[i])
            .max(0.0f32);
        let d2 = d * d;
        sum_d += d as f64;
        sum_d2 += d2 as f64;
        sum_d4 += (d2 * d2) as f64;
    }

    (sum_d, sum_d4, sum_d2)
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn ssim_channel_masked_inner_v3(
    token: archmage::X64V3Token,
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
    mask: &[f32],
) -> (f64, f64, f64) {
    let form = crate::ssim_form::active_luma_form();
    let zero = f32x8::zero_t(token);

    let n = mu1.len();
    let chunks = n / 8;
    let mut sum_d = 0.0f64;
    let mut sum_d4 = 0.0f64;
    let mut sum_d2 = 0.0f64;

    for c in 0..chunks {
        let base = c * 8;
        let m1 = f32x8::from_array_t(token, mu1[base..][..8].try_into().unwrap());
        let m2 = f32x8::from_array_t(token, mu2[base..][..8].try_into().unwrap());
        let ssq = f32x8::from_array_t(token, sum_sq[base..][..8].try_into().unwrap());
        let s12v = f32x8::from_array_t(token, s12[base..][..8].try_into().unwrap());
        let mv = f32x8::from_array_t(token, mask[base..][..8].try_into().unwrap());

        let d = ((ssim_dissim8(token, form, m1, m2, ssq, s12v)) * mv).max(zero);
        let d2 = d * d;
        let d4 = d2 * d2;

        sum_d += d.reduce_add() as f64;
        sum_d2 += d2.reduce_add() as f64;
        sum_d4 += d4.reduce_add() as f64;
    }

    for i in (chunks * 8)..n {
        let d = ((ssim_dissim_raw_scalar(form, mu1[i], mu2[i], sum_sq[i], s12[i])) * mask[i])
            .max(0.0f32);
        let d2 = d * d;
        sum_d += d as f64;
        sum_d2 += d2 as f64;
        sum_d4 += (d2 * d2) as f64;
    }

    (sum_d, sum_d4, sum_d2)
}

#[magetypes(neon, wasm128, scalar)]
fn ssim_channel_masked_inner(
    token: Token,
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
    mask: &[f32],
) -> (f64, f64, f64) {
    let form = crate::ssim_form::active_luma_form();
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let zero = f32x8::zero_t(token);

    let (mu1_chunks, mu1_tail) = mu1.as_chunks::<8>();
    let (mu2_chunks, _) = mu2.as_chunks::<8>();
    let (ssq_chunks, _) = sum_sq.as_chunks::<8>();
    let (s12_chunks, _) = s12.as_chunks::<8>();
    let (mask_chunks, _) = mask.as_chunks::<8>();

    let mut sum_d = 0.0f64;
    let mut sum_d4 = 0.0f64;
    let mut sum_d2 = 0.0f64;

    for ((((m1c, m2c), ssqc), s12c), mc) in mu1_chunks
        .iter()
        .zip(mu2_chunks)
        .zip(ssq_chunks)
        .zip(s12_chunks)
        .zip(mask_chunks)
    {
        let m1 = f32x8::from_array_t(token, *m1c);
        let m2 = f32x8::from_array_t(token, *m2c);
        let ssq = f32x8::from_array_t(token, *ssqc);
        let s12v = f32x8::from_array_t(token, *s12c);
        let mv = f32x8::from_array_t(token, *mc);

        let d = ((ssim_dissim8(token, form, m1, m2, ssq, s12v)) * mv).max(zero);
        let d2 = d * d;
        let d4 = d2 * d2;

        sum_d += d.reduce_add() as f64;
        sum_d2 += d2.reduce_add() as f64;
        sum_d4 += d4.reduce_add() as f64;
    }

    let off = mu1_chunks.len() * 8;
    for (i, &m1v) in mu1_tail.iter().enumerate() {
        let j = off + i;
        let d =
            ((ssim_dissim_raw_scalar(form, m1v, mu2[j], sum_sq[j], s12[j])) * mask[j]).max(0.0f32);
        let d2 = d * d;
        sum_d += d as f64;
        sum_d2 += d2 as f64;
        sum_d4 += (d2 * d2) as f64;
    }

    (sum_d, sum_d4, sum_d2)
}

// === ssim_channel_masked_2: fused 2-mask variant ===
// Computes unweighted SSIM distance once, applies two masks, accumulates
// into two separate (sum_d, sum_d4, sum_d2) tuples in one pass.

#[cfg(target_arch = "x86_64")]
#[arcane]
fn ssim_channel_masked_2_inner_v4(
    token: archmage::X64V4Token,
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
    mask_a: &[f32],
    mask_b: &[f32],
) -> ((f64, f64, f64), (f64, f64, f64)) {
    let form = crate::ssim_form::active_luma_form();
    let zero = f32x16::zero_t(token);

    let n = mu1.len();
    let chunks = n / 16;
    let mut sum_da = 0.0f64;
    let mut sum_d4a = 0.0f64;
    let mut sum_d2a = 0.0f64;
    let mut sum_db = 0.0f64;
    let mut sum_d4b = 0.0f64;
    let mut sum_d2b = 0.0f64;

    for c in 0..chunks {
        let base = c * 16;
        let m1 = f32x16::from_array_t(token, mu1[base..][..16].try_into().unwrap());
        let m2 = f32x16::from_array_t(token, mu2[base..][..16].try_into().unwrap());
        let ssq = f32x16::from_array_t(token, sum_sq[base..][..16].try_into().unwrap());
        let s12v = f32x16::from_array_t(token, s12[base..][..16].try_into().unwrap());
        let mva = f32x16::from_array_t(token, mask_a[base..][..16].try_into().unwrap());
        let mvb = f32x16::from_array_t(token, mask_b[base..][..16].try_into().unwrap());

        // Shared unweighted distance ∈ [0, 2]
        let d_raw = ssim_dissim16(token, form, m1, m2, ssq, s12v);

        let da = (d_raw * mva).max(zero);
        let d2a = da * da;
        let d4a = d2a * d2a;
        let db = (d_raw * mvb).max(zero);
        let d2b = db * db;
        let d4b = d2b * d2b;

        sum_da += da.reduce_add() as f64;
        sum_d2a += d2a.reduce_add() as f64;
        sum_d4a += d4a.reduce_add() as f64;
        sum_db += db.reduce_add() as f64;
        sum_d2b += d2b.reduce_add() as f64;
        sum_d4b += d4b.reduce_add() as f64;
    }

    for i in (chunks * 16)..n {
        let d_raw = ssim_dissim_raw_scalar(form, mu1[i], mu2[i], sum_sq[i], s12[i]);
        let da = (d_raw * mask_a[i]).max(0.0f32);
        let d2a = da * da;
        let db = (d_raw * mask_b[i]).max(0.0f32);
        let d2b = db * db;
        sum_da += da as f64;
        sum_d2a += d2a as f64;
        sum_d4a += (d2a * d2a) as f64;
        sum_db += db as f64;
        sum_d2b += d2b as f64;
        sum_d4b += (d2b * d2b) as f64;
    }

    ((sum_da, sum_d4a, sum_d2a), (sum_db, sum_d4b, sum_d2b))
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn ssim_channel_masked_2_inner_v3(
    token: archmage::X64V3Token,
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
    mask_a: &[f32],
    mask_b: &[f32],
) -> ((f64, f64, f64), (f64, f64, f64)) {
    let form = crate::ssim_form::active_luma_form();
    let zero = f32x8::zero_t(token);

    let n = mu1.len();
    let chunks = n / 8;
    let mut sum_da = 0.0f64;
    let mut sum_d4a = 0.0f64;
    let mut sum_d2a = 0.0f64;
    let mut sum_db = 0.0f64;
    let mut sum_d4b = 0.0f64;
    let mut sum_d2b = 0.0f64;

    for c in 0..chunks {
        let base = c * 8;
        let m1 = f32x8::from_array_t(token, mu1[base..][..8].try_into().unwrap());
        let m2 = f32x8::from_array_t(token, mu2[base..][..8].try_into().unwrap());
        let ssq = f32x8::from_array_t(token, sum_sq[base..][..8].try_into().unwrap());
        let s12v = f32x8::from_array_t(token, s12[base..][..8].try_into().unwrap());
        let mva = f32x8::from_array_t(token, mask_a[base..][..8].try_into().unwrap());
        let mvb = f32x8::from_array_t(token, mask_b[base..][..8].try_into().unwrap());

        let d_raw = ssim_dissim8(token, form, m1, m2, ssq, s12v);

        let da = (d_raw * mva).max(zero);
        let d2a = da * da;
        let d4a = d2a * d2a;
        let db = (d_raw * mvb).max(zero);
        let d2b = db * db;
        let d4b = d2b * d2b;

        sum_da += da.reduce_add() as f64;
        sum_d2a += d2a.reduce_add() as f64;
        sum_d4a += d4a.reduce_add() as f64;
        sum_db += db.reduce_add() as f64;
        sum_d2b += d2b.reduce_add() as f64;
        sum_d4b += d4b.reduce_add() as f64;
    }

    for i in (chunks * 8)..n {
        let d_raw = ssim_dissim_raw_scalar(form, mu1[i], mu2[i], sum_sq[i], s12[i]);
        let da = (d_raw * mask_a[i]).max(0.0f32);
        let d2a = da * da;
        let db = (d_raw * mask_b[i]).max(0.0f32);
        let d2b = db * db;
        sum_da += da as f64;
        sum_d2a += d2a as f64;
        sum_d4a += (d2a * d2a) as f64;
        sum_db += db as f64;
        sum_d2b += d2b as f64;
        sum_d4b += (d2b * d2b) as f64;
    }

    ((sum_da, sum_d4a, sum_d2a), (sum_db, sum_d4b, sum_d2b))
}

#[magetypes(neon, wasm128, scalar)]
fn ssim_channel_masked_2_inner(
    token: Token,
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
    mask_a: &[f32],
    mask_b: &[f32],
) -> ((f64, f64, f64), (f64, f64, f64)) {
    let form = crate::ssim_form::active_luma_form();
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let zero = f32x8::zero_t(token);

    let (mu1_chunks, mu1_tail) = mu1.as_chunks::<8>();
    let (mu2_chunks, _) = mu2.as_chunks::<8>();
    let (ssq_chunks, _) = sum_sq.as_chunks::<8>();
    let (s12_chunks, _) = s12.as_chunks::<8>();
    let (mask_a_chunks, _) = mask_a.as_chunks::<8>();
    let (mask_b_chunks, _) = mask_b.as_chunks::<8>();

    let mut sum_da = 0.0f64;
    let mut sum_d4a = 0.0f64;
    let mut sum_d2a = 0.0f64;
    let mut sum_db = 0.0f64;
    let mut sum_d4b = 0.0f64;
    let mut sum_d2b = 0.0f64;

    for (((((m1c, m2c), ssqc), s12c), mac), mbc) in mu1_chunks
        .iter()
        .zip(mu2_chunks)
        .zip(ssq_chunks)
        .zip(s12_chunks)
        .zip(mask_a_chunks)
        .zip(mask_b_chunks)
    {
        let m1 = f32x8::from_array_t(token, *m1c);
        let m2 = f32x8::from_array_t(token, *m2c);
        let ssq = f32x8::from_array_t(token, *ssqc);
        let s12v = f32x8::from_array_t(token, *s12c);
        let mva = f32x8::from_array_t(token, *mac);
        let mvb = f32x8::from_array_t(token, *mbc);

        let d_raw = ssim_dissim8(token, form, m1, m2, ssq, s12v);

        let da = (d_raw * mva).max(zero);
        let d2a = da * da;
        let d4a = d2a * d2a;
        let db = (d_raw * mvb).max(zero);
        let d2b = db * db;
        let d4b = d2b * d2b;

        sum_da += da.reduce_add() as f64;
        sum_d2a += d2a.reduce_add() as f64;
        sum_d4a += d4a.reduce_add() as f64;
        sum_db += db.reduce_add() as f64;
        sum_d2b += d2b.reduce_add() as f64;
        sum_d4b += d4b.reduce_add() as f64;
    }

    let off = mu1_chunks.len() * 8;
    for (i, &m1v) in mu1_tail.iter().enumerate() {
        let j = off + i;
        let d_raw = ssim_dissim_raw_scalar(form, m1v, mu2[j], sum_sq[j], s12[j]);
        let da = (d_raw * mask_a[j]).max(0.0f32);
        let d2a = da * da;
        let db = (d_raw * mask_b[j]).max(0.0f32);
        let d2b = db * db;
        sum_da += da as f64;
        sum_d2a += d2a as f64;
        sum_d4a += (d2a * d2a) as f64;
        sum_db += db as f64;
        sum_d2b += d2b as f64;
        sum_d4b += (d2b * d2b) as f64;
    }

    ((sum_da, sum_d4a, sum_d2a), (sum_db, sum_d4b, sum_d2b))
}

// === edge_diff_channel_masked: returns (art, art4, det, det4, art2, det2) ===

#[cfg(target_arch = "x86_64")]
#[arcane]
fn edge_diff_masked_inner_v4(
    token: archmage::X64V4Token,
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    mask: &[f32],
) -> (f64, f64, f64, f64, f64, f64) {
    let one = f32x16::splat_t(token, 1.0);
    let zero = f32x16::zero_t(token);

    let n = img1.len();
    let chunks = n / 16;
    let mut sum_art = 0.0f64;
    let mut sum_art4 = 0.0f64;
    let mut sum_art2 = 0.0f64;
    let mut sum_det = 0.0f64;
    let mut sum_det4 = 0.0f64;
    let mut sum_det2 = 0.0f64;

    for c in 0..chunks {
        let base = c * 16;
        let i1 = f32x16::from_array_t(token, img1[base..][..16].try_into().unwrap());
        let i2 = f32x16::from_array_t(token, img2[base..][..16].try_into().unwrap());
        let m1 = f32x16::from_array_t(token, mu1[base..][..16].try_into().unwrap());
        let m2 = f32x16::from_array_t(token, mu2[base..][..16].try_into().unwrap());
        let mv = f32x16::from_array_t(token, mask[base..][..16].try_into().unwrap());

        let diff1 = (i1 - m1).abs();
        let diff2 = (i2 - m2).abs();
        let d1 = ((one + diff2) / (one + diff1) - one) * mv;

        let artifact = d1.max(zero);
        let detail_lost = (-d1).max(zero);

        let a2 = artifact * artifact;
        let dl2 = detail_lost * detail_lost;

        sum_art += artifact.reduce_add() as f64;
        sum_art2 += a2.reduce_add() as f64;
        sum_art4 += (a2 * a2).reduce_add() as f64;
        sum_det += detail_lost.reduce_add() as f64;
        sum_det2 += dl2.reduce_add() as f64;
        sum_det4 += (dl2 * dl2).reduce_add() as f64;
    }

    for i in (chunks * 16)..n {
        let diff1 = (img1[i] - mu1[i]).abs();
        let diff2 = (img2[i] - mu2[i]).abs();
        let d1 = ((1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32) * mask[i];

        let artifact = d1.max(0.0f32);
        let detail_lost = (-d1).max(0.0f32);
        let a2 = artifact * artifact;
        let dl2 = detail_lost * detail_lost;
        sum_art += artifact as f64;
        sum_art2 += a2 as f64;
        sum_art4 += (a2 * a2) as f64;
        sum_det += detail_lost as f64;
        sum_det2 += dl2 as f64;
        sum_det4 += (dl2 * dl2) as f64;
    }

    (sum_art, sum_art4, sum_det, sum_det4, sum_art2, sum_det2)
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn edge_diff_masked_inner_v3(
    token: archmage::X64V3Token,
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    mask: &[f32],
) -> (f64, f64, f64, f64, f64, f64) {
    let one = f32x8::splat_t(token, 1.0);
    let zero = f32x8::zero_t(token);

    let n = img1.len();
    let chunks = n / 8;
    let mut sum_art = 0.0f64;
    let mut sum_art4 = 0.0f64;
    let mut sum_art2 = 0.0f64;
    let mut sum_det = 0.0f64;
    let mut sum_det4 = 0.0f64;
    let mut sum_det2 = 0.0f64;

    for c in 0..chunks {
        let base = c * 8;
        let i1 = f32x8::from_array_t(token, img1[base..][..8].try_into().unwrap());
        let i2 = f32x8::from_array_t(token, img2[base..][..8].try_into().unwrap());
        let m1 = f32x8::from_array_t(token, mu1[base..][..8].try_into().unwrap());
        let m2 = f32x8::from_array_t(token, mu2[base..][..8].try_into().unwrap());
        let mv = f32x8::from_array_t(token, mask[base..][..8].try_into().unwrap());

        let diff1 = (i1 - m1).abs();
        let diff2 = (i2 - m2).abs();
        let d1 = ((one + diff2) / (one + diff1) - one) * mv;

        let artifact = d1.max(zero);
        let detail_lost = (-d1).max(zero);

        let a2 = artifact * artifact;
        let dl2 = detail_lost * detail_lost;

        sum_art += artifact.reduce_add() as f64;
        sum_art2 += a2.reduce_add() as f64;
        sum_art4 += (a2 * a2).reduce_add() as f64;
        sum_det += detail_lost.reduce_add() as f64;
        sum_det2 += dl2.reduce_add() as f64;
        sum_det4 += (dl2 * dl2).reduce_add() as f64;
    }

    for i in (chunks * 8)..n {
        let diff1 = (img1[i] - mu1[i]).abs();
        let diff2 = (img2[i] - mu2[i]).abs();
        let d1 = ((1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32) * mask[i];

        let artifact = d1.max(0.0f32);
        let detail_lost = (-d1).max(0.0f32);
        let a2 = artifact * artifact;
        let dl2 = detail_lost * detail_lost;
        sum_art += artifact as f64;
        sum_art2 += a2 as f64;
        sum_art4 += (a2 * a2) as f64;
        sum_det += detail_lost as f64;
        sum_det2 += dl2 as f64;
        sum_det4 += (dl2 * dl2) as f64;
    }

    (sum_art, sum_art4, sum_det, sum_det4, sum_art2, sum_det2)
}

#[magetypes(neon, wasm128, scalar)]
fn edge_diff_masked_inner(
    token: Token,
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    mask: &[f32],
) -> (f64, f64, f64, f64, f64, f64) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let one = f32x8::splat_t(token, 1.0);
    let zero = f32x8::zero_t(token);

    let (i1_chunks, i1_tail) = img1.as_chunks::<8>();
    let (i2_chunks, _) = img2.as_chunks::<8>();
    let (m1_chunks, _) = mu1.as_chunks::<8>();
    let (m2_chunks, _) = mu2.as_chunks::<8>();
    let (mask_chunks, _) = mask.as_chunks::<8>();

    let mut sum_art = 0.0f64;
    let mut sum_art4 = 0.0f64;
    let mut sum_art2 = 0.0f64;
    let mut sum_det = 0.0f64;
    let mut sum_det4 = 0.0f64;
    let mut sum_det2 = 0.0f64;

    for ((((i1c, i2c), m1c), m2c), mc) in i1_chunks
        .iter()
        .zip(i2_chunks)
        .zip(m1_chunks)
        .zip(m2_chunks)
        .zip(mask_chunks)
    {
        let i1 = f32x8::from_array_t(token, *i1c);
        let i2 = f32x8::from_array_t(token, *i2c);
        let m1 = f32x8::from_array_t(token, *m1c);
        let m2 = f32x8::from_array_t(token, *m2c);
        let mv = f32x8::from_array_t(token, *mc);

        let diff1 = (i1 - m1).abs();
        let diff2 = (i2 - m2).abs();
        let d1 = ((one + diff2) / (one + diff1) - one) * mv;

        let artifact = d1.max(zero);
        let detail_lost = (-d1).max(zero);

        let a2 = artifact * artifact;
        let dl2 = detail_lost * detail_lost;

        sum_art += artifact.reduce_add() as f64;
        sum_art2 += a2.reduce_add() as f64;
        sum_art4 += (a2 * a2).reduce_add() as f64;
        sum_det += detail_lost.reduce_add() as f64;
        sum_det2 += dl2.reduce_add() as f64;
        sum_det4 += (dl2 * dl2).reduce_add() as f64;
    }

    let off = i1_chunks.len() * 8;
    for (i, _) in i1_tail.iter().enumerate() {
        let j = off + i;
        let diff1 = (img1[j] - mu1[j]).abs();
        let diff2 = (img2[j] - mu2[j]).abs();
        let d1 = ((1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32) * mask[j];

        let artifact = d1.max(0.0f32);
        let detail_lost = (-d1).max(0.0f32);
        let a2 = artifact * artifact;
        let dl2 = detail_lost * detail_lost;
        sum_art += artifact as f64;
        sum_art2 += a2 as f64;
        sum_art4 += (a2 * a2) as f64;
        sum_det += detail_lost as f64;
        sum_det2 += dl2 as f64;
        sum_det4 += (dl2 * dl2) as f64;
    }

    (sum_art, sum_art4, sum_det, sum_det4, sum_art2, sum_det2)
}

// === edge_diff_channel_masked_2_art4_det4: fused 2-mask variant ===
// Computes the unweighted edge response once, applies two masks, returns
// only the 4th-power pools (the only ones consumed by the streaming caller
// on the masked + IW paths). Returns ((art4_a, det4_a), (art4_b, det4_b)).

#[cfg(target_arch = "x86_64")]
#[arcane]
fn edge_diff_masked_2_art4_det4_inner_v4(
    token: archmage::X64V4Token,
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    mask_a: &[f32],
    mask_b: &[f32],
) -> ((f64, f64), (f64, f64)) {
    let one = f32x16::splat_t(token, 1.0);
    let zero = f32x16::zero_t(token);

    let n = img1.len();
    let chunks = n / 16;
    let mut sum_art4a = 0.0f64;
    let mut sum_det4a = 0.0f64;
    let mut sum_art4b = 0.0f64;
    let mut sum_det4b = 0.0f64;

    for c in 0..chunks {
        let base = c * 16;
        let i1 = f32x16::from_array_t(token, img1[base..][..16].try_into().unwrap());
        let i2 = f32x16::from_array_t(token, img2[base..][..16].try_into().unwrap());
        let m1 = f32x16::from_array_t(token, mu1[base..][..16].try_into().unwrap());
        let m2 = f32x16::from_array_t(token, mu2[base..][..16].try_into().unwrap());
        let mva = f32x16::from_array_t(token, mask_a[base..][..16].try_into().unwrap());
        let mvb = f32x16::from_array_t(token, mask_b[base..][..16].try_into().unwrap());

        // Shared unweighted edge response ∈ [..., ...]
        let diff1 = (i1 - m1).abs();
        let diff2 = (i2 - m2).abs();
        let d_raw = (one + diff2) / (one + diff1) - one;

        let da = d_raw * mva;
        let artifact_a = da.max(zero);
        let detail_a = (-da).max(zero);
        let a2a = artifact_a * artifact_a;
        let dl2a = detail_a * detail_a;
        sum_art4a += (a2a * a2a).reduce_add() as f64;
        sum_det4a += (dl2a * dl2a).reduce_add() as f64;

        let db = d_raw * mvb;
        let artifact_b = db.max(zero);
        let detail_b = (-db).max(zero);
        let a2b = artifact_b * artifact_b;
        let dl2b = detail_b * detail_b;
        sum_art4b += (a2b * a2b).reduce_add() as f64;
        sum_det4b += (dl2b * dl2b).reduce_add() as f64;
    }

    for i in (chunks * 16)..n {
        let diff1 = (img1[i] - mu1[i]).abs();
        let diff2 = (img2[i] - mu2[i]).abs();
        let d_raw = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;

        let da = d_raw * mask_a[i];
        let aa = da.max(0.0f32);
        let dla = (-da).max(0.0f32);
        let a2a = aa * aa;
        let dl2a = dla * dla;
        sum_art4a += (a2a * a2a) as f64;
        sum_det4a += (dl2a * dl2a) as f64;

        let db = d_raw * mask_b[i];
        let ab = db.max(0.0f32);
        let dlb = (-db).max(0.0f32);
        let a2b = ab * ab;
        let dl2b = dlb * dlb;
        sum_art4b += (a2b * a2b) as f64;
        sum_det4b += (dl2b * dl2b) as f64;
    }

    ((sum_art4a, sum_det4a), (sum_art4b, sum_det4b))
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn edge_diff_masked_2_art4_det4_inner_v3(
    token: archmage::X64V3Token,
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    mask_a: &[f32],
    mask_b: &[f32],
) -> ((f64, f64), (f64, f64)) {
    let one = f32x8::splat_t(token, 1.0);
    let zero = f32x8::zero_t(token);

    let n = img1.len();
    let chunks = n / 8;
    let mut sum_art4a = 0.0f64;
    let mut sum_det4a = 0.0f64;
    let mut sum_art4b = 0.0f64;
    let mut sum_det4b = 0.0f64;

    for c in 0..chunks {
        let base = c * 8;
        let i1 = f32x8::from_array_t(token, img1[base..][..8].try_into().unwrap());
        let i2 = f32x8::from_array_t(token, img2[base..][..8].try_into().unwrap());
        let m1 = f32x8::from_array_t(token, mu1[base..][..8].try_into().unwrap());
        let m2 = f32x8::from_array_t(token, mu2[base..][..8].try_into().unwrap());
        let mva = f32x8::from_array_t(token, mask_a[base..][..8].try_into().unwrap());
        let mvb = f32x8::from_array_t(token, mask_b[base..][..8].try_into().unwrap());

        let diff1 = (i1 - m1).abs();
        let diff2 = (i2 - m2).abs();
        let d_raw = (one + diff2) / (one + diff1) - one;

        let da = d_raw * mva;
        let artifact_a = da.max(zero);
        let detail_a = (-da).max(zero);
        let a2a = artifact_a * artifact_a;
        let dl2a = detail_a * detail_a;
        sum_art4a += (a2a * a2a).reduce_add() as f64;
        sum_det4a += (dl2a * dl2a).reduce_add() as f64;

        let db = d_raw * mvb;
        let artifact_b = db.max(zero);
        let detail_b = (-db).max(zero);
        let a2b = artifact_b * artifact_b;
        let dl2b = detail_b * detail_b;
        sum_art4b += (a2b * a2b).reduce_add() as f64;
        sum_det4b += (dl2b * dl2b).reduce_add() as f64;
    }

    for i in (chunks * 8)..n {
        let diff1 = (img1[i] - mu1[i]).abs();
        let diff2 = (img2[i] - mu2[i]).abs();
        let d_raw = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;

        let da = d_raw * mask_a[i];
        let aa = da.max(0.0f32);
        let dla = (-da).max(0.0f32);
        let a2a = aa * aa;
        let dl2a = dla * dla;
        sum_art4a += (a2a * a2a) as f64;
        sum_det4a += (dl2a * dl2a) as f64;

        let db = d_raw * mask_b[i];
        let ab = db.max(0.0f32);
        let dlb = (-db).max(0.0f32);
        let a2b = ab * ab;
        let dl2b = dlb * dlb;
        sum_art4b += (a2b * a2b) as f64;
        sum_det4b += (dl2b * dl2b) as f64;
    }

    ((sum_art4a, sum_det4a), (sum_art4b, sum_det4b))
}

#[magetypes(neon, wasm128, scalar)]
fn edge_diff_masked_2_art4_det4_inner(
    token: Token,
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    mask_a: &[f32],
    mask_b: &[f32],
) -> ((f64, f64), (f64, f64)) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let one = f32x8::splat_t(token, 1.0);
    let zero = f32x8::zero_t(token);

    let (i1_chunks, i1_tail) = img1.as_chunks::<8>();
    let (i2_chunks, _) = img2.as_chunks::<8>();
    let (m1_chunks, _) = mu1.as_chunks::<8>();
    let (m2_chunks, _) = mu2.as_chunks::<8>();
    let (ma_chunks, _) = mask_a.as_chunks::<8>();
    let (mb_chunks, _) = mask_b.as_chunks::<8>();

    let mut sum_art4a = 0.0f64;
    let mut sum_det4a = 0.0f64;
    let mut sum_art4b = 0.0f64;
    let mut sum_det4b = 0.0f64;

    for (((((i1c, i2c), m1c), m2c), mac), mbc) in i1_chunks
        .iter()
        .zip(i2_chunks)
        .zip(m1_chunks)
        .zip(m2_chunks)
        .zip(ma_chunks)
        .zip(mb_chunks)
    {
        let i1 = f32x8::from_array_t(token, *i1c);
        let i2 = f32x8::from_array_t(token, *i2c);
        let m1 = f32x8::from_array_t(token, *m1c);
        let m2 = f32x8::from_array_t(token, *m2c);
        let mva = f32x8::from_array_t(token, *mac);
        let mvb = f32x8::from_array_t(token, *mbc);

        let diff1 = (i1 - m1).abs();
        let diff2 = (i2 - m2).abs();
        let d_raw = (one + diff2) / (one + diff1) - one;

        let da = d_raw * mva;
        let artifact_a = da.max(zero);
        let detail_a = (-da).max(zero);
        let a2a = artifact_a * artifact_a;
        let dl2a = detail_a * detail_a;
        sum_art4a += (a2a * a2a).reduce_add() as f64;
        sum_det4a += (dl2a * dl2a).reduce_add() as f64;

        let db = d_raw * mvb;
        let artifact_b = db.max(zero);
        let detail_b = (-db).max(zero);
        let a2b = artifact_b * artifact_b;
        let dl2b = detail_b * detail_b;
        sum_art4b += (a2b * a2b).reduce_add() as f64;
        sum_det4b += (dl2b * dl2b).reduce_add() as f64;
    }

    let off = i1_chunks.len() * 8;
    for (i, _) in i1_tail.iter().enumerate() {
        let j = off + i;
        let diff1 = (img1[j] - mu1[j]).abs();
        let diff2 = (img2[j] - mu2[j]).abs();
        let d_raw = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;

        let da = d_raw * mask_a[j];
        let aa = da.max(0.0f32);
        let dla = (-da).max(0.0f32);
        let a2a = aa * aa;
        let dl2a = dla * dla;
        sum_art4a += (a2a * a2a) as f64;
        sum_det4a += (dl2a * dl2a) as f64;

        let db = d_raw * mask_b[j];
        let ab = db.max(0.0f32);
        let dlb = (-db).max(0.0f32);
        let a2b = ab * ab;
        let dl2b = dlb * dlb;
        sum_art4b += (a2b * a2b) as f64;
        sum_det4b += (dl2b * dl2b) as f64;
    }

    ((sum_art4a, sum_det4a), (sum_art4b, sum_det4b))
}

// === build_weights_and_mse: fused weight-build + MSE ===
// In one SIMD pass, given the blurred activity buffer:
// - writes mask_out[i] = 1 / (1 + k_mask * activity[i])
// - writes iw_out[i]   = 1 + k_iw   * activity[i]
// - accumulates Σ (src-dst)² · mask  and  Σ (src-dst)² · iw_weight
// Returns (masked_mse_sum, iw_mse_sum).

#[cfg(target_arch = "x86_64")]
#[arcane]
fn build_weights_and_mse_inner_v4(
    token: archmage::X64V4Token,
    activity: &[f32],
    k_mask: f32,
    k_iw: f32,
    src: &[f32],
    dst: &[f32],
    mask_out: &mut [f32],
    iw_out: &mut [f32],
) -> (f64, f64) {
    let one = f32x16::splat_t(token, 1.0);
    let kmv = f32x16::splat_t(token, k_mask);
    let kiv = f32x16::splat_t(token, k_iw);

    let n = activity.len();
    let chunks = n / 16;
    let mut mse_mask = 0.0f64;
    let mut mse_iw = 0.0f64;

    for c in 0..chunks {
        let base = c * 16;
        let av = f32x16::from_array_t(token, activity[base..][..16].try_into().unwrap());
        let sv = f32x16::from_array_t(token, src[base..][..16].try_into().unwrap());
        let dv = f32x16::from_array_t(token, dst[base..][..16].try_into().unwrap());

        // mask = 1 / (1 + k_mask * a)  → use rcp via division (denominator never zero
        //   because k_mask ≥ 0, a ≥ 0, +1 floor)
        let mask = one / kmv.mul_add_relaxed(av, one);
        // iw_weight = 1 + k_iw * a
        let iw = kiv.mul_add_relaxed(av, one);
        let diff = sv - dv;
        let d2 = diff * diff;
        mse_mask += (d2 * mask).reduce_add() as f64;
        mse_iw += (d2 * iw).reduce_add() as f64;
        mask_out[base..base + 16].copy_from_slice(&mask.to_array());
        iw_out[base..base + 16].copy_from_slice(&iw.to_array());
    }

    for i in (chunks * 16)..n {
        let a = activity[i];
        let mask = 1.0f32 / (1.0f32 + k_mask * a);
        let iw = 1.0f32 + k_iw * a;
        mask_out[i] = mask;
        iw_out[i] = iw;
        let d = src[i] - dst[i];
        let d2 = d * d;
        mse_mask += (d2 * mask) as f64;
        mse_iw += (d2 * iw) as f64;
    }

    (mse_mask, mse_iw)
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn build_weights_and_mse_inner_v3(
    token: archmage::X64V3Token,
    activity: &[f32],
    k_mask: f32,
    k_iw: f32,
    src: &[f32],
    dst: &[f32],
    mask_out: &mut [f32],
    iw_out: &mut [f32],
) -> (f64, f64) {
    let one = f32x8::splat_t(token, 1.0);
    let kmv = f32x8::splat_t(token, k_mask);
    let kiv = f32x8::splat_t(token, k_iw);

    let n = activity.len();
    let chunks = n / 8;
    let mut mse_mask = 0.0f64;
    let mut mse_iw = 0.0f64;

    for c in 0..chunks {
        let base = c * 8;
        let av = f32x8::from_array_t(token, activity[base..][..8].try_into().unwrap());
        let sv = f32x8::from_array_t(token, src[base..][..8].try_into().unwrap());
        let dv = f32x8::from_array_t(token, dst[base..][..8].try_into().unwrap());

        let mask = one / kmv.mul_add_relaxed(av, one);
        let iw = kiv.mul_add_relaxed(av, one);
        let diff = sv - dv;
        let d2 = diff * diff;
        mse_mask += (d2 * mask).reduce_add() as f64;
        mse_iw += (d2 * iw).reduce_add() as f64;
        mask_out[base..base + 8].copy_from_slice(&mask.to_array());
        iw_out[base..base + 8].copy_from_slice(&iw.to_array());
    }

    for i in (chunks * 8)..n {
        let a = activity[i];
        let mask = 1.0f32 / (1.0f32 + k_mask * a);
        let iw = 1.0f32 + k_iw * a;
        mask_out[i] = mask;
        iw_out[i] = iw;
        let d = src[i] - dst[i];
        let d2 = d * d;
        mse_mask += (d2 * mask) as f64;
        mse_iw += (d2 * iw) as f64;
    }

    (mse_mask, mse_iw)
}

#[magetypes(neon, wasm128, scalar)]
fn build_weights_and_mse_inner(
    token: Token,
    activity: &[f32],
    k_mask: f32,
    k_iw: f32,
    src: &[f32],
    dst: &[f32],
    mask_out: &mut [f32],
    iw_out: &mut [f32],
) -> (f64, f64) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let one = f32x8::splat_t(token, 1.0);
    let kmv = f32x8::splat_t(token, k_mask);
    let kiv = f32x8::splat_t(token, k_iw);

    let (a_chunks, a_tail) = activity.as_chunks::<8>();
    let (s_chunks, _) = src.as_chunks::<8>();
    let (d_chunks, _) = dst.as_chunks::<8>();
    let (mo_chunks, mo_tail) = mask_out.as_chunks_mut::<8>();
    let (io_chunks, io_tail) = iw_out.as_chunks_mut::<8>();

    let mut mse_mask = 0.0f64;
    let mut mse_iw = 0.0f64;

    for ((((ac, sc), dc), mo), io) in a_chunks
        .iter()
        .zip(s_chunks)
        .zip(d_chunks)
        .zip(mo_chunks)
        .zip(io_chunks)
    {
        let av = f32x8::from_array_t(token, *ac);
        let sv = f32x8::from_array_t(token, *sc);
        let dv = f32x8::from_array_t(token, *dc);

        let mask = one / kmv.mul_add_relaxed(av, one);
        let iw = kiv.mul_add_relaxed(av, one);
        let diff = sv - dv;
        let d2 = diff * diff;
        mse_mask += (d2 * mask).reduce_add() as f64;
        mse_iw += (d2 * iw).reduce_add() as f64;
        mask.store(mo);
        iw.store(io);
    }

    let off = a_chunks.len() * 8;
    for ((i, &a), (mo, io)) in a_tail
        .iter()
        .enumerate()
        .zip(mo_tail.iter_mut().zip(io_tail.iter_mut()))
    {
        let j = off + i;
        let mask = 1.0f32 / (1.0f32 + k_mask * a);
        let iw = 1.0f32 + k_iw * a;
        *mo = mask;
        *io = iw;
        let d = src[j] - dst[j];
        let d2 = d * d;
        mse_mask += (d2 * mask) as f64;
        mse_iw += (d2 * iw) as f64;
    }

    (mse_mask, mse_iw)
}

// === build_mask_weight_and_mse: single-mask variant ===

#[cfg(target_arch = "x86_64")]
#[arcane]
fn build_mask_weight_and_mse_inner_v4(
    token: archmage::X64V4Token,
    activity: &[f32],
    k_mask: f32,
    src: &[f32],
    dst: &[f32],
    mask_out: &mut [f32],
) -> f64 {
    let one = f32x16::splat_t(token, 1.0);
    let kmv = f32x16::splat_t(token, k_mask);
    let n = activity.len();
    let chunks = n / 16;
    let mut mse = 0.0f64;
    for c in 0..chunks {
        let base = c * 16;
        let av = f32x16::from_array_t(token, activity[base..][..16].try_into().unwrap());
        let sv = f32x16::from_array_t(token, src[base..][..16].try_into().unwrap());
        let dv = f32x16::from_array_t(token, dst[base..][..16].try_into().unwrap());
        let mask = one / kmv.mul_add_relaxed(av, one);
        let diff = sv - dv;
        let d2 = diff * diff;
        mse += (d2 * mask).reduce_add() as f64;
        mask_out[base..base + 16].copy_from_slice(&mask.to_array());
    }
    for i in (chunks * 16)..n {
        let a = activity[i];
        let mask = 1.0f32 / (1.0f32 + k_mask * a);
        mask_out[i] = mask;
        let d = src[i] - dst[i];
        mse += (d * d * mask) as f64;
    }
    mse
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn build_mask_weight_and_mse_inner_v3(
    token: archmage::X64V3Token,
    activity: &[f32],
    k_mask: f32,
    src: &[f32],
    dst: &[f32],
    mask_out: &mut [f32],
) -> f64 {
    let one = f32x8::splat_t(token, 1.0);
    let kmv = f32x8::splat_t(token, k_mask);
    let n = activity.len();
    let chunks = n / 8;
    let mut mse = 0.0f64;
    for c in 0..chunks {
        let base = c * 8;
        let av = f32x8::from_array_t(token, activity[base..][..8].try_into().unwrap());
        let sv = f32x8::from_array_t(token, src[base..][..8].try_into().unwrap());
        let dv = f32x8::from_array_t(token, dst[base..][..8].try_into().unwrap());
        let mask = one / kmv.mul_add_relaxed(av, one);
        let diff = sv - dv;
        let d2 = diff * diff;
        mse += (d2 * mask).reduce_add() as f64;
        mask_out[base..base + 8].copy_from_slice(&mask.to_array());
    }
    for i in (chunks * 8)..n {
        let a = activity[i];
        let mask = 1.0f32 / (1.0f32 + k_mask * a);
        mask_out[i] = mask;
        let d = src[i] - dst[i];
        mse += (d * d * mask) as f64;
    }
    mse
}

#[magetypes(neon, wasm128, scalar)]
fn build_mask_weight_and_mse_inner(
    token: Token,
    activity: &[f32],
    k_mask: f32,
    src: &[f32],
    dst: &[f32],
    mask_out: &mut [f32],
) -> f64 {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let one = f32x8::splat_t(token, 1.0);
    let kmv = f32x8::splat_t(token, k_mask);
    let (a_chunks, a_tail) = activity.as_chunks::<8>();
    let (s_chunks, _) = src.as_chunks::<8>();
    let (d_chunks, _) = dst.as_chunks::<8>();
    let (mo_chunks, mo_tail) = mask_out.as_chunks_mut::<8>();
    let mut mse = 0.0f64;
    for (((ac, sc), dc), mo) in a_chunks.iter().zip(s_chunks).zip(d_chunks).zip(mo_chunks) {
        let av = f32x8::from_array_t(token, *ac);
        let sv = f32x8::from_array_t(token, *sc);
        let dv = f32x8::from_array_t(token, *dc);
        let mask = one / kmv.mul_add_relaxed(av, one);
        let diff = sv - dv;
        let d2 = diff * diff;
        mse += (d2 * mask).reduce_add() as f64;
        mask.store(mo);
    }
    let off = a_chunks.len() * 8;
    for ((i, &a), mo) in a_tail.iter().enumerate().zip(mo_tail.iter_mut()) {
        let j = off + i;
        let mask = 1.0f32 / (1.0f32 + k_mask * a);
        *mo = mask;
        let d = src[j] - dst[j];
        mse += (d * d * mask) as f64;
    }
    mse
}

// === build_iw_weight_and_mse: single-IW variant ===

#[cfg(target_arch = "x86_64")]
#[arcane]
fn build_iw_weight_and_mse_inner_v4(
    token: archmage::X64V4Token,
    activity: &[f32],
    k_iw: f32,
    src: &[f32],
    dst: &[f32],
    iw_out: &mut [f32],
) -> f64 {
    let one = f32x16::splat_t(token, 1.0);
    let kiv = f32x16::splat_t(token, k_iw);
    let n = activity.len();
    let chunks = n / 16;
    let mut mse = 0.0f64;
    for c in 0..chunks {
        let base = c * 16;
        let av = f32x16::from_array_t(token, activity[base..][..16].try_into().unwrap());
        let sv = f32x16::from_array_t(token, src[base..][..16].try_into().unwrap());
        let dv = f32x16::from_array_t(token, dst[base..][..16].try_into().unwrap());
        let iw = kiv.mul_add_relaxed(av, one);
        let diff = sv - dv;
        let d2 = diff * diff;
        mse += (d2 * iw).reduce_add() as f64;
        iw_out[base..base + 16].copy_from_slice(&iw.to_array());
    }
    for i in (chunks * 16)..n {
        let a = activity[i];
        let iw = 1.0f32 + k_iw * a;
        iw_out[i] = iw;
        let d = src[i] - dst[i];
        mse += (d * d * iw) as f64;
    }
    mse
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn build_iw_weight_and_mse_inner_v3(
    token: archmage::X64V3Token,
    activity: &[f32],
    k_iw: f32,
    src: &[f32],
    dst: &[f32],
    iw_out: &mut [f32],
) -> f64 {
    let one = f32x8::splat_t(token, 1.0);
    let kiv = f32x8::splat_t(token, k_iw);
    let n = activity.len();
    let chunks = n / 8;
    let mut mse = 0.0f64;
    for c in 0..chunks {
        let base = c * 8;
        let av = f32x8::from_array_t(token, activity[base..][..8].try_into().unwrap());
        let sv = f32x8::from_array_t(token, src[base..][..8].try_into().unwrap());
        let dv = f32x8::from_array_t(token, dst[base..][..8].try_into().unwrap());
        let iw = kiv.mul_add_relaxed(av, one);
        let diff = sv - dv;
        let d2 = diff * diff;
        mse += (d2 * iw).reduce_add() as f64;
        iw_out[base..base + 8].copy_from_slice(&iw.to_array());
    }
    for i in (chunks * 8)..n {
        let a = activity[i];
        let iw = 1.0f32 + k_iw * a;
        iw_out[i] = iw;
        let d = src[i] - dst[i];
        mse += (d * d * iw) as f64;
    }
    mse
}

#[magetypes(neon, wasm128, scalar)]
fn build_iw_weight_and_mse_inner(
    token: Token,
    activity: &[f32],
    k_iw: f32,
    src: &[f32],
    dst: &[f32],
    iw_out: &mut [f32],
) -> f64 {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let one = f32x8::splat_t(token, 1.0);
    let kiv = f32x8::splat_t(token, k_iw);
    let (a_chunks, a_tail) = activity.as_chunks::<8>();
    let (s_chunks, _) = src.as_chunks::<8>();
    let (d_chunks, _) = dst.as_chunks::<8>();
    let (io_chunks, io_tail) = iw_out.as_chunks_mut::<8>();
    let mut mse = 0.0f64;
    for (((ac, sc), dc), io) in a_chunks.iter().zip(s_chunks).zip(d_chunks).zip(io_chunks) {
        let av = f32x8::from_array_t(token, *ac);
        let sv = f32x8::from_array_t(token, *sc);
        let dv = f32x8::from_array_t(token, *dc);
        let iw = kiv.mul_add_relaxed(av, one);
        let diff = sv - dv;
        let d2 = diff * diff;
        mse += (d2 * iw).reduce_add() as f64;
        iw.store(io);
    }
    let off = a_chunks.len() * 8;
    for ((i, &a), io) in a_tail.iter().enumerate().zip(io_tail.iter_mut()) {
        let j = off + i;
        let iw = 1.0f32 + k_iw * a;
        *io = iw;
        let d = src[j] - dst[j];
        mse += (d * d * iw) as f64;
    }
    mse
}

// === ssim_channel_extended: returns (sum_d, sum_d4, sum_d2, sum_d8, max_d) ===

#[cfg(target_arch = "x86_64")]
#[arcane]
fn ssim_channel_extended_inner_v4(
    token: archmage::X64V4Token,
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
) -> (f64, f64, f64, f64, f32) {
    let form = crate::ssim_form::active_luma_form();
    let zero = f32x16::zero_t(token);

    let n = mu1.len();
    let chunks = n / 16;
    let mut sum_d = 0.0f64;
    let mut sum_d4 = 0.0f64;
    let mut sum_d2 = 0.0f64;
    let mut sum_d8 = 0.0f64;
    let mut max_d_vec = zero;

    for c in 0..chunks {
        let base = c * 16;
        let m1 = f32x16::from_array_t(token, mu1[base..][..16].try_into().unwrap());
        let m2 = f32x16::from_array_t(token, mu2[base..][..16].try_into().unwrap());
        let ssq = f32x16::from_array_t(token, sum_sq[base..][..16].try_into().unwrap());
        let s12v = f32x16::from_array_t(token, s12[base..][..16].try_into().unwrap());

        let d = (ssim_dissim16(token, form, m1, m2, ssq, s12v)).max(zero);
        let d2 = d * d;
        let d4 = d2 * d2;
        let d8 = d4 * d4;

        sum_d += d.reduce_add() as f64;
        sum_d4 += d4.reduce_add() as f64;
        sum_d2 += d2.reduce_add() as f64;
        sum_d8 += d8.reduce_add() as f64;
        max_d_vec = max_d_vec.max(d);
    }

    let mut max_d = max_d_vec.reduce_max();
    for i in (chunks * 16)..n {
        let d = (ssim_dissim_raw_scalar(form, mu1[i], mu2[i], sum_sq[i], s12[i])).max(0.0f32);
        let d2 = d * d;
        let d4 = d2 * d2;
        sum_d += d as f64;
        sum_d4 += d4 as f64;
        sum_d2 += d2 as f64;
        sum_d8 += (d4 * d4) as f64;
        max_d = max_d.max(d);
    }

    (sum_d, sum_d4, sum_d2, sum_d8, max_d)
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn ssim_channel_extended_inner_v3(
    token: archmage::X64V3Token,
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
) -> (f64, f64, f64, f64, f32) {
    let form = crate::ssim_form::active_luma_form();
    let zero = f32x8::zero_t(token);

    let n = mu1.len();
    let chunks = n / 8;
    let mut sum_d = 0.0f64;
    let mut sum_d4 = 0.0f64;
    let mut sum_d2 = 0.0f64;
    let mut sum_d8 = 0.0f64;
    let mut max_d_vec = zero;

    for c in 0..chunks {
        let base = c * 8;
        let m1 = f32x8::from_array_t(token, mu1[base..][..8].try_into().unwrap());
        let m2 = f32x8::from_array_t(token, mu2[base..][..8].try_into().unwrap());
        let ssq = f32x8::from_array_t(token, sum_sq[base..][..8].try_into().unwrap());
        let s12v = f32x8::from_array_t(token, s12[base..][..8].try_into().unwrap());

        let d = (ssim_dissim8(token, form, m1, m2, ssq, s12v)).max(zero);
        let d2 = d * d;
        let d4 = d2 * d2;
        let d8 = d4 * d4;

        sum_d += d.reduce_add() as f64;
        sum_d4 += d4.reduce_add() as f64;
        sum_d2 += d2.reduce_add() as f64;
        sum_d8 += d8.reduce_add() as f64;
        max_d_vec = max_d_vec.max(d);
    }

    let mut max_d = max_d_vec.reduce_max();
    for i in (chunks * 8)..n {
        let d = (ssim_dissim_raw_scalar(form, mu1[i], mu2[i], sum_sq[i], s12[i])).max(0.0f32);
        let d2 = d * d;
        let d4 = d2 * d2;
        sum_d += d as f64;
        sum_d4 += d4 as f64;
        sum_d2 += d2 as f64;
        sum_d8 += (d4 * d4) as f64;
        max_d = max_d.max(d);
    }

    (sum_d, sum_d4, sum_d2, sum_d8, max_d)
}

#[magetypes(neon, wasm128, scalar)]
fn ssim_channel_extended_inner(
    token: Token,
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
) -> (f64, f64, f64, f64, f32) {
    let form = crate::ssim_form::active_luma_form();
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let zero = f32x8::zero_t(token);

    let (mu1_chunks, mu1_tail) = mu1.as_chunks::<8>();
    let (mu2_chunks, _) = mu2.as_chunks::<8>();
    let (ssq_chunks, _) = sum_sq.as_chunks::<8>();
    let (s12_chunks, _) = s12.as_chunks::<8>();

    let mut sum_d = 0.0f64;
    let mut sum_d4 = 0.0f64;
    let mut sum_d2 = 0.0f64;
    let mut sum_d8 = 0.0f64;
    let mut max_d_vec = zero;

    for (((m1c, m2c), ssqc), s12c) in mu1_chunks
        .iter()
        .zip(mu2_chunks)
        .zip(ssq_chunks)
        .zip(s12_chunks)
    {
        let m1 = f32x8::from_array_t(token, *m1c);
        let m2 = f32x8::from_array_t(token, *m2c);
        let ssq = f32x8::from_array_t(token, *ssqc);
        let s12v = f32x8::from_array_t(token, *s12c);

        let d = (ssim_dissim8(token, form, m1, m2, ssq, s12v)).max(zero);
        let d2 = d * d;
        let d4 = d2 * d2;
        let d8 = d4 * d4;

        sum_d += d.reduce_add() as f64;
        sum_d4 += d4.reduce_add() as f64;
        sum_d2 += d2.reduce_add() as f64;
        sum_d8 += d8.reduce_add() as f64;
        max_d_vec = max_d_vec.max(d);
    }

    let mut max_d = max_d_vec.reduce_max();
    let off = mu1_chunks.len() * 8;
    for (i, &m1v) in mu1_tail.iter().enumerate() {
        let j = off + i;
        let d = (ssim_dissim_raw_scalar(form, m1v, mu2[j], sum_sq[j], s12[j])).max(0.0f32);
        let d2 = d * d;
        let d4 = d2 * d2;
        sum_d += d as f64;
        sum_d4 += d4 as f64;
        sum_d2 += d2 as f64;
        sum_d8 += (d4 * d4) as f64;
        max_d = max_d.max(d);
    }

    (sum_d, sum_d4, sum_d2, sum_d8, max_d)
}

// === edge_diff_channel_extended: returns (art, art4, det, det4, art2, det2, art8, det8, max_art, max_det) ===

#[cfg(target_arch = "x86_64")]
#[arcane]
fn edge_diff_extended_inner_v4(
    token: archmage::X64V4Token,
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f32, f32) {
    let one = f32x16::splat_t(token, 1.0);
    let zero = f32x16::zero_t(token);

    let n = img1.len();
    let chunks = n / 16;
    let mut sum_art = 0.0f64;
    let mut sum_art4 = 0.0f64;
    let mut sum_art2 = 0.0f64;
    let mut sum_art8 = 0.0f64;
    let mut sum_det = 0.0f64;
    let mut sum_det4 = 0.0f64;
    let mut sum_det2 = 0.0f64;
    let mut sum_det8 = 0.0f64;
    let mut max_art_vec = zero;
    let mut max_det_vec = zero;

    for c in 0..chunks {
        let base = c * 16;
        let i1 = f32x16::from_array_t(token, img1[base..][..16].try_into().unwrap());
        let i2 = f32x16::from_array_t(token, img2[base..][..16].try_into().unwrap());
        let m1 = f32x16::from_array_t(token, mu1[base..][..16].try_into().unwrap());
        let m2 = f32x16::from_array_t(token, mu2[base..][..16].try_into().unwrap());

        let diff1 = (i1 - m1).abs();
        let diff2 = (i2 - m2).abs();
        let d1 = (one + diff2) / (one + diff1) - one;

        let artifact = d1.max(zero);
        let detail_lost = (-d1).max(zero);

        let a2 = artifact * artifact;
        let a4 = a2 * a2;
        let a8 = a4 * a4;
        let dl2 = detail_lost * detail_lost;
        let dl4 = dl2 * dl2;
        let dl8 = dl4 * dl4;

        sum_art += artifact.reduce_add() as f64;
        sum_art4 += a4.reduce_add() as f64;
        sum_art2 += a2.reduce_add() as f64;
        sum_art8 += a8.reduce_add() as f64;
        sum_det += detail_lost.reduce_add() as f64;
        sum_det4 += dl4.reduce_add() as f64;
        sum_det2 += dl2.reduce_add() as f64;
        sum_det8 += dl8.reduce_add() as f64;
        max_art_vec = max_art_vec.max(artifact);
        max_det_vec = max_det_vec.max(detail_lost);
    }

    let mut max_art = max_art_vec.reduce_max();
    let mut max_det = max_det_vec.reduce_max();

    for i in (chunks * 16)..n {
        let diff1 = (img1[i] - mu1[i]).abs();
        let diff2 = (img2[i] - mu2[i]).abs();
        let d1 = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;

        let artifact = d1.max(0.0f32);
        let detail_lost = (-d1).max(0.0f32);
        let a2 = artifact * artifact;
        let a4 = a2 * a2;
        let dl2 = detail_lost * detail_lost;
        let dl4 = dl2 * dl2;
        sum_art += artifact as f64;
        sum_art4 += a4 as f64;
        sum_art2 += a2 as f64;
        sum_art8 += (a4 * a4) as f64;
        sum_det += detail_lost as f64;
        sum_det4 += dl4 as f64;
        sum_det2 += dl2 as f64;
        sum_det8 += (dl4 * dl4) as f64;
        max_art = max_art.max(artifact);
        max_det = max_det.max(detail_lost);
    }

    (
        sum_art, sum_art4, sum_det, sum_det4, sum_art2, sum_det2, sum_art8, sum_det8, max_art,
        max_det,
    )
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn edge_diff_extended_inner_v3(
    token: archmage::X64V3Token,
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f32, f32) {
    let one = f32x8::splat_t(token, 1.0);
    let zero = f32x8::zero_t(token);

    let n = img1.len();
    let chunks = n / 8;
    let mut sum_art = 0.0f64;
    let mut sum_art4 = 0.0f64;
    let mut sum_art2 = 0.0f64;
    let mut sum_art8 = 0.0f64;
    let mut sum_det = 0.0f64;
    let mut sum_det4 = 0.0f64;
    let mut sum_det2 = 0.0f64;
    let mut sum_det8 = 0.0f64;
    let mut max_art_vec = zero;
    let mut max_det_vec = zero;

    for c in 0..chunks {
        let base = c * 8;
        let i1 = f32x8::from_array_t(token, img1[base..][..8].try_into().unwrap());
        let i2 = f32x8::from_array_t(token, img2[base..][..8].try_into().unwrap());
        let m1 = f32x8::from_array_t(token, mu1[base..][..8].try_into().unwrap());
        let m2 = f32x8::from_array_t(token, mu2[base..][..8].try_into().unwrap());

        let diff1 = (i1 - m1).abs();
        let diff2 = (i2 - m2).abs();
        let d1 = (one + diff2) / (one + diff1) - one;

        let artifact = d1.max(zero);
        let detail_lost = (-d1).max(zero);

        let a2 = artifact * artifact;
        let a4 = a2 * a2;
        let a8 = a4 * a4;
        let dl2 = detail_lost * detail_lost;
        let dl4 = dl2 * dl2;
        let dl8 = dl4 * dl4;

        sum_art += artifact.reduce_add() as f64;
        sum_art4 += a4.reduce_add() as f64;
        sum_art2 += a2.reduce_add() as f64;
        sum_art8 += a8.reduce_add() as f64;
        sum_det += detail_lost.reduce_add() as f64;
        sum_det4 += dl4.reduce_add() as f64;
        sum_det2 += dl2.reduce_add() as f64;
        sum_det8 += dl8.reduce_add() as f64;
        max_art_vec = max_art_vec.max(artifact);
        max_det_vec = max_det_vec.max(detail_lost);
    }

    let mut max_art = max_art_vec.reduce_max();
    let mut max_det = max_det_vec.reduce_max();

    for i in (chunks * 8)..n {
        let diff1 = (img1[i] - mu1[i]).abs();
        let diff2 = (img2[i] - mu2[i]).abs();
        let d1 = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;

        let artifact = d1.max(0.0f32);
        let detail_lost = (-d1).max(0.0f32);
        let a2 = artifact * artifact;
        let a4 = a2 * a2;
        let dl2 = detail_lost * detail_lost;
        let dl4 = dl2 * dl2;
        sum_art += artifact as f64;
        sum_art4 += a4 as f64;
        sum_art2 += a2 as f64;
        sum_art8 += (a4 * a4) as f64;
        sum_det += detail_lost as f64;
        sum_det4 += dl4 as f64;
        sum_det2 += dl2 as f64;
        sum_det8 += (dl4 * dl4) as f64;
        max_art = max_art.max(artifact);
        max_det = max_det.max(detail_lost);
    }

    (
        sum_art, sum_art4, sum_det, sum_det4, sum_art2, sum_det2, sum_art8, sum_det8, max_art,
        max_det,
    )
}

#[magetypes(neon, wasm128, scalar)]
fn edge_diff_extended_inner(
    token: Token,
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f32, f32) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let one = f32x8::splat_t(token, 1.0);
    let zero = f32x8::zero_t(token);

    let (i1_chunks, i1_tail) = img1.as_chunks::<8>();
    let (i2_chunks, _) = img2.as_chunks::<8>();
    let (m1_chunks, _) = mu1.as_chunks::<8>();
    let (m2_chunks, _) = mu2.as_chunks::<8>();

    let mut sum_art = 0.0f64;
    let mut sum_art4 = 0.0f64;
    let mut sum_art2 = 0.0f64;
    let mut sum_art8 = 0.0f64;
    let mut sum_det = 0.0f64;
    let mut sum_det4 = 0.0f64;
    let mut sum_det2 = 0.0f64;
    let mut sum_det8 = 0.0f64;
    let mut max_art_vec = zero;
    let mut max_det_vec = zero;

    for (((i1c, i2c), m1c), m2c) in i1_chunks
        .iter()
        .zip(i2_chunks)
        .zip(m1_chunks)
        .zip(m2_chunks)
    {
        let i1 = f32x8::from_array_t(token, *i1c);
        let i2 = f32x8::from_array_t(token, *i2c);
        let m1 = f32x8::from_array_t(token, *m1c);
        let m2 = f32x8::from_array_t(token, *m2c);

        let diff1 = (i1 - m1).abs();
        let diff2 = (i2 - m2).abs();
        let d1 = (one + diff2) / (one + diff1) - one;

        let artifact = d1.max(zero);
        let detail_lost = (-d1).max(zero);

        let a2 = artifact * artifact;
        let a4 = a2 * a2;
        let a8 = a4 * a4;
        let dl2 = detail_lost * detail_lost;
        let dl4 = dl2 * dl2;
        let dl8 = dl4 * dl4;

        sum_art += artifact.reduce_add() as f64;
        sum_art4 += a4.reduce_add() as f64;
        sum_art2 += a2.reduce_add() as f64;
        sum_art8 += a8.reduce_add() as f64;
        sum_det += detail_lost.reduce_add() as f64;
        sum_det4 += dl4.reduce_add() as f64;
        sum_det2 += dl2.reduce_add() as f64;
        sum_det8 += dl8.reduce_add() as f64;
        max_art_vec = max_art_vec.max(artifact);
        max_det_vec = max_det_vec.max(detail_lost);
    }

    let mut max_art = max_art_vec.reduce_max();
    let mut max_det = max_det_vec.reduce_max();

    let off = i1_chunks.len() * 8;
    for (i, _) in i1_tail.iter().enumerate() {
        let j = off + i;
        let diff1 = (img1[j] - mu1[j]).abs();
        let diff2 = (img2[j] - mu2[j]).abs();
        let d1 = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;

        let artifact = d1.max(0.0f32);
        let detail_lost = (-d1).max(0.0f32);
        let a2 = artifact * artifact;
        let a4 = a2 * a2;
        let dl2 = detail_lost * detail_lost;
        let dl4 = dl2 * dl2;
        sum_art += artifact as f64;
        sum_art4 += a4 as f64;
        sum_art2 += a2 as f64;
        sum_art8 += (a4 * a4) as f64;
        sum_det += detail_lost as f64;
        sum_det4 += dl4 as f64;
        sum_det2 += dl2 as f64;
        sum_det8 += (dl4 * dl4) as f64;
        max_art = max_art.max(artifact);
        max_det = max_det.max(detail_lost);
    }

    (
        sum_art, sum_art4, sum_det, sum_det4, sum_art2, sum_det2, sum_art8, sum_det8, max_art,
        max_det,
    )
}

// ============================================================================
// IW-INLINE fusion kernels (added 2026-05-22)
//
// These kernels eliminate the per-pixel `iw_weight` plane round-trip. The IW
// weight is `1 + k_iw * activity[i]` — a single FMA per pixel, cheaper to
// recompute inline at the consumer than to write to a plane and read back.
//
// **Reduction-order invariant:** these kernels compute `iw_v = kiv.mul_add(
// av, one)` per SIMD lane and then `(d * iw_v).reduce_add()`. The previous
// build-then-consume path wrote `iw = kiv.mul_add(av, one)` to `iw_out[]`
// (bit-identical f32 result), then a separate kernel loaded `iw_out[]` and
// performed the same per-lane multiply + reduce. Because SIMD lane width and
// chunk size match exactly between the two implementations, the per-chunk
// `reduce_add()` sees the same lane values in the same order. Both paths are
// **byte-exact equivalent** within f32 precision — there is no
// non-associative-reduction drift at the chunk boundary.
//
// Used by the streaming hot path in `streaming.rs` when
// `config.compute_iw_features` is on. Replaces `build_weights_and_mse` +
// `ssim_channel_masked_2` + `edge_diff_channel_masked_2_art4_det4` (or the
// IW-alone counterparts) with their fused inline-IW variants — eliminating
// the `bufs.iw_weight` plane write/read entirely.
// ============================================================================

/// Like `build_weights_and_mse` but **only writes mask plane** (no IW plane).
/// IW weight is computed inline + folded into the MSE accumulator, then
/// thrown away. Saves the `iw_out` plane write entirely.
///
/// **Deprecated for the streaming hot path (Phase 2, 2026-05-22)** —
/// [`build_inline_mse`] performs the same accumulation WITHOUT writing
/// the mask plane either. Kept for API stability and unit-test coverage.
///
/// Returns `(mse_mask, mse_iw)`.
#[allow(dead_code)]
pub(crate) fn build_mask_and_iw_mse_inline(
    activity: &[f32],
    k_mask: f32,
    k_iw: f32,
    src: &[f32],
    dst: &[f32],
    mask_out: &mut [f32],
) -> (f64, f64) {
    incant!(
        build_mask_and_iw_mse_inline_inner(activity, k_mask, k_iw, src, dst, mask_out),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// IW-only fused MSE — no plane writes at all. Computes
/// `Σ (src-dst)² · (1 + k_iw * activity[i])` directly.
pub(crate) fn build_iw_mse_only(activity: &[f32], k_iw: f32, src: &[f32], dst: &[f32]) -> f64 {
    incant!(
        build_iw_mse_only_inner(activity, k_iw, src, dst),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// SSIM with mask plane + IW weight derived inline from activity.
/// Returns `((masked d/d4/d2), (iw d/d4/d2))`.
///
/// **Deprecated for the streaming hot path (Phase 2, 2026-05-22)** — see
/// [`ssim_channel_inline_both`] which derives the mask weight inline
/// from activity as well. Kept for API stability and unit-test coverage.
#[allow(dead_code)]
pub(crate) fn ssim_channel_masked_with_iw_inline(
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
    mask: &[f32],
    activity: &[f32],
    k_iw: f32,
) -> ((f64, f64, f64), (f64, f64, f64)) {
    incant!(
        ssim_channel_masked_with_iw_inline_inner(mu1, mu2, sum_sq, s12, mask, activity, k_iw),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// SSIM with IW weight derived inline from activity (no mask).
/// Returns `(sum_d, sum_d4, sum_d2)`.
pub(crate) fn ssim_channel_iw_inline(
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
    activity: &[f32],
    k_iw: f32,
) -> (f64, f64, f64) {
    incant!(
        ssim_channel_iw_inline_inner(mu1, mu2, sum_sq, s12, activity, k_iw),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// Edge-diff with mask plane + IW weight derived inline from activity.
/// Returns `((art4_m, det4_m), (art4_i, det4_i))`.
///
/// **Deprecated for the streaming hot path (Phase 2, 2026-05-22)** — see
/// [`edge_diff_channel_inline_both`] which derives the mask weight
/// inline from activity as well. Kept for API stability and unit-test
/// coverage.
#[allow(dead_code)]
pub(crate) fn edge_diff_channel_masked_with_iw_inline(
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    mask: &[f32],
    activity: &[f32],
    k_iw: f32,
) -> ((f64, f64), (f64, f64)) {
    incant!(
        edge_diff_channel_masked_with_iw_inline_inner(img1, img2, mu1, mu2, mask, activity, k_iw),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// Edge-diff with IW weight derived inline from activity (no mask).
/// Returns `(art, art4, det, det4, art2, det2)`.
pub(crate) fn edge_diff_channel_iw_inline(
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    activity: &[f32],
    k_iw: f32,
) -> (f64, f64, f64, f64, f64, f64) {
    incant!(
        edge_diff_channel_iw_inline_inner(img1, img2, mu1, mu2, activity, k_iw),
        [v4, v3, neon, wasm128, scalar]
    )
}

// --- build_mask_and_iw_mse_inline ---

#[magetypes(v4, v3, neon, wasm128, scalar)]
fn build_mask_and_iw_mse_inline_inner(
    token: Token,
    activity: &[f32],
    k_mask: f32,
    k_iw: f32,
    src: &[f32],
    dst: &[f32],
    mask_out: &mut [f32],
) -> (f64, f64) {
    let one = f32x16::splat_t(token, 1.0);
    let kmv = f32x16::splat_t(token, k_mask);
    let kiv = f32x16::splat_t(token, k_iw);

    let (a_chunks, a_tail) = activity.as_chunks::<16>();
    let (s_chunks, _) = src.as_chunks::<16>();
    let (d_chunks, _) = dst.as_chunks::<16>();
    let (mo_chunks, mo_tail) = mask_out.as_chunks_mut::<16>();

    let mut mse_mask = 0.0f64;
    let mut mse_iw = 0.0f64;

    for (((ac, sc), dc), mo) in a_chunks.iter().zip(s_chunks).zip(d_chunks).zip(mo_chunks) {
        let av = f32x16::from_array_t(token, *ac);
        let sv = f32x16::from_array_t(token, *sc);
        let dv = f32x16::from_array_t(token, *dc);

        let mask = one / kmv.mul_add_relaxed(av, one);
        // IW weight computed inline — bit-identical to `kiv.mul_add(av, one)`
        // produced by `build_weights_and_mse_inner_v4`, but never stored.
        let iw = kiv.mul_add_relaxed(av, one);
        let diff = sv - dv;
        let d2 = diff * diff;
        mse_mask += (d2 * mask).reduce_add() as f64;
        mse_iw += (d2 * iw).reduce_add() as f64;
        mask.store(mo);
    }

    let off = a_chunks.len() * 16;
    for ((i, &a), mo) in a_tail.iter().enumerate().zip(mo_tail.iter_mut()) {
        let j = off + i;
        let mask = 1.0f32 / (1.0f32 + k_mask * a);
        let iw = 1.0f32 + k_iw * a;
        *mo = mask;
        let d = src[j] - dst[j];
        let d2 = d * d;
        mse_mask += (d2 * mask) as f64;
        mse_iw += (d2 * iw) as f64;
    }

    (mse_mask, mse_iw)
}

// --- build_iw_mse_only ---

#[magetypes(v4, v3, neon, wasm128, scalar)]
fn build_iw_mse_only_inner(
    token: Token,
    activity: &[f32],
    k_iw: f32,
    src: &[f32],
    dst: &[f32],
) -> f64 {
    let one = f32x16::splat_t(token, 1.0);
    let kiv = f32x16::splat_t(token, k_iw);
    let (a_chunks, a_tail) = activity.as_chunks::<16>();
    let (s_chunks, _) = src.as_chunks::<16>();
    let (d_chunks, _) = dst.as_chunks::<16>();
    let mut mse = 0.0f64;
    for ((ac, sc), dc) in a_chunks.iter().zip(s_chunks).zip(d_chunks) {
        let av = f32x16::from_array_t(token, *ac);
        let sv = f32x16::from_array_t(token, *sc);
        let dv = f32x16::from_array_t(token, *dc);
        let iw = kiv.mul_add_relaxed(av, one);
        let diff = sv - dv;
        let d2 = diff * diff;
        mse += (d2 * iw).reduce_add() as f64;
    }
    let off = a_chunks.len() * 16;
    for (i, &a) in a_tail.iter().enumerate() {
        let j = off + i;
        let iw = 1.0f32 + k_iw * a;
        let d = src[j] - dst[j];
        mse += (d * d * iw) as f64;
    }
    mse
}

// --- ssim_channel_masked_with_iw_inline ---

#[magetypes(v4, v3, neon, wasm128, scalar)]
fn ssim_channel_masked_with_iw_inline_inner(
    token: Token,
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
    mask: &[f32],
    activity: &[f32],
    k_iw: f32,
) -> ((f64, f64, f64), (f64, f64, f64)) {
    let form = crate::ssim_form::active_luma_form();
    let one = f32x16::splat_t(token, 1.0);
    let zero = f32x16::zero_t(token);
    let kiv = f32x16::splat_t(token, k_iw);

    let (mu1_chunks, mu1_tail) = mu1.as_chunks::<16>();
    let (mu2_chunks, _) = mu2.as_chunks::<16>();
    let (ssq_chunks, _) = sum_sq.as_chunks::<16>();
    let (s12_chunks, _) = s12.as_chunks::<16>();
    let (mask_chunks, _) = mask.as_chunks::<16>();
    let (act_chunks, _) = activity.as_chunks::<16>();

    let mut sum_da = 0.0f64;
    let mut sum_d4a = 0.0f64;
    let mut sum_d2a = 0.0f64;
    let mut sum_db = 0.0f64;
    let mut sum_d4b = 0.0f64;
    let mut sum_d2b = 0.0f64;

    for (((((m1c, m2c), ssqc), s12c), mac), ac) in mu1_chunks
        .iter()
        .zip(mu2_chunks)
        .zip(ssq_chunks)
        .zip(s12_chunks)
        .zip(mask_chunks)
        .zip(act_chunks)
    {
        let m1 = f32x16::from_array_t(token, *m1c);
        let m2 = f32x16::from_array_t(token, *m2c);
        let ssq = f32x16::from_array_t(token, *ssqc);
        let s12v = f32x16::from_array_t(token, *s12c);
        let mva = f32x16::from_array_t(token, *mac);
        let av = f32x16::from_array_t(token, *ac);
        let mvb = kiv.mul_add_relaxed(av, one); // IW weight inline

        let d_raw = ssim_dissim16(token, form, m1, m2, ssq, s12v);

        let da = (d_raw * mva).max(zero);
        let d2a = da * da;
        let d4a = d2a * d2a;
        let db = (d_raw * mvb).max(zero);
        let d2b = db * db;
        let d4b = d2b * d2b;

        sum_da += da.reduce_add() as f64;
        sum_d2a += d2a.reduce_add() as f64;
        sum_d4a += d4a.reduce_add() as f64;
        sum_db += db.reduce_add() as f64;
        sum_d2b += d2b.reduce_add() as f64;
        sum_d4b += d4b.reduce_add() as f64;
    }

    let off = mu1_chunks.len() * 16;
    for (i, &m1v) in mu1_tail.iter().enumerate() {
        let j = off + i;
        let d_raw = ssim_dissim_raw_scalar(form, m1v, mu2[j], sum_sq[j], s12[j]);
        let iw = 1.0f32 + k_iw * activity[j];
        let da = (d_raw * mask[j]).max(0.0f32);
        let d2a = da * da;
        let db = (d_raw * iw).max(0.0f32);
        let d2b = db * db;
        sum_da += da as f64;
        sum_d2a += d2a as f64;
        sum_d4a += (d2a * d2a) as f64;
        sum_db += db as f64;
        sum_d2b += d2b as f64;
        sum_d4b += (d2b * d2b) as f64;
    }

    ((sum_da, sum_d4a, sum_d2a), (sum_db, sum_d4b, sum_d2b))
}

// --- ssim_channel_iw_inline ---

#[magetypes(v4, v3, neon, wasm128, scalar)]
fn ssim_channel_iw_inline_inner(
    token: Token,
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
    activity: &[f32],
    k_iw: f32,
) -> (f64, f64, f64) {
    let form = crate::ssim_form::active_luma_form();
    let one = f32x16::splat_t(token, 1.0);
    let zero = f32x16::zero_t(token);
    let kiv = f32x16::splat_t(token, k_iw);

    let (mu1_chunks, mu1_tail) = mu1.as_chunks::<16>();
    let (mu2_chunks, _) = mu2.as_chunks::<16>();
    let (ssq_chunks, _) = sum_sq.as_chunks::<16>();
    let (s12_chunks, _) = s12.as_chunks::<16>();
    let (act_chunks, _) = activity.as_chunks::<16>();

    let mut sum_d = 0.0f64;
    let mut sum_d4 = 0.0f64;
    let mut sum_d2 = 0.0f64;

    for ((((m1c, m2c), ssqc), s12c), ac) in mu1_chunks
        .iter()
        .zip(mu2_chunks)
        .zip(ssq_chunks)
        .zip(s12_chunks)
        .zip(act_chunks)
    {
        let m1 = f32x16::from_array_t(token, *m1c);
        let m2 = f32x16::from_array_t(token, *m2c);
        let ssq = f32x16::from_array_t(token, *ssqc);
        let s12v = f32x16::from_array_t(token, *s12c);
        let av = f32x16::from_array_t(token, *ac);
        let mv = kiv.mul_add_relaxed(av, one);

        let d = ((ssim_dissim16(token, form, m1, m2, ssq, s12v)) * mv).max(zero);
        let d2 = d * d;
        let d4 = d2 * d2;

        sum_d += d.reduce_add() as f64;
        sum_d2 += d2.reduce_add() as f64;
        sum_d4 += d4.reduce_add() as f64;
    }

    let off = mu1_chunks.len() * 16;
    for (i, &m1v) in mu1_tail.iter().enumerate() {
        let j = off + i;
        let iw = 1.0f32 + k_iw * activity[j];
        let d = ((ssim_dissim_raw_scalar(form, m1v, mu2[j], sum_sq[j], s12[j])) * iw).max(0.0f32);
        let d2 = d * d;
        sum_d += d as f64;
        sum_d2 += d2 as f64;
        sum_d4 += (d2 * d2) as f64;
    }

    (sum_d, sum_d4, sum_d2)
}

// --- edge_diff_channel_masked_with_iw_inline ---

#[magetypes(v4, v3, neon, wasm128, scalar)]
fn edge_diff_channel_masked_with_iw_inline_inner(
    token: Token,
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    mask: &[f32],
    activity: &[f32],
    k_iw: f32,
) -> ((f64, f64), (f64, f64)) {
    let one = f32x16::splat_t(token, 1.0);
    let zero = f32x16::zero_t(token);
    let kiv = f32x16::splat_t(token, k_iw);

    let (i1_chunks, i1_tail) = img1.as_chunks::<16>();
    let (i2_chunks, _) = img2.as_chunks::<16>();
    let (m1_chunks, _) = mu1.as_chunks::<16>();
    let (m2_chunks, _) = mu2.as_chunks::<16>();
    let (mask_chunks, _) = mask.as_chunks::<16>();
    let (act_chunks, _) = activity.as_chunks::<16>();

    let mut sum_art4a = 0.0f64;
    let mut sum_det4a = 0.0f64;
    let mut sum_art4b = 0.0f64;
    let mut sum_det4b = 0.0f64;

    for (((((i1c, i2c), m1c), m2c), mac), ac) in i1_chunks
        .iter()
        .zip(i2_chunks)
        .zip(m1_chunks)
        .zip(m2_chunks)
        .zip(mask_chunks)
        .zip(act_chunks)
    {
        let i1 = f32x16::from_array_t(token, *i1c);
        let i2 = f32x16::from_array_t(token, *i2c);
        let m1 = f32x16::from_array_t(token, *m1c);
        let m2 = f32x16::from_array_t(token, *m2c);
        let mva = f32x16::from_array_t(token, *mac);
        let av = f32x16::from_array_t(token, *ac);
        let mvb = kiv.mul_add_relaxed(av, one);

        let diff1 = (i1 - m1).abs();
        let diff2 = (i2 - m2).abs();
        let d_raw = (one + diff2) / (one + diff1) - one;

        let da = d_raw * mva;
        let artifact_a = da.max(zero);
        let detail_a = (-da).max(zero);
        let a2a = artifact_a * artifact_a;
        let dl2a = detail_a * detail_a;
        sum_art4a += (a2a * a2a).reduce_add() as f64;
        sum_det4a += (dl2a * dl2a).reduce_add() as f64;

        let db = d_raw * mvb;
        let artifact_b = db.max(zero);
        let detail_b = (-db).max(zero);
        let a2b = artifact_b * artifact_b;
        let dl2b = detail_b * detail_b;
        sum_art4b += (a2b * a2b).reduce_add() as f64;
        sum_det4b += (dl2b * dl2b).reduce_add() as f64;
    }

    let off = i1_chunks.len() * 16;
    for (i, _) in i1_tail.iter().enumerate() {
        let j = off + i;
        let diff1 = (img1[j] - mu1[j]).abs();
        let diff2 = (img2[j] - mu2[j]).abs();
        let d_raw = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;
        let iw = 1.0f32 + k_iw * activity[j];

        let da = d_raw * mask[j];
        let aa = da.max(0.0f32);
        let dla = (-da).max(0.0f32);
        let a2a = aa * aa;
        let dl2a = dla * dla;
        sum_art4a += (a2a * a2a) as f64;
        sum_det4a += (dl2a * dl2a) as f64;

        let db = d_raw * iw;
        let ab = db.max(0.0f32);
        let dlb = (-db).max(0.0f32);
        let a2b = ab * ab;
        let dl2b = dlb * dlb;
        sum_art4b += (a2b * a2b) as f64;
        sum_det4b += (dl2b * dl2b) as f64;
    }

    ((sum_art4a, sum_det4a), (sum_art4b, sum_det4b))
}

// --- edge_diff_channel_iw_inline ---

#[magetypes(v4, v3, neon, wasm128, scalar)]
fn edge_diff_channel_iw_inline_inner(
    token: Token,
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    activity: &[f32],
    k_iw: f32,
) -> (f64, f64, f64, f64, f64, f64) {
    let one = f32x16::splat_t(token, 1.0);
    let zero = f32x16::zero_t(token);
    let kiv = f32x16::splat_t(token, k_iw);

    let (i1_chunks, i1_tail) = img1.as_chunks::<16>();
    let (i2_chunks, _) = img2.as_chunks::<16>();
    let (m1_chunks, _) = mu1.as_chunks::<16>();
    let (m2_chunks, _) = mu2.as_chunks::<16>();
    let (act_chunks, _) = activity.as_chunks::<16>();

    let mut sum_art = 0.0f64;
    let mut sum_art4 = 0.0f64;
    let mut sum_art2 = 0.0f64;
    let mut sum_det = 0.0f64;
    let mut sum_det4 = 0.0f64;
    let mut sum_det2 = 0.0f64;

    for ((((i1c, i2c), m1c), m2c), ac) in i1_chunks
        .iter()
        .zip(i2_chunks)
        .zip(m1_chunks)
        .zip(m2_chunks)
        .zip(act_chunks)
    {
        let i1 = f32x16::from_array_t(token, *i1c);
        let i2 = f32x16::from_array_t(token, *i2c);
        let m1 = f32x16::from_array_t(token, *m1c);
        let m2 = f32x16::from_array_t(token, *m2c);
        let av = f32x16::from_array_t(token, *ac);
        let mv = kiv.mul_add_relaxed(av, one);

        let diff1 = (i1 - m1).abs();
        let diff2 = (i2 - m2).abs();
        let d1 = ((one + diff2) / (one + diff1) - one) * mv;

        let artifact = d1.max(zero);
        let detail_lost = (-d1).max(zero);

        let a2 = artifact * artifact;
        let dl2 = detail_lost * detail_lost;

        sum_art += artifact.reduce_add() as f64;
        sum_art2 += a2.reduce_add() as f64;
        sum_art4 += (a2 * a2).reduce_add() as f64;
        sum_det += detail_lost.reduce_add() as f64;
        sum_det2 += dl2.reduce_add() as f64;
        sum_det4 += (dl2 * dl2).reduce_add() as f64;
    }

    let off = i1_chunks.len() * 16;
    for (i, _) in i1_tail.iter().enumerate() {
        let j = off + i;
        let diff1 = (img1[j] - mu1[j]).abs();
        let diff2 = (img2[j] - mu2[j]).abs();
        let iw = 1.0f32 + k_iw * activity[j];
        let d1 = ((1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32) * iw;

        let artifact = d1.max(0.0f32);
        let detail_lost = (-d1).max(0.0f32);
        let a2 = artifact * artifact;
        let dl2 = detail_lost * detail_lost;
        sum_art += artifact as f64;
        sum_art2 += a2 as f64;
        sum_art4 += (a2 * a2) as f64;
        sum_det += detail_lost as f64;
        sum_det2 += dl2 as f64;
        sum_det4 += (dl2 * dl2) as f64;
    }

    (sum_art, sum_art4, sum_det, sum_det4, sum_art2, sum_det2)
}

// === Phase 2 (2026-05-22) — Mask-plane elimination kernels ===
//
// These kernels take the blurred-activity buffer (`activity`) and derive
// BOTH the masked weight (`1 / (1 + k_mask * a)`) AND the IW weight
// (`1 + k_iw * a`) inline in SIMD registers. The mask plane is never
// materialized — saving 1 full plane write (4 MB at 1024²·strip) and
// up to 3 full plane reads per channel per scale.
//
// Drift vs the plane-stored variant: f32 reciprocal differs slightly
// in reduction order. Worst observed relative drift on the byte-exact
// strip gate is well under 1e-6.

/// Masked + IW combined: build neither plane. Reads activity, src, dst.
/// Computes `(1 / (1 + k_mask * a))` and `(1 + k_iw * a)` inline,
/// accumulates `Σ (src-dst)² · mask` AND `Σ (src-dst)² · iw`.
/// Returns `(mse_mask, mse_iw)`.
pub(crate) fn build_inline_mse(
    activity: &[f32],
    k_mask: f32,
    k_iw: f32,
    src: &[f32],
    dst: &[f32],
) -> (f64, f64) {
    incant!(
        build_inline_mse_inner(activity, k_mask, k_iw, src, dst),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// Mask-only: no plane writes. Computes `1 / (1 + k_mask * a)` inline,
/// accumulates `Σ (src-dst)² · mask`. Returns `mse_mask`.
pub(crate) fn build_inline_mask_mse(
    activity: &[f32],
    k_mask: f32,
    src: &[f32],
    dst: &[f32],
) -> f64 {
    incant!(
        build_inline_mask_mse_inner(activity, k_mask, src, dst),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// SSIM both: derive mask + IW weights inline from activity. No mask plane reads.
/// Returns `((masked d/d4/d2), (iw d/d4/d2))`.
pub(crate) fn ssim_channel_inline_both(
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
    activity: &[f32],
    k_mask: f32,
    k_iw: f32,
) -> ((f64, f64, f64), (f64, f64, f64)) {
    incant!(
        ssim_channel_inline_both_inner(mu1, mu2, sum_sq, s12, activity, k_mask, k_iw),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// Weighted masked + IW pools over a RETAINED per-pixel SSIM signal.
///
/// The [`crate::feature_defs::FormulaRevision::Rev3`] counterpart of
/// [`ssim_channel_inline_both`]: the dissimilarity is no longer re-derived
/// from `(mu1, mu2, ssq, s12)` here, because under Rev3 the ONE canonical
/// value was already formed by [`crate::ssim_form::stable_ssim_plane`] and
/// retained. Reductions, weights and accumulation order are the legacy
/// reducer's, so the only difference between the two is where `d_raw` comes
/// from — which is exactly the property the parity controls assert.
///
/// `signal` and `activity` are matching contiguous inner-band pixels.
pub(crate) fn ssim_signal_inline_both(
    signal: &[f32],
    activity: &[f32],
    k_mask: f32,
    k_iw: f32,
) -> ((f64, f64, f64), (f64, f64, f64)) {
    assert_eq!(signal.len(), activity.len());
    incant!(
        ssim_signal_inline_both_inner(signal, activity, k_mask, k_iw),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// Masked-only pool over a retained Rev3 SSIM signal — the
/// [`ssim_channel_inline_mask`] counterpart.
pub(crate) fn ssim_signal_inline_mask(
    signal: &[f32],
    activity: &[f32],
    k_mask: f32,
) -> (f64, f64, f64) {
    assert_eq!(signal.len(), activity.len());
    incant!(
        ssim_signal_inline_mask_inner(signal, activity, k_mask),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// IW-only pool over a retained Rev3 SSIM signal — the
/// [`ssim_channel_iw_inline`] counterpart.
pub(crate) fn ssim_signal_iw_inline(
    signal: &[f32],
    activity: &[f32],
    k_iw: f32,
) -> (f64, f64, f64) {
    assert_eq!(signal.len(), activity.len());
    incant!(
        ssim_signal_iw_inline_inner(signal, activity, k_iw),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// SSIM mask-only: derive mask weight inline. Returns `(sum_d, sum_d4, sum_d2)`.
pub(crate) fn ssim_channel_inline_mask(
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
    activity: &[f32],
    k_mask: f32,
) -> (f64, f64, f64) {
    incant!(
        ssim_channel_inline_mask_inner(mu1, mu2, sum_sq, s12, activity, k_mask),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// Edge both: derive mask + IW weights inline. Returns `((art4_m, det4_m), (art4_i, det4_i))`.
pub(crate) fn edge_diff_channel_inline_both(
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    activity: &[f32],
    k_mask: f32,
    k_iw: f32,
) -> ((f64, f64), (f64, f64)) {
    incant!(
        edge_diff_channel_inline_both_inner(img1, img2, mu1, mu2, activity, k_mask, k_iw),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// Edge mask-only: derive mask weight inline. Returns the full 6-tuple
/// `(art, art4, det, det4, art2, det2)` matching `edge_diff_channel_masked`.
pub(crate) fn edge_diff_channel_inline_mask(
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    activity: &[f32],
    k_mask: f32,
) -> (f64, f64, f64, f64, f64, f64) {
    incant!(
        edge_diff_channel_inline_mask_inner(img1, img2, mu1, mu2, activity, k_mask),
        [v4, v3, neon, wasm128, scalar]
    )
}

// --- build_inline_mse ---

#[magetypes(v4, v3, neon, wasm128, scalar)]
fn build_inline_mse_inner(
    token: Token,
    activity: &[f32],
    k_mask: f32,
    k_iw: f32,
    src: &[f32],
    dst: &[f32],
) -> (f64, f64) {
    let one = f32x16::splat_t(token, 1.0);
    let kmv = f32x16::splat_t(token, k_mask);
    let kiv = f32x16::splat_t(token, k_iw);

    let (a_chunks, a_tail) = activity.as_chunks::<16>();
    let (s_chunks, _) = src.as_chunks::<16>();
    let (d_chunks, _) = dst.as_chunks::<16>();

    let mut mse_mask = 0.0f64;
    let mut mse_iw = 0.0f64;

    for ((ac, sc), dc) in a_chunks.iter().zip(s_chunks).zip(d_chunks) {
        let av = f32x16::from_array_t(token, *ac);
        let sv = f32x16::from_array_t(token, *sc);
        let dv = f32x16::from_array_t(token, *dc);

        let mask = one / kmv.mul_add_relaxed(av, one);
        let iw = kiv.mul_add_relaxed(av, one);
        let diff = sv - dv;
        let d2 = diff * diff;
        mse_mask += (d2 * mask).reduce_add() as f64;
        mse_iw += (d2 * iw).reduce_add() as f64;
    }

    let off = a_chunks.len() * 16;
    for (i, &a) in a_tail.iter().enumerate() {
        let j = off + i;
        let mask = 1.0f32 / (1.0f32 + k_mask * a);
        let iw = 1.0f32 + k_iw * a;
        let d = src[j] - dst[j];
        let d2 = d * d;
        mse_mask += (d2 * mask) as f64;
        mse_iw += (d2 * iw) as f64;
    }

    (mse_mask, mse_iw)
}

// --- build_inline_mask_mse ---

#[magetypes(v4, v3, neon, wasm128, scalar)]
fn build_inline_mask_mse_inner(
    token: Token,
    activity: &[f32],
    k_mask: f32,
    src: &[f32],
    dst: &[f32],
) -> f64 {
    let one = f32x16::splat_t(token, 1.0);
    let kmv = f32x16::splat_t(token, k_mask);

    let (a_chunks, a_tail) = activity.as_chunks::<16>();
    let (s_chunks, _) = src.as_chunks::<16>();
    let (d_chunks, _) = dst.as_chunks::<16>();

    let mut mse_mask = 0.0f64;

    for ((ac, sc), dc) in a_chunks.iter().zip(s_chunks).zip(d_chunks) {
        let av = f32x16::from_array_t(token, *ac);
        let sv = f32x16::from_array_t(token, *sc);
        let dv = f32x16::from_array_t(token, *dc);

        let mask = one / kmv.mul_add_relaxed(av, one);
        let diff = sv - dv;
        let d2 = diff * diff;
        mse_mask += (d2 * mask).reduce_add() as f64;
    }

    let off = a_chunks.len() * 16;
    for (i, &a) in a_tail.iter().enumerate() {
        let j = off + i;
        let mask = 1.0f32 / (1.0f32 + k_mask * a);
        let d = src[j] - dst[j];
        mse_mask += (d * d * mask) as f64;
    }

    mse_mask
}

// --- ssim_channel_inline_both ---

#[magetypes(v4, v3, neon, wasm128, scalar)]
fn ssim_channel_inline_both_inner(
    token: Token,
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
    activity: &[f32],
    k_mask: f32,
    k_iw: f32,
) -> ((f64, f64, f64), (f64, f64, f64)) {
    let form = crate::ssim_form::active_luma_form();
    let one = f32x16::splat_t(token, 1.0);
    let zero = f32x16::zero_t(token);
    let kmv = f32x16::splat_t(token, k_mask);
    let kiv = f32x16::splat_t(token, k_iw);

    let (mu1_chunks, mu1_tail) = mu1.as_chunks::<16>();
    let (mu2_chunks, _) = mu2.as_chunks::<16>();
    let (ssq_chunks, _) = sum_sq.as_chunks::<16>();
    let (s12_chunks, _) = s12.as_chunks::<16>();
    let (act_chunks, _) = activity.as_chunks::<16>();

    let mut sum_da = 0.0f64;
    let mut sum_d4a = 0.0f64;
    let mut sum_d2a = 0.0f64;
    let mut sum_db = 0.0f64;
    let mut sum_d4b = 0.0f64;
    let mut sum_d2b = 0.0f64;

    for ((((m1c, m2c), ssqc), s12c), ac) in mu1_chunks
        .iter()
        .zip(mu2_chunks)
        .zip(ssq_chunks)
        .zip(s12_chunks)
        .zip(act_chunks)
    {
        let m1 = f32x16::from_array_t(token, *m1c);
        let m2 = f32x16::from_array_t(token, *m2c);
        let ssq = f32x16::from_array_t(token, *ssqc);
        let s12v = f32x16::from_array_t(token, *s12c);
        let av = f32x16::from_array_t(token, *ac);

        let mva = one / kmv.mul_add_relaxed(av, one); // mask weight inline
        let mvb = kiv.mul_add_relaxed(av, one); // IW weight inline

        let d_raw = ssim_dissim16(token, form, m1, m2, ssq, s12v);

        let da = (d_raw * mva).max(zero);
        let d2a = da * da;
        let d4a = d2a * d2a;
        let db = (d_raw * mvb).max(zero);
        let d2b = db * db;
        let d4b = d2b * d2b;

        sum_da += da.reduce_add() as f64;
        sum_d2a += d2a.reduce_add() as f64;
        sum_d4a += d4a.reduce_add() as f64;
        sum_db += db.reduce_add() as f64;
        sum_d2b += d2b.reduce_add() as f64;
        sum_d4b += d4b.reduce_add() as f64;
    }

    let off = mu1_chunks.len() * 16;
    for (i, &m1v) in mu1_tail.iter().enumerate() {
        let j = off + i;
        let d_raw = ssim_dissim_raw_scalar(form, m1v, mu2[j], sum_sq[j], s12[j]);
        let mask = 1.0f32 / (1.0f32 + k_mask * activity[j]);
        let iw = 1.0f32 + k_iw * activity[j];
        let da = (d_raw * mask).max(0.0f32);
        let d2a = da * da;
        let db = (d_raw * iw).max(0.0f32);
        let d2b = db * db;
        sum_da += da as f64;
        sum_d2a += d2a as f64;
        sum_d4a += (d2a * d2a) as f64;
        sum_db += db as f64;
        sum_d2b += d2b as f64;
        sum_d4b += (d2b * d2b) as f64;
    }

    ((sum_da, sum_d4a, sum_d2a), (sum_db, sum_d4b, sum_d2b))
}

// --- ssim_channel_inline_mask ---

#[magetypes(v4, v3, neon, wasm128, scalar)]
fn ssim_channel_inline_mask_inner(
    token: Token,
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
    activity: &[f32],
    k_mask: f32,
) -> (f64, f64, f64) {
    let form = crate::ssim_form::active_luma_form();
    let one = f32x16::splat_t(token, 1.0);
    let zero = f32x16::zero_t(token);
    let kmv = f32x16::splat_t(token, k_mask);

    let (mu1_chunks, mu1_tail) = mu1.as_chunks::<16>();
    let (mu2_chunks, _) = mu2.as_chunks::<16>();
    let (ssq_chunks, _) = sum_sq.as_chunks::<16>();
    let (s12_chunks, _) = s12.as_chunks::<16>();
    let (act_chunks, _) = activity.as_chunks::<16>();

    let mut sum_d = 0.0f64;
    let mut sum_d4 = 0.0f64;
    let mut sum_d2 = 0.0f64;

    for ((((m1c, m2c), ssqc), s12c), ac) in mu1_chunks
        .iter()
        .zip(mu2_chunks)
        .zip(ssq_chunks)
        .zip(s12_chunks)
        .zip(act_chunks)
    {
        let m1 = f32x16::from_array_t(token, *m1c);
        let m2 = f32x16::from_array_t(token, *m2c);
        let ssq = f32x16::from_array_t(token, *ssqc);
        let s12v = f32x16::from_array_t(token, *s12c);
        let av = f32x16::from_array_t(token, *ac);
        let mv = one / kmv.mul_add_relaxed(av, one);

        let d = ((ssim_dissim16(token, form, m1, m2, ssq, s12v)) * mv).max(zero);
        let d2 = d * d;
        let d4 = d2 * d2;

        sum_d += d.reduce_add() as f64;
        sum_d2 += d2.reduce_add() as f64;
        sum_d4 += d4.reduce_add() as f64;
    }

    let off = mu1_chunks.len() * 16;
    for (i, &m1v) in mu1_tail.iter().enumerate() {
        let j = off + i;
        let mask = 1.0f32 / (1.0f32 + k_mask * activity[j]);
        let d = ((ssim_dissim_raw_scalar(form, m1v, mu2[j], sum_sq[j], s12[j])) * mask).max(0.0f32);
        let d2 = d * d;
        sum_d += d as f64;
        sum_d2 += d2 as f64;
        sum_d4 += (d2 * d2) as f64;
    }

    (sum_d, sum_d4, sum_d2)
}

// --- Rev3 retained-signal pools ---
//
// One canonical `d_raw` per pixel, formed once by
// `ssim_form::stable_ssim_plane` and retained, replaces the per-pool
// re-derivation from `(mu1, mu2, ssq, s12)`. Everything downstream of
// `d_raw` — the inline mask/IW weights, the `.max(0)` floor, the d/d2/d4
// tiers and the f64 accumulation order — is copied unchanged from the
// legacy reducers above, so a parity test that feeds the legacy `d_raw`
// through these is exact, and the only measured difference is the signal.

#[magetypes(v4, v3, neon, wasm128, scalar)]
fn ssim_signal_inline_both_inner(
    token: Token,
    signal: &[f32],
    activity: &[f32],
    k_mask: f32,
    k_iw: f32,
) -> ((f64, f64, f64), (f64, f64, f64)) {
    let one = f32x16::splat_t(token, 1.0);
    let zero = f32x16::zero_t(token);
    let kmv = f32x16::splat_t(token, k_mask);
    let kiv = f32x16::splat_t(token, k_iw);

    let (sig_chunks, sig_tail) = signal.as_chunks::<16>();
    let (act_chunks, _) = activity.as_chunks::<16>();

    let mut sum_da = 0.0f64;
    let mut sum_d4a = 0.0f64;
    let mut sum_d2a = 0.0f64;
    let mut sum_db = 0.0f64;
    let mut sum_d4b = 0.0f64;
    let mut sum_d2b = 0.0f64;

    for (sc, ac) in sig_chunks.iter().zip(act_chunks) {
        let d_raw = f32x16::from_array_t(token, *sc);
        let av = f32x16::from_array_t(token, *ac);

        let mva = one / kmv.mul_add_relaxed(av, one); // mask weight inline
        let mvb = kiv.mul_add_relaxed(av, one); // IW weight inline

        let da = (d_raw * mva).max(zero);
        let d2a = da * da;
        let d4a = d2a * d2a;
        let db = (d_raw * mvb).max(zero);
        let d2b = db * db;
        let d4b = d2b * d2b;

        sum_da += da.reduce_add() as f64;
        sum_d2a += d2a.reduce_add() as f64;
        sum_d4a += d4a.reduce_add() as f64;
        sum_db += db.reduce_add() as f64;
        sum_d2b += d2b.reduce_add() as f64;
        sum_d4b += d4b.reduce_add() as f64;
    }

    let off = sig_chunks.len() * 16;
    for (i, &d_raw) in sig_tail.iter().enumerate() {
        let j = off + i;
        let mask = 1.0f32 / (1.0f32 + k_mask * activity[j]);
        let iw = 1.0f32 + k_iw * activity[j];
        let da = (d_raw * mask).max(0.0f32);
        let d2a = da * da;
        let db = (d_raw * iw).max(0.0f32);
        let d2b = db * db;
        sum_da += da as f64;
        sum_d2a += d2a as f64;
        sum_d4a += (d2a * d2a) as f64;
        sum_db += db as f64;
        sum_d2b += d2b as f64;
        sum_d4b += (d2b * d2b) as f64;
    }

    ((sum_da, sum_d4a, sum_d2a), (sum_db, sum_d4b, sum_d2b))
}

#[magetypes(v4, v3, neon, wasm128, scalar)]
fn ssim_signal_inline_mask_inner(
    token: Token,
    signal: &[f32],
    activity: &[f32],
    k_mask: f32,
) -> (f64, f64, f64) {
    let one = f32x16::splat_t(token, 1.0);
    let zero = f32x16::zero_t(token);
    let kmv = f32x16::splat_t(token, k_mask);

    let (sig_chunks, sig_tail) = signal.as_chunks::<16>();
    let (act_chunks, _) = activity.as_chunks::<16>();

    let mut sum_d = 0.0f64;
    let mut sum_d4 = 0.0f64;
    let mut sum_d2 = 0.0f64;

    for (sc, ac) in sig_chunks.iter().zip(act_chunks) {
        let av = f32x16::from_array_t(token, *ac);
        let mv = one / kmv.mul_add_relaxed(av, one);

        let d = (f32x16::from_array_t(token, *sc) * mv).max(zero);
        let d2 = d * d;
        let d4 = d2 * d2;

        sum_d += d.reduce_add() as f64;
        sum_d2 += d2.reduce_add() as f64;
        sum_d4 += d4.reduce_add() as f64;
    }

    let off = sig_chunks.len() * 16;
    for (i, &d_raw) in sig_tail.iter().enumerate() {
        let j = off + i;
        let mask = 1.0f32 / (1.0f32 + k_mask * activity[j]);
        let d = (d_raw * mask).max(0.0f32);
        let d2 = d * d;
        sum_d += d as f64;
        sum_d2 += d2 as f64;
        sum_d4 += (d2 * d2) as f64;
    }

    (sum_d, sum_d4, sum_d2)
}

#[magetypes(v4, v3, neon, wasm128, scalar)]
fn ssim_signal_iw_inline_inner(
    token: Token,
    signal: &[f32],
    activity: &[f32],
    k_iw: f32,
) -> (f64, f64, f64) {
    let one = f32x16::splat_t(token, 1.0);
    let zero = f32x16::zero_t(token);
    let kiv = f32x16::splat_t(token, k_iw);

    let (sig_chunks, sig_tail) = signal.as_chunks::<16>();
    let (act_chunks, _) = activity.as_chunks::<16>();

    let mut sum_d = 0.0f64;
    let mut sum_d4 = 0.0f64;
    let mut sum_d2 = 0.0f64;

    for (sc, ac) in sig_chunks.iter().zip(act_chunks) {
        let av = f32x16::from_array_t(token, *ac);
        let wv = kiv.mul_add_relaxed(av, one);

        let d = (f32x16::from_array_t(token, *sc) * wv).max(zero);
        let d2 = d * d;
        let d4 = d2 * d2;

        sum_d += d.reduce_add() as f64;
        sum_d2 += d2.reduce_add() as f64;
        sum_d4 += d4.reduce_add() as f64;
    }

    let off = sig_chunks.len() * 16;
    for (i, &d_raw) in sig_tail.iter().enumerate() {
        let j = off + i;
        let iw = 1.0f32 + k_iw * activity[j];
        let d = (d_raw * iw).max(0.0f32);
        let d2 = d * d;
        sum_d += d as f64;
        sum_d2 += d2 as f64;
        sum_d4 += (d2 * d2) as f64;
    }

    (sum_d, sum_d4, sum_d2)
}

// --- edge_diff_channel_inline_both ---

#[magetypes(v4, v3, neon, wasm128, scalar)]
fn edge_diff_channel_inline_both_inner(
    token: Token,
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    activity: &[f32],
    k_mask: f32,
    k_iw: f32,
) -> ((f64, f64), (f64, f64)) {
    let one = f32x16::splat_t(token, 1.0);
    let zero = f32x16::zero_t(token);
    let kmv = f32x16::splat_t(token, k_mask);
    let kiv = f32x16::splat_t(token, k_iw);

    let (i1_chunks, i1_tail) = img1.as_chunks::<16>();
    let (i2_chunks, _) = img2.as_chunks::<16>();
    let (m1_chunks, _) = mu1.as_chunks::<16>();
    let (m2_chunks, _) = mu2.as_chunks::<16>();
    let (act_chunks, _) = activity.as_chunks::<16>();

    let mut sum_art4a = 0.0f64;
    let mut sum_det4a = 0.0f64;
    let mut sum_art4b = 0.0f64;
    let mut sum_det4b = 0.0f64;

    for ((((i1c, i2c), m1c), m2c), ac) in i1_chunks
        .iter()
        .zip(i2_chunks)
        .zip(m1_chunks)
        .zip(m2_chunks)
        .zip(act_chunks)
    {
        let i1 = f32x16::from_array_t(token, *i1c);
        let i2 = f32x16::from_array_t(token, *i2c);
        let m1 = f32x16::from_array_t(token, *m1c);
        let m2 = f32x16::from_array_t(token, *m2c);
        let av = f32x16::from_array_t(token, *ac);

        let mva = one / kmv.mul_add_relaxed(av, one);
        let mvb = kiv.mul_add_relaxed(av, one);

        let diff1 = (i1 - m1).abs();
        let diff2 = (i2 - m2).abs();
        let d_raw = (one + diff2) / (one + diff1) - one;

        let da = d_raw * mva;
        let artifact_a = da.max(zero);
        let detail_a = (-da).max(zero);
        let a2a = artifact_a * artifact_a;
        let dl2a = detail_a * detail_a;
        sum_art4a += (a2a * a2a).reduce_add() as f64;
        sum_det4a += (dl2a * dl2a).reduce_add() as f64;

        let db = d_raw * mvb;
        let artifact_b = db.max(zero);
        let detail_b = (-db).max(zero);
        let a2b = artifact_b * artifact_b;
        let dl2b = detail_b * detail_b;
        sum_art4b += (a2b * a2b).reduce_add() as f64;
        sum_det4b += (dl2b * dl2b).reduce_add() as f64;
    }

    let off = i1_chunks.len() * 16;
    for (i, _) in i1_tail.iter().enumerate() {
        let j = off + i;
        let diff1 = (img1[j] - mu1[j]).abs();
        let diff2 = (img2[j] - mu2[j]).abs();
        let d_raw = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;
        let mask = 1.0f32 / (1.0f32 + k_mask * activity[j]);
        let iw = 1.0f32 + k_iw * activity[j];

        let da = d_raw * mask;
        let aa = da.max(0.0f32);
        let dla = (-da).max(0.0f32);
        let a2a = aa * aa;
        let dl2a = dla * dla;
        sum_art4a += (a2a * a2a) as f64;
        sum_det4a += (dl2a * dl2a) as f64;

        let db = d_raw * iw;
        let ab = db.max(0.0f32);
        let dlb = (-db).max(0.0f32);
        let a2b = ab * ab;
        let dl2b = dlb * dlb;
        sum_art4b += (a2b * a2b) as f64;
        sum_det4b += (dl2b * dl2b) as f64;
    }

    ((sum_art4a, sum_det4a), (sum_art4b, sum_det4b))
}

// --- edge_diff_channel_inline_mask ---

#[magetypes(v4, v3, neon, wasm128, scalar)]
fn edge_diff_channel_inline_mask_inner(
    token: Token,
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    activity: &[f32],
    k_mask: f32,
) -> (f64, f64, f64, f64, f64, f64) {
    let one = f32x16::splat_t(token, 1.0);
    let zero = f32x16::zero_t(token);
    let kmv = f32x16::splat_t(token, k_mask);

    let (i1_chunks, i1_tail) = img1.as_chunks::<16>();
    let (i2_chunks, _) = img2.as_chunks::<16>();
    let (m1_chunks, _) = mu1.as_chunks::<16>();
    let (m2_chunks, _) = mu2.as_chunks::<16>();
    let (act_chunks, _) = activity.as_chunks::<16>();

    let mut sum_art = 0.0f64;
    let mut sum_art4 = 0.0f64;
    let mut sum_art2 = 0.0f64;
    let mut sum_det = 0.0f64;
    let mut sum_det4 = 0.0f64;
    let mut sum_det2 = 0.0f64;

    for ((((i1c, i2c), m1c), m2c), ac) in i1_chunks
        .iter()
        .zip(i2_chunks)
        .zip(m1_chunks)
        .zip(m2_chunks)
        .zip(act_chunks)
    {
        let i1 = f32x16::from_array_t(token, *i1c);
        let i2 = f32x16::from_array_t(token, *i2c);
        let m1 = f32x16::from_array_t(token, *m1c);
        let m2 = f32x16::from_array_t(token, *m2c);
        let av = f32x16::from_array_t(token, *ac);
        let mv = one / kmv.mul_add_relaxed(av, one);

        let diff1 = (i1 - m1).abs();
        let diff2 = (i2 - m2).abs();
        let d1 = ((one + diff2) / (one + diff1) - one) * mv;

        let artifact = d1.max(zero);
        let detail_lost = (-d1).max(zero);

        let a2 = artifact * artifact;
        let dl2 = detail_lost * detail_lost;

        sum_art += artifact.reduce_add() as f64;
        sum_art2 += a2.reduce_add() as f64;
        sum_art4 += (a2 * a2).reduce_add() as f64;
        sum_det += detail_lost.reduce_add() as f64;
        sum_det2 += dl2.reduce_add() as f64;
        sum_det4 += (dl2 * dl2).reduce_add() as f64;
    }

    let off = i1_chunks.len() * 16;
    for (i, _) in i1_tail.iter().enumerate() {
        let j = off + i;
        let diff1 = (img1[j] - mu1[j]).abs();
        let diff2 = (img2[j] - mu2[j]).abs();
        let mask = 1.0f32 / (1.0f32 + k_mask * activity[j]);
        let d1 = ((1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32) * mask;

        let artifact = d1.max(0.0f32);
        let detail_lost = (-d1).max(0.0f32);
        let a2 = artifact * artifact;
        let dl2 = detail_lost * detail_lost;
        sum_art += artifact as f64;
        sum_art2 += a2 as f64;
        sum_art4 += (a2 * a2) as f64;
        sum_det += detail_lost as f64;
        sum_det2 += dl2 as f64;
        sum_det4 += (dl2 * dl2) as f64;
    }

    (sum_art, sum_art4, sum_det, sum_det4, sum_art2, sum_det2)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(mu1, mu2, sum_sq, s12, mask_a, mask_b)` synthetic plane stats.
    type TestData = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

    fn mk_test_data(n: usize) -> TestData {
        let mut mu1 = Vec::with_capacity(n);
        let mut mu2 = Vec::with_capacity(n);
        let mut sum_sq = Vec::with_capacity(n);
        let mut s12 = Vec::with_capacity(n);
        let mut mask_a = Vec::with_capacity(n);
        let mut mask_b = Vec::with_capacity(n);
        for i in 0..n {
            let x = (i as f32) / (n as f32);
            mu1.push(0.5 + 0.1 * x);
            mu2.push(0.5 + 0.1 * x + 0.01 * (x - 0.5));
            sum_sq.push(0.25 + 0.02 * x);
            s12.push(0.24 + 0.02 * x);
            mask_a.push(0.5 + 0.5 * x); // smooth mask in [0.5, 1.0]
            mask_b.push(1.0 + 2.0 * x); // IW-style weight in [1.0, 3.0]
        }
        (mu1, mu2, sum_sq, s12, mask_a, mask_b)
    }

    /// Activity plane for the Rev3 pool controls: `mask_a` is already a
    /// smooth `[0.5, 1.0]` ramp, which is exactly the shape `activity` takes.
    fn mk_activity(n: usize) -> Vec<f32> {
        (0..n).map(|i| 0.75 * (i as f32) / (n as f32)).collect()
    }

    /// **The Rev3 pool contract, stated as a test.** Feed the legacy reducer
    /// and its retained-signal counterpart the SAME `d_raw` and they must
    /// agree — because the ONLY thing Rev3 changes is where `d_raw` comes
    /// from, never the weights, the floor, the tiers or the accumulation.
    ///
    /// `d_raw` is taken from `ssim_dissim_raw_scalar`, the very expression
    /// the legacy reducer's scalar tail uses, so any disagreement beyond
    /// SIMD-vs-scalar rounding of that one expression is a defect in the new
    /// reducers. The bound is relative and tight; the residual is the
    /// `mul_add` contraction difference between the vector and scalar
    /// spellings of `ssim_dissim`, not a difference in pooling.
    #[test]
    fn retained_signal_pools_agree_with_the_legacy_pools_on_the_same_d_raw() {
        let form = crate::ssim_form::active_luma_form();
        for &n in &[1usize, 15, 16, 17, 31, 32, 33, 100, 256, 1000, 1024] {
            let (mu1, mu2, sum_sq, s12, _, _) = mk_test_data(n);
            let act = mk_activity(n);
            let signal: Vec<f32> = (0..n)
                .map(|j| ssim_dissim_raw_scalar(form, mu1[j], mu2[j], sum_sq[j], s12[j]).max(0.0))
                .collect();

            let close = |got: f64, want: f64, what: &str| {
                let tol = 1e-12 + 1e-6 * want.abs();
                assert!(
                    (got - want).abs() <= tol,
                    "n={n} {what}: retained {got} vs legacy {want}"
                );
            };

            let ((m, m4, m2), (i, i4, i2)) =
                ssim_channel_inline_both(&mu1, &mu2, &sum_sq, &s12, &act, 4.0, 4.0);
            let ((sm, sm4, sm2), (si, si4, si2)) = ssim_signal_inline_both(&signal, &act, 4.0, 4.0);
            close(sm, m, "both/masked d");
            close(sm4, m4, "both/masked d4");
            close(sm2, m2, "both/masked d2");
            close(si, i, "both/iw d");
            close(si4, i4, "both/iw d4");
            close(si2, i2, "both/iw d2");

            let (mm, mm4, mm2) = ssim_channel_inline_mask(&mu1, &mu2, &sum_sq, &s12, &act, 4.0);
            let (sd, sd4, sd2) = ssim_signal_inline_mask(&signal, &act, 4.0);
            close(sd, mm, "mask-only d");
            close(sd4, mm4, "mask-only d4");
            close(sd2, mm2, "mask-only d2");

            let (wi, wi4, wi2) = ssim_channel_iw_inline(&mu1, &mu2, &sum_sq, &s12, &act, 4.0);
            let (sw, sw4, sw2) = ssim_signal_iw_inline(&signal, &act, 4.0);
            close(sw, wi, "iw-only d");
            close(sw4, wi4, "iw-only d4");
            close(sw2, wi2, "iw-only d2");
        }
    }

    /// The retained-signal reducers against a plain scalar reference, so
    /// their weights and tiers are pinned independently of the legacy
    /// reducers they were derived from. Also covers the vector/tail split at
    /// every residue mod 16 and the single-arm reducers' agreement with the
    /// combined one.
    ///
    /// The bound is f32-relative (`1e-6`), NOT bit-exact, and deliberately:
    /// the reducers' vector body forms the weight with `mul_add` (one
    /// rounding) while their scalar tail spells `1 + k*a` (two) — a
    /// pre-existing property they inherit unchanged from the legacy trio, so
    /// no single scalar reference can be bit-exact against both halves. The
    /// bound is still ~5 orders of magnitude tighter than any real pooling
    /// error: dropping one tail element of a 17-element band moves the sum
    /// ~6 %, and a wrong tier or weight moves it further.
    #[test]
    fn retained_signal_pools_match_a_direct_scalar_reference() {
        for &n in &[1usize, 7, 16, 23, 48, 64, 129, 512] {
            let signal: Vec<f32> = (0..n)
                .map(|i| (i as f32 * 0.013).sin().abs() * 0.4)
                .collect();
            let act = mk_activity(n);
            let (k, k_iw) = (4.0f32, 4.0f32);

            let (mut rm, mut rm4, mut rm2) = (0.0f64, 0.0f64, 0.0f64);
            let (mut ri, mut ri4, mut ri2) = (0.0f64, 0.0f64, 0.0f64);
            for j in 0..n {
                // Spelled exactly as the reducers spell it: the mask weight
                // is formed first and MULTIPLIED in. `signal / (1 + k*a)`
                // is the same value in exact arithmetic but a different f32
                // rounding, and this test is about the pooling, not about
                // which of two equivalent divisions rounds where.
                let wm = 1.0f32 / (1.0 + k * act[j]);
                let wi = 1.0f32 + k_iw * act[j];
                let dm = (signal[j] * wm).max(0.0);
                let di = (signal[j] * wi).max(0.0);
                rm += dm as f64;
                rm2 += (dm * dm) as f64;
                rm4 += (dm * dm * dm * dm) as f64;
                ri += di as f64;
                ri2 += (di * di) as f64;
                ri4 += (di * di * di * di) as f64;
            }

            let close = |got: f64, want: f64, what: &str| {
                let tol = 1e-12 + 1e-6 * want.abs();
                assert!(
                    (got - want).abs() <= tol,
                    "n={n} {what}: got {got} want {want}"
                );
            };

            let ((m, m4, m2), (i, i4, i2)) = ssim_signal_inline_both(&signal, &act, k, k_iw);
            close(m, rm, "masked d");
            close(m4, rm4, "masked d4");
            close(m2, rm2, "masked d2");
            close(i, ri, "iw d");
            close(i4, ri4, "iw d4");
            close(i2, ri2, "iw d2");

            // The single-arm reducers must equal the combined one exactly:
            // same lanes, same order, same expression.
            let (om, om4, om2) = ssim_signal_inline_mask(&signal, &act, k);
            assert_eq!(
                (om.to_bits(), om4.to_bits(), om2.to_bits()),
                (m.to_bits(), m4.to_bits(), m2.to_bits()),
                "n={n} mask-only differs from the combined masked half"
            );
            let (oi, oi4, oi2) = ssim_signal_iw_inline(&signal, &act, k_iw);
            assert_eq!(
                (oi.to_bits(), oi4.to_bits(), oi2.to_bits()),
                (i.to_bits(), i4.to_bits(), i2.to_bits()),
                "n={n} iw-only differs from the combined IW half"
            );
        }
    }

    #[test]
    fn ssim_channel_masked_2_matches_two_single_calls() {
        for &n in &[16usize, 17, 32, 100, 256, 1024] {
            let (mu1, mu2, sum_sq, s12, mask_a, mask_b) = mk_test_data(n);
            let (da_ref, d4a_ref, d2a_ref) =
                ssim_channel_masked(&mu1, &mu2, &sum_sq, &s12, &mask_a);
            let (db_ref, d4b_ref, d2b_ref) =
                ssim_channel_masked(&mu1, &mu2, &sum_sq, &s12, &mask_b);
            let ((da, d4a, d2a), (db, d4b, d2b)) =
                ssim_channel_masked_2(&mu1, &mu2, &sum_sq, &s12, &mask_a, &mask_b);
            let rel = |a: f64, b: f64| (a - b).abs() / a.abs().max(1e-9);
            assert!(
                rel(da, da_ref) < 1e-5,
                "n={}: da {} vs ref {}",
                n,
                da,
                da_ref
            );
            assert!(
                rel(d4a, d4a_ref) < 1e-5,
                "n={}: d4a {} vs ref {}",
                n,
                d4a,
                d4a_ref
            );
            assert!(
                rel(d2a, d2a_ref) < 1e-5,
                "n={}: d2a {} vs ref {}",
                n,
                d2a,
                d2a_ref
            );
            assert!(
                rel(db, db_ref) < 1e-5,
                "n={}: db {} vs ref {}",
                n,
                db,
                db_ref
            );
            assert!(
                rel(d4b, d4b_ref) < 1e-5,
                "n={}: d4b {} vs ref {}",
                n,
                d4b,
                d4b_ref
            );
            assert!(
                rel(d2b, d2b_ref) < 1e-5,
                "n={}: d2b {} vs ref {}",
                n,
                d2b,
                d2b_ref
            );
        }
    }

    #[test]
    fn edge_diff_channel_masked_2_matches_two_single_calls() {
        for &n in &[16usize, 17, 32, 100, 256, 1024] {
            let mut img1 = Vec::with_capacity(n);
            let mut img2 = Vec::with_capacity(n);
            let mut mu1 = Vec::with_capacity(n);
            let mut mu2 = Vec::with_capacity(n);
            let mut mask_a = Vec::with_capacity(n);
            let mut mask_b = Vec::with_capacity(n);
            for i in 0..n {
                let x = (i as f32) / (n as f32);
                img1.push(0.5 + 0.05 * (x * 7.0).sin());
                img2.push(0.5 + 0.05 * (x * 7.0).sin() + 0.01 * x);
                mu1.push(0.5 + 0.1 * x);
                mu2.push(0.5 + 0.1 * x + 0.01 * (x - 0.5));
                mask_a.push(0.5 + 0.5 * x);
                mask_b.push(1.0 + 2.0 * x);
            }
            let (_, art4_a_ref, _, det4_a_ref, _, _) =
                edge_diff_channel_masked(&img1, &img2, &mu1, &mu2, &mask_a);
            let (_, art4_b_ref, _, det4_b_ref, _, _) =
                edge_diff_channel_masked(&img1, &img2, &mu1, &mu2, &mask_b);
            let ((art4_a, det4_a), (art4_b, det4_b)) =
                edge_diff_channel_masked_2_art4_det4(&img1, &img2, &mu1, &mu2, &mask_a, &mask_b);
            let rel = |a: f64, b: f64| (a - b).abs() / a.abs().max(1e-9);
            assert!(
                rel(art4_a, art4_a_ref) < 1e-5,
                "n={}: art4_a {} vs ref {}",
                n,
                art4_a,
                art4_a_ref
            );
            assert!(
                rel(det4_a, det4_a_ref) < 1e-5,
                "n={}: det4_a {} vs ref {}",
                n,
                det4_a,
                det4_a_ref
            );
            assert!(
                rel(art4_b, art4_b_ref) < 1e-5,
                "n={}: art4_b {} vs ref {}",
                n,
                art4_b,
                art4_b_ref
            );
            assert!(
                rel(det4_b, det4_b_ref) < 1e-5,
                "n={}: det4_b {} vs ref {}",
                n,
                det4_b,
                det4_b_ref
            );
        }
    }

    #[test]
    fn build_weights_and_mse_matches_scalar() {
        for &n in &[16usize, 17, 32, 100, 256, 1024] {
            let mut activity = Vec::with_capacity(n);
            let mut src = Vec::with_capacity(n);
            let mut dst = Vec::with_capacity(n);
            for i in 0..n {
                let x = (i as f32) / (n as f32);
                activity.push(x * 0.5);
                src.push(0.5 + 0.05 * x);
                dst.push(0.5 + 0.05 * x + 0.01 * (x - 0.5));
            }
            let k_mask = 4.0f32;
            let k_iw = 4.0f32;
            // Scalar reference
            let mut mask_ref = vec![0f32; n];
            let mut iw_ref = vec![0f32; n];
            let mut mse_mask_ref = 0.0f64;
            let mut mse_iw_ref = 0.0f64;
            for i in 0..n {
                let a = activity[i];
                mask_ref[i] = 1.0 / (1.0 + k_mask * a);
                iw_ref[i] = 1.0 + k_iw * a;
                let d = src[i] - dst[i];
                mse_mask_ref += (d * d * mask_ref[i]) as f64;
                mse_iw_ref += (d * d * iw_ref[i]) as f64;
            }
            // SIMD path
            let mut mask_out = vec![0f32; n];
            let mut iw_out = vec![0f32; n];
            let (mse_mask, mse_iw) = build_weights_and_mse(
                &activity,
                k_mask,
                k_iw,
                &src,
                &dst,
                &mut mask_out,
                &mut iw_out,
            );
            let rel = |a: f64, b: f64| (a - b).abs() / a.abs().max(1e-9);
            for i in 0..n {
                assert!(
                    (mask_out[i] - mask_ref[i]).abs() < 1e-6,
                    "n={}: mask[{}] {} vs ref {}",
                    n,
                    i,
                    mask_out[i],
                    mask_ref[i]
                );
                assert!(
                    (iw_out[i] - iw_ref[i]).abs() < 1e-6,
                    "n={}: iw[{}] {} vs ref {}",
                    n,
                    i,
                    iw_out[i],
                    iw_ref[i]
                );
            }
            assert!(
                rel(mse_mask, mse_mask_ref) < 1e-5,
                "n={}: mse_mask {} vs ref {}",
                n,
                mse_mask,
                mse_mask_ref
            );
            assert!(
                rel(mse_iw, mse_iw_ref) < 1e-5,
                "n={}: mse_iw {} vs ref {}",
                n,
                mse_iw,
                mse_iw_ref
            );
        }
    }

    #[test]
    fn build_iw_weight_and_mse_matches_scalar() {
        for &n in &[16usize, 17, 32, 100, 256, 1024] {
            let mut activity = Vec::with_capacity(n);
            let mut src = Vec::with_capacity(n);
            let mut dst = Vec::with_capacity(n);
            for i in 0..n {
                let x = (i as f32) / (n as f32);
                activity.push(x * 0.5);
                src.push(0.5 + 0.05 * x);
                dst.push(0.5 + 0.05 * x + 0.01 * (x - 0.5));
            }
            let k_iw = 4.0f32;
            // Scalar reference
            let mut iw_ref = vec![0f32; n];
            let mut mse_iw_ref = 0.0f64;
            for i in 0..n {
                let a = activity[i];
                iw_ref[i] = 1.0 + k_iw * a;
                let d = src[i] - dst[i];
                mse_iw_ref += (d * d * iw_ref[i]) as f64;
            }
            let mut iw_out = vec![0f32; n];
            let mse_iw = build_iw_weight_and_mse(&activity, k_iw, &src, &dst, &mut iw_out);
            let rel = |a: f64, b: f64| (a - b).abs() / a.abs().max(1e-9);
            for i in 0..n {
                assert!(
                    (iw_out[i] - iw_ref[i]).abs() < 1e-6,
                    "n={}: iw[{}] {} vs ref {}",
                    n,
                    i,
                    iw_out[i],
                    iw_ref[i]
                );
            }
            assert!(
                rel(mse_iw, mse_iw_ref) < 1e-5,
                "n={}: mse_iw {} vs ref {}",
                n,
                mse_iw,
                mse_iw_ref
            );
        }
    }
}
