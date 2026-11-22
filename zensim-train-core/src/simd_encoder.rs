//! f32 SIMD-accelerated encoder kernels.
//!
//! Mirrors `simd_encoder.rs` but operates on f32 slices, doubling the
//! SIMD lane count: f32x16 on AVX-512 (vs f64x8), f32x8 on AVX2 (vs
//! f64x4). Expected ~2× throughput on compute-bound inner loops.
//!
//! The f32 precision is sufficient for the encoder forward/backward:
//! input features are standardized (mean=0, std=1) and the bake output
//! is already f32. The only precision-sensitive operation is the Adam
//! optimizer (momentum/variance accumulators), which can stay f64 if
//! needed via mixed-precision training.

use archmage::{incant, magetypes};
use magetypes::simd::generic::f32x8 as GenericF32x8;
#[cfg(target_arch = "x86_64")] // only the AVX-512 (`X64V4Token`) kernels are 16-wide
use magetypes::simd::generic::f32x16 as GenericF32x16;

/// f32 encoder forward: same semantics as the f64 version.
#[inline]
pub fn encoder_forward_f32(
    x: &[f32],
    w1: &[f32],
    b1: &[f32],
    n_features: usize,
    n_hidden: usize,
    leaky_alpha: f32,
) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(x.len(), n_features);
    debug_assert_eq!(b1.len(), n_hidden);
    debug_assert_eq!(w1.len(), n_features * n_hidden);

    let mut h_pre = b1.to_vec();
    incant!(accumulate_rows_f32(x, w1, &mut h_pre, n_features, n_hidden), [+v4]);
    let h = incant!(apply_leaky_relu_f32(&h_pre, leaky_alpha), [+v4]);
    (h_pre, h)
}

/// f32 dot product with bias.
#[inline]
pub fn dot_bias_f32(h: &[f32], w: &[f32], bias: f32) -> f32 {
    debug_assert_eq!(h.len(), w.len());
    bias + incant!(dot_product_f32(h, w), [+v4])
}

// =============================================================================
// AVX-512 path — f32x16 (16 lanes per instruction)
// =============================================================================

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn accumulate_rows_f32_v4(
    token: archmage::X64V4Token,
    x: &[f32],
    w1: &[f32],
    h_pre: &mut [f32],
    n_features: usize,
    n_hidden: usize,
) {
    // BIT-PARITY RULE (tier-parity lane): the AVX2 (`_v3`) kernel's fused
    // domain is lanes j < n8 = n_hidden - n_hidden%8; the tail is scalar
    // mul+add. The 16-lane loop leaves n_hidden%16 ∈ [0,15) elements; when
    // it is >= 8 the first eight run ONE f32x8 fused group (same
    // single-rounding mul_add the AVX2 lanes use), and only the final
    // n_hidden%8 take mul+add.
    let chunks = n_hidden / 16;
    let tail = chunks * 16;
    let n8 = n_hidden - (n_hidden % 8);
    let mid_fused = tail + 8 <= n8;
    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 {
            continue;
        }
        let s_v = GenericF32x16::splat_t(token, s);
        let row = &w1[i * n_hidden..(i + 1) * n_hidden];
        for c in 0..chunks {
            let off = c * 16;
            let h_block: &mut [f32; 16] = (&mut h_pre[off..off + 16]).try_into().unwrap();
            let w_block: &[f32; 16] = (&row[off..off + 16]).try_into().unwrap();
            let h_v = GenericF32x16::load_t(token, h_block);
            let w_v = GenericF32x16::load_t(token, w_block);
            s_v.mul_add_relaxed(w_v, h_v).store(h_block);
        }
        if mid_fused {
            let s8 = GenericF32x8::splat_t(token, s);
            let h_block: &mut [f32; 8] = (&mut h_pre[tail..tail + 8]).try_into().unwrap();
            let w_block: &[f32; 8] = (&row[tail..tail + 8]).try_into().unwrap();
            let h_v = GenericF32x8::load_t(token, h_block);
            let w_v = GenericF32x8::load_t(token, w_block);
            s8.mul_add_relaxed(w_v, h_v).store(h_block);
        }
        for j in n8..n_hidden {
            h_pre[j] += s * row[j];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn apply_leaky_relu_f32_v4(
    token: archmage::X64V4Token,
    h_pre: &[f32],
    leaky_alpha: f32,
) -> Vec<f32> {
    let mut h = vec![0.0f32; h_pre.len()];
    let chunks = h_pre.len() / 16;
    let tail = chunks * 16;
    let leaky_v = GenericF32x16::splat_t(token, leaky_alpha);
    let zero_v = GenericF32x16::zero_t(token);
    for c in 0..chunks {
        let off = c * 16;
        let in_block: &[f32; 16] = (&h_pre[off..off + 16]).try_into().unwrap();
        let out_block: &mut [f32; 16] = (&mut h[off..off + 16]).try_into().unwrap();
        let v = GenericF32x16::load_t(token, in_block);
        let scaled = v * leaky_v;
        let neg_mask = v.simd_lt(zero_v);
        GenericF32x16::blend(neg_mask, scaled, v).store(out_block);
    }
    for j in tail..h_pre.len() {
        let v = h_pre[j];
        h[j] = if v >= 0.0 { v } else { leaky_alpha * v };
    }
    h
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn dot_product_f32_v4(token: archmage::X64V4Token, h: &[f32], w: &[f32]) -> f32 {
    // BIT-PARITY RULE (tier-parity lane): reproduce the AVX2 (`_v3`)
    // kernel exactly — an 8-lane accumulator where lane k sums
    // h[8c+k]*w[8c+k] fused, the pairwise tree
    // ((l0+l1)+(l2+l3))+((l4+l5)+(l6+l7)), and the sequential mul+add
    // tail over the last len%8 elements. A 16-lane accumulator is a
    // different summation order and is out of parity. Each 16-element
    // group therefore feeds TWO f32x8 fused mul_adds into the same
    // 8-lane accumulator, preserving per-lane chain order.
    let chunks16 = h.len() / 16;
    let n8 = h.len() - (h.len() % 8);
    let mut acc = GenericF32x8::zero_t(token);
    for c in 0..chunks16 {
        let off = c * 16;
        let lo_h: &[f32; 8] = (&h[off..off + 8]).try_into().unwrap();
        let hi_h: &[f32; 8] = (&h[off + 8..off + 16]).try_into().unwrap();
        let lo_w: &[f32; 8] = (&w[off..off + 8]).try_into().unwrap();
        let hi_w: &[f32; 8] = (&w[off + 8..off + 16]).try_into().unwrap();
        let h_lo = GenericF32x8::load_t(token, lo_h);
        let w_lo = GenericF32x8::load_t(token, lo_w);
        acc = h_lo.mul_add_relaxed(w_lo, acc);
        let h_hi = GenericF32x8::load_t(token, hi_h);
        let w_hi = GenericF32x8::load_t(token, hi_w);
        acc = h_hi.mul_add_relaxed(w_hi, acc);
    }
    // Remaining 8-lane chunk when len%16 >= 8 (fused domain [16k, n8)).
    if chunks16 * 16 + 8 <= n8 {
        let off = chunks16 * 16;
        let h_block: &[f32; 8] = (&h[off..off + 8]).try_into().unwrap();
        let w_block: &[f32; 8] = (&w[off..off + 8]).try_into().unwrap();
        let h_v = GenericF32x8::load_t(token, h_block);
        let w_v = GenericF32x8::load_t(token, w_block);
        acc = h_v.mul_add_relaxed(w_v, acc);
    }
    let mut lanes = [0.0f32; 8];
    acc.store(&mut lanes);
    let lane_sum = (lanes[0] + lanes[1])
        + (lanes[2] + lanes[3])
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    let mut tail_sum = 0.0f32;
    for j in n8..h.len() {
        tail_sum += h[j] * w[j];
    }
    lane_sum + tail_sum
}

// =============================================================================
// AVX2 path — f32x8
// =============================================================================

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn accumulate_rows_f32_v3(
    token: archmage::X64V3Token,
    x: &[f32],
    w1: &[f32],
    h_pre: &mut [f32],
    n_features: usize,
    n_hidden: usize,
) {
    let chunks = n_hidden / 8;
    let tail = chunks * 8;
    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 {
            continue;
        }
        let s_v = GenericF32x8::splat_t(token, s);
        let row = &w1[i * n_hidden..(i + 1) * n_hidden];
        for c in 0..chunks {
            let off = c * 8;
            let h_block: &mut [f32; 8] = (&mut h_pre[off..off + 8]).try_into().unwrap();
            let w_block: &[f32; 8] = (&row[off..off + 8]).try_into().unwrap();
            let h_v = GenericF32x8::load_t(token, h_block);
            let w_v = GenericF32x8::load_t(token, w_block);
            s_v.mul_add_relaxed(w_v, h_v).store(h_block);
        }
        for j in tail..n_hidden {
            h_pre[j] += s * row[j];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn apply_leaky_relu_f32_v3(
    token: archmage::X64V3Token,
    h_pre: &[f32],
    leaky_alpha: f32,
) -> Vec<f32> {
    let mut h = vec![0.0f32; h_pre.len()];
    let chunks = h_pre.len() / 8;
    let tail = chunks * 8;
    let leaky_v = GenericF32x8::splat_t(token, leaky_alpha);
    let zero_v = GenericF32x8::zero_t(token);
    for c in 0..chunks {
        let off = c * 8;
        let in_block: &[f32; 8] = (&h_pre[off..off + 8]).try_into().unwrap();
        let out_block: &mut [f32; 8] = (&mut h[off..off + 8]).try_into().unwrap();
        let v = GenericF32x8::load_t(token, in_block);
        let scaled = v * leaky_v;
        let neg_mask = v.simd_lt(zero_v);
        GenericF32x8::blend(neg_mask, scaled, v).store(out_block);
    }
    for j in tail..h_pre.len() {
        let v = h_pre[j];
        h[j] = if v >= 0.0 { v } else { leaky_alpha * v };
    }
    h
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn dot_product_f32_v3(token: archmage::X64V3Token, h: &[f32], w: &[f32]) -> f32 {
    let chunks = h.len() / 8;
    let tail = chunks * 8;
    let mut acc = GenericF32x8::zero_t(token);
    for c in 0..chunks {
        let off = c * 8;
        let h_block: &[f32; 8] = (&h[off..off + 8]).try_into().unwrap();
        let w_block: &[f32; 8] = (&w[off..off + 8]).try_into().unwrap();
        let h_v = GenericF32x8::load_t(token, h_block);
        let w_v = GenericF32x8::load_t(token, w_block);
        acc = h_v.mul_add_relaxed(w_v, acc);
    }
    let mut lanes = [0.0f32; 8];
    acc.store(&mut lanes);
    let lane_sum = (lanes[0] + lanes[1])
        + (lanes[2] + lanes[3])
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    let mut tail_sum = 0.0f32;
    for j in tail..h.len() {
        tail_sum += h[j] * w[j];
    }
    lane_sum + tail_sum
}

// =============================================================================
// NEON fallback — f32x8 generic. NEON `mul_add` lowers to fused `vfmla`,
// matching the `_v3` kernel bit-for-bit (same fused domain [0, n8),
// same mul+add tail).
// =============================================================================

#[magetypes(neon, -scalar)]
fn accumulate_rows_f32(
    token: Token,
    x: &[f32],
    w1: &[f32],
    h_pre: &mut [f32],
    n_features: usize,
    n_hidden: usize,
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let chunks = n_hidden / 8;
    let tail = chunks * 8;
    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 {
            continue;
        }
        let s_v = f32x8::splat_t(token, s);
        let row = &w1[i * n_hidden..(i + 1) * n_hidden];
        for c in 0..chunks {
            let off = c * 8;
            let h_block: &mut [f32; 8] = (&mut h_pre[off..off + 8]).try_into().unwrap();
            let w_block: &[f32; 8] = (&row[off..off + 8]).try_into().unwrap();
            let h_v = f32x8::load_t(token, h_block);
            let w_v = f32x8::load_t(token, w_block);
            s_v.mul_add_relaxed(w_v, h_v).store(h_block);
        }
        for j in tail..n_hidden {
            h_pre[j] += s * row[j];
        }
    }
}

#[magetypes(neon, wasm128, scalar)]
fn apply_leaky_relu_f32(token: Token, h_pre: &[f32], leaky_alpha: f32) -> Vec<f32> {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let mut h = vec![0.0f32; h_pre.len()];
    let chunks = h_pre.len() / 8;
    let tail = chunks * 8;
    let leaky_v = f32x8::splat_t(token, leaky_alpha);
    let zero_v = f32x8::zero_t(token);
    for c in 0..chunks {
        let off = c * 8;
        let in_block: &[f32; 8] = (&h_pre[off..off + 8]).try_into().unwrap();
        let out_block: &mut [f32; 8] = (&mut h[off..off + 8]).try_into().unwrap();
        let v = f32x8::load_t(token, in_block);
        let scaled = v * leaky_v;
        let neg_mask = v.simd_lt(zero_v);
        f32x8::blend(neg_mask, scaled, v).store(out_block);
    }
    for j in tail..h_pre.len() {
        let v = h_pre[j];
        h[j] = if v >= 0.0 { v } else { leaky_alpha * v };
    }
    h
}

#[magetypes(neon, -scalar)]
fn dot_product_f32(token: Token, h: &[f32], w: &[f32]) -> f32 {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let chunks = h.len() / 8;
    let tail = chunks * 8;
    let mut acc = f32x8::zero_t(token);
    for c in 0..chunks {
        let off = c * 8;
        let h_block: &[f32; 8] = (&h[off..off + 8]).try_into().unwrap();
        let w_block: &[f32; 8] = (&w[off..off + 8]).try_into().unwrap();
        let h_v = f32x8::load_t(token, h_block);
        let w_v = f32x8::load_t(token, w_block);
        acc = h_v.mul_add_relaxed(w_v, acc);
    }
    let mut lanes = [0.0f32; 8];
    acc.store(&mut lanes);
    let lane_sum = (lanes[0] + lanes[1])
        + (lanes[2] + lanes[3])
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    let mut tail_sum = 0.0f32;
    for j in tail..h.len() {
        tail_sum += h[j] * w[j];
    }
    lane_sum + tail_sum
}

// =============================================================================
// wasm128 / scalar fallback — per-lane fused arithmetic.
//
// `GenericF32x8<Token>::mul_add` lowers to plain `a*b+c` on these tiers
// (no fused FMA in wasm SIMD; the scalar polyfill avoids the libm call),
// which is NOT the canonical `_v3` arithmetic. These tiers therefore run
// per-element `f32::mul_add` — single-rounding fused on every target —
// over the canonical fused domain, with the same mul+add tail, the same
// 8-lane accumulator, and the same pairwise reduction tree as `_v3`.
// =============================================================================

fn accumulate_rows_f32_fallback(
    x: &[f32],
    w1: &[f32],
    h_pre: &mut [f32],
    n_features: usize,
    n_hidden: usize,
) {
    let n8 = n_hidden - (n_hidden % 8);
    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 {
            continue;
        }
        let row = &w1[i * n_hidden..(i + 1) * n_hidden];
        for j in 0..n8 {
            h_pre[j] = s.mul_add(row[j], h_pre[j]);
        }
        for j in n8..n_hidden {
            h_pre[j] += s * row[j];
        }
    }
}

fn dot_product_f32_fallback(h: &[f32], w: &[f32]) -> f32 {
    let n8 = h.len() - (h.len() % 8);
    // 8-lane accumulator emulation — identical chain order to `_v3`.
    let mut acc = [0.0f32; 8];
    for c in 0..n8 / 8 {
        for k in 0..8 {
            acc[k] = h[8 * c + k].mul_add(w[8 * c + k], acc[k]);
        }
    }
    let lane_sum = (acc[0] + acc[1]) + (acc[2] + acc[3]) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    let mut tail_sum = 0.0f32;
    for j in n8..h.len() {
        tail_sum += h[j] * w[j];
    }
    lane_sum + tail_sum
}

/// `wasm128` tier variant called by `incant!` dispatch.
#[cfg(target_arch = "wasm32")]
fn accumulate_rows_f32_wasm128(
    _token: archmage::Wasm128Token,
    x: &[f32],
    w1: &[f32],
    h_pre: &mut [f32],
    n_features: usize,
    n_hidden: usize,
) {
    accumulate_rows_f32_fallback(x, w1, h_pre, n_features, n_hidden);
}

/// `scalar` tier variant called by `incant!` dispatch.
fn accumulate_rows_f32_scalar(
    _token: archmage::ScalarToken,
    x: &[f32],
    w1: &[f32],
    h_pre: &mut [f32],
    n_features: usize,
    n_hidden: usize,
) {
    accumulate_rows_f32_fallback(x, w1, h_pre, n_features, n_hidden);
}

/// `wasm128` tier variant called by `incant!` dispatch.
#[cfg(target_arch = "wasm32")]
fn dot_product_f32_wasm128(_token: archmage::Wasm128Token, h: &[f32], w: &[f32]) -> f32 {
    dot_product_f32_fallback(h, w)
}

/// `scalar` tier variant called by `incant!` dispatch.
fn dot_product_f32_scalar(_token: archmage::ScalarToken, h: &[f32], w: &[f32]) -> f32 {
    dot_product_f32_fallback(h, w)
}

/// f32 encoder backprop: accumulate gw1/gb1 from dl_dh_pre.
#[inline]
pub fn encoder_backprop_layer1_f32(
    x: &[f32],
    dl_dh_pre: &[f32],
    gw1: &mut [f32],
    gb1: &mut [f32],
    n_features: usize,
    n_hidden: usize,
) {
    debug_assert_eq!(x.len(), n_features);
    debug_assert_eq!(dl_dh_pre.len(), n_hidden);
    debug_assert_eq!(gw1.len(), n_features * n_hidden);
    debug_assert_eq!(gb1.len(), n_hidden);

    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 {
            continue;
        }
        let row = &mut gw1[i * n_hidden..(i + 1) * n_hidden];
        for (g, &dh) in row.iter_mut().zip(dl_dh_pre.iter()) {
            *g += s * dh;
        }
    }
    for (g, &dh) in gb1.iter_mut().zip(dl_dh_pre.iter()) {
        *g += dh;
    }
}

/// f32 LeakyReLU backward.
#[inline]
pub fn leaky_relu_backward_f32(dl_dh: &[f32], h_pre: &[f32], leaky_alpha: f32) -> Vec<f32> {
    debug_assert_eq!(dl_dh.len(), h_pre.len());
    dl_dh
        .iter()
        .zip(h_pre.iter())
        .map(|(&dh, &hp)| if hp >= 0.0 { dh } else { leaky_alpha * dh })
        .collect()
}

// =============================================================================
// 2-layer encoder (f32) — wrapper around the 1-layer f32 fns. Mirrors the
// f64 `encoder_forward_2layer` / `encoder_backprop_2layer` in `simd_encoder`.
// This closes the perf gap on the per-sample-α 2-layer training path which
// was forced onto the f64 fns because the f32 module previously only had
// 1-layer fns (the documented "1.36× over f64" speedup never reached
// real training).
// =============================================================================

/// 2-layer f32 forward: `x → LeakyReLU(W₁x+b₁) → LeakyReLU(W₂h₁+b₂)`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn encoder_forward_2layer_f32(
    x: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
    n_features: usize,
    n_hidden1: usize,
    n_hidden2: usize,
    leaky_alpha: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (h1_pre, h1) = encoder_forward_f32(x, w1, b1, n_features, n_hidden1, leaky_alpha);
    let (h2_pre, h2) = encoder_forward_f32(&h1, w2, b2, n_hidden1, n_hidden2, leaky_alpha);
    (h1_pre, h1, h2_pre, h2)
}

/// 2-layer f32 backprop. Given `dl/dh2_pre` from the head, accumulates
/// gradients into both layers' (gw, gb) buffers.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn encoder_backprop_2layer_f32(
    x: &[f32],
    h1_pre: &[f32],
    h1: &[f32],
    h2_pre: &[f32],
    dl_dh2: &[f32],
    w2: &[f32],
    gw1: &mut [f32],
    gb1: &mut [f32],
    gw2: &mut [f32],
    gb2: &mut [f32],
    n_features: usize,
    n_hidden1: usize,
    n_hidden2: usize,
    leaky_alpha: f32,
) {
    // Layer 2 backprop: dl/dh2 → dl/dh2_pre → gw2, gb2.
    let dl_dh2_pre = leaky_relu_backward_f32(dl_dh2, h2_pre, leaky_alpha);
    encoder_backprop_layer1_f32(h1, &dl_dh2_pre, gw2, gb2, n_hidden1, n_hidden2);
    // Propagate to h1: dl/dh1[j] = Σ_k dl/dh2_pre[k] · w2[j*n2 + k].
    let mut dl_dh1 = vec![0.0f32; n_hidden1];
    for j in 0..n_hidden1 {
        let row = &w2[j * n_hidden2..(j + 1) * n_hidden2];
        let mut sum = 0.0f32;
        for k in 0..n_hidden2 {
            sum += dl_dh2_pre[k] * row[k];
        }
        dl_dh1[j] = sum;
    }
    // Layer 1 backprop: dl/dh1 → dl/dh1_pre → gw1, gb1.
    let dl_dh1_pre = leaky_relu_backward_f32(&dl_dh1, h1_pre, leaky_alpha);
    encoder_backprop_layer1_f32(x, &dl_dh1_pre, gw1, gb1, n_features, n_hidden1);
}

// =============================================================================
// Skip connection (f32) — mirrors `skip_forward`/`skip_backward` in the
// f64 `simd_encoder` module. Final gap-fill so the f32 API is a complete
// drop-in for the f64 API, unblocking removal of the f64 duplicate.
// =============================================================================

/// Skip-connection forward (f32): `b_skip + Σ_i x[i] · w_skip[i]`.
#[inline]
pub fn skip_forward_f32(x: &[f32], w_skip: &[f32], b_skip: f32) -> f32 {
    dot_bias_f32(x, w_skip, b_skip)
}

/// Skip-connection backward (f32): accumulate `dl_dy_skip · x[i]` into
/// `gw_skip[i]`, and `dl_dy_skip` into `*gb_skip`.
#[inline]
pub fn skip_backward_f32(x: &[f32], dl_dy_skip: f32, gw_skip: &mut [f32], gb_skip: &mut f32) {
    for (g, &xi) in gw_skip.iter_mut().zip(x.iter()) {
        *g += dl_dy_skip * xi;
    }
    *gb_skip += dl_dy_skip;
}

// =============================================================================
// f64 compatibility wrappers (NEW — replace the deleted `simd_encoder.rs`
// which had its own duplicate f64 SIMD implementations of every fn in
// this module). The trainer currently holds weights as `Vec<f64>`; these
// wrappers preserve the historical f64 API while routing through the
// single SIMD implementation (f32) below. The wrappers do NOT affect the
// shipped METRIC's precision — the runtime
// (`apply_per_sample_alpha_head_runtime`) already operates on f32 from the
// bake bytes, so this consolidation is a pure deduplication of the
// trainer-internal SIMD code. New trainer call sites should prefer the
// `*_f32` fns directly to skip the per-call cast overhead.
// =============================================================================

#[inline]
fn cast_f32(src: &[f64]) -> Vec<f32> {
    src.iter().map(|&v| v as f32).collect()
}

#[inline]
fn cast_f64(src: &[f32]) -> Vec<f64> {
    src.iter().map(|&v| v as f64).collect()
}

/// f64 wrapper around [`encoder_forward_f32`].
#[inline]
pub fn encoder_forward(
    x: &[f64],
    w1: &[f64],
    b1: &[f64],
    n_features: usize,
    n_hidden: usize,
    leaky_alpha: f64,
) -> (Vec<f64>, Vec<f64>) {
    let (h_pre, h) = encoder_forward_f32(
        &cast_f32(x),
        &cast_f32(w1),
        &cast_f32(b1),
        n_features,
        n_hidden,
        leaky_alpha as f32,
    );
    (cast_f64(&h_pre), cast_f64(&h))
}

/// f64 wrapper around [`encoder_forward_2layer_f32`].
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn encoder_forward_2layer(
    x: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    n_features: usize,
    n_hidden1: usize,
    n_hidden2: usize,
    leaky_alpha: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (h1_pre, h1, h2_pre, h2) = encoder_forward_2layer_f32(
        &cast_f32(x),
        &cast_f32(w1),
        &cast_f32(b1),
        &cast_f32(w2),
        &cast_f32(b2),
        n_features,
        n_hidden1,
        n_hidden2,
        leaky_alpha as f32,
    );
    (
        cast_f64(&h1_pre),
        cast_f64(&h1),
        cast_f64(&h2_pre),
        cast_f64(&h2),
    )
}

/// f64 wrapper around [`encoder_backprop_layer1_f32`]. Accumulates into f64 buffers.
#[inline]
pub fn encoder_backprop_layer1(
    x: &[f64],
    dl_dh_pre: &[f64],
    gw1: &mut [f64],
    gb1: &mut [f64],
    n_features: usize,
    n_hidden: usize,
) {
    let mut gw32 = vec![0.0f32; gw1.len()];
    let mut gb32 = vec![0.0f32; gb1.len()];
    encoder_backprop_layer1_f32(
        &cast_f32(x),
        &cast_f32(dl_dh_pre),
        &mut gw32,
        &mut gb32,
        n_features,
        n_hidden,
    );
    for (dst, &v) in gw1.iter_mut().zip(gw32.iter()) {
        *dst += v as f64;
    }
    for (dst, &v) in gb1.iter_mut().zip(gb32.iter()) {
        *dst += v as f64;
    }
}

/// f64 wrapper around [`encoder_backprop_2layer_f32`]. Accumulates into all
/// four caller-supplied f64 gradient buffers.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn encoder_backprop_2layer(
    x: &[f64],
    h1_pre: &[f64],
    h1: &[f64],
    h2_pre: &[f64],
    dl_dh2: &[f64],
    w2: &[f64],
    gw1: &mut [f64],
    gb1: &mut [f64],
    gw2: &mut [f64],
    gb2: &mut [f64],
    n_features: usize,
    n_hidden1: usize,
    n_hidden2: usize,
    leaky_alpha: f64,
) {
    let mut gw1_32 = vec![0.0f32; gw1.len()];
    let mut gb1_32 = vec![0.0f32; gb1.len()];
    let mut gw2_32 = vec![0.0f32; gw2.len()];
    let mut gb2_32 = vec![0.0f32; gb2.len()];
    encoder_backprop_2layer_f32(
        &cast_f32(x),
        &cast_f32(h1_pre),
        &cast_f32(h1),
        &cast_f32(h2_pre),
        &cast_f32(dl_dh2),
        &cast_f32(w2),
        &mut gw1_32,
        &mut gb1_32,
        &mut gw2_32,
        &mut gb2_32,
        n_features,
        n_hidden1,
        n_hidden2,
        leaky_alpha as f32,
    );
    for (dst, &v) in gw1.iter_mut().zip(gw1_32.iter()) {
        *dst += v as f64;
    }
    for (dst, &v) in gb1.iter_mut().zip(gb1_32.iter()) {
        *dst += v as f64;
    }
    for (dst, &v) in gw2.iter_mut().zip(gw2_32.iter()) {
        *dst += v as f64;
    }
    for (dst, &v) in gb2.iter_mut().zip(gb2_32.iter()) {
        *dst += v as f64;
    }
}

/// f64 wrapper around [`leaky_relu_backward_f32`].
#[inline]
pub fn leaky_relu_backward(dl_dh: &[f64], h_pre: &[f64], leaky_alpha: f64) -> Vec<f64> {
    cast_f64(&leaky_relu_backward_f32(
        &cast_f32(dl_dh),
        &cast_f32(h_pre),
        leaky_alpha as f32,
    ))
}

/// f64 wrapper around [`dot_bias_f32`].
#[inline]
pub fn dot_bias(h: &[f64], w: &[f64], bias: f64) -> f64 {
    dot_bias_f32(&cast_f32(h), &cast_f32(w), bias as f32) as f64
}

/// f64 wrapper around [`skip_forward_f32`].
#[inline]
pub fn skip_forward(x: &[f64], w_skip: &[f64], b_skip: f64) -> f64 {
    skip_forward_f32(&cast_f32(x), &cast_f32(w_skip), b_skip as f32) as f64
}

/// f64 wrapper around [`skip_backward_f32`]. Accumulates into f64 buffers.
#[inline]
pub fn skip_backward(x: &[f64], dl_dy_skip: f64, gw_skip: &mut [f64], gb_skip: &mut f64) {
    let mut gw32 = vec![0.0f32; gw_skip.len()];
    let mut gb32 = 0.0f32;
    skip_backward_f32(&cast_f32(x), dl_dy_skip as f32, &mut gw32, &mut gb32);
    for (dst, &v) in gw_skip.iter_mut().zip(gw32.iter()) {
        *dst += v as f64;
    }
    *gb_skip += gb32 as f64;
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    struct Xs32(u64);
    impl Xs32 {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn next_unit(&mut self) -> f32 {
            (self.next_u64() as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
        }
    }

    fn random_vec_f32(rng: &mut Xs32, n: usize) -> Vec<f32> {
        (0..n).map(|_| rng.next_unit()).collect()
    }

    fn random_sparse_f32(rng: &mut Xs32, n: usize, zero_frac: f64) -> Vec<f32> {
        let thresh = (2.0 * zero_frac - 1.0) as f32;
        (0..n)
            .map(|_| {
                if rng.next_unit() < thresh {
                    0.0
                } else {
                    rng.next_unit()
                }
            })
            .collect()
    }

    #[test]
    fn encoder_forward_f32_production_shape() {
        let n_features = 372;
        let n_hidden = 128;
        let alpha = 0.01f32;
        let mut rng = Xs32::new(0xCAFE);
        let x = random_sparse_f32(&mut rng, n_features, 0.3);
        let w1 = random_vec_f32(&mut rng, n_features * n_hidden);
        let b1 = random_vec_f32(&mut rng, n_hidden);
        let (h_pre, h) = encoder_forward_f32(&x, &w1, &b1, n_features, n_hidden, alpha);
        assert_eq!(h_pre.len(), n_hidden);
        assert_eq!(h.len(), n_hidden);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn dot_bias_f32_matches_scalar() {
        let n = 128;
        let mut rng = Xs32::new(0xBEEF);
        let h = random_vec_f32(&mut rng, n);
        let w = random_vec_f32(&mut rng, n);
        let bias = rng.next_unit();
        let scalar = bias + h.iter().zip(w.iter()).map(|(&a, &b)| a * b).sum::<f32>();
        let simd = dot_bias_f32(&h, &w, bias);
        let rel = (scalar - simd).abs() / scalar.abs().max(1e-6);
        assert!(rel < 1e-5, "scalar={scalar} simd={simd} rel={rel:e}");
    }

    /// f32 2-layer forward + backward must agree with the f64 versions to
    /// f32 precision — the trainer's per-sample-α path runs 2-layer, so
    /// this parity is what gates wiring f32 into real training.
    #[test]
    fn encoder_2layer_f32_matches_f64() {
        use crate::simd_encoder;
        let n_features = 372;
        let n_hidden1 = 128;
        let n_hidden2 = 64;
        let alpha32 = 0.01f32;
        let alpha64 = 0.01f64;
        let mut rng = Xs32::new(0xC0FFEE);
        let x32 = random_sparse_f32(&mut rng, n_features, 0.3);
        let w1_32 = random_vec_f32(&mut rng, n_features * n_hidden1);
        let b1_32 = random_vec_f32(&mut rng, n_hidden1);
        let w2_32 = random_vec_f32(&mut rng, n_hidden1 * n_hidden2);
        let b2_32 = random_vec_f32(&mut rng, n_hidden2);
        let cast = |v: &[f32]| -> Vec<f64> { v.iter().map(|&a| a as f64).collect() };
        let x64 = cast(&x32);
        let w1_64 = cast(&w1_32);
        let b1_64 = cast(&b1_32);
        let w2_64 = cast(&w2_32);
        let b2_64 = cast(&b2_32);

        let (h1_pre_32, h1_32, h2_pre_32, h2_32) = encoder_forward_2layer_f32(
            &x32, &w1_32, &b1_32, &w2_32, &b2_32, n_features, n_hidden1, n_hidden2, alpha32,
        );
        let (h1_pre_64, h1_64, h2_pre_64, h2_64) = simd_encoder::encoder_forward_2layer(
            &x64, &w1_64, &b1_64, &w2_64, &b2_64, n_features, n_hidden1, n_hidden2, alpha64,
        );
        // f32 forward must agree with f64 within f32 precision (sub-ulp drift
        // accumulates across the 47k+8k MAC chain; allow ~1e-3 relative).
        let max_rel_err = |a32: &[f32], a64: &[f64]| -> f64 {
            a32.iter()
                .zip(a64.iter())
                .map(|(&a, &b)| ((a as f64) - b).abs() / b.abs().max(1e-6))
                .fold(0.0_f64, f64::max)
        };
        let e_h1 = max_rel_err(&h1_32, &h1_64);
        let e_h2 = max_rel_err(&h2_32, &h2_64);
        assert!(e_h1 < 1e-3, "h1 max rel err {e_h1:e} too large");
        assert!(e_h2 < 1e-3, "h2 max rel err {e_h2:e} too large");

        // Backward parity: zero grad buffers and verify gw1/gw2 match.
        let mut gw1_32 = vec![0.0f32; w1_32.len()];
        let mut gb1_32 = vec![0.0f32; b1_32.len()];
        let mut gw2_32 = vec![0.0f32; w2_32.len()];
        let mut gb2_32 = vec![0.0f32; b2_32.len()];
        let dl_dh2_32 = random_vec_f32(&mut rng, n_hidden2);
        encoder_backprop_2layer_f32(
            &x32,
            &h1_pre_32,
            &h1_32,
            &h2_pre_32,
            &dl_dh2_32,
            &w2_32,
            &mut gw1_32,
            &mut gb1_32,
            &mut gw2_32,
            &mut gb2_32,
            n_features,
            n_hidden1,
            n_hidden2,
            alpha32,
        );
        let mut gw1_64 = vec![0.0f64; w1_64.len()];
        let mut gb1_64 = vec![0.0f64; b1_64.len()];
        let mut gw2_64 = vec![0.0f64; w2_64.len()];
        let mut gb2_64 = vec![0.0f64; b2_64.len()];
        let dl_dh2_64 = cast(&dl_dh2_32);
        simd_encoder::encoder_backprop_2layer(
            &x64,
            &h1_pre_64,
            &h1_64,
            &h2_pre_64,
            &dl_dh2_64,
            &w2_64,
            &mut gw1_64,
            &mut gb1_64,
            &mut gw2_64,
            &mut gb2_64,
            n_features,
            n_hidden1,
            n_hidden2,
            alpha64,
        );
        let e_gw1 = max_rel_err(&gw1_32, &gw1_64);
        let e_gw2 = max_rel_err(&gw2_32, &gw2_64);
        assert!(e_gw1 < 1e-3, "gw1 max rel err {e_gw1:e} too large");
        assert!(e_gw2 < 1e-3, "gw2 max rel err {e_gw2:e} too large");
    }

    #[test]
    #[ignore = "performance microbench"]
    fn encoder_f32_speedup_vs_f64() {
        use crate::simd_encoder;
        use std::time::Instant;
        let n_features = 372;
        let n_hidden = 128;
        let n_iters = 5000;
        let mut rng = Xs32::new(0x1234);
        let x32 = random_sparse_f32(&mut rng, n_features, 0.3);
        let w32 = random_vec_f32(&mut rng, n_features * n_hidden);
        let b32 = random_vec_f32(&mut rng, n_hidden);
        let x64: Vec<f64> = x32.iter().map(|&v| v as f64).collect();
        let w64: Vec<f64> = w32.iter().map(|&v| v as f64).collect();
        let b64: Vec<f64> = b32.iter().map(|&v| v as f64).collect();

        for _ in 0..200 {
            std::hint::black_box(encoder_forward_f32(
                &x32, &w32, &b32, n_features, n_hidden, 0.01,
            ));
        }
        for _ in 0..200 {
            std::hint::black_box(simd_encoder::encoder_forward(
                &x64, &w64, &b64, n_features, n_hidden, 0.01,
            ));
        }

        let t0 = Instant::now();
        for _ in 0..n_iters {
            std::hint::black_box(simd_encoder::encoder_forward(
                &x64, &w64, &b64, n_features, n_hidden, 0.01,
            ));
        }
        let f64_ns = t0.elapsed().as_nanos() as f64 / n_iters as f64;

        let t1 = Instant::now();
        for _ in 0..n_iters {
            std::hint::black_box(encoder_forward_f32(
                &x32, &w32, &b32, n_features, n_hidden, 0.01,
            ));
        }
        let f32_ns = t1.elapsed().as_nanos() as f64 / n_iters as f64;

        eprintln!(
            "encoder_forward @ 372×128:\n  f64: {:.2} µs\n  f32: {:.2} µs\n  speedup: {:.2}×",
            f64_ns / 1e3,
            f32_ns / 1e3,
            f64_ns / f32_ns
        );
    }

    // =====================================================================
    // Tier parity — the AVX2 (`_v3`, f32x8) kernels are the canonical
    // arithmetic; every other tier the dispatcher can pick must reproduce
    // them BIT FOR BIT.
    // =====================================================================

    /// Ordered-int ULP distance between two f32 values.
    fn ulp_diff_f32(a: f32, b: f32) -> u32 {
        fn ord(v: f32) -> i32 {
            let b = v.to_bits() as i32;
            if b < 0 { i32::MIN - b } else { b }
        }
        ord(a).abs_diff(ord(b))
    }

    fn max_ulp_f32(a: &[f32], b: &[f32]) -> (u32, usize) {
        let mut worst = (0u32, usize::MAX);
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            let d = ulp_diff_f32(x, y);
            if d > worst.0 {
                worst = (d, i);
            }
        }
        worst
    }

    /// Shapes: production 372×128 plus every remainder class so `n_hidden`
    /// exercises the v4 16-lane boundary vs the v3 8-lane boundary
    /// (nh%16 ∈ {0..15}, including 8..15 where v4's tail must re-enter
    /// the fused domain at the v3 boundary).
    #[cfg(target_arch = "x86_64")]
    const ENC_SHAPES: &[(usize, usize)] = &[
        (372, 128),
        (16, 8),
        (16, 16),
        (16, 12),
        (16, 20),
        (16, 24),
        (16, 28),
        (8, 15),
        (8, 23),
        (8, 31),
        (4, 9),
        (4, 13),
        (64, 40),
        (128, 24),
    ];

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn encoder_all_tiers_bit_identical() {
        use archmage::SimdToken;
        // Hold the token lock for the whole body: `encoder_dispatch_bit_identical_under_token_permutations`
        // disables tokens process-wide, and a `summon()` racing it would silently skip the v4 comparison.
        let _lock = archmage::testing::lock_token_testing();
        let v3 = archmage::X64V3Token::summon();
        let v4 = archmage::X64V4Token::summon();
        assert!(
            v3.is_some(),
            "this test needs an x86-64-v3 (AVX2+FMA) host: it is the canonical tier every other tier must reproduce"
        );
        let alpha = 0.01f32;
        let mut failures = Vec::new();
        for &(nf, nh) in ENC_SHAPES {
            for &seed in &[0x51EDu64, 0xABCD_1234, 0x0000_00FF] {
                let mut rng = Xs32::new(seed ^ (nf * 31 + nh) as u64);
                let x = random_sparse_f32(&mut rng, nf, 0.35);
                let w1 = random_vec_f32(&mut rng, nf * nh);
                let h_pre0 = random_vec_f32(&mut rng, nh);

                // ---- accumulate_rows_f32 ----
                let mut ref_rows = h_pre0.clone();
                accumulate_rows_f32_v3(v3.unwrap(), &x, &w1, &mut ref_rows, nf, nh);

                let mut sca_rows = h_pre0.clone();
                accumulate_rows_f32_scalar(
                    archmage::ScalarToken::summon().unwrap(),
                    &x,
                    &w1,
                    &mut sca_rows,
                    nf,
                    nh,
                );
                let (u, i) = max_ulp_f32(&sca_rows, &ref_rows);
                if u != 0 {
                    failures.push(format!(
                        "accumulate_rows scalar-vs-v3 [{nf}x{nh} seed={seed:#x}] \
                         max_ulp={u} at {i}"
                    ));
                }

                if let Some(t4) = v4 {
                    let mut v4_rows = h_pre0.clone();
                    accumulate_rows_f32_v4(t4, &x, &w1, &mut v4_rows, nf, nh);
                    let (u, i) = max_ulp_f32(&v4_rows, &ref_rows);
                    if u != 0 {
                        failures.push(format!(
                            "accumulate_rows v4-vs-v3 [{nf}x{nh} seed={seed:#x}] \
                             max_ulp={u} at {i}"
                        ));
                    }
                }

                // ---- apply_leaky_relu_f32 ----
                let rl_v3 = apply_leaky_relu_f32_v3(v3.unwrap(), &h_pre0, alpha);
                let rl_s = apply_leaky_relu_f32_scalar(
                    archmage::ScalarToken::summon().unwrap(),
                    &h_pre0,
                    alpha,
                );
                let (u, i) = max_ulp_f32(&rl_s, &rl_v3);
                if u != 0 {
                    failures.push(format!(
                        "leaky_relu scalar-vs-v3 [{nf}x{nh} seed={seed:#x}] \
                         max_ulp={u} at {i}"
                    ));
                }
                if let Some(t4) = v4 {
                    let rl_v4 = apply_leaky_relu_f32_v4(t4, &h_pre0, alpha);
                    let (u, i) = max_ulp_f32(&rl_v4, &rl_v3);
                    if u != 0 {
                        failures.push(format!(
                            "leaky_relu v4-vs-v3 [{nf}x{nh} seed={seed:#x}] \
                             max_ulp={u} at {i}"
                        ));
                    }
                }

                // ---- dot_product_f32 ----
                let h = rl_v3.clone();
                let w2 = random_vec_f32(&mut rng, nh);
                let d_v3 = dot_product_f32_v3(v3.unwrap(), &h, &w2);
                let d_s = dot_product_f32_scalar(archmage::ScalarToken::summon().unwrap(), &h, &w2);
                let u = ulp_diff_f32(d_s, d_v3);
                if u != 0 {
                    failures.push(format!(
                        "dot_product scalar-vs-v3 [{nf}x{nh} seed={seed:#x}] \
                         ulp={u} (scalar={d_s:e} v3={d_v3:e})"
                    ));
                }
                if let Some(t4) = v4 {
                    let d_v4 = dot_product_f32_v4(t4, &h, &w2);
                    let u = ulp_diff_f32(d_v4, d_v3);
                    if u != 0 {
                        failures.push(format!(
                            "dot_product v4-vs-v3 [{nf}x{nh} seed={seed:#x}] \
                             ulp={u} (v4={d_v4:e} v3={d_v3:e})"
                        ));
                    }
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// The dispatched wrappers (`encoder_forward_f32`/`dot_bias_f32`) must
    /// return identical bytes under every host token permutation.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn encoder_dispatch_bit_identical_under_token_permutations() {
        use archmage::SimdToken;
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
        assert!(
            archmage::X64V3Token::summon().is_some(),
            "this test needs an x86-64-v3 (AVX2+FMA) host: it is the canonical tier every other tier must reproduce"
        );
        let (nf, nh) = (372usize, 128usize);
        let alpha = 0.01f32;
        let mut rng = Xs32::new(0xE11C_0DE1);
        let x = random_sparse_f32(&mut rng, nf, 0.35);
        let w1 = random_vec_f32(&mut rng, nf * nh);
        let b1 = random_vec_f32(&mut rng, nh);
        let w2 = random_vec_f32(&mut rng, nh);
        let bias = rng.next_unit();

        let mut baseline: Option<Vec<u32>> = None;
        let mut failures = Vec::new();
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let (hp, h) = encoder_forward_f32(&x, &w1, &b1, nf, nh, alpha);
            let y = dot_bias_f32(&h, &w2, bias);
            let mut bits: Vec<u32> = hp.iter().map(|f| f.to_bits()).collect();
            bits.extend(h.iter().map(|f| f.to_bits()));
            bits.push(y.to_bits());
            if let Some(b) = &baseline {
                if bits != *b {
                    failures.push(format!("encoder dispatch diverged: {}", perm.label));
                }
            } else {
                baseline = Some(bits);
            }
            eprintln!("perm [{}] -> y={:e} bits={:#x}", perm.label, y, y.to_bits());
        });
        eprintln!("permutations run: {}", report.permutations_run);
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}
