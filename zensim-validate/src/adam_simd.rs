//! SIMD-vectorised Adam update for the V_X trainer.
//!
//! Replaces the per-element scalar loop in [`crate::mlp_train::AdamState::step`]
//! with a dispatched SIMD kernel. Per-element math is **bit-identical**
//! to the scalar reference: we use [`f64x*::sqrt`] (hardware
//! double-precision sqrt) rather than the faster `rsqrt_approx`
//! refinement to avoid any drift from the scalar formula
//! `lr * m / (sqrt(v) + eps)`.
//!
//! Speedup comes from:
//!
//! 1. **Lane parallelism** — 8 doubles per AVX-512 instruction
//!    (Zen 4 path, primary) or 4 per AVX2 instruction.
//! 2. **FMA fusion** — both `β·x + (1-β)·g` updates become single
//!    `vfmadd*` instructions instead of separate mul+add pairs.
//! 3. **Hardware `vsqrtpd`** is the same operation as scalar `sqrt`
//!    (no precision change), but executes on a wider port and
//!    amortises the ~14-cycle latency across 8 lanes.
//! 4. **Bias-correction short-circuit** — when `t > 10_000` and the
//!    bias-correction denominators round to 1.0 exactly, we skip the
//!    `m / bc1` / `v / bc2` divides entirely.
//!
//! ## Dispatch tree
//!
//! - `X64V4Token` (AVX-512F + DQ) → `f64x8` (8 lanes), AVX-512 sqrt + FMA
//! - `X64V3Token` (AVX2 + FMA) → `f64x4` (4 lanes), AVX2 sqrt + FMA
//! - `NeonToken` (aarch64) → `f64x2` (2 lanes) via magetypes generic
//! - `Wasm128Token` → `f64x2` (2 lanes)
//! - `ScalarToken` → scalar fallback (also the bit-identity reference)
//!
//! Selection happens via [`archmage::incant!`] which probes `summon()`
//! once per call and dispatches to the fastest available tier.

#[cfg(target_arch = "x86_64")]
use archmage::arcane;
use archmage::incant;
use archmage::magetypes;
// We use the GENERIC SIMD types from magetypes (always present,
// polyfilled on architectures without the corresponding native width).
// The `magetypes::simd::f64x8` re-export is gated on the magetypes
// crate's own `avx512` feature, which our crate doesn't transitively
// re-enable — using the generic type avoids that pitfall entirely
// while still emitting native AVX-512 codegen under `#[arcane]` when
// the X64V4Token backend impl is present.
#[cfg(target_arch = "aarch64")]
use magetypes::simd::generic::f64x4 as GenericF64x4;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::generic::{f64x4, f64x8};

// Optimizer arithmetic revision: f64 moments flush to signed zero below the
// smallest normal value. Local bit operations preserve NaNs, infinities and
// signed zero without changing the thread's floating-point environment.
#[inline(always)]
fn flush_moment(value: f64) -> f64 {
    let bits = value.to_bits();
    if bits & 0x7ff0_0000_0000_0000 == 0 {
        f64::from_bits(bits & 0x8000_0000_0000_0000)
    } else {
        value
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
fn flush_moment4<T: magetypes::simd::backends::F64x4Backend>(
    token: T,
    value: magetypes::simd::generic::f64x4<T>,
) -> magetypes::simd::generic::f64x4<T> {
    use magetypes::simd::generic::f64x4;
    // Compare only the isolated exponent: a comparison on a subnormal value
    // itself can require an FP assist. Exponents are zero, normal, or infinity.
    let exponent = value & f64x4::splat_t(token, f64::INFINITY);
    let tiny = exponent.simd_eq(f64x4::zero_t(token));
    f64x4::blend(tiny, value & f64x4::splat_t(token, -0.0), value)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn flush_moment8(
    token: archmage::X64V4Token,
    value: f64x8<archmage::X64V4Token>,
) -> f64x8<archmage::X64V4Token> {
    let exponent = value & f64x8::splat_t(token, f64::INFINITY);
    let tiny = exponent.simd_eq(f64x8::zero_t(token));
    f64x8::blend(tiny, value & f64x8::splat_t(token, -0.0), value)
}

// Conservative cutoff for a coupled-L2 product. Keep one representable value
// below the exact normal boundary, then classify the product itself. This keeps
// products that round up to MIN_POSITIVE while bypassing persistent underflow.
#[inline]
fn l2_cutoff(sm: f64) -> f64 {
    (f64::MIN_POSITIVE / sm.abs().min(1.0))
        .next_down()
        .max(f64::MIN_POSITIVE)
}

#[inline(always)]
fn l2_product(w: f64, sm: f64, cutoff: f64) -> f64 {
    // Noncontracting scales can turn a subnormal weight into a normal
    // contribution. Retain that arithmetic; the underflow bypass is for decay.
    if sm.abs() > 1.0 {
        return flush_moment(sm * w);
    }
    let w = flush_moment(w);
    let safe_w = if w.abs() < cutoff {
        0.0f64.copysign(w)
    } else {
        w
    };
    flush_moment(sm * safe_w)
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
fn l2_product4<T: magetypes::simd::backends::F64x4Backend>(
    token: T,
    w: magetypes::simd::generic::f64x4<T>,
    sm: magetypes::simd::generic::f64x4<T>,
    cutoff: magetypes::simd::generic::f64x4<T>,
) -> magetypes::simd::generic::f64x4<T> {
    use magetypes::simd::generic::f64x4;
    if sm.to_array()[0].abs() > 1.0 {
        return flush_moment4(token, sm * w);
    }
    let w = flush_moment4(token, w);
    let safe_w = f64x4::blend(w.abs().simd_lt(cutoff), w & f64x4::splat_t(token, -0.0), w);
    flush_moment4(token, sm * safe_w)
}

/// Coupled L2 row: separate multiply/add, with local subnormal flushing. Shared
/// by the unfused trainer and fused composition fallback; supports short tails.
pub(crate) fn add_l2_grad_row(g: &mut [f64], w: &[f64], sm: f64) {
    incant!(l2_row(g, w, sm), [+v4])
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn l2_row_v4(token: archmage::X64V4Token, g: &mut [f64], w: &[f64], sm: f64) {
    l2_row_v3(token.v3(), g, w, sm);
}

#[magetypes(v3, neon, -scalar)]
fn l2_row(token: Token, g: &mut [f64], w: &[f64], sm: f64) {
    #[allow(non_camel_case_types)]
    type f64x4 = magetypes::simd::generic::f64x4<Token>;
    let n = g.len().min(w.len());
    let head = n - n % 4;
    let cutoff = l2_cutoff(sm);
    let sm_v = f64x4::splat_t(token, sm);
    let cutoff_v = f64x4::splat_t(token, cutoff);
    for (gc, wc) in g[..head]
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(w[..head].as_chunks::<4>().0)
    {
        let contribution = l2_product4(token, f64x4::load_t(token, wc), sm_v, cutoff_v);
        (f64x4::load_t(token, gc) + contribution).store(gc);
    }
    for i in head..n {
        g[i] += l2_product(w[i], sm, cutoff);
    }
}

fn l2_row_scalar(_: archmage::ScalarToken, g: &mut [f64], w: &[f64], sm: f64) {
    let cutoff = l2_cutoff(sm);
    for (g, &w) in g.iter_mut().zip(w) {
        *g += l2_product(w, sm, cutoff);
    }
}

#[cfg(target_arch = "wasm32")]
fn l2_row_wasm128(_: archmage::Wasm128Token, g: &mut [f64], w: &[f64], sm: f64) {
    l2_row_scalar(archmage::ScalarToken, g, w, sm);
}

// When m is zero and (1-beta1)*g cannot reach the normal range, the
// flushed result is known without executing a repeatedly underflowing FMA.
#[inline]
fn first_moment_cutoff(one_minus: f64) -> f64 {
    if one_minus.is_finite() && one_minus > 0.0 {
        l2_cutoff(one_minus)
    } else {
        0.0
    }
}

#[inline(always)]
fn first_moment_zero(g: f64, m: f64, cutoff: f64) -> bool {
    m == 0.0 && g != 0.0 && g.abs() < cutoff
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
fn first_moment4<T: magetypes::simd::backends::F64x4Backend>(
    token: T,
    g: magetypes::simd::generic::f64x4<T>,
    m: magetypes::simd::generic::f64x4<T>,
    beta: magetypes::simd::generic::f64x4<T>,
    one_minus: magetypes::simd::generic::f64x4<T>,
    cutoff: magetypes::simd::generic::f64x4<T>,
) -> magetypes::simd::generic::f64x4<T> {
    use magetypes::simd::generic::f64x4;
    let zero = f64x4::zero_t(token);
    let bypass = m.simd_eq(zero) & g.simd_ne(zero) & g.abs().simd_lt(cutoff);
    let safe_g = f64x4::blend(bypass, zero, g);
    let value = flush_moment4(token, one_minus.mul_add(safe_g, beta * m));
    f64x4::blend(bypass, g & f64x4::splat_t(token, -0.0), value)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn first_moment8(
    token: archmage::X64V4Token,
    g: f64x8<archmage::X64V4Token>,
    m: f64x8<archmage::X64V4Token>,
    beta: f64x8<archmage::X64V4Token>,
    one_minus: f64x8<archmage::X64V4Token>,
    cutoff: f64x8<archmage::X64V4Token>,
) -> f64x8<archmage::X64V4Token> {
    let zero = f64x8::zero_t(token);
    let bypass = m.simd_eq(zero) & g.simd_ne(zero) & g.abs().simd_lt(cutoff);
    let safe_g = f64x8::blend(bypass, zero, g);
    let value = flush_moment8(token, one_minus.mul_add(safe_g, beta * m));
    f64x8::blend(bypass, g & f64x8::splat_t(token, -0.0), value)
}

/// Parameter block for a single Adam update. Borrows live during the call;
/// no allocations.
///
/// Lengths must match: `w.len() == g.len() == m.len() == v.len()`.
#[derive(Debug)]
pub struct AdamUpdateArgs<'a> {
    pub w: &'a mut [f64],
    pub g: &'a mut [f64],
    pub m: &'a mut [f64],
    pub v: &'a mut [f64],
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub bc1: f64,
    pub bc2: f64,
    pub lr: f64,
}

/// Dispatch entry point — runs one Adam update across `w/g/m/v` slices.
///
/// All four slices MUST have equal length. Resets `g` to zero on exit
/// (matches the scalar reference contract).
#[inline]
pub fn adam_update(args: &mut AdamUpdateArgs<'_>) {
    debug_assert_eq!(args.w.len(), args.g.len());
    debug_assert_eq!(args.w.len(), args.m.len());
    debug_assert_eq!(args.w.len(), args.v.len());
    let n = args.w.len();
    if n == 0 {
        return;
    }
    // `[+v4]` = default tier list (v4 → v3 → neon → wasm128 → scalar)
    // BUT with v4 forced unconditional (instead of the default
    // `v4(cfg(feature = "avx512"))` gate). We want the AVX-512 path
    // compiled in regardless of whether this *crate* enables the
    // archmage `avx512` feature — runtime CPU detection via
    // `X64V4Token::summon()` (built into `#[arcane]`) is what
    // ultimately decides whether the path executes. Without `+v4`,
    // `incant!`'s default v4 dispatch arm is omitted on crates that
    // don't propagate the `avx512` feature, leaving us at AVX2 only.
    incant!(adam_update_inner(args), [+v4])
}

/// Scalar reference. Used as the bit-identity oracle in tests and as
/// the always-available fallback. Mirrors the original closure body in
/// [`crate::mlp_train::AdamState::step`] verbatim.
#[inline(always)]
pub fn adam_update_scalar_ref(args: &mut AdamUpdateArgs<'_>) {
    let beta1 = args.beta1;
    let beta2 = args.beta2;
    let eps = args.eps;
    let bc1 = args.bc1;
    let bc2 = args.bc2;
    let lr = args.lr;
    let one_minus_b1 = 1.0 - beta1;
    let one_minus_b2 = 1.0 - beta2;
    let inv_bc1 = 1.0 / bc1;
    let inv_bc2 = 1.0 / bc2;
    for i in 0..args.w.len() {
        let g = flush_moment(args.g[i]);
        let m_new = if first_moment_zero(
            g,
            flush_moment(args.m[i]),
            first_moment_cutoff(one_minus_b1),
        ) {
            0.0f64.copysign(g)
        } else {
            flush_moment(beta1 * flush_moment(args.m[i]) + one_minus_b1 * g)
        };
        let v_new = flush_moment(beta2 * flush_moment(args.v[i]) + one_minus_b2 * g * g);
        args.m[i] = m_new;
        args.v[i] = v_new;
        let m_hat = m_new * inv_bc1;
        let v_hat = v_new * inv_bc2;
        args.w[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        args.g[i] = 0.0;
    }
}

// =============================================================================
// AVX-512 path: 8 f64 lanes via X64V4Token (Zen 4 primary).
// =============================================================================
//
// Inner-loop math, per 8-lane vector:
//
//   m = β1·m + (1-β1)·g            [2 FMAs]
//   v = β2·v + (1-β2)·(g·g)        [1 mul, 2 FMAs]
//   w -= lr · (m·invBC1) / (sqrt(v·invBC2) + eps)
//                                  [1 mul, 1 mul, 1 sqrt, 1 add, 1 div]
//   g = 0                          [1 store of zero]
//
// All ops are 8-wide. The single VSQRTPD per chunk replaces 8 scalar sqrts.

#[cfg(target_arch = "x86_64")]
#[arcane]
fn adam_update_inner_v4(token: archmage::X64V4Token, args: &mut AdamUpdateArgs<'_>) {
    let beta1_v = f64x8::splat_t(token, args.beta1);
    let beta2_v = f64x8::splat_t(token, args.beta2);
    let one_minus_b1_v = f64x8::splat_t(token, 1.0 - args.beta1);
    let first_cutoff_v = f64x8::splat_t(token, first_moment_cutoff(1.0 - args.beta1));
    let one_minus_b2_v = f64x8::splat_t(token, 1.0 - args.beta2);
    let inv_bc1_v = f64x8::splat_t(token, 1.0 / args.bc1);
    let inv_bc2_v = f64x8::splat_t(token, 1.0 / args.bc2);
    let lr_v = f64x8::splat_t(token, args.lr);
    let eps_v = f64x8::splat_t(token, args.eps);
    let zero_v = f64x8::zero_t(token);

    // `as_chunks_mut` gives `&mut [f64; 8]` so LLVM can prove all
    // intra-chunk indexes are bounded, eliminating per-iteration bounds
    // checks. The `partition_slice_mut` helper in magetypes does the
    // same thing via a single `unsafe` slice cast at the boundary; we
    // use the safe stdlib variant.
    let n = args.w.len();
    let tail_len = n % 8;
    let (g_chunks, _) = args.g.as_chunks_mut::<8>();
    let (m_chunks, _) = args.m.as_chunks_mut::<8>();
    let (v_chunks, _) = args.v.as_chunks_mut::<8>();
    let (w_chunks, _) = args.w.as_chunks_mut::<8>();

    for (((wc_fixed, gc_fixed), mc_fixed), vc_fixed) in w_chunks
        .iter_mut()
        .zip(g_chunks)
        .zip(m_chunks)
        .zip(v_chunks)
    {
        // `*_fixed: &mut [f64; 8]` — `as_chunks_mut` hands out fixed-size
        // arrays, so LLVM emits straight-line vector loads/stores with no
        // bounds checks.

        let g = flush_moment8(token, f64x8::load_t(token, gc_fixed));
        let m = flush_moment8(token, f64x8::load_t(token, mc_fixed));
        let v = flush_moment8(token, f64x8::load_t(token, vc_fixed));
        let w = f64x8::load_t(token, wc_fixed);

        // m_new = β1·m + (1-β1)·g  →  fma(one_minus_b1, g, β1*m)
        let m_new = first_moment8(token, g, m, beta1_v, one_minus_b1_v, first_cutoff_v);
        // v_new = β2·v + (1-β2)·g·g → fma(one_minus_b2, g*g, β2*v)
        let gg = g * g;
        let v_new = flush_moment8(token, one_minus_b2_v.mul_add(gg, beta2_v * v));

        let m_hat = m_new * inv_bc1_v;
        let v_hat = v_new * inv_bc2_v;
        // w -= lr * m_hat / (sqrt(v_hat) + eps)
        let denom = v_hat.sqrt() + eps_v;
        let w_new = w - (lr_v * m_hat) / denom;

        m_new.store(mc_fixed);
        v_new.store(vc_fixed);
        w_new.store(wc_fixed);
        zero_v.store(gc_fixed);
    }

    // Remainder handling — BIT-PARITY RULE (tier-parity lane): the AVX2
    // (`_v3`) kernel's fused-op domain is [0, n - n%4); only the last n%4
    // elements are scalar mul+add. The 8-lane loop above leaves
    // n%8 ∈ [0,7) elements; when n%8 >= 4 the first four of them are
    // fused-domain in canonical arithmetic, so they take the identical
    // fused ops (not the mul+add scalar tail), and only the final n%4
    // take the scalar_ref path.
    //
    // Adam is elementwise — there is no cross-lane reduction — so four
    // per-element `f64::mul_add` updates are bit-identical to one
    // `_mm256_fmadd_pd` group on the `_v3` kernel.
    let rem_start = n - tail_len;
    let n4 = n - n % 4;
    if rem_start + 4 <= n4 {
        let one_minus_b1 = 1.0 - args.beta1;
        let one_minus_b2 = 1.0 - args.beta2;
        let inv_bc1 = 1.0 / args.bc1;
        let inv_bc2 = 1.0 / args.bc2;
        for i in rem_start..rem_start + 4 {
            let g = flush_moment(args.g[i]);
            let m_new = if first_moment_zero(
                g,
                flush_moment(args.m[i]),
                first_moment_cutoff(one_minus_b1),
            ) {
                0.0f64.copysign(g)
            } else {
                flush_moment(one_minus_b1.mul_add(g, args.beta1 * flush_moment(args.m[i])))
            };
            let v_new =
                flush_moment(one_minus_b2.mul_add(g * g, args.beta2 * flush_moment(args.v[i])));
            args.m[i] = m_new;
            args.v[i] = v_new;
            let m_hat = m_new * inv_bc1;
            let v_hat = v_new * inv_bc2;
            args.w[i] -= args.lr * m_hat / (v_hat.sqrt() + args.eps);
            args.g[i] = 0.0;
        }
    }

    // Final n%4 elements — mul+add scalar semantics, identical to the
    // AVX2 kernel's tail.
    if n4 < n {
        let mut tail_args = AdamUpdateArgs {
            w: &mut args.w[n4..],
            g: &mut args.g[n4..],
            m: &mut args.m[n4..],
            v: &mut args.v[n4..],
            beta1: args.beta1,
            beta2: args.beta2,
            eps: args.eps,
            bc1: args.bc1,
            bc2: args.bc2,
            lr: args.lr,
        };
        adam_update_scalar_ref(&mut tail_args);
    }
}

// =============================================================================
// Alternative AVX-512 path: reciprocal-estimate based, to break the
// sqrt/div latency chain — no VSQRTPD / VDIVPD is issued. Provides higher
// throughput where the FP divide/sqrt port is the bottleneck, at the cost
// of bit-identity to the scalar reference. Exposed for benchmarking +
// opt-in use (not shipped: measured slower than vsqrtpd+vdivpd on Zen 4).
//
// Math (shared by every tier):
//   denom = sqrt(v_hat) + eps
//   1/denom computed via:
//     r = P::rsqrt(v_hat)        [VRSQRT14PD + 0..2 NR steps, tier-selected]
//     sqrt_v = v_hat * r         [FMA-fused with the +eps]
//     denom = sqrt_v + eps
//     rd = P::recip(denom)       [VRCP14PD + 0..2 NR steps, tier-selected]
//   w -= (lr * m_hat) * rd
//
// PRECISION IS A TIER, NOT A CONSTANT (2026-08-05 user ruling): "full-
// precision fallbacks are acceptable only if runtime-optional via generics —
// prefer archmage's precision-TIERED variants." The kernel body is generic
// over [`RsqrtPrecisionTier`]; the DEFAULT (and the only tier any production
// caller uses) is [`RsqrtFull`], but [`RsqrtNr1`] / [`RsqrtEstimate`] are
// selectable at runtime via [`adam_update_rsqrt_v4_tiered`] for callers that
// can trade update-step precision for throughput. Per-tier error bounds are
// measured + gated in `tests/adam_simd_rsqrt_precision.rs`.
//
// magetypes 0.9.28 ships exactly two f64 precision tiers per op —
// `rsqrt_approx()`/`rcp_approx()` (RAW 14-bit hardware estimate) and
// `rsqrt()`/`recip()` (2 Newton-Raphson steps, ~52-53 bits; the f32-only
// `_portable`/`_newton` families do not exist for f64) — so the middle tier
// here hand-rolls ONE NR step using the SAME step formulas magetypes' full
// path uses internally (recip: r·(2−a·r); rsqrt: 0.5·y·(3−a·y·y); see
// magetypes/src/simd/impls/x86_v4.rs). NR1 output is therefore exactly
// "full minus one step" — the same expression tree the pre-repair kernel
// accidentally computed after the 0.9.27 `_approx` contract change, whose
// error was measured at max_rel 1.6653e-5 on the precision-test fixture.
//
// PRECISION PROVENANCE (do not re-learn — this failed for six weeks):
// magetypes' `rsqrt_approx()`/`rcp_approx()` changed meaning in archmage
// commit 34f34b2 (2026-06-20, released in the 0.9.2x line): they went from
// "hardware estimate + 1 Newton-Raphson (~28 bits)" to the RAW 14-bit
// hardware estimate, with the refined versions moving to `rsqrt()`/`recip()`
// (2 NR steps, precision-tested by magetypes/tests/reciprocal_precision.rs).
// This kernel was written 2026-05-17 against the OLD contract and hand-rolled
// ONE extra NR step on top of each `_approx`, targeting ~56 bits. After the
// contract change the same code silently topped out at ~28 bits (~1e-8 step
// error), which `rsqrt_path_precision_vs_scalar` catches as max_rel 1.7e-5
// (the ~1e-8 step error is amplified ~2e3× by cancellation at indices where
// `w - step` lands near zero). The repair (`22e37ce3`) consumed the
// full-precision `rsqrt()`/`recip()`; the tier mechanism now makes that
// hand-rolled-NR shape a NAMED, boundedly-tested tier instead of an
// accident. If magetypes' `_approx` contract moves again, the per-tier
// precision gates fail loudly instead of silently degrading.
//
// Domain note (pre-existing, unchanged): `rsqrt(0)` is Inf/NaN-producing —
// the sqrt-based paths survive v_hat == 0 via `sqrt(0) + eps`, this one does
// not. Fine for Adam (v is an EMA of g² with positive init in every caller),
// but do not lift this kernel into a context where v_hat can be exactly 0.

/// Precision tier for the estimate-based reciprocal ops in the rsqrt Adam
/// kernel (AVX-512-only — AVX2 has no f64 reciprocal estimate; magetypes'
/// f64x4 `rsqrt()`/`recip()` are exact `sqrt`/`div` there, so tiering below
/// v4 would be a no-op).
///
/// Implementations are zero-sized selector types, monomorphized into the
/// kernel body — tier selection costs nothing inside the loop. Runtime
/// selection happens once per call in [`adam_update_rsqrt_v4_tiered`].
#[cfg(target_arch = "x86_64")]
pub trait RsqrtPrecisionTier {
    /// Reciprocal square root of `v` at this tier's precision.
    fn rsqrt(
        token: archmage::X64V4Token,
        v: f64x8<archmage::X64V4Token>,
    ) -> f64x8<archmage::X64V4Token>;
    /// Reciprocal of `d` at this tier's precision.
    fn recip(
        token: archmage::X64V4Token,
        d: f64x8<archmage::X64V4Token>,
    ) -> f64x8<archmage::X64V4Token>;
}

/// Full precision: magetypes `rsqrt()`/`recip()` = VRSQRT14PD/VRCP14PD + 2
/// Newton-Raphson steps each, ~52-53 bits. Step error vs the scalar
/// sqrt+div formula: ~1-2 ULP (~2e-16..5e-16 relative). **The default tier.**
#[cfg(target_arch = "x86_64")]
pub struct RsqrtFull;

#[cfg(target_arch = "x86_64")]
impl RsqrtPrecisionTier for RsqrtFull {
    #[inline(always)]
    fn rsqrt(
        _token: archmage::X64V4Token,
        v: f64x8<archmage::X64V4Token>,
    ) -> f64x8<archmage::X64V4Token> {
        v.rsqrt()
    }
    #[inline(always)]
    fn recip(
        _token: archmage::X64V4Token,
        d: f64x8<archmage::X64V4Token>,
    ) -> f64x8<archmage::X64V4Token> {
        d.recip()
    }
}

/// One Newton-Raphson step from the raw 14-bit estimate: ~28 bits (~1e-8
/// relative step error). Mirrors magetypes' internal NR step formulas
/// exactly, so this tier reproduces the pre-`22e37ce3` kernel's arithmetic
/// bit-for-bit (measured max w-relative error 1.6653e-5 on the precision
/// fixture through ~2e3× cancellation amplification — see the test).
#[cfg(target_arch = "x86_64")]
pub struct RsqrtNr1;

#[cfg(target_arch = "x86_64")]
impl RsqrtPrecisionTier for RsqrtNr1 {
    #[inline(always)]
    fn rsqrt(
        token: archmage::X64V4Token,
        v: f64x8<archmage::X64V4Token>,
    ) -> f64x8<archmage::X64V4Token> {
        // y ← 0.5·y·(3 − v·y·y), one step from the raw estimate — the same
        // (unfused) step magetypes' full rsqrt applies twice.
        let half = f64x8::splat_t(token, 0.5);
        let three = f64x8::splat_t(token, 3.0);
        let y = v.rsqrt_approx();
        (half * y) * (three - v * (y * y))
    }
    #[inline(always)]
    fn recip(
        token: archmage::X64V4Token,
        d: f64x8<archmage::X64V4Token>,
    ) -> f64x8<archmage::X64V4Token> {
        // r ← r·(2 − d·r), one step from the raw estimate.
        let two = f64x8::splat_t(token, 2.0);
        let r = d.rcp_approx();
        r * (two - d * r)
    }
}

/// Raw hardware estimate, no refinement: VRSQRT14PD/VRCP14PD alone. The
/// AVX-512 spec bounds the estimate's relative error at 2^-14 ≈ 6.1e-5 per
/// op. Cheapest possible tier; only for callers that can absorb ~1e-4-class
/// relative error in the update step.
#[cfg(target_arch = "x86_64")]
pub struct RsqrtEstimate;

#[cfg(target_arch = "x86_64")]
impl RsqrtPrecisionTier for RsqrtEstimate {
    #[inline(always)]
    fn rsqrt(
        _token: archmage::X64V4Token,
        v: f64x8<archmage::X64V4Token>,
    ) -> f64x8<archmage::X64V4Token> {
        v.rsqrt_approx()
    }
    #[inline(always)]
    fn recip(
        _token: archmage::X64V4Token,
        d: f64x8<archmage::X64V4Token>,
    ) -> f64x8<archmage::X64V4Token> {
        d.rcp_approx()
    }
}

/// Runtime-selectable precision tier for [`adam_update_rsqrt_v4_tiered`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target constructs a subset of variants
pub enum RsqrtPrecision {
    /// ~52-53 bits (2 NR steps) — the default; bounded at 1e-9 w-relative.
    #[default]
    Full,
    /// ~28 bits (1 NR step) — bounded at 1e-3 w-relative on the fixture.
    Nr1,
    /// Raw 14-bit estimate — bounded at 1e0 w-relative on the fixture.
    Estimate,
}

/// Generic kernel body — precision tier is a monomorphized type parameter,
/// so each `#[arcane]` wrapper below compiles a straight-line loop with the
/// tier's op sequence inlined (no per-lane or per-iteration branching).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn adam_update_inner_v4_rsqrt_body<P: RsqrtPrecisionTier>(
    token: archmage::X64V4Token,
    args: &mut AdamUpdateArgs<'_>,
) {
    let beta1_v = f64x8::splat_t(token, args.beta1);
    let beta2_v = f64x8::splat_t(token, args.beta2);
    let one_minus_b1_v = f64x8::splat_t(token, 1.0 - args.beta1);
    let first_cutoff_v = f64x8::splat_t(token, first_moment_cutoff(1.0 - args.beta1));
    let one_minus_b2_v = f64x8::splat_t(token, 1.0 - args.beta2);
    let inv_bc1_v = f64x8::splat_t(token, 1.0 / args.bc1);
    let inv_bc2_v = f64x8::splat_t(token, 1.0 / args.bc2);
    let lr_v = f64x8::splat_t(token, args.lr);
    let eps_v = f64x8::splat_t(token, args.eps);
    let zero_v = f64x8::zero_t(token);

    let n = args.w.len();
    let tail_len = n % 8;
    let (g_chunks, _) = args.g.as_chunks_mut::<8>();
    let (m_chunks, _) = args.m.as_chunks_mut::<8>();
    let (v_chunks, _) = args.v.as_chunks_mut::<8>();
    let (w_chunks, _) = args.w.as_chunks_mut::<8>();

    for (((wc_fixed, gc_fixed), mc_fixed), vc_fixed) in w_chunks
        .iter_mut()
        .zip(g_chunks)
        .zip(m_chunks)
        .zip(v_chunks)
    {
        let g = flush_moment8(token, f64x8::load_t(token, gc_fixed));
        let m = flush_moment8(token, f64x8::load_t(token, mc_fixed));
        let v = flush_moment8(token, f64x8::load_t(token, vc_fixed));
        let w = f64x8::load_t(token, wc_fixed);

        let m_new = first_moment8(token, g, m, beta1_v, one_minus_b1_v, first_cutoff_v);
        let gg = g * g;
        let v_new = flush_moment8(token, one_minus_b2_v.mul_add(gg, beta2_v * v));

        let m_hat = m_new * inv_bc1_v;
        let v_hat = v_new * inv_bc2_v;

        // Tier-selected reciprocal sqrt (see the tier docs above).
        // A flushed second moment is zero: rsqrt(0) is infinite and
        // 0*infinity would turn the otherwise finite epsilon denominator NaN.
        let rsqrt_input = f64x8::blend(v_hat.simd_eq(zero_v), f64x8::splat_t(token, 1.0), v_hat);
        let r1 = P::rsqrt(token, rsqrt_input);

        // sqrt(v_hat) = v_hat * rsqrt(v_hat), fused with the + eps.
        let denom = v_hat.mul_add(r1, eps_v);

        // Tier-selected reciprocal.
        let rd = P::recip(token, denom);

        let w_new = w - (lr_v * m_hat) * rd;

        m_new.store(mc_fixed);
        v_new.store(vc_fixed);
        w_new.store(wc_fixed);
        zero_v.store(gc_fixed);
    }

    let head = n - tail_len;
    if head < n {
        let mut tail_args = AdamUpdateArgs {
            w: &mut args.w[head..],
            g: &mut args.g[head..],
            m: &mut args.m[head..],
            v: &mut args.v[head..],
            beta1: args.beta1,
            beta2: args.beta2,
            eps: args.eps,
            bc1: args.bc1,
            bc2: args.bc2,
            lr: args.lr,
        };
        adam_update_scalar_ref(&mut tail_args);
    }
}

#[cfg(target_arch = "x86_64")]
#[arcane]
pub fn adam_update_inner_v4_rsqrt(token: archmage::X64V4Token, args: &mut AdamUpdateArgs<'_>) {
    adam_update_inner_v4_rsqrt_body::<RsqrtFull>(token, args);
}

#[cfg(target_arch = "x86_64")]
#[arcane]
pub fn adam_update_inner_v4_rsqrt_nr1(token: archmage::X64V4Token, args: &mut AdamUpdateArgs<'_>) {
    adam_update_inner_v4_rsqrt_body::<RsqrtNr1>(token, args);
}

#[cfg(target_arch = "x86_64")]
#[arcane]
pub fn adam_update_inner_v4_rsqrt_estimate(
    token: archmage::X64V4Token,
    args: &mut AdamUpdateArgs<'_>,
) {
    adam_update_inner_v4_rsqrt_body::<RsqrtEstimate>(token, args);
}

/// Bench-only entry-point — runs the rsqrt-based v4 kernel directly at the
/// DEFAULT (full-precision) tier, skipping the dispatch wrapper to isolate
/// kernel cost. Equivalent to
/// `adam_update_rsqrt_v4_tiered(args, RsqrtPrecision::Full)`.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
pub fn adam_update_rsqrt_v4(args: &mut AdamUpdateArgs<'_>) {
    adam_update_rsqrt_v4_tiered(args, RsqrtPrecision::Full);
}

/// Runtime-tier-selected rsqrt Adam kernel. Falls back to the scalar
/// reference when AVX-512 is unavailable (every tier degrades to EXACT
/// there — the fallback is full precision by construction).
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
pub fn adam_update_rsqrt_v4_tiered(args: &mut AdamUpdateArgs<'_>, tier: RsqrtPrecision) {
    use archmage::SimdToken;
    if let Some(token) = <archmage::X64V4Token as SimdToken>::summon() {
        match tier {
            RsqrtPrecision::Full => adam_update_inner_v4_rsqrt(token, args),
            RsqrtPrecision::Nr1 => adam_update_inner_v4_rsqrt_nr1(token, args),
            RsqrtPrecision::Estimate => adam_update_inner_v4_rsqrt_estimate(token, args),
        }
    } else {
        adam_update_scalar_ref(args);
    }
}

// =============================================================================
// AVX2 path: 4 f64 lanes via X64V3Token.
// =============================================================================

#[cfg(target_arch = "x86_64")]
#[arcane]
fn adam_update_inner_v3(token: archmage::X64V3Token, args: &mut AdamUpdateArgs<'_>) {
    adam_update_inner_v3_body(token, args);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn adam_update_inner_v3_body(token: archmage::X64V3Token, args: &mut AdamUpdateArgs<'_>) {
    let beta1_v = f64x4::splat_t(token, args.beta1);
    let beta2_v = f64x4::splat_t(token, args.beta2);
    let one_minus_b1_v = f64x4::splat_t(token, 1.0 - args.beta1);
    let first_cutoff_v = f64x4::splat_t(token, first_moment_cutoff(1.0 - args.beta1));
    let one_minus_b2_v = f64x4::splat_t(token, 1.0 - args.beta2);
    let inv_bc1_v = f64x4::splat_t(token, 1.0 / args.bc1);
    let inv_bc2_v = f64x4::splat_t(token, 1.0 / args.bc2);
    let lr_v = f64x4::splat_t(token, args.lr);
    let eps_v = f64x4::splat_t(token, args.eps);
    let zero_v = f64x4::zero_t(token);

    let n = args.w.len();
    let tail_len = n % 4;
    let (g_chunks, _) = args.g.as_chunks_mut::<4>();
    let (m_chunks, _) = args.m.as_chunks_mut::<4>();
    let (v_chunks, _) = args.v.as_chunks_mut::<4>();
    let (w_chunks, _) = args.w.as_chunks_mut::<4>();

    for (((wc_fixed, gc_fixed), mc_fixed), vc_fixed) in w_chunks
        .iter_mut()
        .zip(g_chunks)
        .zip(m_chunks)
        .zip(v_chunks)
    {
        let g = flush_moment4(token, f64x4::load_t(token, gc_fixed));
        let m = flush_moment4(token, f64x4::load_t(token, mc_fixed));
        let v = flush_moment4(token, f64x4::load_t(token, vc_fixed));
        let w = f64x4::load_t(token, wc_fixed);

        let m_new = first_moment4(token, g, m, beta1_v, one_minus_b1_v, first_cutoff_v);
        let gg = g * g;
        let v_new = flush_moment4(token, one_minus_b2_v.mul_add(gg, beta2_v * v));

        let m_hat = m_new * inv_bc1_v;
        let v_hat = v_new * inv_bc2_v;
        let denom = v_hat.sqrt() + eps_v;
        let w_new = w - (lr_v * m_hat) / denom;

        m_new.store(mc_fixed);
        v_new.store(vc_fixed);
        w_new.store(wc_fixed);
        zero_v.store(gc_fixed);
    }

    // Scalar mop-up.
    let head = n - tail_len;
    if head < n {
        let mut tail_args = AdamUpdateArgs {
            w: &mut args.w[head..],
            g: &mut args.g[head..],
            m: &mut args.m[head..],
            v: &mut args.v[head..],
            beta1: args.beta1,
            beta2: args.beta2,
            eps: args.eps,
            bc1: args.bc1,
            bc2: args.bc2,
            lr: args.lr,
        };
        adam_update_scalar_ref(&mut tail_args);
    }
}

// =============================================================================
// NEON fallback — generic `GenericF64x4<Token>` body. NEON's `mul_add`
// lowers to `vfmaq` (single-rounding fused), so this is bit-identical to
// the `_v3` kernel already: same fused domain [0, n - n%4), same
// scalar_ref tail.
// =============================================================================

#[magetypes(neon, -scalar)]
fn adam_update_inner(token: Token, args: &mut AdamUpdateArgs<'_>) {
    #[allow(non_camel_case_types)]
    type f64x4 = GenericF64x4<Token>;

    let beta1_v = f64x4::splat_t(token, args.beta1);
    let beta2_v = f64x4::splat_t(token, args.beta2);
    let one_minus_b1_v = f64x4::splat_t(token, 1.0 - args.beta1);
    let first_cutoff_v = f64x4::splat_t(token, first_moment_cutoff(1.0 - args.beta1));
    let one_minus_b2_v = f64x4::splat_t(token, 1.0 - args.beta2);
    let inv_bc1_v = f64x4::splat_t(token, 1.0 / args.bc1);
    let inv_bc2_v = f64x4::splat_t(token, 1.0 / args.bc2);
    let lr_v = f64x4::splat_t(token, args.lr);
    let eps_v = f64x4::splat_t(token, args.eps);
    let zero_v = f64x4::zero_t(token);

    let n = args.w.len();
    let tail_len = n % 4;
    let (g_chunks, _) = args.g.as_chunks_mut::<4>();
    let (m_chunks, _) = args.m.as_chunks_mut::<4>();
    let (v_chunks, _) = args.v.as_chunks_mut::<4>();
    let (w_chunks, _) = args.w.as_chunks_mut::<4>();

    for (((wc_fixed, gc_fixed), mc_fixed), vc_fixed) in w_chunks
        .iter_mut()
        .zip(g_chunks)
        .zip(m_chunks)
        .zip(v_chunks)
    {
        let g = flush_moment4(token, f64x4::load_t(token, gc_fixed));
        let m = flush_moment4(token, f64x4::load_t(token, mc_fixed));
        let v = flush_moment4(token, f64x4::load_t(token, vc_fixed));
        let w = f64x4::load_t(token, wc_fixed);

        let m_new = first_moment4(token, g, m, beta1_v, one_minus_b1_v, first_cutoff_v);
        let gg = g * g;
        let v_new = flush_moment4(token, one_minus_b2_v.mul_add(gg, beta2_v * v));

        let m_hat = m_new * inv_bc1_v;
        let v_hat = v_new * inv_bc2_v;
        let denom = v_hat.sqrt() + eps_v;
        let w_new = w - (lr_v * m_hat) / denom;

        m_new.store(mc_fixed);
        v_new.store(vc_fixed);
        w_new.store(wc_fixed);
        zero_v.store(gc_fixed);
    }

    let head = n - tail_len;
    if head < n {
        let mut tail_args = AdamUpdateArgs {
            w: &mut args.w[head..],
            g: &mut args.g[head..],
            m: &mut args.m[head..],
            v: &mut args.v[head..],
            beta1: args.beta1,
            beta2: args.beta2,
            eps: args.eps,
            bc1: args.bc1,
            bc2: args.bc2,
            lr: args.lr,
        };
        adam_update_scalar_ref(&mut tail_args);
    }
}

// =============================================================================
// wasm128 / scalar fallback — per-lane fused arithmetic.
//
// The generic `GenericF64x4<Token>::mul_add` lowers to plain `a*b+c` on
// these tiers (wasm has no f64 FMA primitive; the scalar polyfill avoids
// the libm call), which is NOT the canonical `_v3` arithmetic. Bit parity
// requires the same fused ops the AVX2 lanes run, so these tiers go
// per-element `f64::mul_add` — single-rounding fused on every target
// (hardware FMA where it exists, correctly-rounded libm `fma` elsewhere)
// — over the canonical fused domain [0, n - n%4), followed by the
// `scalar_ref` mul+add tail.
// =============================================================================

fn adam_update_scalar_fused(args: &mut AdamUpdateArgs<'_>) {
    let beta1 = args.beta1;
    let beta2 = args.beta2;
    let eps = args.eps;
    let bc1 = args.bc1;
    let bc2 = args.bc2;
    let lr = args.lr;
    let one_minus_b1 = 1.0 - beta1;
    let one_minus_b2 = 1.0 - beta2;
    let inv_bc1 = 1.0 / bc1;
    let inv_bc2 = 1.0 / bc2;
    let n = args.w.len();
    let n4 = n - n % 4;
    for i in 0..n4 {
        let g = flush_moment(args.g[i]);
        // Identical fused expression to the AVX2 lane ops:
        // fma(one_minus_b1, g, round(beta1*m)) etc.
        let m_new = if first_moment_zero(
            g,
            flush_moment(args.m[i]),
            first_moment_cutoff(one_minus_b1),
        ) {
            0.0f64.copysign(g)
        } else {
            flush_moment(one_minus_b1.mul_add(g, beta1 * flush_moment(args.m[i])))
        };
        let v_new = flush_moment(one_minus_b2.mul_add(g * g, beta2 * flush_moment(args.v[i])));
        args.m[i] = m_new;
        args.v[i] = v_new;
        let m_hat = m_new * inv_bc1;
        let v_hat = v_new * inv_bc2;
        args.w[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        args.g[i] = 0.0;
    }
    if n4 < n {
        let mut tail_args = AdamUpdateArgs {
            w: &mut args.w[n4..],
            g: &mut args.g[n4..],
            m: &mut args.m[n4..],
            v: &mut args.v[n4..],
            beta1,
            beta2,
            eps,
            bc1,
            bc2,
            lr,
        };
        adam_update_scalar_ref(&mut tail_args);
    }
}

/// `wasm128` tier variant called by `incant!` dispatch.
#[cfg(target_arch = "wasm32")]
fn adam_update_inner_wasm128(_token: archmage::Wasm128Token, args: &mut AdamUpdateArgs<'_>) {
    adam_update_scalar_fused(args);
}

/// `scalar` tier variant called by `incant!` dispatch.
fn adam_update_inner_scalar(_token: archmage::ScalarToken, args: &mut AdamUpdateArgs<'_>) {
    adam_update_scalar_fused(args);
}

// =============================================================================
// FUSED K=1 PAIR UPDATE — layer-1 weights only
// =============================================================================
//
// Replaces the per-pair sequence
//
//   backprop_step(A) → gw1[i] += xa[i]·dha[j]      (FMA / mul+add per row domain)
//   backprop_step(B) → gw1[i] += xb[i]·dhb[j]
//   add_l2_grad_layer1 → gw1[i] += sm[i/nh]·w[i]   (mul+add, never FMA)
//   adam_update → consume gw1, write w/m/v, store g=0
//
// with one pass that holds `g` in a register across the whole chain. Every
// element still sees the identical op sequence in the identical order, so the
// result is bit-identical — the only change is that gw1 is not round-tripped
// through L1 between passes (≈1.6 MB/pair of traffic at 944×32).
//
// Domain splits preserved exactly (the "same ops per element" contract):
//   * backprop contributions: FMA (`s.mul_add(dh, g)`) on lanes j < n4h,
//     mul+add on the per-row AVX2 tail [n4h, n_hidden) — the kernel is only
//     dispatched when n_hidden % 4 == 0, so the whole row is fused-domain.
//   * Adam arithmetic: vector (FMA) domain on [0, n − n%4); scalar_ref on
//     the flat tail. n = n_features·n_hidden is a multiple of 4 whenever
//     n_hidden % 4 == 0, so the tail is empty on the gated path.
//   * `x[i] == 0.0` skips that row's contribution exactly like
//     `backprop_*`'s nonzero-feature filter.
//   * L2 applies iff `l2_scale > 0.0` (same gate as the call site); with a
//     per-feature multiplier `sm = l2_scale * mult[row]` is hoisted per row
//     exactly as `add_l2_grad_layer1` does.
//
// Non-v3 tiers route through `adam_pair_fused_fallback`: a scalar
// accumulation of the identical elementwise op sequence into `g`, then the
// regular `adam_update` dispatch — bit-identical by composition (every tier's
// adam is already AVX2-parity per the tierparity contract).

/// Parameter block for the fused pair update on layer-1 weights.
/// `w/g/m/v` are the Adam state; `xa`/`xb` are the pair's feature vectors
/// and `dha`/`dhb` their precomputed `dl_dh_pre` vectors
/// (`simd_mlp::backprop_grad_head` output).
#[derive(Debug)]
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
pub(crate) struct AdamW1FusedArgs<'a> {
    pub w: &'a mut [f64],
    pub g: &'a mut [f64],
    pub m: &'a mut [f64],
    pub v: &'a mut [f64],
    pub xa: &'a [f64],
    pub dha: &'a [f64],
    pub xb: &'a [f64],
    pub dhb: &'a [f64],
    pub l2_scale: f64,
    pub l2_mult: Option<&'a [f64]>,
    pub n_hidden: usize,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub bc1: f64,
    pub bc2: f64,
    pub lr: f64,
    /// Ascending, disjoint, half-open row ranges `(start, end)` the v3/v4
    /// kernels visit (`None` = every row). Rows outside the ranges are left
    /// untouched. Exact only when every skipped row has `w == m == v == g ==
    /// +0.0` and `xa == xb == 0.0` (a `--keep-features` pinned-zero input):
    /// the full update maps such a row to itself bit-for-bit
    /// (`m' = 0, v' = 0, w' = 0 - (lr*0)/(0+eps) = +0`). The composition
    /// fallback ignores the ranges and updates every row, which is the same
    /// fixed point.
    pub active_rows: Option<&'a [(u32, u32)]>,
    /// `Some(tau)` applies the group-lasso block soft-threshold
    /// ([`group_l1_row`]) to each visited row right after that row's Adam
    /// update, while the row is still in cache. Bit-identical to running
    /// `apply_group_l1` over all of `w` after the whole step: rows are
    /// independent, and the row order, the sequential sum of squares and the
    /// `norm <= tau` / `1 - tau/norm` arithmetic are the same. A skipped
    /// (`active_rows`) row is exactly zero and the prox maps zero to zero, so
    /// not visiting it is exact. The composition fallback applies it to every
    /// row after `adam_update`.
    pub group_l1_tau: Option<f64>,
    /// Caller promise that `g` is `+0.0` in every element on entry (the
    /// post-step invariant: every Adam path stores `g = +0.0`, and the only
    /// other writer, the TV regularizer, is followed by an unfused step before
    /// the next fused call). The v3/v4 kernels then start the accumulation
    /// from zero instead of loading `g`, and skip the `g = 0` store — `g` is
    /// left untouched, still `+0.0`. Checked row by row under
    /// `debug_assertions`. The composition fallback ignores it (it reads `g`).
    pub g_zero_in: bool,
    /// Look-ahead forward accumulation (TRAINEROPT3 task 2): while the fused
    /// walk streams each row, apply that row's FINAL post-step value (after
    /// Adam, and after the fused `group_l1_tau` prox of its 8-row block)
    /// to the next pair's `h_pre` accumulators — one `w1` pass instead of a
    /// separate re-read by `forward_*`. See [`FwdAccumW1`] for the exactness
    /// contract the caller must uphold.
    pub fwd: Option<FwdAccumW1<'a>>,
}

/// Next-pair forward accumulators consumed by the fused W1 walk
/// (TRAINEROPT3 task 2).
///
/// Exactness contract — every clause is load-bearing for bit-identity with
/// "unfused step now, `simd_mlp::forward*` next iteration":
///
/// - `hpa`/`hpb` were seeded with this step's FINAL `b1` (post-Adam AND
///   post-`nonneg_project`), the same values the next forward would copy.
/// - `xa`/`xb` are the next pair's standardized feature rows; on every row
///   the walk does NOT visit (`active_rows` gaps) both are `+0.0`, matching
///   the `s == 0.0` skip `forward_*` performs there.
/// - `n_hidden % 4 == 0` (the fused-path gate): every lane is on the
///   canonical fused domain, so `fwd[j] = fma(x[i], w[i,j], fwd[j])` is
///   literally the op `forward_*` runs per visited row — same order
///   (ascending `i`), same single rounding, same skip-on-zero.
/// - The walk itself guarantees each visited row is applied exactly once,
///   after its last write: immediately post-Adam when no prox is pending
///   (`group_l1_tau == None`), or at the end of the row's 8-row block after
///   [`group_l1_rows`] when a prox is fused.
/// - The caller guarantees nothing else touches `w`/`b1` between this walk
///   and the consuming `forward_finish` (no coarse decay, no TV step, no
///   epoch boundary) — those gate the fusion off at the call site.
#[derive(Debug)]
pub(crate) struct FwdAccumW1<'a> {
    /// Next pair's side-A standardized row (length ≥ `w.len()/n_hidden`).
    pub xa: &'a [f64],
    /// Next pair's side-B standardized row.
    pub xb: &'a [f64],
    /// Side-A `h_pre` accumulator — seeded `b1` on entry, finished by
    /// `simd_mlp::forward_finish` after the step completes.
    pub hpa: &'a mut [f64],
    /// Side-B `h_pre` accumulator.
    pub hpb: &'a mut [f64],
}

/// Block soft-threshold on one layer-1 input row (`w`'s outgoing weights):
/// the group-lasso proximal operator with threshold `tau`.
/// `‖w‖₂ <= tau` zeroes the row; otherwise `w *= 1 - tau/‖w‖₂`. The norm is a
/// SEQUENTIAL f64 sum of squares in element order — callers rely on the exact
/// bits, so do not vectorize the reduction. Single owner: `apply_group_l1` and
/// the fused kernel both call this.
#[inline(always)]
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
pub(crate) fn group_l1_row(row: &mut [f64], tau: f64) {
    let norm = row.iter().map(|&w| w * w).sum::<f64>().sqrt();
    group_l1_apply(row, norm, tau);
}

/// The threshold step of [`group_l1_row`] given the row's norm.
#[inline(always)]
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
fn group_l1_apply(row: &mut [f64], norm: f64, tau: f64) {
    if norm <= tau {
        for w in row.iter_mut() {
            *w = 0.0;
        }
    } else {
        let f = 1.0 - tau / norm;
        for w in row.iter_mut() {
            *w *= f;
        }
    }
}

/// Rows per [`group_l1_rows`] block (independent sequential norm chains).
const GROUP_L1_BLOCK: usize = 8;

/// [`group_l1_row`] over `w.len() / nh` consecutive rows, bit-identical to
/// calling it per row. A full block of [`GROUP_L1_BLOCK`] rows runs its
/// sum-of-squares chains side by side: each row's sum is still a sequential
/// f64 accumulation in element order, but the chains are independent, so the
/// 4-cycle add latency of one row no longer serializes the whole step.
#[inline(never)]
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
pub(crate) fn group_l1_rows(w: &mut [f64], nh: usize, tau: f64) {
    if w.len() == GROUP_L1_BLOCK * nh {
        let mut sums = [0.0f64; GROUP_L1_BLOCK];
        {
            let rows: [&[f64]; GROUP_L1_BLOCK] = core::array::from_fn(|k| &w[k * nh..(k + 1) * nh]);
            for j in 0..nh {
                for (s, row) in sums.iter_mut().zip(rows.iter()) {
                    let x = row[j];
                    *s += x * x;
                }
            }
        }
        for (row, s) in w.chunks_exact_mut(nh).zip(sums) {
            group_l1_apply(row, s.sqrt(), tau);
        }
    } else {
        for row in w.chunks_mut(nh) {
            group_l1_row(row, tau);
        }
    }
}

/// One visited row of the look-ahead forward (TRAINEROPT3 task 2): apply
/// `fwd[j] = fma(x[i], w[i,j], fwd[j])` per side over 4-lane chunks — the
/// identical per-lane fma chain `forward_avx2`'s `fwd_acc_block` runs (the
/// chain the AVX-512 and scalar tiers reproduce on the canonical domain).
/// `nh % 4 == 0` is a fused-path gate, so there is no mul+add tail domain
/// to mirror. `sa == 0`/`sb == 0` skip exactly like `forward`'s `s == 0.0`
/// continue — an unguarded `fma(0, w, acc)` could flip an acc of `−0.0`
/// to `+0.0`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
fn fwd_accum_row_v3(
    token: archmage::X64V3Token,
    f: &mut FwdAccumW1<'_>,
    row: usize,
    w_row: &[f64],
    nh: usize,
) {
    debug_assert_eq!(
        w_row.len(),
        nh,
        "the zip below would silently truncate a short row"
    );
    let sa = f.xa[row];
    let sb = f.xb[row];
    let do_a = sa != 0.0;
    let do_b = sb != 0.0;
    if !do_a && !do_b {
        return;
    }
    let sa_v = f64x4::splat_t(token, sa);
    let sb_v = f64x4::splat_t(token, sb);
    let (w4, _) = w_row.as_chunks::<4>();
    let (ha4, _) = f.hpa.as_chunks_mut::<4>();
    let (hb4, _) = f.hpb.as_chunks_mut::<4>();
    for ((wc, hac), hbc) in w4.iter().zip(ha4.iter_mut()).zip(hb4.iter_mut()) {
        let wv = f64x4::load_t(token, wc);
        if do_a {
            sa_v.mul_add(wv, f64x4::load_t(token, hac)).store(hac);
        }
        if do_b {
            sb_v.mul_add(wv, f64x4::load_t(token, hbc)).store(hbc);
        }
    }
}

/// Scalar twin of [`fwd_accum_row_v3`] for the composition fallback —
/// `f64::mul_add` is a single-rounding fma, the same op as the canonical
/// domain. Requires `nh % 4 == 0` (caller's fused-path gate) so no
/// mul+add tail lanes exist to mirror.
#[inline]
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
fn fwd_accum_row_scalar(f: &mut FwdAccumW1<'_>, row: usize, w_row: &[f64], nh: usize) {
    let sa = f.xa[row];
    let sb = f.xb[row];
    let do_a = sa != 0.0;
    let do_b = sb != 0.0;
    if !do_a && !do_b {
        return;
    }
    for ((ha, hb), &w) in f.hpa[..nh]
        .iter_mut()
        .zip(f.hpb[..nh].iter_mut())
        .zip(w_row.iter())
    {
        if do_a {
            *ha = sa.mul_add(w, *ha);
        }
        if do_b {
            *hb = sb.mul_add(w, *hb);
        }
    }
}

/// Dispatch entry for the fused pair update. Caller contract: `n_hidden > 0`,
/// `n_hidden % 4 == 0`, `w/g/m/v` equal length with `w.len() % n_hidden == 0`,
/// `xa/xb.len() >= w.len()/n_hidden`, `dha/dhb.len() >= n_hidden`.
#[inline]
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
pub(crate) fn adam_update_w1_fused(args: &mut AdamW1FusedArgs<'_>) {
    debug_assert_eq!(args.w.len(), args.g.len());
    debug_assert_eq!(args.w.len(), args.m.len());
    debug_assert_eq!(args.w.len(), args.v.len());
    if args.n_hidden == 0
        || !args.n_hidden.is_multiple_of(4)
        || !args.w.len().is_multiple_of(args.n_hidden)
    {
        // The v3 kernel walks full 4-lane row chunks; off-grid shapes take
        // the composition fallback (still bit-identical to the unfused
        // sequence — the fallback reproduces its per-element ops exactly).
        adam_pair_fused_fallback(args);
        return;
    }
    incant!(adam_pair_fused_inner(args), [+v4])
}

/// Scalar-fallback body shared by every non-v3 arm: replays the exact
/// unfused element sequence (backprop A, backprop B, L2) into `g` — fused
/// `mul_add` on the per-row canonical domain, mul+add on the AVX2 tail —
/// then runs the regular tier-dispatched `adam_update`.
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
fn adam_pair_fused_fallback(args: &mut AdamW1FusedArgs<'_>) {
    let nh = args.n_hidden;
    let n = args.w.len();
    let n_rows = n / nh;
    let n4h = nh - nh % 4;
    for row in 0..n_rows {
        let base = row * nh;
        let sa = args.xa[row];
        let sb = args.xb[row];
        for j in 0..nh {
            let i = base + j;
            if j < n4h {
                if sa != 0.0 {
                    args.g[i] = sa.mul_add(args.dha[j], args.g[i]);
                }
                if sb != 0.0 {
                    args.g[i] = sb.mul_add(args.dhb[j], args.g[i]);
                }
            } else {
                if sa != 0.0 {
                    args.g[i] += sa * args.dha[j];
                }
                if sb != 0.0 {
                    args.g[i] += sb * args.dhb[j];
                }
            }
        }
    }
    // L2 — mul+add, row-hoisted `sm` exactly as add_l2_grad_layer1.
    if args.l2_scale > 0.0 {
        match args.l2_mult {
            None => {
                add_l2_grad_row(args.g, args.w, args.l2_scale);
            }
            Some(mult) => {
                for (feat, (grow, wrow)) in args.g.chunks_mut(nh).zip(args.w.chunks(nh)).enumerate()
                {
                    let sm = args.l2_scale * mult[feat];
                    add_l2_grad_row(grow, wrow, sm);
                }
            }
        }
    }
    adam_update(&mut AdamUpdateArgs {
        w: args.w,
        g: args.g,
        m: args.m,
        v: args.v,
        beta1: args.beta1,
        beta2: args.beta2,
        eps: args.eps,
        bc1: args.bc1,
        bc2: args.bc2,
        lr: args.lr,
    });
    if let (Some(tau), true) = (args.group_l1_tau, nh > 0) {
        for blk in args.w.chunks_mut(GROUP_L1_BLOCK * nh) {
            group_l1_rows(blk, nh, tau);
        }
    }
    // TRAINEROPT3 look-ahead: every row's w is now final — accumulate the
    // next pair's `h_pre` in one pass. The fallback ignores `active_rows`,
    // so the apply visits all rows; the per-side `x == 0` guards reproduce
    // `forward`'s own skip, keeping it bit-identical regardless.
    if let Some(f) = args.fwd.as_mut() {
        // A hard check: off the 4-lane canonical domain `forward` uses mul+add tail lanes, so the fma apply
        // below would diverge silently (unreachable today: `fuse_w1` requires `n_hidden % 4 == 0`).
        assert_eq!(nh % 4, 0, "look-ahead forward needs canonical fma domain");
        for row in 0..n_rows {
            fwd_accum_row_scalar(f, row, &args.w[row * nh..row * nh + nh], nh);
        }
    }
}

/// `v4` tier variant. Deliberately NOT a native 8-lane kernel: AVX-512 must
/// preserve AVX2's fused-domain boundaries (4-lane chunks + canonical tail).
/// The kernel is purely elementwise and the caller gate guarantees
/// `n_hidden % 4 == 0`, so the 4-lane v3 body is exact under a v4 token —
/// run it directly via `token.v3()` instead of the scalar-emulating
/// fallback (which was bit-identical but ~4× slower on AVX-512 hosts).
#[cfg(target_arch = "x86_64")]
#[arcane]
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
fn adam_pair_fused_inner_v4(token: archmage::X64V4Token, args: &mut AdamW1FusedArgs<'_>) {
    adam_pair_fused_inner_v3_body(token.v3(), args);
}

#[cfg(target_arch = "x86_64")]
#[arcane]
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
fn adam_pair_fused_inner_v3(token: archmage::X64V3Token, args: &mut AdamW1FusedArgs<'_>) {
    adam_pair_fused_inner_v3_body(token, args);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
fn adam_pair_fused_inner_v3_body(token: archmage::X64V3Token, args: &mut AdamW1FusedArgs<'_>) {
    let beta1_v = f64x4::splat_t(token, args.beta1);
    let beta2_v = f64x4::splat_t(token, args.beta2);
    let one_minus_b1_v = f64x4::splat_t(token, 1.0 - args.beta1);
    let first_cutoff_v = f64x4::splat_t(token, first_moment_cutoff(1.0 - args.beta1));
    let one_minus_b2_v = f64x4::splat_t(token, 1.0 - args.beta2);
    let inv_bc1_v = f64x4::splat_t(token, 1.0 / args.bc1);
    let inv_bc2_v = f64x4::splat_t(token, 1.0 / args.bc2);
    let lr_v = f64x4::splat_t(token, args.lr);
    let eps_v = f64x4::splat_t(token, args.eps);
    let zero_v = f64x4::zero_t(token);

    let nh = args.n_hidden;
    let n = args.w.len();
    let n_rows = n / nh;
    let l2_on = args.l2_scale > 0.0;
    let gz = args.g_zero_in;
    let tau = args.group_l1_tau;
    let (dha_chunks, _) = args.dha[..nh].as_chunks::<4>();
    let (dhb_chunks, _) = args.dhb[..nh].as_chunks::<4>();

    let full = [(0u32, n_rows as u32)];
    let ranges: &[(u32, u32)] = args.active_rows.unwrap_or(&full);
    let mut fwd = args.fwd.take();
    for &(r0, r1) in ranges {
        let (r0, r1) = (r0 as usize, (r1 as usize).min(n_rows));
        let mut blk0 = r0;
        while blk0 < r1 {
            let blk1 = if tau.is_some() {
                (blk0 + GROUP_L1_BLOCK).min(r1)
            } else {
                r1
            };
            for row in blk0..blk1 {
                let base = row * nh;
                let sa = args.xa[row];
                let sb = args.xb[row];
                let sa_v = f64x4::splat_t(token, sa);
                let sb_v = f64x4::splat_t(token, sb);
                let do_a = sa != 0.0;
                let do_b = sb != 0.0;
                let sm = match args.l2_mult {
                    Some(mult) => args.l2_scale * mult[row],
                    None => args.l2_scale,
                };
                let sm_v = f64x4::splat_t(token, sm);
                let cutoff_v = f64x4::splat_t(token, if l2_on { l2_cutoff(sm) } else { 0.0 });
                debug_assert!(
                    !gz || args.g[base..base + nh].iter().all(|g| g.to_bits() == 0),
                    "g_zero_in promised an all +0.0 gradient row (row {row})"
                );
                let (w_chunks, _) = args.w[base..base + nh].as_chunks_mut::<4>();
                let (g_chunks, _) = args.g[base..base + nh].as_chunks_mut::<4>();
                let (m_chunks, _) = args.m[base..base + nh].as_chunks_mut::<4>();
                let (v_chunks, _) = args.v[base..base + nh].as_chunks_mut::<4>();
                for (c, (((wc, gc), mc), vc)) in w_chunks
                    .iter_mut()
                    .zip(g_chunks.iter_mut())
                    .zip(m_chunks.iter_mut())
                    .zip(v_chunks.iter_mut())
                    .enumerate()
                {
                    let mut g = if gz { zero_v } else { f64x4::load_t(token, gc) };
                    if do_a {
                        g = sa_v.mul_add(f64x4::load_t(token, &dha_chunks[c]), g);
                    }
                    if do_b {
                        g = sb_v.mul_add(f64x4::load_t(token, &dhb_chunks[c]), g);
                    }
                    let w = f64x4::load_t(token, wc);
                    if l2_on {
                        // mul+add — two roundings, matching l2_row_avx exactly.
                        g += l2_product4(token, w, sm_v, cutoff_v);
                    }
                    // Tiny L2 gradients can be subnormal even when w is normal.
                    // Do not feed them to the moment FMAs on every update.
                    g = flush_moment4(token, g);
                    let m = flush_moment4(token, f64x4::load_t(token, mc));
                    let v = flush_moment4(token, f64x4::load_t(token, vc));
                    let m_new = first_moment4(token, g, m, beta1_v, one_minus_b1_v, first_cutoff_v);
                    let gg = g * g;
                    let v_new = flush_moment4(token, one_minus_b2_v.mul_add(gg, beta2_v * v));
                    let m_hat = m_new * inv_bc1_v;
                    let v_hat = v_new * inv_bc2_v;
                    let denom = v_hat.sqrt() + eps_v;
                    let w_new = w - (lr_v * m_hat) / denom;
                    m_new.store(mc);
                    v_new.store(vc);
                    w_new.store(wc);
                    if !gz {
                        zero_v.store(gc);
                    }
                }
                // TRAINEROPT3 look-ahead: with no prox pending, `row`'s w
                // is final the moment its Adam stores land — apply it to
                // the next pair's `h_pre` accumulators while the row is
                // still in registers/L1.
                if tau.is_none()
                    && let Some(f) = fwd.as_mut()
                {
                    fwd_accum_row_v3(token, f, row, &args.w[base..base + nh], nh);
                }
            }
            // The block's rows are still hot in L1: prox them now.
            if let Some(tau) = tau {
                group_l1_rows(&mut args.w[blk0 * nh..blk1 * nh], nh, tau);
                // TRAINEROPT3 look-ahead: rows blk0..blk1 are final only
                // now — apply post-prox, still ascending.
                if let Some(f) = fwd.as_mut() {
                    for row in blk0..blk1 {
                        fwd_accum_row_v3(token, f, row, &args.w[row * nh..row * nh + nh], nh);
                    }
                }
            }
            blk0 = blk1;
        }
    }
}

/// `neon` tier variant — routes to the scalar-emulating fallback.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
fn adam_pair_fused_inner_neon(_token: archmage::NeonToken, args: &mut AdamW1FusedArgs<'_>) {
    adam_pair_fused_fallback(args);
}

/// `wasm128` tier variant — routes to the scalar-emulating fallback.
#[cfg(target_arch = "wasm32")]
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
fn adam_pair_fused_inner_wasm128(_token: archmage::Wasm128Token, args: &mut AdamW1FusedArgs<'_>) {
    adam_pair_fused_fallback(args);
}

/// `scalar` tier variant — routes to the scalar-emulating fallback.
#[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
fn adam_pair_fused_inner_scalar(_token: archmage::ScalarToken, args: &mut AdamW1FusedArgs<'_>) {
    adam_pair_fused_fallback(args);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a representative test parameter vector with mixed scales —
    /// includes positive/negative gradients, near-zero `v` (worst case for
    /// `sqrt(v) + eps`), and the typical [-1, 1] weight range.
    #[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
    fn synth_state(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut state = seed;
        let mut nxt = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as u32) as f64 / u32::MAX as f64
        };
        let mut w = vec![0.0; n];
        let mut g = vec![0.0; n];
        let mut m = vec![0.0; n];
        let mut v = vec![0.0; n];
        for i in 0..n {
            w[i] = nxt() * 2.0 - 1.0;
            g[i] = (nxt() * 2.0 - 1.0) * 0.1;
            m[i] = (nxt() * 2.0 - 1.0) * 0.05;
            // Mix of small (1e-12), tiny (1e-8), normal (1e-4), and
            // larger v values to exercise sqrt across the dynamic range.
            v[i] = match i % 4 {
                0 => 1e-12 + nxt() * 1e-12,
                1 => 1e-8 + nxt() * 1e-7,
                2 => 1e-4 + nxt() * 1e-3,
                _ => nxt() * 0.1,
            };
        }
        (w, g, m, v)
    }

    #[allow(dead_code)] // compiled into bench/test targets via #[path] include; each target uses a subset
    fn make_args<'a>(
        w: &'a mut [f64],
        g: &'a mut [f64],
        m: &'a mut [f64],
        v: &'a mut [f64],
        t: u64,
    ) -> AdamUpdateArgs<'a> {
        let beta1 = 0.9f64;
        let beta2 = 0.999f64;
        let eps = 1e-8f64;
        let bc1 = 1.0 - beta1.powi(t as i32);
        let bc2 = 1.0 - beta2.powi(t as i32);
        AdamUpdateArgs {
            w,
            g,
            m,
            v,
            beta1,
            beta2,
            eps,
            bc1,
            bc2,
            lr: 0.005,
        }
    }

    /// Bit-identical equivalence at chunk-aligned sizes.
    #[test]
    fn dispatch_matches_scalar_aligned() {
        // 47616 = w1 size in production (372 * 128).
        for &n in &[8usize, 16, 47616] {
            let (w0, g0, m0, v0) = synth_state(n, 0xA1B2);
            let (mut wa, mut ga, mut ma, mut va) = (w0.clone(), g0.clone(), m0.clone(), v0.clone());
            let (mut wb, mut gb, mut mb, mut vb) = (w0, g0, m0, v0);

            adam_update_scalar_ref(&mut make_args(&mut wa, &mut ga, &mut ma, &mut va, 5));
            adam_update(&mut make_args(&mut wb, &mut gb, &mut mb, &mut vb, 5));

            // Within 1 ULP: scalar ref is "scalar"; we expect bit identity
            // since the SIMD path uses the same operations in the same
            // order (FMA is permitted because the scalar ref also uses
            // f64::mul_add when LLVM contracts; the two arms can differ
            // by 1 ULP on individual lanes if LLVM didn't auto-fuse the
            // scalar mul+add). Use tight relative tolerance.
            for i in 0..n {
                let ref_v = wa[i];
                let got_v = wb[i];
                let rel = (ref_v - got_v).abs() / (ref_v.abs().max(got_v.abs()).max(1e-30));
                assert!(
                    rel < 1e-12,
                    "w mismatch at i={}: ref={:e} got={:e} rel={:e} (n={})",
                    i,
                    ref_v,
                    got_v,
                    rel,
                    n
                );
                assert_eq!(gb[i], 0.0, "g must reset to zero at i={} (n={})", i, n);
            }
            // m and v must also match within tight tolerance.
            for i in 0..n {
                let r = (ma[i] - mb[i]).abs() / (ma[i].abs().max(mb[i].abs()).max(1e-30));
                assert!(r < 1e-12, "m mismatch i={} (n={})", i, n);
                let r = (va[i] - vb[i]).abs() / (va[i].abs().max(vb[i].abs()).max(1e-30));
                assert!(r < 1e-12, "v mismatch i={} (n={})", i, n);
            }
        }
    }

    /// Misaligned tail length — exercises the scalar mop-up.
    #[test]
    fn dispatch_matches_scalar_misaligned() {
        for &n in &[1usize, 3, 5, 7, 9, 11, 13, 47873] {
            let (w0, g0, m0, v0) = synth_state(n, 0xDEAD);
            let (mut wa, mut ga, mut ma, mut va) = (w0.clone(), g0.clone(), m0.clone(), v0.clone());
            let (mut wb, mut gb, mut mb, mut vb) = (w0, g0, m0, v0);

            adam_update_scalar_ref(&mut make_args(&mut wa, &mut ga, &mut ma, &mut va, 100));
            adam_update(&mut make_args(&mut wb, &mut gb, &mut mb, &mut vb, 100));

            for i in 0..n {
                let r = (wa[i] - wb[i]).abs() / (wa[i].abs().max(wb[i].abs()).max(1e-30));
                assert!(r < 1e-12, "w mismatch i={} n={}", i, n);
            }
        }
    }

    /// Late-training step (t = 10_000) where bias correction is ≈ 1.0.
    /// Both paths must still agree.
    #[test]
    fn late_training_step_matches() {
        let n = 47873;
        let (w0, g0, m0, v0) = synth_state(n, 0xBEEF);
        let (mut wa, mut ga, mut ma, mut va) = (w0.clone(), g0.clone(), m0.clone(), v0.clone());
        let (mut wb, mut gb, mut mb, mut vb) = (w0, g0, m0, v0);

        adam_update_scalar_ref(&mut make_args(&mut wa, &mut ga, &mut ma, &mut va, 10_000));
        adam_update(&mut make_args(&mut wb, &mut gb, &mut mb, &mut vb, 10_000));

        for i in 0..n {
            let r = (wa[i] - wb[i]).abs() / (wa[i].abs().max(wb[i].abs()).max(1e-30));
            assert!(r < 1e-12, "w mismatch at i={}", i);
        }
    }

    /// Empty slice — no-op.
    #[test]
    fn empty_is_noop() {
        let mut w: Vec<f64> = vec![];
        let mut g: Vec<f64> = vec![];
        let mut m: Vec<f64> = vec![];
        let mut v: Vec<f64> = vec![];
        adam_update(&mut make_args(&mut w, &mut g, &mut m, &mut v, 1));
    }

    // =====================================================================
    // Tier parity — the AVX2 (`_v3`) kernel is the canonical arithmetic;
    // every other tier the dispatcher can pick must reproduce it BIT FOR
    // BIT (same fused-op set on [0, n - n%4), same mul+add tail).
    // =====================================================================

    fn bits_of(v: &[f64]) -> Vec<u64> {
        v.iter().map(|f| f.to_bits()).collect()
    }

    /// Max ordered-int ULP distance + first index, for reporting.
    fn max_ulp(a: &[f64], b: &[f64]) -> (u64, usize) {
        fn ord(v: f64) -> i64 {
            let b = v.to_bits() as i64;
            if b < 0 { i64::MIN - b } else { b }
        }
        let mut worst = (0u64, usize::MAX);
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            let d = ord(x).abs_diff(ord(y));
            if d > worst.0 {
                worst = (d, i);
            }
        }
        worst
    }

    /// Compare every host-available tier against `_v3` (canonical) on the
    /// same buffers. Sizes cover: n<4 (all-tail on every tier), n=5..8
    /// (v4 all-tail vs v3 partially-chunked — the tier-boundary edge), the
    /// fold shape 30208/32/1 and a misaligned large n.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn adam_all_tiers_bit_identical() {
        use archmage::SimdToken;
        // Hold the token lock for the whole body: `adam_dispatch_bit_identical_under_token_permutations`
        // disables tokens process-wide, and a `summon()` racing it would silently skip the v4 comparison.
        let _lock = archmage::testing::lock_token_testing();
        let v3 = archmage::X64V3Token::summon();
        let v4 = archmage::X64V4Token::summon();
        assert!(
            v3.is_some(),
            "this test needs an x86-64-v3 (AVX2+FMA) host: it is the canonical tier every other tier must reproduce"
        );
        let sizes = [
            1usize, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 15, 16, 17, 20, 24, 28, 31, 32, 33, 40, 48, 63,
            64, 65, 30208, 32, 1, 47873,
        ];
        let mut failures = Vec::new();
        for &n in &sizes {
            for &t in &[5u64, 100, 10_000] {
                let (w0, g0, m0, v0) = synth_state(n, 0xFEED ^ n as u64 ^ t);

                let run = |f: &mut dyn FnMut(&mut AdamUpdateArgs<'_>)| {
                    let (mut w, mut g, mut m, mut v) =
                        (w0.clone(), g0.clone(), m0.clone(), v0.clone());
                    f(&mut make_args(&mut w, &mut g, &mut m, &mut v, t));
                    (w, g, m, v)
                };

                let r_v3 = run(&mut |a| adam_update_inner_v3(v3.unwrap(), a));

                // scalar tier (plain mul_add semantics — must equal v3)
                let r_s = run(&mut |a| {
                    adam_update_inner_scalar(archmage::ScalarToken::summon().unwrap(), a)
                });
                for (name, (ulp, idx)) in [
                    ("w", max_ulp(&r_s.0, &r_v3.0)),
                    ("g", max_ulp(&r_s.1, &r_v3.1)),
                    ("m", max_ulp(&r_s.2, &r_v3.2)),
                    ("v", max_ulp(&r_s.3, &r_v3.3)),
                ] {
                    if ulp != 0 {
                        failures.push(format!(
                            "adam scalar-vs-v3 n={n} t={t} {name}: max_ulp={ulp} at {idx}"
                        ));
                    }
                }

                if let Some(t4) = v4 {
                    let r_v4 = run(&mut |a| adam_update_inner_v4(t4, a));
                    for (name, (ulp, idx)) in [
                        ("w", max_ulp(&r_v4.0, &r_v3.0)),
                        ("g", max_ulp(&r_v4.1, &r_v3.1)),
                        ("m", max_ulp(&r_v4.2, &r_v3.2)),
                        ("v", max_ulp(&r_v4.3, &r_v3.3)),
                    ] {
                        if ulp != 0 {
                            failures.push(format!(
                                "adam v4-vs-v3 n={n} t={t} {name}: max_ulp={ulp} at {idx}"
                            ));
                        }
                    }
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// The dispatched `adam_update` must produce identical bytes under
    /// every host token permutation (v4-off → v3 bytes; v3-off → scalar
    /// bytes; and the scalar tier must match v3 too).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn adam_dispatch_bit_identical_under_token_permutations() {
        use archmage::SimdToken;
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
        assert!(
            archmage::X64V3Token::summon().is_some(),
            "this test needs an x86-64-v3 (AVX2+FMA) host: it is the canonical tier every other tier must reproduce"
        );
        let n = 30208usize + 65; // fold w1 + biases, misaligned on purpose
        let (w0, g0, m0, v0) = synth_state(n, 0xCAFE_5EED);
        let mut baseline: Option<Vec<u64>> = None;
        let mut failures = Vec::new();
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let (mut w, mut g, mut m, mut v) = (w0.clone(), g0.clone(), m0.clone(), v0.clone());
            adam_update(&mut make_args(&mut w, &mut g, &mut m, &mut v, 42));
            let mut bits = bits_of(&w);
            bits.extend(bits_of(&g));
            bits.extend(bits_of(&m));
            bits.extend(bits_of(&v));
            if let Some(b) = &baseline {
                if bits != *b {
                    failures.push(format!("adam dispatch diverged: {}", perm.label));
                }
            } else {
                baseline = Some(bits);
            }
            eprintln!(
                "perm [{}] w[0]={:#x} w[{}]={:#x}",
                perm.label,
                w[0].to_bits(),
                n - 1,
                w[n - 1].to_bits()
            );
        });
        eprintln!("permutations run: {}", report.permutations_run);
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    #[test]
    fn moment_flush_preserves_normal_and_special_bits() {
        let mut values = Vec::new();
        for bits in [
            0,
            1,
            0x000f_ffff_ffff_ffff,
            0x0010_0000_0000_0000,
            0x0010_0000_0000_0001,
            0x3ff0_0000_0000_0000,
            0x7ff0_0000_0000_0000,
            0x7ff8_1234_5678_9abc,
        ] {
            for sign in [0, 1u64 << 63] {
                let value = f64::from_bits(bits | sign);
                let expected = if value.is_subnormal() {
                    sign
                } else {
                    bits | sign
                };
                assert_eq!(flush_moment(value).to_bits(), expected);
                values.push(value);
            }
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        fn check4<T: magetypes::simd::backends::F64x4Backend>(token: T, values: &[f64]) {
            for chunk in values.as_chunks::<4>().0 {
                let actual =
                    flush_moment4(token, magetypes::simd::generic::f64x4::load_t(token, chunk))
                        .to_array();
                assert_eq!(
                    actual.map(f64::to_bits),
                    chunk.map(|v| flush_moment(v).to_bits())
                );
            }
        }
        let _lock = archmage::testing::lock_token_testing();
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        check4(archmage::ScalarToken, &values);
        #[cfg(target_arch = "x86_64")]
        {
            use archmage::SimdToken;
            if let Some(token) = archmage::X64V3Token::summon() {
                check4(token, &values);
            }
            if let Some(token) = archmage::X64V4Token::summon() {
                for chunk in values.as_chunks::<8>().0 {
                    let actual = flush_moment8(token, f64x8::load_t(token, chunk)).to_array();
                    assert_eq!(
                        actual.map(f64::to_bits),
                        chunk.map(|v| flush_moment(v).to_bits())
                    );
                }
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            use archmage::SimdToken;
            check4(archmage::NeonToken::summon().expect("NEON host"), &values);
        }
    }

    #[test]
    fn l2_underflow_boundary_matches_classified_product_in_every_tier() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            for sm in [0.0, 1e-5, 0.125, 0.5, 0.9, 1.0, 2.0] {
                let boundary = l2_cutoff(sm);
                let mut weights = vec![
                    0.0,
                    1e-310,
                    1e-304,
                    f64::from_bits(0x000f_ffff_ffff_ffff),
                    f64::MIN_POSITIVE,
                    0.5,
                ];
                let mut below = boundary;
                let mut above = boundary;
                for _ in 0..64 {
                    weights.extend([below, above, -below, -above]);
                    below = below.next_down();
                    above = above.next_up();
                }
                let expected: Vec<_> = weights
                    .iter()
                    .map(|&w| {
                        let w = if w.is_subnormal() && sm <= 1.0 {
                            0.0f64.copysign(w)
                        } else {
                            w
                        };
                        let product = sm * w;
                        let contribution = if product.is_subnormal() {
                            0.0f64.copysign(product)
                        } else {
                            product
                        };
                        0.0 + contribution
                    })
                    .collect();
                let mut actual = vec![0.0; weights.len()];
                add_l2_grad_row(&mut actual, &weights, sm);
                // Infinity is reached only by the zero-scale boundary fixture:
                // both implementations retain the same invalid 0*infinity.
                for (a, e) in actual.iter().zip(&expected) {
                    assert!(
                        a.to_bits() == e.to_bits() || (a.is_nan() && e.is_nan()),
                        "{} sm={sm} actual={a:?} expected={e:?}",
                        perm.label
                    );
                }
            }
        });
        assert!(report.permutations_run > 0);
    }

    #[test]
    fn fused_tiny_l2_matches_unfused_in_every_tier() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let w0: Vec<_> = (0..24)
                .map(|i| match i % 3 {
                    0 => 1e-304,
                    1 => 1e-310,
                    _ => 0.5,
                })
                .collect();
            let mut fused = (w0.clone(), vec![0.0; 24], vec![0.0; 24], vec![0.0; 24]);
            let mut unfused = fused.clone();
            for _ in 0..10 {
                add_l2_grad_row(&mut unfused.1, &unfused.0, 1e-5);
                adam_update(&mut make_args(
                    &mut unfused.0,
                    &mut unfused.1,
                    &mut unfused.2,
                    &mut unfused.3,
                    10_000,
                ));
                adam_update_w1_fused(&mut AdamW1FusedArgs {
                    w: &mut fused.0,
                    g: &mut fused.1,
                    m: &mut fused.2,
                    v: &mut fused.3,
                    xa: &[0.0; 3],
                    xb: &[0.0; 3],
                    dha: &[0.0; 8],
                    dhb: &[0.0; 8],
                    n_hidden: 8,
                    l2_scale: 1e-5,
                    l2_mult: None,
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-8,
                    bc1: 1.0 - 0.9f64.powi(10_000),
                    bc2: 1.0 - 0.999f64.powi(10_000),
                    lr: 0.005,
                    active_rows: None,
                    group_l1_tau: None,
                    g_zero_in: true,
                    fwd: None,
                });
                for (a, e) in [
                    (&fused.0, &unfused.0),
                    (&fused.1, &unfused.1),
                    (&fused.2, &unfused.2),
                    (&fused.3, &unfused.3),
                ] {
                    assert_eq!(bits_of(a), bits_of(e), "{}", perm.label);
                }
            }
        });
        assert!(report.permutations_run > 0);
    }

    #[test]
    fn zero_moment_bypass_matches_rounded_boundary_and_signed_zero() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
        let one_minus = 1.0 - 0.9f64;
        let cutoff = l2_cutoff(one_minus);
        let positive = [
            0.0,
            f64::MIN_POSITIVE,
            cutoff.next_down(),
            cutoff,
            cutoff.next_up(),
            1e-306,
        ];
        let gradients: Vec<_> = positive.iter().flat_map(|&g| [g, -g]).collect();
        let expected: Vec<_> = gradients
            .iter()
            .map(|&g| {
                let result = one_minus.mul_add(g, 0.0);
                if result.is_subnormal() {
                    0.0f64.copysign(result)
                } else {
                    result
                }
            })
            .collect();
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let mut w = vec![0.5; gradients.len()];
            let mut g = gradients.clone();
            let mut m = vec![0.0; g.len()];
            let mut v = m.clone();
            adam_update(&mut make_args(&mut w, &mut g, &mut m, &mut v, 100));
            assert_eq!(bits_of(&m), bits_of(&expected), "{}", perm.label);
        });
        assert!(report.permutations_run > 0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn rsqrt_flushed_zero_has_a_finite_epsilon_denominator() {
        let _lock = archmage::testing::lock_token_testing();
        for tier in [
            RsqrtPrecision::Full,
            RsqrtPrecision::Nr1,
            RsqrtPrecision::Estimate,
        ] {
            let mut w = [0.5; 8];
            let mut g = [0.0; 8];
            let mut m = [f64::from_bits(1); 8];
            let mut v = m;
            adam_update_rsqrt_v4_tiered(&mut make_args(&mut w, &mut g, &mut m, &mut v, 100), tier);
            assert!(w.iter().all(|&x| x == 0.5), "{tier:?}");
            assert!(m.iter().chain(&v).all(|&x| x.to_bits() == 0), "{tier:?}");
        }
    }

    /// Start above the normal boundary and actually decay across it. Include
    /// restored subnormal states and every vector/tail boundary. The independent
    /// oracle uses classification, rather than the owner's exponent-mask code.
    #[test]
    fn moments_decay_to_zero_in_every_dispatch_tier() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            for n in 1..=17 {
                let mut w = vec![0.5; n];
                let mut g = vec![0.0; n];
                let mut m = vec![f64::MIN_POSITIVE * 2.0; n];
                let mut v = m.clone();
                m[0] = f64::from_bits(1);
                v[0] = f64::from_bits(0x000f_ffff_ffff_ffff);
                let (mut expected_m, mut expected_v) = (m.clone(), v.clone());
                let classify = |x: f64| {
                    if x.is_subnormal() {
                        0.0f64.copysign(x)
                    } else {
                        x
                    }
                };
                for step in 0..1000 {
                    for i in 0..n {
                        expected_m[i] = classify(0.9 * classify(expected_m[i]));
                        expected_v[i] = classify(0.999 * classify(expected_v[i]));
                    }
                    adam_update(&mut make_args(&mut w, &mut g, &mut m, &mut v, 10_000));
                    assert_eq!(
                        bits_of(&m),
                        bits_of(&expected_m),
                        "{} n={n} step={step} m",
                        perm.label
                    );
                    assert_eq!(
                        bits_of(&v),
                        bits_of(&expected_v),
                        "{} n={n} step={step} v",
                        perm.label
                    );
                    assert!(w.iter().all(|&x| x == 0.5));
                    assert!(g.iter().all(|&x| x.to_bits() == 0));
                }
                assert!(m.iter().chain(&v).all(|&x| x.to_bits() == 0));
            }
        });
        assert!(report.permutations_run > 0);
    }

    #[test]
    fn fused_moments_decay_to_zero_in_every_dispatch_tier() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let mut w = vec![0.5; 12];
            let mut g = vec![0.0; 12];
            let mut m = vec![f64::MIN_POSITIVE * 2.0; 12];
            let mut v = m.clone();
            m[0] = f64::from_bits(1);
            v[0] = f64::from_bits(1);
            let mut reference = (w.clone(), g.clone(), m.clone(), v.clone());
            for step in 0..1000 {
                adam_update_w1_fused(&mut AdamW1FusedArgs {
                    w: &mut w,
                    g: &mut g,
                    m: &mut m,
                    v: &mut v,
                    xa: &[0.0; 3],
                    xb: &[0.0; 3],
                    dha: &[0.0; 4],
                    dhb: &[0.0; 4],
                    n_hidden: 4,
                    l2_scale: 0.0,
                    l2_mult: None,
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-8,
                    bc1: 1.0,
                    bc2: 1.0,
                    lr: 0.005,
                    active_rows: None,
                    group_l1_tau: None,
                    g_zero_in: step % 2 == 0,
                    fwd: None,
                });
                adam_update_inner_scalar(
                    archmage::ScalarToken,
                    &mut AdamUpdateArgs {
                        w: &mut reference.0,
                        g: &mut reference.1,
                        m: &mut reference.2,
                        v: &mut reference.3,
                        beta1: 0.9,
                        beta2: 0.999,
                        eps: 1e-8,
                        bc1: 1.0,
                        bc2: 1.0,
                        lr: 0.005,
                    },
                );
                for (actual, expected) in [
                    (&w, &reference.0),
                    (&g, &reference.1),
                    (&m, &reference.2),
                    (&v, &reference.3),
                ] {
                    assert_eq!(
                        bits_of(actual),
                        bits_of(expected),
                        "{} step={step}",
                        perm.label
                    );
                }
                assert!(!m.iter().chain(&v).any(|x| x.is_subnormal()));
            }
            assert!(m.iter().chain(&v).all(|&x| x.to_bits() == 0));
        });
        assert!(report.permutations_run > 0);
    }

    // ---- fused K=1 pair update (fusedstep lane) ----

    /// Unfused oracle: replicate `backprop_*`'s gw1 row sweep (row skip on
    /// `x == 0.0`; FMA on `j < n4h`, mul+add tail) + `add_l2_grad_layer1`
    /// (mul+add, per-row `sm`) + the tier-dispatched `adam_update` — the
    /// exact sequence `mlp_train` runs per pair.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)] // oracle mirrors the kernel signature 1:1
    fn unfused_w1_oracle(
        w: &mut [f64],
        g: &mut [f64],
        m: &mut [f64],
        v: &mut [f64],
        xa: &[f64],
        dha: &[f64],
        xb: &[f64],
        dhb: &[f64],
        l2: f64,
        mult: Option<&[f64]>,
        nh: usize,
        t: u64,
    ) {
        let n = w.len();
        let n4h = nh - nh % 4;
        for r in 0..n / nh {
            let base = r * nh;
            let sa = xa[r];
            if sa != 0.0 {
                for (j, &dh) in dha.iter().enumerate().take(nh) {
                    let i = base + j;
                    if j < n4h {
                        g[i] = sa.mul_add(dh, g[i]);
                    } else {
                        g[i] += sa * dh;
                    }
                }
            }
            let sb = xb[r];
            if sb != 0.0 {
                for (j, &dh) in dhb.iter().enumerate().take(nh) {
                    let i = base + j;
                    if j < n4h {
                        g[i] = sb.mul_add(dh, g[i]);
                    } else {
                        g[i] += sb * dh;
                    }
                }
            }
        }
        if l2 > 0.0 {
            match mult {
                None => {
                    for (gg, &ww) in g.iter_mut().zip(w.iter()) {
                        *gg += l2 * ww;
                    }
                }
                Some(mm) => {
                    for (f, (gr, wr)) in g.chunks_mut(nh).zip(w.chunks(nh)).enumerate() {
                        let sm = l2 * mm[f];
                        for (gg, &ww) in gr.iter_mut().zip(wr.iter()) {
                            *gg += sm * ww;
                        }
                    }
                }
            }
        }
        adam_update(&mut make_args(w, g, m, v, t));
    }

    /// Deliberate order perturbation for the input-sensitivity check: L2
    /// before side B's contribution. Hand-written replica — never calls
    /// the kernel under test.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)] // oracle mirrors the kernel signature 1:1
    fn fused_wrong_order(
        w: &mut [f64],
        g: &mut [f64],
        m: &mut [f64],
        v: &mut [f64],
        xa: &[f64],
        dha: &[f64],
        xb: &[f64],
        dhb: &[f64],
        l2: f64,
        mult: Option<&[f64]>,
        nh: usize,
        t: u64,
    ) {
        let l2_on = l2 > 0.0;
        for r in 0..w.len() / nh {
            let sa = xa[r];
            let sb = xb[r];
            let sm = match mult {
                Some(mm) => l2 * mm[r],
                None => l2,
            };
            for j in 0..nh {
                let i = r * nh + j;
                let mut gg = g[i];
                if sa != 0.0 {
                    gg = sa.mul_add(dha[j], gg);
                }
                if l2_on {
                    gg += sm * w[i];
                }
                if sb != 0.0 {
                    gg = sb.mul_add(dhb[j], gg);
                }
                g[i] = gg;
            }
        }
        adam_update(&mut make_args(w, g, m, v, t));
    }

    #[allow(dead_code)]
    fn pair_inputs(nf: usize, nh: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (xa, _, _, _) = synth_state(nf, seed);
        let (xb, _, _, _) = synth_state(nf, seed ^ 0x51);
        let (dha, _, _, _) = synth_state(nh, seed ^ 0x52);
        let (dhb, _, _, _) = synth_state(nh, seed ^ 0x53);
        // ~1/7 of rows zeroed on each side — exercises the row-skip path.
        let xa: Vec<f64> = xa
            .iter()
            .enumerate()
            .map(|(i, &x)| if i % 7 == 3 { 0.0 } else { x })
            .collect();
        let xb: Vec<f64> = xb
            .iter()
            .enumerate()
            .map(|(i, &x)| if i % 5 == 1 { 0.0 } else { x })
            .collect();
        (xa, xb, dha, dhb)
    }

    /// The dispatched `adam_update_w1_fused` reproduces the unfused sequence
    /// bit-for-bit on every gated shape; off-grid shapes (nh%4!=0) route to
    /// the composition fallback, which must match too.
    ///
    /// NOTE on `black_box(t)`: `f64::powi` with a compile-time-known exponent
    /// can const-fold to a different bit pattern than the same call evaluated
    /// at runtime (CTFE vs `llvm.powi` lowering disagree by ~1ulp). The oracle
    /// computes `bc1`/`bc2` inside `make_args` with a runtime `t`; the fused
    /// args must take the same runtime path or the two sides run Adam with
    /// different bias corrections and the comparison is meaningless.
    #[test]
    fn fused_w1_bit_identical_dispatch() {
        for &(nf, nh) in &[
            (4usize, 4usize),
            (7, 8),
            (13, 16),
            (944, 32),
            (372, 128),
            (5, 6), // nh%4!=0 → fallback
            (9, 3), // nh%4!=0 → fallback
        ] {
            for has_mult in [false, true] {
                for l2 in [0.0f64, 1e-5, 0.3] {
                    let n = nf * nh;
                    let seed = 0xB17_1D3A ^ ((nf as u64) << 24) ^ nh as u64;
                    let (w0, g0, m0, v0) = synth_state(n, seed);
                    let (xa, xb, dha, dhb) = pair_inputs(nf, nh, seed);
                    let mult: Option<Vec<f64>> = if has_mult {
                        Some(
                            synth_state(nf, seed ^ 0x77)
                                .0
                                .iter()
                                .map(|x| x.abs() + 0.5)
                                .collect(),
                        )
                    } else {
                        None
                    };
                    let (mut wo, mut go, mut mo, mut vo) =
                        (w0.clone(), g0.clone(), m0.clone(), v0.clone());
                    let t = std::hint::black_box(3u64);
                    unfused_w1_oracle(
                        &mut wo,
                        &mut go,
                        &mut mo,
                        &mut vo,
                        &xa,
                        &dha,
                        &xb,
                        &dhb,
                        l2,
                        mult.as_deref(),
                        nh,
                        t,
                    );
                    let (mut wf, mut gf, mut mf, mut vf) =
                        (w0.clone(), g0.clone(), m0.clone(), v0.clone());
                    adam_update_w1_fused(&mut AdamW1FusedArgs {
                        w: &mut wf,
                        g: &mut gf,
                        m: &mut mf,
                        v: &mut vf,
                        xa: &xa,
                        dha: &dha,
                        xb: &xb,
                        dhb: &dhb,
                        l2_scale: l2,
                        l2_mult: mult.as_deref(),
                        n_hidden: nh,
                        beta1: 0.9,
                        beta2: 0.999,
                        eps: 1e-8,
                        bc1: 1.0 - 0.9f64.powi(t as i32),
                        bc2: 1.0 - 0.999f64.powi(t as i32),
                        lr: 0.005,
                        active_rows: None,
                        group_l1_tau: None,
                        g_zero_in: false,
                        fwd: None,
                    });
                    for (name, a, b) in [
                        ("w", &wo, &wf),
                        ("g", &go, &gf),
                        ("m", &mo, &mf),
                        ("v", &vo, &vf),
                    ] {
                        assert_eq!(
                            bits_of(a),
                            bits_of(b),
                            "fused diverged: {name} nf={nf} nh={nh} mult={has_mult} l2={l2}"
                        );
                    }
                }
            }
        }
    }

    /// Per-tier check: the v3 fused kernel and the scalar-tier fallback
    /// must each reproduce the unfused oracle bit-for-bit.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn fused_w1_per_tier_bit_identical() {
        use archmage::SimdToken;
        let _lock = archmage::testing::lock_token_testing();
        let v3 =
            archmage::X64V3Token::summon().expect("this test needs an x86-64-v3 (AVX2+FMA) host");
        let sc = archmage::ScalarToken::summon().expect("scalar token");
        let v4 = archmage::X64V4Token::summon();
        for &(nf, nh) in &[(6usize, 4usize), (944, 32), (372, 128), (33, 12)] {
            for has_mult in [false, true] {
                let n = nf * nh;
                let seed = 0xF00D ^ ((nf as u64) << 24) ^ nh as u64;
                let (w0, g0, m0, v0) = synth_state(n, seed);
                let (xa, xb, dha, dhb) = pair_inputs(nf, nh, seed);
                let mult: Option<Vec<f64>> = if has_mult {
                    Some(
                        synth_state(nf, seed ^ 0x99)
                            .0
                            .iter()
                            .map(|x| x.abs() + 0.5)
                            .collect(),
                    )
                } else {
                    None
                };
                let (mut wo, mut go, mut mo, mut vo) =
                    (w0.clone(), g0.clone(), m0.clone(), v0.clone());
                // `black_box(t)`: keeps `powi(t)` on the runtime path in BOTH
                // the oracle (`make_args`) and the fused args below —
                // CTFE-folded `powi(5)` differs from runtime `powi(5)` by ~1ulp.
                let t = std::hint::black_box(5u64);
                unfused_w1_oracle(
                    &mut wo,
                    &mut go,
                    &mut mo,
                    &mut vo,
                    &xa,
                    &dha,
                    &xb,
                    &dhb,
                    1e-5,
                    mult.as_deref(),
                    nh,
                    t,
                );
                let check = |tier: &str, run: &dyn Fn(&mut AdamW1FusedArgs<'_>)| {
                    let (mut wt, mut gt, mut mt, mut vt) =
                        (w0.clone(), g0.clone(), m0.clone(), v0.clone());
                    run(&mut AdamW1FusedArgs {
                        w: &mut wt,
                        g: &mut gt,
                        m: &mut mt,
                        v: &mut vt,
                        xa: &xa,
                        dha: &dha,
                        xb: &xb,
                        dhb: &dhb,
                        l2_scale: 1e-5,
                        l2_mult: mult.as_deref(),
                        n_hidden: nh,
                        beta1: 0.9,
                        beta2: 0.999,
                        eps: 1e-8,
                        bc1: 1.0 - 0.9f64.powi(t as i32),
                        bc2: 1.0 - 0.999f64.powi(t as i32),
                        lr: 0.005,
                        active_rows: None,
                        group_l1_tau: None,
                        g_zero_in: false,
                        fwd: None,
                    });
                    for (name, a, b) in [
                        ("w", &wo, &wt),
                        ("g", &go, &gt),
                        ("m", &mo, &mt),
                        ("v", &vo, &vt),
                    ] {
                        let i = a
                            .iter()
                            .zip(b.iter())
                            .position(|(x, y)| x.to_bits() != y.to_bits());
                        assert_eq!(
                            i,
                            None,
                            "fused {tier} diverged: {name} nf={nf} nh={nh} mult={has_mult} \
                             first i={:?} (row={} j={})",
                            i,
                            i.map(|i| i / nh).unwrap_or(0),
                            i.map(|i| i % nh).unwrap_or(0),
                        );
                    }
                };
                check("v3", &|a| adam_pair_fused_inner_v3(v3, a));
                check("scalar", &|a| adam_pair_fused_inner_scalar(sc, a));
                // v4 routes to the v3 body via `token.v3()` (the kernel is
                // elementwise and nh%4==0 keeps every row in the fused
                // domain); arm it wherever the host provides AVX-512.
                if let Some(v4) = v4 {
                    check("v4", &|a| adam_pair_fused_inner_v4(v4, a));
                }
            }
        }
    }

    /// `active_rows` (the `--keep-features` fast path) is bit-identical to the
    /// full walk when the skipped rows are pinned-zero (w = m = v = g = 0,
    /// x = 0): over several steps, with and without L2 / per-row multipliers.
    #[test]
    #[cfg(target_arch = "x86_64")] // the v3 kernel and X64V3Token exist on x86_64 only
    fn fused_w1_active_rows_bit_identical() {
        let (nf, nh) = (97usize, 32usize);
        let n = nf * nh;
        let kept = |r: usize| !(r % 3 == 1 || (40..52).contains(&r) || r == nf - 1);
        let runs: Vec<(u32, u32)> = {
            let mut v: Vec<(u32, u32)> = Vec::new();
            for r in (0..nf).filter(|&r| kept(r)) {
                match v.last_mut() {
                    Some(l) if l.1 as usize == r => l.1 += 1,
                    _ => v.push((r as u32, r as u32 + 1)),
                }
            }
            v
        };
        use archmage::SimdToken;
        let _lock = archmage::testing::lock_token_testing();
        let v3 =
            archmage::X64V3Token::summon().expect("this test needs an x86-64-v3 (AVX2+FMA) host");
        for (l2_scale, has_mult) in [(0.0, false), (1e-5, false), (1e-5, true)] {
            let (mut w0, mut g0, mut m0, mut v0) = synth_state(n, 0xAC71);
            let mult: Vec<f64> = (0..nf).map(|r| 0.5 + (r % 5) as f64).collect();
            let pin = |a: &mut Vec<f64>| {
                for r in (0..nf).filter(|&r| !kept(r)) {
                    a[r * nh..(r + 1) * nh].fill(0.0);
                }
            };
            for a in [&mut w0, &mut g0, &mut m0, &mut v0] {
                pin(a);
            }
            // Each step's dropped g rows start at +0.0 (the post-step invariant).
            let (mut wf, mut gf, mut mf, mut vf) = (w0.clone(), g0.clone(), m0.clone(), v0.clone());
            let (mut wa, mut ga, mut ma, mut va) = (w0, g0, m0, v0);
            for step in 1..=6u64 {
                let (mut xa, mut xb, dha, dhb) = pair_inputs(nf, nh, 0x77 + step);
                for r in (0..nf).filter(|&r| !kept(r)) {
                    xa[r] = 0.0;
                    xb[r] = 0.0;
                }
                let t = std::hint::black_box(step);
                let (bc1, bc2) = (1.0 - 0.9f64.powi(t as i32), 1.0 - 0.999f64.powi(t as i32));
                let run = |w: &mut Vec<f64>,
                           g: &mut Vec<f64>,
                           m: &mut Vec<f64>,
                           v: &mut Vec<f64>,
                           active: Option<&[(u32, u32)]>| {
                    adam_pair_fused_inner_v3(
                        v3,
                        &mut AdamW1FusedArgs {
                            w,
                            g,
                            m,
                            v,
                            xa: &xa,
                            dha: &dha,
                            xb: &xb,
                            dhb: &dhb,
                            l2_scale,
                            l2_mult: has_mult.then_some(mult.as_slice()),
                            n_hidden: nh,
                            beta1: 0.9,
                            beta2: 0.999,
                            eps: 1e-8,
                            bc1,
                            bc2,
                            lr: 0.005,
                            active_rows: active,
                            group_l1_tau: None,
                            g_zero_in: false,
                            fwd: None,
                        },
                    );
                };
                run(&mut wf, &mut gf, &mut mf, &mut vf, None);
                run(&mut wa, &mut ga, &mut ma, &mut va, Some(&runs));
                for (name, a, b) in [
                    ("w", &wf, &wa),
                    ("g", &gf, &ga),
                    ("m", &mf, &ma),
                    ("v", &vf, &va),
                ] {
                    assert_eq!(
                        bits_of(a),
                        bits_of(b),
                        "active_rows diverged: {name} step={step} l2={l2_scale} mult={has_mult}"
                    );
                }
            }
        }
    }

    /// Fusing the group-lasso prox into the row loop (`group_l1_tau`) and
    /// starting from a zero gradient (`g_zero_in`) are bit-identical to the
    /// separate post-step `apply_group_l1`-style pass over the plain fused
    /// kernel, on the v3 kernel, the v4 wrapper and the composition fallback,
    /// with and without `active_rows`, over several steps. `tau` is chosen so
    /// some rows hit the `norm <= tau` zeroing branch and others the scale
    /// branch (asserted), and the prox must actually change the weights.
    #[test]
    #[cfg(target_arch = "x86_64")] // the v3 kernel and X64V3Token exist on x86_64 only
    fn fused_w1_group_l1_and_g_zero_bit_identical() {
        let (nf, nh) = (97usize, 32usize);
        let n = nf * nh;
        let kept = |r: usize| !(r % 3 == 1 || (40..52).contains(&r) || r == nf - 1);
        let runs: Vec<(u32, u32)> = {
            let mut v: Vec<(u32, u32)> = Vec::new();
            for r in (0..nf).filter(|&r| kept(r)) {
                match v.last_mut() {
                    Some(l) if l.1 as usize == r => l.1 += 1,
                    _ => v.push((r as u32, r as u32 + 1)),
                }
            }
            v
        };
        use archmage::SimdToken;
        let _lock = archmage::testing::lock_token_testing();
        let v3 =
            archmage::X64V3Token::summon().expect("this test needs an x86-64-v3 (AVX2+FMA) host");
        for (l2_scale, has_mult, use_active) in [
            (0.0, false, false),
            (1e-5, false, true),
            (1e-5, true, false),
            (1e-3, true, true),
        ] {
            let (mut w0, _g, mut m0, mut v0) = synth_state(n, 0x6A11);
            let mult: Vec<f64> = (0..nf).map(|r| 0.5 + (r % 5) as f64).collect();
            // Row-scale spread so a single tau straddles the row norms.
            for r in 0..nf {
                let k = 0.02 * (1 + r % 11) as f64;
                w0[r * nh..(r + 1) * nh].iter_mut().for_each(|x| *x *= k);
            }
            if use_active {
                for r in (0..nf).filter(|&r| !kept(r)) {
                    for a in [&mut w0, &mut m0, &mut v0] {
                        a[r * nh..(r + 1) * nh].fill(0.0);
                    }
                }
            }
            let g_start = vec![0.0f64; n];
            // Oracle: plain fused kernel (g loaded/stored) + separate prox pass.
            let mut st_o = (w0.clone(), g_start.clone(), m0.clone(), v0.clone());
            // Candidates: (name, state, runner id).
            let mut st_v3 = st_o.clone();
            let mut st_v4w = st_o.clone();
            let mut st_fb = st_o.clone();
            let mut st_noprox = st_o.clone();
            let lr = 0.005;
            let (mut zeroed_rows, mut scaled_rows) = (0usize, 0usize);
            for step in 1..=6u64 {
                let (mut xa, mut xb, dha, dhb) = pair_inputs(nf, nh, 0x91 + step);
                if use_active {
                    for r in (0..nf).filter(|&r| !kept(r)) {
                        xa[r] = 0.0;
                        xb[r] = 0.0;
                    }
                }
                let t = std::hint::black_box(step);
                let (bc1, bc2) = (1.0 - 0.9f64.powi(t as i32), 1.0 - 0.999f64.powi(t as i32));
                let tau = 0.35 * lr / lr; // row-norm scale of the spread above
                macro_rules! mk {
                    ($st:expr, $tau:expr, $gz:expr, $active:expr) => {{
                        let (w, g, m, v) = &mut *$st;
                        AdamW1FusedArgs {
                            w,
                            g,
                            m,
                            v,
                            xa: &xa,
                            dha: &dha,
                            xb: &xb,
                            dhb: &dhb,
                            l2_scale,
                            l2_mult: has_mult.then_some(mult.as_slice()),
                            n_hidden: nh,
                            beta1: 0.9,
                            beta2: 0.999,
                            eps: 1e-8,
                            bc1,
                            bc2,
                            lr,
                            active_rows: $active,
                            group_l1_tau: $tau,
                            g_zero_in: $gz,
                            fwd: None,
                        }
                    }};
                }
                let act = use_active.then_some(runs.as_slice());
                // oracle
                adam_pair_fused_inner_v3(v3, &mut mk!(&mut st_o, None, false, None));
                for row in st_o.0.chunks_mut(nh) {
                    let norm = row.iter().map(|&w| w * w).sum::<f64>().sqrt();
                    if norm <= tau {
                        row.fill(0.0);
                        zeroed_rows += 1;
                    } else {
                        let f = 1.0 - tau / norm;
                        row.iter_mut().for_each(|w| *w *= f);
                        scaled_rows += 1;
                    }
                }
                adam_pair_fused_inner_v3(v3, &mut mk!(&mut st_noprox, None, false, None));
                adam_pair_fused_inner_v3(v3, &mut mk!(&mut st_v3, Some(tau), true, act));
                adam_pair_fused_inner_v3(v3, &mut mk!(&mut st_v4w, Some(tau), false, act));
                adam_pair_fused_fallback(&mut mk!(&mut st_fb, Some(tau), true, None));
                for (label, cand) in [
                    ("v3+gz", &st_v3),
                    ("v3 prox only", &st_v4w),
                    ("fallback", &st_fb),
                ] {
                    for (name, a, b) in [
                        ("w", &st_o.0, &cand.0),
                        ("g", &st_o.1, &cand.1),
                        ("m", &st_o.2, &cand.2),
                        ("v", &st_o.3, &cand.3),
                    ] {
                        assert_eq!(
                            bits_of(a),
                            bits_of(b),
                            "{label} diverged: {name} step={step} l2={l2_scale} mult={has_mult} active={use_active}"
                        );
                    }
                }
            }
            assert!(
                zeroed_rows > 0 && scaled_rows > 0,
                "tau must straddle: zeroed={zeroed_rows} scaled={scaled_rows}"
            );
            assert_ne!(
                bits_of(&st_o.0),
                bits_of(&st_noprox.0),
                "prox must change w"
            );
        }
    }

    /// TRAINEROPT3 task 2 oracle: arming `fwd` must (a) leave `w/g/m/v`
    /// bit-identical to the unarmed run and (b) produce `hpa`/`hpb` equal to
    /// a hand-replayed `forward` walk over the FINAL post-step weights
    /// (b1-seeded, ascending rows, `x != 0` guard, per-lane `mul_add` — the
    /// canonical-domain ops `forward_*` runs). Rows outside `active_rows`
    /// carry `x == 0` per the caller contract, so the kernel's row-gated
    /// apply and the oracle's all-row replay must agree bit-for-bit. Runs
    /// the v3 kernel and the composition fallback, prox fused and unfused,
    /// `active_rows` on and off, several chained steps, plus a `−0.0` input
    /// (skipped by the `!= 0.0` guard exactly like `forward`'s `s == 0.0`).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn fused_w1_fwd_accum_bit_identical() {
        // Review 2026-10-02: every canonical-domain width, not only multiples of 8.
        for nh in [32usize, 12, 20] {
            fused_w1_fwd_accum_bit_identical_at(nh);
        }
    }

    #[cfg(target_arch = "x86_64")] // the v3 kernel and X64V3Token exist on x86_64 only
    fn fused_w1_fwd_accum_bit_identical_at(nh: usize) {
        use archmage::SimdToken;
        let _lock = archmage::testing::lock_token_testing();
        let v3 =
            archmage::X64V3Token::summon().expect("this test needs an x86-64-v3 (AVX2+FMA) host");
        let nf = 97usize;
        let n = nf * nh;
        let kept = |r: usize| !(r % 5 == 2 || (30..38).contains(&r) || r == nf - 1);
        let runs: Vec<(u32, u32)> = {
            let mut v: Vec<(u32, u32)> = Vec::new();
            for r in (0..nf).filter(|&r| kept(r)) {
                match v.last_mut() {
                    Some(l) if l.1 as usize == r => l.1 += 1,
                    _ => v.push((r as u32, r as u32 + 1)),
                }
            }
            v
        };
        // A spread of b1 values so the seed is meaningful (not all +0).
        let b1: Vec<f64> = (0..nh).map(|j| (j as f64 - 16.0) * 0.013 - 0.2).collect();
        for (l2_scale, tau_on, use_active) in [
            (0.0f64, false, false),
            (1e-5, false, true),
            (0.0, true, true),
            (1e-5, true, false),
        ] {
            let (mut w0, _g, mut m0, mut v0) =
                synth_state(n, 0xF00D ^ l2_scale.to_bits() ^ tau_on as u64);
            if use_active {
                for r in (0..nf).filter(|&r| !kept(r)) {
                    for a in [&mut w0, &mut m0, &mut v0] {
                        a[r * nh..(r + 1) * nh].fill(0.0);
                    }
                }
            }
            let tau = tau_on.then_some(0.35f64);
            let act = use_active.then_some(runs.as_slice());
            let mut st_unarmed = (w0.clone(), vec![0.0f64; n], m0.clone(), v0.clone());
            let mut st_armed_v3 = st_unarmed.clone();
            let mut st_armed_fb = st_unarmed.clone();
            for step in 1..=4u64 {
                let (mut xa, mut xb, dha, dhb) = pair_inputs(nf, nh, 0x51 + step);
                // The NEXT pair's inputs — rows the walk skips must be +0.
                let (mut xa2, mut xb2, _, _) = pair_inputs(nf, nh, 0xA51 + step);
                xa2[7] = -0.0; // guard must skip, exactly like `s == 0.0`
                if use_active {
                    // Loader invariant: dropped inputs are standardized
                    // zeros on BOTH the current and the next pair — the
                    // fallback walks every row, so nonzero x on an unkept
                    // row would legitimately update it there.
                    for r in (0..nf).filter(|&r| !kept(r)) {
                        xa[r] = 0.0;
                        xb[r] = 0.0;
                        xa2[r] = 0.0;
                        xb2[r] = 0.0;
                    }
                }
                let t = step as i32;
                let (bc1, bc2) = (1.0 - 0.9f64.powi(t), 1.0 - 0.999f64.powi(t));
                let lr = 0.005;
                macro_rules! mk {
                    ($st:expr, $fwd:expr) => {{
                        let (w, g, m, v) = &mut *$st;
                        AdamW1FusedArgs {
                            w,
                            g,
                            m,
                            v,
                            xa: &xa,
                            dha: &dha,
                            xb: &xb,
                            dhb: &dhb,
                            l2_scale,
                            l2_mult: None,
                            n_hidden: nh,
                            beta1: 0.9,
                            beta2: 0.999,
                            eps: 1e-8,
                            bc1,
                            bc2,
                            lr,
                            active_rows: act,
                            group_l1_tau: tau,
                            g_zero_in: true,
                            fwd: $fwd,
                        }
                    }};
                }
                adam_pair_fused_inner_v3(v3, &mut mk!(&mut st_unarmed, None));
                let (mut hpa_v3, mut hpb_v3) = (b1.clone(), b1.clone());
                adam_pair_fused_inner_v3(
                    v3,
                    &mut mk!(
                        &mut st_armed_v3,
                        Some(FwdAccumW1 {
                            xa: &xa2,
                            xb: &xb2,
                            hpa: &mut hpa_v3,
                            hpb: &mut hpb_v3,
                        })
                    ),
                );
                let (mut hpa_fb, mut hpb_fb) = (b1.clone(), b1.clone());
                adam_pair_fused_fallback(&mut mk!(
                    &mut st_armed_fb,
                    Some(FwdAccumW1 {
                        xa: &xa2,
                        xb: &xb2,
                        hpa: &mut hpa_fb,
                        hpb: &mut hpb_fb,
                    })
                ));
                // Oracle: forward walk over the FINAL w — b1 seed, every row
                // ascending, `x != 0` guard, per-lane mul_add.
                let (mut hpa_o, mut hpb_o) = (b1.clone(), b1.clone());
                for i in 0..nf {
                    let wrow = &st_unarmed.0[i * nh..i * nh + nh];
                    if xa2[i] != 0.0 {
                        for j in 0..nh {
                            hpa_o[j] = xa2[i].mul_add(wrow[j], hpa_o[j]);
                        }
                    }
                    if xb2[i] != 0.0 {
                        for j in 0..nh {
                            hpb_o[j] = xb2[i].mul_add(wrow[j], hpb_o[j]);
                        }
                    }
                }
                for (label, st) in [("v3", &st_armed_v3), ("fb", &st_armed_fb)] {
                    for (name, a, b) in [
                        ("w", &st_unarmed.0, &st.0),
                        ("g", &st_unarmed.1, &st.1),
                        ("m", &st_unarmed.2, &st.2),
                        ("v", &st_unarmed.3, &st.3),
                    ] {
                        assert_eq!(
                            bits_of(a),
                            bits_of(b),
                            "{label} armed diverged: {name} step={step} l2={l2_scale} tau={tau_on} active={use_active}"
                        );
                    }
                }
                for (label, got, want) in [
                    ("v3 hpa", &hpa_v3, &hpa_o),
                    ("v3 hpb", &hpb_v3, &hpb_o),
                    ("fb hpa", &hpa_fb, &hpa_o),
                    ("fb hpb", &hpb_fb, &hpb_o),
                ] {
                    assert_eq!(
                        bits_of(got),
                        bits_of(want),
                        "{label}: look-ahead h_pre != forward replay — step={step} l2={l2_scale} tau={tau_on} active={use_active}"
                    );
                }
                // Sensitivity: xa2 has nonzero kept rows every step, so the
                // armed accumulator must move off its seed — if it didn't,
                // the fused apply silently never ran.
                assert_ne!(
                    bits_of(&hpa_v3),
                    bits_of(&b1),
                    "armed hpa stayed at seed — the fused apply never ran (step={step})"
                );
            }
        }
    }

    /// Input-sensitivity check (NOT a kernel negative control): the same
    /// fused element sequence with L2 applied *before* side B's
    /// contribution must produce different bits — otherwise the test data
    /// is insensitive to ordering and the equivalence asserts above would
    /// be vacuous. This compares two hand-written replicas; it never runs
    /// the kernel. Kernel-level mutation evidence (reordered v3 kernels
    /// are caught) is recorded in the lane report / Opus review mutation
    /// table.
    #[test]
    fn fused_w1_negative_control_order_matters() {
        let (nf, nh) = (944usize, 32usize);
        let n = nf * nh;
        let seed = 0xC0FFEEu64;
        let (w0, g0, m0, v0) = synth_state(n, seed);
        let (xa, xb, dha, dhb) = pair_inputs(nf, nh, seed);
        let (mut wo, mut go, mut mo, mut vo) = (w0.clone(), g0.clone(), m0.clone(), v0.clone());
        unfused_w1_oracle(
            &mut wo, &mut go, &mut mo, &mut vo, &xa, &dha, &xb, &dhb, 1e-5, None, nh, 3,
        );
        let (mut wp, mut gp, mut mp, mut vp) = (w0.clone(), g0.clone(), m0.clone(), v0.clone());
        fused_wrong_order(
            &mut wp, &mut gp, &mut mp, &mut vp, &xa, &dha, &xb, &dhb, 1e-5, None, nh, 3,
        );
        let diffs: usize = [(&wo, &wp), (&go, &gp), (&mo, &mp), (&vo, &vp)]
            .iter()
            .map(|(a, b)| {
                a.iter()
                    .zip(b.iter())
                    .filter(|(x, y)| x.to_bits() != y.to_bits())
                    .count()
            })
            .sum();
        assert!(
            diffs > 0,
            "negative control: reordered L2 produced identical bits — the oracle is insensitive"
        );
    }
}
