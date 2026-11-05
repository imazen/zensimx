//! **THE one owner of the SCORE path's arithmetic.**
//!
//! Five pieces of float math turn a bake's forward-pass output into a shipped
//! zensim score: the two mixing heads (per-sample-α and hybrid), the tanh
//! output pin, the PCHIP output-calibration spline, and the distance→score
//! mapping. Every one of them was implemented TWICE — once here (reached
//! through [`crate::metric`], the product runtime) and once in
//! `zensim-validate`, whose bake-evaluation tooling scores runtime-loaded bake
//! bytes over stored parquet rows and therefore cannot call the product
//! entry points, which consume a compile-time `ZensimProfile` and a
//! `ZensimResult`.
//!
//! # The defect this module exists for
//!
//! The duplication was documented as bit-exact **in prose, with no test
//! binding the two**, and F19 turned that prose into a live wrong-number bug:
//! [`crate::det_math::PowForm`] routes every transcendental on this path
//! through one selectable form so a score stops being a function of which
//! libm the binary linked against, and `metric.rs` follows it while the
//! mirror did not. Both defaults are [`PowForm::LibmPowf`] today, so the two
//! agree; the day [`crate::ssim_form::SHIPPED_REVISION`] flips they would
//! silently disagree, and every published verdict would describe arithmetic
//! the product does not run. `det_math`'s own exposure table named that fork
//! *"a BLOCKER on flipping `SHIPPED_REVISION`"*.
//!
//! MEASURED before the fix (2026-09-06, six shipped/board bakes ×
//! `cid22,kadid,tid,konjnd,aic3`): `bake_verdict --full-json` was
//! **byte-identical under `ZENSIM_POW_FORM=libm` and `=pure` on all six** —
//! i.e. the evaluation tooling was completely insensitive to the form the
//! product runtime obeys. That is the fork, observed rather than argued.
//!
//! A second divergence fell out of the same consolidation, REAL in code and
//! LATENT in practice: the validate-side PCHIP evaluator capped its upper
//! *extrapolation* at 100 (the 2026-07-04 fix) but left the *interior*
//! segment uncapped, while [`pchip_eval_capped`] — the product's — caps both.
//! The reachable trigger is **a knot whose `y` exceeds 100**, which the wire
//! format permits (`parse_payload` bounds `x` strictly increasing and both
//! finite, and bounds `y` not at all). It is NOT Hermite overshoot: the
//! Fritsch-Carlson derivative rule keeps the interpolant inside its
//! bracketing knots by construction, and a first draft of
//! `the_pchip_interior_is_capped_at_one_hundred` that tried to build an
//! overshoot fixture failed its own vacuity guard proving it.
//!
//! MEASURED over all 49 bakes on disk (`zensim/weights`, its `archive/`, and
//! `zensim-experimental/weights`): **0 declare a knot above 100**, so no
//! published verdict moved. The divergence was a loaded gun, not a fired one.
//! Unifying removes it by construction.
//!
//! # Shape
//!
//! The head parameters arrive in two different owned layouts — `metric.rs`
//! parses them into private structs, `zensim-validate` into public tuples —
//! so the entry points here take **borrowed** parameter views
//! ([`PerSampleAlphaParams`], [`HybridHeadParams`]). Both callers build one
//! for free; neither allocates, and neither has to adopt the other's storage.
//!
//! The [`PowForm`] is an explicit argument rather than a
//! [`crate::det_math::active_pow_form`] read inside, for two reasons: it is
//! the discipline `det_math` already documents (read the `OnceLock` ONCE,
//! above any loop — LLVM will not hoist it for you), and it is what lets a
//! test exercise BOTH arms in one process, which a process-wide `OnceLock`
//! forbids.
//!
//! # `#[doc(hidden)]`
//!
//! Internal machinery, not product surface. The supported API
//! (`docs/public-api/zensim.txt`) is unchanged; these items are registered in
//! `docs/public-api/zensim.internal.txt`.

use crate::det_math::{DetPow, PowForm};

/// Pool-statistic sigma floor shared by both mixing heads.
///
/// Mirrors `zensim-train-core::pool_head::POOL_STD_FLOOR`. It is inlined
/// rather than imported so the zensim runtime keeps no dependency on the
/// trainer crate; `metric.rs` previously carried it TWICE (once per head) and
/// `zensim-validate` a third time.
pub const POOL_STD_FLOOR: f64 = 0.0026;

/// Borrowed per-sample-α head parameters.
///
/// `y = α · y_rank + (1 − α) · y_pool` with `α = σ(h · w_alpha + b_alpha)`
/// computed PER SAMPLE from the hidden vector. Payload layout and provenance:
/// `zensim-train-core::per_sample_alpha_head`.
#[derive(Debug, Clone, Copy)]
pub struct PerSampleAlphaParams<'a> {
    /// Gate weights, one per hidden unit.
    pub w_alpha: &'a [f32],
    /// Gate bias.
    pub b_alpha: f32,
    /// Rank-head weights, one per hidden unit.
    pub rank_w: &'a [f32],
    /// Rank-head bias.
    pub rank_b: f32,
    /// Pool-head reducer weights over `[μ, σ, max, p-norm]`.
    pub reducer_w: [f32; 4],
    /// Pool-head reducer bias.
    pub reducer_b: f32,
    /// The pool head's p-norm exponent.
    pub p_norm: f32,
}

/// Borrowed hybrid-head parameters.
///
/// Same mix as [`PerSampleAlphaParams`] but `α = σ(alpha_logit)` is a single
/// LEARNED SCALAR rather than a per-sample gate. Provenance:
/// `zensim-train-core::hybrid_head`.
#[derive(Debug, Clone, Copy)]
pub struct HybridHeadParams<'a> {
    /// Rank-head weights, one per hidden unit.
    pub rank_w: &'a [f32],
    /// Rank-head bias.
    pub rank_b: f32,
    /// The scalar gate logit.
    pub alpha_logit: f32,
    /// Pool-head reducer weights over `[μ, σ, max, p-norm]`.
    pub reducer_w: [f32; 4],
    /// Pool-head reducer bias.
    pub reducer_b: f32,
    /// The pool head's p-norm exponent.
    pub p_norm: f32,
}

/// Apply the per-sample-α head to a hidden vector `h`.
///
/// Returns `f64::NAN` when `h` is empty or either weight vector disagrees
/// with its length. **That is a widening, and it is deliberate**: the
/// validate-side mirror already returned NaN on mismatch, while `metric.rs`
/// carried `debug_assert_eq!` plus an unchecked index — so in release a
/// `w_alpha` length mismatch was an out-of-bounds panic there. Both shapes
/// are unreachable through either caller (each parse guarantees both vectors
/// are `n_hidden` long, and `metric.rs` re-checks `rank_w` before calling),
/// so no reachable path changes; the checked form is simply the one that
/// cannot be worse.
///
/// Read the [`PowForm`] ONCE above your own loop and pass it in — see the
/// module docs.
pub fn per_sample_alpha_head(h: &[f32], p: &PerSampleAlphaParams<'_>, form: PowForm) -> f64 {
    let n = h.len();
    if n == 0 || p.rank_w.len() != n || p.w_alpha.len() != n {
        return f64::NAN;
    }

    let mut y_rank = p.rank_b as f64;
    let mut alpha_logit = p.b_alpha as f64;
    let mut sum = 0.0_f64;
    let mut max_v = f64::NEG_INFINITY;
    let mut sum_p = 0.0_f64;
    let pn = p.p_norm as f64;
    for (j, &hj) in h.iter().enumerate() {
        let hjf = hj as f64;
        y_rank += hjf * p.rank_w[j] as f64;
        alpha_logit += hjf * p.w_alpha[j] as f64;
        sum += hjf;
        if hjf > max_v {
            max_v = hjf;
        }
        sum_p += hjf.abs().det_powf(pn, form);
    }
    let nf = n as f64;
    let mu = sum / nf;
    let mut var = 0.0_f64;
    for &hj in h.iter() {
        let d = hj as f64 - mu;
        var += d * d;
    }
    let sigma = (var / nf).sqrt().max(POOL_STD_FLOOR);
    let p_norm_stat = (sum_p / nf).det_powf(1.0 / pn, form);

    let y_pool = mu * p.reducer_w[0] as f64
        + sigma * p.reducer_w[1] as f64
        + max_v * p.reducer_w[2] as f64
        + p_norm_stat * p.reducer_w[3] as f64
        + p.reducer_b as f64;

    // sigmoid with clamp (matches the trainer's `sigmoid` helper).
    let alpha = {
        let xc = alpha_logit.clamp(-20.0, 20.0);
        1.0 / (1.0 + (-xc).det_exp(form))
    };
    alpha * y_rank + (1.0 - alpha) * y_pool
}

/// Apply the hybrid head to a hidden vector `h`.
///
/// Returns `f64::NAN` when `h` is empty or `rank_w` disagrees with its
/// length — see [`per_sample_alpha_head`] for why the check is here.
pub fn hybrid_head(h: &[f32], p: &HybridHeadParams<'_>, form: PowForm) -> f64 {
    let n = h.len();
    if n == 0 || p.rank_w.len() != n {
        return f64::NAN;
    }

    let mut y_rank = p.rank_b as f64;
    let mut sum = 0.0_f64;
    let mut max_v = f64::NEG_INFINITY;
    let mut sum_p = 0.0_f64;
    let pn = p.p_norm as f64;
    for (j, &hj) in h.iter().enumerate() {
        let hjf = hj as f64;
        y_rank += hjf * p.rank_w[j] as f64;
        sum += hjf;
        if hjf > max_v {
            max_v = hjf;
        }
        sum_p += hjf.abs().det_powf(pn, form);
    }
    let nf = n as f64;
    let mu = sum / nf;
    let mut var = 0.0_f64;
    for &hj in h.iter() {
        let d = hj as f64 - mu;
        var += d * d;
    }
    let sigma = (var / nf).sqrt().max(POOL_STD_FLOOR);
    let p_norm_stat = (sum_p / nf).det_powf(1.0 / pn, form);

    let y_pool = mu * p.reducer_w[0] as f64
        + sigma * p.reducer_w[1] as f64
        + max_v * p.reducer_w[2] as f64
        + p_norm_stat * p.reducer_w[3] as f64
        + p.reducer_b as f64;

    // sigmoid with clamp (matches the trainer's `sigmoid` helper).
    let alpha = {
        let xc = (p.alpha_logit as f64).clamp(-20.0, 20.0);
        1.0 / (1.0 + (-xc).det_exp(form))
    };
    alpha * y_rank + (1.0 - alpha) * y_pool
}

/// The tanh-pinned `[0, 100]` sigmoid wrap: `100 · σ(y_pre / scale)`.
///
/// Matches `zensim-train-core::per_sample_alpha_head::
/// bake_per_sample_alpha_head_v3_with_tanh` at training time.
///
/// NaN in, NaN out — by arithmetic, not by a guard: `f64::clamp` propagates
/// NaN, `exp(NaN)` is NaN, and `100 · NaN` is NaN. A caller that wants to
/// short-circuit on NaN may still do so; that is a shape choice, not a
/// numerical one.
pub fn tanh_output_pin(y_pre: f64, scale: f64, form: PowForm) -> f64 {
    let xc = (y_pre / scale).clamp(-30.0, 30.0);
    let s = 1.0 / (1.0 + (-xc).det_exp(form));
    100.0 * s
}

/// Map a raw weighted distance to the quality score: `100 − a · d^b`.
///
/// Nominally 0–100 but UNCLAMPED below zero — the magnitude below zero is
/// informative (it separates "slightly wrong" from "completely wrong"), so
/// clamping is the caller's policy, not this function's. `d ≤ 0` returns
/// exactly `100.0`.
pub fn distance_to_score_mapped(raw_distance: f64, a: f64, b: f64, form: PowForm) -> f64 {
    if raw_distance <= 0.0 {
        100.0
    } else {
        100.0 - a * raw_distance.det_powf(b, form)
    }
}

/// Fritsch–Carlson monotone-preserving derivatives at each knot.
///
/// Standard PCHIP recipe: a weighted harmonic mean of adjacent slopes where
/// they share a sign (zero at an extremum), with three-point endpoint
/// estimates clamped to preserve monotonicity.
///
/// # Panics
///
/// Panics if `xs.len() < 2` or `ys.len() != xs.len()` — both are parse-time
/// invariants of every caller (`n_knots >= 2`, strictly increasing `x`).
///
/// Contains no transcendental, so it does not take a [`PowForm`]: it is
/// `+ − × ÷` only, which IEEE-754 requires to be correctly rounded, and is
/// therefore already libc-independent (`det_math`'s exposure table records
/// the same audit for the spline).
pub fn pchip_derivs(xs: &[f64], ys: &[f64]) -> Vec<f64> {
    let n = xs.len();
    assert_eq!(ys.len(), n, "pchip_derivs: xs/ys length mismatch");
    assert!(n >= 2, "pchip_derivs: needs at least 2 knots");
    if n == 2 {
        let s = (ys[1] - ys[0]) / (xs[1] - xs[0]);
        return vec![s, s];
    }
    // Per-segment slopes h_k = (y_{k+1} - y_k) / (x_{k+1} - x_k).
    let mut h = Vec::with_capacity(n - 1);
    let mut s = Vec::with_capacity(n - 1);
    for k in 0..n - 1 {
        let hk = xs[k + 1] - xs[k];
        h.push(hk);
        s.push((ys[k + 1] - ys[k]) / hk);
    }
    let mut d = vec![0.0_f64; n];
    // Interior: weighted harmonic mean when adjacent slopes share sign,
    // else 0 (extremum).
    for k in 1..n - 1 {
        if s[k - 1] * s[k] <= 0.0 {
            d[k] = 0.0;
        } else {
            let w1 = 2.0 * h[k] + h[k - 1];
            let w2 = h[k] + 2.0 * h[k - 1];
            d[k] = (w1 + w2) / (w1 / s[k - 1] + w2 / s[k]);
        }
    }
    // Endpoints — three-point estimate, clamped to preserve mono.
    d[0] = pchip_endpoint(h[0], h[1], s[0], s[1]);
    d[n - 1] = pchip_endpoint(h[n - 2], h[n - 3], s[n - 2], s[n - 3]);
    d
}

fn pchip_endpoint(h0: f64, h1: f64, s0: f64, s1: f64) -> f64 {
    let d = ((2.0 * h0 + h1) * s0 - h0 * s1) / (h0 + h1);
    if d * s0 <= 0.0 {
        0.0
    } else if s0 * s1 <= 0.0 && d.abs() > 3.0 * s0.abs() {
        3.0 * s0
    } else {
        d
    }
}

/// Evaluate the output-calibration PCHIP spline at `x`, capped at `≤ 100` on
/// the upper side and FLOORED one calibrated-dial-range below the bottom knot.
///
/// A perceptual score can never exceed 100 (nothing is more similar than
/// identical), so **every** branch — lower extrapolation, interior segment,
/// upper extrapolation — is capped. The lower side is allowed to go NEGATIVE
/// (an input more dissimilar than the worst codec output is meaningful signal,
/// not a tie at zero) but is floored at `ys[0] − (ys[n−1] − ys[0])` as an OOD
/// safety net for a pathological raw that slipped past the winsor guard.
/// Monotone: `min`/`max` with a constant preserves rank.
///
/// **The interior cap was a divergence — real in code, latent in practice.**
/// The product runtime has always capped the interior; the validate-side
/// mirror capped only the upper extrapolation (the 2026-07-04 fix stopped one
/// branch short), so `bake_verdict` could report a score the shipped runtime
/// reports as exactly 100. The reachable trigger is a KNOT above 100 — which
/// the wire format permits and nothing rejects — not Hermite overshoot, which
/// the Fritsch-Carlson rule forbids. Measured: 0 of the 49 bakes on disk
/// declare one, so no published number moved.
///
/// **A known asymmetry, deliberately unchanged.** The lower branch's
/// `floor = ys[0] − (ys[n−1] − ys[0])` is a FLOOR only for an increasing
/// spline. On a DECREASING one it lands above `ys[0]` and the `.max` makes it
/// a hard value: seven `zensim-experimental` bakes return exactly 200.0 at
/// `x == xs[0]`. That behaviour is IDENTICAL in both implementations, so it
/// is not an owner divergence and changing it would move product numbers —
/// recorded here, not fixed here. No shipped profile has a decreasing spline.
///
/// # Panics
///
/// Panics if the three arrays disagree in length or hold fewer than 2 knots —
/// parse-time invariants of every caller.
pub fn pchip_eval_capped(x: f64, xs: &[f64], ys: &[f64], derivs: &[f64]) -> f64 {
    let n = xs.len();
    assert!(n >= 2, "pchip_eval_capped: needs at least 2 knots");
    assert_eq!(ys.len(), n, "pchip_eval_capped: xs/ys length mismatch");
    assert_eq!(
        derivs.len(),
        n,
        "pchip_eval_capped: xs/derivs length mismatch"
    );
    if !x.is_finite() {
        return x;
    }
    if x <= xs[0] {
        let floor = ys[0] - (ys[n - 1] - ys[0]);
        return (ys[0] + derivs[0] * (x - xs[0])).max(floor);
    }
    if x >= xs[n - 1] {
        return (ys[n - 1] + derivs[n - 1] * (x - xs[n - 1])).min(100.0);
    }
    // Find the segment via binary search.
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let h = xs[hi] - xs[lo];
    let t = (x - xs[lo]) / h;
    // Cubic Hermite basis on [0, 1].
    let h00 = (1.0 + 2.0 * t) * (1.0 - t).powi(2);
    let h10 = t * (1.0 - t).powi(2);
    let h01 = t.powi(2) * (3.0 - 2.0 * t);
    let h11 = t.powi(2) * (t - 1.0);
    (h00 * ys[lo] + h10 * h * derivs[lo] + h01 * ys[hi] + h11 * h * derivs[hi]).min(100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic stand-in for a bake's hidden vector: `n` values
    /// spanning the sign and magnitude range a post-LeakyReLU layer produces.
    fn hidden(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                // (-4, 4), with the negative tail LeakyReLU actually emits.
                let u = (s >> 11) as f64 / ((1u64 << 53) as f64);
                ((u * 8.0 - 4.0) as f32) * if s & 0x10 == 0 { 1.0 } else { 0.1 }
            })
            .collect()
    }

    fn psa_params<'a>(w: &'a [f32], r: &'a [f32]) -> PerSampleAlphaParams<'a> {
        PerSampleAlphaParams {
            w_alpha: w,
            b_alpha: -0.25,
            rank_w: r,
            rank_b: 3.5,
            reducer_w: [0.7, -1.3, 0.4, 0.9],
            reducer_b: -2.0,
            p_norm: 6.0,
        }
    }

    /// The heads are NOT form-invariant — the property whose absence made the
    /// validate-side fork invisible. If a future edit routes them back through
    /// `f64::powf`, both arms would agree and this fails.
    #[test]
    fn both_heads_respond_to_the_pow_form() {
        let mut psa_moved = 0usize;
        let mut hyb_moved = 0usize;
        for seed in 0..2_000u64 {
            let n = 8 + (seed as usize % 121);
            let h = hidden(n, seed);
            let w = hidden(n, seed ^ 0xAAAA);
            let r = hidden(n, seed ^ 0x5555);
            let p = psa_params(&w, &r);
            let a = per_sample_alpha_head(&h, &p, PowForm::LibmPowf);
            let b = per_sample_alpha_head(&h, &p, PowForm::PureRust);
            if a.to_bits() != b.to_bits() {
                psa_moved += 1;
            }
            let hp = HybridHeadParams {
                rank_w: &r,
                rank_b: 3.5,
                alpha_logit: 0.75,
                reducer_w: [0.7, -1.3, 0.4, 0.9],
                reducer_b: -2.0,
                p_norm: 6.0,
            };
            let c = hybrid_head(&h, &hp, PowForm::LibmPowf);
            let d = hybrid_head(&h, &hp, PowForm::PureRust);
            if c.to_bits() != d.to_bits() {
                hyb_moved += 1;
            }
        }
        assert!(
            psa_moved > 0,
            "per_sample_alpha_head is form-INVARIANT — it is not routed through det_math"
        );
        assert!(
            hyb_moved > 0,
            "hybrid_head is form-INVARIANT — it is not routed through det_math"
        );
    }

    /// The two arms are a rounding question, never a semantic one.
    #[test]
    fn the_two_pow_arms_stay_close_on_the_heads() {
        for seed in 0..2_000u64 {
            let n = 8 + (seed as usize % 121);
            let h = hidden(n, seed);
            let w = hidden(n, seed ^ 0xAAAA);
            let r = hidden(n, seed ^ 0x5555);
            let p = psa_params(&w, &r);
            let a = per_sample_alpha_head(&h, &p, PowForm::LibmPowf);
            let b = per_sample_alpha_head(&h, &p, PowForm::PureRust);
            assert!(a.is_finite() && b.is_finite(), "seed {seed}: non-finite");
            let rel = (a - b).abs() / a.abs().max(b.abs()).max(1.0);
            assert!(rel < 1e-12, "seed {seed}: {a} vs {b} (rel {rel:e})");
        }
    }

    #[test]
    fn heads_return_nan_on_a_length_mismatch() {
        let h = vec![0.5_f32; 4];
        let w = vec![1.0_f32; 3];
        let p = psa_params(&w, &w);
        assert!(per_sample_alpha_head(&h, &p, PowForm::LibmPowf).is_nan());
        let hp = HybridHeadParams {
            rank_w: &w,
            rank_b: 0.0,
            alpha_logit: 0.0,
            reducer_w: [1.0; 4],
            reducer_b: 0.0,
            p_norm: 2.0,
        };
        assert!(hybrid_head(&h, &hp, PowForm::LibmPowf).is_nan());
        let empty: [f32; 0] = [];
        let ep = psa_params(&empty, &empty);
        assert!(per_sample_alpha_head(&empty, &ep, PowForm::LibmPowf).is_nan());
    }

    #[test]
    fn tanh_pin_is_zero_to_hundred_and_propagates_nan() {
        for form in [PowForm::LibmPowf, PowForm::PureRust] {
            assert!((tanh_output_pin(0.0, 1.0, form) - 50.0).abs() < 1e-12);
            assert!(tanh_output_pin(f64::NAN, 1.0, form).is_nan());
            for i in -1000..=1000 {
                let y = tanh_output_pin(i as f64 * 0.5, 3.0, form);
                assert!((0.0..=100.0).contains(&y), "out of band at {i}: {y}");
            }
        }
    }

    #[test]
    fn distance_mapping_is_exactly_a_hundred_at_or_below_zero() {
        for form in [PowForm::LibmPowf, PowForm::PureRust] {
            assert_eq!(distance_to_score_mapped(0.0, 18.0, 0.7, form), 100.0);
            assert_eq!(distance_to_score_mapped(-1.0, 18.0, 0.7, form), 100.0);
            assert_eq!(distance_to_score_mapped(-0.0, 18.0, 0.7, form), 100.0);
            // Unclamped below zero on purpose.
            assert!(distance_to_score_mapped(1e6, 18.0, 0.7, form) < 0.0);
        }
    }

    /// The interior cap that the validate-side mirror was missing.
    ///
    /// The reachable trigger is a KNOT above 100 — the wire format permits one
    /// and nothing rejects it. An earlier draft of this test tried to build a
    /// Hermite OVERSHOOT instead and failed its own vacuity guard (max 99.5 on
    /// knots 0/99/99.5), which is the Fritsch-Carlson monotonicity guarantee
    /// showing up as a test failure: the interpolant cannot leave its
    /// bracketing knots. The guard is kept below for the same reason.
    #[test]
    fn the_pchip_interior_is_capped_at_one_hundred() {
        let xs = [0.0_f64, 50.0, 100.0];
        let ys = [0.0_f64, 80.0, 130.0];
        let d = pchip_derivs(&xs, &ys);
        // The old validate-side interior branch, verbatim and uncapped.
        let uncapped = |x: f64, lo: usize, hi: usize| {
            let h = xs[hi] - xs[lo];
            let t = (x - xs[lo]) / h;
            let h00 = (1.0 + 2.0 * t) * (1.0 - t).powi(2);
            let h10 = t * (1.0 - t).powi(2);
            let h01 = t.powi(2) * (3.0 - 2.0 * t);
            let h11 = t.powi(2) * (t - 1.0);
            h00 * ys[lo] + h10 * h * d[lo] + h01 * ys[hi] + h11 * h * d[hi]
        };
        let worst = (1..1000)
            .map(|i| uncapped(50.0 + i as f64 * 0.05, 1, 2))
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            worst > 100.0,
            "fixture no longer exceeds 100 in the interior (max {worst}); \
             the cap test would be vacuous"
        );
        for i in 1..1000 {
            let x = 50.0 + i as f64 * 0.05;
            let y = pchip_eval_capped(x, &xs, &ys, &d);
            assert!(y <= 100.0, "interior above 100 at x={x}: {y}");
        }
    }

    /// The Fritsch-Carlson rule keeps the interpolant inside its bracketing
    /// knots — pinned, because the interior-cap test's shape depends on it and
    /// a future change to `pchip_derivs` that broke it would otherwise be
    /// invisible.
    #[test]
    fn pchip_never_leaves_its_bracketing_knots_on_monotone_data() {
        let xs = [0.0_f64, 10.0, 10.5, 40.0, 41.0, 90.0];
        let ys = [0.0_f64, 99.0, 99.2, 99.4, 99.5, 99.9];
        let d = pchip_derivs(&xs, &ys);
        for seg in 0..xs.len() - 1 {
            for i in 0..=200 {
                let x = xs[seg] + (xs[seg + 1] - xs[seg]) * (i as f64 / 200.0);
                let y = pchip_eval_capped(x, &xs, &ys, &d);
                assert!(
                    y >= ys[seg] - 1e-9 && y <= ys[seg + 1] + 1e-9,
                    "segment {seg} left [{}, {}] at x={x}: {y}",
                    ys[seg],
                    ys[seg + 1]
                );
            }
        }
    }

    #[test]
    fn the_pchip_lower_tail_stays_negative_but_floored() {
        let xs = [0.0_f64, 50.0, 100.0];
        let ys = [10.0_f64, 55.0, 100.0];
        let d = pchip_derivs(&xs, &ys);
        let floor = ys[0] - (ys[2] - ys[0]);
        let deep = pchip_eval_capped(-1.0e6, &xs, &ys, &d);
        assert_eq!(deep, floor, "lower tail is not floored at {floor}");
        // A modest OOD input still goes negative — the neg-tail is intact.
        assert!(pchip_eval_capped(-20.0, &xs, &ys, &d) < 0.0);
    }

    #[test]
    fn pchip_passes_through_its_knots() {
        let xs = [4.5_f64, 7.2, 15.3, 28.1, 38.7, 55.0, 68.4, 88.9];
        let ys = [0.0_f64, 10.0, 30.0, 50.0, 60.0, 80.0, 90.0, 100.0];
        let d = pchip_derivs(&xs, &ys);
        for (x, y) in xs.iter().zip(ys.iter()) {
            let got = pchip_eval_capped(*x, &xs, &ys, &d);
            assert!((got - *y).abs() < 1e-9, "knot {x}: {got} != {y}");
        }
    }
}
