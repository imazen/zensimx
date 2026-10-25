// ============================================================================
// THE SCALAR ORACLE (era-2, 2026-08-31) — `benchmarks/era2_perf_break_2026-08-31.md` §11
//
// TEST-ONLY. Never in the product build. Two levels, because "correct" must
// not itself be an approximation:
//
//   L1 `Neumaier` — a readable scalar reference in unambiguous left-to-right
//      order with Kahan-Babuska-NEUMAIER compensation (chosen over plain
//      Kahan, which loses the correction when the running total is smaller in
//      magnitude than the addend — a real case here, since these sums start
//      at 0 and the first terms can dominate).
//   L2 `Exact`   — Shewchuk non-overlapping expansion via TwoSum. The
//      expansion holds the EXACT sum of the f64 terms; `value()` rounds it
//      once by summing components smallest-first.
//
// SCOPE, stated precisely: the oracle is exact with respect to SUMMATION, and
// the per-pixel formulas are evaluated in f64. That is the right reference,
// because what era-2 changes IS the summation order and the precision of the
// accumulation — not the formulas. Where a production tier evaluates the
// per-pixel terms in f32 lanes (the POOL_SIMD pool path), its deviation from
// this oracle legitimately includes that term-precision difference, and §10.4
// of the design says so.
//
// STANDING ROLE: this is the regression instrument for ALL future perf work on
// these kernels, not an era-2 one-off. Measure a new kernel against the oracle
// BEFORE measuring it against a bench.
#[cfg(any(test, feature = "oracle"))]
// The oracle is a RULER, not semantics: its consumers are `cfg(test)` gates and
// the `oracle`-feature harnesses. Building the lib with `oracle` on but without
// `cfg(test)` therefore leaves most of it unreferenced, which is correct rather
// than a smell.
#[allow(dead_code)]
pub(crate) mod oracle {
    use super::*;

    /// Number of accumulator slots: 13 plain sums + 11 `WeightedSum` × (num, den).
    pub(crate) const N_SLOTS: usize = 13 + 11 * 2;

    /// Slot names, index-aligned with [`Slots`], grouped by FAMILY so the gate
    /// can report per-family worst cases (design §11.2).
    pub(crate) const SLOT_NAMES: [&str; N_SLOTS] = [
        "sum_d",
        "sum_d2",
        "sum_d3",
        "sum_d4",
        "sum_art",
        "sum_det",
        "sum_mse",
        "sum_hf_gain",
        "sum_hf_loss",
        "sum_hf_mag_loss",
        "sum_pjnd",
        "sum_pjnd_lo",
        "sum_pjnd_hi",
        "ws_peak_ssim.num",
        "ws_peak_ssim.den",
        "ws_peak_art.num",
        "ws_peak_art.den",
        "ws_peak_det.num",
        "ws_peak_det.den",
        "ws_mask_ssim.num",
        "ws_mask_ssim.den",
        "ws_mask_art.num",
        "ws_mask_art.den",
        "ws_mask_det.num",
        "ws_mask_det.den",
        "ws_mask_mse.num",
        "ws_mask_mse.den",
        "ws_iw_ssim.num",
        "ws_iw_ssim.den",
        "ws_iw_art.num",
        "ws_iw_art.den",
        "ws_iw_det.num",
        "ws_iw_det.den",
        "ws_iw_mse.num",
        "ws_iw_mse.den",
    ];

    /// The family each slot belongs to, for per-family bound reporting.
    pub(crate) fn slot_family(i: usize) -> &'static str {
        match i {
            0..=6 => "core",
            7..=9 => "hf",
            10..=12 => "pjnd",
            _ => "pools",
        }
    }

    /// A compensated or exact accumulator.
    pub(crate) trait Acc: Default + Clone {
        fn add(&mut self, x: f64);
        fn value(&self) -> f64;
    }

    /// L1: Kahan-Babuska-Neumaier compensated summation.
    #[derive(Clone, Default, Debug)]
    pub(crate) struct Neumaier {
        sum: f64,
        c: f64,
    }
    impl Acc for Neumaier {
        #[inline]
        fn add(&mut self, x: f64) {
            let t = self.sum + x;
            // The Neumaier branch: pick the operand that is larger in
            // magnitude as the one whose bits survive, so the correction is
            // captured in BOTH orderings (plain Kahan drops it in one).
            if self.sum.abs() >= x.abs() {
                self.c += (self.sum - t) + x;
            } else {
                self.c += (x - t) + self.sum;
            }
            self.sum = t;
        }
        #[inline]
        fn value(&self) -> f64 {
            self.sum + self.c
        }
    }

    /// `a + b` exactly, as (rounded sum, exact error). Knuth's TwoSum — no
    /// assumption about the relative magnitudes of `a` and `b`.
    #[inline]
    fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let bb = s - a;
        let err = (a - (s - bb)) + (b - bb);
        (s, err)
    }

    /// L2: Shewchuk non-overlapping expansion. The component list holds the
    /// EXACT sum; nothing is discarded.
    #[derive(Clone, Default, Debug)]
    pub(crate) struct Exact {
        parts: Vec<f64>,
    }
    impl Acc for Exact {
        fn add(&mut self, x: f64) {
            if x == 0.0 {
                return;
            }
            let mut carry = x;
            let mut keep = 0usize;
            for j in 0..self.parts.len() {
                let (hi, lo) = two_sum(carry, self.parts[j]);
                if lo != 0.0 {
                    self.parts[keep] = lo;
                    keep += 1;
                }
                carry = hi;
            }
            self.parts.truncate(keep);
            if carry != 0.0 {
                self.parts.push(carry);
            }
        }
        fn value(&self) -> f64 {
            // Components are non-overlapping and increasing in magnitude;
            // summing smallest-first rounds the exact total once.
            let mut s = 0.0;
            for &p in &self.parts {
                s += p;
            }
            s
        }
    }

    /// One accumulator per slot.
    #[derive(Clone)]
    pub(crate) struct Slots<A: Acc> {
        pub acc: Vec<A>,
    }
    impl<A: Acc> Default for Slots<A> {
        fn default() -> Self {
            Self {
                acc: vec![A::default(); N_SLOTS],
            }
        }
    }
    impl<A: Acc> Slots<A> {
        #[inline]
        fn add(&mut self, i: usize, x: f64) {
            self.acc[i].add(x);
        }
        pub(crate) fn values(&self) -> [f64; N_SLOTS] {
            let mut out = [0.0f64; N_SLOTS];
            for (o, a) in out.iter_mut().zip(self.acc.iter()) {
                *o = a.value();
            }
            out
        }
    }

    /// Flatten a production [`DenseAccum`] into the same slot order.
    pub(super) fn dense_accum_slots(a: &DenseAccum) -> [f64; N_SLOTS] {
        [
            a.sum_d,
            a.sum_d2,
            a.sum_d3,
            a.sum_d4,
            a.sum_art,
            a.sum_det,
            a.sum_mse,
            a.sum_hf_gain,
            a.sum_hf_loss,
            a.sum_hf_mag_loss,
            a.sum_pjnd,
            a.sum_pjnd_lo,
            a.sum_pjnd_hi,
            a.ws_peak_ssim.num,
            a.ws_peak_ssim.den,
            a.ws_peak_art.num,
            a.ws_peak_art.den,
            a.ws_peak_det.num,
            a.ws_peak_det.den,
            a.ws_mask_ssim.num,
            a.ws_mask_ssim.den,
            a.ws_mask_art.num,
            a.ws_mask_art.den,
            a.ws_mask_det.num,
            a.ws_mask_det.den,
            a.ws_mask_mse.num,
            a.ws_mask_mse.den,
            a.ws_iw_ssim.num,
            a.ws_iw_ssim.den,
            a.ws_iw_art.num,
            a.ws_iw_art.den,
            a.ws_iw_det.num,
            a.ws_iw_det.den,
            a.ws_iw_mse.num,
            a.ws_iw_mse.den,
        ]
    }

    /// The reference implementation of the dense pooled-feature math.
    ///
    /// Plain scalar loops, row-major, unambiguous left-to-right order, every
    /// per-pixel term evaluated in f64 with the SAME formulas the production
    /// scalar tail uses (`ssim_d_local`, `bounded_sim`, `saturate`,
    /// `bounded_excess_pair`, `bounded_excess`, `pjnd_transducer`) — this is a
    /// reference for the ACCUMULATION, so the formulas are shared rather than
    /// re-derived, and any formula change is caught by the production tests
    /// rather than silently diverging here.
    ///
    /// Also returns `sum_abs`, the per-slot `Σ|xᵢ|`, which is what the error
    /// bounds in design §10.5 are proportional to — so the gate can compute
    /// each slot's bound from the same run that measures its deviation.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dense_reference<A: Acc>(
        src: &[f32],
        dst: &[f32],
        mu1: &[f32],
        mu2: &[f32],
        ssq: &[f32],
        s12: &[f32],
        activity: &[f32],
        width: usize,
        height: usize,
        transducer_bank: bool,
    ) -> ([f64; N_SLOTS], [f64; N_SLOTS]) {
        let mut s: Slots<A> = Slots::default();
        let mut sum_abs = [0.0f64; N_SLOTS];
        let push = |s: &mut Slots<A>, sum_abs: &mut [f64; N_SLOTS], i: usize, v: f64| {
            s.add(i, v);
            sum_abs[i] += v.abs();
        };

        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                let sv = src[i] as f64;
                let dv = dst[i] as f64;
                let m1 = mu1[i] as f64;
                let m2 = mu2[i] as f64;
                let act = activity[i] as f64;

                let d = ssim_d_local(
                    m1,
                    m2,
                    s12[i] as f64,
                    ssq[i] as f64,
                    crate::ssim_form::active_revision()
                        >= crate::feature_defs::FormulaRevision::Rev3, false);
                push(&mut s, &mut sum_abs, 0, d);
                push(&mut s, &mut sum_abs, 1, d * d);
                push(&mut s, &mut sum_abs, 2, d * d * d);
                push(&mut s, &mut sum_abs, 3, d * d * d * d);

                let diff_src = (sv - m1).abs();
                let diff_dst = (dv - m2).abs();
                let edge_dissim = bounded_dissim(diff_src, diff_dst, C_EDGE, false);
                let (mut art_i, mut det_i) = (0.0, 0.0);
                if diff_dst > diff_src {
                    art_i = edge_dissim;
                } else if diff_dst < diff_src {
                    det_i = edge_dissim;
                }
                push(&mut s, &mut sum_abs, 4, art_i);
                push(&mut s, &mut sum_abs, 5, det_i);

                let raw_sq_err = (sv - dv) * (sv - dv);
                let mse_i = saturate(raw_sq_err, C_MSE);
                push(&mut s, &mut sum_abs, 6, mse_i);

                let hf_src = sv - m1;
                let hf_dst = dv - m2;
                let (hf_gain_i, hf_loss_i) =
                    bounded_excess_pair(hf_dst * hf_dst, hf_src * hf_src, C_HF);
                push(&mut s, &mut sum_abs, 7, hf_gain_i);
                push(&mut s, &mut sum_abs, 8, hf_loss_i);
                push(
                    &mut s,
                    &mut sum_abs,
                    9,
                    bounded_excess(hf_src.abs(), hf_dst.abs(), C_HF),
                );

                let raw_abs_err = (sv - dv).abs();
                push(
                    &mut s,
                    &mut sum_abs,
                    10,
                    pjnd_transducer(raw_abs_err, act, K_PJND_MASK, C_PJND_CLAMP),
                );
                if transducer_bank {
                    push(
                        &mut s,
                        &mut sum_abs,
                        11,
                        pjnd_transducer(raw_abs_err, act, K_PJND_MASK_LOW, C_PJND_CLAMP),
                    );
                    push(
                        &mut s,
                        &mut sum_abs,
                        12,
                        pjnd_transducer(raw_abs_err, act, K_PJND_MASK_HIGH, C_PJND_CLAMP),
                    );
                }

                // Weighted pools, in the production order.
                let sat_act = saturate(act, C_ACTIVITY);
                let mask_w = 1.0 - sat_act;
                let iw_w = sat_act + IW_WEIGHT_FLOOR;
                let sal_ssim = saturate(d, C_PEAK);
                let sal_art = saturate(art_i, C_PEAK);
                let sal_det = saturate(det_i, C_PEAK);
                for (base, w, v) in [
                    (13, sal_ssim, d),
                    (15, sal_art, art_i),
                    (17, sal_det, det_i),
                    (19, mask_w, d),
                    (21, mask_w, art_i),
                    (23, mask_w, det_i),
                    (25, mask_w, mse_i),
                    (27, iw_w, d),
                    (29, iw_w, art_i),
                    (31, iw_w, det_i),
                    (33, iw_w, mse_i),
                ] {
                    push(&mut s, &mut sum_abs, base, w * v);
                    push(&mut s, &mut sum_abs, base + 1, w);
                }
            }
        }
        (s.values(), sum_abs)
    }
}

// ============================================================================
// ERA-2 DENSE KERNEL (`benchmarks/era2_perf_break_2026-08-31.md` §2, §13.2, §14)
//
// NOT YET THE DEFAULT. Built alongside era-1 so it can be gated and measured
// without moving a byte; the flip is a separate, sequenced commit (design §8 —
// it lands after the fold-engine lane pins its parity stages).
//
// THE SHAPE, and every part of it is semantics rather than implementation:
//   * 8 f32 virtual lanes per accumulator, term at pixel x -> lane[x mod 8],
//     entered via `as_chunks::<8>` — NEVER modulo indexing, which was MEASURED
//     11x slower because it defeats vectorisation entirely (§13);
//   * the scalar tail folds into the SAME lanes, so the shape is
//     width-independent (era-1's tail landed in the f64 running total, which is
//     what made its grouping depend on `width % 8`);
//   * a FIXED reduction tree — `era2_reduce8`, never `reduce_add()`, which is
//     TIER-DEPENDENT (§14.2: x86_v3 pairs lane i with i+4 first, wasm/scalar
//     pair adjacent lanes — different groupings of the same eight addends);
//   * an f64 BAND layer: rows fold into a band partial, band partials merge in
//     band order. Band layout is a pure function of (height, ERA2_BAND_ROWS),
//     never of thread count.
//
// PASS STRUCTURE IS A PERF KNOB, NOT SEMANTICS. Splitting the row into several
// passes changes neither which terms an accumulator sees nor their order — each
// accumulator still receives its terms in increasing x — so any pass split
// yields IDENTICAL bytes. That frees the split to be tuned per tier for
// register pressure without touching the era.

/// era-2 band height. A pure function of geometry; deliberately NOT derived
/// from the thread count, which is what makes thread-invariance structural.
// NOTE ON `dead_code`: these primitives are consumed by the era-2 dense kernel,
// which lands in the SEQUENCED flip commit (design §8 — after the fold-engine
// lane pins its parity stages). Until then they are exercised only by the
// era-2 structural gates, so a non-test build sees them as unused. The
// allow comes off with the kernel.
#[allow(dead_code)]
pub(crate) const ERA2_BAND_ROWS: usize = 32;

/// The era-2 horizontal reduction — defined in [`crate::featcanon`], whose
/// canonical kernels need it in every build (this module exists only with
/// `feature-regime-v2`); re-exported here under its era-2 name.
#[allow(unused_imports)]
pub(crate) use crate::featcanon::era2_reduce8;

/// One era-2 accumulator: 8 f32 virtual lanes.
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct Lanes8(pub [f32; 8]);
#[allow(dead_code)]
impl Lanes8 {
    #[inline(always)]
    fn zero() -> Self {
        Self([0.0; 8])
    }
    /// Add one full 8-wide chunk, lane-wise.
    #[inline(always)]
    fn add_chunk(&mut self, c: &[f32; 8]) {
        for (l, &v) in self.0.iter_mut().zip(c.iter()) {
            *l += v;
        }
    }
    /// Fold a partial tail into the SAME lanes: tail element k -> lane k.
    #[inline(always)]
    fn add_tail(&mut self, t: &[f32]) {
        for (l, &v) in self.0.iter_mut().zip(t.iter()) {
            *l += v;
        }
    }
    #[inline(always)]
    fn reduce(self) -> f64 {
        era2_reduce8(self.0)
    }
}

/// **THE ERA-2 DENSE KERNEL.**
///
/// One code path, no tier-specific bodies. Written over `[f32; 8]` fixed
/// arrays so cross-tier bit-identity is not a proof obligation but a
/// tautology — there is only one implementation — while LLVM still
/// auto-vectorises the fixed-size chunks (the workspace's fixed-size-array
/// pattern). If a tier ever needs a hand-written body for speed, it must be
/// proven identical to this one; the oracle gate is where that happens.
///
/// SHAPE (all of it semantics, see `benchmarks/era2_perf_break_2026-08-31.md`):
/// 8 f32 virtual lanes entered via `as_chunks::<8>` with the tail folded into
/// the same lanes; `era2_reduce8`'s fixed tree, never `reduce_add()`; per-row
/// reduction into an f64 band partial (bounding the f32 lane depth exactly as
/// era-1 did, so accuracy is unchanged); band partials merged in band index
/// order, with `ERA2_BAND_ROWS` a pure function of geometry.
///
/// PASS SPLIT (perf only — §14.4 proves any split gives identical bytes):
/// **two passes per row.** Pass A evaluates the terms once, accumulates the 13
/// core families, and writes the four reusable per-pixel values to a row
/// scratch; pass B reads that scratch for the 22 pool accumulators. This keeps
/// each pass's live accumulator set inside a 16-register tier's budget without
/// re-deriving the division-heavy terms, and it is the split to revisit first
/// if the tier numbers ask for it.
/// Cross-vendor / cross-tier harness access: all 35 dense accumulator slots
/// for both eras, so a static binary can print per-slot hashes on two boxes.
/// `benchmarks/era2_perf_break_2026-08-31.md` §15.
#[cfg(any(test, feature = "oracle"))]
#[allow(clippy::too_many_arguments)]
pub fn harness_dense_slots(
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
    era2: bool,
) -> [f64; 35] {
    let a = if era2 {
        dense_block_kernel_era2(
            src,
            dst,
            mu1,
            mu2,
            ssq,
            s12,
            activity,
            width,
            height,
            transducer_bank,
            None,
        )
    } else {
        dense_block_kernel_era1(
            src,
            dst,
            mu1,
            mu2,
            ssq,
            s12,
            activity,
            width,
            height,
            transducer_bank,
            None,
        )
    };
    oracle::dense_accum_slots(&a)
}

/// Which SIMD tier the dispatcher actually selected on this host — reported by
/// the harness so a cross-box difference is never mistaken for a vendor effect
/// when it is a TIER effect (they are confounded on real hardware).
#[cfg(any(test, feature = "oracle"))]
pub fn harness_active_tier() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken as _;
        // Tier labels corrected 2026-09-01 (Profile-D no-tax refactor):
        // archmage's own token docs (`X64V3Token` = "x86-64-v3", AVX2+FMA+
        // BMI1/2, Haswell 2013 / Zen 1 2017; `X64V4Token` = "AVX-512"
        // baseline; `X64V4xToken` = "x86-64-v4x", AVX-512 + VBMI2/GFNI/VNNI)
        // — this function previously reported "v4 (AVX2)" / "v3 (SSE4.2)",
        // both one tier off (v4 is AVX-512, not AVX2; v3 is AVX2, not
        // SSE4.2). There is no dedicated SSE4.2-only tier in current
        // archmage/magetypes; x86-64-v2 (SSE4.2+POPCNT) exists as a token but
        // is not part of this dispatcher's ladder.
        if archmage::X64V4xToken::summon().is_some() {
            return "v4x (AVX-512+VBMI2/GFNI/VNNI)";
        }
        if archmage::X64V4Token::summon().is_some() {
            return "v4 (AVX-512)";
        }
        if archmage::X64V3Token::summon().is_some() {
            return "v3 (AVX2+FMA)";
        }
    }
    "scalar/other"
}

/// Bench/gate access to the era-1 dispatched kernel (see `dense_block_kernel`).
#[cfg(any(test, feature = "oracle"))]
#[allow(clippy::too_many_arguments)]
pub fn bench_dense_era1(
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
) -> f64 {
    let a = dense_block_kernel_era1(
        src,
        dst,
        mu1,
        mu2,
        ssq,
        s12,
        activity,
        width,
        height,
        transducer_bank,
        None,
    );
    a.sum_d + a.ws_iw_mse.num
}

/// Bench/gate access to the era-2 kernel.
#[cfg(any(test, feature = "oracle"))]
#[allow(clippy::too_many_arguments)]
pub fn bench_dense_era2(
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
) -> f64 {
    let a = dense_block_kernel_era2(
        src,
        dst,
        mu1,
        mu2,
        ssq,
        s12,
        activity,
        width,
        height,
        transducer_bank,
        None,
    );
    a.sum_d + a.ws_iw_mse.num
}

/// featacc cost bench (`benches/featacc_extract_ab.rs`): set the measurement
/// override in-process so one zenbench group can interleave candidates.
/// `mode`: "prod" (env-unset — shipped Rev4 arithmetic), "off", "c32",
/// "c64", "neum", "exact". `blur`: "rec", "rec64", "fresh", or `None` to
/// keep the mode's default. Returns false on an unrecognised name.
/// Measurement-only — exists solely under `oracle`.
#[cfg(feature = "oracle")]
pub fn bench_featcanon(mode: &str, blur: Option<&str>) -> bool {
    use crate::featcanon::{BlurMode, Mode};
    let m = match mode {
        "prod" => None,
        "off" => Some(Mode::Off),
        "c32" => Some(Mode::Canon32),
        "c64" => Some(Mode::Canon64),
        "neum" => Some(Mode::CanonNeum),
        "exact" => Some(Mode::Exact),
        _ => return false,
    };
    let b = blur.map(|s| match s {
        "rec" => BlurMode::Rec,
        "rec64" => BlurMode::Rec64,
        "fresh" => BlurMode::Fresh,
        _ => BlurMode::Rec,
    });
    if matches!(blur, Some("rec" | "rec64" | "fresh") | None) {
        crate::featcanon::bench_set_measurement(m, b);
        true
    } else {
        false
    }
}

/// Tier dispatch for the era-2 kernel. The BODY is one source shared by every
/// tier (see `dense_block_kernel_era2_inner`); `#[magetypes]` exists here only
/// to put that body inside each tier's `target_feature` region.
///
/// **This is what the first measurement was missing.** Plain Rust compiles to
/// baseline x86-64 (SSE2) no matter what the host supports, which measured
/// **4.4x slower** than era-1's `#[arcane]`-wrapped kernel — not a shape
/// problem, an ISA problem. Cross-tier bit-identity survives because the
/// source is identical and Rust does not contract `a*b+c` into an FMA without
/// fast-math (which is off); it is VERIFIED, not assumed, by
/// `era2_vendor_probe` across the AMD/Intel pair.
#[allow(clippy::too_many_arguments)]
fn dense_block_kernel_era2(
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
    r4: Option<Rev4Dense<'_>>,
) -> DenseAccum {
    incant!(
        dense_block_kernel_era2_entry(
            src,
            dst,
            mu1,
            mu2,
            ssq,
            s12,
            activity,
            width,
            height,
            transducer_bank,
            r4
        ),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

// Two disjoint-tier entries emitting the same base name (the era-1
// `dense_block_kernel_entry` pattern), kept because §14.4's byte-neutrality
// claim is asserted against BOTH instantiations by
// `era2_fused_and_two_pass_are_bit_identical`.
//
// BOTH ARE CURRENTLY `false` (two-pass), because the fused hypothesis was
// MEASURED AND FALSIFIED. The reasoning was that v4x's 32 SIMD registers would
// hold 13 core + 16 pool accumulators, so it could skip the scratch
// round-trip. Measured at 576x128: fused **445.5 us** against two-pass
// **226.1 us** — 2x WORSE, taking era-2 from 2.12x to 3.98x of era-1. 29
// `Lanes8` accumulators is 232 live f32 values; 32 registers do not hold that,
// and it spills. This is the same wall §A.14 hit when it scalarised the pools
// in the first place, and why POOL_SIMD was capped at 16 accumulators rather
// than 22. The split is not a 16-register concession — it is the right
// structure on every tier.
#[magetypes(+v4x, -v4, -v3, -neon, -wasm128, -scalar)]
#[allow(clippy::too_many_arguments, dead_code)]
fn dense_block_kernel_era2_entry(
    token: Token,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
    r4: Option<Rev4Dense<'_>>,
) -> DenseAccum {
    dense_block_kernel_era2_generic::<_, false>(
        token,
        src,
        dst,
        mu1,
        mu2,
        ssq,
        s12,
        activity,
        width,
        height,
        transducer_bank,
        r4,
    )
}

#[magetypes(v4, v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments, dead_code)]
fn dense_block_kernel_era2_entry(
    token: Token,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
    r4: Option<Rev4Dense<'_>>,
) -> DenseAccum {
    dense_block_kernel_era2_generic::<_, false>(
        token,
        src,
        dst,
        mu1,
        mu2,
        ssq,
        s12,
        activity,
        width,
        height,
        transducer_bank,
        r4,
    )
}

/// Accumulate the first `n` lanes of a term array into a `V8` accumulator,
/// preserving lane identity and add order.
///
/// Used ONLY for the row tail. Zero-padding the tail into a full `V8` add is
/// NOT equivalent: the pool weights are `mask_w = 1 - saturate(act)`, and
/// `saturate(0) = 0`, so a padded lane contributes `mask_w = 1.0` (and
/// `iw_w = IW_WEIGHT_FLOOR`) to the weight denominators. Those are not
/// accumulator-neutral, so the tail is masked rather than padded.
#[inline(always)]
#[allow(dead_code)]
fn v8_add_first<T: F32x8Backend + Copy>(token: T, acc: V8<T>, t: &[f32; 8], n: usize) -> V8<T> {
    let mut a = acc.to_array();
    for k in 0..n {
        a[k] += t[k];
    }
    V8::<T>::from_array_t(token, a)
}

/// `#[inline(always)]`, not `#[inline]` — this body exists ONLY to fuse into
/// its entry's `target_feature` region. era-1 measured a **5.3x** whole-
/// extraction regression when the equivalent hint stopped being honoured and
/// the vector ops compiled into calls outside the feature region; the same
/// trap applies here verbatim.
///
/// V8 REWRITE (era-2 stage B): accumulators are `V8<T>` magetypes vectors, not
/// `[f32; 8]` arrays. `V8` **is** the 8 virtual lanes, so every semantic is
/// unchanged — lane `j` still takes the terms at `x ≡ j (mod 8)` in increasing
/// `x`, the reduction is still the explicit `era2_reduce8` tree over
/// `to_array()` and never `reduce_add()` (which is tier-dependent, §14.2), the
/// tail still folds into lane `k`, and band partials still merge in band order.
/// Term evaluation moves to the `_v` helpers, i.e. **the same functions era-1's
/// SIMD path uses**, so the per-pixel math is era-1's exactly.
#[inline(always)]
#[allow(clippy::too_many_arguments, dead_code)]
fn dense_block_kernel_era2_generic<T: F32x8Backend + Copy, const FUSED: bool>(
    token: T,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
    mut r4: Option<Rev4Dense<'_>>,
) -> DenseAccum {
    // Revision 3 (issue #61): the `s12` plane is the direct error moment, so
    // the v2 SSIM signal must read it as such. Hoisted once per call;
    // loop-invariant, so every call site below unswitches on it.
    let stable = crate::ssim_form::active_revision().stable_forms();
    let direct = crate::ssim_form::active_revision() >= crate::feature_defs::FormulaRevision::Rev3;
    let _ = FUSED; // the split is chosen by tier; see the entries above
    let zero = V8::<T>::zero_t(token);
    let one = V8::<T>::splat_t(token, 1.0);
    let c1 = V8::<T>::splat_t(token, C1_V2 as f32);
    let c2 = V8::<T>::splat_t(token, C2_V2 as f32);
    let c_edge = V8::<T>::splat_t(token, C_EDGE as f32);
    let c_mse = V8::<T>::splat_t(token, C_MSE as f32);
    let c_hf = V8::<T>::splat_t(token, C_HF as f32);
    let c_pj = V8::<T>::splat_t(token, C_PJND_CLAMP as f32);
    let k_mid = V8::<T>::splat_t(token, K_PJND_MASK as f32);
    let k_lo = V8::<T>::splat_t(token, K_PJND_MASK_LOW as f32);
    let k_hi = V8::<T>::splat_t(token, K_PJND_MASK_HIGH as f32);
    let c_act = V8::<T>::splat_t(token, C_ACTIVITY as f32);
    let c_peak = V8::<T>::splat_t(token, C_PEAK as f32);
    let iw_floor = V8::<T>::splat_t(token, IW_WEIGHT_FLOOR as f32);

    let mut acc = DenseAccum::default();
    let mut b0 = 0usize;
    while b0 < height {
        let b1 = (b0 + ERA2_BAND_ROWS).min(height);
        let mut band = [0.0f64; 35];

        for y in b0..b1 {
            let row = y * width;
            // 13 core + 16 pool accumulators, all V8.
            let mut r = [zero; 13];
            let mut p_mw = zero;
            let mut p_iw = zero;
            let mut p_m = [zero; 4];
            let mut p_i = [zero; 4];
            let mut p_kn = [zero; 3];
            let mut p_kd = [zero; 3];

            // One term-evaluation closure, shared by the full-chunk and tail
            // paths so they cannot drift. Returns every term as a V8.
            macro_rules! terms {
                ($s:expr, $dd:expr, $m1:expr, $m2:expr, $q:expr, $p:expr, $act:expr) => {{
                    let (s, dd, m1, m2, q, p, act) = ($s, $dd, $m1, $m2, $q, $p, $act);
                    let d = ssim_d_local_v(token, m1, m2, p, q, c1, c2, direct, stable);
                    let d2 = d * d;
                    let diff_src = (s - m1).abs();
                    let diff_dst = (dd - m2).abs();
                    let edge_dissim = bounded_dissim_v(token, diff_src, diff_dst, c_edge, stable);
                    let gt = diff_dst.simd_gt(diff_src);
                    let lt = diff_dst.simd_lt(diff_src);
                    let art_i = V8::<T>::blend(gt, edge_dissim, zero);
                    let det_i = V8::<T>::blend(lt, edge_dissim, zero);
                    let raw_diff = s - dd;
                    let mse_i = saturate_v(token, raw_diff * raw_diff, c_mse);
                    let hf_src = s - m1;
                    let hf_dst = dd - m2;
                    let (hfg, hfl) =
                        bounded_excess_pair_v(token, hf_dst * hf_dst, hf_src * hf_src, c_hf);
                    let hfm = bounded_excess_v(token, hf_src.abs(), hf_dst.abs(), c_hf);
                    let rae = raw_diff.abs();
                    let pj = pjnd_transducer_v(token, rae, act, k_mid, c_pj);
                    let (pjl, pjh) = if transducer_bank {
                        (
                            pjnd_transducer_v(token, rae, act, k_lo, c_pj),
                            pjnd_transducer_v(token, rae, act, k_hi, c_pj),
                        )
                    } else {
                        (zero, zero)
                    };
                    // d³ and d⁴ share `d2`, matching era-1's SIMD path exactly
                    // (`d2 * d`, `d2 * d2`) rather than re-multiplying from `d`.
                    (
                        [
                            d,
                            d2,
                            d2 * d,
                            d2 * d2,
                            art_i,
                            det_i,
                            mse_i,
                            hfg,
                            hfl,
                            hfm,
                            pj,
                            pjl,
                            pjh,
                        ],
                        art_i,
                        det_i,
                        mse_i,
                        d,
                        act,
                    )
                }};
            }

            // Pool contributions from already-evaluated terms.
            macro_rules! pools {
                ($d:expr, $art:expr, $det:expr, $mse:expr, $act:expr) => {{
                    let (d, art_i, det_i, mse_i, act) = ($d, $art, $det, $mse, $act);
                    let sat = saturate_v(token, act, c_act);
                    let m_w = one - sat;
                    let i_w = sat + iw_floor;
                    let vals = [d, art_i, det_i, mse_i];
                    let mut mv = [zero; 4];
                    let mut iv = [zero; 4];
                    for j in 0..4 {
                        mv[j] = m_w * vals[j];
                        iv[j] = i_w * vals[j];
                    }
                    let mut kn = [zero; 3];
                    let mut kd = [zero; 3];
                    for j in 0..3 {
                        let sal = saturate_v(token, vals[j], c_peak);
                        kn[j] = sal * vals[j];
                        kd[j] = sal;
                    }
                    (m_w, i_w, mv, iv, kn, kd)
                }};
            }

            let n8 = width - width % 8;
            let mut x = 0usize;
            macro_rules! ld {
                ($plane:expr, $off:expr) => {
                    V8::<T>::from_array_t(token, $plane[$off..$off + 8].try_into().unwrap())
                };
            }
            while x < n8 {
                let i = row + x;
                let (t, art_i, det_i, mse_i, d, act) = terms!(
                    ld!(src, i),
                    ld!(dst, i),
                    ld!(mu1, i),
                    ld!(mu2, i),
                    ld!(ssq, i),
                    ld!(s12, i),
                    ld!(activity, i)
                );
                for j in 0..13 {
                    r[j] += t[j];
                }
                let (m_w, i_w, mv, iv, kn, kd) = pools!(d, art_i, det_i, mse_i, act);
                p_mw += m_w;
                p_iw += i_w;
                for j in 0..4 {
                    p_m[j] += mv[j];
                    p_i[j] += iv[j];
                }
                for j in 0..3 {
                    p_kn[j] += kn[j];
                    p_kd[j] += kd[j];
                }
                // REV4: C3 histogram + C4 flat-HF — scalarized per lane
                // (the §A.14 pattern; the bit-domain edges cannot
                // vectorize). `t[7]` is `hfg`. One Option check per
                // chunk; `None` is the pre-REV4 op sequence bit for bit.
                if let Some(r4) = r4.as_mut() {
                    let d_a = d.to_array();
                    let art_a = art_i.to_array();
                    let det_a = det_i.to_array();
                    let mse_a = mse_i.to_array();
                    let hfg_a = t[7].to_array();
                    let act_a = act.to_array();
                    for lane in 0..8 {
                        rev4_dense_pixel(
                            r4,
                            d_a[lane] as f64,
                            art_a[lane] as f64,
                            det_a[lane] as f64,
                            mse_a[lane] as f64,
                            hfg_a[lane] as f64,
                            act_a[lane] as f64,
                            lane,
                        );
                    }
                }
                x += 8;
            }
            // TAIL: masked, never zero-padded (see `v8_add_first`).
            let rem = width - n8;
            if rem != 0 {
                let pad = |pl: &[f32]| -> V8<T> {
                    let mut o = [0.0f32; 8];
                    o[..rem].copy_from_slice(&pl[row + n8..row + width]);
                    V8::<T>::from_array_t(token, o)
                };
                let (t, art_i, det_i, mse_i, d, act) = terms!(
                    pad(src),
                    pad(dst),
                    pad(mu1),
                    pad(mu2),
                    pad(ssq),
                    pad(s12),
                    pad(activity)
                );
                for j in 0..13 {
                    r[j] = v8_add_first(token, r[j], &t[j].to_array(), rem);
                }
                let (m_w, i_w, mv, iv, kn, kd) = pools!(d, art_i, det_i, mse_i, act);
                p_mw = v8_add_first(token, p_mw, &m_w.to_array(), rem);
                p_iw = v8_add_first(token, p_iw, &i_w.to_array(), rem);
                for j in 0..4 {
                    p_m[j] = v8_add_first(token, p_m[j], &mv[j].to_array(), rem);
                    p_i[j] = v8_add_first(token, p_i[j], &iv[j].to_array(), rem);
                }
                for j in 0..3 {
                    p_kn[j] = v8_add_first(token, p_kn[j], &kn[j].to_array(), rem);
                    p_kd[j] = v8_add_first(token, p_kd[j], &kd[j].to_array(), rem);
                }
                // REV4 tail: only the first `rem` lanes are real pixels
                // (the padded lanes' values are discarded, matching the
                // masked-fold convention).
                if let Some(r4) = r4.as_mut() {
                    let d_a = d.to_array();
                    let art_a = art_i.to_array();
                    let det_a = det_i.to_array();
                    let mse_a = mse_i.to_array();
                    let hfg_a = t[7].to_array();
                    let act_a = act.to_array();
                    for lane in 0..rem {
                        rev4_dense_pixel(
                            r4,
                            d_a[lane] as f64,
                            art_a[lane] as f64,
                            det_a[lane] as f64,
                            mse_a[lane] as f64,
                            hfg_a[lane] as f64,
                            act_a[lane] as f64,
                            lane,
                        );
                    }
                }
            }

            // Per-row reduction into the f64 band partial, via the FIXED tree.
            for j in 0..13 {
                band[j] += era2_reduce8(r[j].to_array());
            }
            for j in 0..3 {
                band[13 + j * 2] += era2_reduce8(p_kn[j].to_array());
                band[14 + j * 2] += era2_reduce8(p_kd[j].to_array());
            }
            let mw_row = era2_reduce8(p_mw.to_array());
            let iw_row = era2_reduce8(p_iw.to_array());
            for j in 0..4 {
                band[19 + j * 2] += era2_reduce8(p_m[j].to_array());
                band[20 + j * 2] += mw_row;
                band[27 + j * 2] += era2_reduce8(p_i[j].to_array());
                band[28 + j * 2] += iw_row;
            }
        }

        // Band partials merge in BAND INDEX ORDER.
        let s = &mut acc;
        s.sum_d += band[0];
        s.sum_d2 += band[1];
        s.sum_d3 += band[2];
        s.sum_d4 += band[3];
        s.sum_art += band[4];
        s.sum_det += band[5];
        s.sum_mse += band[6];
        s.sum_hf_gain += band[7];
        s.sum_hf_loss += band[8];
        s.sum_hf_mag_loss += band[9];
        s.sum_pjnd += band[10];
        s.sum_pjnd_lo += band[11];
        s.sum_pjnd_hi += band[12];
        for (w, o) in [
            (&mut s.ws_peak_ssim, 13),
            (&mut s.ws_peak_art, 15),
            (&mut s.ws_peak_det, 17),
            (&mut s.ws_mask_ssim, 19),
            (&mut s.ws_mask_art, 21),
            (&mut s.ws_mask_det, 23),
            (&mut s.ws_mask_mse, 25),
            (&mut s.ws_iw_ssim, 27),
            (&mut s.ws_iw_art, 29),
            (&mut s.ws_iw_det, 31),
            (&mut s.ws_iw_mse, 33),
        ] {
            w.num += band[o];
            w.den += band[o + 1];
        }
        b0 = b1;
    }
    acc
}

// ============================================================================
// featacc dense candidates — measurement-only bodies
// ============================================================================
//
// `dense_block_kernel_era2_generic` IS the shipped canonical arithmetic for
// the dense block: f32 SIMD lanes (`x mod 8`) reduced per row through the
// fixed `era2_reduce8` tree into f64 band partials. The mirrors below put
// the accumulation on the `Pool` axis and the element eval on its own axis
// so the candidates can be measured instead of asserted:
//
// * [`dense_block_kernel_canon`]<P>: per-pixel SCALAR f32 element eval in
//   the SIMD body's exact op order — magetypes `*`/`+`/`/` are unfused on
//   every backend, so the scalar replay is element-bitwise with era-2 on
//   every tier — then `P::add(x & 7, term)` into per-row pools, `fin()` per
//   row, `band[j] +=`, strip `+=` — the era-2 accumulation structure with
//   only the pool arithmetic swapped. `P = LanesF32` replays era-2 bit for
//   bit; `LanesF64`/`Neum64` are candidates (e)/(f).
// * [`dense_block_kernel_exact`]: f64 element eval via the existing f64
//   scalar helpers (`ssim_d_local`, `bounded_sim`, …), Neumaier pools —
//   the oracle arm.

/// The 13 core + 16 pool terms of one dense pixel, f32 scalar — mirrors
/// `dense_block_kernel_era2_generic`'s `terms!`/`pools!` op-for-op.
#[inline(always)]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub(crate) fn dense_terms32(
    s: f32,
    dd: f32,
    m1: f32,
    m2: f32,
    q: f32,
    p: f32,
    act: f32,
    transducer_bank: bool,
    direct: bool,
    stable: bool,
) -> (
    [f32; 13],
    f32,
    f32,
    f32,
    f32,
    f32,
    [f32; 4],
    [f32; 4],
    [f32; 3],
    [f32; 3],
) {
    let c1 = C1_V2 as f32;
    let c2 = C2_V2 as f32;
    // `ssim_d_local_v` — identical op order (p = s12, q = ssq).
    let d = {
        let a = 2.0f32 * m1 * m2 + c1;
        let b = m1 * m1 + m2 * m2 + c1;
        let dl = q - m1 * m1 - m2 * m2 + c2;
        if direct {
            let mu_diff = m1 - m2;
            let err_var = (p - mu_diff * mu_diff).max(0.0);
            let lum = if stable { mu_diff * mu_diff } else { b - a };
            ((dl * lum + a * err_var) / (b * dl)).max(0.0)
        } else {
            let cov = p - m1 * m2;
            let c = 2.0f32 * cov + c2;
            let local = (a * c) / (b * dl);
            (1.0f32 - local).max(0.0)
        }
    };
    let d2 = d * d;
    let diff_src = (s - m1).abs();
    let diff_dst = (dd - m2).abs();
    // `bounded_dissim_v` — `1 − (2ab+c)/(a²+b²+c)`, or revision 7's exact
    // `(a−b)²/(a²+b²+c)`, plain ops in the same order.
    let edge_dissim = crate::feature_v2::bounded_dissim32(diff_src, diff_dst, C_EDGE as f32, stable);
    // `simd_gt`/`blend` — scalar select, same values.
    let (art_i, det_i) = if diff_dst > diff_src {
        (edge_dissim, 0.0f32)
    } else if diff_dst < diff_src {
        (0.0f32, edge_dissim)
    } else {
        (0.0f32, 0.0f32)
    };
    let raw_diff = s - dd;
    // `saturate_v(raw_diff², c_mse)` — `.max(0)` then `x/(x+c)`.
    let raw_sq = raw_diff * raw_diff;
    let mse_i = raw_sq.max(0.0) / (raw_sq.max(0.0) + C_MSE as f32);
    let hf_src = s - m1;
    let hf_dst = dd - m2;
    // `bounded_excess_pair_v` — ONE denom, TWO divisions (no recip-mul).
    let (hfg, hfl) = {
        let a = hf_dst * hf_dst;
        let b = hf_src * hf_src;
        let denom = a + b + C_HF as f32;
        ((a - b).max(0.0) / denom, (b - a).max(0.0) / denom)
    };
    let hfm = (hf_src.abs() - hf_dst.abs()).max(0.0) / (hf_src.abs() + hf_dst.abs() + C_HF as f32);
    let rae = raw_diff.abs();
    // `pjnd_transducer_v` — `rae / (rae + c*(1 + k*act))`.
    let pj = rae / (rae + C_PJND_CLAMP as f32 * (1.0f32 + K_PJND_MASK as f32 * act));
    let (pjl, pjh) = if transducer_bank {
        (
            rae / (rae + C_PJND_CLAMP as f32 * (1.0f32 + K_PJND_MASK_LOW as f32 * act)),
            rae / (rae + C_PJND_CLAMP as f32 * (1.0f32 + K_PJND_MASK_HIGH as f32 * act)),
        )
    } else {
        (0.0, 0.0)
    };
    let t = [
        d,
        d2,
        d2 * d,
        d2 * d2,
        art_i,
        det_i,
        mse_i,
        hfg,
        hfl,
        hfm,
        pj,
        pjl,
        pjh,
    ];
    // `pools!` — mask/IW weights + soft-peak saliencies, f32.
    let sat = act.max(0.0) / (act.max(0.0) + C_ACTIVITY as f32);
    let m_w = 1.0f32 - sat;
    let i_w = sat + IW_WEIGHT_FLOOR as f32;
    let vals = [d, art_i, det_i, mse_i];
    let mut mv = [0.0f32; 4];
    let mut iv = [0.0f32; 4];
    for j in 0..4 {
        mv[j] = m_w * vals[j];
        iv[j] = i_w * vals[j];
    }
    let mut kn = [0.0f32; 3];
    let mut kd = [0.0f32; 3];
    for j in 0..3 {
        let sal = vals[j].max(0.0) / (vals[j].max(0.0) + C_PEAK as f32);
        kn[j] = sal * vals[j];
        kd[j] = sal;
    }
    (t, art_i, det_i, mse_i, m_w, i_w, mv, iv, kn, kd)
}

/// `Pool`-generic scalar dense kernel — see the module note above.
#[allow(clippy::too_many_arguments)]
fn dense_block_kernel_canon<P: crate::featcanon::Pool>(
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
    mut r4: Option<Rev4Dense<'_>>,
    revision: FormulaRevision,
) -> DenseAccum {
    use crate::featcanon::{Lanes16F32, ROW_LANES_FOLD};
    let direct = revision >= FormulaRevision::Rev3;
    let stable_moments = revision >= FormulaRevision::Rev5;
    let forms7 = revision.stable_forms();
    // Revision 7's row lanes (`crate::featcanon::Lanes16F32`): the columns of
    // the row's complete eight-column chunks add in f32 lanes, folded into the
    // f64 pools every `ROW_LANES_FOLD` columns and at the end of the chunks;
    // the f64 pools live for the band and finish once per band; the central
    // moments are two-pass per row.
    let rev7 = revision.row_lanes();
    let chunk_end = (width / 8) * 8;
    let mut dvals = if rev7 { vec![0.0f64; width] } else { Vec::new() };
    let mut acc = DenseAccum::default();
    let mut b0 = 0usize;
    while b0 < height {
        let b1 = (b0 + ERA2_BAND_ROWS).min(height);
        let mut band = [0.0f64; 35];
        // Revision 7: the f64 lanes live for the whole band (each row's row
        // lanes fold into them), and each pool is finished once per band.
        let mut band_pools = [P::zero(); 29];
        for y in b0..b1 {
            let row = y * width;
            // `[r 0..13 | mw | iw | m 0..4 | i 0..4 | kn 0..3 | kd 0..3]`, the
            // vector body's layout.
            let mut row_pools = [P::zero(); 29];
            let pools = if rev7 {
                &mut band_pools
            } else {
                &mut row_pools
            };
            let (r, rest) = pools.split_at_mut(13);
            let (mw_iw, rest) = rest.split_at_mut(2);
            let (p_m, rest) = rest.split_at_mut(4);
            let (p_i, rest) = rest.split_at_mut(4);
            let (p_kn, p_kd) = rest.split_at_mut(3);
            let (p_mw, p_iw) = mw_iw.split_at_mut(1);
            let r: &mut [P; 13] = r.try_into().unwrap();
            let (p_mw, p_iw) = (&mut p_mw[0], &mut p_iw[0]);
            let (p_m, p_i): (&mut [P; 4], &mut [P; 4]) =
                (p_m.try_into().unwrap(), p_i.try_into().unwrap());
            let (p_kn, p_kd): (&mut [P; 3], &mut [P; 3]) =
                (p_kn.try_into().unwrap(), p_kd.try_into().unwrap());
            let mut local = [0.0f64; 16];
            let mut r32 = [Lanes16F32::default(); 13];
            let (mut p_mw32, mut p_iw32) = (Lanes16F32::default(), Lanes16F32::default());
            let (mut p_m32, mut p_i32) = ([Lanes16F32::default(); 4], [Lanes16F32::default(); 4]);
            let (mut p_kn32, mut p_kd32) = ([Lanes16F32::default(); 3], [Lanes16F32::default(); 3]);
            for x in 0..width {
                let i = row + x;
                let d = if rev7 && x < chunk_end {
                    let d = dense_elem_canon(
                        src[i],
                        dst[i],
                        mu1[i],
                        mu2[i],
                        ssq[i],
                        s12[i],
                        activity[i],
                        x,
                        transducer_bank,
                        direct,
                        forms7,
                        &mut r4,
                        &mut r32,
                        &mut p_mw32,
                        &mut p_iw32,
                        &mut p_m32,
                        &mut p_i32,
                        &mut p_kn32,
                        &mut p_kd32,
                    );
                    if (x + 1).is_multiple_of(ROW_LANES_FOLD) || x + 1 == chunk_end {
                        for (l32, l) in r32.iter_mut().zip(r.iter_mut()) {
                            l32.fold_into(l);
                        }
                        p_mw32.fold_into(p_mw);
                        p_iw32.fold_into(p_iw);
                        for (l32, l) in p_m32.iter_mut().zip(p_m.iter_mut()) {
                            l32.fold_into(l);
                        }
                        for (l32, l) in p_i32.iter_mut().zip(p_i.iter_mut()) {
                            l32.fold_into(l);
                        }
                        for (l32, l) in p_kn32.iter_mut().zip(p_kn.iter_mut()) {
                            l32.fold_into(l);
                        }
                        for (l32, l) in p_kd32.iter_mut().zip(p_kd.iter_mut()) {
                            l32.fold_into(l);
                        }
                    }
                    d
                } else {
                    dense_elem_canon(
                        src[i],
                        dst[i],
                        mu1[i],
                        mu2[i],
                        ssq[i],
                        s12[i],
                        activity[i],
                        x,
                        transducer_bank,
                        direct,
                        forms7,
                        &mut r4,
                        r,
                        p_mw,
                        p_iw,
                        p_m,
                        p_i,
                        p_kn,
                        p_kd,
                    )
                };
                if rev7 {
                    dvals[x] = d as f64;
                } else if stable_moments {
                    local[x & 15] = d as f64;
                    if x & 15 == 15 || x + 1 == width {
                        acc.central
                            .merge(OnlineMoments::block(&local[..(x & 15) + 1]));
                    }
                }
            }
            if rev7 {
                acc.central.merge(row_moments_canon(&dvals));
            } else {
                dense_band_add(&mut band, &row_pools, rev7);
            }
        }
        if rev7 {
            dense_band_add(&mut band, &band_pools, rev7);
        }
        let s = &mut acc;
        s.sum_d += band[0];
        s.sum_d2 += band[1];
        s.sum_d3 += band[2];
        s.sum_d4 += band[3];
        s.sum_art += band[4];
        s.sum_det += band[5];
        s.sum_mse += band[6];
        s.sum_hf_gain += band[7];
        s.sum_hf_loss += band[8];
        s.sum_hf_mag_loss += band[9];
        s.sum_pjnd += band[10];
        s.sum_pjnd_lo += band[11];
        s.sum_pjnd_hi += band[12];
        for (w, o) in [
            (&mut s.ws_peak_ssim, 13),
            (&mut s.ws_peak_art, 15),
            (&mut s.ws_peak_det, 17),
            (&mut s.ws_mask_ssim, 19),
            (&mut s.ws_mask_art, 21),
            (&mut s.ws_mask_det, 23),
            (&mut s.ws_mask_mse, 25),
            (&mut s.ws_iw_ssim, 27),
            (&mut s.ws_iw_art, 29),
            (&mut s.ws_iw_det, 31),
            (&mut s.ws_iw_mse, 33),
        ] {
            w.num += band[o];
            w.den += band[o + 1];
        }
        b0 = b1;
    }
    acc
}

/// The `f32x8` sibling of [`dense_terms32`] — identical op order (the
/// scalar mirror was written to era-2's `terms!`/`pools!`; this calls the
/// same `*_v` helpers). All plain unfused ops — no `mul_add` anywhere —
/// so every lane IS one scalar element's rounding.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))] // only the x86_64 lane-parallel canon bodies call it
#[inline(always)]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn dense_terms32_v<V: Vf32>(
    token: V::Tok,
    s: V,
    dd: V,
    m1: V,
    m2: V,
    q: V,
    p: V,
    act: V,
    transducer_bank: bool,
    direct: bool,
    stable: bool,
) -> (
    [V; 13],
    V,
    V,
    V,
    V,
    V,
    [V; 4],
    [V; 4],
    [V; 3],
    [V; 3],
    V,
) {
    let c1 = V::splat_t(token, C1_V2 as f32);
    let c2 = V::splat_t(token, C2_V2 as f32);
    let c_edge = V::splat_t(token, C_EDGE as f32);
    let c_mse = V::splat_t(token, C_MSE as f32);
    let c_hf = V::splat_t(token, C_HF as f32);
    let c_pj = V::splat_t(token, C_PJND_CLAMP as f32);
    let k_mid = V::splat_t(token, K_PJND_MASK as f32);
    let k_lo = V::splat_t(token, K_PJND_MASK_LOW as f32);
    let k_hi = V::splat_t(token, K_PJND_MASK_HIGH as f32);
    let c_act = V::splat_t(token, C_ACTIVITY as f32);
    let c_peak = V::splat_t(token, C_PEAK as f32);
    let iw_floor = V::splat_t(token, IW_WEIGHT_FLOOR as f32);
    let one = V::splat_t(token, 1.0);
    let zero = V::zero_t(token);
    let d = ssim_d_local_v(token, m1, m2, p, q, c1, c2, direct, stable);
    let d2 = d * d;
    let diff_src = (s - m1).abs();
    let diff_dst = (dd - m2).abs();
    let edge_dissim = bounded_dissim_v(token, diff_src, diff_dst, c_edge, stable);
    let gt = diff_dst.simd_gt(diff_src);
    let lt = diff_dst.simd_lt(diff_src);
    let art_i = V::blend(gt, edge_dissim, zero);
    let det_i = V::blend(lt, edge_dissim, zero);
    let raw_diff = s - dd;
    let mse_i = saturate_v(token, raw_diff * raw_diff, c_mse);
    let hf_src = s - m1;
    let hf_dst = dd - m2;
    let (hfg, hfl) = bounded_excess_split_v(token, hf_dst * hf_dst, hf_src * hf_src, c_hf);
    let hfm = bounded_excess_v(token, hf_src.abs(), hf_dst.abs(), c_hf);
    let rae = raw_diff.abs();
    let pj = pjnd_transducer_v(token, rae, act, k_mid, c_pj);
    let (pjl, pjh) = if transducer_bank {
        (
            pjnd_transducer_v(token, rae, act, k_lo, c_pj),
            pjnd_transducer_v(token, rae, act, k_hi, c_pj),
        )
    } else {
        (zero, zero)
    };
    let t = [
        d,
        d2,
        d2 * d,
        d2 * d2,
        art_i,
        det_i,
        mse_i,
        hfg,
        hfl,
        hfm,
        pj,
        pjl,
        pjh,
    ];
    let sat = saturate_v(token, act, c_act);
    let m_w = one - sat;
    let i_w = sat + iw_floor;
    let vals = [d, art_i, det_i, mse_i];
    let mut mv = [zero; 4];
    let mut iv = [zero; 4];
    for j in 0..4 {
        mv[j] = m_w * vals[j];
        iv[j] = i_w * vals[j];
    }
    // `art_i` and `det_i` are `edge_dissim` or `+0`, never both nonzero, and
    // `saturate(+0)` is `+0`: one division gives both soft-peak saturations.
    let sal_edge = saturate_v(token, edge_dissim, c_peak);
    let sal = [
        saturate_v(token, d, c_peak),
        V::blend(gt, sal_edge, zero),
        V::blend(lt, sal_edge, zero),
    ];
    let mut kn = [zero; 3];
    let mut kd = [zero; 3];
    for j in 0..3 {
        kn[j] = sal[j] * vals[j];
        kd[j] = sal[j];
    }
    (t, art_i, det_i, mse_i, m_w, i_w, mv, iv, kn, kd, act)
}

/// One column's contribution to the dense canonical pools — the scalar
/// element step shared by [`dense_block_kernel_canon`]'s `Pool`-generic
/// body and the canon64 vector body's `width % 8` tail.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn dense_elem_canon<P: crate::featcanon::Pool>(
    s: f32,
    dd: f32,
    m1: f32,
    m2: f32,
    q: f32,
    p: f32,
    act: f32,
    x: usize,
    transducer_bank: bool,
    direct: bool,
    stable: bool,
    r4: &mut Option<Rev4Dense<'_>>,
    r: &mut [P; 13],
    p_mw: &mut P,
    p_iw: &mut P,
    p_m: &mut [P; 4],
    p_i: &mut [P; 4],
    p_kn: &mut [P; 3],
    p_kd: &mut [P; 3],
) -> f32 {
    let lane = x & (P::LANES - 1);
    let (t, art_i, det_i, mse_i, m_w, i_w, mv, iv, kn, kd) =
        dense_terms32(s, dd, m1, m2, q, p, act, transducer_bank, direct, stable);
    for j in 0..13 {
        r[j].add(lane, t[j]);
    }
    p_mw.add(lane, m_w);
    p_iw.add(lane, i_w);
    for j in 0..4 {
        p_m[j].add(lane, mv[j]);
        p_i[j].add(lane, iv[j]);
    }
    for j in 0..3 {
        p_kn[j].add(lane, kn[j]);
        p_kd[j].add(lane, kd[j]);
    }
    if let Some(r4) = r4.as_mut() {
        rev4_dense_pixel(
            r4,
            t[0] as f64,
            art_i as f64,
            det_i as f64,
            mse_i as f64,
            t[7] as f64,
            act as f64,
            lane,
        );
    }
    t[0]
}

/// **canon64 vector body** — [`dense_block_kernel_canon`]'s `P = LanesF64`
/// arm with the term evaluation run eight columns at a time through
/// `f32x8` and the `width % 8` tail on [`dense_elem_canon`]. Every lane is
/// one column's scalar formula (all unfused ops; the `simd_gt`/`blend`
/// selection is the scalar `if/else` — `gt`/`lt` are both false on
/// equality, matching `(0, 0)`); `x & 7` chunk lanes are the canonical
/// lanes in order, so each `add_chunk` is the same eight `add(l, v)` calls
/// in the same per-pool order. `rev4_dense_pixel` stays scalar per
/// element (its `scatter` cannot vectorize). `_scalar`/`_wasm128` keep the
/// scalar canonical body.
// The AVX-512 builds keep the per-element pool scatter (`rev4_dense_pixel`,
// the `add(l, v)` lane pools) scalar — LLVM otherwise fuses the 29 pool
// updates into `vgatherdps`/`vscatterqpd` (4.3x slower than v3, measured
// 2026-10-03 on the separate-pool form). The chunk update below goes
// through `f64x8` repr ops (same `vwin_slide64x8!` pattern as fused.rs):
// each pool is one vector load/add/store, so there is no scalar per-lane
// RMW for LLVM to vectorize into gather/scatter.
/// Chunks of eight columns the sixteen-lane dense pools buffer between pool
/// updates (even, so a block starts on lanes 0..8; see the kernel below).
const DENSE_ILV: usize = 16;

/// The dense pools revision 7 accumulates: all but `d²`, `d³` and `d⁴`
/// (pools 1–3). From revision 5 on the SSIM deviation features read the
/// central moments, and nothing reads those three raw sums, so revision 7
/// leaves `sum_d2`, `sum_d3` and `sum_d4` at zero.
const DENSE_POOLS7: [usize; 26] = [
    0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
];

/// Add the 29 dense pools' finished sums to a band's 35 sums (`[r 0..13 | kn,kd
/// pairs | m,mw pairs | i,iw pairs]`; the mean-weight and importance-weight
/// sums are each the denominator of four weighted pools). Revision 7
/// (`rev7`) skips the raw `d²`, `d³`, `d⁴` pools ([`DENSE_POOLS7`]).
#[inline(always)]
fn dense_band_add<P: crate::featcanon::Pool>(band: &mut [f64; 35], pools: &[P; 29], rev7: bool) {
    for j in 0..13 {
        if rev7 && (1..4).contains(&j) {
            continue;
        }
        band[j] += pools[j].fin();
    }
    for j in 0..3 {
        band[13 + j * 2] += pools[23 + j].fin();
        band[14 + j * 2] += pools[26 + j].fin();
    }
    let mw = pools[13].fin();
    let iw = pools[14].fin();
    for j in 0..4 {
        band[19 + j * 2] += pools[15 + j].fin();
        band[20 + j * 2] += mw;
        band[27 + j * 2] += pools[19 + j].fin();
        band[28 + j * 2] += iw;
    }
}

#[magetypes(define(f32x8, f32x16, f64x8), v4x, v4, v3, -scalar)]
#[allow(clippy::too_many_arguments)]
fn dense_block_kernel_canon64_vec<P: crate::featcanon::Pool64, const R7: bool>(
    token: Token,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
    r4: Option<Rev4Dense<'_>>,
    revision: FormulaRevision,
) -> DenseAccum {
    let mut r4 = r4;
    let direct = revision >= FormulaRevision::Rev3;
    let stable_moments = P::LANES == 16;
    let forms7 = revision.stable_forms();
    let mut acc = DenseAccum::default();
    // Provable-length plane slices: the chunk loop reads `i + 8 <= row +
    // full * 8` and the tail reads `row + x < height * width` — both literal
    // once the planes are cut to their exact spans, so no per-element
    // `slice_index_fail` path remains.
    let src = &src[..height * width];
    let dst = &dst[..height * width];
    let mu1 = &mu1[..height * width];
    let mu2 = &mu2[..height * width];
    let ssq = &ssq[..height * width];
    let s12 = &s12[..height * width];
    let activity = &activity[..height * width];
    // Blocked interchange for the sixteen-lane pools (DENSEPOOL,
    // `benchmarks/dense_pool_brief_2026-10-24.md`): each chunk's 29 term
    // vectors go to an L1-resident buffer, and every `DENSE_ILV` chunks each
    // pool loads its two lane halves once, adds the buffered chunks in order
    // and stores once. Lane `l` still adds column `x` (`x & 15 == l`) in
    // ascending `x` from the row's zero, so every sum is bit-identical; only
    // the pool memory traffic changes (a pool round trip per chunk was most
    // of this kernel's time). The eight-lane pools keep the per-chunk update.
    let mut ilv = [[0.0f32; 8]; 29 * DENSE_ILV];
    // The stable central moments of `d` (Rev5+): eight full 16-column
    // blocks at a time, one block per `f64x8` lane (`cols[k][b]` is value
    // `k` of block `b`). Each lane runs `OnlineMoments::block`'s scalar
    // sequence op for op (the same `-0.0`-seeded sum of `v − v₀`, mean and
    // `m2/m3/m4` adds), and the blocks merge in raster order, so the moments
    // are bit-identical; the serial per-block chain is what costs.
    let mut cols = [[0.0f64; 8]; 16];
    let central8 = |cols: &[[f64; 8]; 16], blocks: usize, central: &mut OnlineMoments| {
        if blocks == 0 {
            return;
        }
        let origin = f64x8::load_t(token, &cols[0]);
        let mut sum = f64x8::splat_t(token, -0.0);
        for col in cols {
            sum += f64x8::load_t(token, col) - origin;
        }
        let mean = origin + sum / f64x8::splat_t(token, 16.0);
        let zero = f64x8::zero_t(token);
        let (mut m2, mut m3, mut m4) = (zero, zero, zero);
        for col in cols {
            let d = f64x8::load_t(token, col) - mean;
            let d2 = d * d;
            m2 += d2;
            m3 += d2 * d;
            m4 += d2 * d2;
        }
        let (mean, m2, m3, m4) = (mean.to_array(), m2.to_array(), m3.to_array(), m4.to_array());
        for b in 0..blocks {
            central.merge(OnlineMoments {
                n: 16.0,
                mean: mean[b],
                m2: m2[b],
                m3: m3[b],
                m4: m4[b],
            });
        }
    };
    // Revision 7 (`rowlanes`, `crate::featcanon::Lanes16F32`): the 29 pools
    // add each complete chunk's columns in f32 row lanes, folded into the f64
    // lanes every `ROW_LANES_FOLD` columns and before the tail (the f64 lanes
    // live for the band, finished once per band), and the central moments of
    // `d` are two-pass per row (`row_moments_canon`'s ops, lane-parallel)
    // with one merge per row. Revisions 5 and 6 keep their bits.
    // `R7` is `revision.row_lanes()` at compile time, so a revision-7 instance
    // carries none of the eight-lane and interleave buffers below.
    let rev7 = R7;
    debug_assert_eq!(rev7, revision.row_lanes());
    debug_assert!(!rev7 || stable_moments, "revision 7 runs on the sixteen-lane pools");
    let mut dvals = if rev7 { vec![0.0f64; width] } else { Vec::new() };
    let row_moments = |v: &[f64]| -> OnlineMoments {
        let n = v.len();
        let full16 = n / 16;
        let zero = f64x8::zero_t(token);
        let lanes = |lo: f64x8, hi: f64x8| -> [f64; 16] {
            let mut l = [0.0f64; 16];
            l[..8].copy_from_slice(&lo.to_array());
            l[8..].copy_from_slice(&hi.to_array());
            l
        };
        let (mut lo, mut hi) = (zero, zero);
        for b in 0..full16 {
            lo += f64x8::load_t(token, v[16 * b..16 * b + 8].try_into().unwrap());
            hi += f64x8::load_t(token, v[16 * b + 8..16 * b + 16].try_into().unwrap());
        }
        let mut s1 = lanes(lo, hi);
        for x in full16 * 16..n {
            s1[x & 15] += v[x];
        }
        let mean = crate::featcanon::Pool::fin(crate::featcanon::Lanes16F64(s1)) / n as f64;
        let m = f64x8::splat_t(token, mean);
        let (mut a2, mut b2, mut a3, mut b3, mut a4, mut b4) = (zero, zero, zero, zero, zero, zero);
        for b in 0..full16 {
            let dl = f64x8::load_t(token, v[16 * b..16 * b + 8].try_into().unwrap()) - m;
            let dh = f64x8::load_t(token, v[16 * b + 8..16 * b + 16].try_into().unwrap()) - m;
            let (ql, qh) = (dl * dl, dh * dh);
            a2 += ql;
            b2 += qh;
            a3 += ql * dl;
            b3 += qh * dh;
            a4 += ql * ql;
            b4 += qh * qh;
        }
        let (mut s2, mut s3, mut s4) = (lanes(a2, b2), lanes(a3, b3), lanes(a4, b4));
        for x in full16 * 16..n {
            let d = v[x] - mean;
            let q = d * d;
            s2[x & 15] += q;
            s3[x & 15] += q * d;
            s4[x & 15] += q * q;
        }
        OnlineMoments {
            n: n as f64,
            mean,
            m2: crate::featcanon::Pool::fin(crate::featcanon::Lanes16F64(s2)),
            m3: crate::featcanon::Pool::fin(crate::featcanon::Lanes16F64(s3)),
            m4: crate::featcanon::Pool::fin(crate::featcanon::Lanes16F64(s4)),
        }
    };
    // Revision 7's f32 row lanes; every row's fold clears them.
    let mut pools32 = [crate::featcanon::Lanes16F32::default(); 29];
    let mut b0 = 0usize;
    while b0 < height {
        let b1 = (b0 + ERA2_BAND_ROWS).min(height);
        let mut band = [0.0f64; 35];
        // Revision 7: the f64 lanes live for the whole band (each row's row
        // lanes fold into them), and each pool is finished once per band.
        let mut band_pools = [P::zero(); 29];
        for y in b0..b1 {
            let row = y * width;
            // One flat accumulator row: `[r 0..13 | mw | iw | m 0..4 |
            // i 0..4 | kn 0..3 | kd 0..3]`. The chunk update goes through
            // `f64x8` repr ops — one vector load/add/store per pool — so
            // there is no scalar per-lane RMW for the AVX-512 build to
            // fuse into `vgatherqpd`/`vscatterqpd` (the 4.3x regression
            // the former v3-only dispatch worked around). The tail slices
            // give `dense_elem_canon` the same typed pool refs — zero cost.
            let mut row_pools = [P::zero(); 29];
            let pools = if rev7 { &mut band_pools } else { &mut row_pools };
            let mut local = [0.0f64; 16];
            let mut blocks = 0usize;
            let full = width / 8;
            let srow = &src[row..row + full * 8];
            let drow = &dst[row..row + full * 8];
            let m1row = &mu1[row..row + full * 8];
            let m2row = &mu2[row..row + full * 8];
            let qrow = &ssq[row..row + full * 8];
            let prow = &s12[row..row + full * 8];
            let arow = &activity[row..row + full * 8];
            // Per eight-column chunk `c`: the pool, moment and rev4 updates.
            // `bufs` holds the chunk's 29 pooled term vectors in pool order
            // and `act8` its activity (the rev4 bank also reads `bufs` 0 and
            // 4–7: `d`, `art`, `det`, `mse` and the HF gain).
            macro_rules! chunk8_post {
                ($c:expr, $bufs:expr, $act8:expr) => {{
                    let c: usize = $c;
                    let i = c * 8;
                    let bufs: [[f32; 8]; 29] = $bufs;
                if rev7 {
                    let off = i & 15;
                    for j in DENSE_POOLS7 {
                        let lanes = pools32[j].chunk8(off);
                        (f32x8::load_t(token, lanes) + f32x8::load_t(token, &bufs[j])).store(lanes);
                    }
                    let dv: &mut [f64; 8] = (&mut dvals[i..i + 8]).try_into().unwrap();
                    for l in 0..8 {
                        dv[l] = bufs[0][l] as f64;
                    }
                    if (i + 8).is_multiple_of(crate::featcanon::ROW_LANES_FOLD) {
                        for j in DENSE_POOLS7 {
                            pools32[j].fold_into(&mut pools[j]);
                        }
                    }
                } else if stable_moments {
                    // Chunk `c` adds to lanes `(c & 1) * 8..`; a block starts
                    // at an even chunk (`DENSE_ILV` is even), so a buffered
                    // chunk's parity is its position's.
                    let k = c % DENSE_ILV;
                    for j in 0..29 {
                        ilv[j * DENSE_ILV + k] = bufs[j];
                    }
                    if k + 1 == DENSE_ILV || c + 1 == full {
                        for j in 0..29 {
                            let blk = &ilv[j * DENSE_ILV..j * DENSE_ILV + k + 1];
                            let mut lo = f64x8::load_t(token, pools[j].chunk8(0));
                            let mut hi = f64x8::load_t(token, pools[j].chunk8(8));
                            let (pairs, rest) = blk.as_chunks::<2>();
                            for [a, b] in pairs {
                                let a: [f64; 8] = std::array::from_fn(|l| a[l] as f64);
                                let b: [f64; 8] = std::array::from_fn(|l| b[l] as f64);
                                lo += f64x8::from_array_t(token, a);
                                hi += f64x8::from_array_t(token, b);
                            }
                            if let [last] = rest {
                                let a: [f64; 8] = std::array::from_fn(|l| last[l] as f64);
                                lo += f64x8::from_array_t(token, a);
                            }
                            lo.store(pools[j].chunk8(0));
                            hi.store(pools[j].chunk8(8));
                        }
                    }
                } else {
                    for j in 0..29 {
                        let a: [f64; 8] = std::array::from_fn(|l| bufs[j][l] as f64);
                        let lanes = pools[j].chunk8(i & (P::LANES - 1));
                        (f64x8::load_t(token, lanes) + f64x8::from_array_t(token, a))
                            .store(lanes);
                    }
                }
                if stable_moments && !rev7 {
                    let off = i & 15;
                    for l in 0..8 {
                        local[off + l] = bufs[0][l] as f64;
                        cols[off + l][blocks] = bufs[0][l] as f64;
                    }
                    if off == 8 {
                        blocks += 1;
                        if blocks == 8 {
                            central8(&cols, blocks, &mut acc.central);
                            blocks = 0;
                        }
                    } else if c + 1 == full && width.is_multiple_of(8) {
                        // The row ends on an eight-value block.
                        central8(&cols, blocks, &mut acc.central);
                        blocks = 0;
                        acc.central.merge(OnlineMoments::block(&local[..8]));
                    }
                }
                    if let Some(r4) = r4.as_mut() {
                        rev4_dense_chunk8(r4, &bufs[0], &bufs[4], &bufs[5], &bufs[6], &bufs[7], $act8);
                    }
                }};
            }
            let mut c = 0usize;
            // Revision 7 on AVX-512 tiers: sixteen columns per step (two
            // chunks). Every term is element-wise, so each lane computes the
            // eight-wide step's bits; the sixteen values go to the sixteen f32
            // row lanes in one vector op (lane `x & 15`, the two chunks'
            // order), or, with the rev4 bank on, back through the two chunks'
            // updates. Revisions 5 and 6 keep the eight-wide loop below.
            while rev7 && steps_sixteen::<Token>() && c + 2 <= full {
                let i = c * 8;
                macro_rules! ld16 {
                    ($r:expr) => {
                        f32x16::load_t(token, $r[i..i + 16].try_into().unwrap())
                    };
                }
                let (t, _, _, _, m_w, i_w, mv, iv, kn, kd, act_v) = dense_terms32_v(
                    token,
                    ld16!(srow),
                    ld16!(drow),
                    ld16!(m1row),
                    ld16!(m2row),
                    ld16!(qrow),
                    ld16!(prow),
                    ld16!(arow),
                    transducer_bank,
                    direct,
                    forms7,
                );
                let vecs: [f32x16; 29] = [
                    t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11], t[12],
                    m_w, i_w, mv[0], mv[1], mv[2], mv[3], iv[0], iv[1], iv[2], iv[3], kn[0], kn[1],
                    kn[2], kd[0], kd[1], kd[2],
                ];
                if rev7 && r4.is_none() {
                    for j in DENSE_POOLS7 {
                        let lanes = &mut pools32[j].0;
                        (f32x16::load_t(token, lanes) + vecs[j]).store(lanes);
                    }
                    let d = vecs[0].to_array();
                    let dv: &mut [f64; 16] = (&mut dvals[i..i + 16]).try_into().unwrap();
                    for l in 0..16 {
                        dv[l] = d[l] as f64;
                    }
                    if (i + 16).is_multiple_of(crate::featcanon::ROW_LANES_FOLD) {
                        for j in DENSE_POOLS7 {
                            pools32[j].fold_into(&mut pools[j]);
                        }
                    }
                } else {
                    let arrs: [[f32; 16]; 29] = vecs.map(|v| v.to_array());
                    let act16 = act_v.to_array();
                    for h in 0..2 {
                        let bufs: [[f32; 8]; 29] =
                            std::array::from_fn(|j| arrs[j][8 * h..8 * h + 8].try_into().unwrap());
                        let act8: [f32; 8] = act16[8 * h..8 * h + 8].try_into().unwrap();
                        chunk8_post!(c + h, bufs, &act8);
                    }
                }
                c += 2;
            }
            // The remaining chunk (an odd count) and every chunk on v3: the
            // eight-wide step, unchanged.
            for c in c..full {
                let i = c * 8;
                let sv = f32x8::load_t(token, srow[i..i + 8].try_into().unwrap());
                let dv = f32x8::load_t(token, drow[i..i + 8].try_into().unwrap());
                let m1v = f32x8::load_t(token, m1row[i..i + 8].try_into().unwrap());
                let m2v = f32x8::load_t(token, m2row[i..i + 8].try_into().unwrap());
                let qv = f32x8::load_t(token, qrow[i..i + 8].try_into().unwrap());
                let pv = f32x8::load_t(token, prow[i..i + 8].try_into().unwrap());
                let actv = f32x8::load_t(token, arow[i..i + 8].try_into().unwrap());
                let (t, art_i, det_i, mse_i, m_w, i_w, mv, iv, kn, kd, act_v) = dense_terms32_v(
                    token,
                    sv,
                    dv,
                    m1v,
                    m2v,
                    qv,
                    pv,
                    actv,
                    transducer_bank,
                    direct,
                    forms7,
                );
                let bufs: [[f32; 8]; 29] = [
                    t[0].to_array(),
                    t[1].to_array(),
                    t[2].to_array(),
                    t[3].to_array(),
                    t[4].to_array(),
                    t[5].to_array(),
                    t[6].to_array(),
                    t[7].to_array(),
                    t[8].to_array(),
                    t[9].to_array(),
                    t[10].to_array(),
                    t[11].to_array(),
                    t[12].to_array(),
                    m_w.to_array(),
                    i_w.to_array(),
                    mv[0].to_array(),
                    mv[1].to_array(),
                    mv[2].to_array(),
                    mv[3].to_array(),
                    iv[0].to_array(),
                    iv[1].to_array(),
                    iv[2].to_array(),
                    iv[3].to_array(),
                    kn[0].to_array(),
                    kn[1].to_array(),
                    kn[2].to_array(),
                    kd[0].to_array(),
                    kd[1].to_array(),
                    kd[2].to_array(),
                ];
                if rev7 {
                    let off = i & 15;
                    for j in DENSE_POOLS7 {
                        let lanes = pools32[j].chunk8(off);
                        (f32x8::load_t(token, lanes) + f32x8::load_t(token, &bufs[j])).store(lanes);
                    }
                    let dv: &mut [f64; 8] = (&mut dvals[i..i + 8]).try_into().unwrap();
                    for l in 0..8 {
                        dv[l] = bufs[0][l] as f64;
                    }
                    if (i + 8).is_multiple_of(crate::featcanon::ROW_LANES_FOLD) {
                        for j in DENSE_POOLS7 {
                            pools32[j].fold_into(&mut pools[j]);
                        }
                    }
                } else if stable_moments {
                    // Chunk `c` adds to lanes `(c & 1) * 8..`; a block starts
                    // at an even chunk (`DENSE_ILV` is even), so a buffered
                    // chunk's parity is its position's.
                    let k = c % DENSE_ILV;
                    for j in 0..29 {
                        ilv[j * DENSE_ILV + k] = bufs[j];
                    }
                    if k + 1 == DENSE_ILV || c + 1 == full {
                        for j in 0..29 {
                            let blk = &ilv[j * DENSE_ILV..j * DENSE_ILV + k + 1];
                            let mut lo = f64x8::load_t(token, pools[j].chunk8(0));
                            let mut hi = f64x8::load_t(token, pools[j].chunk8(8));
                            let (pairs, rest) = blk.as_chunks::<2>();
                            for [a, b] in pairs {
                                let a: [f64; 8] = std::array::from_fn(|l| a[l] as f64);
                                let b: [f64; 8] = std::array::from_fn(|l| b[l] as f64);
                                lo += f64x8::from_array_t(token, a);
                                hi += f64x8::from_array_t(token, b);
                            }
                            if let [last] = rest {
                                let a: [f64; 8] = std::array::from_fn(|l| last[l] as f64);
                                lo += f64x8::from_array_t(token, a);
                            }
                            lo.store(pools[j].chunk8(0));
                            hi.store(pools[j].chunk8(8));
                        }
                    }
                } else {
                    for j in 0..29 {
                        let a: [f64; 8] = std::array::from_fn(|l| bufs[j][l] as f64);
                        let lanes = pools[j].chunk8(i & (P::LANES - 1));
                        (f64x8::load_t(token, lanes) + f64x8::from_array_t(token, a))
                            .store(lanes);
                    }
                }
                if stable_moments && !rev7 {
                    let off = i & 15;
                    for l in 0..8 {
                        local[off + l] = bufs[0][l] as f64;
                        cols[off + l][blocks] = bufs[0][l] as f64;
                    }
                    if off == 8 {
                        blocks += 1;
                        if blocks == 8 {
                            central8(&cols, blocks, &mut acc.central);
                            blocks = 0;
                        }
                    } else if c + 1 == full && width.is_multiple_of(8) {
                        // The row ends on an eight-value block.
                        central8(&cols, blocks, &mut acc.central);
                        blocks = 0;
                        acc.central.merge(OnlineMoments::block(&local[..8]));
                    }
                }
                if let Some(r4) = r4.as_mut() {
                    rev4_dense_chunk8(
                        r4,
                        &t[0].to_array(),
                        &art_i.to_array(),
                        &det_i.to_array(),
                        &mse_i.to_array(),
                        &t[7].to_array(),
                        &act_v.to_array(),
                    );
                }
            }
            // Full blocks before the tail's own blocks, in raster order.
            central8(&cols, blocks, &mut acc.central);
            // The end of the chunks, unless the last chunk already folded.
            if rev7 && !(full * 8).is_multiple_of(crate::featcanon::ROW_LANES_FOLD) {
                for j in DENSE_POOLS7 {
                    pools32[j].fold_into(&mut pools[j]);
                }
            }
            // Scalar tail — the canonical element step verbatim.
            {
                let (r, rest) = pools.split_at_mut(13);
                let (mw_iw, rest) = rest.split_at_mut(2);
                let (p_m, rest) = rest.split_at_mut(4);
                let (p_i, rest) = rest.split_at_mut(4);
                let (p_kn, p_kd) = rest.split_at_mut(3);
                let (mw, iw) = mw_iw.split_at_mut(1);
                for x in full * 8..width {
                    let i = row + x;
                    let d = dense_elem_canon(
                        src[i],
                        dst[i],
                        mu1[i],
                        mu2[i],
                        ssq[i],
                        s12[i],
                        activity[i],
                        x,
                        transducer_bank,
                        direct,
                        forms7,
                        &mut r4,
                        r.try_into().unwrap(),
                        &mut mw[0],
                        &mut iw[0],
                        p_m.try_into().unwrap(),
                        p_i.try_into().unwrap(),
                        p_kn.try_into().unwrap(),
                        p_kd.try_into().unwrap(),
                    );
                    if rev7 {
                        dvals[x] = d as f64;
                    } else if stable_moments {
                        local[x & 15] = d as f64;
                        if x & 15 == 15 || x + 1 == width {
                            acc.central
                                .merge(OnlineMoments::block(&local[..(x & 15) + 1]));
                        }
                    }
                }
            }
            if rev7 {
                acc.central.merge(row_moments(&dvals[..width]));
            } else {
                dense_band_add(&mut band, pools, rev7);
            }
        }
        if rev7 {
            dense_band_add(&mut band, &band_pools, rev7);
        }
        let s = &mut acc;
        s.sum_d += band[0];
        s.sum_d2 += band[1];
        s.sum_d3 += band[2];
        s.sum_d4 += band[3];
        s.sum_art += band[4];
        s.sum_det += band[5];
        s.sum_mse += band[6];
        s.sum_hf_gain += band[7];
        s.sum_hf_loss += band[8];
        s.sum_hf_mag_loss += band[9];
        s.sum_pjnd += band[10];
        s.sum_pjnd_lo += band[11];
        s.sum_pjnd_hi += band[12];
        for (w, o) in [
            (&mut s.ws_peak_ssim, 13),
            (&mut s.ws_peak_art, 15),
            (&mut s.ws_peak_det, 17),
            (&mut s.ws_mask_ssim, 19),
            (&mut s.ws_mask_art, 21),
            (&mut s.ws_mask_det, 23),
            (&mut s.ws_mask_mse, 25),
            (&mut s.ws_iw_ssim, 27),
            (&mut s.ws_iw_art, 29),
            (&mut s.ws_iw_det, 31),
            (&mut s.ws_iw_mse, 33),
        ] {
            w.num += band[o];
            w.den += band[o + 1];
        }
        b0 = b1;
    }
    acc
}

/// Scalar-tier siblings of [`dense_block_kernel_canon64_vec`]: scalar and
/// wasm128 keep the `LanesF64` canonical body.
#[allow(clippy::too_many_arguments)]
fn dense_block_kernel_canon64_vec_scalar<P: crate::featcanon::Pool64, const R7: bool>(
    _token: archmage::ScalarToken,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
    r4: Option<Rev4Dense<'_>>,
    revision: FormulaRevision,
) -> DenseAccum {
    dense_block_kernel_canon::<P>(
        src,
        dst,
        mu1,
        mu2,
        ssq,
        s12,
        activity,
        width,
        height,
        transducer_bank,
        r4,
        revision,
    )
}

/// Wasm128 sibling of [`dense_block_kernel_canon64_vec_scalar`].
#[allow(clippy::too_many_arguments, dead_code)]
fn dense_block_kernel_canon64_vec_wasm128<P: crate::featcanon::Pool64, const R7: bool>(
    _token: archmage::Wasm128Token,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
    r4: Option<Rev4Dense<'_>>,
    revision: FormulaRevision,
) -> DenseAccum {
    dense_block_kernel_canon::<P>(
        src,
        dst,
        mu1,
        mu2,
        ssq,
        s12,
        activity,
        width,
        height,
        transducer_bank,
        r4,
        revision,
    )
}

/// NEON runs the scalar canonical body (2026-10-03): NEON vector `max`/`min` propagate NaN where the canon's `f32::max`/
/// `f64::max` drop it (the REV4VEC CI aarch64 failure), so only the x86_64 fused tiers run a lane-parallel body.
#[allow(clippy::too_many_arguments, dead_code)]
fn dense_block_kernel_canon64_vec_neon<P: crate::featcanon::Pool64, const R7: bool>(
    _token: archmage::NeonToken,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
    r4: Option<Rev4Dense<'_>>,
    revision: FormulaRevision,
) -> DenseAccum {
    dense_block_kernel_canon::<P>(
        src,
        dst,
        mu1,
        mu2,
        ssq,
        s12,
        activity,
        width,
        height,
        transducer_bank,
        r4,
        revision,
    )
}

/// f64-exact sibling of [`dense_block_kernel_canon`] — the oracle arm.
/// Elements via the existing f64 scalar helpers (`ssim_d_local`,
/// `bounded_sim`, `bounded_excess`, `saturate`, `pjnd_transducer`); pools
/// Neumaier-compensated; the band structure is identical.
#[cfg(feature = "oracle")]
#[allow(clippy::too_many_arguments)]
fn dense_block_kernel_exact(
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
    mut r4: Option<Rev4Dense<'_>>,
) -> DenseAccum {
    use crate::featcanon::{Neum64, Pool as _};
    let direct = crate::ssim_form::active_revision() >= crate::feature_defs::FormulaRevision::Rev3;
    let stable_moments = crate::ssim_form::active_revision() >= FormulaRevision::Rev5;
    let forms7 = crate::ssim_form::active_revision().stable_forms();
    let mut acc = DenseAccum::default();
    let mut b0 = 0usize;
    while b0 < height {
        let b1 = (b0 + ERA2_BAND_ROWS).min(height);
        let mut band = [0.0f64; 35];
        for y in b0..b1 {
            let row = y * width;
            let mut r = [Neum64::zero(); 13];
            let mut p_mw = Neum64::zero();
            let mut p_iw = Neum64::zero();
            let mut p_m = [Neum64::zero(); 4];
            let mut p_i = [Neum64::zero(); 4];
            let mut p_kn = [Neum64::zero(); 3];
            let mut p_kd = [Neum64::zero(); 3];
            let mut local = [0.0f64; 16];
            for x in 0..width {
                let i = row + x;
                let lane = x & 7;
                let s = src[i] as f64;
                let dd = dst[i] as f64;
                let m1 = mu1[i] as f64;
                let m2 = mu2[i] as f64;
                let act = activity[i] as f64;
                let d = ssim_d_local(m1, m2, s12[i] as f64, ssq[i] as f64, direct, forms7);
                if stable_moments {
                    local[x & 15] = d;
                    if x & 15 == 15 || x + 1 == width {
                        acc.central
                            .merge(OnlineMoments::block(&local[..(x & 15) + 1]));
                    }
                }
                let d2 = d * d;
                let diff_src = (s - m1).abs();
                let diff_dst = (dd - m2).abs();
                let edge_dissim = bounded_dissim(diff_src, diff_dst, C_EDGE, forms7);
                let (art_i, det_i) = if diff_dst > diff_src {
                    (edge_dissim, 0.0)
                } else if diff_dst < diff_src {
                    (0.0, edge_dissim)
                } else {
                    (0.0, 0.0)
                };
                let raw_diff = s - dd;
                let mse_i = saturate(raw_diff * raw_diff, C_MSE);
                let hf_src = s - m1;
                let hf_dst = dd - m2;
                let (hfg, hfl) = bounded_excess_pair(hf_dst * hf_dst, hf_src * hf_src, C_HF);
                let hfm = bounded_excess(hf_src.abs(), hf_dst.abs(), C_HF);
                let rae = raw_diff.abs();
                let pj = pjnd_transducer(rae, act, K_PJND_MASK, C_PJND_CLAMP);
                let (pjl, pjh) = if transducer_bank {
                    (
                        pjnd_transducer(rae, act, K_PJND_MASK_LOW, C_PJND_CLAMP),
                        pjnd_transducer(rae, act, K_PJND_MASK_HIGH, C_PJND_CLAMP),
                    )
                } else {
                    (0.0, 0.0)
                };
                let t = [
                    d,
                    d2,
                    d2 * d,
                    d2 * d2,
                    art_i,
                    det_i,
                    mse_i,
                    hfg,
                    hfl,
                    hfm,
                    pj,
                    pjl,
                    pjh,
                ];
                for j in 0..13 {
                    r[j].add64(lane, t[j]);
                }
                // f64 pools: `weighted_pool_accumulate_scalar`'s formula.
                let mask_w = 1.0 - saturate(act, C_ACTIVITY);
                let iw_w = saturate(act, C_ACTIVITY) + IW_WEIGHT_FLOOR;
                p_mw.add64(lane, mask_w);
                p_iw.add64(lane, iw_w);
                let vals = [d, art_i, det_i, mse_i];
                for j in 0..4 {
                    p_m[j].add64(lane, mask_w * vals[j]);
                    p_i[j].add64(lane, iw_w * vals[j]);
                }
                for j in 0..3 {
                    let sal = saturate(vals[j], C_PEAK);
                    p_kn[j].add64(lane, sal * vals[j]);
                    p_kd[j].add64(lane, sal);
                }
                if let Some(r4) = r4.as_mut() {
                    rev4_dense_pixel(r4, d, art_i, det_i, mse_i, hfg, act, x);
                }
            }
            for j in 0..13 {
                band[j] += r[j].fin();
            }
            for j in 0..3 {
                band[13 + j * 2] += p_kn[j].fin();
                band[14 + j * 2] += p_kd[j].fin();
            }
            let mw_row = p_mw.fin();
            let iw_row = p_iw.fin();
            for j in 0..4 {
                band[19 + j * 2] += p_m[j].fin();
                band[20 + j * 2] += mw_row;
                band[27 + j * 2] += p_i[j].fin();
                band[28 + j * 2] += iw_row;
            }
        }
        let s = &mut acc;
        s.sum_d += band[0];
        s.sum_d2 += band[1];
        s.sum_d3 += band[2];
        s.sum_d4 += band[3];
        s.sum_art += band[4];
        s.sum_det += band[5];
        s.sum_mse += band[6];
        s.sum_hf_gain += band[7];
        s.sum_hf_loss += band[8];
        s.sum_hf_mag_loss += band[9];
        s.sum_pjnd += band[10];
        s.sum_pjnd_lo += band[11];
        s.sum_pjnd_hi += band[12];
        for (w, o) in [
            (&mut s.ws_peak_ssim, 13),
            (&mut s.ws_peak_art, 15),
            (&mut s.ws_peak_det, 17),
            (&mut s.ws_mask_ssim, 19),
            (&mut s.ws_mask_art, 21),
            (&mut s.ws_mask_det, 23),
            (&mut s.ws_mask_mse, 25),
            (&mut s.ws_iw_ssim, 27),
            (&mut s.ws_iw_art, 29),
            (&mut s.ws_iw_det, 31),
            (&mut s.ws_iw_mse, 33),
        ] {
            w.num += band[o];
            w.den += band[o + 1];
        }
        b0 = b1;
    }
    acc
}

/// featcanon-fix D1 at the crate-internal walk (`validate_wide_revision`): the
/// public entries refuse a Rev4 request first (D2), so this pins the walk's
/// OWN refusal — the guard `research::extract` and every internal caller rely
/// on — in both directions.
#[cfg(all(test, feature = "feature-regime-v2"))]
mod featcanon_contract_tests {
    use super::*;
    use crate::feature_defs::FormulaRevision;
    use crate::source::RgbSlice;

    fn walk(revision: FormulaRevision, v1_only: bool) -> Result<ZensimV2Result, ZensimError> {
        let (w, h) = (97, 63);
        let s = tests::textured_image(w, h, 7);
        let d = tests::quantize_distort(&s, w, h);
        let toggles = V2NewFeatureToggles {
            v1_only,
            formula_revision: revision,
            ..Default::default()
        };
        let mut scratch = V2Scratch::new();
        compute_folded720_streaming_impl(
            &RgbSlice::new(&s, w, h),
            &RgbSlice::new(&d, w, h),
            None,
            false,
            toggles,
            &mut scratch,
            None,
        )
    }

    fn assert_mix_refused(r: Result<ZensimV2Result, ZensimError>, what: &str) {
        match r {
            Err(ZensimError::ModelLoadFailed { reason }) => {
                assert_eq!(reason, crate::ssim_form::REV4_MIX, "{what}")
            }
            Err(e) => panic!("{what}: refused for another reason: {e:?}"),
            Ok(_) => panic!("{what}: served a Rev4 mix"),
        }
    }

    /// Direction 2: a Rev4 request in a Rev3 process — v1-only included,
    /// which `validate_wide_revision` used to wave through.
    #[test]
    fn the_walk_refuses_a_rev4_request_in_a_rev3_process() {
        const SENTINEL: &str = "WALK-REV4-IN-REV3-RAN";
        if !crate::ssim_form::run_at_revision(
            "3",
            "feature_v2::featcanon_contract_tests::the_walk_refuses_a_rev4_request_in_a_rev3_process",
            SENTINEL,
        ) {
            return;
        }
        for v1_only in [true, false] {
            assert_mix_refused(
                walk(FormulaRevision::Rev4, v1_only),
                &format!("Rev4 request, v1_only={v1_only}"),
            );
        }
        walk(FormulaRevision::Rev3, true).expect("the matching v1-only request serves");
        walk(FormulaRevision::Rev3, false).expect("the matching wide request serves");
        println!("{SENTINEL}");
    }

    /// Direction 1: a Rev1/Rev2/Rev3 request in a Rev4 process; the matching
    /// Rev4 request (the research walk) serves.
    #[test]
    fn the_walk_refuses_an_earlier_request_in_a_rev4_process() {
        const SENTINEL: &str = "WALK-EARLIER-IN-REV4-RAN";
        if !crate::ssim_form::run_at_revision(
            "4",
            "feature_v2::featcanon_contract_tests::the_walk_refuses_an_earlier_request_in_a_rev4_process",
            SENTINEL,
        ) {
            return;
        }
        for rev in [
            FormulaRevision::Rev1,
            FormulaRevision::Rev2,
            FormulaRevision::Rev3,
        ] {
            assert_mix_refused(walk(rev, true), &format!("{rev:?} v1-only request"));
        }
        walk(FormulaRevision::Rev4, true).expect("the matching v1-only request serves");
        walk(FormulaRevision::Rev4, false).expect("the matching wide request serves");
        println!("{SENTINEL}");
    }

    /// REV4VEC2 bit-exactness gate: `*_canon64_vec` tier variants must
    /// reproduce the scalar `LanesF64` canonical body **bit for bit** —
    /// every `DenseAccum`/`GradientAccum` field plus the Rev4 hook state
    /// (`TailAccum` hist/max, `FlatAccum`, `RingAccum` bins, `BleedAccum`,
    /// bank/chroma cells). Each compiled variant is invoked directly on
    /// its summoned token.
    mod rev4vec2 {
        use super::*;
        use crate::featcanon::{LanesF64, Mode, SumVar};
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

        fn sumvar_bits(v: &SumVar) -> Vec<u64> {
            vec![v.fin().to_bits()]
        }

        fn dense_bits(a: &DenseAccum) -> Vec<u64> {
            let mut v = vec![
                a.sum_d,
                a.sum_d2,
                a.sum_d3,
                a.sum_d4,
                a.sum_art,
                a.sum_det,
                a.sum_mse,
                a.sum_hf_gain,
                a.sum_hf_loss,
                a.sum_hf_mag_loss,
                a.sum_pjnd,
                a.sum_pjnd_lo,
                a.sum_pjnd_hi,
            ]
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<u64>>();
            for w in [
                &a.ws_peak_ssim,
                &a.ws_peak_art,
                &a.ws_peak_det,
                &a.ws_mask_ssim,
                &a.ws_mask_art,
                &a.ws_mask_det,
                &a.ws_mask_mse,
                &a.ws_iw_ssim,
                &a.ws_iw_art,
                &a.ws_iw_det,
                &a.ws_iw_mse,
            ] {
                v.push(w.num.to_bits());
                v.push(w.den.to_bits());
            }
            v
        }

        fn tail_bits(t: &TailAccum) -> Vec<u64> {
            let mut v = Vec::new();
            for m in 0..4 {
                v.extend(t.hist[m].iter().map(|&b| b as u64));
                v.push(t.max[m].to_bits());
            }
            v
        }

        fn flat_bits(f: &FlatAccum) -> Vec<u64> {
            let mut v = sumvar_bits(&f.hfg);
            v.push(f.n);
            v
        }

        fn ring_bits(r: &RingAccum) -> Vec<u64> {
            r.bins.iter().flat_map(sumvar_bits).collect()
        }

        fn bleed_bits(b: &BleedAccum) -> Vec<u64> {
            [sumvar_bits(&b.out_src), sumvar_bits(&b.out_dst)].concat()
        }

        fn grad_bits(a: &GradientAccum) -> Vec<u64> {
            let mut v = [
                a.sum_gms,
                a.sum_gms2,
                a.sum_ringing,
                a.sum_banding,
                a.sum_grad_src,
                a.sum_grad_dst,
                a.sum_bv_gain,
                a.sum_bv_loss,
            ]
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<u64>>();
            for c in a.bank.iter().chain(a.chroma.iter()) {
                v.push(c.loss.to_bits());
                v.push(c.gain.to_bits());
                v.push(c.n);
                v.push(c.mean.to_bits());
                v.push(c.m2.to_bits());
            }
            v
        }

        /// One dense geometry+flag case. `r4_kind`: 0=None, 1=tail+flat.
        fn run_dense(
            f: impl Fn(
                &[f32],
                &[f32],
                &[f32],
                &[f32],
                &[f32],
                &[f32],
                &[f32],
                usize,
                usize,
                bool,
                Option<Rev4Dense<'_>>,
            ) -> DenseAccum,
            planes: &[Vec<f32>; 6],
            w: usize,
            h: usize,
            bank: bool,
            r4_kind: usize,
        ) -> (DenseAccum, Option<(TailAccum, FlatAccum)>) {
            let mut cell = Rev4CellAccum::default();
            let r4 = if r4_kind == 1 {
                Some(Rev4Dense {
                    tail: Some(&mut cell.tail),
                    flat: Some(&mut cell.flat),
                    edges: tail_edges(),
                })
            } else {
                None
            };
            let acc = f(
                &planes[0], &planes[1], &planes[2], &planes[3], &planes[4], &planes[5], &planes[5],
                w, h, bank, r4,
            );
            let hooks = (r4_kind == 1).then_some((cell.tail, cell.flat));
            (acc, hooks)
        }

        #[test]
        fn dense_canon64_vec_matches_scalar_on_every_compiled_tier() {
            let mut rng = Rng(0x4D45_4E53_455F_4341);
            let cases: Vec<(usize, usize, bool, usize)> = vec![
                (8, 4, false, 0),
                (9, 5, true, 1),
                (17, 3, true, 1),
                (5, 33, false, 1),
                (16, 34, true, 0),
                (3, 65, true, 1),
                (33, 40, true, 1),
            ];
            let mut compiled = 0usize;
            macro_rules! check_variant {
                ($name:literal, $f:expr) => {{
                    compiled += 1;
                    for &(w, h, bank, r4_kind) in &cases {
                        let n = w * h;
                        let planes: [Vec<f32>; 6] =
                            std::array::from_fn(|_| (0..n).map(|_| rng.f32v()).collect());
                        let (want_acc, want_hooks) = run_dense(
                            |s, d, m1, m2, q, p, _act_dup, w, h, bank, r4| {
                                dense_block_kernel_canon::<LanesF64>(
                                    s,
                                    d,
                                    m1,
                                    m2,
                                    q,
                                    p,
                                    &planes[5],
                                    w,
                                    h,
                                    bank,
                                    r4,
                                    crate::ssim_form::active_revision(),
                                )
                            },
                            &planes,
                            w,
                            h,
                            bank,
                            r4_kind,
                        );
                        let (got_acc, got_hooks) = run_dense(&$f, &planes, w, h, bank, r4_kind);
                        assert_eq!(
                            dense_bits(&got_acc),
                            dense_bits(&want_acc),
                            "{}: accum differs (w={w} h={h} bank={bank} r4={r4_kind})",
                            $name
                        );
                        match (got_hooks, want_hooks) {
                            (Some((gt, gf)), Some((wt, wf))) => {
                                assert_eq!(
                                    tail_bits(&gt),
                                    tail_bits(&wt),
                                    "{}: tail differs",
                                    $name
                                );
                                assert_eq!(
                                    flat_bits(&gf),
                                    flat_bits(&wf),
                                    "{}: flat differs",
                                    $name
                                );
                            }
                            (None, None) => {}
                            _ => panic!("{}: hook presence mismatch", $name),
                        }
                    }
                }};
            }
            check_variant!(
                "scalar",
                |s: &[f32],
                 d: &[f32],
                 m1: &[f32],
                 m2: &[f32],
                 q: &[f32],
                 p: &[f32],
                 act: &[f32],
                 w: usize,
                 h: usize,
                 bank: bool,
                 r4: Option<Rev4Dense<'_>>| {
                    dense_block_kernel_canon64_vec_scalar::<crate::featcanon::LanesF64, false>(
                        archmage::ScalarToken::summon().expect("infallible"),
                        s,
                        d,
                        m1,
                        m2,
                        q,
                        p,
                        act,
                        w,
                        h,
                        bank,
                        r4,
                        crate::ssim_form::active_revision(),
                    )
                }
            );
            #[cfg(target_arch = "x86_64")]
            if let Some(t) = archmage::X64V3Token::summon() {
                check_variant!(
                    "v3",
                    move |s: &[f32],
                          d: &[f32],
                          m1: &[f32],
                          m2: &[f32],
                          q: &[f32],
                          p: &[f32],
                          act: &[f32],
                          w: usize,
                          h: usize,
                          bank: bool,
                          r4: Option<Rev4Dense<'_>>| {
                        dense_block_kernel_canon64_vec_v3::<crate::featcanon::LanesF64, false>(
                            t,
                            s,
                            d,
                            m1,
                            m2,
                            q,
                            p,
                            act,
                            w,
                            h,
                            bank,
                            r4,
                            crate::ssim_form::active_revision(),
                        )
                    }
                );
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            if let Some(t) = archmage::X64V4Token::summon() {
                check_variant!(
                    "v4",
                    move |s: &[f32],
                          d: &[f32],
                          m1: &[f32],
                          m2: &[f32],
                          q: &[f32],
                          p: &[f32],
                          act: &[f32],
                          w: usize,
                          h: usize,
                          bank: bool,
                          r4: Option<Rev4Dense<'_>>| {
                        dense_block_kernel_canon64_vec_v4::<crate::featcanon::LanesF64, false>(
                            t,
                            s,
                            d,
                            m1,
                            m2,
                            q,
                            p,
                            act,
                            w,
                            h,
                            bank,
                            r4,
                            crate::ssim_form::active_revision(),
                        )
                    }
                );
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            if let Some(t) = archmage::X64V4xToken::summon() {
                check_variant!(
                    "v4x",
                    move |s: &[f32],
                          d: &[f32],
                          m1: &[f32],
                          m2: &[f32],
                          q: &[f32],
                          p: &[f32],
                          act: &[f32],
                          w: usize,
                          h: usize,
                          bank: bool,
                          r4: Option<Rev4Dense<'_>>| {
                        dense_block_kernel_canon64_vec_v4x::<crate::featcanon::LanesF64, false>(
                            t,
                            s,
                            d,
                            m1,
                            m2,
                            q,
                            p,
                            act,
                            w,
                            h,
                            bank,
                            r4,
                            crate::ssim_form::active_revision(),
                        )
                    }
                );
            }
            assert!(compiled >= 1);
        }

        /// One gradient case: all six const combos, `r4_kind` 0=None,
        /// 1=ring+bleed+bank-with-chroma, 2=ring only.
        #[allow(clippy::too_many_arguments)]
        fn run_gradient<const B: bool, const V: bool, const K: bool>(
            f: impl Fn(
                &[f32],
                &[f32],
                &[f32],
                &[f32],
                usize,
                usize,
                f32,
                f32,
                Option<Rev4Grad<'_>>,
            ) -> GradientAccum,
            src_h: &[f32],
            dst_h: &[f32],
            act: &[f32],
            actd: &[f32],
            bleed_mask: &[f32],
            chroma_r: &[f32],
            chroma_d: &[f32],
            w: usize,
            h: usize,
            bv_lo: f32,
            bv_hi: f32,
            r4_kind: usize,
        ) -> (GradientAccum, Option<(RingAccum, BleedAccum)>) {
            let mut cell = Rev4CellAccum::default();
            let r4 = match r4_kind {
                0 => None,
                _ => Some(Rev4Grad {
                    ring: Some(&mut cell.ring),
                    bleed: if r4_kind == 1 {
                        Some((bleed_mask, &mut cell.bleed))
                    } else {
                        None
                    },
                    bank: if r4_kind == 1 {
                        GmsBankWork::new(
                            1,
                            Some(ChromaStrip {
                                reference: chroma_r,
                                distorted: chroma_d,
                            }),
                        )
                    } else {
                        GmsBankWork::new(0, None)
                    },
                }),
            };
            let acc = f(src_h, dst_h, act, actd, w, h, bv_lo, bv_hi, r4);
            let hooks = (r4_kind != 0).then_some((cell.ring, cell.bleed));
            (acc, hooks)
        }

        macro_rules! gradient_cases {
            ($name:literal, $f:ident, $tok:expr, $rng:expr, $compiled:expr) => {{
                $compiled += 1;
                // (w, h, r4_kind) — interior needs w>2; borders-only at
                // w≤2; chroma/bleed paths at r4=1.
                for &(w, h, r4_kind) in &[
                    (3usize, 4usize, 0usize),
                    (9, 5, 1),
                    (10, 3, 1),
                    (17, 6, 2),
                    (18, 4, 1),
                    (5, 9, 1),
                    (33, 5, 1),
                    (2, 6, 1),
                    (1, 3, 0),
                    (26, 7, 1),
                ] {
                    let n = w * h;
                    let nh = (h + 2) * w;
                    let src_h: Vec<f32> = (0..nh).map(|_| $rng.f32v()).collect();
                    let dst_h: Vec<f32> = (0..nh).map(|_| $rng.f32v()).collect();
                    let act: Vec<f32> = (0..n).map(|_| $rng.f32v()).collect();
                    let actd: Vec<f32> = (0..n).map(|_| $rng.f32v()).collect();
                    let bleed: Vec<f32> = (0..n).map(|_| $rng.f32v()).collect();
                    let cr: Vec<f32> = (0..n).map(|_| $rng.f32v()).collect();
                    let cd: Vec<f32> = (0..n).map(|_| $rng.f32v()).collect();
                    let mu1: Vec<f32> = (0..n).map(|_| $rng.f32v()).collect();
                    let mu2: Vec<f32> = (0..n).map(|_| $rng.f32v()).collect();
                    let (bv_lo, bv_hi) = (0.03f32, 0.21f32);

                    macro_rules! run_one {
                        ($want:ident, $got:ident, $B:literal, $V:literal, $K:literal, $R:literal) => {{
                            let ($want, want_hooks) = run_gradient::<$B, $V, $K>(
                                |s, d, a, ad, w, h, lo, hi, r4| {
                                    gradient_block_kernel_canon::<LanesF64, $B, $V, $K, $R, false>(
                                        s,
                                        d,
                                        a,
                                        ad,
                                        &mu1,
                                        &mu2,
                                        w,
                                        h,
                                        lo,
                                        hi,
                                        r4,
                                        Mode::Canon64,
                                    )
                                },
                                &src_h,
                                &dst_h,
                                &act,
                                &actd,
                                &bleed,
                                &cr,
                                &cd,
                                w,
                                h,
                                bv_lo,
                                bv_hi,
                                r4_kind,
                            );
                            let ($got, got_hooks) = run_gradient::<$B, $V, $K>(
                                |s, d, a, ad, w, h, lo, hi, r4| {
                                    $f::<$B, $V, $K, $R>($tok, s, d, a, ad, &mu1, &mu2, w, h, lo, hi, r4)
                                },
                                &src_h,
                                &dst_h,
                                &act,
                                &actd,
                                &bleed,
                                &cr,
                                &cd,
                                w,
                                h,
                                bv_lo,
                                bv_hi,
                                r4_kind,
                            );
                            assert_eq!(
                                grad_bits(&$got),
                                grad_bits(&$want),
                                "{}: accum differs (B={} V={} K={} R={} w={w} h={h} r4={r4_kind})",
                                $name,
                                $B,
                                $V,
                                $K,
                                $R
                            );
                            match (got_hooks, want_hooks) {
                                (Some((gr, gb)), Some((wr, wb))) => {
                                    assert_eq!(
                                        ring_bits(&gr),
                                        ring_bits(&wr),
                                        "{}: ring differs",
                                        $name
                                    );
                                    assert_eq!(
                                        bleed_bits(&gb),
                                        bleed_bits(&wb),
                                        "{}: bleed differs",
                                        $name
                                    );
                                }
                                (None, None) => {}
                                _ => panic!("{}: hook presence mismatch", $name),
                            }
                        }};
                    }
                    for b in [false, true] {
                        for v in [false, true] {
                            for k in [false, true] {
                                // Const-generic combos cannot be runtime
                                // booleans — dispatch on the 8 tuples.
                                match (b, v, k) {
                                    (false, false, false) => {
                                        run_one!(w0, g0, false, false, false, false);
                                        run_one!(w0, g0, false, false, false, true)
                                    }
                                    (false, false, true) => {
                                        run_one!(w1, g1, false, false, true, false);
                                        run_one!(w1, g1, false, false, true, true)
                                    }
                                    (false, true, false) => {
                                        run_one!(w2, g2, false, true, false, false);
                                        run_one!(w2, g2, false, true, false, true)
                                    }
                                    (false, true, true) => {
                                        run_one!(w3, g3, false, true, true, false);
                                        run_one!(w3, g3, false, true, true, true)
                                    }
                                    (true, false, false) => {
                                        run_one!(w4, g4, true, false, false, false);
                                        run_one!(w4, g4, true, false, false, true)
                                    }
                                    (true, false, true) => {
                                        run_one!(w5, g5, true, false, true, false);
                                        run_one!(w5, g5, true, false, true, true)
                                    }
                                    (true, true, false) => {
                                        run_one!(w6, g6, true, true, false, false);
                                        run_one!(w6, g6, true, true, false, true)
                                    }
                                    (true, true, true) => {
                                        run_one!(w7, g7, true, true, true, false);
                                        run_one!(w7, g7, true, true, true, true)
                                    }
                                }
                            }
                        }
                    }
                }
            }};
        }

        #[test]
        fn gradient_canon64_vec_matches_scalar_on_every_compiled_tier() {
            fn go_scalar<const B: bool, const V: bool, const K: bool, const R: bool>(
                _t: archmage::ScalarToken,
                s: &[f32],
                d: &[f32],
                a: &[f32],
                ad: &[f32],
                m1: &[f32],
                m2: &[f32],
                w: usize,
                h: usize,
                lo: f32,
                hi: f32,
                r4: Option<Rev4Grad<'_>>,
            ) -> GradientAccum {
                gradient_block_kernel_canon64_vec_scalar::<crate::featcanon::LanesF64, B, V, K, R, false>(
                    _t,
                    s,
                    d,
                    a,
                    ad,
                    m1,
                    m2,
                    w,
                    h,
                    lo,
                    hi,
                    r4,
                    Mode::Canon64,
                )
            }
            #[cfg(target_arch = "x86_64")]
            fn go_v3<const B: bool, const V: bool, const K: bool, const R: bool>(
                t: archmage::X64V3Token,
                s: &[f32],
                d: &[f32],
                a: &[f32],
                ad: &[f32],
                m1: &[f32],
                m2: &[f32],
                w: usize,
                h: usize,
                lo: f32,
                hi: f32,
                r4: Option<Rev4Grad<'_>>,
            ) -> GradientAccum {
                gradient_block_kernel_canon64_vec_v3::<crate::featcanon::LanesF64, B, V, K, R, false>(
                    t,
                    s,
                    d,
                    a,
                    ad,
                    m1,
                    m2,
                    w,
                    h,
                    lo,
                    hi,
                    r4,
                    Mode::Canon64,
                )
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            fn go_v4<const B: bool, const V: bool, const K: bool, const R: bool>(
                t: archmage::X64V4Token,
                s: &[f32],
                d: &[f32],
                a: &[f32],
                ad: &[f32],
                m1: &[f32],
                m2: &[f32],
                w: usize,
                h: usize,
                lo: f32,
                hi: f32,
                r4: Option<Rev4Grad<'_>>,
            ) -> GradientAccum {
                gradient_block_kernel_canon64_vec_v4::<crate::featcanon::LanesF64, B, V, K, R, false>(
                    t,
                    s,
                    d,
                    a,
                    ad,
                    m1,
                    m2,
                    w,
                    h,
                    lo,
                    hi,
                    r4,
                    Mode::Canon64,
                )
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            fn go_v4x<const B: bool, const V: bool, const K: bool, const R: bool>(
                t: archmage::X64V4xToken,
                s: &[f32],
                d: &[f32],
                a: &[f32],
                ad: &[f32],
                m1: &[f32],
                m2: &[f32],
                w: usize,
                h: usize,
                lo: f32,
                hi: f32,
                r4: Option<Rev4Grad<'_>>,
            ) -> GradientAccum {
                gradient_block_kernel_canon64_vec_v4x::<crate::featcanon::LanesF64, B, V, K, R, false>(
                    t,
                    s,
                    d,
                    a,
                    ad,
                    m1,
                    m2,
                    w,
                    h,
                    lo,
                    hi,
                    r4,
                    Mode::Canon64,
                )
            }
            let mut rng = Rng(0x6752_4144_5F43_4153);
            let mut compiled = 0usize;
            gradient_cases!(
                "scalar",
                go_scalar,
                archmage::ScalarToken::summon().expect("infallible"),
                rng,
                compiled
            );
            #[cfg(target_arch = "x86_64")]
            if let Some(t) = archmage::X64V3Token::summon() {
                gradient_cases!("v3", go_v3, t, rng, compiled);
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            if let Some(t) = archmage::X64V4Token::summon() {
                gradient_cases!("v4", go_v4, t, rng, compiled);
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            if let Some(t) = archmage::X64V4xToken::summon() {
                gradient_cases!("v4x", go_v4x, t, rng, compiled);
            }
            assert!(compiled >= 1);
        }
    }
}

#[cfg(all(test, feature = "threads"))]
#[path = "rev5_scaling_tests.rs"]
mod rev5_scaling_tests;
