// ============================================================================
// Phase-4 (§A.14): magetypes-SIMD formula pass — f32x8-lane vectorized
// counterparts of the four scalar formulas above. Same algebra, f32
// precision (matches the source planes' own f32 storage), 8 pixels/call
// instead of 1. Generic over any token implementing `F32x8Backend` so
// `#[magetypes(v4x, v4, v3, neon, wasm128)]` (see
// `compute_channel_scale_v2_kernel_entry` below) monomorphizes one body
// per tier — this is the SAME pattern as `~/work/archmage/docs/site/
// content/magetypes/examples/plane-ops.md`/`gaussian-blur.md`.
// ============================================================================

type V8<T> = GenericF32x8<T>;

/// The element ops the vector term helpers use, at either vector width.
/// magetypes' 8- and 16-lane generic types implement each with the same
/// per-lane semantics (the same `max` operand order, `GT_OQ`/`LT_OQ`
/// compares, mask select; v3's 16-lane type is two 8-lane halves), so a
/// helper computes the same bits per lane at either width.
trait Vf32:
    Copy
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
{
    type Tok: Copy;
    fn splat_t(token: Self::Tok, v: f32) -> Self;
    fn zero_t(token: Self::Tok) -> Self;
    fn max(self, other: Self) -> Self;
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn simd_gt(self, other: Self) -> Self;
    fn simd_lt(self, other: Self) -> Self;
    fn blend(mask: Self, if_true: Self, if_false: Self) -> Self;
}

macro_rules! impl_vf32 {
    ($ty:ident, $backend:ident) => {
        impl<T: $backend + Copy> Vf32 for $ty<T> {
            type Tok = T;
            #[inline(always)]
            fn splat_t(token: T, v: f32) -> Self {
                $ty::<T>::splat_t(token, v)
            }
            #[inline(always)]
            fn zero_t(token: T) -> Self {
                $ty::<T>::zero_t(token)
            }
            #[inline(always)]
            fn max(self, other: Self) -> Self {
                $ty::<T>::max(self, other)
            }
            #[inline(always)]
            fn abs(self) -> Self {
                $ty::<T>::abs(self)
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                $ty::<T>::sqrt(self)
            }
            #[inline(always)]
            fn simd_gt(self, other: Self) -> Self {
                $ty::<T>::simd_gt(self, other)
            }
            #[inline(always)]
            fn simd_lt(self, other: Self) -> Self {
                $ty::<T>::simd_lt(self, other)
            }
            #[inline(always)]
            fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
                $ty::<T>::blend(mask, if_true, if_false)
            }
        }
    };
}
impl_vf32!(GenericF32x8, F32x8Backend);
impl_vf32!(GenericF32x16, F32x16Backend);

/// Whether a tier's kernels step sixteen f32 columns at a time: true where
/// `T`'s target features include AVX-512F (v4, v4x), so one register holds
/// sixteen lanes. v3's sixteen-lane type is two 256-bit halves, and the wider
/// step measured slower there (B at 2048², revision 7: 80.4 → 84.9 ms).
pub(crate) const fn steps_sixteen<T: archmage::SimdToken>() -> bool {
    let f = T::TARGET_FEATURES.as_bytes();
    let pat = b"avx512f";
    let mut i = 0;
    while i + pat.len() <= f.len() {
        let mut j = 0;
        while j < pat.len() && f[i + j] == pat[j] {
            j += 1;
        }
        if j == pat.len() {
            return true;
        }
        i += 1;
    }
    false
}

// ============================================================================
// Phase-6 (§A.16): division-cost reduction for the SIMD formula family.
//
// SHIPPED: ALGEBRAIC SHARING ONLY (exact, no precision tradeoff beyond
// ordinary FP reassociation already tolerated by this file's 5e-4 policy):
// a chained `(a/b)*(c/d)` becomes one division `(a*c)/(b*d)`; two calls
// that share a denominator by construction (`bounded_excess(a,b,c)` +
// `bounded_excess(b,a,c)`) compute it once; a chained ratio-then-saturate
// (`saturate(x/(1+k*act), c)`) collapses to one division by multiplying
// through by the inner denominator. See `ssim_d_local`/
// `bounded_excess_pair`/`pjnd_transducer` (scalar siblings) for the exact
// derivations — same algebra here, f32 SIMD lanes. This drops the
// per-pixel division count from 13 to 8 (with `transducer_bank` on, the
// default) for FREE — native division stays correctly-rounded throughout,
// so no bounds-safety concern beyond what already existed pre-phase-6.
//
// TRIED AND DROPPED: reciprocal-estimate + Newton-Raphson (`V8::<T>::
// recip`, magetypes `archmage/magetypes/src/simd/generic/generated/
// f32x8_impl.rs:371` — every dispatched backend overrides it with a
// hardware estimate refined by one NR step, full f32 precision). Two
// findings, both measured on this box (Ryzen 9 7950X, whatever tier
// `incant!` picks):
//   1. A per-pixel `.min(...)` clamp after every recip-based division
//      (the naive "guard the ~24-bit estimate against overshooting a
//      documented bound" approach) caused a SEVERE regression: 1024²
//      1-thread went 147.0ms -> 738.2ms (~5x worse). `cargo asm` showed
//      `dense_block_kernel_generic`'s AVX2 monomorphization bloated to
//      4268 instructions / 186 un-inlined `call`s — even `_mm256_mul_ps`/
//      `_mm256_add_ps` stopped inlining, 2112-byte stack frame. The extra
//      splat+min at every division site pushed the function past LLVM's
//      inliner threshold and it gave up wholesale.
//   2. Removing the per-pixel clamps (relying on the mean/weighted-mean
//      accumulation structure to keep any epsilon overshoot from
//      amplifying, plus one cheap clamp on the FINAL feature value)
//      recovered MOST but not all of the regression (524.9ms). The
//      remaining gap was `#[inline]` being too weak a hint once these
//      functions grew larger from the algebra fusion — switching to
//      `#[inline(always)]` (KEPT, see below) fixed it completely: 147.2ms,
//      matching plain-division's 144.3ms within noise (both single
//      4-round 1024² measurements, ~5ms MAD). Precision was never the
//      problem — measured max deviation from exact division was 3.246e-7
//      on real AVX2 hardware (`_mm256_rcp_ps` + 1 NR step), five orders of
//      magnitude under the smoke test's 1e-6 slack margin.
//   3. CONCLUSION: recip() is numerically safe and, once the inlining bug
//      is fixed, performance-NEUTRAL on this hardware — not a regression,
//      but not a measurable win either (147.2ms vs 144.3ms, within noise).
//      AMD Zen4's AVX2 divider unit is evidently fast enough that trading
//      1 division for ~1 estimate + 2-3 dependent multiply/subtract ops
//      does not pay for itself. Not shipped: it adds real complexity (the
//      bounds-safety story, an extra formula-precision test) for zero
//      measured benefit on the hardware this was tested on. Kept as a
//      documented, reproducible negative result rather than silently
//      dropped — a future session on different hardware (older x86 with
//      slower `vdivps`, or a tier where `rcp_approx` is a bigger win vs.
//      that tier's native divide) could reopen this with the inlining
//      fix already known. `#[inline(always)]` is KEPT on these six
//      functions regardless of the recip question — it is what actually
//      fixed the regression and is cheap, harmless insurance against the
//      same bloat recurring as the algebra-fusion body grew larger than
//      the pre-phase-6 originals.
// ============================================================================

/// Vectorized [`ssim_d_local`] — fused to one division via `(a*c)/(b*d)`.
/// Identical algebra, f32 lanes. Bounded `[0, 2]`.
#[inline(always)]
fn ssim_d_local_v<V: Vf32>(
    token: V::Tok,
    mu1: V,
    mu2: V,
    s12: V,
    ssq: V,
    c1: V,
    c2: V,
    // Revision 3: `s12` carries the direct error moment; see `ssim_d_local`.
    direct: bool,
    // Revision 7: the luminance term without cancellation; see `ssim_d_local`.
    stable: bool,
) -> V {
    let two = V::splat_t(token, 2.0);
    let one = V::splat_t(token, 1.0);
    let zero = V::zero_t(token);
    let a = two * mu1 * mu2 + c1;
    let b = mu1 * mu1 + mu2 * mu2 + c1;
    let d = ssq - mu1 * mu1 - mu2 * mu2 + c2;
    if direct {
        let mu_diff = mu1 - mu2;
        let err_var = (s12 - mu_diff * mu_diff).max(zero);
        let lum = if stable { mu_diff * mu_diff } else { b - a };
        ((d * lum + a * err_var) / (b * d)).max(zero)
    } else {
        let cov = s12 - mu1 * mu2;
        let c = two * cov + c2;
        let local = (a * c) / (b * d);
        (one - local).max(zero)
    }
}

/// Vectorized [`bounded_sim`] — `(2ab+c)/(a²+b²+c)`. Bounded `(0, 1]`.
#[inline(always)]
fn bounded_sim_v<V: Vf32>(token: V::Tok, a: V, b: V, c: V) -> V {
    let two = V::splat_t(token, 2.0);
    (two * a * b + c) / (a * a + b * b + c)
}

/// Vectorized [`bounded_dissim`]: `one − bounded_sim_v`, or the exact
/// `(a−b)²/(a²+b²+c)` when `stable` (revision 7).
#[inline(always)]
fn bounded_dissim_v<V: Vf32>(
    token: V::Tok,
    a: V,
    b: V,
    c: V,
    stable: bool,
) -> V {
    if stable {
        let d = a - b;
        d * d / (a * a + b * b + c)
    } else {
        V::splat_t(token, 1.0) - bounded_sim_v(token, a, b, c)
    }
}

/// Vectorized [`bounded_excess`] — `max(0, a-b) / (a+b+c)`. Standalone (no
/// shared-denominator partner — see [`bounded_excess_pair_v`] for the
/// paired form). Bounded `[0, 1)`.
#[inline(always)]
fn bounded_excess_v<V: Vf32>(token: V::Tok, a: V, b: V, c: V) -> V {
    let zero = V::zero_t(token);
    (a - b).max(zero) / (a + b + c)
}

/// Vectorized [`bounded_excess_pair`] — `bounded_excess(a,b,c)` PAIRED with
/// `bounded_excess(b,a,c)`, one shared division instead of two (phase-6
/// §A.16: the two calls' denominators are the SAME value by commutativity
/// of addition). Returns `(bounded_excess(a,b,c), bounded_excess(b,a,c))`.
/// Bounded `[0, 1)` each.
#[inline(always)]
fn bounded_excess_pair_v<V: Vf32>(
    token: V::Tok,
    a: V,
    b: V,
    c: V,
) -> (V, V) {
    let zero = V::zero_t(token);
    let denom = a + b + c;
    ((a - b).max(zero) / denom, (b - a).max(zero) / denom)
}

/// [`bounded_excess_pair_v`] for non-negative `a` and `b` (squares), with one
/// division. At most one of `(a − b).max(0)` and `(b − a).max(0)` is nonzero,
/// and `b − a` is exactly `−(a − b)`, so `|a − b| / (a + b + c)` selected by
/// `a > b` (or `b > a`) is that quotient's bits; the other side is the `+0`
/// that `0 / (a + b + c)` gives for the positive denominator.
#[inline(always)]
fn bounded_excess_split_v<V: Vf32>(token: V::Tok, a: V, b: V, c: V) -> (V, V) {
    let zero = V::zero_t(token);
    let q = (a - b).abs() / (a + b + c);
    (V::blend(a.simd_gt(b), q, zero), V::blend(b.simd_gt(a), q, zero))
}

/// Vectorized [`saturate`] — `max(x,0)/(max(x,0)+c)`. Bounded `[0, 1)`.
#[inline(always)]
fn saturate_v<V: Vf32>(token: V::Tok, x: V, c: V) -> V {
    let zero = V::zero_t(token);
    let x = x.max(zero);
    x / (x + c)
}

/// Vectorized MSCN divisive normalizer — `resid / sqrt(var + c)`.
///
/// Deliberately IEEE `sqrt` + `div` (both correctly rounded on every vendor
/// and every tier), NOT `resid * (var + c).rsqrt()`. Under the pinned
/// magetypes 0.9.28 contract `rsqrt()` is the hardware estimate
/// (`vrsqrtps`, whose seed table is CPU-VENDOR-specific) plus one
/// Newton-Raphson step, which leaves a ~1e-8 rel vendor residue — the
/// AMD-vs-Intel divergence measured on exactly the `MSCN_DIFF_MEAN` /
/// `MSCN_DIFF_L2` append slots in the bf944 wave (imazen/zensim#56,
/// `benchmarks/backfill944_bigcodec_2026-08-02.md`); on NEON it is
/// `1/sqrt` then a multiply (double rounding), a third distinct result. The
/// scalar tail, the f64 reference in `attr_pass_b_main_px`, and the
/// attribution pass-B SIMD kernel all use this exact single-rounding form
/// already; this makes the production kernel match. Gated bit-exact against
/// the scalar IEEE result on every tier by
/// `mscn_norm_v_is_correctly_rounded_on_every_tier`.
#[inline(always)]
fn mscn_norm_v<V: Vf32>(_token: V::Tok, resid: V, var: V, c: V) -> V {
    resid / (var + c).sqrt()
}

/// Vectorized [`pjnd_transducer`] — fused `saturate(raw_abs_err/(1+k*act),
/// c)` collapsed to one division (phase-6 §A.16, exact — see the scalar
/// sibling's doc for the derivation). Bounded `[0, 1)`.
#[inline(always)]
fn pjnd_transducer_v<V: Vf32>(
    token: V::Tok,
    raw_abs_err: V,
    act: V,
    k: V,
    c: V,
) -> V {
    let one = V::splat_t(token, 1.0);
    let denom = raw_abs_err + c * (one + k * act);
    raw_abs_err / denom
}

/// Terriberry (2007) single-pass online moments: tracks `n`, `mean`, and
/// the 2nd/3rd/4th central-moment sums `M2/M3/M4 = Σ(x-mean)^k`. `M3` is
/// tracked only because `M4`'s incremental update formula depends on its
/// PRE-update value — not used as an output here. This is what made
/// `ssim_dev2`/`ssim_dev4` (the D8 GMSD-style deviation moments) genuinely
/// single-pass and O(1)-space in iteration 2, unlike iteration 1's
/// store-then-revisit design.
///
/// Phase-4 (§A.14): no longer on the hot path — `compute_channel_scale_v2`
/// now computes `ssim_dev2`/`ssim_dev4` from SIMD-accumulated RAW power
/// sums (Σd, Σd², Σd³, Σd⁴; see `dense_block_kernel_generic`'s doc), since
/// Terriberry's update is inherently sequential and doesn't vectorize
/// across lanes. Kept as the reference implementation and cross-checked
/// against the raw-moment path in
/// `tests::raw_moment_reformulation_matches_terriberry` — this is the
/// proof that the phase-4 reformulation is numerically equivalent, not
/// just "removed and hoped for the best".
#[derive(Default, Clone, Copy)]
#[cfg_attr(not(test), allow(dead_code))] // test-referenced reference impl (raw_moment_reformulation_matches_terriberry)
pub(crate) struct OnlineMoments {
    n: f64,
    mean: f64,
    m2: f64,
    m3: f64,
    m4: f64,
}

#[cfg_attr(not(test), allow(dead_code))] // see struct note: exercised by the Terriberry parity test
impl OnlineMoments {
    /// Two-pass moments within a small block; no subtraction of raw powers.
    #[inline(never)]
    fn block(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        let n = values.len() as f64;
        let origin = values[0];
        let mean = origin + values.iter().map(|v| v - origin).sum::<f64>() / n;
        let mut out = Self {
            n,
            mean,
            ..Self::default()
        };
        for &v in values {
            let d = v - mean;
            let d2 = d * d;
            out.m2 += d2;
            out.m3 += d2 * d;
            out.m4 += d2 * d2;
        }
        out
    }

    /// Chan/Pébay merge, always called in raster block/strip order.
    fn merge(&mut self, b: Self) {
        if b.n == 0.0 {
            return;
        }
        if self.n == 0.0 {
            *self = b;
            return;
        }
        let a = *self;
        let n = a.n + b.n;
        let d = b.mean - a.mean;
        let d2 = d * d;
        let d3 = d2 * d;
        let d4 = d2 * d2;
        self.mean = a.mean + d * (b.n / n);
        self.m4 = a.m4
            + b.m4
            + d4 * a.n * b.n * (a.n * a.n - a.n * b.n + b.n * b.n) / (n * n * n)
            + 6.0 * d2 * (a.n * a.n * b.m2 + b.n * b.n * a.m2) / (n * n)
            + 4.0 * d * (a.n * b.m3 - b.n * a.m3) / n;
        self.m3 = a.m3
            + b.m3
            + d3 * a.n * b.n * (a.n - b.n) / (n * n)
            + 3.0 * d * (a.n * b.m2 - b.n * a.m2) / n;
        self.m2 = a.m2 + b.m2 + d2 * a.n * b.n / n;
        self.n = n;
    }

    #[inline]
    fn update(&mut self, x: f64) {
        let n1 = self.n;
        self.n += 1.0;
        let delta = x - self.mean;
        let delta_n = delta / self.n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * n1;
        self.mean += delta_n;
        self.m4 += term1 * delta_n2 * (self.n * self.n - 3.0 * self.n + 3.0)
            + 6.0 * delta_n2 * self.m2
            - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (self.n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;
    }

    /// `(mean, dev2, dev4)` — `dev2 = sqrt(M2/n)`, `dev4 = (M4/n)^0.25`,
    /// the GMSD-style deviation-from-mean moments (D8 fix).
    #[inline]
    fn finish(&self) -> (f64, f64, f64) {
        if self.n < 1.0 {
            return (0.0, 0.0, 0.0);
        }
        let dev2 = (self.m2 / self.n).max(0.0).sqrt();
        let dev4 = (self.m4 / self.n)
            .max(0.0)
            .quarter_root(crate::det_math::active_root_form());
        (self.mean, dev2, dev4)
    }
}

/// Revision 7's per-row central moments of `d` (the `rowlanes` era): a
/// sixteen-lane f64 sum (lane `x & 15`, ascending `x`, closed by the
/// [`crate::featcanon::Lanes16F64`] tree) gives the mean, then the centred
/// `(d−μ)²`, `(d−μ)³`, `(d−μ)⁴` sums in sixteen f64 lanes the same way. The
/// canonical scalar form; the vector dense kernel runs the same ops per lane.
pub(crate) fn row_moments_canon(v: &[f64]) -> OnlineMoments {
    use crate::featcanon::{Lanes16F64, Pool as _};
    let n = v.len();
    if n == 0 {
        return OnlineMoments::default();
    }
    let mut s1 = [0.0f64; 16];
    for (x, &d) in v.iter().enumerate() {
        s1[x & 15] += d;
    }
    let mean = Lanes16F64(s1).fin() / n as f64;
    let (mut s2, mut s3, mut s4) = ([0.0f64; 16], [0.0f64; 16], [0.0f64; 16]);
    for (x, &v) in v.iter().enumerate() {
        let d = v - mean;
        let q = d * d;
        s2[x & 15] += q;
        s3[x & 15] += q * d;
        s4[x & 15] += q * q;
    }
    OnlineMoments {
        n: n as f64,
        mean,
        m2: Lanes16F64(s2).fin(),
        m3: Lanes16F64(s3).fin(),
        m4: Lanes16F64(s4).fin(),
    }
}

/// Running `(Σw·v, Σw)` accumulator for the canonical weighted-mean
/// pooling form — the same `Σw·v/Σw` shape as
/// [`crate::iw_pool::WeightedPool::mean`] (iteration 1's promoted
/// canonical pooling helper), computed incrementally instead of from
/// materialized weight/value arrays.
///
/// **This is a gated mirror, not a duplicate**, per this crate's
/// no-duplication policy ("a second implementation is legitimate only
/// when it exists for a measured engineering reason AND a test holds it
/// bit-exact against the owner"): iteration 1's array-based
/// `WeightedPool::mean` was measured (`docs/FEATURE_V2_SPEC_2026-07-18.md`
/// §A.12) to make v2 2-5x SLOWER than v1 — materializing nine full-image
/// `Vec<f32>` maps per channel-scale to feed `WeightedPool::mean` is
/// exactly the cost iteration 2 removes. `WeightedSum` computes the
/// IDENTICAL formula (`Σw·v/Σw`) online, O(1) space. The equivalence is
/// pinned by `tests::weighted_sum_matches_weighted_pool_mean_exactly`.
#[derive(Default, Clone, Copy)]
pub(crate) struct WeightedSum {
    pub(crate) num: f64,
    pub(crate) den: f64,
}
impl WeightedSum {
    #[inline]
    fn add(&mut self, weight: f64, value: f64) {
        self.num += weight * value;
        self.den += weight;
    }
    #[inline]
    fn finish(&self) -> f64 {
        if self.den < 1e-12 {
            0.0
        } else {
            self.num / self.den
        }
    }
    /// Phase-5 (§A.15): fold another (partial-strip) accumulator's
    /// `(num, den)` in directly — `Σw·v` and `Σw` are both plain sums, so
    /// combining N strips' partial `WeightedSum`s is exactly equivalent to
    /// one whole-image `WeightedSum` (no reassociation beyond ordinary
    /// floating-point addition, same class of drift as any other
    /// SIMD-lane-then-strip reduction already tolerated in this file).
    #[inline]
    fn accumulate(&mut self, other: &WeightedSum) {
        self.num += other.num;
        self.den += other.den;
    }
}

// ============================================================================
// FeatureRegime + result + explicit-regime view
// ============================================================================

/// Explicit feature-extraction regime tag. See
/// `docs/FEATURE_V2_SPEC_2026-07-18.md` §(d): disambiguation is primarily
/// by distinct Rust type ([`ZensimV2Result`]/[`FeatureViewV2`] vs v1's
/// `ZensimResult`/`FeatureView`), this enum is the belt-and-suspenders
/// runtime tag for generic/dynamic call sites.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureRegime {
    /// The frozen, byte-stable v1 372-feature extraction (`metric.rs`).
    V1,
    /// The v2 "bounded" extraction this module implements.
    V2Bounded,
    /// The folded 720-layout extraction (2026-07-24): ONE v2-walk pass
    /// emitting `[f0..156) = v1 basic` (computed inside the v2 strip walk
    /// via the same `fused_blur_h_ssim` H-planes + v1's own
    /// `fused_vblur_features_ssim` kernel — parity-gated against the v1
    /// path, NOT byte-frozen), `[156..372) = 0.0` (v1's peak/masked/IW
    /// pool blocks, deprecated: no current model reads them — see
    /// `benchmarks/corruption_head_2026-07-24.md` "the f156..371 block can
    /// be dropped entirely"), `[372..720) = v2-348`. Index-stable
    /// emit-0, per the append-only feature-numbering rule.
    Folded720,
    /// [`Folded720`](Self::Folded720) plus the f720+ append block
    /// (gap-audit additions, 2026-07-26): `[720..924) = append-204`
    /// (17/ch/scale, see [`idx_append`]). The first 720 slots are
    /// bit-identical to a [`Folded720`](Self::Folded720) extraction of
    /// the same pair (`append_first720_bit_stable` gates this).
    Folded720Append,
    /// [`Folded720Append`](Self::Folded720Append) plus the f924+ append2
    /// block ([`idx_append2`]; 924 + 4×[`APPEND2_PER_SCALE`] = 944 slots).
    /// Additive-only and default-OFF — new-regime rows only (the HDR
    /// backfill wave); never mixed into 924-regime tables.
    Folded720Append2,
    /// [`Folded720Append2`](Self::Folded720Append2) plus the f944+ CSFW
    /// block ([`idx_csfw`]; 944 + 4×[`CSFW_PER_SCALE`] = 956 slots —
    /// chunk-3 tier-1, Y-only luminance-CSF-weighted GLOBAL_* twins).
    /// Additive-only and default-OFF — new-regime rows only (the HDR
    /// backfill wave); never mixed into 944- or 924-regime tables. The
    /// chroma tiers' old f956..f979 claim was released (CSF design doc);
    Folded720Csfw,
    /// feature-bank families (f986..f1321: `gridblk` 96, `ringbasis` 72,
    /// `tailhist` 144, `arttype` 24 — see `docs/REV4_FEATURE_BANK_PLAN_*
    /// .md`). Assigned only when ALL FOUR rev4 layout flags are on
    /// (the full 1322 row, also used by the additive C8 1502 row); a rev4-subset request keeps the regime of
    /// its widest fully-registered block — the slot provenance still
    /// disambiguates exactly which tail cells are populated.
    Folded720Rev4,
}

/// Result of [`compute_v2_features_impl`]/[`crate::Zensim::compute_v2_features`].
#[derive(Debug, Clone)]
pub struct ZensimV2Result {
    features: Vec<f64>,
    n_scales: usize,
    regime: FeatureRegime,
    /// `V2NewFeatureToggles::v1_pools`: which of `f156..372` carry v1's
    /// live pool values (a distinct extraction regime per mode at the SAME
    /// layout width — see the toggle's doc).
    v1_pools: V1PoolsMode,
}

impl ZensimV2Result {
    pub fn features(&self) -> &[f64] {
        &self.features
    }
    pub fn into_features(self) -> Vec<f64> {
        self.features
    }
    pub fn n_scales(&self) -> usize {
        self.n_scales
    }
    pub fn regime(&self) -> FeatureRegime {
        self.regime
    }
    /// Which of `f156..372` hold v1's LIVE pool values (the
    /// [`V2NewFeatureToggles::v1_pools`] regimes) rather than the folded
    /// walk's structural zeros. Same layout width either way — this is the
    /// regime-purity marker for that block.
    pub fn v1_pools(&self) -> V1PoolsMode {
        self.v1_pools
    }
    /// `v1_pools() != Off`.
    pub fn v1_pools_live(&self) -> bool {
        self.v1_pools != V1PoolsMode::Off
    }
    /// Explicit-regime named view over this result's features.
    ///
    /// For [`FeatureRegime::Folded720`] results the view covers the v2
    /// block (the vector's tail); the v1 basic block has no v2-named
    /// accessors and is read positionally (`features()[0..156]`).
    pub fn view(&self) -> FeatureViewV2<'_> {
        let v2_len = self.n_scales * 3 * FEATURES_PER_CHANNEL_V2_TOTAL;
        let v2_block = match self.regime {
            FeatureRegime::Folded720 => &self.features[self.features.len() - v2_len..],
            FeatureRegime::Folded720Append => {
                let append_len = self.n_scales * 3 * FEATURES_PER_CHANNEL_APPEND;
                let end = self.features.len() - append_len;
                &self.features[end - v2_len..end]
            }
            FeatureRegime::Folded720Append2 => {
                let tail = self.n_scales * (3 * FEATURES_PER_CHANNEL_APPEND + APPEND2_PER_SCALE);
                let end = self.features.len() - tail;
                &self.features[end - v2_len..end]
            }
            FeatureRegime::Folded720Csfw => {
                let tail = self.n_scales
                    * (3 * FEATURES_PER_CHANNEL_APPEND + APPEND2_PER_SCALE + CSFW_PER_SCALE);
                let end = self.features.len() - tail;
                &self.features[end - v2_len..end]
            }
            // Both 1322 and 1502 retain the same fixed v2 block at f372.
            FeatureRegime::Folded720Rev4 => &self.features[372..372 + v2_len],
            _ => &self.features[..],
        };
        FeatureViewV2::new(v2_block, self.n_scales)
            .expect("compute_v2_features always emits the v2-total layout for its own n_scales")
    }

    /// The f720+ append block (`n_scales × 3 × 17` slots, layout
    /// `scale*(3*17) + ch*17 + local` with [`idx_append`] locals), or
    /// `None` for regimes without one.
    pub fn append_features(&self) -> Option<&[f64]> {
        match self.regime {
            FeatureRegime::Folded720Append => {
                let append_len = self.n_scales * 3 * FEATURES_PER_CHANNEL_APPEND;
                Some(&self.features[self.features.len() - append_len..])
            }
            FeatureRegime::Folded720Append2 => {
                let append_len = self.n_scales * 3 * FEATURES_PER_CHANNEL_APPEND;
                let start = self.features.len() - append_len - self.n_scales * APPEND2_PER_SCALE;
                Some(&self.features[start..start + append_len])
            }
            FeatureRegime::Folded720Csfw => {
                let append_len = self.n_scales * 3 * FEATURES_PER_CHANNEL_APPEND;
                let start = self.features.len()
                    - append_len
                    - self.n_scales * (APPEND2_PER_SCALE + CSFW_PER_SCALE);
                Some(&self.features[start..start + append_len])
            }
            _ => None,
        }
    }

    /// The f924+ append2 slots (`n_scales × APPEND2_PER_SCALE`), when the
    /// result carries them ([`FeatureRegime::Folded720Append2`],
    /// [`FeatureRegime::Folded720Csfw`], or
    pub fn append2_features(&self) -> Option<&[f64]> {
        match self.regime {
            FeatureRegime::Folded720Append2 => {
                let len = self.n_scales * APPEND2_PER_SCALE;
                Some(&self.features[self.features.len() - len..])
            }
            FeatureRegime::Folded720Csfw => {
                let len = self.n_scales * APPEND2_PER_SCALE;
                let start = self.features.len() - len - self.n_scales * CSFW_PER_SCALE;
                Some(&self.features[start..start + len])
            }
            _ => None,
        }
    }

    /// The f944+ CSFW slots (`n_scales × CSFW_PER_SCALE`), when the
    /// result carries them ([`FeatureRegime::Folded720Csfw`] or
    pub fn csfw_features(&self) -> Option<&[f64]> {
        match self.regime {
            FeatureRegime::Folded720Csfw => {
                let len = self.n_scales * CSFW_PER_SCALE;
                Some(&self.features[self.features.len() - len..])
            }
            _ => None,
        }
    }
}

/// Named, explicit-regime view over a v2 feature vector. Unlike v1's
/// [`crate::FeatureView`], [`FeatureViewV2::new`] VALIDATES the exact
/// expected length rather than guessing a tier from an ambiguous length.
#[derive(Debug, Clone, Copy)]
pub struct FeatureViewV2<'a> {
    features: &'a [f64],
    n_scales: usize,
}

impl<'a> FeatureViewV2<'a> {
    pub fn new(features: &'a [f64], n_scales: usize) -> Option<Self> {
        let expected = n_scales * 3 * FEATURES_PER_CHANNEL_V2_TOTAL;
        if features.len() != expected {
            return None;
        }
        Some(Self { features, n_scales })
    }

    pub fn n_scales(&self) -> usize {
        self.n_scales
    }

    #[inline]
    fn at(&self, scale: usize, ch: usize, local: usize) -> f64 {
        self.features
            [scale * 3 * FEATURES_PER_CHANNEL_V2_TOTAL + ch * FEATURES_PER_CHANNEL_V2_TOTAL + local]
    }

    pub fn ssim_mean(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::SSIM_MEAN)
    }
    pub fn ssim_dev2(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::SSIM_DEV2)
    }
    pub fn ssim_dev4(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::SSIM_DEV4)
    }
    pub fn art(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::ART)
    }
    pub fn det(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::DET)
    }
    pub fn mse(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::MSE)
    }
    pub fn hf_gain(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::HF_GAIN)
    }
    pub fn hf_loss(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::HF_LOSS)
    }
    pub fn hf_mag_loss(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::HF_MAG_LOSS)
    }
    pub fn ssim_soft_peak(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::SSIM_SOFT_PEAK)
    }
    pub fn art_soft_peak(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::ART_SOFT_PEAK)
    }
    pub fn det_soft_peak(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::DET_SOFT_PEAK)
    }
    pub fn masked_ssim(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::MASKED_SSIM)
    }
    pub fn masked_art(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::MASKED_ART)
    }
    pub fn masked_det(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::MASKED_DET)
    }
    pub fn masked_mse(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::MASKED_MSE)
    }
    pub fn iw_ssim(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::IW_SSIM)
    }
    pub fn iw_art(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::IW_ART)
    }
    pub fn iw_det(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::IW_DET)
    }
    pub fn iw_mse(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::IW_MSE)
    }
    pub fn pjnd_transducer(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::PJND_TRANSDUCER)
    }
    pub fn pjnd_fragility(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::PJND_FRAGILITY)
    }
    /// Gradient-magnitude similarity (candidate 1). On `ch=0` (X) or
    /// `ch=2` (B) this doubles as chroma-edge similarity (candidate 3).
    pub fn gms(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::GMS)
    }
    pub fn pjnd_transducer_low_k(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::PJND_TRANSDUCER_LOW_K)
    }
    pub fn pjnd_transducer_high_k(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::PJND_TRANSDUCER_HIGH_K)
    }
    pub fn blockiness(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::BLOCKINESS)
    }
    pub fn ringing(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::RINGING)
    }
    pub fn banding(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::BANDING)
    }
    pub fn edge_width_change(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::EDGE_WIDTH_CHANGE)
    }
}

// ============================================================================
// Per-channel-per-scale computation — SINGLE PASS, O(1) accumulators
// ============================================================================

/// How a declared-HDR source's pixel values map to absolute display light
/// (cd/m²) on the folded/append STREAMING HDR route (HDR_PLAN chunk 2).
/// The declaration is explicit at the entry — the route never guesses a
/// transfer from pixel bytes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HdrEncoding {
    /// Pixels are already absolute linear light in cd/m² (the
    /// [`crate::Zensim::compute_pu_linear`] contract; e.g. decoded EXR).
    Linear,
    /// Pixels are PQ (SMPTE ST 2084) code values in `[0, 1]`, decoded per
    /// channel through the display model (`min(EOTF_PQ, peak) + black +
    /// reflection`). `peak_nits = 10000.0` decodes at spec peak with no
    /// display clamp; `1000.0` matches pycvvdp's `standard_hdr_pq`.
    Pq {
        /// Display peak luminance, cd/m².
        peak_nits: f32,
    },
    /// Pixels are HLG (ITU-R BT.2100) signal values in `[0, 1]`, decoded
    /// through the reference OOTF (`F_D = peak·Y_s^(γ−1)·E_s`) with the
    /// BT.2100 system gamma for `peak_nits`/`ambient_lux`.
    Hlg {
        /// Display peak luminance, cd/m².
        peak_nits: f32,
        /// Ambient illuminance, lux (5.0 = reference environment).
        ambient_lux: f32,
    },
}

/// Which FORMULA REVISION an extraction computes — re-exported so
/// [`V2NewFeatureToggles::formula_revision`] is nameable by the callers that
/// set it.
///
/// `#[doc(hidden)]`: the revision axis is research/extraction machinery, not
/// product surface. A default build's `Zensim::compute` resolves it per bake
/// and never asks a caller.
#[doc(hidden)]
pub use crate::feature_defs::FormulaRevision;
/// Runtime toggles for the phase-2 new-feature GROUPS (A.10 candidates),
/// grouped by shared per-pixel computation rather than 1:1 with feature
/// slots (per `docs/FEATURE_V2_SPEC_2026-07-18.md` §A.12's per-group
/// marginal-cost measurement):
///
/// - `gradient_features`: GMS + chroma-edge-GMS (free via channel axis) +
///   ringing + banding + edge-width-change. These all consume the same
///   `sqrt`-based gradient magnitude, so they are gated together — turning
///   this off skips the `sqrt` calls entirely, not just the 4 accumulator
///   adds.
/// - `transducer_bank`: the 2 EXTRA masking-transducer k values (the core
///   `k=4.0` transducer, `PJND_TRANSDUCER`, is always computed — it is not
///   a phase-2 addition, iteration 1 already shipped it).
/// - `blockiness`: oriented blockiness (shares neighbor loads with the
///   gradient group but not the `sqrt`, so toggled independently).
///
/// Default: all `true` (every candidate on). A group that fails the speed
/// gate is set `false` here — NOT deleted — per the phase-2 brief ("If a
/// feature blows the budget, keep it OFF by default and say so").
#[derive(Debug, Clone, Copy)]
pub struct V2NewFeatureToggles {
    pub gradient_features: bool,
    pub transducer_bank: bool,
    pub blockiness: bool,
    /// Emit the masking transducers (core + fragility + bank) on the LUMA
    /// (Y) channel only; zero them on X/B. The 2026-07-19 ablation +
    /// combined-model steering analysis found the chroma transducers cost
    /// CID22 and carry near-zero steering mass in a combined model (masking
    /// is fundamentally a luma phenomenon), while the Y transducer carries
    /// the CSIQ signal. Index-stable: the X/B slots stay at their indices,
    /// emitted as 0 (deprecate-by-mask, never renumber). Default OFF — this
    /// is a compression↔general-FR trade (CID22 up, CSIQ down), so it is
    /// opt-in until the ship recipe fixes the operating point.
    /// See `benchmarks/v2_trainability_ab_2026-07-19.md`.
    pub transducers_luma_only: bool,
    /// Emit the f720+ append block (17/ch/scale — see [`idx_append`] and
    /// the module-level "f720+ APPEND block" section). Default OFF: with
    /// this false every existing path, layout, and byte is unchanged.
    /// When on, the feature vector grows by `n_scales × 3 × 17` slots
    /// appended AFTER the existing layout, and the per-strip append
    /// kernel + the `d2` (blur(dst²)) σ-split chain run.
    pub append_block: bool,
    /// Emit the f924+ append2 block ([`idx_append2`]: BANDVIS gain/loss,
    /// the free luminance conditioner, HDR-gated highlight bins —
    /// `APPEND2_PER_SCALE` Y-only slots per scale). Default OFF: with
    /// this false every existing path, layout, and byte — SDR and HDR
    /// routes — is unchanged. Requires `append_block` (asserted): the
    /// block sits at f924+ and reuses append accumulators. Additive-only;
    /// joins the NEXT extraction regime wave (the HDR backfill).
    pub append2_block: bool,
    /// Emit the f944+ CSFW block ([`idx_csfw`]: chunk-3 tier-1 —
    /// luminance-CSF-weighted GLOBAL_* pooling twins, `CSFW_PER_SCALE`
    /// Y-only slots per scale, both routes with route-local derived φ
    /// constants). Default OFF: with this false the CSFW pass never runs
    /// and every existing path, layout, and byte — SDR and HDR routes —
    /// is unchanged. Requires `append2_block` (asserted): the block sits
    /// at f944+ after append2. Additive-only; joins the NEXT extraction
    /// regime wave (the HDR backfill).
    pub csfw_block: bool,
    /// BANDVIS dst-activity plane (the recorded V3(b)/(c) cross-fire fix
    /// from `benchmarks/append2_bandvis_gates_2026-07-27.md` REMAINDERS
    /// #3, implemented + adjudicated 2026-08-02 for the SOTA-944 P1.5
    /// sequencing decision): compute a distorted-side activity plane
    /// (`box_blur(|dst − mu2|)` — the exact dst twin of the existing
    /// ref-side `activity` chain, Y channel only, ≈+5%-class cost) and
    /// re-weight the BANDVIS **GAIN** polarity by the dst's own local
    /// flatness as a per-pixel POOLING weight:
    /// `gain_px = excess(band_dst, band_src)·(1 − sat(act_dst, C_ACTIVITY))`.
    /// The weight sits OUTSIDE the FR ratio — the adjudication measured
    /// that a flatness mask INSIDE `bounded_excess` is ratio-cancelled
    /// (scale-invariance) and suppresses real banding MORE than the
    /// dither it targets. **LOSS stays bit-identical to the OFF math**:
    /// the dst/src-side weights both measured direction-inverting on the
    /// deband credit, and LOSS is the LYB-validated workhorse polarity.
    /// Identity pairs stay exactly 0 either way.
    ///
    /// ADJUDICATION VERDICT (2026-08-02): both pre-registered masking
    /// arms FAILED their suppression gates at the resonant scale (banding
    /// contours ARE local activity — the plane cannot separate sparse
    /// contours from dense texture); production extraction (bigcodec +
    /// every P1 backfill) runs with this OFF. The toggle is retained as
    /// the P3/LOO research surface: the shipped GAIN combine is the
    /// strongest candidate measured (geometry cross-fire 0.33×, deband
    /// margin 2.2× OFF's). Default OFF: no dst-activity plane is
    /// computed and every path/byte — both routes, all regimes — is
    /// unchanged (const-split kernel; the OFF instantiations emit
    /// today's exact operation sequences). When ON, ONLY the four
    /// per-scale `idx_append2::BANDVIS_GAIN` slots change; every other
    /// slot INCLUDING `BANDVIS_LOSS` stays bit-identical. Requires
    /// `append2_block` (asserted).
    /// Record: `benchmarks/bandvis_dst_activity_2026-08-02.md`.
    pub append2_dst_activity: bool,
    /// Emit v1's peak / masked / IW pool blocks (`f156..372`) LIVE inside
    /// the folded walk instead of the structural zeros (2026-08-30, the
    /// carrier lane: `benchmarks/balance_campaign_2026-08-28.md` "carriers
    /// named + costed" → "un-zero the native slots under a regime flag").
    /// Per v1-aligned band the fold hook replays v1's extended strip
    /// section — see [`fold_v1_basic_bands`] — so at
    /// `pyramid_plane_stride(w) == w` the block is BIT-IDENTICAL to v1's 372
    /// extraction (`folded720_v1_pools_match_v1_path`). Default OFF: rows
    /// with the block live are their OWN extraction regime — never
    /// column-mix them with zeroed-block folded rows
    /// (`ZensimV2Result::v1_pools`; extractor modes `foldapp2carriers` /
    /// `foldapp2pools`). [`V1PoolsMode::Carriers`] emits ONLY the ten
    /// carrier slots the `fused944native` tables carry (the peaks are free;
    /// the masked/IW art-L4 slots need the activity map + the fused edge
    /// kernel at scales 0-1 only); [`V1PoolsMode::Full`] emits all 216.
    /// MEASURED (zenbench paired, `benches/fold_pools_bench.rs`): the full
    /// block is NOT free — +25-32% @576², +34-40% @1152² over the zeroed
    /// fold; see the campaign ledger for the carriers-only cost.
    /// **Which FORMULA REVISION this extraction computes.**
    ///
    /// The revision axis is a PER-REQUEST property, not a process-global one,
    /// and the reason is a measurement: `bake_block_profile` shows all three
    /// shipped 944 bakes (`c_sdr_mlp944_corrmix`, `c_hdr_l1t1944`,
    /// `c_sdr_purity944`) reading the full `GLOBAL_DMEAN`/`CGAIN`/`CLOSS` set
    /// — 33 slots each — so a GLOBAL flip to `Rev2` would move 22 of 33
    /// inputs per bake and silently re-price every one of them. A bake must
    /// be able to declare the revision it was trained against and get it,
    /// while a bake refit at a later revision coexists in the same process.
    ///
    /// Defaults to [`crate::ssim_form::active_revision`], so every existing
    /// construction is byte-identical AND `ZENSIM_FORMULA_REV` keeps working
    /// as the whole-process pin.
    ///
    /// **Honest scope: this carries the F5 axis, not F4's.** `Rev2`'s
    /// `paired_global_contrast` is a FINALISER parameter, so it threads
    /// per-walk for free. Its luma-form half (`ssim_form::active_luma_form`)
    /// is a `OnceLock` read inside the SIMD kernels, and making THAT
    /// per-request is a change to the kernel dispatch this lane does not own.
    /// So a per-bake `Rev2` gets rev2's global-contrast arithmetic and the
    /// process's luma form; `feature_plan` refuses rather than pretending
    /// otherwise when the two disagree.
    pub formula_revision: crate::feature_defs::FormulaRevision,
    pub v1_pools: V1PoolsMode,
    /// **TEST/BENCH INSTRUMENTATION — NOT A PRODUCT MODE** (`#[doc(hidden)]`
    /// 2026-08-30 by user decision: 944 with all pools live is
    /// the ONLY product mode; there is no 372-only product path and no public
    /// or CLI surface offers one). Retained solely as the control arm for
    /// `free_extras_are_pure_addition_to_the_v1_only_walk` and for pricing what the v2-era
    /// blocks cost inside the one product walk.
    ///
    /// It stays `pub` only because the struct is constructed by external
    /// crates with `..Default::default()`, and functional record update
    /// requires every field to be visible — making it `pub(crate)` breaks
    /// every out-of-crate constructor, zenmetrics included. `#[doc(hidden)]`
    /// takes it off the documented surface, which is as far as this can go
    /// without a builder-style redesign of the struct (listed for approval).
    ///
    /// BLOCK-SKIPPING (2026-08-30): compute ONLY v1's blocks — `f0..156`
    /// basic plus, with [`Self::v1_pools`], `f156..372` pools — and skip
    /// every v2-era block (dense-348, gradient, append, append2/BANDVIS,
    /// CSFW). Skips their upstream work too: phase A drops the four
    /// `box_blur_v_from_copy` sweeps that produce the V-blurred
    /// `mu1/mu2/ssq/s12` strips AND the v2 activity chain, because nothing
    /// v1 needs reads them (`fold_v1_basic_bands` takes the H-blurred
    /// planes and computes its own activity internally).
    ///
    /// PURE COMPUTE-SKIPPING — the slots that ARE emitted are bit-identical
    /// to the same request with this off (gated by
    /// `free_extras_are_pure_addition_to_the_v1_only_walk`). Semantics are untouched; this
    /// is not the padded-width question.
    ///
    /// The emitted vector keeps the requesting regime's WIDTH, with
    /// `f372..` left at the structural 0.0 — the final 372-width plumbing
    /// waits on the pad-semantics decision (campaign §8), so that this
    /// lever can be measured and shipped independently of it.
    #[doc(hidden)]
    pub v1_only: bool,
    /// FREE-EXTRAS (2026-09-01): emit the v2-era slots that a v1-only walk
    /// can produce for free — see [`V1FreeExtras`]. `#[doc(hidden)] pub` for
    /// the same reason [`Self::v1_only`] is: this struct is constructed
    /// out-of-crate with `..Default::default()`, which requires every field
    /// to be visible.
    #[doc(hidden)]
    pub free_extras: V1FreeExtras,
    /// REV4 CANDIDATE BANK (2026-09-23, `benchmarks/rev4_featbank_impl_
    /// 2026-09-23.md`): the f986+ grid-phase blocking block
    /// ([`crate::feature_defs::BLOCKS`] `gridblk` — 8 signals per
    /// (scale, channel): 6 signed log-magnitude bins of the
    /// activity-normalised lattice-boundary step excess at the estimated
    /// grid phase, plus on-grid mean and on/off ratio). Default OFF: with
    /// this false every existing path, layout, and byte is unchanged.
    /// Independent of the other rev4 families — the emit tail is
    /// absolute-positioned, so any subset stays index-stable; requires
    #[doc(hidden)]
    pub rev4_gridblk: bool,
    /// REV4 C2 ringing-magnitude basis (f1082+ — 6 triangular
    /// log-magnitude bins of the gradient kernel's own per-pixel RINGING
    /// term, per (scale, channel)). Independent of the other rev4
    #[doc(hidden)]
    pub rev4_ringbasis: bool,
    /// REV4 C3 tail pooling (f1154+ — per (scale, channel): p95/p99/max
    /// of the SSIM/ART/DET/MSE per-pixel maps from the family's fixed
    /// 32-bin log-edged histogram). Independent of the other rev4
    #[doc(hidden)]
    pub rev4_tailhist: bool,
    /// REV4 C4 artifact-type descriptors (f1298+ — per scale: Y
    /// EDGE_WIDTH_CHANGE×HF_LOSS blur, flat-region HF_GAIN noise per
    /// channel, dst-chroma gradient excess outside a ±1-px dilated
    /// dst-luma-edge mask for X/B bleeding). Independent of the other
    #[doc(hidden)]
    pub rev4_arttype: bool,
    /// Experimental C8 GMSBANK f1322..f1501, default off. The 2026-09-24
    /// revision carries native Y and coarse X/Y/B gradient loss/gain/deviation,
    /// plus coarse joint X/B chromaticity loss/deviation, each at five pinned
    /// stabilizers. Earlier C8 sidecars are incompatible despite equal width.
    #[doc(hidden)]
    pub gmsbank: bool,
    /// Restored cut A1 `mapdev` f1502..1561: per-(scale, channel) population
    /// std of the raw squared-error and the four HF maps. Default OFF. The
    /// layout requires the full C8 prefix (f0..f1501).
    #[doc(hidden)]
    pub mapdev: bool,
    /// Restored cut B2 `z1max` f1562..1789: the basic+peaks surface pooled
    /// over ungated 5x5 block maxima. Default OFF. Requires `mapdev`'s layout.
    #[doc(hidden)]
    pub z1max: bool,
    /// Restored cut `gmsnative` f1790..1819: C8's X/B gradient bank at native
    /// scale. Default OFF. Requires `z1max`'s layout.
    #[doc(hidden)]
    pub gmsnative: bool,
    /// `globaff` f1867..1890 ([`crate::globaff`]): the global affine-change
    /// descriptor, formula revision 6 only. Default OFF. Requires
    /// `gmsnative`'s layout; f1820..1866 (reserved and the research palette)
    /// are emitted as structural zeros before it.
    #[doc(hidden)]
    pub globaff: bool,
}

/// Which of v1's pool slots (`f156..372`) the folded walk emits live.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum V1PoolsMode {
    /// Structural zeros (the production folded regimes).
    #[default]
    Off,
    /// The ten carrier slots of the `fused944native` regime — `art_l8` at
    /// (s1,c0) (s1,c2) (s2,c0) (s3,c2) = f178/190/196/226,
    /// `masked_art_4th` at s0 c0-2 = f231/237/243, `iw_art_4th` at (s0,c0)
    /// (s1,c0) (s1,c2) = f303/321/333 — every other pool slot stays 0.
    Carriers,
    /// **The peak block only** (`f156..228`) — masked and IW stay 0.
    ///
    /// Costs the SAME as [`Self::Off`]: the peak accumulators (`ssim_d8`,
    /// `edge_art8`, `edge_det8` and the three running maxima) are produced
    /// UNCONDITIONALLY by `fused::fused_vblur_features_ssim` on every path
    /// and merged unconditionally by `V1BasicSums::accumulate`, so `Off` has
    /// been paying for them all along and simply declined to emit them. What
    /// this mode skips relative to [`Self::Full`] is the masked/IW pass
    /// group — the band activity chain (`abs_diff_into` +
    /// `box_blur_1pass_into`), the fused kernel's `store_mu`/`store_sigma`
    /// side-outputs, and the three `*_inline_both` kernels — which is the
    /// whole measurable cost of the pool block.
    ///
    /// The band scratch is still handed to the band so the memory-traffic
    /// self-blur shape (`FoldHSource::SelfBlur`) stays available; it owns
    /// only the four H planes there, never the pool planes.
    Peaks,
    /// All 216 slots: peaks (72) + masked (72) + IW (72), v1-exact.
    Full,
}

impl V1PoolsMode {
    /// The ten carrier slots (`Carriers`), in v1's 372 layout.
    pub const CARRIER_SLOTS: [usize; 10] = [178, 190, 196, 226, 231, 237, 243, 303, 321, 333];
}

/// Which v2-era slots a **v1-only** walk emits from byproducts the v1 fused
/// kernel can carry without a new plane, a new load or a new pass.
///
/// The classification behind this type is
/// `benchmarks/free_features_2026-09-01.md`. Read from source: of the 788
/// slots above `f156`, exactly **40** are a function of the RAW `src`/`dst`
/// planes alone — every other one needs either the V-blurred moment planes
/// (which a v1 band computes at a DIFFERENT accumulation origin than phase A,
/// so its value would not be the 944 table's value), the ref-side activity
/// plane, the `bs2` σ-split plane, or a neighbour stencil. Those 40 are the
/// three `GLOBAL_*` append slots and append2's `LUMA_MEAN_REF`, and all four
/// finalize from the same four sums.
///
/// This is a COMPUTE-SET selector at a FIXED layout, exactly like
/// [`V1PoolsMode`]: the emitted vector keeps the requesting regime's width
/// and every other slot stays at its structural 0.0. Rows extracted with it
/// ON are their own regime — never column-mix them with `Off` rows, whose
/// `GLOBAL_*` are structural zeros rather than values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum V1FreeExtras {
    /// Nothing extra. The fused kernel runs the identical instruction
    /// sequence it ran before this type existed.
    #[default]
    Off,
    /// Accumulate Σsrc, Σdst, Σsrc², Σdst² in the fused kernel and finalize
    /// `GLOBAL_DMEAN` / `GLOBAL_CGAIN` / `GLOBAL_CLOSS` (`f720+`, 3 per
    /// (scale, channel), minus the `APPEND_SKIP_B_SCALE0` cell the 944 walk
    /// also skips) and `LUMA_MEAN_REF` (`f924+`, one per scale, Y-only).
    ///
    /// Only writes slots whose OWNING block is not running: with the append
    /// block on, the append kernel owns `GLOBAL_*` and this writes nothing
    /// there. So a full 944 walk is unaffected whatever this is set to.
    RawMoments,
    /// [`Self::RawMoments`] PLUS the class-C **bounded-error** tranche —
    /// a strict superset, so a request for this carries the raw moments too.
    ///
    /// Adds `Σ sat((s−d)², C_MSE)` on every channel and, on the Y channel
    /// only, the dark/bright Bernstein-weighted sums of that same per-pixel
    /// value. Those finalize:
    ///
    /// * the v2-348 `MSE` slot, 12 (scale, channel) cells;
    /// * the append block's `LUM_DARK_ERR` / `LUM_MID_ERR` /
    ///   `LUM_BRIGHT_ERR` trio at the Y channel, 4 scales = 12 slots
    ///   (`LUM_MID_ERR` DERIVED from the partition of unity exactly as
    ///   [`finish_append`] derives it).
    ///
    /// **Y only for the bins, by construction, not by choice.** The weight
    /// is `sat(ref_Y, C_LUM_T)`, and the Y channel's own `src` plane IS the
    /// reference luma — so on Y the value rides registers the walk already
    /// holds. On X/B the same slots would need the ref-Y plane streamed
    /// into a per-channel kernel: no new *compute*, but a new *load stream*,
    /// which is outside the class-C ("no new plane, load or pass")
    /// definition. Those 21 slots stay at their structural zeros; see
    /// `benchmarks/free_features_classC_2026-09-04.md` §4.
    ///
    /// No slot is renumbered or invented: every one of the 24 is an existing
    /// 944-layout position, filled by a cheaper route.
    RawMomentsPlusBoundedErr,
}

/// Indices of the 40 FREE slots ([`V1FreeExtras::RawMoments`]) in the 944
/// layout, for an `n_scales`-scale pyramid: the three `GLOBAL_*` append
/// slots per (scale, channel) plus append2's per-scale `LUMA_MEAN_REF`. The
/// `APPEND_SKIP_B_SCALE0` cell is excluded — the 944 walk never computes
/// that append cell, so its slots are structural zeros on BOTH sides and
/// including them would make a parity gate weaker, not stronger.
///
/// Shared by the registered feature definitions and free-accumulator gates.
pub(crate) fn free_slot_indices(n_scales: usize) -> Vec<usize> {
    // v1 (31/ch/scale) ++ v2 (29) ++ append (17) ++ append2 (5/scale).
    let append0 = n_scales * 3 * (31 + FEATURES_PER_CHANNEL_V2_TOTAL);
    let append2_0 = append0 + n_scales * 3 * FEATURES_PER_CHANNEL_APPEND;
    let mut v = Vec::new();
    for scale in 0..n_scales {
        for ch in 0..3 {
            if !append_cell_active(true, ch, scale) {
                continue;
            }
            let base = append0
                + scale * 3 * FEATURES_PER_CHANNEL_APPEND
                + ch * FEATURES_PER_CHANNEL_APPEND;
            for local in [
                idx_append::GLOBAL_DMEAN,
                idx_append::GLOBAL_CGAIN,
                idx_append::GLOBAL_CLOSS,
            ] {
                v.push(base + local);
            }
        }
        v.push(append2_0 + scale * APPEND2_PER_SCALE + idx_append2::LUMA_MEAN_REF);
    }
    v
}

/// Indices of the 24 CLASS-C slots
/// ([`V1FreeExtras::RawMomentsPlusBoundedErr`]'s addition over
/// [`V1FreeExtras::RawMoments`]) in the 944 layout: the v2-348 `MSE` slot
/// per (scale, channel), plus the append block's three luminance-binned
/// error slots at the Y channel only.
///
/// Same contract as [`free_slot_indices`] — ONE derivation, reused by the
/// gate tests and by [`crate::feature_plan::Plan::for_bake`], never
/// re-typed as a literal list.
pub(crate) fn class_c_slot_indices(n_scales: usize) -> Vec<usize> {
    let v1_total = n_scales * 3 * 31;
    let append0 = v1_total + n_scales * 3 * FEATURES_PER_CHANNEL_V2_TOTAL;
    let mut v = Vec::new();
    for scale in 0..n_scales {
        for ch in 0..3 {
            v.push(
                v1_total
                    + scale * 3 * FEATURES_PER_CHANNEL_V2_TOTAL
                    + ch * FEATURES_PER_CHANNEL_V2_TOTAL
                    + idx::MSE,
            );
        }
        // Luminance bins: Y channel only (the register-carry constraint —
        // see `V1FreeExtras::RawMomentsPlusBoundedErr`), and only where the
        // 944 walk itself computes the append cell.
        let ch = APPEND2_CHANNEL;
        if append_cell_active(true, ch, scale) {
            let base = append0
                + scale * 3 * FEATURES_PER_CHANNEL_APPEND
                + ch * FEATURES_PER_CHANNEL_APPEND;
            for local in [
                idx_append::LUM_DARK_ERR,
                idx_append::LUM_MID_ERR,
                idx_append::LUM_BRIGHT_ERR,
            ] {
                v.push(base + local);
            }
        }
    }
    v
}
impl V2NewFeatureToggles {
    /// The **feature-set id** an extraction with these toggles PRODUCES —
    /// `<compute>@w<layout>/<era>#<slots-hash8>`, the string an extractor
    /// stamps into its output table's `_MANIFEST.json`
    /// (`docs/FEATURE_SET_IDS.md` §6.1).
    ///
    /// `era` is the caller's: only the caller knows which extractor build it
    /// is, and an unregistered/unknown era must be passed as
    /// [`crate::feature_set_id::ERA_UNKNOWN`] rather than guessed — a wrong
    /// era label is worse than none.
    ///
    /// `None` only when `era` is not a valid token (`[a-z0-9_]+`).
    ///
    /// `#[doc(hidden)]`: an extractor-side provenance helper, not a product
    /// knob — the same treatment `fold_engine::ScoringEngine` gets.
    #[doc(hidden)]
    pub fn feature_set_id(
        &self,
        n_scales: usize,
        layout_width: usize,
        era: &str,
    ) -> Option<crate::feature_set_id::FeatureSetId> {
        ComputeSet::from_toggles(*self).feature_set_id(n_scales, layout_width, era)
    }

    /// The slots an extraction with these toggles POPULATES at `n_scales`,
    /// clipped to `layout_width` — the producer half of a feature-set id, for
    /// a manifest that wants the explicit list beside the hash.
    #[doc(hidden)]
    pub fn populated_slots(
        &self,
        n_scales: usize,
        layout_width: usize,
    ) -> crate::feature_set_id::SlotSet {
        ComputeSet::from_toggles(*self).populated_slots(n_scales, layout_width)
    }
}

impl Default for V2NewFeatureToggles {
    fn default() -> Self {
        Self {
            gradient_features: true,
            transducer_bank: true,
            blockiness: true,
            transducers_luma_only: false,
            append_block: false,
            append2_block: false,
            csfw_block: false,
            append2_dst_activity: false,
            formula_revision: crate::ssim_form::active_revision(),
            v1_pools: V1PoolsMode::Off,
            v1_only: false,
            free_extras: V1FreeExtras::Off,
            rev4_gridblk: false,
            rev4_ringbasis: false,
            rev4_tailhist: false,
            rev4_arttype: false,
            gmsbank: false,
            mapdev: false,
            z1max: false,
            gmsnative: false,
            globaff: false,
        }
    }
}

/// Zero the masking-transducer slots (core, fragility, low-k, high-k) for a
/// non-luma channel when `transducers_luma_only` is set. Index-stable
/// deprecate-by-mask: the slots keep their positions, emitted as 0. `ch==1`
/// is the Y (luma) channel — left untouched. No-op when the toggle is off.
#[inline]
pub(crate) fn apply_transducer_luma_gate(out: &mut [f64], ch: usize, toggles: V2NewFeatureToggles) {
    if toggles.transducers_luma_only && ch != 1 {
        out[idx::PJND_TRANSDUCER] = 0.0;
        out[idx::PJND_FRAGILITY] = 0.0;
        out[idx::PJND_TRANSDUCER_LOW_K] = 0.0;
        out[idx::PJND_TRANSDUCER_HIGH_K] = 0.0;
    }
}

/// Perf-pass (phase 3, `docs/FEATURE_V2_SPEC_2026-07-18.md` §A.13) reusable
/// scratch buffers for [`compute_channel_scale_v2`], sized once for the
/// LARGEST scale (scale 0) and reused (sliced down to the current scale's
/// `n`) across all `n_scales * 3` calls in
/// [`compute_v2_features_impl_with_toggles`]. Eliminates 9 `vec![0.0f32; n]`
/// heap allocations (+ their zero-fill) per call — 108 allocations per
/// `compute_v2_features_impl_with_toggles` invocation before this change, 9
/// after (one-time, at the scratch's construction). Pure memory-management
/// change: does not alter any arithmetic or its order, so it carries zero
/// numerical risk (and zero risk to v1's golden gate, since v1 never
/// calls into this module).
///
/// Phase-5 (§A.15) RESIZED: was sized for the FULL channel-scale plane
/// (`width*height`, up to 4.2M elements at 2048²); now sized for one
/// STRIP-plus-halo (`width*(STRIP_ROWS+2*HALO_P)`, ≤ ~172K elements even
/// at 2048² width) — the actual memory-traffic reduction the bandwidth
/// hypothesis predicts should matter.
///
/// Ref-reuse pass: the halo-gather targets (`src_wide`/`dst_wide`) moved IN
/// here (they were per-call locals). The old aliasing objection — a
/// `&mut ScratchV2Strip` borrow against the `&[f32]` slices `run_blur_pass`
/// needs — is dissolved by `run_blur_pass_strip`, which destructures the
/// struct ONCE into disjoint field borrows (src/dst read-only, moment
/// buffers `&mut`), so no cell tricks are needed. Moving them in makes the
/// whole scratch set reusable ACROSS pairs via [`V2Scratch`], eliminating
/// the last per-call `vec![0.0f32; ..]` traffic (2 buffers × 3 channels ×
/// 4 scales = 24 zero-filled allocations per pair).
struct ScratchV2Strip {
    /// Halo-gathered strip inputs (`gather_strip_halo` targets) — the
    /// mirror-padded local windows of the source/distorted planes that
    /// every downstream stage of one strip iteration reads.
    src_wide: Vec<f32>,
    dst_wide: Vec<f32>,
    /// H-blur-only intermediates from [`crate::blur::fused_blur_h_ssim`]
    /// (see below) — analogous to v2 iteration-2's old per-call `tmp`, but
    /// now 4-wide since the fused kernel produces all four H-blurred planes
    /// in one pass.
    mu1_h: Vec<f32>,
    mu2_h: Vec<f32>,
    ssq_h: Vec<f32>,
    s12_h: Vec<f32>,
    /// Fully (H+V) blurred planes — same meaning as iteration-2's original
    /// `mu1`/`mu2`/`ssq`/`s12` locals.
    mu1: Vec<f32>,
    mu2: Vec<f32>,
    ssq: Vec<f32>,
    s12: Vec<f32>,
    /// Activity path (unchanged from iteration 2: `abs(src - mu1)` then
    /// blurred) — kept as its own two-buffer pair since
    /// `box_blur_1pass_into` still needs a temp scratch distinct from the
    /// 4-wide fused-H outputs above.
    abs_src: Vec<f32>,
    activity_tmp: Vec<f32>,
    activity: Vec<f32>,
    /// Streamed-walk σ-split plane: this kernel strip's `blur(src²)` wide
    /// buffer, filled per strip by the streaming walk (square + H + V
    /// inside the wide window — the P2 tiling) after the activity chain
    /// frees `abs_src`/`activity_tmp` for reuse as its temps. Unused by
    /// the materialized walk (which reads cached/replayed full planes).
    bs2: Vec<f32>,
    /// Distorted-side activity plane (`box_blur(|dst − mu2|)`) for the
    /// BANDVIS dst self-mask ([`V2NewFeatureToggles::append2_dst_activity`],
    /// Y channel only). LAZILY grown — allocated empty so the default-OFF
    /// heap profile (12 MP heaptrack 221.04 MB, append2 gates V5) is
    /// bit-for-bit unchanged; sized on first use by `stream_phase_a`.
    activity_dst: Vec<f32>,
}
/// **Item D — the compute-set descriptor.** WHAT one extraction request
/// actually computes, as one value, derived once.
///
/// This is the named home for the **compute** half of the layout-vs-compute
/// split the walk already documents: the emitted vector keeps the WIDTH and
/// REGIME the caller asked for (a `v1_only` 944 request is still a 944 row
/// with `f372..` at the structural 0.0), while this decides what work runs.
/// Before it, that decision lived in six ad-hoc locals recomputed from
/// [`V2NewFeatureToggles`] at the top of the walk and re-derived again inside
/// the strip loop; `from_toggles` is now the single derivation and
/// `compute_set_matches_legacy_derivation` holds it to the old one.
///
/// **Why it is worth a type.** Two things the charter asks for are naturally
/// expressed here and nowhere else:
///
/// 1. **Per-model drops (era-2 item E).** The v1 masked/IW/soft-peak pool pass
///    is **41.2 ms — 13.6 % of the tiled 5 MP walk** — and it is worth
///    *exactly nothing* to the 944 MLPs (they read those slots with zero
///    weight) and 0.399 CID22 to `B`. So the right shipping form is neither
///    "drop" nor "keep" but **"let the request say"**, computed from the
///    model's declared read IDs through `Plan::for_bake`.
/// 2. **HDR toggles.** The HDR append is a future regime whose blocks extend
///    this struct **append-only**, exactly as the feature numbering does. The
///    HDR front end already flows into the walk (`FrontEnd::Hdr` selects the
///    BANDVIS deltas and the `hl_bins` lane); an HDR block set becomes fields
///    here rather than more booleans on the public toggle struct.
///
/// **Public surface: none yet, deliberately.** `V2NewFeatureToggles` stays the
/// public request type and is unchanged; this is `pub(crate)` and derived from
/// it. Serving uses the internal feature planner; callers keep their existing
/// public extraction APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ComputeSet {
    /// The formula revision this request computes — see
    /// [`V2NewFeatureToggles::formula_revision`].
    pub formula_revision: crate::feature_defs::FormulaRevision,
    /// The v1 basic fold (`f0..155`). Always on today; a field so that a
    /// v2-only request is expressible rather than impossible.
    pub v1_basic: bool,
    /// Full-resolution X/B moments. Plans may omit them for basic/peak subsets.
    /// Conversion and the coarser XYB pyramid always remain complete.
    pub full_res_xb: bool,
    /// Coarse scales whose basic/peak moments need only Y (bits 1..3).
    pub coarse_y_only_scales: u8,
    /// Only basic local SSIM/edge moments and MSE; no peaks or HF ratios.
    pub local_only: bool,
    pub omit_edges: bool,
    pub sampling: Option<crate::sampling::Sampling>,
    /// v1's masked/IW/soft-peak pool slots — the 13.6 % pass of item E.
    pub v1_pools: V1PoolsMode,
    /// Scales that run the shared masked/IW activity chain in Full mode.
    /// Public extraction retains all scales; declared-ID plans narrow this.
    pub v1_full_scales: u8,
    /// Scales at which any v1-era slot (basic, peaks, masked, IW) is read.
    /// Public extraction keeps all scales; declared-ID plans narrow it, and the
    /// walk skips a scale's strips whole when neither this nor `v2_scales`
    /// names it (the pyramid is still built through it).
    pub v1_scales: u8,
    /// Actual v2-era work, including adjacent-scale gradient dependencies.
    pub v2_scales: u8,
    /// The v2-era blocks as a group (`f372..`): false for a `v1_only`
    /// request, which then computes NOTHING v2-era.
    pub v2_blocks: bool,
    pub gradient: bool,
    pub blockiness: bool,
    pub transducer_bank: bool,
    pub transducers_luma_only: bool,
    /// Declared plans may omit the reference-only PJND fragility finalizer.
    pub omit_pjnd_fragility: bool,
    pub append: bool,
    pub append2: bool,
    pub append2_dst_activity: bool,
    pub csfw: bool,
    /// REV4 C1 grid-phase blocking (f986+, `gridblk`): the strip kernel
    /// over the wide src/dst/activity windows. Independent compute gate —
    /// the emit tail is absolute-positioned, so each family can run alone.
    pub gridblk: bool,
    /// REV4 C2 ringing-magnitude bins (f1082+): the gradient kernel's
    /// per-pixel RINGING term folded into 6 triangular log-magnitude bins.
    pub ringbasis: bool,
    /// REV4 C3 tail pooling (f1154+): the dense kernel's d/art/det/mse
    /// per-pixel values folded into the family's 32-bin log histograms.
    pub tailhist: bool,
    /// REV4 C4 artifact-type descriptors (f1298+): flat-region HF_GAIN
    /// noise in the dense kernel, dst-chroma bleed outside the dilated
    /// dst-luma-edge mask in the gradient kernel, and the finalize-time
    /// EWC×HF_LOSS blur product.
    pub arttype: bool,
    pub gmsbank: bool,
    /// Restored cut A1 (`mapdev`): the per-band map side pass runs.
    pub mapdev: bool,
    /// Restored cut B2 (`z1max`): the block-max pooled side pass runs.
    pub z1max: bool,
    /// Restored cut `gmsnative`: C8's native-scale X/B gradient bank runs.
    pub gmsnative: bool,
    /// The `globaff` strip moment sums run (formula revision 6). Independent
    /// of the v2 kernels: it reads the planes of every channel at every scale.
    pub globaff: bool,
    /// The free v2-era slots a v1-only walk emits ([`V1FreeExtras`]). Held
    /// here rather than re-read from the toggles at each site, so
    /// `raw_moments` has ONE derivation.
    pub free_extras: V1FreeExtras,
}

/// **The revision a BAKE declares**, from its `zentrain.formula_revision`
/// metadata, defaulting to [`crate::ssim_form::SHIPPED_REVISION`] when it
/// carries none.
///
/// Per-bake rather than per-process because the alternative was measured to
/// be unshippable: `bake_block_profile` shows all three shipped 944 bakes
/// reading the full `GLOBAL_DMEAN`/`CGAIN`/`CLOSS` set (33 slots each), so a
/// global flip to `Rev2` would move 22 of 33 inputs per bake and silently
/// re-price all of them. A bake declares the revision it was trained against
/// and is served it; a bake refit later declares the newer one; both coexist.
///
/// An ABSENT value is the registered pre-stamp era (the shipped revision). A
/// PRESENT value this build does not know is refused, never guessed: until
/// featcanon-fix (2026-09-26) it fell back to the shipped revision, so a bake
/// stamped `4` was silently read as Rev1 by `bake_verdict` while
/// `Plan::for_bake` refused it. [`crate::feature_layout::formula_revision`] is
/// the one parser; this is its name at the feature layer.
pub(crate) fn bake_formula_revision(
    model: &crate::mlp::Model,
) -> Result<crate::feature_defs::FormulaRevision, ZensimError> {
    crate::feature_layout::formula_revision(model)
}

/// [`bake_formula_revision`], for consumers outside the crate (the verdict
/// tool's per-bake revision gate). Errors on a present value this build does
/// not know.
///
/// `#[doc(hidden)]`: revision plumbing, not product surface.
///
/// # Errors
///
/// [`ZensimError::ModelLoadFailed`] when `zentrain.formula_revision` is
/// present but not a known revision.
#[doc(hidden)]
pub fn bake_formula_revision_public(
    model: &zenpredict::Model,
) -> Result<FormulaRevision, ZensimError> {
    bake_formula_revision(model)
}

/// The formula revision THIS BUILD computes by default — the shipped one,
/// or whatever `ZENSIM_FORMULA_REV` pins.
///
/// `#[doc(hidden)]`: as above.
#[doc(hidden)]
#[must_use]
pub fn active_formula_revision() -> FormulaRevision {
    crate::ssim_form::active_revision()
}

impl ComputeSet {
    pub(crate) const ALL_SCALES: u8 = (1 << crate::NUM_SCALES) - 1;

    /// Narrow kernels without changing layout or the within-scale arithmetic.
    pub(crate) fn at_scale(self, scale: usize) -> Self {
        let live = self.v2_scales & (1 << scale) != 0;
        // A dead scale (see `scale_dead`) is skipped whole by the walk, so nothing (not even the v1 fold's free
        // extras) is computed there. Every scale the walk still runs keeps its view unchanged.
        let dead = self.scale_dead(scale);
        Self {
            v1_basic: self.v1_basic && !dead,
            v1_pools: if dead {
                V1PoolsMode::Off
            } else {
                self.pools_at(scale)
            },
            free_extras: if dead {
                V1FreeExtras::Off
            } else {
                self.free_extras
            },
            v2_blocks: self.v2_blocks && live,
            gradient: self.gradient && live,
            blockiness: self.blockiness && live,
            append: self.append && live,
            append2: self.append2 && live,
            csfw: self.csfw && live,
            gridblk: self.gridblk && live,
            ringbasis: self.ringbasis && live,
            tailhist: self.tailhist && live,
            arttype: self.arttype && live,
            gmsbank: self.gmsbank && live,
            mapdev: self.mapdev && live,
            z1max: self.z1max && live,
            gmsnative: self.gmsnative && live,
            ..self
        }
    }

    /// Only Full has a per-scale restriction. Peaks keeps the same basic
    /// and peak accumulators while omitting the activity/weighted-pool chain.
    pub(crate) fn pools_at(&self, scale: usize) -> V1PoolsMode {
        if self.v1_pools == V1PoolsMode::Full && self.v1_full_scales & (1 << scale) == 0 {
            V1PoolsMode::Peaks
        } else {
            self.v1_pools
        }
    }

    pub(crate) fn full_pool_scales(&self) -> u8 {
        match self.v1_pools {
            V1PoolsMode::Full => self.v1_full_scales,
            V1PoolsMode::Carriers => 0b0011,
            _ => 0,
        }
    }

    /// The ONE derivation from the public request type. Every `&&
    /// v2_blocks` here was previously written out at a call site; the
    /// invariant is that a v1-only request forces every v2-era block off
    /// rather than asserting they are already off, so `v1_only` composes
    /// with any toggle set (which is what makes it a drop-in bench A/B).
    pub(crate) fn from_toggles(t: V2NewFeatureToggles) -> Self {
        let v2_blocks = !t.v1_only;
        let append = t.append_block && v2_blocks;
        let append2 = t.append2_block && v2_blocks;
        let mut set = Self {
            formula_revision: t.formula_revision,
            v1_basic: true,
            full_res_xb: true,
            coarse_y_only_scales: 0,
            local_only: false,
            omit_edges: false,
            sampling: None,
            v1_pools: t.v1_pools,
            v1_full_scales: Self::ALL_SCALES,
            v1_scales: Self::ALL_SCALES,
            v2_scales: Self::ALL_SCALES,
            v2_blocks,
            gradient: t.gradient_features && v2_blocks,
            blockiness: t.blockiness && v2_blocks,
            transducer_bank: t.transducer_bank,
            transducers_luma_only: t.transducers_luma_only,
            omit_pjnd_fragility: false,
            append,
            append2,
            append2_dst_activity: t.append2_dst_activity && append2,
            csfw: t.csfw_block && v2_blocks,
            gridblk: t.rev4_gridblk && v2_blocks,
            ringbasis: t.rev4_ringbasis && v2_blocks,
            tailhist: t.rev4_tailhist && v2_blocks,
            arttype: t.rev4_arttype && v2_blocks,
            gmsbank: t.gmsbank && v2_blocks,
            mapdev: t.mapdev && v2_blocks,
            z1max: t.z1max && v2_blocks,
            gmsnative: t.gmsnative && v2_blocks,
            globaff: t.globaff,
            free_extras: t.free_extras,
        };
        set.rev5_scope();
        set
    }

    /// The two-family scope of [`crate::feature_defs::FormulaRevision::Rev5`]
    /// (`benchmarks/rev5_spec_2026-10-04.md` §1): `basic + peaks + v2`.
    ///
    /// Every flag naming a family outside that set is cleared, the
    /// masked/IW pool modes degrade to `Peaks`, and the free extras are
    /// off — so a plan whose LAYOUT reaches further than the request (the
    /// full-width `basic,peaks,v2` extraction is the driving case: the
    /// layout turns on every block flag the width reaches, and without
    /// this the walk would compute families nobody asked for) still
    /// computes only them. [`Self::from_toggles`] applies it under the
    /// Rev5 revision. Explicit unsupported requests are rejected first by
    /// `feature_plan` (`PlanError::UnsupportedAtRev5`) or, at raw extraction
    /// boundaries, `validate_raw_rev5_request`. Only validated plans may use
    /// unsupported layout flags without requesting those families.
    ///
    /// Sub-toggles of the supported families (`gradient`, `blockiness`,
    /// `transducer_bank`, `transducers_luma_only`, `v1_only` via
    /// `v2_blocks`, `local_only`, `omit_edges`, `sampling`, and the
    /// per-scale masks) are untouched — they narrow work inside the
    /// supported families, which is exactly what a read-set-derived plan
    /// is for.
    pub(crate) fn rev5_scope(&mut self) {
        if self.formula_revision < crate::feature_defs::FormulaRevision::Rev5 {
            return;
        }
        if self.v1_pools != V1PoolsMode::Off {
            self.v1_pools = V1PoolsMode::Peaks;
        }
        self.v1_full_scales = Self::ALL_SCALES;
        self.append = false;
        self.append2 = false;
        self.append2_dst_activity = false;
        self.csfw = false;
        self.gridblk = false;
        self.ringbasis = false;
        self.tailhist = false;
        self.arttype = false;
        self.gmsbank = false;
        self.mapdev = false;
        self.z1max = false;
        self.gmsnative = false;
        if !self.formula_revision.computes_globaff() {
            self.globaff = false;
        }
        self.free_extras = V1FreeExtras::Off;
    }

    /// Does the v1 fused kernel need to carry the four raw moments?
    ///
    /// Only when something will READ them: the free-extras request is on AND
    /// at least one owning block is off (with `append` on, the append kernel
    /// owns `GLOBAL_*`; with `append2` on it owns `LUMA_MEAN_REF`). So a full
    /// 944 walk never pays for this however the toggle is set.
    pub(crate) fn raw_moments(&self) -> bool {
        self.free_extras != V1FreeExtras::Off && !(self.append && self.append2)
    }

    /// Does the v1 fused kernel need to carry the class-C bounded per-pixel
    /// error `Σ sat((s−d)², C_MSE)`?
    ///
    /// Only when something will READ it: the request asks for the
    /// bounded-error tranche AND at least one owning block is off — the v2
    /// dense kernel owns the `MSE` slot, the append kernel owns the
    /// luminance bins that weight the same per-pixel value. A full 944 walk
    /// never pays for this however the toggle is set, exactly like
    /// [`Self::raw_moments`].
    pub(crate) fn bounded_err(&self) -> bool {
        self.free_extras == V1FreeExtras::RawMomentsPlusBoundedErr
            && (!self.v2_blocks || !self.append)
    }

    /// The per-channel free-extras work order handed to the fused kernel.
    /// `ch` decides the luminance bins alone: they read `sat(ref_Y, ..)`,
    /// which is a register carry ONLY on the channel whose `src` IS the
    /// reference luma ([`APPEND2_CHANNEL`]). Their owning block is the
    /// append kernel, so they are also skipped whenever it runs.
    pub(crate) fn free_work(&self, ch: usize, scale: usize) -> crate::fused::FreeExtrasWork {
        if self.formula_revision >= FormulaRevision::Rev5 && scale == 0 && ch != 1 {
            // Count actual scale-zero chroma consumers, regardless of plan intent.
            crate::fold_timing::work(crate::fold_timing::Work::Scale0XbCell, 1);
        }
        let bounded_err = self.bounded_err();
        crate::fused::FreeExtrasWork {
            revision: Some(self.formula_revision),
            omit_peaks: self.formula_revision >= FormulaRevision::Rev5
                && self.v1_pools == V1PoolsMode::Off,
            local_only: self.local_only,
            omit_edges: self.omit_edges,
            raw_moments: self.raw_moments(),
            bounded_err,
            lum_bins: bounded_err && !self.append && ch == APPEND2_CHANNEL,
        }
    }

    /// Which plane GROUPS the strip scratch must hold, given whether the v1
    /// fold is running self-blur bands (in which case phase A is skipped
    /// whole and its four fused-H outputs are never written).
    pub(crate) fn plane_needs(&self, self_blur: bool) -> StripPlaneNeeds {
        StripPlaneNeeds {
            h: !self_blur,
            v2: self.v2_blocks,
        }
    }

    /// True when the fold can own its H planes band-locally — the shape that
    /// skips phase A entirely. Needs the band pool scratch, which
    /// [`V1PoolsMode::Full`] and [`V1PoolsMode::Peaks`] both provide
    /// (`Peaks` hands the band the scratch for the four H planes alone —
    /// [`BandPoolWork::HOnly`]); [`V1PoolsMode::Off`] hands the band no
    /// scratch at all, which is why `fold_engine::pools_mode_for_need` never
    /// returns it.
    ///
    /// **THE owner of this predicate.** The strip loop's `fuse_channels` arm
    /// reads it rather than re-deriving it — those were two separate
    /// derivations until 2026-09-05 and they DISAGREED on `Peaks`: this one
    /// said `Full` only, the loop said `Full | Peaks`. Since this one sizes
    /// the strip scratch (via [`Self::plane_needs`]) and the loop decides
    /// whether phase A runs, a `Peaks` walk reserved phase A's four fused-H
    /// planes and then skipped phase A whole — allocating and first-touching
    /// planes nothing ever wrote, on the one mode the fast class ships.
    /// `self_blur_sizing_predicate_matches_the_strip_loop_predicate` pins
    /// them together; it fails against the pre-fix definition.
    pub(crate) fn self_blur_eligible(&self) -> bool {
        !self.v2_blocks
            && self.v1_basic
            && matches!(self.v1_pools, V1PoolsMode::Full | V1PoolsMode::Peaks)
    }

    /// The COMPUTE half of this request's **feature-set id**
    /// (`docs/FEATURE_SET_IDS.md` §2.1) — which registered slot families this
    /// walk actually populates.
    ///
    /// Derived from the SAME predicates the walk itself dispatches on
    /// ([`Self::raw_moments`] / [`Self::bounded_err`]), not from the raw
    /// toggle fields, so a full 944 walk correctly names NO free tranche: the
    /// append kernel owns `GLOBAL_*` there, and the free accumulators are not
    /// run at all.
    ///
    /// Sub-toggles (`gradient` / `blockiness` / `transducer_bank` /
    /// `append2_dst_activity`) are deliberately not tokens — they reshape a
    /// family from the inside, which moves the populated slot set, which moves
    /// the id's hash. The name is a handle; the hash is the identity.
    pub(crate) fn compute_parts(&self) -> crate::feature_set_id::ComputeParts {
        use crate::feature_set_id::{ComputeParts, ComputeToken as T};
        let mut p = ComputeParts::EMPTY;
        if self.v1_basic {
            p = p.with(T::Basic);
        }
        match self.v1_pools {
            V1PoolsMode::Off => {}
            V1PoolsMode::Carriers => p = p.with(T::Carriers),
            V1PoolsMode::Peaks => p = p.with(T::Peaks),
            V1PoolsMode::Full => p = p.with(T::Peaks).with(T::Masked).with(T::Iw),
        }
        if self.v2_blocks {
            p = p.with(T::V2);
        }
        if self.append {
            p = p.with(T::Append);
        }
        if self.append2 {
            p = p.with(T::Append2);
        }
        if self.csfw {
            p = p.with(T::Csfw);
        }
        if self.gridblk {
            p = p.with(T::Gridblk);
        }
        if self.ringbasis {
            p = p.with(T::Ringbasis);
        }
        if self.tailhist {
            p = p.with(T::Tailhist);
        }
        if self.arttype {
            p = p.with(T::Arttype);
        }
        if self.gmsbank {
            p = p.with(T::Gmsbank);
        }
        if self.mapdev {
            p = p.with(T::Mapdev);
        }
        if self.z1max {
            p = p.with(T::Z1max);
        }
        if self.gmsnative {
            p = p.with(T::Gmsnative);
        }
        if self.globaff {
            p = p.with(T::GlobAff);
        }
        if self.raw_moments() {
            p = p.with(T::Moments);
        }
        if self.bounded_err() {
            p = p.with(T::ClassC);
        }
        p
    }

    /// The slot set this request POPULATES, clipped to `layout_width`.
    ///
    /// Clipping is the LAYOUT rule: a family that does not exist at this width
    /// cannot be populated at it (a 372-wide request's free tranches vanish
    /// entirely — their slots live at `f720+`).
    ///
    /// Block bounds are composed from the same named constants the emit sites
    /// use, never re-typed as literals; the two scattered tranches come from
    /// [`free_slot_indices`] / [`class_c_slot_indices`], the SAME derivations
    /// `Plan::for_bake` checks a bake against.
    pub(crate) fn populated_slots(
        &self,
        n_scales: usize,
        layout_width: usize,
    ) -> crate::feature_set_id::SlotSet {
        use crate::feature_set_id::SlotSet;
        if self.v2_scales != Self::ALL_SCALES || self.v1_scales != Self::ALL_SCALES {
            return SlotSet::from_slots((0..n_scales).flat_map(|scale| {
                let local = Self {
                    v1_scales: Self::ALL_SCALES,
                    v2_scales: Self::ALL_SCALES,
                    ..self.at_scale(scale)
                };
                local
                    .populated_slots(n_scales, layout_width)
                    .iter_slots()
                    .filter(move |&id| {
                        crate::feature_defs::def_at(id, n_scales)
                            .is_some_and(|d| usize::from(d.scale) == scale)
                    })
                    .collect::<Vec<_>>()
            }));
        }
        let basic = n_scales * 3 * crate::metric::FEATURES_PER_CHANNEL_BASIC;
        let peaks_end = n_scales * 3 * crate::metric::FEATURES_PER_CHANNEL_WITH_PEAKS;
        let masked_end = n_scales * 3 * crate::metric::FEATURES_PER_CHANNEL_EXTENDED;
        let v1_total = masked_end + n_scales * 3 * crate::metric::FEATURES_PER_CHANNEL_IW;
        let v2_end = v1_total + n_scales * 3 * FEATURES_PER_CHANNEL_V2_TOTAL;
        let append_end = v2_end + n_scales * 3 * FEATURES_PER_CHANNEL_APPEND;
        let append2_end = append_end + n_scales * APPEND2_PER_SCALE;
        let csfw_end = append2_end + n_scales * CSFW_PER_SCALE;
        // Flat block: 30 slots at `csfw_end` regardless of `n_scales`
        // (`feature_defs::Replication::Flat` owns the same arithmetic).
        let reserved_end = csfw_end + 30;
        // REV4 families: registered (absolute) bases — the emit tail is
        // absolute-positioned, so each family's range lands at its
        // registered slots whether or not the families before it ran.
        let gridblk_end = reserved_end + n_scales * 3 * GRIDBLK_PER_CELL;
        let ringbasis_end = gridblk_end + n_scales * 3 * RINGBASIS_PER_CELL;
        let tailhist_end = ringbasis_end + n_scales * 3 * TAILHIST_PER_CELL;
        let arttype_end = tailhist_end + n_scales * ARTTYPE_PER_SCALE;
        let gmsbank_end = arttype_end + gmsbank_width(n_scales);
        let mapdev_end = gmsbank_end + n_scales * 3 * MAPDEV_PER_CELL;
        let z1max_end = mapdev_end + n_scales * 3 * Z1MAX_PER_CELL;
        let gmsnative_end = z1max_end + GMSNATIVE_WIDTH;

        let mut ranges: Vec<(usize, usize)> = Vec::new();
        let mut scattered: Vec<usize> = Vec::new();
        if self.v1_basic {
            ranges.push((0, basic));
        }
        match self.v1_pools {
            V1PoolsMode::Off => {}
            V1PoolsMode::Carriers => scattered.extend(V1PoolsMode::CARRIER_SLOTS),
            V1PoolsMode::Peaks => ranges.push((basic, peaks_end)),
            V1PoolsMode::Full => ranges.push((basic, v1_total)),
        }
        if self.v2_blocks {
            ranges.push((v1_total, v2_end));
        }
        if self.append {
            ranges.push((v2_end, append_end));
        }
        if self.append2 {
            ranges.push((append_end, append2_end));
        }
        if self.csfw {
            ranges.push((append2_end, csfw_end));
        }
        if self.gridblk {
            ranges.push((reserved_end, gridblk_end));
        }
        if self.ringbasis {
            ranges.push((gridblk_end, ringbasis_end));
        }
        if self.tailhist {
            ranges.push((ringbasis_end, tailhist_end));
        }
        if self.arttype {
            ranges.push((tailhist_end, arttype_end));
        }
        if self.gmsbank {
            ranges.push((arttype_end, gmsbank_end));
        }
        if self.mapdev {
            ranges.push((gmsbank_end, mapdev_end));
        }
        if self.z1max {
            ranges.push((mapdev_end, z1max_end));
        }
        if self.gmsnative {
            ranges.push((z1max_end, gmsnative_end));
        }
        // Registered base (after the reserved slots and the research palette
        // block), so the range follows the registry rather than a re-typed
        // offset. Every channel at every scale: the channel masks below are
        // kernel masks and do not apply to it.
        let globaff_base = crate::feature_defs::block_base(
            crate::feature_set_id::ComputeToken::GlobAff,
            n_scales,
        )
        .expect("globaff is a registered block")
        .0;
        let globaff_range = if self.globaff {
            globaff_base..globaff_base + n_scales * 3 * crate::globaff::GLOBAFF_PER_CELL
        } else {
            0..0
        };
        if self.globaff {
            ranges.push((globaff_range.start, globaff_range.end));
        }
        if self.raw_moments() {
            scattered.extend(free_slot_indices(n_scales));
        }
        if self.bounded_err() {
            scattered.extend(class_c_slot_indices(n_scales));
        }
        let slots = SlotSet::from_ranges(ranges)
            .union(&SlotSet::from_slots(scattered))
            .clipped_to(layout_width);
        SlotSet::from_slots(slots.iter_slots().filter(|&id| {
            globaff_range.contains(&id)
                || ((!self.omit_pjnd_fragility
                || !(v1_total..v2_end).contains(&id)
                || (id - v1_total) % FEATURES_PER_CHANNEL_V2_TOTAL != idx::PJND_FRAGILITY)
                && (!self.local_only
                    || (id < basic && id % crate::metric::FEATURES_PER_CHANNEL_BASIC < 10))
                && (!self.omit_edges || (id < basic && matches!(id % 13, 0..=2 | 9)))
                && (self.full_res_xb || !Self::is_full_res_xb(id, n_scales))
                && !crate::feature_defs::def_at(id, n_scales).is_some_and(|d| {
                    d.scale > 0
                        && self.coarse_y_only_scales & (1 << d.scale) != 0
                        && matches!(
                            d.channel,
                            crate::feature_defs::Channel::X | crate::feature_defs::Channel::B
                        )
                })
                && !(self.v1_pools == V1PoolsMode::Full
                    && (peaks_end..v1_total).contains(&id)
                    && crate::feature_defs::def_at(id, n_scales)
                        .is_some_and(|d| self.pools_at(usize::from(d.scale)) != V1PoolsMode::Full)))
        }))
    }

    /// The v1 activity chains are channel-local, so masked/IW subsets may
    /// also omit finest X/B. Cross-channel and free-extra families retain
    /// the complete walk until their dependencies have their own plan.
    pub(crate) fn allows_full_res_y_subset(&self) -> bool {
        (!self.v2_blocks || self.v2_scales & 1 == 0 || self.chroma_local_families())
            && self.free_extras == V1FreeExtras::Off
            && matches!(
                self.v1_pools,
                V1PoolsMode::Off | V1PoolsMode::Peaks | V1PoolsMode::Full
            )
    }

    /// True when no family that reads across channels is computed: the v1 blocks and the v2 core are
    /// channel-local (each channel's slots read only that channel's planes), so X/B work may be skipped
    /// families keep the complete walk until they have their own channel plan.
    pub(crate) fn chroma_local_families(&self) -> bool {
        !(self.append
            || self.append2
            || self.csfw
            || self.gridblk
            || self.ringbasis
            || self.tailhist
            || self.arttype
            || self.gmsbank
            || self.mapdev
            || self.z1max
            || self.gmsnative)
    }

    #[inline]
    pub(crate) fn channel_active(&self, scale: usize, channel: usize) -> bool {
        !self.scale_dead(scale)
            && (channel == 1
                || if scale == 0 {
                    self.full_res_xb
                } else {
                    self.coarse_y_only_scales & (1 << scale) == 0
                })
    }

    /// A scale at which nothing is read: no v1-era slot (`v1_scales`) and no v2-era work (`v2_blocks` at a
    /// `v2_scales` bit). Every walk computes no channel there; the pyramid is still built through it. Only
    /// declared-ID plans narrow `v1_scales`, so public extraction never has a dead scale.
    pub(crate) fn scale_dead(&self, scale: usize) -> bool {
        let bit = 1 << scale;
        self.v1_scales & bit == 0 && !(self.v2_blocks && self.v2_scales & bit != 0)
    }

    /// The REV4 work order for one (scale, channel) phase-B call — the
    /// SINGLE owner of the channel-dependent parts (C1's luma-vs-chroma
    /// lattice period, C4 bleed's chroma-only mask). `mask` is the strip's
    /// dilated dst-luma-edge mask built by the caller; it is forwarded
    /// only when `arttype` is on and `ch` is chroma (0 or 2), so the Y
    /// channel can be handed the same slice and still take the `None`
    /// path.
    fn rev4_work<'m>(
        &self,
        scale: usize,
        ch: usize,
        mask: Option<&'m [f32]>,
        chroma: Option<ChromaStrip<'m>>,
    ) -> Rev4Work<'m> {
        let period = (if ch == 1 {
            BLOCK_LATTICE
        } else {
            2 * BLOCK_LATTICE
        }) >> scale;
        Rev4Work {
            gridblk_period: if self.gridblk && period >= 2 {
                period
            } else {
                0
            },
            ringbasis: self.ringbasis,
            tailhist: self.tailhist,
            flat: self.arttype,
            bleed_mask: mask.filter(|_| self.arttype && ch != 1),
            gmsbank: gmsbank_cell_live(self, scale, ch),
            bank: GmsBankWork::new(ch, chroma.filter(|_| self.gmsbank && scale > 0 && ch == 0)),
        }
    }

    pub(crate) fn is_full_res_xb(id: usize, n_scales: usize) -> bool {
        use crate::feature_defs::Channel;
        crate::feature_defs::def_at(id, n_scales)
            .is_some_and(|d| d.scale == 0 && matches!(d.channel, Channel::X | Channel::B))
    }

    /// This request's full feature-set id at `era` — the producer-side id an
    /// extractor stamps into a table's `_MANIFEST.json`.
    pub(crate) fn feature_set_id(
        &self,
        n_scales: usize,
        layout_width: usize,
        era: &str,
    ) -> Option<crate::feature_set_id::FeatureSetId> {
        // Family tokens cannot reconstruct a channel subset. Its explicit
        // feature IDs and per-slot provenance remain the authoritative identity;
        // do not issue a shorthand that Request::for_set cannot reproduce.
        if !self.full_res_xb
            || self.coarse_y_only_scales != 0
            || self.local_only
            || self.sampling.is_some()
            || (self.v1_pools == V1PoolsMode::Full && self.v1_full_scales != Self::ALL_SCALES)
            || self.v1_scales != Self::ALL_SCALES
        {
            return None;
        }
        // The emitted width rides along as the legacy `@w<N>` hint — a
        // PRODUCER's id is exactly where recording it pays, because a reader
        // rebuilding this sparse set from its compute tokens needs the clip.
        // It is not part of the id's identity.
        crate::feature_set_id::FeatureSetId::from_slots_with_layout(
            self.compute_parts(),
            layout_width,
            era,
            &self.populated_slots(n_scales, layout_width),
        )
    }
}

/// Which groups of [`ScratchV2Strip`]'s planes a walk will actually write.
///
/// The set is a property of the REQUEST, and the fold-footprint lane measured
/// what ignoring it costs: a `v1_only` + [`V1PoolsMode::Full`] score — the
/// fold-backed scoring walk — runs self-blur bands, so phase A never executes
/// and the ONLY planes it writes are `src_wide` / `dst_wide`. The other twelve
/// were still allocated at `3 × 148 × width × 4` bytes apiece: **21,312·width
/// bytes untouched**, 24.6 MB at 1152² and 49.1 MB at 2304².
///
/// Under stock glibc those pages are demand-zero (`vec![0.0; n]` →
/// `alloc_zeroed` → `mmap`) and never fault, so they cost address space rather
/// than RSS — but that is an ALLOCATOR POLICY, not a property of the program.
/// MEASURED: with `MALLOC_ARENA_MAX=1` the same binary's `score_fold` peak RSS
/// at 1152²/1T goes 61,952 KiB → 71,768 KiB, because the allocation now comes
/// from the main heap and `calloc` has to memset it. Not asking for the planes
/// is the only way to not depend on that.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct StripPlaneNeeds {
    /// `mu1_h` / `mu2_h` / `ssq_h` / `s12_h` — phase A's fused-H outputs.
    /// False only when every v1 band blurs its own rows ([`FoldHSource::SelfBlur`]).
    h: bool,
    /// `mu1` / `mu2` / `ssq` / `s12` / `abs_src` / `activity_tmp` / `activity`
    /// / `bs2` — everything below `run_blur_pass_inner`'s `want_v2` early
    /// return, plus the σ-split and dst-activity buffers that reuse its temps.
    v2: bool,
}

impl StripPlaneNeeds {
    /// Every plane — what the materialized walks and the reference-moment
    /// cache ask for, and the safe default for any new caller.
    const ALL: Self = Self { h: true, v2: true };
    const NONE: Self = Self {
        h: false,
        v2: false,
    };
    fn union(self, o: Self) -> Self {
        Self {
            h: self.h || o.h,
            v2: self.v2 || o.v2,
        }
    }
}

impl ScratchV2Strip {
    fn new(max_n: usize) -> Self {
        Self::new_for(max_n, StripPlaneNeeds::ALL)
    }

    fn new_for(max_n: usize, needs: StripPlaneNeeds) -> Self {
        let hn = if needs.h { max_n } else { 0 };
        let vn = if needs.v2 { max_n } else { 0 };
        crate::fold_timing::work(
            crate::fold_timing::Work::ScratchZeroElements,
            2 * max_n + 4 * hn + 8 * vn,
        );
        Self {
            src_wide: vec![0.0f32; max_n],
            dst_wide: vec![0.0f32; max_n],
            mu1_h: vec![0.0f32; hn],
            mu2_h: vec![0.0f32; hn],
            ssq_h: vec![0.0f32; hn],
            s12_h: vec![0.0f32; hn],
            mu1: vec![0.0f32; vn],
            mu2: vec![0.0f32; vn],
            ssq: vec![0.0f32; vn],
            s12: vec![0.0f32; vn],
            abs_src: vec![0.0f32; vn],
            activity_tmp: vec![0.0f32; vn],
            activity: vec![0.0f32; vn],
            bs2: vec![0.0f32; vn],
            activity_dst: Vec::new(),
        }
    }
}

/// Reusable cross-pair scratch for the v2 extraction kernels — all 13
/// strip-sized working buffers for all 3 channels. Create ONCE per worker
/// thread and pass to
/// [`Zensim::compute_v2_features_with_ref_and_scratch`](crate::Zensim::compute_v2_features_with_ref_and_scratch)
/// for every pair that thread scores: buffers are lazily (re)sized to the
/// largest image seen and reused verbatim afterwards, so steady-state
/// extraction performs ZERO scratch allocation. Contents are overwritten
/// before every read (each strip iteration fully writes the ranges it
/// consumes), so reuse across unrelated pairs cannot leak data between
/// them — guarded by `scratch_reuse_matches_fresh_scratch` in this file's
/// tests.
pub struct V2Scratch {
    strips: [ScratchV2Strip; 3],
    sized_for: usize,
    /// The UNION of every plane set asked for so far — grow-only, exactly like
    /// `sized_for`, so a driver that alternates a v1-only score with a 944
    /// extraction converges to [`StripPlaneNeeds::ALL`] and never thrashes.
    sized_needs: StripPlaneNeeds,
    /// Recycled rolling-plane buffers for the STREAMING walk's
    /// [`crate::feature_v2_stream::StripPlaneProducer`] (drained at
    /// construction, refilled by `recycle` at walk end) — steady-state
    /// batch extraction performs zero producer allocation.
    stream_pool: Vec<Vec<f32>>,
    #[cfg(feature = "threads")]
    rev5_jobs: Vec<Rev5StripJob>,
    /// Recycled row-block buffers for [`crate::feature_v2_blocks`].
    #[cfg(feature = "threads")]
    block_pool: std::sync::Arc<crate::feature_v2_blocks::BlockPool>,
    #[cfg(all(test, feature = "threads"))]
    rev5_job_budget: Option<usize>,
}

impl V2Scratch {
    /// New, empty scratch. Buffers are allocated on first use.
    pub fn new() -> Self {
        Self {
            strips: [
                ScratchV2Strip::new(0),
                ScratchV2Strip::new(0),
                ScratchV2Strip::new(0),
            ],
            sized_for: 0,
            sized_needs: StripPlaneNeeds::NONE,
            stream_pool: Vec::new(),
            #[cfg(feature = "threads")]
            rev5_jobs: Vec::new(),
            #[cfg(feature = "threads")]
            block_pool: Default::default(),
            #[cfg(all(test, feature = "threads"))]
            rev5_job_budget: None,
        }
    }

    /// Grow (never shrink) every buffer to hold `strip_max_n` elements.
    fn ensure(&mut self, strip_max_n: usize) {
        self.ensure_for(strip_max_n, StripPlaneNeeds::ALL);
    }

    /// [`Self::ensure`] restricted to the plane groups the caller will write
    /// (see [`StripPlaneNeeds`]). Both the size and the set are grow-only, so
    /// this can only ever allocate LESS than `ensure` — never differently.
    fn ensure_for(&mut self, strip_max_n: usize, needs: StripPlaneNeeds) {
        let want = self.sized_needs.union(needs);
        if strip_max_n > self.sized_for || want != self.sized_needs {
            let n = strip_max_n.max(self.sized_for);
            self.strips = [
                ScratchV2Strip::new_for(n, want),
                ScratchV2Strip::new_for(n, want),
                ScratchV2Strip::new_for(n, want),
            ];
            self.sized_for = n;
            self.sized_needs = want;
        }
    }
}

impl Default for V2Scratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Runs the blur pass (fused H-blur + 4x V-blur + activity) on a LOCAL
/// image — external `src`/`dst` variant, used by the phase-6 whole-image
/// bypass path (which reads the real planes directly, no halo copy).
/// Identical algebra to phases 3/4's inline version; writes `n = width *
/// height_local` elements into each of `scratch`'s 11 moment buffers.
fn run_blur_pass(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height_local: usize,
    scratch: &mut ScratchV2Strip,
) {
    let ScratchV2Strip {
        mu1_h,
        mu2_h,
        ssq_h,
        s12_h,
        mu1,
        mu2,
        ssq,
        s12,
        abs_src,
        activity_tmp,
        activity,
        ..
    } = scratch;
    run_blur_pass_inner(
        src,
        dst,
        width,
        height_local,
        mu1_h,
        mu2_h,
        ssq_h,
        s12_h,
        mu1,
        mu2,
        ssq,
        s12,
        abs_src,
        activity_tmp,
        activity,
        // materialized walk: always wants the v2 planes
        true,
        // the materialized walk parallelises at the channel/scale level
        // above this call; a nested band fan-out here is not its axis.
        false,
        0,
        crate::ssim_form::active_revision(),
    );
}

/// Strip-path variant of [`run_blur_pass`]: the inputs are `scratch`'s own
/// `src_wide`/`dst_wide` halo buffers (filled by `gather_strip_halo` just
/// before this call). Destructuring the struct once yields disjoint
/// borrows — read-only input fields alongside `&mut` moment fields — which
/// is what lets the halo buffers live inside the reusable scratch at all.
fn run_blur_pass_strip(width: usize, height_local: usize, scratch: &mut ScratchV2Strip) {
    let n = width * height_local;
    let ScratchV2Strip {
        src_wide,
        dst_wide,
        mu1_h,
        mu2_h,
        ssq_h,
        s12_h,
        mu1,
        mu2,
        ssq,
        s12,
        abs_src,
        activity_tmp,
        activity,
        ..
    } = scratch;
    run_blur_pass_inner(
        &src_wide[..n],
        &dst_wide[..n],
        width,
        height_local,
        mu1_h,
        mu2_h,
        ssq_h,
        s12_h,
        mu1,
        mu2,
        ssq,
        s12,
        abs_src,
        activity_tmp,
        activity,
        // materialized walk: always wants the v2 planes
        true,
        // the materialized walk parallelises at the channel/scale level
        // above this call; a nested band fan-out here is not its axis.
        false,
        0,
        crate::ssim_form::active_revision(),
    );
}

/// Strip-path blur pass when the reference-side moments (`mu1`,
/// `activity`) come from a [`V2PreparedReference`] cache: 3-output fused
/// H (`fused_blur_h_ssim3` — the mu1 chain is compiled out on the v4x
/// tier; `mu1_h` is only fallback scratch) + V-blur of the 3
/// distorted/joint planes only. The
/// mu1 V-blur and the whole activity chain (abs-diff + 2-pass blur) are
/// SKIPPED — their values are read from the cache instead, which was
/// filled by replaying this exact strip walk (see
/// [`compute_ref_moments_channel`]), so consumers see bit-identical
/// inputs.
fn run_blur_pass_strip_cached_ref(width: usize, height_local: usize, scratch: &mut ScratchV2Strip) {
    let n = width * height_local;
    let ScratchV2Strip {
        src_wide,
        dst_wide,
        mu1_h,
        mu2_h,
        ssq_h,
        s12_h,
        mu2,
        ssq,
        s12,
        ..
    } = scratch;
    crate::blur::fused_blur_h_ssim3(
        &src_wide[..n],
        &dst_wide[..n],
        &mut mu1_h[..n],
        &mut mu2_h[..n],
        &mut ssq_h[..n],
        &mut s12_h[..n],
        width,
        height_local,
        BLUR_RADIUS,
        // The wide walk's revision IS the process revision: every entry to it
        // runs `validate_wide_revision` first, same as `run_blur_pass_strip`.
        crate::ssim_form::active_revision(),
    );
    crate::blur::box_blur_v_from_copy(&mu2_h[..n], &mut mu2[..n], width, height_local, BLUR_RADIUS);
    crate::blur::box_blur_v_from_copy(&ssq_h[..n], &mut ssq[..n], width, height_local, BLUR_RADIUS);
    crate::blur::box_blur_v_from_copy(&s12_h[..n], &mut s12[..n], width, height_local, BLUR_RADIUS);
}

/// Fill one channel-scale's cached reference moments by REPLAYING the
/// strip walk on `(src, src)`: same `gather_strip_halo` geometry, same
/// `fused_blur_h_ssim` sliding-sum chains (its `sum_s`/mu1 accumulator
/// reads ONLY the first input, so feeding `src` twice yields a `mu1_h`
/// bit-identical to the pair path's — the other three outputs are
/// discarded), same V-blur and activity functions. Bit-exactness of the
/// cache is BY CONSTRUCTION, not by tolerance: every arithmetic op that
/// produces a cached value is the same op, in the same order, on the same
/// input as the per-pair kernel would have executed.
fn compute_ref_moments_channel(
    src: &[f32],
    width: usize,
    height: usize,
    scratch: &mut ScratchV2Strip,
) -> V2RefMoments {
    let n = width * height;
    let mut mu1_full = vec![0.0f32; n];
    let mut act_full = vec![0.0f32; n];

    // Mirror the (currently disabled) whole-image bypass: below the
    // threshold the kernels blur the full plane directly, so the cache
    // must too.
    #[allow(clippy::absurd_extreme_comparisons)]
    if height <= STRIP_BYPASS_HEIGHT {
        run_blur_pass(src, src, width, height, scratch);
        mu1_full.copy_from_slice(&scratch.mu1[..n]);
        act_full.copy_from_slice(&scratch.activity[..n]);
        return V2RefMoments {
            mu1: mu1_full,
            activity: act_full,
        };
    }

    let mut y0 = 0usize;
    while y0 < height {
        let strip_h = STRIP_ROWS.min(height - y0);
        let wide_h = strip_h + 2 * HALO_P;
        let n_wide = width * wide_h;
        gather_strip_halo(
            src,
            width,
            height,
            y0,
            wide_h,
            HALO_P,
            &mut scratch.src_wide[..n_wide],
        );
        {
            let ScratchV2Strip {
                src_wide,
                mu1_h,
                mu2_h,
                ssq_h,
                s12_h,
                mu1,
                abs_src,
                activity_tmp,
                activity,
                ..
            } = scratch;
            crate::blur::fused_blur_h_ssim(
                &src_wide[..n_wide],
                &src_wide[..n_wide],
                &mut mu1_h[..n_wide],
                &mut mu2_h[..n_wide],
                &mut ssq_h[..n_wide],
                &mut s12_h[..n_wide],
                width,
                wide_h,
                BLUR_RADIUS,
            );
            crate::blur::box_blur_v_from_copy(
                &mu1_h[..n_wide],
                &mut mu1[..n_wide],
                width,
                wide_h,
                BLUR_RADIUS,
            );
            crate::simd_ops::abs_diff_into(
                &src_wide[..n_wide],
                &mu1[..n_wide],
                &mut abs_src[..n_wide],
            );
            crate::blur::box_blur_1pass_into(
                &abs_src[..n_wide],
                &mut activity[..n_wide],
                &mut activity_tmp[..n_wide],
                width,
                wide_h,
                BLUR_RADIUS,
            );
        }
        let off = HALO_P * width;
        let strip_n = width * strip_h;
        let out_base = y0 * width;
        mu1_full[out_base..out_base + strip_n].copy_from_slice(&scratch.mu1[off..off + strip_n]);
        act_full[out_base..out_base + strip_n]
            .copy_from_slice(&scratch.activity[off..off + strip_n]);
        y0 += strip_h;
    }

    V2RefMoments {
        mu1: mu1_full,
        activity: act_full,
    }
}

/// Build the full per-scale/per-channel moment cache for a prepared
/// reference (see [`V2PreparedReference::moments`]).
fn fill_ref_moments(scales: &[([Vec<f32>; 3], usize, usize)]) -> Vec<[V2RefMoments; 3]> {
    let (_, w0, h0) = &scales[0];
    #[allow(clippy::absurd_extreme_comparisons, clippy::unnecessary_min_or_max)]
    let bypass_rows = (*h0).min(STRIP_BYPASS_HEIGHT);
    let strip_max_n = w0 * (STRIP_ROWS + 2 * HALO_P).max(bypass_rows);
    let mut scratch = ScratchV2Strip::new(strip_max_n);
    scales
        .iter()
        .map(|(planes, width, height)| {
            std::array::from_fn(|ch| {
                compute_ref_moments_channel(&planes[ch], *width, *height, &mut scratch)
            })
        })
        .collect()
}

/// Row-band-parallel wrapper for [`crate::blur::fused_blur_h_ssim`].
///
/// **Bit-exact by construction, and the construction is the whole argument.**
/// A horizontal box blur is an independent running-sum recurrence per row, and
/// the kernels walk rows in groups of [`H_BLUR_ROW_GROUP`]
/// (`fused_blur_h_ssim_*_inner`'s `run_group(rg * 8, 8)` + one partial tail).
/// Bands here start at multiples of that group size, so every band's internal
/// grouping is a **sub-sequence of the whole-plane call's** — each row lands in
/// a group of the same size, at the same offset within it, on the same SIMD
/// tier. Only the LAST band can hold a partial group, which is exactly the
/// partial group the whole-plane call's tail would have run.
/// `phase_a_blur_bands_are_bit_exact` pins that over every band size and
/// geometry; `box_blur_h_ring_matches_regathered_reference` independently pins
/// the per-row claim the construction rests on.
///
/// Predecessor context (`extraction_perf_and_buffered_removal_2026-08-30.md`
/// §11.1): this lever was implemented, measured NEUTRAL and reverted **in the
/// 944-full walk**, where phase A is a small share behind `dense_block_kernel`.
/// In the `v1_only` SCORING request that backs the fold engine there is no
/// dense kernel at all and phase A is 36 % of the walk at 16 threads with an
/// occupancy of 0.157 — a different measurement, not a re-run of that one.
#[allow(clippy::too_many_arguments)]
fn fused_blur_h_ssim_banded(
    src: &[f32],
    dst: &[f32],
    mu1_h: &mut [f32],
    mu2_h: &mut [f32],
    ssq_h: &mut [f32],
    s12_h: &mut [f32],
    width: usize,
    height_local: usize,
    #[allow(unused_variables)] parallel: bool,
    revision: FormulaRevision,
) {
    #[cfg(feature = "threads")]
    if parallel && height_local > H_BLUR_BAND_ROWS && width > 0 {
        use rayon::prelude::*;
        let n = H_BLUR_BAND_ROWS * width;
        mu1_h
            .par_chunks_mut(n)
            .zip(mu2_h.par_chunks_mut(n))
            .zip(ssq_h.par_chunks_mut(n))
            .zip(s12_h.par_chunks_mut(n))
            .enumerate()
            .for_each(|(b, (((m1, m2), sq), s12))| {
                let __t = crate::fold_timing::start();
                let y0 = b * H_BLUR_BAND_ROWS;
                let rows = m1.len() / width;
                let lo = y0 * width;
                let hi = lo + rows * width;
                // The tile lives on `fused_blur_h_ssim` itself, so it nests
                // inside this row band automatically: the band gives the
                // thread its rows, the tile keeps that band's 6-plane window
                // dense. Each worker owns its own thread-local tile arena.
                crate::blur::fused_blur_h_ssim_at_revision(
                    &src[lo..hi],
                    &dst[lo..hi],
                    m1,
                    m2,
                    sq,
                    s12,
                    width,
                    rows,
                    BLUR_RADIUS,
                    revision,
                );
                crate::fold_timing::stop(__t, crate::fold_timing::Phase::BlurBandBusy, 0);
            });
        return;
    }
    crate::blur::fused_blur_h_ssim_at_revision(
        src,
        dst,
        mu1_h,
        mu2_h,
        ssq_h,
        s12_h,
        width,
        height_local,
        BLUR_RADIUS,
        revision,
    );
}

/// Shared body for [`run_blur_pass`]/[`run_blur_pass_strip`] — the actual
/// fused H-blur + 4x V-blur + activity sequence, over explicit buffers.
#[allow(clippy::too_many_arguments)]
fn run_blur_pass_inner(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height_local: usize,
    mu1_h: &mut [f32],
    mu2_h: &mut [f32],
    ssq_h: &mut [f32],
    s12_h: &mut [f32],
    mu1: &mut [f32],
    mu2: &mut [f32],
    ssq: &mut [f32],
    s12: &mut [f32],
    abs_src: &mut [f32],
    activity_tmp: &mut [f32],
    activity: &mut [f32],
    want_v2: bool,
    parallel: bool,
    // Diagnostic-only: which `fold_timing` scale slot this call's v2-plane
    // chain is attributed to. Never read by any kernel.
    t_scale: usize,
    revision: FormulaRevision,
) {
    if revision >= FormulaRevision::Rev5 && want_v2 {
        crate::fold_timing::work(crate::fold_timing::Work::V2Cell, 1);
    }

    let n = width * height_local;
    let mu1_h = &mut mu1_h[..n];
    let mu2_h = &mut mu2_h[..n];
    let ssq_h = &mut ssq_h[..n];
    let s12_h = &mut s12_h[..n];
    let __t_h = crate::fold_timing::start();
    fused_blur_h_ssim_banded(
        src,
        dst,
        mu1_h,
        mu2_h,
        ssq_h,
        s12_h,
        width,
        height_local,
        parallel,
        revision,
    );
    crate::fold_timing::stop(__t_h, crate::fold_timing::Phase::BlurHWall, 0);

    // BLOCK-SKIPPING: everything below this line feeds v2-era kernels ONLY.
    // `fold_v1_basic_bands` reads the H-blurred planes above and computes its
    // own activity internally, so a v1-only request can stop here — four
    // whole-window `box_blur_v_from_copy` sweeps and the activity chain
    // (abs-diff + a full 1-pass blur) are skipped, not merely unread.
    if !want_v2 {
        return;
    }

    let __t_pa = crate::fold_timing::start();
    let mu1 = &mut mu1[..n];
    crate::blur::box_blur_v_from_copy_at_revision(
        mu1_h,
        mu1,
        width,
        height_local,
        BLUR_RADIUS,
        revision,
    );
    let mu2 = &mut mu2[..n];
    crate::blur::box_blur_v_from_copy_at_revision(
        mu2_h,
        mu2,
        width,
        height_local,
        BLUR_RADIUS,
        revision,
    );
    let ssq = &mut ssq[..n];
    crate::blur::box_blur_v_from_copy_at_revision(
        ssq_h,
        ssq,
        width,
        height_local,
        BLUR_RADIUS,
        revision,
    );
    let s12 = &mut s12[..n];
    crate::blur::box_blur_v_from_copy_at_revision(
        s12_h,
        s12,
        width,
        height_local,
        BLUR_RADIUS,
        revision,
    );

    crate::fold_timing::work(crate::fold_timing::Work::ActivityChain, 1);
    let abs_src = &mut abs_src[..n];
    crate::simd_ops::abs_diff_into(src, mu1, abs_src);
    let activity = &mut activity[..n];
    crate::blur::box_blur_1pass_at_revision(
        abs_src,
        activity,
        &mut activity_tmp[..n],
        width,
        height_local,
        BLUR_RADIUS,
        revision,
    );
    crate::fold_timing::stop(__t_pa, crate::fold_timing::Phase::PhaseAV2Planes, t_scale);
}

// ============================================================================
// Phase-4 (§A.14): magetypes-SIMD dense formula pass + gradient pass.
//
// Two separate `#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]` hot loops,
// matching the brief's "compute_channel_scale_v2's formula pass + the shared
// gradient pass" scope exactly:
//   - `dense_block_kernel`: SSIM raw moments, edge artifact/detail, bounded
//     MSE, HF gain/loss/mag-loss, PJND core + transducer-bank, and all 11
//     masked/IW/soft-peak weighted-pool pairs. ALWAYS runs (matches the
//     scalar loop's unconditional block).
//   - `gradient_block_kernel`: GMS, ringing, banding, grad_src/grad_dst sums.
//     Only invoked when `toggles.gradient_features` — the toggle check is
//     hoisted OUTSIDE the pixel loop entirely (was a per-pixel branch in the
//     scalar version; now a single runtime branch per `compute_channel_
//     scale_v2` call), which is itself a small, free improvement.
//
// Blockiness is NOT ported to SIMD this phase — it is inherently sparse
// (only x%8==0 or y%8==0 lattice positions contribute), so a dense 8-wide
// SIMD pass would spend 7/8 of its lanes on masked-out zero contributions.
// It is restructured into its own small SCALAR pass that visits ONLY
// lattice positions (§A.14 "side quest not taken" — noted, not pursued,
// per the phase-4 addendum's single-lever discipline) — this is still
// strictly cheaper than the OLD dense-scalar-with-modulo-branch version it
// replaces, since it no longer visits all `width*height` pixels to find
// the ~1/8+1/8 fraction that matter.
//
// SSIM raw-moment reformulation (deliberate v2-only numeric shift,
// documented per this file's own 5e-4-tolerance allowance): Terriberry's
// online update is inherently sequential (each sample's update depends on
// the running n), so it does not vectorize across lanes. Instead, each
// lane accumulates plain running sums of d, d^2, d^3, d^4 (trivially
// SIMD — no cross-lane dependency), reduced ONCE PER ROW (not once per
// whole image — bounds the f32 accumulator's magnitude to ~width/8
// increments of a small [0,2]-bounded value, avoiding "large-swallows-
// small" f32 summation error) into a running f64 total. At the very end,
// the standard raw-to-central moment identities recover (mean, M2, M4):
//   mean = Sum_d / n
//   M2/n = Sum_d2/n - mean^2
//   M4/n = Sum_d4/n - 4*mean*(Sum_d3/n) + 6*mean^2*(Sum_d2/n) - 3*mean^4
// Matches Terriberry's OWN output up to floating-point reassociation
// (verified within 5e-4 relative on the fixture pairs — see the phase-4
// test `simd_matches_scalar_within_tolerance`).
// ============================================================================

/// Per-row-reduced f64 accumulator for the dense (always-on) block.
#[derive(Default, Clone, Copy)]
pub(crate) struct DenseAccum {
    pub(crate) central: OnlineMoments,
    pub(crate) sum_d: f64,
    pub(crate) sum_d2: f64,
    pub(crate) sum_d3: f64,
    pub(crate) sum_d4: f64,
    pub(crate) sum_art: f64,
    pub(crate) sum_det: f64,
    pub(crate) sum_mse: f64,
    pub(crate) sum_hf_gain: f64,
    pub(crate) sum_hf_loss: f64,
    pub(crate) sum_hf_mag_loss: f64,
    pub(crate) sum_pjnd: f64,
    pub(crate) sum_pjnd_lo: f64,
    pub(crate) sum_pjnd_hi: f64,
    pub(crate) ws_peak_ssim: WeightedSum,
    pub(crate) ws_peak_art: WeightedSum,
    pub(crate) ws_peak_det: WeightedSum,
    pub(crate) ws_mask_ssim: WeightedSum,
    pub(crate) ws_mask_art: WeightedSum,
    pub(crate) ws_mask_det: WeightedSum,
    pub(crate) ws_mask_mse: WeightedSum,
    pub(crate) ws_iw_ssim: WeightedSum,
    pub(crate) ws_iw_art: WeightedSum,
    pub(crate) ws_iw_det: WeightedSum,
    pub(crate) ws_iw_mse: WeightedSum,
}

impl DenseAccum {
    /// Phase-5 (§A.15): fold one strip's `DenseAccum` into the running
    /// whole-image total. Every field is a plain sum (or a `WeightedSum`,
    /// itself two plain sums), so summing N strips' partials is exactly
    /// equivalent to accumulating over the whole image in one pass —
    /// strip order does not change WHICH additions happen, only their
    /// grouping (associativity-level float reassociation only, same class
    /// already tolerated for the phase-4 SIMD-lane-then-row reduction).
    #[inline]
    pub(crate) fn accumulate(&mut self, other: &DenseAccum) {
        self.central.merge(other.central);
        self.sum_d += other.sum_d;
        self.sum_d2 += other.sum_d2;
        self.sum_d3 += other.sum_d3;
        self.sum_d4 += other.sum_d4;
        self.sum_art += other.sum_art;
        self.sum_det += other.sum_det;
        self.sum_mse += other.sum_mse;
        self.sum_hf_gain += other.sum_hf_gain;
        self.sum_hf_loss += other.sum_hf_loss;
        self.sum_hf_mag_loss += other.sum_hf_mag_loss;
        self.sum_pjnd += other.sum_pjnd;
        self.sum_pjnd_lo += other.sum_pjnd_lo;
        self.sum_pjnd_hi += other.sum_pjnd_hi;
        self.ws_peak_ssim.accumulate(&other.ws_peak_ssim);
        self.ws_peak_art.accumulate(&other.ws_peak_art);
        self.ws_peak_det.accumulate(&other.ws_peak_det);
        self.ws_mask_ssim.accumulate(&other.ws_mask_ssim);
        self.ws_mask_art.accumulate(&other.ws_mask_art);
        self.ws_mask_det.accumulate(&other.ws_mask_det);
        self.ws_mask_mse.accumulate(&other.ws_mask_mse);
        self.ws_iw_ssim.accumulate(&other.ws_iw_ssim);
        self.ws_iw_art.accumulate(&other.ws_iw_art);
        self.ws_iw_det.accumulate(&other.ws_iw_det);
        self.ws_iw_mse.accumulate(&other.ws_iw_mse);
    }
}

/// Scalar weighted-pool accumulation for ONE pixel — shared by the SIMD
/// kernel's per-lane extraction loop AND the scalar row tail (§A.14
/// register-pressure fix, see `dense_block_kernel_generic`'s doc: the 11
/// masked/IW/soft-peak `(Σw, Σwv)` pairs are 22 of the ~35 SIMD lane
/// accumulators live in the dense kernel's inner loop — measured (256²
/// iteration signal) to cause severe register-pressure regression when
/// vectorized alongside everything else. Scalarizing JUST this block
/// (extract the SIMD-computed `d`/`art_i`/`det_i`/`mse_i`/`act` lanes via
/// `to_array()`, accumulate per-lane here) keeps the division-heavy core
/// formulas (SSIM moments, edge artifact/detail, MSE, HF, PJND — the
/// higher-arithmetic-intensity part) vectorized while removing the 22
/// accumulators that were the dominant register-pressure source.
#[inline]
fn weighted_pool_accumulate_scalar(
    acc: &mut DenseAccum,
    d: f64,
    art_i: f64,
    det_i: f64,
    mse_i: f64,
    act: f64,
) {
    let mask_w = 1.0 - saturate(act, C_ACTIVITY);
    let iw_w = saturate(act, C_ACTIVITY) + IW_WEIGHT_FLOOR;
    acc.ws_mask_ssim.add(mask_w, d);
    acc.ws_mask_art.add(mask_w, art_i);
    acc.ws_mask_det.add(mask_w, det_i);
    acc.ws_mask_mse.add(mask_w, mse_i);
    acc.ws_iw_ssim.add(iw_w, d);
    acc.ws_iw_art.add(iw_w, art_i);
    acc.ws_iw_det.add(iw_w, det_i);
    acc.ws_iw_mse.add(iw_w, mse_i);

    let sal_ssim = saturate(d, C_PEAK);
    let sal_art = saturate(art_i, C_PEAK);
    let sal_det = saturate(det_i, C_PEAK);
    acc.ws_peak_ssim.add(sal_ssim, d);
    acc.ws_peak_art.add(sal_art, art_i);
    acc.ws_peak_det.add(sal_det, det_i);
}

/// Dense-block SIMD kernel body — generic over any `F32x8Backend` token.
/// Processes one row at a time: zero f32-lane accumulators, sweep the row
/// in 8-wide chunks (+ scalar tail for `width % 8 != 0`), reduce once per
/// row into the f64 [`DenseAccum`] running totals.
///
/// `POOL_SIMD` (register-assignment experiment, 2026-07-21): when `true`,
/// the 11 masked/IW/soft-peak weighted pools are accumulated in f32 SIMD
/// lanes (reduced per row to f64) instead of the §A.14 per-lane scalar
/// extraction. The pool block then needs only **16** lane accumulators —
/// not the 22 that caused §A.14's register-pressure regression — because
/// the mask family shares one `Σw` (same `mask_w` for all 4 pools), the
/// IW family shares one `Σw`, and both weights derive from a single
/// `saturate(act)` (1 division for 8 pixels instead of 16 scalar f64
/// divisions). Total live vectors ≈ 13 core + 16 pool + constants — fits
/// the 32 SIMD registers of AVX-512(VL); on 16-register tiers (AVX2/NEON)
/// it would spill, so the entry dispatch enables it for the v4x tier
/// only. Numerics: pool sums move from f64-per-pixel to
/// f32-lane-then-f64-row accumulation — the SAME reassociation class as
/// the phase-4 core-moment change, inside this module's documented 5e-4
/// tolerance (fixture tests + `pool_simd_drift_within_policy` gate it).
/// The scalar row tail keeps the exact f64 path in both modes.
///
/// `#[inline(always)]`, not `#[inline]`: this body exists ONLY to fuse
/// into its `#[arcane]`/magetypes entry's `target_feature` region. When
/// the POOL_SIMD variant pushed the body past LLVM's inline-cost
/// threshold, the hint stopped being honored and every V8 operator
/// compiled into a CALL to a non-inlined `core::arch` shim outside the
/// feature region — measured 5.3x whole-extraction regression (38.2s vs
/// 7.2s on 100 aic3 pairs; perf showed `__mm256_add_ps` as a standalone
/// 26% symbol). Forcing the inline is the entire fix.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn dense_block_kernel_generic<T: F32x8Backend + Copy, const POOL_SIMD: bool>(
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
    let direct = crate::ssim_form::active_revision() >= crate::feature_defs::FormulaRevision::Rev3;
    // Revision 7's cancellation-free forms, by the same process revision.
    let stable = crate::ssim_form::active_revision().stable_forms();
    let zero = V8::<T>::zero_t(token);
    let one = V8::<T>::splat_t(token, 1.0);
    let c1 = V8::<T>::splat_t(token, C1_V2 as f32);
    let c2 = V8::<T>::splat_t(token, C2_V2 as f32);
    let c_edge = V8::<T>::splat_t(token, C_EDGE as f32);
    let c_mse = V8::<T>::splat_t(token, C_MSE as f32);
    let c_hf = V8::<T>::splat_t(token, C_HF as f32);
    let c_pjnd_clamp = V8::<T>::splat_t(token, C_PJND_CLAMP as f32);
    let k_mid = V8::<T>::splat_t(token, K_PJND_MASK as f32);
    let k_lo = V8::<T>::splat_t(token, K_PJND_MASK_LOW as f32);
    let k_hi = V8::<T>::splat_t(token, K_PJND_MASK_HIGH as f32);
    // POOL_SIMD-only constants (const-folded out otherwise).
    let c_activity = V8::<T>::splat_t(token, C_ACTIVITY as f32);
    let c_peak = V8::<T>::splat_t(token, C_PEAK as f32);
    let iw_floor = V8::<T>::splat_t(token, IW_WEIGHT_FLOOR as f32);

    let mut acc = DenseAccum::default();
    let width8 = width - (width % 8);

    for y in 0..height {
        let row = y * width;

        // Row-local f32 lane accumulators — zeroed each row and reduced
        // ONCE at the end of the row (bounds the f32 summation magnitude
        // to ~width/8 terms, see the module-level note above).
        //
        // §A.14 register-pressure fix: the masked/IW/soft-peak weighted-pool
        // accumulators (11 pairs = 22 registers in the first version of this
        // kernel) are NOT here -- they're accumulated scalar, per lane,
        // immediately after each SIMD chunk (see `weighted_pool_accumulate_
        // scalar` below), which measurably fixed a severe register-pressure
        // regression (256x256 iteration signal: 30.4ms fused-block-only vs
        // this version's number in §A.14's bench table).
        let (mut r_d, mut r_d2, mut r_d3, mut r_d4) = (zero, zero, zero, zero);
        let (mut r_art, mut r_det, mut r_mse) = (zero, zero, zero);
        let (mut r_hfg, mut r_hfl, mut r_hfm) = (zero, zero, zero);
        let (mut r_pjnd, mut r_pjnd_lo, mut r_pjnd_hi) = (zero, zero, zero);

        // POOL_SIMD lane accumulators (16 — see the function doc). Dead
        // (const-folded away, zero register cost) when POOL_SIMD=false.
        let (mut p_mask_w, mut p_mask_d, mut p_mask_art, mut p_mask_det, mut p_mask_mse) =
            (zero, zero, zero, zero, zero);
        let (mut p_iw_w, mut p_iw_d, mut p_iw_art, mut p_iw_det, mut p_iw_mse) =
            (zero, zero, zero, zero, zero);
        let (
            mut p_sal_d,
            mut p_sal_dv,
            mut p_sal_art,
            mut p_sal_artv,
            mut p_sal_det,
            mut p_sal_detv,
        ) = (zero, zero, zero, zero, zero, zero);

        let mut x = 0usize;
        while x < width8 {
            let i = row + x;
            macro_rules! ld {
                ($plane:expr) => {
                    V8::<T>::from_array_t(token, $plane[i..i + 8].try_into().unwrap())
                };
            }
            let s = ld!(src);
            let dd = ld!(dst);
            let m1 = ld!(mu1);
            let m2 = ld!(mu2);
            let act = ld!(activity);

            let d = ssim_d_local_v(token, m1, m2, ld!(s12), ld!(ssq), c1, c2, direct, stable);
            r_d += d;
            let d2 = d * d;
            r_d2 += d2;
            r_d3 += d2 * d;
            r_d4 += d2 * d2;

            let diff_src = (s - m1).abs();
            let diff_dst = (dd - m2).abs();
            let edge_dissim = bounded_dissim_v(token, diff_src, diff_dst, c_edge, stable);
            let gt = diff_dst.simd_gt(diff_src);
            let lt = diff_dst.simd_lt(diff_src);
            let art_i = V8::<T>::blend(gt, edge_dissim, zero);
            let det_i = V8::<T>::blend(lt, edge_dissim, zero);
            r_art += art_i;
            r_det += det_i;

            let raw_diff = s - dd;
            let raw_sq_err = raw_diff * raw_diff;
            let mse_i = saturate_v(token, raw_sq_err, c_mse);
            r_mse += mse_i;

            let hf_src = s - m1;
            let hf_dst = dd - m2;
            let hf_src_sq = hf_src * hf_src;
            let hf_dst_sq = hf_dst * hf_dst;
            // Phase-6 (§A.16 lever A): hf_gain/hf_loss share one denominator
            // by construction -- one recip instead of two.
            let (hfg_i, hfl_i) = bounded_excess_pair_v(token, hf_dst_sq, hf_src_sq, c_hf);
            r_hfg += hfg_i;
            r_hfl += hfl_i;
            r_hfm += bounded_excess_v(token, hf_src.abs(), hf_dst.abs(), c_hf);

            // Phase-6 (§A.16 lever A): each band's raw-ratio-then-saturate
            // (2 divisions) fuses to one via `pjnd_transducer_v` -- see
            // `pjnd_transducer`'s doc for the derivation.
            let raw_abs_err = raw_diff.abs();
            r_pjnd += pjnd_transducer_v(token, raw_abs_err, act, k_mid, c_pjnd_clamp);
            if transducer_bank {
                r_pjnd_lo += pjnd_transducer_v(token, raw_abs_err, act, k_lo, c_pjnd_clamp);
                r_pjnd_hi += pjnd_transducer_v(token, raw_abs_err, act, k_hi, c_pjnd_clamp);
            }

            if POOL_SIMD {
                // Register-assignment variant: pools in-lane. ONE division
                // yields both family weights (mask_w = 1-sat, iw_w =
                // sat+floor); three more give the soft-peak saliencies —
                // 4 vector divisions per 8 pixels vs 40 scalar f64
                // divisions on the extraction path.
                let sat_act = saturate_v(token, act, c_activity);
                let mask_w = one - sat_act;
                let iw_w = sat_act + iw_floor;
                p_mask_w += mask_w;
                p_mask_d += mask_w * d;
                p_mask_art += mask_w * art_i;
                p_mask_det += mask_w * det_i;
                p_mask_mse += mask_w * mse_i;
                p_iw_w += iw_w;
                p_iw_d += iw_w * d;
                p_iw_art += iw_w * art_i;
                p_iw_det += iw_w * det_i;
                p_iw_mse += iw_w * mse_i;
                let sal_d = saturate_v(token, d, c_peak);
                let sal_art = saturate_v(token, art_i, c_peak);
                let sal_det = saturate_v(token, det_i, c_peak);
                p_sal_d += sal_d;
                p_sal_dv += sal_d * d;
                p_sal_art += sal_art;
                p_sal_artv += sal_art * art_i;
                p_sal_det += sal_det;
                p_sal_detv += sal_det * det_i;
                // REV4 (POOL_SIMD arm): the lane arrays this arm skips are
                // extracted only when the hook is live — one Option check
                // per chunk, identical op sequence when None.
                if let Some(r4) = r4.as_mut() {
                    let d_arr = d.to_array();
                    let art_arr = art_i.to_array();
                    let det_arr = det_i.to_array();
                    let mse_arr = mse_i.to_array();
                    let hfg_arr = hfg_i.to_array();
                    let act_arr = act.to_array();
                    for lane in 0..8 {
                        rev4_dense_pixel(
                            r4,
                            d_arr[lane] as f64,
                            art_arr[lane] as f64,
                            det_arr[lane] as f64,
                            mse_arr[lane] as f64,
                            hfg_arr[lane] as f64,
                            act_arr[lane] as f64,
                            lane,
                        );
                    }
                }
            } else {
                // §A.14 register-pressure fix (see the row-header comment
                // above): extract this chunk's d/art_i/det_i/mse_i/act lanes
                // and accumulate the 11 weighted-pool pairs scalar, per lane
                // -- reuses the EXACT SAME formula as the scalar tail below
                // via `weighted_pool_accumulate_scalar`, not a re-derivation.
                let d_arr = d.to_array();
                let art_arr = art_i.to_array();
                let det_arr = det_i.to_array();
                let mse_arr = mse_i.to_array();
                let act_arr = act.to_array();
                for lane in 0..8 {
                    weighted_pool_accumulate_scalar(
                        &mut acc,
                        d_arr[lane] as f64,
                        art_arr[lane] as f64,
                        det_arr[lane] as f64,
                        mse_arr[lane] as f64,
                        act_arr[lane] as f64,
                    );
                }
                // REV4: same six lanes plus `hfg_i` — the C3 histogram
                // scatters every pixel, C4-noise counts only the flat.
                if let Some(r4) = r4.as_mut() {
                    let hfg_arr = hfg_i.to_array();
                    for lane in 0..8 {
                        rev4_dense_pixel(
                            r4,
                            d_arr[lane] as f64,
                            art_arr[lane] as f64,
                            det_arr[lane] as f64,
                            mse_arr[lane] as f64,
                            hfg_arr[lane] as f64,
                            act_arr[lane] as f64,
                            lane,
                        );
                    }
                }
            }

            x += 8;
        }

        // Reduce this row's f32 lane partials into the f64 running totals.
        acc.sum_d += r_d.reduce_add() as f64;
        acc.sum_d2 += r_d2.reduce_add() as f64;
        acc.sum_d3 += r_d3.reduce_add() as f64;
        acc.sum_d4 += r_d4.reduce_add() as f64;
        acc.sum_art += r_art.reduce_add() as f64;
        acc.sum_det += r_det.reduce_add() as f64;
        acc.sum_mse += r_mse.reduce_add() as f64;
        acc.sum_hf_gain += r_hfg.reduce_add() as f64;
        acc.sum_hf_loss += r_hfl.reduce_add() as f64;
        acc.sum_hf_mag_loss += r_hfm.reduce_add() as f64;
        acc.sum_pjnd += r_pjnd.reduce_add() as f64;
        acc.sum_pjnd_lo += r_pjnd_lo.reduce_add() as f64;
        acc.sum_pjnd_hi += r_pjnd_hi.reduce_add() as f64;
        if POOL_SIMD {
            // Fold the 16 pool lane-sums into the per-pool WeightedSums.
            // The mask/IW family `den`s are the SAME row sum replicated
            // (that sharing is what makes 16 accumulators suffice).
            let mask_w_sum = p_mask_w.reduce_add() as f64;
            acc.ws_mask_ssim.num += p_mask_d.reduce_add() as f64;
            acc.ws_mask_ssim.den += mask_w_sum;
            acc.ws_mask_art.num += p_mask_art.reduce_add() as f64;
            acc.ws_mask_art.den += mask_w_sum;
            acc.ws_mask_det.num += p_mask_det.reduce_add() as f64;
            acc.ws_mask_det.den += mask_w_sum;
            acc.ws_mask_mse.num += p_mask_mse.reduce_add() as f64;
            acc.ws_mask_mse.den += mask_w_sum;
            let iw_w_sum = p_iw_w.reduce_add() as f64;
            acc.ws_iw_ssim.num += p_iw_d.reduce_add() as f64;
            acc.ws_iw_ssim.den += iw_w_sum;
            acc.ws_iw_art.num += p_iw_art.reduce_add() as f64;
            acc.ws_iw_art.den += iw_w_sum;
            acc.ws_iw_det.num += p_iw_det.reduce_add() as f64;
            acc.ws_iw_det.den += iw_w_sum;
            acc.ws_iw_mse.num += p_iw_mse.reduce_add() as f64;
            acc.ws_iw_mse.den += iw_w_sum;
            acc.ws_peak_ssim.num += p_sal_dv.reduce_add() as f64;
            acc.ws_peak_ssim.den += p_sal_d.reduce_add() as f64;
            acc.ws_peak_art.num += p_sal_artv.reduce_add() as f64;
            acc.ws_peak_art.den += p_sal_art.reduce_add() as f64;
            acc.ws_peak_det.num += p_sal_detv.reduce_add() as f64;
            acc.ws_peak_det.den += p_sal_det.reduce_add() as f64;
        }
        // (POOL_SIMD=false: weighted-pool accumulators already folded in
        // scalar, per-lane, inside the chunk loop above -- §A.14 fix note.)

        // Scalar tail: remaining `width % 8` pixels in this row, using the
        // EXACT SAME scalar formulas as the pre-phase-4 kernel (bit-for-bit
        // identical code path — not a re-derivation).
        for x in width8..width {
            let i = row + x;
            let s = src[i] as f64;
            let dd = dst[i] as f64;
            let m1 = mu1[i] as f64;
            let m2 = mu2[i] as f64;
            let act = activity[i] as f64;

            let d = ssim_d_local(m1, m2, s12[i] as f64, ssq[i] as f64, direct, stable);
            acc.sum_d += d;
            acc.sum_d2 += d * d;
            acc.sum_d3 += d * d * d;
            acc.sum_d4 += d * d * d * d;

            let diff_src = (s - m1).abs();
            let diff_dst = (dd - m2).abs();
            let edge_dissim = bounded_dissim(diff_src, diff_dst, C_EDGE, stable);
            let (mut art_i, mut det_i) = (0.0, 0.0);
            if diff_dst > diff_src {
                art_i = edge_dissim;
            } else if diff_dst < diff_src {
                det_i = edge_dissim;
            }
            acc.sum_art += art_i;
            acc.sum_det += det_i;

            let raw_sq_err = (s - dd) * (s - dd);
            let mse_i = saturate(raw_sq_err, C_MSE);
            acc.sum_mse += mse_i;

            let hf_src = s - m1;
            let hf_dst = dd - m2;
            let hf_src_sq = hf_src * hf_src;
            let hf_dst_sq = hf_dst * hf_dst;
            // Phase-6 (§A.16 lever A step 1): shared-denominator pair, one
            // division instead of two (mirrors the SIMD path above).
            let (hf_gain_i, hf_loss_i) = bounded_excess_pair(hf_dst_sq, hf_src_sq, C_HF);
            acc.sum_hf_gain += hf_gain_i;
            acc.sum_hf_loss += hf_loss_i;
            acc.sum_hf_mag_loss += bounded_excess(hf_src.abs(), hf_dst.abs(), C_HF);

            // Phase-6 (§A.16 lever A step 1): fused raw-ratio+saturate, one
            // division per band instead of two (mirrors the SIMD path above).
            let raw_abs_err = (s - dd).abs();
            acc.sum_pjnd += pjnd_transducer(raw_abs_err, act, K_PJND_MASK, C_PJND_CLAMP);
            if transducer_bank {
                acc.sum_pjnd_lo += pjnd_transducer(raw_abs_err, act, K_PJND_MASK_LOW, C_PJND_CLAMP);
                acc.sum_pjnd_hi +=
                    pjnd_transducer(raw_abs_err, act, K_PJND_MASK_HIGH, C_PJND_CLAMP);
            }

            weighted_pool_accumulate_scalar(&mut acc, d, art_i, det_i, mse_i, act);
            // REV4: the tail's f64 values go through the same per-pixel
            // helper — no re-derivation.
            if let Some(r4) = r4.as_mut() {
                rev4_dense_pixel(r4, d, art_i, det_i, mse_i, hf_gain_i, act, x);
            }
        }
    }

    acc
}

// Two disjoint-tier blocks emit the same base name (the
// `downscale_2x_into_inner` pattern): AVX-512(VL) has 32 SIMD registers,
// so the v4x tier runs the POOL_SIMD=true body (13 core + 16 pool lane
// accumulators live comfortably); every 16-register tier keeps the §A.14
// scalar-pool body, whose extraction fix exists precisely because 22+
// live vectors spill there.
#[magetypes(+v4x, -v4, -v3, -neon, -wasm128, -scalar)]
#[allow(clippy::too_many_arguments)]
fn dense_block_kernel_entry(
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
    dense_block_kernel_generic::<_, true>(
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
#[allow(clippy::too_many_arguments)]
fn dense_block_kernel_entry(
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
    dense_block_kernel_generic::<_, false>(
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

/// Test/measurement-only dispatch that FORCES the §A.14 scalar-pool body
/// on every tier — the baseline side of the POOL_SIMD drift + timing
/// comparisons (`pool_simd_drift_within_policy`, `v2_pool_simd_ab` bench).
#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
fn dense_block_kernel_entry_pools_scalar(
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
    dense_block_kernel_generic::<_, false>(
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

/// Forced-scalar-pool sibling of [`dense_block_kernel`] (see
/// [`dense_block_kernel_entry_pools_scalar`]).
#[allow(clippy::too_many_arguments, dead_code)]
fn dense_block_kernel_pools_scalar(
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
        dense_block_kernel_entry_pools_scalar(
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

#[allow(clippy::too_many_arguments)]
/// **The era-2 accumulation switch.** `ZENSIM_ERA2_DENSE=1` routes every
/// production dense call through [`dense_block_kernel_era2`] — the fixed
/// 8-virtual-lane accumulation with the `era2_reduce8` pairwise tree and
/// `ERA2_BAND_ROWS` bands — instead of era-1's per-row f64 reduction.
///
/// **It is on the ENTRY, for the reason §27 established the hard way.** The
/// column tile was first applied at selected call sites and silently split the
/// v1 reference path from the fold; a switch that some dense call sites honour
/// and others do not is the same defect. There are four production dense sites
/// (the materialized strip walk, the whole-image bypass, `stream_phase_b`, and
/// the plain-v2 walk) and this is the only place they can disagree.
///
/// Until this existed the era-2 kernel was **unreachable outside `mod tests`**
/// — every call site was a test or `#[cfg(feature = "oracle")]` — so the
/// accumulation could not be rank-checked at all. That was an instrument gap,
/// not an evidence gap.
///
/// Tests that compare the two accumulations must call
/// [`dense_block_kernel_era1`] / [`dense_block_kernel_era2`] explicitly, so
/// their comparison is not silently turned into an identity by the switch.
#[inline]
fn era2_dense_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    // ERA-2 DEFAULT (2026-08-31): ON. `ZENSIM_ERA2_DENSE=0` restores era-1's
    // per-row f64 reduction for measurement only.
    *ON.get_or_init(|| std::env::var("ZENSIM_ERA2_DENSE").as_deref() != Ok("0"))
}

/// The production dense entry: dispatches on [`era2_dense_enabled`].
/// `r4`: REV4's C3 tail-histogram / C4 flat-HF-gain hook bundle — `None`
/// on every non-rev4 call (one loop-invariant Option check per chunk,
/// the `transducer_bank` shape).
#[cfg(feature = "oracle")]
#[allow(clippy::too_many_arguments)]
fn audit_rev5_moments(
    acc: &DenseAccum,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    act: &[f32],
    n: usize,
    bank: bool,
) {
    if std::env::var("ZENSIM_FEATCANON_MOMENT_CHECK").as_deref() != Ok("1") {
        return;
    }
    // Independent two-pass f64 ruler over the SAME f32 signal leaves.
    // The exact arithmetic axis separately measures leaf error against f64.
    let values: Vec<f64> = (0..n)
        .map(|i| {
            dense_terms32(
                src[i], dst[i], mu1[i], mu2[i], ssq[i], s12[i], act[i], bank, true, false,
            )
            .0[0] as f64
        })
        .collect();
    let mean = values.iter().sum::<f64>() / n as f64;
    let m2 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>();
    let m4 = values.iter().map(|x| (x - mean).powi(4)).sum::<f64>();
    let mut max_relative = 0.0_f64;
    for (got, reference) in [(acc.central.m2, m2), (acc.central.m4, m4)] {
        let relative = (got - reference).abs() / reference.abs().max(1e-30);
        assert!(
            relative <= 1e-6,
            "Rev5 moment {n} samples: {got} vs {reference}, relative={relative}"
        );
        max_relative = max_relative.max(relative);
    }
    eprintln!(
        "REV5_MOMENT_GATE pixels={n} max_relative={max_relative:.9e} dev4={:.12e}",
        (m4 / n as f64).sqrt().sqrt()
    );
}

/// Production dense evaluator for exact local-refinement strip replacement.
#[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn refinement_dense_strip(
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
) -> DenseAccum {
    dense_block_kernel(
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
}

/// Production gradient evaluator with its one-row halo contract.
#[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
pub(crate) fn refinement_gradient_strip(
    src: &[f32],
    dst: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
) -> GradientAccum {
    gradient_block_kernel(src, dst, activity, width, height, None, None, None, false)
}

#[allow(clippy::too_many_arguments)]
fn dense_block_kernel(
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
    dense_block_kernel_at_revision(
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
        crate::ssim_form::active_revision(),
    )
}

#[allow(clippy::too_many_arguments)]
fn dense_block_kernel_at_revision(
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
    // rev4canon: the computation's own mode selects the scalar
    // `Pool`-generic canon body — `LanesF64` (c64) at Rev4, the oracle arms
    // under `ZENSIM_FEATCANON` (`LanesF32` replays era-2 bitwise — the
    // superseded c32 canon). `compute_mode()` is `mode(comp_rev)` here (see
    // it for the guarantee); `Off` at every revision < 4 keeps the shipped
    // era-1/era-2 dispatch byte-for-byte.
    let mode = crate::featcanon::mode(revision);
    if revision >= FormulaRevision::Rev5 && matches!(mode, crate::featcanon::Mode::Canon64) {
        let acc = if revision.row_lanes() {
            incant!(
                dense_block_kernel_canon64_vec::<crate::featcanon::Lanes16F64, true>(
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
                ),
                [v4x, v4, v3, neon, wasm128, scalar]
            )
        } else {
            incant!(
                dense_block_kernel_canon64_vec::<crate::featcanon::Lanes16F64, false>(
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
                ),
                [v4x, v4, v3, neon, wasm128, scalar]
            )
        };
        #[cfg(feature = "oracle")]
        audit_rev5_moments(
            &acc,
            src,
            dst,
            mu1,
            mu2,
            ssq,
            s12,
            activity,
            width * height,
            transducer_bank,
        );
        return acc;
    }
    if mode.active() {
        match mode {
            crate::featcanon::Mode::Canon64 => {
                // canon64: the `LanesF64` canonical arm — tier-dispatched;
                // scalar/wasm128 keep the generic scalar body.
                return incant!(
                    dense_block_kernel_canon64_vec::<crate::featcanon::LanesF64, false>(
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
                    ),
                    [v4x, v4, v3, neon, wasm128, scalar]
                );
            }
            #[cfg(feature = "oracle")]
            crate::featcanon::Mode::Canon32 => {
                return dense_block_kernel_canon::<crate::featcanon::LanesF32>(
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
                );
            }
            #[cfg(feature = "oracle")]
            crate::featcanon::Mode::CanonNeum => {
                return dense_block_kernel_canon::<crate::featcanon::Neum64>(
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
                );
            }
            #[cfg(feature = "oracle")]
            crate::featcanon::Mode::Exact => {
                return dense_block_kernel_exact(
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
                );
            }
            crate::featcanon::Mode::Off => unreachable!("active() excludes Off"),
        }
    }
    if era2_dense_enabled() {
        return dense_block_kernel_era2(
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
        );
    }
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
        r4,
    )
}

/// Era-1's per-row-reduced accumulation — the shipped path, and the explicit
/// arm for any test that means "era 1" rather than "whatever is switched on".
#[allow(clippy::too_many_arguments)]
fn dense_block_kernel_era1(
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
        dense_block_kernel_entry(
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

/// Co-sited B samples for the X gradient walk. Packed f32 XYB B, one sample
/// per pixel, with the same width/height as the current strip (no halo).
#[derive(Clone, Copy)]
struct ChromaStrip<'a> {
    reference: &'a [f32],
    distorted: &'a [f32],
}

#[derive(Clone, Copy)]
struct GmsBankWork<'a> {
    gradient: &'static [f64; 5],
    chroma: Option<ChromaStrip<'a>>,
}

impl<'a> GmsBankWork<'a> {
    fn new(channel: usize, chroma: Option<ChromaStrip<'a>>) -> Self {
        Self {
            gradient: match channel {
                0 => &GMSBANK_X_C,
                1 => &GMSBANK_C,
                2 => &GMSBANK_B_C,
                _ => unreachable!(),
            },
            chroma,
        }
    }

    fn rows(self, start: usize, end: usize) -> Self {
        Self {
            chroma: self.chroma.map(|p| ChromaStrip {
                reference: &p.reference[start..end],
                distorted: &p.distorted[start..end],
            }),
            ..self
        }
    }
}

/// Difference-form joint opponent loss. Equal constants reproduce 1-MDSI CS
/// on the author's unshifted H/M coordinates. XYB uses its declared centering.
/// Called inside the existing magetypes/arcane gradient entries; the generic
/// backend cannot carry a standalone rite token attribute.
#[inline(always)]
fn chromaticity_loss(reference: [f64; 2], distorted: [f64; 2], constants: [f64; 2]) -> f64 {
    let [xr, br] = reference;
    let [xd, bd] = distorted;
    let [cx, cb] = constants;
    let dx = xr - xd;
    let db = br - bd;
    (dx * dx / cx + db * db / cb)
        / (xr * xr / cx + xd * xd / cx + br * br / cb + bd * bd / cb + 1.0)
}

/// C8 per-constant pair of signed gradient-change means and a Welford
/// population variance of `1-GMS`. Variance is unchanged by the translation
/// `GMS = 1 - delta`, while this form makes identity exactly zero.
#[derive(Default, Clone, Copy)]
struct GmsBankCell {
    loss: f64,
    gain: f64,
    n: u64,
    mean: f64,
    m2: f64,
}

impl GmsBankCell {
    #[inline]
    fn push(&mut self, delta: f64, loss: bool) {
        if loss {
            self.loss += delta;
        } else {
            self.gain += delta;
        }
        self.n += 1;
        // std(GMS) == std(1-GMS) exactly. Welford on delta=1-GMS
        // preserves small deviations that would round away in 1-delta.
        let change = delta - self.mean;
        self.mean += change / self.n as f64;
        self.m2 += change * (delta - self.mean);
    }

    #[inline]
    fn merge(&mut self, other: &Self) {
        if other.n == 0 {
            return;
        }
        self.loss += other.loss;
        self.gain += other.gain;
        if self.n == 0 {
            self.n = other.n;
            self.mean = other.mean;
            self.m2 = other.m2;
            return;
        }
        let total = self.n + other.n;
        let shift = other.mean - self.mean;
        self.m2 += other.m2 + shift * shift * (self.n as f64 * other.n as f64 / total as f64);
        self.mean += shift * (other.n as f64 / total as f64);
        self.n = total;
    }
}

#[inline(always)]
fn gmsbank_pixel(cells: &mut [GmsBankCell; 5], mr: f64, md: f64, constants: &[f64; 5]) {
    let diff = mr - md;
    let numer = diff * diff;
    let denom_base = mr * mr + md * md;
    for (cell, c) in cells.iter_mut().zip(constants) {
        cell.push(numer / (denom_base + c), md < mr);
    }
}

/// featacc: measurement-mode sibling of [`GmsBankCell`] — the `loss`/`gain`
/// sums ride [`SumVar::f64_elems`] (f64-native deltas: `c32` stays
/// sequential so the f32 column never conflates element precision with
/// accumulation order) and the Welford `(n, mean, m2)` triple rides
/// [`WelfordVar`]. Resolves to the production `GmsBankCell` shape at the
/// strip boundary so `GradientAccum::accumulate`'s plain-f64 fold is
/// unchanged for every mode.
struct GmsBankCellVar {
    loss: crate::featcanon::SumVar,
    gain: crate::featcanon::SumVar,
    wf: crate::featcanon::WelfordVar,
}

impl GmsBankCellVar {
    #[inline]
    fn for_mode(mode: crate::featcanon::Mode) -> Self {
        Self {
            loss: crate::featcanon::SumVar::f64_elems(mode),
            gain: crate::featcanon::SumVar::f64_elems(mode),
            wf: crate::featcanon::WelfordVar::for_mode(mode),
        }
    }

    /// [`GmsBankCell::push`] with the canonical lane.
    #[inline]
    fn push(&mut self, lane: usize, delta: f64, is_loss: bool) {
        if is_loss {
            self.loss.add64(lane, delta);
        } else {
            self.gain.add64(lane, delta);
        }
        self.wf.push(lane, delta);
    }

    /// [`GmsBankCell::merge`] against a same-mode row cell.
    #[inline]
    fn merge_var(&mut self, o: &Self) {
        self.loss.merge_from(&o.loss);
        self.gain.merge_from(&o.gain);
        self.wf.merge_var(&o.wf);
    }

    /// Resolve to the production cell shape (`n`, `mean`, `m2` from
    /// [`WelfordVar::stats`]).
    #[inline]
    fn resolve(&self) -> GmsBankCell {
        let s = self.wf.stats();
        GmsBankCell {
            loss: self.loss.fin(),
            gain: self.gain.fin(),
            n: s.n,
            mean: s.mean,
            m2: s.m2,
        }
    }
}

/// [`gmsbank_pixel`] for the var cells — identical element math, canonical
/// lane forwarded.
#[inline(always)]
fn gmsbank_pixel_var(
    cells: &mut [GmsBankCellVar; 5],
    mr: f64,
    md: f64,
    constants: &[f64; 5],
    lane: usize,
) {
    let diff = mr - md;
    let numer = diff * diff;
    let denom_base = mr * mr + md * md;
    for (cell, c) in cells.iter_mut().zip(constants) {
        cell.push(lane, numer / (denom_base + c), md < mr);
    }
}

/// Per-row-reduced f64 accumulator for the gradient block.
#[derive(Default, Clone, Copy)]
pub(crate) struct GradientAccum {
    pub(crate) sum_gms: f64,
    /// Σ gms² — second raw moment of the SAME per-pixel gms values, for
    /// the append block's GMSD-style deviation pooling
    /// ([`idx_append::GMS_DEV2`]). Accumulated unconditionally (the
    /// marginal cost is one FMA per pixel on values already in
    /// registers); only consumed when the append block is on. Adding it
    /// does not perturb `sum_gms` — the shared `g` is computed by the
    /// identical operations in the identical order.
    pub(crate) sum_gms2: f64,
    pub(crate) sum_ringing: f64,
    pub(crate) sum_banding: f64,
    pub(crate) sum_grad_src: f64,
    pub(crate) sum_grad_dst: f64,
    /// append2 BANDVIS sums — accumulated ONLY by the `BANDVIS = true`
    /// kernel instantiation (Y channel with `append2_block` on); zero and
    /// untouched everywhere else, so the existing lanes' operations are
    /// bit-identical when append2 is off.
    sum_bv_gain: f64,
    sum_bv_loss: f64,
    bank: [GmsBankCell; 5],
    chroma: [GmsBankCell; 5],
}

impl GradientAccum {
    /// Phase-5 (§A.15): fold one strip's `GradientAccum` into the running
    /// whole-image total — same reasoning as `DenseAccum::accumulate`.
    #[inline]
    pub(crate) fn accumulate(&mut self, other: &GradientAccum) {
        self.sum_gms += other.sum_gms;
        self.sum_gms2 += other.sum_gms2;
        self.sum_ringing += other.sum_ringing;
        self.sum_banding += other.sum_banding;
        self.sum_grad_src += other.sum_grad_src;
        self.sum_grad_dst += other.sum_grad_dst;
        self.sum_bv_gain += other.sum_bv_gain;
        self.sum_bv_loss += other.sum_bv_loss;
        for (mine, next) in self.bank.iter_mut().zip(&other.bank) {
            mine.merge(next);
        }
        for (mine, next) in self.chroma.iter_mut().zip(&other.chroma) {
            mine.merge(next);
        }
    }
}

fn finish_gmsbank_cell(bank: &[GmsBankCell; 5], n_px: usize, out: &mut [f64]) {
    debug_assert_eq!(out.len(), GMSBANK_PER_CELL);
    for (k, cell) in bank.iter().enumerate() {
        assert_eq!(cell.n as usize, n_px, "C8 gradient sample count");
        let base = k * 3;
        out[base] = cell.loss / n_px as f64;
        out[base + 1] = cell.gain / n_px as f64;
        out[base + 2] = (cell.m2.max(0.0) / n_px as f64).sqrt();
    }
}

fn finish_chroma_cell(cells: &[GmsBankCell; 5], n: usize, out: &mut [f64]) {
    assert_eq!(out.len(), 10);
    for (k, cell) in cells.iter().enumerate() {
        assert_eq!(cell.n as usize, n, "C8 chromaticity sample count");
        out[2 * k] = cell.loss / n as f64;
        out[2 * k + 1] = (cell.m2.max(0.0) / n as f64).sqrt();
    }
}

/// Gradient-block SIMD kernel body. Phase-5 (§A.15) NEW CONTRACT: `src_h`/
/// `dst_h` carry ONE HALO ROW top and bottom (`width*(height+2)` total —
/// local row `y+1` is the strip's real row `y`; local rows `0` and
/// `height+1` are the caller-provided neighbor context, real data for an
/// interior strip or a properly reflected row at the true image edge —
/// see `gather_strip_halo`). This REMOVES the old `y==0`/`y==height-1`
/// special-casing entirely: every output row now uses the identical
/// "interior" halo-relative indexing, since the caller guarantees valid
/// neighbor rows for ALL of them, not just strip-interior ones.
/// `activity` is NOT halo-padded (`width*height`, exactly the strip) — it
/// is read only at each pixel's own row, never a neighbor row (confirmed
/// by inspection: only `ld_at!(activity, row + x)`/`activity[i]` below,
/// no `row_u`/`row_d` indexing into it).
///
/// Interior pixels (`1..width-1`) are vectorized via SHIFTED unaligned
/// loads for the x-neighbors (`row[x-1..x+7]` / `row[x+1..x+9]`) and
/// full-row-offset loads for the y-neighbors (`row_u[x..x+8]` /
/// `row_d[x..x+8]`) — the same shifted-window-read pattern as
/// `~/work/archmage/docs/site/content/magetypes/examples/gaussian-blur.md`'s
/// vertical pass. The single-pixel column border (`x==0`, `x==width-1`)
/// still uses the scalar
/// reflect-boundary formulas (`saturating_sub`/`.min(width-1)`) exactly as
/// before — a tiny fraction of pixels (2 columns + 2 rows out of
/// width*height), not worth the complexity of a SIMD boundary-clamp.
/// `BV_DSTACT` (BANDVIS dst-activity combine, `append2_dst_activity`):
/// when true, `act_dst` carries the distorted-side activity plane (same
/// shape as `activity` — `width*height`, no halo, own-row reads only) and
/// the BANDVIS GAIN polarity becomes the pure-band FR excess pooled under
/// the dst's own flatness weight `1 − sat(act_dst, C_ACTIVITY)` (LOSS
/// keeps the toggle-off math bit-exactly — see the adjudication record in
/// `benchmarks/bandvis_dst_activity_2026-08-02.md` for why both
/// in-ratio masking arms were rejected). Only meaningful with
/// `BANDVIS = true`; with `BV_DSTACT = false` the parameter is dead
/// (`&[]`) and the emitted operation sequence is the pre-fix one, bit
/// for bit.
#[inline]
fn gradient_block_kernel_generic<
    T: F32x8Backend + Copy,
    const BANDVIS: bool,
    const BV_DSTACT: bool,
    const BANK: bool,
>(
    token: T,
    src_h: &[f32],
    dst_h: &[f32],
    activity: &[f32],
    act_dst: &[f32],
    width: usize,
    height: usize,
    bv_delta_lo: f32,
    bv_delta_hi: f32,
    mut r4: Option<Rev4Grad<'_>>,
) -> GradientAccum {
    // Revision 7's cancellation-free forms, by the process revision.
    let stable = crate::ssim_form::active_revision().stable_forms();
    debug_assert_eq!(src_h.len(), width * (height + 2));
    debug_assert_eq!(dst_h.len(), width * (height + 2));
    debug_assert_eq!(activity.len(), width * height);
    if let Some((m, _)) = r4.as_ref().and_then(|r| r.bleed.as_ref()) {
        debug_assert_eq!(m.len(), width * height);
    }
    if BV_DSTACT {
        debug_assert!(BANDVIS, "BV_DSTACT is a BANDVIS refinement");
        debug_assert_eq!(act_dst.len(), width * height);
    }

    let zero = V8::<T>::zero_t(token);
    let one = V8::<T>::splat_t(token, 1.0);
    let c_gms = V8::<T>::splat_t(token, C_GMS as f32);
    let c_ring_err = V8::<T>::splat_t(token, C_RING_ERR as f32);
    let c_activity = V8::<T>::splat_t(token, C_ACTIVITY as f32);
    let c_ring_edge = V8::<T>::splat_t(token, C_RING_EDGE as f32);
    let c_band_dst = V8::<T>::splat_t(token, C_BAND_DST as f32);
    let c_band_src = V8::<T>::splat_t(token, C_BAND_SRC as f32);
    // append2 BANDVIS constants (dead splats when `BANDVIS == false`).
    let bv_lo = V8::<T>::splat_t(token, bv_delta_lo);
    let bv_hi = V8::<T>::splat_t(token, bv_delta_hi);
    let c_bv = V8::<T>::splat_t(token, C_BV as f32);

    let mut acc = GradientAccum::default();

    let bank = r4
        .as_ref()
        .map_or_else(|| GmsBankWork::new(1, None), |r| r.bank);
    if let Some(chroma) = bank.chroma {
        debug_assert_eq!(chroma.reference.len(), width * height);
        debug_assert_eq!(chroma.distorted.len(), width * height);
    }
    let chroma_pixel = |x: usize, y: usize, cells: &mut [GmsBankCell; 5]| {
        if let Some(chroma) = bank.chroma {
            let i = y * width + x;
            let h = i + width;
            let reference = [
                f64::from(src_h[h]) - f64::from(0.42_f32),
                f64::from(chroma.reference[i]) - f64::from(0.55_f32),
            ];
            let distorted = [
                f64::from(dst_h[h]) - f64::from(0.42_f32),
                f64::from(chroma.distorted[i]) - f64::from(0.55_f32),
            ];
            for (cell, constants) in cells.iter_mut().zip(GMSBANK_CS_C) {
                cell.push(chromaticity_loss(reference, distorted, constants), true);
            }
        }
    };

    // Scalar helper for one pixel — used for the first/last COLUMN of
    // every row (x-axis boundary only; the y-axis "first/last row"
    // special case from phase 4 is GONE, per this function's new halo
    // contract — every row `y` has valid `row_u`/`row_d` neighbor data
    // in `src_h`/`dst_h` by construction, either real interior-strip
    // rows or a caller-supplied reflected true-edge row).
    let scalar_pixel = |x: usize,
                        y: usize,
                        acc: &mut GradientAccum,
                        r4: &mut Option<Rev4Grad<'_>>,
                        bank_row: &mut [GmsBankCell; 5],
                        chroma_row: &mut [GmsBankCell; 5]| {
        let xl = x.saturating_sub(1);
        let xr = (x + 1).min(width - 1);
        let row = (y + 1) * width; // +1: src_h/dst_h carry 1 halo row up front
        let row_u = y * width; // y-1, in halo-relative terms
        let row_d = (y + 2) * width; // y+1, in halo-relative terms
        let i = row + x;
        let s = src_h[i] as f64;
        let dd = dst_h[i] as f64;
        let act = activity[y * width + x] as f64;
        let sxl = src_h[row + xl] as f64;
        let sxr = src_h[row + xr] as f64;
        let syu = src_h[row_u + x] as f64;
        let syd = src_h[row_d + x] as f64;
        let dxl = dst_h[row + xl] as f64;
        let dxr = dst_h[row + xr] as f64;
        let dyu = dst_h[row_u + x] as f64;
        let dyd = dst_h[row_d + x] as f64;

        let gx_src = sxr - sxl;
        let gy_src = syd - syu;
        let grad_src_mag = (gx_src * gx_src + gy_src * gy_src).sqrt();
        let gx_dst = dxr - dxl;
        let gy_dst = dyd - dyu;
        let grad_dst_mag = (gx_dst * gx_dst + gy_dst * gy_dst).sqrt();

        if BANK {
            gmsbank_pixel(bank_row, grad_src_mag, grad_dst_mag, bank.gradient);
            chroma_pixel(x, y, chroma_row);
        }

        acc.sum_grad_src += grad_src_mag;
        acc.sum_grad_dst += grad_dst_mag;
        let g = bounded_dissim(grad_src_mag, grad_dst_mag, C_GMS, stable);
        acc.sum_gms += g;
        acc.sum_gms2 += g * g;

        let raw_abs_err = (s - dd).abs();
        let err_b = saturate(raw_abs_err, C_RING_ERR);
        let act_b = saturate(act, C_ACTIVITY);
        let edge_r = saturate(grad_src_mag, C_RING_EDGE);
        let ring_i = err_b * act_b * (1.0 - edge_r);
        acc.sum_ringing += ring_i;
        // REV4: C2's ring bins + C4's outside-mask bleed — same values,
        // scalar per lane like the SIMD arm's `to_array()` tail.
        if let Some(r4) = r4.as_mut() {
            rev4_grad_pixel(
                r4,
                &RINGBASIS_UC,
                ring_i,
                grad_src_mag,
                grad_dst_mag,
                y * width + x,
                x & 7,
            );
        }

        let edge_excess = bounded_excess(grad_dst_mag, grad_src_mag, C_BAND_DST);
        let src_smooth_b = 1.0 - saturate(grad_src_mag, C_BAND_SRC);
        acc.sum_banding += edge_excess * src_smooth_b;

        if BANDVIS {
            // Soft CURVATURE band-pass × flatness mask, FR excess pair
            // (see `idx_append2::BANDVIS_GAIN`). Second differences, not
            // first: a linear gradient of ANY steepness has |∇²| = 0, so
            // smooth ramps never enter the band (measured to be the
            // load-bearing property — a first-difference band could not
            // separate sub-step smooth gradients from steps; both
            // polarities mis-fired on ramp fixtures). A plateau step
            // reports the FULL step in |∇²| at its flanking pixels, so
            // the one-code-step δ derivation carries verbatim. Scalar
            // mirrors the SIMD formula exactly.
            let d2x_src = sxl + sxr - 2.0 * s;
            let d2y_src = syu + syd - 2.0 * s;
            let curv_src = (d2x_src * d2x_src + d2y_src * d2y_src).sqrt();
            let d2x_dst = dxl + dxr - 2.0 * dd;
            let d2y_dst = dyu + dyd - 2.0 * dd;
            let curv_dst = (d2x_dst * d2x_dst + d2y_dst * d2y_dst).sqrt();
            let flat = 1.0 - act_b;
            let band = |g: f64| -> f64 {
                saturate(g, bv_delta_lo as f64) * (1.0 - saturate(g, bv_delta_hi as f64))
            };
            if BV_DSTACT {
                // `append2_dst_activity` SHIPPED combine (adjudication:
                // `benchmarks/bandvis_dst_activity_2026-08-02.md`).
                // GAIN = arm-2 visibility-weighted POOLING: the FR excess
                // on the PURE band terms, weighted by the DST's own
                // flatness OUTSIDE the ratio — the only place a flatness
                // mask survives `bounded_excess`'s scale-invariance
                // (arm 1, flat multiplied INSIDE the pair, measured
                // ratio-cancelled: it suppressed real banding MORE than
                // dither). Measured: lattice cross-fire 0.33×, deband
                // margin 2.2× the OFF margin.
                // LOSS = the OFF math BIT-EXACTLY (identical expressions,
                // identical order): the arm-2 weight measured
                // direction-INVERTING on the deband credit (the banded
                // src's own contours zero the very pixels whose removal
                // should be credited), and LOSS is the LYB-validated
                // workhorse — it must not move.
                let flat_d = 1.0 - saturate(act_dst[y * width + x] as f64, C_ACTIVITY);
                let b_src = band(curv_src) * flat;
                let b_dst = band(curv_dst) * flat;
                let (_, loss) = bounded_excess_pair(b_dst, b_src, C_BV);
                let (g0, _) = bounded_excess_pair(band(curv_dst), band(curv_src), C_BV);
                acc.sum_bv_gain += g0 * flat_d;
                acc.sum_bv_loss += loss;
            } else {
                let b_src = band(curv_src) * flat;
                let b_dst = band(curv_dst) * flat;
                let (gain, loss) = bounded_excess_pair(b_dst, b_src, C_BV);
                acc.sum_bv_gain += gain;
                acc.sum_bv_loss += loss;
            }
        }
    };

    for y in 0..height {
        let mut bank_row = [GmsBankCell::default(); 5];
        let mut chroma_row = [GmsBankCell::default(); 5];
        let row = (y + 1) * width;
        let row_u = y * width;
        let row_d = (y + 2) * width;
        let act_row = y * width;

        scalar_pixel(0, y, &mut acc, &mut r4, &mut bank_row, &mut chroma_row);
        if width > 2 {
            let interior_end = width - 1;
            let interior_w = interior_end - 1; // pixels [1, width-2]
            let chunk_end = 1 + interior_w - (interior_w % 8);

            let (mut r_gms, mut r_gms2, mut r_ring, mut r_band) = (zero, zero, zero, zero);
            let (mut r_gsrc, mut r_gdst) = (zero, zero);
            let (mut r_bv_gain, mut r_bv_loss) = (zero, zero);

            let mut x = 1usize;
            while x < chunk_end {
                macro_rules! ld_at {
                    ($plane:expr, $off:expr) => {
                        V8::<T>::from_array_t(token, $plane[$off..$off + 8].try_into().unwrap())
                    };
                }
                let sxl = ld_at!(src_h, row + x - 1);
                let sxr = ld_at!(src_h, row + x + 1);
                let syu = ld_at!(src_h, row_u + x);
                let syd = ld_at!(src_h, row_d + x);
                let dxl = ld_at!(dst_h, row + x - 1);
                let dxr = ld_at!(dst_h, row + x + 1);
                let dyu = ld_at!(dst_h, row_u + x);
                let dyd = ld_at!(dst_h, row_d + x);
                let s = ld_at!(src_h, row + x);
                let dd = ld_at!(dst_h, row + x);
                let act = ld_at!(activity, act_row + x);

                let gx_src = sxr - sxl;
                let gy_src = syd - syu;
                let grad_src_mag = (gx_src * gx_src + gy_src * gy_src).sqrt();
                let gx_dst = dxr - dxl;
                let gy_dst = dyd - dyu;
                let grad_dst_mag = (gx_dst * gx_dst + gy_dst * gy_dst).sqrt();

                if BANK {
                    let src_mag = grad_src_mag.to_array();
                    let dst_mag = grad_dst_mag.to_array();
                    for lane in 0..8 {
                        gmsbank_pixel(
                            &mut bank_row,
                            f64::from(src_mag[lane]),
                            f64::from(dst_mag[lane]),
                            bank.gradient,
                        );
                        chroma_pixel(x + lane, y, &mut chroma_row);
                    }
                }

                r_gsrc += grad_src_mag;
                r_gdst += grad_dst_mag;
                let g = bounded_dissim_v(token, grad_src_mag, grad_dst_mag, c_gms, stable);
                r_gms += g;
                r_gms2 += g * g;

                let raw_abs_err = (s - dd).abs();
                let err_b = saturate_v(token, raw_abs_err, c_ring_err);
                let act_b = saturate_v(token, act, c_activity);
                let edge_r = saturate_v(token, grad_src_mag, c_ring_edge);
                let ring_i = err_b * act_b * (one - edge_r);
                r_ring += ring_i;
                // REV4: C2 ring bins + C4 outside-mask bleed — scalarized
                // per lane (the §A.14 register-pressure pattern; the
                // u-domain bit lookup cannot vectorize anyway).
                if let Some(r4) = r4.as_mut() {
                    let ring_a = ring_i.to_array();
                    let gs_a = grad_src_mag.to_array();
                    let gd_a = grad_dst_mag.to_array();
                    for lane in 0..8 {
                        rev4_grad_pixel(
                            r4,
                            &RINGBASIS_UC,
                            ring_a[lane] as f64,
                            gs_a[lane] as f64,
                            gd_a[lane] as f64,
                            act_row + x + lane,
                            lane,
                        );
                    }
                }

                let edge_excess = bounded_excess_v(token, grad_dst_mag, grad_src_mag, c_band_dst);
                let src_smooth_b = one - saturate_v(token, grad_src_mag, c_band_src);
                r_band += edge_excess * src_smooth_b;

                if BANDVIS {
                    // Curvature band (see the scalar sibling's rationale).
                    let two = one + one;
                    let d2x_s = sxl + sxr - two * s;
                    let d2y_s = syu + syd - two * s;
                    let curv_s = (d2x_s * d2x_s + d2y_s * d2y_s).sqrt();
                    let d2x_d = dxl + dxr - two * dd;
                    let d2y_d = dyu + dyd - two * dd;
                    let curv_d = (d2x_d * d2x_d + d2y_d * d2y_d).sqrt();
                    let flat = one - act_b;
                    let band_s =
                        saturate_v(token, curv_s, bv_lo) * (one - saturate_v(token, curv_s, bv_hi));
                    let band_d =
                        saturate_v(token, curv_d, bv_lo) * (one - saturate_v(token, curv_d, bv_hi));
                    if BV_DSTACT {
                        // SHIPPED combine (see the scalar sibling): GAIN
                        // = pure-band FR excess × dst flatness (pooling
                        // weight); LOSS = the OFF math bit-exactly.
                        let actd = ld_at!(act_dst, act_row + x);
                        let flat_d = one - saturate_v(token, actd, c_activity);
                        let b_src = band_s * flat;
                        let b_dst = band_d * flat;
                        let (_, loss) = bounded_excess_pair_v(token, b_dst, b_src, c_bv);
                        let (g0, _) = bounded_excess_pair_v(token, band_d, band_s, c_bv);
                        r_bv_gain += g0 * flat_d;
                        r_bv_loss += loss;
                    } else {
                        let b_src = band_s * flat;
                        let b_dst = band_d * flat;
                        let (gain, loss) = bounded_excess_pair_v(token, b_dst, b_src, c_bv);
                        r_bv_gain += gain;
                        r_bv_loss += loss;
                    }
                }

                x += 8;
            }

            acc.sum_gms += r_gms.reduce_add() as f64;
            acc.sum_gms2 += r_gms2.reduce_add() as f64;
            acc.sum_ringing += r_ring.reduce_add() as f64;
            acc.sum_banding += r_band.reduce_add() as f64;
            acc.sum_grad_src += r_gsrc.reduce_add() as f64;
            acc.sum_grad_dst += r_gdst.reduce_add() as f64;
            if BANDVIS {
                acc.sum_bv_gain += r_bv_gain.reduce_add() as f64;
                acc.sum_bv_loss += r_bv_loss.reduce_add() as f64;
            }

            for x in chunk_end..=interior_end - 1 {
                scalar_pixel(x, y, &mut acc, &mut r4, &mut bank_row, &mut chroma_row);
            }
        }
        scalar_pixel(
            width - 1,
            y,
            &mut acc,
            &mut r4,
            &mut bank_row,
            &mut chroma_row,
        );
        if BANK {
            for (sum, row) in acc.bank.iter_mut().zip(&bank_row) {
                sum.merge(row);
            }
            for (sum, row) in acc.chroma.iter_mut().zip(&chroma_row) {
                sum.merge(row);
            }
        }
    }

    acc
}

#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
fn gradient_block_kernel_entry(
    token: Token,
    src: &[f32],
    dst: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    r4: Option<Rev4Grad<'_>>,
) -> GradientAccum {
    gradient_block_kernel_generic::<_, false, false, false>(
        token,
        src,
        dst,
        activity,
        &[],
        width,
        height,
        0.0,
        0.0,
        r4,
    )
}

#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
fn gradient_block_kernel_entry_bandvis(
    token: Token,
    src: &[f32],
    dst: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    bv_delta_lo: f32,
    bv_delta_hi: f32,
    r4: Option<Rev4Grad<'_>>,
) -> GradientAccum {
    gradient_block_kernel_generic::<_, true, false, false>(
        token,
        src,
        dst,
        activity,
        &[],
        width,
        height,
        bv_delta_lo,
        bv_delta_hi,
        r4,
    )
}

#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
fn gradient_block_kernel_entry_bandvis_dstact(
    token: Token,
    src: &[f32],
    dst: &[f32],
    activity: &[f32],
    act_dst: &[f32],
    width: usize,
    height: usize,
    bv_delta_lo: f32,
    bv_delta_hi: f32,
    r4: Option<Rev4Grad<'_>>,
) -> GradientAccum {
    gradient_block_kernel_generic::<_, true, true, false>(
        token,
        src,
        dst,
        activity,
        act_dst,
        width,
        height,
        bv_delta_lo,
        bv_delta_hi,
        r4,
    )
}

#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
fn gradient_block_kernel_entry_gmsbank(
    token: Token,
    src: &[f32],
    dst: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    r4: Option<Rev4Grad<'_>>,
) -> GradientAccum {
    gradient_block_kernel_generic::<_, false, false, true>(
        token,
        src,
        dst,
        activity,
        &[],
        width,
        height,
        0.0,
        0.0,
        r4,
    )
}

#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
fn gradient_block_kernel_entry_bandvis_gmsbank(
    token: Token,
    src: &[f32],
    dst: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    bv_delta_lo: f32,
    bv_delta_hi: f32,
    r4: Option<Rev4Grad<'_>>,
) -> GradientAccum {
    gradient_block_kernel_generic::<_, true, false, true>(
        token,
        src,
        dst,
        activity,
        &[],
        width,
        height,
        bv_delta_lo,
        bv_delta_hi,
        r4,
    )
}

#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
fn gradient_block_kernel_entry_bandvis_dstact_gmsbank(
    token: Token,
    src: &[f32],
    dst: &[f32],
    activity: &[f32],
    act_dst: &[f32],
    width: usize,
    height: usize,
    bv_delta_lo: f32,
    bv_delta_hi: f32,
    r4: Option<Rev4Grad<'_>>,
) -> GradientAccum {
    gradient_block_kernel_generic::<_, true, true, true>(
        token,
        src,
        dst,
        activity,
        act_dst,
        width,
        height,
        bv_delta_lo,
        bv_delta_hi,
        r4,
    )
}

// ============================================================================
// featacc gradient candidates — measurement-only bodies
// ============================================================================
//
// The production kernel accumulates interior pixels in per-row f32 SIMD
// lanes closed by `reduce_add()` (a TIER-DEPENDENT reduction tree) and the
// two border columns + tail in sequential f64; the C8 bank/chroma cells
// are sequential-f64 `GmsBankCell`s merged per row. The mirrors below put
// the interior accumulation on the `Pool` axis (canonical lane `x mod 8`,
// fixed `era2_reduce8`-shaped close) and the bank/chroma cells on
// [`GmsBankCellVar`]. Borders stay sequential f64 in every mode — they are
// already the exact element form, so only their bank-cell pushes ride the
// mode axis.

/// One gradient pixel's full term vector in f64 — the production
/// `scalar_pixel` element math verbatim (clamped x-neighbors, so it serves
/// both the border columns and the exact interior).
pub(crate) struct GradTerms64 {
    pub(crate) gsrc: f64,
    pub(crate) gdst: f64,
    pub(crate) g: f64,
    pub(crate) g2: f64,
    pub(crate) ring: f64,
    pub(crate) band: f64,
    pub(crate) bv_gain: f64,
    pub(crate) bv_loss: f64,
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn gradient_terms64<const BANDVIS: bool, const BV_DSTACT: bool, const HPR: bool>(
    src_h: &[f32],
    dst_h: &[f32],
    activity: &[f32],
    act_dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    width: usize,
    x: usize,
    y: usize,
    bv_lo: f64,
    bv_hi: f64,
    stable: bool,
) -> GradTerms64 {
    let xl = x.saturating_sub(1);
    let xr = (x + 1).min(width - 1);
    let row = (y + 1) * width;
    let row_u = y * width;
    let row_d = (y + 2) * width;
    let i = row + x;
    let s = src_h[i] as f64;
    let dd = dst_h[i] as f64;
    let act = activity[y * width + x] as f64;
    let sxl = src_h[row + xl] as f64;
    let sxr = src_h[row + xr] as f64;
    let syu = src_h[row_u + x] as f64;
    let syd = src_h[row_d + x] as f64;
    let dxl = dst_h[row + xl] as f64;
    let dxr = dst_h[row + xr] as f64;
    let dyu = dst_h[row_u + x] as f64;
    let dyd = dst_h[row_d + x] as f64;

    let gx_src = sxr - sxl;
    let gy_src = syd - syu;
    let gsrc = (gx_src * gx_src + gy_src * gy_src).sqrt();
    let gx_dst = dxr - dxl;
    let gy_dst = dyd - dyu;
    let gdst = (gx_dst * gx_dst + gy_dst * gy_dst).sqrt();
    let g = bounded_dissim(gsrc, gdst, C_GMS, stable);

    // `RingForm::HighPassMu` (HPR): the error with each image's local mean removed.
    let raw_abs_err = if HPR {
        let k = y * width + x;
        ((s - mu1[k] as f64) - (dd - mu2[k] as f64)).abs()
    } else {
        (s - dd).abs()
    };
    let err_b = saturate(raw_abs_err, C_RING_ERR);
    let act_b = saturate(act, C_ACTIVITY);
    let edge_r = saturate(gsrc, C_RING_EDGE);
    let ring = err_b * act_b * (1.0 - edge_r);

    let edge_excess = bounded_excess(gdst, gsrc, C_BAND_DST);
    let src_smooth_b = 1.0 - saturate(gsrc, C_BAND_SRC);
    let band = edge_excess * src_smooth_b;

    let (bv_gain, bv_loss) = if BANDVIS {
        let d2x_src = sxl + sxr - 2.0 * s;
        let d2y_src = syu + syd - 2.0 * s;
        let curv_src = (d2x_src * d2x_src + d2y_src * d2y_src).sqrt();
        let d2x_dst = dxl + dxr - 2.0 * dd;
        let d2y_dst = dyu + dyd - 2.0 * dd;
        let curv_dst = (d2x_dst * d2x_dst + d2y_dst * d2y_dst).sqrt();
        let flat = 1.0 - act_b;
        let bandf = |g: f64| -> f64 { saturate(g, bv_lo) * (1.0 - saturate(g, bv_hi)) };
        if BV_DSTACT {
            let flat_d = 1.0 - saturate(act_dst[y * width + x] as f64, C_ACTIVITY);
            let b_src = bandf(curv_src) * flat;
            let b_dst = bandf(curv_dst) * flat;
            let (_, loss) = bounded_excess_pair(b_dst, b_src, C_BV);
            let (g0, _) = bounded_excess_pair(bandf(curv_dst), bandf(curv_src), C_BV);
            (g0 * flat_d, loss)
        } else {
            let b_src = bandf(curv_src) * flat;
            let b_dst = bandf(curv_dst) * flat;
            bounded_excess_pair(b_dst, b_src, C_BV)
        }
    } else {
        (0.0, 0.0)
    };

    GradTerms64 {
        gsrc,
        gdst,
        g,
        g2: g * g,
        ring,
        band,
        bv_gain,
        bv_loss,
    }
}

/// f32 scalar helpers mirroring the `*_v` SIMD ops' exact op order.
#[inline(always)]
fn sat32(x: f32, c: f32) -> f32 {
    let x = x.max(0.0);
    x / (x + c)
}
#[inline(always)]
fn bsim32(a: f32, b: f32, c: f32) -> f32 {
    (2.0f32 * a * b + c) / (a * a + b * b + c)
}
#[inline(always)]
fn bex32(a: f32, b: f32, c: f32) -> f32 {
    (a - b).max(0.0) / (a + b + c)
}
#[inline(always)]
fn bex_pair32(a: f32, b: f32, c: f32) -> (f32, f32) {
    let d = a + b + c;
    ((a - b).max(0.0) / d, (b - a).max(0.0) / d)
}

/// Shared border-pixel step for the canon/exact bodies: f64 terms folded
/// straight into `acc` (production's scalar-border shape), plus the bank /
/// chroma / rev4 side effects at canonical lane `x mod 8`.
#[allow(clippy::too_many_arguments)]
fn gradient_border_pixel<
    const BANDVIS: bool,
    const BV_DSTACT: bool,
    const BANK: bool,
    const HPR: bool,
    const R7: bool,
>(
    src_h: &[f32],
    dst_h: &[f32],
    activity: &[f32],
    act_dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    width: usize,
    x: usize,
    y: usize,
    bv_lo: f32,
    bv_hi: f32,
    acc: &mut GradientAccum,
    r4: &mut Option<Rev4Grad<'_>>,
    bank: GmsBankWork<'_>,
    bank_row: &mut [GmsBankCellVar; 5],
    chroma_row: &mut [GmsBankCellVar; 5],
) {
    let t = gradient_terms64::<BANDVIS, BV_DSTACT, HPR>(
        src_h,
        dst_h,
        activity,
        act_dst,
        mu1,
        mu2,
        width,
        x,
        y,
        bv_lo as f64,
        bv_hi as f64,
        R7,
    );
    let lane = x & 7;
    if BANK {
        gmsbank_pixel_var(bank_row, t.gsrc, t.gdst, bank.gradient, lane);
        if let Some(chroma) = bank.chroma {
            let i = y * width + x;
            let h = i + width;
            let reference = [
                f64::from(src_h[h]) - f64::from(0.42_f32),
                f64::from(chroma.reference[i]) - f64::from(0.55_f32),
            ];
            let distorted = [
                f64::from(dst_h[h]) - f64::from(0.42_f32),
                f64::from(chroma.distorted[i]) - f64::from(0.55_f32),
            ];
            for (cell, constants) in chroma_row.iter_mut().zip(GMSBANK_CS_C) {
                cell.push(
                    lane,
                    chromaticity_loss(reference, distorted, constants),
                    true,
                );
            }
        }
    }
    acc.sum_grad_src += t.gsrc;
    acc.sum_grad_dst += t.gdst;
    acc.sum_gms += t.g;
    acc.sum_gms2 += t.g2;
    acc.sum_ringing += t.ring;
    if let Some(r4) = r4.as_mut() {
        rev4_grad_pixel(
            r4,
            &RINGBASIS_UC,
            t.ring,
            t.gsrc,
            t.gdst,
            y * width + x,
            x & 7,
        );
    }
    acc.sum_banding += t.band;
    if BANDVIS {
        acc.sum_bv_gain += t.bv_gain;
        acc.sum_bv_loss += t.bv_loss;
    }
}

/// One interior pixel's contribution to the gradient canonical pools —
/// the scalar element step shared by [`gradient_block_kernel_canon`]'s
/// `Pool`-generic body and the canon64 vector body's lane-1..8 head and
/// `interior_end % 8` tail. Extracted verbatim from the canonical body.
#[allow(clippy::too_many_arguments)]
fn gradient_interior_elem_canon<
    P: crate::featcanon::Pool,
    // The pools revision 7 keeps in f32 row lanes (`P` otherwise).
    Q: crate::featcanon::Pool,
    const BANDVIS: bool,
    const BV_DSTACT: bool,
    const BANK: bool,
    const HPR: bool,
    const R7: bool,
>(
    src_h: &[f32],
    dst_h: &[f32],
    activity: &[f32],
    act_dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    width: usize,
    row: usize,
    row_u: usize,
    row_d: usize,
    act_row: usize,
    x: usize,
    bv_lo: f32,
    bv_hi: f32,
    bank: GmsBankWork<'_>,
    bank_row: &mut [GmsBankCellVar; 5],
    chroma_row: &mut [GmsBankCellVar; 5],
    r4: &mut Option<Rev4Grad<'_>>,
    r_gsrc: &mut Q,
    r_gdst: &mut Q,
    r_gms: &mut Q,
    r_gms2: &mut Q,
    r_ring: &mut Q,
    r_band: &mut Q,
    r_bv_gain: &mut Q,
    r_bv_loss: &mut Q,
    // Revision 7's f32 row lanes: `r_gdst` takes `gdst − gsrc` (exact when
    // the two are within a factor of 2), and the fold adds `r_gsrc`'s lanes to
    // both sums, so the edge-width decay sees the difference summed directly.
    gdiff: bool,
) {
    let lane = x & (P::LANES - 1);
    let sxl = src_h[row + x - 1];
    let sxr = src_h[row + x + 1];
    let syu = src_h[row_u + x];
    let syd = src_h[row_d + x];
    let dxl = dst_h[row + x - 1];
    let dxr = dst_h[row + x + 1];
    let dyu = dst_h[row_u + x];
    let dyd = dst_h[row_d + x];
    let s = src_h[row + x];
    let dd = dst_h[row + x];
    let act = activity[act_row + x];

    let gx_src = sxr - sxl;
    let gy_src = syd - syu;
    let gsrc = (gx_src * gx_src + gy_src * gy_src).sqrt();
    let gx_dst = dxr - dxl;
    let gy_dst = dyd - dyu;
    let gdst = (gx_dst * gx_dst + gy_dst * gy_dst).sqrt();

    if BANK {
        gmsbank_pixel_var(
            bank_row,
            f64::from(gsrc),
            f64::from(gdst),
            bank.gradient,
            lane,
        );
        if let Some(chroma) = bank.chroma {
            let i = act_row + x;
            let h = i + width;
            let reference = [
                f64::from(src_h[h]) - f64::from(0.42_f32),
                f64::from(chroma.reference[i]) - f64::from(0.55_f32),
            ];
            let distorted = [
                f64::from(dst_h[h]) - f64::from(0.42_f32),
                f64::from(chroma.distorted[i]) - f64::from(0.55_f32),
            ];
            for (cell, constants) in chroma_row.iter_mut().zip(GMSBANK_CS_C) {
                cell.push(
                    lane,
                    chromaticity_loss(reference, distorted, constants),
                    true,
                );
            }
        }
    }

    r_gsrc.add(lane, gsrc);
    r_gdst.add(lane, if gdiff { gdst - gsrc } else { gdst });
    // Revision 7: `bounded_dissim_v`'s exact `(a−b)²/(a²+b²+c)`, same op order.
    let g = if R7 {
        crate::feature_v2::bounded_dissim32(gsrc, gdst, C_GMS as f32, true)
    } else {
        1.0f32 - bsim32(gsrc, gdst, C_GMS as f32)
    };
    r_gms.add(lane, g);
    r_gms2.add(lane, g * g);

    let raw_abs_err = if HPR {
        ((s - mu1[act_row + x]) - (dd - mu2[act_row + x])).abs()
    } else {
        (s - dd).abs()
    };
    let err_b = sat32(raw_abs_err, C_RING_ERR as f32);
    let act_b = sat32(act, C_ACTIVITY as f32);
    let edge_r = sat32(gsrc, C_RING_EDGE as f32);
    let ring_i = err_b * act_b * (1.0f32 - edge_r);
    r_ring.add(lane, ring_i);
    if let Some(r4) = r4.as_mut() {
        rev4_grad_pixel(
            r4,
            &RINGBASIS_UC,
            ring_i as f64,
            gsrc as f64,
            gdst as f64,
            act_row + x,
            x & 7,
        );
    }

    let edge_excess = bex32(gdst, gsrc, C_BAND_DST as f32);
    let src_smooth_b = 1.0f32 - sat32(gsrc, C_BAND_SRC as f32);
    r_band.add(lane, edge_excess * src_smooth_b);

    if BANDVIS {
        let two = 2.0f32;
        let d2x_s = sxl + sxr - two * s;
        let d2y_s = syu + syd - two * s;
        let curv_s = (d2x_s * d2x_s + d2y_s * d2y_s).sqrt();
        let d2x_d = dxl + dxr - two * dd;
        let d2y_d = dyu + dyd - two * dd;
        let curv_d = (d2x_d * d2x_d + d2y_d * d2y_d).sqrt();
        let flat = 1.0f32 - act_b;
        let band_s = sat32(curv_s, bv_lo) * (1.0f32 - sat32(curv_s, bv_hi));
        let band_d = sat32(curv_d, bv_lo) * (1.0f32 - sat32(curv_d, bv_hi));
        if BV_DSTACT {
            let actd = act_dst[act_row + x];
            let flat_d = 1.0f32 - sat32(actd, C_ACTIVITY as f32);
            let b_src = band_s * flat;
            let b_dst = band_d * flat;
            let (_, loss) = bex_pair32(b_dst, b_src, C_BV as f32);
            let (g0, _) = bex_pair32(band_d, band_s, C_BV as f32);
            r_bv_gain.add(lane, g0 * flat_d);
            r_bv_loss.add(lane, loss);
        } else {
            let b_src = band_s * flat;
            let b_dst = band_d * flat;
            let (gain, loss) = bex_pair32(b_dst, b_src, C_BV as f32);
            r_bv_gain.add(lane, gain);
            r_bv_loss.add(lane, loss);
        }
    }
}

/// Add the gradient kernel's eight interior pools' finished sums to `acc`
/// (`[gms, gms2, ring, band, gsrc, gdst, bv_gain, bv_loss]`), in the fixed
/// order: once per row, or at revision 7 once per call.
#[inline(always)]
fn gradient_pools_finish<P: crate::featcanon::Pool, const BANDVIS: bool>(
    acc: &mut GradientAccum,
    pools: &[P; 8],
) {
    let [gms, gms2, ring, band, gsrc, gdst, bv_gain, bv_loss] = *pools;
    acc.sum_gms += gms.fin();
    acc.sum_gms2 += gms2.fin();
    acc.sum_ringing += ring.fin();
    acc.sum_banding += band.fin();
    acc.sum_grad_src += gsrc.fin();
    acc.sum_grad_dst += gdst.fin();
    if BANDVIS {
        acc.sum_bv_gain += bv_gain.fin();
        acc.sum_bv_loss += bv_loss.fin();
    }
}

/// `Pool`-generic scalar gradient kernel — f32 elements in the SIMD op
/// order, per-row canonical pools, f64 borders.
#[allow(clippy::too_many_arguments)]
fn gradient_block_kernel_canon<
    P: crate::featcanon::Pool,
    const BANDVIS: bool,
    const BV_DSTACT: bool,
    const BANK: bool,
    const HPR: bool,
    const R7: bool,
>(
    src_h: &[f32],
    dst_h: &[f32],
    activity: &[f32],
    act_dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    width: usize,
    height: usize,
    bv_lo: f32,
    bv_hi: f32,
    mut r4: Option<Rev4Grad<'_>>,
    mode: crate::featcanon::Mode,
) -> GradientAccum {
    let mut acc = GradientAccum::default();
    let bank = r4
        .as_ref()
        .map_or_else(|| GmsBankWork::new(1, None), |r| r.bank);
    let mut bank_acc: [GmsBankCellVar; 5] = std::array::from_fn(|_| GmsBankCellVar::for_mode(mode));
    let mut chroma_acc: [GmsBankCellVar; 5] =
        std::array::from_fn(|_| GmsBankCellVar::for_mode(mode));
    // Revision 7: the interior pools' f64 lanes live for the whole call (each
    // row's f32 row lanes fold into them) and finish once, after the last row.
    let mut call_pools = [P::zero(); 8];
    for y in 0..height {
        let row = (y + 1) * width;
        let row_u = y * width;
        let row_d = (y + 2) * width;
        let act_row = y * width;
        let mut bank_row: [GmsBankCellVar; 5] =
            std::array::from_fn(|_| GmsBankCellVar::for_mode(mode));
        let mut chroma_row: [GmsBankCellVar; 5] =
            std::array::from_fn(|_| GmsBankCellVar::for_mode(mode));

        gradient_border_pixel::<BANDVIS, BV_DSTACT, BANK, HPR, R7>(
            src_h,
            dst_h,
            activity,
            act_dst,
            mu1,
            mu2,
            width,
            0,
            y,
            bv_lo,
            bv_hi,
            &mut acc,
            &mut r4,
            bank,
            &mut bank_row,
            &mut chroma_row,
        );
        if width > 2 {
            let interior_end = width - 1;
            // `[gms, gms2, ring, band, gsrc, gdst, bv_gain, bv_loss]`.
            let mut fresh = [P::zero(); 8];
            let [
                r_gms,
                r_gms2,
                r_ring,
                r_band,
                r_gsrc,
                r_gdst,
                r_bv_gain,
                r_bv_loss,
            ] = if R7 { &mut call_pools } else { &mut fresh };
            // Revision 7: the vector kernel's chunk columns (`head_end` to
            // `chunk_end`) add the six non-cancelling pools in f32 row lanes
            // (`crate::featcanon::Lanes16F32`), folded every `ROW_LANES_FOLD`
            // columns and at the end of the chunks; head and tail stay f64.
            let head_end = interior_end.min(8);
            let chunk_end = head_end + (interior_end - head_end) / 8 * 8;
            let mut q32 = [crate::featcanon::Lanes16F32::default(); 8];
            for x in 1..interior_end {
                if R7 && (head_end..chunk_end).contains(&x) {
                    let [q_gms, q_gms2, q_ring, q_band, q_bvg, q_bvl, q_gsrc, q_gdst] = &mut q32;
                    gradient_interior_elem_canon::<
                        P,
                        crate::featcanon::Lanes16F32,
                        BANDVIS,
                        BV_DSTACT,
                        BANK,
                        HPR,
                        R7,
                    >(
                        src_h,
                        dst_h,
                        activity,
                        act_dst,
                        mu1,
                        mu2,
                        width,
                        row,
                        row_u,
                        row_d,
                        act_row,
                        x,
                        bv_lo,
                        bv_hi,
                        bank,
                        &mut bank_row,
                        &mut chroma_row,
                        &mut r4,
                        q_gsrc,
                        q_gdst,
                        q_gms,
                        q_gms2,
                        q_ring,
                        q_band,
                        q_bvg,
                        q_bvl,
                        true,
                    );
                    if (x + 1).is_multiple_of(crate::featcanon::ROW_LANES_FOLD) || x + 1 == chunk_end {
                        let [q_gms, q_gms2, q_ring, q_band, q_bvg, q_bvl, q_gsrc, q_gdst] = &mut q32;
                        q_gms.fold_into(&mut *r_gms);
                        q_gms2.fold_into(&mut *r_gms2);
                        q_ring.fold_into(&mut *r_ring);
                        q_band.fold_into(&mut *r_band);
                        q_bvg.fold_into(&mut *r_bv_gain);
                        q_bvl.fold_into(&mut *r_bv_loss);
                        // The difference pair: the source lanes to both sums.
                        q_gsrc.add_into(&mut *r_gsrc);
                        q_gsrc.fold_into(&mut *r_gdst);
                        q_gdst.fold_into(&mut *r_gdst);
                    }
                    continue;
                }
                gradient_interior_elem_canon::<P, P, BANDVIS, BV_DSTACT, BANK, HPR, R7>(
                    src_h,
                    dst_h,
                    activity,
                    act_dst,
                    mu1,
                    mu2,
                    width,
                    row,
                    row_u,
                    row_d,
                    act_row,
                    x,
                    bv_lo,
                    bv_hi,
                    bank,
                    &mut bank_row,
                    &mut chroma_row,
                    &mut r4,
                    &mut *r_gsrc,
                    &mut *r_gdst,
                    &mut *r_gms,
                    &mut *r_gms2,
                    &mut *r_ring,
                    &mut *r_band,
                    &mut *r_bv_gain,
                    &mut *r_bv_loss,
                    false,
                );
            }
            if !R7 {
                gradient_pools_finish::<P, BANDVIS>(&mut acc, &fresh);
            }
        }
        gradient_border_pixel::<BANDVIS, BV_DSTACT, BANK, HPR, R7>(
            src_h,
            dst_h,
            activity,
            act_dst,
            mu1,
            mu2,
            width,
            width - 1,
            y,
            bv_lo,
            bv_hi,
            &mut acc,
            &mut r4,
            bank,
            &mut bank_row,
            &mut chroma_row,
        );
        if BANK {
            for (sum, row) in bank_acc.iter_mut().zip(&bank_row) {
                sum.merge_var(row);
            }
            for (sum, row) in chroma_acc.iter_mut().zip(&chroma_row) {
                sum.merge_var(row);
            }
        }
    }
    if R7 && width > 2 {
        gradient_pools_finish::<P, BANDVIS>(&mut acc, &call_pools);
    }
    if BANK {
        for (dst, src) in acc.bank.iter_mut().zip(&bank_acc) {
            *dst = src.resolve();
        }
        for (dst, src) in acc.chroma.iter_mut().zip(&chroma_acc) {
            *dst = src.resolve();
        }
    }
    acc
}

/// The `f32x8` sibling of [`gradient_interior_elem_canon`]'s term
/// evaluation — identical op order on the existing `*_v` helpers
/// (`saturate_v` ≡ `sat32`, `bounded_sim_v` ≡ `bsim32`,
/// `bounded_excess_v`/`bounded_excess_pair_v` ≡ `bex32`/`bex_pair32`; all
/// plain unfused ops, so every lane IS one scalar element's rounding).
/// The BANK / REV4 side effects are NOT in here — they stay scalarized
/// per element in the caller (histogram scatters and Welford pushes
/// cannot vectorize).
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))] // only the x86_64 lane-parallel canon bodies call it
#[inline(always)]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn gradient_terms32_v<
    V: Vf32,
    const BANDVIS: bool,
    const BV_DSTACT: bool,
    const HPR: bool,
>(
    token: V::Tok,
    sxl: V,
    sxr: V,
    syu: V,
    syd: V,
    dxl: V,
    dxr: V,
    dyu: V,
    dyd: V,
    s: V,
    dd: V,
    act: V,
    actd: V,
    m1: V,
    m2: V,
    bv_lo: f32,
    bv_hi: f32,
    stable: bool,
) -> (V, V, V, V, V, V, V, V) {
    let c_gms = V::splat_t(token, C_GMS as f32);
    let c_ring_err = V::splat_t(token, C_RING_ERR as f32);
    let c_act = V::splat_t(token, C_ACTIVITY as f32);
    let c_ring_edge = V::splat_t(token, C_RING_EDGE as f32);
    let c_band_dst = V::splat_t(token, C_BAND_DST as f32);
    let c_band_src = V::splat_t(token, C_BAND_SRC as f32);
    let c_bv = V::splat_t(token, C_BV as f32);
    let bv_lo_v = V::splat_t(token, bv_lo);
    let bv_hi_v = V::splat_t(token, bv_hi);
    let one = V::splat_t(token, 1.0);
    let zero = V::zero_t(token);

    let gx_src = sxr - sxl;
    let gy_src = syd - syu;
    let gsrc = (gx_src * gx_src + gy_src * gy_src).sqrt();
    let gx_dst = dxr - dxl;
    let gy_dst = dyd - dyu;
    let gdst = (gx_dst * gx_dst + gy_dst * gy_dst).sqrt();

    let g = bounded_dissim_v(token, gsrc, gdst, c_gms, stable);
    let g2 = g * g;

    let raw_abs_err = if HPR {
        ((s - m1) - (dd - m2)).abs()
    } else {
        (s - dd).abs()
    };
    let err_b = saturate_v(token, raw_abs_err, c_ring_err);
    let act_b = saturate_v(token, act, c_act);
    let edge_r = saturate_v(token, gsrc, c_ring_edge);
    let ring_i = err_b * act_b * (one - edge_r);

    let edge_excess = bounded_excess_v(token, gdst, gsrc, c_band_dst);
    let src_smooth_b = one - saturate_v(token, gsrc, c_band_src);
    let band = edge_excess * src_smooth_b;

    let (bv_gain, bv_loss) = if BANDVIS {
        let two = V::splat_t(token, 2.0);
        let d2x_s = sxl + sxr - two * s;
        let d2y_s = syu + syd - two * s;
        let curv_s = (d2x_s * d2x_s + d2y_s * d2y_s).sqrt();
        let d2x_d = dxl + dxr - two * dd;
        let d2y_d = dyu + dyd - two * dd;
        let curv_d = (d2x_d * d2x_d + d2y_d * d2y_d).sqrt();
        let flat = one - act_b;
        let band_s =
            saturate_v(token, curv_s, bv_lo_v) * (one - saturate_v(token, curv_s, bv_hi_v));
        let band_d =
            saturate_v(token, curv_d, bv_lo_v) * (one - saturate_v(token, curv_d, bv_hi_v));
        if BV_DSTACT {
            let flat_d = one - saturate_v(token, actd, c_act);
            let b_src = band_s * flat;
            let b_dst = band_d * flat;
            let (_, loss) = bounded_excess_pair_v(token, b_dst, b_src, c_bv);
            let (g0, _) = bounded_excess_pair_v(token, band_d, band_s, c_bv);
            (g0 * flat_d, loss)
        } else {
            let b_src = band_s * flat;
            let b_dst = band_d * flat;
            bounded_excess_pair_v(token, b_dst, b_src, c_bv)
        }
    } else {
        (zero, zero)
    };

    (gsrc, gdst, g, g2, ring_i, band, bv_gain, bv_loss)
}

/// **canon64 vector body** — [`gradient_block_kernel_canon`]'s
/// `P = LanesF64` arm with the interior evaluated eight columns at a time
/// through `f32x8`. The interior `1..width-1` runs a lane-aligned head:
/// `1..8` scalar via [`gradient_interior_elem_canon`], then `x ≡ 0 (mod 8)`
/// chunks where chunk lane `l` is canonical lane `l` exactly (each
/// `add_chunk` is the same eight `add(l, v)` calls in the same per-pool
/// order), then the remainder scalar. `x - 1`/`x + 1` neighbors load
/// unaligned via `from_array` — the same shifted-window-read pattern the
/// production interior already uses. `gmsbank_pixel_var`, the chroma
/// pushes and `rev4_grad_pixel` stay scalar per element. `_scalar` and
/// `_wasm128` keep the scalar canonical body.
#[magetypes(define(f32x8, f32x16), v4x, v4, v3, -scalar)]
#[allow(clippy::too_many_arguments)]
fn gradient_block_kernel_canon64_vec<
    P: crate::featcanon::Pool64,
    const BANDVIS: bool,
    const BV_DSTACT: bool,
    const BANK: bool,
    const HPR: bool,
    const R7: bool,
>(
    token: Token,
    src_h: &[f32],
    dst_h: &[f32],
    activity: &[f32],
    act_dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    width: usize,
    height: usize,
    bv_lo: f32,
    bv_hi: f32,
    r4: Option<Rev4Grad<'_>>,
    mode: crate::featcanon::Mode,
) -> GradientAccum {
    let mut r4 = r4;
    let mut acc = GradientAccum::default();
    let bank = r4
        .as_ref()
        .map_or_else(|| GmsBankWork::new(1, None), |r| r.bank);
    let mut bank_acc: [GmsBankCellVar; 5] = std::array::from_fn(|_| GmsBankCellVar::for_mode(mode));
    let mut chroma_acc: [GmsBankCellVar; 5] =
        std::array::from_fn(|_| GmsBankCellVar::for_mode(mode));
    // Revision 7 (`rowlanes`): the chunks' adds of the six non-cancelling
    // pools go to f32 row lanes, folded into the f64 lanes every
    // `ROW_LANES_FOLD` columns and at the end of the chunks (the scalar body's
    // `head_end..chunk_end` region). Revisions 5 and 6 keep their bits.
    let rev7 = R7;
    macro_rules! gpool_add {
        ($pool:ident, $pool32:ident, $x0:expr, $v:expr) => {{
            let v: f32x8 = $v;
            if rev7 {
                // `$x0` is a multiple of 8: its bit 3 picks the lane half.
                let lanes = &mut $pool32.as_chunks_mut::<8>().0[($x0 >> 3) & 1];
                (f32x8::load_t(token, lanes) + v).store(lanes);
            } else {
                let a = v.to_array();
                let lanes = $pool.chunk8($x0 & (P::LANES - 1));
                for l in 0..8 {
                    lanes[l] += a[l] as f64;
                }
            }
        }};
    }
    // The gradient sums feed a ratio (edge-width change) and keep f64 lanes.
    macro_rules! gpool_add64 {
        ($pool:ident, $x0:expr, $v:expr) => {{
            let a = ($v).to_array();
            let lanes = $pool.chunk8($x0 & (P::LANES - 1));
            for l in 0..8 {
                lanes[l] += a[l] as f64;
            }
        }};
    }
    // `Lanes16F32::fold_into`'s op per lane (`lane += v as f64`); `clear`
    // empties the f32 lanes for the row's next columns.
    macro_rules! gpool_fold {
        (clear; $($pool:ident, $pool32:ident);*) => {{
            $(
                gpool_fold!($pool, $pool32);
                $pool32 = [0.0f32; 16];
            )*
        }};
        (last; $($pool:ident, $pool32:ident);*) => {{
            $(
                gpool_fold!($pool, $pool32);
            )*
        }};
        // The difference pair: the source lanes to both sums, then the
        // difference lanes to the distorted sum.
        (pair; $src:ident, $src32:ident, $dst:ident, $dst32:ident) => {{
            gpool_fold!($src, $src32);
            gpool_fold!($dst, $src32);
            gpool_fold!($dst, $dst32);
        }};
        ($pool:ident, $pool32:ident) => {{
            for off in [0usize, 8] {
                let lanes = $pool.chunk8(off);
                for l in 0..8 {
                    lanes[l] += $pool32[off + l] as f64;
                }
            }
        }};
    }
    // Revision 7: the interior pools' f64 lanes live for the whole call (each
    // row's f32 row lanes fold into them) and finish once, after the last row.
    let mut call_pools = [P::zero(); 8];
    for y in 0..height {
        let row = (y + 1) * width;
        let row_u = y * width;
        let row_d = (y + 2) * width;
        let act_row = y * width;
        let mut bank_row: [GmsBankCellVar; 5] =
            std::array::from_fn(|_| GmsBankCellVar::for_mode(mode));
        let mut chroma_row: [GmsBankCellVar; 5] =
            std::array::from_fn(|_| GmsBankCellVar::for_mode(mode));

        gradient_border_pixel::<BANDVIS, BV_DSTACT, BANK, HPR, R7>(
            src_h,
            dst_h,
            activity,
            act_dst,
            mu1,
            mu2,
            width,
            0,
            y,
            bv_lo,
            bv_hi,
            &mut acc,
            &mut r4,
            bank,
            &mut bank_row,
            &mut chroma_row,
        );
        if width > 2 {
            let interior_end = width - 1;
            // `[gms, gms2, ring, band, gsrc, gdst, bv_gain, bv_loss]`.
            let mut fresh = [P::zero(); 8];
            let [
                r_gms,
                r_gms2,
                r_ring,
                r_band,
                r_gsrc,
                r_gdst,
                r_bv_gain,
                r_bv_loss,
            ] = if R7 { &mut call_pools } else { &mut fresh };
            let (mut r_gms32, mut r_gms232, mut r_ring32, mut r_band32) =
                ([0.0f32; 16], [0.0f32; 16], [0.0f32; 16], [0.0f32; 16]);
            let (mut r_bv_gain32, mut r_bv_loss32) = ([0.0f32; 16], [0.0f32; 16]);
            // Revision 7's difference pair: `gsrc` and `gdst − gsrc`.
            let (mut r_gsrc32, mut r_gdst32) = ([0.0f32; 16], [0.0f32; 16]);
            // Lane-aligned head: x = 1..8 (interior starts one pixel in — the
            // canonical lanes land on chunk lanes only for x ≡ 0 (mod 8) chunk
            // starts). Head and tail add in the f64 lanes; without the bank
            // paths and in rows of at least ten columns they are evaluated as
            // one eight-wide step each (`edge8!`), whose lanes are the scalar
            // element's values, added in ascending x.
            let head_end = interior_end.min(8);
            let vec_edges = !BANK && r4.is_none() && interior_end >= 9;
            macro_rules! edge8 {
                ($xs:expr, $from:expr, $to:expr) => {{
                    let xs: usize = $xs;
                    macro_rules! ld8 {
                        ($pl:expr, $o:expr) => {{
                            let o: usize = $o;
                            f32x8::load_t(token, $pl[o..o + 8].try_into().unwrap())
                        }};
                    }
                    let actd = if BV_DSTACT {
                        ld8!(act_dst, act_row + xs)
                    } else {
                        f32x8::zero_t(token)
                    };
                    let (m1, m2) = if HPR {
                        (ld8!(mu1, act_row + xs), ld8!(mu2, act_row + xs))
                    } else {
                        (f32x8::zero_t(token), f32x8::zero_t(token))
                    };
                    let (gsrc, gdst, g, g2, ring_i, band, bv_gain, bv_loss) =
                        gradient_terms32_v::<_, BANDVIS, BV_DSTACT, HPR>(
                            token,
                            ld8!(src_h, row + xs - 1),
                            ld8!(src_h, row + xs + 1),
                            ld8!(src_h, row_u + xs),
                            ld8!(src_h, row_d + xs),
                            ld8!(dst_h, row + xs - 1),
                            ld8!(dst_h, row + xs + 1),
                            ld8!(dst_h, row_u + xs),
                            ld8!(dst_h, row_d + xs),
                            ld8!(src_h, row + xs),
                            ld8!(dst_h, row + xs),
                            ld8!(activity, act_row + xs),
                            actd,
                            m1,
                            m2,
                            bv_lo,
                            bv_hi,
                            R7,
                        );
                    let (gs, gd, ga, gb) = (gsrc.to_array(), gdst.to_array(), g.to_array(), g2.to_array());
                    let (ri, ba) = (ring_i.to_array(), band.to_array());
                    let (bg, bl) = (bv_gain.to_array(), bv_loss.to_array());
                    for x in $from..$to {
                        let (l, lane) = (x - xs, x & (P::LANES - 1));
                        r_gsrc.add(lane, gs[l]);
                        r_gdst.add(lane, gd[l]);
                        r_gms.add(lane, ga[l]);
                        r_gms2.add(lane, gb[l]);
                        r_ring.add(lane, ri[l]);
                        r_band.add(lane, ba[l]);
                        if BANDVIS {
                            r_bv_gain.add(lane, bg[l]);
                            r_bv_loss.add(lane, bl[l]);
                        }
                    }
                }};
            }
            if vec_edges {
                edge8!(1, 1, head_end);
            }
            for x in 1..if vec_edges { 1 } else { head_end } {
                gradient_interior_elem_canon::<P, P, BANDVIS, BV_DSTACT, BANK, HPR, R7>(
                    src_h,
                    dst_h,
                    activity,
                    act_dst,
                    mu1,
                    mu2,
                    width,
                    row,
                    row_u,
                    row_d,
                    act_row,
                    x,
                    bv_lo,
                    bv_hi,
                    bank,
                    &mut bank_row,
                    &mut chroma_row,
                    &mut r4,
                    &mut *r_gsrc,
                    &mut *r_gdst,
                    &mut *r_gms,
                    &mut *r_gms2,
                    &mut *r_ring,
                    &mut *r_band,
                    &mut *r_bv_gain,
                    &mut *r_bv_loss,
                    false,
                );
            }
            // The chunks, x0 = head_end + 8k for k < n. Every plane's window
            // is cut to its exact span first, so the loads need no per-chunk
            // range checks (they were most of this loop's instructions).
            let n = (interior_end - head_end) / 8;
            let span = n * 8;
            let c0 = row + head_end;
            let a0 = act_row + head_end;
            macro_rules! win {
                ($pl:expr, $start:expr) => {{
                    let w: &[[f32; 8]] = $pl[$start..$start + span].as_chunks::<8>().0;
                    &w[..n]
                }};
            }
            let (w_sxl, w_sxr, w_syu, w_syd, w_s) = (
                win!(src_h, c0 - 1),
                win!(src_h, c0 + 1),
                win!(src_h, row_u + head_end),
                win!(src_h, row_d + head_end),
                win!(src_h, c0),
            );
            let (w_dxl, w_dxr, w_dyu, w_dyd, w_d) = (
                win!(dst_h, c0 - 1),
                win!(dst_h, c0 + 1),
                win!(dst_h, row_u + head_end),
                win!(dst_h, row_d + head_end),
                win!(dst_h, c0),
            );
            let w_act = win!(activity, a0);
            let w_actd: &[[f32; 8]] = if BV_DSTACT { win!(act_dst, a0) } else { &[] };
            let (w_m1, w_m2): (&[[f32; 8]], &[[f32; 8]]) =
                if HPR { (win!(mu1, a0), win!(mu2, a0)) } else { (&[], &[]) };
            // Revision 7 on AVX-512 tiers: the chunks from x = 16 run two at a time,
            // sixteen columns per step, so element e of a step is lane
            // `(x0 + e) & 15`; the first chunk (x = 8) and an odd last one stay
            // eight-wide. Every term is element-wise and every pool lane adds
            // the same values in the same order, so the bits are unchanged.
            // The gmsbank and rev4 bank paths keep the eight-wide step.
            let wide = rev7 && steps_sixteen::<Token>() && !BANK && r4.is_none() && n >= 3;
            let m16 = if wide { (n - 1) / 2 } else { 0 };
            let mut k = 0usize;
            while k < n {
                if k == 1 && m16 > 0 {
                    let span16 = m16 * 16;
                    macro_rules! win16 {
                        ($pl:expr, $start:expr) => {{
                            let w: &[[f32; 16]] =
                                $pl[$start + 8..$start + 8 + span16].as_chunks::<16>().0;
                            &w[..m16]
                        }};
                    }
                    let (v_sxl, v_sxr, v_syu, v_syd, v_s) = (
                        win16!(src_h, c0 - 1),
                        win16!(src_h, c0 + 1),
                        win16!(src_h, row_u + head_end),
                        win16!(src_h, row_d + head_end),
                        win16!(src_h, c0),
                    );
                    let (v_dxl, v_dxr, v_dyu, v_dyd, v_d) = (
                        win16!(dst_h, c0 - 1),
                        win16!(dst_h, c0 + 1),
                        win16!(dst_h, row_u + head_end),
                        win16!(dst_h, row_d + head_end),
                        win16!(dst_h, c0),
                    );
                    let v_act = win16!(activity, a0);
                    let v_actd: &[[f32; 16]] = if BV_DSTACT { win16!(act_dst, a0) } else { &[] };
                    let (v_m1, v_m2): (&[[f32; 16]], &[[f32; 16]]) =
                        if HPR { (win16!(mu1, a0), win16!(mu2, a0)) } else { (&[], &[]) };
                    for j in 0..m16 {
                        let x0 = head_end + 8 + 16 * j;
                        macro_rules! ld {
                            ($w:expr) => {
                                f32x16::load_t(token, &$w[j])
                            };
                        }
                        let actd = if BV_DSTACT { ld!(v_actd) } else { f32x16::zero_t(token) };
                        let (m1, m2) = if HPR {
                            (ld!(v_m1), ld!(v_m2))
                        } else {
                            (f32x16::zero_t(token), f32x16::zero_t(token))
                        };
                        let (gsrc, gdst, g, g2, ring_i, band, bv_gain, bv_loss) =
                            gradient_terms32_v::<_, BANDVIS, BV_DSTACT, HPR>(
                                token,
                                ld!(v_sxl),
                                ld!(v_sxr),
                                ld!(v_syu),
                                ld!(v_syd),
                                ld!(v_dxl),
                                ld!(v_dxr),
                                ld!(v_dyu),
                                ld!(v_dyd),
                                ld!(v_s),
                                ld!(v_d),
                                ld!(v_act),
                                actd,
                                m1,
                                m2,
                                bv_lo,
                                bv_hi,
                                R7,
                            );
                        // f64 lanes: lanes 0..8 then 8..16, each lane one add.
                        macro_rules! add64 {
                            ($pool:ident, $v:expr) => {{
                                let a = ($v).to_array();
                                for (off, half) in [(0usize, &a[..8]), (8, &a[8..])] {
                                    let lanes = $pool.chunk8(off);
                                    for l in 0..8 {
                                        lanes[l] += half[l] as f64;
                                    }
                                }
                            }};
                        }
                        macro_rules! add {
                            ($pool:ident, $pool32:ident, $v:expr) => {{
                                if rev7 {
                                    (f32x16::load_t(token, &$pool32) + $v).store(&mut $pool32);
                                } else {
                                    add64!($pool, $v);
                                }
                            }};
                        }
                        add!(r_gsrc, r_gsrc32, gsrc);
                        add!(r_gdst, r_gdst32, gdst - gsrc);
                        add!(r_gms, r_gms32, g);
                        add!(r_gms2, r_gms232, g2);
                        add!(r_ring, r_ring32, ring_i);
                        add!(r_band, r_band32, band);
                        if BANDVIS {
                            add!(r_bv_gain, r_bv_gain32, bv_gain);
                            add!(r_bv_loss, r_bv_loss32, bv_loss);
                        }
                        if rev7 && (x0 + 16).is_multiple_of(crate::featcanon::ROW_LANES_FOLD) {
                            gpool_fold!(
                                clear;
                                r_gms, r_gms32; r_gms2, r_gms232;
                                r_ring, r_ring32; r_band, r_band32; r_bv_gain, r_bv_gain32;
                                r_bv_loss, r_bv_loss32
                            );
                            gpool_fold!(pair; r_gsrc, r_gsrc32, r_gdst, r_gdst32);
                            r_gsrc32 = [0.0f32; 16];
                            r_gdst32 = [0.0f32; 16];
                        }
                    }
                    k = 1 + 2 * m16;
                    continue;
                }
                let x0 = head_end + 8 * k;
                let sxl = f32x8::load_t(token, &w_sxl[k]);
                let sxr = f32x8::load_t(token, &w_sxr[k]);
                let syu = f32x8::load_t(token, &w_syu[k]);
                let syd = f32x8::load_t(token, &w_syd[k]);
                let dxl = f32x8::load_t(token, &w_dxl[k]);
                let dxr = f32x8::load_t(token, &w_dxr[k]);
                let dyu = f32x8::load_t(token, &w_dyu[k]);
                let dyd = f32x8::load_t(token, &w_dyd[k]);
                let s = f32x8::load_t(token, &w_s[k]);
                let dd = f32x8::load_t(token, &w_d[k]);
                let act = f32x8::load_t(token, &w_act[k]);
                let actd = if BV_DSTACT {
                    f32x8::load_t(token, &w_actd[k])
                } else {
                    f32x8::zero_t(token)
                };
                let (m1, m2) = if HPR {
                    (f32x8::load_t(token, &w_m1[k]), f32x8::load_t(token, &w_m2[k]))
                } else {
                    (f32x8::zero_t(token), f32x8::zero_t(token))
                };
                let (gsrc, gdst, g, g2, ring_i, band, bv_gain, bv_loss) =
                    gradient_terms32_v::<_, BANDVIS, BV_DSTACT, HPR>(
                        token, sxl, sxr, syu, syd, dxl, dxr, dyu, dyd, s, dd, act, actd, m1, m2,
                        bv_lo, bv_hi, R7,
                    );

                if BANK {
                    let gs = gsrc.to_array();
                    let gd = gdst.to_array();
                    for l in 0..8 {
                        gmsbank_pixel_var(
                            &mut bank_row,
                            f64::from(gs[l]),
                            f64::from(gd[l]),
                            bank.gradient,
                            l,
                        );
                    }
                    if let Some(chroma) = bank.chroma {
                        for l in 0..8 {
                            let i = act_row + x0 + l;
                            let h = i + width;
                            let reference = [
                                f64::from(src_h[h]) - f64::from(0.42_f32),
                                f64::from(chroma.reference[i]) - f64::from(0.55_f32),
                            ];
                            let distorted = [
                                f64::from(dst_h[h]) - f64::from(0.42_f32),
                                f64::from(chroma.distorted[i]) - f64::from(0.55_f32),
                            ];
                            for (cell, constants) in chroma_row.iter_mut().zip(GMSBANK_CS_C) {
                                cell.push(
                                    l,
                                    chromaticity_loss(reference, distorted, constants),
                                    true,
                                );
                            }
                        }
                    }
                }

                if rev7 {
                    gpool_add!(r_gsrc, r_gsrc32, x0, gsrc);
                    gpool_add!(r_gdst, r_gdst32, x0, gdst - gsrc);
                } else {
                    gpool_add64!(r_gsrc, x0, gsrc);
                    gpool_add64!(r_gdst, x0, gdst);
                }
                gpool_add!(r_gms, r_gms32, x0, g);
                gpool_add!(r_gms2, r_gms232, x0, g2);
                gpool_add!(r_ring, r_ring32, x0, ring_i);
                if let Some(r4) = r4.as_mut() {
                    let ra = ring_i.to_array();
                    let gs = gsrc.to_array();
                    let gd = gdst.to_array();
                    for l in 0..8 {
                        rev4_grad_pixel(
                            r4,
                            &RINGBASIS_UC,
                            ra[l] as f64,
                            gs[l] as f64,
                            gd[l] as f64,
                            act_row + x0 + l,
                            l,
                        );
                    }
                }
                gpool_add!(r_band, r_band32, x0, band);
                if BANDVIS {
                    gpool_add!(r_bv_gain, r_bv_gain32, x0, bv_gain);
                    gpool_add!(r_bv_loss, r_bv_loss32, x0, bv_loss);
                }
                if rev7 && (x0 + 8).is_multiple_of(crate::featcanon::ROW_LANES_FOLD) {
                    gpool_fold!(
                        clear;
                        r_gms, r_gms32; r_gms2, r_gms232;
                        r_ring, r_ring32; r_band, r_band32; r_bv_gain, r_bv_gain32;
                        r_bv_loss, r_bv_loss32
                    );
                    gpool_fold!(pair; r_gsrc, r_gsrc32, r_gdst, r_gdst32);
                    r_gsrc32 = [0.0f32; 16];
                    r_gdst32 = [0.0f32; 16];
                }
                k += 1;
            }
            let x0 = head_end + span;
            // The end of the chunks, unless the last chunk already folded.
            if rev7 && n > 0 && !x0.is_multiple_of(crate::featcanon::ROW_LANES_FOLD) {
                gpool_fold!(
                    last;
                    r_gms, r_gms32; r_gms2, r_gms232;
                    r_ring, r_ring32; r_band, r_band32; r_bv_gain, r_bv_gain32;
                    r_bv_loss, r_bv_loss32
                );
                gpool_fold!(pair; r_gsrc, r_gsrc32, r_gdst, r_gdst32);
            }
            // The tail: one eight-wide step ending at the interior's end, or
            // the canonical element step per column.
            if vec_edges && x0 < interior_end {
                edge8!(interior_end - 8, x0, interior_end);
            }
            for x in x0..if vec_edges { x0 } else { interior_end } {
                gradient_interior_elem_canon::<P, P, BANDVIS, BV_DSTACT, BANK, HPR, R7>(
                    src_h,
                    dst_h,
                    activity,
                    act_dst,
                    mu1,
                    mu2,
                    width,
                    row,
                    row_u,
                    row_d,
                    act_row,
                    x,
                    bv_lo,
                    bv_hi,
                    bank,
                    &mut bank_row,
                    &mut chroma_row,
                    &mut r4,
                    &mut *r_gsrc,
                    &mut *r_gdst,
                    &mut *r_gms,
                    &mut *r_gms2,
                    &mut *r_ring,
                    &mut *r_band,
                    &mut *r_bv_gain,
                    &mut *r_bv_loss,
                    false,
                );
            }
            if !R7 {
                gradient_pools_finish::<P, BANDVIS>(&mut acc, &fresh);
            }
        }
        gradient_border_pixel::<BANDVIS, BV_DSTACT, BANK, HPR, R7>(
            src_h,
            dst_h,
            activity,
            act_dst,
            mu1,
            mu2,
            width,
            width - 1,
            y,
            bv_lo,
            bv_hi,
            &mut acc,
            &mut r4,
            bank,
            &mut bank_row,
            &mut chroma_row,
        );
        if BANK {
            for (sum, row) in bank_acc.iter_mut().zip(&bank_row) {
                sum.merge_var(row);
            }
            for (sum, row) in chroma_acc.iter_mut().zip(&chroma_row) {
                sum.merge_var(row);
            }
        }
    }
    if R7 && width > 2 {
        gradient_pools_finish::<P, BANDVIS>(&mut acc, &call_pools);
    }
    if BANK {
        for (dst, src) in acc.bank.iter_mut().zip(&bank_acc) {
            *dst = src.resolve();
        }
        for (dst, src) in acc.chroma.iter_mut().zip(&chroma_acc) {
            *dst = src.resolve();
        }
    }
    acc
}

/// Scalar-tier sibling of [`gradient_block_kernel_canon64_vec`].
#[allow(clippy::too_many_arguments)]
fn gradient_block_kernel_canon64_vec_scalar<
    P: crate::featcanon::Pool64,
    const BANDVIS: bool,
    const BV_DSTACT: bool,
    const BANK: bool,
    const HPR: bool,
    const R7: bool,
>(
    _token: archmage::ScalarToken,
    src_h: &[f32],
    dst_h: &[f32],
    activity: &[f32],
    act_dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    width: usize,
    height: usize,
    bv_lo: f32,
    bv_hi: f32,
    r4: Option<Rev4Grad<'_>>,
    mode: crate::featcanon::Mode,
) -> GradientAccum {
    gradient_block_kernel_canon::<P, BANDVIS, BV_DSTACT, BANK, HPR, R7>(
        src_h, dst_h, activity, act_dst, mu1, mu2, width, height, bv_lo, bv_hi, r4, mode,
    )
}

/// Wasm128 sibling of [`gradient_block_kernel_canon64_vec_scalar`].
#[allow(clippy::too_many_arguments, dead_code)]
fn gradient_block_kernel_canon64_vec_wasm128<
    P: crate::featcanon::Pool64,
    const BANDVIS: bool,
    const BV_DSTACT: bool,
    const BANK: bool,
    const HPR: bool,
    const R7: bool,
>(
    _token: archmage::Wasm128Token,
    src_h: &[f32],
    dst_h: &[f32],
    activity: &[f32],
    act_dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    width: usize,
    height: usize,
    bv_lo: f32,
    bv_hi: f32,
    r4: Option<Rev4Grad<'_>>,
    mode: crate::featcanon::Mode,
) -> GradientAccum {
    gradient_block_kernel_canon::<P, BANDVIS, BV_DSTACT, BANK, HPR, R7>(
        src_h, dst_h, activity, act_dst, mu1, mu2, width, height, bv_lo, bv_hi, r4, mode,
    )
}

/// NEON runs the scalar canonical body (2026-10-03): NEON vector `max`/`min` propagate NaN where the canon's `f32::max`/
/// `f64::max` drop it (the REV4VEC CI aarch64 failure), so only the x86_64 fused tiers run a lane-parallel body.
#[allow(clippy::too_many_arguments, dead_code)]
fn gradient_block_kernel_canon64_vec_neon<
    P: crate::featcanon::Pool64,
    const BANDVIS: bool,
    const BV_DSTACT: bool,
    const BANK: bool,
    const HPR: bool,
    const R7: bool,
>(
    _token: archmage::NeonToken,
    src_h: &[f32],
    dst_h: &[f32],
    activity: &[f32],
    act_dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    width: usize,
    height: usize,
    bv_lo: f32,
    bv_hi: f32,
    r4: Option<Rev4Grad<'_>>,
    mode: crate::featcanon::Mode,
) -> GradientAccum {
    gradient_block_kernel_canon::<P, BANDVIS, BV_DSTACT, BANK, HPR, R7>(
        src_h, dst_h, activity, act_dst, mu1, mu2, width, height, bv_lo, bv_hi, r4, mode,
    )
}

/// f64-exact sibling — every pixel through [`gradient_terms64`], interior
/// folded into Neumaier pools, borders direct (identical element math).
#[cfg(feature = "oracle")]
#[allow(clippy::too_many_arguments)]
fn gradient_block_kernel_exact<
    const BANDVIS: bool,
    const BV_DSTACT: bool,
    const BANK: bool,
    const HPR: bool,
    const R7: bool,
>(
    src_h: &[f32],
    dst_h: &[f32],
    activity: &[f32],
    act_dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    width: usize,
    height: usize,
    bv_lo: f32,
    bv_hi: f32,
    mut r4: Option<Rev4Grad<'_>>,
    mode: crate::featcanon::Mode,
) -> GradientAccum {
    use crate::featcanon::{Neum64, Pool as _};
    let mut acc = GradientAccum::default();
    let bank = r4
        .as_ref()
        .map_or_else(|| GmsBankWork::new(1, None), |r| r.bank);
    let mut bank_acc: [GmsBankCellVar; 5] = std::array::from_fn(|_| GmsBankCellVar::for_mode(mode));
    let mut chroma_acc: [GmsBankCellVar; 5] =
        std::array::from_fn(|_| GmsBankCellVar::for_mode(mode));
    for y in 0..height {
        let act_row = y * width;
        let mut bank_row: [GmsBankCellVar; 5] =
            std::array::from_fn(|_| GmsBankCellVar::for_mode(mode));
        let mut chroma_row: [GmsBankCellVar; 5] =
            std::array::from_fn(|_| GmsBankCellVar::for_mode(mode));

        gradient_border_pixel::<BANDVIS, BV_DSTACT, BANK, HPR, R7>(
            src_h,
            dst_h,
            activity,
            act_dst,
            mu1,
            mu2,
            width,
            0,
            y,
            bv_lo,
            bv_hi,
            &mut acc,
            &mut r4,
            bank,
            &mut bank_row,
            &mut chroma_row,
        );
        if width > 2 {
            let interior_end = width - 1;
            let mut r_gms = Neum64::zero();
            let mut r_gms2 = Neum64::zero();
            let mut r_ring = Neum64::zero();
            let mut r_band = Neum64::zero();
            let mut r_gsrc = Neum64::zero();
            let mut r_gdst = Neum64::zero();
            let mut r_bv_gain = Neum64::zero();
            let mut r_bv_loss = Neum64::zero();
            for x in 1..interior_end {
                let lane = x & 7;
                let t = gradient_terms64::<BANDVIS, BV_DSTACT, HPR>(
                    src_h,
                    dst_h,
                    activity,
                    act_dst,
                    mu1,
                    mu2,
                    width,
                    x,
                    y,
                    bv_lo as f64,
                    bv_hi as f64,
                    R7,
                );
                if BANK {
                    gmsbank_pixel_var(&mut bank_row, t.gsrc, t.gdst, bank.gradient, lane);
                    if let Some(chroma) = bank.chroma {
                        let i = act_row + x;
                        let h = i + width;
                        let reference = [
                            f64::from(src_h[h]) - f64::from(0.42_f32),
                            f64::from(chroma.reference[i]) - f64::from(0.55_f32),
                        ];
                        let distorted = [
                            f64::from(dst_h[h]) - f64::from(0.42_f32),
                            f64::from(chroma.distorted[i]) - f64::from(0.55_f32),
                        ];
                        for (cell, constants) in chroma_row.iter_mut().zip(GMSBANK_CS_C) {
                            cell.push(
                                lane,
                                chromaticity_loss(reference, distorted, constants),
                                true,
                            );
                        }
                    }
                }
                r_gsrc.add64(lane, t.gsrc);
                r_gdst.add64(lane, t.gdst);
                r_gms.add64(lane, t.g);
                r_gms2.add64(lane, t.g2);
                r_ring.add64(lane, t.ring);
                if let Some(r4) = r4.as_mut() {
                    rev4_grad_pixel(
                        r4,
                        &RINGBASIS_UC,
                        t.ring,
                        t.gsrc,
                        t.gdst,
                        act_row + x,
                        x & 7,
                    );
                }
                r_band.add64(lane, t.band);
                if BANDVIS {
                    r_bv_gain.add64(lane, t.bv_gain);
                    r_bv_loss.add64(lane, t.bv_loss);
                }
            }
            acc.sum_gms += r_gms.fin();
            acc.sum_gms2 += r_gms2.fin();
            acc.sum_ringing += r_ring.fin();
            acc.sum_banding += r_band.fin();
            acc.sum_grad_src += r_gsrc.fin();
            acc.sum_grad_dst += r_gdst.fin();
            if BANDVIS {
                acc.sum_bv_gain += r_bv_gain.fin();
                acc.sum_bv_loss += r_bv_loss.fin();
            }
        }
        gradient_border_pixel::<BANDVIS, BV_DSTACT, BANK, HPR, R7>(
            src_h,
            dst_h,
            activity,
            act_dst,
            mu1,
            mu2,
            width,
            width - 1,
            y,
            bv_lo,
            bv_hi,
            &mut acc,
            &mut r4,
            bank,
            &mut bank_row,
            &mut chroma_row,
        );
        if BANK {
            for (sum, row) in bank_acc.iter_mut().zip(&bank_row) {
                sum.merge_var(row);
            }
            for (sum, row) in chroma_acc.iter_mut().zip(&chroma_row) {
                sum.merge_var(row);
            }
        }
    }
    if BANK {
        for (dst, src) in acc.bank.iter_mut().zip(&bank_acc) {
            *dst = src.resolve();
        }
        for (dst, src) in acc.chroma.iter_mut().zip(&chroma_acc) {
            *dst = src.resolve();
        }
    }
    acc
}

/// `bandvis`: `Some((δ_lo, δ_hi))` accumulates the append2 BANDVIS pair
/// (Y channel with `append2_block` on — the CROSS-pattern const split, so
/// chroma/off paths pay nothing and change no byte). `bv_act_dst`:
/// `Some(dst-activity strip)` switches the BANDVIS dst band term to the
/// dst self-mask (`append2_dst_activity` — a third const-split
/// instantiation; `None` runs the pre-fix bytes exactly). Ignored without
/// `bandvis`. `r4`: REV4's C2 ring-bin / C4 bleed hook bundle — `None`
/// on every non-rev4 call, and the `Option` check is one loop-invariant
/// branch (the `transducer_bank` shape), so the OFF path's op sequence
/// is the pre-REV4 one bit for bit.
fn gradient_block_kernel(
    src: &[f32],
    dst: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    bandvis: Option<(f32, f32)>,
    bv_act_dst: Option<&[f32]>,
    r4: Option<Rev4Grad<'_>>,
    gmsbank: bool,
) -> GradientAccum {
    gradient_block_kernel_at_revision(
        src,
        dst,
        activity,
        None,
        width,
        height,
        bandvis,
        bv_act_dst,
        r4,
        gmsbank,
        crate::ssim_form::active_revision(),
    )
}

#[allow(clippy::too_many_arguments)]
fn gradient_block_kernel_at_revision(
    src: &[f32],
    dst: &[f32],
    activity: &[f32],
    mu: Option<(&[f32], &[f32])>,
    width: usize,
    height: usize,
    bandvis: Option<(f32, f32)>,
    bv_act_dst: Option<&[f32]>,
    r4: Option<Rev4Grad<'_>>,
    gmsbank: bool,
    revision: FormulaRevision,
) -> GradientAccum {
    // rev4canon: the computation's own mode routes through the scalar
    // `Pool`-generic mirror (same halo contract, same borders) — `LanesF64`
    // (the Rev4 canon) or the oracle arms under `ZENSIM_FEATCANON`.
    // `compute_mode()` is `mode(comp_rev)` here (see it for the guarantee);
    // `Off` below Rev4 keeps the shipped SIMD bodies byte-for-byte.
    // `bv_act_dst` is ignored without `bandvis`, as in the production match
    // below.
    let mode = crate::featcanon::mode(revision);
    // Revision 6 owns the high-pass ringing form; `ZENSIM_RING_FORM=hpm` is
    // the per-process research arm of the same arithmetic at Rev4/Rev5.
    let hp = revision.ring_high_pass() || active_ring_form() == RingForm::HighPassMu;
    #[cfg(feature = "oracle")]
    let exact = matches!(mode, crate::featcanon::Mode::Exact);
    #[cfg(not(feature = "oracle"))]
    let exact = false;
    assert!(
        !hp || ((matches!(mode, crate::featcanon::Mode::Canon64) || exact) && mu.is_some()),
        "the high-pass ringing form (revision 6 or ZENSIM_RING_FORM=hpm) is implemented on \
         the canonical gradient kernel (formula revision 4 or later, default ZENSIM_FEATCANON, \
         or the exact oracle) with local-mean planes only"
    );
    let (mu1, mu2) = mu.unwrap_or((&[], &[]));
    if mode.active() {
        use crate::featcanon::Mode;
        #[cfg(feature = "oracle")]
        use crate::featcanon::{LanesF32, Neum64};
        let m = mode;
        #[cfg(feature = "oracle")]
        macro_rules! go {
            ($p:ty) => {
                match (bandvis, bv_act_dst, gmsbank) {
                    (None, _, false) => gradient_block_kernel_canon::<$p, false, false, false, false, false>(
                        src,
                        dst,
                        activity,
                        &[],
                        &[],
                        &[],
                        width,
                        height,
                        0.0,
                        0.0,
                        r4,
                        m,
                    ),
                    (None, _, true) => gradient_block_kernel_canon::<$p, false, false, true, false, false>(
                        src,
                        dst,
                        activity,
                        &[],
                        &[],
                        &[],
                        width,
                        height,
                        0.0,
                        0.0,
                        r4,
                        m,
                    ),
                    (Some((lo, hi)), None, false) => {
                        gradient_block_kernel_canon::<$p, true, false, false, false, false>(
                            src,
                            dst,
                            activity,
                            &[],
                            &[],
                            &[],
                            width,
                            height,
                            lo,
                            hi,
                            r4,
                            m,
                        )
                    }
                    (Some((lo, hi)), None, true) => {
                        gradient_block_kernel_canon::<$p, true, false, true, false, false>(
                            src,
                            dst,
                            activity,
                            &[],
                            &[],
                            &[],
                            width,
                            height,
                            lo,
                            hi,
                            r4,
                            m,
                        )
                    }
                    (Some((lo, hi)), Some(ad), false) => {
                        gradient_block_kernel_canon::<$p, true, true, false, false, false>(
                            src, dst, activity, ad, &[], &[], width, height, lo, hi, r4, m,
                        )
                    }
                    (Some((lo, hi)), Some(ad), true) => {
                        gradient_block_kernel_canon::<$p, true, true, true, false, false>(
                            src, dst, activity, ad, &[], &[], width, height, lo, hi, r4, m,
                        )
                    }
                }
            };
        }
        // canon64: the `LanesF64` arm, tier-dispatched — scalar/wasm128
        // keep the generic scalar canonical body.
        // `ZENSIM_RING_FORM` selects the ringing arm at compile time per
        // instantiation (`HPR`), so the raw form's code is unchanged.
        macro_rules! go64 {
            ($b:literal, $v:literal, $k:literal, $act_dst:expr, $lo:expr, $hi:expr) => {
                match (revision >= FormulaRevision::Rev5, hp, revision.row_lanes()) {
                    (true, false, false) => incant!(
                        gradient_block_kernel_canon64_vec::<crate::featcanon::Lanes16F64, $b, $v, $k, false, false>(
                            src, dst, activity, $act_dst, mu1, mu2, width, height, $lo, $hi, r4, m,
                        ),
                        [v4x, v4, v3, neon, wasm128, scalar]
                    ),
                    (true, true, false) => incant!(
                        gradient_block_kernel_canon64_vec::<crate::featcanon::Lanes16F64, $b, $v, $k, true, false>(
                            src, dst, activity, $act_dst, mu1, mu2, width, height, $lo, $hi, r4, m,
                        ),
                        [v4x, v4, v3, neon, wasm128, scalar]
                    ),
                    // Revision 7 (`rowlanes`): raw ringing, as revision 5 (owner, 2026-10-25) ...
                    (true, false, true) => incant!(
                        gradient_block_kernel_canon64_vec::<crate::featcanon::Lanes16F64, $b, $v, $k, false, true>(
                            src, dst, activity, $act_dst, mu1, mu2, width, height, $lo, $hi, r4, m,
                        ),
                        [v4x, v4, v3, neon, wasm128, scalar]
                    ),
                    // ... or the high-pass form under the `ZENSIM_RING_FORM=hpm` research arm.
                    (true, true, true) => incant!(
                        gradient_block_kernel_canon64_vec::<crate::featcanon::Lanes16F64, $b, $v, $k, true, true>(
                            src, dst, activity, $act_dst, mu1, mu2, width, height, $lo, $hi, r4, m,
                        ),
                        [v4x, v4, v3, neon, wasm128, scalar]
                    ),
                    (false, _, true) => unreachable!("revision 7 implies revision 5's lanes"),
                    (false, false, false) => incant!(
                        gradient_block_kernel_canon64_vec::<crate::featcanon::LanesF64, $b, $v, $k, false, false>(
                            src, dst, activity, $act_dst, mu1, mu2, width, height, $lo, $hi, r4, m,
                        ),
                        [v4x, v4, v3, neon, wasm128, scalar]
                    ),
                    (false, true, false) => incant!(
                        gradient_block_kernel_canon64_vec::<crate::featcanon::LanesF64, $b, $v, $k, true, false>(
                            src, dst, activity, $act_dst, mu1, mu2, width, height, $lo, $hi, r4, m,
                        ),
                        [v4x, v4, v3, neon, wasm128, scalar]
                    ),
                }
            };
        }
        // The exact oracle follows the revision's ringing form and forms
        // (`HPR`, `R7`), so revisions 6 and 7 have an f64 reference too.
        #[cfg(feature = "oracle")]
        macro_rules! go_exact {
            ($hpr:literal, $r7:literal) => {
                match (bandvis, bv_act_dst, gmsbank) {
                    (None, _, false) => gradient_block_kernel_exact::<false, false, false, $hpr, $r7>(
                        src, dst, activity, &[], mu1, mu2, width, height, 0.0, 0.0, r4, m,
                    ),
                    (None, _, true) => gradient_block_kernel_exact::<false, false, true, $hpr, $r7>(
                        src, dst, activity, &[], mu1, mu2, width, height, 0.0, 0.0, r4, m,
                    ),
                    (Some((lo, hi)), None, false) => gradient_block_kernel_exact::<true, false, false, $hpr, $r7>(
                        src, dst, activity, &[], mu1, mu2, width, height, lo, hi, r4, m,
                    ),
                    (Some((lo, hi)), None, true) => gradient_block_kernel_exact::<true, false, true, $hpr, $r7>(
                        src, dst, activity, &[], mu1, mu2, width, height, lo, hi, r4, m,
                    ),
                    (Some((lo, hi)), Some(ad), false) => gradient_block_kernel_exact::<true, true, false, $hpr, $r7>(
                        src, dst, activity, ad, mu1, mu2, width, height, lo, hi, r4, m,
                    ),
                    (Some((lo, hi)), Some(ad), true) => gradient_block_kernel_exact::<true, true, true, $hpr, $r7>(
                        src, dst, activity, ad, mu1, mu2, width, height, lo, hi, r4, m,
                    ),
                }
            };
        }
        return match mode {
            Mode::Canon64 => match (bandvis, bv_act_dst, gmsbank) {
                (None, _, false) => go64!(false, false, false, &[], 0.0, 0.0),
                (None, _, true) => go64!(false, false, true, &[], 0.0, 0.0),
                (Some((lo, hi)), None, false) => go64!(true, false, false, &[], lo, hi),
                (Some((lo, hi)), None, true) => go64!(true, false, true, &[], lo, hi),
                (Some((lo, hi)), Some(ad), false) => go64!(true, true, false, ad, lo, hi),
                (Some((lo, hi)), Some(ad), true) => go64!(true, true, true, ad, lo, hi),
            },
            #[cfg(feature = "oracle")]
            Mode::Canon32 => go!(LanesF32),
            #[cfg(feature = "oracle")]
            Mode::CanonNeum => go!(Neum64),
            #[cfg(feature = "oracle")]
            Mode::Exact => match (hp, revision.stable_forms()) {
                (false, false) => go_exact!(false, false),
                (true, false) => go_exact!(true, false),
                (true, true) => go_exact!(true, true),
                (false, true) => go_exact!(false, true),
            },
            Mode::Off => unreachable!("active() excludes Off"),
        };
    }
    match (bandvis, bv_act_dst, gmsbank) {
        (None, _, false) => incant!(
            gradient_block_kernel_entry(src, dst, activity, width, height, r4),
            [v4x, v4, v3, neon, wasm128, scalar]
        ),
        (None, _, true) => incant!(
            gradient_block_kernel_entry_gmsbank(src, dst, activity, width, height, r4),
            [v4x, v4, v3, neon, wasm128, scalar]
        ),
        (Some((lo, hi)), None, false) => incant!(
            gradient_block_kernel_entry_bandvis(src, dst, activity, width, height, lo, hi, r4),
            [v4x, v4, v3, neon, wasm128, scalar]
        ),
        (Some((lo, hi)), None, true) => incant!(
            gradient_block_kernel_entry_bandvis_gmsbank(
                src, dst, activity, width, height, lo, hi, r4
            ),
            [v4x, v4, v3, neon, wasm128, scalar]
        ),
        (Some((lo, hi)), Some(act_dst), false) => incant!(
            gradient_block_kernel_entry_bandvis_dstact(
                src, dst, activity, act_dst, width, height, lo, hi, r4
            ),
            [v4x, v4, v3, neon, wasm128, scalar]
        ),
        (Some((lo, hi)), Some(act_dst), true) => incant!(
            gradient_block_kernel_entry_bandvis_dstact_gmsbank(
                src, dst, activity, act_dst, width, height, lo, hi, r4
            ),
            [v4x, v4, v3, neon, wasm128, scalar]
        ),
    }
}

/// Sparse blockiness pass — visits ONLY the 8-pixel-lattice positions
/// (`x % BLOCK_LATTICE == 0`, `y % BLOCK_LATTICE == 0`) instead of every
/// pixel. Deliberately scalar (§A.14: not folded into the SIMD dense pass
/// this phase — see the module-level note above) but strictly cheaper
/// than the pre-phase-4 dense-scalar-with-modulo-branch it replaces, since
/// it no longer visits the 7/8 (or 63/64 for the corner term) of pixels
/// that always contribute zero.
///
/// **Canonical accumulation order (streaming pre-chunk P1, 2026-07-26 —
/// `docs/STREAMING_FOLDAPP_C0_DESIGN_2026-07-26.md` §2):** both term
/// families accumulate ROW-ORDERED (y-outer), into two separate f64 sums
/// combined once at the end:
///
/// - `sum_v` (vertical lattice-BOUNDARY steps, `x % LATTICE == 0, x > 0`,
///   every row): `for y in 0..h { for x in lattice }`.
/// - `sum_h` (horizontal lattice-boundary steps, `y % LATTICE == 0, y > 0`,
///   every column): `for y in lattice { for x in 0..w }` (this family was
///   already row-ordered).
///
/// The previous form iterated the vertical family COLUMN-outer and folded
/// both families into one running f64 — an association no row-ordered
/// strip walk can reproduce. Reordering shifts the pooled BLOCKINESS value
/// by f64 reassociation only (~1e-16 rel; 12 of 720 slots affected, all
/// entry paths shift identically so path-parity gates are unaffected), and
/// lets a strip-fed variant accumulate rows `[y0, y1)` per kernel strip
/// into running `(sum_v, sum_h)` with a bit-identical f64 op sequence.
/// (Also a locality win: the old column-outer walk strode `width` floats
/// per step.)
fn blockiness_sparse(src: &[f32], dst: &[f32], width: usize, height: usize) -> f64 {
    let (sum_v, sum_h) = blockiness_sparse_rows(src, dst, width, 0, height);
    crate::featcanon::SumVar::fin(&sum_v) + crate::featcanon::SumVar::fin(&sum_h)
}

/// Row-range core of [`blockiness_sparse`]: accumulate BOTH term families
/// for image rows `[y0, y1)` (`src`/`dst` hold the full plane here; the
/// streamed walk (C2) will call this shape with strip-local slices and
/// translated row indexing). Returns `(sum_v, sum_h)` so a strip walk can
/// keep the two running sums across strips and combine ONCE at finalize,
/// reproducing [`blockiness_sparse`]'s f64 op sequence exactly.
fn blockiness_sparse_rows(
    src: &[f32],
    dst: &[f32],
    width: usize,
    y0: usize,
    y1: usize,
) -> (crate::featcanon::SumVar, crate::featcanon::SumVar) {
    // featacc: `Seq` (the production `f64 +=` chain) unless a measurement
    // mode is set — the f64-element rule means `c32` is `Seq` too.
    use crate::featcanon::{LanesF64, SumVar};
    let mut sum_v = SumVar::f64_elems_meas();
    let mut sum_h = SumVar::f64_elems_meas();
    let spread = crate::ssim_form::active_revision().row_lanes();
    // Revision 7 with the eight-lane sums: lane partials per `STRIP_ROWS`
    // kernel strip of the plane, added in strip order, as the strip walk
    // does (`blockiness_sparse_strip_wide`).
    let partial = spread && matches!(sum_v, SumVar::L64(_));
    let (mut pv, mut ph) = (SumVar::L64(LanesF64::zero()), SumVar::L64(LanesF64::zero()));
    for y in y0..y1 {
        let (v, h) = if partial {
            (&mut pv, &mut ph)
        } else {
            (&mut sum_v, &mut sum_h)
        };
        blockiness_row(
            src,
            dst,
            y * width,
            width,
            y % BLOCK_LATTICE == 0 && y > 0,
            spread,
            v,
            h,
        );
        if partial && ((y + 1).is_multiple_of(STRIP_ROWS) || y + 1 == y1) {
            sum_v.merge_from(&pv);
            sum_h.merge_from(&ph);
            pv = SumVar::L64(LanesF64::zero());
            ph = SumVar::L64(LanesF64::zero());
        }
    }
    (sum_v, sum_h)
}

/// One row of the blockiness lattice, the shared step of every walk: the
/// vertical steps at the lattice columns `x = 8j` (`j ≥ 1`, `x < width`) of
/// the row starting at index `row`, and on a lattice row (`horizontal`) the
/// steps from the row above (`row − width`) at every column, each
/// `bounded_excess(|Δdst|, |Δsrc|, C_BLOCK)` in f64. Horizontal terms add to
/// lane `x & 7`; vertical terms to lane 0, or at revision 7 (`spread`) to
/// lane `(j − 1) & 7`. Each lane takes its terms in ascending `x`, so the
/// eight-lane accumulators get the same bits from the scalar steps and from
/// [`blockiness_row_l64`]'s eight-wide ones.
#[allow(clippy::too_many_arguments)]
pub(crate) fn blockiness_row(
    src: &[f32],
    dst: &[f32],
    row: usize,
    width: usize,
    horizontal: bool,
    spread: bool,
    sum_v: &mut crate::featcanon::SumVar,
    sum_h: &mut crate::featcanon::SumVar,
) {
    use crate::featcanon::SumVar;
    if let (SumVar::L64(v), SumVar::L64(h)) = (&mut *sum_v, &mut *sum_h) {
        incant!(
            blockiness_row_l64(src, dst, row, width, horizontal, spread, v, h),
            [v4x, v4, v3, scalar]
        );
        return;
    }
    let mut x = BLOCK_LATTICE;
    while x < width {
        let i = row + x;
        let step_dst = (dst[i] as f64 - dst[i - 1] as f64).abs();
        let step_src = (src[i] as f64 - src[i - 1] as f64).abs();
        let lane = if spread {
            (x / BLOCK_LATTICE - 1) & 7
        } else {
            x & 7
        };
        sum_v.add64(lane, bounded_excess(step_dst, step_src, C_BLOCK));
        x += BLOCK_LATTICE;
    }
    if horizontal {
        for x in 0..width {
            let i = row + x;
            let i_up = i - width;
            let step_dst = (dst[i] as f64 - dst[i_up] as f64).abs();
            let step_src = (src[i] as f64 - src[i_up] as f64).abs();
            sum_h.add64(x & 7, bounded_excess(step_dst, step_src, C_BLOCK));
        }
    }
}

/// [`blockiness_row`] for the eight-lane f64 accumulators: eight steps per
/// vector, each lane the scalar step's IEEE operations (exact widening,
/// subtraction, `abs`, `max`, the sum and the division), added to the lanes
/// in the scalar order. The vertical lattice columns are gathered eight at a
/// time; with `spread` their eight terms go to the eight lanes at once
/// (columns `8j … 8(j + 7)` with `j ≡ 1 (mod 8)` map to lanes 0–7), without
/// it to lane 0 in column order.
#[magetypes(define(f64x8), v4x, v4, v3, scalar)]
#[allow(clippy::too_many_arguments)]
fn blockiness_row_l64(
    token: Token,
    src: &[f32],
    dst: &[f32],
    row: usize,
    width: usize,
    horizontal: bool,
    spread: bool,
    v: &mut crate::featcanon::LanesF64,
    h: &mut crate::featcanon::LanesF64,
) {
    let c = f64x8::splat_t(token, C_BLOCK);
    let zero = f64x8::zero_t(token);
    macro_rules! terms {
        ($d1:expr, $d0:expr, $s1:expr, $s0:expr) => {{
            let sd = (f64x8::from_array_t(token, $d1) - f64x8::from_array_t(token, $d0)).abs();
            let ss = (f64x8::from_array_t(token, $s1) - f64x8::from_array_t(token, $s0)).abs();
            ((sd - ss).max(zero) / (sd + ss + c)).to_array()
        }};
    }
    let scalar = |i: usize, j: usize| {
        let step_dst = (dst[i] as f64 - dst[j] as f64).abs();
        let step_src = (src[i] as f64 - src[j] as f64).abs();
        bounded_excess(step_dst, step_src, C_BLOCK)
    };
    // Vertical steps: lattice columns `x = 8j`, `j = 1..=last`.
    let last = width.saturating_sub(1) / BLOCK_LATTICE;
    let mut j = 1;
    while j + 8 <= last + 1 {
        // One checked window per plane covers the eight columns and their
        // left neighbors: element `8l + 1` is column `8(j + l)`.
        let a = row + j * BLOCK_LATTICE - 1;
        let dw: &[f32; 58] = dst[a..a + 58].try_into().unwrap();
        let sw: &[f32; 58] = src[a..a + 58].try_into().unwrap();
        let d1: [f64; 8] = core::array::from_fn(|l| dw[8 * l + 1] as f64);
        let d0: [f64; 8] = core::array::from_fn(|l| dw[8 * l] as f64);
        let s1: [f64; 8] = core::array::from_fn(|l| sw[8 * l + 1] as f64);
        let s0: [f64; 8] = core::array::from_fn(|l| sw[8 * l] as f64);
        let t = terms!(d1, d0, s1, s0);
        if spread {
            for (lane, tl) in v.0.iter_mut().zip(t) {
                *lane += tl;
            }
        } else {
            for tl in t {
                v.0[0] += tl;
            }
        }
        j += 8;
    }
    while j <= last {
        let i = row + j * BLOCK_LATTICE;
        let lane = if spread { (j - 1) & 7 } else { 0 };
        v.0[lane] += scalar(i, i - 1);
        j += 1;
    }
    if horizontal {
        let up = row - width;
        let mut x = 0;
        while x + 8 <= width {
            let w = |p: &[f32], at: usize| -> [f64; 8] {
                let a: &[f32; 8] = p[at..at + 8].try_into().unwrap();
                core::array::from_fn(|l| a[l] as f64)
            };
            let t = terms!(
                w(dst, row + x),
                w(dst, up + x),
                w(src, row + x),
                w(src, up + x)
            );
            for (lane, tl) in h.0.iter_mut().zip(t) {
                *lane += tl;
            }
            x += 8;
        }
        while x < width {
            h.0[x & 7] += scalar(row + x, up + x);
            x += 1;
        }
    }
}

/// Compute all [`FEATURES_PER_CHANNEL_V2_TOTAL`] v2 signals for one
/// channel at one scale (except [`idx::EDGE_WIDTH_CHANGE`], filled in by
/// the caller once the adjacent scale is known — see
/// [`compute_v2_features_impl`]). Writes into `out`. Returns
/// `(mean_grad_src, mean_grad_dst)` for the caller's cross-scale
/// edge-width computation (`(0,0)` when `toggles.gradient_features` is
/// off — the caller must not rely on edge-width in that case).
///
/// Phase-5 (§A.15): the per-pixel pass is now a STRIP LOOP — blur +
/// dense-formula-kernel + gradient-kernel FUSED per strip, so full-image
/// mu1/mu2/ssq/s12/activity planes are never materialized (only one
/// strip's worth, `ScratchV2Strip`, sized `width*(STRIP_ROWS+2*HALO_P)`
/// regardless of the channel-scale's actual `height`). Each strip's
/// `DenseAccum`/`GradientAccum` folds into a running whole-image total
/// via `.accumulate()`. `src`/`dst` (this function's own params) remain
/// the FULL channel-scale planes — unchanged from phases 1-4, still
/// materialized once by the caller's XYB conversion — strip-tiling only
/// removes the DERIVED intermediate planes' full-image footprint.
/// Blockiness stays a single full-image pass (§A.14: already sparse,
/// only reads `src`/`dst` which are full arrays regardless).
/// Phase-6 (§A.16 lever B): DISABLED (0 — the bypass never fires). The
/// hypothesis was that below some height, `compute_channel_scale_v2`
/// should bypass the strip loop entirely via
/// `compute_channel_scale_v2_whole`, avoiding the fixed per-strip
/// halo-gather cost. MEASURED (crossover sweep, §A.16.4): the bypass path
/// LOSES to the strip path at every tested size, 256²-512² (bypass ratio
/// 2.5-2.8x vs. strip's 1.3-1.6x at the SAME sizes, both with lever A's
/// division-fusion applied) — the opposite of the phase-5 hypothesis this
/// lever was built to fix. Root cause identified, NOT fixed this phase
/// (out of budget): `compute_channel_scale_v2_whole`'s gradient-halo
/// construction (`gather_strip_halo(..., height+2, 1, &mut src_g)`) copies
/// essentially the WHOLE image (`O(width*height)`, not the `O(width)` this
/// function's own doc originally (incorrectly) claimed) into a FRESH
/// `vec![0.0f32; width*(height+2)]` allocation, TWICE (src_g + dst_g),
/// EVERY call (up to 12x per `compute_v2_features_impl_with_toggles` — 4
/// scales x 3 channels) — this is exactly the "108 full-image allocations"
/// class of problem phase 3 fixed for the main pipeline, reintroduced here
/// specifically for the bypass's gradient halo. A fix (thread a
/// pre-allocated, scratch-owned buffer for this instead of a fresh `vec!`
/// per call) is a well-characterized, NOT-YET-ATTEMPTED follow-on — see
/// §A.16.4/§A.16.6. `compute_channel_scale_v2_whole` and its dedicated
/// test (`bounded_range_bypass_path_small_image`) are KEPT (dead at
/// runtime with this threshold, but correctness-verified and ready for a
/// future session to re-enable once the allocation bug is fixed) rather
/// than ripped out under time pressure.
const STRIP_BYPASS_HEIGHT: usize = 0;

// ============================================================================
// f720+ APPEND kernel — second formula pass over the cache-hot strip planes.
//
// Deliberately a SEPARATE kernel from `dense_block_kernel_generic`: the dense
// kernel's register budget is a documented hazard (§A.14 scalarized its pool
// block after a measured regression; §A.16 measured a 5.3x collapse when the
// body outgrew LLVM's inliner) — widening it again for 19 more accumulators
// would re-open both. A second pass over strip-resident planes re-reads L2,
// not DRAM (that locality is what §A.15's strip tiling bought), so the
// marginal cost is close to the append block's own arithmetic. It also makes
// the f0..f719 bit-stability guarantee structural: the tuned kernels are not
// edited at all (the one exception, `GradientAccum::sum_gms2`, adds a
// derived accumulator without altering any existing operation).
// ============================================================================

/// Per-row-reduced f64 accumulator for the append block.
#[derive(Default, Clone, Copy)]
struct AppendAccum {
    sum_xmask: f64,
    sum_lumt: f64,
    ws_dark: WeightedSum,
    ws_bright: WeightedSum,
    /// Σ mse_i — the mid-luminance bin is DERIVED at finalize: Bernstein
    /// weights sum to 1, so `Σw_mid = n − Σw_dark − Σw_bright` and
    /// `Σw_mid·v = Σv − Σw_dark·v − Σw_bright·v`. One fewer lane pair in
    /// the kernel; bitwise path-stable because the derivation runs in f64
    /// finalize on sums that are themselves path-bitwise-equal.
    sum_mse: f64,
    sum_mscn: f64,
    sum_mscn2: f64,
    sum_cgain: f64,
    sum_closs: f64,
    sum_tex: f64,
    /// Σ art², Σ det² — second raw moments of per-pixel values that are
    /// bit-identical to the dense kernel's `art_i`/`det_i` (same formula,
    /// same inputs, same order), so `finish_append` can pair them with
    /// `DenseAccum::sum_art`/`sum_det` as the matching first moments.
    sum_art2: f64,
    sum_det2: f64,
    sum_s: f64,
    sum_d: f64,
    sum_s2: f64,
    sum_d2: f64,
    /// Revision 2's paired second moment `Σ(d−s)(d+s)`.
    sum_dd: f64,
    /// Revision 2's paired first difference `Σ(d−s)`.
    sum_ds: f64,
    /// append2 highlight bins (HDR route + `append2_block` only — the
    /// `HL = true` kernel instantiation; untouched zeros otherwise).
    ws_hl1: WeightedSum,
    ws_hl2: WeightedSum,
}

impl AppendAccum {
    /// Strip-partial fold, same reasoning as [`DenseAccum::accumulate`].
    #[inline]
    fn accumulate(&mut self, other: &AppendAccum) {
        self.sum_xmask += other.sum_xmask;
        self.sum_lumt += other.sum_lumt;
        self.ws_dark.accumulate(&other.ws_dark);
        self.ws_bright.accumulate(&other.ws_bright);
        self.sum_mse += other.sum_mse;
        self.sum_mscn += other.sum_mscn;
        self.sum_mscn2 += other.sum_mscn2;
        self.sum_cgain += other.sum_cgain;
        self.sum_closs += other.sum_closs;
        self.sum_tex += other.sum_tex;
        self.sum_art2 += other.sum_art2;
        self.sum_det2 += other.sum_det2;
        self.sum_s += other.sum_s;
        self.sum_d += other.sum_d;
        self.sum_s2 += other.sum_s2;
        self.sum_d2 += other.sum_d2;
        self.sum_dd += other.sum_dd;
        self.sum_ds += other.sum_ds;
        self.ws_hl1.accumulate(&other.ws_hl1);
        self.ws_hl2.accumulate(&other.ws_hl2);
    }
}

/// Append-block SIMD kernel body — generic over any `F32x8Backend` token,
/// same row-lane-then-f64 reduction structure as the dense kernel. `CROSS`
/// const-compiles the cross-channel transducer (and its two extra plane
/// streams) in or out — the Y channel dispatches `true`, X/B `false`, so
/// the chroma channels never touch `act_x`/`act_b` (callers pass `&[]`).
///
/// ~19 row-lane accumulators: over the 16-register budget of AVX2/NEON, so
/// some spill there — accepted for the first landing (the spills are
/// L1-resident row-locals). If the perf gate flags it, the §A.14 remedy
/// (scalarize the 3 luminance pools per lane) is the known fix.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn append_block_kernel_generic<T: F32x8Backend + Copy, const CROSS: bool, const HL: bool>(
    token: T,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    bs2: &[f32],
    activity: &[f32],
    ref_y: &[f32],
    act_x: &[f32],
    act_b: &[f32],
    width: usize,
    height: usize,
) -> AppendAccum {
    let zero = V8::<T>::zero_t(token);
    let one = V8::<T>::splat_t(token, 1.0);
    let c_edge = V8::<T>::splat_t(token, C_EDGE as f32);
    let c_mse = V8::<T>::splat_t(token, C_MSE as f32);
    let c_pjnd_clamp = V8::<T>::splat_t(token, C_PJND_CLAMP as f32);
    let k_mid = V8::<T>::splat_t(token, K_PJND_MASK as f32);
    let c_lum = V8::<T>::splat_t(token, C_LUM_T as f32);
    // pjnd_transducer(err, A, k, c) = err/(err + c·(1 + k·A)); folding the
    // luminance / cross-channel terms into A at ratio k_x/K_PJND_MASK gives
    // the intended `c·(1 + k·act + k_x·extra)` denominator with the SAME
    // fused one-division helper (exact algebra, no new formula family).
    let k_lum_ratio = V8::<T>::splat_t(token, (K_LUM_ADAPT / K_PJND_MASK) as f32);
    let k_xch_ratio = V8::<T>::splat_t(token, (K_XCH / K_PJND_MASK) as f32);
    let c_mvar = V8::<T>::splat_t(token, C_MSCN_VAR as f32);
    let c_mabs = V8::<T>::splat_t(token, C_MSCN_ABS as f32);
    let c_msq = V8::<T>::splat_t(token, C_MSCN_SQ as f32);
    let c_con = V8::<T>::splat_t(token, C_CONTRAST as f32);

    let mut acc = AppendAccum::default();
    let width8 = width - (width % 8);

    for y in 0..height {
        let row = y * width;

        let (mut r_xm, mut r_lumt) = (zero, zero);
        let (mut p_wd, mut p_wdv, mut p_wb, mut p_wbv, mut r_mse) = (zero, zero, zero, zero, zero);
        let (mut r_mscn, mut r_mscn2) = (zero, zero);
        let (mut r_cg, mut r_cl, mut r_tex) = (zero, zero, zero);
        let (mut r_art2, mut r_det2) = (zero, zero);
        let (mut r_s, mut r_d, mut r_s2, mut r_d2) = (zero, zero, zero, zero);
        // Revision 2's paired second moment. See
        // `crate::fused::raw_moments_accumulate16` for the measurement chain
        // that ruled out both accumulation granularity and compensation and
        // landed on reassociation.
        let (mut r_dd, mut r_ds) = (zero, zero);
        let (mut p_hl1w, mut p_hl1v, mut p_hl2w, mut p_hl2v) = (zero, zero, zero, zero);

        let mut x = 0usize;
        while x < width8 {
            let i = row + x;
            macro_rules! ld {
                ($plane:expr) => {
                    V8::<T>::from_array_t(token, $plane[i..i + 8].try_into().unwrap())
                };
            }
            let s = ld!(src);
            let dd = ld!(dst);
            let m1 = ld!(mu1);
            let m2 = ld!(mu2);
            let ry = ld!(ref_y);
            let b2 = ld!(bs2);

            let diff_src = (s - m1).abs();
            let diff_dst = (dd - m2).abs();
            let edge_dissim = bounded_dissim_v(token, diff_src, diff_dst, c_edge, false);
            let gt = diff_dst.simd_gt(diff_src);
            let lt = diff_dst.simd_lt(diff_src);
            let art_i = V8::<T>::blend(gt, edge_dissim, zero);
            let det_i = V8::<T>::blend(lt, edge_dissim, zero);
            r_art2 += art_i * art_i;
            r_det2 += det_i * det_i;

            let raw_diff = s - dd;
            let raw_abs_err = raw_diff.abs();
            let mse_i = saturate_v(token, raw_diff * raw_diff, c_mse);
            let t = saturate_v(token, ry, c_lum);
            // Luminance + cross-channel transducers are Y-only (CROSS is
            // the Y channel's dispatch): the luma-gate ablation measured
            // chroma transducers as a broad CID22 cost. `activity` is
            // only consumed here, so the load lives inside the branch.
            if CROSS {
                let act = ld!(activity);
                r_lumt += pjnd_transducer_v(
                    token,
                    raw_abs_err,
                    t.mul_add(k_lum_ratio, act),
                    k_mid,
                    c_pjnd_clamp,
                );
                let act_c = ld!(act_x) + ld!(act_b);
                r_xm += pjnd_transducer_v(
                    token,
                    raw_abs_err,
                    act_c.mul_add(k_xch_ratio, act),
                    k_mid,
                    c_pjnd_clamp,
                );
            }

            let one_mt = one - t;
            let wd = one_mt * one_mt;
            let wb = t * t;
            p_wd += wd;
            p_wdv += wd * mse_i;
            p_wb += wb;
            p_wbv += wb * mse_i;
            r_mse += mse_i;

            if HL {
                // append2 highlight bins (`idx_append2::HL_BIN1/2`):
                // `w = sat(max(ry − anchor, 0), C_HL)` pooling mse_i.
                let hl1_anchor = V8::<T>::splat_t(token, HL1_Y_ANCHOR);
                let hl2_anchor = V8::<T>::splat_t(token, HL2_Y_ANCHOR);
                let c_hl = V8::<T>::splat_t(token, C_HL as f32);
                let w1 = saturate_v(token, (ry - hl1_anchor).max(zero), c_hl);
                let w2 = saturate_v(token, (ry - hl2_anchor).max(zero), c_hl);
                p_hl1w += w1;
                p_hl1v += w1 * mse_i;
                p_hl2w += w2;
                p_hl2v += w2 * mse_i;
            }

            let var1 = (b2 - m1 * m1).max(zero);
            let var2 = ((ld!(ssq) - b2) - m2 * m2).max(zero);
            // #56: exact sqrt+div, not rsqrt — see `mscn_norm_v`.
            let n1 = mscn_norm_v(token, s - m1, var1, c_mvar);
            let n2 = mscn_norm_v(token, dd - m2, var2, c_mvar);
            let dn = n1 - n2;
            r_mscn += saturate_v(token, dn.abs(), c_mabs);
            r_mscn2 += saturate_v(token, dn * dn, c_msq);

            let (cg, cl) = bounded_excess_pair_v(token, var2, var1, c_con);
            r_cg += cg;
            r_cl += cl;
            r_tex += bounded_dissim_v(token, var1, var2, c_con, false);

            r_s += s;
            r_d += dd;
            r_s2 += s * s;
            r_d2 += dd * dd;
            // Revision 2's paired second moment — see
            // `crate::fused::raw_moments_accumulate16`.
            let df = dd - s;
            r_dd += df * (dd + s);
            r_ds += df;

            x += 8;
        }

        acc.sum_xmask += r_xm.reduce_add() as f64;
        acc.sum_lumt += r_lumt.reduce_add() as f64;
        acc.ws_dark.num += p_wdv.reduce_add() as f64;
        acc.ws_dark.den += p_wd.reduce_add() as f64;
        acc.ws_bright.num += p_wbv.reduce_add() as f64;
        acc.ws_bright.den += p_wb.reduce_add() as f64;
        acc.sum_mse += r_mse.reduce_add() as f64;
        acc.sum_mscn += r_mscn.reduce_add() as f64;
        acc.sum_mscn2 += r_mscn2.reduce_add() as f64;
        acc.sum_cgain += r_cg.reduce_add() as f64;
        acc.sum_closs += r_cl.reduce_add() as f64;
        acc.sum_tex += r_tex.reduce_add() as f64;
        acc.sum_art2 += r_art2.reduce_add() as f64;
        acc.sum_det2 += r_det2.reduce_add() as f64;
        acc.sum_s += r_s.reduce_add() as f64;
        acc.sum_d += r_d.reduce_add() as f64;
        acc.sum_s2 += r_s2.reduce_add() as f64;
        acc.sum_d2 += r_d2.reduce_add() as f64;
        acc.sum_dd += r_dd.reduce_add() as f64;
        acc.sum_ds += r_ds.reduce_add() as f64;
        if HL {
            acc.ws_hl1.num += p_hl1v.reduce_add() as f64;
            acc.ws_hl1.den += p_hl1w.reduce_add() as f64;
            acc.ws_hl2.num += p_hl2v.reduce_add() as f64;
            acc.ws_hl2.den += p_hl2w.reduce_add() as f64;
        }

        // Scalar tail — same formulas via the scalar siblings.
        for x in width8..width {
            let i = row + x;
            let s = src[i] as f64;
            let dd = dst[i] as f64;
            let m1 = mu1[i] as f64;
            let m2 = mu2[i] as f64;
            let act = activity[i] as f64;
            let ry = ref_y[i] as f64;
            let b2 = bs2[i] as f64;
            let sq = ssq[i] as f64;

            let diff_src = (s - m1).abs();
            let diff_dst = (dd - m2).abs();
            let ed = bounded_dissim(diff_src, diff_dst, C_EDGE, false);
            let art_i = if diff_dst > diff_src { ed } else { 0.0 };
            let det_i = if diff_dst < diff_src { ed } else { 0.0 };
            acc.sum_art2 += art_i * art_i;
            acc.sum_det2 += det_i * det_i;

            let raw_diff = s - dd;
            let raw_abs_err = raw_diff.abs();
            let mse_i = saturate(raw_diff * raw_diff, C_MSE);
            let t = saturate(ry, C_LUM_T);
            if CROSS {
                acc.sum_lumt += pjnd_transducer(
                    raw_abs_err,
                    act + t * (K_LUM_ADAPT / K_PJND_MASK),
                    K_PJND_MASK,
                    C_PJND_CLAMP,
                );
                let act_c = act_x[i] as f64 + act_b[i] as f64;
                acc.sum_xmask += pjnd_transducer(
                    raw_abs_err,
                    act + act_c * (K_XCH / K_PJND_MASK),
                    K_PJND_MASK,
                    C_PJND_CLAMP,
                );
            }

            let one_mt = 1.0 - t;
            acc.ws_dark.add(one_mt * one_mt, mse_i);
            acc.ws_bright.add(t * t, mse_i);
            acc.sum_mse += mse_i;
            if HL {
                let w1 = saturate((ry - HL1_Y_ANCHOR as f64).max(0.0), C_HL);
                let w2 = saturate((ry - HL2_Y_ANCHOR as f64).max(0.0), C_HL);
                acc.ws_hl1.add(w1, mse_i);
                acc.ws_hl2.add(w2, mse_i);
            }

            let var1 = (b2 - m1 * m1).max(0.0);
            let var2 = ((sq - b2) - m2 * m2).max(0.0);
            let n1 = (s - m1) / (var1 + C_MSCN_VAR).sqrt();
            let n2 = (dd - m2) / (var2 + C_MSCN_VAR).sqrt();
            let dn = n1 - n2;
            acc.sum_mscn += saturate(dn.abs(), C_MSCN_ABS);
            acc.sum_mscn2 += saturate(dn * dn, C_MSCN_SQ);

            let (cg, cl) = bounded_excess_pair(var2, var1, C_CONTRAST);
            acc.sum_cgain += cg;
            acc.sum_closs += cl;
            acc.sum_tex += bounded_dissim(var1, var2, C_CONTRAST, false);

            acc.sum_s += s;
            acc.sum_d += dd;
            acc.sum_s2 += s * s;
            acc.sum_d2 += dd * dd;
            acc.sum_dd += (dd - s) * (dd + s);
            acc.sum_ds += dd - s;
        }
    }

    acc
}

// ============================================================================
// featacc append candidates — measurement-only bodies
// ============================================================================
//
// Same structure as the dense mirrors: `Pool`-generic per-row accumulation
// (canonical lane `x mod 8`, fixed `era2_reduce8`-shaped close into the f64
// `AppendAccum`), scalar f32 elements in the SIMD body's exact op order —
// including the explicit `mul_add` transducer folds, which `f32::mul_add`
// replays bit-exactly (fused on every hardware tier, correctly rounded on
// the scalar tier). `exact` runs the f64 scalar-sibling elements (the
// production tail's forms) into Neumaier pools. Every pixel goes through
// the pools — the canonical close owns the width tail too.

#[inline(always)]
fn pjnd32(rae: f32, a: f32, k: f32, c: f32) -> f32 {
    rae / (rae + c * (1.0f32 + k * a))
}

/// One append pixel's full term vector in f32 — `append_block_kernel_generic`'s
/// SIMD chunk ops verbatim.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn append_terms32<const CROSS: bool, const HL: bool>(
    s: f32,
    dd: f32,
    m1: f32,
    m2: f32,
    sq_i: f32,
    b2: f32,
    act: f32,
    ry: f32,
    act_c: f32,
) -> [f32; 24] {
    let diff_src = (s - m1).abs();
    let diff_dst = (dd - m2).abs();
    let edge_dissim = 1.0f32 - bsim32(diff_src, diff_dst, C_EDGE as f32);
    let art_i = if diff_dst > diff_src {
        edge_dissim
    } else {
        0.0
    };
    let det_i = if diff_dst < diff_src {
        edge_dissim
    } else {
        0.0
    };
    let raw_diff = s - dd;
    let rae = raw_diff.abs();
    let rsq = raw_diff * raw_diff;
    let mse_i = rsq.max(0.0) / (rsq.max(0.0) + C_MSE as f32);
    let t = sat32(ry, C_LUM_T as f32);
    let k_lum = (K_LUM_ADAPT / K_PJND_MASK) as f32;
    let k_xch = (K_XCH / K_PJND_MASK) as f32;
    let (lumt, xm) = if CROSS {
        (
            pjnd32(
                rae,
                t.mul_add(k_lum, act),
                K_PJND_MASK as f32,
                C_PJND_CLAMP as f32,
            ),
            pjnd32(
                rae,
                act_c.mul_add(k_xch, act),
                K_PJND_MASK as f32,
                C_PJND_CLAMP as f32,
            ),
        )
    } else {
        (0.0, 0.0)
    };
    let one_mt = 1.0f32 - t;
    let wd = one_mt * one_mt;
    let wb = t * t;
    let (hl1w, hl1v, hl2w, hl2v) = if HL {
        let w1 = sat32((ry - HL1_Y_ANCHOR).max(0.0), C_HL as f32);
        let w2 = sat32((ry - HL2_Y_ANCHOR).max(0.0), C_HL as f32);
        (w1, w1 * mse_i, w2, w2 * mse_i)
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };
    let var1 = (b2 - m1 * m1).max(0.0);
    let var2 = ((sq_i - b2) - m2 * m2).max(0.0);
    let n1 = (s - m1) / (var1 + C_MSCN_VAR as f32).sqrt();
    let n2 = (dd - m2) / (var2 + C_MSCN_VAR as f32).sqrt();
    let dn = n1 - n2;
    let mscn = sat32(dn.abs(), C_MSCN_ABS as f32);
    let mscn2 = sat32(dn * dn, C_MSCN_SQ as f32);
    let (cg, cl) = bex_pair32(var2, var1, C_CONTRAST as f32);
    let tex = 1.0f32 - bsim32(var1, var2, C_CONTRAST as f32);
    let df = dd - s;
    [
        xm,
        lumt,
        wd,
        wd * mse_i,
        wb,
        wb * mse_i,
        mse_i,
        mscn,
        mscn2,
        cg,
        cl,
        tex,
        art_i * art_i,
        det_i * det_i,
        s,
        dd,
        s * s,
        dd * dd,
        df * (dd + s),
        df,
        hl1w,
        hl1v,
        hl2w,
        hl2v,
    ]
}

/// `Pool`-generic scalar append kernel — see the module note above.
#[allow(clippy::too_many_arguments)]
fn append_block_kernel_canon<P: crate::featcanon::Pool, const CROSS: bool, const HL: bool>(
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    bs2: &[f32],
    activity: &[f32],
    ref_y: &[f32],
    act_x: &[f32],
    act_b: &[f32],
    width: usize,
    height: usize,
) -> AppendAccum {
    let mut acc = AppendAccum::default();
    for y in 0..height {
        let row = y * width;
        let mut r = [P::zero(); 24];
        for x in 0..width {
            let i = row + x;
            let lane = x & 7;
            let t = append_terms32::<CROSS, HL>(
                src[i],
                dst[i],
                mu1[i],
                mu2[i],
                ssq[i],
                bs2[i],
                if CROSS { activity[i] } else { 0.0 },
                ref_y[i],
                if CROSS { act_x[i] + act_b[i] } else { 0.0 },
            );
            for j in 0..24 {
                r[j].add(lane, t[j]);
            }
        }
        acc.sum_xmask += r[0].fin();
        acc.sum_lumt += r[1].fin();
        acc.ws_dark.den += r[2].fin();
        acc.ws_dark.num += r[3].fin();
        acc.ws_bright.den += r[4].fin();
        acc.ws_bright.num += r[5].fin();
        acc.sum_mse += r[6].fin();
        acc.sum_mscn += r[7].fin();
        acc.sum_mscn2 += r[8].fin();
        acc.sum_cgain += r[9].fin();
        acc.sum_closs += r[10].fin();
        acc.sum_tex += r[11].fin();
        acc.sum_art2 += r[12].fin();
        acc.sum_det2 += r[13].fin();
        acc.sum_s += r[14].fin();
        acc.sum_d += r[15].fin();
        acc.sum_s2 += r[16].fin();
        acc.sum_d2 += r[17].fin();
        acc.sum_dd += r[18].fin();
        acc.sum_ds += r[19].fin();
        if HL {
            acc.ws_hl1.den += r[20].fin();
            acc.ws_hl1.num += r[21].fin();
            acc.ws_hl2.den += r[22].fin();
            acc.ws_hl2.num += r[23].fin();
        }
    }
    acc
}

/// f64-exact append sibling — the production scalar tail's element forms
/// (mul+add transducer arguments, two-division `bounded_excess_pair`) into
/// Neumaier pools.
#[cfg(feature = "oracle")]
#[allow(clippy::too_many_arguments)]
fn append_block_kernel_exact<const CROSS: bool, const HL: bool>(
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    bs2: &[f32],
    activity: &[f32],
    ref_y: &[f32],
    act_x: &[f32],
    act_b: &[f32],
    width: usize,
    height: usize,
) -> AppendAccum {
    use crate::featcanon::{Neum64, Pool as _};
    let mut acc = AppendAccum::default();
    for y in 0..height {
        let row = y * width;
        let mut r = [Neum64::zero(); 24];
        for x in 0..width {
            let i = row + x;
            let lane = x & 7;
            let s = src[i] as f64;
            let dd = dst[i] as f64;
            let m1 = mu1[i] as f64;
            let m2 = mu2[i] as f64;
            let act = activity[i] as f64;
            let ry = ref_y[i] as f64;
            let b2 = bs2[i] as f64;
            let sq = ssq[i] as f64;
            let diff_src = (s - m1).abs();
            let diff_dst = (dd - m2).abs();
            let ed = bounded_dissim(diff_src, diff_dst, C_EDGE, false);
            let art_i = if diff_dst > diff_src { ed } else { 0.0 };
            let det_i = if diff_dst < diff_src { ed } else { 0.0 };
            r[12].add64(lane, art_i * art_i);
            r[13].add64(lane, det_i * det_i);
            let raw_diff = s - dd;
            let rae = raw_diff.abs();
            let mse_i = saturate(raw_diff * raw_diff, C_MSE);
            let t = saturate(ry, C_LUM_T);
            if CROSS {
                r[1].add64(
                    lane,
                    pjnd_transducer(
                        rae,
                        act + t * (K_LUM_ADAPT / K_PJND_MASK),
                        K_PJND_MASK,
                        C_PJND_CLAMP,
                    ),
                );
                let act_c = act_x[i] as f64 + act_b[i] as f64;
                r[0].add64(
                    lane,
                    pjnd_transducer(
                        rae,
                        act + act_c * (K_XCH / K_PJND_MASK),
                        K_PJND_MASK,
                        C_PJND_CLAMP,
                    ),
                );
            }
            let one_mt = 1.0 - t;
            let wd = one_mt * one_mt;
            let wb = t * t;
            r[2].add64(lane, wd);
            r[3].add64(lane, wd * mse_i);
            r[4].add64(lane, wb);
            r[5].add64(lane, wb * mse_i);
            r[6].add64(lane, mse_i);
            if HL {
                let w1 = saturate((ry - HL1_Y_ANCHOR as f64).max(0.0), C_HL);
                let w2 = saturate((ry - HL2_Y_ANCHOR as f64).max(0.0), C_HL);
                r[20].add64(lane, w1);
                r[21].add64(lane, w1 * mse_i);
                r[22].add64(lane, w2);
                r[23].add64(lane, w2 * mse_i);
            }
            let var1 = (b2 - m1 * m1).max(0.0);
            let var2 = ((sq - b2) - m2 * m2).max(0.0);
            let n1 = (s - m1) / (var1 + C_MSCN_VAR).sqrt();
            let n2 = (dd - m2) / (var2 + C_MSCN_VAR).sqrt();
            let dn = n1 - n2;
            r[7].add64(lane, saturate(dn.abs(), C_MSCN_ABS));
            r[8].add64(lane, saturate(dn * dn, C_MSCN_SQ));
            let (cg, cl) = bounded_excess_pair(var2, var1, C_CONTRAST);
            r[9].add64(lane, cg);
            r[10].add64(lane, cl);
            r[11].add64(lane, bounded_dissim(var1, var2, C_CONTRAST, false));
            r[14].add64(lane, s);
            r[15].add64(lane, dd);
            r[16].add64(lane, s * s);
            r[17].add64(lane, dd * dd);
            r[18].add64(lane, (dd - s) * (dd + s));
            r[19].add64(lane, dd - s);
        }
        acc.sum_xmask += r[0].fin();
        acc.sum_lumt += r[1].fin();
        acc.ws_dark.den += r[2].fin();
        acc.ws_dark.num += r[3].fin();
        acc.ws_bright.den += r[4].fin();
        acc.ws_bright.num += r[5].fin();
        acc.sum_mse += r[6].fin();
        acc.sum_mscn += r[7].fin();
        acc.sum_mscn2 += r[8].fin();
        acc.sum_cgain += r[9].fin();
        acc.sum_closs += r[10].fin();
        acc.sum_tex += r[11].fin();
        acc.sum_art2 += r[12].fin();
        acc.sum_det2 += r[13].fin();
        acc.sum_s += r[14].fin();
        acc.sum_d += r[15].fin();
        acc.sum_s2 += r[16].fin();
        acc.sum_d2 += r[17].fin();
        acc.sum_dd += r[18].fin();
        acc.sum_ds += r[19].fin();
        if HL {
            acc.ws_hl1.den += r[20].fin();
            acc.ws_hl1.num += r[21].fin();
            acc.ws_hl2.den += r[22].fin();
            acc.ws_hl2.num += r[23].fin();
        }
    }
    acc
}

#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
fn append_block_kernel_entry_cross(
    token: Token,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    bs2: &[f32],
    activity: &[f32],
    ref_y: &[f32],
    act_x: &[f32],
    act_b: &[f32],
    width: usize,
    height: usize,
) -> AppendAccum {
    append_block_kernel_generic::<_, true, false>(
        token, src, dst, mu1, mu2, ssq, bs2, activity, ref_y, act_x, act_b, width, height,
    )
}

#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
fn append_block_kernel_entry_cross_hl(
    token: Token,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    bs2: &[f32],
    activity: &[f32],
    ref_y: &[f32],
    act_x: &[f32],
    act_b: &[f32],
    width: usize,
    height: usize,
) -> AppendAccum {
    append_block_kernel_generic::<_, true, true>(
        token, src, dst, mu1, mu2, ssq, bs2, activity, ref_y, act_x, act_b, width, height,
    )
}

#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
fn append_block_kernel_entry_nocross(
    token: Token,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    bs2: &[f32],
    activity: &[f32],
    ref_y: &[f32],
    width: usize,
    height: usize,
) -> AppendAccum {
    append_block_kernel_generic::<_, false, false>(
        token,
        src,
        dst,
        mu1,
        mu2,
        ssq,
        bs2,
        activity,
        ref_y,
        &[],
        &[],
        width,
        height,
    )
}

/// Runtime dispatch wrapper for the append kernel (same `incant!` shape as
/// [`dense_block_kernel`]). `cross` carries the X/B activity strips for the
/// Y channel; `None` compiles the cross-channel chain out entirely.
/// `hl`: append2 highlight bins on (Y channel, HDR route, `append2_block`)
/// — requires `cross` (Y-only by construction; asserted).
#[allow(clippy::too_many_arguments)]
fn append_block_kernel(
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    bs2: &[f32],
    activity: &[f32],
    ref_y: &[f32],
    cross: Option<(&[f32], &[f32])>,
    hl: bool,
    width: usize,
    height: usize,
) -> AppendAccum {
    // rev4canon: the computation's own mode selects the scalar
    // `Pool`-generic mirror — `LanesF64` (the Rev4 canon) or the oracle arms
    // under `ZENSIM_FEATCANON`. `compute_mode()` is `mode(comp_rev)` here;
    // `Off` below Rev4 keeps the shipped SIMD bodies byte-for-byte.
    let mode = crate::featcanon::compute_mode();
    if mode.active() {
        #[cfg(feature = "oracle")]
        use crate::featcanon::{LanesF32, Neum64};
        use crate::featcanon::{LanesF64, Mode};
        macro_rules! go {
            ($p:ty) => {
                match (cross, hl) {
                    (Some((ax, ab)), true) => append_block_kernel_canon::<$p, true, true>(
                        src, dst, mu1, mu2, ssq, bs2, activity, ref_y, ax, ab, width, height,
                    ),
                    (Some((ax, ab)), false) => append_block_kernel_canon::<$p, true, false>(
                        src, dst, mu1, mu2, ssq, bs2, activity, ref_y, ax, ab, width, height,
                    ),
                    (None, _) => append_block_kernel_canon::<$p, false, false>(
                        src,
                        dst,
                        mu1,
                        mu2,
                        ssq,
                        bs2,
                        activity,
                        ref_y,
                        &[],
                        &[],
                        width,
                        height,
                    ),
                }
            };
        }
        #[cfg(feature = "oracle")]
        macro_rules! go_exact {
            () => {
                match (cross, hl) {
                    (Some((ax, ab)), true) => append_block_kernel_exact::<true, true>(
                        src, dst, mu1, mu2, ssq, bs2, activity, ref_y, ax, ab, width, height,
                    ),
                    (Some((ax, ab)), false) => append_block_kernel_exact::<true, false>(
                        src, dst, mu1, mu2, ssq, bs2, activity, ref_y, ax, ab, width, height,
                    ),
                    (None, _) => append_block_kernel_exact::<false, false>(
                        src,
                        dst,
                        mu1,
                        mu2,
                        ssq,
                        bs2,
                        activity,
                        ref_y,
                        &[],
                        &[],
                        width,
                        height,
                    ),
                }
            };
        }
        return match mode {
            Mode::Canon64 => go!(LanesF64),
            #[cfg(feature = "oracle")]
            Mode::Canon32 => go!(LanesF32),
            #[cfg(feature = "oracle")]
            Mode::CanonNeum => go!(Neum64),
            #[cfg(feature = "oracle")]
            Mode::Exact => go_exact!(),
            Mode::Off => unreachable!("active() excludes Off"),
        };
    }
    match cross {
        Some((act_x, act_b)) if hl => {
            incant!(
                append_block_kernel_entry_cross_hl(
                    src, dst, mu1, mu2, ssq, bs2, activity, ref_y, act_x, act_b, width, height
                ),
                [v4x, v4, v3, neon, wasm128, scalar]
            )
        }
        Some((act_x, act_b)) => {
            incant!(
                append_block_kernel_entry_cross(
                    src, dst, mu1, mu2, ssq, bs2, activity, ref_y, act_x, act_b, width, height
                ),
                [v4x, v4, v3, neon, wasm128, scalar]
            )
        }
        None => {
            debug_assert!(!hl, "HL bins are Y-only (cross) by construction");
            incant!(
                append_block_kernel_entry_nocross(
                    src, dst, mu1, mu2, ssq, bs2, activity, ref_y, width, height
                ),
                [v4x, v4, v3, neon, wasm128, scalar]
            )
        }
    }
}

// ============================================================================
// CSFW (f944+) kernel — chunk-3 tier-1 second pass over the Y strip rows.
//
// A SEPARATE const-gated pass, not a growth of the append kernel: the append
// kernel already carries ~19 row-lane accumulators against the 16-register
// AVX2/NEON budget with accepted L1 spills — five more lanes land on the
// wrong side of that cliff (design §4.2). A second pass over strip-resident
// planes re-reads L2, not DRAM (§A.15's locality), so the marginal cost is
// close to the 10 ops/px of its own arithmetic. With `csfw_block` off this
// pass is never invoked and every existing kernel is untouched machine code
// — the byte-stability guarantee is structural.
// ============================================================================

/// Per-row-reduced f64 accumulators for the CSFW weighted-pool pass:
/// `Σw, Σw·s, Σw·d, Σw·s², Σw·d²` (design §4.1 — pure sums of products,
/// strip-foldable exactly like [`AppendAccum`]).
#[derive(Default, Clone, Copy)]
struct CsfwAccum {
    sum_w: f64,
    sum_ws: f64,
    sum_wd: f64,
    sum_ws2: f64,
    sum_wd2: f64,
}

impl CsfwAccum {
    /// Strip-partial fold, same reasoning as [`AppendAccum::accumulate`].
    #[inline]
    fn accumulate(&mut self, other: &CsfwAccum) {
        self.sum_w += other.sum_w;
        self.sum_ws += other.sum_ws;
        self.sum_wd += other.sum_wd;
        self.sum_ws2 += other.sum_ws2;
        self.sum_wd2 += other.sum_wd2;
    }
}

/// Per-walk CSFW parameters, derived once from the route: the effective
/// per-scale weight quadratic `w(y) = clamp(b0 + b1·y + b2·y², w_min,
/// w_max)` with `b = [1 + g·c0, g·c1, g·c2]`, `g = κ_Y·λ_b` folded at
/// walk setup (design §4.2: `κ_c·λ_b` folds to one constant per band —
/// no runtime multiply of the two) and `c` the route's derived φ
/// quadratic ([`CSFW_PHI_Y_SDR`]/[`CSFW_PHI_Y_PU`]).
#[derive(Clone, Copy)]
struct CsfwParams {
    eff: [[f64; 3]; crate::NUM_SCALES],
}

impl CsfwParams {
    fn for_phi(phi: [f64; 3]) -> Self {
        let eff = std::array::from_fn(|scale| {
            let g = CSFW_KAPPA_Y * CSFW_LAMBDA_B[scale];
            [1.0 + g * phi[0], g * phi[1], g * phi[2]]
        });
        Self { eff }
    }
}

/// CSFW SIMD kernel body — generic over any `F32x8Backend` token, same
/// row-lane-then-f64 reduction structure as [`append_block_kernel_generic`].
/// `ref_y` is the reference Y strip rows (for the Y channel these are the
/// same values as `src`; the argument stays separate so the weight is
/// explicitly a function of the REFERENCE plane only — design §4.1 — and
/// so a chroma tier can reuse the kernel unchanged).
#[inline(always)]
fn csfw_block_kernel_generic<T: F32x8Backend + Copy>(
    token: T,
    src: &[f32],
    dst: &[f32],
    ref_y: &[f32],
    eff: [f64; 3],
    width: usize,
    height: usize,
) -> CsfwAccum {
    let b0 = V8::<T>::splat_t(token, eff[0] as f32);
    let b1 = V8::<T>::splat_t(token, eff[1] as f32);
    let b2 = V8::<T>::splat_t(token, eff[2] as f32);
    let w_min = V8::<T>::splat_t(token, CSFW_W_MIN as f32);
    let w_max = V8::<T>::splat_t(token, CSFW_W_MAX as f32);

    let mut acc = CsfwAccum::default();
    let width8 = width - (width % 8);

    for y in 0..height {
        let row = y * width;
        let (mut r_w, mut r_ws, mut r_wd, mut r_ws2, mut r_wd2) = (
            V8::<T>::zero_t(token),
            V8::<T>::zero_t(token),
            V8::<T>::zero_t(token),
            V8::<T>::zero_t(token),
            V8::<T>::zero_t(token),
        );

        let mut x = 0usize;
        while x < width8 {
            let i = row + x;
            let s = V8::<T>::from_array_t(token, src[i..i + 8].try_into().unwrap());
            let dd = V8::<T>::from_array_t(token, dst[i..i + 8].try_into().unwrap());
            let ry = V8::<T>::from_array_t(token, ref_y[i..i + 8].try_into().unwrap());

            // w = clamp(b0 + y·(b1 + y·b2), w_min, w_max) — Horner, 2 FMA.
            let w = ry.mul_add(ry.mul_add(b2, b1), b0).max(w_min).min(w_max);
            let ws = w * s;
            let wd = w * dd;
            r_w += w;
            r_ws += ws;
            r_wd += wd;
            r_ws2 += ws * s;
            r_wd2 += wd * dd;

            x += 8;
        }

        acc.sum_w += r_w.reduce_add() as f64;
        acc.sum_ws += r_ws.reduce_add() as f64;
        acc.sum_wd += r_wd.reduce_add() as f64;
        acc.sum_ws2 += r_ws2.reduce_add() as f64;
        acc.sum_wd2 += r_wd2.reduce_add() as f64;

        // Scalar tail — same formulas in f64, the house tail idiom.
        for x in width8..width {
            let i = row + x;
            let s = src[i] as f64;
            let dd = dst[i] as f64;
            let ry = ref_y[i] as f64;
            let w = (eff[0] + ry * (eff[1] + ry * eff[2])).clamp(CSFW_W_MIN, CSFW_W_MAX);
            let ws = w * s;
            let wd = w * dd;
            acc.sum_w += w;
            acc.sum_ws += ws;
            acc.sum_wd += wd;
            acc.sum_ws2 += ws * s;
            acc.sum_wd2 += wd * dd;
        }
    }

    acc
}

#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
fn csfw_block_kernel_entry(
    token: Token,
    src: &[f32],
    dst: &[f32],
    ref_y: &[f32],
    eff: [f64; 3],
    width: usize,
    height: usize,
) -> CsfwAccum {
    csfw_block_kernel_generic(token, src, dst, ref_y, eff, width, height)
}

/// Runtime dispatch wrapper for the CSFW kernel (same `incant!` shape as
/// [`append_block_kernel`]). `revision` is the computation's formula
/// revision: Rev4 runs the canonical body (featcanon D1).
fn csfw_block_kernel(
    src: &[f32],
    dst: &[f32],
    ref_y: &[f32],
    eff: [f64; 3],
    width: usize,
    height: usize,
    revision: crate::feature_defs::FormulaRevision,
) -> CsfwAccum {
    let mode = crate::featcanon::mode(crate::ssim_form::effective_revision(revision));
    if mode.active() {
        return csfw_block_kernel_canon(src, dst, ref_y, eff, width, height, mode);
    }
    incant!(
        csfw_block_kernel_entry(src, dst, ref_y, eff, width, height),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

/// **FEATCANON canonical CSFW kernel.** One plain-Rust body — identical on
/// every tier by construction: the Horner weight `w = b0 + y·(b1 + y·b2)`
/// stays fused (`f32::mul_add`), the per-pixel form is the VECTOR body's
/// (the production scalar tail computes `w` in f64 — a different element
/// value — so canonical pins the chunk form for every pixel), and the five
/// sums run through `featcanon::Pool` (lane = x mod 8, one fixed reduce per
/// row). With the `oracle` feature, `exact` mode evaluates in f64 + Neumaier.
#[allow(clippy::manual_clamp)]
fn csfw_block_kernel_canon(
    src: &[f32],
    dst: &[f32],
    ref_y: &[f32],
    eff: [f64; 3],
    width: usize,
    height: usize,
    mode: crate::featcanon::Mode,
) -> CsfwAccum {
    use crate::featcanon::Mode;
    #[cfg(feature = "oracle")]
    use crate::featcanon::{Neum64, Pool as _};
    let b0 = eff[0] as f32;
    let b1 = eff[1] as f32;
    let b2 = eff[2] as f32;
    let w_min = CSFW_W_MIN as f32;
    let w_max = CSFW_W_MAX as f32;

    let mut acc = CsfwAccum::default();
    for y in 0..height {
        let row = y * width;
        #[cfg(feature = "oracle")]
        if mode.exact() {
            let mut p_w = Neum64::zero();
            let mut p_ws = Neum64::zero();
            let mut p_wd = Neum64::zero();
            let mut p_ws2 = Neum64::zero();
            let mut p_wd2 = Neum64::zero();
            for x in 0..width {
                let i = row + x;
                let s = src[i] as f64;
                let dd = dst[i] as f64;
                let ry = ref_y[i] as f64;
                let w = ry
                    .mul_add(ry.mul_add(eff[2], eff[1]), eff[0])
                    .max(CSFW_W_MIN)
                    .min(CSFW_W_MAX);
                let ws = w * s;
                let wd = w * dd;
                p_w.add64(x, w);
                p_ws.add64(x, ws);
                p_wd.add64(x, wd);
                p_ws2.add64(x, ws * s);
                p_wd2.add64(x, wd * dd);
            }
            acc.sum_w += p_w.fin();
            acc.sum_ws += p_ws.fin();
            acc.sum_wd += p_wd.fin();
            acc.sum_ws2 += p_ws2.fin();
            acc.sum_wd2 += p_wd2.fin();
            continue;
        }
        match mode {
            #[cfg(feature = "oracle")]
            Mode::Canon32 => {
                csfw_row_canon::<crate::featcanon::LanesF32>(
                    &mut acc, src, dst, ref_y, row, width, b0, b1, b2, w_min, w_max,
                );
            }
            Mode::Canon64 => {
                csfw_row_canon::<crate::featcanon::LanesF64>(
                    &mut acc, src, dst, ref_y, row, width, b0, b1, b2, w_min, w_max,
                );
            }
            #[cfg(feature = "oracle")]
            Mode::CanonNeum => {
                csfw_row_canon::<Neum64>(
                    &mut acc, src, dst, ref_y, row, width, b0, b1, b2, w_min, w_max,
                );
            }
            _ => unreachable!("csfw canon called outside a canon mode"),
        }
    }
    acc
}

/// One row of the canonical CSFW accumulation (Pool-selected lanes).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::manual_clamp)]
fn csfw_row_canon<P: crate::featcanon::Pool>(
    acc: &mut CsfwAccum,
    src: &[f32],
    dst: &[f32],
    ref_y: &[f32],
    row: usize,
    width: usize,
    b0: f32,
    b1: f32,
    b2: f32,
    w_min: f32,
    w_max: f32,
) {
    let (mut p_w, mut p_ws, mut p_wd, mut p_ws2, mut p_wd2) =
        (P::zero(), P::zero(), P::zero(), P::zero(), P::zero());
    for x in 0..width {
        let i = row + x;
        let s = src[i];
        let dd = dst[i];
        let ry = ref_y[i];
        let w = ry.mul_add(ry.mul_add(b2, b1), b0).max(w_min).min(w_max);
        let ws = w * s;
        let wd = w * dd;
        p_w.add(x, w);
        p_ws.add(x, ws);
        p_wd.add(x, wd);
        p_ws2.add(x, ws * s);
        p_wd2.add(x, wd * dd);
    }
    acc.sum_w += p_w.fin();
    acc.sum_ws += p_ws.fin();
    acc.sum_wd += p_wd.fin();
    acc.sum_ws2 += p_ws2.fin();
    acc.sum_wd2 += p_wd2.fin();
}

/// Finalize one scale's CSFW block into its 3 output slots — the weighted
/// twins of [`finish_append`]'s GLOBAL_* trio, same constants, same
/// clamps. `Σw ≥ w_min·n > 0` structurally; the [`WeightedSum`]-class
/// 1e-12 denominator floor is kept as defense in depth.
fn finish_csfw(csfw: &CsfwAccum, out: &mut [f64]) {
    debug_assert_eq!(out.len(), CSFW_PER_SCALE);
    if csfw.sum_w < 1e-12 {
        out[idx_csfw::W_GLOBAL_DMEAN] = 0.0;
        out[idx_csfw::W_GLOBAL_CGAIN] = 0.0;
        out[idx_csfw::W_GLOBAL_CLOSS] = 0.0;
        return;
    }
    out[idx_csfw::W_GLOBAL_DMEAN] =
        saturate((csfw.sum_ws - csfw.sum_wd).abs() / csfw.sum_w, C_GDMEAN);
    let wmean_s = csfw.sum_ws / csfw.sum_w;
    let wmean_d = csfw.sum_wd / csfw.sum_w;
    let gvar1_w = (csfw.sum_ws2 / csfw.sum_w - wmean_s * wmean_s).max(0.0);
    let gvar2_w = (csfw.sum_wd2 / csfw.sum_w - wmean_d * wmean_d).max(0.0);
    let (g_cgain, g_closs) = bounded_excess_pair(gvar2_w, gvar1_w, C_GCONTRAST);
    out[idx_csfw::W_GLOBAL_CGAIN] = g_cgain.clamp(0.0, 1.0);
    out[idx_csfw::W_GLOBAL_CLOSS] = g_closs.clamp(0.0, 1.0);
}

/// Finalize one channel-scale's append block into its 17 output slots.
/// `dense`/`grad` supply the first moments matching `app`'s second moments
/// (identical per-pixel values by construction — see [`AppendAccum`]).
/// The three `GLOBAL_*` append slots, from the four raw moments — the ONE
/// implementation, shared by [`finish_append`] (which reads the moments off
/// the append kernel) and the free-extras finalize (which reads the
/// BIT-SAME-DEFINITION moments off the v1 fused kernel). Sharing it is what
/// makes `free_extras_match_append_block` a gate on the ACCUMULATION rather
/// than a comparison of two hand-copied formulas.
///
/// Returns `(GLOBAL_DMEAN, g_cgain, g_closs)`; the caller applies its own
/// clamp to the pair, exactly as before.
#[inline]
fn global_stats_from_raw_moments(
    sum_s: f64,
    sum_d: f64,
    sum_s2: f64,
    sum_d2: f64,
    sum_dd: f64,
    sum_ds: f64,
    n_f: f64,
    paired: bool,
) -> (f64, f64, f64) {
    let dmean = saturate((sum_s - sum_d).abs() / n_f, C_GDMEAN);
    let gmean_s = sum_s / n_f;
    let gmean_d = sum_d / n_f;
    let gvar1 = (sum_s2 / n_f - gmean_s * gmean_s).max(0.0);
    let gvar2 = (sum_d2 / n_f - gmean_d * gmean_d).max(0.0);
    if paired {
        // REVISION 2 (`freecomp`). `bounded_excess_pair` needs the DIFFERENCE
        // `gvar2 - gvar1` and the SUM `gvar1 + gvar2`. Only the difference is
        // ill-conditioned, and only because each variance is itself
        // `sum(x^2)/n - mean^2` — so the difference of two variances is a
        // difference of differences of large nearly-equal numbers.
        //
        // Reassociating removes the amplifier exactly:
        //     gvar2 - gvar1 = [sum(d^2) - sum(s^2)]/n - (md^2 - ms^2)
        //                   = sum((d-s)(d+s))/n - (md - ms)(md + ms)
        // Both terms are now O(distortion) rather than O(mean^2), and
        // `sum_dd` is accumulated per PIXEL so its cancellation happens on one
        // pixel's small difference instead of between two large totals. The
        // SUM keeps the raw form: when the variances are small enough for
        // their error to matter, `C_GCONTRAST` (1e-4) already dominates the
        // denominator.
        //
        // WHY THIS AND NOT THE OBVIOUS TWO, both tried and MEASURED to fail
        // (`benchmarks/feature_rev2_2026-09-05.md`, 60 real source images,
        // paired cells, free-vs-append past the 2e-5 bar):
        //   * matching the append kernel's per-ROW reduction granularity in
        //     the free walk: 4.47 % -> 3.90 %, worst cell WORSE. The error is
        //     within one row, not across rows.
        //   * Kahan-compensating the f32 second moments: compensating the FREE
        //     route alone made agreement WORSE (4.97 % -> 5.39 %), which is
        //     what identified the APPEND route as the inaccurate one;
        //     compensating BOTH still left 9.66 %.
        // The reason both fail is that they chase precision in `sum(s^2)`
        // while the amplification is `mean^2/gvar`. With `sum(s^2)` accurate
        // to one f32 ulp the predicted feature error is
        // `1.2e-7 * mean^2 / (gvar1+gvar2+C)`, which for a flat region is
        // ~3e-4 — and the MEASURED worst cell was 3.70e-4. The arithmetic
        // says no f32 accumulation scheme can fix this; only removing the
        // amplifier can.
        //
        // `GLOBAL_DMEAN` is untouched: it never squares, and it MEASURED
        // clean (0 of 649 cells past the bar) while `GLOBAL_CGAIN` was 39.8 %
        // and `GLOBAL_CLOSS` 11.6 %.
        let diff = sum_dd / n_f - (sum_ds / n_f) * (gmean_d + gmean_s);
        let recip_denom = 1.0 / (gvar1 + gvar2 + C_GCONTRAST);
        return (
            dmean,
            diff.max(0.0) * recip_denom,
            (-diff).max(0.0) * recip_denom,
        );
    }
    let (g_cgain, g_closs) = bounded_excess_pair(gvar2, gvar1, C_GCONTRAST);
    (dmean, g_cgain, g_closs)
}
