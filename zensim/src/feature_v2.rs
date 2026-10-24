//! V2 "bounded" feature extraction — additive regime.
//!
//! Gated behind `feature-regime-v2` (default ON since 2026-09-01 — the
//! Profile-D no-tax refactor, `benchmarks/profile_d_notax_2026-09-01.md`;
//! `--no-default-features` still removes this whole module). Full defect inventory
//! (D1-D9), design principles, per-feature formula/bound/citation table,
//! and iteration-1 as-built: `docs/FEATURE_V2_SPEC_2026-07-18.md` Part A/B.
//! This file is iteration 2 ("phase 2" / "feature-v2b"): §A.11/§A.12 of
//! that doc cover the fused kernel design, spill counts, per-feature
//! marginal cost, and the speed-gate verdict.
//!
//! # Iteration 2: single-pass, O(1)-accumulator kernel
//!
//! Iteration 1 stored NINE full-image `Vec<f32>` per-pixel maps (one per
//! basic-family signal) and processed them in two passes. Measured against
//! v1 (`docs/FEATURE_V2_SPEC_2026-07-18.md` §A.12): 2-5x SLOWER than v1's
//! full 372-feature extraction — an unacceptable baseline for adding more
//! features on top. Iteration 2 replaces this with a genuinely single-pass
//! kernel using O(1) running accumulators:
//!
//! - **Simple means** (art, det, mse, hf_*, pjnd_*, gms, ringing, banding,
//!   blockiness): a running sum, divided by `n` at the end. No storage.
//! - **SSIM mean/dev2/dev4**: Terriberry's (2007) single-pass online
//!   higher-order moments (mean + M2 + M3 + M4 running scalars) — the
//!   GMSD-style deviation-from-mean moments (iteration 1's D8 fix) computed
//!   WITHOUT a second pass or any array, unlike iteration 1's
//!   store-then-revisit approach.
//! - **Masked / IW / soft-peak weighted pooling**: running
//!   `(Σw·v, Σw)` numerator/denominator pairs, divided at the end
//!   (`WeightedPool`'s `Σw·v/Σw` form, computed inline rather than
//!   materializing weight or value arrays).
//!
//! The only per-scale arrays that remain are the blur-pass outputs
//! (`mu1`, `mu2`, `s12`, `ssq`, `activity`) — these are inherently
//! multi-pixel (a box blur cannot be expressed as a running per-pixel
//! accumulator), and v1's own architecture carries the same five arrays
//! for the same reason (`streaming.rs`). Everything downstream of them is
//! now one `for y { for x { ... } }` loop with O(1) extra space.
//!
//! # Iteration 2: seven new candidates (`docs/FEATURE_V2_SPEC_2026-07-18.md`
//! §A.10, phase-congruency explicitly excluded — LARGE/log-Gabor cost)
//!
//! All reuse the SAME per-pixel gradient computed once per (channel,
//! scale) pixel and shared across every gradient-consuming feature (GMS,
//! chroma-edge-GMS [=GMS evaluated on the X/B channels via the existing
//! per-channel loop — no extra computation], ringing, banding, oriented
//! blockiness, edge-width-change) — see `idx` module and
//! `compute_channel_scale_v2`'s per-pixel loop.

use crate::det_math::DetRoots;
use crate::error::ZensimError;
use crate::source::ImageSource;

mod restore_cuts;

use archmage::incant;
use archmage::magetypes;
use magetypes::simd::backends::{F32x8Backend, F32x16Backend};
use magetypes::simd::generic::{f32x8 as GenericF32x8, f32x16 as GenericF32x16};

// ============================================================================
// Layout constants
// ============================================================================

/// Bounded-basic block: SSIM mean + Terriberry 2nd/4th deviation moments
/// (D1, D8 fix), bounded edge artifact/detail (D3 fix), saturating MSE (D6
/// partial fix), bounded HF gain/loss/mag (D2 fix).
pub const FEATURES_PER_CHANNEL_V2_BASIC: usize = 9;
/// Soft-saliency-weighted-mean replacement for v1's hard max/L8 peak block
/// (D4 fix).
pub const FEATURES_PER_CHANNEL_V2_PEAK: usize = 3;
/// Masked (flatness-weighted) block using the canonical `Σw·v/Σw` pooling
/// (D5 fix), bounded activity weight (D6 fix).
pub const FEATURES_PER_CHANNEL_V2_MASKED: usize = 4;
/// IW (texture-weighted) block — same canonical pooling, opposite polarity.
pub const FEATURES_PER_CHANNEL_V2_IW: usize = 4;
/// Near-threshold (PJND) core block: divisive-normalization transducer
/// (`k=4.0`) + reference-only masking-susceptibility.
pub const FEATURES_PER_CHANNEL_V2_PJND: usize = 2;
/// Phase-2 additions (A.10 candidates, phase-congruency excluded): GMS
/// (serves both "gradient-magnitude similarity" and, via the X/B channels
/// of the existing per-channel loop, "chroma-edge similarity" — 1 slot for
/// 2 candidates), masking-transducer bank low/high k (2 slots, 1
/// candidate), oriented blockiness, ringing, banding, edge-width-change (4
/// slots, 4 candidates). 1+2+1+1+1+1 = 7 slots for 7 named candidates.
pub const FEATURES_PER_CHANNEL_V2_NEW: usize = 7;
/// Total v2 signals per channel per scale (9+3+4+4+2+7 = 29).
pub const FEATURES_PER_CHANNEL_V2_TOTAL: usize = FEATURES_PER_CHANNEL_V2_BASIC
    + FEATURES_PER_CHANNEL_V2_PEAK
    + FEATURES_PER_CHANNEL_V2_MASKED
    + FEATURES_PER_CHANNEL_V2_IW
    + FEATURES_PER_CHANNEL_V2_PJND
    + FEATURES_PER_CHANNEL_V2_NEW;

/// Named local offsets within one channel's v2 block.
pub mod idx {
    pub const SSIM_MEAN: usize = 0;
    pub const SSIM_DEV2: usize = 1;
    pub const SSIM_DEV4: usize = 2;
    pub const ART: usize = 3;
    pub const DET: usize = 4;
    pub const MSE: usize = 5;
    pub const HF_GAIN: usize = 6;
    pub const HF_LOSS: usize = 7;
    pub const HF_MAG_LOSS: usize = 8;
    pub const SSIM_SOFT_PEAK: usize = 9;
    pub const ART_SOFT_PEAK: usize = 10;
    pub const DET_SOFT_PEAK: usize = 11;
    pub const MASKED_SSIM: usize = 12;
    pub const MASKED_ART: usize = 13;
    pub const MASKED_DET: usize = 14;
    pub const MASKED_MSE: usize = 15;
    pub const IW_SSIM: usize = 16;
    pub const IW_ART: usize = 17;
    pub const IW_DET: usize = 18;
    pub const IW_MSE: usize = 19;
    pub const PJND_TRANSDUCER: usize = 20;
    pub const PJND_FRAGILITY: usize = 21;
    /// Gradient-magnitude similarity (GMSD, Xue et al. 2013). Evaluated
    /// uniformly across X/Y/B channels via the existing per-channel loop,
    /// so `gms(scale, 0)`/`gms(scale, 2)` ARE the chroma-edge-similarity
    /// values (FSIMc, Zhang 2011) — no separate slot needed.
    pub const GMS: usize = 22;
    /// Masking-transducer bank, low-k member (`k=1.0`, less aggressive
    /// masking than the core `PJND_TRANSDUCER`'s `k=4.0`).
    pub const PJND_TRANSDUCER_LOW_K: usize = 23;
    /// Masking-transducer bank, high-k member (`k=16.0`, more aggressive
    /// masking).
    pub const PJND_TRANSDUCER_HIGH_K: usize = 24;
    /// Oriented blockiness (Wang-Bovik pattern, FR-ized against the
    /// reference's own step energy at the same lattice positions).
    pub const BLOCKINESS: usize = 25;
    /// Ringing (dilated-edge form; `activity` stands in for `dilate`).
    pub const RINGING: usize = 26;
    /// Banding / contour detector (cheap single-scale approximation of
    /// CAMBI's step-energy-in-low-gradient-regions signature).
    pub const BANDING: usize = 27;
    /// Edge-width change (two-scale gradient-decay-ratio approximation).
    /// The ONE scale-level (not per-pixel) exception to full
    /// spatializability in this feature set — see its doc below.
    pub const EDGE_WIDTH_CHANGE: usize = 28;
}

// ============================================================================
// Bounding constants — each documented + cited (full derivations in
// docs/FEATURE_V2_SPEC_2026-07-18.md)
// ============================================================================

/// SSIM luminance stability constant, v2. `C1 = (K1*L)^2`, `K1=0.01` (Wang,
/// Bovik, Sheikh, Simoncelli 2004, IEEE TIP 13(4)), `L=1` (matches this
/// crate's existing `simd_ops::C2 = 0.0009 = (0.03*1)^2` derivation).
/// Restores the standard SSIM luminance form
/// `num_m = (2·μ1·μ2 + C1) / (μ1² + μ2² + C1)` in place of v1's
/// `num_m = 1 - (μ1-μ2)²` (D1 fix). Boundedness requires μ1,μ2 ≥ 0, which
/// holds because [`crate::streaming::convert_source_to_xyb`] exclusively
/// calls the *positive*-XYB conversion variants.
pub const C1_V2: f64 = 0.0001;
/// Structure/contrast stability constant, v2 — same value/derivation as
/// v1's `simd_ops::C2`, kept independent so v2 never silently inherits a
/// future v1-only change.
pub const C2_V2: f64 = 0.0009;
/// GMSD/FSIM/DISTS bounded-similarity stabilizer for edge artifact/detail.
pub const C_EDGE: f64 = 1e-4;
/// Saturating half-point for bounded MSE: `mse=0.5` at squared error
/// `C_MSE` (i.e. `sqrt(C_MSE)=0.1` absolute XYB difference).
pub const C_MSE: f64 = 0.01;
/// Stabilizer for the HF gain/loss/mag-loss bounded-excess forms.
pub const C_HF: f64 = 1e-4;
/// Saturating half-point for the soft-saliency peak weight (D4 fix).
pub const C_PEAK: f64 = 0.05;
/// Saturating half-point bounding the reference-activity signal before use
/// as a masked/IW pooling weight (D6 fix).
pub const C_ACTIVITY: f64 = 0.01;
/// Weight floor for the IW pool (prevents an all-zero weight sum on a
/// flat image).
pub const IW_WEIGHT_FLOOR: f64 = 1e-3;
/// ColorVideoVDP-style (Mantiuk et al. 2024, arXiv:2401.11485) final
/// soft-clamp half-point for the PJND transducer bank.
pub const C_PJND_CLAMP: f64 = 0.1;
/// Saturating half-point for the reference-only gradient-energy signal
/// (Bondžulić et al. 2022).
pub const C_PJND_GRAD: f64 = 0.02;
/// Core transducer masking strength (matches v1's `k_mask`/`k_iw` order of
/// magnitude).
pub const K_PJND_MASK: f64 = 4.0;
/// Masking-transducer bank: low-k member (A.10 "2-3 spaced k values" —
/// geometric spacing ×4 around the core `K_PJND_MASK`). Less aggressive
/// masking normalization — closer to raw error, catches near-threshold
/// distortions the core transducer already partly suppresses.
pub const K_PJND_MASK_LOW: f64 = 1.0;
/// Masking-transducer bank: high-k member. More aggressive masking —
/// only distortion that survives heavy local-activity normalization
/// registers, isolating supra-threshold-only error.
pub const K_PJND_MASK_HIGH: f64 = 16.0;

/// GMSD (Xue, Zhang, Mou, Bovik 2013, arXiv:1308.3052, Eq.4) bounded-
/// similarity stabilizer for gradient magnitude comparison. Same order of
/// magnitude as `C_EDGE` (gradients here are raw-pixel central differences
/// in unit-XYB scale, not blur residuals, but the dynamic range is
/// comparable).
pub const C_GMS: f64 = 1e-4;
use crate::gmsbank_constants::{GMSBANK_B_C, GMSBANK_C, GMSBANK_CS_C, GMSBANK_X_C};
/// Saturating half-point for the reference-edge indicator used by ringing
/// (`edge_r` in the A.10 table's `err · dilate(edge_r) · (1−edge_r)` form).
pub const C_RING_EDGE: f64 = 0.02;
/// Saturating half-point bounding the raw per-pixel error before it enters
/// the ringing product (keeps ringing bounded regardless of input dynamic
/// range, same D6-style rationale as `C_ACTIVITY`).
pub const C_RING_ERR: f64 = 0.05;

/// Which per-pixel error the ringing family measures.
///
/// A measurement arm selected per process by `ZENSIM_RING_FORM`, modelled on
/// `ZENSIM_HF_GAIN` ([`crate::hf_gain_form::active_gain_form`]): no shipping
/// path sets it, and every stored table and bake is [`Self::Raw`].
///
/// imazen/zensim#72 (`benchmarks/byconstruction_2026-10-22.md`): [`Self::Raw`]
/// saturates `|src − dst|`, which a global offset, contrast or saturation
/// change makes uniformly large off edges. Measured on 24 references, a
/// +8 luma shift was ringing's largest response, 2.4× a visible σ1.2 blur.
/// [`Self::HighPassMu`] measures the error after removing each image's SSIM
/// local mean (`mu1`, `mu2`, already computed for the dense kernel), so a
/// constant offset or a linear ramp contributes nothing away from the image
/// border, while noise and oscillation next to edges still do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum RingForm {
    /// `sat(|e|, C_RING_ERR) · sat(act, C_ACTIVITY) · (1 − sat(|∇src|, C_RING_EDGE))`,
    /// `e = src − dst`. The shipped form.
    #[default]
    Raw,
    /// The same product with `e = (src − mu1) − (dst − mu2)`.
    HighPassMu,
}

impl RingForm {
    /// The `ZENSIM_RING_FORM` spelling. Both values have the same byte length
    /// on purpose (see [`crate::ssim_form::active_revision`]).
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::HighPassMu => "hpm",
        }
    }
}

/// **The ring form for this process**, read once.
///
/// `ZENSIM_RING_FORM=hpm` selects [`RingForm::HighPassMu`]; unset or `raw`
/// keeps [`RingForm::Raw`]. Any other value panics, so a typo can never
/// measure the wrong arm silently.
pub(crate) fn active_ring_form() -> RingForm {
    use std::sync::OnceLock;
    static FORM: OnceLock<RingForm> = OnceLock::new();
    *FORM.get_or_init(|| match std::env::var("ZENSIM_RING_FORM").as_deref() {
        Err(_) | Ok("raw") => RingForm::Raw,
        Ok("hpm") => RingForm::HighPassMu,
        Ok(other) => panic!("ZENSIM_RING_FORM={other:?}: expected \"raw\" or \"hpm\""),
    })
}

/// The ringing form of the process-revision legacy paths (the diffmap and the
/// caller-gradient attribution passes, which read
/// [`crate::ssim_form::active_revision`] rather than a request's revision):
/// high-pass under `ZENSIM_RING_FORM=hpm` or in a revision-6 process.
pub(crate) fn process_ring_high_pass() -> bool {
    active_ring_form() == RingForm::HighPassMu
        || crate::ssim_form::active_revision().ring_high_pass()
}

#[cfg(test)]
mod ring_form_tests {
    use super::*;

    /// Deterministic xorshift values in `[lo, hi)`.
    struct Draw(u64);
    impl Draw {
        fn next(&mut self, lo: f64, hi: f64) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (lo + (hi - lo) * ((self.0 >> 11) as f64 / (1u64 << 53) as f64)) as f32
        }
    }

    /// 3×3 box mean with clamped borders: a stand-in local mean. The kernel
    /// takes whatever `mu` planes the walk supplies; these tests only need
    /// `mu(dst) = mu(src) + k` for `dst = src + k`.
    fn box3(p: &[f32], w: usize, h: usize) -> Vec<f32> {
        let mut out = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut s = 0.0f32;
                for dy in [-1i64, 0, 1] {
                    for dx in [-1i64, 0, 1] {
                        let yy = (y as i64 + dy).clamp(0, h as i64 - 1) as usize;
                        let xx = (x as i64 + dx).clamp(0, w as i64 - 1) as usize;
                        s += p[yy * w + xx];
                    }
                }
                out[y * w + x] = s / 9.0;
            }
        }
        out
    }

    /// `(src_h, dst_h, act, mu1, mu2)`.
    type Planes = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

    /// `(src_h, dst_h, act, mu1, mu2)` for a `w×h` interior with one halo row
    /// above and below, `dst = f(src)`.
    fn planes(w: usize, h: usize, seed: u64, f: impl Fn(f32, &mut Draw) -> f32) -> Planes {
        let mut r = Draw(seed);
        let src_h: Vec<f32> = (0..(h + 2) * w).map(|_| r.next(0.2, 0.8)).collect();
        let dst_h: Vec<f32> = src_h.iter().map(|&v| f(v, &mut r)).collect();
        let act: Vec<f32> = (0..w * h).map(|_| r.next(0.0, 0.5)).collect();
        let mu1 = box3(&src_h[w..w + w * h], w, h);
        let mu2 = box3(&dst_h[w..w + w * h], w, h);
        (src_h, dst_h, act, mu1, mu2)
    }

    fn run(hp: bool, p: &Planes, w: usize, h: usize) -> GradientAccum {
        let (src_h, dst_h, act, mu1, mu2) = p;
        let m = crate::featcanon::Mode::Canon64;
        if hp {
            gradient_block_kernel_canon::<
                crate::featcanon::Lanes16F64,
                false,
                false,
                false,
                true,
                false,
            >(src_h, dst_h, act, &[], mu1, mu2, w, h, 0.0, 0.0, None, m)
        } else {
            gradient_block_kernel_canon::<
                crate::featcanon::Lanes16F64,
                false,
                false,
                false,
                false,
                false,
            >(src_h, dst_h, act, &[], mu1, mu2, w, h, 0.0, 0.0, None, m)
        }
    }

    #[test]
    fn a_global_offset_does_not_register_under_hpm_and_only_ringing_moves() {
        let (w, h) = (37usize, 11usize);
        let p = planes(w, h, 0x5249_4e47_4850_3403, |v, _| v + 0.03);
        let (raw, hp) = (run(false, &p, w, h), run(true, &p, w, h));
        assert!(raw.sum_ringing > 1.0, "{}", raw.sum_ringing);
        assert!(
            hp.sum_ringing < 1e-4 * raw.sum_ringing,
            "{} vs {}",
            hp.sum_ringing,
            raw.sum_ringing
        );
        for (a, b) in [
            (raw.sum_gms, hp.sum_gms),
            (raw.sum_gms2, hp.sum_gms2),
            (raw.sum_banding, hp.sum_banding),
            (raw.sum_grad_src, hp.sum_grad_src),
            (raw.sum_grad_dst, hp.sum_grad_dst),
        ] {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn noise_still_registers_under_hpm() {
        let (w, h) = (37usize, 11usize);
        let p = planes(w, h, 0x5249_4e47_4850_3404, |v, r| v + r.next(-0.05, 0.05));
        let (raw, hp) = (run(false, &p, w, h), run(true, &p, w, h));
        assert!(
            hp.sum_ringing > 0.3 * raw.sum_ringing,
            "{} vs {}",
            hp.sum_ringing,
            raw.sum_ringing
        );
    }

    #[test]
    fn hpm_matches_an_independent_per_pixel_sum() {
        // Every pixel's term recomputed here in f64 from the definition; the
        // kernel's f32 terms pooled in f64 must agree to f32 precision.
        let (w, h) = (29usize, 7usize);
        let p = planes(w, h, 0x5249_4e47_4850_3405, |v, r| v + r.next(-0.08, 0.08));
        let (src_h, dst_h, act, mu1, mu2) = &p;
        let mut want = 0.0f64;
        for y in 0..h {
            for x in 0..w {
                let at = |pl: &[f32], yy: usize, xx: usize| pl[yy * w + xx] as f64;
                let (xl, xr) = (x.saturating_sub(1), (x + 1).min(w - 1));
                let gx = at(src_h, y + 1, xr) - at(src_h, y + 1, xl);
                let gy = at(src_h, y + 2, x) - at(src_h, y, x);
                let g = (gx * gx + gy * gy).sqrt();
                let i = y * w + x;
                let e =
                    (at(src_h, y + 1, x) - mu1[i] as f64) - (at(dst_h, y + 1, x) - mu2[i] as f64);
                want += saturate(e.abs(), C_RING_ERR)
                    * saturate(act[i] as f64, C_ACTIVITY)
                    * (1.0 - saturate(g, C_RING_EDGE));
            }
        }
        let got = run(true, &p, w, h).sum_ringing;
        assert!((got - want).abs() <= 1e-5 * want, "{got} vs {want}");
    }
}
/// Saturating half-point for "distorted now has a noticeable edge here"
/// in the banding approximation.
pub const C_BAND_DST: f64 = 0.01;
/// Saturating half-point for "source was smooth here" in the banding
/// approximation.
pub const C_BAND_SRC: f64 = 0.02;
/// Bounded-excess stabilizer for oriented blockiness (FR-ized step-energy
/// comparison at the 8-pixel lattice).
pub const C_BLOCK: f64 = 1e-3;
/// Bounded-similarity stabilizer for the edge-width-change scale-decay
/// comparison.
pub const C_EDGEWIDTH: f64 = 1e-3;
/// Additive floor in the per-scale gradient-decay ratio denominator
/// (`decay = mean_grad(coarser) / (mean_grad(finer) + C_GRAD_DECAY)`).
pub const C_GRAD_DECAY: f64 = 1e-4;

// ============================================================================
// f720+ APPEND block — additions from the 2026-07-26 gap audit
// (`zenpapers/docs/zensim-720-feature-gaps-2026-07-26.md` §5, candidates
// A1-A5 + A9). Append-only: emitted AFTER the frozen f0..f719 layout, in a
// separate region `f720 + scale*(3*17) + ch*17 + local` (scale-major, same
// convention as the v2 block). Gated by `V2NewFeatureToggles::append_block`
// (default OFF — every existing path/byte is unchanged when off). Computed
// by a SEPARATE second kernel pass over the cache-hot strip planes
// (`append_block_kernel`) so the tuned dense/gradient kernels are not
// touched (their §A.14/§A.16 register-pressure story stays intact) and
// f0..f719 bit-stability is structural, not incidental.
// ============================================================================

/// Features per channel per scale in the append block (17 × 3 ch × 4
/// scales = 204; full folded-append vector = 720 + 204 = 924).
pub const FEATURES_PER_CHANNEL_APPEND: usize = 17;

/// Skip the append kernel entirely for the B (yellow-violet) channel at
/// scale 0 — its 17 slots emit 0.0 (index-stable, deprecate-by-absence).
/// Grounds: the yellow-violet foveal resolution limit is ~53 ppd vs 94
/// achromatic (Ashraf/Chapiro/Mantiuk 2025), and butteraugli carries no B
/// channel at all in its two highest-frequency bands — finest-scale B
/// append features would model signal the eye cannot resolve, at 25% of
/// the append block's pixel cost (scale 0 is ~75% of pyramid pixels).
pub const APPEND_SKIP_B_SCALE0: bool = true;

/// Named local offsets within one channel's append block.
pub mod idx_append {
    /// Cross-channel masked transducer (gap-audit A1). Y channel ONLY:
    /// `err/(err + C_PJND_CLAMP·(1 + K_PJND_MASK·act_Y + K_XCH·(act_X +
    /// act_B)))` — the ColorVideoVDP-trained masking direction (chromatic
    /// channels mask achromatic; Switkes 1988; vdp-csf-perceptual-math.md
    /// Eq. 9-11 discussion). X/B slots emit 0.0 (deprecate-by-absence:
    /// luma does NOT mask chroma per the same trained matrix, and the
    /// chroma-side planes are not available without a second sweep).
    pub const XMASK_TRANSDUCER: usize = 0;
    /// Luminance-adapted transducer (A2): activity + local-luminance term
    /// in the divisive denominator, `t = sat(ref_Y, C_LUM_T)` standing in
    /// for the DeVries-Rose/Weber background-luminance dependence.
    /// Y channel ONLY (X/B emit 0.0): the 2026-07-19 luma-gate ablation
    /// measured the v2 chroma transducers as a broad CID22 cost with
    /// near-zero steering mass — the same masking family, so the append
    /// block ships luma-gated from the start
    /// (`zensim:benchmarks/v2_trainability_ab_2026-07-19.md:209-216`).
    pub const LUM_TRANSDUCER: usize = 1;
    /// Bounded error (`mse_i = sat((s−d)², C_MSE)` — the v1/v2 MSE
    /// vocabulary) pooled over the DARK reference-luminance soft bin,
    /// weight `(1−t)²` (A2: shadow banding/blocking that uniform pooling
    /// dilutes). Reference-only weights — same foldable `Σw·v/Σw` shape
    /// as the masked/IW families. Pools `mse_i` rather than a
    /// recomputed SSIM map: the luminance CONDITIONING is the feature;
    /// `mse_i` needs no extra plane loads and keeps the kernel inside
    /// the perf gate.
    pub const LUM_DARK_ERR: usize = 2;
    /// Mid-luminance soft bin, weight `2t(1−t)`.
    pub const LUM_MID_ERR: usize = 3;
    /// Bright soft bin, weight `t²`.
    pub const LUM_BRIGHT_ERR: usize = 4;
    /// Normalize-then-difference divisive comparison (A3, NLPD/MSCN
    /// shape — Laparra et al. 2017; BRISQUE's MSCN): per-pixel
    /// `sat(|r₁/σ₁′ − r₂/σ₂′|, C_MSCN_ABS)` mean, `r_i` the blur
    /// residual, `σ_i′ = sqrt(var_i + C_MSCN_VAR)`. Unlike the pjnd
    /// transducer (reference-masked raw error), BOTH images normalize by
    /// their OWN local energy first. The σ-split comes from the cached /
    /// replayed reference plane `bs2 = blur(src²)`:
    /// `var₁ = bs2 − mu1²`, `var₂ = (ssq − bs2) − mu2²` — no per-pair
    /// blur is ever run for it.
    pub const MSCN_DIFF_MEAN: usize = 5;
    /// Squared variant: `sat((n₁−n₂)², C_MSCN_SQ)` mean.
    pub const MSCN_DIFF_L2: usize = 6;
    /// Local contrast gain (A4): `bounded_excess(var₂, var₁, C_CONTRAST)`
    /// mean — distorted window has MORE energy (over-sharpen, ringing,
    /// added grain) — requires the σ-split (`d2` plane).
    pub const CONTRAST_GAIN: usize = 7;
    /// Local contrast loss: `bounded_excess(var₁, var₂, C_CONTRAST)` mean
    /// (washed-out / smoothed rendering as a field, not via HF proxies).
    pub const CONTRAST_LOSS: usize = 8;
    /// Alignment-free texture-energy dissimilarity (A4, the cheap DISTS
    /// texture term — Ding et al. 2020): `1 − bounded_sim(var₁, var₂,
    /// C_CONTRAST)` mean. Compares local contrast STATISTICS, not
    /// pixel-registered values.
    pub const TEXTURE_DISSIM: usize = 9;
    /// GMSD-style deviation pooling of the GMS map (A5; Xue et al. 2013
    /// §3.3 — std of the map, the documented reason GMSD beats GMSM).
    /// Zero when `gradient_features` is off.
    pub const GMS_DEV2: usize = 10;
    /// Deviation (std) of the edge-artifact map (A5).
    pub const ART_DEV2: usize = 11;
    /// Deviation (std) of the detail-lost map (A5).
    pub const DET_DEV2: usize = 12;
    /// Global per-channel mean shift (A9): `sat(|mean(s)−mean(d)|,
    /// C_GDMEAN)` — the windowed features under-weigh global casts.
    pub const GLOBAL_DMEAN: usize = 13;
    /// Global contrast gain: `bounded_excess(gvar₂, gvar₁, C_GCONTRAST)`
    /// on whole-plane variances.
    pub const GLOBAL_CGAIN: usize = 14;
    /// Global contrast loss: reverse polarity of [`GLOBAL_CGAIN`].
    pub const GLOBAL_CLOSS: usize = 15;
    /// Mean source gradient magnitude, saturated (A2.iv): the >92%
    /// first-JND PSNR predictor (Bondžulić et al. 2022 via
    /// jnd-sur.md:204-210) as a reference-side complexity conditioner.
    /// Reference-only (correct 0 in any steering fold, like
    /// `PJND_FRAGILITY`). Zero when `gradient_features` is off.
    pub const GRAD_SRC_MEAN: usize = 16;
}

/// Cross-channel masking strength for the chroma→luma term (append
/// [`idx_append::XMASK_TRANSDUCER`]). Same order of magnitude as
/// `K_PJND_MASK` (the CVVDP `k_{i,c}` values are trained; we seed at the
/// core masking strength and let the head learn usage via the feature
/// weight).
pub const K_XCH: f64 = 4.0;
/// Luminance-term strength in [`idx_append::LUM_TRANSDUCER`]'s divisive
/// denominator.
pub const K_LUM_ADAPT: f64 = 4.0;
/// Saturating half-point mapping the (cbrt-domain, ~[0,1]) reference Y
/// value to the bin parameter `t = y/(y+C_LUM_T)` used by the luminance
/// soft bins and the luminance transducer term. 0.35 puts `t=0.5` near
/// mid-gray in XYB's cube-root intensity scale.
pub const C_LUM_T: f64 = 0.35;
/// Variance floor inside the MSCN normalizer `σ′ = sqrt(var + C_MSCN_VAR)`
/// (≈ (0.032)² in unit-XYB scale — same role as BRISQUE's C, keeps the
/// normalized residual bounded by `|r|/sqrt(C_MSCN_VAR)`).
pub const C_MSCN_VAR: f64 = 1e-3;
/// Per-pixel saturating half-point for `|n₁−n₂|` (bounded-by-construction
/// per the D6 design principle).
pub const C_MSCN_ABS: f64 = 0.5;
/// Per-pixel saturating half-point for `(n₁−n₂)²`.
pub const C_MSCN_SQ: f64 = 0.25;
/// Stabilizer for the var-based contrast gain/loss/texture-sim family
/// (variances of unit-XYB windows are `activity²`-scale; matches
/// `C_ACTIVITY²` order).
pub const C_CONTRAST: f64 = 1e-4;
/// Saturating half-point for the global |Δmean| feature.
pub const C_GDMEAN: f64 = 0.02;
/// Stabilizer for the global variance gain/loss pair.
pub const C_GCONTRAST: f64 = 1e-4;
/// Saturating half-point for the mean source-gradient conditioner
/// (matches `C_PJND_GRAD` — same signal family).
pub const C_GRADM: f64 = 0.02;

// ============================================================================
// append2 (f924+) constants — BANDVIS + luminance conditioner + HL bins.
// δ values are EMPIRICAL (test `bandvis_delta_derivation_table` prints the
// measurement; derivation committed in
// `benchmarks/append2_bandvis_gates_2026-07-27.md`). PROVISIONAL until the
// measurement lands — the derivation test brackets them against the live
// front-ends so a front-end change that invalidates them fails loudly.
// ============================================================================

/// BANDVIS band-pass lower half-point, SDR/cbrt route: ≈0.5× the Y-plane
/// |∇| of a one-8-bit-code plateau step at mid-gray.
pub(crate) const BV_DELTA_LO_SDR: f32 = 0.00169;
/// Upper half-point, SDR route: ≈5× the one-code step.
pub(crate) const BV_DELTA_HI_SDR: f32 = 0.0169;
/// BANDVIS lower half-point, HDR/PU route: ≈0.5× the PU-Y |∇| of one
/// 10-bit PQ code step at 100 cd/m² (PU uniformity makes one constant
/// serve all luminances — the derivation table shows the spread).
pub(crate) const BV_DELTA_LO_PU: f32 = 0.00124;
/// Upper half-point, HDR/PU route: ≈5× the one-step gradient.
pub(crate) const BV_DELTA_HI_PU: f32 = 0.0124;
/// Stabilizer for the BANDVIS FR `bounded_excess` pair (indicators are
/// [0,1] products — same class as `C_EDGE`).
pub(crate) const C_BV: f64 = 1e-4;
/// HL bin 1 anchor: the HDR route's Y-plane value of GRAY at 100 cd/m²
/// (SDR white in PU-normalized units) — measured by
/// `bandvis_delta_derivation_table`; the bin weighs error ABOVE SDR white.
pub(crate) const HL1_Y_ANCHOR: f32 = 1.01;
/// HL bin 2 anchor: Y-plane value of gray at 1000 cd/m².
pub(crate) const HL2_Y_ANCHOR: f32 = 1.649;
/// Softness half-point for the HL soft steps `u/(u+c)`, `u = max(y − anchor, 0)`
/// — ½ the measured anchor spacing (1.649 − 1.010 ≈ 0.64) so the two bins
/// overlap smoothly (E2 partition caveat: they also overlap the existing
/// bright bin — trainers must treat the three as non-orthogonal).
pub(crate) const C_HL: f64 = 0.32;

/// append2 slots per scale (Y-only block — no channel axis; layout
/// `f924 + scale*APPEND2_PER_SCALE + local`, 4 scales × 5 = 20 slots,
/// full vector = 944). Documented deviation from the append block's
/// scale-major×channel layout: every append2 signal is Y-only by
/// design, so a channel axis would be 2/3 structural zeros.
pub const APPEND2_PER_SCALE: usize = 5;

/// The one channel the append2 block is computed on (Y). The block has no
/// channel axis in its LAYOUT (see [`APPEND2_PER_SCALE`]), but its signals
/// are accumulated from the Y-channel kernels, so the attribution density
/// contributes on this channel only.
///
/// Layout metadata, so it stays compiled in every configuration — but its
/// only non-test reader is the `custom-profiles`-gated attribution-density
/// cluster, hence the targeted dead-code allowance.
#[cfg_attr(not(feature = "custom-profiles"), allow(dead_code))]
pub(crate) const APPEND2_CHANNEL: usize = 1;

/// Named local offsets within one scale's append2 block (all Y-only).
pub mod idx_append2 {
    /// BANDVIS banding-visibility GAIN (gaps-doc §6b design, 2026-07-27;
    /// operator revised to CURVATURE during validation): per-pixel
    /// `band(|∇²Y|; δ_lo, δ_hi) · (1 − sat(act, C_ACTIVITY))` indicators
    /// on src and dst (`band(g) = sat(g, δ_lo)·(1 − sat(g, δ_hi))` —
    /// soft band-pass over the SECOND-difference magnitude, half-points
    /// at the EMPIRICAL one-code-step constants `BV_DELTA_*`, per
    /// route). Second differences, not first: |∇²| of a linear gradient
    /// is exactly 0 at any steepness, so smooth ramps stay out of the
    /// band at every scale (the first-difference form measured
    /// polarity-inverted on ramp fixtures — sub-step smooth gradients
    /// were indistinguishable from steps); a plateau step reports the
    /// full step at its flanking pixels, so the δ derivation is
    /// operator-independent. FR
    /// `bounded_excess(b_dst, b_src, C_BV)` mean-pooled. "NEW visible
    /// steps in flat regions" — the CAMBI job on existing machinery; the
    /// pyramid supplies the window sweep; the activity term supplies
    /// dither masking. Foldable (plain mean).
    ///
    /// MEASURED MECHANICS (2026-07-27 behavioral run — load-bearing for
    /// consumers): this is a COARSE-SCALE detector. At scale 0/1 a
    /// realistic gradient's own per-pixel |∇| sits inside the visibility
    /// band (a smooth ramp shallow enough to band is still ≥ ~¼ code per
    /// 2-px span), so the FR excess largely cancels; each downscale
    /// doubles smooth-gradient |∇| OUT of the band while plateau steps
    /// persist, so scales 2–3 are where posterization fires (exactly
    /// where CAMBI's 63-px windows land vs our radius-5 support). The
    /// scale-0/1 slots are kept index-stable; expect near-zero there.
    /// Steps beyond ~5 8-bit codes attenuate BY DESIGN (the band's upper
    /// edge — CAMBI's own contrast cap is 4 ten-bit levels; giant steps
    /// are edges, GMS/blockiness territory). Each step size therefore has
    /// a RESONANT scale (finer steps → finer scale); consumers should
    /// read the 4 per-scale slots as a response CURVE, not redundant
    /// copies (measured ladder matrix:
    /// `benchmarks/append2_bandvis_gates_2026-07-27.md`).
    ///
    /// OPT-IN VARIANT (`V2NewFeatureToggles::append2_dst_activity`,
    /// default OFF — 2026-08-02 P1.5 adjudication): GAIN becomes the
    /// pure-band FR excess pooled under the DST's own flatness weight
    /// (dst-activity plane). Production extraction runs with it OFF
    /// (adjudicated); rows extracted with it ON are a different feature
    /// definition for this slot — never column-mix. Record:
    /// `benchmarks/bandvis_dst_activity_2026-08-02.md`.
    pub const BANDVIS_GAIN: usize = 0;
    /// Reverse polarity: visible steps REMOVED (debanding credit).
    pub const BANDVIS_LOSS: usize = 1;
    /// Reference-only local-adaptation conditioner: `sat(mean(ref Y),
    /// C_LUM_T)` from the append kernel's existing `sum_s` accumulator
    /// (finalize-only — FREE). Correct-0 in any steering fold, like
    /// `PJND_FRAGILITY`/`GRAD_SRC_MEAN`.
    pub const LUMA_MEAN_REF: usize = 2;
    /// HDR-route-gated highlight bin 1: `Σw·mse_i/Σw` with
    /// `w = sat(max(y_ref − HL1_Y_ANCHOR, 0), C_HL)` — error weighted
    /// ABOVE SDR white (anchor = measured PU-Y of gray at 100 cd/m²).
    /// 0.0 on the SDR route (index-stable). E2 PARTITION CAVEAT: these
    /// bins overlap each other AND the existing `LUM_BRIGHT_ERR` bright
    /// bin — non-orthogonal by design; trainers must not treat the
    /// luminance bins as a partition of unity once append2 is on.
    pub const HL_BIN1: usize = 3;
    /// Highlight bin 2: anchor = measured PU-Y of gray at 1000 cd/m².
    pub const HL_BIN2: usize = 4;
}

// ============================================================================
// CSFW (f944+) constants — chunk-3 luminance-CSF tier-1: Y-only
// luminance-weighted GLOBAL_* pooling lanes
// (`docs/CSF_CHUNK3_DESIGN_2026-07-28.md`; coordinator descope 2026-07-28:
// tier-1 ships the 12 achromatic lanes f944..f955 only — the chroma tiers
// keep the f956..f979 claim). The φ quadratics are DERIVED AND FROZEN, not
// fitted: castleCSF's published achromatic-sustained luminance sensitivity
// (Eq. 21, constants 56.49/7.547/0.1445/5.583e−7/9.669e9) divided by each
// route's own encoding derivative (cube root on the SDR route; PU21
// `banding_glare` from `pu21.rs` on the HDR route), normalized at the
// route anchor (sRGB code 128 = 43.73 cd/m² SDR; 100 cd/m² HDR), quadratic
// least-squares in the ENCODED coordinate over codes 4–255 (SDR) /
// L ∈ [1, 4000] cd/m² (HDR — below ~0.5 cd/m² PU21's glare term and
// castleCSF's glare-free model disagree; the clamp owns that region).
// `csfw_phi_derivation_table` recomputes the derivation against the LIVE
// front-ends and brackets these constants — the `bandvis_delta_derivation_
// table` pattern that keeps them honest.
// ============================================================================

/// SDR-route achromatic weight quadratic `φ_Y(y) = c0 + c1·y + c2·y²`,
/// with `y` the LIVE reference Y-plane value
/// (`cbrt(rel + β) − cbrt(β) + 0.01`, β the opsin bias — measured through
/// `srgb_to_positive_xyb_planar_into`). LSQ over sRGB codes 4–253, this
/// refit: rms 0.092, max 0.431 (dark tail). Implementation-found
/// deviation from the design doc's §5.3 table: those values were fitted
/// in an idealized bias-free `cbrt(rel)` coordinate that the live plane
/// is NOT an affine map of (the opsin bias regularizes the doc's
/// dark-tail derivative collapse); §6's pre-composition rule resolves
/// in favor of the live coordinate, and `csfw_phi_derivation_table`
/// recomputes this exact fit and pins the constants to it.
pub(crate) const CSFW_PHI_Y_SDR: [f64; 3] = [1.77430, -5.81908, 4.04916];
/// HDR/PU-route achromatic weight quadratic, `y` the LIVE reference
/// PU-Y plane value (`pu21(mix)/PU_WHITE + 0.01`). LSQ over L ∈
/// [1, 4000] cd/m² log-uniform, this refit: rms 0.082, max 0.329 —
/// within 0.016 of the design §5.3 values (the doc's PU coordinate was
/// already live-accurate up to the +0.01 bias).
pub(crate) const CSFW_PHI_Y_PU: [f64; 3] = [0.78830, -1.10402, 0.30460];
/// Fitted amplitude for the achromatic luminance modulation — SEED 1.0
/// (design §5.3: `κ = 1` applies the derived castleCSF curve exactly;
/// `κ = 0` reproduces the unweighted features bit-for-bit). Stage-1/2
/// calibration (§9.2) adjusts this from measurement, never by hand.
pub(crate) const CSFW_KAPPA_Y: f64 = 1.0;
/// Per-band (per-scale) strength multipliers `λ_b`, achromatic only —
/// the published chromatic peak frequencies do NOT move with luminance
/// (castleCSF Eq. 24 discussion; design §3.3). `λ_2 ≡ 1` is the
/// identifiability anchor; all SEED 1.0. Pre-registered expectation for
/// the stage-1 fit: increasing toward finer scales (falsifier 2).
pub(crate) const CSFW_LAMBDA_B: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
/// Weight clamp bounds (design §5.3): bracket every derived curve (SDR
/// span 2.96×, HDR chroma 2.5×) and own the sub-0.5 cd/m² glare
/// disagreement + the fit's dark-tail extrapolation. No lane can be
/// annihilated (§3.2: supra-threshold flattening — MAD's premise).
pub(crate) const CSFW_W_MIN: f64 = 0.25;
/// Upper weight clamp bound (see [`CSFW_W_MIN`]).
pub(crate) const CSFW_W_MAX: f64 = 4.0;

/// CSFW slots per scale (Y-only tier-1 block — no channel axis, same
/// documented layout deviation as append2; layout
/// `f944 + scale*CSFW_PER_SCALE + local`, 4 scales × 3 = 12 slots, full
/// vector = 956). The chroma tiers (X/B weighted twins, design §8.1)
/// keep the f956..f979 claim and are NOT emitted by this wave.
pub const CSFW_PER_SCALE: usize = 3;

/// Named local offsets within one scale's CSFW block (all Y-only).
///
/// Each lane is the luminance-CSF-weighted twin of the same scale's
/// unweighted Y-channel [`idx_append`] GLOBAL_* statistic: the pooling
/// mean `Σv/n` becomes `Σw·v/Σw` with the per-pixel REF-side weight
/// `w(y) = clamp(1 + κ_Y·λ_b·φ_Y(y), CSFW_W_MIN, CSFW_W_MAX)` evaluated
/// at the reference Y-plane value (the same per-pixel `ref_y` the HL
/// bins read). Same constants (`C_GDMEAN`/`C_GCONTRAST`), same clamps,
/// same bounds as the unweighted twins. E2-CLASS CAVEAT: the weighted
/// twins are correlated with their unweighted originals BY CONSTRUCTION
/// (`κ→0` makes them identical) — trainers must treat each (weighted,
/// unweighted) pair as non-orthogonal; LOO on a 956 bake adjudicates
/// slot-worth, exactly the BANDVIS discipline.
pub mod idx_csfw {
    /// Luminance-weighted global mean shift:
    /// `sat(|Σw·s − Σw·d| / Σw, C_GDMEAN)` — the weighted twin of
    /// [`super::idx_append::GLOBAL_DMEAN`] (Y). V3's worst cross-route
    /// diverging lane family (SROCC 0.49–0.85,
    /// `benchmarks/hdr_streaming_gates_2026-07-27.md`): a mean shift's
    /// visibility depends on WHERE in the tonal range it lands, and the
    /// cbrt/PU encodings weight that differently — the route-common
    /// physical weight is the fix (design §5.1).
    pub const W_GLOBAL_DMEAN: usize = 0;
    /// Luminance-weighted global contrast gain:
    /// `bounded_excess(gvar₂ʷ, gvar₁ʷ, C_GCONTRAST)` on the
    /// weighted-population variances `gvarᵢʷ = Σw·xᵢ²/Σw − (Σw·xᵢ/Σw)²`
    /// — weighted twin of [`super::idx_append::GLOBAL_CGAIN`] (Y).
    pub const W_GLOBAL_CGAIN: usize = 1;
    /// Reverse polarity of [`W_GLOBAL_CGAIN`] — weighted twin of
    /// [`super::idx_append::GLOBAL_CLOSS`] (Y).
    pub const W_GLOBAL_CLOSS: usize = 2;
}

// ============================================================================
// REV4 candidate bank (2026-09-23 lane — `benchmarks/rev4_featbank_impl_
// 2026-09-23.md`, spec `docs/REV4_FEATURE_BANK_PLAN_2026-09-23.md` §1.1).
// Four append-only families at f986+; every slot is difference-form and
// every flag is independent (the emit tail is absolute-positioned, so any
// subset stays index-stable). Layouts mirror `feature_defs::BLOCKS`:
// gridblk/ringbasis/tailhist are PerChannel (`(scale*3 + ch)*per_cell +
// local`), arttype is PerScale (`scale*ARTTYPE_PER_SCALE + local`).
// ============================================================================

/// Registered base of the rev4 candidate bank (`feature_defs::BLOCKS`
/// derives the same number — asserted in `rev4_base_matches_registry`).
#[cfg_attr(not(test), allow(dead_code))] // the registered rev4 base — read by the parity/layout gates
pub(crate) const REV4_BASE: usize = 986;
#[cfg(test)]
pub(crate) const GMSBANK_BASE: usize = 1322;
pub(crate) const GMSBANK_PER_CELL: usize = 15;
/// mapdev slots per (scale, channel) cell (`feature_defs::MAPDEV_SIGNALS`).
pub(crate) const MAPDEV_PER_CELL: usize = 5;
/// z1max slots per (scale, channel) cell: 13 basic + 6 peaks.
pub(crate) const Z1MAX_PER_CELL: usize = 19;
/// gmsnative slots in total: X then B, 15 each, at scale 0.
pub(crate) const GMSNATIVE_WIDTH: usize = 30;

fn gmsbank_width(n_scales: usize) -> usize {
    crate::feature_defs::block_base(crate::feature_set_id::ComputeToken::Gmsbank, n_scales)
        .expect("registered C8")
        .1
        .width(n_scales)
}

/// gridblk slots per (scale, channel) cell: 6 signed log-magnitude bins
/// of the on-grid boundary step excess + on-grid mean + on/off ratio.
pub(crate) const GRIDBLK_PER_CELL: usize = 8;
/// ringbasis slots per (scale, channel) cell: 6 triangular
/// log-magnitude bins of the RINGING per-pixel term.
pub(crate) const RINGBASIS_PER_CELL: usize = 6;
/// tailhist slots per (scale, channel) cell: p95/p99/max for each of the
/// SSIM/ART/DET/MSE per-pixel maps (4 maps × 3).
pub(crate) const TAILHIST_PER_CELL: usize = 12;
/// arttype slots per scale (no channel axis — like append2/CSFW, each
/// signal names its channel): blur(Y), noise_x/y/b, bleed_x/b.
pub(crate) const ARTTYPE_PER_SCALE: usize = 6;
/// Tail histogram bin count per map (31 true-log interior edges
/// `10^(-6 + k*6/32)`, k = 1..31 → 32 partition cells).
pub(crate) const TAILHIST_BINS: usize = 32;
/// Flat-reference activity ceiling for the C4 noise descriptors — the
/// plan's `C4_FLAT_ACT`, pinned equal to [`C_ACTIVITY`] (design note
/// §C4): "flat" is the same saturator scale the masking/ringing
/// formulas already use.
pub(crate) const C4_FLAT_ACT: f64 = C_ACTIVITY;

/// Named local offsets within one scale's arttype block.
pub(crate) mod idx_arttype {
    /// Y-channel blur descriptor: `EDGE_WIDTH_CHANGE × HF_LOSS`, both
    /// already-emitted v2 slots at the Y cell.
    pub const BLUR: usize = 0;
    /// Flat-reference HF_GAIN mean on the X channel (`act ≤ C4_FLAT_ACT`).
    pub const NOISE_X: usize = 1;
    /// Flat-reference HF_GAIN mean on the Y channel.
    pub const NOISE_Y: usize = 2;
    /// Flat-reference HF_GAIN mean on the B channel.
    pub const NOISE_B: usize = 3;
    /// X-channel colour bleed: `bounded_excess(Σ_out g_dst, Σ_out g_src,
    /// C_GMS)` outside the ±1-px dilated dst-luma-edge mask.
    pub const BLEED_X: usize = 4;
    /// B-channel colour bleed (same form as [`BLEED_X`]).
    pub const BLEED_B: usize = 5;
}

/// Triangular-hat bin count shared by gridblk and ringbasis.
pub(crate) const REV4_HATS: usize = 6;
/// Largest lattice period any rev4 cell can take (chroma at scale 0).
pub(crate) const GRIDBLK_MAX_PERIOD: usize = 16;
/// C1 magnitude-hat centres, as the `u()` bit-domain coordinates of
/// TRAIN on-grid |ẽ| quantiles near 5/20/40/60/80/95% (2026-09-23
/// registry revision; design note §C1).
const GRIDBLK_LEVELS: [f64; REV4_HATS] = [1e-3, 4e-2, 1e-1, 2e-1, 5e-1, 1.0];
/// C2 magnitude-hat centres, as the `u()` coordinates of
/// `[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]` (design note §C2).
const RINGBASIS_LEVELS: [f64; REV4_HATS] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0];

/// The shared rev4 log-magnitude coordinate: the IEEE-754 bit pattern of
/// the positive value, read as an f64 (design note "shared log-magnitude
/// coordinate" — monotone in `v` for `v > 0`, transcendental-free,
/// tier/libc-invariant).
#[inline(always)]
fn rev4_u(v: f64) -> f64 {
    v.to_bits() as f64
}

/// `u(level)` per hat centre — const-computed through `to_bits`/`as`
/// (the same map `rev4_u` applies per pixel).
const fn u_centres(levels: [f64; REV4_HATS]) -> [f64; REV4_HATS] {
    let mut c = [0.0f64; REV4_HATS];
    let mut i = 0;
    while i < REV4_HATS {
        c[i] = levels[i].to_bits() as f64;
        i += 1;
    }
    c
}

/// C1's six u-domain hat centres.
const GRIDBLK_UC: [f64; REV4_HATS] = u_centres(GRIDBLK_LEVELS);
/// C2's six u-domain hat centres.
const RINGBASIS_UC: [f64; REV4_HATS] = u_centres(RINGBASIS_LEVELS);

/// Evaluate all six triangular memberships at `u` — `hat_memberships`
/// side="left" picks the adjacent pair, `1−t`/`t` weights with
/// `t = (u − c_j)/(c_{j+1} − c_j)`. Exactly two entries of the result
/// are nonzero.
fn rev4_hats(c: &[f64; REV4_HATS], u: f64) -> [f64; REV4_HATS] {
    let mut h = [0.0f64; REV4_HATS];
    let uc = u.clamp(c[0], c[REV4_HATS - 1]);
    let i = c.partition_point(|&x| x < uc);
    let j = i.saturating_sub(1).min(REV4_HATS - 2);
    let t = (uc - c[j]) / (c[j + 1] - c[j]);
    h[j] = 1.0 - t;
    h[j + 1] = t;
    h
}

/// Pinned IEEE-754 bit patterns for C3's 31 true-log interior edges
/// `10^(-6 + k*log10(2e6)/32)` (k = 1..31). The 2026-09-23 TRAIN-only
/// registry revision extended the top endpoint from 1 to 2; the final
/// interior edge is 1.2709334445868168. Runtime binning and quantile
/// emission use only this table, independent of platform `powf`.
const TAIL_EDGE_BITS: [u64; 31] = [
    0x3eba66c2a74a2306,
    0x3ec4c5f32ba53f33,
    0x3ed0584963ee7360,
    0x3ed9b8969445d847,
    0x3ee43ce83c83f32c,
    0x3eefd8eaa4b1df44,
    0x3ef90ee7884af44e,
    0x3f03b76562438fa3,
    0x3f0f06d135f4b230,
    0x3f186997e71fea72,
    0x3f2335535097fe64,
    0x3f2e3a21d1f34d40,
    0x3f37c88ad7e2850b,
    0x3f42b69b54e81970,
    0x3f4d72b8c0d9ca6c,
    0x3f572ba43fff3718,
    0x3f623b275257b423,
    0x3f6cb0733676d0d9,
    0x3f7692c8be49a365,
    0x3f81c2e1bdebc593,
    0x3f8bf32f4c29143a,
    0x3f95fddda6357e33,
    0x3fa14db59ac807b4,
    0x3fab3acbfaf4e35d,
    0x3fb56cc8fb2ef3d0,
    0x3fc0db8e76856372,
    0x3fca872915c0bda2,
    0x3fd4df716c11c450,
    0x3fe06c5865a085fb,
    0x3fe9d82743b7eddb,
    0x3ff455be4ebe49c1,
];

/// Binade LUT for the pinned C3 edges, used to bin in ~5 ops.
struct TailEdges {
    /// 31 ascending interior-edge bit patterns.
    edges: [u64; 31],
    /// `base[e]` = #edges strictly below binade `e`'s low end;
    /// `inner[e]` = #edges inside binade `e` (≤ 2: edge spacing
    /// 6/32 decade > binade width 0.301... i.e. at most `ceil(0.301/
    /// 0.1875) = 2` — asserted at build).
    base: Box<[u8; 1024]>,
    inner: Box<[u8; 1024]>,
}

impl TailEdges {
    fn build() -> Self {
        let edges = TAIL_EDGE_BITS;
        let mut base = Box::new([0u8; 1024]);
        let mut inner = Box::new([0u8; 1024]);
        for e in 0..=1023usize {
            let lo = (e as u64) << 52;
            let hi = ((e as u64) + 1) << 52;
            let b = edges.partition_point(|&x| x < lo);
            let n_in = edges[b..].partition_point(|&x| x < hi);
            debug_assert!(
                n_in <= 2,
                "edge spacing admits at most two edges per binade"
            );
            base[e] = b as u8;
            inner[e] = n_in as u8;
        }
        Self { edges, base, inner }
    }

    /// [`bin`](Self::bin) with the `c3negfold` correction: non-positive (and
    /// NaN) values fold into bin 0. The maps are non-negative by construction
    /// up to f32 rounding, which produces small negative `art`/`det` values
    /// (and possibly `-0.0`); the raw `to_bits` compare in `bin` sorts every
    /// sign-bit-set double above the top edge and silently saturates the top
    /// bin. Used at [`FormulaRevision::Rev4`] only (see [`TailAccum::fold`]).
    #[inline(always)]
    fn bin_folded(&self, v: f64) -> usize {
        let ub = v.to_bits();
        if (ub >> 63) != 0 || ub >= (0x7ff8u64 << 48) {
            return 0;
        }
        self.bin(v)
    }

    /// Bin index `#{edges strictly below v}` — the C3 cell convention
    /// (an exact edge hit lands in the cell below it).
    /// `v >= 0` is assumed: a sign-bit-set `v` lands in the TOP bin (the
    /// legacy behaviour every Rev1–Rev3 research extraction keeps, byte for
    /// byte); values above the last edge enter the top cell.
    #[inline(always)]
    fn bin(&self, v: f64) -> usize {
        let ub = v.to_bits();
        let e = ((ub >> 52) as usize).min(1023);
        let b = self.base[e] as usize;
        let inner = self.inner[e];
        let mut bin = b;
        if inner > 0 && ub > self.edges[b] {
            bin += 1;
            if inner > 1 && ub > self.edges[b + 1] {
                bin += 1;
            }
        }
        bin
    }
}

fn tail_edges() -> &'static TailEdges {
    use std::sync::OnceLock;
    static E: OnceLock<TailEdges> = OnceLock::new();
    E.get_or_init(TailEdges::build)
}

// ---------------------------------------------------------------------------
// Calibration diagnostics (registry-revision tooling, 2026-09-24).
// `ZENSIM_REV4_DIAG` set at process start enables per-cell fine-grained
// histogram dumps on stderr at finalize — used only for the C3 top-edge
// and C1 hat-centre calibration over TRAIN corpora. Default off; the
// `Option` check is one predictable branch per scatter/rescan pass.
// ---------------------------------------------------------------------------

/// 64 log bins per diagnostic histogram.
const DIAG_BINS: usize = 64;

fn rev4_diag_enabled() -> bool {
    use std::sync::OnceLock;
    static D: OnceLock<bool> = OnceLock::new();
    *D.get_or_init(|| std::env::var_os("ZENSIM_REV4_DIAG").is_some())
}

/// Diag bin for the C3 maps (range [0, 1]): 16 log bins per decade over
/// [1e-4, 1]. `v <= 0` folds into bin 0, `v >= 1` into the last bin.
/// Diagnostic-only — a `log10` per pixel is fine on the gated path.
fn diag_bin01(v: f64) -> usize {
    if v <= 0.0 || v.is_nan() {
        return 0;
    }
    ((v.log10() + 4.0) * 16.0)
        .floor()
        .clamp(0.0, (DIAG_BINS - 1) as f64) as usize
}

/// Diag bin for C1 `|ẽ|` (unbounded, observed ≪ 100): 8 log bins per
/// decade over [1e-5, 1e3]. `v <= 0` folds into bin 0.
fn diag_bin_abs(v: f64) -> usize {
    if v <= 0.0 || v.is_nan() {
        return 0;
    }
    ((v.log10() + 5.0) * 8.0)
        .floor()
        .clamp(0.0, (DIAG_BINS - 1) as f64) as usize
}

// ---------------------------------------------------------------------------
// REV4 per-(scale, channel) accumulator cell
//
// Merge-free by construction: the streaming and materialized walks both run
// the SAME kernel strips in the SAME order and write into the cell directly
// (the `blockiness_sparse_strip_wide` running-total pattern — "a per-strip
// partial would reassociate"), so serial, parallel-channel and
// streaming-vs-materialized paths all produce identical f64 op sequences.
// ---------------------------------------------------------------------------

/// C3: 32-bin log histograms + exact maxima of the four dense per-pixel
/// maps (`d`, `art`, `det`, `mse` — in emitted order). Integer counts are
/// merge- and order-invariant; the max is order-free.
#[derive(Clone)]
struct TailAccum {
    hist: [[u32; TAILHIST_BINS]; 4],
    max: [f64; 4],
    /// Env-gated fine histogram for the top-edge calibration dump
    /// (`ZENSIM_REV4_DIAG`); `None` in every production/test path.
    diag: Option<Box<[[u32; DIAG_BINS]; 4]>>,
    /// `c3negfold`: bin through [`TailEdges::bin_folded`]. Set from the
    /// computation's revision (`Rev4` only); Rev1–Rev3 keep the legacy
    /// [`TailEdges::bin`] so the landed `rev4bank` Bin slots they emit stay
    /// byte-identical. The process revision equals the computation's
    /// (`ssim_form::refuse_rev4_mix`).
    fold: bool,
}

impl Default for TailAccum {
    fn default() -> Self {
        Self {
            hist: [[0; TAILHIST_BINS]; 4],
            max: [0.0; 4],
            fold: crate::ssim_form::active_revision() >= FormulaRevision::Rev4,
            diag: rev4_diag_enabled().then(|| Box::new([[0; DIAG_BINS]; 4])),
        }
    }
}

impl TailAccum {
    /// One map value into its bin + running max.
    #[inline(always)]
    fn scatter(&mut self, edges: &TailEdges, v: f64, map: usize) {
        let b = if self.fold {
            edges.bin_folded(v)
        } else {
            edges.bin(v)
        };
        self.hist[map][b] += 1;
        self.max[map] = self.max[map].max(v);
        if let Some(d) = self.diag.as_deref_mut() {
            d[map][diag_bin01(v)] += 1;
        }
    }
}

/// C4-noise: flat-reference-region HF-gain mass for one cell.
/// featacc: `hfg` is a `SumVar` (f64-native terms) — `Seq` under production.
#[derive(Clone, Copy)]
struct FlatAccum {
    /// Σ `hf_gain_i` over pixels with `act ≤ C4_FLAT_ACT` (scan order).
    hfg: crate::featcanon::SumVar,
    /// Count of flat pixels.
    n: u64,
}

impl Default for FlatAccum {
    fn default() -> Self {
        Self {
            hfg: crate::featcanon::SumVar::f64_elems_meas(),
            n: 0,
        }
    }
}

/// C2: the 6 triangular `u`-domain bins of the per-pixel RINGING term —
/// `bins[k] = Σ_px ring·m_k(u(ring))` in scan order.
/// featacc: `SumVar` bins — `Seq` under production.
#[derive(Clone, Copy)]
struct RingAccum {
    bins: [crate::featcanon::SumVar; REV4_HATS],
}

impl Default for RingAccum {
    fn default() -> Self {
        Self {
            bins: std::array::from_fn(|_| crate::featcanon::SumVar::f64_elems_meas()),
        }
    }
}

/// C4-bleed: gradient magnitudes summed OUTSIDE the dilated dst-luma-edge
/// mask (scan order). featacc: `SumVar` — `Seq` under production.
#[derive(Clone, Copy)]
struct BleedAccum {
    out_src: crate::featcanon::SumVar,
    out_dst: crate::featcanon::SumVar,
}

impl Default for BleedAccum {
    fn default() -> Self {
        Self {
            out_src: crate::featcanon::SumVar::f64_elems_meas(),
            out_dst: crate::featcanon::SumVar::f64_elems_meas(),
        }
    }
}

/// C1: the stored `ẽ` boundary planes plus the per-phase profiles that
/// pick the winning phase at finalize (design note §C1 "Accumulation" —
/// the winning phase gates the emitted bins but is only known after the
/// whole plane is walked, so the kernel stores `ẽ` per boundary and the
/// rescan touches only on-grid positions: `2n/P` hat evaluations instead
/// of membership work on all `2n` boundaries).
///
/// `plane_v[y*w + x]` holds `ẽ` of the vertical boundary between columns
/// `x−1` and `x` (x ∈ 1..w); `plane_h[y*w + x]` holds the horizontal
/// boundary between rows `y−1` and `y` (y ∈ 1..h). Both are written by
/// `gridblk_strip_wide` in the SAME f32 arithmetic on every engine, so
/// the rescan reads bitwise-identical values in the streaming and
/// materialized walks. `abs_*`/`signed_*` are the per-phase `Σ|ẽ|`/`Σẽ`
/// profiles — f64 running totals fed by the kernel's lane-folded V8
/// accumulators (the lane map `(x−1, chunk parity) → phase` is fixed by
/// construction, so the accumulation order is identical on every tier,
/// engine and thread count — this family's determinism contract).
struct GridblkAccum {
    plane_v: Vec<f32>,
    plane_h: Vec<f32>,
    abs_v: [f64; GRIDBLK_MAX_PERIOD],
    signed_v: [f64; GRIDBLK_MAX_PERIOD],
    abs_h: [f64; GRIDBLK_MAX_PERIOD],
    signed_h: [f64; GRIDBLK_MAX_PERIOD],
    /// featacc `exact`-mode shadow planes: the same `ẽ` grid evaluated in
    /// f64. Empty in every other mode (and in product builds — two dead
    /// `Vec`s, 48 bytes of cell state). The rescan in
    /// [`finish_gridblk_cell`] reads whichever plane family is populated.
    plane_v64: Vec<f64>,
    plane_h64: Vec<f64>,
}

impl Default for GridblkAccum {
    fn default() -> Self {
        Self {
            plane_v: Vec::new(),
            plane_h: Vec::new(),
            abs_v: [0.0; GRIDBLK_MAX_PERIOD],
            signed_v: [0.0; GRIDBLK_MAX_PERIOD],
            abs_h: [0.0; GRIDBLK_MAX_PERIOD],
            signed_h: [0.0; GRIDBLK_MAX_PERIOD],
            plane_v64: Vec::new(),
            plane_h64: Vec::new(),
        }
    }
}

/// All rev4 state for one (scale, channel) cell. Untouched when every
/// rev4 compute flag is off.
#[derive(Default)]
struct Rev4CellAccum {
    grid: GridblkAccum,
    ring: RingAccum,
    tail: TailAccum,
    flat: FlatAccum,
    bleed: BleedAccum,
    bank: [GmsBankCell; 5],
    chroma: [GmsBankCell; 5],
}

/// REV4 hook bundle for the dense kernel — carries C3 (`tail`) and C4's
/// noise descriptor (`flat`) for ONE cell. `None` on every non-rev4 call:
/// inside the kernel the Option is one loop-invariant branch (the same
/// shape `transducer_bank` already carries here).
struct Rev4Dense<'a> {
    /// C3 target when `tailhist` is on.
    tail: Option<&'a mut TailAccum>,
    /// C4-noise target when `arttype` is on.
    flat: Option<&'a mut FlatAccum>,
    /// Resolved once by the caller (a `OnceLock` hit per strip, not per
    /// pixel).
    edges: &'static TailEdges,
}

/// One pixel's contribution to the C3/C4 dense targets — the single code
/// path shared by the era-1 SIMD chunk lanes, the era-1 scalar tail, and
/// the era-2 V8 lanes so the three cannot drift.
#[inline(always)]
fn rev4_dense_pixel(
    r4: &mut Rev4Dense<'_>,
    d: f64,
    art: f64,
    det: f64,
    mse: f64,
    hfg: f64,
    act: f64,
    lane: usize,
) {
    if let Some(t) = r4.tail.as_deref_mut() {
        t.scatter(r4.edges, d, 0);
        t.scatter(r4.edges, art, 1);
        t.scatter(r4.edges, det, 2);
        t.scatter(r4.edges, mse, 3);
    }
    if let Some(f) = r4.flat.as_deref_mut()
        && act <= C4_FLAT_ACT
    {
        f.hfg.add64(lane, hfg);
        f.n += 1;
    }
}

/// One chunk's (8 lanes) worth of [`rev4_dense_pixel`] calls — pulled out
/// of the SIMD chunk loop so the per-element histogram/pool scatter stays
/// scalar on every tier. On AVX-512 LLVM otherwise fuses the lane loop's
/// `pool[lane]` and `hist[bin]` indexed updates into
/// `vgatherqpd`/`vscatterqpd` (4.3x slower than the scalar form,
/// measured 2026-10-03); `black_box` on the trip count makes the loop
/// unvectorizable, and `inline(never)` keeps the mangled shape from being
/// re-created inside the caller. Bit-exact: the calls are identical.
#[inline(never)]
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
fn rev4_dense_chunk8(
    r4: &mut Rev4Dense<'_>,
    d: &[f32; 8],
    art: &[f32; 8],
    det: &[f32; 8],
    mse: &[f32; 8],
    hfg: &[f32; 8],
    act: &[f32; 8],
) {
    for l in 0..std::hint::black_box(8usize) {
        rev4_dense_pixel(
            r4,
            d[l] as f64,
            art[l] as f64,
            det[l] as f64,
            mse[l] as f64,
            hfg[l] as f64,
            act[l] as f64,
            l,
        );
    }
}

/// REV4 hook bundle for the gradient kernel — C2's ring bins on every
/// channel and C4's bleed sums on the chroma channels (`mask` present).
struct Rev4Grad<'a> {
    /// C2 target when `ringbasis` is on.
    ring: Option<&'a mut RingAccum>,
    /// C4 bleed: `(outside-weight mask slice, target)` — `None` on the Y
    /// channel (and whenever `arttype` is off).
    bleed: Option<(&'a [f32], &'a mut BleedAccum)>,
    bank: GmsBankWork<'a>,
}

/// One pixel's contribution to the C2/C4 gradient targets — shared by the
/// scalar border path and the SIMD chunk lanes. `i` is the pixel's
/// strip-local index (`y*width + x`) — the C4 outside-mask weight is
/// `1 − mask[i]`, read inside so the slice borrow never leaves `r4`.
#[inline(always)]
fn rev4_grad_pixel(
    r4: &mut Rev4Grad<'_>,
    centres: &[f64; REV4_HATS],
    ring: f64,
    g_src: f64,
    g_dst: f64,
    i: usize,
    lane: usize,
) {
    if let Some(bins) = r4.ring.as_deref_mut()
        && ring > 0.0
    {
        let h = rev4_hats(centres, rev4_u(ring));
        for (b, h) in bins.bins.iter_mut().zip(h.iter()) {
            b.add64(lane, ring * h);
        }
    }
    if let Some((m, b)) = r4.bleed.as_mut() {
        let out_w = 1.0 - m[i] as f64;
        b.out_src.add64(lane, g_src * out_w);
        b.out_dst.add64(lane, g_dst * out_w);
    }
}

/// One (strip, channel)'s rev4 work order — the at-scale [`ComputeSet`]
/// flags resolved against the channel, handed to [`stream_phase_b`] (and
/// the materialized [`compute_channel_scale_v2`]). `gridblk_period == 0`
/// is the C1-off marker (a period can never legitimately be 0), and
/// `bleed_mask` is `Some` only on the two chroma channels.
#[derive(Clone, Copy)]
struct Rev4Work<'a> {
    /// C1 lattice period for this (channel, scale) — `8 >> s` on Y,
    /// `16 >> s` on X/B — or 0 when `gridblk` is off or degenerate (`< 2`).
    gridblk_period: usize,
    /// C2 ring bins run in the gradient kernel.
    ringbasis: bool,
    /// C3 tail histograms run in the dense kernel.
    tailhist: bool,
    /// C4 flat-region HF-gain noise runs in the dense kernel.
    flat: bool,
    /// C4 dilated dst-luma-edge mask (strip_n bytes of 0.0/1.0), `Some`
    /// only when `arttype` is on AND this is a chroma channel.
    bleed_mask: Option<&'a [f32]>,
    /// C8 gradient-similarity bank runs in the same gradient pass.
    gmsbank: bool,
    bank: GmsBankWork<'a>,
}

/// Does the C8 gradient bank run for this (scale, channel)? C8 proper: every
/// coarse channel plus native Y. `gmsnative` (restored cut) adds the native X
/// and B cells; they reuse the same per-channel stabilisers and accumulators.
fn gmsbank_cell_live(local: &ComputeSet, scale: usize, ch: usize) -> bool {
    (local.gmsbank && (scale > 0 || ch == 1)) || (local.gmsnative && scale == 0 && ch != 1)
}

/// Materialized-path REV4 work order for one (scale, channel) — the
/// [`ComputeSet::rev4_work`] analogue for [`compute_channel_scale_v2`],
/// which resolves flags itself and owns the strip loop the bleed mask is
/// built inside. `bleed_y` is the same-scale dst LUMA plane (full
/// `width*height`), `Some` only on the chroma channels when `arttype` is
/// on; the mask is gathered + built per strip from it.
struct Rev4CellArgs<'a> {
    /// The (scale, channel)'s accumulator — every enabled hook writes here.
    cell: &'a mut Rev4CellAccum,
    /// Same semantics as [`Rev4Work::gridblk_period`].
    gridblk_period: usize,
    /// Same semantics as [`Rev4Work::ringbasis`].
    ringbasis: bool,
    /// Same semantics as [`Rev4Work::tailhist`].
    tailhist: bool,
    /// Same semantics as [`Rev4Work::flat`].
    flat: bool,
    /// Same-scale dst-Y plane for the C4 mask — see the struct doc.
    bleed_y: Option<&'a [f32]>,
    gmsbank: bool,
    bank: GmsBankWork<'a>,
}

/// Resolve one (scale, channel)'s [`Rev4CellArgs`] from the at-scale
/// [`ComputeSet`] — the [`ComputeSet::rev4_work`] flag logic verbatim,
/// `bleed_mask` replaced by `bleed_y` (the materialized strip loop builds
/// the mask itself). Single owner of "which hooks this cell needs" on the
/// materialized side.
fn rev4_cell_args<'a>(
    cell: &'a mut Rev4CellAccum,
    local: &ComputeSet,
    scale: usize,
    ch: usize,
    src_planes: &'a [Vec<f32>],
    dst_planes: &'a [Vec<f32>],
) -> Rev4CellArgs<'a> {
    let period = (if ch == 1 {
        BLOCK_LATTICE
    } else {
        2 * BLOCK_LATTICE
    }) >> scale;
    Rev4CellArgs {
        cell,
        gridblk_period: if local.gridblk && period >= 2 {
            period
        } else {
            0
        },
        ringbasis: local.ringbasis,
        tailhist: local.tailhist,
        flat: local.arttype,
        bleed_y: (local.arttype && ch != 1).then(|| &dst_planes[1][..]),
        gmsbank: gmsbank_cell_live(local, scale, ch),
        bank: GmsBankWork::new(
            ch,
            (local.gmsbank && scale > 0 && ch == 0).then(|| ChromaStrip {
                reference: &src_planes[2],
                distorted: &dst_planes[2],
            }),
        ),
    }
}

// ---------------------------------------------------------------------------
// REV4 emission: one PerChannel cell = gridblk(8) + ringbasis(6) +
// tailhist(12) = 26 slots in that order; arttype is emitted per scale by
// `finish_arttype_scale` (its blur term needs the cross-scale EWC value,
// which only exists at finalize).
// ---------------------------------------------------------------------------

/// `#{x ∈ [1, n) : x mod P == φ}` — the per-phase boundary count of one
/// row (V) or column (H) of the lattice. Closed form: the positions are
/// the lattice, not the data.
fn gridblk_phase_count(n: usize, phi: usize, period: usize) -> usize {
    debug_assert!(phi < period);
    if phi == 0 {
        (n - 1) / period
    } else if phi < n {
        (n - 1 - phi) / period + 1
    } else {
        0
    }
}

/// Strict-`>` argmax over the per-phase `Σ|ẽ|` profile (ties resolve to
/// the lowest phase — `Rev4` spec §C1). Returns `None` when no phase has
/// a boundary (degenerate plane).
fn gridblk_winner(
    abs: &[f64; GRIDBLK_MAX_PERIOD],
    cnt: &[usize; GRIDBLK_MAX_PERIOD],
    period: usize,
) -> Option<usize> {
    let mut best: Option<usize> = None;
    let mut best_sum = 0.0f64;
    for p in 0..period {
        if cnt[p] != 0 && abs[p] > best_sum {
            best_sum = abs[p];
            best = Some(p);
        }
    }
    best
}

/// Emit the 8 `gridblk` slots for one cell. `period < 2` (Y at scale 3)
/// or a degenerate side emits zeros — the cell is still fully registered.
/// The magnitude bins rescan the stored `ẽ` planes at the winning phase's
/// positions only (V: `x ≡ pv`, H: `y ≡ ph`, both in y-major scan order —
/// the fixed rescan order every engine replays).
fn finish_gridblk_cell(
    grid: &GridblkAccum,
    period: usize,
    width: usize,
    height: usize,
    out: &mut [f64; GRIDBLK_PER_CELL],
) -> Option<[u32; DIAG_BINS]> {
    *out = [0.0; GRIDBLK_PER_CELL];
    // featacc `exact` mode fills the f64 shadow planes instead of the f32
    // ones — the rescan reads whichever family is populated.
    let exact64 = !grid.plane_v64.is_empty();
    if period < 2 || (!exact64 && grid.plane_v.is_empty()) {
        return None;
    }
    if exact64 {
        debug_assert_eq!(grid.plane_v64.len(), width * height);
        debug_assert_eq!(grid.plane_h64.len(), width * height);
    } else {
        debug_assert_eq!(grid.plane_v.len(), width * height);
        debug_assert_eq!(grid.plane_h.len(), width * height);
    }
    let mut cnt_v = [0usize; GRIDBLK_MAX_PERIOD];
    let mut cnt_h = [0usize; GRIDBLK_MAX_PERIOD];
    for p in 0..period {
        cnt_v[p] = gridblk_phase_count(width, p, period) * height;
        cnt_h[p] = gridblk_phase_count(height, p, period) * width;
    }
    let (Some(pv), Some(ph)) = (
        gridblk_winner(&grid.abs_v, &cnt_v, period),
        gridblk_winner(&grid.abs_h, &cnt_h, period),
    ) else {
        return None;
    };
    // On-grid = the winning phase of each orientation, V and H pooled.
    let n_on = (cnt_v[pv] + cnt_h[ph]) as f64;
    let (mut off_abs, mut n_off) = (0.0f64, 0.0f64);
    for p in 0..period {
        if p != pv {
            off_abs += grid.abs_v[p];
            n_off += cnt_v[p] as f64;
        }
        if p != ph {
            off_abs += grid.abs_h[p];
            n_off += cnt_h[p] as f64;
        }
    }
    if n_off == 0.0 {
        return None;
    }
    // On-grid rescan: 6 triangular-hat memberships of `u(|ẽ|)`, weighted
    // by the SIGNED `ẽ` — ~2n/P evaluations, the store-then-rescan shape
    // the design note's C1 "Accumulation" paragraph commits.
    let mut bins = [0.0f64; REV4_HATS];
    let mut diag = rev4_diag_enabled().then_some([0u32; DIAG_BINS]);
    let xv0 = if pv == 0 { period } else { pv };
    for y in 0..height {
        let row = y * width;
        let mut x = xv0;
        while x < width {
            let v = if exact64 {
                grid.plane_v64[row + x]
            } else {
                grid.plane_v[row + x] as f64
            };
            if v != 0.0 {
                let a = v.abs();
                let h = rev4_hats(&GRIDBLK_UC, rev4_u(a));
                for k in 0..REV4_HATS {
                    bins[k] += v * h[k];
                }
                if let Some(d) = diag.as_mut() {
                    d[diag_bin_abs(a)] += 1;
                }
            }
            x += period;
        }
    }
    let yh0 = if ph == 0 { period } else { ph };
    for y in (yh0..height).step_by(period) {
        let row = y * width;
        for x in 0..width {
            let v = if exact64 {
                grid.plane_h64[row + x]
            } else {
                grid.plane_h[row + x] as f64
            };
            if v != 0.0 {
                let a = v.abs();
                let h = rev4_hats(&GRIDBLK_UC, rev4_u(a));
                for k in 0..REV4_HATS {
                    bins[k] += v * h[k];
                }
                if let Some(d) = diag.as_mut() {
                    d[diag_bin_abs(a)] += 1;
                }
            }
        }
    }
    for k in 0..REV4_HATS {
        out[k] = bins[k] / n_on;
    }
    // `on_mean` is the SIGNED mean (design note §C1 — a suppression reads
    // negative); `onoff_ratio` is the abs-mean contrast with the C_BLOCK
    // denominator guard, not a bare ratio.
    out[6] = (grid.signed_v[pv] + grid.signed_h[ph]) / n_on;
    out[7] = (grid.abs_v[pv] + grid.abs_h[ph]) / n_on / (off_abs / n_off + C_BLOCK);
    diag
}

/// Emit the 6 `ringbasis` slots for one cell: `bins[k]/n_px`.
fn finish_ringbasis_cell(ring: &RingAccum, n_px: usize, out: &mut [f64; RINGBASIS_PER_CELL]) {
    *out = [0.0; RINGBASIS_PER_CELL];
    if n_px == 0 {
        return;
    }
    let inv = 1.0 / n_px as f64;
    for (o, b) in out.iter_mut().zip(ring.bins.iter()) {
        *o = b.fin() * inv;
    }
}

/// Emit the 12 `tailhist` slots for one cell: p95/p99/max over SSIM-d,
/// ART, DET, MSE in that order. `p_q` is the LOWER EDGE of the first bin
/// whose cumulative count reaches `ceil(q·n)`; bin 0 emits 0.0.
fn finish_tailhist_cell(tail: &TailAccum, n_px: usize, out: &mut [f64; TAILHIST_PER_CELL]) {
    *out = [0.0; TAILHIST_PER_CELL];
    if n_px == 0 {
        return;
    }
    let edges = tail_edges();
    for m in 0..4 {
        let h = &tail.hist[m];
        // Every scattered value lands in exactly one bin, so the quantile walk
        // below always reaches its target (the fallthrough is unreachable).
        debug_assert_eq!(h.iter().map(|&c| c as usize).sum::<usize>(), n_px);
        let p95 = tail_quantile_edge(h, n_px, 0.95, edges);
        let p99 = tail_quantile_edge(h, n_px, 0.99, edges);
        out[m * 3] = p95;
        out[m * 3 + 1] = p99;
        out[m * 3 + 2] = tail.max[m];
    }
}

/// Lower edge (as the true-log value, not bits) of the first bin whose
/// cumulative count reaches `ceil(q·n)`. Bin 0's lower edge is 0.0 (the
/// exact-identity cell). Only called with `n > 0`.
fn tail_quantile_edge(h: &[u32; TAILHIST_BINS], n: usize, q: f64, edges: &TailEdges) -> f64 {
    let target = (q * n as f64).ceil() as u64;
    let mut cum = 0u64;
    for (i, &c) in h.iter().enumerate() {
        cum += c as u64;
        if cum >= target {
            return if i == 0 {
                0.0
            } else {
                f64::from_bits(edges.edges[i - 1])
            };
        }
    }
    // Unreachable given target <= n, but keep the function total.
    f64::from_bits(edges.edges[TAILHIST_BINS - 2])
}

/// Emit the 6 `arttype` slots for one scale. `ewc_y`/`hfl_y` are the
/// already-finished Y EDGE_WIDTH_CHANGE and HF_LOSS values for this scale
/// (the blur term multiplies finished feature values — design note §C4).
/// `flat`/`bleed` are the three channels' accumulator cells in the walk's
/// `accums` order (`[X=0, Y=1, B=2]`).
fn finish_arttype_scale(
    ewc_y: f64,
    hfl_y: f64,
    flat: &[FlatAccum; 3],
    bleed: &[BleedAccum; 3],
    out: &mut [f64; ARTTYPE_PER_SCALE],
) {
    out[idx_arttype::BLUR] = ewc_y * hfl_y;
    // `flat`/`bleed` are indexed by CHANNEL (0 = X, 1 = Y, 2 = B — the
    // walk's `accums` order); the registry's NOISE_* order is X, Y, B.
    out[idx_arttype::NOISE_X] = if flat[0].n == 0 {
        0.0
    } else {
        flat[0].hfg.fin() / flat[0].n as f64
    };
    out[idx_arttype::NOISE_Y] = if flat[1].n == 0 {
        0.0
    } else {
        flat[1].hfg.fin() / flat[1].n as f64
    };
    out[idx_arttype::NOISE_B] = if flat[2].n == 0 {
        0.0
    } else {
        flat[2].hfg.fin() / flat[2].n as f64
    };
    out[idx_arttype::BLEED_X] =
        bounded_excess(bleed[0].out_dst.fin(), bleed[0].out_src.fin(), C_GMS);
    out[idx_arttype::BLEED_B] =
        bounded_excess(bleed[2].out_dst.fin(), bleed[2].out_src.fin(), C_GMS);
}

/// Emit all four rev4 families for one scale into their feature slices.
/// `cells` are the channels' `Rev4CellAccum`s in channel order (0 = X,
/// 1 = Y, 2 = B — the walk's `accums` order); `out_*`
/// are the family slices for this scale (`None` when the family is off —
/// emission is skipped, the slots still exist in the vector as zeros
/// only when the layout declared them).
#[allow(clippy::too_many_arguments)]
fn finish_rev4_scale(
    cs: &ComputeSet,
    scale: usize,
    cells: [&Rev4CellAccum; 3],
    width: usize,
    height: usize,
    ewc_y: f64,
    hfl_y: f64,
    out_grid: Option<&mut [f64]>,
    out_ring: Option<&mut [f64]>,
    out_tail: Option<&mut [f64]>,
    out_art: Option<&mut [f64]>,
) {
    let n_px = width * height;
    let period_y = BLOCK_LATTICE >> scale;
    let period_c = (BLOCK_LATTICE * 2) >> scale;
    if let Some(o) = out_grid {
        debug_assert_eq!(o.len(), 3 * GRIDBLK_PER_CELL);
        if cs.gridblk {
            for ch in 0..3 {
                // `cells`/`o` are channel-ordered [X, Y, B]: only Y is luma.
                let period = if ch == 1 { period_y } else { period_c };
                let diag = finish_gridblk_cell(
                    &cells[ch].grid,
                    period,
                    width,
                    height,
                    (&mut o[ch * GRIDBLK_PER_CELL..(ch + 1) * GRIDBLK_PER_CELL])
                        .try_into()
                        .unwrap(),
                );
                if let Some(d) = diag {
                    eprintln!(
                        "REV4DIAGABS s{scale} c{ch} {}",
                        d.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
                    );
                }
            }
        }
    }
    if let Some(o) = out_ring {
        debug_assert_eq!(o.len(), 3 * RINGBASIS_PER_CELL);
        if cs.ringbasis {
            for ch in 0..3 {
                finish_ringbasis_cell(
                    &cells[ch].ring,
                    n_px,
                    (&mut o[ch * RINGBASIS_PER_CELL..(ch + 1) * RINGBASIS_PER_CELL])
                        .try_into()
                        .unwrap(),
                );
            }
        }
    }
    if let Some(o) = out_tail {
        debug_assert_eq!(o.len(), 3 * TAILHIST_PER_CELL);
        if cs.tailhist {
            for ch in 0..3 {
                finish_tailhist_cell(
                    &cells[ch].tail,
                    n_px,
                    (&mut o[ch * TAILHIST_PER_CELL..(ch + 1) * TAILHIST_PER_CELL])
                        .try_into()
                        .unwrap(),
                );
                if let Some(d) = cells[ch].tail.diag.as_deref() {
                    for (m, h) in d.iter().enumerate() {
                        eprintln!(
                            "REV4DIAGTAIL s{scale} c{ch} m{m} {}",
                            h.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
                        );
                    }
                }
            }
        }
    }
    if let Some(o) = out_art {
        debug_assert_eq!(o.len(), ARTTYPE_PER_SCALE);
        if cs.arttype {
            finish_arttype_scale(
                ewc_y,
                hfl_y,
                &[cells[0].flat, cells[1].flat, cells[2].flat],
                &[cells[0].bleed, cells[1].bleed, cells[2].bleed],
                o.try_into().unwrap(),
            );
        }
    }
}

/// Box blur radius at scale 0 — matches v1's `ZensimConfig::default()`.
pub(crate) const BLUR_RADIUS: usize = 5;
/// Oriented-blockiness lattice period (JPEG's 8x8 MCU grid).
pub(crate) const BLOCK_LATTICE: usize = 8;

// ============================================================================
// Phase-5 (§A.15): strip-tiled pipeline constants + halo-boundary helper.
//
// Bandwidth hypothesis (coordinator, from phase-4's own ratio-vs-size curve
// 1.18x@256^2 -> 3.32x@2048^2 -- growing, not flat, meaning DRAM traffic
// from materializing full-image mu1/mu2/ssq/s12/activity planes, not
// division-bound compute, is the dominant slope term at large sizes):
// process the image in row STRIPS small enough that one strip's live
// planes stay cache-resident, fusing blur -> formula pass -> gradient
// pass -> accumulation per strip so full-image intermediate planes are
// NEVER materialized (`ScratchV2Strip` below is sized per-strip, not
// per-image).
// ============================================================================

/// Strip height in rows. Tunable; swept per §A.15's strip-height table
/// (candidates measured: see the spec doc). Not a public knob — phase-5's
/// scope is proving/falsifying the bandwidth hypothesis, not exposing a
/// user-facing tuning parameter.
pub(crate) const STRIP_ROWS: usize = 128;

/// Half-halo (rows of REAL context loaded beyond the strip on each side).
/// Sized so the existing (UNMODIFIED) `box_blur_v_from_copy`'s own
/// reflect-boundary logic never touches the sub-range this module
/// actually reads back out -- derivation (§A.15): computing mu1 (etc.) at
/// buffer-local row `r` needs H-blurred input at `[r-R, r+R]`; the
/// widest-reaching consumer is `activity`, which needs mu1 at strip-local
/// `[-R, strip_h+R)` (one blur-radius beyond the strip itself), so mu1's
/// OWN computation needs INPUT (mu1_h) at `[-2R, strip_h+2R)`. Padding the
/// wide buffer with `HALO_P = 2*BLUR_RADIUS` real rows on each side keeps
/// every window inside `[0, strip_h + 2*HALO_P)` without ever touching
/// the wide-buffer's own synthetic edge (which would apply the WRONG
/// reflection -- the buffer isn't the true image).
pub(crate) const HALO_P: usize = 2 * BLUR_RADIUS;

/// The COARSEST row-group size any `fused_blur_h_ssim_*` kernel walks in.
///
/// The kernels transpose a group of rows into lanes and run one group at a
/// time plus a partial tail: **16 rows on `v4` and `v4x`**
/// (`blur.rs` `let row_groups = height / 16`), **8 on `v3`, `neon`, `wasm128`
/// and `scalar`** (`run_group(rg * 8, 8)`). Band boundaries in
/// [`fused_blur_h_ssim_banded`] are multiples of this coarsest size, which is
/// a multiple of the other, so on EVERY tier a band's grouping is a
/// sub-sequence of the whole-plane call's: same group size, same offset within
/// it, and only the final band can hold the partial group the whole-plane
/// call's tail would have run.
///
/// (`phase_a_blur_bands_are_bit_exact` additionally MEASURES agreement at band
/// sizes that are not multiples of either — the per-row independence
/// `box_blur_h_ring_matches_regathered_reference` pins is stronger than the
/// alignment argument. The alignment is what makes the claim hold without
/// depending on that, on tiers this box cannot run.)
pub(crate) const H_BLUR_ROW_GROUP: usize = 16;

/// Rows per phase-A H-blur band. A multiple of [`H_BLUR_ROW_GROUP`] — the
/// bit-exactness precondition — and the smallest such value, which is what
/// measurement wanted: at 16 rows a band's four output planes fit L2, and
/// phase-A BUSY time FELL 90.2 → 46.2 ms at 2304²/16T when the split landed,
/// so the bands are cheaper in total work as well as wider.
/// `benchmarks/fold_mt_scaling_2026-08-31.md`.
pub(crate) const H_BLUR_BAND_ROWS: usize = 16;
const _: () = assert!(H_BLUR_BAND_ROWS.is_multiple_of(H_BLUR_ROW_GROUP));

/// Mirror-without-repeating-edge boundary reflection ("reflect_101" /
/// whole-sample-symmetric convention: `-1 -> 1`, `-2 -> 2`, `height ->
/// height-2`, matching the mirror convention `crate::blur`'s own
/// boundary-handling comments describe, e.g. `fused_blur_h_ssim_inner_v4`'s
/// "reflect the index back into the image (`2*(height-1) - add_raw`)").
/// Used ONLY to gather a strip's halo rows from the FULL (already fully
/// materialized -- src/dst planes are the caller's whole-channel-scale
/// input, unchanged from phases 1-4) src/dst planes; every row this
/// pulls in is REAL image data, either directly (interior strips) or via
/// this exact mirror (strips touching the true top/bottom edge) — never
/// synthetic/zero-padding.
#[inline]
pub(crate) fn reflect_101(y: isize, height: usize) -> usize {
    if height <= 1 {
        return 0;
    }
    let h = height as isize;
    let period = 2 * (h - 1);
    let mut m = y.rem_euclid(period);
    if m >= h {
        m = period - m;
    }
    m as usize
}

/// Populate `dst` (`width * h_local` rows) from the FULL `src_full` plane
/// (`width * height_full`), where local row `i` maps to global row
/// `reflect_101(y0 as isize - halo as isize + i as isize, height_full)`.
/// For `i` such that the mapped global row is in-bounds (the common case
/// for interior strips and the strip's own core rows), this is an exact,
/// unreflected copy — `reflect_101` is the identity on `[0, height_full)`.
pub(crate) fn gather_strip_halo(
    src_full: &[f32],
    width: usize,
    height_full: usize,
    y0: usize,
    h_local: usize,
    halo: usize,
    dst: &mut [f32],
) {
    debug_assert_eq!(dst.len(), width * h_local);
    for i in 0..h_local {
        let gy = y0 as isize - halo as isize + i as isize;
        let gy_r = reflect_101(gy, height_full);
        let src_row = &src_full[gy_r * width..(gy_r + 1) * width];
        dst[i * width..(i + 1) * width].copy_from_slice(src_row);
    }
}

// ============================================================================
// Bounded per-pixel formulas
// ============================================================================

/// Per-pixel SSIM-like dissimilarity, C1-bounded (D1 fix). Bounded [0, 2].
///
/// Phase-6 (§A.16 lever A step 1): the original form computes
/// `num_m * (num_s/denom_s)` = `(A/B) * (C/D)` — TWO divisions. Since
/// `(A/B)*(C/D) == (A*C)/(B*D)` exactly (ordinary fraction algebra, no
/// approximation), this collapses to ONE division at the cost of two extra
/// multiplications — a strict win since a multiply is ~3-5x cheaper than a
/// divide on every target ISA this crate SIMD-dispatches to. Kept for the
/// scalar tail loop; see `ssim_d_local_v` for the SIMD sibling, which
/// additionally routes the single division through `.recip()` (step 2).
///
/// `direct` (revision 3, issue #61): the fourth plane is the DIRECT error
/// moment `E[(a-b)²]` rather than `E[ab]` — the H pass accumulates it in the
/// `sigma12` plane's place — so the covariance subtraction, which cancels
/// catastrophically on flat content, is replaced through the identity
/// `2cov = var_sum - err_var`. Then `1 - (a/b)(c/d)` becomes
/// `(d(b-a) + a·err_var) / (b·d)`: still one division, and the same
/// bounded-error contract as the v1 signal (`ssim_form::SsimSplats16::direct`).
#[inline]
fn ssim_d_local(mu1: f64, mu2: f64, s12: f64, ssq: f64, direct: bool, stable: bool) -> f64 {
    let a = 2.0 * mu1 * mu2 + C1_V2;
    let b = mu1 * mu1 + mu2 * mu2 + C1_V2;
    let d = ssq - mu1 * mu1 - mu2 * mu2 + C2_V2;
    if direct {
        let mu_diff = mu1 - mu2;
        let err_var = (s12 - mu_diff * mu_diff).max(0.0);
        // Revision 7: `(m1 − m2)²`, the exact value of `b − a`, which
        // cancels when the means are close (the old form keeps its bits).
        let lum = if stable { mu_diff * mu_diff } else { b - a };
        ((d * lum + a * err_var) / (b * d)).max(0.0)
    } else {
        let cov = s12 - mu1 * mu2;
        let c = 2.0 * cov + C2_V2;
        let local = (a * c) / (b * d);
        (1.0 - local).max(0.0)
    }
}

/// GMSD/FSIM/DISTS canonical bounded-similarity form: `(2ab+c)/(a²+b²+c)`.
/// Bounded `(0, 1]`, `=1` iff `a == b`. Requires `a, b >= 0`.
#[inline]
pub(crate) fn bounded_sim(a: f64, b: f64, c: f64) -> f64 {
    (2.0 * a * b + c) / (a * a + b * b + c)
}

/// `1 − bounded_sim(a, b, c)`, the bounded dissimilarity `(a−b)²/(a²+b²+c)`.
/// Formed as `1 − ratio` it cancels when `a ≈ b` (the ratio rounds near 1);
/// with `stable` (revision 7, [`FormulaRevision::stable_forms`]) it forms the
/// exact quotient instead.
#[inline]
pub(crate) fn bounded_dissim(a: f64, b: f64, c: f64, stable: bool) -> f64 {
    if stable {
        let d = a - b;
        d * d / (a * a + b * b + c)
    } else {
        1.0 - bounded_sim(a, b, c)
    }
}

/// [`bounded_dissim`] in f32, op for op `1 − (2ab + c)/(a² + b² + c)`, or the
/// exact quotient with `stable`.
#[inline(always)]
pub(crate) fn bounded_dissim32(a: f32, b: f32, c: f32, stable: bool) -> f32 {
    if stable {
        let d = a - b;
        d * d / (a * a + b * b + c)
    } else {
        1.0 - (2.0 * a * b + c) / (a * a + b * b + c)
    }
}

/// Bounded, SIGNED normalized-difference form: `max(0, a-b) / (a+b+c)`.
/// Bounded `[0, 1)`. Requires `a, b >= 0`.
#[inline]
pub(crate) fn bounded_excess(a: f64, b: f64, c: f64) -> f64 {
    (a - b).max(0.0) / (a + b + c)
}

/// [`bounded_excess`]`(a,b,c)` PAIRED with [`bounded_excess`]`(b,a,c)`, one
/// shared division instead of two (phase-6 §A.16 lever A step 1: the two
/// calls' denominators — `a+b+c` and `b+a+c` — are the SAME value by
/// commutativity of addition, computed once here). Returns
/// `(bounded_excess(a,b,c), bounded_excess(b,a,c))`. Used for the
/// hf_gain/hf_loss pair, which is exactly this shape (`bounded_excess
/// (dst_sq,src_sq,c)` / `bounded_excess(src_sq,dst_sq,c)`).
#[inline]
fn bounded_excess_pair(a: f64, b: f64, c: f64) -> (f64, f64) {
    let recip_denom = 1.0 / (a + b + c);
    (
        (a - b).max(0.0) * recip_denom,
        (b - a).max(0.0) * recip_denom,
    )
}

/// Michelson-contrast-style saturating ratio `x/(x+c)`, bounded `[0, 1)`
/// for `x >= 0`.
#[inline]
fn saturate(x: f64, c: f64) -> f64 {
    let x = x.max(0.0);
    x / (x + c)
}

/// Fused PJND masking-transducer band: `saturate(raw_abs_err/(1+k*act), c)`
/// collapsed to ONE division (phase-6 §A.16 lever A step 1, exact — not an
/// approximation). Derivation: let `D = 1+k*act` (always `>= 1 > 0` since
/// `k, act >= 0`); `saturate(x/D, c) = (x/D)/((x/D)+c)`; multiplying
/// numerator and denominator by `D` (valid, `D` is a positive nonzero
/// scalar) gives `x/(x + c*D) = x/(x + c + c*k*act)` — the original chained
/// `raw_abs_err/(1+k*act)` THEN `t/(t+c)` (2 divisions) becomes one
/// division against a directly-computed denominator. Bounded `[0, 1)` for
/// `raw_abs_err >= 0`, same contract as `saturate(raw_abs_err/(1+k*act),
/// c)`.
#[inline]
fn pjnd_transducer(raw_abs_err: f64, act: f64, k: f64, c: f64) -> f64 {
    raw_abs_err / (raw_abs_err + c * (1.0 + k * act))
}

include!("feature_v2_kernels.rs");

#[cfg(all(test, feature = "feature-regime-v2"))]
mod global_contrast_rev2_tests {
    use super::*;

    /// **The F5 gate.** Replaces (does not re-run) the synthetic-image
    /// `free_extras_match_the_944_append_block` comparison, which could not
    /// see this defect because synthetic content does not produce the
    /// flat-but-not-constant regions where `mean²/gvar` is large.
    ///
    /// The property under test is the one that actually broke: given raw
    /// moments carrying only f32-grade precision — which is all either route
    /// can deliver, since both accumulate in f32 lanes — how far does the
    /// computed `GLOBAL_CGAIN`/`GLOBAL_CLOSS` land from the exact answer?
    ///
    /// MEASURED consequence on real pixels (60 source images, 1,120 paired
    /// tranche cells, `benchmarks/feature_rev2_2026-09-05.md`): free-vs-append
    /// disagreement past the 2e-5 bar falls from **8.30 % to 0.18 %** and the
    /// worst cell from 2.31e-4 to 2.49e-5.
    #[test]
    fn paired_form_survives_f32_grade_moments_where_the_raw_form_does_not() {
        // A flat-but-not-constant region: large mean, tiny variance, tiny
        // distortion — the configuration where `mean^2/gvar` is the amplifier.
        let n = 65536.0f64;
        let (ms, md) = (0.5f64, 0.5f64 + 3e-5);
        let (var_s, var_d) = (1.0e-6f64, 1.3e-6f64);

        let sum_s = ms * n;
        let sum_d = md * n;
        let sum_s2 = (var_s + ms * ms) * n;
        let sum_d2 = (var_d + md * md) * n;
        let sum_ds = (md - ms) * n;
        // sum((d-s)(d+s)) = sum(d^2) - sum(s^2), formed per pixel.
        let sum_dd = sum_d2 - sum_s2;

        // Exact answer, in full f64 from the variances themselves.
        let (want_gain, want_loss) = bounded_excess_pair(var_d, var_s, C_GCONTRAST);

        // f32-grade moments: round the four raw totals to f32, which is the
        // precision any f32 lane accumulation can deliver no matter how it is
        // ordered or compensated. The PAIRED inputs are per-pixel differences,
        // so they are small and survive the same rounding.
        let f32r = |x: f64| x as f32 as f64;
        let (raw_gain, raw_loss) = {
            let (_, g, l) = global_stats_from_raw_moments(
                f32r(sum_s),
                f32r(sum_d),
                f32r(sum_s2),
                f32r(sum_d2),
                f32r(sum_dd),
                f32r(sum_ds),
                n,
                false,
            );
            (g, l)
        };
        let (pair_gain, pair_loss) = {
            let (_, g, l) = global_stats_from_raw_moments(
                f32r(sum_s),
                f32r(sum_d),
                f32r(sum_s2),
                f32r(sum_d2),
                f32r(sum_dd),
                f32r(sum_ds),
                n,
                true,
            );
            (g, l)
        };

        let raw_err = (raw_gain - want_gain)
            .abs()
            .max((raw_loss - want_loss).abs());
        let pair_err = (pair_gain - want_gain)
            .abs()
            .max((pair_loss - want_loss).abs());
        // The bar is the audit's own 2e-5 route-parity bar, not a number
        // picked to make the test pass: the defect is exactly "the raw form
        // lands further from truth than the bar allows".
        const BAR: f64 = 2e-5;
        assert!(
            raw_err > BAR,
            "the raw form should cross the parity bar here (that IS the defect); got {raw_err}"
        );
        assert!(
            pair_err < BAR / 10.0 && pair_err < raw_err / 10.0,
            "paired form must land an order below the bar: paired {pair_err} vs raw {raw_err}"
        );
    }

    /// The reassociation is an ALGEBRAIC IDENTITY, so with exact inputs the
    /// two forms must agree. If they ever do not, the revision changed the
    /// definition rather than its conditioning.
    #[test]
    fn the_two_forms_are_the_same_function_in_exact_arithmetic() {
        for &(ms, md, var_s, var_d) in &[
            (0.5f64, 0.5f64, 0.01f64, 0.02f64),
            (0.2, 0.9, 0.3, 0.05),
            (0.0, 0.0, 0.0, 0.0),
            (1000.0, 1000.5, 4.0, 3.0),
        ] {
            let n = 4096.0;
            let sum_s = ms * n;
            let sum_d = md * n;
            let sum_s2 = (var_s + ms * ms) * n;
            let sum_d2 = (var_d + md * md) * n;
            let a = global_stats_from_raw_moments(
                sum_s,
                sum_d,
                sum_s2,
                sum_d2,
                sum_d2 - sum_s2,
                (md - ms) * n,
                n,
                false,
            );
            let b = global_stats_from_raw_moments(
                sum_s,
                sum_d,
                sum_s2,
                sum_d2,
                sum_d2 - sum_s2,
                (md - ms) * n,
                n,
                true,
            );
            let tol = 1e-9 * a.1.abs().max(a.2.abs()).max(1.0);
            assert_eq!(a.0, b.0, "GLOBAL_DMEAN must be untouched by the revision");
            assert!(
                (a.1 - b.1).abs() <= tol && (a.2 - b.2).abs() <= tol,
                "{a:?} vs {b:?}"
            );
        }
    }
}

/// The three luminance-binned bounded-error values, from the dark/bright
/// weighted sums plus `Σ mse_i` and `n` — the ONE implementation, shared by
/// [`finish_append`] (reading the append kernel's accumulators) and the
/// free-extras finalize (reading the BIT-SAME-DEFINITION sums off the v1
/// fused kernel). Returns `[dark, mid, bright]`, each already `clamp(0,2)`.
///
/// The mid bin is DERIVED, not accumulated: the Bernstein weights
/// `(1−t)²`, `2t(1−t)`, `t²` are a partition of unity, so
/// `Σw_mid = n − Σw_dark − Σw_bright` and
/// `Σw_mid·v = Σv − Σw_dark·v − Σw_bright·v`.
#[inline]
fn lum_bins_from_weighted_sums(
    ws_dark: WeightedSum,
    ws_bright: WeightedSum,
    sum_mse: f64,
    n_f: f64,
) -> [f64; 3] {
    let ws_mid = WeightedSum {
        num: sum_mse - ws_dark.num - ws_bright.num,
        den: n_f - ws_dark.den - ws_bright.den,
    };
    [
        ws_dark.finish().clamp(0.0, 2.0),
        ws_mid.finish().clamp(0.0, 2.0),
        ws_bright.finish().clamp(0.0, 2.0),
    ]
}

fn finish_append(
    dense: &DenseAccum,
    app: &AppendAccum,
    grad: &GradientAccum,
    n: usize,
    cross: bool,
    toggles: V2NewFeatureToggles,
    out: &mut [f64],
) {
    debug_assert_eq!(out.len(), FEATURES_PER_CHANNEL_APPEND);
    let n_f = n as f64;
    #[inline]
    fn clamp01(v: f64) -> f64 {
        v.clamp(0.0, 1.0)
    }
    #[inline]
    fn dev_from_moments(sum: f64, sum_sq: f64, n_f: f64) -> f64 {
        let mean = sum / n_f;
        ((sum_sq / n_f) - mean * mean).max(0.0).sqrt()
    }

    out[idx_append::XMASK_TRANSDUCER] = if cross {
        clamp01(app.sum_xmask / n_f)
    } else {
        0.0
    };
    out[idx_append::LUM_TRANSDUCER] = if cross {
        clamp01(app.sum_lumt / n_f)
    } else {
        0.0
    };
    // Mid bin derived from the Bernstein partition of unity — see
    // [`lum_bins_from_weighted_sums`], the shared owner of all three.
    let lum = lum_bins_from_weighted_sums(app.ws_dark, app.ws_bright, app.sum_mse, n_f);
    out[idx_append::LUM_DARK_ERR] = lum[0];
    out[idx_append::LUM_MID_ERR] = lum[1];
    out[idx_append::LUM_BRIGHT_ERR] = lum[2];
    out[idx_append::MSCN_DIFF_MEAN] = clamp01(app.sum_mscn / n_f);
    out[idx_append::MSCN_DIFF_L2] = clamp01(app.sum_mscn2 / n_f);
    out[idx_append::CONTRAST_GAIN] = clamp01(app.sum_cgain / n_f);
    out[idx_append::CONTRAST_LOSS] = clamp01(app.sum_closs / n_f);
    out[idx_append::TEXTURE_DISSIM] = clamp01(app.sum_tex / n_f);
    out[idx_append::GMS_DEV2] = if toggles.gradient_features {
        clamp01(dev_from_moments(grad.sum_gms, grad.sum_gms2, n_f))
    } else {
        0.0
    };
    out[idx_append::ART_DEV2] = clamp01(dev_from_moments(dense.sum_art, app.sum_art2, n_f));
    out[idx_append::DET_DEV2] = clamp01(dev_from_moments(dense.sum_det, app.sum_det2, n_f));
    let (g_dmean, g_cgain, g_closs) = global_stats_from_raw_moments(
        app.sum_s,
        app.sum_d,
        app.sum_s2,
        app.sum_d2,
        app.sum_dd,
        app.sum_ds,
        n_f,
        toggles.formula_revision.paired_global_contrast(),
    );
    out[idx_append::GLOBAL_DMEAN] = g_dmean;
    out[idx_append::GLOBAL_CGAIN] = clamp01(g_cgain);
    out[idx_append::GLOBAL_CLOSS] = clamp01(g_closs);
    out[idx_append::GRAD_SRC_MEAN] = if toggles.gradient_features {
        saturate(grad.sum_grad_src / n_f, C_GRADM)
    } else {
        0.0
    };
}

/// `out[i] = input[i]²` — the streamed walk's σ-split square (reads the
/// wide window in place, writes into a free scratch buffer). Plain scalar
/// loop; LLVM auto-vectorizes the independent multiply.
fn square_into(input: &[f32], out: &mut [f32]) {
    for (o, &v) in out.iter_mut().zip(input.iter()) {
        *o = v * v;
    }
}

/// Per-channel, per-scale raw sums for the folded v1 feature blocks.
/// The canonical fused kernel supplies basic and peak (max/L8) accumulators,
/// plus the weighted pools and free extras requested by the feature plan.
/// Finalizers below emit the corresponding slots with global scale
/// normalization. Peak slots 156..228 are supported and consumed by candidate
/// models; they are not deprecated or silently discarded.
#[derive(Debug, Clone, Copy)]
struct V1BasicSums {
    // featacc: every f64 sum slot is a `featcanon::SumVar` — production
    // resolves to `Seq` (the plain `+=` below), a measurement mode to the
    // matching lane/Neumaier form. Two constructors pick the ELEMENT axis:
    // `meas_f64` for slots fed f64 strip partials (`accumulate`), `meas_f32`
    // for slots fed f32-native terms (`accumulate_block` in z1max) — under
    // `c32` the former stays sequential (f64 terms) and the latter takes
    // f32 lanes.
    ssim_d: crate::featcanon::SumVar,
    ssim_d4: crate::featcanon::SumVar,
    ssim_d2: crate::featcanon::SumVar,
    edge_art: crate::featcanon::SumVar,
    edge_art4: crate::featcanon::SumVar,
    edge_art2: crate::featcanon::SumVar,
    edge_det: crate::featcanon::SumVar,
    edge_det4: crate::featcanon::SumVar,
    edge_det2: crate::featcanon::SumVar,
    mse: crate::featcanon::SumVar,
    hf_sq_src: crate::featcanon::SumVar,
    hf_sq_dst: crate::featcanon::SumVar,
    hf_abs_src: crate::featcanon::SumVar,
    hf_abs_dst: crate::featcanon::SumVar,
    // --- v1 pool blocks (`V2NewFeatureToggles::v1_pools`): the peak
    //     accumulators the fused kernel returns anyway, plus the masked /
    //     IW sums v1's extended strip section derives per band
    //     (`streaming::ScaleAccumulators` field-for-field). ---
    ssim_d8: crate::featcanon::SumVar,
    edge_art8: crate::featcanon::SumVar,
    edge_det8: crate::featcanon::SumVar,
    // Maxima are order-free; widened to f64 so the exact map eval can push
    // f64 block maxima without narrowing (production values were f32, and
    // f32→f64 is lossless, so production bits are unchanged).
    ssim_max: f64,
    edge_art_max: f64,
    edge_det_max: f64,
    masked_ssim_d: crate::featcanon::SumVar,
    masked_ssim_d4: crate::featcanon::SumVar,
    masked_ssim_d2: crate::featcanon::SumVar,
    masked_art4: crate::featcanon::SumVar,
    masked_det4: crate::featcanon::SumVar,
    masked_mse: crate::featcanon::SumVar,
    iw_ssim_d: crate::featcanon::SumVar,
    iw_ssim_d4: crate::featcanon::SumVar,
    iw_ssim_d2: crate::featcanon::SumVar,
    iw_art4: crate::featcanon::SumVar,
    iw_det4: crate::featcanon::SumVar,
    iw_mse: crate::featcanon::SumVar,
    // --- FREE raw moments ([`V1FreeExtras::RawMoments`]) ---
    // Σsrc, Σdst, Σsrc², Σdst² over this (scale, channel). Zero unless the
    // request asked for them; the fused kernel's own fields of the same
    // name are their only source.
    sum_s: crate::featcanon::SumVar,
    sum_d: crate::featcanon::SumVar,
    sum_s2: crate::featcanon::SumVar,
    sum_d2: crate::featcanon::SumVar,
    /// Revision 2's paired second moment `Σ(d−s)(d+s)`, carried alongside the
    /// raw pair so `global_stats_from_raw_moments` can pick per revision.
    sum_dd: crate::featcanon::SumVar,
    /// Revision 2's paired first difference `Σ(d−s)`.
    sum_ds: crate::featcanon::SumVar,
    // --- FREE bounded error ([`V1FreeExtras::RawMomentsPlusBoundedErr`]) ---
    // Σ sat((s−d)², C_MSE) over this (scale, channel), and the Y-only
    // dark/bright weighted sums of that same per-pixel value. Zero unless
    // the request asked for them.
    sum_msat: crate::featcanon::SumVar,
    lum_wd_num: crate::featcanon::SumVar,
    lum_wd_den: crate::featcanon::SumVar,
    lum_wb_num: crate::featcanon::SumVar,
    lum_wb_den: crate::featcanon::SumVar,
}

impl V1BasicSums {
    fn basic_raw(&self) -> crate::fused::StripChannelAccum {
        let mut out = crate::fused::StripChannelAccum::zero();
        out.ssim_d = self.ssim_d.fin();
        out.ssim_d4 = self.ssim_d4.fin();
        out.ssim_d2 = self.ssim_d2.fin();
        out.edge_art = self.edge_art.fin();
        out.edge_art4 = self.edge_art4.fin();
        out.edge_art2 = self.edge_art2.fin();
        out.edge_det = self.edge_det.fin();
        out.edge_det4 = self.edge_det4.fin();
        out.edge_det2 = self.edge_det2.fin();
        out.mse = self.mse.fin();
        out.hf_sq_src = self.hf_sq_src.fin();
        out.hf_sq_dst = self.hf_sq_dst.fin();
        out.hf_abs_src = self.hf_abs_src.fin();
        out.hf_abs_dst = self.hf_abs_dst.fin();
        out.ssim_d8 = self.ssim_d8.fin();
        out.edge_art8 = self.edge_art8.fin();
        out.edge_det8 = self.edge_det8.fin();
        out.ssim_max = self.ssim_max as f32;
        out.edge_art_max = self.edge_art_max as f32;
        out.edge_det_max = self.edge_det_max as f32;
        out
    }

    /// Merge-level accumulator fed by `accumulate` (f64 strip partials).
    fn meas_f64() -> Self {
        Self::at_revision(crate::ssim_form::active_revision())
    }

    fn at_revision(revision: FormulaRevision) -> Self {
        use crate::featcanon::SumVar;
        Self {
            ssim_d: SumVar::f64_elems(crate::featcanon::mode(revision)),
            ssim_d4: SumVar::f64_elems(crate::featcanon::mode(revision)),
            ssim_d2: SumVar::f64_elems(crate::featcanon::mode(revision)),
            edge_art: SumVar::f64_elems(crate::featcanon::mode(revision)),
            edge_art4: SumVar::f64_elems(crate::featcanon::mode(revision)),
            edge_art2: SumVar::f64_elems(crate::featcanon::mode(revision)),
            edge_det: SumVar::f64_elems(crate::featcanon::mode(revision)),
            edge_det4: SumVar::f64_elems(crate::featcanon::mode(revision)),
            edge_det2: SumVar::f64_elems(crate::featcanon::mode(revision)),
            mse: SumVar::f64_elems(crate::featcanon::mode(revision)),
            hf_sq_src: SumVar::f64_elems(crate::featcanon::mode(revision)),
            hf_sq_dst: SumVar::f64_elems(crate::featcanon::mode(revision)),
            hf_abs_src: SumVar::f64_elems(crate::featcanon::mode(revision)),
            hf_abs_dst: SumVar::f64_elems(crate::featcanon::mode(revision)),
            ssim_d8: SumVar::f64_elems(crate::featcanon::mode(revision)),
            edge_art8: SumVar::f64_elems(crate::featcanon::mode(revision)),
            edge_det8: SumVar::f64_elems(crate::featcanon::mode(revision)),
            ssim_max: 0.0,
            edge_art_max: 0.0,
            edge_det_max: 0.0,
            masked_ssim_d: SumVar::f64_elems(crate::featcanon::mode(revision)),
            masked_ssim_d4: SumVar::f64_elems(crate::featcanon::mode(revision)),
            masked_ssim_d2: SumVar::f64_elems(crate::featcanon::mode(revision)),
            masked_art4: SumVar::f64_elems(crate::featcanon::mode(revision)),
            masked_det4: SumVar::f64_elems(crate::featcanon::mode(revision)),
            masked_mse: SumVar::f64_elems(crate::featcanon::mode(revision)),
            iw_ssim_d: SumVar::f64_elems(crate::featcanon::mode(revision)),
            iw_ssim_d4: SumVar::f64_elems(crate::featcanon::mode(revision)),
            iw_ssim_d2: SumVar::f64_elems(crate::featcanon::mode(revision)),
            iw_art4: SumVar::f64_elems(crate::featcanon::mode(revision)),
            iw_det4: SumVar::f64_elems(crate::featcanon::mode(revision)),
            iw_mse: SumVar::f64_elems(crate::featcanon::mode(revision)),
            sum_s: SumVar::f64_elems(crate::featcanon::mode(revision)),
            sum_d: SumVar::f64_elems(crate::featcanon::mode(revision)),
            sum_s2: SumVar::f64_elems(crate::featcanon::mode(revision)),
            sum_d2: SumVar::f64_elems(crate::featcanon::mode(revision)),
            sum_dd: SumVar::f64_elems(crate::featcanon::mode(revision)),
            sum_ds: SumVar::f64_elems(crate::featcanon::mode(revision)),
            sum_msat: SumVar::f64_elems(crate::featcanon::mode(revision)),
            lum_wd_num: SumVar::f64_elems(crate::featcanon::mode(revision)),
            lum_wd_den: SumVar::f64_elems(crate::featcanon::mode(revision)),
            lum_wb_num: SumVar::f64_elems(crate::featcanon::mode(revision)),
            lum_wb_den: SumVar::f64_elems(crate::featcanon::mode(revision)),
        }
    }

    /// Term-level accumulator fed f32-native block terms (`accumulate_block`
    /// in the z1max side pass) — `c32` takes the f32-lane form.
    fn meas_f32() -> Self {
        #[allow(unused_mut)] // the per-field overrides exist only under `oracle`
        let mut s = Self::meas_f64();
        // `c32` alone keeps f32 lanes on the term-level slots; under the c64
        // canon f64_elems and f32_elems coincide (`L64`), and `meas_f64`
        // already selected it.
        #[cfg(feature = "oracle")]
        if matches!(
            crate::featcanon::compute_mode(),
            crate::featcanon::Mode::Canon32
        ) {
            s.ssim_d = crate::featcanon::SumVar::f32_elems_meas();
            s.ssim_d4 = crate::featcanon::SumVar::f32_elems_meas();
            s.ssim_d2 = crate::featcanon::SumVar::f32_elems_meas();
            s.edge_art = crate::featcanon::SumVar::f32_elems_meas();
            s.edge_art4 = crate::featcanon::SumVar::f32_elems_meas();
            s.edge_art2 = crate::featcanon::SumVar::f32_elems_meas();
            s.edge_det = crate::featcanon::SumVar::f32_elems_meas();
            s.edge_det4 = crate::featcanon::SumVar::f32_elems_meas();
            s.edge_det2 = crate::featcanon::SumVar::f32_elems_meas();
            s.mse = crate::featcanon::SumVar::f32_elems_meas();
            s.hf_sq_src = crate::featcanon::SumVar::f32_elems_meas();
            s.hf_sq_dst = crate::featcanon::SumVar::f32_elems_meas();
            s.hf_abs_src = crate::featcanon::SumVar::f32_elems_meas();
            s.hf_abs_dst = crate::featcanon::SumVar::f32_elems_meas();
            s.ssim_d8 = crate::featcanon::SumVar::f32_elems_meas();
            s.edge_art8 = crate::featcanon::SumVar::f32_elems_meas();
            s.edge_det8 = crate::featcanon::SumVar::f32_elems_meas();
        }
        s
    }
}

impl V1BasicSums {
    /// Add another band's sums into this one. Used by both paths of
    /// [`fold_v1_basic_bands`]; the parallel path merges IN BAND ORDER so the
    /// f64 addition sequence matches the serial one exactly.
    ///
    /// featacc: `merge_from` folds lane-aligned, so under a lane mode each
    /// lane's own sequence is preserved (the same in-band-order argument, per
    /// lane).
    fn merge(&mut self, o: &V1BasicSums) {
        self.ssim_d.merge_from(&o.ssim_d);
        self.ssim_d4.merge_from(&o.ssim_d4);
        self.ssim_d2.merge_from(&o.ssim_d2);
        self.edge_art.merge_from(&o.edge_art);
        self.edge_art4.merge_from(&o.edge_art4);
        self.edge_art2.merge_from(&o.edge_art2);
        self.edge_det.merge_from(&o.edge_det);
        self.edge_det4.merge_from(&o.edge_det4);
        self.edge_det2.merge_from(&o.edge_det2);
        self.mse.merge_from(&o.mse);
        self.hf_sq_src.merge_from(&o.hf_sq_src);
        self.hf_sq_dst.merge_from(&o.hf_sq_dst);
        self.hf_abs_src.merge_from(&o.hf_abs_src);
        self.hf_abs_dst.merge_from(&o.hf_abs_dst);
        self.ssim_d8.merge_from(&o.ssim_d8);
        self.edge_art8.merge_from(&o.edge_art8);
        self.edge_det8.merge_from(&o.edge_det8);
        self.ssim_max = self.ssim_max.max(o.ssim_max);
        self.edge_art_max = self.edge_art_max.max(o.edge_art_max);
        self.edge_det_max = self.edge_det_max.max(o.edge_det_max);
        self.masked_ssim_d.merge_from(&o.masked_ssim_d);
        self.masked_ssim_d4.merge_from(&o.masked_ssim_d4);
        self.masked_ssim_d2.merge_from(&o.masked_ssim_d2);
        self.masked_art4.merge_from(&o.masked_art4);
        self.masked_det4.merge_from(&o.masked_det4);
        self.masked_mse.merge_from(&o.masked_mse);
        self.iw_ssim_d.merge_from(&o.iw_ssim_d);
        self.iw_ssim_d4.merge_from(&o.iw_ssim_d4);
        self.iw_ssim_d2.merge_from(&o.iw_ssim_d2);
        self.iw_art4.merge_from(&o.iw_art4);
        self.iw_det4.merge_from(&o.iw_det4);
        self.iw_mse.merge_from(&o.iw_mse);
        self.sum_s.merge_from(&o.sum_s);
        self.sum_d.merge_from(&o.sum_d);
        self.sum_s2.merge_from(&o.sum_s2);
        self.sum_d2.merge_from(&o.sum_d2);
        self.sum_dd.merge_from(&o.sum_dd);
        self.sum_ds.merge_from(&o.sum_ds);
        self.sum_msat.merge_from(&o.sum_msat);
        self.lum_wd_num.merge_from(&o.lum_wd_num);
        self.lum_wd_den.merge_from(&o.lum_wd_den);
        self.lum_wb_num.merge_from(&o.lum_wb_num);
        self.lum_wb_den.merge_from(&o.lum_wb_den);
    }

    /// Merge one strip's channel partials. The term-level accumulation axis
    /// lives inside the kernel that produced `s`; this merge step folds each
    /// partial in production order (lane 0 under lane modes — the O(#strips)
    /// merge sequence is unchanged across candidates).
    fn accumulate(&mut self, s: &crate::fused::StripChannelAccum) {
        self.ssim_d.add64(0, s.ssim_d);
        self.ssim_d4.add64(0, s.ssim_d4);
        self.ssim_d2.add64(0, s.ssim_d2);
        self.edge_art.add64(0, s.edge_art);
        self.edge_art4.add64(0, s.edge_art4);
        self.edge_art2.add64(0, s.edge_art2);
        self.edge_det.add64(0, s.edge_det);
        self.edge_det4.add64(0, s.edge_det4);
        self.edge_det2.add64(0, s.edge_det2);
        self.mse.add64(0, s.mse);
        self.hf_sq_src.add64(0, s.hf_sq_src);
        self.hf_sq_dst.add64(0, s.hf_sq_dst);
        self.hf_abs_src.add64(0, s.hf_abs_src);
        self.hf_abs_dst.add64(0, s.hf_abs_dst);
        // Peaks: v1 merges them the same way (`streaming.rs`
        // `accum.ssim_d8[c] += strip_acc.ssim_d8; accum.ssim_max[c] =
        // accum.ssim_max[c].max(strip_acc.ssim_max)`, …) — free to carry.
        self.ssim_d8.add64(0, s.ssim_d8);
        self.edge_art8.add64(0, s.edge_art8);
        self.edge_det8.add64(0, s.edge_det8);
        self.ssim_max = self.ssim_max.max(s.ssim_max as f64);
        self.edge_art_max = self.edge_art_max.max(s.edge_art_max as f64);
        self.edge_det_max = self.edge_det_max.max(s.edge_det_max as f64);
        // FREE raw moments: plain sums, so an unconditional merge is exact
        // whether the kernel filled them (+0.0 otherwise) — same shape as
        // the peaks above.
        self.sum_s.add64(0, s.sum_s);
        self.sum_d.add64(0, s.sum_d);
        self.sum_s2.add64(0, s.sum_s2);
        self.sum_d2.add64(0, s.sum_d2);
        self.sum_dd.add64(0, s.sum_dd);
        self.sum_ds.add64(0, s.sum_ds);
        // FREE bounded error: plain sums, merged the same unconditional way
        // as the raw moments above (+0.0 when the kernel did not fill them).
        self.sum_msat.add64(0, s.sum_msat);
        self.lum_wd_num.add64(0, s.lum_wd_num);
        self.lum_wd_den.add64(0, s.lum_wd_den);
        self.lum_wb_num.add64(0, s.lum_wb_num);
        self.lum_wb_den.add64(0, s.lum_wb_den);
    }

    /// Finalize the v1 pool blocks — peaks (6/ch), masked (6/ch), IW
    /// (6/ch) — into their v1 slot order, replicating
    /// `streaming::ScaleAccumulators::finalize` + `metric.rs`'s pass-2/3/4
    /// pushes (`.abs()` on the masked/IW ssim + art/det L4 slots, none on
    /// the peaks / mse).
    fn finalize_pools_into(
        &self,
        n: usize,
        peaks: &mut [f64],
        masked: &mut [f64],
        iw: &mut [f64],
        revision: crate::feature_defs::FormulaRevision,
    ) {
        debug_assert_eq!(peaks.len(), 6);
        debug_assert_eq!(masked.len(), 6);
        debug_assert_eq!(iw.len(), 6);
        let one_over_n = 1.0 / n as f64;
        // Hoisted out of the nine root calls below: it reads a `OnceLock`,
        // which LLVM cannot hoist for you. Same discipline as `gain_form`.
        let root_form = crate::det_math::RootForm::at_revision(Some(revision));
        peaks[0] = self.ssim_max;
        peaks[1] = self.edge_art_max;
        peaks[2] = self.edge_det_max;
        peaks[3] = (self.ssim_d8.fin() * one_over_n)
            .max(0.0)
            .eighth_root(root_form);
        peaks[4] = (self.edge_art8.fin() * one_over_n)
            .max(0.0)
            .eighth_root(root_form);
        peaks[5] = (self.edge_det8.fin() * one_over_n)
            .max(0.0)
            .eighth_root(root_form);
        masked[0] = (self.masked_ssim_d.fin() * one_over_n).abs();
        masked[1] = (self.masked_ssim_d4.fin() * one_over_n)
            .max(0.0)
            .quarter_root(root_form)
            .abs();
        masked[2] = (self.masked_ssim_d2.fin() * one_over_n)
            .max(0.0)
            .sqrt()
            .abs();
        masked[3] = (self.masked_art4.fin() * one_over_n)
            .max(0.0)
            .quarter_root(root_form)
            .abs();
        masked[4] = (self.masked_det4.fin() * one_over_n)
            .max(0.0)
            .quarter_root(root_form)
            .abs();
        masked[5] = self.masked_mse.fin() * one_over_n;
        iw[0] = (self.iw_ssim_d.fin() * one_over_n).abs();
        iw[1] = (self.iw_ssim_d4.fin() * one_over_n)
            .max(0.0)
            .quarter_root(root_form)
            .abs();
        iw[2] = (self.iw_ssim_d2.fin() * one_over_n).max(0.0).sqrt().abs();
        iw[3] = (self.iw_art4.fin() * one_over_n)
            .max(0.0)
            .quarter_root(root_form)
            .abs();
        iw[4] = (self.iw_det4.fin() * one_over_n)
            .max(0.0)
            .quarter_root(root_form)
            .abs();
        iw[5] = self.iw_mse.fin() * one_over_n;
    }

    /// Finalize the FREE v2-era slots ([`V1FreeExtras::RawMoments`]) this
    /// (scale, channel)'s raw moments can produce: the three `GLOBAL_*`
    /// append values (via the shared [`global_stats_from_raw_moments`]) and
    /// append2's `LUMA_MEAN_REF` (Y-channel only, the same
    /// `saturate(mean ref, C_LUM_T).clamp(0,1)` the append2 finalize applies
    /// to the append kernel's `sum_s`).
    fn finalize_free(
        &self,
        n: usize,
        rev: crate::feature_defs::FormulaRevision,
    ) -> ([f64; 3], f64) {
        let n_f = n as f64;
        let (dmean, cgain, closs) = global_stats_from_raw_moments(
            self.sum_s.fin(),
            self.sum_d.fin(),
            self.sum_s2.fin(),
            self.sum_d2.fin(),
            self.sum_dd.fin(),
            self.sum_ds.fin(),
            n_f,
            rev.paired_global_contrast(),
        );
        (
            [dmean, cgain.clamp(0.0, 1.0), closs.clamp(0.0, 1.0)],
            saturate(self.sum_s.fin() / n_f, C_LUM_T).clamp(0.0, 1.0),
        )
    }

    /// Finalize the class-C bounded-error slots
    /// ([`V1FreeExtras::RawMomentsPlusBoundedErr`]) from the fused kernel's
    /// `Σ mse_i` + luminance-weighted sums: the v2-348 `MSE` value and the
    /// dark / mid / bright luminance-binned means.
    ///
    /// Both routes go through [`lum_bins_from_weighted_sums`] and the same
    /// `clamp01(Σ mse_i / n)` the v2 finalize applies, so the parity gate is
    /// on the ACCUMULATION rather than on two hand-copied formulas — the
    /// discipline `global_stats_from_raw_moments` already established.
    fn finalize_free_bounded(&self, n: usize) -> (f64, [f64; 3]) {
        let n_f = n as f64;
        (
            (self.sum_msat.fin() / n_f).clamp(0.0, 1.0),
            lum_bins_from_weighted_sums(
                WeightedSum {
                    num: self.lum_wd_num.fin(),
                    den: self.lum_wd_den.fin(),
                },
                WeightedSum {
                    num: self.lum_wb_num.fin(),
                    den: self.lum_wb_den.fin(),
                },
                self.sum_msat.fin(),
                n_f,
            ),
        )
    }

    /// Finalize into one channel's 13 basic features, replicating v1's
    /// pooling + assembly EXACTLY (`streaming::ChannelAccum::finalize` +
    /// `metric.rs`'s pass-1 feature push, `.abs()` included): mean, L4
    /// (`(Σx⁴/n)^¼`), L2 (`(Σx²/n)^½`) for ssim/art/det; mse mean; the
    /// three HF ratio features with their `1e-10` guards. `n` is the
    /// scale's full pixel count (`w_s * h_s`) — identical to v1's per-scale
    /// accumulator `n` since both walks cover every row exactly once.
    fn finalize_into(
        &self,
        n: usize,
        out: &mut [f64],
        revision: crate::feature_defs::FormulaRevision,
    ) {
        debug_assert_eq!(out.len(), 13);
        let one_over_n = 1.0 / n as f64;
        let root_form = crate::det_math::RootForm::at_revision(Some(revision));
        out[0] = (self.ssim_d.fin() * one_over_n).abs();
        out[1] = (self.ssim_d4.fin() * one_over_n)
            .max(0.0)
            .quarter_root(root_form)
            .abs();
        out[2] = (self.ssim_d2.fin() * one_over_n).max(0.0).sqrt().abs();
        out[3] = (self.edge_art.fin() * one_over_n).abs();
        out[4] = (self.edge_art4.fin() * one_over_n)
            .max(0.0)
            .quarter_root(root_form)
            .abs();
        out[5] = (self.edge_art2.fin() * one_over_n).max(0.0).sqrt().abs();
        out[6] = (self.edge_det.fin() * one_over_n).abs();
        out[7] = (self.edge_det4.fin() * one_over_n)
            .max(0.0)
            .quarter_root(root_form)
            .abs();
        out[8] = (self.edge_det2.fin() * one_over_n).max(0.0).sqrt().abs();
        out[9] = self.mse.fin() * one_over_n;
        // Same owner as the buffered walk (`crate::hf_gain_form`), which is
        // what keeps the fold's v1 pools bit-identical to v1's under EVERY
        // arm rather than only under the shipped one.
        let gain_form = crate::hf_gain_form::HfGainForm::at_revision(Some(revision));
        let var_src = self.hf_sq_src.fin() * one_over_n;
        let var_dst = self.hf_sq_dst.fin() * one_over_n;
        out[10] = crate::hf_gain_form::hf_energy_loss(var_src, var_dst);
        out[12] = crate::hf_gain_form::hf_energy_gain(gain_form, var_src, var_dst);
        let mad_src = self.hf_abs_src.fin() * one_over_n;
        let mad_dst = self.hf_abs_dst.fin() * one_over_n;
        out[11] = crate::hf_gain_form::hf_mag_loss(mad_src, mad_dst);
    }
}

/// v1's band tiling constants, replicated for the fold. v1 tiles every
/// scale into 32-row bands ([`crate::streaming`]'s `STRIP_INNER`) and
/// V-blurs each band from a buffer extending `overlap = blur_passes(1) ×
/// radius(5) = 5` rows past the band, mirror-clamped at the PLANE bounds.
/// The band layout is part of v1's numerics contract: the f32 sliding
/// V-blur state re-initializes at every band's buffer top, so pooled
/// sums depend on the tiling. The fold reproduces the tiling exactly —
/// same buffer extents, same init points, same band order — which makes
/// the accumulated sums bit-identical to v1's whenever the plane values
/// themselves are (true-width == v1's SIMD-padded width; see the parity
/// test for the padded-width caveat).
const V1_BAND_ROWS: usize = 32;
/// Band slots per strip — the fixed fan-out width of the band-parallel path.
const V1_BANDS_PER_STRIP: usize = STRIP_ROWS / V1_BAND_ROWS;
const V1_BAND_OVERLAP: usize = 5;

/// v1's masking / IW strengths (`metric::config_from_params`:
/// `extended_masking_strength: 4.0`, `iw_strength: 4.0`) — the fold's pool
/// replay uses the same constants so the block reproduces v1's numbers.
const V1_MASK_K: f32 = 4.0;
const V1_IW_K: f32 = 4.0;

/// What a v1 band does with its [`FoldPoolScratch`] — the band-level
/// resolution of [`V1PoolsMode`].
///
/// [`Self::HOnly`] exists because the pool block is NOT three independent
/// jobs. Peaks ride the fused kernel's unconditional L8/max tier, while
/// masked and IW share ONE activity chain, ONE sigma store and three
/// `*_inline_both` kernels that compute both strengths in a single sweep. So
/// the only compute boundary inside `f156..372` is *peaks* vs
/// *masked-and-IW*, and this enum is that boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BandPoolWork {
    /// No pool arithmetic at all. The scratch is present only so a
    /// [`FoldHSource::SelfBlur`] band can own its four H planes.
    HOnly,
    /// Activity chain + `edge_diff_channel_inline_both` (the carrier slots'
    /// `art_4th`); no sigma store, no MSE, no masked/IW SSIM.
    Carriers,
    /// Every pool slot: activity chain, sigma store, and all three
    /// `*_inline_both` kernels.
    Full,
}

#[cfg(test)]
thread_local! {
    // Observe actual weighted-pool execution, not planner declarations.
    static POOL_TEST_WIDTHS: std::cell::RefCell<Option<Vec<usize>>> = const { std::cell::RefCell::new(None) };
    // Scales whose strips the streaming walk skipped (declared plans reading nothing there).
    static SKIPPED_SCALES: std::cell::RefCell<u8> = const { std::cell::RefCell::new(0) };
}

/// Band-local planes for the v1 pool replay (`V2NewFeatureToggles::v1_pools`):
/// sized for one v1 band buffer (`V1_BAND_ROWS + 2 * V1_BAND_OVERLAP` rows ×
/// width), grown on first use per channel accumulator and reused across
/// bands, strips and scales (the widest scale sizes it once).
#[derive(Debug, Clone, Default)]
struct FoldPoolScratch {
    /// V-blurred `mu1` / `mu2` (the fused kernel's `store_mu` side-output).
    stable_sd: Vec<f32>,
    mu1_v: Vec<f32>,
    mu2_v: Vec<f32>,
    /// `|src − H_blur(src)|` (v1's `bufs.mask` role).
    act_raw: Vec<f32>,
    /// One-pass-blurred activity (v1's `bufs.mul_buf` role).
    act: Vec<f32>,
    /// V-blurred `ssq_h` / `s12_h` (v1's `bufs.sigma1_sq` / `bufs.sigma12`);
    /// `ssq_v` doubles as the activity blur's temp before it is filled.
    ssq_v: Vec<f32>,
    s12_v: Vec<f32>,
    /// BAND-LOCAL H-blurred planes, for the self-blur band shape
    /// ([`FoldHSource::SelfBlur`]). Only allocated when that shape runs.
    h: [Vec<f32>; 4],
}

impl FoldPoolScratch {
    fn ensure(&mut self, n: usize) {
        for b in [
            &mut self.mu1_v,
            &mut self.mu2_v,
            &mut self.act_raw,
            &mut self.act,
            &mut self.ssq_v,
            &mut self.s12_v,
        ] {
            if b.len() < n {
                b.resize(n, 0.0);
            }
        }
    }

    /// Grow the band-local H planes to `n` elements. Separate from
    /// [`Self::ensure`] so the precomputed-H shape never allocates them.
    fn ensure_h(&mut self, n: usize) {
        for b in self.h.iter_mut() {
            if b.len() < n {
                b.resize(n, 0.0);
            }
        }
    }
}

/// Where a v1 band's four H-blurred planes come from.
///
/// **This is the fold's memory-traffic lever, and it is a measured one.**
/// The 16-independent-process throughput test at 2304² (see
/// `benchmarks/fold_mt_scaling_2026-08-31.md`) puts the fold's ceiling at
/// **4.2×** on a box where the buffered walk reaches **10.9×** — from the same
/// serial speed. The whole difference is this: buffered blurs a band's rows
/// into band-PRIVATE buffers and consumes them in the SAME task, while the
/// fold blurred the whole 148-row strip window in one set of tasks and read it
/// back in another, so four planes round-tripped through L3/DRAM per strip.
///
/// `SelfBlur` gives the fold buffered's shape: each band H-blurs exactly the
/// `[b0 − overlap, b1 + overlap)` rows it is about to consume, into its own
/// scratch. It costs redundant blur at the band seams — 42 rows blurred per 32
/// consumed against 148 per 128, i.e. **+40 % blur compute** — and buys a
/// self-contained task.
///
/// Bit-exact either way: the H blur is an independent per-row recurrence
/// (`phase_a_blur_bands_are_bit_exact`), so a band's rows carry the same bits
/// whichever call produced them, and the band kernel below reads the same rows
/// at the same offsets. `fold_self_blur_matches_precomputed_h` is the gate.
#[derive(Clone, Copy)]
enum FoldHSource<'a> {
    /// Rev5 consumers share the already V-blurred planes.
    Vertical([&'a [f32]; 4]),
    /// The four planes, over the strip's whole wide window (row 0 of each
    /// plane is wide-window row 0).
    Precomputed([&'a [f32]; 4]),
    /// Each band blurs its own rows from the raw windows into
    /// `FoldPoolScratch::h`.
    SelfBlur,
}

/// One v1-aligned band of the fold hook. Extracted from `fold_v1_basic_bands`
/// verbatim so the band-parallel and serial paths run IDENTICAL code — the
/// only difference between them is which `V1BasicSums` the band accumulates
/// into, and in what order those are merged.
#[allow(clippy::too_many_arguments)]
fn fold_v1_one_band(
    b0: usize,
    width: usize,
    rows_end: usize,
    strip_y0: usize,
    halo_offset: usize,
    height: usize,
    h_src: FoldHSource<'_>,
    raw: [&[f32]; 2],
    sums: &mut V1BasicSums,
    mut pools: Option<(&mut FoldPoolScratch, BandPoolWork)>,
    // `free`: which FREE extra accumulators the fused kernel should carry —
    // the raw moments (Σs, Σd, Σs², Σd²) that finalize the append block's
    // GLOBAL_* trio and append2's LUMA_MEAN_REF, and/or the class-C
    // bounded-error family (v2 `MSE` + the Y luminance bins). None of them
    // needs a new plane, load or pass ([`V1FreeExtras`],
    // [`crate::fused::FreeExtrasWork`]).
    free: crate::fused::FreeExtrasWork,
) -> usize {
    let [src, dst] = raw;
    let rows = 0..rows_end;
    let _ = &rows;
    let b1 = (b0 + V1_BAND_ROWS).min(rows_end);
    // v1's band buffer: [b0 − overlap, b1 + overlap) clamped to the
    // plane — the kernel's own mirror at the buffer edges then
    // reproduces v1's boundary/init behavior bit-for-bit.
    let top = b0.saturating_sub(V1_BAND_OVERLAP);
    let bot = (b1 + V1_BAND_OVERLAP).min(height);
    let lt = top + halo_offset - strip_y0;
    let lb = bot + halo_offset - strip_y0;
    let h_local = bot - top;
    let inner_start = b0 - top;
    let inner_h = b1 - b0;
    let span = lt * width..lb * width;
    let mut empty_mu1: [f32; 0] = [];
    let mut empty_mu2: [f32; 0] = [];
    let mut empty_sd: [f32; 0] = [];
    if let FoldHSource::Vertical([a, b, c, d]) = h_src {
        debug_assert!(free.revision() >= FormulaRevision::Rev5);
        // Radius zero selects the same canonical per-element/pool owner,
        // reading the shared vertical values without another blur sweep.
        sums.accumulate(&crate::fused::fused_vblur_features_ssim(
            &a[span.clone()],
            &b[span.clone()],
            &c[span.clone()],
            &d[span.clone()],
            &src[span.clone()],
            &dst[span],
            width,
            h_local,
            inner_start,
            inner_h,
            0,
            &mut [],
            &mut [],
            false,
            &mut [],
            false,
            &mut [],
            &mut [],
            false,
            free,
            crate::fused::ExtPoolsWork::default(),
            &[],
        ));
        return b1;
    }
    if let Some((ps, work)) = pools.as_mut().map(|(p, f)| (&mut **p, *f)) {
        let band_n = h_local * width;
        // Size to the WIDEST band this plane can ever hold, not to this band's
        // own height (fold-footprint lane). `ensure` grows through
        // `Vec::resize`, which reserves `max(2*cap, need)`: the first band of
        // a strip is TOP-CLAMPED to `V1_BAND_ROWS + V1_BAND_OVERLAP` rows, so
        // slot 0 was sized 37·width and then DOUBLED to 74·width on the first
        // interior band — 76 % more than it can use, on 10 planes × 3
        // channels. Measured (heaptrack, 1152²): 27.65 MB against the 23.22 MB
        // the slots actually hold, i.e. 3840·width bytes of pure growth slack.
        // Asking for the maximum up front makes every slot exactly 42·width.
        //
        // The pool planes are grown BELOW the `HOnly` arm: peaks-only bands
        // never touch them (feature-cost lane), so allocating them here would
        // hand a skipping request the footprint it is skipping.
        let band_cap_n = (V1_BAND_ROWS + 2 * V1_BAND_OVERLAP).min(height.max(1)) * width;
        debug_assert!(
            band_n <= band_cap_n,
            "band {band_n} exceeds the {band_cap_n} capacity bound"
        );
        let self_blur = matches!(h_src, FoldHSource::SelfBlur);
        if self_blur {
            ps.ensure_h(band_cap_n);
        }
        if work == BandPoolWork::HOnly {
            // PEAKS mode. Identical arithmetic to the `pools == None` arm
            // below — every store flag off, no activity, no masked/IW kernel
            // — run on whichever H planes this band is using. The peak sums
            // the caller wants are the kernel's unconditional L8/max tier,
            // so this costs exactly what `V1PoolsMode::Off` costs and the
            // emitted peak slots are bit-identical to `Full`'s.
            if self_blur {
                let [h0, h1, h2, h3] = &mut ps.h;
                crate::blur::fused_blur_h_ssim_at_revision(
                    &src[span.clone()],
                    &dst[span.clone()],
                    &mut h0[..band_n],
                    &mut h1[..band_n],
                    &mut h2[..band_n],
                    &mut h3[..band_n],
                    width,
                    h_local,
                    BLUR_RADIUS,
                    free.revision(),
                );
            }
            let (mu1_h, mu2_h, ssq_h, s12_h, span_h) = if self_blur {
                (
                    &ps.h[0][..band_n],
                    &ps.h[1][..band_n],
                    &ps.h[2][..band_n],
                    &ps.h[3][..band_n],
                    0..band_n,
                )
            } else {
                let FoldHSource::Precomputed([a, b, c, d]) = h_src else {
                    unreachable!("SelfBlur handled above")
                };
                (a, b, c, d, span.clone())
            };
            sums.accumulate(&crate::fused::fused_vblur_features_ssim(
                &mu1_h[span_h.clone()],
                &mu2_h[span_h.clone()],
                &ssq_h[span_h.clone()],
                &s12_h[span_h],
                &src[span.clone()],
                &dst[span],
                width,
                h_local,
                inner_start,
                inner_h,
                BLUR_RADIUS,
                &mut empty_mu1,
                &mut empty_mu2,
                false,
                &mut empty_sd,
                false,
                &mut [],
                &mut [],
                false,
                free,
                crate::fused::ExtPoolsWork::default(),
                &[],
            ));
            return b1;
        }
        #[cfg(test)]
        POOL_TEST_WIDTHS.with_borrow_mut(|widths| {
            if let Some(widths) = widths {
                widths.push(width);
            }
        });
        let full = work == BandPoolWork::Full;
        let stable = free.revision() >= crate::feature_defs::FormulaRevision::Rev3;
        if stable && full {
            ps.stable_sd.resize(band_cap_n, 0.0);
        }
        ps.ensure(band_cap_n);
        // ONE destructure, so the band-local H planes can be READ while the
        // pool planes stay mutable — disjoint fields of the same `&mut`.
        let FoldPoolScratch {
            stable_sd,
            mu1_v,
            mu2_v,
            act_raw,
            act,
            ssq_v,
            s12_v,
            h,
        } = ps;
        let [h0, h1, h2, h3] = h;
        if self_blur {
            // The band's OWN H blur, over exactly the rows it consumes.
            // Bit-identical to reading those rows out of a whole-window call
            // (`phase_a_blur_bands_are_bit_exact`); the point is that these
            // four planes never leave this task.
            crate::blur::fused_blur_h_ssim_at_revision(
                &src[span.clone()],
                &dst[span.clone()],
                &mut h0[..band_n],
                &mut h1[..band_n],
                &mut h2[..band_n],
                &mut h3[..band_n],
                width,
                h_local,
                BLUR_RADIUS,
                free.revision(),
            );
        }
        let (mu1_h, mu2_h, ssq_h, s12_h, span_h) = match h_src {
            FoldHSource::Vertical(_) => unreachable!("shared vertical planes handled above"),
            FoldHSource::Precomputed([a, b, c, d]) => (a, b, c, d, span.clone()),
            FoldHSource::SelfBlur => (
                &h0[..band_n],
                &h1[..band_n],
                &h2[..band_n],
                &h3[..band_n],
                0..band_n,
            ),
        };
        // ORDER: activity FIRST, fused kernel SECOND. The activity blur
        // borrows `ssq_v` as its scratch temp, and the fused kernel then
        // overwrites `ssq_v`'s INNER rows with the real V-blurred sigma
        // (`store_sigma`) — so the two uses are disjoint in time and the
        // temp costs no extra buffer. Reordering is sum-neutral: the
        // `sums` fields written here (masked/IW) are disjoint from the
        // ones `accumulate` writes (basic).
        //
        // v1 computes `|src − H_blur(src)|` with its own H kernel
        // (`box_blur_h_into_abs_diff`); the fold already holds the
        // H-blurred src as `mu1_h` (the shared fused H-pass plane), so the
        // activity is one abs-diff pass — the pool parity gate proves the
        // two H kernels agree bit-for-bit on these planes.
        crate::simd_ops::abs_diff_into(
            &src[span.clone()],
            &mu1_h[span_h.clone()],
            &mut act_raw[..band_n],
        );
        // Revision 3: the SAME fused extension the streaming path runs
        // (`streaming.rs`, `fused_ext`) — the activity's V blur and every
        // masked/IW pool ride inside the SSIM V sweep, over the same bands
        // in the same order, which is what keeps the rev-3 fold/streaming
        // pool parity gate bit-exact. Here `act` then holds the H-BLURRED
        // activity the sweep V-blurs in-register.
        let fused_ext = stable;
        if fused_ext {
            crate::blur::box_blur_h(
                &act_raw[..band_n],
                &mut act[..band_n],
                width,
                h_local,
                BLUR_RADIUS,
            );
        } else {
            crate::blur::box_blur_1pass_into(
                &act_raw[..band_n],
                &mut act[..band_n],
                &mut ssq_v[..band_n],
                width,
                h_local,
                BLUR_RADIUS,
            );
        }
        let ext = crate::fused::ExtPoolsWork {
            on: fused_ext,
            mask: true,
            iw: true,
            k_mask: V1_MASK_K,
            k_iw: V1_IW_K,
        };
        // `store_sigma` replaces the two `box_blur_v_from_copy(ssq_h →
        // ssq_v)` / `(s12_h → s12_v)` band sweeps the `Full` arm used to
        // run after this call: the fused kernel already carries the same
        // running V-blur sums in registers and divides by the same
        // `1.0 / diam`, so the stored planes are BIT-IDENTICAL to what
        // that second sweep produced (and it writes only the inner rows —
        // the only rows the masked/IW SSIM kernel reads). `Carriers`
        // needs no sigma, so it stores none and `ssq_v` simply stays the
        // activity temp.
        let acc = crate::fused::fused_vblur_features_ssim(
            &mu1_h[span_h.clone()],
            &mu2_h[span_h.clone()],
            &ssq_h[span_h.clone()],
            &s12_h[span_h.clone()],
            &src[span.clone()],
            &dst[span.clone()],
            width,
            h_local,
            inner_start,
            inner_h,
            BLUR_RADIUS,
            &mut mu1_v[..band_n],
            &mut mu2_v[..band_n],
            true,
            if stable && full {
                &mut stable_sd[..band_n]
            } else {
                &mut empty_sd
            },
            stable && full,
            &mut ssq_v[..band_n],
            &mut s12_v[..band_n],
            full,
            free,
            ext,
            if fused_ext { &act[..band_n] } else { &[] },
        );
        sums.accumulate(&acc);
        if fused_ext {
            if full {
                sums.masked_mse.add64(0, acc.masked_mse);
                sums.iw_mse.add64(0, acc.iw_mse);
                sums.masked_ssim_d.add64(0, acc.masked_ssim_d);
                sums.masked_ssim_d4.add64(0, acc.masked_ssim_d4);
                sums.masked_ssim_d2.add64(0, acc.masked_ssim_d2);
                sums.iw_ssim_d.add64(0, acc.iw_ssim_d);
                sums.iw_ssim_d4.add64(0, acc.iw_ssim_d4);
                sums.iw_ssim_d2.add64(0, acc.iw_ssim_d2);
            }
            sums.masked_art4.add64(0, acc.masked_art4);
            sums.masked_det4.add64(0, acc.masked_det4);
            sums.iw_art4.add64(0, acc.iw_art4);
            sums.iw_det4.add64(0, acc.iw_det4);
            return b1;
        }
        let inner = inner_start * width..(inner_start + inner_h) * width;
        let inner_src = &src[span.start + inner.start..span.start + inner.end];
        let inner_dst = &dst[span.start + inner.start..span.start + inner.end];
        let inner_mu1 = &mu1_v[inner.clone()];
        let inner_mu2 = &mu2_v[inner.clone()];
        let act_inner = &act[inner.clone()];
        if full {
            // The masked/IW SSIM + MSE slots — the sigma planes are
            // already in `ssq_v` / `s12_v` (the fused kernel's
            // `store_sigma` side-output above), so this arm no longer runs
            // its own two V-blur band sweeps.
            let (mse_m, mse_i) = crate::simd_ops::build_inline_mse(
                act_inner, V1_MASK_K, V1_IW_K, inner_src, inner_dst,
            );
            sums.masked_mse.add64(0, mse_m);
            sums.iw_mse.add64(0, mse_i);
            let ((sd_m, sd4_m, sd2_m), (sd_i, sd4_i, sd2_i)) = if stable {
                // `stable_sd` is sized under `stable && full` above and read
                // here under `stable` — equivalent only because this arm IS
                // the `full` one. Pinned rather than assumed: moving this
                // pooling out of `if full` would otherwise read a zeroed
                // buffer and quietly emit zero masked/IW SSIM.
                debug_assert!(full && stable_sd.len() >= inner.end);
                crate::simd_ops::ssim_signal_inline_both(
                    &stable_sd[inner.clone()],
                    act_inner,
                    V1_MASK_K,
                    V1_IW_K,
                )
            } else {
                crate::simd_ops::ssim_channel_inline_both(
                    inner_mu1,
                    inner_mu2,
                    &ssq_v[inner.clone()],
                    &s12_v[inner.clone()],
                    act_inner,
                    V1_MASK_K,
                    V1_IW_K,
                )
            };
            sums.masked_ssim_d.add64(0, sd_m);
            sums.masked_ssim_d4.add64(0, sd4_m);
            sums.masked_ssim_d2.add64(0, sd2_m);
            sums.iw_ssim_d.add64(0, sd_i);
            sums.iw_ssim_d4.add64(0, sd4_i);
            sums.iw_ssim_d2.add64(0, sd2_i);
        }
        let ((art4_m, det4_m), (art4_i, det4_i)) = crate::simd_ops::edge_diff_channel_inline_both(
            inner_src, inner_dst, inner_mu1, inner_mu2, act_inner, V1_MASK_K, V1_IW_K,
        );
        sums.masked_art4.add64(0, art4_m);
        sums.masked_det4.add64(0, det4_m);
        sums.iw_art4.add64(0, art4_i);
        sums.iw_det4.add64(0, det4_i);
    } else {
        // No pool scratch => no band-local H buffer either, so this arm only
        // serves `FoldHSource::Precomputed`. `fold_v1_basic_bands` refuses the
        // combination up front rather than reaching here.
        let FoldHSource::Precomputed([mu1_h, mu2_h, ssq_h, s12_h]) = h_src else {
            unreachable!("SelfBlur requires pool scratch (checked in fold_v1_basic_bands)")
        };
        sums.accumulate(&crate::fused::fused_vblur_features_ssim(
            &mu1_h[span.clone()],
            &mu2_h[span.clone()],
            &ssq_h[span.clone()],
            &s12_h[span.clone()],
            &src[span.clone()],
            &dst[span],
            width,
            h_local,
            inner_start,
            inner_h,
            BLUR_RADIUS,
            &mut empty_mu1,
            &mut empty_mu2,
            false,
            &mut empty_sd,
            false,
            &mut [],
            &mut [],
            false,
            free,
            crate::fused::ExtPoolsWork::default(),
            &[],
        ));
    }
    b1
}

/// Run v1's fused V-blur + basic-feature kernel over the v1-aligned bands
/// covered by one fold buffer (the FOLD hook): consumes the v2 blur
/// pass's H-planes + halo buffers directly (H-blur is per-row/stateless,
/// so H values are shareable; the V state is NOT, hence the band replay).
///
/// `rows` are the image rows this call must accumulate (the strip's inner
/// rows), `strip_y0`/`halo_offset` map image rows to buffer-local rows
/// (`local = row − strip_y0 + halo_offset`; strip path: `y0`/`HALO_P` —
/// a band's ±5-row extent always lands on real gathered rows because
/// bands and strips are both 32-row aligned and `HALO_P ≥ 5`; whole-plane
/// path: `0`/`0`), and `height` is the full plane height at this scale
/// (band extents clamp against it, exactly like v1 clamps against the
/// plane).
///
/// With `pools == None` the store flags are off, so the kernel's
/// `mu1/mu2/sd` side-outputs are never written — empty slices are safe
/// (every write in every tier is `if store_*`-gated). With `Some`
/// (`V2NewFeatureToggles::v1_pools`) each band ALSO replays v1's extended
/// strip section (`streaming::process_strip_channel`'s `need_activity`
/// block) on the band buffer: the kernel stores its V-blurred mu1/mu2 for
/// the inner rows, the ref-side activity is `|src − H_blur(src)|`
/// one-pass-blurred over the SAME band buffer (mirror-clamped at its
/// edges exactly like v1's strip buffer), the sigma planes are the shared
/// H-planes V-blurred over the band, and the three fused masked+IW
/// kernels (`build_inline_mse`, `ssim_channel_inline_both`,
/// `edge_diff_channel_inline_both`) run on the inner rows at v1's
/// `k = 4` / `k_iw = 4`. Band extents, buffer contents and reduction
/// order are v1's, so the pooled sums are bit-identical to v1's whenever
/// the plane values are.
///
/// `raw = [src, dst]` and, for [`FoldHSource::Precomputed`], the four
/// H-planes, all in the SAME buffer-local coordinate system (`local = row −
/// strip_y0 + halo_offset`): the strip path passes the scratch H-planes +
/// halo buffers; the whole-plane path passes the full-plane H-planes + the
/// real src/dst planes with `strip_y0 = halo_offset = 0`.
/// [`FoldHSource::SelfBlur`] passes no H-planes — each band produces its own
/// from `raw`, which is the memory-traffic shape (see that type's doc).
/// SelfBlur requires `pools` (the band-local H buffer lives in the band's
/// [`FoldPoolScratch`]) and is refused otherwise.
///
/// RAII closer for the [`crate::fold_timing::Phase::FoldWall`] span, so the
/// early `return` in the parallel arm still records it. Diagnostic only.
struct FoldTimingGuard(Option<std::time::Instant>);
impl Drop for FoldTimingGuard {
    fn drop(&mut self) {
        crate::fold_timing::stop(self.0.take(), crate::fold_timing::Phase::FoldWall, 0);
    }
}

#[allow(clippy::too_many_arguments)]
fn fold_v1_basic_bands(
    width: usize,
    rows: core::ops::Range<usize>,
    strip_y0: usize,
    halo_offset: usize,
    height: usize,
    h_src: FoldHSource<'_>,
    raw: [&[f32]; 2],
    sums: &mut V1BasicSums,
    pools: Option<(&mut [FoldPoolScratch], BandPoolWork)>,
    parallel: bool,
    // `free`: see [`fold_v1_one_band`] — forwarded verbatim to every band on
    // both the serial and the band-parallel path, so the two stay identical
    // by construction.
    free: crate::fused::FreeExtrasWork,
) {
    assert!(
        !matches!(h_src, FoldHSource::SelfBlur) || pools.is_some(),
        "FoldHSource::SelfBlur needs the band pool scratch for its H buffer"
    );
    debug_assert_eq!(rows.start % V1_BAND_ROWS, 0, "strips are 32-row aligned");
    let __t_fold = crate::fold_timing::start();
    let __fold_guard = FoldTimingGuard(__t_fold);
    // Band starts, in order. Bands are INDEPENDENT: each reads a clamped
    // window of the shared planes and produces its own sums, so the only
    // cross-band coupling is the accumulation itself.
    //
    // Held in a FIXED array rather than a `Vec`. This function runs once per
    // (strip, channel, scale), and a `Vec::new()` grown by `push` costs one
    // allocation plus up to two reallocs every time — measured at **232
    // allocations per 1152² fast-class compare** against a reused `V2Scratch`
    // (`zensim/tests/fastclass_alloc_steady_state.rs`,
    // `benchmarks/kernel_fastclass_2026-09-05.md` §3.2), i.e. ~4 per
    // (strip, channel), which is exactly this. The bound is structural, not a
    // guess: a strip is at most `STRIP_ROWS` rows and a band is
    // `V1_BAND_ROWS`, so there are at most `V1_BANDS_PER_STRIP` starts. The
    // assert states that rather than trusting it, so a future change to either
    // constant fails loudly here instead of silently truncating a band — which
    // would drop its rows from the sums and move feature bytes.
    let mut starts_buf = [0usize; V1_BANDS_PER_STRIP];
    let mut n_starts = 0usize;
    let mut b = rows.start;
    while b < rows.end {
        assert!(
            n_starts < V1_BANDS_PER_STRIP,
            "band starts overflow: rows {}..{} span {} rows, more than \
             V1_BANDS_PER_STRIP ({V1_BANDS_PER_STRIP}) bands of {V1_BAND_ROWS}",
            rows.start,
            rows.end,
            rows.end - rows.start
        );
        starts_buf[n_starts] = b;
        n_starts += 1;
        b = (b + V1_BAND_ROWS).min(rows.end);
    }
    let starts: &[usize] = &starts_buf[..n_starts];

    // BIT-EXACTNESS OF THE PARALLEL PATH. Each band accumulates into its own
    // zero-initialised `V1BasicSums`, and the merge below runs SEQUENTIALLY IN
    // BAND ORDER — so the sequence of f64 additions is `((0 + b0) + b1) + ...`,
    // exactly what the in-place serial loop performed. (An unordered or tree
    // reduction would NOT be: f64 addition is not associative. The max fields
    // are order-free either way.) Both paths below use the same
    // local-then-merge shape so serial and parallel are identical to each
    // other by construction, not merely by argument.
    #[cfg(feature = "threads")]
    if parallel && starts.len() > 1 {
        use rayon::prelude::*;
        // Each band gets its OWN persistent scratch slot, so nothing is
        // allocated in the hot path. `map_init(FoldPoolScratch::default)` was
        // tried first and measured a NET LOSS: it re-allocates ~580 KB per
        // worker per strip per channel.
        let (slots, work) = match pools {
            Some((s, f)) => (Some(s), f),
            None => (None, BandPoolWork::HOnly),
        };
        let locals: Vec<V1BasicSums> = match slots {
            // CHUNKED over the slots the channel actually keeps
            // (`band_slots_for`). With at least as many slots as bands —
            // every pool of 4+ threads — `per` is 1 and this is exactly the
            // old one-band-per-slot zip. With fewer, a slot's bands run
            // sequentially inside one task, which costs parallelism the pool
            // could not have supplied anyway and saves the slots outright.
            //
            // ORDER, which is the whole bit-exactness argument: `par_chunks`
            // and `par_iter_mut` are both INDEXED, so `collect` yields chunks
            // in band order and each chunk yields its bands in band order —
            // flattening reproduces the serial band sequence exactly, and the
            // merge below is the same left-to-right f64 fold either way.
            Some(slots) => {
                let per = starts.len().div_ceil(slots.len().max(1)).max(1);
                let chunked: Vec<Vec<V1BasicSums>> = starts
                    .par_chunks(per)
                    .zip(slots.par_iter_mut())
                    .map(|(chunk, ps)| {
                        let mut out = Vec::with_capacity(chunk.len());
                        for &b0 in chunk {
                            let __t = crate::fold_timing::start();
                            let mut local = V1BasicSums::at_revision(free.revision());
                            fold_v1_one_band(
                                b0,
                                width,
                                rows.end,
                                strip_y0,
                                halo_offset,
                                height,
                                h_src,
                                raw,
                                &mut local,
                                Some((&mut *ps, work)),
                                free,
                            );
                            crate::fold_timing::stop(__t, crate::fold_timing::Phase::BandBusy, 0);
                            out.push(local);
                        }
                        out
                    })
                    .collect();
                debug_assert_eq!(
                    chunked.iter().map(Vec::len).sum::<usize>(),
                    starts.len(),
                    "chunked band fan-out must cover every band exactly once"
                );
                chunked.into_iter().flatten().collect()
            }
            None => starts
                .par_iter()
                .map(|&b0| {
                    let __t = crate::fold_timing::start();
                    let mut local = V1BasicSums::at_revision(free.revision());
                    fold_v1_one_band(
                        b0,
                        width,
                        rows.end,
                        strip_y0,
                        halo_offset,
                        height,
                        h_src,
                        raw,
                        &mut local,
                        None,
                        free,
                    );
                    crate::fold_timing::stop(__t, crate::fold_timing::Phase::BandBusy, 0);
                    local
                })
                .collect(),
        };
        for l in &locals {
            sums.merge(l);
        }
        return;
    }
    #[cfg(not(feature = "threads"))]
    let _ = parallel;

    let mut pools = pools;
    for (i, &b0) in starts.iter().enumerate() {
        let mut local = V1BasicSums::at_revision(free.revision());
        fold_v1_one_band(
            b0,
            width,
            rows.end,
            strip_y0,
            halo_offset,
            height,
            h_src,
            raw,
            &mut local,
            pools
                .as_mut()
                .map(|(p, f)| (&mut p[i.min(p.len() - 1)], *f)),
            free,
        );
        sums.merge(&local);
    }
}

/// One channel-scale of the plain-v2 (V2Bounded) materialized walk: the
/// strip loop over halo-gathered wide buffers. (The folded/append regimes
/// route through the STREAMING walk since the C5 switchover — this
/// function serves only `compute_v2_features*`.)
fn compute_channel_scale_v2(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    toggles: V2NewFeatureToggles,
    moments: Option<(&[f32], &[f32])>,
    scratch: &mut ScratchV2Strip,
    out: &mut [f64],
    mut r4a: Option<Rev4CellArgs<'_>>,
) -> (f64, f64) {
    let n = width * height;
    assert_eq!(src.len(), n, "src plane length must be width*height");
    assert_eq!(dst.len(), n, "dst plane length must be width*height");
    if let Some((mu1_full, act_full)) = moments {
        assert_eq!(mu1_full.len(), n, "cached mu1 plane must be width*height");
        assert_eq!(
            act_full.len(),
            n,
            "cached activity plane must be width*height"
        );
    }
    assert_eq!(
        out.len(),
        FEATURES_PER_CHANNEL_V2_TOTAL,
        "out slice must hold exactly one channel-scale's v2 block"
    );

    // Phase-6 (§A.16 lever B): small images skip the strip machinery
    // entirely — see `STRIP_BYPASS_HEIGHT`'s doc and `compute_channel_
    // scale_v2_whole`. Currently DISABLED (`STRIP_BYPASS_HEIGHT=0`, see its
    // doc) so this comparison is a permanent `false` today — written
    // generically (not as a dead-code-eliding special case) so a future
    // session re-enabling the lever only needs to change the constant.
    // REV4: the whole-image path carries no rev4 hooks — if the lever is
    // ever re-enabled, a rev4-enabled call must NOT take it (the cells
    // would silently stay zero). `r4a.is_none()` keeps that honest.
    #[allow(clippy::absurd_extreme_comparisons)]
    if height <= STRIP_BYPASS_HEIGHT && r4a.is_none() {
        return compute_channel_scale_v2_whole(src, dst, width, height, toggles, scratch, out);
    }

    let max_wide_h = STRIP_ROWS + 2 * HALO_P;
    assert!(
        width * max_wide_h <= scratch.mu1.len(),
        "scratch buffers must be sized for the largest strip+halo"
    );

    // REV4: the bleed-mask workspace for this cell — allocated once,
    // filled per strip (chroma cells only; `bleed_y` is `Some` exactly
    // there).
    let mut bleed_work = r4a
        .as_ref()
        .and_then(|a| a.bleed_y.map(|_| BleedMaskWork::sized(width, STRIP_ROWS)));

    let mut dense = DenseAccum::default();
    let mut grad = GradientAccum::default();

    let mut y0 = 0usize;
    while y0 < height {
        let strip_h = STRIP_ROWS.min(height - y0);
        let wide_h = strip_h + 2 * HALO_P;
        let n_wide = width * wide_h;

        // --- Gather this strip's halo-padded input (real image data,
        //     mirror-reflected only within BLUR_RADIUS*2 of the TRUE
        //     image top/bottom — see `gather_strip_halo`/`reflect_101`)
        //     into the scratch's own halo buffers (reused across strips,
        //     calls, and — via `V2Scratch` — across pairs). ---
        gather_strip_halo(
            src,
            width,
            height,
            y0,
            wide_h,
            HALO_P,
            &mut scratch.src_wide[..n_wide],
        );
        gather_strip_halo(
            dst,
            width,
            height,
            y0,
            wide_h,
            HALO_P,
            &mut scratch.dst_wide[..n_wide],
        );

        // --- Blur pass on the strip+halo buffer only (§A.15's actual
        //     memory-traffic reduction: `wide_h` is O(STRIP_ROWS), not
        //     O(height)). With cached reference moments the mu1 V-blur +
        //     activity chain drop out of the per-pair cost entirely —
        //     EXCEPT when Rev4 C1 is on: `gridblk_strip_wide` reads the
        //     WIDE activity window including halo rows, and activity is
        //     a sliding-window V-blur whose f32 running-sum history is
        //     local to each strip geometry — `act_full` rows are
        //     bit-identical only where that row was a STRIP row of the
        //     replayed walk, so gathering it would feed divergent
        //     values at halo positions. The full strip pass reproduces
        //     the streaming path's wide window bitwise. ---
        let need_wide_activity = r4a.as_ref().is_some_and(|a| a.gridblk_period >= 2);
        if moments.is_some() && !need_wide_activity {
            run_blur_pass_strip_cached_ref(width, wide_h, scratch);
        } else {
            run_blur_pass_strip(width, wide_h, scratch);
        }

        // --- Slice down to the strip's own real rows (buffer-local
        //     offset HALO_P, matching `gather_strip_halo`'s convention:
        //     local row `HALO_P + k` is global row `y0 + k`). Cached
        //     reference moments are FULL-plane buffers, indexed by global
        //     row (`y0 * width`), not halo-local offset. ---
        let off = HALO_P * width;
        let strip_n = width * strip_h;
        let out_base = y0 * width;
        let src_strip = &scratch.src_wide[off..off + strip_n];
        let dst_strip = &scratch.dst_wide[off..off + strip_n];
        let (mu1_strip, activity_strip) = match moments {
            Some((mu1_full, act_full)) => (
                &mu1_full[out_base..out_base + strip_n],
                &act_full[out_base..out_base + strip_n],
            ),
            None => (
                &scratch.mu1[off..off + strip_n],
                &scratch.activity[off..off + strip_n],
            ),
        };
        let mu2_strip = &scratch.mu2[off..off + strip_n];
        let ssq_strip = &scratch.ssq[off..off + strip_n];
        let s12_strip = &scratch.s12[off..off + strip_n];

        // REV4 C4: dst-Y dilated-edge mask for this strip — gathered with
        // a ±2-row halo from the same-scale luma plane, built by the SAME
        // `dst_y_edge_mask_strip` the streaming walk uses (parity input).
        let bleed_mask: Option<&[f32]> =
            match (r4a.as_ref().and_then(|a| a.bleed_y), bleed_work.as_mut()) {
                (Some(dy), Some(mw)) => {
                    gather_strip_halo(
                        dy,
                        width,
                        height,
                        y0,
                        strip_h + 4,
                        2,
                        &mut mw.dh[..width * (strip_h + 4)],
                    );
                    dst_y_edge_mask_strip(width, y0, strip_h, height, mw);
                    Some(&mw.mask[..strip_n])
                }
                _ => None,
            };

        // REV4 hook bundle — identical construction to `stream_phase_b`'s
        // (`Rev4Dense`/`Rev4Grad` feed the same `rev4_*_pixel` helpers),
        // so a cell's accumulator is bitwise-identical across walks.
        let r4d = match r4a.as_mut() {
            Some(a) if a.tailhist || a.flat => Some(Rev4Dense {
                tail: a.tailhist.then_some(&mut a.cell.tail),
                flat: a.flat.then_some(&mut a.cell.flat),
                edges: tail_edges(),
            }),
            _ => None,
        };
        let strip_dense = dense_block_kernel(
            src_strip,
            dst_strip,
            mu1_strip,
            mu2_strip,
            ssq_strip,
            s12_strip,
            activity_strip,
            width,
            strip_h,
            toggles.transducer_bank,
            r4d,
        );
        dense.accumulate(&strip_dense);

        let bank_on = r4a.as_ref().is_some_and(|a| a.gmsbank);
        if toggles.gradient_features || bank_on {
            // Gradient needs src/dst at [y0-1, y0+strip_h+1) — 1-row
            // halo, comfortably inside the HALO_P(=10)-row buffer we
            // already gathered. Buffer-local offset HALO_P-1.
            let g_off = (HALO_P - 1) * width;
            let g_n = width * (strip_h + 2);
            let src_g = &scratch.src_wide[g_off..g_off + g_n];
            let dst_g = &scratch.dst_wide[g_off..g_off + g_n];
            let r4g = match r4a.as_mut() {
                Some(a) if a.ringbasis || bleed_mask.is_some() || a.gmsbank => Some(Rev4Grad {
                    ring: a.ringbasis.then_some(&mut a.cell.ring),
                    bleed: bleed_mask.map(|m| (m, &mut a.cell.bleed)),
                    bank: a.bank.rows(y0 * width, (y0 + strip_h) * width),
                }),
                _ => None,
            };
            let strip_grad = gradient_block_kernel(
                src_g,
                dst_g,
                activity_strip,
                width,
                strip_h,
                None,
                None,
                r4g,
                bank_on,
            );
            grad.accumulate(&strip_grad);
        }

        // REV4 C1: boundary `ẽ` planes off the just-gathered wide buffers.
        // `scratch.activity` is strip-computed over the same wide window
        // in every blur path that can reach this point (the cached-moments
        // branch is gated off above when C1 is on).
        if let Some(a) = r4a.as_mut()
            && a.gridblk_period >= 2
        {
            gridblk_strip_wide(
                &scratch.src_wide[..n_wide],
                &scratch.dst_wide[..n_wide],
                &scratch.activity[..n_wide],
                width,
                y0,
                strip_h,
                height,
                a.gridblk_period,
                &mut a.cell.grid,
            );
        }

        y0 += strip_h;
    }

    let sum_blockiness = if toggles.blockiness {
        blockiness_sparse(src, dst, width, height)
    } else {
        0.0
    };

    if let Some(a) = r4a.as_mut()
        && a.gmsbank
    {
        a.cell.bank = grad.bank;
        a.cell.chroma = grad.chroma;
    }
    finish_channel_scale(&dense, &grad, sum_blockiness, n, out)
}

/// Phase-6 (§A.16 lever B) whole-image path: below [`STRIP_BYPASS_HEIGHT`],
/// skip the strip loop's halo-gather-and-reblur machinery entirely and run
/// the blur pass + dense/gradient kernels directly on the real image —
/// mirroring phase-4's non-strip design (`compute_channel_scale_v2` at
/// commit `7696b62a`, before phase-5's strip-tiling rewrite), reusing the
/// SAME kernels (`dense_block_kernel`/`gradient_block_kernel`/
/// `blockiness_sparse`/`finish_channel_scale`) rather than re-deriving
/// them: `run_blur_pass`'s `crate::blur` primitives already do their own
/// boundary reflection (identical to how phase-4 called them directly on
/// `src`/`dst`), so no halo copy is needed for the blur+dense pass at all.
/// The one exception is `gradient_block_kernel_generic`'s NEW (phase-5)
/// halo contract, which needs a 1-row-padded buffer — built via
/// `gather_strip_halo(halo=1)`, the SAME helper the strip loop already
/// uses for its own (larger) halo, just with `halo=1` and sourced directly
/// from the real image instead of a strip's local window.
fn compute_channel_scale_v2_whole(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    toggles: V2NewFeatureToggles,
    scratch: &mut ScratchV2Strip,
    out: &mut [f64],
) -> (f64, f64) {
    let n = width * height;
    debug_assert!(
        n <= scratch.mu1.len(),
        "scratch buffers must be sized for at least STRIP_BYPASS_HEIGHT rows \
         (see the scratch allocation in compute_v2_features_impl_with_toggles)"
    );

    run_blur_pass(src, dst, width, height, scratch);

    let mu1 = &scratch.mu1[..n];
    let mu2 = &scratch.mu2[..n];
    let ssq = &scratch.ssq[..n];
    let s12 = &scratch.s12[..n];
    let activity = &scratch.activity[..n];

    let dense = dense_block_kernel(
        src,
        dst,
        mu1,
        mu2,
        ssq,
        s12,
        activity,
        width,
        height,
        toggles.transducer_bank,
        None,
    );

    let grad = if toggles.gradient_features {
        // gradient_block_kernel_generic's contract needs src_h/dst_h sized
        // width*(height+2) with a 1-row halo on each side (§A.15.1) — the
        // ONLY extra copy this bypass path pays (O(width), not
        // O(width*height)), built with the same `gather_strip_halo` the
        // strip loop uses, `halo=1` instead of `HALO_P`.
        let mut src_g = vec![0.0f32; width * (height + 2)];
        let mut dst_g = vec![0.0f32; width * (height + 2)];
        gather_strip_halo(src, width, height, 0, height + 2, 1, &mut src_g);
        gather_strip_halo(dst, width, height, 0, height + 2, 1, &mut dst_g);
        gradient_block_kernel(
            &src_g, &dst_g, activity, width, height, None, None, None, false,
        )
    } else {
        GradientAccum::default()
    };

    let sum_blockiness = if toggles.blockiness {
        blockiness_sparse(src, dst, width, height)
    } else {
        0.0
    };

    finish_channel_scale(&dense, &grad, sum_blockiness, n, out)
}

/// Shared tail for both [`compute_channel_scale_v2`] (strip path) and
/// [`compute_channel_scale_v2_whole`] (phase-6 §A.16 lever B bypass path):
/// converts raw-moment/weighted-pool accumulators into the final 29
/// per-channel-scale feature values. Identical for both callers — the
/// accumulators (`DenseAccum`/`GradientAccum`) are plain sums, so N
/// strips' partials (via `.accumulate()`) or one whole-image pass produce
/// the exact same totals feeding this function; extracting it once avoids
/// a second copy of this ~50-line conversion drifting from the original
/// (the no-duplication policy's concern — see CLAUDE.md "NO DUPLICATE
/// IMPLEMENTATIONS").
#[inline]
pub(crate) fn finish_channel_scale(
    dense: &DenseAccum,
    grad: &GradientAccum,
    sum_blockiness: f64,
    n: usize,
    out: &mut [f64],
) -> (f64, f64) {
    finish_channel_scale_planned(dense, grad, sum_blockiness, n, out, false)
}

/// Same ordered finalizer with a private declared-input restriction.
#[inline]
pub(crate) fn finish_channel_scale_planned(
    dense: &DenseAccum,
    grad: &GradientAccum,
    sum_blockiness: f64,
    n: usize,
    out: &mut [f64],
    omit_pjnd_fragility: bool,
) -> (f64, f64) {
    finish_channel_scale_at_revision(
        dense,
        grad,
        sum_blockiness,
        n,
        out,
        omit_pjnd_fragility,
        crate::ssim_form::active_revision(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_channel_scale_at_revision(
    dense: &DenseAccum,
    grad: &GradientAccum,
    sum_blockiness: f64,
    n: usize,
    out: &mut [f64],
    omit_pjnd_fragility: bool,
    revision: FormulaRevision,
) -> (f64, f64) {
    let n_f = n as f64;

    // --- Raw-moment -> central-moment conversion (§A.14 module doc):
    //     replaces `OnlineMoments::finish()`'s Terriberry-tracked
    //     mean/M2/M4 with the standard identity from the SIMD-accumulated
    //     raw power sums. Bit-for-bit different from Terriberry (different
    //     floating-point operation order), numerically equivalent within
    //     the file's 5e-4 relative tolerance (verified by
    //     `simd_matches_scalar_within_tolerance`). ---
    let mean_d = dense.sum_d / n_f;
    let (m2, m4) = if dense.central.n > 0.0 {
        (dense.central.m2 / n_f, dense.central.m4 / n_f)
    } else {
        let raw2 = dense.sum_d2 / n_f;
        let raw3 = dense.sum_d3 / n_f;
        let raw4 = dense.sum_d4 / n_f;
        (
            (raw2 - mean_d * mean_d).max(0.0),
            (raw4 - 4.0 * mean_d * raw3 + 6.0 * mean_d * mean_d * raw2 - 3.0 * mean_d.powi(4))
                .max(0.0),
        )
    };
    let dev2 = m2.sqrt();
    let dev4 = m4.quarter_root(crate::det_math::RootForm::at_revision(Some(revision)));

    // Phase-6 (§A.16): defensive bounds clamp on the FINAL feature value
    // (called once per channel-scale, O(1) not O(pixels) — cheap). Not
    // load-bearing for the shipped algebra-only reformulation (native `/`
    // stays correctly-rounded, so the pre-phase-6 per-pixel `.max(zero)`
    // guards were already sufficient) — kept as low-cost belt-and-braces
    // hardening for every feature whose per-pixel formula was touched this
    // phase (ssim_d_local_v / bounded_sim_v / bounded_excess_v /
    // bounded_excess_pair_v / saturate_v / pjnd_transducer_v), and because
    // a `.recip()`-based variant of these formulas WAS tried and measured
    // safe-but-neutral (§A.16, module doc above) — this clamp is what that
    // path would have needed had it shipped, left in place in case a
    // future session revisits recip on different hardware.
    #[inline]
    fn clamp01(v: f64) -> f64 {
        v.clamp(0.0, 1.0)
    }
    #[inline]
    fn clamp02(v: f64) -> f64 {
        v.clamp(0.0, 2.0)
    }

    out[idx::SSIM_MEAN] = clamp02(mean_d);
    out[idx::SSIM_DEV2] = clamp02(dev2);
    out[idx::SSIM_DEV4] = clamp02(dev4);
    out[idx::ART] = clamp01(dense.sum_art / n_f);
    out[idx::DET] = clamp01(dense.sum_det / n_f);
    out[idx::MSE] = clamp01(dense.sum_mse / n_f);
    out[idx::HF_GAIN] = clamp01(dense.sum_hf_gain / n_f);
    out[idx::HF_LOSS] = clamp01(dense.sum_hf_loss / n_f);
    out[idx::HF_MAG_LOSS] = clamp01(dense.sum_hf_mag_loss / n_f);
    out[idx::SSIM_SOFT_PEAK] = clamp02(dense.ws_peak_ssim.finish());
    out[idx::ART_SOFT_PEAK] = clamp02(dense.ws_peak_art.finish());
    out[idx::DET_SOFT_PEAK] = clamp02(dense.ws_peak_det.finish());
    out[idx::MASKED_SSIM] = clamp02(dense.ws_mask_ssim.finish());
    out[idx::MASKED_ART] = clamp02(dense.ws_mask_art.finish());
    out[idx::MASKED_DET] = clamp02(dense.ws_mask_det.finish());
    out[idx::MASKED_MSE] = clamp02(dense.ws_mask_mse.finish());
    out[idx::IW_SSIM] = clamp02(dense.ws_iw_ssim.finish());
    out[idx::IW_ART] = clamp02(dense.ws_iw_art.finish());
    out[idx::IW_DET] = clamp02(dense.ws_iw_det.finish());
    out[idx::IW_MSE] = clamp02(dense.ws_iw_mse.finish());
    out[idx::PJND_TRANSDUCER] = clamp01(dense.sum_pjnd / n_f);
    if !omit_pjnd_fragility {
        out[idx::PJND_FRAGILITY] = 1.0 - saturate(grad.sum_grad_src / n_f, C_PJND_GRAD);
    }
    out[idx::GMS] = grad.sum_gms / n_f;
    out[idx::PJND_TRANSDUCER_LOW_K] = clamp01(dense.sum_pjnd_lo / n_f);
    out[idx::PJND_TRANSDUCER_HIGH_K] = clamp01(dense.sum_pjnd_hi / n_f);
    out[idx::BLOCKINESS] = sum_blockiness / n_f;
    out[idx::RINGING] = grad.sum_ringing / n_f;
    out[idx::BANDING] = grad.sum_banding / n_f;
    out[idx::EDGE_WIDTH_CHANGE] = 0.0; // filled in by the caller (needs the adjacent scale)

    (grad.sum_grad_src / n_f, grad.sum_grad_dst / n_f)
}

// ============================================================================
// Per-pixel diffmap generator (v2 core)
//
// Spatializes the SIMPLE-MEAN families of the dense/gradient/blockiness
// kernels above into an explicit per-pixel map, so
// `mean_over_pixels(map) == Σ_local weights[local] * feature_local` for
// every family whose FEATURE is, by construction, the unweighted mean of a
// well-defined per-pixel quantity. Scalar-only (no `#[magetypes]`/`incant!`
// dispatch) — this is the correctness-first reference implementation; a
// SIMD-accelerated sibling is a natural follow-up once this is proven right
// (see `tests::v2_diffmap_block_pool_matches_features`, the gate this
// module's own doc requires before anything here is trusted).
// ============================================================================

/// Per-pixel diffmap for ONE channel-scale, built from the SAME closed-form
/// formulas [`dense_block_kernel_generic`]/[`gradient_block_kernel_generic`]/
/// [`blockiness_sparse`] use to accumulate their pooled means — just written
/// out per-pixel instead of summed. For every SUPPORTED `local` (see below),
/// `dm[i]` carries `weights[local] * M_local(pixel i)`, where `M_local` is
/// the exact per-pixel quantity whose unweighted mean over all
/// `width*height` pixels equals [`finish_channel_scale`]'s `feature_local`
/// (BEFORE that function's mean-level `clamp01`/`clamp02` — see "Clamping"
/// below). Returns `dm[i] = Σ_local weights[local] · M_local(pixel i)`.
///
/// Reuses [`gather_strip_halo`]/[`run_blur_pass`]/[`ScratchV2Strip`] — the
/// SAME strip/halo machinery [`compute_channel_scale_v2`] (the function the
/// real v2 pipeline actually calls) uses to build `mu1`/`mu2`/`ssq`/`s12`/
/// `activity` — so this function's per-pixel inputs are IDENTICAL to the
/// ones the pooled features are computed from, not a fresh (and possibly
/// boundary-divergent) recomputation.
///
/// # Supported (spatialized) families
///
/// - [`idx::SSIM_MEAN`] — `ssim_d_local(mu1, mu2, s12, ssq)` itself (the
///   C1/C2-bounded SSIM dissimilarity), unchanged from
///   [`dense_block_kernel_generic`]'s scalar tail.
/// - [`idx::ART`] / [`idx::DET`] — the edge-artifact/detail split:
///   `1 - bounded_sim(|src-mu1|, |dst-mu2|, C_EDGE)` routed to ART when
///   `|dst-mu2| > |src-mu1|`, to DET when `<`, else neither fires (both 0).
/// - [`idx::MSE`] — `saturate((src-dst)^2, C_MSE)`.
/// - [`idx::HF_GAIN`] / [`idx::HF_LOSS`] — the shared-denominator
///   `bounded_excess_pair(hf_dst^2, hf_src^2, C_HF)` (`hf_x = x - mu_x`).
/// - [`idx::HF_MAG_LOSS`] — `bounded_excess(|hf_src|, |hf_dst|, C_HF)`.
/// - [`idx::PJND_TRANSDUCER`] — the CORE masking transducer only
///   (`k = K_PJND_MASK`): `pjnd_transducer(|src-dst|, activity, K_PJND_MASK,
///   C_PJND_CLAMP)`.
/// - [`idx::GMS`] — `1 - bounded_sim(grad_src_mag, grad_dst_mag, C_GMS)`
///   (central-difference gradient magnitude), using
///   [`gradient_block_kernel_generic`]'s own boundary convention exactly:
///   x-neighbors clamp-to-edge (`saturating_sub`/`.min`), y-neighbors
///   whole-sample-mirror via [`reflect_101`] (sourced from the strip's
///   halo-gathered buffer, so a strip seam or the true image edge both
///   resolve identically to the pooled feature's own computation).
/// - [`idx::BLOCKINESS`] — sparse 8-pixel-lattice step-energy comparison
///   (`bounded_excess(step_dst, step_src, C_BLOCK)`); nonzero ONLY at
///   `x % BLOCK_LATTICE == 0` / `y % BLOCK_LATTICE == 0` positions, matching
///   [`blockiness_sparse`]'s own sparsity — a lattice-corner pixel gets BOTH
///   the vertical- and horizontal-boundary contributions added, exactly as
///   [`blockiness_sparse`]'s two passes both add into the same running sum.
/// - [`idx::RINGING`] — `saturate(|src-dst|,C_RING_ERR) *
///   saturate(activity,C_ACTIVITY) * (1 - saturate(grad_src_mag,C_RING_EDGE))`.
/// - [`idx::BANDING`] — `bounded_excess(grad_dst_mag, grad_src_mag,
///   C_BAND_DST) * (1 - saturate(grad_src_mag, C_BAND_SRC))`.
/// - [`idx::MASKED_SSIM`] / [`idx::MASKED_ART`] / [`idx::MASKED_DET`] /
///   [`idx::MASKED_MSE`] and [`idx::IW_SSIM`] / [`idx::IW_ART`] /
///   [`idx::IW_DET`] / [`idx::IW_MSE`] — the weighted-pool families
///   `Σ(w·v)/Σw`. Additive weighted means, so spatializable as `n·w·v/Σw`:
///   the per-pixel numerators (`mask_w·v` with `mask_w = 1 - sat(act)`,
///   `iw_w·v` with `iw_w = sat(act) + IW_WEIGHT_FLOOR`) and the global `Σw`
///   are accumulated in the strip pass, then normalized by `n/Σw` after — so
///   `mean_over_pixels(map)` equals the pooled ratio exactly (block-pool test).
///   ART/DET route by `diff_dst`≷`diff_src`, matching the pooled kernel.
///
/// # Excluded families (weight ignored, regardless of what the caller passes)
///
/// - [`idx::SSIM_DEV2`] / [`idx::SSIM_DEV4`] — Terriberry/GMSD-style
///   deviation-FROM-THE-MEAN moments (`dev2 = sqrt(M2/n)`,
///   `M2/n = E[d^2] - E[d]^2`): a NONLINEAR function of the whole-image
///   mean, not itself the mean of any single per-pixel quantity.
/// - [`idx::SSIM_SOFT_PEAK`], [`idx::ART_SOFT_PEAK`], [`idx::DET_SOFT_PEAK`]
///   — the saliency-weighted-pool families (`Σw·v/Σw` with `w = saliency`).
///   Additive like masked/iw and spatializable the same way once the
///   per-pixel saliency weight is threaded through — a natural follow-up
///   (masked/iw landed the pattern; soft-peak reuses it with `w = sal`).
/// - [`idx::PJND_FRAGILITY`] — `1 - saturate(mean(grad_src_mag),
///   C_PJND_GRAD)`: reference-only, and a NONLINEAR function of the mean
///   gradient magnitude (not the mean of a per-pixel fragility value).
/// - [`idx::PJND_TRANSDUCER_LOW_K`] / [`idx::PJND_TRANSDUCER_HIGH_K`] — the
///   masking-transducer BANK (extra `k` values). Mechanically these use the
///   exact same [`pjnd_transducer`] per-pixel formula as the core transducer
///   above (swap `K_PJND_MASK` for `K_PJND_MASK_LOW`/`K_PJND_MASK_HIGH`) and
///   would spatialize identically — out of scope for THIS core per the task
///   brief, not for any correctness reason. Natural follow-up.
/// - [`idx::EDGE_WIDTH_CHANGE`] — the one genuinely scale-level (not
///   per-pixel) feature in this set: it compares THIS scale's mean gradient
///   magnitude against the ADJACENT (coarser) scale's, filled in by
///   [`compute_v2_features_impl_with_toggles`] only once that scale exists.
///   No single channel-scale's pixels carry enough information to
///   spatialize it.
///
/// # Clamping
///
/// [`finish_channel_scale`] applies a MEAN-level `clamp01`/`clamp02` to
/// several of the supported families (`out[idx::X] = clamp01(sum_x/n)`) —
/// i.e. the clamp bounds the AVERAGE, not each pixel. This function emits
/// the per-pixel quantity WITHOUT that mean-level clamp (`ssim_d_local`'s
/// own internal `.max(0.0)` guard, and the various `bounded_*`/`saturate`
/// helpers' own per-pixel bounds, still apply — those are part of the
/// formula itself, not the mean-level wrapper). So `mean_over_pixels(dm)`
/// equals `Σ weights[local] * feature_local` exactly when every supported
/// family's mean lands inside its clamp range on the given input (the
/// common case for a real, non-degenerate image pair, and the case this
/// module's own test suite always constructs — see `TOL`'s neighboring
/// comment on FP-reassociation tolerance). Callers relying on bit-parity
/// with heavily clamp-saturating input should expect the mean-level clamp
/// to introduce a (bounded, one-directional) divergence between the pooled
/// feature and this map's mean.
///
/// # Panics
///
/// If `src.len() != width*height` or `dst.len() != width*height`.
pub(crate) fn compute_v2_diffmap_channel_scale(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    weights: &[f64; FEATURES_PER_CHANNEL_V2_TOTAL],
) -> Vec<f32> {
    // Revision 3 (issue #61): the `s12` plane is the direct error moment, so
    // the v2 SSIM signal must read it as such. Hoisted once per call;
    // loop-invariant, so every call site below unswitches on it.
    let direct = crate::ssim_form::active_revision() >= crate::feature_defs::FormulaRevision::Rev3;
    let stable = crate::ssim_form::active_revision().stable_forms();
    let ring_hp = process_ring_high_pass();
    let n = width * height;
    assert_eq!(src.len(), n, "src plane length must be width*height");
    assert_eq!(dst.len(), n, "dst plane length must be width*height");

    let mut out = vec![0.0f32; n];
    if width == 0 || height == 0 {
        return out;
    }

    // Weighted-pool families (masked/iw) are additive weighted means
    // `Σ(w·v)/Σw` — spatializable as `n·w·v/Σw` once the global `Σw` is known.
    // Accumulate per-pixel weighted numerators + the global `Σw` during the
    // strip pass; normalize into `out` after (so the map's mean equals the
    // pooled weighted-mean feature, the block-pool identity the test gates).
    // soft-peak needs the saliency weight (a follow-up); dev/fragility/
    // edge_width are non-additive and stay excluded.
    let has_mask = weights[idx::MASKED_SSIM] != 0.0
        || weights[idx::MASKED_ART] != 0.0
        || weights[idx::MASKED_DET] != 0.0
        || weights[idx::MASKED_MSE] != 0.0;
    let has_iw = weights[idx::IW_SSIM] != 0.0
        || weights[idx::IW_ART] != 0.0
        || weights[idx::IW_DET] != 0.0
        || weights[idx::IW_MSE] != 0.0;
    let mut masked_num = if has_mask {
        vec![0.0f32; n]
    } else {
        Vec::new()
    };
    let mut iw_num = if has_iw { vec![0.0f32; n] } else { Vec::new() };
    let mut sum_mask_w = 0.0f64;
    let mut sum_iw_w = 0.0f64;

    let max_wide_h = STRIP_ROWS + 2 * HALO_P;
    let mut scratch = ScratchV2Strip::new(width * max_wide_h);
    let mut src_wide = vec![0.0f32; width * max_wide_h];
    let mut dst_wide = vec![0.0f32; width * max_wide_h];

    let mut y0 = 0usize;
    while y0 < height {
        let strip_h = STRIP_ROWS.min(height - y0);
        let wide_h = strip_h + 2 * HALO_P;
        let n_wide = width * wide_h;

        // Same halo-gather + blur-pass as `compute_channel_scale_v2`'s
        // strip loop (§A.15) — reused verbatim so this function's
        // mu1/mu2/ssq/s12/activity are IDENTICAL to the ones the pooled
        // features are computed from, not a fresh (and possibly
        // boundary-divergent) recomputation.
        gather_strip_halo(
            src,
            width,
            height,
            y0,
            wide_h,
            HALO_P,
            &mut src_wide[..n_wide],
        );
        gather_strip_halo(
            dst,
            width,
            height,
            y0,
            wide_h,
            HALO_P,
            &mut dst_wide[..n_wide],
        );
        run_blur_pass(
            &src_wide[..n_wide],
            &dst_wide[..n_wide],
            width,
            wide_h,
            &mut scratch,
        );

        let off = HALO_P * width;
        let strip_n = width * strip_h;
        let src_strip = &src_wide[off..off + strip_n];
        let dst_strip = &dst_wide[off..off + strip_n];
        let mu1_strip = &scratch.mu1[off..off + strip_n];
        let mu2_strip = &scratch.mu2[off..off + strip_n];
        let ssq_strip = &scratch.ssq[off..off + strip_n];
        let s12_strip = &scratch.s12[off..off + strip_n];
        let activity_strip = &scratch.activity[off..off + strip_n];

        // 1-row-halo view for the gradient family — same slice convention
        // as `compute_channel_scale_v2`'s call into `gradient_block_kernel`.
        let g_off = (HALO_P - 1) * width;
        let g_n = width * (strip_h + 2);
        let src_g = &src_wide[g_off..g_off + g_n];
        let dst_g = &dst_wide[g_off..g_off + g_n];

        for y_local in 0..strip_h {
            let gy = y0 + y_local;
            let i_strip = y_local * width;
            let i_self = (y_local + 1) * width; // src_g/dst_g: +1 halo row
            let i_up = y_local * width; // src_g/dst_g: row y_local-1 (halo-relative)
            let i_down = (y_local + 2) * width; // src_g/dst_g: row y_local+1 (halo-relative)

            for x in 0..width {
                let i_local = i_strip + x;
                let i_global = gy * width + x;

                let s = src_strip[i_local] as f64;
                let dd = dst_strip[i_local] as f64;
                let m1 = mu1_strip[i_local] as f64;
                let m2 = mu2_strip[i_local] as f64;
                let act = activity_strip[i_local] as f64;

                let mut acc = 0.0f64;

                // --- Dense (always-on) family — bit-for-bit the same
                //     formulas as `dense_block_kernel_generic`'s scalar
                //     tail. ---
                let d = ssim_d_local(
                    m1,
                    m2,
                    s12_strip[i_local] as f64,
                    ssq_strip[i_local] as f64,
                    direct,
                    stable,
                );
                acc += weights[idx::SSIM_MEAN] * d;

                let diff_src = (s - m1).abs();
                let diff_dst = (dd - m2).abs();
                let edge_dissim = bounded_dissim(diff_src, diff_dst, C_EDGE, stable);
                if diff_dst > diff_src {
                    acc += weights[idx::ART] * edge_dissim;
                } else if diff_dst < diff_src {
                    acc += weights[idx::DET] * edge_dissim;
                }

                let raw_diff = s - dd;
                let raw_sq_err = raw_diff * raw_diff;
                let mse_v = saturate(raw_sq_err, C_MSE);
                acc += weights[idx::MSE] * mse_v;

                // Masked / IW weighted-pool numerators (normalized after the
                // pass by n/Σw). Same `mask_w`/`iw_w` weights and ART/DET
                // routing the pooled kernel uses: `mask_w = 1 - sat(act)`,
                // `iw_w = sat(act) + FLOOR`.
                if has_mask || has_iw {
                    let sat_act = saturate(act, C_ACTIVITY);
                    let (edge_art, edge_det) = if diff_dst > diff_src {
                        (edge_dissim, 0.0)
                    } else if diff_dst < diff_src {
                        (0.0, edge_dissim)
                    } else {
                        (0.0, 0.0)
                    };
                    if has_mask {
                        let mask_w = 1.0 - sat_act;
                        sum_mask_w += mask_w;
                        masked_num[i_global] += (mask_w
                            * (weights[idx::MASKED_SSIM] * d
                                + weights[idx::MASKED_ART] * edge_art
                                + weights[idx::MASKED_DET] * edge_det
                                + weights[idx::MASKED_MSE] * mse_v))
                            as f32;
                    }
                    if has_iw {
                        let iw_w = sat_act + IW_WEIGHT_FLOOR;
                        sum_iw_w += iw_w;
                        iw_num[i_global] += (iw_w
                            * (weights[idx::IW_SSIM] * d
                                + weights[idx::IW_ART] * edge_art
                                + weights[idx::IW_DET] * edge_det
                                + weights[idx::IW_MSE] * mse_v))
                            as f32;
                    }
                }

                let hf_src = s - m1;
                let hf_dst = dd - m2;
                let hf_src_sq = hf_src * hf_src;
                let hf_dst_sq = hf_dst * hf_dst;
                let (hf_gain_i, hf_loss_i) = bounded_excess_pair(hf_dst_sq, hf_src_sq, C_HF);
                acc += weights[idx::HF_GAIN] * hf_gain_i;
                acc += weights[idx::HF_LOSS] * hf_loss_i;
                acc += weights[idx::HF_MAG_LOSS] * bounded_excess(hf_src.abs(), hf_dst.abs(), C_HF);

                let raw_abs_err = raw_diff.abs();
                acc += weights[idx::PJND_TRANSDUCER]
                    * pjnd_transducer(raw_abs_err, act, K_PJND_MASK, C_PJND_CLAMP);

                // --- Gradient family — bit-for-bit the same formulas as
                //     `gradient_block_kernel_generic`'s `scalar_pixel`
                //     closure (GMS/ringing/banding), same x-clamp / y-mirror
                //     boundary convention. ---
                let xl = x.saturating_sub(1);
                let xr = (x + 1).min(width - 1);
                let sxl = src_g[i_self + xl] as f64;
                let sxr = src_g[i_self + xr] as f64;
                let syu = src_g[i_up + x] as f64;
                let syd = src_g[i_down + x] as f64;
                let dxl = dst_g[i_self + xl] as f64;
                let dxr = dst_g[i_self + xr] as f64;
                let dyu = dst_g[i_up + x] as f64;
                let dyd = dst_g[i_down + x] as f64;

                let gx_src = sxr - sxl;
                let gy_src = syd - syu;
                let grad_src_mag = (gx_src * gx_src + gy_src * gy_src).sqrt();
                let gx_dst = dxr - dxl;
                let gy_dst = dyd - dyu;
                let grad_dst_mag = (gx_dst * gx_dst + gy_dst * gy_dst).sqrt();

                acc +=
                    weights[idx::GMS] * (bounded_dissim(grad_src_mag, grad_dst_mag, C_GMS, stable));

                let ring_err = if ring_hp {
                    ((s - m1) - (dd - m2)).abs()
                } else {
                    raw_abs_err
                };
                let err_b = saturate(ring_err, C_RING_ERR);
                let act_b = saturate(act, C_ACTIVITY);
                let edge_r = saturate(grad_src_mag, C_RING_EDGE);
                acc += weights[idx::RINGING] * (err_b * act_b * (1.0 - edge_r));

                let edge_excess = bounded_excess(grad_dst_mag, grad_src_mag, C_BAND_DST);
                let src_smooth_b = 1.0 - saturate(grad_src_mag, C_BAND_SRC);
                acc += weights[idx::BANDING] * (edge_excess * src_smooth_b);

                out[i_global] = acc as f32;
            }
        }

        y0 += strip_h;
    }

    // --- Blockiness — sparse 8-pixel-lattice pass over the FULL
    //     (un-strip-tiled) planes, matching `blockiness_sparse`'s own loop
    //     exactly (including the lattice-corner double-add: a pixel with
    //     both x%8==0 and y%8==0 gets a contribution from EACH loop, same
    //     as `blockiness_sparse`'s single running `sum` does). ---
    let w_block = weights[idx::BLOCKINESS];
    let mut x = BLOCK_LATTICE;
    while x < width {
        for y in 0..height {
            let i = y * width + x;
            let step_dst = (dst[i] as f64 - dst[i - 1] as f64).abs();
            let step_src = (src[i] as f64 - src[i - 1] as f64).abs();
            out[i] += (w_block * bounded_excess(step_dst, step_src, C_BLOCK)) as f32;
        }
        x += BLOCK_LATTICE;
    }
    let mut y = BLOCK_LATTICE;
    while y < height {
        for x in 0..width {
            let i = y * width + x;
            let i_up = i - width;
            let step_dst = (dst[i] as f64 - dst[i_up] as f64).abs();
            let step_src = (src[i] as f64 - src[i_up] as f64).abs();
            out[i] += (w_block * bounded_excess(step_dst, step_src, C_BLOCK)) as f32;
        }
        y += BLOCK_LATTICE;
    }

    // Normalize the weighted-pool numerators: feature = Σ(w·v)/Σw, so the
    // per-pixel map is (n/Σw)·Σ_family(weight·w·v). Then mean_over_pixels ==
    // Σ_family weight·(Σ w·v / Σw) == Σ_family weight·feature, matching the
    // block-pool identity for the masked/iw families. No-op if a pool's Σw≈0.
    if has_mask && sum_mask_w > 1e-12 {
        let f = n as f64 / sum_mask_w;
        for i in 0..n {
            out[i] += (f * masked_num[i] as f64) as f32;
        }
    }
    if has_iw && sum_iw_w > 1e-12 {
        let f = n as f64 / sum_iw_w;
        for i in 0..n {
            out[i] += (f * iw_num[i] as f64) as f32;
        }
    }

    out
}

/// Full-image v2 diffmap: the per-pixel contribution of a bake's v2 block to
/// its scalar, at scale-0 (full) resolution. This is the builder the runtime
/// steering fold (`compute_with_ref_and_diffmap`'s `ModelSensitivity` path)
/// needs to make the DEPLOYED diffmap read v2 — currently it folds only the
/// ≤372 v1 block, so `M3` (deployed-map coherence) is ~0 for every v2 bake
/// (and every v1 MLP bake) while `M2` (the gradient ceiling) is ≈1.0.
///
/// `s_v2` is the bake's gradient w.r.t. each v2 feature, laid out exactly as
/// [`FeatureViewV2`] indexes them: `s_v2[scale*3*TOTAL + ch*TOTAL + local]`,
/// length `n_scales*3*`[`FEATURES_PER_CHANNEL_V2_TOTAL`]. The caller slices it
/// out of the full bake gradient (`s[372..]` for a frozen-v1++v2 720 bake).
///
/// Builds the SAME pyramid the feature extractor uses (via
/// [`prepare_v2_reference_impl`] for both images — bit-identical planes), runs
/// [`compute_v2_diffmap_channel_scale`] per channel-scale with that scale's
/// weight slice, and nearest-upsamples each coarser scale's map to full res
/// (mean-preserving on even dims) accumulating into one map.
///
/// # Block-pool identity (the correctness gate)
///
/// `mean_over_pixels(full) == Σ_{scale,ch,local∈SUPPORTED} s_v2[..]·feature`,
/// to FP-reassociation tolerance, because each channel-scale map already
/// satisfies that identity ([`compute_v2_diffmap_channel_scale`]'s own gate)
/// and nearest-upsampling preserves a plane's mean. See
/// `tests::v2_diffmap_full_block_pool_matches_features`.
///
/// Exposed to consumers via the public [`crate::Zensim::compute_v2_diffmap`]
/// (the G-STEER harness + encoder closed loop call it there). The `s_v2`
/// gradient it takes is the `s[372..]` tail of a v1-372 ++ v2 bake's full
/// gradient; the v1 contribution comes from the ordinary `ModelSensitivity`
/// diffmap path, summed with this.
pub(crate) fn compute_v2_diffmap_full(
    reference: &impl ImageSource,
    distorted: &impl ImageSource,
    s_v2: &[f64],
    max_pixels: Option<usize>,
    parallel: bool,
) -> Result<Vec<f32>, ZensimError> {
    let rprep = prepare_v2_reference_impl(reference, max_pixels, parallel, false)?;
    let dprep = prepare_v2_reference_impl(distorted, max_pixels, parallel, false)?;
    let n_scales = rprep.scales.len().min(dprep.scales.len());
    let per_scale = 3 * FEATURES_PER_CHANNEL_V2_TOTAL;
    debug_assert!(
        s_v2.len() >= n_scales * per_scale,
        "s_v2 too short: {} < {}",
        s_v2.len(),
        n_scales * per_scale
    );
    let (w0, h0) = (rprep.scales[0].1, rprep.scales[0].2);
    let mut out = vec![0.0f32; w0 * h0];
    for scale in 0..n_scales {
        let (ref planes_r, ws, hs) = rprep.scales[scale];
        let planes_d = &dprep.scales[scale].0;
        for ch in 0..3 {
            // This channel-scale's weight slice → the [TOTAL] array the core takes.
            let mut weights = [0.0f64; FEATURES_PER_CHANNEL_V2_TOTAL];
            let base = scale * per_scale + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
            weights.copy_from_slice(&s_v2[base..base + FEATURES_PER_CHANNEL_V2_TOTAL]);
            if weights.iter().all(|&w| w == 0.0) {
                continue;
            }
            let cs =
                compute_v2_diffmap_channel_scale(&planes_r[ch], &planes_d[ch], ws, hs, &weights);
            // Nearest-upsample (ws,hs)->(w0,h0), accumulate. Mean-preserving when
            // w0/h0 are integer multiples of ws/hs (the pyramid's even case).
            for y in 0..h0 {
                let sy = if hs == h0 { y } else { y * hs / h0 };
                let row_c = sy * ws;
                let row_o = y * w0;
                for x in 0..w0 {
                    let sx = if ws == w0 { x } else { x * ws / w0 };
                    out[row_o + x] += cs[row_c + sx];
                }
            }
        }
    }
    Ok(out)
}

// ============================================================================
// Top-level entry point
// ============================================================================

/// Precomputed reference-side state for the v2 extraction: the
/// (reflect-padded, XYB-converted) reference planes at every pyramid
/// scale. Build once per reference via
/// [`crate::Zensim::prepare_v2_reference`], then score any number of
/// distorted variants against it with
/// [`crate::Zensim::compute_v2_features_with_ref`] — each call skips the
/// reference-side decode-adjacent work (reflect-pad copy, RGB→XYB
/// conversion, and the 3-level downscale chain) that the pair entry point
/// would otherwise redo per pair.
///
/// Bit-exactness contract: the planes are built by the SAME functions in
/// the SAME order as the pair path (`convert_source_to_xyb` +
/// `downscale_2x` per level), so `compute_v2_features_with_ref(prepare(r),
/// d)` produces features **bit-identical** to `compute_v2_features(r, d)`
/// — guarded by `prepared_ref_bit_identical_to_pair_path` in this file's
/// tests. Memory: ~`5.33 * w * h` f32s (≈21 MB at 1 MP, ≈85 MB at 4 MP).
pub struct V2PreparedReference {
    /// Per pyramid scale: 3 XYB planes at that scale's exact
    /// `(width, height)` — same layout the pair path materializes.
    scales: Vec<([Vec<f32>; 3], usize, usize)>,
    /// Optional cached reference-side blur moments (`mu1` = blurred ref,
    /// `activity` = blurred |ref − mu1|), per scale per channel, at full
    /// plane size. Filled by REPLAYING the strip walk (same halo gather,
    /// same sliding-sum chain starts), so the cached values are
    /// bit-identical to what the kernels would recompute — see
    /// [`fill_ref_moments`]. When present, each per-pair call skips the
    /// mu1 V-blur and the entire activity chain (abs-diff + 2-pass blur)
    /// per channel-scale.
    moments: Option<Vec<[V2RefMoments; 3]>>,
    /// Original (pre reflect-pad) source dimensions; distorted images are
    /// validated against these, exactly like `validate_pair`.
    orig_width: usize,
    orig_height: usize,
}

/// One channel-scale's cached reference-side blur moments (full-plane).
struct V2RefMoments {
    mu1: Vec<f32>,
    activity: Vec<f32>,
}

impl V2PreparedReference {
    /// Reference width in pixels (unpadded). Distorted images scored
    /// against this reference must match exactly.
    pub fn width(&self) -> usize {
        self.orig_width
    }
    /// Reference height in pixels (unpadded).
    pub fn height(&self) -> usize {
        self.orig_height
    }
    /// Whether reference-side blur moments are cached (see
    /// [`crate::Zensim::prepare_v2_reference_with_moments`]).
    pub fn has_cached_moments(&self) -> bool {
        self.moments.is_some()
    }
}

/// Build a [`V2PreparedReference`]: validate, reflect-pad if sub-64px
/// (identity-skip otherwise — the pad is a pure copy at ≥64px and is NOT
/// performed), convert to XYB once, and materialize all
/// [`crate::NUM_SCALES`] pyramid levels.
pub(crate) fn prepare_v2_reference_impl(
    source: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    cache_moments: bool,
) -> Result<V2PreparedReference, ZensimError> {
    prepare_v2_reference_impl_inner(source, max_pixels, parallel, cache_moments)
}

fn prepare_v2_reference_impl_inner(
    source: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    cache_moments: bool,
) -> Result<V2PreparedReference, ZensimError> {
    if source.width() == 0 || source.height() == 0 {
        return Err(ZensimError::ImageTooSmall);
    }
    crate::metric::reject_hdr_input(source)?;
    crate::metric::check_within_max_pixels(source.width(), source.height(), max_pixels)?;

    let (orig_width, orig_height) = (source.width(), source.height());
    let scales = if orig_width < crate::metric::MIN_PYRAMID_DIM
        || orig_height < crate::metric::MIN_PYRAMID_DIM
    {
        build_v2_ref_scales(&crate::metric::reflect_pad_to_min(source), parallel)
    } else {
        build_v2_ref_scales(source, parallel)
    };
    let moments = cache_moments.then(|| fill_ref_moments(&scales));
    Ok(V2PreparedReference {
        scales,
        moments,
        orig_width,
        orig_height,
    })
}

/// Materialize the reference XYB pyramid: scale 0 from
/// `convert_source_to_xyb`, each further level via `downscale_2x_into`
/// (bit-identical arithmetic to the pair path's `downscale_2x_inplace` —
/// both compute `(a + b + c + d) * 0.25` per output element in the same
/// order; the in-place/out-of-place split changes only where the result
/// is stored).
fn build_v2_ref_scales(
    img: &impl ImageSource,
    parallel: bool,
) -> Vec<([Vec<f32>; 3], usize, usize)> {
    assert!(
        !crate::feature_v2_stream::linear_pyramid(),
        "ZENSIM_PYRAMID=linear covers only the streaming and block producers"
    );
    let mut width = img.width();
    let mut height = img.height();
    let mut scales: Vec<([Vec<f32>; 3], usize, usize)> = Vec::with_capacity(crate::NUM_SCALES);
    scales.push((
        crate::streaming::convert_source_to_xyb(
            img,
            width,
            parallel,
            crate::ssim_form::active_revision(),
        ),
        width,
        height,
    ));
    for _ in 1..crate::NUM_SCALES {
        let new_w = width / 2;
        let new_h = height / 2;
        let mut next: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0f32; new_w * new_h]);
        {
            let (prev_planes, _, _) = scales.last().expect("scale 0 pushed above");
            for ch in 0..3 {
                crate::blur::downscale_2x_into(
                    &prev_planes[ch],
                    width,
                    &mut next[ch],
                    new_w,
                    new_h,
                );
            }
        }
        scales.push((next, new_w, new_h));
        width = new_w;
        height = new_h;
    }
    scales
}

/// Compute the v2 "bounded" feature vector for a source/distorted pair.
///
/// Reuses [`crate::metric::validate_pair`] and
/// [`crate::metric::reflect_pad_to_min`] — identical input handling to
/// v1's entry points. Always uses [`crate::NUM_SCALES`] (4) scales and
/// `BLUR_RADIUS=5`, matching v1's defaults.
///
/// Called by [`crate::Zensim::compute_v2_features`] (the public entry
/// point); kept as a free function so it can be unit-tested without a
/// `Zensim` instance.
pub(crate) fn compute_v2_features_impl(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
) -> Result<ZensimV2Result, ZensimError> {
    compute_v2_features_impl_with_toggles(
        source,
        distorted,
        max_pixels,
        parallel,
        V2NewFeatureToggles::default(),
    )
}

/// Like [`compute_v2_features_impl`], but with explicit control over which
/// phase-2 new-feature GROUPS are computed — used by
/// `Zensim::compute_v2_features_with_toggles` (per-group marginal-cost
/// measurement, `docs/FEATURE_V2_SPEC_2026-07-18.md` §A.12) and by the
/// `v2_speed_baseline`/`v2_stage_profile` benches.
///
/// Ref-reuse pass: this is now a thin composition of
/// [`prepare_v2_reference_impl`] + [`compute_v2_features_with_ref_impl`]
/// — the pair path and the prepared path share ONE scale-walk owner, so
/// they cannot drift. `validate_pair`/`check_within_max_pixels` run first
/// to preserve the original error precedence (dimension mismatch before
/// any per-side rejection).
pub(crate) fn compute_v2_features_impl_with_toggles(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    toggles: V2NewFeatureToggles,
) -> Result<ZensimV2Result, ZensimError> {
    if toggles.formula_revision >= FormulaRevision::Rev5 {
        validate_wide_revision(toggles)?;
        validate_raw_rev5_request(toggles)?;
    }
    crate::metric::validate_pair(source, distorted)?;
    crate::metric::check_within_max_pixels(source.width(), source.height(), max_pixels)?;
    let prepared = prepare_v2_reference_impl(source, max_pixels, parallel, false)?;
    let mut scratch = V2Scratch::new();
    compute_v2_features_with_ref_impl(
        &prepared,
        distorted,
        max_pixels,
        parallel,
        toggles,
        &mut scratch,
    )
}

/// Score one distorted image against a prepared reference — the shared
/// scale-walk owner for BOTH the pair entry point (which prepares
/// inline) and the batch/reuse entry points. Only distorted-side work
/// happens per call: conditional reflect-pad (skipped at ≥64px, where it
/// is an identity copy), XYB conversion, per-scale kernels, and the
/// distorted pyramid downscale. Reference planes are read from
/// `prepared`.
pub(crate) fn compute_v2_features_with_ref_impl(
    prepared: &V2PreparedReference,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    toggles: V2NewFeatureToggles,
    scratch: &mut V2Scratch,
) -> Result<ZensimV2Result, ZensimError> {
    // Rev5 serves the bounded-layout API through the canonical fold owner;
    // preserve its 348-slot layout rather than running the historical materialized kernels.
    if crate::ssim_form::effective_revision(toggles.formula_revision) >= FormulaRevision::Rev5 {
        validate_wide_revision(toggles)?;
        validate_raw_rev5_request(toggles)?;
        crate::metric::reject_hdr_input(distorted)?;
        crate::metric::check_within_max_pixels(distorted.width(), distorted.height(), max_pixels)?;
        if (distorted.width(), distorted.height()) != (prepared.orig_width, prepared.orig_height) {
            return Err(ZensimError::DimensionMismatch);
        }
        let run = |image: &_, scratch: &mut V2Scratch| {
            foldapp_streaming_walk(
                image,
                image,
                parallel,
                toggles,
                crate::feature_v2_stream::FrontEnd::Sdr,
                scratch,
                FoldWalkExtras {
                    ref_planes: Some(&prepared.scales),
                    ..Default::default()
                },
            )
        };
        // The two image types differ on the padded branch, so invoke the
        // generic owner directly there.
        let folded = if distorted.width() < crate::metric::MIN_PYRAMID_DIM
            || distorted.height() < crate::metric::MIN_PYRAMID_DIM
        {
            let image = crate::metric::reflect_pad_to_min(distorted);
            foldapp_streaming_walk(
                &image,
                &image,
                parallel,
                toggles,
                crate::feature_v2_stream::FrontEnd::Sdr,
                scratch,
                FoldWalkExtras {
                    ref_planes: Some(&prepared.scales),
                    ..Default::default()
                },
            )
        } else {
            run(distorted, scratch)
        };
        return Ok(ZensimV2Result {
            features: folded.features()[372..720].to_vec(),
            n_scales: crate::NUM_SCALES,
            v1_pools: V1PoolsMode::Off,
            regime: FeatureRegime::V2Bounded,
        });
    }
    // REV4SERVE: the V2Bounded walk is NOT the canonical Rev4 owner — its
    // v1 moments come from this walk's own H-blurred planes (parity-gated
    // against the frozen v1 path, never byte-frozen; measured divergent
    // from `research::extract` at Rev4). The canonical equivalents are the
    // folded entries (`Zensim::compute_folded720_features_streaming`). A
    // Rev4 request refuses rather than serve tier-dispatched arithmetic.
    // Below Rev4 nothing changes.
    if crate::ssim_form::effective_revision(toggles.formula_revision)
        >= crate::feature_defs::FormulaRevision::Rev4
    {
        return Err(ZensimError::ModelLoadFailed {
            reason: "the V2Bounded buffered walk is not canonical at formula revision 4: use the folded entries (compute_folded720_features_streaming) for Rev4 extraction",
        });
    }
    compute_v2_features_with_ref_impl_inner(
        prepared, distorted, max_pixels, parallel, toggles, scratch, None,
    )
}

/// Folded-720 pair entry (see [`FeatureRegime::Folded720`] for the
/// layout) — routes to the STREAMING walk (C5 switchover, 2026-07-26,
/// user-approved: the materialized folded walk and its reference cache
/// are deleted; `benchmarks/streaming_foldapp_gates_2026-07-26.md`).
pub(crate) fn compute_folded720_impl_with_toggles(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    toggles: V2NewFeatureToggles,
) -> Result<ZensimV2Result, ZensimError> {
    validate_wide_revision(toggles)?;
    let mut scratch = V2Scratch::new();
    compute_folded720_streaming_impl(
        source,
        distorted,
        max_pixels,
        parallel,
        toggles,
        &mut scratch,
        None,
    )
}

/// Folded-720-plus-append pair entry (see
/// [`FeatureRegime::Folded720Append`]): the folded 720 layout with the
/// f720+ append block (204 slots) emitted after it. Forces
/// `toggles.append_block = true`. Routes to the STREAMING walk (C5).
pub(crate) fn compute_folded720_append_impl(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    mut toggles: V2NewFeatureToggles,
) -> Result<ZensimV2Result, ZensimError> {
    toggles.append_block = true;
    compute_folded720_impl_with_toggles(source, distorted, max_pixels, parallel, toggles)
}

// ============================================================================
// STREAMING folded-720[+append] walk (C2 of
// docs/STREAMING_FOLDAPP_C0_DESIGN_2026-07-26.md).
//
// Consumes `feature_v2_stream::StripPlaneProducer`'s kernel strips instead
// of materialized full-image pyramids: per emitted strip, phase A fills
// each channel's wide windows (`gather_strip_halo` semantics from the
// rolling planes) + runs the SAME blur pass the materialized walk runs on
// its moments-free path + the P2 strip-tiled bs2 chain; phase B runs the
// SAME dense/gradient/append/fold kernels and folds their per-strip
// accumulators into per-(scale, channel) totals in kernel-strip order —
// the exact f64 op sequence of the materialized walk. Finalize replays
// the materialized walk's per-scale epilogue (finish + fold finalize +
// edge-width chain) on the merged accumulators. Output is BITWISE equal
// to the materialized pair path (test-gated below); no full-image f32
// plane exists on either side at any point.
// ============================================================================

/// append2 per-walk parameters, derived once from the route + toggles:
/// BANDVIS δ half-points (route-specific EMPIRICAL constants) and whether
/// the HDR highlight bins are live.
#[derive(Clone, Copy)]
struct Append2Params {
    bv_delta_lo: f32,
    bv_delta_hi: f32,
    hl_bins: bool,
    /// BANDVIS dst self-mask live ([`V2NewFeatureToggles::append2_dst_activity`]):
    /// phase A computes the Y-channel dst-activity plane, phase B routes it
    /// into the BANDVIS-dstact kernel instantiation. Route-independent.
    dst_activity: bool,
}

/// Per-channel accumulator state for the streaming walk: per-scale totals
/// for every family the walk pools, merged strictly in kernel-strip order.
struct StreamChannelAccums {
    /// Rev5 retaining walks keep the exact coarse strip kernel results.
    retained_strips: Option<Vec<Vec<(DenseAccum, GradientAccum)>>>,
    dense: Vec<DenseAccum>,
    grad: Vec<GradientAccum>,
    app: Vec<AppendAccum>,
    v1: Vec<V1BasicSums>,
    /// Row-ordered blockiness partials `(sum_v, sum_h)` (P1 canonical).
    /// featacc: [`SumVar`] slots — `Seq` unless a measurement mode is set
    /// (f64-native elements: `c32` resolves to `Seq`, i.e. production
    /// accumulation, per the f64-element rule).
    block: Vec<(crate::featcanon::SumVar, crate::featcanon::SumVar)>,
    /// CSFW weighted-pool partials (Y channel + `csfw_block` only —
    /// untouched zeros otherwise).
    csfw: Vec<CsfwAccum>,
    /// REV4 per-scale cells (`gridblk`/`ringbasis`/`tailhist` plus this
    /// channel's share of `arttype`) — written by the dense/gradient hook
    /// bundles and the gridblk strip kernel, all inside `stream_phase_b`.
    /// Untouched zeros when every rev4 compute flag is off.
    rev4: Vec<Rev4CellAccum>,
    /// Band-local planes for the v1 pool replay (`v1_pools`); empty
    /// (never allocated) when the toggle is off.
    /// One scratch per band SLOT within a strip (`STRIP_ROWS /
    /// V1_BAND_ROWS`), so the band-parallel path reuses buffers instead of
    /// allocating ~580 KB per worker per strip — which measured as a net
    /// LOSS before this existed.
    pool_scratch: Vec<FoldPoolScratch>,
}

/// A bounded, reusable owned halo window and its independent v2 strip result.
/// Slots contain no full-image planes. Workers never touch running totals.
#[cfg(feature = "threads")]
struct Rev5StripJob {
    info: crate::feature_v2_stream::StripInfo,
    ch: usize,
    scratch: ScratchV2Strip,
    sized_for: usize,
    result: StreamChannelAccums,
    bands: [V1BasicSums; V1_BANDS_PER_STRIP],
}

// Private policy, chosen by measurement (benchmarks/rev5perf5_budget_design_2026-10-09.md):
// 256 MiB costs 0.7-21.9 % against uncapped at >=4096-wide inputs on 16-32 threads
// (128 MiB: 15.5-46.0 %), at 0.48-0.72x uncapped peak RSS. This bounds owned queue
// payload, not allocator overhead or the whole score.
#[cfg(feature = "threads")]
const REV5_JOB_BUDGET_BYTES: usize = 256 * 1024 * 1024;

#[cfg(feature = "threads")]
fn rev5_job_bytes(max_n: usize) -> Option<usize> {
    use core::mem::size_of;
    // Eleven float planes, the slot itself, and every fixed-length result
    // vector allocated by StreamChannelAccums::new(NUM_SCALES, 1). Queued
    // jobs never allocate the pool scratch's float planes.
    let cell = size_of::<DenseAccum>()
        + size_of::<GradientAccum>()
        + size_of::<AppendAccum>()
        + size_of::<V1BasicSums>()
        + size_of::<(crate::featcanon::SumVar, crate::featcanon::SumVar)>()
        + size_of::<CsfwAccum>()
        + size_of::<Rev4CellAccum>();
    let fixed = size_of::<Rev5StripJob>()
        + crate::NUM_SCALES * cell
        + size_of::<FoldPoolScratch>()
        + crate::NUM_SCALES * size_of::<Vec<(DenseAccum, GradientAccum)>>()
        + 3 * size_of::<(DenseAccum, GradientAccum)>()
        + size_of::<Option<RetainedRows<'static>>>();
    max_n
        .checked_mul(11)?
        .checked_mul(size_of::<f32>())?
        .checked_add(fixed)
}

// Below this many slots the walk keeps the original route. That route runs a
// strip's three channels in parallel (`fuse_channels`) and each channel's v1
// bands in parallel (`band_parallel`); a queued job runs its bands serially.
// Which is faster at a given slot count is measured, not assumed: see
// benchmarks/rev5perf5_budget_design_2026-10-09.md. Both routes produce
// identical bits; the floor changes scheduling only.
#[cfg(feature = "threads")]
const REV5_MIN_JOB_SLOTS: usize = 3;

#[cfg(feature = "threads")]
fn rev5_job_limit(max_n: usize, threads: usize, budget: usize) -> usize {
    let limit = rev5_job_bytes(max_n).map_or(0, |bytes| threads.min(16).min(budget / bytes));
    if limit < REV5_MIN_JOB_SLOTS { 0 } else { limit }
}

#[cfg(feature = "threads")]
fn rev5_prepare_jobs(jobs: &mut Vec<Rev5StripJob>, limit: usize, max_n: usize) {
    // Drop excess slots BEFORE growing parent scratch or new job planes.
    // A narrower subsequent pair cannot keep oversized cached planes outside
    // the current request's byte accounting.
    jobs.truncate(limit);
    for job in jobs.iter_mut() {
        if job.sized_for > max_n {
            job.scratch = ScratchV2Strip::new(0);
            job.sized_for = 0;
        }
    }
    if jobs.capacity() != limit {
        let old = core::mem::take(jobs);
        *jobs = Vec::with_capacity(limit);
        jobs.extend(old);
    }
}

#[cfg(feature = "threads")]
impl Rev5StripJob {
    fn new(info: crate::feature_v2_stream::StripInfo, ch: usize) -> Self {
        Self {
            info,
            ch,
            scratch: ScratchV2Strip::new(0),
            sized_for: 0,
            result: StreamChannelAccums::at_revision(crate::NUM_SCALES, 1, FormulaRevision::Rev5),
            bands: core::array::from_fn(|_| V1BasicSums::at_revision(FormulaRevision::Rev5)),
        }
    }

    /// Take the strip and size the scratch for its window.
    fn prepare(&mut self, info: crate::feature_v2_stream::StripInfo, ch: usize) {
        self.info = info;
        self.ch = ch;
        let n = info.plane_w * info.wide_h();
        if n > self.sized_for {
            // Release the previous allocation before constructing its larger
            // replacement, so both plane sets never coexist inside the cap.
            self.scratch = ScratchV2Strip::new(0);
            self.sized_for = 0;
            self.scratch = rev5_job_scratch(n);
            self.sized_for = n;
        }
    }

    fn gather<S: ImageSource, D: ImageSource>(
        &mut self,
        producer: &crate::feature_v2_stream::StripPlaneProducer<'_, S, D>,
        info: crate::feature_v2_stream::StripInfo,
        ch: usize,
        borrow_source: bool,
    ) {
        use crate::feature_v2_stream::Side;
        self.prepare(info, ch);
        let n = info.plane_w * info.wide_h();
        for (side, out) in [
            (Side::Source, &mut self.scratch.src_wide),
            (Side::Distorted, &mut self.scratch.dst_wide),
        ] {
            if side == Side::Source && borrow_source {
                continue;
            }
            if let Some(window) = producer.wide_window(side, ch, &info) {
                out[..n].copy_from_slice(window);
            } else {
                producer.fill_wide(side, ch, &info, &mut out[..n]);
            }
        }
    }
}

/// Rev5's basic hook reads vertical planes, so horizontal moments are dead
/// after vertical blur. Reuse two of those planes as activity-chain temps;
/// no append bs2 or separate activity temps are retained in queued jobs.
#[cfg(feature = "threads")]
fn rev5_job_scratch(n: usize) -> ScratchV2Strip {
    let mut scratch = ScratchV2Strip::new_for(n, StripPlaneNeeds::NONE);
    for plane in [
        &mut scratch.mu1_h,
        &mut scratch.mu2_h,
        &mut scratch.ssq_h,
        &mut scratch.s12_h,
        &mut scratch.mu1,
        &mut scratch.mu2,
        &mut scratch.ssq,
        &mut scratch.s12,
        &mut scratch.activity,
    ] {
        plane.resize(n, 0.0);
    }
    crate::fold_timing::work(crate::fold_timing::Work::ScratchZeroElements, 9 * n);
    scratch
}

#[cfg(feature = "threads")]
fn rev5_cached_window<'a>(
    reference: Option<&'a [crate::streaming::XybPyramidLevel]>,
    info: &crate::feature_v2_stream::StripInfo,
    ch: usize,
) -> Option<&'a [f32]> {
    if !info.interior() {
        return None;
    }
    let (planes, width, height) = &reference?[info.scale];
    debug_assert_eq!((*width, *height), (info.plane_w, info.plane_h));
    Some(&planes[ch][(info.y0 - HALO_P) * width..(info.y0 + info.strip_h + HALO_P) * width])
}

#[cfg(feature = "threads")]
fn rev5_job_blur(
    info: &crate::feature_v2_stream::StripInfo,
    scratch: &mut ScratchV2Strip,
    revision: FormulaRevision,
    source: Option<&[f32]>,
) {
    let n = info.plane_w * info.wide_h();
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
        activity,
        ..
    } = scratch;
    let src = source.unwrap_or(&src_wide[..n]);
    let start = crate::fold_timing::start();
    // The moments in one pass where both blurs run their per-output trees:
    // the horizontal planes then live a column panel at a time in the first
    // rows of `mu1_h`..`s12_h` instead of filling them. Only pools of eight or
    // more threads take it, where the jobs' planes contend for the cache
    // (paired, B at revision 7, 1507×2048 / 1024²: eight threads −3.6% /
    // +0.1%, sixteen on eight cores −22.0% / −16.4%, 32 −10.8% / −3.1%); four
    // threads ran +2.4% / −3.0% and one thread 1–2% slower on the panels'
    // shorter row runs, so smaller pools and the serial walk keep two passes.
    let fused = rayon::current_num_threads() >= 8
        && crate::blur::fused_blur_hv_ssim_at_revision(
            src,
            &dst_wide[..n],
            &mut mu1[..n],
            &mut mu2[..n],
            &mut ssq[..n],
            &mut s12[..n],
            info.plane_w,
            info.wide_h(),
            BLUR_RADIUS,
            revision,
            [
                &mut mu1_h[..n],
                &mut mu2_h[..n],
                &mut ssq_h[..n],
                &mut s12_h[..n],
            ],
        );
    if !fused {
        fused_blur_h_ssim_banded(
            src,
            &dst_wide[..n],
            &mut mu1_h[..n],
            &mut mu2_h[..n],
            &mut ssq_h[..n],
            &mut s12_h[..n],
            info.plane_w,
            info.wide_h(),
            false,
            revision,
        );
    }
    crate::fold_timing::stop(start, crate::fold_timing::Phase::BlurHWall, info.scale);
    let start = crate::fold_timing::start();
    if !fused {
        for (horizontal, vertical) in [
            (&mu1_h[..n], &mut mu1[..n]),
            (&mu2_h[..n], &mut mu2[..n]),
            (&ssq_h[..n], &mut ssq[..n]),
            (&s12_h[..n], &mut s12[..n]),
        ] {
            crate::blur::box_blur_v_from_copy_at_revision(
                horizontal,
                vertical,
                info.plane_w,
                info.wide_h(),
                BLUR_RADIUS,
                revision,
            );
        }
    }
    // The activity plane in one pass where its blurs run their per-output
    // trees, `mu1_h` and `mu2_h` holding a panel's rows. Pools of sixteen or
    // more threads only (paired against the fused moments alone, B at
    // revision 7, 1507×2048 / 1024²: sixteen on eight cores −13.5% / −4.9%,
    // 32 −8.1% / −7.9%, but eight +1.4% / +1.0%).
    if !(fused
        && rayon::current_num_threads() >= 16
        && crate::blur::activity_at_revision(
            src,
            &mu1[..n],
            &mut activity[..n],
            info.plane_w,
            info.wide_h(),
            BLUR_RADIUS,
            revision,
            &mut mu1_h[..n],
            &mut mu2_h[..n],
        ))
    {
        crate::simd_ops::abs_diff_into(src, &mu1[..n], &mut mu1_h[..n]);
        crate::blur::box_blur_1pass_at_revision(
            &mu1_h[..n],
            &mut activity[..n],
            &mut mu2_h[..n],
            info.plane_w,
            info.wide_h(),
            BLUR_RADIUS,
            revision,
        );
    }
    crate::fold_timing::work(crate::fold_timing::Work::ActivityChain, 1);
    crate::fold_timing::work(crate::fold_timing::Work::V2Cell, 1);
    crate::fold_timing::stop(start, crate::fold_timing::Phase::PhaseAV2Planes, info.scale);
}

/// One queued strip job's work: phase A into its own planes, phase B into its
/// own partials (and its retained rows when `output` is given). Runs on any
/// thread; [`rev5_merge_job`] adds the partials to the walk's totals.
#[cfg(feature = "threads")]
#[allow(clippy::too_many_arguments)]
fn rev5_execute_job(
    job: &mut Rev5StripJob,
    mut output: Option<RetainedRows<'_>>,
    compute: &ComputeSet,
    toggles: V2NewFeatureToggles,
    fold_v1: bool,
    retain: bool,
    reference: Option<&[crate::streaming::XybPyramidLevel]>,
) {
    let busy = crate::fold_timing::start();
    let info = &job.info;
    let scale = info.scale;
    let local = compute.at_scale(scale);
    let toggles = V2NewFeatureToggles {
        v1_pools: local.v1_pools,
        v1_only: false,
        gradient_features: local.gradient,
        // This running sum must be accumulated in original pixel order.
        blockiness: false,
        ..toggles
    };
    job.result.dense[scale] = DenseAccum::default();
    job.result.grad[scale] = GradientAccum::default();
    if retain && (1..=3).contains(&scale) {
        let strips = job
            .result
            .retained_strips
            .get_or_insert_with(|| (0..crate::NUM_SCALES).map(|_| Vec::new()).collect());
        strips[scale].clear();
        if strips[scale].capacity() == 0 {
            strips[scale].reserve_exact(1);
        }
    } else if !retain {
        job.result.retained_strips = None;
    }
    let cached_source = rev5_cached_window(reference, info, job.ch);
    // Revision 7: the strip's blockiness lane partials, here in the
    // parallel job (both branches below); the merge after the batch adds
    // them in strip order.
    if local.blockiness && local.formula_revision.row_lanes() {
        let n = info.plane_w * info.wide_h();
        let src = cached_source.unwrap_or(&job.scratch.src_wide[..n]);
        let mode = crate::featcanon::mode(local.formula_revision);
        let (v, h) = &mut job.result.block[scale];
        *v = crate::featcanon::SumVar::f64_elems(mode);
        *h = crate::featcanon::SumVar::f64_elems(mode);
        blockiness_sparse_strip_wide(
            src,
            &job.scratch.dst_wide[..n],
            info.plane_w,
            info.y0,
            info.strip_h,
            local.formula_revision,
            v,
            h,
        );
    }
    if let Some(output) = output.as_mut()
        && retained_core_admits(&local)
    {
        rev5_blur_into_retained(info, &mut job.scratch, [cached_source, None], output);
        let n = info.plane_w * info.wide_h();
        let src = cached_source.unwrap_or(&job.scratch.src_wide[..n]);
        rev5_retained_phase_b(
            info,
            [src, &job.scratch.dst_wide[..n]],
            output.moment_rows(),
            RetainedCoreWork {
                toggles,
                fold_v1,
                free: local.free_work(job.ch, scale),
            },
            &mut job.result,
            &mut job.bands,
        );
        crate::fold_timing::stop(busy, crate::fold_timing::Phase::BBusy, scale);
        return;
    }
    rev5_job_blur(
        info,
        &mut job.scratch,
        compute.formula_revision,
        cached_source,
    );
    let n = info.plane_w * info.wide_h();
    let src = cached_source.unwrap_or(&job.scratch.src_wide[..n]);
    if let Some(output) = &mut output {
        output.copy(info, &job.scratch, src, &job.scratch.dst_wide[..n], false);
    }
    stream_phase_b(
        &job.scratch,
        src,
        &job.scratch.dst_wide[..n],
        info,
        toggles,
        false,
        true,
        false,
        false,
        false,
        &[],
        None,
        None,
        None,
        local.free_work(job.ch, scale),
        local.rev4_work(scale, job.ch, None, None),
        &mut job.result,
    );
    if fold_v1 {
        let raw = [src, &job.scratch.dst_wide[..n]];
        let vertical = FoldHSource::Vertical([
            &job.scratch.mu1[..n],
            &job.scratch.mu2[..n],
            &job.scratch.ssq[..n],
            &job.scratch.s12[..n],
        ]);
        let mut b0 = info.y0;
        for band in &mut job.bands[..info.strip_h.div_ceil(V1_BAND_ROWS)] {
            *band = V1BasicSums::at_revision(FormulaRevision::Rev5);
            b0 = fold_v1_one_band(
                b0,
                info.plane_w,
                info.y0 + info.strip_h,
                info.y0,
                HALO_P,
                info.plane_h,
                vertical,
                raw,
                band,
                None,
                local.free_work(job.ch, scale),
            );
        }
    }
    crate::fold_timing::stop(busy, crate::fold_timing::Phase::BBusy, scale);
}

/// Add one executed strip job's partials to the walk's totals. Jobs merge
/// once each, in the producer's order; do not reorder or tree-reduce them.
#[cfg(feature = "threads")]
fn rev5_merge_job(
    job: &Rev5StripJob,
    compute: &ComputeSet,
    fold_v1: bool,
    totals: &mut [StreamChannelAccums; 3],
    reference: Option<&[crate::streaming::XybPyramidLevel]>,
) {
    let info = &job.info;
    let scale = info.scale;
    let local = compute.at_scale(scale);
    let total = &mut totals[job.ch];
    total.dense[scale].accumulate(&job.result.dense[scale]);
    total.grad[scale].accumulate(&job.result.grad[scale]);
    if let Some(target) = &mut total.retained_strips
        && let Some(strips) = &job.result.retained_strips
    {
        target[scale].extend_from_slice(&strips[scale]);
    }
    if fold_v1 {
        // Preserve each canonical 32-row band addition, never merge a
        // pre-reduced strip sum into the running basic accumulator.
        for band in &job.bands[..info.strip_h.div_ceil(V1_BAND_ROWS)] {
            total.v1[scale].merge(band);
        }
    }
    if local.blockiness {
        let (sum_v, sum_h) = &mut total.block[scale];
        if local.formula_revision.row_lanes() {
            let (v, h) = &job.result.block[scale];
            sum_v.merge_from(v);
            sum_h.merge_from(h);
        } else {
            // Revisions 4–6 keep one running sum: the strip's windows, here.
            let n = info.plane_w * info.wide_h();
            let src =
                rev5_cached_window(reference, info, job.ch).unwrap_or(&job.scratch.src_wide[..n]);
            let dst = &job.scratch.dst_wide[..n];
            blockiness_sparse_strip_wide(
                src,
                dst,
                info.plane_w,
                info.y0,
                info.strip_h,
                local.formula_revision,
                sum_v,
                sum_h,
            );
        }
    }
}

/// Queued strip jobs without batch barriers, for walks that retain nothing
/// and read no cached reference: each gathered job is spawned on the rayon
/// pool at once, finished jobs come back over a channel and merge in the
/// producer's order ([`rev5_merge_job`], the batch runner's merge), and the
/// producer waits only when `limit` jobs are in flight. The merge order is the
/// batch runner's, so every value has the batch runner's bits; the producer
/// gathers and converts while the pool computes.
#[cfg(feature = "threads")]
struct Rev5Pipeline {
    compute: ComputeSet,
    fold_v1: bool,
    limit: usize,
    free: Vec<Rev5StripJob>,
    in_flight: usize,
    next_seq: usize,
    merge_seq: usize,
    done: std::collections::BTreeMap<usize, Rev5StripJob>,
    tx: std::sync::mpsc::Sender<(usize, std::thread::Result<Rev5StripJob>)>,
    rx: std::sync::mpsc::Receiver<(usize, std::thread::Result<Rev5StripJob>)>,
}

#[cfg(feature = "threads")]
impl Rev5Pipeline {
    fn new(compute: ComputeSet, fold_v1: bool, limit: usize, free: Vec<Rev5StripJob>) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        Self {
            compute,
            fold_v1,
            limit,
            free,
            in_flight: 0,
            next_seq: 0,
            merge_seq: 0,
            done: std::collections::BTreeMap::new(),
            tx,
            rx,
        }
    }

    /// A job slot for the next strip, merging finished jobs (oldest first)
    /// until fewer than `limit` are in flight.
    fn slot(
        &mut self,
        info: crate::feature_v2_stream::StripInfo,
        ch: usize,
        totals: &mut [StreamChannelAccums; 3],
    ) -> Rev5StripJob {
        while self.in_flight >= self.limit {
            self.wait(totals);
        }
        self.free
            .pop()
            .unwrap_or_else(|| Rev5StripJob::new(info, ch))
    }

    /// Spawn a gathered job.
    fn submit(&mut self, mut job: Rev5StripJob, toggles: V2NewFeatureToggles) {
        let (seq, tx, compute, fold_v1) =
            (self.next_seq, self.tx.clone(), self.compute, self.fold_v1);
        self.next_seq += 1;
        self.in_flight += 1;
        rayon::spawn(move || {
            let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                rev5_execute_job(&mut job, None, &compute, toggles, fold_v1, false, None);
            }));
            // The walk owns the receiver until every spawned job reports.
            let _ = tx.send((seq, run.map(|()| job)));
        });
    }

    /// Take at least one finished job and merge every job that is next in
    /// order. A rayon worker waiting here runs pending pool work instead of
    /// blocking, so a walk called from inside the pool cannot starve its own
    /// jobs.
    fn wait(&mut self, totals: &mut [StreamChannelAccums; 3]) {
        let mut got = None;
        while got.is_none() {
            match self.rx.try_recv() {
                Ok(m) => got = Some(m),
                Err(std::sync::mpsc::TryRecvError::Empty) => match rayon::yield_now() {
                    Some(rayon::Yield::Executed) => {}
                    Some(rayon::Yield::Idle) => std::thread::yield_now(),
                    None => got = Some(self.rx.recv().expect("strip job channel")),
                },
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    unreachable!("the pipeline holds a sender")
                }
            }
        }
        let mut next = got;
        while let Some((seq, run)) = next.take() {
            let job = run.unwrap_or_else(|payload| std::panic::resume_unwind(payload));
            self.done.insert(seq, job);
            next = self.rx.try_recv().ok();
        }
        while let Some(job) = self.done.remove(&self.merge_seq) {
            rev5_merge_job(&job, &self.compute, self.fold_v1, totals, None);
            self.merge_seq += 1;
            self.in_flight -= 1;
            self.free.push(job);
        }
    }

    /// Finish and merge every job in flight; returns the job slots.
    fn drain(&mut self, totals: &mut [StreamChannelAccums; 3]) {
        while self.in_flight > 0 {
            self.wait(totals);
        }
    }
}

/// A finished task of [`rev5_block_walk`]: a produced chunk (side, chunk
/// index) or a strip job (sequence number).
#[cfg(feature = "threads")]
enum BlockWalkMsg {
    Chunk(
        usize,
        usize,
        std::thread::Result<crate::feature_v2_blocks::ChunkPlanes>,
    ),
    // Boxed: a job slot is about 21 KB, a chunk message about 120 bytes.
    Job(usize, std::thread::Result<Box<Rev5StripJob>>),
}

/// The plain revision-5+ walk with parallel row production
/// ([`crate::feature_v2_blocks`]): chunk tasks convert rows and downscale them
/// into immutable blocks, strip jobs copy their own windows out of the blocks,
/// and this thread only schedules, adds the `globaff` rows and merges finished
/// jobs. Jobs are submitted, and merged, in the streaming walk's order: strips
/// in ascending order within each scale and channels in order within a strip,
/// so every accumulator receives the same partials in the same order and every
/// value keeps its bits. Production runs at most a few chunks ahead of the
/// rows the next strip needs.
#[cfg(feature = "threads")]
#[allow(clippy::too_many_arguments)]
fn rev5_block_walk<S: ImageSource, D: ImageSource, const ALL_CHANNELS: bool>(
    source: &S,
    distorted: &D,
    dims: &[(usize, usize)],
    compute: &ComputeSet,
    toggles: V2NewFeatureToggles,
    fold_v1: bool,
    limit: usize,
    omit_scale0_xb: bool,
    slots: &mut Vec<Rev5StripJob>,
    pool: &std::sync::Arc<crate::feature_v2_blocks::BlockPool>,
    totals: &mut [StreamChannelAccums; 3],
    globaff: &mut [[crate::globaff::GlobAffAccum; 3]],
    stop: Option<&dyn enough::Stop>,
) {
    use crate::feature_v2_blocks::{CHUNK_ROWS, RowBlock, gather_window, produce_chunk};
    use crate::feature_v2_stream::StripInfo;
    use std::collections::{BTreeMap, VecDeque};
    use std::sync::Arc;

    let n_scales = dims.len();
    let h0 = dims[0].1;
    let n_chunks = h0.div_ceil(CHUNK_ROWS);
    let threads = rayon::current_num_threads().max(2);
    // Production may run this many scale-0 rows past the rows the next
    // strip needs, with at most this many chunk tasks (one per side) queued.
    // At most twelve strip jobs hold scratch at once: each holds about
    // 9.8 MB at 1507 columns, and sixteen at once overflowed a 96 MiB L3 at
    // sixteen threads on eight cores (paired, B at revision 7, 1507×2048:
    // twelve −13.2% and eight −20.8% against sixteen; at 32 threads on two
    // CCDs twelve −2.5% and eight +8.8%). Eight chunks of look-ahead instead
    // of one per thread: −3.6% at sixteen threads, −7.6% at 32 (1024²).
    let ahead = threads.min(8) * CHUNK_ROWS;
    let prod_limit = 2 * threads.min(8);
    let limit = limit.min(12);
    // The streaming producer's conversion revision.
    let revision = toggles.formula_revision;
    let globaff_on = !globaff.is_empty();
    let compute = *compute;

    rayon::in_place_scope(|scope| {
        let (tx, rx) = std::sync::mpsc::channel::<BlockWalkMsg>();
        // Retained blocks per side, channel and scale, in row order; rows
        // `[.., hi[scale])` are produced on both sides.
        let mut planes: [[Vec<VecDeque<Arc<RowBlock>>>; 3]; 2] = core::array::from_fn(|_| {
            core::array::from_fn(|_| (0..n_scales).map(|_| VecDeque::new()).collect())
        });
        let mut hi = vec![0usize; n_scales];
        let mut arrived: BTreeMap<usize, [Option<crate::feature_v2_blocks::ChunkPlanes>; 2]> =
            BTreeMap::new();
        let mut appended = 0usize;
        let mut next_chunk = 0usize;
        let mut prod_in_flight = 0usize;
        let mut next_ks = vec![0usize; n_scales];
        // The next channel to submit within each scale's current strip.
        let mut next_ch = vec![0usize; n_scales];
        let mut free = core::mem::take(slots);
        // Strip scratch released by finished jobs whose merge does not read
        // it (every job but revision-5/6 blockiness, which the merge sums
        // from the job's windows): `limit` bounds the scratch in use, so a
        // finished job never holds its planes while it waits to merge.
        let mut free_scratch: Vec<(ScratchV2Strip, usize)> = Vec::new();
        let (mut in_scratch, mut pending) = (0usize, 0usize);
        let (mut next_seq, mut merge_seq) = (0usize, 0usize);
        let mut done: BTreeMap<usize, Rev5StripJob> = BTreeMap::new();
        let mut stopped = false;
        let keeps_scratch = |scale: usize| {
            let local = compute.at_scale(scale);
            local.blockiness && !local.formula_revision.row_lanes()
        };

        let strip_at = |s: usize, ks: usize| {
            let (plane_w, plane_h) = dims[s];
            let y0 = ks * STRIP_ROWS;
            (y0 < plane_h).then(|| StripInfo {
                scale: s,
                y0,
                strip_h: STRIP_ROWS.min(plane_h - y0),
                plane_w,
                plane_h,
            })
        };
        // The scale-0 rows that make a strip's window complete.
        let need0 = |info: &StripInfo| {
            ((info.y0 + info.strip_h + HALO_P).min(info.plane_h) << info.scale).min(h0)
        };

        loop {
            if !stopped && stop.is_some_and(|token| token.check().is_err()) {
                stopped = true;
            }

            // Production, a bounded distance past the next strip's rows.
            if !stopped {
                let want0 = (0..n_scales)
                    .filter_map(|s| strip_at(s, next_ks[s]).map(|info| need0(&info)))
                    .min()
                    .unwrap_or(h0);
                while next_chunk < n_chunks
                    && prod_in_flight + 2 <= prod_limit
                    && next_chunk * CHUNK_ROWS < want0 + ahead
                {
                    let k = next_chunk;
                    for side in 0..2 {
                        let tx = tx.clone();
                        scope.spawn(move |_| {
                            let run =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    if side == 0 {
                                        produce_chunk(
                                            source,
                                            k,
                                            dims,
                                            omit_scale0_xb,
                                            revision,
                                            pool,
                                        )
                                    } else {
                                        produce_chunk(
                                            distorted,
                                            k,
                                            dims,
                                            omit_scale0_xb,
                                            revision,
                                            pool,
                                        )
                                    }
                                }));
                            // The walk holds the receiver until every task reports.
                            let _ = tx.send(BlockWalkMsg::Chunk(side, k, run));
                        });
                    }
                    next_chunk += 1;
                    prod_in_flight += 2;
                }
            }

            // Every ready strip, shallowest scale first, while job slots last.
            'emit: for s in 0..n_scales {
                if stopped {
                    break;
                }
                while let Some(info) = strip_at(s, next_ks[s]) {
                    if hi[s] < (info.y0 + info.strip_h + HALO_P).min(info.plane_h) {
                        break;
                    }
                    if next_ch[s] == 0 {
                        // `globaff`: every strip of every scale, all three
                        // channels, in strip order, before the dead-scale skip.
                        if globaff_on {
                            let y1 = info.y0 + info.strip_h;
                            for (ch, acc) in globaff[s].iter_mut().enumerate() {
                                let [src, dst] = [&planes[0][ch][s], &planes[1][ch][s]];
                                for (bs, bd) in src.iter().zip(dst) {
                                    let (a, b) = (bs.r0().max(info.y0), bs.end().min(y1));
                                    if a < b {
                                        acc.add_rows(bs.rows(a, b), bd.rows(a, b), info.plane_w);
                                    }
                                }
                            }
                        }
                        if compute.scale_dead(s) {
                            #[cfg(test)]
                            SKIPPED_SCALES.with_borrow_mut(|v| *v |= 1 << s);
                            next_ch[s] = 3;
                        }
                    }
                    let local = compute.at_scale(s);
                    let strip_toggles = V2NewFeatureToggles {
                        v1_pools: local.v1_pools,
                        v1_only: !local.v2_blocks,
                        gradient_features: local.gradient,
                        blockiness: local.blockiness,
                        ..toggles
                    };
                    while next_ch[s] < 3 {
                        let ch = next_ch[s];
                        if !ALL_CHANNELS && !compute.channel_active(s, ch) {
                            next_ch[s] += 1;
                            continue;
                        }
                        if in_scratch >= limit {
                            break 'emit;
                        }
                        let lo = info.y0.saturating_sub(HALO_P);
                        let top = (info.y0 + info.strip_h + HALO_P).min(info.plane_h);
                        let window = |side: usize| -> Vec<Arc<RowBlock>> {
                            planes[side][ch][s]
                                .iter()
                                .filter(|b| b.end() > lo && b.r0() < top)
                                .cloned()
                                .collect()
                        };
                        let (src_blocks, dst_blocks) = (window(0), window(1));
                        let mut job = free.pop().unwrap_or_else(|| Rev5StripJob::new(info, ch));
                        if job.sized_for == 0
                            && let Some((scratch, sized_for)) = free_scratch.pop()
                        {
                            job.scratch = scratch;
                            job.sized_for = sized_for;
                        }
                        job.prepare(info, ch);
                        let (seq, tx) = (next_seq, tx.clone());
                        scope.spawn(move |_| {
                            let run =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    let n = info.plane_w * info.wide_h();
                                    gather_window(
                                        &src_blocks,
                                        &info,
                                        &mut job.scratch.src_wide[..n],
                                    );
                                    gather_window(
                                        &dst_blocks,
                                        &info,
                                        &mut job.scratch.dst_wide[..n],
                                    );
                                    drop((src_blocks, dst_blocks));
                                    rev5_execute_job(
                                        &mut job,
                                        None,
                                        &compute,
                                        strip_toggles,
                                        fold_v1,
                                        false,
                                        None,
                                    );
                                }));
                            let _ = tx.send(BlockWalkMsg::Job(seq, run.map(|()| Box::new(job))));
                        });
                        next_seq += 1;
                        in_scratch += 1;
                        pending += 1;
                        next_ch[s] += 1;
                    }
                    next_ks[s] += 1;
                    next_ch[s] = 0;
                    // Retire the blocks no later strip of this scale reads.
                    let floor = strip_at(s, next_ks[s])
                        .map_or(usize::MAX, |next| next.y0.saturating_sub(HALO_P));
                    for side in &mut planes {
                        for ch in side.iter_mut() {
                            while ch[s].front().is_some_and(|b| b.end() <= floor) {
                                ch[s].pop_front();
                            }
                        }
                    }
                }
            }

            let emitted = (0..n_scales).all(|s| strip_at(s, next_ks[s]).is_none());
            if (emitted || stopped) && pending == 0 && prod_in_flight == 0 {
                break;
            }

            // Wait for at least one finished task, then take every other
            // finished one. A rayon worker waiting here runs pending pool work.
            let mut msg = None;
            while msg.is_none() {
                match rx.try_recv() {
                    Ok(m) => msg = Some(m),
                    Err(std::sync::mpsc::TryRecvError::Empty) => match rayon::yield_now() {
                        Some(rayon::Yield::Executed) => {}
                        Some(rayon::Yield::Idle) => std::thread::yield_now(),
                        None => msg = Some(rx.recv().expect("block walk channel")),
                    },
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        unreachable!("the walk holds a sender")
                    }
                }
            }
            while let Some(m) = msg.take() {
                match m {
                    BlockWalkMsg::Chunk(side, k, run) => {
                        prod_in_flight -= 1;
                        let chunk =
                            run.unwrap_or_else(|payload| std::panic::resume_unwind(payload));
                        arrived.entry(k).or_default()[side] = Some(chunk);
                        // Append whole chunks (both sides) in order.
                        while let Some(both) = arrived.get(&appended)
                            && both.iter().all(Option::is_some)
                        {
                            let both = arrived.remove(&appended).expect("present");
                            for (side, chunk) in both.into_iter().enumerate() {
                                for (ch, scales) in chunk.expect("present").into_iter().enumerate()
                                {
                                    for (s, block) in scales.into_iter().enumerate() {
                                        if let Some(block) = block {
                                            if ch == 1 {
                                                hi[s] = hi[s].max(block.end());
                                            }
                                            planes[side][ch][s].push_back(block);
                                        }
                                    }
                                }
                            }
                            appended += 1;
                        }
                    }
                    BlockWalkMsg::Job(seq, run) => {
                        let mut job =
                            *run.unwrap_or_else(|payload| std::panic::resume_unwind(payload));
                        if !keeps_scratch(job.info.scale) {
                            let scratch =
                                core::mem::replace(&mut job.scratch, ScratchV2Strip::new(0));
                            free_scratch.push((scratch, core::mem::take(&mut job.sized_for)));
                            in_scratch -= 1;
                        }
                        done.insert(seq, job);
                        while let Some(job) = done.remove(&merge_seq) {
                            rev5_merge_job(&job, &compute, fold_v1, totals, None);
                            if keeps_scratch(job.info.scale) {
                                in_scratch -= 1;
                            }
                            merge_seq += 1;
                            pending -= 1;
                            free.push(job);
                        }
                    }
                }
                msg = rx.try_recv().ok();
            }
        }
        // Reattach the released scratch so the next walk's first `limit`
        // slots keep their planes.
        free.sort_by_key(|job| core::cmp::Reverse(job.sized_for));
        for job in free.iter_mut().filter(|job| job.sized_for == 0) {
            let Some((scratch, sized_for)) = free_scratch.pop() else {
                break;
            };
            job.scratch = scratch;
            job.sized_for = sized_for;
        }
        free.sort_by_key(|job| core::cmp::Reverse(job.sized_for));
        *slots = free;
    });
}

#[cfg(feature = "threads")]
fn rev5_run_jobs(
    jobs: &mut [Rev5StripJob],
    compute: &ComputeSet,
    toggles: V2NewFeatureToggles,
    fold_v1: bool,
    totals: &mut [StreamChannelAccums; 3],
    retention: Option<&mut FoldRetention>,
    reference: Option<&[crate::streaming::XybPyramidLevel]>,
) {
    use rayon::prelude::*;
    let wall = crate::fold_timing::start();
    let retain = retention.is_some();
    let outputs = retention.map(|r| r.job_sinks(jobs));
    let execute = |job: &mut Rev5StripJob, output: Option<RetainedRows<'_>>| {
        rev5_execute_job(job, output, compute, toggles, fold_v1, retain, reference)
    };
    if let Some(outputs) = outputs {
        jobs.par_iter_mut()
            .zip(outputs.into_par_iter())
            .for_each(|(job, output)| execute(job, output));
    } else {
        jobs.par_iter_mut().for_each(|job| execute(job, None));
    }
    crate::fold_timing::stop(wall, crate::fold_timing::Phase::BWall, 0);
    // Every cell's kernel partial is merged once, in the producer's order.
    // Do not replace this loop with a parallel/tree reduction.
    for job in jobs.iter() {
        rev5_merge_job(job, compute, fold_v1, totals, reference);
    }
}

impl StreamChannelAccums {
    /// `n_band_slots` is the number of band scratches this channel keeps —
    /// see [`band_slots_for`]. It is a CAPACITY, not a tiling: bands are
    /// chunked across whatever slots exist (`fold_v1_basic_bands`), so any
    /// count in `1..=V1_BANDS_PER_STRIP` computes the same bytes.
    fn at_revision(n_scales: usize, n_band_slots: usize, revision: FormulaRevision) -> Self {
        Self {
            retained_strips: None,
            dense: vec![DenseAccum::default(); n_scales],
            grad: vec![GradientAccum::default(); n_scales],
            app: vec![AppendAccum::default(); n_scales],
            v1: vec![V1BasicSums::at_revision(revision); n_scales],
            block: vec![
                (
                    crate::featcanon::SumVar::f64_elems(crate::featcanon::mode(revision)),
                    crate::featcanon::SumVar::f64_elems(crate::featcanon::mode(revision))
                );
                n_scales
            ],
            csfw: vec![CsfwAccum::default(); n_scales],
            rev4: (0..n_scales).map(|_| Rev4CellAccum::default()).collect(),
            pool_scratch: (0..n_band_slots.clamp(1, V1_BANDS_PER_STRIP))
                .map(|_| FoldPoolScratch::default())
                .collect(),
        }
    }
}

/// How many [`FoldPoolScratch`] slots one channel needs — the fold-footprint
/// lane's answer to "why is the fold heavier than buffered below ~2.5 MP".
///
/// The band scratch is the fold's single largest term (measured: 33 % of the
/// walk's peak heap at every size, `benchmarks/fold_footprint_2026-08-31.md`),
/// and it was sized by the FAN-OUT SHAPE — `V1_BANDS_PER_STRIP` slots per
/// channel × 3 channels = 12 band buffers — regardless of how many could ever
/// be live. Buffered's equivalent (`streaming`'s `ScaleBuffers`, 7 planes over
/// the same 42-row band) is sized by the WORKER COUNT, via `map_init`: one per
/// rayon worker. At one thread that is 12 buffers against buffered's 1, on
/// identical work, and it is most of the gap.
///
/// A band cannot run without a thread to run it on, so `min(bands, threads)`
/// is the count that can ever be simultaneously live inside one channel task
/// (rayon nests: a channel's band `par_iter` can be stolen by every thread in
/// the pool). Above that the extra slots are memory nothing can reach.
///
/// Byte-neutral: which scratch a band borrows cannot reach a feature, because
/// every band fully overwrites the region it reads and merges into its own
/// zero-initialised `V1BasicSums`, which are then reduced in BAND ORDER either
/// way (`fold_v1_basic_bands`). `fold_engine_parity` sweeps rayon pools
/// 1/2/3/8/16, which is now also a sweep of this function's output (1/2/3/4/4).
fn band_slots_for(band_parallel: bool) -> usize {
    #[cfg(feature = "threads")]
    let n = if band_parallel {
        V1_BANDS_PER_STRIP.min(rayon::current_num_threads().max(1))
    } else {
        1
    };
    #[cfg(not(feature = "threads"))]
    let n = {
        let _ = band_parallel;
        1usize
    };
    n
}

/// True when the append kernel runs for this (channel, scale) — mirrors
/// the materialized walk's [`APPEND_SKIP_B_SCALE0`] dispatch.
#[inline]
fn append_cell_active(append_on: bool, ch: usize, scale: usize) -> bool {
    append_on && !(APPEND_SKIP_B_SCALE0 && ch == 2 && scale == 0)
}

/// Retained per-(scale, channel) state from ONE streaming folded walk —
/// the extractor-side hooks of the fused folded-944 score+attribution
/// compare (campaign appendix N). Holds full-image copies of the pyramid
/// core rows and the phase-A V-blur planes — BITWISE the values the walk
/// computed (the C1 producer gate holds strip windows byte-equal to the
/// materialized pyramids, and phase A is the same blur chain
/// `attr_blur_cache_channel` replicates) — plus the EXACT per-cell pooled
/// accumulators and mean-gradients captured at finalize. Buffers are
/// reused across calls (`ensure`); every retained row a fused compare
/// later reads is written each walk, and the only conditionally-computed
/// plane (`bs2` where the append kernel is inactive — the
/// `APPEND_SKIP_B_SCALE0` cell) is zero-filled explicitly so session
/// reuse is deterministic. Memory: 8 planes × 3 ch × Σ scale sizes
/// (~42 MB at 576²) — the same class the standalone attribution's
/// materialized pyramids + plane sets pay.
pub(crate) struct FoldRetention {
    pub(crate) dims: Vec<(usize, usize)>,
    /// `[scale][ch]` source-side pyramid planes (core rows).
    pub(crate) pyr_src: Vec<[Vec<f32>; 3]>,
    /// `[scale][ch]` distorted-side pyramid planes (core rows).
    pub(crate) pyr_dst: Vec<[Vec<f32>; 3]>,
    /// `[scale][ch]` phase-A V-blur planes (mu1/mu2/ssq/s12/act/bs2).
    pub(crate) planes: Vec<[AttrChPlanes; 3]>,
    /// `[scale][ch]` exact pooled cells (walk accumulators + blockiness).
    pub(crate) cells: Vec<[AttrCellSums; 3]>,
    /// `[scale][ch]` (mean grad src, mean grad dst) from finalize.
    pub(crate) mg: Vec<[(f64, f64); 3]>,
    /// Production strip partials, in walk order, for exact coarse refinement.
    pub(crate) strips: Vec<[Vec<(DenseAccum, GradientAccum)>; 3]>,
    /// Prepared finite-query snapshots can move all distorted scale-0 channels.
    pub(crate) complete_distorted0: bool,
    #[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
    pub(crate) snapshot_reuse: Option<crate::local_refine::SnapshotReuse>,
    pub(crate) basic_stats: Vec<crate::metric::ScaleStats>,
    pub(crate) basic_hf: Vec<[(f64, f64); 3]>,
    pub(crate) basic_sd: Vec<[Vec<f32>; 3]>,
    retain_basic: bool,
    rev5: bool,
    /// The revision the retaining walk computed its planes at (the plan's, not the process's).
    pub(crate) revision: FormulaRevision,
    channels: [[bool; 3]; crate::NUM_SCALES],
}

impl Default for FoldRetention {
    fn default() -> Self {
        Self::new()
    }
}

impl FoldRetention {
    pub(crate) fn new() -> Self {
        Self {
            dims: Vec::new(),
            pyr_src: Vec::new(),
            pyr_dst: Vec::new(),
            planes: Vec::new(),
            cells: Vec::new(),
            mg: Vec::new(),
            strips: Vec::new(),
            complete_distorted0: false,
            #[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
            snapshot_reuse: None,
            basic_stats: Vec::new(),
            basic_hf: Vec::new(),
            basic_sd: Vec::new(),
            retain_basic: false,
            rev5: false,
            revision: crate::ssim_form::SHIPPED_REVISION,
            channels: [[false; 3]; crate::NUM_SCALES],
        }
    }

    /// Size every buffer for the walk's scale dims (no-op when unchanged;
    /// cells/mg/planes are fully rewritten by each retaining walk).
    fn ensure(
        &mut self,
        dims: &[(usize, usize)],
        compute: &ComputeSet,
        retain_basic: bool,
        complete_distorted0: bool,
    ) {
        #[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
        crate::local_refine::restore_snapshot_buffers(self);
        self.strips.clear();
        self.strips
            .resize_with(dims.len(), || core::array::from_fn(|_| Vec::new()));
        let rev5 = compute.formula_revision >= FormulaRevision::Rev5;
        self.revision = compute.formula_revision;
        let channels = core::array::from_fn(|scale| {
            core::array::from_fn(|ch| {
                scale < dims.len() && (!rev5 || compute.channel_active(scale, ch))
            })
        });
        if self.dims.as_slice() == dims
            && self.retain_basic == retain_basic
            && self.channels == channels
            && self.rev5 == rev5
            && self.complete_distorted0 == complete_distorted0
        {
            // A steering snapshot may have moved planes out
            // (`LocalRefineSnapshot::capture_taking`); re-create exactly those.
            // Every retaining walk rewrites each element before any read.
            for (scale, &(w, h)) in dims.iter().enumerate() {
                for (ch, &active) in channels[scale].iter().enumerate() {
                    let n = if active { w * h } else { 0 };
                    if self.pyr_src[scale][ch].len() != n {
                        if n == 0 {
                            // Keep retired snapshot storage for the next cached
                            // source-plane copy; unserved channels stay empty.
                            self.pyr_src[scale][ch].clear();
                        } else {
                            self.pyr_src[scale][ch] = vec![0.0; n];
                        }
                    }
                    let dn = if scale == 0 && complete_distorted0 {
                        w * h
                    } else {
                        n
                    };
                    if self.pyr_dst[scale][ch].len() != dn {
                        self.pyr_dst[scale][ch] = vec![0.0; dn];
                    }
                    let p = &self.planes[scale][ch];
                    if [&p.mu1, &p.mu2, &p.ssq, &p.s12, &p.act]
                        .iter()
                        .any(|v| v.len() != n)
                    {
                        self.planes[scale][ch] = AttrChPlanes::new_for(n, !rev5);
                    }
                }
            }
            return;
        }
        self.dims = dims.to_vec();
        let mk = |scale: usize, n: usize| -> [Vec<f32>; 3] {
            core::array::from_fn(|ch| vec![0.0; if channels[scale][ch] { n } else { 0 }])
        };
        self.pyr_src = dims
            .iter()
            .enumerate()
            .map(|(scale, &(w, h))| mk(scale, w * h))
            .collect();
        self.pyr_dst = dims
            .iter()
            .enumerate()
            .map(|(scale, &(w, h))| {
                if scale == 0 && complete_distorted0 {
                    core::array::from_fn(|_| vec![0.0; w * h])
                } else {
                    mk(scale, w * h)
                }
            })
            .collect();
        self.planes = dims
            .iter()
            .enumerate()
            .map(|(scale, &(w, h))| {
                core::array::from_fn(|ch| {
                    AttrChPlanes::new_for(if channels[scale][ch] { w * h } else { 0 }, !rev5)
                })
            })
            .collect();
        self.cells = vec![[AttrCellSums::default(); 3]; dims.len()];
        self.mg = vec![[(0.0, 0.0); 3]; dims.len()];
        self.complete_distorted0 = complete_distorted0;
        self.retain_basic = retain_basic;
        self.rev5 = rev5;
        self.basic_stats = if retain_basic {
            vec![crate::metric::ScaleStats::default(); dims.len()]
        } else {
            Vec::new()
        };
        self.basic_hf = if retain_basic {
            vec![[(0.0, 0.0); 3]; dims.len()]
        } else {
            Vec::new()
        };
        self.basic_sd = if retain_basic {
            dims.iter()
                .enumerate()
                .map(|(scale, &(w, h))| mk(scale, w * h))
                .collect()
        } else {
            Vec::new()
        };
        self.channels = channels;
    }

    /// Copy one (strip, channel)'s core rows out of the walk: pyramid
    /// windows + phase-A planes. Pure copies — the walk's own accumulation
    /// is untouched (G-N1 rests on this).
    fn copy_strip(
        &mut self,
        info: &crate::feature_v2_stream::StripInfo,
        ch: usize,
        scr: &ScratchV2Strip,
        src_win: &[f32],
        dst_win: &[f32],
        want_bs2: bool,
    ) {
        let w = info.plane_w;
        let off = HALO_P * w;
        let n = w * info.strip_h;
        let out = info.y0 * w;
        let scale = info.scale;
        self.pyr_src[scale][ch][out..out + n].copy_from_slice(&src_win[off..off + n]);
        self.pyr_dst[scale][ch][out..out + n].copy_from_slice(&dst_win[off..off + n]);
        let p = &mut self.planes[scale][ch];
        p.mu1[out..out + n].copy_from_slice(&scr.mu1[off..off + n]);
        p.mu2[out..out + n].copy_from_slice(&scr.mu2[off..off + n]);
        p.ssq[out..out + n].copy_from_slice(&scr.ssq[off..off + n]);
        p.s12[out..out + n].copy_from_slice(&scr.s12[off..off + n]);
        p.act[out..out + n].copy_from_slice(&scr.activity[off..off + n]);
        if self.retain_basic {
            let sd = &mut self.basic_sd[scale][ch][out..out + n];
            incant!(
                retain_basic_sd(
                    &p.mu1[out..out + n],
                    &p.mu2[out..out + n],
                    &p.ssq[out..out + n],
                    &p.s12[out..out + n],
                    sd
                ),
                [v4x, v4, v3, scalar]
            );
        }
        if want_bs2 {
            p.bs2[out..out + n].copy_from_slice(&scr.bs2[off..off + n]);
        } else if !self.rev5 {
            // Never computed for this cell (`APPEND_SKIP_B_SCALE0`) — its
            // pass-B terms carry zero coefficients, but zero the plane so
            // reuse across pairs is deterministic.
            p.bs2[out..out + n].fill(0.0);
        }
    }
}

#[cfg(feature = "threads")]
include!("feature_v2_retained_rows.rs");
#[cfg(feature = "threads")]
#[path = "feature_v2_retained_core.rs"]
mod retained_core;
#[cfg(feature = "threads")]
use retained_core::{
    RetainedCoreWork, retained_core_admits, rev5_blur_into_retained, rev5_retained_phase_b,
    rev5_retained_single_channel,
};

// Retain the basic direct-error plane with the same fused expression as its
// fold owner. Setup and correctly rounded scalar tails remain outside SIMD.
fn retain_basic_sd_scalar(
    _token: archmage::ScalarToken,
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    err: &[f32],
    sd: &mut [f32],
) {
    for (i, value) in sd.iter_mut().enumerate() {
        *value = crate::ssim_form::ssim_direct_raw_scalar(
            crate::ssim_form::SsimLumaForm::Clamp,
            mu1[i],
            mu2[i],
            ssq[i],
            err[i],
        )
        .max(0.0);
    }
}
#[magetypes(define(f32x8), v4x, v4, v3, -scalar)]
fn retain_basic_sd(
    token: Token,
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    err: &[f32],
    sd: &mut [f32],
) {
    let form = crate::ssim_form::SsimLumaForm::Clamp;
    let splats = crate::ssim_form::SsimSplats8::new(token, form);
    let zero = f32x8::zero_t(token);
    let mut i = 0;
    while i + 8 <= sd.len() {
        let m1 = f32x8::load_t(token, mu1[i..i + 8].try_into().unwrap());
        let m2 = f32x8::load_t(token, mu2[i..i + 8].try_into().unwrap());
        let sq = f32x8::load_t(token, ssq[i..i + 8].try_into().unwrap());
        let er = f32x8::load_t(token, err[i..i + 8].try_into().unwrap());
        splats
            .direct(m1, m2, sq, er)
            .max(zero)
            .store((&mut sd[i..i + 8]).try_into().unwrap());
        i += 8;
    }
    retain_basic_sd_scalar(
        archmage::ScalarToken,
        &mu1[i..],
        &mu2[i..],
        &ssq[i..],
        &err[i..],
        &mut sd[i..],
    );
}

/// One (strip, channel)'s wide input windows: zero-copy slices of the
/// producer's rolling planes for interior strips, or the scratch
/// `src_wide`/`dst_wide` buffers filled via `fill_wide` when the strip
/// touches the true top/bottom (reflection must be materialized). Both
/// forms hold IDENTICAL bytes for identical rows (C1's producer gate), so
/// every downstream kernel is bitwise-indifferent to which one it reads —
/// the zero-copy path just skips two `n_wide` memcpys per channel-strip.
/// Re-derive one (strip, channel)'s wide windows for a SHARED-borrow
/// consumer (phase B): zero-copy producer slices for interior strips, or
/// the scratch buffers phase A filled for edge strips. Identical bytes
/// either way (C1's producer gate).
fn stream_windows_shared<'w, S: ImageSource, D: ImageSource>(
    producer: &'w crate::feature_v2_stream::StripPlaneProducer<'_, S, D>,
    info: &crate::feature_v2_stream::StripInfo,
    ch: usize,
    scr: &'w ScratchV2Strip,
) -> (&'w [f32], &'w [f32]) {
    use crate::feature_v2_stream::Side;
    let n_wide = info.plane_w * info.wide_h();
    match (
        producer.wide_window(Side::Source, ch, info),
        producer.wide_window(Side::Distorted, ch, info),
    ) {
        (Some(sw), Some(dw)) => (sw, dw),
        _ => (&scr.src_wide[..n_wide], &scr.dst_wide[..n_wide]),
    }
}

/// The wide-window GATHER half of [`stream_phase_a`], for the self-blur band
/// shape that skips the rest of phase A.
///
/// Interior strips take the producer's rolling-plane slices directly and this
/// is a no-op; only a strip touching the true top or bottom needs the
/// reflect-padded copy into the scratch, which is exactly what
/// [`stream_windows_shared`] then reads back. Byte-identical to what phase A
/// does — it is the same two calls, lifted.
///
/// Its only call site is inside a `#[cfg(feature = "threads")]` block, so a
/// `threads`-off build (e.g. the `feature-regime-v2`-alone CI permutation)
/// has no caller for it; `cfg_attr` scopes the allow to exactly that case
/// rather than silencing dead-code detection when `threads` IS on.
#[cfg_attr(not(feature = "threads"), allow(dead_code))]
fn stream_gather_windows<S: ImageSource, D: ImageSource>(
    producer: &crate::feature_v2_stream::StripPlaneProducer<'_, S, D>,
    info: &crate::feature_v2_stream::StripInfo,
    ch: usize,
    scr: &mut ScratchV2Strip,
) {
    use crate::feature_v2_stream::Side;
    let n_wide = info.plane_w * info.wide_h();
    if producer.wide_window(Side::Source, ch, info).is_none()
        || producer.wide_window(Side::Distorted, ch, info).is_none()
    {
        producer.fill_wide(Side::Source, ch, info, &mut scr.src_wide[..n_wide]);
        producer.fill_wide(Side::Distorted, ch, info, &mut scr.dst_wide[..n_wide]);
    }
}

/// Phase A of one (strip, channel): resolve the wide windows (zero-copy
/// producer slices for interior strips; `fill_wide` into the scratch
/// buffers when the strip touches the true top/bottom), then the
/// materialized walk's moments-free blur pass (fused H + 4x V +
/// activity), plus the P2 strip-tiled bs2 σ-split chain when the append
/// cell is active (`abs_src`/`activity_tmp` are dead after the activity
/// chain and serve as its square/H temps).
fn stream_phase_a<S: ImageSource, D: ImageSource>(
    producer: &crate::feature_v2_stream::StripPlaneProducer<'_, S, D>,
    info: &crate::feature_v2_stream::StripInfo,
    ch: usize,
    want_bs2: bool,
    want_act_dst: bool,
    want_v2: bool,
    // `parallel`: row-band-parallelise the H-blur
    // (`fused_blur_h_ssim_banded`). The channel fan-out above is only 3-way,
    // so on a >3-thread pool phase A is the walk's occupancy floor; bands are
    // its second axis.
    parallel: bool,
    scr: &mut ScratchV2Strip,
    revision: FormulaRevision,
) {
    use crate::feature_v2_stream::Side;
    let width = info.plane_w;
    let wide_h = info.wide_h();
    let n_wide = width * wide_h;
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
        bs2,
        activity_dst,
    } = scr;
    let (src_win, dst_win): (&[f32], &[f32]) = match (
        producer.wide_window(Side::Source, ch, info),
        producer.wide_window(Side::Distorted, ch, info),
    ) {
        (Some(sw), Some(dw)) => (sw, dw),
        _ => {
            producer.fill_wide(Side::Source, ch, info, &mut src_wide[..n_wide]);
            producer.fill_wide(Side::Distorted, ch, info, &mut dst_wide[..n_wide]);
            (&src_wide[..n_wide], &dst_wide[..n_wide])
        }
    };
    run_blur_pass_inner(
        src_win,
        dst_win,
        width,
        wide_h,
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
        want_v2,
        parallel,
        info.scale,
        revision,
    );
    // BANDVIS dst self-mask (`append2_dst_activity`, Y channel only): the
    // exact dst twin of the ref activity chain — `box_blur(|dst − mu2|)`
    // over the same wide window, mu2 already produced by the fused blur
    // pass above. `abs_src`/`activity_tmp` are dead after the ref chain
    // and serve as its temps (the same reuse the bs2 fill below makes —
    // which runs after and overwrites them again, so ordering here is
    // load-bearing: act-dst BEFORE bs2).
    let __t_pap = crate::fold_timing::start();
    if want_act_dst {
        if activity_dst.len() < n_wide {
            activity_dst.resize(n_wide, 0.0);
        }
        crate::simd_ops::abs_diff_into(dst_win, &mu2[..n_wide], &mut abs_src[..n_wide]);
        crate::blur::box_blur_1pass_into(
            &abs_src[..n_wide],
            &mut activity_dst[..n_wide],
            &mut activity_tmp[..n_wide],
            width,
            wide_h,
            BLUR_RADIUS,
        );
    }
    if want_bs2 {
        square_into(src_win, &mut abs_src[..n_wide]);
        crate::blur::box_blur_h(
            &abs_src[..n_wide],
            &mut activity_tmp[..n_wide],
            width,
            wide_h,
            BLUR_RADIUS,
        );
        crate::blur::box_blur_v_from_copy(
            &activity_tmp[..n_wide],
            &mut bs2[..n_wide],
            width,
            wide_h,
            BLUR_RADIUS,
        );
    }
    crate::fold_timing::stop(
        __t_pap,
        crate::fold_timing::Phase::PhaseAAppendPlanes,
        info.scale,
    );
}

/// Phase B of one (strip, channel): the materialized walk's kernel set
/// over the phase-A planes, folded into `acc` in kernel-strip order.
///
/// `refy_strip` is the reference Y plane's strip rows (from the
/// producer's rolling plane — valid until the next `next_strip` call);
/// `cross` is `Some((act_x, act_b))` strip slices for the Y channel's
/// cross-masked transducer (X/B's phase-A activity, still live in their
/// scratches — computed once per strip, replacing the materialized pair
/// path's whole-plane replay planes).
///
/// LOCALITY CONTRACT: call this IMMEDIATELY after the same channel's
/// [`stream_phase_a`] — the fold band replay re-reads the H planes and
/// the kernels re-read the V planes while they are cache-hot (the same
/// reason the materialized walk runs its fold hook "before the dense
/// kernel purely for locality"). Interleaving another channel's phase A
/// in between evicts ~9 MB of wide buffers and was measured at
/// +50 ms/pair on aic3-100 (the C3 two-phase draft).
#[allow(clippy::too_many_arguments)]
fn stream_phase_b(
    scr: &ScratchV2Strip,
    src_win: &[f32],
    dst_win: &[f32],
    info: &crate::feature_v2_stream::StripInfo,
    toggles: V2NewFeatureToggles,
    fold_v1: bool,
    v2_blocks: bool,
    band_parallel: bool,
    // `self_blur`: each v1 band produces its OWN H planes instead of reading
    // the strip-wide ones phase A wrote. Only legal when phase A did NOT run
    // (`!v2_blocks`) and `v1_pools` is `Full` (SelfBlur needs the band pool
    // scratch) — `foldapp_streaming_walk` owns that decision.
    self_blur: bool,
    append: bool,
    refy_strip: &[f32],
    cross: Option<(&[f32], &[f32])>,
    append2: Option<Append2Params>,
    csfw: Option<CsfwParams>,
    // The resolved per-channel work order. Do not reconstruct it from public
    // toggles: that loses private subset restrictions and runs dead reductions.
    free: crate::fused::FreeExtrasWork,
    // REV4's resolved per-(strip, channel) work order — `Rev4Work::OFF`
    // on every non-rev4 call (loop-invariant Option branches inside the
    // kernels, same shape as `transducer_bank`).
    r4w: Rev4Work<'_>,
    acc: &mut StreamChannelAccums,
) {
    let width = info.plane_w;
    let strip_h = info.strip_h;
    let y0 = info.y0;
    let scale = info.scale;
    let n_wide = width * info.wide_h();
    let off = HALO_P * width;
    let strip_n = width * strip_h;

    let src_strip = &src_win[off..off + strip_n];
    let dst_strip = &dst_win[off..off + strip_n];

    // BLOCK-SKIPPING: with `v1_only` the phase-A V planes and activity were
    // never produced, so nothing below may touch them. The v1 fold hook
    // reads only the H planes + the raw windows and is run unconditionally
    // below.
    if v2_blocks {
        let mu1_strip = &scr.mu1[off..off + strip_n];
        let mu2_strip = &scr.mu2[off..off + strip_n];
        let ssq_strip = &scr.ssq[off..off + strip_n];
        let s12_strip = &scr.s12[off..off + strip_n];
        let act_strip = &scr.activity[off..off + strip_n];

        let __t_dense = crate::fold_timing::start();
        // REV4 C3/C4-noise hook — `None` when both are off (the
        // `transducer_bank`-shaped invariant branch inside the kernel).
        let cell = &mut acc.rev4[scale];
        let r4d = if r4w.tailhist || r4w.flat {
            Some(Rev4Dense {
                tail: if r4w.tailhist {
                    Some(&mut cell.tail)
                } else {
                    None
                },
                flat: if r4w.flat { Some(&mut cell.flat) } else { None },
                edges: tail_edges(),
            })
        } else {
            None
        };
        let d = dense_block_kernel_at_revision(
            src_strip,
            dst_strip,
            mu1_strip,
            mu2_strip,
            ssq_strip,
            s12_strip,
            act_strip,
            width,
            strip_h,
            toggles.transducer_bank,
            r4d,
            toggles.formula_revision,
        );
        crate::fold_timing::stop(__t_dense, crate::fold_timing::Phase::DenseKernel, scale);
        acc.dense[scale].accumulate(&d);
        let mut retained_grad = GradientAccum::default();

        // REV4 C2/C4-bleed ride the gradient kernel too — it must run
        // whenever either hook has work, even with `gradient_features`
        // off (the hook's `Option` gates keep the rev4-off call
        // byte-identical to the pre-rev4 signature).
        if toggles.gradient_features || r4w.ringbasis || r4w.bleed_mask.is_some() || r4w.gmsbank {
            let g_off = (HALO_P - 1) * width;
            let g_n = width * (strip_h + 2);
            // BANDVIS accumulates only on (Y, append2 on) — the const-split
            // instantiation; every other path runs the byte-identical
            // BANDVIS=false kernel. The dst self-mask plane routes in only
            // when `append2_dst_activity` is live (third instantiation).
            let bandvis = append2
                .filter(|_| cross.is_some())
                .map(|p| (p.bv_delta_lo, p.bv_delta_hi));
            let bv_act_dst = append2
                .filter(|p| cross.is_some() && p.dst_activity)
                .map(|_| &scr.activity_dst[off..off + strip_n]);
            let cell = &mut acc.rev4[scale];
            let r4g = if r4w.ringbasis || r4w.bleed_mask.is_some() || r4w.gmsbank {
                Some(Rev4Grad {
                    ring: if r4w.ringbasis {
                        Some(&mut cell.ring)
                    } else {
                        None
                    },
                    bleed: r4w.bleed_mask.map(|m| (m, &mut cell.bleed)),
                    bank: r4w.bank,
                })
            } else {
                None
            };
            let __t_grad = crate::fold_timing::start();
            let g = gradient_block_kernel_at_revision(
                &src_win[g_off..g_off + g_n],
                &dst_win[g_off..g_off + g_n],
                act_strip,
                Some((mu1_strip, mu2_strip)),
                width,
                strip_h,
                bandvis,
                bv_act_dst,
                r4g,
                r4w.gmsbank,
                toggles.formula_revision,
            );
            crate::fold_timing::stop(__t_grad, crate::fold_timing::Phase::GradKernel, scale);
            acc.grad[scale].accumulate(&g);
            if toggles.gradient_features {
                retained_grad = g;
            }
        }
        if let Some(strips) = &mut acc.retained_strips
            && (1..=3).contains(&scale)
        {
            strips[scale].push((d, retained_grad));
        }

        // REV4 C1 — the full-boundary pass over the wide window (the
        // phase-A activity plane covers the halo rows, which is exactly
        // what `gridblk_strip_wide`'s contract requires).
        if r4w.gridblk_period >= 2 {
            let __t_r4 = crate::fold_timing::start();
            gridblk_strip_wide(
                &src_win[..n_wide],
                &dst_win[..n_wide],
                &scr.activity[..n_wide],
                width,
                y0,
                strip_h,
                info.plane_h,
                r4w.gridblk_period,
                &mut acc.rev4[scale].grid,
            );
            crate::fold_timing::stop(__t_r4, crate::fold_timing::Phase::Rev4Kernel, scale);
        }
    }

    if fold_v1 {
        stream_fold_v1(
            scr,
            src_win,
            dst_win,
            info,
            toggles,
            v2_blocks,
            band_parallel,
            self_blur,
            free,
            acc,
        );
    }

    if append {
        debug_assert!(v2_blocks, "append_block requires the v2 planes");
        debug_assert_eq!(refy_strip.len(), strip_n);
        // Re-bind the phase-A V planes: they are scoped to the `v2_blocks`
        // arm above, and `append` can only be set when that arm ran.
        let mu1_strip = &scr.mu1[off..off + strip_n];
        let mu2_strip = &scr.mu2[off..off + strip_n];
        let ssq_strip = &scr.ssq[off..off + strip_n];
        let act_strip = &scr.activity[off..off + strip_n];
        let hl = append2.map(|p| p.hl_bins).unwrap_or(false) && cross.is_some();
        let __t_app = crate::fold_timing::start();
        let a = append_block_kernel(
            src_strip,
            dst_strip,
            mu1_strip,
            mu2_strip,
            ssq_strip,
            &scr.bs2[off..off + strip_n],
            act_strip,
            refy_strip,
            cross,
            hl,
            width,
            strip_h,
        );
        crate::fold_timing::stop(__t_app, crate::fold_timing::Phase::AppendKernel, scale);
        acc.app[scale].accumulate(&a);

        // CSFW pass accumulates only on (Y, csfw_block on) — a separate
        // kernel invocation over the strip-resident rows (design §4.2);
        // every other path never dispatches it. `cross.is_some()` is the
        // Y-channel discriminant, same as the BANDVIS gate above.
        if let Some(cp) = csfw.filter(|_| cross.is_some()) {
            let __t_csfw = crate::fold_timing::start();
            let c = csfw_block_kernel(
                src_strip,
                dst_strip,
                refy_strip,
                cp.eff[scale],
                width,
                strip_h,
                toggles.formula_revision,
            );
            crate::fold_timing::stop(__t_csfw, crate::fold_timing::Phase::CsfwKernel, scale);
            acc.csfw[scale].accumulate(&c);
        }
    }

    if v2_blocks && toggles.blockiness {
        let (sum_v, sum_h) = &mut acc.block[scale];
        let __t_blk = crate::fold_timing::start();
        blockiness_sparse_strip_wide(
            &src_win[..n_wide],
            &dst_win[..n_wide],
            width,
            y0,
            strip_h,
            toggles.formula_revision,
            sum_v,
            sum_h,
        );
        crate::fold_timing::stop(__t_blk, crate::fold_timing::Phase::BlockKernel, scale);
    }
}

/// Replay each basic band directly into the running total. Keeping this
/// ordered merge separate lets strip tasks evaluate v2 in parallel without
/// reassociating any basic band or sparse blockiness sum.
#[allow(clippy::too_many_arguments)]
fn stream_fold_v1(
    scr: &ScratchV2Strip,
    src_win: &[f32],
    dst_win: &[f32],
    info: &crate::feature_v2_stream::StripInfo,
    toggles: V2NewFeatureToggles,
    v2_blocks: bool,
    band_parallel: bool,
    self_blur: bool,
    free: crate::fused::FreeExtrasWork,
    acc: &mut StreamChannelAccums,
) {
    let width = info.plane_w;
    let y0 = info.y0;
    let strip_h = info.strip_h;
    let scale = info.scale;
    let n_wide = width * info.wide_h();
    fold_v1_basic_bands(
        width,
        y0..y0 + strip_h,
        y0,
        HALO_P,
        info.plane_h,
        if free.revision() >= FormulaRevision::Rev5 && v2_blocks {
            FoldHSource::Vertical([
                &scr.mu1[..n_wide],
                &scr.mu2[..n_wide],
                &scr.ssq[..n_wide],
                &scr.s12[..n_wide],
            ])
        } else if self_blur {
            FoldHSource::SelfBlur
        } else {
            FoldHSource::Precomputed([
                &scr.mu1_h[..n_wide],
                &scr.mu2_h[..n_wide],
                &scr.ssq_h[..n_wide],
                &scr.s12_h[..n_wide],
            ])
        },
        [&src_win[..n_wide], &dst_win[..n_wide]],
        &mut acc.v1[scale],
        match toggles.v1_pools {
            V1PoolsMode::Off => None,
            // Peaks: the scratch travels only so a self-blur band owns
            // its H planes — no pool arithmetic runs (`HOnly`).
            V1PoolsMode::Peaks => Some((&mut acc.pool_scratch[..], BandPoolWork::HOnly)),
            // The carrier slots need the activity + edge kernel at
            // scales 0-1 only (masked_art_4th s0, iw_art_4th s0-s1);
            // the peaks come from the kernel at every scale for free.
            V1PoolsMode::Carriers if scale <= 1 => {
                Some((&mut acc.pool_scratch[..], BandPoolWork::Carriers))
            }
            V1PoolsMode::Carriers => None,
            V1PoolsMode::Full => Some((&mut acc.pool_scratch[..], BandPoolWork::Full)),
        },
        band_parallel,
        // FREE EXTRAS: the fused kernel carries an extra accumulator only
        // when a block that is OFF will finalize it (`ComputeSet::
        // free_work`). With the owning blocks on, their own kernels own
        // those slots and every flag is false, so the full 944 walk is
        // untouched.
        free,
    );
}

/// [`blockiness_sparse_rows`] over a strip's WIDE buffer: buffer row
/// `HALO_P + k` is plane row `y0 + k`. The horizontal family at plane row
/// `y0` (a lattice row whenever `y0 > 0` — kernel strips are 128-aligned)
/// reads its upper neighbor from the halo row `HALO_P − 1`, which is the
/// REAL plane row `y0 − 1` for every strip that can fire it (`y > 0`
/// never fires on the top strip's first row).
///
/// `sum_v`/`sum_h` are the CALLER's running totals, accumulated into
/// directly — NOT per-strip partials folded afterwards — so the global
/// f64 op sequence continues seamlessly across kernel strips and equals
/// [`blockiness_sparse_rows`]'s canonical whole-range form bit-for-bit
/// (a per-strip partial would reassociate: `(Σ strip0) + (Σ strip1)` ≠
/// one running sum — measured as a 1-ULP BLOCKINESS divergence when this
/// function briefly returned partials).
#[allow(clippy::too_many_arguments)]
fn blockiness_sparse_strip_wide(
    src_wide: &[f32],
    dst_wide: &[f32],
    width: usize,
    y0: usize,
    strip_h: usize,
    revision: FormulaRevision,
    sum_v: &mut crate::featcanon::SumVar,
    sum_h: &mut crate::featcanon::SumVar,
) {
    use crate::featcanon::{LanesF64, SumVar};
    let spread = revision.row_lanes();
    // Revision 7: the strip's own lane partials, added to the running lanes
    // once, so a strip's terms can be computed apart from the others (in
    // parallel) and merged in strip order with the same bits.
    let partial = spread && matches!((&*sum_v, &*sum_h), (SumVar::L64(_), SumVar::L64(_)));
    let (mut pv, mut ph) = (SumVar::L64(LanesF64::zero()), SumVar::L64(LanesF64::zero()));
    let (v, h) = if partial {
        (&mut pv, &mut ph)
    } else {
        (&mut *sum_v, &mut *sum_h)
    };
    for k in 0..strip_h {
        let y = y0 + k;
        blockiness_row(
            src_wide,
            dst_wide,
            (HALO_P + k) * width,
            width,
            y.is_multiple_of(BLOCK_LATTICE) && y > 0,
            spread,
            v,
            h,
        );
    }
    if partial {
        sum_v.merge_from(&pv);
        sum_h.merge_from(&ph);
    }
}

// ============================================================================
// REV4 C1 `gridblk` strip kernel (design note §C1).
//
// `ẽ = (|Δdst| − |Δsrc|) / ((act_a + act_b)/2 + C_ACTIVITY)` is computed
// for EVERY vertical boundary (x ∈ 1..w) and horizontal boundary
// (y ∈ 1..h) — not the sparse lattice-only subset `blockiness` visits,
// because the winning phase is data-dependent and the profile needs all
// `2n` boundaries. The winning phase is only known after the whole plane
// is walked, so the kernel STORES `ẽ` into the cell's `plane_v`/`plane_h`
// f32 buffers and `finish_gridblk_cell` rescans the on-grid positions
// only (~2n/P hat evaluations — the store-then-rescan shape).
//
// Wide-window contract (same as `blockiness_sparse_strip_wide`, extended
// by the activity plane): `*_wide` are `width * (strip_h + 2*HALO_P)`,
// buffer row `HALO_P + k` = plane row `y0 + k`. `act_wide` covers the
// identical wide window including halo rows (phase A fills all of
// `scratch.activity`), so the H boundary at plane row `y0` reads the REAL
// plane row `y0 − 1` from the halo, and the scalar/V8 paths index the
// same offsets.
//
// PROFILE DETERMINISM: the emitted values are defined by THIS kernel —
// per-phase `Σ|ẽ|`/`Σẽ` accumulate as: (a) two V8 lane pairs for the
// chunked V span (chunks start at x = 1 + 8c, so lane k of parity-j
// chunks covers positions ≡ (1 + 8j + k) mod period — the fold happens
// once per strip into the f64 phase totals); (b) per-row V8 accumulators
// for H (a row's boundaries share phase `y mod period`), lane-summed in
// fixed f32 order then added to f64; (c) scalar tails fold into the f64
// phase totals in row order. Every engine, tier and thread count replays
// the identical sequence, so the feature is bit-deterministic while its
// values carry f32-lane accumulation precision — the documented
// exception to the row-ordered-f64 convention, chosen because a
// per-boundary scalar scatter cannot meet the family's runtime budget.
// ============================================================================

/// Per-cell C1 accumulation entry — lazily sizes the `ẽ` planes and runs
/// the dispatched strip kernel. `plane_h` is the FULL plane height.
#[allow(clippy::too_many_arguments)]
fn gridblk_strip_wide(
    src_wide: &[f32],
    dst_wide: &[f32],
    act_wide: &[f32],
    width: usize,
    y0: usize,
    strip_h: usize,
    plane_h: usize,
    period: usize,
    acc: &mut GridblkAccum,
) {
    // rev4canon: the computation's own mode selects the scalar canon body —
    // `LanesF64` at Rev4; the oracle arms under `ZENSIM_FEATCANON`.
    // `compute_mode()` is `mode(comp_rev)` here (see it for the guarantee);
    // `Off` below Rev4 runs the shipped SIMD body byte-for-byte.
    let mode = crate::featcanon::compute_mode();
    if mode.active() {
        #[cfg(feature = "oracle")]
        use crate::featcanon::{LanesF32, Neum64};
        use crate::featcanon::{LanesF64, Mode};
        return match mode {
            Mode::Canon64 => gridblk_strip_wide_canon::<LanesF64>(
                src_wide, dst_wide, act_wide, width, y0, strip_h, plane_h, period, acc,
            ),
            #[cfg(feature = "oracle")]
            Mode::Canon32 => gridblk_strip_wide_canon::<LanesF32>(
                src_wide, dst_wide, act_wide, width, y0, strip_h, plane_h, period, acc,
            ),
            #[cfg(feature = "oracle")]
            Mode::CanonNeum => gridblk_strip_wide_canon::<Neum64>(
                src_wide, dst_wide, act_wide, width, y0, strip_h, plane_h, period, acc,
            ),
            #[cfg(feature = "oracle")]
            Mode::Exact => gridblk_strip_wide_exact(
                src_wide, dst_wide, act_wide, width, y0, strip_h, plane_h, period, acc,
            ),
            Mode::Off => unreachable!("active() excludes Off"),
        };
    }
    incant!(
        gridblk_strip_wide_entry(
            src_wide, dst_wide, act_wide, width, y0, strip_h, plane_h, period, acc
        ),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

#[allow(clippy::too_many_arguments)]
#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
fn gridblk_strip_wide_entry(
    token: Token,
    src_wide: &[f32],
    dst_wide: &[f32],
    act_wide: &[f32],
    width: usize,
    y0: usize,
    strip_h: usize,
    plane_h: usize,
    period: usize,
    acc: &mut GridblkAccum,
) {
    gridblk_strip_wide_generic::<_>(
        token, src_wide, dst_wide, act_wide, width, y0, strip_h, plane_h, period, acc,
    )
}

#[allow(clippy::too_many_arguments)]
fn gridblk_strip_wide_generic<T: F32x8Backend + Copy>(
    token: T,
    src_wide: &[f32],
    dst_wide: &[f32],
    act_wide: &[f32],
    width: usize,
    y0: usize,
    strip_h: usize,
    plane_h: usize,
    period: usize,
    acc: &mut GridblkAccum,
) {
    debug_assert!((2..=GRIDBLK_MAX_PERIOD).contains(&period));
    debug_assert!(period.is_power_of_two());
    debug_assert_eq!(src_wide.len(), width * (strip_h + 2 * HALO_P));
    debug_assert_eq!(dst_wide.len(), src_wide.len());
    debug_assert_eq!(act_wide.len(), src_wide.len());
    let n_plane = width * plane_h;
    if acc.plane_v.len() != n_plane {
        acc.plane_v.resize(n_plane, 0.0);
        acc.plane_h.resize(n_plane, 0.0);
    }
    let half = V8::<T>::splat_t(token, 0.5);
    let c_act = V8::<T>::splat_t(token, C_ACTIVITY as f32);
    // Lane-phase accumulators for the chunked V span — folded to f64 once
    // per strip (see module doc above for the lane→phase map).
    let mut acc_v_abs = [V8::<T>::zero_t(token), V8::<T>::zero_t(token)];
    let mut acc_v_sgn = [V8::<T>::zero_t(token), V8::<T>::zero_t(token)];
    macro_rules! ld_at {
        ($plane:expr, $off:expr) => {
            V8::<T>::from_array_t(token, $plane[$off..$off + 8].try_into().unwrap())
        };
    }
    for k in 0..strip_h {
        let y = y0 + k;
        let row = (HALO_P + k) * width;
        let prow = y * width;

        // ---- Vertical boundaries (x ∈ 1..width), phase `x mod period`.
        // Full chunks from x = 1 step 8 keep a fixed lane→phase map.
        let interior_w = width - 1;
        let chunk_end = 1 + interior_w - (interior_w % 8);
        let mut x = 1usize;
        let mut c = 0usize;
        while x < chunk_end {
            let i = row + x;
            let e = (ld_at!(dst_wide, i) - ld_at!(dst_wide, i - 1)).abs()
                - (ld_at!(src_wide, i) - ld_at!(src_wide, i - 1)).abs();
            let norm = (ld_at!(act_wide, i - 1) + ld_at!(act_wide, i)) * half + c_act;
            let et = e / norm;
            et.store(
                (&mut acc.plane_v[prow + x..prow + x + 8])
                    .try_into()
                    .unwrap(),
            );
            let j = c & 1;
            acc_v_abs[j] += et.abs();
            acc_v_sgn[j] += et;
            x += 8;
            c += 1;
        }
        while x < width {
            let i = row + x;
            let e = (dst_wide[i] - dst_wide[i - 1]).abs() - (src_wide[i] - src_wide[i - 1]).abs();
            let et = e / ((act_wide[i - 1] + act_wide[i]) * 0.5 + C_ACTIVITY as f32);
            acc.plane_v[prow + x] = et;
            let p = x % period;
            acc.abs_v[p] += et.abs() as f64;
            acc.signed_v[p] += et as f64;
            x += 1;
        }

        // ---- Horizontal boundaries (row y vs y−1), phase `y mod period`.
        // `y == 0` never fires: the lattice has no boundary above the
        // plane. For interior strips the halo row is the real `y − 1`.
        if y > 0 {
            let ph = y % period;
            let row_u = row - width;
            let mut acc_h_abs = V8::<T>::zero_t(token);
            let mut acc_h_sgn = V8::<T>::zero_t(token);
            let chunk_end_h = width - (width % 8);
            let mut x = 0usize;
            while x < chunk_end_h {
                let i = row + x;
                let e = (ld_at!(dst_wide, i) - ld_at!(dst_wide, row_u + x)).abs()
                    - (ld_at!(src_wide, i) - ld_at!(src_wide, row_u + x)).abs();
                let norm = (ld_at!(act_wide, row_u + x) + ld_at!(act_wide, i)) * half + c_act;
                let et = e / norm;
                et.store(
                    (&mut acc.plane_h[prow + x..prow + x + 8])
                        .try_into()
                        .unwrap(),
                );
                acc_h_abs += et.abs();
                acc_h_sgn += et;
                x += 8;
            }
            let mut t_abs = 0.0f32;
            let mut t_sgn = 0.0f32;
            while x < width {
                let i = row + x;
                let iu = row_u + x;
                let e = (dst_wide[i] - dst_wide[iu]).abs() - (src_wide[i] - src_wide[iu]).abs();
                let et = e / ((act_wide[iu] + act_wide[i]) * 0.5 + C_ACTIVITY as f32);
                acc.plane_h[prow + x] = et;
                t_abs += et.abs();
                t_sgn += et;
                x += 1;
            }
            let a = acc_h_abs.to_array();
            let s = acc_h_sgn.to_array();
            let row_abs = a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7] + t_abs;
            let row_sgn = s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + s[7] + t_sgn;
            acc.abs_h[ph] += row_abs as f64;
            acc.signed_h[ph] += row_sgn as f64;
        }
    }
    // Strip-end lane fold — phase (1 + 8j + k) mod period per doc above.
    for (j, (av, sv)) in acc_v_abs.iter().zip(acc_v_sgn.iter()).enumerate() {
        let a = av.to_array();
        let s = sv.to_array();
        for k in 0..8 {
            let p = (1 + 8 * j + k) % period;
            acc.abs_v[p] += a[k] as f64;
            acc.signed_v[p] += s[k] as f64;
        }
    }
}

// featacc: measurement-mode `gridblk` bodies. The canon body computes `ẽ`
// in scalar f32 (the SIMD body's op order — `(a−b).abs() − (c−d).abs()`,
// `(act_lo + act_hi) * 0.5 + C`, one division), stores the f32 plane, and
// accumulates the per-phase profiles through per-strip `Pool`s at
// canonical lane `x mod 8` (V: one pair of pools per phase, folded once
// per strip like the production lane-parity fold; H: one pool pair per
// row, folded per row like production). `exact` evaluates `ẽ` in f64 into
// the `plane_*64` shadow planes and pools through `Neum64`.
#[allow(clippy::too_many_arguments)]
fn gridblk_strip_wide_canon<P: crate::featcanon::Pool>(
    src_wide: &[f32],
    dst_wide: &[f32],
    act_wide: &[f32],
    width: usize,
    y0: usize,
    strip_h: usize,
    plane_h: usize,
    period: usize,
    acc: &mut GridblkAccum,
) {
    let n_plane = width * plane_h;
    if acc.plane_v.len() != n_plane {
        acc.plane_v.resize(n_plane, 0.0);
        acc.plane_h.resize(n_plane, 0.0);
    }
    let mut pv_abs = [P::zero(); GRIDBLK_MAX_PERIOD];
    let mut pv_sgn = [P::zero(); GRIDBLK_MAX_PERIOD];
    for k in 0..strip_h {
        let y = y0 + k;
        let row = (HALO_P + k) * width;
        let prow = y * width;
        for x in 1..width {
            let i = row + x;
            let e = (dst_wide[i] - dst_wide[i - 1]).abs() - (src_wide[i] - src_wide[i - 1]).abs();
            let et = e / ((act_wide[i - 1] + act_wide[i]) * 0.5 + C_ACTIVITY as f32);
            acc.plane_v[prow + x] = et;
            let p = x % period;
            pv_abs[p].add(x & 7, et.abs());
            pv_sgn[p].add(x & 7, et);
        }
        if y > 0 {
            let ph = y % period;
            let row_u = row - width;
            let mut h_abs = P::zero();
            let mut h_sgn = P::zero();
            for x in 0..width {
                let i = row + x;
                let iu = row_u + x;
                let e = (dst_wide[i] - dst_wide[iu]).abs() - (src_wide[i] - src_wide[iu]).abs();
                let et = e / ((act_wide[iu] + act_wide[i]) * 0.5 + C_ACTIVITY as f32);
                acc.plane_h[prow + x] = et;
                h_abs.add(x & 7, et.abs());
                h_sgn.add(x & 7, et);
            }
            acc.abs_h[ph] += h_abs.fin();
            acc.signed_h[ph] += h_sgn.fin();
        }
    }
    for p in 0..period {
        acc.abs_v[p] += pv_abs[p].fin();
        acc.signed_v[p] += pv_sgn[p].fin();
    }
}

#[cfg(feature = "oracle")]
#[allow(clippy::too_many_arguments)]
fn gridblk_strip_wide_exact(
    src_wide: &[f32],
    dst_wide: &[f32],
    act_wide: &[f32],
    width: usize,
    y0: usize,
    strip_h: usize,
    plane_h: usize,
    period: usize,
    acc: &mut GridblkAccum,
) {
    use crate::featcanon::{Neum64, Pool as _};
    let n_plane = width * plane_h;
    if acc.plane_v64.len() != n_plane {
        acc.plane_v64.resize(n_plane, 0.0);
        acc.plane_h64.resize(n_plane, 0.0);
    }
    let mut pv_abs = [Neum64::zero(); GRIDBLK_MAX_PERIOD];
    let mut pv_sgn = [Neum64::zero(); GRIDBLK_MAX_PERIOD];
    for k in 0..strip_h {
        let y = y0 + k;
        let row = (HALO_P + k) * width;
        let prow = y * width;
        for x in 1..width {
            let i = row + x;
            let e = (dst_wide[i] as f64 - dst_wide[i - 1] as f64).abs()
                - (src_wide[i] as f64 - src_wide[i - 1] as f64).abs();
            let et = e / ((act_wide[i - 1] as f64 + act_wide[i] as f64) * 0.5 + C_ACTIVITY);
            acc.plane_v64[prow + x] = et;
            let p = x % period;
            pv_abs[p].add64(x & 7, et.abs());
            pv_sgn[p].add64(x & 7, et);
        }
        if y > 0 {
            let ph = y % period;
            let row_u = row - width;
            let mut h_abs = Neum64::zero();
            let mut h_sgn = Neum64::zero();
            for x in 0..width {
                let i = row + x;
                let iu = row_u + x;
                let e = (dst_wide[i] as f64 - dst_wide[iu] as f64).abs()
                    - (src_wide[i] as f64 - src_wide[iu] as f64).abs();
                let et = e / ((act_wide[iu] as f64 + act_wide[i] as f64) * 0.5 + C_ACTIVITY);
                acc.plane_h64[prow + x] = et;
                h_abs.add64(x & 7, et.abs());
                h_sgn.add64(x & 7, et);
            }
            acc.abs_h[ph] += h_abs.fin();
            acc.signed_h[ph] += h_sgn.fin();
        }
    }
    for p in 0..period {
        acc.abs_v[p] += pv_abs[p].fin();
        acc.signed_v[p] += pv_sgn[p].fin();
    }
}

// ============================================================================
// REV4 C4 `arttype` bleed mask builder (design note §C4).
//
// The mask marks every pixel within Chebyshev distance 1 of a dst-Y
// edge — where "edge" is the gradient kernel's own test
// (`(gx² + gy²).sqrt() >= C_RING_EDGE`, x-neighbours column-clamped,
// y-neighbours the caller-supplied rows). Built once per strip before
// the channel fan-out so the serial path (X/B processed before Y) has it
// ready for the chroma gradient hooks.
//
// `dh` holds the dst-Y plane rows `reflect_101(y0 − 2 + r, plane_h)` for
// `r ∈ 0..strip_h + 4`, fetched by the caller (producer `rows` in the
// streaming walk, `gather_strip_halo` in the materialized path). The
// edge formula is evaluated in V8 f32 uniformly at every position —
// the deterministic definition; scalar tails use the identical f32 op
// sequence. `mul_add` is deliberately NOT used (its FMA fusion is
// tier-dependent — the same reason the production kernels spell
// `a*b + c` out).
// ============================================================================

/// Per-strip C4 mask scratch (sized by the caller once per walk).
#[derive(Default)]
struct BleedMaskWork {
    /// Dst-Y rows for `reflect_101(y0 − 2 + r)`, r ∈ 0..strip_h+4.
    dh: Vec<f32>,
    /// One raw edge-flag row (width f32, {0,1}).
    e_row: Vec<f32>,
    /// Horizontally-dilated edge rows j ∈ 0..strip_h+2 (E at plane row
    /// `y0 − 1 + j`, zero when that row is out of plane).
    ehd: Vec<f32>,
    /// The finished mask, `width * strip_h`, {0.0, 1.0}.
    mask: Vec<f32>,
}

impl BleedMaskWork {
    /// Size all buffers for `width` × strips of at most `max_strip_h`.
    fn sized(width: usize, max_strip_h: usize) -> Self {
        Self {
            dh: vec![0.0; width * (max_strip_h + 4)],
            e_row: vec![0.0; width],
            ehd: vec![0.0; width * (max_strip_h + 2)],
            mask: vec![0.0; width * max_strip_h],
        }
    }
}

/// Build the strip's dilated dst-Y edge mask into `mw.mask[..n_strip]`.
#[allow(clippy::too_many_arguments)]
fn dst_y_edge_mask_strip(
    width: usize,
    y0: usize,
    strip_h: usize,
    plane_h: usize,
    mw: &mut BleedMaskWork,
) {
    incant!(
        dst_y_edge_mask_entry(
            &mw.dh[..width * (strip_h + 4)],
            width,
            y0,
            strip_h,
            plane_h,
            &mut mw.e_row[..width],
            &mut mw.ehd[..width * (strip_h + 2)],
            &mut mw.mask[..width * strip_h]
        ),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

#[allow(clippy::too_many_arguments)]
#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
fn dst_y_edge_mask_entry(
    token: Token,
    dh: &[f32],
    width: usize,
    y0: usize,
    strip_h: usize,
    plane_h: usize,
    e_row: &mut [f32],
    ehd: &mut [f32],
    mask: &mut [f32],
) {
    dst_y_edge_mask_generic::<_>(token, dh, width, y0, strip_h, plane_h, e_row, ehd, mask)
}

#[allow(clippy::too_many_arguments)]
fn dst_y_edge_mask_generic<T: F32x8Backend + Copy>(
    token: T,
    dh: &[f32],
    width: usize,
    y0: usize,
    strip_h: usize,
    plane_h: usize,
    e_row: &mut [f32],
    ehd: &mut [f32],
    mask: &mut [f32],
) {
    debug_assert_eq!(dh.len(), width * (strip_h + 4));
    debug_assert_eq!(ehd.len(), width * (strip_h + 2));
    debug_assert_eq!(mask.len(), width * strip_h);
    let zero = V8::<T>::zero_t(token);
    let one = V8::<T>::splat_t(token, 1.0);
    let c_edge = V8::<T>::splat_t(token, C_RING_EDGE as f32);
    macro_rules! ld_at {
        ($plane:expr, $off:expr) => {
            V8::<T>::from_array_t(token, $plane[$off..$off + 8].try_into().unwrap())
        };
    }
    // Scalar edge flag at plane row py = y0−1+j, column x — x-neighbours
    // clamped, f32 arithmetic identical to the V8 lanes.
    let edge_at = |j: usize, x: usize| -> f32 {
        let xl = x.saturating_sub(1);
        let xr = (x + 1).min(width - 1);
        let (ru, rc, rd) = (j * width, (j + 1) * width, (j + 2) * width);
        let gx = dh[rc + xr] - dh[rc + xl];
        let gy = dh[rd + x] - dh[ru + x];
        if (gx * gx + gy * gy).sqrt() >= C_RING_EDGE as f32 {
            1.0
        } else {
            0.0
        }
    };
    // Pass 1: edge rows (hdilated) into `ehd`. Row j is E at plane row
    // `y0 − 1 + j`; rows outside the plane stay zero — the plane-clamped
    // dilation of the spec falls out for free.
    for j in 0..strip_h + 2 {
        let py = y0 as isize + j as isize - 1;
        let ehd_row = &mut ehd[j * width..(j + 1) * width];
        if py < 0 || py >= plane_h as isize {
            ehd_row.fill(0.0);
            continue;
        }
        let (ru, rc, rd) = (j * width, (j + 1) * width, (j + 2) * width);
        let mut x = 0usize;
        if width > 2 {
            e_row[0] = edge_at(j, 0);
            let interior_end = width - 1;
            let interior_w = interior_end - 1;
            let chunk_end = 1 + interior_w - (interior_w % 8);
            x = 1;
            while x < chunk_end {
                let gx = ld_at!(dh, rc + x + 1) - ld_at!(dh, rc + x - 1);
                let gy = ld_at!(dh, rd + x) - ld_at!(dh, ru + x);
                let m = (gx * gx + gy * gy).sqrt();
                let flag = V8::<T>::blend(m.simd_ge(c_edge), one, zero);
                flag.store((&mut e_row[x..x + 8]).try_into().unwrap());
                x += 8;
            }
            while x < interior_end {
                e_row[x] = edge_at(j, x);
                x += 1;
            }
            e_row[width - 1] = edge_at(j, width - 1);
        } else {
            while x < width {
                e_row[x] = edge_at(j, x);
                x += 1;
            }
        }
        // Horizontal 3-tap dilation, column-clamped — pure max ops, so
        // the V8 and scalar shapes emit identical bits.
        if width == 1 {
            ehd_row[0] = e_row[0];
        } else if width == 2 {
            let m = e_row[0].max(e_row[1]);
            ehd_row[0] = m;
            ehd_row[1] = m;
        } else {
            ehd_row[0] = e_row[0].max(e_row[1]);
            let interior_w = width - 2;
            let chunk_end = 1 + interior_w - (interior_w % 8);
            let mut x = 1usize;
            while x < chunk_end {
                let m = V8::<T>::from_array_t(token, e_row[x - 1..x + 7].try_into().unwrap())
                    .max(V8::<T>::from_array_t(
                        token,
                        e_row[x..x + 8].try_into().unwrap(),
                    ))
                    .max(V8::<T>::from_array_t(
                        token,
                        e_row[x + 1..x + 9].try_into().unwrap(),
                    ));
                m.store((&mut ehd_row[x..x + 8]).try_into().unwrap());
                x += 8;
            }
            while x < width - 1 {
                ehd_row[x] = e_row[x - 1].max(e_row[x]).max(e_row[x + 1]);
                x += 1;
            }
            ehd_row[width - 1] = e_row[width - 2].max(e_row[width - 1]);
        }
    }
    // Pass 2: vertical 3-tap dilation — mask row k = max(ehd[k..k+3]),
    // the ±1-row neighbourhood (out-of-plane ehd rows are already zero).
    for k in 0..strip_h {
        let (a, b, c) = (k * width, (k + 1) * width, (k + 2) * width);
        let mrow = &mut mask[k * width..(k + 1) * width];
        let mut x = 0usize;
        while x + 8 <= width {
            let m = ld_at!(ehd, a + x)
                .max(ld_at!(ehd, b + x))
                .max(ld_at!(ehd, c + x));
            m.store((&mut mrow[x..x + 8]).try_into().unwrap());
            x += 8;
        }
        while x < width {
            mrow[x] = ehd[a + x].max(ehd[b + x]).max(ehd[c + x]);
            x += 1;
        }
    }
}

// Wide kernels still consult process arithmetic. Never feed them direct-error
// moments prepared under a different explicit revision. Basic-only plans carry
// their revision through both passes and remain safe to serve side by side —
// EXCEPT across the Rev4 boundary (featcanon D1): the basic path's other
// formula gates still read the process switch, so a v1-only Rev4 request in a
// Rev3 process (or the reverse) would be labelled one revision and computed
// as another.
fn validate_wide_revision(toggles: V2NewFeatureToggles) -> Result<(), ZensimError> {
    crate::ssim_form::refuse_rev4_mix(toggles.formula_revision)?;
    if !toggles.v1_only && toggles.formula_revision != crate::ssim_form::active_revision() {
        return Err(ZensimError::ModelLoadFailed {
            reason: "wide feature extraction requires matching process and requested formula revisions",
        });
    }
    Ok(())
}

/// Raw toggles request computation, unlike the layout flags of a validated
/// plan. Refuse unsupported families before `rev5_scope` can narrow work.
fn validate_raw_rev5_request(t: V2NewFeatureToggles) -> Result<(), ZensimError> {
    // `globaff` sits past the whole optional-block layout chain, which no raw
    // Rev5+ request may carry; only a validated revision-6 plan reaches it.
    if t.globaff {
        return Err(ZensimError::ModelLoadFailed {
            reason: "the globaff family is served only through a validated formula revision 6 plan",
        });
    }
    if t.formula_revision >= FormulaRevision::Rev5
        && (matches!(t.v1_pools, V1PoolsMode::Full | V1PoolsMode::Carriers)
            || t.append_block
            || t.append2_block
            || t.append2_dst_activity
            || t.csfw_block
            || t.rev4_gridblk
            || t.rev4_ringbasis
            || t.rev4_tailhist
            || t.rev4_arttype
            || t.gmsbank
            || t.mapdev
            || t.z1max
            || t.gmsnative
            || t.free_extras != V1FreeExtras::Off)
    {
        return Err(ZensimError::ModelLoadFailed {
            reason: "formula revision 5 supports only basic + peaks + v2; raw extraction requested an unsupported family",
        });
    }
    Ok(())
}

/// Streaming folded-720[+append] pair entry: validation + sub-64
/// reflect-pad exactly like the materialized pair entry
/// ([`compute_folded720_impl_with_toggles`] → prepare → with-ref inner),
/// then the streaming walk. Output is bitwise equal to the materialized
/// path (`streamed_foldapp_bitwise_vs_materialized`).
pub(crate) fn compute_folded720_streaming_impl(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    toggles: V2NewFeatureToggles,
    scratch: &mut V2Scratch,
    planned_compute: Option<ComputeSet>,
) -> Result<ZensimV2Result, ZensimError> {
    compute_folded720_streaming_extras(
        source,
        distorted,
        max_pixels,
        parallel,
        toggles,
        scratch,
        FoldWalkExtras {
            compute: planned_compute,
            ..Default::default()
        },
    )
}

/// [`compute_folded720_streaming_impl`] with the caller's
/// [`FoldWalkExtras`] carried into the walk — the training surface's
/// validation, HDR routing and padding; the extras are the only delta.
pub(crate) fn compute_folded720_streaming_extras(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    toggles: V2NewFeatureToggles,
    scratch: &mut V2Scratch,
    extras: FoldWalkExtras<'_>,
) -> Result<ZensimV2Result, ZensimError> {
    if !extras
        .compute
        .is_some_and(|c| c.formula_revision >= FormulaRevision::Rev5)
    {
        validate_wide_revision(toggles)?;
    }
    if extras.compute.is_none() {
        validate_raw_rev5_request(toggles)?;
    }
    crate::metric::validate_pair_dims(source, distorted)?;
    crate::metric::check_within_max_pixels(source.width(), source.height(), max_pixels)?;
    // HDR routing (HDR_PLAN chunk 2) in the exact position the
    // unconditional `reject_hdr_input` calls held: a pair of
    // HDR-declared sources whose pixels are ABSOLUTE LINEAR cd/m²
    // (`LinearF32Rgba` + `is_hdr()`, the `compute_pu_linear` contract)
    // routes to the PU front-end with `HdrEncoding::Linear`. Every other
    // HDR-flagged shape (code-value formats without a declared transfer,
    // translucent alpha, mixed SDR/HDR pairs) still gets
    // `HdrInputRequiresPuPath` — the reject remains for every path the
    // HDR validation did not cover. sRGB sources take the identical
    // SDR path as before this routing existed (byte-stability gate).
    if source.is_hdr() || distorted.is_hdr() {
        fn linear_ok(
            is_hdr: bool,
            fmt: crate::source::PixelFormat,
            alpha: crate::source::AlphaMode,
        ) -> bool {
            is_hdr
                && fmt == crate::source::PixelFormat::LinearF32Rgba
                && matches!(alpha, crate::source::AlphaMode::Opaque)
        }
        if linear_ok(source.is_hdr(), source.pixel_format(), source.alpha_mode())
            && linear_ok(
                distorted.is_hdr(),
                distorted.pixel_format(),
                distorted.alpha_mode(),
            )
        {
            return compute_folded720_hdr_streaming_extras(
                source,
                distorted,
                HdrEncoding::Linear,
                max_pixels,
                parallel,
                toggles,
                scratch,
                extras,
            );
        }
        return Err(ZensimError::HdrInputRequiresPuPath);
    }
    if source.width() < crate::metric::MIN_PYRAMID_DIM
        || source.height() < crate::metric::MIN_PYRAMID_DIM
    {
        let padded_src = crate::metric::reflect_pad_to_min(source);
        let padded_dst = crate::metric::reflect_pad_to_min(distorted);
        return Ok(foldapp_streaming_walk(
            &padded_src,
            &padded_dst,
            parallel,
            toggles,
            crate::feature_v2_stream::FrontEnd::Sdr,
            scratch,
            extras,
        ));
    }
    Ok(foldapp_streaming_walk(
        source,
        distorted,
        parallel,
        toggles,
        crate::feature_v2_stream::FrontEnd::Sdr,
        scratch,
        extras,
    ))
}

/// **The fold-backed v1-372 extraction** (fold-engine lane, 2026-08-30):
/// the streaming walk asked for exactly what a v1 score needs and nothing
/// else — `v1_only` (skip every v2-era block *and* its upstream V-blur /
/// activity work; pure compute-skipping, gated by
/// `free_extras_are_pure_addition_to_the_v1_only_walk`) plus [`V1PoolsMode::Full`] (emit
/// `f156..372` live). Returns `(features, mean_offset)` where
/// `features[0..372]` is **bit-identical to v1's 372-feature extraction** at
/// every width under option C — `v1_372_bit_exact_to_fold_at_every_width` is
/// the gate — and `mean_offset` reproduces
/// `streaming::compute_xyb_mean_offset` bit-exactly (see [`MeanOffsetRows`]).
///
/// The returned vector is the requesting regime's WIDTH (720 here) with
/// `f372..` at the structural `0.0`; the caller truncates to its config's
/// v1 width, which is a prefix (`[0,156)` basic · `[156,228)` peaks ·
/// `[228,300)` masked · `[300,372)` IW).
///
/// SDR only. A declared-HDR pair returns
/// [`ZensimError::HdrInputRequiresPuPath`] — the PU entries keep the
/// buffered walk until the fold's PU front-end grows a matching
/// mean-offset path.
pub(crate) fn compute_folded_v1_372_streaming_impl(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    scratch: &mut V2Scratch,
    // The serving PLAN, resolved by `fold_engine::score_plan` from the
    // profile's bakes. `None` = the unconditional pre-2026-08-31 behaviour
    // (`V1PoolsMode::Full`, v1-only, v1 layout) — spelled out below as a
    // literal rather than reconstructed from a plan, so the default path is
    // the SAME code, not merely an equivalent one.
    plan: Option<&crate::feature_plan::Plan>,
    #[cfg(feature = "custom-profiles")] retention: Option<&mut FoldRetention>,
) -> Result<(Vec<f64>, [f64; 3]), ZensimError> {
    compute_folded_v1_372_streaming_stoppable(
        source,
        distorted,
        max_pixels,
        parallel,
        scratch,
        plan,
        #[cfg(feature = "custom-profiles")]
        retention,
        None,
    )
}

pub(crate) fn compute_folded_v1_372_streaming_stoppable(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    scratch: &mut V2Scratch,
    // The serving PLAN, resolved by `fold_engine::score_plan` from the
    // profile's bakes. `None` = the unconditional pre-2026-08-31 behaviour
    // (`V1PoolsMode::Full`, v1-only, v1 layout) — spelled out below as a
    // literal rather than reconstructed from a plan, so the default path is
    // the SAME code, not merely an equivalent one.
    plan: Option<&crate::feature_plan::Plan>,
    #[cfg(feature = "custom-profiles")] retention: Option<&mut FoldRetention>,
    stop: Option<&dyn enough::Stop>,
) -> Result<(Vec<f64>, [f64; 3]), ZensimError> {
    crate::metric::validate_pair_dims(source, distorted)?;
    crate::metric::check_within_max_pixels(source.width(), source.height(), max_pixels)?;
    if source.is_hdr() || distorted.is_hdr() {
        return Err(ZensimError::HdrInputRequiresPuPath);
    }
    let toggles = match plan {
        // A plan asks for its own (compute, layout) pair — which is how a
        // bake declaring a layout wider than 372 becomes servable at all.
        Some(p) => p.toggles(),
        None => V2NewFeatureToggles {
            v1_pools: V1PoolsMode::Full,
            v1_only: true,
            ..V2NewFeatureToggles::default()
        },
    };
    if let Some(p) = plan
        && let Some(sampling) = p.compute.sampling
    {
        let dims = sampling.dims(source.width(), source.height());
        let mut mo = MeanOffsetRows::at_revision(dims[0].0, dims[0].1, toggles.formula_revision);
        let res = foldapp_streaming_walk(
            source,
            distorted,
            parallel,
            toggles,
            crate::feature_v2_stream::FrontEnd::Sdr,
            scratch,
            FoldWalkExtras {
                stop,
                compute: Some(p.compute),
                mean_offset: Some(&mut mo),
                #[cfg(feature = "custom-profiles")]
                retention,
                ..Default::default()
            },
        );
        return Ok((res.into_features(), mo.finish()));
    }
    // Sub-64 reflect-pad BEFORE the walk, exactly as
    // `metric::compute_with_config_inner` does — so both engines' features
    // AND mean_offset are taken over the same padded plane. The two arms are
    // written out rather than closed over because the padded and unpadded
    // sides are different `ImageSource` types.
    if source.width() < crate::metric::MIN_PYRAMID_DIM
        || source.height() < crate::metric::MIN_PYRAMID_DIM
    {
        let padded_src = crate::metric::reflect_pad_to_min(source);
        let padded_dst = crate::metric::reflect_pad_to_min(distorted);
        let mut mo = MeanOffsetRows::at_revision(
            padded_src.width(),
            padded_src.height(),
            toggles.formula_revision,
        );
        let res = foldapp_streaming_walk(
            &padded_src,
            &padded_dst,
            parallel,
            toggles,
            crate::feature_v2_stream::FrontEnd::Sdr,
            scratch,
            FoldWalkExtras {
                stop,
                compute: plan.map(|p| p.compute),
                mean_offset: Some(&mut mo),
                #[cfg(feature = "custom-profiles")]
                retention,
                ..Default::default()
            },
        );
        return Ok((res.into_features(), mo.finish()));
    }
    let mut mo =
        MeanOffsetRows::at_revision(source.width(), source.height(), toggles.formula_revision);
    let res = foldapp_streaming_walk(
        source,
        distorted,
        parallel,
        toggles,
        crate::feature_v2_stream::FrontEnd::Sdr,
        scratch,
        FoldWalkExtras {
            stop,
            compute: plan.map(|p| p.compute),
            mean_offset: Some(&mut mo),
            #[cfg(feature = "custom-profiles")]
            retention,
            ..Default::default()
        },
    );
    Ok((res.into_features(), mo.finish()))
}

/// Whether a [`crate::streaming::PrecomputedReference`] can feed the fold's
/// source side for a walk over `(w, h)` COMPUTE dims — the ref-cached form's
/// admission test, and **the single owner of that predicate**.
///
/// Requires the cache to hold exactly [`crate::NUM_SCALES`] levels whose
/// dimensions match the fold's own floor-halving recurrence and whose planes
/// are tightly packed at each level's width. `PrecomputedReference` stops its
/// pyramid at `w < 8 || h < 8`, so a cache with fewer levels (or built at
/// different dims) is refused and the caller re-derives from the image
/// instead of scoring a mismatched pyramid.
pub(crate) fn cached_ref_feed_usable(
    scales: &[crate::streaming::XybPyramidLevel],
    w: usize,
    h: usize,
) -> bool {
    if scales.len() != crate::NUM_SCALES {
        return false;
    }
    let (mut sw, mut sh) = (w, h);
    for (planes, cw, ch) in scales.iter() {
        if *cw != sw || *ch != sh {
            return false;
        }
        if planes.iter().any(|p| p.len() != sw * sh) {
            return false;
        }
        sw /= 2;
        sh /= 2;
    }
    true
}

/// **The ref-cached fold-backed v1-372 extraction** (fold-engine lane stage
/// 3): [`compute_folded_v1_372_streaming_impl`] with the source side fed from
/// an already-built XYB pyramid, so N distorted candidates amortise one
/// reference's decode + sRGB→XYB conversion + 3-level downscale chain — the
/// shape `Zensim::precompute_reference` / `compute_with_ref` has always had on
/// the buffered path.
///
/// `distorted` must already be at the cache's COMPUTE dims (the caller
/// reflect-pads a sub-64 distorted, exactly as `Zensim::compute_with_ref`
/// does). Returns `None` when the cache cannot feed the fold
/// ([`cached_ref_feed_usable`]) — the caller then falls back rather than
/// scoring a different pyramid.
///
/// Bit-identical to N independent [`compute_folded_v1_372_streaming_impl`]
/// calls; `fold_ref_cache_matches_independent_computes` is the gate.
#[cfg(any(test, feature = "custom-profiles"))]
pub(crate) fn compute_folded_v1_372_with_ref_impl(
    precomputed: &crate::streaming::PrecomputedReference,
    distorted: &impl ImageSource,
    parallel: bool,
    scratch: &mut V2Scratch,
    pool_mode: Option<V1PoolsMode>,
    plan: Option<&crate::feature_plan::Plan>,
) -> Option<(Vec<f64>, [f64; 3])> {
    compute_folded_v1_372_with_ref_stoppable(
        precomputed,
        distorted,
        parallel,
        scratch,
        pool_mode,
        plan,
        None,
    )
}

pub(crate) fn compute_folded_v1_372_with_ref_stoppable(
    precomputed: &crate::streaming::PrecomputedReference,
    distorted: &impl ImageSource,
    parallel: bool,
    scratch: &mut V2Scratch,
    pool_mode: Option<V1PoolsMode>,
    plan: Option<&crate::feature_plan::Plan>,
    stop: Option<&dyn enough::Stop>,
) -> Option<(Vec<f64>, [f64; 3])> {
    if !ref_feed_admits(precomputed, distorted, plan) {
        return None;
    }
    let (cw, ch) = (precomputed.scales[0].1, precomputed.scales[0].2);
    let toggles = plan.map_or(
        V2NewFeatureToggles {
            v1_pools: pool_mode.unwrap_or(V1PoolsMode::Full),
            v1_only: true,
            ..V2NewFeatureToggles::default()
        },
        crate::feature_plan::Plan::toggles,
    );
    let mut mo = MeanOffsetRows::at_revision(cw, ch, toggles.formula_revision);
    // `source` is unused by the producer on the cached feed (every source-side
    // row is copied from the cache); it is still the type parameter, so pass
    // the distorted image and let the feed override it. The producer's
    // `ref_planes` branch is taken for EVERY side-0 fill, so no pixel of this
    // argument is ever read for side 0.
    let res = foldapp_streaming_walk(
        distorted,
        distorted,
        parallel,
        toggles,
        crate::feature_v2_stream::FrontEnd::Sdr,
        scratch,
        FoldWalkExtras {
            stop,
            mean_offset: Some(&mut mo),
            ref_planes: Some(&precomputed.scales),
            compute: plan.map(|p| p.compute),
            ..Default::default()
        },
    );
    Some((res.into_features(), mo.finish()))
}

/// Whether `precomputed` can feed the source side of a fold walk over
/// `distorted` under `plan` — the ref-cached form's admission test, shared by
/// [`compute_folded_v1_372_with_ref_impl`] and the retaining
/// [`compute_folded_with_ref_retained`].
///
/// Refuses a sampled cache or plan, a pre-Rev5 v2 plan, HDR input, a
/// distorted image not at the cache's compute dims, a cache that fails
/// [`cached_ref_feed_usable`], and a plan with a restored-cut side pass
/// (`mapdev` / `z1max`): those rebuild both pyramids from the source pixels,
/// which a ref-fed walk does not have (the walk asserts on that combination).
pub(crate) fn ref_feed_admits(
    precomputed: &crate::streaming::PrecomputedReference,
    distorted: &impl ImageSource,
    plan: Option<&crate::feature_plan::Plan>,
) -> bool {
    if precomputed.sampling.is_some()
        || plan.is_some_and(|p| {
            p.compute.sampling.is_some()
                || (p.compute.v2_blocks && p.compute.formula_revision < FormulaRevision::Rev5)
                || p.compute.mapdev
                || p.compute.z1max
        })
        || distorted.is_hdr()
    {
        return false;
    }
    let Some(&(_, cw, ch)) = precomputed.scales.first() else {
        return false;
    };
    distorted.width() == cw
        && distorted.height() == ch
        && cached_ref_feed_usable(&precomputed.scales, cw, ch)
}

#[cfg(all(test, feature = "custom-profiles"))]
thread_local! {
    /// Test-only: refuse the prepared-reference feed on this thread, so one
    /// test can serve the same steering session both ways.
    pub(crate) static REF_FEED_OFF: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
    /// Test-only: retaining walks served by the prepared-reference feed on
    /// this thread.
    pub(crate) static REF_FEED_HITS: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

/// Whether a RETAINING walk (prepared steering's) may take its source side
/// from `precomputed`: [`ref_feed_admits`], plus a Rev5+ plan whose revision
/// is the process revision. A steering reference is converted either at the
/// plan's revision (`PrecomputedReference::for_candidate`) or at the process
/// revision (`Zensim::precompute_reference`); requiring both to agree means the
/// cached planes were converted with the walk's own arithmetic, and from Rev4
/// on the conversion is per-pixel (no chunk or band length can move a byte,
/// `streaming::tests::convert_chunk_rows_is_semantics_not_a_knob`), so the
/// cached planes are the bytes the walk would have converted.
#[cfg(feature = "custom-profiles")]
pub(crate) fn retained_ref_feed_admits(
    precomputed: &crate::streaming::PrecomputedReference,
    distorted: &impl ImageSource,
    plan: &crate::feature_plan::Plan,
) -> bool {
    #[cfg(test)]
    if REF_FEED_OFF.with(core::cell::Cell::get) {
        return false;
    }
    plan.compute.formula_revision >= FormulaRevision::Rev5
        && crate::featcanon::mode(precomputed.formula_revision)
            == crate::featcanon::mode(plan.compute.formula_revision)
        && ref_feed_admits(precomputed, distorted, Some(plan))
}

/// [`compute_folded_v1_372_streaming_impl`]'s unpadded walk with the source
/// side copied from `precomputed` and the session's `retention` attached:
/// prepared steering re-scores one reference against many reconstructions,
/// so the reference's conversion and downscale chain run once per session
/// instead of once per call. The caller checks [`retained_ref_feed_admits`].
/// Bit-identical to the uncached walk (same producer over the same planes);
/// `prepared_steering_ref_feed_is_bit_identical` is the gate.
#[cfg(feature = "custom-profiles")]
pub(crate) fn compute_folded_with_ref_retained(
    source: &impl ImageSource,
    precomputed: &crate::streaming::PrecomputedReference,
    distorted: &impl ImageSource,
    parallel: bool,
    scratch: &mut V2Scratch,
    plan: &crate::feature_plan::Plan,
    retention: Option<&mut FoldRetention>,
) -> (Vec<f64>, [f64; 3]) {
    debug_assert!(retained_ref_feed_admits(precomputed, distorted, plan));
    #[cfg(test)]
    REF_FEED_HITS.with(|hits| hits.set(hits.get() + 1));
    let mut mo = MeanOffsetRows::at_revision(
        source.width(),
        source.height(),
        plan.compute.formula_revision,
    );
    let res = foldapp_streaming_walk(
        source,
        distorted,
        parallel,
        plan.toggles(),
        crate::feature_v2_stream::FrontEnd::Sdr,
        scratch,
        FoldWalkExtras {
            compute: Some(plan.compute),
            mean_offset: Some(&mut mo),
            retention,
            ref_planes: Some(&precomputed.scales),
            stop: None,
        },
    );
    (res.into_features(), mo.finish())
}

/// The declared-HDR folded/append streaming entry (HDR_PLAN chunk 2):
/// pixel values → absolute cd/m² per `encoding` → the UPIQ-validated
/// PU-XYB front-end → the UNCHANGED streaming 924 walk (same layout,
/// same formulas, same constants — `FeatureRegime` reports as the same
/// folded regimes; HDR-ness is a property of the extraction run, tracked
/// by the caller exactly like the fold-vs-ext regime rule).
///
/// Accepted sources: `LinearF32Rgba` (f32; cd/m² for `Linear`, code
/// values for `Pq`/`Hlg`) or `Srgb16Rgba` (u16 code values, `Pq`/`Hlg`
/// only) — both with `AlphaMode::Opaque` (the alpha noise-background
/// compositor is `[0,1]`-relative and NOT validated on absolute-light
/// pixels). Declared primaries are converted to linear sRGB before opsin,
/// preserving negative and above-one components (no SDR display clamp).
/// Everything else:
/// `HdrInputRequiresPuPath`.
pub(crate) fn compute_folded720_hdr_streaming_impl(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    encoding: HdrEncoding,
    max_pixels: Option<usize>,
    parallel: bool,
    toggles: V2NewFeatureToggles,
    scratch: &mut V2Scratch,
    planned_compute: Option<ComputeSet>,
) -> Result<ZensimV2Result, ZensimError> {
    compute_folded720_hdr_streaming_extras(
        source,
        distorted,
        encoding,
        max_pixels,
        parallel,
        toggles,
        scratch,
        FoldWalkExtras {
            compute: planned_compute,
            ..Default::default()
        },
    )
}

/// [`compute_folded720_hdr_streaming_impl`] with the caller's
/// [`FoldWalkExtras`] carried into the walk — the training surface's
/// declared-HDR entry.
pub(crate) fn compute_folded720_hdr_streaming_extras(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    encoding: HdrEncoding,
    max_pixels: Option<usize>,
    parallel: bool,
    toggles: V2NewFeatureToggles,
    scratch: &mut V2Scratch,
    extras: FoldWalkExtras<'_>,
) -> Result<ZensimV2Result, ZensimError> {
    if !extras
        .compute
        .is_some_and(|c| c.formula_revision >= FormulaRevision::Rev5)
    {
        validate_wide_revision(toggles)?;
    }
    if extras.compute.is_none() {
        validate_raw_rev5_request(toggles)?;
    }
    // REV4SERVE: the PU front end is canonical at Rev4
    // (`color::pu_xyb_canon` + the `_at_revision` transfer decoders), so the
    // HDR walk computes Rev4 in a Rev4 process — `refuse_rev4_mix` keeps
    // the cross-boundary cases out.
    crate::ssim_form::refuse_rev4_mix(toggles.formula_revision)?;
    validate_hdr_pair(source, distorted, encoding, max_pixels)?;
    let front_end = crate::feature_v2_stream::FrontEnd::Hdr(encoding);
    if source.width() < crate::metric::MIN_PYRAMID_DIM
        || source.height() < crate::metric::MIN_PYRAMID_DIM
    {
        let padded_src = crate::metric::reflect_pad_to_min(source);
        let padded_dst = crate::metric::reflect_pad_to_min(distorted);
        return Ok(foldapp_streaming_walk(
            &padded_src,
            &padded_dst,
            parallel,
            toggles,
            front_end,
            scratch,
            extras,
        ));
    }
    Ok(foldapp_streaming_walk(
        source, distorted, parallel, toggles, front_end, scratch, extras,
    ))
}

pub(crate) fn validate_hdr_pair(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    encoding: HdrEncoding,
    max_pixels: Option<usize>,
) -> Result<(), ZensimError> {
    crate::metric::validate_pair_dims(source, distorted)?;
    crate::metric::check_within_max_pixels(source.width(), source.height(), max_pixels)?;
    let valid_display = match encoding {
        HdrEncoding::Linear => true,
        HdrEncoding::Pq { peak_nits } => peak_nits.is_finite() && peak_nits > 0.0,
        HdrEncoding::Hlg {
            peak_nits,
            ambient_lux,
        } => {
            peak_nits.is_finite()
                && peak_nits > 0.0
                && ambient_lux.is_finite()
                && ambient_lux >= 0.0
        }
    };
    if !valid_display {
        return Err(ZensimError::HdrInputRequiresPuPath);
    }
    let shape_ok = |fmt: crate::source::PixelFormat, alpha: crate::source::AlphaMode| {
        let fmt_ok = match encoding {
            HdrEncoding::Linear => fmt == crate::source::PixelFormat::LinearF32Rgba,
            HdrEncoding::Pq { .. } | HdrEncoding::Hlg { .. } => {
                fmt == crate::source::PixelFormat::LinearF32Rgba
                    || fmt == crate::source::PixelFormat::Srgb16Rgba
            }
        };
        fmt_ok && matches!(alpha, crate::source::AlphaMode::Opaque)
    };
    if !shape_ok(source.pixel_format(), source.alpha_mode())
        || !shape_ok(distorted.pixel_format(), distorted.alpha_mode())
    {
        return Err(ZensimError::HdrInputRequiresPuPath);
    }
    Ok(())
}

/// Folded-720+append+append2 pair entry (944; [`FeatureRegime::
/// Folded720Append2`]): forces `append_block` + `append2_block`.
pub(crate) fn compute_folded720_append2_impl(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    mut toggles: V2NewFeatureToggles,
) -> Result<ZensimV2Result, ZensimError> {
    toggles.append_block = true;
    toggles.append2_block = true;
    compute_folded720_impl_with_toggles(source, distorted, max_pixels, parallel, toggles)
}

/// Declared-HDR 944 entry: [`compute_folded720_hdr_streaming_impl`] with
/// append + append2 forced on (HL bins live; PU-domain BANDVIS δs).
pub(crate) fn compute_folded720_append2_hdr_streaming_impl(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    encoding: HdrEncoding,
    max_pixels: Option<usize>,
    parallel: bool,
    mut toggles: V2NewFeatureToggles,
    scratch: &mut V2Scratch,
) -> Result<ZensimV2Result, ZensimError> {
    toggles.append_block = true;
    toggles.append2_block = true;
    compute_folded720_hdr_streaming_impl(
        source, distorted, encoding, max_pixels, parallel, toggles, scratch, None,
    )
}

/// Streaming folded-944 extraction WITH attribution retention — the fused
/// compare's extractor-side hooks (campaign appendix N). Identical
/// validation + sub-64 reflect-pad to [`compute_folded720_streaming_impl`],
/// `append_block` + `append2_block` forced on, accumulation UNTOUCHED (the
/// hooks only copy), so the output features are BITWISE the canonical 944
/// extraction (G-N1 gate). SDR route only for now: HDR-declared inputs get
/// [`ZensimError::HdrInputRequiresPuPath`] (the 944 set is SDR-by-design;
/// an HDR fused route is registered future work).
///
/// Gated on `custom-profiles`: the fused entry ([`crate::Fused944Session`])
/// lives in the `attribution` module, which only exists under that feature —
/// same both-features rule as the `Fused944Session` re-export (af4417f8).
#[cfg(feature = "custom-profiles")]
pub(crate) fn compute_folded944_streaming_with_retention(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    scratch: &mut V2Scratch,
    retention: &mut FoldRetention,
) -> Result<ZensimV2Result, ZensimError> {
    crate::metric::validate_pair_dims(source, distorted)?;
    crate::metric::check_within_max_pixels(source.width(), source.height(), max_pixels)?;
    if source.is_hdr() || distorted.is_hdr() {
        return Err(ZensimError::HdrInputRequiresPuPath);
    }
    let toggles = V2NewFeatureToggles {
        append_block: true,
        append2_block: true,
        ..V2NewFeatureToggles::default()
    };
    validate_raw_rev5_request(toggles)?;
    if source.width() < crate::metric::MIN_PYRAMID_DIM
        || source.height() < crate::metric::MIN_PYRAMID_DIM
    {
        let padded_src = crate::metric::reflect_pad_to_min(source);
        let padded_dst = crate::metric::reflect_pad_to_min(distorted);
        return Ok(foldapp_streaming_walk(
            &padded_src,
            &padded_dst,
            parallel,
            toggles,
            crate::feature_v2_stream::FrontEnd::Sdr,
            scratch,
            FoldWalkExtras {
                retention: Some(retention),
                ..Default::default()
            },
        ));
    }
    Ok(foldapp_streaming_walk(
        source,
        distorted,
        parallel,
        toggles,
        crate::feature_v2_stream::FrontEnd::Sdr,
        scratch,
        FoldWalkExtras {
            retention: Some(retention),
            ..Default::default()
        },
    ))
}

/// Reusable f32 pass-B scratch for the fused retention path (appendix P
/// lever 1): plane-sized buffers kept across compares in the
/// [`crate::Fused944Session`] so per-iteration callers pay no allocation /
/// page-fault cost.
///
/// `custom-profiles`-gated with the rest of the fused-retention cluster: its
/// only consumer is the `attribution` module (af4417f8 rule).
#[cfg(feature = "custom-profiles")]
#[derive(Default)]
pub(crate) struct PassBScratchF32 {
    pub(crate) canvas: Vec<f32>,
    pub(crate) scale_density: Vec<f32>,
    pub(crate) win_plane: Vec<f32>,
    pub(crate) spread_tmp: Vec<f32>,
    pub(crate) spread_out: Vec<f32>,
    /// Lattice-step boundary terms of the last pass B (see `crate::attribution::LatticeScale`).
    pub(crate) lattice: crate::attribution::LatticeSet,
}

/// The v2/append/append2 attribution density built from walk RETENTION
/// (appendix N): coefficients derived from the EXACT streaming
/// accumulators + finalize mean-gradients, pass B over the retained
/// planes — sharing the coefficient derivation ([`derive_v2app_coeffs`] /
/// [`attr_ew_coeff`]) verbatim with the standalone
/// [`compute_v2_append_attribution`], with the f32 combine kernels
/// (appendix P lever 1: `attr_pass_b_rows_f32` — the standalone keeps the
/// strict f64 pass B). The retained planes are bitwise the standalone's
/// `AttrChPlanes` (producer gate + same blur chain), so the densities
/// differ ONLY by the coefficient inputs (exact accums vs the standalone's
/// 1e-9-parity pass-A replication) and the f32 combine — measured inside
/// the C3a tolerance class (G-N2/G-P2 gate). Returns the TRIMMED f32
/// density at `orig_w × orig_h` (the fused caller adds it into the f32
/// basic canvas; features stay owned by the real extraction).
///
/// `custom-profiles`-gated: the f32 pass B below reaches into
/// `crate::attribution` for its upsample kernel, and that module is
/// configured out without the feature — a `feature-regime-v2`-only build
/// failed with E0433 here until this gate (the 87c5e9ef note; same
/// both-features rule as the `Fused944Session` re-export, af4417f8).
#[cfg(feature = "custom-profiles")]
pub(crate) fn compute_v2_append_attribution_from_retention(
    ret: &FoldRetention,
    s_v2: &[f64],
    s_append: Option<&[f64]>,
    s_append2: Option<&[f64]>,
    parallel: bool,
    orig_w: usize,
    orig_h: usize,
    scratch: &mut PassBScratchF32,
) -> (Vec<f32>, Vec<crate::attribution::LatticeScale>) {
    let (w0, h0) = ret.dims[0];
    let n0 = w0 * h0;
    scratch.canvas.clear();
    scratch.canvas.resize(n0, 0.0);
    let mut canvas = std::mem::take(&mut scratch.canvas);
    retention_pass_b_all_scales(
        ret,
        s_v2,
        s_append,
        s_append2,
        parallel,
        scratch,
        &mut crate::attribution::AttrSinkF32::Canvas(&mut canvas),
        None,
        (orig_w, orig_h),
        crate::map_paint::MapPaint::default(),
    );
    let lattice = scratch.lattice.finish();
    // Trim the (possibly reflect-padded sub-64) canvas to the original.
    let out = if orig_w == w0 && orig_h == h0 {
        canvas.clone()
    } else {
        let mut out = Vec::with_capacity(orig_w * orig_h);
        for y in 0..orig_h.min(h0) {
            out.extend_from_slice(&canvas[y * w0..y * w0 + orig_w.min(w0)]);
        }
        out
    };
    scratch.canvas = canvas;
    (out, lattice)
}

/// Level-2 sibling of
/// [`compute_v2_append_attribution_from_retention`]: pass B folds each
/// scale straight into the caller's [`BinAccum`](crate::attribution::BinAccum)
/// — the session's full-resolution pass-B canvas stays untouched (and
/// unallocated on binned-only sessions), and no trimmed clone is made.
#[cfg(feature = "custom-profiles")]
pub(crate) fn compute_v2_append_attribution_from_retention_into_bins(
    ret: &FoldRetention,
    s_v2: &[f64],
    s_append: Option<&[f64]>,
    s_append2: Option<&[f64]>,
    parallel: bool,
    scratch: &mut PassBScratchF32,
    accum: &mut crate::attribution::BinAccum,
    geometry: Option<&crate::sampling::Geometry>,
    paint: crate::map_paint::MapPaint,
) {
    let logical = accum.logical_dimensions();
    retention_pass_b_all_scales(
        ret,
        s_v2,
        s_append,
        s_append2,
        parallel,
        scratch,
        &mut crate::attribution::AttrSinkF32::Bins(accum),
        geometry,
        logical,
        paint,
    );
    accum.set_lattice(scratch.lattice.finish());
}

/// Whether this cell has any contribution to the requested attribution.
#[cfg(feature = "custom-profiles")]
fn attribution_cell_active(
    s_v2: &[f64],
    s_append: Option<&[f64]>,
    s_append2: Option<&[f64]>,
    scale: usize,
    ch: usize,
) -> bool {
    let active = |values: &[f64], cell: usize, width: usize| {
        values
            .iter()
            .skip(cell * width)
            .take(width)
            .any(|&v| v != 0.0)
    };
    let cell = scale * 3 + ch;
    active(s_v2, cell, FEATURES_PER_CHANNEL_V2_TOTAL)
        || s_append.is_some_and(|s| active(s, cell, FEATURES_PER_CHANNEL_APPEND))
        || (ch == APPEND2_CHANNEL && s_append2.is_some_and(|s| active(s, scale, APPEND2_PER_SCALE)))
}

/// Shared per-scale pass-B loop over the retention for both sinks.
#[cfg(feature = "custom-profiles")]
fn retention_pass_b_all_scales(
    ret: &FoldRetention,
    s_v2: &[f64],
    s_append: Option<&[f64]>,
    s_append2: Option<&[f64]>,
    parallel: bool,
    scratch: &mut PassBScratchF32,
    sink: &mut crate::attribution::AttrSinkF32<'_>,
    geometry: Option<&crate::sampling::Geometry>,
    logical: (usize, usize),
    paint: crate::map_paint::MapPaint,
) {
    scratch
        .lattice
        .reset(logical.0, logical.1, geometry.is_none());
    let n_scales = ret.dims.len();
    let (w0, h0) = ret.dims[0];
    let want_append = s_append.is_some();
    let n0 = w0 * h0;
    scratch.scale_density.resize(n0, 0.0);
    scratch.win_plane.resize(n0, 0.0);
    // Research guided painting (`crate::map_paint`): the source-resolution
    // difference guide, built once per map when that mode is selected.
    let guide0 =
        (paint.paint == crate::map_paint::PaintMode::Guided && geometry.is_none()).then(|| {
            crate::map_paint::difference_guide(
                [&ret.pyr_src[0][0], &ret.pyr_src[0][1], &ret.pyr_src[0][2]],
                [&ret.pyr_dst[0][0], &ret.pyr_dst[0][1], &ret.pyr_dst[0][2]],
                n0,
                paint.guide,
            )
        });
    for scale in 0..n_scales {
        // A selectively omitted cell can have no retained samples. Deriving
        // its coefficients would evaluate 0 * (1 / 0), poisoning an otherwise
        // valid coarse map. Zero sensitivity makes its contribution exactly
        // zero; skip its work before any normalization or plane access.
        if !(0..3).any(|ch| attribution_cell_active(s_v2, s_append, s_append2, scale, ch)) {
            continue;
        }
        attr_pass_b_for_scale_f32(
            scale,
            [
                &ret.pyr_src[scale][0],
                &ret.pyr_src[scale][1],
                &ret.pyr_src[scale][2],
            ],
            [
                &ret.pyr_dst[scale][0],
                &ret.pyr_dst[scale][1],
                &ret.pyr_dst[scale][2],
            ],
            &ret.planes[scale],
            &ret.cells,
            &ret.mg,
            &ret.dims,
            n_scales,
            s_v2,
            s_append,
            s_append2,
            want_append,
            parallel,
            w0,
            h0,
            &mut scratch.scale_density,
            &mut scratch.win_plane,
            sink,
            &mut scratch.spread_tmp,
            &mut scratch.spread_out,
            geometry,
            &mut scratch.lattice,
            guide0.as_deref(),
            paint,
            ret.revision,
        );
    }
}

/// Folded-720+append+append2+CSFW pair entry (956; [`FeatureRegime::
/// Folded720Csfw`]): forces `append_block` + `append2_block` + `csfw_block`.
pub(crate) fn compute_folded720_csfw_impl(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    mut toggles: V2NewFeatureToggles,
) -> Result<ZensimV2Result, ZensimError> {
    toggles.append_block = true;
    toggles.append2_block = true;
    toggles.csfw_block = true;
    compute_folded720_impl_with_toggles(source, distorted, max_pixels, parallel, toggles)
}

/// Declared-HDR 956 entry: [`compute_folded720_hdr_streaming_impl`] with
/// append + append2 + CSFW forced on (PU-route φ constants).
pub(crate) fn compute_folded720_csfw_hdr_streaming_impl(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    encoding: HdrEncoding,
    max_pixels: Option<usize>,
    parallel: bool,
    mut toggles: V2NewFeatureToggles,
    scratch: &mut V2Scratch,
) -> Result<ZensimV2Result, ZensimError> {
    toggles.append_block = true;
    toggles.append2_block = true;
    toggles.csfw_block = true;
    compute_folded720_hdr_streaming_impl(
        source, distorted, encoding, max_pixels, parallel, toggles, scratch, None,
    )
}

/// [`compute_folded720_hdr_streaming_impl`] with the append block forced
/// on (the 924 shape).
pub(crate) fn compute_folded720_append_hdr_streaming_impl(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    encoding: HdrEncoding,
    max_pixels: Option<usize>,
    parallel: bool,
    mut toggles: V2NewFeatureToggles,
    scratch: &mut V2Scratch,
) -> Result<ZensimV2Result, ZensimError> {
    toggles.append_block = true;
    compute_folded720_hdr_streaming_impl(
        source, distorted, encoding, max_pixels, parallel, toggles, scratch, None,
    )
}

/// [`compute_folded720_streaming_impl`] with the append block forced on
/// (the [`FeatureRegime::Folded720Append`] entry shape).
pub(crate) fn compute_folded720_append_streaming_impl(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    mut toggles: V2NewFeatureToggles,
    scratch: &mut V2Scratch,
) -> Result<ZensimV2Result, ZensimError> {
    toggles.append_block = true;
    compute_folded720_streaming_impl(
        source, distorted, max_pixels, parallel, toggles, scratch, None,
    )
}

/// The streaming walk body (inputs already validated + ≥ 64px).
/// `retention`: the fused compare's extractor-side hooks (appendix N) —
/// `Some` copies per-strip planes + finalize accumulators into the
/// [`FoldRetention`]; `None` (every pre-existing caller) is the untouched
/// walk. The hooks only COPY: accumulation, kernels, and finalize are
/// identical either way, so retained and plain walks emit bitwise-equal
/// features (G-N1).
/// Per-scale-0-row `Σ_x (src[x] − dst[x])` sums, in f64, one triple per row —
/// the decomposition that lets the streaming fold reproduce the buffered
/// path's `ZensimResult::mean_offset` **bit-exactly** rather than to a
/// tolerance.
///
/// `streaming::compute_xyb_mean_offset` sums 64-row chunks, each chunk
/// summing per-row f64 sums of `(s[i] − d[i]) as f64` — note the subtraction
/// happens in **f32** and only the difference is widened. Two properties make
/// it reproducible from a strip walk that visits scale-0 rows in a different
/// grouping:
///
/// 1. It is already **thread-invariant**: the rayon arm is an
///    order-preserving `into_par_iter().map().collect()` and the final
///    reduction walks that `Vec` serially in row order, so parallelism never
///    reassociates it.
/// 2. It **decomposes per row**: chunk `acc` is a left-to-right f64 sum of
///    per-row sums, so recording the per-row sums as the strips go past and
///    performing [`Self::finish`]'s identical chunk reduction afterwards
///    lands on the same bits.
///
/// Cost is one `Vec<[f64; 3]>` of `height` entries (55 KB at 2304², against
/// the walk's measured 163 MB peak RSS) plus one f32 subtract + f64 add per
/// scale-0 pixel per channel, taken while the producer's rolling plane rows
/// are already cache-hot.
/// Rows per task in [`MeanOffsetRows::add_strip`]'s fan-out. Purely a
/// schedule parameter — the values are per-row and assignment-only, so no
/// choice here can move a byte (`mean_offset_row_bands_are_bit_exact`).
#[cfg(feature = "threads")]
const MEAN_OFFSET_BAND_ROWS: usize = 16;

pub(crate) struct MeanOffsetRows {
    rows: Vec<[f64; 3]>,
    width: usize,
    height: usize,
}

impl MeanOffsetRows {
    pub(crate) fn new(width: usize, height: usize) -> Self {
        Self::at_revision(width, height, crate::ssim_form::active_revision())
    }

    fn at_revision(width: usize, height: usize, revision: FormulaRevision) -> Self {
        // Rev5 exposes no global XYB offset: the supported model inputs do
        // not consume it. Keep a zero metadata value without an image pass.
        let height = if revision >= FormulaRevision::Rev5 {
            0
        } else {
            height
        };
        Self {
            rows: vec![[0.0f64; 3]; height],
            width,
            height,
        }
    }

    /// Record one scale-0 strip's rows for all three channels. `src`/`dst` are
    /// the producer's rolling-plane rows covering `[y0, y0 + n_rows)` at
    /// `plane_w` stride; `plane_w == width` at scale 0 under option C.
    ///
    /// **Row-parallel, and BIT-EXACT by construction** (fold-MT lane). Every
    /// `rows[y][ch]` is ASSIGNED — not accumulated into — by a pure function of
    /// that one row of `src`/`dst`, so each element is written exactly once
    /// with a value that does not depend on any other row, any other channel,
    /// or the order in which they run. The only ordered arithmetic is the
    /// left-to-right `f64` sum WITHIN a row, and that is never split.
    /// [`Self::finish`]'s 64-row-chunk reduction then reads `rows` in its own
    /// fixed order regardless.
    ///
    /// Why it earns a fan-out: this side channel is a full pass over the
    /// scale-0 planes for all three channels (127 MB read per walk at 2304²)
    /// and it ran SERIALLY between `next_strip` and the channel fan-out — it
    /// was the whole of the 7.1 ms the phase profile could not account for,
    /// i.e. 13 % of a 16-thread compare at degree 1.
    fn add_strip(
        &mut self,
        y0: usize,
        n_rows: usize,
        plane_w: usize,
        src: [&[f32]; 3],
        dst: [&[f32]; 3],
        #[allow(unused_variables)] parallel: bool,
    ) {
        let y1 = (y0 + n_rows).min(self.height);
        if y1 <= y0 {
            return;
        }
        let width = self.width;
        let rows = &mut self.rows[y0..y1];
        let one = |k_base: usize, out: &mut [[f64; 3]]| {
            for (i, o) in out.iter_mut().enumerate() {
                let k = k_base + i;
                for ch in 0..3 {
                    let s = &src[ch][k * plane_w..k * plane_w + width];
                    let d = &dst[ch][k * plane_w..k * plane_w + width];
                    let mut row_sum = 0.0f64;
                    for i in 0..width {
                        row_sum += (s[i] - d[i]) as f64;
                    }
                    o[ch] = row_sum;
                }
            }
        };
        #[cfg(feature = "threads")]
        if parallel && rows.len() > MEAN_OFFSET_BAND_ROWS {
            use rayon::prelude::*;
            rows.par_chunks_mut(MEAN_OFFSET_BAND_ROWS)
                .enumerate()
                .for_each(|(b, out)| one(b * MEAN_OFFSET_BAND_ROWS, out));
            return;
        }
        one(0, rows);
    }

    /// The exact reduction `streaming::compute_xyb_mean_offset` performs:
    /// 64-row chunks in ascending order, per channel a left-to-right sum of
    /// the chunk's row sums, then a left-to-right sum over chunks, then one
    /// divide by `width · height`. Reproduced loop-for-loop on purpose — this
    /// is the function whose *shape* is the guarantee.
    pub(crate) fn finish(&self) -> [f64; 3] {
        if self.height == 0 {
            return [0.0; 3];
        }
        let n = (self.width * self.height) as f64;
        let chunk_rows = 64usize;
        let mut chunks: Vec<[f64; 3]> = Vec::with_capacity(self.height.div_ceil(chunk_rows));
        let mut row_start = 0usize;
        while row_start < self.height {
            let row_end = (row_start + chunk_rows).min(self.height);
            let mut diff = [0.0f64; 3];
            for (c, d) in diff.iter_mut().enumerate() {
                let mut acc = 0.0f64;
                for y in row_start..row_end {
                    acc += self.rows[y][c];
                }
                *d = acc;
            }
            chunks.push(diff);
            row_start += chunk_rows;
        }
        let mut offset = [0.0f64; 3];
        for chunk_diff in &chunks {
            for (o, &d) in offset.iter_mut().zip(chunk_diff.iter()) {
                *o += d;
            }
        }
        for o in &mut offset {
            *o /= n;
        }
        offset
    }
}

/// The fold walk's optional SIDE CHANNELS, bundled so the walk keeps one
/// options parameter instead of growing a positional tail.
///
/// All three are `None` on the plain extraction path, which is then
/// byte-for-byte the walk as it was before any of them existed.
#[derive(Default)]
pub(crate) struct FoldWalkExtras<'a> {
    // A cancelled partial walk is discarded by the named scorer's post-check.
    pub(crate) stop: Option<&'a dyn enough::Stop>,
    /// Resolved model/research plan; raw public extraction keeps all channels.
    pub(crate) compute: Option<ComputeSet>,
    /// Appendix-N retention hooks (the fused-944 attribution session).
    pub(crate) retention: Option<&'a mut FoldRetention>,
    /// Per-scale-0-row `Σ_x (src − dst)` sums, for the fold-backed engine's
    /// bit-exact `mean_offset` (see [`MeanOffsetRows`]).
    pub(crate) mean_offset: Option<&'a mut MeanOffsetRows>,
    /// A pre-built source-side XYB pyramid the producer copies from instead
    /// of decoding + converting + downscaling (the ref-cached fold form).
    pub(crate) ref_planes: Option<&'a [crate::streaming::XybPyramidLevel]>,
}

fn foldapp_streaming_walk<S: ImageSource, D: ImageSource>(
    source: &S,
    distorted: &D,
    parallel: bool,
    toggles: V2NewFeatureToggles,
    front_end: crate::feature_v2_stream::FrontEnd,
    scratch: &mut V2Scratch,
    extras: FoldWalkExtras<'_>,
) -> ZensimV2Result {
    let compute = extras
        .compute
        .unwrap_or_else(|| ComputeSet::from_toggles(toggles));
    if compute.full_res_xb && compute.coarse_y_only_scales == 0 {
        foldapp_streaming_walk_impl::<S, D, true>(
            source, distorted, parallel, toggles, front_end, scratch, extras,
        )
    } else {
        assert!(compute.allows_full_res_y_subset());
        foldapp_streaming_walk_impl::<S, D, false>(
            source, distorted, parallel, toggles, front_end, scratch, extras,
        )
    }
}

fn foldapp_streaming_walk_impl<S: ImageSource, D: ImageSource, const ALL_CHANNELS: bool>(
    source: &S,
    distorted: &D,
    parallel: bool,
    toggles: V2NewFeatureToggles,
    front_end: crate::feature_v2_stream::FrontEnd,
    scratch: &mut V2Scratch,
    extras: FoldWalkExtras<'_>,
) -> ZensimV2Result {
    let FoldWalkExtras {
        stop,
        mut retention,
        mut mean_offset,
        ref_planes,
        compute,
    } = extras;
    use crate::feature_v2_stream::StripPlaneProducer;
    // ITEM D: one derivation of WHAT this request computes. Every local
    // below reads from it instead of re-deriving from `toggles`.
    let mut compute = compute.unwrap_or_else(|| ComputeSet::from_toggles(toggles));
    // rev5: plan-supplied sets are already scoped (`from_toggles` and
    // `Plan::normalized` both apply `rev5_scope`); a hand-built set is
    // clamped here so the walk can never run a family Rev5 does not serve.
    compute.rev5_scope();
    let fold_v1 = compute.v1_basic;
    // BLOCK-SKIPPING: a v1-only request computes NOTHING v2-era. Every
    // v2 toggle is forced off in `ComputeSet::from_toggles` rather than
    // asserted, so the caller can set `v1_only` on top of any existing
    // toggle set and get the v1 blocks of that request — which is what
    // makes it a drop-in A/B in the bench.
    let v2_blocks = compute.v2_blocks;
    // BAND PARALLELISM: the channel fan-out is only 3-way, and the fold was
    // MEASURED to saturate at exactly 3 threads (2.27x) and REGRESS beyond
    // it. Bands inside a channel are independent, so they are the next axis;
    // rayon nests fine inside the channel fan-out.
    let band_parallel = parallel;
    // LAYOUT vs COMPUTE. The emitted vector keeps the WIDTH and REGIME the
    // caller asked for — a v1-only 944 request is still a 944 row, with
    // `f372..` at the structural 0.0 — while the `*_on` flags below drive
    // what is actually COMPUTED. Deriving the regime from the compute flags
    // instead would silently hand back a 720-wide vector for a 944 request.
    let layout_append = toggles.append_block;
    let layout_append2 = toggles.append2_block;
    let layout_csfw = toggles.csfw_block;
    // REV4 family layout flags — independent (each family emits iff its
    // own layout flag, so a rev4-subset request keeps its exact width).
    let layout_gridblk = toggles.rev4_gridblk;
    let layout_ringbasis = toggles.rev4_ringbasis;
    let layout_tailhist = toggles.rev4_tailhist;
    let layout_arttype = toggles.rev4_arttype;
    let layout_gmsbank = toggles.gmsbank;
    let layout_mapdev = toggles.mapdev;
    let layout_z1max = toggles.z1max;
    let layout_gmsnative = toggles.gmsnative;
    // `globaff` (revision 6): the layout flag sets the width; `compute.globaff`
    // decides whether the strip moment sums run.
    let layout_globaff = toggles.globaff;
    let globaff_on = compute.globaff;
    // Only read below under `threads` (the `fuse_channels` derivation a few
    // lines down); the `not(threads)` arm hardcodes `fuse_channels = false`
    // without it, so `--no-default-features --features feature-regime-v2`
    // (no `threads`) sees this as a genuinely unused binding.
    #[cfg(feature = "threads")]
    let append_on = compute.append;
    let append2_on = compute.append2;
    assert!(
        !layout_append2 || layout_append,
        "append2_block requires append_block (f924+ sits after the append block)"
    );
    assert!(
        !toggles.append2_dst_activity || append2_on,
        "append2_dst_activity requires append2_block (it refines the BANDVIS lanes)"
    );
    let append2 = append2_on.then(|| {
        use crate::feature_v2_stream::FrontEnd;
        match front_end {
            FrontEnd::Sdr => Append2Params {
                bv_delta_lo: BV_DELTA_LO_SDR,
                bv_delta_hi: BV_DELTA_HI_SDR,
                hl_bins: false,
                dst_activity: toggles.append2_dst_activity,
            },
            FrontEnd::Hdr(_) => Append2Params {
                bv_delta_lo: BV_DELTA_LO_PU,
                bv_delta_hi: BV_DELTA_HI_PU,
                hl_bins: true,
                dst_activity: toggles.append2_dst_activity,
            },
        }
    });
    let csfw_on = toggles.csfw_block && v2_blocks;
    assert!(
        !layout_csfw || layout_append2,
        "csfw_block requires append2_block (f944+ sits after the append2 block)"
    );
    assert!(
        !layout_gmsbank
            || (layout_gridblk && layout_ringbasis && layout_tailhist && layout_arttype),
        "gmsbank requires the full f1322 prefix layout"
    );
    assert!(
        !layout_mapdev || layout_gmsbank,
        "mapdev requires the C8 layout (f1502 sits after f1501)"
    );
    assert!(
        !layout_z1max || layout_mapdev,
        "z1max requires the mapdev layout (f1562 sits after f1561)"
    );
    assert!(
        !layout_gmsnative || layout_z1max,
        "gmsnative requires the z1max layout (f1790 sits after f1789)"
    );
    assert!(
        !layout_globaff || layout_gmsnative,
        "globaff requires the gmsnative layout (f1867 sits after f1866)"
    );
    assert!(
        !globaff_on || (layout_globaff && compute.formula_revision.computes_globaff()),
        "globaff is computed only at formula revision 6, into its own layout"
    );
    assert!(
        !globaff_on || compute.sampling.is_none(),
        "globaff is defined on the walk's own pyramid, not a sampled one"
    );
    // Route-local derived φ: the SAME weighting mechanism on both routes,
    // pre-composed with each route's own encoding (design §6 — runtime
    // never inverts an encoding; the route-dependence lives entirely in
    // which constant set is selected, the `BV_DELTA_*` pattern).
    let csfw = csfw_on.then(|| {
        use crate::feature_v2_stream::FrontEnd;
        match front_end {
            FrontEnd::Sdr => CsfwParams::for_phi(CSFW_PHI_Y_SDR),
            FrontEnd::Hdr(_) => CsfwParams::for_phi(CSFW_PHI_Y_PU),
        }
    });
    let n_scales = crate::NUM_SCALES;
    let (w0, h0) = (source.width(), source.height());

    // Scale dims: floor-halving, matching both the producer and the
    // materialized walk's `downscale_2x_inplace` returns.
    let mut dims = Vec::with_capacity(n_scales);
    {
        let (mut w, mut h) = (w0, h0);
        for _ in 0..n_scales {
            dims.push((w, h));
            w /= 2;
            h /= 2;
        }
    }

    if let Some(sampling) = compute.sampling {
        dims = sampling.dims(w0, h0);
    }
    let strip_max_n = dims[0].0 * (STRIP_ROWS + 2 * HALO_P);
    // HOISTED so the strip scratch can be sized to the planes this walk will
    // actually write (fold-footprint lane). Both flags are strip-independent —
    // they were computed inside the loop purely because that is where they are
    // read — and the loop now reads these, so there is one definition of each.
    #[cfg(feature = "threads")]
    let fuse_channels = parallel && !append_on;
    #[cfg(not(feature = "threads"))]
    let fuse_channels = false;
    let self_blur = fuse_channels && retention.is_none() && compute.self_blur_eligible();
    // Phase A is skipped WHOLE under self-blur bands, so its four fused-H
    // outputs are never written — `plane_needs` is where that is decided.
    // Bounded strip queue: independent v2 cells retain their original 128-row
    // kernels and strip partials. Basic bands and blockiness still merge in
    // producer order. No image-sized pyramid or thread-dependent reduction.
    #[cfg(feature = "threads")]
    let job_budget = REV5_JOB_BUDGET_BYTES;
    #[cfg(all(test, feature = "threads"))]
    let job_budget = scratch.rev5_job_budget.unwrap_or(job_budget);
    #[cfg(feature = "threads")]
    let batch_limit = if parallel
        && compute.formula_revision >= FormulaRevision::Rev5
        && compute.v2_blocks
        && !layout_append
        && rayon::current_num_threads() >= 2
        && h0 > STRIP_ROWS
    {
        debug_assert!(
            !compute.csfw
                && !compute.append
                && !compute.append2
                && !compute.gridblk
                && !compute.ringbasis
                && !compute.tailhist
                && !compute.arttype,
            "queued merges cover only Rev5's supported families"
        );
        // If this entire walk fits the existing bounded queue, dispatch it
        // together instead of creating a full batch followed by a tiny tail.
        // The byte budget and canonical strip/merge order are unchanged.
        let jobs: usize = dims
            .iter()
            .enumerate()
            .map(|(scale, &(_, h))| {
                if compute.at_scale(scale).v2_blocks {
                    let channels = (0..3)
                        .filter(|&ch| ALL_CHANNELS || compute.channel_active(scale, ch))
                        .count();
                    h.div_ceil(STRIP_ROWS) * channels
                } else {
                    0
                }
            })
            .sum();
        let capacity = rev5_job_limit(strip_max_n, 16, job_budget);
        if jobs >= REV5_MIN_JOB_SLOTS && jobs <= capacity {
            jobs
        } else {
            // One job per thread, at least four (measured on a 1507×2048
            // photo, B at revision 7: two threads 21.2 → 18.7 ms with batches
            // of four; at eight threads batches of sixteen were slower, 7.8 →
            // 10.1 ms, their scratch no longer fitting the cache). The
            // pipeline uses the same number as its in-flight limit; at eight
            // threads limits of four and six were slower than eight.
            rev5_job_limit(strip_max_n, rayon::current_num_threads().max(4), job_budget)
        }
    } else {
        0
    };
    #[cfg(feature = "threads")]
    rev5_prepare_jobs(&mut scratch.rev5_jobs, batch_limit, strip_max_n);
    let needs = compute.plane_needs(self_blur);
    #[cfg(feature = "threads")]
    let needs = if batch_limit > 0 {
        // Only basic-only scale cells can reach the old phase A. They need
        // horizontal planes; queued v2 cells own all their vertical planes.
        StripPlaneNeeds {
            h: needs.h && compute.v2_scales != ComputeSet::ALL_SCALES,
            v2: false,
        }
    } else {
        needs
    };
    scratch.ensure_for(strip_max_n, needs);
    let V2Scratch {
        strips: scratch_strips,
        stream_pool,
        #[cfg(feature = "threads")]
        rev5_jobs,
        #[cfg(feature = "threads")]
        block_pool,
        ..
    } = scratch;

    // Finite queries need the omitted raw scale-0 distorted channels too.
    // Keep them during the prepared walk instead of converting the image again.
    // This changes storage only; channel_active still controls every feature.
    let complete_distorted0 = ref_planes.is_some()
        && retention.is_some()
        && compute.formula_revision >= FormulaRevision::Rev5
        && compute.v2_blocks;
    let mut accums: [StreamChannelAccums; 3] = std::array::from_fn(|_| {
        StreamChannelAccums::at_revision(
            n_scales,
            band_slots_for(band_parallel),
            compute.formula_revision,
        )
    });
    // REV4 C4's strip scratch — sized for the largest plane width; the
    // per-strip build slices it down to `info.plane_w`.
    let mut bleed_work = compute
        .arttype
        .then(|| BleedMaskWork::sized(dims[0].0, STRIP_ROWS));
    if let Some(ret) = retention.as_deref_mut() {
        ret.ensure(
            &dims,
            &compute,
            compute.formula_revision >= FormulaRevision::Rev5 && fold_v1,
            complete_distorted0,
        );
        if compute.formula_revision >= FormulaRevision::Rev5 {
            for (ch, acc) in accums.iter_mut().enumerate() {
                acc.retained_strips = Some(
                    dims.iter()
                        .enumerate()
                        .map(|(scale, &(_, h))| {
                            let active = (1..=3).contains(&scale)
                                && compute.at_scale(scale).v2_blocks
                                && compute.channel_active(scale, ch);
                            Vec::with_capacity(if active { h.div_ceil(STRIP_ROWS) } else { 0 })
                        })
                        .collect(),
                );
            }
        }
    }
    // BANDVIS dst self-mask live: phase A additionally produces the Y
    // channel's dst-activity plane (and ONLY the Y channel's — the block
    // is Y-only, so X/B never pay the chain).
    let act_dst_on = append2.map(|p| p.dst_activity).unwrap_or(false);
    // The restored-cut side passes (mapdev / z1max) rebuild both pyramids from
    // the pair, so they cannot serve a ref-fed walk (no source pixels), a
    // sampled walk, or the HDR front end.
    let restore_side_ok = ref_planes.is_none()
        && compute.sampling.is_none()
        && matches!(front_end, crate::feature_v2_stream::FrontEnd::Sdr);
    let mut producer = StripPlaneProducer::new_with_ref_feed(
        source,
        distorted,
        parallel,
        stream_pool,
        front_end,
        ref_planes,
        compute.sampling,
        toggles.formula_revision,
        // `globaff` reads every channel's rows at every scale, so the
        // scale-zero chroma rows stay in the rolling planes.
        !compute.full_res_xb && !complete_distorted0 && !globaff_on,
    );
    let mut globaff: Vec<[crate::globaff::GlobAffAccum; 3]> = if globaff_on {
        vec![Default::default(); n_scales]
    } else {
        Vec::new()
    };

    #[cfg(feature = "threads")]
    let mut queued = 0;
    // Plain SDR walks whose every live scale runs the v2 blocks produce their
    // rows in parallel chunk tasks, and their strip jobs copy their own
    // windows (`rev5_block_walk`); the streaming producer above is unused.
    #[cfg(feature = "threads")]
    let block_walk = {
        let omit_xb = !compute.full_res_xb
            && !complete_distorted0
            && !globaff_on
            && !crate::feature_v2_stream::linear_pyramid();
        batch_limit > 0
            && retention.is_none()
            && ref_planes.is_none()
            && compute.sampling.is_none()
            && matches!(front_end, crate::feature_v2_stream::FrontEnd::Sdr)
            && !(ALL_CHANNELS && omit_xb)
            && (0..n_scales).all(|s| compute.scale_dead(s) || compute.at_scale(s).v2_blocks)
    };
    // Other plain walks (nothing retained, no cached reference) run their
    // strip jobs through the pipeline instead of batches: same merge order,
    // no barrier between batches, production overlapped with the jobs.
    #[cfg(feature = "threads")]
    let mut pipeline =
        (batch_limit > 0 && retention.is_none() && ref_planes.is_none() && !block_walk)
            .then(|| Rev5Pipeline::new(compute, fold_v1, batch_limit, core::mem::take(rev5_jobs)));

    let __t_walk = crate::fold_timing::start();
    #[cfg(feature = "threads")]
    if block_walk {
        rev5_block_walk::<S, D, ALL_CHANNELS>(
            source,
            distorted,
            &dims,
            &compute,
            toggles,
            fold_v1,
            batch_limit,
            !compute.full_res_xb
                && !complete_distorted0
                && !globaff_on
                && !crate::feature_v2_stream::linear_pyramid(),
            rev5_jobs,
            block_pool,
            &mut accums,
            &mut globaff,
            stop,
        );
    }
    loop {
        #[cfg(feature = "threads")]
        if block_walk {
            break;
        }
        if stop.is_some_and(|token| token.check().is_err()) {
            break;
        }
        let __t_prod = crate::fold_timing::start();
        let next = producer.next_strip();
        let Some(info) = next else {
            crate::fold_timing::stop(__t_prod, crate::fold_timing::Phase::Producer, 0);
            break;
        };
        let scale = info.scale;
        if complete_distorted0 && scale == 0 {
            let y1 = info.y0 + info.strip_h;
            let range = info.y0 * info.plane_w..y1 * info.plane_w;
            let ret = retention.as_deref_mut().expect("prepared retaining walk");
            for ch in 0..3 {
                if !compute.channel_active(0, ch) {
                    let rows = producer.rows(
                        crate::feature_v2_stream::Side::Distorted,
                        ch,
                        0,
                        info.y0,
                        y1,
                    );
                    ret.pyr_dst[0][ch][range.clone()].copy_from_slice(rows);
                }
            }
        }
        // `globaff`: every strip of every scale, all three channels, in
        // strip order — before the dead-scale skip, which knows nothing of it.
        if globaff_on {
            let y1 = info.y0 + info.strip_h;
            for (ch, acc) in globaff[scale].iter_mut().enumerate() {
                acc.add_rows(
                    producer.rows(
                        crate::feature_v2_stream::Side::Source,
                        ch,
                        scale,
                        info.y0,
                        y1,
                    ),
                    producer.rows(
                        crate::feature_v2_stream::Side::Distorted,
                        ch,
                        scale,
                        info.y0,
                        y1,
                    ),
                    info.plane_w,
                );
            }
        }
        let local = compute.at_scale(scale);
        let v2_blocks = local.v2_blocks;
        let append_on = local.append;
        let append2_on = local.append2;
        let append2 = append2.filter(|_| append2_on);
        let csfw = csfw.filter(|_| local.csfw);
        let toggles = V2NewFeatureToggles {
            v1_pools: local.v1_pools,
            v1_only: !v2_blocks,
            gradient_features: local.gradient,
            blockiness: local.blockiness,
            ..toggles
        };
        crate::fold_timing::stop(__t_prod, crate::fold_timing::Phase::Producer, scale);
        // A declared-ID plan that reads no v1-era slot and runs no v2-era work at this scale skips its strips whole
        // (the producer has already carried the pyramid through them). Only the pre-Rev5 scale-0 side channel keeps
        // them.
        if compute.scale_dead(scale)
            && !(scale == 0
                && compute.formula_revision < FormulaRevision::Rev5
                && mean_offset.is_some())
        {
            #[cfg(test)]
            SKIPPED_SCALES.with_borrow_mut(|v| *v |= 1 << scale);
            continue;
        }

        #[cfg(feature = "threads")]
        if let Some(pipe) = pipeline.as_mut() {
            // A basic-only cell keeps its existing path; finish every queued
            // cell first so per-channel/scale ordering is never disturbed.
            if !v2_blocks {
                pipe.drain(&mut accums);
            } else {
                for ch in 0..3 {
                    if !ALL_CHANNELS && !compute.channel_active(scale, ch) {
                        continue;
                    }
                    let mut job = pipe.slot(info, ch, &mut accums);
                    job.gather(&producer, info, ch, false);
                    pipe.submit(job, toggles);
                }
                continue;
            }
        }
        #[cfg(feature = "threads")]
        if batch_limit > 0 && pipeline.is_none() {
            // A basic-only cell keeps its existing path; finish older queued
            // cells first so per-channel/scale ordering is never disturbed.
            if !v2_blocks && queued > 0 {
                rev5_run_jobs(
                    &mut rev5_jobs[..queued],
                    &compute,
                    toggles,
                    fold_v1,
                    &mut accums,
                    retention.as_deref_mut(),
                    ref_planes.filter(|_| compute.sampling.is_none()),
                );
                queued = 0;
            }
            if v2_blocks {
                for ch in 0..3 {
                    if !ALL_CHANNELS && !compute.channel_active(scale, ch) {
                        continue;
                    }
                    if queued == rev5_jobs.len() {
                        rev5_jobs.push(Rev5StripJob::new(info, ch));
                    }
                    let job = &mut rev5_jobs[queued];
                    job.gather(
                        &producer,
                        info,
                        ch,
                        compute.sampling.is_none()
                            && rev5_cached_window(ref_planes, &info, ch).is_some(),
                    );
                    queued += 1;
                    if queued == batch_limit {
                        rev5_run_jobs(
                            &mut rev5_jobs[..queued],
                            &compute,
                            toggles,
                            fold_v1,
                            &mut accums,
                            retention.as_deref_mut(),
                            ref_planes.filter(|_| compute.sampling.is_none()),
                        );
                        queued = 0;
                    }
                }
                continue;
            }
        }

        // mean_offset side-channel (fold-engine lane): the scale-0 strips
        // tile [0, h0) exactly once in ascending order, so each row's
        // Σ_x (src − dst) is recorded exactly once here. Emitted for EVERY
        // strip layout — the walk's toggles never change which strips the
        // producer emits, only what is computed from them.
        if scale == 0
            && compute.formula_revision < FormulaRevision::Rev5
            && let Some(mo) = mean_offset.as_deref_mut()
        {
            let __t_mo = crate::fold_timing::start();
            let y1 = info.y0 + info.strip_h;
            let src: [&[f32]; 3] = core::array::from_fn(|ch| {
                producer.rows(crate::feature_v2_stream::Side::Source, ch, 0, info.y0, y1)
            });
            let dst: [&[f32]; 3] = core::array::from_fn(|ch| {
                producer.rows(
                    crate::feature_v2_stream::Side::Distorted,
                    ch,
                    0,
                    info.y0,
                    y1,
                )
            });
            mo.add_strip(info.y0, info.strip_h, info.plane_w, src, dst, parallel);
            crate::fold_timing::stop(__t_mo, crate::fold_timing::Phase::MeanOffset, scale);
        }

        // REV4 C4: the strip's ±1-px dilated dst-Y edge mask — built once
        // pump's slot), so the serial order's X/B phase B has it ready.
        // Only a live chroma channel consumes it (the mask is dst-LUMA
        // edges read by the chroma gradient hooks).
        let bleed_mask: Option<&[f32]> = if local.arttype
            && (ALL_CHANNELS
                || compute.channel_active(scale, 0)
                || compute.channel_active(scale, 2))
        {
            let __t_m = crate::fold_timing::start();
            let mw = bleed_work.as_mut().expect("arttype implies mask work");
            let w = info.plane_w;
            for r in 0..info.strip_h + 4 {
                let gy = info.y0 as isize - 2 + r as isize;
                let gr = reflect_101(gy, info.plane_h);
                let row = producer.rows(
                    crate::feature_v2_stream::Side::Distorted,
                    1,
                    scale,
                    gr,
                    gr + 1,
                );
                mw.dh[r * w..(r + 1) * w].copy_from_slice(row);
            }
            dst_y_edge_mask_strip(w, info.y0, info.strip_h, info.plane_h, mw);
            crate::fold_timing::stop(__t_m, crate::fold_timing::Phase::Rev4Kernel, scale);
            Some(&mw.mask[..w * info.strip_h])
        } else {
            None
        };

        // C8 reads the co-sited B rows directly from the existing producer.
        // These immutable strip views stay live across the channel fan-out.
        let chroma_strip = (local.gmsbank && scale > 0).then(|| ChromaStrip {
            reference: producer.rows(
                crate::feature_v2_stream::Side::Source,
                2,
                scale,
                info.y0,
                info.y0 + info.strip_h,
            ),
            distorted: producer.rows(
                crate::feature_v2_stream::Side::Distorted,
                2,
                scale,
                info.y0,
                info.y0 + info.strip_h,
            ),
        });

        // ref_y strip rows straight from the producer's rolling plane
        // (valid until the next `next_strip` call).
        let need_refy = append_on;
        let refy = if need_refy {
            producer.rows(
                crate::feature_v2_stream::Side::Source,
                1,
                scale,
                info.y0,
                info.y0 + info.strip_h,
            )
        } else {
            &[][..]
        };

        // FUSED PER-CHANNEL FAN-OUT (fold-MT lane). When no channel reads
        // another channel's phase-A output, each worker can finish its own
        // retention and phase B. A split fan-out costs the walk's
        // locality contract: it holds THREE channels' ~9 MB phase-A buffers
        // live across a barrier, then re-reads them from wherever they landed.
        // The serial arm already fuses A->B per channel for exactly that
        // reason (measured +50 ms/pair on aic3-100 when interleaved); this is
        // the same shape under threads.
        //
        // The two preconditions, checked rather than assumed:
        // * `!append_on` => `cross` is `None` on every channel and `refy` is
        //   empty, so Y needs nothing from X/B — the ONLY cross-channel edge
        //   in the walk.
        // * Retention destinations are disjoint channel/row views, so each
        //   worker copies its own outputs before phase B.
        //
        // Byte-neutral by construction: identical kernels on identical inputs
        // into per-channel accumulators that are already disjoint. Only which
        // thread runs what, and when, changes.
        // `fuse_channels` and `self_blur` are HOISTED above the strip loop —
        // they are strip-independent, and the strip scratch is now SIZED from
        // them (`StripPlaneNeeds`), so there is one definition of each.
        //
        // SELF-BLUR BANDS. With `v1_only` phase A's ONLY output is the four
        // H-blurred planes over the strip's whole 148-row wide window, and the
        // v1 bands are its only consumer. Writing them there and reading them
        // back in a different task is four planes through L3/DRAM per strip —
        // which the 16-independent-process throughput test measures as the
        // fold's ceiling (4.2x against buffered's 10.9x from the same serial
        // speed; `benchmarks/fold_mt_scaling_2026-08-31.md`). Letting each
        // band blur exactly the rows it consumes into its own scratch drops
        // phase A entirely and makes every band a self-contained task — the
        // shape `streaming::process_channel_strip` has always had.
        //
        // `V1PoolsMode::Full` or `Peaks` is required because the band-local
        // H buffer lives in the band's `FoldPoolScratch`, and one of those is
        // what every fold-backed SCORE asks for. (`Peaks` hands the band the
        // scratch for the H planes alone — see `BandPoolWork::HOnly`.)
        // ONE owner: `ComputeSet::self_blur_eligible`, which is also what
        // sized the strip scratch above. Re-deriving it here is what let the
        // two drift apart on `Peaks` (see that function's doc); the local
        // `fuse_channels` conjunct stays because it is a runtime parallelism
        // fact the compute set does not model.
        #[cfg(feature = "threads")]
        let self_blur = fuse_channels && retention.is_none() && local.self_blur_eligible();
        #[cfg(feature = "threads")]
        if fuse_channels && retention.is_some() {
            use rayon::prelude::*;
            let __t_f = crate::fold_timing::start();
            let outputs: [Option<RetainedRows<'_>>; 3] = if let Some(ret) = retention.as_deref_mut()
            {
                let sinks = ret.row_sinks().nth(scale).unwrap();
                let mut sinks = sinks.into_iter().enumerate();
                core::array::from_fn(|_| {
                    let (ch, mut sink) = sinks.next().unwrap();
                    if !ALL_CHANNELS && !compute.channel_active(scale, ch) {
                        return None;
                    }
                    sink.take_prefix(info.y0 * info.plane_w);
                    Some(sink.take_prefix(info.strip_h * info.plane_w))
                })
            } else {
                core::array::from_fn(|_| None)
            };
            scratch_strips
                .par_iter_mut()
                .zip(accums.par_iter_mut())
                .zip(outputs.into_par_iter())
                .enumerate()
                .for_each(|(ch, ((scr, acc), mut output))| {
                    if !ALL_CHANNELS && !compute.channel_active(scale, ch) {
                        return;
                    }
                    let __t = crate::fold_timing::start();
                    if rayon::current_num_threads() == 1
                        && retained_core_admits(&local)
                        && let Some(output) = output.as_mut()
                    {
                        rev5_retained_single_channel(
                            &producer,
                            &info,
                            ch,
                            (scr, acc),
                            output,
                            &local,
                            RetainedCoreWork {
                                toggles,
                                fold_v1,
                                free: local.free_work(ch, scale),
                            },
                        );
                        crate::fold_timing::stop(__t, crate::fold_timing::Phase::BBusy, scale);
                        return;
                    }
                    if self_blur {
                        // Phase A is skipped whole; only the wide-window
                        // gather it also performs is still needed.
                        stream_gather_windows(&producer, &info, ch, scr);
                    } else {
                        stream_phase_a(
                            &producer,
                            &info,
                            ch,
                            false,
                            false,
                            v2_blocks,
                            true,
                            scr,
                            compute.formula_revision,
                        );
                    }
                    let (src_win, dst_win) = stream_windows_shared(&producer, &info, ch, scr);
                    if let Some(output) = &mut output {
                        output.copy(&info, scr, src_win, dst_win, false);
                    }
                    stream_phase_b(
                        scr,
                        src_win,
                        dst_win,
                        &info,
                        toggles,
                        fold_v1,
                        v2_blocks,
                        band_parallel,
                        self_blur,
                        false,
                        &[][..],
                        None,
                        append2,
                        csfw,
                        local.free_work(ch, scale),
                        local.rev4_work(scale, ch, bleed_mask, chroma_strip),
                        acc,
                    );
                    crate::fold_timing::stop(__t, crate::fold_timing::Phase::BBusy, scale);
                });
            crate::fold_timing::stop(__t_f, crate::fold_timing::Phase::BWall, scale);
            continue;
        }

        #[cfg(feature = "threads")]
        if fuse_channels {
            use rayon::prelude::*;
            let __t_f = crate::fold_timing::start();
            scratch_strips
                .par_iter_mut()
                .zip(accums.par_iter_mut())
                .enumerate()
                .for_each(|(ch, (scr, acc))| {
                    if !ALL_CHANNELS && !compute.channel_active(scale, ch) {
                        return;
                    }
                    let __t = crate::fold_timing::start();
                    if self_blur {
                        // Phase A is skipped whole; only the wide-window
                        // gather it also performs is still needed.
                        stream_gather_windows(&producer, &info, ch, scr);
                    } else {
                        stream_phase_a(
                            &producer,
                            &info,
                            ch,
                            false,
                            false,
                            v2_blocks,
                            true,
                            scr,
                            compute.formula_revision,
                        );
                    }
                    let (src_win, dst_win) = stream_windows_shared(&producer, &info, ch, scr);
                    stream_phase_b(
                        scr,
                        src_win,
                        dst_win,
                        &info,
                        toggles,
                        fold_v1,
                        v2_blocks,
                        band_parallel,
                        self_blur,
                        false,
                        &[][..],
                        None,
                        append2,
                        csfw,
                        local.free_work(ch, scale),
                        local.rev4_work(scale, ch, bleed_mask, chroma_strip),
                        acc,
                    );
                    crate::fold_timing::stop(__t, crate::fold_timing::Phase::BBusy, scale);
                });
            crate::fold_timing::stop(__t_f, crate::fold_timing::Phase::BWall, scale);
            continue;
        }

        #[cfg(feature = "threads")]
        let ran_parallel = if parallel {
            use rayon::prelude::*;
            // Parallel fan-out keeps the two-phase shape (A for all
            // channels, then B) — Y's cross inputs need X/B's phase A
            // done. Values are identical to the serial order (pure
            // kernels on identical inputs); only cache locality differs,
            // and parallel wall is not the 1-thread gate path.
            let __t_a = crate::fold_timing::start();
            scratch_strips
                .par_iter_mut()
                .enumerate()
                .for_each(|(ch, scr)| {
                    if !ALL_CHANNELS && !compute.channel_active(scale, ch) {
                        return;
                    }
                    let __t = crate::fold_timing::start();
                    stream_phase_a(
                        &producer,
                        &info,
                        ch,
                        append_cell_active(append_on, ch, scale),
                        ch == 1 && act_dst_on && append_cell_active(append_on, 1, scale),
                        v2_blocks,
                        true,
                        scr,
                        compute.formula_revision,
                    );
                    crate::fold_timing::stop(__t, crate::fold_timing::Phase::ABusy, scale);
                });
            crate::fold_timing::stop(__t_a, crate::fold_timing::Phase::AWall, scale);
            let __t_between = crate::fold_timing::start();
            let scratches: &[ScratchV2Strip; 3] = scratch_strips;
            // Retention hooks (appendix N): serial copies between the two
            // fan-outs — pure reads of the phase-A scratches + windows.
            if let Some(ret) = retention.as_deref_mut() {
                for (ch, scr) in scratches.iter().enumerate() {
                    if !ALL_CHANNELS && !compute.channel_active(scale, ch) {
                        continue;
                    }
                    let (src_win, dst_win) = stream_windows_shared(&producer, &info, ch, scr);
                    ret.copy_strip(
                        &info,
                        ch,
                        scr,
                        src_win,
                        dst_win,
                        append_cell_active(append_on, ch, scale),
                    );
                }
            }
            crate::fold_timing::stop(__t_between, crate::fold_timing::Phase::Between, scale);
            let __t_b = crate::fold_timing::start();
            accums.par_iter_mut().enumerate().for_each(|(ch, acc)| {
                if !ALL_CHANNELS && !compute.channel_active(scale, ch) {
                    return;
                }
                let __t = crate::fold_timing::start();
                let (src_win, dst_win) =
                    stream_windows_shared(&producer, &info, ch, &scratches[ch]);
                let cross = if ch == 1 && append_cell_active(append_on, ch, scale) {
                    let off = HALO_P * info.plane_w;
                    let strip_n = info.plane_w * info.strip_h;
                    Some((
                        &scratches[0].activity[off..off + strip_n],
                        &scratches[2].activity[off..off + strip_n],
                    ))
                } else {
                    None
                };
                stream_phase_b(
                    &scratches[ch],
                    src_win,
                    dst_win,
                    &info,
                    toggles,
                    fold_v1,
                    v2_blocks,
                    band_parallel,
                    false,
                    append_cell_active(append_on, ch, scale),
                    refy,
                    cross,
                    append2,
                    csfw,
                    local.free_work(ch, scale),
                    local.rev4_work(scale, ch, bleed_mask, chroma_strip),
                    acc,
                );
                crate::fold_timing::stop(__t, crate::fold_timing::Phase::BBusy, scale);
            });
            crate::fold_timing::stop(__t_b, crate::fold_timing::Phase::BWall, scale);
            true
        } else {
            false
        };
        #[cfg(not(feature = "threads"))]
        let ran_parallel = false;

        if !ran_parallel {
            // Serial: FUSED phase A+B per channel (the locality contract
            // in `stream_phase_b`'s doc — H/V planes stay cache-hot into
            // the band replay + kernels), and ONE shared scratch set for
            // all three channels — each stage overwrites the same ~9 MB
            // in place, so the working set stays L2/L3-resident across
            // channels AND strips (three per-channel sets measured
            // +15 ms/pair on aic3-100: each channel's buffers were
            // evicted by the other two between its consecutive strips).
            // Order X(0), B(2), then Y(1); Y's cross transducer reads
            // X/B's activity STRIP rows, stashed into the (otherwise
            // idle) second/third scratch sets' activity buffers — a
            // 2×strip_n copy, bitwise-identical values.
            let off = HALO_P * info.plane_w;
            let strip_n = info.plane_w * info.strip_h;
            let (s0, s_rest) = scratch_strips.split_at_mut(1);
            let scr = &mut s0[0];
            let (stash_x_buf, stash_b_buf) = s_rest.split_at_mut(1);
            let y_active = append_cell_active(append_on, 1, scale);
            for ch in [0usize, 2] {
                if !ALL_CHANNELS && !compute.channel_active(scale, ch) {
                    continue;
                }
                let active = append_cell_active(append_on, ch, scale);
                stream_phase_a(
                    &producer,
                    &info,
                    ch,
                    active,
                    false,
                    v2_blocks,
                    false,
                    scr,
                    compute.formula_revision,
                );
                if y_active {
                    let stash = if ch == 0 {
                        &mut stash_x_buf[0].activity
                    } else {
                        &mut stash_b_buf[0].activity
                    };
                    stash[..strip_n].copy_from_slice(&scr.activity[off..off + strip_n]);
                }
                let (src_win, dst_win) = stream_windows_shared(&producer, &info, ch, scr);
                if let Some(ret) = retention.as_deref_mut() {
                    ret.copy_strip(&info, ch, scr, src_win, dst_win, active);
                }
                stream_phase_b(
                    scr,
                    src_win,
                    dst_win,
                    &info,
                    toggles,
                    fold_v1,
                    v2_blocks,
                    band_parallel,
                    false,
                    active,
                    refy,
                    None,
                    append2,
                    csfw,
                    local.free_work(ch, scale),
                    local.rev4_work(scale, ch, bleed_mask, chroma_strip),
                    &mut accums[ch],
                );
            }
            {
                stream_phase_a(
                    &producer,
                    &info,
                    1,
                    y_active,
                    y_active && act_dst_on,
                    v2_blocks,
                    false,
                    scr,
                    compute.formula_revision,
                );
                let cross = if y_active {
                    Some((
                        &stash_x_buf[0].activity[..strip_n],
                        &stash_b_buf[0].activity[..strip_n],
                    ))
                } else {
                    None
                };
                let (src_win, dst_win) = stream_windows_shared(&producer, &info, 1, scr);
                if let Some(ret) = retention.as_deref_mut() {
                    ret.copy_strip(&info, 1, scr, src_win, dst_win, y_active);
                }
                stream_phase_b(
                    scr,
                    src_win,
                    dst_win,
                    &info,
                    toggles,
                    fold_v1,
                    v2_blocks,
                    band_parallel,
                    false,
                    y_active,
                    refy,
                    cross,
                    append2,
                    csfw,
                    local.free_work(1, scale),
                    local.rev4_work(scale, 1, bleed_mask, chroma_strip),
                    &mut accums[1],
                );
            }
        }
    }

    #[cfg(feature = "threads")]
    if queued > 0 {
        rev5_run_jobs(
            &mut rev5_jobs[..queued],
            &compute,
            toggles,
            fold_v1,
            &mut accums,
            retention.as_deref_mut(),
            ref_planes.filter(|_| compute.sampling.is_none()),
        );
    }
    #[cfg(feature = "threads")]
    if let Some(mut pipe) = pipeline.take() {
        pipe.drain(&mut accums);
        *rev5_jobs = core::mem::take(&mut pipe.free);
    }

    if let Some(t) = __t_walk {
        crate::fold_timing::walk_done(t.elapsed().as_nanos() as u64);
    }
    producer.recycle(stream_pool);
    if let Some(ret) = retention.as_deref_mut() {
        for (ch, acc) in accums.iter_mut().enumerate() {
            if let Some(strips) = acc.retained_strips.take() {
                for (scale, values) in strips.into_iter().enumerate() {
                    ret.strips[scale][ch] = values;
                }
            }
        }
    }

    // --- Finalize: the materialized walk's per-scale epilogue, replayed
    //     on the merged accumulators (identical f64 ops on identical
    //     values ⇒ identical bits). ---
    let v1_total = if fold_v1 { n_scales * 3 * 31 } else { 0 };
    let v12_total = v1_total + n_scales * 3 * FEATURES_PER_CHANNEL_V2_TOTAL;
    let append_total = if layout_append {
        n_scales * 3 * FEATURES_PER_CHANNEL_APPEND
    } else {
        0
    };
    let append2_total = if layout_append2 {
        n_scales * APPEND2_PER_SCALE
    } else {
        0
    };
    let csfw_total = if layout_csfw {
        n_scales * CSFW_PER_SCALE
    } else {
        0
    };
    let reserved_total = if layout_gridblk || layout_ringbasis || layout_tailhist || layout_arttype
    {
        30
    } else {
        0
    };
    // REV4 tail blocks — emit order `gridblk | ringbasis | tailhist |
    // except PerScale `arttype`.
    let gridblk_total = if layout_gridblk {
        n_scales * 3 * GRIDBLK_PER_CELL
    } else {
        0
    };
    let ringbasis_total = if layout_ringbasis {
        n_scales * 3 * RINGBASIS_PER_CELL
    } else {
        0
    };
    let tailhist_total = if layout_tailhist {
        n_scales * 3 * TAILHIST_PER_CELL
    } else {
        0
    };
    let arttype_total = if layout_arttype {
        n_scales * ARTTYPE_PER_SCALE
    } else {
        0
    };
    let gmsbank_total = if layout_gmsbank {
        gmsbank_width(n_scales)
    } else {
        0
    };
    let mapdev_total = if layout_mapdev {
        n_scales * 3 * MAPDEV_PER_CELL
    } else {
        0
    };
    let z1max_total = if layout_z1max {
        n_scales * 3 * Z1MAX_PER_CELL
    } else {
        0
    };
    let gmsnative_total = if layout_gmsnative { GMSNATIVE_WIDTH } else { 0 };
    let before_tail = v12_total
        + append_total
        + append2_total
        + csfw_total
        + reserved_total
        + gridblk_total
        + ringbasis_total
        + tailhist_total
        + arttype_total
        + gmsbank_total
        + mapdev_total
        + z1max_total
        + gmsnative_total;
    // `globaff` sits at its registered base: the reserved slots and the
    // research palette block before it (f1820..1866) are structural zeros.
    let (reserved_tail_total, globaff_total) = if layout_globaff {
        let base =
            crate::feature_defs::block_base(crate::feature_set_id::ComputeToken::GlobAff, n_scales)
                .expect("globaff is a registered block")
                .0;
        assert!(
            base >= before_tail,
            "globaff base {base} precedes the walk tail {before_tail}"
        );
        (
            base - before_tail,
            n_scales * 3 * crate::globaff::GLOBAFF_PER_CELL,
        )
    } else {
        (0, 0)
    };
    let mut features = vec![0.0f64; before_tail + reserved_tail_total + globaff_total];
    let (features_v12, features_tail) = features.split_at_mut(v12_total);
    let (features_app, features_tail2) = features_tail.split_at_mut(append_total);
    let (features_app2, features_tail3) = features_tail2.split_at_mut(append2_total);
    let (features_csfw, features_tail4) = features_tail3.split_at_mut(csfw_total);
    let (_reserved, features_tail5) = features_tail4.split_at_mut(reserved_total);
    let (features_gridblk, features_tail6) = features_tail5.split_at_mut(gridblk_total);
    let (features_ring, features_tail7) = features_tail6.split_at_mut(ringbasis_total);
    let (features_thist, features_tail8) = features_tail7.split_at_mut(tailhist_total);
    let (features_art, features_tail9) = features_tail8.split_at_mut(arttype_total);
    let (features_gmsbank, features_tail10) = features_tail9.split_at_mut(gmsbank_total);
    let (features_mapdev, features_tail11) = features_tail10.split_at_mut(mapdev_total);
    let (features_z1max, features_tail12) = features_tail11.split_at_mut(z1max_total);
    let (features_gmsnative, features_tail13) = features_tail12.split_at_mut(gmsnative_total);
    let (_reserved_tail, features_globaff) = features_tail13.split_at_mut(reserved_tail_total);
    let mut prev_grad: [Option<(f64, f64)>; 3] = [None; 3];

    #[allow(clippy::needless_range_loop)] // scale derives 3+ offsets across distinct arrays
    for scale in 0..n_scales {
        let local = compute.at_scale(scale);
        let v2_blocks = local.v2_blocks;
        let append_on = local.append;
        let append2_on = local.append2;
        let toggles = V2NewFeatureToggles {
            v1_pools: local.v1_pools,
            v1_only: !v2_blocks,
            gradient_features: local.gradient,
            blockiness: local.blockiness,
            ..toggles
        };
        let (width, height) = dims[scale];
        let n = width * height;
        let scale_base = scale * 3 * FEATURES_PER_CHANNEL_V2_TOTAL;
        let app_scale_base = scale * 3 * FEATURES_PER_CHANNEL_APPEND;
        let mut grads: [(f64, f64); 3] = [(0.0, 0.0); 3];

        for (ch, acc) in accums.iter().enumerate() {
            if !ALL_CHANNELS && !compute.channel_active(scale, ch) {
                continue;
            }
            let out = &mut features_v12
                [v1_total + scale_base + ch * FEATURES_PER_CHANNEL_V2_TOTAL..]
                [..FEATURES_PER_CHANNEL_V2_TOTAL];
            let sum_blockiness = if toggles.blockiness {
                acc.block[scale].0.fin() + acc.block[scale].1.fin()
            } else {
                0.0
            };
            grads[ch] = finish_channel_scale_at_revision(
                &acc.dense[scale],
                &acc.grad[scale],
                sum_blockiness,
                n,
                out,
                compute.omit_pjnd_fragility,
                compute.formula_revision,
            );
            if append_cell_active(append_on, ch, scale) {
                let out_app = &mut features_app
                    [app_scale_base + ch * FEATURES_PER_CHANNEL_APPEND..]
                    [..FEATURES_PER_CHANNEL_APPEND];
                finish_append(
                    &acc.dense[scale],
                    &acc.app[scale],
                    &acc.grad[scale],
                    n,
                    ch == 1,
                    toggles,
                    out_app,
                );
            }
            apply_transducer_luma_gate(out, ch, toggles);
        }

        // Retention hooks (appendix N): the EXACT pooled cells + the
        // mean-gradients this finalize just derived — the fused compare's
        // coefficient inputs (the standalone derives them from its own
        // 1e-9-parity pass-A replication instead).
        if let Some(ret) = retention.as_deref_mut() {
            for (ch, acc) in accums.iter().enumerate() {
                if !ALL_CHANNELS && !compute.channel_active(scale, ch) {
                    continue;
                }
                ret.cells[scale][ch] = AttrCellSums {
                    dense: acc.dense[scale],
                    grad: acc.grad[scale],
                    app: acc.app[scale],
                    blockiness: if toggles.blockiness {
                        acc.block[scale].0.fin() + acc.block[scale].1.fin()
                    } else {
                        0.0
                    },
                    n,
                };
                ret.mg[scale][ch] = grads[ch];
            }
        }

        if let Some(ret) = retention.as_deref_mut()
            && ret.retain_basic
        {
            let channels = core::array::from_fn(|ch| {
                if ALL_CHANNELS || compute.channel_active(scale, ch) {
                    accums[ch].v1[scale].basic_raw()
                } else {
                    crate::fused::StripChannelAccum::zero()
                }
            });
            ret.basic_hf[scale] = channels
                .each_ref()
                .map(|value| (value.hf_sq_src, value.hf_abs_src));
            ret.basic_stats[scale] = crate::streaming::ScaleAccumulators::from_basic(
                channels,
                n,
                compute.formula_revision,
            );
        }
        if fold_v1 {
            for (ch, acc) in accums.iter().enumerate() {
                if !ALL_CHANNELS && !compute.channel_active(scale, ch) {
                    continue;
                }
                let base = scale * 39 + ch * 13;
                acc.v1[scale].finalize_into(
                    n,
                    &mut features_v12[base..base + 13],
                    compute.formula_revision,
                );
                // FREE EXTRAS ([`V1FreeExtras::RawMoments`]): the v2-era
                // slots this walk can finalize from the fused kernel's raw
                // moments, written ONLY where the owning block is off — so
                // this can never race the append/append2 finalize below.
                // Index-stable: every other slot of those blocks keeps its
                // structural 0.0.
                // `!= Off`, not `== RawMoments`: every variant is a
                // SUPERSET of the raw moments, so an exact-equality test
                // here silently zeroes these 40 slots the moment a wider
                // variant is requested (caught by
                // `class_c_extras_are_pure_addition_to_the_free_walk`).
                if toggles.free_extras != V1FreeExtras::Off {
                    let (globals, luma_mean) =
                        acc.v1[scale].finalize_free(n, toggles.formula_revision);
                    // `GLOBAL_*` live in the append block. The 944 walk skips
                    // the (B, scale 0) append cell entirely
                    // (`APPEND_SKIP_B_SCALE0`), so this must skip it too or
                    // the emitted row would not be the 944 row.
                    if layout_append && !append_on && append_cell_active(true, ch, scale) {
                        let apb = app_scale_base + ch * FEATURES_PER_CHANNEL_APPEND;
                        features_app[apb + idx_append::GLOBAL_DMEAN] = globals[0];
                        features_app[apb + idx_append::GLOBAL_CGAIN] = globals[1];
                        features_app[apb + idx_append::GLOBAL_CLOSS] = globals[2];
                    }
                    // `LUMA_MEAN_REF` is append2's, Y-only.
                    if layout_append2 && !append2_on && ch == APPEND2_CHANNEL {
                        features_app2[scale * APPEND2_PER_SCALE + idx_append2::LUMA_MEAN_REF] =
                            luma_mean;
                    }
                }
                // CLASS-C FREE EXTRAS: the bounded-error slots, written under
                // the same "only where the owning block is off" rule. The v2
                // finalize above already ran and left `MSE` at its structural
                // `clamp01(0/n) = 0.0` when `v2_blocks` is off, so this
                // overwrite is the value's first and only real write.
                if toggles.free_extras == V1FreeExtras::RawMomentsPlusBoundedErr {
                    let (mse, lum) = acc.v1[scale].finalize_free_bounded(n);
                    if !v2_blocks {
                        features_v12[v1_total
                            + scale_base
                            + ch * FEATURES_PER_CHANNEL_V2_TOTAL
                            + idx::MSE] = mse;
                    }
                    // The luminance bins are append's, and Y-only for the
                    // same register-carry reason the kernel is.
                    if layout_append
                        && !append_on
                        && ch == APPEND2_CHANNEL
                        && append_cell_active(true, ch, scale)
                    {
                        let apb = app_scale_base + ch * FEATURES_PER_CHANNEL_APPEND;
                        features_app[apb + idx_append::LUM_DARK_ERR] = lum[0];
                        features_app[apb + idx_append::LUM_MID_ERR] = lum[1];
                        features_app[apb + idx_append::LUM_BRIGHT_ERR] = lum[2];
                    }
                }
                if toggles.v1_pools != V1PoolsMode::Off {
                    // v1's block-major 372 layout: [basic 156][peaks 72]
                    // [masked 72][iw 72], each pool block scale-major then
                    // channel-major, 6 per (scale, ch) (`metric.rs` passes
                    // 2/3/4). `n_scales`-generic: 13/6/6/6 per (scale, ch).
                    let cell = (scale * 3 + ch) * 6;
                    let peaks0 = n_scales * 39 + cell;
                    let masked0 = n_scales * 57 + cell;
                    let iw0 = n_scales * 75 + cell;
                    let mut peaks = [0.0f64; 6];
                    let mut masked = [0.0f64; 6];
                    let mut iw = [0.0f64; 6];
                    acc.v1[scale].finalize_pools_into(
                        n,
                        &mut peaks,
                        &mut masked,
                        &mut iw,
                        compute.formula_revision,
                    );
                    if toggles.v1_pools == V1PoolsMode::Full {
                        features_v12[peaks0..peaks0 + 6].copy_from_slice(&peaks);
                        features_v12[masked0..masked0 + 6].copy_from_slice(&masked);
                        features_v12[iw0..iw0 + 6].copy_from_slice(&iw);
                    } else if toggles.v1_pools == V1PoolsMode::Peaks {
                        // Peaks only; the masked/IW sums were never
                        // accumulated, so their slots stay the structural 0.
                        features_v12[peaks0..peaks0 + 6].copy_from_slice(&peaks);
                    } else {
                        // Carriers: only the ten `fused944native` slots go
                        // live (peak slot 4 = art_l8, masked/iw slot 3 =
                        // art_4th); everything else stays the structural 0.
                        for &slot in V1PoolsMode::CARRIER_SLOTS.iter() {
                            if slot == peaks0 + 4 {
                                features_v12[slot] = peaks[4];
                            } else if slot == masked0 + 3 {
                                features_v12[slot] = masked[3];
                            } else if slot == iw0 + 3 {
                                features_v12[slot] = iw[3];
                            }
                        }
                    }
                }
            }
        }

        for ch in 0..3 {
            let (gsrc, gdst) = grads[ch];
            if let Some((prev_gsrc, prev_gdst)) = prev_grad[ch]
                && compute.at_scale(scale).gradient
                && compute.at_scale(scale - 1).gradient
            {
                let decay_src = gsrc / (prev_gsrc + C_GRAD_DECAY);
                let decay_dst = gdst / (prev_gdst + C_GRAD_DECAY);
                let prev_base = v1_total
                    + (scale - 1) * 3 * FEATURES_PER_CHANNEL_V2_TOTAL
                    + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
                features_v12[prev_base + idx::EDGE_WIDTH_CHANGE] = bounded_dissim(
                    decay_src,
                    decay_dst,
                    C_EDGEWIDTH,
                    compute.formula_revision.stable_forms(),
                );
            }
            prev_grad[ch] = Some((gsrc, gdst));

            if scale == n_scales - 1 && n_scales >= 2 {
                let this_base = v1_total + scale_base + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
                let prev_base = v1_total
                    + (scale - 1) * 3 * FEATURES_PER_CHANNEL_V2_TOTAL
                    + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
                features_v12[this_base + idx::EDGE_WIDTH_CHANGE] =
                    features_v12[prev_base + idx::EDGE_WIDTH_CHANGE];
            }
        }
    }

    // --- append2 finalize (Y-only, per scale): BANDVIS means from the Y
    //     gradient accumulators, the free luminance conditioner from the
    //     Y append accumulator's Σs, and the HDR highlight bins. ---
    if append2_on {
        #[allow(clippy::needless_range_loop)] // scale derives offsets across distinct arrays
        for scale in 0..n_scales {
            if !compute.at_scale(scale).append2 {
                continue;
            }
            let (width, height) = dims[scale];
            let n_f = (width * height) as f64;
            let base = scale * APPEND2_PER_SCALE;
            let gy = &accums[1].grad[scale];
            let ay = &accums[1].app[scale];
            let out2 = &mut features_app2[base..base + APPEND2_PER_SCALE];
            if toggles.gradient_features {
                out2[idx_append2::BANDVIS_GAIN] = (gy.sum_bv_gain / n_f).clamp(0.0, 1.0);
                out2[idx_append2::BANDVIS_LOSS] = (gy.sum_bv_loss / n_f).clamp(0.0, 1.0);
            }
            // Reference-only conditioner: sat(mean ref-Y, C_LUM_T) — the
            // same `t` mapping the luminance bins use per pixel.
            out2[idx_append2::LUMA_MEAN_REF] = saturate(ay.sum_s / n_f, C_LUM_T).clamp(0.0, 1.0);
            // HL bins: WeightedSum.finish() is 0 when Σw ≈ 0 (SDR route
            // or no highlight mass).
            out2[idx_append2::HL_BIN1] = ay.ws_hl1.finish().clamp(0.0, 1.0);
            out2[idx_append2::HL_BIN2] = ay.ws_hl2.finish().clamp(0.0, 1.0);
        }
    }

    // --- CSFW finalize (Y-only, per scale): the weighted GLOBAL_* twins
    //     from the Y CSFW accumulators (design §4.1). ---
    if csfw_on {
        for scale in 0..n_scales {
            if !compute.at_scale(scale).csfw {
                continue;
            }
            let base = scale * CSFW_PER_SCALE;
            finish_csfw(
                &accums[1].csfw[scale],
                &mut features_csfw[base..base + CSFW_PER_SCALE],
            );
        }
    }

    // --- REV4 finalize (per scale): C1 cells rescan their stored `ẽ`
    //     planes; C2/C3 fold their accumulators; C4's blur reads the
    //     already-emitted EWC_Y × HF_LOSS_Y — final only after the scale
    //     loop's cross-scale EWC fill, which is why this sits after it.
    if layout_gridblk || layout_ringbasis || layout_tailhist || layout_arttype {
        #[allow(clippy::needless_range_loop)] // scale derives offsets across distinct arrays
        for scale in 0..n_scales {
            let local = compute.at_scale(scale);
            if !local.gridblk && !local.ringbasis && !local.tailhist && !local.arttype {
                continue;
            }
            let (width, height) = dims[scale];
            let cells = [
                &accums[0].rev4[scale],
                &accums[1].rev4[scale],
                &accums[2].rev4[scale],
            ];
            let (ewc_y, hfl_y) = if local.arttype {
                let yb = v1_total
                    + scale * 3 * FEATURES_PER_CHANNEL_V2_TOTAL
                    + FEATURES_PER_CHANNEL_V2_TOTAL;
                (
                    features_v12[yb + idx::EDGE_WIDTH_CHANGE],
                    features_v12[yb + idx::HF_LOSS],
                )
            } else {
                (0.0, 0.0)
            };
            finish_rev4_scale(
                &local,
                scale,
                cells,
                width,
                height,
                ewc_y,
                hfl_y,
                if layout_gridblk {
                    Some(
                        &mut features_gridblk
                            [scale * 3 * GRIDBLK_PER_CELL..(scale + 1) * 3 * GRIDBLK_PER_CELL],
                    )
                } else {
                    None
                },
                if layout_ringbasis {
                    Some(
                        &mut features_ring
                            [scale * 3 * RINGBASIS_PER_CELL..(scale + 1) * 3 * RINGBASIS_PER_CELL],
                    )
                } else {
                    None
                },
                if layout_tailhist {
                    Some(
                        &mut features_thist
                            [scale * 3 * TAILHIST_PER_CELL..(scale + 1) * 3 * TAILHIST_PER_CELL],
                    )
                } else {
                    None
                },
                if layout_arttype {
                    Some(
                        &mut features_art
                            [scale * ARTTYPE_PER_SCALE..(scale + 1) * ARTTYPE_PER_SCALE],
                    )
                } else {
                    None
                },
            );
        }
    }

    if layout_gmsbank {
        for (scale, &(width, height)) in dims.iter().enumerate().take(n_scales) {
            if !compute.at_scale(scale).gmsbank {
                continue;
            }
            for (ch, channel) in accums.iter().enumerate() {
                if scale == 0 && ch != 1 {
                    continue;
                }
                let base = if scale == 0 {
                    0
                } else {
                    15 + (scale - 1) * 55 + ch * 15
                };
                finish_gmsbank_cell(
                    &channel.grad[scale].bank,
                    width * height,
                    &mut features_gmsbank[base..base + GMSBANK_PER_CELL],
                );
            }
            if scale > 0 {
                let base = 15 + (scale - 1) * 55 + 45;
                finish_chroma_cell(
                    &accums[0].grad[scale].chroma,
                    width * height,
                    &mut features_gmsbank[base..base + 10],
                );
            }
        }
    }

    if layout_gmsnative && compute.at_scale(0).gmsnative {
        assert!(
            compute.full_res_xb,
            "gmsnative needs the native-scale X/B strips (full_res_xb)"
        );
        let (width, height) = dims[0];
        for (cell, ch) in [0usize, 2].into_iter().enumerate() {
            finish_gmsbank_cell(
                &accums[ch].grad[0].bank,
                width * height,
                &mut features_gmsnative[cell * GMSBANK_PER_CELL..(cell + 1) * GMSBANK_PER_CELL],
            );
        }
    }

    let restore_work = restore_cuts::Work {
        mapdev: layout_mapdev && compute.mapdev,
        z1max: layout_z1max && compute.z1max,
    };
    // --- globaff finalize: `(share, explained)` per (scale, channel), in the
    //     registry's `2·(3·scale + channel) + signal` order. ---
    if globaff_on {
        for (scale, cells) in globaff.iter().enumerate() {
            for (ch, acc) in cells.iter().enumerate() {
                let (share, explained) = acc.finish();
                let at = crate::globaff::GLOBAFF_PER_CELL * (3 * scale + ch);
                features_globaff[at] = share;
                features_globaff[at + 1] = explained;
            }
        }
    }
    if restore_work.any() {
        assert!(
            restore_side_ok,
            "mapdev/z1max need the SDR pair path (no ref-fed, sampled or HDR walk)"
        );
        restore_cuts::run(
            source,
            distorted,
            parallel,
            compute.formula_revision,
            restore_work,
            features_mapdev,
            features_z1max,
        );
    }
    ZensimV2Result {
        features,
        n_scales,
        v1_pools: if fold_v1 {
            toggles.v1_pools
        } else {
            V1PoolsMode::Off
        },
        regime: match (
            layout_append,
            layout_append2,
            layout_csfw,
            layout_gridblk && layout_ringbasis && layout_tailhist && layout_arttype,
        ) {
            (true, true, true, true) => FeatureRegime::Folded720Rev4,
            (true, true, true, false) => FeatureRegime::Folded720Csfw,
            (true, true, false, _) => FeatureRegime::Folded720Append2,
            (true, false, _, _) => FeatureRegime::Folded720Append,
            (false, _, _, _) => FeatureRegime::Folded720,
        },
    }
}

fn compute_v2_features_with_ref_impl_inner(
    prepared: &V2PreparedReference,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    toggles: V2NewFeatureToggles,
    scratch: &mut V2Scratch,
    // REV4 cell out-param — `None` on every production call (the
    // materialized walk never emits rev4 slots; the streaming walk owns
    // that tail). `Some(&mut [n_scales * 3 cells])` — flat `(scale*3+ch)`
    // order — turns the same strip walk into the materialized half of the
    // streaming-vs-materialized parity gate: identical hook math into
    // caller-visible accumulators.
    mut rev4_cells: Option<&mut [Rev4CellAccum]>,
) -> Result<ZensimV2Result, ZensimError> {
    if distorted.width() != prepared.orig_width || distorted.height() != prepared.orig_height {
        return Err(ZensimError::DimensionMismatch);
    }
    crate::metric::reject_hdr_input(distorted)?;
    crate::metric::check_within_max_pixels(distorted.width(), distorted.height(), max_pixels)?;

    // Distorted-side reflect-pad only when genuinely below the pyramid
    // minimum. At ≥64px `reflect_pad_to_min` degenerates to an identity
    // copy (reflect_index(i, n) == i for i < n), so skipping it changes
    // no plane values — it only removes a full per-pixel copy of the
    // image from every pair.
    let mut dst_planes = if distorted.width() < crate::metric::MIN_PYRAMID_DIM
        || distorted.height() < crate::metric::MIN_PYRAMID_DIM
    {
        let padded_dst = crate::metric::reflect_pad_to_min(distorted);
        crate::streaming::convert_source_to_xyb(
            &padded_dst,
            padded_dst.width(),
            parallel,
            crate::ssim_form::active_revision(),
        )
    } else {
        crate::streaming::convert_source_to_xyb(
            distorted,
            distorted.width(),
            parallel,
            crate::ssim_form::active_revision(),
        )
    };

    let (mut width, mut height) = (prepared.scales[0].1, prepared.scales[0].2);
    debug_assert_eq!(
        dst_planes[0].len(),
        width * height,
        "distorted planes must match the prepared reference's scale-0 dims \
         (same original dims + same pad rule guarantee this)"
    );

    let n_scales = prepared.scales.len();
    if let Some(c) = rev4_cells.as_ref() {
        assert_eq!(
            c.len(),
            n_scales * 3,
            "rev4 cell slice must carry one cell per (scale, channel)"
        );
    }
    // REV4's resolved compute flags for this call — the same
    // `ComputeSet::at_scale`/`rev4_work` gating the streaming walk uses
    // (minus `bleed_mask`, which `compute_channel_scale_v2`'s strip loop
    // builds from `bleed_y`).
    let rev4_compute = rev4_cells
        .is_some()
        .then(|| ComputeSet::from_toggles(toggles));
    let mut features = vec![0.0f64; n_scales * 3 * FEATURES_PER_CHANNEL_V2_TOTAL];
    // Per-channel (mean_grad_src, mean_grad_dst) from the previous
    // (finer) scale, for the edge-width-change cross-scale comparison.
    let mut prev_grad: [Option<(f64, f64)>; 3] = [None; 3];

    // Perf pass (§A.13) / phase-5 (§A.15): one scratch buffer set PER
    // CHANNEL, sized for one STRIP+halo at the largest (scale-0) `width`
    // (`width * (STRIP_ROWS + 2*HALO_P)` — NOT `width*height` anymore;
    // this is the actual memory-traffic reduction §A.15 exists to
    // measure). 3 separate sets (rather than 1 reused across channels
    // too) so the 3 channels within a scale can run independently — see
    // the `threads` parallel branch below, which needs disjoint `&mut`
    // scratch per closure. The sets live in the caller-supplied
    // [`V2Scratch`] so batch drivers pay the allocation once per worker,
    // not once per pair.
    //
    // Phase-6 (§A.16 lever B): `compute_channel_scale_v2_whole` (the
    // small-image bypass) needs `width*height` rows of scratch — bump the
    // allocation to cover whichever is larger (`STRIP_BYPASS_HEIGHT=0`
    // today makes this `.min()` a permanent no-op).
    #[allow(clippy::unnecessary_min_or_max)]
    let bypass_rows = height.min(STRIP_BYPASS_HEIGHT);
    let strip_max_n = width * (STRIP_ROWS + 2 * HALO_P).max(bypass_rows);
    scratch.ensure(strip_max_n);
    let scratch = &mut scratch.strips;

    for scale in 0..n_scales {
        let scale_base = scale * 3 * FEATURES_PER_CHANNEL_V2_TOTAL;

        // Reference planes for this scale come straight from the prepared
        // pyramid (read-only); only the distorted planes are walked
        // in-place below.
        let (src_planes, ref_w, ref_h) = &prepared.scales[scale];
        debug_assert_eq!(
            (*ref_w, *ref_h),
            (width, height),
            "prepared reference scale dims must track the distorted pyramid walk"
        );
        // Cached reference-side moments for this scale (if prepared with
        // them): per-channel (mu1, activity) full planes.
        let moments_scale = prepared.moments.as_ref().map(|m| &m[scale]);
        let moments_for =
            |ch: usize| moments_scale.map(|ms| (&ms[ch].mu1[..], &ms[ch].activity[..]));

        // Each channel's compute is fully independent WITHIN a scale (only
        // `prev_grad[ch]` from the SAME channel's earlier scale is read,
        // below, after all 3 channels finish, and each channel owns its
        // own scratch set) -- so the 3-channel fan-out can run in
        // parallel when both the `threads` feature and the caller's
        // `parallel` flag are active.
        let mut grads: [(f64, f64); 3] = [(0.0, 0.0); 3];
        let out_region = &mut features[scale_base..][..3 * FEATURES_PER_CHANNEL_V2_TOTAL];

        // REV4's at-scale flags for this scale — resolved once, handed to
        // `rev4_cell_args` per channel (mirrors `rev4_work` minus the mask).
        let local = rev4_compute.as_ref().map(|cs| cs.at_scale(scale));

        #[cfg(feature = "threads")]
        let ran_parallel = if parallel {
            use rayon::prelude::*;
            let cell_opts: Vec<Option<&mut Rev4CellAccum>> = match rev4_cells.as_deref_mut() {
                Some(c) => c[scale * 3..scale * 3 + 3].iter_mut().map(Some).collect(),
                None => vec![None, None, None],
            };
            let results: Vec<(f64, f64)> = out_region
                .chunks_mut(FEATURES_PER_CHANNEL_V2_TOTAL)
                .collect::<Vec<_>>()
                .into_par_iter()
                .zip(scratch.par_iter_mut())
                .zip(cell_opts)
                .enumerate()
                .map(|(ch, ((out, scr), cell))| {
                    let g = compute_channel_scale_v2(
                        &src_planes[ch],
                        &dst_planes[ch],
                        width,
                        height,
                        toggles,
                        moments_for(ch),
                        scr,
                        out,
                        cell.map(|c| {
                            rev4_cell_args(
                                c,
                                local.as_ref().unwrap(),
                                scale,
                                ch,
                                src_planes,
                                &dst_planes,
                            )
                        }),
                    );
                    apply_transducer_luma_gate(out, ch, toggles);
                    g
                })
                .collect();
            grads = [results[0], results[1], results[2]];
            true
        } else {
            false
        };
        #[cfg(not(feature = "threads"))]
        let ran_parallel = false;

        if !ran_parallel {
            for (ch, out) in out_region
                .chunks_mut(FEATURES_PER_CHANNEL_V2_TOTAL)
                .enumerate()
            {
                let r4a = rev4_cells.as_deref_mut().map(|c| {
                    rev4_cell_args(
                        &mut c[scale * 3 + ch],
                        local.as_ref().unwrap(),
                        scale,
                        ch,
                        src_planes,
                        &dst_planes,
                    )
                });
                grads[ch] = compute_channel_scale_v2(
                    &src_planes[ch],
                    &dst_planes[ch],
                    width,
                    height,
                    toggles,
                    moments_for(ch),
                    &mut scratch[ch],
                    out,
                    r4a,
                );
                apply_transducer_luma_gate(out, ch, toggles);
            }
        }

        for ch in 0..3 {
            let (gsrc, gdst) = grads[ch];
            if let Some((prev_gsrc, prev_gdst)) = prev_grad[ch] {
                // Edge-width-change belongs to the FINER (previous) scale's
                // slot -- it's "how much did this scale's edges widen by
                // the time we reach the next coarser scale".
                let decay_src = gsrc / (prev_gsrc + C_GRAD_DECAY);
                let decay_dst = gdst / (prev_gdst + C_GRAD_DECAY);
                let prev_base = (scale - 1) * 3 * FEATURES_PER_CHANNEL_V2_TOTAL
                    + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
                features[prev_base + idx::EDGE_WIDTH_CHANGE] = bounded_dissim(
                    decay_src,
                    decay_dst,
                    C_EDGEWIDTH,
                    toggles.formula_revision.stable_forms(),
                );
            }
            prev_grad[ch] = Some((gsrc, gdst));

            if scale == n_scales - 1 && n_scales >= 2 {
                // Coarsest scale has no next scale to compare against --
                // duplicate the previous (second-coarsest) scale's value
                // (documented approximation; see module/spec doc).
                let this_base = scale_base + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
                let prev_base = (scale - 1) * 3 * FEATURES_PER_CHANNEL_V2_TOTAL
                    + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
                features[this_base + idx::EDGE_WIDTH_CHANGE] =
                    features[prev_base + idx::EDGE_WIDTH_CHANGE];
            }
        }

        if scale + 1 < n_scales {
            // BUG TRAP: all 3 channels must downscale from the SAME
            // (width, height) — only update the outer width/height AFTER
            // the per-channel loop (see the test-suite-caught corruption
            // this comment originally documented).
            //
            // Only the DISTORTED pyramid is walked here — the reference
            // levels were materialized once in `prepared`.
            let mut new_wh = (width, height);
            for plane in dst_planes.iter_mut() {
                new_wh = crate::blur::downscale_2x_inplace(plane, width, height);
            }
            width = new_wh.0;
            height = new_wh.1;
        }
    }

    Ok(ZensimV2Result {
        features,
        n_scales,
        v1_pools: V1PoolsMode::Off,
        regime: FeatureRegime::V2Bounded,
    })
}

// ============================================================================
// Attribution densities for the v2 (f372-719) + append (f720-923) blocks
// (task #67 C2a — the coverage levers the C1 basic density measured missing).
//
// Same contract as `crate::attribution`: per-pixel density whose rectangle
// sum over a block approximates `Σ_k s_k·Δf_k(block)` — the first-order
// score change from refining the block to reference — with TRUE integrands
// and ABSOLUTE normalization. `s` slices are the RAW model gradient
// `∂score/∂f_k`; the sign fold happens via the `density = Σ_k s_k·(δf_k)_i`
// rule ((δf_k)_i = the first-order feature change from refining pixel i's
// content, e.g. `−v_i/N` for a mean-pooled error signal).
//
// Integrand classes (all derived from the production kernels in this file):
//  - mean pools (v2: ssim_mean/art/det/mse/hf×3/pjnd×3/gms/ringing/banding;
//    append: xmask/lumt/mscn×2/contrast×2/tex): (δf)_i = −v_i/N. Exact.
//  - reference-weighted pools (masked/iw, append luminance bins):
//    (δf)_i = −w_i·v_i/Σw (w reference-side). Exact.
//  - self-weighted soft-peak (w_i = sat(v_i, C_PEAK), f = Σw·v/Σw):
//    (δf)_i = w(v_i)·(f − v_i)/Σw — SIGNED; full-plane sum is exactly 0.
//  - deviation pools (v2 SSIM_DEV2/DEV4; append GMS/ART/DET_DEV2):
//    central-moment chain rule, SIGNED (zeroing a mean-valued pixel RAISES
//    the deviation). DEV2: (δf)_i = (2μv−v²)/(2f·N); DEV4 via raw moments.
//  - global slots (append GLOBAL_DMEAN/CGAIN/CLOSS): exact chain rule on
//    whole-plane means/variances, SIGNED.
//  - blockiness: lattice-step terms with one-sided shares per step end plus a per-step pair term
//    applied by rectangle queries at cut steps (exact for footprint-aligned rectangle repairs; the
//    sampling-geometry path keeps the symmetric 50/50 split).
//  - EDGE_WIDTH_CHANGE: exact two-scale chain rule on the adjacent-scale
//    mean-gradient ratios (incl. the last-scale copy's weight).
//  - reference-only (PJND_FRAGILITY, GRAD_SRC_MEAN): (δf)_i = 0 exactly.
//  - structural zeros (X/B transducer slots, the (B, scale 0) append cell):
//    features are constant 0.0 in the regime ⇒ Δf ≡ 0 ⇒ integrand 0,
//    regardless of the probed gradient (mirrors the harness's f156-371
//    structural-zero handling).
//
// The finalize clamps (`clamp01`/`clamp02`) are treated as inert (features
// live strictly inside their bounds on real content; at a saturated clamp
// the true gradient is 0 and the density would over-attribute — accepted
// and documented as an approximation).
//
// Pass A replays the MATERIALIZED strip walk (STRIP_ROWS + HALO_P halos,
// `run_blur_pass_strip`, the σ-split `bs2` chain from `stream_phase_a`) and
// then runs the PRODUCTION kernels over the cached planes, so every pooled
// scalar feeding a coefficient is production-arithmetic (parity-gated by
// `v2_append_attr_features_match_production`). Pass B re-derives the
// per-pixel signals in f64 from the same planes — the density's plane sums
// therefore match production features to per-pixel f32-vs-f64 recompute
// noise (~1e-7 rel), not 1e-9; the pooled-scalar parity is the strict gate.
// ============================================================================

// Coarse per-section timers for ZENSIM_ATTR_PERF=1 (seconds, thread-local).
thread_local! {
    static PERF_BLUR: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
    static PERF_KERN: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
    static PERF_PASSB: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}

/// Output of [`compute_v2_append_attribution`].
#[cfg(feature = "custom-profiles")]
pub(crate) struct V2AppendAttribution {
    /// Lattice-step boundary terms (`crate::attribution::LatticeScale`) belonging to `density`.
    pub lattice: Vec<crate::attribution::LatticeScale>,
    /// Full-resolution density (f64, `width × height`, trimmed to the
    /// original image), in score units per pixel.
    pub density: Vec<f64>,
    pub width: usize,
    pub height: usize,
    /// Pass-A finalized v2 features (`n_scales·3·29`, scale-major) — the
    /// production-kernel pooled values the coefficients were derived from.
    /// Read by the 1e-9 production-parity gate
    /// (`v2_append_attr_features_match_production`); retained on the struct
    /// as the audit surface for any consumer of the density.
    #[allow(dead_code)]
    pub v2_features: Vec<f64>,
    /// Pass-A finalized append features (`n_scales·3·17`), empty when the
    /// append block was not requested. Same audit role as `v2_features`.
    #[allow(dead_code)]
    pub append_features: Vec<f64>,
}

/// Per-(scale, channel) pass-A pooled state.
///
/// Stays compiled in every configuration (it is a [`FoldRetention`] field
/// type, and the walk's retention hooks write it), but its fields are only
/// READ by the `custom-profiles`-gated attribution-density cluster — hence
/// the targeted dead-code allowance.
#[cfg_attr(not(feature = "custom-profiles"), allow(dead_code))]
#[derive(Default, Clone, Copy)]
pub(crate) struct AttrCellSums {
    pub(crate) dense: DenseAccum,
    pub(crate) grad: GradientAccum,
    app: AppendAccum,
    pub(crate) blockiness: f64,
    pub(crate) n: usize,
}

/// Per-scale per-channel cached planes (scale-plane sized slices of
/// scale-0-sized buffers).
pub(crate) struct AttrChPlanes {
    pub(crate) mu1: Vec<f32>,
    pub(crate) mu2: Vec<f32>,
    pub(crate) ssq: Vec<f32>,
    pub(crate) s12: Vec<f32>,
    pub(crate) act: Vec<f32>,
    bs2: Vec<f32>,
}

impl AttrChPlanes {
    #[cfg(feature = "custom-profiles")]
    fn new(n: usize) -> Self {
        Self::new_for(n, true)
    }
    fn new_for(n: usize, want_bs2: bool) -> Self {
        Self {
            mu1: vec![0.0; n],
            mu2: vec![0.0; n],
            ssq: vec![0.0; n],
            s12: vec![0.0; n],
            act: vec![0.0; n],
            bs2: vec![0.0; if want_bs2 { n } else { 0 }],
        }
    }
}

/// Blur + cache one (scale, channel): materialized strip walk
/// (`gather_strip_halo` + `run_blur_pass_strip` + the `stream_phase_a`
/// σ-split chain for `bs2`), core rows copied into `planes`.
#[cfg(feature = "custom-profiles")]
fn attr_blur_cache_channel(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    want_bs2: bool,
    scratch: &mut ScratchV2Strip,
    planes: &mut AttrChPlanes,
) {
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
        gather_strip_halo(
            dst,
            width,
            height,
            y0,
            wide_h,
            HALO_P,
            &mut scratch.dst_wide[..n_wide],
        );
        run_blur_pass_strip(width, wide_h, scratch);
        if want_bs2 {
            // Same chain as `stream_phase_a`: square(src) → H-box → V-box.
            // `abs_src`/`activity_tmp` are free after the activity chain.
            square_into(&scratch.src_wide[..n_wide], &mut scratch.abs_src[..n_wide]);
            crate::blur::box_blur_h(
                &scratch.abs_src[..n_wide],
                &mut scratch.activity_tmp[..n_wide],
                width,
                wide_h,
                BLUR_RADIUS,
            );
            crate::blur::box_blur_v_from_copy(
                &scratch.activity_tmp[..n_wide],
                &mut scratch.bs2[..n_wide],
                width,
                wide_h,
                BLUR_RADIUS,
            );
        }
        let off = HALO_P * width;
        let strip_n = width * strip_h;
        let out = y0 * width;
        planes.mu1[out..out + strip_n].copy_from_slice(&scratch.mu1[off..off + strip_n]);
        planes.mu2[out..out + strip_n].copy_from_slice(&scratch.mu2[off..off + strip_n]);
        planes.ssq[out..out + strip_n].copy_from_slice(&scratch.ssq[off..off + strip_n]);
        planes.s12[out..out + strip_n].copy_from_slice(&scratch.s12[off..off + strip_n]);
        planes.act[out..out + strip_n].copy_from_slice(&scratch.activity[off..off + strip_n]);
        if want_bs2 {
            planes.bs2[out..out + strip_n].copy_from_slice(&scratch.bs2[off..off + strip_n]);
        }
        y0 += strip_h;
    }
}

/// Pass A kernels for one (scale, channel) over cached planes: production
/// dense/gradient/append kernels in kernel-strip order + canonical
/// blockiness. `cross` = `(act_x, act_b)` full planes for the Y channel.
#[cfg(feature = "custom-profiles")]
#[allow(clippy::too_many_arguments)]
fn attr_pass_a_kernels(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    planes: &AttrChPlanes,
    cross: Option<(&[f32], &[f32])>,
    ref_y: &[f32],
    want_append: bool,
    append_active: bool,
    grad_halo: &mut Vec<f32>,
    grad_halo_d: &mut Vec<f32>,
) -> AttrCellSums {
    let mut cell = AttrCellSums {
        n: width * height,
        ..Default::default()
    };
    let mut y0 = 0usize;
    while y0 < height {
        let strip_h = STRIP_ROWS.min(height - y0);
        let base = y0 * width;
        let strip_n = width * strip_h;
        let src_s = &src[base..base + strip_n];
        let dst_s = &dst[base..base + strip_n];
        let d = dense_block_kernel(
            src_s,
            dst_s,
            &planes.mu1[base..base + strip_n],
            &planes.mu2[base..base + strip_n],
            &planes.ssq[base..base + strip_n],
            &planes.s12[base..base + strip_n],
            &planes.act[base..base + strip_n],
            width,
            strip_h,
            true, // transducer_bank (V2NewFeatureToggles::default())
            None,
        );
        cell.dense.accumulate(&d);
        // Gradient needs a 1-row halo; gather from the full planes (real
        // rows except the reflect_101 mirror at the true image edges —
        // identical values to the production wide-window slices).
        let g_n = width * (strip_h + 2);
        grad_halo.resize(g_n, 0.0);
        grad_halo_d.resize(g_n, 0.0);
        gather_strip_halo(src, width, height, y0, strip_h + 2, 1, grad_halo);
        gather_strip_halo(dst, width, height, y0, strip_h + 2, 1, grad_halo_d);
        let g = gradient_block_kernel(
            grad_halo,
            grad_halo_d,
            &planes.act[base..base + strip_n],
            width,
            strip_h,
            None,
            None,
            None,
            false,
        );
        cell.grad.accumulate(&g);
        if want_append && append_active {
            let a = append_block_kernel(
                src_s,
                dst_s,
                &planes.mu1[base..base + strip_n],
                &planes.mu2[base..base + strip_n],
                &planes.ssq[base..base + strip_n],
                &planes.bs2[base..base + strip_n],
                &planes.act[base..base + strip_n],
                &ref_y[base..base + strip_n],
                cross.map(|(x, b)| (&x[base..base + strip_n], &b[base..base + strip_n])),
                false, // hl (append2) — not part of the 924 regime
                width,
                strip_h,
            );
            cell.app.accumulate(&a);
        }
        y0 += strip_h;
    }
    cell.blockiness = blockiness_sparse(src, dst, width, height);
    cell
}

/// All deferred pass-B coefficients for one (scale, channel), derived from
/// the raw gradients and the pass-A pooled sums. Field semantics: each is
/// the multiplier on the named per-pixel term in the pass-B combine (the
/// `s_k` sign fold and `1/N` already applied).
#[cfg(feature = "custom-profiles")]
#[derive(Default, Clone, Copy)]
struct V2AppCoeffs {
    // v2 mean-pool slots: coefficient × per-pixel value.
    c_ssim: f64,
    c_art: f64,
    c_det: f64,
    c_mse: f64,
    c_hf_gain: f64,
    c_hf_loss: f64,
    c_hf_mag: f64,
    c_pjnd: f64,
    c_pjnd_lo: f64,
    c_pjnd_hi: f64,
    c_gms: f64,
    c_ringing: f64,
    c_banding: f64,
    c_blockiness: f64,
    // v2 weighted pools: coefficient × (w · v); w recomputed per pixel.
    c_mask_ssim: f64,
    c_mask_art: f64,
    c_mask_det: f64,
    c_mask_mse: f64,
    c_iw_ssim: f64,
    c_iw_art: f64,
    c_iw_det: f64,
    c_iw_mse: f64,
    // soft-peak: s_k/W and the pooled f per member.
    sp_ssim: (f64, f64),
    sp_art: (f64, f64),
    sp_det: (f64, f64),
    // ssim dev pools: μ, raw2, raw3, the third central moment, f2, f4 and the two s_k.
    dev_mu: f64,
    dev_raw2: f64,
    dev_raw3: f64,
    dev_c3: f64,
    dev_f2: f64,
    dev_f4: f64,
    s_dev2: f64,
    s_dev4: f64,
    // The two deviation pools' density as one polynomial in `e = d − dev_mu`,
    // guards and the map split folded in:
    // `q0 + e·(q1 + e·(q2 + e·(q3 + e·q4)))`.
    dev_q: [f64; 5],
    // append mean slots.
    c_xmask: f64,
    c_lumt: f64,
    c_mscn: f64,
    c_mscn2: f64,
    c_cgain: f64,
    c_closs: f64,
    c_tex: f64,
    // append luminance bins: coefficient × (w · mse_i) per bin.
    c_dark: f64,
    c_mid: f64,
    c_bright: f64,
    // append dev pools (gms/art/det): (s_k, μ, f) per member.
    gd_gms: (f64, f64, f64),
    gd_art: (f64, f64, f64),
    gd_det: (f64, f64, f64),
    // The same pools' densities as `q[0] + e·(q[1] + e·q[2])` in `e = v − q[3]`.
    gq_gms: [f64; 4],
    gq_art: [f64; 4],
    gq_det: [f64; 4],
    // append globals.
    g_dmean: f64, // coefficient × (s − d)
    g_var: f64,   // coefficient × ((s − d) − g_shift)·((s + d) − g_sum)
    g_shift: f64,
    g_sum: f64,
    // edge-width: coefficient × (|∇d| − |∇s|).
    c_edgew: f64,
    // append2 BANDVIS pair (f924+, Y-only): coefficient × the per-pixel
    // FR-excess indicator. Mean pools like the v2 hf pair — `−s_k/N`.
    c_bv_gain: f64,
    c_bv_loss: f64,
    // The forms of the computation that produced the planes, never the process's: whether `s12` holds the
    // direct error moment (Rev3+), and whether ringing reads the high-pass residual (revision 6 or the
    // research ring-form override). A bake served at its declared revision in a process of another one
    // retains planes of its own revision.
    direct: bool,
    ring_hp: bool,
    // Whether the planes' revision uses the cancellation-free forms (revision 7).
    stable: bool,
}

/// A deviation pool's centred quadratic density (`V2AppCoeffs::gq_art`):
/// `q[0] + e·(q[1] + e·q[2])` with `e = v − q[3]`.
#[cfg(feature = "custom-profiles")]
#[inline(always)]
fn gq_density(q: &[f64; 4], v: f64) -> f64 {
    let e = v - q[3];
    q[0] + e * (q[1] + e * q[2])
}

#[cfg(feature = "custom-profiles")]
#[allow(clippy::too_many_arguments)]
fn derive_v2app_coeffs(
    s_v2: &[f64],
    s_append: Option<&[f64]>,
    s_append2: Option<&[f64]>,
    scale: usize,
    ch: usize,
    cell: &AttrCellSums,
    append_active: bool,
    cross: bool,
    edgew_coeff: f64,
    revision: FormulaRevision,
    split: crate::map_paint::SplitMode,
) -> V2AppCoeffs {
    use crate::map_paint::SplitMode;
    let n_f = cell.n as f64;
    let inv_n = 1.0 / n_f;
    let v2b = scale * 3 * FEATURES_PER_CHANNEL_V2_TOTAL + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
    let gv2 = |slot: usize| s_v2.get(v2b + slot).copied().unwrap_or(0.0);
    let apb = scale * 3 * FEATURES_PER_CHANNEL_APPEND + ch * FEATURES_PER_CHANNEL_APPEND;
    let gap = |slot: usize| -> f64 {
        s_append
            .and_then(|s| s.get(apb + slot))
            .copied()
            .unwrap_or(0.0)
    };
    let dense = &cell.dense;
    let app = &cell.app;
    let grad = &cell.grad;

    let mut c = V2AppCoeffs {
        // Mean pools: density += s_k · (−v_i/N).
        c_ssim: -gv2(idx::SSIM_MEAN) * inv_n,
        c_art: -gv2(idx::ART) * inv_n,
        c_det: -gv2(idx::DET) * inv_n,
        c_mse: -gv2(idx::MSE) * inv_n,
        c_hf_gain: -gv2(idx::HF_GAIN) * inv_n,
        c_hf_loss: -gv2(idx::HF_LOSS) * inv_n,
        c_hf_mag: -gv2(idx::HF_MAG_LOSS) * inv_n,
        c_pjnd: -gv2(idx::PJND_TRANSDUCER) * inv_n,
        c_pjnd_lo: -gv2(idx::PJND_TRANSDUCER_LOW_K) * inv_n,
        c_pjnd_hi: -gv2(idx::PJND_TRANSDUCER_HIGH_K) * inv_n,
        c_gms: -gv2(idx::GMS) * inv_n,
        c_ringing: -gv2(idx::RINGING) * inv_n,
        c_banding: -gv2(idx::BANDING) * inv_n,
        c_blockiness: -gv2(idx::BLOCKINESS) * inv_n,
        c_edgew: edgew_coeff,
        direct: revision >= FormulaRevision::Rev3,
        ring_hp: revision.ring_high_pass() || active_ring_form() == RingForm::HighPassMu,
        stable: revision.stable_forms(),
        ..Default::default()
    };

    // ── append2 (f924-943; `idx_append2`), Y-only, no channel axis ────────
    // Layout `f924 + scale*APPEND2_PER_SCALE + local`. Registered
    // per-slot decomposability (campaign appendix E.1 / the
    // `slot_decomposability` TSV) — the block splits, it is not uniform:
    //
    //  * BANDVIS_GAIN / BANDVIS_LOSS — class **E**. Plain mean over the
    //    plane of a per-pixel `bounded_excess_pair` indicator, i.e. the
    //    exact pooling form of the v2 HF_GAIN/HF_LOSS/HF_MAG_LOSS slots:
    //    `(δf)_i = −v_i/N`. Spatialized below.
    //  * LUMA_MEAN_REF — class **N by definition**: `sat(mean(ref Y))` is
    //    REFERENCE-ONLY, so `∂f/∂(distorted) ≡ 0` and a zero density is the
    //    exactly-correct answer (same footing as v2 `PJND_FRAGILITY` and
    //    append `GRAD_SRC_MEAN`, which also carry no term here).
    //  * HL_BIN1 / HL_BIN2 — class **N (structural zero) on this route**.
    //    Their form (`Σw·mse_i/Σw`, reference-side w) IS class-E, identical
    //    to the append luminance bins — but they are computed only under the
    //    HDR const-generic `HL`, and this attribution path is structurally
    //    SDR (`attr_pass_a_kernels` passes `hl = false`; both sides come
    //    through the SDR `prepare_v2_reference_impl`). On the SDR route
    //    `Σw ≡ 0` ⇒ `WeightedSum::finish() ≡ 0` ⇒ the feature is identically
    //    0.0 ⇒ `Δf ≡ 0` for any probed gradient — the same footing as the
    //    X/B transducer slots and the `APPEND_SKIP_B_SCALE0` cell.
    //
    // So exactly 2 of the 5 local slots carry a term, and the other 3 are
    // deliberately, explicitly zero rather than silently unreached.
    if ch == APPEND2_CHANNEL {
        let ap2b = scale * APPEND2_PER_SCALE;
        let gap2 = |slot: usize| -> f64 {
            s_append2
                .and_then(|s| s.get(ap2b + slot))
                .copied()
                .unwrap_or(0.0)
        };
        c.c_bv_gain = -gap2(idx_append2::BANDVIS_GAIN) * inv_n;
        c.c_bv_loss = -gap2(idx_append2::BANDVIS_LOSS) * inv_n;
        // idx_append2::LUMA_MEAN_REF / HL_BIN1 / HL_BIN2: no term, by the
        // classification above. Deliberate omission, not an oversight.
    }

    // Weighted pools: density += s_k · (−w_i·v_i/Σw). Σw is shared per
    // family (same weight for all 4 members).
    let mask_den = dense.ws_mask_ssim.den;
    if mask_den > 0.0 {
        c.c_mask_ssim = -gv2(idx::MASKED_SSIM) / mask_den;
        c.c_mask_art = -gv2(idx::MASKED_ART) / mask_den;
        c.c_mask_det = -gv2(idx::MASKED_DET) / mask_den;
        c.c_mask_mse = -gv2(idx::MASKED_MSE) / mask_den;
    }
    let iw_den = dense.ws_iw_ssim.den;
    if iw_den > 0.0 {
        c.c_iw_ssim = -gv2(idx::IW_SSIM) / iw_den;
        c.c_iw_art = -gv2(idx::IW_ART) / iw_den;
        c.c_iw_det = -gv2(idx::IW_DET) / iw_den;
        c.c_iw_mse = -gv2(idx::IW_MSE) / iw_den;
    }
    // Soft-peak (self-weighted): density += s_k · w(v_i)·(f − v_i)/W.
    let sp = |ws: &WeightedSum, sk: f64| -> (f64, f64) {
        if ws.den > 0.0 {
            (sk / ws.den, ws.num / ws.den)
        } else {
            (0.0, 0.0)
        }
    };
    c.sp_ssim = sp(&dense.ws_peak_ssim, gv2(idx::SSIM_SOFT_PEAK));
    c.sp_art = sp(&dense.ws_peak_art, gv2(idx::ART_SOFT_PEAK));
    c.sp_det = sp(&dense.ws_peak_det, gv2(idx::DET_SOFT_PEAK));
    if split == SplitMode::Exact {
        // The pixel's share `w(v)·v/W` of the soft peak `Σw(v)·v / Σw(v)`.
        c.sp_ssim.1 = 0.0;
        c.sp_art.1 = 0.0;
        c.sp_det.1 = 0.0;
    }
    // SSIM deviation pools (central moments of the d map).
    let mu = dense.sum_d * inv_n;
    let raw2 = dense.sum_d2 * inv_n;
    let raw3 = dense.sum_d3 * inv_n;
    let raw4 = dense.sum_d4 * inv_n;
    let m2 = (raw2 - mu * mu).max(0.0);
    let m4 = (raw4 - 4.0 * mu * raw3 + 6.0 * mu * mu * raw2 - 3.0 * mu.powi(4)).max(0.0);
    let (m2, m4) = if dense.central.n > 0.0 {
        (dense.central.m2 * inv_n, dense.central.m4 * inv_n)
    } else {
        (m2, m4)
    };
    c.dev_mu = mu;
    c.dev_raw2 = raw2;
    c.dev_raw3 = raw3;
    // The third central moment, from the stable block moments when the walk
    // kept them (Rev5+), else from the raw sums.
    c.dev_c3 = if dense.central.n > 0.0 {
        dense.central.m3 * inv_n
    } else {
        raw3 - 3.0 * mu * raw2 + 2.0 * mu.powi(3)
    };
    c.dev_f2 = m2.sqrt();
    c.dev_f4 = m4.quarter_root(crate::det_math::active_root_form());
    c.s_dev2 = gv2(idx::SSIM_DEV2);
    c.s_dev4 = gv2(idx::SSIM_DEV4);
    // Both deviation pools as one polynomial in `e = d − μ`, which does not
    // cancel when the deviation is small against the mean (raw powers of `d`
    // carry coefficients that grow as `1/f³`).
    //  * repair (pixel `i` made perfect, `d → 0`, first order):
    //    `k2·(μ² − e²)` and `k4·(μ⁴ − e⁴ + 4d·c3)`, `k2 = s2/(2·f2·N)`,
    //    `k4 = s4/(4·f4³·N)`;
    //  * exact (the pixel's share `d·∂f/∂d`, negated; both pools are
    //    homogeneous of degree 1 in `d`, so the shares sum to the feature):
    //    `−(s2/(f2·N))·d·e` and `−(s4/(f4³·N))·d·(e³ − c3)`.
    let mut q = [0.0f64; 5];
    if c.s_dev2 != 0.0 && c.dev_f2 > 1e-12 {
        match split {
            SplitMode::Repair => {
                let k = c.s_dev2 * inv_n / (2.0 * c.dev_f2);
                q[0] += k * mu * mu;
                q[2] -= k;
            }
            SplitMode::Exact => {
                // d·e = e² + μ·e
                let k = -c.s_dev2 * inv_n / c.dev_f2;
                q[1] += k * mu;
                q[2] += k;
            }
        }
    }
    if c.s_dev4 != 0.0 && c.dev_f4 > 1e-12 {
        let c3 = c.dev_c3;
        match split {
            SplitMode::Repair => {
                // μ⁴ − e⁴ + 4(e + μ)·c3
                let k = c.s_dev4 * inv_n / (4.0 * c.dev_f4.powi(3));
                q[0] += k * (mu.powi(4) + 4.0 * mu * c3);
                q[1] += k * 4.0 * c3;
                q[4] -= k;
            }
            SplitMode::Exact => {
                // d·(e³ − c3) = e⁴ + μ·e³ − c3·e − μ·c3
                let k = -c.s_dev4 * inv_n / c.dev_f4.powi(3);
                q[0] -= k * mu * c3;
                q[1] -= k * c3;
                q[3] += k * mu;
                q[4] += k;
            }
        }
    }
    c.dev_q = q;

    if append_active {
        if cross {
            c.c_xmask = -gap(idx_append::XMASK_TRANSDUCER) * inv_n;
            c.c_lumt = -gap(idx_append::LUM_TRANSDUCER) * inv_n;
        }
        c.c_mscn = -gap(idx_append::MSCN_DIFF_MEAN) * inv_n;
        c.c_mscn2 = -gap(idx_append::MSCN_DIFF_L2) * inv_n;
        c.c_cgain = -gap(idx_append::CONTRAST_GAIN) * inv_n;
        c.c_closs = -gap(idx_append::CONTRAST_LOSS) * inv_n;
        c.c_tex = -gap(idx_append::TEXTURE_DISSIM) * inv_n;
        if app.ws_dark.den > 0.0 {
            c.c_dark = -gap(idx_append::LUM_DARK_ERR) / app.ws_dark.den;
        }
        let mid_den = n_f - app.ws_dark.den - app.ws_bright.den;
        if mid_den > 0.0 {
            c.c_mid = -gap(idx_append::LUM_MID_ERR) / mid_den;
        }
        if app.ws_bright.den > 0.0 {
            c.c_bright = -gap(idx_append::LUM_BRIGHT_ERR) / app.ws_bright.den;
        }
        // Append deviation pools: (s_k, μ, f).
        let gd = |sk: f64, sum: f64, sum2: f64| -> (f64, f64, f64) {
            let m = sum * inv_n;
            let f = ((sum2 * inv_n) - m * m).max(0.0).sqrt();
            (sk, m, f)
        };
        c.gd_gms = gd(gap(idx_append::GMS_DEV2), grad.sum_gms, grad.sum_gms2);
        c.gd_art = gd(gap(idx_append::ART_DEV2), dense.sum_art, app.sum_art2);
        c.gd_det = gd(gap(idx_append::DET_DEV2), dense.sum_det, app.sum_det2);
        // The same deviation pools as quadratics in `e = v − m`: repair
        // `s·(2m·v − v²)/(2f·N) = −(s/(2fN))·(e² − m²)`; exact, the share
        // `v·∂f/∂v` negated, `−(s/(fN))·v·(v − m) = −(s/(fN))·(e² + m·e)`.
        let gq = |g: (f64, f64, f64)| -> [f64; 4] {
            let (sk, m, f) = g;
            if sk == 0.0 || f <= 1e-12 {
                return [0.0; 4];
            }
            match split {
                SplitMode::Repair => {
                    let k = -sk * inv_n / (2.0 * f);
                    [-k * m * m, 0.0, k, m]
                }
                SplitMode::Exact => {
                    let k = -sk * inv_n / f;
                    [0.0, k * m, k, m]
                }
            }
        };
        c.gq_gms = gq(c.gd_gms);
        c.gq_art = gq(c.gd_art);
        c.gq_det = gq(c.gd_det);
        // Globals. GLOBAL_DMEAN: f = sat(|Δ|, C), Δ = (Σs−Σd)/N. Repair:
        // (δf)_i = −sat'(|Δ|)·sign(Δ)·(s_i−d_i)/N. Exact: the secant share
        // (f/Δ)·(s_i−d_i)/N, with f/Δ = sign(Δ)/(|Δ| + C) bounded at Δ → 0.
        let delta = (app.sum_s - app.sum_d) * inv_n;
        c.g_dmean = match split {
            SplitMode::Repair => {
                let sat_p = C_GDMEAN / ((delta.abs() + C_GDMEAN) * (delta.abs() + C_GDMEAN));
                gap(idx_append::GLOBAL_DMEAN) * (-sat_p * delta.signum() * inv_n)
            }
            SplitMode::Exact if delta != 0.0 => {
                gap(idx_append::GLOBAL_DMEAN)
                    * (-(delta.signum() / (delta.abs() + C_GDMEAN)) * inv_n)
            }
            SplitMode::Exact => 0.0,
        };
        // GLOBAL_CGAIN/CLOSS: chain rule on gvar2 (gvar1 is reference-side).
        let gmean_s = app.sum_s * inv_n;
        let gmean_d = app.sum_d * inv_n;
        let gvar1 = (app.sum_s2 * inv_n - gmean_s * gmean_s).max(0.0);
        let gvar2 = (app.sum_d2 * inv_n - gmean_d * gmean_d).max(0.0);
        let denom = gvar1 + gvar2 + C_GCONTRAST;
        match split {
            SplitMode::Repair => {
                // δgvar2 per refined pixel = (s−d)·((s+d) − 2·gmean_d)/N.
                let df_dg2 = if gvar2 > gvar1 {
                    gap(idx_append::GLOBAL_CGAIN) * ((2.0 * gvar1 + C_GCONTRAST) / (denom * denom))
                } else if gvar1 > gvar2 {
                    gap(idx_append::GLOBAL_CLOSS) * (-(2.0 * gvar1 + C_GCONTRAST) / (denom * denom))
                } else {
                    0.0
                };
                c.g_var = df_dg2 * inv_n;
                c.g_shift = 0.0;
                c.g_sum = 2.0 * gmean_d;
            }
            SplitMode::Exact => {
                // GLOBAL_CGAIN = (gvar2 − gvar1)/denom, GLOBAL_CLOSS its
                // negation: the pixel's share of `gvar2 − gvar1` is
                // ((d−μd)² − (s−μs)²)/N, factored as
                // −((s−d) − (μs−μd))·((s+d) − (μs+μd))/N so it does not cancel.
                c.g_var = if gvar2 > gvar1 {
                    gap(idx_append::GLOBAL_CGAIN) * inv_n / denom
                } else if gvar1 > gvar2 {
                    -gap(idx_append::GLOBAL_CLOSS) * inv_n / denom
                } else {
                    0.0
                };
                c.g_shift = gmean_s - gmean_d;
                c.g_sum = gmean_s + gmean_d;
            }
        }
        // GRAD_SRC_MEAN (slot 16): reference-only ⇒ integrand 0 (no term).
    }
    c
}

/// Pass B for one (scale, channel): f64 per-pixel combine over the cached
/// planes. Three sweeps: the main per-pixel loop, the gradient-family loop
/// (needs ±1-row neighbors), and the sparse blockiness lattice.
///
/// C2b bleed allocation (MEASURED variant: window-only): mass is routed by
/// SUPPORT class into two planes — `id_plane` (everything pixel-supported
/// plus, per the 8-cell A/B below, the residual-form signals) and
/// `win_plane` (the pure blur-window signals: the ssim `d` map, the
/// var-based contrast/texture signals, and the ssim-derived pools — whose
/// per-pixel values are functions of K-blurred planes only). The caller
/// spreads `win_plane` ONCE per scale with the sum-preserving box spread.
/// The 50/50 residual split (art/det/hf/mscn half-spread) was measured and
/// REGRESSED all 8 gate cells (−0.01..−0.08); the pure `I − K` adjoint for
/// residual signals allocates zero net mass and is structurally wrong for
/// removal semantics — both recorded in the C2b benchmark section.
#[cfg(feature = "custom-profiles")]
#[allow(clippy::too_many_arguments)]
fn attr_pass_b_channel(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    planes: &AttrChPlanes,
    cross: Option<(&[f32], &[f32])>,
    ref_y: &[f32],
    co: &V2AppCoeffs,
    parallel: bool,
    id_plane: &mut [f64],
    win_plane: &mut [f64],
    lattice: Option<(&mut crate::attribution::LatticeScale, usize, usize)>,
) {
    // Row-banded parallel main+gradient sweeps (C2b Part 2): each band
    // writes only its own rows of id/win (disjoint), reads shared planes
    // (gradient reads ±1 row of src/dst — immutable). Blockiness stays
    // serial (it writes the row ABOVE at lattice rows). Only the rayon arm
    // bands, so the const is `threads`-gated with it.
    #[cfg(feature = "threads")]
    const BAND: usize = 64;
    #[cfg(feature = "threads")]
    if parallel && height > BAND {
        use rayon::prelude::*;
        id_plane[..width * height]
            .par_chunks_mut(BAND * width)
            .zip(win_plane[..width * height].par_chunks_mut(BAND * width))
            .enumerate()
            .for_each(|(band, (id_rows, win_rows))| {
                let y0 = band * BAND;
                let y1 = (y0 + BAND).min(height);
                attr_pass_b_rows(
                    src, dst, width, height, planes, cross, ref_y, co, y0, y1, id_rows, win_rows,
                );
            });
        attr_pass_b_blockiness(src, dst, width, height, co, id_plane, lattice);
        return;
    }
    let _ = parallel;
    attr_pass_b_rows(
        src, dst, width, height, planes, cross, ref_y, co, 0, height, id_plane, win_plane,
    );
    attr_pass_b_blockiness(src, dst, width, height, co, id_plane, lattice);
}

/// Rows `[y0, y1)` of the main + gradient sweeps; `id_rows`/`win_rows` are
/// the band-local output slices (row `y` writes at `(y − y0) * width`).
#[cfg(feature = "custom-profiles")]
#[allow(clippy::too_many_arguments)]
fn attr_pass_b_rows(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    planes: &AttrChPlanes,
    cross: Option<(&[f32], &[f32])>,
    ref_y: &[f32],
    co: &V2AppCoeffs,
    y0: usize,
    y1: usize,
    id_plane: &mut [f64],
    win_plane: &mut [f64],
) {
    // Revision 3 (issue #61): the `s12` plane is the direct error moment, so
    // the v2 SSIM signal must read it as such. The planes' own revision
    // (`V2AppCoeffs::direct`); loop-invariant, so every call site below
    // unswitches on it.
    let direct = co.direct;
    let ring_hp = co.ring_hp;
    let out_off = y0 * width;
    let cross_on = cross.is_some();
    for y in y0..y1 {
        let row = y * width;
        for x in 0..width {
            let i = row + x;
            let s = src[i] as f64;
            let dd = dst[i] as f64;
            let m1 = planes.mu1[i] as f64;
            let m2 = planes.mu2[i] as f64;
            let sq = planes.ssq[i] as f64;
            let s12v = planes.s12[i] as f64;
            let act = planes.act[i] as f64;

            // Class accumulators: pixel-supported (a_id), window-supported
            // (a_win), and residual-form (a_res; routed to id per the
            // measured A/B — see the fn doc).
            let mut a_id = 0.0f64;
            let mut a_win = 0.0f64;
            let mut a_res = 0.0f64;

            // Dense family (same formulas as `dense_block_kernel`'s tail).
            let d = ssim_d_local(m1, m2, s12v, sq, direct, co.stable);
            let diff_src = (s - m1).abs();
            let diff_dst = (dd - m2).abs();
            let edge_dissim = bounded_dissim(diff_src, diff_dst, C_EDGE, co.stable);
            let (art_i, det_i) = if diff_dst > diff_src {
                (edge_dissim, 0.0)
            } else if diff_dst < diff_src {
                (0.0, edge_dissim)
            } else {
                (0.0, 0.0)
            };
            let raw_diff = s - dd;
            let raw_abs_err = raw_diff.abs();
            let mse_i = saturate(raw_diff * raw_diff, C_MSE);
            a_win += co.c_ssim * d;
            a_res += co.c_art * art_i + co.c_det * det_i;
            a_id += co.c_mse * mse_i;

            let hf_src = s - m1;
            let hf_dst = dd - m2;
            let (hf_gain_i, hf_loss_i) =
                bounded_excess_pair(hf_dst * hf_dst, hf_src * hf_src, C_HF);
            a_res += co.c_hf_gain * hf_gain_i + co.c_hf_loss * hf_loss_i;
            a_res += co.c_hf_mag * bounded_excess(hf_src.abs(), hf_dst.abs(), C_HF);

            a_id += co.c_pjnd * pjnd_transducer(raw_abs_err, act, K_PJND_MASK, C_PJND_CLAMP);
            a_id += co.c_pjnd_lo * pjnd_transducer(raw_abs_err, act, K_PJND_MASK_LOW, C_PJND_CLAMP);
            a_id +=
                co.c_pjnd_hi * pjnd_transducer(raw_abs_err, act, K_PJND_MASK_HIGH, C_PJND_CLAMP);

            // Weighted pools (masked / iw): −s_k·w_i·v_i/Σw.
            let sat_act = saturate(act, C_ACTIVITY);
            let mask_w = 1.0 - sat_act;
            let iw_w = sat_act + IW_WEIGHT_FLOOR;
            // Weights are reference-side; each member routes by its VALUE's
            // support class (ssim → window, art/det → residual, mse → pixel).
            a_win += mask_w * (co.c_mask_ssim * d) + iw_w * (co.c_iw_ssim * d);
            a_res += mask_w * (co.c_mask_art * art_i + co.c_mask_det * det_i)
                + iw_w * (co.c_iw_art * art_i + co.c_iw_det * det_i);
            a_id += mask_w * (co.c_mask_mse * mse_i) + iw_w * (co.c_iw_mse * mse_i);

            // Soft-peak: s_k·w(v)·(f − v)/W  (sp = (s_k/W, f)).
            a_win += co.sp_ssim.0 * saturate(d, C_PEAK) * (co.sp_ssim.1 - d);
            a_res += co.sp_art.0 * saturate(art_i, C_PEAK) * (co.sp_art.1 - art_i);
            a_res += co.sp_det.0 * saturate(det_i, C_PEAK) * (co.sp_det.1 - det_i);

            // SSIM deviation pools (v = d): the centred polynomial
            // `V2AppCoeffs::dev_q` in `e = d − μ`.
            {
                let q = &co.dev_q;
                let e = d - co.dev_mu;
                a_win += q[0] + e * (q[1] + e * (q[2] + e * (q[3] + e * q[4])));
            }

            // Append block.
            let b2 = planes.bs2[i] as f64;
            let var1 = (b2 - m1 * m1).max(0.0);
            let var2 = ((sq - b2) - m2 * m2).max(0.0);
            let n1 = (s - m1) / (var1 + C_MSCN_VAR).sqrt();
            let n2 = (dd - m2) / (var2 + C_MSCN_VAR).sqrt();
            let dn = n1 - n2;
            a_res += co.c_mscn * saturate(dn.abs(), C_MSCN_ABS);
            a_res += co.c_mscn2 * saturate(dn * dn, C_MSCN_SQ);
            let (cg, cl) = bounded_excess_pair(var2, var1, C_CONTRAST);
            a_win += co.c_cgain * cg + co.c_closs * cl;
            a_win += co.c_tex * (bounded_dissim(var1, var2, C_CONTRAST, co.stable));

            if cross_on {
                let ry = ref_y[i] as f64;
                let t = saturate(ry, C_LUM_T);
                a_id += co.c_lumt
                    * pjnd_transducer(
                        raw_abs_err,
                        act + t * (K_LUM_ADAPT / K_PJND_MASK),
                        K_PJND_MASK,
                        C_PJND_CLAMP,
                    );
                if let Some((ax, ab)) = cross {
                    let act_c = ax[i] as f64 + ab[i] as f64;
                    a_id += co.c_xmask
                        * pjnd_transducer(
                            raw_abs_err,
                            act + act_c * (K_XCH / K_PJND_MASK),
                            K_PJND_MASK,
                            C_PJND_CLAMP,
                        );
                }
            }
            // Luminance bins (t from the reference Y plane, all channels).
            {
                let ry = ref_y[i] as f64;
                let t = saturate(ry, C_LUM_T);
                let one_mt = 1.0 - t;
                let wd = one_mt * one_mt;
                let wb = t * t;
                let wm = 1.0 - wd - wb;
                a_id += (co.c_dark * wd + co.c_mid * wm + co.c_bright * wb) * mse_i;
            }
            // Append dev pools for art/det (gms handled in the gradient loop).
            a_res += gq_density(&co.gq_art, art_i) + gq_density(&co.gq_det, det_i);
            // Globals (pixel class — raw whole-plane sums).
            a_id += co.g_dmean * (s - dd);
            a_id += co.g_var * (((s - dd) - co.g_shift) * ((s + dd) - co.g_sum));

            id_plane[i - out_off] += a_id + a_res;
            win_plane[i - out_off] += a_win;
        }
    }

    // Gradient family: gms / ringing / banding / gms-dev / edge-width /
    // append2 BANDVIS.
    let need_bandvis = co.c_bv_gain != 0.0 || co.c_bv_loss != 0.0;
    let need_grad = co.c_gms != 0.0
        || co.c_ringing != 0.0
        || co.c_banding != 0.0
        || co.gd_gms.0 != 0.0
        || co.c_edgew != 0.0
        || need_bandvis;
    if need_grad {
        for y in y0..y1 {
            let row = y * width;
            let y_up = reflect_101(y as isize - 1, height);
            let y_dn = reflect_101(y as isize + 1, height);
            for x in 0..width {
                let i = row + x;
                let xl = x.saturating_sub(1);
                let xr = (x + 1).min(width - 1);
                let sxl = src[row + xl] as f64;
                let sxr = src[row + xr] as f64;
                let syu = src[y_up * width + x] as f64;
                let syd = src[y_dn * width + x] as f64;
                let dxl = dst[row + xl] as f64;
                let dxr = dst[row + xr] as f64;
                let dyu = dst[y_up * width + x] as f64;
                let dyd = dst[y_dn * width + x] as f64;
                let gx_s = sxr - sxl;
                let gy_s = syd - syu;
                let gmag_s = (gx_s * gx_s + gy_s * gy_s).sqrt();
                let gx_d = dxr - dxl;
                let gy_d = dyd - dyu;
                let gmag_d = (gx_d * gx_d + gy_d * gy_d).sqrt();
                let g = bounded_dissim(gmag_s, gmag_d, C_GMS, co.stable);
                let mut acc = co.c_gms * g;
                acc += gq_density(&co.gq_gms, g);
                if co.c_ringing != 0.0 {
                    let s = src[i] as f64;
                    let dd = dst[i] as f64;
                    let act = planes.act[i] as f64;
                    let ring_err = if ring_hp {
                        ((s - planes.mu1[i] as f64) - (dd - planes.mu2[i] as f64)).abs()
                    } else {
                        (s - dd).abs()
                    };
                    let err_b = saturate(ring_err, C_RING_ERR);
                    let act_b = saturate(act, C_ACTIVITY);
                    let edge_r = saturate(gmag_s, C_RING_EDGE);
                    acc += co.c_ringing * (err_b * act_b * (1.0 - edge_r));
                }
                if co.c_banding != 0.0 {
                    let edge_excess = bounded_excess(gmag_d, gmag_s, C_BAND_DST);
                    let src_smooth = 1.0 - saturate(gmag_s, C_BAND_SRC);
                    acc += co.c_banding * (edge_excess * src_smooth);
                }
                if need_bandvis {
                    // append2 BANDVIS (`idx_append2::BANDVIS_GAIN/LOSS`) —
                    // the OFF-toggle production form (`append2_dst_activity`
                    // is default-OFF and adjudicated OFF for every
                    // production extraction, and this path fixes toggles to
                    // `V2NewFeatureToggles::default()`). Second differences
                    // reuse the four neighbour loads the gradient family
                    // already performs, so the terms are near-free and the
                    // neighbour convention (x: clamp, y: reflect_101 via the
                    // production halo rows) matches the kernel exactly.
                    let s_c = src[i] as f64;
                    let d_c = dst[i] as f64;
                    let d2x_src = sxl + sxr - 2.0 * s_c;
                    let d2y_src = syu + syd - 2.0 * s_c;
                    let curv_src = (d2x_src * d2x_src + d2y_src * d2y_src).sqrt();
                    let d2x_dst = dxl + dxr - 2.0 * d_c;
                    let d2y_dst = dyu + dyd - 2.0 * d_c;
                    let curv_dst = (d2x_dst * d2x_dst + d2y_dst * d2y_dst).sqrt();
                    let flat = 1.0 - saturate(planes.act[i] as f64, C_ACTIVITY);
                    let band = |g: f64| -> f64 {
                        saturate(g, BV_DELTA_LO_SDR as f64)
                            * (1.0 - saturate(g, BV_DELTA_HI_SDR as f64))
                    };
                    let b_src = band(curv_src) * flat;
                    let b_dst = band(curv_dst) * flat;
                    let (gain_i, loss_i) = bounded_excess_pair(b_dst, b_src, C_BV);
                    acc += co.c_bv_gain * gain_i + co.c_bv_loss * loss_i;
                }
                // Edge-width: coefficient × (|∇d| − |∇s|) — the per-pixel
                // change of this scale's mean distorted gradient under
                // refinement is −(|∇d|−|∇s|)/N (folded into c_edgew).
                acc += co.c_edgew * (gmag_d - gmag_s);
                id_plane[i - out_off] += acc;
            }
        }
    }
}

/// Blockiness lattice terms. A step between plane pixels `p` and `q` loses all of its excess `E` when both are
/// refined, and keeps `E_p = excess(|d_q - s_p|, |s_q - s_p|)` when only `p` is refined (`E_q` likewise for `q`). Each end gets the one-sided share `g + I/2` (`g_p = E - E_p`,
/// `I = E_p + E_q - E`) and the pair term `I` is recorded in the [`crate::attribution::LatticeScale`] for steps
/// whose both ends lie inside the logical image, so rectangle queries are exact for footprint-aligned repairs.
/// Steps outside it (or when no lattice is passed, as on the sampling-geometry path) keep the symmetric 50/50
/// split. Serial — the horizontal family writes the row above.
#[cfg(feature = "custom-profiles")]
fn attr_pass_b_blockiness(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    co: &V2AppCoeffs,
    id_plane: &mut [f64],
    mut lattice: Option<(&mut crate::attribution::LatticeScale, usize, usize)>,
) {
    if co.c_blockiness == 0.0 {
        return;
    }
    let c = co.c_blockiness;
    // One lattice step between `p` (earlier) and `q`: the density shares and, when the step is fully inside
    // the logical image, the pair term the rectangle query needs (`LatticeScale`). Outside it the legacy
    // 50/50 split stays.
    let step = |ip: usize, iq: usize| -> (f64, f64, f64) {
        let (dp, dq) = (dst[ip] as f64, dst[iq] as f64);
        let (sp, sq) = (src[ip] as f64, src[iq] as f64);
        let step_src = (sq - sp).abs();
        let e = bounded_excess((dq - dp).abs(), step_src, C_BLOCK);
        let e_p = bounded_excess((dq - sp).abs(), step_src, C_BLOCK);
        let e_q = bounded_excess((sq - dp).abs(), step_src, C_BLOCK);
        (
            0.5 * c * (e - e_p + e_q),
            0.5 * c * (e - e_q + e_p),
            c * (e_p + e_q - e),
        )
    };
    for y in 0..height {
        let row = y * width;
        let mut x = BLOCK_LATTICE;
        while x < width {
            let i = row + x;
            match lattice.as_mut() {
                Some((l, lw, lh)) if x < *lw && y < *lh => {
                    let (to_p, to_q, pair) = step(i - 1, i);
                    id_plane[i - 1] += to_p;
                    id_plane[i] += to_q;
                    l.add_vertical_step(x, y, pair);
                }
                _ => {
                    let step_dst = (dst[i] as f64 - dst[i - 1] as f64).abs();
                    let step_src = (src[i] as f64 - src[i - 1] as f64).abs();
                    let v = c * bounded_excess(step_dst, step_src, C_BLOCK);
                    id_plane[i] += 0.5 * v;
                    id_plane[i - 1] += 0.5 * v;
                }
            }
            x += BLOCK_LATTICE;
        }
        if y % BLOCK_LATTICE == 0 && y > 0 {
            for x in 0..width {
                let i = row + x;
                let i_up = i - width;
                match lattice.as_mut() {
                    Some((l, lw, lh)) if x < *lw && y < *lh => {
                        let (to_p, to_q, pair) = step(i_up, i);
                        id_plane[i_up] += to_p;
                        id_plane[i] += to_q;
                        l.add_horizontal_step(y, x, pair);
                    }
                    _ => {
                        let step_dst = (dst[i] as f64 - dst[i_up] as f64).abs();
                        let step_src = (src[i] as f64 - src[i_up] as f64).abs();
                        let v = c * bounded_excess(step_dst, step_src, C_BLOCK);
                        id_plane[i] += 0.5 * v;
                        id_plane[i_up] += 0.5 * v;
                    }
                }
            }
        }
    }
}

/// f32 coefficient pack for the FUSED pass-B kernel (appendix P lever 1):
/// [`V2AppCoeffs`] with the dev-pool / gdp guards PRE-FOLDED into
/// polynomial coefficients so the pixel loop is branch-free, and the
/// cross/lum-transducer terms zeroed for non-Y channels (the f64 kernel's
/// `cross_on` branch, folded into coefficients). Serves the fused
/// folded-944 entry ONLY — the standalone f64 pass-B and its strict tests
/// are untouched (the C3a precedent).
#[cfg(feature = "custom-profiles")]
#[derive(Clone, Copy)]
struct V2AppCoeffsF32 {
    c_ssim: f32,
    c_art: f32,
    c_det: f32,
    c_mse: f32,
    c_hf_gain: f32,
    c_hf_loss: f32,
    c_hf_mag: f32,
    c_pjnd: f32,
    c_pjnd_lo: f32,
    c_pjnd_hi: f32,
    c_gms: f32,
    c_ringing: f32,
    c_banding: f32,
    c_blockiness: f32,
    c_mask_ssim: f32,
    c_mask_art: f32,
    c_mask_det: f32,
    c_mask_mse: f32,
    c_iw_ssim: f32,
    c_iw_art: f32,
    c_iw_det: f32,
    c_iw_mse: f32,
    sp_ssim: (f32, f32),
    sp_art: (f32, f32),
    sp_det: (f32, f32),
    /// ssim-`d` dev-pool polynomial ([`V2AppCoeffs::dev_q`]), centered at the
    /// pool mean `pdm`: `pq[0] + e·(pq[1] + e·(pq[2] + e·(pq[3] + e·pq[4])))`
    /// with `e = d − pdm`, added to the window class. Expanded in raw powers
    /// of `d` these coefficients grow as `1/f³` and the f32 sum cancels
    /// catastrophically once the deviation is small against the mean.
    pq: [f32; 5],
    pdm: f32,
    c_xmask: f32,
    c_lumt: f32,
    c_mscn: f32,
    c_mscn2: f32,
    c_cgain: f32,
    c_closs: f32,
    c_tex: f32,
    c_dark: f32,
    c_mid: f32,
    c_bright: f32,
    /// gdp pools as centred quadratics ([`V2AppCoeffs::gq_art`]):
    /// `q[0] + e·(q[1] + e·q[2])`, `e = v − q[3]` (guards folded).
    ga: [f32; 4],
    gd: [f32; 4],
    gg: [f32; 4],
    g_dmean: f32,
    g_var: f32,
    g_shift: f32,
    g_sum: f32,
    c_edgew: f32,
    c_bv_gain: f32,
    c_bv_loss: f32,
    /// [`V2AppCoeffs::direct`] / [`V2AppCoeffs::ring_hp`] / [`V2AppCoeffs::stable`]: the
    /// planes' own forms.
    direct: bool,
    ring_hp: bool,
    stable: bool,
}

/// Fold a derived [`V2AppCoeffs`] into the f32 pack. `cross_on` = the f64
/// kernel's `cross.is_some()` (Y channel): when false, the lum-transducer
/// and cross-mask terms are zeroed exactly as the f64 branch skips them.
#[cfg(feature = "custom-profiles")]
fn v2app_coeffs_fold_f32(co: &V2AppCoeffs, cross_on: bool) -> V2AppCoeffsF32 {
    let f32x = |q: &[f64]| -> Vec<f32> { q.iter().map(|&v| v as f32).collect() };
    let q4 = |q: &[f64; 4]| -> [f32; 4] { f32x(q).try_into().unwrap() };
    V2AppCoeffsF32 {
        c_ssim: co.c_ssim as f32,
        c_art: co.c_art as f32,
        c_det: co.c_det as f32,
        c_mse: co.c_mse as f32,
        c_hf_gain: co.c_hf_gain as f32,
        c_hf_loss: co.c_hf_loss as f32,
        c_hf_mag: co.c_hf_mag as f32,
        c_pjnd: co.c_pjnd as f32,
        c_pjnd_lo: co.c_pjnd_lo as f32,
        c_pjnd_hi: co.c_pjnd_hi as f32,
        c_gms: co.c_gms as f32,
        c_ringing: co.c_ringing as f32,
        c_banding: co.c_banding as f32,
        c_blockiness: co.c_blockiness as f32,
        c_mask_ssim: co.c_mask_ssim as f32,
        c_mask_art: co.c_mask_art as f32,
        c_mask_det: co.c_mask_det as f32,
        c_mask_mse: co.c_mask_mse as f32,
        c_iw_ssim: co.c_iw_ssim as f32,
        c_iw_art: co.c_iw_art as f32,
        c_iw_det: co.c_iw_det as f32,
        c_iw_mse: co.c_iw_mse as f32,
        sp_ssim: (co.sp_ssim.0 as f32, co.sp_ssim.1 as f32),
        sp_art: (co.sp_art.0 as f32, co.sp_art.1 as f32),
        sp_det: (co.sp_det.0 as f32, co.sp_det.1 as f32),
        pq: f32x(&co.dev_q).try_into().unwrap(),
        pdm: co.dev_mu as f32,
        c_xmask: if cross_on { co.c_xmask as f32 } else { 0.0 },
        c_lumt: if cross_on { co.c_lumt as f32 } else { 0.0 },
        c_mscn: co.c_mscn as f32,
        c_mscn2: co.c_mscn2 as f32,
        c_cgain: co.c_cgain as f32,
        c_closs: co.c_closs as f32,
        c_tex: co.c_tex as f32,
        c_dark: co.c_dark as f32,
        c_mid: co.c_mid as f32,
        c_bright: co.c_bright as f32,
        ga: q4(&co.gq_art),
        gd: q4(&co.gq_det),
        gg: q4(&co.gq_gms),
        g_dmean: co.g_dmean as f32,
        g_var: co.g_var as f32,
        g_shift: co.g_shift as f32,
        g_sum: co.g_sum as f32,
        c_edgew: co.c_edgew as f32,
        c_bv_gain: co.c_bv_gain as f32,
        c_bv_loss: co.c_bv_loss as f32,
        direct: co.direct,
        ring_hp: co.ring_hp,
        stable: co.stable,
    }
}

/// Scalar per-pixel MAIN-sweep integrand of the f32 fused pass-B — shared
/// by the SIMD kernel's row tail and the sub-8-wide fallback, so there is
/// exactly ONE formula source. Returns `(id_add, win_add)` (`id` carries
/// the pixel + residual classes, `win` the window class).
/// Scalar-tail twin of [`ssim_d_local_v`]'s two forms. At revision 3 the `s12` plane carries the direct error
/// moment `E[(a-b)²]`, not `E[ab]`, so the covariance form is wrong there (it read the error moment as a product
/// moment and returned `d` far outside `[0, 2]`, which the dev-pool coefficients then amplified to 1e26 on tail
/// columns; `benchmarks/v2spatial_2026-10-02.md`).
#[cfg(feature = "custom-profiles")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn ssim_d_px(
    na: f32,
    nb: f32,
    nd: f32,
    m1: f32,
    m2: f32,
    s12v: f32,
    direct: bool,
    stable: bool,
) -> f32 {
    if direct {
        let mu_diff = m1 - m2;
        let err_var = (s12v - mu_diff * mu_diff).max(0.0);
        let lum = if stable { mu_diff * mu_diff } else { nb - na };
        ((nd * lum + na * err_var) / (nb * nd)).max(0.0)
    } else {
        let nc = 2.0 * (s12v - m1 * m2) + C2_V2 as f32;
        (1.0 - (na * nc) / (nb * nd)).max(0.0)
    }
}

#[cfg(feature = "custom-profiles")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn attr_pass_b_main_px(
    s: f32,
    dd: f32,
    m1: f32,
    m2: f32,
    sq: f32,
    s12v: f32,
    act: f32,
    b2v: f32,
    axv: f32,
    abv: f32,
    ry: f32,
    co: &V2AppCoeffsF32,
    direct: bool,
    append: bool,
) -> (f32, f32) {
    const C1: f32 = C1_V2 as f32;
    const C2: f32 = C2_V2 as f32;
    const CE: f32 = C_EDGE as f32;
    const CM: f32 = C_MSE as f32;
    const CH: f32 = C_HF as f32;
    const CPC: f32 = C_PJND_CLAMP as f32;
    const KP: f32 = K_PJND_MASK as f32;
    const KPL: f32 = K_PJND_MASK_LOW as f32;
    const KPH: f32 = K_PJND_MASK_HIGH as f32;
    const CA: f32 = C_ACTIVITY as f32;
    const IWF: f32 = IW_WEIGHT_FLOOR as f32;
    const CPK: f32 = C_PEAK as f32;
    const CMV: f32 = C_MSCN_VAR as f32;
    const CMA: f32 = C_MSCN_ABS as f32;
    const CMS: f32 = C_MSCN_SQ as f32;
    const CC: f32 = C_CONTRAST as f32;
    const CLT: f32 = C_LUM_T as f32;
    const KLA_OVER_KP: f32 = (K_LUM_ADAPT / K_PJND_MASK) as f32;
    const KX_OVER_KP: f32 = (K_XCH / K_PJND_MASK) as f32;
    let sat = |x: f32, c: f32| -> f32 {
        let x = x.max(0.0);
        x / (x + c)
    };
    let pjnd = |x: f32, act: f32, k: f32| -> f32 { x / (x + CPC * (1.0 + k * act)) };
    let mut a_id = 0.0f32;
    let mut a_win = 0.0f32;
    let mut a_res = 0.0f32;
    // Dense family.
    let na = 2.0 * m1 * m2 + C1;
    let nb = m1 * m1 + m2 * m2 + C1;
    let nd = sq - m1 * m1 - m2 * m2 + C2;
    let d = ssim_d_px(na, nb, nd, m1, m2, s12v, direct, co.stable);
    let diff_src = (s - m1).abs();
    let diff_dst = (dd - m2).abs();
    let edge_dissim = bounded_dissim32(diff_src, diff_dst, CE, co.stable);
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
    let raw_abs_err = raw_diff.abs();
    let mse_i = sat(raw_diff * raw_diff, CM);
    a_win += co.c_ssim * d;
    a_res += co.c_art * art_i + co.c_det * det_i;
    a_id += co.c_mse * mse_i;
    let hf_src = s - m1;
    let hf_dst = dd - m2;
    let (hf_gain_i, hf_loss_i) = {
        let a2 = hf_dst * hf_dst;
        let b2 = hf_src * hf_src;
        let r = 1.0 / (a2 + b2 + CH);
        ((a2 - b2).max(0.0) * r, (b2 - a2).max(0.0) * r)
    };
    a_res += co.c_hf_gain * hf_gain_i + co.c_hf_loss * hf_loss_i;
    a_res += co.c_hf_mag * {
        let a = hf_src.abs();
        let b = hf_dst.abs();
        (a - b).max(0.0) / (a + b + CH)
    };
    a_id += co.c_pjnd * pjnd(raw_abs_err, act, KP);
    a_id += co.c_pjnd_lo * pjnd(raw_abs_err, act, KPL);
    a_id += co.c_pjnd_hi * pjnd(raw_abs_err, act, KPH);
    // Weighted pools.
    let sat_act = sat(act, CA);
    let mask_w = 1.0 - sat_act;
    let iw_w = sat_act + IWF;
    a_win += mask_w * (co.c_mask_ssim * d) + iw_w * (co.c_iw_ssim * d);
    a_res += mask_w * (co.c_mask_art * art_i + co.c_mask_det * det_i)
        + iw_w * (co.c_iw_art * art_i + co.c_iw_det * det_i);
    a_id += mask_w * (co.c_mask_mse * mse_i) + iw_w * (co.c_iw_mse * mse_i);
    // Soft-peak.
    a_win += co.sp_ssim.0 * sat(d, CPK) * (co.sp_ssim.1 - d);
    a_res += co.sp_art.0 * sat(art_i, CPK) * (co.sp_art.1 - art_i);
    a_res += co.sp_det.0 * sat(det_i, CPK) * (co.sp_det.1 - det_i);
    // Dev polynomial, centered: `V2AppCoeffsF32::pq` in e = d − pdm.
    let e = d - co.pdm;
    let q = &co.pq;
    a_win += q[0] + e * (q[1] + e * (q[2] + e * (q[3] + e * q[4])));
    // Rev5 retains no appendix plane: its coefficients cannot read this block.
    if append {
        // Append block.
        let var1 = (b2v - m1 * m1).max(0.0);
        let var2 = ((sq - b2v) - m2 * m2).max(0.0);
        let n1 = (s - m1) / (var1 + CMV).sqrt();
        let n2 = (dd - m2) / (var2 + CMV).sqrt();
        let dn = n1 - n2;
        a_res += co.c_mscn * sat(dn.abs(), CMA);
        a_res += co.c_mscn2 * sat(dn * dn, CMS);
        let (cg, cl) = {
            let r = 1.0 / (var2 + var1 + CC);
            ((var2 - var1).max(0.0) * r, (var1 - var2).max(0.0) * r)
        };
        a_win += co.c_cgain * cg + co.c_closs * cl;
        a_win += co.c_tex * (bounded_dissim32(var1, var2, CC, co.stable));
        // Luminance transducer + cross-mask (coefficients zero off-Y).
        let t = sat(ry, CLT);
        if co.c_lumt != 0.0 {
            a_id += co.c_lumt * pjnd(raw_abs_err, act + t * KLA_OVER_KP, KP);
        }
        if co.c_xmask != 0.0 {
            let act_c = axv + abv;
            a_id += co.c_xmask * pjnd(raw_abs_err, act + act_c * KX_OVER_KP, KP);
        }
        // Luminance bins.
        let one_mt = 1.0 - t;
        let wd = one_mt * one_mt;
        let wb = t * t;
        let wm = 1.0 - wd - wb;
        a_id += (co.c_dark * wd + co.c_mid * wm + co.c_bright * wb) * mse_i;
        // gdp pools (art/det).
        let gq = |q: &[f32; 4], v: f32| -> f32 {
            let e = v - q[3];
            q[0] + e * (q[1] + e * q[2])
        };
        a_res += gq(&co.ga, art_i) + gq(&co.gd, det_i);
        // Globals (factored g_var form).
        a_id += co.g_dmean * raw_diff;
        a_id += co.g_var * ((raw_diff - co.g_shift) * ((s + dd) - co.g_sum));
    }
    (a_id + a_res, a_win)
}

/// SIMD main sweep of the f32 fused pass-B — generic over any
/// [`F32x8Backend`] token, 8 pixels per step with the scalar
/// [`attr_pass_b_main_px`] tail. Same formulas lane-for-lane (selects via
/// [`GenericF32x8::blend`]); the auto-vectorizer refused the 55-constant
/// monolith (measured scalar `divss` codegen — appendix P lever 1), so
/// this is the explicit magetypes port, the same §A.14 pattern as
/// `dense_block_kernel_generic`.
#[cfg(feature = "custom-profiles")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn attr_pass_b_main_kernel_generic<T: F32x8Backend + Copy>(
    token: T,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    act_p: &[f32],
    bs2: &[f32],
    ax: &[f32],
    ab: &[f32],
    ref_y: &[f32],
    co: &V2AppCoeffsF32,
    width: usize,
    y0: usize,
    y1: usize,
    id_plane: &mut [f32],
    win_plane: &mut [f32],
) {
    // Revision 3 (issue #61): the `s12` plane is the direct error moment, so
    // the v2 SSIM signal must read it as such. The planes' own revision
    // (`V2AppCoeffsF32::direct`); loop-invariant, so every call site below
    // unswitches on it.
    let direct = co.direct;
    let zero = V8::<T>::zero_t(token);
    let one = V8::<T>::splat_t(token, 1.0);
    let sp = |v: f32| V8::<T>::splat_t(token, v);
    let c1 = sp(C1_V2 as f32);
    let c2 = sp(C2_V2 as f32);
    let c_edge = sp(C_EDGE as f32);
    let c_mse = sp(C_MSE as f32);
    let c_hf = sp(C_HF as f32);
    let c_pjnd_clamp = sp(C_PJND_CLAMP as f32);
    let k_mid = sp(K_PJND_MASK as f32);
    let k_lo = sp(K_PJND_MASK_LOW as f32);
    let k_hi = sp(K_PJND_MASK_HIGH as f32);
    let c_activity = sp(C_ACTIVITY as f32);
    let iw_floor = sp(IW_WEIGHT_FLOOR as f32);
    let c_peak = sp(C_PEAK as f32);
    let c_mscn_var = sp(C_MSCN_VAR as f32);
    let c_mscn_abs = sp(C_MSCN_ABS as f32);
    let c_mscn_sq = sp(C_MSCN_SQ as f32);
    let c_contrast = sp(C_CONTRAST as f32);
    let c_lum_t = sp(C_LUM_T as f32);
    let kla = sp((K_LUM_ADAPT / K_PJND_MASK) as f32);
    let kx = sp((K_XCH / K_PJND_MASK) as f32);
    let sat_v = |x: V8<T>, c: V8<T>| -> V8<T> {
        let x = x.max(zero);
        x / (x + c)
    };
    let append = !bs2.is_empty();
    let has_lumt = co.c_lumt != 0.0;
    let has_xmask = co.c_xmask != 0.0;
    let out_off = y0 * width;
    let width8 = width - (width % 8);
    for y in y0..y1 {
        let row = y * width;
        let orow = row - out_off;
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
            let sq = ld!(ssq);
            let act = ld!(act_p);
            let mut a_id = zero;
            let mut a_win = zero;
            let mut a_res = zero;
            // Dense family.
            let d = ssim_d_local_v(token, m1, m2, ld!(s12), sq, c1, c2, direct, co.stable);
            let diff_src = (s - m1).abs();
            let diff_dst = (dd - m2).abs();
            let edge_dissim = bounded_dissim_v(token, diff_src, diff_dst, c_edge, co.stable);
            let art_i = V8::<T>::blend(diff_dst.simd_gt(diff_src), edge_dissim, zero);
            let det_i = V8::<T>::blend(diff_dst.simd_lt(diff_src), edge_dissim, zero);
            let raw_diff = s - dd;
            let raw_abs_err = raw_diff.abs();
            let mse_i = sat_v(raw_diff * raw_diff, c_mse);
            a_win += sp(co.c_ssim) * d;
            a_res += sp(co.c_art) * art_i + sp(co.c_det) * det_i;
            a_id += sp(co.c_mse) * mse_i;
            let hf_src = s - m1;
            let hf_dst = dd - m2;
            let (hf_gain_i, hf_loss_i) =
                bounded_excess_pair_v(token, hf_dst * hf_dst, hf_src * hf_src, c_hf);
            a_res += sp(co.c_hf_gain) * hf_gain_i + sp(co.c_hf_loss) * hf_loss_i;
            a_res += sp(co.c_hf_mag) * bounded_excess_v(token, hf_src.abs(), hf_dst.abs(), c_hf);
            a_id += sp(co.c_pjnd) * pjnd_transducer_v(token, raw_abs_err, act, k_mid, c_pjnd_clamp);
            a_id +=
                sp(co.c_pjnd_lo) * pjnd_transducer_v(token, raw_abs_err, act, k_lo, c_pjnd_clamp);
            a_id +=
                sp(co.c_pjnd_hi) * pjnd_transducer_v(token, raw_abs_err, act, k_hi, c_pjnd_clamp);
            // Weighted pools.
            let sat_act = sat_v(act, c_activity);
            let mask_w = one - sat_act;
            let iw_w = sat_act + iw_floor;
            a_win += mask_w * (sp(co.c_mask_ssim) * d) + iw_w * (sp(co.c_iw_ssim) * d);
            a_res += mask_w * (sp(co.c_mask_art) * art_i + sp(co.c_mask_det) * det_i)
                + iw_w * (sp(co.c_iw_art) * art_i + sp(co.c_iw_det) * det_i);
            a_id += mask_w * (sp(co.c_mask_mse) * mse_i) + iw_w * (sp(co.c_iw_mse) * mse_i);
            // Soft-peak.
            a_win += sp(co.sp_ssim.0) * sat_v(d, c_peak) * (sp(co.sp_ssim.1) - d);
            a_res += sp(co.sp_art.0) * sat_v(art_i, c_peak) * (sp(co.sp_art.1) - art_i);
            a_res += sp(co.sp_det.0) * sat_v(det_i, c_peak) * (sp(co.sp_det.1) - det_i);
            // Dev polynomial, centered (see `V2AppCoeffsF32::pq`).
            let e = d - sp(co.pdm);
            let q = &co.pq;
            a_win += sp(q[0]) + e * (sp(q[1]) + e * (sp(q[2]) + e * (sp(q[3]) + e * sp(q[4]))));
            if append {
                // Append block.
                let b2v = ld!(bs2);
                let var1 = (b2v - m1 * m1).max(zero);
                let var2 = ((sq - b2v) - m2 * m2).max(zero);
                let n1 = (s - m1) / (var1 + c_mscn_var).sqrt();
                let n2 = (dd - m2) / (var2 + c_mscn_var).sqrt();
                let dn = n1 - n2;
                a_res += sp(co.c_mscn) * sat_v(dn.abs(), c_mscn_abs);
                a_res += sp(co.c_mscn2) * sat_v(dn * dn, c_mscn_sq);
                let (cg, cl) = bounded_excess_pair_v(token, var2, var1, c_contrast);
                a_win += sp(co.c_cgain) * cg + sp(co.c_closs) * cl;
                a_win +=
                    sp(co.c_tex) * (bounded_dissim_v(token, var1, var2, c_contrast, co.stable));
                // Luminance transducer + cross-mask (Y only; uniform branches).
                let ry = ld!(ref_y);
                let t = sat_v(ry, c_lum_t);
                if has_lumt {
                    a_id += sp(co.c_lumt)
                        * pjnd_transducer_v(token, raw_abs_err, act + t * kla, k_mid, c_pjnd_clamp);
                }
                if has_xmask {
                    let act_c = ld!(ax) + ld!(ab);
                    a_id += sp(co.c_xmask)
                        * pjnd_transducer_v(
                            token,
                            raw_abs_err,
                            act + act_c * kx,
                            k_mid,
                            c_pjnd_clamp,
                        );
                }
                // Luminance bins.
                let one_mt = one - t;
                let wd = one_mt * one_mt;
                let wb = t * t;
                let wm = one - wd - wb;
                a_id += (sp(co.c_dark) * wd + sp(co.c_mid) * wm + sp(co.c_bright) * wb) * mse_i;
                // gdp pools.
                let ea = art_i - sp(co.ga[3]);
                a_res += sp(co.ga[0]) + ea * (sp(co.ga[1]) + ea * sp(co.ga[2]));
                let ed = det_i - sp(co.gd[3]);
                a_res += sp(co.gd[0]) + ed * (sp(co.gd[1]) + ed * sp(co.gd[2]));
                // Globals (factored g_var form).
                a_id += sp(co.g_dmean) * raw_diff;
                a_id += sp(co.g_var) * ((raw_diff - sp(co.g_shift)) * ((s + dd) - sp(co.g_sum)));
            }
            let o = orow + x;
            let id_new =
                V8::<T>::from_array_t(token, id_plane[o..o + 8].try_into().unwrap()) + a_id + a_res;
            id_plane[o..o + 8].copy_from_slice(&id_new.to_array());
            let win_new =
                V8::<T>::from_array_t(token, win_plane[o..o + 8].try_into().unwrap()) + a_win;
            win_plane[o..o + 8].copy_from_slice(&win_new.to_array());
            x += 8;
        }
        // Scalar tail — the shared per-pixel formula.
        for x in width8..width {
            let i = row + x;
            let (id_add, win_add) = attr_pass_b_main_px(
                src[i],
                dst[i],
                mu1[i],
                mu2[i],
                ssq[i],
                s12[i],
                act_p[i],
                if append { bs2[i] } else { 0.0 },
                ax[i],
                ab[i],
                ref_y[i],
                co,
                direct,
                append,
            );
            id_plane[orow + x] += id_add;
            win_plane[orow + x] += win_add;
        }
    }
}

/// Tiered magetypes entry for the pass-B main sweep.
#[cfg(feature = "custom-profiles")]
#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
fn attr_pass_b_main_entry(
    token: Token,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    act_p: &[f32],
    bs2: &[f32],
    ax: &[f32],
    ab: &[f32],
    ref_y: &[f32],
    co: &V2AppCoeffsF32,
    width: usize,
    y0: usize,
    y1: usize,
    id_plane: &mut [f32],
    win_plane: &mut [f32],
) {
    attr_pass_b_main_kernel_generic(
        token, src, dst, mu1, mu2, ssq, s12, act_p, bs2, ax, ab, ref_y, co, width, y0, y1,
        id_plane, win_plane,
    );
}

/// f32 twin of [`attr_pass_b_rows`] (appendix P lever 1): the SIMD main
/// sweep ([`attr_pass_b_main_entry`], magetypes-tiered) plus the gradient
/// family sweep ([`attr_pass_b_grad_rows_f32`], `#[autoversion]` — its
/// smaller body auto-vectorizes). Same integrand formulas as the f64
/// kernel; the dev/gdp pools are pre-folded polynomials and the `g_var`
/// global uses the factored `(s−d)·((s+d) − 2·gmean_d)` form
/// (algebraically identical, f32-stable). Cross activity slices are
/// dummies (== `ref_y`) for non-Y channels — their coefficients are
/// zeroed at fold time. Precision class: the fused entry's C3a tolerance
/// (G-P2), NOT the standalone's strict identities.
#[cfg(feature = "custom-profiles")]
#[allow(clippy::too_many_arguments)]
fn attr_pass_b_rows_f32(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    act_p: &[f32],
    bs2: &[f32],
    ax: &[f32],
    ab: &[f32],
    ref_y: &[f32],
    co: V2AppCoeffsF32,
    y0: usize,
    y1: usize,
    id_plane: &mut [f32],
    win_plane: &mut [f32],
) {
    incant!(
        attr_pass_b_main_entry(
            src, dst, mu1, mu2, ssq, s12, act_p, bs2, ax, ab, ref_y, &co, width, y0, y1, id_plane,
            win_plane
        ),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
    attr_pass_b_grad_rows_f32(
        src, dst, width, height, act_p, mu1, mu2, co, y0, y1, id_plane,
    );
}

/// Scalar per-pixel gradient-family integrand of the f32 fused pass-B —
/// shared by the SIMD kernel's edge/tail pixels (ONE formula source).
#[cfg(feature = "custom-profiles")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn attr_pass_b_grad_px(
    s_c: f32,
    d_c: f32,
    act_i: f32,
    m1_c: f32,
    m2_c: f32,
    sxl: f32,
    sxr: f32,
    syu: f32,
    syd: f32,
    dxl: f32,
    dxr: f32,
    dyu: f32,
    dyd: f32,
    co: &V2AppCoeffsF32,
    need_bandvis: bool,
) -> f32 {
    const CA: f32 = C_ACTIVITY as f32;
    const CG: f32 = C_GMS as f32;
    const CRE: f32 = C_RING_ERR as f32;
    const CRG: f32 = C_RING_EDGE as f32;
    const CBD: f32 = C_BAND_DST as f32;
    const CBS: f32 = C_BAND_SRC as f32;
    const CBV: f32 = C_BV as f32;
    let sat = |x: f32, c: f32| -> f32 {
        let x = x.max(0.0);
        x / (x + c)
    };
    let gx_s = sxr - sxl;
    let gy_s = syd - syu;
    let gmag_s = (gx_s * gx_s + gy_s * gy_s).sqrt();
    let gx_d = dxr - dxl;
    let gy_d = dyd - dyu;
    let gmag_d = (gx_d * gx_d + gy_d * gy_d).sqrt();
    let g = bounded_dissim32(gmag_s, gmag_d, CG, co.stable);
    let mut acc = co.c_gms * g;
    let eg = g - co.gg[3];
    acc += co.gg[0] + eg * (co.gg[1] + eg * co.gg[2]);
    if co.c_ringing != 0.0 {
        let ring_err = if co.ring_hp {
            ((s_c - m1_c) - (d_c - m2_c)).abs()
        } else {
            (s_c - d_c).abs()
        };
        let err_b = sat(ring_err, CRE);
        let act_b = sat(act_i, CA);
        let edge_r = sat(gmag_s, CRG);
        acc += co.c_ringing * (err_b * act_b * (1.0 - edge_r));
    }
    if co.c_banding != 0.0 {
        let edge_excess = (gmag_d - gmag_s).max(0.0) / (gmag_d + gmag_s + CBD);
        let src_smooth = 1.0 - sat(gmag_s, CBS);
        acc += co.c_banding * (edge_excess * src_smooth);
    }
    if need_bandvis {
        let d2x_src = sxl + sxr - 2.0 * s_c;
        let d2y_src = syu + syd - 2.0 * s_c;
        let curv_src = (d2x_src * d2x_src + d2y_src * d2y_src).sqrt();
        let d2x_dst = dxl + dxr - 2.0 * d_c;
        let d2y_dst = dyu + dyd - 2.0 * d_c;
        let curv_dst = (d2x_dst * d2x_dst + d2y_dst * d2y_dst).sqrt();
        let flat = 1.0 - sat(act_i, CA);
        let band = |g: f32| -> f32 { sat(g, BV_DELTA_LO_SDR) * (1.0 - sat(g, BV_DELTA_HI_SDR)) };
        let b_src = band(curv_src) * flat;
        let b_dst = band(curv_dst) * flat;
        let (gain_i, loss_i) = {
            let r = 1.0 / (b_dst + b_src + CBV);
            ((b_dst - b_src).max(0.0) * r, (b_src - b_dst).max(0.0) * r)
        };
        acc += co.c_bv_gain * gain_i + co.c_bv_loss * loss_i;
    }
    acc + co.c_edgew * (gmag_d - gmag_s)
}

/// SIMD gradient-family sweep — generic over any [`F32x8Backend`] token;
/// interior pixels 8-wide with unaligned ±1 neighbor loads, x-edges and
/// the sub-8 tail through the scalar [`attr_pass_b_grad_px`].
#[cfg(feature = "custom-profiles")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn attr_pass_b_grad_kernel_generic<T: F32x8Backend + Copy>(
    token: T,
    src: &[f32],
    dst: &[f32],
    act_p: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    co: &V2AppCoeffsF32,
    width: usize,
    height: usize,
    y0: usize,
    y1: usize,
    id_plane: &mut [f32],
) {
    let need_bandvis = co.c_bv_gain != 0.0 || co.c_bv_loss != 0.0;
    let has_ringing = co.c_ringing != 0.0;
    let ring_hp = co.ring_hp;
    let has_banding = co.c_banding != 0.0;
    let zero = V8::<T>::zero_t(token);
    let one = V8::<T>::splat_t(token, 1.0);
    let two = V8::<T>::splat_t(token, 2.0);
    let sp = |v: f32| V8::<T>::splat_t(token, v);
    let c_gms = sp(C_GMS as f32);
    let c_activity = sp(C_ACTIVITY as f32);
    let c_ring_err = sp(C_RING_ERR as f32);
    let c_ring_edge = sp(C_RING_EDGE as f32);
    let c_band_dst = sp(C_BAND_DST as f32);
    let c_band_src = sp(C_BAND_SRC as f32);
    let c_bv = sp(C_BV as f32);
    let bv_lo = sp(BV_DELTA_LO_SDR);
    let bv_hi = sp(BV_DELTA_HI_SDR);
    let sat_v = |x: V8<T>, c: V8<T>| -> V8<T> {
        let x = x.max(zero);
        x / (x + c)
    };
    let out_off = y0 * width;
    for y in y0..y1 {
        let row = y * width;
        let ru = reflect_101(y as isize - 1, height) * width;
        let rd = reflect_101(y as isize + 1, height) * width;
        if width == 1 {
            let i = row;
            id_plane[i - out_off] += attr_pass_b_grad_px(
                src[i],
                dst[i],
                act_p[i],
                mu1[i],
                mu2[i],
                src[i],
                src[i],
                src[ru],
                src[rd],
                dst[i],
                dst[i],
                dst[ru],
                dst[rd],
                co,
                need_bandvis,
            );
            continue;
        }
        // x = 0 (xl clamps to 0).
        {
            let i = row;
            id_plane[i - out_off] += attr_pass_b_grad_px(
                src[i],
                dst[i],
                act_p[i],
                mu1[i],
                mu2[i],
                src[i],
                src[i + 1],
                src[ru],
                src[rd],
                dst[i],
                dst[i + 1],
                dst[ru],
                dst[rd],
                co,
                need_bandvis,
            );
        }
        // Interior [1, width−1): 8-wide chunks + scalar remainder.
        let interior_end = width - 1;
        let mut x = 1usize;
        while x + 8 <= interior_end {
            let i = row + x;
            macro_rules! ldo {
                ($plane:expr, $off:expr) => {
                    V8::<T>::from_array_t(token, $plane[$off..$off + 8].try_into().unwrap())
                };
            }
            let s_c = ldo!(src, i);
            let d_c = ldo!(dst, i);
            let sxl = ldo!(src, i - 1);
            let sxr = ldo!(src, i + 1);
            let syu = ldo!(src, ru + x);
            let syd = ldo!(src, rd + x);
            let dxl = ldo!(dst, i - 1);
            let dxr = ldo!(dst, i + 1);
            let dyu = ldo!(dst, ru + x);
            let dyd = ldo!(dst, rd + x);
            let gx_s = sxr - sxl;
            let gy_s = syd - syu;
            let gmag_s = (gx_s * gx_s + gy_s * gy_s).sqrt();
            let gx_d = dxr - dxl;
            let gy_d = dyd - dyu;
            let gmag_d = (gx_d * gx_d + gy_d * gy_d).sqrt();
            let g = bounded_dissim_v(token, gmag_s, gmag_d, c_gms, co.stable);
            let mut acc = sp(co.c_gms) * g;
            let eg = g - sp(co.gg[3]);
            acc += sp(co.gg[0]) + eg * (sp(co.gg[1]) + eg * sp(co.gg[2]));
            if has_ringing {
                let ring_err = if ring_hp {
                    ((s_c - ldo!(mu1, i)) - (d_c - ldo!(mu2, i))).abs()
                } else {
                    (s_c - d_c).abs()
                };
                let err_b = sat_v(ring_err, c_ring_err);
                let act_b = sat_v(ldo!(act_p, i), c_activity);
                let edge_r = sat_v(gmag_s, c_ring_edge);
                acc += sp(co.c_ringing) * (err_b * act_b * (one - edge_r));
            }
            if has_banding {
                let edge_excess = bounded_excess_v(token, gmag_d, gmag_s, c_band_dst);
                let src_smooth = one - sat_v(gmag_s, c_band_src);
                acc += sp(co.c_banding) * (edge_excess * src_smooth);
            }
            if need_bandvis {
                let d2x_src = sxl + sxr - two * s_c;
                let d2y_src = syu + syd - two * s_c;
                let curv_src = (d2x_src * d2x_src + d2y_src * d2y_src).sqrt();
                let d2x_dst = dxl + dxr - two * d_c;
                let d2y_dst = dyu + dyd - two * d_c;
                let curv_dst = (d2x_dst * d2x_dst + d2y_dst * d2y_dst).sqrt();
                let flat = one - sat_v(ldo!(act_p, i), c_activity);
                let b_src = sat_v(curv_src, bv_lo) * (one - sat_v(curv_src, bv_hi)) * flat;
                let b_dst = sat_v(curv_dst, bv_lo) * (one - sat_v(curv_dst, bv_hi)) * flat;
                let (gain_i, loss_i) = bounded_excess_pair_v(token, b_dst, b_src, c_bv);
                acc += sp(co.c_bv_gain) * gain_i + sp(co.c_bv_loss) * loss_i;
            }
            acc += sp(co.c_edgew) * (gmag_d - gmag_s);
            let o = i - out_off;
            let id_new = V8::<T>::from_array_t(token, id_plane[o..o + 8].try_into().unwrap()) + acc;
            id_plane[o..o + 8].copy_from_slice(&id_new.to_array());
            x += 8;
        }
        while x < interior_end {
            let i = row + x;
            id_plane[i - out_off] += attr_pass_b_grad_px(
                src[i],
                dst[i],
                act_p[i],
                mu1[i],
                mu2[i],
                src[i - 1],
                src[i + 1],
                src[ru + x],
                src[rd + x],
                dst[i - 1],
                dst[i + 1],
                dst[ru + x],
                dst[rd + x],
                co,
                need_bandvis,
            );
            x += 1;
        }
        // x = width−1 (xr clamps to width−1).
        {
            let x = width - 1;
            let i = row + x;
            id_plane[i - out_off] += attr_pass_b_grad_px(
                src[i],
                dst[i],
                act_p[i],
                mu1[i],
                mu2[i],
                src[i - 1],
                src[i],
                src[ru + x],
                src[rd + x],
                dst[i - 1],
                dst[i],
                dst[ru + x],
                dst[rd + x],
                co,
                need_bandvis,
            );
        }
    }
}

/// Tiered magetypes entry for the pass-B gradient sweep.
#[cfg(feature = "custom-profiles")]
#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
fn attr_pass_b_grad_entry(
    token: Token,
    src: &[f32],
    dst: &[f32],
    act_p: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    co: &V2AppCoeffsF32,
    width: usize,
    height: usize,
    y0: usize,
    y1: usize,
    id_plane: &mut [f32],
) {
    attr_pass_b_grad_kernel_generic(
        token, src, dst, act_p, mu1, mu2, co, width, height, y0, y1, id_plane,
    );
}

/// Gradient-family sweep of the f32 fused pass-B (gms / gms-dev / ringing
/// / banding / BANDVIS / edge-width) — magetypes-tiered dispatch; skipped
/// entirely when every gradient-family coefficient is zero.
#[cfg(feature = "custom-profiles")]
#[allow(clippy::too_many_arguments)]
fn attr_pass_b_grad_rows_f32(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    act_p: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    co: V2AppCoeffsF32,
    y0: usize,
    y1: usize,
    id_plane: &mut [f32],
) {
    let need_bandvis = co.c_bv_gain != 0.0 || co.c_bv_loss != 0.0;
    let need_grad = co.c_gms != 0.0
        || co.c_ringing != 0.0
        || co.c_banding != 0.0
        || co.gg[..3].iter().any(|&v| v != 0.0)
        || co.c_edgew != 0.0
        || need_bandvis;
    if !need_grad {
        return;
    }
    incant!(
        attr_pass_b_grad_entry(
            src, dst, act_p, mu1, mu2, &co, width, height, y0, y1, id_plane
        ),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}

/// f32 twin of [`attr_pass_b_blockiness`] (serial — the horizontal family
/// writes the row above).
#[cfg(feature = "custom-profiles")]
fn attr_pass_b_blockiness_f32(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    c_blockiness: f32,
    id_plane: &mut [f32],
    mut lattice: Option<(&mut crate::attribution::LatticeScale, usize, usize)>,
) {
    const CB: f32 = C_BLOCK as f32;
    if c_blockiness == 0.0 {
        return;
    }
    let c = c_blockiness;
    let excess = |a: f32, b: f32| (a - b).max(0.0) / (a + b + CB);
    // f32 twin of the f64 step above; the pair term is accumulated in f64.
    let step = |ip: usize, iq: usize| -> (f32, f32, f64) {
        let (dp, dq) = (dst[ip], dst[iq]);
        let (sp, sq) = (src[ip], src[iq]);
        let step_src = (sq - sp).abs();
        let e = excess((dq - dp).abs(), step_src);
        let e_p = excess((dq - sp).abs(), step_src);
        let e_q = excess((sq - dp).abs(), step_src);
        (
            0.5 * c * (e - e_p + e_q),
            0.5 * c * (e - e_q + e_p),
            f64::from(c) * f64::from(e_p + e_q - e),
        )
    };
    for y in 0..height {
        let row = y * width;
        let mut x = BLOCK_LATTICE;
        while x < width {
            let i = row + x;
            match lattice.as_mut() {
                Some((l, lw, lh)) if x < *lw && y < *lh => {
                    let (to_p, to_q, pair) = step(i - 1, i);
                    id_plane[i - 1] += to_p;
                    id_plane[i] += to_q;
                    l.add_vertical_step(x, y, pair);
                }
                _ => {
                    let step_dst = (dst[i] - dst[i - 1]).abs();
                    let step_src = (src[i] - src[i - 1]).abs();
                    let v = c * excess(step_dst, step_src);
                    id_plane[i] += 0.5 * v;
                    id_plane[i - 1] += 0.5 * v;
                }
            }
            x += BLOCK_LATTICE;
        }
        if y % BLOCK_LATTICE == 0 && y > 0 {
            for x in 0..width {
                let i = row + x;
                let i_up = i - width;
                match lattice.as_mut() {
                    Some((l, lw, lh)) if x < *lw && y < *lh => {
                        let (to_p, to_q, pair) = step(i_up, i);
                        id_plane[i_up] += to_p;
                        id_plane[i] += to_q;
                        l.add_horizontal_step(y, x, pair);
                    }
                    _ => {
                        let step_dst = (dst[i] - dst[i_up]).abs();
                        let step_src = (src[i] - src[i_up]).abs();
                        let v = c * excess(step_dst, step_src);
                        id_plane[i] += 0.5 * v;
                        id_plane[i_up] += 0.5 * v;
                    }
                }
            }
        }
    }
}

/// f32 twin of [`attr_pass_b_channel`]: row-banded parallel main+gradient
/// sweeps into f32 id/win planes; blockiness serial.
#[cfg(feature = "custom-profiles")]
#[allow(clippy::too_many_arguments)]
fn attr_pass_b_channel_f32(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    planes: &AttrChPlanes,
    cross: Option<(&[f32], &[f32])>,
    ref_y: &[f32],
    co: V2AppCoeffsF32,
    parallel: bool,
    id_plane: &mut [f32],
    win_plane: &mut [f32],
    lattice: Option<(&mut crate::attribution::LatticeScale, usize, usize)>,
) {
    // Band size for the rayon arm below; `threads`-gated with it.
    #[cfg(feature = "threads")]
    const BAND: usize = 64;
    let (ax, ab) = cross.unwrap_or((ref_y, ref_y));
    let run = |y0: usize, y1: usize, id_rows: &mut [f32], win_rows: &mut [f32]| {
        attr_pass_b_rows_f32(
            src,
            dst,
            width,
            height,
            &planes.mu1,
            &planes.mu2,
            &planes.ssq,
            &planes.s12,
            &planes.act,
            &planes.bs2,
            ax,
            ab,
            ref_y,
            co,
            y0,
            y1,
            id_rows,
            win_rows,
        );
    };
    #[cfg(feature = "threads")]
    if parallel && height > BAND {
        use rayon::prelude::*;
        id_plane[..width * height]
            .par_chunks_mut(BAND * width)
            .zip(win_plane[..width * height].par_chunks_mut(BAND * width))
            .enumerate()
            .for_each(|(band, (id_rows, win_rows))| {
                let y0 = band * BAND;
                let y1 = (y0 + BAND).min(height);
                run(y0, y1, id_rows, win_rows);
            });
        attr_pass_b_blockiness_f32(src, dst, width, height, co.c_blockiness, id_plane, lattice);
        return;
    }
    let _ = parallel;
    run(0, height, id_plane, win_plane);
    attr_pass_b_blockiness_f32(src, dst, width, height, co.c_blockiness, id_plane, lattice);
}

/// Edge-width pass-B coefficient for one (scale `u`, `ch`) — a free fn
/// (extracted 2026-08-05, appendix N) so the fused folded-944 path, which
/// derives its inputs from walk RETENTION instead of the attr pass-A
/// replication, shares the exact formula. Contributions from E(t) at
/// t = u (denominator role of the decay ratio) and t = u−1 (numerator
/// role); E(t) = 1 − bounded_sim(a, b); the LAST scale's slot is a copy
/// of E(n−2) so its gradient weight folds into E(n−2)'s. δmgd per
/// refined pixel = −(|∇d|−|∇s|)_i/N on BOTH terms (sign FD-gate-verified
/// in C2a). `dims[u]` are scale `u`'s plane dims.
#[cfg(feature = "custom-profiles")]
fn attr_ew_coeff(
    s_v2: &[f64],
    mg: &[[(f64, f64); 3]],
    dims: &[(usize, usize)],
    n_scales: usize,
    u: usize,
    ch: usize,
) -> f64 {
    if n_scales < 2 {
        return 0.0;
    }
    let mut coeff = 0.0f64;
    let n_u = (dims[u].0 * dims[u].1) as f64;
    let s_at = |t: usize| -> f64 {
        let v2b = t * 3 * FEATURES_PER_CHANNEL_V2_TOTAL + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
        let mut sk = s_v2
            .get(v2b + idx::EDGE_WIDTH_CHANGE)
            .copied()
            .unwrap_or(0.0);
        if t == n_scales - 2 {
            let lb = (n_scales - 1) * 3 * FEATURES_PER_CHANNEL_V2_TOTAL
                + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
            sk += s_v2
                .get(lb + idx::EDGE_WIDTH_CHANGE)
                .copied()
                .unwrap_or(0.0);
        }
        sk
    };
    // (−∂bs/∂b, pgd, gd) for E(t).
    let de_db_at = |t: usize| -> (f64, f64, f64) {
        let (pgs, pgd) = mg[t][ch];
        let (gs, gd) = mg[t + 1][ch];
        let a = gs / (pgs + C_GRAD_DECAY);
        let b = gd / (pgd + C_GRAD_DECAY);
        let den = a * a + b * b + C_EDGEWIDTH;
        let dbs_db = (2.0 * a * den - (2.0 * a * b + C_EDGEWIDTH) * 2.0 * b) / (den * den);
        (-dbs_db, pgd, gd)
    };
    // E(u): scale u is the DENOMINATOR (mgd_t) of b.
    if u + 1 < n_scales {
        let sk = s_at(u);
        if sk != 0.0 {
            let (de_db, pgd, gd) = de_db_at(u);
            coeff +=
                sk * de_db * (-gd / ((pgd + C_GRAD_DECAY) * (pgd + C_GRAD_DECAY))) * (-1.0 / n_u);
        }
    }
    // E(u−1): scale u is the NUMERATOR (mgd_{t+1}) of b.
    if u >= 1 {
        let sk = s_at(u - 1);
        if sk != 0.0 {
            let (de_db, pgd, _gd) = de_db_at(u - 1);
            coeff += sk * de_db * (1.0 / (pgd + C_GRAD_DECAY)) * (-1.0 / n_u);
        }
    }
    coeff
}

/// Pass B for one scale (combine → spread window mass → upsample into the
/// full-res canvas) — a free fn (extracted 2026-08-05, appendix N) shared
/// verbatim by the standalone attribution pipeline and the fused
/// folded-944 retention path. `rplanes`/`dplanes` are the scale's pyramid
/// planes, `dims` the per-scale plane dims, `(w0, h0)` the scale-0
/// (canvas) dims.
#[cfg(feature = "custom-profiles")]
#[allow(clippy::too_many_arguments)]
fn attr_pass_b_for_scale(
    scale: usize,
    rplanes: [&[f32]; 3],
    dplanes: [&[f32]; 3],
    planes: &[AttrChPlanes; 3],
    cells: &[[AttrCellSums; 3]],
    mg: &[[(f64, f64); 3]],
    dims: &[(usize, usize)],
    n_scales: usize,
    s_v2: &[f64],
    s_append: Option<&[f64]>,
    s_append2: Option<&[f64]>,
    want_append: bool,
    parallel: bool,
    w0: usize,
    h0: usize,
    scale_density: &mut [f64],
    win_plane: &mut [f64],
    sink: &mut crate::attribution::AttrSinkF64<'_>,
    spread_tmp: &mut Vec<f64>,
    lattice: &mut crate::attribution::LatticeSet,
) {
    let tpb = crate::perf_clock::Instant::now();
    let (ws, hs) = dims[scale];
    let n = ws * hs;
    scale_density[..n].fill(0.0);
    win_plane[..n].fill(0.0);
    // Exact lattice-step boundary terms (BLOCKINESS) are collected per scale into `lattice`.
    let mut lat = lattice.take(scale, ws, hs);
    for ch in 0..3 {
        let append_active = append_cell_active(want_append, ch, scale);
        let cross: Option<(&[f32], &[f32])> = if ch == 1 {
            Some((&planes[0].act, &planes[2].act))
        } else {
            None
        };
        let co = derive_v2app_coeffs(
            s_v2,
            s_append,
            s_append2,
            scale,
            ch,
            &cells[scale][ch],
            append_active,
            ch == 1,
            attr_ew_coeff(s_v2, mg, dims, n_scales, scale, ch),
            // The standalone walk computes its planes at the process revision (`validate_wide_revision`).
            crate::ssim_form::active_revision(),
            // The standalone route keeps the removal split (no paint modes).
            crate::map_paint::SplitMode::Repair,
        );
        attr_pass_b_channel(
            rplanes[ch],
            dplanes[ch],
            ws,
            hs,
            &planes[ch],
            cross,
            rplanes[1],
            &co,
            parallel,
            &mut scale_density[..n],
            &mut win_plane[..n],
            lat.as_mut().map(|(l, lw, lh)| (l, *lw, *lh)),
        );
    }
    if let Some((l, _, _)) = lat {
        lattice.put(l);
    }
    crate::blur::box_spread_sum_preserving(&mut win_plane[..n], ws, hs, BLUR_RADIUS, spread_tmp);
    for (d, s) in scale_density[..n].iter_mut().zip(win_plane[..n].iter()) {
        *d += *s;
    }
    PERF_PASSB.with(|c| c.set(c.get() + tpb.elapsed().as_secs_f64()));
    // Sum-preserving footprint upsample (the v2 pyramid floor-halves,
    // so footprints are full 2^s × 2^s blocks; ÷4^s is exact).
    let factor = 1usize << scale;
    match sink {
        crate::attribution::AttrSinkF64::Canvas(canvas) => {
            if factor == 1 {
                for (c, &v) in canvas.iter_mut().zip(scale_density[..n].iter()) {
                    *c += v;
                }
            } else {
                let inv_area = 1.0 / ((factor * factor) as f64);
                for sy in 0..hs {
                    let y0 = sy * factor;
                    let y1 = (y0 + factor).min(h0);
                    for sx in 0..ws {
                        let v = scale_density[sy * ws + sx] * inv_area;
                        let x0 = sx * factor;
                        let x1 = (x0 + factor).min(w0);
                        for row in canvas[y0 * w0..].chunks_mut(w0).take(y1.saturating_sub(y0)) {
                            for slot in &mut row[x0..x1] {
                                *slot += v;
                            }
                        }
                    }
                }
            }
        }
        crate::attribution::AttrSinkF64::Bins(accum) => {
            accum.add_scale_plane_f64(&scale_density[..n], ws, hs, factor);
        }
    }
}

/// f32 twin of [`attr_pass_b_for_scale`] (appendix P lever 1) — the FUSED
/// entry's per-scale pass B: f64 coefficient derivation (identical inputs),
/// folded to the f32 pack, f32 combine kernels, window spread fused with
/// the window→identity merge ([`crate::blur::box_spread_merge_f32`]), f32
/// sum-preserving footprint upsample into the f32 canvas.
///
/// `custom-profiles`-gated: calls `crate::attribution::
/// upsample_add_sum_preserving_f32`, which does not exist without that
/// feature (the fused-entry cluster; af4417f8 rule).
#[cfg(feature = "custom-profiles")]
#[allow(clippy::too_many_arguments)]
fn attr_pass_b_for_scale_f32(
    scale: usize,
    rplanes: [&[f32]; 3],
    dplanes: [&[f32]; 3],
    planes: &[AttrChPlanes; 3],
    cells: &[[AttrCellSums; 3]],
    mg: &[[(f64, f64); 3]],
    dims: &[(usize, usize)],
    n_scales: usize,
    s_v2: &[f64],
    s_append: Option<&[f64]>,
    s_append2: Option<&[f64]>,
    want_append: bool,
    parallel: bool,
    w0: usize,
    h0: usize,
    scale_density: &mut [f32],
    win_plane: &mut [f32],
    sink: &mut crate::attribution::AttrSinkF32<'_>,
    spread_tmp: &mut Vec<f32>,
    spread_out: &mut Vec<f32>,
    geometry: Option<&crate::sampling::Geometry>,
    lattice: &mut crate::attribution::LatticeSet,
    guide0: Option<&[f64]>,
    paint: crate::map_paint::MapPaint,
    revision: FormulaRevision,
) {
    let tpb = crate::perf_clock::Instant::now();
    let (ws, hs) = dims[scale];
    let n = ws * hs;
    scale_density[..n].fill(0.0);
    win_plane[..n].fill(0.0);
    // Exact lattice-step boundary terms need plane pixels to map one-to-one onto source footprints, which a
    // sampling geometry breaks; there the legacy symmetric split stays.
    let mut lat = if geometry.is_none() {
        lattice.take(scale, ws, hs)
    } else {
        None
    };
    for ch in 0..3 {
        if !attribution_cell_active(s_v2, s_append, s_append2, scale, ch) {
            continue;
        }
        let append_active = append_cell_active(want_append, ch, scale);
        let cross: Option<(&[f32], &[f32])> =
            if ch == 1 && planes[0].act.len() >= n && planes[2].act.len() >= n {
                Some((&planes[0].act, &planes[2].act))
            } else {
                None
            };
        let co = derive_v2app_coeffs(
            s_v2,
            s_append,
            s_append2,
            scale,
            ch,
            &cells[scale][ch],
            append_active,
            ch == 1,
            attr_ew_coeff(s_v2, mg, dims, n_scales, scale, ch),
            revision,
            paint.split,
        );
        let co32 = v2app_coeffs_fold_f32(&co, cross.is_some());
        attr_pass_b_channel_f32(
            rplanes[ch],
            dplanes[ch],
            ws,
            hs,
            &planes[ch],
            cross,
            rplanes[1],
            co32,
            parallel,
            &mut scale_density[..n],
            &mut win_plane[..n],
            lat.as_mut().map(|(l, lw, lh)| (l, *lw, *lh)),
        );
    }
    if let Some((l, _, _)) = lat {
        lattice.put(l);
    }
    // Window spread fused with the window→identity merge (value-exact vs
    // spread-then-add; parallel is bitwise-invariant per its gate). The
    // research window modes (`crate::map_paint`) replace the uniform spread.
    crate::map_paint::merge_window(
        paint.window,
        &mut win_plane[..n],
        &mut scale_density[..n],
        ws,
        hs,
        BLUR_RADIUS,
        || crate::map_paint::difference_guide(rplanes, dplanes, n, paint.guide),
        spread_tmp,
        spread_out,
        parallel && n >= crate::blur::SPREAD_PARALLEL_MIN_N,
    );
    PERF_PASSB.with(|c| c.set(c.get() + tpb.elapsed().as_secs_f64()));
    match sink {
        crate::attribution::AttrSinkF32::Canvas(canvas) => {
            crate::attribution::upsample_add_sum_preserving_f32(
                &scale_density[..n],
                ws,
                hs,
                canvas,
                w0,
                h0,
                1usize << scale,
            );
        }
        crate::attribution::AttrSinkF32::Bins(accum) => {
            if let Some(geometry) = geometry {
                let projected = geometry.project(&scale_density[..n], scale);
                let (w, h) = geometry.logical_dimensions();
                accum.add_scale_plane_f32(&projected, w, h, 1);
            } else {
                accum.add_scale_plane_painted_f32(
                    &scale_density[..n],
                    ws,
                    hs,
                    scale,
                    paint.paint,
                    guide0,
                );
            }
        }
    }
}

/// Build the v2 (+ optional append) attribution density for a pair.
///
/// `s_v2` = raw `∂score/∂f` for the v2 block (`f372..720` layout,
/// `scale*87 + ch*29 + slot`, up to 348 entries); `s_append` likewise for
/// the append block (`f720..924`, `scale*51 + ch*17 + slot`, 204 entries);
/// `s_append2` for the append2 block (`f924..944`, `scale*5 + slot`, 20
/// entries, Y-only — see `derive_v2app_coeffs`'s append2 section for which
/// of the 5 local slots carry a term and why the other three are exactly
/// zero). Toggles are fixed to `V2NewFeatureToggles::default()` — the
/// 924/944-regime canon. See the section comment above for integrand
/// classes and the documented approximations.
///
/// The whole attribution-density cluster (this entry, its pass A/B helpers,
/// and the f32 fused twins above) is `custom-profiles`-gated: every non-test
/// consumer lives in `crate::attribution`, which only exists under that
/// feature — the same both-features rule af4417f8 applied to the
/// `Fused944Session` re-export. Without the gates, a
/// `feature-regime-v2`-only build failed to COMPILE (E0433 on
/// `crate::attribution` in the f32 pass B; flagged in 87c5e9ef) and, once
/// that was gated, dead-coded the remaining 27 items of the cluster.
#[cfg(feature = "custom-profiles")]
pub(crate) fn compute_v2_append_attribution(
    reference: &impl ImageSource,
    distorted: &impl ImageSource,
    s_v2: &[f64],
    s_append: Option<&[f64]>,
    s_append2: Option<&[f64]>,
    max_pixels: Option<usize>,
    parallel: bool,
) -> Result<V2AppendAttribution, ZensimError> {
    let (density, v2_features, append_features, lattice) = compute_v2_append_attribution_impl(
        reference, distorted, s_v2, s_append, s_append2, max_pixels, parallel, None,
    )?;
    Ok(V2AppendAttribution {
        lattice,
        density: density.expect("canvas arm always yields a density"),
        width: reference.width(),
        height: reference.height(),
        v2_features,
        append_features,
    })
}

/// Level-2 sibling: the per-scale v2/append mass folds straight into the
/// caller's [`BinAccum`](crate::attribution::BinAccum) — no full-resolution
/// v2 density plane, no trim copy. Returns `(v2_features, append_features)`
/// (the pass-A audit surface).
#[cfg(feature = "custom-profiles")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_v2_append_attribution_into_bins(
    reference: &impl ImageSource,
    distorted: &impl ImageSource,
    s_v2: &[f64],
    s_append: Option<&[f64]>,
    s_append2: Option<&[f64]>,
    max_pixels: Option<usize>,
    parallel: bool,
    accum: &mut crate::attribution::BinAccum,
) -> Result<(Vec<f64>, Vec<f64>), ZensimError> {
    let (_, v2_features, append_features, _) = compute_v2_append_attribution_impl(
        reference,
        distorted,
        s_v2,
        s_append,
        s_append2,
        max_pixels,
        parallel,
        Some(accum),
    )?;
    Ok((v2_features, append_features))
}

/// Shared core of the two entries above: `bins == None` reproduces the
/// pre-Level-2 canvas path byte-identically (alloc → accumulate → trim);
/// `Some(accum)` folds per scale into the bins and allocates no canvas.
#[cfg(feature = "custom-profiles")]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn compute_v2_append_attribution_impl(
    reference: &impl ImageSource,
    distorted: &impl ImageSource,
    s_v2: &[f64],
    s_append: Option<&[f64]>,
    s_append2: Option<&[f64]>,
    max_pixels: Option<usize>,
    parallel: bool,
    bins: Option<&mut crate::attribution::BinAccum>,
) -> Result<
    (
        Option<Vec<f64>>,
        Vec<f64>,
        Vec<f64>,
        Vec<crate::attribution::LatticeScale>,
    ),
    ZensimError,
> {
    // ZENSIM_ATTR_PERF=1: coarse section timing to stderr (perf lever triage).
    let perf_log = std::env::var("ZENSIM_ATTR_PERF").as_deref() == Ok("1");
    let t0 = crate::perf_clock::Instant::now();
    let rprep = prepare_v2_reference_impl(reference, max_pixels, parallel, false)?;
    let dprep = prepare_v2_reference_impl(distorted, max_pixels, parallel, false)?;
    let t_prep = t0.elapsed();
    let n_scales = rprep.scales.len().min(dprep.scales.len());
    let (w0, h0) = (rprep.scales[0].1, rprep.scales[0].2);
    let want_append = s_append.is_some();
    let toggles = V2NewFeatureToggles::default();

    // Single-sweep pipeline (C2b Part 2): pass B(s) needs the edge-width
    // coefficients, which need mean gradients up to scale s+1 — so the
    // schedule is A(0), A(1), B(0), A(2), B(1), ..., B(n−1), with TWO
    // ping-ponged plane sets so scale s's caches survive A(s+1). This
    // removes the former second full blur+cache sweep entirely.
    let mut scratches: [ScratchV2Strip; 3] = [
        ScratchV2Strip::new(w0 * (STRIP_ROWS + 2 * HALO_P)),
        ScratchV2Strip::new(w0 * (STRIP_ROWS + 2 * HALO_P)),
        ScratchV2Strip::new(w0 * (STRIP_ROWS + 2 * HALO_P)),
    ];
    let mut planes_pp: [[AttrChPlanes; 3]; 2] = [
        [
            AttrChPlanes::new(w0 * h0),
            AttrChPlanes::new(w0 * h0),
            AttrChPlanes::new(w0 * h0),
        ],
        [
            AttrChPlanes::new(w0 * h0),
            AttrChPlanes::new(w0 * h0),
            AttrChPlanes::new(w0 * h0),
        ],
    ];
    let mut grad_halo: Vec<f32> = Vec::new();
    let mut grad_halo_d: Vec<f32> = Vec::new();

    let mut cells: Vec<[AttrCellSums; 3]> = vec![[AttrCellSums::default(); 3]; n_scales];
    let mut v2_features = vec![0.0f64; n_scales * 3 * FEATURES_PER_CHANNEL_V2_TOTAL];
    let mut append_features = if want_append {
        vec![0.0f64; n_scales * 3 * FEATURES_PER_CHANNEL_APPEND]
    } else {
        Vec::new()
    };
    // (mean grad src, mean grad dst) per (scale, ch) for edge-width.
    let mut mg = vec![[(0.0f64, 0.0f64); 3]; n_scales];
    let want_canvas = bins.is_none();
    let mut canvas = if want_canvas {
        vec![0.0f64; w0 * h0]
    } else {
        Vec::new()
    };
    let mut sink = match bins {
        Some(accum) => crate::attribution::AttrSinkF64::Bins(accum),
        None => crate::attribution::AttrSinkF64::Canvas(&mut canvas),
    };
    let mut scale_density = vec![0.0f64; w0 * h0];
    let mut win_plane = vec![0.0f64; w0 * h0];
    let mut spread_tmp: Vec<f64> = Vec::new();

    // ── Pass A for one scale into the given plane set. ──
    #[allow(clippy::too_many_arguments)]
    fn pass_a_scale(
        rprep: &V2PreparedReference,
        dprep: &V2PreparedReference,
        scale: usize,
        want_append: bool,
        parallel: bool,
        toggles: V2NewFeatureToggles,
        planes: &mut [AttrChPlanes; 3],
        scratches: &mut [ScratchV2Strip; 3],
        cells: &mut [[AttrCellSums; 3]],
        mg: &mut [[(f64, f64); 3]],
        v2_features: &mut [f64],
        append_features: &mut [f64],
        grad_halo: &mut Vec<f32>,
        grad_halo_d: &mut Vec<f32>,
    ) {
        let (ref rplanes, ws, hs) = rprep.scales[scale];
        let dplanes = &dprep.scales[scale].0;
        let tb = crate::perf_clock::Instant::now();
        // Channel-parallel blur+cache (C2b Part 2): the three channels are
        // independent given per-channel scratch.
        #[cfg(feature = "threads")]
        let did_parallel = if parallel {
            let [p0, p1, p2] = planes;
            let [s0, s1, s2] = scratches;
            rayon::join(
                || attr_blur_cache_channel(&rplanes[0], &dplanes[0], ws, hs, want_append, s0, p0),
                || {
                    rayon::join(
                        || {
                            attr_blur_cache_channel(
                                &rplanes[1],
                                &dplanes[1],
                                ws,
                                hs,
                                want_append,
                                s1,
                                p1,
                            )
                        },
                        || {
                            attr_blur_cache_channel(
                                &rplanes[2],
                                &dplanes[2],
                                ws,
                                hs,
                                want_append,
                                s2,
                                p2,
                            )
                        },
                    )
                },
            );
            true
        } else {
            false
        };
        #[cfg(not(feature = "threads"))]
        let did_parallel = false;
        // `parallel` is only consulted by the rayon arm above; consume it on
        // no-`threads` builds (same idiom as `attr_pass_b_channel`).
        #[cfg(not(feature = "threads"))]
        let _ = parallel;
        if !did_parallel {
            for ch in 0..3 {
                attr_blur_cache_channel(
                    &rplanes[ch],
                    &dplanes[ch],
                    ws,
                    hs,
                    want_append,
                    &mut scratches[0],
                    &mut planes[ch],
                );
            }
        }
        PERF_BLUR.with(|c| c.set(c.get() + tb.elapsed().as_secs_f64()));
        let tk = crate::perf_clock::Instant::now();
        for ch in 0..3 {
            let append_active = append_cell_active(want_append, ch, scale);
            let cross: Option<(&[f32], &[f32])> = if ch == 1 {
                Some((&planes[0].act, &planes[2].act))
            } else {
                None
            };
            let cell = attr_pass_a_kernels(
                &rplanes[ch],
                &dplanes[ch],
                ws,
                hs,
                &planes[ch],
                cross,
                &rplanes[1],
                want_append,
                append_active,
                grad_halo,
                grad_halo_d,
            );
            let v2b =
                scale * 3 * FEATURES_PER_CHANNEL_V2_TOTAL + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
            let (gs, gd) = finish_channel_scale(
                &cell.dense,
                &cell.grad,
                cell.blockiness,
                cell.n,
                &mut v2_features[v2b..v2b + FEATURES_PER_CHANNEL_V2_TOTAL],
            );
            mg[scale][ch] = (gs, gd);
            if want_append {
                let apb =
                    scale * 3 * FEATURES_PER_CHANNEL_APPEND + ch * FEATURES_PER_CHANNEL_APPEND;
                if append_active {
                    finish_append(
                        &cell.dense,
                        &cell.app,
                        &cell.grad,
                        cell.n,
                        ch == 1,
                        toggles,
                        &mut append_features[apb..apb + FEATURES_PER_CHANNEL_APPEND],
                    );
                }
            }
            cells[scale][ch] = cell;
        }
        PERF_KERN.with(|c| c.set(c.get() + tk.elapsed().as_secs_f64()));
    }

    // ── Pass B (combine → spread → upsample): the shared free fn
    //    `attr_pass_b_for_scale` (+ `attr_ew_coeff` inside it) — extracted
    //    2026-08-05 (appendix N) so the fused folded-944 retention path
    //    shares them verbatim. Behavior here is unchanged. ──
    let dims: Vec<(usize, usize)> = rprep.scales.iter().map(|s| (s.1, s.2)).collect();
    let mut lattice = crate::attribution::LatticeSet::default();
    lattice.reset(reference.width(), reference.height(), true);
    let pass_b_scale = |scale: usize,
                        planes: &[AttrChPlanes; 3],
                        cells: &[[AttrCellSums; 3]],
                        mg: &[[(f64, f64); 3]],
                        scale_density: &mut Vec<f64>,
                        win_plane: &mut Vec<f64>,
                        sink: &mut crate::attribution::AttrSinkF64<'_>,
                        spread_tmp: &mut Vec<f64>,
                        lattice: &mut crate::attribution::LatticeSet| {
        let (ref rplanes, _, _) = rprep.scales[scale];
        let dplanes = &dprep.scales[scale].0;
        attr_pass_b_for_scale(
            scale,
            [&rplanes[0], &rplanes[1], &rplanes[2]],
            [&dplanes[0], &dplanes[1], &dplanes[2]],
            planes,
            cells,
            mg,
            &dims,
            n_scales,
            s_v2,
            s_append,
            s_append2,
            want_append,
            parallel,
            w0,
            h0,
            scale_density,
            win_plane,
            sink,
            spread_tmp,
            lattice,
        );
    };

    // ── Pipeline: A(s); once A(s) exists, fill the cross-scale edge-width
    //    FEATURES for scale s−1 (production replication incl. the
    //    last-scale copy), then B(s−1); finish with B(n−1). ──
    for scale in 0..n_scales {
        {
            let [set_a, set_b] = &mut planes_pp;
            let pset = if scale % 2 == 0 { set_a } else { set_b };
            pass_a_scale(
                &rprep,
                &dprep,
                scale,
                want_append,
                parallel,
                toggles,
                pset,
                &mut scratches,
                &mut cells,
                &mut mg,
                &mut v2_features,
                &mut append_features,
                &mut grad_halo,
                &mut grad_halo_d,
            );
        }
        if scale >= 1 {
            #[allow(clippy::needless_range_loop)] // ch indexes two scales of mg + feature bases
            for ch in 0..3 {
                let (pgs, pgd) = mg[scale - 1][ch];
                let (gs, gd) = mg[scale][ch];
                let decay_src = gs / (pgs + C_GRAD_DECAY);
                let decay_dst = gd / (pgd + C_GRAD_DECAY);
                let prev_b = (scale - 1) * 3 * FEATURES_PER_CHANNEL_V2_TOTAL
                    + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
                v2_features[prev_b + idx::EDGE_WIDTH_CHANGE] = bounded_dissim(
                    decay_src,
                    decay_dst,
                    C_EDGEWIDTH,
                    toggles.formula_revision.stable_forms(),
                );
                if scale == n_scales - 1 && n_scales >= 2 {
                    let this_b = scale * 3 * FEATURES_PER_CHANNEL_V2_TOTAL
                        + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
                    v2_features[this_b + idx::EDGE_WIDTH_CHANGE] =
                        v2_features[prev_b + idx::EDGE_WIDTH_CHANGE];
                }
            }
            let prev = scale - 1;
            pass_b_scale(
                prev,
                &planes_pp[prev % 2],
                &cells,
                &mg,
                &mut scale_density,
                &mut win_plane,
                &mut sink,
                &mut spread_tmp,
                &mut lattice,
            );
        }
    }
    {
        let last = n_scales - 1;
        pass_b_scale(
            last,
            &planes_pp[last % 2],
            &cells,
            &mg,
            &mut scale_density,
            &mut win_plane,
            &mut sink,
            &mut spread_tmp,
            &mut lattice,
        );
    }
    // The bin sink takes the boundary terms itself; the canvas arm returns them with its density.
    let mut lattice_out = Vec::new();
    match &mut sink {
        crate::attribution::AttrSinkF64::Bins(accum) => accum.set_lattice(lattice.finish()),
        crate::attribution::AttrSinkF64::Canvas(_) => lattice_out = lattice.finish(),
    }
    if perf_log {
        eprintln!(
            "ATTRPERF v2app: prep {:.1} ms | pipeline {:.1} ms (A: blur+cache {:.1}, kernels {:.1}; B: combine {:.1})",
            t_prep.as_secs_f64() * 1e3,
            t0.elapsed().as_secs_f64() * 1e3 - t_prep.as_secs_f64() * 1e3,
            PERF_BLUR.with(|c| c.get()) * 1e3,
            PERF_KERN.with(|c| c.get()) * 1e3,
            PERF_PASSB.with(|c| c.get()) * 1e3,
        );
        PERF_BLUR.with(|c| c.set(0.0));
        PERF_KERN.with(|c| c.set(0.0));
        PERF_PASSB.with(|c| c.set(0.0));
    }

    // (The sink's borrow of `canvas` ends with its last use above.)
    // Trim the (possibly reflect-padded sub-64) canvas to the original
    // (canvas arm only; the bins arm clipped at fold time).
    let density = if want_canvas {
        let (ow, oh) = (reference.width(), reference.height());
        Some(if ow == w0 && oh == h0 {
            canvas
        } else {
            let mut out = Vec::with_capacity(ow * oh);
            for y in 0..oh.min(h0) {
                out.extend_from_slice(&canvas[y * w0..y * w0 + ow.min(w0)]);
            }
            out
        })
    } else {
        None
    };
    Ok((density, v2_features, append_features, lattice_out))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(feature = "feature-regime-v2")]
include!("feature_v2_tests.rs");

include!("feature_v2_oracle.rs");
