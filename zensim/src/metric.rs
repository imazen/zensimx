//! Core zensim metric computation.
//!
//! Multi-scale SSIM + edge + high-frequency features in XYB color space,
//! with trained weights per feature.
//!
//! # Feature extraction pipeline
//!
//! Both images are converted to the XYB perceptual color space (cube-root LMS,
//! same as ssimulacra2 and butteraugli), then processed at multiple scales.
//! Each scale halves resolution via 2× box downscale. At each scale, 19 features
//! are extracted per XYB channel (X, Y, B): 13 basic + 6 peak/diagnostic,
//! giving **228 features total** (4 scales × 3 channels × 19 features).
//!
//! ## SSIM features (3 per channel per scale)
//!
//! Uses the ssimulacra2 variant of SSIM, which differs from standard SSIM:
//!
//! ```text
//! mu1 = blur(src), mu2 = blur(dst)
//! sigma12 = blur(src * dst)
//! sum_sq  = blur(src² + dst²)     // one blur instead of two
//!
//! num_m   = 1 - (mu1 - mu2)²      // luminance (no C1, no denominator)
//! num_s   = 2·sigma12 - 2·mu1·mu2 + C2   // structure × contrast
//! denom_s = sum_sq - mu1² - mu2² + C2     // = sigma1² + sigma2² + C2
//!
//! d = max(0, 1 - num_m · num_s / denom_s) // per-pixel SSIM error
//! ```
//!
//! The luminance component drops the standard SSIM denominator
//! `(mu1² + mu2² + C1)` — ssimulacra2's reasoning is that the denominator
//! over-weights dark-region errors, which is wrong for perceptually uniform
//! values (XYB is already gamma-like). There is no C1 constant; C2 = 0.0009.
//!
//! The `sum_sq` optimization computes `blur(src² + dst²)` with one blur
//! instead of separate `blur(src²)` and `blur(dst²)`, because the SSIM
//! formula only needs `sigma1² + sigma2²`, not each individually.
//!
//! Three pooling norms capture different aspects of the error distribution:
//! - **ssim_mean** = `mean(d)` — average error
//! - **ssim_4th**  = `(mean(d⁴))^(1/4)` — L4 norm, emphasizes worst-case errors
//! - **ssim_2nd**  = `(mean(d²))^(1/2)` — L2 norm, intermediate sensitivity
//!
//! ## Edge features (6 per channel per scale)
//!
//! Edge detection compares local detail (pixel minus local mean) between
//! source and distorted:
//!
//! ```text
//! diff_src = |src - mu1|    // source edge magnitude
//! diff_dst = |dst - mu2|    // distorted edge magnitude
//!
//! d = (1 + diff_dst) / (1 + diff_src) - 1   // per-pixel edge ratio
//!
//! artifact    = max(0,  d)   // distorted has MORE edge than source
//! detail_lost = max(0, -d)   // distorted has LESS edge than source
//! ```
//!
//! The `1 +` offsets prevent division by zero and dampen sensitivity in flat
//! regions. The ratio formulation is scale-invariant. Splitting into artifact
//! (ringing, banding, blockiness) vs detail_lost (blur, smoothing) lets the
//! model weight them independently.
//!
//! Each is pooled with three norms (mean, L4, L2) = 6 features.
//!
//! ## MSE (1 per channel per scale)
//!
//! Plain mean squared error in XYB space: `mean((src - dst)²)`.
//! No blur dependency, computed directly from pixels.
//!
//! ## High-frequency features (3 per channel per scale)
//!
//! These measure changes in local detail energy by comparing `pixel - blur(pixel)`
//! (the high-frequency residual) between source and distorted. Despite their
//! former names ("variance_loss", "texture_loss", "contrast_increase"), they
//! do NOT measure image variance — they measure the ratio of high-pass energy.
//!
//! ```text
//! hf_src_L2 = Σ(src - mu1)²    // source HF energy (L2)
//! hf_dst_L2 = Σ(dst - mu2)²    // distorted HF energy (L2)
//! hf_src_L1 = Σ|src - mu1|     // source HF magnitude (L1)
//! hf_dst_L1 = Σ|dst - mu2|     // distorted HF magnitude (L1)
//! ```
//!
//! - **hf_energy_loss** = `max(0, 1 - hf_dst_L2 / hf_src_L2)` — detail smoothed away
//! - **hf_mag_loss**    = `max(0, 1 - hf_dst_L1 / hf_src_L1)` — same, L1 (robust to outliers)
//! - **hf_energy_gain** = `max(0, hf_dst_L2 / hf_src_L2 - 1)` — detail added (ringing/sharpening)
//!
//! `hf_energy_loss` and `hf_energy_gain` are the positive and negative halves
//! of the same signal, split by ReLU — this gives the linear model separate
//! knobs for blur vs ringing without needing signed weights.
//!
//! ## Peak features (6 per channel per scale)
//!
//! Computed during the fused V-blur kernel at no extra cost:
//! - **ssim_max**, **art_max**, **det_max** — per-pixel maximum of each error type
//! - **ssim_l8**, **art_l8**, **det_l8** — L8-pooled (near-worst-case) values
//!
//! These capture outlier sensitivity that mean/L2/L4 pooling may miss.
//!
//! ## Scoring
//!
//! All 228 features are multiplied by trained weights, summed, normalized by
//! scale count, then mapped to a 0–100 score via:
//! `score = 100 - a · distance^b` (default a=18.0, b=0.7).

use crate::bake_metadata::*;
use crate::det_math::{DetPow, active_pow_form};
use crate::error::ZensimError;

/// Configuration for zensim computation.
///
/// All computation uses the streaming path, which processes scale 0 in
/// horizontal strips with fused blur+feature extraction for minimal memory
/// traffic. When `blur_passes == 1` (the default), fused H-blur + V-blur+reduce
/// SIMD kernels are used for peak performance.
///
/// Blur kernel shape for local-mean computation.
///
/// Controls how `blur(src)` and `blur(dst)` are computed at each scale.
/// The default `Box` kernel uses iterated box blur, which is O(1) per pixel
/// regardless of radius and has full SIMD optimization.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlurKernel {
    /// Iterated box blur. `passes` controls the kernel shape:
    /// - 1 = rectangular (fastest, enables fused streaming kernels)
    /// - 2 = triangular (~1.5× slower at scale 0)
    /// - 3 = piecewise-quadratic ≈ Gaussian (~2× slower)
    Box {
        /// Number of passes (1 = rectangular, 2 = triangular, 3 ≈ Gaussian).
        passes: u8,
    },
}

impl Default for BlurKernel {
    fn default() -> Self {
        Self::Box { passes: 1 }
    }
}

/// Downscale filter for pyramid construction.
///
/// Controls how each pyramid level is produced from the previous one.
/// `Box2x2` (2×2 block averaging) is the only filter the trained
/// profiles are calibrated for and the only implemented variant; the
/// enum stays `#[non_exhaustive]` so alternative kernels can be added
/// without a breaking change. (Speculative `zenresize`-gated
/// Mitchell/Lanczos variants were removed pre-0.3.0 — imazen/zensimx#47 —
/// because no compute path ever dispatched on them.)
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum DownscaleFilter {
    /// 2×2 box averaging (fastest, current default).
    #[default]
    Box2x2,
}

/// Configuration for the zensim metric computation pipeline.
///
/// Controls blur kernel, pyramid construction, and feature extraction.
/// The defaults match the trained profile and give peak performance;
/// only change these for training or research.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ZensimConfig {
    /// Request-local arithmetic for model-bound serving.
    pub(crate) formula_revision: Option<crate::feature_defs::FormulaRevision>,
    pub(crate) local_only: bool,
    pub(crate) omit_edges: bool,
    pub(crate) attribution_channels: Option<[[bool; 3]; crate::NUM_SCALES]>,
    /// Box blur radius at scale 0 (default: 5, giving an 11-pixel kernel).
    ///
    /// The blur kernel width is `2 * blur_radius + 1`. Larger radii capture
    /// coarser structure but increase computation proportionally.
    /// Both streaming and full-image paths are SIMD-optimized for any radius.
    pub blur_radius: usize,

    /// Number of box blur passes (1, 2, or 3; default: 1).
    ///
    /// Controls the blur kernel shape:
    /// - **1 pass** — rectangular kernel. Enables fused blur+feature SIMD kernels
    ///   in the streaming path (fastest).
    /// - **2 passes** — triangular kernel. Falls back to separate blur+reduce in
    ///   the streaming path (~1.5× slower at scale 0).
    /// - **3 passes** — piecewise-quadratic ≈ Gaussian. Same fallback (~2× slower).
    ///
    /// All three variants have full SIMD optimization (AVX-512 + AVX2 dispatch).
    /// The performance difference comes from whether the fused streaming kernels
    /// can be used, not from the blur itself.
    pub blur_passes: u8,

    /// Blur kernel shape (default: `BlurKernel::Box { passes: 1 }`).
    ///
    /// Overrides `blur_passes` when set. The `blur_radius` field still controls
    /// the kernel width. Currently only the `Box` variant is implemented.
    #[allow(dead_code)] // planned: not yet wired into blur dispatch
    pub blur_kernel: BlurKernel,

    /// Downscale filter for pyramid construction (default: `DownscaleFilter::Box2x2`).
    ///
    /// Controls how each pyramid level is produced. `Box2x2` is currently
    /// the only variant.
    #[allow(dead_code)] // planned: not yet wired into pyramid construction
    pub downscale_filter: DownscaleFilter,

    /// Compute all 156 features even when their weights are zero (default: false).
    ///
    /// When false, channels/features with zero weight are skipped entirely.
    /// Enable for weight training to avoid circular dependency (need all features
    /// to determine which weights should be nonzero).
    pub compute_all_features: bool,

    /// Compute extended features (25 per channel instead of 13; default: false).
    ///
    /// When true, adds 12 extra features per channel per scale:
    /// - 6 masked features (SSIM/edge/MSE weighted by source flatness)
    /// - 6 percentile/max features (worst-case SSIM/edge errors)
    ///
    /// The masking strength for extended features is controlled by
    /// `extended_masking_strength`.
    pub extended_features: bool,

    /// Masking strength for extended masked features (default: 4.0).
    ///
    /// Only used when `extended_features` is true. Controls the flatness mask:
    /// `mask[i] = 1 / (1 + k * blur(|src - mu|))`.
    ///
    /// Higher values = more aggressive masking of textured regions.
    /// Typical range: 2.0–8.0.
    pub extended_masking_strength: f32,

    /// Compute information-content-weighted (IW) features (Wang & Li
    /// 2011, IW-SSIM). When `true`, emits a per-scale per-channel block
    /// of 6 features (`iw_ssim_mean / iw_ssim_4th / iw_ssim_2nd /
    /// iw_art_4th / iw_det_4th / iw_mse`) pooled with weights derived
    /// from the source image's local activity — the OPPOSITE polarity
    /// of the `extended_features` masked block (which suppresses
    /// texture). Texture-rich regions get MORE weight, mirroring
    /// Wang's GSM-on-wavelet info-content estimator.
    ///
    /// At `num_scales = 4` and 3 channels, enabling this adds 72
    /// features (228 → 300 in the basic profile).
    ///
    /// **Zero overhead when `false`** — the per-pixel work is gated by
    /// a constant-per-call branch in the SIMD loop; the branch
    /// predictor sees a fixed direction.
    pub compute_iw_features: bool,

    /// IW weighting strength: `iw_weight[i] = 1.0 + k * blur(|src - mu|)`.
    /// Only used when `compute_iw_features` is true. Higher values
    /// concentrate the pool more aggressively on textured / edge
    /// regions. Default: 4.0 (mirrors `extended_masking_strength`).
    pub iw_strength: f32,

    /// Maximum number of downscale levels (default: 4).
    ///
    /// Each level halves resolution. 4 scales covers 1×, 2×, 4×, 8× — sufficient
    /// for most perceptual effects. The feature vector length scales linearly:
    /// `num_scales × 3 channels × 13 features`.
    ///
    /// Both paths are SIMD-optimized for any scale count.
    pub num_scales: usize,

    /// Score mapping scale factor (default: 18.0).
    ///
    /// Used in the final score formula: `score = 100 - a × d^b`, where `d` is
    /// the raw weighted distance. Larger values spread scores more aggressively.
    pub score_mapping_a: f64,

    /// Score mapping gamma exponent (default: 0.7).
    ///
    /// Used in the final score formula: `score = 100 - a × d^b`. Sub-linear
    /// gamma (< 1.0) compresses high distances, giving more resolution in the
    /// high-quality range.
    pub score_mapping_b: f64,

    /// Enable multi-threaded computation via rayon (default: true).
    pub allow_multithreading: bool,

    /// Correct-by-construction bounded squash for the linear score
    /// (mirrors [`ProfileParams::bounded_squash`](crate::profile::ProfileParams)).
    /// When `true`, `combine_scores` maps the non-negative feature
    /// distance through `100·exp(−(a/100)·d^b)` instead of `100 − a·d^b`,
    /// guaranteeing a bounded `[0, 100]` output. Default `false`.
    pub(crate) bounded_squash: bool,
}

impl ZensimConfig {
    /// The formula revision this computation runs: the request's own
    /// (`formula_revision`), or — when the request names none — the process
    /// revision, which is what `None` has always meant. The value every
    /// revision-dependent leaf on this config's walk receives (featcanon D1).
    pub(crate) fn revision(&self) -> crate::feature_defs::FormulaRevision {
        self.formula_revision
            .unwrap_or_else(crate::ssim_form::active_revision)
    }
}

impl Default for ZensimConfig {
    fn default() -> Self {
        Self {
            formula_revision: None,
            local_only: false,
            omit_edges: false,
            attribution_channels: None,
            blur_radius: 5,
            blur_passes: 1,
            blur_kernel: BlurKernel::default(),
            downscale_filter: DownscaleFilter::default(),
            compute_all_features: false,
            extended_features: false,
            extended_masking_strength: 4.0,
            compute_iw_features: false,
            iw_strength: 4.0,
            num_scales: crate::NUM_SCALES,
            score_mapping_a: 18.0,
            score_mapping_b: 0.7,
            allow_multithreading: true,
            bounded_squash: false,
        }
    }
}

/// Map a raw weighted distance to the 0–100 quality score.
///
/// Uses the default power-law mapping: `score = 100 - 18 * d^0.7`, clamped to \[0, 100\].
/// Identical images (d = 0) score 100.
///
/// For profile-specific mapping, use [`Zensim::compute`] which applies the profile's
/// `score_mapping_a` and `score_mapping_b` automatically.
pub(crate) fn distance_to_score(raw_distance: f64) -> f64 {
    distance_to_score_mapped(raw_distance, 18.0, 0.7)
}

/// Map a raw weighted distance to the quality score with custom parameters.
///
/// `score = 100 - a * d^b`. Nominally 0–100 but can go negative for
/// extreme distortions (the magnitude below zero is informative —
/// it distinguishes "slightly wrong" from "completely wrong").
/// Correct-by-construction bounded squash: `score = 100 · exp(−(a/100) · d^b)`.
///
/// For a non-negative distance `d ≥ 0` (guaranteed when the profile's
/// weights are non-negative and every feature is a non-negative
/// dissimilarity), this maps `d → (0, 100]` strictly-decreasingly, with
/// `score = 100` iff `d = 0` (its unique maximum) and `score → 0` as
/// `d → ∞`. It is a strictly-monotone transform of `d`, so it preserves
/// the *rank* of [`distance_to_score_mapped`] (identical SROCC) while
/// being **bounded `[0, 100]` by construction** — unlike `100 − a·d^b`,
/// which is unbounded below.
///
/// The `a/100` scaling makes the squash agree with `100 − a·d^b` to
/// first order at `d = 0` (`100·e^{−(a/100)·d^b} ≈ 100 − a·d^b` for
/// small `d^b`), so the near-identical regime matches the legacy dial.
/// See `docs/METRIC_INVARIANTS_MECHANISM_AND_REDESIGN_2026-05-26.md` §4.
fn bounded_score_squash(raw_distance: f64, a: f64, b: f64) -> f64 {
    if raw_distance <= 0.0 {
        return 100.0;
    }
    // F19: one form read for the whole mapping, so the `powf` and the `exp`
    // can never be evaluated under two different arms.
    let form = active_pow_form();
    let exponent = (a / 100.0) * raw_distance.det_powf(b, form);
    // exp(−x) ∈ (0, 1]; ×100 ∈ (0, 100]. Finite for all finite x ≥ 0;
    // for x = +inf, exp(−inf) = 0. Defensive clamp catches NaN only.
    let s = 100.0 * (-exponent).det_exp(form);
    if s.is_finite() { s } else { 0.0 }
}

/// Map a raw weighted distance to the quality score with custom parameters.
///
/// `score = 100 - a * d^b`. Nominally 0–100 but can go negative for
/// extreme distortions (the magnitude below zero is informative —
/// it distinguishes "slightly wrong" from "completely wrong").
fn distance_to_score_mapped(raw_distance: f64, a: f64, b: f64) -> f64 {
    crate::score_math::distance_to_score_mapped(raw_distance, a, b, active_pow_form())
}

/// Compute score from raw features using custom weights.
///
/// `features`: raw features from `ZensimResult::features()`.
/// `weights`: one weight per feature; `weights.len()` must equal
/// `features.len()`.
///
/// Returns `(score, raw_distance)` on success, or
/// [`ZensimError::FeatureWeightsLengthMismatch`] if the slices have
/// different lengths.
#[cfg_attr(not(feature = "training"), allow(dead_code))]
pub fn try_score_from_features(
    features: &[f64],
    weights: &[f64],
) -> Result<(f64, f64), ZensimError> {
    if features.len() != weights.len() {
        return Err(ZensimError::FeatureWeightsLengthMismatch);
    }
    let raw_distance: f64 = features
        .iter()
        .zip(weights.iter())
        .map(|(&f, &w)| w * f)
        .sum();
    // Normalize by number of scales.
    // Layout: [scored × N_scales] [peaks × N_scales] [masked × N_scales]
    // 156 = 39×4, 228 = 57×4, 300 = 75×4 — all divide by 4 scales.
    let per_scale_candidates = [
        FEATURES_PER_CHANNEL_EXTENDED * 3,   // 75
        FEATURES_PER_CHANNEL_WITH_PEAKS * 3, // 57
        FEATURES_PER_CHANNEL_BASIC * 3,      // 39
    ];
    let features_per_scale = per_scale_candidates
        .iter()
        .copied()
        .find(|&ps| ps > 0 && features.len().is_multiple_of(ps))
        .unwrap_or(FEATURES_PER_CHANNEL_BASIC * 3);
    let n_scales = features.len() / features_per_scale;
    let raw_distance = raw_distance / n_scales.max(1) as f64;
    Ok((distance_to_score(raw_distance), raw_distance))
}

/// Score a precomputed feature vector under a [`ZensimProfile`].
///
/// **Bit-exact match with `Zensim::new(profile).compute(...)` post-feature-extraction.**
///
/// This is the entry point used by alternative feature backends (e.g.
/// `zensim-gpu`'s CubeCL pipeline) to produce a `0..100` score with
/// the same calibration the CPU path applies — runs the same bake
/// forward pass, per-sample-α / hybrid head, tanh-pin, PCHIP spline,
/// per-codec affine, and clamp / soft-clamp / extrapolate-score
/// disposition the canonical CPU `compute` flow uses.
///
/// `features` must match the profile's MLP input width (228 / 300 /
/// 372 for the V_18 / V_20-extended / V_22-IW families respectively,
/// or +4 if the bake declares size axes). For linear V0_1 / V0_2
/// profiles (no `mlp_bytes`), falls back to the legacy
/// `dot(features, weights)` + `100 − A·d^B` path.
///
/// `width` / `height` are passed through to the bake forward pass
/// for the optional `--mlp-size-axes` feature expansion (4 trailing
/// floats appended to the feature vector when the bake's `n_inputs
/// == features.len() + 4`).
///
/// # Errors
///
/// - [`ZensimError::ModelLoadFailed`] if the bake bytes fail to parse.
/// - [`ZensimError::ModelForwardFailed`] if the bake's input width
///   doesn't match the supplied feature vector (modulo the +4 size-axis
///   convention).
/// - [`ZensimError::FeatureWeightsLengthMismatch`] if the profile is
///   linear (no MLP) and `features.len() != weights.len()`.
pub fn score_features_with_profile(
    profile: crate::profile::ZensimProfile,
    features: &[f64],
    width: u32,
    height: u32,
) -> Result<f64, ZensimError> {
    score_features_with_profile_and_codec(profile, features, width, height, None)
}

/// Same as [`score_features_with_profile`] but accepts an optional
/// codec hint that drives the per-codec post-spline affine calibration
/// (EXP-CROSS-CODEC-V11-E). Pass `Some("jpeg")` / `Some("webp")` /
/// `Some("avif")` / `Some("jxl")` (or their `zenfoo` aliases) when the
/// caller knows which codec produced the distorted image and the
/// profile carries `zentrain.per_codec_calibration` metadata.
pub fn score_features_with_profile_and_codec(
    profile: crate::profile::ZensimProfile,
    features: &[f64],
    width: u32,
    height: u32,
    codec_hint: Option<&str>,
) -> Result<f64, ZensimError> {
    let params = profile.params();

    // Linear (no-MLP) profiles — V0_1 / V0_2. Run the classic
    // `dot(features, weights)` + `100 − A·d^B` path.
    if params.mlp_bytes.is_none() {
        if features.len() != params.weights.len() {
            return Err(ZensimError::FeatureWeightsLengthMismatch);
        }
        let raw_distance: f64 = features
            .iter()
            .zip(params.weights.iter())
            .map(|(&f, &w)| w * f)
            .sum();
        let raw_per_scale = raw_distance / params.num_scales.max(1) as f64;
        let score = if params.bounded_squash {
            bounded_score_squash(
                raw_per_scale,
                params.score_mapping_a,
                params.score_mapping_b,
            )
        } else {
            distance_to_score_mapped(
                raw_per_scale,
                params.score_mapping_a,
                params.score_mapping_b,
            )
        };
        // V0_1 / V0_2 are not on the extrapolate / soft-clamp paths
        // (their `ProfileParams` defaults are both `false`), so a hard
        // [0, 100] clamp matches the canonical `apply_mlp_scoring`
        // disposition.
        return Ok(score.clamp(0.0, 100.0));
    }

    // MLP-scored profile. Mirror `apply_mlp_scoring_with_codec`
    // exactly so the calibration disposition (raw / hard clamp /
    // soft clamp / extrapolate) matches the canonical CPU flow.
    //
    // Build a placeholder ZensimResult to thread through
    // `apply_mlp_scoring_with_codec` — that function reads
    // `result.features()`, calls `forward_one_bake_with_codec`, and
    // writes `set_mlp_score(raw, score)`. We then return the score.
    //
    // (We could also call `forward_one_bake_with_codec` directly +
    // duplicate the post-network dispatch here, but threading through
    // `apply_mlp_scoring_with_codec` ensures every future calibration
    // tweak — ensemble routing, b3 mix, extrapolate-score — flows
    // through this entry point automatically.)
    let mut result = ZensimResult::new(f64::NAN, f64::NAN, features.to_vec(), profile, [0.0; 3]);
    apply_mlp_scoring_with_codec(&mut result, params, width, height, codec_hint)?;
    Ok(result.score())
}

/// Batched central-difference gradient of the profile's score at `base`
/// (campaign appendix Y, L-Y1 — the jxl-loop iteration-0 probe owner).
///
/// Semantics are EXACTLY the sequential recipe of calling
/// [`score_features_with_profile`] `2·N` times with the loop's probe rule:
///
/// ```text
/// eps_k   = max(|base[k]| · 1e-3, 1e-5)
/// grad[k] = (score(base + eps_k·e_k) − score(base − eps_k·e_k)) / (2·eps_k)
///           (0.0 when either forward is non-finite or errors)
/// ```
///
/// but the bake is parsed ONCE and one `Predictor` (+ one f32 input
/// buffer) serves every forward — removing the per-forward
/// `Model::from_bytes` re-parse, `Predictor::new` allocations, and the
/// `ZensimResult` feature-vector clone that made the sequential recipe
/// ~65% of the jxl k3 loop wall. Each forward's arithmetic is identical
/// to the canonical path (shared [`bake_dispatch_one`] +
/// [`prep_bake_input_f32`] + [`dispose_mlp_raw`]), so the gradient is
/// **bitwise-equal** to the sequential recipe — gated by
/// `fd_gradient_bitwise_matches_sequential`.
///
/// Two exact-zero shortcuts (both bitwise-equal to running the forwards):
/// dead-column-pruned bakes (`FeatureTransform::Drop`) and prefix-sized
/// bakes never read the affected column, so `up == dn` bitwise and the
/// component is exactly `0.0` — the forwards are skipped.
///
/// Profiles outside the fast path (linear / ensemble-routed / b3-mixed /
/// min-max-head) fall back to the per-probe canonical path — same
/// result, sequential cost.
///
/// # Errors
///
/// [`ZensimError::ModelLoadFailed`] if the bake bytes fail to parse (the
/// sequential recipe would instead return an all-zero gradient through
/// per-call NaN; callers that want that behavior can map the error to
/// `vec![0.0; base.len()]`).
pub fn score_features_fd_gradient_with_profile(
    profile: crate::profile::ZensimProfile,
    base: &[f64],
    width: u32,
    height: u32,
) -> Result<Vec<f64>, ZensimError> {
    let params = profile.params();
    let n = base.len();
    let mut probe = base.to_vec();

    // Sequential fallback: the canonical per-probe path, verbatim.
    let seq_fallback = |probe: &mut Vec<f64>| -> Vec<f64> {
        let sf =
            |f: &[f64]| score_features_with_profile(profile, f, width, height).unwrap_or(f64::NAN);
        let mut grad = vec![0.0f64; n];
        for (k, gk) in grad.iter_mut().enumerate() {
            let eps = (base[k].abs() * 1e-3).max(1e-5);
            probe[k] = base[k] + eps;
            let up = sf(probe);
            probe[k] = base[k] - eps;
            let dn = sf(probe);
            probe[k] = base[k];
            *gk = if up.is_finite() && dn.is_finite() {
                (up - dn) / (2.0 * eps)
            } else {
                0.0
            };
        }
        grad
    };

    // Fast path: single-bake MLP profiles (no ensemble classifier, no b3
    // secondary, no min-max head — those route through the fallback).
    let fast_bytes: Option<&'static [u8]> = match (
        params.mlp_bytes,
        params.ensemble_classifier_bytes,
        params.mlp_bytes_b3,
    ) {
        (Some(loader), None, None) => Some(loader()),
        _ => None,
    };
    let Some(bytes) = fast_bytes else {
        return Ok(seq_fallback(&mut probe));
    };
    let model = cached_profile_model(bytes)?;
    let bundle = cached_bake_metadata(bytes, &model)?;
    // Derived-input bakes (E33 `fx1`) take the canonical per-probe forward,
    // which owns their model-row construction.
    if bundle.minmax_head.is_some() || bundle.derived_inputs.is_some() {
        return Ok(seq_fallback(&mut probe));
    }
    let n_inputs = model.caller_input_width();
    let needs_transforms = model.has_nontrivial_feature_transforms();
    let mut predictor = crate::mlp::Predictor::new(&model);
    let per_sample_alpha = bundle.per_sample_alpha.as_deref();
    let hybrid_head = bundle.hybrid_head.as_deref();
    let tanh_pin_scale = bundle.tanh_pin_scale;
    let output_spline = bundle.output_spline.as_deref();
    // This entry takes no codec hint (mirrors the loop's probe closure,
    // which calls the hint-less `score_features_with_profile`).
    let per_codec_affine: Option<(f32, f32)> = None;

    // Dropped raw columns (dead-column-pruned bakes): valid only when the
    // transform table indexes the same positions the caller probes.
    let drop_mask: Option<&[zenpredict::FeatureTransform]> = if n_inputs == n {
        model.feature_transforms()
    } else {
        None
    };

    // **LAYOUT — the same gather the single-forward path does.** A bake
    // declaring a DENSE feature set reads its ids at packed positions, so a
    // caller's identity-laid-out probe must be GATHERED, not sliced. Without
    // this the batched entry took `prep_bake_input_f32`'s PREFIX branch
    // (`n_inputs < features.len()` ⇒ `features[..n_inputs]`) and differentiated
    // the wrong features — MEASURED against the sequential recipe the moment
    // shipped `B` went dense: component 0 read −748.7 batched against the
    // correct 0.0, because caller position 0 carries id `f0` and `B` declares
    // `f3..f369`. For an identity-layout bake (no declaration, or one this
    // build cannot reproduce) nothing below changes.
    //
    // **Not `#[cfg]`-conditional, and must never become so again** — see the
    // single-forward path's note and
    // `benchmarks/dense_serving_ungate_2026-09-06.md`.
    let layout = crate::feature_layout::declared_layout(&model);
    let dense_ids: Option<Vec<u16>> = (!layout.is_identity()).then(|| {
        (0..layout.width())
            .filter_map(|p| layout.slot_at(p))
            .collect()
    });
    // Same refusal as the single-forward path, for the same reason — a short
    // row would be silently zero-filled at ids the bake genuinely reads.
    if dense_ids.is_some() && n < layout.walk_width() {
        return Err(ZensimError::ModelForwardFailed {
            reason: "the bake declares feature ids this feature vector does not reach",
        });
    }

    let mut f32buf: Vec<f32> = Vec::with_capacity(n_inputs.max(n));
    let mut gathered: Vec<f64> = Vec::new();
    let mut grad = vec![0.0f64; n];
    for k in 0..n {
        // A DENSE bake reads ids, so a caller position the declaration does
        // not name cannot reach the forward at all — `up == dn` bitwise,
        // component exactly 0.0. This REPLACES the prefix rule below for such
        // a bake: `n_inputs` is its packed width, and `k >= n_inputs` would
        // zero components for ids it very much does read.
        if let Some(ids) = &dense_ids {
            if !ids.iter().any(|&id| usize::from(id) == k) {
                continue;
            }
        } else if n_inputs < n && k >= n_inputs {
            // Prefix-sized bake: columns past `n_inputs` never enter the
            // forward — `up == dn` bitwise, component exactly 0.0.
            continue;
        }
        // Dropped column: the transform removes it before layer 0 —
        // `up == dn` bitwise, component exactly 0.0.
        if let Some(ts) = drop_mask
            && matches!(ts[k], zenpredict::FeatureTransform::Drop)
        {
            continue;
        }
        let eps = (base[k].abs() * 1e-3).max(1e-5);
        let mut fwd = |probe: &[f64], p: &mut crate::mlp::Predictor<'_>| -> f64 {
            let src: &[f64] = if dense_ids.is_some() {
                layout.gather(probe, &mut gathered);
                &gathered
            } else {
                probe
            };
            if prep_bake_input_f32(src, n_inputs, width, height, &mut f32buf).is_err() {
                return f64::NAN;
            }
            match bake_dispatch_one(
                p,
                &f32buf,
                needs_transforms,
                per_sample_alpha,
                hybrid_head,
                tanh_pin_scale,
                output_spline,
                per_codec_affine,
            ) {
                Ok(raw) => dispose_mlp_raw(raw, params),
                Err(_) => f64::NAN,
            }
        };
        probe[k] = base[k] + eps;
        let up = fwd(&probe, &mut predictor);
        probe[k] = base[k] - eps;
        let dn = fwd(&probe, &mut predictor);
        probe[k] = base[k];
        grad[k] = if up.is_finite() && dn.is_finite() {
            (up - dn) / (2.0 * eps)
        } else {
            0.0
        };
    }
    Ok(grad)
}

/// Pre-compute reference with a custom number of pyramid scales.
///
/// Use this when calling [`compute_zensim_with_ref_and_config`] with a non-default
/// `num_scales`. The precomputed data must have at least as many scales as the config
/// requests.
///
/// **Research / feature-extraction API; not stable.** The
/// `ZensimConfig` knobs that this entry exposes (blur kernel, scale
/// count, masking) change metric internals whose outputs are
/// incomparable with the default trained weights. Used by the
/// zensim-validate feature-extraction binaries; not part of the
/// 0-100 score surface.
#[doc(hidden)]
#[cfg_attr(not(feature = "training"), allow(dead_code))]
pub fn precompute_reference_with_scales(
    source: &[[u8; 3]],
    width: usize,
    height: usize,
    num_scales: usize,
) -> Result<crate::streaming::PrecomputedReference, ZensimError> {
    if width < 8 || height < 8 {
        return Err(ZensimError::ImageTooSmall);
    }
    let pixels = width
        .checked_mul(height)
        .ok_or(ZensimError::ImageTooLarge)?;
    // Reject any width/height combination whose padded-plane size would
    // overflow `usize` on the current target.
    check_within_max_pixels(width, height, None)?;
    if source.len() != pixels {
        return Err(ZensimError::InvalidDataLength);
    }
    let src_img = crate::source::RgbSlice::try_new(source, width, height)?;
    Ok(crate::streaming::PrecomputedReference::new(
        &src_img,
        num_scales,
        true,
        crate::ssim_form::active_revision(),
    ))
}

#[cfg(any(feature = "training", test))]
fn refuse_rev5_masked_config(config: &ZensimConfig) -> Result<(), ZensimError> {
    if config.revision() >= crate::feature_defs::FormulaRevision::Rev5
        && (config.extended_features || config.compute_iw_features)
    {
        return Err(ZensimError::ModelLoadFailed {
            reason: "formula revision 5 supports only basic + peaks + v2; raw configuration requests masked or IW features",
        });
    }
    Ok(())
}

/// Compute zensim with a precomputed reference and custom configuration.
///
/// **Research / feature-extraction API; not stable.** The
/// `ZensimConfig` knobs that this entry exposes change metric
/// internals whose outputs are incomparable with the default trained
/// weights. The `config.num_scales` must not exceed the number of
/// scales in `precomputed`. Used by the zensim-validate
/// feature-extraction binaries.
#[doc(hidden)]
#[cfg(feature = "training")]
pub fn compute_zensim_with_ref_and_config(
    precomputed: &crate::streaming::PrecomputedReference,
    distorted: &[[u8; 3]],
    width: usize,
    height: usize,
    config: ZensimConfig,
) -> Result<ZensimResult, ZensimError> {
    // These two entries take a caller-built config rather than a profile, so
    // they never pass through `config_from_params` — the route refusal has to
    // be repeated here or a `blur_passes != 1` config would reach the strip
    // walk under an arithmetic revision that does not serve it.
    crate::ssim_form::check_route(&config)?;
    refuse_rev5_masked_config(&config)?;
    if width < 8 || height < 8 {
        return Err(ZensimError::ImageTooSmall);
    }
    let pixels = width
        .checked_mul(height)
        .ok_or(ZensimError::ImageTooLarge)?;
    if distorted.len() != pixels {
        return Err(ZensimError::InvalidDataLength);
    }
    if precomputed.width() != width || precomputed.height() != height {
        return Err(ZensimError::DimensionMismatch);
    }
    let dst_img = crate::source::RgbSlice::try_new(distorted, width, height)?;
    // The reference was reflect-padded inside `PrecomputedReference::new`;
    // pad the distorted to match or the scale walk asserts on the width
    // mismatch. Same rule as `Zensim::compute_with_ref`.
    let result = if needs_pyramid_pad(width, height, config.num_scales) {
        let d = reflect_pad_for_scales(&dst_img, config.num_scales);
        crate::streaming::compute_zensim_streaming_with_ref(precomputed, &d, &config, WEIGHTS, None)
    } else {
        crate::streaming::compute_zensim_streaming_with_ref(
            precomputed,
            &dst_img,
            &config,
            WEIGHTS,
            None,
        )
    };
    // This entry point scores via the plain WEIGHTS (V0_2 linear) distance,
    // never an MLP forward pass — tag honestly rather than inheriting
    // `combine_scores`'s internal `codec_target()` (`B`) placeholder tag
    // (which every `Zensim::compute*` caller overrides but this standalone
    // function does not go through).
    Ok(result.with_profile(crate::profile::ZensimProfile::LegacyLinearV0_2))
}

/// Per-scale statistics collected during computation.
#[derive(Default, Clone)]
pub(crate) struct ScaleStats {
    /// SSIM statistics: [mean_d, root4_d] per channel = 6 values
    pub(crate) ssim: [f64; 6],
    /// Edge features: [art_mean, art_4th, det_mean, det_4th] per channel = 12 values
    pub(crate) edge: [f64; 12],
    /// Per-channel MSE: mean((src - dst)²) for X, Y, B
    pub(crate) mse: [f64; 3],
    /// High-frequency energy loss (L2): max(0, 1 - Σ(dst-mu_dst)²/Σ(src-mu_src)²) per channel.
    /// Measures loss of local detail energy relative to source. Sensitive to blur/smoothing.
    pub(crate) hf_energy_loss: [f64; 3],
    /// Raw destination HF sum for the exact active gain-form derivative.
    /// Read only by the `custom-profiles` attribution path.
    #[cfg_attr(not(feature = "custom-profiles"), allow(dead_code))]
    pub(crate) hf_sq_dst_sum: [f64; 3],
    /// High-frequency magnitude loss (L1): max(0, 1 - Σ|dst-mu_dst|/Σ|src-mu_src|) per channel.
    /// Like hf_energy_loss but with L1 norm — more robust to outliers.
    pub(crate) hf_mag_loss: [f64; 3],
    /// 2nd-power pooled SSIM: [root2_d] per channel = 3 values
    pub(crate) ssim_2nd: [f64; 3],
    /// Edge 2nd power: [art_2nd, det_2nd] per channel = 6 values
    pub(crate) edge_2nd: [f64; 6],
    /// High-frequency energy gain (L2): max(0, Σ(dst-mu_dst)²/Σ(src-mu_src)² - 1) per channel.
    /// Measures added local detail energy (ringing, sharpening artifacts).
    pub(crate) hf_energy_gain: [f64; 3],
    // --- Extended features (only populated when extended_features=true) ---
    /// Masked SSIM: [mean, 4th, 2nd] per channel = 9 values
    pub(crate) masked_ssim: [f64; 9],
    /// Masked edge artifact L4 per channel = 3 values
    pub(crate) masked_art_4th: [f64; 3],
    /// Masked edge detail_lost L4 per channel = 3 values
    pub(crate) masked_det_4th: [f64; 3],
    /// Masked MSE per channel = 3 values
    pub(crate) masked_mse: [f64; 3],
    /// Max SSIM error per channel = 3 values
    pub(crate) ssim_max: [f64; 3],
    /// Max edge artifact per channel = 3 values
    pub(crate) art_max: [f64; 3],
    /// Max edge detail_lost per channel = 3 values
    pub(crate) det_max: [f64; 3],
    /// L8 power pool SSIM error per channel = 3 values: (Σd⁸/N)^(1/8)
    pub(crate) ssim_p95: [f64; 3],
    /// L8 power pool edge artifact per channel = 3 values: (Σd⁸/N)^(1/8)
    pub(crate) art_p95: [f64; 3],
    /// L8 power pool edge detail_lost per channel = 3 values: (Σd⁸/N)^(1/8)
    pub(crate) det_p95: [f64; 3],
    // --- IW (information-content-weighted) features (only populated when
    //     compute_iw_features=true). Texture-EMPHASISING counterpart to
    //     the masked_* block. Wang & Li 2011 IW-SSIM. ---
    /// IW SSIM: [mean, 4th, 2nd] per channel = 9 values
    pub(crate) iw_ssim: [f64; 9],
    /// IW edge artifact L4 per channel = 3 values
    pub(crate) iw_art_4th: [f64; 3],
    /// IW edge detail_lost L4 per channel = 3 values
    pub(crate) iw_det_4th: [f64; 3],
    /// IW MSE per channel = 3 values
    pub(crate) iw_mse: [f64; 3],
    /// Mean IW **weight** per channel: `Σw / n`, where `w_i = 1 + k_iw · a_i`.
    ///
    /// NOT a feature — diagnostic only, and deliberately so. Every `iw_*` field
    /// above is pooled by `1/n` while a weighted mean is `Σ(w·v)/Σw`, so each
    /// carries a factor of `mean_w^p` (p = 1 for means, 0.5 for 2nd moments,
    /// 0.25 for 4th). Since the weights come from the REFERENCE only, `mean_w`
    /// is a per-reference constant: it cancels within an image and does not
    /// cancel across images, so it lands entirely on pooled (cross-image) rank.
    ///
    /// This field exists to MEASURE that factor against the weights we actually
    /// ship, rather than against `iw_pool.rs`'s different estimator. See
    /// `benchmarks/iw_pooling_normalization_2026-07-15.md`.
    // Read only by the `iw_mean_weight_spread_across_references` diagnostic
    // test (iw_pool.rs); non-test builds only WRITE it, under every feature
    // combination (the old `iw-diagnostics` gate no longer has a non-test
    // reader).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) iw_mean_w: [f64; 3],
}

/// Result from a zensim comparison.
///
/// Contains the final score, the raw distance used to derive it, and the
/// full per-scale feature vector (useful for diagnostics or weight training).
#[derive(Debug, Clone)]
pub struct ZensimResult {
    score: f64,
    raw_distance: f64,
    features: Vec<f64>,
    profile: crate::profile::ZensimProfile,
    mean_offset: [f64; 3],
    /// Set by the identity-image short-circuit in
    /// `compute_with_config_inner` / `compute_zensim_streaming`. When
    /// true, the post-network MLP scoring path in
    /// `apply_mlp_scoring_with_codec` MUST NOT overwrite the result —
    /// running the trained MLP on the all-zero feature vector
    /// produces garbage (the bake has no signal anchoring
    /// "zero feature vector → score 100").
    ///
    /// Replaces the pre-0.3.0 sentinel check
    /// `raw_distance == 0.0 && features.iter().all(== 0.0)` which was
    /// brittle (any future profile that legitimately produces
    /// raw_distance=0.0 + all-zero features on a real image would
    /// silently skip MLP scoring).
    is_identical: bool,
}

impl ZensimResult {
    /// Create a result from computed values. Internal use only.
    pub(crate) fn new(
        score: f64,
        raw_distance: f64,
        features: Vec<f64>,
        profile: crate::profile::ZensimProfile,
        mean_offset: [f64; 3],
    ) -> Self {
        Self {
            score,
            raw_distance,
            features,
            profile,
            mean_offset,
            is_identical: false,
        }
    }

    /// Mark this result as the byte-identical short-circuit output.
    /// `apply_mlp_scoring_with_codec` reads this flag to skip the
    /// MLP forward pass (which would overwrite the (100, 0, zeros)
    /// payload with garbage).
    #[inline]
    pub(crate) fn mark_identical(mut self) -> Self {
        self.is_identical = true;
        self
    }

    /// True iff this result was produced by the byte-identical
    /// short-circuit. Used by `apply_mlp_scoring_with_codec` to
    /// preserve the (score=100, raw_distance=0, zero-features)
    /// invariant regardless of which bake is loaded.
    #[inline]
    pub(crate) fn is_identical(&self) -> bool {
        self.is_identical
    }

    /// Set the profile on this result (builder pattern). Internal use only.
    pub(crate) fn with_profile(mut self, profile: crate::profile::ZensimProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Replace the raw distance and score with values from the MLP
    /// scoring path. Internal use only — called by
    /// [`apply_mlp_scoring`](crate::metric::apply_mlp_scoring).
    pub(crate) fn set_mlp_score(&mut self, raw_distance: f64, score: f64) {
        self.raw_distance = raw_distance;
        self.score = score;
    }

    /// Create a NaN sentinel result (for error/placeholder paths).
    ///
    /// Research / sibling-tooling helper. Used by zensim-validate's
    /// per-pair eval pipeline when a row's image can't be decoded —
    /// the NaN propagates downstream without crashing the sweep. Not
    /// part of the stable public surface.
    #[doc(hidden)]
    pub fn nan() -> Self {
        Self {
            score: f64::NAN,
            raw_distance: f64::NAN,
            features: vec![],
            // Placeholder tag on a NaN result — the canonical codec-target
            // profile (`B`); never the deprecated `A`.
            profile: crate::profile::ZensimProfile::codec_target(),
            mean_offset: [f64::NAN; 3],
            is_identical: false,
        }
    }

    /// Quality score on a 0–100 scale. 100 = identical, 0 = maximally different.
    /// Derived from `raw_distance` via a power-law mapping.
    pub fn score(&self) -> f64 {
        self.score
    }

    /// Raw weighted feature distance before nonlinear mapping. Lower = more similar.
    /// Not bounded to a fixed range; depends on image content and weights.
    pub fn raw_distance(&self) -> f64 {
        self.raw_distance
    }

    /// Per-scale raw features as a slice.
    ///
    /// Layout: 4 scales × 3 channels (X, Y, B) × 19 features per channel = 228.
    /// See [`FeatureView`] for named access.
    pub fn features(&self) -> &[f64] {
        &self.features
    }

    /// Consume the result and return the owned feature vector.
    pub fn into_features(self) -> Vec<f64> {
        self.features
    }

    /// Which profile produced this score.
    pub fn profile(&self) -> crate::profile::ZensimProfile {
        self.profile
    }

    /// Per-channel XYB mean offset: `mean(src_xyb[c]) - mean(dst_xyb[c])`.
    ///
    /// Captures global color/luminance shifts (CMS errors, white balance changes).
    /// Channels: `[X, Y, B]`, signed. Positive = distorted is darker/less saturated.
    pub fn mean_offset(&self) -> [f64; 3] {
        self.mean_offset
    }

    /// Convert the score to a dissimilarity value.
    ///
    /// Dissimilarity is `(100 - score) / 100`: 0 = identical, higher = worse.
    /// This is the inverse of the 0–100 score scale, normalized to 0–1.
    ///
    /// See also [`score_to_dissimilarity`] for the standalone conversion.
    pub fn dissimilarity(&self) -> f64 {
        score_to_dissimilarity(self.score)
    }

    /// Approximate SSIMULACRA2 score from the raw distance.
    ///
    /// Direct power-law fit: `100 - 19.04 × d^0.598`, calibrated on 344k
    /// synthetic pairs. MAE: 4.4 SSIM2 points, Pearson r = 0.974.
    ///
    /// More accurate than `mapping::zensim_to_ssim2(score)` (MAE 4.9, r = 0.932)
    /// because it skips the intermediate score mapping.
    pub fn approx_ssim2(&self) -> f64 {
        if self.raw_distance <= 0.0 {
            return 100.0;
        }
        (100.0 - 19.0379 * self.raw_distance.det_powf(0.5979, active_pow_form())).max(-100.0)
    }

    /// Approximate DSSIM value from the raw distance.
    ///
    /// Direct power-law fit: `0.000922 × d^1.224`, calibrated on 344k
    /// synthetic pairs. MAE: 0.00129, Pearson r = 0.952.
    ///
    /// Significantly more accurate than `mapping::zensim_to_dssim(score)`
    /// (MAE 0.00213, r = 0.719) because DSSIM's natural exponent (1.22)
    /// differs from the score mapping exponent (0.70).
    pub fn approx_dssim(&self) -> f64 {
        if self.raw_distance <= 0.0 {
            return 0.0;
        }
        0.000922 * self.raw_distance.det_powf(1.2244, active_pow_form())
    }

    /// Approximate butteraugli distance from the raw distance.
    ///
    /// Direct power-law fit: `2.365 × d^0.613`, calibrated on 344k
    /// synthetic pairs. MAE: 1.65 distance units, Pearson r = 0.713.
    ///
    /// Butteraugli's weak correlation with our features (r = 0.71) limits
    /// approximation accuracy regardless of mapping choice.
    pub fn approx_butteraugli(&self) -> f64 {
        if self.raw_distance <= 0.0 {
            return 0.0;
        }
        2.365353 * self.raw_distance.det_powf(0.6130, active_pow_form())
    }
}

/// Convert a zensim score (0–100, 100 = identical) to a dissimilarity value
/// (0 = identical, higher = worse).
///
/// Linear conversion: `(100 - score) / 100`.
///
/// | score | dissimilarity |
/// |-------|---------------|
/// | 100.0 | 0.0           |
/// | 99.5  | 0.005         |
/// | 95.0  | 0.05          |
/// | 50.0  | 0.5           |
/// | 0.0   | 1.0           |
pub fn score_to_dissimilarity(score: f64) -> f64 {
    (100.0 - score) / 100.0
}

/// Convert a dissimilarity value (0 = identical, higher = worse) back to a
/// zensim score (0–100, 100 = identical).
///
/// Inverse of [`score_to_dissimilarity`]: `score = 100 * (1 - dissimilarity)`.
pub fn dissimilarity_to_score(dissimilarity: f64) -> f64 {
    (100.0 * (1.0 - dissimilarity)).clamp(0.0, 100.0)
}

/// What kind of perceptual difference dominates between source and distorted.
///
/// Only categories with provably defensible statistical signatures are offered.
/// If no category can be identified with high confidence, `Unclassified` is returned.
#[cfg(feature = "classification")]
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Images are perceptually identical (score ≈ 100).
    Identical,
    /// Max delta ≤ N/255 — integer rounding, LUT precision, truncation.
    RoundingError,
    /// One channel zero-delta, others large — RGB↔BGR swap.
    ChannelSwap,
    /// Alpha compositing error (e.g. straight/premul confusion, wrong background).
    AlphaCompositing,
    /// Images differ but no category reached sufficient confidence.
    Unclassified,
}

/// Decomposed error classification for a source/distorted pair.
///
/// `dominant` is the category with the highest confidence (or `Identical`
/// if the overall score is ≈ 100).
#[cfg(feature = "classification")]
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ErrorClassification {
    /// The dominant error category.
    pub dominant: ErrorCategory,
    /// Overall confidence in the classification (0.0–1.0).
    pub confidence: f64,
    /// Rounding bias analysis (only populated when `dominant == RoundingError`).
    ///
    /// Measures how balanced the rounding errors are across positive and negative
    /// directions. `None` when not a rounding error or insufficient data.
    pub rounding_bias: Option<RoundingBias>,
}

/// Analysis of whether rounding errors are balanced (+/-) or systematic.
///
/// A balanced distribution (roughly equal +1 and -1 counts) indicates normal
/// rounding mode differences — nothing to worry about. A heavily skewed
/// distribution (mostly one direction) suggests systematic truncation or
/// a floor/ceil bias that may indicate a pipeline bug.
#[cfg(feature = "classification")]
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct RoundingBias {
    /// Per-channel ratio of positive-to-total differing pixels.
    ///
    /// 0.5 = perfectly balanced, 0.0 = all negative, 1.0 = all positive.
    /// Channels: `[R, G, B]`.
    pub positive_fraction: [f64; 3],
    /// Whether the rounding appears balanced (within statistical norms).
    ///
    /// `true` means the +/- distribution is consistent with unbiased rounding
    /// and is likely nothing to worry about. `false` means systematic bias
    /// was detected (e.g., all errors in one direction = truncation).
    pub balanced: bool,
}

/// Pixel-level delta analysis for error classification.
///
/// All deltas are `src - dst` (positive = distorted is darker/lower).
/// Values normalized to [0.0, 1.0] regardless of input bit depth.
#[cfg(feature = "classification")]
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct DeltaStats {
    // --- Per-channel [R, G, B] summary stats ---
    /// Mean delta (signed). Positive = dst darker.
    pub mean_delta: [f64; 3],
    /// Standard deviation of delta.
    pub stddev_delta: [f64; 3],
    /// Maximum |delta|.
    pub max_abs_delta: [f64; 3],

    // --- Signed small-delta histogram ---
    /// Per-channel pixel counts for signed deltas -3 to +3 (in 1/native_max units).
    ///
    /// Index mapping: `[0]`=−3, `[1]`=−2, `[2]`=−1, `[3]`=0, `[4]`=+1, `[5]`=+2, `[6]`=+3.
    /// Delta convention: `src - dst`, so +1 means dst is 1 LSB lower than src.
    /// Only counts pixels whose per-channel delta falls in \[−3, +3\]; pixels
    /// outside this range are not tracked here.
    pub signed_small_histogram: [[u64; 7]; 3],

    /// Maximum representable value for the native pixel format.
    ///
    /// 255.0 for u8 formats, 65535.0 for u16, 1.0 for f32/f16.
    /// Used to interpret delta magnitudes at native precision.
    pub native_max: f64,

    // --- Pixel counts ---
    /// Total pixels compared.
    pub pixel_count: u64,
    /// Pixels where any channel differs.
    pub pixels_differing: u64,
    /// Pixels where any channel |delta| > 1/255.
    pub pixels_differing_by_more_than_1: u64,

    // --- Alpha channel ---
    /// Whether the input format has an alpha channel.
    pub has_alpha: bool,
    /// Max |src_alpha - dst_alpha| in 0-255 units. 0 for RGB-only formats.
    pub alpha_max_delta: u8,
    /// Pixels where alpha differs at all. 0 for RGB-only formats.
    pub alpha_pixels_differing: u64,

    // --- Per-channel value histograms (256 bins, quantized to 8-bit) ---
    /// Source image histogram. `[channel][value]`. R=0, G=1, B=2, A=3.
    pub src_histogram: [[u64; 256]; 4],
    /// Distorted image histogram. `[channel][value]`. R=0, G=1, B=2, A=3.
    pub dst_histogram: [[u64; 256]; 4],

    // --- Alpha-stratified stats (only for RGBA/BGRA inputs) ---
    /// Delta stats for fully opaque pixels (A = max).
    pub opaque_stats: Option<AlphaStratifiedStats>,
    /// Delta stats for semitransparent pixels (0 < A < max).
    pub semitransparent_stats: Option<AlphaStratifiedStats>,
    /// Pearson correlation between |delta| and (1 - alpha).
    /// High (> 0.8) = compositing/premul error. None if no alpha.
    pub alpha_error_correlation: Option<f64>,
}

/// Stats for a subset of pixels grouped by alpha.
#[cfg(feature = "classification")]
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct AlphaStratifiedStats {
    /// Number of pixels in this stratum.
    pub pixel_count: u64,
    /// Mean |delta| per channel in this alpha stratum.
    pub mean_abs_delta: [f64; 3],
    /// Max |delta| per channel.
    pub max_abs_delta: [f64; 3],
}

/// Result from `classify()`: the zensim score plus delta analysis and error classification.
#[cfg(feature = "classification")]
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ClassifiedResult {
    /// The standard zensim result (score, features, etc.).
    pub result: ZensimResult,
    /// Error classification with per-category confidence scores.
    pub classification: ErrorClassification,
    /// Pixel-level delta statistics.
    pub delta_stats: DeltaStats,
}

// --- Zensim config struct (primary API) ---

use crate::profile::{ProfileParams, ZensimProfile};
use crate::source::ImageSource;

mod bake;
pub use bake::BakeScorer;
#[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
pub use bake::SteeringSession;

/// Metric configuration. Methods on this struct are the primary API.
///
/// ```no_run
/// use zensim::{Zensim, ZensimProfile, RgbSlice};
/// # let (src, dst) = (vec![[0u8; 3]; 64], vec![[0u8; 3]; 64]);
/// let z = Zensim::new(ZensimProfile::codec_target());
/// let source = RgbSlice::new(&src, 8, 8);
/// let distorted = RgbSlice::new(&dst, 8, 8);
/// let result = z.compute(&source, &distorted).unwrap();
/// println!("{}: {:.2}", result.profile(), result.score());
/// ```
#[derive(Clone, Debug)]
pub struct Zensim {
    profile: ZensimProfile,
    parallel: bool,
    /// Optional cap on total pixels (`width * height`) per image. `None`
    /// disables the cap. Allocation grows roughly `~14 × pixels × 4 B` for
    /// the streaming pipeline, so callers feeding untrusted dimensions
    /// should set this. See [`Zensim::with_max_pixels`].
    max_pixels: Option<usize>,
    /// Optional cooperative-cancellation token, checked at row-band and
    /// scale boundaries inside the streaming walk. See [`Zensim::with_stop`].
    stop: Option<StopHandle>,
    /// Route the scoring entries through the streaming fold instead of the
    /// buffered walk (fold-engine lane, 2026-08-30). `false` for every
    /// profile except [`crate::profile::ZensimProfile::D`], which
    /// `Zensim::new` sets it `true` for by construction (see `fast_by_default`
    /// there) — or wherever something explicitly asks via
    /// `Zensim::with_engine(ScoringEngine::Fold)`. Both routes only exist
    /// under `feature-regime-v2` (default-on since 2026-09-01;
    /// `--no-default-features` removes the field's only readers, so it stays
    /// permanently `false`-equivalent there). Requests the fold cannot serve
    /// bit-identically (`fold_engine::is_fold_backable`) fall back to
    /// buffered even when set.
    pub(crate) fold_engine: bool,
    /// Let a fold-backed SCORE skip computing the v1 pool families the loaded
    /// profile structurally never reads (fold_engine::score_pool_mode).
    /// **Default `false`** for every profile except `D` (same
    /// `fast_by_default` wiring as `fold_engine` above) — opt-in elsewhere via
    /// [`Zensim::with_unread_feature_skipping`], because it leaves the
    /// skipped slots at `0.0` in [`ZensimResult::features`]. The SCORE and
    /// `raw_distance` are bit-identical either way, by construction: a slot
    /// is only skipped when every consumer's weight on it is exactly zero.
    /// Read only by the `feature-regime-v2`-gated fold engine (default-on
    /// since 2026-09-01; inert dead field under `--no-default-features`).
    #[cfg_attr(not(feature = "feature-regime-v2"), allow(dead_code))]
    pub(crate) skip_unread_pools: bool,
    /// An optional companion CORRUPTION HEAD
    /// (`docs/PLAN_CORRHEAD_SERVING_2026-09-06.md`). `None` by default and on
    /// every existing path: `compute` does not read it, so attaching one
    /// changes no score any current caller can observe. The gate is asked for
    /// explicitly, through [`Zensim::corruption_verdict`].
    ///
    /// `Arc` so `Zensim` stays `Clone` without copying a ~200 KB ensemble.
    #[cfg(feature = "corruption-head")]
    corruption_head: Option<std::sync::Arc<crate::corruption_head::CorruptionHead>>,
}

/// Shared cancellation token stored on [`Zensim`]. Wraps the caller's
/// [`enough::Stop`] in an `Arc` so `Zensim` stays `Clone` (clones share
/// the same token). Debug output elides the token (trait objects have no
/// useful Debug form).
#[derive(Clone)]
struct StopHandle(std::sync::Arc<dyn enough::Stop>);

impl core::fmt::Debug for StopHandle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("StopHandle(..)")
    }
}

/// Whether a computation that asked for internal parallel work runs it:
/// `requested`, and the current rayon pool has more than one thread. With a
/// single thread each parallel section would only hand its work to the
/// pool's one worker and block until it returns (most of a small image's
/// time); the sequential path computes the same bits.
#[inline]
pub(crate) fn effective_parallel(requested: bool) -> bool {
    #[cfg(feature = "threads")]
    {
        requested && rayon::current_num_threads() > 1
    }
    #[cfg(not(feature = "threads"))]
    {
        let _ = requested;
        false
    }
}

impl Zensim {
    /// Create a new `Zensim` with the given profile. Parallel by default.
    ///
    /// A default `max_pixels` cap of **120 million** (≈ 120 MP — large enough
    /// for common ~108 MP camera photos) is applied so that forwarding
    /// untrusted dimensions can't drive an unbounded allocation (memory is
    /// ~14×w×h×4 B). Call [`Zensim::with_max_pixels`] to tighten it, or
    /// `with_max_pixels(usize::MAX)` to opt out for trusted input.
    pub fn new(profile: ZensimProfile) -> Self {
        // `ZensimProfile::D`'s whole reason to exist is speed: its bake
        // structurally reads none of the `f156..372` pool block
        // (`Plan::for_bake` / `fold_engine::score_pool_mode`
        // both derive `V1PoolsMode::Peaks` for it), so — unlike every other
        // profile, which defaults to the buffered walk and leaves these
        // `#[doc(hidden)]` knobs to an explicit opt-in — `D` opts itself in.
        // `feature-regime-v2` (the fold engine's gate) is default-on since
        // 2026-09-01 (`benchmarks/profile_d_notax_2026-09-01.md`), so a plain
        // `cargo add zensim` build reaches the fast fold path for `D` with no
        // extra feature flag. On a build with `--no-default-features` (or an
        // explicit re-add list that omits `feature-regime-v2`) these two
        // fields are inert (nothing reads them outside that feature), so `D`
        // still computes correctly via the ordinary buffered walk, just at
        // `B`-class cost rather than the fold's `156`-class speedup. See the
        // `D` variant's rustdoc and
        // `benchmarks/profile_d_and_published_speed_2026-09-01.md` /
        // `benchmarks/profile_d_notax_2026-09-01.md` for the measured
        // buffered-vs-fold and gated-vs-default-build numbers.
        #[cfg(feature = "candidate-profiles")]
        let fast_by_default = matches!(profile, ZensimProfile::D);
        #[cfg(not(feature = "candidate-profiles"))]
        let fast_by_default = false;
        Self {
            profile,
            parallel: true,
            max_pixels: Some(120_000_000),
            stop: None,
            fold_engine: fast_by_default,
            skip_unread_pools: fast_by_default,
            #[cfg(feature = "corruption-head")]
            corruption_head: None,
        }
    }

    /// Attach a companion CORRUPTION HEAD.
    ///
    /// `#[doc(hidden)]` + feature-gated, and **PROPOSED, not approved** — the
    /// exact signatures are registered for the user's sign-off in
    /// `docs/PLAN_CORRHEAD_SERVING_2026-09-06.md` section 3.1. Until then this
    /// is not a supported surface.
    ///
    /// Attaching a head is deliberately NOT the same as turning the gate on:
    /// [`Self::compute`] does not read the head, so no existing caller can
    /// observe a different score. The gate is asked for explicitly, by
    /// [`Self::corruption_verdict`]. That split is what lets the head ship
    /// before the composition is a product decision.
    ///
    /// # Errors
    ///
    /// [`CorruptionHeadError::NotServable`] when the head reads a feature slot
    /// this profile's extraction plan does not populate. Attaching a head must
    /// never widen the walk, so this is a refusal rather than a silent
    /// upgrade — MEASURED for the shipped profiles by
    /// `corruption_head::servability_tests`.
    #[cfg(feature = "corruption-head")]
    #[doc(hidden)]
    pub fn with_corruption_head(
        mut self,
        head: crate::corruption_head::CorruptionHead,
    ) -> Result<Self, crate::corruption_head::CorruptionHeadError> {
        head.check_servable_for_profile(self.profile)?;
        self.corruption_head = Some(std::sync::Arc::new(head));
        Ok(self)
    }

    /// The attached head, if any.
    #[cfg(feature = "corruption-head")]
    #[doc(hidden)]
    pub fn corruption_head(&self) -> Option<&crate::corruption_head::CorruptionHead> {
        self.corruption_head.as_deref()
    }

    /// Apply the attached head to a result THIS `Zensim` produced.
    ///
    /// `None` when no head is attached — never a silently ungated score
    /// dressed up as a gated one.
    ///
    /// The composition is
    /// [`corruption_head::gate_score`](crate::corruption_head::gate_score),
    /// the same function `bake_verdict`'s DEPLOY section calls, so a served
    /// score and a verdict cannot drift.
    ///
    /// # Errors
    ///
    /// [`CorruptionHeadError::FeatureLenMismatch`] when the result's feature
    /// vector is not the width the head declares. Every shipped profile emits
    /// 372 (gated by `corruption_head::probe_width`), so this fires only for a
    /// head baked against a different layout.
    #[cfg(feature = "corruption-head")]
    #[doc(hidden)]
    pub fn corruption_verdict(
        &self,
        result: &ZensimResult,
    ) -> Option<
        Result<
            crate::corruption_head::CorruptionVerdict,
            crate::corruption_head::CorruptionHeadError,
        >,
    > {
        let head = self.corruption_head.as_deref()?;
        Some(head.verdict(result.features(), result.score()))
    }

    /// Select the walk the scoring entries run. See
    /// [`ScoringEngine`](crate::fold_engine::ScoringEngine).
    ///
    /// `#[doc(hidden)]` + `feature-regime-v2`-gated: an internal selector for
    /// the parity gates and the perf comparison, not a product knob. The
    /// default is unchanged (`Buffered`), and a request the fold cannot serve
    /// bit-identically falls back to buffered rather than scoring a different
    /// quantity.
    #[cfg(feature = "feature-regime-v2")]
    #[doc(hidden)]
    pub fn with_engine(mut self, engine: crate::fold_engine::ScoringEngine) -> Self {
        self.fold_engine = matches!(engine, crate::fold_engine::ScoringEngine::Fold);
        self
    }

    /// Let a fold-backed score SKIP computing v1 pool families this
    /// profile's scoring consumers structurally never read.
    ///
    /// `#[doc(hidden)]` + `feature-regime-v2`-gated, and **off by default**,
    /// for one reason: it is not score-visible but it IS feature-visible. A
    /// family is only skipped when every layer-0 weight on it is exactly
    /// zero across every bake the profile forwards, so `score()` and
    /// `raw_distance()` are bit-identical with it on or off — but the
    /// skipped slots come back as `0.0` from
    /// [`ZensimResult::features`](crate::ZensimResult::features), so a
    /// caller reading the vector must opt in knowingly. This is the same
    /// distinction `fold_engine::is_fold_backable` already draws for
    /// `streaming::active_channels`' channel skipping.
    ///
    /// Has no effect on the buffered engine, on
    /// [`Self::compute_extended_features`] (which exists to return the
    /// vector), or on any profile whose bakes read the pool block.
    #[cfg(feature = "feature-regime-v2")]
    #[doc(hidden)]
    pub fn with_unread_feature_skipping(mut self, on: bool) -> Self {
        self.skip_unread_pools = on;
        self
    }

    /// The pool mode a fold-backed score would run for this instance — the
    /// read-back of [`Self::with_unread_feature_skipping`], and the hook the
    /// perf harness prices.
    #[cfg(feature = "feature-regime-v2")]
    #[doc(hidden)]
    pub fn score_pool_mode(&self) -> crate::feature_v2::V1PoolsMode {
        let params = self.profile.params();
        let config = config_from_params(params, self.parallel_now());
        self.scoring_plan(params, &config).map_or_else(
            || crate::fold_engine::score_pool_mode(params, &config, self.skip_unread_pools),
            |plan| plan.compute.v1_pools,
        )
    }

    /// The serving PLAN for a SCORING entry: `None` (today's unconditional v1
    /// walk) unless the extraction must differ — because this instance opted
    /// into pool skipping, or because a bake declares a LAYOUT wider than the
    /// v1 walk emits.
    ///
    /// The second case is what makes a wide bake servable at all. Before it,
    /// a 944-declared bake reached `prep_bake_input_f32` with a 372-wide
    /// vector and was refused with `"bake declares more input features than
    /// the caller supplied"` — a true statement about a situation the runtime
    /// created by never asking for the width the bake declared.
    fn scoring_plan(
        &self,
        params: &ProfileParams,
        config: &ZensimConfig,
    ) -> Option<crate::feature_plan::Plan> {
        crate::fold_engine::score_plan(
            params,
            config,
            self.skip_unread_pools
                || config.revision() >= crate::feature_defs::FormulaRevision::Rev5,
        )
    }

    /// The walk this instance's scoring entries run — the read-back of
    /// [`Self::with_engine`].
    #[cfg(feature = "feature-regime-v2")]
    #[doc(hidden)]
    pub fn engine(&self) -> crate::fold_engine::ScoringEngine {
        if self.fold_engine
            || config_from_params(self.profile.params(), self.parallel_now()).revision()
                >= crate::feature_defs::FormulaRevision::Rev5
        {
            crate::fold_engine::ScoringEngine::Fold
        } else {
            crate::fold_engine::ScoringEngine::Buffered
        }
    }

    /// Enable or disable multi-threaded computation (rayon).
    /// Default: `true`.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Set a cap on total pixels (`width * height`) accepted by every
    /// `compute*` entry point. Images exceeding the cap are rejected with
    /// [`ZensimError::ImageTooLarge`] before any allocation runs.
    ///
    /// Default: **120 MP** (set by [`Zensim::new`]). Zensim allocates
    /// roughly `width × height × 4 bytes × ~14` for the streaming XYB
    /// pyramid plus destination planes, so a 4K (≈8.3 MP) image needs
    /// ~470 MB of scratch. A cap of e.g. 64 MP (8192×8192) keeps the
    /// worst case at ~3.6 GB.
    ///
    /// Pass [`usize::MAX`] explicitly to disable the cap on a previously
    /// configured `Zensim`.
    pub fn with_max_pixels(mut self, max_pixels: usize) -> Self {
        self.max_pixels = Some(max_pixels);
        self
    }

    /// Install a cooperative-cancellation token, checked at row-band and
    /// scale boundaries inside the streaming compute walk. When the token
    /// fires, the walk abandons remaining work promptly and the entry
    /// point returns [`ZensimError::Cancelled`].
    ///
    /// Accepts any [`enough::Stop`] implementation — typically a
    /// `Stopper` from the sibling
    /// [`almost-enough`](https://docs.rs/almost-enough) crate, cancelled
    /// from another thread (or wired to a timeout). Passing
    /// [`Unstoppable`](enough::Unstoppable) (or any token whose
    /// [`may_stop`](enough::Stop::may_stop) is `false`) stores nothing and
    /// keeps the zero-check fast path.
    ///
    /// Coverage: the scoring entry points (`compute`,
    /// `compute_with_codec_hint`, `compute_with_ref`, `compute_pu_linear`,
    /// `compute_pu_linear_planar`) and the diffmap entries
    /// (`compute_with_ref_and_diffmap`,
    /// `compute_with_ref_and_diffmap_linear_planar`). The v2
    /// feature-extraction walks and the caller-paced strip APIs
    /// (`compute_streaming_strips*`, where the caller already controls
    /// cadence between strips) do not check the token yet.
    ///
    /// At Rev5, canonical fold scoring checks the token between strips as
    /// well as before and after the fold. Earlier buffered paths retain
    /// the following band-level cancellation granularity.
    /// Cancellation granularity is one row band (a few dozen rows) at the
    /// current scale; a fired token is detected within roughly one band's
    /// worth of work per rayon worker.
    ///
    /// Below Rev5 a live token therefore keeps every scoring entry on the
    /// buffered walk. A profile whose bake reads past that walk's width
    /// (`C`, `CHdr`) is served through the fold only without a token; with
    /// one, `compute`, `compute_pu_linear` and `compute_pu_linear_planar` all
    /// refuse it with the same error, so one contract covers every entry.
    ///
    /// ```no_run
    /// use zensim::{Zensim, ZensimProfile};
    /// # struct S(std::sync::Arc<std::sync::atomic::AtomicBool>);
    /// # impl enough::Stop for S {
    /// #     fn check(&self) -> Result<(), enough::StopReason> {
    /// #         if self.0.load(std::sync::atomic::Ordering::Relaxed) {
    /// #             Err(enough::StopReason::Cancelled)
    /// #         } else { Ok(()) }
    /// #     }
    /// # }
    /// # let token = S(Default::default());
    /// let z = Zensim::new(ZensimProfile::codec_target()).with_stop(token);
    /// // z.compute(..) now returns Err(ZensimError::Cancelled { .. })
    /// // soon after the token fires.
    /// ```
    pub fn with_stop(mut self, stop: impl enough::Stop + 'static) -> Self {
        // may_stop() == false (e.g. Unstoppable) can never fire — skip
        // storing it so every check stays on the None fast path.
        self.stop = if stop.may_stop() {
            Some(StopHandle(std::sync::Arc::new(stop)))
        } else {
            None
        };
        self
    }

    /// Borrow the installed stop token for threading into the walkers.
    pub(crate) fn stop_ref(&self) -> Option<&dyn enough::Stop> {
        self.stop.as_ref().map(|h| &*h.0 as &dyn enough::Stop)
    }

    /// Return `Err(ZensimError::Cancelled)` if the installed token has
    /// fired. Entry points call this up front (cheap pre-flight) and after
    /// the walk returns (converting an abandoned walk into the error).
    pub(crate) fn check_stop(&self) -> Result<(), ZensimError> {
        match &self.stop {
            Some(h) => h.0.check().map_err(ZensimError::from),
            None => Ok(()),
        }
    }

    /// Current `max_pixels` cap, if any.
    pub fn max_pixels(&self) -> Option<usize> {
        self.max_pixels
    }

    /// Current profile.
    pub fn profile(&self) -> ZensimProfile {
        self.profile
    }

    /// Whether multi-threaded computation is enabled.
    pub fn parallel(&self) -> bool {
        self.parallel
    }

    /// Whether this call runs its internal parallel work: [`Self::parallel`]
    /// and a rayon pool of more than one thread ([`effective_parallel`]).
    #[inline]
    fn parallel_now(&self) -> bool {
        effective_parallel(self.parallel)
    }

    /// Compare source and distorted images.
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError`] if dimensions are mismatched or too small.
    pub fn compute(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
    ) -> Result<ZensimResult, ZensimError> {
        self.compute_with_codec_hint(source, distorted, None)
    }

    /// Compare source and distorted images, supplying an optional
    /// codec hint that drives **per-codec post-spline affine**
    /// calibration (EXP-CROSS-CODEC-V11-E, 2026-05-20).
    ///
    /// When the loaded profile's bake carries
    /// `zentrain.per_codec_calibration` metadata AND `codec_hint`
    /// names a registered codec, the runtime applies an extra
    /// `score = α_c + β_c · spline(raw)` affine after the standard
    /// scoring path. The affine is monotone within codec
    /// (β > 0 by construction) so within-codec rank ordering is
    /// bit-exact preserved; only the cross-codec systematic bias
    /// is pulled toward consensus at JND landmarks.
    ///
    /// Accepted codec hint strings (case-insensitive):
    /// `"jpeg"` / `"jpg"` / `"zenjpeg"` / `"mozjpeg"` / `"libjpeg"`,
    /// `"webp"` / `"zenwebp"`,
    /// `"avif"` / `"zenavif"`,
    /// `"jxl"` / `"zenjxl"` / `"jpegxl"`,
    /// `"png"` / `"zenpng"`. Any other hint is treated as unknown
    /// (no affine applied — identical to calling
    /// [`Self::compute`]).
    ///
    /// Profiles without the metadata silently ignore the hint
    /// (output is identical to [`Self::compute`]).
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError`] if dimensions are mismatched or too small.
    pub fn compute_with_codec_hint(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
        codec_hint: Option<&str>,
    ) -> Result<ZensimResult, ZensimError> {
        // Descriptor-declared HDR (`ImageSource::is_hdr` — the zenpixels
        // transfer signal, never pixel-value sniffing) routes to the
        // PU-linear front-end + the profile's HDR weights (issue #38: the
        // centralised guard now dispatches instead of erroring). The codec
        // hint does not apply on this path (per-codec affines are
        // SDR-calibrated).
        if source.is_hdr() || distorted.is_hdr() {
            if source.is_hdr() != distorted.is_hdr() {
                return Err(ZensimError::HdrInputRequiresPuPath);
            }
            if source.width() != distorted.width() || source.height() != distorted.height() {
                return Err(ZensimError::DimensionMismatch);
            }
            let r = nits_rgb_from_hdr_source(source)?;
            let d = nits_rgb_from_hdr_source(distorted)?;
            let (w, h) = (source.width(), source.height());
            return self.compute_pu_linear(&r, &d, w, h, w * 3, w * 3);
        }
        let params = self.profile.params();
        validate_pair(source, distorted)?;
        check_within_max_pixels(source.width(), source.height(), self.max_pixels)?;
        self.check_stop()?;
        let config = config_from_params(params, self.parallel_now());
        check_profile_route(params, &config)?;
        let mut result = compute_with_config_inner(
            source,
            distorted,
            &config,
            params.weights,
            self.stop_ref(),
            self.fold_engine,
            self.scoring_plan(params, &config).as_ref(),
        );
        self.check_stop()?;
        apply_mlp_scoring_with_codec(
            &mut result,
            params,
            source.width() as u32,
            source.height() as u32,
            codec_hint,
        )?;
        Ok(result.with_profile(self.profile))
    }

    /// Compare source and distorted images, returning the **extended** feature
    /// set (300 features at the default 4-scale, 3-channel layout) instead of
    /// the standard 228 weighted-only set.
    ///
    /// The extended set adds 6 masked features (SSIM/edge/MSE weighted by
    /// source flatness) per channel per scale on top of the 228 features
    /// returned by [`Zensim::compute`]. The score is identical to `compute()`
    /// — only the first 228 features have non-zero weights — but the extra 72
    /// features are useful inputs for retraining downstream models (selectors,
    /// regressors) without re-running the costly multi-scale stats pass.
    ///
    /// Layout (default profile, 4 scales × 3 channels):
    /// - 0..156   — basic features (13/channel/scale): SSIM/edge/HF errors
    /// - 156..228 — peak features (6/channel/scale): SSIM/edge max + p95
    /// - 228..300 — masked features (6/channel/scale): SSIM/edge/MSE flatness-weighted
    ///
    /// Use [`FeatureView`] for named access. Cost overhead vs `compute()` is
    /// modest — the masking pass reuses the already-computed flatness map.
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError`] if dimensions are mismatched or too small.
    /// Rev5 refuses this masked-family request; use `compute_all_features`
    /// (with `training`) or a supported research slot request instead.
    pub fn compute_extended_features(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
    ) -> Result<ZensimResult, ZensimError> {
        if source.is_hdr() || distorted.is_hdr() {
            if source.is_hdr() != distorted.is_hdr() {
                return Err(ZensimError::HdrInputRequiresPuPath);
            }
            let r = nits_rgb_from_hdr_source(source)?;
            let d = nits_rgb_from_hdr_source(distorted)?;
            let (w, h) = (source.width(), source.height());
            return self.compute_pu_linear_extended_features(&r, &d, w, h, w * 3, w * 3);
        }
        let params = self.profile.params();
        validate_pair(source, distorted)?;
        check_within_max_pixels(source.width(), source.height(), self.max_pixels)?;
        let mut config = config_from_params(params, self.parallel_now());
        check_profile_route(params, &config)?;
        config.extended_features = true;
        if config.revision() >= crate::feature_defs::FormulaRevision::Rev5 {
            return Err(ZensimError::ModelLoadFailed {
                reason: "formula revision 5 supports only basic + peaks + v2; extended extraction requests masked features",
            });
        }
        let result = compute_with_config_inner(
            source,
            distorted,
            &config,
            params.weights,
            self.stop_ref(),
            self.fold_engine,
            None,
        );
        self.check_stop()?;
        Ok(result.with_profile(self.profile))
    }

    /// Compute the v2 "bounded" feature vector (`feature-regime-v2`,
    /// iteration 1) for a source/distorted pair.
    ///
    /// This is a SEPARATE, strictly additive extraction regime — see
    /// [`crate::feature_v2`]'s module doc and
    /// `docs/FEATURE_V2_SPEC_2026-07-18.md` for the defect inventory (D1-D9)
    /// it fixes, the per-feature formula/bound/citation table, and the
    /// as-built layout. It does not read from or write to any v1 code path
    /// (`compute`, `compute_extended_features`, `compute_with_ref`, ...);
    /// v1's byte-stable 372-feature output is unaffected whether this
    /// method exists in the build or not.
    ///
    /// Iteration 1 is a scalar correctness-first reference implementation —
    /// not yet fused into the archmage/SIMD streaming hot path, so it is
    /// slower than v1's `compute*` methods. There is no v2 scoring/profile
    /// yet (no bake consumes these features); this method returns raw,
    /// bounded, spatializable features for validation and future trainer
    /// experiments.
    ///
    /// # Errors
    ///
    /// Same error conditions as [`Self::compute_extended_features`]:
    /// [`ZensimError::DimensionMismatch`], [`ZensimError::ImageTooSmall`]
    /// (zero-dimension only — sub-64px reflect-pads), and
    /// [`ZensimError::ImageTooLarge`] if `max_pixels` is exceeded.
    #[cfg(feature = "feature-regime-v2")]
    pub fn compute_v2_features(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
    ) -> Result<crate::feature_v2::ZensimV2Result, ZensimError> {
        crate::feature_v2::compute_v2_features_impl(
            source,
            distorted,
            self.max_pixels,
            self.parallel_now(),
        )
    }

    /// Per-pixel diffmap for the v2 feature block: the contribution of an
    /// append-only bake's v2 weights to its scalar, at full resolution. This
    /// is the steering companion to [`Self::compute_v2_features`] — folding
    /// each v2 family's per-pixel error map, weighted by the bake's gradient
    /// `s_v2` w.r.t. that feature, so an encoder closed loop can read *where*
    /// to refine for the v2 part of the score. The v1 (≤372) contribution
    /// comes from the ordinary [`crate::DiffmapWeighting::ModelSensitivity`]
    /// path; summing the two gives the deployed map for a v1++v2 bake.
    ///
    /// `s_v2` is laid out exactly as the v2 feature view indexes it —
    /// `s_v2[scale*3*29 + ch*29 + local]`, length `n_scales*3*29` — i.e. the
    /// `s[372..]` tail of a frozen-v1-372 ++ v2 bake's full gradient.
    ///
    /// The map spatializes the ADDITIVE families exactly (simple-mean +
    /// masked/iw weighted-pools, block-pool-identity tested); the genuinely
    /// non-additive families (`dev2/dev4`, `fragility`, `edge_width`) carry no
    /// weight, matching their exclusion from an exact per-pixel fold.
    ///
    /// # Errors
    ///
    /// [`ZensimError::ImageTooSmall`], [`ZensimError::HdrInputRequiresPuPath`],
    /// or [`ZensimError::ImageTooLarge`] — same as the feature path.
    #[cfg(feature = "feature-regime-v2")]
    pub fn compute_v2_diffmap(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
        s_v2: &[f64],
    ) -> Result<Vec<f32>, ZensimError> {
        crate::feature_v2::compute_v2_diffmap_full(
            source,
            distorted,
            s_v2,
            self.max_pixels,
            self.parallel_now(),
        )
    }

    /// Like [`Self::compute_v2_features`], but with explicit control over
    /// which phase-2 new-feature GROUPS (`crate::feature_v2::
    /// V2NewFeatureToggles`) are computed. Used for per-group marginal-cost
    /// measurement (`docs/FEATURE_V2_SPEC_2026-07-18.md` §A.12) and to keep
    /// a group that fails the speed gate available (default OFF) without
    /// deleting its implementation. `compute_v2_features` is equivalent to
    /// calling this with `V2NewFeatureToggles::default()` (all groups on).
    #[cfg(feature = "feature-regime-v2")]
    pub fn compute_v2_features_with_toggles(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
        toggles: crate::feature_v2::V2NewFeatureToggles,
    ) -> Result<crate::feature_v2::ZensimV2Result, ZensimError> {
        // REV4SERVE: only a same-revision Rev4 process serves a Rev4 request.
        crate::ssim_form::refuse_rev4_mix(toggles.formula_revision)?;
        crate::feature_v2::compute_v2_features_impl_with_toggles(
            source,
            distorted,
            self.max_pixels,
            self.parallel_now(),
            toggles,
        )
    }

    /// Precompute the v2 reference-side state (reflect-padded XYB planes at
    /// every pyramid scale) for scoring MANY distorted variants against one
    /// reference. Sweep/extraction workloads pair each reference with dozens
    /// of distorted images; preparing once amortizes the reference-side
    /// convert + pyramid work to ~zero per pair.
    ///
    /// Features from [`Self::compute_v2_features_with_ref`] are
    /// **bit-identical** to [`Self::compute_v2_features`] on the same pair —
    /// the prepared planes are built by the same functions in the same
    /// order (test-gated).
    ///
    /// # Errors
    ///
    /// [`ZensimError::ImageTooSmall`] (zero-dimension),
    /// [`ZensimError::HdrInputRequiresPuPath`] (HDR-flagged source), or
    /// [`ZensimError::ImageTooLarge`] (`max_pixels` exceeded).
    #[cfg(feature = "feature-regime-v2")]
    pub fn prepare_v2_reference(
        &self,
        source: &impl ImageSource,
    ) -> Result<crate::feature_v2::V2PreparedReference, ZensimError> {
        crate::feature_v2::prepare_v2_reference_impl(
            source,
            self.max_pixels,
            self.parallel_now(),
            false,
        )
    }

    /// Like [`Self::prepare_v2_reference`], but ALSO caches the
    /// reference-side blur moments (`mu1` + activity/masking planes) at
    /// every scale, so each subsequent
    /// [`Self::compute_v2_features_with_ref`] call skips the reference
    /// V-blur and the whole activity chain per channel-scale. Costs ~2x
    /// the prepare time and ~2x the prepared memory (still amortized to
    /// ~zero across a sweep's variants); features remain bit-identical —
    /// the cache is filled by replaying the exact per-strip kernel walk.
    ///
    /// Serves the plain-v2 (`V2Bounded`) research path ONLY: the
    /// folded-720[+append] extraction is streaming-only since the C5
    /// switchover (2026-07-26) and takes no prepared reference.
    ///
    /// # Errors
    ///
    /// Same as [`Self::prepare_v2_reference`].
    #[cfg(feature = "feature-regime-v2")]
    pub fn prepare_v2_reference_with_moments(
        &self,
        source: &impl ImageSource,
    ) -> Result<crate::feature_v2::V2PreparedReference, ZensimError> {
        crate::feature_v2::prepare_v2_reference_impl(
            source,
            self.max_pixels,
            self.parallel_now(),
            true,
        )
    }

    /// Compute v2 features for one distorted image against a prepared
    /// reference (see [`Self::prepare_v2_reference`]). Convenience form
    /// that allocates its scratch internally and uses default
    /// [`crate::feature_v2::V2NewFeatureToggles`]; batch drivers should
    /// prefer [`Self::compute_v2_features_with_ref_and_scratch`] with one
    /// [`crate::feature_v2::V2Scratch`] per worker thread to also
    /// eliminate per-pair scratch allocation.
    ///
    /// # Errors
    ///
    /// [`ZensimError::DimensionMismatch`] if `distorted` does not match the
    /// prepared reference's dimensions, plus the error conditions of
    /// [`Self::compute_v2_features`].
    #[cfg(feature = "feature-regime-v2")]
    pub fn compute_v2_features_with_ref(
        &self,
        reference: &crate::feature_v2::V2PreparedReference,
        distorted: &impl ImageSource,
    ) -> Result<crate::feature_v2::ZensimV2Result, ZensimError> {
        let mut scratch = crate::feature_v2::V2Scratch::new();
        crate::feature_v2::compute_v2_features_with_ref_impl(
            reference,
            distorted,
            self.max_pixels,
            self.parallel_now(),
            crate::feature_v2::V2NewFeatureToggles::default(),
            &mut scratch,
        )
    }

    /// Full-control batch form of [`Self::compute_v2_features_with_ref`]:
    /// explicit feature-group toggles and a caller-owned reusable
    /// [`crate::feature_v2::V2Scratch`] (one per worker thread), so
    /// steady-state extraction performs zero scratch allocation per pair.
    ///
    /// # Errors
    ///
    /// Same as [`Self::compute_v2_features_with_ref`].
    #[cfg(feature = "feature-regime-v2")]
    pub fn compute_v2_features_with_ref_and_scratch(
        &self,
        reference: &crate::feature_v2::V2PreparedReference,
        distorted: &impl ImageSource,
        toggles: crate::feature_v2::V2NewFeatureToggles,
        scratch: &mut crate::feature_v2::V2Scratch,
    ) -> Result<crate::feature_v2::ZensimV2Result, ZensimError> {
        // REV4SERVE: only a same-revision Rev4 process serves a Rev4 request.
        crate::ssim_form::refuse_rev4_mix(toggles.formula_revision)?;
        crate::feature_v2::compute_v2_features_with_ref_impl(
            reference,
            distorted,
            self.max_pixels,
            self.parallel_now(),
            toggles,
            scratch,
        )
    }

    /// Compute the FOLDED 720-layout feature vector in ONE v2-walk pass:
    /// `[f0..156) = v1 basic` (computed from the v2 walk's shared
    /// H-blurred planes via v1's own fused kernel — parity-gated against
    /// the frozen v1 path, not byte-frozen), `[156..372) = 0.0` (v1's
    /// peak/masked/IW pool blocks, deprecated: no current model reads
    /// them), `[372..720) = v2-348`.
    ///
    /// This is the production extraction for the foldable-384 / 504-class
    /// models (dial + diffmap + corruption head): one pass replaces the
    /// research-only double extraction (`compute_extended_features` ++
    /// `compute_v2_features`), skipping the v1 activity/pool machinery
    /// those models never read. Do NOT mix folded rows into corpora whose
    /// `f156..371` columns are live — this is its own extraction regime
    /// ([`crate::feature_v2::FeatureRegime::Folded720`]).
    ///
    /// STREAMING-ONLY since the C5 switchover (2026-07-26, user-approved
    /// ~1.33× batch-CPU trade for the 4.7× memory reduction + one code
    /// path): routes through the strip-plane producer; there is no
    /// prepared-reference form — batch drivers use
    /// [`Self::compute_folded720_features_streaming`] with a per-worker
    /// [`crate::feature_v2::V2Scratch`].
    ///
    /// # Errors
    ///
    /// Same as [`Self::compute_v2_features`].
    #[cfg(feature = "feature-regime-v2")]
    pub fn compute_folded720_features(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
    ) -> Result<crate::feature_v2::ZensimV2Result, ZensimError> {
        crate::feature_v2::compute_folded720_impl_with_toggles(
            source,
            distorted,
            self.max_pixels,
            self.parallel_now(),
            crate::feature_v2::V2NewFeatureToggles::default(),
        )
    }

    /// [`Self::compute_folded720_features`] plus the f720+ APPEND block
    /// (17/ch/scale — cross-channel masking, luminance conditioning,
    /// MSCN divisive comparison, σ-split contrast/texture, deviation
    /// pooling, global stats; see `feature_v2::idx_append` and the
    /// 2026-07-26 gap audit in
    /// `zenpapers/docs/zensim-720-feature-gaps-2026-07-26.md`). Emits a
    /// 924-slot vector: the first 720 slots are bit-identical to the
    /// plain folded extraction
    /// ([`crate::feature_v2::FeatureRegime::Folded720Append`]).
    ///
    /// # Errors
    ///
    /// Same as [`Self::compute_v2_features`]. Rev5 explicitly refuses append
    /// and every other family outside basic + peaks + v2, on SDR and HDR
    /// raw extraction entries.
    #[cfg(feature = "feature-regime-v2")]
    pub fn compute_folded720_append_features(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
    ) -> Result<crate::feature_v2::ZensimV2Result, ZensimError> {
        crate::feature_v2::compute_folded720_append_impl(
            source,
            distorted,
            self.max_pixels,
            self.parallel_now(),
            crate::feature_v2::V2NewFeatureToggles::default(),
        )
    }

    /// STREAMING form of [`Self::compute_folded720_features`]: the same
    /// 720-layout output, bit-for-bit, computed through the strip-plane
    /// producer (`docs/STREAMING_FOLDAPP_C0_DESIGN_2026-07-26.md`) —
    /// O(width) rolling planes instead of materialized full-image
    /// pyramids, no prepared reference, no moments cache. Peak memory is
    /// the `v1stream` class regardless of image height.
    ///
    /// # Errors
    ///
    /// Same as [`Self::compute_v2_features`].
    #[cfg(feature = "feature-regime-v2")]
    pub fn compute_folded720_features_streaming(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
        toggles: crate::feature_v2::V2NewFeatureToggles,
        scratch: &mut crate::feature_v2::V2Scratch,
    ) -> Result<crate::feature_v2::ZensimV2Result, ZensimError> {
        // REV4SERVE: only a same-revision Rev4 process serves a Rev4 request.
        crate::ssim_form::refuse_rev4_mix(toggles.formula_revision)?;
        crate::feature_v2::compute_folded720_streaming_impl(
            source,
            distorted,
            self.max_pixels,
            self.parallel_now(),
            toggles,
            scratch,
            None,
        )
    }

    /// STREAMING form of [`Self::compute_folded720_append_features`]
    /// (924 slots, `toggles.append_block` forced on): bit-identical
    /// output through the strip-plane producer — the append σ-split's
    /// `bs2 = blur(src²)` and the Y channel's X/B cross-activities are
    /// computed per kernel strip inside the walk, so no reference cache
    /// or replay plane is ever materialized.
    ///
    /// # Errors
    ///
    /// Same as [`Self::compute_v2_features`].
    #[cfg(feature = "feature-regime-v2")]
    pub fn compute_folded720_append_features_streaming(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
        toggles: crate::feature_v2::V2NewFeatureToggles,
        scratch: &mut crate::feature_v2::V2Scratch,
    ) -> Result<crate::feature_v2::ZensimV2Result, ZensimError> {
        // REV4SERVE: only a same-revision Rev4 process serves a Rev4 request.
        crate::ssim_form::refuse_rev4_mix(toggles.formula_revision)?;
        crate::feature_v2::compute_folded720_append_streaming_impl(
            source,
            distorted,
            self.max_pixels,
            self.parallel_now(),
            toggles,
            scratch,
        )
    }

    /// Declared-HDR form of [`Self::compute_folded720_features_streaming`]
    /// (HDR_PLAN chunk 2): pixel values map to absolute display light per
    /// `encoding` (PQ / HLG display model, or already-linear cd/m²), then
    /// the UPIQ-validated PU-XYB front-end (opsin → PU21 `banding_glare`
    /// per channel, normalized so 100 cd/m² → 1.0 — SDR-range content
    /// lands on the scale the downstream formulas were tuned on) feeds
    /// the UNCHANGED streaming 720 walk. Sources: `LinearF32Rgba` (f32)
    /// or `Srgb16Rgba` (code values; `Pq`/`Hlg` only), `AlphaMode::Opaque`,
    /// declared primaries converted to the opsin matrix's linear-sRGB basis
    /// without clipping absolute light to the SDR range. SDR extraction is unchanged.
    ///
    /// # Errors
    ///
    /// [`ZensimError::HdrInputRequiresPuPath`] for unsupported
    /// format/alpha shapes, plus the error conditions of
    /// [`Self::compute_v2_features`].
    #[cfg(feature = "feature-regime-v2")]
    pub fn compute_folded720_features_hdr(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
        encoding: crate::feature_v2::HdrEncoding,
        toggles: crate::feature_v2::V2NewFeatureToggles,
        scratch: &mut crate::feature_v2::V2Scratch,
    ) -> Result<crate::feature_v2::ZensimV2Result, ZensimError> {
        // REV4SERVE: only a same-revision Rev4 process serves a Rev4 request.
        crate::ssim_form::refuse_rev4_mix(toggles.formula_revision)?;
        crate::feature_v2::compute_folded720_hdr_streaming_impl(
            source,
            distorted,
            encoding,
            self.max_pixels,
            self.parallel_now(),
            toggles,
            scratch,
            None,
        )
    }

    /// [`Self::compute_folded720_features_hdr`] with the f720+ append
    /// block forced on — the 924-slot declared-HDR extraction.
    ///
    /// # Errors
    ///
    /// Same as [`Self::compute_folded720_features_hdr`].
    #[cfg(feature = "feature-regime-v2")]
    pub fn compute_folded720_append_features_hdr(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
        encoding: crate::feature_v2::HdrEncoding,
        toggles: crate::feature_v2::V2NewFeatureToggles,
        scratch: &mut crate::feature_v2::V2Scratch,
    ) -> Result<crate::feature_v2::ZensimV2Result, ZensimError> {
        // REV4SERVE: only a same-revision Rev4 process serves a Rev4 request.
        crate::ssim_form::refuse_rev4_mix(toggles.formula_revision)?;
        crate::feature_v2::compute_folded720_append_hdr_streaming_impl(
            source,
            distorted,
            encoding,
            self.max_pixels,
            self.parallel_now(),
            toggles,
            scratch,
        )
    }

    /// Folded-720+append+**append2** extraction (944 slots;
    /// [`crate::feature_v2::FeatureRegime::Folded720Append2`]): the 924
    /// layout plus [`crate::feature_v2::idx_append2`] — BANDVIS banding
    /// gain/loss (empirical per-route δ band-pass × flatness mask), the
    /// free reference-luminance conditioner, and (HDR route only)
    /// highlight bins. OPT-IN and additive-only: joins the NEXT
    /// extraction regime wave; never mix 944 rows into 924 tables.
    ///
    /// # Errors
    ///
    /// Same as [`Self::compute_folded720_append_features`].
    #[cfg(feature = "feature-regime-v2")]
    pub fn compute_folded720_append2_features(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
    ) -> Result<crate::feature_v2::ZensimV2Result, ZensimError> {
        crate::feature_v2::compute_folded720_append2_impl(
            source,
            distorted,
            self.max_pixels,
            self.parallel_now(),
            crate::feature_v2::V2NewFeatureToggles::default(),
        )
    }

    /// Declared-HDR form of [`Self::compute_folded720_append2_features`]
    /// (PU-domain BANDVIS constants; highlight bins live).
    ///
    /// # Errors
    ///
    /// Same as [`Self::compute_folded720_append_features_hdr`].
    #[cfg(feature = "feature-regime-v2")]
    pub fn compute_folded720_append2_features_hdr(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
        encoding: crate::feature_v2::HdrEncoding,
        toggles: crate::feature_v2::V2NewFeatureToggles,
        scratch: &mut crate::feature_v2::V2Scratch,
    ) -> Result<crate::feature_v2::ZensimV2Result, ZensimError> {
        // REV4SERVE: only a same-revision Rev4 process serves a Rev4 request.
        crate::ssim_form::refuse_rev4_mix(toggles.formula_revision)?;
        crate::feature_v2::compute_folded720_append2_hdr_streaming_impl(
            source,
            distorted,
            encoding,
            self.max_pixels,
            self.parallel_now(),
            toggles,
            scratch,
        )
    }

    /// Folded-720+append+append2+**CSFW** extraction (956 slots;
    /// [`crate::feature_v2::FeatureRegime::Folded720Csfw`]): the 944
    /// layout plus [`crate::feature_v2::idx_csfw`] — chunk-3 tier-1
    /// luminance-CSF-weighted GLOBAL_* pooling twins (Y-only, both
    /// routes, derived route-local φ constants —
    /// `docs/CSF_CHUNK3_DESIGN_2026-07-28.md`). OPT-IN and
    /// additive-only: joins the NEXT extraction regime wave; never mix
    /// 956 rows into 944 or 924 tables.
    ///
    /// # Errors
    ///
    /// Same as [`Self::compute_folded720_append_features`].
    #[cfg(feature = "feature-regime-v2")]
    pub fn compute_folded720_csfw_features(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
    ) -> Result<crate::feature_v2::ZensimV2Result, ZensimError> {
        crate::feature_v2::compute_folded720_csfw_impl(
            source,
            distorted,
            self.max_pixels,
            self.parallel_now(),
            crate::feature_v2::V2NewFeatureToggles::default(),
        )
    }

    /// Declared-HDR form of [`Self::compute_folded720_csfw_features`]
    /// (PU-route φ constants; append2's highlight bins live).
    ///
    /// # Errors
    ///
    /// Same as [`Self::compute_folded720_append_features_hdr`].
    #[cfg(feature = "feature-regime-v2")]
    pub fn compute_folded720_csfw_features_hdr(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
        encoding: crate::feature_v2::HdrEncoding,
        toggles: crate::feature_v2::V2NewFeatureToggles,
        scratch: &mut crate::feature_v2::V2Scratch,
    ) -> Result<crate::feature_v2::ZensimV2Result, ZensimError> {
        // REV4SERVE: only a same-revision Rev4 process serves a Rev4 request.
        crate::ssim_form::refuse_rev4_mix(toggles.formula_revision)?;
        crate::feature_v2::compute_folded720_csfw_hdr_streaming_impl(
            source,
            distorted,
            encoding,
            self.max_pixels,
            self.parallel_now(),
            toggles,
            scratch,
        )
    }

    /// Pre-compute reference image data for batch comparison.
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError::ImageTooSmall`] if either dimension is zero.
    pub fn precompute_reference(
        &self,
        source: &impl ImageSource,
    ) -> Result<crate::streaming::PrecomputedReference, ZensimError> {
        let params = self.profile.params();
        if source.width() == 0 || source.height() == 0 {
            return Err(ZensimError::ImageTooSmall);
        }
        // Sub-64px sources are reflect-padded inside `PrecomputedReference::new`
        // (matching the buffered path), so no 8px floor here.
        check_within_max_pixels(source.width(), source.height(), self.max_pixels)?;
        reject_hdr_input(source)?;
        Ok(crate::streaming::PrecomputedReference::new(
            source,
            params.num_scales,
            self.parallel_now(),
            config_from_params(params, self.parallel_now()).revision(),
        ))
    }

    /// The engine-routed body of [`Self::compute_with_ref`]: the fold-backed
    /// ref-cached compare when this instance asks for it AND the fold can
    /// serve the request bit-identically, the buffered `*_with_ref` walk
    /// otherwise. Both consume the SAME `PrecomputedReference` — the pyramid
    /// cache is not buffered-walk state (fold-engine lane stage 3).
    fn compare_against_ref(
        &self,
        precomputed: &crate::streaming::PrecomputedReference,
        distorted: &impl ImageSource,
        config: &ZensimConfig,
        weights: &[f64],
    ) -> ZensimResult {
        self.compare_against_ref_into(precomputed, distorted, config, weights, None)
    }

    /// [`Self::compare_against_ref`] with an optional caller-owned scratch.
    /// `None` allocates a fresh one per call (what `compute_with_ref` does on
    /// both engines); `Some` is the ref-loop shape `compute_with_ref_into`
    /// exposes, and is what makes the fold's ref-cache saving match buffered's
    /// instead of being halved by a per-compare allocation (fold-MT lane).
    fn compare_against_ref_into(
        &self,
        precomputed: &crate::streaming::PrecomputedReference,
        distorted: &impl ImageSource,
        config: &ZensimConfig,
        weights: &[f64],
        scratch: Option<&mut crate::streaming::ZensimScratch>,
    ) -> ZensimResult {
        let mut scratch = scratch;
        if (config.revision() >= crate::feature_defs::FormulaRevision::Rev5
            || (self.fold_engine && self.stop.is_none()))
            && crate::fold_engine::is_fold_backable(config)
        {
            let mut owned;
            let v2 = match scratch.as_mut() {
                Some(s) => &mut s.v2,
                None => {
                    owned = crate::feature_v2::V2Scratch::new();
                    &mut owned
                }
            };
            if config.revision() >= crate::feature_defs::FormulaRevision::Rev5 {
                let plan = self.scoring_plan(self.profile.params(), config);
                if let Some(result) =
                    crate::fold_engine::compute_fold_backed_with_ref_stoppable_plan(
                        precomputed,
                        distorted,
                        config,
                        weights,
                        v2,
                        plan.as_ref().map(|p| p.compute.v1_pools),
                        plan.as_ref(),
                        self.stop_ref(),
                    )
                {
                    return result;
                }
            }
            // The ref-cached walk emits the v1 layout only; a wide bake is
            // not servable through it yet (phase 2 — registered, and the
            // servability census reports it as such rather than silently
            // scoring a truncated vector). Take just the pool axis.
            let pool_mode = self
                .scoring_plan(self.profile.params(), config)
                .map(|p| p.compute.v1_pools);
            if let Some(r) = crate::fold_engine::compute_fold_backed_with_ref(
                precomputed,
                distorted,
                config,
                weights,
                v2,
                pool_mode,
            ) {
                return r;
            }
        }
        crate::streaming::compute_zensim_streaming_with_ref(
            precomputed,
            distorted,
            config,
            weights,
            self.stop_ref(),
        )
    }

    /// Compare a distorted image against a precomputed reference.
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError::ImageTooSmall`] if either dimension is zero.
    /// Returns [`ZensimError::DimensionMismatch`] if `distorted.width()` /
    /// `distorted.height()` differ from the precomputed reference's
    /// dimensions (see
    /// [`PrecomputedReference::width`](crate::PrecomputedReference::width)).
    /// Returns [`ZensimError::ImageTooLarge`] if dimensions exceed the
    /// configured `max_pixels` cap.
    pub fn compute_with_ref(
        &self,
        precomputed: &crate::streaming::PrecomputedReference,
        distorted: &impl ImageSource,
    ) -> Result<ZensimResult, ZensimError> {
        let params = self.profile.params();
        if distorted.width() == 0 || distorted.height() == 0 {
            return Err(ZensimError::ImageTooSmall);
        }
        // Contract: distorted must match the reference's ORIGINAL (unpadded)
        // dims (`PrecomputedReference` keeps those even when its pyramid was
        // built on a reflect-padded source).
        validate_ref_match(precomputed, distorted)?;
        check_within_max_pixels(distorted.width(), distorted.height(), self.max_pixels)?;
        // The precomputed reference was built via `precompute_reference`,
        // which already rejected HDR — so only check the distorted side
        // here. Symmetry across the call path is preserved.
        reject_hdr_input(distorted)?;
        let config = config_from_params(params, self.parallel_now());
        check_profile_route(params, &config)?;
        validate_cached_revision(precomputed, config.revision())?;
        let (ow, oh) = (distorted.width(), distorted.height());
        // Pad a sub-64px distorted to the pyramid minimum so it aligns with the
        // (also-padded) reference pyramid; score with the original dims. The
        // pad is SHARED — it runs before either engine, so both see the same
        // pixels against the same cached pyramid.
        let mut result = if needs_pyramid_pad(ow, oh, config.num_scales) {
            let d = reflect_pad_for_scales(distorted, config.num_scales);
            self.compare_against_ref(precomputed, &d, &config, params.weights)
        } else {
            self.compare_against_ref(precomputed, distorted, &config, params.weights)
        };
        self.check_stop()?;
        apply_mlp_scoring(&mut result, params, ow as u32, oh as u32)?;
        Ok(result.with_profile(self.profile))
    }

    /// Compare source and distorted images using **strip-aggregating
    /// streaming** for very large images (e.g. 80 MP) where the
    /// standard [`compute`](Self::compute) path would OOM.
    ///
    /// Internally splits both images into horizontal Y-strips of
    /// `strip_inner` scale-0 rows (with `strip_margin` rows of overlap
    /// on each side for the blur stencil), runs the existing pipeline
    /// on each strip independently, and aggregates per-scale
    /// accumulators across strips. Memory peak per pair drops from
    /// `O(full_image × 1.33)` to `O(strip_height × 1.33)`.
    ///
    /// # When to use
    ///
    /// - **Image > 16 MP**: use this path. At 80 MP a `compute()` call
    ///   peaks at ~2.5 GB per pair; 16 rayon workers × 2.5 GB exceeds
    ///   reasonable RAM. This path's per-pair peak is ~125 MB with
    ///   `strip_inner = 256, strip_margin = 128`.
    /// - **Image ≤ 16 MP**: prefer [`compute`](Self::compute) — strip
    ///   aggregation has a small per-strip overhead and a minor
    ///   approximation in the strip-boundary blur context that the
    ///   full path avoids.
    ///
    /// # Default strip geometry
    ///
    /// `strip_inner = 256`, `strip_margin = 128` covers the default
    /// 4-scale pyramid with `blur_radius = 5`. For a 4× larger blur
    /// (`config.blur_radius = 20`), bump `strip_margin` proportionally.
    /// `strip_inner` must be at least `2 * blur_radius + 1` rows at
    /// every scale; with the default 4-scale pyramid that means
    /// `strip_inner >= 16 × (2 * blur_radius + 1) = 176` at default
    /// blur. The `256` default is safe for everything we ship.
    ///
    /// # Precision
    ///
    /// Byte-exact equivalent to the full-image [`compute`](Self::compute)
    /// path (within f64 machine epsilon, < 1e-13 rel). Strip-internal
    /// bands tile against the full-image plane layout so each band's
    /// V-blur running-sum init point and advance count match the
    /// full-image path exactly. Per-strip accumulators sum directly
    /// (raw `ScaleAccumulators::merge`) with no per-strip finalize
    /// precision loss.
    ///
    /// # Reference data: strip-per-strip (one-off pair) mode
    ///
    /// This entry builds the reference XYB pyramid per strip — each
    /// strip's `PrecomputedReference` covers JUST that strip's source
    /// rows + margin. Peak per-pair memory is `O(strip_h × width)`,
    /// at the cost of converting + downscaling the reference for each
    /// strip. **Best for one-off pairs** where the reference is not
    /// reused (each call sees a different source/distorted pair).
    ///
    /// For batch quantization loops where many distorted candidates
    /// are scored against the same reference, use
    /// [`Self::compute_with_ref_streaming_strips`] instead — that
    /// variant pre-builds the FULL reference pyramid once and reuses
    /// it across strips (and across calls).
    ///
    /// At Rev5, this delegates to the canonical streaming fold with its
    /// fixed 128-row reduction tree; caller strip geometry does not change
    /// the features or score.
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError`] if dimensions are mismatched or too
    /// small to support the strip geometry.
    pub fn compute_streaming_strips(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
        strip_inner: usize,
        strip_margin: usize,
    ) -> Result<ZensimResult, ZensimError> {
        let params = self.profile.params();
        validate_pair(source, distorted)?;
        // Sub-64px images can't form the pyramid the strip geometry assumes and
        // don't need streaming (they fit in memory) — route them to the
        // buffered path, which reflect-pads to the pyramid minimum.
        if needs_pyramid_pad(source.width(), source.height(), params.num_scales) {
            return self.compute(source, distorted);
        }
        check_within_max_pixels(source.width(), source.height(), self.max_pixels)?;
        let config = config_from_params(params, self.parallel_now());
        check_profile_route(params, &config)?;
        if config.revision() >= crate::feature_defs::FormulaRevision::Rev5 {
            return self.compute(source, distorted);
        }
        crate::ssim_form::refuse_rev4_strips(config.revision())?;
        // Identity is decided by ONE owner (`images_byte_identical` ->
        // `identical_result_at`, inside `compute`). The strip walk has no
        // short-circuit of its own, so a perfect copy scored here returned the
        // model's forward on an all-zero feature row (96.24 for `B`) while
        // `compute` returned exactly 100 — two paths, two answers, same pair.
        // Delegating to `compute` cannot drift from it; `compute` short-circuits
        // immediately so no strip work is done. Same shape as the sub-64px
        // delegation above.
        if images_byte_identical(source, distorted) {
            return self.compute(source, distorted);
        }

        let (stats, mean_offset) = crate::streaming::compute_multiscale_stats_streaming_strips(
            source,
            distorted,
            &config,
            params.weights,
            strip_inner,
            strip_margin,
        );
        let mut result = combine_scores(&stats, params.weights, &config, mean_offset);
        apply_mlp_scoring(
            &mut result,
            params,
            source.width() as u32,
            source.height() as u32,
        )?;
        Ok(result.with_profile(self.profile))
    }

    /// Same as [`compute_streaming_strips`](Self::compute_streaming_strips)
    /// with sensible defaults: `strip_inner = 256`, `strip_margin = 128`.
    /// Use for 80 MP+ images.
    pub fn compute_streaming_strips_default(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
    ) -> Result<ZensimResult, ZensimError> {
        self.compute_streaming_strips(source, distorted, 256, 128)
    }

    /// Strip-aggregating variant that REUSES a full pre-built reference
    /// pyramid across strips — best for batch encoder loops where many
    /// distorted candidates are scored against the same source.
    ///
    /// Unlike [`Self::compute_streaming_strips`] (which builds a fresh
    /// per-strip [`crate::streaming::PrecomputedReference`] on each call,
    /// saving memory but redoing reference XYB conversion + pyramid
    /// downscale for every strip), this entry takes a caller-owned full
    /// `PrecomputedReference` and slices the appropriate Y-range out of
    /// it per strip. The distorted side is still processed per-strip so
    /// the distorted-side memory peak stays bounded.
    ///
    /// # Memory tradeoff
    ///
    /// - **`compute_streaming_strips`** (strip-per-strip ref): peak
    ///   `O(strip_h × width × 4 bytes × 3 channels × 2 sides × 1.33)`
    ///   per pair. ~125 MB at 80 MP with default strip geometry. Best
    ///   for memory-constrained one-off pairs.
    /// - **`compute_with_ref_streaming_strips`** (buffered ref): the
    ///   `PrecomputedReference` holds the FULL ref pyramid (~1.28 GB
    ///   at 80 MP). Per-call overhead is just the distorted side
    ///   (~960 MB at 80 MP), but amortized across N distorted
    ///   candidates the per-call cost drops to one distorted-side
    ///   pass. Best for encoder quantization loops.
    ///
    /// # Precision
    ///
    /// At Rev5, this delegates to the canonical reference-fed fold;
    /// `strip_inner`/`strip_margin` do not change its fixed 128-row tree.
    /// The full reference cache remains resident.
    /// Byte-exact equivalent to [`Self::compute_with_ref`] (within f64
    /// machine epsilon).
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError`] if dimensions are mismatched or too
    /// small to support the strip geometry.
    pub fn compute_with_ref_streaming_strips(
        &self,
        precomputed: &crate::streaming::PrecomputedReference,
        distorted: &impl ImageSource,
        strip_inner: usize,
        strip_margin: usize,
    ) -> Result<ZensimResult, ZensimError> {
        reject_hdr_input(distorted)?;
        let params = self.profile.params();
        if distorted.width() == 0 || distorted.height() == 0 {
            return Err(ZensimError::ImageTooSmall);
        }
        validate_ref_match(precomputed, distorted)?;
        // Sub-64px distorted doesn't need streaming and can't form the strip
        // pyramid — route to the buffered ref path (which pads to align with
        // the also-padded reference).
        if needs_pyramid_pad(distorted.width(), distorted.height(), params.num_scales) {
            return self.compute_with_ref(precomputed, distorted);
        }
        check_within_max_pixels(distorted.width(), distorted.height(), self.max_pixels)?;
        let config = config_from_params(params, self.parallel_now());
        check_profile_route(params, &config)?;
        validate_cached_revision(precomputed, config.revision())?;
        if config.revision() >= crate::feature_defs::FormulaRevision::Rev5 {
            return self.compute_with_ref(precomputed, distorted);
        }
        crate::ssim_form::refuse_rev4_strips(config.revision())?;

        let (stats, mean_offset) =
            crate::streaming::compute_multiscale_stats_streaming_strips_with_ref(
                precomputed,
                distorted,
                &config,
                params.weights,
                strip_inner,
                strip_margin,
            );
        let mut result = combine_scores(&stats, params.weights, &config, mean_offset);
        apply_mlp_scoring(
            &mut result,
            params,
            distorted.width() as u32,
            distorted.height() as u32,
        )?;
        Ok(result.with_profile(self.profile))
    }

    /// Same as [`Self::compute_with_ref_streaming_strips`] with default
    /// strip geometry (`strip_inner = 256`, `strip_margin = 128`).
    pub fn compute_with_ref_streaming_strips_default(
        &self,
        precomputed: &crate::streaming::PrecomputedReference,
        distorted: &impl ImageSource,
    ) -> Result<ZensimResult, ZensimError> {
        self.compute_with_ref_streaming_strips(precomputed, distorted, 256, 128)
    }

    /// Compare against a precomputed reference, reusing caller-owned scratch
    /// buffers. Designed for encoder quantization loops that compare many
    /// distorted candidates against the same reference: the per-call XYB
    /// plane allocation (which can be ~25 MB at 1080p, ~99 MB at 4K) is
    /// kept alive across calls instead of being freed and reallocated.
    ///
    /// The first call costs the same as [`Zensim::compute_with_ref`]; subsequent
    /// calls skip the allocation and the OS page-fault commit.
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError::ImageTooSmall`] if either dimension is zero.
    pub fn compute_with_ref_into(
        &self,
        precomputed: &crate::streaming::PrecomputedReference,
        distorted: &impl ImageSource,
        scratch: &mut crate::streaming::ZensimScratch,
    ) -> Result<ZensimResult, ZensimError> {
        let params = self.profile.params();
        if distorted.width() == 0
            || distorted.height() == 0
            || (crate::ssim_form::active_revision() < crate::feature_defs::FormulaRevision::Rev5
                && (distorted.width() < 8 || distorted.height() < 8))
        {
            return Err(ZensimError::ImageTooSmall);
        }
        validate_ref_match(precomputed, distorted)?;
        check_within_max_pixels(distorted.width(), distorted.height(), self.max_pixels)?;
        reject_hdr_input(distorted)?;
        let config = config_from_params(params, self.parallel_now());
        check_profile_route(params, &config)?;
        validate_cached_revision(precomputed, config.revision())?;
        let (ow, oh) = (distorted.width(), distorted.height());
        // ENGINE ROUTING (fold-MT lane). `compute_with_ref` has routed to the
        // fold since the fold-engine lane; this entry did not, so a ref LOOP —
        // the one shape that exists to amortise work — was the one shape that
        // could not use the fold. It routes through the same
        // `compare_against_ref_into` and hands it `scratch`, so the fold's
        // per-compare `V2Scratch` is reused exactly like buffered's
        // `dst_planes`. Out-of-domain requests still degrade to the buffered
        // walk below, byte-for-byte.
        if (config.revision() >= crate::feature_defs::FormulaRevision::Rev5
            || (self.fold_engine && self.stop.is_none()))
            && crate::fold_engine::is_fold_backable(&config)
        {
            // Same shared reflect-pad `compute_with_ref` applies, so the two
            // ref entries route identically rather than differing on sub-64.
            let mut result = if needs_pyramid_pad(ow, oh, config.num_scales) {
                let d = reflect_pad_for_scales(distorted, config.num_scales);
                self.compare_against_ref_into(
                    precomputed,
                    &d,
                    &config,
                    params.weights,
                    Some(scratch),
                )
            } else {
                self.compare_against_ref_into(
                    precomputed,
                    distorted,
                    &config,
                    params.weights,
                    Some(scratch),
                )
            };
            apply_mlp_scoring(&mut result, params, ow as u32, oh as u32)?;
            return Ok(result.with_profile(self.profile));
        }
        // Pad a sub-pyramid-minimum distorted so it aligns with the (also
        // padded) reference pyramid — the same rule `compute_with_ref`
        // applies. Without it this entry fed a 48-wide plane to a 64-wide
        // reference scale and tripped `scale 0 width mismatch`; scoring still
        // uses the ORIGINAL dims (2026-08-30).
        let (stats, mean_offset) = if needs_pyramid_pad(ow, oh, config.num_scales) {
            let d = reflect_pad_for_scales(distorted, config.num_scales);
            crate::streaming::compute_multiscale_stats_streaming_with_ref_borrowed(
                precomputed,
                &d,
                &mut scratch.dst_planes,
                &config,
                params.weights,
                self.stop_ref(),
            )
        } else {
            crate::streaming::compute_multiscale_stats_streaming_with_ref_borrowed(
                precomputed,
                distorted,
                &mut scratch.dst_planes,
                &config,
                params.weights,
                self.stop_ref(),
            )
        };
        self.check_stop()?;
        let mut result = combine_scores(&stats, params.weights, &config, mean_offset);
        apply_mlp_scoring(&mut result, params, ow as u32, oh as u32)?;
        Ok(result.with_profile(self.profile))
    }

    /// Precompute reference from planar linear RGB f32 data.
    ///
    /// `planes` are `[R, G, B]`, each with at least `stride * height` elements.
    /// `stride` is the number of f32 elements per row (≥ `width`; may be larger
    /// for padded buffers like the encoder's `padded_width`).
    ///
    /// This avoids the interleave-to-RGBA overhead when the caller already has
    /// separate channel buffers in linear light.
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError::ImageTooSmall`] if either dimension is zero.
    pub fn precompute_reference_linear_planar(
        &self,
        planes: [&[f32]; 3],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<crate::streaming::PrecomputedReference, ZensimError> {
        let params = self.profile.params();
        if width < 8 || height < 8 {
            return Err(ZensimError::ImageTooSmall);
        }
        check_within_max_pixels(width, height, self.max_pixels)?;
        // `stride * height` must fit in usize on 32-bit; reject overflow up
        // front so downstream `y * stride + x` arithmetic cannot wrap.
        let row_capacity = stride
            .checked_mul(height)
            .ok_or(ZensimError::ImageTooLarge)?;
        for plane in &planes {
            if plane.len() < row_capacity {
                return Err(ZensimError::InvalidDataLength);
            }
        }
        Ok(crate::streaming::PrecomputedReference::from_linear_planar(
            planes,
            width,
            height,
            stride,
            params.num_scales,
            self.parallel_now(),
            config_from_params(params, self.parallel_now()).revision(),
        ))
    }

    /// **HDR.** Score a pair supplied as **interleaved** absolute-luminance
    /// linear RGB (`[R, G, B, R, G, B, …]` f32, cd/m²), using the PU21
    /// perceptually-uniform front-end (banding_glare) in place of the SDR
    /// cube-root nonlinearity. This is the primary HDR entry point —
    /// interleaved rows are how HDR pixels normally arrive (decoder output,
    /// EXR readers, GPU readbacks), and they feed the PU conversion with no
    /// per-row gather. See `docs/HDR_PLAN.md` §2b; validation:
    /// `benchmarks/upiq_pu_validation_2026-06-01.md`.
    ///
    /// Inputs are display-emitted luminance. PQ/HLG **code values** must
    /// first be mapped to cd/m² by a display model (PQ: EOTF × peak nits;
    /// HLG: OOTF at display peak) — that conversion lives in the caller
    /// (e.g. zenpixels-convert), not here. Absolute-score calibration
    /// against a trained HDR bake is tracked in
    /// [imazen/zensimx#38](https://github.com/imazen/zensimx/issues/38).
    ///
    /// `ref_stride` / `dist_stride` count f32 elements between row starts
    /// (≥ `3 * width`; tightly packed = `3 * width`). Row padding beyond
    /// `3 * width` is ignored, so strided buffers (SIMD-aligned rows,
    /// sub-rect views) score without repacking.
    ///
    /// Sub-64px inputs reflect-pad up to the pyramid minimum exactly like the
    /// SDR entry points — scores stay defined down to 1×1.
    ///
    /// # Errors
    /// [`ZensimError::ImageTooSmall`] (zero-sized),
    /// [`ZensimError::InvalidStride`] (stride < `3 * width`),
    /// [`ZensimError::ImageTooLarge`] (arithmetic overflow), or
    /// [`ZensimError::InvalidDataLength`] (slice too short).
    pub fn compute_pu_linear(
        &self,
        ref_rgb: &[f32],
        dist_rgb: &[f32],
        width: usize,
        height: usize,
        ref_stride: usize,
        dist_stride: usize,
    ) -> Result<ZensimResult, ZensimError> {
        let params = self.profile.params_pu_linear();
        if width == 0 || height == 0 {
            return Err(ZensimError::ImageTooSmall);
        }
        check_within_max_pixels(width, height, self.max_pixels)?;
        let min_stride = width.checked_mul(3).ok_or(ZensimError::ImageTooLarge)?;
        for (slice, stride) in [(ref_rgb, ref_stride), (dist_rgb, dist_stride)] {
            if stride < min_stride {
                return Err(ZensimError::InvalidStride);
            }
            // The last row only needs `3 * width` elements, not a full
            // stride — same convention as `StridedBytes`.
            let required = (height - 1)
                .checked_mul(stride)
                .and_then(|v| v.checked_add(min_stride))
                .ok_or(ZensimError::ImageTooLarge)?;
            if slice.len() < required {
                return Err(ZensimError::InvalidDataLength);
            }
        }
        let config = config_from_params(params, self.parallel_now());
        check_profile_route(params, &config)?;
        // Identical inputs must score exactly 100.0 through the same
        // `mark_identical` contract the SDR path uses (compares only the
        // valid `3 * width` of each row, so stride padding is ignored here
        // too).
        let is_identical = || {
            (0..height).all(|y| {
                ref_rgb[y * ref_stride..y * ref_stride + 3 * width]
                    == dist_rgb[y * dist_stride..y * dist_stride + 3 * width]
            })
        };
        if let Some(route) = self.pu_fold_route(params, &config) {
            if route.short_circuit_identity && is_identical() {
                return self.pu_wide_identity(
                    params,
                    &config,
                    route.plan.as_deref(),
                    width,
                    height,
                );
            }
            let r = nits_image(width, height, |x, y, ch| {
                ref_rgb[y * ref_stride + x * 3 + ch]
            })?;
            let d = nits_image(width, height, |x, y, ch| {
                dist_rgb[y * dist_stride + x * 3 + ch]
            })?;
            return self.compute_pu_fold(&r, &d, params, &config, route.plan.as_deref());
        }
        let identical = is_identical();
        let mut result = if identical {
            identical_result(&config)
        } else {
            let (stats, mean_offset) =
                crate::streaming::compute_multiscale_stats_pu_linear_interleaved(
                    ref_rgb,
                    dist_rgb,
                    width,
                    height,
                    ref_stride,
                    dist_stride,
                    &config,
                    params.weights,
                    self.stop_ref(),
                );
            combine_scores(&stats, params.weights, &config, mean_offset)
        };
        self.check_stop()?;
        apply_mlp_scoring(&mut result, params, width as u32, height as u32)?;
        Ok(result.with_profile(self.profile))
    }

    /// Extended-features variant of
    /// [`compute_pu_linear`](Self::compute_pu_linear) — same absolute-nits
    /// PU-XYB path, with the extended (masked) feature block forced on
    /// exactly like [`compute_extended_features`](Self::compute_extended_features)
    /// does for the sRGB path. The per-scale engine is shared
    /// (`process_scale_bands`), so the emitted feature vector has the same
    /// layout/width as the sRGB extraction under the same profile params
    /// (372 in the `with-iw` regime). This is the transfer-invariant HDR
    /// feature path: no u8 shell, no display-peak anchor at the input.
    /// Rev5 refuses this explicit masked-family extraction request.
    pub fn compute_pu_linear_extended_features(
        &self,
        ref_rgb: &[f32],
        dist_rgb: &[f32],
        width: usize,
        height: usize,
        ref_stride: usize,
        dist_stride: usize,
    ) -> Result<ZensimResult, ZensimError> {
        let params = self.profile.params_pu_linear();
        if width == 0 || height == 0 {
            return Err(ZensimError::ImageTooSmall);
        }
        check_within_max_pixels(width, height, self.max_pixels)?;
        let min_stride = width.checked_mul(3).ok_or(ZensimError::ImageTooLarge)?;
        for (slice, stride) in [(ref_rgb, ref_stride), (dist_rgb, dist_stride)] {
            if stride < min_stride {
                return Err(ZensimError::InvalidStride);
            }
            let required = (height - 1)
                .checked_mul(stride)
                .and_then(|v| v.checked_add(min_stride))
                .ok_or(ZensimError::ImageTooLarge)?;
            if slice.len() < required {
                return Err(ZensimError::InvalidDataLength);
            }
        }
        let mut config = config_from_params(params, self.parallel_now());
        check_profile_route(params, &config)?;
        config.extended_features = true;
        if config.revision() >= crate::feature_defs::FormulaRevision::Rev5 {
            return Err(ZensimError::ModelLoadFailed {
                reason: "formula revision 5 supports only basic + peaks + v2; extended extraction requests masked features",
            });
        }
        if crate::ssim_form::active_revision() >= crate::feature_defs::FormulaRevision::Rev5
            && crate::fold_engine::is_fold_backable(&config)
        {
            let r = nits_image(width, height, |x, y, ch| {
                ref_rgb[y * ref_stride + x * 3 + ch]
            })?;
            let d = nits_image(width, height, |x, y, ch| {
                dist_rgb[y * dist_stride + x * 3 + ch]
            })?;
            let plan = cached_pu_plan(params, &config, true);
            return self.compute_pu_fold(&r, &d, params, &config, plan.as_deref());
        }
        let identical = (0..height).all(|y| {
            ref_rgb[y * ref_stride..y * ref_stride + 3 * width]
                == dist_rgb[y * dist_stride..y * dist_stride + 3 * width]
        });
        let mut result = if identical {
            identical_result(&config)
        } else {
            let (stats, mean_offset) =
                crate::streaming::compute_multiscale_stats_pu_linear_interleaved(
                    ref_rgb,
                    dist_rgb,
                    width,
                    height,
                    ref_stride,
                    dist_stride,
                    &config,
                    params.weights,
                    self.stop_ref(),
                );
            combine_scores(&stats, params.weights, &config, mean_offset)
        };
        self.check_stop()?;
        apply_mlp_scoring(&mut result, params, width as u32, height as u32)?;
        Ok(result.with_profile(self.profile))
    }

    /// Planar variant of [`compute_pu_linear`](Self::compute_pu_linear): the
    /// same PU21 HDR scoring with each image supplied as `[R, G, B]` planes,
    /// each ≥ `stride * height` f32 (`stride` in f32 elements per plane row,
    /// shared by all six planes). For pipelines that already keep channels
    /// separate; scores are identical to the interleaved entry, including the
    /// sub-64px reflect-padding.
    ///
    /// # Errors
    /// [`ZensimError::ImageTooSmall`] (zero-sized),
    /// [`ZensimError::ImageTooLarge`] (overflow), or
    /// [`ZensimError::InvalidDataLength`] (short plane).
    pub fn compute_pu_linear_planar(
        &self,
        ref_planes: [&[f32]; 3],
        dist_planes: [&[f32]; 3],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<ZensimResult, ZensimError> {
        let params = self.profile.params_pu_linear();
        if width == 0 || height == 0 {
            return Err(ZensimError::ImageTooSmall);
        }
        check_within_max_pixels(width, height, self.max_pixels)?;
        if stride < width {
            return Err(ZensimError::InvalidStride);
        }
        let row_capacity = stride
            .checked_mul(height)
            .ok_or(ZensimError::ImageTooLarge)?;
        for plane in ref_planes.iter().chain(dist_planes.iter()) {
            if plane.len() < row_capacity {
                return Err(ZensimError::InvalidDataLength);
            }
        }
        let config = config_from_params(params, self.parallel_now());
        check_profile_route(params, &config)?;
        // Same identity short-circuit as the interleaved entry (valid
        // `width` of each plane row only).
        let is_identical = || {
            ref_planes.iter().zip(dist_planes.iter()).all(|(r, d)| {
                (0..height)
                    .all(|y| r[y * stride..y * stride + width] == d[y * stride..y * stride + width])
            })
        };
        if let Some(route) = self.pu_fold_route(params, &config) {
            if route.short_circuit_identity && is_identical() {
                return self.pu_wide_identity(
                    params,
                    &config,
                    route.plan.as_deref(),
                    width,
                    height,
                );
            }
            let r = nits_image(width, height, |x, y, ch| ref_planes[ch][y * stride + x])?;
            let d = nits_image(width, height, |x, y, ch| dist_planes[ch][y * stride + x])?;
            return self.compute_pu_fold(&r, &d, params, &config, route.plan.as_deref());
        }
        let identical = is_identical();
        let mut result = if identical {
            identical_result(&config)
        } else {
            let (stats, mean_offset) = crate::streaming::compute_multiscale_stats_pu_linear_planar(
                ref_planes,
                dist_planes,
                width,
                height,
                stride,
                &config,
                params.weights,
                self.stop_ref(),
            );
            combine_scores(&stats, params.weights, &config, mean_offset)
        };
        self.check_stop()?;
        apply_mlp_scoring(&mut result, params, width as u32, height as u32)?;
        Ok(result.with_profile(self.profile))
    }

    /// Which walk a PU-linear scoring entry takes: `None` is the legacy
    /// buffered PU walk; `Some` is [`Self::compute_pu_fold`] with the plan to
    /// pass it.
    ///
    /// - **Rev5, fold-backable config:** the canonical fold, as before. Rev5
    ///   identity features come from the same walk.
    /// - **Below Rev5, a plan wider than the v1 vector, no stop token:** CHdr
    ///   and C (944-layout bakes). The legacy walk emits only the v1 width,
    ///   and the dense gather would refuse the bake. This is the rule SDR
    ///   `compute` applies through `compute_with_config_inner`, including its
    ///   two conditions: a live stop token keeps the request off the fold
    ///   (issue #48: the fold has no intra-walk stop hook, so cancellation
    ///   would degrade to the before/after checks), and identical pixels
    ///   short-circuit to zeros at the plan width
    ///   ([`Self::pu_wide_identity`]).
    ///
    /// The plan comes from [`cached_pu_plan`], so a profile the legacy walk
    /// already reaches pays one map lookup per call, not a plan derivation,
    /// and its PU path and output bits are unchanged. The fold's
    /// `HdrEncoding::Linear` walk with the plan's toggles is today's build of
    /// the extraction CHdr was trained and validated on
    /// (`benchmarks/chdr_pu_2026-10-10.md`).
    fn pu_fold_route(
        &self,
        params: &'static ProfileParams,
        config: &ZensimConfig,
    ) -> Option<PuFoldRoute> {
        if crate::ssim_form::active_revision() >= crate::feature_defs::FormulaRevision::Rev5
            && crate::fold_engine::is_fold_backable(config)
        {
            return Some(PuFoldRoute {
                plan: cached_pu_plan(params, config, true),
                short_circuit_identity: false,
            });
        }
        if self.stop.is_some() || !crate::fold_engine::is_fold_backable(config) {
            return None;
        }
        let plan = cached_pu_plan(params, config, self.skip_unread_pools)?;
        (plan.walk_width() > crate::fold_engine::v1_feature_width(config)).then_some(PuFoldRoute {
            plan: Some(plan),
            short_circuit_identity: true,
        })
    }

    /// The below-Rev5 wide-plan identity result: exactly 100 with zeros at
    /// the plan width, as SDR `compute` returns through
    /// `identical_result_at` for the same profiles.
    fn pu_wide_identity(
        &self,
        params: &ProfileParams,
        config: &ZensimConfig,
        plan: Option<&crate::feature_plan::Plan>,
        width: usize,
        height: usize,
    ) -> Result<ZensimResult, ZensimError> {
        let mut result = identical_result_at(config, plan.map_or(0, |p| p.walk_width()));
        self.check_stop()?;
        apply_mlp_scoring(&mut result, params, width as u32, height as u32)?;
        Ok(result.with_profile(self.profile))
    }

    /// Canonical PU fold for the planar/interleaved nits APIs, with the plan
    /// [`Self::pu_fold_route`] resolved: every fold-backable config at Rev5,
    /// and wide-plan bakes below Rev5 without a stop token.
    fn compute_pu_fold(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
        params: &ProfileParams,
        config: &ZensimConfig,
        plan: Option<&crate::feature_plan::Plan>,
    ) -> Result<ZensimResult, ZensimError> {
        self.check_stop()?;
        let toggles = plan.map_or_else(
            || crate::feature_v2::V2NewFeatureToggles {
                v1_only: true,
                ..Default::default()
            },
            |p| p.toggles(),
        );
        let mut scratch = crate::feature_v2::V2Scratch::new();
        let mut mo = crate::feature_v2::MeanOffsetRows::new(source.width(), source.height());
        let mut features = crate::feature_v2::compute_folded720_hdr_streaming_extras(
            source,
            distorted,
            crate::feature_v2::HdrEncoding::Linear,
            self.max_pixels,
            self.parallel_now(),
            toggles,
            &mut scratch,
            crate::feature_v2::FoldWalkExtras {
                compute: plan.map(|p| p.compute),
                mean_offset: Some(&mut mo),
                ..Default::default()
            },
        )?
        .into_features();
        if let Some(p) = plan {
            p.check_emit_covered(features.len())
                .map_err(|_| ZensimError::ModelLoadFailed {
                    reason: "Rev5 PU fold did not materialize the bake's declared reads",
                })?;
            features.resize(
                p.walk_width()
                    .max(crate::fold_engine::v1_feature_width(config)),
                0.0,
            );
        } else {
            features.truncate(crate::fold_engine::v1_feature_width(config));
        }
        let (score, raw_distance) =
            score_v1_layout_features(&mut features, params.weights, config, config.num_scales);
        let mut result =
            ZensimResult::new(score, raw_distance, features, self.profile, mo.finish());
        if images_byte_identical(source, distorted) {
            result.score = 100.0;
            result.raw_distance = 0.0;
            result = result.mark_identical();
        }
        self.check_stop()?;
        apply_mlp_scoring(
            &mut result,
            params,
            source.width() as u32,
            source.height() as u32,
        )?;
        Ok(result)
    }

    /// Like `compute`, but always computes all features regardless of
    /// zero weights (forces every channel active). For training/research.
    /// At Rev5 this emits all supported basic, peak and v2 slots in the
    /// 720 layout; masked/IW slots remain outside the supported scope.
    #[cfg(feature = "training")]
    pub fn compute_all_features(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
    ) -> Result<ZensimResult, ZensimError> {
        let params = self.profile.params();
        validate_pair(source, distorted)?;
        let mut config = config_from_params(params, self.parallel_now());
        check_profile_route(params, &config)?;
        config.compute_all_features = true;
        let extraction_plan = if config.revision() >= crate::feature_defs::FormulaRevision::Rev5 {
            use crate::feature_defs::family_slots;
            use crate::feature_set_id::ComputeToken;
            let want = family_slots(ComputeToken::Basic, config.num_scales)
                .union(&family_slots(ComputeToken::Peaks, config.num_scales))
                .union(&family_slots(ComputeToken::V2, config.num_scales));
            Some(crate::feature_plan::Plan::derive(&want, 720).map_err(|_| {
                ZensimError::ModelLoadFailed {
                    reason: "cannot derive the full supported Rev5 extraction plan",
                }
            })?)
        } else {
            None
        };
        let result = compute_with_config_inner(
            source,
            distorted,
            &config,
            params.weights,
            self.stop_ref(),
            self.fold_engine,
            // Research extraction requests every supported slot, independently
            // of the profile's serving read set.
            extraction_plan.as_ref(),
        );
        self.check_stop()?;
        Ok(result.with_profile(self.profile))
    }
}

#[cfg(feature = "classification")]
impl Zensim {
    /// Compare source and distorted images with full error classification.
    ///
    /// Returns a [`ClassifiedResult`] containing the standard zensim score,
    /// pixel-level delta statistics, and error type classification.
    ///
    /// The `result.score()` is identical to what `compute()` returns — classification
    /// is a separate analysis pass that doesn't affect the score.
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError`] if dimensions are mismatched or too small.
    pub fn classify(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
    ) -> Result<ClassifiedResult, ZensimError> {
        validate_pair(source, distorted)?;

        // Compute delta stats (pixel-level analysis in sRGB space).
        // Returns `ZensimError::UnsupportedPixelFormat` if either input
        // uses a `PixelFormat` the delta-stats extractor doesn't handle.
        let delta_stats = crate::streaming::compute_delta_stats(source, distorted)?;

        // Compute the standard zensim score
        let result = self.compute(source, distorted)?;

        // Derive classification from delta stats and zensim features
        let classification = derive_classification(&delta_stats, &result);

        Ok(ClassifiedResult {
            result,
            classification,
            delta_stats,
        })
    }
}

#[cfg(feature = "training")]
impl Zensim {
    /// Compute with explicit custom params (for training).
    pub fn compute_with_params(
        params: &ProfileParams,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
    ) -> Result<ZensimResult, ZensimError> {
        validate_pair(source, distorted)?;
        let config = config_from_params(params, true);
        check_profile_route(params, &config)?;
        let serving_plan = if config.revision() >= crate::feature_defs::FormulaRevision::Rev5 {
            crate::fold_engine::score_plan(params, &config, true)
        } else {
            None
        };
        let result = compute_with_config_inner(
            source,
            distorted,
            &config,
            params.weights,
            None,
            false,
            serving_plan.as_ref(),
        );
        Ok(result)
    }
}

/// Derive error classification from pixel-level delta statistics.
///
/// Uses only 3 provable detectors with mathematically defensible signatures:
/// 1. **RoundingError** — max delta ≤ 3/255, based on `pixels_differing_by_more_than_1`
/// 2. **ChannelSwap** — one zero-delta channel with large deltas in others
/// 3. **AlphaCompositing** — opaque unchanged, semitransparent changed (tightened)
///
/// No `Mixed` category — highest score wins, or `Unclassified`.
#[cfg(feature = "classification")]
fn derive_classification(delta_stats: &DeltaStats, _result: &ZensimResult) -> ErrorClassification {
    let mut rounding_bias: Option<RoundingBias> = None;

    // Track per-detector scores internally
    let mut score_rounding = 0.0f64;
    let mut score_swap = 0.0f64;
    let mut score_alpha = 0.0f64;

    // If images are identical, short circuit
    if delta_stats.pixels_differing == 0 {
        return ErrorClassification {
            dominant: ErrorCategory::Identical,
            confidence: 1.0,
            rounding_bias: None,
        };
    }

    let max_delta = delta_stats
        .max_abs_delta
        .iter()
        .copied()
        .fold(0.0f64, f64::max);

    // === 1. Rounding error: based on max_delta + pixels_differing_by_more_than_1 ===
    //
    // If no pixel in any channel exceeds 1/255 delta, this is provably off-by-1.
    // The only operations that produce max_delta ≤ 3/255 are: integer rounding
    // mode differences, sRGB LUT precision, float→int truncation.
    if delta_stats.pixels_differing_by_more_than_1 == 0 {
        score_rounding = 1.0;
    } else if max_delta <= 2.0 / 255.0 {
        score_rounding = 0.95;
    } else if max_delta <= 3.0 / 255.0 {
        score_rounding = 0.9;
    }

    // === 2. Channel swap: one zero-delta channel, others large ===
    //
    // The only way to get one channel with zero delta and others with large
    // deltas is a channel swap. No other operation produces this pattern.
    let mut zero_channels = 0u32;
    let mut hot_channels = 0u32;
    for ch in 0..3 {
        if delta_stats.max_abs_delta[ch] < 1.0 / 255.0 {
            zero_channels += 1;
        }
        if delta_stats.max_abs_delta[ch] > 0.1 {
            hot_channels += 1;
        }
    }
    if zero_channels == 1 && hot_channels >= 1 && max_delta > 0.05 {
        score_swap = 0.9;
    }

    // === 3. Alpha compositing: tightened thresholds ===
    //
    // Stratification: opaque pixels unchanged, semitransparent changed.
    // Tightened from 0.01→0.005 opaque threshold, 0.7→0.8 correlation threshold.
    if let Some(ref opaque) = delta_stats.opaque_stats
        && let Some(ref semi) = delta_stats.semitransparent_stats
    {
        let opaque_max = opaque.mean_abs_delta.iter().copied().fold(0.0f64, f64::max);
        let semi_mean = semi.mean_abs_delta.iter().copied().fold(0.0f64, f64::max);
        if opaque_max < 0.005 && semi_mean > 0.02 && semi.pixel_count > 100 {
            score_alpha = 0.9;
        }
    }
    if let Some(corr) = delta_stats.alpha_error_correlation
        && corr > 0.8
    {
        score_alpha = score_alpha.max(corr);
    }

    // === Determine dominant category ===
    // Highest score wins. No Mixed category.
    let scores = [
        (ErrorCategory::RoundingError, score_rounding),
        (ErrorCategory::ChannelSwap, score_swap),
        (ErrorCategory::AlphaCompositing, score_alpha),
    ];

    // NaN-safe: `total_cmp` orders all f64 values (including NaN) without
    // panicking. If any score is NaN, it sorts to the most-negative end
    // and is therefore never selected as the max — the `best_score > 0.0`
    // check below then routes to `Unclassified`.
    let (best_cat, best_score) = scores
        .iter()
        .copied()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("scores is a 3-element array — max is always defined");

    let (dominant, confidence) = if best_score > 0.0 {
        (best_cat, best_score)
    } else {
        (ErrorCategory::Unclassified, 0.0)
    };

    // === Rounding bias analysis ===
    // When RoundingError is detected, analyze the signed small-delta histogram
    // to determine if errors are balanced (+/-) or systematic (one direction).
    if dominant == ErrorCategory::RoundingError {
        rounding_bias = Some(compute_rounding_bias(delta_stats));
    }

    ErrorClassification {
        dominant,
        confidence,
        rounding_bias,
    }
}

/// Compute rounding bias from the signed small-delta histogram.
///
/// Examines the +1/-1, +2/-2, +3/-3 bins per channel to determine whether
/// errors are balanced (unbiased rounding) or systematic (truncation/floor).
#[cfg(feature = "classification")]
fn compute_rounding_bias(delta_stats: &DeltaStats) -> RoundingBias {
    let h = &delta_stats.signed_small_histogram;
    let mut positive_fraction = [0.5f64; 3];
    let mut all_balanced = true;

    for ch in 0..3 {
        // Count positive deltas (+1, +2, +3) and negative deltas (-1, -2, -3)
        let neg = h[ch][0] + h[ch][1] + h[ch][2]; // bins -3, -2, -1
        let pos = h[ch][4] + h[ch][5] + h[ch][6]; // bins +1, +2, +3
        let total_nonzero = neg + pos;

        if total_nonzero == 0 {
            // No differing pixels in this channel — perfectly balanced
            positive_fraction[ch] = 0.5;
            continue;
        }

        positive_fraction[ch] = pos as f64 / total_nonzero as f64;

        // Statistical test: for balanced rounding, we'd expect ~50% positive.
        // With N trials and p=0.5, the standard deviation is sqrt(N)/2.
        // Use a 3-sigma threshold: if |pos_frac - 0.5| > 3 * 0.5 / sqrt(N),
        // consider it unbalanced. But also require a minimum absolute skew
        // (at least 60/40 split) to avoid flagging trivially small deviations
        // in large samples.
        let n = total_nonzero as f64;
        let expected_std = 0.5 / n.sqrt();
        let deviation = (positive_fraction[ch] - 0.5).abs();
        if deviation > 3.0 * expected_std && deviation > 0.1 {
            all_balanced = false;
        }
    }

    RoundingBias {
        positive_fraction,
        balanced: all_balanced,
    }
}

/// Reject HDR-flagged input before it can silently flow through the
/// SDR XYB pipeline and produce meaningless scores.
///
/// Every public entry point (compute, compute_with_ref,
/// precompute_reference, compute_with_ref_into, compute_with_params,
/// compute_extended_features, compute_all_features) calls this
/// either directly or via [`validate_pair`]. The guard is centralised
/// so a future HDR-aware code path can flip it in one place by
/// dispatching to the PU-encoded XYB front-end instead of erroring.
///
/// HDR pairs are scored via [`Zensim::compute_pu_linear`]
/// (absolute-luminance linear RGB); trained-bake calibration is
/// tracked in imazen/zensimx#38. [`ImageSource::is_hdr`] exists
/// specifically to make this refusal possible — without the signal,
/// HDR pixels in a `LinearF32Rgba` source are indistinguishable from
/// SDR and would be silently clamped.
#[inline]
/// Extract interleaved linear RGB f32 (absolute nits) from a
/// descriptor-flagged HDR source. `LinearF32Rgba` is a CONTAINER, not an
/// HDR signal — SDR content ships in it too and flows the SDR pipeline;
/// only `is_hdr()` routes here. On flagged sources the format is validated:
/// nits need f32 linear, so an `is_hdr()` source in a u8/u16 sRGB format is
/// a self-contradictory descriptor and errors rather than guessing.
fn nits_rgb_from_hdr_source(src: &impl ImageSource) -> Result<Vec<f32>, ZensimError> {
    if src.pixel_format() != crate::source::PixelFormat::LinearF32Rgba
        || src.alpha_mode() != crate::source::AlphaMode::Opaque
    {
        return Err(ZensimError::HdrInputRequiresPuPath);
    }
    let (w, h) = (src.width(), src.height());
    let mut out = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        let row = src.row_bytes(y);
        let pixels = crate::source::packed_row::<[f32; 4]>(row, w);
        let px: &[f32] = bytemuck::cast_slice(&pixels);
        for x in 0..w {
            let mut pixel = [px[x * 4], px[x * 4 + 1], px[x * 4 + 2]];
            crate::color::apply_gamut_matrix(
                &mut pixel,
                src.color_primaries(),
                crate::source::GamutMapping::Preserve,
            );
            out.extend_from_slice(&pixel);
        }
    }
    Ok(out)
}

pub(crate) fn reject_hdr_input(source: &impl ImageSource) -> Result<(), ZensimError> {
    if source.is_hdr() {
        return Err(ZensimError::HdrInputRequiresPuPath);
    }
    Ok(())
}

pub(crate) fn validate_pair(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
) -> Result<(), ZensimError> {
    validate_pair_dims(source, distorted)?;
    // Refuse HDR-flagged input on either side — a mixed SDR/HDR pair is
    // just as unscorable by the SDR pipeline as an all-HDR one.
    reject_hdr_input(source)?;
    reject_hdr_input(distorted)?;
    Ok(())
}

/// [`validate_pair`]'s dimension/zero checks WITHOUT the HDR refusal —
/// for the folded/append streaming entries, which ROUTE declared-HDR
/// pairs to the PU front-end instead of refusing them (HDR_PLAN chunk 2).
/// Every other entry keeps [`validate_pair`]'s refusal.
pub(crate) fn validate_pair_dims(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
) -> Result<(), ZensimError> {
    // Empty images are unscoreable; sub-64px images are now handled by
    // reflect-padding in `compute_with_config_inner` (scores down to 1×1),
    // so they are NOT rejected here anymore.
    if source.width() == 0 || source.height() == 0 {
        return Err(ZensimError::ImageTooSmall);
    }
    if source.width() != distorted.width() || source.height() != distorted.height() {
        return Err(ZensimError::DimensionMismatch);
    }
    Ok(())
}

/// Validate that `distorted` matches the dimensions baked into `precomputed`.
///
/// A precomputed reference is built from a specific source's `width × height`
/// (recorded on the struct via
/// [`PrecomputedReference::width`](crate::PrecomputedReference::width) /
/// [`height`](crate::PrecomputedReference::height)). Reusing it against a
/// distorted image with different dims would either silently produce garbage
/// scores (smaller padded width misaligns reference rows) or panic on slice
/// out-of-range. Reject such calls up front.
pub(crate) fn validate_ref_match(
    precomputed: &crate::streaming::PrecomputedReference,
    distorted: &impl ImageSource,
) -> Result<(), ZensimError> {
    if precomputed.sampling.is_some() {
        return Err(ZensimError::ModelLoadFailed {
            reason: "sampling-bound cache requires its BakeScorer surface",
        });
    }
    validate_ref_dimensions(precomputed, distorted)
}

pub(crate) fn validate_cached_revision(
    precomputed: &crate::PrecomputedReference,
    revision: crate::feature_defs::FormulaRevision,
) -> Result<(), ZensimError> {
    if crate::featcanon::mode(precomputed.formula_revision) != crate::featcanon::mode(revision) {
        return Err(ZensimError::ModelLoadFailed {
            reason: "reference cache color conversion revision differs from scorer; precompute with this profile",
        });
    }
    Ok(())
}

pub(crate) fn validate_ref_dimensions(
    precomputed: &crate::streaming::PrecomputedReference,
    distorted: &impl ImageSource,
) -> Result<(), ZensimError> {
    if precomputed.width() != distorted.width() || precomputed.height() != distorted.height() {
        return Err(ZensimError::DimensionMismatch);
    }
    Ok(())
}

/// Reject `width × height` if it overflows `usize` on the current target,
/// exceeds the configured `max_pixels` cap, or if the padded plane size
/// (`pyramid_plane_stride(width) × height`) would overflow during downstream
/// allocation.
///
/// Centralizes the overflow guard so 32-bit / wasm32 builds (where
/// `usize = u32`) cannot wrap silently inside downstream allocation math.
pub(crate) fn check_within_max_pixels(
    width: usize,
    height: usize,
    max_pixels: Option<usize>,
) -> Result<(), ZensimError> {
    let pixels = width
        .checked_mul(height)
        .ok_or(ZensimError::ImageTooLarge)?;
    if let Some(cap) = max_pixels
        && pixels > cap
    {
        return Err(ZensimError::ImageTooLarge);
    }
    // Also confirm the padded plane size fits. `pyramid_plane_stride` rounds
    // up to a multiple of 16 (and may add another 16 above 512 wide), so
    // `padded_width * height` may overflow even when `width * height` did
    // not. Guarding here is cheaper than threading checked-arithmetic
    // through every internal allocation site.
    let padded = crate::blur::pyramid_plane_stride(width);
    crate::blur::checked_padded_plane_len(padded, height)?;
    Ok(())
}

/// Check if source and distorted images have byte-identical pixel data
/// and matching color interpretation (format + primaries).
pub(crate) fn images_byte_identical(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
) -> bool {
    use crate::source::{AlphaMode, PixelFormat};

    let (w, h) = (source.width(), source.height());
    if w != distorted.width() || h != distorted.height() {
        return false;
    }
    if source.pixel_format() != distorted.pixel_format() {
        return false;
    }
    // Different primaries mean different perceptual colors even with identical bytes.
    if source.color_primaries() != distorted.color_primaries() {
        return false;
    }
    let fmt = source.pixel_format();
    let bpp = fmt.bytes_per_pixel();
    let row_len = w * bpp;

    // For RGBA formats with non-opaque alpha: pixels where both have A=0
    // composite to the same background, so they're visually identical
    // regardless of their RGB values.
    let alpha_aware = fmt.has_alpha()
        && !matches!(source.alpha_mode(), AlphaMode::Opaque)
        && !matches!(distorted.alpha_mode(), AlphaMode::Opaque);

    for y in 0..h {
        let sr = source.row_bytes(y);
        let dr = distorted.row_bytes(y);
        if sr[..row_len] == dr[..row_len] {
            continue; // fast path: row is byte-identical
        }
        if !alpha_aware {
            return false;
        }
        // Slow path: check pixel-by-pixel, skipping A=0 pairs
        match fmt {
            PixelFormat::Srgb8Rgba | PixelFormat::Srgb8Bgra => {
                for x in 0..w {
                    let o = x * 4;
                    if sr[o + 3] == 0 && dr[o + 3] == 0 {
                        continue;
                    }
                    if sr[o..o + 4] != dr[o..o + 4] {
                        return false;
                    }
                }
            }
            PixelFormat::Srgb16Rgba => {
                for x in 0..w {
                    let o = x * 8;
                    let sa = u16::from_ne_bytes([sr[o + 6], sr[o + 7]]);
                    let da = u16::from_ne_bytes([dr[o + 6], dr[o + 7]]);
                    if sa == 0 && da == 0 {
                        continue;
                    }
                    if sr[o..o + 8] != dr[o..o + 8] {
                        return false;
                    }
                }
            }
            PixelFormat::LinearF32Rgba => {
                for x in 0..w {
                    let o = x * 16;
                    let sa = f32::from_ne_bytes([sr[o + 12], sr[o + 13], sr[o + 14], sr[o + 15]]);
                    let da = f32::from_ne_bytes([dr[o + 12], dr[o + 13], dr[o + 14], dr[o + 15]]);
                    if sa <= 0.0 && da <= 0.0 {
                        continue;
                    }
                    if sr[o..o + 16] != dr[o..o + 16] {
                        return false;
                    }
                }
            }
            _ => return false,
        }
    }
    true
}

/// Minimum per-dimension size for the full 4-scale pyramid (8·2³ = 64).
pub(crate) const MIN_PYRAMID_DIM: usize = min_pyramid_dim_for_scales(crate::NUM_SCALES);

/// Smallest per-dimension size that still forms `num_scales` pyramid levels.
///
/// The scale walk (`streaming::compute_multiscale_stats_streaming` and every
/// `*_with_ref` sibling) halves `w`/`h` per level and stops at `w < 8 ||
/// h < 8`, so level `num_scales-1` needs `8 · 2^(num_scales-1)` at level 0.
/// At the shipped `NUM_SCALES = 4` this is 64 — the historical
/// [`MIN_PYRAMID_DIM`] constant, unchanged.
#[inline]
pub(crate) const fn min_pyramid_dim_for_scales(num_scales: usize) -> usize {
    // `num_scales` comes from a profile / `ZensimConfig` (1..=6 in practice);
    // saturate rather than shift-overflow on a nonsense value.
    match num_scales {
        0 | 1 => 8,
        n if n <= 32 => 8usize << (n - 1),
        _ => usize::MAX,
    }
}

/// **THE single owner of the "must reflect-pad before the scale walk"
/// decision.** Every v1 entry point that reaches the pyramid asks this.
///
/// A pair that answers `true` and is NOT padded produces a SHORT feature
/// vector (fewer `ScaleStats` → `n_scales · 3 · 31` instead of 372) or, on
/// the `*_with_ref` paths, trips the `scale 0 width mismatch` assertion
/// against an already-padded reference. Both were live defects until
/// 2026-08-30 — see `docs/DATASET_HISTORY.md` §3.26 and
/// `zensim/tests/v1_feature_width_pure_function.rs`.
///
/// Note the asymmetry this hides: the walk starts at
/// `pyramid_plane_stride(width)` but plain `height`, so before the padding was
/// applied `54x96` came out full-width (54 → 64 by SIMD alignment) while
/// `96x54` came out short. Callers must not re-derive the rule.
#[inline]
pub(crate) fn needs_pyramid_pad(width: usize, height: usize, num_scales: usize) -> bool {
    let min_dim = min_pyramid_dim_for_scales(num_scales);
    width > 0 && height > 0 && (width < min_dim || height < min_dim)
}

/// Owned, tightly-packed image presenting a reflect(mirror)-padded view of a
/// sub-64px input so the multi-scale pyramid always forms. This is the
/// bake-/metric-agnostic small-image fix: the pipeline computes genuine scales
/// over reflected content (a constant difference scores identically at every
/// size; scores down to 1×1 with no error) instead of synthesizing a feature
/// vector. NO-OP for images already ≥ 64px in both dims.
pub(crate) struct OwnedImage {
    data: Vec<u8>,
    width: usize,
    height: usize,
    format: crate::source::PixelFormat,
    alpha: crate::source::AlphaMode,
    primaries: crate::source::ColorPrimaries,
    gamut_mapping: crate::source::GamutMapping,
}

/// Preserve absolute linear-sRGB nits when adapting the existing raw APIs
/// to the typed fold front end. These legacy entries materialize an opaque
/// RGBA image; the ImageSource/BakeScorer HDR entries remain row streamed.
fn nits_image(
    width: usize,
    height: usize,
    pixel: impl Fn(usize, usize, usize) -> f32,
) -> Result<OwnedImage, ZensimError> {
    let bytes = width
        .checked_mul(height)
        .and_then(|v| v.checked_mul(16))
        .ok_or(ZensimError::ImageTooLarge)?;
    let mut data = Vec::new();
    data.try_reserve_exact(bytes)
        .map_err(|_| ZensimError::ImageTooLarge)?;
    for y in 0..height {
        for x in 0..width {
            for ch in 0..3 {
                data.extend_from_slice(&pixel(x, y, ch).to_ne_bytes());
            }
            data.extend_from_slice(&1.0f32.to_ne_bytes());
        }
    }
    Ok(OwnedImage {
        data,
        width,
        height,
        format: crate::source::PixelFormat::LinearF32Rgba,
        alpha: crate::source::AlphaMode::Opaque,
        primaries: crate::source::ColorPrimaries::Srgb,
        gamut_mapping: crate::source::GamutMapping::Preserve,
    })
}

impl ImageSource for OwnedImage {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
    fn pixel_format(&self) -> crate::source::PixelFormat {
        self.format
    }
    fn alpha_mode(&self) -> crate::source::AlphaMode {
        self.alpha
    }
    fn color_primaries(&self) -> crate::source::ColorPrimaries {
        self.primaries
    }
    fn gamut_mapping(&self) -> crate::source::GamutMapping {
        self.gamut_mapping
    }
    fn row_bytes(&self, y: usize) -> &[u8] {
        let stride = self.width * self.format.bytes_per_pixel();
        &self.data[y * stride..y * stride + stride]
    }
}

/// reflect-101 index map (mirror without repeating the edge); `n == 1` → 0.
#[inline]
pub(crate) fn reflect_index(i: usize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let mut k = i % period;
    if k >= n {
        k = period - k;
    }
    k
}

/// Reflect(mirror)-pad `src` up to at least [`MIN_PYRAMID_DIM`] in each dim.
pub(crate) fn reflect_pad_to_min(src: &impl ImageSource) -> OwnedImage {
    reflect_pad_for_scales(src, crate::NUM_SCALES)
}

/// Reflect(mirror)-pad `src` up to [`min_pyramid_dim_for_scales`] in each dim.
pub(crate) fn reflect_pad_for_scales(src: &impl ImageSource, num_scales: usize) -> OwnedImage {
    reflect_pad_to_size(src, min_pyramid_dim_for_scales(num_scales))
}

pub(crate) fn reflect_pad_to_size(src: &impl ImageSource, min_dim: usize) -> OwnedImage {
    let (w, h) = (src.width(), src.height());
    let bpp = src.pixel_format().bytes_per_pixel();
    let (bw, bh) = (w.max(min_dim), h.max(min_dim));
    let mut data = vec![0u8; bw * bh * bpp];
    for y in 0..bh {
        let srow = src.row_bytes(reflect_index(y, h));
        for x in 0..bw {
            let sx = reflect_index(x, w) * bpp;
            let dx = (y * bw + x) * bpp;
            data[dx..dx + bpp].copy_from_slice(&srow[sx..sx + bpp]);
        }
    }
    OwnedImage {
        data,
        width: bw,
        height: bh,
        format: src.pixel_format(),
        alpha: src.alpha_mode(),
        primaries: src.color_primaries(),
        gamut_mapping: src.gamut_mapping(),
    }
}

/// **The shared front of every SDR scoring entry**, with the engine selector
/// (fold-engine lane, 2026-08-30). The reflect-pad and the byte-identical
/// short-circuit run BEFORE either walk, so both engines see exactly the same
/// pixels and the same identical-pair payload. Only the walk differs.
///
/// `use_fold == false` is the pre-2026-08-30 path, byte-for-byte.
fn compute_with_config_inner(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    config: &ZensimConfig,
    weights: &[f64],
    stop: Option<&dyn enough::Stop>,
    use_fold: bool,
    // The serving plan, resolved by the CALLER (only `Zensim` knows the
    // profile's bakes). `None` = today's unconditional behaviour.
    // Deliberately a resolved plan rather than a flag: the decision belongs
    // to `fold_engine::score_plan`, and passing the answer keeps this shared
    // front ignorant of bakes.
    plan: Option<&crate::feature_plan::Plan>,
) -> ZensimResult {
    // Sub-64px inputs can't form the 4-scale pyramid. Reflect-pad both sides
    // to the pyramid minimum so every metric/bake gets 4 genuinely-computed
    // scales. NO-OP at ≥ 64px in both dims (the common path is untouched).
    let (w, h) = (source.width(), source.height());
    if needs_pyramid_pad(w, h, config.num_scales) {
        let s = reflect_pad_for_scales(source, config.num_scales);
        let d = reflect_pad_for_scales(distorted, config.num_scales);
        return compute_with_config_core(&s, &d, config, weights, stop, use_fold, plan);
    }
    compute_with_config_core(source, distorted, config, weights, stop, use_fold, plan)
}

fn compute_with_config_core(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    config: &ZensimConfig,
    weights: &[f64],
    stop: Option<&dyn enough::Stop>,
    use_fold: bool,
    plan: Option<&crate::feature_plan::Plan>,
) -> ZensimResult {
    // Rev5 identity features are computed by the same canonical walk as
    // nonidentity pairs. The canonical walk checks a live stop token between
    // strips; public entries discard partial results when cancellation fires.
    if config.revision() >= crate::feature_defs::FormulaRevision::Rev5
        && crate::fold_engine::is_fold_backable(config)
    {
        let mut scratch = crate::feature_v2::V2Scratch::new();
        if let Ok(mut result) = crate::fold_engine::compute_fold_backed_stoppable(
            source,
            distorted,
            config,
            weights,
            &mut scratch,
            plan,
            stop,
        ) {
            if images_byte_identical(source, distorted) {
                result.score = 100.0;
                result.raw_distance = 0.0;
                result = result.mark_identical();
            }
            return result;
        }
    }

    // Identical images must score exactly 100.0 — short-circuit before
    // floating-point arithmetic introduces sub-ULP noise in SSIM/edge features.
    if images_byte_identical(source, distorted) {
        {
            // An identity result must be as wide as the bake that will read
            // it, or a wide bake is refused on a PERFECT COPY — the one pair
            // every metric must handle. The extra slots are zeros either way;
            // what changes is the length.
            return identical_result_at(config, plan.map_or(0, |p| p.walk_width()));
        }
    }

    // A live cancellation token keeps the request on buffered: issue #48's
    // contract is cooperative cancellation INSIDE the walk (row-band and
    // scale boundaries) and the fold has no stop hook, so routing a
    // stoppable request to it would silently downgrade cancellation to the
    // caller's before/after `check_stop`. Named in `is_fold_backable`'s
    // caller rather than hidden.
    {
        // A plan whose layout is wider than the buffered walk can emit MUST
        // take the fold: falling back to buffered there is not a safe default,
        // it is a refusal. `use_fold` still governs the narrow case, where both
        // walks can serve and buffered is the default for every profile
        // except `D`.
        //
        // The whole decision is one `cfg` BLOCK rather than an attribute on
        // the `if`: an attribute placed before a `let` binds to the `let`,
        // which silently leaves the `if` unconditional on a
        // `--no-default-features` build (caught by zensim-experimental, which
        // builds zensim without this feature).
        let plan_needs_fold =
            plan.is_some_and(|p| p.walk_width() > crate::fold_engine::v1_feature_width(config));
        if (use_fold || plan_needs_fold)
            && stop.is_none()
            && crate::fold_engine::is_fold_backable(config)
        {
            let mut scratch = crate::feature_v2::V2Scratch::new();
            // The fold's only failure modes here are the dimension/HDR checks
            // the caller already made; a surprise falls back rather than
            // erroring out of a path whose signature is infallible.
            if let Ok(r) = crate::fold_engine::compute_fold_backed_stoppable(
                source,
                distorted,
                config,
                weights,
                &mut scratch,
                plan,
                stop,
            ) {
                return r;
            }
        }
    }

    crate::streaming::compute_zensim_streaming_stoppable(source, distorted, config, weights, stop)
}

/// The identical-pair short-circuit payload, shared by the SDR byte-identical
/// check and the PU entries' element-identical checks: `(score=100,
/// raw_distance=0, all-zero features)`, flagged via `mark_identical` so
/// `apply_mlp_scoring_with_codec` skips the MLP forward pass (which has no
/// signal anchoring "zero feature vector → 100" and would overwrite the
/// payload with garbage).
///
/// The feature width depends on which flags are enabled: basic
/// (228 = 4 × 3 × 19) is always present, extended adds 72 masked features
/// (300 total), and compute_iw_features adds 72 IW-pool features (372
/// total). On identical inputs every feature is zero (all error metrics
/// are 0). PreviewV0_5Compression (V_22-372feat) needs the full 372-width
/// vector; PreviewV0_5 / PreviewV0_5Balanced (V_22-mix-LARGE+iwssim,
/// 300-input) needs only the extended 300-width vector. A missing block
/// triggers InvalidDataLength downstream when the bake's MLP runs.
fn identical_result(config: &ZensimConfig) -> ZensimResult {
    identical_result_at(config, 0)
}

/// [`identical_result`] widened to at least `min_width` slots.
///
/// A perfect copy is the one pair every metric must handle, and a wide bake
/// was refused on it for the same reason it was refused everywhere else: the
/// vector handed to `prep_bake_input_f32` was the v1 width. The extra slots
/// are zeros either way — what changes is the LENGTH, which is what the bake
/// checks. `min_width = 0` is exactly today's behaviour.
fn identical_result_at(config: &ZensimConfig, min_width: usize) -> ZensimResult {
    let fpc = match (config.extended_features, config.compute_iw_features) {
        (true, true) => FEATURES_PER_CHANNEL_EXTENDED + FEATURES_PER_CHANNEL_IW,
        (true, false) => FEATURES_PER_CHANNEL_EXTENDED,
        (false, _) => FEATURES_PER_CHANNEL_WITH_PEAKS,
    };
    let num_features = (config.num_scales * 3 * fpc).max(min_width);
    ZensimResult::new(
        100.0,
        0.0,
        vec![0.0; num_features],
        // Placeholder tag (overridden by the compute* caller); the
        // canonical codec-target profile, never the deprecated `A`.
        ZensimProfile::codec_target(),
        [0.0; 3],
    )
    .mark_identical()
}

// Named Rev5 models own their declared plans. Raw research configs retain the process-mix guard.
pub(crate) fn check_profile_route(
    params: &ProfileParams,
    config: &ZensimConfig,
) -> Result<(), ZensimError> {
    if params.mlp_bytes.is_some()
        && config
            .formula_revision
            .is_some_and(|r| r >= crate::feature_defs::FormulaRevision::Rev5)
        && crate::fold_engine::is_fold_backable(config)
    {
        return Ok(());
    }
    crate::ssim_form::check_route(config)
}

pub(crate) fn config_from_params(params: &ProfileParams, parallel: bool) -> ZensimConfig {
    ZensimConfig {
        formula_revision: params
            .mlp_bytes
            .and_then(|load| cached_profile_model(load()).ok())
            .and_then(|model| crate::feature_layout::formula_revision(&model).ok())
            .filter(|revision| *revision >= crate::feature_defs::FormulaRevision::Rev5),
        local_only: false,
        omit_edges: false,
        attribution_channels: None,
        blur_radius: params.blur_radius,
        blur_passes: params.blur_passes,
        blur_kernel: BlurKernel::Box {
            passes: params.blur_passes,
        },
        downscale_filter: DownscaleFilter::default(),
        // MLP-scored profiles need the full feature vector populated;
        // the linear path can skip features it doesn't use.
        compute_all_features: params.mlp_bytes.is_some(),
        extended_features: params.extended_features,
        extended_masking_strength: 4.0,
        compute_iw_features: params.compute_iw_features,
        iw_strength: 4.0,
        num_scales: params.num_scales,
        score_mapping_a: params.score_mapping_a,
        score_mapping_b: params.score_mapping_b,
        allow_multithreading: parallel,
        bounded_squash: params.bounded_squash,
    }
}

/// Logistic soft-clamp from raw score into the `[0, 100]` range.
///
/// `f(x) = 100 / (1 + exp(-(x - 50) / 20))`.
///
/// The scale parameter `20` gives a gentle squash centered at 50; at
/// the band centers the deviation from `x` is small but NON-ZERO:
///
/// | raw | f(raw) | Δ |
/// |---:|---:|---:|
/// | -1000 | 0.0     | -1000 |
/// |    0  | 7.586   | +7.6 |
/// |   25  | 28.110  | +3.1 |
/// |   50  | 50.0    | 0 |
/// |   75  | 71.890  | -3.1 |
/// |  100  | 92.414  | -7.6 |
/// | 1000  | 100.0   | -900 |
///
/// The interior shift is the cost of the formula: a profile that
/// enables `soft_clamp_score = true` ships a recalibration alongside
/// the rank-preservation fix. Downstream consumers that read a fixed
/// score-to-codec-parameter table will see a few-percentage-point
/// shift in the mapped codec settings.
///
/// **The function strictly preserves rank order globally** (monotone
/// increasing for all finite `x`), so SROCC / KROCC / PWRC are
/// invariant under the transform. PLCC and Z-RMSE shift modestly
/// because they're calibration-sensitive.
///
/// **Why the transform helps**: hard `raw.clamp(0, 100)` creates tie
/// blocks at exactly 0 or 100 when many out-of-range raw values
/// collapse to the boundary. The PreviewV0_4 multi-bake at α=0.4
/// raw-space sees this on heavy-distortion pairs (V_20 IS B3
/// specialist extrapolates past 100) and the resulting tie block
/// collapses SROCC to 0 on the affected band (observed on TID B0/B1
/// 2026-05-15). Soft-clamp distributes those raw values uniquely
/// into the high tail, so SROCC stays defined.
///
/// Cost: one `exp` call (~1 ns on modern hardware). Used when
/// `ProfileParams::soft_clamp_score` is `true`.
#[inline]
fn soft_clamp_score(raw: f64) -> f64 {
    100.0 / (1.0 + (-(raw - 50.0) / 20.0).det_exp(active_pow_form()))
}

/// Replace `result.raw_distance` and `result.score` with the MLP forward
/// pass output when the profile uses an MLP scorer. No-op for linear
/// profiles.
///
/// When the loaded MLP's `n_inputs` exceeds `result.features().len()`
/// by exactly 4, this function appends size-axis features derived
/// from the source `(width, height)` before scoring. This matches
/// the V0_4 trainer's `--mlp-size-axes` mode (228 → 232 inputs):
/// log2(pixels), log2(min_dim), log2(max_dim), and signed
/// log2(max/min) by aspect orientation.
pub(crate) fn apply_mlp_scoring(
    result: &mut ZensimResult,
    params: &crate::profile::ProfileParams,
    width: u32,
    height: u32,
) -> Result<(), ZensimError> {
    apply_mlp_scoring_with_codec(result, params, width, height, None)
}

/// Same as [`apply_mlp_scoring`] but accepts an optional codec hint
/// that drives the per-codec post-spline affine calibration
/// (EXP-CROSS-CODEC-V11-E). The hint is threaded into every
/// `forward_one_bake_with_codec` call below — both the
/// ensemble-routing branches and the primary/b3 mix branch — so
/// every loaded bake gets a chance to apply per-codec affine if
/// it carries the metadata.
pub(crate) fn apply_mlp_scoring_with_codec(
    result: &mut ZensimResult,
    params: &crate::profile::ProfileParams,
    width: u32,
    height: u32,
    codec_hint: Option<&str>,
) -> Result<(), ZensimError> {
    let Some(loader) = params.mlp_bytes else {
        return Ok(());
    };
    // A built-in profile's bake declares no revision, so it IS revision 1.
    // `ZENSIM_FORMULA_REV` pins the PIXELS, not the coefficients — pinning a
    // research revision and reading a profile score therefore prices revision-1
    // weights against another era's features. `BakeScorer` refuses that
    // outright; this path cannot, because the same call also produces the
    // features a research extraction is there to collect. So it says so, once.
    if config_from_params(params, false).formula_revision.is_none() {
        crate::ssim_form::warn_pinned_revision_scoring_once();
    }
    // **Score-disposition guard (D9).** A bake carrying
    // `zentrain.output_calibration_spline` emits an already-calibrated
    // 0-100 score. Scoring it with `skip_score_mapping == false` applies
    // the legacy `100 - A*d^B` DISTANCE mapping on top of that spline,
    // sending every realistic score deeply negative; the default hard
    // clamp then pins it to exactly 0.0. The observed signature is a
    // model that looks catastrophically broken: identity 100, the very
    // worst quality point scoring ~9, and every other point exactly
    // 0.000000 (ADD156 ship audit 2026-08-31, defect D9 — reproduced
    // first try, because the naive builder configuration IS the wrong
    // one: `ProfileParams::builder()` defaults BOTH `skip_score_mapping`
    // and `extrapolate_score` to `false`).
    //
    // Silently-wrong pixels/scores are a shipping bug, never a tolerated
    // tradeoff, so we refuse the combination instead of returning the
    // zero. The check runs only when `skip_score_mapping == false`, which
    // no spline-carrying profile in this crate sets, so the shipped path
    // pays nothing.
    if !params.skip_score_mapping {
        check_score_disposition(params, loader)?;
    }
    // **Identity-image short-circuit guard.** When the byte-identical
    // short-circuit in `compute_with_config_inner` /
    // `compute_zensim_with_config` fires, it produces
    // `(score=100.0, raw_distance=0.0, features=[0.0; N])`. Running
    // the MLP forward pass on the all-zero feature vector then
    // overwrites those values with garbage — typically ~0 on
    // V0_5Balanced (MSE bake on z-scored zeros), ~2 on
    // V0_5Compression / V0_5Ensemble — because the trained MLP has
    // no signal anchoring "zero feature vector → score 100". The
    // bake's biases dominate, producing an off-scale output that
    // `skip_score_mapping=true` returns verbatim.
    //
    // We detect the post-short-circuit state via the explicit
    // `is_identical` flag set by `mark_identical`. The pre-0.3.0 code
    // used a sentinel check (`raw_distance == 0.0 && features.all
    // (== 0.0)`); the flag is robust against any future profile that
    // legitimately produces an all-zero feature vector on a real
    // image.
    if result.is_identical() {
        return Ok(());
    }
    {
        // **Ensemble routing path** (PreviewV0_5Ensemble) — when an
        // `ensemble_classifier_bytes` is present, run the classifier
        // first and route to either the primary (`mlp_bytes`,
        // balanced) or the alternative (`mlp_bytes_compression`)
        // based on the classifier's sign. The classifier output is a
        // pre-sigmoid logit; `logit > 0` ⇔ `sigmoid(logit) > 0.5` →
        // compression.
        //
        // Both target bakes MUST accept the same input feature
        // shape; the classifier runs over the same vector (its own
        // `n_inputs` selects the prefix). Documented in
        // `benchmarks/exp_ensemble_v05_eval_2026-05-18.md`.
        let raw = if let (Some(clf_loader), Some(cmp_loader)) = (
            params.ensemble_classifier_bytes,
            params.mlp_bytes_compression,
        ) {
            // Classifier dispatch — classifier bake is a router; it
            // does NOT receive the per-codec hint (its output is a
            // pre-sigmoid logit, not a score). The routed target bake
            // (primary or compression) receives the hint and may apply
            // the per-codec affine if its metadata is present.
            let logit = forward_one_bake(clf_loader(), result.features(), width, height)?;
            if logit > 0.0 {
                forward_one_bake_with_codec(
                    cmp_loader(),
                    result.features(),
                    width,
                    height,
                    codec_hint,
                )?
            } else {
                forward_one_bake_with_codec(loader(), result.features(), width, height, codec_hint)?
            }
        } else {
            let raw_primary = forward_one_bake_with_codec(
                loader(),
                result.features(),
                width,
                height,
                codec_hint,
            )?;
            // Optional secondary bake (D2 multi-output ensemble, e.g.
            // V_20 IS B3 specialist) — its forward path runs over the
            // SAME 228-feature vector but applies its own
            // `feature_transforms` metadata. Both bakes' raw outputs
            // are mixed linearly at `mlp_primary_mix` weight on the
            // primary. The codec hint is passed identically to both.
            if let Some(b3_loader) = params.mlp_bytes_b3 {
                let raw_b3 = forward_one_bake_with_codec(
                    b3_loader(),
                    result.features(),
                    width,
                    height,
                    codec_hint,
                )?;
                let a = params.mlp_primary_mix as f64;
                a * raw_primary + (1.0 - a) * raw_b3
            } else {
                raw_primary
            }
        };

        let score = dispose_mlp_raw(raw, params);
        result.set_mlp_score(raw, score);
    }
    Ok(())
}

/// Post-forward score disposition shared by [`apply_mlp_scoring_with_codec`]
/// and the batched FD-gradient entry
/// ([`score_features_fd_gradient_with_profile`]) — extracted verbatim so the
/// two paths cannot drift (campaign appendix Y, L-Y1).
pub(crate) fn dispose_mlp_raw(raw: f64, params: &crate::profile::ProfileParams) -> f64 {
    let pre_bound = if params.skip_score_mapping {
        // The bake is already MCOS-calibrated (V0_8+); the raw
        // output IS the final score. Skipping the
        // `100 − A·d^B` transform avoids producing garbage
        // (e.g. raw=90 → mapped=-374).
        raw
    } else {
        distance_to_score_mapped(raw, params.score_mapping_a, params.score_mapping_b)
    };
    // Bound the score to [0, 100] before handing it back. Two
    // policies exist:
    //   - Hard clamp (default, legacy): out-of-range outputs pin
    //     to 0 or 100. Cheap, MCOS-natural. Tie blocks at the
    //     boundary collapse SROCC to 0 on affected bands when
    //     many pairs extrapolate (multi-bake V_20 IS regime).
    //   - Soft clamp (V_20+ multi-bake): logistic squash through
    //     `100 / (1 + exp(-(raw - 50) / 20))`. The output deviates
    //     from `raw` by < 1.5 units in the [5, 95] interior so
    //     interior calibration is unchanged; only the tails are
    //     reshaped. Preserves rank ordering at the extremes
    //     (no ties → SROCC stays defined). Costs one `exp` per
    //     score (~1 ns).
    // The choice is per-profile via `ProfileParams::soft_clamp_score`.
    // PreviewV0_4 (V_18 + V_20 IS multi-bake) sets `true`;
    // the V_18 ship single-bake and earlier set `false`.
    //
    // EXP-CROSS-CODEC-V10 (2026-05-20): `extrapolate_score` overrides
    // both hard- and soft-clamp paths. The PCHIP spline's linear
    // extrapolation past its endpoint knots flows through, so
    // "pathological" codec output (worst codec at q=0, butter > 12)
    // maps to a negative score instead of collapsing to a tie at 0.
    // V10 profiles (BalancedV3 / CompressionV3 / TunerV4) set this.
    if params.extrapolate_score {
        pre_bound
    } else if params.soft_clamp_score {
        soft_clamp_score(pre_bound)
    } else {
        pre_bound.clamp(0.0, 100.0)
    }
}

// ============================================================================
// Per-sample-α head runtime dispatch (V_24-per-sample-α bakes)
// ============================================================================
//
// Bakes trained with `train_mlp_per_sample_alpha_head_with_tv` in
// zensim-train-core attach a `zentrain.per_sample_alpha_head`
// metadata entry. The bake's final layer is a `n_hidden × n_hidden`
// identity matrix (passthrough), so the predictor's output IS the
// post-LeakyReLU hidden vector `h`. The runtime then mixes a rank
// head (`y_rank = h · rank_w + rank_b`) and a pool head (`y_pool =
// stats(h) · reducer_w + reducer_b`) via a per-sample gate
// `α = σ(h · w_α + b_α)`:
//
//     y_final = α · y_rank + (1 − α) · y_pool
//
// Payload layout (f32 little-endian):
//   [w_α[0..n_hidden]] [b_α] [rank_w[0..n_hidden]] [rank_b]
//   [reducer_w[0..4]] [reducer_b] [p_norm]
// Total size = (2·n_hidden + 8) × 4 bytes.
//
// The pool sigma floor mirrors `zensim-train-core::pool_head::POOL_STD_FLOOR`
// and is owned once by `crate::score_math::POOL_STD_FLOOR` (it used to be
// declared here TWICE, once per head, and a third time in `zensim-validate`).

/// Apply the tanh-pinned [0, 100] sigmoid wrap: `y_score = 100 · σ(y_pre / scale)`.
/// Matches `zensim_train_core::per_sample_alpha_head::bake_per_sample_alpha_head_v3_with_tanh`
/// at training time and bit-exact with the train-time pin (single-precision sigmoid
/// pin via clamp [−30, 30] in y_pre/scale, sigmoid in f64).
fn apply_tanh_output_pin(y_pre: f64, scale: f64) -> f64 {
    crate::score_math::tanh_output_pin(y_pre, scale, active_pow_form())
}

/// Evaluate the PCHIP spline at `x`, clamped to `≤ 100` on the UPPER
/// side only.
///
/// A perceptual score can never exceed 100 (nothing is more similar than
/// identical), so the upper extrapolation is clamped at 100 — this is the
/// correct-by-construction upper bound that fixes the `> 100` defect
/// (root cause: V39's degenerate spline linearly extrapolated past 100).
/// The LOWER extrapolation is allowed to go **negative** so that inputs more
/// dissimilar than a simple low-quality encode score below zero (a desired
/// property of the codec-target dial — "worse than the worst codec output" is
/// meaningful signal, not a tie at 0), but is FLOORED one calibrated-dial-range
/// below the bottom knot as an OOD safety net (a pathological raw that slips
/// past the winsor guard can't produce a wild score). The result stays monotone
/// (min/max with a constant preserves order).
fn apply_output_calibration_spline(x: f64, spline: &OutputCalibrationSpline) -> f64 {
    // ONE owner (`crate::score_math::pchip_eval_capped`), reached identically
    // from `zensim-validate::output_calibration_spline`. The mirror there had
    // the upper EXTRAPOLATION capped (2026-07-04) but not the INTERIOR, so a
    // spline declaring a knot above 100 — which the wire format permits —
    // could be published above 100 by the evaluation tooling while the product
    // runtime reported exactly 100. LATENT: 0 of the 49 bakes on disk declare
    // one. Not Hermite overshoot; Fritsch-Carlson forbids that.
    crate::score_math::pchip_eval_capped(x, &spline.xs, &spline.ys, &spline.derivs)
}

// ============================================================================
// Min-max monotone head runtime dispatch (Sill 1998; 2026-07-16)
// ============================================================================
//
// A min-max monotone bake REPLACES the MLP layer stack: the score is
// `min over K groups  max over J pieces  (w[g][h]·x + b[g][h])` on the
// standardized+transformed feature vector, with each `w` sign-constrained so
// the score is monotone in codec quality BY CONSTRUCTION (see
// `zensim_validate::mlp_train::minmax_monotone`). Because there is no matmul
// forward, the runtime BYPASSES `Predictor::predict`: it reads the bake's
// scaler + feature-transforms directly, applies transform→scale→clamp±8→min-max,
// then the shared PCHIP output spline. The ±8 clamp mirrors the trainer's
// standardize clamp (`train_minmax`) so runtime scoring is bit-exact with the
// held-out eval that qualified the bake.
//
// Payload layout (little-endian):
//   `[u32 k, u32 j, u32 n, w[k·j·n] f32, b[k·j] f32]`
// so `12 + 4·(k·j·n + k·j)` bytes. `n` MUST equal the bake's `n_inputs`.

/// Standardized-feature clamp applied before the min-max forward. Mirrors
/// `train_minmax`'s `.clamp(-8.0, 8.0)` on the standardized feature vector so
/// the shipped bake scores identically to the held-out eval that qualified it.
const MINMAX_STD_CLAMP: f64 = 8.0;

/// Min-max forward: `min_g max_h (w[g][h]·x + b[g][h])`. `x` is the clamped
/// standardized feature vector (length `meta.n`). Mirrors
/// `MinMaxMonotone::forward` (weights are f32 in the bake, accumulated in f64).
fn apply_minmax_runtime(x: &[f64], meta: &MinMaxHeadMeta) -> f64 {
    let n = meta.n;
    let mut best_min = f64::INFINITY;
    for g in 0..meta.k {
        let mut best_max = f64::NEG_INFINITY;
        for h in 0..meta.j {
            let base = (g * meta.j + h) * n;
            let mut acc = meta.b[g * meta.j + h] as f64;
            for (wv, xv) in meta.w[base..base + n].iter().zip(&x[..n]) {
                acc += *wv as f64 * xv;
            }
            if acc > best_max {
                best_max = acc;
            }
        }
        if best_max < best_min {
            best_min = best_max;
        }
    }
    best_min
}

/// Look up the per-codec affine for the given codec hint. Returns
/// `None` for unrecognized codecs (caller substitutes identity).
///
/// Match is case-insensitive and accepts common aliases (the table's
/// own name plus standard codec-family aliases). The metadata stores
/// canonical names (e.g. "jpeg", "webp", "avif", "jxl"); aliases like
/// "zenjpeg" / "mozjpeg" / "libjpeg" / "jpg" all map to "jpeg" etc.
fn lookup_per_codec_affine(cal: &PerCodecCalibration, codec_hint: &str) -> Option<(f32, f32)> {
    let lower = codec_hint.to_ascii_lowercase();
    let canon: &str = match lower.as_str() {
        "jpeg" | "jpg" | "zenjpeg" | "mozjpeg" | "libjpeg" => "jpeg",
        "webp" | "zenwebp" => "webp",
        "avif" | "zenavif" => "avif",
        "jxl" | "zenjxl" | "jpegxl" | "jpeg-xl" => "jxl",
        "png" | "zenpng" => "png",
        other => other,
    };
    for entry in &cal.entries {
        if entry.name == canon {
            return Some((entry.alpha, entry.beta));
        }
    }
    None
}

impl PerSampleAlphaMeta {
    /// Borrowed view for [`crate::score_math`], the owner of the arithmetic.
    /// Free — no allocation, no copy of either weight vector.
    fn params(&self) -> crate::score_math::PerSampleAlphaParams<'_> {
        crate::score_math::PerSampleAlphaParams {
            w_alpha: &self.w_alpha,
            b_alpha: self.b_alpha,
            rank_w: &self.rank_w,
            rank_b: self.rank_b,
            reducer_w: self.reducer_w,
            reducer_b: self.reducer_b,
            p_norm: self.p_norm,
        }
    }
}

/// Apply the per-sample-α runtime to a hidden vector `h`. Returns
/// the final mixed score `y`. Bit-exact match with
/// `zensim_train_core::per_sample_alpha_head::apply_per_sample_alpha_head_runtime`
/// (asserted by the canonical regression test in
/// `tests/per_sample_alpha_runtime.rs`).
fn apply_per_sample_alpha_runtime(h: &[f32], meta: &PerSampleAlphaMeta) -> f64 {
    debug_assert_eq!(meta.rank_w.len(), h.len());
    debug_assert_eq!(meta.w_alpha.len(), h.len());
    debug_assert!(!h.is_empty());
    // ONE owner (`crate::score_math`), reached identically from here and from
    // `zensim-validate`'s bake-evaluation runtime. The arithmetic used to live
    // in both places, bit-exact in PROSE only, with no test binding them — and
    // F19's `PowForm` made that a live wrong-number risk the moment
    // `SHIPPED_REVISION` flips. See `score_math`'s module docs.
    crate::score_math::per_sample_alpha_head(h, &meta.params(), active_pow_form())
}

// ============================================================================
// Hybrid-head runtime dispatch (V_24-hybrid bakes)
// ============================================================================
//
// Bakes trained with `train_mlp_hybrid_head_with_tv` in
// zensim-train-core attach a `zentrain.hybrid_head` metadata
// entry. Like the per-sample-α head, the bake's final layer is an
// `n_hidden × n_hidden` identity matrix (passthrough), so the
// predictor's output IS the post-LeakyReLU hidden vector `h`. The
// runtime then mixes a rank head (`y_rank = h · rank_w + rank_b`)
// and a pool head (`y_pool = stats(h) · reducer_w + reducer_b`)
// via a single LEARNED SCALAR gate `α = σ(α_logit)` (different
// from the per-sample-α head, where α is computed per-sample from
// the hidden vector):
//
//     y_final = α · y_rank + (1 − α) · y_pool
//
// Payload layout (f32 little-endian):
//   [rank_w[0..n_hidden]] [rank_b] [α_logit]
//   [reducer_w[0..4]] [reducer_b] [p_norm]
// Total size = (n_hidden + 8) × 4 bytes.
//
// The pool sigma floor is owned by `crate::score_math::POOL_STD_FLOOR`.

impl HybridHeadMeta {
    /// Borrowed view for [`crate::score_math`] — see
    /// [`PerSampleAlphaMeta::params`].
    fn params(&self) -> crate::score_math::HybridHeadParams<'_> {
        crate::score_math::HybridHeadParams {
            rank_w: &self.rank_w,
            rank_b: self.rank_b,
            alpha_logit: self.alpha_logit,
            reducer_w: self.reducer_w,
            reducer_b: self.reducer_b,
            p_norm: self.p_norm,
        }
    }
}

/// Apply the hybrid-head runtime to a hidden vector `h`. Returns the
/// final mixed score `y`. Bit-exact match with
/// `zensim_train_core::hybrid_head::apply_hybrid_head_runtime`
/// (asserted by the canonical regression test in
/// `tests/hybrid_head_runtime.rs`).
fn apply_hybrid_head_runtime(h: &[f32], meta: &HybridHeadMeta) -> f64 {
    debug_assert_eq!(meta.rank_w.len(), h.len());
    debug_assert!(!h.is_empty());
    // ONE owner — see `apply_per_sample_alpha_runtime` above.
    crate::score_math::hybrid_head(h, &meta.params(), active_pow_form())
}

/// Forward one bake over `features` and return the raw output scalar.
/// Handles the three accepted input-width shapes (n_inputs == features,
/// n_inputs == features+4 with size-axes, n_inputs < features prefix)
/// and dispatches to `predict_transformed` when the bake declares
/// non-trivial `feature_transforms` metadata (V_20+ input-shaping
/// bakes); otherwise uses the plain `predict` path with zero overhead.
///
/// Bakes carrying `zentrain.per_sample_alpha_head` metadata
/// (V_24-per-sample-α architecture) take a separate dispatch path:
/// the forward returns `h` (the post-LeakyReLU hidden vector, since
/// the bake's layer 2 is an identity passthrough) and the runtime
/// mixes a rank-head + pool-head pair via a per-sample sigmoid gate.
/// See `apply_per_sample_alpha_runtime` above for the formula and
/// payload layout.
///
/// Bakes carrying `zentrain.hybrid_head` metadata (V_24-hybrid
/// architecture) take a similar dispatch path, but the α gate is a
/// single learned SCALAR (`α = σ(α_logit)`) shared across all
/// samples rather than computed per-sample. See
/// `apply_hybrid_head_runtime` above for the formula and payload
/// layout.
/// True iff `bytes` parses as a bake carrying a
/// `zentrain.output_calibration_spline` metadata entry — i.e. its raw
/// output is already a calibrated 0-100 score rather than a distance.
///
/// A parse failure returns `false`: the forward pass that follows will
/// surface the real parse error with a better diagnostic than this guard
/// could, and this helper must never turn a load problem into a
/// configuration complaint.
fn bake_carries_output_spline(bytes: &'static [u8]) -> bool {
    cached_profile_model(bytes)
        .map(|model| {
            model
                .metadata()
                .get(OUTPUT_CALIBRATION_SPLINE_KEY)
                .is_some()
        })
        .unwrap_or(false)
}

/// Refuse a profile configuration that would silently produce garbage
/// scores (ADD156 ship audit 2026-08-31, defect **D9**).
///
/// Every bake whose raw output feeds [`dispose_mlp_raw`] is checked: if it
/// carries an output-calibration spline while the profile asks for the
/// legacy distance mapping, the two calibrations compose into nonsense and
/// the hard clamp hides it as an exact `0.0`. Fail loud, naming the
/// setting that is missing.
///
/// The ensemble CLASSIFIER bake is deliberately not checked — its output is
/// a routing logit that never reaches the score disposition.
fn check_score_disposition(
    params: &crate::profile::ProfileParams,
    primary: fn() -> &'static [u8],
) -> Result<(), ZensimError> {
    const MISCONFIGURED: &str = "bake carries zentrain.output_calibration_spline \
         (its raw output is already a calibrated 0-100 score) but \
         ProfileParams::skip_score_mapping is false, so the legacy 100-A*d^B \
         distance mapping would be applied on top of the spline and clamp \
         virtually every score to exactly 0 — set .skip_score_mapping(true), \
         and .extrapolate_score(true) to keep the negative tail";

    for loader in [
        Some(primary),
        params.mlp_bytes_b3,
        params.mlp_bytes_compression,
    ]
    .into_iter()
    .flatten()
    {
        if bake_carries_output_spline(loader()) {
            return Err(ZensimError::ModelForwardFailed {
                reason: MISCONFIGURED,
            });
        }
    }
    Ok(())
}

fn forward_one_bake(
    bytes: &'static [u8],
    features: &[f64],
    width: u32,
    height: u32,
) -> Result<f64, ZensimError> {
    forward_one_bake_with_codec(bytes, features, width, height, None)
}

/// The PU-linear route [`Zensim::pu_fold_route`] resolved for one call.
pub(crate) struct PuFoldRoute {
    plan: Option<std::sync::Arc<crate::feature_plan::Plan>>,
    /// Below Rev5 a wide-plan pair of identical pixels returns 100 with zeros
    /// at the plan width, without walking, as SDR `compute` does. At Rev5
    /// identity features come from the canonical walk.
    short_circuit_identity: bool,
}

/// [`crate::fold_engine::score_plan`] for a `&'static` profile, derived once
/// per process.
///
/// The plan is a function of the profile's bakes, the v1 width of the config
/// (extended/IW flags and scale count), the pool-skipping flag and the
/// process formula revision, which is fixed for the process. Keying on the
/// params address, the skip flag and the v1 width covers every input, so the
/// PU entries no longer derive a plan (`Plan::for_bake`) on every call. The
/// cache holds at most a few entries per shipped profile.
pub(crate) fn cached_pu_plan(
    params: &'static ProfileParams,
    config: &ZensimConfig,
    skip_unread: bool,
) -> Option<std::sync::Arc<crate::feature_plan::Plan>> {
    use std::collections::HashMap;
    use std::sync::{Arc, OnceLock, RwLock};
    type Key = (usize, bool, usize);
    type Cache = RwLock<HashMap<Key, Option<Arc<crate::feature_plan::Plan>>>>;
    static CACHE: OnceLock<Cache> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    let key = (
        params as *const ProfileParams as usize,
        skip_unread,
        crate::fold_engine::v1_feature_width(config),
    );
    if let Ok(read) = cache.read()
        && let Some(plan) = read.get(&key)
    {
        return plan.clone();
    }
    let plan = crate::fold_engine::score_plan(params, config, skip_unread).map(Arc::new);
    if let Ok(mut write) = cache.write() {
        return write.entry(key).or_insert(plan).clone();
    }
    plan
}

/// The parsed [`crate::mlp::Model`] of a `&'static` profile bake, interned by
/// the bytes' data pointer and length — the same cache shape (and stability
/// argument) as [`cached_bake_metadata`] and `fold_engine::cached_bake_pool_need`:
/// profile bake slices come from `&'static` slots via `ProfileParams`, so the
/// key is stable and unique per slot. The cache holds one entry per distinct
/// `&'static` bake slice: the built-in profiles, plus any loaders a caller
/// registers through `custom-profiles`. A loader that leaks a fresh slice on
/// every call (already a leak) adds one parsed, decoded model per call.
/// Parsing is deterministic, so a cached model is
/// the model a fresh parse would return; the scoring arithmetic is unchanged.
///
/// Without it every `Zensim::compute` of an MLP profile parsed its bake up to
/// three times (serving plan, disposition check, forward), which became the
/// dominant small-image cost once a parse decodes f16 layers.
///
/// A parse failure is returned, never cached. Dynamically loaded bakes do not
/// come through here; they own their `Model` through `BakeScorer`.
pub(crate) fn cached_profile_model(
    bytes: &'static [u8],
) -> Result<std::sync::Arc<crate::mlp::Model>, ZensimError> {
    use std::collections::HashMap;
    use std::sync::{Arc, OnceLock, RwLock};

    type Cache = RwLock<HashMap<(usize, usize), Arc<crate::mlp::Model>>>;
    static CACHE: OnceLock<Cache> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    let key = (bytes.as_ptr() as usize, bytes.len());
    if let Ok(read) = cache.read()
        && let Some(model) = read.get(&key)
    {
        return Ok(Arc::clone(model));
    }
    let parsed = Arc::new(crate::mlp::Model::from_bytes(bytes).map_err(|_| {
        ZensimError::ModelLoadFailed {
            reason: "Model::from_bytes failed to parse the bake header or layer table",
        }
    })?);
    if let Ok(mut write) = cache.write() {
        // Another thread's parse of the same slot is equally valid; keep the first.
        return Ok(Arc::clone(write.entry(key).or_insert(parsed)));
    }
    // Poisoned lock: serve the fresh parse uncached.
    Ok(parsed)
}

/// Lookup the parsed metadata for `bytes` from the static interner,
/// or parse it on first sight and cache. Keyed by the bytes' data
/// pointer (bake byte slices always come from `&'static` slots via
/// `ProfileParams::mlp_bytes`, so the pointer is stable and unique
/// per slot).
///
/// Returns an `Arc<ScoreMetadata>` — the caller drops the cache
/// lock before doing the forward pass.
fn cached_bake_metadata(
    bytes: &[u8],
    model: &crate::mlp::Model,
) -> Result<std::sync::Arc<ScoreMetadata>, ZensimError> {
    use std::collections::HashMap;
    use std::sync::{Arc, OnceLock, RwLock};

    static CACHE: OnceLock<RwLock<HashMap<usize, Arc<ScoreMetadata>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    let key = bytes.as_ptr() as usize;

    // Fast path: read-only lookup. Multiple readers can pass through
    // simultaneously — only the first-sight parse holds the write lock.
    if let Ok(read) = cache.read()
        && let Some(entry) = read.get(&key)
    {
        return Ok(Arc::clone(entry));
    }

    // Slow path: parse + insert. Re-check after acquiring the write
    // lock — another thread may have raced ahead of us.
    let parsed = std::sync::Arc::new(parse_bake_metadata(model)?);

    if let Ok(mut write) = cache.write() {
        // Idempotent: another thread's parse for the same key is also
        // valid; we just keep whichever Arc landed first.
        let entry = write.entry(key).or_insert_with(|| Arc::clone(&parsed));
        return Ok(Arc::clone(entry));
    }
    // Poisoned lock — return the freshly parsed bundle without
    // caching. Forward correctness is preserved at the cost of one
    // re-parse next call.
    Ok(parsed)
}

/// Same as [`forward_one_bake`] but accepts an optional codec hint
/// that drives the per-codec post-spline affine calibration
/// (EXP-CROSS-CODEC-V11-E). When `codec_hint` is `Some` and the
/// bake carries `zentrain.per_codec_calibration` metadata, the
/// matched codec's `(alpha, beta)` affine is applied AFTER the
/// PCHIP spline; otherwise the score is returned identical to the
/// hint-less path.
fn forward_one_bake_with_codec(
    bytes: &'static [u8],
    features: &[f64],
    width: u32,
    height: u32,
    codec_hint: Option<&str>,
) -> Result<f64, ZensimError> {
    let model = cached_profile_model(bytes)?;
    // Legacy profiles do not bind their front end/reference cache to this
    // contract. Complete sampling candidates must use BakeScorer directly.
    if model.metadata().get(crate::sampling::KEY).is_some() {
        return Err(ZensimError::ModelLoadFailed {
            reason: "sampling-bound bake requires the BakeScorer pixel surface",
        });
    }
    let bundle = cached_bake_metadata(bytes, &model)?;
    let mut scorer = BakeScorer::with_metadata(&model, bundle)?;
    scorer.score_features(features, width, height, codec_hint)
}

// One inference dispatch for named profiles, dynamically loaded bakes and eval.
// The reusable surface owns these buffers; the named-profile adapter borrows them.
#[allow(clippy::too_many_arguments)]
fn forward_model_with_codec(
    model: &crate::mlp::Model,
    predictor: &mut crate::mlp::Predictor<'_>,
    bundle: &ScoreMetadata,
    layout: &crate::feature_layout::Layout,
    gathered: &mut Vec<f64>,
    f32_features: &mut Vec<f32>,
    x_std: &mut Vec<f64>,
    features: &[f64],
    width: u32,
    height: u32,
    codec_hint: Option<&str>,
) -> Result<f64, ZensimError> {
    let n_inputs = model.caller_input_width();
    let needs_transforms = model.has_nontrivial_feature_transforms();
    let per_sample_alpha = bundle.per_sample_alpha.as_deref();
    let hybrid_head = bundle.hybrid_head.as_deref();
    let tanh_pin_scale = bundle.tanh_pin_scale;
    let output_spline = bundle.output_spline.as_deref();
    // EXP-CROSS-CODEC-V11-E (2026-05-20): per-codec post-spline affine.
    // Applies AFTER the PCHIP spline. Only active when the bake carries
    // `zentrain.per_codec_calibration` metadata AND the caller supplied
    // a codec hint that maps to a registered codec entry. The
    // calibration table itself is cached; only the codec-hint lookup
    // runs per call.
    let per_codec_affine: Option<(f32, f32)> = match (&bundle.per_codec_calibration, codec_hint) {
        (Some(cal), Some(hint)) => lookup_per_codec_affine(cal, hint),
        _ => None,
    };

    // **LAYOUT — resolved ONCE, before any head branches.** A bake declaring a
    // DENSE feature set reads its ids at packed positions, so the caller's
    // identity-laid-out vector must be GATHERED rather than sliced.
    // `declared_layout` returns the identity layout for a bake with no id, an
    // unreproducible id, or a `zentrain.feature_set_id` whose slot count
    // differs from the declared width — for those, `feats` IS `features` and
    // nothing below changes by a bit.
    //
    // **Not `#[cfg]`-conditional, and must never become so again.** The
    // shipped `A`, `B`, `BHdr` and `D` bakes have been dense since `cb2f412d`;
    // while this was gated on `feature-regime-v2`, a build without that
    // feature fell through to `prep_bake_input_f32`'s PREFIX branch and served
    // every one of them the wrong features with no error — MEASURED at up to
    // 261 zensim points and a sign flip on `D`
    // (`benchmarks/dense_serving_ungate_2026-09-06.md`).
    //
    // **Hoisted above the min-max branch on purpose.** It used to sit below,
    // so a dense bake carrying a min-max head hit that branch's
    // `features.len() != n` guard and was REFUSED — loud, so never a wrong
    // number, but also not servable, and a second head added below the gather
    // would have inherited the same hazard silently. No shipped profile has a
    // min-max head, so this moves no shipped byte — which is also why the
    // dense × min-max combination is UNTESTED: it has no artifact to test
    // against. What is proven is that the identity path is unchanged (every
    // gate below, plus `serving`'s pinned scores). Gated by
    // `feature_layout::tests::every_shipped_bake_resolves_to_its_own_declared_width`,
    // by `serving::tests::every_shipped_profile_scores_its_pinned_value` (which
    // runs under EVERY feature permutation, which is the point), and end to end
    // by `dense_layout_round_trip` + `scripts/serving_matrix.sh`.
    let feats: &[f64] = if layout.is_identity() {
        features
    } else {
        // **The row must REACH every declared id, or this is a refusal.**
        // `Layout::gather` writes the structural `0.0` for an id the walk did
        // not reach, which is correct for a DECLARED GAP and wrong for a
        // caller who simply handed over a short vector: every id a dense bake
        // declares is an id it reads, so filling one is exactly the "a
        // consumer cannot tell this zero from a measured zero" failure the
        // dense contract exists to end. Loud, named, and never a silent prefix.
        if features.len() < layout.walk_width() {
            return Err(ZensimError::ModelForwardFailed {
                reason: "the bake declares feature ids this feature vector does not reach",
            });
        }
        layout.gather(features, gathered);
        gathered.as_slice()
    };

    // E33 `fx1`: a derived-input bake's model row is built from its gathered
    // declared layout (direct ids and same-cell f32 products). Load refused
    // every head, transform and malformed declaration this cannot serve.
    if let Some(derived) = bundle.derived_inputs.as_deref() {
        if features.len() < layout.walk_width() {
            return Err(ZensimError::ModelForwardFailed {
                reason: "the bake declares feature ids this feature vector does not reach",
            });
        }
        if layout.is_identity() {
            layout.gather(features, gathered);
        }
        derived.fill_f32(gathered, f32_features);
        return bake_dispatch_one(
            predictor,
            f32_features,
            needs_transforms,
            per_sample_alpha,
            hybrid_head,
            tanh_pin_scale,
            output_spline,
            per_codec_affine,
        );
    }

    // Min-max monotone head (Sill 1998): the score REPLACES the layer forward,
    // so bypass the Predictor entirely — read the bake's scaler + feature
    // transforms directly, apply transform→scale→clamp±8→min-max, then the
    // shared PCHIP spline + per-codec affine. The clamp mirrors the trainer's
    // standardize clamp so runtime scoring is bit-exact with the qualifying eval.
    if let Some(mm) = bundle.minmax_head.as_deref() {
        // INTERNAL layer-0 width on purpose (NOT `caller_input_width()`):
        // this path indexes the transform array by layer-0 position, so it
        // needs the two to be equal and refuses the bake below when they are
        // not. See campaign appendix E.9 for the distinction.
        let n = model.n_inputs();
        // The loop below indexes `transforms[i]` at layer-0 index `i`, which
        // only aligns for a 1:1 transform pipeline. A variable-arity bake
        // (expander, or pruned with `drop`) breaks the alignment AND panics
        // in the scalar `apply_with_params`; refuse instead.
        if mm.n != n || feats.len() < n || model.caller_input_width() != n {
            return Err(ZensimError::ModelForwardFailed {
                reason: "min-max head n does not match bake n_inputs / feature length",
            });
        }
        let transforms = model.feature_transforms();
        let params = model.feature_transform_params();
        let mean = model.scaler_mean();
        let scale = model.scaler_scale();
        x_std.resize(n, 0.0);
        for (i, slot) in x_std.iter_mut().enumerate() {
            // transform in f32 (matches the trainer's `apply_with_params(x as f32)`)
            let raw = feats[i] as f32;
            let t = match (transforms, params) {
                (Some(tf), Some(p)) => tf[i].apply_with_params(raw, &p[i]),
                (Some(tf), None) => tf[i].apply_with_params(raw, &[]),
                (None, _) => raw,
            } as f64;
            // standardize (sklearn zero-variance guard) + clamp, matching train_minmax
            let s = scale[i] as f64;
            let safe = if s == 0.0 { 1.0 } else { s };
            *slot = ((t - mean[i] as f64) / safe).clamp(-MINMAX_STD_CLAMP, MINMAX_STD_CLAMP);
        }
        let raw = apply_minmax_runtime(x_std, mm);
        let y_pre = tanh_pin_scale.map_or(raw, |scale| apply_tanh_output_pin(raw, scale));
        let y_after_spline = match output_spline {
            Some(spline) => apply_output_calibration_spline(y_pre, spline),
            None => y_pre,
        };
        return Ok(match per_codec_affine {
            Some((alpha, beta)) => (alpha as f64) + (beta as f64) * y_after_spline,
            None => y_after_spline,
        });
    }

    let dispatch = |p: &mut crate::mlp::Predictor<'_>, x: &[f32]| -> Result<f64, ZensimError> {
        bake_dispatch_one(
            p,
            x,
            needs_transforms,
            per_sample_alpha,
            hybrid_head,
            tanh_pin_scale,
            output_spline,
            per_codec_affine,
        )
    };
    prep_bake_input_f32(feats, n_inputs, width, height, f32_features)?;
    dispatch(predictor, f32_features)
}

/// Size a caller feature vector to a bake's declared input width and
/// convert to f32 into `out` (cleared first). Extracted from
/// [`forward_one_bake_with_codec`] so the batched FD-gradient entry
/// reuses the identical sizing semantics without a per-forward
/// allocation (campaign appendix Y, L-Y1). Branches:
/// exact width / +4 size-axes / prefix / loud mismatch.
fn prep_bake_input_f32(
    features: &[f64],
    n_inputs: usize,
    width: u32,
    height: u32,
    out: &mut Vec<f32>,
) -> Result<(), ZensimError> {
    out.clear();
    if n_inputs == features.len() {
        out.extend(features.iter().map(|&v| v as f32));
        Ok(())
    } else if n_inputs == features.len() + 4 {
        let mut augmented = features.to_vec();
        append_mlp_size_axes(&mut augmented, width, height);
        out.extend(augmented.iter().map(|&v| v as f32));
        Ok(())
    } else if n_inputs < features.len() {
        out.extend(features[..n_inputs].iter().map(|&v| v as f32));
        Ok(())
    } else {
        // n_inputs > features.len() + 4 — bake expects more features than
        // the caller supplied, including the optional 4 size axes. This
        // is a feature-shape mismatch on a successfully-loaded bake.
        Err(ZensimError::ModelForwardFailed {
            reason: "bake declares more input features than the caller supplied",
        })
    }
}

/// The per-forward post-network dispatch shared by
/// [`forward_one_bake_with_codec`] and the batched FD-gradient entry
/// ([`score_features_fd_gradient_with_profile`]) — extracted verbatim
/// from the former's closure so the two paths cannot drift (campaign
/// appendix Y, L-Y1). `x` must already be sized to the bake's input
/// width (see [`prep_bake_input_f32`]).
#[allow(clippy::too_many_arguments)]
fn bake_dispatch_one(
    p: &mut crate::mlp::Predictor<'_>,
    x: &[f32],
    needs_transforms: bool,
    per_sample_alpha: Option<&PerSampleAlphaMeta>,
    hybrid_head: Option<&HybridHeadMeta>,
    tanh_pin_scale: Option<f64>,
    output_spline: Option<&OutputCalibrationSpline>,
    per_codec_affine: Option<(f32, f32)>,
) -> Result<f64, ZensimError> {
    let out = if needs_transforms {
        p.predict_transformed(x)
            .map_err(|_| ZensimError::ModelForwardFailed {
                reason: "Predictor::predict_transformed failed",
            })?
    } else {
        p.predict(x).map_err(|_| ZensimError::ModelForwardFailed {
            reason: "Predictor::predict failed",
        })?
    };
    finish_network_output(
        out,
        per_sample_alpha,
        hybrid_head,
        tanh_pin_scale,
        output_spline,
        per_codec_affine,
    )
}

fn finish_network_output(
    out: &[f32],
    per_sample_alpha: Option<&PerSampleAlphaMeta>,
    hybrid_head: Option<&HybridHeadMeta>,
    tanh_pin_scale: Option<f64>,
    output_spline: Option<&OutputCalibrationSpline>,
    per_codec_affine: Option<(f32, f32)>,
) -> Result<f64, ZensimError> {
    let y_pre = if let Some(meta) = per_sample_alpha {
        // `out` is the hidden vector h (n_hidden floats). Apply
        // the per-sample-α runtime formula.
        if out.len() != meta.rank_w.len() {
            // Shape mismatch between metadata's declared n_hidden
            // and bake's n_outputs — bake is malformed.
            return Err(ZensimError::ModelForwardFailed {
                reason: "per-sample-α metadata n_hidden does not match bake output length",
            });
        }
        apply_per_sample_alpha_runtime(out, meta)
    } else if let Some(meta) = hybrid_head {
        // `out` is the hidden vector h. Apply the scalar-α
        // hybrid runtime formula.
        if out.len() != meta.rank_w.len() {
            return Err(ZensimError::ModelForwardFailed {
                reason: "hybrid-head metadata n_hidden does not match bake output length",
            });
        }
        apply_hybrid_head_runtime(out, meta)
    } else {
        out.first()
            .copied()
            .ok_or(ZensimError::ModelForwardFailed {
                reason: "model has no score output",
            })? as f64
    };
    let y_after_pin = if let Some(scale) = tanh_pin_scale {
        apply_tanh_output_pin(y_pre, scale)
    } else {
        y_pre
    };
    let y_after_spline = if let Some(spline) = output_spline {
        apply_output_calibration_spline(y_after_pin, spline)
    } else {
        y_after_pin
    };
    // EXP-CROSS-CODEC-V11-E: per-codec post-spline affine. When
    // the metadata is present AND the caller supplied a known
    // codec hint, pull the per-codec output toward consensus.
    // Otherwise pass through unchanged. The affine is monotone
    // (beta > 0 by parse) so within-codec rank ordering is
    // preserved bit-exact.
    Ok(if let Some((alpha, beta)) = per_codec_affine {
        (alpha as f64) + (beta as f64) * y_after_spline
    } else {
        y_after_spline
    })
}

/// Append `(log2(pixels), log2(min_dim), log2(max_dim), signed
/// log2(max/min))` to a feature vector. Mirrors the trainer-side
/// `append_size_axes` in `zensim-validate/src/main.rs` so a model
/// trained with `--mlp-size-axes` produces the same input layout
/// at runtime.
fn append_mlp_size_axes(features: &mut Vec<f64>, width: u32, height: u32) {
    if width == 0 || height == 0 {
        features.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
        return;
    }
    let w = width as f64;
    let h = height as f64;
    let pixels = w * h;
    let min_dim = w.min(h);
    let max_dim = w.max(h);
    // F19: these four are MLP INPUTS, so they reach a shipped score through
    // the forward pass exactly as the mapping exponents do.
    let form = active_pow_form();
    let log2_pixels = pixels.det_log2(form);
    let log2_min = min_dim.det_log2(form);
    let log2_max = max_dim.det_log2(form);
    let log_aspect_signed = (max_dim / min_dim).det_log2(form) * if w >= h { 1.0 } else { -1.0 };
    features.extend_from_slice(&[log2_pixels, log2_min, log2_max, log_aspect_signed]);
}

/// Features per channel per scale: 19 features always emitted.
///
/// ```text
///  Index  Name             Pooling  Source
///  ─────  ───────────────  ───────  ──────────────────
///   0     ssim_mean        mean     SSIM error map
///   1     ssim_4th         L4       SSIM error map
///   2     ssim_2nd         L2       SSIM error map
///   3     art_mean         mean     edge artifact (ringing)
///   4     art_4th          L4       edge artifact
///   5     art_2nd          L2       edge artifact
///   6     det_mean         mean     edge detail lost (blur)
///   7     det_4th          L4       edge detail lost
///   8     det_2nd          L2       edge detail lost
///   9     mse              mean     (src - dst)²
///  10     hf_energy_loss   ratio    1 - Σ(dst-mu)²/Σ(src-mu)²
///  11     hf_mag_loss      ratio    1 - Σ|dst-mu|/Σ|src-mu|
///  12     hf_energy_gain   ratio    Σ(dst-mu)²/Σ(src-mu)² - 1
///  13     ssim_max         max      per-pixel SSIM error
///  14     art_max          max      per-pixel edge artifact
///  15     det_max          max      per-pixel edge detail_lost
///  16     ssim_l8          L8       (Σd⁸/N)^(1/8) SSIM error
///  17     art_l8           L8       (Σd⁸/N)^(1/8) edge artifact
///  18     det_l8           L8       (Σd⁸/N)^(1/8) edge detail_lost
/// ```
///
/// Total features = `num_scales × 3 channels × 13` = 156 at 4 scales.
///
/// Note: 6 additional "peak" features (max/l8) are always computed
/// but only included when `compute_all_features` is true. This keeps
/// the default feature vector compatible with existing profiles.
pub const FEATURES_PER_CHANNEL_BASIC: usize = 13;

/// Features per channel when `compute_all_features` is true: 19 features
/// (13 basic + 6 peak/l8). Peak features are always computed (near-zero cost)
/// but excluded from the default feature vector for profile compatibility.
pub const FEATURES_PER_CHANNEL_WITH_PEAKS: usize = 19;

/// Extended features per channel per scale: 25 features (19 with peaks + 6 masked).
///
/// ```text
///  Index  Name               Pooling  Source
///  ─────  ─────────────────  ───────  ──────────────────
///  0–12   (same as basic 13)
///  13–18  (same as peak features: max/l8)
///  19     masked_ssim_mean   mean     SSIM × flatness mask
///  20     masked_ssim_4th    L4       SSIM × flatness mask
///  21     masked_ssim_2nd    L2       SSIM × flatness mask
///  22     masked_art_4th     L4       edge artifact × flatness mask
///  23     masked_det_4th     L4       edge detail_lost × flatness mask
///  24     masked_mse         mean     (src-dst)² × flatness mask
/// ```
///
/// Total features = `num_scales × 3 channels × 25` = 300 at 4 scales.
pub const FEATURES_PER_CHANNEL_EXTENDED: usize = 25;

/// Information-content-weighted (IW) features per channel per scale.
/// Wang & Li 2011 — IW-SSIM. Same 6 feature slots as the `masked_*`
/// block but with the weight polarity flipped: texture-rich regions
/// get MORE weight.
///
/// ```text
///  Index  Name               Pooling  Source
///  ─────  ─────────────────  ───────  ──────────────────
///  0      iw_ssim_mean       mean     SSIM × iw_weight
///  1      iw_ssim_4th        L4       SSIM × iw_weight
///  2      iw_ssim_2nd        L2       SSIM × iw_weight
///  3      iw_art_4th         L4       edge artifact × iw_weight
///  4      iw_det_4th         L4       edge detail_lost × iw_weight
///  5      iw_mse             mean     (src-dst)² × iw_weight
/// ```
///
/// Enabled via `ZensimConfig::compute_iw_features`. At `num_scales = 4`
/// and 3 channels, adds 72 features → total profile becomes 300
/// (basic + peaks + IW).
pub const FEATURES_PER_CHANNEL_IW: usize = 6;

/// Named view over a flat feature vector.
///
/// Provides ergonomic access to features by name, scale, and channel
/// without changing the underlying storage format.
///
/// ```ignore
/// let result = z.compute_all_features(&src, &dst)?;
/// let view = FeatureView::new(result.features(), 4)?;
/// let ssim_mean_s0_y = view.ssim_mean(0, 1);
/// let ssim_max_s2_x = view.ssim_max(0, 2).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct FeatureView<'a> {
    features: &'a [f64],
    n_scales: usize,
    /// Number of features in the scored block
    scored_total: usize,
    /// Number of features in the peaks block (0 if not present)
    peaks_total: usize,
}

/// XYB channel index: X (red-green chrominance).
#[cfg(feature = "training")]
pub const CH_X: usize = 0;
/// XYB channel index: Y (luminance).
#[cfg(feature = "training")]
pub const CH_Y: usize = 1;
/// XYB channel index: B (blue-yellow chrominance).
#[cfg(feature = "training")]
pub const CH_B: usize = 2;

impl<'a> FeatureView<'a> {
    /// Create a view over a feature vector.
    ///
    /// Automatically detects the tier (peaks/extended) from length.
    /// Returns `None` if the length doesn't match any valid layout.
    /// Peaks are always present (basic-only 156-element vectors are no longer generated).
    pub fn new(features: &'a [f64], n_scales: usize) -> Option<Self> {
        let basic_total = n_scales * 3 * FEATURES_PER_CHANNEL_BASIC;
        let peaks_total = n_scales * 3 * 6;
        let masked_total = n_scales * 3 * 6;

        let (scored_total, peaks_total) = if features.len() == basic_total {
            // Legacy basic-only layout (backward compat)
            (basic_total, 0)
        } else if features.len() == basic_total + peaks_total
            || features.len() == basic_total + peaks_total + masked_total
        {
            (basic_total, peaks_total)
        } else {
            return None;
        };

        Some(Self {
            features,
            n_scales,
            scored_total,
            peaks_total,
        })
    }

    /// Number of scales in this feature vector.
    pub fn n_scales(&self) -> usize {
        self.n_scales
    }

    /// Whether peak features (max/L8) are present.
    pub fn has_peaks(&self) -> bool {
        self.peaks_total > 0
    }

    /// Whether masked features are present.
    pub fn has_masked(&self) -> bool {
        self.features.len() > self.scored_total + self.peaks_total
    }

    // --- Scored features (always present) ---

    fn scored_idx(&self, scale: usize, ch: usize, offset: usize) -> usize {
        scale * 3 * FEATURES_PER_CHANNEL_BASIC + ch * FEATURES_PER_CHANNEL_BASIC + offset
    }

    /// SSIM error, mean pooling.
    pub fn ssim_mean(&self, scale: usize, ch: usize) -> f64 {
        self.features[self.scored_idx(scale, ch, 0)]
    }
    /// SSIM error, L4 norm.
    pub fn ssim_4th(&self, scale: usize, ch: usize) -> f64 {
        self.features[self.scored_idx(scale, ch, 1)]
    }
    /// SSIM error, L2 norm.
    pub fn ssim_2nd(&self, scale: usize, ch: usize) -> f64 {
        self.features[self.scored_idx(scale, ch, 2)]
    }
    /// Edge artifact (ringing), mean pooling.
    pub fn art_mean(&self, scale: usize, ch: usize) -> f64 {
        self.features[self.scored_idx(scale, ch, 3)]
    }
    /// Edge artifact, L4 norm.
    pub fn art_4th(&self, scale: usize, ch: usize) -> f64 {
        self.features[self.scored_idx(scale, ch, 4)]
    }
    /// Edge artifact, L2 norm.
    pub fn art_2nd(&self, scale: usize, ch: usize) -> f64 {
        self.features[self.scored_idx(scale, ch, 5)]
    }
    /// Edge detail lost (blur), mean pooling.
    pub fn det_mean(&self, scale: usize, ch: usize) -> f64 {
        self.features[self.scored_idx(scale, ch, 6)]
    }
    /// Edge detail lost, L4 norm.
    pub fn det_4th(&self, scale: usize, ch: usize) -> f64 {
        self.features[self.scored_idx(scale, ch, 7)]
    }
    /// Edge detail lost, L2 norm.
    pub fn det_2nd(&self, scale: usize, ch: usize) -> f64 {
        self.features[self.scored_idx(scale, ch, 8)]
    }
    /// Mean squared error.
    pub fn mse(&self, scale: usize, ch: usize) -> f64 {
        self.features[self.scored_idx(scale, ch, 9)]
    }
    /// High-frequency energy loss ratio.
    pub fn hf_energy_loss(&self, scale: usize, ch: usize) -> f64 {
        self.features[self.scored_idx(scale, ch, 10)]
    }
    /// High-frequency magnitude loss ratio.
    pub fn hf_mag_loss(&self, scale: usize, ch: usize) -> f64 {
        self.features[self.scored_idx(scale, ch, 11)]
    }
    /// High-frequency energy gain ratio.
    pub fn hf_energy_gain(&self, scale: usize, ch: usize) -> f64 {
        self.features[self.scored_idx(scale, ch, 12)]
    }

    // --- Peak features (always present) ---

    fn peak_idx(&self, scale: usize, ch: usize, offset: usize) -> Option<usize> {
        if self.peaks_total == 0 {
            return None;
        }
        Some(self.scored_total + scale * 3 * 6 + ch * 6 + offset)
    }

    /// SSIM error, pixel-wise max.
    pub fn ssim_max(&self, scale: usize, ch: usize) -> Option<f64> {
        self.peak_idx(scale, ch, 0).map(|i| self.features[i])
    }
    /// Edge artifact, pixel-wise max.
    pub fn art_max(&self, scale: usize, ch: usize) -> Option<f64> {
        self.peak_idx(scale, ch, 1).map(|i| self.features[i])
    }
    /// Edge detail lost, pixel-wise max.
    pub fn det_max(&self, scale: usize, ch: usize) -> Option<f64> {
        self.peak_idx(scale, ch, 2).map(|i| self.features[i])
    }
    /// SSIM error, L8 norm `(Σd⁸/N)^(1/8)`.
    pub fn ssim_l8(&self, scale: usize, ch: usize) -> Option<f64> {
        self.peak_idx(scale, ch, 3).map(|i| self.features[i])
    }
    /// Edge artifact, L8 norm.
    pub fn art_l8(&self, scale: usize, ch: usize) -> Option<f64> {
        self.peak_idx(scale, ch, 4).map(|i| self.features[i])
    }
    /// Edge detail lost, L8 norm.
    pub fn det_l8(&self, scale: usize, ch: usize) -> Option<f64> {
        self.peak_idx(scale, ch, 5).map(|i| self.features[i])
    }

    // --- Masked features (require extended_features) ---

    fn masked_idx(&self, scale: usize, ch: usize, offset: usize) -> Option<usize> {
        if !self.has_masked() {
            return None;
        }
        Some(self.scored_total + self.peaks_total + scale * 3 * 6 + ch * 6 + offset)
    }

    /// Masked SSIM error, mean pooling.
    pub fn masked_ssim_mean(&self, scale: usize, ch: usize) -> Option<f64> {
        self.masked_idx(scale, ch, 0).map(|i| self.features[i])
    }
    /// Masked SSIM error, L4 norm.
    pub fn masked_ssim_4th(&self, scale: usize, ch: usize) -> Option<f64> {
        self.masked_idx(scale, ch, 1).map(|i| self.features[i])
    }
    /// Masked SSIM error, L2 norm.
    pub fn masked_ssim_2nd(&self, scale: usize, ch: usize) -> Option<f64> {
        self.masked_idx(scale, ch, 2).map(|i| self.features[i])
    }
    /// Masked edge artifact, L4 norm.
    pub fn masked_art_4th(&self, scale: usize, ch: usize) -> Option<f64> {
        self.masked_idx(scale, ch, 3).map(|i| self.features[i])
    }
    /// Masked edge detail lost, L4 norm.
    pub fn masked_det_4th(&self, scale: usize, ch: usize) -> Option<f64> {
        self.masked_idx(scale, ch, 4).map(|i| self.features[i])
    }
    /// Masked MSE.
    pub fn masked_mse(&self, scale: usize, ch: usize) -> Option<f64> {
        self.masked_idx(scale, ch, 5).map(|i| self.features[i])
    }

    /// Get the scored features slice (first N features, WEIGHTS-compatible).
    pub fn scored_features(&self) -> &[f64] {
        &self.features[..self.scored_total]
    }

    /// Get the peak features slice, if present.
    pub fn peak_features(&self) -> Option<&[f64]> {
        if self.peaks_total == 0 {
            None
        } else {
            Some(&self.features[self.scored_total..self.scored_total + self.peaks_total])
        }
    }

    /// Get the masked features slice, if present.
    pub fn masked_features(&self) -> Option<&[f64]> {
        if !self.has_masked() {
            None
        } else {
            Some(&self.features[self.scored_total + self.peaks_total..])
        }
    }
}

/// Compute zensim with custom configuration (training API).
///
/// Uses the v0.2 weights (latest general-purpose profile).
///
/// **Research / feature-extraction API; not stable.** Used by the
/// zensim-validate feature-extraction binaries and inline tests; the
/// `ZensimConfig` knobs change metric internals whose outputs are
/// incomparable with the default trained weights.
#[doc(hidden)]
#[cfg(any(feature = "training", test))]
pub fn compute_zensim_with_config(
    source: &[[u8; 3]],
    distorted: &[[u8; 3]],
    width: usize,
    height: usize,
    config: ZensimConfig,
) -> Result<ZensimResult, ZensimError> {
    // Validation. The route refusal comes first: this entry takes a
    // caller-built config, so it never passes through `config_from_params`
    // and would otherwise carry a `blur_passes != 1` config into the strip
    // walk under an arithmetic revision that does not serve it.
    crate::ssim_form::check_route(&config)?;
    refuse_rev5_masked_config(&config)?;
    if width < 8 || height < 8 {
        return Err(ZensimError::ImageTooSmall);
    }
    let pixels = width
        .checked_mul(height)
        .ok_or(ZensimError::ImageTooLarge)?;
    // Also reject overflow on the padded plane size (pyramid_plane_stride(width) * height).
    check_within_max_pixels(width, height, None)?;
    if source.len() != pixels {
        return Err(ZensimError::InvalidDataLength);
    }
    if distorted.len() != pixels {
        return Err(ZensimError::InvalidDataLength);
    }
    if source.len() != distorted.len() {
        return Err(ZensimError::DimensionMismatch);
    }

    // Identical images must score exactly 100.0 — short-circuit before
    // floating-point arithmetic introduces sub-ULP noise in SSIM/edge features.
    if source == distorted {
        // Match the feature width to the enabled config flags. See the
        // sister short-circuit above `compute_zensim_streaming` for the
        // bug history (the 372-input compute_iw_features path was
        // discovered when a 372-input bake failed with InvalidDataLength
        // because the short-circuit only counted basic+extended = 300).
        let fpc = match (config.extended_features, config.compute_iw_features) {
            (true, true) => FEATURES_PER_CHANNEL_EXTENDED + FEATURES_PER_CHANNEL_IW,
            (true, false) => FEATURES_PER_CHANNEL_EXTENDED,
            (false, _) => FEATURES_PER_CHANNEL_WITH_PEAKS,
        };
        let num_features = config.num_scales * 3 * fpc;
        // This entry point scores via the plain WEIGHTS (V0_2 linear)
        // distance, never an MLP forward pass — tag with the honest
        // legacy-linear profile rather than `A` (the canonical MLP-scored
        // profile).
        return Ok(ZensimResult::new(
            100.0,
            0.0,
            vec![0.0; num_features],
            ZensimProfile::LegacyLinearV0_2,
            [0.0; 3],
        )
        .mark_identical());
    }

    let src_img = crate::source::RgbSlice::new(source, width, height);
    let dst_img = crate::source::RgbSlice::new(distorted, width, height);

    // Reflect-pad a sub-pyramid-minimum pair, exactly as
    // `compute_with_config_inner` (every `Zensim::compute*`) does. Without
    // this the scale walk stopped early and this entry returned a SHORT
    // feature vector — 93 / 186 / 279 wide instead of 372 — with no error.
    // Both v1-372 extractors call this, which is what made ~6.5 % of the
    // R1b eval-slice rows ragged (docs/DATASET_HISTORY.md §3.26).
    let result = if needs_pyramid_pad(width, height, config.num_scales) {
        let s = reflect_pad_for_scales(&src_img, config.num_scales);
        let d = reflect_pad_for_scales(&dst_img, config.num_scales);
        crate::streaming::compute_zensim_streaming(&s, &d, &config, WEIGHTS)
    } else {
        crate::streaming::compute_zensim_streaming(&src_img, &dst_img, &config, WEIGHTS)
    };
    // See the identical-shortcut branch above: this function scores via
    // plain linear WEIGHTS, not an MLP bake, so it does not inherit
    // `combine_scores`'s internal `codec_target()` (`B`) placeholder tag
    // (which every `Zensim::compute*` caller overrides but this standalone
    // function does not go through).
    Ok(result.with_profile(ZensimProfile::LegacyLinearV0_2))
}

/// Combine per-scale statistics into a final score.
///
/// Uses learned weights that balance:
/// - Per-channel sensitivity (Y > X > B, matching human vision)
/// - Per-scale importance (medium scales most important)
/// - SSIM vs edge features
/// - Mean vs 4th-power pooling
///
/// Weights are trained against synthetic quality scores (see `weights/` directory).
/// Features per scale for the default scoring profile (3 channels × 13 features = 39).
#[cfg_attr(not(feature = "training"), allow(dead_code))]
pub const FEATURES_PER_SCALE: usize = FEATURES_PER_CHANNEL_WITH_PEAKS * 3;

/// Default scoring weights — references the latest profile weights.
///
/// Layout: 4 scales × 3 channels (X,Y,B) × 13 basic features, then
///         4 scales × 3 channels × 6 peak features = 228 total.
#[cfg(any(feature = "training", test))]
pub const WEIGHTS: &[f64; 228] = &crate::profile::WEIGHTS_PREVIEW_V0_2;

/// **THE single owner of "turn a v1-layout feature vector into
/// `(score, raw_distance)`".** Extracted verbatim from [`combine_scores`]
/// (2026-08-30, the fold-engine lane) so the buffered walk and the
/// fold-backed engine share one implementation of the linear scoring tail
/// instead of each restating the dot product — per the repo's
/// NO-DUPLICATE-IMPLEMENTATIONS rule, where a second *Rust* site counts as a
/// duplicate exactly as much as a Python one.
///
/// Does three things, in this order and no other:
///
/// 1. **Sanitize non-finite values** (NaN/Inf → 0) in place, before scoring
///    *and* before the MLP forward reads the same slice. A single bad feature
///    would otherwise poison the MLP (NaN cascades through every weighted
///    sum) or the dot product (Inf swamps it). Replacing with 0 matches
///    `zensim_gpu`'s defensive behaviour on degenerate inputs (one corpus
///    pair, 2048×1365 JPEG q60, hit Inf at masked-ssim_mean B-channel).
/// 2. **Dot with the weights** over the scored prefix — basic + peak, i.e.
///    `n_scales · 3 · FEATURES_PER_CHANNEL_WITH_PEAKS` slots, truncated to
///    `weights.len()` — in **ascending index order**, then divide by
///    `max(n_scales, 1)`. The order is load-bearing: it is what makes the
///    two engines' `raw_distance` bit-identical rather than
///    equal-to-an-epsilon, so do not reassociate it.
/// 3. **Map distance to score**: the correct-by-construction bounded squash
///    (linear profiles that opt in) or the legacy unbounded `100 − a·d^b`.
///    The squash is a strictly-monotone transform of the same
///    `raw_distance`, so the *rank* is identical; only boundedness differs.
///    See [`bounded_score_squash`] and
///    `docs/METRIC_INVARIANTS_MECHANISM_AND_REDESIGN_2026-05-26.md`.
///
/// `features` may be any v1-layout width (228 / 300 / 372) — the scored
/// prefix is the same in all of them, and the tail is sanitized but not
/// weighted.
pub(crate) fn score_v1_layout_features(
    features: &mut [f64],
    weights: &[f64],
    config: &ZensimConfig,
    n_scales: usize,
) -> (f64, f64) {
    for f in features.iter_mut() {
        if !f.is_finite() {
            *f = 0.0;
        }
    }

    let scored_total = n_scales * 3 * FEATURES_PER_CHANNEL_WITH_PEAKS;
    let n_score = scored_total.min(weights.len()).min(features.len());
    let mut raw_distance = 0.0f64;
    for (i, &feat) in features[..n_score].iter().enumerate() {
        raw_distance += feat * weights[i];
    }
    raw_distance /= n_scales.max(1) as f64;

    let score = if config.bounded_squash {
        bounded_score_squash(raw_distance, config.score_mapping_a, config.score_mapping_b)
    } else {
        distance_to_score_mapped(raw_distance, config.score_mapping_a, config.score_mapping_b)
    };
    (score, raw_distance)
}

pub(crate) fn combine_scores(
    scale_stats: &[ScaleStats],
    weights: &[f64],
    config: &ZensimConfig,
    mean_offset: [f64; 3],
) -> ZensimResult {
    let extended = config.extended_features;
    let iw = config.compute_iw_features;

    // Feature vector layout (in order they appear in the Vec):
    //   [0..N_basic)        — 13/ch × 3ch × n_scales (basic features)
    //   [N_basic..N_peaks)  — 6/ch × 3ch × n_scales peak features (always included)
    //   [N_peaks..N_masked) — 6/ch × 3ch × n_scales masked features (if extended)
    //   [N_masked..N_iw)    — 6/ch × 3ch × n_scales IW features (if compute_iw_features)
    //
    // Both basic and peak features are scored: features[0..WEIGHTS.len()]
    // produces the dot product used for the final score. Masked + IW are
    // training-only feature additions consumed by MLP-scored profiles.
    let n_scales = scale_stats.len();
    let basic_per_ch = FEATURES_PER_CHANNEL_BASIC; // 13
    let basic_total = n_scales * basic_per_ch * 3;
    let peak_total = n_scales * 6 * 3;
    let masked_total = if extended { n_scales * 6 * 3 } else { 0 };
    let iw_total = if iw {
        n_scales * FEATURES_PER_CHANNEL_IW * 3
    } else {
        0
    };
    let total = basic_total + peak_total + masked_total + iw_total;

    let mut features = Vec::with_capacity(total);

    // Pass 1: scored features (13/ch, weight-compatible order)
    for ss in scale_stats.iter() {
        for c in 0..3 {
            features.push(ss.ssim[c * 2].abs());
            features.push(ss.ssim[c * 2 + 1].abs());
            features.push(ss.ssim_2nd[c].abs());
            features.push(ss.edge[c * 4].abs());
            features.push(ss.edge[c * 4 + 1].abs());
            features.push(ss.edge_2nd[c * 2].abs());
            features.push(ss.edge[c * 4 + 2].abs());
            features.push(ss.edge[c * 4 + 3].abs());
            features.push(ss.edge_2nd[c * 2 + 1].abs());
            features.push(ss.mse[c]);
            features.push(ss.hf_energy_loss[c]);
            features.push(ss.hf_mag_loss[c]);
            features.push(ss.hf_energy_gain[c]);
        }
    }

    // Pass 2: peak features (6/ch — max + L8, always computed at near-zero cost)
    for ss in scale_stats.iter() {
        for c in 0..3 {
            features.push(ss.ssim_max[c]);
            features.push(ss.art_max[c]);
            features.push(ss.det_max[c]);
            features.push(ss.ssim_p95[c]);
            features.push(ss.art_p95[c]);
            features.push(ss.det_p95[c]);
        }
    }

    // Pass 3: masked features (6/ch — expensive, training only)
    if extended {
        for ss in scale_stats.iter() {
            for c in 0..3 {
                features.push(ss.masked_ssim[c * 3].abs());
                features.push(ss.masked_ssim[c * 3 + 1].abs());
                features.push(ss.masked_ssim[c * 3 + 2].abs());
                features.push(ss.masked_art_4th[c].abs());
                features.push(ss.masked_det_4th[c].abs());
                features.push(ss.masked_mse[c]);
            }
        }
    }

    // Pass 4: IW (information-content-weighted) features (6/ch —
    // texture-emphasising counterpart to masked). Wang & Li 2011.
    if iw {
        for ss in scale_stats.iter() {
            for c in 0..3 {
                features.push(ss.iw_ssim[c * 3].abs());
                features.push(ss.iw_ssim[c * 3 + 1].abs());
                features.push(ss.iw_ssim[c * 3 + 2].abs());
                features.push(ss.iw_art_4th[c].abs());
                features.push(ss.iw_det_4th[c].abs());
                features.push(ss.iw_mse[c]);
            }
        }
    }

    let (score, raw_distance) =
        score_v1_layout_features(&mut features, weights, config, scale_stats.len());
    debug_assert_eq!(
        basic_total + peak_total,
        n_scales * 3 * FEATURES_PER_CHANNEL_WITH_PEAKS
    );

    // Placeholder profile tag — every `Zensim::compute*` caller overrides
    // it via `with_profile(self.profile)` before returning to the user.
    // Uses the canonical codec-target profile (`B`), never the deprecated `A`.
    ZensimResult::new(
        score,
        raw_distance,
        features,
        ZensimProfile::codec_target(),
        mean_offset,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One parse per `&'static` profile slot, and the cached model is the model a
    /// fresh parse returns: identical forward bits on a fixed row. (Scores stay
    /// pinned by `serving::tests::every_shipped_profile_scores_its_pinned_value`.)
    #[test]
    fn profile_model_cache_interns_one_parse_per_slot() {
        let loader = crate::profile::ZensimProfile::B
            .params()
            .mlp_bytes
            .expect("B scores through a bake");
        let first = cached_profile_model(loader()).unwrap();
        let second = cached_profile_model(loader()).unwrap();
        assert!(std::sync::Arc::ptr_eq(&first, &second));
        let fresh = crate::mlp::Model::from_bytes(loader()).unwrap();
        let row: Vec<f32> = (0..fresh.caller_input_width())
            .map(|i| (i as f32 * 0.37).sin())
            .collect();
        let cached_out = crate::mlp::Predictor::new(&first)
            .predict(&row)
            .unwrap()
            .to_vec();
        let fresh_out = crate::mlp::Predictor::new(&fresh)
            .predict(&row)
            .unwrap()
            .to_vec();
        assert_eq!(
            cached_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            fresh_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    /// **D9 (ADD156 ship audit, `benchmarks/add156_ship_audit_2026-08-31.md`).**
    /// A bake carrying `zentrain.output_calibration_spline` already emits a
    /// calibrated 0–100 score. Configuring it with
    /// `skip_score_mapping == false` re-applies the legacy `100 − A·d^B`
    /// *distance* mapping on top of the spline, which drives every realistic
    /// score deeply negative; the default hard clamp then pins it to
    /// **exactly 0.0**. The audit reproduced this first try on ADD156:
    /// identity 100.000000, q5 = 9.300537, and **q10 through q100 all
    /// exactly 0.000000**, with no error and no warning — a model that looks
    /// catastrophically broken when only its wiring is.
    ///
    /// Two assertions, and both must hold:
    ///  1. the pathology itself is PINNED — a real spline-bake output run
    ///     through the misconfigured disposition really is exactly `0.0`, so
    ///     nobody can "fix" this by quietly changing the mapping and leave
    ///     the silent-zero class alive;
    ///  2. the combination is now REFUSED with a diagnostic that names the
    ///     missing setting, instead of returning that zero.
    ///
    /// Assertion 2 is the one that fails before the fix.
    #[test]
    fn d9_spline_bake_without_skip_score_mapping_is_refused_not_silently_zeroed() {
        use crate::source::RgbSlice;

        // A deterministic non-degenerate pair. The distorted side is a
        // coarse quantization of the reference, which lands the score well
        // inside the dial's interior (nowhere near 0 or 100).
        const W: usize = 64;
        const H: usize = 64;
        let mut refpx = vec![[0u8; 3]; W * H];
        let mut dstpx = vec![[0u8; 3]; W * H];
        for y in 0..H {
            for x in 0..W {
                let i = y * W + x;
                let r = ((x * 3 + y) % 256) as u8;
                let g = ((x + y * 5) % 256) as u8;
                let b = ((x * 7 + y * 2) % 256) as u8;
                refpx[i] = [r, g, b];
                // 5-bit quantize + a structured ripple: a real, visible distortion.
                let q = |v: u8| (v & 0xF8).saturating_add(if (x + y) % 3 == 0 { 9 } else { 0 });
                dstpx[i] = [q(r), q(g), q(b)];
            }
        }
        let src = RgbSlice::new(&refpx, W, H);
        let dst = RgbSlice::new(&dstpx, W, H);

        // Control: the SHIPPED, correct configuration (PROFILE_B sets
        // `skip_score_mapping: true` + `extrapolate_score: true`).
        let historical =
            crate::mlp::Model::from_bytes(crate::profile::linear_bake_b_cid80()).unwrap();
        let control = BakeScorer::new(&historical)
            .unwrap()
            .with_parallel(false)
            .compute(&src, &dst, None)
            .expect("control compute with the shipped B profile");
        let good_score = control.score();
        assert!(
            good_score.is_finite() && good_score > 1.0 && good_score < 99.0,
            "control score {good_score} is not in the dial interior — the \
             fixture no longer exercises the path this test is about"
        );
        let feats = control.features().to_vec();
        assert_eq!(
            feats.len(),
            372,
            "B is a 372-input profile; got {} features",
            feats.len()
        );

        // The misconfiguration: identical to PROFILE_B except for the two
        // dispositions a caller using `ProfileParams::builder()` gets BY
        // DEFAULT (both `false`).
        let broken = crate::profile::ProfileParams {
            weights: &crate::profile::WEIGHTS_PREVIEW_V0_2,
            blur_radius: 5,
            blur_passes: 1,
            num_scales: 4,
            bounded_squash: false,
            score_mapping_a: 18.0,
            score_mapping_b: 0.7,
            skip_score_mapping: false, // <-- the defect
            mlp_bytes: Some(crate::profile::linear_bake_b_cid80),
            mlp_bytes_b3: None,
            mlp_primary_mix: 1.0,
            extended_features: true,
            compute_iw_features: true,
            soft_clamp_score: false,
            extrapolate_score: false, // <-- the defect's silencer
            ensemble_classifier_bytes: None,
            mlp_bytes_compression: None,
        };

        // (1) Pin the pathology. The dynamic control score is the bake's
        // post-spline raw output — already a 0–100 score. Running it through
        // the misconfigured disposition yields EXACTLY zero.
        let raw = good_score;
        assert!(
            raw > 7.5,
            "fixture raw output {raw} is too small to trigger the pathology"
        );
        let silent_garbage = dispose_mlp_raw(raw, &broken);
        assert_eq!(
            silent_garbage, 0.0,
            "the D9 pathology no longer reproduces: raw {raw} disposed to \
             {silent_garbage} instead of exactly 0.0"
        );

        // (2) The guard: scoring under that configuration must FAIL LOUD.
        let mut result =
            ZensimResult::new(0.0, 0.0, feats, ZensimProfile::B, control.mean_offset());
        let outcome = apply_mlp_scoring(&mut result, &broken, W as u32, H as u32);
        let err = match outcome {
            Err(e) => e,
            Ok(()) => panic!(
                "D9 REGRESSION: a spline-carrying bake scored with \
                 skip_score_mapping=false returned Ok and score {} — silent \
                 garbage instead of a diagnostic",
                result.score()
            ),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("skip_score_mapping"),
            "the diagnostic must name the setting that was missing; got: {msg}"
        );
        assert!(
            msg.contains("output_calibration_spline"),
            "the diagnostic must name what the bake carries; got: {msg}"
        );
    }

    /// `soft_clamp_score` matches its documented boundary values and
    /// stays monotone strictly increasing — the rank-preservation
    /// property that the V_20+ multi-bake regime relies on.
    #[test]
    fn soft_clamp_score_boundary_values() {
        // Documented anchor values in the function docstring.
        assert!((soft_clamp_score(-1000.0) - 0.0).abs() < 1e-9);
        assert!((soft_clamp_score(0.0) - 7.585818002124355).abs() < 1e-9);
        assert!((soft_clamp_score(50.0) - 50.0).abs() < 1e-9);
        assert!((soft_clamp_score(100.0) - 92.41418199787566).abs() < 1e-9);
        assert!((soft_clamp_score(1000.0) - 100.0).abs() < 1e-9);
    }

    /// OOD safety net (2026-07-10): the output-calibration spline's lower
    /// extrapolation is floored one calibrated-dial-range below the bottom
    /// knot, so a pathological out-of-distribution raw that slips past the
    /// winsor guard produces a bounded score instead of a wild one (a raw of
    /// −8.63 previously → ~−1131). Legitimate / modest negatives (the neg-tail)
    /// are preserved; only the wild extreme is clamped. Mirrors
    /// `zensim_validate::output_calibration_spline`.
    #[test]
    fn output_spline_lower_extrapolation_floored_ood_safety_net() {
        // 2-knot identity-ish (0,0)-(100,100): floor = ys[0] - (ys[n-1]-ys[0])
        // = 0 - 100 = -100.
        let mut bytes = (2u32).to_le_bytes().to_vec();
        for (x, y) in [(0.0f32, 0.0f32), (100.0f32, 100.0f32)] {
            bytes.extend_from_slice(&x.to_le_bytes());
            bytes.extend_from_slice(&y.to_le_bytes());
        }
        let spline = parse_output_calibration_spline(&bytes).expect("parse");
        // Modest negatives extrapolate linearly (neg-tail intact).
        assert!((apply_output_calibration_spline(-10.0, &spline) - (-10.0)).abs() < 1e-6);
        assert!((apply_output_calibration_spline(-8.63, &spline) - (-8.63)).abs() < 1e-6);
        // Pathological far-negative raw floored at -100, not wild.
        assert!(
            (apply_output_calibration_spline(-1.0e6, &spline) - (-100.0)).abs() < 1e-6,
            "OOD pathology must floor at -100, not extrapolate to a wild score"
        );
        // Monotone across the floor transition.
        let mut prev = f64::NEG_INFINITY;
        for i in 0..=200 {
            let x = -200.0 + i as f64;
            let y = apply_output_calibration_spline(x, &spline);
            assert!(y >= prev - 1e-9, "non-monotone at x={x}");
            prev = y;
        }
    }

    /// Soft-clamp is strictly monotone — adjacent raw scores keep
    /// their order, so SROCC (rank-based) is preserved through the
    /// transform. The hard-clamp pathology (tie blocks at 0/100)
    /// disappears: out-of-range raw values stay distinct.
    #[test]
    fn soft_clamp_score_monotone_across_full_range() {
        let mut prev = f64::NEG_INFINITY;
        // Sweep a wide grid: deep below 0, through the saturation band,
        // and far above 100. 1e6 raw is plausible for accidentally
        // distance-shaped bakes wired in by mistake — must still rank.
        for &raw in &[
            -1e6, -1000.0, -100.0, -50.0, -10.0, 0.0, 1.0, 25.0, 49.999, 50.0, 50.001, 75.0, 90.0,
            99.999, 100.0, 100.001, 110.0, 200.0, 1000.0, 1e6,
        ] {
            let s = soft_clamp_score(raw);
            assert!(
                s > prev || (s.is_finite() && prev.is_finite() && (s - prev).abs() < 1e-12),
                "soft_clamp not monotone: f({raw}) = {s}, prev = {prev}"
            );
            assert!(
                (0.0..=100.0).contains(&s),
                "soft_clamp out of [0, 100]: f({raw}) = {s}"
            );
            prev = s;
        }
    }

    /// The interior shift is non-zero but bounded. Inside `[25, 75]`
    /// the soft-clamp shift stays under ±3.2 units; at the boundary
    /// it grows to ±7.6 (raw = 0 → 7.586, raw = 100 → 92.414).
    /// Profiles that enable `soft_clamp_score` accept this
    /// recalibration as the price of rank preservation at the tails.
    #[test]
    fn soft_clamp_score_interior_shift_bounded() {
        for tenths in 250..=750 {
            let raw = tenths as f64 / 10.0;
            let s = soft_clamp_score(raw);
            assert!(
                (s - raw).abs() <= 3.2,
                "soft_clamp shift >3.2 in [25,75]: f({raw}) = {s} (Δ={})",
                (s - raw).abs()
            );
        }
    }

    /// Verify compute_all_features produces same score as default (weight-skipped) path.
    /// This exercises the multi-SSIM channel code path where ssim_chs.len() > 1.
    #[test]
    fn compute_all_matches_default() {
        // Generate a simple test pattern: gradient source, slightly different distorted
        let w = 128;
        let h = 128;
        let n = w * h;
        let mut src = vec![[128u8, 128, 128]; n];
        let mut dst = vec![[128u8, 128, 128]; n];
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 255) / w) as u8;
                let g = ((y * 255) / h) as u8;
                let b = 128;
                src[y * w + x] = [r, g, b];
                // Slight distortion
                dst[y * w + x] = [r.saturating_add(5), g, b.saturating_sub(3)];
            }
        }

        let default_result =
            compute_zensim_with_config(&src, &dst, w, h, ZensimConfig::default()).unwrap();
        let all_result = compute_zensim_with_config(
            &src,
            &dst,
            w,
            h,
            ZensimConfig {
                compute_all_features: true,
                ..Default::default()
            },
        )
        .unwrap();

        // Same score (default weights skip zero-weight channels; compute_all computes them
        // but zero weights still produce same weighted distance)
        assert!(
            (default_result.score - all_result.score).abs() < 0.01,
            "default {} vs all_features {}",
            default_result.score,
            all_result.score,
        );

        // Both default and compute_all now include peak features (228)
        assert_eq!(all_result.features.len(), 228);
        assert_eq!(default_result.features.len(), 228);
        // With compute_all, previously-skipped channels should now have nonzero features
        let all_nonzero = all_result
            .features
            .iter()
            .filter(|f| f.abs() > 1e-12)
            .count();
        let default_nonzero = default_result
            .features
            .iter()
            .filter(|f| f.abs() > 1e-12)
            .count();
        assert!(
            all_nonzero >= default_nonzero,
            "compute_all should have >= features: {} vs {}",
            all_nonzero,
            default_nonzero,
        );
    }

    /// `compute_zensim_with_config` scores via plain linear `WEIGHTS`
    /// (V0_2), never an MLP forward pass — it must NOT claim the
    /// MLP-scored `ZensimProfile::A` tag (a prior bug hardcoded `A` here).
    /// Covers both the general path and the identical-images
    /// short-circuit, which had its own separate hardcoded `A`.
    #[test]
    fn compute_zensim_with_config_tags_legacy_linear_profile_not_a() {
        let w = 128;
        let h = 128;
        let n = w * h;
        let mut src = vec![[128u8, 128, 128]; n];
        let mut dst = vec![[128u8, 128, 128]; n];
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 255) / w) as u8;
                src[y * w + x] = [r, 128, 128];
                dst[y * w + x] = [r.saturating_add(5), 128, 128];
            }
        }

        // General (non-identical) path.
        let result = compute_zensim_with_config(&src, &dst, w, h, ZensimConfig::default()).unwrap();
        assert_eq!(
            result.profile(),
            crate::profile::ZensimProfile::LegacyLinearV0_2,
            "linear-scored result must not be tagged as MLP-scored profile A"
        );

        // Identical-images short-circuit path (its own hardcoded tag).
        let identical =
            compute_zensim_with_config(&src, &src, w, h, ZensimConfig::default()).unwrap();
        assert_eq!(
            identical.profile(),
            crate::profile::ZensimProfile::LegacyLinearV0_2,
            "identical-images short-circuit must not be tagged as MLP-scored profile A"
        );
        assert_eq!(identical.score(), 100.0);
    }

    /// Same honesty check for `compute_zensim_with_ref_and_config`.
    #[cfg(feature = "training")]
    #[test]
    fn compute_zensim_with_ref_and_config_tags_legacy_linear_profile_not_a() {
        let w = 128;
        let h = 128;
        let n = w * h;
        let mut src = vec![[128u8, 128, 128]; n];
        let mut dst = vec![[128u8, 128, 128]; n];
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 255) / w) as u8;
                src[y * w + x] = [r, 128, 128];
                dst[y * w + x] = [r.saturating_add(5), 128, 128];
            }
        }
        let config = ZensimConfig::default();
        let precomputed = precompute_reference_with_scales(&src, w, h, config.num_scales).unwrap();
        let result = compute_zensim_with_ref_and_config(&precomputed, &dst, w, h, config).unwrap();
        assert_eq!(
            result.profile(),
            crate::profile::ZensimProfile::LegacyLinearV0_2,
            "linear-scored result must not be tagged as MLP-scored profile A"
        );
    }

    /// V0_20a IW pool integration: `compute_iw_features=true` must
    /// emit 72 additional features (228 → 300 at 4 scales × 3 ch × 6),
    /// AND those features must differ from the masked block when both
    /// are enabled (weight polarity is opposite).
    #[test]
    fn compute_iw_features_emits_300_when_enabled() {
        let w = 128;
        let h = 128;
        let n = w * h;
        let mut src = vec![[128u8, 128, 128]; n];
        let mut dst = vec![[128u8, 128, 128]; n];
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 255) / w) as u8;
                let g = ((y * 255) / h) as u8;
                // Add a textured patch so the IW weights actually vary.
                let noise = if (x % 16) < 8 && (y % 16) < 8 { 32 } else { 0 };
                src[y * w + x] = [r.saturating_add(noise), g, 128];
                dst[y * w + x] = [r.saturating_add(noise).saturating_add(5), g, 125];
            }
        }

        // Off → 228 features (basic + peaks, masked OFF, iw OFF)
        let off = compute_zensim_with_config(
            &src,
            &dst,
            w,
            h,
            ZensimConfig {
                compute_all_features: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(off.features.len(), 228);

        // IW only → 228 + 72 = 300 features
        let iw_only = compute_zensim_with_config(
            &src,
            &dst,
            w,
            h,
            ZensimConfig {
                compute_all_features: true,
                compute_iw_features: true,
                iw_strength: 4.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(iw_only.features.len(), 300);

        // First 228 must match exactly (IW must not perturb basic/peaks).
        for i in 0..228 {
            let d = (off.features[i] - iw_only.features[i]).abs();
            assert!(
                d < 1e-9,
                "IW must not affect basic/peak features; index {} differs by {}",
                i,
                d,
            );
        }

        // IW block must produce non-zero output (we have textured patches).
        let iw_block_nonzero = iw_only.features[228..]
            .iter()
            .filter(|f| f.abs() > 1e-12)
            .count();
        assert!(iw_block_nonzero > 0, "IW block was all zeros");

        // When both extended_features and compute_iw_features are on,
        // total = 228 + 72 (masked) + 72 (IW) = 372. Masked and IW
        // pools have OPPOSITE weight polarity, so the two 72-feature
        // blocks must differ.
        let both = compute_zensim_with_config(
            &src,
            &dst,
            w,
            h,
            ZensimConfig {
                compute_all_features: true,
                extended_features: true,
                compute_iw_features: true,
                iw_strength: 4.0,
                extended_masking_strength: 4.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(both.features.len(), 372);

        let mut masked_iw_diff_count = 0usize;
        for i in 0..72 {
            let masked = both.features[228 + i];
            let iw = both.features[228 + 72 + i];
            if (masked - iw).abs() > 1e-9 {
                masked_iw_diff_count += 1;
            }
        }
        assert!(
            masked_iw_diff_count >= 60,
            "masked vs IW features should differ; only {}/72 differed",
            masked_iw_diff_count,
        );
    }

    /// `Zensim::compute_extended_features` returns 300 features at the
    /// default 4-scale, 3-channel layout, and produces a score that matches
    /// `Zensim::compute` to within a small numerical tolerance (the extra
    /// 72 features have zero weight, so the weighted score is unchanged).
    #[test]
    fn compute_extended_features_returns_300() {
        use crate::source::RgbSlice;
        let w = 64;
        let h = 64;
        let (src, dst) = make_gradient_pair(w, h);
        let src_img = RgbSlice::new(&src, w, h);
        let dst_img = RgbSlice::new(&dst, w, h);

        // Linear-profile-specific test: the score-equality assertion
        // only holds for profiles whose scoring is linear-weighted on
        // the 228 features (the extended 300 features just have
        // weight=0 in those slots). MLP profiles (A +) score
        // via a forward pass — extended features change the input
        // shape and break the equality. Pin to V0_2.
        let z = crate::Zensim::new(crate::ZensimProfile::PreviewV0_2);
        let standard = z.compute(&src_img, &dst_img).unwrap();
        let extended = z.compute_extended_features(&src_img, &dst_img).unwrap();

        assert_eq!(standard.features().len(), 228);
        assert_eq!(extended.features().len(), 300);
        // Score should match: extra features have zero weight in the
        // weighted-distance calculation.
        assert!(
            (standard.score() - extended.score()).abs() < 0.01,
            "standard score {} vs extended score {}",
            standard.score(),
            extended.score()
        );
        // Sanity: features 0..228 should agree wherever the standard path
        // populated them. The standard path skips channels whose weights are
        // all-zero (leaving 0.0 in those slots); the extended path forces
        // every channel/feature to be computed. So we only enforce equality
        // on slots the standard path actually filled.
        for i in 0..228 {
            let a = standard.features()[i];
            let b = extended.features()[i];
            if a != 0.0 {
                assert!(
                    (a - b).abs() < 1e-6,
                    "feature {i}: standard {a} vs extended {b}"
                );
            }
        }
        // The trailing 72 masked features should be non-negative.
        for (i, &f) in extended.features()[228..].iter().enumerate() {
            assert!(f >= 0.0, "extended feature {} is negative: {}", 228 + i, f);
        }
    }

    /// Helper: create a gradient test image pair.
    fn make_gradient_pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
        let n = w * h;
        let mut src = vec![[128u8, 128, 128]; n];
        let mut dst = vec![[128u8, 128, 128]; n];
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 255) / w) as u8;
                let g = ((y * 255) / h) as u8;
                let b = 128;
                src[y * w + x] = [r, g, b];
                dst[y * w + x] = [
                    r.saturating_add(10),
                    g.saturating_sub(5),
                    b.saturating_add(3),
                ];
            }
        }
        (src, dst)
    }

    /// Numerical equivalence: when both `extended_features` and
    /// `compute_iw_features` are enabled, the fused 2-mask SIMD path
    /// must produce the same masked block as running with
    /// `extended_features` alone, and the same IW block as running
    /// with `compute_iw_features` alone. This is the critical
    /// invariant for the 2026-05-15 perf optimization landing fused
    /// 2-mask SSIM/edge/MSE kernels.
    #[test]
    fn fused_2mask_matches_separate_paths() {
        let (w, h) = (128, 128);
        let n = w * h;
        let mut src = vec![[128u8, 128, 128]; n];
        let mut dst = vec![[128u8, 128, 128]; n];
        // Build a textured image so weights actually vary.
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 255) / w) as u8;
                let g = ((y * 255) / h) as u8;
                let b = (((x + y) * 255) / (2 * w)) as u8;
                let noise = if (x % 16) < 8 && (y % 16) < 8 { 32 } else { 0 };
                src[y * w + x] = [r.saturating_add(noise), g, b];
                dst[y * w + x] = [
                    r.saturating_add(noise).saturating_add(5),
                    g.saturating_sub(3),
                    b.saturating_add(2),
                ];
            }
        }

        let cfg_ext = ZensimConfig {
            extended_features: true,
            compute_iw_features: false,
            compute_all_features: true,
            extended_masking_strength: 4.0,
            iw_strength: 4.0,
            ..Default::default()
        };
        let cfg_iw = ZensimConfig {
            extended_features: false,
            compute_iw_features: true,
            compute_all_features: true,
            extended_masking_strength: 4.0,
            iw_strength: 4.0,
            ..Default::default()
        };
        let cfg_both = ZensimConfig {
            extended_features: true,
            compute_iw_features: true,
            compute_all_features: true,
            extended_masking_strength: 4.0,
            iw_strength: 4.0,
            ..Default::default()
        };
        let r_ext = compute_zensim_with_config(&src, &dst, w, h, cfg_ext).unwrap();
        let r_iw = compute_zensim_with_config(&src, &dst, w, h, cfg_iw).unwrap();
        let r_both = compute_zensim_with_config(&src, &dst, w, h, cfg_both).unwrap();

        assert_eq!(r_ext.features.len(), 300);
        assert_eq!(r_iw.features.len(), 300);
        assert_eq!(r_both.features.len(), 372);

        // First 228 features (basic + peaks) must agree across all 3.
        for i in 0..228 {
            let de = (r_ext.features[i] - r_both.features[i]).abs();
            let di = (r_iw.features[i] - r_both.features[i]).abs();
            assert!(
                de < 1e-9,
                "basic feature {} disagrees: ext_only {} vs both {}",
                i,
                r_ext.features[i],
                r_both.features[i]
            );
            assert!(
                di < 1e-9,
                "basic feature {} disagrees: iw_only {} vs both {}",
                i,
                r_iw.features[i],
                r_both.features[i]
            );
        }

        // Masked block (228..300 in r_ext, 228..300 in r_both) must agree
        // — same kernel called identically.
        let mut max_diff = 0.0f64;
        for i in 0..72 {
            let d = (r_ext.features[228 + i] - r_both.features[228 + i]).abs();
            let denom = r_ext.features[228 + i].abs().max(1e-9);
            // Use a relative tolerance — small numerical drift from
            // FMA reordering in the fused 2-mask vs single-mask kernels
            // is acceptable. The bound of 1e-4 relative captures any
            // structural correctness bug while permitting last-bit FMA
            // variability.
            let rel = d / denom;
            max_diff = max_diff.max(rel);
            assert!(
                rel < 1e-4,
                "masked feature {} differs: ext_only {} vs both {} (rel diff {:e})",
                i,
                r_ext.features[228 + i],
                r_both.features[228 + i],
                rel,
            );
        }
        // IW block (228..300 in r_iw, 300..372 in r_both) must agree.
        for i in 0..72 {
            let d = (r_iw.features[228 + i] - r_both.features[300 + i]).abs();
            let denom = r_iw.features[228 + i].abs().max(1e-9);
            let rel = d / denom;
            max_diff = max_diff.max(rel);
            assert!(
                rel < 1e-4,
                "iw feature {} differs: iw_only {} vs both {} (rel diff {:e})",
                i,
                r_iw.features[228 + i],
                r_both.features[300 + i],
                rel,
            );
        }
        // Sanity: not all zero
        assert!(
            (0.0..1e-4).contains(&max_diff),
            "max_diff out of range: {}",
            max_diff
        );
    }

    /// Extended features: default config produces same score as non-extended.
    #[test]
    fn extended_features_backward_compat() {
        let (w, h) = (64, 64);
        let (src, dst) = make_gradient_pair(w, h);

        let basic = compute_zensim_with_config(&src, &dst, w, h, ZensimConfig::default()).unwrap();

        let extended = compute_zensim_with_config(
            &src,
            &dst,
            w,
            h,
            ZensimConfig {
                extended_features: false,
                compute_all_features: true,
                ..Default::default()
            },
        )
        .unwrap();

        // Both produce 228 features now (peaks always included)
        assert_eq!(basic.features.len(), 228);
        assert_eq!(extended.features.len(), 228);
        // Score should be the same — compute_all forces all channels active but result is same
        assert!(
            (basic.score - extended.score).abs() < 0.01,
            "basic {} vs compute_all {}",
            basic.score,
            extended.score,
        );
    }

    /// Extended features produce 300 values and all are non-negative.
    #[test]
    fn extended_features_count_and_nonneg() {
        let (w, h) = (64, 64);
        let (src, dst) = make_gradient_pair(w, h);

        let result = compute_zensim_with_config(
            &src,
            &dst,
            w,
            h,
            ZensimConfig {
                extended_features: true,
                compute_all_features: true,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(
            result.features.len(),
            300,
            "Expected 25 × 3 × 4 = 300 features"
        );
        for (i, &f) in result.features.iter().enumerate() {
            assert!(f >= 0.0, "Feature {} is negative: {}", i, f);
        }
    }

    /// ssim_max >= ssim_4th >= ssim_mean ordering.
    #[test]
    fn extended_features_ordering() {
        let (w, h) = (64, 64);
        let (src, dst) = make_gradient_pair(w, h);

        let result = compute_zensim_with_config(
            &src,
            &dst,
            w,
            h,
            ZensimConfig {
                extended_features: true,
                compute_all_features: true,
                ..Default::default()
            },
        )
        .unwrap();

        // Feature layout (block-separated):
        //   [0..156)   scored: 13/ch × 3ch × 4 scales
        //   [156..228)  peaks: 6/ch × 3ch × 4 scales
        //   [228..300) masked: 6/ch × 3ch × 4 scales
        let scored_per_ch = FEATURES_PER_CHANNEL_BASIC; // 13
        let peaks_offset = 4 * scored_per_ch * 3; // 156
        let peaks_per_ch = 6;
        for scale in 0..4 {
            for ch in 0..3 {
                let scored_base = scale * scored_per_ch * 3 + ch * scored_per_ch;
                let peaks_base = peaks_offset + scale * peaks_per_ch * 3 + ch * peaks_per_ch;
                let ssim_mean = result.features[scored_base]; // scored[0]
                let ssim_4th = result.features[scored_base + 1]; // scored[1]
                let ssim_max = result.features[peaks_base]; // peaks[0]
                let ssim_p95 = result.features[peaks_base + 3]; // peaks[3]

                // max >= 4th >= mean (4th is L4 norm, always >= mean for non-negative values)
                assert!(
                    ssim_max >= ssim_4th - 1e-10,
                    "s{} c{}: max {:.6} < 4th {:.6}",
                    scale,
                    ch,
                    ssim_max,
                    ssim_4th,
                );
                assert!(
                    ssim_4th >= ssim_mean - 1e-10,
                    "s{} c{}: 4th {:.6} < mean {:.6}",
                    scale,
                    ch,
                    ssim_4th,
                    ssim_mean,
                );
                // p95 between 4th and max
                assert!(
                    ssim_p95 <= ssim_max + 1e-10,
                    "s{} c{}: p95 {:.6} > max {:.6}",
                    scale,
                    ch,
                    ssim_p95,
                    ssim_max,
                );
            }
        }
    }

    /// Identical images: all features zero.
    #[test]
    fn extended_features_identical_zero() {
        let (w, h) = (64, 64);
        let (src, _) = make_gradient_pair(w, h);

        let result = compute_zensim_with_config(
            &src,
            &src,
            w,
            h,
            ZensimConfig {
                extended_features: true,
                compute_all_features: true,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(result.score, 100.0);
        assert_eq!(result.features.len(), 300);
        for (i, &f) in result.features.iter().enumerate() {
            assert!(
                f.abs() < 1e-10,
                "Feature {} not zero for identical: {}",
                i,
                f
            );
        }
    }

    /// Masked features <= unmasked features (masking reduces).
    #[test]
    fn extended_masked_leq_unmasked() {
        let (w, h) = (64, 64);
        let (src, dst) = make_gradient_pair(w, h);

        let result = compute_zensim_with_config(
            &src,
            &dst,
            w,
            h,
            ZensimConfig {
                extended_features: true,
                compute_all_features: true,
                ..Default::default()
            },
        )
        .unwrap();

        // Feature layout (block-separated):
        //   [0..156)   scored: 13/ch × 3ch × 4 scales
        //   [156..228)  peaks: 6/ch × 3ch × 4 scales
        //   [228..300) masked: 6/ch × 3ch × 4 scales
        let scored_per_ch = FEATURES_PER_CHANNEL_BASIC; // 13
        let masked_offset = 4 * scored_per_ch * 3 + 4 * 6 * 3; // 156 + 72 = 228
        let masked_per_ch = 6;
        for scale in 0..4 {
            for ch in 0..3 {
                let scored_base = scale * scored_per_ch * 3 + ch * scored_per_ch;
                let masked_base = masked_offset + scale * masked_per_ch * 3 + ch * masked_per_ch;
                let ssim_mean = result.features[scored_base]; // scored[0]
                let ssim_4th = result.features[scored_base + 1]; // scored[1]
                let ssim_2nd = result.features[scored_base + 2]; // scored[2]
                let masked_ssim_mean = result.features[masked_base]; // masked[0]
                let masked_ssim_4th = result.features[masked_base + 1]; // masked[1]
                let masked_ssim_2nd = result.features[masked_base + 2]; // masked[2]

                // Masked values should be <= unmasked (mask weights ∈ [0,1])
                assert!(
                    masked_ssim_mean <= ssim_mean + 1e-10,
                    "s{} c{}: masked_mean {:.6} > mean {:.6}",
                    scale,
                    ch,
                    masked_ssim_mean,
                    ssim_mean,
                );
                assert!(
                    masked_ssim_4th <= ssim_4th + 1e-10,
                    "s{} c{}: masked_4th {:.6} > 4th {:.6}",
                    scale,
                    ch,
                    masked_ssim_4th,
                    ssim_4th,
                );
                assert!(
                    masked_ssim_2nd <= ssim_2nd + 1e-10,
                    "s{} c{}: masked_2nd {:.6} > 2nd {:.6}",
                    scale,
                    ch,
                    masked_ssim_2nd,
                    ssim_2nd,
                );
            }
        }
    }

    /// Regression: `derive_classification` must not panic when any
    /// per-detector score is NaN. The pre-cleanup implementation used
    /// `partial_cmp(...).unwrap()` which would unwind on NaN.
    ///
    /// We feed `alpha_error_correlation = NaN` (the only path where a
    /// NaN can reach `score_alpha`). The function must return cleanly
    /// — the NaN score is sorted to the most-negative end by
    /// `total_cmp`, so the legitimate non-NaN scores still pick the
    /// dominant category.
    #[cfg(feature = "classification")]
    #[test]
    fn derive_classification_is_nan_safe() {
        let delta_stats = DeltaStats {
            mean_delta: [0.0; 3],
            stddev_delta: [0.0; 3],
            max_abs_delta: [0.5 / 255.0; 3],
            signed_small_histogram: [[0u64; 7]; 3],
            native_max: 255.0,
            pixel_count: 10_000,
            pixels_differing: 100,
            pixels_differing_by_more_than_1: 0,
            has_alpha: true,
            alpha_max_delta: 0,
            alpha_pixels_differing: 0,
            src_histogram: [[0u64; 256]; 4],
            dst_histogram: [[0u64; 256]; 4],
            opaque_stats: None,
            semitransparent_stats: None,
            // Inject NaN — the audit-flagged failure mode. The pre-fix
            // code called `partial_cmp(...).unwrap()` and panicked here.
            alpha_error_correlation: Some(f64::NAN),
        };
        let dummy_result = ZensimResult::nan();
        let cls = derive_classification(&delta_stats, &dummy_result);
        // Must NOT have selected a NaN-confidence category. With the
        // `total_cmp` ordering, NaN sorts low and never wins.
        assert!(
            cls.confidence.is_finite(),
            "confidence must be finite even when an internal score is NaN, got {}",
            cls.confidence
        );
    }
}

#[cfg(test)]
mod pu_linear_extended_tests {
    use super::*;
    use crate::{PixelFormat, StridedBytes};

    #[test]
    fn pu_linear_extended_features_width_matches_srgb_extraction() {
        let (w, h) = (64usize, 64usize);
        let mut refe = vec![0u8; w * h * 3];
        for (i, v) in refe.iter_mut().enumerate() {
            *v = ((i * 37) % 251) as u8;
        }
        let mut dist = refe.clone();
        for v in dist.iter_mut().step_by(7) {
            *v = v.saturating_sub(12);
        }
        let z = Zensim::new(ZensimProfile::PreviewV0_2);
        let src = StridedBytes::try_new(&refe, w, h, w * 3, PixelFormat::Srgb8Rgb).unwrap();
        let dst = StridedBytes::try_new(&dist, w, h, w * 3, PixelFormat::Srgb8Rgb).unwrap();
        let srgb = z.compute_extended_features(&src, &dst).unwrap();

        let to_nits = |px: &[u8]| -> Vec<f32> {
            px.iter()
                .map(|&v| {
                    let x = v as f32 / 255.0;
                    let lin = if x <= 0.04045 {
                        x / 12.92
                    } else {
                        ((x + 0.055) / 1.055).powf(2.4)
                    };
                    lin * 203.0
                })
                .collect()
        };
        let rn = to_nits(&refe);
        let dn = to_nits(&dist);
        let pu = z
            .compute_pu_linear_extended_features(&rn, &dn, w, h, w * 3, w * 3)
            .unwrap();
        assert_eq!(
            srgb.features().len(),
            pu.features().len(),
            "feature layout must match across paths"
        );
        assert!(pu.features().len() >= 300, "extended block missing");
        // Identical-input contract holds on the PU path too.
        let ident = z
            .compute_pu_linear_extended_features(&rn, &rn, w, h, w * 3, w * 3)
            .unwrap();
        assert!((ident.score() - 100.0).abs() < 1e-9);
    }
}
