//! # zensim-experimental
//!
//! Historical and experimental [`zensim`] metric profiles, preserved for
//! long-term research reference.
//!
//! These profiles and their trained MLP bakes were moved out of the
//! published `zensim` crate to keep its download size and permanent API
//! surface minimal. The published `zensim` keeps only the shipping profile
//! `A` (plus the `Custom` extension point). Everything else — the
//! back-compat linear `PreviewV0_2` (reconstructed via [`preview_v0_2`]),
//! the `PreviewV0_4` D2 ensemble, the whole `PreviewV0_5*` SOTA-trail
//! matrix, `A_Phone`, `LinearBounded`, and `PreviewV0_5Linear` — lives
//! here. (The linear `PreviewV0_1` profile was dropped: its weight vector
//! was removed from `zensim` and no research tooling here needs it.)
//!
//! Every profile is reconstructed through zensim's stable
//! [`zensim::profile::ProfileParams::builder`] +
//! [`zensim::ZensimProfile::Custom`] extension point, pointing at a bake
//! embedded in this crate via `include_bytes!`. Because all the head /
//! spline / per-codec-calibration behaviour lives in the bake metadata,
//! each profile here is **bit-identical** to when it lived in `zensim`.
//!
//! This crate is **never published** (`publish = false`). It exists so the
//! research lineage stays runnable from the workspace — training, eval, and
//! regression tooling construct these profiles for comparison — without
//! shipping ~7.7 MB of historical bakes to every `zensim` consumer.
//!
//! ## Usage
//!
//! ```ignore
//! use zensim::{Zensim, RgbSlice};
//! let z = Zensim::new(zensim_experimental::preview_v0_5_tuner_v4());
//! // ... z.compute(&reference, &distorted)? as with any built-in profile.
//! ```
//!
//! Each function returns a ready-to-use [`zensim::ZensimProfile::Custom`]
//! whose `name()` matches the original built-in variant's name string
//! (e.g. `"zensim-preview-v0.5-tuner-v4"`), so existing tooling that keys
//! reports / filenames on the profile name keeps working unchanged.

#![forbid(unsafe_code)]

// Per-codec affine calibration for the Tuner profile. Relocated from
// `zensim::codec_calibration` (2026-06-12, imazen/zensimx#47): the zensim
// runtime parses its own private per-codec calibration from bake metadata,
// so these standalone types' only consumer is this crate's tooling/examples.
pub mod codec_calibration;

use std::sync::OnceLock;
use zensim::ZensimProfile;
use zensim::profile::ProfileParams;

/// Embedded historical bakes. Each is a thin `include_bytes!` wrapper so
/// the bytes resolve lazily through the `fn() -> &'static [u8]` builder
/// slots. The byte files live in `zensim-experimental/weights/`.
mod bakes {
    // PreviewV0_4 D2 multi-bake: V39 primary + V_20 input-shaping secondary.
    pub fn v0_4_primary() -> &'static [u8] {
        include_bytes!("../weights/v39_v32plus_spline_seed17_2026-05-25.bin")
    }
    pub fn v0_4_secondary() -> &'static [u8] {
        include_bytes!("../weights/v0_20_is_calibrated_2026-05-15.bin")
    }
    // PreviewV0_5 balanced / compression ships + the ensemble classifier.
    pub fn balanced() -> &'static [u8] {
        include_bytes!("../weights/v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin")
    }
    pub fn compression() -> &'static [u8] {
        include_bytes!("../weights/v_compression_persample_2026-05-18.bin")
    }
    pub fn ensemble_classifier() -> &'static [u8] {
        include_bytes!("../weights/v05_ensemble_classifier_2026-05-18.bin")
    }
    // Tuner trail.
    pub fn tuner() -> &'static [u8] {
        include_bytes!("../weights/v_tuner_2026-05-18.bin")
    }
    pub fn cross_codec() -> &'static [u8] {
        include_bytes!("../weights/v_cross_codec_2026-05-19.bin")
    }
    pub fn tuner_v2() -> &'static [u8] {
        include_bytes!("../weights/v_tuner_v6_2026-05-19.bin")
    }
    pub fn tuner_v3() -> &'static [u8] {
        include_bytes!("../weights/v_tuner_v9_2026-05-20.bin")
    }
    // V10 score-space reallocation ships.
    pub fn balanced_v2() -> &'static [u8] {
        include_bytes!("../weights/v_balanced_v2_2026-05-20.bin")
    }
    pub fn compression_v2() -> &'static [u8] {
        include_bytes!("../weights/v_compression_v2_2026-05-20.bin")
    }
    pub fn balanced_v3() -> &'static [u8] {
        include_bytes!("../weights/v_balanced_v3_2026-05-20.bin")
    }
    pub fn compression_v3() -> &'static [u8] {
        include_bytes!("../weights/v_compression_v3_2026-05-20.bin")
    }
    pub fn tuner_v4() -> &'static [u8] {
        include_bytes!("../weights/v_tuner_v10_2026-05-20.bin")
    }
    // V11-E per-codec-affine variants (opt-in research artifacts).
    pub fn tuner_v4_calibrated() -> &'static [u8] {
        include_bytes!("../weights/v_tuner_v4_per_codec_2026-05-20.bin")
    }
    pub fn balanced_v3_calibrated() -> &'static [u8] {
        include_bytes!("../weights/v_balanced_v3_per_codec_2026-05-20.bin")
    }
    pub fn compression_v3_calibrated() -> &'static [u8] {
        include_bytes!("../weights/v_compression_v3_per_codec_2026-05-20.bin")
    }
    // Cell 5 linear + A_Phone display target.
    pub fn cell5_linear() -> &'static [u8] {
        include_bytes!("../weights/v02_372feat_cell5_2026-05-28.bin")
    }
    pub fn a_phone() -> &'static [u8] {
        include_bytes!("../weights/zensim_b_phone_oled_2026-05-26.bin")
    }
    // The CVVDP-desktop bake (`zensim_b_desktop_2026-05-25.bin`) is preserved
    // on disk for reference but was never wired to a `ProfileParams`, so it
    // has no profile function here.
}

macro_rules! profile_fn {
    ($(#[$m:meta])* $fn:ident, $name:literal, $build:expr) => {
        $(#[$m])*
        pub fn $fn() -> ZensimProfile {
            static PARAMS: OnceLock<ProfileParams> = OnceLock::new();
            ZensimProfile::Custom {
                params: PARAMS.get_or_init(|| $build),
                name: $name,
            }
        }
    };
}

profile_fn!(
    /// `A_Phone` — generation-A, modern-phone OLED display target
    /// (`modern_oled_phone_indoor`, ~110 PPD, 400 nit indoor SDR). Same
    /// 372-feature architecture as `ZensimProfile::A`; phone-CVVDP bake.
    a_phone,
    "zensim-a-phone",
    ProfileParams::builder()
        .mlp(bakes::a_phone)
        .skip_score_mapping(true)
        .extended_features(true)
        .compute_iw_features(true)
        .extrapolate_score(true)
        .build()
);

profile_fn!(
    /// `PreviewV0_2` — the historical linear general-ranking profile,
    /// reconstructed as a `Custom` profile after it was removed as a
    /// built-in `zensim` variant. The `ProfileParams::builder()` defaults
    /// (the `WEIGHTS_PREVIEW_V0_2` vector, `blur_radius = 5`,
    /// `blur_passes = 1`, `num_scales = 4`, `100 − 18·d^0.7`, all
    /// dispositions off, no MLP) are byte-for-byte the former
    /// `PROFILE_PREVIEW_V0_2` static, so scores are identical to the
    /// removed built-in. Concordance-filtered 218k pairs, Nelder-Mead
    /// SROCC = 0.9960.
    preview_v0_2,
    "zensim-preview-v0.2",
    ProfileParams::builder().build()
);

profile_fn!(
    /// `LinearBounded` — correct-by-construction bounded squash
    /// `100·exp(−(a/100)·d^b)` over the V0_2 non-negative weights. Bounded
    /// `[0, 100]`, equals 100 iff identical, monotone non-increasing in
    /// every error feature — by construction, on the whole input domain.
    /// Same rank order as [`preview_v0_2`].
    linear_bounded,
    "zensim-linear-bounded",
    ProfileParams::builder().bounded_squash(true).build()
);

profile_fn!(
    /// Preview v0.4 — D2 multi-bake ensemble: V39 primary + V_20
    /// input-shaping secondary, raw-output mix at primary weight 0.4,
    /// soft-clamped.
    preview_v0_4,
    "zensim-preview-v0.4",
    ProfileParams::builder()
        .mlp(bakes::v0_4_primary)
        .mlp_secondary(bakes::v0_4_secondary, 0.4)
        .skip_score_mapping(true)
        .soft_clamp_score(true)
        .build()
);

profile_fn!(
    /// Preview v0.5 — balanced-trail ship (back-compat name for callers
    /// that matched `PreviewV0_5` before the two-trail framework). Same
    /// bake/params as [`preview_v0_5_balanced`].
    preview_v0_5,
    "zensim-preview-v0.5",
    ProfileParams::builder()
        .mlp(bakes::balanced)
        .skip_score_mapping(true)
        .extended_features(true)
        .build()
);

profile_fn!(
    /// Preview v0.5 balanced trail — V_22-mix-LARGE+iwssim, 300→128→1
    /// score-shaped MLP. Best balanced-corpus coverage.
    preview_v0_5_balanced,
    "zensim-preview-v0.5-balanced",
    ProfileParams::builder()
        .mlp(bakes::balanced)
        .skip_score_mapping(true)
        .extended_features(true)
        .build()
);

profile_fn!(
    /// Preview v0.5 compression trail — V_24-per-sample-α, 300→128→128
    /// (per_sample_alpha_head), soft-clamped. Wins CID22 / AIC-3.
    preview_v0_5_compression,
    "zensim-preview-v0.5-compression",
    ProfileParams::builder()
        .mlp(bakes::compression)
        .skip_score_mapping(true)
        .extended_features(true)
        .soft_clamp_score(true)
        .build()
);

profile_fn!(
    /// Preview v0.5 ensemble — a 300→64→1 classifier routes each pair to
    /// either the balanced (primary) or compression bake by its sign.
    preview_v0_5_ensemble,
    "zensim-preview-v0.5-ensemble",
    ProfileParams::builder()
        .mlp(bakes::balanced)
        .ensemble(bakes::ensemble_classifier, bakes::compression)
        .skip_score_mapping(true)
        .extended_features(true)
        .soft_clamp_score(true)
        .build()
);

profile_fn!(
    /// Preview v0.5 tuner — V_24-per-sample-α s2, affine-calibrated to the
    /// 0..100 dial. 372-feature input.
    preview_v0_5_tuner,
    "zensim-preview-v0.5-tuner",
    ProfileParams::builder()
        .mlp(bakes::tuner)
        .skip_score_mapping(true)
        .extended_features(true)
        .compute_iw_features(true)
        .build()
);

profile_fn!(
    /// Preview v0.5 cross-codec (opt-in) — per-sample-α + cross-codec
    /// equivalence-pair loss. Soft-clamped. 372-feature input.
    preview_v0_5_cross_codec,
    "zensim-preview-v0.5-cross-codec",
    ProfileParams::builder()
        .mlp(bakes::cross_codec)
        .skip_score_mapping(true)
        .extended_features(true)
        .compute_iw_features(true)
        .soft_clamp_score(true)
        .build()
);

profile_fn!(
    /// Preview v0.5 tuner v2 — per-sample-α + tanh-output-head pin.
    /// 372-feature input.
    preview_v0_5_tuner_v2,
    "zensim-preview-v0.5-tuner-v2",
    ProfileParams::builder()
        .mlp(bakes::tuner_v2)
        .skip_score_mapping(true)
        .extended_features(true)
        .compute_iw_features(true)
        .build()
);

profile_fn!(
    /// Preview v0.5 tuner v3 — tuner v2 + post-network PCHIP spline
    /// (JND→60, JOD→30). 372-feature input.
    preview_v0_5_tuner_v3,
    "zensim-preview-v0.5-tuner-v3",
    ProfileParams::builder()
        .mlp(bakes::tuner_v3)
        .skip_score_mapping(true)
        .extended_features(true)
        .compute_iw_features(true)
        .build()
);

profile_fn!(
    /// Preview v0.5 balanced v2 — balanced bake + 7-knot PCHIP spline.
    /// 300-feature input (no IW pool).
    preview_v0_5_balanced_v2,
    "zensim-preview-v0.5-balanced-v2",
    ProfileParams::builder()
        .mlp(bakes::balanced_v2)
        .skip_score_mapping(true)
        .extended_features(true)
        .build()
);

profile_fn!(
    /// Preview v0.5 compression v2 — compression bake + 7-knot PCHIP
    /// spline. 300-feature input (no IW pool).
    preview_v0_5_compression_v2,
    "zensim-preview-v0.5-compression-v2",
    ProfileParams::builder()
        .mlp(bakes::compression_v2)
        .skip_score_mapping(true)
        .extended_features(true)
        .build()
);

profile_fn!(
    /// Preview v0.5 balanced v3 (V10 score-space) — unclamped spline
    /// extrapolation (pathological input maps below 0). 300-feature input.
    preview_v0_5_balanced_v3,
    "zensim-preview-v0.5-balanced-v3",
    ProfileParams::builder()
        .mlp(bakes::balanced_v3)
        .skip_score_mapping(true)
        .extended_features(true)
        .extrapolate_score(true)
        .build()
);

profile_fn!(
    /// Preview v0.5 compression v3 (V10 score-space) — unclamped spline
    /// extrapolation. 300-feature input.
    preview_v0_5_compression_v3,
    "zensim-preview-v0.5-compression-v3",
    ProfileParams::builder()
        .mlp(bakes::compression_v3)
        .skip_score_mapping(true)
        .extended_features(true)
        .extrapolate_score(true)
        .build()
);

profile_fn!(
    /// Preview v0.5 tuner v4 (V10 score-space) — tuner v3 network + V10
    /// spline, unclamped. 372-feature input.
    preview_v0_5_tuner_v4,
    "zensim-preview-v0.5-tuner-v4",
    ProfileParams::builder()
        .mlp(bakes::tuner_v4)
        .skip_score_mapping(true)
        .extended_features(true)
        .compute_iw_features(true)
        .extrapolate_score(true)
        .build()
);

profile_fn!(
    /// Preview v0.5 tuner v4, per-codec-calibrated (V11-E, opt-in
    /// research artifact). Falsified: per-codec affine raised holdout
    /// cross-codec stddev. 372-feature input.
    preview_v0_5_tuner_v4_calibrated,
    "zensim-preview-v0.5-tuner-v4-calibrated",
    ProfileParams::builder()
        .mlp(bakes::tuner_v4_calibrated)
        .skip_score_mapping(true)
        .extended_features(true)
        .compute_iw_features(true)
        .extrapolate_score(true)
        .build()
);

profile_fn!(
    /// Preview v0.5 balanced v3, per-codec-calibrated (V11-E, opt-in).
    /// NOTE: this preserves the original profile's `compute_iw_features =
    /// true` (372-feature) disposition, which differs from the
    /// non-calibrated [`preview_v0_5_balanced_v3`] (300-feature) — a latent
    /// inconsistency carried over verbatim from the pre-relocation source.
    preview_v0_5_balanced_v3_calibrated,
    "zensim-preview-v0.5-balanced-v3-calibrated",
    ProfileParams::builder()
        .mlp(bakes::balanced_v3_calibrated)
        .skip_score_mapping(true)
        .extended_features(true)
        .compute_iw_features(true)
        .extrapolate_score(true)
        .build()
);

profile_fn!(
    /// Preview v0.5 compression v3, per-codec-calibrated (V11-E, opt-in).
    /// NOTE: preserves the original `compute_iw_features = true`
    /// (372-feature) disposition, differing from the non-calibrated
    /// [`preview_v0_5_compression_v3`] (300-feature) — verbatim carry-over.
    preview_v0_5_compression_v3_calibrated,
    "zensim-preview-v0.5-compression-v3-calibrated",
    ProfileParams::builder()
        .mlp(bakes::compression_v3_calibrated)
        .skip_score_mapping(true)
        .extended_features(true)
        .compute_iw_features(true)
        .extrapolate_score(true)
        .build()
);

profile_fn!(
    /// Preview v0.5 linear ("Cell 5") — pure 372→1 BVLS linear fit + 18-knot
    /// PCHIP dial spline, 4.8 KB. Lowest CID22 Z-RMSE of any zensim bake.
    preview_v0_5_linear,
    "zensim-preview-v0.5-linear",
    ProfileParams::builder()
        .mlp(bakes::cell5_linear)
        .skip_score_mapping(true)
        .extended_features(true)
        .compute_iw_features(true)
        .extrapolate_score(true)
        .build()
);
