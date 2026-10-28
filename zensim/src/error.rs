//! Error types for zensim image comparison.

/// Errors from zensim computation.
///
/// This enum is `#[non_exhaustive]` — additional variants may be added in
/// future minor releases without a major version bump. Match with a `_`
/// arm to remain forward-compatible.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ZensimError {
    /// The configured integrity head rejected a steering reconstruction.
    /// This is a model detection, not proof of the underlying software cause.
    #[error("corruption head rejected the steering reconstruction")]
    CorruptionDetected,
    /// Source and distorted images have different pixel counts.
    #[error("Source and distorted images must have the same dimensions")]
    DimensionMismatch,

    /// Width or height is zero. Since 0.3.0, sub-64px images (down to
    /// 1×1) are reflect-padded to the pyramid minimum and score normally;
    /// only empty (zero-dimension) inputs are rejected.
    #[error("Images must have non-zero width and height")]
    ImageTooSmall,

    /// `pixels.len()` does not equal `width * height`.
    #[error("Image data length does not match width * height")]
    InvalidDataLength,

    /// Row stride is smaller than `width * bytes_per_pixel`.
    #[error("Row stride is smaller than width * bytes_per_pixel")]
    InvalidStride,

    /// Image dimensions exceed the configured `max_pixels` cap, or the
    /// notional pixel/byte count overflows `usize` on the current target
    /// (e.g. `width * height` wraps on 32-bit / wasm32). Use
    /// [`Zensim::with_max_pixels`](crate::Zensim::with_max_pixels) to
    /// raise or remove the cap.
    #[error("Image dimensions exceed the configured maximum or overflow usize")]
    ImageTooLarge,

    /// `features.len()` does not equal `weights.len()` in
    /// `try_score_from_features` (a `training`-feature API). Both
    /// slices must have the same length (one weight per feature).
    #[error("features and weights must have the same length")]
    FeatureWeightsLengthMismatch,

    /// A pixel format reached a code path that does not yet handle it.
    ///
    /// This currently fires from the optional `classification` feature's
    /// delta-stats pass when a format other than sRGB-8 (RGB/RGBA/BGRA),
    /// sRGB-16 (RGBA), or linear-F32 (RGBA) is presented. The enum is
    /// `#[non_exhaustive]`, so this variant is non-breaking — match it
    /// alongside a `_` arm if you handle it.
    #[error("pixel format is not supported by this code path")]
    UnsupportedPixelFormat,

    /// Loading a trained MLP bake's bytes failed.
    ///
    /// Distinct from [`Self::InvalidDataLength`] — covers
    /// header / version / structural parse failures, not feature-length
    /// mismatches. `reason` is a `&'static str` describing the parse
    /// stage that failed (kept static so the error type stays `Copy`).
    #[error("failed to load MLP bake: {reason}")]
    ModelLoadFailed { reason: &'static str },

    /// Running an MLP forward pass on a loaded bake failed.
    ///
    /// Covers shape mismatches between the bake's declared layers, the
    /// supplied features, and any per-sample-α / hybrid-head metadata
    /// (e.g. metadata declares `n_hidden=32` but the bake's final layer
    /// outputs 16 values). Distinct from [`Self::ModelLoadFailed`] —
    /// the bake parsed cleanly but the forward call could not produce
    /// a score. `reason` is a `&'static str` describing the failure
    /// site.
    #[error("MLP forward failed: {reason}")]
    ModelForwardFailed { reason: &'static str },

    /// An [`ImageSource`](crate::ImageSource) flagged itself HDR
    /// ([`ImageSource::is_hdr`](crate::ImageSource::is_hdr)), which the SDR
    /// entry points cannot score: they expect display-encoded SDR data, and
    /// running the SDR pipeline on HDR-coded values would silently produce
    /// meaningless scores — we refuse instead. Fires when either side of a
    /// pair is HDR-flagged.
    ///
    /// HDR pairs ARE scorable: decode to **absolute-luminance linear RGB
    /// (cd/m²)** and call
    /// [`Zensim::compute_pu_linear`](crate::Zensim::compute_pu_linear)
    /// (interleaved; planar variant
    /// [`compute_pu_linear_planar`](crate::Zensim::compute_pu_linear_planar)) —
    /// the PU21 front-end. Its output calibration against a trained HDR
    /// bake is still open — see
    /// [imazen/zensimx#38](https://github.com/imazen/zensimx/issues/38).
    #[error(
        "HDR input cannot be scored by the SDR entry points — decode to \
         absolute-luminance linear RGB and use compute_pu_linear"
    )]
    HdrInputRequiresPuPath,

    /// The computation was cancelled through the cooperative-cancellation
    /// token installed via [`Zensim::with_stop`](crate::Zensim::with_stop).
    ///
    /// `reason` is the [`enough::StopReason`] reported by the token
    /// (explicit cancellation vs. timeout). Partial results are discarded;
    /// re-run the computation to retry.
    #[error("computation cancelled: {reason:?}")]
    Cancelled {
        /// Why the stop token fired.
        reason: enough::StopReason,
    },
}

impl From<enough::StopReason> for ZensimError {
    fn from(reason: enough::StopReason) -> Self {
        ZensimError::Cancelled { reason }
    }
}

/// Pixel format conversion error from the zenpixels adapter.
#[cfg(feature = "zenpixels")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("Unsupported pixel format: {0}")]
pub struct UnsupportedFormat(pub &'static str);
