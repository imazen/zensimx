//! # zensim
//!
//! Fast psychovisual image similarity metric combining ideas from
//! SSIMULACRA2 and butteraugli. Multi-scale SSIM + edge + high-frequency
//! features in XYB color space, with trained weights and AVX2/AVX-512 SIMD.
//!
//! ## Quick start
//!
//! ```
//! use zensim::{Zensim, ZensimProfile, RgbSlice};
//! # let (src_pixels, dst_pixels) = (vec![[0u8; 3]; 64], vec![[0u8; 3]; 64]);
//! let z = Zensim::new(ZensimProfile::codec_target());
//! let source = RgbSlice::new(&src_pixels, 8, 8);
//! let distorted = RgbSlice::new(&dst_pixels, 8, 8);
//! let result = z.compute(&source, &distorted)?;
//! println!("{}: {:.2}", result.profile(), result.score());
//! # Ok::<(), zensim::ZensimError>(())
//! ```
//!
//! ## Batch comparison (one reference, many distorted)
//!
//! ```
//! use zensim::{Zensim, ZensimProfile, RgbSlice};
//! # let (ref_pixels, width, height) = (vec![[0u8; 3]; 64], 8usize, 8usize);
//! # let distorted_images: Vec<Vec<[u8; 3]>> = vec![];
//! let z = Zensim::new(ZensimProfile::codec_target());
//! let source = RgbSlice::new(&ref_pixels, width, height);
//! let precomputed = z.precompute_reference(&source)?;
//! for dst_pixels in &distorted_images {
//!     let dst = RgbSlice::new(dst_pixels, width, height);
//!     let result = z.compute_with_ref(&precomputed, &dst)?;
//!     println!("score: {:.2}", result.score());
//! }
//! # Ok::<(), zensim::ZensimError>(())
//! ```
//!
//! ## Encoder closed-loop pattern (per-window quality probe)
//!
//! For codec quantization controllers that want a per-window quality signal
//! mid-encode (rather than one global score post-encode), pre-slice the
//! source into row-windows that match the encoder's natural emission
//! cadence, build a [`PrecomputedReference`] per window once, then call
//! [`compute_with_ref_into`](crate::Zensim::compute_with_ref_into) per
//! window with a shared [`ZensimScratch`] to amortize allocations.
//!
//! ```
//! use zensim::{Zensim, ZensimProfile, RgbSlice, ZensimScratch};
//! # let (source_pixels, width, height) = (vec![[0u8; 3]; 16 * 8], 16usize, 8usize);
//! # const WINDOW_ROWS: usize = 4;
//! let z = Zensim::new(ZensimProfile::codec_target());
//!
//! // Up front: pre-slice the source into row-windows. Build one
//! // PrecomputedReference per window (each is small — only that
//! // window's pyramid).
//! let mut window_refs = Vec::new();
//! let mut y = 0;
//! while y < height {
//!     let h = (height - y).min(WINDOW_ROWS);
//!     if h < 8 { break; }  // skip tail slivers (they'd reflect-pad, costing accuracy)
//!     let start = y * width;
//!     let end = start + h * width;
//!     let win = RgbSlice::new(&source_pixels[start..end], width, h);
//!     window_refs.push((y, h, z.precompute_reference(&win)?));
//!     y += WINDOW_ROWS;
//! }
//!
//! // Per encode iMCU window i: compute canonical zensim on the slice,
//! // reusing scratch buffers across all calls.
//! let mut scratch = ZensimScratch::new();
//! for (y0, h, pre) in &window_refs {
//!     # let distorted_window: Vec<[u8; 3]> = vec![[0u8; 3]; (*h) * width];
//!     let dst = RgbSlice::new(&distorted_window, width, *h);
//!     let result = z.compute_with_ref_into(pre, &dst, &mut scratch)?;
//!     // Feed result.score() into your AQ controller…
//!     let _ = result.score();
//! }
//! # Ok::<(), zensim::ZensimError>(())
//! ```
//!
//! ### Caveats — read before deploying as a control signal
//!
//! Empirical validation against full-image diffmap as ground-truth (see
//! parked branch `explored/issue-16-option-d-slice-canonical-and-zenwebp-data`
//! and the `slice_localized_distortion.rs` / `slice_real_codec_localization.rs`
//! examples in `zensim-regress`):
//!
//! - **Truncated pyramid context.** A 64-row window's pyramid is 64→32→16→8
//!   rows at scales 0–3. SSIM uses an 11×11 window (radius 5), so scale 3
//!   has 0 valid SSIM rows in 8 input rows; scale 2 has ~6 valid rows of 16.
//!   The features at coarser scales are dominated by mirror-padded boundary
//!   data, not content. The trained weights at scales 1–3 carry ~94% of
//!   the weight mass.
//!
//! - **On *synthetic* injected-distortion** (one window heavily damaged,
//!   others clean): per-window canonical correctly identifies the damaged
//!   window 100% of the time. The clean-vs-damaged gap dwarfs any
//!   pad-noise contribution.
//!
//! - **On *real* codec output** (mozjpeg / zenjpeg sRGB / zenjpeg XYB /
//!   zenavif at q60–q90): per-window canonical's top-1 ranking matches
//!   ground-truth (full-image diffmap aggregated per window) only ~24%
//!   of the time. Top-3 overlap is 1.71/3 ≈ 57%. Mean SROCC = 0.57.
//!
//! - **Treat the per-window signal as a *guidance* signal, not a per-window
//!   precision oracle.** Pair with EMA smoothing across windows and an
//!   iteration-boundary canonical check ([`Zensim::compute_with_ref`])
//!   for the global score.
//!
//! - **Cost.** With K windows of ~`H/K` rows each, total per-iteration
//!   compute is roughly the same as one full-image canonical compute,
//!   distributed across the encoder's iMCU emissions. The
//!   [`PrecomputedReference`] builds amortize across all encoder
//!   iterations against the same source.
//!
//! ## RGBA support
//!
//! ```
//! use zensim::{Zensim, ZensimProfile, RgbaSlice};
//! # let (src_rgba, dst_rgba) = (vec![[0u8; 4]; 64], vec![[0u8; 4]; 64]);
//! let z = Zensim::new(ZensimProfile::codec_target());
//! let source = RgbaSlice::new(&src_rgba, 8, 8);
//! let distorted = RgbaSlice::new(&dst_rgba, 8, 8);
//! let result = z.compute(&source, &distorted)?;
//! # Ok::<(), zensim::ZensimError>(())
//! ```
//!
//! ## zenpixels support
//!
//! With the `zenpixels` feature, any `zenpixels::PixelSlice` or
//! `zenpixels::PixelBuffer` can be used directly via `ZenpixelsSource`:
//!
//! ```ignore
//! use zensim::{Zensim, ZensimProfile, ZenpixelsSource};
//!
//! let source = ZenpixelsSource::try_from_slice(&pixel_slice)?;
//! let distorted = ZenpixelsSource::try_from_slice(&other_slice)?;
//! let result = Zensim::new(ZensimProfile::codec_target()).compute(&source, &distorted)?;
//! ```
//!
//! Supported: Rgb8, Rgba8, Bgra8, Rgbx8, Bgrx8, Rgba16, RgbaF32 (sRGB/BT.709/linear).
//! Premultiplied alpha is un-premultiplied automatically. RGBX/BGRX padding bytes
//! are treated as opaque automatically. HDR (PQ, HLG) and grayscale are rejected
//! with `UnsupportedFormat` (zenpixels feature).
//!
//! ## Input requirements
//!
//! - **Color space:** All inputs must be **sRGB-encoded** (gamma ~2.2) — the
//!   standard output of JPEG, PNG, and WebP decoders. For linear-light data,
//!   use `PixelFormat::LinearF32Rgba` via [`StridedBytes`].
//! - **Wide gamut:** Display P3 and BT.2020 primaries are accepted via
//!   [`ColorPrimaries`] on [`StridedBytes`] — gamut-mapped to sRGB internally.
//!   Passing wide-gamut data as sRGB will produce incorrect scores.
//! - **Pixel formats:** [`RgbSlice`] (sRGB u8), [`RgbaSlice`] (sRGB u8 + alpha),
//!   `imgref::ImgRef` (sRGB u8, stride-aware, default feature),
//!   `ZenpixelsSource` (zenpixels `PixelSlice`/`PixelBuffer`, `zenpixels` feature),
//!   [`StridedBytes`] (any of `Srgb8Rgb`, `Srgb8Rgba`, `Srgb8Bgra`,
//!   `Srgb16Rgba`, `LinearF32Rgba`), or implement [`ImageSource`] directly.
//! - **Alpha:** RGBA inputs are composited over a deterministic noise
//!   background so alpha differences are detected without the structured-pattern
//!   amplification of a checkerboard. Supports `Straight` and `Opaque` alpha modes.
//! - **Dimensions:** Both images must be the same width × height. Any
//!   non-zero size scores: sub-64px inputs (down to 1×1) are reflect-padded
//!   to the pyramid minimum internally.
//!
//! ## Score semantics
//!
//! 100 = identical, higher = more similar. How a raw result becomes a
//! score depends on the profile:
//!
//! - [`ZensimProfile::B`] (the canonical profile — what
//!   [`ZensimProfile::codec_target`] and [`ZensimProfile::latest_preview`]
//!   return): a deterministic LINEAR core over the 372 features + a monotone
//!   PCHIP dial spline. Byte-reproducible (no training seed). SDR content;
//!   HDR (absolute-nits) content routes to [`ZensimProfile::BHdr`]
//!   automatically. Pathological inputs can score below 0 (the spline
//!   extrapolates past its knots).
//! - [`ZensimProfile::A`] (**deprecated** — the prior canonical profile,
//!   the v47 MLP): MLP forward pass + monotone PCHIP dial spline, identity
//!   at ≈ 97.7. Behind the default-on `deprecated-profiles` feature; disable
//!   it to drop `A`. Superseded by `B`.
//! - [`ZensimProfile::PreviewV0_1`] / [`ZensimProfile::PreviewV0_2`]
//!   (linear profiles): `100 - 18 × d^0.7` where `d` is the per-scale
//!   weighted feature distance. Calibrated from 0–100 on 344k training
//!   pairs; extreme distortions can score below 0. Externally-defined
//!   linear/MLP profiles are also reachable via [`ZensimProfile::Custom`]
//!   (reconstructed in the `zensim-experimental` crate).
//!
//! [`ZensimResult`] also provides [`approx_ssim2()`](ZensimResult::approx_ssim2),
//! [`approx_dssim()`](ZensimResult::approx_dssim), and
//! [`approx_butteraugli()`](ZensimResult::approx_butteraugli) for direct
//! metric approximations. The [`mapping`] module has bidirectional interpolation
//! tables for score-level conversions.
//!
//! ## Determinism
//!
//! Deterministic for the same input on the same architecture. Cross-architecture
//! results (e.g. AVX2 vs scalar vs AVX-512) may differ by small ULP due to
//! different FMA contraction behavior.
//!
//! ## Design
//!
//! - **XYB color space** — cube root LMS, same perceptual space as ssimulacra2/butteraugli
//! - **Modified SSIM** — ssimulacra2's variant: drops the luminance denominator
//!   (no C1), uses `1 - (mu1-mu2)²` directly. Correct for perceptually-uniform spaces.
//! - **19 features per channel per scale** — 13 basic (SSIM, edge artifact/detail
//!   loss, MSE, high-frequency) + 6 peak features, all scored
//! - **4-scale pyramid** — 1×, 2×, 4×, 8× via box downscale (ssimulacra2 uses 6)
//! - **O(1)-per-pixel box blur** — single-pass with fused SIMD kernel
//! - **228 trained weights** — optimized on 344k synthetic pairs across 6 codecs
//! - **AVX2/AVX-512 SIMD** throughout via [archmage](https://crates.io/crates/archmage)
//!
//! See the `metric` module source for the full feature extraction math.

#![forbid(unsafe_code)]

/// Attribution-density steering map (research; `custom-profiles`).
#[cfg(feature = "custom-profiles")]
mod attribution;
mod blur;
mod color;
pub(crate) mod featcanon;
// The companion CORRUPTION HEAD (`docs/PLAN_CORRHEAD_SERVING_2026-09-06.md`):
// a gradient-boosted tree ensemble in its own `ZCTH` wire format, plus the ONE
// owner of the deploy composition `gate_score`. Gated and `#[doc(hidden)]`:
// nothing here is on `Zensim::compute`'s default path and the public API delta
// on a default build is ZERO. The public shape is PROPOSED in the plan doc and
// awaits the user's approval before it becomes a supported surface.
#[cfg(feature = "corruption-head")]
#[doc(hidden)]
pub mod corruption_head;
// **THE one owner of libc-independent arithmetic on the feature and score
// paths** (F19). `#[doc(hidden)] pub` since 2026-09-06 so `zensim-validate`'s
// bake-evaluation runtime can route through the SAME form the product runtime
// does instead of re-implementing `powf`/`exp` beside it — the F19 module's
// own exposure table named that fork "a BLOCKER on flipping
// `SHIPPED_REVISION`" and named this `pub` surface as the fix. Internal
// machinery, not product surface: the SUPPORTED API
// (`docs/public-api/zensim.txt`) is unchanged by it.
#[doc(hidden)]
pub mod det_math;
mod diffmap;
mod error;
mod fused;
/// THE v1 HF-energy ratio family — one owner, and the home of the F17 fix.
/// Unconditional: `contrast_inc` is a v1 basic slot, so it exists on a
/// `--no-default-features` build too.
pub(crate) mod hf_gain_form;
// The `globaff` family's strip moment sums (formula revision 6).
pub(crate) mod globaff;
// Research painting modes for the candidate attribution map.
mod iw_pool;
mod jnd;
#[cfg(feature = "custom-profiles")]
pub(crate) mod map_paint;
pub mod mapping;
mod metric;
// MLP runtime is internal — V0_4+ dispatch is exposed only through
// `ZensimProfile::PreviewV0_*` variants. Consumers wanting to bake or
// load custom MLP weights should depend on the `zenpredict` crate
// directly. zenpredict is MIT/Apache-2.0 (matching zensim), so the
// runtime is an unconditional dependency in 0.3.0+.
pub(crate) mod mlp;
mod pool;
pub mod profile;
// **Shared test fixtures, declared once at the crate root.** The geometry
// matrix and the pixel generators are OWNED by `tests/common/*` — the
// integration suites' owner — and are reached from in-crate tests through
// `#[path]` rather than retyped. A `#[path]` on a NESTED inline module cannot
// be used here: rustc resolves it under `src/<parent mod>/`, a directory that
// does not exist, so the `..` components fail to canonicalise. Declaring them
// at the root (where `src/` is real) is the form that resolves.
mod simd_ops;
pub mod source;
#[cfg(all(test, feature = "training", feature = "feature-regime-v2"))]
mod source_rows_tests;
// **THE one owner of the SCORE path's arithmetic** — the two mixing heads,
// the tanh output pin, the distance→score mapping, and the PCHIP output
// spline. `#[doc(hidden)] pub` for the same reason `det_math` is: the
// evaluation tooling in `zensim-validate` scores runtime-loaded bake bytes
// and MUST reach the product's own arithmetic rather than mirror it.
// Internal machinery, not product surface.
/// Internal wire-format views used by the Rust baking and diagnostics tools.
#[doc(hidden)]
pub mod bake_metadata;
mod sampling;
#[doc(hidden)]
pub mod score_math;
pub(crate) mod ssim_form;
mod streaming;
#[cfg(test)]
#[path = "../tests/common/generators.rs"]
pub(crate) mod test_generators;
#[cfg(test)]
#[path = "../tests/common/parity_cells.rs"]
pub(crate) mod test_parity_cells;
// HDR foundation: transfer functions + display model (code values → absolute
// luminance). Still foundation-only — the PU entry points take already-linear
// cd/m², so code-value decoding stays with the caller until a code-value
// entry lands. See `docs/HDR_PLAN.md`.
mod transfer;
// PU21 perceptually-uniform encoding — the HDR-path replacement for the
// cube-root nonlinearity, consumed by `Zensim::compute_pu_linear{,_planar}`.
mod pu21;
// EX-4 extended feature modules — XYB/LMS front-end stats + CVVDP-shape
// per-pair signals (DKL, Weber-contrast band ratios, mutual-masking
// residuals, Minkowski β=3 pool). Gated behind the `training` feature
// because they're only meaningful inside the feature-extract pipeline;
// the metric hot path never calls them.
#[cfg(feature = "training")]
pub mod cvvdp_features;
#[cfg(feature = "training")]
pub mod xyb_lms_features;

// Feature-set identifiers — the naming system that replaces "944"/"156+free"
// (docs/FEATURE_SET_IDS.md). NOT feature-gated: the grammar and the hash are
// pure data, and every consumer (bake_verdict, the trainer, the extractors,
// the board) needs them on a `--no-default-features` build too.
pub mod feature_set_id;

// The feature DEFINITION registry — what each slot IS (name, family, scale,
// channel, statistic, cost class, difference-form, direction, owning kernel,
// revision history) and the one owner of the layout arithmetic that expands
// signals into slot ids. NOT feature-gated, for the same reason
// `feature_set_id` is not: it is pure data, and the plan derivation and every
// consumer need it on a `--no-default-features` build too.
// docs/FEATURE_SYSTEM_DESIGN_2026-09-05.md
//
// `allow(dead_code)` while increment 1a stands alone: the registry is a DATA
// surface whose consumer is the plan derivation (increment 1b). Some of it is
// unread even by the gates — `Form::Undeclared` and `Direction::Undeclared`
// are deliberately unconstructed today (no signal's identity behaviour is
// unestablished yet) and must stay in the vocabulary so a future signal can
// say "not measured" instead of claiming a form nobody checked.
// NARROW this the moment the plan consumes the registry — a permanent blanket
// allow here would hide genuinely-unreachable definitions later.
#[allow(dead_code)]
pub(crate) mod feature_defs;

mod gmsbank_constants;

// The Rev5 runtime is mandatory for B; its research API remains opt-in. See
// feature_v2.rs's module doc and docs/FEATURE_V2_SPEC_2026-07-18.md.
#[cfg(feature = "feature-regime-v2")]
pub mod feature_v2;
#[cfg(not(feature = "feature-regime-v2"))]
#[allow(dead_code)] // private runtime; research API stays opt-in
mod feature_v2;

// Streaming strip-plane producer for the folded-720+append walk
// (docs/STREAMING_FOLDAPP_C0_DESIGN_2026-07-26.md).
pub(crate) mod feature_v2_stream;
// Pyramid rows produced by parallel chunk tasks for the strip-job pipeline.
#[cfg(feature = "threads")]
pub(crate) mod feature_v2_blocks;

// NEIGHSTEER (lane 2026-10-04): exact local refinement of the coarse-scale
// v2 pooled features for steering — crate-private, default-on for Rev5.
// Older revisions opt in via `ZENSIM_NEIGHBOUR_EXACT=1`. An explicit
// override enables replay only on exactly "1"; =0 or empty leaves the
// frozen density whole at every revision.
// benchmarks/neighsteer_2026-10-04.md
// The module is gated on `custom-profiles` too: its only caller is
// `BakeScorer::compute_attribution_input` (`all(custom-profiles,
// feature-regime-v2)`), so building it under v2 without custom-profiles
// (e.g. `candidate-profiles` alone) would be dead code end to end.
#[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
pub(crate) mod local_refine;

// The extraction PLAN: one derivation from "what a consumer needs" to "what
// the walk must run and how wide the answer is" — the pair (compute, layout)
// the runtime was missing. Gated with `feature_v2` because it resolves to a
// `ComputeSet`. docs/FEATURE_SYSTEM_DESIGN_2026-09-05.md
// `allow(dead_code)`: `Plan::derive` / `Plan::v1` / `Plan::covers` are the
// request-side half of the contract, exercised by the plan gates and the
// servability census; the runtime reaches a plan through `Plan::for_bake`.
// NARROW when a runtime caller takes an explicit slot request (phase 2's
// research engine is the first).
// **Layer 2 — the LAYOUT** (phase 4 of the feature-system refactor): a
// declared mapping from feature ids to positions in an emitted vector. Phase 1
// landed this as a bare `layout_width` and recorded WHY (every registered
// layout was the identity mapping, so a full map had no consumer that could
// distinguish it from the width). `Layout::dense` is that consumer.
//
// **UNGATED since 2026-09-06, and that is a correctness fix, not tidiness.**
// It was gated on `feature-regime-v2` while the bakes it serves are not: since
// `cb2f412d` the shipped `A`, `B`, `BHdr` and `D` are DENSE, and a build
// without that feature resolved their width disagreement POSITIONALLY —
// `prep_bake_input_f32`'s prefix branch — which is a silently-wrong score, the
// one outcome the dense contract forbids. MEASURED before the ungate, all 48
// A/B/BHdr/D cells wrong (`benchmarks/dense_serving_ungate_2026-09-06.md` §2):
// B — what `codec_target()` returns — 48.17 -> 13.49 on a blur; D 48.43 ->
// -213.15, a sign flip; A 94.11 -> 86.44; BHdr 95.12 -> 81.52. This module
// depends only on `feature_set_id` + `feature_defs` + `mlp::Model`, none of
// which is v2-gated, and every shipped dense bake's highest declared id is
// < 372 — a width the legacy v1 walk already emits — so serving a declaration
// correctly costs a default build +3,540 B of `.rlib` (+0.05 %) and a
// `--no-default-features` build +134,772 B (+4.29 %) — the latter because
// ungating this makes `feature_defs`'s registry reachable where nothing had
// referenced it. Same-parent A/B, `benchmarks/dense_serving_ungate_2026-09-06.md`.
pub(crate) mod feature_layout;

// E33 `fx1`: explicit model-input lists (direct ids and same-cell
// difference × reference-only products) declared in bake metadata. Not
// feature-gated for the same reason as `feature_layout`: it changes which
// values a declared bake is served.
pub(crate) mod derived_inputs;

// **The SERVABILITY census** — the roster of shipped profiles and the gate
// that every one of them serves a real pair. Always compiled, deliberately:
// it lived inside the `feature-regime-v2`-gated `feature_plan` until
// 2026-09-06, so the one instrument that asks "can this build serve what it
// ships?" was blind in exactly the builds that could not. The plan-shaped
// gates stay in `feature_plan`; only the profile-shaped ones moved.
//
// `allow(dead_code)` only in NON-test builds: every item here exists so the
// gates share ONE roster instead of each keeping a `#[cfg]`-dependent copy
// (the drift this module's own doc warns about). Scoped to `not(test)` so a
// genuinely-unused item still shows up where it can be seen.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod serving;

/// THE metadata key a bake declares its read set under — an ascending decimal
/// feature-id list, one per line.
///
/// Re-exported (doc-hidden, not part of the supported surface) because the key
/// has exactly ONE owner and the tool that WRITES it lives in a different
/// crate (`zensim-validate`'s `bake_dial_refit densify`) from the runtime that
/// READS it (`feature_layout::declared_layout`). A second string literal in
/// the writer is the duplication this repo's one-owner rule exists to prevent.
#[doc(hidden)]
pub use feature_layout::FEATURE_IDS_KEY as ZENTRAIN_FEATURE_IDS_KEY;

/// The feature ids a bake DECLARES, in ascending order — or `None` when it
/// declares the identity layout, which is every bake that shipped before
/// 2026-09-06.
///
/// **This is the consumer-side half of the dense contract, and it must exist
/// or dense bakes are silently mis-scored.** `zensim`'s own scoring path
/// gathers a declared bake's ids out of the walk's identity-laid-out vector
/// (`metric.rs`, the `declared_layout` branch). Every EVALUATION tool in
/// `zensim-validate` builds its own scratch and copies `row[..n_inputs]`
/// positionally — correct for an identity layout, and wrong by construction
/// for a dense one, where position 3 is whatever id the bake declared third.
/// Exported (doc-hidden, not the supported surface) so those tools gather
/// through the SAME owner rather than re-parsing the metadata key.
///
/// `None` means "copy positionally, exactly as before" — so a caller that
/// threads this through changes no byte for any existing bake.
#[doc(hidden)]
pub fn declared_feature_ids(model: &crate::mlp::Model) -> Option<Vec<u16>> {
    let layout = feature_layout::declared_layout(model);
    if layout.is_identity() {
        return None;
    }
    Some(
        (0..layout.width())
            .map(|pos| layout.slot_at(pos).unwrap_or(u16::MAX))
            .collect(),
    )
}

#[allow(dead_code)]
pub(crate) mod feature_plan;

// Env-gated (`ZENSIM_FOLD_TIMING`) per-phase wall/busy accounting for the
// streaming fold walk — diagnostic only, no public surface, and every hook is
// a resolved-`OnceLock` load when disabled.
// benchmarks/fold_mt_scaling_2026-08-31.md
pub(crate) mod fold_timing;
// A clock for the attribution path's timing counters that also runs in wasm.
#[cfg(feature = "custom-profiles")]
pub(crate) mod perf_clock;

// The fold-backed scoring engine (benchmarks/fold_engine_2026-08-31.md):
// gives the streaming fold a `score()` by attaching it under the SHARED
// scoring tail + bake forward, rather than duplicating either.
//
// `#[doc(hidden)]`: its one public item (`ScoringEngine`) is an internal
// engine selector for the parity gates and the perf comparison, not a product
// knob. `Buffered` is the default for every profile except `ZensimProfile::D`
// (`benchmarks/profile_d_notax_2026-09-01.md` — `D`'s whole reason to exist is
// speed, so `Zensim::new` opts it into `Fold` itself). The feature gates the
// public engine selector; private fold execution is also compiled in minimal
// builds because B requires its model-bound Rev5 extraction.
#[cfg(feature = "feature-regime-v2")]
#[doc(hidden)]
pub mod fold_engine;
#[cfg(not(feature = "feature-regime-v2"))]
#[allow(dead_code)] // private runtime; engine selector stays opt-in
mod fold_engine;

// **The RESEARCH engine** (phase 2 of the feature-system refactor,
// `docs/PLAN_FEATURE_SYSTEM_2026-09-05.md`): one plan-driven, full-width,
// deterministic entry that computes every registered signal and reports what
// each emitted value IS — id, name, family, statistic, cost, form, direction,
// owning kernel, resolved revision, live defect, and whether the plan
// populated the position or left the layout's structural zero.
//
// `#[doc(hidden)]`: research and extraction machinery, not product surface —
// the SUPPORTED API (`docs/public-api/zensim.txt`) is unchanged by it. Gated
// on `feature-regime-v2` because the walk it drives is; a
// `--no-default-features` build has neither.
#[cfg(feature = "feature-regime-v2")]
#[doc(hidden)]
pub mod research;

// --- Primary API ---
/// Cooperative-cancellation vocabulary, re-exported from the
/// [`enough`](https://docs.rs/enough) crate for use with
/// [`Zensim::with_stop`]. Construct a real cancellable token with the
/// sibling [`almost-enough`](https://docs.rs/almost-enough) crate's
/// `Stopper` (any `impl Stop` works).
pub use enough::{Stop, StopReason, Unstoppable};
pub use error::ZensimError;
#[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
pub use metric::SteeringSession;
pub use metric::{
    BakeScorer, FeatureView, Zensim, ZensimResult, dissimilarity_to_score, score_to_dissimilarity,
};

#[doc(hidden)]
pub use color::{bench_pu_xyb_dispatch, bench_pu_xyb_scalar};

/// Dev-only stage-level exports for per-stage SIMD benchmarking.
///
/// NOT part of the public API and NOT semver-covered — same status as the
/// `bench_pu_xyb_*` exports above, which set this precedent.
///
/// These exist because the whole-pipeline tier bench established that NEON is
/// worth only ~1.26x here while butteraugli — a comparable multi-scale
/// XYB+blur metric — gets 3.4x on the same host, and that the shortfall is
/// inside the image kernels rather than in scoring. Attributing it further
/// needs per-stage numbers, and on macOS the usual profilers (dtrace via
/// cargo-flamegraph) require SIP to be disabled. Exporting the stage entry
/// points lets `zensim-bench`'s `stage_isolation` bench get the same
/// attribution with an A/B instead of a profiler.
#[doc(hidden)]
pub mod __bench_stages {
    pub use crate::blur::{
        box_blur_1pass_into, box_spread_merge_f32, downscale_2x_into, fused_blur_h_ssim,
    };
    pub use crate::color::srgb_to_positive_xyb_planar_into;

    /// Exact native SDR linear input for peer audits. Supports u16 sRGB and
    /// f32 linear RGBA, declared primaries and the scorer's alpha composite.
    /// Refuses HDR, preserve-gamut mode, invalid geometry and nonfinite input.
    pub fn native_sdr_linear_rgb(
        source: &impl crate::ImageSource,
    ) -> Result<Vec<[f32; 3]>, crate::ZensimError> {
        crate::streaming::native_sdr_linear_rgb(source)
    }
    pub use crate::simd_ops::{abs_diff_sum, mul_into, sq_diff_sum, sq_sum_into};
}
pub use jnd::{JndEstimate, JndMap};
/// Classification API — requires `features = ["classification"]`.
///
/// Exposes `classify()`, error categorization, and per-pixel delta statistics
/// for regression testing workflows.
#[cfg(feature = "classification")]
pub use metric::{
    AlphaStratifiedStats, ClassifiedResult, DeltaStats, ErrorCategory, ErrorClassification,
    RoundingBias,
};
pub use profile::ZensimProfile;
pub use source::{
    AlphaMode, ColorPrimaries, GamutMapping, ImageSource, PixelFormat, RgbSlice, RgbaSlice,
    StridedBytes,
};

// `attribution` (the module) exists only under `custom-profiles`, and the
// fused folded-944 entry inside it additionally needs `feature-regime-v2` —
// gate the re-export on BOTH, else a `feature-regime-v2`-only build fails
// to compile (caught 2026-08-05 by the Profile C folded-944 test build).
#[cfg(feature = "custom-profiles")]
pub use attribution::{AttributionResult, AttributionSession};
#[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
pub use attribution::{Fused944Session, ScoredAttribution};
pub use diffmap::{DiffmapOptions, DiffmapResult, DiffmapWeighting};
pub use streaming::{PrecomputedReference, ZensimScratch};

/// Score a precomputed feature vector under a [`ZensimProfile`] —
/// the entry point alternative feature backends (e.g. `zensim-gpu`)
/// use to produce a bit-exact CPU-equivalent `0..100` score.
///
/// Runs the full bake forward pass (per-sample-α / hybrid head,
/// tanh-pin, PCHIP spline, per-codec affine, clamp / soft-clamp /
/// extrapolate disposition) — same dispatch the canonical
/// `Zensim::compute(...)` flow applies after feature extraction.
pub use metric::{
    score_features_fd_gradient_with_profile, score_features_with_profile,
    score_features_with_profile_and_codec,
};

/// Training/research API — requires `features = ["training"]`.
///
/// These items expose metric internals (blur kernel shape, scale count,
/// masking, weight vectors) that change metric behavior. Scores produced
/// with non-default `ZensimConfig` are **not comparable** to the default
/// trained weights or the 0-100 score scale.
///
/// The historical `score_from_features` (the panicking variant) was
/// removed in 0.3.0; use [`try_score_from_features`] instead.
#[cfg(feature = "training")]
pub use metric::{
    BlurKernel, CH_B, CH_X, CH_Y, DownscaleFilter, FEATURES_PER_CHANNEL_BASIC,
    FEATURES_PER_CHANNEL_EXTENDED, FEATURES_PER_CHANNEL_WITH_PEAKS, FEATURES_PER_SCALE, WEIGHTS,
    ZensimConfig, compute_zensim_with_config, compute_zensim_with_ref_and_config,
    precompute_reference_with_scales, try_score_from_features,
};

/// IW-weight estimator types — requires `features = ["training"]`.
///
/// Exposes the Wang & Li 2011 info-content weight estimators
/// ([`iw_pool::IwWeightKind`], [`iw_pool::IwWeightConfig`]) and the
/// `compute_iw_weights` entry point used for research experiments.
/// Including the steerable-pyramid GSM approximation spike added
/// 2026-05-15 — see `benchmarks/iw_pyramid_spike_methodology_2026-05-15.md`.
///
/// `WeightedPool` and `IwSsimFeatures` (the offline-experiment pool +
/// pooled-feature struct) are crate-internal as of 0.3.0 — the
/// streaming hot path emits the same numbers directly via
/// `streaming::process_strip_into_accum`, and no external workspace
/// caller used them.
#[cfg(feature = "training")]
pub use iw_pool::{IwWeightConfig, IwWeightKind, compute_iw_weights};

#[cfg(feature = "zenpixels")]
mod zenpixels_compat;
#[cfg(feature = "zenpixels")]
pub use error::UnsupportedFormat;
#[cfg(feature = "zenpixels")]
pub use zenpixels_compat::ZenpixelsSource;

/// Number of downscale levels. Each level halves resolution.
/// 4 scales covers 1x, 2x, 4x, 8x — sufficient for most perceptual effects.
pub(crate) const NUM_SCALES: usize = 4;

#[cfg(feature = "training")]
mod palette;
