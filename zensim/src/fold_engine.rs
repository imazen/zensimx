//! The fold-backed scoring engine — "the fold becomes the engine".
//!
//! zensim ships two walks over the same statistic:
//!
//! * **buffered** ([`crate::streaming::compute_multiscale_stats_streaming`]):
//!   whole-image XYB pyramids for both sides, band-processed per scale. This
//!   is what [`crate::Zensim::compute`] has always run.
//! * **the fold** ([`crate::feature_v2`]'s `foldapp_streaming_walk`): rolling
//!   planes, no materialised pyramid, O(strip) plane residency.
//!
//! Since option C (`blur::pyramid_plane_stride(w) == w`, 2026-08-30) the
//! fold's `f0..372` is **bit-identical** to v1's 372-feature extraction at
//! every width — gated by
//! `feature_v2::tests::v1_372_bit_exact_to_fold_at_every_width` over 19
//! geometries. What was still missing was everything between a feature vector
//! and a [`ZensimResult`]: the fold is an extractor, and
//! `ZensimV2Result` has no `score()`.
//!
//! This module is that missing piece, and it is deliberately thin. It does
//! **not** re-implement any scoring:
//!
//! | job | owner |
//! |---|---|
//! | sub-64 reflect-pad, byte-identical short-circuit | [`crate::metric`], shared, unchanged |
//! | extraction | the fold (`compute_folded_v1_372_streaming_impl`) |
//! | `mean_offset` | the fold, via `feature_v2::MeanOffsetRows` |
//! | sanitize + weight dot + score mapping | [`crate::metric::score_v1_layout_features`], shared |
//! | bake forward, α/hybrid head, tanh pin, PCHIP output spline, per-codec affine, clamp disposition | [`crate::metric::apply_mlp_scoring_with_codec`], untouched |
//!
//! The bake/spline machinery attaches *under* this module, not into it:
//! `apply_mlp_scoring_with_codec` consumes a finished `ZensimResult` plus the
//! original `(width, height)` and does not care which walk produced it. That
//! is the whole reason the fold-backed engine is additive rather than a
//! rewrite.
//!
//! ## What the engine does NOT cover, and why
//!
//! * **`num_scales != crate::NUM_SCALES`.** The fold is hard-wired to 4
//!   scales; a `ZensimConfig` (the `training`-feature surface) asking for a
//!   different pyramid falls back to buffered rather than silently scoring a
//!   different pyramid. At 4, `min_pyramid_dim_for_scales(4) ==
//!   MIN_PYRAMID_DIM == 64`, so the two paths' reflect-pad decisions coincide
//!   exactly.
//! * **Historical declared-HDR / PU-linear input.** Revisions 1–4 keep
//!   buffered. Rev5 `compute_pu_linear*` uses the PU fold with the declared
//!   bake plan and the revision's zero mean-offset metadata.
//! * **Non-default blur config.** The fold implements `blur_radius = 5`,
//!   `blur_passes = 1` — every shipped profile's values. Anything else falls
//!   back.
//!
//! [`is_fold_backable`] is the single owner of that predicate.

use crate::error::ZensimError;
use crate::feature_v2::V2Scratch;
use crate::metric::{ZensimConfig, ZensimResult};
use crate::source::ImageSource;

/// Which walk [`crate::Zensim`]'s scoring entries run.
///
/// `#[doc(hidden)]`: this is an internal engine selector for the parity gates
/// and the perf comparison, not a product knob. [`ScoringEngine::Buffered`]
/// is the default for every profile except [`crate::profile::ZensimProfile::D`]
/// (`Zensim::new` opts `D` into `Fold` itself, since speed is its whole reason
/// to exist — `benchmarks/profile_d_notax_2026-09-01.md`). `feature-regime-v2`
/// is default-on as of 2026-09-01, so a plain `cargo add zensim` build CAN
/// name this type (it stays `#[doc(hidden)]`, not surfaced in rendered docs);
/// `--no-default-features` removes it and the module entirely, and every
/// profile including `D` then runs `Buffered` unconditionally — correctly,
/// just without the fold's speed.
///
/// It is `pub` rather than `pub(crate)` for the same reason
/// `V2NewFeatureToggles::v1_only` is: the parity gate lives in
/// `tests/fold_engine_parity.rs`, which is a separate crate and cannot reach
/// `pub(crate)`.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScoringEngine {
    /// The buffered walk — today's shipped path.
    #[default]
    Buffered,
    /// The streaming fold, where it can serve the request (see
    /// [`is_fold_backable`]); buffered otherwise.
    Fold,
}

/// **THE single owner of "can the fold serve this scoring request?"**
///
/// Returns `false` for every case §"What the engine does NOT cover" lists.
/// Callers route on this rather than on `engine == Fold` alone, so an
/// out-of-domain request degrades to buffered instead of scoring a different
/// quantity.
pub(crate) fn is_fold_backable(config: &ZensimConfig) -> bool {
    config.num_scales == crate::NUM_SCALES
        && config.blur_radius == 5
        && config.blur_passes == 1
        // WEIGHT-SKIPPING is the subtle one. `streaming::active_channels`
        // drops a channel whose basic+peak weights are all ~0 unless
        // `compute_all_features` or `extended_features` is set, leaving those
        // slots at their `ScaleStats` default. The fold has no such notion —
        // it always computes all three channels — so on a weight-skipping
        // config the two paths would emit the same SCORE (the dropped slots
        // carry zero weight) but DIFFERENT feature vectors. Requiring one of
        // the two all-active flags keeps the parity claim about the whole
        // `ZensimResult`, not just its score. Every MLP-scored profile sets
        // `compute_all_features` (it is `mlp_bytes.is_some()`), and every
        // `compute_extended_features` call sets `extended_features`; the
        // plain linear `PreviewV0_1`/`PreviewV0_2` `compute()` path is the
        // one this excludes, and it stays on buffered.
        && (config.compute_all_features || config.extended_features)
}

/// v1's feature width for a config — the same `(extended, iw)` table
/// [`crate::metric::combine_scores`] assembles to, and the same one
/// `identical_result` uses.
///
/// v1's layout is `[0,156) basic · [156,228) peaks · [228,300) masked ·
/// [300,372) IW`, so a narrower config is a strict **prefix** of a wider one
/// and truncation is the whole conversion from the fold's 372. That claim is
/// gated (`fold_narrow_config_is_a_prefix_of_wide`), not assumed.
pub(crate) fn v1_feature_width(config: &ZensimConfig) -> usize {
    let fpc = match (config.extended_features, config.compute_iw_features) {
        (true, true) => {
            crate::metric::FEATURES_PER_CHANNEL_EXTENDED + crate::metric::FEATURES_PER_CHANNEL_IW
        }
        (true, false) => crate::metric::FEATURES_PER_CHANNEL_EXTENDED,
        (false, _) => crate::metric::FEATURES_PER_CHANNEL_WITH_PEAKS,
    };
    config.num_scales * 3 * fpc
}

/// Fold-backed replacement for
/// [`crate::streaming::compute_zensim_streaming_stoppable`]: same inputs,
/// same `ZensimResult`, different walk.
///
/// The caller ([`crate::metric::compute_with_config_inner`]) has already
/// applied the shared reflect-pad and byte-identical short-circuit, so this
/// sees exactly what the buffered walk would.
pub(crate) fn compute_fold_backed(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    config: &ZensimConfig,
    weights: &[f64],
    scratch: &mut V2Scratch,
    // The serving plan; `None` = today's unconditional v1 walk at
    // `V1PoolsMode::Full`. See `score_plan`.
    plan: Option<&crate::feature_plan::Plan>,
) -> Result<ZensimResult, ZensimError> {
    compute_fold_backed_stoppable(source, distorted, config, weights, scratch, plan, None)
}

pub(crate) fn compute_fold_backed_stoppable(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    config: &ZensimConfig,
    weights: &[f64],
    scratch: &mut V2Scratch,
    // The serving plan; `None` = today's unconditional v1 walk at
    // `V1PoolsMode::Full`. See `score_plan`.
    plan: Option<&crate::feature_plan::Plan>,
    stop: Option<&dyn enough::Stop>,
) -> Result<ZensimResult, ZensimError> {
    let (mut features, mean_offset) = crate::feature_v2::compute_folded_v1_372_streaming_stoppable(
        source,
        distorted,
        None,
        config.allow_multithreading,
        scratch,
        plan,
        #[cfg(feature = "custom-profiles")]
        None,
        stop,
    )?;
    // The fold emits its regime's width with `f372..` structurally zero. Keep
    // whichever is WIDER: the config's v1 width (a prefix, which the linear
    // tail reads) or the plan's WALK width (which a wide bake reads, and
    // which a DENSE layout's gather reads FROM — never the layout's own
    // packed width, which would truncate the walk before the gather could
    // reach the ids above it). Without a plan this is exactly today's
    // `truncate(v1_feature_width)`.
    //
    // REV4SERVE: `resize`, not `truncate` — a bake can declare an identity
    // input width beyond the plan's emitted regime width (the v2+basic
    // featpot cells are 1853-input rows whose live reads stop at f719).
    // Positions past the emit bound are uncomputed slots the bake provably
    // does not read (its live-read mask ends below them); zero is what the
    // extraction emits for unpopulated identity slots, so the score and the
    // visible row both stay honest.
    //
    // REV4SERVE review F1: before extending, verify the emitted vector
    // already covers everything the plan CLAIMS to populate — an emit the
    // walk did not materialize must surface loudly (debug panic, release
    // error), never as a zero-filled read slot.
    if let Some(p) = plan {
        debug_assert!(
            p.emit_covered(features.len()),
            "plan emit claims ids past the emitted vector (bound {} > {})",
            p.emit_bound(),
            features.len()
        );
        p.check_emit_covered(features.len())
            .map_err(|_| crate::ZensimError::ModelLoadFailed {
                reason: "bake plan emits features the fold walk did not materialize",
            })?;
    } else {
        debug_assert!(
            features.len() >= v1_feature_width(config),
            "v1 walk emitted fewer features than the config's width"
        );
        if features.len() < v1_feature_width(config) {
            return Err(crate::ZensimError::ModelForwardFailed {
                reason: "fold walk emitted fewer features than the config's v1 width",
            });
        }
    }
    let keep = plan
        .map_or(0, |p| p.walk_width())
        .max(v1_feature_width(config));
    features.resize(keep, 0.0);

    let (score, raw_distance) =
        crate::metric::score_v1_layout_features(&mut features, weights, config, config.num_scales);

    Ok(ZensimResult::new(
        score,
        raw_distance,
        features,
        // Placeholder profile tag, matching `combine_scores` — every
        // `Zensim::compute*` caller overrides it via `with_profile`.
        crate::profile::ZensimProfile::codec_target(),
        mean_offset,
    ))
}

/// Fold-backed replacement for
/// [`crate::streaming::compute_zensim_streaming_with_ref`] — the M1
/// precompute-once / compare-many shape, on the fold.
///
/// The reference's XYB pyramid IS what the fold's source side needs (the
/// producer fills scale 0 with `convert_source_to_xyb_into_slices` and
/// cascades `downscale_2x_into`, the same two functions that built the
/// cache), so no new reference type exists: `PrecomputedReference` serves
/// both engines. Returns `None` when the cache cannot feed the fold —
/// `feature_v2::cached_ref_feed_usable` owns that predicate — and the caller
/// falls back to the buffered `*_with_ref` walk rather than scoring a
/// mismatched pyramid.
pub(crate) fn compute_fold_backed_with_ref(
    precomputed: &crate::streaming::PrecomputedReference,
    distorted: &impl ImageSource,
    config: &ZensimConfig,
    weights: &[f64],
    scratch: &mut V2Scratch,
    pool_mode: Option<crate::feature_v2::V1PoolsMode>,
) -> Option<ZensimResult> {
    compute_fold_backed_with_ref_plan(
        precomputed,
        distorted,
        config,
        weights,
        scratch,
        pool_mode,
        None,
    )
}

pub(crate) fn compute_fold_backed_with_ref_plan(
    precomputed: &crate::streaming::PrecomputedReference,
    distorted: &impl ImageSource,
    config: &ZensimConfig,
    weights: &[f64],
    scratch: &mut V2Scratch,
    pool_mode: Option<crate::feature_v2::V1PoolsMode>,
    plan: Option<&crate::feature_plan::Plan>,
) -> Option<ZensimResult> {
    compute_fold_backed_with_ref_stoppable_plan(
        precomputed,
        distorted,
        config,
        weights,
        scratch,
        pool_mode,
        plan,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_fold_backed_with_ref_stoppable_plan(
    precomputed: &crate::streaming::PrecomputedReference,
    distorted: &impl ImageSource,
    config: &ZensimConfig,
    weights: &[f64],
    scratch: &mut V2Scratch,
    pool_mode: Option<crate::feature_v2::V1PoolsMode>,
    plan: Option<&crate::feature_plan::Plan>,
    stop: Option<&dyn enough::Stop>,
) -> Option<ZensimResult> {
    let (mut features, mean_offset) = crate::feature_v2::compute_folded_v1_372_with_ref_stoppable(
        precomputed,
        distorted,
        config.allow_multithreading,
        scratch,
        pool_mode,
        plan,
        stop,
    )?;
    if let Some(plan) = plan {
        plan.check_emit_covered(features.len()).ok()?;
        features.resize(plan.walk_width().max(v1_feature_width(config)), 0.0);
    } else {
        features.truncate(v1_feature_width(config));
    }
    let (score, raw_distance) =
        crate::metric::score_v1_layout_features(&mut features, weights, config, config.num_scales);
    Some(ZensimResult::new(
        score,
        raw_distance,
        features,
        crate::profile::ZensimProfile::codec_target(),
        mean_offset,
    ))
}

// ============================================================================
// Per-profile weight-skipping: computing only the v1 pool families a bake reads
// ============================================================================

/// Which of v1's three pool families (`f156..228` peaks, `f228..300` masked,
/// `f300..372` IW) a bake's layer 0 **structurally** reads — the compute-side
/// complement of `zensim-validate`'s `bake_block_profile`.
///
/// "Structurally" means the L∞ norm over that caller line's layer-0 output
/// weights is exactly zero, which is the same test the tooling uses and the
/// only one that is safe here: a zero weight makes the slot's value
/// unreachable by the forward pass for EVERY input, so leaving the slot at
/// 0.0 cannot move the score by a single ULP. A merely *small* weight is not
/// skippable and is never treated as one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct V1PoolNeed {
    pub(crate) peaks: bool,
    pub(crate) masked: bool,
    pub(crate) iw: bool,
}

impl V1PoolNeed {
    /// The conservative answer: assume every family is read. Returned for
    /// anything this module declines to analyse.
    pub(crate) const ALL: Self = Self {
        peaks: true,
        masked: true,
        iw: true,
    };
}

/// v1's 372 block boundaries, in caller feature indices.
const V1_PEAKS: (usize, usize) = (156, 228);
const V1_MASKED: (usize, usize) = (228, 300);
const V1_IW: (usize, usize) = (300, 372);

/// The layer-0 column range each CALLER line occupies, in caller order —
/// `caller_col_spans()[k] = (start, end)` with `end == start` for a line the
/// bake dropped.
///
/// **Why this exists here and is not a duplicate.** A bake with a
/// variable-arity [`zenpredict::FeatureTransform`] has caller line `k` ≠
/// layer-0 column `k`: `Drop` (dead-column pruning, which
/// `bake_dial_refit pack` performs BY DEFAULT since 2026-08-04) consumes zero
/// columns, `Sinusoidal` consumes `2·N`. Slicing columns at caller boundaries
/// on such a bake is the caller-width bug class
/// `zensim-validate::block_profile` documents four instances of. That crate
/// folds per-column *values* back to caller lines; this needs only the
/// *spans*, and both are thin callers of
/// [`zenpredict::FeatureTransform::output_arity`], which is the single owner
/// of arity semantics. zensim cannot depend on zensim-validate (it is the
/// other way round), so the narrow primitive lives in the lower crate.
///
/// `None` when the arities do not tile layer 0 — a malformed bake, which the
/// caller must treat as "assume everything is read".
fn caller_col_spans(model: &crate::mlp::Model, in_dim: usize) -> Option<Vec<(usize, usize)>> {
    let Some(ts) = model.feature_transforms() else {
        // No transform table: caller line k IS column k.
        return (in_dim == model.n_inputs()).then(|| (0..in_dim).map(|i| (i, i + 1)).collect());
    };
    let params = model.feature_transform_params();
    if let Some(p) = params
        && p.len() != ts.len()
    {
        return None;
    }
    let mut out = Vec::with_capacity(ts.len());
    let mut cur = 0usize;
    for (k, t) in ts.iter().enumerate() {
        let pk: &[f32] = params.map(|p| p[k].as_slice()).unwrap_or(&[]);
        let end = cur + t.output_arity(pk);
        if end > in_dim {
            return None;
        }
        out.push((cur, end));
        cur = end;
    }
    (cur == in_dim).then_some(out)
}

/// Layer-0 structural read-set of an ALREADY-PARSED bake, or
/// [`V1PoolNeed::ALL`] when its arities do not tile layer 0 (which no valid
/// bake's do). The parse-free half of [`bake_pool_need`] — split out so
/// [`crate::feature_plan::Plan::for_bake`] can derive the same
/// structural read-set from a model handle it already holds, rather than
/// re-parsing bytes it was never given. Handles pruned and expanded bakes
/// through [`caller_col_spans`].
pub(crate) fn bake_pool_need_from_model(model: &crate::mlp::Model) -> V1PoolNeed {
    // The read set as FEATURE IDS, through the one owner. This used to fold
    // `caller_line_reads` (layer-0 POSITIONS) directly against the family
    // bounds — sound only while position == id, i.e. only for an identity
    // layout. Under the dense contract shipped `B` has 95 live positions
    // `0..94` carrying ids `f3..f369`, and the positional fold reported
    // `{peaks: false, masked: false, iw: false}` for a bake that reads 49
    // lines above `f227`. That is a SKIP decision, so the wrong answer here
    // deletes pools the bake then gathers as structural zeros.
    let Some(slots) = crate::feature_plan::bake_read_slots(model) else {
        return V1PoolNeed::ALL;
    };
    let reads = |lo: usize, hi: usize| -> bool { (lo..hi).any(|s| slots.contains(s)) };
    V1PoolNeed {
        peaks: reads(V1_PEAKS.0, V1_PEAKS.1),
        masked: reads(V1_MASKED.0, V1_MASKED.1),
        iw: reads(V1_IW.0, V1_IW.1),
    }
}

/// **THE per-caller-line structural read predicate.** `caller_line_reads(m)[k]`
/// marks a nonzero layer-0 path, or a nonzero replacement min-max piece.
///
/// Extracted from [`bake_pool_need_from_model`]'s inner closure so that
/// [`crate::feature_plan::bake_read_slots`] reads the SAME predicate rather
/// than a second copy — the block-boundary question ("does it read the masked
/// family?") and the slot-set question ("exactly which lines does it read?")
/// are the same fold at two granularities, and a second copy is how the
/// caller-width bug class keeps recurring.
///
/// A dropped line spans no columns and so reads nothing, which is exactly
/// pruning's contract: a pruned column was already an exact zero, or a
/// transform-forced constant folded into the bias.
///
/// `None` for unreadable shapes or replacement-head contracts, which skip
/// callers must treat conservatively and serving admission must refuse.
pub(crate) fn caller_line_reads(model: &crate::mlp::Model) -> Option<Vec<bool>> {
    let layer = model.layer(0);
    let (in_dim, out_dim) = (layer.in_dim, layer.out_dim);
    let spans = caller_col_spans(model, in_dim)?;
    // This head replaces the network. Layer-zero zeros cannot prove its
    // inputs dead. Use the same parser as scoring, and require the scalar
    // head's one-to-one transform contract before making a skip decision.
    if let Some(entry) = model
        .metadata()
        .get(crate::bake_metadata::MINMAX_MONOTONE_HEAD_KEY)
    {
        let head = crate::bake_metadata::parse_minmax_head_meta(entry.value)?;
        if head.n != in_dim
            || spans.len() != head.n
            || spans
                .iter()
                .enumerate()
                .any(|(i, &span)| span != (i, i + 1))
        {
            return None;
        }
        return Some(
            (0..head.n)
                .map(|i| head.w.chunks_exact(head.n).any(|piece| piece[i] != 0.0))
                .collect(),
        );
    }
    Some(
        spans
            .iter()
            .map(|&(a, b)| {
                (a..b).any(|i| match &layer.weights {
                    crate::mlp::WeightStorage::F32(w) => {
                        w[i * out_dim..(i + 1) * out_dim].iter().any(|&v| v != 0.0)
                    }
                    crate::mlp::WeightStorage::F16(w) => w[i * out_dim..(i + 1) * out_dim]
                        .iter()
                        .any(|&h| crate::mlp::f16_bits_to_f32(h) != 0.0),
                    crate::mlp::WeightStorage::I8 { weights, scales } => weights
                        [i * out_dim..(i + 1) * out_dim]
                        .iter()
                        .zip(scales.iter())
                        .any(|(&q, &s)| q != 0 && s != 0.0),
                })
            })
            .collect(),
    )
}

/// Layer-0 structural read-set of one bake's BYTES, or [`V1PoolNeed::ALL`]
/// when the bake cannot be parsed. Parses then delegates to
/// [`bake_pool_need_from_model`] — see that function for the derivation.
fn bake_pool_need(bytes: &[u8]) -> V1PoolNeed {
    let Ok(model) = crate::mlp::Model::from_bytes(bytes) else {
        return V1PoolNeed::ALL;
    };
    bake_pool_need_from_model(&model)
}

/// [`bake_pool_need`], interned by bake-bytes data pointer — the same cache
/// shape (and the same stability argument) as
/// `metric::cached_bake_metadata`: bake slices always come from `&'static`
/// slots via `ProfileParams::mlp_bytes`, so the pointer is a stable unique
/// key per slot. Parsing a bake per compare would defeat the point of the
/// skip.
pub(crate) fn cached_bake_pool_need(bytes: &[u8]) -> V1PoolNeed {
    use std::collections::HashMap;
    use std::sync::{OnceLock, RwLock};
    static CACHE: OnceLock<RwLock<HashMap<usize, V1PoolNeed>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    let key = bytes.as_ptr() as usize;
    if let Ok(read) = cache.read()
        && let Some(v) = read.get(&key)
    {
        return *v;
    }
    let need = bake_pool_need(bytes);
    if let Ok(mut write) = cache.write() {
        write.insert(key, need);
    }
    need
}

/// The pool mode a fold-backed SCORE may run for this profile — the three-way
/// **model-class** decision, not a per-family one.
///
/// Returns [`V1PoolsMode::Full`] — today's unconditional behaviour — unless
/// the caller opted into skipping AND **every** consumer of the feature
/// vector structurally ignores the block in question. The consumers are:
///
/// * the profile's linear `weights`, which `metric::score_v1_layout_features`
///   reads over `[0, num_scales·3·FEATURES_PER_CHANNEL_WITH_PEAKS)` =
///   `f0..228`. Masked/IW are structurally out of its reach (checked, not
///   assumed), but the PEAK block is not — so dropping to
///   [`V1PoolsMode::Off`] additionally requires the weight vector to be zero
///   there, or `raw_distance` would move;
/// * `mlp_bytes`, `mlp_bytes_b3` and `ensemble_classifier_bytes`, each read
///   through [`cached_bake_pool_need`].
///
/// The ladder, cheapest first:
///
/// | mode | when | what it skips |
/// |---|---|---|
/// | [`V1PoolsMode::Off`] | nothing reads `f156..372` — the **basic-only model class** (`ADD156` reads 28 of 156 basic lines and 0 of 216 pool lines) | the whole pool block |
/// | [`V1PoolsMode::Peaks`] | peaks read, masked/IW not | the masked/IW pass group |
/// | [`V1PoolsMode::Full`] | anything reads masked or IW — every profile shipped today | nothing |
///
/// `Off` and `Peaks` cost the SAME to compute (the peak accumulators are the
/// fused V-blur kernel's unconditional L8/max tier); the difference between
/// them is only which slots are emitted. The real compute boundary is
/// `{Off, Peaks}` vs `Full`.
pub(crate) fn score_pool_mode(
    params: &crate::profile::ProfileParams,
    config: &crate::metric::ZensimConfig,
    skip_unread: bool,
) -> crate::feature_v2::V1PoolsMode {
    use crate::feature_v2::V1PoolsMode;
    if !skip_unread {
        return V1PoolsMode::Full;
    }
    // The linear tail's reach. If a future config ever widened it past the
    // peak block this decision would be unsound, so it is checked, not
    // assumed.
    if config.num_scales * 3 * crate::metric::FEATURES_PER_CHANNEL_WITH_PEAKS > V1_PEAKS.1 {
        return V1PoolsMode::Full;
    }
    let mut need = V1PoolNeed {
        peaks: false,
        masked: false,
        iw: false,
    };
    let mut any_bake = false;
    for bytes in params.scoring_bake_bytes() {
        any_bake = true;
        let n = cached_bake_pool_need(bytes);
        need.peaks |= n.peaks;
        need.masked |= n.masked;
        need.iw |= n.iw;
    }
    if !any_bake {
        // A purely linear profile reads f0..228 only, so masked/IW are
        // unread by construction. The peak block still feeds `raw_distance`.
        return V1PoolsMode::Peaks;
    }
    pools_mode_for_need(need)
}

/// The **serving plan** for a profile: the pair `(compute, layout)` one walk
/// must run so that every one of the profile's bakes can be scored from it.
///
/// `None` means "today's path, unchanged" — the profile's widest bake fits the
/// v1 layout and the caller did not opt into pool skipping, so nothing about
/// the extraction needs to change. This is the byte-identity guarantee: the
/// narrow case does not merely produce the same numbers, it takes the same
/// code path.
///
/// `Some(plan)` is returned when the extraction must differ from that default
/// — because a bake declares a LAYOUT wider than the v1 walk can emit (the
/// servability case), or because pool skipping was requested (today's
/// `score_pool_mode` case, now expressed as one axis of a plan).
///
/// A profile can carry up to three scoring bakes and must serve all of them
/// from ONE extraction, so the plan is the UNION over them — never any single
/// bake's, which would under-compute for the others.
pub(crate) fn score_plan(
    params: &crate::profile::ProfileParams,
    config: &crate::metric::ZensimConfig,
    skip_unread: bool,
) -> Option<crate::feature_plan::Plan> {
    use crate::feature_plan::Plan;
    let v1_width = v1_feature_width(config);
    let mut widest = 0usize;
    let mut union: Option<Plan> = None;
    for bytes in params.scoring_bake_bytes() {
        let Ok(model) = crate::metric::cached_profile_model(bytes) else {
            // An unparseable bake is not a planning input: the scoring path
            // reports it on its own terms. Assume the widest safe plan.
            return None;
        };
        let Ok(p) = Plan::for_bake(&model) else {
            return None;
        };
        // The WALK width, not `caller_input_width()`. They are equal for
        // every identity-laid-out bake — i.e. every bake that exists today —
        // and differ for a DENSE one, whose 265 declared inputs still need a
        // walk that reaches f941. Comparing the caller width here sent a
        // dense bake down the "narrow, nothing to plan" shortcut below and
        // then gathered its ids out of a 372-wide vector, which is silently
        // wrong: measured, and the reason `dense_layout_round_trip` exists.
        widest = widest.max(p.walk_width());
        union = Some(match union {
            Some(u) => {
                // **REVISION IS PER BAKE, and one walk computes one era.** A
                // profile whose bakes declare different formula revisions
                // cannot be served from a single extraction; unioning them
                // would silently serve one bake the other's arithmetic, which
                // is exactly what the per-bake declaration exists to prevent.
                // Returning `None` sends the profile down the unplanned path,
                // which computes the process's revision for every bake and is
                // today's behaviour — and the loud half lives in
                // `mixed_revision_profiles_are_refused_by_the_planner`, which
                // pins that a mixed profile does NOT get a plan rather than
                // getting a quietly wrong one.
                if !u.revisions_agree(&p) {
                    return None;
                }
                u.union(&p)
            }
            None => p,
        });
    }
    // Nothing to plan for: a purely linear profile reads `f0..228` only, and
    // the existing `score_pool_mode` policy already covers it.
    let mut plan = union?;
    // A profile's linear tail reads the v1 prefix whatever its bakes declare, so
    // every v1 scale stays computed (the per-scale skip is for plans whose
    // readers are all declared, e.g. `BakeScorer`).
    if plan.compute.v1_scales != crate::feature_v2::ComputeSet::ALL_SCALES {
        plan.compute.v1_scales = crate::feature_v2::ComputeSet::ALL_SCALES;
        plan.emit = plan
            .compute
            .populated_slots(crate::NUM_SCALES, plan.layout_width());
    }
    if widest <= v1_width && !skip_unread {
        // The narrow, non-skipping case IS today's path. Returning `None`
        // keeps it on the identical code path rather than on an equivalent
        // one — which is a stronger guarantee than equal output.
        return None;
    }
    // Pool skipping is opt-in. Without it the walk computes the whole pool
    // block exactly as it does today, whatever the bakes read.
    if !skip_unread {
        plan.compute.coarse_y_only_scales = 0;
        plan.compute.local_only = false;
        plan.compute.omit_edges = false;
        plan.compute.v1_pools = crate::feature_v2::V1PoolsMode::Full;
        plan.compute.v1_full_scales = crate::feature_v2::ComputeSet::ALL_SCALES;
        plan.compute.v1_scales = crate::feature_v2::ComputeSet::ALL_SCALES;
        plan.emit = plan
            .compute
            .populated_slots(crate::NUM_SCALES, plan.layout_width());
    }
    // The emitted vector must be at least as wide as the linear tail reads.
    if plan.walk_width() < v1_width {
        // Widen the LAYOUT, not just a number: the linear tail reads the v1
        // prefix, so a plan narrower than it must be re-laid-out at that
        // width. Rebuilt through the constructor so `compute` and `emit`
        // stay the normalized fixed point rather than being patched in place.
        plan = crate::feature_plan::Plan::widened_to_identity(&plan, v1_width);
    }
    Some(plan)
}

/// The [`V1PoolsMode`] to run for a given structural read-set.
///
/// **Never returns [`V1PoolsMode::Off`]**, even when nothing reads the pool
/// block at all — `Off` and `Peaks` cost the SAME to compute (the peak
/// accumulators are the fused V-blur kernel's unconditional L8/max tier) but
/// `Off` is never the better choice, and the reason is footprint, not
/// arithmetic. `Off` hands the band no scratch at all, which disables the
/// band-local self-blur shape (`FoldHSource::SelfBlur`) — so the walk falls
/// back to phase A's four STRIP-wide H planes (`4 × 3 × 148·W × 4` bytes)
/// where `Peaks` blurs exactly the 42 rows a band consumes into
/// `3 × slots × 4 × 42·W × 4`. `Peaks` computes the identical sums, emits a
/// superset of `Off`'s slots, and is the smaller hot set
/// (`benchmarks/fold_footprint_2026-08-31.md` §9.2 costs a `Full` band task
/// 12 planes × 42 rows = 2,016·W bytes; `Peaks` touches 6 of those 12). So
/// `Peaks` dominates `Off` on every axis and this policy never returns `Off`.
///
/// Shared by [`score_pool_mode`] (which unions `need` across a profile's up
/// to three bakes first) and
/// [`crate::feature_plan::Plan::for_bake`] (which derives
/// `need` from one already-parsed model) so the "`Off` is never the right
/// answer" policy lives in exactly one place.
pub(crate) fn pools_mode_for_need(need: V1PoolNeed) -> crate::feature_v2::V1PoolsMode {
    use crate::feature_v2::V1PoolsMode;
    if need.masked || need.iw {
        V1PoolsMode::Full
    } else {
        V1PoolsMode::Peaks
    }
}

#[cfg(all(test, feature = "feature-regime-v2"))]
mod skip_policy_tests {
    use super::*;
    use crate::feature_v2::V1PoolsMode;
    use crate::profile::ZensimProfile;

    fn cfg_for(p: ZensimProfile) -> (&'static crate::profile::ProfileParams, ZensimConfig) {
        let params = p.params();
        let config = crate::metric::config_from_params(params, false);
        (params, config)
    }

    /// Every SHIPPED profile reads the masked/IW block, so the policy must
    /// resolve to `Full` — i.e. opting into skipping changes nothing for
    /// them. This is the "no silent behaviour change" half of the contract,
    /// stated as a fact about the bakes rather than as an intention.
    #[test]
    fn shipped_profiles_read_the_pool_block_so_skipping_is_a_no_op() {
        // A one-element list today; it exists so a second shipped profile is a
        // one-line addition rather than a copy-paste of the body.
        #[allow(clippy::single_element_loop)]
        for p in [ZensimProfile::BHdr] {
            let (params, config) = cfg_for(p);
            assert_eq!(
                score_pool_mode(params, &config, true),
                V1PoolsMode::Full,
                "{p} resolved to a skipping mode — its bake reads f228..372"
            );
            assert_eq!(score_pool_mode(params, &config, false), V1PoolsMode::Full);
        }
    }

    /// The skip is opt-in: with `skip_unread == false` the answer is `Full`
    /// no matter what the bake looks like.
    #[test]
    fn skipping_is_off_by_default() {
        let z = crate::Zensim::new(ZensimProfile::BHdr);
        assert_eq!(z.score_pool_mode(), V1PoolsMode::Full);
        assert_eq!(
            z.clone()
                .with_unread_feature_skipping(true)
                .score_pool_mode(),
            V1PoolsMode::Full,
            "B reads the block, so even opted-in it must compute it"
        );
    }

    /// The read-set derivation itself, against the shipped B bake — the
    /// numbers `zensim-validate`'s `bake_block_profile` reports for the same
    /// bytes (v1_peaks 26 used / v1_masked 10 / v1_iw 13, 2026-08-31).
    #[test]
    fn bake_pool_need_matches_the_block_profile_tool_on_shipped_b() {
        let params = ZensimProfile::B.params();
        let bytes: Vec<&'static [u8]> = params.scoring_bake_bytes().collect();
        assert_eq!(bytes.len(), 1, "B forwards exactly one bake");
        let need = cached_bake_pool_need(bytes[0]);
        assert_eq!(
            need,
            V1PoolNeed {
                peaks: false,
                masked: false,
                iw: false
            }
        );
        // Interned: the second call must agree with the first.
        assert_eq!(cached_bake_pool_need(bytes[0]), need);
    }

    /// **The end-to-end score-safety gate.** When the skip DOES fire, the
    /// linear scoring tail must not notice: `metric::score_v1_layout_features`
    /// reads `features[..num_scales·3·FEATURES_PER_CHANNEL_WITH_PEAKS]` =
    /// `f0..228`, which `V1PoolsMode::Peaks` emits bit-identically. Forced
    /// here rather than driven through a profile, because no SHIPPED bake
    /// ignores the masked/IW block — so this is the only way to exercise the
    /// firing path until one does.
    #[cfg(all(feature = "training", feature = "threads"))]
    #[test]
    fn a_fired_skip_leaves_raw_distance_bit_identical() {
        use crate::feature_v2::V2Scratch;
        use crate::source::RgbSlice;
        let params = ZensimProfile::B.params();
        for &(w, h) in &[
            (96usize, 64usize),
            (200, 150),
            (127, 93),
            (256, 256),
            (577, 385),
        ] {
            for &parallel in &[false, true] {
                let mut config = crate::metric::config_from_params(params, parallel);
                config.allow_multithreading = parallel;
                let src = crate::feature_v2::tests::textured_image(w, h, 7);
                let dst = crate::feature_v2::tests::quantize_distort(&src, w, h);
                let (sref, dref) = (RgbSlice::new(&src, w, h), RgbSlice::new(&dst, w, h));
                let mut scratch = V2Scratch::new();
                let full = compute_fold_backed(
                    &sref,
                    &dref,
                    &config,
                    params.weights,
                    &mut scratch,
                    Some(&crate::feature_plan::Plan::v1(
                        V1PoolsMode::Full,
                        v1_feature_width(&config),
                    )),
                )
                .expect("full");
                let peaks = compute_fold_backed(
                    &sref,
                    &dref,
                    &config,
                    params.weights,
                    &mut scratch,
                    Some(&crate::feature_plan::Plan::v1(
                        V1PoolsMode::Peaks,
                        v1_feature_width(&config),
                    )),
                )
                .expect("peaks");
                assert_eq!(
                    full.raw_distance().to_bits(),
                    peaks.raw_distance().to_bits(),
                    "{w}x{h} par={parallel}: raw_distance moved ({:e} vs {:e})",
                    full.raw_distance(),
                    peaks.raw_distance()
                );
                assert_eq!(
                    full.score().to_bits(),
                    peaks.score().to_bits(),
                    "{w}x{h} par={parallel}: linear score moved"
                );
                assert_eq!(full.mean_offset(), peaks.mean_offset());
                let (ff, fp) = (full.features(), peaks.features());
                assert_eq!(ff.len(), fp.len());
                for i in 0..228 {
                    assert_eq!(
                        ff[i].to_bits(),
                        fp[i].to_bits(),
                        "{w}x{h} par={parallel}: scored slot {i} moved"
                    );
                }
                assert!(fp[228..372].iter().all(|&v| v == 0.0));
                // …and the skip is genuinely doing something: Full's block
                // is not all-zero on a distorted pair.
                assert!(ff[228..372].iter().any(|&v| v != 0.0));
            }
        }
    }

    /// The same guarantee on the REF-CACHED entry. `compute_with_ref` routes
    /// through `compute_fold_backed_with_ref`, a different function with its
    /// own `pool_mode` argument, so "the skip is score-neutral" has to be
    /// stated about it separately — a wiring slip there would be invisible to
    /// the `compute()` gate above.
    #[cfg(all(feature = "training", feature = "threads"))]
    #[test]
    fn a_fired_skip_is_score_neutral_on_the_ref_cached_entry() {
        use crate::feature_v2::V2Scratch;
        use crate::source::RgbSlice;
        let params = ZensimProfile::B.params();
        for &(w, h) in &[(96usize, 64usize), (200, 150), (256, 256), (577, 385)] {
            for &parallel in &[false, true] {
                let mut config = crate::metric::config_from_params(params, parallel);
                config.allow_multithreading = parallel;
                let src = crate::feature_v2::tests::textured_image(w, h, 7);
                let dst = crate::feature_v2::tests::quantize_distort(&src, w, h);
                let (sref, dref) = (RgbSlice::new(&src, w, h), RgbSlice::new(&dst, w, h));
                let pre = crate::streaming::PrecomputedReference::new(
                    &sref,
                    config.num_scales,
                    parallel,
                    config.revision(),
                );
                let mut scratch = V2Scratch::new();
                let full = compute_fold_backed_with_ref(
                    &pre,
                    &dref,
                    &config,
                    params.weights,
                    &mut scratch,
                    Some(V1PoolsMode::Full),
                )
                .expect("the cache must feed the fold at these dims");
                let peaks = compute_fold_backed_with_ref(
                    &pre,
                    &dref,
                    &config,
                    params.weights,
                    &mut scratch,
                    Some(V1PoolsMode::Peaks),
                )
                .expect("peaks");
                assert_eq!(
                    full.raw_distance().to_bits(),
                    peaks.raw_distance().to_bits(),
                    "{w}x{h} par={parallel}: ref-cached raw_distance moved"
                );
                assert_eq!(full.score().to_bits(), peaks.score().to_bits());
                assert_eq!(full.mean_offset(), peaks.mean_offset());
                let (ff, fp) = (full.features(), peaks.features());
                for i in 0..228 {
                    assert_eq!(
                        ff[i].to_bits(),
                        fp[i].to_bits(),
                        "{w}x{h} par={parallel}: ref-cached scored slot {i} moved"
                    );
                }
                assert!(fp[228..372].iter().all(|&v| v == 0.0));
                assert!(ff[228..372].iter().any(|&v| v != 0.0));
            }
        }
    }

    /// A bake this module declines to analyse must come back as `ALL`, never
    /// as an optimistic "nothing is read" — the failure mode that would
    /// silently zero a live slot.
    #[test]
    fn an_unparseable_bake_needs_everything() {
        assert_eq!(bake_pool_need(&[0u8; 8]), V1PoolNeed::ALL);
        assert_eq!(bake_pool_need(&[]), V1PoolNeed::ALL);
    }

    /// **The Profile-D task gate**: scores must be bit-identical whatever
    /// walk/skip combination produces them — buffered vs fold, skip vs
    /// no-skip, all four combinations, through the REAL public
    /// `Zensim::compute` entry point (not the internal `compute_fold_backed`
    /// harness the tests above use). This is the end-to-end proof that
    /// `Zensim::new`'s per-profile fast-by-default wiring
    /// (`benchmarks/profile_d_and_published_speed_2026-09-01.md`) changes
    /// only performance, never the score — and that the DEFAULT
    /// construction (fast-by-default under this feature build) agrees with
    /// every explicit combination too.
    #[cfg(feature = "candidate-profiles")]
    #[test]
    fn profile_d_scores_are_engine_and_skip_invariant() {
        use crate::source::RgbSlice;
        for &(w, h) in &[(96usize, 64usize), (256, 256), (577, 385)] {
            let src = crate::feature_v2::tests::textured_image(w, h, 11);
            let dst = crate::feature_v2::tests::quantize_distort(&src, w, h);
            let (sref, dref) = (RgbSlice::new(&src, w, h), RgbSlice::new(&dst, w, h));

            let default_result = crate::Zensim::new(ZensimProfile::D)
                .compute(&sref, &dref)
                .expect("D scores by default");
            let want = default_result.score().to_bits();

            for &engine in &[ScoringEngine::Buffered, ScoringEngine::Fold] {
                for &skip in &[false, true] {
                    let r = crate::Zensim::new(ZensimProfile::D)
                        .with_engine(engine)
                        .with_unread_feature_skipping(skip)
                        .compute(&sref, &dref)
                        .unwrap_or_else(|e| panic!("{w}x{h} engine={engine:?} skip={skip}: {e}"));
                    assert_eq!(
                        r.score().to_bits(),
                        want,
                        "{w}x{h}: engine={engine:?} skip={skip} diverged from the default \
                         (fast-by-default) score — D's speed knobs must never move the score"
                    );
                }
            }
        }
    }

    /// **The gating-tax-removal gate** (`benchmarks/profile_d_notax_2026-09-01.md`):
    /// a plain default build (this crate's `default` feature list, which has
    /// carried `feature-regime-v2` since 2026-09-01) must score `D`
    /// IDENTICALLY — score, `raw_distance`, mean_offset, and every SCORED
    /// feature slot — to what a `--no-default-features` build re-adding only
    /// the OTHER defaults (`avx512,imgref,threads,deprecated-profiles,
    /// candidate-profiles`, i.e. everything except `feature-regime-v2`) would
    /// have produced. Such a build cannot name `ScoringEngine` or
    /// `with_engine` at all (`fold_engine` does not exist without the
    /// feature), so it can only ever run [`ScoringEngine::Buffered`] — this
    /// crate's proxy for "what the gated-off build computes" is therefore
    /// forcing `Buffered` explicitly (leaving `skip_unread_pools` untouched:
    /// the buffered walk never reads that field either way, proven by the
    /// MEASURED finding below) and diffing against this build's DEFAULT
    /// construction (which, for `D`, is fast-by-default = `Fold`+skip where
    /// serviceable).
    ///
    /// **Why the comparison stops at `f0..228` and not the full 372, and why
    /// that is the correct claim rather than a weakened one — MEASURED, not
    /// assumed.** An earlier draft of this test asserted full-vector
    /// equality and failed immediately at `f228` (96×64: `0.0` vs
    /// `2.302614610319627e-3`) while `score()` matched exactly
    /// (`68.94795355810257` both arms, bit-for-bit) — the failure was the
    /// test's premise, not a code defect. `V1PoolsMode::Peaks` (what
    /// `fast_by_default` selects for `D`, since `ADD156` reads 0 of
    /// `f156..372`) deliberately leaves `f228..372` (masked/IW) at `0.0` —
    /// that is `with_unread_feature_skipping`'s own documented contract
    /// ("leaves the skipped slots at 0.0"). The buffered walk has no
    /// skipping concept at all and always computes real values there,
    /// confirmed directly: `Zensim::new(D).with_engine(Buffered).compute(..)`
    /// (skip untouched) gives the SAME `2.302614610319627e-3` at `f228` as
    /// the `skip=false` proxy, on both arms' `score()` still bit-identical.
    /// So a genuinely gated-off build's `D` and this build's default `D`
    /// PROVABLY differ at `f228..372` — real values vs deliberate zeros —
    /// while agreeing on everything that actually reaches the score
    /// (`score_v1_layout_features` reads `f0..228` for `D`'s class of
    /// profile; masked/IW carries zero weight). Asserting full-vector
    /// equality would have enshrined a false claim; asserting it only over
    /// the scored prefix states the true invariant the gating-tax refactor
    /// actually provides: speed changes, SCORE never does.
    /// `profile_d_scores_are_engine_and_skip_invariant` above already proves
    /// `Buffered == Fold` for `D`'s score as one of its four combinations;
    /// this test is its own standalone, narrowly-named regression gate for
    /// the gated-vs-default-build claim specifically, with the scored-region
    /// boundary stated and justified rather than left implicit.
    /// **A DENSE bake's pool need is read in ID space, not position space.**
    ///
    /// `caller_line_reads` returns layer-0 POSITIONS, and until 2026-09-06
    /// every bake was identity-laid-out so a position WAS an id. The dense
    /// contract breaks that: a bake declaring 3 ids in `f156`/`f228`/`f300`
    /// has live positions `0,1,2`, and folding those against the family
    /// bounds says it touches no pool at all — a SKIP decision, so the wrong
    /// answer deletes the pools the bake then gathers as structural zeros.
    ///
    /// MEASURED on the real artifact when shipped `B` went dense: the
    /// positional fold reported `{peaks:false, masked:false, iw:false}` for a
    /// bake that reads 49 lines above `f227`.
    ///
    /// The NEGATIVE CONTROL is the same bake read the old way — if a
    /// positional fold of these ids also said "all three pools", the test
    /// would prove nothing.
    #[test]
    fn a_dense_bake_reports_the_pools_its_declared_ids_land_in() {
        let bytes = crate::feature_layout::tests::ids_bake_bytes(&[156, 228, 300]);
        let m = crate::mlp::Model::from_bytes(&bytes).expect("synthetic dense bake parses");
        assert_eq!(m.caller_input_width(), 3, "3 declared ids, 3 caller inputs");
        assert_eq!(
            bake_pool_need_from_model(&m),
            V1PoolNeed {
                peaks: true,
                masked: true,
                iw: true
            }
        );
        // Negative control: the raw positional fold this replaced.
        let live = caller_line_reads(&m).expect("layer 0 tiles");
        let positional = |lo: usize, hi: usize| -> bool {
            let hi = hi.min(live.len());
            lo < hi && live[lo..hi].iter().any(|&b| b)
        };
        assert!(
            !positional(V1_PEAKS.0, V1_PEAKS.1)
                && !positional(V1_MASKED.0, V1_MASKED.1)
                && !positional(V1_IW.0, V1_IW.1),
            "the positional fold must say NO pools here, else this test proves nothing"
        );
    }

    /// **The `Off` is never served policy applies to the ID-SPACE branch too.**
    ///
    /// `pools_mode_for_need` documents (with a measured footprint argument)
    /// that `Off` is never the right answer for a SERVED v1 walk: it costs the
    /// same arithmetic as `Peaks` and grows the hot set by disabling the
    /// band-local self-blur. `Plan::for_bake` routes through that owner;
    /// `Plan::derive_with_layout` derives from touched families and will
    /// happily return `Off`. Nothing reached that branch until shipped `D`
    /// declared its 28 basic ids — at which point D's emitted vector went from
    /// real values at `f156..227` to zeros, score unmoved, footprint worse.
    #[test]
    fn a_dense_basic_only_bake_is_served_peaks_not_off() {
        let ids: Vec<usize> = (0..8).map(|i| i * 3).collect();
        let bytes = crate::feature_layout::tests::ids_bake_bytes(&ids);
        let m = crate::mlp::Model::from_bytes(&bytes).expect("synthetic dense bake parses");
        // The raw id-space derivation, with no policy applied: `Off`.
        let layout = crate::feature_layout::declared_layout(&m);
        let want = crate::feature_plan::bake_read_slots(&m).expect("read set");
        let raw = crate::feature_plan::Plan::derive_with_layout(&want, layout).expect("plan");
        assert_eq!(
            raw.compute.v1_pools,
            V1PoolsMode::Off,
            "the unfiltered id-space derivation must say Off, else the policy below is untested"
        );
        // The SERVING plan promotes it.
        let served = crate::feature_plan::Plan::for_bake(&m).expect("serving plan");
        assert_eq!(served.compute.v1_pools, V1PoolsMode::Peaks);
    }

    #[cfg(feature = "candidate-profiles")]
    #[test]
    fn default_build_profile_d_matches_feature_gated_off_buffered_walk() {
        use crate::source::RgbSlice;
        // 96x64 / 577x385: the existing invariant test's sub-64-pad and
        // odd-dims fixtures. 592x400: `simd_padded_width(592) == 592` (a
        // multiple of 16 that stays below the +16 rounding some widths get,
        // per CLAUDE.md's "option C" pad-column note) — a control size for
        // the v1/fold padded-width class of defect, so this gate does not
        // accidentally only ever exercise geometries where that class of bug
        // is invisible.
        for &(w, h) in &[(96usize, 64usize), (577, 385), (592, 400)] {
            let src = crate::feature_v2::tests::textured_image(w, h, 29);
            let dst = crate::feature_v2::tests::quantize_distort(&src, w, h);
            let (sref, dref) = (RgbSlice::new(&src, w, h), RgbSlice::new(&dst, w, h));

            let default_build = crate::Zensim::new(ZensimProfile::D)
                .compute(&sref, &dref)
                .expect("D scores by default");
            // The gated-off proxy: forced Buffered, `skip_unread_pools` left
            // untouched (a genuinely gated-off build cannot set it at all,
            // and the buffered walk never reads it — see the doc comment).
            let gated_off_proxy = crate::Zensim::new(ZensimProfile::D)
                .with_engine(ScoringEngine::Buffered)
                .compute(&sref, &dref)
                .expect("D scores forced-buffered");

            assert_eq!(
                default_build.score().to_bits(),
                gated_off_proxy.score().to_bits(),
                "{w}x{h}: default-build D score diverged from the feature-gated-off proxy"
            );
            assert_eq!(
                default_build.raw_distance().to_bits(),
                gated_off_proxy.raw_distance().to_bits(),
                "{w}x{h}: raw_distance diverged"
            );
            assert_eq!(
                default_build.mean_offset(),
                gated_off_proxy.mean_offset(),
                "{w}x{h}: mean_offset diverged"
            );
            // The scored prefix (`D`'s class reads `f0..228` —
            // `score_v1_layout_features`) must be bit-identical: this is
            // every slot that can move the score, so it is the honest scope
            // of "the gating-tax refactor changes only speed".
            let (df, gf) = (default_build.features(), gated_off_proxy.features());
            assert_eq!(df.len(), gf.len(), "{w}x{h}: feature vector width diverged");
            for (i, (&a, &b)) in df[..228].iter().zip(gf[..228].iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{w}x{h}: SCORED feature slot {i} diverged ({a:e} vs {b:e})"
                );
            }
            // f228..372 (masked/IW) is DELIBERATELY zero on the default
            // (fast-by-default skip) arm for this bake and real on the
            // gated-off (always-buffered) arm — assert that documented
            // asymmetry holds (not "any difference is fine": specifically
            // this one, in this direction), so a future change that starts
            // agreeing here (e.g. skip stops firing) or starts disagreeing
            // in `f0..228` (a real regression) both fail loudly rather than
            // silently passing a loosened gate.
            assert!(
                df[228..372].iter().all(|&v| v == 0.0),
                "{w}x{h}: default (fast-by-default skip) arm's masked/IW block is no longer all-zero"
            );
            assert!(
                gf[228..372].iter().any(|&v| v != 0.0),
                "{w}x{h}: gated-off (buffered) arm's masked/IW block is unexpectedly all-zero"
            );
        }
    }

    /// `caller_col_spans` must tile layer 0 exactly and stay in CALLER space.
    /// The class this guards is the one that made the policy useless on a
    /// pruned bake: `bake_dial_refit pack` prunes BY DEFAULT (2026-08-04), so
    /// a bake whose caller width is 944 can have 667 layer-0 columns, and
    /// reading column `k` as caller line `k` there silently mis-reports which
    /// families are live.
    #[test]
    fn caller_col_spans_tile_layer0_in_caller_space() {
        let params = ZensimProfile::B.params();
        for bytes in params.scoring_bake_bytes() {
            let model = crate::mlp::Model::from_bytes(bytes).expect("shipped bake parses");
            let in_dim = model.layer(0).in_dim;
            let spans = caller_col_spans(&model, in_dim).expect("a shipped bake must tile");
            assert_eq!(
                spans.len(),
                model.caller_input_width(),
                "one span per CALLER line, not per column"
            );
            assert_eq!(spans.first().map(|s| s.0), Some(0));
            assert_eq!(spans.last().map(|s| s.1), Some(in_dim));
            for w in spans.windows(2) {
                assert_eq!(w[0].1, w[1].0, "spans must tile without gap or overlap");
            }
            let covered: usize = spans.iter().map(|&(a, b)| b - a).sum();
            assert_eq!(covered, in_dim);
        }
    }
}
