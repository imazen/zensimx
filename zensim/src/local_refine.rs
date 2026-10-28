//! NEIGHSTEER (lane 2026-10-04): exact local coarse-scale refinement of the
//! v2 pooled features for prepared steering.
//!
//! [`crate::ScoredAttribution::refinement_gain`] predicts the score change
//! from repairing a source rectangle through frozen base-image signals. For
//! an 8×8 block, pyramid scales 1–3 hold 4×4, 2×2 and 1×1 changed coarse
//! pixels inside 11×11 blur windows whose other taps are unrepaired
//! neighbours — a neighbourhood effect the frozen density cannot express
//! (`benchmarks/neighsteer_2026-10-04.md`: M3f 0.81 → 0.98 oracle lift at
//! KADID JPEG 03). This module computes those true finite changes exactly
//! from the fold walk's [`FoldRetention`]: a candidate-distorted local
//! pyramid, locally recomputed phase-A planes, the production per-pixel
//! kernels evaluated old-and-new by the same routine, exact f64 deltas of
//! every cell accumulator the finish consumes, and the production
//! [`feature_v2::finish_channel_scale`] (clamps, soft-peak ratios, weighted
//! pools, edge-width cross-scale chain, transducer luma gate) on both
//! sides.
//!
//! Rev5 uses a finite `[C - radius, C + radius)` window halo instead of
//! the historical recurrence's residue cone. It retains production strip
//! partials, replaces touched partials, and merges them in the original order;
//! stable central moments and sixteen-lane pools are never approximated by raw
//! power subtraction. Gradient halos and the cross-scale edge-width chain are
//! replayed too. HDR snapshots use the same PU conversion as the scored walk.
//!
//! # Precision contract
//!
//! - **Pyramid** — pinned 2×2 box cascade ([`crate::blur::downscale_2x_into`],
//!   `(a+b+c+d)*0.25` left-to-right f32, floor-dropped odd rows/cols). A
//!   coarse pixel changes iff its `2^s` source footprint meets the rect.
//! - **Phase A** — `mu2`/`ssq`/`s12` recomputed over the full **residue
//!   cone**: production's blur is a sliding-sum recurrence
//!   (`sum = sum + add − rem`, `out = sum·inv`), so a changed tap perturbs
//!   the running sum's rounding for every later output of that H row (to
//!   the row end — and below Rev4 the H pass is column-tiled at
//!   `blur::H_TILE_WIDTH`/`ZENSIM_H_TILE`, restarting the sum per tile)
//!   and V column (to the strip's bottom — strips re-init each
//!   `STRIP_ROWS` band). The cone is `[c.x0−r, w)` over every strip
//!   whose `2·HALO_P` halo gather touches `[c.y0, c.y1)`. The blur pass
//!   itself is NOT reimplemented here: each touched strip's mixed wide
//!   window is fed to the production
//!   [`crate::blur::fused_blur_h_ssim_at_revision`] and
//!   [`crate::blur::box_blur_v_from_copy`] over the strip's exact
//!   geometry, so the column tile, the env override, the per-tier
//!   `mul_add` semantics (magetypes' unfused `a*b+c` on scalar/wasm128,
//!   fused FMA elsewhere) and the oracle/canon axis dispatch are
//!   production's by construction on every tier (scalar, v3, v4, v4x,
//!   neon, wasm128). A per-cell `new != old` mask then limits term
//!   evaluation to cells that actually differ (plus the changed set
//!   itself). Replays are bit-exact against an intervened walk's
//!   retention (`cone_planes_bit_exact_and_complete`). `mu1`, `act` are
//!   reference-only and reused (so is `bs2`, which the served families
//!   never read). Audit note (fix round, 2026-10-04): after the blur
//!   rewrite the engine keeps **no f32 arithmetic of its own** — every
//!   f32 value is produced by a production kernel call (per-pixel deltas are
//!   taken after widening to f64, which is exact for two f32s), and every
//!   remaining hand-written reduction is scalar f64, which is IEEE-754
//!   identical on every dispatch tier (the magetypes `mul_add` tier
//!   divergence is an f32-only phenomenon in this codebase). The only
//!   per-pixel term evaluators are production's too:
//! - **Terms** — [`feature_v2::dense_terms32`] (f32; the canonical per-pixel
//!   evaluator at every revision — canon64 differs only in lane
//!   accumulation, which f64 deltas absorb to ~1e-9), and
//!   [`feature_v2::gradient_terms64`] (f64; the gradient kernel's own
//!   border-path evaluator) for OLD and NEW inputs by the same routine.
//! - **Blockiness** — [`feature_v2::bounded_excess`] step terms re-evaluated
//!   at the 8-lattice positions whose (x−1,x)/(y−1,y) dst pairs touch the
//!   changed set.
//! - **Finish** — base cells + f64 deltas through the production finish;
//!   unserved (inactive or v2-off) cells emit literal zero deltas because
//!   the walk leaves those feature slots zero in both computations —
//!   EXCEPT `edge_width_change`, whose finalize writes every channel
//!   (gated only on the per-scale `gradient` pair): an inactive channel's
//!   EWC is still computed, with `(0,0)` gradients, and the engine
//!   reproduces that exactly.
//!
//! Refusal: [`LocalRefineSnapshot::capture`] returns `None` for sampling
//! plans, plans with v2 off, missing/foreign retention dims (including the
//! sub-`MIN_PYRAMID_DIM` reflect-pad, whose scale-0 dims no longer match the
//! image), and [`Self::deltas_range`] refuses scales outside `1..=3` —
//! unsupported work is never silently zero.

use crate::feature_defs::FormulaRevision;
use crate::feature_v2::{
    self, AttrCellSums, BLOCK_LATTICE, BLUR_RADIUS, C_BLOCK, C_EDGEWIDTH, C_GRAD_DECAY, DenseAccum,
    FEATURES_PER_CHANNEL_V2_TOTAL as CH_W, FoldRetention, GradientAccum, V2NewFeatureToggles, idx,
};

/// Coarse pyramid scales the engine serves: `1..=LOCAL_SCALES`.
pub(crate) const LOCAL_SCALES: usize = 3;

/// `f372` — the v2 block's first feature ID at `crate::NUM_SCALES == 4`
/// (`BLOCK_END_V1_POOLS`, feature_defs.rs), so
/// `id = V2_ID_BASE + scale * 87 + ch * 29 + slot`.
const V2_ID_BASE: usize = 372;
/// Features per channel at a scale (the `idx::` layout width).
const SCALE_W: usize = 3 * CH_W;

/// One channel-scale's retained phase-A planes, minus the append-only
/// `bs2` (never read by a served v2 family).
#[derive(Clone, Default)]
struct ChPlanes {
    mu1: Vec<f32>,
    mu2: Vec<f32>,
    ssq: Vec<f32>,
    s12: Vec<f32>,
    act: Vec<f32>,
}

/// Convert `img` to positive XYB at scale 0 (the walk's conversion: HDR
/// through `hdr_source_to_xyb`, SDR through `convert_source_to_xyb_into_slices`)
/// and keep only the channels the walk did not retain (`!channel_active(0, ch)`).
fn fill_inactive_scale0(
    img: &impl crate::source::ImageSource,
    encoding: Option<feature_v2::HdrEncoding>,
    c: &crate::feature_v2::ComputeSet,
    (w0, h0): (usize, usize),
    out: &mut [Vec<f32>; 3],
) {
    let mut planes: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0; w0 * h0]);
    if let Some(encoding) = encoding {
        crate::feature_v2_stream::hdr_source_to_xyb(img, encoding, &mut planes, c.formula_revision);
    } else {
        let [p0, p1, p2] = &mut planes;
        crate::streaming::convert_source_to_xyb_into_slices(
            img,
            p0,
            p1,
            p2,
            w0,
            false,
            0,
            c.formula_revision,
        );
    }
    for ch in 0..3 {
        if !c.channel_active(0, ch) {
            out[ch] = core::mem::take(&mut planes[ch]);
        }
    }
}

/// The retained planes a snapshot keeps, copied from or moved out of a
/// [`FoldRetention`].
pub(crate) struct RetainedParts {
    pyr_src0: [Vec<f32>; 3],
    pyr_dst0: [Vec<f32>; 3],
    pyr_src: [[Vec<f32>; 3]; LOCAL_SCALES],
    pyr_dst: [[Vec<f32>; 3]; LOCAL_SCALES],
    planes: [[ChPlanes; 3]; LOCAL_SCALES],
    strips: [[Vec<(DenseAccum, GradientAccum)>; 3]; LOCAL_SCALES],
}

/// The coarse-scale slice of a walk's [`FoldRetention`] that exact local
/// refinement needs: scale-0 source+distorted planes (the candidate
/// cascade's base), scales 1–3 pyramids, phase-A planes and exact pooled
/// cells, per-scale mean gradients for [`idx::EDGE_WIDTH_CHANGE`], and the
/// plan facts (`served` mask, toggle flags, revision) the finish depends
/// on. Rev5 attribution captures this by default; an explicit
/// `ZENSIM_NEIGHBOUR_EXACT` value enables it only when exactly `"1"`.
/// Older revisions require that opt-in. See [`Self::capture`] for refusals.
///
/// Sparse-retention note: plans with `full_res_xb == false` leave scale-0
/// X/B retention planes **zero-filled** (the walk only retains channels
/// `channel_active(0, ch)`), and `coarse_y_only_scales` plans leave X/B
/// pyramid planes at the masked scales equally unretained — but the 2×2
/// cascade still consumed the real pixels at every level.
/// [`Self::capture`] rebuilds missing scale-0 channels from the
/// caller-supplied images through the producer's own
/// `convert_source_to_xyb` conversion, then re-runs the production
/// `downscale_2x_into` cascade for every unretained coarse channel
/// ([`Self::rebuild_unretained_coarse`]) — the snapshot's pyramid planes
/// are bit-identical to what the walk used at every scale, on every
/// plan, and an inactive channel's `mg` is forced to the `(0,0)` the
/// walk's finalize chain actually uses.
pub(crate) struct LocalRefineSnapshot {
    /// Scale-0 dims — the rectangle coordinate space.
    dims0: (usize, usize),
    /// `dims[scale - 1]` for scales `1..=LOCAL_SCALES`.
    dims: [(usize, usize); LOCAL_SCALES],
    /// Scale-0 pyramids (source = the default `Reference` candidate).
    pyr_src0: [Vec<f32>; 3],
    pyr_dst0: [Vec<f32>; 3],
    /// `pyr_*[scale - 1][ch]` for scales `1..=LOCAL_SCALES`.
    pyr_src: [[Vec<f32>; 3]; LOCAL_SCALES],
    pyr_dst: [[Vec<f32>; 3]; LOCAL_SCALES],
    /// `planes[scale - 1][ch]` — mu1/mu2/ssq/s12/act.
    planes: [[ChPlanes; 3]; LOCAL_SCALES],
    /// `cells[scale - 1][ch]` — exact pooled `AttrCellSums`.
    cells: [[AttrCellSums; 3]; LOCAL_SCALES],
    /// `mg[scale - 1][ch]` — (mean grad src, mean grad dst).
    mg: [[(f64, f64); 3]; LOCAL_SCALES],
    /// `served[scale - 1][ch]` = `v2_blocks(scale) && channel_active(scale, ch)`.
    served: [[bool; 3]; LOCAL_SCALES],
    /// Per-scale `compute.at_scale(scale).gradient` / `.blockiness`.
    gradient_on: [bool; LOCAL_SCALES],
    blockiness_on: [bool; LOCAL_SCALES],
    /// The walk's toggles (transducer_bank, transducers_luma_only, ...).
    toggles: V2NewFeatureToggles,
    /// The computation's formula revision (blur arithmetic + `direct`).
    revision: FormulaRevision,
    omit_pjnd_fragility: bool,
    snapshot_reuse: Option<SnapshotReuse>,
    /// Rev5 ordered production strip partials; central moments are replaced, never subtracted.
    strips: [[Vec<(DenseAccum, GradientAccum)>; 3]; LOCAL_SCALES],
}

/// One retired snapshot's owned buffers per retaining session. No snapshot
/// carries a reference to another snapshot, and live outputs never enter it.
pub(crate) type SnapshotReuse = std::sync::Arc<std::sync::Mutex<Option<RetainedParts>>>;

/// Restore only storage. The next walk rewrites every active element, and
/// `FoldRetention::ensure` checks every length and plan restriction afterwards.
pub(crate) fn restore_snapshot_buffers(ret: &mut FoldRetention) {
    let Some(pool) = &ret.snapshot_reuse else {
        return;
    };
    let Some(parts) = pool.lock().expect("snapshot reuse lock").take() else {
        return;
    };
    if ret.pyr_src.len() <= LOCAL_SCALES
        || ret.pyr_dst.len() <= LOCAL_SCALES
        || ret.planes.len() <= LOCAL_SCALES
    {
        return;
    }
    ret.pyr_src[0] = parts.pyr_src0;
    ret.pyr_dst[0] = parts.pyr_dst0;
    for (si, ((src, dst), planes)) in parts
        .pyr_src
        .into_iter()
        .zip(parts.pyr_dst)
        .zip(parts.planes)
        .enumerate()
    {
        ret.pyr_src[si + 1] = src;
        ret.pyr_dst[si + 1] = dst;
        for (out, p) in ret.planes[si + 1].iter_mut().zip(planes) {
            out.mu1 = p.mu1;
            out.mu2 = p.mu2;
            out.ssq = p.ssq;
            out.s12 = p.s12;
            out.act = p.act;
        }
    }
}

impl Drop for LocalRefineSnapshot {
    fn drop(&mut self) {
        let Some(pool) = self.snapshot_reuse.take() else {
            return;
        };
        let mut slot = pool.lock().expect("snapshot reuse lock");
        if slot.is_none() {
            *slot = Some(RetainedParts {
                pyr_src0: core::mem::take(&mut self.pyr_src0),
                pyr_dst0: core::mem::take(&mut self.pyr_dst0),
                pyr_src: core::mem::take(&mut self.pyr_src),
                pyr_dst: core::mem::take(&mut self.pyr_dst),
                planes: core::mem::take(&mut self.planes),
                strips: core::mem::take(&mut self.strips),
            });
        }
    }
}

/// Scale-0 distorted values for the queried rectangle — the intervention
/// the engine prices, in the positive-XYB space `FoldRetention::pyr_dst`
/// holds, planar per channel with `rect_w * rect_h` elements each.
pub(crate) enum Candidate<'a> {
    /// Full reference repair — the steering map's default intervention.
    Reference,
    /// Explicit values (e.g. a different-quality decode of the rect) —
    /// exercised by the golden tests' non-reference candidates.
    #[cfg_attr(not(test), allow(dead_code))]
    Planar([&'a [f32]; 3]),
}

/// Clipped, nonempty changed-pixel rectangle at one scale.
#[derive(Clone, Copy)]
struct Chg {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}

/// A rectangle's `2^s`-footprint descent: scale-`s` pixels whose footprint
/// `[x·2^s, (x+1)·2^s)` at scale 0 meets `[x0,x1) × [y0,y1)`. An empty
/// rect meets no footprint (`x0 == x1` is `None`, not a 1-column touch).
fn scale_rect(
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    s: usize,
    w: usize,
    h: usize,
) -> Option<Chg> {
    if x0 >= x1 || y0 >= y1 {
        return None;
    }
    let cx0 = x0 >> s;
    let cy0 = y0 >> s;
    let cx1 = x1.saturating_add((1 << s) - 1) >> s;
    let cy1 = y1.saturating_add((1 << s) - 1) >> s;
    let (cx1, cy1) = (cx1.min(w), cy1.min(h));
    if cx0 >= cx1 || cy0 >= cy1 {
        return None;
    }
    Some(Chg {
        x0: cx0,
        y0: cy0,
        x1: cx1,
        y1: cy1,
    })
}

impl LocalRefineSnapshot {
    /// Retain the coarse-scale pieces of `ret` the engine needs.
    ///
    /// `src_dims` is the steering session's IMAGE dims: when the walk
    /// reflect-padded the pair (below `crate::metric::MIN_PYRAMID_DIM`),
    /// `ret.dims[0]` is the padded size and rectangle coordinates no longer
    /// index the retained scale-0 pyramid — refused. Also refused: sampling
    /// plans (`compute.sampling`), plans with v2 off (`compute.v2_blocks`),
    /// and foreign/empty retentions (`ret.dims.len() != NUM_SCALES`).
    /// SDR convenience form. Rev5 HDR steering calls `capture_with_encoding`
    /// so missing channels are rebuilt in the scored PU domain.
    ///
    /// `source`/`distorted` are the session's pixel inputs. Retention only
    /// copies a channel's pyramid rows where `channel_active` holds, so a
    /// `full_res_xb`-off plan leaves scale-0 X/B as zeros — but every
    /// served coarse cell still cascades FROM those pixels. Missing
    /// channels are rebuilt here through the walk's own
    /// [`crate::streaming::convert_source_to_xyb_into_slices`], which is
    /// what the producer's `convert_side_scale0` calls — bit-identical
    /// values, computed once per session rather than per query.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn capture(
        ret: &FoldRetention,
        plan: &crate::feature_plan::Plan,
        src_dims: (usize, usize),
        source: &impl crate::source::ImageSource,
        distorted: &impl crate::source::ImageSource,
    ) -> Option<Self> {
        Self::capture_with_encoding(ret, plan, src_dims, source, distorted, None)
    }

    pub(crate) fn capture_with_encoding(
        ret: &FoldRetention,
        plan: &crate::feature_plan::Plan,
        src_dims: (usize, usize),
        source: &impl crate::source::ImageSource,
        distorted: &impl crate::source::ImageSource,
        encoding: Option<feature_v2::HdrEncoding>,
    ) -> Option<Self> {
        if !Self::capture_admits(ret, plan, src_dims) {
            return None;
        }
        let parts = RetainedParts {
            strips: std::array::from_fn(|s| ret.strips[s + 1].clone()),
            pyr_src0: ret.pyr_src[0].clone(),
            pyr_dst0: ret.pyr_dst[0].clone(),
            pyr_src: std::array::from_fn(|s| ret.pyr_src[s + 1].clone()),
            pyr_dst: std::array::from_fn(|s| ret.pyr_dst[s + 1].clone()),
            planes: std::array::from_fn(|s| {
                std::array::from_fn(|ch| {
                    let p = &ret.planes[s + 1][ch];
                    ChPlanes {
                        mu1: p.mu1.clone(),
                        mu2: p.mu2.clone(),
                        ssq: p.ssq.clone(),
                        s12: p.s12.clone(),
                        act: p.act.clone(),
                    }
                })
            }),
        };
        Self::capture_from_parts(
            ret, plan, src_dims, source, distorted, encoding, parts, None,
        )
    }

    /// [`Self::capture_with_encoding`], but MOVING the planes out of `ret`
    /// instead of copying them: the steering session's last use of its
    /// retention in a call, so the snapshot and the session never hold two
    /// copies. The moved values are the ones a copy would
    /// hold, so the snapshot is byte-identical. `ret` keeps its dims and
    /// metadata; `FoldRetention::ensure` re-creates exactly the moved buffers
    /// on the next walk, which rewrites every element before any read.
    ///
    /// `cached_source0` is the source's scale-0 XYB from a prepared reference,
    /// passed only when `feature_v2::retained_ref_feed_admits` holds: those
    /// planes are the bytes the walk's conversion produces, so the source's
    /// unretained channels are copied from it instead of converted again.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn capture_taking(
        ret: &mut FoldRetention,
        plan: &crate::feature_plan::Plan,
        src_dims: (usize, usize),
        source: &impl crate::source::ImageSource,
        distorted: &impl crate::source::ImageSource,
        encoding: Option<feature_v2::HdrEncoding>,
        cached_source0: Option<&[Vec<f32>; 3]>,
    ) -> Option<Self> {
        if !Self::capture_admits(ret, plan, src_dims) {
            return None;
        }
        let snapshot_reuse = ret.complete_distorted0.then(|| {
            std::sync::Arc::clone(
                ret.snapshot_reuse
                    .get_or_insert_with(|| std::sync::Arc::new(std::sync::Mutex::new(None))),
            )
        });
        let take = core::mem::take::<Vec<f32>>;
        let parts = RetainedParts {
            strips: std::array::from_fn(|s| ret.strips[s + 1].each_mut().map(core::mem::take)),
            pyr_src0: ret.pyr_src[0].each_mut().map(take),
            pyr_dst0: ret.pyr_dst[0].each_mut().map(take),
            pyr_src: std::array::from_fn(|s| ret.pyr_src[s + 1].each_mut().map(take)),
            pyr_dst: std::array::from_fn(|s| ret.pyr_dst[s + 1].each_mut().map(take)),
            planes: std::array::from_fn(|s| {
                std::array::from_fn(|ch| {
                    let p = &mut ret.planes[s + 1][ch];
                    ChPlanes {
                        mu1: take(&mut p.mu1),
                        mu2: take(&mut p.mu2),
                        ssq: take(&mut p.ssq),
                        s12: take(&mut p.s12),
                        act: take(&mut p.act),
                    }
                })
            }),
        };
        let mut snapshot = Self::capture_from_parts(
            ret,
            plan,
            src_dims,
            source,
            distorted,
            encoding,
            parts,
            cached_source0,
        )?;
        snapshot.snapshot_reuse = snapshot_reuse;
        Some(snapshot)
    }

    /// The one capture body; `parts` are the retained planes, copied or moved.
    #[allow(clippy::too_many_arguments)]
    fn capture_from_parts(
        ret: &FoldRetention,
        plan: &crate::feature_plan::Plan,
        src_dims: (usize, usize),
        source: &impl crate::source::ImageSource,
        distorted: &impl crate::source::ImageSource,
        encoding: Option<feature_v2::HdrEncoding>,
        parts: RetainedParts,
        cached_source0: Option<&[Vec<f32>; 3]>,
    ) -> Option<Self> {
        let c = &plan.compute;
        let (w0, h0) = src_dims;
        let RetainedParts {
            strips,
            mut pyr_src0,
            mut pyr_dst0,
            pyr_src,
            pyr_dst,
            planes,
        } = parts;
        if (0..3).any(|ch| !c.channel_active(0, ch)) {
            // One image at a time: each conversion fills all three channels
            // but only the inactive ones are kept, so dropping one side's
            // planes before converting the other halves the transient.
            match cached_source0 {
                Some(cached) => {
                    for ch in 0..3 {
                        if !c.channel_active(0, ch) {
                            debug_assert_eq!(cached[ch].len(), w0 * h0);
                            pyr_src0[ch].clone_from(&cached[ch]);
                        }
                    }
                }
                None => fill_inactive_scale0(source, encoding, c, (w0, h0), &mut pyr_src0),
            }
            if !ret.complete_distorted0 {
                fill_inactive_scale0(distorted, encoding, c, (w0, h0), &mut pyr_dst0);
            }
        }
        let mut served = [[false; 3]; LOCAL_SCALES];
        let mut gradient_on = [false; LOCAL_SCALES];
        let mut blockiness_on = [false; LOCAL_SCALES];
        for s in 0..LOCAL_SCALES {
            let local = c.at_scale(s + 1);
            gradient_on[s] = local.gradient;
            blockiness_on[s] = local.blockiness;
            for (ch, srv) in served[s].iter_mut().enumerate() {
                *srv = local.v2_blocks && c.channel_active(s + 1, ch);
            }
        }
        let mut this = Self {
            dims0: ret.dims[0],
            dims: [ret.dims[1], ret.dims[2], ret.dims[3]],
            pyr_src0,
            pyr_dst0,
            pyr_src,
            pyr_dst,
            planes,
            cells: std::array::from_fn(|s| ret.cells[s + 1]),
            mg: std::array::from_fn(|s| ret.mg[s + 1]),
            served,
            gradient_on,
            blockiness_on,
            toggles: plan.toggles(),
            revision: c.formula_revision,
            omit_pjnd_fragility: c.omit_pjnd_fragility,
            snapshot_reuse: None,
            strips,
        };
        if !this.rebuild_unretained_coarse(c) {
            return None;
        }
        if this.revision >= FormulaRevision::Rev5 {
            for si in 0..LOCAL_SCALES {
                for ch in 0..3 {
                    if this.served[si][ch] {
                        let (_, h) = this.dims[si];
                        assert_eq!(
                            this.strips[si][ch].len(),
                            h.div_ceil(feature_v2::STRIP_ROWS),
                            "retaining walk must supply every served production strip"
                        );
                    }
                }
            }
        }
        Some(this)
    }

    /// Whether [`Self::capture_with_encoding`] returns `Some` for this
    /// retention, plan and source size. It is decided by the plan and the
    /// retained pyramid dimensions alone, so a caller can know before capturing
    /// (prepared steering zeroes the frozen v2 density only when a capture will
    /// replace it, and captures after the map pass so the two never coexist).
    /// The one owner of the refusal rule: the capture itself calls it, and
    /// `rebuild_unretained_coarse`'s dimension test is the same halving check.
    pub(crate) fn capture_admits(
        ret: &FoldRetention,
        plan: &crate::feature_plan::Plan,
        src_dims: (usize, usize),
    ) -> bool {
        let c = &plan.compute;
        if c.sampling.is_some()
            || !c.v2_blocks
            || ret.dims.len() != crate::NUM_SCALES
            || ret.dims[0] != src_dims
        {
            return false;
        }
        (1..=LOCAL_SCALES).all(|s| {
            let (pw, ph) = ret.dims[s - 1];
            (pw / 2, ph / 2) == ret.dims[s]
        })
    }

    /// Replay a production strip with retained reference planes and optional
    /// candidate values. The caller replaces only strips meeting the dependency cone.
    fn strip_accumulators(
        &self,
        si: usize,
        ch: usize,
        strip: usize,
        query: Option<(&Query<'_>, &Planes)>,
    ) -> (DenseAccum, GradientAccum) {
        let (w, h) = self.dims[si];
        let y0 = strip * feature_v2::STRIP_ROWS;
        let sh = feature_v2::STRIP_ROWS.min(h - y0);
        let base = y0 * w;
        let n = sh * w;
        let p = &self.planes[si][ch];
        let src = &self.pyr_src[si][ch];
        // Snapshot construction reads the untouched strip. Only a repair
        // query needs owned copies in which to replace the changed samples.
        let mut dst = std::borrow::Cow::Borrowed(&self.pyr_dst[si][ch][base..base + n]);
        let mut mu2 = std::borrow::Cow::Borrowed(&p.mu2[base..base + n]);
        let mut ssq = std::borrow::Cow::Borrowed(&p.ssq[base..base + n]);
        let mut s12 = std::borrow::Cow::Borrowed(&p.s12[base..base + n]);
        if let Some((q, planes)) = query {
            let dst = dst.to_mut();
            let mu2 = mu2.to_mut();
            let ssq = ssq.to_mut();
            let s12 = s12.to_mut();
            let changed = q.changed[si].expect("a replay strip has a changed coarse rectangle");
            for y in y0.max(changed.y0)..(y0 + sh).min(changed.y1) {
                for x in changed.x0..changed.x1 {
                    dst[(y - y0) * w + x] = q.dst_at(si + 1, ch, y, x);
                }
            }
            for y in y0.max(planes.ry0)..(y0 + sh).min(planes.ry0 + planes.rh) {
                let i = (y - y0) * w + planes.rx0;
                let j = (y - planes.ry0) * planes.rw;
                mu2[i..i + planes.rw].copy_from_slice(&planes.mu2[j..j + planes.rw]);
                ssq[i..i + planes.rw].copy_from_slice(&planes.ssq[j..j + planes.rw]);
                s12[i..i + planes.rw].copy_from_slice(&planes.s12[j..j + planes.rw]);
            }
        }
        let dense = feature_v2::refinement_dense_strip(
            &src[base..base + n],
            &dst,
            &p.mu1[base..base + n],
            &mu2,
            &ssq,
            &s12,
            &p.act[base..base + n],
            w,
            sh,
            self.toggles.transducer_bank,
        );
        let mut grad = GradientAccum::default();
        if self.gradient_on[si] {
            let mut sg = vec![0.0; (sh + 2) * w];
            let mut dg = vec![0.0; (sh + 2) * w];
            for j in 0..sh + 2 {
                let y = feature_v2::reflect_101(y0 as isize + j as isize - 1, h);
                sg[j * w..(j + 1) * w].copy_from_slice(&src[y * w..(y + 1) * w]);
                dg[j * w..(j + 1) * w].copy_from_slice(&self.pyr_dst[si][ch][y * w..(y + 1) * w]);
                if let Some((q, _)) = query {
                    let c = q.changed[si].expect("a replay strip has a changed coarse rectangle");
                    if y >= c.y0 && y < c.y1 {
                        for x in c.x0..c.x1 {
                            dg[j * w + x] = q.dst_at(si + 1, ch, y, x);
                        }
                    }
                }
            }
            grad = feature_v2::refinement_gradient_strip(&sg, &dg, &p.act[base..base + n], w, sh);
        }
        (dense, grad)
    }

    /// Rebuild every coarse plane the engine reads that the walk did not
    /// retain (fix-round BLOCKER, 2026-10-04). `channel_active(s, ch)`
    /// gates [`FoldRetention::copy_strip`], so a `coarse_y_only_scales`
    /// plan leaves that channel's `pyr_src`/`pyr_dst` rows **zero-filled**
    /// in a fresh retention — or **stale from a previous pair** when a
    /// `FoldRetention` is reused at the same dims (`ensure` is then a
    /// no-op). The cascade still consumed the real values: the strip
    /// producer builds every channel's pyramid unconditionally through
    /// [`crate::blur::downscale_2x_into`] (a pure function of the parent
    /// rows), so re-running it level by level from the scale-0 planes —
    /// complete after [`Self::capture`]'s channel rebuild — reproduces
    /// the walked planes bit-exactly. `pyr_src` is included even though
    /// only `pyr_dst` feeds `dst_at` today: the plane is cheap, and a
    /// retained-zero that is never read is still a lie waiting for a
    /// caller. `mg` needs no rebuild — an inactive channel's walk-side
    /// `grads` entry is exactly `(0,0)` — it is FORCED to `(0,0)` here so
    /// a stale retention cannot leak a previous pair's gradients into the
    /// [`idx::EDGE_WIDTH_CHANGE`] chain. Returns `false` (capture
    /// refuses) when a level's dims do not halve its parent's — a
    /// retention the cascade cannot rebuild exactly is never read.
    fn rebuild_unretained_coarse(&mut self, c: &crate::feature_v2::ComputeSet) -> bool {
        for s in 1..=LOCAL_SCALES {
            let (pw, ph) = if s == 1 { self.dims0 } else { self.dims[s - 2] };
            let (cw, chh) = self.dims[s - 1];
            if (pw / 2, ph / 2) != (cw, chh) {
                return false;
            }
            for ch in 0..3 {
                if c.channel_active(s, ch) {
                    continue;
                }
                for side in 0..2 {
                    let (planes, plane0) = if side == 0 {
                        (&mut self.pyr_src, &mut self.pyr_src0)
                    } else {
                        (&mut self.pyr_dst, &mut self.pyr_dst0)
                    };
                    let (parent, child) = if s == 1 {
                        (plane0[ch].as_slice(), &mut planes[0][ch])
                    } else {
                        let (head, tail) = planes.split_at_mut(s - 1);
                        (head[s - 2][ch].as_slice(), &mut tail[0][ch])
                    };
                    child.resize(cw * chh, 0.0);
                    crate::blur::downscale_2x_into(parent, pw, child, cw, chh);
                }
                self.mg[s - 1][ch] = (0.0, 0.0);
            }
        }
        true
    }

    /// Directly owned heap bytes (excluding the shared session reuse slot) — the number the lane's cost section
    /// reports as "snapshot memory" (scale-0 src+dst, scales 1–3 src/dst +
    /// five phase-A planes; cells/mg are a few KB). Read by the cost
    /// measurement test.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn heap_bytes(&self) -> usize {
        let p3 = |p: &[Vec<f32>; 3]| -> usize { p.iter().map(|v| v.len() * 4).sum() };
        let mut n = p3(&self.pyr_src0) + p3(&self.pyr_dst0);
        for s in 0..LOCAL_SCALES {
            n += p3(&self.pyr_src[s]) + p3(&self.pyr_dst[s]);
            for ch in 0..3 {
                let p = &self.planes[s][ch];
                n += self.strips[s][ch].len() * core::mem::size_of::<(DenseAccum, GradientAccum)>();
                n += 4 * (p.mu1.len() + p.mu2.len() + p.ssq.len() + p.s12.len() + p.act.len());
            }
        }
        n
    }

    /// `Σ_k s_k · Δf_k` over every v2 pooled feature at scales `1..=3`, the
    /// term [`crate::ScoredAttribution::refinement_gain`] adds when neighbour
    /// replay is enabled. `s` is the identity-indexed sensitivity row.
    pub(crate) fn weighted_gain(
        &self,
        s: &[f64],
        rect: (usize, usize, usize, usize),
        candidate: &Candidate,
    ) -> Option<f64> {
        let deltas = self.deltas(rect, candidate)?;
        let mut g = 0.0f64;
        for (id, d) in deltas {
            if let Some(&sk) = s.get(id) {
                g += sk * d;
            }
        }
        Some(g)
    }

    /// Exact finite deltas `(id, Δ)` for every v2 pooled feature at scales
    /// `1..=3` — the task's default scale range.
    pub(crate) fn deltas(
        &self,
        rect: (usize, usize, usize, usize),
        candidate: &Candidate,
    ) -> Option<Vec<(usize, f64)>> {
        self.deltas_range(rect, candidate, 1..=LOCAL_SCALES)
    }

    /// [`Self::deltas`] over an explicit scale range; refuses anything
    /// outside `1..=3` (`None`) — the engine implements those scales only.
    /// `rect` is the half-open scale-0 rectangle `(x0, y0, x1, y1)`;
    /// coordinates are clamped to the image. `candidate` supplies the
    /// distorted scale-0 values inside it (`None` = reference repair).
    pub(crate) fn deltas_range(
        &self,
        rect: (usize, usize, usize, usize),
        candidate: &Candidate,
        scales: core::ops::RangeInclusive<usize>,
    ) -> Option<Vec<(usize, f64)>> {
        if *scales.start() < 1 || *scales.end() > LOCAL_SCALES || scales.is_empty() {
            return None;
        }
        let (w0, h0) = self.dims0;
        let x0 = rect.0.min(w0);
        let y0 = rect.1.min(h0);
        let x1 = rect.2.min(w0).max(x0);
        let y1 = rect.3.min(h0).max(y0);
        // The candidate the cascade reads inside `rect`: explicit planes,
        // or the retained scale-0 source (full reference repair).
        let cand: Option<[&[f32]; 3]> = match candidate {
            Candidate::Reference => None,
            Candidate::Planar(p) => {
                let rw = x1 - x0;
                let rh = y1 - y0;
                if p.iter().any(|v| v.len() != rw * rh) {
                    return None;
                }
                Some(*p)
            }
        };
        let mut q = Query {
            snap: self,
            rect: (x0, y0, x1, y1),
            cand,
            changed: [None; LOCAL_SCALES],
            new_dst: std::array::from_fn(|_| std::array::from_fn(|_| Vec::new())),
            err: false,
        };
        let revision = crate::ssim_form::effective_revision(self.revision);
        q.err = revision >= FormulaRevision::Rev3;
        for (si, chg) in q.changed.iter_mut().enumerate() {
            let (w, h) = self.dims[si];
            *chg = scale_rect(x0, y0, x1, y1, si + 1, w, h);
        }
        q.build_changed();

        // Per-cell finishes. `outs_*` are the production-finished feature
        // values; EWC is patched afterwards from the mg arrays.
        let mut outs_base = [[[0.0f64; CH_W]; 3]; LOCAL_SCALES];
        let mut outs_new = [[[0.0f64; CH_W]; 3]; LOCAL_SCALES];
        let mut mg_new = [[(0.0f64, 0.0f64); 3]; LOCAL_SCALES];
        for si in 0..LOCAL_SCALES {
            for ch in 0..3 {
                if !self.served[si][ch] {
                    continue; // unserved cells: base == new == the walk's zero slots
                }
                let cell = &self.cells[si][ch];
                feature_v2::finish_channel_scale_planned(
                    &cell.dense,
                    &cell.grad,
                    cell.blockiness,
                    cell.n,
                    &mut outs_base[si][ch],
                    self.omit_pjnd_fragility,
                );
                feature_v2::apply_transducer_luma_gate(&mut outs_base[si][ch], ch, self.toggles);
                let mut dense_n = cell.dense;
                let mut grad_n = cell.grad;
                let mut dblock = 0.0f64;
                if let Some(c) = q.changed[si] {
                    let planes = q.recompute_planes(si, ch, &c);
                    if self.revision >= FormulaRevision::Rev5 {
                        dense_n = DenseAccum::default();
                        grad_n = GradientAccum::default();
                        for (strip, &(d, g)) in self.strips[si][ch].iter().enumerate() {
                            let sy = strip * feature_v2::STRIP_ROWS;
                            let end = (sy + feature_v2::STRIP_ROWS).min(self.dims[si].1);
                            let touches_dense = sy < planes.ry0 + planes.rh && end > planes.ry0;
                            let touches_grad = sy < (c.y1 + 1).min(self.dims[si].1)
                                && end > c.y0.saturating_sub(1);
                            let (d, g) = if touches_dense || touches_grad {
                                self.strip_accumulators(si, ch, strip, Some((&q, &planes)))
                            } else {
                                (d, g)
                            };
                            dense_n.accumulate(&d);
                            grad_n.accumulate(&g);
                        }
                    } else {
                        let mut dacc = DenseAccum::default();
                        let mut gacc = GradientAccum::default();
                        q.dense_delta(si, ch, &planes, &c, &mut dacc);
                        if self.gradient_on[si] {
                            q.grad_delta(si, ch, &c, &mut gacc);
                        }
                        dense_n.accumulate(&dacc);
                        grad_n.accumulate(&gacc);
                    }
                    if self.blockiness_on[si] {
                        dblock = q.blockiness_delta(si, ch, &c);
                    }
                }
                mg_new[si][ch] = feature_v2::finish_channel_scale_planned(
                    &dense_n,
                    &grad_n,
                    cell.blockiness + dblock,
                    cell.n,
                    &mut outs_new[si][ch],
                    self.omit_pjnd_fragility,
                );
                feature_v2::apply_transducer_luma_gate(&mut outs_new[si][ch], ch, self.toggles);
            }
        }

        // EDGE_WIDTH_CHANGE: `features[prev+EWC] =
        // 1 - bsim(mg[s+1].0/(mg[s].0 + C_DECAY), mg[s+1].1/(mg[s].1 +
        // C_DECAY), C_EW)`; the coarsest scale copies the second-coarsest
        // (production foldapp finalize). As in production, the slot is
        // written only where both mixed scales run gradient work and the
        // channel at both; elsewhere it keeps the structural 0, which the
        // coarsest-scale copy then repeats.
        for si in 0..LOCAL_SCALES {
            for ch in 0..3 {
                if si + 1 < LOCAL_SCALES {
                    if !(self.gradient_on[si]
                        && self.gradient_on[si + 1]
                        && self.served[si][ch]
                        && self.served[si + 1][ch])
                    {
                        continue;
                    }
                    let (bs, bd) = self.mg[si + 1][ch];
                    let (ps, pd) = self.mg[si][ch];
                    let ds = bs / (ps + C_GRAD_DECAY);
                    let dd = bd / (pd + C_GRAD_DECAY);
                    outs_base[si][ch][idx::EDGE_WIDTH_CHANGE] =
                        feature_v2::bounded_dissim(ds, dd, C_EDGEWIDTH, revision.stable_forms());
                    let (bs, bd) = mg_new[si + 1][ch];
                    let (ps, pd) = mg_new[si][ch];
                    let ds = bs / (ps + C_GRAD_DECAY);
                    let dd = bd / (pd + C_GRAD_DECAY);
                    outs_new[si][ch][idx::EDGE_WIDTH_CHANGE] =
                        feature_v2::bounded_dissim(ds, dd, C_EDGEWIDTH, revision.stable_forms());
                } else {
                    outs_base[si][ch][idx::EDGE_WIDTH_CHANGE] =
                        outs_base[si - 1][ch][idx::EDGE_WIDTH_CHANGE];
                    outs_new[si][ch][idx::EDGE_WIDTH_CHANGE] =
                        outs_new[si - 1][ch][idx::EDGE_WIDTH_CHANGE];
                }
            }
        }

        let mut out = Vec::with_capacity(LOCAL_SCALES * 3 * CH_W);
        for si in 0..LOCAL_SCALES {
            if !scales.contains(&(si + 1)) {
                continue;
            }
            for ch in 0..3 {
                for slot in 0..CH_W {
                    out.push((
                        V2_ID_BASE + (si + 1) * SCALE_W + ch * CH_W + slot,
                        outs_new[si][ch][slot] - outs_base[si][ch][slot],
                    ));
                }
            }
        }
        Some(out)
    }
}

/// The recomputed dst-dependent phase-A planes over the **residue cone**
/// `[rx0, w) × [ry0, ry1)`: the blur's sliding-sum recurrence carries a
/// rounding residue downstream of any changed tap — H rows differ to the
/// row end, V columns to the strip end — so the plane values that differ
/// from the retained walk live in that cone, not merely `dilate(C, r)`.
/// `diff` marks cells where any recomputed plane differs from the
/// retained walk (the dense-delta eval set; `C` is unioned in by the
/// caller for the dst-only terms).
struct Planes {
    mu2: Vec<f32>,
    ssq: Vec<f32>,
    s12: Vec<f32>,
    diff: Vec<u8>,
    rx0: usize,
    ry0: usize,
    rw: usize,
    rh: usize,
}

/// One query's working state: the clipped intervention rectangle, the
/// per-scale changed sets and their new distorted values (built by the
/// pinned cascade), and the revision flag shared by the per-pixel terms.
struct Query<'a> {
    snap: &'a LocalRefineSnapshot,
    /// Clipped scale-0 rectangle.
    rect: (usize, usize, usize, usize),
    /// Explicit candidate planes (None = reference repair via `pyr_src0`).
    cand: Option<[&'a [f32]; 3]>,
    /// `changed[scale - 1]` — `None` where the footprint misses the plane.
    changed: [Option<Chg>; LOCAL_SCALES],
    /// `new_dst[scale - 1][ch]` — the cascade's new values over `changed`.
    new_dst: [[Vec<f32>; 3]; LOCAL_SCALES],
    /// `revision >= Rev3` — the `s12` direct-error and `dense_terms32`
    /// `direct` flag.
    err: bool,
}

impl<'a> Query<'a> {
    /// The (possibly candidate-substituted) distorted pyramid value.
    /// `scale` is the ABSOLUTE pyramid level (`0` = candidate rect space).
    #[inline]
    fn dst_at(&self, scale: usize, ch: usize, y: usize, x: usize) -> f32 {
        if scale == 0 {
            let (x0, y0, x1, y1) = self.rect;
            if y >= y0 && y < y1 && x >= x0 && x < x1 {
                return match self.cand {
                    Some(p) => p[ch][(y - y0) * (x1 - x0) + (x - x0)],
                    // Full reference repair: the retained scale-0 source.
                    None => self.snap.pyr_src0[ch][y * self.snap.dims0.0 + x],
                };
            }
            return self.snap.pyr_dst0[ch][y * self.snap.dims0.0 + x];
        }
        if let Some(c) = self.changed[scale - 1]
            && y >= c.y0
            && y < c.y1
            && x >= c.x0
            && x < c.x1
        {
            return self.new_dst[scale - 1][ch][(y - c.y0) * (c.x1 - c.x0) + (x - c.x0)];
        }
        let (w, _h) = self.snap.dims[scale - 1];
        self.snap.pyr_dst[scale - 1][ch][y * w + x]
    }

    /// Build every `new_dst` rect: for scale `s`, output pixel `(x, y)` is
    /// the pinned 2×2 box over level `s−1` rows `2y, 2y+1` — which is
    /// exactly the output rows of `downscale_2x_into` applied to the
    /// `[2·cy0, 2·cy1)` input band of the mixed level `s−1`.
    fn build_changed(&mut self) {
        for ch in 0..3 {
            for si in 0..LOCAL_SCALES {
                let Some(c) = self.changed[si] else { continue };
                let (wp, _hp) = if si == 0 {
                    self.snap.dims0
                } else {
                    self.snap.dims[si - 1]
                };
                let (ws, _hs) = self.snap.dims[si];
                let band_h = 2 * (c.y1 - c.y0);
                let mut band = vec![0.0f32; band_h * wp];
                for j in 0..band_h {
                    let y = 2 * c.y0 + j;
                    for x in 0..wp {
                        band[j * wp + x] = self.dst_at(si, ch, y, x);
                    }
                }
                let out_w = wp / 2;
                debug_assert_eq!(out_w, ws);
                let out_h = band_h / 2;
                let mut out = vec![0.0f32; out_w * out_h];
                crate::blur::downscale_2x_into(&band, wp, &mut out, out_w, out_h);
                let cw = c.x1 - c.x0;
                let mut newd = vec![0.0f32; cw * (c.y1 - c.y0)];
                for y in 0..c.y1 - c.y0 {
                    newd[y * cw..(y + 1) * cw]
                        .copy_from_slice(&out[y * out_w + c.x0..y * out_w + c.x0 + cw]);
                }
                self.new_dst[si][ch] = newd;
            }
        }
    }

    /// First wide-buffer row whose gathered image row lies inside
    /// `[c.y0, c.y1)` — `None` when this strip's halo'd range misses the
    /// changed rows entirely (no tap can differ → strip untouched).
    fn strip_first_changed(strip_y0: usize, strip_h: usize, h: usize, c: &Chg) -> Option<usize> {
        let wide_h = strip_h + 2 * feature_v2::HALO_P;
        (0..wide_h).find(|&j| {
            let gy = feature_v2::reflect_101(
                strip_y0 as isize - feature_v2::HALO_P as isize + j as isize,
                h,
            );
            gy >= c.y0 && gy < c.y1
        })
    }

    /// Recompute the dst-dependent phase-A planes over the residue cone
    /// for one (scale, channel), reproducing the walk's strip blur pass
    /// **through production's own kernels**. The sliding-sum recurrences
    /// make the true diff region downstream-transitive: a changed tap
    /// perturbs the running sum's rounding for every later output of
    /// that H row (to the row end — and below Rev4 the H pass is
    /// column-TILED at `blur::H_TILE_WIDTH`/`ZENSIM_H_TILE`, restarting
    /// the sum per tile) and of that V column (to the strip's bottom —
    /// strips re-init each `STRIP_ROWS` band). The region is therefore
    /// `[c.x0-r, w)` over every strip whose halo gather touches
    /// `[c.y0, c.y1)`, from that strip's top to its bottom.
    ///
    /// The blur pass is NOT hand-copied: each touched strip gathers its
    /// wide halo window (the producer's reflect-101 shape) with the
    /// candidate cascade spliced into `dst`, then
    /// [`crate::blur::fused_blur_h_ssim_at_revision`] and
    /// [`crate::blur::box_blur_v_from_copy`] run over exactly the strip's
    /// geometry — so the column tile, the env override, the per-tier
    /// `mul_add` semantics (magetypes' unfused `a*b+c` on scalar/wasm128,
    /// fused FMA elsewhere) and the oracle/canon axis dispatch are all
    /// production's BY CONSTRUCTION, on every tier. The V input is the
    /// cone's column slice `[rx0, w)`: V-blur columns are independent,
    /// so the slice is bit-identical to the full-width pass's.
    fn recompute_planes(&self, si: usize, ch: usize, c: &Chg) -> Planes {
        if self.snap.revision >= FormulaRevision::Rev5 {
            return self.recompute_local_planes(si, ch, c);
        }
        let (w, h) = self.snap.dims[si];
        let rx0 = c.x0.saturating_sub(BLUR_RADIUS);
        let strips: Vec<(usize, usize)> = (0..h.div_ceil(feature_v2::STRIP_ROWS))
            .map(|s| {
                let sy0 = s * feature_v2::STRIP_ROWS;
                (sy0, feature_v2::STRIP_ROWS.min(h - sy0))
            })
            .collect();
        let first = strips
            .iter()
            .position(|&(sy0, sh)| Self::strip_first_changed(sy0, sh, h, c).is_some());
        let last = strips
            .iter()
            .rposition(|&(sy0, sh)| Self::strip_first_changed(sy0, sh, h, c).is_some());
        let (Some(first), Some(last)) = (first, last) else {
            // Unreachable: `c` is nonempty inside the plane, so the strip
            // containing `c.y0` gathers it. Keep the refusal honest.
            return Planes {
                mu2: Vec::new(),
                ssq: Vec::new(),
                s12: Vec::new(),
                diff: Vec::new(),
                rx0,
                ry0: 0,
                rw: 0,
                rh: 0,
            };
        };
        let ry0 = strips[first].0;
        let ry1 = strips[last].0 + strips[last].1;
        let local = self.snap.revision >= FormulaRevision::Rev5;
        let rx1 = if local {
            (c.x1 + BLUR_RADIUS).min(w)
        } else {
            w
        };
        let ry0 = if local {
            c.y0.saturating_sub(BLUR_RADIUS)
        } else {
            ry0
        };
        let ry1 = if local {
            (c.y1 + BLUR_RADIUS).min(h)
        } else {
            ry1
        };
        let rw = rx1 - rx0;
        let rh = ry1 - ry0;
        let old = &self.snap.planes[si][ch];
        let src_plane = &self.snap.pyr_src[si][ch];
        let mut planes = Planes {
            mu2: vec![0.0; rw * rh],
            ssq: vec![0.0; rw * rh],
            s12: vec![0.0; rw * rh],
            diff: vec![0u8; rw * rh],
            rx0,
            ry0,
            rw,
            rh,
        };
        for &(strip_y0, strip_h) in &strips[first..=last] {
            if Self::strip_first_changed(strip_y0, strip_h, h, c).is_none() {
                // A strip inside the span whose halo misses the changed
                // rows: every output is retained-identical — copy the
                // retained rows so region indexing stays uniform.
                for y in strip_y0.max(ry0)..(strip_y0 + strip_h).min(ry1) {
                    let o = (y - ry0) * rw;
                    let i = y * w + rx0;
                    planes.mu2[o..o + rw].copy_from_slice(&old.mu2[i..i + rw]);
                    planes.ssq[o..o + rw].copy_from_slice(&old.ssq[i..i + rw]);
                    planes.s12[o..o + rw].copy_from_slice(&old.s12[i..i + rw]);
                }
                continue;
            }
            // Gather the strip's wide halo window — the same reflect-101
            // gather production's `fill_wide` runs, with the candidate
            // cascade spliced into the dst rows.
            let wide_h = strip_h + 2 * feature_v2::HALO_P;
            // Local windows need only the cone columns and their H halo.
            // Gather true image reflection explicitly; the production H pass
            // reads interior windows at offset radius and cannot see this
            // temporary buffer's boundaries.
            let hw = if local { rw + 2 * BLUR_RADIUS } else { w };
            let hoff = if local { BLUR_RADIUS } else { rx0 };
            let mut src_wide = vec![0.0f32; wide_h * hw];
            let mut dst_wide = vec![0.0f32; wide_h * hw];
            for j in 0..wide_h {
                let gy = feature_v2::reflect_101(
                    strip_y0 as isize - feature_v2::HALO_P as isize + j as isize,
                    h,
                );
                for x in 0..hw {
                    let gx = if local {
                        feature_v2::reflect_101(rx0 as isize - BLUR_RADIUS as isize + x as isize, w)
                    } else {
                        x
                    };
                    src_wide[j * hw + x] = src_plane[gy * w + gx];
                    dst_wide[j * hw + x] = self.dst_at(si + 1, ch, gy, gx);
                }
            }
            // H over the full strip rows — the whole width, because the
            // running-sum init and the column tile are part of the
            // semantics (mu1 is emitted but src-only; retained instead).
            let mut h_mu1 = vec![0.0f32; wide_h * hw];
            let mut h_mu2 = vec![0.0f32; wide_h * hw];
            let mut h_ssq = vec![0.0f32; wide_h * hw];
            let mut h_s12 = vec![0.0f32; wide_h * hw];
            crate::blur::fused_blur_h_ssim_at_revision(
                &src_wide,
                &dst_wide,
                &mut h_mu1,
                &mut h_mu2,
                &mut h_ssq,
                &mut h_s12,
                hw,
                wide_h,
                BLUR_RADIUS,
                self.snap.revision,
            );
            // V per plane over the cone's column slice (independent
            // columns → slice equals the full pass). Outputs for plane
            // rows [strip_y0, strip_y0+strip_h) land at wide rows
            // [HALO_P, HALO_P+strip_h).
            let mut v_in = vec![0.0f32; wide_h * rw];
            let mut v_out = vec![0.0f32; wide_h * rw];
            for (h_plane, out_plane) in [
                (&h_mu2, &mut planes.mu2),
                (&h_ssq, &mut planes.ssq),
                (&h_s12, &mut planes.s12),
            ] {
                for j in 0..wide_h {
                    v_in[j * rw..(j + 1) * rw]
                        .copy_from_slice(&h_plane[j * hw + hoff..j * hw + hoff + rw]);
                }
                crate::blur::box_blur_v_from_copy(&v_in, &mut v_out, rw, wide_h, BLUR_RADIUS);
                for y in strip_y0.max(ry0)..(strip_y0 + strip_h).min(ry1) {
                    let j = y - strip_y0 + feature_v2::HALO_P;
                    let o = (y - ry0) * rw;
                    out_plane[o..o + rw].copy_from_slice(&v_out[j * rw..(j + 1) * rw]);
                }
            }
            for y in strip_y0.max(ry0)..(strip_y0 + strip_h).min(ry1) {
                let o = (y - ry0) * rw;
                for col in 0..rw {
                    let j = o + col;
                    let i = y * w + rx0 + col;
                    planes.diff[j] |= (planes.mu2[j] != old.mu2[i]
                        || planes.ssq[j] != old.ssq[i]
                        || planes.s12[j] != old.s12[i]) as u8;
                }
            }
        }
        planes
    }

    /// A Rev5 window has no strip-phase dependence: gather exactly the
    /// finite output cone plus one H/V halo, reflecting against the image.
    /// Core outputs never read temporary-buffer boundaries. Production
    /// pair-tree kernels therefore reproduce the full fold bit for bit.
    fn recompute_local_planes(&self, si: usize, ch: usize, c: &Chg) -> Planes {
        let (w, h) = self.snap.dims[si];
        let r = BLUR_RADIUS;
        let (rx0, ry0) = (c.x0.saturating_sub(r), c.y0.saturating_sub(r));
        let (rw, rh) = ((c.x1 + r).min(w) - rx0, (c.y1 + r).min(h) - ry0);
        let (hw, hh) = (rw + 2 * r, rh + 2 * r);
        let mut src = vec![0.0f32; hw * hh];
        let mut dst = vec![0.0f32; hw * hh];
        for y in 0..hh {
            let gy = feature_v2::reflect_101(ry0 as isize + y as isize - r as isize, h);
            for x in 0..hw {
                let gx = feature_v2::reflect_101(rx0 as isize + x as isize - r as isize, w);
                src[y * hw + x] = self.snap.pyr_src[si][ch][gy * w + gx];
                dst[y * hw + x] = self.dst_at(si + 1, ch, gy, gx);
            }
        }
        let mut hm1 = vec![0.0f32; hw * hh];
        let mut hm2 = vec![0.0f32; hw * hh];
        let mut hssq = vec![0.0f32; hw * hh];
        let mut hs12 = vec![0.0f32; hw * hh];
        crate::blur::fused_blur_h_ssim_at_revision(
            &src,
            &dst,
            &mut hm1,
            &mut hm2,
            &mut hssq,
            &mut hs12,
            hw,
            hh,
            r,
            self.snap.revision,
        );
        let mut result = Planes {
            mu2: vec![0.0; rw * rh],
            ssq: vec![0.0; rw * rh],
            s12: vec![0.0; rw * rh],
            diff: vec![0; rw * rh],
            rx0,
            ry0,
            rw,
            rh,
        };
        let mut vin = vec![0.0f32; rw * hh];
        let mut vout = vec![0.0f32; rw * hh];
        for (hp, out) in [
            (&hm2, &mut result.mu2),
            (&hssq, &mut result.ssq),
            (&hs12, &mut result.s12),
        ] {
            for y in 0..hh {
                vin[y * rw..(y + 1) * rw].copy_from_slice(&hp[y * hw + r..y * hw + r + rw]);
            }
            crate::blur::box_blur_v_from_copy(&vin, &mut vout, rw, hh, r);
            for y in 0..rh {
                out[y * rw..(y + 1) * rw].copy_from_slice(&vout[(y + r) * rw..(y + r + 1) * rw]);
            }
        }
        let old = &self.snap.planes[si][ch];
        for y in 0..rh {
            for x in 0..rw {
                let o = y * rw + x;
                let i = (ry0 + y) * w + rx0 + x;
                result.diff[o] = (result.mu2[o] != old.mu2[i]
                    || result.ssq[o] != old.ssq[i]
                    || result.s12[o] != old.s12[i]) as u8;
            }
        }
        result
    }

    /// Fold the per-pixel dense-term differences over the residue cone
    /// into `acc` — every accumulator `finish_channel_scale` reads: the
    /// 13 raw pools, the three soft-peak num/den pairs, the shared
    /// mask/IW weight denominators and their four weighted numerators
    /// each. The eval set is `planes.diff ∪ C` — cells where any
    /// recomputed plane differs (plus the changed pixels themselves,
    /// whose `dd` term input differs under a candidate).
    fn dense_delta(&self, si: usize, ch: usize, planes: &Planes, c: &Chg, acc: &mut DenseAccum) {
        let (w, _h) = self.snap.dims[si];
        let p = &self.snap.planes[si][ch];
        let src = &self.snap.pyr_src[si][ch];
        let dst = &self.snap.pyr_dst[si][ch];
        let tb = self.snap.toggles.transducer_bank;
        let rw = planes.rw;
        if rw == 0 || planes.rh == 0 {
            return;
        }
        // Union the changed cells into the eval mask (their `dd` differs;
        // plane inputs may not).
        let mut mask = planes.diff.clone();
        for y in c.y0..c.y1 {
            let o = (y - planes.ry0) * rw + (c.x0 - planes.rx0);
            for m in &mut mask[o..o + (c.x1 - c.x0)] {
                *m = 1;
            }
        }
        for ry in 0..planes.rh {
            for rx in 0..rw {
                if mask[ry * rw + rx] == 0 {
                    continue;
                }
                let y = planes.ry0 + ry;
                let x = planes.rx0 + rx;
                let i = y * w + x;
                let j = ry * rw + rx;
                let d_new = self.dst_at(si + 1, ch, y, x);
                let (to, _, _, _, mwo, iwo, mvo, ivo, kno, kdo) = feature_v2::dense_terms32(
                    src[i], dst[i], p.mu1[i], p.mu2[i], p.ssq[i], p.s12[i], p.act[i], tb, self.err,
                    false,
                );
                let (tn, _, _, _, mwn, iwn, mvn, ivn, knn, kdn) = feature_v2::dense_terms32(
                    src[i],
                    d_new,
                    p.mu1[i],
                    planes.mu2[j],
                    planes.ssq[j],
                    planes.s12[j],
                    p.act[i],
                    tb,
                    self.err,
                    // Refinement refuses revision 6+ (high-pass ringing), so
                    // revision 7's forms never reach it.
                    false,
                );
                acc.sum_d += f64::from(tn[0]) - f64::from(to[0]);
                acc.sum_d2 += f64::from(tn[1]) - f64::from(to[1]);
                acc.sum_d3 += f64::from(tn[2]) - f64::from(to[2]);
                acc.sum_d4 += f64::from(tn[3]) - f64::from(to[3]);
                acc.sum_art += f64::from(tn[4]) - f64::from(to[4]);
                acc.sum_det += f64::from(tn[5]) - f64::from(to[5]);
                acc.sum_mse += f64::from(tn[6]) - f64::from(to[6]);
                acc.sum_hf_gain += f64::from(tn[7]) - f64::from(to[7]);
                acc.sum_hf_loss += f64::from(tn[8]) - f64::from(to[8]);
                acc.sum_hf_mag_loss += f64::from(tn[9]) - f64::from(to[9]);
                acc.sum_pjnd += f64::from(tn[10]) - f64::from(to[10]);
                acc.sum_pjnd_lo += f64::from(tn[11]) - f64::from(to[11]);
                acc.sum_pjnd_hi += f64::from(tn[12]) - f64::from(to[12]);
                acc.ws_peak_ssim.num += f64::from(knn[0]) - f64::from(kno[0]);
                acc.ws_peak_ssim.den += f64::from(kdn[0]) - f64::from(kdo[0]);
                acc.ws_peak_art.num += f64::from(knn[1]) - f64::from(kno[1]);
                acc.ws_peak_art.den += f64::from(kdn[1]) - f64::from(kdo[1]);
                acc.ws_peak_det.num += f64::from(knn[2]) - f64::from(kno[2]);
                acc.ws_peak_det.den += f64::from(kdn[2]) - f64::from(kdo[2]);
                let dm = f64::from(mwn) - f64::from(mwo);
                let di = f64::from(iwn) - f64::from(iwo);
                acc.ws_mask_ssim.num += f64::from(mvn[0]) - f64::from(mvo[0]);
                acc.ws_mask_art.num += f64::from(mvn[1]) - f64::from(mvo[1]);
                acc.ws_mask_det.num += f64::from(mvn[2]) - f64::from(mvo[2]);
                acc.ws_mask_mse.num += f64::from(mvn[3]) - f64::from(mvo[3]);
                acc.ws_iw_ssim.num += f64::from(ivn[0]) - f64::from(ivo[0]);
                acc.ws_iw_art.num += f64::from(ivn[1]) - f64::from(ivo[1]);
                acc.ws_iw_det.num += f64::from(ivn[2]) - f64::from(ivo[2]);
                acc.ws_iw_mse.num += f64::from(ivn[3]) - f64::from(ivo[3]);
                for w in [
                    &mut acc.ws_mask_ssim,
                    &mut acc.ws_mask_art,
                    &mut acc.ws_mask_det,
                    &mut acc.ws_mask_mse,
                ] {
                    w.den += dm;
                }
                for w in [
                    &mut acc.ws_iw_ssim,
                    &mut acc.ws_iw_art,
                    &mut acc.ws_iw_det,
                    &mut acc.ws_iw_mse,
                ] {
                    w.den += di;
                }
            }
        }
    }

    /// Fold the gradient-term differences over `dilate(C,1)` into `acc`:
    /// `sum_grad_src`/`sum_grad_dst` (the edge-width means), `sum_gms`,
    /// `sum_gms2`, `sum_ringing`, `sum_banding`. `src`/`activity` are
    /// reference-only (retained); the dst ±1 halo is rebuilt mixed.
    fn grad_delta(&self, si: usize, ch: usize, c: &Chg, acc: &mut GradientAccum) {
        let (w, h) = self.snap.dims[si];
        let gx0 = c.x0.saturating_sub(1);
        let gx1 = (c.x1 + 1).min(w);
        let gy0 = c.y0.saturating_sub(1);
        let gy1 = (c.y1 + 1).min(h);
        // Halo rows [gy0-1, gy1+1] via reflect_101 — `gradient_terms64`'s
        // `(y+1)`-offset convention with `y` the local core row index.
        let gh = gy1 - gy0 + 2;
        let mut src_g = vec![0.0f32; gh * w];
        let mut dst_old = vec![0.0f32; gh * w];
        let mut dst_new = vec![0.0f32; gh * w];
        let src_plane = &self.snap.pyr_src[si][ch];
        let dst_plane = &self.snap.pyr_dst[si][ch];
        for j in 0..gh {
            let gy = feature_v2::reflect_101(gy0 as isize - 1 + j as isize, h);
            src_g[j * w..j * w + w].copy_from_slice(&src_plane[gy * w..gy * w + w]);
            dst_old[j * w..j * w + w].copy_from_slice(&dst_plane[gy * w..gy * w + w]);
            for x in 0..w {
                dst_new[j * w + x] = self.dst_at(si + 1, ch, gy, x);
            }
        }
        let act = &self.snap.planes[si][ch].act;
        // `RingForm::HighPassMu` reads the local-mean planes, which a change
        // moves over the blur radius, not this ±1 dilation.
        assert!(
            feature_v2::active_ring_form() == feature_v2::RingForm::Raw
                && !self.snap.revision.ring_high_pass(),
            "local refinement supports the raw ringing form only \
             (not revision 6 or ZENSIM_RING_FORM=hpm)"
        );
        for y in gy0..gy1 {
            let ly = y - gy0;
            for x in gx0..gx1 {
                let terms = feature_v2::gradient_terms64::<false, false, false>;
                let o = terms(
                    &src_g,
                    &dst_old,
                    &act[gy0 * w..],
                    &[],
                    &[],
                    &[],
                    w,
                    x,
                    ly,
                    0.0,
                    0.0,
                    false,
                );
                let n = terms(
                    &src_g,
                    &dst_new,
                    &act[gy0 * w..],
                    &[],
                    &[],
                    &[],
                    w,
                    x,
                    ly,
                    0.0,
                    0.0,
                    false,
                );
                acc.sum_grad_src += n.gsrc - o.gsrc;
                acc.sum_grad_dst += n.gdst - o.gdst;
                acc.sum_gms += n.g - o.g;
                acc.sum_gms2 += n.g2 - o.g2;
                acc.sum_ringing += n.ring - o.ring;
                acc.sum_banding += n.band - o.band;
            }
        }
    }

    /// The blockiness delta: `bounded_excess(step_dst, step_src, C_BLOCK)`
    /// re-evaluated at every 8-lattice position whose step pair touches
    /// the changed set. Vertical steps at lattice columns `x = 8k` inside
    /// `[c.x0, c.x1]`; horizontal steps at lattice rows `y = 8k` inside
    /// `[c.y0, c.y1]` — `blockiness_sparse_rows`' exact shape.
    fn blockiness_delta(&self, si: usize, ch: usize, c: &Chg) -> f64 {
        let (w, h) = self.snap.dims[si];
        let src = &self.snap.pyr_src[si][ch];
        let dst = &self.snap.pyr_dst[si][ch];
        let mut d = 0.0f64;
        let lat = BLOCK_LATTICE;
        let mut x = c.x0.div_ceil(lat).max(1) * lat;
        while x < w && x <= c.x1 {
            for y in c.y0..c.y1 {
                let i = y * w + x;
                // Production subtracts in f64 BEFORE widening
                // (`blockiness_sparse_rows`, feature_v2.rs): all three
                // sides must do the same or a no-op candidate emits
                // nonzero deltas.
                let ss = (src[i] as f64 - src[i - 1] as f64).abs();
                let so = (dst[i] as f64 - dst[i - 1] as f64).abs();
                let sn = (self.dst_at(si + 1, ch, y, x) as f64
                    - self.dst_at(si + 1, ch, y, x - 1) as f64)
                    .abs();
                d += feature_v2::bounded_excess(sn, ss, C_BLOCK)
                    - feature_v2::bounded_excess(so, ss, C_BLOCK);
            }
            x += lat;
        }
        let mut y = c.y0.div_ceil(lat).max(1) * lat;
        while y < h && y <= c.y1 {
            for x in c.x0..c.x1 {
                let i = y * w + x;
                let ss = (src[i] as f64 - src[i - w] as f64).abs();
                let so = (dst[i] as f64 - dst[i - w] as f64).abs();
                let sn = (self.dst_at(si + 1, ch, y, x) as f64
                    - self.dst_at(si + 1, ch, y - 1, x) as f64)
                    .abs();
                d += feature_v2::bounded_excess(sn, ss, C_BLOCK)
                    - feature_v2::bounded_excess(so, ss, C_BLOCK);
            }
            y += lat;
        }
        d
    }
}

// ============================================================================
// Golden gate: engine deltas vs FULL recomputation.
//
// Every test that compares two computations holds
// `archmage::testing::lock_token_testing()` (the lane's hard rule — token
// serialisation keeps the SIMD dispatch observations consistent), and the
// Rev3/Rev4 coverage uses the `ssim_form::run_at_revision` child-process
// pattern because `ZENSIM_FORMULA_REV` is process-global.
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_v2::{FoldRetention, V2Scratch};
    use crate::source::RgbSlice;

    #[test]
    fn rev5_exact_refinement_goldens() {
        if !crate::ssim_form::run_at_revision(
            "5",
            "local_refine::tests::rev5_exact_refinement_goldens",
            "REV5_LOCAL_EXACT_OK",
        ) {
            return;
        }
        golden_body();
        {
            let _token = archmage::testing::lock_token_testing();
            cone_body();
        }
        noop_candidate_yields_exact_zero_deltas();
        coarse_y_only_plan_coarse_channels_rebuilt();
        sparse_retention_scale0_channels_rebuilt();
        println!("REV5_LOCAL_EXACT_OK");
    }

    #[test]
    fn rev5_retained_strips_match_snapshot_replay_bits() {
        if !crate::ssim_form::run_at_revision(
            "5",
            "local_refine::tests::rev5_retained_strips_match_snapshot_replay_bits",
            "REV5_RETAINED_STRIPS_OK",
        ) {
            return;
        }
        let _token = archmage::testing::lock_token_testing();
        let plan = full_plan();
        let mut scratch = V2Scratch::new();
        let mut ret = FoldRetention::default();
        // Multiple strips, odd edges, changed dimensions and moved-buffer reuse.
        for (iteration, (w, h)) in [(257, 521), (137, 101), (257, 521)].into_iter().enumerate() {
            let src = textured(w, h, 5 + iteration as u32);
            let dst = distort(&src, w, h);
            let source = RgbSlice::new(&src, w, h);
            let distorted = RgbSlice::new(&dst, w, h);
            crate::feature_v2::compute_folded_v1_372_streaming_impl(
                &source,
                &distorted,
                None,
                iteration != 1,
                &mut scratch,
                Some(&plan),
                Some(&mut ret),
            )
            .expect("retaining production walk");
            let mut replay = LocalRefineSnapshot::capture(&ret, &plan, (w, h), &source, &distorted)
                .expect("copy capture");
            // Reproduce the original capture's independent dense/gradient pass.
            for si in 0..LOCAL_SCALES {
                for ch in 0..3 {
                    if replay.served[si][ch] {
                        for strip in 0..replay.strips[si][ch].len() {
                            replay.strips[si][ch][strip] =
                                replay.strip_accumulators(si, ch, strip, None);
                        }
                    }
                }
            }
            let actual = LocalRefineSnapshot::capture_taking(
                &mut ret,
                &plan,
                (w, h),
                &source,
                &distorted,
                None,
                None,
            )
            .expect("moving capture");
            let y = if h > 280 { 240 } else { 32 };
            for rect in [(0, 0, 8, 8), (16, y, 24, y + 40), (w - 9, h - 9, w, h)] {
                let rw = rect.2 - rect.0;
                let rh = rect.3 - rect.1;
                let planes: [Vec<f32>; 3] =
                    std::array::from_fn(|ch| vec![0.25 + ch as f32 * 0.125; rw * rh]);
                for candidate in [
                    Candidate::Reference,
                    Candidate::Planar(planes.each_ref().map(Vec::as_slice)),
                ] {
                    let bits = |snapshot: &LocalRefineSnapshot| {
                        snapshot
                            .deltas(rect, &candidate)
                            .expect("finite refinement")
                            .into_iter()
                            .map(|(id, value)| (id, value.to_bits()))
                            .collect::<Vec<_>>()
                    };
                    assert_eq!(
                        bits(&actual),
                        bits(&replay),
                        "iteration{iteration} rect{rect:?}"
                    );
                }
            }
        }
        println!("REV5_RETAINED_STRIPS_OK");
    }

    #[test]
    fn prepared_scale0_distorted_planes_match_reconversion_bits() {
        if !crate::ssim_form::run_at_revision(
            "5",
            "local_refine::tests::prepared_scale0_distorted_planes_match_reconversion_bits",
            "PREPARED_RAW0_OK",
        ) {
            return;
        }
        let _token = archmage::testing::lock_token_testing();
        let ids = (13..26).chain(39..156).chain(401..430).chain(459..720);
        let plan = crate::feature_plan::Plan::derive(
            &crate::feature_set_id::SlotSet::from_slots(ids),
            720,
        )
        .expect("sparse prepared plan");
        assert!(!plan.compute.full_res_xb);
        let mut scratch = V2Scratch::new();
        let mut ret = FoldRetention::default();
        for (w, h) in [(64, 64), (257, 521), (137, 101), (257, 521)] {
            let src = textured(w, h, 41);
            let dst = distort(&src, w, h);
            let source = RgbSlice::new(&src, w, h);
            let distorted = RgbSlice::new(&dst, w, h);
            let reference = crate::PrecomputedReference::for_candidate(
                &source,
                false,
                None,
                FormulaRevision::Rev5,
            );
            let prepared = crate::feature_v2::compute_folded_with_ref_retained(
                &source,
                &reference,
                &distorted,
                false,
                &mut scratch,
                &plan,
                Some(&mut ret),
            );
            assert!(ret.complete_distorted0);
            let actual = LocalRefineSnapshot::capture_with_encoding(
                &ret,
                &plan,
                (w, h),
                &source,
                &distorted,
                None,
            )
            .expect("raw retained capture");
            ret.complete_distorted0 = false;
            let expected = LocalRefineSnapshot::capture_with_encoding(
                &ret,
                &plan,
                (w, h),
                &source,
                &distorted,
                None,
            )
            .expect("original reconverting capture");
            for ch in 0..3 {
                assert_eq!(
                    actual.pyr_dst0[ch]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    expected.pyr_dst0[ch]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    "{w}x{h} channel{ch}",
                );
            }
            for rect in [(0, 0, 8, 8), (16, 16, 24, 24), (w - 9, h - 9, w, h)] {
                let bits = |snapshot: &LocalRefineSnapshot| {
                    snapshot
                        .deltas(rect, &Candidate::Reference)
                        .expect("finite repair")
                        .into_iter()
                        .map(|(id, v)| (id, v.to_bits()))
                        .collect::<Vec<_>>()
                };
                assert_eq!(bits(&actual), bits(&expected), "{w}x{h} {rect:?}");
            }
            // The ordinary sparse walk still uses the original omitted-plane path.
            let ordinary = crate::feature_v2::compute_folded_v1_372_streaming_impl(
                &source,
                &distorted,
                None,
                false,
                &mut scratch,
                Some(&plan),
                Some(&mut ret),
            )
            .expect("ordinary sparse walk");
            assert_eq!(
                prepared.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                ordinary.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "prepared features must preserve the ordinary sparse walk",
            );
            assert_eq!(prepared.1.map(f64::to_bits), ordinary.1.map(f64::to_bits));
            assert!(!ret.complete_distorted0);
            assert!(ret.pyr_dst[0][0].is_empty() && ret.pyr_dst[0][2].is_empty());
        }
        println!("PREPARED_RAW0_OK");
    }

    #[test]
    fn retired_snapshot_buffers_reuse_without_changing_live_results() {
        if !crate::ssim_form::run_at_revision(
            "5",
            "local_refine::tests::retired_snapshot_buffers_reuse_without_changing_live_results",
            "SNAPSHOT_REUSE_OK",
        ) {
            return;
        }
        let _token = archmage::testing::lock_token_testing();
        let plan = crate::feature_plan::Plan::derive(
            &crate::feature_set_id::SlotSet::from_slots(
                (13..26).chain(39..156).chain(401..430).chain(459..720),
            ),
            720,
        )
        .expect("sparse prepared plan");
        fn take_pair(
            src: &[[u8; 3]],
            dst: &[[u8; 3]],
            dims: (usize, usize),
            plan: &crate::feature_plan::Plan,
            scratch: &mut V2Scratch,
            ret: &mut FoldRetention,
        ) -> (LocalRefineSnapshot, LocalRefineSnapshot) {
            let (w, h) = dims;
            let source = RgbSlice::new(src, w, h);
            let distorted = RgbSlice::new(dst, w, h);
            let reference = crate::PrecomputedReference::for_candidate(
                &source,
                false,
                None,
                FormulaRevision::Rev5,
            );
            let prepared = crate::feature_v2::compute_folded_with_ref_retained(
                &source,
                &reference,
                &distorted,
                false,
                scratch,
                plan,
                Some(ret),
            );
            let actual = LocalRefineSnapshot::capture_taking(
                ret,
                plan,
                dims,
                &source,
                &distorted,
                None,
                Some(&reference.scales[0].0),
            )
            .expect("moving prepared capture");
            // A fresh ordinary retention has no recycle slot or prepared raw0 path.
            let mut fresh = FoldRetention::default();
            let ordinary = crate::feature_v2::compute_folded_v1_372_streaming_impl(
                &source,
                &distorted,
                None,
                false,
                &mut V2Scratch::new(),
                Some(plan),
                Some(&mut fresh),
            )
            .expect("fresh ordinary walk");
            assert_eq!(
                prepared.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                ordinary.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            );
            let expected = LocalRefineSnapshot::capture(&fresh, plan, dims, &source, &distorted)
                .expect("independent ordinary capture");
            (actual, expected)
        }
        let bits = |snapshot: &LocalRefineSnapshot| {
            [(0, 0, 8, 8), (16, 16, 24, 24), (31, 40, 47, 56)]
                .into_iter()
                .flat_map(|rect| {
                    snapshot
                        .deltas(rect, &Candidate::Reference)
                        .expect("finite repair")
                        .into_iter()
                        .map(|(id, v)| (id, v.to_bits()))
                })
                .collect::<Vec<_>>()
        };
        let dims = (257, 521);
        let src = textured(dims.0, dims.1, 91);
        let dst = distort(&src, dims.0, dims.1);
        let mut scratch = V2Scratch::new();
        let mut ret = FoldRetention::default();
        let (a, a_expected) = take_pair(&src, &dst, dims, &plan, &mut scratch, &mut ret);
        let a_bits = bits(&a_expected);
        assert_eq!(bits(&a), a_bits);
        let mut changed = dst.clone();
        for px in &mut changed {
            px[0] ^= 17;
        }
        let (b, b_expected) = take_pair(&src, &changed, dims, &plan, &mut scratch, &mut ret);
        assert_eq!(bits(&b), bits(&b_expected));
        let address = b.pyr_dst0[1].as_ptr() as usize;
        let source_address = b.pyr_src0[0].as_ptr() as usize;
        assert!(
            ret.snapshot_reuse
                .as_ref()
                .expect("reuse slot")
                .lock()
                .expect("lock")
                .is_none()
        );
        // Returning from another thread must not mutate the still-live A result.
        std::thread::spawn(move || drop(b))
            .join()
            .expect("drop thread");
        assert!(
            ret.snapshot_reuse
                .as_ref()
                .expect("reuse slot")
                .lock()
                .expect("lock")
                .is_some()
        );
        let mut changed_source = src.clone();
        for px in &mut changed_source {
            px[2] ^= 37;
        }
        let (c, c_expected) = take_pair(&changed_source, &dst, dims, &plan, &mut scratch, &mut ret);
        assert_eq!(
            c.pyr_dst0[1].as_ptr() as usize,
            address,
            "reuse the retired allocation"
        );
        assert_eq!(
            c.pyr_src0[0].as_ptr() as usize,
            source_address,
            "reuse source storage while copying the current reference"
        );
        assert_eq!(bits(&c), bits(&c_expected));
        assert_eq!(bits(&a), a_bits, "live results must not be recycled");
        drop(a);
        let small = (137, 101);
        let other_src = textured(small.0, small.1, 117);
        let other_dst = distort(&other_src, small.0, small.1);
        let (d, d_expected) =
            take_pair(&other_src, &other_dst, small, &plan, &mut scratch, &mut ret);
        assert_eq!(
            bits(&d),
            bits(&d_expected),
            "resize and overwrite retired storage"
        );
        assert_eq!(
            bits(&c),
            bits(&c_expected),
            "older live geometry remains valid"
        );
        drop(d);
        drop(c);
        let slot = ret
            .snapshot_reuse
            .as_ref()
            .expect("reuse slot")
            .lock()
            .expect("lock");
        assert_eq!(
            slot.as_ref().expect("one retired result").pyr_dst0[1].len(),
            small.0 * small.1,
            "a later drop must not accumulate a second retired buffer set"
        );
        println!("SNAPSHOT_REUSE_OK");
    }

    /// `feature_v2::tests::textured_image`'s fixture family, duplicated
    /// here so the test does not reach across modules: gradients + edges +
    /// LCG noise so every feature family sees real signal.
    fn textured(w: usize, h: usize, seed: u32) -> Vec<[u8; 3]> {
        let mut state = seed | 1;
        let mut px = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let noise = (state >> 24) as u8;
                let grad = ((x * 255) / w.max(1)) as u8;
                let edge = if (x / 9 + y / 7) % 2 == 0 { 200 } else { 40 };
                px.push([
                    grad.wrapping_add(noise / 4),
                    edge,
                    (((y * 255) / h.max(1)) as u8) ^ (noise / 8),
                ]);
            }
        }
        px
    }

    /// Codec-flavored quantize distortion (the sibling test's shape).
    fn distort(src: &[[u8; 3]], w: usize, h: usize) -> Vec<[u8; 3]> {
        let mut out = src.to_vec();
        for y in 0..h {
            for x in 0..w {
                let p = &mut out[y * w + x];
                for c in p.iter_mut() {
                    *c = (*c / 24) * 24 + ((x % 8 == 0 || y % 8 == 0) as u8) * 6;
                }
            }
        }
        out
    }

    /// A REAL zenjpeg 4:4:4 round-trip — the same crate + path
    /// `gen_jpeg_distortion --subsampling 444 --decoded-out` drives
    /// (`EncoderConfig::ycbcr(q, ChromaSubsampling::None)` → `Decoder`).
    fn jpeg444(px: &[[u8; 3]], w: usize, h: usize, q: u8) -> Vec<[u8; 3]> {
        use enough::Unstoppable;
        use zenjpeg::decoder::Decoder;
        use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
        let flat: Vec<u8> = px.iter().flat_map(|p| p.iter().copied()).collect();
        let config = EncoderConfig::ycbcr(q, ChromaSubsampling::None);
        let mut enc = config
            .encode_from_bytes(w as u32, h as u32, PixelLayout::Rgb8Srgb)
            .expect("zenjpeg encoder init");
        enc.push_packed(&flat, Unstoppable).expect("zenjpeg push");
        let bytes = enc.finish().expect("zenjpeg finish");
        let dec = Decoder::new()
            .decode(&bytes, Unstoppable)
            .expect("zenjpeg decode");
        dec.pixels_u8()
            .expect("u8 jpeg output")
            .as_chunks::<3>()
            .0
            .iter()
            .map(|c| [c[0], c[1], c[2]])
            .collect()
    }

    /// A real-image pair: `v1_golden_real_ref.png` (96×96 real content)
    /// cropped to non-multiple-of-8 dims, JPEG q10 4:4:4 as the distorted.
    fn real_jpeg_pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
        let img =
            image::load_from_memory(include_bytes!("../tests/fixtures/v1_golden_real_ref.png"))
                .expect("fixture decodes")
                .to_rgb8();
        let (fw, fh) = (img.width() as usize, img.height() as usize);
        assert!(w <= fw && h <= fh, "crop {w}x{h} exceeds fixture {fw}x{fh}");
        let mut src = vec![[0u8; 3]; w * h];
        for y in 0..h {
            for x in 0..w {
                let p = img.get_pixel(x as u32, y as u32);
                src[y * w + x] = [p[0], p[1], p[2]];
            }
        }
        let dst = jpeg444(&src, w, h, 10);
        (src, dst)
    }

    /// The full fold walk, with retention, at the process's active revision.
    fn walk(
        src: &[[u8; 3]],
        dst: &[[u8; 3]],
        w: usize,
        h: usize,
    ) -> (crate::feature_v2::ZensimV2Result, FoldRetention) {
        let source = RgbSlice::new(src, w, h);
        let distorted = RgbSlice::new(dst, w, h);
        let mut scratch = V2Scratch::new();
        let mut retention = FoldRetention::default();
        let res = if crate::ssim_form::active_revision() >= FormulaRevision::Rev5 {
            // Wider storage is valid when a plan explicitly requests only
            // the supported 576 slots. The raw 944 request must refuse.
            let plan = full_plan();
            crate::feature_v2::compute_folded720_streaming_extras(
                &source,
                &distorted,
                None,
                false,
                plan.toggles(),
                &mut scratch,
                crate::feature_v2::FoldWalkExtras {
                    compute: Some(plan.compute),
                    retention: Some(&mut retention),
                    ..Default::default()
                },
            )
        } else {
            crate::feature_v2::compute_folded944_streaming_with_retention(
                &source,
                &distorted,
                None,
                false,
                &mut scratch,
                &mut retention,
            )
        }
        .expect("fold walk computes the requested supported slots");
        (res, retention)
    }

    /// The full-944 plan every snapshot in this module is captured against.
    fn full_plan() -> crate::feature_plan::Plan {
        let ids = if crate::ssim_form::active_revision() >= FormulaRevision::Rev5 {
            crate::feature_set_id::SlotSet::from_slots((0..228).chain(372..720))
        } else {
            crate::feature_set_id::SlotSet::from_slots(0..944)
        };
        crate::feature_plan::Plan::derive(&ids, 944).expect("full plan derives")
    }

    /// Golden tolerance, per feature id: relative `1e-6` on the delta
    /// plus an absolute floor of `4·f32::EPSILON·max(1, |feature|)` — four
    /// ulps of an f32 at the feature's own magnitude. Justification: the
    /// served features finish in f64 over f32 phase-A planes, and the
    /// engine-vs-full error is pool-order noise — f64 reassociation of
    /// the same per-cell sums (each `~1e-13` relative at these
    /// magnitudes) plus a single f32→f64 rounding of near-identical
    /// pooled values — so a few f32 ulps of the feature magnitude bounds
    /// the worst honest divergence with room to spare, while still
    /// catching any real term/plane defect (which lands at ~1e-3+ of the
    /// delta or wildly more). The floor bottoms out at ~4.8e-7 for
    /// sub-unit features.
    fn delta_tol(base_f: f64, full_f: f64, full_d: f64) -> f64 {
        let mag = base_f.abs().max(full_f.abs()).max(1.0);
        4.0 * f64::from(f32::EPSILON) * mag + 1e-6 * full_d.abs()
    }

    /// The golden check: `deltas` must match the full walk's feature
    /// differences on the intervened image within [`delta_tol`]
    /// (relative 1e-6 + an f32-epsilon floor) for every v2 id at
    /// scales 1–3.
    fn check_pair(
        name: &str,
        src: &[[u8; 3]],
        dst: &[[u8; 3]],
        alt_rect: Option<&[[u8; 3]]>,
        w: usize,
        h: usize,
        rects: &[(usize, usize, usize, usize)],
    ) {
        let (base, retention) = walk(src, dst, w, h);
        let plan = full_plan();
        let snap = LocalRefineSnapshot::capture(
            &retention,
            &plan,
            (w, h),
            &RgbSlice::new(src, w, h),
            &RgbSlice::new(dst, w, h),
        )
        .unwrap_or_else(|| panic!("{name}: capture must accept a full v2 plan"));
        for &(x0, y0, x1, y1) in rects {
            // The intervened image: dst with the rect's pixels replaced by
            // the candidate (reference, or the alt image's pixels there).
            let cx0 = x0.min(w);
            let cy0 = y0.min(h);
            let cx1 = x1.min(w).max(cx0);
            let cy1 = y1.min(h).max(cy0);
            let mut intervened = dst.to_vec();
            for y in cy0..cy1 {
                for x in cx0..cx1 {
                    intervened[y * w + x] = match alt_rect {
                        Some(alt) => alt[y * w + x],
                        None => src[y * w + x],
                    };
                }
            }
            let (full, ret2) = walk(src, &intervened, w, h);
            let candidate = match alt_rect {
                Some(_) => {
                    // The engine prices values in positive-XYB: the new
                    // walk's retained scale-0 dst, cropped to the rect.
                    let (rw, rh) = (cx1 - cx0, cy1 - cy0);
                    let mut planes: [Vec<f32>; 3] =
                        [vec![0.0; rw * rh], vec![0.0; rw * rh], vec![0.0; rw * rh]];
                    for (ch, plane) in planes.iter_mut().enumerate() {
                        for y in 0..rh {
                            for x in 0..rw {
                                plane[y * rw + x] = ret2.pyr_dst[0][ch][(cy0 + y) * w + cx0 + x];
                            }
                        }
                    }
                    Candidate::Planar([
                        Box::leak(planes[0].clone().into_boxed_slice()),
                        Box::leak(planes[1].clone().into_boxed_slice()),
                        Box::leak(planes[2].clone().into_boxed_slice()),
                    ])
                }
                None => Candidate::Reference,
            };
            let deltas = snap
                .deltas((x0, y0, x1, y1), &candidate)
                .unwrap_or_else(|| panic!("{name}: deltas must serve"));
            assert_eq!(
                deltas.len(),
                3 * 3 * 29,
                "{name}: one delta per v2 pooled feature at scales 1-3"
            );
            let mut worst = (0.0f64, 0usize, 0.0f64, 0.0f64);
            let mut bad = 0usize;
            for (id, d) in &deltas {
                let full_d = full.features()[*id] - base.features()[*id];
                let tol = if crate::ssim_form::active_revision() >= FormulaRevision::Rev5 {
                    1e-10 + full_d.abs() * 1e-6
                } else {
                    delta_tol(base.features()[*id], full.features()[*id], full_d)
                };
                let err = (d - full_d).abs();
                if err > tol {
                    bad += 1;
                    if err > worst.0 {
                        worst = (err, *id, *d, full_d);
                    }
                }
            }
            if bad > 0 {
                // The full mismatch list — the first ten — before failing.
                for (id, d) in deltas
                    .iter()
                    .filter(|(id, d)| {
                        let fd = full.features()[*id] - base.features()[*id];
                        (d - fd).abs() > delta_tol(base.features()[*id], full.features()[*id], fd)
                    })
                    .take(10)
                {
                    let full_d = full.features()[*id] - base.features()[*id];
                    eprintln!(
                        "  {name} rect ({x0},{y0},{x1},{y1}) f{id}: local {d:e} vs full {full_d:e}"
                    );
                }
                panic!(
                    "{name} rect ({x0},{y0},{x1},{y1}): {bad} ids over tolerance; \
                     worst f{}: local {:e} vs full {:e}",
                    worst.1, worst.2, worst.3
                );
            }
        }
    }

    fn golden_body() {
        let _token = archmage::testing::lock_token_testing();
        // Case 1: synthetic textured, dims not multiples of 8/16.
        let (w, h) = (137usize, 101usize);
        let src = textured(w, h, 0xBEEF);
        let dst = distort(&src, w, h);
        check_pair(
            "textured-137x101",
            &src,
            &dst,
            None,
            w,
            h,
            &[
                (0, 0, 1, 1),                   // 1x1 at the corner
                (0, 0, 8, 8),                   // 8x8 at the edge
                (16, 16, 24, 24),               // 8x8 aligned
                (3, 5, 26, 22),                 // 23x17 unaligned
                (32, 8, 48, 24),                // 16x16
                (w - 9, h - 9, w, h),           // unaligned far edge
                (w - 4, h - 4, w + 12, h + 12), // clipped past the edge
                (48, 40, 80, 72),               // 32x32
            ],
        );

        // Case 2: real 4:4:4 zenjpeg q10 pair, non-multiple dims.
        let (w, h) = (91usize, 87usize);
        let (src, dst) = real_jpeg_pair(w, h);
        check_pair(
            "jpeg-q10-91x87",
            &src,
            &dst,
            None,
            w,
            h,
            &[
                (0, 0, 8, 8),
                (16, 16, 24, 24),
                (3, 5, 26, 22),
                (w - 9, h - 9, w, h),
                (48, 40, 80, 72),
            ],
        );

        // Case 3: candidate pixels that are NOT the reference — a q30
        // 4:4:4 decode of the same source pasted inside the rect.
        let (w, h) = (121usize, 93usize);
        let src = textured(w, h, 0xCAFE);
        let dst = distort(&src, w, h);
        let alt = jpeg444(&src, w, h, 30);
        check_pair(
            "jpeg-q30-candidate-121x93",
            &src,
            &dst,
            Some(&alt),
            w,
            h,
            &[
                (8, 8, 16, 16),
                (0, 0, 8, 8),
                (3, 5, 26, 22),
                (60, 30, 92, 62),
            ],
        );

        // Case 4: multi-strip — 301 tall puts scale-1 at 150 rows (two
        // 128-row strips). The 240..280 rect's scale-1 footprint straddles
        // the strip boundary at row 128, where each strip's sliding-sum
        // residue ends and the next strip re-initialises.
        let (w, h) = (141usize, 301usize);
        let src = textured(w, h, 0xF00D);
        let dst = distort(&src, w, h);
        check_pair(
            "textured-141x301-multistrip",
            &src,
            &dst,
            None,
            w,
            h,
            &[
                (0, 240, 16, 280),  // straddles the s1 strip boundary
                (0, 0, 8, 8),       // far above the boundary
                (40, 296, 56, 301), // bottom edge, last strip only
            ],
        );
    }

    /// Rev3: the shipped recurrence path (f32 blur, no canon mode).
    #[test]
    fn golden_reference_and_candidate_match_full_recompute_rev3() {
        if !crate::ssim_form::run_at_revision(
            "3",
            "local_refine::tests::golden_reference_and_candidate_match_full_recompute_rev3",
            "NEIGHSTEER-GOLDEN-REV3-RAN",
        ) {
            return;
        }
        golden_body();
        println!("NEIGHSTEER-GOLDEN-REV3-RAN");
    }

    /// Rev4: Canon64 + Rec64 blur arithmetic.
    #[test]
    fn golden_reference_and_candidate_match_full_recompute_rev4() {
        if !crate::ssim_form::run_at_revision(
            "4",
            "local_refine::tests::golden_reference_and_candidate_match_full_recompute_rev4",
            "NEIGHSTEER-GOLDEN-REV4-RAN",
        ) {
            return;
        }
        golden_body();
        println!("NEIGHSTEER-GOLDEN-REV4-RAN");
    }

    /// Refusals: never silently zero — sampling plans, v2-off plans,
    /// reflect-padded pairs, out-of-range scales, malformed candidates.
    #[test]
    fn refusals_return_none_not_zeros() {
        let _token = archmage::testing::lock_token_testing();
        let (w, h) = (96usize, 96usize);
        let src = textured(w, h, 7);
        let dst = distort(&src, w, h);
        let (_base, retention) = walk(&src, &dst, w, h);
        let plan = full_plan();
        let snap = LocalRefineSnapshot::capture(
            &retention,
            &plan,
            (w, h),
            &RgbSlice::new(&src, w, h),
            &RgbSlice::new(&dst, w, h),
        )
        .unwrap();
        // Scale range outside 1..=3.
        assert!(
            snap.deltas_range((0, 0, 8, 8), &Candidate::Reference, 0..=2)
                .is_none()
        );
        assert!(
            snap.deltas_range((0, 0, 8, 8), &Candidate::Reference, 4..=4)
                .is_none()
        );
        // Malformed candidate planes (wrong length).
        let bad = [vec![0.0f32; 8], vec![0.0; 64], vec![0.0; 64]];
        assert!(
            snap.deltas(
                (0, 0, 8, 8),
                &Candidate::Planar([&bad[0], &bad[1], &bad[2]])
            )
            .is_none()
        );
        // A v1-only plan never captures (v2_blocks off).
        let v1_only = crate::feature_plan::Plan::derive(
            &crate::feature_set_id::SlotSet::from_slots(0..156),
            944,
        )
        .unwrap();
        assert!(
            LocalRefineSnapshot::capture(
                &retention,
                &v1_only,
                (w, h),
                &RgbSlice::new(&src, w, h),
                &RgbSlice::new(&dst, w, h),
            )
            .is_none()
        );
        // Reflect-padded pairs: the retained scale-0 dims exceed the
        // image, so a foreign `src_dims` — or a genuinely padded walk —
        // refuses.
        let (sw, sh) = (40usize, 32usize);
        let ssrc = textured(sw, sh, 3);
        let sdst = distort(&ssrc, sw, sh);
        let (_pbase, pret) = walk(&ssrc, &sdst, sw, sh);
        assert!(
            pret.dims[0] != (sw, sh),
            "small pairs must run the reflect-pad path"
        );
        assert!(
            LocalRefineSnapshot::capture(
                &pret,
                &plan,
                (sw, sh),
                &RgbSlice::new(&ssrc, sw, sh),
                &RgbSlice::new(&sdst, sw, sh),
            )
            .is_none()
        );
        // Foreign dims on a normal pair.
        assert!(
            LocalRefineSnapshot::capture(
                &retention,
                &plan,
                (w + 2, h),
                &RgbSlice::new(&src, w, h),
                &RgbSlice::new(&dst, w, h),
            )
            .is_none()
        );
    }

    /// Empty and fully-covered rects: empty is an exact zero (not a
    /// refusal); a whole-image rect changes every coarse pixel.
    #[test]
    fn empty_and_full_rects() {
        let _token = archmage::testing::lock_token_testing();
        let (w, h) = (97usize, 83usize);
        let src = textured(w, h, 11);
        let dst = distort(&src, w, h);
        let (_base, retention) = walk(&src, &dst, w, h);
        let plan = full_plan();
        let snap = LocalRefineSnapshot::capture(
            &retention,
            &plan,
            (w, h),
            &RgbSlice::new(&src, w, h),
            &RgbSlice::new(&dst, w, h),
        )
        .unwrap();
        let empty = snap
            .deltas((20, 20, 20, 24), &Candidate::Reference)
            .unwrap();
        assert!(
            empty.iter().all(|(_, d)| *d == 0.0),
            "empty rect must be an exact no-op"
        );
        // Whole-image: the changed set is the entire coarse planes; deltas
        // must still match full recomputation (the src==dst limit is
        // degenerate, so instead verify finiteness + nonzero where a real
        // distortion exists).
        let whole = snap.deltas((0, 0, w, h), &Candidate::Reference).unwrap();
        assert!(whole.iter().all(|(_, d)| d.is_finite()));
        assert!(
            whole.iter().any(|(_, d)| d.abs() > 1e-9),
            "a whole-image reference repair must move real features"
        );
    }

    /// Plane-level invariant (the golden gate's microscope): for one
    /// (pair, rect) case the recomputed `mu2`/`ssq`/`s12` planes must be
    /// **bit-identical** to the intervened walk's own retained planes at
    /// every residue-cone pixel, and no plane difference may exist
    /// outside the cone — the sliding-sum recurrences' downstream
    /// propagation is exactly what the cone models. Cell deltas then
    /// carry only lane-pool-order noise.
    fn cone_case(
        name: &str,
        src: &[[u8; 3]],
        dst: &[[u8; 3]],
        w: usize,
        h: usize,
        rect: (usize, usize, usize, usize),
    ) {
        let mut intervened = dst.to_vec();
        for y in rect.1..rect.3 {
            for x in rect.0..rect.2 {
                intervened[y * w + x] = src[y * w + x];
            }
        }
        let (_base, ret) = walk(src, dst, w, h);
        let (_full, ret2) = walk(src, &intervened, w, h);
        let plan = full_plan();
        let snap = LocalRefineSnapshot::capture(
            &ret,
            &plan,
            (w, h),
            &RgbSlice::new(src, w, h),
            &RgbSlice::new(dst, w, h),
        )
        .unwrap_or_else(|| panic!("{name}: capture must accept a full v2 plan"));

        let mut q = Query {
            snap: &snap,
            rect,
            cand: None,
            changed: [None; LOCAL_SCALES],
            new_dst: std::array::from_fn(|_| std::array::from_fn(|_| Vec::new())),
            err: false,
        };
        let revision = crate::ssim_form::effective_revision(snap.revision);
        q.err = revision >= FormulaRevision::Rev3;
        for (si, chg) in q.changed.iter_mut().enumerate() {
            let (ww, hh) = snap.dims[si];
            *chg = scale_rect(rect.0, rect.1, rect.2, rect.3, si + 1, ww, hh);
        }
        q.build_changed();

        for si in 0..LOCAL_SCALES {
            for ch in 0..3 {
                let Some(c) = q.changed[si] else { continue };
                let planes = q.recompute_planes(si, ch, &c);
                let (sw, sh) = snap.dims[si];
                let mut outside = 0usize;
                let mut unmasked = 0usize;
                for y in 0..sh {
                    for x in 0..sw {
                        let i = y * sw + x;
                        let in_cone = x >= planes.rx0
                            && x < planes.rx0 + planes.rw
                            && y >= planes.ry0
                            && y < planes.ry0 + planes.rh;
                        for (a, b, maskable) in [
                            (
                                ret.planes[si + 1][ch].mu2[i],
                                ret2.planes[si + 1][ch].mu2[i],
                                true,
                            ),
                            (
                                ret.planes[si + 1][ch].ssq[i],
                                ret2.planes[si + 1][ch].ssq[i],
                                true,
                            ),
                            (
                                ret.planes[si + 1][ch].s12[i],
                                ret2.planes[si + 1][ch].s12[i],
                                true,
                            ),
                            (
                                ret.planes[si + 1][ch].mu1[i],
                                ret2.planes[si + 1][ch].mu1[i],
                                false,
                            ),
                            (
                                ret.planes[si + 1][ch].act[i],
                                ret2.planes[si + 1][ch].act[i],
                                false,
                            ),
                        ] {
                            if a != b {
                                if !in_cone {
                                    outside += 1;
                                } else if maskable {
                                    let j = (y - planes.ry0) * planes.rw + (x - planes.rx0);
                                    if planes.diff[j] == 0 {
                                        unmasked += 1;
                                    }
                                }
                            }
                        }
                        if in_cone {
                            let j = (y - planes.ry0) * planes.rw + (x - planes.rx0);
                            for (mine, truth) in [
                                (planes.mu2[j], ret2.planes[si + 1][ch].mu2[i]),
                                (planes.ssq[j], ret2.planes[si + 1][ch].ssq[i]),
                                (planes.s12[j], ret2.planes[si + 1][ch].s12[i]),
                            ] {
                                assert_eq!(
                                    mine,
                                    truth,
                                    "{name} s{}c{} plane at ({x},{y}): recomputed {mine:e} != walk {truth:e}",
                                    si + 1,
                                    ch
                                );
                            }
                        }
                    }
                }
                assert_eq!(
                    outside, 0,
                    "{name} s{si}c{ch}: plane diffs leaked outside the residue cone"
                );
                assert_eq!(
                    unmasked, 0,
                    "{name} s{si}c{ch}: real plane diffs not marked in the eval mask"
                );
            }
        }
    }

    /// The shared cone body — run by every revision/tier control below.
    /// Aligned, unaligned and edge-touching rects on a real JPEG pair,
    /// plus a multi-strip image (301 rows → scale-1 spans the 128-row
    /// strip boundary) whose rects straddle that boundary. The caller
    /// holds `lock_token_testing` where token state matters (the scalar
    /// control); the bodies themselves do not mutate tokens.
    fn cone_body() {
        let (w, h) = (91usize, 87usize);
        let (src, dst) = real_jpeg_pair(w, h);
        for rect in [
            (0usize, 0usize, 8usize, 8usize),
            (3, 5, 26, 22),
            (w - 9, h - 9, w, h),
        ] {
            cone_case("jpeg-91x87", &src, &dst, w, h, rect);
        }
        let (w, h) = (141usize, 301usize);
        let src = textured(w, h, 0xF00D);
        let dst = distort(&src, w, h);
        for rect in [
            (0usize, 240usize, 16usize, 280usize), // straddles the s1 strip boundary
            (40, 296, 56, 301),                    // last strip, edge-touching
            (3, 5, 26, 22),                        // unaligned, first strip
        ] {
            cone_case("textured-141x301", &src, &dst, w, h, rect);
        }
    }

    /// Rev3: the shipped recurrence path — f32 blur, column tiling on.
    #[test]
    fn cone_planes_bit_exact_and_complete_rev3() {
        if !crate::ssim_form::run_at_revision(
            "3",
            "local_refine::tests::cone_planes_bit_exact_and_complete_rev3",
            "NEIGHSTEER-CONE-REV3-RAN",
        ) {
            return;
        }
        let _token = archmage::testing::lock_token_testing();
        cone_body();
        println!("NEIGHSTEER-CONE-REV3-RAN");
    }

    /// Rev4: canon Rec64/Canon64 blur arithmetic through the same cone.
    #[test]
    fn cone_planes_bit_exact_and_complete_rev4() {
        if !crate::ssim_form::run_at_revision(
            "4",
            "local_refine::tests::cone_planes_bit_exact_and_complete_rev4",
            "NEIGHSTEER-CONE-REV4-RAN",
        ) {
            return;
        }
        let _token = archmage::testing::lock_token_testing();
        cone_body();
        println!("NEIGHSTEER-CONE-REV4-RAN");
    }

    /// The same bit-exact cone at the FORCED SCALAR dispatch tier, at both
    /// revisions: the pinned magetypes scalar `mul_add` is an unfused
    /// `a*b + c` — the i686/wasm128 arithmetic class — so this is the
    /// stand-in for those targets on x86 CI. x86-only because the token
    /// disable is an x86 concept; aarch64's equivalent is neon-only.
    /// The production blur calls inside `recompute_planes` are exactly
    /// what makes this possible: the engine cannot drift from the tier's
    /// arithmetic because it has none of its own.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn cone_planes_bit_exact_scalar_tier() {
        const PATH: &str = "local_refine::tests::cone_planes_bit_exact_scalar_tier";
        const SENT: &str = "NEIGHSTEER-CONE-SCALAR-RAN";
        // Disables SIMD tiers: always alone in its own process, once per revision,
        // even when this process is already at Rev3/Rev4 (e.g. a revision child
        // running many tests), so no other test in this binary sees the change.
        if !crate::ssim_form::run_in_own_process(PATH, SENT, Some("3")) {
            assert!(!crate::ssim_form::run_in_own_process(PATH, SENT, Some("4")));
            return;
        }
        let _token = archmage::testing::lock_token_testing();
        /// Disabling X64V2+ forces every dispatch to the scalar
        /// implementation. `Drop` restores the process-wide tokens even
        /// if the body panics, so the token leak can't poison siblings.
        struct Restore;
        impl Drop for Restore {
            fn drop(&mut self) {
                let _ = archmage::X64V2Token::dangerously_disable_token_process_wide(false);
            }
        }
        archmage::X64V2Token::dangerously_disable_token_process_wide(true)
            .expect("token disable under the testing lock");
        let _restore = Restore;
        cone_body();
        println!("{SENT}");
    }

    /// Wide Rev3: source width over 2048 puts scale-1 above
    /// `blur::H_TILE_WIDTH` (1024), so the production H pass column-tiles
    /// and every tile edge's restarted running sum must replay exactly —
    /// the defect the fix routes through production's own kernel for.
    /// Rects straddle the tile edge (scale-1 x = 1024 ↔ source x = 2048).
    #[test]
    fn wide_rev3_scale1_over_h_tile() {
        if !crate::ssim_form::run_at_revision(
            "3",
            "local_refine::tests::wide_rev3_scale1_over_h_tile",
            "NEIGHSTEER-WIDE-REV3-RAN",
        ) {
            return;
        }
        let _token = archmage::testing::lock_token_testing();
        let (w, h) = (2060usize, 98usize);
        assert!(
            w / 2 > crate::blur::H_TILE_WIDTH,
            "scale-1 width {} must exceed the H tile {}",
            w / 2,
            crate::blur::H_TILE_WIDTH
        );
        let src = textured(w, h, 0xBEE);
        let dst = distort(&src, w, h);
        check_pair(
            "wide-2060x98",
            &src,
            &dst,
            None,
            w,
            h,
            &[
                (2040, 10, 2060, 40), // s1 footprint crosses the x=1024 tile edge
                (1997, 7, 2055, 47),  // unaligned, crosses it too
                (0, 0, 8, 8),         // far left of the tile
                (w - 9, h - 9, w, h), // unaligned far edge
            ],
        );
        println!("NEIGHSTEER-WIDE-REV3-RAN");
    }

    /// `coarse_y_only_scales` plans skip X/B retention at the masked
    /// scales; the cascade still consumed real pixels there. `capture`
    /// must rebuild every unretained coarse channel through the
    /// production downscale chain — including when a REUSED retention
    /// buffer holds a previous pair's stale values — and an inactive
    /// channel's `mg` must be the `(0,0)` the walk's finalize chain uses.
    /// Emitted deltas must equal a full plan'd recomputation.
    #[test]
    fn coarse_y_only_plan_coarse_channels_rebuilt() {
        let _token = archmage::testing::lock_token_testing();
        let (w, h) = (137usize, 101usize);
        let src = textured(w, h, 5);
        let dst = distort(&src, w, h);
        // v2 Y at scales 1-3 plus v2 X/B at scale 2 only: the chroma mask
        // is {2} so `coarse_y_only_scales` = {1,3} — scale-1/3 X/B are
        // computed but never retained. (No scale-0 X/B in the request
        // either → the scale-0 rebuild path runs too.)
        //   scale-s block: 372 + s*87; Y +29, B +58.
        // Slot 28 (`edge_width_change`) is EXCLUDED from the scale-2 X/B
        // blocks: requesting it would legitimately pull scale-3's chroma
        // back on — EWC at scale s reads scale s+1's gradient sums, which
        // is exactly the plan rule this test exercises around.
        let ids: Vec<usize> = (488..517)
            .chain(546..574)
            .chain(575..632)
            .chain(662..691)
            .collect();
        let plan = crate::feature_plan::Plan::derive(
            &crate::feature_set_id::SlotSet::from_slots(ids),
            692,
        )
        .unwrap();
        assert_eq!(
            plan.compute.coarse_y_only_scales & 0b1010,
            0b1010,
            "plan must actually skip X/B at scales 1 and 3 (got {:#b})",
            plan.compute.coarse_y_only_scales
        );
        let source = RgbSlice::new(&src, w, h);
        let distorted = RgbSlice::new(&dst, w, h);
        let mut scratch = V2Scratch::new();
        let mut ret = FoldRetention::default();
        // STALE-buffer precondition: fill the retention with a different
        // pair's values under a FULL plan, so the coarse-plan walk below
        // leaves foreign pixels in the unretained channels — the defect's
        // worst case, not just zeros.
        {
            let stale_src = textured(w, h, 9);
            let stale_dst = distort(&stale_src, w, h);
            let (_f, _m) = crate::feature_v2::compute_folded_v1_372_streaming_impl(
                &RgbSlice::new(&stale_src, w, h),
                &RgbSlice::new(&stale_dst, w, h),
                None,
                false,
                &mut scratch,
                Some(&full_plan()),
                Some(&mut ret),
            )
            .expect("full walk computes");
        }
        let (base_f, _mo) = crate::feature_v2::compute_folded_v1_372_streaming_impl(
            &source,
            &distorted,
            None,
            false,
            &mut scratch,
            Some(&plan),
            Some(&mut ret),
        )
        .expect("plan'd walk computes");
        // The defect's precondition: stale foreign data really is sitting
        // in the unretained channels when capture runs.
        if crate::ssim_form::active_revision() >= FormulaRevision::Rev5 {
            assert!(ret.pyr_dst[1][0].is_empty(), "Rev5 omits inactive storage");
        } else {
            assert!(
                ret.pyr_dst[1][0].iter().any(|&v| v != 0.0),
                "stale scale-1 X must be present in retention for this test to mean anything"
            );
        }
        let snap = LocalRefineSnapshot::capture(&ret, &plan, (w, h), &source, &distorted)
            .expect("capture accepts the coarse-y-only plan");
        // Ground truth: an ALL_CHANNELS walk's retained planes.
        let (_fb, ret_all) = walk(&src, &dst, w, h);
        for ch in [0usize, 2] {
            assert_eq!(
                snap.pyr_dst0[ch], ret_all.pyr_dst[0][ch],
                "rebuilt dst scale-0 ch{ch} must equal the producer's plane"
            );
            assert_eq!(
                snap.pyr_src0[ch], ret_all.pyr_src[0][ch],
                "rebuilt src scale-0 ch{ch} must equal the producer's plane"
            );
            for si in 0..LOCAL_SCALES {
                assert_eq!(
                    snap.pyr_dst[si][ch],
                    ret_all.pyr_dst[si + 1][ch],
                    "rebuilt dst scale-{} ch{ch} must equal the producer's plane",
                    si + 1
                );
                assert_eq!(
                    snap.pyr_src[si][ch],
                    ret_all.pyr_src[si + 1][ch],
                    "rebuilt src scale-{} ch{ch} must equal the producer's plane",
                    si + 1
                );
            }
        }
        // Inactive channels' `mg` must be the (0,0) production uses.
        for si in 0..LOCAL_SCALES {
            for ch in [0usize, 2] {
                if !plan.compute.channel_active(si + 1, ch) {
                    assert_eq!(
                        snap.mg[si][ch],
                        (0.0, 0.0),
                        "inactive mg s{} ch{ch} must be (0,0)",
                        si + 1
                    );
                }
            }
        }
        // Emitted deltas equal a full plan'd recomputation — the review's
        // example rect plus a second unaligned one.
        for &(x0, y0, x1, y1) in &[(2usize, 0usize, 10usize, 8usize), (40, 30, 63, 47)] {
            let mut intervened = dst.clone();
            for y in y0..y1 {
                for x in x0..x1 {
                    intervened[y * w + x] = src[y * w + x];
                }
            }
            let mut scratch2 = V2Scratch::new();
            let (full_f, _m2) = crate::feature_v2::compute_folded_v1_372_streaming_impl(
                &source,
                &RgbSlice::new(&intervened, w, h),
                None,
                false,
                &mut scratch2,
                Some(&plan),
                None,
            )
            .expect("plan'd walk on intervened image");
            let deltas = snap
                .deltas((x0, y0, x1, y1), &Candidate::Reference)
                .expect("deltas for served plan");
            assert!(!deltas.is_empty());
            for (id, d) in &deltas {
                let fd = full_f[*id] - base_f[*id];
                let tol = delta_tol(base_f[*id], full_f[*id], fd);
                assert!(
                    (d - fd).abs() <= tol,
                    "f{id} rect {x0},{y0}-{x1},{y1}: local {d:e} vs full {fd:e} (tol {tol:e})"
                );
            }
        }
    }

    /// No-op candidate: substituting the rect's own distorted pixels must
    /// produce EXACTLY zero deltas on every emitted id — the cascade and
    /// every term evaluator receive bit-identical inputs, and the f64
    /// blockiness subtractions cancel exactly. This is the sharpest
    /// possible check that no term fabricates a value.
    #[test]
    fn noop_candidate_yields_exact_zero_deltas() {
        let _token = archmage::testing::lock_token_testing();
        let (w, h) = (97usize, 83usize);
        let src = textured(w, h, 0xAB);
        let dst = distort(&src, w, h);
        let (_base, retention) = walk(&src, &dst, w, h);
        let plan = full_plan();
        let snap = LocalRefineSnapshot::capture(
            &retention,
            &plan,
            (w, h),
            &RgbSlice::new(&src, w, h),
            &RgbSlice::new(&dst, w, h),
        )
        .unwrap();
        for &(x0, y0, x1, y1) in &[
            (8usize, 8usize, 16usize, 16usize),
            (3, 5, 26, 22),
            (0, 0, 8, 8),
        ] {
            let (rw, rh) = (x1 - x0, y1 - y0);
            let planes: [Vec<f32>; 3] = std::array::from_fn(|ch| {
                let mut p = vec![0.0f32; rw * rh];
                for y in 0..rh {
                    p[y * rw..(y + 1) * rw].copy_from_slice(
                        &retention.pyr_dst[0][ch][(y0 + y) * w + x0..(y0 + y) * w + x0 + rw],
                    );
                }
                p
            });
            let cand = Candidate::Planar([&planes[0], &planes[1], &planes[2]]);
            let deltas = snap
                .deltas((x0, y0, x1, y1), &cand)
                .expect("no-op candidate must serve");
            assert!(!deltas.is_empty());
            assert!(
                deltas.iter().all(|(_, d)| *d == 0.0),
                "rect ({x0},{y0},{x1},{y1}): no-op candidate must be exactly zero"
            );
        }
    }

    /// Regression for the live `prepare_steering` failure (2026-10-04):
    /// plans that serve no scale-0 X/B feature (`full_res_xb == false`,
    /// e.g. the `by_v2fy` bakes) leave those scale-0 pyramid rows
    /// zero-filled in retention — the walk still computes the channels
    /// internally, it just doesn't retain them. The cascade reads scale-0
    /// pixels for every coarse change, so `capture` must rebuild missing
    /// channels from the pixel inputs. This walks the exact production
    /// path (`compute_folded_v1_372_streaming_impl` with the plan's
    /// compute set) and compares `deltas` against a full recomputation
    /// under the same plan.
    #[test]
    fn sparse_retention_scale0_channels_rebuilt() {
        let _token = archmage::testing::lock_token_testing();
        let (w, h) = (137usize, 101usize);
        let src = textured(w, h, 5);
        let dst = distort(&src, w, h);
        // The `by_v2fy` bakes' exact request (parsed out of
        // `human-by_v2fy-h128-full-s5101.bin`): basic-Y at scale 0,
        // all-channel basic at scales 1–3, v2 luma at scale 0, every v2
        // channel at scales 1–3 → `full_res_xb == false` → scale-0 X/B
        // unretained.
        let ids: Vec<usize> = (13..26)
            .chain(39..156)
            .chain(401..430)
            .chain(459..720)
            .collect();
        let plan = crate::feature_plan::Plan::derive(
            &crate::feature_set_id::SlotSet::from_slots(ids),
            720,
        )
        .unwrap();
        assert!(
            !plan.compute.full_res_xb,
            "plan must exercise the missing-scale-0-channel path"
        );
        let source = RgbSlice::new(&src, w, h);
        let distorted = RgbSlice::new(&dst, w, h);
        let mut scratch = V2Scratch::new();
        let mut ret = FoldRetention::default();
        let (base_f, _mo) = crate::feature_v2::compute_folded_v1_372_streaming_impl(
            &source,
            &distorted,
            None,
            false,
            &mut scratch,
            Some(&plan),
            Some(&mut ret),
        )
        .expect("plan'd walk computes");
        // Precondition of the bug: scale-0 X/B really are unretained.
        assert!(
            ret.pyr_dst[0][0].iter().all(|&v| v == 0.0)
                && ret.pyr_dst[0][2].iter().all(|&v| v == 0.0),
            "retention should lack scale-0 X/B under this plan"
        );
        let snap = LocalRefineSnapshot::capture(&ret, &plan, (w, h), &source, &distorted)
            .expect("capture accepts a v2 plan");
        // The snapshot's rebuilt scale-0 planes must match what a full
        // (ALL_CHANNELS) walk retains.
        let (_full_base, ret_all) = walk(&src, &dst, w, h);
        for ch in [0usize, 2] {
            assert_eq!(
                snap.pyr_dst0[ch], ret_all.pyr_dst[0][ch],
                "rebuilt scale-0 dst ch{ch} must equal the producer's plane"
            );
            assert_eq!(
                snap.pyr_src0[ch], ret_all.pyr_src[0][ch],
                "rebuilt scale-0 src ch{ch} must equal the producer's plane"
            );
        }
        for &(x0, y0, x1, y1) in &[
            (16usize, 16usize, 24usize, 24usize),
            (0, 0, 8, 8),
            (40, 30, 63, 47),
        ] {
            let mut intervened = dst.clone();
            for y in y0..y1 {
                for x in x0..x1 {
                    intervened[y * w + x] = src[y * w + x];
                }
            }
            let mut scratch2 = V2Scratch::new();
            let (full_f, _mo2) = crate::feature_v2::compute_folded_v1_372_streaming_impl(
                &source,
                &RgbSlice::new(&intervened, w, h),
                None,
                false,
                &mut scratch2,
                Some(&plan),
                None,
            )
            .expect("plan'd walk on intervened image");
            let deltas = snap
                .deltas((x0, y0, x1, y1), &Candidate::Reference)
                .expect("deltas for served plan");
            assert!(!deltas.is_empty());
            for (id, d) in &deltas {
                let fd = full_f[*id] - base_f[*id];
                let tol = delta_tol(base_f[*id], full_f[*id], fd);
                assert!(
                    (d - fd).abs() <= tol,
                    "f{id} rect {x0},{y0}-{x1},{y1}: local {d:e} vs full {fd:e} (tol {tol:e})"
                );
            }
        }
    }

    /// Manual end-to-end diagnostic — the exact context the lane measured:
    /// real KADID I01_10_05 + a real by_v2fy bake through
    /// `prepare_steering` + `refinement_gain`, comparing `weighted_gain`
    /// against per-block true feature deltas. LOUD-GATED, never silently
    /// skipped: runs only when explicitly requested (`--ignored`) AND all
    /// three env vars are set —
    ///   `NEIGHSTEER_KADID_DIR` (dir containing `I01.png`, `I01_10_05.png`),
    ///   `NEIGHSTEER_BAKE` (a `.bin` bake file),
    ///   `ZENSIM_NEIGHBOUR_EXACT=1` (exactly "1"),
    /// and it PANICS when any is missing or its path does not exist. Set
    /// `ZENSIM_FORMULA_REV=4` alongside for the reported configuration.
    #[test]
    #[ignore = "external dataset — set NEIGHSTEER_KADID_DIR + NEIGHSTEER_BAKE + ZENSIM_NEIGHBOUR_EXACT=1"]
    fn kadid_case_debug() {
        let _token = archmage::testing::lock_token_testing();
        let dir = std::env::var("NEIGHSTEER_KADID_DIR").expect(
            "kadid_case_debug requires NEIGHSTEER_KADID_DIR=<KADID images dir> \
             (set it or do not run this test)",
        );
        let bake_path = std::env::var("NEIGHSTEER_BAKE").expect(
            "kadid_case_debug requires NEIGHSTEER_BAKE=<path to a .bin bake> \
             (set it or do not run this test)",
        );
        assert_eq!(
            std::env::var("ZENSIM_NEIGHBOUR_EXACT").as_deref(),
            Ok("1"),
            "kadid_case_debug requires ZENSIM_NEIGHBOUR_EXACT=1 exactly"
        );
        let load = |p: &std::path::Path| {
            let img = image::open(p)
                .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
                .to_rgb8();
            let (w, h) = (img.width() as usize, img.height() as usize);
            let v: Vec<[u8; 3]> = img.pixels().map(|p| [p[0], p[1], p[2]]).collect();
            (v, w, h)
        };
        let dir = std::path::Path::new(&dir);
        let (src, w, h) = load(&dir.join("I01.png"));
        let (dst, w2, h2) = load(&dir.join("I01_10_05.png"));
        assert_eq!((w, h), (w2, h2));
        let bytes = std::fs::read(&bake_path).unwrap_or_else(|e| panic!("{bake_path}: {e}"));
        let model = zenpredict::Model::from_bytes(&bytes).unwrap();
        let mut scorer = crate::BakeScorer::new(&model).unwrap().with_parallel(false);
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let mut worker = scorer.prepare_steering(&rs, 8).unwrap();
        let scored = worker.compute(&ds, None).unwrap();
        let Some(snap) = scored.neighbour_exact.as_ref() else {
            panic!("no snapshot captured");
        };
        // A handful of 8x8 blocks across the top row — the same grid the
        // example iterates.
        for bx in [0usize, 8, 16, 24, 32] {
            let rect = (bx, 0usize, bx + 8, 8usize);
            let mut intervened = dst.clone();
            for y in rect.1..rect.3 {
                for x in rect.0..rect.2 {
                    intervened[y * w + x] = src[y * w + x];
                }
            }
            let full = scorer
                .compute(&rs, &RgbSlice::new(&intervened, w, h), None)
                .unwrap();
            let base_f = scored.result().features();
            let deltas = snap.deltas(rect, &Candidate::Reference).unwrap();
            // True delta restricted to the emitted ids.
            let mut sum_s_true = 0.0f64;
            let mut sum_s_local = 0.0f64;
            let mut bad = 0usize;
            let s = scored.sensitivities();
            for (id, d) in &deltas {
                let fd = full.features()[*id] - base_f[*id];
                sum_s_true += s[*id] * fd;
                sum_s_local += s[*id] * d;
                if (d - fd).abs() > 1e-6 + 1e-4 * fd.abs() {
                    if bad < 8 {
                        eprintln!("    block[{bx}] f{id}: local {d:e} vs full {fd:e}");
                    }
                    bad += 1;
                }
            }
            // Whole-row true linear gain vs score delta.
            let lin_all: f64 = (0..base_f.len().min(s.len()))
                .map(|id| s[id] * (full.features()[id] - base_f[id]))
                .sum();
            let ds_true = full.score() - scored.result().score();
            eprintln!(
                "block[{bx}]: ds {ds_true:e}  lin_all {lin_all:e}  \
                 Σs·Δ_true(emitted) {sum_s_true:e}  Σs·Δ_engine {sum_s_local:e}  bad {bad}"
            );
            let rg = scored.refinement_gain(rect.0, rect.1, rect.2, rect.3);
            let dens = scored
                .attribution()
                .query_rect(rect.0, rect.1, rect.2, rect.3);
            eprintln!("    refinement_gain {rg:e}  density {dens:e}");
            assert_eq!(
                bad, 0,
                "block[{bx}]: {bad} engine deltas disagree with full recomputation"
            );
        }
    }

    /// The lane's cost numbers: snapshot heap bytes, and µs per 8×8 and
    /// 32×32 reference-repair query at ~1 MP (1024×1024 textured —
    /// `#[ignore]`d so it runs on demand, not in the gate). Reported in
    /// `benchmarks/neighsteer_WORKLOG.md` / `NEIGHSTEER_DONE.md`.
    #[test]
    #[ignore = "cost measurement — run explicitly"]
    fn cost_per_query_1mp() {
        let _token = archmage::testing::lock_token_testing();
        let (w, h) = (1024usize, 1024usize);
        let src = textured(w, h, 0x51);
        let dst = distort(&src, w, h);
        let (_base, retention) = walk(&src, &dst, w, h);
        let plan = full_plan();
        let snap = LocalRefineSnapshot::capture(
            &retention,
            &plan,
            (w, h),
            &RgbSlice::new(&src, w, h),
            &RgbSlice::new(&dst, w, h),
        )
        .unwrap();
        eprintln!("snapshot heap: {} bytes", snap.heap_bytes());
        // Full-walk timing next to the per-query timing — the records'
        // "100–1000× cheaper than a recompute" claim must be measured,
        // not asserted.
        let mut walk_us = f64::INFINITY;
        for _ in 0..3 {
            let t = std::time::Instant::now();
            std::hint::black_box(walk(&src, &dst, w, h));
            walk_us = walk_us.min(t.elapsed().as_secs_f64() * 1e6);
        }
        eprintln!("full fold-944 walk: {:.1} ms", walk_us / 1e3);
        for (name, rect) in [
            ("8x8", (400usize, 400usize, 408usize, 408usize)),
            ("32x32", (400usize, 400usize, 432usize, 432usize)),
        ] {
            // Warm-up + measure.
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                std::hint::black_box(snap.deltas(rect, &Candidate::Reference).unwrap());
                best = best.min(t.elapsed().as_secs_f64() * 1e6);
            }
            eprintln!(
                "{name} query: {best:.1} us  (walk/query = {:.1}x)",
                walk_us / best
            );
        }
    }
}
