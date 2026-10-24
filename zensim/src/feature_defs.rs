//! **The feature DEFINITION registry** — what each slot IS, one entry per
//! signal, and the layout arithmetic that expands signals into slot ids.
//!
//! Design: `docs/FEATURE_SYSTEM_DESIGN_2026-09-05.md` §2. Phases + gates:
//! `docs/PLAN_FEATURE_SYSTEM_2026-09-05.md`.
//!
//! [`crate::feature_set_id`] answers *"are these two vectors the same kind of
//! thing?"* at the granularity of a slot FAMILY. This module answers *"what IS
//! slot 353?"* — which, before it landed, nothing in-tree could. Per-slot
//! naming existed only as a 13-entry array inside a `#[cfg(test)]` diagnostic
//! (`streaming.rs`) and four `pub mod idx*` constant blocks that give a LOCAL
//! offset inside one block.
//!
//! ## Signals, not slots
//!
//! A 944-wide vector holds 944 slots but only **85 distinct signals**: the
//! layout replicates most of them across 4 pyramid scales × 3 channels. The
//! registry stores the 85 and expands them, so declaring a new v2 signal is
//! ONE entry rather than twelve, and the (scale, channel) arithmetic has one
//! owner instead of being written out at every emit site.
//!
//! ## The layout arithmetic, in one place
//!
//! Block-major, exactly as `feature_v2`'s emit sites document it:
//!
//! ```text
//!   [0, 156)    basic    scale-major, then channel:  scale*39 + ch*13 + local
//!   [156, 228)  peaks    (scale*3 + ch)*6 + local, based at n_scales*39
//!   [228, 300)  masked   same shape,               based at n_scales*57
//!   [300, 372)  iw       same shape,               based at n_scales*75
//!   [372, 720)  v2       scale*3*29 + ch*29 + local
//!   [720, 924)  append   scale*3*17 + ch*17 + local
//!   [924, 944)  append2  scale*5 + local     (Y-only: one cell per scale)
//!   [944, 956)  csfw     scale*3 + local     (Y-only)
//!   [956, 986)  reserved; later IDs retain their original positions.
//! ```
//!
//! ## Why the geometry constants are DERIVED here, not copied
//!
//! `feature_v2::FEATURES_PER_CHANNEL_APPEND` is `17` because there are 17
//! append signals. This module knows that number by COUNTING its own append
//! entries, which is a derivation rather than a second copy — and
//! [`tests::geometry_matches_the_block_constants`] proves the two agree, so a
//! new signal that forgets to bump the constant fails the build rather than
//! silently shifting every slot above it.
//!
//! ## Append-only
//!
//! Ids are never renumbered (user directive 2026-07-19: *"new v2 features use
//! indices after all v1 features; we deprecate not renumber"*). A retired
//! signal keeps its `block_local` forever and gets [`SignalDef::deprecated`];
//! a revision APPENDS to [`SignalDef::revisions`] and never edits history.

use crate::feature_set_id::{ComputeToken, SlotSet};

/// Which colour plane a slot is pooled over.
///
/// [`Channel::Scalar`] is not "no channel" — it is the per-SCALE blocks
/// (append2, csfw) whose signals exist once per scale rather than once per
/// (scale, channel). They are computed on Y (`feature_v2::APPEND2_CHANNEL`)
/// but occupy one slot, not three.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Channel {
    /// XYB X (red-green chroma).
    X,
    /// XYB Y (luma).
    Y,
    /// XYB B (blue-yellow chroma).
    B,
    /// A per-scale slot with no channel axis.
    Scalar,
}

impl Channel {
    /// The per-(scale, channel) blocks index channels in this order.
    pub(crate) const TRIPLE: [Channel; 3] = [Channel::X, Channel::Y, Channel::B];

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Channel::X => "x",
            Channel::Y => "y",
            Channel::B => "b",
            Channel::Scalar => "s",
        }
    }
}

/// How a signal pools its per-pixel term over the plane.
///
/// Recorded because it is the difference between two slots that share a name
/// stem and are otherwise indistinguishable in the vector (`ssim_mean` vs
/// `ssim_4th` vs `ssim_2nd` are one per-pixel term under three pooling
/// exponents), and because the pooling exponent is what a future revision is
/// most likely to change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Statistic {
    /// Arithmetic mean of the per-pixel term.
    Mean,
    /// `(Σx²/n)^½`.
    L2,
    /// `(Σx⁴/n)^¼`.
    L4,
    /// `(Σx⁸/n)^⅛`.
    L8,
    /// `(Σm⁴/n)^¼` pooled over per-BLOCK maxima (the restored `z1max` family).
    /// Distinct from [`Statistic::L4`] on purpose: it shares v1's `RootForm`
    /// handling but is NOT attached to the `v1detroot`/F18 registry gates,
    /// whose pinned counts are measurements over the v1/v2 slots
    /// (`registered_defects_cover_exactly_the_audited_slots`).
    BlockL4,
    /// `(Σm⁸/n)^⅛` over per-block maxima; see [`Statistic::BlockL4`].
    BlockL8,
    /// Plane maximum.
    Max,
    /// `Σw·v / Σw` — the canonical weighted pooling the masked/IW blocks use.
    WeightedMean,
    /// A ratio of two pooled quantities (the HF gain/loss/mag family).
    Ratio,
    /// A mean restricted to a luminance bin.
    Bin,
    /// A whole-plane scalar that is not a pooled per-pixel term (the
    /// `GLOBAL_*` / `W_GLOBAL_*` families, computed from raw moments).
    Global,
}

impl Statistic {
    /// Stable lower-case token, for provenance output.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Statistic::Mean => "mean",
            Statistic::L2 => "l2",
            Statistic::L4 => "l4",
            Statistic::L8 => "l8",
            Statistic::BlockL4 => "block_l4",
            Statistic::BlockL8 => "block_l8",
            Statistic::Max => "max",
            Statistic::WeightedMean => "weighted_mean",
            Statistic::Ratio => "ratio",
            Statistic::Bin => "bin",
            Statistic::Global => "global",
        }
    }
}

/// What a slot costs when its OWNING BLOCK runs.
///
/// This is the signal's base cost. It is NOT the whole cost story: a slot can
/// also be [`Tranche`]-harvestable, i.e. obtainable for free by a walk that
/// never runs its owning block at all. See [`cost_of`], which is the honest
/// per-SLOT answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CostClass {
    /// Falls out of an accumulator the plan already runs for another slot.
    Free,
    /// Shares an existing sweep; no new pass over the plane.
    Cheap,
    /// Needs its own pass, plane, or kernel.
    Expensive,
}

impl CostClass {
    /// Stable lower-case token, for provenance output.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            CostClass::Free => "free",
            CostClass::Cheap => "cheap",
            CostClass::Expensive => "expensive",
        }
    }
}

/// Whether a slot can be harvested by a v1-only walk that never runs the
/// block the slot lives in — `feature_v2::V1FreeExtras`'s two tranches.
///
/// **This is why cost is a property of a SLOT, not of a SIGNAL.** The
/// registry's first draft declared it per signal and the cross-check gate
/// caught it: `lum_dark_err` is harvestable at Y and not at X/B;
/// `global_dmean` is harvestable everywhere EXCEPT the `(B, scale 0)` append
/// cell the 944 walk skips entirely; and the v2-348 `mse` slot is harvestable
/// despite living in the most expensive block in the layout. A per-signal
/// cost field is wrong by 16 slots on the raw-moment tranche alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tranche {
    /// Not harvestable — needs its owning block.
    None,
    /// `V1FreeExtras::RawMoments`: derivable from the fused kernel's raw
    /// `Σs`, `Σd`, `Σs²`, `Σd²`.
    RawMoments,
    /// `V1FreeExtras::RawMomentsPlusBoundedErr`'s addition: derivable from
    /// the fused kernel's saturating-MSE and luminance-weighted sums.
    ClassC,
}

impl Tranche {
    /// Stable lower-case token, for provenance output.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Tranche::None => "none",
            Tranche::RawMoments => "raw_moments",
            Tranche::ClassC => "class_c",
        }
    }
}

/// Which (scale, channel) placements of a signal its [`Tranche`] can serve.
///
/// Mirrors the two real restrictions in `feature_v2`'s owner functions:
/// `APPEND_SKIP_B_SCALE0` (the 944 walk never computes that append cell, so
/// its slots are structural zeros on both sides) and the Y-only register-carry
/// constraint on the luminance bins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Placement {
    /// Every cell the block occupies.
    AllCells,
    /// Every cell except the append block's `(B, scale 0)`.
    SkipBScale0,
    /// The Y channel only, and only where the append cell is computed.
    YOnlyActive,
}

/// Whether a slot is a difference between reference and distorted, and so
/// whether an identity pair must drive it to zero.
///
/// Reference-only slots can remain nonzero on a perfect copy. For example,
/// `append2::LUMA_MEAN_REF` reports the reference's luminance, and Rev5
/// `PJND_FRAGILITY` reports reference structure. The historical 372 all-zero
/// identity probe captured the product shortcut's placeholder features;
/// it does not establish that canonical extracted identity features are zero.
///
/// [`Form::Undeclared`] is an HONEST state, not a default to fill in later: a
/// signal whose identity behaviour has not been established (or, for the
/// restored-cut slots, depends on the formula revision) reports that, rather
/// than claiming a form nobody measured. The declared ones are pinned
/// by [`tests::declared_difference_forms_are_zero_on_an_identity_pair`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Form {
    /// A difference/dissimilarity: exactly `0.0` on an identity pair.
    Difference,
    /// A similarity: `1.0`-ish on an identity pair, higher is better.
    Similarity,
    /// Computed from the reference alone; an identity pair does not zero it.
    ReferenceOnly,
    /// Not yet established. Never treated as any of the above.
    ///
    /// Declared by the restored-cut families (2026-09-25), the only committed
    /// uses: the `mapdev` distorted-side deviations (`hfsq_dst_dev`,
    /// `hfabs_dst_dev`: a property of one image, no reference/distorted
    /// relation to zero) and the SSIM-derived `z1max` slots (block-locals 0, 1,
    /// 2, 13, 16), whose identity value depends on the formula revision —
    /// exactly 0 at Rev3 (direct-error form), nonzero at Rev1 — and which are
    /// not attached to the F4 defect gate (see [`Statistic::BlockL4`]).
    Undeclared,
}

impl Form {
    /// Stable lower-case token, for provenance output.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Form::Difference => "difference",
            Form::Similarity => "similarity",
            Form::ReferenceOnly => "reference_only",
            Form::Undeclared => "undeclared",
        }
    }
}

/// The monotone expectation a dial gate may rely on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Direction {
    /// Larger value = more distortion.
    HigherIsWorse,
    /// Larger value = better fidelity.
    HigherIsBetter,
    /// No signed expectation (reference-only conditioners, raw statistics).
    Unsigned,
    /// Not yet established.
    ///
    /// Reserved for the Phase-2 research engine's provenance output
    /// (`docs/PLAN_FEATURE_SYSTEM_2026-09-05.md`); no signal in the committed
    /// registry currently declares it, so nothing constructs it yet.
    #[allow(dead_code)]
    Undeclared,
}

impl Direction {
    /// Stable lower-case token, for provenance output.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Direction::HigherIsWorse => "higher_is_worse",
            Direction::HigherIsBetter => "higher_is_better",
            Direction::Unsigned => "unsigned",
            Direction::Undeclared => "undeclared",
        }
    }
}

/// The kernel that owns a signal's accumulation — the "who computes this"
/// link the plan follows to decide what must run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KernelId {
    /// v1's fused basic/edge/HF accumulation (`fused.rs`, replayed by the
    /// fold's `fold_v1_basic_bands`).
    V1Fused,
    /// v1's peak tier (max + L8), unconditional inside the fused kernel.
    V1Peaks,
    /// v1's shared activity chain feeding masked and IW together.
    V1MaskIw,
    /// The v2-348 dense block kernel.
    V2Dense,
    /// The v2 gradient kernel (GMS, ringing, banding, edge width).
    V2Gradient,
    /// The append-204 kernel.
    Append,
    /// The append2 / BANDVIS kernel.
    Append2,
    /// The CSFW tier-1 kernel.
    Csfw,
    /// The free raw-moment accumulator a v1-only walk finalizes.
    FreeRawMoments,
    /// The free bounded-error (class C) accumulator.
    FreeBoundedErr,
    /// The Rev4 gridblk boundary-excess pass (C1's own kernel).
    Gridblk,
    /// The restored-cut per-band error-map side pass (`feature_v2::restore_cuts`):
    /// the v1 fused kernel's own per-pixel maps consumed by deviation and
    /// block-max pooling.
    RestoreMaps,
    /// Research palette signature and population transport.
    Palette,
    /// The walk's per-strip global-affine moment sums (`crate::globaff`).
    GlobAff,
}

impl KernelId {
    /// Stable lower-case token, for provenance output.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            KernelId::V1Fused => "v1_fused",
            KernelId::V1Peaks => "v1_peaks",
            KernelId::V1MaskIw => "v1_mask_iw",
            KernelId::V2Dense => "v2_dense",
            KernelId::V2Gradient => "v2_gradient",
            KernelId::Append => "append",
            KernelId::Append2 => "append2",
            KernelId::Csfw => "csfw",
            KernelId::FreeRawMoments => "free_raw_moments",
            KernelId::FreeBoundedErr => "free_bounded_err",
            KernelId::Gridblk => "gridblk",
            KernelId::RestoreMaps => "restore_maps",
            KernelId::Palette => "palette",
            KernelId::GlobAff => "globaff",
        }
    }
}

/// Whether a [`Revision`] has LANDED (the values moved) or is PROPOSED (the
/// defect is known and modelled, the fix is not applied).
///
/// Modelling a proposed revision is not the same as making one. The
/// architecture's job is to make "which slots would this change, and what
/// would it cost to re-extract?" a lookup — so a fix can be scheduled with its
/// blast radius known, instead of discovered after it lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RevisionStatus {
    /// The values moved; tables built before it read a different quantity.
    Landed,
    /// Registered, NOT applied. Shipped bytes are unchanged.
    Proposed,
}

impl RevisionStatus {
    /// Stable lower-case token, for provenance output.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            RevisionStatus::Landed => "landed",
            RevisionStatus::Proposed => "proposed",
        }
    }
}

/// A known defect in a signal's current definition.
///
/// Sourced from `docs/FEATURE_DEFECTS_AUDIT_2026-09-05.md`, whose ids
/// (`F4`, `F5`, `F15`, …) are the keys. A defect is a property of the SIGNAL,
/// so the registry is where it belongs: "does anything I read have a live
/// defect?" becomes a query over a bake's read set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Defect {
    /// The audit's id.
    pub id: &'static str,
    /// What is wrong, and what currently absorbs it.
    pub note: &'static str,
}

/// One registered change to how a signal's value is computed.
///
/// The era axis the identity layer records per TABLE, recorded here per
/// SIGNAL — so *"is my number affected by era X?"* is a lookup instead of a
/// re-extraction. Append-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Revision {
    /// The `feature_set_id` era token this landed in, or the token a proposed
    /// revision would create.
    pub era: &'static str,
    /// The byte-changing commit, or `"-"` for a proposal.
    pub commit: &'static str,
    /// Landed, or registered-not-applied.
    pub status: RevisionStatus,
    /// What changed (or would), and what it means for numbers read before it.
    pub note: &'static str,
}

/// WHICH registered revision's semantics a computation uses.
///
/// The [`Revision`] entries above record, per signal, which eras MOVED it.
/// This is the knob that SELECTS one — phase 3 of
/// `docs/PLAN_FEATURE_SYSTEM_2026-09-05.md`, executed by
/// `docs/PLAN_FEATURE_REV2_2026-09-05.md`.
///
/// It is deliberately a small closed enum rather than a set of era tokens:
/// eras in this crate have only ever moved forward together, and a research
/// extraction needs to reproduce "the semantics of era X", not an arbitrary
/// per-slot mixture. [`Self::era_tokens`] is the bridge back to the registry,
/// so "which slots does this revision move?" stays a lookup over the signal
/// table rather than a second list that can drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum FormulaRevision {
    /// Everything shipped through era `v1postc` — the semantics every stored
    /// table and every published verdict was read at.
    #[default]
    Rev1,
    /// The 2026-09-05 batched arithmetic-fix era: the audit's F4, F5 and F17
    /// fixes landing together, so exactly one era boundary exists rather than
    /// one per defect — one recalculation, not three.
    Rev2,
    /// Revision 2 plus stable f64 pairwise-error moments for v1 SSIM
    /// basic, peak, masked and IW signals. Requires freshly extracted data.
    Rev3,
    /// Revision 3 formulas under **canonical arithmetic** (the `tiercanon`
    /// era), served and steered in a Rev4 process (REV4SERVE).
    ///
    /// The canonical leaves — opsin XYB, the fused H/V SSIM blurs and pools,
    /// the free raw-moment / bounded-error / luma-bin accumulators, the CSFW
    /// pass, and the PU front end (`color::pu_xyb_canon` over the
    /// `_at_revision` transfer decoders) — run ONE source-level body with
    /// fused `f32::mul_add`, fixed 8-virtual-lane pools and a fixed pairwise
    /// reduce. Measured, not proven: `research::extract`'s full 1825-slot
    /// SDR vector is bit-identical across x86 v4x/v4/v3/scalar and wasm32
    /// simd128/scalar on the measured pairs (wasm32: 6 pairs, 8×8 to
    /// 512×384).
    ///
    /// A Rev4 computation must run in a Rev4 process (and vice versa):
    /// `ssim_form::refuse_rev4_mix` refuses every cross-boundary mix because
    /// the process-global gates (`compute_mode`, `RootForm`, `PowForm`,
    /// `HfGainForm`, the luma form) still read `ZENSIM_FORMULA_REV`.
    /// `streaming::active_channels` routes every active channel through the
    /// fused SSIM kernels at Rev4, so the tier-dispatched edge-only
    /// (`blur::fused_blur_h_mu`) and MSE-only (`sq_diff_sum`) routes are
    /// never reached. Requires freshly extracted data: 837 of 1825 slots
    /// move vs production v3 (union over the lane's 11 audit pairs). Only
    /// for the fused-V-blur pools was the accumulation order shown not to
    /// matter against the f64 exact oracle; see
    /// `benchmarks/featcanon_WORKLOG.md`.
    Rev4,
    /// Revision 4's formulas under **local-window arithmetic** (the
    /// `localwin` era, `benchmarks/rev5_spec_2026-10-04.md`), computed for
    /// the `basic` (f0..227), `peaks` (f156..227) and `v2` (f372..719)
    /// families only.
    ///
    /// Rev5 replaces the sliding-sum blur recurrences with a per-output
    /// 11-tap f32 pair tree over the window's own inputs (production's
    /// mirror padding), replaces the 8-lane pools with 16 fixed virtual
    /// lanes and one fixed pairwise reduce, allows fused multiply-add
    /// wherever the same fused expression is written, and carries the
    /// stable central-moment finalizers. Bit-identity across tiers is a
    /// property of the single fixed order, not a hope.
    ///
    /// The two-family scope is deliberate: a plan or bake whose read set
    /// leaves `basic + peaks + v2` is refused at plan time
    /// ([`crate::feature_plan::PlanError::UnsupportedAtRev5`]) rather than
    /// computed at a lower revision behind the caller's back. The process
    /// model is Rev4's: `ZENSIM_FORMULA_REV=5` pins it and
    /// `ssim_form::refuse_rev4_mix` refuses every cross-boundary mix.
    Rev5,
    /// Revision 5 with the corrected ringing measurement (the `ringhpm` era,
    /// imazen/zensim#72, `benchmarks/byconstruction_2026-10-22.md`).
    ///
    /// The twelve v2 `ringing` slots measure the error after removing each
    /// image's SSIM local mean, `(src − mu1) − (dst − mu2)`, instead of
    /// `src − dst` (`feature_v2::RingForm::HighPassMu`), so a global offset,
    /// contrast or saturation change no longer reads as ringing away from
    /// edges. Every other slot keeps Rev5's arithmetic bit for bit. It also
    /// admits the `globaff` family (f1867..1890): per scale and XYB channel,
    /// the share of the mean squared error a global least-squares fit
    /// `dst ≈ a·src + b` explains, and that explained error.
    ///
    /// Rev5's scope and process model otherwise apply unchanged:
    /// `ZENSIM_FORMULA_REV=6` pins it for research, and a bake declaring
    /// revision 6 selects it per call.
    Rev6,
    /// Revision 6 with faster and cancellation-free pooled arithmetic (the
    /// `rowlanes` era, `benchmarks/byconstruction_2026-10-22.md`, "Speed pass
    /// before revision 7"):
    ///
    /// * **f32 row lanes.** The dense, fused-SSIM and gradient pools add
    ///   the columns of each row's complete 8-column chunks to sixteen f32
    ///   lanes (`x & 15`) and fold every lane into its f64 lane at the end
    ///   of the chunks and after every 4,096th column, instead of adding
    ///   every column in f64; the remaining columns add in f64 as before.
    ///   The f64 lanes live for a band of 32 rows (dense) or for the call
    ///   (fused SSIM, gradient: a 128-row strip) and are finished once there
    ///   instead of once per row. Pools whose features difference or ratio
    ///   two large sums (v1
    ///   variance, texture and contrast terms, the edge-width change, the
    ///   band-lifetime raw moments) keep f64 lanes.
    /// * **Blockiness lattice lanes.** The vertical lattice step at column
    ///   `8j` adds to f64 lane `(j − 1) & 7` instead of lane 0, so the eight
    ///   lanes take eight steps at once; each 128-row kernel strip sums into
    ///   its own lanes, added to the running lanes in strip order (strips run
    ///   in parallel, with the same bits at any thread count).
    /// * **Central moments per row.** The SSIM deviation moments are
    ///   two-pass per row (a sixteen-lane f64 mean, then the centred sums)
    ///   with one Chan/Pébay merge per row instead of one per 16 columns.
    /// * **Cancellation-free forms.** The v2 SSIM luminance term is
    ///   `(m1 − m2)²`, the bounded dissimilarities `(a − b)²/(a² + b² + c)`
    ///   and the v1 edge ratio `(x − y)/(1 + y)`: the exact values the old
    ///   forms computed by subtracting nearly equal numbers.
    /// * **Opsin cube root with one division**
    ///   ([`crate::det_math::cbrt_newton_f32`]): within 3 ulp like
    ///   `cbrt_midp`, correctly rounded more often (72.7% against 56%), with
    ///   one division instead of two.
    ///
    /// Any live slot may move (an arithmetic era); scope and process model
    /// are Rev6's (`ZENSIM_FORMULA_REV=7`, or a bake declaring revision 7).
    Rev7,
}

impl FormulaRevision {
    /// The era tokens this revision introduces, relative to [`Self::Rev1`].
    ///
    /// Empty for `Rev1`, which is the baseline rather than a change.
    pub(crate) const fn era_tokens(self) -> &'static [&'static str] {
        match self {
            Self::Rev1 => &[],
            Self::Rev2 => &["v1ssimcap", "freecomp", "v1hfgain", "v1detroot", "scorepow"],
            Self::Rev3 => &[
                "v1ssimcap",
                "freecomp",
                "v1hfgain",
                "v1detroot",
                "scorepow",
                "v1ssimstable",
                "v2ssimstable",
                "v1extfused",
            ],
            Self::Rev4 => &[
                "v1ssimcap",
                "freecomp",
                "v1hfgain",
                "v1detroot",
                "scorepow",
                "v1ssimstable",
                "v2ssimstable",
                "v1extfused",
                "tiercanon",
                "c3negfold",
            ],
            Self::Rev5 => &[
                "v1ssimcap",
                "freecomp",
                "v1hfgain",
                "v1detroot",
                "scorepow",
                "v1ssimstable",
                "v2ssimstable",
                "v1extfused",
                "tiercanon",
                "c3negfold",
                "localwin",
            ],
            Self::Rev6 => &[
                "v1ssimcap",
                "freecomp",
                "v1hfgain",
                "v1detroot",
                "scorepow",
                "v1ssimstable",
                "v2ssimstable",
                "v1extfused",
                "tiercanon",
                "c3negfold",
                "localwin",
                "ringhpm",
            ],
            Self::Rev7 => &[
                "v1ssimcap",
                "freecomp",
                "v1hfgain",
                "v1detroot",
                "scorepow",
                "v1ssimstable",
                "v2ssimstable",
                "v1extfused",
                "tiercanon",
                "c3negfold",
                "localwin",
                "ringhpm",
                "rowlanes",
            ],
        }
    }

    /// Whether the v2 `ringing` family measures the local-mean-removed error
    /// ([`Self::Rev6`]'s `ringhpm` era) rather than the raw pixel error.
    pub(crate) const fn ring_high_pass(self) -> bool {
        matches!(self, Self::Rev6 | Self::Rev7)
    }

    /// Whether the `globaff` family (f1867..1890) is computed at this revision.
    pub(crate) const fn computes_globaff(self) -> bool {
        matches!(self, Self::Rev6 | Self::Rev7)
    }

    /// Whether the pools add columns in f32 row lanes and the central
    /// moments are two-pass per row ([`Self::Rev7`]'s `rowlanes` era).
    pub(crate) const fn row_lanes(self) -> bool {
        matches!(self, Self::Rev7)
    }

    /// Whether the opsin conversion takes its cube root from
    /// [`crate::det_math::cbrt_newton_f32`] (a Newton then a Halley step, one
    /// division) instead of `cbrt_midp` ([`Self::Rev7`]).
    pub(crate) const fn cbrt_newton(self) -> bool {
        matches!(self, Self::Rev7)
    }

    /// Whether the cancellation-free forms replace the subtractive ones:
    /// `(m1 − m2)²` for the v2 SSIM luminance term, `(a − b)²/(a² + b² + c)`
    /// for the bounded dissimilarities and `(x − y)/(1 + y)` for the v1 edge
    /// ratio ([`Self::Rev7`]).
    pub(crate) const fn stable_forms(self) -> bool {
        matches!(self, Self::Rev7)
    }

    /// Whether `GLOBAL_CGAIN`/`GLOBAL_CLOSS` are formed from the PAIRED
    /// second moment `Σ(d−s)(d+s)` instead of the difference of two raw
    /// variances — the `freecomp` half of revision 2.
    ///
    /// F5: `fused.rs`'s own comment claims the band-batched reorder "costs
    /// negligible precision: worst |Δ| vs the 944 append block is 5.35e-6".
    /// That was measured through `free_extras_match_the_944_append_block`,
    /// which is SYNTHETIC-IMAGE-ONLY. On real pairs the audit measures 2,607
    /// of 28,601 cells (9.12 %) past the 2e-5 bar, worst 3.63e-3 — because
    /// `Σs²/n − mean²` amplifies the accumulated f32 error by `mean²/var`,
    /// which is unbounded as a region flattens, and synthetic images do not
    /// produce the flat-but-not-constant regions that make it large.
    ///
    /// **Granularity was tried first and MEASURED not to be the lever**:
    /// matching the append kernel's per-row reduction moves the paired
    /// disagreement only 4.47 % -> 3.90 % past the bar and leaves the worst
    /// cell worse, because the error lives in the f32 accumulation WITHIN one
    /// row. Compensation removes the depth dependence instead. See
    /// [`crate::fused`]'s `kahan_add16` for the full measurement.
    ///
    /// It changes **no shipped byte**: only the free walk is compensated, the
    /// append route is untouched, so no 944 table moves. That also makes the
    /// free route the more accurate of the two, which is what turns the
    /// remaining free-vs-append gap into a MEASUREMENT of the append route's
    /// own error (plan R4) rather than an unattributed disagreement.
    pub(crate) const fn paired_global_contrast(self) -> bool {
        matches!(
            self,
            Self::Rev2 | Self::Rev3 | Self::Rev4 | Self::Rev5 | Self::Rev6 | Self::Rev7
        )
    }

    /// Every slot id this revision moves, derived from the signal table's own
    /// [`Revision`] entries — never from a separate list. The union over
    /// [`Self::era_tokens`], so a revision that batches three defects returns
    /// all three defects' slots.
    ///
    /// This is the predicate gate **G3.1** checks against a re-extraction: a
    /// slot that moves and is not returned here is a FAILURE.
    ///
    /// A revision that carries an ARITHMETIC era ([`ARITHMETIC_REVISIONS`],
    /// Rev4's `tiercanon`) moves every live slot, so it returns them all.
    pub(crate) fn moved_slots(self, width: u16, n_scales: usize) -> Vec<u16> {
        let tokens = self.era_tokens();
        let all = tokens.iter().any(|t| is_arithmetic_era(t));
        (0..width as usize)
            .filter_map(|id| def_at(id, n_scales))
            .filter(|d| all || d.signal.revisions.iter().any(|r| tokens.contains(&r.era)))
            .map(|d| d.id)
            .collect()
    }
}

/// Every slot id ONE registered era moves.
///
/// [`FormulaRevision::moved_slots`] is the union over a revision's eras, which
/// is what a recalculation needs — but a G3.1 gate is a claim about ONE defect
/// ("F4 moves these 132"), and once a revision batches three of them the union
/// can no longer express that claim. Added when `v1hfgain` (F17) joined
/// revision 2 and the F4 gate's 132 became 144: the gate was asserting an
/// era-level fact through a revision-level instrument, which was correct only
/// while the revision had one v1 era in it.
pub(crate) fn era_moved_slots(era: &str, width: u16, n_scales: usize) -> Vec<u16> {
    // An arithmetic era (`tiercanon`) is not a per-signal era: canonical
    // pooling order can move ANY slot a canonical leaf feeds (837/1825
    // measured on the audit set, content-dependent), so the honest
    // registration is every live slot. See [`ARITHMETIC_REVISIONS`].
    if is_arithmetic_era(era) {
        return (0..width as usize)
            .filter_map(|id| def_at(id, n_scales))
            .map(|d| d.id)
            .collect();
    }
    (0..width as usize)
        .filter_map(|id| def_at(id, n_scales))
        .filter(|d| d.signal.revisions.iter().any(|r| r.era == era))
        .map(|d| d.id)
        .collect()
}

/// Internal dispatch family. Palette has no supported public token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Family {
    Supported(ComputeToken),
    Palette,
}

impl From<ComputeToken> for Family {
    fn from(token: ComputeToken) -> Self {
        Self::Supported(token)
    }
}

impl PartialEq<ComputeToken> for Family {
    fn eq(&self, token: &ComputeToken) -> bool {
        *self == Self::Supported(*token)
    }
}

impl Family {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Supported(token) => token.as_str(),
            Self::Palette => "palette",
        }
    }
}

impl core::fmt::Display for Family {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One registered signal: a feature definition before the layout replicates
/// it across scales and channels.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SignalDef {
    /// The slot family — the [`ComputeToken`] vocabulary, one owner.
    pub family: Family,
    /// Index within the family's per-cell block (the `idx*` constant).
    pub block_local: u16,
    /// Name stem, `[a-z0-9_]+`, unique within the family.
    pub name: &'static str,
    // `statistic`, `form`, `direction`, `kernel`, and `deprecated` are all
    // constructed (every signal declaration sets them) but not yet read back
    // anywhere — they are Phase 1a's declarative data, awaiting the Phase-2
    // research engine's provenance consumer
    // (`docs/PLAN_FEATURE_SYSTEM_2026-09-05.md`). Not dead: reserved.
    #[allow(dead_code)]
    pub statistic: Statistic,
    /// Cost when the signal's OWNING BLOCK runs. See [`cost_of`] for the
    /// per-slot answer, which also accounts for [`SignalDef::tranche`].
    pub cost: CostClass,
    /// Whether a v1-only walk can harvest this signal without its block.
    pub tranche: Tranche,
    /// Which placements the tranche can serve.
    pub placement: Placement,
    #[allow(dead_code)]
    pub form: Form,
    #[allow(dead_code)]
    pub direction: Direction,
    #[allow(dead_code)]
    pub kernel: KernelId,
    /// Retired: the slot keeps its id and its structural zero forever.
    #[allow(dead_code)]
    pub deprecated: bool,
    /// A live defect in the current definition, from the defect audit.
    pub defect: Option<Defect>,
    pub revisions: &'static [Revision],
}

/// One slot: a [`SignalDef`] placed at a (scale, channel) by the layout.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FeatureDef {
    /// The stable slot id. Append-only, never renumbered.
    ///
    /// Constructed by every layout expansion but not yet read back anywhere
    /// — reserved for the Phase-2 provenance consumer
    /// (`docs/PLAN_FEATURE_SYSTEM_2026-09-05.md`).
    #[allow(dead_code)]
    pub id: u16,
    pub signal: &'static SignalDef,
    pub scale: u8,
    pub channel: Channel,
}

impl FeatureDef {
    /// The slot's unique name: `<family>_<signal>_s<scale>_<channel>`.
    ///
    /// The family prefix is what makes it unique — `basic`'s `ssim_mean` and
    /// `v2`'s `ssim_mean` are different quantities under the same stem.
    /// Uniqueness over every registered layout is gated by
    /// [`tests::names_are_unique_and_valid`].
    pub(crate) fn name(&self) -> String {
        format!(
            "{}_{}_s{}_{}",
            self.signal.family.as_str(),
            self.signal.name,
            self.scale,
            self.channel.as_str()
        )
    }
}

// ============================================================================
// The signal table
// ============================================================================

const NO_REV: &[Revision] = &[];

/// The commit that introduced the Rev4 feature bank — recorded so
/// `Revision::commit` names a real hash rather than a placeholder (a landed
/// revision must name the byte-changing commit; the registry test enforces
/// `!= "-"`). Pinned by the follow-up commit once the landing commit exists.
const REV4BANK_COMMIT: &str = "adef35be";

/// The Rev4 feature-bank introduction era. Every slot of the four appended
/// families (f986..1321 at 4 scales) is born carrying it, so
/// `era_moved_slots("rev4bank", …)` enumerates exactly the appended range —
/// the same machine-checkable answer the `v1ssimcap`/`v1hfgain` entries give
/// for their eras.
const REV4BANK: &[Revision] = &[Revision {
    era: "rev4bank",
    commit: REV4BANK_COMMIT,
    status: RevisionStatus::Landed,
    note: "append-only introduction of the four Rev4 candidate families \
           (gridblk, ringbasis, tailhist, arttype) at f986..1321. No existing \
           slot's value moves; disabled families emit nothing and cost \
           nothing. Spec: docs/REV4_FEATURE_BANK_PLAN_2026-09-23.md; design: \
           benchmarks/rev4_featbank_impl_2026-09-23.md.",
}];

/// C3 fix: the tailhist histogram's bin lookup read `f64::to_bits` directly,
/// so any value with the sign bit set (tiny negatives from f32 rounding in
/// `edge_dissim`, NaN) sorted above every positive edge and landed in the
/// top bin — REVIEW_PARTB (2026-09-25) measured 2.01-3.36% top-bin cells on
/// KADID/TID `art`/`det` in the pre-fix bank; on TRAIN data with the fold
/// the top-edge saturation is <= 0.81% (benchmarks/rev4canon_WORKLOG.md).
/// The corrected lookup folds non-positive and NaN values into bin 0. The
/// exact maxima never changed, so the era attaches to the eight `Bin`
/// (`*_p95`/`*_p99`) signals only.
///
/// **Scoped to [`FormulaRevision::Rev4`].** The tailhist family is the
/// landed `rev4bank` era and is computed at EVERY formula revision, so the
/// fold is gated on the computation's revision: Rev1-Rev3 research
/// extractions keep the legacy bins byte for byte (Known Bugs, zensim
/// `CLAUDE.md`).
const REV_C3NEGFOLD: Revision = Revision {
    era: "c3negfold",
    commit: "-",
    status: RevisionStatus::Proposed,
    note: "tailhist bins sign-bit-set and NaN values into bin 0 instead of \
           the top bin, at FormulaRevision::Rev4 only (`TailAccum::fold`); \
           Rev1-Rev3 keep the legacy bins byte for byte. Moves only the Bin \
           signals; `*_max` is exact and unchanged. Root cause of the \
           phantom C3 top-bin saturation measured in REVIEW_PARTB \
           2026-09-25; record in benchmarks/rev4canon_WORKLOG.md.",
};

/// The defect behind the `ringhpm` era: the ringing error is the raw pixel
/// error, so a global edit reads as ringing.
const DEFECT_RING_GLOBAL: Defect = Defect {
    id: "RINGGLOBAL",
    note: "v2 `ringing` saturates the raw error `|src − dst|`, which a global \
           offset, contrast or saturation change makes uniformly large away \
           from edges. MEASURED on 24 references (feature_probe.py, \
           benchmarks/byconstruction_2026-10-22.md): a +8 luma shift was \
           ringing's largest response, 2.4x a visible blur sigma 1.2 \
           (median 0.334 vs 0.139). imazen/zensim#72.",
};

/// The `ringhpm` era: ringing measures the error after removing each image's
/// SSIM local mean ([`FormulaRevision::Rev6`]).
const REV_RINGHPM: Revision = Revision {
    era: "ringhpm",
    commit: "-",
    status: RevisionStatus::Proposed,
    note: "ringing's per-pixel error becomes `(src − mu1) − (dst − mu2)` with \
           the SSIM local means the dense kernel already holds; the three \
           saturations are unchanged. Moves only the 12 ringing slots \
           (+8 shift median 0.334 -> 0.017, blur unchanged 0.139 -> 0.143). \
           Rev5 keeps the raw form byte for byte; `ZENSIM_RING_FORM=hpm` is \
           the per-process research arm of the same arithmetic. Record: \
           benchmarks/byconstruction_2026-10-22.md.",
};
const RINGING_REVS: &[Revision] = &[REV_RINGHPM];

/// The tailhist `Bin` signals carry `rev4bank` plus the C3 binning fix;
/// `Max` signals keep `REV4BANK` alone (exact maxima are unaffected).
const TAILHIST_BIN_REVS: &[Revision] = &[REV4BANK[0], REV_C3NEGFOLD];

/// Pinned to the byte-changing implementation commit in the qualification
/// follow-up after the local quarantine commit has a stable hash.
const GMSBANK_COMMIT: &str = "5cdcf70a";
const GMSBANK: &[Revision] = &[
    Revision {
        era: "gmsbank",
        commit: GMSBANK_COMMIT,
        status: RevisionStatus::Landed,
        note: "append-only introduction of the five-constant GMS similarity bank, \
           loss/gain/population-deviation at f1322..1501. No earlier slot moves. \
           Design: benchmarks/rev4_gmsbank_design_2026-09-23.md.",
    },
    Revision {
        era: "gmsbank-chroma-2026-09-24",
        commit: "7b8e8a4f",
        status: RevisionStatus::Landed,
        note: "Quarantined implementation revision: native Y gradients, coarse X/Y/B \
           gradients with per-channel TRAIN stabilisers, and coarse joint X/B \
           chromaticity mean-loss/Welford deviation. Width 180 at four scales. \
           No f0..f1321 change; no consumer or promotion. \
           Preregistration: benchmarks/gmsd-chroma_prereg_2026-09-24.md.",
    },
];

/// Restored-cut families (COST_CUTS_AUDIT, 2026-09-24 user ruling: "remember
/// not to reject things for the cost budget ... we can optimize and make
/// things optional"). Each is a default-off, append-only family. The commit
/// is pinned to the byte-changing implementation commit once it exists.
const RESTORE_COMMIT: &str = "384d15e1";
const MAPDEV: &[Revision] = &[Revision {
    era: "mapdev",
    commit: RESTORE_COMMIT,
    status: RevisionStatus::Landed,
    note: "append-only introduction of the per-scale, per-channel population \
           standard deviation (per-row Welford, row-ordered Chan merge) of the \
           raw squared-error map and the four HF energy/magnitude maps, \
           f1502..1561. Restores COST_CUTS_AUDIT A1 (gmsd lane A_dev, the part \
           that is NOT a function of existing mean/L2 columns). No earlier \
           slot moves.",
}];
const Z1MAX: &[Revision] = &[Revision {
    era: "z1max",
    commit: RESTORE_COMMIT,
    status: RevisionStatus::Landed,
    note: "append-only introduction of the 228-slot basic+peaks surface pooled \
           over the (0,0)-anchored ungated 5x5 block-MAX lattice (partial \
           border blocks dropped) instead of pixels, f1562..1789. Restores \
           COST_CUTS_AUDIT B2 (zgeom lane z1max). No earlier slot moves. The \
           4th/8th-root slots use `Statistic::BlockL4/BlockL8` and go through \
           the same RootForm as v1's; they are deliberately NOT attached to the \
           v1detroot/F18 gates (pinned v1/v2 counts) until the family is \
           promoted from an optional research arm.",
}];
const GMSNATIVE: &[Revision] = &[Revision {
    era: "gmsnative",
    commit: RESTORE_COMMIT,
    status: RevisionStatus::Landed,
    note: "append-only introduction of C8's X and B gradient loss/gain/deviation \
           bank at NATIVE scale, f1790..1819 (the 30 slots the chroma revision \
           dropped). Same per-channel stabilisers as the coarse X/B cells. \
           Restores COST_CUTS_AUDIT Ambiguous 7. No earlier slot moves.",
}];

/// The v1 option-C revision: v1 stopped pooling mirror-padded phantom
/// columns, which moves every pooled v1 slot at any non-tight width.
const REV_OPTION_C: &[Revision] = &[Revision {
    era: "v1postc",
    commit: "56bbcda2",
    status: RevisionStatus::Landed,
    note: "option C: v1 stopped pooling mirror-padded phantom columns. Moves \
           every pooled v1 slot at widths where simd_padded_width(w) != w \
           (512/576/1152/2304 among them). Numbers read on v1pre/v1cur cannot \
           be corrected into v1postc, only re-extracted.",
}];

/// **F4** — the one LIVE arithmetic defect
/// (`docs/FEATURE_DEFECTS_AUDIT_2026-09-05.md`).
const DEFECT_F4: Defect = Defect {
    id: "F4",
    note: "v1's SSIM per-pixel dissimilarity has a `.max(0)` floor and NO upper \
           cap, and `num_m = 1 - (mu1-mu2)^2` carries no C1. L4 pooling then \
           amplifies it: f313 reads 5,814,302 on scanned rows against a \
           photographic p99.9 of 0.48. Absorbed today by the bake-side winsor \
           guard, not by the feature. BLAST RADIUS CORRECTED 2026-09-05: the \
           audit attached this to the masked and IW blocks only (72 slots); a \
           bounded-form re-extraction MEASURES 132 — basic 36, peaks 24, \
           masked 36, IW 36 — because every one of them is pooled from the \
           SAME per-pixel `d`. The peaks pair is `ssim_max`/`ssim_l8`, which a \
           per-block reading of the audit would not have predicted.",
};

/// The PROPOSED fix for [`DEFECT_F4`] — registered, deliberately NOT applied.
const REV_F4_PROPOSED: &[Revision] = &[
    Revision {
        era: "v1postc",
        commit: "56bbcda2",
        status: RevisionStatus::Landed,
        note: "option C: v1 stopped pooling mirror-padded phantom columns.",
    },
    REV_F4_ENTRY,
    REV_SSIM_STABLE,
];

/// **F18** — the pooled 4th/8th roots make the extractor LIBC-DEPENDENT.
///
/// Not in the 2026-09-05 audit's F1..F17 inventory; found 2026-09-06 by the
/// rev2 fleet wave's own rev1 correctness gate, which is the only reason
/// anyone was looking — nothing weaker than `to_bits()` equality would have
/// seen it.
const DEFECT_F18: Defect = Defect {
    id: "F18",
    note: "`powf` is not correctly rounded and no standard requires two libcs \
           to agree on it, so `(sum x^4/n).powf(0.25)` and \
           `(sum x^8/n).powf(0.125)` make a feature a function of WHICH LIBM \
           the binary linked against. MEASURED: a static-musl build of the \
           same source differs from a glibc build on 77/322,152 csiq cells \
           (0.0239 %) and 328/1,116,000 tid cells (0.0294 %), every delta \
           exactly one ULP at f64; inside the basic block the differing \
           positions are exactly `= 1, 4, 7 (mod 13)`, i.e. only the \
           `.powf(0.25)` slots. Mechanism closed by a probe with nothing to do \
           with zensim: `x ** 0.25` over 400,000 doubles disagrees on 276 \
           between glibc 2.43 and musl and on 0 between two glibc versions. \
           Consequence: the fleet had to build its Feature executor against \
           glibc to reproduce the dev box bit-for-bit. Every OTHER pooled v1 \
           slot uses `sqrt`, which IEEE-754 requires to be correctly rounded \
           and which is a hardware instruction, so it is libc-independent by \
           construction - these are the whole exposure. Record: \
           benchmarks/libm_pow_nondeterminism_2026-09-06.md.",
};

/// The PROPOSED fix for [`DEFECT_F18`] on a v1 slot that carries NO other
/// defect — registered, deliberately NOT applied.
const REV_DETROOT_PROPOSED: &[Revision] = &[
    Revision {
        era: "v1postc",
        commit: "56bbcda2",
        status: RevisionStatus::Landed,
        note: "option C: v1 stopped pooling mirror-padded phantom columns.",
    },
    REV_DETROOT,
];

/// The PROPOSED fix for [`DEFECT_F18`] on a slot that ALSO carries F4 — the
/// SSIM-derived `L4`/`L8` pools, which are in both eras.
const REV_F4_AND_DETROOT: &[Revision] = &[
    Revision {
        era: "v1postc",
        commit: "56bbcda2",
        status: RevisionStatus::Landed,
        note: "option C: v1 stopped pooling mirror-padded phantom columns.",
    },
    REV_F4_ENTRY,
    REV_SSIM_STABLE,
    REV_DETROOT,
];

/// [`DEFECT_F18`]'s fix on a v2-era slot (`ssim_dev4`), which option C never
/// touched.
const REV_V2_DETROOT: &[Revision] = &[REV_DETROOT];

/// The v2 block's SSIM-derived signals under revision 3. They read the same
/// `sigma12` plane the v1 signal does, and under revision 3 that plane holds
/// the direct error moment — so `feature_v2::ssim_d_local` forms the v2
/// dissimilarity from it directly rather than as `cov = s12 - mu1*mu2`. A
/// separate era from `v1ssimstable` because it names a different kernel and a
/// different slot family; both are in revision 3's list.
const REV_EXT_FUSED: Revision = Revision {
    era: "v1extfused",
    commit: "-",
    status: RevisionStatus::Proposed,
    note: "Rev3 (issue #61 perf campaign): the masked/IW extension is FUSED \
           into the SSIM V sweep (`fused::ExtPoolsWork`). The activity is \
           `blur(|src - H(src)|)` exactly as before — the H-only mu plane the \
           H pass already wrote, H-blurred once, V-blurred inside the sweep \
           by the same `sum + add - rem` recurrence — and every masked/IW \
           pool is formed from the same per-pixel values the separate passes \
           formed. What moves is the ORDER the f64 chunk sums are added in \
           (column-group-major inside the sweep, row-major in the passes), so \
           these slots move at the last f64 bits on every image. The masked/\
           IW SSIM pools are already `v1ssimstable` movers; this era adds \
           the v1 MASKED/IW `edge_art_4th`, `edge_det_4th` and `mse` slots. STILL PROPOSED: `SHIPPED_REVISION` \
           is `Rev1`.",
};
/// `v1()`'s default list (`v1postc`) plus the fused-extension era: the v1
/// MASKED/IW `mse` slots.
const REV_OPTION_C_AND_EXT_FUSED: &[Revision] = &[
    Revision {
        era: "v1postc",
        commit: "56bbcda2",
        status: RevisionStatus::Landed,
        note: "option C: v1 stopped pooling mirror-padded phantom columns.",
    },
    REV_EXT_FUSED,
];
/// `v1()`'s pooled-root list (`v1postc` + F18) plus the fused-extension era:
/// the v1 MASKED/IW `edge_art_4th` / `edge_det_4th` slots.
const REV_DETROOT_AND_EXT_FUSED: &[Revision] = &[
    Revision {
        era: "v1postc",
        commit: "56bbcda2",
        status: RevisionStatus::Landed,
        note: "option C: v1 stopped pooling mirror-padded phantom columns.",
    },
    REV_DETROOT,
    REV_EXT_FUSED,
];

const REV_V2_SSIM_STABLE: Revision = Revision {
    era: "v2ssimstable",
    commit: "-",
    status: RevisionStatus::Proposed,
    note: "Rev3 (issue #61): the v2 dense kernel's SSIM dissimilarity \
           (`ssim_d_local` / `ssim_d_local_v`) reads the fourth moment plane \
           as the DIRECT error `E[(a-b)^2]` and forms \
           `(d(b-a) + a*err_var)/(b*d)` instead of recovering the covariance \
           by subtraction. Same bounded-error contract as `v1ssimstable`. \
           Registered after the first fused build was caught (by reading, \
           not by a gate) still computing `cov = s12 - mu1*mu2` on the \
           redefined plane; `rev3_moves_exactly_the_registered_slots_on_the_944_layout` \
           now covers the v2 block. Proposed: nothing is trained against it.",
};
const REV_V2_SSIM: &[Revision] = &[REV_V2_SSIM_STABLE];
/// `ssim_dev4` carries F18's pooled-root era AND the v2 SSIM era.
const REV_V2_SSIM_AND_DETROOT: &[Revision] = &[REV_DETROOT, REV_V2_SSIM_STABLE];

/// The one text of the `v1detroot` era, so the three lists above cannot drift.
const REV_DETROOT: Revision = Revision {
    era: "v1detroot",
    commit: "-",
    status: RevisionStatus::Proposed,
    note: "F18 fix: evaluate the pooled 4th/8th roots as `sqrt.sqrt` and \
           `sqrt.sqrt.sqrt`. THE ARM IS DERIVED, NOT CHOSEN - IEEE-754 \
           requires `sqrt` to be correctly rounded, so for these two \
           exponents the replacement is unique, bit-identical on every libc, \
           and cheaper. NOT more accurate: `sqrt.sqrt` double-rounds where \
           `powf` rounds once, and MEASURED over 4,000 log-uniform doubles \
           against a 60-digit reference the two agree on 3,455 while glibc's \
           `pow` is nearer the truth in 544 of the 545 that differ - so the \
           case is determinism and a BOUNDED error, not accuracy (the \
           discovery record's contrary claim is falsified in \
           benchmarks/libc_determinism_2026-09-06.md section 3). Owner: \
           `crate::det_math::RootForm`. Its cost is that it moves revision-1 \
           bytes by up to one ULP - which is why it is an era item and not a \
           plain fix. It rides in revision 2 with F4, F5 and F17 so \
           there is ONE era boundary and one recalculation. MIGRATION: this \
           era invalidates any `ZENSIM_FORMULA_REV=2` table extracted BEFORE \
           it was registered (the R6b lane has some); \
           `ZENSIM_ROOT_FORM=libm` reproduces those exactly. STILL PROPOSED: \
           `ssim_form::SHIPPED_REVISION` is `Rev1`, so no shipped byte moves.",
};

/// **The one text of the `v1ssimcap` era.** Both [`REV_F4_PROPOSED`] (a slot
/// in F4 alone) and [`REV_F4_AND_DETROOT`] (a slot in F4 *and* F18) name this
/// constant, so the two lists cannot state the same era differently — the
/// exact drift a second copy of the text would invite.
const REV_SSIM_STABLE: Revision = Revision {
    era: "v1ssimstable",
    commit: "-",
    status: RevisionStatus::Proposed,
    note: "Rev3 (issue #61), FUSED form: the v1 SSIM dissimilarity is formed \
           from a DIRECT error moment. `blur::fused_blur_h_ssim` accumulates \
           `sum (a-b)^2` in the fourth plane in place of `sum a*b` (same two \
           FMAs), and `fused::fused_vblur_ssim_inner` forms \
           `loss + (1-loss)*E_err/(var1+var2+C2)` from the same four f32 \
           planes it always V-blurred — no second traversal. Basic, peak, \
           masked, IW and the attribution planes all read that one retained \
           value. BOUNDED, not exact, by user directive 2026-09-09 (\"bounded \
           error is fine, speed above minor flaws\"): registered on the \
           locality fixture, peak out-of-support movement <= 2e-5 (measured \
           4.277e-6; shipped rev 1 measures 4.886e-4), max abs error vs the \
           exact f64 reference kernel <= 1e-3, all-equal-window residue \
           <= 1e-5 (measured 3.689e-6). The exact f64 second-pass kernel \
           `ssim_form::stable_ssim_plane` measured 0 movement but cost \
           +27..87% of extraction (22% of the walk in perf) and is retained \
           only as the reference the bounds are measured against. This era \
           moves every SSIM-derived slot on every image, unlike `v1ssimcap`. \
           STILL PROPOSED: `ssim_form::SHIPPED_REVISION` is `Rev1`, no table \
           has been re-extracted and no bake refit against it.",
};

const REV_F4_ENTRY: Revision = Revision {
    era: "v1ssimcap",
    commit: "-",
    status: RevisionStatus::Proposed,
    note: "F4 fix: bound the per-pixel dissimilarity's luminance term. The \
           ARM IS DECIDED — `SsimLumaForm::Clamp`, i.e. `max(0, 1 - D^2)` \
           — by measurement over 217,756 rows \
           (benchmarks/f4_arm_decision_2026-09-05.md). Clamp is \
           BIT-IDENTICAL to revision 1 wherever `D^2 <= 1`, and no corpus \
           with local pixels reaches past that, so this era moves NOTHING \
           on content resembling the shipped corpora; the 5.8e6 outlier \
           that motivates it lives in the bigcodec sweep. STILL PROPOSED, \
           not landed: `ssim_form::SHIPPED_REVISION` is `Rev1` until the \
           recalculation and refit land, and Proposed is what `shipped \
           bytes are unchanged` means.",
};

/// **F5** — the free-40 raw-moment route-parity skew.
const DEFECT_F5: Defect = Defect {
    id: "F5",
    note: "`global_stats_from_raw_moments` uses the catastrophic-cancellation \
           form `sum(s^2)/n - (sum(s)/n)^2`, and the two routes reduce at \
           different granularities (append kernel per ROW, fused kernel per \
           BAND). 2,607 of 28,601 cells (9.12 %) exceed the 2e-5 parity bar on \
           real pairs, worst 3.63e-3. The class-C 24 slots are clean (0/18,552).",
};

/// The PROPOSED fix for [`DEFECT_F5`] — the CHEAPEST window, because no
/// shipped bake reads these slots yet.
const REV_F5_PROPOSED: &[Revision] = &[Revision {
    era: "freecomp",
    commit: "-",
    status: RevisionStatus::Proposed,
    note: "F5 fix: compensated (or two-pass) accumulation for the raw second \
           moment, so the free route and the append kernel agree. Migration \
           cost is ZERO shipped bytes TODAY — no shipped bake reads the \
           raw-moment tranche — and rises the moment one does. Land it before \
           a bake reads these slots, not after.",
}];

/// **F17** — the v1 HF-energy GAIN member is unbounded above.
///
/// Not in the 2026-09-05 audit's F1..F16 inventory; found by R6 while reading
/// its own "max over ALL slots" column, and registered here by R6b.
/// **F19** — the SCORE path is libc-dependent for the same reason F18's
/// features were, and [`DEFECT_F18`]'s fix does not reach it.
///
/// Named in F18's own record as a known, unmeasured exposure. Measured
/// 2026-09-06 by extending that lane's cross-libc gate to the score column it
/// was already dumping.
///
/// # ★ THIS DEFECT MOVES NO FEATURE SLOT — deliberately
///
/// Every other entry in this registry attaches to a [`SignalDef`], so
/// [`era_moved_slots`] answers "which slots?" by derivation. F19 attaches to
/// none, because the score is not a feature: it is
/// `metric.rs`'s mapping of a raw distance, the MLP head runtimes' p-norm
/// pooling and sigmoids, and the four `--mlp-size-axes` inputs. Its era is
/// registered in [`SCORE_PATH_REVISIONS`] instead, and
/// [`scorepow_moves_no_feature_slot`] pins the emptiness so a later reader
/// cannot mistake it for an oversight.
const DEFECT_F19: Defect = Defect {
    id: "F19",
    note: "The raw-distance -> score mapping calls `powf` at exponents that \
           are not powers of two - `score_mapping_b` = 0.7 on EVERY shipped \
           profile, plus 0.5979 / 1.2244 / 0.6130 in the `approx_*` helpers \
           and a bake-supplied p-norm `p` in the per-sample-alpha and hybrid \
           head runtimes - and `exp` in the bounded squash, the soft clamp, \
           the tanh output pin and both head sigmoids, and `log2` in the four \
           `--mlp-size-axes` MLP inputs. `powf`, `exp` and `log2` are all \
           un-correctly-rounded libm entry points, so a SCORE is a function of \
           which libc the binary linked against. Unlike F18 there is NO \
           derivation available: no chain of correctly-rounded operations \
           evaluates x^0.7, so the arm is an ALGORITHM choice and the property \
           bought is sameness, not accuracy. MEASURED (the F18 gate extended \
           to its score column, 220 procedural cells): glibc vs static-musl \
           differ on 1 of 220 scores under BOTH root arms - i.e. fixing the \
           features did not fix the score - and on 0 of 220 under the pure \
           arm. Owner: `crate::det_math::PowForm`.",
};

/// [`DEFECT_F19`]'s era. Registered in [`SCORE_PATH_REVISIONS`], not on any
/// signal — the score is not a slot.
const REV_SCOREPOW: Revision = Revision {
    era: "scorepow",
    commit: "-",
    status: RevisionStatus::Proposed,
    note: "F19 fix: evaluate the score path's `powf` / `exp` / `log2` with \
           `libm::{pow, exp, log2}` - the pure-Rust port of musl's fdlibm - \
           instead of the platform libm. THE ARM IS CHOSEN, NOT DERIVED, and \
           that is the difference from `v1detroot`: x^0.7 has no finite \
           correctly-rounded evaluation, so the only property available is \
           that every target run THE SAME SOURCE. `libm::pow` carries no \
           `select_implementation!` and no fma; `libm::exp`/`log2` carry one, \
           gated `x86_no_sse`, which no target this crate ships for selects. \
           NOT more accurate than glibc: MEASURED against a 60-digit reference \
           over the score's own domain, glibc's `pow` is nearer the truth more \
           often than the port is - the case is determinism and a measured \
           bound. MIGRATION: this era moves SCORE bytes by a few ULP; it moves \
           NO stored feature table, because it touches no feature. \
           `ZENSIM_POW_FORM=libm` reproduces any pre-era score exactly. STILL \
           PROPOSED: `ssim_form::SHIPPED_REVISION` is `Rev1`, so no shipped \
           byte moves. BLOCKER ON FLIPPING, registered 2026-09-06: \
           `zensim-validate`'s `bake_runtime` (and its `bake_compare` fork) \
           re-implement the per-sample-alpha and hybrid HEAD math and document \
           themselves bit-exact with `metric.rs`. They do NOT follow \
           `PowForm`, and no test holds them together - so the claim is true \
           today (both defaults are `LibmPowf`) and becomes false the instant \
           this era activates, which would make a VERDICT disagree with the \
           score the product returns. Routing them needs a `pub` surface on \
           `det_math`. Same shape, one repo further out and NOT fixable from \
           here: `zenpredict::feature_transform`'s `cbrt`/`powf`/`ln`/`ln_1p` \
           are on the PRODUCT path via `predict_transformed` and are LIVE in \
           Profiles A, BHdr and C (B - the default - and D declare only \
           `winsor_p99`, a clamp, and are clean).",
};

/// The registered eras that change a **score** rather than a feature slot.
///
/// [`era_moved_slots`] derives a feature era's blast radius from the signal
/// table. A score era has no slots to derive from, so it needs a home of its
/// own — and `research.rs`'s `era_is_registered` consults BOTH, which is what
/// keeps "every active era token is registered" a real invariant instead of
/// one that a score era would have silently broken.
pub(crate) const SCORE_PATH_REVISIONS: &[(Defect, Revision)] = &[(DEFECT_F19, REV_SCOREPOW)];

/// Whether `era` names a registered SCORE-path era.
pub(crate) fn is_score_path_era(era: &str) -> bool {
    SCORE_PATH_REVISIONS.iter().any(|(_, r)| r.era == era)
}

/// The defect behind the `tiercanon` era: the same feature computed on two
/// SIMD tiers differs in its low bits.
///
/// Registered on the registry rather than on a signal because it is a
/// property of the ARITHMETIC every canonical leaf shares, not of one
/// formula. Sources: `REVIEW_TIERPARITY.md` and the featcanon lane
/// (`benchmarks/featcanon_WORKLOG.md`).
const DEFECT_TIER_DIVERGENCE: Defect = Defect {
    id: "TIERDIV",
    note: "Feature bits depend on the SIMD tier: `mul_add` is fused on \
           v3/v4/neon and `a*b+c` on the magetypes scalar and wasm128 \
           backends; `reduce_add`'s tree is backend-defined; tail/chunk \
           geometry differs per width. MEASURED at production Rev3 on 11 \
           audit pairs: v4x vs v3 differ on 0-297 of 1825 research slots per \
           pair (the v1 SSIM pools), scalar vs v3 on 125-1615 (1523 minimum on \
           a non-identity pair) across every family. Not an accuracy defect: \
           a sameness defect.",
};

/// The `tiercanon` era: Rev3 formulas under canonical arithmetic.
const REV_TIERCANON: Revision = Revision {
    era: "tiercanon",
    commit: "-",
    status: RevisionStatus::Proposed,
    note: "featcanon: every canonical leaf (opsin XYB, the fused H/V SSIM \
           blurs and pools, the free raw-moment / bounded-error / luma-bin \
           accumulators, the CSFW pass) runs ONE plain-Rust body with fused \
           `f32::mul_add`, 8 fixed virtual f32 lanes (lane = x mod 8) and the \
           fixed `era2_reduce8` tree, so the 1825-slot research vector is \
           bit-identical across x86 v4x/v4/v3/scalar (measured on the lane's \
           11 audit pairs and the review's 12 probe pairs) and wasm32 \
           simd128/scalar (measured, 5 crops) - by \
           measurement, not by construction, since leaves outside this era \
           837 of 1825 slots vs production v3 on the audit pairs (union), so \
           it is registered as moving every live slot. REV4SERVE made the \
           served, HDR, diffmap and attribution leaves canonical at Rev4 \
           (`color::pu_xyb_canon` + `_at_revision` transfer decoders; \
           `streaming::active_channels` routes every active channel through \
           the fused SSIM kernels); served paths accept Rev4 in a Rev4 \
           process and refuse only a cross-boundary mix \
           (`ssim_form::refuse_rev4_mix`). STILL PROPOSED.",
};

/// The defect behind the `localwin` era: window sums evaluated by updating
/// the PREVIOUS window (`sum + add − rem`), so rounding error accrues with
/// position instead of being re-bounded per output, and raw-power-sum →
/// central-moment conversions that catastrophically cancel.
///
/// Registered on the registry rather than on a signal for the same reason
/// [`DEFECT_TIER_DIVERGENCE`] is: it is a property of the arithmetic shape,
/// not of one formula. Sources: `benchmarks/featcanon_WORKLOG.md` (f32
/// storage of the sliding running sum is the dominant structured error;
/// `ssim_dev4` raw-moment cancellation), `benchmarks/rev5_spec_2026-10-04.md`.
const DEFECT_WINDOW_DRIFT: Defect = Defect {
    id: "WINDRIFT",
    note: "The box blurs evaluate each window by updating the previous \
           window's running sum, so f32 rounding error accrues with position \
           rather than re-bounding per output; `ssim_dev4`-style raw-moment \
           conversions (`raw4 − 4µ·raw3 + 6µ²·raw2 − 3µ⁴`) cancel \
           catastrophically on near-constant content (the FEATACC 17×9 crop \
           collapses a positive ~3.76e-4 to 0).",
};

/// The `localwin` era: Rev4's formulas under per-output local-window
/// arithmetic — Rev5.
const REV_LOCALWIN: Revision = Revision {
    era: "localwin",
    commit: "-",
    status: RevisionStatus::Proposed,
    note: "rev5: the 11-tap box blurs (H and V, and the fused SSIM planes) \
           sum each output's own 11 inputs in a fixed f32 pair tree with \
           production's mirror padding — no running-sum recurrence, so an \
           output depends only on its window. Pools carry 16 fixed virtual \
           lanes closed by one fixed pairwise reduce; fused multiply-add is \
           written wherever it is the same fused expression. Elements stay \
           f32. The central moments use block-local moments merged through \
           the fixed-order Chan/Pébay formulas — no raw-power-sum expansion. \
           Scoped to the basic + peaks + v2 families: a plan or bake whose \
           read set leaves them is refused at plan time rather than computed \
           at a lower revision behind the caller's back. Moves every live \
           slot (an arithmetic era); tier-identical by construction of the \
           one fixed order.",
};

/// The registered eras that change the ARITHMETIC of every canonical leaf
/// rather than one signal's formula.
///
/// Like [`SCORE_PATH_REVISIONS`], a home for eras the per-signal table cannot
/// express. An arithmetic era moves every live slot ([`era_moved_slots`],
/// [`FormulaRevision::moved_slots`]), is registered for
/// `research::era_is_registered`, and — when the build computes it — is every
/// slot's provenance era (`research::current_era_of`).
/// The defect behind the `rowlanes` era: subtractive forms that cancel, and
/// pooled arithmetic that costs more than its precision needs.
const DEFECT_FORM_CANCELLATION: Defect = Defect {
    id: "FORMCANCEL",
    note: "Three formulas compute a small value by subtracting nearly equal \
           numbers: the v2 SSIM luminance term `(m1² + m2² + C1) − (2·m1·m2 + \
           C1)`, the bounded dissimilarity `1 − (2ab + c)/(a² + b² + c)` and \
           the v1 edge ratio `(1 + x)/(1 + y) − 1`. MEASURED \
           (byconstruction record, \"Formula revision 7\"): +336% / −87% on \
           near-identical chroma features of 1e-8 to 1e-9; scores move by at \
           most 7.9e-5 points. Separately, every pool column was added in f64 \
           (a conversion and an f64 add per term) and the central moments \
           merged once per 16 columns, most of the pooled kernels' time.",
};

/// The `rowlanes` era: Rev6 with f32 row lanes, per-row central moments and
/// the cancellation-free forms ([`FormulaRevision::Rev7`]).
const REV_ROWLANES: Revision = Revision {
    era: "rowlanes",
    commit: "-",
    status: RevisionStatus::Proposed,
    note: "rev7: the dense, fused-SSIM and gradient pools add the columns of \
           each row's complete 8-column chunks to sixteen f32 lanes (`x & 15`, \
           ascending x), folded into f64 lanes at the end of the chunks and \
           after every 4,096th column, except pools whose features difference \
           or ratio two large sums (f64, as before); the f64 lanes live for a \
           32-row band (dense) or the call (fused SSIM, gradient) and finish \
           once there; the v1 HF and edge-width sums add source sums plus \
           differences; the blockiness lattice's vertical steps use all \
           eight f64 lanes, summed per 128-row strip; the SSIM deviation moments are two-pass per row with \
           one Chan/Pébay merge per row; the three subtractive forms become \
           their exact cancellation-free equivalents; the opsin cube root \
           takes one division (`cbrt_newton_f32`). MEASURED (byconstruction \
           record, \"Revision 7 speed after the implementation\"): against \
           revision 6 on B's 55 dumps, 22,190 of 39,600 feature values move, \
           by a median 1.1e-6 relative (largest 4.0e-3, an absolute 2.3e-8), \
           and the score by at most 4.0e-5 points; against the exact f64 \
           oracle (93 near-identical pairs) the per-feature median error's \
           p90 falls from 2.80e-6 to 2.39e-6 and its worst feature rises from \
           5.10e-5 to 5.91e-5 (`peaks_ssim_max_s3_b`, the cube root's error \
           pattern). Registered as moving every live slot (an arithmetic era); \
           tier-identical by the same fixed order.",
};

pub(crate) const ARITHMETIC_REVISIONS: &[(Defect, Revision)] = &[
    (DEFECT_TIER_DIVERGENCE, REV_TIERCANON),
    (DEFECT_WINDOW_DRIFT, REV_LOCALWIN),
    (DEFECT_FORM_CANCELLATION, REV_ROWLANES),
];

/// Whether `era` names a registered ARITHMETIC era.
pub(crate) fn is_arithmetic_era(era: &str) -> bool {
    ARITHMETIC_REVISIONS.iter().any(|(_, r)| r.era == era)
}

/// The registered arithmetic-era entry named `era`, if any.
pub(crate) fn arithmetic_revision(era: &str) -> Option<&'static Revision> {
    ARITHMETIC_REVISIONS
        .iter()
        .find(|(_, r)| r.era == era)
        .map(|(_, r)| r)
}

const DEFECT_F17: Defect = Defect {
    id: "F17",
    note: "v1's `contrast_inc` = `max(0, var_dst/var_src - 1)` divides by the \
           SOURCE term, so its numerator is unbounded by its denominator. Its \
           two siblings in the same family divide by the source term too, but \
           their numerators are `max(0, src - dst)`, which the denominator \
           bounds — measured max exactly 1.000000 for both. The `> 1e-10` gate \
           is a threshold, not a stabiliser. MEASURED over 216,756 real pairs \
           on 8 corpora: the twelve `contrast_inc` slots are the TOP TWELVE of \
           all 372 by maximum (worst 36,465.74) and the thirteenth is 1.972 — \
           x105,127 the gold holdout's own p99.9 of 0.34687. Unlike F4 it \
           fires on real corpora, on 0.0198 % of cells. Absorbed by a \
           bake-side winsor guard on Profiles A and B ONLY: Profile D (the SDR \
           default) has no transform block at all and reads f116 (max 1,380) \
           and f155 (max 2,127) raw, and CHdr reads all twelve at `identity`.",
};

/// The PROPOSED fix for [`DEFECT_F17`] — registered, deliberately NOT applied.
const REV_F17_PROPOSED: &[Revision] = &[
    Revision {
        era: "v1postc",
        commit: "56bbcda2",
        status: RevisionStatus::Landed,
        note: "option C: v1 stopped pooling mirror-padded phantom columns.",
    },
    Revision {
        era: "v1hfgain",
        commit: "-",
        status: RevisionStatus::Proposed,
        note: "F17 fix: bound the HF-energy gain member. THE ARM IS DECIDED — \
               `HfGainForm::SaturatingExcess`, i.e. `g/(g+1)` — by measurement \
               over 216,756 rows, and it is the ONLY arm passing H3+H4+H5 \
               (docs/PLAN_FEATURE_REV2_2026-09-05.md section 11, \
               benchmarks/feature_rev2_2026-09-05.md section 11). Unlike \
               `v1ssimcap` this era moves values on EVERY corpus with pixels — \
               `contrast_inc` is nonzero on 12 % (CID22) to 52 % (KADID) of \
               cells — so a rev2 flip re-extracts these twelve slots for real. \
               MEASURED benefit and cost: LIVE 0.7357 -> 0.9500 (+0.214), TID \
               +0.033, KADID +0.021, CID22 +0.0027..+0.0090 CI-excluding; and \
               KonJND REGRESSES -0.013..-0.080 on EVERY arm, which makes it a \
               property of bounding rather than of this arm. It rides in \
               revision 2 with F4 and F5 so there is ONE era boundary and one \
               recalculation. STILL PROPOSED: \
               `ssim_form::SHIPPED_REVISION` is `Rev1`.",
    },
];

/// **F15** — `PJND_FRAGILITY` is nonzero on an identity pair.
const DEFECT_F15: Defect = Defect {
    id: "F15",
    note: "The identity-pair value is NOT a defect: the slot is \
           1 − saturate(mean grad_src_mag), a property of the reference, and on \
           the Rev4 bank's 88 identical keys it equals every same-reference \
           sibling bit for bit (2,376 cells; REVIEW_PARTB 2026-09-25). The earlier \
           premise that it should read 0 there was wrong. The defect is the other \
           artifact: a v1-only 944 walk reads exactly 1.0 from zeroed \
           accumulators, a value emitted without its inputs.",
};

/// **THE one owner of "does this slot use a pooled 4th/8th root?"** — derived
/// from the slot's own [`Statistic`], never from a hand-kept list.
///
/// `L4` is `(sum x^4/n)^(1/4)` and `L8` is `(sum x^8/n)^(1/8)`; every other v1
/// statistic is a mean, a max, a ratio or an `L2` (`sqrt`, correctly rounded
/// by IEEE-754). So the registry's answer to "which slots does `v1detroot`
/// move?" is a property of the statistic and cannot drift away from the
/// kernels — which is the failure mode a second list would have.
const fn uses_pooled_root(statistic: Statistic) -> bool {
    matches!(statistic, Statistic::L4 | Statistic::L8)
}

const fn v1(
    family: ComputeToken,
    block_local: u16,
    name: &'static str,
    statistic: Statistic,
    kernel: KernelId,
) -> SignalDef {
    SignalDef {
        family: Family::Supported(family),
        block_local,
        name,
        statistic,
        cost: CostClass::Cheap,
        tranche: Tranche::None,
        placement: Placement::AllCells,
        // Every v1 slot is a difference form: the 372 identity probe is the
        // ZERO vector on all 38 dial-grid references, measured
        // (benchmarks/dial_addressability_gate_2026-09-04.md §15).
        form: Form::Difference,
        direction: Direction::HigherIsWorse,
        kernel,
        deprecated: false,
        // F18: an `L4`/`L8` slot with no other defect carries the libc
        // exposure alone. F4 slots are handled by `v1_defect`, which keeps
        // `Defect::F4` (the larger one) in the single `defect` field and
        // records F18 through the revision list — `SignalDef` holds ONE
        // defect, so a slot in two defects names the bigger and registers
        // both eras.
        defect: if uses_pooled_root(statistic) {
            Some(DEFECT_F18)
        } else {
            None
        },
        revisions: if uses_pooled_root(statistic) {
            REV_DETROOT_PROPOSED
        } else {
            REV_OPTION_C
        },
    }
}

/// [`v1`] with an explicit revision list — for slots whose era membership is
/// not implied by their statistic (the fused-extension pools). The F18
/// defect is still derived from the statistic exactly as [`v1`] derives it.
const fn v1_rev(
    family: ComputeToken,
    block_local: u16,
    name: &'static str,
    statistic: Statistic,
    kernel: KernelId,
    revisions: &'static [Revision],
) -> SignalDef {
    SignalDef {
        family: Family::Supported(family),
        block_local,
        name,
        statistic,
        cost: CostClass::Cheap,
        tranche: Tranche::None,
        placement: Placement::AllCells,
        form: Form::Difference,
        direction: Direction::HigherIsWorse,
        kernel,
        deprecated: false,
        defect: if uses_pooled_root(statistic) {
            Some(DEFECT_F18)
        } else {
            None
        },
        revisions,
    }
}

/// A v1 signal carrying a named defect and its proposed revision.
const fn v1_with_defect(
    family: ComputeToken,
    block_local: u16,
    name: &'static str,
    statistic: Statistic,
    kernel: KernelId,
    defect: Defect,
    revisions: &'static [Revision],
) -> SignalDef {
    SignalDef {
        family: Family::Supported(family),
        block_local,
        name,
        statistic,
        cost: CostClass::Cheap,
        tranche: Tranche::None,
        placement: Placement::AllCells,
        form: Form::Difference,
        direction: Direction::HigherIsWorse,
        kernel,
        deprecated: false,
        defect: Some(defect),
        revisions,
    }
}

/// A v1 signal carrying the F4 defect and its proposed revision.
const fn v1_defect(
    family: ComputeToken,
    block_local: u16,
    name: &'static str,
    statistic: Statistic,
    kernel: KernelId,
) -> SignalDef {
    v1_with_defect(
        family,
        block_local,
        name,
        statistic,
        kernel,
        DEFECT_F4,
        // An SSIM-derived `L4`/`L8` pool is in BOTH `v1ssimcap` (F4) and
        // `v1detroot` (F18): the same slot, two independent arithmetic
        // problems. The eras therefore OVERLAP, which is why
        // `rev2_carries_all_eras_*` checks a union and not a sum.
        if uses_pooled_root(statistic) {
            REV_F4_AND_DETROOT
        } else {
            REV_F4_PROPOSED
        },
    )
}

const fn v2sig(
    block_local: u16,
    name: &'static str,
    statistic: Statistic,
    form: Form,
    direction: Direction,
    kernel: KernelId,
    tranche: Tranche,
    defect: Option<Defect>,
) -> SignalDef {
    SignalDef {
        family: Family::Supported(ComputeToken::V2),
        block_local,
        name,
        statistic,
        cost: CostClass::Expensive,
        tranche,
        placement: Placement::AllCells,
        form,
        direction,
        kernel,
        deprecated: false,
        // F18 reaches the v2 block too: `ssim_dev4` is `(M4/n)^(1/4)`, the
        // same `powf(0.25)` in three finalizers (`OnlineMoments::finish`, the
        // dense kernel, the append kernel). The caller's explicit `defect`
        // wins when it has one — a v2 slot with a named defect keeps it.
        defect: match defect {
            Some(d) => Some(d),
            None => {
                if uses_pooled_root(statistic) {
                    Some(DEFECT_F18)
                } else {
                    None
                }
            }
        },
        revisions: if uses_pooled_root(statistic) {
            REV_V2_DETROOT
        } else {
            NO_REV
        },
    }
}

/// [`v2sig`] with an explicit `revisions` list, for the v2 signals an era
/// names by KERNEL rather than by statistic (the SSIM-derived six).
#[allow(clippy::too_many_arguments)]
const fn v2sig_rev(
    block_local: u16,
    name: &'static str,
    statistic: Statistic,
    form: Form,
    direction: Direction,
    kernel: KernelId,
    tranche: Tranche,
    defect: Option<Defect>,
    revisions: &'static [Revision],
) -> SignalDef {
    let base = v2sig(
        block_local,
        name,
        statistic,
        form,
        direction,
        kernel,
        tranche,
        defect,
    );
    SignalDef { revisions, ..base }
}

const fn app(
    block_local: u16,
    name: &'static str,
    statistic: Statistic,
    form: Form,
    direction: Direction,
    kernel: KernelId,
    tranche: Tranche,
    placement: Placement,
    defect: Option<Defect>,
    revisions: &'static [Revision],
) -> SignalDef {
    SignalDef {
        family: Family::Supported(ComputeToken::Append),
        block_local,
        name,
        statistic,
        cost: CostClass::Expensive,
        tranche,
        placement,
        form,
        direction,
        kernel,
        deprecated: false,
        defect,
        revisions,
    }
}

/// The 13 v1 basic signals per (scale, channel), in `V1BasicSums::finalize_into`
/// order. Names follow the in-tree diagnostic array in `streaming.rs`.
pub(crate) static BASIC: [SignalDef; 13] = {
    use ComputeToken::Basic as F;
    use KernelId::V1Fused as K;
    use Statistic::{L2, L4, Mean, Ratio};
    [
        // F4 reaches EVERY slot built from the per-pixel SSIM dissimilarity,
        // not only the masked and IW blocks the audit named — MEASURED, see
        // `f4_moves_exactly_the_registered_slots`.
        v1_defect(F, 0, "ssim_mean", Mean, K),
        v1_defect(F, 1, "ssim_4th", L4, K),
        v1_defect(F, 2, "ssim_2nd", L2, K),
        v1(F, 3, "edge_art_mean", Mean, K),
        v1(F, 4, "edge_art_4th", L4, K),
        v1(F, 5, "edge_art_2nd", L2, K),
        v1(F, 6, "edge_det_mean", Mean, K),
        v1(F, 7, "edge_det_4th", L4, K),
        v1(F, 8, "edge_det_2nd", L2, K),
        v1(F, 9, "mse", Mean, K),
        // The HF-energy ratio family. The two `loss` members are bounded by
        // construction (their numerator is `max(0, src - dst)`, which their
        // own denominator bounds); the `gain` member's is `max(0, dst - src)`,
        // which it does not — that asymmetry IS F17, and it is why only the
        // third of these three carries a defect.
        v1(F, 10, "var_loss", Ratio, K),
        v1(F, 11, "tex_loss", Ratio, K),
        v1_with_defect(
            F,
            12,
            "contrast_inc",
            Ratio,
            K,
            DEFECT_F17,
            REV_F17_PROPOSED,
        ),
    ]
};

/// The 6 v1 peak signals per (scale, channel), in `finalize_pools_into` order.
pub(crate) static PEAKS: [SignalDef; 6] = {
    use ComputeToken::Peaks as F;
    use KernelId::V1Peaks as K;
    use Statistic::{L8, Max};
    [
        // Same source expression as BASIC's ssim_* — see the F4 note there.
        v1_defect(F, 0, "ssim_max", Max, K),
        v1(F, 1, "edge_art_max", Max, K),
        v1(F, 2, "edge_det_max", Max, K),
        v1_defect(F, 3, "ssim_l8", L8, K),
        v1(F, 4, "edge_art_l8", L8, K),
        v1(F, 5, "edge_det_l8", L8, K),
    ]
};

/// The 6 v1 masked signals per (scale, channel).
pub(crate) static MASKED: [SignalDef; 6] = {
    use ComputeToken::Masked as F;
    use KernelId::V1MaskIw as K;
    use Statistic::{L2, L4, Mean};
    [
        // F4: the per-pixel SSIM dissimilarity these three pool is uncapped
        // and carries no C1. f241 (masked ssim_4th) and f313 (iw ssim_4th)
        // are the two worst slots the defect audit scanned.
        v1_defect(F, 0, "ssim_mean", Mean, K),
        v1_defect(F, 1, "ssim_4th", L4, K),
        v1_defect(F, 2, "ssim_2nd", L2, K),
        v1_rev(F, 3, "edge_art_4th", L4, K, REV_DETROOT_AND_EXT_FUSED),
        v1_rev(F, 4, "edge_det_4th", L4, K, REV_DETROOT_AND_EXT_FUSED),
        v1_rev(F, 5, "mse", Mean, K, REV_OPTION_C_AND_EXT_FUSED),
    ]
};

/// The 6 v1 IW signals per (scale, channel) — same shape as [`MASKED`],
/// opposite pooling polarity.
pub(crate) static IW: [SignalDef; 6] = {
    use ComputeToken::Iw as F;
    use KernelId::V1MaskIw as K;
    use Statistic::{L2, L4, Mean};
    [
        // F4: the per-pixel SSIM dissimilarity these three pool is uncapped
        // and carries no C1. f241 (masked ssim_4th) and f313 (iw ssim_4th)
        // are the two worst slots the defect audit scanned.
        v1_defect(F, 0, "ssim_mean", Mean, K),
        v1_defect(F, 1, "ssim_4th", L4, K),
        v1_defect(F, 2, "ssim_2nd", L2, K),
        v1_rev(F, 3, "edge_art_4th", L4, K, REV_DETROOT_AND_EXT_FUSED),
        v1_rev(F, 4, "edge_det_4th", L4, K, REV_DETROOT_AND_EXT_FUSED),
        v1_rev(F, 5, "mse", Mean, K, REV_OPTION_C_AND_EXT_FUSED),
    ]
};

/// The 29 v2 signals per (scale, channel), index-aligned with
/// `feature_v2::idx`.
pub(crate) static V2: [SignalDef; 29] = {
    use Direction::{HigherIsWorse, Unsigned};
    use Form::{Difference, ReferenceOnly};
    use KernelId::{V2Dense, V2Gradient};
    use Statistic::{L2, L4, Mean, WeightedMean};
    [
        // Bounded-basic block (idx 0..8).
        v2sig_rev(
            0,
            "ssim_mean",
            Mean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
            REV_V2_SSIM,
        ),
        v2sig_rev(
            1,
            "ssim_dev2",
            L2,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
            REV_V2_SSIM,
        ),
        v2sig_rev(
            2,
            "ssim_dev4",
            L4,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
            REV_V2_SSIM_AND_DETROOT,
        ),
        v2sig(
            3,
            "art",
            Mean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        v2sig(
            4,
            "det",
            Mean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        // Class C: harvestable from the fused kernel's saturating-MSE sum,
        // despite living in the most expensive block in the layout.
        v2sig(
            5,
            "mse",
            Mean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::ClassC,
            None,
        ),
        v2sig(
            6,
            "hf_gain",
            Mean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        v2sig(
            7,
            "hf_loss",
            Mean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        v2sig(
            8,
            "hf_mag_loss",
            Mean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        // Soft-saliency peak block (idx 9..11).
        v2sig_rev(
            9,
            "ssim_soft_peak",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
            REV_V2_SSIM,
        ),
        v2sig(
            10,
            "art_soft_peak",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        v2sig(
            11,
            "det_soft_peak",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        // Masked block (idx 12..15).
        v2sig_rev(
            12,
            "masked_ssim",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
            REV_V2_SSIM,
        ),
        v2sig(
            13,
            "masked_art",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        v2sig(
            14,
            "masked_det",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        v2sig(
            15,
            "masked_mse",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        // IW block (idx 16..19).
        v2sig_rev(
            16,
            "iw_ssim",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
            REV_V2_SSIM,
        ),
        v2sig(
            17,
            "iw_art",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        v2sig(
            18,
            "iw_det",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        v2sig(
            19,
            "iw_mse",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        // Near-threshold core (idx 20..21). FRAGILITY is reference-only
        // masking-susceptibility — see `idx::PJND_FRAGILITY`'s own doc.
        v2sig(
            20,
            "pjnd_transducer",
            Mean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        // F15: ReferenceOnly, and its nonzero identity-pair value is correct
        // (it equals the same-reference siblings). The defect is the 1.0 a
        // v1-only walk reads from zeroed accumulators.
        v2sig(
            21,
            "pjnd_fragility",
            Mean,
            ReferenceOnly,
            Unsigned,
            V2Dense,
            Tranche::None,
            Some(DEFECT_F15),
        ),
        // Phase-2 additions (idx 22..28).
        v2sig(
            22,
            "gms",
            Mean,
            Difference,
            HigherIsWorse,
            V2Gradient,
            Tranche::None,
            None,
        ),
        v2sig(
            23,
            "pjnd_transducer_low_k",
            Mean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        v2sig(
            24,
            "pjnd_transducer_high_k",
            Mean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        v2sig(
            25,
            "blockiness",
            Mean,
            Difference,
            HigherIsWorse,
            V2Gradient,
            Tranche::None,
            None,
        ),
        v2sig_rev(
            26,
            "ringing",
            Mean,
            Difference,
            HigherIsWorse,
            V2Gradient,
            Tranche::None,
            Some(DEFECT_RING_GLOBAL),
            RINGING_REVS,
        ),
        v2sig(
            27,
            "banding",
            Mean,
            Difference,
            HigherIsWorse,
            V2Gradient,
            Tranche::None,
            None,
        ),
        // The one scale-level (not per-pixel) signal in the set — see
        // `idx::EDGE_WIDTH_CHANGE`'s doc and the v2 spec.
        v2sig(
            28,
            "edge_width_change",
            Mean,
            Difference,
            HigherIsWorse,
            V2Gradient,
            Tranche::None,
            None,
        ),
    ]
};

/// The 17 append signals per (scale, channel), index-aligned with
/// `feature_v2::idx_append`.
///
/// The three `GLOBAL_*` and `GRAD_SRC_MEAN` are the tranche a v1-only walk can
/// finalize from raw moments — [`CostClass::Free`] — which is what makes the
/// `basic+peaks+moments` plan reachable without the append kernel.
pub(crate) static APPEND: [SignalDef; 17] = {
    use Direction::{HigherIsWorse, Unsigned};
    use Form::{Difference, ReferenceOnly};
    use KernelId::{Append as KA, FreeBoundedErr as KB, FreeRawMoments as KR};
    use Placement::{AllCells, SkipBScale0, YOnlyActive};
    use Statistic::{Bin, Global, L2, Mean};
    use Tranche::{ClassC, None as NoTranche, RawMoments};
    [
        app(
            0,
            "xmask_transducer",
            Mean,
            Difference,
            HigherIsWorse,
            KA,
            NoTranche,
            AllCells,
            None,
            NO_REV,
        ),
        app(
            1,
            "lum_transducer",
            Mean,
            Difference,
            HigherIsWorse,
            KA,
            NoTranche,
            AllCells,
            None,
            NO_REV,
        ),
        // The three luminance-binned error means are the class-C tranche, and
        // ONLY at Y: the register-carry constraint the free accumulator works
        // under has no room for the chroma pair.
        app(
            2,
            "lum_dark_err",
            Bin,
            Difference,
            HigherIsWorse,
            KB,
            ClassC,
            YOnlyActive,
            None,
            NO_REV,
        ),
        app(
            3,
            "lum_mid_err",
            Bin,
            Difference,
            HigherIsWorse,
            KB,
            ClassC,
            YOnlyActive,
            None,
            NO_REV,
        ),
        app(
            4,
            "lum_bright_err",
            Bin,
            Difference,
            HigherIsWorse,
            KB,
            ClassC,
            YOnlyActive,
            None,
            NO_REV,
        ),
        app(
            5,
            "mscn_diff_mean",
            Mean,
            Difference,
            HigherIsWorse,
            KA,
            NoTranche,
            AllCells,
            None,
            NO_REV,
        ),
        app(
            6,
            "mscn_diff_l2",
            L2,
            Difference,
            HigherIsWorse,
            KA,
            NoTranche,
            AllCells,
            None,
            NO_REV,
        ),
        app(
            7,
            "contrast_gain",
            Mean,
            Difference,
            HigherIsWorse,
            KA,
            NoTranche,
            AllCells,
            None,
            NO_REV,
        ),
        app(
            8,
            "contrast_loss",
            Mean,
            Difference,
            HigherIsWorse,
            KA,
            NoTranche,
            AllCells,
            None,
            NO_REV,
        ),
        app(
            9,
            "texture_dissim",
            Mean,
            Difference,
            HigherIsWorse,
            KA,
            NoTranche,
            AllCells,
            None,
            NO_REV,
        ),
        app(
            10,
            "gms_dev2",
            L2,
            Difference,
            HigherIsWorse,
            KA,
            NoTranche,
            AllCells,
            None,
            NO_REV,
        ),
        app(
            11,
            "art_dev2",
            L2,
            Difference,
            HigherIsWorse,
            KA,
            NoTranche,
            AllCells,
            None,
            NO_REV,
        ),
        app(
            12,
            "det_dev2",
            L2,
            Difference,
            HigherIsWorse,
            KA,
            NoTranche,
            AllCells,
            None,
            NO_REV,
        ),
        // The raw-moment tranche — every append cell the 944 walk computes,
        // i.e. all but `(B, scale 0)` (`APPEND_SKIP_B_SCALE0`).
        app(
            13,
            "global_dmean",
            Global,
            Difference,
            HigherIsWorse,
            KR,
            RawMoments,
            SkipBScale0,
            Some(DEFECT_F5),
            REV_F5_PROPOSED,
        ),
        app(
            14,
            "global_cgain",
            Global,
            Difference,
            HigherIsWorse,
            KR,
            RawMoments,
            SkipBScale0,
            Some(DEFECT_F5),
            REV_F5_PROPOSED,
        ),
        app(
            15,
            "global_closs",
            Global,
            Difference,
            HigherIsWorse,
            KR,
            RawMoments,
            SkipBScale0,
            Some(DEFECT_F5),
            REV_F5_PROPOSED,
        ),
        app(
            16,
            "grad_src_mean",
            Mean,
            ReferenceOnly,
            Unsigned,
            KA,
            NoTranche,
            AllCells,
            None,
            NO_REV,
        ),
    ]
};

/// The 5 append2 signals per SCALE (Y-only), index-aligned with
/// `feature_v2::idx_append2`.
pub(crate) static APPEND2: [SignalDef; 5] = {
    use ComputeToken::Append2 as F;
    use Direction::{HigherIsWorse, Unsigned};
    use Form::{Difference, ReferenceOnly};
    use KernelId::{Append2 as KA, FreeRawMoments as KR};
    use Statistic::{Bin, Mean};
    const fn a2(
        block_local: u16,
        name: &'static str,
        statistic: Statistic,
        form: Form,
        direction: Direction,
        kernel: KernelId,
        tranche: Tranche,
        defect: Option<Defect>,
    ) -> SignalDef {
        SignalDef {
            family: Family::Supported(F),
            block_local,
            name,
            statistic,
            cost: CostClass::Expensive,
            tranche,
            // append2 is a per-SCALE block, so there is no channel axis to
            // restrict: every cell it has is a Y cell.
            placement: Placement::AllCells,
            form,
            direction,
            kernel,
            deprecated: false,
            defect,
            revisions: NO_REV,
        }
    }
    [
        a2(
            0,
            "bandvis_gain",
            Mean,
            Difference,
            HigherIsWorse,
            KA,
            Tranche::None,
            None,
        ),
        a2(
            1,
            "bandvis_loss",
            Mean,
            Difference,
            HigherIsWorse,
            KA,
            Tranche::None,
            None,
        ),
        // THE slot that makes the 944 identity probe non-zero: a property of
        // the reference alone, so a perfect copy leaves it at its reference
        // value. Free from the raw moments a v1-only walk already has.
        a2(
            2,
            "luma_mean_ref",
            Mean,
            ReferenceOnly,
            Unsigned,
            KR,
            Tranche::RawMoments,
            Some(DEFECT_F5),
        ),
        a2(
            3,
            "hl_bin1",
            Bin,
            Difference,
            HigherIsWorse,
            KA,
            Tranche::None,
            None,
        ),
        a2(
            4,
            "hl_bin2",
            Bin,
            Difference,
            HigherIsWorse,
            KA,
            Tranche::None,
            None,
        ),
    ]
};

/// The 3 CSFW tier-1 signals per SCALE (Y-only), index-aligned with
/// `feature_v2::idx_csfw`.
pub(crate) static CSFW: [SignalDef; 3] = {
    use ComputeToken::Csfw as F;
    use Direction::HigherIsWorse;
    use Form::Difference;
    use KernelId::Csfw as K;
    use Statistic::Global;
    const fn cs(block_local: u16, name: &'static str) -> SignalDef {
        SignalDef {
            family: Family::Supported(F),
            block_local,
            name,
            statistic: Global,
            cost: CostClass::Expensive,
            tranche: Tranche::None,
            placement: Placement::AllCells,
            form: Difference,
            direction: HigherIsWorse,
            kernel: K,
            deprecated: false,
            defect: None,
            revisions: NO_REV,
        }
    }
    [
        cs(0, "w_global_dmean"),
        cs(1, "w_global_cgain"),
        cs(2, "w_global_closs"),
    ]
};

// ============================================================================
// Rev4 candidate feature bank (f986..1321 at 4 scales)
// ============================================================================
//
// Spec: `docs/REV4_FEATURE_BANK_PLAN_2026-09-23.md` §1.1 / §2.4–2.5; pinned
// geometry: `benchmarks/rev4_featbank_impl_2026-09-23.md`. All four families
// are difference-form by construction (identity pair → 0.0), carry no
// tranche (a v1-only walk cannot harvest any of them — they live on the
// v2-era passes), and declare `Placement::AllCells`: gridblk's `(Y, scale 3)`
// cell is registered-with-stated-zero (period 1 has no phase contrast), not
// skipped, so the slot exists and emits 0.0 deliberately.

/// The 8 gridblk signals per (scale, channel) cell — the C1 phase-aligned
/// blocking family. `mag_bin*` are SIGNED mass bins of the activity-scaled
/// boundary step excess (`ẽ` may be negative = suppression), so they are
/// `Unsigned`; `on_mean` is signed for the same reason; `onoff_ratio` is a
/// magnitude ratio, `HigherIsWorse`.
pub(crate) static GRIDBLK: [SignalDef; 8] = {
    use ComputeToken::Gridblk as F;
    use Form::Difference;
    use KernelId::Gridblk as K;
    const fn gb(
        block_local: u16,
        name: &'static str,
        statistic: Statistic,
        direction: Direction,
    ) -> SignalDef {
        SignalDef {
            family: Family::Supported(F),
            block_local,
            name,
            statistic,
            cost: CostClass::Expensive,
            tranche: Tranche::None,
            placement: Placement::AllCells,
            form: Difference,
            direction,
            kernel: K,
            deprecated: false,
            defect: None,
            revisions: REV4BANK,
        }
    }
    [
        gb(0, "mag_bin1", Statistic::Bin, Direction::Unsigned),
        gb(1, "mag_bin2", Statistic::Bin, Direction::Unsigned),
        gb(2, "mag_bin3", Statistic::Bin, Direction::Unsigned),
        gb(3, "mag_bin4", Statistic::Bin, Direction::Unsigned),
        gb(4, "mag_bin5", Statistic::Bin, Direction::Unsigned),
        gb(5, "mag_bin6", Statistic::Bin, Direction::Unsigned),
        gb(6, "on_mean", Statistic::Mean, Direction::Unsigned),
        gb(7, "onoff_ratio", Statistic::Ratio, Direction::HigherIsWorse),
    ]
};

/// The 6 ringbasis signals per (scale, channel) cell — the C2
/// ringing-magnitude basis. Triangular log-axis bins of the gradient
/// kernel's own per-pixel ringing term; every bin is a nonnegative pooled
/// mass, so all are `HigherIsWorse`.
pub(crate) static RINGBASIS: [SignalDef; 6] = {
    use ComputeToken::Ringbasis as F;
    use Direction::HigherIsWorse;
    use Form::Difference;
    use KernelId::V2Gradient as K;
    use Statistic::Bin;
    const fn rb(block_local: u16, name: &'static str) -> SignalDef {
        SignalDef {
            family: Family::Supported(F),
            block_local,
            name,
            statistic: Bin,
            cost: CostClass::Expensive,
            tranche: Tranche::None,
            placement: Placement::AllCells,
            form: Difference,
            direction: HigherIsWorse,
            kernel: K,
            deprecated: false,
            defect: None,
            revisions: REV4BANK,
        }
    }
    [
        rb(0, "mag_bin1"),
        rb(1, "mag_bin2"),
        rb(2, "mag_bin3"),
        rb(3, "mag_bin4"),
        rb(4, "mag_bin5"),
        rb(5, "mag_bin6"),
    ]
};

/// The 12 tailhist signals per (scale, channel) cell — the C3 tail profile
/// of the four existing dense per-pixel maps (`ssim` dissimilarity,
/// `art`, `det`, `mse`). `p95`/`p99` are read off the family's 32-bin
/// log-edged histogram (a bin statistic); `max` is the exact map max.
pub(crate) static TAILHIST: [SignalDef; 12] = {
    use ComputeToken::Tailhist as F;
    use Direction::HigherIsWorse;
    use Form::Difference;
    use KernelId::V2Dense as K;
    const fn th(
        block_local: u16,
        name: &'static str,
        statistic: Statistic,
        revisions: &'static [Revision],
    ) -> SignalDef {
        SignalDef {
            family: Family::Supported(F),
            block_local,
            name,
            statistic,
            cost: CostClass::Expensive,
            tranche: Tranche::None,
            placement: Placement::AllCells,
            form: Difference,
            direction: HigherIsWorse,
            kernel: K,
            deprecated: false,
            defect: None,
            revisions,
        }
    }
    [
        th(0, "ssim_p95", Statistic::Bin, TAILHIST_BIN_REVS),
        th(1, "ssim_p99", Statistic::Bin, TAILHIST_BIN_REVS),
        th(2, "ssim_max", Statistic::Max, REV4BANK),
        th(3, "art_p95", Statistic::Bin, TAILHIST_BIN_REVS),
        th(4, "art_p99", Statistic::Bin, TAILHIST_BIN_REVS),
        th(5, "art_max", Statistic::Max, REV4BANK),
        th(6, "det_p95", Statistic::Bin, TAILHIST_BIN_REVS),
        th(7, "det_p99", Statistic::Bin, TAILHIST_BIN_REVS),
        th(8, "det_max", Statistic::Max, REV4BANK),
        th(9, "mse_p95", Statistic::Bin, TAILHIST_BIN_REVS),
        th(10, "mse_p99", Statistic::Bin, TAILHIST_BIN_REVS),
        th(11, "mse_max", Statistic::Max, REV4BANK),
    ]
};

/// The 6 arttype signals per SCALE (channel `Scalar`) — the C4 artifact-type
/// descriptors: blur (EWC × HF_LOSS product at that scale), flat-region HF
/// noise per channel, and chroma bleed outside the dilated luma-edge mask.
pub(crate) static ARTTYPE: [SignalDef; 6] = {
    use ComputeToken::Arttype as F;
    use Direction::HigherIsWorse;
    use Form::Difference;
    const fn at(
        block_local: u16,
        name: &'static str,
        statistic: Statistic,
        kernel: KernelId,
    ) -> SignalDef {
        SignalDef {
            family: Family::Supported(F),
            block_local,
            name,
            statistic,
            cost: CostClass::Expensive,
            tranche: Tranche::None,
            placement: Placement::AllCells,
            form: Difference,
            direction: HigherIsWorse,
            kernel,
            deprecated: false,
            defect: None,
            revisions: REV4BANK,
        }
    }
    [
        at(0, "blur", Statistic::Global, KernelId::V2Dense),
        at(1, "noise_x", Statistic::Mean, KernelId::V2Dense),
        at(2, "noise_y", Statistic::Mean, KernelId::V2Dense),
        at(3, "noise_b", Statistic::Mean, KernelId::V2Dense),
        at(4, "bleed_x", Statistic::Ratio, KernelId::V2Gradient),
        at(5, "bleed_b", Statistic::Ratio, KernelId::V2Gradient),
    ]
};

/// C8: sparse native-Y/coarse-chroma gradient and chromaticity signals.
pub(crate) static GMSBANK_SIGNALS: [SignalDef; 25] = {
    use ComputeToken::Gmsbank as F;
    use Direction::HigherIsWorse;
    use Form::Difference;
    use KernelId::V2Gradient as K;
    const fn gb(block_local: u16, name: &'static str, statistic: Statistic) -> SignalDef {
        SignalDef {
            family: Family::Supported(F),
            block_local,
            name,
            statistic,
            cost: CostClass::Expensive,
            tranche: Tranche::None,
            placement: Placement::AllCells,
            form: Difference,
            direction: HigherIsWorse,
            kernel: K,
            deprecated: false,
            defect: None,
            revisions: GMSBANK,
        }
    }
    [
        gb(0, "loss0", Statistic::Mean),
        gb(1, "gain0", Statistic::Mean),
        gb(2, "dev0", Statistic::Global),
        gb(3, "loss1", Statistic::Mean),
        gb(4, "gain1", Statistic::Mean),
        gb(5, "dev1", Statistic::Global),
        gb(6, "loss2", Statistic::Mean),
        gb(7, "gain2", Statistic::Mean),
        gb(8, "dev2", Statistic::Global),
        gb(9, "loss3", Statistic::Mean),
        gb(10, "gain3", Statistic::Mean),
        gb(11, "dev3", Statistic::Global),
        gb(12, "loss4", Statistic::Mean),
        gb(13, "gain4", Statistic::Mean),
        gb(14, "dev4", Statistic::Global),
        gb(15, "cs_loss0", Statistic::Mean),
        gb(16, "cs_dev0", Statistic::Global),
        gb(17, "cs_loss1", Statistic::Mean),
        gb(18, "cs_dev1", Statistic::Global),
        gb(19, "cs_loss2", Statistic::Mean),
        gb(20, "cs_dev2", Statistic::Global),
        gb(21, "cs_loss3", Statistic::Mean),
        gb(22, "cs_dev3", Statistic::Global),
        gb(23, "cs_loss4", Statistic::Mean),
        gb(24, "cs_dev4", Statistic::Global),
    ]
};

/// A1 (restored): standard deviation of five per-pixel maps of the v1 band
/// kernel per (scale, channel). `mse_dev` is exactly zero on an identity pair;
/// the four HF maps are reference/distorted properties, so on an identity
/// pair `hfsq_src_dev == hfsq_dst_dev` and neither is zero.
pub(crate) static MAPDEV_SIGNALS: [SignalDef; 5] = {
    use ComputeToken::Mapdev as F;
    use Statistic::Global;
    const fn md(
        block_local: u16,
        name: &'static str,
        form: Form,
        direction: Direction,
    ) -> SignalDef {
        SignalDef {
            family: Family::Supported(F),
            block_local,
            name,
            statistic: Global,
            cost: CostClass::Expensive,
            tranche: Tranche::None,
            placement: Placement::AllCells,
            form,
            direction,
            kernel: KernelId::RestoreMaps,
            deprecated: false,
            defect: None,
            revisions: MAPDEV,
        }
    }
    [
        md(0, "mse_dev", Form::Difference, Direction::HigherIsWorse),
        md(1, "hfsq_src_dev", Form::ReferenceOnly, Direction::Unsigned),
        md(2, "hfsq_dst_dev", Form::Undeclared, Direction::Unsigned),
        md(3, "hfabs_src_dev", Form::ReferenceOnly, Direction::Unsigned),
        md(4, "hfabs_dst_dev", Form::Undeclared, Direction::Unsigned),
    ]
};

/// B2 (restored): the 13 basic + 6 peak v1 signals pooled over ungated 5x5
/// block maxima, in `BASIC` then `PEAKS` order.
pub(crate) static Z1MAX_SIGNALS: [SignalDef; 19] = {
    use ComputeToken::Z1max as F;
    use Statistic::{BlockL4 as L4, BlockL8 as L8, L2, Max, Mean, Ratio};
    const fn zm(
        block_local: u16,
        name: &'static str,
        statistic: Statistic,
        form: Form,
    ) -> SignalDef {
        SignalDef {
            family: Family::Supported(F),
            block_local,
            name,
            statistic,
            cost: CostClass::Expensive,
            tranche: Tranche::None,
            placement: Placement::AllCells,
            form,
            direction: Direction::HigherIsWorse,
            kernel: KernelId::RestoreMaps,
            deprecated: false,
            defect: None,
            revisions: Z1MAX,
        }
    }
    use Form::{Difference as D, Undeclared as U};
    [
        zm(0, "ssim_mean", Mean, U),
        zm(1, "ssim_4th", L4, U),
        zm(2, "ssim_2nd", L2, U),
        zm(3, "edge_art_mean", Mean, D),
        zm(4, "edge_art_4th", L4, D),
        zm(5, "edge_art_2nd", L2, D),
        zm(6, "edge_det_mean", Mean, D),
        zm(7, "edge_det_4th", L4, D),
        zm(8, "edge_det_2nd", L2, D),
        zm(9, "mse", Mean, D),
        zm(10, "var_loss", Ratio, D),
        zm(11, "tex_loss", Ratio, D),
        zm(12, "contrast_inc", Ratio, D),
        zm(13, "ssim_max", Max, U),
        zm(14, "edge_art_max", Max, D),
        zm(15, "edge_det_max", Max, D),
        zm(16, "ssim_l8", L8, U),
        zm(17, "edge_art_l8", L8, D),
        zm(18, "edge_det_l8", L8, D),
    ]
};

/// Ambiguous 7 (restored): C8's per-channel gradient bank at native scale for
/// the X and B channels (15 slots each), identical to the coarse X/B cells'
/// definition.
pub(crate) static GMSNATIVE_SIGNALS: [SignalDef; 15] = {
    use ComputeToken::Gmsnative as F;
    use Direction::HigherIsWorse;
    use Form::Difference;
    use KernelId::V2Gradient as K;
    const fn gn(block_local: u16, name: &'static str, statistic: Statistic) -> SignalDef {
        SignalDef {
            family: Family::Supported(F),
            block_local,
            name,
            statistic,
            cost: CostClass::Expensive,
            tranche: Tranche::None,
            placement: Placement::AllCells,
            form: Difference,
            direction: HigherIsWorse,
            kernel: K,
            deprecated: false,
            defect: None,
            revisions: GMSNATIVE,
        }
    }
    [
        gn(0, "loss0", Statistic::Mean),
        gn(1, "gain0", Statistic::Mean),
        gn(2, "dev0", Statistic::Global),
        gn(3, "loss1", Statistic::Mean),
        gn(4, "gain1", Statistic::Mean),
        gn(5, "dev1", Statistic::Global),
        gn(6, "loss2", Statistic::Mean),
        gn(7, "gain2", Statistic::Mean),
        gn(8, "dev2", Statistic::Global),
        gn(9, "loss3", Statistic::Mean),
        gn(10, "gain3", Statistic::Mean),
        gn(11, "dev3", Statistic::Global),
        gn(12, "loss4", Statistic::Mean),
        gn(13, "gain4", Statistic::Mean),
        gn(14, "dev4", Statistic::Global),
    ]
};

// Palette IDs are explicit N-major placements, with their own research semantics.
const PALETTE_REVISION: &[Revision] = &[
    Revision {
        era: "palette_v1",
        commit: "8a4f8314",
        status: RevisionStatus::Landed,
        note: "Initial research producer; superseded after a signed-population regression. Retained as historical provenance.",
    },
    Revision {
        era: "palette_v2",
        commit: "f9a38a33",
        status: RevisionStatus::Landed,
        note: "Signed lightness/chroma compare independently population-weighted means; producer BUILD_COMMIT pins the implementation. Existing families unchanged.",
    },
];
const fn palette_signal(block_local: u16, name: &'static str, direction: Direction) -> SignalDef {
    SignalDef {
        family: Family::Palette,
        block_local,
        name,
        statistic: Statistic::Global,
        cost: CostClass::Expensive,
        tranche: Tranche::None,
        placement: Placement::AllCells,
        form: Form::Difference,
        direction,
        kernel: KernelId::Palette,
        deprecated: false,
        defect: None,
        revisions: PALETTE_REVISION,
    }
}
pub(crate) static PALETTE_SIGNALS: [SignalDef; 42] = [
    palette_signal(0, "n2_shift", Direction::HigherIsWorse),
    palette_signal(1, "n2_lightness_signed", Direction::Undeclared),
    palette_signal(2, "n2_chroma_signed", Direction::Undeclared),
    palette_signal(3, "n2_hue_signed", Direction::Undeclared),
    palette_signal(4, "n2_weight_emd", Direction::HigherIsWorse),
    palette_signal(5, "n2_largest_shift", Direction::HigherIsWorse),
    palette_signal(6, "n3_shift", Direction::HigherIsWorse),
    palette_signal(7, "n3_lightness_signed", Direction::Undeclared),
    palette_signal(8, "n3_chroma_signed", Direction::Undeclared),
    palette_signal(9, "n3_hue_signed", Direction::Undeclared),
    palette_signal(10, "n3_weight_emd", Direction::HigherIsWorse),
    palette_signal(11, "n3_largest_shift", Direction::HigherIsWorse),
    palette_signal(12, "n4_shift", Direction::HigherIsWorse),
    palette_signal(13, "n4_lightness_signed", Direction::Undeclared),
    palette_signal(14, "n4_chroma_signed", Direction::Undeclared),
    palette_signal(15, "n4_hue_signed", Direction::Undeclared),
    palette_signal(16, "n4_weight_emd", Direction::HigherIsWorse),
    palette_signal(17, "n4_largest_shift", Direction::HigherIsWorse),
    palette_signal(18, "n5_shift", Direction::HigherIsWorse),
    palette_signal(19, "n5_lightness_signed", Direction::Undeclared),
    palette_signal(20, "n5_chroma_signed", Direction::Undeclared),
    palette_signal(21, "n5_hue_signed", Direction::Undeclared),
    palette_signal(22, "n5_weight_emd", Direction::HigherIsWorse),
    palette_signal(23, "n5_largest_shift", Direction::HigherIsWorse),
    palette_signal(24, "n6_shift", Direction::HigherIsWorse),
    palette_signal(25, "n6_lightness_signed", Direction::Undeclared),
    palette_signal(26, "n6_chroma_signed", Direction::Undeclared),
    palette_signal(27, "n6_hue_signed", Direction::Undeclared),
    palette_signal(28, "n6_weight_emd", Direction::HigherIsWorse),
    palette_signal(29, "n6_largest_shift", Direction::HigherIsWorse),
    palette_signal(30, "n7_shift", Direction::HigherIsWorse),
    palette_signal(31, "n7_lightness_signed", Direction::Undeclared),
    palette_signal(32, "n7_chroma_signed", Direction::Undeclared),
    palette_signal(33, "n7_hue_signed", Direction::Undeclared),
    palette_signal(34, "n7_weight_emd", Direction::HigherIsWorse),
    palette_signal(35, "n7_largest_shift", Direction::HigherIsWorse),
    palette_signal(36, "n8_shift", Direction::HigherIsWorse),
    palette_signal(37, "n8_lightness_signed", Direction::Undeclared),
    palette_signal(38, "n8_chroma_signed", Direction::Undeclared),
    palette_signal(39, "n8_hue_signed", Direction::Undeclared),
    palette_signal(40, "n8_weight_emd", Direction::HigherIsWorse),
    palette_signal(41, "n8_largest_shift", Direction::HigherIsWorse),
];

/// The `globaff` introduction (formula revision 6): an append-only family.
const GLOBAFF_REVISION: &[Revision] = &[Revision {
    era: "globaff",
    commit: "-",
    status: RevisionStatus::Proposed,
    note: "append-only introduction of the global affine-change descriptor, \
           f1867..1890, computed at formula revision 6 only: per scale and XYB \
           channel the share of the mean squared error a least-squares fit \
           dst = a*src + b explains, and that explained error, from five f64 \
           sums per cell accumulated over the walk's strips (`crate::globaff`). \
           It replaces the research sidecar zensim-bench/examples/byc_globaff.rs \
           (two-pass moments) at the same ids. No earlier slot moves.",
}];
const fn globaff_signal(block_local: u16, name: &'static str, direction: Direction) -> SignalDef {
    SignalDef {
        family: Family::Supported(ComputeToken::GlobAff),
        block_local,
        name,
        statistic: Statistic::Global,
        cost: CostClass::Cheap,
        tranche: Tranche::None,
        placement: Placement::AllCells,
        form: Form::Difference,
        direction,
        kernel: KernelId::GlobAff,
        deprecated: false,
        defect: None,
        revisions: GLOBAFF_REVISION,
    }
}
/// `share` (block-local 0) and `explained` (1) per (scale, channel), in the
/// sidecar's order `2·(3·scale + channel) + signal`.
pub(crate) static GLOBAFF_SIGNALS: [SignalDef; crate::globaff::GLOBAFF_PER_CELL] = [
    globaff_signal(0, "share", Direction::Unsigned),
    globaff_signal(1, "explained", Direction::HigherIsWorse),
];

// ============================================================================
// Layout arithmetic — THE owner
// ============================================================================

/// How a family is replicated across the (scale, channel) axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Replication {
    /// One cell per (scale, channel): 3 cells per scale.
    PerChannel,
    /// One cell per scale (the Y-only blocks).
    PerScale,
    /// family's own pyramid levels are its signal axis, not the walk's
    /// scales, so there is nothing to replicate). [`def_at`] reports
    /// `scale = 0, channel = Scalar` for every flat slot, which is also what
    /// `ComputeSet::at_scale` narrowing needs: a flat family's slots live
    /// and die with the scale-0 walk rows that feed them.
    Flat,
    /// Native Y gradients, then coarse X/Y/B gradients and joint chroma CS.
    GmsbankChroma,
    /// Two cells total, at scale 0: X then B (the native-scale chroma cells
    /// the coarse-only C8 layout omits).
    NativeXb,
}

/// A registered block: a signal table plus where it sits in the layout.
pub(crate) struct BlockDef {
    pub family: Family,
    pub signals: &'static [SignalDef],
    pub replication: Replication,
}

/// Every block, in LAYOUT order. The base offset of each is the sum of the
/// widths of the ones before it, so inserting a block anywhere but the end
/// would renumber — which the append-only directive forbids.
pub(crate) static BLOCKS: &[BlockDef] = &[
    BlockDef {
        family: Family::Supported(ComputeToken::Basic),
        signals: &BASIC,
        replication: Replication::PerChannel,
    },
    BlockDef {
        family: Family::Supported(ComputeToken::Peaks),
        signals: &PEAKS,
        replication: Replication::PerChannel,
    },
    BlockDef {
        family: Family::Supported(ComputeToken::Masked),
        signals: &MASKED,
        replication: Replication::PerChannel,
    },
    BlockDef {
        family: Family::Supported(ComputeToken::Iw),
        signals: &IW,
        replication: Replication::PerChannel,
    },
    BlockDef {
        family: Family::Supported(ComputeToken::V2),
        signals: &V2,
        replication: Replication::PerChannel,
    },
    BlockDef {
        family: Family::Supported(ComputeToken::Append),
        signals: &APPEND,
        replication: Replication::PerChannel,
    },
    BlockDef {
        family: Family::Supported(ComputeToken::Append2),
        signals: &APPEND2,
        replication: Replication::PerScale,
    },
    BlockDef {
        family: Family::Supported(ComputeToken::Csfw),
        signals: &CSFW,
        replication: Replication::PerScale,
    },
    BlockDef {
        family: Family::Supported(ComputeToken::Gridblk),
        signals: &GRIDBLK,
        replication: Replication::PerChannel,
    },
    BlockDef {
        family: Family::Supported(ComputeToken::Ringbasis),
        signals: &RINGBASIS,
        replication: Replication::PerChannel,
    },
    BlockDef {
        family: Family::Supported(ComputeToken::Tailhist),
        signals: &TAILHIST,
        replication: Replication::PerChannel,
    },
    BlockDef {
        family: Family::Supported(ComputeToken::Arttype),
        signals: &ARTTYPE,
        replication: Replication::PerScale,
    },
    BlockDef {
        family: Family::Supported(ComputeToken::Gmsbank),
        signals: &GMSBANK_SIGNALS,
        replication: Replication::GmsbankChroma,
    },
    BlockDef {
        family: Family::Supported(ComputeToken::Mapdev),
        signals: &MAPDEV_SIGNALS,
        replication: Replication::PerChannel,
    },
    BlockDef {
        family: Family::Supported(ComputeToken::Z1max),
        signals: &Z1MAX_SIGNALS,
        replication: Replication::PerChannel,
    },
    BlockDef {
        family: Family::Supported(ComputeToken::Gmsnative),
        signals: &GMSNATIVE_SIGNALS,
        replication: Replication::NativeXb,
    },
    BlockDef {
        family: Family::Palette,
        signals: &PALETTE_SIGNALS,
        replication: Replication::Flat,
    },
    BlockDef {
        family: Family::Supported(ComputeToken::GlobAff),
        signals: &GLOBAFF_SIGNALS,
        replication: Replication::PerChannel,
    },
];

impl BlockDef {
    /// Slots this block occupies at `n_scales`.
    pub(crate) fn width(&self, n_scales: usize) -> usize {
        let cells = match self.replication {
            Replication::PerChannel => n_scales * 3,
            Replication::PerScale => n_scales,
            Replication::Flat => 1,
            Replication::NativeXb => {
                return if n_scales == 0 {
                    0
                } else {
                    2 * self.signals.len()
                };
            }
            Replication::GmsbankChroma => {
                return if n_scales == 0 {
                    0
                } else {
                    15 + (n_scales - 1) * 55
                };
            }
        };
        cells * self.signals.len()
    }
}

/// Retired slots stay unassigned so later feature IDs keep their meaning.
fn reserved_before(family: Family) -> usize {
    match family {
        Family::Supported(ComputeToken::Gridblk) => 30,
        Family::Palette => 5,
        _ => 0,
    }
}

/// The base slot id of `family` at `n_scales`, and its block.
pub(crate) fn block_base(
    family: impl Into<Family>,
    n_scales: usize,
) -> Option<(usize, &'static BlockDef)> {
    let family = family.into();
    let mut base = 0usize;
    for b in BLOCKS {
        base += reserved_before(b.family);
        if b.family == family {
            return Some((base, b));
        }
        base += b.width(n_scales);
    }
    None
}

/// Every emitted-vector width a registered PRODUCER set has ever used —
/// the candidate clip widths [`crate::feature_layout::slots_of`] searches when
/// an id does not carry a legacy `@w<N>` component.
///
/// **This bounds a search; it is not an identity.** A feature-set id's identity
/// is its `slots_hash8`, so a width on this list can only ever CONFIRM a
/// reconstruction, never change which set an id names. An id whose sparse clip
/// width is not here is reported unreproducible by this build, which is the
/// same answer it would give for an unknown compute token — and the honest one,
/// because guessing would name a different set.
///
/// Sourced from `benchmarks/feature_sets_registry.json`'s `sets[].layout`
/// (append-only; 2026-09-19: 372, 720, 924, 944, 956, 986; 2026-09-23: the
/// Rev4 feature bank's full width 1322) plus C8's width 1502, the by-construction research widths 1891 (globaff),
/// 2301 (the gated globaff research set) and 2337 (the byc_orient research set) and the restored-cut families' widths
/// 1562 (mapdev),
/// `zensim-validate`'s
/// `every_registered_layout_width_is_a_candidate` holds the two in sync, so
/// registering a set at a new width fails the build rather than silently
/// becoming unreproducible.
pub(crate) const REGISTERED_LAYOUT_WIDTHS: &[usize] = &[
    372, 720, 924, 944, 956, 986, 1322, 1502, 1562, 1790, 1820, 1825, 1867, 1891, 2301, 2337,
];

/// Total layout width at `n_scales` with every registered block present.
pub(crate) fn full_width(n_scales: usize) -> usize {
    BLOCKS
        .iter()
        .map(|b| reserved_before(b.family) + b.width(n_scales))
        .sum()
}

/// The slot id of one signal placement.
///
/// `channel` is ignored for [`Replication::PerScale`] and
/// [`Replication::Flat`] blocks; `Flat` additionally requires `scale == 0`
/// (it has no scale axis — see the variant's doc).
pub(crate) fn slot_id(
    family: impl Into<Family>,
    block_local: usize,
    scale: usize,
    channel: usize,
    n_scales: usize,
) -> Option<usize> {
    let (base, block) = block_base(family, n_scales)?;
    if block_local >= block.signals.len() {
        return None;
    }
    let per = block.signals.len();
    Some(match block.replication {
        Replication::GmsbankChroma => {
            if scale >= n_scales {
                return None;
            }
            if scale == 0 {
                if channel != 1 || block_local >= 15 {
                    return None;
                }
                base + block_local
            } else if block_local < 15 {
                if channel >= 3 {
                    return None;
                }
                base + 15 + (scale - 1) * 55 + channel * 15 + block_local
            } else {
                base + 15 + (scale - 1) * 55 + 45 + block_local - 15
            }
        }
        Replication::NativeXb => {
            if scale != 0 || !matches!(channel, 0 | 2) {
                return None;
            }
            base + (channel / 2) * per + block_local
        }
        Replication::PerChannel => {
            if scale >= n_scales || channel >= 3 {
                return None;
            }
            base + (scale * 3 + channel) * per + block_local
        }
        Replication::PerScale => {
            if scale >= n_scales {
                return None;
            }
            base + scale * per + block_local
        }
        // One cell: no (scale, channel) axes. `scale == 0` is required so
        // this round-trips [`def_at`]'s flat-block reading, and so a
        // per-scale enumeration of a flat block does not silently answer
        // the same slot once per scale.
        Replication::Flat => {
            if scale != 0 {
                return None;
            }
            base + block_local
        }
    })
}

/// The definition of slot `id` at `n_scales`, or `None` past the full width.
///
/// The inverse of [`slot_id`], and gated to round-trip against it.
pub(crate) fn def_at(id: usize, n_scales: usize) -> Option<FeatureDef> {
    let mut base = 0usize;
    for block in BLOCKS {
        base += reserved_before(block.family);
        if id < base {
            return None;
        }
        let w = block.width(n_scales);
        if id < base + w {
            let off = id - base;
            let per = block.signals.len();
            let (scale, channel, local) = match block.replication {
                Replication::GmsbankChroma => {
                    if off < 15 {
                        (0, Channel::Y, off)
                    } else {
                        let scale = 1 + (off - 15) / 55;
                        let within = (off - 15) % 55;
                        if within < 45 {
                            (scale, Channel::TRIPLE[within / 15], within % 15)
                        } else {
                            (scale, Channel::Scalar, 15 + within - 45)
                        }
                    }
                }
                Replication::NativeXb => {
                    let cell = off / per;
                    (0, [Channel::X, Channel::B][cell], off % per)
                }
                Replication::PerChannel => {
                    let cell = off / per;
                    (cell / 3, Channel::TRIPLE[cell % 3], off % per)
                }
                Replication::PerScale => (off / per, Channel::Scalar, off % per),
                Replication::Flat => (0, Channel::Scalar, off),
            };
            return Some(FeatureDef {
                id: u16::try_from(id).ok()?,
                signal: &block.signals[local],
                scale: u8::try_from(scale).ok()?,
                channel,
            });
        }
        base += w;
    }
    None
}

/// The ten `fused944native` CARRIER slots.
///
/// A hand-picked subset of peaks/masked/iw rather than a derivable rule — the
/// slots the `fused944native` tables actually carry (peak `art_l8`, and the
/// masked/IW `art_4th` at the scales the fused edge kernel reaches). Declared
/// here so the registry is complete on a `--no-default-features` build, and
/// gated equal to `V1PoolsMode::CARRIER_SLOTS` — the owner — by
/// [`tests::carrier_slots_match_the_feature_v2_owner`].
pub(crate) const CARRIER_SLOTS: [usize; 10] = [178, 190, 196, 226, 231, 237, 243, 303, 321, 333];

/// Every slot of `family` at `n_scales`, as a [`SlotSet`].
///
/// THE family→slots derivation. Covers all twelve [`ComputeToken`]s: the
/// eight contiguous blocks, the two scattered free tranches, and the carrier
/// subset. Gated equal to `ComputeSet::populated_slots` for every token by
/// [`tests::family_slots_match_compute_set_populated_slots`].
pub(crate) fn family_slots(family: impl Into<Family>, n_scales: usize) -> SlotSet {
    match family.into() {
        Family::Supported(ComputeToken::Moments) => tranche_slots(Tranche::RawMoments, n_scales),
        Family::Supported(ComputeToken::ClassC) => tranche_slots(Tranche::ClassC, n_scales),
        Family::Supported(ComputeToken::Carriers) => SlotSet::from_slots(CARRIER_SLOTS),
        // `Hdr` is RESERVED — registered in the vocabulary, emitted by
        // nothing, so its slot set is legitimately empty rather than absent.
        Family::Supported(ComputeToken::Hdr) => SlotSet::default(),
        other => match block_base(other, n_scales) {
            Some((base, block)) => SlotSet::from_ranges([(base, base + block.width(n_scales))]),
            None => SlotSet::default(),
        },
    }
}

/// Does this signal's [`Placement`] admit `(scale, channel)`?
fn placement_admits(placement: Placement, scale: usize, channel: Channel) -> bool {
    match placement {
        Placement::AllCells => true,
        // `APPEND_SKIP_B_SCALE0`: the 944 walk never computes that append
        // cell, so its slots are structural zeros on BOTH sides — including
        // them would make a parity gate weaker, not stronger.
        Placement::SkipBScale0 => !(channel == Channel::B && scale == 0),
        Placement::YOnlyActive => channel == Channel::Y,
    }
}

/// The slots a v1-only walk can harvest at `tranche`, at `n_scales`.
///
/// **Derived, never listed.** Gated bit-for-bit against `feature_v2`'s owner
/// functions (`free_slot_indices` / `class_c_slot_indices`) by
/// [`tests::tranche_slots_match_the_feature_v2_owners`], so the registry
/// carries the same answer without the registry being a second copy of the
/// arithmetic: it derives the set from per-signal declarations, and the gate
/// is what makes the two provably equal.
///
/// [`Tranche::ClassC`] is the ADDITION over [`Tranche::RawMoments`], matching
/// `V1FreeExtras::RawMomentsPlusBoundedErr`'s own definition — not a superset.
pub(crate) fn tranche_slots(tranche: Tranche, n_scales: usize) -> SlotSet {
    let mut out = Vec::new();
    for id in 0..full_width(n_scales) {
        let Some(d) = def_at(id, n_scales) else {
            continue;
        };
        if d.signal.tranche == tranche
            && placement_admits(d.signal.placement, usize::from(d.scale), d.channel)
        {
            out.push(id);
        }
    }
    SlotSet::from_slots(out)
}

/// The honest per-SLOT cost: [`CostClass::Free`] when a tranche can harvest
/// this placement, else the signal's own base cost.
///
/// A per-signal cost field cannot express this — see [`Tranche`]'s doc for
/// the three ways it is wrong.
pub(crate) fn cost_of(id: usize, n_scales: usize) -> Option<CostClass> {
    let d = def_at(id, n_scales)?;
    let harvestable = d.signal.tranche != Tranche::None
        && placement_admits(d.signal.placement, usize::from(d.scale), d.channel);
    Some(if harvestable {
        CostClass::Free
    } else {
        d.signal.cost
    })
}

/// Every registered signal, in layout order — the registry's iteration entry.
///
/// Not yet called (reserved for the Phase-2 research engine's iteration over
/// the registry, `docs/PLAN_FEATURE_SYSTEM_2026-09-05.md`); kept because it is
/// the documented entry point for that consumer, not because it runs today.
#[allow(dead_code)]
pub(crate) fn signals() -> impl Iterator<Item = &'static SignalDef> {
    BLOCKS.iter().flat_map(|b| b.signals.iter())
}

#[cfg(test)]
mod tests {
    /// **G3.1 for revision 2's `v1ssimcap` era** — the registry's claim about
    /// WHICH slots the F4 fix moves, checked against a re-extraction.
    ///
    /// MEASURED 2026-09-05 (`/mnt/v/output/zensim/rev2-2026-09-05/`): the
    /// `feature_invariant_probe dump` set, 22,396 `to_bits()` cells over 3
    /// routes, re-run with `ZENSIM_SSIM_LUMA=c1` and diffed against the
    /// shipped form, moves exactly **132** slots — basic 36, peaks 24, masked
    /// 36, IW 36.
    ///
    /// **This gate FAILED when it was first written, and that is the finding.**
    /// `docs/FEATURE_DEFECTS_AUDIT_2026-09-05.md` scopes F4 to "the `ssim_*`
    /// family in `f228..299` + `f300..371`" — 72 slots — because that is where
    /// the 5.8e6 outliers were observed. But the basic block's
    /// `ssim_mean/4th/2nd` and the peaks block's `ssim_max`/`ssim_l8` are
    /// pooled from the SAME per-pixel `d`, so a bounded form moves them too.
    /// Observing where a defect's SYMPTOM is largest is not the same as
    /// deriving where its CAUSE reaches; the registry now carries the derived
    /// answer.
    ///
    /// The peaks pair is the part no per-block reading would have predicted:
    /// `ssim_max` and `ssim_l8` are local 0 and local 3 of a 6-signal block,
    /// so the moved ids are an every-third-slot comb, not a contiguous run.
    ///
    /// ⚠ The instrument changed on 2026-09-05 when F17's `v1hfgain` joined
    /// revision 2: this asserts an ERA's slot set, so it now reads
    /// [`super::era_moved_slots`]`("v1ssimcap", …)` rather than
    /// `Rev2.moved_slots`, whose union is 144. The CLAIM (132, and which 132)
    /// is unchanged and is still checked exactly — see `era_moved_slots`'s own
    /// doc for why the revision-level form could no longer express it.
    #[test]
    fn f4_moves_exactly_the_registered_slots() {
        use crate::NUM_SCALES;
        let got = super::era_moved_slots("v1ssimcap", 372, NUM_SCALES);

        // Derived independently of the signal table, from the layout the
        // measurement exposed: (block base, per-cell width, local ids).
        let mut want: Vec<u16> = Vec::new();
        for (base, per_cell, locals) in [
            (0u16, 13u16, &[0u16, 1, 2][..]), // basic  ssim_mean/4th/2nd
            (156, 6, &[0, 3][..]),            // peaks  ssim_max / ssim_l8
            (228, 6, &[0, 1, 2][..]),         // masked ssim_mean/4th/2nd
            (300, 6, &[0, 1, 2][..]),         // IW     ssim_mean/4th/2nd
        ] {
            for cell in 0..(NUM_SCALES as u16 * 3) {
                for &l in locals {
                    want.push(base + cell * per_cell + l);
                }
            }
        }
        want.sort_unstable();

        assert_eq!(
            got, want,
            "the registry's v1ssimcap slot set disagrees with the measured one"
        );
        assert_eq!(
            got.len(),
            132,
            "expected 132 moved slots, got {}",
            got.len()
        );
        // The audit's two worst columns must be in it.
        for id in [241u16, 313] {
            assert!(
                got.contains(&id),
                "f{id} (an audit-named worst column) is missing"
            );
        }
    }

    /// **The registry half of G3.1 for revision 3's `v1ssimstable` era.**
    ///
    /// Revision 3 changes the PRECISION of the same per-pixel `d` that F4's
    /// bounded form changes, so it reaches exactly the same slots — asserted
    /// against `v1ssimcap`'s measured set rather than against a second
    /// hand-written list.
    ///
    /// The two eras are NOT interchangeable despite sharing a slot set:
    /// `v1ssimcap` is bit-identical to revision 1 wherever `D^2 <= 1`, i.e.
    /// on all ordinary content, while `v1ssimstable` moves every one of these
    /// columns on every image. Same blast radius, very different blast.
    ///
    /// The MEASURED half — that the extractor really does move this set and
    /// nothing else — is
    /// `streaming::tests::rev3_moves_exactly_the_registered_slots`, which
    /// re-extracts at revision 3 in a child process and diffs.
    #[test]
    fn ssim_stable_moves_exactly_the_f4_slots() {
        use super::FormulaRevision;
        use crate::NUM_SCALES;
        let stable = super::era_moved_slots("v1ssimstable", 372, NUM_SCALES);
        let f4 = super::era_moved_slots("v1ssimcap", 372, NUM_SCALES);
        assert_eq!(
            stable, f4,
            "v1ssimstable and v1ssimcap read the same per-pixel signal, so they \
             must declare the same slots"
        );
        assert_eq!(stable.len(), 132);
        // And it is one of revision 3's eras, so a recalculation planned from
        // the revision (not the era) still sees these columns.
        assert!(
            FormulaRevision::Rev3
                .moved_slots(372, NUM_SCALES)
                .contains(&241)
        );
        assert!(FormulaRevision::Rev3.era_tokens().contains(&"v1ssimstable"));
    }

    /// Revision 1 is the baseline, so it moves nothing by definition.
    #[test]
    fn rev1_moves_no_slots() {
        use super::FormulaRevision;
        use crate::NUM_SCALES;
        assert!(
            FormulaRevision::Rev1
                .moved_slots(944, NUM_SCALES)
                .is_empty()
        );
        assert!(FormulaRevision::Rev1.era_tokens().is_empty());
    }

    /// **G3.1 for revision 2's `v1hfgain` era** — F17's twelve slots, derived
    /// from the layout rather than copied from a list.
    ///
    /// `contrast_inc` is block-local 12 of the 13-signal basic block, so the
    /// moved ids are `13·cell + 12` for the twelve (scale, channel) cells.
    /// Unlike `v1ssimcap`, this set does NOT change with pool state: the basic
    /// block is present in every registered layout, so a 944 read moves the
    /// same twelve and only those twelve.
    #[test]
    fn f17_moves_exactly_the_twelve_contrast_inc_slots() {
        use crate::NUM_SCALES;
        let want: Vec<u16> = (0..NUM_SCALES as u16 * 3).map(|c| c * 13 + 12).collect();
        assert_eq!(want.len(), 12);
        for width in [372u16, 944] {
            let got = super::era_moved_slots("v1hfgain", width, NUM_SCALES);
            assert_eq!(
                got, want,
                "v1hfgain slot set at width {width} disagrees with the layout"
            );
        }
        // The four slots the R6 record names as the corpus-wise worst, and the
        // two the SDR default reads unguarded.
        for id in [12u16, 38, 129, 116, 155] {
            assert!(want.contains(&id), "f{id} missing from the F17 set");
        }
        // Its two bounded siblings must NOT be in it — F17 is the asymmetry,
        // and a fix that moved `var_loss` would be a different, larger change.
        for id in [10u16, 11, 23, 24] {
            assert!(!want.contains(&id), "f{id} (a loss member) must not move");
        }
    }

    /// Revision 2 carries all THREE registered eras, and `freecomp`'s slots are
    /// reachable only at a width that has an append block — which is exactly
    /// why F5 is free today and stops being free the moment a bake declares
    /// one.
    /// **F19's era moves ZERO feature slots, and that is the claim.**
    ///
    /// Every other registered era answers "which slots?" by derivation from
    /// the signal table. `scorepow` answers "none" — it is a SCORE era. This
    /// pins the emptiness at every registered width so that (a) a future
    /// reader cannot mistake it for a registration someone forgot to finish,
    /// and (b) attaching F19 to a signal — which would be wrong, and would
    /// invalidate stored feature tables for a defect that cannot reach them —
    /// fails here.
    #[test]
    fn scorepow_moves_no_feature_slot() {
        use crate::NUM_SCALES;
        for width in [372u16, 720, 924, 944] {
            assert!(
                super::era_moved_slots("scorepow", width, NUM_SCALES).is_empty(),
                "scorepow must move no feature slot at width {width} — it is a \
                 SCORE era; if this fires, F19 was attached to a SignalDef"
            );
        }
        // But it IS registered, in the other registry — otherwise
        // `research::era_is_registered` would call revision 2 incoherent.
        assert!(super::is_score_path_era("scorepow"));
        assert!(!super::is_score_path_era("v1detroot"));
        assert_eq!(super::SCORE_PATH_REVISIONS.len(), 1);
        assert_eq!(super::SCORE_PATH_REVISIONS[0].0.id, "F19");
        assert_eq!(
            super::SCORE_PATH_REVISIONS[0].1.status,
            super::RevisionStatus::Proposed,
            "SHIPPED_REVISION is Rev1; F19 must stay Proposed"
        );
    }

    /// featcanon D3: `tiercanon` is REGISTERED (the research provenance and
    /// `era_is_registered` can name it) as an ARITHMETIC era that moves every
    /// live slot, it is Rev4's and only Rev4's, and it is still Proposed.
    #[test]
    fn tiercanon_is_a_registered_arithmetic_era_moving_every_live_slot() {
        use super::FormulaRevision;
        use crate::NUM_SCALES;
        assert!(super::is_arithmetic_era("tiercanon"));
        assert!(!super::is_arithmetic_era("v1ssimstable"));
        assert!(!super::is_score_path_era("tiercanon"));
        let r = super::arithmetic_revision("tiercanon").expect("registered");
        assert_eq!(r.status, super::RevisionStatus::Proposed);
        assert_eq!(super::ARITHMETIC_REVISIONS[0].0.id, "TIERDIV");
        // No per-signal entry claims it: the registry-level entry is the one
        // owner, so the two cannot drift apart.
        assert!(
            super::signals().all(|s| s.revisions.iter().all(|r| r.era != "tiercanon")),
            "tiercanon must not also be attached to a SignalDef"
        );
        for rev in [
            FormulaRevision::Rev1,
            FormulaRevision::Rev2,
            FormulaRevision::Rev3,
        ] {
            assert!(!rev.era_tokens().contains(&"tiercanon"), "{rev:?}");
        }
        assert!(FormulaRevision::Rev4.era_tokens().contains(&"tiercanon"));
        // Rev4 = Rev3's eras + tiercanon, in that order.
        assert_eq!(
            &FormulaRevision::Rev4.era_tokens()[..FormulaRevision::Rev3.era_tokens().len()],
            FormulaRevision::Rev3.era_tokens()
        );
        for width in [372u16, 944, 1825] {
            let live: Vec<u16> = (0..width as usize)
                .filter_map(|id| super::def_at(id, NUM_SCALES))
                .map(|d| d.id)
                .collect();
            assert_eq!(super::era_moved_slots("tiercanon", width, NUM_SCALES), live);
            assert_eq!(FormulaRevision::Rev4.moved_slots(width, NUM_SCALES), live);
            assert!(
                FormulaRevision::Rev3.moved_slots(width, NUM_SCALES).len() < live.len(),
                "Rev3's moved set must stay per-signal at width {width}"
            );
        }
    }

    /// rev5: `localwin` is REGISTERED as the second ARITHMETIC era — it
    /// moves every live slot, it is Rev5's and only Rev5's, and it is still
    /// Proposed. Same invariants the `tiercanon` gate pins, one era later.
    #[test]
    fn localwin_is_a_registered_arithmetic_era_moving_every_live_slot() {
        use super::FormulaRevision;
        use crate::NUM_SCALES;
        assert!(super::is_arithmetic_era("localwin"));
        assert!(!super::is_score_path_era("localwin"));
        let r = super::arithmetic_revision("localwin").expect("registered");
        assert_eq!(r.status, super::RevisionStatus::Proposed);
        assert_eq!(super::ARITHMETIC_REVISIONS[1].0.id, "WINDRIFT");
        // No per-signal entry claims it: the registry-level entry is the one
        // owner, so the two cannot drift apart.
        assert!(
            super::signals().all(|s| s.revisions.iter().all(|r| r.era != "localwin")),
            "localwin must not also be attached to a SignalDef"
        );
        for rev in [
            FormulaRevision::Rev1,
            FormulaRevision::Rev2,
            FormulaRevision::Rev3,
            FormulaRevision::Rev4,
        ] {
            assert!(!rev.era_tokens().contains(&"localwin"), "{rev:?}");
        }
        // Rev5 = Rev4's eras + localwin, in that order — `tiercanon` stays
        // in the token list because Rev5 builds on the canonical leaves.
        assert!(FormulaRevision::Rev5.era_tokens().contains(&"localwin"));
        assert_eq!(
            &FormulaRevision::Rev5.era_tokens()[..FormulaRevision::Rev4.era_tokens().len()],
            FormulaRevision::Rev4.era_tokens()
        );
        for width in [372u16, 944, 1825] {
            let live: Vec<u16> = (0..width as usize)
                .filter_map(|id| super::def_at(id, NUM_SCALES))
                .map(|d| d.id)
                .collect();
            assert_eq!(super::era_moved_slots("localwin", width, NUM_SCALES), live);
            assert_eq!(FormulaRevision::Rev5.moved_slots(width, NUM_SCALES), live);
        }
    }

    #[test]
    fn rev2_carries_all_eras_and_f5_needs_the_append_block() {
        use super::FormulaRevision;
        use crate::NUM_SCALES;
        let toks = FormulaRevision::Rev2.era_tokens();
        for want in ["v1ssimcap", "freecomp", "v1hfgain", "v1detroot", "scorepow"] {
            assert!(toks.contains(&want), "{want} missing from {toks:?}");
        }

        let at372 = FormulaRevision::Rev2.moved_slots(372, NUM_SCALES);
        let at944 = FormulaRevision::Rev2.moved_slots(944, NUM_SCALES);
        // THREE v1 eras at 372, and they are NOT disjoint: `v1detroot` (F18,
        // the libc-dependent pooled roots) OVERLAPS `v1ssimcap` (F4) on every
        // SSIM-derived `L4`/`L8` slot — the same slot, two independent
        // arithmetic problems. So a revision's `moved_slots` is a UNION, and
        // the sum-of-eras identity this test used to assert would now
        // double-count. The inclusion-exclusion below is the replacement, and
        // it is a strictly stronger claim: it pins each era's size AND the
        // exact size of the one overlap.
        let cap = super::era_moved_slots("v1ssimcap", 372, NUM_SCALES);
        let gain = super::era_moved_slots("v1hfgain", 372, NUM_SCALES);
        let root = super::era_moved_slots("v1detroot", 372, NUM_SCALES);
        assert_eq!(
            cap.len(),
            132,
            "v1ssimcap: basic 36 + peaks 24 + masked 36 + IW 36"
        );
        assert_eq!(gain.len(), 12, "v1hfgain: the twelve `contrast_inc` slots");
        assert_eq!(
            root.len(),
            144,
            "v1detroot: the L4/L8 pools — basic 36 + peaks 36 + masked 36 + IW 36"
        );
        let overlap = cap.iter().filter(|id| root.contains(id)).count();
        assert_eq!(
            overlap, 48,
            "the four SSIM L4/L8 signals (basic ssim_4th, peaks ssim_l8, masked \
             ssim_4th, IW ssim_4th) x 12 (scale, channel) cells"
        );
        assert_eq!(
            gain.iter()
                .filter(|id| cap.contains(id) || root.contains(id))
                .count(),
            0,
            "v1hfgain is a Ratio family and must not touch either pooled-root era"
        );
        assert_eq!(
            at372.len(),
            cap.len() + gain.len() + root.len() - overlap,
            "moved_slots must be the UNION of its eras, not the sum"
        );
        assert_eq!(at372.len(), 240, "132 + 12 + 144 - 48");
        assert!(
            at944.len() > at372.len(),
            "944 must additionally reach freecomp's raw-moment slots: {} vs {}",
            at944.len(),
            at372.len()
        );
        // Every 372 slot is still in the 944 set — a revision appends, it
        // never drops.
        for id in &at372 {
            assert!(at944.contains(id), "f{id} vanished at 944");
        }
    }

    use super::*;

    const NS: usize = crate::NUM_SCALES;

    /// **G1.4** — `def_at` and `slot_id` are inverses on every slot of the
    /// full layout. A layout-arithmetic typo cannot survive this.
    #[test]
    fn id_arithmetic_round_trips_on_every_slot() {
        let w = full_width(NS);
        assert_eq!(
            block_base(Family::Palette, NS).unwrap().0,
            1825,
            "legacy width stays frozen"
        );
        assert_eq!(
            block_base(ComputeToken::GlobAff, NS).unwrap().0,
            1867,
            "globaff sits after the palette block"
        );
        assert_eq!(w, 1891, "full registered width with palette and globaff");
        for id in 0..w {
            let Some(d) = def_at(id, NS) else {
                assert!((956..986).contains(&id) || (1820..1825).contains(&id));
                continue;
            };
            let ch = match d.channel {
                Channel::X => 0,
                Channel::Y => 1,
                Channel::B => 2,
                Channel::Scalar => 0,
            };
            let back = slot_id(
                d.signal.family,
                usize::from(d.signal.block_local),
                usize::from(d.scale),
                ch,
                NS,
            )
            .unwrap_or_else(|| panic!("slot_id failed for slot {id}"));
            assert_eq!(back, id, "round-trip for slot {id} ({})", d.name());
        }
        assert!(def_at(w, NS).is_none(), "past the full width");
    }

    #[test]
    fn gmsbank_sparse_chroma_placement_at_every_scale_count() {
        for ns in 1..=4 {
            let (base, block) = block_base(ComputeToken::Gmsbank, ns).unwrap();
            assert_eq!(block.width(ns), [15, 70, 125, 180][ns - 1]);
            for local in 0..25 {
                assert!(slot_id(ComputeToken::Gmsbank, local, 0, 0, ns).is_none());
                assert!(slot_id(ComputeToken::Gmsbank, local, 0, 2, ns).is_none());
                assert_eq!(
                    slot_id(ComputeToken::Gmsbank, local, 0, 1, ns).is_some(),
                    local < 15
                );
            }
            for id in base..base + block.width(ns) {
                let d = def_at(id, ns).unwrap();
                let channel = match d.channel {
                    Channel::X | Channel::Scalar => 0,
                    Channel::Y => 1,
                    Channel::B => 2,
                };
                assert_eq!(
                    slot_id(
                        ComputeToken::Gmsbank,
                        usize::from(d.signal.block_local),
                        usize::from(d.scale),
                        channel,
                        ns
                    ),
                    Some(id)
                );
                assert!(usize::from(d.scale) < ns);
                if d.signal.block_local >= 15 {
                    assert!(d.scale > 0);
                    assert_eq!(d.channel, Channel::Scalar);
                }
            }
        }
    }

    /// The block bases are the numbers every emit site in `feature_v2` writes
    /// out by hand. Pinned here so the registry cannot drift from them.
    #[test]
    fn block_bases_match_the_documented_layout() {
        let expect = [
            (Family::Supported(ComputeToken::Basic), 0usize, 156usize),
            (Family::Supported(ComputeToken::Peaks), 156, 72),
            (Family::Supported(ComputeToken::Masked), 228, 72),
            (Family::Supported(ComputeToken::Iw), 300, 72),
            (Family::Supported(ComputeToken::V2), 372, 348),
            (Family::Supported(ComputeToken::Append), 720, 204),
            (Family::Supported(ComputeToken::Append2), 924, 20),
            (Family::Supported(ComputeToken::Csfw), 944, 12),
            (Family::Supported(ComputeToken::Gridblk), 986, 96),
            (Family::Supported(ComputeToken::Ringbasis), 1082, 72),
            (Family::Supported(ComputeToken::Tailhist), 1154, 144),
            (Family::Supported(ComputeToken::Arttype), 1298, 24),
            (Family::Supported(ComputeToken::Gmsbank), 1322, 180),
            (Family::Supported(ComputeToken::Mapdev), 1502, 60),
            (Family::Supported(ComputeToken::Z1max), 1562, 228),
            (Family::Supported(ComputeToken::Gmsnative), 1790, 30),
            (Family::Palette, 1825, 42),
        ];
        for (family, base, width) in expect {
            let (b, blk) = block_base(family, NS).expect("registered family");
            assert_eq!(b, base, "{family} base");
            assert_eq!(blk.width(NS), width, "{family} width");
        }
    }

    /// **G1.3** — every slot name is unique and obeys the `zenanalyze-api`
    /// name charset (`[a-z0-9_]+`), which the feature-set id grammar reuses.
    #[test]
    fn names_are_unique_and_valid() {
        let mut seen = std::collections::BTreeSet::new();
        for id in 0..full_width(NS) {
            let Some(def) = def_at(id, NS) else {
                assert!((956..986).contains(&id) || (1820..1825).contains(&id));
                continue;
            };
            let n = def.name();
            assert!(
                crate::feature_set_id::is_valid_token(&n),
                "slot {id} name {n:?} is not [a-z0-9_]+"
            );
            assert!(seen.insert(n.clone()), "duplicate slot name {n:?} at {id}");
        }
        // 1832 through the palette, plus globaff's 24 (2 signals × 4 scales × 3 channels).
        assert_eq!(seen.len(), 1832 + 24);
    }

    /// Signal names are unique WITHIN a family (the family prefix is what
    /// disambiguates `basic`'s `ssim_mean` from `v2`'s).
    #[test]
    fn signal_names_are_unique_within_each_family() {
        for block in BLOCKS {
            let mut seen = std::collections::BTreeSet::new();
            for s in block.signals {
                assert!(
                    seen.insert(s.name),
                    "{} has two signals named {:?}",
                    block.family,
                    s.name
                );
            }
        }
    }

    /// `block_local` is the array index — i.e. the registry's order IS the
    /// `idx*` constant order, which is what index-aligns it with the emit
    /// sites.
    #[test]
    fn block_local_equals_the_array_index() {
        for block in BLOCKS {
            for (i, s) in block.signals.iter().enumerate() {
                assert_eq!(usize::from(s.block_local), i, "{} signal {i}", block.family);
                assert_eq!(s.family, block.family, "{} signal {i} family", block.family);
            }
        }
    }

    /// The free tranches are DERIVED from the per-signal declarations, and
    /// the counts are the ones `feature_v2`'s own docs state.
    #[test]
    fn free_tranches_are_derivable_from_the_declarations() {
        let raw = tranche_slots(Tranche::RawMoments, NS);
        let bounded = tranche_slots(Tranche::ClassC, NS);
        // 3 GLOBAL_* over the 11 append cells the 944 walk computes (all but
        // `(B, scale 0)`) = 33, plus LUMA_MEAN_REF per scale = 4.
        assert_eq!(raw.len(), 37, "raw-moment tranche");
        // The v2-348 MSE slot per (scale, channel) = 12, plus the three
        // luminance bins at Y only = 12.
        assert_eq!(bounded.len(), 24, "class-C tranche");
        for id in raw.iter_slots().chain(bounded.iter_slots()) {
            assert_eq!(
                cost_of(id, NS).expect("cost"),
                CostClass::Free,
                "slot {id} is in a tranche but not costed Free"
            );
        }
    }

    /// The two tranches are DISJOINT — class C is the ADDITION over the raw
    /// moments, matching `V1FreeExtras::RawMomentsPlusBoundedErr`'s own
    /// definition rather than being a superset.
    #[test]
    fn the_two_tranches_are_disjoint() {
        let raw = tranche_slots(Tranche::RawMoments, NS);
        let bounded = tranche_slots(Tranche::ClassC, NS);
        assert!(
            raw.intersect(&bounded).is_empty(),
            "tranches overlap: {}",
            raw.intersect(&bounded)
        );
    }

    /// A slot outside every tranche keeps its block's base cost — the
    /// negative control for `cost_of`.
    #[test]
    fn slots_outside_a_tranche_keep_their_block_cost() {
        // `(B, scale 0)` global_dmean is in the tranche's SIGNAL but not its
        // placement: the 944 walk never computes that append cell.
        let skipped = slot_id(ComputeToken::Append, 13, 0, 2, NS).expect("slot");
        assert_eq!(cost_of(skipped, NS), Some(CostClass::Expensive));
        assert!(!tranche_slots(Tranche::RawMoments, NS).contains(skipped));
        // The same signal at `(X, scale 0)` IS harvestable.
        let kept = slot_id(ComputeToken::Append, 13, 0, 0, NS).expect("slot");
        assert_eq!(cost_of(kept, NS), Some(CostClass::Free));
        // A luminance bin at X is not harvestable; at Y it is.
        let lum_x = slot_id(ComputeToken::Append, 2, 1, 0, NS).expect("slot");
        let lum_y = slot_id(ComputeToken::Append, 2, 1, 1, NS).expect("slot");
        assert_eq!(cost_of(lum_x, NS), Some(CostClass::Expensive));
        assert_eq!(cost_of(lum_y, NS), Some(CostClass::Free));
    }

    /// **The identity decomposition, pinned to the registry.**
    ///
    /// The audit measured that a 944-wide identity vector resolves into
    /// exactly three populations: 15 reference-only slots (correct by design),
    /// 12 `PJND_FRAGILITY` slots (defect F15), and fp residue <= 1.12e-3. The
    /// registry's `Form` flag must reproduce the first two counts from its own
    /// declarations, or it does not encode what the identity probe measures.
    ///
    /// 15 = `grad_src_mean` at the 11 append cells the 944 walk computes
    /// (12 minus the `(B, scale 0)` cell) + `luma_mean_ref` once per scale.
    #[test]
    fn identity_nonzero_slots_decompose_exactly_as_the_audit_measured() {
        let w944 = 944usize;
        let mut ref_only = Vec::new();
        let mut fragility = Vec::new();
        for id in 0..w944 {
            let Some(d) = def_at(id, NS) else {
                assert!((956..986).contains(&id) || (1820..1825).contains(&id));
                continue;
            };
            if d.signal.name == "pjnd_fragility" {
                fragility.push(id);
                continue;
            }
            if d.signal.form == Form::ReferenceOnly
                && placement_admits(d.signal.placement, usize::from(d.scale), d.channel)
                // `grad_src_mean` lives in the append block, which the 944
                // walk skips at `(B, scale 0)` — the same cell the raw-moment
                // tranche skips, for the same reason.
                && !(d.signal.family == ComputeToken::Append
                    && d.channel == Channel::B
                    && d.scale == 0)
            {
                ref_only.push(id);
            }
        }
        assert_eq!(
            ref_only.len(),
            15,
            "reference-only slots at 944 (audit measured 15): {ref_only:?}"
        );
        assert_eq!(
            fragility.len(),
            12,
            "PJND_FRAGILITY slots at 944 (audit measured 12): {fragility:?}"
        );
        // And the artifact is registered as a defect, not silently accepted.
        let f = def_at(fragility[0], NS).expect("def");
        assert_eq!(
            f.signal.defect.map(|d| d.id),
            Some("F15"),
            "PJND_FRAGILITY's identity artifact must be a REGISTERED defect"
        );
    }

    /// Every audit defect the registry models is attached to the slots the
    /// audit named, and to no others.
    #[test]
    fn registered_defects_cover_exactly_the_audited_slots() {
        let mut by_id: std::collections::BTreeMap<&str, Vec<usize>> = Default::default();
        for id in 0..full_width(NS) {
            if let Some(d) = def_at(id, NS).and_then(|def| def.signal.defect) {
                by_id.entry(d.id).or_default().push(id);
            }
        }
        // F4: every slot pooled from the per-pixel SSIM dissimilarity.
        //
        // WIDENED 2026-09-05 from 72 to 132, by MEASUREMENT rather than by
        // re-reading the audit. This assertion previously encoded the audit's
        // own scoping — "the `ssim_*` family in `f228..299` + `f300..371`",
        // i.e. masked + IW only — which is where the 5.8e6 SYMPTOM was
        // observed. Re-extracting the probe's 22,396-cell dump under a bounded
        // luminance form moves 132 slots: basic 36 (`ssim_mean/4th/2nd`),
        // peaks 24 (`ssim_max`, `ssim_l8`), masked 36, IW 36. They share one
        // per-pixel `d`, so they share its defect.
        //
        // This is a WIDENING of the claim, not a relaxation: the registry now
        // asserts the defect reaches 60 MORE slots than before, and
        // `f4_moves_exactly_the_registered_slots` holds that set against a
        // re-extraction so it cannot drift back to the smaller number.
        assert_eq!(by_id.get("F4").map(Vec::len), Some(132), "F4 slot count");
        // The audit's two worst scanned slots must be inside it.
        for named in [241usize, 313] {
            assert!(
                by_id["F4"].contains(&named),
                "F4 must cover f{named}, the audit's worst scanned slot"
            );
        }
        // F5: the three GLOBAL_* signals over the 11 computed append cells,
        // plus `luma_mean_ref` per scale — the raw-moment tranche exactly.
        let f5: std::collections::BTreeSet<usize> = by_id["F5"].iter().copied().collect();
        let tranche: std::collections::BTreeSet<usize> = tranche_slots(Tranche::RawMoments, NS)
            .iter_slots()
            .collect();
        assert!(
            tranche.is_subset(&f5),
            "F5 must cover the whole raw-moment tranche"
        );
        // F15: 12 slots.
        assert_eq!(by_id.get("F15").map(Vec::len), Some(12), "F15 slot count");
        // F18: the pooled 4th/8th roots. `SignalDef` holds ONE `defect`, so
        // the 48 SSIM `L4`/`L8` slots that are in BOTH eras name F4 (the
        // larger problem) here and carry F18 only through their revision
        // list — which is why this count is 108 and not 156. The
        // era-complete answer is `era_moved_slots("v1detroot", …)`, checked
        // immediately below against the statistic it is derived from.
        assert_eq!(by_id.get("F18").map(Vec::len), Some(108), "F18 slot count");
        let detroot: std::collections::BTreeSet<usize> = (0..full_width(NS))
            .filter(|&id| {
                def_at(id, NS)
                    .is_some_and(|d| d.signal.revisions.iter().any(|r| r.era == "v1detroot"))
            })
            .collect();
        let l4l8: std::collections::BTreeSet<usize> = (0..full_width(NS))
            .filter(|&id| {
                def_at(id, NS).is_some_and(|d| super::uses_pooled_root(d.signal.statistic))
            })
            .collect();
        assert_eq!(
            detroot, l4l8,
            "the `v1detroot` era must be EXACTLY the L4/L8 slots — derived from \
             the statistic, never from a second list"
        );
        assert_eq!(detroot.len(), 156, "144 v1 L4/L8 + 12 v2 `ssim_dev4`");
        assert!(
            by_id["F18"].iter().all(|id| detroot.contains(id)),
            "every F18-labelled slot must be inside the era it belongs to"
        );
        // F17: the twelve `contrast_inc` slots — and ONLY those. The two
        // bounded members of the same family must stay defect-free, because
        // the asymmetry is the defect: attaching F17 to `var_loss` would
        // triple a 12-slot recalculation to fix nothing.
        assert_eq!(by_id.get("F17").map(Vec::len), Some(12), "F17 slot count");
        for id in [12usize, 38, 116, 129, 155] {
            assert!(by_id["F17"].contains(&id), "F17 must cover f{id}");
        }
        for id in 0..full_width(NS) {
            let Some(d) = def_at(id, NS) else {
                assert!((956..986).contains(&id) || (1820..1825).contains(&id));
                continue;
            };
            if d.signal.name == "var_loss" || d.signal.name == "tex_loss" {
                assert!(
                    d.signal.defect.is_none(),
                    "f{id} ({}) is bounded by construction and must carry no defect",
                    d.signal.name
                );
            }
        }
    }

    /// A PROPOSED revision must not claim to have landed, and every LANDED
    /// revision must name a real commit. This is what keeps "modelled" and
    /// "applied" from blurring: F4's fix is registered and deliberately NOT
    /// applied, and the registry has to say so in a machine-checkable way.
    #[test]
    fn proposed_revisions_are_distinguishable_from_landed_ones() {
        let (mut landed, mut proposed) = (0usize, 0usize);
        for sig in signals() {
            for r in sig.revisions {
                match r.status {
                    RevisionStatus::Landed => {
                        assert_ne!(
                            r.commit, "-",
                            "{}: landed revision needs a commit",
                            sig.name
                        );
                        landed += 1;
                    }
                    RevisionStatus::Proposed => {
                        assert_eq!(
                            r.commit, "-",
                            "{}: a PROPOSED revision must not name a commit",
                            sig.name
                        );
                        proposed += 1;
                    }
                }
            }
        }
        assert!(landed > 0 && proposed > 0, "expected both kinds");
        // The three proposals revision 2 batches.
        let eras: std::collections::BTreeSet<&str> = signals()
            .flat_map(|s| s.revisions.iter())
            .filter(|r| r.status == RevisionStatus::Proposed)
            .map(|r| r.era)
            .collect();
        assert!(eras.contains("v1ssimcap"), "F4's proposed revision");
        assert!(eras.contains("freecomp"), "F5's proposed revision");
        assert!(eras.contains("v1hfgain"), "F17's proposed revision");
    }

    /// Every v1 slot carries the option-C revision; nothing else claims it.
    #[test]
    fn option_c_revision_is_scoped_to_the_v1_families() {
        for block in BLOCKS {
            let v1_family = matches!(
                block.family,
                Family::Supported(
                    ComputeToken::Basic
                        | ComputeToken::Peaks
                        | ComputeToken::Masked
                        | ComputeToken::Iw
                )
            );
            for s in block.signals {
                let has_c = s.revisions.iter().any(|r| r.commit == "56bbcda2");
                assert_eq!(
                    has_c, v1_family,
                    "{}::{} option-C revision presence",
                    block.family, s.name
                );
            }
        }
    }
}

/// Cross-checks against the `feature_v2` owners — the gates that make the
/// registry's independent derivations provably equal to the code that
/// actually emits the rows, rather than a plausible second opinion.
///
/// Gated on `feature-regime-v2` because the owners are: the registry itself
/// is pure data and builds without it.
#[cfg(all(test, feature = "feature-regime-v2"))]
mod owner_gates {
    use super::*;
    use crate::feature_set_id::ComputeToken as T;
    use crate::feature_v2;

    const NS: usize = crate::NUM_SCALES;

    /// The block widths this registry derives BY COUNTING its own signals must
    /// equal the geometry constants `feature_v2` and `metric` declare. A new
    /// signal that forgets to bump its block's constant fails here rather than
    /// silently shifting every slot above it.
    #[test]
    fn geometry_matches_the_block_constants() {
        assert_eq!(BASIC.len(), crate::metric::FEATURES_PER_CHANNEL_BASIC);
        assert_eq!(
            BASIC.len() + PEAKS.len(),
            crate::metric::FEATURES_PER_CHANNEL_WITH_PEAKS
        );
        assert_eq!(
            BASIC.len() + PEAKS.len() + MASKED.len(),
            crate::metric::FEATURES_PER_CHANNEL_EXTENDED
        );
        assert_eq!(IW.len(), crate::metric::FEATURES_PER_CHANNEL_IW);
        assert_eq!(V2.len(), feature_v2::FEATURES_PER_CHANNEL_V2_TOTAL);
        assert_eq!(APPEND.len(), feature_v2::FEATURES_PER_CHANNEL_APPEND);
        assert_eq!(APPEND2.len(), feature_v2::APPEND2_PER_SCALE);
        assert_eq!(CSFW.len(), feature_v2::CSFW_PER_SCALE);
    }

    /// The registry's `block_local` ordering IS the `idx*` constant ordering —
    /// which is what index-aligns a `FeatureDef` with the value at that slot.
    /// Spot-checked at every block's named boundaries.
    #[test]
    fn block_local_indices_match_the_idx_modules() {
        use feature_v2::{idx, idx_append, idx_append2, idx_csfw};
        let find = |t: &[SignalDef], name: &str| -> usize {
            usize::from(
                t.iter()
                    .find(|s| s.name == name)
                    .unwrap_or_else(|| panic!("no signal {name}"))
                    .block_local,
            )
        };
        assert_eq!(find(&V2, "ssim_mean"), idx::SSIM_MEAN);
        assert_eq!(find(&V2, "mse"), idx::MSE);
        assert_eq!(find(&V2, "hf_gain"), idx::HF_GAIN);
        assert_eq!(find(&V2, "pjnd_fragility"), idx::PJND_FRAGILITY);
        assert_eq!(find(&V2, "gms"), idx::GMS);
        assert_eq!(find(&V2, "edge_width_change"), idx::EDGE_WIDTH_CHANGE);
        assert_eq!(
            find(&APPEND, "xmask_transducer"),
            idx_append::XMASK_TRANSDUCER
        );
        assert_eq!(find(&APPEND, "lum_dark_err"), idx_append::LUM_DARK_ERR);
        assert_eq!(find(&APPEND, "lum_bright_err"), idx_append::LUM_BRIGHT_ERR);
        assert_eq!(find(&APPEND, "global_dmean"), idx_append::GLOBAL_DMEAN);
        assert_eq!(find(&APPEND, "global_closs"), idx_append::GLOBAL_CLOSS);
        assert_eq!(find(&APPEND, "grad_src_mean"), idx_append::GRAD_SRC_MEAN);
        assert_eq!(find(&APPEND2, "bandvis_gain"), idx_append2::BANDVIS_GAIN);
        assert_eq!(find(&APPEND2, "luma_mean_ref"), idx_append2::LUMA_MEAN_REF);
        assert_eq!(find(&APPEND2, "hl_bin2"), idx_append2::HL_BIN2);
        assert_eq!(find(&CSFW, "w_global_dmean"), idx_csfw::W_GLOBAL_DMEAN);
        assert_eq!(find(&CSFW, "w_global_closs"), idx_csfw::W_GLOBAL_CLOSS);
    }

    /// **The tranche gate.** The registry derives the free tranches from
    /// per-signal `Tranche` + `Placement` declarations; `feature_v2` derives
    /// them from hand-written loops. They must be the SAME SET.
    ///
    /// This gate is what caught the registry's first draft, which declared
    /// cost per SIGNAL and was wrong by 16 slots (it gave the luminance bins
    /// to all three channels and the `GLOBAL_*` family the `(B, scale 0)` cell
    /// the 944 walk never computes).
    #[test]
    fn tranche_slots_match_the_feature_v2_owners() {
        let raw_owner = SlotSet::from_slots(feature_v2::free_slot_indices(NS));
        assert_eq!(
            tranche_slots(Tranche::RawMoments, NS),
            raw_owner,
            "raw-moment tranche disagrees with feature_v2::free_slot_indices"
        );
        let cc_owner = SlotSet::from_slots(feature_v2::class_c_slot_indices(NS));
        assert_eq!(
            tranche_slots(Tranche::ClassC, NS),
            cc_owner,
            "class-C tranche disagrees with feature_v2::class_c_slot_indices"
        );
    }

    /// The carrier list is the owner's list.
    #[test]
    fn carrier_slots_match_the_feature_v2_owner() {
        assert_eq!(
            CARRIER_SLOTS,
            feature_v2::V1PoolsMode::CARRIER_SLOTS,
            "carrier slot list disagrees with V1PoolsMode::CARRIER_SLOTS"
        );
    }

    /// **G1.1** — for every registered `ComputeToken`, the registry's slot set
    /// equals what a `ComputeSet` with exactly that family on POPULATES.
    ///
    /// This is the load-bearing one: it makes the registry's family→slots
    /// derivation and `ComputeSet::populated_slots` provably the same
    /// function, so the plan can be built from either without them drifting.
    #[test]
    fn family_slots_match_compute_set_populated_slots() {
        use feature_v2::{ComputeSet, V1FreeExtras, V1PoolsMode};
        let width = full_width(NS);
        let off = ComputeSet {
            formula_revision: crate::ssim_form::active_revision(),
            full_res_xb: true,
            coarse_y_only_scales: 0,
            local_only: false,
            omit_edges: false,
            sampling: None,
            v1_basic: false,
            v1_pools: V1PoolsMode::Off,
            v1_full_scales: ComputeSet::ALL_SCALES,
            v1_scales: ComputeSet::ALL_SCALES,
            v2_scales: ComputeSet::ALL_SCALES,
            v2_blocks: false,
            gradient: false,
            blockiness: false,
            transducer_bank: false,
            transducers_luma_only: false,
            omit_pjnd_fragility: false,
            append: false,
            append2: false,
            append2_dst_activity: false,
            csfw: false,
            gridblk: false,
            ringbasis: false,
            tailhist: false,
            arttype: false,
            gmsbank: false,
            mapdev: false,
            z1max: false,
            gmsnative: false,
            globaff: false,
            free_extras: V1FreeExtras::Off,
        };
        // One `ComputeSet` per token that turns on EXACTLY that family.
        let cases: [(T, ComputeSet); 18] = [
            (
                T::Basic,
                ComputeSet {
                    v1_basic: true,
                    ..off
                },
            ),
            (
                T::Carriers,
                ComputeSet {
                    v1_pools: V1PoolsMode::Carriers,
                    ..off
                },
            ),
            (
                T::V2,
                ComputeSet {
                    v2_blocks: true,
                    ..off
                },
            ),
            (
                T::Append,
                ComputeSet {
                    append: true,
                    ..off
                },
            ),
            (
                T::Append2,
                ComputeSet {
                    append2: true,
                    ..off
                },
            ),
            (T::Csfw, ComputeSet { csfw: true, ..off }),
            (
                T::Gridblk,
                ComputeSet {
                    gridblk: true,
                    ..off
                },
            ),
            (
                T::Ringbasis,
                ComputeSet {
                    ringbasis: true,
                    ..off
                },
            ),
            (
                T::Tailhist,
                ComputeSet {
                    tailhist: true,
                    ..off
                },
            ),
            (
                T::Arttype,
                ComputeSet {
                    arttype: true,
                    ..off
                },
            ),
            (
                T::Gmsbank,
                ComputeSet {
                    gmsbank: true,
                    ..off
                },
            ),
            (
                T::Mapdev,
                ComputeSet {
                    mapdev: true,
                    ..off
                },
            ),
            (T::Z1max, ComputeSet { z1max: true, ..off }),
            (
                T::Gmsnative,
                ComputeSet {
                    gmsnative: true,
                    ..off
                },
            ),
            (
                T::GlobAff,
                ComputeSet {
                    globaff: true,
                    ..off
                },
            ),
            // `Peaks` alone is expressible; `Masked`/`Iw` are not (v1's pool
            // modes turn the two on together), so they are checked as the
            // DIFFERENCE between `Full` and `Peaks` below.
            (
                T::Peaks,
                ComputeSet {
                    v1_pools: V1PoolsMode::Peaks,
                    ..off
                },
            ),
            (
                T::Moments,
                ComputeSet {
                    free_extras: V1FreeExtras::RawMoments,
                    ..off
                },
            ),
            (
                T::ClassC,
                ComputeSet {
                    free_extras: V1FreeExtras::RawMomentsPlusBoundedErr,
                    ..off
                },
            ),
        ];
        for (token, cs) in cases {
            let owner = cs.populated_slots(NS, width);
            let mine = match token {
                // `RawMomentsPlusBoundedErr` populates BOTH tranches, so the
                // registry's comparison set is the union — matching the
                // owner's own `raw_moments() || bounded_err()` shape.
                T::ClassC => family_slots(T::Moments, NS).union(&family_slots(T::ClassC, NS)),
                t => family_slots(t, NS),
            };
            assert_eq!(mine, owner, "family {token}: registry vs populated_slots");
        }
        // `Masked` + `Iw` = `Full` minus `Peaks`.
        let full = ComputeSet {
            v1_pools: V1PoolsMode::Full,
            ..off
        }
        .populated_slots(NS, width);
        let peaks = ComputeSet {
            v1_pools: V1PoolsMode::Peaks,
            ..off
        }
        .populated_slots(NS, width);
        let mask_iw = SlotSet::from_slots(full.iter_slots().filter(|s| !peaks.contains(*s)));
        assert_eq!(
            family_slots(T::Masked, NS).union(&family_slots(T::Iw, NS)),
            mask_iw,
            "masked+iw: registry vs (Full minus Peaks)"
        );
    }

    /// A minimal field reader for the committed registry JSON — this crate
    /// has no serde dep and is not getting one for a test.
    fn json_str_field(obj: &str, key: &str) -> Option<String> {
        let pat = format!("\"{key}\"");
        let at = obj.find(&pat)? + pat.len();
        let rest = &obj[at..];
        let colon = rest.find(':')? + 1;
        let rest = rest[colon..].trim_start();
        // A JSON `null` is a real value here (a `kind: "class"` entry pins the
        // class and deliberately records no slot set), not a missing field —
        // returning the NEXT field's string for it is how a naive scan lies.
        if rest.starts_with("null") {
            return None;
        }
        let open = rest.find('"')? + 1;
        let rest = &rest[open..];
        let close = rest.find('"')?;
        Some(rest[..close].to_string())
    }

    /// Each `sets` entry as `(key, object-text)`.
    fn registry_sets(json: &str) -> Vec<(String, String)> {
        let sets_at = json.find("\"sets\"").expect("registry has a sets section");
        let body = &json[sets_at..];
        let mut out = Vec::new();
        let bytes = body.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            // A key is a quoted string containing "@w" followed by `: {`.
            if bytes[i] == b'"' {
                let Some(close) = body[i + 1..].find('"') else {
                    break;
                };
                let key = &body[i + 1..i + 1 + close];
                let after = &body[i + 1 + close + 1..];
                let trimmed = after.trim_start();
                if key.contains("@w") && trimmed.starts_with(':') {
                    let obj_rel = after.find('{');
                    if let Some(rel) = obj_rel {
                        let obj_start = i + 1 + close + 1 + rel;
                        let mut depth = 0i32;
                        let mut j = obj_start;
                        while j < bytes.len() {
                            match bytes[j] {
                                b'{' => depth += 1,
                                b'}' => {
                                    depth -= 1;
                                    if depth == 0 {
                                        break;
                                    }
                                }
                                _ => {}
                            }
                            j += 1;
                        }
                        out.push((
                            key.to_string(),
                            body[obj_start..=j.min(bytes.len() - 1)].to_string(),
                        ));
                        i = j + 1;
                        continue;
                    }
                }
                i += 1 + close + 1;
                continue;
            }
            i += 1;
        }
        out
    }

    /// **G1.2** — every feature-set id registered in
    /// `benchmarks/feature_sets_registry.json` is reproducible from THIS
    /// registry, at the right strength for its role.
    ///
    /// The two roles are checked differently, and the difference is the
    /// finding that shaped this gate. A **producer** id (an extractor, a
    /// features root) names what it POPULATES, which IS derivable from its
    /// compute tokens and layout width — so it is checked for EQUALITY. A
    /// **consumer** id (a bake) names what it READS, which is an arbitrary
    /// subset of some producer's set and is NOT derivable from any name — so
    /// it is checked for COVERAGE against its producer.
    ///
    /// The registry carries a live instance of exactly that distinction:
    /// `basic+peaks+moments@w944/era2r4` (producer, 265 slots, `#4fcef1d6`)
    /// and `…#0b476506` (consumer, 261 slots — the same set minus the four
    /// `LUMA_MEAN_REF` slots, which are the free set's only reference-absolute
    /// statistic and so its only slots that do not vanish at identity). Same
    /// handle, different hash, and the hash is what tells them apart — which
    /// is what `docs/FEATURE_SET_IDS.md` §2.4 means by "the name is a handle;
    /// the hash is the identity".
    #[test]
    fn registered_sets_are_reproducible_from_the_definition_registry() {
        let json = include_str!("../../benchmarks/feature_sets_registry.json");
        let (mut producers, mut consumers, mut classes) = (0usize, 0usize, 0usize);
        for (key, obj) in registry_sets(json) {
            let compute = json_str_field(&obj, "compute").expect("compute");
            let era = json_str_field(&obj, "era").expect("era");
            let role = json_str_field(&obj, "role").unwrap_or_default();
            let kind = json_str_field(&obj, "kind").unwrap_or_default();
            // A `kind: "class"` entry pins the handle without an exact slot
            // set (a bake's exact read set is derived from its bytes). There
            // is nothing to check, and inventing a set for it would be a
            // fabrication rather than a gate.
            let (Some(slots_s), Some(hash_s)) = (
                json_str_field(&obj, "slots"),
                json_str_field(&obj, "slots_hash8"),
            ) else {
                assert_eq!(
                    kind, "class",
                    "{key}: no slots/hash recorded but kind is {kind:?}, not \"class\""
                );
                classes += 1;
                continue;
            };
            let width: usize = obj
                .split("\"layout\"")
                .nth(1)
                .and_then(|t| t.split(':').nth(1))
                .and_then(|t| t.trim().trim_end_matches(',').split(',').next())
                .and_then(|t| t.trim().parse().ok())
                .expect("layout width");
            let parts = crate::feature_set_id::ComputeParts::parse(&compute)
                .unwrap_or_else(|| panic!("{key}: unparseable compute {compute:?}"));
            let recorded =
                SlotSet::parse(&slots_s).unwrap_or_else(|| panic!("{key}: unparseable slots"));
            let expect_hash = u32::from_str_radix(&hash_s, 16).expect("hex hash");

            // (a) The recorded slots hash to the recorded hash — the identity
            //     owner (`slots_hash8`) agrees with the committed value.
            assert_eq!(
                recorded.hash8(),
                expect_hash,
                "{key}: recorded slots hash {:08x} != recorded slots_hash8 {hash_s}",
                recorded.hash8()
            );

            // (b) The token-derived PRODUCER set for this compute + width.
            let mut derived = SlotSet::default();
            for t in parts.iter() {
                derived = derived.union(&family_slots(t, NS));
            }
            if parts.has_palette() {
                derived = derived.union(&family_slots(Family::Palette, NS));
            }
            let mut derived = derived.clipped_to(width);
            if let Some(selection) = json_str_field(&obj, "slot_selection") {
                derived =
                    crate::feature_plan::apply_registered_slot_selection(&selection, derived, NS);
            }
            // A set may declare `research_slots`: columns a sidecar outside zensim supplies (the by-construction
            // byc_globaff descriptor). zensim cannot derive them; everything else must still match exactly.
            let recorded = match json_str_field(&obj, "research_slots") {
                Some(r) => SlotSet::parse(&r)
                    .unwrap_or_else(|| panic!("{key}: unparseable research_slots"))
                    .missing_from(&recorded),
                None => recorded,
            };

            if role == "consumer" {
                // A read set is a subset of what its producer populates.
                assert!(
                    derived.covers(&recorded),
                    "{key}: consumer reads slots no producer with compute                      {compute:?} populates: {}",
                    derived.missing_from(&recorded)
                );
                consumers += 1;
            } else {
                assert_eq!(
                    derived, recorded,
                    "{key}: registry-derived producer slots != recorded slots"
                );
                producers += 1;
            }
            assert!(
                crate::feature_set_id::is_valid_token(&era),
                "{key}: era {era:?} is not [a-z0-9_]+"
            );
        }
        assert!(
            producers >= 10 && consumers >= 1,
            "expected the registered sets; saw {producers} producers, \
             {consumers} consumers, {classes} classes"
        );
    }
}
