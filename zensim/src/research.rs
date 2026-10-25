// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.

//! **The RESEARCH engine** — one named, complete entry that computes every
//! registered signal at any registered revision, with per-feature provenance.
//!
//! Design: `docs/FEATURE_SYSTEM_DESIGN_2026-09-05.md` §5.3. Phase + gates:
//! `docs/PLAN_FEATURE_SYSTEM_2026-09-05.md` phase 2.
//!
//! ## What this is, and what it deliberately is NOT
//!
//! The design sketched the research engine as *"buffered / oracle-backed,
//! reference semantics"* against a production engine on *"the fold, minimal
//! plan"*. **Two of those three words did not survive contact with the code,
//! and the reasons are measurements, not opinions:**
//!
//! * **It cannot be the buffered walk.** `streaming::compute_multiscale_
//!   stats_streaming` — the BUFFERED path, whose name says "streaming" for a
//!   different reason (see CLAUDE.md's naming-trap note) — has no
//!   `V2NewFeatureToggles` parameter and contains no reference to
//!   `append_block` / `csfw_block` at all. It is structurally v1-only, i.e.
//!   it tops out at 372 of the 956 registered slots. A research engine that
//!   cannot compute two thirds of the registry is not comprehensive.
//! * **It cannot be oracle-backed by default.** `feature_v2::oracle`'s
//!   `Neumaier` and `Exact` accumulators produce *different bits* from the
//!   production reduction — that is their whole purpose as a ruler. Making
//!   them the research engine's arithmetic would make G2.1 (bit-exact parity
//!   with production on every shared id) unsatisfiable **by construction**.
//!   The oracle stays what it already is: the standing precision ruler,
//!   gated separately (`era2_oracle_bounds_hold_for_every_pool_shape`).
//!
//! So the research engine is a **plan-driven, full-width, deterministic entry
//! to the fold walk** — which is exactly the machinery that already computes
//! every registered block. What it ADDS over calling the walk by hand is:
//!
//! 1. it takes a [`Plan`](crate::feature_plan::Plan) rather than a hand-built
//!    toggle struct, so "what runs" has one owner and cannot drift from "what
//!    was asked for";
//! 2. **per-feature provenance** — id, name, family, scale, channel,
//!    statistic, cost, form, direction, owning kernel, resolved revision,
//!    live defect, and whether the plan populated it or left a structural
//!    zero;
//! 3. a **revision selector** that resolves each slot's semantics from the
//!    registry and refuses LOUDLY, naming slots, when a caller asks for an
//!    era this build cannot reproduce.
//!
//! ## Determinism
//!
//! [`Request::everything`] is single-threaded by default, so a research
//! extraction is deterministic by construction. Thread invariance is still
//! GATED rather than assumed ([`tests::research_output_is_thread_invariant`]):
//! the same request with `parallel` on must be bit-identical across rayon
//! pool sizes 1/2/3/8/16 **and** equal to the serial answer — the standard
//! `v1_feature_width_pure_function.rs` already holds the v1 extractor to.

use crate::ZensimError;
use crate::feature_defs::{self, Channel, CostClass, Family, RevisionStatus};
use crate::feature_plan::{Plan, PlanError};
use crate::feature_set_id::{ComputeToken, FeatureSetId, SlotSet};
use crate::source::ImageSource;

/// The build commit this binary was compiled from, when the build environment
/// recorded one (`ZENSIM_BUILD_COMMIT`).
///
/// `None` is an HONEST state and is reported as `"unrecorded"` — never
/// guessed, and never silently filled from a runtime `git` call at this
/// layer. An extractor that CAN resolve the commit at run time should record
/// its own value beside this one and report any disagreement, which is what
/// `examples/v2_ab_extract.rs` does.
pub const BUILD_COMMIT: Option<&str> = option_env!("ZENSIM_BUILD_COMMIT");

/// The era token a signal reports when no revision has ever moved it.
///
/// Not a placeholder: a slot that has never been revised computes the same
/// quantity in every era, so it is compatible with any [`RevisionRef`].
pub const BASE_REVISION_ERA: &str = "base";

/// Which era's semantics a request wants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RevisionRef {
    /// Whatever this build computes — the current registry state.
    Current,
    /// A named era token. Servable only when every requested slot's latest
    /// LANDED revision is that era (or the slot has never been revised);
    /// otherwise the request is refused, naming the incompatible slots.
    Named(String),
}

/// Why a research extraction could not be performed.
#[derive(Debug, Clone)]
pub enum ResearchError {
    /// No registered block populates the requested slots.
    Plan(String),
    /// The named revision is not what this build computes for these slots.
    ///
    /// Carries the slots whose latest landed revision differs, so the caller
    /// learns *which* features it would have been served wrong values for
    /// rather than a bare "unsupported".
    RevisionUnavailable {
        /// The era that was asked for.
        wanted: String,
        /// Slots whose current semantics are a different era.
        incompatible: SlotSet,
        /// The DISTINCT eras those slots actually compute. A per-slot list
        /// would be 372 identical entries — measured, the first draft printed
        /// exactly that and the message ran to 6 KB.
        actual: Vec<String>,
    },
    /// The era token appears nowhere in the registry.
    RevisionUnregistered {
        /// The era that was asked for.
        wanted: String,
    },
    /// The walk itself refused (dimension mismatch, HDR on an SDR entry, …).
    Compute(ZensimError),
    /// A bake's layer-0 arities do not tile its declared input width.
    UnreadableBake,
}

impl core::fmt::Display for ResearchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ResearchError::Plan(m) => write!(f, "research: cannot plan: {m}"),
            ResearchError::RevisionUnavailable {
                wanted,
                incompatible,
                actual,
            } => write!(
                f,
                "research: revision '{wanted}' is registered but is not what \
                 this build computes for {} slot(s) — {} compute(s) {} \
                 instead. No era other than the current one can be reproduced \
                 until it is registered as a LANDED revision (a Proposed one \
                 is a priced design, not an implementation).",
                incompatible.len(),
                incompatible,
                actual.join(" / ")
            ),
            ResearchError::RevisionUnregistered { wanted } => write!(
                f,
                "research: revision '{wanted}' appears in no registry entry"
            ),
            ResearchError::Compute(e) => write!(f, "research: walk refused: {e:?}"),
            ResearchError::UnreadableBake => {
                f.write_str("research: bake layer-0 arities do not tile its input width")
            }
        }
    }
}

impl std::error::Error for ResearchError {}

impl From<PlanError> for ResearchError {
    fn from(e: PlanError) -> Self {
        match e {
            PlanError::UnreadableBake => ResearchError::UnreadableBake,
            other => ResearchError::Plan(other.to_string()),
        }
    }
}

/// A research extraction request: which slots, at which layout width, at
/// which revision.
#[derive(Debug, Clone)]
pub struct Request {
    want: SlotSet,
    layout_width: usize,
    revision: RevisionRef,
    era_label: String,
    parallel: bool,
    dense: bool,
}

impl Request {
    /// **The comprehensive request** — every registered slot, at the full
    /// registered layout width, at the current revision, single-threaded.
    #[must_use]
    pub fn everything() -> Request {
        let w = full_width();
        // `globaff` exists only from formula revision 6; below it, everything
        // is every other registered slot.
        let globaff = feature_defs::family_slots(ComputeToken::GlobAff, crate::NUM_SCALES);
        let computes_globaff = crate::ssim_form::active_revision().computes_globaff();
        Request {
            want: SlotSet::from_slots((0..w).filter(|&id| {
                feature_defs::def_at(id, crate::NUM_SCALES).is_some()
                    && (computes_globaff || !globaff.contains(id))
            })),
            layout_width: w,
            revision: RevisionRef::Current,
            era_label: crate::feature_set_id::ERA_UNKNOWN.to_string(),
            parallel: false,
            dense: false,
        }
    }

    /// An explicit slot request at an explicit layout width.
    #[must_use]
    pub fn for_slots(want: SlotSet, layout_width: usize) -> Request {
        Request {
            want,
            layout_width,
            revision: RevisionRef::Current,
            era_label: crate::feature_set_id::ERA_UNKNOWN.to_string(),
            parallel: false,
            dense: false,
        }
    }

    /// Reproduce a registered feature set: the union of its compute tokens'
    /// slots, at its declared layout width.
    ///
    /// The id's own slot hash is CHECKED against the reconstruction — a
    /// `FeatureSetId` whose hash disagrees with its compute tokens describes
    /// a set this build cannot reproduce, and saying so is the point of the
    /// identity layer.
    pub fn for_set(id: &FeatureSetId) -> Result<Request, ResearchError> {
        // ONE owner for "which slots does this id name" —
        // `feature_layout::slots_of`, which tries both the sparse and the
        // dense reading and lets the recorded hash decide. Reconstructing it
        // here as well is exactly the drift the no-duplicate rule stops: the
        // first draft did, clipped to `layout_width` unconditionally, and so
        // refused every DENSE id.
        let Some(want) = crate::feature_layout::slots_of(id) else {
            return Err(ResearchError::Plan(format!(
                "feature-set id {id} names a slot set this build cannot \
                 reproduce — neither the sparse reading (clip to its declared \
                 width) nor the dense one (packed, clipped to the registry) \
                 hashes to {}",
                crate::feature_set_id::hex8(id.slots_hash())
            )));
        };
        // A dense id names FEWER positions than the highest id it carries,
        // and reproducing it means emitting into that dense layout — not into
        // an identity layout of its packed width, which would be a different
        // table entirely.
        //
        // **Sparse-vs-dense CANNOT be recovered from the slots.** The same 265
        // ids are a sparse layout at w944 (their own indices, 679 structural
        // fills) and a dense one at w265 (packed, no gaps) — two different
        // tables. That is precisely why the layout survives as an optional
        // `@w<N>` hint on an id whose IDENTITY no longer includes it: the hash
        // says WHICH set, the width says HOW it was written.
        //
        // With a width: the historical rule, unchanged — dense iff the width
        // equals the member count and the set is not already the identity
        // range. Without one: the SPARSE reading, because it is the
        // conservative default (emitting a set at its own indices can only add
        // structural fills, while guessing "dense" would PERMUTE the vector).
        // A caller who means the dense layout says so, either by keeping the
        // `@w<packed>` hint on the id or by using
        // [`Request::for_slots`] + [`Request::dense`].
        let (dense, layout_width) = match id.layout_width() {
            Some(w) => (
                want.len() == w && want.iter_slots().last() != Some(w - 1),
                w,
            ),
            None => (false, want.iter_slots().last().map_or(0, |s| s + 1)),
        };
        Ok(Request {
            want,
            layout_width,
            revision: RevisionRef::Current,
            era_label: id.era().to_string(),
            parallel: false,
            dense,
        })
    }

    /// The read set of a serialized bake, at its declared caller width — the
    /// research-side twin of the runtime's servability entry.
    pub fn for_bake_bytes(bytes: &[u8]) -> Result<Request, ResearchError> {
        let model =
            crate::mlp::Model::from_bytes(bytes).map_err(|_| ResearchError::UnreadableBake)?;
        // This untyped slot request cannot carry a sampling contract. Use the
        // complete BakeScorer pixel surface for sampling-bound extraction.
        if crate::sampling::Sampling::from_model(&model)
            .map_err(|_| ResearchError::UnreadableBake)?
            .is_some()
        {
            return Err(ResearchError::UnreadableBake);
        }
        let want =
            crate::feature_plan::bake_read_slots(&model).ok_or(ResearchError::UnreadableBake)?;
        // `want` is in ID space. For an identity bake the caller width IS the
        // walk width, but a DENSE bake's caller width is its packed size — 28
        // for shipped `D`, whose ids reach `f155` — so clipping to it would
        // silently delete every id above the packed count and hand back a
        // request for a fraction of what the bake reads. The layout's
        // `walk_width` (one past the highest id it carries) is the width that
        // makes the clip a no-op for both kinds.
        let layout = crate::feature_layout::declared_layout(&model);
        let layout_width = layout.walk_width();
        Ok(Request {
            want: want.clipped_to(layout_width),
            layout_width,
            revision: RevisionRef::Current,
            era_label: crate::feature_set_id::ERA_UNKNOWN.to_string(),
            parallel: false,
            dense: false,
        })
    }

    /// Emit into a **DENSE** layout: the requested ids packed in ascending
    /// order with no gaps, so the vector is `want.len()` wide instead of the
    /// declared width's worth of mostly-structural-fill.
    ///
    /// This is what retires "944-with-structural-zeros as the wire format"
    /// for NEW artifacts. It changes no existing one: every stored table and
    /// every shipped bake is in a legacy IDENTITY layout, which remains a
    /// declared layout over the same ids.
    #[must_use]
    pub fn dense(mut self) -> Request {
        self.dense = true;
        self
    }

    /// Ask for a specific era's semantics.
    #[must_use]
    pub fn at_revision(mut self, revision: RevisionRef) -> Request {
        self.revision = revision;
        self
    }

    /// Stamp the extraction with an EXTRACTOR era token (the
    /// `feature_set_id` grammar's third field, e.g. `era2r4`). Orthogonal to
    /// [`RevisionRef`], which is the per-SIGNAL axis.
    #[must_use]
    pub fn with_era_label(mut self, era: impl Into<String>) -> Request {
        self.era_label = era.into();
        self
    }

    /// Run the walk multi-threaded. OFF by default: a research extraction is
    /// deterministic by construction, and thread invariance is a gate rather
    /// than an assumption.
    #[must_use]
    pub fn with_parallel(mut self, parallel: bool) -> Request {
        self.parallel = parallel;
        self
    }

    /// The slots asked for.
    #[must_use]
    pub fn want(&self) -> &SlotSet {
        &self.want
    }

    /// The declared emit width.
    #[must_use]
    pub fn layout_width(&self) -> usize {
        self.layout_width
    }

    /// The [`Layout`](crate::feature_layout::Layout) this request emits into.
    fn layout(&self) -> crate::feature_layout::Layout {
        if self.dense {
            crate::feature_layout::Layout::dense(&self.want)
        } else {
            crate::feature_layout::Layout::identity(self.layout_width)
        }
    }

    // Research-only extension; the existing fold plan sees exactly its old IDs
    // and layout. This deliberately does not relax Plan::for_bake's scope.
    fn plan(&self) -> Result<Plan, ResearchError> {
        let globaff = self.want.intersect(&feature_defs::family_slots(
            ComputeToken::GlobAff,
            crate::NUM_SCALES,
        ));
        if !globaff.is_empty() && !crate::ssim_form::active_revision().computes_globaff() {
            return Err(ResearchError::Plan(format!(
                "globaff slot(s) {globaff} are computed only at formula revision 6 \
                 (ZENSIM_FORMULA_REV=6)"
            )));
        }
        let palette = self.want.intersect(&feature_defs::family_slots(
            Family::Palette,
            crate::NUM_SCALES,
        ));
        if palette.is_empty() {
            return Ok(Plan::derive_with_layout(&self.want, self.layout())?);
        }
        if self.era_label == "palette_v1" {
            return Err(ResearchError::RevisionUnavailable {
                wanted: self.era_label.clone(),
                incompatible: palette,
                actual: vec!["palette_v2".to_string()],
            });
        }
        #[cfg(not(feature = "training"))]
        return Err(ResearchError::Plan(
            "palette requires the training research build".into(),
        ));
        #[cfg(feature = "training")]
        {
            // Every walk-computed slot: below the palette, plus `globaff`
            // after it (whose identity layout then spans the palette's zeros,
            // filled below).
            let legacy = palette.missing_from(&self.want);
            let legacy_layout = if self.dense {
                crate::feature_layout::Layout::dense(&legacy)
            } else if globaff.is_empty() {
                crate::feature_layout::Layout::identity(self.layout_width.min(crate::palette::BASE))
            } else {
                crate::feature_layout::Layout::identity(self.layout_width)
            };
            let mut plan = Plan::derive_with_layout(&legacy, legacy_layout)?;
            plan.layout = self.layout();
            // An empty legacy request must not acquire the old plan's implicit
            // basic carriers in a sparse palette-only layout.
            if legacy.is_empty() {
                plan.emit = SlotSet::default();
            }
            plan.emit = plan.emit.union(&palette.clipped_to(plan.walk_width()));
            Ok(plan)
        }
    }

    /// Check the request WITHOUT touching an image: can it be planned, and
    /// can this build reproduce the revision it names?
    ///
    /// Exists because the alternative is finding out per pair. A batch
    /// extractor that only learns its request is unservable inside the
    /// per-pair body prints one refusal per row — measured at 4 identical
    /// multi-kilobyte messages on a 4-pair smoke run, and it would be 200,000
    /// on a real corpus, after paying for every decode.
    ///
    /// Returns the resolved plan's emitted slot set on success.
    ///
    /// # Errors
    ///
    /// [`ResearchError::Plan`] / [`ResearchError::RevisionUnavailable`] /
    /// [`ResearchError::RevisionUnregistered`], exactly as [`extract`] would.
    pub fn validate(&self) -> Result<SlotSet, ResearchError> {
        let plan = self.plan()?;
        check_revision(self, &plan.emit)?;
        Ok(plan.emit)
    }
}

/// Everything the registry knows about one emitted position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureProvenance {
    /// The stable slot id — equal to the position under every layout
    /// registered today (all five legacy widths are identity mappings).
    pub id: u16,
    /// `<family>_<signal>_s<scale>_<channel>`.
    pub name: String,
    /// The `ComputeToken` vocabulary token.
    pub family: &'static str,
    /// Pyramid scale.
    pub scale: u8,
    /// `x` | `y` | `b` | `s` (per-scale slots have no channel axis).
    pub channel: &'static str,
    /// How the signal pools its per-pixel term.
    pub statistic: &'static str,
    /// The honest PER-SLOT cost — `free` when a tranche harvests this
    /// placement without its owning block, else the signal's base cost.
    pub cost: &'static str,
    /// Which free tranche can harvest it, if any.
    pub tranche: &'static str,
    /// Identity-pair behaviour: `difference` slots are 0 on a perfect copy,
    /// `reference_only` ones are not.
    pub form: &'static str,
    /// The monotone expectation a dial gate may rely on.
    pub direction: &'static str,
    /// The kernel that owns this signal's accumulation.
    pub kernel: &'static str,
    /// The era whose semantics this value carries — the latest LANDED
    /// revision, or [`BASE_REVISION_ERA`] when the signal has never moved.
    pub revision_era: String,
    /// The byte-changing commit of that revision, or `"-"`.
    pub revision_commit: &'static str,
    /// A registered PROPOSED revision that has NOT been applied, if any.
    /// Present means "a fix is designed and priced", never "a fix is in".
    ///
    /// It reports the FIRST registered-but-unapplied era on this signal.
    ///
    /// **It under-reports a slot that is in two proposed eras**, and 48 slots
    /// now are: every SSIM-derived `L4`/`L8` pool is in `v1ssimcap` (F4, the
    /// unbounded luminance term) AND `v1detroot` (F18, the libc-dependent
    /// pooled root). This field reports `"v1ssimcap"` for them and says
    /// nothing about the second. Widening it to a list would change the
    /// public API, which this crate's snapshot gate forbids, so the
    /// limitation is recorded rather than papered over: the COMPLETE answer
    /// is `feature_defs::era_moved_slots(<era>, …)`, which reads the whole
    /// revision list, and `registered_defects_cover_exactly_the_audited_slots`
    /// holds it against the statistic it is derived from.
    pub proposed_revision: Option<&'static str>,
    /// A live defect id from `docs/FEATURE_DEFECTS_AUDIT_2026-09-05.md`.
    pub defect: Option<&'static str>,
    /// Retired slot: keeps its id and its structural zero forever.
    pub deprecated: bool,
    /// Did the plan POPULATE this position, or is the value a structural
    /// zero the layout declares but the plan does not compute?
    pub populated: bool,
}

/// The result of a research extraction: values, and what each one IS.
#[derive(Debug, Clone)]
pub struct Extraction {
    values: Vec<f64>,
    provenance: Vec<FeatureProvenance>,
    layout: crate::feature_layout::Layout,
    n_scales: usize,
    emitted: SlotSet,
    feature_set_id: Option<FeatureSetId>,
    revision: RevisionRef,
    build_commit: Option<&'static str>,
}

impl Extraction {
    /// The emitted vector, `layout_width` long.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Consume into the emitted vector.
    #[must_use]
    pub fn into_values(self) -> Vec<f64> {
        self.values
    }

    /// One entry per emitted position, index-aligned with [`Self::values`].
    #[must_use]
    pub fn provenance(&self) -> &[FeatureProvenance] {
        &self.provenance
    }

    /// The declared layout width.
    #[must_use]
    pub fn layout_width(&self) -> usize {
        self.layout.width()
    }

    /// The layout's stable name — `w944`, `dense265`.
    #[must_use]
    pub fn layout_name(&self) -> &str {
        self.layout.name()
    }

    /// Where feature `id` sits in this extraction's vector, or `None` when
    /// the layout does not carry it.
    ///
    /// The reverse index a manifest reader wants: *"which column is f353?"*
    /// is the identity under every legacy width and is NOT under a dense one,
    /// which is exactly why asking is better than assuming.
    #[must_use]
    pub fn position_of(&self, id: u16) -> Option<usize> {
        self.layout.pos_of(id)
    }

    /// Pyramid scale count.
    #[must_use]
    pub fn n_scales(&self) -> usize {
        self.n_scales
    }

    /// The slots the plan populated. Everything else in the layout is a
    /// structural zero.
    #[must_use]
    pub fn emitted(&self) -> &SlotSet {
        &self.emitted
    }

    /// The producer feature-set id, when the compute set has one.
    #[must_use]
    pub fn feature_set_id(&self) -> Option<&FeatureSetId> {
        self.feature_set_id.as_ref()
    }

    /// The revision that was asked for.
    #[must_use]
    pub fn revision(&self) -> &RevisionRef {
        &self.revision
    }

    /// The recorded build commit, or `"unrecorded"`.
    #[must_use]
    pub fn build_commit(&self) -> &str {
        self.build_commit.unwrap_or("unrecorded")
    }

    /// A JSON object carrying everything a table manifest needs to say what
    /// its columns ARE — the provenance half of `_MANIFEST.json`.
    ///
    /// Hand-rendered rather than `serde`-derived: `zensim` carries no JSON
    /// dependency, and adding one to ship a manifest writer would be a
    /// dependency for a string.
    #[must_use]
    pub fn manifest_json(&self) -> String {
        let mut s = String::with_capacity(256 * self.provenance.len() + 512);
        s.push_str("{\n");
        s.push_str(&format!(
            "  \"layout\": \"{}\",\n  \"layout_width\": {},\n  \"layout_is_identity\": {},\n",
            self.layout.name(),
            self.layout.width(),
            self.layout.is_identity()
        ));
        s.push_str(&format!("  \"n_scales\": {},\n", self.n_scales));
        s.push_str(&format!(
            "  \"feature_set_id\": {},\n",
            match &self.feature_set_id {
                Some(id) => format!("\"{id}\""),
                None => "null".to_string(),
            }
        ));
        s.push_str(&format!("  \"emitted_slots\": \"{}\",\n", self.emitted));
        s.push_str(&format!(
            "  \"emitted_slot_count\": {},\n",
            self.emitted.len()
        ));
        s.push_str(&format!(
            "  \"revision\": \"{}\",\n",
            match &self.revision {
                RevisionRef::Current => "current".to_string(),
                RevisionRef::Named(n) => n.clone(),
            }
        ));
        s.push_str(&format!(
            "  \"zensim_build_commit\": \"{}\",\n",
            self.build_commit()
        ));
        // WHICH ARITHMETIC RAN. Distinct from `revision` above (what was
        // ASKED for): the revision lane's switch can be pinned per process
        // with `ZENSIM_FORMULA_REV`, so a table's manifest must record the
        // formula era it was actually produced under, not the default.
        s.push_str(&format!(
            "  \"formula_revision\": \"{:?}\",\n  \"formula_revision_eras\": [{}],\n",
            crate::ssim_form::active_revision(),
            active_era_tokens()
                .iter()
                .map(|t| format!("\"{t}\""))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        // The ringing measurement arm (`ZENSIM_RING_FORM`), so a table made
        // under the high-pass arm can never pass for a shipped-form table.
        s.push_str(&format!(
            "  \"ring_form\": \"{}\",\n",
            if crate::feature_v2::process_ring_high_pass() {
                crate::feature_v2::RingForm::HighPassMu.as_str()
            } else {
                crate::feature_v2::RingForm::Raw.as_str()
            }
        ));
        // The pyramid arm (`ZENSIM_PYRAMID`): coarse levels averaged in XYB
        // (the shipped form) or in linear RGB.
        s.push_str(&format!(
            "  \"pyramid\": \"{}\",\n",
            if crate::feature_v2_stream::linear_pyramid() {
                "linear"
            } else {
                "xyb"
            }
        ));
        s.push_str("  \"features\": [\n");
        for (i, p) in self.provenance.iter().enumerate() {
            s.push_str(&format!(
                "    {{\"pos\": {}, \"id\": {}, \"name\": \"{}\", \"family\": \"{}\", \
                 \"scale\": {}, \"channel\": \"{}\", \"statistic\": \"{}\", \
                 \"cost\": \"{}\", \"tranche\": \"{}\", \"form\": \"{}\", \
                 \"direction\": \"{}\", \"kernel\": \"{}\", \"revision\": \"{}\", \
                 \"revision_commit\": \"{}\", \"proposed_revision\": {}, \
                 \"defect\": {}, \"deprecated\": {}, \"populated\": {}}}{}\n",
                i,
                p.id,
                p.name,
                p.family,
                p.scale,
                p.channel,
                p.statistic,
                p.cost,
                p.tranche,
                p.form,
                p.direction,
                p.kernel,
                p.revision_era,
                p.revision_commit,
                match p.proposed_revision {
                    Some(r) => format!("\"{r}\""),
                    None => "null".to_string(),
                },
                match p.defect {
                    Some(d) => format!("\"{d}\""),
                    None => "null".to_string(),
                },
                p.deprecated,
                p.populated,
                if i + 1 == self.provenance.len() {
                    ""
                } else {
                    ","
                }
            ));
        }
        s.push_str("  ]\n}\n");
        s
    }
}

/// Does THIS build compute a given registered revision?
///
/// A LANDED revision is by definition what the shipped build computes. A
/// PROPOSED one is not — *unless* the revision lane's arithmetic switch is
/// pinned to the [`feature_defs::FormulaRevision`] that introduces it, which
/// is exactly what `ZENSIM_FORMULA_REV=2` does. Asking that switch instead of
/// assuming the default is what makes the two selectors compose.
fn build_computes(r: &feature_defs::Revision) -> bool {
    r.status == RevisionStatus::Landed || active_era_tokens().contains(&r.era)
}

/// The era a signal's values carry **in this build**: its latest revision
/// that this build actually computes, or [`BASE_REVISION_ERA`].
///
/// Not "latest LANDED": under `ZENSIM_FORMULA_REV=2` a Proposed entry IS what
/// runs, and reporting the landed one would make the provenance lie about the
/// bytes beside it.
fn current_era_of(signal: &'static feature_defs::SignalDef) -> &'static str {
    if signal.family == Family::Palette {
        return "palette_v2";
    }
    // An active ARITHMETIC era (Rev4's `tiercanon`) moved every slot, so it
    // is every slot's era — registered in `feature_defs::ARITHMETIC_REVISIONS`,
    // not per signal.
    if let Some(r) = active_arithmetic_revision() {
        return r.era;
    }
    signal
        .revisions
        .iter()
        .rfind(|r| build_computes(r))
        .map_or(BASE_REVISION_ERA, |r| r.era)
}

/// The commit of a signal's effective revision, or `"-"`.
fn current_commit_of(signal: &'static feature_defs::SignalDef) -> &'static str {
    if signal.family == Family::Palette {
        return BUILD_COMMIT.unwrap_or("unrecorded");
    }
    if let Some(r) = active_arithmetic_revision() {
        return r.commit;
    }
    signal
        .revisions
        .iter()
        .rfind(|r| build_computes(r))
        .map_or("-", |r| r.commit)
}

/// Is a signal's value in THIS build the same value era `wanted` defines?
///
/// Two ways to be compatible, and the second is the one the first draft
/// missed:
///
/// 1. the signal's EFFECTIVE era is exactly `wanted`; or
/// 2. **`wanted` never touched this signal** — no revision entry names it —
///    in which case the signal computes the same quantity in `wanted`'s world
///    as in any other, so serving it is correct.
///
/// The first draft used only rule 1 and so refused 156 of 156 basic slots for
/// `v1ssimcap` when only **36** carry it; under `ZENSIM_FORMULA_REV=2` it
/// still refused 120 slots the fix does not move. An era is a boundary in
/// TIME, not a label every slot must wear.
///
/// An ARITHMETIC era touches every signal, so rule 2 never applies to it — and
/// while one is active, it has touched every signal for every other `wanted`
/// too: no slot's value is any earlier era's value.
fn signal_matches_era(signal: &'static feature_defs::SignalDef, wanted: &str) -> bool {
    if current_era_of(signal) == wanted {
        return true;
    }
    if feature_defs::is_arithmetic_era(wanted) || active_arithmetic_revision().is_some() {
        return false;
    }
    !signal.revisions.iter().any(|r| r.era == wanted)
}

/// The arithmetic era this build computes, if any (the latest one named by
/// the active revision's tokens).
fn active_arithmetic_revision() -> Option<&'static feature_defs::Revision> {
    active_era_tokens()
        .iter()
        .rev()
        .find_map(|t| feature_defs::arithmetic_revision(t))
}

/// The first registered PROPOSED revision for a signal, if any.
fn proposed_era_of(signal: &'static feature_defs::SignalDef) -> Option<&'static str> {
    signal
        .revisions
        .iter()
        .find(|r| r.status == RevisionStatus::Proposed)
        .map(|r| r.era)
}

/// Is `era` a token any registry entry names?
fn era_is_registered(era: &str) -> bool {
    // BOTH registries. A feature era is derivable from the signal table; a
    // SCORE era (F19's `scorepow`) has no slots to derive from and lives in
    // `feature_defs::SCORE_PATH_REVISIONS`. Consulting only the first would
    // have made this invariant fail the moment a score era joined a revision
    // — which is a registry gap, not a reason to weaken the assertion.
    feature_defs::signals().any(|s| s.revisions.iter().any(|r| r.era == era))
        || feature_defs::is_score_path_era(era)
        || feature_defs::is_arithmetic_era(era)
}

/// Era tokens this BUILD is actually computing right now.
///
/// The revision lane owns the arithmetic switch
/// (`ssim_form::active_revision`, pinnable with `ZENSIM_FORMULA_REV`); this
/// asks it, rather than assuming the shipped revision. So when a build is
/// pinned to `Rev2`, a `RevisionRef::Named("v1ssimcap")` request is SERVABLE
/// here with no change to this module — which is the whole point of the two
/// selectors being separate: theirs decides which formula runs, mine decides
/// whether the caller is allowed to be told it got the era it asked for.
fn active_era_tokens() -> &'static [&'static str] {
    crate::ssim_form::active_revision().era_tokens()
}

/// Resolve the request's [`RevisionRef`] against the registry.
///
/// A `Named` era is servable only when every requested slot's CURRENT
/// semantics are that era, or the slot has never been revised. Anything else
/// is refused, naming the slots — because reproducing a superseded era needs
/// the superseded code, and this build does not have it.
fn check_revision(req: &Request, emit: &SlotSet) -> Result<(), ResearchError> {
    let RevisionRef::Named(wanted) = &req.revision else {
        return Ok(());
    };
    // A token no registry entry names is a TYPO, and saying so is more
    // actionable than listing the slots it disagrees with — every slot
    // disagrees with a name that does not exist. Decided FIRST, so the
    // spelling error is never buried inside a slot list.
    if !era_is_registered(wanted) {
        return Err(ResearchError::RevisionUnregistered {
            wanted: wanted.clone(),
        });
    }
    let ns = crate::NUM_SCALES;
    let mut bad = Vec::new();
    let mut actual: Vec<String> = Vec::new();
    for id in emit.iter_slots() {
        let Some(d) = feature_defs::def_at(id, ns) else {
            continue;
        };
        if signal_matches_era(d.signal, wanted) {
            continue;
        }
        bad.push(id);
        // DISTINCT eras only — the per-slot form is unreadable and carries no
        // extra information (`incompatible` already names every slot, as a
        // compact range list).
        let era = current_era_of(d.signal);
        if !actual.iter().any(|a| a == era) {
            actual.push(era.to_string());
        }
    }
    if bad.is_empty() {
        return Ok(());
    }
    Err(ResearchError::RevisionUnavailable {
        wanted: wanted.clone(),
        incompatible: SlotSet::from_slots(bad),
        actual,
    })
}

/// Provenance for every POSITION of a plan's layout.
///
/// Position and id are the same number under every layout that exists today
/// (all five legacy widths are identity mappings) and are NOT the same number
/// under a dense one — so the id comes from `layout.slot_at(pos)`, never from
/// the position, and `populated` is asked of the id.
fn provenance_for(plan: &Plan, n_scales: usize) -> Vec<FeatureProvenance> {
    (0..plan.layout_width())
        .map(|pos| {
            let id = plan.layout.slot_at(pos).map(usize::from);
            let d = id.and_then(|id| feature_defs::def_at(id, n_scales));
            match d {
                Some(d) => FeatureProvenance {
                    id: d.id,
                    name: d.name(),
                    family: d.signal.family.as_str(),
                    scale: d.scale,
                    channel: d.channel.as_str(),
                    statistic: d.signal.statistic.as_str(),
                    cost: id
                        .and_then(|id| feature_defs::cost_of(id, n_scales))
                        .unwrap_or(CostClass::Expensive)
                        .as_str(),
                    tranche: d.signal.tranche.as_str(),
                    form: d.signal.form.as_str(),
                    direction: d.signal.direction.as_str(),
                    kernel: d.signal.kernel.as_str(),
                    revision_era: current_era_of(d.signal).to_string(),
                    revision_commit: current_commit_of(d.signal),
                    proposed_revision: proposed_era_of(d.signal),
                    defect: d.signal.defect.map(|x| x.id),
                    deprecated: d.signal.deprecated,
                    populated: id.is_some_and(|id| plan.emit.contains(id)),
                },
                // Past the registry: an UNREGISTERED position. Reported as
                // such rather than omitted, so a width that outruns the
                // registry is visible in the manifest instead of silent.
                None => FeatureProvenance {
                    id: id.and_then(|i| u16::try_from(i).ok()).unwrap_or(u16::MAX),
                    name: match id {
                        Some(i) => format!("unregistered_f{i}"),
                        None => format!("declared_gap_at_{pos}"),
                    },
                    family: "unregistered",
                    scale: 0,
                    channel: Channel::Scalar.as_str(),
                    statistic: "undeclared",
                    cost: "undeclared",
                    tranche: "none",
                    form: "undeclared",
                    direction: "undeclared",
                    kernel: "none",
                    revision_era: BASE_REVISION_ERA.to_string(),
                    revision_commit: "-",
                    proposed_revision: None,
                    defect: None,
                    deprecated: false,
                    populated: false,
                },
            }
        })
        .collect()
}

/// Run a research extraction over one SDR pair.
///
/// The walk is the same folded owner as `Zensim::compute_folded720_features_streaming`,
/// driven by the complete plan (including its channel selection), so
/// bit-exact parity on every shared id is a property of the code rather than
/// of a second implementation kept in step by hand.
///
/// # Errors
///
/// [`ResearchError`] when the request cannot be planned, the revision cannot
/// be reproduced, or the walk itself refuses.
pub fn extract(
    req: &Request,
    source: &impl ImageSource,
    distorted: &impl ImageSource,
) -> Result<Extraction, ResearchError> {
    extract_impl(req, source, distorted, None)
}

/// Extract an explicit research read set through the production native HDR walk.
///
/// Input formats, alpha, primaries and stride follow
/// [`crate::Zensim::compute_folded720_features_hdr`]. The explicit encoding
/// supplies the absolute-light display model; no SDR conversion is performed.
/// Planning, gathering and provenance are shared with [`extract`].
///
/// # Errors
/// Refuses unsupported HDR input, unreproducible revisions, or invalid plans.
pub fn extract_hdr(
    req: &Request,
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    encoding: crate::feature_v2::HdrEncoding,
) -> Result<Extraction, ResearchError> {
    extract_impl(req, source, distorted, Some(encoding))
}

fn extract_impl(
    req: &Request,
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    hdr: Option<crate::feature_v2::HdrEncoding>,
) -> Result<Extraction, ResearchError> {
    let ns = crate::NUM_SCALES;
    let plan = req.plan()?;
    check_revision(req, &plan.emit)?;

    let mut scratch = crate::feature_v2::V2Scratch::new();
    let toggles = plan.toggles();
    #[cfg(feature = "training")]
    let extras = crate::feature_v2::FoldWalkExtras {
        compute: Some(plan.compute),
        ..Default::default()
    };
    #[cfg(not(feature = "training"))]
    let extras = crate::feature_v2::FoldWalkExtras {
        compute: Some(plan.compute),
        ..Default::default()
    };
    let palette_slots = feature_defs::family_slots(Family::Palette, ns);
    let palette_only =
        !plan.emit.is_empty() && plan.emit.iter_slots().all(|id| palette_slots.contains(id));
    #[allow(unused_mut)] // Mutated by the training-only palette block.
    let mut walk = if palette_only {
        Vec::new()
    } else {
        let result = if let Some(encoding) = hdr {
            crate::feature_v2::compute_folded720_hdr_streaming_extras(
                source,
                distorted,
                encoding,
                Some(120_000_000),
                req.parallel,
                toggles,
                &mut scratch,
                extras,
            )
        } else {
            crate::feature_v2::compute_folded720_streaming_extras(
                source,
                distorted,
                Some(120_000_000),
                req.parallel,
                toggles,
                &mut scratch,
                extras,
            )
        }
        .map_err(ResearchError::Compute)?;
        result.into_features()
    };

    // GATHER into the declared LAYOUT. The walk emits at its own identity
    // width; the layout says which id lives at which position, so a narrower
    // declared layout takes the prefix, a wider one gets the structural fill
    // it declares, and a DENSE one is packed. `plan.emit` remains the
    // authority on which positions carry a computed number — the byte at an
    // unpopulated position is its finaliser's degenerate value, which is not
    // always `0.0` (see `nonzero_structural_fill_slots`).
    #[cfg(feature = "training")]
    if !plan
        .emit
        .intersect(&feature_defs::family_slots(
            Family::Palette,
            crate::NUM_SCALES,
        ))
        .is_empty()
    {
        if hdr.is_some() {
            return Err(ResearchError::Plan(
                "palette_v2 does not support native HDR".into(),
            ));
        }
        let palette = crate::palette::extract(source, distorted)?;
        walk.resize(
            walk.len().max(crate::palette::BASE + crate::palette::WIDTH),
            0.0,
        );
        for (offset, value) in palette.into_iter().enumerate() {
            if plan.emit.contains(crate::palette::BASE + offset) {
                walk[crate::palette::BASE + offset] = value;
            }
        }
    }
    let mut values = Vec::new();
    plan.layout.gather(&walk, &mut values);

    let provenance = provenance_for(&plan, ns);
    // The producer id: COMPUTE tokens from the plan, LAYOUT width from the
    // layout, and the slot hash from what was actually POPULATED.
    //
    // The two-step exists because `ComputeSet::feature_set_id` derives the
    // hash from `populated_slots` clipped to the width it is handed, and for
    // a DENSE layout that width is the PACKED count — clipping the family
    // union to 265 reconstructs a completely different set. MEASURED: the
    // first version emitted `basic+peaks+moments@w265#3fb78648` where the
    // set's real hash is `#4fcef1d6`, and worse, `declared_layout` would then
    // have accepted the wrong reconstruction because it also has 265 members.
    // So the tokens come from the walk-width call and the hash comes from
    // `plan.emit`, which is already in id space.
    let feature_set_id = plan
        .compute
        .feature_set_id(ns, plan.walk_width(), &req.era_label)
        .and_then(|id| {
            // The emitted width is recorded as the legacy `@w<N>` hint so a
            // reader can rebuild a SPARSE set without searching candidate
            // widths; it is not part of the id's identity.
            FeatureSetId::new_with_layout(
                if plan
                    .emit
                    .intersect(&feature_defs::family_slots(
                        Family::Palette,
                        crate::NUM_SCALES,
                    ))
                    .is_empty()
                {
                    id.compute()
                } else {
                    if palette_only {
                        crate::feature_set_id::ComputeParts::EMPTY.with_palette()
                    } else {
                        id.compute().with_palette()
                    }
                },
                plan.layout_width(),
                &req.era_label,
                plan.emit.hash8(),
            )
        })
        .or_else(|| {
            if plan
                .emit
                .intersect(&feature_defs::family_slots(
                    Family::Palette,
                    crate::NUM_SCALES,
                ))
                .is_empty()
            {
                None
            } else {
                FeatureSetId::new_with_layout(
                    if palette_only {
                        crate::feature_set_id::ComputeParts::EMPTY.with_palette()
                    } else {
                        plan.compute.compute_parts().with_palette()
                    },
                    plan.layout_width(),
                    &req.era_label,
                    plan.emit.hash8(),
                )
            }
        });

    Ok(Extraction {
        values,
        provenance,
        layout: plan.layout,
        n_scales: ns,
        emitted: plan.emit,
        feature_set_id,
        revision: req.revision.clone(),
        build_commit: BUILD_COMMIT,
    })
}

/// The slots whose UNPOPULATED value is **not** `0.0`.
///
/// "A position the plan does not compute is a structural zero" is the
/// documented contract (`ZensimV2Result`'s own doc: *"a v1-only 944 request
/// is still a 944 row with `f372..` at the structural 0.0"*). MEASURED
/// 2026-09-05, it is true of **560 of the 572** positions a `v1_only` 944
/// walk leaves alone, and FALSE of twelve: the `pjnd_fragility` slots come
/// back as exactly `1.0` — one per (scale, channel) cell — because the v2
/// dense block's fragility finaliser returns `1.0` for its degenerate
/// no-samples case, and it runs whether or not the kernel that fills its
/// accumulators did.
///
/// Those twelve are exactly the slots the defect audit already tagged
/// **F15** (*"`PJND_FRAGILITY` is nonzero on an identity pair"*) — the same
/// finaliser, observed in a second place. So this set is DERIVED from the
/// registry's defect field rather than hard-coded as twelve indices, and it
/// will follow the fix when the revision lane lands one.
///
/// This lane does NOT change the finaliser: kernel arithmetic belongs to the
/// revision lane, and a fix here would move shipped bytes. What it does is
/// stop the gates from asserting something false, and name why.
#[must_use]
pub fn nonzero_structural_fill_slots() -> SlotSet {
    let ns = crate::NUM_SCALES;
    SlotSet::from_slots((0..feature_defs::full_width(ns)).filter(|&id| {
        feature_defs::def_at(id, ns)
            .and_then(|d| d.signal.defect)
            .is_some_and(|x| x.id == "F15")
    }))
}

/// Every registered [`ComputeToken`], for callers building a request from a
/// family list.
#[must_use]
pub fn family_slots(family: ComputeToken) -> SlotSet {
    feature_defs::family_slots(family, crate::NUM_SCALES)
}

/// The full registered layout width at the shipped scale count.
#[must_use]
pub fn full_width() -> usize {
    #[cfg(feature = "training")]
    {
        feature_defs::full_width(crate::NUM_SCALES)
    }
    #[cfg(not(feature = "training"))]
    {
        feature_defs::block_base(Family::Palette, crate::NUM_SCALES)
            .expect("palette block")
            .0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RgbSlice;

    /// A deterministic non-identical SDR pair.
    fn pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
        let mut src = vec![[0u8; 3]; w * h];
        let mut dst = vec![[0u8; 3]; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let v = ((x * 7 + y * 13) % 251) as u8;
                src[i] = [v, v.wrapping_add(40), v.wrapping_mul(3)];
                dst[i] = [
                    v & 0xF0,
                    v.wrapping_add(37),
                    v.wrapping_mul(3).wrapping_sub(9),
                ];
            }
        }
        (src, dst)
    }

    /// **G2.3 (completeness)** — the research engine serves
    /// `Request::everything()` and its emitted slot set is the registry's.
    #[test]
    fn everything_covers_the_whole_registry() {
        let w = full_width();
        let (s, d) = pair(64, 64);
        let e = extract(
            &Request::everything(),
            &RgbSlice::new(&s, 64, 64),
            &RgbSlice::new(&d, 64, 64),
        )
        .expect("everything must be servable");
        assert_eq!(e.layout_width(), w, "full width");
        assert_eq!(e.values().len(), w);
        assert!((956..986).chain(1820..1825).all(|id| e.values()[id] == 0.0));
        assert_eq!(e.provenance().len(), w);
        // Every registered slot is populated by the everything plan.
        // `globaff` (f1867..1890) is computed only at formula revision 6.
        let globaff = family_slots(ComputeToken::GlobAff);
        let all = SlotSet::from_ranges([(0, 956), (986, 1820), (1825, w)]);
        let all = if crate::ssim_form::active_revision().computes_globaff() {
            all
        } else {
            globaff.missing_from(&all)
        };
        assert!(
            Request::for_slots(SlotSet::from_ranges([(956, 986), (1820, 1825)]), w)
                .validate()
                .is_err()
        );
        assert!(
            e.emitted().covers(&all),
            "everything must populate every registered slot; missing {}",
            e.emitted().missing_from(&all)
        );
        // Exactly the reserved positions are unregistered; other IDs keep their definitions.
        assert!(
            e.provenance()
                .iter()
                .enumerate()
                .all(|(id, p)| (p.family == "unregistered")
                    == ((956..986).contains(&id) || (1820..1825).contains(&id))),
            "only the reserved positions may be unregistered"
        );
    }

    /// The revision selector: `Current` and the current era by NAME agree
    /// bit-for-bit, an unavailable era is refused naming the slots, and an
    /// unregistered token is refused as a typo.
    #[test]
    fn revision_selection_is_checked_and_refuses_loudly() {
        let (w, h) = (64usize, 64usize);
        let (s, d) = pair(w, h);
        let (rs, rd) = (RgbSlice::new(&s, w, h), RgbSlice::new(&d, w, h));

        // The 372 layout's pooled slots carry the option-C landed revision.
        let want = SlotSet::from_ranges([(0, 372)]);
        let cur = extract(&Request::for_slots(want.clone(), 372), &rs, &rd)
            .expect("current")
            .into_values();
        let named = extract(
            &Request::for_slots(want.clone(), 372)
                .at_revision(RevisionRef::Named("v1postc".into())),
            &rs,
            &rd,
        )
        .expect("naming the current era must be servable")
        .into_values();
        for (i, (a, b)) in cur.iter().zip(named.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "f{i} moved under a named era");
        }

        // A registered-but-not-current era is refused, naming slots.
        let err = extract(
            &Request::for_slots(want.clone(), 372)
                .at_revision(RevisionRef::Named("v1ssimcap".into())),
            &rs,
            &rd,
        )
        .expect_err("a proposed-only era must be refused");
        match err {
            ResearchError::RevisionUnavailable {
                ref incompatible, ..
            } => {
                assert!(!incompatible.is_empty(), "must name the slots");
            }
            other => panic!("wrong error: {other}"),
        }

        // An unregistered token is a typo, not a request — and it is
        // decided BEFORE the slot comparison, so the spelling error is not
        // buried inside a list of slots that all disagree with a name that
        // does not exist.
        let err = extract(
            &Request::for_slots(SlotSet::from_ranges([(0, 156)]), 372)
                .at_revision(RevisionRef::Named("no-such-era".into())),
            &rs,
            &rd,
        )
        .expect_err("an unregistered era must be refused");
        assert!(
            matches!(err, ResearchError::RevisionUnregistered { .. }),
            "wrong error: {err}"
        );
    }

    /// The two revision selectors COMPOSE: pinning the arithmetic era makes
    /// the corresponding named request servable, with no change here.
    ///
    /// `ZENSIM_FORMULA_REV` is read once per process into a `OnceLock`, so
    /// this cannot flip it at runtime; it asserts the LOGIC against the
    /// build's actual active revision instead, which is the half this module
    /// owns. The arithmetic half has its own gate in the revision lane.
    #[test]
    fn the_request_selector_follows_the_arithmetic_selector() {
        let active = active_era_tokens();
        let shipped = crate::ssim_form::active_revision();
        // Whatever this build is pinned to, every era it computes must be a
        // registered token — an active era nothing declares would mean the
        // two selectors had drifted apart.
        for t in active {
            assert!(
                era_is_registered(t),
                "the arithmetic selector claims era {t}, which no registry \
                 entry names"
            );
        }
        // And under the SHIPPED revision the active set is empty, which is
        // what makes "no era other than the current one can be reproduced"
        // true rather than merely asserted.
        if shipped == crate::feature_defs::FormulaRevision::Rev1 {
            assert!(
                active.is_empty(),
                "Rev1 is the baseline and introduces no era token"
            );
        }
    }

    /// **The refusal's slot list IS the registry's declared blast radius.**
    ///
    /// Under the shipped revision, asking for an era `Rev2` introduces must
    /// refuse exactly the slots that ERA moves — the revision lane's own
    /// owner — and no others. A selector that named a different set would be
    /// refusing the wrong work, in either direction: too many (blocking
    /// servable slots) or too few (serving a caller the wrong era silently).
    ///
    /// ⚠ The comparison target changed on 2026-09-05 when F17's `v1hfgain`
    /// joined revision 2. A `RevisionRef::Named` request is ERA-scoped, so it
    /// is compared against [`feature_defs::era_moved_slots`]; it used to read
    /// `FormulaRevision::Rev2::moved_slots`, whose union is now 144 and which
    /// could never have been the right target for a single-era request — it
    /// was merely equal to one while revision 2 had one v1 era in it. The
    /// numbers asserted (132 / 36) are unchanged, and the F17 era is asserted
    /// beside them rather than folded into them.
    ///
    /// MEASURED end-to-end alongside this: with `ZENSIM_FORMULA_REV=2` the
    /// request succeeds and exactly the named columns move — 36 for a
    /// `basic`-only request (F4's three `ssim_*` signals × 12 cells) and
    /// **132** for the full `0..372`, which is F4's blast radius CORRECTED by
    /// the revision lane's own measurement from the audit's original 72.
    #[test]
    fn the_refusal_names_exactly_the_registrys_moved_slots() {
        if crate::ssim_form::active_revision() != crate::feature_defs::FormulaRevision::Rev1 {
            // This build is pinned to a non-shipped revision; the request
            // would SUCCEED and there is no refusal to compare. Reported, not
            // silently skipped — a graceful skip is what CLAUDE.md bans.
            panic!(
                "this gate requires the shipped revision; ZENSIM_FORMULA_REV \
                 is pinned to {:?}",
                crate::ssim_form::active_revision()
            );
        }
        let ns = crate::NUM_SCALES;
        let want = SlotSet::from_ranges([(0, 372)]);
        let req = Request::for_slots(want, 372).at_revision(RevisionRef::Named("v1ssimcap".into()));
        let err = req.validate().expect_err("Rev1 cannot serve a Rev2 era");
        let ResearchError::RevisionUnavailable { incompatible, .. } = err else {
            panic!("wrong error: {err}");
        };
        let declared = SlotSet::from_slots(
            crate::feature_defs::era_moved_slots("v1ssimcap", 372, ns)
                .into_iter()
                .map(usize::from),
        );
        assert_eq!(
            incompatible, declared,
            "the refusal's slots and the registry's declared blast radius \
             disagree"
        );
        // 132 = F4's measured blast radius at the 372 layout: 36 in `basic`
        // plus 96 across the masked and IW pool blocks. The audit's original
        // figure was 72; the revision lane corrected it by measurement, and
        // this gate reads the corrected owner rather than a copied number.
        assert_eq!(incompatible.len(), 132, "F4's blast radius at 372");
        // The `basic`-only sub-request is the CLI case, and is a strict
        // subset — named separately because it is the number the end-to-end
        // demonstration printed.
        let basic_only = Request::for_slots(SlotSet::from_ranges([(0, 156)]), 372)
            .at_revision(RevisionRef::Named("v1ssimcap".into()));
        let err = basic_only.validate().expect_err("still unservable");
        let ResearchError::RevisionUnavailable { incompatible, .. } = err else {
            panic!("wrong error: {err}");
        };
        assert_eq!(incompatible.len(), 36, "F4 inside the basic block");

        // The same property for revision 2's OTHER v1 era, F17's `v1hfgain`:
        // twelve `contrast_inc` slots, all of them inside `basic`, so the
        // basic-only sub-request refuses the same twelve rather than a subset.
        let f17 = Request::for_slots(SlotSet::from_ranges([(0, 372)]), 372)
            .at_revision(RevisionRef::Named("v1hfgain".into()));
        let ResearchError::RevisionUnavailable { incompatible, .. } =
            f17.validate().expect_err("Rev1 cannot serve v1hfgain")
        else {
            panic!("wrong error kind for v1hfgain");
        };
        assert_eq!(
            incompatible,
            SlotSet::from_slots(
                crate::feature_defs::era_moved_slots("v1hfgain", 372, ns)
                    .into_iter()
                    .map(usize::from),
            )
        );
        assert_eq!(incompatible.len(), 12, "F17's blast radius at 372");
        let f17_basic = Request::for_slots(SlotSet::from_ranges([(0, 156)]), 372)
            .at_revision(RevisionRef::Named("v1hfgain".into()));
        let ResearchError::RevisionUnavailable { incompatible, .. } =
            f17_basic.validate().expect_err("still unservable")
        else {
            panic!("wrong error kind for the v1hfgain basic request");
        };
        assert_eq!(
            incompatible.len(),
            12,
            "every F17 slot is inside the basic block"
        );

        // And for revision 2's THIRD v1 era, F18's `v1detroot` — the pooled
        // 4th/8th roots that made the extractor libc-dependent. 144 at the
        // 372 layout (basic 36 + peaks 36 + masked 36 + IW 36); the extra 12
        // this era moves are the v2 `ssim_dev4` cells, which do not exist at
        // 372. It OVERLAPS `v1ssimcap` on the 48 SSIM L4/L8 slots, which is
        // why each era is checked against its own `era_moved_slots` and never
        // against the revision's union.
        let f18 = Request::for_slots(SlotSet::from_ranges([(0, 372)]), 372)
            .at_revision(RevisionRef::Named("v1detroot".into()));
        let ResearchError::RevisionUnavailable { incompatible, .. } =
            f18.validate().expect_err("Rev1 cannot serve v1detroot")
        else {
            panic!("wrong error kind for v1detroot");
        };
        assert_eq!(
            incompatible,
            SlotSet::from_slots(
                crate::feature_defs::era_moved_slots("v1detroot", 372, ns)
                    .into_iter()
                    .map(usize::from),
            )
        );
        assert_eq!(incompatible.len(), 144, "F18's blast radius at 372");
    }

    /// A `FeatureSetId` round-trips into a request that reproduces it.
    #[test]
    fn a_registered_set_id_round_trips_through_the_request() {
        let (w, h) = (64usize, 64usize);
        let (s, d) = pair(w, h);
        let (rs, rd) = (RgbSlice::new(&s, w, h), RgbSlice::new(&d, w, h));
        // basic+peaks+moments at w944 — the campaign's free-set producer.
        let want = SlotSet::from_ranges([(0, 228)])
            .union(&family_slots(ComputeToken::Moments))
            .clipped_to(944);
        let id = FeatureSetId::from_slots_with_layout(
            crate::feature_set_id::ComputeParts::EMPTY
                .with(ComputeToken::Basic)
                .with(ComputeToken::Peaks)
                .with(ComputeToken::Moments),
            944,
            "era2r4",
            &want,
        )
        .expect("a registered set id");
        let req = Request::for_set(&id).expect("registered set must plan");
        assert_eq!(req.layout_width(), 944);
        let e = extract(&req, &rs, &rd).expect("extract");
        assert_eq!(e.emitted().len(), 265, "the free-set arm emits 265 slots");
        assert_eq!(e.layout_width(), 944);
        // The producer id it reports must name the same slot hash.
        let got = e.feature_set_id().expect("producer id");
        assert_eq!(got.slots_hash(), id.slots_hash());
    }

    /// Provenance is index-aligned, complete, and reports structural zeros as
    /// NOT populated rather than omitting them.
    #[test]
    fn provenance_is_aligned_and_marks_structural_zeros() {
        let (w, h) = (64usize, 64usize);
        let (s, d) = pair(w, h);
        let want = SlotSet::from_ranges([(0, 156)]);
        let e = extract(
            &Request::for_slots(want, 944),
            &RgbSlice::new(&s, w, h),
            &RgbSlice::new(&d, w, h),
        )
        .expect("extract");
        assert_eq!(e.provenance().len(), 944);
        assert_eq!(e.values().len(), 944);
        for (pos, p) in e.provenance().iter().enumerate() {
            assert_eq!(usize::from(p.id), pos, "identity layout at position {pos}");
            assert_eq!(
                p.populated,
                e.emitted().contains(pos),
                "populated flag disagrees with the plan at f{pos}"
            );
            // The structural fill is 0.0 EXCEPT on the twelve F15 slots —
            // see `nonzero_structural_fill_slots` and
            // `the_structural_fill_value_is_zero_except_on_the_f15_slots`,
            // which pins both halves of that measurement.
            if !p.populated && !nonzero_structural_fill_slots().contains(pos) {
                assert_eq!(
                    e.values()[pos],
                    0.0,
                    "an unpopulated position must be a structural zero (f{pos}, {})",
                    p.name
                );
            }
        }
        // The manifest renders and names the build commit state honestly.
        let m = e.manifest_json();
        assert!(m.contains("\"layout_width\": 944"));
        assert!(m.contains("\"zensim_build_commit\""));
        assert!(m.contains("\"populated\": false"));
    }

    /// **The structural fill is NOT a constant** — measured, and pinned so a
    /// walk change cannot silently move it.
    ///
    /// A `v1_only` 944 request leaves 572 positions uncomputed. 560 of them
    /// come back as exactly `0.0`; twelve come back as exactly `1.0`, and
    /// they are precisely the defect-F15 `pjnd_fragility` slots. Both halves
    /// are asserted: a new nonzero position, or an F15 slot that starts
    /// reading zero, fails here.
    #[test]
    fn the_structural_fill_value_is_zero_except_on_the_f15_slots() {
        let (w, h) = (64usize, 64usize);
        let (s, d) = pair(w, h);
        let e = extract(
            &Request::for_slots(SlotSet::from_ranges([(0, 156)]), 944),
            &RgbSlice::new(&s, w, h),
            &RgbSlice::new(&d, w, h),
        )
        .expect("extract");
        let exceptions = nonzero_structural_fill_slots();
        assert_eq!(exceptions.len(), 12, "F15 covers twelve slots");
        let mut nonzero = Vec::new();
        for pos in 0..e.layout_width() {
            if !e.emitted().contains(pos) && e.values()[pos] != 0.0 {
                nonzero.push(pos);
            }
        }
        assert_eq!(
            SlotSet::from_slots(nonzero.iter().copied()),
            exceptions,
            "the set of nonzero unpopulated positions moved"
        );
        for pos in exceptions.iter_slots() {
            assert_eq!(
                e.values()[pos],
                1.0,
                "f{pos} ({}) — the fragility finaliser's degenerate value",
                e.provenance()[pos].name
            );
            assert_eq!(e.provenance()[pos].defect, Some("F15"));
        }
    }

    /// **G4.1 — a dense layout and its `w944` equivalent carry the SAME
    /// VALUES**, position-for-position through the layout's own index.
    ///
    /// This is what makes "retire 944-with-structural-zeros as the wire
    /// format" a rename rather than a re-extraction: a `dense265` table is
    /// the `w944` table with its 679 structural fills removed, and the
    /// layout says where each surviving id went.
    #[test]
    fn a_dense_layout_carries_the_same_values_as_its_sparse_equivalent() {
        let (w, h) = (128usize, 128usize);
        let (s, d) = pair(w, h);
        let (rs, rd) = (RgbSlice::new(&s, w, h), RgbSlice::new(&d, w, h));
        let want = SlotSet::from_ranges([(0, 228)])
            .union(&family_slots(ComputeToken::Moments))
            .clipped_to(944);

        let sparse =
            extract(&Request::for_slots(want.clone(), 944), &rs, &rd).expect("the w944 arm");
        let dense = extract(&Request::for_slots(want.clone(), 944).dense(), &rs, &rd)
            .expect("the dense arm");

        assert_eq!(sparse.layout_width(), 944);
        assert_eq!(sparse.layout_name(), "w944");
        assert_eq!(dense.layout_width(), 265, "the dense arm has no gaps");
        assert_eq!(dense.layout_name(), "dense265");
        // The two arms populate the SAME ids.
        assert_eq!(sparse.emitted(), dense.emitted());
        assert_eq!(dense.emitted().len(), 265);

        // Every carried id holds the bit-identical value in both arms, found
        // through each layout's own index rather than by assuming positions.
        for id in want.iter_slots() {
            let sp = sparse.position_of(id as u16).expect("sparse position");
            let dp = dense.position_of(id as u16).expect("dense position");
            assert_eq!(sp, id, "the w944 layout is the identity");
            assert_eq!(
                sparse.values()[sp].to_bits(),
                dense.values()[dp].to_bits(),
                "f{id} differs between the w944 and dense265 layouts \
                 (positions {sp} vs {dp})"
            );
        }
        // And the dense arm carries NOTHING else: 265 values, 265 ids.
        assert_eq!(dense.values().len(), 265);
        assert_eq!(dense.provenance().len(), 265);
        assert!(
            dense.provenance().iter().all(|p| p.populated),
            "a dense layout over a plan's own emit set has no structural fill"
        );
        // The 679 positions the sparse arm carries and the dense one drops
        // are exactly the ones the plan never populated.
        let dropped = 944 - 265;
        assert_eq!(
            sparse.provenance().iter().filter(|p| !p.populated).count(),
            dropped
        );
    }

    /// The producer id a DENSE extraction reports names the same slot HASH as
    /// its sparse twin, differing only in the declared width — which is the
    /// identity layer working as designed, and is what makes
    /// `feature_layout::slots_of` able to read the dense interpretation back.
    #[test]
    fn the_dense_producer_id_carries_the_same_slot_hash_as_the_sparse_one() {
        let (w, h) = (64usize, 64usize);
        let (s, d) = pair(w, h);
        let (rs, rd) = (RgbSlice::new(&s, w, h), RgbSlice::new(&d, w, h));
        let want = SlotSet::from_ranges([(0, 228)])
            .union(&family_slots(ComputeToken::Moments))
            .clipped_to(944);
        let sparse = extract(
            &Request::for_slots(want.clone(), 944).with_era_label("era2r4"),
            &rs,
            &rd,
        )
        .expect("sparse");
        let dense = extract(
            &Request::for_slots(want.clone(), 944)
                .dense()
                .with_era_label("era2r4"),
            &rs,
            &rd,
        )
        .expect("dense");
        let (a, b) = (
            sparse.feature_set_id().expect("sparse id"),
            dense.feature_set_id().expect("dense id"),
        );
        assert_eq!(a.slots_hash(), b.slots_hash(), "same slots, same hash");
        assert_eq!(a.slots_hash(), want.hash8());
        // The layout is a HINT, not the identity — so the two ids are EQUAL
        // even though they were emitted at different widths, and the recorded
        // hint still says which width each came out at.
        assert_eq!(a, b, "same compute, same era, same slots => the SAME id");
        assert_eq!(a.layout_width(), Some(944));
        assert_eq!(b.layout_width(), Some(265));
        assert_eq!(a.compute(), b.compute());
        // And the dense id reads BACK to the same slot set — the property
        // `declared_layout` depends on.
        assert_eq!(
            crate::feature_layout::slots_of(b).as_ref(),
            Some(&want),
            "the dense id must reconstruct its own slot set"
        );
    }

    /// A dense layout still plans the CHEAP walk — packing the output does
    /// not change what runs.
    #[test]
    fn a_dense_layout_does_not_change_the_compute_set() {
        let want = SlotSet::from_ranges([(0, 228)])
            .union(&family_slots(ComputeToken::Moments))
            .clipped_to(944);
        let a = Request::for_slots(want.clone(), 944)
            .validate()
            .expect("sparse");
        let b = Request::for_slots(want, 944)
            .dense()
            .validate()
            .expect("dense");
        assert_eq!(a, b, "the same ids are emitted either way");
    }

    /// **REV5 addendum (coordinator, 2026-10-04): a Rev5 `basic,peaks,v2`
    /// request at the registry's full width plans ONLY the three supported
    /// families** — even though the layout's width reaches the append block
    /// and the Rev4 feature banks (the layout flags would turn them on in
    /// the toggles the plan emits, which is exactly the over-computation
    /// `ComputeSet::rev5_scope` exists to kill). `emitted()` is exactly the
    /// 576 requested slots, the producer id names `basic+peaks+v2` only,
    /// every unrequested position stays a structural zero in the
    /// provenance, and a request reaching past the three families is
    /// refused at plan time — never computed at a lower revision behind
    /// the caller's back.
    #[test]
    #[cfg(feature = "feature-regime-v2")]
    fn rev5_basic_peaks_v2_request_computes_only_the_three_families() {
        if !crate::ssim_form::run_at_revision(
            "5",
            "research::tests::rev5_basic_peaks_v2_request_computes_only_the_three_families",
            "REV5-EXTRACT-RAN",
        ) {
            return;
        }
        let (w, h) = (64usize, 64usize);
        let (s, d) = pair(w, h);
        let (rs, rd) = (RgbSlice::new(&s, w, h), RgbSlice::new(&d, w, h));
        let want = family_slots(ComputeToken::Basic)
            .union(&family_slots(ComputeToken::Peaks))
            .union(&family_slots(ComputeToken::V2));
        assert_eq!(want.len(), 576, "the spec's three-family slot count");
        let e = extract(&Request::for_slots(want.clone(), full_width()), &rs, &rd)
            .expect("the Rev5 supported set must extract");
        // Exactly the requested slots — nothing the wider layout reaches.
        assert_eq!(e.emitted(), &want);
        assert_eq!(e.emitted().len(), 576);
        // The producer id names only the three families — no append,
        // layout width.
        let id = e.feature_set_id().expect("a full-family request has an id");
        let only = crate::feature_set_id::ComputeParts::EMPTY
            .with(ComputeToken::Basic)
            .with(ComputeToken::Peaks)
            .with(ComputeToken::V2);
        assert_eq!(
            id.compute(),
            only,
            "the id must name ONLY the supported families: {id}"
        );
        // Every unrequested position stays a structural zero — the
        // assembler turns exactly these into NaN.
        for p in e.provenance() {
            assert_eq!(
                p.populated,
                want.contains(usize::from(p.id)),
                "position {} ({}) populated flag disagrees with the request",
                p.id,
                p.name
            );
        }
        // Requests reaching past basic+peaks+v2 refuse — one slot from each
        // unsupported family Rev5 does not compute.
        for extra in [228usize, 300, 720, 924, 944, 1100] {
            let ask = want.clone().union(&SlotSet::from_slots([extra]));
            let err = extract(&Request::for_slots(ask, full_width()), &rs, &rd)
                .err()
                .unwrap_or_else(|| {
                    panic!("slot {extra} outside basic+peaks+v2 must refuse at Rev5")
                });
            assert!(
                matches!(&err, ResearchError::Plan(m) if m.contains("basic + peaks + v2")),
                "slot {extra}: expected the Rev5 scope refusal, got {err:?}"
            );
        }
        println!("REV5-EXTRACT-RAN");
    }

    /// Every provenance name is unique across the full width — the property
    /// that makes a manifest's column names a usable key.
    #[test]
    fn provenance_names_are_unique_at_the_full_width() {
        let (w, h) = (64usize, 64usize);
        let (s, d) = pair(w, h);
        let e = extract(
            &Request::everything(),
            &RgbSlice::new(&s, w, h),
            &RgbSlice::new(&d, w, h),
        )
        .expect("extract");
        let mut seen = std::collections::HashSet::new();
        for p in e.provenance() {
            assert!(seen.insert(p.name.clone()), "duplicate name {}", p.name);
        }
        assert_eq!(seen.len(), full_width());
    }
}
