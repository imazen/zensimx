// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.

//! **Layer 2 — the LAYOUT**, a declared mapping from feature ids to positions
//! in an emitted vector.
//!
//! Design: `docs/FEATURE_SYSTEM_DESIGN_2026-09-05.md` §4. Phase 4 of
//! `docs/PLAN_FEATURE_SYSTEM_2026-09-05.md`.
//!
//! ## What this makes expressible that was not
//!
//! Phase 1 landed `layout_width` + `LayoutBlocks::for_width` and recorded WHY
//! (`PLAN_FEATURE_SYSTEM` §1b): every layout registered at the time was the
//! IDENTITY mapping over its range, so a full `id -> position` map would have
//! been a type with no consumer and no test that could distinguish it from the
//! width it wrapped. That is no longer true — a **dense** layout has a
//! different mapping, and phase 4 is where it gets its first real consumer.
//!
//! Three facts follow, none of which held before:
//!
//! 1. **The legacy widths are DECLARATIONS, not code paths.** `w156`, `w372`,
//!    `w504`, `w720`, `w924`, `w944`, `w956` are `Layout`s whose `slot_at[i]
//!    == Some(i)`. Every stored table, every shipped bake and every board row
//!    is already in one of them, so **no stored byte moves**.
//! 2. **A structural zero is a LAYOUT property, not a data property.** A
//!    position whose id the plan does not compute is written `0.0` *because
//!    the layout declares that id lives there and the plan does not populate
//!    it* — the distinction `FEATURE_SET_IDS.md` §3.1 says is missing when
//!    seven different feature sets are all called "944".
//! 3. **A dense subset is expressible.** [`Layout::dense`] packs a read set
//!    with no gaps, which is what retires "944-with-structural-zeros as the
//!    wire format" for NEW artifacts without touching the 944 that exists.
//!
//! ## Why the gather is a real safety fix, not a tidiness one
//!
//! Before this, `metric::prep_bake_input_f32` resolved a width disagreement
//! POSITIONALLY: `n_inputs < features.len()` took the first `n_inputs`
//! positions. For every bake that exists today that is correct, because every
//! declared width is a PREFIX of the walk's identity layout. The moment a
//! dense table exists it stops being correct — a 265-input bake over
//! `basic+peaks+moments` would silently be served `f0..265`, which is
//! `basic + peaks + 37 slots of masked` — plausible numbers, wrong features,
//! no error. A declared layout turns that into a gather, and an *undeclarable*
//! width into a loud refusal.

use crate::feature_set_id::{FeatureSetId, SlotSet};

/// Resolve a bake's arithmetic era. Absence is the registered pre-stamp
/// legacy era; a present unknown value is never a request for that default.
pub(crate) fn formula_revision(
    model: &crate::mlp::Model,
) -> Result<crate::feature_defs::FormulaRevision, crate::ZensimError> {
    use crate::feature_defs::FormulaRevision;
    if model.metadata().get("zentrain.formula_revision").is_none() {
        return Ok(crate::ssim_form::SHIPPED_REVISION);
    }
    match model
        .metadata()
        .get_utf8("zentrain.formula_revision")
        .ok()
        .map(str::trim)
    {
        Some("1" | "rev1" | "Rev1") => Ok(FormulaRevision::Rev1),
        Some("2" | "rev2" | "Rev2") => Ok(FormulaRevision::Rev2),
        Some("3" | "rev3" | "Rev3") => Ok(FormulaRevision::Rev3),
        Some("4" | "rev4" | "Rev4") => Ok(FormulaRevision::Rev4),
        Some("5" | "rev5" | "Rev5") => Ok(FormulaRevision::Rev5),
        Some("6" | "rev6" | "Rev6") => Ok(FormulaRevision::Rev6),
        Some("7" | "rev7" | "Rev7") => Ok(FormulaRevision::Rev7),
        _ => Err(crate::ZensimError::ModelLoadFailed {
            reason: "unknown zentrain.formula_revision",
        }),
    }
}

/// A declared mapping from feature ids to positions in an emitted vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Layout {
    /// Position -> feature id. `None` is a DECLARED GAP: a position the
    /// layout reserves that carries no registered feature (only reachable
    /// past the registry's full width).
    slot_at: Vec<Option<u16>>,
    /// `(id, position)` sorted by id — the reverse index. A `Vec` + binary
    /// search rather than a `HashMap`: a layout is built once per plan, never
    /// per pixel, and the ids are already sorted for every layout anyone can
    /// construct here.
    by_id: Vec<(u16, u32)>,
    /// A stable name for manifests and error messages.
    name: String,
}

impl Layout {
    /// The identity layout of `width` — `slot_at[i] == Some(i)`.
    ///
    /// Every layout registered before phase 4 is one of these, which is why
    /// declaring them changes no stored byte.
    pub(crate) fn identity(width: usize) -> Layout {
        let slot_at: Vec<Option<u16>> = (0..width).map(|i| u16::try_from(i).ok()).collect();
        let by_id = slot_at
            .iter()
            .enumerate()
            .filter_map(|(pos, id)| id.map(|id| (id, pos as u32)))
            .collect();
        Layout {
            slot_at,
            by_id,
            name: format!("w{width}"),
        }
    }

    /// A DENSE layout over `slots`: the requested ids packed in ascending
    /// order with no gaps, so the emitted width is `slots.len()`.
    pub(crate) fn dense(slots: &SlotSet) -> Layout {
        let slot_at: Vec<Option<u16>> = slots.iter_slots().map(|s| u16::try_from(s).ok()).collect();
        let by_id = slot_at
            .iter()
            .enumerate()
            .filter_map(|(pos, id)| id.map(|id| (id, pos as u32)))
            .collect();
        Layout {
            name: format!("dense{}", slot_at.len()),
            slot_at,
            by_id,
        }
    }

    /// The emitted width.
    pub(crate) fn width(&self) -> usize {
        self.slot_at.len()
    }

    /// A stable name for manifests and messages (`w944`, `dense265`).
    // Consumers (`feature_plan`, `research`) are `feature-regime-v2`-gated; the
    // LAYOUT itself is not, because the shipped dense bakes it serves are not
    // (2026-09-06 ungate). Scoped so a genuinely-dead accessor still shows up
    // on a default build.
    #[cfg_attr(not(feature = "feature-regime-v2"), allow(dead_code))]
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    /// The feature id at `pos`, or `None` for a declared gap.
    pub(crate) fn slot_at(&self, pos: usize) -> Option<u16> {
        self.slot_at.get(pos).copied().flatten()
    }

    /// The position of feature `id`, or `None` when this layout does not
    /// carry it.
    // Consumers (`feature_plan`, `research`) are `feature-regime-v2`-gated; the
    // LAYOUT itself is not, because the shipped dense bakes it serves are not
    // (2026-09-06 ungate). Scoped so a genuinely-dead accessor still shows up
    // on a default build.
    #[cfg_attr(not(feature = "feature-regime-v2"), allow(dead_code))]
    pub(crate) fn pos_of(&self, id: u16) -> Option<usize> {
        self.by_id
            .binary_search_by_key(&id, |&(i, _)| i)
            .ok()
            .map(|k| self.by_id[k].1 as usize)
    }

    /// Every id this layout carries.
    // Consumers (`feature_plan`, `research`) are `feature-regime-v2`-gated; the
    // LAYOUT itself is not, because the shipped dense bakes it serves are not
    // (2026-09-06 ungate). Scoped so a genuinely-dead accessor still shows up
    // on a default build.
    #[cfg_attr(not(feature = "feature-regime-v2"), allow(dead_code))]
    pub(crate) fn ids(&self) -> SlotSet {
        SlotSet::from_slots(self.slot_at.iter().flatten().map(|&id| usize::from(id)))
    }

    /// Is this the identity mapping over `0..width`?
    ///
    /// The fast path the runtime takes for every artifact that exists today.
    pub(crate) fn is_identity(&self) -> bool {
        self.slot_at
            .iter()
            .enumerate()
            .all(|(pos, id)| *id == u16::try_from(pos).ok())
    }

    /// The identity width the WALK must emit for this layout to be fillable:
    /// one past the highest id it carries.
    ///
    /// Distinct from [`Self::width`] the moment a layout is dense — a
    /// `dense265` layout over `basic+peaks+moments` still needs a walk that
    /// reaches slot 943.
    pub(crate) fn walk_width(&self) -> usize {
        self.slot_at
            .iter()
            .flatten()
            .map(|&id| usize::from(id) + 1)
            .max()
            .unwrap_or(0)
    }

    /// Gather a walk's identity-laid-out vector into this layout.
    ///
    /// A position whose id the walk did not reach, and a declared gap, both
    /// become the structural `0.0` — which is what the layout DECLARES, not a
    /// silently-dropped value: `Plan::emit` is what says whether a position
    /// carries a computed number.
    pub(crate) fn gather(&self, walk: &[f64], out: &mut Vec<f64>) {
        out.clear();
        out.reserve(self.slot_at.len());
        for id in &self.slot_at {
            out.push(match id {
                Some(id) => walk.get(usize::from(*id)).copied().unwrap_or(0.0),
                None => 0.0,
            });
        }
    }
}

/// The slot set a registered [`FeatureSetId`] names, or `None` when this build
/// cannot reproduce it.
///
/// The id carries COMPUTE tokens, a WIDTH, an era and a slot hash; the slots
/// follow from the registry. **The width means two different things depending
/// on the layout, and the HASH is what decides which** — which is exactly the
/// job the identity layer exists to do:
///
/// * for a SPARSE (identity) layout the width is also the clip: a
///   `basic+peaks+moments@w944` id names the 265 slots of those families that
///   fall below 944, laid out at their own indices with 679 structural fills;
/// * for a DENSE layout the width is the PACKED count: the same 265 slots,
///   at consecutive positions, so the id reads `…@w265` — and clipping the
///   family union to 265 would reconstruct a completely different set.
///
/// So both readings are tried and the recorded `slots_hash8` picks the one
/// the id actually names. An id neither reading reproduces describes a set
/// this build does not have, and saying so is the point.
// Consumers (`feature_plan::Plan`, `research`) are `feature-regime-v2`-gated;
// see `Layout::name`'s note.
#[cfg_attr(not(feature = "feature-regime-v2"), allow(dead_code))]
pub(crate) fn slots_of(id: &FeatureSetId) -> Option<SlotSet> {
    let ns = crate::NUM_SCALES;
    let mut union = SlotSet::from_slots([]);
    for t in id.compute().iter() {
        union = union.union(&crate::feature_defs::family_slots(t, ns));
    }
    if id.compute().has_palette() {
        union = union.union(&crate::feature_defs::family_slots(
            crate::feature_defs::Family::Palette,
            ns,
        ));
    }
    let dense = union.clipped_to(crate::feature_defs::full_width(ns));
    // The SPARSE reading needs a clip width. A legacy `@w<N>` id names it; the
    // canonical layout-free form does not, so try every width any registered
    // producer set has used. The hash decides — this list only bounds the
    // search, and an id whose clip width is not on it is reported
    // unreproducible rather than guessed at.
    let candidates: &[usize] = match id.layout_width() {
        Some(w) => &[w],
        None => crate::feature_defs::REGISTERED_LAYOUT_WIDTHS,
    };
    for &w in candidates {
        let sparse = union.clipped_to(w);
        if sparse.hash8() == id.slots_hash() {
            return Some(sparse);
        }
    }
    // The DENSE reading. The recorded width is NOT re-checked here: it is not
    // part of the identity, the hash already pins the exact slot list, and the
    // one caller that needs "this set has as many members as the bake asks
    // for" ([`declared_layout`]) applies that filter itself against the BAKE's
    // width, which is the number that actually matters.
    (dense.hash8() == id.slots_hash()).then_some(dense)
}

/// The layout a loaded bake DECLARES.
///
/// Conservative by construction, and that is the safety property: a dense
/// layout is used **only** when the bake carries a `zentrain.feature_set_id`
/// that this build can reproduce AND whose slot count equals the bake's
/// caller-facing width exactly. Anything else — no id, an unparseable id, an
/// id whose hash disagrees, or an id whose slot count differs from the width
/// (which is the NORMAL case: the metadata records the id of the tables the
/// bake was TRAINED on, and a `944`-with-structural-zeros producer names 265
/// slots at width 944) — resolves to the IDENTITY layout at
/// `caller_input_width()`, which is exactly what every bake in existence gets
/// today.
///
/// So this function returns a non-identity layout for **zero** currently
/// shipped or board bakes, by design; it is the hook a dense-table bake needs
/// in order to be servable at all, and
/// [`tests::every_shipped_bake_resolves_to_an_identity_layout`] pins that.
pub(crate) fn declared_layout(model: &crate::mlp::Model) -> Layout {
    let width = model.caller_input_width();
    // THE declaration, preferred over every inference: an explicit ascending
    // id list. A family-token id can only name a FAMILY UNION, so it cannot
    // express a read set like Profile D's 28 basic slots; `feature_ids` can,
    // and it is what the cruft purge stamps onto every densified bake.
    if let Some(slots) = declared_ids(model)
        && slots.len() == width
    {
        return Layout::dense(&slots);
    }
    // E33 `fx1`: a derived-input bake's declared ids are its READ SET and its
    // gather layout; the model's caller width counts derived entries instead.
    // Validity (every id read, entry count == caller width) is checked where
    // the bake is loaded (`bake_metadata::parse_bake_metadata`), which refuses
    // a malformed declaration before any forward.
    if model
        .metadata()
        .get(crate::derived_inputs::DERIVED_INPUTS_KEY)
        .is_some()
        && let Some(slots) = declared_ids(model)
    {
        return Layout::dense(&slots);
    }
    let dense = model
        .metadata()
        .get_utf8("zentrain.feature_set_id")
        .ok()
        .and_then(FeatureSetId::parse)
        .as_ref()
        .and_then(dense_slots_of)
        .filter(|slots| slots.len() == width);
    match dense {
        Some(slots) => Layout::dense(&slots),
        None => Layout::identity(width),
    }
}

/// The bake's EXPLICIT declared read set, from `zentrain.feature_ids`.
///
/// Format: ascending decimal feature ids, one per line (whitespace-separated
/// is accepted so the payload can be written either way). **Strict**: a
/// duplicate, a descending pair, an unparseable token or an id past the
/// registry's full width returns `None` and the caller falls back to the
/// identity layout — a declaration this build cannot reproduce must never be
/// half-believed, because a wrong id list permutes the vector the bake is
/// served.
///
/// This is a METADATA convention, not a format change: ZNPR v3 metadata is
/// free-form utf8 key/value, and `zenpredict` is untouched.
pub(crate) fn declared_ids(model: &crate::mlp::Model) -> Option<SlotSet> {
    let text = model.metadata().get_utf8(FEATURE_IDS_KEY).ok()?;
    parse_feature_ids(text, crate::feature_defs::full_width(crate::NUM_SCALES))
}

/// The metadata key carrying a bake's explicit declared read set.
pub const FEATURE_IDS_KEY: &str = "zentrain.feature_ids";

/// Parse + VALIDATE an explicit id list. Separated from [`declared_ids`] so
/// the strictness is testable without building a `Model`.
pub(crate) fn parse_feature_ids(text: &str, full_width: usize) -> Option<SlotSet> {
    let mut ids: Vec<usize> = Vec::new();
    for tok in text.split_ascii_whitespace() {
        let v: usize = tok.parse().ok()?;
        if v >= full_width {
            return None;
        }
        if let Some(&last) = ids.last() {
            // Strictly ascending: rejects both duplicates and disorder, which
            // is what makes position `j` provably the `j`-th smallest id and
            // so makes `Layout::dense`'s packing the same mapping the writer
            // intended.
            if v <= last {
                return None;
            }
        }
        ids.push(v);
    }
    (!ids.is_empty()).then(|| SlotSet::from_slots(ids))
}

/// The DENSE reading of an id, and only the dense one.
///
/// [`slots_of`] tries both readings because a caller reproducing a registered
/// set wants whichever one the id names. [`declared_layout`] must not: a
/// SPARSE id whose reconstruction happens to have as many members as the
/// bake's width would otherwise be read as a dense layout and permute the
/// vector the bake is served. So the layout decision takes the strict form —
/// clip to the registry's full width, require the hash, require the packed
/// count — and anything else stays the identity layout.
fn dense_slots_of(id: &FeatureSetId) -> Option<SlotSet> {
    let ns = crate::NUM_SCALES;
    let mut union = SlotSet::from_slots([]);
    for t in id.compute().iter() {
        union = union.union(&crate::feature_defs::family_slots(t, ns));
    }
    if id.compute().has_palette() {
        union = union.union(&crate::feature_defs::family_slots(
            crate::feature_defs::Family::Palette,
            ns,
        ));
    }
    let dense = union.clipped_to(crate::feature_defs::full_width(ns));
    // A set that IS the identity range is not a dense layout, it is the
    // identity one under another name — `Layout::dense(0..372)` and
    // `Layout::identity(372)` are the same mapping, and returning the former
    // would only make `is_identity` say something different about a vector
    // nothing did to.
    let is_identity_range = dense.iter_slots().last() == Some(dense.len().saturating_sub(1));
    // The recorded width is not consulted: it is not part of the identity, and
    // `declared_layout` — the only caller — filters on the BAKE's caller width,
    // which is the number a mis-sized layout would actually break.
    (dense.hash8() == id.slots_hash() && !is_identity_range).then_some(dense)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::feature_set_id::ComputeToken;

    /// **G4.2, the legacy half** — every registered legacy width is the
    /// IDENTITY mapping over its range, so declaring it moves no stored byte.
    ///
    /// Generalizes phase 1's G1.7 from a width to a `Layout`.
    #[test]
    fn every_legacy_width_is_the_identity_mapping() {
        for w in [156usize, 372, 504, 720, 924, 944, 956] {
            let l = Layout::identity(w);
            assert_eq!(l.width(), w);
            assert_eq!(l.walk_width(), w, "identity: walk width IS the width");
            assert!(l.is_identity(), "w{w} must be identity");
            for pos in 0..w {
                assert_eq!(l.slot_at(pos), u16::try_from(pos).ok());
                assert_eq!(l.pos_of(pos as u16), Some(pos));
            }
        }
    }

    /// A dense layout packs its ids with no gaps, and round-trips both ways.
    #[test]
    fn a_dense_layout_packs_and_round_trips() {
        let ns = crate::NUM_SCALES;
        let want = SlotSet::from_ranges([(0, 228)])
            .union(&crate::feature_defs::family_slots(
                ComputeToken::Moments,
                ns,
            ))
            .clipped_to(944);
        let l = Layout::dense(&want);
        assert_eq!(l.width(), 265, "the free-set arm packs to 265");
        // 942, not 944: the highest raw-moment slot is f941, so the WALK
        // must reach 942 — a fact worth having a number for, because
        // `LayoutBlocks::for_width(942)` still reaches append2 (base 924) and
        // still does NOT reach CSFW (base 944), which is exactly right for
        // this set.
        assert_eq!(
            l.walk_width(),
            942,
            "the walk must reach one past the highest id the layout carries"
        );
        assert!(!l.is_identity());
        assert_eq!(l.ids(), want);
        // Every id round-trips through its position, in both directions.
        for (pos, id) in want.iter_slots().enumerate() {
            assert_eq!(l.slot_at(pos), u16::try_from(id).ok());
            assert_eq!(l.pos_of(id as u16), Some(pos));
        }
        // And an id the layout does not carry has no position.
        assert_eq!(l.pos_of(300), None, "f300 is IW, not in this set");
    }

    /// The gather moves values to their declared positions and fills the rest
    /// with the structural zero — never a stale or shifted value.
    #[test]
    fn the_gather_places_ids_at_their_declared_positions() {
        let want = SlotSet::from_ranges([(0, 4), (10, 12)]);
        let l = Layout::dense(&want);
        assert_eq!(l.width(), 6);
        let walk: Vec<f64> = (0..12).map(|i| i as f64 * 100.0).collect();
        let mut out = Vec::new();
        l.gather(&walk, &mut out);
        assert_eq!(out, vec![0.0, 100.0, 200.0, 300.0, 1000.0, 1100.0]);
        // An id past the walk's reach becomes the structural zero, not a
        // panic and not a wrapped read.
        let short = vec![1.0, 2.0];
        l.gather(&short, &mut out);
        assert_eq!(out, vec![1.0, 2.0, 0.0, 0.0, 0.0, 0.0]);
        // The identity layout's gather is a plain copy.
        let id = Layout::identity(12);
        id.gather(&walk, &mut out);
        assert_eq!(out, walk);
    }

    /// A bake DECLARING a dense feature set resolves to that dense layout —
    /// the positive control for
    /// [`tests::every_shipped_bake_resolves_to_an_identity_layout`], without
    /// which that test would pass just as happily if `declared_layout` always
    /// returned identity.
    #[test]
    fn a_bake_declaring_a_dense_set_resolves_to_that_layout() {
        let ns = crate::NUM_SCALES;
        let slots = SlotSet::from_ranges([(0, 228)])
            .union(&crate::feature_defs::family_slots(
                ComputeToken::Moments,
                ns,
            ))
            .clipped_to(944);
        assert_eq!(slots.len(), 265);
        let id = crate::feature_set_id::FeatureSetId::from_slots_with_layout(
            crate::feature_set_id::ComputeParts::EMPTY
                .with(ComputeToken::Basic)
                .with(ComputeToken::Peaks)
                .with(ComputeToken::Moments),
            265,
            "era2r4",
            &slots,
        )
        .expect("a dense id");
        // The id must round-trip through its own grammar, and `slots_of` must
        // read the DENSE interpretation out of it.
        let reparsed = crate::feature_set_id::FeatureSetId::parse(&id.to_string())
            .expect("the dense id must reparse");
        assert_eq!(
            slots_of(&reparsed).as_ref(),
            Some(&slots),
            "slots_of must read the dense interpretation"
        );

        let bytes = dense_bake_bytes(&id.to_string(), 265);
        let m = crate::mlp::Model::from_bytes(&bytes).expect("bake parses");
        assert_eq!(m.caller_input_width(), 265);
        assert_eq!(
            m.metadata().get_utf8("zentrain.feature_set_id").ok(),
            Some(id.to_string().as_str()),
            "the metadata entry must survive the bake"
        );
        let l = declared_layout(&m);
        assert!(!l.is_identity(), "must resolve to a DENSE layout");
        assert_eq!(l.width(), 265);
        assert_eq!(l.walk_width(), 942);
        assert_eq!(l.ids(), slots);
    }

    /// A 1-layer identity ZNPR v3 bake of `n` inputs carrying a declared
    /// feature-set id. Built through the mandated JSON pipeline.
    fn dense_bake_bytes(set_id: &str, n: usize) -> Vec<u8> {
        let arr = |v: &[f32]| -> String {
            let mut s = String::from("[");
            for (k, x) in v.iter().enumerate() {
                if k > 0 {
                    s.push(',');
                }
                s.push_str(&x.to_string());
            }
            s.push(']');
            s
        };
        let json = format!(
            r#"{{
                "schema_hash": 1,
                "scaler_mean": {mean},
                "scaler_scale": {scale},
                "metadata": [{{"key": "zentrain.feature_set_id", "type": "utf8", "text": "{set_id}"}}],
                "layers": [
                    {{"in_dim": {n}, "out_dim": 1, "activation": "identity",
                      "dtype": "f32", "weights": {w}, "biases": [0.0]}}
                ]
            }}"#,
            mean = arr(&vec![0.0f32; n]),
            scale = arr(&vec![1.0f32; n]),
            w = arr(&vec![1.0f32; n]),
        );
        zenpredict_bake::bake_from_json_str(&json).expect("synthetic dense bake must build")
    }

    /// **B.4 (the declaration is READ)** — an explicit `zentrain.feature_ids`
    /// list resolves to exactly that dense layout, and REMOVING it collapses
    /// the same bake to the identity layout. That pair is the proof the
    /// runtime gathers by the declaration rather than by a positional prefix:
    /// the first assertion alone would pass just as happily if
    /// `declared_layout` ignored the key and the width happened to match.
    #[test]
    fn an_explicit_feature_id_list_resolves_to_that_dense_layout() {
        // Profile D's shape: a scattered read set inside `basic`, which no
        // FAMILY-token id can name — the reason the explicit list exists.
        let ids: Vec<usize> = vec![0, 3, 7, 12, 40, 41, 99, 155];
        let bytes = ids_bake_bytes(&ids);
        let m = crate::mlp::Model::from_bytes(&bytes).expect("bake parses");
        assert_eq!(m.caller_input_width(), ids.len());
        let l = declared_layout(&m);
        assert!(!l.is_identity(), "must resolve to a DENSE layout");
        assert_eq!(l.width(), ids.len());
        assert_eq!(l.walk_width(), 156, "one past the highest declared id");
        assert_eq!(l.ids(), SlotSet::from_slots(ids.iter().copied()));
        for (pos, &id) in ids.iter().enumerate() {
            assert_eq!(l.slot_at(pos), u16::try_from(id).ok());
            assert_eq!(l.pos_of(id as u16), Some(pos));
        }

        // NEGATIVE CONTROL: the same widths, the same weights, no declaration.
        let plain =
            crate::mlp::Model::from_bytes(&ids_bake_bytes_no_decl(ids.len())).expect("bake parses");
        let pl = declared_layout(&plain);
        assert!(
            pl.is_identity(),
            "without the declaration the SAME width must fall back to identity"
        );
        assert_ne!(pl.ids(), l.ids(), "the two layouts must differ");
    }

    /// The parse is STRICT, and each rejection is its own reason. A
    /// half-believed id list permutes the vector a bake is served, so every
    /// one of these must fall back to identity rather than guess.
    #[test]
    fn the_feature_id_list_parse_refuses_anything_it_cannot_prove() {
        let full = crate::feature_defs::full_width(crate::NUM_SCALES);
        assert_eq!(
            parse_feature_ids("0 1 2 155", full),
            Some(SlotSet::from_slots([0usize, 1, 2, 155]))
        );
        // Newline-separated is the written form; whitespace-separated parses
        // the same, so the payload can be either.
        assert_eq!(
            parse_feature_ids("0\n1\n2\n155\n", full),
            Some(SlotSet::from_slots([0usize, 1, 2, 155]))
        );
        assert_eq!(parse_feature_ids("0 1 1 2", full), None, "duplicate");
        assert_eq!(parse_feature_ids("0 5 3", full), None, "descending");
        assert_eq!(parse_feature_ids("0 x 3", full), None, "unparseable");
        assert_eq!(parse_feature_ids("", full), None, "empty");
        assert_eq!(parse_feature_ids("-1 3", full), None, "negative");
        assert_eq!(
            parse_feature_ids(&format!("0 {full}"), full),
            None,
            "past the registry's full width"
        );
        assert_eq!(
            parse_feature_ids(&format!("0 {}", full - 1), full),
            Some(SlotSet::from_slots([0usize, full - 1])),
            "the last registered id is in range"
        );
    }

    /// A 1-layer bake of `ids.len()` inputs declaring those ids explicitly.
    pub(crate) fn ids_bake_bytes(ids: &[usize]) -> Vec<u8> {
        let list = ids
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        bake_bytes_with_metadata(
            ids.len(),
            &format!(
                r#"{{"key": "{FEATURE_IDS_KEY}", "type": "utf8", "text": "{}"}}"#,
                list.replace('\n', "\\n")
            ),
        )
    }

    fn ids_bake_bytes_no_decl(n: usize) -> Vec<u8> {
        bake_bytes_with_metadata(n, "")
    }

    fn bake_bytes_with_metadata(n: usize, md: &str) -> Vec<u8> {
        let arr = |v: &[f32]| -> String {
            let mut s = String::from("[");
            for (k, x) in v.iter().enumerate() {
                if k > 0 {
                    s.push(',');
                }
                s.push_str(&x.to_string());
            }
            s.push(']');
            s
        };
        let json = format!(
            r#"{{
                "schema_hash": 1,
                "scaler_mean": {mean},
                "scaler_scale": {scale},
                "metadata": [{md}],
                "layers": [
                    {{"in_dim": {n}, "out_dim": 1, "activation": "identity",
                      "dtype": "f32", "weights": {w}, "biases": [0.0]}}
                ]
            }}"#,
            mean = arr(&vec![0.0f32; n]),
            scale = arr(&vec![1.0f32; n]),
            w = arr(&vec![1.0f32; n]),
        );
        zenpredict_bake::bake_from_json_str(&json).expect("synthetic bake must build")
    }

    /// **G4.2, the bake half** — every shipped bake resolves to a layout whose
    /// width is EXACTLY its declared caller width, dense or identity.
    ///
    /// This asserted `is_identity()` for every bake until 2026-09-06, when
    /// `A` / `B` / `BHdr` / `D` were flipped to the dense contract
    /// (`benchmarks/dense_bake_flip_2026-09-06.md`). The invariant that
    /// actually protects a served vector is not "identity" — it is that the
    /// layout the runtime resolves and the width the bake asks for are the
    /// same number, so a bake is never handed a vector shaped for something
    /// else. That holds for both kinds and is what is asserted now.
    ///
    /// The safety argument the old name pointed at is kept as its own
    /// assertion: a DENSE layout is only ever read from an explicit
    /// `zentrain.feature_ids` list, never inferred from a
    /// `zentrain.feature_set_id` that happens to name fewer slots than the
    /// declared width (a `944`-with-structural-zeros producer names 265 at
    /// width 944, and reading that as dense would permute the vector).
    #[test]
    fn every_shipped_bake_resolves_to_its_own_declared_width() {
        let mut n = 0usize;
        let mut dense = 0usize;
        for (name, p) in crate::serving::shipped_profiles() {
            for bytes in p.params().scoring_bake_bytes() {
                let Ok(m) = crate::mlp::Model::from_bytes(bytes) else {
                    continue;
                };
                let l = declared_layout(&m);
                let derived = crate::derived_inputs::from_model(&m).unwrap();
                let gather = derived.as_ref().map_or(m.caller_input_width(), |d| {
                    assert_eq!(d.model_width(), m.caller_input_width());
                    d.gather_width()
                });
                assert_eq!(
                    l.width(),
                    gather,
                    "{name}: layout {} is {} wide but the bake asks for {}",
                    l.name(),
                    l.width(),
                    m.caller_input_width()
                );
                if !l.is_identity() {
                    dense += 1;
                    // A dense layout must come from the explicit id list.
                    assert!(
                        crate::declared_feature_ids(&m).is_some(),
                        "{name}: resolved to the dense layout {} without an explicit \
                         `zentrain.feature_ids` declaration — a dense layout inferred \
                         from anything else would permute the vector it is served",
                        l.name()
                    );
                    assert_eq!(
                        l.width(),
                        gather,
                        "{name}: a dense bake must fill every gathered input"
                    );
                }
                n += 1;
            }
        }
        use crate::serving::{expected_min_bake_count, expected_min_dense_count};
        assert!(
            n >= expected_min_bake_count(),
            "the census must actually see bakes, saw {n}, expected >= {}",
            expected_min_bake_count()
        );
        assert!(
            dense >= expected_min_dense_count(),
            "A/B/BHdr/D ship DENSE since 2026-09-06 (of those reachable under the active \
             features) — saw {dense} dense of {n}, expected >= {}; if a flip was reverted, \
             say so here rather than letting this pass quietly",
            expected_min_dense_count()
        );
    }
}
