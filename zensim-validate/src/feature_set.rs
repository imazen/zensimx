//! **Feature-set identity for bakes, features roots and grids** — the
//! consumer of `zensim::feature_set_id`, and the check that closes the
//! `--regime 944` silent-mis-scoring bug at the root.
//!
//! Design + the alias table: `docs/FEATURE_SET_IDS.md`. Registry:
//! `benchmarks/feature_sets_registry.json`, embedded here with `include_str!`
//! so a binary is never one missing file away from silently having no
//! registry.
//!
//! # Producer vs consumer
//!
//! Both are a [`FeatureSetRef`]; they differ only in what the slot set means.
//!
//! * a **PRODUCER** (an extractor, a features root, a dial grid) — its slots
//!   are what it POPULATES;
//! * a **CONSUMER** (a bake) — its slots are what it READS
//!   ([`crate::block_profile::used_caller_lines`]).
//!
//! A read is sound iff the widths agree, `consumer.slots ⊆ producer.slots`,
//! and the eras agree. The middle one is the bug: shipped **B** reads 49 lines
//! in `f156..371`, the `ext944` root populates none of them, and the resulting
//! CID22 **0.3862** (true: **0.8764**) came back with no error and no warning.
//!
//! # Why the registry is embedded, not read from disk
//!
//! The check is a REFUSAL surface. A registry that can be absent is a check
//! that can be skipped by deleting a file; embedding it makes "no registry" a
//! compile error instead of a silent pass. It is also what lets
//! `every_registered_set_agrees_with_the_hash_owner` hold the committed JSON
//! to `zensim::feature_set_id::slots_hash8` — nothing here re-derives the hash.

use std::collections::BTreeMap;
use std::path::Path;

use zenpredict::Model;
use zensim::feature_set_id::{ComputeParts, ComputeToken, ERA_UNKNOWN, FeatureSetId, SlotSet};

/// The committed registry, embedded at build time.
const REGISTRY_JSON: &str = include_str!("../../benchmarks/feature_sets_registry.json");

/// A registered feature set. `pinned` is `Some` for `kind: "set"` (slots and
/// hash pinned) and `None` for `kind: "class"` (compute + layout + era only —
/// a MODEL class whose exact read-set differs per bake, so the name is the
/// class and the hash is the instance).
#[derive(Debug, Clone)]
pub struct RegisteredSet {
    /// The registry key — the canonical id string, or the class form for a
    /// `class` entry.
    pub key: String,
    /// Compute + layout + era, always present.
    pub id: ClassId,
    /// `|slots|` as recorded, for a pinned set.
    pub n_slots: Option<usize>,
    /// `"producer"` (populates) or `"consumer"` (reads).
    pub role: String,
    /// The legacy count/name this set has been called.
    pub legacy_name: String,
    /// The doc/record that establishes the set exists.
    pub evidence: String,
    pinned: Option<FeatureSetRef>,
}

impl RegisteredSet {
    /// The fully-pinned reference, for a `kind: "set"` entry.
    #[must_use]
    pub fn as_ref(&self) -> Option<&FeatureSetRef> {
        self.pinned.as_ref()
    }
}

/// Compute + layout + era, without a slot hash — what a `class` entry pins and
/// what the canonical id string renders minus its `#hash8`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassId {
    /// The COMPUTE token set.
    pub compute: ComputeParts,
    /// The emitted vector width, when the registry entry records one. A
    /// RECONSTRUCTION HINT, not part of the identity — see
    /// `zensim::feature_set_id`'s module docs.
    pub layout: Option<usize>,
    /// The registered extractor-era token.
    pub era: String,
}

impl ClassId {
    /// `"<compute>/<era>"` — the canonical class form, without the hash and
    /// without the layout.
    #[must_use]
    pub fn to_string_class(&self) -> String {
        format!("{}/{}", self.compute, self.era)
    }

    /// The LEGACY class form `"<compute>@w<layout>/<era>"`, when a layout is
    /// recorded. Registry keys were written this way and are append-only, so
    /// a lookup must still be able to spell one.
    #[must_use]
    pub fn to_string_class_legacy(&self) -> Option<String> {
        self.layout
            .map(|w| format!("{}@w{}/{}", self.compute, w, self.era))
    }
}

/// A resolved feature-set reference: the id, the slot set it stands for, and
/// HOW we learned it.
#[derive(Debug, Clone)]
pub struct FeatureSetRef {
    /// The full id (compute + layout + era + slot hash).
    pub id: FeatureSetId,
    /// The slots this reference stands for — populated (producer) or read
    /// (consumer).
    pub slots: SlotSet,
    /// **The ARTIFACT's emitted/declared vector width**, when it is known — a
    /// bake's `caller_input_width()`, a table's `feat_*` column count.
    ///
    /// This left [`FeatureSetId`] on 2026-09-06 because it is not part of a
    /// feature set's identity: a densified shipped `B` and its wide twin
    /// produced the same compute tokens and the same slots hash at `@w95` and
    /// `@w372`, and [`check`] reported a mismatch for a difference that cannot
    /// make a read unsound. Here it is a property of the thing on disk, and
    /// only a real SHORTFALL (a consumer needing a wider row than the producer
    /// emits) is a finding.
    pub layout: Option<usize>,
    /// Human description of where the id came from, printed beside every use.
    pub source: String,
    /// `true` when the id was INFERRED (from the registry's root/regime tables
    /// or an alias) rather than read from a stored `feature_set_id`. An
    /// inferred id is evidence about the artifact's NAME, never about its
    /// BYTES — every consumer must badge it.
    pub inferred: bool,
}

/// What kind of disagreement a [`check`] found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MismatchKind {
    /// The consumer's declared width differs from the producer's layout. The
    /// runtime already fails loud on this (`FeatureLenMismatch`); reported
    /// here so a dry check sees it too.
    LayoutDiffers,
    /// **The bug.** The consumer reads slots the producer does not populate.
    SlotsNotPopulated,
    /// Same slots, different extractor era — the values differ, and the shift
    /// is model-specific, so the number cannot be corrected, only re-verdicted.
    EraDiffers,
    /// One side's era is `unknown`. Never silently a match.
    EraUnknown,
    /// The consumer's COMPUTE tokens are not a subset of the producer's. A
    /// weaker, more legible restatement of `SlotsNotPopulated`, reported
    /// alongside it.
    ComputeNotCovered,
}

/// One disagreement, with the actionable detail.
#[derive(Debug, Clone)]
pub struct Mismatch {
    /// Which kind.
    pub kind: MismatchKind,
    /// One line naming exactly what differs.
    pub detail: String,
}

impl std::fmt::Display for Mismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.detail)
    }
}

/// Compare a CONSUMER (a bake) against a PRODUCER (a features root / grid).
///
/// Returns every disagreement, most-severe first. An empty result means the
/// read is sound on all three axes of `docs/FEATURE_SET_IDS.md` §4.
#[must_use]
pub fn check(consumer: &FeatureSetRef, producer: &FeatureSetRef) -> Vec<Mismatch> {
    let mut out = Vec::new();
    // **A SHORTFALL, not a difference.** This used to fire whenever the two
    // ids' `@w<N>` components differed, which after the dense-bake flip meant
    // every dense-bake/wide-table pair — a mismatch reported inside a REFUSAL
    // surface for a difference that cannot make a read unsound (a dense bake
    // GATHERS its ids out of a wider row; that is the whole design). The
    // unsound direction is the other one: a consumer that needs a wider row
    // than the producer emits. Both widths must be KNOWN to say anything.
    if let (Some(c), Some(p)) = (consumer.layout, producer.layout)
        && c > p
    {
        out.push(Mismatch {
            kind: MismatchKind::LayoutDiffers,
            detail: format!(
                "the bake needs a {c}-wide row and the table emits {p} — the shortfall is \
                 filled structurally, so every number below it is computed from fill"
            ),
        });
    }
    if !producer.slots.covers(&consumer.slots) {
        let missing = producer.slots.missing_from(&consumer.slots);
        out.push(Mismatch {
            kind: MismatchKind::SlotsNotPopulated,
            detail: format!(
                "the bake READS {} slot(s) the table does not POPULATE: {missing}",
                missing.len()
            ),
        });
        let uncovered: Vec<&str> = ComputeToken::ALL
            .iter()
            .filter(|t| consumer.id.compute().contains(**t) && !producer.id.compute().contains(**t))
            .map(|t| t.as_str())
            .collect();
        if !uncovered.is_empty() {
            out.push(Mismatch {
                kind: MismatchKind::ComputeNotCovered,
                detail: format!(
                    "compute families read but not produced: {}",
                    uncovered.join("+")
                ),
            });
        }
    }
    if consumer.id.era() == ERA_UNKNOWN || producer.id.era() == ERA_UNKNOWN {
        out.push(Mismatch {
            kind: MismatchKind::EraUnknown,
            detail: format!(
                "era not established (bake {}, table {}) — 'we do not know which extractor made this' \
                 has the same consequence for a published number as 'a different one'",
                consumer.id.era(),
                producer.id.era()
            ),
        });
    } else if consumer.id.era() != producer.id.era() {
        out.push(Mismatch {
            kind: MismatchKind::EraDiffers,
            detail: format!(
                "bake trained on era {}, table is era {} — the shift is model-specific, so the \
                 number cannot be corrected across the boundary, only re-verdicted",
                consumer.id.era(),
                producer.id.era()
            ),
        });
    }
    out
}

/// Derive a bake's CONSUMER reference from its bytes.
///
/// Slots = the structurally-used caller lines
/// ([`crate::block_profile::used_caller_lines`], the SAME definition
/// `bake_block_profile` tabulates). COMPUTE = the families those reads fall
/// in. LAYOUT = `Model::caller_input_width()` — never `n_inputs()`, which is
/// the pruned internal width and is a third, different number.
pub fn bake_feature_set_ref(model: &Model, era: &str) -> Result<FeatureSetRef, String> {
    let width = model.caller_input_width();
    // E33 `fx1`: a derived-input bake's positions are entries over its declared ids, so the
    // positional mapping below would mis-name its reads. zensim's read-set owner (the serving
    // planner) answers directly: a live `in` reads its id, a live `product` reads both factors.
    if model
        .metadata()
        .get(crate::derived_inputs::DERIVED_INPUTS_KEY)
        .is_some()
    {
        let scorer = zensim::BakeScorer::new(model).map_err(|e| e.to_string())?;
        let ids = scorer.consumed_feature_ids().map_err(|e| e.to_string())?;
        // The wire is the gathered declared read set, as for a dense bake. `caller_input_width()`
        // counts model inputs (direct entries plus products) and is not a row width.
        let declared = zensim::declared_feature_ids(model)
            .ok_or("derived-input bake without zentrain.feature_ids")?
            .len();
        let slots = SlotSet::from_slots(ids.into_iter().map(usize::from));
        let compute = compute_parts_for_slots(&slots);
        let id = FeatureSetId::from_slots(compute, era, &slots)
            .ok_or_else(|| format!("invalid era token {era:?} (charset [a-z0-9_]+)"))?;
        return Ok(FeatureSetRef {
            id,
            slots,
            layout: Some(declared),
            source: "derived from bake bytes (zensim read set of a derived-input bake)".to_string(),
            inferred: false,
        });
    }
    let used = crate::block_profile::used_caller_lines(model)?;
    // `used_caller_lines` returns POSITIONS. For every identity-layout bake a
    // position IS a feature id, which is why this was sound for four months.
    // For a bake that DECLARES a dense read set it is not: under `dense95`,
    // position 3 is whatever id the bake declared third, and deriving the id
    // from positions would name `basic@w95` (slots `0-94`) for a bake that
    // actually reads `f3..f369` across four families. Translate through the
    // declaration when there is one — the same `zensim::declared_feature_ids`
    // owner the runtime and the scorers read.
    let (slots, source) = match zensim::declared_feature_ids(model) {
        Some(ids) => (
            SlotSet::from_slots(
                used.iter()
                    .filter_map(|&pos| ids.get(pos).map(|&id| usize::from(id))),
            ),
            "derived from bake bytes (used caller lines, mapped through the \
             bake's declared feature ids)"
                .to_string(),
        ),
        None => (
            SlotSet::from_slots(used),
            "derived from bake bytes (structurally-used caller lines)".to_string(),
        ),
    };
    let compute = compute_parts_for_slots(&slots);
    // The bake's caller width rides on the REF, not on the id: it describes
    // this artifact's wire, and a dense bake's is deliberately narrower than
    // the table it reads.
    let id = FeatureSetId::from_slots(compute, era, &slots)
        .ok_or_else(|| format!("invalid era token {era:?} (charset [a-z0-9_]+)"))?;
    Ok(FeatureSetRef {
        id,
        slots,
        layout: Some(width),
        source,
        inferred: false,
    })
}

/// The era a bake declares it was TRAINED on, from its embedded
/// `zentrain.feature_set_id` metadata, or [`ERA_UNKNOWN`].
///
/// The stored value is the PRODUCER id of the tables the bake trained on — the
/// bake's own consumer id is always derivable from its bytes, so it is not
/// stored.
#[must_use]
pub fn bake_declared_training_set(model: &Model) -> Option<FeatureSetId> {
    let md = model.metadata();
    let entry = md.get("zentrain.feature_set_id")?;
    FeatureSetId::parse(core::str::from_utf8(entry.value).ok()?.trim())
}

/// Classify a slot set into COMPUTE tokens: a token is named when the set
/// touches its family at all.
///
/// Family granularity is deliberate — this is the NAME, and the exact read set
/// is the hash. The two scattered tranches (`moments` / `classc`) are named
/// only when their owning block is NOT also touched, mirroring
/// `ComputeSet::raw_moments` / `bounded_err`: when `append` runs it owns
/// `GLOBAL_*`, so a set covering the append block is `append`, not
/// `append+moments`.
#[must_use]
pub fn compute_parts_for_slots(slots: &SlotSet) -> ComputeParts {
    use ComputeToken as T;
    let reg = registry();
    let touches = |tok: T| -> bool {
        reg.token_slots(tok)
            .map(|s| !s.intersect(slots).is_empty())
            .unwrap_or(false)
    };
    let mut p = ComputeParts::EMPTY;
    for t in [
        T::Basic,
        T::Peaks,
        T::Masked,
        T::Iw,
        T::V2,
        T::Append,
        T::Append2,
        T::Csfw,
    ] {
        if touches(t) {
            p = p.with(t);
        }
    }
    // `carriers` is a SUBSET of the pool block: name it only when the pool
    // reads are exactly the ten carrier slots (otherwise `peaks`/`masked`/`iw`
    // already describe them).
    if let (Some(carriers), Some(pool)) = (reg.token_slots(T::Carriers), reg.pool_block())
        && !slots.intersect(pool).is_empty()
        && slots.intersect(pool) == *carriers
    {
        p = ComputeParts::EMPTY
            .with(T::Carriers)
            .with_all_except_pools(p);
    }
    if !p.contains(T::Append)
        && let Some(m) = reg.token_slots(T::Moments)
        && !m.intersect(slots).is_empty()
    {
        p = p.with(T::Moments);
    }
    if !p.contains(T::V2)
        && let Some(c) = reg.token_slots(T::ClassC)
        && !c.intersect(slots).is_empty()
    {
        p = p.with(T::ClassC);
    }
    p
}

/// Small local extension so `compute_parts_for_slots` can swap the pool
/// tokens for `carriers` without reaching into `ComputeParts`' bits.
trait ComputePartsExt {
    fn with_all_except_pools(self, other: ComputeParts) -> ComputeParts;
}
impl ComputePartsExt for ComputeParts {
    fn with_all_except_pools(self, other: ComputeParts) -> ComputeParts {
        use ComputeToken as T;
        let mut out = self;
        for t in other.iter() {
            if !matches!(t, T::Peaks | T::Masked | T::Iw) {
                out = out.with(t);
            }
        }
        out
    }
}

/// Resolve a features root's PRODUCER reference.
///
/// Order, most authoritative first:
/// 1. the root's `_MANIFEST.json` `"feature_set_id"` key (asserted, not
///    inferred) — what every new artifact carries per the design's §6.1;
/// 2. the registry's `roots` table, keyed on the exact path (INFERRED);
/// 3. the root's `_MANIFEST.json` `"regime"` prose string, via the registry's
///    `regime_strings` table (INFERRED).
///
/// `None` when nothing resolves — which callers must report as "era not
/// established", never as a pass.
#[must_use]
pub fn root_feature_set_ref(root: &Path) -> Option<FeatureSetRef> {
    let reg = registry();
    let manifest = std::fs::read_to_string(root.join("_MANIFEST.json")).ok();
    // 1. an asserted, stored id.
    if let Some(txt) = manifest.as_deref()
        && let Some(v) = json_string_field(txt, "feature_set_id")
        && let Some(id) = FeatureSetId::parse(v.trim())
    {
        return resolve_declared_set(
            id,
            format!("{}/_MANIFEST.json feature_set_id", root.display()),
        );
    }

    // 2. the registry's root table.
    let key = root.to_string_lossy();
    let key = key.trim_end_matches('/');
    if let Some(set_id) = reg.roots.get(key)
        && let Some(set) = reg.set(set_id)
        && let Some(r) = set.as_ref()
    {
        let mut r = r.clone();
        r.source = format!("registry roots[{key}]");
        r.inferred = true;
        return Some(r);
    }
    // 3. the manifest's prose regime string.
    if let Some(txt) = manifest.as_deref()
        && let Some(regime) = json_string_field(txt, "regime")
        && let Some((compute, layout, slots)) = reg.regime(regime.trim())
    {
        let era = reg
            .era_for_root(key)
            .unwrap_or_else(|| ERA_UNKNOWN.to_string());
        let id = FeatureSetId::from_slots(compute, &era, &slots)?;
        return Some(FeatureSetRef {
            id,
            slots,
            layout: Some(layout),
            source: format!("{}/_MANIFEST.json regime {regime:?}", root.display()),
            inferred: true,
        });
    }
    None
}

fn resolve_declared_set(id: FeatureSetId, source: String) -> Option<FeatureSetRef> {
    let reg = registry();
    // The slot set comes from the registry when the id is registered; an
    // unregistered id still identifies itself, with an empty slot set that
    // makes every coverage check report rather than silently pass.
    //
    // Look the id up BOTH ways. Registry keys are the CLASS form
    // (`<compute>@w<layout>/<era>`, hash in the value's `slots_hash8`), so
    // an exact-key lookup on the full id -- which is what a manifest
    // declares -- matched NOTHING, and every manifest-declared root
    // resolved to an EMPTY slot set. That is not a cosmetic miss: an empty
    // producer set makes `check` report `SlotsNotPopulated` for every slot
    // the consumer reads, so the FIRST root to carry the key the design
    // calls most authoritative got 28 spurious "not populated" slots
    // (2026-09-05, the postC 372 root). Fall back to the class form, and
    // only accept it when the registered `slots_hash8` AGREES with the
    // declared one -- a disagreement means the name and the bytes have
    // come apart, which must not resolve silently.
    // Registry keys are append-only and were written in the LEGACY
    // `@w<N>` spelling, so a lookup tries every spelling of the same id:
    // the string as declared, the canonical layout-free form, and both
    // class forms. `Registry::set` also indexes the layout-free spelling
    // of every legacy key, so an id declared canonically finds a key
    // written with a width.
    let class_forms: Vec<String> = [
        Some(format!("{}/{}", id.compute(), id.era())),
        id.layout_width()
            .map(|w| format!("{}@w{}/{}", id.compute(), w, id.era())),
    ]
    .into_iter()
    .flatten()
    .collect();
    let slots = reg
        .set(&id.to_string())
        .or_else(|| reg.set(&id.clone().layout_free().to_string()))
        .or_else(|| {
            class_forms.iter().find_map(|c| {
                reg.set(c).filter(|s| {
                    s.as_ref()
                        .is_some_and(|r| r.id.slots_hash() == id.slots_hash())
                })
            })
        })
        .and_then(|s| s.as_ref().map(|r| r.slots.clone()))
        .unwrap_or_default();
    let layout = id.layout_width();
    Some(FeatureSetRef {
        id,
        slots,
        layout,
        source,
        inferred: false,
    })
}

/// Resolve a specific table's declarations. Per-file sidecars may refine a
/// root, but conflicting explicit IDs/revisions are never guessed away.
pub fn table_feature_set_ref(path: &Path) -> Result<Option<FeatureSetRef>, String> {
    let metadata = table_metadata(path)?;
    if let Some(r) = crate::palette_training::reference(
        &metadata,
        format!("{}: explicit E32 research projection", path.display()),
    )? {
        return Ok(Some(r));
    }
    match metadata.get("feature_set_id") {
        Some(value) => {
            let id = value
                .as_str()
                .and_then(FeatureSetId::parse)
                .ok_or_else(|| format!("{}: malformed feature_set_id", path.display()))?;
            Ok(resolve_declared_set(
                id,
                format!("{}: stored table declaration", path.display()),
            ))
        }
        None => Ok(root_feature_set_ref(
            path.parent().unwrap_or(Path::new(".")),
        )),
    }
}

pub(crate) fn table_metadata(path: &Path) -> Result<serde_json::Value, String> {
    let root = path.parent().unwrap_or(Path::new("."));
    let mut merged = serde_json::Map::new();
    for manifest in [
        root.join("_MANIFEST.json"),
        std::path::PathBuf::from(format!("{}.manifest.json", path.display())),
        std::path::PathBuf::from(format!("{}._MANIFEST.json", path.display())),
    ] {
        let bytes = match std::fs::read(&manifest) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(format!("{}: {e}", manifest.display())),
        };
        let value: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| format!("{}: invalid manifest: {e}", manifest.display()))?;
        if !value.is_object() {
            return Err(format!(
                "{}: manifest must be an object",
                manifest.display()
            ));
        }
        let filename = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let per_file = value.get("files").and_then(|v| v.get(filename));
        for scope in [Some(&value), per_file].into_iter().flatten() {
            for key in [
                "feature_set_id",
                "formula_revision",
                "decoder_era",
                "decoder_revision",
                "sampling",
                "research_palette",
                "data_role",
                "data_role_decision_required",
                "human_sources",
                "table_sha256",
                "keys_sha256",
                "row_keys_sha256",
                "row_selection_sha256",
                "row_selection",
                "rows",
                "rows_kept",
                "observations",
            ] {
                if !matches!(
                    key,
                    "feature_set_id"
                        | "formula_revision"
                        | "decoder_era"
                        | "decoder_revision"
                        | "sampling"
                ) && scope.get("research_palette").is_none()
                {
                    continue;
                }
                let Some(v) = scope.get(key) else { continue };
                if let Some(old) = merged.get(key) {
                    // Formula revisions occur as both strings and integers.
                    let norm = |v: &serde_json::Value| {
                        v.as_str()
                            .map(str::to_owned)
                            .unwrap_or_else(|| v.to_string())
                    };
                    if norm(old) != norm(v) {
                        return Err(format!(
                            "{}: conflicting {key}: {old} vs {v}",
                            manifest.display()
                        ));
                    }
                }
                merged.insert(key.into(), v.clone());
            }
        }
    }
    Ok(merged.into())
}

/// Read only a top-level string; nested historical notes cannot impersonate
/// a declaration. Malformed present manifests are refused by table admission.
fn json_string_field(txt: &str, key: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(txt)
        .ok()?
        .get(key)?
        .as_str()
        .map(str::to_owned)
}

/// The parsed registry.
#[derive(Debug)]
pub struct Registry {
    formula_revisions: BTreeMap<String, u8>,
    sets: BTreeMap<String, RegisteredSet>,
    /// The LAYOUT-FREE spelling of every declared key → that key. Built at
    /// load time so a canonical id resolves to an append-only legacy entry.
    layout_free_keys: BTreeMap<String, String>,
    /// features-root path → set key.
    pub roots: BTreeMap<String, String>,
    regimes: BTreeMap<String, (ComputeParts, usize, SlotSet)>,
    aliases: BTreeMap<String, Vec<String>>,
    token_slots: BTreeMap<&'static str, SlotSet>,
}

impl Registry {
    /// Arithmetic revision of a recorded table era; absent remains unknown.
    pub fn formula_revision(&self, era: &str) -> Option<u8> {
        self.formula_revisions.get(era).copied()
    }

    /// A registered set by any spelling of its key.
    ///
    /// **Registry keys are APPEND-ONLY and were all written in the legacy
    /// `<compute>@w<N>/<era>[#hash]` form.** The canonical id form dropped the
    /// `@w<N>` component on 2026-09-06, so a caller holding a canonical id
    /// would otherwise miss every entry. This resolves both, by consulting an
    /// index built at load time from the layout-free spelling of each key —
    /// which is what makes "registry aliases for every existing `@w<N>` id"
    /// a lookup rather than a migration. Nothing on disk is edited.
    #[must_use]
    pub fn set(&self, key: &str) -> Option<&RegisteredSet> {
        self.sets.get(key).or_else(|| {
            self.layout_free_keys
                .get(key)
                .and_then(|k| self.sets.get(k))
        })
    }
    /// Every registered set, key-ordered. Iterates the DECLARED keys only —
    /// the layout-free aliases are a lookup index, not extra sets.
    pub fn sets(&self) -> impl Iterator<Item = (&str, &RegisteredSet)> {
        self.sets.iter().map(|(k, v)| (k.as_str(), v))
    }
    /// Every layout-free alias → declared key, alias-ordered. The alias table
    /// the id-form change rests on, exposed so a test can hold it.
    pub fn layout_free_aliases(&self) -> impl Iterator<Item = (&str, &str)> {
        self.layout_free_keys
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
    }
    /// Every alias, name-ordered.
    pub fn aliases(&self) -> impl Iterator<Item = (&str, &[String])> {
        self.aliases.iter().map(|(k, v)| (k.as_str(), v.as_slice()))
    }
    /// The ids a legacy name has meant. Length > 1 is AMBIGUOUS by
    /// construction, and a consumer must say so rather than pick one.
    #[must_use]
    pub fn aliases_for(&self, name: &str) -> &[String] {
        self.aliases.get(name).map(Vec::as_slice).unwrap_or(&[])
    }
    /// The canonical slot set of one COMPUTE token, at 4 scales.
    #[must_use]
    pub fn token_slots(&self, t: ComputeToken) -> Option<&SlotSet> {
        self.token_slots.get(t.as_str())
    }
    /// The whole v1 pool block (`peaks ∪ masked ∪ iw`).
    #[must_use]
    pub fn pool_block(&self) -> Option<&SlotSet> {
        self.token_slots.get("__pool_block")
    }
    /// Resolve a manifest `regime` prose string to `(compute, layout, slots)`.
    #[must_use]
    pub fn regime(&self, s: &str) -> Option<(ComputeParts, usize, SlotSet)> {
        self.regimes.get(s).cloned()
    }
    /// The era token registered for a features-root path, via the `roots`
    /// table's target set.
    #[must_use]
    pub fn era_for_root(&self, path: &str) -> Option<String> {
        let key = self.roots.get(path)?;
        Some(self.sets.get(key)?.id.era.clone())
    }
}

/// The embedded registry, parsed once.
#[must_use]
pub fn registry() -> &'static Registry {
    use std::sync::OnceLock;
    static REG: OnceLock<Registry> = OnceLock::new();
    REG.get_or_init(|| parse_registry(REGISTRY_JSON).expect("the COMMITTED registry must parse"))
}

/// The arithmetic revisions this validator admits in stored data.
///
/// ONE list. The three admission sites below (registry parse, per-table
/// metadata, root manifest) previously spelled `matches!(n, 1 | 2)` out
/// separately, which is exactly how a new revision comes to be accepted in
/// one place and silently rejected in another. Mirrors
/// `zensim::feature_v2::FormulaRevision`; extend both together.
pub const ADMITTED_FORMULA_REVISIONS: &[u64] = &[1, 2, 3, 4, 5, 6, 7];

/// Whether a declared `formula_revision` value names a revision this build
/// knows how to read.
pub fn is_admitted_formula_revision(n: u64) -> bool {
    ADMITTED_FORMULA_REVISIONS.contains(&n)
}

fn parse_registry(txt: &str) -> Result<Registry, String> {
    let v: serde_json::Value =
        serde_json::from_str(txt).map_err(|e| format!("feature_sets_registry.json: {e}"))?;
    let formula_revisions = v
        .get("eras")
        .and_then(|x| x.as_object())
        .into_iter()
        .flat_map(|eras| eras.iter())
        .filter_map(|(era, value)| {
            value
                .get("formula_revision")
                .and_then(|x| x.as_u64())
                .filter(|n| is_admitted_formula_revision(*n))
                .map(|n| (era.clone(), n as u8))
        })
        .collect();
    let mut token_slots = BTreeMap::new();
    if let Some(map) = v.get("compute_tokens").and_then(|x| x.as_object()) {
        for (k, e) in map {
            if let Some(s) = e.get("slots").and_then(|x| x.as_str())
                && let Some(set) = SlotSet::parse(s)
                && let Some(tok) = ComputeToken::parse(k)
            {
                token_slots.insert(tok.as_str(), set);
            }
        }
    }
    // The v1 pool block, composed from its three registered tokens — never a
    // second literal.
    let pool = [ComputeToken::Peaks, ComputeToken::Masked, ComputeToken::Iw]
        .iter()
        .filter_map(|t| token_slots.get(t.as_str()))
        .fold(SlotSet::default(), |acc, s| acc.union(s));
    token_slots.insert("__pool_block", pool);

    let mut sets = BTreeMap::new();
    if let Some(map) = v.get("sets").and_then(|x| x.as_object()) {
        for (key, e) in map {
            let compute = e
                .get("compute")
                .and_then(|x| x.as_str())
                .and_then(ComputeParts::parse)
                .ok_or_else(|| format!("{key}: bad compute"))?;
            let layout = e
                .get("layout")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| format!("{key}: bad layout"))? as usize;
            let era = e
                .get("era")
                .and_then(|x| x.as_str())
                .ok_or_else(|| format!("{key}: bad era"))?
                .to_string();
            let id = ClassId {
                compute,
                layout: Some(layout),
                era: era.clone(),
            };
            let pinned = match e.get("slots").and_then(|x| x.as_str()) {
                None => None,
                Some(s) => {
                    let slots =
                        SlotSet::parse(s).ok_or_else(|| format!("{key}: unparseable slots"))?;
                    let fid = FeatureSetId::from_slots_with_layout(compute, layout, &era, &slots)
                        .ok_or_else(|| format!("{key}: bad era token"))?;
                    let stored = e.get("slots_hash8").and_then(|x| x.as_str());
                    if let Some(h) = stored
                        && h != format!("{:08x}", fid.slots_hash())
                    {
                        return Err(format!(
                            "{key}: stored slots_hash8 {h} != owner {:08x}",
                            fid.slots_hash()
                        ));
                    }
                    Some(FeatureSetRef {
                        id: fid,
                        slots,
                        layout: Some(layout),
                        source: format!("registry sets[{key}]"),
                        inferred: false,
                    })
                }
            };
            sets.insert(
                key.clone(),
                RegisteredSet {
                    key: key.clone(),
                    id,
                    n_slots: e
                        .get("n_slots")
                        .and_then(serde_json::Value::as_u64)
                        .map(|n| n as usize),
                    role: e
                        .get("role")
                        .and_then(|x| x.as_str())
                        .unwrap_or("producer")
                        .to_string(),
                    legacy_name: e
                        .get("legacy_name")
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string(),
                    evidence: e
                        .get("evidence")
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string(),
                    pinned,
                },
            );
        }
    }
    let mut roots = BTreeMap::new();
    if let Some(map) = v.get("roots").and_then(|x| x.as_object()) {
        for (k, e) in map {
            if let Some(s) = e.as_str() {
                roots.insert(k.clone(), s.to_string());
            }
        }
    }
    let mut regimes = BTreeMap::new();
    if let Some(map) = v.get("regime_strings").and_then(|x| x.as_object()) {
        for (k, e) in map {
            let (Some(c), Some(l), Some(s)) = (
                e.get("compute")
                    .and_then(|x| x.as_str())
                    .and_then(ComputeParts::parse),
                e.get("layout").and_then(serde_json::Value::as_u64),
                e.get("slots")
                    .and_then(|x| x.as_str())
                    .and_then(SlotSet::parse),
            ) else {
                continue;
            };
            regimes.insert(k.clone(), (c, l as usize, s));
        }
    }
    let mut aliases = BTreeMap::new();
    if let Some(map) = v.get("aliases").and_then(|x| x.as_object()) {
        for (k, e) in map {
            let ids: Vec<String> = e
                .get("ids")
                .and_then(|x| x.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            aliases.insert(k.clone(), ids);
        }
    }
    // The layout-free alias index. A key is `<compute>[@w<N>]/<era>[#hash8]`;
    // dropping `@w<N>` is a pure string edit at a fixed position, so the index
    // cannot disagree with the parse. A COLLISION (two declared keys whose
    // layout-free spellings are equal) would make a canonical lookup
    // ambiguous, so it is an ERROR rather than a last-writer-wins.
    let mut layout_free_keys: BTreeMap<String, String> = BTreeMap::new();
    for k in sets.keys() {
        let Some((head, rest)) = k.split_once('/') else {
            continue;
        };
        let Some((compute, _w)) = head.split_once('@') else {
            continue; // already layout-free
        };
        let alias = format!("{compute}/{rest}");
        if let Some(prev) = layout_free_keys.insert(alias.clone(), k.clone()) {
            return Err(format!(
                "registry keys {prev:?} and {k:?} have the same layout-free spelling                  {alias:?} — a canonical id could not resolve to one of them"
            ));
        }
    }
    Ok(Registry {
        formula_revisions,
        sets,
        layout_free_keys,
        roots,
        regimes,
        aliases,
        token_slots,
    })
}

// ── Resolving a bake's EVAL features root FROM THE BAKE (2026-09-05) ───────
//
// `scripts/run_full_eval.sh` used to hard-code one features root per regime,
// so a bake trained at a NON-default root had two possible outcomes and no
// third: a wrong-regime read (which `bake_verdict` correctly refuses) or no
// board cell at all. The concrete casualty was **A3b** — the replication
// wave's one genuinely-k=1 recipe, trained on
// `/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01`, which scores fine at
// its native root (CID22 0.88-0.89) and has no board row
// (`benchmarks/replication_wave_2026-09-05.md` §4c.3, "named gap, not a silent
// omission").
//
// The root is therefore DERIVED FROM THE BAKE, in this one owner:
//
//   1. an explicit caller override — always wins, always reported;
//   2. the bake's declared `zentrain.feature_set_id`, matched against the
//      registry's `roots` table (the id-first path, live for every bake
//      trained after the feature-set-id lane);
//   3. the bake's embedded `zentrain.repro` training-input paths, matched
//      against that same `roots` table (the path-first path, which is what a
//      pre-id bake like A3b has);
//   4. the regime default — but only as a DETERMINATION: the bake carries a
//      repro and its training inputs name no registered features root, i.e.
//      it trained on a training corpus that is not an eval root (every 372/720
//      bake). Reported with that reason.
//
// A bake with NO repro at all is `Err`: its root genuinely cannot be
// determined, and the caller must refuse rather than default silently.
// Ambiguity (two different registered roots in one repro) is also `Err` —
// picking one would be a guess.

/// Where a resolved features root came from. A caller that cannot say WHERE
/// the root came from cannot say whether it is right, so this is never a bare
/// path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RootSource {
    /// The caller passed it.
    Explicit,
    /// The bake's `zentrain.feature_set_id` names a registered set, and
    /// exactly one registered root produces that set.
    DeclaredFeatureSetId(String),
    /// The bake's `zentrain.repro` training inputs live under exactly one
    /// registered features root.
    ReproTrainingPaths(String),
    /// The bake HAS a repro and its training inputs name no registered
    /// features root — so it trained on a corpus that is not an eval root and
    /// the regime default applies. Carries the reason.
    RegimeDefault(String),
}

impl RootSource {
    /// One line for a log, so every run says which rule fired.
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            RootSource::Explicit => "explicit --features-root (caller override)".into(),
            RootSource::DeclaredFeatureSetId(id) => {
                format!("bake's zentrain.feature_set_id {id} -> registry roots")
            }
            RootSource::ReproTrainingPaths(p) => {
                format!("bake's zentrain.repro training inputs under {p}")
            }
            RootSource::RegimeDefault(why) => format!("regime default ({why})"),
        }
    }
}

/// A features root plus the rule that produced it.
#[derive(Debug, Clone)]
pub struct ResolvedRoot {
    pub root: std::path::PathBuf,
    pub source: RootSource,
}

/// Every path-shaped string in a bake's `zentrain.repro` that could name a
/// training input: the `--group`/`--val-group` argv values (`name:path:...`)
/// and the `inputs[].path` entries. Deliberately over-collects — matching is
/// done against the registry, so a non-root string simply never matches.
fn repro_candidate_paths(repro: &serde_json::Value) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    if let Some(a) = repro.get("argv").and_then(|x| x.as_array()) {
        for t in a.iter().filter_map(serde_json::Value::as_str) {
            // A `--group` value is `name:path[:w:w:mode]`; a bare path is also
            // accepted. Split on ':' and keep every absolute-looking piece.
            for piece in t.split(':') {
                if piece.starts_with('/') {
                    out.push(piece.to_string());
                }
            }
        }
    }
    for key in ["inputs", "files"] {
        if let Some(a) = repro.get(key).and_then(|x| x.as_array()) {
            for it in a {
                for k in ["path", "file", "input"] {
                    if let Some(sv) = it.get(k).and_then(serde_json::Value::as_str) {
                        out.push(sv.to_string());
                    }
                }
                if let Some(sv) = it.as_str() {
                    out.push(sv.to_string());
                }
            }
        }
    }
    out
}

/// The registered features roots any of `paths` lives under, longest-prefix
/// first so a nested registered root (`.../era2r4/foldapp2_views`) wins over
/// its parent.
fn registered_roots_covering(paths: &[String]) -> Vec<String> {
    let reg = registry();
    let mut keys: Vec<&String> = reg.roots.keys().collect();
    keys.sort_by_key(|k| std::cmp::Reverse(k.len()));
    let mut hits: Vec<String> = Vec::new();
    for p in paths {
        if let Some(k) = keys
            .iter()
            .find(|k| p.as_str() == k.as_str() || p.starts_with(&format!("{k}/")))
            && !hits.contains(k)
        {
            hits.push((*k).clone());
        }
    }
    hits
}

/// Resolve the EVAL features root for `model` — see the module note above for
/// the four-step rule and why a missing repro is an error rather than a
/// default.
///
/// `regime_default` is the root the caller's regime would have used; it is
/// returned only under [`RootSource::RegimeDefault`].
pub fn resolve_features_root(
    model: &Model,
    regime_default: &Path,
    explicit: Option<&Path>,
) -> Result<ResolvedRoot, String> {
    if let Some(p) = explicit {
        return Ok(ResolvedRoot {
            root: p.to_path_buf(),
            source: RootSource::Explicit,
        });
    }
    let reg = registry();
    // 2. the declared training set id.
    if let Some(id) = bake_declared_training_set(model) {
        let want = id.to_string();
        let mut hits: Vec<&String> = reg
            .roots
            .iter()
            .filter(|(_, v)| **v == want)
            .map(|(k, _)| k)
            .collect();
        hits.sort();
        hits.dedup();
        match hits.len() {
            1 => {
                return Ok(ResolvedRoot {
                    root: std::path::PathBuf::from(hits[0]),
                    source: RootSource::DeclaredFeatureSetId(want),
                });
            }
            0 => {} // declared but not a registered ROOT — fall through to repro
            _ => {
                return Err(format!(
                    "the bake declares training set {want}, and {} registered roots produce it \
                     ({}). Pass --features-root explicitly; picking one would be a guess.",
                    hits.len(),
                    hits.iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }
    }
    // 3./4. the embedded repro.
    let md = model.metadata();
    let Some(entry) = md.get("zentrain.repro") else {
        return Err(
            "the bake carries no zentrain.repro, so its features root cannot be determined. \
             Pass --features-root explicitly (never let it silently default)."
                .into(),
        );
    };
    let txt = core::str::from_utf8(entry.value)
        .map_err(|e| format!("zentrain.repro is not UTF-8: {e}"))?;
    let repro: serde_json::Value =
        serde_json::from_str(txt).map_err(|e| format!("zentrain.repro is not JSON: {e}"))?;
    let cands = repro_candidate_paths(&repro);
    let hits = registered_roots_covering(&cands);
    match hits.len() {
        1 => Ok(ResolvedRoot {
            root: std::path::PathBuf::from(&hits[0]),
            source: RootSource::ReproTrainingPaths(hits[0].clone()),
        }),
        0 => Ok(ResolvedRoot {
            root: regime_default.to_path_buf(),
            source: RootSource::RegimeDefault(
                "the bake's training inputs name no registered features root — it trained on a \
                 corpus that is not an eval root"
                    .into(),
            ),
        }),
        _ => Err(format!(
            "the bake's training inputs span {} registered features roots ({}). Pass \
             --features-root explicitly; picking one would be a guess.",
            hits.len(),
            hits.join(", ")
        )),
    }
}

/// What a bake NEEDS, derived from its own bytes — the replacement for asking
/// a caller to type `--regime`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedRegime {
    /// The narrowest registered layout width that can carry every id the bake
    /// reads. This is the number `--regime` was being used to spell.
    pub regime: usize,
    /// The highest feature id the bake reads.
    pub max_read_id: usize,
    /// How many ids it reads.
    pub n_read: usize,
    /// True when the bake reads any of `f156..371`. A folded regime ZEROES
    /// that block, so such a bake cannot be scored at 720/924/944 — this is
    /// the `--regime 944` silent-mis-scoring class (shipped `B`: CID22 0.3862
    /// against its true 0.8764).
    pub reads_pool_block: bool,
}

/// Derive [`DerivedRegime`] from a bake's bytes.
///
/// **`--regime N` was always a fact about the BAKE, typed by hand.** The bake
/// knows its own read set — literally, since 2026-09-06, for the four shipped
/// profiles that declare their ids — so the width it needs is derivable, and a
/// derived value cannot be typed wrong. Every candidate comes from
/// `zensim::feature_set_id::registered_layout_widths`, so this cannot invent a
/// regime the registry does not know.
///
/// The `era` argument only labels the returned reference internally; it does
/// not affect the width.
pub fn derive_regime(model: &Model) -> Result<DerivedRegime, String> {
    let r = bake_feature_set_ref(model, ERA_UNKNOWN)?;
    let max_read_id = r
        .slots
        .ranges()
        .last()
        .map(|(_, end)| end - 1)
        .ok_or_else(|| "the bake reads no feature ids".to_string())?;
    let mut widths: Vec<usize> = zensim::feature_set_id::registered_layout_widths().to_vec();
    widths.sort_unstable();
    let regime = widths
        .iter()
        .copied()
        .find(|w| max_read_id < *w)
        .ok_or_else(|| {
            format!("the bake reads f{max_read_id}, past every registered layout width {widths:?}")
        })?;
    let reads_pool_block = (156..372).any(|s| r.slots.contains(s));
    Ok(DerivedRegime {
        regime,
        max_read_id,
        n_read: r.slots.len(),
        reads_pool_block,
    })
}

#[cfg(test)]
mod derive_regime_tests {
    use super::*;

    fn bake(width: usize, live: impl Fn(usize) -> bool) -> Vec<u8> {
        let arr = |v: &[f32]| {
            let mut s = String::from("[");
            for (i, x) in v.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(&x.to_string());
            }
            s.push(']');
            s
        };
        let w: Vec<f32> = (0..width)
            .map(|i| if live(i) { 1.0 } else { 0.0 })
            .collect();
        let json = format!(
            r#"{{"schema_hash":1,"scaler_mean":{},"scaler_scale":{},"metadata":[],
                 "layers":[{{"in_dim":{width},"out_dim":1,"activation":"identity",
                 "dtype":"f32","weights":{},"biases":[0.0]}}]}}"#,
            arr(&vec![0.0; width]),
            arr(&vec![1.0; width]),
            arr(&w),
        );
        zenpredict_bake::bake_from_json_str(&json).expect("synthetic bake")
    }

    /// The width a caller used to type is a FACT ABOUT THE BAKE, and it is
    /// derivable — including the case the `--regime 944` bug is about: a
    /// 372-class bake that reads the pool block cannot be scored at a folded
    /// regime, and the derivation says so without being told.
    #[test]
    fn the_regime_is_derived_from_the_read_set() {
        // Basic-only: everything it reads fits in 372.
        let m = Model::from_bytes(&bake(372, |i| i < 28)).expect("parse");
        let d = derive_regime(&m).expect("derive");
        assert_eq!(d.regime, 372);
        assert_eq!(d.max_read_id, 27);
        assert_eq!(d.n_read, 28);
        assert!(!d.reads_pool_block);

        // Reads the IW pool: still 372, and flagged — this is shipped B's
        // shape, the one `--regime 944` silently mis-scored.
        let m = Model::from_bytes(&bake(372, |i| i == 3 || i == 369)).expect("parse");
        let d = derive_regime(&m).expect("derive");
        assert_eq!(d.regime, 372);
        assert_eq!(d.max_read_id, 369);
        assert!(d.reads_pool_block, "f369 is in the IW pool");

        // A 944-class bake needs 944, derived rather than typed.
        let m = Model::from_bytes(&bake(944, |i| !(156..700).contains(&i))).expect("parse");
        let d = derive_regime(&m).expect("derive");
        assert_eq!(d.regime, 944);
        assert_eq!(d.max_read_id, 943);
        assert!(!d.reads_pool_block, "it reads no f156..371");
    }

    /// Every candidate comes from the registry's width list, so the derivation
    /// cannot invent a regime nothing is registered at.
    #[test]
    fn the_derived_regime_is_always_a_registered_width() {
        for hi in [0usize, 155, 371, 500, 719, 900, 943, 955] {
            let m = Model::from_bytes(&bake(956, |i| i == hi)).expect("parse");
            let d = derive_regime(&m).expect("derive");
            assert!(
                zensim::feature_set_id::registered_layout_widths().contains(&d.regime),
                "f{hi} derived regime {} is not a registered width",
                d.regime
            );
            assert!(d.regime > hi, "f{hi} must fit in regime {}", d.regime);
        }
    }
}

/// Admit all training/selection table roots before reading their rows. Replay
/// preserves historical recipes while recording every unresolved era; it
/// never turns incompatible data into a qualification pass.
pub fn admit_training_tables(
    paths: &[std::path::PathBuf],
    replay: Option<&str>,
    selected_ids: Option<&[usize]>,
    max_features: Option<usize>,
) -> Result<serde_json::Value, String> {
    if replay.is_some_and(|s| s.trim().is_empty()) {
        return Err("--historical-replay requires a nonempty reason and recorded recipe".into());
    }
    // All E32 roles/maps are checked before keys or any training-table opens.
    // Legacy tables keep their original admission JSON and execution path.
    let declarations = paths
        .iter()
        .map(|p| table_metadata(p))
        .collect::<Result<Vec<_>, _>>()?;
    let palette = declarations
        .iter()
        .map(crate::palette_training::validate)
        .collect::<Result<Vec<_>, _>>()?;
    if palette.iter().any(|&p| p) {
        if replay.is_some()
            || !palette.iter().all(|&p| p)
            || selected_ids != Some(crate::palette_training::primary_ids().as_slice())
            || max_features != Some(1867)
        {
            return Err("E32 requires all projected legs, exactly the registered 462 IDs/width and no replay".into());
        }
        for (path, metadata) in paths.iter().zip(&declarations) {
            crate::palette_training::verify_keys(path, metadata)?;
        }
        for (path, metadata) in paths.iter().zip(&declarations) {
            if crate::train_manifest::sha256_file(path).map_err(|e| e.to_string())?
                != metadata["table_sha256"].as_str().unwrap()
            {
                return Err("E32 projected table byte pin changed".into());
            }
        }
    }
    let mut tables = Vec::new();
    let mut issues = Vec::new();
    let mut eras = std::collections::BTreeSet::new();
    let mut revisions = std::collections::BTreeSet::new();
    for path in paths {
        let metadata = table_metadata(path)?;
        let resolved = table_feature_set_ref(path)?;
        let actual_width = max_features
            .map(|_| crate::parquet_loader::table_feature_width(path))
            .transpose()?;
        let requested = actual_width.map(|w| {
            selected_ids.map_or_else(
                || (0..w.min(max_features.unwrap())).collect::<Vec<_>>(),
                |ids| ids.to_vec(),
            )
        });
        if let (Some(w), Some(ids)) = (actual_width, requested.as_ref())
            && let Some(id) = ids.iter().find(|&&id| id >= w)
        {
            return Err(format!(
                "{}: required feature f{id} is missing from {w}-column table",
                path.display()
            ));
        }
        if let Some(r) = &resolved {
            eras.insert(r.id.era().to_owned());
            if r.slots.is_empty() {
                issues.push(format!("{}: producer slots are unknown", path.display()));
            }
            if let Some(ids) = &requested {
                let want = SlotSet::from_slots(ids.iter().copied());
                if !r.slots.covers(&want) {
                    issues.push(format!(
                        "{}: requested IDs not populated by producer: {}",
                        path.display(),
                        r.slots.missing_from(&want)
                    ));
                }
            }
            let registered = registry().formula_revision(r.id.era());
            let declared = metadata
                .get("formula_revision")
                .map(|v| {
                    v.as_u64()
                        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
                        .filter(|n| is_admitted_formula_revision(*n))
                        .map(|n| n as u8)
                        .ok_or_else(|| format!("{}: invalid formula_revision {v}", path.display()))
                })
                .transpose()?;
            if let (Some(a), Some(b)) = (registered, declared)
                && a != b
            {
                return Err(format!(
                    "{}: formula revision {b} conflicts with registered era revision {a}",
                    path.display()
                ));
            }
            if let Some(rev) = declared.or(registered) {
                revisions.insert(rev);
            } else {
                issues.push(format!("{}: formula revision is unknown", path.display()));
            }
        } else {
            issues.push(format!(
                "{}: feature-set identity is unknown",
                path.display()
            ));
        }
        tables.push(serde_json::json!({
            "path": path, "feature_set_id": resolved.as_ref().map(|r| r.id.to_string()),
            "inferred": resolved.as_ref().map(|r| r.inferred),
            "source": resolved.as_ref().map(|r| &r.source),
            "stored_declarations": metadata, "actual_width": actual_width,
            "requested_ids": requested,
        }));
    }
    let samplings: std::collections::BTreeSet<_> = tables
        .iter()
        .map(|t| {
            t["stored_declarations"]
                .get("sampling")
                .cloned()
                .unwrap_or(serde_json::Value::Null)
                .to_string()
        })
        .collect();
    if samplings.len() > 1 {
        return Err("mixed sampling contracts in training tables".into());
    }
    let sampling = tables
        .first()
        .and_then(|t| t["stored_declarations"].get("sampling"))
        .cloned();
    if revisions.len() > 1 {
        issues.push(format!("mixed formula revisions: {revisions:?}"));
    }
    let decoders: std::collections::BTreeSet<_> = tables
        .iter()
        .filter_map(|t| {
            t["stored_declarations"]
                .get("decoder_era")
                .or_else(|| t["stored_declarations"].get("decoder_revision"))
        })
        .map(|v| v.to_string())
        .collect();
    if decoders.len() > 1 {
        issues.push(format!("mixed decoder declarations: {decoders:?}"));
    }
    if eras.len() > 1 {
        issues.push(format!("mixed table eras: {eras:?}"));
    }
    if !issues.is_empty() && replay.is_none() {
        return Err(format!(
            "table admission refused: {}. Resolve provenance, or use --historical-replay REASON for an explicitly unqualified recorded recipe",
            issues.join("; ")
        ));
    }
    let revision = if revisions.len() == 1 && issues.is_empty() {
        revisions.first().copied()
    } else {
        None
    };
    let mut admission = serde_json::json!({"tables": tables, "issues": issues, "historical_replay": replay,
        "formula_revision": revision, "sampling": sampling, "qualified_provenance": issues.is_empty() && replay.is_none() && tables.iter().all(|t|
            ["decoder_era", "decoder_revision"].iter().any(|key|
                t["stored_declarations"].get(key).is_some_and(|v| !v.is_null() && v.as_str() != Some(""))))});
    if palette.iter().any(|&p| p) {
        admission["research_family"] = serde_json::json!("palette_v2");
        admission["serving_allowed"] = serde_json::json!(false);
    }
    Ok(admission)
}

#[cfg(test)]
mod training_admission_tests {
    use super::*;
    #[test]
    fn assessment_projection_is_registered_by_exact_consumed_ids() {
        let v: serde_json::Value = serde_json::from_str(include_str!(
            "../../benchmarks/costset2_2026-10-03.candidate_ids.json"
        ))
        .unwrap();
        let ids: Vec<usize> = serde_json::from_value(v["candidates"]["by_v2fy"].clone()).unwrap();
        let slots = SlotSet::from_slots(ids);
        let id = FeatureSetId::from_slots_with_layout(
            compute_parts_for_slots(&slots),
            720,
            "rev5_localwin",
            &slots,
        )
        .unwrap();
        assert_eq!(id.to_string(), "basic+v2@w720/rev5_localwin#62adfc93");
        assert_eq!(
            resolve_declared_set(id, "explicit assessment".into())
                .unwrap()
                .slots,
            slots
        );
    }

    #[test]
    #[ignore = "requires explicitly pinned features-only assessment manifest"]
    fn external_assessment_tables_admission() {
        let p = std::path::PathBuf::from(std::env::var("SHIPPATH7_ASSESSMENT").unwrap());
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        assert_eq!(v["features_only"], true);
        assert_eq!(v["labels_read"], false);
        let ids: Vec<usize> = serde_json::from_value(v["feature_ids"].clone()).unwrap();
        assert_eq!(ids.len(), 420);
        let paths: Vec<std::path::PathBuf> = v["tables"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| {
                for spec in [t, &t["declaration"], &t["keys"]] {
                    let path = std::path::Path::new(spec["path"].as_str().unwrap());
                    assert!(
                        !path
                            .components()
                            .any(|c| c.as_os_str().to_string_lossy().contains("_sealed"))
                    );
                    assert_eq!(
                        crate::train_manifest::sha256_file(path).unwrap(),
                        spec["sha256"].as_str().unwrap()
                    );
                }
                std::path::PathBuf::from(t["path"].as_str().unwrap())
            })
            .collect();
        let mut admitted = admit_training_tables(&paths, None, Some(&ids), Some(720)).unwrap();
        for (table, binding) in admitted["tables"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .zip(v["tables"].as_array().unwrap())
        {
            table["sha256"] = binding["sha256"].clone();
            table["declaration"] = binding["declaration"].clone();
            table["keys"] = binding["keys"].clone();
        }
        assert!(admit_training_tables(&paths, None, Some(&[0]), Some(720)).is_err());
        assert!(admit_training_tables(&paths, None, None, Some(720)).is_err());
        assert_eq!(admitted["qualified_provenance"], true);
        assert_eq!(admitted["formula_revision"], 5);
        assert_eq!(admitted["historical_replay"], serde_json::Value::Null);
        std::fs::write(
            std::env::var("SHIPPATH7_ADMISSION_OUT").unwrap(),
            serde_json::to_vec_pretty(&admitted).unwrap(),
        )
        .unwrap();
        println!(
            "SHIPPATH7 actual Rust admission: {} declared features-only tables",
            paths.len()
        );
    }

    #[test]
    fn rev5_measured_subset_admits_real_ids_and_refuses_nan_padding() {
        let dir = std::env::temp_dir().join(format!("zensim-admit-rev5-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("teacher.csv");
        let header = (0..1853)
            .map(|i| format!("f{i}"))
            .collect::<Vec<_>>()
            .join(",");
        std::fs::write(&path, format!("human_score,{header}\n")).unwrap();
        let sidecar = std::path::PathBuf::from(format!("{}.manifest.json", path.display()));
        let mut metadata = serde_json::json!({
            "feature_set_id": "basic+peaks+v2@w1825/rev5_localwin#36c3f3af",
            "formula_revision": 5,
            "decoder_era": "fixture-executable-binding"
        });
        std::fs::write(&sidecar, metadata.to_string()).unwrap();
        let paths = vec![path];
        let ids = [0, 155, 156, 227, 372, 719];
        let admitted = admit_training_tables(&paths, None, Some(&ids), Some(1853)).unwrap();
        assert_eq!(admitted["qualified_provenance"], true);
        assert_eq!(admitted["formula_revision"], 5);
        assert_eq!(admitted["tables"][0]["inferred"], false);
        for absent in [228, 371, 720, 1824, 1852] {
            let err = admit_training_tables(&paths, None, Some(&[absent]), Some(1853)).unwrap_err();
            assert!(err.contains("not populated"), "{err}");
        }
        let replay =
            admit_training_tables(&paths, Some("diagnostic"), Some(&ids), Some(1853)).unwrap();
        assert_eq!(replay["qualified_provenance"], false);
        metadata["formula_revision"] = serde_json::json!(4);
        std::fs::write(&sidecar, metadata.to_string()).unwrap();
        assert!(
            admit_training_tables(&paths, None, Some(&ids), Some(1853))
                .unwrap_err()
                .contains("conflicts")
        );
        metadata["formula_revision"] = serde_json::json!(5);
        metadata.as_object_mut().unwrap().remove("decoder_era");
        std::fs::write(&sidecar, metadata.to_string()).unwrap();
        let incomplete = admit_training_tables(&paths, None, Some(&ids), Some(1853)).unwrap();
        assert_eq!(incomplete["qualified_provenance"], false);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn training_sampling_contract_cannot_be_mixed_even_in_replay() {
        let dir =
            std::env::temp_dir().join(format!("zensim-admit-sampling-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let paths = vec![dir.join("a.csv"), dir.join("b.csv")];
        let header = (0..372)
            .map(|i| format!("f{i}"))
            .collect::<Vec<_>>()
            .join(",");
        let identity = format!(
            "basic+peaks@w372/sampling_v1_xyb_triangle_2#{:08x}",
            zensim::feature_set_id::slots_hash8(0..228)
        );
        let metadata = serde_json::json!({"feature_set_id": identity, "formula_revision": 3,
            "decoder_era": "fixture-pinned", "sampling": "v1:xyb:triangle:2"});
        for path in &paths {
            std::fs::write(path, format!("{header}\n")).unwrap();
            std::fs::write(
                format!("{}.manifest.json", path.display()),
                metadata.to_string(),
            )
            .unwrap();
        }
        let admitted = admit_training_tables(&paths, None, Some(&[0, 227]), Some(372)).unwrap();
        assert_eq!(admitted["sampling"], "v1:xyb:triangle:2");
        for sampling in [
            serde_json::json!("v1:xyb:triangle:3"),
            serde_json::Value::Null,
        ] {
            let mut other = metadata.clone();
            other["sampling"] = sampling;
            std::fs::write(
                format!("{}.manifest.json", paths[1].display()),
                other.to_string(),
            )
            .unwrap();
            for replay in [None, Some("frozen test recipe")] {
                let err =
                    admit_training_tables(&paths, replay, Some(&[0, 227]), Some(372)).unwrap_err();
                assert!(err.contains("mixed sampling"), "{err}");
            }
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn real_headers_and_file_declarations_are_admitted_before_rows() {
        let dir = std::env::temp_dir().join(format!("zensim-admission-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("features.csv");
        let header = (0..956)
            .map(|i| format!("f{i}"))
            .collect::<Vec<_>>()
            .join(",");
        std::fs::write(&path, format!("human_score,{header}\n")).unwrap();
        let manifest = dir.join("_MANIFEST.json");
        std::fs::write(
            &manifest,
            serde_json::json!({
                "feature_set_id": "basic+peaks+masked+iw@w372/v1postc#d16a1091",
                "formula_revision": 1, "decoder_era": "fixture-pinned"
            })
            .to_string(),
        )
        .unwrap();
        let paths = vec![path.clone()];
        let admitted = admit_training_tables(&paths, None, Some(&[0, 371]), Some(956)).unwrap();
        assert_eq!(admitted["qualified_provenance"], true);
        assert_eq!(admitted["formula_revision"], 1);
        let absent = admit_training_tables(&paths, None, Some(&[950]), Some(956)).unwrap_err();
        assert!(absent.contains("not populated"), "{absent}");
        // Replay records semantic mismatch, but cannot fabricate a missing column.
        let replay =
            admit_training_tables(&paths, Some("frozen test recipe"), Some(&[950]), Some(956))
                .unwrap();
        assert_eq!(replay["qualified_provenance"], false);
        let absent =
            admit_training_tables(&paths, Some("frozen test recipe"), Some(&[956]), Some(957))
                .unwrap_err();
        assert!(absent.contains("required feature f956"), "{absent}");
        let sidecar = dir.join("features.csv.manifest.json");
        for v in [
            serde_json::json!({"formula_revision":2}),
            serde_json::json!({"decoder_era":"another-decoder"}),
            serde_json::json!({"feature_set_id":"malformed"}),
        ] {
            std::fs::write(&sidecar, v.to_string()).unwrap();
            assert!(
                admit_training_tables(&paths, Some("frozen test recipe"), None, Some(372)).is_err()
            );
        }
        std::fs::remove_file(&sidecar).unwrap();
        // Gapped CSV has the same refusal as parquet, before any body is read.
        std::fs::write(&path, "human_score,f0,f1,f8\n").unwrap();
        let err =
            admit_training_tables(&paths, Some("frozen test recipe"), None, Some(372)).unwrap_err();
        assert!(err.contains("DENSE-BY-ID"), "{err}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// **The admitted-revision list is ONE list and covers every registered revision** (1–7 since the
    /// by-construction revisions 6 and 7 joined `zensim::feature_defs::FormulaRevision`).
    ///
    /// Before this, the registry parse, the per-table metadata check and the
    /// root-manifest read each spelled `matches!(n, 1 | 2)` separately — which
    /// is exactly how a new revision comes to be accepted in one place and
    /// silently rejected in another. This pins that all three read the same
    /// list, and that an unregistered value is still refused rather than
    /// defaulted (a mislabelled table must fail loudly, not become revision 1).
    #[test]
    fn the_admitted_revision_list_covers_every_registered_revision() {
        assert_eq!(ADMITTED_FORMULA_REVISIONS, &[1, 2, 3, 4, 5, 6, 7]);
        for n in ADMITTED_FORMULA_REVISIONS {
            assert!(
                is_admitted_formula_revision(*n),
                "revision {n} is registered but not admitted"
            );
        }
        for n in [0u64, 8, 99] {
            assert!(
                !is_admitted_formula_revision(n),
                "revision {n} must not be admitted"
            );
        }

        // The root-manifest reader is one of the three sites; check it end to
        // end rather than trusting that it calls the helper.
        let dir = std::env::temp_dir().join(format!("zensim-admit-rev-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for (value, want_ok) in [
            ("3", true),
            ("2", true),
            ("4", true),
            ("5", true),
            ("7", true),
            ("8", false),
        ] {
            std::fs::write(
                dir.join("_MANIFEST.json"),
                format!("{{\"formula_revision\": {value}}}"),
            )
            .unwrap();
            let got = root_formula_revision(&dir);
            assert_eq!(
                got.is_ok(),
                want_ok,
                "formula_revision {value} admission: {got:?}"
            );
            if want_ok {
                assert_eq!(got.unwrap(), Some(value.parse::<u8>().unwrap()));
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn unknown_and_mixed_eras_require_explicit_replay() {
        let unknown = vec![std::path::PathBuf::from("/not-registered/features.parquet")];
        assert!(admit_training_tables(&unknown, None, None, None).is_err());
        let replay =
            admit_training_tables(&unknown, Some("reproduce frozen recipe"), None, None).unwrap();
        assert_eq!(replay["qualified_provenance"], false);
        assert!(replay["formula_revision"].is_null());
        let paths: Vec<_> = registry()
            .roots
            .keys()
            .filter(|p| {
                p.contains("2026-09-05-full-features-372-postC")
                    || p.contains("canonical-2026-05-21/train")
            })
            .map(|p| Path::new(p).join("probe.parquet"))
            .collect();
        assert_eq!(paths.len(), 2);
        assert!(admit_training_tables(&paths, None, None, None).is_err());
        assert!(
            admit_training_tables(&paths, Some("fixed historical control"), None, None).is_ok()
        );
    }
}

/// Revision of cached values, read from their manifest or the registered era.
/// The process's extraction setting cannot change an already-written table.
pub fn root_formula_revision(root: &Path) -> Result<Option<u8>, String> {
    if let Ok(bytes) = std::fs::read(root.join("_MANIFEST.json")) {
        let v: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", root.display()))?;
        if let Some(value) = v.get("formula_revision") {
            let n = value
                .as_u64()
                .or_else(|| value.as_str().and_then(|s| s.parse().ok()));
            return match n {
                Some(v) if is_admitted_formula_revision(v) => Ok(Some(v as u8)),
                _ => Err(format!(
                    "{}: invalid formula_revision {value}",
                    root.display()
                )),
            };
        }
    }
    Ok(root_feature_set_ref(root).and_then(|r| registry().formula_revision(r.id.era())))
}
