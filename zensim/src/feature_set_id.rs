//! **Feature-set identifiers** — the replacement for naming feature sets by a
//! count.
//!
//! Design + rationale + the alias table: [`docs/FEATURE_SET_IDS.md`]. Registry:
//! `benchmarks/feature_sets_registry.json`.
//!
//! A count answers ONE question (how wide is the vector?) and has repeatedly
//! been read as three. "944" has named six different feature sets in this repo
//! — two extractor eras of the same compute set, an all-pools variant whose
//! `f156..371` is LIVE where the others are structural zeros, a carriers
//! variant, and two free-set arms that share only `f0..227` with any of them.
//! "156+free" names 156 basic + 109 free slots inside a 944-wide vector.
//! `--regime 944` silently mis-scores a 372 bake that reads `f156..371`
//! (shipped `B`: CID22 0.3862 against its true 0.8764).
//!
//! An id has three parts, plus an OPTIONAL fourth:
//!
//! ```text
//!   basic+peaks+moments/era2r4#a1b2c3d4          <- canonical, layout-free
//!   basic+peaks+moments@w944/era2r4#a1b2c3d4     <- legacy alias, still parses
//!   └─COMPUTE──────────┘ └LAYOUT┘ └ERA─┘ └hash8┘
//! ```
//!
//! * **COMPUTE** — which slot families are actually populated
//!   ([`ComputeToken`]), `+`-joined in registry order.
//! * **ERA** — the registered extractor-era token. Load-bearing because the
//!   shift between eras is model-specific, not a constant offset
//!   (`benchmarks/eval372_current_root_2026-08-30.md`).
//! * **`#hash8`** — 8 lowercase hex of [`slots_hash8`] over the sorted,
//!   de-duplicated slot-id list. The NAME is a handle; the HASH is the
//!   identity, so two sets with the same name and different slot lists cannot
//!   collide.
//! * **LAYOUT** (optional, `@w<N>`) — the emitted vector width. **Not part of
//!   the identity**: two ids that differ only here are EQUAL, hash equal, and
//!   name the same set.
//!
//! ## Why the layout left the identity (2026-09-06)
//!
//! The layout is the WIRE SHAPE, not the feature set, and with a dense wire it
//! carries nothing the hash does not. It also lied inside a refusal surface:
//! the cruft purge measured a densified shipped `B` and its wide twin as
//! `basic+peaks+masked+iw@w95/unknown#9403d2a7` and
//! `…@w372/unknown#9403d2a7` — the same compute tokens and the SAME slots hash
//! — so `feature_set::check` reported a `LayoutDiffers` mismatch on every
//! dense-bake/wide-table pair, for a difference that cannot make a read
//! unsound. The width now belongs to the ARTIFACT
//! (`zensim_validate::feature_set::FeatureSetRef::layout`), where a shortfall
//! is a real finding.
//!
//! It is still ACCEPTED, and every `@w<N>` string ever written still parses,
//! because it remains a **reconstruction aid**: [`crate::feature_layout`]
//! rebuilds a registered set from its compute tokens, and a SPARSE id's set is
//! the family union CLIPPED to its width, so knowing the width turns a search
//! over candidate widths into a single check. Identity is the hash; the width
//! is a hint about how to reproduce it.
//!
//! ## Substrate: `zenanalyze-api`, reused not paralleled
//!
//! The FROZEN `zenanalyze-api` contract already owns per-feature identity as
//! `name@hex8` with charset `[a-z0-9_]+`, 8 strictly-lowercase hex digits, and
//! a mandatory 64→32 fold `(h >> 32) ^ (h & 0xffff_ffff)`
//! (`NamedFeature::fold_hash`). All four decisions are reused verbatim here.
//! A feature-set id is a COARSER handle than a `NamedFeature` — a whole vector
//! at a width and an era the API crate deliberately does not model — so it
//! lives here and the API crate is untouched.

use core::fmt;

/// One registered COMPUTE token: a slot family whose presence changes which
/// slots are POPULATED (not merely their values).
///
/// Sub-toggles that reshape a family from the inside (`gradient`,
/// `blockiness`, `transducer_bank`, `append2_dst_activity`) are deliberately
/// absent: they move the populated set, so they move the hash, so they cannot
/// collide — and keeping them out of the short form is what keeps a 944 name
/// readable. See `docs/FEATURE_SET_IDS.md` §2.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum ComputeToken {
    /// v1 basic fold, `f0..155`.
    Basic,
    /// The ten `fused944native` carrier slots (`V1PoolsMode::Carriers`).
    Carriers,
    /// v1 soft-peak pool, `f156..227` (`V1PoolsMode::Peaks`).
    Peaks,
    /// v1 masked pool, `f228..299`.
    Masked,
    /// v1 IW pool, `f300..371`.
    Iw,
    /// v2-348 dense block, `f372..719`.
    V2,
    /// append-204, `f720..923`.
    Append,
    /// append2 / BANDVIS, `f924..943`.
    Append2,
    /// CSFW tier-1, `f944..955`.
    Csfw,
    /// The free raw-moment tranche (`V1FreeExtras::RawMoments`) — 37 scattered
    /// slots a v1-only walk finalizes without the v2-era passes.
    Moments,
    /// The class-C bounded-error tranche
    /// (`V1FreeExtras::RawMomentsPlusBoundedErr`) — 24 scattered slots.
    ClassC,
    /// RESERVED for the future HDR block. Append-only; its enum position
    Hdr,
    /// Rev4 grid-aligned blocking phase profile, `f986..1081`.
    ///
    Gridblk = 13,
    /// Rev4 ringing-magnitude basis histogram, `f1082..1153`.
    Ringbasis,
    /// Rev4 tail histograms of the dense per-pixel maps, `f1154..1297`.
    Tailhist,
    /// Rev4 artifact-type descriptors, `f1298..1321`.
    Arttype,
    /// Experimental GMSBANK gradient similarity basis, `f1322..1501`.
    Gmsbank,
    /// Restored cut (COST_CUTS_AUDIT A1): per-scale, per-channel deviation
    /// (std) of the MSE and HF-energy/magnitude maps, `f1502..1561`.
    Mapdev,
    /// Restored cut (COST_CUTS_AUDIT B2): the 228-slot basic+peaks surface
    /// pooled over ungated 5x5 block maxima instead of pixels, `f1562..1789`.
    Z1max,
    /// Restored cut (COST_CUTS_AUDIT Ambiguous 7): C8's X/B gradient
    /// loss/gain/deviation bank at native scale, `f1790..1819`.
    Gmsnative,
    /// Global affine-change descriptor, `f1867..1890` (formula revision 6):
    /// per scale and XYB channel, the share of the mean squared error a
    /// least-squares fit `dst ≈ a·src + b` over the whole plane explains, and
    /// that explained error.
    GlobAff,
}

impl ComputeToken {
    /// Every token, in the canonical registry order the short form renders in.
    pub const ALL: &'static [ComputeToken] = &[
        ComputeToken::Basic,
        ComputeToken::Carriers,
        ComputeToken::Peaks,
        ComputeToken::Masked,
        ComputeToken::Iw,
        ComputeToken::V2,
        ComputeToken::Append,
        ComputeToken::Append2,
        ComputeToken::Csfw,
        ComputeToken::Gridblk,
        ComputeToken::Ringbasis,
        ComputeToken::Tailhist,
        ComputeToken::Arttype,
        ComputeToken::Gmsbank,
        ComputeToken::Mapdev,
        ComputeToken::Z1max,
        ComputeToken::Gmsnative,
        ComputeToken::GlobAff,
        ComputeToken::Moments,
        ComputeToken::ClassC,
        ComputeToken::Hdr,
    ];

    /// The token's wire spelling (charset `[a-z0-9_]+`, per the
    /// `zenanalyze-api` name rule).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            ComputeToken::Basic => "basic",
            ComputeToken::Carriers => "carriers",
            ComputeToken::Peaks => "peaks",
            ComputeToken::Masked => "masked",
            ComputeToken::Iw => "iw",
            ComputeToken::V2 => "v2",
            ComputeToken::Append => "append",
            ComputeToken::Append2 => "append2",
            ComputeToken::Csfw => "csfw",
            ComputeToken::Gridblk => "gridblk",
            ComputeToken::Ringbasis => "ringbasis",
            ComputeToken::Tailhist => "tailhist",
            ComputeToken::Arttype => "arttype",
            ComputeToken::Gmsbank => "gmsbank",
            ComputeToken::Mapdev => "mapdev",
            ComputeToken::Z1max => "z1max",
            ComputeToken::Gmsnative => "gmsnative",
            ComputeToken::GlobAff => "globaff",
            ComputeToken::Moments => "moments",
            ComputeToken::ClassC => "classc",
            ComputeToken::Hdr => "hdr",
        }
    }

    /// Parse one token. Unknown spellings are `None` — the vocabulary is
    /// CLOSED, so a typo is an error rather than a new set.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|t| t.as_str() == s)
    }

    const fn bit(self) -> u32 {
        1u32 << (self as u32)
    }
}

impl fmt::Display for ComputeToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A set of [`ComputeToken`]s. Renders in registry order regardless of
/// insertion order; the empty set renders `none`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash, PartialOrd, Ord)]
pub struct ComputeParts(u32);

impl ComputeParts {
    /// No families at all.
    pub const EMPTY: Self = ComputeParts(0);

    /// This set plus `t`.
    #[must_use]
    pub const fn with(self, t: ComputeToken) -> Self {
        ComputeParts(self.0 | t.bit())
    }

    /// Is `t` in the set?
    #[must_use]
    pub const fn contains(self, t: ComputeToken) -> bool {
        self.0 & t.bit() != 0
    }

    // Private research wire bit: no supported ComputeToken variant or iterator
    // item. Existing public token bits/order remain exactly unchanged.
    pub(crate) const fn with_palette(self) -> Self {
        Self(self.0 | (1 << 22))
    }

    pub(crate) const fn has_palette(self) -> bool {
        self.0 & (1 << 22) != 0
    }

    /// Is the set empty?
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The tokens, in registry order.
    pub fn iter(self) -> impl Iterator<Item = ComputeToken> {
        ComputeToken::ALL
            .iter()
            .copied()
            .filter(move |t| self.contains(*t))
    }

    /// Parse a `+`-joined token list (order-tolerant), or `none`.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        if s == "none" {
            return Some(Self::EMPTY);
        }
        if s.is_empty() {
            return None;
        }
        let mut out = Self::EMPTY;
        for part in s.split('+') {
            out = if part == "palette" {
                out.with_palette()
            } else {
                out.with(ComputeToken::parse(part)?)
            };
        }
        Some(out)
    }
}

impl fmt::Display for ComputeParts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("none");
        }
        let mut first = true;
        for t in ComputeToken::ALL.iter().copied() {
            if t == ComputeToken::Moments && self.has_palette() {
                if !first {
                    f.write_str("+")?;
                }
                f.write_str("palette")?;
                first = false;
            }
            if !self.contains(t) {
                continue;
            }
            if !first {
                f.write_str("+")?;
            }
            first = false;
            f.write_str(t.as_str())?;
        }
        Ok(())
    }
}

/// A canonical set of slot indices: sorted, de-duplicated, stored as merged
/// half-open ranges and rendered compactly (`"0-155,720-923"`, a lone slot as
/// `"42"`, the empty set as `""`).
///
/// This is the machine half of a feature-set id — the thing the `#hash8`
/// hashes and the thing [`SlotSet::covers`] compares. A PRODUCER's slot set is
/// what it populates; a CONSUMER's (a bake's) is what it reads.
#[derive(Debug, Clone, PartialEq, Eq, Default, Hash)]
pub struct SlotSet {
    /// Sorted, disjoint, merged, half-open `[start, end)`.
    ranges: Vec<(usize, usize)>,
}

impl SlotSet {
    /// Canonicalise an arbitrary slot iterator (any order, duplicates fine).
    #[must_use]
    pub fn from_slots(slots: impl IntoIterator<Item = usize>) -> Self {
        let mut v: Vec<usize> = slots.into_iter().collect();
        v.sort_unstable();
        v.dedup();
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for s in v {
            match ranges.last_mut() {
                Some(last) if last.1 == s => last.1 = s + 1,
                _ => ranges.push((s, s + 1)),
            }
        }
        Self { ranges }
    }

    /// Canonicalise a half-open range list (any order, overlaps fine).
    #[must_use]
    pub fn from_ranges(ranges: impl IntoIterator<Item = (usize, usize)>) -> Self {
        let mut v: Vec<(usize, usize)> = ranges.into_iter().filter(|(a, b)| a < b).collect();
        v.sort_unstable();
        let mut out: Vec<(usize, usize)> = Vec::new();
        for (a, b) in v {
            match out.last_mut() {
                Some(last) if a <= last.1 => last.1 = last.1.max(b),
                _ => out.push((a, b)),
            }
        }
        Self { ranges: out }
    }

    /// Number of slots in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ranges.iter().map(|(a, b)| b - a).sum()
    }

    /// Is the set empty?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Is `slot` in the set?
    #[must_use]
    pub fn contains(&self, slot: usize) -> bool {
        self.ranges.iter().any(|(a, b)| slot >= *a && slot < *b)
    }

    /// Every slot, ascending.
    pub fn iter_slots(&self) -> impl Iterator<Item = usize> + '_ {
        self.ranges.iter().flat_map(|(a, b)| *a..*b)
    }

    /// The merged half-open ranges.
    #[must_use]
    pub fn ranges(&self) -> &[(usize, usize)] {
        &self.ranges
    }

    /// Does this set contain every slot of `other`? **The compatibility test:**
    /// a read is sound only when the producer covers the consumer.
    #[must_use]
    pub fn covers(&self, other: &SlotSet) -> bool {
        other.iter_slots().all(|s| self.contains(s))
    }

    /// The slots of `other` this set does NOT contain — the actionable half of
    /// a failed [`covers`](Self::covers).
    #[must_use]
    pub fn missing_from(&self, other: &SlotSet) -> SlotSet {
        SlotSet::from_slots(other.iter_slots().filter(|s| !self.contains(*s)))
    }

    /// Intersection.
    #[must_use]
    pub fn intersect(&self, other: &SlotSet) -> SlotSet {
        SlotSet::from_slots(self.iter_slots().filter(|s| other.contains(*s)))
    }

    /// Union.
    #[must_use]
    pub fn union(&self, other: &SlotSet) -> SlotSet {
        SlotSet::from_ranges(
            self.ranges
                .iter()
                .copied()
                .chain(other.ranges.iter().copied()),
        )
    }

    /// Everything below `width` — the LAYOUT clip. A family that does not
    /// exist at this width cannot be populated at it.
    #[must_use]
    pub fn clipped_to(&self, width: usize) -> SlotSet {
        SlotSet::from_ranges(
            self.ranges
                .iter()
                .filter(|(a, _)| *a < width)
                .map(|(a, b)| (*a, (*b).min(width))),
        )
    }

    /// This set's content hash — [`slots_hash8`], the ONE owner.
    #[must_use]
    pub fn hash8(&self) -> u32 {
        slots_hash8(self.iter_slots())
    }

    /// Parse the compact form (`"0-155,720-923"`, `"42"`, `""`). Rejects
    /// descending ranges; tolerates unsorted / overlapping input by
    /// canonicalising.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        if s.is_empty() {
            return Some(Self::default());
        }
        let mut ranges = Vec::new();
        for part in s.split(',') {
            let part = part.trim();
            match part.split_once('-') {
                Some((a, b)) => {
                    let (a, b) = (
                        a.trim().parse::<usize>().ok()?,
                        b.trim().parse::<usize>().ok()?,
                    );
                    if b < a {
                        return None;
                    }
                    ranges.push((a, b + 1));
                }
                None => {
                    let a = part.parse::<usize>().ok()?;
                    ranges.push((a, a + 1));
                }
            }
        }
        Some(Self::from_ranges(ranges))
    }
}

impl fmt::Display for SlotSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for (a, b) in &self.ranges {
            if !first {
                f.write_str(",")?;
            }
            first = false;
            if b - a == 1 {
                write!(f, "{a}")?;
            } else {
                write!(f, "{}-{}", a, b - 1)?;
            }
        }
        Ok(())
    }
}

/// **THE hash owner.** 32-bit content hash of a slot set.
///
/// FNV-1a/64 over the canonical rendering of the SORTED, DE-DUPLICATED slot
/// list (decimal, comma-separated), folded to 32 bits by the exact rule
/// `zenanalyze_api::NamedFeature::fold_hash` mandates:
/// `(h >> 32) ^ (h & 0xffff_ffff)`.
///
/// Set semantics: order and duplicates in the input cannot move the hash.
/// Ordering *within* the emitted vector is the LAYOUT's job, not this one's.
///
/// Every producer and consumer calls this. Nothing re-derives it — a silent
/// hash mismatch is worse than no hash (the same discipline `zenanalyze`
/// applies to `feature_qualified_names`, which ships a committed TSV rather
/// than letting Python recompute the fold).
#[must_use]
pub fn slots_hash8(slots: impl IntoIterator<Item = usize>) -> u32 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut v: Vec<usize> = slots.into_iter().collect();
    v.sort_unstable();
    v.dedup();
    let mut h = FNV_OFFSET;
    let mut first = true;
    for s in v {
        if !first {
            h ^= u64::from(b',');
            h = h.wrapping_mul(FNV_PRIME);
        }
        first = false;
        for byte in itoa_bytes(s) {
            h ^= u64::from(byte);
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    // The mandated 64→32 fold — identical to `NamedFeature::fold_hash`.
    ((h >> 32) ^ (h & 0xffff_ffff)) as u32
}

/// Decimal bytes of `n`, most-significant first, without allocating a String.
fn itoa_bytes(n: usize) -> impl Iterator<Item = u8> {
    let mut buf = [0u8; 20];
    let mut i = buf.len();
    let mut n = n;
    loop {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
        if n == 0 {
            break;
        }
    }
    (i..buf.len()).map(move |k| buf[k])
}

/// Render a 32-bit hash as the 8-lowercase-hex form the id carries.
#[must_use]
pub fn hex8(h: u32) -> String {
    format!("{h:08x}")
}

/// Is `s` a valid COMPUTE/ERA token spelling? Charset `[a-z0-9_]+`, exactly
/// `zenanalyze_api::NamedFeature::is_valid_name`.
#[must_use]
pub fn is_valid_token(s: &str) -> bool {
    !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
}

/// A feature-set identifier: `<compute>/<era>#<hash8>`, optionally carrying a
/// legacy `@w<layout>` width.
///
/// **Equality is over compute + era + hash, and NOT over the layout.** Same
/// slots at a different era is a DIFFERENT id, which is the whole point (the
/// `ext944` / `era2r4` pair is bit-identical except `f720..923`, and a
/// published number cannot be corrected across the boundary, only
/// re-verdicted). Same slots at a different WIRE WIDTH is the SAME id, which
/// is what makes every `@w<N>` spelling an alias of its layout-free form
/// rather than a different set — see the module docs for the measurement that
/// forced this.
#[derive(Debug, Clone, Eq)]
pub struct FeatureSetId {
    compute: ComputeParts,
    /// The legacy wire width, when the id string carried one. A
    /// RECONSTRUCTION AID, never part of the identity: excluded from
    /// [`PartialEq`] and [`Hash`] by hand below, so an id parsed from
    /// `…@w372/…` and the same id parsed from `…/…` are interchangeable in
    /// every set, map and comparison.
    layout_width: Option<usize>,
    era: String,
    slots_hash: u32,
}

impl PartialEq for FeatureSetId {
    fn eq(&self, other: &Self) -> bool {
        self.compute == other.compute
            && self.era == other.era
            && self.slots_hash == other.slots_hash
    }
}

impl core::hash::Hash for FeatureSetId {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.compute.hash(state);
        self.era.hash(state);
        self.slots_hash.hash(state);
    }
}

impl FeatureSetId {
    /// Build the canonical, LAYOUT-FREE id from an already-computed slot hash.
    /// `None` when `era` is not a valid token.
    #[must_use]
    pub fn new(compute: ComputeParts, era: &str, slots_hash: u32) -> Option<Self> {
        is_valid_token(era).then(|| Self {
            compute,
            layout_width: None,
            era: era.to_string(),
            slots_hash,
        })
    }

    /// [`new`](Self::new) carrying the legacy `@w<layout>` width — a
    /// reconstruction aid, never part of the identity. Use it when you KNOW
    /// the wire width and want a reader to be able to rebuild the set without
    /// searching candidate widths.
    #[must_use]
    pub fn new_with_layout(
        compute: ComputeParts,
        layout_width: usize,
        era: &str,
        slots_hash: u32,
    ) -> Option<Self> {
        Self::new(compute, era, slots_hash).map(|id| id.with_layout(layout_width))
    }

    /// This id carrying `layout_width`. Changes the rendering, never the
    /// identity: the result compares and hashes EQUAL to `self`.
    #[must_use]
    pub fn with_layout(mut self, layout_width: usize) -> Self {
        self.layout_width = Some(layout_width);
        self
    }

    /// This id without its layout — the canonical form. Compares and hashes
    /// EQUAL to `self`; the difference is only what [`Display`] writes.
    #[must_use]
    pub fn layout_free(mut self) -> Self {
        self.layout_width = None;
        self
    }

    /// Build from the slot set itself — hashes through [`slots_hash8`].
    #[must_use]
    pub fn from_slots(compute: ComputeParts, era: &str, slots: &SlotSet) -> Option<Self> {
        Self::new(compute, era, slots.hash8())
    }

    /// [`from_slots`](Self::from_slots) carrying the legacy width.
    #[must_use]
    pub fn from_slots_with_layout(
        compute: ComputeParts,
        layout_width: usize,
        era: &str,
        slots: &SlotSet,
    ) -> Option<Self> {
        Self::from_slots(compute, era, slots).map(|id| id.with_layout(layout_width))
    }

    /// The COMPUTE token set.
    #[must_use]
    pub fn compute(&self) -> ComputeParts {
        self.compute
    }
    /// The legacy emitted-vector width, when the id string carried one.
    ///
    /// `None` for the canonical form. **Never a tie-breaker in a comparison**
    /// — it is excluded from equality and hashing on purpose; it exists so a
    /// reader can rebuild a SPARSE set (the family union clipped to this
    /// width) without searching.
    #[must_use]
    pub fn layout_width(&self) -> Option<usize> {
        self.layout_width
    }
    /// The registered extractor-era token.
    #[must_use]
    pub fn era(&self) -> &str {
        &self.era
    }
    /// The slot-set content hash.
    #[must_use]
    pub fn slots_hash(&self) -> u32 {
        self.slots_hash
    }

    /// Same compute + slots, ignoring the era — the "is this the same feature
    /// set, made by a possibly-different extractor?" question. The layout was
    /// dropped from this comparison with the rest of the identity: a set does
    /// not stop being itself because it was written into a narrower vector.
    #[must_use]
    pub fn same_set_ignoring_era(&self, other: &Self) -> bool {
        self.compute == other.compute && self.slots_hash == other.slots_hash
    }

    /// Parse either form. Strict, exactly like
    /// `zenanalyze_api::NamedFeature::parse`: 8 LOWERCASE hex digits, a closed
    /// token vocabulary, `[a-z0-9_]+` eras — and, when the legacy `@w<N>`
    /// component is present, a mandatory `w` on the width.
    ///
    /// **Both spellings of the same set parse to EQUAL ids**, which is what
    /// makes every `@w<N>` string ever written an alias rather than a
    /// different set.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        // The `@w<N>` component is optional. Split on the FIRST `/` after the
        // compute part: everything before it is `<compute>` or
        // `<compute>@w<N>`.
        let (head, rest) = s.split_once('/')?;
        let (compute_s, layout_width) = match head.split_once('@') {
            Some((c, layout_s)) => (c, Some(layout_s.strip_prefix('w')?.parse::<usize>().ok()?)),
            None => (head, None),
        };
        let (era, hash_s) = rest.split_once('#')?;
        let compute = ComputeParts::parse(compute_s)?;
        if hash_s.len() != 8
            || !hash_s
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return None;
        }
        let slots_hash = u32::from_str_radix(hash_s, 16).ok()?;
        let id = Self::new(compute, era, slots_hash)?;
        Some(match layout_width {
            Some(w) => id.with_layout(w),
            None => id,
        })
    }
}

impl fmt::Display for FeatureSetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.layout_width {
            Some(w) => write!(
                f,
                "{}@w{}/{}#{:08x}",
                self.compute, w, self.era, self.slots_hash
            ),
            None => write!(f, "{}/{}#{:08x}", self.compute, self.era, self.slots_hash),
        }
    }
}

/// Every emitted-vector width a registered PRODUCER set has used — the
/// candidate clip widths a reader searches when an id carries no legacy
/// `@w<N>` component.
///
/// **This bounds a search; it is never an identity.** A width here can only
/// CONFIRM a reconstruction (the `slots_hash8` decides), so adding one cannot
/// change which set any existing id names. `#[doc(hidden)]` and not the
/// supported surface: it exists so `zensim-validate`'s
/// `every_registered_layout_width_is_a_candidate` can hold this list and
/// `benchmarks/feature_sets_registry.json` in sync — registering a set at a
/// new width fails that gate instead of silently becoming unreproducible.
#[doc(hidden)]
#[must_use]
pub fn registered_layout_widths() -> &'static [usize] {
    crate::feature_defs::REGISTERED_LAYOUT_WIDTHS
}

/// The canonical era token for "we do not know which extractor made this".
///
/// Never silently treated as a match: "we do not know" and "we know it was a
/// different one" have the same consequence for a published number.
pub const ERA_UNKNOWN: &str = "unknown";
