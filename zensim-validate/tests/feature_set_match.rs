//! `zensim_validate::feature_set` — deriving a feature-set id for a BAKE and
//! for a FEATURES ROOT, and the compatibility check that closes the
//! `--regime 944` silent-mis-scoring bug at the root.
//!
//! Design: `docs/FEATURE_SET_IDS.md`. Registry:
//! `benchmarks/feature_sets_registry.json`.
//!
//! The class under test has two published instances: `ebothg_m504` (a
//! wrong-root board row) and appendix U.R0 — shipped **B**, a 372-input bake
//! with 49 structurally-used lines in `f156..371`, scored at `--regime 944`
//! where that block is STRUCTURAL ZEROS, reading CID22 **0.3862** against its
//! true **0.8764**. No error, no warning, a plausible number half a point low.

use std::path::Path;
use zenpredict::{Activation, Model, WeightDtype};
use zenpredict_bake::{BakeLayer, BakeRequest, bake};
use zensim::feature_set_id::{ComputeToken as T, SlotSet};
use zensim_validate::feature_set;

const OUT: usize = 2;
const W1: [f32; OUT] = [0.7, -0.3];
const B1: [f32; 1] = [0.1];

/// Single hidden layer + identity head over `n_raw` caller lines; layer-0
/// weight on caller line `k` iff `live(k)`.
fn build_model(n_raw: usize, live: impl Fn(usize) -> bool) -> Model {
    let mut w0 = vec![0.0f32; n_raw * OUT];
    for k in 0..n_raw {
        if live(k) {
            for o in 0..OUT {
                w0[k * OUT + o] = 0.05 + ((k * 31 + o * 7) % 97) as f32 / 100.0;
            }
        }
    }
    let b0 = vec![0.0f32; OUT];
    let mean = vec![0.0f32; n_raw];
    let scale = vec![1.0f32; n_raw];
    let layers = [
        BakeLayer {
            in_dim: n_raw,
            out_dim: OUT,
            activation: Activation::LeakyRelu,
            dtype: WeightDtype::F32,
            weights: &w0,
            biases: &b0,
        },
        BakeLayer {
            in_dim: OUT,
            out_dim: 1,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &W1,
            biases: &B1,
        },
    ];
    let bytes = bake(&BakeRequest {
        schema_hash: 0x5eed_0000_c105_0001,
        flags: 0,
        scaler_mean: &mean,
        scaler_scale: &scale,
        layers: &layers,
        feature_bounds: &[],
        metadata: &[],
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
        feature_order: None,
        output_order: None,
        compressed: false,
        hu_permutations: None,
    })
    .expect("bake");
    Model::from_bytes(&bytes).expect("parse")
}

/// **THE bug.** A 372-wide bake that structurally reads `f156..371` (shipped
/// B's shape: 49 live lines in that block), checked against the `ext944` root
/// whose f156..371 is structural zeros.
#[test]
fn a_372_bake_reading_f156_371_mismatches_the_944_root() {
    let m = build_model(372, |k| k < 156 || (300..349).contains(&k));
    let bake_ref = feature_set::bake_feature_set_ref(&m, "v1cur").expect("derive");
    assert!(
        bake_ref.id.compute().contains(T::Iw),
        "the bake reads the IW block"
    );
    let root = feature_set::registry()
        .set("basic+v2+append+append2@w944/ext944")
        .expect("registered")
        .as_ref()
        .expect("a pinned set has slots");

    let ms = feature_set::check(&bake_ref, root);
    assert!(!ms.is_empty(), "reading a zeroed block MUST be reported");
    let joined = ms
        .iter()
        .map(|m| m.to_string())
        .collect::<Vec<_>>()
        .join(" | ");
    assert!(
        joined.contains("300-348"),
        "must name the unpopulated slots: {joined}"
    );
    assert!(
        ms.iter()
            .any(|m| m.kind == feature_set::MismatchKind::SlotsNotPopulated),
        "the slot-coverage mismatch is the load-bearing one: {joined}"
    );
}

/// A basic-only reader (ADD156's shape) is safe at every root that populates
/// `f0..155` — the T.R4 bridge cell measured ≤0.0008 cross-root drift for
/// exactly this shape.
#[test]
fn basic_only_bake_compatibility_respects_partial_producers() {
    let m = build_model(372, |k| k < 28);
    let bake_ref = feature_set::bake_feature_set_ref(&m, "v1cur").expect("derive");
    assert_eq!(bake_ref.id.compute().to_string(), "basic");
    for (id, set) in feature_set::registry().sets() {
        let Some(p) = set.as_ref() else { continue };
        if p.layout.is_none_or(|w| w < 372) {
            continue;
        }
        let slot_fails = feature_set::check(&bake_ref, p)
            .into_iter()
            .filter(|m| m.kind == feature_set::MismatchKind::SlotsNotPopulated)
            .count();
        // September 13 sampling producers intentionally omit fine X/B, and the
        // SHIPPATH7 by_v2fy projection roots (`slot_selection: by_v2fy_420`, at
        // revision 5 and the by-construction revision-7 re-extraction) populate
        // only the 420 by_v2fy IDs (basic 0-12 and 26-38 are NaN).
        // The research-only palette family (`palette@...`) populates only its
        // own f1825-f1866 slots.
        // Width 372 does not authorize reading those unpopulated basic slots:
        // the check must report the missing coverage for these producers.
        let expected = usize::from(
            id.contains("/sampling_v1_y_")
                || id.contains("/rev5_localwin#62adfc93")
                || id.contains("/rev7_rowlanes#62adfc93")
                || id.starts_with("palette@"),
        );
        assert_eq!(slot_fails, expected, "basic slot coverage for {id}");
    }
}

/// Same slots, different era ⇒ still reported, because the values differ even
/// though the layout and the populated set do not.
#[test]
fn an_era_difference_is_reported_even_when_the_slots_match() {
    let m = build_model(372, |k| k < 156);
    let a = feature_set::bake_feature_set_ref(&m, "v1cur").expect("derive");
    let root = feature_set::registry()
        .set("basic+peaks+masked+iw@w372/v1pre")
        .expect("registered")
        .as_ref()
        .expect("pinned");
    let ms = feature_set::check(&a, root);
    assert!(
        ms.iter()
            .any(|m| m.kind == feature_set::MismatchKind::EraDiffers)
    );
    assert!(
        !ms.iter()
            .any(|m| m.kind == feature_set::MismatchKind::SlotsNotPopulated)
    );
}

/// An UNKNOWN era is never silently a match.
#[test]
fn an_unknown_era_is_reported_not_waved_through() {
    let m = build_model(372, |k| k < 156);
    let a =
        feature_set::bake_feature_set_ref(&m, zensim::feature_set_id::ERA_UNKNOWN).expect("derive");
    let root = feature_set::registry()
        .set("basic+peaks+masked+iw@w372/v1cur")
        .expect("registered")
        .as_ref()
        .expect("pinned");
    assert!(
        feature_set::check(&a, root)
            .iter()
            .any(|m| m.kind == feature_set::MismatchKind::EraUnknown)
    );
}

/// The committed registry is checked against the OWNER: every pinned set's
/// stored `slots_hash8` must equal `zensim::feature_set_id::slots_hash8` over
/// its own `slots`, and its canonical id string must render back to its key.
#[test]
fn every_registered_set_agrees_with_the_hash_owner() {
    let reg = feature_set::registry();
    let mut pinned = 0usize;
    for (key, set) in reg.sets() {
        // Registry keys are APPEND-ONLY and were all written in the legacy
        // `@w<N>` spelling; the canonical id form dropped that component on
        // 2026-09-06. A key must render back as one of the two, and which one
        // is a property of when it was written, not of what it names.
        let key_class = key.split('#').next().unwrap();
        let legacy = set.id.to_string_class_legacy();
        assert!(
            key_class == set.id.to_string_class() || legacy.as_deref() == Some(key_class),
            "the registry key must be the canonical id or its legacy \
             `@w<N>` spelling (modulo the hash): key {key_class:?}, canonical {:?}, \
             legacy {legacy:?}",
            set.id.to_string_class()
        );
        if let Some(p) = set.as_ref() {
            pinned += 1;
            assert_eq!(
                p.slots.hash8(),
                p.id.slots_hash(),
                "{key}: stored hash disagrees with the owner"
            );
            assert_eq!(
                p.slots.len(),
                set.n_slots.expect("pinned sets carry n_slots")
            );
            assert!(
                p.layout
                    .is_some_and(|w| w >= p.slots.ranges().last().map(|r| r.1).unwrap_or(0)),
                "{key}: slots overflow the declared layout"
            );
        }
    }
    assert!(
        pinned >= 12,
        "the registry must pin every set of the doc's §3 table"
    );
    // Every alias target resolves.
    for (name, ids) in reg.aliases() {
        for id in ids {
            assert!(reg.set(id).is_some(), "alias {name} -> unregistered {id}");
        }
    }
    // "944" is the ambiguous one, by construction.
    assert!(
        reg.aliases_for("944").len() >= 6,
        "944 has had at least six meanings"
    );
    assert_eq!(reg.aliases_for("720").len(), 1);
}

/// A features root resolves through its `_MANIFEST.json` `feature_set_id` key
/// when it has one, and through the registry's root/regime tables when it does
/// not — the latter marked `inferred`.
#[test]
fn a_registered_root_path_resolves_and_is_marked_inferred() {
    let p = Path::new("/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01");
    let Some(r) = feature_set::root_feature_set_ref(p) else {
        // The root is on block storage; skip only when the REGISTRY itself
        // cannot answer, which would be a registry defect, not a missing disk.
        panic!("the registry alone must resolve a registered root path");
    };
    assert_eq!(
        r.id.to_string(),
        "basic+v2+append+append2@w944/ext944#7ed470b4"
    );
    assert!(
        r.inferred,
        "a root with no stored id is INFERRED, never asserted"
    );
    assert!(
        !r.slots.contains(300),
        "f156..371 is not populated at ext944"
    );
    assert!(r.slots.covers(&SlotSet::parse("0-155").unwrap()));
}

/// A root whose `_MANIFEST.json` DECLARES a `feature_set_id` resolves to the
/// registry's slot set — not to an empty one.
///
/// FAILS BEFORE the 2026-09-05 fix. `root_feature_set_ref` looked the declared
/// id up with `reg.set(&id.to_string())`, i.e. the FULL id including `#hash8`,
/// while every registry key is the CLASS form with the hash in the value. So
/// the branch the design calls "most authoritative" always fell through to an
/// EMPTY slot set, and `check` then reported every slot the consumer reads as
/// `SlotsNotPopulated`. Measured on the first root to carry the key: 28
/// spurious "not populated" slots for a 372-input bake against a 372-wide root
/// that populates all 372.
///
/// Written against a manifest this test WRITES, so it exercises the resolution
/// path itself and needs nothing from block storage.
#[test]
fn a_manifest_declared_feature_set_id_resolves_to_the_registered_slots() {
    // CARGO_TARGET_TMPDIR, not `std::env::temp_dir()` — `/tmp` is banned as
    // scratch in this workspace (CLAUDE.md), and this keeps the fixture under
    // the build directory where it is cleaned with everything else.
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("zv_fsid_declared_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("_MANIFEST.json"),
        r#"{"regime": "not a registered regime string",
            "feature_set_id": "basic+peaks+masked+iw@w372/v1cur#d16a1091"}"#,
    )
    .unwrap();
    let r = feature_set::root_feature_set_ref(&dir).expect("declared id must resolve");
    assert!(
        !r.inferred,
        "a root that DECLARES its id is asserted, never inferred"
    );
    assert_eq!(
        r.id.to_string(),
        "basic+peaks+masked+iw@w372/v1cur#d16a1091"
    );
    assert_eq!(
        r.slots,
        SlotSet::parse("0-371").unwrap(),
        "the declared id must carry the REGISTERED slot set, not an empty one"
    );
    // The concrete regression: a 372-wide basic-only consumer must be covered.
    assert!(r.slots.covers(&SlotSet::parse("0-155").unwrap()));

    // NEGATIVE CONTROL: a declared id whose hash disagrees with the registry
    // must NOT borrow the class entry's slots — the name and the bytes have
    // come apart and that has to stay visible.
    std::fs::write(
        dir.join("_MANIFEST.json"),
        r#"{"feature_set_id": "basic+peaks+masked+iw@w372/v1cur#deadbeef"}"#,
    )
    .unwrap();
    let bad = feature_set::root_feature_set_ref(&dir).expect("still identifies itself");
    assert!(
        bad.slots.is_empty(),
        "a hash that disagrees with the registry must resolve to NO slots"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// **The alias contract, held on the committed registry.** Every declared key
/// is the legacy `<compute>@w<N>/<era>[#hash]` spelling; its LAYOUT-FREE
/// spelling must resolve to the SAME entry, and the two id strings must parse
/// to EQUAL ids naming the same slots and the same hash.
///
/// This is the whole justification for dropping the layout component: if a
/// canonical id could not find an append-only legacy key, the change would be
/// a migration rather than a rename, and every published id would need
/// rewriting.
#[test]
fn every_legacy_at_w_key_resolves_from_its_layout_free_spelling() {
    use zensim::feature_set_id::FeatureSetId;
    let reg = feature_set::registry();
    let (mut checked, mut both, mut canonical_only) = (0usize, 0usize, 0usize);
    for (key, set) in reg.sets() {
        let Some((head, rest)) = key.split_once('/') else {
            continue;
        };
        let Some((compute, w)) = head.split_once('@') else {
            continue; // already layout-free
        };
        assert!(
            w.starts_with('w'),
            "{key}: malformed layout component {w:?}"
        );
        let alias = format!("{compute}/{rest}");
        let via_alias = reg
            .set(&alias)
            .unwrap_or_else(|| panic!("{key}: layout-free spelling {alias:?} does not resolve"));
        assert_eq!(
            via_alias.key, set.key,
            "{alias} resolved to a different set"
        );
        checked += 1;

        // Both id STRINGS parse, and to equal ids.
        if let (Some(a), Some(b)) = (FeatureSetId::parse(key), FeatureSetId::parse(&alias)) {
            assert_eq!(a, b, "{key}: the two spellings must be the SAME id");
            assert_eq!(a.slots_hash(), b.slots_hash());
            assert_eq!(a.compute(), b.compute());
            assert_eq!(a.era(), b.era());
            assert_eq!(a.layout_width(), Some(w[1..].parse::<usize>().unwrap()));
            assert_eq!(
                b.layout_width(),
                None,
                "the canonical form carries no width"
            );
            // Reconstruction: when BOTH spellings rebuild a set, it must be
            // the SAME set. The layout-free form must also never reconstruct
            // LESS often — dropping a hint cannot lose information, because
            // the hint only picks one candidate out of the list the canonical
            // form searches.
            let ra = zensim::research::Request::for_set(&a)
                .map(|r| r.want().clone())
                .ok();
            let rb = zensim::research::Request::for_set(&b)
                .map(|r| r.want().clone())
                .ok();
            if let (Some(x), Some(y)) = (&ra, &rb) {
                assert_eq!(
                    x, y,
                    "{key}: the two spellings reconstruct different slot sets"
                );
                both += 1;
            }
            assert!(
                !(ra.is_some() && rb.is_none()),
                "{key}: the legacy spelling reconstructs and the canonical one does not — \
                 dropping a hint must not lose information"
            );
            if ra.is_none() && rb.is_some() {
                // MEASURED, and it is the canonical form being STRICTLY
                // better rather than a defect: this set is the family union at
                // a clip that is not the width the id records. The one
                // instance today is the `#0b476506` CONSUMER entry — the 265
                // free set minus the four LUMA_MEAN_REF slots — which is
                // exactly `union.clipped_to(924)`, while its `@w944` records
                // the WIRE width of the tables it reads. The legacy spelling
                // therefore pins the wrong candidate and fails; the canonical
                // one searches and finds it.
                canonical_only += 1;
            }
        }
    }
    assert!(
        checked >= 12,
        "the registry must carry legacy keys, saw {checked}"
    );
    // MEASURED, not aspirational: most registered sets are NOT reconstructible
    // from their compute tokens at any clip — they are PINNED slot lists (the
    // carriers set's scattered slots, the consumer read sets), which is why
    // the registry stores `slots` at all. Reconstruction is a convenience for
    // the family-union sets; the pinned list is the truth for the rest. The
    // properties that matter are the two asserted inside the loop: equal when
    // both rebuild, and never worse without the hint.
    assert!(
        both >= 1,
        "at least one key must reconstruct both ways, saw {both}"
    );
    assert_eq!(
        canonical_only, 1,
        "exactly one registered key is reconstructible ONLY from its layout-free spelling \
         (the #0b476506 consumer set, whose @w944 is a wire width and not its clip); a new \
          one is a finding to record, not a number to bump"
    );
}

/// The candidate clip-width list `zensim` searches for a layout-free id must
/// cover every width the committed registry actually uses. Registering a set
/// at a new width without extending the list would make that set
/// unreproducible from its canonical id — silently, because the search would
/// simply not find a hash match.
#[test]
fn every_registered_layout_width_is_a_candidate() {
    let candidates = zensim::feature_set_id::registered_layout_widths();
    let reg = feature_set::registry();
    let mut seen = 0usize;
    for (key, set) in reg.sets() {
        let Some(w) = set.id.layout else { continue };
        seen += 1;
        assert!(
            candidates.contains(&w),
            "{key}: layout w{w} is registered but not in \
             zensim::feature_set_id::registered_layout_widths() {candidates:?} — a canonical \
             (layout-free) id for this set could not be reconstructed"
        );
    }
    assert!(seen >= 12, "the registry must record layouts, saw {seen}");
}

/// **The 39 ids a 944 table DECLARES and the walk never writes** — the
/// table-side instance of the shape the cruft-purge ruling names ("a layout
/// where … features aren't computed").
///
/// MEASURED 2026-09-06 over every leg of
/// `/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01` (a full-column scan,
/// not a sample): 39 of 944 `f*` columns are exactly `0.0` on every row of
/// every leg, and they are the same 39 everywhere. They are the
/// `APPEND_SKIP_B_SCALE0` cells (`feature_defs::placement_admits` — "the 944
/// walk never computes that append cell") plus the reference-only /
/// HDR-gated append2 slots. The registry's `folded720append2pools` regime
/// declares `0-943`, and `Plan::emit` agrees with it, so BOTH declarations
/// overstate what is populated by 39.
///
/// This test PINS the state rather than asserting the fix, because fixing it
/// changes a refusal surface: with `populated_slots` modelling the placement
/// rules, `check` would begin reporting `SlotsNotPopulated` for any bake
/// reading one of the 39 — and shipped **CHdr reads 8 of them** (`f927/928`,
/// `f932/933`, `f937/938`, `f942/943`: `HL_BIN1`/`HL_BIN2` per scale, which
/// are structurally zero on an SDR route). That is a live product decision,
/// not a lane's, so it is registered with its blast radius measured.
///
/// When the fix lands, this test fails and its expectations are updated in the
/// same commit — which is the point of pinning it.
#[test]
fn a_944_regime_declares_39_ids_the_walk_never_writes() {
    /// The 39, measured all-zero on every leg of the ext944-era2r4 root.
    const NEVER_WRITTEN: &[usize] = &[
        720, 721, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769,
        770, 771, 772, 805, 806, 822, 823, 856, 857, 873, 874, 907, 908, 927, 928, 932, 933, 937,
        938, 942, 943,
    ];

    let (_, layout, declared) = feature_set::registry()
        .regime("folded720append2pools")
        .expect("the 944 regime is registered");
    assert_eq!(layout, 944);
    assert_eq!(
        declared.len(),
        944,
        "the registry declares every id populated"
    );
    for &s in NEVER_WRITTEN {
        assert!(
            declared.contains(s),
            "f{s} is declared populated by the registry — that is the overstatement"
        );
    }

    // `Plan::emit` agrees with the registry, so the owner does not model the
    // placement rules either. Asserting this is what makes the fix visible.
    let emit = zensim::research::Request::for_slots(declared.clone(), layout)
        .validate()
        .expect("the 944 request plans");
    assert_eq!(
        emit.len(),
        944,
        "Plan::emit does not model APPEND_SKIP_B_SCALE0 / reference-only slots yet"
    );

    // The blast radius of the fix, pinned on the shipped artifacts.
    let read_count = |rel: &str| -> usize {
        let bytes = std::fs::read(rel).unwrap_or_else(|e| panic!("{rel}: {e}"));
        let m = zenpredict::Model::from_bytes(&bytes).expect("parse bake");
        let used = zensim_validate::block_profile::used_caller_lines(&m).expect("caller lines");
        NEVER_WRITTEN.iter().filter(|s| used.contains(s)).count()
    };
    assert_eq!(
        read_count(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../zensim/weights/c_sdr_purity944_2026-08-29.bin"
        )),
        0,
        "shipped C reads none of the never-written slots"
    );
    assert_eq!(
        read_count(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../zensim/weights/c_hdr_l1t1944_2026-08-29.bin"
        )),
        8,
        "shipped CHdr reads EIGHT of them (HL_BIN1/HL_BIN2 per scale) — the measured blast \
         radius of modelling the placement rules in populated_slots"
    );
}
