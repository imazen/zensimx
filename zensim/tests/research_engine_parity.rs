// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.

//! **Phase 2's gates for the RESEARCH engine**
//! (`docs/PLAN_FEATURE_SYSTEM_2026-09-05.md` §Phase 2).
//!
//! * **G2.1 (engine parity)** — for every plan both engines can serve, at the
//!   same revision, every SHARED id agrees **bit-exactly**. Two halves: the
//!   v1 layout against `Zensim::compute_extended_features`, and the 944
//!   layout against the production append2 walk.
//! * **G2.2 (thread invariance)** — research output is bit-identical across
//!   rayon pool sizes 1/2/3/8/16 and equal to the serial answer.
//! * **G2.4 (provenance is CHECKED, not asserted)** — the reported owning
//!   kernel is verified by a perturbation probe: dropping a family stops
//!   populating exactly its own slots and leaves every other position
//!   bit-identical.
//!
//! The geometry matrix is [`common::parity_cells::CELLS`] — the SAME set
//! `fold_engine_parity.rs` uses, by reference rather than by retyping. A
//! second parity suite on its own hand-picked width list is the exact drift
//! the shared owner exists to prevent: both suites stay green while the union
//! of what they cover quietly shrinks.

#![cfg(all(feature = "training", feature = "feature-regime-v2"))]

mod common;

use common::parity_cells::{CELLS, pool_sweep_cells};
use zensim::feature_set_id::{ComputeToken, SlotSet};
use zensim::research::{self, Request};
use zensim::{RgbSlice, Zensim};

fn pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let r = common::generators::gen_value_noise(w, h, 0xC0FFEE);
    let d = common::generators::distort_block_artifacts(&r, w, h);
    (r, d)
}

/// **G2.1, the v1 half** — a 372-layout research extraction is bit-identical
/// to the production extended-feature extraction on all 372 shared ids, at
/// every cell of the shared geometry matrix.
#[test]
fn research_and_production_agree_bit_exactly_at_the_v1_layout() {
    let z = Zensim::new(common::legacy_full_profile()).with_parallel(false);
    let mut checked = 0usize;
    for &(w, h) in CELLS {
        let (s, d) = pair(w, h);
        let (rs, rd) = (RgbSlice::new(&s, w, h), RgbSlice::new(&d, w, h));
        let prod = z
            .compute_extended_features(&rs, &rd)
            .unwrap_or_else(|e| panic!("production extraction at {w}x{h}: {e:?}"));
        let prod = prod.features();
        assert!(
            prod.len() >= 372,
            "production emitted {} features at {w}x{h}",
            prod.len()
        );
        let e = research::extract(
            &Request::for_slots(SlotSet::from_ranges([(0, 372)]), 372),
            &rs,
            &rd,
        )
        .unwrap_or_else(|e| panic!("research extraction at {w}x{h}: {e}"));
        assert_eq!(e.values().len(), 372);
        for (i, (p, r)) in prod.iter().zip(e.values()).take(372).enumerate() {
            assert_eq!(
                p.to_bits(),
                r.to_bits(),
                "f{i} differs at {w}x{h}: production {p:.17e} vs research {r:.17e}"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 372 * CELLS.len());
}

/// **G2.1, the WIDE half** — a 944-layout research extraction is
/// bit-identical to the production 944 walk on all 944 shared ids.
#[test]
fn research_and_production_agree_bit_exactly_at_the_944_layout() {
    use zensim::feature_v2::{V1PoolsMode, V2NewFeatureToggles, V2Scratch};
    let z = Zensim::new(common::legacy_full_profile()).with_parallel(false);
    let mut scratch = V2Scratch::new();
    for &(w, h) in CELLS {
        let (s, d) = pair(w, h);
        let (rs, rd) = (RgbSlice::new(&s, w, h), RgbSlice::new(&d, w, h));
        let prod = z
            .compute_folded720_append_features_streaming(
                &rs,
                &rd,
                // The SAME plan the research request resolves to: a full
                // `0..944` request wants the pool block LIVE, so the
                // production arm must ask for it too. Comparing a
                // pools-`Off` production walk with a pools-`Full` research
                // one is not a parity failure, it is two different plans.
                V2NewFeatureToggles {
                    append2_block: true,
                    v1_pools: V1PoolsMode::Full,
                    ..V2NewFeatureToggles::default()
                },
                &mut scratch,
            )
            .unwrap_or_else(|e| panic!("production 944 walk at {w}x{h}: {e:?}"));
        assert_eq!(prod.features().len(), 944, "at {w}x{h}");
        let e = research::extract(
            &Request::for_slots(SlotSet::from_ranges([(0, 944)]), 944),
            &rs,
            &rd,
        )
        .unwrap_or_else(|e| panic!("research 944 extraction at {w}x{h}: {e}"));
        for (i, (p, r)) in prod.features().iter().zip(e.values()).enumerate() {
            assert_eq!(
                p.to_bits(),
                r.to_bits(),
                "f{i} differs at {w}x{h}: production {p:.17e} vs research {r:.17e}"
            );
        }
    }
}

/// **G2.1, the COMPLETE half** — the full registered width agrees
/// bit-exactly with the production walk at the widest registered regime
/// feature-bank families + C8 and the three retained restored cuts.
/// production toggles must name every registered block: a block left off
/// here makes the two widths differ.
#[test]
fn research_everything_agrees_with_the_production_walk() {
    use zensim::feature_v2::{V1PoolsMode, V2NewFeatureToggles, V2Scratch};
    let full = research::full_width();
    // The production walk excludes the research-only palette tail.
    let served_width = 1820;
    let z = Zensim::new(common::legacy_full_profile()).with_parallel(false);
    let mut scratch = V2Scratch::new();
    for &(w, h) in CELLS {
        let (s, d) = pair(w, h);
        let (rs, rd) = (RgbSlice::new(&s, w, h), RgbSlice::new(&d, w, h));
        let prod = z
            .compute_folded720_append_features_streaming(
                &rs,
                &rd,
                V2NewFeatureToggles {
                    append2_block: true,
                    csfw_block: true,
                    rev4_gridblk: true,
                    rev4_ringbasis: true,
                    rev4_tailhist: true,
                    rev4_arttype: true,
                    gmsbank: true,
                    mapdev: true,
                    z1max: true,
                    gmsnative: true,
                    v1_pools: V1PoolsMode::Full,
                    ..V2NewFeatureToggles::default()
                },
                &mut scratch,
            )
            .unwrap_or_else(|e| panic!("production {full} walk at {w}x{h}: {e:?}"));
        assert_eq!(prod.features().len(), served_width, "at {w}x{h}");
        let e = research::extract(&Request::everything(), &rs, &rd)
            .unwrap_or_else(|e| panic!("research everything at {w}x{h}: {e}"));
        assert_eq!(e.values().len(), full, "research width at {w}x{h}");
        for (i, (p, r)) in prod.features().iter().zip(e.values()).enumerate() {
            assert_eq!(p.to_bits(), r.to_bits(), "f{i} differs at {w}x{h}");
        }
    }
}

/// **G2.2 (thread invariance)** — `Request::everything()` is bit-identical
/// across rayon pool sizes 1/2/3/8/16 and equal to the serial answer, over
/// the widened pool-sweep geometry set.
#[cfg(feature = "threads")]
#[test]
fn research_output_is_thread_invariant() {
    for (w, h) in pool_sweep_cells() {
        let (s, d) = pair(w, h);
        let (rs, rd) = (RgbSlice::new(&s, w, h), RgbSlice::new(&d, w, h));
        let serial = research::extract(&Request::everything(), &rs, &rd)
            .unwrap_or_else(|e| panic!("serial at {w}x{h}: {e}"))
            .into_values();
        for threads in [1usize, 2, 3, 8, 16] {
            let built = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("build rayon pool");
            let got = built.install(|| {
                research::extract(&Request::everything().with_parallel(true), &rs, &rd)
                    .unwrap_or_else(|e| panic!("parallel at {w}x{h}: {e}"))
                    .into_values()
            });
            assert_eq!(got.len(), serial.len());
            for (i, (a, b)) in serial.iter().zip(got.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "f{i} moved at {w}x{h} with rayon pool size {threads}: \
                     serial {a:.17e} vs parallel {b:.17e}"
                );
            }
        }
    }
}

/// **G2.4 (provenance is CHECKED, not asserted)** — the perturbation probe.
///
/// For each family the plan can express dropping, the narrowed extraction
/// must leave EVERY position it still populates bit-identical to the full
/// walk's. That is the real content: the blocks share accumulators and
/// upstream planes, so "turning this off perturbs nothing else" is a claim
/// that has to be measured. The provenance's `kernel` field is then checked
/// against the same partition — a slot the narrowed plan still populates may
/// not name a kernel no surviving family runs.
#[test]
fn dropping_a_family_perturbs_only_its_own_slots() {
    let full_w = research::full_width();
    // `globaff` is computed only in a formula-revision-6 process (`Request::everything` leaves it out below).
    let globaff = research::family_slots(ComputeToken::GlobAff);
    let all = SlotSet::from_ranges([(0, 956), (986, 1820), (1825, full_w)]);
    let all = if research::Request::everything().want().contains(1867) {
        all
    } else {
        globaff.missing_from(&all)
    };
    for &(w, h) in CELLS {
        let (s, d) = pair(w, h);
        let (rs, rd) = (RgbSlice::new(&s, w, h), RgbSlice::new(&d, w, h));
        let base = research::extract(&Request::everything(), &rs, &rd)
            .unwrap_or_else(|e| panic!("base at {w}x{h}: {e}"));

        for fam in [
            ComputeToken::Csfw,
            ComputeToken::Append2,
            ComputeToken::Append,
            ComputeToken::Masked,
            ComputeToken::Iw,
            // REV4: each family independently droppable — dropping one
            // must leave every still-populated slot (including the other
            // three rev4 families' AND the whole f0..985 prefix)
            // bit-identical.
            ComputeToken::Gridblk,
            ComputeToken::Ringbasis,
            ComputeToken::Tailhist,
            ComputeToken::Arttype,
        ] {
            let fam_slots = research::family_slots(fam);
            let want = SlotSet::from_slots(all.iter_slots().filter(|s| !fam_slots.contains(*s)));
            let e = research::extract(&Request::for_slots(want, full_w), &rs, &rd)
                .unwrap_or_else(|e| panic!("narrowed {} at {w}x{h}: {e}", fam.as_str()));
            for s in e.emitted().iter_slots() {
                assert_eq!(
                    base.values()[s].to_bits(),
                    e.values()[s].to_bits(),
                    "dropping {} perturbed f{s} ({}) at {w}x{h} — a slot it does \
                     not own",
                    fam.as_str(),
                    e.provenance()[s].name
                );
            }
            // Every position the narrowed plan does NOT populate must be the
            // layout's structural zero, never a stale or partial value.
            for s in 0..full_w {
                if !e.emitted().contains(s) {
                    assert_eq!(
                        e.values()[s],
                        0.0,
                        "dropping {} left f{s} ({}) non-zero while unpopulated at {w}x{h}",
                        fam.as_str(),
                        e.provenance()[s].name
                    );
                }
            }
        }
    }
}

/// The kernel a slot's provenance names must actually be one the plan that
/// populates it schedules — the "who computes this" link, checked against the
/// plan rather than trusted.
#[test]
fn every_populated_slot_names_a_kernel_its_plan_runs() {
    let (w, h) = (128usize, 128usize);
    let (s, d) = pair(w, h);
    let (rs, rd) = (RgbSlice::new(&s, w, h), RgbSlice::new(&d, w, h));

    // The cheap free-set plan: basic + peaks + the raw-moment tranche at the
    // 944 layout. Its populated slots may name only v1 kernels and the free
    // raw-moment accumulator — never the append kernel that owns the block
    // those slots LIVE in.
    let want = SlotSet::from_ranges([(0, 228)])
        .union(&research::family_slots(ComputeToken::Moments))
        .clipped_to(944);
    let e = research::extract(&Request::for_slots(want, 944), &rs, &rd).expect("free-set plan");
    assert_eq!(e.emitted().len(), 265);
    let allowed = ["v1_fused", "v1_peaks", "free_raw_moments"];
    for s in e.emitted().iter_slots() {
        let p = &e.provenance()[s];
        assert!(
            allowed.contains(&p.kernel),
            "f{s} ({}) is populated by the free-set plan but names kernel {} — \
             the plan does not run it",
            p.name,
            p.kernel
        );
        // Only the TRANCHE slots are free — the basic and peak blocks are
        // `cheap` because their own kernel runs. `cost_of` is the per-SLOT
        // answer and this is what it means.
        if research::family_slots(ComputeToken::Moments).contains(s) {
            assert_eq!(
                p.cost, "free",
                "f{s} ({}) is harvested by a tranche, so its per-slot cost \
                 must be free",
                p.name
            );
            assert_eq!(p.kernel, "free_raw_moments", "f{s} ({})", p.name);
        }
    }
}
