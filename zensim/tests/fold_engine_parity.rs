//! **The stage-2 gate of the fold-engine lane** (design note:
//! `benchmarks/fold_engine_2026-08-31.md`): every scoring entry, run through
//! the streaming fold, must return a `ZensimResult` **bit-identical** to the
//! buffered walk's.
//!
//! ## Why bit-identical and not "within the golden tolerance"
//!
//! The golden gate's `max(1e-6 abs, 1e-5·scale)` tolerance exists for
//! cross-*environment* drift — the measured 241-246-of-372 divergence between
//! CPU-vendor classes (`benchmarks/v1_golden_env_triage_2026-08-05.md`). It is
//! not a licence for two code paths on the SAME box to disagree. Since option
//! C (`blur::pyramid_plane_stride(w) == w`) the fold and the buffered walk
//! compute the same statistic over the same pixels, and
//! `feature_v2::tests::v1_372_bit_exact_to_fold_at_every_width` already proves
//! the feature block agrees to the bit at 19 geometries. This file extends
//! that from the FEATURE VECTOR to the whole `ZensimResult` — `score`,
//! `raw_distance`, every feature, and `mean_offset` — across the product
//! entry points, so a fold-backed `compute()` is a substitution rather than
//! an approximation.
//!
//! Anything the fold cannot serve bit-identically must FALL BACK to buffered
//! rather than answer differently; `fold_engine::is_fold_backable` owns that
//! predicate and `fold_falls_back_on_a_weight_skipping_profile` is its gate.

#![cfg(all(feature = "training", feature = "feature-regime-v2"))]

mod common;

use zensim::fold_engine::ScoringEngine;
use zensim::{RgbSlice, Zensim, ZensimProfile, ZensimResult};

fn pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let r = common::generators::gen_value_noise(w, h, 0xC0FFEE);
    let d = common::generators::distort_block_artifacts(&r, w, h);
    (r, d)
}

fn load_png_rgb8(path: &std::path::Path) -> (Vec<[u8; 3]>, usize, usize) {
    use zenpixels::SampleType;
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let cfg = zenpng::PngDecodeConfig::default();
    let out = zenpng::decode(&bytes, &cfg, &enough::Unstoppable).expect("zenpng decode");
    let (w, h) = (out.info.width as usize, out.info.height as usize);
    let layout = out.pixels.spec().layout;
    assert_eq!(layout.sample, SampleType::U8);
    let slice = out.pixels.as_view();
    let (channels, has_alpha) = (layout.channels(), layout.order.has_alpha_slot());
    let mut rgb = Vec::with_capacity(w * h);
    for y in 0..h as u32 {
        let row = slice.row(y);
        match (channels, has_alpha) {
            (4, true) => {
                for px in row.as_chunks::<4>().0.iter().take(w) {
                    rgb.push([px[0], px[1], px[2]]);
                }
            }
            (3, false) => {
                for px in row.as_chunks::<3>().0.iter().take(w) {
                    rgb.push([px[0], px[1], px[2]]);
                }
            }
            other => panic!("unsupported PNG channel layout {other:?}"),
        }
    }
    (rgb, w, h)
}

/// The real-photo golden fixture (96×96 gb82 `city.png` / `city_q50.jpg`
/// crop) — the same bytes `v1_golden_bytes::GOLDEN_REAL` pins.
fn golden_real() -> (Vec<[u8; 3]>, Vec<[u8; 3]>, usize, usize) {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let (r, rw, rh) = load_png_rgb8(&manifest.join("tests/fixtures/v1_golden_real_ref.png"));
    let (d, dw, dh) = load_png_rgb8(&manifest.join("tests/fixtures/v1_golden_real_dist.png"));
    assert_eq!((rw, rh), (dw, dh), "fixture dimension mismatch");
    (r, d, rw, rh)
}

/// Every field of a `ZensimResult`, to the bit. `to_bits()` rather than `==`
/// so a NaN can never pass as "equal" and a `-0.0`/`0.0` swap is caught.
#[track_caller]
fn assert_result_bit_identical(ctx: &str, buffered: &ZensimResult, fold: &ZensimResult) {
    assert_eq!(
        buffered.score().to_bits(),
        fold.score().to_bits(),
        "{ctx}: score buffered {:.17e} vs fold {:.17e}",
        buffered.score(),
        fold.score()
    );
    assert_eq!(
        buffered.raw_distance().to_bits(),
        fold.raw_distance().to_bits(),
        "{ctx}: raw_distance buffered {:.17e} vs fold {:.17e}",
        buffered.raw_distance(),
        fold.raw_distance()
    );
    let (bf, ff) = (buffered.features(), fold.features());
    assert_eq!(
        bf.len(),
        ff.len(),
        "{ctx}: feature width buffered {} vs fold {}",
        bf.len(),
        ff.len()
    );
    let mut bad = Vec::new();
    for (i, (&b, &f)) in bf.iter().zip(ff.iter()).enumerate() {
        if b.to_bits() != f.to_bits() {
            bad.push((i, b, f));
        }
    }
    assert!(
        bad.is_empty(),
        "{ctx}: {} of {} features differ between the buffered walk and the fold. \
         These paths compute the same statistic over the same pixels since option C — \
         a difference here means padding was reintroduced, a summation grouping moved, \
         or the fold was asked for a request it cannot serve. Do NOT widen this to a \
         tolerance. First 12: {:?}",
        bad.len(),
        bf.len(),
        &bad[..bad.len().min(12)]
    );
    for c in 0..3 {
        assert_eq!(
            buffered.mean_offset()[c].to_bits(),
            fold.mean_offset()[c].to_bits(),
            "{ctx}: mean_offset[{c}] buffered {:.17e} vs fold {:.17e} — the fold's \
             per-row decomposition must reproduce compute_xyb_mean_offset's 64-row \
             chunk reduction EXACTLY, not to an epsilon",
            buffered.mean_offset()[c],
            fold.mean_offset()[c]
        );
    }
}

use common::parity_cells::CELLS;

fn run_both<F>(profile: ZensimProfile, parallel: bool, f: F) -> (ZensimResult, ZensimResult)
where
    F: Fn(&Zensim) -> ZensimResult,
{
    let buffered = Zensim::new(profile).with_parallel(parallel);
    let fold = Zensim::new(profile)
        .with_parallel(parallel)
        .with_engine(ScoringEngine::Fold);
    assert_eq!(buffered.engine(), ScoringEngine::Buffered);
    assert_eq!(fold.engine(), ScoringEngine::Fold);
    (f(&buffered), f(&fold))
}

/// `Zensim::compute` — the product entry, profile B (372 inputs, MLP/linear
/// bake + PCHIP output spline + extrapolate disposition). The bake forward
/// and the spline are SHARED code that runs after both walks, so this test
/// covers the whole chain, not just extraction.
#[test]
fn compute_is_bit_identical_across_engines() {
    for &parallel in &[false, true] {
        for &(w, h) in CELLS {
            let (r, d) = pair(w, h);
            let (b, f) = run_both(common::legacy_full_profile(), parallel, |z| {
                z.compute(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h))
                    .expect("compute")
            });
            assert_result_bit_identical(&format!("compute B {w}x{h} par={parallel}"), &b, &f);
        }
    }
}

/// `Zensim::compute_extended_features` at both widths the shipped profiles
/// produce: 372 (profile B — extended + IW) and 300 (`PreviewV0_2` —
/// extended, no IW). The 300 case is the §2.4 prefix claim under test: the
/// fold emits 372 and truncates, so if v1's narrow layout were NOT a prefix
/// of its wide one this fails here.
#[test]
fn extended_features_are_bit_identical_across_engines() {
    for &(profile, want) in &[
        (common::legacy_full_profile(), 372usize),
        (ZensimProfile::PreviewV0_2, 300usize),
    ] {
        for &parallel in &[false, true] {
            for &(w, h) in CELLS {
                let (r, d) = pair(w, h);
                let (b, f) = run_both(profile, parallel, |z| {
                    z.compute_extended_features(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h))
                        .expect("compute_extended_features")
                });
                assert_eq!(
                    b.features().len(),
                    want,
                    "{profile:?} {w}x{h}: buffered width"
                );
                assert_result_bit_identical(
                    &format!("extended {profile:?} {w}x{h} par={parallel}"),
                    &b,
                    &f,
                );
            }
        }
    }
}

/// The REAL-photo golden fixture, both entries. Procedural noise and a real
/// JPEG-artefacted photo have very different statistics; the golden set keeps
/// both for that reason and so does this gate.
#[test]
fn golden_real_fixture_is_bit_identical_across_engines() {
    let (r, d, w, h) = golden_real();
    for &parallel in &[false, true] {
        let (b, f) = run_both(common::legacy_full_profile(), parallel, |z| {
            z.compute(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h))
                .expect("compute")
        });
        assert_result_bit_identical(&format!("GOLDEN_REAL compute par={parallel}"), &b, &f);
        let (b, f) = run_both(common::legacy_full_profile(), parallel, |z| {
            z.compute_extended_features(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h))
                .expect("compute_extended_features")
        });
        assert_result_bit_identical(&format!("GOLDEN_REAL extended par={parallel}"), &b, &f);
    }
}

/// The fold-backed result must not depend on how many threads rayon happens
/// to have — the same invariant `v1_372_is_bit_identical_across_rayon_pool_sizes`
/// states for the buffered path. Both engines are checked in the same loop so
/// a failure names which one moved.
#[cfg(feature = "threads")]
#[test]
fn both_engines_are_bit_identical_across_rayon_pool_sizes() {
    // WIDENED by the fold-MT lane (`benchmarks/fold_mt_scaling_2026-08-31.md`)
    // from four hand-picked shapes to the WHOLE geometry set, plus the two
    // large shapes the scaling work is measured on. Every lever that lane
    // ships is a schedule change in the PARALLEL arm — H-blur row bands, the
    // two-sided producer front end, the six-way downscale cascade, the fused
    // per-channel fan-out — so a band boundary interacting with a geometry is
    // exactly the failure mode, and only a pool sweep over every geometry can
    // see it.
    let cells = common::parity_cells::pool_sweep_cells();
    for &(w, h) in &cells {
        let (r, d) = pair(w, h);
        let mut first: Option<(ZensimResult, ZensimResult)> = None;
        for threads in [1usize, 2, 3, 8, 16] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("build rayon pool");
            let (b, f) = pool.install(|| {
                run_both(common::legacy_full_profile(), true, |z| {
                    z.compute(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h))
                        .expect("compute")
                })
            });
            // Cross-engine parity holds at every thread count.
            assert_result_bit_identical(&format!("{w}x{h} @ {threads}T"), &b, &f);
            match &first {
                None => first = Some((b, f)),
                Some((b0, f0)) => {
                    assert_result_bit_identical(
                        &format!("{w}x{h}: BUFFERED moved between 1 and {threads} threads"),
                        b0,
                        &b,
                    );
                    assert_result_bit_identical(
                        &format!("{w}x{h}: FOLD moved between 1 and {threads} threads"),
                        f0,
                        &f,
                    );
                }
            }
        }
    }
}

/// The fallback contract. `PreviewV0_2`'s plain `compute()` config is
/// weight-skipping (`compute_all_features == false`, `extended_features ==
/// false`), which `streaming::active_channels` honours by leaving
/// zero-weight channels' slots at their default while the fold always
/// computes all three. `is_fold_backable` rejects that config, so asking for
/// the fold must return the BUFFERED answer — identical, not merely
/// score-equal.
#[test]
fn fold_falls_back_on_a_weight_skipping_profile() {
    for &(w, h) in &[(200usize, 150usize), (256, 256)] {
        let (r, d) = pair(w, h);
        let (b, f) = run_both(ZensimProfile::PreviewV0_2, true, |z| {
            z.compute(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h))
                .expect("compute")
        });
        assert_eq!(b.features().len(), 228, "{w}x{h}: PreviewV0_2 is 228-wide");
        assert_result_bit_identical(&format!("fallback PreviewV0_2 {w}x{h}"), &b, &f);
    }
}

/// The identical-pair short-circuit is SHARED (it runs in
/// `compute_with_config_core` before either walk), so both engines must
/// return the same `mark_identical` payload — score exactly 100, an all-zero
/// feature vector of the config's width, and a zero mean_offset.
#[test]
fn identical_pair_short_circuit_is_shared() {
    let (w, h) = (200usize, 150usize);
    let (r, _) = pair(w, h);
    let (b, f) = run_both(common::legacy_full_profile(), true, |z| {
        z.compute(&RgbSlice::new(&r, w, h), &RgbSlice::new(&r, w, h))
            .expect("compute")
    });
    assert_eq!(b.score(), 100.0);
    assert_result_bit_identical("identical pair", &b, &f);
}

/// Classification rides `Zensim::compute`, so it is fold-backed for free —
/// stated as its own gate so a future refactor that gives `classify` its own
/// walk cannot silently skip the parity check. The delta-stats half is a pure
/// per-pixel pass over the two sources with no pyramid, so it is
/// engine-independent by construction; asserted anyway.
#[cfg(feature = "classification")]
#[test]
fn classify_is_bit_identical_across_engines() {
    for &(w, h) in &[(200usize, 150usize), (96, 64), (127, 93)] {
        let (r, d) = pair(w, h);
        let buffered = Zensim::new(common::legacy_full_profile());
        let fold = Zensim::new(common::legacy_full_profile()).with_engine(ScoringEngine::Fold);
        let cb = buffered
            .classify(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h))
            .expect("classify");
        let cf = fold
            .classify(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h))
            .expect("classify");
        assert_result_bit_identical(&format!("classify {w}x{h}"), &cb.result, &cf.result);
        assert_eq!(
            format!("{:?}", cb.classification),
            format!("{:?}", cf.classification),
            "{w}x{h}: classification differs between engines"
        );
    }
}

// ── STAGE 3: the ref-cached fold form ────────────────────────────────────

/// **The stage-3 gate.** N distorted candidates scored against ONE
/// `precompute_reference` must equal N independent `compute` calls, on the
/// fold, **bit-for-bit** — every field, at every geometry.
///
/// This is a strictly STRONGER property than the buffered path has for
/// itself: buffered's `compute_with_ref` derives `mean_offset` from a
/// strip-wise `offset_sums / pixel_count` accumulation rather than
/// `compute_xyb_mean_offset`'s 64-row chunk reduction, so
/// `cross_platform::mean_offset_precomputed_ref` can only assert `< 1e-10`
/// between buffered's own two entries. The fold's ref-cached form feeds the
/// SAME producer over the SAME planes as its direct form, so it has nothing
/// to round differently.
#[test]
fn fold_ref_cache_matches_independent_computes() {
    for &(w, h) in CELLS {
        let (r, _) = pair(w, h);
        // Several distinct distortions against one reference — the encoder
        // loop's actual shape, not a single-candidate smoke test.
        let dists: Vec<Vec<[u8; 3]>> = (0..4)
            .map(|k| {
                let mut d = common::generators::distort_block_artifacts(&r, w, h);
                for (i, px) in d.iter_mut().enumerate() {
                    if i % (3 + k) == 0 {
                        px[k % 3] = px[k % 3].saturating_add(7 * (k as u8 + 1));
                    }
                }
                d
            })
            .collect();
        for parallel in [false, true] {
            let z = Zensim::new(common::legacy_full_profile())
                .with_parallel(parallel)
                .with_engine(ScoringEngine::Fold);
            let rs = RgbSlice::new(&r, w, h);
            let precomputed = z.precompute_reference(&rs).expect("precompute");
            for (k, d) in dists.iter().enumerate() {
                let ds = RgbSlice::new(d, w, h);
                let direct = z.compute(&rs, &ds).expect("compute");
                let with_ref = z.compute_with_ref(&precomputed, &ds).expect("with_ref");
                assert_result_bit_identical(
                    &format!("fold ref-cache {w}x{h} cand{k} par={parallel}"),
                    &direct,
                    &with_ref,
                );
            }
        }
    }
}

/// **The fold-MT lane's scratch-reuse gate.** `compute_with_ref_into` now
/// routes to the fold and hands it the caller's `ZensimScratch`, so a fold
/// ref-loop reuses its `V2Scratch` across compares instead of re-paying the
/// first-touch page-fault commit on ~61 MB of strip planes every call.
///
/// Reuse cannot move a byte — every kernel fully overwrites the buffers it
/// reads — but "cannot" is what a gate is for. Three claims, all `to_bits`:
///
/// 1. `compute_with_ref_into` with a REUSED scratch equals `compute_with_ref`
///    (a fresh allocation per call) for every candidate in the loop.
/// 2. It also equals a plain `compute`, so the routing did not quietly land
///    on a different walk.
/// 3. Reuse across DIFFERENT geometries in one scratch is safe (the scratch
///    only ever grows), which is the shape a batch consumer actually has.
#[test]
fn fold_ref_scratch_reuse_is_bit_identical() {
    for parallel in [false, true] {
        let z = Zensim::new(common::legacy_full_profile())
            .with_parallel(parallel)
            .with_engine(ScoringEngine::Fold);
        // ONE scratch across every geometry and every candidate — claim 3.
        let mut scratch = zensim::ZensimScratch::new();
        for &(w, h) in CELLS {
            let (r, _) = pair(w, h);
            let rs = RgbSlice::new(&r, w, h);
            let precomputed = z.precompute_reference(&rs).expect("precompute");
            for k in 0..4usize {
                let mut d = common::generators::distort_block_artifacts(&r, w, h);
                for (i, px) in d.iter_mut().enumerate() {
                    if i % (3 + k) == 0 {
                        px[k % 3] = px[k % 3].saturating_add(7 * (k as u8 + 1));
                    }
                }
                let ds = RgbSlice::new(&d, w, h);
                let fresh = z.compute_with_ref(&precomputed, &ds).expect("with_ref");
                let reused = z
                    .compute_with_ref_into(&precomputed, &ds, &mut scratch)
                    .expect("with_ref_into");
                assert_result_bit_identical(
                    &format!("fold scratch reuse {w}x{h} cand{k} par={parallel}"),
                    &fresh,
                    &reused,
                );
                let direct = z.compute(&rs, &ds).expect("compute");
                assert_result_bit_identical(
                    &format!("fold scratch reuse vs direct {w}x{h} cand{k} par={parallel}"),
                    &direct,
                    &reused,
                );
            }
        }
    }
}

/// Cross-engine at the `compute_with_ref` entry. Features, score and
/// `raw_distance` are bit-identical; `mean_offset` is the one field where the
/// two engines' *ref-cached* forms genuinely differ, because buffered's
/// `*_with_ref` accumulates it strip-wise while the fold reproduces
/// `compute_xyb_mean_offset` exactly — the same divergence
/// `cross_platform::mean_offset_precomputed_ref` already bounds at `1e-10`
/// between buffered's OWN two entries. Asserted at that same bound, and the
/// observed maximum is printed so a drift shows up as a number rather than as
/// a pass.
#[test]
fn compute_with_ref_cross_engine() {
    let mut worst = 0.0f64;
    for &(w, h) in CELLS {
        let (r, d) = pair(w, h);
        let rs = RgbSlice::new(&r, w, h);
        let ds = RgbSlice::new(&d, w, h);
        for parallel in [false, true] {
            let zb = Zensim::new(common::legacy_full_profile()).with_parallel(parallel);
            let zf = Zensim::new(common::legacy_full_profile())
                .with_parallel(parallel)
                .with_engine(ScoringEngine::Fold);
            let pb = zb.precompute_reference(&rs).expect("precompute");
            let pf = zf.precompute_reference(&rs).expect("precompute");
            let b = zb.compute_with_ref(&pb, &ds).expect("with_ref");
            let f = zf.compute_with_ref(&pf, &ds).expect("with_ref");
            let (bf, ff) = (b.features(), f.features());
            assert_eq!(bf.len(), ff.len());
            for (i, (&x, &y)) in bf.iter().zip(ff.iter()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "{w}x{h} par={parallel}: with_ref f{i} buffered {x:.17e} vs fold {y:.17e}"
                );
            }
            assert_eq!(
                b.score().to_bits(),
                f.score().to_bits(),
                "{w}x{h} par={parallel}: with_ref score"
            );
            assert_eq!(
                b.raw_distance().to_bits(),
                f.raw_distance().to_bits(),
                "{w}x{h} par={parallel}: with_ref raw_distance"
            );
            for c in 0..3 {
                let diff = (b.mean_offset()[c] - f.mean_offset()[c]).abs();
                worst = worst.max(diff);
                assert!(
                    diff < 1e-10,
                    "{w}x{h} par={parallel}: with_ref mean_offset[{c}] buffered {:.17e} vs \
                     fold {:.17e} (diff {diff:.3e})",
                    b.mean_offset()[c],
                    f.mean_offset()[c]
                );
            }
        }
    }
    println!("compute_with_ref cross-engine worst |Δmean_offset| = {worst:.3e}");
}

// ── STAGE 4: the attribution canvas ──────────────────────────────────────

/// **The stage-4 structural claim, stated as a test.** The predecessor lane
/// recorded "attribution's basic canvas is buffered-native" as retirement
/// blocker 4. Read at source that is true of the concrete TYPE and not of the
/// WALK: `build_attribution_into_sink` calls only `crate::blur` — never
/// `compute_multiscale_stats_streaming`, `process_scale_bands`, or any other
/// walk function — and its only buffered dependency is a pyramid to read.
/// This lane made that structural by giving the builder a
/// `&impl MultiScaleRef` signature.
///
/// The observable consequence, gated here: `compute_attribution_density*` is
/// **engine-independent** — bit-identical density and SAT block sums under
/// both engines, at per-pixel and binned resolution, for the basic-only and
/// the full-coverage (v2 + append + append2) entries. A future change that
/// routes the canvas through a walk would break this.
#[cfg(feature = "custom-profiles")]
#[test]
fn attribution_density_is_engine_independent() {
    let s_basic: Vec<f64> = (0..156)
        .map(|k| if k % 3 == 0 { -1.0 } else { -0.25 } * (1.0 + (k % 7) as f64 * 0.1))
        .collect();
    let mut s_full = vec![0.0f64; 944];
    for (k, v) in s_full.iter_mut().enumerate() {
        *v = ((k % 13) as f64 - 6.0) * 0.05;
    }
    for &(w, h) in &[(150usize, 170usize), (200, 150), (96, 64), (127, 93)] {
        let (r, d) = pair(w, h);
        let rs = RgbSlice::new(&r, w, h);
        let ds = RgbSlice::new(&d, w, h);
        let zb = Zensim::new(common::legacy_full_profile());
        let zf = Zensim::new(common::legacy_full_profile()).with_engine(ScoringEngine::Fold);

        let cmp = |name: &str, a: &zensim::AttributionResult, b: &zensim::AttributionResult| {
            assert_eq!(
                (a.width(), a.height()),
                (b.width(), b.height()),
                "{name} dims"
            );
            for (i, (&x, &y)) in a.density().iter().zip(b.density().iter()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "{name} {w}x{h}: density[{i}] buffered {x:e} vs fold {y:e} — the \
                     attribution canvas reads a pyramid, not a walk, so it must not \
                     depend on the engine at all"
                );
            }
            let (ba, bb) = (a.block_sums(16), b.block_sums(16));
            for (i, (&x, &y)) in ba.iter().zip(bb.iter()).enumerate() {
                assert_eq!(x.to_bits(), y.to_bits(), "{name} {w}x{h}: block_sums[{i}]");
            }
        };

        cmp(
            "density",
            &zb.compute_attribution_density(&rs, &ds, &s_basic).unwrap(),
            &zf.compute_attribution_density(&rs, &ds, &s_basic).unwrap(),
        );
        cmp(
            "density_binned8",
            &zb.compute_attribution_density_binned(&rs, &ds, &s_basic, 8)
                .unwrap(),
            &zf.compute_attribution_density_binned(&rs, &ds, &s_basic, 8)
                .unwrap(),
        );
        cmp(
            "density_full",
            &zb.compute_attribution_density_full(&rs, &ds, &s_full)
                .unwrap(),
            &zf.compute_attribution_density_full(&rs, &ds, &s_full)
                .unwrap(),
        );
    }
}

/// **The retirement migration path for the fused compare, measured as a
/// gate.** `compute_with_ref_score_and_attribution` is the one genuinely
/// walk-bound attribution consumer: it folds the map in-strip inside
/// `streaming::compute_zensim_streaming_with_ref_and_attr_{planes,fold}`,
/// whose band tiling (`BAND_ROWS == streaming::STRIP_INNER`) is chosen to be
/// bit-compatible with the buffered extractor. It therefore stays on buffered
/// under `ScoringEngine::Fold` (the fallback), and a caller that wants the
/// fold has to SPLIT it into a fold-backed score plus a standalone map.
///
/// This gates that the split is sound: the fold-backed `compute_with_ref`
/// score is **bit-identical** to the fused compare's score, so only the map
/// half changes — and the map half's difference is already characterised by
/// `attribution::tests::fused_matches_standalone_attribution` (f32-combine
/// precision, 3e-5·max_abs). What the split COSTS is the bench's
/// `fused_buffered` vs `split_fold` arms; this test is what it does not cost.
#[cfg(feature = "custom-profiles")]
#[test]
fn fused_compare_splits_into_fold_score_plus_standalone_map() {
    let s: Vec<f64> = (0..156)
        .map(|k| if k % 3 == 0 { -1.0 } else { -0.25 } * (1.0 + (k % 7) as f64 * 0.1))
        .collect();
    for &(w, h) in &[(150usize, 170usize), (200, 150), (127, 93)] {
        let (r, d) = pair(w, h);
        let rs = RgbSlice::new(&r, w, h);
        let ds = RgbSlice::new(&d, w, h);
        let zb = Zensim::new(common::legacy_full_profile());
        let zf = Zensim::new(common::legacy_full_profile()).with_engine(ScoringEngine::Fold);
        let pre_b = zb.precompute_reference(&rs).unwrap();
        let pre_f = zf.precompute_reference(&rs).unwrap();

        let (fused_res, _fused_map) = zb
            .compute_with_ref_score_and_attribution(&pre_b, &ds, &s)
            .unwrap();
        let split_res = zf.compute_with_ref(&pre_f, &ds).unwrap();
        assert_eq!(
            fused_res.score().to_bits(),
            split_res.score().to_bits(),
            "{w}x{h}: fused-compare score {:.17e} vs fold-backed compute_with_ref {:.17e} — \
             the split must not move the number the product ships",
            fused_res.score(),
            split_res.score()
        );

        // …and the fused entry itself is engine-invariant, because it falls
        // back: asking for the fold returns the buffered answer verbatim.
        let (fused_f, _) = zf
            .compute_with_ref_score_and_attribution(&pre_f, &ds, &s)
            .unwrap();
        assert_eq!(
            fused_res.score().to_bits(),
            fused_f.score().to_bits(),
            "{w}x{h}: the fused compare must fall back to buffered identically"
        );
    }
}

/// **Per-profile weight-skipping must be INERT for every shipped profile**
/// (feature-cost lane, 2026-08-31).
///
/// `Zensim::with_unread_feature_skipping(true)` lets a fold-backed score drop
/// the v1 masked/IW pass group when the profile's bakes structurally ignore
/// `f228..372`. Profile B does NOT ignore it (its layer 0 carries 10 live
/// masked lines and 13 live IW lines — `bake_block_profile`, 2026-08-31), so
/// the policy must resolve to `V1PoolsMode::Full` and the opted-in result must
/// be bit-identical to the default one — on BOTH engines, at every geometry,
/// at every rayon pool size. That is the "no silent behaviour change for
/// existing callers" half of the contract, gated as a fact rather than an
/// intention: if a future bake, repack or prune zeroed those columns, this
/// test would keep passing (the skip would then be genuinely score-neutral),
/// but a policy bug that fired on a live block fails it immediately.
#[cfg(feature = "threads")]
#[test]
fn unread_feature_skipping_is_inert_on_a_profile_that_reads_the_block() {
    assert_eq!(
        Zensim::new(common::legacy_full_profile())
            .with_unread_feature_skipping(true)
            .score_pool_mode(),
        zensim::feature_v2::V1PoolsMode::Full,
        "B reads f228..372 — the policy must refuse to skip it"
    );
    let cells = common::parity_cells::pool_sweep_cells();
    for &(w, h) in &cells {
        let (r, d) = pair(w, h);
        for threads in [1usize, 2, 3, 8, 16] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("build rayon pool");
            pool.install(|| {
                for engine in [ScoringEngine::Buffered, ScoringEngine::Fold] {
                    let base = Zensim::new(common::legacy_full_profile())
                        .with_parallel(true)
                        .with_engine(engine);
                    let skipping = Zensim::new(common::legacy_full_profile())
                        .with_parallel(true)
                        .with_engine(engine)
                        .with_unread_feature_skipping(true);
                    let a = base
                        .compute(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h))
                        .expect("compute");
                    let b = skipping
                        .compute(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h))
                        .expect("compute");
                    assert_result_bit_identical(
                        &format!("{w}x{h} @ {threads}T {engine:?}: skipping moved the result"),
                        &a,
                        &b,
                    );
                    // And the block it would have skipped is genuinely live.
                    assert!(
                        a.features()[228..372].iter().any(|&v| v != 0.0),
                        "{w}x{h}: masked/IW block is all-zero — the gate would be vacuous"
                    );
                }
            });
        }
    }
}

/// **The 944 walk's pool-size gate** (v2-cost lane, 2026-08-31 —
/// `benchmarks/v2_block_cost_2026-08-31.md`).
///
/// `both_engines_are_bit_identical_across_rayon_pool_sizes` above sweeps
/// pools for the SCORING path, which runs `v1_only` — so it exercises the
/// producer, the H blur and the v1 band replay, and **not one v2-era
/// kernel**. The 944 extraction walk is the arm that runs `dense`,
/// `gradient`, `append`, the append2/BANDVIS instantiation, `blockiness`,
/// and phase A's four `box_blur_v_from_copy` sweeps plus the activity and
/// `bs2` chains; before this test the only thread coverage those had was
/// `append2_layout_identity_and_first924_bit_stable`'s single serial-vs-
/// parallel pair at the ambient pool size.
///
/// That is the gap this closes, and it is not hypothetical: the v2 block's
/// per-strip partials are merged by `DenseAccum::accumulate` and its
/// siblings, whose own doc records that strip order changes the GROUPING of
/// the merge. Anything that re-schedules the phase-A/phase-B fan-out, the
/// H-blur row bands, or the band-parallel slot count is therefore one
/// grouping change away from silently moving `f372..943` — the exact class
/// of change the era-2 restructure (`benchmarks/era2_perf_break_2026-08-31.md`)
/// and the column-tiling lane are both aiming at. A pool sweep over the
/// whole geometry set is what makes such a move fail loudly instead of
/// landing in a training table.
///
/// Asserted on `to_bits()` over all 944 slots, so a NaN slot must also match
/// its payload rather than comparing unequal to itself.
#[cfg(feature = "threads")]
#[test]
fn folded944_is_bit_identical_across_rayon_pool_sizes() {
    use zensim::feature_v2::{V1PoolsMode, V2NewFeatureToggles, V2Scratch};
    // `fold944_full` as `benches/extract_paths_bench.rs` defines it: the one
    // product mode (all pools live) plus the append and append2 blocks.
    let toggles = V2NewFeatureToggles {
        v1_pools: V1PoolsMode::Full,
        append_block: true,
        append2_block: true,
        ..Default::default()
    };
    let cells = common::parity_cells::pool_sweep_cells();
    assert!(cells.len() >= 18, "geometry set shrank to {}", cells.len());
    for &(w, h) in &cells {
        let (r, d) = pair(w, h);
        let mut first: Option<Vec<u64>> = None;
        for threads in [1usize, 2, 3, 8, 16] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("build rayon pool");
            let bits: Vec<u64> = pool.install(|| {
                let z = Zensim::new(common::legacy_full_profile()).with_parallel(true);
                let mut scratch = V2Scratch::new();
                let v2 = z
                    .compute_folded720_features_streaming(
                        &RgbSlice::new(&r, w, h),
                        &RgbSlice::new(&d, w, h),
                        toggles,
                        &mut scratch,
                    )
                    .expect("944 fold walk");
                v2.features().iter().map(|v| v.to_bits()).collect()
            });
            assert_eq!(bits.len(), 944, "{w}x{h} @ {threads}T: width moved");
            match &first {
                None => first = Some(bits),
                Some(b0) => {
                    if let Some(i) = (0..944).find(|&i| b0[i] != bits[i]) {
                        panic!(
                            "{w}x{h}: f{i} moved between 1 and {threads} threads \
                             ({:e} -> {:e})",
                            f64::from_bits(b0[i]),
                            f64::from_bits(bits[i])
                        );
                    }
                }
            }
        }
    }
}
