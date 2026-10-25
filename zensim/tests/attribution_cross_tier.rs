//! Attribution-density parity under every SIMD-token permutation.
//!
//! The attribution walk (`compute_attribution_density`) bands the plane
//! differently from the streaming score path (whole-plane H-blur vs
//! 32 + 2·r-row strips), so any kernel whose tail rows carry different
//! numerics from its vector body breaks the "sum of the density equals the
//! production feature" identity — but only on tiers where the tail arithmetic
//! differs from the vector arithmetic. On this host's top tier the mismatch
//! was invisible; the i686 CI run (scalar-only dispatch) caught it in
//! `attribution::tests::sum_preservation_*`. This test runs the same
//! identities under `for_each_token_permutation` so the scalar tier is
//! exercised on every CI host, not just 32-bit.
//!
//! Tolerances mirror the unit tests in `zensim/src/attribution.rs`.
#![cfg(feature = "custom-profiles")]

mod common;

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use std::sync::OnceLock;
use zensim::profile::ProfileParams;
use zensim::{RgbSlice, Zensim, ZensimProfile};

/// Extended-features profile (all channels/scales active) — the same
/// construction as `attribution::tests::test_zensim`.
fn extended_zensim() -> Zensim {
    static PARAMS: OnceLock<ProfileParams> = OnceLock::new();
    Zensim::new(ZensimProfile::Custom {
        params: PARAMS.get_or_init(|| ProfileParams::builder().extended_features(true).build()),
        name: "attribution-cross-tier",
    })
}

/// Deterministic textured pair (blocky JPEG-ish distortion) — the same
/// construction as `attribution::tests::test_pair`.
fn test_pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let mut src = Vec::with_capacity(w * h);
    let mut dst = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let base = ((x * 255) / w) as u8;
            let tex = (((x * 7 + y * 13) % 32) * 3) as u8;
            let edge = if (y / 16) % 2 == 0 { 40 } else { 0 };
            let px = [
                base.wrapping_add(tex),
                base.wrapping_add(edge),
                (255 - base).wrapping_add(tex / 2),
            ];
            src.push(px);
            let q = |v: u8| (v / 12) * 12;
            let mut d = [q(px[0]), q(px[1]), q(px[2])];
            if x < w / 2 && y < h / 2 {
                d[0] = d[0].saturating_add(18);
            }
            dst.push(d);
        }
    }
    (src, dst)
}

#[test]
fn attribution_identities_hold_on_every_tier() {
    let mut failures: Vec<String> = Vec::new();
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let label = &perm.label;

        // Candidate max/L8 coverage through the public surface. Dispatch changes
        // are process-wide, so these probes belong in this isolated executable,
        // never the parallel library unit-test process.
        #[cfg(feature = "feature-regime-v2")]
        {
            let (w, h) = (128, 128);
            let (src, dst) = test_pair(w, h);
            let rs = RgbSlice::new(&src, w, h);
            let ds = RgbSlice::new(&dst, w, h);
            let mut session = zensim::Fused944Session::new();
            for revision in ["1", "3"] {
                for cell in 0..12 {
                    for slot in 0..6 {
                        let id = 156 + cell * 6 + slot;
                        let weight = if id % 2 == 0 { -1.0 } else { 0.75 };
                        let recipe = serde_json::json!({
                            "schema_hash": 1, "scaler_mean": [0.0], "scaler_scale": [1.0],
                            "metadata": [{"key":"zentrain.feature_ids","type":"utf8","text":id.to_string()},
                                         {"key":"zentrain.formula_revision","type":"utf8","text":revision}],
                            "layers": [{"in_dim":1,"out_dim":1,"activation":"identity",
                                        "dtype":"f32","weights":[weight],"biases":[0.0]}]
                        });
                        let bytes =
                            zenpredict_bake::bake_from_json_str(&recipe.to_string()).unwrap();
                        let model = zenpredict::Model::from_bytes(&bytes).unwrap();
                        let mut scorer = zensim::BakeScorer::new(&model).unwrap();
                        let pre = scorer.precompute_reference(&rs).unwrap();
                        let scalar = scorer.compute(&rs, &ds, None).unwrap();
                        let scored = scorer
                            .compute_with_ref_and_attribution(&rs, &pre, &ds, None, &mut session, 8)
                            .unwrap();
                        assert_eq!(scalar.score().to_bits(), scored.result().score().to_bits());
                        assert_eq!(scalar.features(), scored.result().features());
                        assert!(scored.unsupported_refinement_feature_ids().is_empty());
                        let (actual, divisor) = if slot < 3 {
                            assert_eq!(scored.unsupported_feature_ids(), &[id]);
                            assert!(scored.attribution().density().iter().all(|v| *v == 0.0));
                            (scored.refinement_gain(0, 0, w, h), 1.0)
                        } else {
                            assert!(scored.unsupported_feature_ids().is_empty());
                            assert_eq!(
                                scored.refinement_gain(0, 0, w, h),
                                scored.attribution().query_rect(0, 0, w, h)
                            );
                            // The served map's exact split gives each pixel its share
                            // of the L8 pool, so the shares sum to the feature
                            // (default since 2026-10-25); the removal split's
                            // first-order effects sum to f/8, checked below.
                            (scored.attribution().query_rect(0, 0, w, h), 1.0)
                        };
                        let expected =
                            -scored.sensitivities()[id] * scalar.features()[id] / divisor;
                        if (actual - expected).abs() > 2e-5 * expected.abs().max(1e-12) {
                            failures.push(format!(
                                "{label} candidate peak f{id}: {actual} != {expected}"
                            ));
                        }
                        if slot >= 3 {
                            let mut removal_scorer = zensim::BakeScorer::new(&model)
                                .unwrap()
                                .with_map_paint("box", "nearest", "repair")
                                .unwrap();
                            let removal_pre = removal_scorer.precompute_reference(&rs).unwrap();
                            let removal = removal_scorer
                                .compute_with_ref_and_attribution(
                                    &rs,
                                    &removal_pre,
                                    &ds,
                                    None,
                                    &mut session,
                                    8,
                                )
                                .unwrap();
                            let got = removal.attribution().query_rect(0, 0, w, h);
                            let want = -removal.sensitivities()[id] * scalar.features()[id] / 8.0;
                            if (got - want).abs() > 2e-5 * want.abs().max(1e-12) {
                                failures.push(format!(
                                    "{label} removal-split candidate peak f{id}: {got} != {want}"
                                ));
                            }
                        }
                        if slot >= 3 {
                            let finite = scorer
                                .with_finite_moment_refinement(true)
                                .compute_with_ref_and_attribution(
                                    &rs,
                                    &pre,
                                    &ds,
                                    None,
                                    &mut session,
                                    8,
                                )
                                .unwrap();
                            assert_eq!(finite.result().score().to_bits(), scalar.score().to_bits());
                            assert_eq!(finite.result().features(), scalar.features());
                            assert_eq!(
                                finite.attribution().density(),
                                scored.attribution().density()
                            );
                            for (x1, y1) in [(w, h), (w / 2, h / 2), (w / 3, h / 3)] {
                                let corrected = finite.refinement_gain(0, 0, x1, y1);
                                let old = scored.refinement_gain(0, 0, x1, y1);
                                assert!(corrected.is_finite(), "{label}: finite moment f{id}");
                                // Removing frozen nonnegative mass has curvature in
                                // the direction determined by the signed sensitivity.
                                assert!((corrected - old) * finite.sensitivities()[id] <= 0.0);
                            }
                        }
                    }
                }
            }
        }

        // Fresh vs precomputed-reference scoring: bit-identical features.
        {
            let (w, h) = (64, 64);
            let (src, dst) = test_pair(w, h);
            let z = extended_zensim();
            let rs = RgbSlice::new(&src, w, h);
            let ds = RgbSlice::new(&dst, w, h);
            let fresh = z.compute(&rs, &ds).unwrap();
            let pre = z.precompute_reference(&rs).unwrap();
            let with_ref = z.compute_with_ref(&pre, &ds).unwrap();
            if fresh.features() != with_ref.features() {
                failures.push(format!("{label}: fresh vs with_ref features differ"));
            }
        }

        // Sum preservation: the full-image density query lands on the
        // production feature (mean slots) or f/p (p-pooled slots).
        {
            let (w, h) = (64, 64);
            let (src, dst) = test_pair(w, h);
            let z = extended_zensim();
            let rs = RgbSlice::new(&src, w, h);
            let ds = RgbSlice::new(&dst, w, h);
            let feats = z.compute_extended_features(&rs, &ds).unwrap();
            let feats = feats.features();
            let mean_cases = [
                (13usize, 1e-9),     // scale 0, ch Y, slot 0 (ssim mean)
                (39 + 13, 1e-9),     // scale 1, ch Y, slot 0
                (2 * 39 + 13, 1e-9), // scale 2, ch Y, slot 0
                (9, 1e-6),           // scale 0, ch X, slot 9 (mse)
                (39 + 26 + 9, 1e-6), // scale 1, ch B, slot 9
            ];
            for &(k, tol) in &mean_cases {
                let mut s = vec![0.0f64; 156];
                s[k] = -1.0;
                let attr = z.compute_attribution_density(&rs, &ds, &s).unwrap();
                let full = attr.query_rect(0, 0, w, h);
                let expect = feats[k];
                if (full - expect).abs() > tol * expect.abs().max(1e-12) {
                    failures.push(format!(
                        "{label}: mean slot k={k}: full-image query {full} vs feature {expect}"
                    ));
                }
            }
            for &(k, p) in &[(14usize, 4.0), (15, 2.0), (18, 2.0)] {
                let mut s = vec![0.0f64; 156];
                s[k] = -1.0;
                let attr = z.compute_attribution_density(&rs, &ds, &s).unwrap();
                let full = attr.query_rect(0, 0, w, h);
                let expect = feats[k] / p;
                if (full - expect).abs() > 1e-5 * expect.abs().max(1e-12) {
                    failures.push(format!(
                        "{label}: p-pooled slot k={k}: full-image query {full} vs f/p {expect}"
                    ));
                }
            }
        }

        // Fused score+attribution equals the standalone density.
        {
            let (w, h) = (150, 170);
            let (src, dst) = test_pair(w, h);
            let z = Zensim::new(common::legacy_full_profile());
            let rs = RgbSlice::new(&src, w, h);
            let ds = RgbSlice::new(&dst, w, h);
            let pre = z.precompute_reference(&rs).unwrap();
            let mut s = vec![0.0f64; 156];
            for (k, v) in s.iter_mut().enumerate() {
                *v = if k % 3 == 0 { -1.0 } else { -0.25 } * (1.0 + (k % 7) as f64 * 0.1);
            }
            let std_attr = z.compute_attribution_density(&rs, &ds, &s).unwrap();
            let (_, fused_attr) = z
                .compute_with_ref_score_and_attribution(&pre, &ds, &s)
                .unwrap();
            let max_abs = std_attr
                .density()
                .iter()
                .fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(max_abs > 0.0);
            let budget = 3e-5 * max_abs + 1e-9;
            let (worst, worst_i) = fused_attr
                .density()
                .iter()
                .zip(std_attr.density().iter())
                .enumerate()
                .fold((0.0f32, 0usize), |(m, mi), (i, (a, b))| {
                    let d = (a - b).abs();
                    if d > m { (d, i) } else { (m, mi) }
                });
            if worst > budget {
                failures.push(format!(
                    "{label}: fused vs standalone density: |diff| {worst} at pixel {worst_i} exceeds {budget}"
                ));
            }
        }
    });
    eprintln!("{report}");
    assert!(report.permutations_run >= 1);
    assert!(failures.is_empty(), "{failures:#?}");
}
