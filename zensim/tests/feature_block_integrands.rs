//! `BakeScorer::feature_block_integrands` (E36 experiment 3): the per-feature block integrands, weighted by the
//! served sensitivities, reproduce the served density's block sums, and features the map cannot spatialize are
//! reported rather than returned as zero columns.
#![cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]

use zensim::RgbSlice;

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

fn linear_bake(ids: &[usize], weights: &[f32], revision: &str) -> Vec<u8> {
    let recipe = serde_json::json!({
        "schema_hash": 1, "scaler_mean": vec![0.0; ids.len()], "scaler_scale": vec![1.0; ids.len()],
        "metadata": [{"key":"zentrain.feature_ids","type":"utf8",
                      "text": ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" ")},
                     {"key":"zentrain.formula_revision","type":"utf8","text":revision}],
        "layers": [{"in_dim":ids.len(),"out_dim":1,"activation":"identity",
                    "dtype":"f32","weights":weights,"biases":[0.0]}]
    });
    zenpredict_bake::bake_from_json_str(&recipe.to_string()).unwrap()
}

#[test]
fn weighted_integrands_reproduce_the_served_density() {
    let (w, h) = (96, 80);
    let (src, dst) = test_pair(w, h);
    let rs = RgbSlice::new(&src, w, h);
    let ds = RgbSlice::new(&dst, w, h);
    // Basic (3, 40, 100) and v2 at scale 0 (380, 400), the families F3 reads; its coarse v2 (f459-719) is left
    // out because the served revision-7 density zeroes it for the neighbour-exact replay.
    let ids = [3usize, 40, 100, 380, 400];
    let weights = [-0.8f32, 0.5, -0.3, 0.4, -0.2];
    for revision in ["7"] {
        let bytes = linear_bake(&ids, &weights, revision);
        let model = zenpredict::Model::from_bytes(&bytes).unwrap();
        let mut scorer = zensim::BakeScorer::new(&model).unwrap();
        let pre = scorer.precompute_reference(&rs).unwrap();
        let mut session = zensim::Fused944Session::new();
        let served = scorer
            .compute_with_ref_and_attribution(&rs, &pre, &ds, None, &mut session, 8)
            .unwrap();
        assert!(served.unsupported_feature_ids().is_empty(), "rev {revision}");
        let expected = served.attribution().block_sums(8);
        let s = served.sensitivities().to_vec();

        let mut fresh = zensim::Fused944Session::new();
        let integrands = scorer
            .feature_block_integrands(&rs, &pre, &ds, &mut fresh, 8, &ids)
            .unwrap();
        assert_eq!(integrands.ids, ids, "rev {revision}");
        assert!(integrands.unsupported.is_empty());
        assert_eq!((integrands.blocks_x, integrands.blocks_y), (12, 10));
        let scale = expected.iter().fold(0f64, |m, v| m.max(v.abs()));
        assert!(scale > 0.0);
        for (b, want) in expected.iter().enumerate() {
            let got: f64 = integrands
                .ids
                .iter()
                .zip(&integrands.values)
                .map(|(&id, v)| s[id] * v[b])
                .sum();
            assert!(
                (got - want).abs() <= 1e-5 * scale,
                "rev {revision} block {b}: {got} vs {want}"
            );
        }
        // Each feature carries signal somewhere.
        for (id, v) in integrands.ids.iter().zip(&integrands.values) {
            assert!(v.iter().any(|x| *x != 0.0), "rev {revision} f{id} has an all-zero integrand");
        }
    }
}

#[test]
fn unspatializable_ids_are_reported() {
    let (w, h) = (64, 64);
    let (src, dst) = test_pair(w, h);
    let rs = RgbSlice::new(&src, w, h);
    let ds = RgbSlice::new(&dst, w, h);
    let bytes = linear_bake(&[3, 156], &[-0.5, -0.5], "7");
    let model = zenpredict::Model::from_bytes(&bytes).unwrap();
    let mut scorer = zensim::BakeScorer::new(&model).unwrap();
    let pre = scorer.precompute_reference(&rs).unwrap();
    let mut session = zensim::Fused944Session::new();
    // f156 is an L-infinity peak (hard maximum): refinement only, no density.
    let integrands = scorer
        .feature_block_integrands(&rs, &pre, &ds, &mut session, 16, &[3, 156])
        .unwrap();
    assert_eq!(integrands.ids, [3]);
    assert_eq!(integrands.unsupported, [156]);
    assert!(
        scorer
            .feature_block_integrands(&rs, &pre, &ds, &mut session, 0, &[3])
            .is_err()
    );
}
