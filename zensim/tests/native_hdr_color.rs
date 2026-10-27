//! Public HDR color-basis gates with independent f64 colorimetry.
#![cfg(feature = "feature-regime-v2")]
use zensim::feature_v2::{HdrEncoding, V2NewFeatureToggles, V2Scratch};
use zensim::{
    AlphaMode, ColorPrimaries, ImageSource, PixelFormat, StridedBytes, Zensim, ZensimProfile,
};

#[path = "fixtures/hdr_pairs.rs"]
#[allow(dead_code)]
mod hdr_pairs;

fn matrix(p: ColorPrimaries) -> [[f64; 3]; 3] {
    match p {
        ColorPrimaries::Srgb => [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        ColorPrimaries::DisplayP3 => [
            [1.2249401763, -0.2249401763, 0.],
            [-0.0420569547, 1.0420569547, 0.],
            [-0.0196375546, -0.0786360456, 1.0982736001],
        ],
        ColorPrimaries::Bt2020 => [
            [1.6604910021, -0.5876411388, -0.0728498633],
            [-0.1245504745, 1.1328998971, -0.0083494226],
            [-0.0181507634, -0.100578898, 1.1187296614],
        ],
        _ => unreachable!(),
    }
}
// Materialize the independently checked f32 matrix representation before
// comparing downstream features. Higher-order moments on tiny pyramids can
// amplify sub-ULP color differences; they are not a pixel-color error bound.
fn transform(pixels: &[[f32; 4]], primaries: ColorPrimaries) -> Vec<[f32; 4]> {
    let exact = matrix(primaries);
    let m: [[f32; 3]; 3] = match primaries {
        ColorPrimaries::Srgb => [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        ColorPrimaries::DisplayP3 => [
            [1.224_940_2, -0.224_940_2, 0.],
            [-0.042_056_955, 1.042_056_9, 0.],
            [-0.019_637_555, -0.078_636_04, 1.098_273_6],
        ],
        ColorPrimaries::Bt2020 => [
            [1.660_491, -0.587_641_1, -0.072_849_9],
            [-0.124_550_5, 1.132_899_9, -0.008_349_4],
            [-0.018_151, -0.100_578_6, 1.118_729_6],
        ],
        _ => unreachable!(),
    };
    pixels
        .iter()
        .map(|p| {
            let c: [f32; 3] = core::array::from_fn(|c| {
                let actual = m[c][0] * p[0] + m[c][1] * p[1] + m[c][2] * p[2];
                let expected: f64 = (0..3).map(|i| exact[c][i] * f64::from(p[i])).sum();
                assert!(
                    (f64::from(actual) - expected).abs()
                        < 2e-6
                            * p[..3]
                                .iter()
                                .map(|v| f64::from(v.abs()))
                                .fold(1.0, f64::max)
                );
                actual
            });
            [c[0], c[1], c[2], 1.]
        })
        .collect()
}

/// Independent f64 display light for native PQ/HLG codes on the default
/// display (1000-nit peak, black plus reflected-ambient lift), expressed in
/// sRGB primaries through the checked f32 matrix.
fn independent_display_light(
    pixels: &[[u16; 4]],
    encoding: HdrEncoding,
    prim: ColorPrimaries,
) -> Vec<[f32; 4]> {
    let lift = 0.005 + 250. * 0.005 / std::f64::consts::PI;
    let v: Vec<[f32; 4]> = pixels
        .iter()
        .map(|p| {
            let c: [f64; 3] = core::array::from_fn(|i| f64::from(p[i]) / 65535.);
            let rgb = match encoding {
                HdrEncoding::Pq { .. } => c.map(|v| {
                    let t = v.powf(32. / 2523.);
                    let l = 10000.
                        * ((t - 3424. / 4096.).max(0.) / (2413. / 128. - 2392. / 128. * t))
                            .powf(16384. / 2610.);
                    l.min(1000.) + lift
                }),
                HdrEncoding::Hlg { .. } => {
                    let c = c.map(|v| {
                        if v <= 0.5 {
                            v * v / 3.
                        } else {
                            ((v - 0.55991073) / 0.17883277).exp() / 12.
                                + (1. - 4. * 0.17883277) / 12.
                        }
                    });
                    let y = match prim {
                        ColorPrimaries::Srgb => [0.2126390059, 0.7151686788, 0.0721923154],
                        ColorPrimaries::DisplayP3 => [0.2289745641, 0.6917385218, 0.0792869141],
                        _ => [0.2627, 0.6780, 0.0593],
                    };
                    let l = (0..3).map(|i| y[i] * c[i]).sum::<f64>();
                    c.map(|v| 1000. * l.powf(0.2) * v + lift)
                }
                _ => unreachable!(),
            };
            [rgb[0] as f32, rgb[1] as f32, rgb[2] as f32, 1.]
        })
        .collect();
    transform(&v, prim)
}

fn source(pixels: &[[f32; 4]], w: usize, h: usize, p: ColorPrimaries) -> StridedBytes<'_> {
    StridedBytes::with_alpha_mode(
        bytemuck::cast_slice(pixels),
        w,
        h,
        w * 16,
        PixelFormat::LinearF32Rgba,
        AlphaMode::Opaque,
    )
    .with_color_primaries(p)
}
fn features(a: &impl ImageSource, b: &impl ImageSource, encoding: HdrEncoding) -> Vec<f64> {
    Zensim::new(ZensimProfile::B)
        .with_parallel(false)
        .compute_folded720_append_features_hdr(
            a,
            b,
            encoding,
            V2NewFeatureToggles::default(),
            &mut V2Scratch::new(),
        )
        .unwrap()
        .features()
        .to_vec()
}
#[test]
fn hdr_public_features_match_independent_common_primaries_without_sdr_clipping() {
    for (w, h) in [(17, 9), (97, 65)] {
        let (a, b) = hdr_pairs::native_color(w, h);
        for p in [ColorPrimaries::DisplayP3, ColorPrimaries::Bt2020] {
            let ar = transform(&a, p);
            let br = transform(&b, p);
            let actual = features(
                &source(&a, w, h, p),
                &source(&b, w, h, p),
                HdrEncoding::Linear,
            );
            let expected = features(
                &source(&ar, w, h, ColorPrimaries::Srgb),
                &source(&br, w, h, ColorPrimaries::Srgb),
                HdrEncoding::Linear,
            );
            for (id, (x, y)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    x.is_finite() && (x - y).abs() < 2e-4 * y.abs().max(1.),
                    "{p:?} {w}x{h} f{id}: {x} vs {y}"
                );
            }
            #[cfg(feature = "custom-profiles")]
            {
                let recipe = serde_json::json!({
                    "schema_hash":1, "scaler_mean":[0.], "scaler_scale":[1.],
                    "metadata":[
                        {"key":"zentrain.feature_ids","type":"utf8","text":"9"},
                        {"key":"zentrain.formula_revision","type":"utf8","text":std::env::var("ZENSIM_FORMULA_REV").unwrap_or_else(|_|"1".into())}
                    ],
                    "layers":[{"in_dim":1,"out_dim":1,"activation":"identity","dtype":"f32","weights":[-10.],"biases":[100.]}]
                });
                let bytes = zenpredict_bake::bake_from_json_str(&recipe.to_string()).unwrap();
                let model = zenpredict::Model::from_bytes(&bytes).unwrap();
                let mut scorer = zensim::BakeScorer::new(&model)
                    .unwrap()
                    .with_parallel(false);
                let native = scorer
                    .compute_hdr(
                        &source(&a, w, h, p),
                        &source(&b, w, h, p),
                        HdrEncoding::Linear,
                        None,
                    )
                    .unwrap();
                let common = scorer
                    .compute_hdr(
                        &source(&ar, w, h, ColorPrimaries::Srgb),
                        &source(&br, w, h, ColorPrimaries::Srgb),
                        HdrEncoding::Linear,
                        None,
                    )
                    .unwrap();
                let cached = scorer
                    .score_features(&actual, w as u32, h as u32, None)
                    .unwrap();
                assert_eq!(native.to_bits(), common.to_bits());
                assert_eq!(native.to_bits(), cached.to_bits());
            }
            assert!(actual[9] > 0., "absolute-light error was clipped away");
            let mixed = features(
                &source(&a, w, h, p),
                &source(&a, w, h, ColorPrimaries::Srgb),
                HdrEncoding::Linear,
            );
            assert!(
                mixed[9] > 1e-6,
                "identical codes in distinct primaries were treated as identical colors"
            );
        }
    }
}

#[test]
fn pq_and_hlg_native_u16_agree_with_independent_display_light() {
    let (w, h) = (97, 65);
    let a: Vec<[u16; 4]> = (0..w * h)
        .map(|i| {
            [
                10000 + (i * 37 % 35000) as u16,
                15000 + (i * 19 % 30000) as u16,
                12000 + (i * 11 % 28000) as u16,
                65535,
            ]
        })
        .collect();
    let b: Vec<_> = a
        .iter()
        .enumerate()
        .map(|(i, p)| [p[0] - 400, p[1] + (i % 500) as u16, p[2] - 200, 65535])
        .collect();
    for prim in [
        ColorPrimaries::Srgb,
        ColorPrimaries::DisplayP3,
        ColorPrimaries::Bt2020,
    ] {
        for encoding in [
            HdrEncoding::Pq { peak_nits: 1000. },
            HdrEncoding::Hlg {
                peak_nits: 1000.,
                ambient_lux: 5.,
            },
        ] {
            let decode = |pixels: &[[u16; 4]]| independent_display_light(pixels, encoding, prim);
            let ar = decode(&a);
            let br = decode(&b);
            let native = |p: &[[u16; 4]]| {
                features(
                    &StridedBytes::with_alpha_mode(
                        bytemuck::cast_slice(&a),
                        w,
                        h,
                        w * 8,
                        PixelFormat::Srgb16Rgba,
                        AlphaMode::Opaque,
                    )
                    .with_color_primaries(prim),
                    &StridedBytes::with_alpha_mode(
                        bytemuck::cast_slice(p),
                        w,
                        h,
                        w * 8,
                        PixelFormat::Srgb16Rgba,
                        AlphaMode::Opaque,
                    )
                    .with_color_primaries(prim),
                    encoding,
                )
            };
            let actual = native(&b);
            let to_float = |pixels: &[[u16; 4]]| -> Vec<[f32; 4]> {
                pixels
                    .iter()
                    .map(|p| {
                        [
                            f32::from(p[0]) * (1.0 / 65535.0),
                            f32::from(p[1]) * (1.0 / 65535.0),
                            f32::from(p[2]) * (1.0 / 65535.0),
                            1.,
                        ]
                    })
                    .collect()
            };
            let (af, bf) = (to_float(&a), to_float(&b));
            let float_codes =
                features(&source(&af, w, h, prim), &source(&bf, w, h, prim), encoding);
            assert_eq!(
                actual, float_codes,
                "native u16 vs float code features {encoding:?} {prim:?}"
            );
            let expected = features(
                &source(&ar, w, h, ColorPrimaries::Srgb),
                &source(&br, w, h, ColorPrimaries::Srgb),
                HdrEncoding::Linear,
            );
            for (id, (x, y)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    x.is_finite() && (x - y).abs() < 1e-3 * y.abs().max(1.),
                    "{prim:?} {encoding:?} f{id}: {x} vs {y}"
                );
            }
        }
    }
}

#[test]
fn hdr_invalid_display_parameters_refuse_before_scoring() {
    let pixels = vec![[1., 2., 3., 1.]; 16 * 16];
    let src = source(&pixels, 16, 16, ColorPrimaries::Bt2020);
    for encoding in [
        HdrEncoding::Pq { peak_nits: 0. },
        HdrEncoding::Pq {
            peak_nits: f32::NAN,
        },
        HdrEncoding::Hlg {
            peak_nits: 1000.,
            ambient_lux: -1.,
        },
        HdrEncoding::Hlg {
            peak_nits: f32::INFINITY,
            ambient_lux: 5.,
        },
    ] {
        assert!(
            Zensim::new(ZensimProfile::B)
                .compute_folded720_append_features_hdr(
                    &src,
                    &src,
                    encoding,
                    V2NewFeatureToggles::default(),
                    &mut V2Scratch::new()
                )
                .is_err()
        );
    }
}

#[test]
fn public_wide_extraction_refuses_mixed_formula_revisions() {
    let p = vec![[20., 40., 10., 1.]; 17 * 9];
    let src = source(&p, 17, 9, ColorPrimaries::Bt2020);
    let current = zensim::feature_v2::active_formula_revision();
    let toggles = V2NewFeatureToggles {
        formula_revision: if current == zensim::feature_v2::FormulaRevision::Rev3 {
            zensim::feature_v2::FormulaRevision::Rev1
        } else {
            zensim::feature_v2::FormulaRevision::Rev3
        },
        ..V2NewFeatureToggles::default()
    };
    let z = Zensim::new(ZensimProfile::B);
    assert!(
        z.compute_folded720_append_features_hdr(
            &src,
            &src,
            HdrEncoding::Linear,
            toggles,
            &mut V2Scratch::new()
        )
        .is_err()
    );
}

#[cfg(feature = "custom-profiles")]
#[test]
fn prepared_native_maps_match_canonical_features_and_survive_failed_calls() {
    let revision = format!("{:?}", zensim::feature_v2::active_formula_revision());
    let revision = revision.trim_start_matches("Rev");
    let ids: Vec<usize> = (0..228).collect();
    let recipe = serde_json::json!({
        "schema_hash":1, "scaler_mean":vec![0.;228], "scaler_scale":vec![1.;228],
        "metadata":[
            {"key":"zentrain.feature_ids","type":"utf8","text":ids.iter().map(usize::to_string).collect::<Vec<_>>().join(" ")},
            {"key":"zentrain.formula_revision","type":"utf8","text":revision}
        ],
        "layers":[{"in_dim":228,"out_dim":1,"activation":"identity","dtype":"f32",
            "weights":(0..228).map(|i| -(i as f32+1.)/100.).collect::<Vec<_>>(),"biases":[100.]}]
    });
    let bytes = zenpredict_bake::bake_from_json_str(&recipe.to_string()).unwrap();
    let model = zenpredict::Model::from_bytes(&bytes).unwrap();
    for (w, h) in [(17, 9), (97, 65), (128, 129)] {
        for encoding in [
            HdrEncoding::Linear,
            HdrEncoding::Pq { peak_nits: 1000. },
            HdrEncoding::Hlg {
                peak_nits: 1000.,
                ambient_lux: 5.,
            },
        ] {
            let factor = if matches!(encoding, HdrEncoding::Linear) {
                1000.
            } else {
                1.
            };
            let a: Vec<[f32; 4]> = (0..w * h)
                .map(|i| {
                    [
                        factor * (0.05 + (i * 17 % 233) as f32 / 256.),
                        factor * (0.01 + (i * 37 % 239) as f32 / 256.),
                        factor * (0.02 + (i * 53 % 241) as f32 / 256.),
                        1.,
                    ]
                })
                .collect();
            let mut b = a.clone();
            for y in h / 3..h * 2 / 3 {
                for x in w / 3..w * 2 / 3 {
                    b[y * w + x][0] *= 0.8;
                    b[y * w + x][2] *= 0.9;
                }
            }
            let src = source(&a, w, h, ColorPrimaries::Bt2020);
            let dst = source(&b, w, h, ColorPrimaries::Bt2020);
            let canonical = Zensim::new(ZensimProfile::B)
                .with_parallel(false)
                .compute_folded720_append_features_hdr(
                    &src,
                    &dst,
                    encoding,
                    V2NewFeatureToggles {
                        v1_pools: zensim::feature_v2::V1PoolsMode::Full,
                        ..Default::default()
                    },
                    &mut V2Scratch::new(),
                )
                .unwrap()
                .features()
                .to_vec();
            let expected = zensim::BakeScorer::new(&model)
                .unwrap()
                .with_parallel(false)
                .compute_hdr(&src, &dst, encoding, None)
                .unwrap();
            if !matches!(encoding, HdrEncoding::Linear) {
                let codes = |pixels: &[[f32; 4]]| {
                    pixels
                        .iter()
                        .map(|p| {
                            [
                                (p[0] * 65535.).round() as u16,
                                (p[1] * 65535.).round() as u16,
                                (p[2] * 65535.).round() as u16,
                                65535,
                            ]
                        })
                        .collect::<Vec<_>>()
                };
                let (a16, b16) = (codes(&a), codes(&b));
                let src16 = StridedBytes::with_alpha_mode(
                    bytemuck::cast_slice(&a16),
                    w,
                    h,
                    w * 8,
                    PixelFormat::Srgb16Rgba,
                    AlphaMode::Opaque,
                )
                .with_color_primaries(ColorPrimaries::Bt2020);
                let dst16 = StridedBytes::with_alpha_mode(
                    bytemuck::cast_slice(&b16),
                    w,
                    h,
                    w * 8,
                    PixelFormat::Srgb16Rgba,
                    AlphaMode::Opaque,
                )
                .with_color_primaries(ColorPrimaries::Bt2020);
                let mut scalar = zensim::BakeScorer::new(&model)
                    .unwrap()
                    .with_parallel(false);
                let expected16 = scalar.compute_hdr(&src16, &dst16, encoding, None).unwrap();
                let canonical16 = Zensim::new(ZensimProfile::B)
                    .with_parallel(false)
                    .compute_folded720_append_features_hdr(
                        &src16,
                        &dst16,
                        encoding,
                        V2NewFeatureToggles {
                            v1_pools: zensim::feature_v2::V1PoolsMode::Full,
                            ..Default::default()
                        },
                        &mut V2Scratch::new(),
                    )
                    .unwrap();
                let mut worker = scalar.prepare_steering_hdr(&src16, encoding, 8).unwrap();
                let actual16 = worker.compute(&dst16, None).unwrap();
                assert_eq!(actual16.result().score(), expected16);
                for id in 0..228 {
                    assert!(
                        (actual16.result().features()[id] - canonical16.features()[id]).abs()
                            <= 2e-9 * canonical16.features()[id].abs().max(1.),
                        "native16 {encoding:?} {w}x{h} f{id}"
                    );
                }
            }
            for parallel in [false, true] {
                for bin in [1, 8] {
                    let mut scorer = zensim::BakeScorer::new(&model)
                        .unwrap()
                        .with_parallel(parallel)
                        .with_finite_moment_refinement(true);
                    let mut worker = scorer.prepare_steering_hdr(&src, encoding, bin).unwrap();
                    let first = worker.compute(&dst, None).unwrap();
                    for (id, &value) in canonical[..228].iter().enumerate() {
                        let actual = first.result().features()[id];
                        assert!(
                            (actual - value).abs() <= 2e-9 * value.abs().max(1.),
                            "{encoding:?} {w}x{h} f{id}: {actual} vs {value}"
                        );
                    }
                    assert_eq!(first.result().score(), expected, "{encoding:?} {w}x{h}");
                    assert!(first.unsupported_refinement_feature_ids().is_empty());
                    assert!(first.refinement_gain(0, 0, w, h).is_finite());
                    let identity = worker.compute(&src, None).unwrap();
                    assert_eq!(identity.result().score(), 100.);
                    assert_eq!(identity.refinement_gain(0, 0, w, h), 0.);
                    let bad = [[0u8; 3]; 64];
                    assert!(
                        worker
                            .compute(&zensim::RgbSlice::new(&bad, 8, 8), None)
                            .is_err()
                    );
                    let again = worker.compute(&dst, None).unwrap();
                    assert_eq!(first.result().features(), again.result().features());
                    assert_eq!(
                        first.refinement_gain(0, 0, w, h),
                        again.refinement_gain(0, 0, w, h)
                    );
                }
            }
        }
        let a = vec![[130u8, 111, 95]; w * h];
        let mut b = a.clone();
        b[w * h / 2] = [100, 107, 105];
        let src = zensim::RgbSlice::new(&a, w, h);
        let dst = zensim::RgbSlice::new(&b, w, h);
        let mut scorer = zensim::BakeScorer::new(&model)
            .unwrap()
            .with_parallel(false);
        let expected = scorer.compute(&src, &dst, None).unwrap();
        let mut worker = scorer.prepare_steering(&src, 8).unwrap();
        let actual = worker.compute(&dst, None).unwrap();
        for id in 0..228 {
            assert!(
                (actual.result().features()[id] - expected.features()[id]).abs()
                    <= 2e-9 * expected.features()[id].abs().max(1.),
                "SDR {w}x{h} f{id}"
            );
        }
        assert_eq!(actual.result().score(), expected.score());
        drop(worker);
        // Public session scratch may alternate between candidate and legacy
        // folded calls; retained candidate signals must not leak into a new pair.
        let mut scratch = zensim::Fused944Session::new();
        let pre = scorer.precompute_reference(&src).unwrap();
        scorer
            .compute_with_ref_and_attribution(&src, &pre, &dst, None, &mut scratch, 8)
            .unwrap();
        let z = Zensim::new(ZensimProfile::B).with_parallel(false);
        let legacy_pre = z.precompute_reference(&src).unwrap();
        let sensitivities = vec![-1.; 944];
        let reused = z
            .compute_folded944_score_and_attribution_binned(
                &src,
                &legacy_pre,
                &src,
                &sensitivities,
                &mut scratch,
                8,
            )
            .unwrap();
        let fresh = z
            .compute_folded944_score_and_attribution_binned(
                &src,
                &legacy_pre,
                &src,
                &sensitivities,
                &mut zensim::Fused944Session::new(),
                8,
            )
            .unwrap();
        assert_eq!(reused.0.features(), fresh.0.features());
        assert_eq!(
            reused.2.query_rect(0, 0, w, h),
            fresh.2.query_rect(0, 0, w, h)
        );
    }
}

/// The append-family tests above intentionally cover their legacy contract.
/// Rev7 admits basic/peaks/v2 instead: exercise the supported 720 entry without
/// enabling append families or changing the legacy expectations.
#[test]
fn rev7_native_u16_preserves_one_code_steps_through_supported_hdr_entry() {
    fn native(p: &[[u16; 4]], w: usize, h: usize, prim: ColorPrimaries) -> StridedBytes<'_> {
        StridedBytes::with_alpha_mode(
            bytemuck::cast_slice(p),
            w,
            h,
            w * 8,
            PixelFormat::Srgb16Rgba,
            AlphaMode::Opaque,
        )
        .with_color_primaries(prim)
    }
    use zensim::feature_v2::FormulaRevision;
    // This test is independently invoked under Rev7 by mlic-native-input-tests.
    // Explicitly bind its arithmetic, rather than silently testing the default.
    let revision = zensim::feature_v2::active_formula_revision();
    if revision != FormulaRevision::Rev7 {
        // Exercise this revision in a child so the ordinary suite still covers it.
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .env("ZENSIM_FORMULA_REV", "7")
            .args([
                "--exact",
                "rev7_native_u16_preserves_one_code_steps_through_supported_hdr_entry",
                "--test-threads=1",
            ])
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    let (w, h) = (97, 65);
    let a: Vec<[u16; 4]> = (0..w * h)
        .map(|i| {
            [
                (i * 37 % 65535) as u16,
                (i * 19 % 65535) as u16,
                (i * 11 % 65535) as u16,
                65535,
            ]
        })
        .collect();
    let b: Vec<_> = a.iter().map(|p| [p[0] + 1, p[1], p[2], p[3]]).collect();
    let as_float = |pixels: &[[u16; 4]]| -> Vec<[f32; 4]> {
        pixels
            .iter()
            .map(|p| {
                [
                    f32::from(p[0]) * (1.0 / 65535.0),
                    f32::from(p[1]) * (1.0 / 65535.0),
                    f32::from(p[2]) * (1.0 / 65535.0),
                    1.,
                ]
            })
            .collect()
    };
    let (af, bf) = (as_float(&a), as_float(&b));
    for prim in [
        ColorPrimaries::Srgb,
        ColorPrimaries::DisplayP3,
        ColorPrimaries::Bt2020,
    ] {
        for encoding in [
            HdrEncoding::Pq { peak_nits: 1000. },
            HdrEncoding::Hlg {
                peak_nits: 1000.,
                ambient_lux: 5.,
            },
        ] {
            let z = Zensim::new(ZensimProfile::B).with_parallel(false);
            let toggles = V2NewFeatureToggles::default();
            let actual = z
                .compute_folded720_features_hdr(
                    &native(&a, w, h, prim),
                    &native(&b, w, h, prim),
                    encoding,
                    toggles,
                    &mut V2Scratch::new(),
                )
                .unwrap();
            let expected = z
                .compute_folded720_features_hdr(
                    &source(&af, w, h, prim),
                    &source(&bf, w, h, prim),
                    encoding,
                    toggles,
                    &mut V2Scratch::new(),
                )
                .unwrap();
            assert_eq!(
                actual.features(),
                expected.features(),
                "{encoding:?} {prim:?}"
            );
            assert!(actual.features().iter().all(|v| v.is_finite()));
            let identity = z
                .compute_folded720_features_hdr(
                    &native(&a, w, h, prim),
                    &native(&a, w, h, prim),
                    encoding,
                    toggles,
                    &mut V2Scratch::new(),
                )
                .unwrap();
            assert_ne!(
                actual.features(),
                identity.features(),
                "one u16 code step disappeared: {encoding:?} {prim:?}"
            );
        }
    }
}

/// Rev7's supported HDR entry decodes native PQ/HLG codes to the same display
/// light as the independent f64 reference, in every tested container, within
/// the legacy test's unchanged tolerance.
#[test]
fn rev7_native_pq_hlg_match_independent_display_light_through_supported_hdr_entry() {
    use zensim::feature_v2::FormulaRevision;
    if zensim::feature_v2::active_formula_revision() != FormulaRevision::Rev7 {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .env("ZENSIM_FORMULA_REV", "7")
            .args([
                "--exact",
                "rev7_native_pq_hlg_match_independent_display_light_through_supported_hdr_entry",
                "--test-threads=1",
            ])
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    let (w, h) = (97, 65);
    let a: Vec<[u16; 4]> = (0..w * h)
        .map(|i| {
            [
                10000 + (i * 37 % 35000) as u16,
                15000 + (i * 19 % 30000) as u16,
                12000 + (i * 11 % 28000) as u16,
                65535,
            ]
        })
        .collect();
    let b: Vec<_> = a
        .iter()
        .enumerate()
        .map(|(i, p)| [p[0] - 400, p[1] + (i % 500) as u16, p[2] - 200, 65535])
        .collect();
    fn native(p: &[[u16; 4]], w: usize, h: usize, prim: ColorPrimaries) -> StridedBytes<'_> {
        StridedBytes::with_alpha_mode(
            bytemuck::cast_slice(p),
            w,
            h,
            w * 8,
            PixelFormat::Srgb16Rgba,
            AlphaMode::Opaque,
        )
        .with_color_primaries(prim)
    }
    let z = Zensim::new(ZensimProfile::B).with_parallel(false);
    let toggles = V2NewFeatureToggles::default();
    for prim in [
        ColorPrimaries::Srgb,
        ColorPrimaries::DisplayP3,
        ColorPrimaries::Bt2020,
    ] {
        for encoding in [
            HdrEncoding::Pq { peak_nits: 1000. },
            HdrEncoding::Hlg {
                peak_nits: 1000.,
                ambient_lux: 5.,
            },
        ] {
            let actual = z
                .compute_folded720_features_hdr(
                    &native(&a, w, h, prim),
                    &native(&b, w, h, prim),
                    encoding,
                    toggles,
                    &mut V2Scratch::new(),
                )
                .unwrap();
            let (ar, br) = (
                independent_display_light(&a, encoding, prim),
                independent_display_light(&b, encoding, prim),
            );
            let expected = z
                .compute_folded720_features_hdr(
                    &source(&ar, w, h, ColorPrimaries::Srgb),
                    &source(&br, w, h, ColorPrimaries::Srgb),
                    HdrEncoding::Linear,
                    toggles,
                    &mut V2Scratch::new(),
                )
                .unwrap();
            assert_eq!(actual.features().len(), expected.features().len());
            for (id, (x, y)) in actual
                .features()
                .iter()
                .zip(expected.features())
                .enumerate()
            {
                assert!(
                    x.is_finite() && (x - y).abs() < 1e-3 * y.abs().max(1.),
                    "Rev7 {prim:?} {encoding:?} f{id}: {x} vs {y}"
                );
            }
        }
    }
}

/// Rev7 display model at the declared PQ peak: code differences entirely above
/// the 1000-nit peak clip to the same display light (identical features), while
/// a step below the peak remains visible.
#[test]
fn rev7_pq_steps_above_the_declared_peak_clip_and_below_it_stay_visible() {
    use zensim::feature_v2::FormulaRevision;
    if zensim::feature_v2::active_formula_revision() != FormulaRevision::Rev7 {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .env("ZENSIM_FORMULA_REV", "7")
            .args([
                "--exact",
                "rev7_pq_steps_above_the_declared_peak_clip_and_below_it_stay_visible",
                "--test-threads=1",
            ])
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    fn native(p: &[[u16; 4]], w: usize, h: usize) -> StridedBytes<'_> {
        StridedBytes::with_alpha_mode(
            bytemuck::cast_slice(p),
            w,
            h,
            w * 8,
            PixelFormat::Srgb16Rgba,
            AlphaMode::Opaque,
        )
        .with_color_primaries(ColorPrimaries::Bt2020)
    }
    // 1000 nits is PQ code 0.7518 of full scale (about 49,271 of 65,535): every
    // channel of the right third sits above 1,500 nits (code 52,400 and up), the
    // rest below 400 nits (code 42,299 and down) even after the +300 step.
    let (w, h) = (96, 64);
    let a: Vec<[u16; 4]> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let base = if x >= 2 * w / 3 {
                54000 + (x * 37 + y * 11) % 6000
            } else {
                20000 + (x * 211 + y * 97) % 22000
            };
            [
                base as u16,
                (base - 800) as u16,
                (base - 1600) as u16,
                65535,
            ]
        })
        .collect();
    let lift = |above: bool| -> Vec<[u16; 4]> {
        a.iter()
            .enumerate()
            .map(|(i, p)| {
                let x = i % w;
                if (x >= 2 * w / 3) == above {
                    [p[0] + 300, p[1] + 300, p[2] + 300, p[3]]
                } else {
                    *p
                }
            })
            .collect()
    };
    let z = Zensim::new(ZensimProfile::B).with_parallel(false);
    let toggles = V2NewFeatureToggles::default();
    let encoding = HdrEncoding::Pq { peak_nits: 1000. };
    let features = |b: &[[u16; 4]]| {
        z.compute_folded720_features_hdr(
            &native(&a, w, h),
            &native(b, w, h),
            encoding,
            toggles,
            &mut V2Scratch::new(),
        )
        .unwrap()
        .features()
        .to_vec()
    };
    let identity = features(&a);
    assert_eq!(
        features(&lift(true)),
        identity,
        "steps entirely above the declared peak must clip to the same display light"
    );
    let below = features(&lift(false));
    assert!(below.iter().all(|v| v.is_finite()));
    assert_ne!(
        below, identity,
        "a step below the declared peak must stay visible"
    );
}

/// Rev7's supported HDR entry reads strided native rows exactly as packed ones:
/// padded rows (with garbage in the padding) give bit-identical features.
#[test]
fn rev7_native_hdr_strided_rows_match_packed_rows() {
    use zensim::feature_v2::FormulaRevision;
    if zensim::feature_v2::active_formula_revision() != FormulaRevision::Rev7 {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .env("ZENSIM_FORMULA_REV", "7")
            .args([
                "--exact",
                "rev7_native_hdr_strided_rows_match_packed_rows",
                "--test-threads=1",
            ])
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    let (w, h, pad) = (97usize, 65usize, 5usize);
    let packed = |seed: usize| -> Vec<[u16; 4]> {
        (0..w * h)
            .map(|i| {
                [
                    (8000 + (i * 37 + seed) % 40000) as u16,
                    (9000 + (i * 19 + seed) % 38000) as u16,
                    (7000 + (i * 11 + seed) % 36000) as u16,
                    65535,
                ]
            })
            .collect()
    };
    let padded = |p: &[[u16; 4]]| -> Vec<[u16; 4]> {
        let mut out = Vec::with_capacity((w + pad) * h);
        for row in p.chunks_exact(w) {
            out.extend_from_slice(row);
            out.extend((0..pad).map(|k| [(k * 7919) as u16, 65535, (k * 104729) as u16, 0]));
        }
        out
    };
    let (a, b) = (packed(0), packed(13));
    let (ap, bp) = (padded(&a), padded(&b));
    fn view(
        p: &[[u16; 4]],
        w: usize,
        h: usize,
        stride: usize,
        prim: ColorPrimaries,
    ) -> StridedBytes<'_> {
        StridedBytes::with_alpha_mode(
            bytemuck::cast_slice(p),
            w,
            h,
            stride * 8,
            PixelFormat::Srgb16Rgba,
            AlphaMode::Opaque,
        )
        .with_color_primaries(prim)
    }
    let z = Zensim::new(ZensimProfile::B).with_parallel(false);
    let toggles = V2NewFeatureToggles::default();
    for prim in [
        ColorPrimaries::Srgb,
        ColorPrimaries::DisplayP3,
        ColorPrimaries::Bt2020,
    ] {
        for encoding in [
            HdrEncoding::Pq { peak_nits: 1000. },
            HdrEncoding::Hlg {
                peak_nits: 1000.,
                ambient_lux: 5.,
            },
        ] {
            let tight = z
                .compute_folded720_features_hdr(
                    &view(&a, w, h, w, prim),
                    &view(&b, w, h, w, prim),
                    encoding,
                    toggles,
                    &mut V2Scratch::new(),
                )
                .unwrap();
            let strided = z
                .compute_folded720_features_hdr(
                    &view(&ap, w, h, w + pad, prim),
                    &view(&bp, w, h, w + pad, prim),
                    encoding,
                    toggles,
                    &mut V2Scratch::new(),
                )
                .unwrap();
            assert_eq!(
                tight.features(),
                strided.features(),
                "{encoding:?} {prim:?}"
            );
        }
    }
}
