//! Extra COSTCMP arms; all scores use the existing public serving surfaces.
use serde_json::{Value, json};
use zenpredict::Model;
use zensim::{RgbSlice, Zensim, ZensimProfile};

type Action = Box<dyn FnMut() -> f64>;
const FAST_MAIN: &str = "09ec3e7c78bd230b1545572a188e162313c59fda";

#[allow(deprecated)] // Profile A is explicitly requested as the historical control.
pub(super) fn action(
    arm: &str,
    src: &'static [[u8; 3]],
    dst: &'static [[u8; 3]],
    w: usize,
    h: usize,
) -> Option<(Action, Value)> {
    match arm {
        "e34_butter_parity" => {
            let params = butteraugli::ButteraugliParams::default().with_compute_diffmap(true);
            let source: &'static [u8] = bytemuck::cast_slice(src);
            let distorted: &'static [u8] = bytemuck::cast_slice(dst);
            let cached = butteraugli::ButteraugliReference::new(source, w, h, params.clone())
                .unwrap()
                .compare(distorted)
                .unwrap();
            let direct = butteraugli::butteraugli(
                imgref::Img::new(bytemuck::cast_slice::<[u8; 3], rgb::RGB8>(src), w, h),
                imgref::Img::new(bytemuck::cast_slice::<[u8; 3], rgb::RGB8>(dst), w, h),
                &params,
            )
            .unwrap();
            let score_delta = (cached.pnorm_3 - direct.pnorm_3).abs();
            let cm = cached.diffmap.as_ref().unwrap();
            let dm = direct.diffmap.as_ref().unwrap();
            assert_eq!((cm.width(), cm.height()), (w, h));
            assert_eq!((dm.width(), dm.height()), (w, h));
            let map_delta = cm
                .buf()
                .iter()
                .zip(dm.buf())
                .map(|(a, b)| f64::from((a - b).abs()))
                .fold(0.0_f64, f64::max);
            assert!(
                score_delta < 1e-10 && map_delta < 1e-6,
                "published FIR cache parity bounds"
            );
            let score = cached.pnorm_3;
            Some((
                Box::new(move || score),
                json!({"score_abs_delta":score_delta,
                "map_max_abs_delta":map_delta,"score_tolerance":1e-10,"map_tolerance":1e-6,
                "kind":"setup-only parity oracle; never timed or RSS-measured"}),
            ))
        }
        "e34_butter_scalar" | "e34_butter_map" | "e34_butter_baseline" => {
            let params = butteraugli::ButteraugliParams::default();
            let source: &'static [u8] = bytemuck::cast_slice(src);
            let distorted: &'static [u8] = bytemuck::cast_slice(dst);
            let info = json!({"revision":"5","peer":"butteraugli0.9.3",
                "vcs":"6e3a7bd27ef247e6bc285d9473a691facdb984f8",
                "source_sha256":"914c4086ea33acfef5c3ea38393c6f5c0ec99d32155377818f1723ec1bd565e3",
                "kind":"reference_metric",
                "score":"pnorm3","map_bin":1,"surface":arm,
                "output_retention":"complete native result until call return"});
            let action: Action = match arm {
                "e34_butter_baseline" => Box::new(move || {
                    std::hint::black_box((source, distorted));
                    0.0
                }),
                "e34_butter_map" => {
                    let reference = butteraugli::ButteraugliReference::new(
                        source,
                        w,
                        h,
                        params.with_compute_diffmap(true),
                    )
                    .unwrap();
                    Box::new(move || {
                        let result = reference.compare(distorted).unwrap();
                        assert_eq!(result.diffmap.as_ref().unwrap().width(), w);
                        assert_eq!(result.diffmap.as_ref().unwrap().height(), h);
                        let score = result.pnorm_3;
                        std::hint::black_box(result);
                        score
                    })
                }
                _ => Box::new(move || {
                    let result = butteraugli::butteraugli(
                        imgref::Img::new(bytemuck::cast_slice::<[u8; 3], rgb::RGB8>(src), w, h),
                        imgref::Img::new(bytemuck::cast_slice::<[u8; 3], rgb::RGB8>(dst), w, h),
                        &params,
                    )
                    .unwrap();
                    let score = result.pnorm_3;
                    std::hint::black_box(result);
                    score
                }),
            };
            Some((action, info))
        }
        "e34_control"
        | "e34_candidate"
        | "e34_candidate_map"
        | "e34_candidate_map_audit"
        | "e34_candidate_map_before"
        | "e34_candidate_baseline"
        | "e34_candidate_map_retained"
        | "e34_candidate_cached"
        | "e34_candidate_prepare"
        | "e34_candidate_serial"
        | "e34_candidate_map_serial"
        | "e34_candidate_map_serial_audit"
        | "e34_candidate_map_bin2"
        | "e34_candidate_map_bin8"
        | "e34_candidate_map_bin16" => {
            // E34's registered widths are explicit caller bindings. Keep the E33
            // H128 admission below unchanged; all E34 measurements use these
            // same serving surfaces and SPEEDQ's existing paired-round owner.
            let label = if arm == "e34_control" {
                "CONTROL"
            } else {
                "CANDIDATE"
            };
            let path =
                std::env::var(format!("ZEN_S2_E34_BAKE_{label}")).expect("pinned E34 model path");
            let expected =
                std::env::var(format!("ZEN_S2_E34_SHA_{label}")).expect("pinned E34 model SHA");
            let hidden: usize = std::env::var(format!("ZEN_S2_E34_HIDDEN_{label}"))
                .expect("registered E34 hidden width")
                .parse()
                .unwrap();
            assert!(matches!(hidden, 1 | 32 | 128), "registered E34 widths");
            let bytes = std::fs::read(path).unwrap();
            assert_eq!(super::digest(&bytes), expected, "E34 model bytes changed");
            assert_eq!(std::env::var("ZENSIM_FORMULA_REV").unwrap(), "5");
            let model: &'static Model = Box::leak(Box::new(Model::from_bytes(&bytes).unwrap()));
            assert_eq!(model.layers().next().unwrap().out_dim, hidden);
            assert_eq!(model.n_outputs(), 1);
            let scorer: &'static mut zensim::BakeScorer<'static> = Box::leak(Box::new(
                zensim::BakeScorer::new(model)
                    .unwrap()
                    .with_parallel(!matches!(
                        arm,
                        "e34_candidate_serial"
                            | "e34_candidate_map_serial"
                            | "e34_candidate_map_serial_audit"
                    )),
            ));
            let bin = match arm {
                "e34_candidate_map_bin2" => 2,
                "e34_candidate_map_bin8" => 8,
                "e34_candidate_map_bin16" => 16,
                "e34_candidate_map_audit" => std::env::var("ZEN_S2_MAP_BIN")
                    .map(|s| s.parse::<usize>().expect("explicit map bin"))
                    .unwrap_or(1),
                _ => 1,
            };
            let mut info = json!({"source_sha256":expected,"revision":"5","hidden":hidden,
                "n_inputs":model.n_inputs(),"source_bytes":bytes.len(),"map_bin":bin,
                "surface":arm});
            let source: &'static RgbSlice<'static> = Box::leak(Box::new(RgbSlice::new(src, w, h)));
            let action: Action = match arm {
                "e34_candidate_map_audit" | "e34_candidate_map_serial_audit" => {
                    // Untimed cross-build audit of every density/feature/gradient
                    // bit and rectangle queries, including finite neighbour replay.
                    let mut session = scorer.prepare_steering(source, bin).unwrap();
                    let result = session.compute(&RgbSlice::new(dst, w, h), None).unwrap();
                    let mut bytes = Vec::new();
                    for value in result.attribution().density() {
                        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
                    }
                    info["density_sha256"] = json!(super::digest(&bytes));
                    bytes.clear();
                    for value in result
                        .result()
                        .features()
                        .iter()
                        .chain(result.sensitivities())
                    {
                        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
                    }
                    info["features_gradient_sha256"] = json!(super::digest(&bytes));
                    info["feature_bits"] = json!(
                        result
                            .result()
                            .features()
                            .iter()
                            .map(|v| format!("{:016x}", v.to_bits()))
                            .collect::<Vec<_>>()
                    );
                    info["gradient_bits"] = json!(
                        result
                            .sensitivities()
                            .iter()
                            .map(|v| format!("{:016x}", v.to_bits()))
                            .collect::<Vec<_>>()
                    );
                    bytes.clear();
                    for block in [1usize, 8, 16, 32, 64] {
                        for y in (0..h).step_by(block) {
                            for x in (0..w).step_by(block) {
                                let v = result.attribution().query_rect(
                                    x,
                                    y,
                                    (x + block).min(w),
                                    (y + block).min(h),
                                );
                                bytes.extend_from_slice(&v.to_bits().to_le_bytes());
                            }
                        }
                    }
                    info["queries_sha256"] = json!(super::digest(&bytes));
                    bytes.clear();
                    for (x, y) in [(0, 0), (w / 2, h / 2), (w - 8, h - 8)] {
                        let v = result.refinement_gain(x, y, (x + 8).min(w), (y + 8).min(h));
                        bytes.extend_from_slice(&v.to_bits().to_le_bytes());
                    }
                    info["refinement_sha256"] = json!(super::digest(&bytes));
                    if bin != 1 {
                        let mut exact_scorer = zensim::BakeScorer::new(model).unwrap();
                        let mut exact_session = exact_scorer.prepare_steering(source, 1).unwrap();
                        let exact = exact_session
                            .compute(&RgbSlice::new(dst, w, h), None)
                            .unwrap();
                        assert_eq!(
                            exact.result().score().to_bits(),
                            result.result().score().to_bits()
                        );
                        let mut query_max = 0.0_f64;
                        let mut query_sum = 0.0_f64;
                        let mut queries = 0usize;
                        let mut refine_max = 0.0_f64;
                        for y in (3..h).step_by(17) {
                            for x in (3..w).step_by(17) {
                                let x1 = (x + 13).min(w);
                                let y1 = (y + 19).min(h);
                                let delta = (result.attribution().query_rect(x, y, x1, y1)
                                    - exact.attribution().query_rect(x, y, x1, y1))
                                .abs();
                                assert!(delta.is_finite(), "nonfinite map-query movement");
                                query_max = query_max.max(delta);
                                query_sum += delta;
                                queries += 1;
                                let refinement_delta = (result.refinement_gain(x, y, x1, y1)
                                    - exact.refinement_gain(x, y, x1, y1))
                                .abs();
                                assert!(
                                    refinement_delta.is_finite(),
                                    "nonfinite refinement movement"
                                );
                                refine_max = refine_max.max(refinement_delta);
                            }
                        }
                        info["bin_movement"] = json!({"score_bits_equal":true,
                            "unaligned_rectangles":queries,"finite_rectangles":queries,"query_max_abs":query_max,
                            "query_mean_abs":query_sum/queries as f64,
                            "refinement_max_abs":refine_max});
                    }
                    let score = result.result().score();
                    Box::new(move || score)
                }
                "e34_candidate_cached" => {
                    let reference = scorer.precompute_reference(source).unwrap();
                    let direct = scorer
                        .compute(source, &RgbSlice::new(dst, w, h), None)
                        .unwrap();
                    let cached = scorer
                        .compute_with_ref(source, &reference, &RgbSlice::new(dst, w, h), None)
                        .unwrap();
                    assert_eq!(direct.score().to_bits(), cached.score().to_bits());
                    assert_eq!(
                        direct.raw_distance().to_bits(),
                        cached.raw_distance().to_bits()
                    );
                    assert_eq!(
                        direct
                            .features()
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        cached
                            .features()
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(
                        direct.mean_offset().map(f64::to_bits),
                        cached.mean_offset().map(f64::to_bits)
                    );
                    info["cached_complete_result_parity"] = json!(true);
                    Box::new(move || {
                        let result = scorer
                            .compute_with_ref(source, &reference, &RgbSlice::new(dst, w, h), None)
                            .unwrap();
                        let score = result.score();
                        std::hint::black_box(result);
                        score
                    })
                }
                "e34_candidate_prepare" => Box::new(move || {
                    std::hint::black_box(scorer.precompute_reference(source).unwrap());
                    0.0
                }),
                "e34_candidate_baseline" => Box::new(move || {
                    std::hint::black_box(&scorer);
                    0.0
                }),
                "e34_candidate_map" => {
                    let mut session = scorer.prepare_steering(source, bin).unwrap();
                    Box::new(move || {
                        session
                            .compute(&RgbSlice::new(dst, w, h), None)
                            .unwrap()
                            .result()
                            .score()
                    })
                }
                "e34_candidate_map_retained"
                | "e34_candidate_map_serial"
                | "e34_candidate_map_before"
                | "e34_candidate_map_bin2"
                | "e34_candidate_map_bin8"
                | "e34_candidate_map_bin16" => {
                    let mut session = scorer.prepare_steering(source, bin).unwrap();
                    Box::new(move || {
                        let result = session.compute(&RgbSlice::new(dst, w, h), None).unwrap();
                        let score = result.result().score();
                        std::hint::black_box(result);
                        score
                    })
                }
                _ => Box::new(move || {
                    scorer
                        .compute(source, &RgbSlice::new(dst, w, h), None)
                        .unwrap()
                        .score()
                }),
            };
            Some((action, info))
        }
        "zensim_A" | "zensim_B" | "zensim_D" => {
            // These are the exact slots named by profile.rs and serving.rs.
            // Pin their bytes as metadata; scoring still calls Zensim::compute.
            let (profile, bytes, name): (ZensimProfile, &[u8], &str) = if arm == "zensim_A" {
                (
                    ZensimProfile::A,
                    include_bytes!(
                        "../../../zensim/weights/v47_strict_qat_native_byid_2026-09-06.bin"
                    ),
                    "A",
                )
            } else if arm == "zensim_D" {
                // The scorecard's "frozen D" runtime comparator (MODEL_SELECTION_SCORECARD).
                (
                    ZensimProfile::D,
                    include_bytes!(
                        "../../../zensim/weights/d_sdr_add156_id100_negrich_dial_byid_2026-09-06.bin"
                    ),
                    "D",
                )
            } else {
                (
                    ZensimProfile::B,
                    include_bytes!(
                        "../../../zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_byid_2026-09-06.bin"
                    ),
                    "B",
                )
            };
            let model = Model::from_bytes(bytes).unwrap();
            let declared = model.metadata().get_utf8("zentrain.formula_revision").ok();
            // Missing metadata is the registered shipped Rev1 era. No stamp.
            let revision = declared.unwrap_or("1");
            assert_eq!(revision, "1", "serving profile arithmetic changed");
            assert_eq!(std::env::var("ZENSIM_FORMULA_REV").unwrap(), revision);
            let info = json!({"source_sha256":super::digest(bytes),"revision":revision,
                "declared_revision":declared,"profile":name,"n_inputs":model.n_inputs(),
                "surface":"Zensim::compute; complete named serving profile"});
            let z = Zensim::new(profile).with_parallel(true);
            let action: Action = Box::new(move || {
                z.compute(&RgbSlice::new(src, w, h), &RgbSlice::new(dst, w, h))
                    .unwrap()
                    .score()
            });
            Some((action, info))
        }
        "fast_ssim2_main" => {
            let source = fast_ssim2_main::PixelSlice::new(
                bytemuck::cast_slice(src),
                w as u32,
                h as u32,
                w * 3,
                fast_ssim2_main::PixelDescriptor::RGB8_SRGB,
            )
            .unwrap();
            let distorted = fast_ssim2_main::PixelSlice::new(
                bytemuck::cast_slice(dst),
                w as u32,
                h as u32,
                w * 3,
                fast_ssim2_main::PixelDescriptor::RGB8_SRGB,
            )
            .unwrap();
            let action: Action = Box::new(move || {
                fast_ssim2_main::compute_ssimulacra2(&source, &distorted).unwrap()
            });
            Some((
                action,
                json!({"commit":FAST_MAIN,"version":"0.9.0","surface":"compute_ssimulacra2 RGB8 one-shot","rayon":true}),
            ))
        }
        "fast_ssim2" => {
            let action: Action = Box::new(move || {
                fast_ssim2::compute_ssimulacra2(
                    imgref::Img::new(src, w, h),
                    imgref::Img::new(dst, w, h),
                )
                .unwrap()
            });
            Some((
                action,
                json!({"version":"0.8.2","surface":"compute_ssimulacra2 RGB8 one-shot","rayon":cfg!(feature="ssim2-rayon")}),
            ))
        }
        "e33_baseline_a" | "e33_baseline_c" | "e33_baseline_seed0" => {
            // RSS baseline for the scorecard's INCREMENTAL memory bar: the same process with inputs, model and
            // scorer constructed, scoring nothing. Incremental RSS = an arm's peak minus this peak.
            let label = arm.trim_start_matches("e33_baseline_").to_uppercase();
            let path =
                std::env::var(format!("ZEN_S2_E33_BAKE_{label}")).expect("pinned candidate path");
            let expected =
                std::env::var(format!("ZEN_S2_E33_SHA_{label}")).expect("pinned candidate SHA");
            let bytes = std::fs::read(path).unwrap();
            assert_eq!(super::digest(&bytes), expected, "candidate bytes changed");
            let model: &'static Model = Box::leak(Box::new(Model::from_bytes(&bytes).unwrap()));
            let scorer = zensim::BakeScorer::new(model).unwrap().with_parallel(true);
            let info = json!({"source_sha256":expected,"revision":"5","surface":"baseline: inputs + model + BakeScorer, no scoring"});
            let action: Action = Box::new(move || {
                std::hint::black_box(&scorer);
                0.0
            });
            Some((action, info))
        }
        "e33_a_map" | "e33_c_map" | "e33_seed0_map" => {
            // Scorecard spatial cost: cached-reference complete score + map. The reference is prepared
            // once, untimed (prepare_steering, map bin 1 as in G-STEER); each call scores and maps.
            let label = arm
                .trim_start_matches("e33_")
                .trim_end_matches("_map")
                .to_uppercase();
            let path =
                std::env::var(format!("ZEN_S2_E33_BAKE_{label}")).expect("pinned candidate path");
            let expected =
                std::env::var(format!("ZEN_S2_E33_SHA_{label}")).expect("pinned candidate SHA");
            let bytes = std::fs::read(path).unwrap();
            assert_eq!(super::digest(&bytes), expected, "candidate bytes changed");
            assert_eq!(
                std::env::var("ZENSIM_FORMULA_REV").unwrap(),
                "5",
                "candidates serve Rev5"
            );
            let model: &'static Model = Box::leak(Box::new(Model::from_bytes(&bytes).unwrap()));
            let scorer: &'static mut zensim::BakeScorer<'static> = Box::leak(Box::new(
                zensim::BakeScorer::new(model).unwrap().with_parallel(true),
            ));
            let source: &'static RgbSlice<'static> = Box::leak(Box::new(RgbSlice::new(src, w, h)));
            let mut session = scorer.prepare_steering(source, 1).unwrap();
            let info = json!({"source_sha256":expected,"revision":"5","map_bin":1,"source_bytes":bytes.len(),
                "surface":"BakeScorer::prepare_steering once, then SteeringSession::compute per call (score + map)"});
            let action: Action = Box::new(move || {
                session
                    .compute(&RgbSlice::new(dst, w, h), None)
                    .unwrap()
                    .result()
                    .score()
            });
            Some((action, info))
        }
        "e33_a" | "e33_c" => {
            // E33 section 9.3: a registered full-data seed-0 candidate, pinned by path and SHA from the caller,
            // served through the same dynamic surface as production (BakeScorer, Rev5, Rayon enabled).
            let label = arm.trim_start_matches("e33_").to_uppercase();
            let path = std::env::var(format!("ZEN_S2_E33_BAKE_{label}"))
                .expect("pinned E33 candidate path");
            let expected =
                std::env::var(format!("ZEN_S2_E33_SHA_{label}")).expect("pinned E33 candidate SHA");
            let bytes = std::fs::read(path).unwrap();
            assert_eq!(
                super::digest(&bytes),
                expected,
                "E33 candidate bytes changed"
            );
            assert_eq!(
                std::env::var("ZENSIM_FORMULA_REV").unwrap(),
                "5",
                "E33 candidates serve Rev5"
            );
            let model = Box::leak(Box::new(Model::from_bytes(&bytes).unwrap()));
            assert_eq!(model.layers().next().unwrap().out_dim, 128, "H128 shape");
            assert_eq!(model.n_outputs(), 1);
            let mut scorer = zensim::BakeScorer::new(model).unwrap().with_parallel(true);
            let consumed = scorer.consumed_feature_ids().unwrap();
            let info = json!({"source_sha256":expected,"revision":"5","hidden":128,
                "consumed_ids":consumed.len(),"n_inputs":model.n_inputs(),"source_bytes":bytes.len(),
                "surface":"BakeScorer::compute; E33 registered candidate"});
            let action: Action = Box::new(move || {
                scorer
                    .compute(&RgbSlice::new(src, w, h), &RgbSlice::new(dst, w, h), None)
                    .unwrap()
                    .score()
            });
            Some((action, info))
        }
        "zensim_B_rev5" => {
            // The current default B (diff+frag-820, selected 2026-10-19): report-only context for the
            // two curated revision-7 models, served through the named profile as callers see it.
            let bytes: &[u8] =
                include_bytes!("../../../zensim/weights/b_sdr_diff_frag_820_rev5_2026-10-19.bin");
            let model = Model::from_bytes(bytes).unwrap();
            let declared = model.metadata().get_utf8("zentrain.formula_revision").ok();
            assert_eq!(declared, Some("5"), "default B arithmetic changed");
            assert_eq!(std::env::var("ZENSIM_FORMULA_REV").unwrap(), "5");
            let info = json!({"source_sha256":super::digest(bytes),"revision":"5",
                "declared_revision":declared,"profile":"B","n_inputs":model.n_inputs(),
                "surface":"Zensim::compute; current default B (report-only)"});
            let z = Zensim::new(ZensimProfile::B).with_parallel(true);
            let action: Action = Box::new(move || {
                z.compute(&RgbSlice::new(src, w, h), &RgbSlice::new(dst, w, h))
                    .unwrap()
                    .score()
            });
            Some((action, info))
        }
        "r7_general"
        | "r7_codec"
        | "r7_general_cached"
        | "r7_codec_cached"
        | "r7_general_map"
        | "r7_codec_map"
        | "r7_baseline_general"
        | "r7_baseline_codec" => {
            // The two curated revision-7 models (general: x256 + rehearsal e59; codec only: AIC-only
            // e29), pinned by path and SHA from the caller and served through BakeScorer at their
            // declared revision. Scorecard rows: scalar performance (uncached; cached reported
            // separately) and spatial/memory cost (cached-reference score + map; RSS baselines).
            let label = if arm.contains("general") {
                "GENERAL"
            } else {
                "CODEC"
            };
            let path = std::env::var(format!("ZEN_S2_R7_BAKE_{label}"))
                .expect("pinned curated model path");
            let expected =
                std::env::var(format!("ZEN_S2_R7_SHA_{label}")).expect("pinned curated model SHA");
            let bytes = std::fs::read(path).unwrap();
            assert_eq!(
                super::digest(&bytes),
                expected,
                "curated model bytes changed"
            );
            assert_eq!(
                std::env::var("ZENSIM_FORMULA_REV").unwrap(),
                "7",
                "curated models serve revision 7"
            );
            let model: &'static Model = Box::leak(Box::new(Model::from_bytes(&bytes).unwrap()));
            assert_eq!(
                model.metadata().get_utf8("zentrain.formula_revision").ok(),
                Some("7"),
                "curated model declares revision 7"
            );
            assert_eq!(model.layers().next().unwrap().out_dim, 128, "H128 shape");
            assert_eq!(model.n_outputs(), 1);
            let scorer: &'static mut zensim::BakeScorer<'static> = Box::leak(Box::new(
                zensim::BakeScorer::new(model).unwrap().with_parallel(true),
            ));
            let consumed = scorer.consumed_feature_ids().unwrap().len();
            let mut info = json!({"source_sha256":expected,"revision":"7","hidden":128,
                "consumed_ids":consumed,"n_inputs":model.n_inputs(),"source_bytes":bytes.len(),
                "surface":arm});
            let source: &'static RgbSlice<'static> = Box::leak(Box::new(RgbSlice::new(src, w, h)));
            let action: Action = if arm.starts_with("r7_baseline_") {
                // RSS baseline for the incremental memory bar: inputs, model and scorer, no scoring.
                Box::new(move || {
                    std::hint::black_box(&scorer);
                    0.0
                })
            } else if arm.ends_with("_cached") {
                let reference = scorer.precompute_reference(source).unwrap();
                let direct = scorer
                    .compute(source, &RgbSlice::new(dst, w, h), None)
                    .unwrap();
                let cached = scorer
                    .compute_with_ref(source, &reference, &RgbSlice::new(dst, w, h), None)
                    .unwrap();
                assert_eq!(direct.score().to_bits(), cached.score().to_bits());
                assert_eq!(
                    direct.raw_distance().to_bits(),
                    cached.raw_distance().to_bits()
                );
                assert_eq!(
                    direct
                        .features()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    cached
                        .features()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
                info["cached_complete_result_parity"] = json!(true);
                Box::new(move || {
                    let result = scorer
                        .compute_with_ref(source, &reference, &RgbSlice::new(dst, w, h), None)
                        .unwrap();
                    let score = result.score();
                    std::hint::black_box(result);
                    score
                })
            } else if arm.ends_with("_map") {
                // The reference is prepared once, untimed (map bin 1, as in G-STEER); each call
                // returns the complete score and map, retained until the call returns.
                info["map_bin"] = json!(1);
                let mut session = scorer.prepare_steering(source, 1).unwrap();
                Box::new(move || {
                    let result = session.compute(&RgbSlice::new(dst, w, h), None).unwrap();
                    let score = result.result().score();
                    std::hint::black_box(result);
                    score
                })
            } else {
                Box::new(move || {
                    scorer
                        .compute(source, &RgbSlice::new(dst, w, h), None)
                        .unwrap()
                        .score()
                })
            };
            Some((action, info))
        }
        _ => None,
    }
}
