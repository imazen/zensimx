// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.
//! Head-to-head of the two feature-extraction FAMILIES, paired/interleaved
//! in one process so shared-box noise cancels
//! (`benchmarks/extraction_perf_and_buffered_removal_2026-08-30.md`).
//!
//! Extraction controls per size, one interleaved group:
//!
//! | arm | path | features |
//! |---|---|---|
//! | `buf_v1_228`  | BUFFERED v1 (`compute_zensim_with_config`, whole-image pyramids) | 228 |
//! | `buf_v1_372`  | same, `extended_features` + `compute_iw_features` | 372 (156 basic + 216 pools) |
//! | `fold944_off` | STREAMING fold, `V1PoolsMode::Off` (`folded720append2`) | 944, f156..371 = structural zeros |
//! | `fold944_full`| STREAMING fold, `V1PoolsMode::Full` (`folded720append2pools`) | 944, all live |
//! | `fast_ssim2`  | the OPPONENT: `fast_ssim2::compute_ssimulacra2`, same pixels | — |
//!
//! Pool deltas price the extra accumulators. The v1-only arms also measure
//! cheap-wide raw-moment and bounded-error producers against the full walk.
//! `fold156_basic` retains raw Off as a historical diagnostic; current serving
//! promotes Off to Peaks for its smaller working set.
//!
//! The `fast_ssim2` arm (added 2026-08-31 by the ssim2-replacement-bar lane,
//! `benchmarks/ssim2_replacement_bar_2026-08-31.md`) is the speed row of the
//! "as good or better than ssim2" exam. It belongs in THIS group rather than a
//! separate bench because a head-to-head against an external opponent is
//! exactly where an isolated back-to-back measurement would bake the box's
//! thermal/neighbour state into the answer — interleaving it with the walks it
//! is being compared to is the whole point. fast-ssim2 is measured at its
//! DEFAULT features, i.e. single-threaded (its `rayon` feature parallelises the
//! Gaussian blur and is off); `zensim-bench`'s `ssim2_speed_bar` prices the
//! threaded variant, since a cargo feature cannot vary per arm.
//!
//! WIDTH NOTE, load-bearing for reading these numbers: the buffered path
//! walks `simd_padded_width(w)` columns, the fold walks `w`. At 576/1152/2304
//! that is +16 columns (592/1168/2320), so the buffered arm does 2.8 % / 1.4 %
//! / 0.7 % more column work — and its pool VALUES differ from the fold's
//! (`folded720_v1_pools_match_v1_path` is bit-exact only where
//! `simd_padded_width(w) == w`).
//!
//! Thread count comes from `RAYON_NUM_THREADS` (both families take
//! `parallel = true`); run the binary once per count.
//!
//! Run:
//! ```text
//! cargo bench --bench extract_paths_bench -p zensim \
//!   --features custom-profiles,feature-regime-v2,threads,training
//! ```
//! Peak-RSS mode (one arm, no zenbench, for `/usr/bin/time -v`):
//! `ZEN_XP_RSS=<arm> ZEN_XP_SIZE=1152 ZEN_XP_ITERS=20 <the bench binary>`
use zensim::profile::ProfileParams;
use zensim::{RgbSlice, Zensim, ZensimConfig, ZensimProfile, compute_zensim_with_config};

// Reuse the established example/benchmark decode owner; native timing accepts
// only already admitted RGB8 PNG pairs, never original codec inputs or HDR.
#[path = "../examples/support/zen_io.rs"]
mod zen_io;

struct TimingPair {
    label: String,
    width: usize,
    height: usize,
    source: Vec<[u8; 3]>,
    distorted: Vec<[u8; 3]>,
}

fn timing_sha(bytes: &[u8]) -> String {
    use sha2::Digest;
    sha2::Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// ZEN_XP_PAIRS supplies a registered TRAIN-development JSON manifest.
/// ZEN_XP_PAIRS_SHA256 binds its exact bytes. All role metadata is checked
/// before pixel reads. Native geometry is never resized or implicitly replaced.
fn timing_pairs(sizes: &[usize]) -> Vec<TimingPair> {
    let Some(path) = std::env::var_os("ZEN_XP_PAIRS") else {
        assert!(
            std::env::var_os("ZEN_XP_PAIRS_SHA256").is_none(),
            "pair hash without manifest"
        );
        return sizes
            .iter()
            .map(|&n| {
                let (source, distorted) = test_pair(n, n);
                TimingPair {
                    label: n.to_string(),
                    width: n,
                    height: n,
                    source,
                    distorted,
                }
            })
            .collect();
    };
    assert!(
        std::env::var_os("ZEN_XP_SIZES").is_none(),
        "native pairs conflict with synthetic sizes"
    );
    let bytes = std::fs::read(path).expect("pair manifest");
    assert_eq!(
        timing_sha(&bytes),
        std::env::var("ZEN_XP_PAIRS_SHA256").expect("pair manifest hash"),
        "pair manifest hash mismatch"
    );
    let manifest: serde_json::Value = serde_json::from_slice(&bytes).expect("pair JSON");
    assert_eq!(
        manifest["role"], "train",
        "native timing permits TRAIN only"
    );
    assert_eq!(
        manifest["partition"], "development",
        "native timing partition"
    );
    let cases = manifest["cases"].as_array().expect("pair cases");
    assert!(!cases.is_empty(), "empty pair manifest");
    let mut labels = std::collections::HashSet::new();
    for case in cases {
        assert_eq!(case["role"], "train", "native timing permits TRAIN only");
        assert_eq!(case["partition"], "development", "native timing partition");
        let label = case["id"].as_str().expect("pair id");
        assert!(
            !label.is_empty()
                && label
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'),
            "invalid pair id"
        );
        assert!(labels.insert(label), "duplicate pair id");
        for field in [
            "reference",
            "png",
            "reference_sha256",
            "png_sha256",
            "reference_pixels_sha256",
            "distorted_pixels_sha256",
        ] {
            assert!(case[field].as_str().is_some(), "missing pair field {field}");
        }
        for field in ["width", "height"] {
            assert!(
                case[field]
                    .as_u64()
                    .is_some_and(|n| n > 0 && usize::try_from(n).is_ok()),
                "invalid dimension"
            );
        }
    }
    cases
        .iter()
        .map(|case| {
            let width = case["width"].as_u64().unwrap() as usize;
            let height = case["height"].as_u64().unwrap() as usize;
            let decode = |path_key: &str, bytes_key: &str, pixels_key: &str| {
                let path = std::path::Path::new(case[path_key].as_str().unwrap());
                let bytes = std::fs::read(path).expect("native PNG");
                assert_eq!(
                    timing_sha(&bytes),
                    case[bytes_key].as_str().unwrap(),
                    "native PNG hash mismatch"
                );
                assert!(
                    bytes.len() >= 33
                        && &bytes[..8] == b"\x89PNG\r\n\x1a\n"
                        && &bytes[12..16] == b"IHDR"
                        && bytes[24] == 8
                        && bytes[25] == 2,
                    "native timing requires RGB8 PNG"
                );
                let (pixels, w, h) = zen_io::decode_rgb8(path);
                assert_eq!((w, h), (width, height), "native dimensions mismatch");
                assert_eq!(
                    timing_sha(bytemuck::cast_slice(&pixels)),
                    case[pixels_key].as_str().unwrap(),
                    "native pixel hash mismatch"
                );
                pixels
            };
            let source = decode("reference", "reference_sha256", "reference_pixels_sha256");
            let distorted = decode("png", "png_sha256", "distorted_pixels_sha256");
            let label = format!("native_{}", case["id"].as_str().unwrap());
            eprintln!("# admitted {label} {width}x{height} RGB8, byte and pixel hashes verified");
            TimingPair {
                label,
                width,
                height,
                source,
                distorted,
            }
        })
        .collect()
}

/// Deterministic textured pair — the same content family the attribution
/// tests and `fold_pools_bench` use, so numbers are comparable across both.
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

fn v1_cfg(extended: bool, iw: bool) -> ZensimConfig {
    // `ZensimConfig` is `#[non_exhaustive]` — build then set.
    let mut c = ZensimConfig::default();
    c.extended_features = extended;
    c.compute_iw_features = iw;
    c.allow_multithreading = true;
    c
}

fn fold_zensim() -> &'static Zensim {
    let params = ProfileParams::builder()
        .skip_score_mapping(true)
        .extrapolate_score(true)
        .extended_features(true)
        .build();
    let params: &'static ProfileParams = Box::leak(Box::new(params));
    Box::leak(Box::new(
        Zensim::new(ZensimProfile::Custom {
            name: "extract_paths_bench",
            params,
        })
        .with_parallel(true),
    ))
}

// Explicit producer controls; serving derives the needed work by ID.
fn toggles_off() -> zensim::feature_v2::V2NewFeatureToggles {
    zensim::feature_v2::V2NewFeatureToggles {
        append_block: true,
        append2_block: true,
        ..Default::default()
    }
}

fn toggles_full() -> zensim::feature_v2::V2NewFeatureToggles {
    zensim::feature_v2::V2NewFeatureToggles {
        v1_pools: zensim::feature_v2::V1PoolsMode::Full,
        ..toggles_off()
    }
}

fn toggles_csfw() -> zensim::feature_v2::V2NewFeatureToggles {
    zensim::feature_v2::V2NewFeatureToggles {
        csfw_block: true,
        ..toggles_off()
    }
}

/// f1322 shape: the rev4 feature-bank ON arm — all four families over the
fn toggles_rev4_all() -> zensim::feature_v2::V2NewFeatureToggles {
    zensim::feature_v2::V2NewFeatureToggles {
        rev4_gridblk: true,
        rev4_ringbasis: true,
        rev4_tailhist: true,
        rev4_arttype: true,
        ..toggles_csfw()
    }
}

fn toggles_gmsbank() -> zensim::feature_v2::V2NewFeatureToggles {
    zensim::feature_v2::V2NewFeatureToggles {
        gmsbank: true,
        ..toggles_rev4_all()
    }
}

/// `ZEN_XP_ARMS=a,b,c` restricts the interleaved groups to the named arms
/// (unset = every arm, so an unqualified run is unchanged). A full-zoo group
/// at 4096^2 holds the exclusive zenbench lock for hours; a cost comparison of
/// a handful of arms does not need the rest.
fn arm_on(name: &str) -> bool {
    match std::env::var("ZEN_XP_ARMS") {
        Ok(list) => list.split(',').any(|a| a == name),
        Err(_) => true,
    }
}

fn bench_arm<F>(group: &mut zenbench::BenchGroup, name: impl Into<String>, f: F)
where
    F: FnMut(&mut zenbench::Bencher) + Send + 'static,
{
    let name = name.into();
    if arm_on(&name) {
        group.bench(name, f);
    }
}

/// Restored cuts (COST_CUTS_AUDIT): the layout chain is nested, so each arm
/// carries every earlier restored family; a family's marginal cost is the
/// difference to the arm before it (the C8-on `fold1502_gmsbank` arm is the
/// baseline of the first).
fn toggles_restore(upto: usize) -> zensim::feature_v2::V2NewFeatureToggles {
    zensim::feature_v2::V2NewFeatureToggles {
        mapdev: upto >= 1,
        z1max: upto >= 2,
        gmsnative: upto >= 3,
        ..toggles_gmsbank()
    }
}

fn restore_arm(name: &str) -> Option<usize> {
    match name {
        "fold1562_mapdev" => Some(1),
        "fold1790_z1max" => Some(2),
        "fold1820_gmsnative" => Some(3),
        _ => None,
    }
}

fn toggles_rev4_family(family: &str) -> zensim::feature_v2::V2NewFeatureToggles {
    let mut t = toggles_rev4_all();
    match family {
        "gridblk" => t.rev4_gridblk = true,
        "ringbasis" => t.rev4_ringbasis = true,
        "tailhist" => t.rev4_tailhist = true,
        "arttype" => t.rev4_arttype = true,
        other => panic!("unknown rev4 family: {other}"),
    }
    t
}

/// Raw extraction controls. Serving additionally derives its plan, gathers
/// declared IDs and applies the complete scoring composition.
fn toggles_v1_only(
    pools: zensim::feature_v2::V1PoolsMode,
) -> zensim::feature_v2::V2NewFeatureToggles {
    zensim::feature_v2::V2NewFeatureToggles {
        v1_only: true,
        v1_pools: pools,
        ..Default::default()
    }
}

fn free_toggles(arm: &str) -> zensim::feature_v2::V2NewFeatureToggles {
    use zensim::feature_v2::{V1FreeExtras, V1PoolsMode};
    zensim::feature_v2::V2NewFeatureToggles {
        // Retain the 944 identity layout while v1_only suppresses its work.
        append_block: true,
        append2_block: true,
        free_extras: match arm {
            "fold228_moments" => V1FreeExtras::RawMoments,
            "fold228_classc" => V1FreeExtras::RawMomentsPlusBoundedErr,
            _ => unreachable!("free control arm"),
        },
        ..toggles_v1_only(V1PoolsMode::Peaks)
    }
}

/// `(max_rounds, min_rounds, max_wall_seconds)` — the defaults, or the
/// `ZEN_XP_ROUNDS` / `ZEN_XP_MIN_ROUNDS` / `ZEN_XP_WALL_S` overrides. `min_rounds` follows
/// `max_rounds` down (a min above the max would never terminate).
fn bench_budget() -> (usize, usize, u64) {
    let max_r: usize = std::env::var("ZEN_XP_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let wall_s: u64 = std::env::var("ZEN_XP_WALL_S")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(600);
    let min_r: usize = std::env::var("ZEN_XP_MIN_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(25);
    (max_r, min_r.min(max_r), wall_s)
}

/// One-arm loop for external peak-RSS measurement (`/usr/bin/time -v`) and
/// for hardware-counter profiling (`perf stat`/`perf record`).
///
/// `ZEN_XP_SIZE` is the square side. `ZEN_XP_W` / `ZEN_XP_H` override the
/// width / height independently, which is what the branch-behaviour lane
/// needs: it has to compare a width that is a multiple of the SIMD lane
/// count (8 f32 lanes on `v4`, 16 on `v4x`) against ones that are not, so
/// the row-tail hypothesis can be tested rather than assumed.
fn rss_mode(arm: &str) {
    let size: usize = std::env::var("ZEN_XP_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1152);
    let w: usize = std::env::var("ZEN_XP_W")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(size);
    let h: usize = std::env::var("ZEN_XP_H")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(size);
    let iters: usize = std::env::var("ZEN_XP_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let (src, dst) = test_pair(w, h);
    if arm == "bake" {
        let models: Vec<_> = std::env::var("ZEN_XP_BAKE")
            .expect("ZEN_XP_BAKE")
            .split(',')
            .map(|path| zenpredict::Model::from_bytes(&std::fs::read(path).unwrap()).unwrap())
            .collect();
        let parallel = std::env::var("RAYON_NUM_THREADS").as_deref() != Ok("1");
        let weights: Option<Vec<f64>> = std::env::var("ZEN_XP_WEIGHTS")
            .ok()
            .map(|v| v.split(',').map(|x| x.parse().unwrap()).collect());
        let mut scorer = zensim::BakeScorer::ensemble(&models, weights.as_deref())
            .unwrap()
            .with_parallel(parallel)
            .with_finite_moment_refinement(std::env::var_os("ZEN_XP_FINITE_MOMENTS").is_some());
        let input = std::env::var("ZEN_XP_INPUT").unwrap_or_else(|_| "rgb8".into());
        assert!(matches!(
            input.as_str(),
            "rgb8" | "sdr16" | "linear-p3" | "hdr-pq16" | "hdr-linear2020"
        ));
        let (rs, ds) = (
            timing_source(Box::leak(src.into_boxed_slice()), w, h, &input),
            timing_source(Box::leak(dst.into_boxed_slice()), w, h, &input),
        );
        let encoding = match input.as_str() {
            "hdr-pq16" => Some(zensim::feature_v2::HdrEncoding::Pq { peak_nits: 1000. }),
            "hdr-linear2020" => Some(zensim::feature_v2::HdrEncoding::Linear),
            _ => None,
        };
        if let Some(encoding) = encoding {
            if std::env::var_os("ZEN_XP_SPATIAL").is_some() {
                let mut worker = scorer.prepare_steering_hdr(&rs, encoding, 8).unwrap();
                for _ in 0..iters {
                    zenbench::black_box(worker.compute(&ds, None).unwrap());
                }
            } else {
                for _ in 0..iters {
                    zenbench::black_box(scorer.compute_hdr(&rs, &ds, encoding, None).unwrap());
                }
            }
            return;
        }
        if std::env::var_os("ZEN_XP_SPATIAL").is_some() {
            let mut worker = scorer.prepare_steering(&rs, 8).unwrap();
            run_profile_calls(iters, || {
                zenbench::black_box(worker.compute(&ds, None).unwrap());
            });
        } else {
            run_profile_calls(iters, || {
                zenbench::black_box(scorer.compute(&rs, &ds, None).unwrap());
            });
        }
        return;
    }
    let _ = size;
    let size = w; // reported below; the walk uses `w`/`h`
    let z = fold_zensim();
    let mut scratch = zensim::feature_v2::V2Scratch::new();
    let mut sink = 0.0f64;
    for _ in 0..iters {
        match arm {
            "buf_v1_228" => {
                let r = compute_zensim_with_config(&src, &dst, w, h, v1_cfg(false, false))
                    .expect("buf_v1_228");
                sink += r.features()[0];
            }
            "buf_v1_372" => {
                let r = compute_zensim_with_config(&src, &dst, w, h, v1_cfg(true, true))
                    .expect("buf_v1_372");
                sink += r.features()[371];
            }
            "fold156_basic" | "fold228_peaks" | "fold372_full" => {
                use zensim::feature_v2::V1PoolsMode;
                let t = toggles_v1_only(match arm {
                    "fold156_basic" => V1PoolsMode::Off,
                    "fold228_peaks" => V1PoolsMode::Peaks,
                    _ => V1PoolsMode::Full,
                });
                let rsv = RgbSlice::new(&src, w, h);
                let dsv = RgbSlice::new(&dst, w, h);
                let v2 = z
                    .compute_folded720_features_streaming(&rsv, &dsv, t, &mut scratch)
                    .expect("fold v1_only");
                sink += v2.features()[0];
            }
            "fold228_moments" | "fold228_classc" => {
                let r = z
                    .compute_folded720_features_streaming(
                        &RgbSlice::new(&src, w, h),
                        &RgbSlice::new(&dst, w, h),
                        free_toggles(arm),
                        &mut scratch,
                    )
                    .expect("free control");
                sink += r.features()[941];
            }
            "fold944_off" | "fold944_full" => {
                let t = if arm == "fold944_full" {
                    toggles_full()
                } else {
                    toggles_off()
                };
                let rsv = RgbSlice::new(&src, w, h);
                let dsv = RgbSlice::new(&dst, w, h);
                let v2 = z
                    .compute_folded720_features_streaming(&rsv, &dsv, t, &mut scratch)
                    .expect("fold");
                sink += v2.features()[943] as f64;
            }
            "fold956_csfw" => {
                let rsv = RgbSlice::new(&src, w, h);
                let dsv = RgbSlice::new(&dst, w, h);
                let v2 = z
                    .compute_folded720_features_streaming(&rsv, &dsv, toggles_csfw(), &mut scratch)
                    .expect("fold csfw");
                sink += v2.features()[v2.features().len() - 1];
            }
            "fold986_gridblk" | "fold986_ringbasis" | "fold986_tailhist" | "fold986_arttype"
            | "fold1322_rev4" | "fold1502_gmsbank" => {
                let t = if arm == "fold1502_gmsbank" {
                    toggles_gmsbank()
                } else if arm == "fold1322_rev4" {
                    toggles_rev4_all()
                } else {
                    toggles_rev4_family(arm.strip_prefix("fold986_").unwrap())
                };
                let rsv = RgbSlice::new(&src, w, h);
                let dsv = RgbSlice::new(&dst, w, h);
                let v2 = z
                    .compute_folded720_features_streaming(&rsv, &dsv, t, &mut scratch)
                    .expect("fold rev4");
                sink += v2.features()[v2.features().len() - 1] as f64;
            }
            other if restore_arm(other).is_some() => {
                let t = toggles_restore(restore_arm(other).unwrap());
                let rsv = RgbSlice::new(&src, w, h);
                let dsv = RgbSlice::new(&dst, w, h);
                let v2 = z
                    .compute_folded720_features_streaming(&rsv, &dsv, t, &mut scratch)
                    .expect("fold restore");
                sink += v2.features()[v2.features().len() - 1] as f64;
            }
            other => panic!("unknown ZEN_XP_RSS arm: {other}"),
        }
    }
    println!("{arm} size={size} w={w} h={h} iters={iters} sink={sink:e}");
}

/// Diagnostic bakes exercise the actual model-selected extraction path.
/// Nonzero weights on every declared ID prevent dead-input pruning from
/// turning the full control into the subset. These are checksums, not fits.
fn subset_model(full: bool) -> zenpredict::Model {
    let ids: Vec<usize> = (0..228)
        .filter(|&i| full || !matches!(i, 0..=12 | 26..=38 | 156..=161 | 168..=173))
        .collect();
    let n = ids.len();
    let recipe = serde_json::json!({
        "schema_hash": 1, "scaler_mean": vec![0.0; n], "scaler_scale": vec![1.0; n],
        "metadata": [
            {"key":"zentrain.feature_ids","type":"utf8","text":ids.iter().map(usize::to_string).collect::<Vec<_>>().join("\n")},
            {"key":"zentrain.formula_revision","type":"utf8","text":std::env::var("ZENSIM_FORMULA_REV").unwrap_or_else(|_| "1".into())}
        ],
        "layers": [{"in_dim": n, "out_dim": 1, "activation":"identity", "dtype":"f32",
                    "weights": vec![0.01; n], "biases":[0.0]}]
    });
    let bytes = zenpredict_bake::bake_from_json_str(&recipe.to_string()).unwrap();
    zenpredict::Model::from_bytes(&bytes).unwrap()
}

/// Opt-in paired comparison, so the unrelated wide arms do not consume the
/// measurement budget. Includes planning, allocations, extraction and forward
/// through BakeScorer::compute, just as a caller uses it. Raw fold control
/// reuses scratch and is explicitly distinguished from these serving arms.
fn subset_bench(sizes: &[usize]) {
    let full = Box::leak(Box::new(subset_model(true)));
    let subset = Box::leak(Box::new(subset_model(false)));
    let z = fold_zensim();
    // zenbench 0.1.9 mistakes its own lock-heartbeat task for a rival bench.
    // Keep its exclusive lock and paired statistics; run this opt-in group
    // on a quiet host without the broken process-name scan.
    let result = zenbench::run_gated(zenbench::GateConfig::disabled(), |suite| {
        for &n in sizes {
            let (src, dst) = test_pair(n, n);
            let src: &'static [[u8; 3]] = Box::leak(src.into_boxed_slice());
            let dst: &'static [[u8; 3]] = Box::leak(dst.into_boxed_slice());
            suite.compare(format!("fullres_y_{n}"), |group| {
                let (max_r, min_r, wall_s) = bench_budget();
                group
                    .config()
                    .max_rounds(max_r)
                    .min_rounds(min_r)
                    .max_wall_time(std::time::Duration::from_secs(wall_s));
                for (name, model) in [("bake228_full", &*full), ("bake190_y", &*subset)] {
                    bench_arm(group, name, move |b| {
                        let mut scorer = zensim::BakeScorer::new(model).unwrap();
                        b.iter(move || {
                            let r = scorer
                                .compute(&RgbSlice::new(src, n, n), &RgbSlice::new(dst, n, n), None)
                                .unwrap();
                            zenbench::black_box(r.score());
                        });
                    });
                }
                bench_arm(group, "fold228_reused_scratch", move |b| {
                    let mut scratch = zensim::feature_v2::V2Scratch::new();
                    b.iter(move || {
                        let r = z
                            .compute_folded720_features_streaming(
                                &RgbSlice::new(src, n, n),
                                &RgbSlice::new(dst, n, n),
                                toggles_v1_only(zensim::feature_v2::V1PoolsMode::Peaks),
                                &mut scratch,
                            )
                            .unwrap();
                        zenbench::black_box(r.features()[14]);
                    });
                });
            });
        }
    });
    if let Ok(path) = std::env::var("ZENBENCH_RESULT_PATH") {
        result.save(&path).unwrap();
    }
}

/// Synthetic native-format throughput probes. Values exercise low u16 bits;
/// they do not carry human labels or establish cross-input calibration.
fn timing_source(
    pixels: &'static [[u8; 3]],
    w: usize,
    h: usize,
    input: &str,
) -> zensim::StridedBytes<'static> {
    use zensim::{AlphaMode, ColorPrimaries, PixelFormat, StridedBytes};
    if input == "rgb8" {
        return StridedBytes::with_alpha_mode(
            bytemuck::cast_slice(pixels),
            w,
            h,
            w * 3,
            PixelFormat::Srgb8Rgb,
            AlphaMode::Opaque,
        );
    }
    if matches!(input, "sdr16" | "hdr-pq16") {
        let pixels: Vec<[u16; 4]> = pixels
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let c: [u16; 3] = core::array::from_fn(|c| {
                    u16::from(p[c]) * 256 + ((i * 37 + c * 53) % 256) as u16
                });
                [c[0], c[1], c[2], 65535]
            })
            .collect();
        let pixels = Box::leak(pixels.into_boxed_slice());
        return StridedBytes::with_alpha_mode(
            bytemuck::cast_slice(pixels),
            w,
            h,
            w * 8,
            PixelFormat::Srgb16Rgba,
            AlphaMode::Opaque,
        )
        .with_color_primaries(if input == "hdr-pq16" {
            ColorPrimaries::Bt2020
        } else {
            ColorPrimaries::Srgb
        });
    }
    let pixels: Vec<[f32; 4]> = pixels
        .iter()
        .map(|p| {
            let factor = if input == "hdr-linear2020" { 1000. } else { 1. };
            let c = p.map(|v| linear_srgb::default::srgb_u8_to_linear(v) * factor);
            [c[0], c[1], c[2], 1.]
        })
        .collect();
    let pixels = Box::leak(pixels.into_boxed_slice());
    StridedBytes::with_alpha_mode(
        bytemuck::cast_slice(pixels),
        w,
        h,
        w * 16,
        PixelFormat::LinearF32Rgba,
        AlphaMode::Opaque,
    )
    .with_color_primaries(if input == "hdr-linear2020" {
        ColorPrimaries::Bt2020
    } else {
        ColorPrimaries::DisplayP3
    })
}

/// Complete public-API model comparisons, including optional cached spatial
/// maps. Manifest rows are `name<TAB>bake_path[,bake_path...]`, without a
/// header. An optional third TAB column supplies comma-separated weights.
/// Without weights, multiple paths use the public uniform ensemble composition.
/// A fourth column (`0` or `1`) selects finite-moment refinement for paired arms.
fn sampling_models_bench(sizes: &[usize], manifest: &str) {
    struct Case {
        name: String,
        models: &'static [zenpredict::Model],
        weights: Option<&'static [f64]>,
        finite_moments: bool,
    }
    let models: Vec<Case> = std::fs::read_to_string(manifest)
        .unwrap()
        .lines()
        .map(|line| {
            let mut fields = line.split('\t');
            let name = fields.next().expect("name");
            let path = fields.next().expect("bake paths");
            let weights = fields.next().map(|s| {
                let values: Vec<f64> = s.split(',').map(|v| v.parse().unwrap()).collect();
                &*Box::leak(values.into_boxed_slice())
            });
            let finite_moments = match fields.next() {
                None | Some("0") => false,
                Some("1") => true,
                _ => panic!("finite moments must be 0 or 1"),
            };
            assert!(fields.next().is_none(), "unexpected model manifest column");
            let members: Vec<_> = path
                .split(',')
                .map(|path| {
                    let bytes = std::fs::read(path).unwrap();
                    zenpredict::Model::from_bytes(&bytes).unwrap()
                })
                .collect();
            Case {
                name: name.to_string(),
                models: Box::leak(members.into_boxed_slice()),
                weights,
                finite_moments,
            }
        })
        .collect();
    let input = std::env::var("ZEN_XP_INPUT").unwrap_or_else(|_| "rgb8".into());
    assert!(
        matches!(
            input.as_str(),
            "rgb8" | "sdr16" | "linear-p3" | "hdr-pq16" | "hdr-linear2020"
        ),
        "unknown native input probe"
    );
    assert!(
        input == "rgb8" || std::env::var_os("ZEN_XP_PAIRS").is_none(),
        "native-format synthetic probes cannot relabel manifest pixels"
    );
    assert!(
        input == "rgb8" || std::env::var_os("ZEN_XP_CONTROLS").is_none(),
        "RGB8 peer controls require RGB8 input"
    );
    let encoding = match input.as_str() {
        "hdr-pq16" => Some(zensim::feature_v2::HdrEncoding::Pq { peak_nits: 1000. }),
        "hdr-linear2020" => Some(zensim::feature_v2::HdrEncoding::Linear),
        _ => None,
    };
    let spatial = std::env::var_os("ZEN_XP_SPATIAL").is_some();
    let prepared = std::env::var_os("ZEN_XP_PREPARED").is_some();
    assert!(
        !spatial || encoding.is_none() || prepared,
        "HDR spatial timing requires ZEN_XP_PREPARED"
    );
    let parallel = std::env::var("RAYON_NUM_THREADS").as_deref() != Ok("1");
    let pairs = timing_pairs(sizes);
    let result_path = std::env::var_os("ZENBENCH_RESULT_PATH").map(std::path::PathBuf::from);
    if let Some(path) = &result_path {
        assert!(!path.exists(), "refusing to overwrite timing evidence");
    }
    let result = zenbench::run_gated(zenbench::GateConfig::strict(), |suite| {
        for pair in pairs {
            let (w, h, label) = (pair.width, pair.height, pair.label);
            let (src, dst) = (pair.source, pair.distorted);
            let src: &'static [[u8; 3]] = Box::leak(src.into_boxed_slice());
            let dst: &'static [[u8; 3]] = Box::leak(dst.into_boxed_slice());
            let (native_src, native_dst) = (
                timing_source(src, w, h, &input),
                timing_source(dst, w, h, &input),
            );
            let label = if input == "rgb8" {
                label
            } else {
                format!("{label}_{input}")
            };
            suite.compare(
                format!(
                    "sampling_{}_{label}",
                    if spatial { "spatial" } else { "scalar" }
                ),
                |group| {
                    let (max_r, min_r, wall_s) = bench_budget();
                    group
                        .config()
                        .max_rounds(max_r)
                        .min_rounds(min_r)
                        .max_wall_time(std::time::Duration::from_secs(wall_s));
                    // This model comparison reports per-call latency, never
                    // percentiles of automatically batched per-iteration means.
                    group.config().min_iterations = 1;
                    group.config().max_iterations = 1;

                    if std::env::var_os("ZEN_XP_CONTROLS").is_some() && !spatial {
                        #[cfg(feature = "candidate-profiles")]
                        bench_arm(group, "D_current_revision", move |b| {
                            let z = Zensim::new(ZensimProfile::D).with_parallel(parallel);
                            let (rs, ds) = (RgbSlice::new(src, w, h), RgbSlice::new(dst, w, h));
                            b.iter(move || {
                                zenbench::black_box(z.compute(&rs, &ds).unwrap().score())
                            });
                        });
                        bench_arm(group, "fast_ssim2_st", move |b| {
                            let (rs, ds) =
                                (imgref::Img::new(src, w, h), imgref::Img::new(dst, w, h));
                            b.iter(move || {
                                zenbench::black_box(
                                    fast_ssim2::compute_ssimulacra2(rs, ds).unwrap(),
                                )
                            });
                        });
                    }
                    for case in &models {
                        let (model, weights) = (case.models, case.weights);
                        let finite_moments = case.finite_moments;
                        bench_arm(group, case.name.clone(), move |b| {
                            let mut scorer = zensim::BakeScorer::ensemble(model, weights)
                                .unwrap()
                                .with_parallel(parallel)
                                .with_finite_moment_refinement(finite_moments);
                            let (rs, ds) = (native_src, native_dst);
                            if let Some(encoding) = encoding {
                                if spatial {
                                    let mut worker =
                                        scorer.prepare_steering_hdr(&rs, encoding, 8).unwrap();
                                    b.iter(move || {
                                        let r = worker.compute(&ds, None).unwrap();
                                        zenbench::black_box(r.refinement_gain(0, 0, w / 2, h / 2));
                                    });
                                } else {
                                    b.iter(move || {
                                        zenbench::black_box(
                                            scorer.compute_hdr(&rs, &ds, encoding, None).unwrap(),
                                        )
                                    });
                                }
                                return;
                            }
                            if spatial && prepared {
                                let mut worker = scorer.prepare_steering(&rs, 8).unwrap();
                                b.iter(move || {
                                    let r = worker.compute(&ds, None).unwrap();
                                    zenbench::black_box(r.refinement_gain(0, 0, w / 2, h / 2));
                                });
                            } else if spatial {
                                let pre = scorer.precompute_reference(&rs).unwrap();
                                let mut session = zensim::Fused944Session::new();
                                b.iter(move || {
                                    let r = scorer
                                        .compute_with_ref_and_attribution(
                                            &rs,
                                            &pre,
                                            &ds,
                                            None,
                                            &mut session,
                                            8,
                                        )
                                        .unwrap();
                                    zenbench::black_box(r.refinement_gain(0, 0, w / 2, h / 2));
                                });
                            } else {
                                b.iter(move || {
                                    zenbench::black_box(
                                        scorer.compute(&rs, &ds, None).unwrap().score(),
                                    );
                                });
                            }
                        });
                    }
                },
            );
        }
    });
    if let Some(path) = result_path {
        result.save(path).unwrap();
    }
}

// Persistent isolated revision owners let zenbench interleave Rev3/4/5
// without mutating a process-wide formula gate. Setup and warmup are untimed;
// the parent measures a call plus pipe round-trip (reported explicitly).
fn run_profile_calls(count: usize, mut action: impl FnMut()) {
    if std::env::var_os("ZEN_XP_ROUND_WORKER").is_none() {
        for _ in 0..count {
            action();
        }
        return;
    }
    use std::io::{BufRead, Write};
    for _ in 0..3 {
        action();
    }
    let mut out = std::io::stdout().lock();
    writeln!(out, "READY").unwrap();
    out.flush().unwrap();
    for line in std::io::stdin().lock().lines().map_while(Result::ok) {
        if line == "quit" {
            break;
        }
        let start = std::time::Instant::now();
        action();
        writeln!(out, "{}", start.elapsed().as_nanos()).unwrap();
        out.flush().unwrap();
    }
}

struct RevisionWorker {
    process: std::process::Child,
    input: std::process::ChildStdin,
    output: std::io::BufReader<std::process::ChildStdout>,
}
impl RevisionWorker {
    fn new(binary: &str, revision: &str, bake: &str, size: usize, spatial: bool) -> Self {
        use std::io::BufRead;
        let mut command = std::process::Command::new(binary);
        command
            .env("ZEN_XP_RSS", "bake")
            .env("ZEN_XP_ROUND_WORKER", "1")
            .env("ZEN_XP_BAKE", bake)
            .env("ZEN_XP_SIZE", size.to_string())
            .env("ZENSIM_FORMULA_REV", revision)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped());
        if spatial {
            command.env("ZEN_XP_SPATIAL", "1");
        } else {
            command.env_remove("ZEN_XP_SPATIAL");
        }
        let mut process = command.spawn().unwrap();
        let input = process.stdin.take().unwrap();
        let mut output = std::io::BufReader::new(process.stdout.take().unwrap());
        let mut ready = String::new();
        output.read_line(&mut ready).unwrap();
        assert_eq!(ready.trim(), "READY");
        Self {
            process,
            input,
            output,
        }
    }
    fn call(&mut self) -> u128 {
        use std::io::{BufRead, Write};
        writeln!(self.input, "call").unwrap();
        self.input.flush().unwrap();
        let mut response = String::new();
        self.output.read_line(&mut response).unwrap();
        response.trim().parse().unwrap()
    }
}
impl Drop for RevisionWorker {
    fn drop(&mut self) {
        use std::io::Write;
        let _ = writeln!(self.input, "quit");
        let _ = self.input.flush();
        let _ = self.process.wait();
    }
}

fn revision_models_bench(sizes: &[usize], manifest: &str) {
    let entries: Vec<Vec<String>> = std::fs::read_to_string(manifest)
        .unwrap()
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .map(|line| line.split('\t').map(str::to_string).collect())
        .collect();
    assert!(
        entries.iter().all(|row| row.len() == 4),
        "name, binary, revision, bake TSV"
    );
    let path = std::env::var("ZENBENCH_RESULT_PATH").expect("retain interleaved rounds");
    assert!(
        !std::path::Path::new(&path).exists(),
        "refusing to overwrite timing evidence"
    );
    let spatial = std::env::var_os("ZEN_XP_SPATIAL").is_some();
    let result = zenbench::run_gated(zenbench::GateConfig::strict(), |suite| {
        for &size in sizes {
            suite.compare(
                format!("revisions_{}_{size}", if spatial { "map" } else { "score" }),
                |group| {
                    let (rounds, min_rounds, wall) = bench_budget();
                    group
                        .config()
                        .max_rounds(rounds)
                        .min_rounds(min_rounds)
                        .max_wall_time(std::time::Duration::from_secs(wall));
                    group.config().min_iterations = 1;
                    group.config().max_iterations = 1;
                    for row in &entries {
                        let mut worker =
                            RevisionWorker::new(&row[1], &row[2], &row[3], size, spatial);
                        bench_arm(group, row[0].clone(), move |b| b.iter(|| worker.call()));
                    }
                },
            );
        }
    });
    result.save(path).unwrap();
}

/// Kernel study uses upstream zenresize's own coefficient tables and float
/// resizer. No copied filters, RGB transfer, quantization, or model scoring.
fn resize_filter_bench(sizes: &[usize]) {
    use zenresize::filter::InterpolationDetails;
    use zenresize::weights::F32WeightTable;
    use zenresize::{Filter, PixelFormat, ResizeConfig, Resizer};
    let filters = [
        Filter::Box,
        Filter::Triangle,
        Filter::Mitchell,
        Filter::RobidouxFast,
        Filter::RobidouxSharp,
        Filter::Lanczos2,
        Filter::Lanczos,
    ];
    let mut response = Vec::new();
    for filter in filters {
        for (num, den) in [(3usize, 2usize), (2, 1), (3, 1)] {
            let d = num as f64 / den as f64;
            // Interior exact-ratio polyphase coefficients from the real owner.
            let table = F32WeightTable::new(
                576,
                (576 * den / num) as u32,
                &InterpolationDetails::create(filter),
            );
            for phase in 0..den {
                let i = 96 * den / num + phase;
                let taps = table.weights(i);
                let left = table.left[i];
                let gain = |f: f64| {
                    let (re, im) = taps
                        .iter()
                        .enumerate()
                        .fold((0.0, 0.0), |(re, im), (j, &h)| {
                            let theta = std::f64::consts::TAU * f * (f64::from(left) + j as f64);
                            (
                                re + f64::from(h) * theta.cos(),
                                im + f64::from(h) * theta.sin(),
                            )
                        });
                    re.hypot(im)
                };
                let gains: Vec<_> = [4, 8, 16, 32]
                    .into_iter()
                    .map(|b| serde_json::json!({"period":b,"amplitude":gain(1.0/f64::from(b))}))
                    .collect();
                // Actual float resize: all numerator² input phases of an impulse, so phase
                // sensitivity and signed lobes are measured rather than inferred.
                let n = 96usize;
                let m = n * den / num;
                let cfg = ResizeConfig::builder(n as u32, n as u32, m as u32, m as u32)
                    .format(PixelFormat::GRAYF32_LINEAR)
                    .filter(filter)
                    .build();
                let mut resizer = Resizer::new(&cfg);
                let mut input = vec![0.0; n * n];
                let mut output = vec![0.0; m * m];
                let mut impulses = Vec::new();
                for py in 0..num {
                    for px in 0..num {
                        input.fill(0.0);
                        input[(48 + py) * n + 48 + px] = 1.0;
                        resizer.resize_f32_into(&input, &mut output);
                        let max = output.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let min = output.iter().copied().fold(f32::INFINITY, f32::min);
                        let energy: f64 = output.iter().map(|&x| f64::from(x).powi(2)).sum();
                        impulses.push(serde_json::json!({"phase":[px,py],"max":max,"min":min,"mse_ratio":energy*d*d}));
                    }
                }
                for (i, p) in input.iter_mut().enumerate() {
                    *p = if (i % n + i / n).is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    };
                }
                resizer.resize_f32_into(&input, &mut output);
                let checker_mse = (8..m - 8)
                    .flat_map(|y| (8..m - 8).map(move |x| y * m + x))
                    .map(|i| f64::from(output[i]).powi(2))
                    .sum::<f64>()
                    / ((m - 16) * (m - 16)) as f64;
                response.push(serde_json::json!({
                "filter":filter.name(),"stride":d,"ratio":[num,den],"output_phase":phase,"taps":taps,"left":left,
                "tap_count":taps.len(),"dc_gain":gain(0.0),"gains":gains,
                "checkerboard_amplitude":gain(0.5).powi(2),"actual_checkerboard_mse":checker_mse,
                "white_noise_variance_gain_2d":taps.iter().map(|&h| f64::from(h).powi(2)).sum::<f64>().powi(2),
                "impulses":impulses
            }));
            }
        }
    }
    let path = std::env::var("ZEN_XP_FILTER_RESPONSE")
        .expect("set ZEN_XP_FILTER_RESPONSE to an output JSON path");
    std::fs::write(path, serde_json::to_vec_pretty(&response).unwrap()).unwrap();
    let result = zenbench::run_gated(zenbench::GateConfig::disabled(), |suite| {
        for &n in sizes {
            assert_eq!(n % 6, 0, "filter sizes must divide exactly by 2 and 3");
            let input: &'static [f32] = Box::leak(
                (0..n * n)
                    .map(|i| ((i % n * 7 + i / n * 13) % 257) as f32 / 256.0)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            );
            suite.compare(format!("resize_float_plane_{n}"), |group| {
                let (max_r, min_r, wall_s) = bench_budget();
                group
                    .config()
                    .max_rounds(max_r)
                    .min_rounds(min_r)
                    .max_wall_time(std::time::Duration::from_secs(wall_s));
                for filter in filters {
                    for (num, den) in [(3usize, 2usize), (2, 1), (3, 1)] {
                        let d = num as f64 / den as f64;
                        let m = n * den / num;
                        let cfg = ResizeConfig::builder(n as u32, n as u32, m as u32, m as u32)
                            .format(PixelFormat::GRAYF32_LINEAR)
                            .filter(filter)
                            .build();
                        bench_arm(group, format!("{}_{}x", filter.name(), d), move |b| {
                            let mut resizer = Resizer::new(&cfg);
                            let mut output = vec![0.0; m * m];
                            b.iter(move || {
                                resizer.resize_f32_into(input, &mut output);
                                zenbench::black_box(&output);
                            });
                        });
                    }
                }
            });
        }
    });
    if let Ok(path) = std::env::var("ZENBENCH_RESULT_PATH") {
        result.save(&path).unwrap();
    }
}

/// `ZENSIM_MAX_TIER=v3` disables the AVX-512 tokens for this process — the
/// callgrind profiling tier (valgrind cannot execute AVX-512). Mirror of
/// `featcost::apply_tier_cap`, optional here so the same binary still benches
/// at the native tier when the variable is unset.
fn apply_tier_cap_from_env() {
    match std::env::var("ZENSIM_MAX_TIER").as_deref() {
        Err(_) => {}
        Ok("v3") => {
            use archmage::{X64V4Token, X64V4xToken};
            X64V4xToken::dangerously_disable_token_process_wide(true).expect("cap v4x");
            X64V4Token::dangerously_disable_token_process_wide(true).expect("cap v4");
            eprintln!("tier cap: AVX-512 tokens disabled (ZENSIM_MAX_TIER=v3)");
        }
        Ok(o) => panic!("ZENSIM_MAX_TIER={o:?}: only v3"),
    }
}

fn main() {
    apply_tier_cap_from_env();
    if let Ok(arm) = std::env::var("ZEN_XP_RSS") {
        rss_mode(&arm);
        return;
    }
    let sizes: Vec<usize> = std::env::var("ZEN_XP_SIZES")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![576, 1152, 2304]);
    if let Ok(manifest) = std::env::var("ZEN_XP_EXTERNAL_MODELS") {
        revision_models_bench(&sizes, &manifest);
        return;
    }
    if let Ok(manifest) = std::env::var("ZEN_XP_MODELS") {
        sampling_models_bench(&sizes, &manifest);
        return;
    }
    if std::env::var_os("ZEN_XP_FILTERS").is_some() {
        resize_filter_bench(&sizes);
        return;
    }
    if std::env::var_os("ZEN_XP_SUBSET").is_some() {
        subset_bench(&sizes);
        return;
    }
    // `ZENBENCH_RESULT_PATH` saves the completed paired rounds — raw per-round
    // durations, call counts, execution order and each round's pre-round gate
    // status — the same contract `zensim-bench/benches/ssim2_speed_bar.rs`
    // honours. Until 2026-09-18 this arm dropped its `RunResult` on the floor,
    // so a matrix run across thread counts and revisions produced only the
    // terminal summary table: medians survived as text, every raw round and
    // every gate flag did not, and no percentile could be recomputed. Checked
    // BEFORE the run, so a path collision costs nothing instead of discarding
    // a finished sweep.
    let result_path = std::env::var_os("ZENBENCH_RESULT_PATH").map(std::path::PathBuf::from);
    if let Some(path) = &result_path {
        assert!(!path.exists(), "refusing to overwrite benchmark evidence");
    }
    let z = fold_zensim();
    let (off, full) = (toggles_off(), toggles_full());
    let result = zenbench::run(|suite| {
        for &n in &sizes {
            let (src, dst) = test_pair(n, n);
            let src_s: &'static [[u8; 3]] = Box::leak(src.into_boxed_slice());
            let dst_s: &'static [[u8; 3]] = Box::leak(dst.into_boxed_slice());
            suite.compare(format!("extract_paths_{n}"), |group| {
                // Same budget the pools lane raised fold_pools_bench to: the
                // default 120 s / 4-usable-rounds cannot resolve a few-percent
                // lever on a shared box.
                //
                // OVERRIDABLE (feature-cost lane): this bench takes an
                // exclusive zenbench lock, so a full 3-size × 3-thread matrix
                // at the default budget holds it for hours and starves every
                // other lane on the box. `ZEN_XP_ROUNDS` / `ZEN_XP_WALL_S`
                // let a matrix run at a coarser budget; the defaults are
                // unchanged, so an unqualified invocation reproduces the
                // published single-shot numbers exactly.
                let (max_r, min_r, wall_s) = bench_budget();
                group
                    .config()
                    .max_rounds(max_r)
                    .min_rounds(min_r)
                    .max_wall_time(std::time::Duration::from_secs(wall_s));
                bench_arm(group, "buf_v1_228", move |b| {
                    b.iter(move || {
                        let r =
                            compute_zensim_with_config(src_s, dst_s, n, n, v1_cfg(false, false))
                                .unwrap();
                        zenbench::black_box(r.features()[0]);
                    })
                });
                bench_arm(group, "buf_v1_372", move |b| {
                    b.iter(move || {
                        let r = compute_zensim_with_config(src_s, dst_s, n, n, v1_cfg(true, true))
                            .unwrap();
                        zenbench::black_box(r.features()[371]);
                    })
                });
                for (name, pools) in [
                    ("fold156_basic", zensim::feature_v2::V1PoolsMode::Off),
                    ("fold228_peaks", zensim::feature_v2::V1PoolsMode::Peaks),
                    ("fold372_full", zensim::feature_v2::V1PoolsMode::Full),
                ] {
                    let t = toggles_v1_only(pools);
                    bench_arm(group, name, move |b| {
                        let mut scratch = zensim::feature_v2::V2Scratch::new();
                        b.iter(move || {
                            let rsv = RgbSlice::new(src_s, n, n);
                            let dsv = RgbSlice::new(dst_s, n, n);
                            let v2 = z
                                .compute_folded720_features_streaming(&rsv, &dsv, t, &mut scratch)
                                .unwrap();
                            zenbench::black_box(v2.features()[0]);
                        })
                    });
                }
                for name in ["fold228_moments", "fold228_classc"] {
                    let t = free_toggles(name);
                    bench_arm(group, name, move |b| {
                        let mut scratch = zensim::feature_v2::V2Scratch::new();
                        b.iter(move || {
                            let v = z
                                .compute_folded720_features_streaming(
                                    &RgbSlice::new(src_s, n, n),
                                    &RgbSlice::new(dst_s, n, n),
                                    t,
                                    &mut scratch,
                                )
                                .unwrap();
                            zenbench::black_box(v.features()[941]);
                        })
                    });
                }
                bench_arm(group, "fold944_off", move |b| {
                    let mut scratch = zensim::feature_v2::V2Scratch::new();
                    b.iter(move || {
                        let rsv = RgbSlice::new(src_s, n, n);
                        let dsv = RgbSlice::new(dst_s, n, n);
                        let v2 = z
                            .compute_folded720_features_streaming(&rsv, &dsv, off, &mut scratch)
                            .unwrap();
                        zenbench::black_box(v2.features()[943]);
                    })
                });
                bench_arm(group, "fold944_full", move |b| {
                    let mut scratch = zensim::feature_v2::V2Scratch::new();
                    b.iter(move || {
                        let rsv = RgbSlice::new(src_s, n, n);
                        let dsv = RgbSlice::new(dst_s, n, n);
                        let v2 = z
                            .compute_folded720_features_streaming(&rsv, &dsv, full, &mut scratch)
                            .unwrap();
                        zenbench::black_box((v2.features()[178], v2.features()[943]));
                    })
                });
                // 30-slot block (986). Same pixels, same process, paired.
                bench_arm(group, "fold956_csfw", move |b| {
                    let mut scratch = zensim::feature_v2::V2Scratch::new();
                    let t = toggles_csfw();
                    b.iter(move || {
                        let rsv = RgbSlice::new(src_s, n, n);
                        let dsv = RgbSlice::new(dst_s, n, n);
                        let v2 = z
                            .compute_folded720_features_streaming(&rsv, &dsv, t, &mut scratch)
                            .unwrap();
                        zenbench::black_box(v2.features()[955]);
                    })
                });
                // Rev4 feature-bank cost gate (2026-09-23): the OFF arm is
                // Each family arm flips exactly one rev4 toggle; the full arm
                // runs all four (f1322). Same pixels, same process, paired.
                for (name, t) in [
                    ("fold986_gridblk", toggles_rev4_family("gridblk")),
                    ("fold986_ringbasis", toggles_rev4_family("ringbasis")),
                    ("fold986_tailhist", toggles_rev4_family("tailhist")),
                    ("fold986_arttype", toggles_rev4_family("arttype")),
                    ("fold1322_rev4", toggles_rev4_all()),
                    ("fold1502_gmsbank", toggles_gmsbank()),
                    ("fold1562_mapdev", toggles_restore(1)),
                    ("fold1790_z1max", toggles_restore(2)),
                    ("fold1820_gmsnative", toggles_restore(3)),
                ] {
                    bench_arm(group, name, move |b| {
                        let mut scratch = zensim::feature_v2::V2Scratch::new();
                        b.iter(move || {
                            let rsv = RgbSlice::new(src_s, n, n);
                            let dsv = RgbSlice::new(dst_s, n, n);
                            let v2 = z
                                .compute_folded720_features_streaming(&rsv, &dsv, t, &mut scratch)
                                .unwrap();
                            zenbench::black_box(v2.features()[v2.features().len() - 1]);
                        })
                    });
                }
                // The opponent. Same pixels, same process, same round.
                bench_arm(group, "fast_ssim2", move |b| {
                    b.iter(move || {
                        let s = imgref::Img::new(src_s, n, n);
                        let d = imgref::Img::new(dst_s, n, n);
                        zenbench::black_box(fast_ssim2::compute_ssimulacra2(s, d).unwrap())
                    })
                });
            });
        }
    });
    match result_path {
        Some(path) => result.save(path).expect("save benchmark evidence"),
        None => drop(result),
    }
}
