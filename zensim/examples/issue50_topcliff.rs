//! Issue #50 instrument: decompose the near-100 "top cliff".
//!
//! Reproduces the issue's perturbation grid (shift X% of pixels by ±N
//! codes, random sign, clamped) on real images and scores every pair
//! twice through the PUBLIC scoring pipeline:
//!
//! 1. `ZensimProfile::B` (the default `codec_target`) → the shipped score
//!    (post output-calibration-spline; B sets `skip_score_mapping`, so
//!    `score == raw_distance == spline(model_raw)`).
//! 2. A `ZensimProfile::Custom` clone of B whose bake had its
//!    `zentrain.output_calibration_spline` metadata stripped via the
//!    canonical `bake_dial_refit strip` — same features, same forward,
//!    NO spline → the PRE-spline model raw.
//!
//! Comparing the two columns per perturbation separates "the model's raw
//! output saturates" from "the spline compresses the top".
//!
//! Usage:
//! ```sh
//! bake_dial_refit strip \
//!     --in zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin \
//!     --out /tmp/b_nospline.bin
//! cargo run --release --features custom-profiles --example issue50_topcliff -- \
//!     --stripped-bake /tmp/b_nospline.bin image1.png [image2.png ...]
//! ```
//!
//! Findings + mechanism: `benchmarks/issue50_topcliff_2026-08-02.md`.

use std::sync::OnceLock;

use zensim::profile::ProfileParams;
use zensim::{RgbSlice, Zensim, ZensimProfile};

fn load_png_rgb8(path: &str) -> (Vec<[u8; 3]>, usize, usize) {
    use zenpixels::SampleType;
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let cfg = zenpng::PngDecodeConfig::default();
    let out = zenpng::decode(&bytes, &cfg, &enough::Unstoppable).expect("zenpng decode");
    let (w, h) = (out.info.width as usize, out.info.height as usize);
    let layout = out.pixels.spec().layout;
    assert_eq!(layout.sample, SampleType::U8);
    let slice = out.pixels.as_view();
    let channels = layout.channels();
    let has_alpha = layout.order.has_alpha_slot();
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

/// Deterministic xorshift64* PRNG — the perturbation pattern is fixed per
/// (seed, grid cell) so runs are exactly reproducible.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Shift `frac` of pixels by ±`codes` on every channel (random sign per
/// pixel, clamped) — the issue #50 repro perturbation.
fn perturb(src: &[[u8; 3]], frac: f64, codes: i16, seed: u64) -> Vec<[u8; 3]> {
    let mut rng = Rng(seed | 1);
    let mut out = src.to_vec();
    for px in out.iter_mut() {
        if rng.unit() < frac {
            let delta = if rng.unit() < 0.5 { codes } else { -codes };
            for c in px.iter_mut() {
                *c = (*c as i16 + delta).clamp(0, 255) as u8;
            }
        }
    }
    out
}

static STRIPPED_BAKE: OnceLock<Vec<u8>> = OnceLock::new();
fn stripped_bake_bytes() -> &'static [u8] {
    STRIPPED_BAKE.get().expect("stripped bake loaded in main")
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut stripped_path: Option<String> = None;
    if let Some(i) = args.iter().position(|a| a == "--stripped-bake") {
        args.remove(i);
        stripped_path = Some(args.remove(i));
    }
    if args.is_empty() {
        eprintln!("usage: issue50_topcliff [--stripped-bake b_nospline.bin] <image.png>...");
        std::process::exit(2);
    }

    // Optional pre-spline arm: a Custom profile = B's exact runtime flags,
    // bake = B minus its output-calibration spline.
    let raw_profile: Option<ZensimProfile> = stripped_path.map(|p| {
        let bytes = std::fs::read(&p).unwrap_or_else(|e| panic!("read {p}: {e}"));
        STRIPPED_BAKE.set(bytes).expect("set once");
        let params: &'static ProfileParams = Box::leak(Box::new(
            ProfileParams::builder()
                .mlp(stripped_bake_bytes)
                .extended_features(true)
                .compute_iw_features(true)
                .skip_score_mapping(true)
                .extrapolate_score(true)
                .build(),
        ));
        ZensimProfile::Custom {
            name: "b-nospline",
            params,
        }
    });

    // The issue's grid plus a few extra near-top cells.
    let grid: &[(f64, i16)] = &[
        (0.0, 0),
        (0.0001, 1),
        (0.001, 1),
        (0.01, 1),
        (0.05, 1),
        (0.25, 1),
        (1.0, 1),
        (0.05, 2),
        (0.25, 2),
        (1.0, 2),
        (1.0, 4),
        (1.0, 8),
    ];

    println!("image\tfrac\tcodes\tscore_B\traw_pre_spline");
    for path in &args {
        let (src, w, h) = load_png_rgb8(path);
        let name = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(path);
        let zb = Zensim::new(ZensimProfile::B);
        let zraw = raw_profile.map(Zensim::new);
        // 944-regime bakes (SOTA-944 near-top arm): the Custom compute path
        // can't extract folded-append2 features itself, so score them via the
        // CANONICAL streaming extractor + score_features_with_profile (the
        // diffmap_block_coherence pattern). Raw == output for spline-less MLPs.
        #[cfg(feature = "feature-regime-v2")]
        let raw_n_in: usize = zraw
            .as_ref()
            .map(|_| {
                zenpredict::Model::from_bytes(STRIPPED_BAKE.get().unwrap())
                    .expect("parse bake")
                    .n_inputs()
            })
            .unwrap_or(0);
        #[cfg(feature = "feature-regime-v2")]
        let mut v2_scratch = zensim::feature_v2::V2Scratch::new();
        for &(frac, codes) in grid {
            let dst = if codes == 0 {
                src.clone()
            } else {
                perturb(&src, frac, codes, 0x5EED_0050)
            };
            let sref = RgbSlice::new(&src, w, h);
            let sdst = RgbSlice::new(&dst, w, h);
            let score = zb.compute(&sref, &sdst).expect("B compute").score();
            #[cfg(feature = "feature-regime-v2")]
            let raw = zraw.as_ref().map(|z| {
                if raw_n_in > 720 {
                    let toggles = zensim::feature_v2::V2NewFeatureToggles {
                        append2_block: raw_n_in == 944,
                        ..Default::default()
                    };
                    let feats = z
                        .compute_folded720_append_features_streaming(
                            &sref,
                            &sdst,
                            toggles,
                            &mut v2_scratch,
                        )
                        .expect("folded features")
                        .features()
                        .to_vec();
                    zensim::score_features_with_profile(
                        z.profile(),
                        &feats[..raw_n_in],
                        w as u32,
                        h as u32,
                    )
                    .expect("bake forward")
                } else {
                    z.compute(&sref, &sdst).expect("raw compute").score()
                }
            });
            #[cfg(not(feature = "feature-regime-v2"))]
            let raw = zraw
                .as_ref()
                .map(|z| z.compute(&sref, &sdst).expect("raw compute").score());
            match raw {
                Some(r) => println!("{name}\t{frac}\t{codes}\t{score:.4}\t{r:.6}"),
                None => println!("{name}\t{frac}\t{codes}\t{score:.4}\t-"),
            }
        }
    }
}
