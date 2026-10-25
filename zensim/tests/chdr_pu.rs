//! CHdr through the PU-linear HDR entry points (CHDR_PU, 2026-10-10).
//!
//! `ZensimProfile::CHdr` is the HDR candidate-of-record and declares 697 ids
//! of the 944 folded layout. Its training and validation features came from
//! the fleet's `zensim-foldapp2` HDR route: zenmetrics
//! `hdr::hdr_foldapp_features`, which calls
//! [`Zensim::compute_folded720_append2_features_hdr`] with
//! `HdrEncoding::Linear` (absolute cd/m²) and default toggles
//! (`benchmarks/hdr944_bake_wave_2026-08-27.md`, "hdrfeat944 (944-regime,
//! `zensim-foldapp2`, HdrEncoding::Linear arms)"). The PU entries must serve
//! CHdr, every input layout must agree bit for bit, and the served features
//! and score must equal today's build of that extraction. (The fleet-era
//! build differed by at most 1.1e-5 at 320 of the 697 ids, 1.5e-4 points of
//! score, from two later deliberate arithmetic changes:
//! `benchmarks/chdr_pu_2026-10-10.md`.)
#![cfg(feature = "candidate-profiles")]

use zensim::feature_v2::{HdrEncoding, V2NewFeatureToggles, V2Scratch};
use zensim::source::{AlphaMode, ImageSource, PixelFormat};
use zensim::{Zensim, ZensimProfile};

const CHDR_BAKE: &[u8] = include_bytes!("../weights/c_hdr_l1t1944_byid_2026-09-07.bin");

/// Absolute-linear cd/m² RGBA f32 source: the zenmetrics extractor's adapter
/// shape (`LinearF32Rgba`, opaque, declared HDR).
struct Nits {
    bytes: Vec<u8>,
    w: usize,
    h: usize,
}

impl Nits {
    fn new(rgb: &[f32], w: usize, h: usize) -> Self {
        let mut bytes = Vec::with_capacity(w * h * 16);
        for p in rgb.as_chunks::<3>().0 {
            for v in [p[0], p[1], p[2], 1.0f32] {
                bytes.extend_from_slice(&v.to_ne_bytes());
            }
        }
        Self { bytes, w, h }
    }
}

impl ImageSource for Nits {
    fn width(&self) -> usize {
        self.w
    }
    fn height(&self) -> usize {
        self.h
    }
    fn pixel_format(&self) -> PixelFormat {
        PixelFormat::LinearF32Rgba
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }
    fn is_hdr(&self) -> bool {
        true
    }
    fn row_bytes(&self, y: usize) -> &[u8] {
        &self.bytes[y * self.w * 16..(y + 1) * self.w * 16]
    }
}

#[path = "fixtures/hdr_pairs.rs"]
#[allow(dead_code)]
mod hdr_pairs;
use hdr_pairs::chdr_pu as pair;

fn planes(rgb: &[f32], w: usize, h: usize, stride: usize) -> [Vec<f32>; 3] {
    core::array::from_fn(|c| {
        let mut p = vec![-1.0f32; stride * h];
        for y in 0..h {
            for x in 0..w {
                p[y * stride + x] = rgb[(y * w + x) * 3 + c];
            }
        }
        p
    })
}

fn padded(rgb: &[f32], w: usize, h: usize, stride: usize) -> Vec<f32> {
    let mut v = vec![-1.0f32; stride * h];
    for y in 0..h {
        v[y * stride..y * stride + 3 * w].copy_from_slice(&rgb[y * 3 * w..(y + 1) * 3 * w]);
    }
    v
}

#[test]
fn chdr_scores_through_the_pu_entries_and_matches_its_validation_extraction() {
    let model = zenpredict::Model::from_bytes(CHDR_BAKE).expect("CHdr bake parses");
    let declared = zensim::declared_feature_ids(&model).expect("CHdr declares explicit ids");
    assert_eq!(declared.len(), 697);
    for (w, h, seed) in [
        (17usize, 19usize, 0x9e37_79b9_7f4a_7c15u64),
        (64, 64, 0x2545_f491_4f6c_dd1d),
        (97, 65, 0x1234_5678_9abc_def1),
        (256, 128, 0x0bad_c0de_dead_beef),
    ] {
        let tag = format!("{w}x{h}");
        let (r, d) = pair(w, h, seed);
        let z = Zensim::new(ZensimProfile::CHdr);

        let served = z
            .compute_pu_linear(&r, &d, w, h, 3 * w, 3 * w)
            .unwrap_or_else(|e| panic!("{tag}: CHdr refused by compute_pu_linear: {e:?}"));
        let score = served.score();
        assert!(score.is_finite(), "{tag}");

        // Every input layout serves the same bits.
        let (rs, ds) = (3 * w + 5, 3 * w + 5);
        let strided = z
            .compute_pu_linear(&padded(&r, w, h, rs), &padded(&d, w, h, ds), w, h, rs, ds)
            .unwrap();
        let (rp, dp) = (planes(&r, w, h, w), planes(&d, w, h, w));
        let planar = z
            .compute_pu_linear_planar([&rp[0], &rp[1], &rp[2]], [&dp[0], &dp[1], &dp[2]], w, h, w)
            .unwrap_or_else(|e| panic!("{tag}: CHdr refused by compute_pu_linear_planar: {e:?}"));
        let ps = w + 3;
        let (rq, dq) = (planes(&r, w, h, ps), planes(&d, w, h, ps));
        let planar_strided = z
            .compute_pu_linear_planar([&rq[0], &rq[1], &rq[2]], [&dq[0], &dq[1], &dq[2]], w, h, ps)
            .unwrap();
        let descriptor = z
            .compute(&Nits::new(&r, w, h), &Nits::new(&d, w, h))
            .unwrap();
        for (name, other) in [
            ("strided interleaved", &strided),
            ("planar", &planar),
            ("strided planar", &planar_strided),
            ("descriptor-HDR compute", &descriptor),
        ] {
            assert_eq!(
                other.score().to_bits(),
                score.to_bits(),
                "{tag}: {name} disagrees with interleaved"
            );
            assert_eq!(
                other.features(),
                served.features(),
                "{tag}: {name} features"
            );
        }

        // Today's build of the validation extraction (zenmetrics `hdr_foldapp_features`).
        let mut scratch = V2Scratch::new();
        let reference = z
            .compute_folded720_append2_features_hdr(
                &Nits::new(&r, w, h),
                &Nits::new(&d, w, h),
                HdrEncoding::Linear,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap()
            .into_features();
        assert_eq!(reference.len(), 944);
        for &id in &declared {
            let id = usize::from(id);
            assert_eq!(
                served.features()[id].to_bits(),
                reference[id].to_bits(),
                "{tag}: served f{id} {} vs the reference extraction {}",
                served.features()[id],
                reference[id]
            );
        }
        let recomputed = zensim::score_features_with_profile(
            ZensimProfile::CHdr,
            &reference,
            w as u32,
            h as u32,
        )
        .unwrap();
        assert_eq!(
            score.to_bits(),
            recomputed.to_bits(),
            "{tag}: served {score} vs feature-path recomputation {recomputed}"
        );

        // Identity stays exactly 100 on the PU entry.
        assert_eq!(
            z.compute_pu_linear(&r, &r, w, h, 3 * w, 3 * w)
                .unwrap()
                .score(),
            100.0
        );
    }
}

/// A live token that never fires: `may_stop()` is true, so `with_stop`
/// installs it, and every check passes.
struct NeverStop;

impl zensim::Stop for NeverStop {
    fn check(&self) -> Result<(), zensim::StopReason> {
        Ok(())
    }
}

/// A token that has already fired.
struct Fired;

impl zensim::Stop for Fired {
    fn check(&self) -> Result<(), zensim::StopReason> {
        Err(zensim::StopReason::Cancelled)
    }
}

/// Below Rev5 the wide-plan PU route follows SDR `compute`'s cancellation
/// contract (issue #48): the fold has no intra-walk stop hook, so a request
/// carrying a live stop token stays on the buffered walk. For C and CHdr that
/// walk cannot reach the declared ids, so the stoppable request is refused,
/// with exactly the error SDR `compute` gives the same profile and token.
/// Without a token both serve. A fired token returns `Cancelled` before any
/// work, and B (legacy PU walk) keeps its bits under a never-firing token.
#[test]
fn a_stoppable_wide_plan_pu_request_follows_the_sdr_cancellation_contract() {
    let (w, h) = (97usize, 65usize);
    let (r, d) = pair(w, h, 0x1234_5678_9abc_def1);
    let src: Vec<[u8; 3]> = (0..w * h)
        .map(|i| [(i * 7) as u8, (i * 13) as u8, (i * 3) as u8])
        .collect();
    let dst: Vec<[u8; 3]> = src
        .iter()
        .map(|p| [p[0].wrapping_add(9), p[1], p[2]])
        .collect();
    let (s8, d8) = (
        zensim::RgbSlice::new(&src, w, h),
        zensim::RgbSlice::new(&dst, w, h),
    );
    for (name, profile) in [("CHdr", ZensimProfile::CHdr), ("C", ZensimProfile::C)] {
        let plain = Zensim::new(profile);
        assert!(
            plain.compute_pu_linear(&r, &d, w, h, 3 * w, 3 * w).is_ok(),
            "{name}"
        );
        assert!(plain.compute(&s8, &d8).is_ok(), "{name}");

        let stoppable = Zensim::new(profile).with_stop(NeverStop);
        let sdr = format!("{:?}", stoppable.compute(&s8, &d8).unwrap_err());
        let pu = format!(
            "{:?}",
            stoppable
                .compute_pu_linear(&r, &d, w, h, 3 * w, 3 * w)
                .unwrap_err()
        );
        let (rp, dp) = (planes(&r, w, h, w), planes(&d, w, h, w));
        let planar = format!(
            "{:?}",
            stoppable
                .compute_pu_linear_planar(
                    [&rp[0], &rp[1], &rp[2]],
                    [&dp[0], &dp[1], &dp[2]],
                    w,
                    h,
                    w
                )
                .unwrap_err()
        );
        let desc = format!(
            "{:?}",
            stoppable
                .compute(&Nits::new(&r, w, h), &Nits::new(&d, w, h))
                .unwrap_err()
        );
        assert!(
            sdr.contains("does not reach"),
            "{name}: SDR refusal changed: {sdr}"
        );
        for (entry, e) in [
            ("interleaved", &pu),
            ("planar", &planar),
            ("descriptor", &desc),
        ] {
            assert_eq!(
                e, &sdr,
                "{name}: stoppable PU {entry} must refuse as SDR compute does"
            );
        }

        let fired = Zensim::new(profile).with_stop(Fired);
        assert!(
            matches!(
                fired.compute_pu_linear(&r, &d, w, h, 3 * w, 3 * w),
                Err(zensim::ZensimError::Cancelled { .. })
            ),
            "{name}: a fired token must cancel before any work"
        );
    }
    let b = Zensim::new(ZensimProfile::B);
    let b_stoppable = Zensim::new(ZensimProfile::B).with_stop(NeverStop);
    assert_eq!(
        b.compute_pu_linear(&r, &d, w, h, 3 * w, 3 * w)
            .unwrap()
            .score()
            .to_bits(),
        b_stoppable
            .compute_pu_linear(&r, &d, w, h, 3 * w, 3 * w)
            .unwrap()
            .score()
            .to_bits()
    );
}

/// Below Rev5 a wide-plan identity pair short-circuits as SDR `compute` does:
/// exactly 100, with zeros at the plan width rather than a computed walk.
#[test]
fn wide_plan_pu_identity_matches_sdr_identity() {
    let (w, h) = (64usize, 64usize);
    let (r, _) = pair(w, h, 0x2545_f491_4f6c_dd1d);
    let src: Vec<[u8; 3]> = (0..w * h)
        .map(|i| [(i * 5) as u8, (i * 11) as u8, i as u8])
        .collect();
    let s8 = zensim::RgbSlice::new(&src, w, h);
    // C fed nits is served by CHdr's weights and plan (PU-linear routes C to
    // CHdr since 2026-10-11), so its PU identity matches CHdr's SDR identity.
    for (profile, served) in [
        (ZensimProfile::CHdr, ZensimProfile::CHdr),
        (ZensimProfile::C, ZensimProfile::CHdr),
    ] {
        let pu = Zensim::new(profile)
            .compute_pu_linear(&r, &r, w, h, 3 * w, 3 * w)
            .unwrap();
        let sdr = Zensim::new(served).compute(&s8, &s8).unwrap();
        assert_eq!(pu.score(), 100.0);
        assert_eq!(pu.features().len(), sdr.features().len());
        assert!(pu.features().iter().all(|&v| v == 0.0));
    }
}

/// PU-linear maps built-in profile C to CHdr, as it maps B to BHdr (owner
/// decision 2026-10-11): C fed nits through every PU-linear entry returns
/// exactly CHdr's score and features through the same entry, tagged `C`.
/// C's SDR path keeps C's own weights.
#[test]
fn c_routes_to_chdr_through_every_pu_entry_bit_identically() {
    for (w, h, seed) in [
        (17usize, 19usize, 0x9e37_79b9_7f4a_7c15u64),
        (97, 65, 0x1234_5678_9abc_def1),
        (256, 128, 0x0bad_c0de_dead_beef),
    ] {
        let tag = format!("{w}x{h}");
        let (r, d) = pair(w, h, seed);
        let (c, chdr) = (
            Zensim::new(ZensimProfile::C),
            Zensim::new(ZensimProfile::CHdr),
        );
        let (rs, ds) = (3 * w + 5, 3 * w + 5);
        let (rp, dp) = (planes(&r, w, h, w), planes(&d, w, h, w));
        type Entry<'a> = &'a dyn Fn(&Zensim) -> Result<zensim::ZensimResult, zensim::ZensimError>;
        let entries: [(&str, bool, Entry); 5] = [
            ("interleaved", true, &|z| {
                z.compute_pu_linear(&r, &d, w, h, 3 * w, 3 * w)
            }),
            ("strided interleaved", true, &|z| {
                z.compute_pu_linear(&padded(&r, w, h, rs), &padded(&d, w, h, ds), w, h, rs, ds)
            }),
            ("planar", true, &|z| {
                z.compute_pu_linear_planar(
                    [&rp[0], &rp[1], &rp[2]],
                    [&dp[0], &dp[1], &dp[2]],
                    w,
                    h,
                    w,
                )
            }),
            ("extended features", false, &|z| {
                z.compute_pu_linear_extended_features(&r, &d, w, h, 3 * w, 3 * w)
            }),
            ("descriptor-HDR compute", true, &|z| {
                z.compute(&Nits::new(&r, w, h), &Nits::new(&d, w, h))
            }),
        ];
        for (name, serves, entry) in entries {
            let (got, want) = match (entry(&c), entry(&chdr)) {
                (Ok(got), Ok(want)) => (got, want),
                // CHdr's extended-features PU entry refuses (it predates this
                // routing); C must refuse with exactly the same error.
                (Err(got), Err(want)) if !serves => {
                    assert_eq!(format!("{got:?}"), format!("{want:?}"), "{tag} {name}");
                    continue;
                }
                (got, want) => panic!(
                    "{tag} {name}: C {:?} vs CHdr {:?}",
                    got.map(|r| r.score()),
                    want.map(|r| r.score())
                ),
            };
            assert!(
                serves,
                "{tag} {name}: CHdr now serves this entry; update the test"
            );
            assert_eq!(
                got.score().to_bits(),
                want.score().to_bits(),
                "{tag} {name}: C {} vs CHdr {}",
                got.score(),
                want.score()
            );
            assert_eq!(got.features(), want.features(), "{tag} {name}: features");
            assert!(
                matches!(got.profile(), ZensimProfile::C),
                "{tag} {name}: the result keeps the C tag"
            );
        }
    }

    // SDR input still scores with C's own weights, not CHdr's.
    let (w, h) = (97usize, 65usize);
    let src: Vec<[u8; 3]> = (0..w * h)
        .map(|i| [(i * 7) as u8, (i * 13) as u8, (i * 3) as u8])
        .collect();
    let dst: Vec<[u8; 3]> = src
        .iter()
        .map(|p| [p[0].wrapping_add(9), p[1], p[2]])
        .collect();
    let (s8, d8) = (
        zensim::RgbSlice::new(&src, w, h),
        zensim::RgbSlice::new(&dst, w, h),
    );
    let c_sdr = Zensim::new(ZensimProfile::C).compute(&s8, &d8).unwrap();
    let recomputed =
        zensim::score_features_with_profile(ZensimProfile::C, c_sdr.features(), w as u32, h as u32)
            .unwrap();
    assert_eq!(c_sdr.score().to_bits(), recomputed.to_bits());
    let chdr_sdr = Zensim::new(ZensimProfile::CHdr).compute(&s8, &d8).unwrap();
    assert_ne!(
        c_sdr.score().to_bits(),
        chdr_sdr.score().to_bits(),
        "C's SDR path must not route to CHdr"
    );
    assert_ne!(c_sdr.features().len(), chdr_sdr.features().len());
}
