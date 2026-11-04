//! `approx_ssim2` / `approx_dssim` / `approx_butteraugli` on a bake-scored
//! profile.
//!
//! The three power laws were fitted on the linear preview-v0.2 distance. A
//! bake-scored profile (A, B, C, D and their HDR variants) computes that
//! distance and then `set_mlp_score` replaces `raw_distance` with the bake's
//! output, which is a calibrated score. Fed that score, `approx_ssim2`
//! floored at -100 and `approx_butteraugli` read about 30 on ordinary JPEG
//! pairs, and a negative score read as a perfect 100 / 0. Every SDR bake
//! profile carries the same linear weights and front end as `PreviewV0_2`, so
//! its approximations must match that profile's. Not bit for bit: the
//! 372-wide extraction fills a few basic slots slightly differently (up to
//! 0.02 on these images), which moves the approximations by about 1e-7
//! relative, against a bug of 167 ssimulacra2 points on real JPEG pairs.
//! BHdr and CHdr are left out: they want absolute-luminance input.

use zensim::{RgbSlice, Zensim, ZensimProfile};

const W: usize = 192;
const H: usize = 160;

/// Every SDR bake-scored profile this build carries.
#[allow(deprecated)] // `A` stays behind the default-on `deprecated-profiles` feature
fn bake_profiles() -> Vec<(&'static str, ZensimProfile)> {
    let mut v = vec![("B", ZensimProfile::B)];
    #[cfg(feature = "deprecated-profiles")]
    v.push(("A", ZensimProfile::A));
    #[cfg(feature = "candidate-profiles")]
    v.extend([("C", ZensimProfile::C), ("D", ZensimProfile::D)]);
    v
}

/// Deterministic textured image: two gradients, a sine pattern and hashed
/// noise, so every scale has detail to lose.
fn source() -> Vec<[u8; 3]> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    (0..W * H)
        .map(|i| {
            let (x, y) = ((i % W) as f64, (i / W) as f64);
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let n = (state >> 59) as f64;
            let s = 40.0 * ((x * 0.31).sin() * (y * 0.17).cos());
            let px = |base: f64| (base + s + n).clamp(0.0, 255.0) as u8;
            [px(x), px(y * 1.4), px(128.0 - x * 0.3 + y * 0.2)]
        })
        .collect()
}

/// Separable box blur of the given radius: a monotone distortion ladder.
fn blur(src: &[[u8; 3]], r: usize) -> Vec<[u8; 3]> {
    let pass = |img: &[[u8; 3]], horizontal: bool| -> Vec<[u8; 3]> {
        (0..W * H)
            .map(|i| {
                let (x, y) = (i % W, i / W);
                let mut acc = [0u32; 3];
                let mut n = 0u32;
                for k in 0..=2 * r {
                    let (xx, yy) = if horizontal {
                        ((x + k).saturating_sub(r).min(W - 1), y)
                    } else {
                        (x, (y + k).saturating_sub(r).min(H - 1))
                    };
                    let p = img[yy * W + xx];
                    for c in 0..3 {
                        acc[c] += p[c] as u32;
                    }
                    n += 1;
                }
                [0, 1, 2].map(|c| ((acc[c] + n / 2) / n) as u8)
            })
            .collect()
    };
    pass(&pass(src, true), false)
}

fn compute(profile: ZensimProfile, a: &[[u8; 3]], b: &[[u8; 3]]) -> zensim::ZensimResult {
    Zensim::new(profile)
        .with_parallel(false)
        .compute(&RgbSlice::new(a, W, H), &RgbSlice::new(b, W, H))
        .unwrap()
}

#[test]
fn bake_approximations_equal_preview_v0_2() {
    let src = source();
    for r in [1, 2, 4] {
        let dst = blur(&src, r);
        let v02 = compute(ZensimProfile::PreviewV0_2, &src, &dst);
        for (p, profile) in bake_profiles() {
            let b = compute(profile, &src, &dst);
            // Same weights, same front end: the linear distance matches, so
            // the approximations do too. The bake's raw_distance is not that
            // distance.
            assert!(
                (b.raw_distance() - v02.raw_distance()).abs() > 1.0,
                "{p} r={r}"
            );
            for (name, x, y) in [
                ("approx_ssim2", b.approx_ssim2(), v02.approx_ssim2()),
                ("approx_dssim", b.approx_dssim(), v02.approx_dssim()),
                (
                    "approx_butteraugli",
                    b.approx_butteraugli(),
                    v02.approx_butteraugli(),
                ),
            ] {
                assert!(
                    (x - y).abs() <= 1e-6 * y.abs().max(1e-3),
                    "{name} {p} r={r}: {x} vs PreviewV0_2 {y}"
                );
            }
        }
    }
}

#[test]
fn b_approximations_track_distortion() {
    let src = source();
    let results: Vec<_> = [1, 2, 4]
        .map(|r| compute(ZensimProfile::B, &src, &blur(&src, r)))
        .into();
    for w in results.windows(2) {
        assert!(w[1].approx_ssim2() < w[0].approx_ssim2());
        assert!(w[1].approx_butteraugli() > w[0].approx_butteraugli());
        assert!(w[1].approx_dssim() > w[0].approx_dssim());
    }
    for r in &results {
        assert!(
            (-100.0..100.0).contains(&r.approx_ssim2()),
            "{}",
            r.approx_ssim2()
        );
        assert!(r.approx_butteraugli() > 0.0 && r.approx_butteraugli().is_finite());
    }
}

/// A negative bake score used to land on the `raw_distance <= 0` branch and
/// read as a perfect pair. D scores this blur at about -15.
#[cfg(feature = "candidate-profiles")]
#[test]
fn negative_bake_score_is_not_a_perfect_pair() {
    let src = source();
    let d = compute(ZensimProfile::D, &src, &blur(&src, 4));
    assert!(
        d.score() < 0.0,
        "fixture no longer drives D negative: {}",
        d.score()
    );
    assert!(d.approx_ssim2() < 50.0, "{}", d.approx_ssim2());
    assert!(d.approx_butteraugli() > 1.0, "{}", d.approx_butteraugli());
}

#[test]
fn b_identical_pair_is_exact() {
    let src = source();
    let r = compute(ZensimProfile::B, &src, &src);
    assert_eq!(r.approx_ssim2(), 100.0);
    assert_eq!(r.approx_dssim(), 0.0);
    assert_eq!(r.approx_butteraugli(), 0.0);
}
