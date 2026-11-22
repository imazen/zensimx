use super::kernels::RowKernels;
use super::{
    FEATURE_NAMES, IntensityTarget, RgbRows, Variant, compute, extract_fx2, kernels, stream,
};
use archmage::{
    ScalarToken,
    testing::{CompileTimePolicy, for_each_token_permutation},
};
use magetypes::simd::generic::f32x16;

const VARIANTS: [Variant; 3] = [Variant::AlgS4, Variant::AlgS5, Variant::HybridS5];

fn view(pixels: &[[u8; 3]], w: usize, h: usize) -> RgbRows<'_> {
    RgbRows::new(pixels, w, h, w).unwrap()
}

fn gray(w: usize, h: usize, mut f: impl FnMut(usize, usize) -> u8) -> Vec<[u8; 3]> {
    (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .map(|(x, y)| [f(x, y); 3])
        .collect()
}

fn pair(w: usize, h: usize, recipe: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let mut state = (w as u32) ^ (h as u32).rotate_left(7) ^ (recipe as u32 + 1);
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state >> 24) as u8
    };
    let mut a = Vec::new();
    let mut b = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let pa = match recipe {
                0 => [0; 3],
                1 => [255; 3],
                2 => [128; 3],
                3 => [if (x + y) % 2 == 0 { 16 } else { 240 }; 3],
                4 => [next(), next(), next()],
                5 => [(x * 4) as u8, (y * 3) as u8, ((x + y) * 2) as u8],
                6 => [((x / 8 + y / 8) * 31) as u8; 3],
                _ => [0, if (x + y) % 2 == 0 { 255 } else { 64 }, 255],
            };
            let pb = match recipe {
                0 => [u8::from(x == w / 2 && y == h / 2); 3],
                1 => [254; 3],
                2 => [129; 3],
                3 => [116; 3],
                4 => pa.map(|v| v.saturating_add(next() & 15)),
                5 => pa.map(|v| v / 7 * 7),
                6 => [(pa[0] as i16 + (x % 5) as i16 * 6 - 12).clamp(0, 255) as u8; 3],
                _ => [0, pa[1], 252],
            };
            a.push(pa);
            b.push(pb);
        }
    }
    (a, b)
}

fn checked_values(result: &super::Result) -> Vec<f64> {
    assert!(result.score.is_finite() && (0.0..=100.0).contains(&result.score));
    let mut values = vec![result.score];
    for group in result.features.iter().flatten() {
        for (value, name) in group.iter().zip(FEATURE_NAMES) {
            assert!(value.is_finite() && *value >= 0.0, "{name}={value}");
            values.push(*value);
        }
    }
    values
}

fn assert_bits(a: &super::Result, b: &super::Result) {
    assert_eq!(a.score.to_bits(), b.score.to_bits());
    for (a, b) in a.features.iter().flatten().zip(b.features.iter().flatten()) {
        for ((a, b), name) in a.iter().zip(b).zip(FEATURE_NAMES) {
            assert_eq!(a.to_bits(), b.to_bits(), "{name}: {a} vs {b}");
        }
    }
}

fn cases() -> impl Iterator<Item = (usize, usize, usize)> {
    [
        (1, 1),
        (3, 3),
        (17, 13),
        (64, 64),
        (65, 65),
        (256, 256),
        (256, 191),
        (840, 944),
    ]
    .into_iter()
    .flat_map(|(w, h)| (0..if h == 944 { 2 } else { 8 }).map(move |recipe| (w, h, recipe)))
}

#[test]
fn exact_symmetry_identity_and_thread_trees() {
    #[cfg(feature = "threads")]
    let pools = [1, 2, 4, 8, 16].map(|n| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap()
    });
    let mut pairs = 0;
    for (w, h, recipe) in cases() {
        let (a, b) = pair(w, h, recipe);
        let a = view(&a, w, h);
        let b = view(&b, w, h);
        for variant in VARIANTS {
            let ab = compute(a, b, variant, false).unwrap();
            let ba = compute(b, a, variant, false).unwrap();
            let _ = checked_values(&ab);
            assert_bits(&ab, &ba);
            let aa = compute(a, a, variant, false).unwrap();
            assert_eq!(aa.score.to_bits(), 100.0_f64.to_bits());
            assert!(
                aa.features
                    .iter()
                    .flatten()
                    .flatten()
                    .all(|v| v.to_bits() == 0)
            );
            #[cfg(feature = "threads")]
            if matches!(
                (w, h, recipe),
                (256, 256, 4) | (256, 191, 3) | (840, 944, 1)
            ) {
                for pool in &pools {
                    let parallel = pool.install(|| compute(a, b, variant, true).unwrap());
                    let parallel_reverse = pool.install(|| compute(b, a, variant, true).unwrap());
                    assert_bits(&ab, &parallel);
                    assert_bits(&ab, &parallel_reverse);
                }
            }
            if recipe == 2 || recipe == 1 {
                assert!(
                    ab.score < 100.0 && ab.features[0][1][8] > 0.0,
                    "{w}x{h} {recipe}"
                );
            }
        }
        pairs += 1;
    }
    assert!(pairs >= 50);
    println!("REV8_SYMMETRY {pairs} RGB8 pairs, three variants");
}

#[test]
fn stride_validation_and_reflect_singletons() {
    let (a, b) = pair(17, 13, 4);
    let stride = 23;
    let mut pad = vec![[0, 255, 12]; 12 * stride + 17];
    for (i, row) in a.as_chunks::<17>().0.iter().enumerate() {
        pad[i * stride..i * stride + 17].copy_from_slice(row);
    }
    assert!(RgbRows::new(&pad, 17, 13, 16).is_none());
    assert!(RgbRows::new(&pad[..pad.len() - 1], 17, 13, stride).is_none());
    assert!(RgbRows::new(&pad, 17, 0, stride).is_none());
    assert!(RgbRows::new(&pad, 1, usize::MAX, usize::MAX).is_none());
    assert!(compute(view(&a, 17, 13), view(&b, 13, 17), Variant::AlgS5, false).is_none());
    for variant in VARIANTS {
        let wide = compute(
            RgbRows::new(&pad, 17, 13, stride).unwrap(),
            view(&b, 17, 13),
            variant,
            false,
        )
        .unwrap();
        let tight = compute(view(&a, 17, 13), view(&b, 17, 13), variant, false).unwrap();
        assert_bits(&wide, &tight);
        for (w, h) in [(1, 3), (11, 1), (1, 41), (41, 1)] {
            let flat = vec![[17, 208, 53]; w * h];
            let result = compute(view(&flat, w, h), view(&flat, w, h), variant, false).unwrap();
            assert_eq!(result.score, 100.0);
        }
    }
}

#[test]
fn reciprocal_cube_root_exp_mask_and_negative_variance_rails() {
    let math = kernels::RowMath(ScalarToken);
    let splat = |v| f32x16::splat_t(ScalarToken, v);
    let eval = |v: f32x16<ScalarToken>| v.to_array()[0];
    for x in [
        0.0,
        -0.0,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        1e-12,
        0.0009,
        0.25,
        1.0,
        17.0,
        1e20,
    ] {
        let got = eval(math.rcp_nr1(splat(x), 1e-12));
        let expected = 1.0 / x.max(1e-12);
        assert!(
            (got - expected).abs() <= 3.0 * f32::EPSILON * expected,
            "rcp {x}: {got} != {expected}"
        );
    }
    assert_eq!(eval(math.rcp_nr1(splat(f32::INFINITY), 1e-12)).to_bits(), 0);
    for x in [-1.0f32, -0.003_793_073_4, -0.0, 0.0, 0.003_793_073_4, 1.0] {
        let actual = eval(math.cube_root_and_add(splat(x), 0.0));
        if x == 0.0 {
            assert_eq!(actual.to_bits(), x.to_bits());
        } else {
            assert!(
                (actual - x.cbrt()).abs() <= 4.0 * f32::EPSILON,
                "{x}: {actual}"
            );
        }
    }
    let mut black = [vec![0.0; 32], vec![0.0; 32], vec![0.0; 32]];
    math.color::<false>(&[[0; 3]], &mut black);
    assert_eq!(black.map(|channel| channel[5]), [0.42, 0.01, 0.55]);

    let mut black500 = [vec![0.0; 32], vec![0.0; 32], vec![0.0; 32]];
    math.color::<true>(&[[0; 3]], &mut black500);
    assert_eq!(black500.map(|channel| channel[5]), [0.42, 0.01, 0.55]);

    let mut white255 = [vec![0.0; 32], vec![0.0; 32], vec![0.0; 32]];
    let mut white500 = [vec![0.0; 32], vec![0.0; 32], vec![0.0; 32]];
    math.color::<false>(&[[255; 3]], &mut white255);
    math.color::<true>(&[[255; 3]], &mut white500);
    for c in 0..3 {
        assert!(
            (white255[c][5] - white500[c][5]).abs() <= 2e-7,
            "white ch {c}"
        );
    }

    let exp = |x| eval(math.negative_exp(splat(x)));
    assert_eq!(exp(0.0), 1.0);
    assert_eq!(exp(f32::NEG_INFINITY).to_bits(), 0);
    assert_eq!(exp(kernels::EXP_CUTOFF).to_bits(), 0);
    assert!(exp(kernels::EXP_CUTOFF.next_up()) > 0.0);
    assert!(exp(f32::NAN).is_nan() && exp(0.1).is_nan());
    let mut last = 0.0;
    for i in (-3200..=0).map(|i| i as f32 / 100.0) {
        let actual = exp(i);
        assert!(actual >= last, "{i}: {last} > {actual}");
        assert!((actual - i.exp()).abs() <= 4e-7);
        last = actual;
    }
    let flat = 0.7f32;
    assert!((-flat).mul_add(flat, flat * flat) < 0.0);
    assert_eq!(
        eval(math.moment_variance(
            splat(flat * 121.0),
            splat(((flat * flat) * 121.0).next_down().next_down().next_down())
        )),
        0.0
    );
    for sigma in [0.0, 1e-12, 0.03, 0.05, 0.1, 0.6] {
        let weights = math.visibility(splat(sigma)).map(eval);
        for ((w, a), p) in weights
            .into_iter()
            .zip([0.03, 0.0025, 0.0001])
            .zip([1, 2, 4])
        {
            let expected = (a / sigma.max(1e-12).powi(p)).min(1.0);
            assert!(w.is_finite() && (w - expected).abs() <= 4e-7);
        }
    }
}

#[test]
fn window_rebase_preserves_entering_tail() {
    let math = kernels::RowMath(ScalarToken);
    let source = |v| core::array::from_fn(|_| vec![v; stream::LANES]);
    let big: stream::Moments = source(1.0);
    let small = source(1e-8);
    let zero = source(0.0);
    let mut output = source(0.0);
    let mut taps = [&big; 11];
    math.vertical(&taps, &mut output, stream::LANES);
    assert_eq!(output[0][0], 11.0);
    taps = [&zero; 11];
    taps[10] = &small;
    math.vertical(&taps, &mut output, stream::LANES);
    assert_eq!(output[0][0].to_bits(), 1e-8_f32.to_bits());

    let reference = [0.7_f32; 121];
    let mut changed = reference;
    changed[61] = changed[61].next_up();
    let e = changed[61] - reference[61];
    let expected = f64::from(e).powi(2) * (120.0 / 121.0_f64.powi(2));
    let from_error = math
        .moment_variance(
            f32x16::splat_t(ScalarToken, e),
            f32x16::splat_t(ScalarToken, e * e),
        )
        .to_array()[0];
    assert!((f64::from(from_error) / expected - 1.0).abs() < 2e-7);
}

#[test]
fn exact_orientation_ties_and_huge_gaps() {
    let f = [0.01, 0.1, 1.0, 10.0];
    let r = [3.0, 2.0, 1.0, 0.4];
    for (a, b) in [
        (0.0, -0.0),
        (0.01, 0.01_f64.next_up()),
        (0.015, 0.016),
        (1.0, -1.0),
        (f64::MAX, f64::MIN),
    ] {
        let forward = stream::blend(a, b, f, r);
        let reverse = stream::blend(b, a, r, f);
        assert_eq!(forward.map(f64::to_bits), reverse.map(f64::to_bits));
        assert!(forward.into_iter().all(f64::is_finite));
    }
    assert_eq!(
        stream::blend(0.4, 0.4, f, r),
        core::array::from_fn(|i| (f[i] + r[i]) * 0.5)
    );
}

#[test]
fn spatial_diagnostics_for_fragility_and_discount() {
    let (w, h) = (64, 64);
    let flat = gray(w, h, |_, _| 128);
    let transition = gray(w, h, |x, y| {
        if x < 32 {
            128
        } else {
            96 + ((x + y) % 2) as u8 * 64
        }
    });
    let periodic = gray(w, h, |x, y| {
        [103, 108, 121, 135, 149, 153, 149, 135, 121, 108][(x + y) % 10]
    });
    let blur = gray(w, h, |x, y| {
        ([(x + 63) % w, x, (x + 1) % w]
            .into_iter()
            .map(|x| u16::from(periodic[y * w + x][0]))
            .sum::<u16>()
            / 3) as u8
    });
    let grain = gray(w, h, |x, y| {
        let n = ((x * 157 + y * 131 + x * y * 31) % 29) as i16 - 14;
        (i16::from(periodic[y * w + x][0]) + n) as u8
    });
    let shifted = periodic
        .iter()
        .map(|p| p.map(|v| v + 16))
        .collect::<Vec<_>>();
    let exposure = periodic
        .iter()
        .map(|p| p.map(|v| (u16::from(v) * 7 / 4).min(255) as u8))
        .collect::<Vec<_>>();
    let clipped = periodic
        .iter()
        .map(|p| p.map(|v| v.min(130)))
        .collect::<Vec<_>>();
    let luma_control = gray(w, h, |x, y| periodic[y * w + (x + 1) % w][0]);
    let blocks = gray(w, h, |x, _| 112 + 32 * ((x / 8) % 2) as u8);
    let rephased = gray(w, h, |x, _| 112 + 32 * (((x + 3) / 8) % 2) as u8);
    let non_grid = gray(w, h, |x, _| 112 + 32 * (((x + 2) / 7) % 2) as u8);
    let ringing = gray(w, h, |x, _| [108, 144, 121, 128, 128, 128, 128, 128][x % 8]);
    let noise = gray(w, h, |x, y| {
        92 + ((x * 157 + y * 131 + x * y * 31) % 73) as u8
    });
    let mut gaps = Vec::new();
    for (name, a, b) in [
        ("flat-texture", &flat, &transition),
        ("blur-grain", &blur, &grain),
        ("brightness", &periodic, &shifted),
        ("exposure", &periodic, &exposure),
        ("clipping", &periodic, &clipped),
        ("luminance-preserving", &periodic, &luma_control),
        ("grid-rephase", &blocks, &rephased),
        ("off-grid", &blocks, &non_grid),
        ("ringing", &flat, &ringing),
        ("noise-periodic", &noise, &periodic),
        ("tie", &flat, &flat),
    ] {
        let witness = stream::diagnostic(view(a, w, h), view(b, w, h));
        let [a, b] = witness.orientation_parts;
        for (phi, [activity, bk, rg]) in witness.orientation.into_iter().zip([a, b]) {
            assert!(
                (phi - (activity - f64::from(kernels::ALPHA) * bk - f64::from(kernels::BETA) * rg))
                    .abs()
                    < 1e-7
            );
        }
        let max = witness.features[18];
        let mean = witness.mean_fragility;
        assert!(max >= mean);
        if matches!(name, "flat-texture" | "blur-grain" | "clipping") {
            assert!(max > mean && mean > 0.0);
        }
        if name == "grid-rephase" {
            assert!(a[1] > 0.0 && b[1] == 0.0);
        }
        if name == "off-grid" {
            assert!(a[1] > b[1] && b[1] > 0.0);
        }
        if name == "ringing" {
            assert!(a[2] == 0.0 && b[2] > 0.0);
        }
        let gap = (witness.orientation[0] - witness.orientation[1]).abs();
        gaps.push(gap);
        println!(
            "REV8_WITNESS {name} frag_max={max:.9e} frag_mean={mean:.9e} A={a:?} B={b:?} phi={:?} |gap|={gap:.9e}",
            witness.orientation
        );
    }
    gaps.sort_by(f64::total_cmp);
    println!(
        "REV8_TIES {} cases, exact={}, min={:.9e}, median={:.9e}, max={:.9e}",
        gaps.len(),
        gaps.iter().filter(|&&v| v == 0.0).count(),
        gaps[0],
        gaps[gaps.len() / 2],
        gaps[gaps.len() - 1]
    );
}

#[test]
fn tier_permutations_measure_drift_and_require_exact_swap() {
    const ISOLATED: &str = "ZENSIM_REV8_TIER_CHILD";
    if std::env::var_os(ISOLATED).is_none() {
        // Token overrides are process-wide. Isolate the sweep from the
        // shipping tests even when cargo test uses many test threads.
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "rev8::tests::tier_permutations_measure_drift_and_require_exact_swap",
                "--nocapture",
            ])
            .env(ISOLATED, "1")
            .output()
            .unwrap();
        print!("{}", String::from_utf8_lossy(&output.stdout));
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
        assert!(output.status.success());
        return;
    }
    let mut baseline = None;
    let mut seen = std::collections::BTreeSet::new();
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |permutation| {
        let tier = super::tier();
        if !seen.insert(tier) {
            return;
        }
        let mut values = Vec::new();
        for (w, h, recipe) in cases() {
            let (a, b) = pair(w, h, recipe);
            let a = view(&a, w, h);
            let b = view(&b, w, h);
            for variant in VARIANTS {
                let ab = compute(a, b, variant, false).unwrap();
                assert_bits(&ab, &compute(b, a, variant, false).unwrap());
                values.extend(checked_values(&ab));
            }
        }
        if let Some((reference_tier, reference)) = &baseline {
            let mut max_abs = 0.0_f64;
            let mut max_rel = 0.0_f64;
            let mut max_f32_ulp = 0;
            let mut max_f64_ulp = 0;
            let mut changed = 0;
            let mut worst = (0, 0.0, 0.0);
            for (i, (&a, &b)) in values.iter().zip(reference as &Vec<f64>).enumerate() {
                let abs = (a - b).abs();
                if abs > max_abs {
                    max_abs = abs;
                    worst = (i, a, b);
                }
                if a.max(b) > 0.0 {
                    max_rel = max_rel.max(abs / a.max(b));
                }
                max_f64_ulp = max_f64_ulp.max(a.to_bits().abs_diff(b.to_bits()));
                max_f32_ulp = max_f32_ulp.max((a as f32).to_bits().abs_diff((b as f32).to_bits()));
                changed += usize::from(a.to_bits() != b.to_bits());
            }
            println!(
                "REV8_TIER {tier} vs {reference_tier} {} changed={changed} max_abs={max_abs:.9e} max_rel={max_rel:.9e} f32_ulp={max_f32_ulp} f64_ulp={max_f64_ulp} worst={worst:?} ({})",
                values.len(),
                permutation.label
            );
            assert!(max_abs < 1e-4, "{tier}: {max_abs}");
        } else {
            baseline = Some((tier, values));
        }
    });
    assert!(report.permutations_run >= seen.len() && !seen.is_empty());
}

#[test]
fn fixture_pchip_slopes_match_canonical_owner() {
    let xs = [0.0, 0.125, 0.5, 1.5];
    let ys = [100.0, 90.0, 55.0, 0.0];
    let expected = crate::score_math::pchip_derivs(&xs, &ys);
    for (a, b) in [
        (-230.0 / 3.0),
        (-6720.0 / 79.0),
        (-101640.0 / 1411.0),
        (-895.0 / 33.0),
    ]
    .into_iter()
    .zip(expected)
    {
        assert!((a - b).abs() < 2e-14);
    }
}

#[test]
fn streamed_moments_track_direct_f64_windows() {
    use crate::color::*;

    let xy = |p: [u8; 3]| {
        let [r, g, b] = p.map(|x| {
            let v = f64::from(x) / 255.0;
            if v <= 0.04045 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        });
        let bias = f64::from(K_B0);
        let sample =
            |a, d, c| (r * f64::from(a) + g * f64::from(d) + b * f64::from(c) + bias).cbrt();
        let l = sample(K_M00, K_M01, K_M02) - bias.cbrt();
        let m = sample(K_M10, K_M11, K_M12) - bias.cbrt();
        let s = sample(K_M20, K_M21, K_M22) - bias.cbrt();
        let y = (l + m) * 0.5;
        [(l - m) * 7.0 + 0.42, y + 0.01, s - y + 0.55]
    };
    let reflect = |mut x: isize, n: usize| {
        if n == 1 {
            return 0;
        }
        let high = n as isize - 1;
        while x < 0 || x > high {
            x = if x < 0 { -x } else { 2 * high - x };
        }
        x as usize
    };
    let mut max_error = [0.0_f64; 3];
    for (w, h, recipe) in [(1, 3, 2), (3, 1, 5), (17, 13, 4), (31, 23, 3), (33, 17, 7)] {
        let (a, b) = pair(w, h, recipe);
        let result = compute(view(&a, w, h), view(&b, w, h), Variant::AlgS5, false).unwrap();
        let mut images = [
            a.into_iter().map(xy).collect::<Vec<_>>(),
            b.into_iter().map(xy).collect(),
        ];
        let (mut w, mut h) = (w, h);
        for (s, scale) in result.features.iter().enumerate() {
            for (c, terms) in scale.iter().enumerate() {
                let mut expect = [0.0; 3];
                for y in 0..h {
                    for x in 0..w {
                        let mut a = [0.0; 121];
                        let mut b = [0.0; 121];
                        for (i, (dy, dx)) in (-5..=5)
                            .flat_map(|dy| (-5..=5).map(move |dx| (dy, dx)))
                            .enumerate()
                        {
                            let ixy = reflect(y as isize + dy, h) * w + reflect(x as isize + dx, w);
                            a[i] = images[0][ixy][c];
                            b[i] = images[1][ixy][c];
                        }
                        let error: [f64; 121] = core::array::from_fn(|i| a[i] - b[i]);
                        let mean = |v: &[f64; 121]| v.iter().sum::<f64>() / 121.0;
                        let variance = |v: &[f64; 121]| {
                            let m = mean(v);
                            v.iter().map(|v| (v - m).powi(2)).sum::<f64>() / 121.0
                        };
                        expect[0] +=
                            mean(&error).powi(2) / (mean(&a).powi(2) + mean(&b).powi(2) + 0.0009);
                        expect[1] += variance(&error) / (variance(&a) + variance(&b) + 0.0009);
                        expect[2] += (images[0][y * w + x][c] - images[1][y * w + x][c]).powi(2);
                    }
                }
                for (i, idx) in [0, 3, 8].into_iter().enumerate() {
                    let expected = expect[i] / (w * h) as f64;
                    let actual = terms[idx];
                    let abs = (actual - expected).abs();
                    assert!(
                        abs < 2e-5,
                        "{w}x{h} scale {s} channel {c} feature {idx}: {actual} vs {expected}"
                    );
                    max_error[i] = max_error[i].max(abs);
                }
            }
            let downsample = |image: Vec<[f64; 3]>| {
                let mut out = Vec::new();
                for y in (0..h).step_by(2) {
                    for x in (0..w).step_by(2) {
                        let right = (x + 1).min(w - 1);
                        let down = (y + 1).min(h - 1);
                        out.push(core::array::from_fn(|c| {
                            (image[y * w + x][c]
                                + image[y * w + right][c]
                                + image[down * w + x][c]
                                + image[down * w + right][c])
                                / 4.0
                        }));
                    }
                }
                out
            };
            images = images.map(downsample);
            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }
    }
    println!("REV8_F64_WINDOW max-abs [L1,D21,MSE]={max_error:?}");
}

#[test]
fn fx2_and_500nit_symmetry_identity_and_thread_parity() {
    #[cfg(feature = "threads")]
    let pools = [1, 2, 4, 8, 16].map(|n| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap()
    });
    for (w, h, recipe) in [(1, 1, 4), (17, 13, 4), (64, 64, 5), (256, 191, 3)] {
        let (a, b) = pair(w, h, recipe);
        let a_view = view(&a, w, h);
        let b_view = view(&b, w, h);

        let base = compute(a_view, b_view, Variant::AlgS4, false).unwrap();
        let fx255_ab = extract_fx2(
            a_view,
            b_view,
            Variant::AlgS4,
            IntensityTarget::Nits255,
            false,
        )
        .unwrap();
        let fx255_ba = extract_fx2(
            b_view,
            a_view,
            Variant::AlgS4,
            IntensityTarget::Nits255,
            false,
        )
        .unwrap();
        for s in 0..4 {
            for c in 0..3 {
                for k in 0..super::COMMON {
                    assert_eq!(
                        base.features[s][c][k].to_bits(),
                        fx255_ab.features[s][c][k].to_bits(),
                        "Nits255 feature mismatch at s={s} c={c} k={k}"
                    );
                }
                assert_eq!(
                    fx255_ab.scale_fragility[s][c].to_bits(),
                    fx255_ba.scale_fragility[s][c].to_bits(),
                );
                let f = fx255_ab.scale_fragility[s][c];
                assert!(f.is_finite() && f > 0.0 && f <= 1.0, "frag={f}");
            }
        }
        assert_eq!(
            fx255_ab.alg_s4_fx2_inputs().map(f64::to_bits),
            fx255_ba.alg_s4_fx2_inputs().map(f64::to_bits),
        );

        let fx500_ab = extract_fx2(
            a_view,
            b_view,
            Variant::AlgS4,
            IntensityTarget::Nits500,
            false,
        )
        .unwrap();
        let fx500_ba = extract_fx2(
            b_view,
            a_view,
            Variant::AlgS4,
            IntensityTarget::Nits500,
            false,
        )
        .unwrap();
        assert_eq!(
            fx500_ab.alg_s4_fx2_inputs().map(f64::to_bits),
            fx500_ba.alg_s4_fx2_inputs().map(f64::to_bits),
        );
        let fx500_aa = extract_fx2(
            a_view,
            a_view,
            Variant::AlgS4,
            IntensityTarget::Nits500,
            false,
        )
        .unwrap();
        assert!(
            fx500_aa
                .alg_s4_fx2_inputs()
                .iter()
                .all(|v| v.to_bits() == 0)
        );
        for scale_frag in fx500_ab.scale_fragility.iter().take(4) {
            for &f in scale_frag {
                assert!(f.is_finite() && f > 0.0 && f <= 1.0, "frag500={f}");
            }
        }

        #[cfg(feature = "threads")]
        if (w, h) == (256, 191) {
            for pool in &pools {
                for target in [IntensityTarget::Nits255, IntensityTarget::Nits500] {
                    let expected =
                        extract_fx2(a_view, b_view, Variant::AlgS4, target, false).unwrap();
                    let par = pool
                        .install(|| extract_fx2(a_view, b_view, Variant::AlgS4, target, true))
                        .unwrap();
                    let par_swap = pool
                        .install(|| extract_fx2(b_view, a_view, Variant::AlgS4, target, true))
                        .unwrap();
                    assert_eq!(
                        expected.alg_s4_fx2_inputs().map(f64::to_bits),
                        par.alg_s4_fx2_inputs().map(f64::to_bits),
                    );
                    assert_eq!(
                        expected.alg_s4_fx2_inputs().map(f64::to_bits),
                        par_swap.alg_s4_fx2_inputs().map(f64::to_bits),
                    );
                }
            }
        }
    }
}
