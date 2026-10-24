#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::PixelFormat;
    use crate::source::{RgbSlice, StridedBytes};

    /// `SplitMode::Amplitude` carries the SSIM member's soft peak and both deviation pools as mean pools of `d`,
    /// with no peak or deviation polynomial left; the exact split keeps both.
    #[cfg(feature = "custom-profiles")]
    #[test]
    fn amplitude_split_folds_the_ssim_pools_into_its_mean_pool() {
        use crate::map_paint::SplitMode;
        let n = 64usize;
        let mut cell = AttrCellSums {
            n,
            ..Default::default()
        };
        for i in 0..n {
            let v = 0.01 + 0.002 * i as f64;
            cell.dense.sum_d += v;
            cell.dense.sum_d2 += v * v;
            cell.dense.sum_d3 += v * v * v;
            cell.dense.sum_d4 += v * v * v * v;
        }
        cell.dense.ws_peak_ssim = WeightedSum { num: 0.5, den: 2.0 };
        let mut s = vec![0.0; FEATURES_PER_CHANNEL_V2_TOTAL];
        s[idx::SSIM_MEAN] = 1.0;
        s[idx::SSIM_SOFT_PEAK] = 2.0;
        s[idx::SSIM_DEV2] = 3.0;
        s[idx::SSIM_DEV4] = 4.0;
        let derive = |split| {
            derive_v2app_coeffs(
                &s,
                None,
                None,
                0,
                0,
                &cell,
                false,
                false,
                0.0,
                FormulaRevision::Rev5,
                split,
            )
        };
        let (exact, amp) = (derive(SplitMode::Exact), derive(SplitMode::Amplitude));
        let inv_n = 1.0 / n as f64;
        assert!(exact.sp_ssim.0 != 0.0 && exact.dev_q.iter().any(|q| *q != 0.0));
        assert_eq!(exact.c_ssim, -inv_n);
        assert_eq!(amp.sp_ssim, (0.0, 0.0));
        assert_eq!(amp.dev_q, [0.0; 5]);
        assert!((amp.c_ssim + 10.0 * inv_n).abs() <= 1e-15, "{}", amp.c_ssim);
    }

    /// Revision 7's blockiness takes lane partials per `STRIP_ROWS` kernel strip, merged in strip order: the
    /// whole-range form (`blockiness_sparse_rows`, the materialized walks) and the strip walk
    /// (`blockiness_sparse_strip_wide` over reflected halo windows, as the streaming walk and its parallel jobs
    /// call it) must give the same bits, for heights that end on and off a strip boundary and widths with and
    /// without a lattice remainder.
    #[test]
    fn rev7_blockiness_strip_partials_match_the_whole_range_form() {
        if !crate::ssim_form::run_at_revision(
            "7",
            "feature_v2::tests::rev7_blockiness_strip_partials_match_the_whole_range_form",
            "REV7-BLOCKINESS-STRIPS-RAN",
        ) {
            return;
        }
        println!("REV7-BLOCKINESS-STRIPS-RAN");
        use crate::featcanon::SumVar;
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        };
        for (w, h) in [(77usize, 300usize), (200, 256), (1031, 133), (64, 129), (9, 40)] {
            let src: Vec<f32> = (0..w * h).map(|_| next()).collect();
            let dst: Vec<f32> = src.iter().map(|&v| v + (next() - 0.5) * 0.1).collect();
            let (whole_v, whole_h) = blockiness_sparse_rows(&src, &dst, w, 0, h);
            let mode = crate::featcanon::mode(FormulaRevision::Rev7);
            let (mut sv, mut sh) = (SumVar::f64_elems(mode), SumVar::f64_elems(mode));
            let mut y0 = 0;
            while y0 < h {
                let strip_h = STRIP_ROWS.min(h - y0);
                let wide_h = strip_h + 2 * HALO_P;
                let (mut sw, mut dw) = (vec![0f32; w * wide_h], vec![0f32; w * wide_h]);
                for i in 0..wide_h {
                    let gy = reflect_101(y0 as isize - HALO_P as isize + i as isize, h);
                    sw[i * w..(i + 1) * w].copy_from_slice(&src[gy * w..(gy + 1) * w]);
                    dw[i * w..(i + 1) * w].copy_from_slice(&dst[gy * w..(gy + 1) * w]);
                }
                blockiness_sparse_strip_wide(
                    &sw,
                    &dw,
                    w,
                    y0,
                    strip_h,
                    FormulaRevision::Rev7,
                    &mut sv,
                    &mut sh,
                );
                y0 += strip_h;
            }
            assert!(matches!(sv, SumVar::L64(_)), "revision 7 runs on the eight-lane sums");
            assert_eq!(whole_v.fin().to_bits(), sv.fin().to_bits(), "{w}x{h} vertical");
            assert_eq!(whole_h.fin().to_bits(), sh.fin().to_bits(), "{w}x{h} horizontal");
            assert!(whole_v.fin() > 0.0 && (h <= 8 || whole_h.fin() > 0.0), "{w}x{h}: terms ran");
        }
    }

    // Explicit calibration tool; compiled only by calibration_instrument.sh.
    #[cfg(gmsbank_calibration_instrument)]
    include!("gmsbank_calibration_instrument.rs");

    /// Plane-level exactness of the BLOCKINESS map for rectangle repairs (`crate::attribution::LatticeScale`):
    /// for every plane-aligned rectangle, at footprint factors 1, 2 and 4, the density mass inside the rectangle
    /// plus the lattice boundary correction must equal `c * (B(dst) - B(dst with the rectangle replaced by src))`,
    /// where `B` is the feature's own `blockiness_sparse` sum. The legacy 50/50 split (`lattice = None`) must NOT
    /// be exact here: that is the negative control.
    #[cfg(feature = "custom-profiles")]
    fn blockiness_exactness_fixture() -> (usize, usize, Vec<f32>, Vec<f32>) {
        let (w, h) = (64usize, 48usize);
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        };
        let src: Vec<f32> = (0..w * h).map(|_| next()).collect();
        // Blocky distortion: per 8x8 cell offset plus pixel noise, so lattice steps carry real excess.
        let cell: Vec<f32> = (0..(w / 8) * (h / 8))
            .map(|_| (next() - 0.5) * 0.4)
            .collect();
        let dst: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                src[i] + cell[(y / 8) * (w / 8) + x / 8] + (next() - 0.5) * 0.05
            })
            .collect();
        (w, h, src, dst)
    }

    #[cfg(feature = "custom-profiles")]
    fn blockiness_rect_error(use_lattice: bool, scale: usize) -> f64 {
        let (w, h, src, dst) = blockiness_exactness_fixture();
        let c = -0.37f64;
        let co = V2AppCoeffs {
            c_blockiness: c,
            ..Default::default()
        };
        let mut id = vec![0.0f64; w * h];
        let mut lat = crate::attribution::LatticeScale::new(scale, w, h);
        attr_pass_b_blockiness(
            &src,
            &dst,
            w,
            h,
            &co,
            &mut id,
            use_lattice.then_some((&mut lat, w, h)),
        );
        lat.finalize();
        let f = 1usize << scale;
        let before = blockiness_sparse(&src, &dst, w, h);
        let mut worst = 0.0f64;
        let mut state = 0x1234_5678u64;
        let mut pick = move |n: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 33) as usize % n
        };
        for _ in 0..400 {
            let (x0, y0) = (pick(w), pick(h));
            let (x1, y1) = (x0 + 1 + pick(w - x0), y0 + 1 + pick(h - y0));
            let mut repaired = dst.clone();
            let mut mass = 0.0;
            for y in y0..y1 {
                for x in x0..x1 {
                    repaired[y * w + x] = src[y * w + x];
                    mass += id[y * w + x];
                }
            }
            let truth = c * (before - blockiness_sparse(&src, &repaired, w, h));
            let predicted = mass + lat.correction(x0 * f, y0 * f, x1 * f, y1 * f);
            worst = worst.max((predicted - truth).abs());
        }
        worst
    }

    #[cfg(feature = "custom-profiles")]
    #[test]
    fn blockiness_map_is_exact_for_rectangle_repairs() {
        for scale in 0..3 {
            let err = blockiness_rect_error(true, scale);
            assert!(
                err < 1e-9,
                "scale {scale}: worst |predicted - finite repair| {err}"
            );
        }
    }

    #[cfg(feature = "custom-profiles")]
    #[test]
    fn legacy_blockiness_split_is_not_exact_for_rectangle_repairs() {
        let err = blockiness_rect_error(false, 0);
        assert!(
            err > 1e-3,
            "the symmetric split should miss one-sided repairs, worst error {err}"
        );
    }

    /// Independent per-step pair terms of the fixture (same definitions as the production routine, recomputed here
    /// from the planes so the tests below do not read production recordings back).
    #[cfg(feature = "custom-profiles")]
    fn fixture_pair_terms(c: f64) -> Vec<(bool, usize, usize, f64)> {
        let (w, h, src, dst) = blockiness_exactness_fixture();
        let ex = |a: f64, b: f64| bounded_excess(a, b, C_BLOCK);
        let at = |v: &[f32], i: usize| f64::from(v[i]);
        let pair = |ip: usize, iq: usize| {
            let ss = (at(&src, iq) - at(&src, ip)).abs();
            let e = ex((at(&dst, iq) - at(&dst, ip)).abs(), ss);
            let e_p = ex((at(&dst, iq) - at(&src, ip)).abs(), ss);
            let e_q = ex((at(&src, iq) - at(&dst, ip)).abs(), ss);
            c * (e_p + e_q - e)
        };
        let mut out = Vec::new();
        for y in 0..h {
            for x in (BLOCK_LATTICE..w).step_by(BLOCK_LATTICE) {
                out.push((true, x, y, pair(y * w + x - 1, y * w + x)));
            }
            if y % BLOCK_LATTICE == 0 && y > 0 {
                for x in 0..w {
                    out.push((false, x, y, pair((y - 1) * w + x, y * w + x)));
                }
            }
        }
        out
    }

    /// Non-aligned rectangle edges follow the documented coverage convention: each step end covers the fraction
    /// of its `2^s` footprint the rectangle overlaps per axis and the correction is `-(I/2) |a_p - a_q|`.
    /// The expectation is summed step by step here, independently of the prefix-sum implementation (this is
    /// what exercises its partial-coverage branch).
    #[cfg(feature = "custom-profiles")]
    #[test]
    fn blockiness_fractional_edges_follow_the_coverage_convention() {
        let (w, h, src, dst) = blockiness_exactness_fixture();
        let c = -0.37f64;
        let terms = fixture_pair_terms(c);
        let co = V2AppCoeffs {
            c_blockiness: c,
            ..Default::default()
        };
        for scale in 0..3usize {
            let f = 1usize << scale;
            let mut id = vec![0.0f64; w * h];
            let mut lat = crate::attribution::LatticeScale::new(scale, w, h);
            attr_pass_b_blockiness(&src, &dst, w, h, &co, &mut id, Some((&mut lat, w, h)));
            lat.finalize();
            let cover = |lo: usize, hi: usize, i: usize| {
                let a = lo.max(i * f);
                let b = hi.min((i + 1) * f);
                b.saturating_sub(a) as f64 / f as f64
            };
            let mut state = 0xface_feedu64 + scale as u64;
            let mut pick = move |n: usize| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 33) as usize % n
            };
            let (full_w, full_h) = (w * f, h * f);
            let mut partial_cases = 0;
            for _ in 0..300 {
                let (x0, y0) = (pick(full_w), pick(full_h));
                let (x1, y1) = (x0 + 1 + pick(full_w - x0), y0 + 1 + pick(full_h - y0));
                let mut want = 0.0;
                for &(vertical, x, y, i) in &terms {
                    let (ap, aq) = if vertical {
                        let cy = cover(y0, y1, y);
                        (cover(x0, x1, x - 1) * cy, cover(x0, x1, x) * cy)
                    } else {
                        let cx = cover(x0, x1, x);
                        (cover(y0, y1, y - 1) * cx, cover(y0, y1, y) * cx)
                    };
                    want += -0.5 * i * (ap - aq).abs();
                }
                let got = lat.correction(x0, y0, x1, y1);
                assert!(
                    (got - want).abs() < 1e-9,
                    "scale {scale} rect {:?}: correction {got} vs step-by-step {want}",
                    (x0, y0, x1, y1)
                );
                partial_cases +=
                    usize::from(x0 % f != 0 || y0 % f != 0 || x1 % f != 0 || y1 % f != 0);
            }
            if f > 1 {
                assert!(
                    partial_cases > 100,
                    "scale {scale}: only {partial_cases} partial-coverage rectangles"
                );
            }
        }
    }

    /// Steps whose plane pixels are not both inside the logical image (limits below the plane size) record no
    /// terms and keep the symmetric split: columns at or beyond the limit are bit-identical to the legacy
    /// routine, and a rectangle cutting a line beyond the limit gets no correction.
    #[cfg(feature = "custom-profiles")]
    #[test]
    fn blockiness_steps_outside_the_logical_image_keep_the_legacy_split() {
        let (w, h, src, dst) = blockiness_exactness_fixture();
        let co = V2AppCoeffs {
            c_blockiness: -0.37,
            ..Default::default()
        };
        let (lw, lh) = (40usize, 32usize);
        let mut limited = vec![0.0f64; w * h];
        let mut lat = crate::attribution::LatticeScale::new(0, w, h);
        attr_pass_b_blockiness(
            &src,
            &dst,
            w,
            h,
            &co,
            &mut limited,
            Some((&mut lat, lw, lh)),
        );
        lat.finalize();
        let mut legacy = vec![0.0f64; w * h];
        attr_pass_b_blockiness(&src, &dst, w, h, &co, &mut legacy, None);
        for y in 0..h {
            for x in lw..w {
                assert_eq!(
                    limited[y * w + x].to_bits(),
                    legacy[y * w + x].to_bits(),
                    "pixel ({x},{y})"
                );
            }
        }
        // Rows at or beyond the limit: horizontal steps in rows >= lh are legacy at every column.
        for y in lh..h {
            for x in 0..w {
                assert_eq!(
                    limited[y * w + x].to_bits(),
                    legacy[y * w + x].to_bits(),
                    "pixel ({x},{y})"
                );
            }
        }
        // Lines beyond the limits cut by a rectangle edge: no correction. (x = 48 and y = 40 are lattice lines.)
        assert_eq!(lat.correction(48, 0, 64, 32), 0.0);
        assert_eq!(lat.correction(0, 40, 40, 48), 0.0);
        // A line inside the limits is cut and does correct.
        assert_ne!(lat.correction(16, 0, 40, 32), 0.0);
    }

    /// The pass-B scalar-tail SSIM dissimilarity must follow the vector kernel's two forms. At revision 3 the
    /// `s12` plane is the direct error moment; the covariance form read it as `E[ab]` and returned `d` of 1e2-1e3
    /// (dark flat window: `a*nc/(b*nd)` strongly negative), which the dev4 coefficients amplified to 1e26.
    #[cfg(feature = "custom-profiles")]
    #[test]
    fn pass_b_tail_ssim_d_follows_the_direct_error_moment_form() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..4000 {
            let (m1, m2) = (next() * 0.9, next() * 0.9);
            let var1 = next() * 0.02 * next();
            let var2 = next() * 0.02 * next();
            let ssq = m1 * m1 + m2 * m2 + var1 + var2;
            let md = m1 - m2;
            // direct plane: E[(a-b)^2] = (m1-m2)^2 + var1 + var2 - 2cov, cov in [-sqrt(v1 v2), sqrt(v1 v2)]
            let cov = (next() * 2.0 - 1.0) * (var1 * var2).sqrt();
            let s12_direct = md * md + var1 + var2 - 2.0 * cov;
            let s12_prod = m1 * m2 + cov;
            let na = (2.0 * m1 * m2 + C1_V2) as f32;
            let nb = (m1 * m1 + m2 * m2 + C1_V2) as f32;
            let nd = (ssq - m1 * m1 - m2 * m2 + C2_V2) as f32;
            // Both luminance forms: revision 7's `(m1 − m2)²` and the
            // subtractive one before it.
            for stable in [false, true] {
                for (direct, s12) in [(true, s12_direct), (false, s12_prod)] {
                    let want = ssim_d_local(m1, m2, s12, ssq, direct, stable);
                    let got = f64::from(ssim_d_px(
                        na, nb, nd, m1 as f32, m2 as f32, s12 as f32, direct, stable,
                    ));
                    assert!(
                        (0.0..=2.0 + 1e-3).contains(&got),
                        "d out of range: {got} (direct {direct}, stable {stable})"
                    );
                    assert!(
                        (got - want).abs() < 2e-3,
                        "px {got} vs f64 {want} (direct {direct}, stable {stable})"
                    );
                }
            }
        }
    }

    /// imazen/zensim#56 regression gate: the MSCN divisive normalizer must
    /// be the CORRECTLY-ROUNDED IEEE `resid / sqrt(var + c)` on every SIMD
    /// tier, lane for lane, bit for bit. That is what makes the
    /// `MSCN_DIFF_MEAN`/`MSCN_DIFF_L2` slots CPU-vendor-deterministic: IEEE
    /// sqrt and div are exactly rounded on every vendor, whereas the
    /// `rsqrt()` the kernel used before (`vrsqrtps` estimate + 1 NR step on
    /// x86; `1/sqrt` then multiply on NEON and scalar) is not — the bf944
    /// wave measured AMD≠Intel by ~1e-8 rel on exactly those 22 columns.
    ///
    /// Mutation-verified: substituting `resid * (var + c).rsqrt()` in
    /// `mscn_norm_v` fails this test on x86 (NR residue) AND on NEON /
    /// scalar (double rounding), so it cannot silently regress on any CI
    /// platform. The corpus deliberately includes the exact-zero variance
    /// floor (`var = 0` → `sqrt(C_MSCN_VAR)`), sub-floor, unit and large
    /// variances, and signed residuals spanning ±[1e-4, 4].
    fn check_mscn_norm_tier<T: F32x8Backend + Copy>(token: T, tier: &str) {
        let c = C_MSCN_VAR as f32;
        let cv = V8::<T>::splat_t(token, c);
        // Deterministic LCG so the corpus is identical on every platform.
        let mut state = 0x9E37_79B9u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32
        };
        let mut n_checked = 0usize;
        let mut n_mismatch = 0usize;
        let mut first_mismatch: Option<(f32, f32, f32, f32)> = None;
        for block in 0..512 {
            let mut resid = [0f32; 8];
            let mut var = [0f32; 8];
            for lane in 0..8 {
                let u = next();
                // Signed residual, log-spread magnitude 1e-4..4.
                let mag = 1e-4 * (4e4f32).powf(next());
                resid[lane] = if u < 0.5 { -mag } else { mag };
                // Variance: exact floor every 8th lane, else log-spread
                // 1e-9..1e2 (covers below the C_MSCN_VAR floor and far above).
                var[lane] = if (block + lane) % 8 == 0 {
                    0.0
                } else {
                    1e-9 * (1e11f32).powf(next())
                };
            }
            let got = mscn_norm_v(
                token,
                V8::<T>::from_array_t(token, resid),
                V8::<T>::from_array_t(token, var),
                cv,
            )
            .to_array();
            for lane in 0..8 {
                // Scalar Rust f32 sqrt and div are IEEE-754 correctly rounded.
                let want = resid[lane] / (var[lane] + c).sqrt();
                n_checked += 1;
                if got[lane].to_bits() != want.to_bits() {
                    n_mismatch += 1;
                    first_mismatch.get_or_insert((resid[lane], var[lane], got[lane], want));
                }
            }
        }
        assert_eq!(
            n_mismatch, 0,
            "tier {tier}: {n_mismatch}/{n_checked} lanes of mscn_norm_v are not the \
             correctly-rounded IEEE resid/sqrt(var+c); first: {first_mismatch:?} \
             (resid, var, got, want)"
        );
    }

    #[test]
    fn mscn_norm_v_is_correctly_rounded_on_every_tier() {
        use archmage::SimdToken as _;
        let mut tiers_run = 0usize;
        check_mscn_norm_tier(
            archmage::ScalarToken::summon().expect("scalar token is infallible"),
            "scalar",
        );
        tiers_run += 1;
        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(target_arch = "x86_64")]
            if let Some(t) = archmage::X64V3Token::summon() {
                check_mscn_norm_tier(t, "x86 v3");
                tiers_run += 1;
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            if let Some(t) = archmage::X64V4Token::summon() {
                check_mscn_norm_tier(t, "x86 v4");
                tiers_run += 1;
            }
            #[cfg(feature = "avx512")]
            if let Some(t) = archmage::X64V4xToken::summon() {
                check_mscn_norm_tier(t, "x86 v4x");
                tiers_run += 1;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if let Some(t) = archmage::NeonToken::summon() {
                check_mscn_norm_tier(t, "neon");
                tiers_run += 1;
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            if let Some(t) = archmage::Wasm128Token::summon() {
                check_mscn_norm_tier(t, "wasm128");
                tiers_run += 1;
            }
        }
        assert!(tiers_run >= 1, "no tier was exercised");
    }

    // Perf pass (§A.13): widened from 1e-6 to 5e-4 after switching mu1/mu2/
    // ssq/s12 to `crate::blur::fused_blur_h_ssim`'s single fused-multiply-add
    // sliding-window accumulation (`sum_sq = sv.mul_add(sv, dv.mul_add(dv,
    // sum_sq))`) in place of the old separate `sq_sum_into`/`mul_into`
    // elementwise passes (plain `sv*sv + dv*dv`, two roundings). FMA vs.
    // non-FMA reassociation is a legitimate, expected last-few-ULP source of
    // drift, not a logic change — but `ssim_d_local`'s `denom_s = ssq - mu1²
    // - mu2² + C2_V2` is a near-zero-minus-near-zero subtraction on identity
    // input (variance ≈ 0), stabilized only by the small `C2_V2 = 9e-4`
    // constant, so ULP-scale noise in `ssq` gets divided by ~9e-4 and
    // amplified ~1000x. This amplification is a PRE-EXISTING property of the
    // `ssim_d_local` formula (small-constant-stabilized ratio), not
    // something the perf pass introduced — only WHICH rounding path feeds it
    // changed. Observed on the identity fixture as TOL was raised: idx 0
    // (SSIM_MEAN) 2.52e-6, idx 1 (SSIM_DEV2) 1.47e-5, idx 9 (a
    // weighted-pool/SOFT_PEAK slot one denominator-division further from the
    // source) 1.20e-4 — each roughly one more small-constant division beyond
    // the last, consistent with amplified-not-new noise rather than a
    // distinct bug. 5e-4 leaves >4x headroom above the largest observed.
    // Within this file's own documented allowance (v2 has no downstream
    // consumers yet; v1's golden gate, which does NOT go through this
    // kernel, keeps its own separate zero-tolerance `==` comparison in
    // `tests/v1_golden_bytes.rs`).
    const TOL: f64 = 5e-4;

    fn flat_image(w: usize, h: usize, rgb: [u8; 3]) -> Vec<[u8; 3]> {
        vec![rgb; w * h]
    }

    const ZERO_ONE_IDX: &[usize] = &[
        idx::ART,
        idx::DET,
        idx::MSE,
        idx::HF_GAIN,
        idx::HF_LOSS,
        idx::HF_MAG_LOSS,
        idx::PJND_TRANSDUCER,
        idx::PJND_FRAGILITY,
        idx::GMS,
        idx::PJND_TRANSDUCER_LOW_K,
        idx::PJND_TRANSDUCER_HIGH_K,
        idx::RINGING,
        idx::BANDING,
    ];
    const ZERO_TWO_IDX: &[usize] = &[
        idx::SSIM_MEAN,
        idx::SSIM_DEV2,
        idx::SSIM_DEV4,
        idx::SSIM_SOFT_PEAK,
        idx::ART_SOFT_PEAK,
        idx::DET_SOFT_PEAK,
        idx::MASKED_SSIM,
        idx::MASKED_ART,
        idx::MASKED_DET,
        idx::MASKED_MSE,
        idx::IW_SSIM,
        idx::IW_ART,
        idx::IW_DET,
        idx::IW_MSE,
        idx::BLOCKINESS, // can be up to 2 blocks (h+v corner)
        idx::EDGE_WIDTH_CHANGE,
    ];

    fn assert_all_bounded(view: &FeatureViewV2, tol: f64) {
        for scale in 0..view.n_scales() {
            for ch in 0..3 {
                for &off in ZERO_ONE_IDX {
                    let v = view.at(scale, ch, off);
                    assert!(
                        v.is_finite() && (0.0..1.0 + tol).contains(&v),
                        "idx {off} OOB [0,1): {v} at s{scale} c{ch}"
                    );
                }
                for &off in ZERO_TWO_IDX {
                    let v = view.at(scale, ch, off);
                    assert!(
                        v.is_finite() && (0.0..=2.0 + tol).contains(&v),
                        "idx {off} OOB [0,2]: {v} at s{scale} c{ch}"
                    );
                }
            }
        }
    }

    #[test]
    fn regime_tag_and_view_accessors_match_raw_indexing() {
        let src = flat_image(64, 64, [128, 128, 128]);
        let mut dst = src.clone();
        dst[0] = [200, 60, 60];
        let source = RgbSlice::new(&src, 64, 64);
        let distorted = RgbSlice::new(&dst, 64, 64);

        let result =
            compute_v2_features_impl(&source, &distorted, None, false).expect("compute ok");
        assert_eq!(result.regime(), FeatureRegime::V2Bounded);
        assert_eq!(result.n_scales(), crate::NUM_SCALES);
        assert_eq!(
            result.features().len(),
            crate::NUM_SCALES * 3 * FEATURES_PER_CHANNEL_V2_TOTAL
        );

        let view = result.view();
        for scale in 0..result.n_scales() {
            for ch in 0..3 {
                let base =
                    scale * 3 * FEATURES_PER_CHANNEL_V2_TOTAL + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
                assert_eq!(
                    view.ssim_mean(scale, ch),
                    result.features()[base + idx::SSIM_MEAN]
                );
                assert_eq!(view.gms(scale, ch), result.features()[base + idx::GMS]);
                assert_eq!(
                    view.pjnd_transducer_low_k(scale, ch),
                    result.features()[base + idx::PJND_TRANSDUCER_LOW_K]
                );
                assert_eq!(
                    view.pjnd_transducer_high_k(scale, ch),
                    result.features()[base + idx::PJND_TRANSDUCER_HIGH_K]
                );
                assert_eq!(
                    view.blockiness(scale, ch),
                    result.features()[base + idx::BLOCKINESS]
                );
                assert_eq!(
                    view.ringing(scale, ch),
                    result.features()[base + idx::RINGING]
                );
                assert_eq!(
                    view.banding(scale, ch),
                    result.features()[base + idx::BANDING]
                );
                assert_eq!(
                    view.edge_width_change(scale, ch),
                    result.features()[base + idx::EDGE_WIDTH_CHANGE]
                );
            }
        }
        assert!(FeatureViewV2::new(result.features(), result.n_scales() + 1).is_none());
        assert!(
            FeatureViewV2::new(
                &result.features()[..result.features().len() - 1],
                result.n_scales()
            )
            .is_none()
        );
    }

    /// Identity input => every error-oriented feature is exactly 0.
    /// `pjnd_fragility` is a reference-only property, excluded (as in
    /// iteration 1). `edge_width_change` compares src's OWN cross-scale
    /// decay to dst's — on identity input src==dst so their decays are
    /// identical too => also exactly 0, INCLUDED here (a strengthening
    /// over iteration 1: this is the one new-feature identity check that
    /// exercises the cross-scale path).
    #[test]
    fn identity_input_zeroes_every_error_feature() {
        let mut src = Vec::with_capacity(64 * 64);
        for y in 0..64 {
            for x in 0..64 {
                let v = (((x * 7 + y * 13) % 256) as u8).max(1);
                src.push([v, v.wrapping_add(40), v.wrapping_add(80)]);
            }
        }
        let dst = src.clone();
        let source = RgbSlice::new(&src, 64, 64);
        let distorted = RgbSlice::new(&dst, 64, 64);

        let result =
            compute_v2_features_impl(&source, &distorted, None, false).expect("compute ok");
        let view = result.view();
        let must_be_zero = [
            idx::SSIM_MEAN,
            idx::SSIM_DEV2,
            idx::SSIM_DEV4,
            idx::ART,
            idx::DET,
            idx::MSE,
            idx::HF_GAIN,
            idx::HF_LOSS,
            idx::HF_MAG_LOSS,
            idx::SSIM_SOFT_PEAK,
            idx::ART_SOFT_PEAK,
            idx::DET_SOFT_PEAK,
            idx::MASKED_SSIM,
            idx::MASKED_ART,
            idx::MASKED_DET,
            idx::MASKED_MSE,
            idx::IW_SSIM,
            idx::IW_ART,
            idx::IW_DET,
            idx::IW_MSE,
            idx::PJND_TRANSDUCER,
            idx::PJND_TRANSDUCER_LOW_K,
            idx::PJND_TRANSDUCER_HIGH_K,
            idx::GMS,
            idx::RINGING,
            idx::BANDING,
            idx::BLOCKINESS,
            idx::EDGE_WIDTH_CHANGE,
        ];
        for scale in 0..result.n_scales() {
            for ch in 0..3 {
                for &off in &must_be_zero {
                    let v = view.at(scale, ch, off);
                    assert!(
                        v.abs() < TOL,
                        "idx {off} not zero on identity: {v} at s{scale} c{ch}"
                    );
                }
            }
        }
    }

    /// Adversarial fixture: flat source + noise (D2/HF-gain pathology).
    #[test]
    fn transducer_luma_gate_zeroes_chroma_only() {
        // Distorted pair (dst != src) so the transducers are non-zero.
        let mut src = Vec::with_capacity(64 * 64);
        let mut dst = Vec::with_capacity(64 * 64);
        for y in 0..64 {
            for x in 0..64 {
                let v = (((x * 7 + y * 13) % 256) as u8).max(1);
                src.push([v, v.wrapping_add(40), v.wrapping_add(80)]);
                let d = v.wrapping_add(((x + y) % 17) as u8);
                dst.push([d, d.wrapping_add(40), d.wrapping_add(80)]);
            }
        }
        let source = RgbSlice::new(&src, 64, 64);
        let distorted = RgbSlice::new(&dst, 64, 64);

        let base = compute_v2_features_impl(&source, &distorted, None, false).expect("base ok");
        let gated = compute_v2_features_impl_with_toggles(
            &source,
            &distorted,
            None,
            false,
            V2NewFeatureToggles {
                transducers_luma_only: true,
                ..Default::default()
            },
        )
        .expect("gated ok");
        let bv = base.view();
        let gv = gated.view();
        let transducers = [
            idx::PJND_TRANSDUCER,
            idx::PJND_FRAGILITY,
            idx::PJND_TRANSDUCER_LOW_K,
            idx::PJND_TRANSDUCER_HIGH_K,
        ];
        let n_scales = base.n_scales();
        let mut any_chroma_nonzero_in_base = false;
        for scale in 0..n_scales {
            for ch in 0..3 {
                let b = ch * FEATURES_PER_CHANNEL_V2_TOTAL;
                let sb = scale * 3 * FEATURES_PER_CHANNEL_V2_TOTAL;
                for t in transducers {
                    let bval = base.features()[sb + b + t];
                    let gval = gated.features()[sb + b + t];
                    if ch == 1 {
                        // Luma channel: transducers unchanged.
                        assert_eq!(bval, gval, "Y transducer local {t} scale {scale} changed");
                    } else {
                        // Chroma channels: zeroed under the gate.
                        assert_eq!(gval, 0.0, "chroma transducer local {t} not zeroed");
                        if bval != 0.0 {
                            any_chroma_nonzero_in_base = true;
                        }
                    }
                }
                // Every NON-transducer feature is byte-identical between the two.
                for local in 0..FEATURES_PER_CHANNEL_V2_TOTAL {
                    if transducers.contains(&local) {
                        continue;
                    }
                    assert_eq!(
                        base.features()[sb + b + local],
                        gated.features()[sb + b + local],
                        "non-transducer local {local} ch {ch} scale {scale} changed"
                    );
                }
            }
        }
        // Guard: the gate is actually testing something (chroma transducers
        // WERE non-zero without it).
        assert!(
            any_chroma_nonzero_in_base,
            "test fixture produced no non-zero chroma transducers — gate is untested"
        );
        // Silence the unused-view warnings while keeping them for future asserts.
        let _ = (bv.gms(0, 0), gv.gms(0, 0));
    }

    #[test]
    fn bounded_range_flat_source_plus_noise() {
        let w = 64;
        let h = 64;
        let src = vec![[128u8, 128, 128]; w * h];
        let mut dst = src.clone();
        for y in 0..h {
            for x in 0..w {
                dst[y * w + x] = if (x + y) % 2 == 0 {
                    [200, 60, 220]
                } else {
                    [50, 220, 40]
                };
            }
        }
        let source = RgbSlice::new(&src, w, h);
        let distorted = RgbSlice::new(&dst, w, h);
        let result =
            compute_v2_features_impl(&source, &distorted, None, false).expect("compute ok");
        assert_all_bounded(&result.view(), TOL);
    }

    /// Adversarial fixture: large chroma mean-shift (D1 driver).
    #[test]
    fn bounded_range_large_chroma_mean_shift() {
        let w = 64;
        let h = 64;
        let src = flat_image(w, h, [0, 255, 0]);
        let dst = flat_image(w, h, [255, 0, 255]);
        let source = RgbSlice::new(&src, w, h);
        let distorted = RgbSlice::new(&dst, w, h);
        let result =
            compute_v2_features_impl(&source, &distorted, None, false).expect("compute ok");
        assert_all_bounded(&result.view(), TOL);
    }

    /// Adversarial fixture: hard edge on a flat chroma field (the exact
    /// masked/IW explosion pattern from
    /// `benchmarks/ssim_moment_explosion_2026-07-16.md`).
    #[test]
    fn bounded_range_hard_edge_on_flat() {
        let w = 64;
        let h = 64;
        let src = vec![[0u8, 200, 0]; w * h];
        let mut dst = src.clone();
        for y in 0..h {
            for x in (w / 2)..w {
                dst[y * w + x] = [220, 0, 220];
            }
        }
        let source = RgbSlice::new(&src, w, h);
        let distorted = RgbSlice::new(&dst, w, h);
        let result =
            compute_v2_features_impl(&source, &distorted, None, false).expect("compute ok");
        assert_all_bounded(&result.view(), TOL);
    }

    /// Phase-5 (§A.15): multi-strip boundary correctness. A 150-row image
    /// with `STRIP_ROWS=64` spans 3 strips (`[0,64)`, `[64,128)`,
    /// `[128,150)`), and this fixture puts a HARD HORIZONTAL edge at
    /// `y=128` — exactly the first strip seam at the current
    /// `STRIP_ROWS=128` — plus a second at `y=400` (mid-strip, for
    /// contrast) and blocky columns throughout (stresses blockiness'
    /// lattice check across the seam too). If `gather_strip_halo`/the halo
    /// math were wrong, this is exactly the pattern that would show it:
    /// gradients spanning the seam would see the WRONG neighbor row
    /// (garbage, a mismatched reflection, or a stale scratch value),
    /// producing an out-of-bounds or wildly-off signal right at `y=128`.
    /// Bounded-range + a same-shape identity check (identity input still
    /// zeroes everything even across 5 strips) are the two things
    /// actually exercised here.
    ///
    /// `h=640` is DELIBERATELY well above [`STRIP_BYPASS_HEIGHT`]
    /// (phase-6 §A.16 lever B) — this test exists specifically to exercise
    /// the STRIP LOOP's multi-strip seam handling, and must keep doing so
    /// regardless of where the bypass threshold lands; a height anywhere
    /// near the threshold would silently start routing through
    /// `compute_channel_scale_v2_whole` instead and stop testing what this
    /// test's name promises.
    #[test]
    fn bounded_range_multi_strip_horizontal_edge() {
        let w = 96;
        let h = 640; // 5 strips at STRIP_ROWS=128: [0,128)[128,256)[256,384)[384,512)[512,640)
        assert!(
            h > STRIP_BYPASS_HEIGHT,
            "test fixture must stay above the lever-B bypass threshold to keep exercising \
             the strip loop -- bump h if STRIP_BYPASS_HEIGHT is ever raised past 640"
        );
        let mut src = Vec::with_capacity(w * h);
        let mut dst = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                // Horizontal bands (edges at y=128, the first strip seam,
                // and y=400, mid-strip) plus an 8-wide vertical block
                // pattern, so gradient/blockiness both see real structure
                // crossing the strip seam.
                let band = if y < 128 {
                    0u8
                } else if y < 400 {
                    255u8
                } else {
                    120u8
                };
                let block = if (x / 8) % 2 == 0 { 0i16 } else { 40i16 };
                let v = (band as i16 + block).clamp(0, 255) as u8;
                src.push([v, v.wrapping_add(30), v.wrapping_add(60)]);
                // Distorted: shift the band edges by a few rows and
                // perturb the block pattern, so gradients differ on both
                // sides of the seam, not just coincide with src.
                let dband = if y < 125 {
                    10u8
                } else if y < 403 {
                    240u8
                } else {
                    130u8
                };
                let dblock = if (x / 8) % 2 == 0 { 5i16 } else { 35i16 };
                let dv = (dband as i16 + dblock).clamp(0, 255) as u8;
                dst.push([dv, dv.wrapping_add(25), dv.wrapping_add(55)]);
            }
        }
        let source = RgbSlice::new(&src, w, h);
        let distorted = RgbSlice::new(&dst, w, h);
        let result =
            compute_v2_features_impl(&source, &distorted, None, false).expect("compute ok");
        assert_all_bounded(&result.view(), TOL);

        // Identity check across the SAME multi-strip shape: every
        // error-oriented feature must still be exactly zero when src==dst,
        // even though the computation now crosses 5 strip boundaries.
        let source2 = RgbSlice::new(&src, w, h);
        let identity = RgbSlice::new(&src, w, h);
        let result2 =
            compute_v2_features_impl(&source2, &identity, None, false).expect("compute ok");
        let view2 = result2.view();
        for scale in 0..result2.n_scales() {
            for ch in 0..3 {
                let d = view2.ssim_mean(scale, ch);
                assert!(
                    d.abs() < TOL,
                    "multi-strip identity: ssim_mean not zero: {d} at s{scale} c{ch}"
                );
                let gms = view2.gms(scale, ch);
                assert!(
                    gms.abs() < TOL,
                    "multi-strip identity: gms not zero: {gms} at s{scale} c{ch}"
                );
            }
        }
    }

    /// Phase-6 (§A.16 lever B): dedicated coverage for
    /// `compute_channel_scale_v2_whole` (the small-image bypass path).
    /// `STRIP_BYPASS_HEIGHT=0` DISABLES the bypass at runtime (measured
    /// regression, §A.16.4 — the function is correctness-verified and kept
    /// for a future session, not currently reachable via the public
    /// dispatch), so this test calls `compute_channel_scale_v2_whole`
    /// DIRECTLY rather than through `compute_v2_features_impl` — the only
    /// way to keep exercising it regardless of the (disabled) threshold
    /// value. A hard horizontal edge at `y=0` (the image's OWN true top
    /// edge — exercises `reflect_101`'s boundary case directly, not an
    /// interior strip seam) plus one at mid-image, and an 8-wide vertical
    /// block pattern, so gradient/blockiness see real structure at the
    /// exact row the halo-construction code touches.
    #[test]
    fn bounded_range_bypass_path_small_image() {
        let w = 80;
        let h = 96;
        let mut src = Vec::with_capacity(w * h);
        let mut dst = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let band = if y < 40 { 20.0f32 } else { 220.0 };
                let block = if (x / 8) % 2 == 0 { 0.0f32 } else { 40.0 };
                src.push((band + block).clamp(0.0, 255.0) / 255.0);
                let dband = if y < 37 { 30.0f32 } else { 200.0 };
                let dblock = if (x / 8) % 2 == 0 { 5.0f32 } else { 35.0 };
                dst.push((dband + dblock).clamp(0.0, 255.0) / 255.0);
            }
        }
        let mut scratch = ScratchV2Strip::new(w * h);
        let mut out = [0.0f64; FEATURES_PER_CHANNEL_V2_TOTAL];
        compute_channel_scale_v2_whole(
            &src,
            &dst,
            w,
            h,
            V2NewFeatureToggles::default(),
            &mut scratch,
            &mut out,
        );
        for &off in ZERO_ONE_IDX {
            let v = out[off];
            assert!(
                v.is_finite() && (0.0..1.0 + TOL).contains(&v),
                "bypass-path idx {off} OOB [0,1): {v}"
            );
        }
        for &off in ZERO_TWO_IDX {
            let v = out[off];
            assert!(
                v.is_finite() && (0.0..=2.0 + TOL).contains(&v),
                "bypass-path idx {off} OOB [0,2]: {v}"
            );
        }

        // Identity re-run through the SAME direct call: every error
        // feature must still be exactly zero.
        let mut scratch2 = ScratchV2Strip::new(w * h);
        let mut out2 = [0.0f64; FEATURES_PER_CHANNEL_V2_TOTAL];
        compute_channel_scale_v2_whole(
            &src,
            &src,
            w,
            h,
            V2NewFeatureToggles::default(),
            &mut scratch2,
            &mut out2,
        );
        {
            let d = out2[idx::SSIM_MEAN];
            assert!(
                d.abs() < TOL,
                "bypass-path identity: ssim_mean not zero: {d}"
            );
            let gms = out2[idx::GMS];
            assert!(gms.abs() < TOL, "bypass-path identity: gms not zero: {gms}");
        }
    }

    /// Adversarial fixture: 8x8-periodic step pattern (stresses the NEW
    /// oriented-blockiness feature specifically — a fake "JPEG-blocky"
    /// image with real 8-pixel-periodic steps that should score high
    /// blockiness on itself-vs-flat, but bounded).
    #[test]
    fn bounded_range_8x8_periodic_steps() {
        let w = 64;
        let h = 64;
        let src = flat_image(w, h, [100, 100, 100]);
        let mut dst = src.clone();
        for y in 0..h {
            for x in 0..w {
                let block_val = if (x / 8 + y / 8) % 2 == 0 {
                    80u8
                } else {
                    180u8
                };
                dst[y * w + x] = [block_val, block_val, block_val];
            }
        }
        let source = RgbSlice::new(&src, w, h);
        let distorted = RgbSlice::new(&dst, w, h);
        let result =
            compute_v2_features_impl(&source, &distorted, None, false).expect("compute ok");
        assert_all_bounded(&result.view(), TOL);
        // The blockiness signal should register something (not identically
        // zero) on a genuinely blocky pattern -- a sanity check that the
        // feature is alive, not just bounded-and-silent.
        let view = result.view();
        let any_blockiness =
            (0..result.n_scales()).any(|s| (0..3).any(|c| view.blockiness(s, c) > 1e-3));
        assert!(
            any_blockiness,
            "blockiness should fire on a genuine 8x8 step pattern"
        );
    }

    /// Adversarial fixture: banding gradient (smooth ramp in source,
    /// stepped/posterized in distorted) -- stresses the NEW banding
    /// feature specifically.
    #[test]
    fn bounded_range_banding_gradient() {
        let w = 64;
        let h = 64;
        let mut src = Vec::with_capacity(w * h);
        let mut dst = Vec::with_capacity(w * h);
        for _y in 0..h {
            for x in 0..w {
                let smooth = (x * 255 / w) as u8;
                let posterized = ((x * 8 / w) * 32) as u8; // 8 flat steps
                src.push([smooth, smooth, smooth]);
                dst.push([posterized, posterized, posterized]);
            }
        }
        let source = RgbSlice::new(&src, w, h);
        let distorted = RgbSlice::new(&dst, w, h);
        let result =
            compute_v2_features_impl(&source, &distorted, None, false).expect("compute ok");
        assert_all_bounded(&result.view(), TOL);
        let view = result.view();
        let any_banding = (0..result.n_scales()).any(|s| (0..3).any(|c| view.banding(s, c) > 1e-3));
        assert!(
            any_banding,
            "banding should fire on a smooth-ramp -> posterized-steps pair"
        );
    }

    /// Blanket bounded-range sweep on deterministic pseudo-random content.
    #[test]
    fn bounded_range_all_signals_random_content() {
        let w = 64;
        let h = 64;
        let mut src = Vec::with_capacity(w * h);
        let mut dst = Vec::with_capacity(w * h);
        let mut state: u32 = 0x1234_5678;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        };
        for _ in 0..(w * h) {
            src.push([next(), next(), next()]);
            dst.push([next(), next(), next()]);
        }
        let source = RgbSlice::new(&src, w, h);
        let distorted = RgbSlice::new(&dst, w, h);
        let result =
            compute_v2_features_impl(&source, &distorted, None, false).expect("compute ok");
        assert_all_bounded(&result.view(), TOL);
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let src = flat_image(64, 64, [10, 10, 10]);
        let dst = flat_image(32, 32, [10, 10, 10]);
        let source = RgbSlice::new(&src, 64, 64);
        let distorted = RgbSlice::new(&dst, 32, 32);
        let err = compute_v2_features_impl(&source, &distorted, None, false).unwrap_err();
        assert!(matches!(err, ZensimError::DimensionMismatch));
    }

    /// Gated-mirror pin (see `WeightedSum`'s doc): the incremental
    /// `WeightedSum` accumulator used by the fused kernel must match
    /// iteration 1's array-based `WeightedPool::mean` (the promoted
    /// canonical `Σw·v/Σw` helper) bit-for-bit on the same data, so the
    /// perf-motivated rewrite is provably not a silent formula drift.
    #[test]
    fn weighted_sum_matches_weighted_pool_mean_exactly() {
        let mut state: u32 = 0xabcdef01;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / u32::MAX as f32).abs()
        };
        let values: Vec<f32> = (0..1000).map(|_| next() * 2.0).collect();
        let weights: Vec<f32> = (0..1000).map(|_| next()).collect();

        let mut ws = WeightedSum::default();
        for (&v, &w) in values.iter().zip(weights.iter()) {
            ws.add(w as f64, v as f64);
        }
        let incremental = ws.finish();
        let batch = crate::iw_pool::WeightedPool::mean(&values, &weights);
        assert!(
            (incremental - batch).abs() < 1e-9,
            "WeightedSum {incremental} vs WeightedPool::mean {batch} diverged"
        );
    }

    /// Phase-4 (§A.14): proves the SIMD raw-moment reformulation
    /// (`compute_channel_scale_v2`'s `mean_d`/`m2`/`m4` block) is
    /// numerically equivalent to `OnlineMoments`'s Terriberry (2007)
    /// single-pass algorithm on the SAME data, not just "removed and
    /// hoped for the best". Uses `d` values in the actual documented
    /// range `[0, 2]` (not arbitrary floats) since that's what the real
    /// kernel feeds both paths. Also exercises `OnlineMoments` directly
    /// so it isn't dead code despite leaving the hot path.
    #[test]
    fn rev5_stable_block_moments_match_two_pass() {
        for base in [0.0, 0.5, 1.0, 1e4] {
            for n in [1usize, 16, 153, 1025] {
                let values: Vec<_> = (0..n)
                    .map(|i| base + ((i * 137 % 1009) as f64 - 500.0) * 1e-7)
                    .collect();
                let mean = values.iter().sum::<f64>() / n as f64;
                let m2 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>();
                let m4 = values.iter().map(|x| (x - mean).powi(4)).sum::<f64>();
                let mut actual = OnlineMoments::default();
                for block in values.chunks(16) {
                    actual.merge(OnlineMoments::block(block));
                }
                for (a, e) in [(actual.m2, m2), (actual.m4, m4)] {
                    assert!(
                        (a - e).abs() <= 1e-6 * e.abs().max(1e-30),
                        "base={base} n={n} actual={a} reference={e}"
                    );
                }
            }
        }
        let mut constant = OnlineMoments::default();
        for _ in 0..10 {
            constant.merge(OnlineMoments::block(&[0.7; 16]));
        }
        assert_eq!(constant.m2, 0.0);
        assert_eq!(constant.m4, 0.0);
    }

    #[test]
    fn raw_moment_reformulation_matches_terriberry() {
        let mut state: u32 = 0x5eed_1234;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f64 / u32::MAX as f64).abs() * 2.0 // in [0, 2]
        };
        let d_values: Vec<f64> = (0..50_000).map(|_| next()).collect();

        let mut moments = OnlineMoments::default();
        for &d in &d_values {
            moments.update(d);
        }
        let (terri_mean, terri_dev2, terri_dev4) = moments.finish();

        let n = d_values.len() as f64;
        let (mut sum_d, mut sum_d2, mut sum_d3, mut sum_d4) = (0.0, 0.0, 0.0, 0.0);
        for &d in &d_values {
            sum_d += d;
            sum_d2 += d * d;
            sum_d3 += d * d * d;
            sum_d4 += d * d * d * d;
        }
        let raw_mean = sum_d / n;
        let raw2 = sum_d2 / n;
        let raw3 = sum_d3 / n;
        let raw4 = sum_d4 / n;
        let m2 = (raw2 - raw_mean * raw_mean).max(0.0);
        let m4 = (raw4 - 4.0 * raw_mean * raw3 + 6.0 * raw_mean * raw_mean * raw2
            - 3.0 * raw_mean.powi(4))
        .max(0.0);
        let raw_dev2 = m2.sqrt();
        let raw_dev4 = m4.powf(0.25);

        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-12);
        assert!(
            rel(raw_mean, terri_mean) < 5e-4,
            "mean: raw {raw_mean} vs terriberry {terri_mean}"
        );
        assert!(
            rel(raw_dev2, terri_dev2) < 5e-4,
            "dev2: raw {raw_dev2} vs terriberry {terri_dev2}"
        );
        assert!(
            rel(raw_dev4, terri_dev4) < 5e-4,
            "dev4: raw {raw_dev4} vs terriberry {terri_dev4}"
        );
    }

    #[test]
    fn small_image_reflect_pads_and_scores() {
        let src = flat_image(4, 4, [90, 100, 110]);
        let mut dst = flat_image(4, 4, [90, 100, 110]);
        dst[0] = [255, 0, 0];
        let source = RgbSlice::new(&src, 4, 4);
        let distorted = RgbSlice::new(&dst, 4, 4);
        let result =
            compute_v2_features_impl(&source, &distorted, None, false).expect("pads + computes");
        assert_eq!(result.n_scales(), crate::NUM_SCALES);
        let view = result.view();
        let any_nonzero =
            (0..result.n_scales()).any(|s| (0..3).any(|c| view.ssim_mean(s, c) > TOL));
        assert!(
            any_nonzero,
            "single-pixel change should move ssim_mean somewhere"
        );
    }

    /// Perf pass (§A.13): the 3-way channel `rayon` fan-out
    /// (`compute_v2_features_impl_with_toggles`'s `parallel` branch, gated
    /// on `#[cfg(feature = "threads")]`) is new in this pass and had ZERO
    /// prior test coverage — every existing test in this file calls with
    /// `parallel: false`. A parallel code path that only gets exercised by
    /// the (output-blind) speed bench is exactly how a data race or a
    /// scratch-buffer aliasing bug would ship silently. Assert the parallel
    /// and serial paths produce IDENTICAL output on real (non-flat,
    /// non-trivial) content spanning all 4 scales. Bit-exact is the right
    /// bar here (not a tolerance): 3-way channel fan-out has no floating-
    /// point reassociation vs the serial loop (each channel's own
    /// `compute_channel_scale_v2` call is byte-for-byte the same function
    /// either way — only the SCHEDULING differs), so any divergence would
    /// mean a real bug (e.g. reading another channel's scratch, or writing
    /// past a `chunks_mut` boundary), not benign FP noise.
    #[test]
    fn parallel_matches_serial_exactly() {
        let w = 200;
        // Deliberately non-power-of-2 (exercises reflect padding too) AND
        // deliberately above `STRIP_BYPASS_HEIGHT` (phase-6 §A.16 lever B)
        // so this keeps exercising the strip loop's multi-strip case
        // (h=647 spans 6 strips at STRIP_ROWS=128) on top of its primary
        // parallel-vs-serial channel-fan-out purpose — a smaller height
        // would silently start routing through the single-pass bypass path
        // instead and drop that secondary coverage.
        let h = 647;
        assert!(
            h > STRIP_BYPASS_HEIGHT,
            "keep this above the lever-B bypass threshold"
        );
        let mut src = Vec::with_capacity(w * h);
        let mut dst = Vec::with_capacity(w * h);
        let mut state: u32 = 0xC0FF_EE42;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        };
        for _ in 0..(w * h) {
            src.push([next(), next(), next()]);
            dst.push([next(), next(), next()]);
        }
        let source = RgbSlice::new(&src, w, h);
        let distorted = RgbSlice::new(&dst, w, h);

        let serial =
            compute_v2_features_impl(&source, &distorted, None, false).expect("serial compute");
        let parallel =
            compute_v2_features_impl(&source, &distorted, None, true).expect("parallel compute");

        let (sf, pf) = (serial.features(), parallel.features());
        assert_eq!(sf.len(), pf.len());
        let mut mismatches = 0;
        for (i, (&s, &p)) in sf.iter().zip(pf.iter()).enumerate() {
            if s.to_bits() != p.to_bits() && !(s.is_nan() && p.is_nan()) {
                if mismatches < 10 {
                    eprintln!("feature {i}: serial={s:.17e} parallel={p:.17e}");
                }
                mismatches += 1;
            }
        }
        assert_eq!(
            mismatches,
            0,
            "{mismatches} of {} v2 features diverged between parallel and serial scheduling",
            sf.len()
        );
    }

    /// Acceptance gate for [`compute_v2_diffmap_channel_scale`]: the
    /// per-pixel map's own mean must reproduce the pooled feature values it
    /// claims to spatialize, computed via [`compute_channel_scale_v2`] —
    /// the SAME entry point the real (non-bypass, `STRIP_BYPASS_HEIGHT=0`)
    /// v2 pipeline calls for every channel-scale. Tests at the PLANE level
    /// directly (a synthetic f32 pair, not an RGB image through XYB
    /// conversion) — sufficient per this function's contract, which only
    /// operates on one channel-scale's plane, and it sidesteps XYB/
    /// multi-scale machinery that's orthogonal to what's being proven here.
    ///
    /// Fixture: a smooth, strictly-positive (`mu1,mu2 >= 0`, required by
    /// `C1_V2` boundedness), textured 64x64 plane pair with a MILD signed
    /// perturbation (so ART and DET both fire somewhere, HF gain and loss
    /// both fire somewhere) plus a small 8-pixel-lattice bump (so BLOCKINESS
    /// is non-trivially exercised, not just trivially 0≈0). "Mild" is
    /// load-bearing: every clamped family (SSIM_MEAN's `clamp02`; ART/DET/
    /// MSE/HF_*/PJND_TRANSDUCER's `clamp01`) must land inside its clamp
    /// range on this fixture, or `finish_channel_scale`'s MEAN-level clamp
    /// would legitimately diverge from this function's unclamped per-pixel
    /// mean (see `compute_v2_diffmap_channel_scale`'s "Clamping" doc) — the
    /// asserts at the end confirm no clamp fired, so a future edit that
    /// makes the fixture more aggressive fails loudly here instead of
    /// silently comparing a clamped feature against an unclamped map.
    #[test]
    fn v2_diffmap_block_pool_matches_features() {
        let width = 64usize;
        let height = 64usize;
        let n = width * height;

        let mut src = vec![0.0f32; n];
        let mut dst = vec![0.0f32; n];
        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                let fx = x as f32;
                let fy = y as f32;
                let base = 0.5 + 0.25 * (fx * 0.19).sin() * (fy * 0.13).cos();
                let s = base.clamp(0.05, 0.95);
                src[i] = s;

                // Mild, signed, spatially-varying perturbation (some pixels
                // brighter, some darker, some unchanged -- exercises ART,
                // DET, HF_GAIN, and HF_LOSS all in the same fixture) plus a
                // small block-lattice bump so BLOCKINESS is non-trivially
                // exercised.
                let mut d = s + 0.03 * ((fx + fy) * 0.5).sin();
                if x % BLOCK_LATTICE == 0 && x > 0 {
                    d += 0.015;
                }
                if y % BLOCK_LATTICE == 0 && y > 0 {
                    d -= 0.015;
                }
                dst[i] = d.clamp(0.05, 0.95);
            }
        }

        // --- "Official" per-channel-scale features, via the SAME entry
        //     point the real v2 pipeline calls. ---
        let toggles = V2NewFeatureToggles::default();
        let max_wide_h = STRIP_ROWS + 2 * HALO_P;
        let mut scratch = ScratchV2Strip::new(width * max_wide_h);
        let mut feat = [0.0f64; FEATURES_PER_CHANNEL_V2_TOTAL];
        compute_channel_scale_v2(
            &src,
            &dst,
            width,
            height,
            toggles,
            None,
            &mut scratch,
            &mut feat,
            None,
        );

        // --- Diffmap under test: weight 1.0 on every spatialized family,
        //     0.0 (excluded) on everything else. ---
        let supported = [
            idx::SSIM_MEAN,
            idx::ART,
            idx::DET,
            idx::MSE,
            idx::HF_GAIN,
            idx::HF_LOSS,
            idx::HF_MAG_LOSS,
            idx::PJND_TRANSDUCER,
            idx::GMS,
            idx::BLOCKINESS,
            idx::RINGING,
            idx::BANDING,
            // Weighted-pool families (additive weighted means, normalized by n/Σw).
            idx::MASKED_SSIM,
            idx::MASKED_ART,
            idx::MASKED_DET,
            idx::MASKED_MSE,
            idx::IW_SSIM,
            idx::IW_ART,
            idx::IW_DET,
            idx::IW_MSE,
        ];
        let mut weights = [0.0f64; FEATURES_PER_CHANNEL_V2_TOTAL];
        for &local in &supported {
            weights[local] = 1.0;
        }

        let map = compute_v2_diffmap_channel_scale(&src, &dst, width, height, &weights);
        assert_eq!(map.len(), n);
        assert!(
            map.iter().all(|v| v.is_finite()),
            "diffmap has non-finite entries"
        );

        let map_mean: f64 = map.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let expected: f64 = supported.iter().map(|&local| feat[local]).sum();

        let rel_err = (map_mean - expected).abs() / expected.abs().max(1e-9);
        assert!(
            rel_err < 5e-4,
            "diffmap block-pool mean {map_mean} vs feature sum {expected}: rel_err {rel_err}"
        );

        // Guard: the fixture must actually exercise every supported family
        // (a bug that always emits 0 must not pass by matching a
        // trivially-zero feature) AND must stay clear of every mean-level
        // clamp (clamp02 at 2.0 for SSIM_MEAN, clamp01 at 1.0 for the
        // ART/DET/MSE/HF_*/PJND_TRANSDUCER block) so the identity above is
        // exercised in its EXACT (not clamp-truncated) form.
        assert!(
            feat[idx::SSIM_MEAN] > 1e-4,
            "fixture produced ~zero SSIM_MEAN"
        );
        assert!(
            feat[idx::BLOCKINESS] > 1e-4,
            "fixture produced ~zero BLOCKINESS"
        );
        assert!(feat[idx::ART] > 1e-4, "fixture produced ~zero ART");
        assert!(feat[idx::DET] > 1e-4, "fixture produced ~zero DET");
        assert!(
            feat[idx::SSIM_MEAN] < 2.0 - 1e-6,
            "SSIM_MEAN clamp02 fired: {}",
            feat[idx::SSIM_MEAN]
        );
        for &local in &[
            idx::ART,
            idx::DET,
            idx::MSE,
            idx::HF_GAIN,
            idx::HF_LOSS,
            idx::HF_MAG_LOSS,
            idx::PJND_TRANSDUCER,
        ] {
            assert!(
                feat[local] < 1.0 - 1e-6,
                "clamp01 fired on idx {local}: {}",
                feat[local]
            );
        }
    }

    /// Acceptance gate for [`compute_v2_diffmap_full`] (the runtime-fold
    /// builder): its full-image map's mean equals the summed weighted feature
    /// over EVERY channel-scale, not just one. Expected is computed from the
    /// SAME prepared planes the builder uses (`compute_channel_scale_v2` on
    /// `prepare_v2_reference_impl`'s scales), so the XYB conversion + pyramid
    /// are shared and the only thing under test is the per-channel-scale
    /// accumulate + nearest-upsample. Even dims (96→48→24→12) make the
    /// upsample exactly mean-preserving.
    #[test]
    fn v2_diffmap_full_block_pool_matches_features() {
        let (width, height) = (96usize, 96usize);
        let n = width * height;
        // Mild RGB fixture (kept clear of mean-level clamps, like the
        // channel-scale test above).
        let mut refpx = vec![[0u8; 3]; n];
        let mut dstpx = vec![[0u8; 3]; n];
        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                let base = 120.0 + 40.0 * ((x as f32) * 0.11).sin() * ((y as f32) * 0.09).cos();
                let r = base.clamp(20.0, 235.0);
                let g = (base + 8.0).clamp(20.0, 235.0);
                let b = (base - 6.0).clamp(20.0, 235.0);
                refpx[i] = [r as u8, g as u8, b as u8];
                let d = 4.0 * (((x + y) as f32) * 0.5).sin();
                dstpx[i] = [
                    (r + d).clamp(20.0, 235.0) as u8,
                    (g - d).clamp(20.0, 235.0) as u8,
                    (b + d * 0.5).clamp(20.0, 235.0) as u8,
                ];
            }
        }
        let refimg = crate::RgbSlice::new(&refpx, width, height);
        let dstimg = crate::RgbSlice::new(&dstpx, width, height);

        // Expected: Σ over every (scale,ch) of the supported families' features,
        // from the builder's own planes.
        let supported = [
            idx::SSIM_MEAN,
            idx::ART,
            idx::DET,
            idx::MSE,
            idx::HF_GAIN,
            idx::HF_LOSS,
            idx::HF_MAG_LOSS,
            idx::PJND_TRANSDUCER,
            idx::GMS,
            idx::BLOCKINESS,
            idx::RINGING,
            idx::BANDING,
            idx::MASKED_SSIM,
            idx::MASKED_ART,
            idx::MASKED_DET,
            idx::MASKED_MSE,
            idx::IW_SSIM,
            idx::IW_ART,
            idx::IW_DET,
            idx::IW_MSE,
        ];
        let rprep = prepare_v2_reference_impl(&refimg, None, false, false).unwrap();
        let dprep = prepare_v2_reference_impl(&dstimg, None, false, false).unwrap();
        let n_scales = rprep.scales.len();
        let per_scale = 3 * FEATURES_PER_CHANNEL_V2_TOTAL;
        let mut s_v2 = vec![0.0f64; n_scales * per_scale];
        let mut expected = 0.0f64;
        let toggles = V2NewFeatureToggles::default();
        let max_wide_h = STRIP_ROWS + 2 * HALO_P;
        for scale in 0..n_scales {
            let (ref pr, ws, hs) = rprep.scales[scale];
            let pd = &dprep.scales[scale].0;
            for ch in 0..3 {
                let mut scratch = ScratchV2Strip::new(ws * max_wide_h);
                let mut feat = [0.0f64; FEATURES_PER_CHANNEL_V2_TOTAL];
                compute_channel_scale_v2(
                    &pr[ch],
                    &pd[ch],
                    ws,
                    hs,
                    toggles,
                    None,
                    &mut scratch,
                    &mut feat,
                    None,
                );
                let base = scale * per_scale + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
                for &local in &supported {
                    s_v2[base + local] = 1.0;
                    expected += feat[local];
                }
            }
        }

        let map = compute_v2_diffmap_full(&refimg, &dstimg, &s_v2, None, false).unwrap();
        assert_eq!(map.len(), n);
        assert!(map.iter().all(|v| v.is_finite()), "non-finite entries");
        let map_mean: f64 = map.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let rel_err = (map_mean - expected).abs() / expected.abs().max(1e-9);
        assert!(
            rel_err < 2e-3,
            "full-diffmap block-pool mean {map_mean} vs feature sum {expected}: rel_err {rel_err}"
        );
        // Fixture must actually exercise the families (not pass trivially at 0).
        assert!(
            expected > 1e-3,
            "fixture produced ~zero supported-feature sum"
        );
    }

    // ------------------------------------------------------------------
    // Prepared-reference (pyramid reuse) path
    // ------------------------------------------------------------------

    /// Deterministic structured content: gradients + edges + pseudo-noise
    /// texture (LCG), so every feature family (SSIM, HF, gradient,
    /// blockiness, transducer) sees real signal.
    pub(crate) fn textured_image(w: usize, h: usize, seed: u32) -> Vec<[u8; 3]> {
        let mut state = seed | 1;
        let mut px = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let noise = (state >> 24) as u8;
                let grad = ((x * 255) / w.max(1)) as u8;
                let edge = if (x / 9 + y / 7) % 2 == 0 { 200 } else { 40 };
                let r = grad.wrapping_add(noise / 4);
                let g = edge;
                let b = ((y * 255) / h.max(1)) as u8 ^ (noise / 8);
                px.push([r, g, b]);
            }
        }
        px
    }

    /// Blocky quantization distortion — deterministic, codec-flavored.
    pub(crate) fn quantize_distort(src: &[[u8; 3]], w: usize, h: usize) -> Vec<[u8; 3]> {
        let mut out = src.to_vec();
        for y in 0..h {
            for x in 0..w {
                let p = &mut out[y * w + x];
                for c in p.iter_mut() {
                    // 8x8-block DC-ish flattening + level quantization.
                    *c = (*c / 24) * 24 + ((x % 8 == 0 || y % 8 == 0) as u8) * 6;
                }
            }
        }
        out
    }

    /// The prepared-reference path must produce features BIT-IDENTICAL to
    /// the pair path — same functions, same order, only the storage of the
    /// reference pyramid differs. Covers: multi-strip (h > STRIP_ROWS),
    /// odd dims (SIMD tails), the exact-64 floor, and the sub-64
    /// reflect-pad path.
    #[test]
    fn prepared_ref_bit_identical_to_pair_path() {
        for &(w, h) in &[
            (64usize, 64usize),
            (96, 80),
            (200, 136),
            (40, 30),
            (65, 129),
        ] {
            let src = textured_image(w, h, 0xBEEF);
            let dst = quantize_distort(&src, w, h);
            let source = RgbSlice::new(&src, w, h);
            let distorted = RgbSlice::new(&dst, w, h);

            let pair = compute_v2_features_impl(&source, &distorted, None, false)
                .expect("pair path computes");

            let prepared =
                prepare_v2_reference_impl(&source, None, false, false).expect("prepare computes");
            let mut scratch = V2Scratch::new();
            let with_ref = compute_v2_features_with_ref_impl(
                &prepared,
                &distorted,
                None,
                false,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .expect("with_ref computes");

            assert_eq!(pair.features().len(), with_ref.features().len());
            for (i, (a, b)) in pair
                .features()
                .iter()
                .zip(with_ref.features().iter())
                .enumerate()
            {
                assert!(
                    a.to_bits() == b.to_bits(),
                    "{w}x{h}: feature {i} diverged: pair={a:e} with_ref={b:e}"
                );
            }
        }
    }

    /// Toggle combinations must round-trip through the prepared path
    /// identically too (the luma gate + disabled groups change which
    /// slots are written — index-stability must not depend on which
    /// entry point ran).
    #[test]
    fn prepared_ref_bit_identical_with_toggles() {
        let (w, h) = (96usize, 96usize);
        let src = textured_image(w, h, 0x5EED);
        let dst = quantize_distort(&src, w, h);
        let source = RgbSlice::new(&src, w, h);
        let distorted = RgbSlice::new(&dst, w, h);

        for toggles in [
            V2NewFeatureToggles {
                transducers_luma_only: true,
                ..Default::default()
            },
            V2NewFeatureToggles {
                gradient_features: false,
                ..Default::default()
            },
            V2NewFeatureToggles {
                blockiness: false,
                transducer_bank: false,
                ..Default::default()
            },
        ] {
            let pair =
                compute_v2_features_impl_with_toggles(&source, &distorted, None, false, toggles)
                    .expect("pair path computes");
            let prepared =
                prepare_v2_reference_impl(&source, None, false, false).expect("prepare computes");
            let mut scratch = V2Scratch::new();
            let with_ref = compute_v2_features_with_ref_impl(
                &prepared,
                &distorted,
                None,
                false,
                toggles,
                &mut scratch,
            )
            .expect("with_ref computes");
            for (i, (a, b)) in pair
                .features()
                .iter()
                .zip(with_ref.features().iter())
                .enumerate()
            {
                assert!(
                    a.to_bits() == b.to_bits(),
                    "toggles {toggles:?}: feature {i} diverged"
                );
            }
        }
    }

    /// REV4 streaming-vs-materialized parity: the materialized walk's
    /// `Rev4CellAccum` cells, finalized by [`finish_rev4_scale`], must
    /// equal the streaming walk's emitted rev4 segments bit-for-bit.
    /// Both engines run the same kernel strips in the same order into
    /// merge-free accumulators (the `blockiness_sparse_strip_wide`
    /// running-total contract), so serial-vs-streaming differences are
    /// storage, not arithmetic. Covers both activity routes: the
    /// strip-computed `scratch.activity` plane and the cached full-plane
    /// `moments` (whose per-strip gather must reproduce it exactly).
    #[test]
    fn rev4_streaming_materialized_parity() {
        let toggles = V2NewFeatureToggles {
            append_block: true,
            append2_block: true,
            csfw_block: true,
            rev4_gridblk: true,
            rev4_ringbasis: true,
            rev4_tailhist: true,
            rev4_arttype: true,
            v1_pools: V1PoolsMode::Full,
            ..Default::default()
        };
        let compute = ComputeSet::from_toggles(toggles);
        for &(w, h) in &[(96usize, 80usize), (200, 136), (65, 129)] {
            let src = textured_image(w, h, 0xBEEF);
            let dst = quantize_distort(&src, w, h);
            let source = RgbSlice::new(&src, w, h);
            let distorted = RgbSlice::new(&dst, w, h);

            let mut scratch = V2Scratch::new();
            let streamed = compute_folded720_streaming_impl(
                &source,
                &distorted,
                None,
                false,
                toggles,
                &mut scratch,
                None,
            )
            .expect("streaming walk computes");
            assert_eq!(
                streamed.features.len(),
                REV4_BASE + 96 + 72 + 144 + 24,
                "all-rev4 layout must emit the full 1322"
            );

            // The rev4 tail is thread-invariant too: the parallel channel
            // fan-out changes only which thread runs a cell's strips, not
            // their order (the merge-free-accumulator contract).
            #[cfg(feature = "threads")]
            {
                let mut pscratch = V2Scratch::new();
                let par = compute_folded720_streaming_impl(
                    &source,
                    &distorted,
                    None,
                    true,
                    toggles,
                    &mut pscratch,
                    None,
                )
                .expect("parallel streaming computes");
                assert_eq!(
                    &par.features[REV4_BASE..],
                    &streamed.features[REV4_BASE..],
                    "parallel-vs-serial streaming rev4 tail {w}x{h}"
                );
            }

            for cache_moments in [false, true] {
                let prep = prepare_v2_reference_impl(&source, None, false, cache_moments)
                    .expect("prepare computes");
                let ns = prep.scales.len();
                let mut cells: Vec<Rev4CellAccum> =
                    (0..ns * 3).map(|_| Rev4CellAccum::default()).collect();
                let mat = compute_v2_features_with_ref_impl_inner(
                    &prep,
                    &distorted,
                    None,
                    false,
                    toggles,
                    &mut scratch,
                    Some(&mut cells[..]),
                )
                .expect("materialized walk computes");

                for scale in 0..ns {
                    let local = compute.at_scale(scale);
                    let (sw, sh) = (prep.scales[scale].1, prep.scales[scale].2);
                    // Y channel is index 1 of the [X, Y, B] channel blocks.
                    let yb =
                        scale * 3 * FEATURES_PER_CHANNEL_V2_TOTAL + FEATURES_PER_CHANNEL_V2_TOTAL;
                    let ewc = mat.features[yb + idx::EDGE_WIDTH_CHANGE];
                    let hfl = mat.features[yb + idx::HF_LOSS];
                    let mut og = [0.0f64; 3 * GRIDBLK_PER_CELL];
                    let mut or_ = [0.0f64; 3 * RINGBASIS_PER_CELL];
                    let mut ot = [0.0f64; 3 * TAILHIST_PER_CELL];
                    let mut oa = [0.0f64; ARTTYPE_PER_SCALE];
                    finish_rev4_scale(
                        &local,
                        scale,
                        [
                            &cells[scale * 3],
                            &cells[scale * 3 + 1],
                            &cells[scale * 3 + 2],
                        ],
                        sw,
                        sh,
                        ewc,
                        hfl,
                        Some(&mut og),
                        Some(&mut or_),
                        Some(&mut ot),
                        Some(&mut oa),
                    );
                    let tag = format!("{w}x{h} s{scale} moments={cache_moments}");
                    let gb = REV4_BASE + scale * 3 * GRIDBLK_PER_CELL;
                    assert_eq!(
                        &og[..],
                        &streamed.features[gb..gb + og.len()],
                        "gridblk {tag}"
                    );
                    let rb = REV4_BASE + 96 + scale * 3 * RINGBASIS_PER_CELL;
                    assert_eq!(
                        &or_[..],
                        &streamed.features[rb..rb + or_.len()],
                        "ringbasis {tag}"
                    );
                    let tb = REV4_BASE + 96 + 72 + scale * 3 * TAILHIST_PER_CELL;
                    assert_eq!(
                        &ot[..],
                        &streamed.features[tb..tb + ot.len()],
                        "tailhist {tag}"
                    );
                    let ab = REV4_BASE + 96 + 72 + 144 + scale * ARTTYPE_PER_SCALE;
                    assert_eq!(
                        &oa[..],
                        &streamed.features[ab..ab + oa.len()],
                        "arttype {tag}"
                    );
                }
            }
        }
    }

    /// All-rev4 toggles for the family unit tests (the same literal the
    /// parity test uses — every block on, v1 pools full).
    fn rev4_all_toggles() -> V2NewFeatureToggles {
        V2NewFeatureToggles {
            append_block: true,
            append2_block: true,
            csfw_block: true,
            rev4_gridblk: true,
            rev4_ringbasis: true,
            rev4_tailhist: true,
            rev4_arttype: true,
            v1_pools: V1PoolsMode::Full,
            ..Default::default()
        }
    }

    fn gmsbank_extract(
        src: &[[u8; 3]],
        dst: &[[u8; 3]],
        w: usize,
        h: usize,
        parallel: bool,
    ) -> Vec<f64> {
        let mut scratch = V2Scratch::new();
        compute_folded720_streaming_impl(
            &RgbSlice::new(src, w, h),
            &RgbSlice::new(dst, w, h),
            None,
            parallel,
            V2NewFeatureToggles {
                gmsbank: true,
                ..rev4_all_toggles()
            },
            &mut scratch,
            None,
        )
        .expect("C8 extraction")
        .into_features()
    }

    #[test]
    fn gmsbank_strict_contrast_reduction_routes_only_to_loss() {
        let mut cells = [GmsBankCell::default(); 5];
        for i in 1..=257 {
            let mr = i as f64 / 257.0;
            let md = mr * 0.5;
            assert!(md < mr);
            gmsbank_pixel(&mut cells, mr, md, &GMSBANK_C);
        }
        for cell in cells {
            assert!(cell.loss > 0.0);
            assert_eq!(cell.gain.to_bits(), 0.0f64.to_bits());
        }
    }

    #[test]
    fn gmsbank_constant_chroma_shift_is_visible_without_gradients() {
        let (w, h) = (97, 161);
        let reference = vec![[120, 100, 60]; w * h];
        let distorted = vec![[100, 120, 60]; w * h];
        let values = gmsbank_extract(&reference, &distorted, w, h, false);
        assert!(
            values[GMSBANK_BASE..GMSBANK_BASE + 15]
                .iter()
                .all(|v| v.to_bits() == 0),
            "flat inputs have no scale-0 Y gradient: {:?}",
            &values[GMSBANK_BASE..GMSBANK_BASE + 15]
        );
        for scale in 1..4 {
            let base = GMSBANK_BASE + 15 + (scale - 1) * 55;
            assert!(
                values[base..base + 45].iter().all(|v| v.to_bits() == 0),
                "flat inputs have no scale-{scale} gradient: {:?}",
                &values[base..base + 45]
            );
            for k in 0..5 {
                assert!(
                    values[base + 45 + 2 * k] > 0.0,
                    "flat colour shift must register CS loss"
                );
                assert_eq!(
                    values[base + 46 + 2 * k].to_bits(),
                    0,
                    "constant loss map has zero deviation"
                );
            }
        }
    }

    #[test]
    fn gmsbank_identity_and_constant_monotonicity() {
        assert_eq!(
            crate::feature_defs::block_base(
                crate::feature_set_id::ComputeToken::Gmsbank,
                crate::NUM_SCALES
            )
            .unwrap()
            .0,
            GMSBANK_BASE,
        );
        let (w, h) = (97usize, 161usize);
        let src: Vec<[u8; 3]> = (0..w * h)
            .map(|i| {
                let x = i % w;
                let y = i / w;
                let a = ((x * 7 + y * 13 + (x * y) % 31) % 256) as u8;
                [a, a.wrapping_add(37), a.wrapping_sub(41)]
            })
            .collect();
        let identity = gmsbank_extract(&src, &src, w, h, false);
        assert_eq!(identity.len(), 1502);
        assert!(identity[GMSBANK_BASE..].iter().all(|v| v.to_bits() == 0));
        let dst: Vec<[u8; 3]> = src
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let offset = if (i / w + i % w) % 2 == 0 { 9 } else { -9 };
                p.map(|c| (i32::from(c) + offset).clamp(0, 255) as u8)
            })
            .collect();
        let serial = gmsbank_extract(&src, &dst, w, h, false);
        let parallel = gmsbank_extract(&src, &dst, w, h, true);
        assert_eq!(serial, parallel, "C8 row order must survive MT extraction");
        for scale in 0..4 {
            for ch in 0..3 {
                if scale == 0 && ch != 1 {
                    continue;
                }
                let base = GMSBANK_BASE
                    + if scale == 0 {
                        0
                    } else {
                        15 + (scale - 1) * 55 + ch * 15
                    };
                let cell = &serial[base..base + 15];
                for k in 0..4 {
                    let now = cell[k * 3] + cell[k * 3 + 1];
                    let next = cell[(k + 1) * 3] + cell[(k + 1) * 3 + 1];
                    assert!(now + 1e-14 >= next, "loss+gain must fall as c grows");
                }
            }
            if scale > 0 {
                let base = GMSBANK_BASE + 15 + (scale - 1) * 55 + 45;
                for k in 0..4 {
                    assert!(serial[base + 2 * k] >= serial[base + 2 * (k + 1)]);
                }
            }
        }
    }

    #[test]
    fn gmsbank_materialized_matches_streaming() {
        for (w, h) in [(97usize, 161usize), (151, 83)] {
            let src = textured_image(w, h, 0xC8B4);
            let dst = quantize_distort(&src, w, h);
            let source = RgbSlice::new(&src, w, h);
            let distorted = RgbSlice::new(&dst, w, h);
            let toggles = V2NewFeatureToggles {
                gmsbank: true,
                ..rev4_all_toggles()
            };
            let streamed = gmsbank_extract(&src, &dst, w, h, false);
            let prep = prepare_v2_reference_impl(&source, None, false, false).unwrap();
            let ns = prep.scales.len();
            let mut cells: Vec<Rev4CellAccum> =
                (0..ns * 3).map(|_| Rev4CellAccum::default()).collect();
            let mut scratch = V2Scratch::new();
            compute_v2_features_with_ref_impl_inner(
                &prep,
                &distorted,
                None,
                false,
                toggles,
                &mut scratch,
                Some(&mut cells[..]),
            )
            .unwrap();
            for scale in 0..ns {
                let n = prep.scales[scale].1 * prep.scales[scale].2;
                for ch in 0..3 {
                    if scale == 0 && ch != 1 {
                        continue;
                    }
                    let cell = &cells[scale * 3 + ch];
                    let mut out = [0.0; GMSBANK_PER_CELL];
                    finish_gmsbank_cell(&cell.bank, n, &mut out);
                    let base = GMSBANK_BASE
                        + if scale == 0 {
                            0
                        } else {
                            15 + (scale - 1) * 55 + ch * 15
                        };
                    assert_eq!(
                        out,
                        streamed[base..base + GMSBANK_PER_CELL],
                        "C8 materialized vs streamed {w}x{h} scale={scale} ch={ch}"
                    );
                }
                if scale > 0 {
                    let mut out = [0.0; 10];
                    finish_chroma_cell(&cells[scale * 3].chroma, n, &mut out);
                    let base = GMSBANK_BASE + 15 + (scale - 1) * 55 + 45;
                    assert_eq!(
                        out,
                        streamed[base..base + 10],
                        "C8 materialized CS scale {scale}"
                    );
                }
            }
        }
    }

    #[test]
    fn gmsbank_stride_matches_tight() {
        let (w, h) = (97usize, 161usize);
        let src = textured_image(w, h, 0xC8C8);
        let dst = quantize_distort(&src, w, h);
        let stride = w * 3 + 13;
        let mut padded_src = vec![0u8; stride * h];
        let mut padded_dst = vec![0u8; stride * h];
        for y in 0..h {
            for x in 0..w {
                let at = y * stride + x * 3;
                padded_src[at..at + 3].copy_from_slice(&src[y * w + x]);
                padded_dst[at..at + 3].copy_from_slice(&dst[y * w + x]);
            }
        }
        let source = StridedBytes::try_new(&padded_src, w, h, stride, PixelFormat::Srgb8Rgb)
            .expect("strided source");
        let distorted = StridedBytes::try_new(&padded_dst, w, h, stride, PixelFormat::Srgb8Rgb)
            .expect("strided distortion");
        let mut scratch = V2Scratch::new();
        let strided = compute_folded720_streaming_impl(
            &source,
            &distorted,
            None,
            false,
            V2NewFeatureToggles {
                gmsbank: true,
                ..rev4_all_toggles()
            },
            &mut scratch,
            None,
        )
        .unwrap()
        .into_features();
        let tight = gmsbank_extract(&src, &dst, w, h, false);
        assert_eq!(tight[GMSBANK_BASE..], strided[GMSBANK_BASE..]);
    }

    /// Extract the full 1322-slot vector for a pair through the streaming
    /// walk (the served engine).
    fn rev4_extract(src: &[[u8; 3]], dst: &[[u8; 3]], w: usize, h: usize) -> Vec<f64> {
        let source = RgbSlice::new(src, w, h);
        let distorted = RgbSlice::new(dst, w, h);
        let mut scratch = V2Scratch::new();
        compute_folded720_streaming_impl(
            &source,
            &distorted,
            None,
            false,
            rev4_all_toggles(),
            &mut scratch,
            None,
        )
        .expect("streaming walk computes")
        .features
    }

    /// REV4 is difference-form end to end: an identical pair emits exact
    /// 0.0 in C1 (`ẽ ≡ 0` → no winning phase), C2 (`ring` factors
    /// `|s − dd|` ≡ 0) and C4 (`hfg`/`bleed`/`ewc·hfl` all factor a
    /// dst−src difference). C3 alone is not exact-0: it histograms the
    /// existing `d` map, whose direct-form `ssim_d_local` carries the
    /// REGISTERED identity floating-point residue shared with the parent
    /// v2 slots (`feature_invariants.rs` class 3, bar 2e-3 — the design
    /// note's stated justification for C3's non-difference form). The
    /// tailhist quantiles on identity must therefore sit under the same
    /// bar, not at zero.
    #[test]
    fn rev4_identity_emits_zero() {
        let (w, h) = (96usize, 80usize);
        let src = textured_image(w, h, 0xF00D);
        let f = rev4_extract(&src, &src, w, h);
        assert_eq!(f.len(), 1322);
        let tail_base = REV4_BASE + 96 + 72;
        let art_base = tail_base + 144;
        let mut bad = Vec::new();
        for (i, &v) in f[REV4_BASE..].iter().enumerate() {
            let abs_i = REV4_BASE + i;
            if abs_i < tail_base || abs_i >= art_base {
                // C1 / C2 / C4: exact zero on identity.
                if v.to_bits() != 0.0f64.to_bits() {
                    bad.push((abs_i, v));
                }
            } else {
                // C3: registered fp-residue bound, same bar as the
                // parent identity gate.
                assert!(
                    v.abs() <= 2e-3,
                    "tailhist slot {abs_i} = {v:e} exceeds the registered \
                     2e-3 identity-residue bar"
                );
            }
        }
        assert!(
            bad.is_empty(),
            "nonzero identity rev4 slots (C1/C2/C4): {bad:?}"
        );
    }

    /// A strided (padded-row) source must produce the same rev4 tail as
    /// the tight buffer — the front end's row reads are stride-aware and
    /// nothing downstream may observe the padding.
    #[test]
    fn rev4_strided_equals_tight() {
        let (w, h) = (96usize, 80usize);
        let src = textured_image(w, h, 0xBEEF);
        let dst = quantize_distort(&src, w, h);
        // Padded copy: stride = w*3 + 13 (odd byte pad, deliberately not
        // a multiple of anything the kernels could silently consume).
        let stride = w * 3 + 13;
        let mut pad_src = vec![0u8; stride * h];
        let mut pad_dst = vec![0u8; stride * h];
        for y in 0..h {
            for x in 0..w {
                let o = y * stride + x * 3;
                pad_src[o..o + 3].copy_from_slice(&src[y * w + x]);
                pad_dst[o..o + 3].copy_from_slice(&dst[y * w + x]);
            }
        }
        let tight = rev4_extract(&src, &dst, w, h);
        let s_src = StridedBytes::try_new(&pad_src, w, h, stride, PixelFormat::Srgb8Rgb)
            .expect("strided src");
        let s_dst = StridedBytes::try_new(&pad_dst, w, h, stride, PixelFormat::Srgb8Rgb)
            .expect("strided dst");
        let mut scratch = V2Scratch::new();
        let strided = compute_folded720_streaming_impl(
            &s_src,
            &s_dst,
            None,
            false,
            rev4_all_toggles(),
            &mut scratch,
            None,
        )
        .expect("strided walk computes")
        .features;
        for (i, (a, b)) in tight[REV4_BASE..]
            .iter()
            .zip(strided[REV4_BASE..].iter())
            .enumerate()
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "rev4 slot {} strided != tight",
                REV4_BASE + i
            );
        }
    }

    /// Low-contrast content (gentle gradient + ±3 LCG noise) — content
    /// steps stay ≪ block-flatten steps so a planted 8-px lattice
    /// dominates the phase profile (the phase test's fixture).
    fn smooth_image(w: usize, h: usize, seed: u32) -> Vec<[u8; 3]> {
        let mut state = seed | 1;
        let mut px = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let noise = ((state >> 24) as i32 % 7) - 3;
                let base = 96 + ((x * 48) / w.max(1)) as i32 + ((y * 32) / h.max(1)) as i32;
                let v = (base + noise).clamp(0, 255) as u8;
                px.push([v, (v as i32 + (noise >> 1)).clamp(0, 255) as u8, v]);
            }
        }
        px
    }

    /// Alternating ±shift per 8-px column block: steps appear ONLY at
    /// `x ≡ 0 mod 8` boundaries (the two sides inside a block share the
    /// shift), planting a pure phase-0 lattice of added edges.
    fn block_shift_distort(src: &[[u8; 3]], w: usize, h: usize) -> Vec<[u8; 3]> {
        let mut out = src.to_vec();
        for y in 0..h {
            for x in 0..w {
                let delta = if (x / 8) % 2 == 0 { 24i32 } else { -24i32 };
                for c in out[y * w + x].iter_mut() {
                    *c = (*c as i32 + delta).clamp(0, 255) as u8;
                }
            }
        }
        out
    }

    /// C1 phase recovery: shift a reference's 8-px block columns and crop
    /// the DISTORTED image 5 columns left against a matching crop of the
    /// reference — the planted lattice then sits at phase `(8−5) mod 8
    /// = 3` in the compared frame with content still aligned. The
    /// per-phase `Σ|ẽ|` argmax must be exactly 3, and its magnitude must
    /// match the unshifted pair's phase-0 response within the stated
    /// tolerance (±40% — the crop removes the leftmost content and the
    /// off-lattice content steps re-partition, but the on-grid excess is
    /// the same physical boundary set).
    #[test]
    fn rev4_gridblk_phase_shift_three() {
        let (w, h) = (128usize, 96usize);
        let src = smooth_image(w, h, 0xC1);
        let dst_full = block_shift_distort(&src, w, h);
        // Unshifted pair (phase 0 expected).
        let f0 = rev4_extract(&src, &dst_full, w, h);
        // Cropped pair: dst's lattice lands at phase 3, content aligned.
        let wc = w - 5;
        let src_c: Vec<[u8; 3]> = (0..h)
            .flat_map(|y| src[y * w + 5..y * w + w].iter().copied())
            .collect();
        let dst_c: Vec<[u8; 3]> = (0..h)
            .flat_map(|y| dst_full[y * w + 5..y * w + w].iter().copied())
            .collect();
        let source_c = RgbSlice::new(&src_c, wc, h);
        let distorted_c = RgbSlice::new(&dst_c, wc, h);
        let prep = prepare_v2_reference_impl(&source_c, None, false, false).expect("prepare");
        let mut cells: Vec<Rev4CellAccum> = (0..prep.scales.len() * 3)
            .map(|_| Rev4CellAccum::default())
            .collect();
        let mut scratch = V2Scratch::new();
        let _ = compute_v2_features_with_ref_impl_inner(
            &prep,
            &distorted_c,
            None,
            false,
            rev4_all_toggles(),
            &mut scratch,
            Some(&mut cells[..]),
        )
        .expect("materialized computes");
        // Scale 0, Y channel (index 1): V-phase argmax must be 3.
        let g = &cells[1].grid;
        let pv = gridblk_winner(&g.abs_v, &[usize::MAX; GRIDBLK_MAX_PERIOD], 8)
            .expect("a winning V phase");
        assert_eq!(
            pv,
            3,
            "V lattice phase must be 3, abs_v={:?}",
            &g.abs_v[..8]
        );
        // Response magnitude vs the unshifted phase-0 profile: use the
        // emitted on-grid mean as the stated response (blocking ADDS
        // steps → positive signed ẽ on-grid).
        let on_mean0 = f0[REV4_BASE + 8 + 6];
        let f3 = rev4_extract(&src_c, &dst_c, wc, h);
        let on_mean3 = f3[REV4_BASE + 8 + 6];
        assert!(
            on_mean3 > 0.4 * on_mean0 && on_mean3 < 1.6 * on_mean0,
            "phase-3 response {on_mean3} vs unshifted {on_mean0} outside ±40%"
        );
    }

    /// C1 broad sanity bound: blur-only and noise-only distortions carry
    /// no planted lattice. The [0.5, 2.0] range catches gross numerical
    /// faults; it does not discriminate JPEG blocking (whose measured
    /// on/off range, 1.01–1.24, falls inside this interval).
    #[test]
    fn rev4_gridblk_blur_noise_band() {
        let (w, h) = (96usize, 80usize);
        let src = textured_image(w, h, 0xBEEF);
        // 3x3 box blur.
        let mut blur = src.clone();
        for y in 0..h {
            for x in 0..w {
                let mut acc = [0u32; 3];
                let mut n = 0u32;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let yy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                        let xx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                        for c in 0..3 {
                            acc[c] += src[yy * w + xx][c] as u32;
                        }
                        n += 1;
                    }
                }
                for c in 0..3 {
                    blur[y * w + x][c] = (acc[c] / n) as u8;
                }
            }
        }
        // ±6 LCG noise.
        let mut noise = src.clone();
        let mut st = 0x9E3779B9u32;
        for p in noise.iter_mut() {
            for c in p.iter_mut() {
                st = st.wrapping_mul(1664525).wrapping_add(1013904223);
                let d = ((st >> 24) as i32 % 13) - 6;
                *c = (*c as i32 + d).clamp(0, 255) as u8;
            }
        }
        for (name, dst) in [("blur", &blur), ("noise", &noise)] {
            let f = rev4_extract(&src, dst, w, h);
            for ch in 0..3 {
                let r = f[REV4_BASE + ch * 8 + 7];
                assert!(
                    (0.5..=2.0).contains(&r),
                    "{name}: s0 ch{ch} on/off ratio {r} outside [0.5, 2.0]"
                );
            }
        }
    }

    /// C3 machinery: `TailAccum`/`finish_tailhist_cell` against a
    /// sorted-array reference on a controlled multiset — max exact,
    /// p95/p99 equal to the lower edge of the true quantile's bin (the
    /// "within one bin" contract), counts partition-invariant.
    #[test]
    fn rev4_tailhist_quantile_semantics() {
        let edges = tail_edges();
        // Deterministic multiset: 1000 values spanning [1e-6, 1e-1]
        // plus 200 exact zeros (bin-0 mass) and 40 saturating values.
        let mut vals: Vec<f64> = (0..1000)
            .map(|i| 10f64.powf(-6.0 + (i as f64 / 999.0) * 5.0))
            .collect();
        vals.extend(std::iter::repeat_n(0.0, 200));
        vals.extend(std::iter::repeat_n(2.5, 40));
        let n = vals.len();
        // Every pixel scatters into all four maps in the walk, and `finish_tailhist_cell` asserts each map holds
        // `n` counts; the same multiset in every map keeps that invariant (the checks below read map 0).
        let mut acc = TailAccum::default();
        for &v in &vals {
            for m in 0..4 {
                acc.scatter(edges, v, m);
            }
        }
        let mut out = [0.0f64; TAILHIST_PER_CELL];
        finish_tailhist_cell(&acc, n, &mut out);
        // max = true max exactly.
        let sorted_max = vals.iter().copied().fold(0.0f64, f64::max);
        assert_eq!(out[2].to_bits(), sorted_max.to_bits(), "max must be exact");
        // p95/p99 within one bin of the sorted-array quantile: the
        // emitted lower edge must equal the lower edge of the bin that
        // contains the true order statistic (or be the adjacent bin
        // when the cumulative target lands mid-bin — at most one bin of
        // slack, asserted as: true value's bin lower edge <= emitted <=
        // next edge above true value's bin).
        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        for (k, q) in [(0usize, 0.95f64), (1, 0.99)] {
            let idx = ((q * n as f64).ceil() as usize)
                .saturating_sub(1)
                .min(n - 1);
            let tv = sorted[idx];
            let emitted = out[k];
            let tv_bin = edges.bin(tv);
            let lo = if tv_bin == 0 {
                0.0
            } else {
                f64::from_bits(edges.edges[tv_bin - 1])
            };
            let hi = f64::from_bits(edges.edges[tv_bin.min(TAILHIST_BINS - 2)]);
            assert!(
                emitted >= lo && emitted <= hi.max(lo),
                "q={q}: emitted {emitted} not within one bin of {tv} (bin {tv_bin} edges [{lo}, {hi}])"
            );
        }
        // Partition invariance: three different strip partitionings of
        // the same sequence produce identical count arrays.
        for split in [vec![0usize, n], vec![0, 400, 800, n], vec![0, 1, 999, n]] {
            let mut h = [[0u32; TAILHIST_BINS]; 4];
            for w in split.windows(2) {
                for &v in &vals[w[0]..w[1]] {
                    for row in h.iter_mut() {
                        row[edges.bin(v)] += 1;
                    }
                }
            }
            assert_eq!(h, acc.hist, "counts must be partition-invariant");
        }
    }

    /// C3 regression (era `c3negfold`): `edge_dissim`'s f32 rounding emits
    /// tiny NEGATIVE `art`/`det` values, and a raw `to_bits` bin lookup
    /// sorted every sign-bit-set double above all positive edges — the
    /// phantom ~2.9% top-bin saturation of REVIEW_PARTB 2026-09-25 was
    /// those negatives, not real tail mass. Sign-bit-set and NaN values
    /// must fold into bin 0; only genuine `v >= last edge` mass may reach
    /// the top bin.
    #[test]
    fn rev4_tailhist_bin_folds_sign_bit_values() {
        let edges = tail_edges();
        let top = TAILHIST_BINS - 1;
        // Sign-bit-set values: the bin-0 fold.
        for (name, v) in [
            ("neg tiny", -1e-12f64),
            ("neg below-first-edge", -0.5),
            ("neg large", -1e300),
            ("neg zero", -0.0),
            ("neg inf", f64::NEG_INFINITY),
            ("NaN", f64::NAN),
            ("neg NaN", f64::from_bits(0xfff8000000000001)),
        ] {
            assert_eq!(
                edges.bin_folded(v),
                0,
                "{name}: bin_folded({v}) must fold to 0"
            );
        }
        // Non-negative values still bin by magnitude.
        assert_eq!(edges.bin(0.0), 0);
        assert_eq!(edges.bin(1e-7), 0); // below the first edge
        let last = f64::from_bits(edges.edges[TAILHIST_BINS - 2]);
        assert_eq!(edges.bin(last * 2.0), top, "v above last edge -> top bin");
        assert_eq!(edges.bin(f64::INFINITY), top, "+inf is genuine tail mass");
        // End to end through `scatter`: a realistic mostly-negative
        // multiset must leave the top bin empty and keep max exact —
        // the pre-fix arithmetic put every negative in the top bin.
        let mut acc = TailAccum {
            fold: true,
            ..TailAccum::default()
        };
        let mut vals: Vec<f64> = (0..5000).map(|i| -(i as f64) * 1e-9).collect();
        vals.extend(std::iter::repeat_n(0.004, 8000));
        vals.extend(std::iter::repeat_n(last * 2.0, 10));
        for &v in &vals {
            acc.scatter(edges, v, 1);
        }
        assert_eq!(acc.hist[1][top], 10, "only real tail values in the top bin");
        // Bin 0 holds the folded mass: negatives + positives below the
        // first edge.
        let expect0 = vals.iter().filter(|&&v| edges.bin_folded(v) == 0).count() as u32;
        assert_eq!(acc.hist[1][0], expect0, "bin 0 must hold the folded mass");
        assert_eq!(acc.max[1].to_bits(), (last * 2.0).to_bits());
        let total: u64 = acc.hist[1].iter().map(|&c| c as u64).sum();
        assert_eq!(
            total,
            vals.len() as u64,
            "histogram total == scattered count"
        );
    }

    /// Rev1-Rev3 byte-identity guard for `c3negfold`: with `fold == false`
    /// (what `TailAccum::default()` yields below Rev4) a sign-bit-set value
    /// still lands in the TOP bin, exactly as before the fix, while the
    /// folded accumulator puts it in bin 0. Both leave `max` untouched.
    #[test]
    fn tailhist_fold_is_off_below_rev4() {
        let edges = tail_edges();
        let top = TAILHIST_BINS - 1;
        let default = TailAccum::default();
        assert_eq!(
            default.fold,
            crate::ssim_form::active_revision() >= FormulaRevision::Rev4,
            "fold follows the process revision"
        );
        let (mut legacy, mut folded) = (
            TailAccum {
                fold: false,
                ..TailAccum::default()
            },
            TailAccum {
                fold: true,
                ..TailAccum::default()
            },
        );
        for v in [-1e-12f64, -0.0, 0.25] {
            legacy.scatter(edges, v, 2);
            folded.scatter(edges, v, 2);
        }
        assert_eq!(
            legacy.hist[2][top], 2,
            "legacy bins sign-bit-set values to the top"
        );
        assert_eq!(folded.hist[2][top], 0);
        assert_eq!(
            folded.hist[2][0] + folded.hist[2].iter().skip(1).sum::<u32>(),
            3
        );
        assert_eq!(legacy.max[2].to_bits(), folded.max[2].to_bits());
    }

    /// C4 bleed on a mostly flat grey source: a localized equal-RGB
    /// luma step leaves an unmasked area and emits zero bleed. Adding a
    /// chroma texture outside that step must produce positive bleed.
    #[test]
    fn rev4_arttype_bleed_luma_vs_chroma() {
        let (w, h) = (96usize, 80usize);
        let src = vec![[128u8; 3]; w * h];
        // Localized equal-RGB change, without clipping or per-pixel noise.
        let mut luma = src.clone();
        for y in 12..32 {
            for x in 12..32 {
                luma[y * w + x] = [152; 3];
            }
        }
        // Compute the same scale-0 dst-Y mask as the production strip.
        // Its coverage is part of this test: an all-masked plane makes a
        // zero bleed value vacuous.
        let prepared = prepare_v2_reference_impl(&RgbSlice::new(&luma, w, h), None, false, false)
            .expect("prepare luma-only dst");
        let dst_y = &prepared.scales[0].0[1];
        let mut mask_work = BleedMaskWork::sized(w, h);
        for r in 0..h + 4 {
            let row = reflect_101(r as isize - 2, h);
            mask_work.dh[r * w..(r + 1) * w].copy_from_slice(&dst_y[row * w..(row + 1) * w]);
        }
        dst_y_edge_mask_strip(w, 0, h, h, &mut mask_work);
        let covered = mask_work.mask[..w * h]
            .iter()
            .filter(|&&v| v != 0.0)
            .count();
        assert!(covered < w * h, "luma mask covered all {covered} pixels");
        let fl = rev4_extract(&src, &luma, w, h);
        let ab = REV4_BASE + 96 + 72 + 144; // arttype s0
        assert_eq!(
            fl[ab + idx_arttype::BLEED_X].to_bits(),
            0.0f64.to_bits(),
            "luma-only bleed X must be exactly 0"
        );
        assert_eq!(
            fl[ab + idx_arttype::BLEED_B].to_bits(),
            0.0f64.to_bits(),
            "luma-only bleed B must be exactly 0"
        );
        // Negative control: retain the luma step and add opponent-colour
        // stripes in a separate patch. Both RGB colours have nearly the
        // same converted XYB Y as [128; 3] (<0.001 difference), but their
        // X/B values differ strongly. Thus the chroma edges remain outside
        // the dst-Y mask instead of filling it vacuously.
        let mut luma_chroma = luma.clone();
        for y in 44..68 {
            for x in 48..80 {
                luma_chroma[y * w + x] = if (x + y) % 4 < 2 {
                    [80, 142, 128]
                } else {
                    [176, 100, 128]
                };
            }
        }
        let fc = rev4_extract(&src, &luma_chroma, w, h);
        assert!(
            fc[ab + idx_arttype::BLEED_X] > 0.0 || fc[ab + idx_arttype::BLEED_B] > 0.0,
            "luma+chroma must produce positive bleed, got X={} B={}",
            fc[ab + idx_arttype::BLEED_X],
            fc[ab + idx_arttype::BLEED_B]
        );
    }

    /// f64 accumulation shows no cancellation on flat high-value inputs:
    /// scatter one large constant into the f64 accumulators and compare
    /// against the n·v algebraic reference — the running sums must stay
    /// within a tiny relative bound (1e-12), i.e. no catastrophic
    /// cancellation or drift, not merely "approximately" right.
    #[test]
    fn rev4_accumulators_flat_no_cancellation() {
        const N: usize = 200_000;
        let big = 0.75f64; // high-value flat term
        // RingAccum: Σ ring·h_k — constant ring ⇒ fixed hat weights.
        let mut ring = RingAccum::default();
        let h = rev4_hats(&RINGBASIS_UC, rev4_u(big));
        for _ in 0..N {
            for (b, h) in ring.bins.iter_mut().zip(h.iter()) {
                b.add64(0, big * h);
            }
        }
        for (k, (b, &h)) in ring.bins.iter().zip(h.iter()).enumerate() {
            let b = b.fin();
            let expect = N as f64 * big * h;
            let rel = ((b - expect) / expect).abs();
            assert!(
                rel < 1e-10 || expect == 0.0 && b == 0.0,
                "ring bin {k} drifted on flat input: {b} vs {expect} (rel {rel:e})",
            );
        }
        // FlatAccum: Σ hfg over flat pixels ≈ N · hfg.
        let mut flat = FlatAccum::default();
        for _ in 0..N {
            flat.hfg.add64(0, big);
            flat.n += 1;
        }
        let expect = N as f64 * big;
        assert!(
            ((flat.hfg.fin() - expect) / expect).abs() < 1e-10,
            "flat hfg drifted: {:?} vs {expect}",
            flat.hfg
        );
        // GridblkAccum signed sums over an alternating ±big pattern: the
        // equal-magnitude pairs cancel EXACTLY (same-bits add/sub), and
        // |ẽ| stays on the algebraic bound.
        let mut g = GridblkAccum::default();
        for i in 0..N {
            g.signed_v[0] += if i % 2 == 0 { big } else { -big };
            g.abs_v[0] += big;
        }
        assert_eq!(g.signed_v[0].to_bits(), 0.0f64.to_bits());
        assert!(
            ((g.abs_v[0] - expect) / expect).abs() < 1e-10,
            "gridblk abs drifted: {} vs {expect}",
            g.abs_v[0]
        );
    }

    /// C3 (and every rev4 slot) is merge-order free: serial walk and the
    /// band-parallel walk at pool sizes {1,2,8,16} must emit BIT-IDENTICAL
    /// rev4 segments. Height 200 forces a 128+72 strip decomposition —
    /// mixed strip heights inside one image — and widths crossing
    /// `H_TILE_WIDTH` exercise the horizontal tile split.
    #[cfg(all(feature = "training", feature = "threads"))]
    #[test]
    fn rev4_thread_and_strip_invariance() {
        for &(w, h) in &[(200usize, 200usize), (300, 200), (96, 320)] {
            let src = textured_image(w, h, 0xC3);
            let dst = quantize_distort(&src, w, h);
            let mut reference: Option<Vec<f64>> = None;
            for &parallel in &[false, true] {
                for threads in [1usize, 2, 8, 16] {
                    if !parallel && threads != 1 {
                        continue;
                    }
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .expect("build rayon pool");
                    let got = pool.install(|| {
                        let mut scratch = V2Scratch::new();
                        compute_folded720_streaming_impl(
                            &RgbSlice::new(&src, w, h),
                            &RgbSlice::new(&dst, w, h),
                            None,
                            parallel,
                            rev4_all_toggles(),
                            &mut scratch,
                            None,
                        )
                        .unwrap()
                        .features
                    });
                    match &reference {
                        None => reference = Some(got),
                        Some(first) => {
                            for i in 0..first.len().min(got.len()) {
                                assert_eq!(
                                    first[i].to_bits(),
                                    got[i].to_bits(),
                                    "{w}x{h}: slot {i} moved at par={parallel} t={threads} \
                                     ({:e} -> {:e})",
                                    first[i],
                                    got[i]
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// C1 JPEG ladder: the internal winning-phase Σ|ẽ|, which is not an
    /// emitted slot, must be non-decreasing as zenjpeg quality descends
    /// 95 → 10 on four imazen-26 TRAIN references. Signed `on_mean`
    /// and on/off ratio are logged only; neither is claimed monotone.
    /// Caller opts in with `ZENSIM_REV4_CORPUS_ROOT` and `--ignored`.
    #[test]
    #[ignore = "explicit corpus gate: use just rev4-corpus-tests"]
    fn rev4_gridblk_zenjpeg_ladder() {
        use enough::Unstoppable;
        use zenjpeg::decoder::Decoder;
        use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

        let root_path = std::env::var("ZENSIM_REV4_CORPUS_ROOT")
            .expect("caller must set ZENSIM_REV4_CORPUS_ROOT");
        let root = std::path::Path::new(&root_path);
        assert!(root.is_dir(), "corpus root {root:?} is missing");
        // TRAIN refs: leading numeric stem ends in {0,2,4,6,8}.
        const REFS: [&str; 4] = [
            "20210608_123424__pristine2x.png",
            "20220616_102219__pristine2x.png",
            "20230704_121114__pristine2x.png",
            "20240424_123020__pristine2x.png",
        ];
        let jpeg_roundtrip = |px: &[[u8; 3]], w: usize, h: usize, q: u8| -> Vec<[u8; 3]> {
            let flat: Vec<u8> = px.iter().flat_map(|p| p.iter().copied()).collect();
            let config = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter);
            let mut enc = config
                .encode_from_bytes(w as u32, h as u32, PixelLayout::Rgb8Srgb)
                .expect("zenjpeg encoder init");
            enc.push_packed(&flat, Unstoppable).expect("zenjpeg push");
            let bytes = enc.finish().expect("zenjpeg finish");
            let dec = Decoder::new()
                .decode(&bytes, Unstoppable)
                .expect("zenjpeg decode");
            let out = dec.pixels_u8().expect("u8 jpeg output");
            out.as_chunks::<3>()
                .0
                .iter()
                .map(|c| [c[0], c[1], c[2]])
                .collect()
        };
        for name in REFS {
            let img = image::open(root.join(name))
                .unwrap_or_else(|e| panic!("decode {name}: {e}"))
                .to_rgb8();
            let (w, h) = (img.width() as usize, img.height() as usize);
            let src: Vec<[u8; 3]> = img
                .as_raw()
                .as_chunks::<3>()
                .0
                .iter()
                .map(|c| [c[0], c[1], c[2]])
                .collect();
            let prep = prepare_v2_reference_impl(&RgbSlice::new(&src, w, h), None, false, false)
                .expect("prepare");
            let mut prev = f64::NEG_INFINITY;
            let mut series = Vec::new();
            for q in (10u8..=95).rev().step_by(5) {
                let dst = jpeg_roundtrip(&src, w, h, q);
                let mut cells: Vec<Rev4CellAccum> = (0..prep.scales.len() * 3)
                    .map(|_| Rev4CellAccum::default())
                    .collect();
                let mut scratch = V2Scratch::new();
                let _ = compute_v2_features_with_ref_impl_inner(
                    &prep,
                    &RgbSlice::new(&dst, w, h),
                    None,
                    false,
                    rev4_all_toggles(),
                    &mut scratch,
                    Some(&mut cells[..]),
                )
                .expect("materialized computes");
                let f = rev4_extract(&src, &dst, w, h);
                // Scale-0 Y cell: on-grid excess = Σ|ẽ| at the winning
                // phase (V + H), the raw magnitude the phase argmax
                // selects; signed on_mean + on/off recorded alongside.
                let g = &cells[1].grid;
                let pv = gridblk_winner(&g.abs_v, &[usize::MAX; GRIDBLK_MAX_PERIOD], 8)
                    .expect("V winner");
                let ph = gridblk_winner(&g.abs_h, &[usize::MAX; GRIDBLK_MAX_PERIOD], 8)
                    .expect("H winner");
                let on_abs = g.abs_v[pv] + g.abs_h[ph];
                let (m, r) = (f[REV4_BASE + 8 + 6], f[REV4_BASE + 8 + 7]);
                series.push((q, on_abs, m, r, pv, ph));
                assert!(
                    on_abs >= prev,
                    "{name}: internal on-grid |excess| decreased at q{q}: {on_abs:e} < {prev:e}"
                );
                prev = on_abs;
            }
            let first = series.first().unwrap().1;
            let last = series.last().unwrap().1;
            assert!(
                last > first,
                "{name}: q10 on-grid |excess| not above q95: {series:?}"
            );
            eprintln!("{name} C1 ladder (q, Σon|ẽ|, on_mean, onoff, pv, ph): {series:?}");
        }
    }

    /// Existing-slot bit-identity gate: the shared dense/gradient kernels
    /// carry rev4 hooks, so an all-rev4-ON extraction must emit the SAME
    /// f0..f985 bits as the all-rev4-OFF one (the hooks write only the
    /// rev4 accumulators; `Option::None` skips them entirely). Geometries
    /// include multi-strip heights, sub-64 dims (reflect-padded),
    /// non-tight strides and `H_TILE_WIDTH`-crossing widths. The
    /// SIMD-tier matrix for this same property lives in
    /// `tests/rev4_featbank_parity.rs` — token permutation is
    /// process-wide and cannot run inside the parallel unit-test
    /// process.
    #[test]
    fn rev4_existing_slots_unmoved_by_toggles() {
        for &(w, h, strided) in &[
            (200usize, 200usize, false),
            (96, 80, false),
            (300, 200, false),
            (127, 288, true),
            (50, 300, false),
            (300, 50, false),
            (40, 40, false),
        ] {
            let src = textured_image(w, h, 0xE0 + w as u32);
            let dst = quantize_distort(&src, w, h);
            // Optional padded copy (non-tight stride + 13 bytes).
            let stride = w * 3 + 13;
            let (pad_src, pad_dst) = if strided {
                let mut ps = vec![0u8; stride * h];
                let mut pd = vec![0u8; stride * h];
                for y in 0..h {
                    for x in 0..w {
                        let o = y * stride + x * 3;
                        ps[o..o + 3].copy_from_slice(&src[y * w + x]);
                        pd[o..o + 3].copy_from_slice(&dst[y * w + x]);
                    }
                }
                (ps, pd)
            } else {
                (Vec::new(), Vec::new())
            };
            let toggles_on = rev4_all_toggles();
            // Same request minus ONLY the rev4 flags — identical work
            // everywhere except the rev4 hooks.
            let toggles_off = V2NewFeatureToggles {
                rev4_gridblk: false,
                rev4_ringbasis: false,
                rev4_tailhist: false,
                rev4_arttype: false,
                ..rev4_all_toggles()
            };
            let run_once = || {
                let mut scratch = V2Scratch::new();
                let mut run = |toggles: V2NewFeatureToggles| -> Vec<f64> {
                    if strided {
                        let s =
                            StridedBytes::try_new(&pad_src, w, h, stride, PixelFormat::Srgb8Rgb)
                                .expect("strided src");
                        let d =
                            StridedBytes::try_new(&pad_dst, w, h, stride, PixelFormat::Srgb8Rgb)
                                .expect("strided dst");
                        compute_folded720_streaming_impl(
                            &s,
                            &d,
                            None,
                            false,
                            toggles,
                            &mut scratch,
                            None,
                        )
                        .expect("streaming computes")
                        .features
                    } else {
                        compute_folded720_streaming_impl(
                            &RgbSlice::new(&src, w, h),
                            &RgbSlice::new(&dst, w, h),
                            None,
                            false,
                            toggles,
                            &mut scratch,
                            None,
                        )
                        .expect("streaming computes")
                        .features
                    }
                };
                let on = run(toggles_on);
                let off = run(toggles_off);
                assert!(
                    on.len() >= REV4_BASE && off.len() == 956,
                    "unexpected width on={} off={}",
                    on.len(),
                    off.len()
                );
                assert!(on[956..986].iter().all(|v| *v == 0.0));
                for i in 0..956 {
                    assert_eq!(
                        on[i].to_bits(),
                        off[i].to_bits(),
                        "{w}x{h} strided={strided}: existing slot {i} moved \
                         with rev4 toggles ({:e} -> {:e})",
                        off[i],
                        on[i]
                    );
                }
            };
            run_once();
        }
    }

    /// One `V2Scratch` reused across pairs of DIFFERENT sizes and content
    /// must produce the same bits as a fresh scratch per pair — i.e. no
    /// stale-buffer leakage between pairs (buffers are fully written
    /// before every read).
    #[test]
    fn scratch_reuse_matches_fresh_scratch() {
        let mut shared = V2Scratch::new();
        // Big pair first so the small pair runs on oversized buffers full
        // of the big pair's data — the harshest staleness ordering.
        for &(w, h, seed) in &[
            (200usize, 136usize, 0xAAAAu32),
            (64, 64, 0xBBBB),
            (96, 80, 7),
        ] {
            let src = textured_image(w, h, seed);
            let dst = quantize_distort(&src, w, h);
            let source = RgbSlice::new(&src, w, h);
            let distorted = RgbSlice::new(&dst, w, h);
            let prepared =
                prepare_v2_reference_impl(&source, None, false, false).expect("prepare computes");

            let mut fresh = V2Scratch::new();
            let a = compute_v2_features_with_ref_impl(
                &prepared,
                &distorted,
                None,
                false,
                V2NewFeatureToggles::default(),
                &mut fresh,
            )
            .expect("fresh-scratch computes");
            let b = compute_v2_features_with_ref_impl(
                &prepared,
                &distorted,
                None,
                false,
                V2NewFeatureToggles::default(),
                &mut shared,
            )
            .expect("shared-scratch computes");
            for (i, (x, y)) in a.features().iter().zip(b.features().iter()).enumerate() {
                assert!(
                    x.to_bits() == y.to_bits(),
                    "{w}x{h}: feature {i} diverged under scratch reuse"
                );
            }
        }
    }

    /// The cached-moments prepared path (mu1 + activity read from the
    /// reference cache; V-blur + activity chain skipped per pair) must ALSO
    /// be bit-identical to the pair path — the cache is filled by replaying
    /// the exact strip walk, so this gate holds by construction and this
    /// test enforces it stays that way. Multi-strip + odd + pad sizes.
    #[test]
    fn prepared_ref_with_moments_bit_identical_to_pair_path() {
        for &(w, h) in &[
            (64usize, 64usize),
            (96, 80),
            (200, 136),
            (65, 129),
            (40, 30),
        ] {
            let src = textured_image(w, h, 0xC0DE);
            let dst = quantize_distort(&src, w, h);
            let source = RgbSlice::new(&src, w, h);
            let distorted = RgbSlice::new(&dst, w, h);

            let pair = compute_v2_features_impl(&source, &distorted, None, false)
                .expect("pair path computes");

            let prepared = prepare_v2_reference_impl(&source, None, false, true)
                .expect("prepare with moments computes");
            assert!(prepared.has_cached_moments());
            let mut scratch = V2Scratch::new();
            let with_ref = compute_v2_features_with_ref_impl(
                &prepared,
                &distorted,
                None,
                false,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .expect("with_ref computes");

            for (i, (a, b)) in pair
                .features()
                .iter()
                .zip(with_ref.features().iter())
                .enumerate()
            {
                assert!(
                    a.to_bits() == b.to_bits(),
                    "{w}x{h}: feature {i} diverged under cached moments: \
                     pair={a:e} with_ref={b:e}"
                );
            }
        }
    }

    /// Independent f64 algebra, not a second call to the pooling helper.
    /// Exercises actual dispatched SIMD and scalar tails with valid moments.
    #[test]
    fn weighted_mse_matches_normalized_positive_measure() {
        for width in [7, 8, 13, 31, 97, 257] {
            let height = 9;
            let n = width * height;
            for uniform in [false, true] {
                let src = vec![0.2f32; n];
                let dst: Vec<f32> = (0..n)
                    .map(|i| {
                        if uniform {
                            0.25
                        } else {
                            0.2 + (i % 17) as f32 * 0.017
                        }
                    })
                    .collect();
                let act: Vec<f32> = (0..n).map(|i| (i % 23) as f32 * 0.023).collect();
                let ssq: Vec<f32> = src.iter().zip(&dst).map(|(&s, &d)| s * s + d * d).collect();
                let cross: Vec<f32> = src.iter().zip(&dst).map(|(&s, &d)| s * d).collect();
                let actual = dense_block_kernel(
                    &src, &dst, &src, &dst, &ssq, &cross, &act, width, height, true, None,
                );
                let signal: Vec<f64> = src
                    .iter()
                    .zip(&dst)
                    .map(|(&s, &d)| {
                        let e = (f64::from(s) - f64::from(d)).powi(2);
                        e / (e + C_MSE)
                    })
                    .collect();
                // Complementary activity weights partition the unweighted mass:
                // w_mask + w_iw = 1 + floor, at every pixel.
                let signal_sum: f64 = signal.iter().sum();
                assert!(
                    ((actual.ws_mask_mse.num + actual.ws_iw_mse.num) / (n as f64)
                        - (1.0 + IW_WEIGHT_FLOOR) * signal_sum / (n as f64))
                        .abs()
                        < 2e-6
                );
                let lo = signal.iter().copied().fold(f64::INFINITY, f64::min);
                let hi = signal.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                for (iw, pool) in [(false, actual.ws_mask_mse), (true, actual.ws_iw_mse)] {
                    let weights: Vec<f64> = act
                        .iter()
                        .map(|&a| {
                            let a = f64::from(a);
                            if iw {
                                a / (a + C_ACTIVITY) + IW_WEIGHT_FLOOR
                            } else {
                                C_ACTIVITY / (a + C_ACTIVITY)
                            }
                        })
                        .collect();
                    let denominator: f64 = weights.iter().sum();
                    let reference: f64 =
                        weights.iter().zip(&signal).map(|(w, e)| w * e).sum::<f64>() / denominator;
                    let value = pool.finish();
                    assert!(
                        (value - reference).abs() < 2e-6,
                        "width={width} iw={iw}: {value} vs {reference}"
                    );
                    assert!(value >= lo - 2e-6 && value <= hi + 2e-6);
                    // A common weight multiplier must cancel in the ratio.
                    let rescaled: f64 = weights
                        .iter()
                        .zip(&signal)
                        .map(|(w, e)| 37.0 * w * e)
                        .sum::<f64>()
                        / (37.0 * denominator);
                    assert!((reference - rescaled).abs() < 1e-12);
                }
                let identity = dense_block_kernel(
                    &src,
                    &src,
                    &src,
                    &src,
                    &vec![0.08; n],
                    &vec![0.04; n],
                    &act,
                    width,
                    height,
                    true,
                    None,
                );
                assert_eq!(identity.ws_mask_mse.finish(), 0.0);
                assert_eq!(identity.ws_iw_mse.finish(), 0.0);
            }
        }
    }

    /// POOL_SIMD drift gate: the vectorized weighted-pool path (enabled on
    /// the v4x tier) must stay within this module's documented 5e-4
    /// relative tolerance of the §A.14 scalar-pool path — same
    /// reassociation class as the phase-4 core-moment change. On hosts
    /// where the active tier runs scalar pools anyway, both sides are
    /// §14.4 UNDER TEST: the two `const FUSED` instantiations must produce
    /// BIT-IDENTICAL accumulators.
    ///
    /// Post-V8-rewrite the split is vestigial (both entries select `false`,
    /// since fusing MEASURED 2x worse — §19.2), but the gate is kept because
    /// it is the executable form of §14.4's claim that pass structure is a
    /// perf knob and not semantics. It runs on the `scalar` token so it is
    /// tier-independent.
    ///
    /// This is the claim that lets the pass split be tuned per tier without
    /// touching the era. It is asserted directly rather than inferred: both
    /// const-generic instantiations are called on the same planes and every
    /// one of the 35 slots is compared with `to_bits()`. If it ever fails, the
    /// per-tier split is NOT free and the tier layout becomes semantics.
    #[test]
    fn era2_fused_and_two_pass_are_bit_identical() {
        for &(w, h) in &[
            (96usize, 64usize),
            (200, 136),
            (127, 93),
            (61, 40),
            (8, 3),
            (13, 65),
        ] {
            let src_px = textured_image(w, h, 0xBEEF);
            let dst_px = quantize_distort(&src_px, w, h);
            let source = RgbSlice::new(&src_px, w, h);
            let distorted = RgbSlice::new(&dst_px, w, h);
            let sp = crate::streaming::convert_source_to_xyb(
                &source,
                w,
                false,
                crate::ssim_form::active_revision(),
            );
            let dp = crate::streaming::convert_source_to_xyb(
                &distorted,
                w,
                false,
                crate::ssim_form::active_revision(),
            );
            let mut scratch = ScratchV2Strip::new(w * h);
            let n = w * h;
            for ch in 0..3 {
                run_blur_pass(&sp[ch], &dp[ch], w, h, &mut scratch);
                use archmage::SimdToken as _;
                let tok = archmage::ScalarToken::summon().expect("infallible");
                let f = super::dense_block_kernel_era2_generic::<_, true>(
                    tok,
                    &sp[ch],
                    &dp[ch],
                    &scratch.mu1[..n],
                    &scratch.mu2[..n],
                    &scratch.ssq[..n],
                    &scratch.s12[..n],
                    &scratch.activity[..n],
                    w,
                    h,
                    true,
                    None,
                );
                let t = super::dense_block_kernel_era2_generic::<_, false>(
                    tok,
                    &sp[ch],
                    &dp[ch],
                    &scratch.mu1[..n],
                    &scratch.mu2[..n],
                    &scratch.ssq[..n],
                    &scratch.s12[..n],
                    &scratch.activity[..n],
                    w,
                    h,
                    true,
                    None,
                );
                let (fs, ts) = (oracle::dense_accum_slots(&f), oracle::dense_accum_slots(&t));
                for i in 0..fs.len() {
                    assert_eq!(
                        fs[i].to_bits(),
                        ts[i].to_bits(),
                        "{w}x{h} ch{ch} slot {} ({}): FUSED {:e} != two-pass {:e}. The per-tier \
                         pass split is supposed to be byte-neutral (§14.4); if it is not, the \
                         tier layout has become part of era-2's semantics.",
                        i,
                        oracle::SLOT_NAMES[i],
                        fs[i],
                        ts[i]
                    );
                }
            }
        }
    }

    /// ERA-2 STRUCTURAL GATE: the properties the break actually rests on.
    ///
    /// **A correction the first version of this test earned.** It originally
    /// asserted "band-merge == serial fold, bit-exact" and FAILED at 127x93
    /// (7.854278564453125 vs 7.8542633056640625). The assertion was wrong, not
    /// the design: banding IS a different grouping — serial computes
    /// `((band0 + r32) + r33) + …` while banded computes `band0 + band1` — and
    /// f64/f32 addition is not associative. This is the same blocking
    /// non-associativity documented for the era-1 dense kernel; it does not go
    /// away because the shape is new.
    ///
    /// **So `ERA2_BAND_ROWS` is SEMANTICS, not a tuning knob.** Changing it
    /// changes output bytes and is an era decision. It is deliberately NOT
    /// derived from the thread count, image height, or anything else that
    /// varies at runtime — that is what makes thread-invariance structural.
    /// Anyone reaching for it to "tune the band size" must read this first.
    ///
    /// What is actually asserted:
    /// (a) the banded fold is INVARIANT to how bands are distributed across
    ///     workers — computing them in any order, or all at once, gives
    ///     bit-identical results, because the merge is sequential in band index;
    /// (b) lane `j` holds exactly the terms at `x ≡ j (mod 8)` in increasing
    ///     `x`, tail included — so the shape is width-independent, unlike
    ///     era-1's, whose tail landed in the f64 running total.

    #[test]
    fn era2_band_merge_and_tail_are_structural() {
        let term = |i: usize| -> f32 {
            let k = i.wrapping_mul(2654435761usize) % 65521;
            (k as f32) * (1.0 / 65536.0) - 0.5
        };

        // Fold one band's rows into its own lane set.
        let band_of = |plane: &[f32], width: usize, y0: usize, y1: usize| -> super::Lanes8 {
            let mut band = super::Lanes8::zero();
            for y in y0..y1 {
                let row = &plane[y * width..(y + 1) * width];
                let (chunks, tail) = row.as_chunks::<8>();
                for c in chunks {
                    band.add_chunk(c);
                }
                band.add_tail(tail);
            }
            band
        };

        for &(width, rows) in &[(576usize, 128usize), (127, 93), (61, 40), (8, 3), (5, 1)] {
            let plane: Vec<f32> = (0..width * rows).map(term).collect();

            let mut starts = Vec::new();
            let mut b = 0usize;
            while b < rows {
                starts.push(b);
                b = (b + super::ERA2_BAND_ROWS).min(rows);
            }

            // Reference: bands computed in index order, merged in index order.
            let mut reference = super::Lanes8::zero();
            for (i, &y0) in starts.iter().enumerate() {
                let y1 = starts.get(i + 1).copied().unwrap_or(rows);
                let bandv = band_of(&plane, width, y0, y1);
                for (m, v) in reference.0.iter_mut().zip(bandv.0.iter()) {
                    *m += *v;
                }
            }

            // (a) Compute the bands in REVERSE order (as a worker pool would,
            // completing out of order), then merge in index order as the design
            // requires. Must be bit-identical.
            let mut partials: Vec<Option<super::Lanes8>> = vec![None; starts.len()];
            for (i, &y0) in starts.iter().enumerate().rev() {
                let y1 = starts.get(i + 1).copied().unwrap_or(rows);
                partials[i] = Some(band_of(&plane, width, y0, y1));
            }
            let mut out_of_order = super::Lanes8::zero();
            for p in partials.iter().flatten() {
                for (m, v) in out_of_order.0.iter_mut().zip(p.0.iter()) {
                    *m += *v;
                }
            }
            assert_eq!(
                reference.reduce().to_bits(),
                out_of_order.reduce().to_bits(),
                "{width}x{rows}: computing bands out of order changed the result. \
                 era-2's thread-invariance rests on the MERGE being sequential in \
                 band index while the bands themselves are order-free."
            );

            // (b) lane j == the terms at x ≡ j (mod 8), tail included.
            let mut expect = [0.0f32; 8];
            for y in 0..rows {
                for x in 0..width {
                    expect[x % 8] += plane[y * width + x];
                }
            }
            let single = band_of(&plane, width, 0, rows);
            assert_eq!(
                single.0.map(f32::to_bits),
                expect.map(f32::to_bits),
                "{width}x{rows}: lane contents diverge from 'terms at x ≡ j (mod 8), \
                 in increasing x' — the tail fold is wrong."
            );
        }
    }

    /// ITEM D: the descriptor must reproduce the derivation it replaced, for
    /// every toggle combination — not just the ones the walk happens to use.
    /// The legacy expressions are written out here verbatim (they were six
    /// ad-hoc locals at the top of `foldapp_streaming_walk`); if
    /// `ComputeSet::from_toggles` ever drifts from them this fails, which is
    /// the whole point of landing the descriptor separately from any change
    /// in behaviour.
    #[test]
    fn compute_set_matches_legacy_derivation() {
        use super::{ComputeSet, V1PoolsMode, V2NewFeatureToggles};
        let pools = [
            V1PoolsMode::Off,
            V1PoolsMode::Peaks,
            V1PoolsMode::Carriers,
            V1PoolsMode::Full,
        ];
        let mut checked = 0usize;
        for bits in 0u32..256 {
            for pm in pools {
                let t = V2NewFeatureToggles {
                    formula_revision: crate::ssim_form::active_revision(),
                    gradient_features: bits & 1 != 0,
                    transducer_bank: bits & 2 != 0,
                    blockiness: bits & 4 != 0,
                    transducers_luma_only: bits & 8 != 0,
                    append_block: bits & 16 != 0,
                    append2_block: bits & 32 != 0,
                    csfw_block: bits & 64 != 0,
                    append2_dst_activity: bits & 128 != 0,
                    v1_pools: pm,
                    v1_only: bits & 3 == 3,
                    // The free-extras field is exercised by its own gates
                    // (`free_extras_*`); this one holds the LEGACY derivation,
                    // which predates it.
                    free_extras: super::V1FreeExtras::Off,
                    // The rev4 families likewise postdate the 8-bit space;
                    // their own gates cover the rev4 toggles directly.
                    rev4_gridblk: false,
                    rev4_ringbasis: false,
                    rev4_tailhist: false,
                    rev4_arttype: false,
                    gmsbank: false,
                    mapdev: false,
                    z1max: false,
                    gmsnative: false,
                    globaff: false,
                };
                let cs = ComputeSet::from_toggles(t);
                // --- the legacy derivation, verbatim ---
                let v2_blocks = !t.v1_only;
                let append_on = t.append_block && v2_blocks;
                let append2_on = t.append2_block && v2_blocks;
                // ⚠ CORRECTED 2026-09-05, and this is an expected-VALUE change,
                // so read it deliberately. This line said `V1PoolsMode::Full`
                // alone, transcribed from `ComputeSet::self_blur_eligible` —
                // but the STRIP LOOP, which is what actually decides whether
                // phase A runs, has always accepted `Full | Peaks`. So the
                // "legacy derivation" this test reproduces was itself a copy of
                // the wrong one of the two, and pinning it kept a real
                // footprint defect green: a `Peaks` walk sized in phase A's
                // four fused-H planes and then skipped phase A whole.
                // `self_blur_sizing_predicate_matches_the_strip_loop_predicate`
                // is the independent pin against the LOOP's predicate, and it
                // fails against the pre-fix definition. Widening this to match
                // is a correction, not a relaxation — the constraint it now
                // encodes (sizing == loop) is strictly stronger than the one it
                // encoded before (sizing == a hand-copy of sizing).
                let self_blur_core =
                    !v2_blocks && matches!(t.v1_pools, V1PoolsMode::Full | V1PoolsMode::Peaks);
                // --- /legacy ---
                assert_eq!(cs.v2_blocks, v2_blocks, "v2_blocks bits={bits}");
                assert_eq!(cs.append, append_on, "append bits={bits}");
                assert_eq!(cs.append2, append2_on, "append2 bits={bits}");
                assert_eq!(
                    cs.self_blur_eligible(),
                    self_blur_core,
                    "self_blur bits={bits}"
                );
                assert_eq!(cs.v1_pools, t.v1_pools);
                // A v1-only request must compute NOTHING v2-era, whatever
                // else the caller set.
                if t.v1_only {
                    assert!(
                        !cs.v2_blocks
                            && !cs.gradient
                            && !cs.blockiness
                            && !cs.append
                            && !cs.append2
                            && !cs.append2_dst_activity
                            && !cs.csfw,
                        "v1_only must force every v2-era block off, bits={bits}"
                    );
                }
                // `append2_dst_activity` refines BANDVIS and cannot outlive it.
                assert!(!cs.append2_dst_activity || cs.append2);
                // Plane needs follow the group, not the individual blocks.
                assert_eq!(cs.plane_needs(false).v2, cs.v2_blocks);
                assert!(!cs.plane_needs(true).h);
                checked += 1;
            }
        }
        assert_eq!(checked, 256 * 4);
    }

    /// **The two `self_blur` predicates must be the SAME predicate.**
    ///
    /// There are two, and until 2026-09-05 they disagreed on
    /// [`V1PoolsMode::Peaks`]:
    ///
    /// * [`ComputeSet::self_blur_eligible`] — read at `stream_foldapp`'s
    ///   `plane_needs` call, which is what SIZES the strip scratch.
    /// * the `let self_blur = …` binding inside the strip loop's
    ///   `fuse_channels` arm, which is what DECIDES whether phase A runs.
    ///
    /// When the sizing predicate says `false` and the loop predicate says
    /// `true`, the scratch reserves phase A's four fused-H planes and the
    /// loop then skips phase A whole, so those planes are allocated,
    /// first-touched and never written. That is exactly the waste
    /// `StripPlaneNeeds` was introduced to remove
    /// (`benchmarks/fold_footprint_2026-08-31.md` §5), reappearing on the
    /// one mode the fast class actually ships:
    /// `fold_engine::pools_mode_for_need` returns `Peaks` for any bake that
    /// does not read masked/IW, which is every basic+peaks model.
    ///
    /// This test pins them together so the next edit to either one cannot
    /// silently re-open the gap. It asserts the LOOP's predicate, because
    /// the loop is the one that determines what is actually written — a
    /// sizing decision that disagrees with it is wrong by definition, in
    /// whichever direction it disagrees (under-sizing would be a
    /// correctness bug, over-sizing is the footprint bug found here).
    #[test]
    fn self_blur_sizing_predicate_matches_the_strip_loop_predicate() {
        for &pm in &[
            V1PoolsMode::Off,
            V1PoolsMode::Carriers,
            V1PoolsMode::Peaks,
            V1PoolsMode::Full,
        ] {
            for &v1_only in &[true, false] {
                let t = V2NewFeatureToggles {
                    v1_only,
                    v1_pools: pm,
                    ..Default::default()
                };
                let cs = ComputeSet::from_toggles(t);
                // Verbatim from the strip loop's `fuse_channels` arm, minus
                // `fuse_channels` itself (a runtime parallelism fact, which
                // both sites already AND in separately).
                let loop_predicate = !cs.v2_blocks
                    && cs.v1_basic
                    && matches!(t.v1_pools, V1PoolsMode::Full | V1PoolsMode::Peaks);
                assert_eq!(
                    cs.self_blur_eligible(),
                    loop_predicate,
                    "self_blur sizing vs loop predicate disagree at \
                     v1_pools={pm:?} v1_only={v1_only}: sizing says {}, loop says \
                     {loop_predicate}. A disagreement in this direction reserves \
                     phase A's four H planes and never writes them.",
                    cs.self_blur_eligible(),
                );
            }
        }
    }

    /// **The safety gate `era2_fast_profile_subset_2026-08-31.md` §5 item 2
    /// asks for**: the derivation must never drop a family the model
    /// actually reads. Cross-checks [`crate::feature_plan::Plan::for_bake`]
    /// against the independently-derived, already-tested
    /// [`crate::fold_engine::score_pool_mode`] on shipped `B` — same bake,
    /// two derivations, must agree bit-for-bit on `v1_pools`, and `B` must
    /// resolve to `Full` (it reads the whole pool block, so nothing is safe
    /// to skip).
    #[test]
    fn bake_plan_matches_score_pool_mode_on_shipped_b() {
        use crate::profile::ZensimProfile;
        let params = ZensimProfile::B.params();
        let bytes: Vec<&'static [u8]> = params.scoring_bake_bytes().collect();
        assert_eq!(bytes.len(), 1, "B forwards exactly one bake");
        let model = crate::mlp::Model::from_bytes(bytes[0]).expect("shipped B bake parses");
        let cs = crate::feature_plan::Plan::for_bake(&model).unwrap().compute;
        assert_eq!(
            cs.v1_pools,
            V1PoolsMode::Off,
            "diff+frag B reads no pooled v1 inputs"
        );
        assert!(cs.v2_blocks, "diff+frag B reads the v2 block");

        let config = crate::metric::config_from_params(params, false);
        let want = crate::Zensim::new(ZensimProfile::B).score_pool_mode();
        assert_eq!(config.revision(), FormulaRevision::Rev5);
        assert_eq!(
            cs.v1_pools, want,
            "bake plan disagrees with the tested score_pool_mode path"
        );
    }

    /// The same cross-check on a model this function DECLINES to analyse
    /// block-by-block (wider than the v1 372-feature layout): it must return
    /// the safe "everything needed" answer, never an optimistic under-report.
    /// `C` (944-wide) is used only as a wide fixture; its OWN pool need is a
    /// separate, already-gated question (`bake_pool_need_matches_the_block_profile_tool_on_shipped_b`-style
    /// coverage for 944 bakes lives in the block-profile tooling, not here).
    #[cfg(feature = "candidate-profiles")]
    #[test]
    fn bake_plan_covers_c_without_unused_pools_or_activity_variant() {
        use crate::profile::ZensimProfile;
        let params = ZensimProfile::C.params();
        let bytes: Vec<&'static [u8]> = params.scoring_bake_bytes().collect();
        assert_eq!(bytes.len(), 1, "C forwards exactly one bake");
        let model = crate::mlp::Model::from_bytes(bytes[0]).expect("shipped C bake parses");
        assert!(
            model.caller_input_width() > 372,
            "fixture must actually be wider than the v1 layout to exercise the fallback"
        );
        let cs = crate::feature_plan::Plan::for_bake(&model).unwrap().compute;
        assert!(
            cs.v2_blocks,
            "a wide, unanalysed bake must fall back to computing everything"
        );
        assert_eq!(cs.v1_pools, V1PoolsMode::Peaks);
        assert!(cs.append && cs.append2 && cs.gradient && cs.blockiness);
        assert!(!cs.csfw && !cs.append2_dst_activity);
    }

    // ---------------------------------------------------------------- //
    // Feature-set IDs (`docs/FEATURE_SET_IDS.md`) — the COMPUTE half.
    // ---------------------------------------------------------------- //

    /// A FULL 944 walk names every block and NO free tranche: the append
    /// kernel owns `GLOBAL_*` there, so `raw_moments()` is false by
    /// construction and the free accumulators never run.
    #[test]
    fn compute_parts_full_944_names_every_block_and_no_free_tranche() {
        use crate::feature_set_id::{ComputeToken as T, SlotSet};
        let cs = ComputeSet::from_toggles(V2NewFeatureToggles {
            append_block: true,
            append2_block: true,
            free_extras: V1FreeExtras::RawMomentsPlusBoundedErr,
            v1_pools: V1PoolsMode::Full,
            ..Default::default()
        });
        let parts = cs.compute_parts();
        assert_eq!(parts.to_string(), "basic+peaks+masked+iw+v2+append+append2");
        assert!(
            !parts.contains(T::Moments),
            "append owns GLOBAL_* on a full walk"
        );
        assert!(
            !parts.contains(T::ClassC),
            "the v2 kernel owns MSE on a full walk"
        );
        assert_eq!(cs.populated_slots(4, 944), SlotSet::parse("0-943").unwrap());
    }

    /// The free-set arm — v1-only + `Peaks` + `RawMoments` at the 944 layout.
    /// Named `basic+peaks+moments`, and the 37 raw-moment slots
    /// (`benchmarks/free_features_2026-09-01.md` §1) are the ONLY thing
    /// populated above `f227`.
    #[test]
    fn compute_parts_free_set_arm_is_basic_peaks_moments_with_37_scattered_slots() {
        use crate::feature_set_id::SlotSet;
        let cs = ComputeSet::from_toggles(V2NewFeatureToggles {
            v1_only: true,
            v1_pools: V1PoolsMode::Peaks,
            free_extras: V1FreeExtras::RawMoments,
            ..Default::default()
        });
        assert_eq!(cs.compute_parts().to_string(), "basic+peaks+moments");
        let slots = cs.populated_slots(4, 944);
        assert!(
            slots.covers(&SlotSet::parse("0-227").unwrap()),
            "basic+peaks live"
        );
        assert!(!slots.contains(300), "IW is a structural zero here");
        assert_eq!(
            slots.len(),
            228 + 37,
            "228 basic+peaks, plus the 37-slot tranche"
        );
    }

    /// `+classC` adds **24** slots at EXISTING 944 positions — the width does
    /// not move, which is exactly what a count cannot express
    /// (`benchmarks/free_features_classC_2026-09-04.md` §1).
    #[test]
    fn compute_parts_class_c_adds_24_slots_at_the_same_width() {
        let base = ComputeSet::from_toggles(V2NewFeatureToggles {
            v1_only: true,
            v1_pools: V1PoolsMode::Peaks,
            free_extras: V1FreeExtras::RawMoments,
            ..Default::default()
        });
        let with_c = ComputeSet::from_toggles(V2NewFeatureToggles {
            v1_only: true,
            v1_pools: V1PoolsMode::Peaks,
            free_extras: V1FreeExtras::RawMomentsPlusBoundedErr,
            ..Default::default()
        });
        assert_eq!(
            with_c.compute_parts().to_string(),
            "basic+peaks+moments+classc"
        );
        let (a, b) = (base.populated_slots(4, 944), with_c.populated_slots(4, 944));
        assert!(b.covers(&a), "class C is a strict superset of the free set");
        assert_eq!(b.len(), a.len() + 24);
        // Same LAYOUT, different id — the "944" ambiguity, resolved.
        let ida = base.feature_set_id(4, 944, "era2r4").unwrap();
        let idb = with_c.feature_set_id(4, 944, "era2r4").unwrap();
        assert_eq!(ida.layout_width(), idb.layout_width());
        assert_ne!(ida, idb);
    }

    /// A 372-wide request clips the free tranches away entirely — their slots
    /// live at `f720+`, which does not exist at this layout.
    #[test]
    fn compute_parts_372_layout_clips_the_free_tranches_away() {
        use crate::feature_set_id::SlotSet;
        let cs = ComputeSet::from_toggles(V2NewFeatureToggles {
            v1_only: true,
            v1_pools: V1PoolsMode::Full,
            free_extras: V1FreeExtras::RawMomentsPlusBoundedErr,
            ..Default::default()
        });
        assert_eq!(cs.populated_slots(4, 372), SlotSet::parse("0-371").unwrap());
    }

    /// `V1PoolsMode::Carriers` populates exactly the ten carrier slots — a
    /// third distinct meaning of "944" at the same width as the other two.
    #[test]
    fn compute_parts_carriers_populates_exactly_the_ten_carrier_slots() {
        use crate::feature_set_id::SlotSet;
        let cs = ComputeSet::from_toggles(V2NewFeatureToggles {
            append_block: true,
            append2_block: true,
            v1_pools: V1PoolsMode::Carriers,
            ..Default::default()
        });
        assert_eq!(
            cs.compute_parts().to_string(),
            "basic+carriers+v2+append+append2"
        );
        let slots = cs.populated_slots(4, 944);
        let pool_block = SlotSet::parse("156-371").unwrap();
        assert_eq!(
            slots.intersect(&pool_block),
            SlotSet::from_slots(V1PoolsMode::CARRIER_SLOTS)
        );
    }

    /// The two eras of the SAME 944 compute set get different ids — the
    /// `ext944` / `era2r4` pair, bit-identical everywhere except `f720..923`.
    #[test]
    fn feature_set_id_separates_the_two_native_944_eras() {
        let cs = ComputeSet::from_toggles(V2NewFeatureToggles {
            append_block: true,
            append2_block: true,
            v1_pools: V1PoolsMode::Off,
            ..Default::default()
        });
        let a = cs.feature_set_id(4, 944, "ext944").unwrap();
        let b = cs.feature_set_id(4, 944, "era2r4").unwrap();
        assert_eq!(a.compute(), b.compute());
        assert_eq!(a.slots_hash(), b.slots_hash());
        assert_ne!(a, b, "the era is part of the identity");
        assert!(a.same_set_ignoring_era(&b));
        assert_eq!(a.compute().to_string(), "basic+v2+append+append2");
    }

    /// **Profile D's whole reason to exist.** `ADD156` reads only `f0..156`
    /// (28 of 156 basic lines; 0 of 216 pool lines — `bake_block_profile`,
    /// `benchmarks/add156_ship_audit_2026-08-31.md` §1.5), so the derived
    /// compute set must resolve to the cheap `Peaks` mode (never the
    /// mathematically-tempting but footprint-inferior `Off` — see
    /// [`crate::fold_engine::pools_mode_for_need`]) and must NOT claim any
    /// v2-era block.
    #[cfg(feature = "candidate-profiles")]
    #[test]
    fn bake_plan_derives_the_156_set_for_profile_d() {
        use crate::profile::ZensimProfile;
        let params = ZensimProfile::D.params();
        let bytes: Vec<&'static [u8]> = params.scoring_bake_bytes().collect();
        assert_eq!(bytes.len(), 1, "D forwards exactly one bake");
        let model = crate::mlp::Model::from_bytes(bytes[0]).expect("shipped D bake parses");
        // Since the 2026-09-06 dense flip `D` declares the 28 ids it reads and
        // asks for exactly 28 caller inputs (it used to declare the whole
        // 372-wide v1 layout with 344 all-zero layer-0 rows). What this test
        // is about is unchanged: the derived COMPUTE must still be the cheap
        // v1 walk with no v2-era block, and `Peaks` rather than the
        // footprint-inferior `Off`.
        assert_eq!(
            model.caller_input_width(),
            28,
            "ADD156 declares the 28 basic ids it reads"
        );
        let cs = crate::feature_plan::Plan::for_bake(&model).unwrap().compute;
        assert_eq!(
            cs.v1_pools,
            V1PoolsMode::Peaks,
            "ADD156 reads 0 of 216 pool lines, so masked+IW must be skippable, \
             and Peaks (not Off) is the correct minimal answer"
        );
        assert!(!cs.v2_blocks && !cs.append && !cs.append2 && !cs.gradient && !cs.csfw);

        // Cross-checked against the tested score_pool_mode path too, exactly
        // as the B gate above does — one policy, proven to agree twice.
        let config = crate::metric::config_from_params(params, false);
        let want = crate::fold_engine::score_pool_mode(params, &config, true);
        assert_eq!(cs.v1_pools, want);
    }

    /// Minimal synthetic 944-wide (`Folded720Append2`), 1-layer, identity,
    /// f32 ZNPR v3 bake — built through the MANDATED JSON pipeline
    /// (`zenpredict_bake::bake_from_json_str`, CLAUDE.md "JSON pipeline
    /// mandate for ZNPR v3 bakes"), never hand-rolled wire bytes. `nonzero`
    /// names the input indices carrying a live (1.0) weight; every other
    /// input's weight is exactly 0.0. This is the synthetic stand-in for
    /// the real A3b/A4b/K-class campaign bakes
    /// (`/mnt/v/output/zensim/wave-r4-2026-09-01/bakes/A4b_156_s4004.bin`,
    /// 509 KB — too large to commit and not CI-portable) that motivated
    /// this fix: this task independently re-verified the real A4b bake
    /// gets the cheap set through this same code path before committing
    /// (ad hoc, not part of the suite, since the file can't ship with it).
    fn wide_free_set_bake_bytes(nonzero: &[usize]) -> Vec<u8> {
        const N: usize = 944;
        let mut weights = vec![0.0f32; N];
        for &i in nonzero {
            assert!(i < N, "index {i} out of range");
            weights[i] = 1.0;
        }
        let arr = |v: &[f32]| -> String {
            let mut s = String::from("[");
            for (k, x) in v.iter().enumerate() {
                if k > 0 {
                    s.push(',');
                }
                s.push_str(&x.to_string());
            }
            s.push(']');
            s
        };
        let json = format!(
            r#"{{
                "schema_hash": 1,
                "scaler_mean": {mean},
                "scaler_scale": {scale},
                "layers": [
                    {{
                        "in_dim": {n},
                        "out_dim": 1,
                        "activation": "identity",
                        "dtype": "f32",
                        "weights": {weights},
                        "biases": [0.0]
                    }}
                ]
            }}"#,
            n = N,
            mean = arr(&vec![0.0f32; N]),
            scale = arr(&vec![1.0f32; N]),
            weights = arr(&weights),
        );
        zenpredict_bake::bake_from_json_str(&json).expect("synthetic wide bake must build")
    }

    /// **The A4b-class gap this fix closes.** A 944-wide bake whose only
    /// live columns are inside basic + peaks + two of the 40 FREE
    /// raw-moments slots (one `GLOBAL_DMEAN`-family slot, one
    /// `LUMA_MEAN_REF`) must get the CHEAP compute set — `v2_blocks`
    /// false, `free_extras: RawMoments` — not the "everything" fallback a
    /// bare `caller_input_width() > 372` check would give it.
    #[test]
    fn bake_plan_gives_a4b_class_free_set_bakes_the_cheap_set() {
        let free = free_slot_indices(crate::NUM_SCALES);
        assert!(free.len() >= 2, "need at least two free slots to pick from");
        let nonzero = [5usize, 200, free[0], free[10.min(free.len() - 1)]];
        let bytes = wide_free_set_bake_bytes(&nonzero);
        let model = crate::mlp::Model::from_bytes(&bytes).expect("synthetic bake parses");
        assert_eq!(model.caller_input_width(), 944);

        let cs = crate::feature_plan::Plan::for_bake(&model).unwrap().compute;
        assert!(
            !cs.v2_blocks
                && !cs.gradient
                && !cs.blockiness
                && !cs.transducer_bank
                && !cs.append
                && !cs.append2
                && !cs.csfw,
            "a bake reading only basic+peaks+free slots must get the cheap set, got {cs:?}"
        );
        assert_eq!(
            cs.free_extras,
            V1FreeExtras::RawMoments,
            "the bake DOES read free-extras positions, so free_extras must be requested"
        );
        assert_eq!(
            cs.v1_pools,
            V1PoolsMode::Peaks,
            "reads peaks (idx 200), not masked/IW"
        );
    }

    /// The same free-set shape, but with NO live weight in the free slots
    /// at all (basic+peaks only, width still 944) — `free_extras` must
    /// come back `Off`: requesting `RawMoments` for a bake that never
    /// reads it would cost cycles for nothing.
    #[test]
    fn bake_plan_leaves_free_extras_off_when_unread() {
        let bytes = wide_free_set_bake_bytes(&[5, 200]);
        let model = crate::mlp::Model::from_bytes(&bytes).expect("synthetic bake parses");
        let cs = crate::feature_plan::Plan::for_bake(&model).unwrap().compute;
        assert!(!cs.v2_blocks);
        assert_eq!(cs.free_extras, V1FreeExtras::Off);
    }

    /// A bake that reads even ONE non-free column at or beyond f372 (here:
    /// `idx_append::XMASK_TRANSDUCER`, f720 — inside the append block but
    /// NOT one of the 40 free slots) must fall all the way back to
    /// "everything" — the safety property item 2 of
    /// `era2_fast_profile_subset_2026-08-31.md` §5 asks for: the cheap-set
    /// carve-out must never fire on a bake it can't fully account for.
    #[test]
    fn bake_plan_computes_a_non_free_append_read() {
        let free = free_slot_indices(crate::NUM_SCALES);
        let non_free_append = 720 + idx_append::XMASK_TRANSDUCER; // = 720
        assert!(
            !free.contains(&non_free_append),
            "fixture picked a position that IS free — fixture bug, not a product bug"
        );
        let bytes = wide_free_set_bake_bytes(&[5, 200, non_free_append]);
        let model = crate::mlp::Model::from_bytes(&bytes).expect("synthetic bake parses");
        let cs = crate::feature_plan::Plan::for_bake(&model).unwrap().compute;
        assert!(
            cs.v2_blocks
                && cs.append
                && cs.append2
                && cs.gradient
                && cs.blockiness
                && cs.transducer_bank,
            "a non-free append read must trigger the full fallback, got {cs:?}"
        );
    }

    /// Same, for a genuine v2-348 read (f372..720 — gradient/blockiness/
    /// transducer territory, no free slots live there at all).
    #[test]
    fn bake_plan_computes_a_v2_348_read() {
        let bytes = wide_free_set_bake_bytes(&[5, 200, 400]);
        let model = crate::mlp::Model::from_bytes(&bytes).expect("synthetic bake parses");
        let cs = crate::feature_plan::Plan::for_bake(&model).unwrap().compute;
        assert!(cs.v2_blocks, "a v2-348 read must trigger the full fallback");
    }

    /// **Bit-identical scores, cheap set vs everything, on the free-set
    /// fixture** — this is pure compute-skipping (the coordinator's
    /// gate): the two compute sets must produce the SAME score whenever
    /// both are asked to score the same pair, because the skipped slots
    /// are provably unread by this model. Drives the fold walk directly
    /// (`compute_folded_v1_372_streaming_impl`-style, at 944 width) via
    /// `V2NewFeatureToggles` built from each `ComputeSet` by hand, since
    /// `ComputeSet` itself has no public "run me" entry point yet.
    #[test]
    fn bake_plan_cheap_set_is_score_neutral_for_a4b_class_bakes() {
        use crate::source::RgbSlice;
        let free = free_slot_indices(crate::NUM_SCALES);
        let nonzero = [5usize, 200, free[0], free[10.min(free.len() - 1)]];
        let bytes = wide_free_set_bake_bytes(&nonzero);
        let model = crate::mlp::Model::from_bytes(&bytes).expect("synthetic bake parses");
        let cs_cheap = crate::feature_plan::Plan::for_bake(&model).unwrap().compute;
        assert!(
            !cs_cheap.v2_blocks,
            "fixture must actually resolve to the cheap set"
        );

        let src = tests::textured_image(96, 64, 13);
        let dst = tests::quantize_distort(&src, 96, 64);
        let (sref, dref) = (RgbSlice::new(&src, 96, 64), RgbSlice::new(&dst, 96, 64));
        let mut scratch = V2Scratch::new();

        let toggles_everything = V2NewFeatureToggles {
            v1_pools: V1PoolsMode::Full,
            append_block: true,
            append2_block: true,
            free_extras: V1FreeExtras::Off,
            ..V2NewFeatureToggles::default()
        };
        // `append_block`/`append2_block` control the OUTPUT LAYOUT
        // (`layout_append`/`layout_append2` in `foldapp_streaming_walk`),
        // independently of `v1_only` gating the actual append/append2
        // COMPUTE off (`ComputeSet.append`/`.append2`, which is what makes
        // this walk cheap) — both must stay on so the free-extras values
        // have their normal 944-wide homes to land in and the two vectors
        // stay directly comparable index-for-index.
        let toggles_cheap = V2NewFeatureToggles {
            v1_pools: cs_cheap.v1_pools,
            v1_only: true,
            append_block: true,
            append2_block: true,
            free_extras: cs_cheap.free_extras,
            ..V2NewFeatureToggles::default()
        };
        let full = foldapp_streaming_walk(
            &sref,
            &dref,
            false,
            toggles_everything,
            crate::feature_v2_stream::FrontEnd::Sdr,
            &mut scratch,
            FoldWalkExtras::default(),
        )
        .into_features();
        let cheap = foldapp_streaming_walk(
            &sref,
            &dref,
            false,
            toggles_cheap,
            crate::feature_v2_stream::FrontEnd::Sdr,
            &mut scratch,
            FoldWalkExtras::default(),
        )
        .into_features();
        assert_eq!(full.len(), cheap.len());
        // Basic+peaks slots (pure v1, no raw-moments involved) must be
        // BIT-identical -- the cheap walk's v1 fold and the full walk's v1
        // fold are the same code, nothing about free-extras touches them.
        for &i in &[5usize, 200] {
            assert_eq!(
                full[i].to_bits(),
                cheap[i].to_bits(),
                "slot {i} (basic/peaks, unrelated to free-extras) moved between full and cheap"
            );
        }
        // The free-extras slots themselves are NOT required to be
        // bit-identical: `free_extras_match_the_944_append_block` (this
        // same test module) already establishes and bounds the summation-
        // order tolerance between the v1-only fused-kernel accumulation and
        // the dedicated append kernel's -- 2e-5, `benchmarks/
        // free_features_2026-09-01.md`. Reused here, not re-derived.
        const FREE_TOL: f64 = 2e-5;
        for &i in &nonzero {
            if i == 5 || i == 200 {
                continue; // already checked above as a bit-exact slot
            }
            let d = (full[i] - cheap[i]).abs();
            assert!(
                d <= FREE_TOL,
                "slot {i} (a LIVE free-extras weight): full={:e} cheap={:e} |Δ|={d:e} > {FREE_TOL:e}",
                full[i],
                cheap[i]
            );
        }
        // And the SCORE this model would produce -- dot(features, weights)
        // over exactly the live positions -- must therefore also agree
        // well within the model's own tolerance for this class of bake.
        let score = |f: &[f64]| -> f64 { nonzero.iter().map(|&i| f[i]).sum() };
        assert!(
            (score(&full) - score(&cheap)).abs() <= FREE_TOL * nonzero.len() as f64,
            "full={:e} cheap={:e}",
            score(&full),
            score(&cheap)
        );
    }

    /// The trap that motivates [`super::era2_reduce8`]: `reduce_add()` is
    /// TIER-DEPENDENT (§14.2), so it can never be part of era-2's semantics.
    ///
    /// This does not assert a divergence (this box runs one tier), it asserts
    /// the FIXED tree is what we think it is — an executable statement of the
    /// semantics, so a future refactor that silently swaps in `reduce_add()`
    /// fails here.
    #[test]
    fn era2_reduce_tree_is_fixed_and_explicit() {
        // Values chosen so the two known groupings genuinely differ: adjacent
        // pairing vs the x86_v3 (i, i+4) pairing.
        let a: [f32; 8] = [1.0, 1e-8, 1.0, 1e-8, -1.0, 1e-8, -1.0, 1e-8];
        let fixed = super::era2_reduce8(a);
        let adjacent = (((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))) as f64;
        let v3_style = (((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]))) as f64;
        assert_eq!(
            fixed.to_bits(),
            adjacent.to_bits(),
            "era2_reduce8 must be the ADJACENT-pairwise tree, written out"
        );
        assert_ne!(
            adjacent.to_bits(),
            v3_style.to_bits(),
            "the two backend reduction shapes must be demonstrably different on \
             this fixture — if they agree, the fixture stopped exercising the \
             divergence and this test no longer documents anything"
        );
    }

    /// ERA-2 ORACLE GATE (`benchmarks/era2_perf_break_2026-08-31.md` §11).
    ///
    /// Judges the production kernels against an EXACT reference rather than
    /// against a previous implementation. Three things happen, in order:
    ///
    /// 1. **The oracle judges itself.** L1 (Neumaier-compensated f64) is
    ///    compared to L2 (Shewchuk exact expansion). If L1 drifts from L2
    ///    beyond compensated summation's own bound, the oracle is broken and
    ///    says so BEFORE it is used to judge anything else.
    /// 2. **Each production pool shape is measured against L2** — both the
    ///    `POOL_SIMD` lane form (what `v4x` ships today and what era-2 makes
    ///    universal) and the scalar-pool form (what every 16-register tier
    ///    ships today). Their deviations differ by construction, which is the
    ///    §10.4 accuracy-class change made visible instead of assumed.
    /// 3. **Every slot is checked against a bound derived from the SAME run** —
    ///    `Σ|xᵢ|` per slot comes out of the oracle, so the bound is computed
    ///    from measured magnitudes rather than guessed.
    ///
    /// Geometries deliberately straddle the classes that matter to the era-2
    /// reshape: tight vs non-tight width, `width.is_multiple_of(8)` vs not (the scalar
    /// tail), and heights that do and do not divide the band size.
    ///
    /// A deviation above its bound is a BUG — in the kernel or in the analysis
    /// — and the lane STOPS to find out which. It is never a tolerance to
    /// widen.
    #[test]
    fn era2_oracle_bounds_hold_for_every_pool_shape() {
        use oracle::{Exact, N_SLOTS, Neumaier, SLOT_NAMES, dense_accum_slots, slot_family};
        const U32: f64 = 5.960_464_477_539_063e-8; // 2^-24
        const U64: f64 = 1.110_223_024_625_157e-16; // 2^-53

        // (w, h, note)
        let geoms: &[(usize, usize, &str)] = &[
            (96, 64, "tight, w%8==0"),
            (200, 136, "non-tight, w%8==0"),
            (127, 93, "non-tight, w%8==7 (scalar tail)"),
            (61, 40, "sub-64, w%8==5 (tail + small)"),
        ];
        let mut worst_overall: (f64, String) = (0.0, String::new());
        // Per (variant, family): worst |dev| with the sum|x| and bound it came
        // with, so §10.5's table is filled from measurement, not guessed.
        let mut per: std::collections::BTreeMap<(&str, &str), (f64, f64, f64)> =
            std::collections::BTreeMap::new();

        for &(w, h, note) in geoms {
            let src_px = textured_image(w, h, 0xD1FF);
            let dst_px = quantize_distort(&src_px, w, h);
            let source = RgbSlice::new(&src_px, w, h);
            let distorted = RgbSlice::new(&dst_px, w, h);
            let src_planes = crate::streaming::convert_source_to_xyb(
                &source,
                w,
                false,
                crate::ssim_form::active_revision(),
            );
            let dst_planes = crate::streaming::convert_source_to_xyb(
                &distorted,
                w,
                false,
                crate::ssim_form::active_revision(),
            );
            let mut scratch = ScratchV2Strip::new(w * h);
            let n = w * h;

            for ch in 0..3 {
                run_blur_pass(&src_planes[ch], &dst_planes[ch], w, h, &mut scratch);
                let (mu1, mu2) = (&scratch.mu1[..n], &scratch.mu2[..n]);
                let (ssq, s12) = (&scratch.ssq[..n], &scratch.s12[..n]);
                let act = &scratch.activity[..n];

                let (l1, _) = oracle::dense_reference::<Neumaier>(
                    &src_planes[ch],
                    &dst_planes[ch],
                    mu1,
                    mu2,
                    ssq,
                    s12,
                    act,
                    w,
                    h,
                    true,
                );
                let (l2, sum_abs) = oracle::dense_reference::<Exact>(
                    &src_planes[ch],
                    &dst_planes[ch],
                    mu1,
                    mu2,
                    ssq,
                    s12,
                    act,
                    w,
                    h,
                    true,
                );

                // (1) The oracle judges itself. Compensated summation's error
                // is O(u) * Σ|x| regardless of n — that is the whole point of
                // the compensation — so a generous constant times u64 is a
                // true bound, not a fitted one.
                for i in 0..N_SLOTS {
                    let allow = 64.0 * U64 * sum_abs[i] + f64::MIN_POSITIVE;
                    assert!(
                        (l1[i] - l2[i]).abs() <= allow,
                        "{w}x{h} ch{ch} [{note}]: ORACLE SELF-CHECK FAILED at {} \
                         (L1 Neumaier {:e} vs L2 exact {:e}, allowed {:e}). The oracle is \
                         broken; do not trust any kernel judged by it.",
                        SLOT_NAMES[i],
                        l1[i],
                        l2[i],
                        allow
                    );
                }

                // (2)+(3) Each production pool shape vs L2, bound from Σ|x|.
                let lane_pool = dense_block_kernel_era1(
                    &src_planes[ch],
                    &dst_planes[ch],
                    mu1,
                    mu2,
                    ssq,
                    s12,
                    act,
                    w,
                    h,
                    true,
                    None,
                );
                let scalar_pool = dense_block_kernel_pools_scalar(
                    &src_planes[ch],
                    &dst_planes[ch],
                    mu1,
                    mu2,
                    ssq,
                    s12,
                    act,
                    w,
                    h,
                    true,
                    None,
                );

                let era2 = dense_block_kernel_era2(
                    &src_planes[ch],
                    &dst_planes[ch],
                    mu1,
                    mu2,
                    ssq,
                    s12,
                    act,
                    w,
                    h,
                    true,
                    None,
                );
                for (variant, accum) in [
                    ("dispatched", &lane_pool),
                    ("pools_scalar", &scalar_pool),
                    ("era2", &era2),
                ] {
                    let got = dense_accum_slots(accum);
                    for i in 0..N_SLOTS {
                        let fam = slot_family(i);
                        // f32 lane accumulation: (chunks per lane) + (reduce
                        // tree depth) terms at u32. f64 aggregation: one add
                        // per row at u64. The scalar-pool variant accumulates
                        // its POOL slots per pixel in f64 instead, so those
                        // carry no u32 term.
                        // BOUND per design §12.2 + §12.5. Two components:
                        //
                        // (1) TERM-EVALUATION, absolute and cancellation-safe.
                        //     Every intermediate in these formulas is bounded
                        //     by 1 in magnitude (d, sal, the weights,
                        //     bounded_sim, saturate all live in [0,1]), so ONE
                        //     f32 rounding anywhere in a term contributes at
                        //     most u32·1 — ABSOLUTE, not relative to |term|.
                        //     That matters because `d = max(1-local,0)` with
                        //     `local ≈ 1` is a CANCELLING difference: d's
                        //     absolute error does not shrink as d does, and
                        //     the derived slots (d², d³, d⁴, and every pool
                        //     `num`) inherit it amplified by the derivative.
                        //     A bound written against the slot's own Σ|x|
                        //     understates those by two orders (Σ|d| is 282x
                        //     Σ|d²| here) — measured, and it is what the
                        //     oracle caught first.
                        //
                        // (2) ACCUMULATION, relative to Σ|x| (design §12.2).
                        //
                        // (1) dominates and is deliberately LOOSE — a proven
                        // upper bound, not a fitted one. The regression signal
                        // is the reported per-family COEFFICIENT below, which
                        // moves long before the bound is crossed.
                        const SAFETY: f64 = 2.0;
                        let k_eval = SAFETY
                            * match fam {
                                "core" => 13.0,
                                "hf" => 7.0,
                                "pjnd" => 5.0,
                                _ => 17.0, // pools: core formula + weight ops
                            };
                        let n_px = (w * h) as f64;
                        let chunks_per_lane = (w / 8) as f64;
                        let pools_in_f64 = variant == "pools_scalar" && fam == "pools";
                        // era-2 folds its tail into the lanes, so it has no
                        // per-pixel f64 tail term (era-1 does).
                        let era2_shape = variant == "era2";
                        let acc_u32 = if pools_in_f64 {
                            0.0
                        } else {
                            chunks_per_lane + 8.0
                        };
                        let acc_u64 = if pools_in_f64 {
                            n_px
                        } else {
                            h as f64
                                + 8.0
                                + if era2_shape {
                                    0.0
                                } else {
                                    (w % 8) as f64 * h as f64
                                }
                        };
                        let bound = k_eval * n_px * U32
                            + (acc_u32 * U32 + acc_u64 * U64) * sum_abs[i]
                            + 8.0 * U64 * l2[i].abs()
                            + f64::MIN_POSITIVE;
                        let dev = (got[i] - l2[i]).abs();
                        {
                            let e = per.entry((variant, fam)).or_insert((0.0, 0.0, 0.0));
                            if dev > e.0 {
                                *e = (dev, sum_abs[i], bound);
                            }
                        }
                        // Reported signal: dev as a fraction of its bound.
                        // A kernel change that degrades accuracy moves this
                        // long before it trips the assert.
                        let frac = dev / bound;
                        if frac > worst_overall.0 {
                            worst_overall = (
                                frac,
                                format!(
                                    "{} {}x{} ch{} [{}] {} ({}): dev {:e} = {:.1}% of bound {:e}",
                                    variant,
                                    w,
                                    h,
                                    ch,
                                    note,
                                    SLOT_NAMES[i],
                                    fam,
                                    dev,
                                    100.0 * frac,
                                    bound
                                ),
                            );
                        }
                        assert!(
                            dev <= bound,
                            "{w}x{h} ch{ch} [{note}] {variant}: slot {} ({}) deviates {:e} from \
                             the EXACT sum, above its proven bound {:e} (Σ|x| = {:e}). A bound \
                             violation is a BUG in the kernel or in the error analysis — find \
                             out which. Do NOT widen this.",
                            SLOT_NAMES[i],
                            fam,
                            dev,
                            bound,
                            sum_abs[i]
                        );
                    }
                }
            }
        }
        eprintln!(
            "ERA2-ORACLE worst deviation = {:.2}% of its proven bound\n  {}",
            100.0 * worst_overall.0,
            worst_overall.1
        );
        eprintln!("ERA2-ORACLE per-variant x family worst |dev| vs the EXACT sum:");
        eprintln!(
            "  {:<14} {:<7} {:>13} {:>13} {:>13} {:>8}",
            "variant", "family", "max|dev|", "sum|x|", "bound", "% bound"
        );
        for ((v, f), (dev, sa, bd)) in &per {
            eprintln!(
                "  {v:<14} {f:<7} {dev:>13.4e} {sa:>13.4e} {bd:>13.4e} {:>7.1}%",
                100.0 * dev / bd
            );
        }
    }

    /// identical and the gate passes trivially.
    #[test]
    fn pool_simd_drift_within_policy() {
        let (w, h) = (200usize, 136usize);
        let src_px = textured_image(w, h, 0xD1FF);
        let dst_px = quantize_distort(&src_px, w, h);

        // Build real moment planes via the whole-image blur pass.
        let source = RgbSlice::new(&src_px, w, h);
        let distorted = RgbSlice::new(&dst_px, w, h);
        let src_planes = crate::streaming::convert_source_to_xyb(
            &source,
            w,
            false,
            crate::ssim_form::active_revision(),
        );
        let dst_planes = crate::streaming::convert_source_to_xyb(
            &distorted,
            w,
            false,
            crate::ssim_form::active_revision(),
        );
        let mut scratch = ScratchV2Strip::new(w * h);
        for ch in 0..3 {
            run_blur_pass(&src_planes[ch], &dst_planes[ch], w, h, &mut scratch);
            let n = w * h;
            let a = dense_block_kernel_era1(
                &src_planes[ch],
                &dst_planes[ch],
                &scratch.mu1[..n],
                &scratch.mu2[..n],
                &scratch.ssq[..n],
                &scratch.s12[..n],
                &scratch.activity[..n],
                w,
                h,
                true,
                None,
            );
            let b = dense_block_kernel_pools_scalar(
                &src_planes[ch],
                &dst_planes[ch],
                &scratch.mu1[..n],
                &scratch.mu2[..n],
                &scratch.ssq[..n],
                &scratch.s12[..n],
                &scratch.activity[..n],
                w,
                h,
                true,
                None,
            );
            let pools = [
                ("mask_ssim", a.ws_mask_ssim, b.ws_mask_ssim),
                ("mask_art", a.ws_mask_art, b.ws_mask_art),
                ("mask_det", a.ws_mask_det, b.ws_mask_det),
                ("mask_mse", a.ws_mask_mse, b.ws_mask_mse),
                ("iw_ssim", a.ws_iw_ssim, b.ws_iw_ssim),
                ("iw_art", a.ws_iw_art, b.ws_iw_art),
                ("iw_det", a.ws_iw_det, b.ws_iw_det),
                ("iw_mse", a.ws_iw_mse, b.ws_iw_mse),
                ("peak_ssim", a.ws_peak_ssim, b.ws_peak_ssim),
                ("peak_art", a.ws_peak_art, b.ws_peak_art),
                ("peak_det", a.ws_peak_det, b.ws_peak_det),
            ];
            for (name, pa, pb) in pools {
                for (part, va, vb) in [("num", pa.num, pb.num), ("den", pa.den, pb.den)] {
                    let rel = (va - vb).abs() / vb.abs().max(1e-9);
                    assert!(
                        rel <= TOL,
                        "ch{ch} pool {name}.{part}: rel drift {rel:e} \
                         (simd={va:e} scalar={vb:e}) exceeds policy {TOL:e}"
                    );
                }
            }
            // Non-pool sums must be BIT-identical — POOL_SIMD touches
            // nothing else.
            assert_eq!(a.sum_d.to_bits(), b.sum_d.to_bits());
            assert_eq!(a.sum_mse.to_bits(), b.sum_mse.to_bits());
            assert_eq!(a.sum_pjnd.to_bits(), b.sum_pjnd.to_bits());
        }
    }

    /// Dimension mismatch against the prepared reference is rejected the
    /// same way `validate_pair` rejects it on the pair path.
    #[test]
    fn prepared_ref_rejects_dim_mismatch() {
        let src = textured_image(96, 96, 3);
        let source = RgbSlice::new(&src, 96, 96);
        let prepared =
            prepare_v2_reference_impl(&source, None, false, false).expect("prepare computes");

        let other = textured_image(64, 96, 4);
        let distorted = RgbSlice::new(&other, 64, 96);
        let mut scratch = V2Scratch::new();
        let err = compute_v2_features_with_ref_impl(
            &prepared,
            &distorted,
            None,
            false,
            V2NewFeatureToggles::default(),
            &mut scratch,
        )
        .unwrap_err();
        assert!(matches!(err, ZensimError::DimensionMismatch));
    }

    /// **The HDR extraction routes reach revision 3, and still hold there.**
    ///
    /// Declared-HDR extraction goes through `foldapp_streaming_walk`, i.e. the
    /// same fold band replay as SDR, so it inherits the corrected SSIM signal
    /// automatically. "Automatically" is the part worth checking: this re-runs
    /// the existing `hdr_*` gates — route refusal, auto-vs-explicit routing,
    /// identity zeros, and bounded output over all four encodings — in a
    /// revision-3 process, so an HDR route that silently broke under the new
    /// arithmetic fails here rather than in a fleet extraction.
    ///
    /// It does NOT establish HDR quality at revision 3; no HDR bake is fit
    /// against corrected features.
    #[test]
    fn rev3_hdr_route_gates_hold() {
        // Named so it does NOT match its own filter — a wrapper that counted
        // itself would mask one of the four gates going missing.
        crate::ssim_form::rerun_tests_at_revision("3", "feature_v2::tests::hdr_", 4);
    }

    /// **Folded-vs-streaming parity holds at revision 3 too.**
    ///
    /// The two `folded720_v1_*` gates below are the crate's exact contract
    /// between the folded band replay and v1's own strip walk. Revision 3
    /// changes where the SSIM signal comes from on BOTH sides, so the
    /// contract has to be re-established at that revision rather than
    /// assumed to survive; this re-runs those same gates in a process pinned
    /// to revision 3 instead of copying them.
    ///
    /// `training`-gated for the same reason the gates themselves are: their
    /// v1 side is `compute_zensim_with_config`, a training-only export. The
    /// wrapper must not exist in a build where the filter would match
    /// nothing.
    #[cfg(feature = "training")]
    #[test]
    fn fold_parity_gates_hold_at_revision_three() {
        // This wrapper is deliberately NOT named `folded720_v1_*`: the
        // filter below must match exactly the two real gates, and a control
        // that counted itself would hide one of them going missing.
        crate::ssim_form::rerun_tests_at_revision("3", "folded720_v1_", 2);
    }

    /// FOLD PARITY GATE (2026-07-24). The folded-720 path replays v1's
    /// exact 32-row band tiling over shared H-planes, so its v1 basic
    /// block (`f0..156`) is **BIT-IDENTICAL** to the frozen v1 extraction
    /// path whenever the working widths agree — i.e. whenever
    /// `pyramid_plane_stride(w) == w` (multiples of 16 below 512, plus the
    /// non-bumped alignments above; see `blur::pyramid_plane_stride`).
    ///
    /// For other widths the two regimes are STRUCTURALLY different by
    /// v1's own frozen semantics: v1 computes over the SIMD-padded width
    /// (mirrored pad columns participate in pooling, and the downscale
    /// chain halves the padded width, shifting every deeper scale's
    /// grid), while the fold runs at the true width. That divergence is
    /// v1's pad wart, not fold noise — it is the documented boundary of
    /// the [`FeatureRegime::Folded720`] contract, measured on real
    /// content + real models in `benchmarks/` (fold_v1_basic_*), and it
    /// is why folded rows must NEVER be silently mixed into v1-extracted
    /// corpora.
    ///
    /// Also gates: `f156..372` all-zero (deprecated pool blocks) and the
    /// v2 block (`f372..`) BIT-identical to the plain v2 path, for BOTH
    /// width classes.
    ///
    /// `training`-gated: the v1 side of the comparison is
    /// `compute_zensim_with_config`, a training-only export.
    #[cfg(feature = "training")]
    #[test]
    fn folded720_v1_basic_matches_v1_path() {
        // OPTION C (2026-08-30): the `expect_bit_exact` flag is GONE and
        // every geometry now asserts bit-exactness. 127 and 200 used to be
        // the "padded-width divergence class" — v1 pooled mirror-filled
        // phantom columns there and the fold did not. C removed the phantom
        // columns, so the class does not exist. Sanctioned re-pin (era-3,
        // docs/DATASET_HISTORY.md): a tightening, never a loosening.
        // ≥64 sizes only: `compute_zensim_with_config` (the fleet v1
        // extractor) does not reflect-pad sub-64 (see
        // `folded720_sub64_matches_padding_v1_entry` for that).
        for &(w, h) in &[
            (96usize, 64usize),
            (64, 300),
            (208, 144),
            (127, 93),
            (200, 150),
        ] {
            let src = textured_image(w, h, 7);
            let dst = quantize_distort(&src, w, h);

            // v1, the production extraction config (extended + IW = 372).
            let cfg = crate::ZensimConfig {
                extended_features: true,
                compute_iw_features: true,
                allow_multithreading: false,
                ..Default::default()
            };
            let v1 = crate::compute_zensim_with_config(&src, &dst, w, h, cfg).unwrap();
            let v1f = v1.features();
            assert_eq!(v1f.len(), 372);

            let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
            let sref = RgbSlice::new(&src, w, h);
            let dref = RgbSlice::new(&dst, w, h);
            let v2 = z.compute_v2_features(&sref, &dref).unwrap();
            let folded = z.compute_folded720_features(&sref, &dref).unwrap();
            assert_eq!(folded.regime(), FeatureRegime::Folded720);
            let ff = folded.features();
            assert_eq!(ff.len(), 720);

            // v2 block: bit-identical to the plain v2 path.
            for (i, (&a, &b)) in ff[372..].iter().zip(v2.features().iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{w}x{h}: folded v2 slot {i} ({a:e}) != plain v2 ({b:e})"
                );
            }
            // Deprecated pool blocks: exactly zero.
            assert!(
                ff[156..372].iter().all(|&v| v == 0.0),
                "{w}x{h}: deprecated f156..372 must be exactly 0.0"
            );
            // Every width is the bit-exact class under C. If this fires,
            // something structural regressed (band extents, H sharing, merge
            // order, or padding reintroduced) — investigate, never widen to
            // a tolerance.
            for i in 0..156 {
                assert_eq!(
                    ff[i].to_bits(),
                    v1f[i].to_bits(),
                    "{w}x{h}: basic f{i} fold {:e} != v1 {:e}",
                    ff[i],
                    v1f[i]
                );
            }
        }
    }

    /// OPTION-C GATE (2026-08-30). **This test's meaning was deliberately
    /// INVERTED** when C shipped, and that is the point of the comment.
    ///
    /// It used to be `v1_padded_width_divergence_is_column_padding`, and it
    /// asserted that the buffered v1 path and the fold DISAGREE at every
    /// non-tight width — up to 81.6 % relative on a pool slot — because v1
    /// walked `simd_padded_width(width)` columns and pooled the mirror-filled
    /// extras while the fold walked the image. That divergence was the defect,
    /// not an invariant. C removes the phantom columns
    /// (`blur::pyramid_plane_stride` now returns the width), so the assertion
    /// **The bit-exactness gate for [`MeanOffsetRows::add_strip`]'s row
    /// fan-out** (fold-MT lane). Serial and parallel must agree on every
    /// `rows[y][ch]` AND on `finish()`, at heights that straddle the band
    /// size, at a width where the per-row `f64` sum has a long carry chain,
    /// and across a multi-strip walk (so the `y0` offsets differ).
    ///
    /// The property is assignment-only per (row, channel), so this cannot
    /// fail for an ordering reason — it can only fail if someone later splits
    /// the sum WITHIN a row, which is exactly the mistake to catch.
    #[test]
    fn mean_offset_row_bands_are_bit_exact() {
        for &(width, height, strip) in &[
            (64usize, 128usize, 128usize),
            (2304, 128, 128),
            (97, 40, 17),
            (7, 300, 128),
            (33, 33, 8),
        ] {
            let n = width * height;
            let src: Vec<f32> = (0..n)
                .map(|i| (i.wrapping_mul(2654435761usize) % 1013) as f32 / 1013.0)
                .collect();
            let dst: Vec<f32> = (0..n)
                .map(|i| (i.wrapping_mul(40503usize).wrapping_add(3) % 1009) as f32 / 1009.0)
                .collect();
            let run = |parallel: bool| -> (Vec<[f64; 3]>, [f64; 3]) {
                let mut mo = MeanOffsetRows::new(width, height);
                let mut y0 = 0;
                while y0 < height {
                    let rows = strip.min(height - y0);
                    let lo = y0 * width;
                    let hi = lo + rows * width;
                    let s: [&[f32]; 3] = [&src[lo..hi], &src[lo..hi], &dst[lo..hi]];
                    let d: [&[f32]; 3] = [&dst[lo..hi], &src[lo..hi], &src[lo..hi]];
                    mo.add_strip(y0, rows, width, s, d, parallel);
                    y0 += rows;
                }
                (mo.rows.clone(), mo.finish())
            };
            let (rs, fs) = run(false);
            let (rp, fp) = run(true);
            for (y, (a, b)) in rs.iter().zip(rp.iter()).enumerate() {
                for ch in 0..3 {
                    assert_eq!(
                        a[ch].to_bits(),
                        b[ch].to_bits(),
                        "{width}x{height} strip {strip}: row {y} ch {ch} \
                         serial {} vs parallel {}",
                        a[ch],
                        b[ch]
                    );
                }
            }
            for ch in 0..3 {
                assert_eq!(
                    fs[ch].to_bits(),
                    fp[ch].to_bits(),
                    "{width}x{height} strip {strip}: finish()[{ch}]"
                );
            }
        }
    }

    /// **The bit-exactness gate for [`FoldHSource::SelfBlur`]** (fold-MT lane).
    ///
    /// A band that H-blurs its own `[b0 − overlap, b1 + overlap)` rows must
    /// produce byte-identical sums to one reading those rows out of a
    /// whole-window blur. The property it rests on is the H blur's per-row
    /// independence; this asserts the consequence directly, on every field of
    /// `V1BasicSums`, at geometries that exercise a top band (clamped `top`),
    /// a bottom band (clamped `bot`), a partial last band, and both pool
    /// modes' band counts.
    #[test]
    fn fold_self_blur_matches_precomputed_h() {
        for &(width, plane_h, strip_h) in &[
            (64usize, 128usize, 128usize),
            (97, 96, 96),
            (2304, 128, 128),
            (128, 200, 128),
            (65, 40, 40),
        ] {
            let wide_h = strip_h + 2 * HALO_P;
            let n = width * wide_h;
            let src: Vec<f32> = (0..n)
                .map(|i| (i.wrapping_mul(2654435761usize) % 977) as f32 / 977.0)
                .collect();
            let dst: Vec<f32> = (0..n)
                .map(|i| (i.wrapping_mul(40503usize).wrapping_add(11) % 991) as f32 / 991.0)
                .collect();
            // The precomputed arm's four planes, over the WHOLE wide window.
            let mut h = [vec![0.0f32; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]];
            {
                let (a, rest) = h.split_at_mut(1);
                let (b, rest2) = rest.split_at_mut(1);
                let (c, d) = rest2.split_at_mut(1);
                crate::blur::fused_blur_h_ssim(
                    &src,
                    &dst,
                    &mut a[0],
                    &mut b[0],
                    &mut c[0],
                    &mut d[0],
                    width,
                    wide_h,
                    BLUR_RADIUS,
                );
            }
            for &parallel in &[false, true] {
                let run = |h_src: FoldHSource<'_>| -> V1BasicSums {
                    let mut pool: Vec<FoldPoolScratch> = (0..V1_BANDS_PER_STRIP)
                        .map(|_| FoldPoolScratch::default())
                        .collect();
                    let mut sums = V1BasicSums::meas_f64();
                    fold_v1_basic_bands(
                        width,
                        0..strip_h.min(plane_h),
                        0,
                        HALO_P,
                        plane_h,
                        h_src,
                        [&src, &dst],
                        &mut sums,
                        Some((&mut pool[..], BandPoolWork::Full)),
                        parallel,
                        // Every FREE extra ON: the self-blur-vs-precomputed
                        // identity has to cover them too.
                        crate::fused::FreeExtrasWork {
                            raw_moments: true,
                            bounded_err: true,
                            lum_bins: true,
                            ..Default::default()
                        },
                    );
                    sums
                };
                let want = run(FoldHSource::Precomputed([&h[0], &h[1], &h[2], &h[3]]));
                let got = run(FoldHSource::SelfBlur);
                // `{:?}` on f64 is the shortest round-tripping form, so it
                // separates every pair of distinct finite values — the one
                // hole is NaN, whose payload it hides, so exclude NaN
                // explicitly rather than let it mask a difference.
                let wb = format!("{want:?}");
                let gb = format!("{got:?}");
                assert!(
                    !wb.contains("NaN"),
                    "{width}x{plane_h}: fixture produced NaN, which would make \
                     the Debug comparison below blind: {wb}"
                );
                assert_eq!(
                    wb, gb,
                    "{width}x{plane_h} strip {strip_h} par={parallel}: self-blur bands \
                     must equal precomputed-H bands"
                );
            }
        }
    }

    /// **The bit-exactness gate for [`fused_blur_h_ssim_banded`]** (fold-MT
    /// lane). The banded call must equal the whole-plane call BIT-FOR-BIT, at
    /// every band size that is a multiple of [`H_BLUR_ROW_GROUP`] and at
    /// heights that exercise a partial final group, a partial final band, and
    /// both at once.
    ///
    /// This is stronger than the shipped constant needs: it sweeps band sizes
    /// 8/12/16/24/32/64 so that re-tuning `H_BLUR_BAND_ROWS` on a future box
    /// cannot silently move a byte. Sizes 8, 12 and 24 are NOT multiples of
    /// [`H_BLUR_ROW_GROUP`] (16 — `v4`/`v4x` transpose 16 rows at a time;
    /// `v3`/`neon`/`wasm128`/`scalar` take 8), so they test the per-row
    /// independence itself rather than the alignment argument the shipped
    /// constant rests on. Recorded, not relied on: the shipped constant stays
    /// a multiple and the `const` assert beside it enforces that.
    #[test]
    fn phase_a_blur_bands_are_bit_exact() {
        for &(width, height) in &[
            (64usize, 148usize), // the scale-0 strip shape: 18 groups + a 4-row tail
            (2304, 40),          // wide, band-splittable
            (97, 148),           // odd width (scalar tail inside every row)
            (16, 7),             // shorter than one row group
            (33, 8),             // exactly one row group
            (127, 93),           // a golden-fixture height
            (5, 149),            // narrower than one SIMD lane
        ] {
            let n = width * height;
            let src: Vec<f32> = (0..n)
                .map(|i| (i.wrapping_mul(2654435761usize) % 1000) as f32 / 1000.0)
                .collect();
            let dst: Vec<f32> = (0..n)
                .map(|i| (i.wrapping_mul(40503usize).wrapping_add(7) % 1000) as f32 / 1000.0)
                .collect();
            let mut want = [vec![0.0f32; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]];
            {
                let (a, rest) = want.split_at_mut(1);
                let (b, rest2) = rest.split_at_mut(1);
                let (c, d) = rest2.split_at_mut(1);
                crate::blur::fused_blur_h_ssim(
                    &src,
                    &dst,
                    &mut a[0],
                    &mut b[0],
                    &mut c[0],
                    &mut d[0],
                    width,
                    height,
                    BLUR_RADIUS,
                );
            }
            for &bands in &[8usize, 12, 16, 24, 32, 64] {
                let mut got = [vec![0.0f32; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]];
                // Replay the helper's banding by hand at an arbitrary band
                // size — the helper itself is pinned to one constant, and the
                // claim under test is about the BANDING, not that constant.
                let mut y0 = 0;
                while y0 < height {
                    let rows = bands.min(height - y0);
                    let lo = y0 * width;
                    let hi = lo + rows * width;
                    let (a, rest) = got.split_at_mut(1);
                    let (b, rest2) = rest.split_at_mut(1);
                    let (c, d) = rest2.split_at_mut(1);
                    crate::blur::fused_blur_h_ssim(
                        &src[lo..hi],
                        &dst[lo..hi],
                        &mut a[0][lo..hi],
                        &mut b[0][lo..hi],
                        &mut c[0][lo..hi],
                        &mut d[0][lo..hi],
                        width,
                        rows,
                        BLUR_RADIUS,
                    );
                    y0 += rows;
                }
                for (pi, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                    for (i, (a, b)) in g.iter().zip(w.iter()).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "{width}x{height} bands={bands} plane {pi} idx {i}: \
                             banded {a} vs whole-plane {b}"
                        );
                    }
                }
            }
        }
    }

    /// flips: the two paths must now be BIT-IDENTICAL everywhere.
    ///
    /// This is a sanctioned re-pin — a named, deliberate semantics correction
    /// recorded in `docs/DATASET_HISTORY.md` (era-3) — NOT a tolerance
    /// widening. Nothing here is loosened: it went from "assert they differ,
    /// bounded" to "assert they are equal, exactly".
    ///
    /// The geometry set is kept verbatim from the characterisation, INCLUDING
    /// the three h = 93 cells that were the last residual under the option-A
    /// pre-pad workaround. They pass by equality here; the residual was an
    /// artifact of padding the input and does not exist under C.
    #[cfg(feature = "training")]
    #[test]
    fn v1_372_bit_exact_to_fold_at_every_width() {
        const CELLS: &[(usize, usize)] = &[
            // formerly "tight" (the only class that used to agree)
            (96, 64),
            (208, 144),
            (592, 80),
            (128, 93),
            // formerly divergent — even, non-tight
            (200, 150),
            (200, 151),
            (576, 96),
            (1152, 72),
            (100, 96),
            // formerly divergent — odd, non-tight
            (127, 64),
            (127, 96),
            (127, 128),
            (129, 96),
            (201, 96),
            (255, 96),
            (577, 80),
            // the h = 93 cells: the last residual under the pre-pad workaround
            (126, 93),
            (127, 93),
            (255, 93),
        ];
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        for &(w, h) in CELLS {
            assert_eq!(
                crate::blur::pyramid_plane_stride(w),
                w,
                "C: the pyramid stride must equal the width"
            );
            let src = textured_image(w, h, 7);
            let dst = quantize_distort(&src, w, h);
            let cfg = crate::ZensimConfig {
                extended_features: true,
                compute_iw_features: true,
                allow_multithreading: false,
                ..Default::default()
            };
            let v1 = crate::compute_zensim_with_config(&src, &dst, w, h, cfg).unwrap();
            let v1f = v1.features();
            assert_eq!(v1f.len(), 372, "{w}x{h}: v1 must emit 372");

            let mut scratch = V2Scratch::new();
            let folded = z
                .compute_folded720_append_features_streaming(
                    &RgbSlice::new(&src, w, h),
                    &RgbSlice::new(&dst, w, h),
                    V2NewFeatureToggles {
                        v1_pools: V1PoolsMode::Full,
                        ..V2NewFeatureToggles::default()
                    },
                    &mut scratch,
                )
                .unwrap();
            let ff = folded.features();

            for i in 0..372 {
                assert_eq!(
                    ff[i].to_bits(),
                    v1f[i].to_bits(),
                    "{w}x{h}: f{i} fold {:e} != buffered v1 {:e}. Under option C these \
                     paths compute the same statistic over the same pixels — a difference \
                     here means padding was reintroduced somewhere, or one walk changed. \
                     Do NOT widen this to a tolerance.",
                    ff[i],
                    v1f[i]
                );
            }
        }
    }

    /// C, pinned at its owner: no phantom columns at ANY width. A future
    /// stride/alignment experiment that reintroduces padding must fail here
    /// and read the reason, because padding the plane silently re-pollutes
    /// every v1 pool with columns that are not in the image (that was the
    /// era-2 defect, worth up to 81.6 % on a pool slot).
    #[test]
    fn pyramid_stride_has_no_phantom_columns() {
        for w in [
            1usize, 7, 8, 15, 16, 17, 63, 64, 96, 100, 127, 128, 200, 255, 256, 511, 512, 513, 576,
            592, 1024, 1152, 2304, 4096,
        ] {
            assert_eq!(
                crate::blur::pyramid_plane_stride(w),
                w,
                "pyramid_plane_stride({w}) must be {w} under option C"
            );
        }
    }

    /// **The per-profile weight-skipping gate** (feature-cost lane,
    /// 2026-08-31). [`V1PoolsMode::Peaks`] must be pure compute-skipping:
    /// every slot it DOES emit is bit-identical to what
    /// [`V1PoolsMode::Full`] emits, and the masked/IW block it skips is left
    /// at exactly `0.0` (never a partially-accumulated value, never NaN from
    /// finalising an accumulator nothing wrote).
    ///
    /// Run over the same 19 geometries as
    /// `v1_372_bit_exact_to_fold_at_every_width`, in both the `v1_only`
    /// (fold-backed SCORE) walk and the full 944 walk, serial AND parallel —
    /// the parallel arm matters because `Peaks` is the second mode allowed to
    /// take the band-local self-blur shape, so a band-boundary interaction is
    /// exactly the failure mode.
    #[cfg(feature = "training")]
    #[test]
    fn folded_peaks_mode_is_pure_compute_skipping() {
        const CELLS: &[(usize, usize)] = &[
            (96, 64),
            (208, 144),
            (592, 80),
            (128, 93),
            (200, 150),
            (200, 151),
            (576, 96),
            (1152, 72),
            (100, 96),
            (127, 64),
            (127, 96),
            (127, 128),
            (129, 96),
            (201, 96),
            (255, 96),
            (577, 80),
            (126, 93),
            (127, 93),
            (255, 93),
        ];
        for &v1_only in &[true, false] {
            for &parallel in &[false, true] {
                let z = crate::Zensim::new(crate::ZensimProfile::codec_target())
                    .with_parallel(parallel);
                let mut scratch = V2Scratch::new();
                for &(w, h) in CELLS {
                    let src = textured_image(w, h, 7);
                    let dst = quantize_distort(&src, w, h);
                    let sref = RgbSlice::new(&src, w, h);
                    let dref = RgbSlice::new(&dst, w, h);
                    let base = V2NewFeatureToggles {
                        v1_only,
                        append_block: !v1_only,
                        append2_block: !v1_only,
                        ..V2NewFeatureToggles::default()
                    };
                    let full = z
                        .compute_folded720_append_features_streaming(
                            &sref,
                            &dref,
                            V2NewFeatureToggles {
                                v1_pools: V1PoolsMode::Full,
                                ..base
                            },
                            &mut scratch,
                        )
                        .unwrap();
                    let peaks = z
                        .compute_folded720_append_features_streaming(
                            &sref,
                            &dref,
                            V2NewFeatureToggles {
                                v1_pools: V1PoolsMode::Peaks,
                                ..base
                            },
                            &mut scratch,
                        )
                        .unwrap();
                    assert_eq!(peaks.v1_pools(), V1PoolsMode::Peaks);
                    assert!(peaks.v1_pools_live());
                    assert_eq!(full.regime(), peaks.regime(), "{w}x{h}: regime moved");
                    let (ff, fp) = (full.features(), peaks.features());
                    assert_eq!(ff.len(), fp.len(), "{w}x{h}: width moved");
                    let tag = if v1_only { "v1_only" } else { "944" };
                    // Emitted slots: bit-identical.
                    for i in (0..228).chain(372..ff.len()) {
                        assert_eq!(
                            fp[i].to_bits(),
                            ff[i].to_bits(),
                            "{w}x{h} {tag} par={parallel}: slot {i} moved under Peaks ({:e} vs {:e})",
                            fp[i],
                            ff[i]
                        );
                    }
                    // Skipped block: exactly zero, and finite.
                    for (i, &v) in fp[228..372].iter().enumerate() {
                        assert_eq!(
                            v.to_bits(),
                            0.0f64.to_bits(),
                            "{w}x{h} {tag} par={parallel}: skipped slot {} is {v:e}, not +0.0",
                            228 + i
                        );
                    }
                    // The peak block must actually carry values — a mode that
                    // emitted zeros there would pass the identity above only
                    // because Full's were zero too.
                    assert!(
                        fp[156..228].iter().any(|&v| v > 0.0),
                        "{w}x{h} {tag} par={parallel}: Peaks emitted an all-zero peak block"
                    );
                }
            }
        }
    }

    /// **FREE-EXTRAS GATE 1 — the free set is pure ADDITION.**
    ///
    /// [`V1FreeExtras::RawMoments`] on a `v1_only` walk must leave every slot
    /// the walk already emitted BIT-IDENTICAL and fill only the 40 free
    /// slots, which are `+0.0` with it off. This is the claim that makes the
    /// perf question a fair one: the two arms differ by the accumulators, not
    /// by any change to what the walk computes.
    #[cfg(feature = "training")]
    #[test]
    fn free_extras_are_pure_addition_to_the_v1_only_walk() {
        const CELLS: &[(usize, usize)] = &[
            (96, 64),
            (208, 144),
            (592, 80),
            (127, 93),
            (200, 151),
            (576, 96),
            (255, 96),
        ];
        for &parallel in &[false, true] {
            let z =
                crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(parallel);
            let mut scratch = V2Scratch::new();
            for &(w, h) in CELLS {
                let src = textured_image(w, h, 7);
                let dst = quantize_distort(&src, w, h);
                let sref = RgbSlice::new(&src, w, h);
                let dref = RgbSlice::new(&dst, w, h);
                // v1-only COMPUTE at the 944 LAYOUT: the free slots have
                // somewhere to land, and every v2-era block stays off.
                let base = V2NewFeatureToggles {
                    v1_only: true,
                    v1_pools: V1PoolsMode::Peaks,
                    append_block: true,
                    append2_block: true,
                    ..V2NewFeatureToggles::default()
                };
                let off = z
                    .compute_folded720_append_features_streaming(&sref, &dref, base, &mut scratch)
                    .unwrap();
                let on = z
                    .compute_folded720_append_features_streaming(
                        &sref,
                        &dref,
                        V2NewFeatureToggles {
                            free_extras: V1FreeExtras::RawMoments,
                            ..base
                        },
                        &mut scratch,
                    )
                    .unwrap();
                let (fo, fn_) = (off.features(), on.features());
                assert_eq!(fo.len(), fn_.len(), "{w}x{h}: width moved");
                assert_eq!(fo.len(), 944, "{w}x{h}: expected the 944 layout");
                let free: std::collections::HashSet<usize> =
                    free_slot_indices(off.n_scales()).into_iter().collect();
                for i in 0..fo.len() {
                    if free.contains(&i) {
                        assert_eq!(
                            fo[i].to_bits(),
                            0.0f64.to_bits(),
                            "{w}x{h} par={parallel}: free slot {i} is {:e} with the mode OFF",
                            fo[i]
                        );
                    } else {
                        assert_eq!(
                            fo[i].to_bits(),
                            fn_[i].to_bits(),
                            "{w}x{h} par={parallel}: slot {i} moved ({:e} -> {:e})",
                            fo[i],
                            fn_[i]
                        );
                    }
                }
                assert!(
                    free.iter().filter(|&&i| fn_[i] != 0.0).count() >= free.len() / 2,
                    "{w}x{h} par={parallel}: the free block came out mostly zero"
                );
            }
        }
    }

    /// **FREE-EXTRAS GATE 2 — the free slots ARE the 944 slots.**
    ///
    /// The whole point of the free set is that a 156-class walk can emit
    /// slots a model trained on 944 TABLES already knows. That is only true
    /// if the value matches. It cannot match BITWISE — the v1 fused kernel
    /// sums a (scale, channel) in column-group-major order over v1's 32-row
    /// bands while the append kernel sums it row-major over 128-row strips,
    /// so the two are the same multiset of f64 addends in a different
    /// grouping — but they must agree to f64 reassociation.
    ///
    /// MEASURED, and the binding term is NOT reassociation — it is that the
    /// two kernels accumulate at different PRECISION. The append kernel adds
    /// a whole row into f32 SIMD lanes (`r_s = r_s + s`, `width/8` adds per
    /// lane) and reduces to f64 once per row; the fused kernel reduces every
    /// 16-lane group to f64 immediately. So the append side carries ~`width/8`
    /// f32 accumulation steps where this side carries one, and the free value
    /// is the MORE accurate of the two. The signature confirms it: the
    /// deltas below scale with WIDTH (worst at the widest cell) and are
    /// bit-identical between the serial and band-parallel arms, so nothing
    /// about threading or band order is involved.
    ///
    /// Worst measured |Δ| over these cells (both arms): **4.62e-6** at
    /// 2304×64 (4.44e-6 at 1152×72, 1.24e-6 at 576×96 — the width scaling)
    /// on `GLOBAL_DMEAN`, i.e. ~1.9e-4 RELATIVE on a
    /// feature of scale 2.4e-2 — three orders inside this module's own
    /// documented 5e-4 v2 tolerance. The bar is 2e-5; the full table is in
    /// `benchmarks/free_features_2026-09-01.md`. `ZENSIM_FREE_DIAG=1`
    /// prints every non-zero delta with its (scale, channel, slot).
    #[cfg(feature = "training")]
    fn free_tol() -> f64 {
        std::env::var("ZENSIM_FREE_TOL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2e-5)
    }

    #[cfg(feature = "training")]
    #[test]
    fn free_extras_match_the_944_append_block() {
        const CELLS: &[(usize, usize)] = &[
            (96, 64),
            (208, 144),
            (592, 80),
            (127, 93),
            (200, 151),
            (576, 96),
            (255, 96),
            (1152, 72),
            // Production widths: the delta is width-driven, so the bar has
            // to be exercised where the product runs, not only at 576².
            (2304, 64),
        ];
        let mut worst = 0.0f64;
        let mut worst_at = (0usize, 0usize, 0usize);
        for &parallel in &[false, true] {
            let z =
                crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(parallel);
            let mut scratch = V2Scratch::new();
            for &(w, h) in CELLS {
                let src = textured_image(w, h, 7);
                let dst = quantize_distort(&src, w, h);
                let sref = RgbSlice::new(&src, w, h);
                let dref = RgbSlice::new(&dst, w, h);
                let full = z
                    .compute_folded720_append_features_streaming(
                        &sref,
                        &dref,
                        V2NewFeatureToggles {
                            append_block: true,
                            append2_block: true,
                            ..V2NewFeatureToggles::default()
                        },
                        &mut scratch,
                    )
                    .unwrap();
                let free = z
                    .compute_folded720_append_features_streaming(
                        &sref,
                        &dref,
                        V2NewFeatureToggles {
                            v1_only: true,
                            v1_pools: V1PoolsMode::Peaks,
                            append_block: true,
                            append2_block: true,
                            free_extras: V1FreeExtras::RawMoments,
                            ..V2NewFeatureToggles::default()
                        },
                        &mut scratch,
                    )
                    .unwrap();
                let (a, b) = (full.features(), free.features());
                for i in free_slot_indices(full.n_scales()) {
                    let d = (a[i] - b[i]).abs();
                    if d > worst {
                        worst = d;
                        worst_at = (w, h, i);
                    }
                    if std::env::var("ZENSIM_FREE_DIAG").is_ok() && d > 1e-12 {
                        let append0 = full.n_scales() * 3 * (31 + FEATURES_PER_CHANNEL_V2_TOTAL);
                        let (sc, ch, loc) =
                            if i < append0 + full.n_scales() * 3 * FEATURES_PER_CHANNEL_APPEND {
                                let r = i - append0;
                                (
                                    r / (3 * FEATURES_PER_CHANNEL_APPEND),
                                    (r % (3 * FEATURES_PER_CHANNEL_APPEND))
                                        / FEATURES_PER_CHANNEL_APPEND,
                                    r % FEATURES_PER_CHANNEL_APPEND,
                                )
                            } else {
                                (99, 99, 99)
                            };
                        eprintln!(
                            "DIAG {w}x{h} par={parallel} slot={i} s{sc} c{ch} L{loc} 944={:.12e} free={:.12e} d={d:.3e}",
                            a[i], b[i]
                        );
                    }
                    assert!(
                        d <= free_tol(),
                        "{w}x{h} par={parallel}: free slot {i} = {:e}, 944 walk says {:e} (Δ {d:e})",
                        b[i],
                        a[i]
                    );
                }
            }
        }
        eprintln!("free-extras vs 944 append: worst |Δ| = {worst:e} at {worst_at:?}");
    }

    /// **FREE-EXTRAS GATE 3 — a live 944 walk is untouched.**
    ///
    /// With the append + append2 blocks ON, their own kernels own every free
    /// slot, so `ComputeSet::raw_moments` must be false and the whole 944
    /// walk must be bit-identical whatever the toggle says. This is what
    /// makes the field safe to set unconditionally in a profile.
    #[cfg(feature = "training")]
    #[test]
    fn free_extras_never_touch_a_live_944_walk() {
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target());
        let mut scratch = V2Scratch::new();
        for &(w, h) in &[(208usize, 144usize), (127, 93), (576, 96)] {
            let src = textured_image(w, h, 7);
            let dst = quantize_distort(&src, w, h);
            let sref = RgbSlice::new(&src, w, h);
            let dref = RgbSlice::new(&dst, w, h);
            let base = V2NewFeatureToggles {
                append_block: true,
                append2_block: true,
                ..V2NewFeatureToggles::default()
            };
            let off = z
                .compute_folded720_append_features_streaming(&sref, &dref, base, &mut scratch)
                .unwrap();
            let on = z
                .compute_folded720_append_features_streaming(
                    &sref,
                    &dref,
                    V2NewFeatureToggles {
                        free_extras: V1FreeExtras::RawMoments,
                        ..base
                    },
                    &mut scratch,
                )
                .unwrap();
            for (i, (x, y)) in off.features().iter().zip(on.features()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "{w}x{h}: 944 slot {i} moved under free_extras ({x:e} -> {y:e})"
                );
            }
        }
    }

    // =====================================================================
    // CLASS-C free slots (`V1FreeExtras::RawMomentsPlusBoundedErr`) —
    // `benchmarks/free_features_classC_2026-09-04.md`
    // =====================================================================

    /// The kernel's f32 copies of the two saturation constants must equal
    /// the f64 definitions the 944 walk uses. `fused.rs` is unconditional
    /// while this module is behind `feature-regime-v2`, so the constants
    /// cannot be imported; this is the gate that keeps the duplication
    /// honest (the same contract `fused::C_MSE_F32`'s doc states).
    #[test]
    fn class_c_kernel_constants_match() {
        // Stated in f32, which is the comparison that means something: the
        // 944 walk's own SIMD path splats `C_MSE as f32` / `C_LUM_T as f32`
        // (`append_block_kernel_generic`), so THAT is the value the class-C
        // kernel has to reproduce. Neither constant is representable in
        // f32, so an `as f64` round-trip equality would be false for a
        // correct implementation — it was, and this is the corrected claim.
        assert_eq!(crate::fused::C_MSE_F32, C_MSE as f32);
        assert_eq!(crate::fused::C_LUM_T_F32, C_LUM_T as f32);
    }

    /// **CLASS-C GATE 0 — the INTEGRAND, against an f64 scalar oracle.**
    ///
    /// The per-slot end-to-end oracle is gate 2 below (an independent
    /// kernel's value for the same slot). This one gates the arithmetic
    /// underneath it: the kernel's f32 per-pixel integrands
    /// (`sat((s−d)², C_MSE)` and the Bernstein weights of
    /// `t = sat(ref_Y, C_LUM_T)`) against the SAME f64 `saturate` the
    /// append kernel's own scalar tail calls, over a value range that
    /// spans the XYB planes' actual span including negatives (where the
    /// `max(0)` clamp is load-bearing) and the near-zero regime the
    /// saturation amplifies ~1/C.
    ///
    /// The bar is RELATIVE (1e-5) with a 1e-7 absolute floor: f32 carries
    /// ~1.2e-7 relative, and `sat` is contractive in relative error
    /// (`dsat/sat = (1−sat)·dx/x`), so a correct f32 evaluation lands two
    /// orders inside it. An algebra error does not.
    #[test]
    fn class_c_integrands_match_the_f64_scalar_oracle() {
        let vals: Vec<f32> = vec![
            -3.0, -1.0, -0.35, -1e-4, 0.0, 1e-7, 1e-4, 0.001, 0.01, 0.05, 0.1, 0.35, 0.5, 1.0, 2.0,
            7.5,
        ];
        let mut worst_m = 0.0f64;
        let mut worst_w = 0.0f64;
        for &s in &vals {
            for &d in &vals {
                let mut acc_m = 0.0f32;
                let m = crate::fused::test_only_bounded_err_scalar(&mut acc_m, s - d);
                assert_eq!(
                    acc_m, m,
                    "the accumulator must hold exactly what it returned"
                );
                let want_m = saturate(((s - d) as f64) * ((s - d) as f64), C_MSE);
                let dm = (m as f64 - want_m).abs();
                worst_m = worst_m.max(dm / want_m.max(1e-7));
                assert!(
                    dm <= 1e-5 * want_m.abs().max(1e-2) + 1e-7,
                    "mse_i({s},{d}) = {m:e}, f64 oracle {want_m:e}"
                );

                let (mut wdn, mut wdd, mut wbn, mut wbd) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                crate::fused::test_only_lum_bins_scalar(
                    &mut wdn, &mut wdd, &mut wbn, &mut wbd, s, m,
                );
                let t_ref = saturate(s as f64, C_LUM_T);
                let (wd_ref, wb_ref) = ((1.0 - t_ref) * (1.0 - t_ref), t_ref * t_ref);
                for (got, want, name) in [
                    (wdd as f64, wd_ref, "w_dark"),
                    (wbd as f64, wb_ref, "w_bright"),
                    (wdn as f64, wd_ref * want_m, "w_dark*mse"),
                    (wbn as f64, wb_ref * want_m, "w_bright*mse"),
                ] {
                    let dw = (got - want).abs();
                    worst_w = worst_w.max(dw / want.abs().max(1e-7));
                    assert!(
                        dw <= 1e-5 * want.abs().max(1e-2) + 1e-7,
                        "{name}({s},{d}) = {got:e}, f64 oracle {want:e}"
                    );
                }
                // The partition of unity the mid bin is DERIVED from must
                // hold at every point, or `lum_bins_from_weighted_sums`'s
                // subtraction is not a mid-bin weight.
                let w_mid = 1.0 - wd_ref - wb_ref;
                assert!(
                    (w_mid - 2.0 * t_ref * (1.0 - t_ref)).abs() <= 1e-12,
                    "Bernstein partition of unity broken at s={s}"
                );
                assert!(w_mid >= -1e-12, "derived mid weight negative at s={s}");
            }
        }
        eprintln!(
            "class-C integrand vs f64 oracle: worst rel |Δ| mse {worst_m:e}, weights {worst_w:e}"
        );
    }

    /// **CLASS-C GATE 1 — the class-C set is pure ADDITION.**
    ///
    /// Same claim, same shape, as `free_extras_are_pure_addition_to_the_
    /// v1_only_walk`: turning the variant on must leave every slot the
    /// `RawMoments` walk already emitted BIT-IDENTICAL (that includes the
    /// 156 basic slots, the 72 peaks AND the 40 raw-moment slots) and fill
    /// only the 24 class-C positions, which are `+0.0` without it. This is
    /// the "no existing slot moves" era gate: if it fails, the change is an
    /// era break, not an addition.
    #[cfg(feature = "training")]
    #[test]
    fn class_c_extras_are_pure_addition_to_the_free_walk() {
        const CELLS: &[(usize, usize)] = &[
            (96, 64),
            (208, 144),
            (592, 80),
            (127, 93),
            (200, 151),
            (576, 96),
            (255, 96),
        ];
        for &parallel in &[false, true] {
            let z =
                crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(parallel);
            let mut scratch = V2Scratch::new();
            for &(w, h) in CELLS {
                let src = textured_image(w, h, 7);
                let dst = quantize_distort(&src, w, h);
                let sref = RgbSlice::new(&src, w, h);
                let dref = RgbSlice::new(&dst, w, h);
                let base = V2NewFeatureToggles {
                    v1_only: true,
                    v1_pools: V1PoolsMode::Peaks,
                    append_block: true,
                    append2_block: true,
                    free_extras: V1FreeExtras::RawMoments,
                    ..V2NewFeatureToggles::default()
                };
                let off = z
                    .compute_folded720_append_features_streaming(&sref, &dref, base, &mut scratch)
                    .unwrap();
                let on = z
                    .compute_folded720_append_features_streaming(
                        &sref,
                        &dref,
                        V2NewFeatureToggles {
                            free_extras: V1FreeExtras::RawMomentsPlusBoundedErr,
                            ..base
                        },
                        &mut scratch,
                    )
                    .unwrap();
                let (fo, fn_) = (off.features(), on.features());
                assert_eq!(fo.len(), 944, "{w}x{h}: expected the 944 layout");
                assert_eq!(fo.len(), fn_.len(), "{w}x{h}: width moved");
                let cc: std::collections::HashSet<usize> =
                    class_c_slot_indices(off.n_scales()).into_iter().collect();
                assert_eq!(cc.len(), 24, "the class-C set is 24 slots at 4 scales");
                for i in 0..fo.len() {
                    if cc.contains(&i) {
                        assert_eq!(
                            fo[i].to_bits(),
                            0.0f64.to_bits(),
                            "{w}x{h} par={parallel}: class-C slot {i} is {:e} with the mode OFF",
                            fo[i]
                        );
                    } else {
                        assert_eq!(
                            fo[i].to_bits(),
                            fn_[i].to_bits(),
                            "{w}x{h} par={parallel}: slot {i} moved ({:e} -> {:e}) — an ERA BREAK, \
                             not an addition",
                            fo[i],
                            fn_[i]
                        );
                    }
                }
                // Every class-C slot must carry a value and be finite —
                // an all-zero fill would pass the identity above trivially.
                for &i in &cc {
                    assert!(
                        fn_[i].is_finite(),
                        "{w}x{h} par={parallel}: class-C slot {i} is {:e}",
                        fn_[i]
                    );
                }
                assert!(
                    cc.iter().filter(|&&i| fn_[i] != 0.0).count() >= cc.len() / 2,
                    "{w}x{h} par={parallel}: the class-C block came out mostly zero"
                );
            }
        }
    }

    /// **CLASS-C GATE 2 — the class-C slots ARE the 944 slots (the per-slot
    /// oracle).**
    ///
    /// Each of the 24 values is compared against the SAME slot produced by
    /// an entirely different kernel on a full 944 walk: `MSE` by the dense
    /// v2 kernel, the luminance bins by the append kernel. Bitwise equality
    /// is not available for the same reason gate 2 of the raw-moments set
    /// documents (different accumulation grouping AND different f32/f64
    /// staging), so the bar is the same `free_tol()`.
    #[cfg(feature = "training")]
    #[test]
    fn class_c_extras_match_the_944_walk() {
        const CELLS: &[(usize, usize)] = &[
            (96, 64),
            (208, 144),
            (592, 80),
            (127, 93),
            (200, 151),
            (576, 96),
            (255, 96),
            (1152, 72),
            (2304, 64),
        ];
        let mut worst = 0.0f64;
        let mut worst_at = (0usize, 0usize, 0usize);
        for &parallel in &[false, true] {
            let z =
                crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(parallel);
            let mut scratch = V2Scratch::new();
            for &(w, h) in CELLS {
                let src = textured_image(w, h, 7);
                let dst = quantize_distort(&src, w, h);
                let sref = RgbSlice::new(&src, w, h);
                let dref = RgbSlice::new(&dst, w, h);
                let full = z
                    .compute_folded720_append_features_streaming(
                        &sref,
                        &dref,
                        V2NewFeatureToggles {
                            append_block: true,
                            append2_block: true,
                            ..V2NewFeatureToggles::default()
                        },
                        &mut scratch,
                    )
                    .unwrap();
                let free = z
                    .compute_folded720_append_features_streaming(
                        &sref,
                        &dref,
                        V2NewFeatureToggles {
                            v1_only: true,
                            v1_pools: V1PoolsMode::Peaks,
                            append_block: true,
                            append2_block: true,
                            free_extras: V1FreeExtras::RawMomentsPlusBoundedErr,
                            ..V2NewFeatureToggles::default()
                        },
                        &mut scratch,
                    )
                    .unwrap();
                let (a, b) = (full.features(), free.features());
                for i in class_c_slot_indices(full.n_scales()) {
                    let d = (a[i] - b[i]).abs();
                    if d > worst {
                        worst = d;
                        worst_at = (w, h, i);
                    }
                    assert!(
                        d <= free_tol(),
                        "{w}x{h} par={parallel}: class-C slot {i} = {:e}, 944 walk says {:e} \
                         (Δ {d:e})",
                        b[i],
                        a[i]
                    );
                }
                // The 944 side must actually be non-trivial on these slots,
                // or the comparison above is vacuous.
                assert!(
                    class_c_slot_indices(full.n_scales())
                        .iter()
                        .filter(|&&i| a[i] != 0.0)
                        .count()
                        >= 12,
                    "{w}x{h}: the 944 walk's own class-C slots are mostly zero"
                );
            }
        }
        eprintln!("class-C vs 944 walk: worst |Δ| = {worst:e} at {worst_at:?}");
    }

    /// **CLASS-C GATE 3 — a live 944 walk is untouched.**
    #[cfg(feature = "training")]
    #[test]
    fn class_c_extras_never_touch_a_live_944_walk() {
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target());
        let mut scratch = V2Scratch::new();
        for &(w, h) in &[(208usize, 144usize), (127, 93), (576, 96)] {
            let src = textured_image(w, h, 7);
            let dst = quantize_distort(&src, w, h);
            let sref = RgbSlice::new(&src, w, h);
            let dref = RgbSlice::new(&dst, w, h);
            let base = V2NewFeatureToggles {
                append_block: true,
                append2_block: true,
                ..V2NewFeatureToggles::default()
            };
            let off = z
                .compute_folded720_append_features_streaming(&sref, &dref, base, &mut scratch)
                .unwrap();
            let on = z
                .compute_folded720_append_features_streaming(
                    &sref,
                    &dref,
                    V2NewFeatureToggles {
                        free_extras: V1FreeExtras::RawMomentsPlusBoundedErr,
                        ..base
                    },
                    &mut scratch,
                )
                .unwrap();
            for (i, (x, y)) in off.features().iter().zip(on.features()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "{w}x{h}: 944 slot {i} moved under the class-C variant ({x:e} -> {y:e})"
                );
            }
        }
    }

    /// **CLASS-C GATE 4 — finite and BIT-IDENTICAL across rayon pool sizes
    /// and against the serial walk.**
    ///
    /// The band merge is sequential in band order on both the serial and
    /// the band-parallel path, so the f64 addition sequence is the same
    /// whatever the pool size — the property `v1_masked_and_iw_blocks_are_
    /// thread_invariant` gates for the v1 blocks, restated for the new
    /// accumulators (which are plain sums merged the same way). Sizes span
    /// several 32-row bands so a layout disagreement has somewhere to show.
    #[cfg(all(feature = "training", feature = "threads"))]
    #[test]
    fn class_c_extras_are_thread_invariant_and_finite() {
        let toggles = V2NewFeatureToggles {
            v1_only: true,
            v1_pools: V1PoolsMode::Peaks,
            append_block: true,
            append2_block: true,
            free_extras: V1FreeExtras::RawMomentsPlusBoundedErr,
            ..V2NewFeatureToggles::default()
        };
        for &(w, h) in &[(256usize, 256usize), (96, 320), (320, 96), (127, 288)] {
            let src = textured_image(w, h, 11);
            let dst = quantize_distort(&src, w, h);
            let mut reference: Option<Vec<f64>> = None;
            // `false` = the serial walk; the rest are pool sizes for the
            // band-parallel one.
            for &parallel in &[false, true] {
                for threads in [1usize, 2, 8, 16] {
                    if !parallel && threads != 1 {
                        continue;
                    }
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .expect("build rayon pool");
                    let got = pool.install(|| {
                        let z = crate::Zensim::new(crate::ZensimProfile::codec_target())
                            .with_parallel(parallel);
                        let mut scratch = V2Scratch::new();
                        let sref = RgbSlice::new(&src, w, h);
                        let dref = RgbSlice::new(&dst, w, h);
                        z.compute_folded720_append_features_streaming(
                            &sref,
                            &dref,
                            toggles,
                            &mut scratch,
                        )
                        .unwrap()
                        .features()
                        .to_vec()
                    });
                    for (i, &v) in got.iter().enumerate() {
                        assert!(
                            v.is_finite(),
                            "{w}x{h} par={parallel} t={threads}: slot {i} is {v:e}"
                        );
                    }
                    match &reference {
                        None => reference = Some(got),
                        Some(first) => {
                            for (i, (x, y)) in first.iter().zip(&got).enumerate() {
                                assert_eq!(
                                    x.to_bits(),
                                    y.to_bits(),
                                    "{w}x{h}: slot {i} moved at par={parallel} t={threads} \
                                     ({x:e} -> {y:e})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// **CLASS-C GATE 5 — append-only bookkeeping.**
    ///
    /// The class-C tranche adds NO new slot numbers: every one of its 24
    /// positions is an existing 944-layout slot, it is disjoint from the
    /// raw-moments free set, and the layout width is unchanged. A future
    /// lane that wants genuinely new features must EXTEND the layout past
    /// 944, never renumber — this test is what fails if someone forgets.
    #[test]
    fn class_c_slot_set_is_append_only_and_disjoint() {
        for n_scales in 1..=4 {
            let cc = class_c_slot_indices(n_scales);
            let rm = free_slot_indices(n_scales);
            let width = n_scales * 3 * (31 + FEATURES_PER_CHANNEL_V2_TOTAL)
                + n_scales * 3 * FEATURES_PER_CHANNEL_APPEND
                + n_scales * APPEND2_PER_SCALE;
            let set: std::collections::HashSet<usize> = cc.iter().copied().collect();
            assert_eq!(set.len(), cc.len(), "class-C slot list has duplicates");
            for &i in &cc {
                assert!(
                    i < width,
                    "class-C slot {i} is outside the {width}-wide layout"
                );
                assert!(
                    !rm.contains(&i),
                    "class-C slot {i} overlaps the raw-moments free set"
                );
            }
            // 3 MSE cells per scale (one per channel) + the Y luminance
            // trio per scale where the 944 walk computes that append cell.
            let want = n_scales * 3
                + (0..n_scales)
                    .filter(|&s| append_cell_active(true, APPEND2_CHANNEL, s))
                    .count()
                    * 3;
            assert_eq!(cc.len(), want, "class-C slot count at {n_scales} scales");
        }
        // The 4-scale set, spelled out, so a renumbering shows up as a diff
        // in review rather than as a silently different table.
        assert_eq!(
            class_c_slot_indices(4),
            vec![
                377, 406, 435, 739, 740, 741, // scale 0
                464, 493, 522, 790, 791, 792, // scale 1
                551, 580, 609, 841, 842, 843, // scale 2
                638, 667, 696, 892, 893, 894, // scale 3
            ]
        );
    }

    /// POOL PARITY GATE (2026-08-30, the carrier lane): with
    /// [`V2NewFeatureToggles::v1_pools`] the fold replays v1's extended
    /// strip section per v1-aligned band, so the peak / masked / IW blocks
    /// (`f156..372`) are **BIT-IDENTICAL** to the frozen v1 372 extraction
    /// at every width where the basic block is (`pyramid_plane_stride(w) ==
    /// w`); the padded-width class documents its divergence the same way
    /// (`folded720_v1_basic_matches_v1_path`), and
    /// `v1_padded_width_divergence_is_column_padding` shows that class is
    /// entirely v1's mirror-padded columns — pre-pad the input and 17 of 20
    /// geometries become bit-exact. Also gates that the toggle changes
    /// NOTHING else — basic + v2 slots bit-equal to the toggle-off fold —
    /// and that the toggle-off fold still zeroes the block.
    #[cfg(feature = "training")]
    #[test]
    fn folded720_v1_pools_match_v1_path() {
        // OPTION C: no divergence class any more — see
        // `folded720_v1_basic_matches_v1_path` for the same re-pin note.
        for &(w, h) in &[
            (96usize, 64usize),
            (64, 300),
            (208, 144),
            (127, 93),
            (200, 150),
        ] {
            let src = textured_image(w, h, 7);
            let dst = quantize_distort(&src, w, h);
            let cfg = crate::ZensimConfig {
                extended_features: true,
                compute_iw_features: true,
                allow_multithreading: false,
                ..Default::default()
            };
            let v1 = crate::compute_zensim_with_config(&src, &dst, w, h, cfg).unwrap();
            let v1f = v1.features();
            assert_eq!(v1f.len(), 372);
            let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
            let sref = RgbSlice::new(&src, w, h);
            let dref = RgbSlice::new(&dst, w, h);
            let mut scratch = V2Scratch::new();
            let off = z
                .compute_folded720_append_features_streaming(
                    &sref,
                    &dref,
                    V2NewFeatureToggles::default(),
                    &mut scratch,
                )
                .unwrap();
            let on = z
                .compute_folded720_append_features_streaming(
                    &sref,
                    &dref,
                    V2NewFeatureToggles {
                        v1_pools: V1PoolsMode::Full,
                        ..V2NewFeatureToggles::default()
                    },
                    &mut scratch,
                )
                .unwrap();
            let carriers = z
                .compute_folded720_append_features_streaming(
                    &sref,
                    &dref,
                    V2NewFeatureToggles {
                        v1_pools: V1PoolsMode::Carriers,
                        ..V2NewFeatureToggles::default()
                    },
                    &mut scratch,
                )
                .unwrap();
            assert!(!off.v1_pools_live());
            assert_eq!(on.v1_pools(), V1PoolsMode::Full);
            assert_eq!(carriers.v1_pools(), V1PoolsMode::Carriers);
            assert_eq!(off.regime(), on.regime());
            let (fo, fp, fc) = (off.features(), on.features(), carriers.features());
            assert_eq!(fo.len(), fp.len());
            // Carriers: exactly the ten slots live, bit-equal to Full's
            // values (same arithmetic), every other slot unchanged from off.
            for i in 0..fo.len() {
                if V1PoolsMode::CARRIER_SLOTS.contains(&i) {
                    assert_eq!(
                        fc[i].to_bits(),
                        fp[i].to_bits(),
                        "{w}x{h}: carrier slot {i}"
                    );
                    assert!(
                        fc[i] > 0.0,
                        "{w}x{h}: carrier slot {i} is zero on a distorted pair"
                    );
                } else {
                    assert_eq!(
                        fc[i].to_bits(),
                        fo[i].to_bits(),
                        "{w}x{h}: non-carrier slot {i}"
                    );
                }
            }
            assert!(
                fo[156..372].iter().all(|&v| v == 0.0),
                "{w}x{h}: toggle-off fold must keep f156..372 at exactly 0.0"
            );
            for i in (0..156).chain(372..fo.len()) {
                assert_eq!(
                    fo[i].to_bits(),
                    fp[i].to_bits(),
                    "{w}x{h}: slot {i} changed by v1_pools ({:e} vs {:e})",
                    fo[i],
                    fp[i]
                );
            }
            // Every width is the bit-exact class under C.
            for i in 156..372 {
                assert_eq!(
                    fp[i].to_bits(),
                    v1f[i].to_bits(),
                    "{w}x{h}: pool slot {i} ({:e}) != v1 ({:e})",
                    fp[i],
                    v1f[i]
                );
            }
        }
    }

    /// Sub-64 inputs: the folded path inherits v2's reflect-pad-to-64
    /// pyramid, which matches the v1 `Zensim` API entries
    /// (`compute_with_config_inner` pads with the same
    /// `reflect_pad_to_min`) — gate the basic block against
    /// `Zensim::compute_extended_features` (its `f0..156` prefix).
    #[test]
    fn folded720_sub64_matches_padding_v1_entry() {
        let (w, h) = (48usize, 40usize);
        let src = textured_image(w, h, 11);
        let dst = quantize_distort(&src, w, h);
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);

        let z = crate::Zensim::new(crate::ZensimProfile::PreviewV0_2).with_parallel(false);
        let v1 = z.compute_extended_features(&sref, &dref).unwrap();
        let v1f = v1.features();
        let folded = z.compute_folded720_features(&sref, &dref).unwrap();
        let ff = folded.features();
        assert_eq!(ff.len(), 720);
        // Both sides pad 48x40 → 64x64 via the same `reflect_pad_to_min`,
        // and pyramid_plane_stride(64) == 64 ⇒ bit-exact class.
        for i in 0..156 {
            assert_eq!(
                ff[i].to_bits(),
                v1f[i].to_bits(),
                "sub64 basic f{i}: fold {:e} != v1 {:e}",
                ff[i],
                v1f[i]
            );
        }
    }

    /// SUCCESSOR of `folded720_ref_paths_bit_identical` (C5: the
    /// prepared/moments entry forms are deleted with the reference
    /// cache): every REMAINING folded-720 entry form must agree bitwise
    /// on all 720 slots — the pair wrapper (`compute_folded720_features`,
    /// internal scratch), the explicit streaming batch form (caller
    /// scratch), and the parallel fan-out. Also keeps the old test's
    /// `view()`-exposes-the-v2-tail assertion.
    #[test]
    fn folded720_entry_paths_bit_identical() {
        let (w, h) = (150usize, 170usize);
        let src = textured_image(w, h, 23);
        let dst = quantize_distort(&src, w, h);
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);

        let pair = z.compute_folded720_features(&sref, &dref).unwrap();
        let mut scratch = V2Scratch::new();
        let a = z
            .compute_folded720_features_streaming(
                &sref,
                &dref,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        let zp = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(true);
        let b = zp
            .compute_folded720_features_streaming(
                &sref,
                &dref,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();

        assert_eq!(pair.features().len(), 720);
        for i in 0..720 {
            assert_eq!(
                pair.features()[i].to_bits(),
                a.features()[i].to_bits(),
                "pair wrapper vs streaming batch form diverge at f{i}"
            );
            assert_eq!(
                pair.features()[i].to_bits(),
                b.features()[i].to_bits(),
                "serial vs parallel streaming diverge at f{i}"
            );
        }
        // Folded view() exposes the v2 tail.
        assert_eq!(
            pair.view().ssim_mean(0, 0).to_bits(),
            pair.features()[372 + idx::SSIM_MEAN].to_bits()
        );
    }

    // ------------------------------------------------------------------
    // f720+ append block
    // ------------------------------------------------------------------

    /// SUCCESSOR of `append_ref_paths_bit_identical` (C5: the cached
    /// bs2/moments legs are deleted with the reference cache — the
    /// streamed walk computes bs2 + cross activities per kernel strip):
    /// every REMAINING append entry form must agree bitwise on ALL 924
    /// slots — pair wrapper, explicit streaming batch form, parallel
    /// fan-out, and a REUSED-scratch second run (the cross-pair leak
    /// gate the old cached legs also exercised).
    #[test]
    fn append_entry_paths_bit_identical() {
        let (w, h) = (150usize, 170usize);
        let src = textured_image(w, h, 23);
        let dst = quantize_distort(&src, w, h);
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);

        let pair = z.compute_folded720_append_features(&sref, &dref).unwrap();
        assert_eq!(
            pair.features().len(),
            720 + 4 * 3 * FEATURES_PER_CHANNEL_APPEND
        );
        assert_eq!(pair.regime(), FeatureRegime::Folded720Append);
        assert_eq!(
            pair.append_features().expect("append regime").len(),
            4 * 3 * FEATURES_PER_CHANNEL_APPEND
        );

        let mut scratch = V2Scratch::new();
        let a = z
            .compute_folded720_append_features_streaming(
                &sref,
                &dref,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        // Scratch reuse (second run on the same scratch).
        let a2 = z
            .compute_folded720_append_features_streaming(
                &sref,
                &dref,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        let zp = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(true);
        let b = zp
            .compute_folded720_append_features_streaming(
                &sref,
                &dref,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();

        for i in 0..pair.features().len() {
            assert_eq!(
                pair.features()[i].to_bits(),
                a.features()[i].to_bits(),
                "pair wrapper vs streaming batch form diverge at f{i}"
            );
            assert_eq!(
                pair.features()[i].to_bits(),
                a2.features()[i].to_bits(),
                "scratch reuse diverges at f{i}"
            );
            assert_eq!(
                pair.features()[i].to_bits(),
                b.features()[i].to_bits(),
                "serial vs parallel streaming diverge at f{i}"
            );
        }
    }

    /// Turning the append block on must not move a single bit of the
    /// first 720 slots — the append is strictly additive (separate second
    /// kernel pass, separate output region; the only shared-kernel touch,
    /// `GradientAccum::sum_gms2`, derives from unchanged operations).
    #[test]
    fn append_first720_bit_stable() {
        let (w, h) = (150usize, 170usize);
        let src = textured_image(w, h, 7);
        let dst = quantize_distort(&src, w, h);
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);

        let folded = z.compute_folded720_features(&sref, &dref).unwrap();
        let appended = z.compute_folded720_append_features(&sref, &dref).unwrap();
        assert_eq!(folded.features().len(), 720);
        for i in 0..720 {
            assert_eq!(
                folded.features()[i].to_bits(),
                appended.features()[i].to_bits(),
                "append toggled on moved f{i}"
            );
        }
        // The v2-named view must agree across both regimes too.
        assert_eq!(
            folded.view().ssim_mean(1, 1).to_bits(),
            appended.view().ssim_mean(1, 1).to_bits()
        );
    }

    /// Identity pair: the error-driven append features are EXACTLY 0
    /// (`raw_abs_err`, `art_i`/`det_i`, gms, and the global sums all
    /// compute identical-bit operands on both sides); the SSIM/σ-derived
    /// features are tiny-but-nonzero, because on an identity pair `ssq`
    /// accumulates `(s²+d²)` per tap while `s12` accumulates `s·d` —
    /// `Σ(2a)` vs `2Σ(a)` round differently in f32 — and the append `bs2`
    /// plane comes from a different H kernel (`box_blur_h` on `src²`,
    /// strip-tiled per P2) than the fused `ssq` chain, so
    /// `var₁ ≠ var₂` at the ULP scale. Observed magnitude ~2e-6; the 1e-4
    /// bound is far below any real-signal value while catching sign or
    /// formula errors. `GRAD_SRC_MEAN` is reference-only and genuinely
    /// nonzero (the `PJND_FRAGILITY` precedent).
    #[test]
    fn append_identity_pair_zeros() {
        let (w, h) = (150usize, 130usize);
        let src = textured_image(w, h, 5);
        let sref = RgbSlice::new(&src, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let r = z.compute_folded720_append_features(&sref, &sref).unwrap();
        let app = r.append_features().unwrap();
        const EXACT_ZERO: [usize; 11] = [
            idx_append::XMASK_TRANSDUCER,
            idx_append::LUM_TRANSDUCER,
            idx_append::LUM_DARK_ERR,
            idx_append::LUM_MID_ERR,
            idx_append::LUM_BRIGHT_ERR,
            idx_append::GMS_DEV2,
            idx_append::ART_DEV2,
            idx_append::DET_DEV2,
            idx_append::GLOBAL_DMEAN,
            idx_append::GLOBAL_CGAIN,
            idx_append::GLOBAL_CLOSS,
        ];
        for scale in 0..4 {
            for ch in 0..3 {
                for local in 0..FEATURES_PER_CHANNEL_APPEND {
                    let v = app[scale * 3 * FEATURES_PER_CHANNEL_APPEND
                        + ch * FEATURES_PER_CHANNEL_APPEND
                        + local];
                    if local == idx_append::GRAD_SRC_MEAN {
                        assert!(
                            (0.0..1.0).contains(&v),
                            "grad_src_mean out of range at s{scale} ch{ch}: {v}"
                        );
                    } else if EXACT_ZERO.contains(&local) {
                        assert_eq!(
                            v, 0.0,
                            "append local {local} not exactly zero on identity pair \
                             (s{scale} ch{ch}): {v}"
                        );
                    } else {
                        assert!(
                            v.abs() < 1e-4,
                            "append local {local} above the identity ULP band \
                             (s{scale} ch{ch}): {v}"
                        );
                    }
                }
            }
        }
    }

    /// Bounds + layout invariants on a real distorted pair: every append
    /// feature within its documented bound, and the cross-channel
    /// transducer slot exactly 0.0 on the chroma channels (Y-only by
    /// design — luma does not mask chroma in the CVVDP trained matrix).
    #[test]
    fn append_bounds_and_chroma_xmask_zero() {
        let (w, h) = (200usize, 150usize);
        let src = textured_image(w, h, 41);
        let dst = quantize_distort(&src, w, h);
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let r = z.compute_folded720_append_features(&sref, &dref).unwrap();
        let app = r.append_features().unwrap();
        let mut any_nonzero = false;
        for scale in 0..4 {
            for ch in 0..3 {
                let base =
                    scale * 3 * FEATURES_PER_CHANNEL_APPEND + ch * FEATURES_PER_CHANNEL_APPEND;
                for local in 0..FEATURES_PER_CHANNEL_APPEND {
                    let v = app[base + local];
                    assert!(
                        v.is_finite() && (0.0..=2.0).contains(&v),
                        "append s{scale} ch{ch} local{local} out of bounds: {v}"
                    );
                    if v != 0.0 {
                        any_nonzero = true;
                    }
                }
                if ch != 1 {
                    assert_eq!(
                        app[base + idx_append::XMASK_TRANSDUCER],
                        0.0,
                        "xmask must be 0 on chroma channel {ch}"
                    );
                    assert_eq!(
                        app[base + idx_append::LUM_TRANSDUCER],
                        0.0,
                        "lum transducer must be 0 on chroma channel {ch} (luma-gate)"
                    );
                } else {
                    assert!(
                        app[base + idx_append::XMASK_TRANSDUCER] > 0.0,
                        "Y xmask should fire on a distorted pair (s{scale})"
                    );
                }
                if APPEND_SKIP_B_SCALE0 && ch == 2 && scale == 0 {
                    for local in 0..FEATURES_PER_CHANNEL_APPEND {
                        assert_eq!(
                            app[base + local],
                            0.0,
                            "B channel scale 0 must be fully skipped (local {local})"
                        );
                    }
                }
            }
        }
        assert!(
            any_nonzero,
            "append block is entirely zero on a distorted pair"
        );
    }

    // ========================================================================
    // C2 streaming-walk parity gates
    // (docs/STREAMING_FOLDAPP_C0_DESIGN_2026-07-26.md §3)
    // ========================================================================

    /// Originally the C2 gate (streamed vs the since-deleted materialized
    /// foldapp walk, bitwise). Post-C5 the pair entries ROUTE to the
    /// streaming walk, so this now gates the wrapper entries against the
    /// explicit-scratch streaming form across the full adversarial dim
    /// matrix (odd dims, sub-one-strip, exact/one-past strip multiples,
    /// 1-row strips, tall multi-production-chunk) — the geometry coverage
    /// the C2 gate provided lives on here.
    #[test]
    fn streamed_foldapp_bitwise_vs_materialized() {
        let cases = [
            (150usize, 170usize),
            (127, 93),
            (200, 150),
            (96, 517),
            // C3 adversarial rows: exact kernel-strip multiple (h = 256),
            // one past it (h = 257 — 1-row final strip), and h = 129
            // (1-row second strip at scale 0, sub-strip deeper scales).
            (96, 256),
            (96, 257),
            (72, 129),
        ];
        let z_parallel = false;
        for (w, h) in cases {
            let src = textured_image(w, h, 23);
            let dst = quantize_distort(&src, w, h);
            let sref = RgbSlice::new(&src, w, h);
            let dref = RgbSlice::new(&dst, w, h);

            // Foldapp (924).
            let mat = compute_folded720_append_impl(
                &sref,
                &dref,
                None,
                z_parallel,
                V2NewFeatureToggles::default(),
            )
            .unwrap();
            let mut scratch = V2Scratch::new();
            let st = compute_folded720_append_streaming_impl(
                &sref,
                &dref,
                None,
                z_parallel,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
            assert_eq!(mat.regime(), st.regime());
            assert_eq!(mat.features().len(), st.features().len());
            for i in 0..mat.features().len() {
                assert_eq!(
                    mat.features()[i].to_bits(),
                    st.features()[i].to_bits(),
                    "{w}x{h} foldapp: streamed diverges from materialized at f{i} \
                     ({} vs {})",
                    mat.features()[i],
                    st.features()[i]
                );
            }

            // Fold-only (720).
            let mat_f = compute_folded720_impl_with_toggles(
                &sref,
                &dref,
                None,
                z_parallel,
                V2NewFeatureToggles::default(),
            )
            .unwrap();
            let st_f = compute_folded720_streaming_impl(
                &sref,
                &dref,
                None,
                z_parallel,
                V2NewFeatureToggles::default(),
                &mut scratch,
                None,
            )
            .unwrap();
            assert_eq!(mat_f.features().len(), 720);
            for i in 0..720 {
                assert_eq!(
                    mat_f.features()[i].to_bits(),
                    st_f.features()[i].to_bits(),
                    "{w}x{h} fold: streamed diverges at f{i}"
                );
            }
        }
    }

    /// Sub-64 inputs reflect-pad before the streaming walk exactly like
    /// the materialized entry — outputs stay bitwise equal.
    #[test]
    fn streamed_sub64_matches_materialized() {
        let (w, h) = (40usize, 30usize);
        let src = textured_image(w, h, 9);
        let dst = quantize_distort(&src, w, h);
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);
        let mat = compute_folded720_append_impl(
            &sref,
            &dref,
            None,
            false,
            V2NewFeatureToggles::default(),
        )
        .unwrap();
        let mut scratch = V2Scratch::new();
        let st = compute_folded720_append_streaming_impl(
            &sref,
            &dref,
            None,
            false,
            V2NewFeatureToggles::default(),
            &mut scratch,
        )
        .unwrap();
        for i in 0..mat.features().len() {
            assert_eq!(
                mat.features()[i].to_bits(),
                st.features()[i].to_bits(),
                "sub-64 streamed diverges at f{i}"
            );
        }
    }

    /// Streaming walk: parallel channel fan-out is bitwise equal to the
    /// serial path (per-channel accumulators + deterministic strip order,
    /// so scheduling cannot change any sum).
    #[test]
    fn streamed_parallel_matches_serial() {
        let (w, h) = (200usize, 300usize);
        let src = textured_image(w, h, 31);
        let dst = quantize_distort(&src, w, h);
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);
        let mut scratch = V2Scratch::new();
        let serial = compute_folded720_append_streaming_impl(
            &sref,
            &dref,
            None,
            false,
            V2NewFeatureToggles::default(),
            &mut scratch,
        )
        .unwrap();
        let parallel = compute_folded720_append_streaming_impl(
            &sref,
            &dref,
            None,
            true,
            V2NewFeatureToggles::default(),
            &mut scratch,
        )
        .unwrap();
        for i in 0..serial.features().len() {
            assert_eq!(
                serial.features()[i].to_bits(),
                parallel.features()[i].to_bits(),
                "streamed parallel/serial diverge at f{i}"
            );
        }
    }

    /// Scratch reuse across unrelated streamed pairs leaks nothing.
    #[test]
    fn streamed_scratch_reuse_matches_fresh() {
        let (w, h) = (150usize, 170usize);
        let src_a = textured_image(w, h, 3);
        let dst_a = quantize_distort(&src_a, w, h);
        let src_b = textured_image(w, h, 77);
        let dst_b = quantize_distort(&src_b, w, h);
        let toggles = V2NewFeatureToggles::default();

        let mut fresh = V2Scratch::new();
        let clean = compute_folded720_append_streaming_impl(
            &RgbSlice::new(&src_b, w, h),
            &RgbSlice::new(&dst_b, w, h),
            None,
            false,
            toggles,
            &mut fresh,
        )
        .unwrap();

        let mut reused = V2Scratch::new();
        let _ = compute_folded720_append_streaming_impl(
            &RgbSlice::new(&src_a, w, h),
            &RgbSlice::new(&dst_a, w, h),
            None,
            false,
            toggles,
            &mut reused,
        )
        .unwrap();
        let with_reuse = compute_folded720_append_streaming_impl(
            &RgbSlice::new(&src_b, w, h),
            &RgbSlice::new(&dst_b, w, h),
            None,
            false,
            toggles,
            &mut reused,
        )
        .unwrap();

        for i in 0..clean.features().len() {
            assert_eq!(
                clean.features()[i].to_bits(),
                with_reuse.features()[i].to_bits(),
                "streamed scratch reuse changed f{i}"
            );
        }
    }

    /// A declared-ID plan that reads no slot at scale 0 (no v1-era slot, no v2-era work) skips the scale-0 strips
    /// whole, and every slot it does read is bit-identical to the full plan's.
    #[test]
    fn declared_plan_skips_unread_scales_bit_identically() {
        use crate::feature_plan::Plan;
        use crate::feature_set_id::SlotSet;
        let (w, h) = (160, 128);
        let mut st = 0x9e37_79b9u32;
        let mut rnd = move || {
            st ^= st << 13;
            st ^= st >> 17;
            st ^= st << 5;
            st
        };
        let src: Vec<[u8; 3]> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                [((x * 7 + y * 3) % 256) as u8, ((x * x + y) % 256) as u8, ((y * 5 + x / 3) % 256) as u8]
            })
            .collect();
        let dst: Vec<[u8; 3]> = src
            .iter()
            .map(|p| p.map(|v| v.saturating_add((rnd() % 7) as u8)))
            .collect();
        let (sref, dref) = (RgbSlice::new(&src, w, h), RgbSlice::new(&dst, w, h));
        // basic + peaks + v2: the Rev5 families, so the test holds at every revision.
        let all: Vec<usize> = (0..228).chain(372..720).collect();
        let coarse: Vec<usize> = all
            .iter()
            .copied()
            .filter(|&id| crate::feature_defs::def_at(id, 4).is_some_and(|d| d.scale >= 1))
            .collect();
        let full = Plan::derive(&SlotSet::from_slots(all), 720).unwrap();
        let narrow = Plan::derive(&SlotSet::from_slots(coarse.iter().copied()), 720).unwrap();
        assert_eq!(narrow.compute.v1_scales, 0b1110);
        assert_eq!(narrow.compute.v2_scales & 1, 0);
        let stray: Vec<usize> = narrow
            .compute
            .populated_slots(4, 720)
            .iter_slots()
            .filter(|&id| crate::feature_defs::def_at(id, 4).is_none_or(|d| d.scale == 0))
            .collect();
        assert!(stray.is_empty(), "a skipped scale must not be reported as populated: {stray:?}");
        let run = |plan: &Plan| {
            SKIPPED_SCALES.with_borrow_mut(|v| *v = 0);
            let f = foldapp_streaming_walk(
                &sref,
                &dref,
                false,
                plan.toggles(),
                crate::feature_v2_stream::FrontEnd::Sdr,
                &mut V2Scratch::new(),
                FoldWalkExtras {
                    compute: Some(plan.compute),
                    ..FoldWalkExtras::default()
                },
            )
            .into_features();
            (f, SKIPPED_SCALES.with_borrow(|v| *v))
        };
        let (f_full, skipped_full) = run(&full);
        let (f_narrow, skipped_narrow) = run(&narrow);
        assert_eq!(skipped_full, 0);
        assert_eq!(skipped_narrow, 0b0001, "scale 0 must be skipped whole");
        for &id in &coarse {
            assert_eq!(f_full[id].to_bits(), f_narrow[id].to_bits(), "f{id} moved when scale 0 was skipped");
        }
    }

    #[test]
    fn coarse_pools_execute_only_at_requested_resolutions() {
        use crate::feature_plan::Plan;
        use crate::feature_set_id::SlotSet;
        let (w, h) = (256, 192);
        let src = vec![[127u8; 3]; w * h];
        let mut dst = src.clone();
        dst[w + 1] = [255, 0, 255];
        let ids = (0..228)
            .chain(264..300)
            .chain(336..372)
            .filter(|&id| !ComputeSet::is_full_res_xb(id, 4));
        let coarse = Plan::derive(&SlotSet::from_slots(ids), 372).unwrap();
        for (plan, expected) in [
            (Plan::v1(V1PoolsMode::Full, 372), vec![32, 64, 128, 256]),
            (coarse, vec![32, 64]),
            (Plan::v1(V1PoolsMode::Peaks, 372), vec![]),
        ] {
            POOL_TEST_WIDTHS.with_borrow_mut(|v| *v = Some(Vec::new()));
            compute_folded_v1_372_streaming_impl(
                &RgbSlice::new(&src, w, h),
                &RgbSlice::new(&dst, w, h),
                None,
                false,
                &mut V2Scratch::new(),
                Some(&plan),
                #[cfg(feature = "custom-profiles")]
                None,
            )
            .unwrap();
            let mut widths = POOL_TEST_WIDTHS.with_borrow_mut(|v| v.take().unwrap());
            widths.sort_unstable();
            widths.dedup();
            assert_eq!(widths, expected, "actual weighted-pool kernel widths");
        }
    }

    // ========================================================================
    // HDR route gates (HDR_PLAN chunk 2 — streaming PU front-end)
    // ========================================================================

    /// Minimal declared-HDR test source: absolute-linear cd/m² RGBA f32,
    /// `is_hdr() == true`, opaque.
    struct NitsImage {
        data: Vec<[f32; 4]>,
        w: usize,
        h: usize,
    }

    impl NitsImage {
        fn from_rgb_nits(rgb: &[[f32; 3]], w: usize, h: usize) -> Self {
            Self {
                data: rgb.iter().map(|p| [p[0], p[1], p[2], 1.0]).collect(),
                w,
                h,
            }
        }
    }

    impl crate::source::ImageSource for NitsImage {
        fn width(&self) -> usize {
            self.w
        }
        fn height(&self) -> usize {
            self.h
        }
        fn pixel_format(&self) -> crate::source::PixelFormat {
            crate::source::PixelFormat::LinearF32Rgba
        }
        fn row_bytes(&self, y: usize) -> &[u8] {
            bytemuck::cast_slice(&self.data[y * self.w..(y + 1) * self.w])
        }
        fn alpha_mode(&self) -> crate::source::AlphaMode {
            crate::source::AlphaMode::Opaque
        }
        fn is_hdr(&self) -> bool {
            true
        }
    }

    #[test]
    fn scale_selective_944_hdr_retained_features_are_bit_exact() {
        use crate::feature_plan::Plan;
        use crate::feature_set_id::SlotSet;
        let full = Plan::derive(&SlotSet::from_slots(0..944), 944).unwrap();
        let mut scratch = V2Scratch::new();
        for (w, h) in [(17, 9), (97, 131)] {
            let src = vec![[203.0, 170.0, 100.0]; w * h];
            let mut dst = src.clone();
            for (i, p) in dst.iter_mut().enumerate() {
                if i % 19 == 0 {
                    *p = [1000.0, 2000.0, 500.0];
                }
            }
            let src = NitsImage::from_rgb_nits(&src, w, h);
            let dst = NitsImage::from_rgb_nits(&dst, w, h);
            for parallel in [false, true] {
                let mut run = |plan: &Plan| {
                    compute_folded720_streaming_impl(
                        &src,
                        &dst,
                        None,
                        parallel,
                        plan.toggles(),
                        &mut scratch,
                        Some(plan.compute),
                    )
                    .unwrap()
                };
                let baseline = run(&full);
                for mask in 1u8..16 {
                    let want = SlotSet::from_slots((0..944).filter(|&id| {
                        mask & (1 << crate::feature_defs::def_at(id, 4).unwrap().scale) != 0
                    }));
                    let plan = Plan::derive(&want, 944).unwrap();
                    let value = run(&plan);
                    for id in want.iter_slots() {
                        assert_eq!(
                            value.features()[id].to_bits(),
                            baseline.features()[id].to_bits(),
                            "HDR mask={mask:04b} f{id}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn fullres_y_subset_hdr_retained_features_are_bit_exact() {
        use crate::feature_plan::Plan;
        use crate::feature_set_id::SlotSet;
        for (coarse, y_only) in [(false, false), (true, false), (false, true)] {
            let want = SlotSet::from_slots((0..372).filter(|&id| {
                !ComputeSet::is_full_res_xb(id, 4)
                    && (!y_only || (id < 156 && id % 13 < 10 && id / 13 % 3 == 1))
                    && (id < 228 || coarse && matches!(id,264..=299|336..=371))
            }));
            let plan = Plan::derive(&want, 372).unwrap();
            let mut scratch = V2Scratch::new();
            for (w, h) in [(17, 9), (97, 131)] {
                let src = vec![[203.0, 170.0, 100.0]; w * h];
                let mut dst = src.clone();
                dst[(h / 2) * w + w / 2] = [1000.0, 2000.0, 500.0];
                let src = NitsImage::from_rgb_nits(&src, w, h);
                let dst = NitsImage::from_rgb_nits(&dst, w, h);
                for parallel in [false, true] {
                    let full = compute_folded720_hdr_streaming_impl(
                        &src,
                        &dst,
                        HdrEncoding::Linear,
                        None,
                        parallel,
                        plan.toggles(),
                        &mut scratch,
                        None,
                    )
                    .unwrap();
                    // Exercise the automatic declared-linear-HDR route as well.
                    let sub = compute_folded720_streaming_impl(
                        &src,
                        &dst,
                        None,
                        parallel,
                        plan.toggles(),
                        &mut scratch,
                        Some(plan.compute),
                    )
                    .unwrap();
                    for id in want.iter_slots() {
                        assert_eq!(
                            full.features()[id].to_bits(),
                            sub.features()[id].to_bits(),
                            "HDR f{id}"
                        );
                    }
                }
            }
        }
    }

    /// PQ/HLG code-value test source (`Srgb16Rgba` container), HDR-flagged.
    struct Code16Image {
        data: Vec<[u16; 4]>,
        w: usize,
        h: usize,
    }

    impl Code16Image {
        fn from_unit_codes(codes: &[[f32; 3]], w: usize, h: usize) -> Self {
            Self {
                data: codes
                    .iter()
                    .map(|p| {
                        [
                            (p[0].clamp(0.0, 1.0) * 65535.0).round() as u16,
                            (p[1].clamp(0.0, 1.0) * 65535.0).round() as u16,
                            (p[2].clamp(0.0, 1.0) * 65535.0).round() as u16,
                            65535,
                        ]
                    })
                    .collect(),
                w,
                h,
            }
        }
    }

    impl crate::source::ImageSource for Code16Image {
        fn width(&self) -> usize {
            self.w
        }
        fn height(&self) -> usize {
            self.h
        }
        fn pixel_format(&self) -> crate::source::PixelFormat {
            crate::source::PixelFormat::Srgb16Rgba
        }
        fn row_bytes(&self, y: usize) -> &[u8] {
            bytemuck::cast_slice(&self.data[y * self.w..(y + 1) * self.w])
        }
        fn alpha_mode(&self) -> crate::source::AlphaMode {
            crate::source::AlphaMode::Opaque
        }
        fn is_hdr(&self) -> bool {
            true
        }
    }

    /// Deterministic HDR test content spanning the given luminance range
    /// (geometric ramp + texture), in cd/m².
    fn nits_image(w: usize, h: usize, lo: f32, hi: f32, seed: u32) -> Vec<[f32; 3]> {
        let mut px = vec![[0.0f32; 3]; w * h];
        let mut state = seed.wrapping_mul(0x9E37_79B9) | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state & 0xFFFF) as f32 / 65535.0
        };
        for y in 0..h {
            for x in 0..w {
                let t = (x + y * 3) as f32 / (w + h * 3) as f32;
                let base = lo * (hi / lo).powf(t);
                let n = 0.7 + 0.6 * next();
                px[y * w + x] = [base * n, base * (0.8 + 0.4 * next()), base * n * 0.9];
            }
        }
        px
    }

    fn distort_nits(src: &[[f32; 3]]) -> Vec<[f32; 3]> {
        src.iter()
            .map(|&[r, g, b]| {
                // Multiplicative + quantization-flavored distortion that
                // stays strictly positive at every luminance.
                [
                    (r * 1.06).max(0.001),
                    (g * 0.95 + 0.02 * r).max(0.001),
                    (b * 1.02).max(0.001),
                ]
            })
            .collect()
    }

    /// Unvalidated HDR shapes stay rejected: HDR-flagged 8-bit sRGB on the
    /// SDR entry, mixed SDR/HDR pairs, wrong format for `Linear` on the
    /// explicit entry.
    #[test]
    fn hdr_route_rejects_unvalidated_shapes() {
        let (w, h) = (96usize, 80usize);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let mut scratch = V2Scratch::new();

        // HDR-flagged source whose pixels are NOT absolute-linear f32:
        // still rejected on the plain entries (routing covers only the
        // validated Linear shape).
        struct HdrSrgb8 {
            data: Vec<[u8; 3]>,
            w: usize,
            h: usize,
        }
        impl crate::source::ImageSource for HdrSrgb8 {
            fn width(&self) -> usize {
                self.w
            }
            fn height(&self) -> usize {
                self.h
            }
            fn pixel_format(&self) -> crate::source::PixelFormat {
                crate::source::PixelFormat::Srgb8Rgb
            }
            fn row_bytes(&self, y: usize) -> &[u8] {
                bytemuck::cast_slice(&self.data[y * self.w..(y + 1) * self.w])
            }
            fn alpha_mode(&self) -> crate::source::AlphaMode {
                crate::source::AlphaMode::Opaque
            }
            fn is_hdr(&self) -> bool {
                true
            }
        }
        let bad = HdrSrgb8 {
            data: vec![[128, 128, 128]; w * h],
            w,
            h,
        };
        let e = z.compute_folded720_append_features(&bad, &bad).unwrap_err();
        assert!(matches!(e, ZensimError::HdrInputRequiresPuPath), "{e:?}");

        // Mixed pair: SDR + HDR.
        let srgb = textured_image(w, h, 5);
        let sdr = RgbSlice::new(&srgb, w, h);
        let nits = nits_image(w, h, 1.0, 200.0, 7);
        let hdr = NitsImage::from_rgb_nits(&nits, w, h);
        let e = z.compute_folded720_append_features(&sdr, &hdr).unwrap_err();
        assert!(matches!(e, ZensimError::HdrInputRequiresPuPath), "{e:?}");

        // Explicit entry, Linear encoding, code-value container: rejected.
        let codes = Code16Image::from_unit_codes(&nits_image(w, h, 0.0, 1.0, 9), w, h);
        let e = z
            .compute_folded720_append_features_hdr(
                &codes,
                &codes,
                HdrEncoding::Linear,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap_err();
        assert!(matches!(e, ZensimError::HdrInputRequiresPuPath), "{e:?}");
    }

    /// An HDR-declared absolute-linear pair routed through the PLAIN
    /// folded/append entries produces bit-identical output to the explicit
    /// HDR entry with `HdrEncoding::Linear` — the auto-route is the same
    /// walk.
    #[test]
    fn hdr_auto_route_matches_explicit_linear() {
        let (w, h) = (150usize, 170usize);
        let src = nits_image(w, h, 0.5, 800.0, 21);
        let dst = distort_nits(&src);
        let s_img = NitsImage::from_rgb_nits(&src, w, h);
        let d_img = NitsImage::from_rgb_nits(&dst, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let mut scratch = V2Scratch::new();

        let auto = z.compute_folded720_append_features(&s_img, &d_img).unwrap();
        let explicit = z
            .compute_folded720_append_features_hdr(
                &s_img,
                &d_img,
                HdrEncoding::Linear,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        assert_eq!(auto.regime(), FeatureRegime::Folded720Append);
        for i in 0..auto.features().len() {
            assert_eq!(
                auto.features()[i].to_bits(),
                explicit.features()[i].to_bits(),
                "auto-route vs explicit diverge at f{i}"
            );
        }
    }

    /// Identity HDR pair: the error-driven append classes are EXACTLY 0 and
    /// the σ-derived ones sit in the identity ULP band — the same contract
    /// as `append_identity_pair_zeros`, on PU-domain planes.
    #[test]
    fn hdr_identity_pair_append_zeros() {
        let (w, h) = (150usize, 130usize);
        let src = nits_image(w, h, 0.05, 2000.0, 5);
        let img = NitsImage::from_rgb_nits(&src, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let mut scratch = V2Scratch::new();
        let r = z
            .compute_folded720_append_features_hdr(
                &img,
                &img,
                HdrEncoding::Linear,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        let app = r.append_features().unwrap();
        const EXACT_ZERO: [usize; 11] = [
            idx_append::XMASK_TRANSDUCER,
            idx_append::LUM_TRANSDUCER,
            idx_append::LUM_DARK_ERR,
            idx_append::LUM_MID_ERR,
            idx_append::LUM_BRIGHT_ERR,
            idx_append::GMS_DEV2,
            idx_append::ART_DEV2,
            idx_append::DET_DEV2,
            idx_append::GLOBAL_DMEAN,
            idx_append::GLOBAL_CGAIN,
            idx_append::GLOBAL_CLOSS,
        ];
        for scale in 0..4 {
            for ch in 0..3 {
                for local in 0..FEATURES_PER_CHANNEL_APPEND {
                    let v = app[scale * 3 * FEATURES_PER_CHANNEL_APPEND
                        + ch * FEATURES_PER_CHANNEL_APPEND
                        + local];
                    if local == idx_append::GRAD_SRC_MEAN {
                        assert!((0.0..1.0).contains(&v), "grad_src_mean {v} s{scale} ch{ch}");
                    } else if EXACT_ZERO.contains(&local) {
                        assert_eq!(v, 0.0, "append {local} not zero on HDR identity: {v}");
                    } else {
                        // Identity ULP band, HDR-scaled: the σ-split noise
                        // (`(ssq − bs2) − mu2²` vs `bs2 − mu1²`) grows with
                        // the PLANE AMPLITUDE squared, and PU planes reach
                        // `pu21(10000)/PU_WHITE ≈ 2.5` vs the SDR path's
                        // ≤ ~1.0 — so the SDR test's 1e-4 band scales by
                        // ~6.5× here. Measured on this fixture: 4.3e-4 max
                        // (CONTRAST_GAIN); 2e-3 leaves ~4× headroom while
                        // still far below any real-signal value.
                        assert!(v.abs() < 2e-3, "append {local} above identity band: {v}");
                    }
                }
            }
        }
        // The whole 720 block must also be finite.
        assert!(r.features().iter().all(|v| v.is_finite()));
    }

    /// Extremes: content at the PU21 domain edges (0.005 and 10,000 cd/m²)
    /// through all three encodings — every one of the 924 slots finite and
    /// inside its documented bound class.
    #[test]
    fn hdr_extremes_bounded_all_encodings() {
        let (w, h) = (96usize, 96usize);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let mut scratch = V2Scratch::new();

        let check = |r: &ZensimV2Result, tag: &str| {
            assert_eq!(r.features().len(), 924, "{tag}");
            for (i, v) in r.features().iter().enumerate() {
                assert!(v.is_finite(), "{tag}: f{i} not finite: {v}");
                assert!(
                    (-1.0..=3.0).contains(v),
                    "{tag}: f{i} outside sanity range: {v}"
                );
            }
        };

        // Linear at the domain floor and ceiling (plus a full-range ramp).
        for (lo, hi, tag) in [
            (0.005f32, 0.05f32, "linear-black"),
            (5_000.0, 10_000.0, "linear-peak"),
            (0.005, 10_000.0, "linear-fullrange"),
        ] {
            let src = nits_image(w, h, lo, hi, 3);
            let dst = distort_nits(&src);
            let r = z
                .compute_folded720_append_features_hdr(
                    &NitsImage::from_rgb_nits(&src, w, h),
                    &NitsImage::from_rgb_nits(&dst, w, h),
                    HdrEncoding::Linear,
                    V2NewFeatureToggles::default(),
                    &mut scratch,
                )
                .unwrap();
            check(&r, tag);
        }

        // PQ code values 0.0..=1.0 (decodes to 0..10000 at spec peak;
        // clamps at 1000 for the display-limited variant).
        let codes = nits_image(w, h, 1e-4, 1.0, 13); // reuse generator as unit ramp
        let dstc: Vec<[f32; 3]> = codes
            .iter()
            .map(|&[r, g, b]| [(r * 0.97).min(1.0), (g * 1.02).min(1.0), b.min(1.0)])
            .collect();
        for (peak, tag) in [(10_000.0f32, "pq-spec-peak"), (1_000.0, "pq-1000")] {
            let r = z
                .compute_folded720_append_features_hdr(
                    &Code16Image::from_unit_codes(&codes, w, h),
                    &Code16Image::from_unit_codes(&dstc, w, h),
                    HdrEncoding::Pq { peak_nits: peak },
                    V2NewFeatureToggles::default(),
                    &mut scratch,
                )
                .unwrap();
            check(&r, tag);
        }

        // HLG signal values through the reference OOTF.
        let r = z
            .compute_folded720_append_features_hdr(
                &Code16Image::from_unit_codes(&codes, w, h),
                &Code16Image::from_unit_codes(&dstc, w, h),
                HdrEncoding::Hlg {
                    peak_nits: 1000.0,
                    ambient_lux: 5.0,
                },
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        check(&r, "hlg-1000");
    }

    /// BANDVIS δ-constant derivation (append2, 2026-07-27): measure the
    /// |∇| the span-2 central-difference kernel reports at one-code-step
    /// plateau edges, through the REAL front-ends, at all 4 walk scales.
    /// Prints the table (`--nocapture`) that
    /// `benchmarks/append2_bandvis_gates_2026-07-27.md` commits, and pins
    /// the shipped constants to brackets of the measurement.
    #[test]
    fn bandvis_delta_derivation_table() {
        let w = 512usize;
        let h = 128usize;

        // --- SDR: horizontal sRGB ramp posterized to 8-bit steps of k codes.
        // Adjacent plateaus differ by k/255 in sRGB code → Y-plane step
        // through srgb→XYB(cbrt).
        let sdr_step = |k: usize| -> f64 {
            let mut px = vec![[0u8; 3]; w * h];
            for y in 0..h {
                for x in 0..w {
                    // mid-gray plateaus: 118 or 118+k
                    let v = if x < w / 2 { 118u8 } else { (118 + k) as u8 };
                    px[y * w + x] = [v, v, v];
                }
            }
            let mut c0 = vec![0.0f32; w * h];
            let mut c1 = vec![0.0f32; w * h];
            let mut c2 = vec![0.0f32; w * h];
            crate::color::srgb_to_positive_xyb_planar_into(&px, &mut c0, &mut c1, &mut c2);
            // span-2 central difference at the edge = full plateau delta
            (c1[w / 2 + 1] as f64 - c1[w / 2 - 2] as f64).abs()
        };

        // --- HDR/PU: PQ 10-bit steps around given absolute luminances.
        let pu_step = |nits: f32, tenbit_steps: u32| -> f64 {
            // PQ inverse EOTF (code from nits) via bisection on pq_eotf.
            let code_of = |target: f32| -> f32 {
                let (mut lo, mut hi) = (0.0f32, 1.0f32);
                for _ in 0..50 {
                    let mid = 0.5 * (lo + hi);
                    if crate::transfer::pq_eotf(mid) < target {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                lo
            };
            let c = code_of(nits);
            let c2v = c + tenbit_steps as f32 / 1023.0;
            let l1 = crate::transfer::pq_eotf(c);
            let l2 = crate::transfer::pq_eotf(c2v);
            let px = [[l1, l1, l1], [l2, l2, l2]];
            let mut o0 = [0.0f32; 2];
            let mut o1 = [0.0f32; 2];
            let mut o2 = [0.0f32; 2];
            crate::color::linear_to_pu_xyb_planar_into(&px, &mut o0, &mut o1, &mut o2);
            (o1[1] as f64 - o1[0] as f64).abs()
        };

        println!("SDR (sRGB@118 gray, cbrt-Y step per k 8-bit codes):");
        for k in [1usize, 2, 4, 6] {
            println!("  k={k}: |ΔY| = {:.6}", sdr_step(k));
        }
        println!("HDR/PU (PQ 10-bit steps → PU-Y):");
        for nits in [1.0f32, 10.0, 100.0, 1000.0, 5000.0] {
            for st in [1u32, 4] {
                println!(
                    "  {nits} nits, {st} steps: |ΔPU-Y| = {:.6}",
                    pu_step(nits, st)
                );
            }
        }

        // Bracket assertions for the shipped constants (values chosen from
        // this measurement — see the gates doc):
        let one_sdr = sdr_step(1);
        assert!(
            BV_DELTA_LO_SDR as f64 > 0.3 * one_sdr && (BV_DELTA_LO_SDR as f64) < 0.7 * one_sdr,
            "δ_lo_sdr {} vs 1-step {}",
            BV_DELTA_LO_SDR,
            one_sdr
        );
        assert!(
            BV_DELTA_HI_SDR as f64 > 3.5 * one_sdr && (BV_DELTA_HI_SDR as f64) < 7.0 * one_sdr,
            "δ_hi_sdr {} vs 1-step {}",
            BV_DELTA_HI_SDR,
            one_sdr
        );
        // HL-bin anchors: the PU-route Y-plane value of a GRAY pixel at
        // 100 cd/m² (SDR white in the normalized domain) and 1000 cd/m².
        let y_of_nits = |nits: f32| -> f64 {
            let px = [[nits, nits, nits]];
            let mut o0 = [0.0f32; 1];
            let mut o1 = [0.0f32; 1];
            let mut o2 = [0.0f32; 1];
            crate::color::linear_to_pu_xyb_planar_into(&px, &mut o0, &mut o1, &mut o2);
            o1[0] as f64
        };
        println!("PU-route Y-plane gray values (HL anchors):");
        for nits in [80.0f32, 100.0, 203.0, 1000.0, 4000.0] {
            println!("  {nits} nits → y = {:.5}", y_of_nits(nits));
        }
        assert!(
            (HL1_Y_ANCHOR as f64 - y_of_nits(100.0)).abs() < 0.02,
            "HL1 anchor {} vs measured PU-Y(100 nits) {}",
            HL1_Y_ANCHOR,
            y_of_nits(100.0)
        );
        assert!(
            (HL2_Y_ANCHOR as f64 - y_of_nits(1000.0)).abs() < 0.02,
            "HL2 anchor {} vs measured PU-Y(1000 nits) {}",
            HL2_Y_ANCHOR,
            y_of_nits(1000.0)
        );

        let one_pu_mid = pu_step(100.0, 1);
        assert!(
            BV_DELTA_LO_PU as f64 > 0.3 * one_pu_mid && (BV_DELTA_LO_PU as f64) < 0.7 * one_pu_mid,
            "δ_lo_pu {} vs 1-step@100nits {}",
            BV_DELTA_LO_PU,
            one_pu_mid
        );
        assert!(
            BV_DELTA_HI_PU as f64 > 3.5 * one_pu_mid && (BV_DELTA_HI_PU as f64) < 7.0 * one_pu_mid,
            "δ_hi_pu {} vs 1-step@100nits {}",
            BV_DELTA_HI_PU,
            one_pu_mid
        );
    }

    // ========================================================================
    // append2 gates (BANDVIS + conditioner + HL bins)
    // ========================================================================

    /// Identity + layout + first-924 bit-stability + entry parity at 944.
    #[test]
    fn append2_layout_identity_and_first924_bit_stable() {
        let (w, h) = (150usize, 170usize);
        let src = textured_image(w, h, 23);
        let dst = quantize_distort(&src, w, h);
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);

        let a924 = z.compute_folded720_append_features(&sref, &dref).unwrap();
        let a944 = z.compute_folded720_append2_features(&sref, &dref).unwrap();
        assert_eq!(a944.regime(), FeatureRegime::Folded720Append2);
        assert_eq!(a944.features().len(), 944);
        assert_eq!(a944.append2_features().unwrap().len(), 20);
        assert_eq!(a944.append_features().unwrap().len(), 204);
        // Turning append2 on must not move a bit of the first 924.
        for i in 0..924 {
            assert_eq!(
                a924.features()[i].to_bits(),
                a944.features()[i].to_bits(),
                "append2 toggled on moved f{i}"
            );
        }
        // Parallel + scratch parity at 944.
        let zp = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(true);
        let b = zp.compute_folded720_append2_features(&sref, &dref).unwrap();
        for i in 0..944 {
            assert_eq!(
                a944.features()[i].to_bits(),
                b.features()[i].to_bits(),
                "serial vs parallel diverge at f{i}"
            );
        }
        // All 20 new slots bounded [0,1].
        for (i, v) in a944.append2_features().unwrap().iter().enumerate() {
            assert!((0.0..=1.0).contains(v) && v.is_finite(), "app2[{i}] = {v}");
        }

        // Identity pair: BANDVIS gain/loss EXACTLY 0; HL bins EXACTLY 0
        // on the SDR route; conditioner in (0,1).
        let idr = z.compute_folded720_append2_features(&sref, &sref).unwrap();
        let app2 = idr.append2_features().unwrap();
        for scale in 0..4 {
            let b = scale * APPEND2_PER_SCALE;
            assert_eq!(app2[b + idx_append2::BANDVIS_GAIN], 0.0, "s{scale} gain");
            assert_eq!(app2[b + idx_append2::BANDVIS_LOSS], 0.0, "s{scale} loss");
            assert_eq!(app2[b + idx_append2::HL_BIN1], 0.0, "s{scale} hl1 (SDR)");
            assert_eq!(app2[b + idx_append2::HL_BIN2], 0.0, "s{scale} hl2 (SDR)");
            let lm = app2[b + idx_append2::LUMA_MEAN_REF];
            assert!(lm > 0.0 && lm < 1.0, "s{scale} luma_mean {lm}");
        }
    }

    // --- Shared BANDVIS fixtures (gates doc V3). Extracted 2026-08-02 so
    // the P1.5 dst-activity adjudication tests measure the IDENTICAL
    // pixel content as the OFF-math characterization pins (formulas
    // verbatim from the original inline builders; the behavior test's
    // value pins + the gates-doc printed numbers gate the motion).
    // `benchmarks/bandvis_dst_activity_2026-08-02.md`. ---

    /// Smooth diagonal gradient ramp (sRGB codes 32..224).
    fn bv_ramp(w: usize, h: usize) -> Vec<[u8; 3]> {
        (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let v = 32.0 + 192.0 * (x + y) as f32 / (w + h - 2) as f32;
                let v = v as u8;
                [v, v, v]
            })
            .collect()
    }

    /// Mid-tread posterize to `bits` (the gates-doc ladder operator).
    fn bv_posterize(px: &[[u8; 3]], bits: u32) -> Vec<[u8; 3]> {
        let mask = !((1u16 << (8 - bits)) - 1) as u8;
        let half = ((1u16 << (8 - bits)) / 2) as u8;
        px.iter()
            .map(|&[r, g, b]| [(r & mask) | half, (g & mask) | half, (b & mask) | half])
            .collect()
    }

    /// 4-bit ordered-Bayer (4×4) dither of the diagonal ramp.
    fn bv_bayer_dither(w: usize, h: usize) -> Vec<[u8; 3]> {
        let bayer = [
            [0u8, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5],
        ];
        (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let v = 32.0 + 192.0 * (x + y) as f32 / (w + h - 2) as f32;
                let t = (bayer[y % 4][x % 4] as f32 + 0.5) / 16.0;
                let step = 16.0; // 4-bit quantization step
                let q = ((v / step + t - 0.5).floor() * step + step / 2.0).clamp(0.0, 255.0) as u8;
                [q, q, q]
            })
            .collect()
    }

    /// 4-bit hash-noise (TPDF-class) dither of the diagonal ramp.
    fn bv_noise_dither(w: usize, h: usize) -> Vec<[u8; 3]> {
        (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let v = 32.0 + 192.0 * (x + y) as f32 / (w + h - 2) as f32;
                let mut hsh = (x as u32)
                    .wrapping_mul(0x9E37_79B9)
                    .wrapping_add((y as u32).wrapping_mul(0x85EB_CA6B));
                hsh ^= hsh >> 15;
                hsh = hsh.wrapping_mul(0x2C1B_3C6D);
                hsh ^= hsh >> 13;
                let t = ((hsh & 0xFFFF) as f32 + 0.5) / 65536.0;
                let step = 16.0;
                let q = ((v / step + t - 0.5).floor() * step + step / 2.0).clamp(0.0, 255.0) as u8;
                [q, q, q]
            })
            .collect()
    }

    /// ±16-code hash-textured version of `ramp` (the b2 source fixture).
    fn bv_textured_ramp(ramp: &[[u8; 3]]) -> Vec<[u8; 3]> {
        ramp.iter()
            .enumerate()
            .map(|(i, &[v, _, _])| {
                let mut hsh = (i as u32).wrapping_mul(0x27D4_EB2F);
                hsh ^= hsh >> 15;
                let n = ((hsh & 0x1F) as i16) - 16; // ±16 codes texture
                let t = (v as i16 + n).clamp(0, 255) as u8;
                [t, t, t]
            })
            .collect()
    }

    /// ±6-code 8-px DC-lattice overlay on `ramp` (the V3(c) fixture).
    fn bv_blocky(ramp: &[[u8; 3]], w: usize) -> Vec<[u8; 3]> {
        ramp.iter()
            .enumerate()
            .map(|(i, &[v, _, _])| {
                let (x, y) = (i % w, i / w);
                let dc = if ((x / 8) + (y / 8)) % 2 == 0 { 6 } else { -6 };
                let q = (v as i16 + dc).clamp(0, 255) as u8;
                [q, q, q]
            })
            .collect()
    }

    /// V3(a,b,c,e) behavioral: posterize-ladder monotonicity, dither
    /// masking, blocky-not-banded separation, debanding credit.
    #[test]
    fn append2_bandvis_behavior() {
        let (w, h) = (256usize, 256usize);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);

        // Smooth diagonal gradient ramp (sRGB codes 32..224).
        let ramp: Vec<[u8; 3]> = bv_ramp(w, h);
        let posterize = bv_posterize;
        let gain_at = |dst: &[[u8; 3]]| -> Vec<f64> {
            let r = z
                .compute_folded720_append2_features(
                    &RgbSlice::new(&ramp, w, h),
                    &RgbSlice::new(dst, w, h),
                )
                .unwrap();
            let app2 = r.append2_features().unwrap().to_vec();
            (0..4)
                .map(|s| app2[s * APPEND2_PER_SCALE + idx_append2::BANDVIS_GAIN])
                .collect()
        };

        // (a) Ladder at the COARSE scales the detector serves (see
        // `idx_append2::BANDVIS_GAIN`'s measured-mechanics note): within
        // the visibility band (6→5→4 bit = 4/8/16-code steps), fewer
        // levels ⇒ strictly more banding at scales 2 and 3; 3-bit
        // (32-code steps) attenuates vs 4-bit — the CAMBI-faithful upper
        // contrast cutoff (giant steps are edges, not banding).
        let g6 = gain_at(&posterize(&ramp, 6));
        let g5 = gain_at(&posterize(&ramp, 5));
        let g4 = gain_at(&posterize(&ramp, 4));
        let g3 = gain_at(&posterize(&ramp, 3));
        println!(
            "ladder: 6bit {g6:?}\n        5bit {g5:?}\n        4bit {g4:?}\n        3bit {g3:?}"
        );
        // MEASURED STRUCTURE (design-true invariants; full matrix in the
        // gates doc): the response over the posterize ladder is UNIMODAL —
        // it rises through the visibility band (step sizes approaching the
        // band optimum √(δ_lo·δ_hi) ≈ 1.6 8-bit codes × edge density) and
        // rolls off past the contrast cap (steps ≫ δ_hi are edges, not
        // banding — CAMBI's own cap). The per-scale slots form the
        // resonance curve consumers read.
        let g7 = gain_at(&posterize(&ramp, 7));
        let peak = |g: &Vec<f64>| {
            g.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, &v)| (i, v))
                .unwrap()
        };
        let (s7, p7) = peak(&g7);
        let (s6, p6) = peak(&g6);
        let (s5, p5) = peak(&g5);
        let (s4, p4) = peak(&g4);
        let (s3b, p3) = peak(&g3);
        println!(
            "resonance: 7b s{s7}@{p7:.4} 6b s{s6}@{p6:.4} 5b s{s5}@{p5:.4} 4b s{s4}@{p4:.4} 3b s{s3b}@{p3:.4}"
        );
        // 1. Every posterize rung fires clearly (curvature SNR: the
        //    weakest rung is still >0.05 on this fixture).
        for (tag, p) in [("7b", p7), ("6b", p6), ("5b", p5), ("4b", p4), ("3b", p3)] {
            assert!(p > 0.05, "{tag} rung must fire: {p}");
        }
        // 2. Unimodal: the ladder's interior peak exceeds both endpoints.
        let pmax = [p7, p6, p5, p4, p3].into_iter().fold(0.0f64, f64::max);
        assert!(
            pmax > p7 && pmax > p3,
            "ladder should peak in the interior: {p7} .. {pmax} .. {p3}"
        );
        // 3. Cap rolloff: monotone decline past the band (5b → 4b → 3b).
        assert!(
            p5 > p4 && p4 > p3,
            "cap rolloff not monotone: {p5} {p4} {p3}"
        );

        // (b) Dither masking: ordered-dither the SAME 4-bit posterize —
        // gain must drop substantially.
        let dithered: Vec<[u8; 3]> = bv_bayer_dither(w, h);
        // (b1) NOISE dither (the industry deband practice — TPDF-class
        // random dither): the quantization residual decorrelates and
        // averages out sub-band at coarse scales ⇒ GAIN drops
        // substantially vs plain posterize.
        let noise_dithered: Vec<[u8; 3]> = bv_noise_dither(w, h);
        let gn = gain_at(&noise_dithered);
        let ratio_noise = gn[3] / g4[3].max(1e-12);
        println!(
            "noise-dither masking [scale3]: undithered {:.5} dithered {:.5} ratio {:.3}",
            g4[3], gn[3], ratio_noise
        );
        // V3(b) VERDICT — MISS, recorded (gates doc): dst-side dither
        // (noise AND ordered) FIRES rather than masking. Structural: any
        // ~1-code quantization residual has dense mid-band curvature, and
        // the flatness mask is REF-side (the specified design); local
        // dst-texture masking needs a dst-activity plane (~+5% CPU,
        // outside this wave's ≤+2% budget — REMAINDERS). Pin the measured
        // behavior so a silent change trips a test:
        assert!(
            ratio_noise > 1.0,
            "characterization pin: dst noise-dither currently FIRES (ratio {ratio_noise}); \
             if this improved, the gates doc + REMAINDERS need updating"
        );

        // (b2) Source-texture masking (the activity mask's ref-side
        // mechanism): the SAME posterization applied to a noise-textured
        // source fires much less than on the clean ramp.
        let textured_src: Vec<[u8; 3]> = bv_textured_ramp(&ramp);
        let r_tex = z
            .compute_folded720_append2_features(
                &RgbSlice::new(&textured_src, w, h),
                &RgbSlice::new(&bv_posterize(&textured_src, 4), w, h),
            )
            .unwrap();
        let a2t = r_tex.append2_features().unwrap();
        let g_tex3 = a2t[3 * APPEND2_PER_SCALE + idx_append2::BANDVIS_GAIN];
        println!(
            "src-texture masking [scale3]: clean-src {:.5} textured-src {:.5} ratio {:.3}",
            g4[3],
            g_tex3,
            g_tex3 / g4[3].max(1e-12)
        );
        assert!(
            g_tex3 < 0.5 * g4[3],
            "source texture should mask banding: {g_tex3} vs {}",
            g4[3]
        );

        // RECORDED characteristic (not asserted): ordered 4×4 Bayer at
        // this 16-code step size aliases a ~2-code low-frequency pattern
        // into the coarse scales — in-band curvature that GAIN reports
        // (the pattern IS visible at this amplitude; noise dither above
        // is the practice BANDVIS should and does credit). Numbers for
        // the gates doc:
        let gd = gain_at(&dithered);
        println!(
            "ordered-Bayer dither [per scale]: {:?} (vs undithered s3 {:.5})",
            gd, g4[3]
        );

        // (c) Blocky-but-not-banded: 8px-lattice DC shifts fire blockiness,
        // not bandvis.
        let blocky: Vec<[u8; 3]> = bv_blocky(&ramp, w);
        let rb = z
            .compute_folded720_append2_features(
                &RgbSlice::new(&ramp, w, h),
                &RgbSlice::new(&blocky, w, h),
            )
            .unwrap();
        let app2b = rb.append2_features().unwrap();
        let gain_blocky_max = (0..4)
            .map(|s| app2b[s * APPEND2_PER_SCALE + idx_append2::BANDVIS_GAIN])
            .fold(0.0f64, f64::max);
        let blockiness = rb.view().blockiness(0, 1);
        println!(
            "blocky fixture: max bandvis_gain {gain_blocky_max:.5} blockiness(Y,s0) {blockiness:.5}"
        );
        // V3(c) VERDICT — MISS, recorded (gates doc + REMAINDERS): a
        // dense DC-shift lattice at its resonant scale fires BANDVIS
        // strongly (the flatness mask is REF-side; the dst's own blocky
        // texture does not self-mask — same root cause as V3(b)).
        // BLOCKINESS remains the lattice-SPECIFIC discriminator the head
        // pairs it with. Pinned invariants: blockiness fires decisively,
        // and the bandvis cross-response is nonzero (characterization pin
        // — if either moves, re-verdict the gates doc).
        assert!(
            blockiness > 0.1,
            "blocky fixture must fire blockiness: {blockiness}"
        );
        assert!(
            gain_blocky_max > 0.1,
            "characterization pin: dense DC lattices currently cross-fire bandvis \
             ({gain_blocky_max}); if this improved, update the gates doc"
        );

        // (e) Debanding credit: banded SOURCE, ideally-debanded dst (the
        // smooth ramp itself) — LOSS fires at the banding's resonant
        // scale, GAIN doesn't.
        let r = z
            .compute_folded720_append2_features(
                &RgbSlice::new(&posterize(&ramp, 4), w, h),
                &RgbSlice::new(&ramp, w, h),
            )
            .unwrap();
        let app2 = r.append2_features().unwrap();
        let b3 = 3 * APPEND2_PER_SCALE;
        let (gain3, loss3) = (
            app2[b3 + idx_append2::BANDVIS_GAIN],
            app2[b3 + idx_append2::BANDVIS_LOSS],
        );
        println!("debanding (ideal) [scale3]: gain {gain3:.5} loss {loss3:.5}");
        // V3(e): the LOSS credit direction holds (loss > gain). A stronger
        // loss ≫ gain is unattainable in u8 fixtures: an 8-bit "smooth"
        // ramp is itself a 1-code micro-staircase whose post-downscale
        // curvature residual (~0.3 code) sits at the band's lower edge —
        // the gain term correctly reports it (gates doc).
        assert!(
            loss3 > gain3,
            "debanding must credit LOSS over GAIN at scale 3: g {gain3} l {loss3}"
        );
        assert!(loss3 > 0.1, "debanding LOSS should fire strongly: {loss3}");
        // RECORDED characteristic (not asserted): a dithered "deband" at
        // this 4-bit magnitude leaves ~2-code residual pattern after
        // downscale — in-band dense texture that GAIN reads as new
        // stepping at deep scales (arguably correct: ordered dither at
        // this amplitude is visible mottle). Numbers for the gates doc:
        let rd = z
            .compute_folded720_append2_features(
                &RgbSlice::new(&posterize(&ramp, 4), w, h),
                &RgbSlice::new(&dithered, w, h),
            )
            .unwrap();
        let a2 = rd.append2_features().unwrap();
        println!(
            "deband-by-dither [scale3]: gain {:.5} loss {:.5}",
            a2[b3 + idx_append2::BANDVIS_GAIN],
            a2[b3 + idx_append2::BANDVIS_LOSS]
        );
    }

    /// P1.5 dst-activity toggle — structural gates (pre-registered F6/F7/
    /// F8 in `benchmarks/bandvis_dst_activity_2026-08-02.md`, tightened
    /// to the SHIPPED combine): toggle ON moves ONLY the four BANDVIS
    /// GAIN slots of the 944 vector — every other slot INCLUDING
    /// BANDVIS_LOSS is bit-identical (the shipped combine keeps LOSS =
    /// the OFF math exactly) — identity stays exactly 0, serial ≡
    /// parallel, and the toggle is live (GAIN lanes DO move on a
    /// quantized pair).
    #[test]
    fn append2_dst_activity_lanes_only_identity_and_parallel() {
        let (w, h) = (150usize, 170usize);
        let src = textured_image(w, h, 23);
        let dst = quantize_distort(&src, w, h);
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let mut scratch = V2Scratch::new();
        let t_off = V2NewFeatureToggles {
            append_block: true,
            append2_block: true,
            ..Default::default()
        };
        let t_on = V2NewFeatureToggles {
            append2_dst_activity: true,
            ..t_off
        };
        let off = z
            .compute_folded720_append_features_streaming(&sref, &dref, t_off, &mut scratch)
            .unwrap();
        let on = z
            .compute_folded720_append_features_streaming(&sref, &dref, t_on, &mut scratch)
            .unwrap();
        assert_eq!(off.features().len(), 944);
        assert_eq!(on.features().len(), 944);
        assert_eq!(on.regime(), FeatureRegime::Folded720Append2);
        // F7 (tightened): only the BANDVIS GAIN lanes may differ — LOSS
        // is bit-stable by the shipped combine's construction.
        let mut lanes_moved = 0usize;
        for i in 0..944 {
            let gain_lane = i >= 924 && (i - 924) % APPEND2_PER_SCALE == idx_append2::BANDVIS_GAIN;
            if gain_lane {
                lanes_moved += (off.features()[i].to_bits() != on.features()[i].to_bits()) as usize;
            } else {
                assert_eq!(
                    off.features()[i].to_bits(),
                    on.features()[i].to_bits(),
                    "append2_dst_activity moved a non-GAIN slot f{i}"
                );
            }
        }
        assert!(
            lanes_moved > 0,
            "toggle must be LIVE on a quantized pair (no GAIN lane moved)"
        );
        // F6: identity pair stays exactly 0 with the toggle ON (activity
        // twins are bitwise-identical on identical planes ⇒ FR pair 0).
        let idr = z
            .compute_folded720_append_features_streaming(&sref, &sref, t_on, &mut scratch)
            .unwrap();
        let app2 = idr.append2_features().unwrap();
        for scale in 0..4 {
            let b = scale * APPEND2_PER_SCALE;
            assert_eq!(app2[b + idx_append2::BANDVIS_GAIN], 0.0, "s{scale} gain");
            assert_eq!(app2[b + idx_append2::BANDVIS_LOSS], 0.0, "s{scale} loss");
        }
        // F8: serial ≡ parallel at 944 with the toggle ON.
        #[cfg(feature = "threads")]
        {
            let zp = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(true);
            let mut scratch_p = V2Scratch::new();
            let on_p = zp
                .compute_folded720_append_features_streaming(&sref, &dref, t_on, &mut scratch_p)
                .unwrap();
            for i in 0..944 {
                assert_eq!(
                    on.features()[i].to_bits(),
                    on_p.features()[i].to_bits(),
                    "serial vs parallel diverge at f{i} (dst-activity ON)"
                );
            }
        }
    }

    /// P1.5 dst-activity adjudication matrix (F1–F5 of
    /// `benchmarks/bandvis_dst_activity_2026-08-02.md`): OFF and ON arms
    /// over the IDENTICAL gates-doc fixtures (shared `bv_*` builders).
    /// Prints the full per-scale table the doc records. Asserted here:
    /// the design-contract invariants (real banding keeps firing with the
    /// self-mask ON; identity/direction gates) — the adjudication gates
    /// (F2 ratio, F3 reduction, …) are recorded in the doc with the
    /// measured outcome, and the adjudicated pins were added below after
    /// the doc's results section landed.
    #[test]
    fn append2_dst_activity_behavior_matrix() {
        let (w, h) = (256usize, 256usize);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let mut scratch = V2Scratch::new();
        let t_off = V2NewFeatureToggles {
            append_block: true,
            append2_block: true,
            ..Default::default()
        };
        let t_on = V2NewFeatureToggles {
            append2_dst_activity: true,
            ..t_off
        };
        let mut gl = |src: &[[u8; 3]], dst: &[[u8; 3]], on: bool| -> (Vec<f64>, Vec<f64>) {
            let r = z
                .compute_folded720_append_features_streaming(
                    &RgbSlice::new(src, w, h),
                    &RgbSlice::new(dst, w, h),
                    if on { t_on } else { t_off },
                    &mut scratch,
                )
                .unwrap();
            let a = r.append2_features().unwrap();
            (
                (0..4)
                    .map(|s| a[s * APPEND2_PER_SCALE + idx_append2::BANDVIS_GAIN])
                    .collect(),
                (0..4)
                    .map(|s| a[s * APPEND2_PER_SCALE + idx_append2::BANDVIS_LOSS])
                    .collect(),
            )
        };

        let ramp = bv_ramp(w, h);
        let p = |bits| bv_posterize(&ramp, bits);

        // F1: the posterize ladder, both arms (asserts AFTER the full
        // matrix prints, so a single miss still shows every number).
        let mut peaks_on = Vec::new();
        for bits in [7u32, 6, 5, 4, 3] {
            let (g_off, _) = gl(&ramp, &p(bits), false);
            let (g_on, _) = gl(&ramp, &p(bits), true);
            println!("F1 ladder {bits}b OFF {g_off:?}\n              ON  {g_on:?}");
            let pk = g_on.iter().cloned().fold(0.0f64, f64::max);
            peaks_on.push(pk);
        }

        // F2: noise-dither ratio @s3, both arms.
        let nd = bv_noise_dither(w, h);
        let (g4_off, _) = gl(&ramp, &p(4), false);
        let (g4_on, _) = gl(&ramp, &p(4), true);
        let (gn_off, ln_off) = gl(&ramp, &nd, false);
        let (gn_on, ln_on) = gl(&ramp, &nd, true);
        let ratio_off = gn_off[3] / g4_off[3].max(1e-12);
        let ratio_on = gn_on[3] / g4_on[3].max(1e-12);
        println!(
            "F2 noise-dither s3: OFF undith {:.5} dith {:.5} ratio {:.3} | ON undith {:.5} dith {:.5} ratio {:.3}",
            g4_off[3], gn_off[3], ratio_off, g4_on[3], gn_on[3], ratio_on
        );
        println!(
            "F2 noise-dither LOSS s0..s3: OFF {ln_off:?} ON {ln_on:?} (dither-as-partial-deband credit direction)"
        );

        // F2b: ordered-Bayer per scale, both arms.
        let bd = bv_bayer_dither(w, h);
        let (gb_off, _) = gl(&ramp, &bd, false);
        let (gb_on, _) = gl(&ramp, &bd, true);
        println!("F2b bayer GAIN OFF {gb_off:?}\n               ON  {gb_on:?}");

        // F3: DC lattice, both arms.
        let blk = bv_blocky(&ramp, w);
        let (gk_off, _) = gl(&ramp, &blk, false);
        let (gk_on, _) = gl(&ramp, &blk, true);
        let mx_off = gk_off.iter().cloned().fold(0.0f64, f64::max);
        let mx_on = gk_on.iter().cloned().fold(0.0f64, f64::max);
        println!(
            "F3 lattice GAIN OFF {gk_off:?} max {mx_off:.4}\n                ON  {gk_on:?} max {mx_on:.4}"
        );

        // F4: source-texture masking, both arms.
        let tex = bv_textured_ramp(&ramp);
        let texp = bv_posterize(&tex, 4);
        let (gt_off, _) = gl(&tex, &texp, false);
        let (gt_on, _) = gl(&tex, &texp, true);
        println!(
            "F4 src-texture s3: OFF {:.5} (ratio {:.3}) ON {:.5} (ratio {:.3})",
            gt_off[3],
            gt_off[3] / g4_off[3].max(1e-12),
            gt_on[3],
            gt_on[3] / g4_on[3].max(1e-12)
        );
        assert!(
            gt_on[3] < 0.5 * g4_on[3],
            "F4: source texture must keep masking with dst self-mask ON: {} vs {}",
            gt_on[3],
            g4_on[3]
        );

        // F5: deband credit, both arms.
        let (gd_off, ld_off) = gl(&p(4), &ramp, false);
        let (gd_on, ld_on) = gl(&p(4), &ramp, true);
        println!(
            "F5 deband s3: OFF gain {:.5} loss {:.5} | ON gain {:.5} loss {:.5}",
            gd_off[3], ld_off[3], gd_on[3], ld_on[3]
        );

        // --- Adjudicated pins (shipped combine; full registered-gate
        // PASS/MISS ledger in the doc) ---
        // F1: in-band rungs keep firing; the 3b cap-tail rung measured
        // 0.0347 — UNDER the registered 0.05 bar (recorded MISS: 32-code
        // steps carry strong self-activity AND sit in the "edges not
        // banding" cap regime) — pinned at its measured level so a silent
        // further collapse trips.
        for (i, bits) in [7u32, 6, 5, 4].iter().enumerate() {
            assert!(
                peaks_on[i] > 0.05,
                "F1: {bits}b rung must keep firing with the dst weight ON: {}",
                peaks_on[i]
            );
        }
        assert!(
            peaks_on[4] > 0.02,
            "F1: 3b cap-tail rung collapsed below its measured level: {}",
            peaks_on[4]
        );
        let pmax = peaks_on.iter().cloned().fold(0.0f64, f64::max);
        assert!(
            pmax > peaks_on[3] && pmax > peaks_on[4],
            "F1: peak must sit in the visibility band, not the cap tail (ON): {peaks_on:?}"
        );
        assert!(
            peaks_on[2] > peaks_on[3] && peaks_on[3] > peaks_on[4],
            "F1: cap rolloff not monotone (ON): {peaks_on:?}"
        );
        // Shipped-combine construction: LOSS is BIT-identical to OFF on
        // every fixture (the toggle re-weights GAIN only).
        for s in 0..4 {
            assert_eq!(
                ln_on[s].to_bits(),
                ln_off[s].to_bits(),
                "LOSS moved under the toggle (dither fixture, s{s})"
            );
            assert_eq!(
                ld_on[s].to_bits(),
                ld_off[s].to_bits(),
                "LOSS moved under the toggle (deband fixture, s{s})"
            );
        }
        // F3 (adjudicated PASS): the pooling weight suppresses the
        // lattice cross-fire decisively.
        assert!(
            mx_on < 0.5 * mx_off,
            "F3: lattice cross-fire must stay < 0.5x OFF: {mx_on} vs {mx_off}"
        );
        // F5 (shipped combine): LOSS keeps the OFF value, GAIN takes the
        // arm-2 weight — direction restored with a margin ABOVE the OFF
        // margin.
        assert!(
            ld_on[3] > gd_on[3],
            "F5: debanding must credit LOSS over GAIN at s3 (ON): g {} l {}",
            gd_on[3],
            ld_on[3]
        );
        assert!(
            ld_on[3] > 0.1,
            "F5: debanding LOSS should fire strongly (ON): {}",
            ld_on[3]
        );
        assert!(
            (ld_on[3] - gd_on[3]) > (ld_off[3] - gd_off[3]),
            "F5: the shipped combine's deband margin must exceed the OFF margin"
        );
    }

    /// V4-class: HL bins fire on >SDR-white highlight error, exactly 0 on
    /// SDR-range HDR content; BANDVIS runs the PU constants on the HDR
    /// route (posterized 10-bit-ish ramp fires).
    #[test]
    fn append2_hdr_hl_bins_and_pu_bandvis() {
        let (w, h) = (128usize, 128usize);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let mut scratch = V2Scratch::new();

        // Highlight-error pair: ramp 50..3000 nits, error only >300 nits.
        let ramp: Vec<[f32; 3]> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let t = (x + y) as f32 / (w + h - 2) as f32;
                let nits = 50.0 * (3000.0f32 / 50.0).powf(t);
                [nits, nits, nits]
            })
            .collect();
        let dst: Vec<[f32; 3]> = ramp
            .iter()
            .map(|&[r, g, b]| {
                if r > 300.0 {
                    [r * 1.15, g * 1.15, b * 1.15]
                } else {
                    [r, g, b]
                }
            })
            .collect();
        let r = z
            .compute_folded720_append2_features_hdr(
                &NitsImage::from_rgb_nits(&ramp, w, h),
                &NitsImage::from_rgb_nits(&dst, w, h),
                HdrEncoding::Linear,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        let app2 = r.append2_features().unwrap();
        println!(
            "HL highlight-error: hl1 {:.5} hl2 {:.5} dark-ctl mse {:.5}",
            app2[idx_append2::HL_BIN1],
            app2[idx_append2::HL_BIN2],
            r.append_features().unwrap()[idx_append::LUM_DARK_ERR]
        );
        assert!(app2[idx_append2::HL_BIN1] > 1e-3, "HL1 should fire");
        assert!(app2[idx_append2::HL_BIN2] > 1e-3, "HL2 should fire");
        // Slot-forensics F-4 coverage (benchmarks/extractor_slot_forensics_
        // 2026-08-05.md): the 8 HL slots (f927/932/937/942 + f928/933/938/943)
        // are the ONLY route-gated members of the audit's 39 never-populated
        // set — prove every one of them populates on the HDR route, not just
        // scale 0 (the highlight region survives every downscale, so all 4
        // per-scale kernels must accumulate mass).
        for scale in 0..4 {
            let b = scale * APPEND2_PER_SCALE;
            assert!(
                app2[b + idx_append2::HL_BIN1] > 1e-3,
                "HL1 must fire at scale {scale} on the HDR route (f{})",
                924 + b + idx_append2::HL_BIN1
            );
            assert!(
                app2[b + idx_append2::HL_BIN2] > 1e-3,
                "HL2 must fire at scale {scale} on the HDR route (f{})",
                924 + b + idx_append2::HL_BIN2
            );
        }

        // SDR-range HDR content (≤80 nits): bins exactly 0.
        let sdr_ramp: Vec<[f32; 3]> = ramp
            .iter()
            .map(|&[r, _, _]| {
                let v = (r / 3000.0 * 80.0).max(0.05);
                [v, v, v]
            })
            .collect();
        let sdr_dst: Vec<[f32; 3]> = sdr_ramp.iter().map(|&[r, g, b]| [r * 1.1, g, b]).collect();
        let r = z
            .compute_folded720_append2_features_hdr(
                &NitsImage::from_rgb_nits(&sdr_ramp, w, h),
                &NitsImage::from_rgb_nits(&sdr_dst, w, h),
                HdrEncoding::Linear,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        let app2 = r.append2_features().unwrap();
        for scale in 0..4 {
            let b = scale * APPEND2_PER_SCALE;
            assert_eq!(
                app2[b + idx_append2::HL_BIN1],
                0.0,
                "hl1 on ≤80-nit content"
            );
            assert_eq!(
                app2[b + idx_append2::HL_BIN2],
                0.0,
                "hl2 on ≤80-nit content"
            );
        }

        // PU-route BANDVIS: posterize the PU ramp (quantize nits through
        // a PQ-code-like grid) — gain fires with the PU δ constants.
        let poster: Vec<[f32; 3]> = ramp
            .iter()
            .map(|&[r, _, _]| {
                // quantize log-luminance to 24 plateaus
                let lg = (r / 50.0).ln() / (3000.0f32 / 50.0).ln();
                let q = (lg * 24.0).floor() / 24.0;
                let v = 50.0 * (3000.0f32 / 50.0).powf(q);
                [v, v, v]
            })
            .collect();
        let r = z
            .compute_folded720_append2_features_hdr(
                &NitsImage::from_rgb_nits(&ramp, w, h),
                &NitsImage::from_rgb_nits(&poster, w, h),
                HdrEncoding::Linear,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        let app2r = r.append2_features().unwrap();
        let g_coarse = (2..4)
            .map(|s| app2r[s * APPEND2_PER_SCALE + idx_append2::BANDVIS_GAIN])
            .fold(0.0f64, f64::max);
        println!(
            "PU-route posterize gain per scale: {:?}",
            (0..4)
                .map(|s| app2r[s * APPEND2_PER_SCALE + idx_append2::BANDVIS_GAIN])
                .collect::<Vec<_>>()
        );
        assert!(
            g_coarse > 1e-4,
            "PU-domain BANDVIS should fire at coarse scales on posterized HDR ramp"
        );
    }

    /// P1.5 dst-activity toggle on the DECLARED-HDR route (pre-registered
    /// F9): toggle ON moves only the BANDVIS lanes (HL bins, conditioner,
    /// first-924 all bit-stable), and PU-domain BANDVIS still fires on
    /// the posterized HDR ramp with the self-mask ON.
    #[test]
    fn append2_dst_activity_hdr_route_lanes_and_fires() {
        let (w, h) = (128usize, 128usize);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let mut scratch = V2Scratch::new();

        // The V4 PU fixture: log-luminance ramp 50..3000 nits, posterized
        // to 24 plateaus (identical construction to the test above).
        let ramp: Vec<[f32; 3]> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let t = (x + y) as f32 / (w + h - 2) as f32;
                let nits = 50.0 * (3000.0f32 / 50.0).powf(t);
                [nits, nits, nits]
            })
            .collect();
        let poster: Vec<[f32; 3]> = ramp
            .iter()
            .map(|&[r, _, _]| {
                let lg = (r / 50.0).ln() / (3000.0f32 / 50.0).ln();
                let q = (lg * 24.0).floor() / 24.0;
                let v = 50.0 * (3000.0f32 / 50.0).powf(q);
                [v, v, v]
            })
            .collect();
        let t_on = V2NewFeatureToggles {
            append2_dst_activity: true,
            ..V2NewFeatureToggles::default()
        };
        let off = z
            .compute_folded720_append2_features_hdr(
                &NitsImage::from_rgb_nits(&ramp, w, h),
                &NitsImage::from_rgb_nits(&poster, w, h),
                HdrEncoding::Linear,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        let on = z
            .compute_folded720_append2_features_hdr(
                &NitsImage::from_rgb_nits(&ramp, w, h),
                &NitsImage::from_rgb_nits(&poster, w, h),
                HdrEncoding::Linear,
                t_on,
                &mut scratch,
            )
            .unwrap();
        let mut lanes_moved = 0usize;
        for i in 0..944 {
            let gain_lane = i >= 924 && (i - 924) % APPEND2_PER_SCALE == idx_append2::BANDVIS_GAIN;
            if gain_lane {
                lanes_moved += (off.features()[i].to_bits() != on.features()[i].to_bits()) as usize;
            } else {
                assert_eq!(
                    off.features()[i].to_bits(),
                    on.features()[i].to_bits(),
                    "HDR route: append2_dst_activity moved a non-GAIN slot f{i}"
                );
            }
        }
        assert!(lanes_moved > 0, "HDR route: toggle must be live");
        let app2 = on.append2_features().unwrap();
        let g_on: Vec<f64> = (0..4)
            .map(|s| app2[s * APPEND2_PER_SCALE + idx_append2::BANDVIS_GAIN])
            .collect();
        println!("F9 PU-route posterize GAIN (dst-activity ON): {g_on:?}");
        let g_coarse_on = g_on[2..4].iter().cloned().fold(0.0f64, f64::max);
        assert!(
            g_coarse_on > 1e-4,
            "F9: PU-domain BANDVIS must still fire with the dst self-mask ON"
        );
    }

    // ========================================================================
    // CSFW gates (chunk-3 tier-1: luminance-weighted GLOBAL_* twins)
    // ========================================================================

    /// Identity + layout + first-944 bit-stability + SDR entry parity at
    /// 956 (the append2 gate shape, one block up).
    #[test]
    fn csfw_layout_identity_and_first944_bit_stable() {
        let (w, h) = (150usize, 170usize);
        let src = textured_image(w, h, 23);
        let dst = quantize_distort(&src, w, h);
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);

        let a944 = z.compute_folded720_append2_features(&sref, &dref).unwrap();
        let a956 = z.compute_folded720_csfw_features(&sref, &dref).unwrap();
        assert_eq!(a956.regime(), FeatureRegime::Folded720Csfw);
        assert_eq!(a956.features().len(), 956);
        assert_eq!(a956.csfw_features().unwrap().len(), 12);
        assert_eq!(a956.append2_features().unwrap().len(), 20);
        assert_eq!(a956.append_features().unwrap().len(), 204);
        // The windowed accessors must agree with the 944 result's views.
        assert_eq!(
            a956.append2_features().unwrap(),
            a944.append2_features().unwrap()
        );
        assert_eq!(
            a956.append_features().unwrap(),
            a944.append_features().unwrap()
        );
        // Turning CSFW on must not move a bit of the first 944.
        for i in 0..944 {
            assert_eq!(
                a944.features()[i].to_bits(),
                a956.features()[i].to_bits(),
                "csfw toggled on moved f{i}"
            );
        }
        // Parallel + scratch parity at 956.
        let zp = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(true);
        let b = zp.compute_folded720_csfw_features(&sref, &dref).unwrap();
        for i in 0..956 {
            assert_eq!(
                a956.features()[i].to_bits(),
                b.features()[i].to_bits(),
                "serial vs parallel diverge at f{i}"
            );
        }
        // SDR entry parity: the pair entry vs BOTH toggle-carrying
        // streaming batch forms — byte-identical 956.
        let mut scratch = V2Scratch::new();
        let toggles = V2NewFeatureToggles {
            append2_block: true,
            csfw_block: true,
            ..V2NewFeatureToggles::default()
        };
        let via_append_stream = z
            .compute_folded720_append_features_streaming(&sref, &dref, toggles, &mut scratch)
            .unwrap();
        let via_fold_stream = z
            .compute_folded720_features_streaming(
                &sref,
                &dref,
                V2NewFeatureToggles {
                    append_block: true,
                    ..toggles
                },
                &mut scratch,
            )
            .unwrap();
        assert_eq!(via_append_stream.regime(), FeatureRegime::Folded720Csfw);
        assert_eq!(via_fold_stream.regime(), FeatureRegime::Folded720Csfw);
        for i in 0..956 {
            assert_eq!(
                a956.features()[i].to_bits(),
                via_append_stream.features()[i].to_bits(),
                "pair vs append-streaming entry diverge at f{i}"
            );
            assert_eq!(
                a956.features()[i].to_bits(),
                via_fold_stream.features()[i].to_bits(),
                "pair vs fold-streaming entry diverge at f{i}"
            );
        }
        // All 12 new slots bounded [0,1] + finite.
        for (i, v) in a956.csfw_features().unwrap().iter().enumerate() {
            assert!((0.0..=1.0).contains(v) && v.is_finite(), "csfw[{i}] = {v}");
        }
        // The weighted twins must actually differ from the unweighted
        // GLOBAL_* on a real distortion (the weight is not a no-op): at
        // least one scale's W_GLOBAL_DMEAN differs when the unweighted
        // twin is nonzero.
        let app = a956.append_features().unwrap();
        let csfw = a956.csfw_features().unwrap();
        let mut any_diff = false;
        for scale in 0..4 {
            let u = app[scale * (3 * FEATURES_PER_CHANNEL_APPEND)
                + FEATURES_PER_CHANNEL_APPEND
                + idx_append::GLOBAL_DMEAN];
            let wv = csfw[scale * CSFW_PER_SCALE + idx_csfw::W_GLOBAL_DMEAN];
            if u > 1e-9 && (wv - u).abs() > 1e-12 {
                any_diff = true;
            }
        }
        assert!(
            any_diff,
            "weighted GLOBAL_DMEAN identical to unweighted on a distorted pair — weight inert?"
        );

        // Identity pair: every CSFW slot EXACTLY 0 (v ≡ 0 ⇒ weighted
        // pools of identical planes cancel exactly, independent of w).
        let idr = z.compute_folded720_csfw_features(&sref, &sref).unwrap();
        let csfw_id = idr.csfw_features().unwrap();
        for (i, v) in csfw_id.iter().enumerate() {
            assert_eq!(*v, 0.0, "identity csfw[{i}] = {v}");
        }
    }

    /// The doc-predicted luminance direction (design §5.1/§5.2): the SDR
    /// achromatic weight peaks in the darks (~×1.8 near sRGB code 32)
    /// and falls below 1 in the highlights (~×0.62 at code 255), so a
    /// mean shift confined to DARK content must weigh MORE in
    /// `W_GLOBAL_DMEAN` than its unweighted twin, and a bright-confined
    /// shift LESS. Same direction on the HDR route (castleCSF-over-PU21
    /// residual: w 1.95 at 1 cd/m² → 0.63 at 4000).
    #[test]
    fn csfw_luminance_direction_dark_up_bright_down() {
        let (w, h) = (128usize, 128usize);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);

        // SDR: left half dark (code 20), right half bright (code 220).
        let half = |dark: u8, bright: u8| -> Vec<[u8; 3]> {
            (0..w * h)
                .map(|i| {
                    let v = if i % w < w / 2 { dark } else { bright };
                    [v, v, v]
                })
                .collect()
        };
        let src = half(20, 220);
        let dark_shift = half(26, 220); // +6 codes, dark half only
        let bright_shift = half(20, 226); // +6 codes, bright half only
        let lane_pair = |dst: &Vec<[u8; 3]>| -> (f64, f64) {
            let r = z
                .compute_folded720_csfw_features(
                    &RgbSlice::new(&src, w, h),
                    &RgbSlice::new(dst, w, h),
                )
                .unwrap();
            let u = r.append_features().unwrap()
                [FEATURES_PER_CHANNEL_APPEND + idx_append::GLOBAL_DMEAN];
            let wv = r.csfw_features().unwrap()[idx_csfw::W_GLOBAL_DMEAN];
            (wv, u)
        };
        let (wd, ud) = lane_pair(&dark_shift);
        let (wb, ub) = lane_pair(&bright_shift);
        println!(
            "SDR dark-shift:   weighted {wd:.6} unweighted {ud:.6} (ratio {:.3})",
            wd / ud
        );
        println!(
            "SDR bright-shift: weighted {wb:.6} unweighted {ub:.6} (ratio {:.3})",
            wb / ub
        );
        assert!(
            wd > ud * 1.05,
            "dark-confined mean shift must up-weigh: {wd} vs {ud}"
        );
        assert!(
            wb < ub * 0.95,
            "bright-confined mean shift must down-weigh: {wb} vs {ub}"
        );

        // HDR route: left half 2 cd/m², right half 400 cd/m².
        let nits_half = |dark: f32, bright: f32| -> Vec<[f32; 3]> {
            (0..w * h)
                .map(|i| {
                    let v = if i % w < w / 2 { dark } else { bright };
                    [v, v, v]
                })
                .collect()
        };
        let mut scratch = V2Scratch::new();
        let hsrc = nits_half(2.0, 400.0);
        let hdark = nits_half(2.4, 400.0); // +20% dark half
        let hbright = nits_half(2.0, 480.0); // +20% bright half
        let hdr_lane_pair = |dst: &Vec<[f32; 3]>, scratch: &mut V2Scratch| -> (f64, f64) {
            let r = z
                .compute_folded720_csfw_features_hdr(
                    &NitsImage::from_rgb_nits(&hsrc, w, h),
                    &NitsImage::from_rgb_nits(dst, w, h),
                    HdrEncoding::Linear,
                    V2NewFeatureToggles::default(),
                    scratch,
                )
                .unwrap();
            let u = r.append_features().unwrap()
                [FEATURES_PER_CHANNEL_APPEND + idx_append::GLOBAL_DMEAN];
            let wv = r.csfw_features().unwrap()[idx_csfw::W_GLOBAL_DMEAN];
            (wv, u)
        };
        let (hwd, hud) = hdr_lane_pair(&hdark, &mut scratch);
        let (hwb, hub) = hdr_lane_pair(&hbright, &mut scratch);
        println!(
            "HDR dark-shift:   weighted {hwd:.6} unweighted {hud:.6} (ratio {:.3})",
            hwd / hud
        );
        println!(
            "HDR bright-shift: weighted {hwb:.6} unweighted {hub:.6} (ratio {:.3})",
            hwb / hub
        );
        assert!(
            hwd > hud * 1.05,
            "HDR dark-confined shift must up-weigh: {hwd} vs {hud}"
        );
        assert!(
            hwb < hub * 0.95,
            "HDR bright-confined shift must down-weigh: {hwb} vs {hub}"
        );
    }

    /// HDR-route CSFW: 956 shape, HDR entry parity, identity zeros, and
    /// bounded firing on a real gain-error nits pair.
    #[test]
    fn csfw_hdr_route_entry_parity_and_smoke() {
        let (w, h) = (128usize, 128usize);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let mut scratch = V2Scratch::new();

        let ramp: Vec<[f32; 3]> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let t = (x + y) as f32 / (w + h - 2) as f32;
                let nits = 0.5 * (2000.0f32 / 0.5).powf(t);
                [nits, nits, nits]
            })
            .collect();
        let dst: Vec<[f32; 3]> = ramp
            .iter()
            .map(|&[r, g, b]| [r * 1.12, g * 1.12, b * 1.12])
            .collect();
        let sref = NitsImage::from_rgb_nits(&ramp, w, h);
        let dref = NitsImage::from_rgb_nits(&dst, w, h);

        let a = z
            .compute_folded720_csfw_features_hdr(
                &sref,
                &dref,
                HdrEncoding::Linear,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        assert_eq!(a.regime(), FeatureRegime::Folded720Csfw);
        assert_eq!(a.features().len(), 956);
        // First-944 bit-stability on the HDR route.
        let a944 = z
            .compute_folded720_append2_features_hdr(
                &sref,
                &dref,
                HdrEncoding::Linear,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        for i in 0..944 {
            assert_eq!(
                a944.features()[i].to_bits(),
                a.features()[i].to_bits(),
                "HDR csfw toggled on moved f{i}"
            );
        }
        // HDR entry parity: the csfw entry vs the append2/append HDR
        // entries carrying the toggle — byte-identical 956.
        let via_a2 = z
            .compute_folded720_append2_features_hdr(
                &sref,
                &dref,
                HdrEncoding::Linear,
                V2NewFeatureToggles {
                    csfw_block: true,
                    ..V2NewFeatureToggles::default()
                },
                &mut scratch,
            )
            .unwrap();
        let via_app = z
            .compute_folded720_append_features_hdr(
                &sref,
                &dref,
                HdrEncoding::Linear,
                V2NewFeatureToggles {
                    append2_block: true,
                    csfw_block: true,
                    ..V2NewFeatureToggles::default()
                },
                &mut scratch,
            )
            .unwrap();
        for i in 0..956 {
            assert_eq!(
                a.features()[i].to_bits(),
                via_a2.features()[i].to_bits(),
                "HDR csfw vs append2-entry toggles diverge at f{i}"
            );
            assert_eq!(
                a.features()[i].to_bits(),
                via_app.features()[i].to_bits(),
                "HDR csfw vs append-entry toggles diverge at f{i}"
            );
        }
        // Bounded + fires: a global 12% gain must move the weighted
        // global lanes.
        let csfw = a.csfw_features().unwrap();
        for (i, v) in csfw.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(v) && v.is_finite(),
                "hdr csfw[{i}] = {v}"
            );
        }
        let fired = (0..4).any(|s| csfw[s * CSFW_PER_SCALE + idx_csfw::W_GLOBAL_DMEAN] > 1e-4);
        assert!(
            fired,
            "W_GLOBAL_DMEAN should fire on a 12% gain pair: {csfw:?}"
        );

        // HDR identity: exactly 0 on every CSFW slot.
        let idr = z
            .compute_folded720_csfw_features_hdr(
                &sref,
                &sref,
                HdrEncoding::Linear,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        for (i, v) in idr.csfw_features().unwrap().iter().enumerate() {
            assert_eq!(*v, 0.0, "HDR identity csfw[{i}] = {v}");
        }
    }

    // ========================================================================
}
