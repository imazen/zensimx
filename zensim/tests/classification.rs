#![cfg(feature = "classification")]
#![allow(deprecated)]
// exercises the deprecated `ZensimProfile::A` (shipped behind the default-on `deprecated-profiles` feature)
//! Error classification tests against PNG fixture corpus.
//!
//! Validates that:
//! 1. DeltaStats features match expected patterns for each distortion type
//! 2. Classification assigns correct ErrorCategory
//! 3. Negative controls (noise) don't false-positive
//! 4. classify().result.score() matches compute().score()
//!
//! Run with: `cargo test -p zensim --all-features --test classification -- --nocapture`

mod common;

use image::ImageReader;
use std::path::Path;
use zensim::{ClassifiedResult, ErrorCategory, RgbSlice, RgbaSlice, Zensim, ZensimProfile};

fn fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn load_rgb(name: &str) -> (Vec<[u8; 3]>, usize, usize) {
    let path = fixtures_dir().join(name);
    let img = ImageReader::open(&path)
        .unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()))
        .decode()
        .unwrap_or_else(|e| panic!("Failed to decode {}: {e}", path.display()))
        .to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let pixels: Vec<[u8; 3]> = img.pixels().map(|p| p.0).collect();
    (pixels, w, h)
}

fn load_rgba(name: &str) -> (Vec<[u8; 4]>, usize, usize) {
    let path = fixtures_dir().join(name);
    let img = ImageReader::open(&path)
        .unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()))
        .decode()
        .unwrap_or_else(|e| panic!("Failed to decode {}: {e}", path.display()))
        .to_rgba8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let pixels: Vec<[u8; 4]> = img.pixels().map(|p| p.0).collect();
    (pixels, w, h)
}

fn classify_rgb(src_name: &str, dst_name: &str) -> ClassifiedResult {
    let (src, w, h) = load_rgb(src_name);
    let (dst, dw, dh) = load_rgb(dst_name);
    assert_eq!(
        (w, h),
        (dw, dh),
        "dimension mismatch: {src_name} vs {dst_name}"
    );
    let z = Zensim::new(ZensimProfile::A);
    let src_img = RgbSlice::new(&src, w, h);
    let dst_img = RgbSlice::new(&dst, w, h);
    z.classify(&src_img, &dst_img).expect("classify failed")
}

fn classify_rgba(src_name: &str, dst_name: &str) -> ClassifiedResult {
    let (src, w, h) = load_rgba(src_name);
    let (dst, dw, dh) = load_rgba(dst_name);
    assert_eq!(
        (w, h),
        (dw, dh),
        "dimension mismatch: {src_name} vs {dst_name}"
    );
    let z = Zensim::new(ZensimProfile::A);
    let src_img = RgbaSlice::new(&src, w, h);
    let dst_img = RgbaSlice::new(&dst, w, h);
    z.classify(&src_img, &dst_img).expect("classify failed")
}

fn print_delta_stats(name: &str, cr: &ClassifiedResult) {
    let ds = &cr.delta_stats;
    println!("  {name}:");
    println!("    score: {:.4}", cr.result.score());
    println!(
        "    dominant: {:?} (confidence: {:.3})",
        cr.classification.dominant, cr.classification.confidence,
    );
    println!(
        "    mean_delta:     R={:.6} G={:.6} B={:.6}",
        ds.mean_delta[0], ds.mean_delta[1], ds.mean_delta[2],
    );
    println!(
        "    stddev_delta:   R={:.6} G={:.6} B={:.6}",
        ds.stddev_delta[0], ds.stddev_delta[1], ds.stddev_delta[2],
    );
    println!(
        "    max_abs_delta:  R={:.6} G={:.6} B={:.6}",
        ds.max_abs_delta[0], ds.max_abs_delta[1], ds.max_abs_delta[2],
    );
    println!(
        "    pixels_differing: {} / {} ({:.1}%)",
        ds.pixels_differing,
        ds.pixel_count,
        ds.pixels_differing as f64 / ds.pixel_count as f64 * 100.0,
    );
    println!(
        "    pixels_differing_by_more_than_1: {}",
        ds.pixels_differing_by_more_than_1,
    );
    // Signed small-delta histogram per channel
    let labels = ["-3", "-2", "-1", " 0", "+1", "+2", "+3"];
    for ch_name in ["R", "G", "B"] {
        let ch = match ch_name {
            "R" => 0,
            "G" => 1,
            _ => 2,
        };
        let h = &ds.signed_small_histogram[ch];
        let parts: Vec<String> = (0..7).map(|i| format!("{}:{}", labels[i], h[i])).collect();
        println!("    signed_small[{ch_name}]: {}", parts.join("  "));
    }
    println!(
        "    classification: {:?} (confidence={:.3})",
        cr.classification.dominant, cr.classification.confidence,
    );
    if let Some(ref bias) = cr.classification.rounding_bias {
        println!(
            "    rounding_bias: pos_frac=[{:.3}, {:.3}, {:.3}] balanced={}",
            bias.positive_fraction[0],
            bias.positive_fraction[1],
            bias.positive_fraction[2],
            bias.balanced,
        );
    }
}

// ─── classify() score identity ─────────────────────────────────────────

/// classify().result.score() must be bit-identical to compute().score().
#[test]
fn classify_score_matches_compute() {
    let z = Zensim::new(ZensimProfile::A);
    let (src, w, h) = load_rgb("gradient.png");
    let (dst, _, _) = load_rgb("gradient_gamma22.png");
    let src_img = RgbSlice::new(&src, w, h);
    let dst_img = RgbSlice::new(&dst, w, h);

    let compute_result = z.compute(&src_img, &dst_img).expect("compute failed");
    let classify_result = z.classify(&src_img, &dst_img).expect("classify failed");

    assert_eq!(
        compute_result.score().to_bits(),
        classify_result.result.score().to_bits(),
        "classify score must be bit-identical to compute score: {} vs {}",
        compute_result.score(),
        classify_result.result.score(),
    );
    assert_eq!(
        compute_result.raw_distance().to_bits(),
        classify_result.result.raw_distance().to_bits(),
        "classify raw_distance must be bit-identical",
    );
    println!("  score identity verified: {:.6}", compute_result.score());
}

// ─── Identical images ──────────────────────────────────────────────────

#[test]
fn identical_images_classified_as_identical() {
    let (src, w, h) = load_rgb("gradient.png");
    let z = Zensim::new(ZensimProfile::A);
    let src_img = RgbSlice::new(&src, w, h);
    let copy = src.clone();
    let dst_img = RgbSlice::new(&copy, w, h);
    let cr = z.classify(&src_img, &dst_img).expect("classify failed");

    assert_eq!(cr.classification.dominant, ErrorCategory::Identical);
    assert_eq!(cr.delta_stats.pixels_differing, 0);
    assert_eq!(cr.result.score(), 100.0);
    println!("  identical: score=100, dominant=Identical");
}

// ─── Transfer function errors (no longer detected — Unclassified) ──────

#[test]
fn gamma_22_is_unclassified() {
    let cr = classify_rgb("gradient.png", "gradient_gamma22.png");
    print_delta_stats("gradient → gamma22", &cr);

    // Gamma errors produce nonlinear deltas but we no longer offer TF classification
    // because real-world validation showed 0/5 correct (all false positives).
    assert_eq!(
        cr.classification.dominant,
        ErrorCategory::Unclassified,
        "gamma 2.2 should be Unclassified (TF detector removed), got {:?}",
        cr.classification.dominant,
    );
}

#[test]
fn gamma_18_is_unclassified() {
    let cr = classify_rgb("gradient.png", "gradient_gamma18.png");
    print_delta_stats("gradient → gamma18", &cr);
    assert_eq!(
        cr.classification.dominant,
        ErrorCategory::Unclassified,
        "gamma 1.8 should be Unclassified (TF detector removed), got {:?}",
        cr.classification.dominant,
    );
}

#[test]
fn double_srgb_is_unclassified() {
    let cr = classify_rgb("gradient.png", "gradient_double_srgb.png");
    print_delta_stats("gradient → double_srgb", &cr);
    assert_eq!(
        cr.classification.dominant,
        ErrorCategory::Unclassified,
        "double sRGB should be Unclassified (TF detector removed), got {:?}",
        cr.classification.dominant,
    );
}

#[test]
fn linear_as_srgb_is_unclassified() {
    let cr = classify_rgb("gradient.png", "gradient_linear_as_srgb.png");
    print_delta_stats("gradient → linear_as_srgb", &cr);
    assert!(
        cr.delta_stats.max_abs_delta.iter().any(|d| *d > 0.1),
        "linear_as_srgb should have large max delta",
    );
    assert_eq!(
        cr.classification.dominant,
        ErrorCategory::Unclassified,
        "linear_as_srgb should be Unclassified (TF detector removed), got {:?}",
        cr.classification.dominant,
    );
}

// ─── Channel swap ──────────────────────────────────────────────────────

#[test]
fn rgb_bgr_detected_as_channel_swap() {
    let cr = classify_rgb("blocks.png", "blocks_rgb_bgr.png");
    print_delta_stats("blocks → rgb_bgr", &cr);

    // G channel should have zero delta (it's unchanged in RGB↔BGR)
    assert!(
        cr.delta_stats.max_abs_delta[1] < 0.01,
        "G channel should be unchanged in RGB↔BGR swap, max_abs_delta[G]={:.6}",
        cr.delta_stats.max_abs_delta[1],
    );
    // R and B should have large deltas
    assert!(
        cr.delta_stats.max_abs_delta[0] > 0.05,
        "R channel should differ in RGB↔BGR, max_abs_delta[R]={:.6}",
        cr.delta_stats.max_abs_delta[0],
    );
    assert!(
        cr.classification.confidence > 0.3,
        "RGB↔BGR should have high confidence, got {:.3}",
        cr.classification.confidence,
    );
    assert_eq!(
        cr.classification.dominant,
        ErrorCategory::ChannelSwap,
        "RGB↔BGR dominant should be ChannelSwap, got {:?}",
        cr.classification.dominant,
    );
}

#[test]
fn cbcr_swap_has_channel_asymmetry() {
    let cr = classify_rgb("blocks.png", "blocks_cbcr_swap.png");
    print_delta_stats("blocks → cbcr_swap", &cr);

    // CbCr swap should produce visible differences
    let max_delta = cr
        .delta_stats
        .max_abs_delta
        .iter()
        .copied()
        .fold(0.0f64, f64::max);
    assert!(
        max_delta > 0.05,
        "CbCr swap should produce visible deltas, max={max_delta:.4}",
    );
}

// ─── Bit depth / quantization ──────────────────────────────────────────

#[test]
fn truncate_lsb_detected_as_rounding_error() {
    let cr = classify_rgb("gradient.png", "gradient_truncate.png");
    print_delta_stats("gradient → truncate", &cr);

    let ds = &cr.delta_stats;
    // Max delta should be <= 1/255
    let max_d = ds.max_abs_delta.iter().copied().fold(0.0f64, f64::max);
    assert!(
        max_d < 2.0 / 255.0,
        "truncation max delta should be ~1/255, got {max_d:.6}",
    );
    // No pixel differs by more than 1 — provably off-by-1
    assert_eq!(
        ds.pixels_differing_by_more_than_1, 0,
        "truncation should have zero pixels differing by more than 1",
    );
    assert_eq!(
        cr.classification.confidence, 1.0,
        "truncation should have confidence = 1.0, got {:.3}",
        cr.classification.confidence,
    );
    assert_eq!(
        cr.classification.dominant,
        ErrorCategory::RoundingError,
        "truncation dominant should be RoundingError, got {:?}",
        cr.classification.dominant,
    );
    // Rounding bias should be present
    let bias = cr
        .classification
        .rounding_bias
        .as_ref()
        .expect("rounding_bias should be populated for RoundingError");
    // LSB truncation is unidirectional — all errors should be in one direction
    // (positive, since truncation rounds toward zero = dst is lower)
    for (ch, &pos_frac) in bias.positive_fraction.iter().enumerate() {
        assert!(
            pos_frac > 0.9 || pos_frac < 0.1,
            "truncation should be heavily skewed, ch={ch} pos_frac={:.3}",
            pos_frac,
        );
    }
    assert!(
        !bias.balanced,
        "truncation should NOT be balanced (it's systematic)",
    );
}

#[test]
fn depth4_is_unclassified() {
    let cr = classify_rgb("gradient.png", "gradient_depth4.png");
    print_delta_stats("gradient → depth4", &cr);

    // 4-bit quantization: max delta up to 8/255, too large for RoundingError
    let max_d = cr
        .delta_stats
        .max_abs_delta
        .iter()
        .copied()
        .fold(0.0f64, f64::max);
    assert!(
        max_d > 0.01 && max_d < 0.1,
        "depth4 max delta should be moderate, got {max_d:.4}",
    );
    assert_eq!(
        cr.classification.dominant,
        ErrorCategory::Unclassified,
        "depth4 should be Unclassified (max_delta > 3/255), got {:?}",
        cr.classification.dominant,
    );
}

#[test]
fn depth2_is_unclassified() {
    let cr = classify_rgb("gradient.png", "gradient_depth2.png");
    print_delta_stats("gradient → depth2", &cr);

    // 2-bit quantization: very heavy posterization, well beyond rounding
    let max_d = cr
        .delta_stats
        .max_abs_delta
        .iter()
        .copied()
        .fold(0.0f64, f64::max);
    assert!(
        max_d > 0.05,
        "depth2 should have large max delta, got {max_d:.4}",
    );
    assert_eq!(
        cr.classification.dominant,
        ErrorCategory::Unclassified,
        "depth2 should be Unclassified (max_delta >> 3/255), got {:?}",
        cr.classification.dominant,
    );
}

#[test]
fn expand256_classified_correctly() {
    let cr = classify_rgb("gradient.png", "gradient_expand256.png");
    print_delta_stats("gradient → expand256", &cr);

    // expand256: very subtle error, delta proportional to value
    let ds = &cr.delta_stats;
    let max_d = ds.max_abs_delta.iter().copied().fold(0.0f64, f64::max);
    assert!(
        max_d < 5.0 / 255.0,
        "expand256 max delta should be small, got {max_d:.6}",
    );
    // If max_delta ≤ 3/255, should be RoundingError; otherwise Unclassified
    if max_d <= 3.0 / 255.0 {
        assert_eq!(
            cr.classification.dominant,
            ErrorCategory::RoundingError,
            "expand256 with max_delta ≤ 3/255 should be RoundingError, got {:?}",
            cr.classification.dominant,
        );
    } else {
        assert_eq!(
            cr.classification.dominant,
            ErrorCategory::Unclassified,
            "expand256 with max_delta > 3/255 should be Unclassified, got {:?}",
            cr.classification.dominant,
        );
    }
}

// ─── Color space / ICC matrix ──────────────────────────────────────────

#[test]
fn adobe_as_srgb_detected() {
    let cr = classify_rgb("blocks.png", "blocks_adobe_as_srgb.png");
    print_delta_stats("blocks → adobe_as_srgb", &cr);

    // NOTE: IM6 `-set colorspace AdobeRGB -colorspace sRGB` just relabels
    // data without ICC profile conversion, so images may be identical.
    // This test validates the fixture loads and classify() doesn't crash.
    // A real AdobeRGB→sRGB conversion would produce visible differences
    // on non-primary colors.
    let max_d = cr
        .delta_stats
        .max_abs_delta
        .iter()
        .copied()
        .fold(0.0f64, f64::max);
    println!("    (max_d = {max_d:.6}, IM6 may not apply ICC conversion)");
}

// ─── Alpha compositing ────────────────────────────────────────────────

#[test]
fn premul_as_straight_detected() {
    let cr = classify_rgba("alpha_patches.png", "alpha_patches_premul_as_straight.png");
    print_delta_stats("alpha_patches → premul_as_straight", &cr);

    // Opaque pixels should have zero delta
    if let Some(ref opaque) = cr.delta_stats.opaque_stats {
        let opaque_max = opaque.mean_abs_delta.iter().copied().fold(0.0f64, f64::max);
        assert!(
            opaque_max < 0.01,
            "premul_as_straight: opaque pixels should have near-zero delta, got {opaque_max:.6}",
        );
    }
    // Semitransparent should have large delta
    if let Some(ref semi) = cr.delta_stats.semitransparent_stats {
        let semi_mean = semi.mean_abs_delta.iter().copied().fold(0.0f64, f64::max);
        assert!(
            semi_mean > 0.01,
            "premul_as_straight: semitransparent should have visible delta, got {semi_mean:.6}",
        );
    }
    assert!(
        cr.classification.confidence > 0.3,
        "premul_as_straight should have high confidence, got {:.3}",
        cr.classification.confidence,
    );
    assert_eq!(
        cr.classification.dominant,
        ErrorCategory::AlphaCompositing,
        "premul_as_straight dominant should be AlphaCompositing, got {:?}",
        cr.classification.dominant,
    );
}

#[test]
fn straight_as_premul_detected() {
    let cr = classify_rgba("alpha_patches.png", "alpha_patches_straight_as_premul.png");
    print_delta_stats("alpha_patches → straight_as_premul", &cr);

    // Opaque should be unchanged
    if let Some(ref opaque) = cr.delta_stats.opaque_stats {
        let opaque_max = opaque.mean_abs_delta.iter().copied().fold(0.0f64, f64::max);
        assert!(
            opaque_max < 0.01,
            "straight_as_premul: opaque should be unchanged, got {opaque_max:.6}",
        );
    }
    assert!(
        cr.classification.confidence > 0.3,
        "straight_as_premul should have high confidence, got {:.3}",
        cr.classification.confidence,
    );
    assert_eq!(
        cr.classification.dominant,
        ErrorCategory::AlphaCompositing,
        "straight_as_premul dominant should be AlphaCompositing, got {:?}",
        cr.classification.dominant,
    );
}

#[test]
fn alpha_gradient_premul_error() {
    let cr = classify_rgba(
        "alpha_gradient.png",
        "alpha_gradient_premul_as_straight.png",
    );
    print_delta_stats("alpha_gradient → premul_as_straight", &cr);

    // Error should correlate with alpha
    if let Some(corr) = cr.delta_stats.alpha_error_correlation {
        println!("    alpha_error_correlation: {corr:.4}");
        // For premul error on gradient alpha, correlation should be positive
        // (more transparent = more error)
        assert!(
            corr > 0.3,
            "premul error should correlate with transparency, got {corr:.4}",
        );
    }
    assert_eq!(
        cr.classification.dominant,
        ErrorCategory::AlphaCompositing,
        "alpha_gradient premul dominant should be AlphaCompositing, got {:?}",
        cr.classification.dominant,
    );
}

#[test]
fn wrong_bg_black_detected() {
    let cr = classify_rgba("alpha_gradient.png", "alpha_gradient_wrong_bg_black.png");
    print_delta_stats("alpha_gradient → wrong_bg_black", &cr);

    // Error proportional to (1-alpha)
    if let Some(corr) = cr.delta_stats.alpha_error_correlation {
        println!("    alpha_error_correlation: {corr:.4}");
        assert!(
            corr > 0.8,
            "wrong_bg_black should have high alpha-error correlation, got {corr:.4}",
        );
    }
    assert_eq!(
        cr.classification.dominant,
        ErrorCategory::AlphaCompositing,
        "wrong_bg_black dominant should be AlphaCompositing, got {:?}",
        cr.classification.dominant,
    );
}

// ─── Noise (no longer detected — Unclassified) ─────────────────────────

#[test]
fn low_noise_is_unclassified() {
    let cr = classify_rgb("gradient.png", "gradient_noise_low.png");
    print_delta_stats("gradient → noise_low", &cr);

    // Noise detector removed — no defensible signature for random noise
    assert_eq!(
        cr.classification.dominant,
        ErrorCategory::Unclassified,
        "low noise should be Unclassified (noise detector removed), got {:?}",
        cr.classification.dominant,
    );
}

#[test]
fn high_noise_is_unclassified() {
    let cr = classify_rgb("gradient.png", "gradient_noise_high.png");
    print_delta_stats("gradient → noise_high", &cr);

    assert_eq!(
        cr.classification.dominant,
        ErrorCategory::Unclassified,
        "high noise should be Unclassified (noise detector removed), got {:?}",
        cr.classification.dominant,
    );
}

// ─── Determinism ───────────────────────────────────────────────────────

#[test]
fn delta_stats_deterministic() {
    let (src, w, h) = load_rgb("gradient.png");
    let (dst, _, _) = load_rgb("gradient_gamma22.png");
    let z = Zensim::new(ZensimProfile::A);
    let src_img = RgbSlice::new(&src, w, h);
    let dst_img = RgbSlice::new(&dst, w, h);

    let r1 = z.classify(&src_img, &dst_img).unwrap();
    let r2 = z.classify(&src_img, &dst_img).unwrap();
    let r3 = z.classify(&src_img, &dst_img).unwrap();

    // Delta stats must be deterministic across runs
    for c in 0..3 {
        assert_eq!(
            r1.delta_stats.mean_delta[c].to_bits(),
            r2.delta_stats.mean_delta[c].to_bits(),
            "mean_delta[{c}] not deterministic",
        );
        assert_eq!(
            r1.delta_stats.mean_delta[c].to_bits(),
            r3.delta_stats.mean_delta[c].to_bits(),
            "mean_delta[{c}] not deterministic (run 3)",
        );
        assert_eq!(
            r1.delta_stats.stddev_delta[c].to_bits(),
            r2.delta_stats.stddev_delta[c].to_bits(),
            "stddev_delta[{c}] not deterministic",
        );
    }
    assert_eq!(r1.delta_stats.pixel_count, r2.delta_stats.pixel_count);
    assert_eq!(
        r1.delta_stats.pixels_differing,
        r2.delta_stats.pixels_differing
    );
    println!("  delta stats deterministic across 3 runs");
}

// ─── Flatten over background (IM-generated) ────────────────────────────

#[test]
fn flatten_over_black_vs_white() {
    // These are RGB images (alpha removed by flatten), so we compare
    // the two composited results to show different background effects
    let cr = classify_rgb(
        "alpha_patches_over_black.png",
        "alpha_patches_over_white.png",
    );
    print_delta_stats("over_black → over_white", &cr);

    // They should differ significantly
    let max_d = cr
        .delta_stats
        .max_abs_delta
        .iter()
        .copied()
        .fold(0.0f64, f64::max);
    assert!(
        max_d > 0.1,
        "flatten over black vs white should differ significantly, max={max_d:.4}",
    );
}

// ─── Round half up ────────────────────────────────────────────────────

#[test]
fn round_half_up_detected_as_rounding_error() {
    let cr = classify_rgb("gradient.png", "gradient_round_half_up.png");
    print_delta_stats("gradient → round_half_up", &cr);

    // Off-by-1 rounding: max delta ≤ 1/255, zero pixels > 1 delta
    let max_d = cr
        .delta_stats
        .max_abs_delta
        .iter()
        .copied()
        .fold(0.0f64, f64::max);
    assert!(
        max_d <= 2.0 / 255.0,
        "round_half_up max delta should be tiny, got {max_d:.6}",
    );
    assert_eq!(
        cr.classification.dominant,
        ErrorCategory::RoundingError,
        "round_half_up should be RoundingError, got {:?}",
        cr.classification.dominant,
    );
    // Rounding bias should be present
    let bias = cr
        .classification
        .rounding_bias
        .as_ref()
        .expect("rounding_bias should be populated for RoundingError");
    // round_half_up vs default rounding: errors should be unidirectional
    // (all negative, since round_half_up rounds UP = dst is higher)
    for (ch, &pos_frac) in bias.positive_fraction.iter().enumerate() {
        assert!(
            pos_frac > 0.9 || pos_frac < 0.1,
            "round_half_up should be heavily skewed, ch={ch} pos_frac={:.3}",
            pos_frac,
        );
    }
}

// ─── Gray ramp as source ───────────────────────────────────────────────

#[test]
fn gray_ramp_self_identical() {
    let (src, w, h) = load_rgb("gray_ramp.png");
    let z = Zensim::new(ZensimProfile::A);
    let src_img = RgbSlice::new(&src, w, h);
    let copy = src.clone();
    let dst_img = RgbSlice::new(&copy, w, h);
    let cr = z.classify(&src_img, &dst_img).unwrap();

    assert_eq!(cr.classification.dominant, ErrorCategory::Identical);
    assert_eq!(cr.delta_stats.pixels_differing, 0);
    println!("  gray_ramp: identical verified");
}
