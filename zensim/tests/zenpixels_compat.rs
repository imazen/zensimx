#![cfg(feature = "zenpixels")]
#![allow(deprecated)]
// exercises the deprecated `ZensimProfile::A` (shipped behind the default-on `deprecated-profiles` feature)
//! End-to-end integration tests for the zenpixels ImageSource adapter.
//!
//! Validates that `ZenpixelsSource` produces identical scores to the
//! equivalent native zensim types (RgbSlice, RgbaSlice, StridedBytes).
//!
//! Run with: `cargo test -p zensim --features zenpixels --test zenpixels_compat`

use zenpixels::{AlphaMode as ZpAlpha, ImageSpec, PixelBuf, PixelView, Transfer, specs};
use zensim::{
    AlphaMode, ImageSource, RgbSlice, RgbaSlice, StridedBytes, ZenpixelsSource, Zensim,
    ZensimProfile,
};

fn zensim() -> Zensim {
    Zensim::new(ZensimProfile::A)
}

/// Generate deterministic 16x16 test pixels (gradient pattern).
fn test_pixels_rgb() -> Vec<[u8; 3]> {
    (0..16 * 16)
        .map(|i| {
            let x = (i % 16) as u8;
            let y = (i / 16) as u8;
            [
                x.wrapping_mul(17),
                y.wrapping_mul(17),
                (x ^ y).wrapping_mul(11),
            ]
        })
        .collect()
}

fn test_pixels_rgba() -> Vec<[u8; 4]> {
    test_pixels_rgb()
        .into_iter()
        .map(|[r, g, b]| [r, g, b, 255])
        .collect()
}

fn test_pixels_rgba_with_alpha() -> Vec<[u8; 4]> {
    (0..16 * 16)
        .map(|i| {
            let x = (i % 16) as u8;
            let y = (i / 16) as u8;
            let a = if x < 8 { 255 } else { 128 };
            [
                x.wrapping_mul(17),
                y.wrapping_mul(17),
                (x ^ y).wrapping_mul(11),
                a,
            ]
        })
        .collect()
}

fn distorted_pixels_rgb() -> Vec<[u8; 3]> {
    test_pixels_rgb()
        .into_iter()
        .map(|[r, g, b]| [r.wrapping_add(5), g, b.wrapping_sub(3)])
        .collect()
}

fn distorted_pixels_rgba() -> Vec<[u8; 4]> {
    distorted_pixels_rgb()
        .into_iter()
        .map(|[r, g, b]| [r, g, b, 255])
        .collect()
}

#[test]
fn rgb8_matches_rgbslice() {
    let z = zensim();
    let src = test_pixels_rgb();
    let dst = distorted_pixels_rgb();

    // Native
    let native = z
        .compute(&RgbSlice::new(&src, 16, 16), &RgbSlice::new(&dst, 16, 16))
        .unwrap();

    // Via zenpixels
    let src_bytes: Vec<u8> = src.iter().flat_map(|p| p.iter().copied()).collect();
    let dst_bytes: Vec<u8> = dst.iter().flat_map(|p| p.iter().copied()).collect();
    let src_slice = PixelView::new(&src_bytes, 16, 16, 16 * 3, &specs::RGB8_SRGB).unwrap();
    let dst_slice = PixelView::new(&dst_bytes, 16, 16, 16 * 3, &specs::RGB8_SRGB).unwrap();
    let zp_src = ZenpixelsSource::try_from_slice(&src_slice).unwrap();
    let zp_dst = ZenpixelsSource::try_from_slice(&dst_slice).unwrap();
    let zp_result = z.compute(&zp_src, &zp_dst).unwrap();

    assert!(
        (native.score() - zp_result.score()).abs() < 0.001,
        "RGB8 score mismatch: native={}, zenpixels={}",
        native.score(),
        zp_result.score()
    );
}

#[test]
fn rgba8_matches_rgbaslice() {
    let z = zensim();
    let src = test_pixels_rgba();
    let dst = distorted_pixels_rgba();

    let native = z
        .compute(&RgbaSlice::new(&src, 16, 16), &RgbaSlice::new(&dst, 16, 16))
        .unwrap();

    let src_bytes: Vec<u8> = src.iter().flat_map(|p| p.iter().copied()).collect();
    let dst_bytes: Vec<u8> = dst.iter().flat_map(|p| p.iter().copied()).collect();
    let src_slice = PixelView::new(&src_bytes, 16, 16, 16 * 4, &specs::RGBA8_SRGB).unwrap();
    let dst_slice = PixelView::new(&dst_bytes, 16, 16, 16 * 4, &specs::RGBA8_SRGB).unwrap();
    let zp_src = ZenpixelsSource::try_from_slice(&src_slice).unwrap();
    let zp_dst = ZenpixelsSource::try_from_slice(&dst_slice).unwrap();
    let zp_result = z.compute(&zp_src, &zp_dst).unwrap();

    assert!(
        (native.score() - zp_result.score()).abs() < 0.001,
        "RGBA8 score mismatch: native={}, zenpixels={}",
        native.score(),
        zp_result.score()
    );
}

#[test]
fn rgbx8_treated_as_opaque() {
    let z = zensim();
    let src = test_pixels_rgba();
    let dst = distorted_pixels_rgba();

    // Native with explicit Opaque
    let native = z
        .compute(
            &RgbaSlice::with_alpha_mode(&src, 16, 16, AlphaMode::Opaque),
            &RgbaSlice::with_alpha_mode(&dst, 16, 16, AlphaMode::Opaque),
        )
        .unwrap();

    // RGBX8 — padding byte should be ignored
    let src_bytes: Vec<u8> = src.iter().flat_map(|p| p.iter().copied()).collect();
    let dst_bytes: Vec<u8> = dst.iter().flat_map(|p| p.iter().copied()).collect();
    let src_slice = PixelView::new(&src_bytes, 16, 16, 16 * 4, &specs::RGBX8_SRGB).unwrap();
    let dst_slice = PixelView::new(&dst_bytes, 16, 16, 16 * 4, &specs::RGBX8_SRGB).unwrap();
    let zp_src = ZenpixelsSource::try_from_slice(&src_slice).unwrap();
    let zp_dst = ZenpixelsSource::try_from_slice(&dst_slice).unwrap();

    assert_eq!(zp_src.alpha_mode(), AlphaMode::Opaque);
    let zp_result = z.compute(&zp_src, &zp_dst).unwrap();

    assert!(
        (native.score() - zp_result.score()).abs() < 0.001,
        "RGBX8 score mismatch: native={}, zenpixels={}",
        native.score(),
        zp_result.score()
    );
}

#[test]
fn bgra8_matches_stridedbytes() {
    let z = zensim();
    // BGRA layout: [B, G, R, A]
    let src_bgra: Vec<u8> = test_pixels_rgba()
        .iter()
        .flat_map(|[r, g, b, a]| [*b, *g, *r, *a])
        .collect();
    let dst_bgra: Vec<u8> = distorted_pixels_rgba()
        .iter()
        .flat_map(|[r, g, b, a]| [*b, *g, *r, *a])
        .collect();

    let native_src = StridedBytes::with_alpha_mode(
        &src_bgra,
        16,
        16,
        16 * 4,
        zensim::PixelFormat::Srgb8Bgra,
        AlphaMode::Straight,
    );
    let native_dst = StridedBytes::with_alpha_mode(
        &dst_bgra,
        16,
        16,
        16 * 4,
        zensim::PixelFormat::Srgb8Bgra,
        AlphaMode::Straight,
    );
    let native = z.compute(&native_src, &native_dst).unwrap();

    let src_slice = PixelView::new(&src_bgra, 16, 16, 16 * 4, &specs::BGRA8_SRGB).unwrap();
    let dst_slice = PixelView::new(&dst_bgra, 16, 16, 16 * 4, &specs::BGRA8_SRGB).unwrap();
    let zp_src = ZenpixelsSource::try_from_slice(&src_slice).unwrap();
    let zp_dst = ZenpixelsSource::try_from_slice(&dst_slice).unwrap();
    let zp_result = z.compute(&zp_src, &zp_dst).unwrap();

    assert!(
        (native.score() - zp_result.score()).abs() < 0.001,
        "BGRA8 score mismatch: native={}, zenpixels={}",
        native.score(),
        zp_result.score()
    );
}

#[test]
fn identical_images_score_100() {
    let z = zensim();
    let px = test_pixels_rgba();
    let bytes: Vec<u8> = px.iter().flat_map(|p| p.iter().copied()).collect();
    let slice = PixelView::new(&bytes, 16, 16, 16 * 4, &specs::RGBA8_SRGB).unwrap();
    let src = ZenpixelsSource::try_from_slice(&slice).unwrap();

    let slice2 = PixelView::new(&bytes, 16, 16, 16 * 4, &specs::RGBA8_SRGB).unwrap();
    let dst = ZenpixelsSource::try_from_slice(&slice2).unwrap();

    let result = z.compute(&src, &dst).unwrap();
    assert_eq!(result.score(), 100.0);
}

#[test]
fn different_images_score_below_100() {
    let z = zensim();
    let src_bytes: Vec<u8> = test_pixels_rgba()
        .iter()
        .flat_map(|p| p.iter().copied())
        .collect();
    let dst_bytes: Vec<u8> = distorted_pixels_rgba()
        .iter()
        .flat_map(|p| p.iter().copied())
        .collect();
    let s = PixelView::new(&src_bytes, 16, 16, 16 * 4, &specs::RGBA8_SRGB).unwrap();
    let d = PixelView::new(&dst_bytes, 16, 16, 16 * 4, &specs::RGBA8_SRGB).unwrap();
    let src = ZenpixelsSource::try_from_slice(&s).unwrap();
    let dst = ZenpixelsSource::try_from_slice(&d).unwrap();
    let result = z.compute(&src, &dst).unwrap();
    assert!(
        result.score() < 100.0,
        "different images should score < 100"
    );
}

#[test]
fn premultiplied_matches_straight() {
    let z = zensim();
    let straight = test_pixels_rgba_with_alpha();

    // Create premultiplied version (floor-quantized, as stores commonly are)
    let premul: Vec<u8> = straight
        .iter()
        .flat_map(|[r, g, b, a]| {
            let af = *a as f32 / 255.0;
            [
                (*r as f32 * af) as u8,
                (*g as f32 * af) as u8,
                (*b as f32 * af) as u8,
                *a,
            ]
        })
        .collect();

    let straight_bytes: Vec<u8> = straight.iter().flat_map(|p| p.iter().copied()).collect();

    // Premultiplied via zenpixels adapter (un-premultiplies internally)
    let premul_desc = ImageSpec::RGBA8_SRGB.with_alpha(ZpAlpha::Premultiplied);
    let p_slice = PixelView::new(&premul, 16, 16, 16 * 4, &premul_desc).unwrap();
    let zp_src = ZenpixelsSource::try_from_slice(&p_slice).unwrap();

    // (1) Pixel-exact adapter contract: recovery must match a reference
    // round-to-nearest un-premultiply byte-for-byte. zensim scores identical
    // buffers at exactly 100, and even a single 1-level pixel difference on
    // 16x16 drops the score to ~95.8 (measured), so == 100.0 is a byte-exact
    // assertion. This is strictly stronger than the old score-threshold form.
    let reference: Vec<u8> = premul
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|px| {
            let a = px[3];
            if a == 0 {
                [0, 0, 0, a]
            } else if a < 255 {
                let inv = 255.0 / a as f32;
                [
                    (px[0] as f32 * inv).round().min(255.0) as u8,
                    (px[1] as f32 * inv).round().min(255.0) as u8,
                    (px[2] as f32 * inv).round().min(255.0) as u8,
                    a,
                ]
            } else {
                [px[0], px[1], px[2], a]
            }
        })
        .collect();
    let r_slice = PixelView::new(&reference, 16, 16, 16 * 4, &specs::RGBA8_SRGB).unwrap();
    let r_src = ZenpixelsSource::try_from_slice(&r_slice).unwrap();
    let exact = z.compute(&zp_src, &r_src).unwrap();
    assert_eq!(
        exact.score(),
        100.0,
        "adapter un-premultiply must match the round-to-nearest reference exactly"
    );

    // (2) Catastrophic canary vs the straight original. The test's premultiply
    // floor-quantizes, so optimal recovery is within 2 LSB on ~36% of channels
    // — Profile A scores that 93.4 (a half-image +1 LSB control scores 94.5),
    // while a SKIPPED un-premultiply (premul scored as straight) scores 12.1.
    // The 90 floor separates those regimes without re-coupling this test to
    // metric-calibration drift; (1) above carries the exactness guarantee.
    let s_slice = PixelView::new(&straight_bytes, 16, 16, 16 * 4, &specs::RGBA8_SRGB).unwrap();
    let zp_dst = ZenpixelsSource::try_from_slice(&s_slice).unwrap();
    let zp_result = z.compute(&zp_src, &zp_dst).unwrap();
    assert!(
        zp_result.score() > 90.0,
        "premul un-premultiply should produce near-identical result, got {}",
        zp_result.score()
    );
}

#[test]
fn hdr_transfer_rejected() {
    let desc = ImageSpec::RGBA8_SRGB.with_transfer(Transfer::Pq);
    let data = vec![0u8; 16 * 16 * 4];
    let slice = PixelView::new(&data, 16, 16, 16 * 4, &desc).unwrap();
    let result = ZenpixelsSource::try_from_slice(&slice);
    assert!(result.is_err());
}

#[test]
fn grayscale_rejected() {
    let data = vec![128u8; 16 * 16];
    let slice = PixelView::new(&data, 16, 16, 16, &specs::GRAY8_SRGB).unwrap();
    let result = ZenpixelsSource::try_from_slice(&slice);
    assert!(result.is_err());
}

#[test]
fn pixel_buffer_end_to_end() {
    let z = zensim();
    let px: Vec<u8> = test_pixels_rgba()
        .iter()
        .flat_map(|p| p.iter().copied())
        .collect();
    let buf = PixelBuf::from_bytes(px.clone(), 16, 16, ImageSpec::RGBA8_SRGB).unwrap();
    let src = ZenpixelsSource::try_from_buffer(&buf).unwrap();

    let buf2 = PixelBuf::from_bytes(px, 16, 16, ImageSpec::RGBA8_SRGB).unwrap();
    let dst = ZenpixelsSource::try_from_buffer(&buf2).unwrap();

    let result = z.compute(&src, &dst).unwrap();
    assert_eq!(result.score(), 100.0);
}
