//! Adapter for zenpixels pixel types.
//!
//! Provides [`ZenpixelsSource`], a validated wrapper that implements
//! [`ImageSource`] for any zenpixels [`PixelView`] or [`PixelBuf`].
//!
//! # Supported formats
//!
//! | zenpixels format | transfer | result |
//! |------------------|----------|--------|
//! | Rgb8 | sRGB/BT.709 | `Srgb8Rgb`, Opaque |
//! | Rgba8 | sRGB/BT.709 | `Srgb8Rgba`, from alpha mode |
//! | Bgra8 | sRGB/BT.709 | `Srgb8Bgra`, from alpha mode |
//! | Rgbx8 | sRGB/BT.709 | `Srgb8Rgba`, **Opaque** (automatic) |
//! | Bgrx8 | sRGB/BT.709 | `Srgb8Bgra`, **Opaque** (automatic) |
//! | Rgba16 | sRGB/BT.709 | `Srgb16Rgba`, from alpha mode |
//! | RgbaF32 | Linear | `LinearF32Rgba`, from alpha mode |
//!
//! Premultiplied alpha is un-premultiplied automatically (allocates a copy).
//! RGBX/BGRX padding bytes are ignored automatically (no user action needed).
//!
//! # Rejected formats
//!
//! - HDR transfers (PQ, HLG) — the `ImageSource` pipeline is SDR; decode HDR
//!   to absolute-luminance linear RGB and score it via
//!   [`Zensim::compute_pu_linear`](crate::Zensim::compute_pu_linear) instead
//! - Unknown transfer — can't convert losslessly
//! - Grayscale — not supported
//! - Narrow signal range — requires expansion first

use std::borrow::Cow;

use zenpixels::{
    AlphaMode as ZpAlpha, ChannelOrder, ColorSpace, ImageSpec, PixelBuf, PixelView,
    Primaries as ZpPrimaries, SampleLayout, SignalRange, Transfer,
};

use crate::error::UnsupportedFormat;
use crate::source::{AlphaMode, ColorPrimaries, ImageSource, PixelFormat};

/// Validated adapter wrapping zenpixels pixel data for use with zensim.
///
/// Created via [`try_from_slice`](Self::try_from_slice) or
/// [`try_from_buffer`](Self::try_from_buffer). Validates the image spec
/// and maps it to the correct zensim format, alpha mode, and color primaries.
///
/// For premultiplied alpha, the data is un-premultiplied into an internal
/// buffer (one-time allocation). For all other formats, the original data
/// is borrowed with zero copies.
pub struct ZenpixelsSource<'a> {
    data: Cow<'a, [u8]>,
    width: usize,
    height: usize,
    stride: usize,
    pixel_format: PixelFormat,
    alpha_mode: AlphaMode,
    color_primaries: ColorPrimaries,
}

impl<'a> ZenpixelsSource<'a> {
    /// Create from a borrowed [`PixelView`].
    ///
    /// Validates the image spec and rejects unsupported formats.
    /// For premultiplied alpha, allocates and un-premultiplies the data.
    pub fn try_from_slice(slice: &'a PixelView<'a>) -> Result<Self, UnsupportedFormat> {
        let spec = slice.spec();
        let (pixel_format, alpha_mode, color_primaries) = map_descriptor(spec)?;

        let width = slice.width() as usize;
        let height = slice.rows() as usize;
        let stride = slice.stride();
        let raw = slice.as_strided_bytes();

        let data = if spec.color.alpha() == ZpAlpha::Premultiplied {
            Cow::Owned(unpremultiply(raw, width, height, stride, spec))
        } else {
            Cow::Borrowed(raw)
        };

        Ok(Self {
            data,
            width,
            height,
            stride,
            pixel_format,
            alpha_mode,
            color_primaries,
        })
    }

    /// Create from an owned [`PixelBuf`].
    ///
    /// Same validation as [`try_from_slice`](Self::try_from_slice).
    /// The buffer's backing memory is borrowed for the lifetime of the source.
    pub fn try_from_buffer(buf: &'a PixelBuf) -> Result<Self, UnsupportedFormat> {
        let spec = buf.spec();
        let (pixel_format, alpha_mode, color_primaries) = map_descriptor(spec)?;

        let width = buf.width() as usize;
        let height = buf.height() as usize;
        let stride = buf.stride();

        // PixelBuf may have internal offset; get contiguous bytes.
        let raw = match buf.as_view().as_contiguous_bytes() {
            Some(bytes) => bytes,
            None => {
                return Err(UnsupportedFormat("non-contiguous PixelBuf layout"));
            }
        };

        let data = if spec.color.alpha() == ZpAlpha::Premultiplied {
            Cow::Owned(unpremultiply(raw, width, height, stride, spec))
        } else {
            Cow::Borrowed(raw)
        };

        Ok(Self {
            data,
            width,
            height,
            stride,
            pixel_format,
            alpha_mode,
            color_primaries,
        })
    }
}

impl ImageSource for ZenpixelsSource<'_> {
    #[inline]
    fn width(&self) -> usize {
        self.width
    }
    #[inline]
    fn height(&self) -> usize {
        self.height
    }
    #[inline]
    fn pixel_format(&self) -> PixelFormat {
        self.pixel_format
    }
    #[inline]
    fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }
    #[inline]
    fn color_primaries(&self) -> ColorPrimaries {
        self.color_primaries
    }
    #[inline]
    fn row_bytes(&self, y: usize) -> &[u8] {
        let bpp = self.pixel_format.bytes_per_pixel();
        let start = y * self.stride;
        &self.data[start..start + self.width * bpp]
    }
}

/// Map a zenpixels [`ImageSpec`] to zensim types.
fn map_descriptor(
    spec: &ImageSpec,
) -> Result<(PixelFormat, AlphaMode, ColorPrimaries), UnsupportedFormat> {
    if matches!(
        spec.layout.order,
        ChannelOrder::Gray | ChannelOrder::GrayAlpha
    ) || matches!(spec.color, ColorSpace::Gray { .. })
    {
        return Err(UnsupportedFormat("grayscale formats not supported"));
    }

    let ColorSpace::Rgb {
        primaries,
        transfer,
        range,
        alpha,
        ..
    } = spec.color
    else {
        return Err(UnsupportedFormat("unsupported color space — expected RGB"));
    };

    // Reject unsupported transfers
    match transfer {
        Transfer::Srgb | Transfer::Bt709 => {}
        Transfer::Linear => {
            if spec.layout != SampleLayout::RGBAF32 {
                return Err(UnsupportedFormat("linear transfer requires RgbaF32 format"));
            }
        }
        Transfer::Pq | Transfer::Hlg(_) => {
            return Err(UnsupportedFormat(
                "HDR transfers (PQ, HLG) are not supported by the SDR ImageSource pipeline — \
                 decode to absolute-luminance linear RGB and use Zensim::compute_pu_linear",
            ));
        }
        _ => {
            return Err(UnsupportedFormat(
                "unknown transfer function — cannot convert losslessly",
            ));
        }
    }

    // Reject narrow signal range
    if range != SignalRange::Full {
        return Err(UnsupportedFormat(
            "narrow/limited signal range not supported — expand to full range first",
        ));
    }

    // Map pixel format
    let pixel_format = match spec.layout {
        SampleLayout::RGB8 => PixelFormat::Srgb8Rgb,
        SampleLayout::RGBA8 | SampleLayout::RGBX8 => PixelFormat::Srgb8Rgba,
        SampleLayout::BGRA8 | SampleLayout::BGRX8 => PixelFormat::Srgb8Bgra,
        SampleLayout::RGBA16 => PixelFormat::Srgb16Rgba,
        SampleLayout::RGBAF32 => PixelFormat::LinearF32Rgba,
        _ => {
            return Err(UnsupportedFormat("unsupported pixel format"));
        }
    };

    // Map alpha mode
    let alpha_mode = if matches!(spec.layout, SampleLayout::RGBX8 | SampleLayout::BGRX8) {
        AlphaMode::Opaque
    } else {
        match alpha {
            ZpAlpha::None | ZpAlpha::Undefined | ZpAlpha::Opaque => AlphaMode::Opaque,
            ZpAlpha::Straight | ZpAlpha::Premultiplied => AlphaMode::Straight,
            _ => AlphaMode::Straight,
        }
    };

    // Map color primaries
    let color_primaries = match primaries {
        ZpPrimaries::Bt709 => ColorPrimaries::Srgb,
        ZpPrimaries::DisplayP3 => ColorPrimaries::DisplayP3,
        ZpPrimaries::Bt2020 => ColorPrimaries::Bt2020,
        _ => ColorPrimaries::Srgb,
    };

    Ok((pixel_format, alpha_mode, color_primaries))
}

/// Un-premultiply pixel data, returning a new buffer with straight alpha.
///
/// Divides each RGB channel by alpha in the native domain (sRGB for u8/u16,
/// linear for f32). Fully transparent pixels (A=0) get RGB zeroed.
fn unpremultiply(
    data: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    spec: &ImageSpec,
) -> Vec<u8> {
    let mut out = data.to_vec();

    match spec.layout {
        SampleLayout::RGBA8 | SampleLayout::RGBX8 | SampleLayout::BGRA8 | SampleLayout::BGRX8 => {
            for y in 0..height {
                let row_start = y * stride;
                for x in 0..width {
                    let off = row_start + x * 4;
                    let a = out[off + 3];
                    if a == 0 {
                        out[off] = 0;
                        out[off + 1] = 0;
                        out[off + 2] = 0;
                    } else if a < 255 {
                        let inv = 255.0 / a as f32;
                        // Round-to-nearest: premultiplied stores are already
                        // quantized, so truncating here would compound the
                        // loss (up to ~2 LSB at mid alpha). Rounding is the
                        // closest reconstruction of the straight value.
                        out[off] = (out[off] as f32 * inv).round().min(255.0) as u8;
                        out[off + 1] = (out[off + 1] as f32 * inv).round().min(255.0) as u8;
                        out[off + 2] = (out[off + 2] as f32 * inv).round().min(255.0) as u8;
                    }
                }
            }
        }
        SampleLayout::RGBA16 => {
            for y in 0..height {
                let row_start = y * stride;
                for x in 0..width {
                    let off = row_start + x * 8;
                    let a = u16::from_ne_bytes([out[off + 6], out[off + 7]]);
                    if a == 0 {
                        out[off..off + 6].fill(0);
                    } else if a < 65535 {
                        let inv = 65535.0 / a as f32;
                        for c in 0..3 {
                            let co = off + c * 2;
                            let v = u16::from_ne_bytes([out[co], out[co + 1]]);
                            let unpremul = (v as f32 * inv).round().min(65535.0) as u16;
                            out[co..co + 2].copy_from_slice(&unpremul.to_ne_bytes());
                        }
                    }
                }
            }
        }
        SampleLayout::RGBAF32 => {
            for y in 0..height {
                let row_start = y * stride;
                for x in 0..width {
                    let off = row_start + x * 16;
                    let a = f32::from_ne_bytes([
                        out[off + 12],
                        out[off + 13],
                        out[off + 14],
                        out[off + 15],
                    ]);
                    if a <= 0.0 {
                        out[off..off + 12].fill(0);
                    } else if a < 1.0 {
                        let inv = 1.0 / a;
                        for c in 0..3 {
                            let co = off + c * 4;
                            let v = f32::from_ne_bytes([
                                out[co],
                                out[co + 1],
                                out[co + 2],
                                out[co + 3],
                            ]);
                            out[co..co + 4].copy_from_slice(&(v * inv).to_ne_bytes());
                        }
                    }
                }
            }
        }
        _ => {}
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgbx_maps_to_opaque() {
        let desc = ImageSpec::RGBX8_SRGB;
        let (fmt, alpha, _) = map_descriptor(&desc).unwrap();
        assert_eq!(fmt, PixelFormat::Srgb8Rgba);
        assert_eq!(alpha, AlphaMode::Opaque);
    }

    #[test]
    fn bgrx_maps_to_opaque() {
        let desc = ImageSpec::BGRX8_SRGB;
        let (fmt, alpha, _) = map_descriptor(&desc).unwrap();
        assert_eq!(fmt, PixelFormat::Srgb8Bgra);
        assert_eq!(alpha, AlphaMode::Opaque);
    }

    #[test]
    fn rgba_straight_maps_correctly() {
        let desc = ImageSpec::RGBA8_SRGB;
        let (fmt, alpha, _) = map_descriptor(&desc).unwrap();
        assert_eq!(fmt, PixelFormat::Srgb8Rgba);
        assert_eq!(alpha, AlphaMode::Straight);
    }

    #[test]
    fn hdr_rejected() {
        let desc = ImageSpec::RGBA8_SRGB.with_transfer(Transfer::Pq);
        assert!(map_descriptor(&desc).is_err());
    }

    #[test]
    fn unknown_transfer_rejected() {
        let desc = ImageSpec::RGBA8_SRGB.with_transfer(Transfer::Unknown);
        assert!(map_descriptor(&desc).is_err());
    }

    #[test]
    fn grayscale_rejected() {
        let desc = ImageSpec::GRAY8_SRGB;
        assert!(map_descriptor(&desc).is_err());
    }

    #[test]
    fn p3_primaries_mapped() {
        let desc = ImageSpec::RGBA8_SRGB.with_primaries(ZpPrimaries::DisplayP3);
        let (_, _, primaries) = map_descriptor(&desc).unwrap();
        assert_eq!(primaries, ColorPrimaries::DisplayP3);
    }

    #[test]
    fn unpremultiply_u8_roundtrip() {
        // Premultiplied: R=100 means straight R = 100 * 255/200 ≈ 127
        let data = vec![100u8, 50, 25, 200];
        let result = unpremultiply(&data, 1, 1, 4, &ImageSpec::RGBA8_SRGB);
        assert_eq!(result[3], 200); // alpha unchanged
        assert_eq!(result[0], 128); // 100 * 255/200 = 127.5 → 128 (round-to-nearest)
        assert_eq!(result[1], 64); // 50 * 255/200 = 63.75 → 64 (round-to-nearest)
    }

    #[test]
    fn unpremultiply_a0_clears_rgb() {
        let data = vec![255u8, 128, 64, 0];
        let result = unpremultiply(&data, 1, 1, 4, &ImageSpec::RGBA8_SRGB);
        assert_eq!(result, [0, 0, 0, 0]);
    }
}
