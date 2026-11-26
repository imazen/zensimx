//! Pixel operations — `image`-crate-free RGBA8 canvas + ops.
//!
//! Provides:
//! - [`Bitmap`] — packed RGBA8 buffer (`Vec<u8>`, row-major, no padding).
//!   Public API of `zensim-regress` returns this type; consumers can
//!   inspect raw bytes via [`Bitmap::as_raw`] / [`Bitmap::into_raw`]
//!   or load/save PNG via [`Bitmap::open`] / [`Bitmap::save`].
//! - [`overlay`] — straight-alpha source-over composite.
//! - [`resize`] / [`resize_gray`] — `zenresize`-backed resampling.
//! - [`crop`], [`flip_horizontal`], [`flip_vertical`], [`rotate90`],
//!   [`rotate180`], [`rotate270`] — hand-rolled, byte-level.
//! - [`fill_rect`], [`draw_rect_border`] — drawing primitives.
//! - [`PngError`] — PNG IO error type, hides `zenpng` internals.
//!
//! **Privacy:** [`overlay`] and the basic ops are `pub(crate)`; only
//! [`Bitmap`] and [`PngError`] are `pub` (the public canvas types
//! exposed via `MontageOptions::render` etc.). Per the migration
//! contract: NO `zen*` crate types appear in the public surface.

use std::fs;
use std::io;
use std::path::Path;

use enough::Unstoppable;
use rgb::Rgba;
use zenpixels::PixelBuf;
use zenpng::{EncodeConfig, PngDecodeConfig};

/// Resampling filter for [`resize`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ResampleFilter {
    /// Nearest-neighbor — pixelated, fastest.
    #[allow(dead_code)]
    Nearest,
    /// Bilinear — fast, soft, no overshoot.
    Triangle,
    /// Mitchell-Netravali (B=1/3, C=1/3) — gentle ~1% ringing,
    /// slightly soft. Good default for content with sharp edges
    /// (glyphs, thin lines) where Lanczos's overshoot is too aggressive.
    Mitchell,
    /// Lanczos-3 — sharpest, with ~5% over/undershoot at sharp edges.
    /// Ideal for natural images; produces visible halo rings on
    /// 1-bit-ish content like glyph strips.
    Lanczos3,
}

impl ResampleFilter {
    fn to_zenresize(self) -> zenresize::Filter {
        self.to_zenresize_filter()
    }

    /// Map to the matching `zenresize::Filter`. Crate-internal so callers
    /// (like `font::build_scaled_strip_per_cell`) can drive zenresize
    /// directly with custom config (e.g., input-side crop) instead of
    /// going through `pixel_ops::resize`.
    pub(crate) fn to_zenresize_filter(self) -> zenresize::Filter {
        match self {
            ResampleFilter::Nearest => zenresize::Filter::Box,
            ResampleFilter::Triangle => zenresize::Filter::Triangle,
            ResampleFilter::Mitchell => zenresize::Filter::Mitchell,
            ResampleFilter::Lanczos3 => zenresize::Filter::Lanczos,
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Bitmap — public RGBA8 canvas type
// ════════════════════════════════════════════════════════════════════════

/// Packed-RGBA8 bitmap. Row-major, no padding.
///
/// This is the canvas type returned by [`crate::diff_image::MontageOptions::render`]
/// and friends. To interop with other crates, use [`Bitmap::as_raw`]
/// to inspect the underlying RGBA bytes, or [`Bitmap::open`] /
/// [`Bitmap::save`] for PNG IO.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bitmap {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl Bitmap {
    /// Create a new bitmap zero-filled (transparent black).
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![0u8; (width as usize) * (height as usize) * 4],
        }
    }

    /// Create a bitmap filled with a single color.
    pub fn from_pixel(width: u32, height: u32, color: [u8; 4]) -> Self {
        let n = (width as usize) * (height as usize);
        let mut pixels = Vec::with_capacity(n * 4);
        for _ in 0..n {
            pixels.extend_from_slice(&color);
        }
        Self {
            width,
            height,
            pixels,
        }
    }

    /// Create a bitmap from raw RGBA8 bytes (packed, no row padding).
    /// Returns `None` if `pixels.len() != width * height * 4`.
    pub fn from_raw(width: u32, height: u32, pixels: Vec<u8>) -> Option<Self> {
        if pixels.len() != (width as usize) * (height as usize) * 4 {
            return None;
        }
        Some(Self {
            width,
            height,
            pixels,
        })
    }

    /// Copy a packed RGBA8 byte slice into a new owned bitmap.
    ///
    /// Convenience for callers who hold `&[u8]` (e.g., decoder output)
    /// and want an owned [`Bitmap`] for use with [`crate::diff_image`]
    /// APIs. For zero-copy borrowing with stride support, use
    /// [`BitmapRef::from_borrowed_rgba8_packed`] /
    /// [`BitmapRef::from_borrowed_rgba8_strided`].
    pub fn from_rgba_slice(rgba: &[u8], width: u32, height: u32) -> Result<Self, BitmapError> {
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(4))
            .ok_or(BitmapError::InvalidDimensions { width, height })?;
        if rgba.len() != expected {
            return Err(BitmapError::BufferTooSmall {
                required: expected,
                actual: rgba.len(),
            });
        }
        Ok(Self {
            width,
            height,
            pixels: rgba.to_vec(),
        })
    }

    /// Create a bitmap from a function `f(x, y) -> [r, g, b, a]`.
    pub fn from_fn<F>(width: u32, height: u32, f: F) -> Self
    where
        F: Fn(u32, u32) -> [u8; 4],
    {
        let mut pixels = Vec::with_capacity((width as usize) * (height as usize) * 4);
        for y in 0..height {
            for x in 0..width {
                pixels.extend_from_slice(&f(x, y));
            }
        }
        Self {
            width,
            height,
            pixels,
        }
    }

    /// Width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// `(width, height)` in pixels.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Borrow the underlying RGBA8 byte buffer.
    pub fn as_raw(&self) -> &[u8] {
        &self.pixels
    }

    /// Mutable borrow of the underlying RGBA8 byte buffer.
    pub fn as_raw_mut(&mut self) -> &mut [u8] {
        &mut self.pixels
    }

    /// Take ownership of the underlying RGBA8 byte buffer.
    pub fn into_raw(self) -> Vec<u8> {
        self.pixels
    }

    /// Read pixel `(x, y)`. Returns transparent black if out of bounds.
    pub fn get_pixel(&self, x: u32, y: u32) -> [u8; 4] {
        if x >= self.width || y >= self.height {
            return [0, 0, 0, 0];
        }
        let off = ((y * self.width + x) * 4) as usize;
        [
            self.pixels[off],
            self.pixels[off + 1],
            self.pixels[off + 2],
            self.pixels[off + 3],
        ]
    }

    /// Write pixel `(x, y)`. No-op if out of bounds.
    pub fn put_pixel(&mut self, x: u32, y: u32, p: [u8; 4]) {
        if x >= self.width || y >= self.height {
            return;
        }
        let off = ((y * self.width + x) * 4) as usize;
        self.pixels[off..off + 4].copy_from_slice(&p);
    }

    /// Decode a PNG file into an RGBA8 [`Bitmap`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self, PngError> {
        let bytes = fs::read(path.as_ref()).map_err(PngError::Io)?;
        Self::from_png_bytes(&bytes)
    }

    /// Decode a PNG byte buffer into an RGBA8 [`Bitmap`].
    pub fn from_png_bytes(bytes: &[u8]) -> Result<Self, PngError> {
        let cfg = PngDecodeConfig::default();
        let out = zenpng::decode(bytes, &cfg, &Unstoppable)
            .map_err(|e| PngError::Decode(format!("{e:?}")))?;
        let (w, h) = (out.info.width, out.info.height);
        let rgba = pixel_buffer_to_rgba8(&out.pixels, w, h)?;
        Self::from_raw(w, h, rgba)
            .ok_or_else(|| PngError::Decode("unexpected dimension mismatch".into()))
    }

    /// Encode this bitmap as PNG and write it to `path`.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), PngError> {
        let bytes = self.to_png_bytes()?;
        fs::write(path.as_ref(), bytes).map_err(PngError::Io)
    }

    /// Encode this bitmap as PNG bytes (no IO).
    pub fn to_png_bytes(&self) -> Result<Vec<u8>, PngError> {
        let pixels: &[Rgba<u8>] = bytemuck::cast_slice(&self.pixels);
        let img = imgref::Img::new(pixels, self.width as usize, self.height as usize);
        zenpng::encode_rgba8(
            img,
            None,
            &EncodeConfig::default(),
            &Unstoppable,
            &Unstoppable,
        )
        .map_err(|e| PngError::Encode(format!("{e:?}")))
    }

    /// Borrow this bitmap as a [`BitmapRef`] (zero-copy, packed stride).
    pub fn as_ref(&self) -> BitmapRef<'_> {
        BitmapRef {
            width: self.width,
            height: self.height,
            stride_bytes: self.width.saturating_mul(4),
            data: &self.pixels,
        }
    }
}

impl<'a> From<&'a Bitmap> for BitmapRef<'a> {
    fn from(b: &'a Bitmap) -> Self {
        b.as_ref()
    }
}

// ════════════════════════════════════════════════════════════════════════
// BitmapRef — borrowed RGBA8 view, stride-aware
// ════════════════════════════════════════════════════════════════════════

/// Borrowed RGBA8 bitmap view with stride support.
///
/// Wraps an external `&[u8]` buffer with arbitrary row stride for
/// **zero-copy interop** with strided pixel sources (e.g.,
/// `zenpixels::PixelSlice`, mmap'd image rows, decoder line buffers).
///
/// # Constructing from `zenpixels::PixelSlice`
///
/// `zenpixels` types are intentionally not part of this crate's public
/// API surface (to keep version coalescing safe for downstream
/// consumers). Callers that depend on both crates can build a
/// [`BitmapRef`] from a `PixelSlice` using only its public accessors:
///
/// ```ignore
/// // The caller depends on both zenpixels and zensim-regress directly.
/// let bm = BitmapRef::from_borrowed_rgba8_strided(
///     slice.as_strided_bytes(),
///     slice.width(),
///     slice.rows(),
///     slice.stride() as u32,
/// )?;
/// ```
///
/// To feed a `BitmapRef` into a montage / diff API that expects an owned
/// [`Bitmap`], call [`BitmapRef::to_owned`] (compacts strided rows into a
/// packed `Vec<u8>`).
#[derive(Copy, Clone, Debug)]
pub struct BitmapRef<'a> {
    width: u32,
    height: u32,
    stride_bytes: u32,
    data: &'a [u8],
}

impl<'a> BitmapRef<'a> {
    /// Borrow a packed RGBA8 buffer (stride = `width * 4`).
    pub fn from_borrowed_rgba8_packed(
        data: &'a [u8],
        width: u32,
        height: u32,
    ) -> Result<Self, BitmapError> {
        let stride = width
            .checked_mul(4)
            .ok_or(BitmapError::InvalidDimensions { width, height })?;
        Self::from_borrowed_rgba8_strided(data, width, height, stride)
    }

    /// Borrow a strided RGBA8 buffer.
    ///
    /// `stride_bytes` is the byte distance between the start of
    /// consecutive rows; it must be at least `width * 4`. The buffer
    /// must contain at least `(height - 1) * stride_bytes + width * 4`
    /// bytes (the last row need not have its trailing padding present).
    pub fn from_borrowed_rgba8_strided(
        data: &'a [u8],
        width: u32,
        height: u32,
        stride_bytes: u32,
    ) -> Result<Self, BitmapError> {
        let row_bytes = width
            .checked_mul(4)
            .ok_or(BitmapError::InvalidDimensions { width, height })?;
        if stride_bytes < row_bytes {
            return Err(BitmapError::StrideTooSmall {
                stride_bytes,
                min_stride: row_bytes,
            });
        }
        if height == 0 || width == 0 {
            return Ok(Self {
                width,
                height,
                stride_bytes,
                data: &data[..0],
            });
        }
        let required = (height as usize - 1)
            .checked_mul(stride_bytes as usize)
            .and_then(|n| n.checked_add(row_bytes as usize))
            .ok_or(BitmapError::InvalidDimensions { width, height })?;
        if data.len() < required {
            return Err(BitmapError::BufferTooSmall {
                required,
                actual: data.len(),
            });
        }
        Ok(Self {
            width,
            height,
            stride_bytes,
            data,
        })
    }

    /// Width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }
    /// Height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }
    /// `(width, height)` in pixels.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
    /// Stride between consecutive rows, in bytes.
    pub fn stride_bytes(&self) -> u32 {
        self.stride_bytes
    }
    /// `true` if `stride_bytes == width * 4` (no row padding).
    pub fn is_packed(&self) -> bool {
        self.stride_bytes == self.width.saturating_mul(4)
    }
    /// Borrow row `y` as `width * 4` bytes (no trailing padding).
    /// Returns an empty slice if `y >= height`.
    pub fn row(&self, y: u32) -> &[u8] {
        if y >= self.height {
            return &[];
        }
        let off = (y as usize) * (self.stride_bytes as usize);
        let end = off + (self.width as usize) * 4;
        &self.data[off..end]
    }

    /// Compact this strided view into an owned, packed [`Bitmap`].
    ///
    /// For packed inputs (`is_packed() == true`) this still allocates;
    /// use [`Bitmap::from_rgba_slice`] if you'd rather copy from `&[u8]`
    /// directly.
    pub fn to_owned(&self) -> Bitmap {
        let row_bytes = (self.width as usize) * 4;
        let mut pixels = Vec::with_capacity(row_bytes * (self.height as usize));
        for y in 0..self.height {
            pixels.extend_from_slice(self.row(y));
        }
        Bitmap {
            width: self.width,
            height: self.height,
            pixels,
        }
    }
}

/// Errors from constructing a [`Bitmap`] / [`BitmapRef`] from raw bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BitmapError {
    /// `width * height * 4` overflowed `usize`.
    InvalidDimensions {
        /// The width that caused overflow.
        width: u32,
        /// The height that caused overflow.
        height: u32,
    },
    /// Provided buffer was smaller than required.
    BufferTooSmall {
        /// Bytes required to address all `height` rows of `width` pixels at the given stride.
        required: usize,
        /// Bytes actually provided.
        actual: usize,
    },
    /// Provided stride was smaller than `width * 4`.
    StrideTooSmall {
        /// Stride bytes provided.
        stride_bytes: u32,
        /// Minimum stride bytes (`width * 4`).
        min_stride: u32,
    },
}

impl std::fmt::Display for BitmapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BitmapError::InvalidDimensions { width, height } => {
                write!(f, "invalid bitmap dimensions: {width}×{height}")
            }
            BitmapError::BufferTooSmall { required, actual } => {
                write!(
                    f,
                    "buffer too small: required {required} bytes, got {actual}"
                )
            }
            BitmapError::StrideTooSmall {
                stride_bytes,
                min_stride,
            } => write!(
                f,
                "stride too small: {stride_bytes} bytes, must be at least {min_stride}"
            ),
        }
    }
}

impl std::error::Error for BitmapError {}

/// PNG encode/decode error. Hides `zenpng` internals from the public
/// API.
#[derive(Debug)]
pub enum PngError {
    /// Underlying file IO error.
    Io(io::Error),
    /// PNG decode error (malformed, unsupported, etc.).
    Decode(String),
    /// PNG encode error (resource limit, etc.).
    Encode(String),
}

impl std::fmt::Display for PngError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PngError::Io(e) => write!(f, "PNG IO error: {e}"),
            PngError::Decode(s) => write!(f, "PNG decode error: {s}"),
            PngError::Encode(s) => write!(f, "PNG encode error: {s}"),
        }
    }
}

impl std::error::Error for PngError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PngError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for PngError {
    fn from(e: io::Error) -> Self {
        PngError::Io(e)
    }
}

/// Convert zenpng's erased `PixelBuf` into RGBA8 packed bytes.
/// Handles RGBA8, RGB8, Gray8, GrayAlpha8 (the formats PNG can yield
/// for typical assets). 16-bit and float formats are not supported.
fn pixel_buffer_to_rgba8(pixels: &PixelBuf, w: u32, h: u32) -> Result<Vec<u8>, PngError> {
    use zenpixels::SampleType;
    let layout = pixels.spec().layout;
    if layout.sample != SampleType::U8 {
        return Err(PngError::Decode(format!(
            "unsupported channel type {:?}; only U8 supported",
            layout.sample
        )));
    }
    let n = (w as usize) * (h as usize);
    let slice = pixels.as_view();
    let mut out = Vec::with_capacity(n * 4);
    let channels = layout.channels();
    let has_alpha = layout.order.has_alpha_slot();
    for y in 0..h {
        let row = slice.row(y);
        match (channels, has_alpha) {
            // Rgba
            (4, true) => out.extend_from_slice(&row[..(w as usize) * 4]),
            // Rgb
            (3, false) => {
                for px in row.as_chunks::<3>().0.iter().take(w as usize) {
                    out.extend_from_slice(&[px[0], px[1], px[2], 255]);
                }
            }
            // Gray
            (1, false) => {
                for &g in row.iter().take(w as usize) {
                    out.extend_from_slice(&[g, g, g, 255]);
                }
            }
            // GrayAlpha
            (2, true) => {
                for px in row.as_chunks::<2>().0.iter().take(w as usize) {
                    out.extend_from_slice(&[px[0], px[0], px[0], px[1]]);
                }
            }
            other => {
                return Err(PngError::Decode(format!(
                    "unsupported channel layout (channels={}, has_alpha={})",
                    other.0, other.1
                )));
            }
        }
    }
    Ok(out)
}

// ════════════════════════════════════════════════════════════════════════
// GrayBitmap — single-channel u8 (used for the embedded font strip)
// ════════════════════════════════════════════════════════════════════════

/// Packed-Gray8 bitmap. Used for the embedded font glyph strip.
#[derive(Clone, Debug)]
pub(crate) struct GrayBitmap {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pixels: Vec<u8>,
}

impl GrayBitmap {
    /// Decode a PNG byte buffer into a single-channel [`GrayBitmap`].
    /// Used by [`crate::font`] for the embedded glyph strip.
    pub(crate) fn from_png_bytes(bytes: &[u8]) -> Result<Self, PngError> {
        let cfg = PngDecodeConfig::default();
        let out = zenpng::decode(bytes, &cfg, &Unstoppable)
            .map_err(|e| PngError::Decode(format!("{e:?}")))?;
        let (w, h) = (out.info.width, out.info.height);
        let pixels = pixel_buffer_to_gray8(&out.pixels)?;
        Ok(Self {
            width: w,
            height: h,
            pixels,
        })
    }

    #[allow(dead_code)] // used by font test only (font_strip is RGBA now)
    pub(crate) fn width(&self) -> u32 {
        self.width
    }

    #[allow(dead_code)] // used in tests only
    pub(crate) fn height(&self) -> u32 {
        self.height
    }

    #[allow(dead_code)] // used by font test only
    pub(crate) fn get_pixel(&self, x: u32, y: u32) -> u8 {
        if x >= self.width || y >= self.height {
            return 0;
        }
        self.pixels[(y * self.width + x) as usize]
    }
}

fn pixel_buffer_to_gray8(pixels: &PixelBuf) -> Result<Vec<u8>, PngError> {
    use zenpixels::SampleType;
    let layout = pixels.spec().layout;
    if layout.sample != SampleType::U8 {
        return Err(PngError::Decode(format!(
            "expected U8 channels, got {:?}",
            layout.sample
        )));
    }
    let slice = pixels.as_view();
    let w = slice.width() as usize;
    let h = slice.rows() as usize;
    let channels = layout.channels();
    let has_alpha = layout.order.has_alpha_slot();
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h as u32 {
        let row = slice.row(y);
        match (channels, has_alpha) {
            (1, false) => out.extend_from_slice(&row[..w]),
            (2, true) => {
                for px in row.as_chunks::<2>().0.iter().take(w) {
                    out.push(px[0]);
                }
            }
            (3, false) => {
                for px in row.as_chunks::<3>().0.iter().take(w) {
                    out.push(px[1]); // green as luminance proxy
                }
            }
            (4, true) => {
                for px in row.as_chunks::<4>().0.iter().take(w) {
                    out.push(px[1]);
                }
            }
            other => {
                return Err(PngError::Decode(format!(
                    "unsupported gray decode channel layout ({}, alpha={})",
                    other.0, other.1
                )));
            }
        }
    }
    Ok(out)
}

// ════════════════════════════════════════════════════════════════════════
// Resize — backed by zenresize
// ════════════════════════════════════════════════════════════════════════

/// Resample `src` to `(w, h)` using `filter`. Returns a fresh buffer.
pub(crate) fn resize(src: &Bitmap, w: u32, h: u32, filter: ResampleFilter) -> Bitmap {
    if src.width == 0 || src.height == 0 || w == 0 || h == 0 {
        return Bitmap::new(w, h);
    }
    if (src.width, src.height) == (w, h) {
        return src.clone();
    }
    let config = zenresize::ResizeConfig::builder(src.width, src.height, w, h)
        .filter(filter.to_zenresize())
        .input(zenresize::PixelDescriptor::RGBA8_SRGB)
        .build();
    let pixels = zenresize::Resizer::new(&config).resize(&src.pixels);
    Bitmap {
        width: w,
        height: h,
        pixels,
    }
}

/// Byte-linear resample — treat u8 RGBA as already-linear values and
/// skip the gamma round-trip. Use only for byte-space diagnostic
/// pipelines where the downstream consumer (e.g. `abs(e - a)` on u8
/// pixels) is itself byte-linear; otherwise prefer [`resize`].
pub(crate) fn resize_byte_linear(src: &Bitmap, w: u32, h: u32, filter: ResampleFilter) -> Bitmap {
    if src.width == 0 || src.height == 0 || w == 0 || h == 0 {
        return Bitmap::new(w, h);
    }
    if (src.width, src.height) == (w, h) {
        return src.clone();
    }
    let desc =
        zenresize::PixelDescriptor::RGBA8_SRGB.with_transfer(zenresize::TransferFunction::Linear);
    let config = zenresize::ResizeConfig::builder(src.width, src.height, w, h)
        .filter(filter.to_zenresize())
        .input(desc)
        .build();
    let pixels = zenresize::Resizer::new(&config).resize(&src.pixels);
    Bitmap {
        width: w,
        height: h,
        pixels,
    }
}

/// Resample a [`GrayBitmap`]. Currently unused — the font module
/// switched to an RGBA8 strip so the resize uses the correct
/// linear-alpha pipeline. Kept for future callers that genuinely need
/// gamma-aware single-channel resize (no double-encoding trap as long
/// as the caller passes the right `TransferFunction`).
#[allow(dead_code)]
pub(crate) fn resize_gray(src: &GrayBitmap, w: u32, h: u32, filter: ResampleFilter) -> GrayBitmap {
    if src.width == 0 || src.height == 0 || w == 0 || h == 0 {
        return GrayBitmap {
            width: w,
            height: h,
            pixels: vec![0u8; (w * h) as usize],
        };
    }
    if (src.width, src.height) == (w, h) {
        return src.clone();
    }
    let config = zenresize::ResizeConfig::builder(src.width, src.height, w, h)
        .filter(filter.to_zenresize())
        .input(zenresize::PixelDescriptor::GRAY8_SRGB)
        .build();
    let pixels = zenresize::Resizer::new(&config).resize(&src.pixels);
    GrayBitmap {
        width: w,
        height: h,
        pixels,
    }
}

// ════════════════════════════════════════════════════════════════════════
// Overlay (linear-light premultiplied source-over) + crop + flips + rotates
// ════════════════════════════════════════════════════════════════════════

/// Convert one sRGB straight-alpha u8 RGBA pixel to premultiplied linear f32.
#[inline]
fn srgb_u8_to_linear_premul(rgba: &[u8]) -> [f32; 4] {
    debug_assert_eq!(rgba.len(), 4);
    let a = rgba[3] as f32 * (1.0 / 255.0);
    let r = linear_srgb::default::srgb_u8_to_linear(rgba[0]) * a;
    let g = linear_srgb::default::srgb_u8_to_linear(rgba[1]) * a;
    let b = linear_srgb::default::srgb_u8_to_linear(rgba[2]) * a;
    [r, g, b, a]
}

/// Convert one premultiplied linear f32 RGBA pixel back to sRGB straight-alpha u8.
#[inline]
fn linear_premul_to_srgb_u8(rgba: [f32; 4]) -> [u8; 4] {
    let a = rgba[3].clamp(0.0, 1.0);
    if a < 1.0 / 512.0 {
        // Vanishing alpha — write a fully-transparent pixel rather than divide by ~0.
        return [0, 0, 0, 0];
    }
    let inv_a = 1.0 / a;
    let r_lin = (rgba[0] * inv_a).clamp(0.0, 1.0);
    let g_lin = (rgba[1] * inv_a).clamp(0.0, 1.0);
    let b_lin = (rgba[2] * inv_a).clamp(0.0, 1.0);
    [
        linear_srgb::default::linear_to_srgb_u8(r_lin),
        linear_srgb::default::linear_to_srgb_u8(g_lin),
        linear_srgb::default::linear_to_srgb_u8(b_lin),
        (a * 255.0 + 0.5) as u8,
    ]
}

/// Source-over composite of `src` onto `canvas` at signed `(dx, dy)`.
///
/// Both inputs are sRGB-encoded straight-alpha u8 RGBA. The composite is
/// performed in **premultiplied linear-light f32** via `zenblend::blend_row`
/// — this avoids the under-darkening artifact you get blending sRGB-encoded
/// straight-alpha values directly. Pixels with source alpha 0 or 255 take
/// fast paths that skip the linear conversion (pure transparent / pure
/// opaque cases are bit-identical to a direct copy).
///
/// Out-of-bounds pixels are clipped.
pub(crate) fn overlay(canvas: &mut Bitmap, src: &Bitmap, dx: i64, dy: i64) {
    let cw = canvas.width as i64;
    let ch = canvas.height as i64;
    let sw = src.width as i64;
    let sh = src.height as i64;
    if dx >= cw || dy >= ch || dx + sw <= 0 || dy + sh <= 0 {
        return;
    }

    // Clipped destination + matching source offset.
    let x_start = dx.max(0);
    let y_start = dy.max(0);
    let x_end = (dx + sw).min(cw);
    let y_end = (dy + sh).min(ch);
    let sx_start = (x_start - dx) as u32;
    let sy_start = (y_start - dy) as u32;
    let cw_u = canvas.width as usize;
    let sw_u = src.width as usize;
    let row_w = (x_end - x_start) as usize;
    if row_w == 0 {
        return;
    }

    // Per-row scratch buffers. Reused across all destination rows. Holds
    // partial-alpha pixels; the fully-transparent / fully-opaque fast paths
    // never touch these.
    let mut src_lin = vec![0.0f32; row_w * 4];
    let mut dst_lin = vec![0.0f32; row_w * 4];
    // For each x in [0, row_w) we record whether the pixel needed the
    // linear path. Used to write only the partial-alpha pixels back from
    // the linear scratch (the fast paths already wrote their result
    // directly into canvas).
    let mut needs_linear = vec![false; row_w];

    for y_dst in y_start..y_end {
        let y_src = sy_start + (y_dst - y_start) as u32;
        let dst_row_off = (y_dst as usize) * cw_u * 4;
        let src_row_off = (y_src as usize) * sw_u * 4;

        let mut any_partial = false;
        for (i, x_dst) in (x_start..x_end).enumerate() {
            let x_src = sx_start + (x_dst - x_start) as u32;
            let d_off = dst_row_off + (x_dst as usize) * 4;
            let s_off = src_row_off + (x_src as usize) * 4;
            let sa = src.pixels[s_off + 3];
            needs_linear[i] = false;
            if sa == 0 {
                continue;
            }
            if sa == 255 {
                canvas.pixels[d_off..d_off + 4].copy_from_slice(&src.pixels[s_off..s_off + 4]);
                continue;
            }
            // Partial alpha — convert into the linear scratch.
            let s_lin = srgb_u8_to_linear_premul(&src.pixels[s_off..s_off + 4]);
            let d_lin = srgb_u8_to_linear_premul(&canvas.pixels[d_off..d_off + 4]);
            src_lin[i * 4..i * 4 + 4].copy_from_slice(&s_lin);
            dst_lin[i * 4..i * 4 + 4].copy_from_slice(&d_lin);
            needs_linear[i] = true;
            any_partial = true;
        }

        if !any_partial {
            continue;
        }

        // Blend partial-alpha src OVER dst — zenblend mutates the first
        // operand in place, so after this `src_lin` holds the composite.
        zenblend::blend_row(&mut src_lin, &dst_lin, zenblend::BlendMode::SrcOver);

        // Convert blended pixels back and write to canvas.
        for i in 0..row_w {
            if !needs_linear[i] {
                continue;
            }
            let x_dst = x_start + i as i64;
            let d_off = dst_row_off + (x_dst as usize) * 4;
            let blended = [
                src_lin[i * 4],
                src_lin[i * 4 + 1],
                src_lin[i * 4 + 2],
                src_lin[i * 4 + 3],
            ];
            canvas.pixels[d_off..d_off + 4].copy_from_slice(&linear_premul_to_srgb_u8(blended));
        }
    }
}

/// Crop `src` to `(x, y, w, h)`. Returns a fresh buffer; does not
/// modify `src`. Out-of-bounds requests yield a partial / empty crop.
pub(crate) fn crop(src: &Bitmap, x: u32, y: u32, w: u32, h: u32) -> Bitmap {
    let x_end = x.saturating_add(w).min(src.width);
    let y_end = y.saturating_add(h).min(src.height);
    let cw = x_end.saturating_sub(x);
    let ch = y_end.saturating_sub(y);
    let mut out = Bitmap::new(w, h);
    let row_stride_src = (src.width as usize) * 4;
    let row_stride_dst = (w as usize) * 4;
    let copy_bytes = (cw as usize) * 4;
    for row in 0..ch {
        let src_off = ((y + row) as usize) * row_stride_src + (x as usize) * 4;
        let dst_off = (row as usize) * row_stride_dst;
        out.pixels[dst_off..dst_off + copy_bytes]
            .copy_from_slice(&src.pixels[src_off..src_off + copy_bytes]);
    }
    out
}

/// Mirror `src` horizontally (left-right swap).
pub(crate) fn flip_horizontal(src: &Bitmap) -> Bitmap {
    let mut out = Bitmap::new(src.width, src.height);
    let stride = (src.width as usize) * 4;
    for y in 0..src.height as usize {
        let row_off = y * stride;
        for x in 0..src.width as usize {
            let s = row_off + x * 4;
            let d = row_off + (src.width as usize - 1 - x) * 4;
            out.pixels[d..d + 4].copy_from_slice(&src.pixels[s..s + 4]);
        }
    }
    out
}

/// Mirror `src` vertically (top-bottom swap).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn flip_vertical(src: &Bitmap) -> Bitmap {
    let mut out = Bitmap::new(src.width, src.height);
    let stride = (src.width as usize) * 4;
    for y in 0..src.height as usize {
        let s = y * stride;
        let d = (src.height as usize - 1 - y) * stride;
        out.pixels[d..d + stride].copy_from_slice(&src.pixels[s..s + stride]);
    }
    out
}

/// Rotate `src` by 90° clockwise.
pub(crate) fn rotate90(src: &Bitmap) -> Bitmap {
    let (sw, sh) = (src.width as usize, src.height as usize);
    let mut out = Bitmap::new(src.height, src.width);
    let dst_stride = sh * 4;
    let src_stride = sw * 4;
    for y in 0..sh {
        for x in 0..sw {
            let s = y * src_stride + x * 4;
            // (x, y) → (sh-1-y, x) in the rotated frame.
            let d = x * dst_stride + (sh - 1 - y) * 4;
            out.pixels[d..d + 4].copy_from_slice(&src.pixels[s..s + 4]);
        }
    }
    out
}

/// Rotate `src` by 180°.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn rotate180(src: &Bitmap) -> Bitmap {
    let (sw, sh) = (src.width as usize, src.height as usize);
    let mut out = Bitmap::new(src.width, src.height);
    let stride = sw * 4;
    for y in 0..sh {
        for x in 0..sw {
            let s = y * stride + x * 4;
            let d = (sh - 1 - y) * stride + (sw - 1 - x) * 4;
            out.pixels[d..d + 4].copy_from_slice(&src.pixels[s..s + 4]);
        }
    }
    out
}

/// Rotate `src` by 270° clockwise (≡ 90° counter-clockwise).
pub(crate) fn rotate270(src: &Bitmap) -> Bitmap {
    let (sw, sh) = (src.width as usize, src.height as usize);
    let mut out = Bitmap::new(src.height, src.width);
    let dst_stride = sh * 4;
    let src_stride = sw * 4;
    for y in 0..sh {
        for x in 0..sw {
            let s = y * src_stride + x * 4;
            // (x, y) → (y, sw-1-x).
            let d = (sw - 1 - x) * dst_stride + y * 4;
            out.pixels[d..d + 4].copy_from_slice(&src.pixels[s..s + 4]);
        }
    }
    out
}

// ════════════════════════════════════════════════════════════════════════
// Drawing primitives
// ════════════════════════════════════════════════════════════════════════

/// Fill the rectangle `(x, y, w, h)` of `img` with `color`. Clips.
///
/// For fully-opaque colors (`color[3] == 255`) this stamps bytes
/// directly. For partial-alpha colors the fill is composited in
/// linear-light premultiplied f32 (`color` SrcOver canvas) so that
/// translucent fills don't overwrite the destination's existing alpha
/// or under-darken edges.
pub(crate) fn fill_rect(img: &mut Bitmap, x: u32, y: u32, w: u32, h: u32, color: [u8; 4]) {
    let img_w = img.width;
    let img_h = img.height;
    let x_end = x.saturating_add(w).min(img_w);
    let y_end = y.saturating_add(h).min(img_h);
    if x >= x_end || y >= y_end {
        return;
    }
    let stride = (img_w as usize) * 4;
    let row_w = (x_end - x) as usize;

    // Fast path: fully opaque or fully transparent — byte stamp / no-op.
    if color[3] == 255 {
        let mut row = Vec::with_capacity(row_w * 4);
        for _ in 0..row_w {
            row.extend_from_slice(&color);
        }
        for yy in y..y_end {
            let off = (yy as usize) * stride + (x as usize) * 4;
            img.pixels[off..off + row_w * 4].copy_from_slice(&row);
        }
        return;
    }
    if color[3] == 0 {
        return;
    }

    // Partial-alpha: composite via zenblend in linear-premul f32.
    let solid_lin = srgb_u8_to_linear_premul(&color);
    // Pre-fill an fg scratch with the solid color repeated.
    let mut fg = vec![0.0f32; row_w * 4];
    for px in fg.as_chunks_mut::<4>().0.iter_mut() {
        px.copy_from_slice(&solid_lin);
    }
    let mut bg = vec![0.0f32; row_w * 4];
    let mut blended = vec![0.0f32; row_w * 4];

    for yy in y..y_end {
        let off = (yy as usize) * stride + (x as usize) * 4;
        // Convert canvas row → linear premul into bg.
        for (i, dst_px) in bg.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let p_off = off + i * 4;
            dst_px.copy_from_slice(&srgb_u8_to_linear_premul(&img.pixels[p_off..p_off + 4]));
        }
        // Re-seed blended from fg (zenblend mutates the first operand).
        blended.copy_from_slice(&fg);
        zenblend::blend_row(&mut blended, &bg, zenblend::BlendMode::SrcOver);
        // Convert blended → sRGB-u8 back into canvas.
        for (i, src_px) in blended.as_chunks::<4>().0.iter().enumerate() {
            let p_off = off + i * 4;
            let pix = [src_px[0], src_px[1], src_px[2], src_px[3]];
            img.pixels[p_off..p_off + 4].copy_from_slice(&linear_premul_to_srgb_u8(pix));
        }
    }
}

/// Draw a 1-px outline at the rectangle `(x, y, w, h)`.
///
/// For fully-opaque colors this stamps bytes directly. For partial-
/// alpha colors each border pixel is composited in linear-light
/// premultiplied f32 — same semantics as [`fill_rect`].
pub(crate) fn draw_rect_border(img: &mut Bitmap, x: u32, y: u32, w: u32, h: u32, color: [u8; 4]) {
    if w == 0 || h == 0 {
        return;
    }
    let img_w = img.width;
    let img_h = img.height;
    let x_end = x.saturating_add(w).min(img_w);
    let y_end = y.saturating_add(h).min(img_h);
    let bot = y_end.saturating_sub(1);
    let right = x_end.saturating_sub(1);

    // For partial-alpha borders we'd duplicate the fill_rect blend logic
    // for every single pixel. Instead, route each side through fill_rect
    // (1-pixel-thick rect) — fully opaque hits the fast byte path; partial
    // alpha gets the correct linear-premul composite. Layout splits so
    // each border pixel is touched exactly once (no corner double-blend):
    //   top:    full row at y
    //   bottom: full row at bot (if bot != y)
    //   left:   column slice strictly between top and bottom
    //   right:  column slice strictly between top and bottom (if right != x)
    if x_end > x && y < img_h {
        fill_rect(img, x, y, x_end - x, 1, color);
    }
    if x_end > x && bot < img_h && bot > y {
        fill_rect(img, x, bot, x_end - x, 1, color);
    }
    // Interior column range (rows strictly between top and bottom).
    let interior_top = y.saturating_add(1);
    let interior_bot = bot; // exclusive
    if interior_bot > interior_top {
        let interior_h = interior_bot - interior_top;
        if x < img_w {
            fill_rect(img, x, interior_top, 1, interior_h, color);
        }
        if right < img_w && right > x {
            fill_rect(img, right, interior_top, 1, interior_h, color);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, c: [u8; 4]) -> Bitmap {
        Bitmap::from_pixel(w, h, c)
    }

    #[test]
    fn bitmap_new_is_zero_filled() {
        let b = Bitmap::new(2, 2);
        assert_eq!(b.dimensions(), (2, 2));
        assert_eq!(b.as_raw(), &[0u8; 16]);
    }

    #[test]
    fn bitmap_from_pixel_is_solid() {
        let b = Bitmap::from_pixel(2, 2, [10, 20, 30, 40]);
        assert_eq!(b.get_pixel(0, 0), [10, 20, 30, 40]);
        assert_eq!(b.get_pixel(1, 1), [10, 20, 30, 40]);
    }

    #[test]
    fn bitmap_from_raw_validates_size() {
        assert!(Bitmap::from_raw(2, 2, vec![0u8; 16]).is_some());
        assert!(Bitmap::from_raw(2, 2, vec![0u8; 12]).is_none());
    }

    #[test]
    fn bitmap_from_rgba_slice_copies() {
        let bytes = [1, 2, 3, 255, 4, 5, 6, 255];
        let b = Bitmap::from_rgba_slice(&bytes, 2, 1).unwrap();
        assert_eq!(b.get_pixel(0, 0), [1, 2, 3, 255]);
        assert_eq!(b.get_pixel(1, 0), [4, 5, 6, 255]);
    }

    #[test]
    fn bitmap_from_rgba_slice_rejects_undersize() {
        assert_eq!(
            Bitmap::from_rgba_slice(&[0u8; 4], 2, 1).unwrap_err(),
            BitmapError::BufferTooSmall {
                required: 8,
                actual: 4
            }
        );
    }

    #[test]
    fn bitmapref_packed_round_trip() {
        let bytes: Vec<u8> = (0..2 * 3 * 4).map(|i| i as u8).collect();
        let r = BitmapRef::from_borrowed_rgba8_packed(&bytes, 2, 3).unwrap();
        assert_eq!(r.dimensions(), (2, 3));
        assert!(r.is_packed());
        let owned = r.to_owned();
        assert_eq!(owned.as_raw(), bytes);
    }

    #[test]
    fn bitmapref_strided_compacts_row_padding() {
        // 2x2 RGBA with 12-byte stride (2*4 = 8 bytes of pixels + 4 bytes pad).
        let mut bytes = vec![0u8; 24];
        bytes[0..8].copy_from_slice(&[1, 2, 3, 255, 4, 5, 6, 255]); // row 0
        bytes[12..20].copy_from_slice(&[7, 8, 9, 255, 10, 11, 12, 255]); // row 1
        let r = BitmapRef::from_borrowed_rgba8_strided(&bytes, 2, 2, 12).unwrap();
        assert!(!r.is_packed());
        assert_eq!(r.row(0), &[1, 2, 3, 255, 4, 5, 6, 255]);
        assert_eq!(r.row(1), &[7, 8, 9, 255, 10, 11, 12, 255]);
        let owned = r.to_owned();
        assert_eq!(
            owned.as_raw(),
            &[1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9, 255, 10, 11, 12, 255]
        );
    }

    #[test]
    fn bitmapref_rejects_undersize_stride() {
        let bytes = [0u8; 32];
        assert!(matches!(
            BitmapRef::from_borrowed_rgba8_strided(&bytes, 2, 2, 4),
            Err(BitmapError::StrideTooSmall { .. })
        ));
    }

    #[test]
    fn bitmapref_rejects_short_buffer() {
        let bytes = [0u8; 8];
        // Need at least (2-1)*12 + 2*4 = 20 bytes for a 2x2 strided bitmap.
        assert!(matches!(
            BitmapRef::from_borrowed_rgba8_strided(&bytes, 2, 2, 12),
            Err(BitmapError::BufferTooSmall { .. })
        ));
    }

    #[test]
    fn bitmap_as_ref_is_packed_view() {
        let b = Bitmap::from_pixel(3, 2, [9, 9, 9, 255]);
        let r: BitmapRef<'_> = (&b).into();
        assert_eq!(r.dimensions(), (3, 2));
        assert!(r.is_packed());
        assert_eq!(r.stride_bytes(), 12);
        assert_eq!(r.row(0), b.as_raw()[..12].to_vec().as_slice());
    }

    #[test]
    fn bitmap_from_fn() {
        let b = Bitmap::from_fn(2, 2, |x, y| [x as u8, y as u8, 0, 255]);
        assert_eq!(b.get_pixel(1, 0), [1, 0, 0, 255]);
        assert_eq!(b.get_pixel(0, 1), [0, 1, 0, 255]);
    }

    #[test]
    fn put_pixel_round_trips() {
        let mut b = Bitmap::new(4, 4);
        b.put_pixel(2, 1, [255, 128, 64, 200]);
        assert_eq!(b.get_pixel(2, 1), [255, 128, 64, 200]);
    }

    #[test]
    fn overlay_blits_opaque() {
        let mut canvas = solid(10, 10, [0, 0, 0, 255]);
        let src = solid(4, 4, [255, 0, 0, 255]);
        overlay(&mut canvas, &src, 2, 3);
        assert_eq!(canvas.get_pixel(2, 3), [255, 0, 0, 255]);
        assert_eq!(canvas.get_pixel(5, 6), [255, 0, 0, 255]);
        assert_eq!(canvas.get_pixel(0, 0), [0, 0, 0, 255]);
    }

    #[test]
    fn overlay_skips_transparent_source() {
        let mut canvas = solid(4, 4, [0, 0, 0, 255]);
        let src = solid(2, 2, [255, 0, 0, 0]);
        overlay(&mut canvas, &src, 1, 1);
        assert_eq!(canvas.get_pixel(1, 1), [0, 0, 0, 255]);
    }

    #[test]
    fn overlay_clips_negative_offsets() {
        let mut canvas = solid(4, 4, [0, 0, 0, 255]);
        let src = solid(4, 4, [255, 0, 0, 255]);
        overlay(&mut canvas, &src, -2, -2);
        assert_eq!(canvas.get_pixel(0, 0), [255, 0, 0, 255]);
        assert_eq!(canvas.get_pixel(1, 1), [255, 0, 0, 255]);
        assert_eq!(canvas.get_pixel(3, 3), [0, 0, 0, 255]);
    }

    #[test]
    fn overlay_partial_alpha_uses_linear_compositing() {
        // 50%-alpha white over solid black. The naïve sRGB straight-alpha
        // formula gives sRGB ~128 (linear ~0.216 — way too dark). The
        // correct linear-light average of 0 and 1 at 50% is 0.5 → sRGB 188.
        let mut canvas = solid(2, 2, [0, 0, 0, 255]);
        let src = solid(2, 2, [255, 255, 255, 128]);
        overlay(&mut canvas, &src, 0, 0);
        let p = canvas.get_pixel(0, 0);
        assert_eq!(p[3], 255, "alpha out should saturate (over opaque dst)");
        assert!(
            p[0] >= 180 && p[0] <= 195,
            "expected linear-correct gray near 188, got {}",
            p[0]
        );
        assert_eq!(p[0], p[1]);
        assert_eq!(p[1], p[2]);
    }

    #[test]
    fn overlay_partial_alpha_dark_over_light_doesnt_under_darken() {
        // Inverse: 50%-alpha black over solid white. Naïve sRGB average
        // yields ~128 sRGB. Linear-correct is sRGB ~188.
        let mut canvas = solid(2, 2, [255, 255, 255, 255]);
        let src = solid(2, 2, [0, 0, 0, 128]);
        overlay(&mut canvas, &src, 0, 0);
        let p = canvas.get_pixel(0, 0);
        assert!(
            p[0] >= 180 && p[0] <= 195,
            "expected linear-correct gray near 188, got {}",
            p[0]
        );
    }

    #[test]
    fn fill_rect_partial_alpha_blends_in_linear() {
        // 50%-alpha white over solid black should land near sRGB 188,
        // matching overlay's linear-correct semantics. The old impl
        // overwrote bytes and would give sRGB 255 (alpha 128).
        let mut img = solid(4, 4, [0, 0, 0, 255]);
        fill_rect(&mut img, 0, 0, 4, 4, [255, 255, 255, 128]);
        let p = img.get_pixel(1, 1);
        assert_eq!(p[3], 255, "dst alpha must stay 255 (over opaque)");
        assert!(
            p[0] >= 180 && p[0] <= 195,
            "expected linear-correct gray near 188, got {}",
            p[0]
        );
    }

    #[test]
    fn fill_rect_opaque_fast_path_unchanged() {
        // Fully-opaque fills must still stamp bytes exactly.
        let mut img = solid(4, 4, [10, 20, 30, 255]);
        fill_rect(&mut img, 1, 1, 2, 2, [50, 60, 70, 255]);
        assert_eq!(img.get_pixel(1, 1), [50, 60, 70, 255]);
        assert_eq!(img.get_pixel(0, 0), [10, 20, 30, 255]);
    }

    #[test]
    fn draw_rect_border_partial_alpha_blends() {
        // Partial-alpha border on opaque canvas — every border pixel
        // should land near sRGB 188 (linear midpoint), not 255.
        let mut img = solid(6, 6, [0, 0, 0, 255]);
        draw_rect_border(&mut img, 1, 1, 4, 4, [255, 255, 255, 128]);
        let p = img.get_pixel(1, 1);
        assert!(
            p[0] >= 180 && p[0] <= 195,
            "border partial-alpha expected ~188, got {}",
            p[0]
        );
        // Interior must remain untouched by border draw.
        assert_eq!(img.get_pixel(2, 2), [0, 0, 0, 255]);
    }

    #[test]
    fn resize_same_dim_returns_clone() {
        let b = solid(8, 8, [128; 4]);
        let r = resize(&b, 8, 8, ResampleFilter::Triangle);
        assert_eq!(r.as_raw(), b.as_raw());
    }

    #[test]
    fn resize_changes_dimensions() {
        let b = solid(4, 4, [128, 128, 128, 255]);
        let r = resize(&b, 8, 8, ResampleFilter::Lanczos3);
        assert_eq!(r.dimensions(), (8, 8));
    }

    #[test]
    fn crop_extracts_region() {
        let mut src = solid(8, 8, [0, 0, 0, 255]);
        src.put_pixel(3, 3, [255, 255, 255, 255]);
        let cropped = crop(&src, 2, 2, 4, 4);
        assert_eq!(cropped.dimensions(), (4, 4));
        assert_eq!(cropped.get_pixel(1, 1), [255, 255, 255, 255]);
    }

    #[test]
    fn flip_horizontal_swaps_columns() {
        let mut src = solid(4, 1, [0, 0, 0, 255]);
        src.put_pixel(0, 0, [255, 0, 0, 255]);
        let flipped = flip_horizontal(&src);
        assert_eq!(flipped.get_pixel(3, 0), [255, 0, 0, 255]);
    }

    #[test]
    fn flip_vertical_swaps_rows() {
        let mut src = solid(1, 4, [0, 0, 0, 255]);
        src.put_pixel(0, 0, [255, 0, 0, 255]);
        let flipped = flip_vertical(&src);
        assert_eq!(flipped.get_pixel(0, 3), [255, 0, 0, 255]);
    }

    #[test]
    fn rotate90_then_270_is_identity() {
        let mut src = Bitmap::from_fn(3, 5, |x, y| [x as u8 * 60, y as u8 * 50, 128, 255]);
        src.put_pixel(0, 0, [255, 128, 64, 255]);
        let r = rotate90(&rotate270(&src));
        assert_eq!(r.dimensions(), src.dimensions());
        assert_eq!(r.get_pixel(0, 0), src.get_pixel(0, 0));
        assert_eq!(r.get_pixel(2, 4), src.get_pixel(2, 4));
    }

    #[test]
    fn rotate180_inverts() {
        let src = Bitmap::from_fn(4, 3, |x, y| [x as u8 * 30, y as u8 * 50, 0, 255]);
        let r = rotate180(&src);
        assert_eq!(r.get_pixel(0, 0), src.get_pixel(3, 2));
        assert_eq!(r.get_pixel(3, 2), src.get_pixel(0, 0));
    }

    #[test]
    fn fill_rect_clips_to_bounds() {
        let mut img = solid(10, 10, [0, 0, 0, 255]);
        fill_rect(&mut img, 2, 2, 4, 4, [255, 0, 0, 255]);
        assert_eq!(img.get_pixel(3, 3), [255, 0, 0, 255]);
        assert_eq!(img.get_pixel(0, 0), [0, 0, 0, 255]);
        assert_eq!(img.get_pixel(6, 6), [0, 0, 0, 255]);
    }

    #[test]
    fn draw_rect_border_outlines_only() {
        let mut img = solid(10, 10, [0, 0, 0, 255]);
        draw_rect_border(&mut img, 1, 1, 8, 8, [255, 0, 0, 255]);
        assert_eq!(img.get_pixel(1, 1), [255, 0, 0, 255]);
        assert_eq!(img.get_pixel(8, 8), [255, 0, 0, 255]);
        assert_eq!(img.get_pixel(5, 5), [0, 0, 0, 255]);
    }

    #[test]
    fn png_roundtrip() {
        let src = Bitmap::from_fn(8, 8, |x, y| [x as u8 * 32, y as u8 * 32, 128, 255]);
        let bytes = src.to_png_bytes().expect("encode");
        let decoded = Bitmap::from_png_bytes(&bytes).expect("decode");
        assert_eq!(decoded.dimensions(), src.dimensions());
        assert_eq!(decoded.as_raw(), src.as_raw());
    }
}
