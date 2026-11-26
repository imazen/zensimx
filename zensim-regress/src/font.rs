//! Monospace font rendering for annotating diff images.
//!
//! Embeds a Consolas glyph strip (48px, ASCII 32–126) as a PNG, decoded
//! once on first use into an RGBA8 [`Bitmap`] with `(R, G, B) = (255, 255,
//! 255)` (white-tinted) and `A = source PNG value` (linear coverage).
//! Renders text at any size by downscaling via `pixel_ops::resize`
//! (Mitchell-Netravali) — the base is larger than any render target, so
//! we always downsample. Mitchell over Lanczos because Lanczos's
//! negative side-lobes produce visible halo rings outside glyph edges
//! at small char_h.
//!
//! Resampling is correct because the strip is RGBA: zenresize's
//! `RGBA8_SRGB` pipeline treats alpha as linear (the right semantics
//! for a coverage mask), premultiplies, resamples in linear-light, and
//! unpremultiplies on output. The earlier `GRAY8_SRGB` path applied an
//! sRGB curve to the coverage values, which is mathematically wrong
//! since coverage is a linear quantity by definition.
//!
//! Composite is gamma-correct: per glyph pixel, read the strip's alpha
//! as coverage and lerp `fg`/`bg` in linear-sRGB space via the LUT-
//! backed `blend_channel_gamma_correct` helper (private module fn).
//! A naive
//! `fg*a + bg*(1-a)` lerp in sRGB-encoded byte space produces muddy
//! halos at mid-alpha because sRGB→linear is non-linear.
//!
//! Word-wraps on spaces when a max width is specified.

use std::sync::{Arc, Mutex, OnceLock};

use crate::pixel_ops::{Bitmap, GrayBitmap, ResampleFilter};

/// Precomputed sRGB-u8 → linear-f32 lookup table — reused for every
/// glyph composite to avoid per-pixel pow() calls. Built lazily on
/// first use; ~1KB, computed in microseconds.
fn srgb_to_linear_lut() -> &'static [f32; 256] {
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0.0f32; 256];
        for (i, slot) in t.iter_mut().enumerate() {
            *slot = linear_srgb::default::srgb_u8_to_linear(i as u8);
        }
        t
    })
}

/// Gamma-correct blend of a foreground over a background using a glyph
/// alpha as the coverage mask. Returns the blended sRGB-u8 channel.
///
/// `alpha` is the coverage from the glyph rasterizer (treated as
/// linear coverage — anti-aliasing area-of-overlap, not an sRGB
/// quantity). `fg` and `bg` are sRGB-u8.
#[inline]
fn blend_channel_gamma_correct(fg: u8, bg: u8, alpha: u8) -> u8 {
    let lut = srgb_to_linear_lut();
    let fg_lin = lut[fg as usize];
    let bg_lin = lut[bg as usize];
    let a = (alpha as f32) * (1.0 / 255.0);
    let blended = fg_lin * a + bg_lin * (1.0 - a);
    linear_srgb::default::linear_to_srgb_u8(blended)
}

/// Base glyph width in the embedded strip (before scaling).
const BASE_CHAR_W: u32 = 26;
/// Base glyph height in the embedded strip (before scaling).
const BASE_CHAR_H: u32 = 54;
/// Total characters in the strip: ASCII 32-126 (95) + extras (Δ).
const CHAR_COUNT: u32 = 96;
/// First printable ASCII codepoint. (Composers map through
/// `crate::glyph_map`; this remains for the strip-layout tests.)
#[cfg(test)]
const FIRST_CHAR: u32 = 0x20;

/// The embedded font strip PNG (Consolas 48px, grayscale).
static FONT_PNG: &[u8] = include_bytes!("font_strip.png");

/// Decoded font strip as RGBA8 (white-tinted, alpha = coverage).
///
/// The PNG is grayscale where each value represents the glyph's
/// coverage at that pixel — a *linear* quantity (area-of-overlap),
/// NOT sRGB-encoded brightness. We store it as `(255, 255, 255, value)`
/// so that:
/// 1. Resampling via zenresize uses the `RGBA8_SRGB` pipeline that
///    correctly treats alpha as linear (premultiply, resample, unpremultiply).
///    The previous `GRAY8_SRGB` path mistakenly applied an sRGB gamma
///    curve to the coverage values, producing slightly-wrong alpha
///    after downsample.
/// 2. The composite step reads only the alpha channel as the coverage
///    mask. The chosen `fg` color is the actual fg — the strip's RGB
///    just signals "white-tinted glyph" semantically.
///
/// Decoded once on first use; subsequent calls reuse the static.
#[cfg_attr(feature = "sdf-font", allow(dead_code))]
fn font_strip() -> &'static Bitmap {
    static STRIP: OnceLock<Bitmap> = OnceLock::new();
    STRIP.get_or_init(|| {
        let gray =
            GrayBitmap::from_png_bytes(FONT_PNG).expect("embedded font strip PNG is invalid");
        let (w, h) = (gray.width, gray.height);
        let mut rgba = Vec::with_capacity((w as usize) * (h as usize) * 4);
        for &v in gray.pixels.iter() {
            rgba.extend_from_slice(&[255, 255, 255, v]);
        }
        Bitmap::from_raw(w, h, rgba).expect("strip dimensions match buffer")
    })
}

/// Render a string into an RGBA pixel buffer at the base font size.
///
/// Returns `(pixels, width, height)`. Newlines are supported.
pub fn render_text(text: &str, fg: [u8; 4], bg: [u8; 4]) -> (Vec<u8>, u32, u32) {
    render_text_height(text, fg, bg, BASE_CHAR_H)
}

/// Render text word-wrapped to fit within `max_width_px` pixels.
///
/// Lines are broken on spaces. If a single word is wider than `max_width_px`,
/// it is placed on its own line (not split mid-word). Lines stack tightly
/// — for typographic leading, see [`render_text_wrapped_lh`].
///
/// Returns `(pixels, width, height)`.
pub fn render_text_wrapped(
    text: &str,
    fg: [u8; 4],
    bg: [u8; 4],
    target_char_h: u32,
    max_width_px: u32,
) -> (Vec<u8>, u32, u32) {
    render_text_wrapped_lh(text, fg, bg, target_char_h, max_width_px, 1.0)
}

/// Same as [`render_text_wrapped`], with a CSS-`line-height` ratio
/// applied to the inter-line stride.
pub fn render_text_wrapped_lh(
    text: &str,
    fg: [u8; 4],
    bg: [u8; 4],
    target_char_h: u32,
    max_width_px: u32,
    line_height: f32,
) -> (Vec<u8>, u32, u32) {
    if target_char_h == 0 || max_width_px == 0 {
        return (vec![], 0, 0);
    }

    let scaled_char_w = char_width_for_height(target_char_h);
    let max_cols = (max_width_px / scaled_char_w.max(1)) as usize;
    let wrapped = wrap_text(text, max_cols.max(1));
    render_text_height_lh(&wrapped, fg, bg, target_char_h, line_height)
}

/// Render a string at a given pixel height.
///
/// The font is scaled so each character is `target_char_h` pixels tall.
/// Character width scales proportionally to maintain the monospace aspect ratio.
/// Multi-line input stacks lines tightly (no inter-line leading) — for a
/// typographic gap, use [`render_text_height_lh`] with `line_height > 1.0`.
///
/// Returns `(pixels, width, height)`.
pub fn render_text_height(
    text: &str,
    fg: [u8; 4],
    bg: [u8; 4],
    target_char_h: u32,
) -> (Vec<u8>, u32, u32) {
    render_text_height_lh(text, fg, bg, target_char_h, 1.0)
}

/// Same as [`render_text_height`], but stacks multi-line input with a
/// CSS-`line-height` ratio (`line_height = 1.2` ≈ default browser
/// leading). The first line still occupies `target_char_h` pixels;
/// each subsequent line advances by `round(target_char_h *
/// line_height)`, clamped to ≥ 1px. The output buffer height is
/// `(num_lines - 1) * line_advance + target_char_h` — no trailing
/// leading.
pub fn render_text_height_lh(
    text: &str,
    fg: [u8; 4],
    bg: [u8; 4],
    target_char_h: u32,
    line_height: f32,
) -> (Vec<u8>, u32, u32) {
    if target_char_h == 0 {
        return (vec![], 0, 0);
    }

    let lines: Vec<&str> = text.lines().collect();
    let max_cols = lines
        .iter()
        .map(|l| crate::glyph_map::cell_count(l))
        .max()
        .unwrap_or(0);
    let num_lines = lines.len() as u32;

    if max_cols == 0 || num_lines == 0 {
        return (vec![], 0, 0);
    }

    let scaled_char_w = char_width_for_height(target_char_h);
    let scaled_char_h = target_char_h;
    let line_advance = line_advance_px(scaled_char_h, line_height);

    // Glyph cells are resampled IN ISOLATION (see
    // `build_scaled_run_per_cell` for the why) and fetched lazily in
    // SCALE_BATCH runs, so per-size work tracks the glyphs this text
    // actually uses.
    //
    // Mitchell-Netravali — minimal overshoot, no halo rings. Lanczos
    // produces visible side-lobe ringing on sharp glyph edges; Triangle
    // is artifact-free but visibly soft.
    let mut run: Option<(u32, Arc<Bitmap>)> = None;

    let out_w = max_cols * scaled_char_w;
    let out_h = stacked_height_px(scaled_char_h, line_advance, num_lines);

    let mut buf = vec![0u8; (out_w * out_h * 4) as usize];

    // Fill background
    for pixel in buf.as_chunks_mut::<4>().0.iter_mut() {
        pixel.copy_from_slice(&bg);
    }

    // Blit characters
    for (line_idx, line) in lines.iter().enumerate() {
        let y_base = line_idx as u32 * line_advance;
        let mut col = 0u32;
        for ch in line.chars() {
            let x_base = col * scaled_char_w;
            let glyph_idx = match crate::glyph_map::map_char(ch) {
                crate::glyph_map::Mapped::Glyph(idx) => idx,
                crate::glyph_map::Mapped::Skip => continue,
                crate::glyph_map::Mapped::NotDef(cp) => {
                    crate::glyph_map::draw_notdef(
                        &mut buf,
                        out_w,
                        out_h,
                        x_base,
                        y_base,
                        scaled_char_w,
                        scaled_char_h,
                        cp,
                        fg,
                    );
                    col += 1;
                    continue;
                }
            };
            col += 1;
            let batch = glyph_idx / SCALE_BATCH;
            let scaled_run = match &run {
                Some((b, bmp)) if *b == batch => bmp,
                _ => {
                    let bmp = cached_glyph_batch(
                        batch,
                        scaled_char_w,
                        scaled_char_h,
                        ResampleFilter::Mitchell,
                    );
                    &run.insert((batch, bmp)).1
                }
            };
            let src_x = (glyph_idx - batch * SCALE_BATCH) * scaled_char_w;

            for gy in 0..scaled_char_h {
                for gx in 0..scaled_char_w {
                    let sx = src_x + gx;
                    if sx >= scaled_run.width() {
                        continue;
                    }
                    // Read coverage from the RGBA strip's alpha channel.
                    // RGB is constant 255 (white-tinted) and is unused
                    // at composite — we tint with `fg` directly.
                    let alpha = scaled_run.get_pixel(sx, gy)[3];
                    if alpha == 0 {
                        continue;
                    }

                    let px = x_base + gx;
                    let py = y_base + gy;
                    if px >= out_w || py >= out_h {
                        continue;
                    }

                    let off = ((py * out_w + px) * 4) as usize;
                    if alpha == 255 {
                        buf[off..off + 4].copy_from_slice(&fg);
                    } else {
                        // Gamma-correct blend in linear-sRGB space for
                        // RGB; alpha (channel 3) blends linearly since
                        // it isn't gamma-encoded.
                        buf[off] = blend_channel_gamma_correct(fg[0], bg[0], alpha);
                        buf[off + 1] = blend_channel_gamma_correct(fg[1], bg[1], alpha);
                        buf[off + 2] = blend_channel_gamma_correct(fg[2], bg[2], alpha);
                        let a = alpha as u16;
                        let inv_a = 255 - a;
                        buf[off + 3] = ((fg[3] as u16 * a + bg[3] as u16 * inv_a) / 255) as u8;
                    }
                }
            }
        }
    }

    (buf, out_w, out_h)
}

/// Render a string at an integer multiple of the base font size.
pub fn render_text_scaled(text: &str, fg: [u8; 4], bg: [u8; 4], scale: u32) -> (Vec<u8>, u32, u32) {
    let target_h = BASE_CHAR_H * scale.max(1);
    render_text_height(text, fg, bg, target_h)
}

// ─── Word wrap ──────────────────────────────────────────────────────────

/// Word-wrap text to fit within `max_cols` characters per line.
///
/// Breaks on spaces. Explicit newlines in the input are preserved.
/// Words longer than `max_cols` are placed on their own line (not split).
fn wrap_text(text: &str, max_cols: usize) -> String {
    let mut result = String::new();

    for input_line in text.lines() {
        if !result.is_empty() {
            result.push('\n');
        }

        let mut col = 0usize;
        for (i, word) in input_line.split(' ').enumerate() {
            let wlen = word.len();

            if i == 0 {
                // First word on this line — always emit
                result.push_str(word);
                col = wlen;
            } else if col + 1 + wlen <= max_cols {
                // Fits on current line with a space
                result.push(' ');
                result.push_str(word);
                col += 1 + wlen;
            } else {
                // Wrap to next line
                result.push('\n');
                result.push_str(word);
                col = wlen;
            }
        }
    }

    result
}

// ─── Helpers ────────────────────────────────────────────────────────────

/// Build a downscaled glyph strip by resampling each glyph cell IN
/// ISOLATION, then concatenating them horizontally.
///
/// Why per-cell: the source strip is `BASE_CHAR_W * CHAR_COUNT` pixels
/// wide with adjacent glyph cells touching. A single resize of the
/// whole strip lets a multi-tap reconstruction filter (Mitchell, Lanczos,
/// etc.) sample across cell boundaries — at the rightmost column of
/// glyph N, the filter footprint extends into the leftmost column of
/// glyph N+1, pulling a thin sliver of N+1's alpha into N's output.
/// At small char_h this manifests as visible "edges of other characters"
/// inside each glyph.
///
/// Implementation: wrap the strip as a `zenpixels::PixelSlice` (zero-
/// copy strided view), then `crop_view(...)` for each glyph cell — that
/// also-zero-copy adjusts only the slice's offset+stride. Drive
/// `zenresize::StreamingResize` row-by-row, feeding `cell.row(y)`
/// directly (a 26×4 byte sub-slice of the original strip's bytes — no
/// copy, no full-strip-width row reads).
///
/// The resampler clamps at the cell edge because the streaming
/// resizer's source dims are the *cell's* dims (26×54), not the full
/// strip. So the filter footprint can never reach into adjacent glyphs.
/// Scaled-strip cache keyed by canvas (target glyph dims + filter).
///
/// `MontageOptions::render` issues many text renders per call, almost
/// all at the same `char_h` (panel labels at one size, heatmap stats
/// at another). The strip itself is color-independent — RGB is constant
/// 255, alpha carries coverage, tinting happens at composite — so the
/// cache key is purely `(scaled_char_w, scaled_char_h, filter)`.
///
/// Glyphs are scaled lazily in RUNS of `SCALE_BATCH` consecutive
/// indices: batch `b` covers `[b·SCALE_BATCH, min((b+1)·SCALE_BATCH,
/// CHAR_COUNT))`. Per-size work then tracks the glyphs a label actually
/// uses instead of the whole (growing) atlas — a 20-glyph label at a
/// fresh size scales ~3 batches, not 96+ cells.
///
/// Why 4 (measured, `examples/glyph_batch_bench.rs`, with per-run
/// weight-table reuse in the producer): per-run overhead `a ≈ 2–3 µs`
/// (the once-per-run weight build), per-glyph `b` = 4.2–10.7 µs
/// across 12–54 px. Small batches win the COLD workload model (B=1:
/// 0.09–0.18 ms per fresh size for a 10-label set; whole-strip:
/// 0.40–1.03 ms) — but batch=1 defeats the compose loop's same-batch
/// memoization and makes the warm path pay a cache lookup per
/// character against a much larger entry set. B=4 keeps cold within
/// ~2.2× of the B=1 ideal (and ~2.6× better than whole-strip) while
/// text's index clustering (lowercase/digit runs) amortizes lookups.
pub(crate) const SCALE_BATCH: u32 = 4;

/// Byte budget for the scaled-glyph cache (sum of cached run bitmaps).
/// Generous for many (size × batch) combinations while bounding a
/// pathological many-sizes sequence; eviction is LRU.
const GLYPH_CACHE_BUDGET: usize = 6 << 20;

/// Fetch (building on miss) the scaled run for `batch` at the given
/// cell size. Returned bitmap holds `run_len(batch)` cells side by
/// side; glyph `idx` sits at `(idx - batch·SCALE_BATCH) * char_w`.
///
/// Bounded byte-budget LRU. Vec-based: entry count stays small enough
/// that linear scan beats a HashMap. Built outside the lock to avoid
/// serialising concurrent renders.
fn cached_glyph_batch(batch: u32, char_w: u32, char_h: u32, filter: ResampleFilter) -> Arc<Bitmap> {
    type Key = (u32, u32, u32, ResampleFilter);
    struct Cache {
        entries: Vec<(Key, Arc<Bitmap>)>,
        bytes: usize,
    }
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();

    let key: Key = (batch, char_w, char_h, filter);
    let cache = CACHE.get_or_init(|| {
        Mutex::new(Cache {
            entries: Vec::new(),
            bytes: 0,
        })
    });

    if let Ok(mut guard) = cache.lock()
        && let Some(idx) = guard.entries.iter().position(|(k, _)| *k == key)
    {
        // Bump to MRU (back of vec).
        let entry = guard.entries.remove(idx);
        guard.entries.push(entry);
        return guard.entries.last().unwrap().1.clone();
    }

    let start = batch * SCALE_BATCH;
    let count = SCALE_BATCH.min(CHAR_COUNT.saturating_sub(start));
    let scaled = Arc::new(build_scaled_run(start, count, char_w, char_h, filter));

    if let Ok(mut guard) = cache.lock() {
        // Race-tolerant: another thread may have inserted the same key
        // in the gap between the miss and the build. Dedup before push.
        if !guard.entries.iter().any(|(k, _)| *k == key) {
            let sz = scaled.as_raw().len();
            guard.entries.push((key, scaled.clone()));
            guard.bytes += sz;
            while guard.bytes > GLYPH_CACHE_BUDGET && guard.entries.len() > 1 {
                let (_, old) = guard.entries.remove(0);
                guard.bytes -= old.as_raw().len();
            }
        }
    }
    scaled
}

/// Build the scaled cells for glyph indices `[start, start+count)` at
/// the given cell size — the single producer both paths implement.
///
/// The `sdf-font` feature swaps the producer only: same dimensions,
/// same RGBA-with-coverage-alpha layout, same cache and composite path
/// downstream. `filter` stays in the cache key but does not affect SDF
/// output.
fn build_scaled_run(
    start: u32,
    count: u32,
    char_w: u32,
    char_h: u32,
    #[cfg_attr(feature = "sdf-font", allow(unused_variables))] filter: ResampleFilter,
) -> Bitmap {
    #[cfg(feature = "sdf-font")]
    {
        crate::sdf_font::build_scaled_run_sdf(start, count, char_w, char_h)
    }
    #[cfg(not(feature = "sdf-font"))]
    {
        build_scaled_run_per_cell(font_strip(), start, count, char_w, char_h, filter)
    }
}

/// Internal bench hook for `examples/glyph_batch_bench.rs` — times the
/// producer at arbitrary run lengths without going through the cache.
#[cfg(feature = "_internal_api")]
#[doc(hidden)]
pub fn bench_build_scaled_run(start: u32, count: u32, char_w: u32, char_h: u32) -> usize {
    build_scaled_run(start, count, char_w, char_h, ResampleFilter::Mitchell)
        .as_raw()
        .len()
}

/// Full-strip builder retained for the batch/strip equivalence test.
#[cfg(all(test, not(feature = "sdf-font")))]
fn build_scaled_strip_per_cell(
    strip: &Bitmap,
    scaled_char_w: u32,
    scaled_char_h: u32,
    filter: ResampleFilter,
) -> Bitmap {
    build_scaled_run_per_cell(strip, 0, CHAR_COUNT, scaled_char_w, scaled_char_h, filter)
}

#[cfg_attr(feature = "sdf-font", allow(dead_code))]
fn build_scaled_run_per_cell(
    strip: &Bitmap,
    start: u32,
    count: u32,
    scaled_char_w: u32,
    scaled_char_h: u32,
    filter: ResampleFilter,
) -> Bitmap {
    let scaled_strip_w = scaled_char_w * count.max(1);
    if scaled_char_w == 0 || scaled_char_h == 0 || count == 0 {
        return Bitmap::new(scaled_strip_w.max(1), scaled_char_h.max(1));
    }
    let mut out = vec![0u8; (scaled_strip_w as usize) * (scaled_char_h as usize) * 4];
    let scaled_strip_w_usize = scaled_strip_w as usize;
    let scaled_char_w_usize = scaled_char_w as usize;

    // Strided zero-copy view of the strip.
    let strip_slice = zenpixels::PixelView::new(
        strip.as_raw(),
        strip.width(),
        strip.height(),
        (strip.width() as usize) * 4,
        &zenpixels::formats::RGBA8_SRGB,
    )
    .expect("strip slice dims valid");

    let cfg =
        zenresize::ResizeConfig::builder(BASE_CHAR_W, BASE_CHAR_H, scaled_char_w, scaled_char_h)
            .filter(filter.to_zenresize_filter())
            .input(zenresize::PixelDescriptor::RGBA8_SRGB)
            .build();

    // One streaming resizer for the whole run: `reset()` between cells
    // preserves the filter weight tables and internal buffers (every
    // cell has identical dims/filter), so weights are computed once per
    // run instead of once per glyph.
    let mut sr = zenresize::StreamingResize::new(&cfg);

    for i in 0..count {
        let glyph_idx = (start + i).min(CHAR_COUNT - 1);
        // crop: zero-copy strided sub-view of cell `glyph_idx`.
        let cell = strip_slice
            .crop(glyph_idx * BASE_CHAR_W, 0, BASE_CHAR_W, BASE_CHAR_H)
            .expect("cell bounds valid");

        sr.reset();
        let cell_x = (i as usize) * scaled_char_w_usize;
        let mut output_y = 0u32;

        let drain = |sr: &mut zenresize::StreamingResize, output_y: &mut u32, out: &mut [u8]| {
            while let Some(row) = sr.next_output_row() {
                if *output_y < scaled_char_h {
                    let dst_off = (*output_y as usize * scaled_strip_w_usize + cell_x) * 4;
                    let n = scaled_char_w_usize * 4;
                    out[dst_off..dst_off + n].copy_from_slice(row);
                    *output_y += 1;
                }
            }
        };

        for y in 0..BASE_CHAR_H {
            // Zero-copy row slice — points into the strip's underlying
            // bytes via the cell's stride; no row-bytes copy here.
            let row = cell.row(y);
            sr.push_row(row).expect("push_row");
            drain(&mut sr, &mut output_y, &mut out);
        }
        sr.finish();
        drain(&mut sr, &mut output_y, &mut out);
    }
    Bitmap::from_raw(scaled_strip_w, scaled_char_h, out)
        .expect("per-cell strip dimensions match buffer")
}

/// Compute the character width for a given target character height.
fn char_width_for_height(target_char_h: u32) -> u32 {
    (BASE_CHAR_W * target_char_h + BASE_CHAR_H / 2) / BASE_CHAR_H
}

/// Vertical stride between consecutive lines (`char_h * line_height`,
/// clamped to ≥ 1px and ≥ char_h so lines never overlap). A
/// non-finite, non-positive, or NaN `line_height` is treated as 1.0.
fn line_advance_px(char_h: u32, line_height: f32) -> u32 {
    if char_h == 0 {
        return 0;
    }
    let lh = if line_height.is_finite() && line_height > 0.0 {
        line_height
    } else {
        1.0
    };
    let advance = ((char_h as f32) * lh).round() as u32;
    advance.max(char_h.max(1))
}

/// Total stacked height for `n` lines: `(n - 1) * line_advance + char_h`.
/// No trailing leading; a single line is exactly `char_h` tall.
fn stacked_height_px(char_h: u32, line_advance: u32, n_lines: u32) -> u32 {
    if n_lines == 0 {
        return 0;
    }
    n_lines
        .saturating_sub(1)
        .saturating_mul(line_advance)
        .saturating_add(char_h)
}

/// Map a character to its index in the font strip. Test-only: the
/// composers route through [`crate::glyph_map::map_char`], which adds
/// format-char skipping, fullwidth folding, emoji twins, and the
/// hex-in-box notdef (the old behavior silently rendered unknown
/// codepoints as space/Δ).
#[cfg(test)]
fn char_index(ch: char) -> u32 {
    let code = ch as u32;
    if (FIRST_CHAR..FIRST_CHAR + 95).contains(&code) {
        code - FIRST_CHAR
    } else {
        match ch {
            '\u{0394}' => 95, // Δ (Greek capital delta)
            _ => 0,           // space for unknown
        }
    }
}

/// Render multiple lines, each with its own color, at a char height that
/// makes the longest line fit within `max_width_px`.
///
/// Returns `(pixels, width, height)`. No word-wrapping — each line is
/// rendered as-is. The font size is computed from the longest line.
/// Lines stack tightly; for typographic leading, see
/// [`render_lines_fitted_lh`].
pub fn render_lines_fitted(
    lines: &[(&str, [u8; 4])],
    bg: [u8; 4],
    max_width_px: u32,
) -> (Vec<u8>, u32, u32) {
    render_lines_fitted_lh(lines, bg, max_width_px, 1.0)
}

/// Same as [`render_lines_fitted`], with a CSS-`line-height` ratio
/// applied to the inter-line stride.
pub fn render_lines_fitted_lh(
    lines: &[(&str, [u8; 4])],
    bg: [u8; 4],
    max_width_px: u32,
    line_height: f32,
) -> (Vec<u8>, u32, u32) {
    if lines.is_empty() || max_width_px == 0 {
        return (vec![], 0, 0);
    }
    let (char_w, char_h) = fit_char_h_for_lines(lines, max_width_px);
    if char_w == 0 || char_h == 0 {
        return (vec![], 0, 0);
    }
    let line_advance = line_advance_px(char_h, line_height);
    let longest = lines
        .iter()
        .map(|(s, _)| crate::glyph_map::cell_count(s))
        .max()
        .unwrap_or(0);

    let out_w = longest * char_w;
    let out_h = stacked_height_px(char_h, line_advance, lines.len() as u32);

    if out_w == 0 || out_h == 0 {
        return (vec![], 0, 0);
    }

    // Per-cell isolated resize, fetched lazily in SCALE_BATCH runs —
    // see `cached_glyph_batch` and `build_scaled_run_per_cell`.
    let mut run: Option<(u32, Arc<Bitmap>)> = None;

    let mut buf = vec![0u8; (out_w * out_h * 4) as usize];
    for pixel in buf.as_chunks_mut::<4>().0.iter_mut() {
        pixel.copy_from_slice(&bg);
    }

    for (line_idx, (text, fg)) in lines.iter().enumerate() {
        let y_base = line_idx as u32 * line_advance;
        // Center each line within the output width
        let line_w = crate::glyph_map::cell_count(text) * char_w;
        let x_offset = (out_w.saturating_sub(line_w)) / 2;
        let mut col = 0u32;
        for ch in text.chars() {
            let x_base = x_offset + col * char_w;
            let glyph_idx = match crate::glyph_map::map_char(ch) {
                crate::glyph_map::Mapped::Glyph(idx) => idx,
                crate::glyph_map::Mapped::Skip => continue,
                crate::glyph_map::Mapped::NotDef(cp) => {
                    crate::glyph_map::draw_notdef(
                        &mut buf, out_w, out_h, x_base, y_base, char_w, char_h, cp, *fg,
                    );
                    col += 1;
                    continue;
                }
            };
            col += 1;
            let batch = glyph_idx / SCALE_BATCH;
            let scaled_run = match &run {
                Some((b, bmp)) if *b == batch => bmp,
                _ => {
                    let bmp = cached_glyph_batch(batch, char_w, char_h, ResampleFilter::Mitchell);
                    &run.insert((batch, bmp)).1
                }
            };
            let src_x = (glyph_idx - batch * SCALE_BATCH) * char_w;

            for gy in 0..char_h {
                for gx in 0..char_w {
                    let sx = src_x + gx;
                    if sx >= scaled_run.width() {
                        continue;
                    }
                    // Coverage from the RGBA strip's alpha channel.
                    let alpha = scaled_run.get_pixel(sx, gy)[3];
                    if alpha == 0 {
                        continue;
                    }
                    let px = x_base + gx;
                    let py = y_base + gy;
                    if px >= out_w || py >= out_h {
                        continue;
                    }
                    let off = ((py * out_w + px) * 4) as usize;
                    if alpha == 255 {
                        buf[off..off + 4].copy_from_slice(fg);
                    } else {
                        // Gamma-correct RGB blend; linear alpha blend.
                        buf[off] = blend_channel_gamma_correct(fg[0], bg[0], alpha);
                        buf[off + 1] = blend_channel_gamma_correct(fg[1], bg[1], alpha);
                        buf[off + 2] = blend_channel_gamma_correct(fg[2], bg[2], alpha);
                        let a = alpha as u16;
                        let inv_a = 255 - a;
                        buf[off + 3] = ((fg[3] as u16 * a + bg[3] as u16 * inv_a) / 255) as u8;
                    }
                }
            }
        }
    }

    (buf, out_w, out_h)
}

/// Base character width in pixels (before scaling).
pub const GLYPH_W: u32 = BASE_CHAR_W;
/// Base character height in pixels (before scaling).
pub const GLYPH_H: u32 = BASE_CHAR_H;

/// Typographic vertical-centering adjustment as a fraction of char
/// height. Empirically derived from the embedded Consolas strip: the
/// inked cap-mid (cap-top + baseline)/2 sits at ~42.6% of cell height,
/// while the geometric cell mid is 50%. To make capitals appear
/// visually centered when an enclosing layout uses geometric
/// centering, shift the cell down by this fraction.
pub const TYPO_CAP_MID_OFFSET: f32 = 0.074;

// ─── Cheap measurement (no rasterization) ──────────────────────────────

/// Return the `(w, h)` a [`render_text_height`] call would produce —
/// without actually rasterizing. Useful for two-pass layout where the
/// measure pass needs sizes but the render pass will rasterize anyway.
pub fn measure_text_height(text: &str, target_char_h: u32) -> (u32, u32) {
    measure_text_height_lh(text, target_char_h, 1.0)
}

/// Same as [`measure_text_height`], with a CSS-`line-height` ratio
/// applied to the inter-line stride.
pub fn measure_text_height_lh(text: &str, target_char_h: u32, line_height: f32) -> (u32, u32) {
    if target_char_h == 0 {
        return (0, 0);
    }
    let lines: Vec<&str> = text.lines().collect();
    let max_cols = lines
        .iter()
        .map(|l| crate::glyph_map::cell_count(l))
        .max()
        .unwrap_or(0);
    let num_lines = lines.len() as u32;
    if max_cols == 0 || num_lines == 0 {
        return (0, 0);
    }
    let scaled_char_w = char_width_for_height(target_char_h);
    let line_advance = line_advance_px(target_char_h, line_height);
    (
        max_cols * scaled_char_w,
        stacked_height_px(target_char_h, line_advance, num_lines),
    )
}

/// Return the `(w, h)` a [`render_text_wrapped`] call would produce.
pub fn measure_text_wrapped(text: &str, target_char_h: u32, max_width_px: u32) -> (u32, u32) {
    measure_text_wrapped_lh(text, target_char_h, max_width_px, 1.0)
}

/// Same as [`measure_text_wrapped`], with a CSS-`line-height` ratio
/// applied to the inter-line stride.
pub fn measure_text_wrapped_lh(
    text: &str,
    target_char_h: u32,
    max_width_px: u32,
    line_height: f32,
) -> (u32, u32) {
    if target_char_h == 0 || max_width_px == 0 {
        return (0, 0);
    }
    let scaled_char_w = char_width_for_height(target_char_h);
    let max_cols = (max_width_px / scaled_char_w.max(1)) as usize;
    let wrapped = wrap_text(text, max_cols.max(1));
    measure_text_height_lh(&wrapped, target_char_h, line_height)
}

/// Return the `(w, h)` a [`render_lines_fitted`] call would produce.
pub fn measure_lines_fitted(lines: &[(&str, [u8; 4])], max_width_px: u32) -> (u32, u32) {
    measure_lines_fitted_lh(lines, max_width_px, 1.0)
}

/// Same as [`measure_lines_fitted`], with a CSS-`line-height` ratio
/// applied to the inter-line stride.
pub fn measure_lines_fitted_lh(
    lines: &[(&str, [u8; 4])],
    max_width_px: u32,
    line_height: f32,
) -> (u32, u32) {
    let (char_w, char_h) = fit_char_h_for_lines(lines, max_width_px);
    if char_w == 0 || char_h == 0 {
        return (0, 0);
    }
    let longest = lines.iter().map(|(s, _)| s.len()).max().unwrap_or(0) as u32;
    let line_advance = line_advance_px(char_h, line_height);
    (
        longest * char_w,
        stacked_height_px(char_h, line_advance, lines.len() as u32),
    )
}

/// Internal: derive `(char_w, char_h)` such that
/// `longest * char_w ≤ max_width_px`. `char_w` is computed via
/// [`char_width_for_height`] which rounds half-up, so the naive formula
/// `floor(max_width_px / longest * BASE_H/BASE_W)` can overshoot by 1
/// when char_w rounds up. We post-correct by decrementing `char_h`
/// until `longest * char_w ≤ max_width_px`.
pub(crate) fn fit_char_h_for_lines(lines: &[(&str, [u8; 4])], max_width_px: u32) -> (u32, u32) {
    if lines.is_empty() || max_width_px == 0 {
        return (0, 0);
    }
    let longest = lines
        .iter()
        .map(|(s, _)| crate::glyph_map::cell_count(s))
        .max()
        .unwrap_or(1);
    if longest == 0 {
        return (0, 0);
    }
    let mut char_h = (max_width_px as f32 / longest as f32
        * (BASE_CHAR_H as f32 / BASE_CHAR_W as f32))
        .floor() as u32;
    char_h = char_h.clamp(BASE_CHAR_H / 4, BASE_CHAR_H);
    // Post-correct: char_width_for_height rounds half-up, so the
    // initial char_h can produce `longest * char_w > max_width_px` by
    // 1 character pixel. Decrement until it fits or we hit the floor.
    while char_h > BASE_CHAR_H / 4 && longest * char_width_for_height(char_h) > max_width_px {
        char_h -= 1;
    }
    (char_width_for_height(char_h), char_h)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── glyph_map integration through the public composers ─────────

    #[test]
    fn emoji_renders_hex_box_not_delta() {
        // 🚀 previously clamped to a strip glyph; it must now render a
        // notdef box (which differs from Δ's pixels) one cell wide.
        let (rocket, w1, _) = render_text_height("🚀", [255; 4], [0; 4], 40);
        let (delta, w2, _) = render_text_height("Δ", [255; 4], [0; 4], 40);
        assert_eq!(w1, w2, "notdef occupies exactly one cell");
        assert_ne!(rocket, delta, "notdef must not masquerade as Δ");
        assert!(
            rocket.as_chunks::<4>().0.iter().any(|p| p[0] == 255),
            "box drawn"
        );
    }

    #[test]
    fn distinct_emoji_render_distinct_hex_boxes() {
        let (a, ..) = render_text_height("🚀", [255; 4], [0; 4], 40);
        let (b, ..) = render_text_height("🎉", [255; 4], [0; 4], 40);
        assert_ne!(a, b);
    }

    #[test]
    fn variation_selector_is_zero_width() {
        let (plain, w1, h1) = render_text_height("✔", [255; 4], [0; 4], 30);
        let (vs16, w2, h2) = render_text_height("✔\u{FE0F}", [255; 4], [0; 4], 30);
        assert_eq!((w1, h1), (w2, h2), "VS16 must not add a cell");
        assert_eq!(plain, vs16, "VS16 must not change pixels");
    }

    #[test]
    fn fullwidth_folds_render_as_ascii() {
        let (fw, w1, _) = render_text_height("（ｘ）", [255; 4], [0; 4], 30);
        let (ascii, w2, _) = render_text_height("(x)", [255; 4], [0; 4], 30);
        assert_eq!(w1, w2);
        assert_eq!(fw, ascii);
    }

    #[test]
    fn twin_maps_ornament_question_mark() {
        let (twin, ..) = render_text_height("❓", [255; 4], [0; 4], 30);
        let (ascii, ..) = render_text_height("?", [255; 4], [0; 4], 30);
        assert_eq!(twin, ascii);
    }

    #[test]
    fn measure_matches_render_for_mapped_text() {
        let text = "a🚀✔\u{FE0F}（）";
        let (buf, w, h) = render_text_height(text, [255; 4], [0; 4], 24);
        let (mw, mh) = measure_text_height(text, 24);
        assert_eq!((w, h), (mw, mh));
        assert_eq!(buf.len(), (w * h * 4) as usize);
    }

    /// Lazy batches must be pixel-identical to the full-strip build —
    /// every glyph, every pixel, at a representative size.
    #[test]
    fn batched_runs_match_full_strip() {
        let (w, h) = (13u32, 27u32);
        #[cfg(not(feature = "sdf-font"))]
        let full = build_scaled_strip_per_cell(font_strip(), w, h, ResampleFilter::Mitchell);
        #[cfg(feature = "sdf-font")]
        let full = crate::sdf_font::build_scaled_strip_sdf(w, h);
        for g in 0..CHAR_COUNT {
            let b = g / SCALE_BATCH;
            let run = cached_glyph_batch(b, w, h, ResampleFilter::Mitchell);
            let off = (g - b * SCALE_BATCH) * w;
            for y in 0..h {
                for x in 0..w {
                    assert_eq!(
                        run.get_pixel(off + x, y),
                        full.get_pixel(g * w + x, y),
                        "glyph {g} at ({x},{y})"
                    );
                }
            }
        }
    }

    #[test]
    fn font_strip_loads() {
        let strip = font_strip();
        assert_eq!(strip.width(), BASE_CHAR_W * CHAR_COUNT);
        assert_eq!(strip.height(), BASE_CHAR_H);
    }

    #[test]
    fn render_empty_string() {
        let (buf, w, h) = render_text("", [255; 4], [0; 4]);
        assert_eq!(w, 0);
        assert_eq!(h, 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn render_single_char() {
        let (buf, w, h) = render_text("A", [255, 255, 255, 255], [0, 0, 0, 255]);
        assert_eq!(w, BASE_CHAR_W);
        assert_eq!(h, BASE_CHAR_H);
        assert_eq!(buf.len(), (w * h * 4) as usize);
        let white_count = buf.chunks(4).filter(|px| px[0] > 128).count();
        assert!(
            white_count > 10,
            "A glyph should have lit pixels, got {white_count}"
        );
    }

    #[test]
    fn render_multiline() {
        let (buf, w, h) = render_text("AB\nCD", [255; 4], [0; 4]);
        assert_eq!(w, 2 * BASE_CHAR_W);
        assert_eq!(h, 2 * BASE_CHAR_H);
        assert_eq!(buf.len(), (w * h * 4) as usize);
    }

    #[test]
    fn render_height_custom() {
        let target_h = 40;
        let (buf, w, h) = render_text_height("AB", [255; 4], [0; 4], target_h);
        assert_eq!(h, target_h);
        assert!(w > 0);
        assert_eq!(buf.len(), (w * h * 4) as usize);
    }

    #[test]
    fn digits_have_pixels() {
        let strip = font_strip();
        for d in b'0'..=b'9' {
            let idx = char_index(d as char);
            let x0 = idx * BASE_CHAR_W;
            let mut lit = 0u32;
            for y in 0..BASE_CHAR_H {
                for x in x0..x0 + BASE_CHAR_W {
                    // Strip is RGBA — coverage is in alpha.
                    if strip.get_pixel(x, y)[3] > 128 {
                        lit += 1;
                    }
                }
            }
            assert!(
                lit > 10,
                "digit '{}' should have lit pixels, got {lit}",
                d as char
            );
        }
    }

    #[test]
    fn wrap_short_text_unchanged() {
        assert_eq!(wrap_text("hello world", 80), "hello world");
    }

    #[test]
    fn wrap_breaks_on_space() {
        assert_eq!(wrap_text("aaa bbb ccc", 7), "aaa bbb\nccc");
    }

    #[test]
    fn wrap_preserves_newlines() {
        assert_eq!(wrap_text("aaa\nbbb ccc", 7), "aaa\nbbb ccc");
    }

    #[test]
    fn wrap_long_word_own_line() {
        assert_eq!(wrap_text("hi verylongword ok", 8), "hi\nverylongword\nok");
    }

    #[test]
    fn wrapped_render_produces_more_lines() {
        let text = "zdsim:0.13 delta:[80,40,0] 23.4% differ category:perceptual";
        let narrow = render_text_wrapped(text, [255; 4], [0; 4], 20, 200);
        let wide = render_text_wrapped(text, [255; 4], [0; 4], 20, 2000);
        // Narrow should be taller (more lines)
        assert!(
            narrow.2 > wide.2,
            "narrow {} should be taller than wide {}",
            narrow.2,
            wide.2
        );
    }
}
