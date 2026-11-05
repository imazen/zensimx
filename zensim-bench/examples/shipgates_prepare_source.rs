//! Explicit source preparation by an opaque crop or lossless PNG8 expansion.
#![forbid(unsafe_code)]
use rgb::FromSlice;
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::PathBuf};
use zenpixels::{
    PixelDescriptor,
    policy::{AlphaPolicy, ConvertOptions},
};
use zenpixels_convert::{PixelBufferConvertTypedExt, RowConverter};

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.get(1).is_some_and(|mode| mode == "--bmp-list") {
        if args.len() != 4 {
            return Err("--bmp-list LIST OUTPUT_DIRECTORY required".into());
        }
        return prepare_bmp_list(&PathBuf::from(&args[2]), &PathBuf::from(&args[3]));
    }
    let expand = args.get(1).is_some_and(|mode| mode == "--expand-png8");
    if args.len() != 4 || (!expand && args[1] != "--crop-opaque") {
        return Err("expected --crop-opaque or --expand-png8, input PNG and fresh output PNG; matte conversion is not admitted".into());
    }
    let input = PathBuf::from(&args[2]);
    let output = PathBuf::from(&args[3]);
    let bytes = fs::read(&input)?;
    let image = zenpng::decode(
        &bytes,
        &zenpng::PngDecodeConfig::default(),
        &enough::Unstoppable,
    )?;
    let info = &image.info;
    let decoded_eight = info.bit_depth == 8
        || (expand && matches!(info.color_type, 0 | 3) && matches!(info.bit_depth, 1 | 2 | 4));
    if !decoded_eight
        || info.sequence.is_animation()
        || info.icc_profile.is_some()
        || info.exif.is_some()
        || info.cicp.is_some()
        || info.chromaticities.is_some()
        || info.content_light_level.is_some()
        || info.mastering_display.is_some()
        || info.source_gamma.is_some_and(|v| v != 45455)
    {
        return Err("outside static sRGB8 source contract".into());
    }
    let rgba = image.pixels.to_rgba8().copy_to_contiguous_bytes();
    let rect = if expand {
        if rgba.as_chunks::<4>().0.iter().any(|pixel| pixel[3] != 255) {
            return Err("PNG8 expansion requires every pixel opaque".into());
        }
        Rect {
            x: 0,
            y: 0,
            width: info.width as usize,
            height: info.height as usize,
        }
    } else {
        largest_opaque(imgref::ImgRef::new(
            rgba.as_rgba(),
            info.width as usize,
            info.height as usize,
        ))
        .ok_or("source has no opaque rectangle")?
    };
    let (width, height) = (rect.width as u32, rect.height as u32);
    let mut rgb = vec![0u8; width as usize * height as usize * 3];
    let options = ConvertOptions::retain_channels().with_alpha_policy(AlphaPolicy::DiscardIfOpaque);
    let mut converter = RowConverter::new_explicit(
        PixelDescriptor::RGBA8_SRGB,
        PixelDescriptor::RGB8_SRGB,
        &options,
    )?;
    converter.convert_rows(
        &rgba[(rect.y * info.width as usize + rect.x) * 4..],
        info.width as usize * 4,
        &mut rgb,
        width as usize * 3,
        width,
        height,
    )?;
    let pixels = imgref::ImgRef::new(rgb.as_rgb(), width as usize, height as usize);
    let png = if expand {
        canonical_png8(pixels)?
    } else {
        zenpng::encode_rgb8(
            pixels,
            None,
            &zenpng::EncodeConfig::default().with_compression(zenpng::Compression::Fastest),
            &enough::Unstoppable,
            &enough::Unstoppable,
        )?
    };
    let back = zenpng::decode(
        &png,
        &zenpng::PngDecodeConfig::default(),
        &enough::Unstoppable,
    )?;
    let check = back.pixels.to_rgb8().copy_to_contiguous_bytes();
    if check != rgb
        || back.info.width != width
        || back.info.height != height
        || (expand && (back.info.bit_depth != 8 || !matches!(back.info.color_type, 0 | 2)))
    {
        return Err("prepared PNG readback differs".into());
    }
    fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&output)?
        .write_all(&png)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
        "input":input,"input_sha256":sha(&bytes),"input_rgba_sha256":sha(&rgba),
        "output":output,"output_sha256":sha(&png),"output_rgb_sha256":sha(&rgb),
        "width":width,"height":height,"crop":{"x":rect.x,"y":rect.y,"width":rect.width,"height":rect.height},
        "mode":if expand {"expand-png8"} else {"crop-opaque"},
        "source_png_bit_depth":info.bit_depth,"source_png_color_type":info.color_type,
        "output_png_bit_depth":back.info.bit_depth,"output_png_color_type":back.info.color_type,
        "recipe":if expand {"lossless palette/gray expansion; no crop; opaque PNG8 via zenpng"} else {"maximal opaque rectangle; alpha discard verified by RowConverter; zenpng RGB8"},
        "driver_sha256":sha(&fs::read(std::env::current_exe()?)?)}))?
    );
    Ok(())
}

/// Existing first-party stats encoding avoids palette selection. Explicit gray8
/// encoding prevents automatic sub-byte grayscale. Neither path changes samples.
fn canonical_png8(
    pixels: imgref::ImgRef<'_, rgb::RGB8>,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let config = zenpng::EncodeConfig::default().with_compression(zenpng::Compression::Fastest);
    let result = if pixels
        .rows()
        .all(|row| row.iter().all(|p| p.r == p.g && p.r == p.b))
    {
        let gray: Vec<u8> = pixels
            .rows()
            .flat_map(|row| row.iter().map(|p| p.r))
            .collect();
        zenpng::encode_gray8(
            imgref::ImgRef::new(gray.as_gray(), pixels.width(), pixels.height()),
            None,
            &config,
            &enough::Unstoppable,
            &enough::Unstoppable,
        )
    } else {
        zenpng::encode_rgb8_with_stats(
            pixels,
            None,
            &config,
            &enough::Unstoppable,
            &enough::Unstoppable,
        )
        .map(|result| result.0)
    };
    Ok(result?)
}

#[derive(Clone, Copy, Debug)]
struct Rect {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

/// Largest all-opaque rectangle, with deterministic top/left tie breaking.
/// Row iteration preserves arbitrary source strides.
fn largest_opaque(pixels: imgref::ImgRef<'_, rgb::RGBA8>) -> Option<Rect> {
    let mut heights = vec![0; pixels.width()];
    let mut best: Option<Rect> = None;
    for (y, row) in pixels.rows().enumerate() {
        for (height, pixel) in heights.iter_mut().zip(row) {
            *height = if pixel.a == 255 { *height + 1 } else { 0 };
        }
        let mut stack: Vec<(usize, usize)> = Vec::with_capacity(pixels.width());
        for x in 0..=pixels.width() {
            let height = heights.get(x).copied().unwrap_or(0);
            let mut start = x;
            while stack.last().is_some_and(|&(_, h)| h > height) {
                let (left, h) = stack.pop().expect("nonempty histogram stack");
                let candidate = Rect {
                    x: left,
                    y: y + 1 - h,
                    width: x - left,
                    height: h,
                };
                if best.is_none_or(|b| {
                    candidate.width * candidate.height > b.width * b.height
                        || (candidate.width * candidate.height == b.width * b.height
                            && (candidate.y, candidate.x, candidate.height, candidate.width)
                                < (b.y, b.x, b.height, b.width))
                }) {
                    best = Some(candidate);
                }
                start = left;
            }
            if height > 0 && stack.last().is_none_or(|&(_, h)| h < height) {
                stack.push((start, height));
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canonical_png8_preserves_strided_palette_and_gray_samples_without_subbyte_output() {
        for gray in [false, true] {
            let mut pixels = vec![rgb::RGB8::new(91, 92, 93); 10];
            for y in 0..2 {
                for x in 0..3 {
                    pixels[y * 5 + x] = if gray {
                        rgb::RGB8::new(
                            if x == 0 { 0 } else { 255 },
                            if x == 0 { 0 } else { 255 },
                            if x == 0 { 0 } else { 255 },
                        )
                    } else if x == 0 {
                        rgb::RGB8::new(255, 0, 0)
                    } else {
                        rgb::RGB8::new(0, 0, 255)
                    };
                }
            }
            let view = imgref::ImgRef::new_stride(&pixels, 3, 2, 5);
            let png = canonical_png8(view).unwrap();
            let back = zenpng::decode(
                &png,
                &zenpng::PngDecodeConfig::default(),
                &enough::Unstoppable,
            )
            .unwrap();
            assert_eq!(back.info.bit_depth, 8);
            assert_eq!(back.info.color_type, if gray { 0 } else { 2 });
            let expected: Vec<u8> = view
                .rows()
                .flat_map(|row| row.iter().flat_map(|p| [p.r, p.g, p.b]))
                .collect();
            assert_eq!(back.pixels.to_rgb8().copy_to_contiguous_bytes(), expected);
        }
    }
    #[test]
    fn every_three_by_three_alpha_mask_matches_exhaustive_strided_oracle() {
        for mask in 0..512 {
            let mut pixels = vec![rgb::RGBA8::new(0, 0, 0, 0); 15];
            for y in 0..3 {
                for x in 0..3 {
                    pixels[y * 5 + x].a = if mask & (1 << (y * 3 + x)) != 0 {
                        255
                    } else {
                        0
                    };
                }
            }
            let found = largest_opaque(imgref::ImgRef::new_stride(&pixels, 3, 3, 5));
            let mut maximum = 0;
            for y0 in 0..3 {
                for y1 in y0 + 1..=3 {
                    for x0 in 0..3 {
                        for x1 in x0 + 1..=3 {
                            if (y0..y1).all(|y| (x0..x1).all(|x| pixels[y * 5 + x].a == 255)) {
                                maximum = maximum.max((y1 - y0) * (x1 - x0));
                            }
                        }
                    }
                }
            }
            assert_eq!(
                found.map_or(0, |r| r.width * r.height),
                maximum,
                "mask{mask}"
            );
            if let Some(r) = found {
                assert!(
                    (r.y..r.y + r.height)
                        .all(|y| (r.x..r.x + r.width).all(|x| pixels[y * 5 + x].a == 255))
                );
            }
        }
    }
}

// Developer-only transport: preserve original BMP samples in a PNG container.
fn prepare_bmp_list(
    list: &std::path::Path,
    output: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::collections::BTreeSet;
    let inputs: Vec<PathBuf> = fs::read_to_string(list)?
        .lines()
        .map(PathBuf::from)
        .collect();
    let mut names = BTreeSet::new();
    for p in &inputs {
        let name = p.file_stem().ok_or("missing filename")?;
        if !names.insert(name.to_owned()) {
            return Err("ambiguous output filename".into());
        }
    }
    fs::create_dir(output)?;
    let mut receipts = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output.join("PIXELS.jsonl"))?;
    for (i, input) in inputs.iter().enumerate() {
        let bytes = fs::read(input)?;
        let image = zenbitmaps::decode_bmp(&bytes, enough::Unstoppable)?;
        if image.layout != zenbitmaps::PixelLayout::Rgb8 {
            return Err("RGB8 BMP required; no conversion".into());
        }
        let rgb = image.pixels();
        let pixels = imgref::ImgRef::new(rgb.as_rgb(), image.width as usize, image.height as usize);
        let png = zenpng::encode_rgb8(
            pixels,
            None,
            &zenpng::EncodeConfig::default().with_compression(zenpng::Compression::Fastest),
            &enough::Unstoppable,
            &enough::Unstoppable,
        )?;
        let back = zenpng::decode(
            &png,
            &zenpng::PngDecodeConfig::default(),
            &enough::Unstoppable,
        )?;
        if back.info.width != image.width
            || back.info.height != image.height
            || back.pixels.to_rgb8().copy_to_contiguous_bytes() != rgb
        {
            return Err("BMP to PNG changed decoded pixels".into());
        }
        let dest = output
            .join(input.file_stem().ok_or("missing filename")?)
            .with_extension("png");
        fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&dest)?
            .write_all(&png)?;
        writeln!(
            receipts,
            "{}",
            serde_json::json!({"input":input,"output":dest,"source_sha256":sha(&bytes),"png_sha256":sha(&png),"width":image.width,"height":image.height,"rgb_sha256":sha(rgb),"roundtrip_exact":true})
        )?;
        if (i + 1) % 1000 == 0 {
            eprintln!("BMP transport {}/{}", i + 1, inputs.len());
        }
    }
    eprintln!(
        "BMP transport complete: {} inputs; exact PNG readback for every input",
        inputs.len()
    );
    Ok(())
}
