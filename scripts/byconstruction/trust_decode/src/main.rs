//! Native RGB8 ingress for dataset assessments. JPEG uses pinned zenjpeg main;
//! PNG uses zenpng and the feature extractor's canonical row conversion.
//! The dataset trust screen (zensimx `scripts/byconstruction/dataset_trust.py`): the pinned judge build
//! reads PNG, JPEG and WebP only, so BMP and PNM inputs are decoded here with zenbitmaps and written as RGB8 PNG;
//! the frozen DVIFM runtime also lacks WebP, so WebP inputs are decoded with zenwebp (`oneshot::decode_rgba`, its
//! default upsampling) for that runtime alone.
//!
//!     trust_decode LIST OUT_DIR > MAP.tsv
//!
//! LIST holds one input path per line. Each output is `OUT_DIR/<sha256 of the packed RGB8 pixels>.png`; MAP.tsv has
//! `input`, `png`, `width`, `height`, `rgb8_sha256` per input. Gray inputs are expanded to RGB; an alpha channel is
//! dropped only when every pixel is opaque, otherwise refused.
#![forbid(unsafe_code)]
use std::borrow::Cow;
use std::io::{BufRead, Write};
use std::path::Path;

use sha2::{Digest, Sha256};
use zencodec::decode::{Decode, DecodeJob, DecoderConfig};
use zenpixels::{PixelBuffer, PixelDescriptor};

fn decode(bytes: &[u8]) -> Result<PixelBuffer, String> {
    macro_rules! run {
        ($cfg:expr) => {
            $cfg.job()
                .decoder(Cow::Borrowed(bytes), &[])
                .map_err(|e| format!("job: {e}"))?
                .decode()
                .map(|o| o.into_buffer())
                .map_err(|e| format!("decode: {e}"))
        };
    }
    match bytes {
        [137, b'P', b'N', b'G', ..] => zenpng::decode(
            bytes,
            &zenpng::PngDecodeConfig::default(),
            &enough::Unstoppable,
        )
        .map(|o| o.pixels)
        .map_err(|e| format!("png decode: {e:?}")),
        [b'B', b'M', ..] => run!(zenbitmaps::BmpDecoderConfig::new()),
        [b'P', b'1'..=b'7', ..] => run!(zenbitmaps::PnmDecoderConfig::new()),
        _ => Err("neither BMP nor PNM by magic bytes".into()),
    }
}

/// Packed RGB8, read row by row through the buffer's stride.
fn rgb8(pb: &PixelBuffer) -> Result<Vec<u8>, String> {
    let desc = pb.descriptor();
    let (w, h) = (pb.width() as usize, pb.height() as usize);
    let slice = pb.as_slice();
    let (stride, data) = (slice.stride(), slice.as_strided_bytes());
    let mut out = Vec::with_capacity(w * h * 3);
    if desc.layout_compatible(PixelDescriptor::RGB8)
        || desc.layout_compatible(PixelDescriptor::RGB8_SRGB)
    {
        for row in 0..h {
            out.extend_from_slice(&data[row * stride..row * stride + w * 3]);
        }
    } else if desc.layout_compatible(PixelDescriptor::GRAY8)
        || desc.layout_compatible(PixelDescriptor::GRAY8_SRGB)
    {
        for row in 0..h {
            for &v in &data[row * stride..row * stride + w] {
                out.extend_from_slice(&[v, v, v]);
            }
        }
    } else {
        // The feature extractor's canonical row converter owns narrowing.
        // Preserve this tool's refusal of alpha and unsupported channel layouts.
        if desc.has_alpha() || !matches!(desc.channels(), 1 | 3) {
            return Err(format!("unsupported decoded layout {desc:?}"));
        }
        use zenpixels_convert::converter::RowConverter;
        out.resize(w * h * 3, 0);
        let mut converter = RowConverter::new(desc, PixelDescriptor::RGB8_SRGB)
            .map_err(|e| format!("row conversion plan: {e}"))?;
        converter
            .convert_rows(data, stride, &mut out, w * 3, w as u32, h as u32)
            .map_err(|e| format!("row conversion: {e}"))?;
    }
    Ok(out)
}

/// WebP through zenwebp's native one-shot decoder (RGBA8, alpha 255 where opaque), as packed RGB8.
fn webp_rgb8(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    let (rgba, w, h) =
        zenwebp::oneshot::decode_rgba(bytes).map_err(|e| format!("webp decode: {e:?}"))?;
    let mut out = Vec::with_capacity(w as usize * h as usize * 3);
    for px in rgba.as_chunks::<4>().0 {
        if px[3] != 255 {
            return Err("webp has non-opaque alpha".into());
        }
        out.extend_from_slice(&px[..3]);
    }
    Ok((w, h, out))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 && !(args.len() == 4 && args[3] == "--preserve-rgb8-png") {
        eprintln!("usage: trust_decode LIST OUT_DIR [--preserve-rgb8-png] > MAP.tsv");
        std::process::exit(2);
    }
    let out_dir = Path::new(&args[2]);
    std::fs::create_dir_all(out_dir).expect("create OUT_DIR");
    let list = std::io::BufReader::new(std::fs::File::open(&args[1]).expect("open LIST"));
    let mut stdout = std::io::stdout().lock();
    writeln!(
        stdout,
        "input\tpng\twidth\theight\trgb8_sha256\tencoded_sha256"
    )
    .unwrap();
    for (index, line) in list.lines().enumerate() {
        let input = line.expect("read LIST");
        if input.is_empty() {
            continue;
        }
        let bytes = std::fs::read(&input).unwrap_or_else(|e| panic!("{input}: {e}"));
        let mut original_png = false;
        let (w, h, px) = if args.len() == 4 && bytes.starts_with(b"\x89PNG") {
            let pb = decode(&bytes).unwrap_or_else(|e| panic!("{input}: {e}"));
            original_png = pb.descriptor().layout_compatible(PixelDescriptor::RGB8)
                || pb
                    .descriptor()
                    .layout_compatible(PixelDescriptor::RGB8_SRGB);
            (
                pb.width(),
                pb.height(),
                rgb8(&pb).unwrap_or_else(|e| panic!("{input}: {e}")),
            )
        } else if bytes.starts_with(&[0xff, 0xd8]) {
            let decoded = zenjpeg::decoder::Decoder::new()
                .decode(&bytes, enough::Unstoppable)
                .unwrap_or_else(|e| panic!("{input}: jpeg decode: {e:?}"));
            let (w, h) = decoded.dimensions();
            let pixels = decoded.pixels_u8().expect("JPEG RGB8 output").to_vec();
            assert_eq!(
                pixels.len(),
                w as usize * h as usize * 3,
                "{input}: JPEG RGB8 length"
            );
            (w, h, pixels)
        } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
            webp_rgb8(&bytes).unwrap_or_else(|e| panic!("{input}: {e}"))
        } else {
            let pb = decode(&bytes).unwrap_or_else(|e| panic!("{input}: {e}"));
            let px = rgb8(&pb).unwrap_or_else(|e| panic!("{input}: {e}"));
            (pb.width(), pb.height(), px)
        };
        let digest: String = Sha256::digest(&px)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let dst = if original_png {
            std::path::PathBuf::from(&input)
        } else {
            out_dir.join(format!("{digest}.png"))
        };
        if !dst.exists() {
            use rgb::FromSlice;
            let img = imgref::ImgRef::new(px.as_rgb(), w as usize, h as usize);
            let png = zenpng::encode_rgb8(
                img,
                None,
                &zenpng::EncodeConfig::default(),
                &enough::Unstoppable,
                &enough::Unstoppable,
            )
            .unwrap_or_else(|e| panic!("{input}: png encode: {e:?}"));
            let tmp = dst.with_extension("png.part");
            std::fs::write(&tmp, png).expect("write png");
            std::fs::rename(&tmp, &dst).expect("rename png");
        }
        let encoded: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        writeln!(
            stdout,
            "{input}\t{}\t{w}\t{h}\t{digest}\t{encoded}",
            dst.display()
        )
        .unwrap();
        if (index + 1) % 100 == 0 {
            stdout.flush().unwrap();
            eprintln!("decoded {} inputs", index + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_rgb16_narrows_at_half_step_boundaries() {
        let values = [0u16, 128, 129, 256, 257, 384, 385, 65534, 65535];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut buffer = PixelBuffer::new(3, 1, PixelDescriptor::RGB16_SRGB);
        buffer.as_slice_mut().row_mut(0)[..bytes.len()].copy_from_slice(&bytes);
        assert_eq!(rgb8(&buffer).unwrap(), [0, 0, 1, 1, 1, 1, 1, 255, 255]);
    }
}
