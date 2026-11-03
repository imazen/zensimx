//! The dataset trust screen's decoder (zensimx `scripts/byconstruction/dataset_trust.py`): the pinned judge build
//! reads PNG, JPEG and WebP only, so BMP and PNM inputs are decoded here with zenbitmaps and written as RGB8 PNG.
//!
//!     trust_decode LIST OUT_DIR > MAP.tsv
//!
//! LIST holds one input path per line. Each output is `OUT_DIR/<sha256 of the packed RGB8 pixels>.png`; MAP.tsv has
//! `input`, `png`, `width`, `height`, `rgb8_sha256` per input. Gray inputs are expanded to RGB; alpha is refused.
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
    if desc.layout_compatible(PixelDescriptor::RGB8) || desc.layout_compatible(PixelDescriptor::RGB8_SRGB) {
        for row in 0..h {
            out.extend_from_slice(&data[row * stride..row * stride + w * 3]);
        }
    } else if desc.layout_compatible(PixelDescriptor::GRAY8) || desc.layout_compatible(PixelDescriptor::GRAY8_SRGB) {
        for row in 0..h {
            for &v in &data[row * stride..row * stride + w] {
                out.extend_from_slice(&[v, v, v]);
            }
        }
    } else {
        return Err(format!("unsupported decoded layout {desc:?}"));
    }
    Ok(out)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: trust_decode LIST OUT_DIR > MAP.tsv");
        std::process::exit(2);
    }
    let out_dir = Path::new(&args[2]);
    std::fs::create_dir_all(out_dir).expect("create OUT_DIR");
    let list = std::io::BufReader::new(std::fs::File::open(&args[1]).expect("open LIST"));
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "input\tpng\twidth\theight\trgb8_sha256").unwrap();
    for line in list.lines() {
        let input = line.expect("read LIST");
        if input.is_empty() {
            continue;
        }
        let bytes = std::fs::read(&input).unwrap_or_else(|e| panic!("{input}: {e}"));
        let pb = decode(&bytes).unwrap_or_else(|e| panic!("{input}: {e}"));
        let (w, h) = (pb.width(), pb.height());
        let px = rgb8(&pb).unwrap_or_else(|e| panic!("{input}: {e}"));
        let digest: String = Sha256::digest(&px).iter().map(|b| format!("{b:02x}")).collect();
        let dst = out_dir.join(format!("{digest}.png"));
        if !dst.exists() {
            use rgb::FromSlice;
            let img = imgref::ImgRef::new(px.as_rgb(), w as usize, h as usize);
            let png = zenpng::encode_rgb8(img, None, &zenpng::EncodeConfig::default(), &enough::Unstoppable,
                                          &enough::Unstoppable)
                .unwrap_or_else(|e| panic!("{input}: png encode: {e:?}"));
            let tmp = dst.with_extension("png.part");
            std::fs::write(&tmp, png).expect("write png");
            std::fs::rename(&tmp, &dst).expect("rename png");
        }
        writeln!(stdout, "{input}\t{}\t{w}\t{h}\t{digest}", dst.display()).unwrap();
    }
}
