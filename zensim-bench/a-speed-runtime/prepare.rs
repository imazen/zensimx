//! Fixture preparation is outside timed scoring. Pixels and JPEG bytes
//! remain in the caller's artifact directory; no encoded images enter git.
#![forbid(unsafe_code)]
use enough::Unstoppable;
use std::{error::Error, path::Path};
use zenpixels_convert::PixelBufferConvertTypedExt;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    let sources: Vec<String> = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let root = Path::new(&args[2]);
    std::fs::create_dir_all(root)?;
    assert!(
        !root.join("pairs.json").exists(),
        "preserve existing fixture manifests"
    );
    let mut pairs = Vec::new();
    let sizes: Vec<usize> = std::env::var("A_SPEED_SIZES")
        .unwrap_or_else(|_| "64,256,1024,4096".into())
        .split(',')
        .map(|n| n.parse().expect("positive integer fixture size"))
        .collect();
    assert!(!sizes.is_empty() && sizes.iter().all(|&n| n > 0));
    let mut counts = vec![0usize; sizes.len()];
    for (index, source) in sources.iter().enumerate() {
        let bytes = std::fs::read(source)?;
        let decoded = zenpng::decode(&bytes, &zenpng::PngDecodeConfig::default(), &Unstoppable)?;
        let rgb = decoded.pixels.to_rgb8();
        let view = rgb.as_imgref();
        let (w, h) = (view.width(), view.height());
        let input: Vec<u8> = view
            .rows()
            .flat_map(|row| row.iter().flat_map(|p| [p.r, p.g, p.b]))
            .collect();
        for (slot, &n) in sizes.iter().enumerate() {
            if counts[slot] >= 20
                || (n < 4096 && (w < n || h < n))
                || (n == 4096 && (!source.contains("png-v3/") || w.max(h) < 3840))
            {
                continue;
            }
            // A central square crop preserves geometry and never enlarges.
            let side = w.min(h);
            let (out_w, out_h, resize_input, in_w, in_h) = if n == 4096 {
                let scale = (4096.0 / w.max(h) as f64).min(1.0);
                (
                    (w as f64 * scale).round() as usize,
                    (h as f64 * scale).round() as usize,
                    input.clone(),
                    w,
                    h,
                )
            } else {
                let mut crop = Vec::with_capacity(side * side * 3);
                for y in (h - side) / 2..(h - side) / 2 + side {
                    let start = (y * w + (w - side) / 2) * 3;
                    crop.extend_from_slice(&input[start..start + side * 3]);
                }
                (n, n, crop, side, side)
            };
            let config = zenresize::ResizeConfig::builder(
                in_w as u32,
                in_h as u32,
                out_w as u32,
                out_h as u32,
            )
            .filter(zenresize::Filter::Mitchell)
            .format(zenresize::PixelDescriptor::RGB8_SRGB)
            .build();
            let reference = zenresize::Resizer::new(&config).resize(&resize_input);
            let cfg = zenjpeg::encoder::EncoderConfig::ycbcr(
                40.0,
                zenjpeg::encoder::ChromaSubsampling::Quarter,
            );
            let mut encoder = cfg.encode_from_bytes(
                out_w as u32,
                out_h as u32,
                zenjpeg::encoder::PixelLayout::Rgb8Srgb,
            )?;
            encoder.push_packed(&reference, Unstoppable)?;
            let jpeg = encoder.finish()?;
            // Independent readback checks that the generated JPEG is accepted
            // by another implementation. Only zenjpeg's decoded bytes become
            // the distorted benchmark input.
            let oracle = image::load_from_memory(&jpeg)?;
            assert_eq!(
                (oracle.width(), oracle.height()),
                (out_w as u32, out_h as u32)
            );
            let distorted = zenjpeg::decoder::Decoder::new()
                .decode(&jpeg, Unstoppable)?
                .into_pixels_u8()
                .ok_or("no RGB8 JPEG pixels")?;
            assert_eq!(distorted.len(), reference.len());
            let id = format!("{n}_{index:03}");
            let a = format!("{id}.ref.rgb8");
            let b = format!("{id}.dist.rgb8");
            std::fs::write(root.join(&a), reference)?;
            std::fs::write(root.join(&b), distorted)?;
            std::fs::write(root.join(format!("{id}.q40.jpg")), jpeg)?;
            let crop_rect = if n == 4096 {
                [0, 0, w, h]
            } else {
                [(w - side) / 2, (h - side) / 2, side, side]
            };
            pairs.push(serde_json::json!({"id":id,"size":n,"width":out_w,"height":out_h,"source":source,"reference":a,"distorted":b,"quality":40,"subsampling":"420","crop":crop_rect,"resize":"zenresize Mitchell RGB8_SRGB"}));
            counts[slot] += 1;
            eprintln!("prepared {id} from {source}");
        }
    }
    assert_eq!(counts, vec![20; sizes.len()]);
    std::fs::write(root.join("pairs.json"), serde_json::to_vec_pretty(&pairs)?)?;
    Ok(())
}
