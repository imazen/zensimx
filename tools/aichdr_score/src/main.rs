//! Score the AIC-HDR2025 stimuli (jpeg-aic/AIC-HDR2025, Rec. 2100 PQ, 16-bit RGB PNG)
//! with the published `zensim` crate's `Zensim::compute_hdr`.
//!
//! Each distorted image is scored against its source's `*_0ref_00.png`. The PNG
//! samples are passed as PQ code values (`HdrEncoding::Pq`) with BT.2020 primaries;
//! the dataset's ICC profile (Rec2100PQ.icc) declares exactly that encoding.
//!
//! Usage: aichdr-score <dataset_root> <profile:A> <peak_nits>... > scores.tsv
//! Output: `image	profile	peak_nits	score	raw_distance` per distorted image.

use std::path::{Path, PathBuf};

use zensim::{
    AlphaMode, ColorPrimaries, HdrEncoding, PixelFormat, StridedBytes, Zensim, ZensimProfile,
};

struct Rgba16 {
    w: usize,
    h: usize,
    bytes: Vec<u8>,
}

fn load(path: &Path) -> Result<Rgba16, String> {
    let data = std::fs::read(path).map_err(|e| format!("{path:?}: {e}"))?;
    let decoded = zenpng::decode(
        &data,
        &zenpng::PngDecodeConfig::default(),
        &enough::Unstoppable,
    )
    .map_err(|e| format!("{path:?}: {e}"))?;
    let px = decoded.pixels;
    let d = px.descriptor();
    if d.channel_type() != zenpixels::ChannelType::U16
        || d.layout() != zenpixels::ChannelLayout::Rgb
    {
        return Err(format!("{path:?}: expected RGB16, got {d:?}"));
    }
    let (w, h) = (px.width() as usize, px.height() as usize);
    let view = px.as_slice();
    let mut bytes = Vec::with_capacity(w * h * 8);
    let opaque = u16::MAX.to_ne_bytes();
    for y in 0..h {
        let row = view.row(y as u32);
        let (chunks, rest) = row.as_chunks::<6>();
        if chunks.len() != w || !rest.is_empty() {
            return Err(format!("{path:?}: row {y} is not {w} RGB16 pixels"));
        }
        for p in chunks {
            bytes.extend_from_slice(p);
            bytes.extend_from_slice(&opaque);
        }
    }
    Ok(Rgba16 { w, h, bytes })
}

fn source(img: &Rgba16) -> Result<StridedBytes<'_>, String> {
    Ok(StridedBytes::try_with_alpha_mode(
        &img.bytes,
        img.w,
        img.h,
        img.w * 8,
        PixelFormat::Srgb16Rgba,
        AlphaMode::Opaque,
    )
    .map_err(|e| format!("{e:?}"))?
    .with_color_primaries(ColorPrimaries::Bt2020))
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        return Err("usage: aichdr-score <dataset_root> A <peak_nits>...".into());
    }
    let root = PathBuf::from(&args[0]);
    let profile = match args[1].as_str() {
        "A" => ZensimProfile::A,
        other => return Err(format!("unsupported profile {other}")),
    };
    let peaks: Vec<f32> = args[2..]
        .iter()
        .map(|s| s.parse::<f32>().map_err(|e| format!("{s}: {e}")))
        .collect::<Result<_, _>>()?;
    let z = Zensim::new(profile);

    let mut names: Vec<String> = std::fs::read_dir(root.join("images"))
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .filter(|n| n.ends_with(".png") && !n.contains("_0ref_"))
        .collect();
    names.sort();
    if names.len() != 100 {
        return Err(format!("expected 100 distorted images, found {}", names.len()));
    }
    println!("image\tprofile\tpeak_nits\tscore\traw_distance");
    let mut cached: Option<(String, Rgba16)> = None;
    for name in &names {
        let sid = &name["HDR_PTC_".len().."HDR_PTC_00001".len()];
        let ref_name = format!("HDR_PTC_{sid}_0ref_00.png");
        if cached.as_ref().map(|(n, _)| n != &ref_name).unwrap_or(true) {
            cached = Some((ref_name.clone(), load(&root.join("images").join(&ref_name))?));
        }
        let reference = &cached.as_ref().unwrap().1;
        let distorted = load(&root.join("images").join(name))?;
        if (distorted.w, distorted.h) != (reference.w, reference.h) {
            return Err(format!("{name}: size mismatch"));
        }
        for &peak in &peaks {
            let r = z
                .compute_hdr(
                    &source(reference)?,
                    &source(&distorted)?,
                    HdrEncoding::Pq { peak_nits: peak },
                )
                .map_err(|e| format!("{name}: {e:?}"))?;
            println!(
                "{name}\t{}\t{peak}\t{:.10}\t{:.10}",
                args[1],
                r.score(),
                r.raw_distance()
            );
        }
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("aichdr-score: {e}");
        std::process::exit(1);
    }
}
