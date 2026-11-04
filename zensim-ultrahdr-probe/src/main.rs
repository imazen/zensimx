//! Ultra HDR same-base probe (owner question, 2026-11-04): "how well does an Ultra HDR image compressed
//! (keeping the gain map intact) score when compared in SDR vs HDR (applied gain map) with the same base
//! SDR encode?"
//!
//! For every input container and every quality on the ladder: decode the base, re-encode it with zenjpeg
//! (4:2:0) at that quality, reassemble the container with the ORIGINAL gain map JPEG and metadata, decode
//! both containers the same way, and score
//! - SDR: original base vs re-encoded base (zensim `F`, SSIMULACRA2, butteraugli 3-norm and max), and
//! - HDR: the gain map applied to both bases at a display boost (zensim `F` through `compute_hdr`,
//!   SSIMULACRA2's native absolute-light path, butteraugli's linear path at the boost's peak),
//!   at a 4× boost (812 cd/m² over the BT.2408 203 cd/m² SDR white) and at the gain map's full boost, and
//! - SDR fed as HDR (owner question): display boost 1.0, so the gain map contributes nothing and the base's own
//!   light is scored through the HDR paths, once with SDR white at 203 cd/m² and once at 100 cd/m².
//!
//! usage: zensim-ultrahdr-probe <out.tsv> <workers> <file.jpg>...   (QS=35,45,... overrides the ladder)
use std::error::Error;
use std::fmt::Display;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;
use ultrahdr_rs::{Decoder, Encoder};
use zensim::source::{AlphaMode, ImageSource, PixelFormat};
use zensim::{HdrEncoding, RgbSlice, Zensim, ZensimProfile};

/// BT.2408 diffuse white: ultrahdr-rs's linear 1.0 (the convention zenmetrics-cli's HDR path uses too).
const SDR_WHITE_NITS: f32 = 203.0;
const DISPLAY_BOOST: f32 = 4.0;

type R<T> = Result<T, Box<dyn Error + Send + Sync>>;

fn s<E: Display>(e: E) -> Box<dyn Error + Send + Sync> {
    e.to_string().into()
}

/// Absolute-light RGBA f32 for zensim's `compute_hdr` (`LinearF32Rgba`, opaque, sRGB primaries).
struct NitsRgba {
    data: Vec<[f32; 4]>,
    w: usize,
    h: usize,
}

impl ImageSource for NitsRgba {
    fn width(&self) -> usize {
        self.w
    }
    fn height(&self) -> usize {
        self.h
    }
    fn pixel_format(&self) -> PixelFormat {
        PixelFormat::LinearF32Rgba
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }
    fn row_bytes(&self, y: usize) -> &[u8] {
        bytemuck::cast_slice(&self.data[y * self.w..(y + 1) * self.w])
    }
}

/// One decoded container: the base as packed RGB8 and, per boost, the gain-mapped light in cd/m²
/// as packed RGB f32 (the reference metrics) and RGBA f32 (zensim).
struct Rendered {
    w: usize,
    h: usize,
    rgb8: Vec<u8>,
    hdr: Vec<(HdrArm, Vec<f32>, NitsRgba)>,
}

/// One HDR scoring arm: the display boost handed to the gain map and the luminance of SDR white (ultrahdr-rs's linear 1.0).
#[derive(Clone, Copy)]
struct HdrArm {
    boost: f32,
    white_nits: f32,
}

fn render(bytes: &[u8], arms: &[HdrArm]) -> R<Rendered> {
    let dec = Decoder::new(bytes).map_err(s)?;
    if !dec.is_ultrahdr() {
        return Err("not an Ultra HDR container".into());
    }
    let sdr = dec.decode_sdr().map_err(s)?;
    let sl = sdr.as_slice();
    let (w, h, stride) = (sl.width() as usize, sl.rows() as usize, sl.stride());
    let bpp = sl.descriptor().format.bytes_per_pixel();
    let bytes_sdr = sl.as_strided_bytes();
    let mut rgb8 = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        let row = &bytes_sdr[y * stride..y * stride + w * bpp];
        for px in row.chunks_exact(bpp) {
            rgb8.extend_from_slice(&px[..3]);
        }
    }
    let mut hdr = Vec::new();
    for &arm in arms {
        let buf = dec.decode_hdr(arm.boost).map_err(s)?;
        let sl = buf.as_slice();
        let (hw, hh, hstride) = (sl.width() as usize, sl.rows() as usize, sl.stride());
        if hw != w || hh != h {
            return Err(format!("HDR render {hw}x{hh} differs from the base {w}x{h}").into());
        }
        let hbpp = sl.descriptor().format.bytes_per_pixel();
        if hbpp != 16 {
            return Err(format!("expected RGBA f32 from decode_hdr, got {hbpp} bytes per pixel").into());
        }
        let raw = sl.as_strided_bytes();
        let mut rgb = Vec::with_capacity(w * h * 3);
        let mut rgba = Vec::with_capacity(w * h);
        for y in 0..h {
            let row: &[[f32; 4]] = bytemuck::cast_slice(&raw[y * hstride..y * hstride + w * 16]);
            for p in row {
                let (r, g, b) = (p[0] * arm.white_nits, p[1] * arm.white_nits, p[2] * arm.white_nits);
                rgb.extend_from_slice(&[r, g, b]);
                rgba.push([r, g, b, 1.0]);
            }
        }
        hdr.push((arm, rgb, NitsRgba { data: rgba, w, h }));
    }
    Ok(Rendered { w, h, rgb8, hdr })
}

fn ssim2_sdr(a: &[u8], b: &[u8], w: usize, h: usize) -> R<f64> {
    let d = fast_ssim2::PixelDescriptor::RGB8_SRGB;
    let sa = fast_ssim2::PixelSlice::new(a, w as u32, h as u32, w * 3, d).map_err(s)?;
    let sb = fast_ssim2::PixelSlice::new(b, w as u32, h as u32, w * 3, d).map_err(s)?;
    fast_ssim2::compute_ssimulacra2(&sa, &sb).map_err(s)
}

/// SSIMULACRA2's native HDR path: linear f32 in cd/m² with a 1 cd/m² diffuse white (the buffer's unit is one nit).
fn ssim2_hdr(a: &[f32], b: &[f32], w: usize, h: usize) -> R<f64> {
    let d = fast_ssim2::PixelDescriptor::RGBF32_LINEAR;
    let white = zenpixels::DiffuseWhite::new(1.0);
    let sa = fast_ssim2::PixelSlice::new(bytemuck::cast_slice(a), w as u32, h as u32, w * 12, d)
        .map_err(s)?
        .with_diffuse_white(white);
    let sb = fast_ssim2::PixelSlice::new(bytemuck::cast_slice(b), w as u32, h as u32, w * 12, d)
        .map_err(s)?
        .with_diffuse_white(white);
    fast_ssim2::compute_ssimulacra2(&sa, &sb).map_err(s)
}

fn display_relative(nits: &[f32], peak: f32) -> Vec<f32> {
    let inv = 1.0 / peak;
    nits.iter().map(|&v| (v * inv).clamp(0.0, 1.0)).collect()
}

fn encode_base(rgb8: &[u8], w: usize, h: usize, q: u32) -> R<Vec<u8>> {
    use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Unstoppable};
    let cfg = EncoderConfig::ycbcr(q as f32, ChromaSubsampling::Quarter);
    let mut enc = cfg
        .encode_from_bytes(w as u32, h as u32, PixelLayout::Rgb8Srgb)
        .map_err(s)?;
    enc.push_packed(rgb8, Unstoppable).map_err(s)?;
    enc.finish().map_err(s)
}

fn probe(path: &Path, qs: &[u32]) -> R<Vec<String>> {
    let t0 = Instant::now();
    let bytes = std::fs::read(path)?;
    let dec = Decoder::new(&bytes).map_err(s)?;
    if !dec.is_ultrahdr() {
        return Err(format!("{}: not an Ultra HDR container", path.display()).into());
    }
    let meta = dec.metadata().cloned().ok_or("missing gain-map metadata")?;
    let gm_jpeg = dec.gainmap_jpeg().ok_or("missing gain-map JPEG")?.to_vec();
    let base_jpeg_len = dec.primary_jpeg().ok_or("missing primary JPEG")?.len();
    let gm = dec.decode_gainmap().map_err(s)?;
    let full = ultrahdr_core::full_reconstruction_boost(&meta);
    let full = if full.is_finite() && full >= 1.0 { full } else { 1.0 };
    let arms = [
        HdrArm { boost: 1.0, white_nits: SDR_WHITE_NITS },
        HdrArm { boost: 1.0, white_nits: 100.0 },
        HdrArm { boost: DISPLAY_BOOST, white_nits: SDR_WHITE_NITS },
        HdrArm { boost: full, white_nits: SDR_WHITE_NITS },
    ];
    let orig = render(&bytes, &arms)?;
    let (w, h) = (orig.w, orig.h);
    let z = Zensim::new(ZensimProfile::F).with_parallel(false);
    let butter_sdr = butteraugli::ButteraugliReference::new(
        &orig.rgb8,
        w,
        h,
        butteraugli::ButteraugliParams::default(),
    )
    .map_err(s)?;
    let mut butter_hdr = Vec::new();
    for (arm, rgb, _) in &orig.hdr {
        let peak = arm.boost * arm.white_nits;
        let rel = display_relative(rgb, peak);
        butter_hdr.push(
            butteraugli::ButteraugliReference::new_linear(
                &rel,
                w,
                h,
                butteraugli::ButteraugliParams::default().with_intensity_target(peak),
            )
            .map_err(s)?,
        );
    }
    let src_sdr = RgbSlice::new(bytemuck::cast_slice(&orig.rgb8), w, h);
    let mut rows = Vec::new();
    for &q in qs {
        let base_q = encode_base(&orig.rgb8, w, h, q)?;
        let mut ue = Encoder::new();
        ue.set_base_jpeg(base_q.clone());
        ue.set_existing_gainmap_jpeg(gm_jpeg.clone(), meta.clone());
        let container = ue.encode_from_jpegs().map_err(s)?;
        let cand = render(&container, &arms)?;
        if (cand.w, cand.h) != (w, h) {
            return Err("re-encoded base changed size".into());
        }
        let dst_sdr = RgbSlice::new(bytemuck::cast_slice(&cand.rgb8), w, h);
        let zs = z.compute(&src_sdr, &dst_sdr).map_err(s)?.score();
        let s2 = ssim2_sdr(&orig.rgb8, &cand.rgb8, w, h)?;
        let bs = butter_sdr.compare(&cand.rgb8).map_err(s)?;
        let (b3s, bms) = (bs.pnorm(3.0).unwrap_or(f64::NAN), bs.max_norm());
        let mut zh_first = f64::NAN;
        for (i, (arm, rgb_q, rgba_q)) in cand.hdr.iter().enumerate() {
            let (_, rgb_o, rgba_o) = &orig.hdr[i];
            let (boost, white) = (arm.boost, arm.white_nits);
            let peak = boost * white;
            let zh = z.compute_hdr(rgba_o, rgba_q, HdrEncoding::Linear).map_err(s)?.score();
            if i == 0 {
                zh_first = zh;
            }
            let s2h = ssim2_hdr(rgb_o, rgb_q, w, h)?;
            let bh = butter_hdr[i]
                .compare_linear(&display_relative(rgb_q, peak))
                .map_err(s)?;
            rows.push(format!(
                "{}\t{w}\t{h}\t{q}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{boost:.4}\t{white:.1}\t{peak:.1}\t{zs:.4}\t{s2:.4}\t{b3s:.5}\t{bms:.5}\t{zh:.4}\t{s2h:.4}\t{:.5}\t{:.5}",
                path.display(),
                base_q.len(),
                container.len(),
                bytes.len(),
                base_jpeg_len,
                gm_jpeg.len(),
                gm.width,
                gm.height,
                gm.channels,
                full,
                bh.pnorm(3.0).unwrap_or(f64::NAN),
                bh.max_norm(),
            ));
        }
        eprintln!(
            "{} q{q}: sdr zensim {zs:.2} ssim2 {s2:.2} | sdr-as-hdr(203) zensim {zh_first:.2} | {:.0}s",
            path.file_name().unwrap_or_default().to_string_lossy(),
            t0.elapsed().as_secs_f64()
        );
    }
    Ok(rows)
}

const HEADER: &str = "file\twidth\theight\tq\tbase_q_bytes\tcontainer_q_bytes\torig_container_bytes\torig_base_bytes\tgainmap_bytes\tgainmap_w\tgainmap_h\tgainmap_channels\tfull_boost\tboost\twhite_nits\tpeak_nits\tzensim_sdr\tssim2_sdr\tbutter3_sdr\tbuttermax_sdr\tzensim_hdr\tssim2_hdr\tbutter3_hdr\tbuttermax_hdr";

fn main() -> R<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: zensim-ultrahdr-probe <out.tsv> <workers> <file.jpg>...   (QS=35,45,... overrides the ladder)");
        std::process::exit(2);
    }
    let out = PathBuf::from(&args[0]);
    let workers: usize = args[1].parse()?;
    let files: Vec<PathBuf> = args[2..].iter().map(PathBuf::from).collect();
    let qs: Vec<u32> = std::env::var("QS")
        .unwrap_or_else(|_| "35,45,55,65,75,82,88,94".into())
        .split(',')
        .map(|v| v.trim().parse::<u32>())
        .collect::<Result<_, _>>()?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(workers).build()?;
    let results: Vec<(PathBuf, R<Vec<String>>)> =
        pool.install(|| files.par_iter().map(|f| (f.clone(), probe(f, &qs))).collect());
    let mut file = std::fs::File::create(&out)?;
    writeln!(file, "{HEADER}")?;
    let mut failures = 0;
    for (f, r) in &results {
        match r {
            Ok(rows) => {
                for row in rows {
                    writeln!(file, "{row}")?;
                }
            }
            Err(e) => {
                failures += 1;
                eprintln!("FAILED {}: {e}", f.display());
            }
        }
    }
    eprintln!(
        "wrote {} ({} files, {} failed, ladder {:?})",
        out.display(),
        results.len() - failures,
        failures,
        qs
    );
    if failures > 0 {
        std::process::exit(1);
    }
    Ok(())
}
