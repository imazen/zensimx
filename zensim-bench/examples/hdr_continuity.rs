//! HC1 continuity check (benchmarks/e36_subset_test_2026-11-01.md, "HC1", step 2 reads): an SDR image shown as HDR
//! at the SDR white must score as its SDR self. Each 8-bit sRGB PNG is distorted in-process (box blur r=1 and r=2,
//! deterministic ±8 noise, and the identity pair), then every pair is scored with a bake three ways: the SDR path
//! (`BakeScorer::compute`), `compute_hdr` on linear nits = srgb_eotf(v) × white through the cube-root front end at
//! that white, and `compute_hdr` through the PU21 front end. Output TSV: bake, file, distortion, sdr, hdr_cuberoot,
//! hdr_pu21, gap_cuberoot, gap_pu21; a summary line per bake with the mean and max |gap|.
//!
//!   ZENSIM_FORMULA_REV=7 hdr_continuity --bake NAME=PATH [--bake ...] [--white 203] --out OUT.tsv IMG.png...
use std::path::{Path, PathBuf};

use zensim::feature_v2::{HdrEncoding, HdrFrontEnd};
use zensim::{BakeScorer, RgbSlice};

struct Linear {
    data: Vec<[f32; 4]>,
    w: usize,
    h: usize,
}

impl zensim::source::ImageSource for Linear {
    fn width(&self) -> usize {
        self.w
    }
    fn height(&self) -> usize {
        self.h
    }
    fn pixel_format(&self) -> zensim::source::PixelFormat {
        zensim::source::PixelFormat::LinearF32Rgba
    }
    fn row_bytes(&self, y: usize) -> &[u8] {
        bytemuck::cast_slice(&self.data[y * self.w..(y + 1) * self.w])
    }
    fn alpha_mode(&self) -> zensim::source::AlphaMode {
        zensim::source::AlphaMode::Opaque
    }
    fn is_hdr(&self) -> bool {
        true
    }
    fn color_primaries(&self) -> zensim::ColorPrimaries {
        zensim::ColorPrimaries::Srgb
    }
}

fn srgb_eotf(v: u8) -> f32 {
    let c = v as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn to_linear(rgb: &[[u8; 3]], w: usize, h: usize, white: f32) -> Linear {
    let lut: Vec<f32> = (0..=255u8).map(|v| srgb_eotf(v) * white).collect();
    Linear {
        data: rgb
            .iter()
            .map(|p| {
                [
                    lut[p[0] as usize],
                    lut[p[1] as usize],
                    lut[p[2] as usize],
                    1.0,
                ]
            })
            .collect(),
        w,
        h,
    }
}

fn decode_png8(path: &Path) -> Result<(Vec<[u8; 3]>, usize, usize), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {path:?}: {e}"))?;
    let out = zenpng::decode(
        &bytes,
        &zenpng::PngDecodeConfig::default(),
        &enough::Unstoppable,
    )
    .map_err(|e| format!("{path:?}: {e}"))?;
    if out.info.bit_depth != 8 {
        return Err(format!("{path:?}: 8-bit PNG required"));
    }
    let buf = out.pixels;
    let (w, h) = (buf.width() as usize, buf.height() as usize);
    let raw = buf.copy_to_contiguous_bytes();
    let channels = raw.len() / (w * h);
    if !matches!(channels, 3 | 4) {
        return Err(format!("{path:?}: RGB or RGBA required"));
    }
    Ok((
        raw.chunks_exact(channels)
            .map(|p| [p[0], p[1], p[2]])
            .collect(),
        w,
        h,
    ))
}

/// Separable box blur of radius `r`, edge-clamped, rounded to u8.
fn blur(rgb: &[[u8; 3]], w: usize, h: usize, r: usize) -> Vec<[u8; 3]> {
    let n = (2 * r + 1) as f32;
    let mut tmp = vec![[0f32; 3]; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0f32; 3];
            for dx in -(r as isize)..=(r as isize) {
                let xx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                for c in 0..3 {
                    acc[c] += rgb[y * w + xx][c] as f32;
                }
            }
            tmp[y * w + x] = [acc[0] / n, acc[1] / n, acc[2] / n];
        }
    }
    let mut out = vec![[0u8; 3]; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0f32; 3];
            for dy in -(r as isize)..=(r as isize) {
                let yy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                for c in 0..3 {
                    acc[c] += tmp[yy * w + x][c];
                }
            }
            out[y * w + x] = [
                (acc[0] / n).round().clamp(0.0, 255.0) as u8,
                (acc[1] / n).round().clamp(0.0, 255.0) as u8,
                (acc[2] / n).round().clamp(0.0, 255.0) as u8,
            ];
        }
    }
    out
}

/// Deterministic additive noise in [-amp, amp] (LCG), the same for every run.
fn noise(rgb: &[[u8; 3]], amp: i32, mut state: u64) -> Vec<[u8; 3]> {
    rgb.iter()
        .map(|p| {
            let mut q = [0u8; 3];
            for c in 0..3 {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let d = ((state >> 33) % (2 * amp as u64 + 1)) as i32 - amp;
                q[c] = (p[c] as i32 + d).clamp(0, 255) as u8;
            }
            q
        })
        .collect()
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut bakes: Vec<(String, PathBuf)> = Vec::new();
    let mut white = 203.0f32;
    let mut out: Option<PathBuf> = None;
    let mut files: Vec<PathBuf> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--bake" => {
                let (n, p) = args[i + 1].split_once('=').ok_or("--bake NAME=PATH")?;
                bakes.push((n.to_string(), PathBuf::from(p)));
                i += 2;
            }
            "--white" => {
                white = args[i + 1].parse().map_err(|_| "--white nits")?;
                i += 2;
            }
            "--out" => {
                out = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            f => {
                files.push(PathBuf::from(f));
                i += 1;
            }
        }
    }
    let out = out.ok_or("--out required")?;
    if bakes.is_empty() || files.is_empty() {
        return Err(
            "usage: hdr_continuity --bake NAME=PATH [--white 203] --out OUT.tsv IMG.png...".into(),
        );
    }
    let models: Vec<(String, zenpredict::Model)> = bakes
        .iter()
        .map(|(n, p)| {
            let b = std::fs::read(p).map_err(|e| format!("{p:?}: {e}"))?;
            Ok((
                n.clone(),
                zenpredict::Model::from_bytes(&b).map_err(|e| format!("{p:?}: {e:?}"))?,
            ))
        })
        .collect::<Result<_, String>>()?;
    let mut rows = vec![
        "bake\tfile\tdistortion\tsdr\thdr_cuberoot\thdr_pu21\tgap_cuberoot\tgap_pu21".to_string(),
    ];
    let mut summary = Vec::new();
    for (name, model) in &models {
        let mut sdr = BakeScorer::new(model)
            .map_err(|e| format!("{e:?}"))?
            .with_parallel(false);
        let mut cube = BakeScorer::new(model)
            .map_err(|e| format!("{e:?}"))?
            .with_parallel(false)
            .with_hdr_front_end(HdrFrontEnd::CubeRoot { unit_nits: white });
        let mut pu = BakeScorer::new(model)
            .map_err(|e| format!("{e:?}"))?
            .with_parallel(false);
        let (mut gaps_c, mut gaps_p) = (Vec::new(), Vec::new());
        for f in &files {
            let (rgb, w, h) = decode_png8(f)?;
            let ref_lin = to_linear(&rgb, w, h, white);
            let variants: [(&str, Vec<[u8; 3]>); 4] = [
                ("identity", rgb.clone()),
                ("blur1", blur(&rgb, w, h, 1)),
                ("blur2", blur(&rgb, w, h, 2)),
                ("noise8", noise(&rgb, 8, 0x5eed_1234_abcd_0001)),
            ];
            for (label, dist) in &variants {
                let s = sdr
                    .compute(&RgbSlice::new(&rgb, w, h), &RgbSlice::new(dist, w, h), None)
                    .map_err(|e| format!("sdr {f:?}: {e:?}"))?
                    .score();
                let dist_lin = to_linear(dist, w, h, white);
                let c = cube
                    .compute_hdr(&ref_lin, &dist_lin, HdrEncoding::Linear, None)
                    .map_err(|e| format!("cube-root {f:?}: {e:?}"))?;
                let p = pu
                    .compute_hdr(&ref_lin, &dist_lin, HdrEncoding::Linear, None)
                    .map_err(|e| format!("pu21 {f:?}: {e:?}"))?;
                gaps_c.push((c - s).abs());
                gaps_p.push((p - s).abs());
                rows.push(format!(
                    "{name}\t{}\t{label}\t{s:.4}\t{c:.4}\t{p:.4}\t{:.4}\t{:.4}",
                    f.file_name().unwrap().to_string_lossy(),
                    c - s,
                    p - s
                ));
            }
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let max = |v: &[f64]| v.iter().cloned().fold(0.0, f64::max);
        summary.push(format!(
            "# {name}: pairs {} white {white} cd/m2 | cube-root mean|gap| {:.4} max {:.4} | pu21 mean|gap| {:.4} max {:.4}",
            gaps_c.len(),
            mean(&gaps_c),
            max(&gaps_c),
            mean(&gaps_p),
            max(&gaps_p)
        ));
    }
    rows.extend(summary.iter().cloned());
    std::fs::write(&out, rows.join("\n") + "\n").map_err(|e| e.to_string())?;
    for s in &summary {
        println!("{s}");
    }
    Ok(())
}
