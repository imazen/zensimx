//! Orientation-split damage sidecar for the by-construction exploration (`benchmarks/byconstruction_2026-10-22.md`,
//! "Orientation-selective bands"). Research only: not a registered feature family, and nothing reads it at serving
//! time.
//!
//! For each pair, both images are decoded by the shared decode owner (legacy RGB8, as the feature extractor does),
//! converted to positive XYB with zensim's own conversion, and reduced to the walk's 4-scale pyramid with zensim's own
//! 2× downscale (dimensions halve with floor; a level below 8 px is not built). At every scale and channel the three
//! Haar detail bands of each 2×2 block (horizontal edges: top minus bottom rows; vertical edges: left minus right
//! columns; diagonal) are taken of the error `dst − src` (the bands are linear, so this is the band difference), and the
//! root mean square of each band is written: a non-negative damage value per orientation. Sums are f64.
//!
//! ```text
//! byc_orient --pairs PAIRS.tsv --out OUT.csv
//! ```
//!
//! `PAIRS.tsv` has columns `ref_path`, `dist_path`, `row_id` (others ignored). The output has `row_id` then
//! `o0..o35`, index `3·(3·scale + channel) + {0: horizontal, 1: vertical, 2: diagonal}`, channels X, Y, B; a missing
//! scale is NaN.

use std::path::{Path, PathBuf};

use rayon::prelude::*;

#[path = "shared/zen_decode.rs"]
mod zen_decode;

const SCALES: usize = 4;
const N: usize = 3 * 3 * SCALES;

fn planes(path: &Path) -> Result<(usize, usize, [Vec<f32>; 3]), String> {
    let img = zen_decode::decode_rgb8_path(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (w, h) = (img.width as usize, img.height as usize);
    let px: &[[u8; 3]] = img.pixels.as_chunks::<3>().0;
    let mut out = [
        vec![0.0f32; w * h],
        vec![0.0f32; w * h],
        vec![0.0f32; w * h],
    ];
    let [x, y, b] = &mut out;
    zensim::__bench_stages::srgb_to_positive_xyb_planar_into(px, x, y, b);
    Ok((w, h, out))
}

/// RMS of the horizontal, vertical and diagonal Haar bands of `d − s` over the plane's whole 2×2 blocks.
fn orient(s: &[f32], d: &[f32], w: usize, h: usize) -> [f64; 3] {
    let mut acc = [0.0f64; 3];
    for y in (0..h / 2 * 2).step_by(2) {
        for x in (0..w / 2 * 2).step_by(2) {
            let e = |dy: usize, dx: usize| {
                let i = (y + dy) * w + x + dx;
                f64::from(d[i]) - f64::from(s[i])
            };
            let (p, q, r, t) = (e(0, 0), e(0, 1), e(1, 0), e(1, 1));
            let band = [
                0.25 * (p + q - r - t),
                0.25 * (p - q + r - t),
                0.25 * (p - q - r + t),
            ];
            for (a, v) in acc.iter_mut().zip(band) {
                *a += v * v;
            }
        }
    }
    let blocks = ((w / 2) * (h / 2)).max(1) as f64;
    acc.map(|a| (a / blocks).sqrt())
}

fn descriptor(reference: &Path, distorted: &Path) -> Result<[f64; N], String> {
    let (w, h, mut src) = planes(reference)?;
    let (w2, h2, mut dst) = planes(distorted)?;
    if (w, h) != (w2, h2) {
        return Err(format!(
            "dimension mismatch {w}x{h} vs {w2}x{h2}: {}",
            distorted.display()
        ));
    }
    let mut out = [f64::NAN; N];
    let (mut cw, mut ch) = (w, h);
    for scale in 0..SCALES {
        if cw < 8 || ch < 8 {
            break;
        }
        for c in 0..3 {
            let v = orient(&src[c], &dst[c], cw, ch);
            out[3 * (3 * scale + c)..3 * (3 * scale + c) + 3].copy_from_slice(&v);
        }
        let (nw, nh) = (cw / 2, ch / 2);
        if scale + 1 < SCALES && nw >= 8 && nh >= 8 {
            for planes in [&mut src, &mut dst] {
                for p in planes.iter_mut() {
                    let mut next = vec![0.0f32; nw * nh];
                    zensim::__bench_stages::downscale_2x_into(p, cw, &mut next, nw, nh);
                    *p = next;
                }
            }
        }
        (cw, ch) = (nw, nh);
    }
    Ok(out)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut pairs, mut out) = (None, None);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--pairs" => pairs = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from),
            other => panic!("unknown argument {other}"),
        }
    }
    let (pairs, out) = (
        pairs.expect("--pairs PAIRS.tsv"),
        out.expect("--out OUT.csv"),
    );
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(&pairs)
        .expect("open pairs");
    let headers = rdr.headers().expect("headers").clone();
    let col = |n: &str| {
        headers
            .iter()
            .position(|h| h == n)
            .unwrap_or_else(|| panic!("{n} column"))
    };
    let (ri, di, ii) = (col("ref_path"), col("dist_path"), col("row_id"));
    let rows: Vec<(i64, PathBuf, PathBuf)> = rdr
        .records()
        .map(|r| {
            let r = r.expect("record");
            (
                r[ii].parse().expect("row_id"),
                PathBuf::from(&r[ri]),
                PathBuf::from(&r[di]),
            )
        })
        .collect();
    let results: Vec<(i64, [f64; N])> = rows
        .par_iter()
        .map(|(id, r, d)| {
            (
                *id,
                descriptor(r, d).unwrap_or_else(|e| panic!("row {id}: {e}")),
            )
        })
        .collect();
    let mut w = csv::Writer::from_path(&out).expect("create output");
    let mut header = vec!["row_id".to_string()];
    header.extend((0..N).map(|i| format!("o{i}")));
    w.write_record(&header).expect("header");
    for (id, v) in results {
        let mut rec = vec![id.to_string()];
        rec.extend(v.iter().map(|x| format!("{x:e}")));
        w.write_record(&rec).expect("row");
    }
    w.flush().expect("flush");
}
