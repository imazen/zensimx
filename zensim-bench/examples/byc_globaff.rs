//! Global affine-change descriptor sidecar for the by-construction exploration
//! (`benchmarks/byconstruction_2026-10-22.md`). Research only: not a registered feature family, and nothing reads it at
//! serving time.
//!
//! For each pair, both images are decoded by the shared decode owner (legacy RGB8, as the feature extractor does),
//! converted to positive XYB with zensim's own conversion, and reduced to the walk's 4-scale pyramid with zensim's own
//! 2× downscale (dimensions halve with floor; a level below 8 px is not built). At every scale and channel a least
//! squares fit `dst ≈ a·src + b` is taken over the whole plane, and two numbers are written:
//!
//! - `share`: the fraction of the mean squared error `E[(src − dst)²]` the fit explains, in `[0, 1]`
//!   (0 when the error is zero). An offset, a contrast change or a saturation change is explained almost entirely;
//!   noise and blur mostly are not.
//! - `explained`: that explained mean squared error, `E[(src − dst)²] − min_{a,b} E[(dst − a·src − b)²]`.
//!
//! Moments are two-pass f64 sums. Usage:
//!
//! ```text
//! byc_globaff --pairs PAIRS.tsv --out OUT.csv
//! ```
//!
//! `PAIRS.tsv` has columns `ref_path`, `dist_path`, `row_id` (others ignored). The output has `row_id` then
//! `g0..g23`, index `2·(3·scale + channel) + {0: share, 1: explained}`, channels X, Y, B; a missing scale is NaN.

use std::path::{Path, PathBuf};

use rayon::prelude::*;

#[path = "shared/zen_decode.rs"]
mod zen_decode;

const SCALES: usize = 4;

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

/// `(share, explained)` of the global affine fit of `d` on `s`.
fn affine(s: &[f32], d: &[f32]) -> (f64, f64) {
    let n = s.len() as f64;
    let (ms, md) = (
        s.iter().map(|&v| v as f64).sum::<f64>() / n,
        d.iter().map(|&v| v as f64).sum::<f64>() / n,
    );
    let (mut vss, mut vdd, mut vsd) = (0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in s.iter().zip(d) {
        let (a, b) = (a as f64 - ms, b as f64 - md);
        vss += a * a;
        vdd += b * b;
        vsd += a * b;
    }
    let (vss, vdd, vsd) = (vss / n, vdd / n, vsd / n);
    let mse = (ms - md) * (ms - md) + vss + vdd - 2.0 * vsd;
    let resid = if vss > 1e-12 {
        (vdd - vsd * vsd / vss).max(0.0)
    } else {
        vdd
    };
    let explained = (mse - resid).max(0.0);
    let share = if mse > 1e-12 {
        (explained / mse).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (share, explained)
}

fn descriptor(reference: &Path, distorted: &Path) -> Result<[f64; 2 * 3 * SCALES], String> {
    let (w, h, mut src) = planes(reference)?;
    let (w2, h2, mut dst) = planes(distorted)?;
    if (w, h) != (w2, h2) {
        return Err(format!(
            "dimension mismatch {w}x{h} vs {w2}x{h2}: {}",
            distorted.display()
        ));
    }
    let mut out = [f64::NAN; 2 * 3 * SCALES];
    let (mut cw, mut ch) = (w, h);
    for scale in 0..SCALES {
        if cw < 8 || ch < 8 {
            break;
        }
        for c in 0..3 {
            let (share, explained) = affine(&src[c], &dst[c]);
            out[2 * (3 * scale + c)] = share;
            out[2 * (3 * scale + c) + 1] = explained;
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
    let results: Vec<(i64, [f64; 2 * 3 * SCALES])> = rows
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
    header.extend((0..2 * 3 * SCALES).map(|i| format!("g{i}")));
    w.write_record(&header).expect("header");
    for (id, v) in results {
        let mut rec = vec![id.to_string()];
        rec.extend(v.iter().map(|x| format!("{x:e}")));
        w.write_record(&rec).expect("row");
    }
    w.flush().expect("flush");
}
