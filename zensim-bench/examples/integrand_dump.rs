//! Per-feature block integrands for supervising a bake's attribution map (E36 experiment 3, step 2;
//! `benchmarks/e36_subset_test_2026-11-01.md`).
//!
//! For each pair: [`BakeScorer::feature_block_integrands`] for every feature the bake reads, at `--block` source
//! pixels, the bake's own sensitivities and score, and the butteraugli diffmap's block sums and block maxima on the
//! same grid.
//!
//! ```text
//! integrand_dump --pairs PAIRS.tsv --out DIR --bake BAKE.bin [--block 8] [--window W --paint P --split S]
//! ```
//!
//! `PAIRS.tsv` has a header with `key`, `ref_path`, `dist_path` (map_dump's format). Per pair:
//! `DIR/<key>.integrands.f32` (row-major little-endian f32, one row per spatialized feature in the JSON's `ids`
//! order, `blocks_x * blocks_y` columns), `DIR/<key>.butter_sum.f32` and `DIR/<key>.butter_max.f32` (one value per
//! block), and one JSON object per pair on stdout. Without the paint options the scorer's served map modes apply.
//! Inputs are PNG files decoded by the `image` crate. No labels are read.

use std::io::Write;
use std::path::{Path, PathBuf};

use butteraugli::{ButteraugliParams, Img, RGB8, butteraugli};
use rayon::prelude::*;
use zensim::{BakeScorer, Fused944Session, RgbSlice};

fn write_f32(path: &Path, v: impl Iterator<Item = f32>) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("create output"));
    for x in v {
        f.write_all(&x.to_le_bytes()).expect("write output");
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut pairs, mut out, mut bake) = (None, None, None);
    let mut block = 8usize;
    let mut paint: Option<[String; 3]> = None;
    let mut modes = [String::from("box"), String::from("nearest"), String::from("repair")];
    while let Some(a) = args.next() {
        match a.as_str() {
            "--pairs" => pairs = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from),
            "--bake" => bake = args.next().map(PathBuf::from),
            "--block" => block = args.next().expect("--block N").parse().expect("block"),
            "--window" | "--paint" | "--split" => {
                let i = ["--window", "--paint", "--split"].iter().position(|f| *f == a).unwrap();
                modes[i] = args.next().expect("mode");
                paint = Some(modes.clone());
            }
            other => panic!("unknown argument {other}"),
        }
    }
    let (pairs, out) = (pairs.expect("--pairs"), out.expect("--out"));
    std::fs::create_dir_all(&out).expect("out dir");
    let bytes = std::fs::read(bake.expect("--bake")).expect("read bake");
    let model = zenpredict::Model::from_bytes(&bytes).expect("parse bake");
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(&pairs)
        .expect("pairs");
    let hdr = rdr.headers().expect("header").clone();
    let col = |n: &str| hdr.iter().position(|h| h == n).unwrap_or_else(|| panic!("{n} column"));
    let (ki, ri, di) = (col("key"), col("ref_path"), col("dist_path"));
    let rows: Vec<(String, PathBuf, PathBuf)> = rdr
        .records()
        .map(|r| {
            let r = r.expect("row");
            (r[ki].to_string(), PathBuf::from(&r[ri]), PathBuf::from(&r[di]))
        })
        .collect();
    let lines: Vec<String> = rows
        .par_iter()
        .map(|(key, reference, distorted)| {
            let rd = image::open(reference).expect("decode ref").to_rgb8();
            let dd = image::open(distorted).expect("decode dist").to_rgb8();
            let (w, h) = (rd.width() as usize, rd.height() as usize);
            assert_eq!((w, h), (dd.width() as usize, dd.height() as usize), "{key}: size mismatch");
            let rpx: &[[u8; 3]] = rd.as_raw().as_chunks::<3>().0;
            let dpx: &[[u8; 3]] = dd.as_raw().as_chunks::<3>().0;
            let (rs, ds) = (RgbSlice::new(rpx, w, h), RgbSlice::new(dpx, w, h));
            let mut scorer = BakeScorer::new(&model).expect("bake scorer").with_parallel(false);
            if let Some([window, paint, split]) = &paint {
                scorer = scorer.with_map_paint(window, paint, split).expect("paint modes");
            }
            let pre = scorer.precompute_reference(&rs).expect("reference");
            let mut session = Fused944Session::new();
            let served = scorer
                .compute_with_ref_and_attribution(&rs, &pre, &ds, None, &mut session, block)
                .expect("attribution");
            let sens = served.sensitivities().to_vec();
            let reads: Vec<usize> = scorer
                .consumed_feature_ids()
                .expect("read set")
                .into_iter()
                .map(usize::from)
                .collect();
            let mut fresh = Fused944Session::new();
            let integrands = scorer
                .feature_block_integrands(&rs, &pre, &ds, &mut fresh, block, &reads)
                .expect("integrands");
            write_f32(
                &out.join(format!("{key}.integrands.f32")),
                integrands.values.iter().flatten().map(|&v| v as f32),
            );
            let to_rgb8 = |px: &[[u8; 3]]| -> Vec<RGB8> { px.iter().map(|q| RGB8::new(q[0], q[1], q[2])).collect() };
            let (r8, d8) = (to_rgb8(rpx), to_rgb8(dpx));
            let b = butteraugli(
                Img::new(&r8[..], w, h),
                Img::new(&d8[..], w, h),
                &ButteraugliParams::default().with_compute_diffmap(true),
            )
            .expect("butteraugli");
            let bm = b.diffmap.expect("diffmap requested");
            let (bx, by) = (integrands.blocks_x, integrands.blocks_y);
            let (mut sum, mut max) = (vec![0f64; bx * by], vec![0f32; bx * by]);
            for (y, row) in bm.buf().chunks(w).enumerate() {
                for (x, &v) in row.iter().enumerate() {
                    let i = (y / block) * bx + x / block;
                    sum[i] += v as f64;
                    max[i] = max[i].max(v);
                }
            }
            write_f32(&out.join(format!("{key}.butter_sum.f32")), sum.iter().map(|&v| v as f32));
            write_f32(&out.join(format!("{key}.butter_max.f32")), max.iter().copied());
            serde_json::json!({
                "key": key, "width": w, "height": h, "block": block, "blocks": [bx, by],
                "ids": integrands.ids, "unsupported": integrands.unsupported,
                "sensitivities": integrands.ids.iter().map(|&i| sens[i]).collect::<Vec<_>>(),
                "score": served.result().score(), "butteraugli": b.score,
            })
            .to_string()
        })
        .collect();
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    for l in lines {
        writeln!(lock, "{l}").expect("stdout");
    }
}
