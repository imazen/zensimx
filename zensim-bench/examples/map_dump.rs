//! Spatial maps for the map-quality study (by-construction exploration,
//! `benchmarks/byconstruction_2026-10-22.md`, "Maps").
//!
//! For each pair: a candidate bake's attribution map
//! (`BakeScorer::compute_with_ref_and_attribution`, under the process's
//! `ZENSIM_MAP_WINDOW` / `ZENSIM_MAP_PAINT` research modes), and on request
//! the judges — the butteraugli diffmap, the SSIMULACRA2 score-mass field (a
//! sum-preserving split of fast-ssim2 0.8.2's pooled terms, replicated from the
//! 2026-09-20 steering study's `ssim2_field`) and, with the `dvifm-judge`
//! feature, DVIFM 0.3's painted map — plus profile B's basic-plane diffmap
//! (`--legacy`, what the zenmetrics workbench shows for zensim B).
//!
//! ```text
//! map_dump --pairs PAIRS.tsv --out DIR --bake BAKE.bin --tag NAME [--bin 1] [--window box|center|guided]
//!          [--paint nearest|guided] [--split repair|exact] [--guide sq|abs] [--smooth SIGMA] [--legacy] [--judges]
//! ```
//!
//! `PAIRS.tsv` has a header with `key`, `ref_path`, `dist_path`. Each map is
//! written to `DIR/<key>.<name>.f32` (row-major little-endian f32: `w × h`, or
//! the candidate's `ceil(w/bin) × ceil(h/bin)` bin grid), and one JSON object
//! per pair goes to stdout. Inputs are PNG files decoded by the `image` crate
//! (lossless: any conforming decoder gives the same pixels). No labels are read.

use std::io::Write;
use std::path::{Path, PathBuf};

use butteraugli::{ButteraugliParams, Img, RGB8, butteraugli};
use fast_ssim2::{Blur, Ssimulacra2Reference};
use imgref::ImgRef;
use rayon::prelude::*;
use zensim::{BakeScorer, Fused944Session, RgbSlice, Zensim, ZensimProfile};

/// fast-ssim2 0.8.2's `WEIGHT` table (the published ssimulacra2/libjxl
/// weights), walked linearly in (channel, scale, norm, term) order as
/// `Msssim::score` walks it.
const SSIM2_WEIGHT: [f64; 108] = [
    0.0,
    0.000_737_660_670_740_658_6,
    0.0,
    0.0,
    0.000_779_348_168_286_730_9,
    0.0,
    0.0,
    0.000_437_115_573_010_737_9,
    0.0,
    1.104_172_642_665_734_6,
    0.000_662_848_341_292_71,
    0.000_152_316_327_837_187_52,
    0.0,
    0.001_640_643_745_659_975_4,
    0.0,
    1.842_245_552_053_929_8,
    11.441_172_603_757_666,
    0.0,
    0.000_798_910_943_601_516_3,
    0.000_176_816_438_078_653,
    0.0,
    1.878_759_497_954_638_7,
    10.949_069_906_051_42,
    0.0,
    0.000_728_934_699_150_807_2,
    0.967_793_708_062_683_3,
    0.0,
    0.000_140_034_242_854_358_84,
    0.998_176_697_785_496_7,
    0.000_319_497_559_344_350_53,
    0.000_455_099_211_379_206_3,
    0.0,
    0.0,
    0.001_364_876_616_324_339_8,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    7.466_890_328_078_848,
    0.0,
    17.445_833_984_131_262,
    0.000_623_560_163_404_146_6,
    0.0,
    0.0,
    6.683_678_146_179_332,
    0.000_377_244_079_796_112_96,
    1.027_889_937_768_264,
    225.205_153_008_492_74,
    0.0,
    0.0,
    19.213_238_186_143_016,
    0.001_140_152_458_661_836_1,
    0.001_237_755_635_509_985,
    176.393_175_984_506_94,
    0.0,
    0.0,
    24.433_009_998_704_76,
    0.285_208_026_121_177_57,
    0.000_448_543_692_383_340_8,
    0.0,
    0.0,
    0.0,
    34.779_063_444_837_72,
    44.835_625_328_877_896,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.000_868_055_657_329_169_8,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.000_531_319_187_435_874_7,
    0.0,
    0.000_165_338_141_613_791_12,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.000_417_917_180_325_133_6,
    0.001_729_082_823_472_283_3,
    0.0,
    0.002_082_700_584_663_643_7,
    0.0,
    0.0,
    8.826_982_764_996_862,
    23.192_433_439_989_26,
    0.0,
    95.108_049_881_108_6,
    0.986_397_803_440_068_2,
    0.983_438_279_246_535_3,
    0.001_228_640_504_827_849_3,
    171.266_725_589_730_7,
    0.980_785_887_243_537_9,
    0.0,
    0.0,
    0.0,
    0.000_513_006_458_899_067_9,
    0.0,
    0.000_108_540_578_584_115_37,
];
const SSIM2_C2: f64 = 0.0009;

/// The SSIMULACRA2 score-mass density at source resolution: every pooled
/// term's sum-preserving per-pixel split (L1 terms `w·x/N`, L4 terms
/// `w·x⁴/(N·r4³)`), each scale's mass spread over its source footprint.
fn ssim2_field(r: ImgRef<'_, [u8; 3]>, d: ImgRef<'_, [u8; 3]>) -> Result<(Vec<f64>, f64), String> {
    let r1 = Ssimulacra2Reference::new(r).map_err(|e| format!("ssim2 ref: {e}"))?;
    let r2 = Ssimulacra2Reference::new(d).map_err(|e| format!("ssim2 dist: {e}"))?;
    let ns = r1.num_scales();
    let (w, h) = (r.width(), r.height());
    let mut density = vec![0.0f64; w * h];
    let mut blur = Blur::new(w, h);
    let mut mul = [
        vec![0.0f32; w * h],
        vec![0.0f32; w * h],
        vec![0.0f32; w * h],
    ];
    let mut raw = 0.0f64;
    let mut wi = 0usize;
    let mut terms: Vec<[f64; 18]> = Vec::new();
    let mut sev_scales: Vec<(usize, usize, Vec<f64>)> = Vec::new();
    for s in 0..ns {
        let v1 = r1.scale_planes(s).expect("scale");
        let v2 = r2.scale_planes(s).expect("scale");
        let (sw, sh) = (v1.width, v1.height);
        let n = sw * sh;
        for (c, m) in mul.iter_mut().enumerate() {
            m.truncate(n);
            for ((o, &a), &b) in m
                .iter_mut()
                .zip(&v1.img1_planar[c][..n])
                .zip(&v2.img1_planar[c][..n])
            {
                *o = a * b;
            }
        }
        blur.shrink_to(sw, sh);
        let s12 = blur.blur(&mul);
        let px = |c: usize, i: usize| -> (f64, f64, f64) {
            let (mu1, mu2) = (v1.mu1[c][i] as f64, v2.mu1[c][i] as f64);
            let (s11, s22, sx) = (
                v1.sigma1_sq[c][i] as f64,
                v2.sigma1_sq[c][i] as f64,
                s12[c][i] as f64,
            );
            let md = mu1 - mu2;
            let num_m = (-md).mul_add(md, 1.0);
            let num_s = 2.0f64.mul_add(sx - mu1 * mu2, SSIM2_C2);
            let den_s = (s11 - mu1 * mu1) + (s22 - mu2 * mu2) + SSIM2_C2;
            let dd = (1.0 - (num_m * num_s) / den_s).max(0.0);
            let d1 = (1.0 + (v2.img1_planar[c][i] as f64 - mu2).abs())
                / (1.0 + (v1.img1_planar[c][i] as f64 - mu1).abs())
                - 1.0;
            (dd, d1.max(0.0), (-d1).max(0.0))
        };
        let nf = n as f64;
        let mut t = [0.0f64; 18];
        let mut sev = vec![0.0f64; n];
        for c in 0..3 {
            let (mut sd, mut sd4, mut sa, mut sa4, mut sl, mut sl4) =
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            for i in 0..n {
                let (dd, a, l) = px(c, i);
                sd += dd;
                sd4 += dd.powi(4);
                sa += a;
                sa4 += a.powi(4);
                sl += l;
                sl4 += l.powi(4);
            }
            let (md, r4d) = (sd / nf, (sd4 / nf).sqrt().sqrt());
            let (ma, r4a) = (sa / nf, (sa4 / nf).sqrt().sqrt());
            let (ml, r4l) = (sl / nf, (sl4 / nf).sqrt().sqrt());
            t[c * 6..c * 6 + 6].copy_from_slice(&[md, r4d, ma, r4a, ml, r4l]);
            let wat = |norm: usize, m: usize| SSIM2_WEIGHT[((c * ns + s) * 2 + norm) * 3 + m];
            for (i, sv) in sev.iter_mut().enumerate() {
                let (dd, a, l) = px(c, i);
                // L1 terms split as w·x/N, L4 terms as w·x⁴/(N·r4³): each sums
                // back to its pooled term.
                let mut v = (wat(0, 0) * dd + wat(0, 1) * a + wat(0, 2) * l) / nf;
                if r4d > 0.0 {
                    v += wat(1, 0) * dd.powi(4) / (nf * r4d.powi(3));
                }
                if r4a > 0.0 {
                    v += wat(1, 1) * a.powi(4) / (nf * r4a.powi(3));
                }
                if r4l > 0.0 {
                    v += wat(1, 2) * l.powi(4) / (nf * r4l.powi(3));
                }
                *sv += v;
            }
        }
        terms.push(t);
        sev_scales.push((sw, sh, sev));
    }
    for c in 0..3 {
        for t in &terms {
            for norm in 0..2 {
                for m in 0..3 {
                    raw = SSIM2_WEIGHT[wi].mul_add(t[c * 6 + m * 2 + norm].abs(), raw);
                    wi += 1;
                }
            }
        }
    }
    for (s, (sw, sh, sev)) in sev_scales.iter().enumerate() {
        let step = 1usize << s;
        for sy in 0..*sh {
            let (y0, y1) = (sy * step, ((sy + 1) * step).min(h));
            for sx in 0..*sw {
                let (x0, x1) = (sx * step, ((sx + 1) * step).min(w));
                let share = sev[sy * sw + sx] / ((x1 - x0) * (y1 - y0)) as f64;
                for y in y0..y1 {
                    for x in x0..x1 {
                        density[y * w + x] += share;
                    }
                }
            }
        }
    }
    Ok((density, raw))
}

fn write_f32(path: &Path, v: impl Iterator<Item = f32>) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("create map"));
    for x in v {
        f.write_all(&x.to_le_bytes()).expect("write map");
    }
}

struct Pair {
    key: String,
    reference: PathBuf,
    distorted: PathBuf,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut pairs, mut out, mut bake, mut tag) = (None, None, None, String::from("attr"));
    let (mut bin, mut legacy, mut judges) = (1usize, false, false);
    let (mut window, mut paint, mut split) = (
        String::from("box"),
        String::from("nearest"),
        String::from("repair"),
    );
    let mut smooth = 0.0f32;
    let mut guide = String::from("sq");
    while let Some(a) = args.next() {
        match a.as_str() {
            "--pairs" => pairs = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from),
            "--bake" => bake = args.next().map(PathBuf::from),
            "--tag" => tag = args.next().expect("--tag NAME"),
            "--bin" => bin = args.next().expect("--bin N").parse().expect("bin"),
            "--window" => window = args.next().expect("--window MODE"),
            "--paint" => paint = args.next().expect("--paint MODE"),
            "--split" => split = args.next().expect("--split MODE"),
            "--guide" => guide = args.next().expect("--guide sq|abs"),
            "--smooth" => {
                smooth = args
                    .next()
                    .expect("--smooth SIGMA")
                    .parse()
                    .expect("--smooth SIGMA")
            }
            "--legacy" => legacy = true,
            "--judges" => judges = true,
            other => panic!("unknown argument {other}"),
        }
    }
    let (pairs, out) = (pairs.expect("--pairs"), out.expect("--out"));
    std::fs::create_dir_all(&out).expect("out dir");
    let model_bytes = bake.as_ref().map(|p| std::fs::read(p).expect("read bake"));
    let model = model_bytes
        .as_ref()
        .map(|b| zenpredict::Model::from_bytes(b).expect("parse bake"));
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(&pairs)
        .expect("pairs");
    let hdr = rdr.headers().expect("header").clone();
    let col = |n: &str| {
        hdr.iter()
            .position(|h| h == n)
            .unwrap_or_else(|| panic!("{n} column"))
    };
    let (ki, ri, di) = (col("key"), col("ref_path"), col("dist_path"));
    let rows: Vec<Pair> = rdr
        .records()
        .map(|r| {
            let r = r.expect("row");
            Pair {
                key: r[ki].to_string(),
                reference: PathBuf::from(&r[ri]),
                distorted: PathBuf::from(&r[di]),
            }
        })
        .collect();
    let lines: Vec<String> = rows
        .par_iter()
        .map(|p| {
            let rd = image::open(&p.reference).expect("decode ref").to_rgb8();
            let dd = image::open(&p.distorted).expect("decode dist").to_rgb8();
            let (w, h) = (rd.width() as usize, rd.height() as usize);
            assert_eq!((w, h), (dd.width() as usize, dd.height() as usize), "{}: size mismatch", p.key);
            let rpx: &[[u8; 3]] = rd.as_raw().as_chunks::<3>().0;
            let dpx: &[[u8; 3]] = dd.as_raw().as_chunks::<3>().0;
            let (rs, ds) = (RgbSlice::new(rpx, w, h), RgbSlice::new(dpx, w, h));
            let mut row = serde_json::json!({"key": p.key, "width": w, "height": h});
            if let Some(model) = &model {
                let mut scorer = BakeScorer::new(model)
                    .expect("bake scorer")
                    .with_parallel(false)
                    .with_map_paint(&window, &paint, &split)
                    .expect("paint modes")
                    .with_map_guide(&guide)
                    .expect("guide norm")
                    .with_map_smooth(smooth)
                    .expect("smooth sigma");
                let pre = scorer.precompute_reference(&rs).expect("reference");
                let mut session = Fused944Session::new();
                let att = scorer
                    .compute_with_ref_and_attribution(&rs, &pre, &ds, None, &mut session, bin)
                    .expect("attribution");
                let a = att.attribution();
                let total: f64 = a.density().iter().map(|&v| v as f64).sum::<f64>() * (bin * bin) as f64;
                write_f32(&out.join(format!("{}.{tag}.f32", p.key)), a.density().iter().copied());
                let sens = att.sensitivities();
                let (smax_i, smax) = sens
                    .iter()
                    .enumerate()
                    .fold((0usize, 0.0f64), |acc, (i, &v)| if v.abs() > acc.1 { (i, v.abs()) } else { acc });
                row[tag.as_str()] = serde_json::json!({"score": att.result().score(), "window": window, "paint": paint, "split": split, "guide": guide, "smooth": smooth,
                    "sens_max": smax, "sens_max_id": smax_i, "grid": [a.grid_width(), a.grid_height()],
                    "bin": a.bin(), "mass": total, "unsupported": att.unsupported_feature_ids().len()});
            }
            if legacy {
                let z = Zensim::new(ZensimProfile::B).with_parallel(false);
                let pre = z.precompute_reference(&rs).expect("legacy reference");
                let r = z
                    .compute_with_ref_and_diffmap(&pre, &ds, zensim::DiffmapWeighting::default())
                    .expect("legacy diffmap");
                write_f32(&out.join(format!("{}.legacy.f32", p.key)), r.diffmap().iter().copied());
                row["legacy"] = serde_json::json!({"score": r.score()});
            }
            if judges {
                let to_rgb8 = |px: &[[u8; 3]]| -> Vec<RGB8> { px.iter().map(|q| RGB8::new(q[0], q[1], q[2])).collect() };
                let (r8, d8) = (to_rgb8(rpx), to_rgb8(dpx));
                let b = butteraugli(
                    Img::new(&r8[..], w, h),
                    Img::new(&d8[..], w, h),
                    &ButteraugliParams::default().with_compute_diffmap(true),
                )
                .expect("butteraugli");
                let bm = b.diffmap.expect("diffmap requested");
                write_f32(&out.join(format!("{}.butter.f32", p.key)), bm.buf().iter().copied());
                let (field, raw) = ssim2_field(ImgRef::new(rpx, w, h), ImgRef::new(dpx, w, h)).expect("ssim2 field");
                write_f32(&out.join(format!("{}.ssim2.f32", p.key)), field.iter().map(|&v| v as f32));
                row["butteraugli"] = serde_json::json!(b.score);
                row["ssim2_raw"] = serde_json::json!(raw);
                #[cfg(feature = "dvifm-judge")]
                {
                    // DVIFM 0.3 in port mode: the author's v0.3 Laplacian variant, as the frozen
                    // 2026-10-20 runtime scored it (`--params` port-lap-v0.3).
                    let params = dvifmish::port::presets::get("port-lap-v0.3").expect("DVIFM 0.3 port preset");
                    let flat = |px: &[[u8; 3]]| -> Vec<u8> { px.iter().flatten().copied().collect() };
                    let (rf, df) = (flat(rpx), flat(dpx));
                    let ri = dvifmish::Image::new(w, h, &rf).expect("dvifm ref");
                    let di = dvifmish::Image::new(w, h, &df).expect("dvifm dist");
                    let (score, map) = dvifmish::port::score_with_map(&params, &ri, &di).expect("dvifm");
                    let painted = map.painted(None, None);
                    write_f32(&out.join(format!("{}.dvifm.f32", p.key)), painted.iter().map(|&v| v as f32));
                    row["dvifm"] = serde_json::json!({"raw": score.raw, "jnd": score.jnd, "map_total": map.total()});
                }
            }
            row.to_string()
        })
        .collect();
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    for l in lines {
        writeln!(lock, "{l}").expect("stdout");
    }
}
