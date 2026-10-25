//! Where a bake's signed attribution map puts its extremes for one pair, under
//! each map mode the workbench offers (window, paint, split, smoothing): the
//! density's minimum and maximum with their pixels, the share of its total in
//! a border band, and the values along a probe row. For the loop-convergence
//! investigation (`loop-converge`), whose map maxima sat at the image edge.
//!
//! ```text
//! cargo run --release --manifest-path zensim-workbench/loop-converge/Cargo.toml --bin map_probe -- \
//!     REF.png DIST.rgb BAKE.bin [ROW]
//! ```
//!
//! `DIST.rgb` is packed sRGB8 RGB of the reference's size (`loop-converge`'s
//! `LOOP_CONVERGE_DUMP` frames).

use zensim::{BakeScorer, Fused944Session, RgbSlice};

fn read_png(path: &str) -> (Vec<[u8; 3]>, usize, usize) {
    let mut decoder = png::Decoder::new(std::io::BufReader::new(
        std::fs::File::open(path).expect("open image"),
    ));
    // Palette and sub-byte images expand to 8-bit RGB(A); 16-bit strips.
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().expect("png header");
    let mut buf = vec![0; reader.output_buffer_size().expect("png size")];
    let info = reader.next_frame(&mut buf).expect("png frame");
    assert_eq!(info.color_type, png::ColorType::Rgb, "RGB8 PNG expected");
    let px = buf[..info.buffer_size()].as_chunks::<3>().0.to_vec();
    (px, info.width as usize, info.height as usize)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: map_probe REF.png DIST.rgb BAKE.bin [ROW]");
        std::process::exit(2);
    }
    let (reference, w, h) = read_png(&args[1]);
    let dist_bytes = std::fs::read(&args[2]).expect("read dist");
    assert_eq!(
        dist_bytes.len(),
        w * h * 3,
        "DIST.rgb must match the reference size"
    );
    let distorted = dist_bytes.as_chunks::<3>().0.to_vec();
    let bake = std::fs::read(&args[3]).expect("read bake");
    let model = zenpredict::Model::from_bytes(&bake).expect("bake");
    let row: usize = args.get(4).map_or(h / 2, |s| s.parse().expect("row"));
    let (rs, ds) = (
        RgbSlice::new(&reference, w, h),
        RgbSlice::new(&distorted, w, h),
    );
    let variants = [
        ("center", "guided", "exact", 3.0f32),
        ("center", "guided", "exact", 0.0),
        ("center", "nearest", "exact", 0.0),
        ("box", "nearest", "exact", 0.0),
        ("box", "nearest", "repair", 0.0),
        ("center", "guided", "repair", 0.0),
    ];
    // PROBE_REPAIR=x,y: the features of the pair with that pixel copied from
    // the reference, against the pair's own, weighted by the pair's
    // sensitivities (the first-order score change of the repair, per feature).
    if let Ok(xy) = std::env::var("PROBE_REPAIR") {
        let (x, y) = xy.split_once(',').expect("PROBE_REPAIR=x,y");
        let i = y.parse::<usize>().expect("y") * w + x.parse::<usize>().expect("x");
        let mut fixed = distorted.clone();
        fixed[i] = reference[i];
        let mut scorer = BakeScorer::new(&model)
            .expect("scorer")
            .with_parallel(false);
        let pre = scorer.precompute_reference(&rs).expect("reference");
        let mut session = Fused944Session::new();
        let att = scorer
            .compute_with_ref_and_attribution(&rs, &pre, &ds, None, &mut session, 1)
            .expect("attribution");
        let sens = att.sensitivities().to_vec();
        let f0 = att.result().features().to_vec();
        let fs = RgbSlice::new(&fixed, w, h);
        let mut session = Fused944Session::new();
        let att1 = scorer
            .compute_with_ref_and_attribution(&rs, &pre, &fs, None, &mut session, 1)
            .expect("attribution");
        let f1 = att1.result().features().to_vec();
        println!(
            "repair ({xy}): score {:.6} -> {:.6}; {} features, {} sensitivities",
            att.result().score(),
            att1.result().score(),
            f0.len(),
            sens.len()
        );
        let mut rows: Vec<(usize, f64, f64, f64)> = (0..f0.len().min(sens.len()))
            .map(|k| (k, sens[k], f1[k] - f0[k], sens[k] * (f1[k] - f0[k])))
            .collect();
        rows.sort_by(|a, b| b.3.abs().total_cmp(&a.3.abs()));
        for (k, s, df, c) in rows.iter().take(12) {
            println!(
                "  f{k}: sensitivity {s:.4e} change {df:.4e} ({:.4e} -> {:.4e}) contribution {c:.4e}",
                f0[*k], f1[*k]
            );
        }
        return;
    }
    for (window, paint, split, smooth) in variants {
        let mut scorer = BakeScorer::new(&model)
            .expect("scorer")
            .with_parallel(false)
            .with_map_paint(window, paint, split)
            .and_then(|s| s.with_map_smooth(smooth))
            .expect("map modes");
        let pre = scorer.precompute_reference(&rs).expect("reference");
        let mut session = Fused944Session::new();
        let att = scorer
            .compute_with_ref_and_attribution(&rs, &pre, &ds, None, &mut session, 1)
            .expect("attribution");
        let a = att.attribution();
        let d = a.density();
        let n = (w * h) as f64;
        let (mut lo, mut hi) = ((f32::MAX, 0usize), (f32::MIN, 0usize));
        for (i, &v) in d.iter().enumerate() {
            if v < lo.0 {
                lo = (v, i);
            }
            if v > hi.0 {
                hi = (v, i);
            }
        }
        let total: f64 = d.iter().map(|&v| f64::from(v)).sum();
        let band: f64 = d
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                let (x, y) = (i % w, i / w);
                x < 12 || y < 12 || x + 12 >= w || y + 12 >= h
            })
            .map(|(_, &v)| f64::from(v))
            .sum();
        let abs_band: f64 = d
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                let (x, y) = (i % w, i / w);
                x < 12 || y < 12 || x + 12 >= w || y + 12 >= h
            })
            .map(|(_, &v)| f64::from(v).abs())
            .sum();
        let abs_total: f64 = d.iter().map(|&v| f64::from(v).abs()).sum();
        let probe: Vec<String> = (0..12)
            .map(|x| format!("{:.2}", f64::from(d[row * w + x]) * n))
            .collect();
        println!(
            "{window}/{paint}/{split}/σ{smooth}: score {:.4}; density x N min {:.3} at ({},{}) max {:.3} at ({},{}); \
             total {:.5} (border-12 band {:.5}); |d| border share {:.1}%; row {row} x0..11: [{}]",
            att.result().score(),
            f64::from(lo.0) * n,
            lo.1 % w,
            lo.1 / w,
            f64::from(hi.0) * n,
            hi.1 % w,
            hi.1 / w,
            total,
            band,
            100.0 * abs_band / abs_total,
            probe.join(", ")
        );
    }
}
