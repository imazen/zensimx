//! `loop_map_values_linear` against `loop_map_values` on an image both can
//! express: the distorted image's sRGB8 bytes through the exact sRGB transfer
//! (f64, rounded once to f32), so the two inputs carry the same pixels. Prints
//! one JSON line per metric and map (the largest absolute difference, the 8-bit
//! map's largest value, and the two scores) and exits non-zero when any
//! difference exceeds 1e-4 of that map's largest value (my bound: the two
//! paths convert to XYB through different code, so equality is not bitwise).
//! Record: benchmarks/byconstruction_2026-10-22.md.
//!
//! ```text
//! ZENSIM_WORKBENCH_BAKES=... cargo run --release --manifest-path zensim-workbench/Cargo.toml --example linear_parity
//! ```

fn srgb_to_linear(v: u8) -> f32 {
    let c = f64::from(v) / 255.0;
    (if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }) as f32
}

fn main() {
    let (w, h) = (96usize, 80usize);
    let (r, d) = pair(w, h);
    let plane = |c: usize| -> Vec<f32> {
        d.as_chunks::<4>()
            .0
            .iter()
            .map(|p| srgb_to_linear(p[c]))
            .collect()
    };
    let (pr, pg, pb) = (plane(0), plane(1), plane(2));
    let metrics: Vec<String> =
        serde_json::from_str(&zensim_workbench::available_metrics()).expect("metrics");
    let mut worst = 0f64;
    for metric in &metrics {
        for map in zensim_workbench::map_names() {
            let (a, sa) = zensim_workbench::loop_map_values(
                metric,
                &map,
                &r,
                &d,
                w as u32,
                h as u32,
                w * 4,
                1.0,
                1.0,
            )
            .expect("8-bit map");
            let (b, sb) = zensim_workbench::loop_map_values_linear(
                metric,
                &map,
                &r,
                w * 4,
                &pr,
                &pg,
                &pb,
                w,
                w as u32,
                h as u32,
                1.0,
                1.0,
            )
            .expect("linear map");
            let top = a.iter().copied().fold(0f32, f32::max) as f64;
            let diff = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
                .fold(0f64, f64::max);
            let rel = if top > 0.0 { diff / top } else { diff };
            worst = worst.max(rel);
            println!(
                "{}",
                serde_json::json!({"metric": metric, "map": map, "max_abs_diff": diff, "map_max": top,
                                   "relative": rel, "score_8bit": sa, "score_linear": sb})
            );
        }
    }
    if worst > 1e-4 {
        eprintln!("linear and 8-bit maps differ by {worst:e} of a map's largest value");
        std::process::exit(1);
    }
}

/// The smoke check's pair: a ramp-and-texture source and a lightly blurred,
/// offset copy, RGBA.
fn pair(w: usize, h: usize) -> (Vec<u8>, Vec<u8>) {
    let px = |x: usize, y: usize| -> [u8; 4] {
        let v = ((x * 7 + y * 13) % 200 + 20) as u8;
        [v, v / 2 + 40, 255 - v, 255]
    };
    let mut r = Vec::with_capacity(w * h * 4);
    let mut d = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            r.extend_from_slice(&px(x, y));
            let (a, b) = (px(x, y), px((x + 1).min(w - 1), y));
            d.extend_from_slice(&[
                ((u16::from(a[0]) + u16::from(b[0])) / 2) as u8,
                ((u16::from(a[1]) + u16::from(b[1])) / 2 + 2).min(255) as u8,
                ((u16::from(a[2]) + u16::from(b[2])) / 2) as u8,
                255,
            ]);
        }
    }
    (r, d)
}
