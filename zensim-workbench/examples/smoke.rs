//! Native side of the wasm smoke check (`just byc-workbench-smoke`): scores a
//! fixed synthetic RGBA pair with every embedded metric and prints, per
//! metric, the score JSON, each loop map's sum and each worker map's cell sum,
//! one JSON line per metric. The node side (`smoke.mjs`) prints the score and
//! loop maps from the wasm module; `workbench_check.py` compares the worker's
//! maps in headless Chromium.

fn main() {
    let (w, h) = (96u32, 80u32);
    let (r, d) = pair(w as usize, h as usize);
    let metrics: Vec<String> =
        serde_json::from_str(&zensim_workbench::available_metrics()).expect("metrics");
    for metric in metrics {
        let score: serde_json::Value = serde_json::from_str(
            &zensim_workbench::score_json(&metric, &r, &d, w, h, w as usize * 4).expect("score"),
        )
        .expect("score json");
        let maps: serde_json::Map<String, serde_json::Value> = zensim_workbench::map_names()
            .iter()
            .map(|m| {
                let (values, _) = zensim_workbench::loop_map_values(
                    &metric,
                    m,
                    &r,
                    &d,
                    w,
                    h,
                    w as usize * 4,
                    1.0,
                    1.0,
                )
                .expect("map");
                (
                    m.clone(),
                    serde_json::json!(values.iter().map(|&v| f64::from(v)).sum::<f64>()),
                )
            })
            .collect();
        // The maps the worker shows: each one's cell values summed.
        let rgb = |v: &[u8]| -> Vec<[u8; 3]> {
            v.as_chunks::<4>()
                .0
                .iter()
                .map(|p| [p[0], p[1], p[2]])
                .collect()
        };
        let (rr, dd) = (rgb(&r), rgb(&d));
        let spatial: serde_json::Map<String, serde_json::Value> = zensim_workbench::map_names()
            .iter()
            .map(|m| {
                let map =
                    zensim_workbench::compute_map(&metric, m, &rr, &dd, w, h).expect("spatial map");
                (
                    m.clone(),
                    serde_json::json!(map.values().iter().map(|&v| f64::from(v)).sum::<f64>()),
                )
            })
            .collect();
        println!(
            "{}",
            serde_json::json!({"metric": metric, "score": score, "maps": maps, "spatial": spatial})
        );
    }
}

/// A ramp-and-texture source and a lightly blurred, offset copy, RGBA.
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
