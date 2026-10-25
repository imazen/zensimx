//! The workbench's JPEG XL loop run natively, to see where it lands against
//! the requested distance: jxl-encoder's perceptual loop at each distance,
//! scored by its built-in butteraugli or, through the adapter zenmetrics'
//! wasm codec uses (`web/codecs/jxl/src/lib.rs`: each reconstruction as sRGB8
//! RGBA, the metric's loop map on butteraugli's scale, the score its largest
//! value), by an embedded metric's map at its fitted placement
//! (`loop_map_defaults`). Per distance it prints every iteration's score, the
//! largest per-block tile distance (what the loop steers to the target) and
//! how many blocks are still above it, then the final score against the
//! request.
//!
//! ```text
//! ZENSIM_WORKBENCH_BAKES=... cargo run --release --manifest-path zensim-workbench/loop-converge/Cargo.toml -- \
//!     IMAGE.png METRIC MAP DISTANCES [ITERS=4] [EFFORT=7]
//! ```
//!
//! `METRIC` is an embedded metric's name, or `butteraugli` for the built-in
//! loop (`MAP` is then ignored); `DISTANCES` is comma-separated. The
//! workbench's loop player defaults to effort 7 and 4 iterations.

use jxl_encoder::{ExternalMetric, LoopTrace, LossyConfig, PixelLayout};
use zensim_workbench_loop_converge::{ErrorSlot, MapScorer, read_png, rgba as rgba_ref, srgb8};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: loop_converge IMAGE.png METRIC MAP DISTANCES [ITERS] [EFFORT]");
        std::process::exit(2);
    }
    let (rgb, width, height) = read_png(&args[1]);
    let (metric, map) = (args[2].clone(), args[3].clone());
    let distances: Vec<f32> = args[4]
        .split(',')
        .map(|d| d.parse().expect("distance"))
        .collect();
    let iters: u32 = args.get(5).map_or(4, |s| s.parse().expect("iters"));
    let effort: u8 = args.get(6).map_or(7, |s| s.parse().expect("effort"));
    // LOOP_PLACEMENT=SCALE,EXPONENT[;DIST=SCALE,EXPONENT...]: placements to
    // use instead of the module's fitted one, per requested distance when
    // keyed (the unkeyed entry is the fallback).
    let overrides: Vec<(Option<f32>, f64, f64)> = std::env::var("LOOP_PLACEMENT")
        .map(|v| {
            v.split(';')
                .map(|e| {
                    let (key, se) = match e.split_once('=') {
                        Some((k, se)) => (Some(k.parse::<f32>().expect("distance key")), se),
                        None => (None, e),
                    };
                    let (sc, ex) = se.split_once(',').expect("SCALE,EXPONENT");
                    (
                        key,
                        sc.parse().expect("scale"),
                        ex.parse().expect("exponent"),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    let placement = (metric != "butteraugli").then(|| {
        let defaults: Vec<serde_json::Value> =
            serde_json::from_str(&zensim_workbench::loop_map_defaults()).expect("defaults");
        let d = defaults
            .iter()
            .find(|d| d["metric"] == metric.as_str() && d["map"] == map.as_str())
            .unwrap_or_else(|| panic!("no placement for {metric} {map}"));
        (
            d["scale"].as_f64().expect("scale"),
            d["exponent"].as_f64().expect("exponent"),
        )
    });
    let rgba = rgba_ref(&rgb);
    println!(
        "image {} {width}x{height}; metric {metric} {}; iters {iters}; effort {effort}",
        args[1],
        placement.map_or("built-in".into(), |(s, e)| format!(
            "{map} scale {s} exponent {e}"
        ))
    );
    for &distance in &distances {
        let trace = LoopTrace::new();
        let config = LossyConfig::new(distance)
            .with_effort(effort)
            .with_butteraugli_iters(iters);
        let request = config
            .encode_request(width, height, PixelLayout::Rgb8)
            .with_loop_trace(&trace);
        let placed = placement.map(|p| {
            overrides
                .iter()
                .find(|o| o.0 == Some(distance))
                .or_else(|| overrides.iter().find(|o| o.0.is_none()))
                .map_or(p, |o| (o.1, o.2))
        });
        let error = ErrorSlot::default();
        let external = placed.map(|(scale, exponent)| {
            ExternalMetric::new(MapScorer {
                metric: metric.clone(),
                map: map.clone(),
                scale,
                exponent,
                reference: rgba.clone(),
                linear: false,
                error: error.clone(),
            })
        });
        let request = match &external {
            Some(m) => request.with_external_metric(m),
            None => request,
        };
        let file = request.encode(&rgb).expect("encode");
        if let Some(e) = error.lock().expect("error slot").take() {
            panic!("d {distance}: the steering map failed: {e}");
        }
        let frames = trace.take();
        // LOOP_CONVERGE_DUMP=DIR: every frame's reconstruction (sRGB8 RGB, raw)
        // and map (f32 LE) as d<distance>_s<seed>_i<iter>.{rgb,map}.
        if let Some(dir) = std::env::var_os("LOOP_CONVERGE_DUMP") {
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir).expect("dump dir");
            for f in &frames {
                let stem = format!("d{distance}_s{}_i{}", f.seed, f.iter);
                let [r, g, b] = &f.recon;
                let rgb: Vec<u8> = (0..r.len())
                    .flat_map(|i| [srgb8(r[i]), srgb8(g[i]), srgb8(b[i])])
                    .collect();
                std::fs::write(dir.join(format!("{stem}.rgb")), rgb).expect("write rgb");
                let map: Vec<u8> = f.diffmap.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(dir.join(format!("{stem}.map")), map).expect("write map");
            }
        }
        let mut line = format!("d {distance}: {} bytes;", file.len());
        if let Some((sc, ex)) = placed {
            line += &format!(" placement {sc:.4e}/{ex};");
        }
        for f in &frames {
            let tile_max = f.tile_dist.iter().copied().fold(0f32, f32::max);
            let above = f
                .tile_dist
                .iter()
                .filter(|&&t| f64::from(t) > f.target)
                .count();
            line += &format!(
                " i{} score {:.3} tile_max {:.3} above {};",
                f.iter, f.score, tile_max, above
            );
        }
        // LOOP_CONVERGE_SCORE=METRIC: that embedded metric's score of the
        // first and last frames' reconstructions (the first is the encode
        // before any loop adjustment).
        if let Ok(judge) = std::env::var("LOOP_CONVERGE_SCORE") {
            for f in [frames.first(), frames.last()].into_iter().flatten() {
                let [r, g, b] = &f.recon;
                let rgba: Vec<u8> = (0..r.len())
                    .flat_map(|i| [srgb8(r[i]), srgb8(g[i]), srgb8(b[i]), 255])
                    .collect();
                let json = zensim_workbench::score_json(
                    &judge,
                    &rgba_ref(&rgb),
                    &rgba,
                    width,
                    height,
                    width as usize * 4,
                )
                .expect("score");
                line += &format!(" [{judge} i{}: {json}]", f.iter);
            }
        }
        let last = frames.last().expect("frames");
        println!(
            "{line} final {:.3} = {:.3} x requested (block target {:.3})",
            last.score,
            last.score / f64::from(distance),
            last.target
        );
    }
}
