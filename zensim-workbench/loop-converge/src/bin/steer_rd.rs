//! Rate–distortion of JPEG XL loop steering maps: every image of a corpus
//! encoded at each distance by jxl-encoder's perceptual loop under one
//! steering arm, then decoded, in the layout the shared judge panel reads
//! (`scripts/v_next/rd_probe_2026-07-18.sh judge`, analysis
//! `scripts/v_next/rd_probe_analyze_2026-07-18.py`). Every arm runs in this
//! one binary, so the encoder build is the same for all of them.
//!
//! ```text
//! ZENSIM_WORKBENCH_BAKES=... cargo run --release --manifest-path zensim-workbench/loop-converge/Cargo.toml \
//!     --bin steer_rd -- --corpus CORPUS.tsv --out RD/jxl --label LABEL --steer STEER \
//!     [--distances 0.3,0.45,...] [--iters 4] [--effort 7] [--threads 8] [--placement SCALE,EXPONENT] \
//!     [--probe workbench:METRIC:MAP ...] [--frames DIR] [--linear] [--server CMD ARG...]
//! ```
//!
//! `STEER` is `none` (no loop: zero iterations), `butteraugli` (the built-in
//! loop), `workbench:METRIC:MAP` (a zensim-workbench map at its fitted
//! placement, `loop_map_defaults`) or `server` (a [`MapServer`] per worker
//! thread, started from the arguments after `--server`, which must come
//! last). `CORPUS.tsv` lines are `path`, `name`, `class` (tab-separated, no
//! header; `scripts/byconstruction/jxl_map_corpus.py` writes it). Defaults:
//! the workbench loop player's effort 7 and 4 iterations; distances
//! 0.3–10.
//!
//! Writes under `--out`: `ref/NAME.png` (the source as loaded; an existing one
//! must hold the same pixels), `decoded/LABEL__NAME__dD.png` (the bitstream
//! decoded by zenjxl-decoder to dithered sRGB8, as the earlier harness
//! `zensim_diffmap_rd` decodes), `manifest_LABEL.tsv` (the judge's input),
//! `artifacts/jxl/SHA256.jxl` (every bitstream), `loop_LABEL.tsv` (per encode:
//! the bitstream's SHA-256, the loop's frames and seeds, and its last frame's
//! score, target, largest tile distance and blocks above the target) and
//! `run_LABEL.json` (arguments, placement or server header, the embedded
//! bakes with their SHA-256, this executable's SHA-256).
//!
//! `--linear` scores a workbench arm's reconstructions as the loop holds them
//! (linear-light floats, not rounded to sRGB8: errors under half an 8-bit step
//! still count); a server gets floats when its header asks for
//! `"input":"linear-rgb-f32"`. Both are recorded in `run_LABEL.json`.
//!
//! `--placement` replaces a workbench arm's fitted placement. Each `--probe`
//! (a workbench map at its fitted placement, or `workbench:METRIC:MAP@SCALE,EXPONENT`
//! at that one) is computed on the *decoded*
//! output, what a viewer gets, and adds three columns to `loop_LABEL.tsv`: the
//! map's largest value (the workbench's loop score), its largest 8×8 tile
//! distance (1.2 × the block's 16-norm, the loop's rule, over plain 8×8
//! blocks) and the number of tiles above the requested distance. `--frames`
//! writes every loop frame's per-block arrays to
//! `DIR/LABEL__NAME__dD.frames` (little-endian; per frame: `u32` iteration,
//! `f64` seed, score and target, `u32` block count `n`, then `n` `f32` each of
//! tile distances, the quant field and the next quant field, NaN when the
//! frame has none).
//!
//! A steering map that fails stops the run: the loop itself would only end
//! the seed and keep the previous iteration.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use jxl_encoder::{ExternalMetric, LoopTrace, LossyConfig, PixelLayout};
use sha2::{Digest, Sha256};
use zensim_workbench_loop_converge::{
    ErrorSlot, MapScorer, MapServer, ServerScorer, read_png, rgba,
};

const DISTANCES: &str = "0.3,0.45,0.6,1.0,1.6,2.5,4.0,6.4,10.0";

enum Steer {
    None,
    Butteraugli,
    Workbench {
        metric: String,
        map: String,
        scale: f64,
        exponent: f64,
    },
    Server(Vec<String>),
}

struct Image {
    name: String,
    class: String,
    rgb: Vec<u8>,
    width: u32,
    height: u32,
}

struct Encoded {
    image: usize,
    distance: usize,
    bytes: usize,
    sha256: String,
    encode_ms: f64,
    dist_png: PathBuf,
    frames: usize,
    seeds: usize,
    last: Option<(f64, f64, f32, usize)>,
    probes: Vec<(f64, f32, usize)>,
}

/// A workbench map at its fitted placement.
struct Placed {
    metric: String,
    map: String,
    scale: f64,
    exponent: f64,
}

fn fitted(metric: &str, map: &str) -> Placed {
    let defaults: Vec<serde_json::Value> =
        serde_json::from_str(&zensim_workbench::loop_map_defaults()).expect("defaults");
    let d = defaults
        .iter()
        .find(|d| d["metric"] == metric && d["map"] == map)
        .unwrap_or_else(|| panic!("no placement for {metric} {map}"));
    assert!(d["r"].is_number(), "{metric} {map} has no fitted placement");
    Placed {
        metric: metric.to_string(),
        map: map.to_string(),
        scale: d["scale"].as_f64().expect("scale"),
        exponent: d["exponent"].as_f64().expect("exponent"),
    }
}

/// Each 8×8 block's tile distance, `1.2 · (Σ v¹⁶ / n)^(1/16)` (zenmetrics'
/// `tile_norms`, the loop's rule over plain 8×8 blocks).
fn tile_norms(values: &[f32], width: usize, height: usize) -> Vec<f32> {
    let (bx, by) = (width.div_ceil(8), height.div_ceil(8));
    let mut sum = vec![0f64; bx * by];
    let mut count = vec![0u32; bx * by];
    for y in 0..height {
        for x in 0..width {
            let b = (y / 8) * bx + x / 8;
            sum[b] += f64::from(values[y * width + x]).powi(16);
            count[b] += 1;
        }
    }
    sum.iter()
        .zip(&count)
        .map(|(&s, &n)| (1.2 * (s / f64::from(n)).powf(1.0 / 16.0)) as f32)
        .collect()
}

fn write_frames(path: &Path, frames: &[jxl_encoder::LoopFrame]) {
    let mut out = Vec::new();
    for f in frames {
        let n = f.tile_dist.len();
        assert_eq!(f.quant_field.len(), n, "quant field per block");
        out.extend(f.iter.to_le_bytes());
        for v in [f.seed, f.score, f.target] {
            out.extend(v.to_le_bytes());
        }
        out.extend((n as u32).to_le_bytes());
        for v in &f.tile_dist {
            out.extend(v.to_le_bytes());
        }
        for v in &f.quant_field {
            out.extend(v.to_le_bytes());
        }
        match &f.next_quant_field {
            Some(q) => q.iter().for_each(|v| out.extend(v.to_le_bytes())),
            None => (0..n).for_each(|_| out.extend(f32::NAN.to_le_bytes())),
        }
    }
    std::fs::write(path, out).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .fold(String::new(), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

fn write_png(path: &Path, rgb: &[u8], width: u32, height: u32) {
    let file = std::fs::File::create(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().expect("png header");
    writer.write_image_data(rgb).expect("png data");
}

/// The bitstream decoded to dithered sRGB8 RGB, as `zensim_diffmap_rd`'s
/// canonical decode.
fn decode(bitstream: &[u8], width: u32, height: u32) -> Vec<u8> {
    let image = zenjxl_decoder::decode_with(
        bitstream,
        zenjxl_decoder::api::JxlDecoderOptions::default().with_dither_u8(true),
    )
    .expect("decode");
    assert!(
        (image.width, image.height) == (width as usize, height as usize)
            && matches!(image.channels, 2 | 4)
            && image.data.len() == image.width * image.height * image.channels,
        "decoded shape"
    );
    let mut rgb = Vec::with_capacity(image.width * image.height * 3);
    for p in image.data.chunks_exact(image.channels) {
        assert_eq!(p[image.channels - 1], 255, "opaque output expected");
        if image.channels == 2 {
            rgb.extend_from_slice(&[p[0]; 3]);
        } else {
            rgb.extend_from_slice(&p[..3]);
        }
    }
    rgb
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut corpus, mut out, mut label, mut steer) = (None, None, None, None);
    let (mut distances, mut iters, mut effort, mut threads) =
        (DISTANCES.to_string(), 4u32, 7u8, 8usize);
    let mut server = Vec::new();
    let (mut placement_override, mut probe_specs, mut frames_dir) = (None, Vec::new(), None);
    let mut linear = false;
    while let Some(a) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--corpus" => corpus = Some(value()),
            "--out" => out = Some(PathBuf::from(value())),
            "--label" => label = Some(value()),
            "--steer" => steer = Some(value()),
            "--distances" => distances = value(),
            "--iters" => iters = value().parse().expect("--iters"),
            "--effort" => effort = value().parse().expect("--effort"),
            "--threads" => threads = value().parse().expect("--threads"),
            "--placement" => placement_override = Some(value()),
            "--probe" => probe_specs.push(value()),
            "--frames" => frames_dir = Some(PathBuf::from(value())),
            "--linear" => linear = true,
            "--server" => {
                server = args.by_ref().collect();
                break;
            }
            other => panic!("unknown argument {other}"),
        }
    }
    let corpus = corpus.expect("--corpus CORPUS.tsv");
    let out = out.expect("--out DIR");
    let label = label.expect("--label LABEL");
    let steer_spec = steer.expect("--steer none|butteraugli|workbench:METRIC:MAP|server");
    let distances: Vec<f32> = distances
        .split(',')
        .map(|d| d.parse().expect("distance"))
        .collect();
    let steer = match steer_spec.split(':').collect::<Vec<_>>().as_slice() {
        ["none"] => Steer::None,
        ["butteraugli"] => Steer::Butteraugli,
        ["workbench", metric, map] => {
            // An explicit placement needs no fitted one.
            let (scale, exponent) = match &placement_override {
                Some(se) => {
                    let (sc, ex) = se.split_once(',').expect("--placement SCALE,EXPONENT");
                    (sc.parse().expect("scale"), ex.parse().expect("exponent"))
                }
                None => {
                    let p = fitted(metric, map);
                    (p.scale, p.exponent)
                }
            };
            Steer::Workbench {
                metric: metric.to_string(),
                map: map.to_string(),
                scale,
                exponent,
            }
        }
        ["server"] => {
            assert!(
                !server.is_empty(),
                "--steer server needs --server CMD ARG..."
            );
            Steer::Server(server.clone())
        }
        _ => panic!("unknown --steer {steer_spec}"),
    };
    assert!(
        placement_override.is_none() || matches!(steer, Steer::Workbench { .. }),
        "--placement applies to a workbench arm"
    );
    assert!(
        !linear || matches!(steer, Steer::Workbench { .. }),
        "--linear applies to a workbench arm (a server asks for floats in its header)"
    );
    let iters = if matches!(steer, Steer::None) {
        0
    } else {
        iters
    };
    let probes: Vec<Placed> = probe_specs
        .iter()
        .map(
            |spec| match spec.split(':').collect::<Vec<_>>().as_slice() {
                ["workbench", metric, map] => match map.split_once('@') {
                    // `MAP@SCALE,EXPONENT`: the map at that placement instead of its fitted one.
                    Some((map, se)) => {
                        let (sc, ex) = se.split_once(',').expect("MAP@SCALE,EXPONENT");
                        Placed {
                            metric: metric.to_string(),
                            map: map.to_string(),
                            scale: sc.parse().expect("scale"),
                            exponent: ex.parse().expect("exponent"),
                        }
                    }
                    None => fitted(metric, map),
                },
                _ => panic!("unknown --probe {spec}"),
            },
        )
        .collect();
    if let Some(dir) = &frames_dir {
        std::fs::create_dir_all(dir).expect("frames dir");
    }

    let corpus_text = std::fs::read_to_string(&corpus).unwrap_or_else(|e| panic!("{corpus}: {e}"));
    let images: Vec<Image> = corpus_text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let c: Vec<&str> = l.split('\t').collect();
            assert!(c.len() >= 3, "corpus line {l:?}: path, name, class");
            let (rgb, width, height) = read_png(c[0]);
            Image {
                name: c[1].to_string(),
                class: c[2].to_string(),
                rgb,
                width,
                height,
            }
        })
        .collect();
    for dir in ["ref", "decoded", "artifacts/jxl"] {
        std::fs::create_dir_all(out.join(dir)).expect("output dirs");
    }
    for im in &images {
        let path = out.join("ref").join(format!("{}.png", im.name));
        if path.exists() {
            let (rgb, w, h) = read_png(path.to_str().expect("utf-8 path"));
            assert!(
                (w, h) == (im.width, im.height) && rgb == im.rgb,
                "{}: an existing reference holds other pixels",
                path.display()
            );
        } else {
            write_png(&path, &im.rgb, im.width, im.height);
        }
    }

    // A server's header is read once here, for the record; every worker starts its own.
    let (header, server_linear) = match &steer {
        Steer::Server(cmd) => {
            let probe = MapServer::spawn(cmd);
            (Some(probe.header.clone()), probe.linear)
        }
        _ => (None, false),
    };
    let jobs: Vec<(usize, usize)> = (0..images.len())
        .flat_map(|i| (0..distances.len()).map(move |d| (i, d)))
        .collect();
    let next = AtomicUsize::new(0);
    let results = Mutex::new(Vec::with_capacity(jobs.len()));
    let started = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| {
                let server = match &steer {
                    Steer::Server(cmd) => Some(Arc::new(Mutex::new(MapServer::spawn(cmd)))),
                    _ => None,
                };
                loop {
                    let j = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&(i, di)) = jobs.get(j) else { break };
                    let im = &images[i];
                    let distance = distances[di];
                    let trace = LoopTrace::new();
                    let error = ErrorSlot::default();
                    let config = LossyConfig::new(distance)
                        .with_effort(effort)
                        .with_butteraugli_iters(iters);
                    let request = config
                        .encode_request(im.width, im.height, PixelLayout::Rgb8)
                        .with_loop_trace(&trace);
                    let external = match &steer {
                        Steer::None | Steer::Butteraugli => None,
                        Steer::Workbench {
                            metric,
                            map,
                            scale,
                            exponent,
                        } => Some(ExternalMetric::new(MapScorer {
                            metric: metric.clone(),
                            map: map.clone(),
                            scale: *scale,
                            exponent: *exponent,
                            reference: rgba(&im.rgb),
                            linear,
                            error: error.clone(),
                        })),
                        Steer::Server(_) => Some(ExternalMetric::new(ServerScorer {
                            server: server.clone().expect("server"),
                            reference: rgba(&im.rgb),
                            error: error.clone(),
                        })),
                    };
                    let request = match &external {
                        Some(m) => request.with_external_metric(m),
                        None => request,
                    };
                    let t = Instant::now();
                    let bitstream = request.encode(&im.rgb).expect("encode");
                    let encode_ms = t.elapsed().as_secs_f64() * 1000.0;
                    if let Some(e) = error.lock().expect("error slot").take() {
                        panic!("{} d {distance}: the steering map failed: {e}", im.name);
                    }
                    let frames = trace.take();
                    let mut seeds: Vec<u64> = frames.iter().map(|f| f.seed.to_bits()).collect();
                    seeds.dedup();
                    let last = frames.last().map(|f| {
                        let tile_max = f.tile_dist.iter().copied().fold(0f32, f32::max);
                        let above = f
                            .tile_dist
                            .iter()
                            .filter(|&&t| f64::from(t) > f.target)
                            .count();
                        (f.score, f.target, tile_max, above)
                    });
                    let sha256 = hex(&bitstream);
                    let artifact = out.join("artifacts/jxl").join(format!("{sha256}.jxl"));
                    if !artifact.exists() {
                        std::fs::write(&artifact, &bitstream).expect("write bitstream");
                    }
                    let decoded = decode(&bitstream, im.width, im.height);
                    let (w, h) = (im.width as usize, im.height as usize);
                    let probe_values: Vec<(f64, f32, usize)> = probes
                        .iter()
                        .map(|p| {
                            let (values, score) = zensim_workbench::loop_map_values(
                                &p.metric,
                                &p.map,
                                &rgba(&im.rgb),
                                &rgba(&decoded),
                                im.width,
                                im.height,
                                w * 4,
                                p.scale,
                                p.exponent,
                            )
                            .unwrap_or_else(|e| panic!("probe {} {}: {e}", p.metric, p.map));
                            let tiles = tile_norms(&values, w, h);
                            let tile_max = tiles.iter().copied().fold(0f32, f32::max);
                            let above = tiles.iter().filter(|&&t| t > distance).count();
                            (score, tile_max, above)
                        })
                        .collect();
                    if let Some(dir) = &frames_dir {
                        write_frames(
                            &dir.join(format!("{label}__{}__d{distance:.2}.frames", im.name)),
                            &frames,
                        );
                    }
                    let dist_png = out
                        .join("decoded")
                        .join(format!("{label}__{}__d{distance:.2}.png", im.name));
                    write_png(&dist_png, &decoded, im.width, im.height);
                    eprintln!(
                        "  {label} {} d={distance:.2} bytes={} encode={encode_ms:.0}ms frames={}",
                        im.name,
                        bitstream.len(),
                        frames.len()
                    );
                    results.lock().expect("results").push(Encoded {
                        image: i,
                        distance: di,
                        bytes: bitstream.len(),
                        sha256,
                        encode_ms,
                        dist_png,
                        frames: frames.len(),
                        seeds: seeds.len(),
                        last,
                        probes: probe_values,
                    });
                }
            });
        }
    });
    let mut results = results.into_inner().expect("results");
    assert_eq!(results.len(), jobs.len(), "every encode finished");
    results.sort_by_key(|r| (r.image, r.distance));

    let mut manifest =
        String::from("ref_path\tdist_path\tlabel\timage\tcorpus\tdistance\tbytes\tencode_ms\n");
    let mut lp = String::from(
        "image\tclass\tdistance\tbytes\tbitstream_sha256\tencode_ms\tframes\tseeds\tlast_score\tlast_target\tlast_tile_max\tlast_blocks_above",
    );
    for p in &probes {
        let tag = format!("{}:{}@{:e},{}", p.metric, p.map, p.scale, p.exponent);
        let _ = write!(
            lp,
            "\tprobe_max[{tag}]\tprobe_tile_max[{tag}]\tprobe_above[{tag}]"
        );
    }
    lp.push('\n');
    for r in &results {
        let im = &images[r.image];
        let d = distances[r.distance];
        let ref_png = out.join("ref").join(format!("{}.png", im.name));
        let _ = writeln!(
            manifest,
            "{}\t{}\t{label}\t{}\t{}\t{d:.2}\t{}\t{:.1}",
            ref_png.display(),
            r.dist_png.display(),
            im.name,
            im.class,
            r.bytes,
            r.encode_ms
        );
        let (score, target, tile, above) = r.last.map_or(
            ("".into(), "".into(), "".into(), "".into()),
            |(s, t, m, a)| {
                (
                    format!("{s:.6}"),
                    format!("{t:.6}"),
                    format!("{m:.6}"),
                    a.to_string(),
                )
            },
        );
        let _ = write!(
            lp,
            "{}\t{}\t{d:.2}\t{}\t{}\t{:.1}\t{}\t{}\t{score}\t{target}\t{tile}\t{above}",
            im.name, im.class, r.bytes, r.sha256, r.encode_ms, r.frames, r.seeds
        );
        for (score, tile, above) in &r.probes {
            let _ = write!(lp, "\t{score:.6}\t{tile:.6}\t{above}");
        }
        lp.push('\n');
    }
    std::fs::write(out.join(format!("manifest_{label}.tsv")), manifest).expect("write manifest");
    std::fs::write(out.join(format!("loop_{label}.tsv")), lp).expect("write loop table");
    let placement = match &steer {
        Steer::Workbench {
            metric,
            map,
            scale,
            exponent,
        } => {
            serde_json::json!({"metric": metric, "map": map, "scale": scale, "exponent": exponent})
        }
        _ => serde_json::Value::Null,
    };
    let exe = std::fs::read("/proc/self/exe")
        .map(|b| hex(&b))
        .unwrap_or_default();
    let run = serde_json::json!({
        "label": label,
        "steer": steer_spec,
        "placement": placement,
        "server": if server.is_empty() { serde_json::Value::Null } else { serde_json::json!(server) },
        "server_header": header,
        "linear_input": linear || server_linear,
        "probes": probes.iter().map(|p| serde_json::json!({"metric": p.metric, "map": p.map, "scale": p.scale, "exponent": p.exponent})).collect::<Vec<_>>(),
        "frames_dir": frames_dir,
        "iters": iters,
        "effort": effort,
        "distances": distances,
        "corpus": corpus,
        "corpus_sha256": hex(corpus_text.as_bytes()),
        "images": images.len(),
        "encodes": results.len(),
        "threads": threads,
        "wall_seconds": started.elapsed().as_secs_f64(),
        "decoder": "zenjxl-decoder, dither_u8 on",
        "embedded_bakes": zensim_workbench::BAKES.iter().map(|b| serde_json::json!({"name": b.0, "sha256": b.2})).collect::<Vec<_>>(),
        "exe_sha256": exe,
    });
    std::fs::write(
        out.join(format!("run_{label}.json")),
        serde_json::to_string_pretty(&run).expect("json"),
    )
    .expect("write run record");
    eprintln!(
        "[steer_rd] {label}: {} encodes in {:.0} s",
        results.len(),
        started.elapsed().as_secs_f64()
    );
}
