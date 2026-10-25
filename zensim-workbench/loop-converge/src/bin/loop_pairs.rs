//! Calibration pairs for the loop maps' placement on butteraugli's scale, in
//! the `--save-pairs` format of zenmetrics' `loop_norm_calibrate` (which
//! zensim-workbench's `examples/loop_calibrate.rs` reads), at any distances.
//! Each image is center-cropped to at most 512×512 with even sides, as there.
//! The distorted image and butteraugli's block distances come from the loop
//! itself: frame 0 of a one-iteration butteraugli loop at effort 7 is the
//! encode before any adjustment, as the loop reconstructs it (sRGB8, as the
//! workbench hands it to a steering metric) with the loop's own butteraugli
//! diffmap, reduced per 8×8 block by the 16-norm × 1.2.
//!
//! ```text
//! cargo run --release --manifest-path zensim-workbench/loop-converge/Cargo.toml --bin loop_pairs -- \
//!     OUT_DIR 0.1,0.2,0.3 --fit A.png B.png --hold C.png
//! ```
//!
//! Image paths are written relative to `CORPUS_ROOT` when they lie under it.

use jxl_encoder::{LoopTrace, LossyConfig, PixelLayout};
use std::fmt::Write as _;

fn srgb8(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let e = if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (e * 255.0 + 0.5) as u8
}

/// The image's center crop (at most 512×512, even sides) as RGBA8.
fn load(path: &str) -> (Vec<u8>, u32, u32) {
    let mut decoder = png::Decoder::new(std::io::BufReader::new(
        std::fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}")),
    ));
    // Palette and sub-byte images expand to 8-bit RGB(A); 16-bit strips.
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().expect("png header");
    let mut buf = vec![0; reader.output_buffer_size().expect("png size")];
    let info = reader.next_frame(&mut buf).expect("png frame");
    assert_eq!(
        info.bit_depth,
        png::BitDepth::Eight,
        "{path}: 8-bit PNG expected"
    );
    let channels = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        other => panic!("{path}: unsupported color type {other:?}"),
    };
    let (w, h) = (info.width, info.height);
    let (cw, ch) = (w.min(512) & !1, h.min(512) & !1);
    let (x0, y0) = ((w - cw) / 2, (h - ch) / 2);
    let mut rgba = Vec::with_capacity((cw * ch * 4) as usize);
    for y in y0..y0 + ch {
        for x in x0..x0 + cw {
            let i = ((y * w + x) as usize) * channels;
            rgba.extend([buf[i], buf[i + 1], buf[i + 2], 255]);
        }
    }
    (rgba, cw, ch)
}

/// Per 8×8 block: the 16-norm of `values` times 1.2 (zenmetrics'
/// `loop_tile_norms`, jxl-encoder's butteraugli tile reduction).
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
        .map(|(&s, &n)| 1.2 * (s / f64::from(n)).powf(1.0 / 16.0) as f32)
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 4 {
        eprintln!("usage: loop_pairs OUT_DIR DISTANCES --fit IMAGES... [--hold IMAGES...]");
        std::process::exit(2);
    }
    let out = &args[0];
    let distances: Vec<f32> = args[1]
        .split(',')
        .map(|d| d.parse().expect("distance"))
        .collect();
    let mut set = "";
    let mut images: Vec<(&str, &str)> = Vec::new();
    for a in &args[2..] {
        match a.as_str() {
            "--fit" => set = "fit",
            "--hold" => set = "hold",
            path => {
                assert!(!set.is_empty(), "images follow --fit or --hold");
                images.push((set, path));
            }
        }
    }
    let root = std::env::var("CORPUS_ROOT").unwrap_or_default();
    std::fs::create_dir_all(out).expect("out dir");
    let mut index = String::from("index\tset\timage\tdistance\twidth\theight\n");
    let mut i = 0;
    for (set, path) in images {
        let (reference, width, height) = load(path);
        let rgb: Vec<u8> = reference
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect();
        let name = path
            .strip_prefix(root.as_str())
            .map_or(path, |p| p.trim_start_matches('/'));
        for &distance in &distances {
            let trace = LoopTrace::new();
            let config = LossyConfig::new(distance)
                .with_effort(7)
                .with_butteraugli_iters(1);
            config
                .encode_request(width, height, PixelLayout::Rgb8)
                .with_loop_trace(&trace)
                .encode(&rgb)
                .expect("encode");
            let frames = trace.take();
            let f = frames.first().expect("frame 0");
            let [r, g, b] = &f.recon;
            let distorted: Vec<u8> = (0..r.len())
                .flat_map(|k| [srgb8(r[k]), srgb8(g[k]), srgb8(b[k]), 255])
                .collect();
            let butter = tile_norms(&f.diffmap, width as usize, height as usize);
            let file = |ext: &str| format!("{out}/{i:03}.{ext}");
            std::fs::write(file("ref.rgba"), &reference).expect("write");
            std::fs::write(file("dist.rgba"), &distorted).expect("write");
            let bytes: Vec<u8> = butter.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(file("butter.f32"), bytes).expect("write");
            writeln!(
                index,
                "{i:03}\t{set}\t{name}\t{distance}\t{width}\t{height}"
            )
            .unwrap();
            i += 1;
        }
        eprintln!("{name}: {} distances", distances.len());
    }
    std::fs::write(format!("{out}/pairs.tsv"), index).expect("write pairs.tsv");
}
