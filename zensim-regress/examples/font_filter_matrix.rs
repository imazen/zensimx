//! Render text through the font strip with a matrix of zenresize
//! filters (including sharpened kernels), one PGM per (filter, size).
//! Replicates the production per-cell path from `src/font.rs`: cells
//! are resized in isolation as RGBA (white, alpha = coverage) through
//! the `RGBA8_SRGB` pipeline, then composited white-over-black with a
//! gamma-correct (linear-light) blend.
//!
//! ```text
//! cargo run -p zensim-regress --example font_filter_matrix -- \
//!     /mnt/v/output/zensim-regress/sdf-explainer "Rag7" 12 18 27
//! ```
//!
//! Companion to `font_size_ladder` (which dumps the actual shipping
//! path — Mitchell); the `mitchell` output here should match it to
//! within LUT rounding, validating this harness.

use std::env;
use std::fs;
use std::io::Write as _;
use std::path::PathBuf;

const BASE_CHAR_W: u32 = 26;
const BASE_CHAR_H: u32 = 54;
const FIRST_CHAR: u32 = 0x20;
const CHAR_COUNT: u32 = 96;

static FONT_PNG: &[u8] = include_bytes!("../src/font_strip.png");

/// Mirror of `font::char_width_for_height`.
fn char_width_for_height(h: u32) -> u32 {
    (BASE_CHAR_W * h + BASE_CHAR_H / 2) / BASE_CHAR_H
}

fn srgb_encode(a: f32) -> u8 {
    let a = a.clamp(0.0, 1.0);
    let v = if a <= 0.003_130_8 {
        12.92 * a
    } else {
        1.055 * a.powf(1.0 / 2.4) - 0.055
    };
    (v * 255.0).round() as u8
}

fn main() {
    let mut args = env::args().skip(1);
    let out_dir = PathBuf::from(args.next().expect("arg 1: output dir"));
    let text = args.next().expect("arg 2: text to render");
    let sizes: Vec<u32> = args
        .map(|s| s.parse().expect("size args must be u32 px"))
        .collect();
    assert!(!sizes.is_empty(), "pass at least one size");
    fs::create_dir_all(&out_dir).expect("create out dir");

    let filters: &[(&str, zenresize::Filter)] = &[
        ("triangle", zenresize::Filter::Triangle),
        ("catmullrom", zenresize::Filter::CatmullRom),
        ("mitchell", zenresize::Filter::Mitchell),
        ("lanczos", zenresize::Filter::Lanczos),
        ("lanczos_sharp", zenresize::Filter::LanczosSharp),
        ("robidoux_sharp", zenresize::Filter::RobidouxSharp),
    ];

    // Decode the strip; coverage = R channel of the expanded RGBA.
    let strip = zensim_regress::Bitmap::from_png_bytes(FONT_PNG).expect("embedded strip decodes");
    assert_eq!(
        (strip.width(), strip.height()),
        (BASE_CHAR_W * CHAR_COUNT, BASE_CHAR_H)
    );

    // One isolated RGBA cell (white, alpha = coverage) per glyph index.
    let cell_rgba = |glyph: u32| -> Vec<u8> {
        let x0 = glyph * BASE_CHAR_W;
        let mut buf = Vec::with_capacity((BASE_CHAR_W * BASE_CHAR_H * 4) as usize);
        for y in 0..BASE_CHAR_H {
            for x in 0..BASE_CHAR_W {
                let cov = strip.get_pixel(x0 + x, y)[0];
                buf.extend_from_slice(&[255, 255, 255, cov]);
            }
        }
        buf
    };

    for &(name, filter) in filters {
        for &char_h in &sizes {
            let char_w = char_width_for_height(char_h).max(1);
            let cfg = zenresize::ResizeConfig::builder(BASE_CHAR_W, BASE_CHAR_H, char_w, char_h)
                .filter(filter)
                .format(zenresize::PixelFormat::RGBA8_SRGB)
                .build();
            let mut resizer = zenresize::Resizer::new(&cfg);

            let out_w = char_w * text.chars().count() as u32;
            let mut gray = vec![0u8; (out_w * char_h) as usize];
            for (i, ch) in text.chars().enumerate() {
                let glyph = (ch as u32).saturating_sub(FIRST_CHAR).min(CHAR_COUNT - 1);
                let scaled = resizer.resize(&cell_rgba(glyph));
                for y in 0..char_h as usize {
                    for x in 0..char_w as usize {
                        let a = scaled[(y * char_w as usize + x) * 4 + 3] as f32 / 255.0;
                        gray[y * out_w as usize + i * char_w as usize + x] = srgb_encode(a);
                    }
                }
            }

            let path = out_dir.join(format!("filtermx_{name}_{char_h}px.pgm"));
            let mut f = fs::File::create(&path).expect("create pgm");
            writeln!(f, "P5 {out_w} {char_h} 255").expect("header");
            f.write_all(&gray).expect("pixels");
            println!("{} ({out_w}x{char_h})", path.display());
        }
    }
}
