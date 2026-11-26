//! One-shot tool: print v1's full 372-feature vector for the phase-3 golden
//! fixtures, as a Rust array literal, for pasting into
//! `zensim/tests/v1_golden_bytes.rs`. Run ONCE on unmodified v1 code before
//! any v1-touching refactor; never run again except to deliberately update
//! the golden (which must never happen silently -- see that test's doc).
//!
//! ```sh
//! cargo run --release -p zensim --features training --example capture_v1_golden
//! ```

use zensim::{ZensimConfig, compute_zensim_with_config};

#[path = "../tests/common/generators.rs"]
mod generators;

fn v1_config() -> ZensimConfig {
    let mut cfg = ZensimConfig::default();
    cfg.compute_all_features = true;
    cfg.extended_features = true;
    cfg.compute_iw_features = true;
    cfg
}

fn print_vec(name: &str, v: &[f64]) {
    println!("pub const {name}: [f64; {}] = [", v.len());
    for chunk in v.chunks(4) {
        let line: Vec<String> = chunk.iter().map(|x| format!("{x:.17e}")).collect();
        println!("    {},", line.join(", "));
    }
    println!("];");
}

fn load_png_rgb8(path: &str) -> (Vec<[u8; 3]>, usize, usize) {
    use zenpixels::SampleType;
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let cfg = zenpng::PngDecodeConfig::default();
    let out = zenpng::decode(&bytes, &cfg, &enough::Unstoppable).expect("zenpng decode");
    let (w, h) = (out.info.width as usize, out.info.height as usize);
    let layout = out.pixels.format().layout;
    assert_eq!(layout.sample, SampleType::U8);
    let slice = out.pixels.as_view();
    let channels = layout.channels();
    let has_alpha = layout.order.has_alpha_slot();
    let mut rgb = Vec::with_capacity(w * h);
    for y in 0..h as u32 {
        let row = slice.row(y);
        match (channels, has_alpha) {
            (4, true) => {
                for px in row.as_chunks::<4>().0.iter().take(w) {
                    rgb.push([px[0], px[1], px[2]]);
                }
            }
            (3, false) => {
                for px in row.as_chunks::<3>().0.iter().take(w) {
                    rgb.push([px[0], px[1], px[2]]);
                }
            }
            other => panic!("unsupported PNG channel layout {other:?}"),
        }
    }
    (rgb, w, h)
}

fn main() {
    // --- Synthetic deterministic fixture (reuses tests/common/generators.rs
    //     -- already-tested, already-in-repo procedural generators; no new
    //     synthetic-image code). ---
    let (w, h) = (64, 64);
    let syn_ref = generators::gen_value_noise(w, h, 0xC0FFEE);
    let syn_dist = generators::distort_block_artifacts(&syn_ref, w, h);

    let cfg = v1_config();
    let syn_result =
        compute_zensim_with_config(&syn_ref, &syn_dist, w, h, cfg).expect("synthetic v1 compute");
    print_vec("GOLDEN_SYNTHETIC", syn_result.features());
    eprintln!(
        "synthetic: score={} n_features={}",
        syn_result.score(),
        syn_result.features().len()
    );

    // --- NON-TIGHT fixture (era-3 / option C, 2026-08-30). 200x150: under
    //     era-2 this width was padded to 208 and the phantom columns were
    //     POOLED, which is exactly the class the other two fixtures could not
    //     see (64 and 96 are both stride-invariant, so their goldens are
    //     UNCHANGED by C). Procedural, so no fixture file is needed. ---
    let (nw, nh) = (200usize, 150usize);
    let nt_ref = generators::gen_value_noise(nw, nh, 0xC0FFEE);
    let nt_dist = generators::distort_block_artifacts(&nt_ref, nw, nh);
    let nt_result = compute_zensim_with_config(&nt_ref, &nt_dist, nw, nh, v1_config())
        .expect("non-tight v1 compute");
    print_vec("GOLDEN_NONTIGHT", nt_result.features());
    eprintln!(
        "non-tight: {nw}x{nh} score={} n_features={}",
        nt_result.score(),
        nt_result.features().len()
    );

    // --- Real fixture (cropped from gb82 city.png / city_q50.jpg, committed
    //     at tests/fixtures/v1_golden_real_{ref,dist}.png, <10KB combined). ---
    let manifest = env!("CARGO_MANIFEST_DIR");
    let (real_ref, rw, rh) =
        load_png_rgb8(&format!("{manifest}/tests/fixtures/v1_golden_real_ref.png"));
    let (real_dist, dw, dh) = load_png_rgb8(&format!(
        "{manifest}/tests/fixtures/v1_golden_real_dist.png"
    ));
    assert_eq!((rw, rh), (dw, dh));
    let real_result = compute_zensim_with_config(&real_ref, &real_dist, rw, rh, v1_config())
        .expect("real v1 compute");
    print_vec("GOLDEN_REAL", real_result.features());
    eprintln!(
        "real: {}x{} score={} n_features={}",
        rw,
        rh,
        real_result.score(),
        real_result.features().len()
    );
}
