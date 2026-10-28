//! Per-block L0 mass compare for the YJ-autotransforms vs v11-ship bakes
//! (task #214 Phase 2).
//!
//! Loads both bakes via zenpredict, computes
//! `importance[i] = scaler_scale[i] * Σ_h |L0[h, i]|` per input feature,
//! and aggregates into the canonical 372-feature block schema:
//!   - basic    : f0..f155   (156 features)
//!   - peak     : f156..f227  (72 features)
//!   - masked   : f228..f299  (72 features)
//!   - iw_pool  : f300..f371  (72 features)
//!
//! Emits a per-block table to stdout + TSV at the path passed via
//! the first positional arg (optional).
//!
//! Usage:
//!   cargo run -p zensim-validate --release --example l0_per_block_compare \
//!       [out.tsv]
//!
//! Bake paths are hard-coded (this is a one-shot Phase 2 verification tool).

use std::env;
use std::path::Path;

use zenpredict::{Model, WeightStorage};

fn block_for(idx: usize) -> &'static str {
    if idx < 156 {
        "basic"
    } else if idx < 228 {
        "peak"
    } else if idx < 300 {
        "masked"
    } else {
        "iw_pool"
    }
}

/// `importance[i] = scaler_scale[i] * Σ_h |L0[h, i]|`. Returns per-feature
/// importance for the first layer.
fn l0_importance(bake_path: &Path) -> Vec<f64> {
    let bytes = std::fs::read(bake_path).expect("read bake");
    let model = Model::from_bytes_with_schema(&bytes, 0).expect("parse bake");
    let n_inputs = model.n_inputs();
    let scaler_scale: Vec<f64> = model.scaler_scale().iter().map(|&s| s as f64).collect();
    assert_eq!(scaler_scale.len(), n_inputs);

    // Layer 0 weights, dequantized to f32 if needed.
    let layer0 = model.layer(0);
    let in_dim = layer0.in_dim;
    let out_dim = layer0.out_dim;
    assert_eq!(in_dim, n_inputs);

    let mut sum_abs_w = vec![0.0_f64; n_inputs];
    match &layer0.weights {
        WeightStorage::F32(w) => {
            assert_eq!(w.len(), in_dim * out_dim);
            for h in 0..out_dim {
                for i in 0..in_dim {
                    let v = w[h * in_dim + i] as f64;
                    sum_abs_w[i] += v.abs();
                }
            }
        }
        WeightStorage::F16(w) => {
            assert_eq!(w.len(), in_dim * out_dim);
            // Inline f16 → f64 (mimics IEEE 754 half precision).
            for h in 0..out_dim {
                for i in 0..in_dim {
                    let bits = w[h * in_dim + i];
                    let sign = (bits >> 15) & 1;
                    let exp = (bits >> 10) & 0x1f;
                    let mant = bits & 0x3ff;
                    let v_f32 = if exp == 0 {
                        let m = mant as f32 / 1024.0;
                        let s = if sign == 0 { 1.0 } else { -1.0 };
                        s * m * 2.0_f32.powi(-14)
                    } else if exp == 31 {
                        if mant == 0 {
                            if sign == 0 {
                                f32::INFINITY
                            } else {
                                f32::NEG_INFINITY
                            }
                        } else {
                            f32::NAN
                        }
                    } else {
                        let m = 1.0 + mant as f32 / 1024.0;
                        let s = if sign == 0 { 1.0 } else { -1.0 };
                        let e = exp as i32 - 15;
                        s * m * 2.0_f32.powi(e)
                    };
                    sum_abs_w[i] += (v_f32 as f64).abs();
                }
            }
        }
        WeightStorage::I8 { weights, scales } => {
            assert_eq!(weights.len(), in_dim * out_dim);
            // Scales are typically per-output-channel (`out_dim` values).
            // Dequant: w_f = w_i8 * scale[h] (per-channel).
            for h in 0..out_dim {
                let scale = scales[h] as f64;
                for i in 0..in_dim {
                    let v = weights[h * in_dim + i] as f64 * scale;
                    sum_abs_w[i] += v.abs();
                }
            }
        }
    }

    sum_abs_w
        .iter()
        .zip(scaler_scale.iter())
        .map(|(&w, &s)| s * w)
        .collect()
}

fn main() {
    let ship = Path::new(
        "/home/lilith/work/zen/zensimx/zensim-experimental/weights/v_tuner_v11_2026-05-24.bin",
    );
    let cand = Path::new(
        "/home/lilith/work/zen/zensimx/zensim-experimental/weights/v_tuner_v11_yj_autotransforms_2026-05-25.bin",
    );

    let ship_imp = l0_importance(ship);
    let cand_imp = l0_importance(cand);
    assert_eq!(ship_imp.len(), 372);
    assert_eq!(cand_imp.len(), 372);

    // Aggregate per block
    let blocks = ["basic", "peak", "masked", "iw_pool"];
    let mut ship_block = [0.0_f64; 4];
    let mut cand_block = [0.0_f64; 4];
    for i in 0..372 {
        let bi = match block_for(i) {
            "basic" => 0,
            "peak" => 1,
            "masked" => 2,
            "iw_pool" => 3,
            _ => unreachable!(),
        };
        ship_block[bi] += ship_imp[i];
        cand_block[bi] += cand_imp[i];
    }

    let ship_total: f64 = ship_block.iter().sum();
    let cand_total: f64 = cand_block.iter().sum();

    println!("Per-block L0 mass (importance = scaler_scale × Σ|W|):");
    println!(
        "{:<10} | {:>11} | {:>11} | {:>9} | {:>9} | {:>9}",
        "block", "ship", "candidate", "Δ (abs)", "ship %", "cand %"
    );
    println!("{:-<70}", "");
    for (bi, b) in blocks.iter().enumerate() {
        let sp = ship_block[bi] / ship_total * 100.0;
        let cp = cand_block[bi] / cand_total * 100.0;
        println!(
            "{:<10} | {:>11.4} | {:>11.4} | {:>+9.4} | {:>8.2}% | {:>8.2}%",
            b,
            ship_block[bi],
            cand_block[bi],
            cand_block[bi] - ship_block[bi],
            sp,
            cp,
        );
    }
    println!("{:-<70}", "");
    println!(
        "{:<10} | {:>11.4} | {:>11.4}",
        "total", ship_total, cand_total
    );
    println!();
    println!("Normalized share-of-mass Δ (percentage points):");
    println!(
        "{:<10} | {:>10} | {:>10} | {:>+8}",
        "block", "ship %", "cand %", "Δ pp"
    );
    for (bi, b) in blocks.iter().enumerate() {
        let sp = ship_block[bi] / ship_total * 100.0;
        let cp = cand_block[bi] / cand_total * 100.0;
        println!(
            "{:<10} | {:>9.2}% | {:>9.2}% | {:>+7.2}",
            b,
            sp,
            cp,
            cp - sp
        );
    }

    // TSV
    if let Some(out_arg) = env::args().nth(1) {
        let out_path = Path::new(&out_arg);
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent).expect("mkdir");
        }
        let mut tsv = String::new();
        tsv.push_str(
            "block\tship_mass\tcand_mass\tdelta_abs\tship_share_pct\tcand_share_pct\tdelta_pp\n",
        );
        for (bi, b) in blocks.iter().enumerate() {
            let sp = ship_block[bi] / ship_total * 100.0;
            let cp = cand_block[bi] / cand_total * 100.0;
            tsv.push_str(&format!(
                "{}\t{:.4}\t{:.4}\t{:+.4}\t{:.2}\t{:.2}\t{:+.2}\n",
                b,
                ship_block[bi],
                cand_block[bi],
                cand_block[bi] - ship_block[bi],
                sp,
                cp,
                cp - sp,
            ));
        }
        std::fs::write(out_path, tsv).expect("write tsv");
        eprintln!("wrote {}", out_path.display());
    }
}
