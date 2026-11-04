//! zenbench A/B: MLP forward time, dead-column-pruned bake vs its
//! un-pruned twin.
//!
//! Pruning's storage win is small once the ZNPR payload is LZ4-compressed
//! (a run of zero `f16` weights compresses to nearly nothing). The win
//! that compression *cannot* give you is inference: layer 0 dominates a
//! `944 → 128 → 1` net, and the forward kernel's `if s == 0.0 { continue }`
//! fast path does NOT fire on a dead column — the standardized input
//! `(x − mean)/scale` is generically nonzero even when the whole weight
//! row is zero, so an un-pruned bake pays 944 SAXPY rows to accumulate
//! 277 rows of zeros.
//!
//! Both arms run `Predictor::predict_transformed` over the SAME feature
//! rows, interleaved round-robin by zenbench so thermal/turbo drift hits
//! both equally.
//!
//! Usage:
//!   cargo run --release -p zensim-validate --example prune_forward_bench -- \
//!       <unpruned.bin> <pruned.bin> [rows] [results.json]
//! R12: prepend `--heads` to compare A-MLP against a candidate head, without pruning labels.
//!
//! Produce the pair with:
//!   bake_dial_refit pack --in X_dial.bin --out unpruned.bin --no-prune ...
//!   bake_dial_refit pack --in X_dial.bin --out pruned.bin ...

use std::env;

use zenpredict::{Model, Predictor, WeightStorage};

/// Deterministic pseudo-random feature rows. Values are irrelevant to
/// the timing (the kernel is branch-free per weight) as long as the
/// standardized inputs are generically nonzero, which is the whole
/// point — a zero input would let the `s == 0.0` fast path skip the
/// row and hide the cost pruning removes.
fn make_rows(n_rows: usize, width: usize, nonnegative: bool) -> Vec<Vec<f32>> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    (0..n_rows)
        .map(|_| {
            (0..width)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    // [-4, 4): a plausible standardized-feature range.
                    if nonnegative {
                        // R12 positive damage inputs; fixed synthetic bank, no extraction.
                        ((state >> 40) as f32 + 1.0) / 16_777_216.0
                    } else {
                        ((state >> 40) as f32 / 1024.0) - 4.0
                    }
                })
                .collect()
        })
        .collect()
}

fn main() {
    let mut args: Vec<String> = env::args().collect();
    if args.get(1).is_some_and(|s| s == "--inspect") {
        let model =
            Model::from_bytes(&std::fs::read(&args[2]).expect("read bake")).expect("parse bake");
        let layers: Vec<_> = (0..model.n_layers()).map(|i| {
            let layer = model.layer(i);
            let weights: Vec<f32> = match layer.weights {
                WeightStorage::F32(w) => w.to_vec(),
                WeightStorage::F16(w) => w.iter().map(|&v| zenpredict::f16_bits_to_f32(v)).collect(),
                WeightStorage::I8 { .. } => panic!("R12 inspection expects f32/f16"),
            };
            serde_json::json!({"in_dim":layer.in_dim,"out_dim":layer.out_dim,
                "activation":format!("{:?}",layer.activation),"weights":weights,"biases":layer.biases})
        }).collect();
        let spline = zensim_validate::output_calibration_spline::extract(&model)
            .map(|s| serde_json::json!({"xs":s.xs,"ys":s.ys,"derivs":s.derivs}));
        let doc = serde_json::json!({"scaler_mean":model.scaler_mean(),"scaler_scale":model.scaler_scale(),
            "layers":layers,"derived_inputs":model.metadata().get_utf8("zensim.derived_inputs").expect("derived declaration"),
            "spline":spline});
        std::fs::write(
            &args[3],
            serde_json::to_vec_pretty(&doc).expect("serialize"),
        )
        .expect("write inspection");
        return;
    }
    let heads = args.get(1).is_some_and(|s| s == "--heads");
    if heads {
        args.remove(1);
    }
    let a_label = if heads { "A-MLP" } else { "unpruned" };
    let b_label = if heads { "candidate" } else { "pruned" };
    if args.len() < 3 {
        eprintln!(
            "usage: prune_forward_bench <unpruned.bin> <pruned.bin> [rows]\n\
             both bakes must share a caller_input_width"
        );
        std::process::exit(2);
    }
    let n_rows: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);

    let a_bytes: &'static [u8] = Box::leak(
        std::fs::read(&args[1])
            .expect("read unpruned bake")
            .into_boxed_slice(),
    );
    let b_bytes: &'static [u8] = Box::leak(
        std::fs::read(&args[2])
            .expect("read pruned bake")
            .into_boxed_slice(),
    );
    let a_model: &'static Model = Box::leak(Box::new(
        Model::from_bytes(a_bytes).expect("parse unpruned"),
    ));
    let b_model: &'static Model =
        Box::leak(Box::new(Model::from_bytes(b_bytes).expect("parse pruned")));

    let width = a_model.caller_input_width();
    assert_eq!(
        width,
        b_model.caller_input_width(),
        "the two bakes must accept the same caller width — that is the whole contract"
    );
    eprintln!(
        "{a_label}: caller_width {width} layer0 {}x{} ({} B file)",
        a_model.layer(0).in_dim,
        a_model.layer(0).out_dim,
        a_bytes.len()
    );
    eprintln!(
        "{b_label}: caller_width {width} layer0 {}x{} ({} B file); rows={n_rows}",
        b_model.layer(0).in_dim,
        b_model.layer(0).out_dim,
        b_bytes.len()
    );

    let rows: &'static [Vec<f32>] = Box::leak(make_rows(n_rows, width, heads).into_boxed_slice());

    let result = zenbench::run(|suite| {
        suite.compare("predict_transformed", |group| {
            group.config().max_rounds(200);
            group.bench(a_label, move |b| {
                let mut p = Predictor::new(a_model);
                b.iter(move || {
                    let mut acc = 0f32;
                    for row in rows {
                        acc += p.predict_transformed(row).expect("forward")[0];
                    }
                    zenbench::black_box(acc);
                })
            });
            group.bench(b_label, move |b| {
                let mut p = Predictor::new(b_model);
                b.iter(move || {
                    let mut acc = 0f32;
                    for row in rows {
                        acc += p.predict_transformed(row).expect("forward")[0];
                    }
                    zenbench::black_box(acc);
                })
            });
        });
    });
    if let Some(path) = args.get(4) {
        result.save(path).expect("save retained rounds");
    }
}
