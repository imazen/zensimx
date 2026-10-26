//! Fixed-work counter probe of the actual trainer kernels; no duplicate Adam.
#![forbid(unsafe_code)]

#[allow(dead_code)]
#[path = "../../../zensim-validate/src/adam_simd.rs"]
mod adam_simd;

use adam_simd::{AdamUpdateArgs, AdamW1FusedArgs, adam_update, adam_update_w1_fused};
use std::hint::black_box;

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    if matches!(
        argv.get(1).map(String::as_str),
        Some("--repro" | "--normalize-repro" | "--inspect")
    ) {
        use std::io::Write;
        assert_eq!(
            argv.len(),
            4,
            "usage: probe --repro|--normalize-repro|--inspect MODEL OUTPUT"
        );
        let bytes = std::fs::read(&argv[2]).expect("read model");
        let model =
            zenpredict::Model::from_bytes(&bytes).expect("parse model with the native owner");
        let metadata = model.metadata();
        let repro = metadata
            .get_utf8("zentrain.repro")
            .expect("embedded trainer provenance");
        let output = if argv[1] == "--inspect" {
            use std::fmt::Write;
            let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            let mut out = format!(
                "scaler_mean={:?}\nscaler_scale={:?}\nfeature_bounds={:?}\noutput_specs={:?}\nfeature_columns={:?}\nfeature_transforms={:?}\nfeature_transform_params={:?}\n",
                bits(model.scaler_mean()),
                bits(model.scaler_scale()),
                model.feature_bounds(),
                model.output_specs(),
                model.feature_columns().collect::<Vec<_>>(),
                model.feature_transforms(),
                model.feature_transform_params(),
            );
            for (index, layer) in model.layers().enumerate() {
                use zenpredict::WeightStorage;
                let weights = match layer.weights {
                    WeightStorage::F32(values) => format!("F32:{:?}", bits(values)),
                    WeightStorage::F16(values) => format!("F16:{values:?}"),
                    WeightStorage::I8 { weights, scales } => {
                        format!("I8:{weights:?}:{:?}", bits(scales))
                    }
                };
                writeln!(
                    &mut out,
                    "layer:{index}={},{},{:?};{weights};biases:{:?}",
                    layer.in_dim,
                    layer.out_dim,
                    layer.activation,
                    bits(layer.biases)
                )
                .unwrap();
            }
            for entry in metadata
                .iter()
                .filter(|entry| entry.key != "zentrain.repro")
            {
                write!(&mut out, "metadata:{}={:?}:", entry.key, entry.kind).unwrap();
                for byte in entry.value {
                    write!(&mut out, "{byte:02x}").unwrap();
                }
                out.push('\n');
            }
            out.into_bytes()
        } else if argv[1] == "--repro" {
            repro.as_bytes().to_vec()
        } else {
            // The owner's splice preserves every non-metadata section byte,
            // including packed weight bits and permutation sections.
            zenpredict_bake::append_metadata_utf8(&bytes, "zentrain.repro", "{}")
                .expect("replace provenance without rebaking model content")
        };
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&argv[3])
            .expect("fresh provenance output")
            .write_all(&output)
            .expect("write provenance");
        return;
    }
    assert!(
        matches!(argv.len(), 5 | 6),
        "usage: probe fused|unfused normal|normal-l2|decay|tiny-l2|tiny-moment|subnormal-weight STEPS v3|scalar|native [lazy|touched]"
    );
    let steps: usize = argv[3].parse().unwrap();
    let _guard = archmage::testing::lock_token_testing();
    #[cfg(target_arch = "x86_64")]
    match argv[4].as_str() {
        "v3" => archmage::X64V4Token::dangerously_disable_token_process_wide(true).unwrap(),
        "scalar" => {
            archmage::X64V4Token::dangerously_disable_token_process_wide(true).unwrap();
            archmage::X64V3Token::dangerously_disable_token_process_wide(true).unwrap();
        }
        "native" => {}
        _ => panic!("unknown tier"),
    }
    let n = 820 * 128;
    let mut w = vec![
        match argv[2].as_str() {
            "tiny-l2" => 1e-304,
            "tiny-moment" => 1e-302,
            "subnormal-weight" => 1e-310,
            _ => 0.5,
        };
        n
    ];
    let mut g = vec![0.0; n];
    let initial = match argv[2].as_str() {
        "normal" | "normal-l2" => 0.01,
        "decay" => f64::MIN_POSITIVE * 2.0,
        "tiny-l2" | "tiny-moment" | "subnormal-weight" => 0.0,
        _ => panic!("unknown state"),
    };
    let mut m = vec![initial; n];
    let mut v = vec![initial; n];
    let pages = argv.get(5).map(String::as_str).unwrap_or("lazy");
    match pages {
        "touched" => {
            // Force actual writes even when calloc supplied shared zero pages.
            // Training normally touches moment pages before they decay to zero.
            for value in m.iter_mut().chain(v.iter_mut()) {
                *black_box(value) = black_box(initial);
            }
        }
        "lazy" => {}
        _ => panic!("unknown moment-page setup"),
    }
    let x = vec![0.0; 820];
    let dh = vec![0.0; 128];
    assert!(argv[2] != "normal-l2" || argv[1] == "fused");
    for _ in 0..steps {
        match argv[1].as_str() {
            "fused" => adam_update_w1_fused(&mut AdamW1FusedArgs {
                w: &mut w,
                g: &mut g,
                m: &mut m,
                v: &mut v,
                xa: &x,
                xb: &x,
                dha: &dh,
                dhb: &dh,
                n_hidden: 128,
                l2_scale: if matches!(
                    argv[2].as_str(),
                    "normal-l2" | "tiny-l2" | "tiny-moment" | "subnormal-weight"
                ) {
                    1e-5
                } else {
                    0.0
                },
                l2_mult: None,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                bc1: 1.0,
                bc2: 1.0,
                lr: 0.005,
                active_rows: None,
                group_l1_tau: None,
                g_zero_in: true,
                fwd: None,
            }),
            "unfused" => adam_update(&mut AdamUpdateArgs {
                w: &mut w,
                g: &mut g,
                m: &mut m,
                v: &mut v,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                bc1: 1.0,
                bc2: 1.0,
                lr: 0.005,
            }),
            _ => panic!("unknown kernel"),
        }
        black_box((&w, &m, &v));
    }
    println!(
        "kernel={} state={} steps={steps} tier={} pages={pages} params={n} w_bits={:016x} m_bits={:016x} v_bits={:016x}",
        argv[1],
        argv[2],
        argv[4],
        w[0].to_bits(),
        m[0].to_bits(),
        v[0].to_bits()
    );
}
