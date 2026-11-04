#![cfg(feature = "real-bake-tests")]
use std::{error::Error, path::PathBuf, process::Command};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
#[allow(dead_code)]
#[path = "../export/pixels.rs"]
mod pixels;

#[test]
fn retained_export_matches_plain_extraction_on_odd_sizes_and_strided_rows() {
    let bake = std::env::var("VIZ_F_BAKE").expect("VIZ_F_BAKE is required");
    let bank = std::env::var("VIZ_F_BANK").expect("VIZ_F_BANK is required");
    let root = PathBuf::from(std::env::var("TMPDIR").expect("caller must set disk-backed TMPDIR"))
        .join(format!(
            "atlas-parity-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
    std::fs::create_dir_all(&root).unwrap();
    for (w, h) in [(65usize, 67usize), (79, 53), (129, 97)] {
        let pixels: Vec<_> = (0..h)
            .flat_map(|y| {
                (0..w).map(move |x| {
                    [
                        ((x * 7 + y * 3) % 256) as u8,
                        ((x / 8 * 31 + y * 5) % 256) as u8,
                        ((x * 13 + y / 8 * 17) % 256) as u8,
                    ]
                })
            })
            .collect();
        let a = pixels::RgbImage {
            pixels: pixels.clone(),
            width: w,
            height: h,
            stride: w,
        };
        let mut b = pixels::RgbImage {
            pixels,
            width: w,
            height: h,
            stride: w,
        };
        for y in h / 3..h * 2 / 3 {
            for x in w / 3..w * 2 / 3 {
                b.pixels[y * w + x][1] = b.pixels[y * w + x][1].saturating_add(23);
            }
        }
        let reference = root.join(format!("{w}x{h}-reference.png"));
        let distorted = root.join(format!("{w}x{h}-distorted.png"));
        pixels::write(&reference, &a).unwrap();
        pixels::write(&distorted, &b).unwrap();
        let mut tight_bits = None;
        for padding in [0, 5] {
            let out = root.join(format!("{w}x{h}-padding{padding}"));
            let result = Command::new(env!("CARGO_BIN_EXE_zensim-viz-export"))
                .arg("atlas")
                .arg(&bake)
                .arg("--bank")
                .arg(&bank)
                .arg("--reference")
                .arg(&reference)
                .arg("--distorted")
                .arg(&distorted)
                .arg("--out")
                .arg(&out)
                .arg("--max-rows")
                .arg("200")
                .arg("--row-padding")
                .arg(padding.to_string())
                .env("ZENSIM_FORMULA_REV", "7")
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            let meta: serde_json::Value =
                serde_json::from_slice(&std::fs::read(out.join("meta.json")).unwrap()).unwrap();
            assert_eq!(meta["pooled_bit_identity"], true);
            let control: serde_json::Value =
                serde_json::from_slice(&std::fs::read(out.join("control.json")).unwrap()).unwrap();
            let hooked: serde_json::Value =
                serde_json::from_slice(&std::fs::read(out.join("instrumented.json")).unwrap())
                    .unwrap();
            assert_eq!(control["bits"], hooked["bits"]);
            assert_eq!(meta["pooled_bits_checked"], control["emitted_slot_count"]);
            assert_eq!(
                meta["layout_bits_checked"],
                control["bits"].as_array().unwrap().len()
            );
            if let Some(ref bits) = tight_bits {
                assert_eq!(
                    bits, &control["bits"],
                    "stride padding changed feature bits at {w}x{h}"
                );
            } else {
                tight_bits = Some(control["bits"].clone());
            }
            let features = meta["features"].as_array().unwrap();
            for feature in features {
                let id = feature["id"].as_u64().unwrap() as usize;
                let expected = control["bits"][id]
                    .as_str()
                    .unwrap()
                    .parse::<u64>()
                    .unwrap();
                assert_eq!(
                    feature["pooled_value"].as_f64().unwrap().to_bits(),
                    expected,
                    "displayed pooled f{id} differs from plain extraction"
                );
                if !feature["fragility"].as_bool().unwrap() {
                    assert_eq!(feature["value"].as_f64().unwrap().to_bits(), expected);
                }
            }

            assert!(features.iter().any(|f| f["map_file"].is_string()));
            assert!(features.iter().all(|f| f["value"].is_number()));
            for f in features {
                let range = f["bank_plot_ranges"]
                    .as_object()
                    .expect("native bank p1–p99 required");
                assert!(!range.is_empty());
                for bounds in range.values() {
                    assert!(bounds[0].as_f64().unwrap() <= bounds[1].as_f64().unwrap());
                }
                if f["map_file"].is_string() {
                    let bounds = f["map_robust_range"]
                        .as_array()
                        .expect("native map p2–p98 required");
                    let lo = f["map_range"][0].as_f64().unwrap();
                    let hi = f["map_range"][1].as_f64().unwrap();
                    assert!(lo <= bounds[0].as_f64().unwrap());
                    assert!(bounds[0].as_f64().unwrap() <= bounds[1].as_f64().unwrap());
                    assert!(bounds[1].as_f64().unwrap() <= hi);
                }
            }
            let map_records: Vec<serde_json::Value> =
                serde_json::from_slice(&std::fs::read(out.join("maps/maps.json")).unwrap())
                    .unwrap();
            for f in features.iter().filter(|f| f["map_file"].is_string()) {
                let image = pixels::decode(&out.join(f["map_file"].as_str().unwrap())).unwrap();
                assert!(image.width <= 512 && image.height <= 512);
                assert!(
                    f["map_range"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .all(|v| v.is_number())
                );
                if f["id"].as_u64().unwrap() < 156 {
                    let header_path =
                        out.join(f["terms_file"].as_str().expect("basic terms required"));
                    let header: serde_json::Value =
                        serde_json::from_slice(&std::fs::read(&header_path).unwrap()).unwrap();
                    assert_eq!(header["formula_replay_bit_identity"], true);
                    assert_eq!(
                        header["formula_replay_bits_checked"].as_u64().unwrap() as usize,
                        image.width * image.height
                    );
                    assert_eq!(header["native_grid"], f["native_grid"]);
                    let bytes = std::fs::read(
                        header_path
                            .parent()
                            .unwrap()
                            .join(header["data_file"].as_str().unwrap()),
                    )
                    .unwrap();
                    let names = header["planes"].as_array().unwrap();
                    assert_eq!(bytes.len(), image.width * image.height * names.len() * 4);
                    let local = f["id"].as_u64().unwrap() % 13;
                    let result = match local {
                        0..=2 => "d1",
                        3..=5 => "artifact",
                        6..=8 => "detail",
                        9 => "mse",
                        _ => unreachable!(),
                    };
                    let plane = names.iter().position(|name| name == result).unwrap();
                    let n = image.width * image.height;
                    // Product inputs share the raw detector's normalized PNG;
                    // their display range is multiplied by reference fragility.
                    // Terms describe the raw local formula, before that product.
                    let raw = map_records.iter().find(|r| r["id"] == f["id"]).unwrap();
                    let lo = raw["map_range"][0].as_f64().unwrap() as f32;
                    let hi = raw["map_range"][1].as_f64().unwrap() as f32;
                    if f["fragility"].as_bool().unwrap() {
                        let fragility = f["fragility_value"].as_f64().unwrap();
                        for k in 0..2 {
                            assert_eq!(
                                f["map_range"][k].as_f64().unwrap().to_bits(),
                                (raw["map_range"][k].as_f64().unwrap() * fragility).to_bits()
                            );
                        }
                    }
                    for (i, pixel) in image.pixels.iter().enumerate() {
                        let offset = (plane * n + i) * 4;
                        let value =
                            f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                        let q = if hi == lo {
                            0
                        } else {
                            ((value - lo) / (hi - lo) * 255.0).round() as u8
                        };
                        assert_eq!(
                            *pixel, [q; 3],
                            "basic map/term disagreement f{} at {i}: value={value:?}, range={lo:?}..{hi:?}, image={}x{}, header={header_path:?}",
                            f["id"], image.width, image.height
                        );
                    }
                }
            }
        }
    }
}
