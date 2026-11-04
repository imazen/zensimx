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
            }
        }
    }
}
