//! Explicit, label-free E34 pixel preparation. Decode/resize/PNG I/O and
//! feature extraction stay in their existing native owners. Outputs are fresh.
#[path = "support/zen_io.rs"]
mod zen_io;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path};
use zensim::{BakeScorer, RgbSlice};

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}
fn text<'a>(v: &'a Value, key: &str) -> &'a str {
    v[key].as_str().unwrap()
}
fn read(v: &Value, key: &str) -> (Vec<[u8; 3]>, usize, usize) {
    let path = Path::new(text(v, key));
    assert_eq!(
        sha(&fs::read(path).unwrap()),
        text(v, &format!("{key}_sha256"))
    );
    zen_io::decode_rgb8_with_max_pixels(path, v.get("max_decode_pixels").and_then(Value::as_u64))
}
fn source_orientation(bytes: &[u8]) -> Option<zenpixels::Orientation> {
    let parsed = zencodec::Exif::parse(bytes)?;
    if let Some(orientation) = parsed.orientation() {
        return Some(orientation);
    }
    // Distinguish valid absence from an invalid orientation tag through the
    // structured native parser/filter, without a second TIFF scanner.
    let orientation_only = zenmeta::exif::serialize(&zenmeta::exif::filter(
        &parsed,
        &zencodec::ExifPolicy::ORIENTATION_ONLY,
    ));
    let without_tags = zenmeta::exif::serialize(&zenmeta::exif::filter(
        &parsed,
        &zencodec::ExifPolicy::DISCARD_ALL,
    ));
    (orientation_only == without_tags).then_some(zenpixels::Orientation::Identity)
}

#[cfg(test)]
fn source_orientation_is_identity(bytes: &[u8]) -> bool {
    source_orientation(bytes).is_some_and(|o| o.is_identity())
}

// Metadata admission is shared by probing and pixel normalization.
fn source_plan(
    info: &zenpng::PngInfo,
) -> Option<(zenpixels::ColorPrimaries, zenpixels::Orientation)> {
    use zenpixels::ColorPrimaries;
    if info.has_alpha {
        return None;
    } // E36 excludes unqualified matte conversion.
    let orientation = match info.exif.as_deref() {
        Some(bytes) => source_orientation(bytes)?,
        None => zenpixels::Orientation::Identity,
    };
    if info.source_gamma.is_some() || info.chromaticities.is_some() {
        return None;
    }
    let mut primaries = ColorPrimaries::Bt709;
    if let Some(c) = info.cicp {
        if info.icc_profile.is_some()
            || (
                c.transfer_characteristics,
                c.matrix_coefficients,
                c.full_range,
            ) != (13, 0, true)
        {
            return None;
        }
        primaries = match c.color_primaries {
            1 => ColorPrimaries::Bt709,
            12 => ColorPrimaries::DisplayP3,
            _ => return None,
        };
    }
    if let Some(icc) = &info.icc_profile {
        let id = zenpixels::icc::identify_common(icc)?;
        if !id.is_srgb()
            || !matches!(
                id.valid_use,
                zenpixels::icc::IdentificationUse::MatrixTrcSubstitution
            )
        {
            return None;
        }
    }
    Some((primaries, orientation))
}

// Explicit E36 SDR ingress delegates pixel math to existing native owners.
fn read_srgb_opaque(v: &Value) -> (Vec<[u8; 3]>, usize, usize) {
    use zenpixels::{PixelDescriptor, TransferFunction};
    use zenpixels_convert::{AlphaPolicy, ConvertOptions, RowConverter};
    let bytes = fs::read(text(v, "source")).unwrap();
    assert_eq!(sha(&bytes), text(v, "source_sha256"));
    let cfg = zenpng::PngDecodeConfig::default().with_max_pixels(120_000_000);
    let decoded = zenpng::decode(&bytes, &cfg, &enough::Unstoppable).unwrap();
    let info = &decoded.info;
    let (primaries, orientation) =
        source_plan(info).expect("source color/orientation admission refused");
    // Explicit caller contract: absent profile/gamma/chromaticities means sRGB.
    // All original metadata remains in the hashed probe receipt.
    let oriented = (!orientation.is_identity()).then(|| {
        zenpixels_convert::orient::apply_orientation(decoded.pixels.as_slice(), orientation)
    });
    let pixels = oriented.as_ref().unwrap_or(&decoded.pixels);
    let desc = pixels
        .descriptor()
        .with_primaries(primaries)
        .with_transfer(TransferFunction::Srgb);
    let options = ConvertOptions::permissive().with_alpha_policy(AlphaPolicy::Forbid);
    let mut converter =
        RowConverter::new_explicit(desc, PixelDescriptor::RGB8_SRGB, &options).unwrap();
    let (ow, oh) = orientation.output_dimensions(info.width, info.height);
    let (w, h) = (ow as usize, oh as usize);
    let mut rgb = vec![[0u8; 3]; w * h];
    let input = pixels.as_slice();
    for y in 0..h {
        converter.convert_row(
            input.row(y as u32),
            bytemuck::cast_slice_mut(&mut rgb[y * w..(y + 1) * w]),
            w as u32,
        );
    }
    (rgb, w, h)
}

fn fresh(path: &str) -> fs::File {
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap()
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 2, "e34_pixels JOB.json");
    let job: Value = serde_json::from_slice(&fs::read(&args[1]).unwrap()).unwrap();
    let mut output = fresh(text(&job, "output"));
    match text(&job, "mode") {
        "probe" => {
            for row in job["rows"].as_array().unwrap() {
                let bytes = fs::read(text(row, "source")).unwrap();
                assert_eq!(sha(&bytes), text(row, "source_sha256"));
                let info = zenpng::probe(&bytes).unwrap();
                writeln!(
                    output,
                    "{}",
                    json!({
                        "source": row["source"], "source_sha256": sha(&bytes),
                        "width": info.width, "height": info.height,
                        "bit_depth": info.bit_depth, "has_alpha": info.has_alpha,
                        "icc_sha256": info.icc_profile.as_ref().map(|v| sha(v)),
                    "icc_native_identification": info.icc_profile.as_ref().map(|v| format!("{:?}", zenpixels::icc::identify_common(v))),
                        "srgb_intent": info.srgb_intent,
                        "source_gamma": info.source_gamma,
                        "cicp": format!("{:?}", info.cicp),
                        "chromaticities": format!("{:?}", info.chromaticities),
                        "exif_sha256": info.exif.as_ref().map(|v| sha(v)),
                    "exif_native_orientation": info.exif.as_deref().map(|v| format!("{:?}", source_orientation(v))),
                    "srgb_opaque_admitted": source_plan(&info).is_some()
                    })
                )
                .unwrap();
            }
        }
        "normalize" => {
            for row in job["rows"].as_array().unwrap() {
                let (mut pixels, mut w, mut h) = match job["color_policy"].as_str() {
                    Some("srgb-opaque-v1") => read_srgb_opaque(row),
                    None => read(row, "source"), // historical E34 caller contract
                    Some(other) => panic!("unregistered E36 color policy: {other}"),
                };
                let edge = job["longest_side"].as_u64().unwrap() as usize;
                if edge > 0 {
                    if job["never_upscale"].as_bool().unwrap_or(false) {
                        assert!(edge <= w.max(h), "registered no-upscale boundary");
                    }
                    let factor = edge as f64 / w.max(h) as f64;
                    let nw = (w as f64 * factor).round().max(1.) as usize;
                    let nh = (h as f64 * factor).round().max(1.) as usize;
                    pixels = zen_io::resize_rgb8(&pixels, w, h, nw, nh);
                    w = nw;
                    h = nh;
                }
                let bytes = zen_io::encode_png_rgb8(&pixels, w, h);
                fresh(text(row, "destination")).write_all(&bytes).unwrap();
                writeln!(output,"{}",json!({"source":row["source"],"source_sha256":row["source_sha256"],
                "destination":row["destination"],"sha256":sha(&bytes),"pixels_sha256":sha(bytemuck::cast_slice(&pixels)),
                "width":w,"height":h,"color_policy":job["color_policy"]})).unwrap();
            }
        }
        "features" | "repairs" => {
            assert_eq!(std::env::var("ZENSIM_FORMULA_REV").as_deref(), Ok("5"));
            assert_eq!(std::env::var("ZENSIM_MAX_TIER").as_deref(), Ok("v3"));
            #[cfg(target_arch = "x86_64")]
            {
                archmage::X64V4xToken::dangerously_disable_token_process_wide(true).unwrap();
                archmage::X64V4Token::dangerously_disable_token_process_wide(true).unwrap();
                eprintln!("E34 extraction: v4/v4x disabled, registered v3 cap");
            }
            let blob = fs::read(text(&job, "model")).unwrap();
            assert_eq!(sha(&blob), text(&job, "model_sha256"));
            let model = zenpredict::Model::from_bytes(&blob).unwrap();
            let mut scorer = BakeScorer::new(&model).unwrap().with_parallel(false);
            let ids: Vec<usize> = job["feature_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            let width = job["width"].as_u64().unwrap() as usize;
            let extract = |features: &[f64]| -> Vec<f32> {
                let mut row = vec![0.; width];
                for &id in &ids {
                    assert!(features[id].is_finite());
                    row[id] = features[id] as f32;
                }
                row
            };
            for row in job["rows"].as_array().unwrap() {
                let (r, w, h) = read(row, "reference");
                let (d, dw, dh) = read(row, "distorted");
                assert_eq!((w, h), (dw, dh));
                if text(&job, "mode") == "repairs" {
                    assert_eq!(w.max(h), 256);
                }
                let rs = RgbSlice::new(&r, w, h);
                let base = extract(
                    scorer
                        .compute(&rs, &RgbSlice::new(&d, w, h), None)
                        .unwrap()
                        .features(),
                );
                if text(&job, "mode") == "features" {
                    let mut pixel_pair = Sha256::new();
                    pixel_pair.update(bytemuck::cast_slice(&r));
                    pixel_pair.update(bytemuck::cast_slice(&d));
                    let mut metric_keys = serde_json::Map::new();
                    for id in job["metric_ids"].as_array().unwrap() {
                        let id = id.as_str().unwrap();
                        let mut hash = pixel_pair.clone();
                        hash.update(id.as_bytes());
                        let key: String =
                            hash.finalize().iter().map(|v| format!("{v:02x}")).collect();
                        metric_keys.insert(id.to_owned(), json!(key));
                    }
                    writeln!(
                        output,
                        "{}",
                        json!({
                            "key": row["key"], "reference_sha256": row["reference_sha256"],
                            "distorted_sha256": row["distorted_sha256"],
                            "width": w, "height": h, "features": base,
                        "reference_pixels_sha256": sha(bytemuck::cast_slice(&r)),
                        "distorted_pixels_sha256": sha(bytemuck::cast_slice(&d)),
                        "metric_pixel_keys": metric_keys
                        })
                    )
                    .unwrap();
                    output.flush().unwrap();
                    continue;
                }
                let mut repaired = d.clone();
                for block in row["blocks"].as_array().unwrap() {
                    let b: Vec<usize> = block
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_u64().unwrap() as usize)
                        .collect();
                    assert_eq!(b.len(), 3);
                    let (x, y, size) = (b[0], b[1], b[2]);
                    assert!(
                        [8, 16, 32, 64].contains(&size)
                            && x < w
                            && y < h
                            && x % size == 0
                            && y % size == 0
                    );
                    for yy in y..(y + size).min(h) {
                        let range = yy * w + x..yy * w + (x + size).min(w);
                        repaired[range.clone()].copy_from_slice(&r[range]);
                    }
                    if let Some(path) = row.get("parity_png").and_then(Value::as_str) {
                        assert_eq!(row["blocks"].as_array().unwrap().len(), 1);
                        fresh(path)
                            .write_all(&zen_io::encode_png_rgb8(&repaired, w, h))
                            .unwrap();
                    }
                    let features = extract(
                        scorer
                            .compute(&rs, &RgbSlice::new(&repaired, w, h), None)
                            .unwrap()
                            .features(),
                    );
                    let identical = base
                        .iter()
                        .zip(&features)
                        .all(|(a, b)| a.to_bits() == b.to_bits());
                    writeln!(output,"{}",json!({"base_id":row["base_id"],"block_size":size,"block":[x,y],
                        "base_features":base,"repair_features":features,"drop_bit_identical":identical})).unwrap();
                    for yy in y..(y + size).min(h) {
                        let range = yy * w + x..yy * w + (x + size).min(w);
                        repaired[range.clone()].copy_from_slice(&d[range]);
                    }
                }
                output.flush().unwrap();
            }
        }
        _ => panic!("unknown pixel preparation mode"),
    }
}

#[cfg(test)]
mod tests {
    use super::source_orientation_is_identity;
    #[test]
    fn real_native_png_alpha_is_refused_for_e36() {
        let rgba = [rgb::RGBA {
            r: 120u8,
            g: 40,
            b: 10,
            a: 128,
        }; 4];
        let bytes = zenpng::encode_rgba8(
            imgref::ImgRef::new(&rgba, 2, 2),
            None,
            &zenpng::EncodeConfig::default(),
            &enough::Unstoppable,
            &enough::Unstoppable,
        )
        .unwrap();
        let info = zenpng::probe(&bytes).unwrap();
        assert!(info.has_alpha);
        assert!(super::source_plan(&info).is_none());
        let opaque = super::zen_io::encode_png_rgb8(&[[120, 40, 10]; 4], 2, 2);
        assert!(super::source_plan(&zenpng::probe(&opaque).unwrap()).is_some());
    }

    #[test]
    fn native_exif_absence_identity_rotation_and_malformed() {
        let mut exif = zencodec::Exif::new(zencodec::exif::TextEncoding::Ascii);
        assert!(source_orientation_is_identity(&zenmeta::exif::serialize(
            &exif
        )));
        exif.set_orientation(zenpixels::Orientation::Rotate90);
        assert!(!source_orientation_is_identity(&zenmeta::exif::serialize(
            &exif
        )));
        exif.set_orientation(zenpixels::Orientation::Identity);
        assert!(source_orientation_is_identity(&zenmeta::exif::serialize(
            &exif
        )));
        assert!(!source_orientation_is_identity(b"malformed"));
    }
}
