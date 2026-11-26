//! Score the UPIQ HDR subset with zensim's PU21 HDR front-end and emit a CSV
//! the `scripts/upiq_eval.py` harness can validate against the JOD truth.
//!
//! UPIQ EXRs are stored in absolute photometric units (cd/m²), so they feed
//! the named-profile `Zensim::compute_pu_linear_planar` control, or the complete
//! `BakeScorer::compute_hdr` composition with linear BT.709 nits. Decode uses
//! zenexr (the upstream Rust EXR implementation), preserving float precision.
//! Output: `condition_id,zensim_hdr` per scored pair; decode failures abort.
//!
//! Usage:
//!   upiq_pu_score \
//!     --images /mnt/v/datasets/upiq_extracted/upiq_dataset/images \
//!     --subjective /mnt/v/datasets/upiq/upiq_subjective_scores.csv \
//!     --out /tmp/zensim_hdr_scores.csv \
//!     [--corpus narwaria,korshunov] [--profile bhdr]
//!
//! Frozen candidate serving requires `--composition manifest.json` and
//! `--input-contract upiq-exr-bt709-nits-v1`; the manifest binds all member
//! hashes and weights. The photometric contract belongs to this corpus, not
//! to arbitrary EXR files.

use std::collections::HashMap;
use std::path::Path;

use zensim::source::{AlphaMode, ImageSource, PixelFormat};
use zensim::{Zensim, ZensimProfile};

#[path = "upiq_pu_score/ingest.rs"]
mod ingest;

struct Rgb {
    w: usize,
    h: usize,
    r: Vec<f32>,
    g: Vec<f32>,
    b: Vec<f32>,
    rgba: Vec<u8>,
}

impl ImageSource for Rgb {
    fn width(&self) -> usize {
        self.w
    }
    fn height(&self) -> usize {
        self.h
    }
    fn pixel_format(&self) -> PixelFormat {
        PixelFormat::LinearF32Rgba
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }
    fn is_hdr(&self) -> bool {
        true
    }
    fn row_bytes(&self, y: usize) -> &[u8] {
        &self.rgba[y * self.w * 16..(y + 1) * self.w * 16]
    }
}

/// Load an absolute-luminance EXR into planar f32 RGB (values preserved, HDR
/// magnitudes > 1 kept). Uses zenexr's upstream Rust exr decoder. The UPIQ
/// contract is absolute nits in BT.709; conflicting color or alpha is refused.
fn load_exr_rgb(path: &Path) -> Result<Rgb, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    decode_exr_rgb(path, &bytes)
}

fn decode_exr_rgb(path: &Path, bytes: &[u8]) -> Result<Rgb, String> {
    let decoded = zenexr::ExrDecoderConfig::new()
        .decode(bytes, &enough::Unstoppable)
        .map_err(|e| e.to_string())?;
    let pixels = decoded.pixels();
    if !matches!(
        pixels.spec().color,
        zenpixels::ColorSpace::Rgb {
            primaries: zenpixels::Primaries::Bt709,
            ..
        }
    ) {
        return Err(format!(
            "{path:?}: non-BT709/unknown EXR chromaticities require an explicit color contract"
        ));
    }
    let channels = if pixels.spec().layout.order.has_alpha_slot() {
        4
    } else {
        3
    };
    let (w, h) = (pixels.width() as usize, pixels.height() as usize);
    let mut raw = Vec::with_capacity(w * h * 3);
    let view = pixels.as_view();
    for y in 0..h {
        for p in view.row(y as u32).chunks_exact(channels * 4) {
            let values: [f32; 3] = core::array::from_fn(|c| {
                f32::from_ne_bytes(p[c * 4..c * 4 + 4].try_into().unwrap())
            });
            if !values.iter().all(|v| v.is_finite())
                || (channels == 4 && f32::from_ne_bytes(p[12..16].try_into().unwrap()) != 1.)
            {
                return Err("UPIQ requires finite opaque samples".into());
            }
            raw.extend_from_slice(&values[..3]);
        }
    }
    let n = w * h;
    let mut r = vec![0.0f32; n];
    let mut g = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    let mut rgba = Vec::with_capacity(n * 16);
    for i in 0..n {
        r[i] = raw[3 * i];
        g[i] = raw[3 * i + 1];
        b[i] = raw[3 * i + 2];
        for value in [r[i], g[i], b[i], 1.] {
            rgba.extend_from_slice(&value.to_ne_bytes());
        }
    }
    Ok(Rgb {
        w,
        h,
        r,
        g,
        b,
        rgba,
    })
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if ingest::requested(&args) {
        ingest::run(&args).expect("UPIQ-380 allowlisted extraction");
        return;
    }
    let images = arg(&args, "--images")
        .unwrap_or_else(|| "/mnt/v/datasets/upiq_extracted/upiq_dataset/images".into());
    let subjective = arg(&args, "--subjective")
        .unwrap_or_else(|| "/mnt/v/datasets/upiq/upiq_subjective_scores.csv".into());
    let out = arg(&args, "--out").unwrap_or_else(|| "/tmp/zensim_hdr_scores.csv".into());
    let corpora: Vec<String> = arg(&args, "--corpus")
        .map(|s| s.split(',').map(str::to_string).collect())
        .unwrap_or_else(|| vec!["narwaria".into(), "korshunov".into()]);
    // HDR subset is currently always on (the former `--hdr-only` flag was a
    // no-op: `... || true`); a future `--sdr` toggle can reintroduce a parse.
    let hdr_only = true;
    let composition = arg(&args, "--composition").map(|p| {
        serde_json::from_slice::<serde_json::Value>(&std::fs::read(p).expect("composition"))
            .expect("composition JSON")
    });
    if composition.is_some() {
        assert_eq!(
            arg(&args, "--input-contract").as_deref(),
            Some("upiq-exr-bt709-nits-v1"),
            "candidate requires explicit UPIQ nits contract"
        );
    }
    let model_bytes: Vec<Vec<u8>> = composition.as_ref().map_or_else(Vec::new, |c| {
        use sha2::{Digest, Sha256};
        c["members"]
            .as_array()
            .expect("members")
            .iter()
            .map(|m| {
                let b = std::fs::read(m["path"].as_str().expect("path")).expect("model");
                let h: String = Sha256::digest(&b)
                    .iter()
                    .map(|x| format!("{x:02x}"))
                    .collect();
                assert_eq!(h, m["sha256"].as_str().expect("hash"));
                b
            })
            .collect()
    });
    assert!(
        composition.is_none() || !model_bytes.is_empty(),
        "empty ensemble"
    );
    let models: Vec<_> = model_bytes
        .iter()
        .map(|b| zenpredict::Model::from_bytes(b).expect("model"))
        .collect();
    let weights: Option<Vec<f64>> = composition.as_ref().map(|c| {
        c["weights"]
            .as_array()
            .expect("weights")
            .iter()
            .map(|v| v.as_f64().expect("weight"))
            .collect()
    });
    let mut scorer = (!models.is_empty()).then(|| {
        zensim::BakeScorer::ensemble(&models, weights.as_deref())
            .expect("servable ensemble")
            .with_parallel(false)
    });

    // Score each pair in one EXR-decode pass (decode is the bottleneck).
    // A = 372-feature MLP, the deprecated prior shipping profile — kept here
    // for the historical UPIQ comparison. (For the current default use
    // `ZensimProfile::codec_target()`; the historical linear V0_1 / V0_2
    // profiles live in `zensim_experimental`.)
    #[allow(deprecated)]
    let profiles: [(&str, Zensim); 1] = match arg(&args, "--profile").as_deref() {
        None | Some("a") => [("zensim_a", Zensim::new(ZensimProfile::A))],
        Some("bhdr") => [("zensim_bhdr", Zensim::new(ZensimProfile::BHdr))],
        Some(other) => panic!("unsupported named HDR control: {other}"),
    };

    let mut rdr = csv::Reader::from_path(&subjective).expect("open subjective csv");
    let headers = rdr.headers().expect("headers").clone();
    let col = |name: &str| headers.iter().position(|h| h == name).expect(name);
    let (c_cid, c_ds, c_hdr, c_test, c_ref) = (
        col("condition_id"),
        col("dataset"),
        col("is_hdr"),
        col("test_file"),
        col("reference_file"),
    );

    // Cache reference images (one ref → many distorted).
    let mut ref_cache: HashMap<String, Rgb> = HashMap::new();
    let mut rows: Vec<(String, Vec<f64>)> = Vec::new();
    let (mut ok, mut skip, mut err) = (0usize, 0usize, 0usize);

    for rec in rdr.records() {
        let rec = rec.expect("record");
        let dataset = &rec[c_ds];
        let is_hdr = &rec[c_hdr] == "1";
        if !corpora.iter().any(|c| c == dataset) {
            continue;
        }
        if hdr_only && !is_hdr {
            continue;
        }
        let cid = rec[c_cid].to_string();
        let test_path = Path::new(&images).join(&rec[c_test]);
        let ref_rel = rec[c_ref].to_string();
        let ref_path = Path::new(&images).join(&ref_rel);

        if !ref_cache.contains_key(&ref_rel) {
            match load_exr_rgb(&ref_path) {
                Ok(img) => {
                    ref_cache.insert(ref_rel.clone(), img);
                }
                Err(e) => {
                    eprintln!("REF FAIL {e}");
                    err += 1;
                    continue;
                }
            }
        }
        let r = &ref_cache[&ref_rel];
        let d = match load_exr_rgb(&test_path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("DIST FAIL {e}");
                err += 1;
                continue;
            }
        };
        if d.w != r.w || d.h != r.h {
            skip += 1;
            continue;
        }
        let mut scores = Vec::with_capacity(profiles.len());
        let mut failed = false;
        if let Some(scorer) = &mut scorer {
            scores.push(
                scorer
                    .compute_hdr(r, &d, zensim::feature_v2::HdrEncoding::Linear, None)
                    .expect("public HDR scoring"),
            );
        } else {
            for (_, z) in &profiles {
                match z.compute_pu_linear_planar(
                    [&r.r, &r.g, &r.b],
                    [&d.r, &d.g, &d.b],
                    r.w,
                    r.h,
                    r.w,
                ) {
                    Ok(res) => scores.push(res.score()),
                    Err(e) => {
                        eprintln!("SCORE FAIL {cid}: {e:?}");
                        failed = true;
                        break;
                    }
                }
            }
        }
        if failed {
            err += 1;
            continue;
        }
        rows.push((cid, scores));
        ok += 1;
        if ok % 50 == 0 {
            eprintln!("scored {ok} pairs…");
        }
    }

    let mut w = String::from("condition_id");
    if composition.is_some() {
        w.push_str(",zensim_candidate");
    } else {
        for (name, _) in &profiles {
            w.push(',');
            w.push_str(name);
        }
    }
    w.push('\n');
    for (cid, scores) in &rows {
        w.push_str(cid);
        for s in scores {
            w.push_str(&format!(",{s}"));
        }
        w.push('\n');
    }
    assert!(
        ok > 0 && skip == 0 && err == 0,
        "refusing incomplete UPIQ output: {ok} scored, {skip} skipped, {err} errors"
    );
    use std::io::Write;
    let mut output = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&out)
        .expect("fresh output");
    output.write_all(w.as_bytes()).expect("write out");
    eprintln!("done: {ok} scored, {skip} dim-skipped, {err} errored → {out}");
}
