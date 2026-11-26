//! Label-free Rev5 extraction of the external HDR video panels (HDR-VDC,
//! AVT-VQDB-UHD-1-HDR) from HDRVID display frames.
//!
//! Inputs are the 16-bit RGB PNGs written by `tools/hdrvid_decode`: full-range
//! PQ (SMPTE ST 2084) code values, BT.2020 primaries, at the registered display
//! frame. Every admitted row is one (video, display configuration) with eight
//! reference/distorted frame pairs. Each pair runs the production native HDR
//! walk (`research::extract_hdr`, `HdrEncoding::Pq { peak_nits }`) at
//! `FormulaRevision::Rev5` for the exact by_v2fy 420 IDs, in the frozen
//! 1825-slot f64 transport (unrequested slots NaN), like the UPIQ-380 owner.
//!
//! Display configurations follow the July HDR-VDC registration:
//! A = 4K, Pq{1000}; B = 4K, Pq{700}; C = 4K, Pq{700}, dimmed; D = 1080p,
//! Pq{700}; E = 1080p, Pq{700}, dimmed. AVT uses A only. "Dimmed" is the
//! experiment's shader applied to both images: PQ decode at the 10 000 cd/m²
//! spec peak, linear / 8, PQ re-encode (linear-srgb ST 2084 transfer).
//!
//! The admission (schema, set, role, contract, revision, IDs, row census and
//! every frame's original-byte SHA-256) is validated completely before any
//! frame is read. Output: `key, config, j, f0..f1824` per frame pair, plus
//! `<out>.manifest.json`. No label file is an input.
//!
//! Usage:
//!   hdrvid_extract --admission A.json --admission-sha256 HEX --out features.tsv

use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rayon::prelude::*;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use zensim::source::{AlphaMode, ImageSource, PixelFormat};

const TRANSPORT_WIDTH: usize = 1825;
const CANDIDATES: &str = include_str!("../../../benchmarks/costset2_2026-10-03.candidate_ids.json");
const CANDIDATES_SHA256: &str = "0a6a20dc356acef3bef9deffc411f03189813e8b924fddcf7b22f7efea6b9f17";
const CONTRACT: &str = "hdrvid-pq-png16-bt2020-display-v1";

fn canonical_ids() -> Vec<usize> {
    assert_eq!(digest(CANDIDATES.as_bytes()), CANDIDATES_SHA256);
    let candidates: serde_json::Value =
        serde_json::from_str(CANDIDATES).expect("pinned candidate JSON");
    serde_json::from_value(candidates["candidates"]["by_v2fy"].clone()).expect("pinned by_v2fy IDs")
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[derive(Deserialize)]
struct Admission {
    schema: String,
    set: String,
    role: String,
    input_contract: String,
    formula_revision: u8,
    frame_root: PathBuf,
    requested_ids: Vec<usize>,
    rows: Vec<Row>,
}

#[derive(Deserialize)]
struct Row {
    key: String,
    content: String,
    config: String,
    frames: Vec<FramePair>,
}

#[derive(Deserialize)]
struct FramePair {
    j: usize,
    reference: Frame,
    distorted: Frame,
}

#[derive(Deserialize)]
struct Frame {
    rel: String,
    sha256: String,
}

/// (display width, display height, peak nits, dimmed) per registered config.
fn config(set: &str, name: &str) -> Option<(usize, usize, f32, bool)> {
    match (set, name) {
        (_, "A") => Some((3840, 2160, 1000.0, false)),
        ("hdrvdc", "B") => Some((3840, 2160, 700.0, false)),
        ("hdrvdc", "C") => Some((3840, 2160, 700.0, true)),
        ("hdrvdc", "D") => Some((1920, 1080, 700.0, false)),
        ("hdrvdc", "E") => Some((1920, 1080, 700.0, true)),
        _ => None,
    }
}

fn validate(a: &Admission) -> Result<(), String> {
    let (rows, configs): (usize, &[&str]) = match a.set.as_str() {
        "hdrvdc" => (116 * 5, &["A", "B", "C", "D", "E"]),
        "avt" => (195, &["A"]),
        _ => return Err("unregistered external HDR video set".into()),
    };
    if a.schema != "hdrvid-extraction-admission-v1"
        || a.role != "external-eval"
        || a.input_contract != CONTRACT
        || a.formula_revision != 5
        || !a.frame_root.is_absolute()
        || a.requested_ids != canonical_ids()
        || a.rows.len() != rows
    {
        return Err("not the registered external Rev5 HDR video contract".into());
    }
    let mut seen = BTreeSet::new();
    for row in &a.rows {
        if !configs.contains(&row.config.as_str())
            || row.frames.len() != 8
            || row.frames.iter().enumerate().any(|(j, f)| f.j != j)
            || !seen.insert((row.key.clone(), row.config.clone()))
            || row.content.is_empty()
            || row.content.contains('/')
        {
            return Err(format!(
                "row {} / {}: member/config/frame mismatch",
                row.key, row.config
            ));
        }
        let (w, _, _, _) = config(&a.set, &row.config).expect("checked above");
        let dir = if w == 3840 { "frames" } else { "frames-1080" };
        for f in &row.frames {
            for frame in [&f.reference, &f.distorted] {
                let expected = format!("{dir}/{}/{}/", a.set, row.content);
                if !frame.rel.starts_with(&expected)
                    || frame.rel.contains("..")
                    || !frame.rel.ends_with(&format!("_f{}.png", f.j))
                    || frame.sha256.len() != 64
                    || !frame
                        .sha256
                        .bytes()
                        .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
                {
                    return Err(format!(
                        "{}: frame path/hash outside the admission",
                        row.key
                    ));
                }
            }
            if f.reference.rel == f.distorted.rel {
                return Err(format!("{}: identity pair", row.key));
            }
        }
    }
    if seen.iter().map(|(k, _)| k).collect::<BTreeSet<_>>().len() * configs.len() != rows {
        return Err("every video needs every registered configuration".into());
    }
    Ok(())
}

/// PQ code-value image over interleaved RGBA f32 (the `LinearF32Rgba` +
/// `HdrEncoding::Pq` contract of the declared-HDR route).
struct CodeImage {
    w: usize,
    h: usize,
    data: Vec<[f32; 4]>,
}

impl ImageSource for CodeImage {
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
        bytemuck::cast_slice(&self.data[y * self.w..(y + 1) * self.w])
    }
}

/// The experiment's dimming shader on one code value.
fn dim(v: f32) -> f32 {
    use linear_srgb::default::{linear_to_pq, pq_to_linear};
    linear_to_pq(pq_to_linear(v) / 8.0)
}

fn load(
    root: &Path,
    frame: &Frame,
    size: (usize, usize),
    dimmed: bool,
) -> Result<CodeImage, String> {
    let path = root.join(&frame.rel);
    let mut component = root.to_owned();
    for part in Path::new(&frame.rel).components() {
        component.push(part);
        if std::fs::symlink_metadata(&component)
            .map_err(|e| format!("{component:?}: {e}"))?
            .file_type()
            .is_symlink()
        {
            return Err("symlink below admitted frame root refused".into());
        }
    }
    let bytes = std::fs::read(&path).map_err(|e| format!("{path:?}: {e}"))?;
    if digest(&bytes) != frame.sha256 {
        return Err(format!("frame bytes changed: {path:?}"));
    }
    let decoded = zenpng::decode(
        &bytes,
        &zenpng::PngDecodeConfig::default(),
        &enough::Unstoppable,
    )
    .map_err(|e| format!("{path:?}: {e}"))?;
    let pixels = decoded.pixels;
    let layout = pixels.spec().layout;
    if layout.sample != zenpixels::SampleType::U16
        || layout.order != zenpixels::ChannelOrder::Rgb
        || (pixels.width() as usize, pixels.height() as usize) != size
    {
        return Err(format!(
            "{path:?}: not a {}x{} RGB16 display frame ({:?})",
            size.0,
            size.1,
            pixels.spec()
        ));
    }
    let (w, h) = size;
    let view = pixels.as_view();
    let mut data = Vec::with_capacity(w * h);
    for y in 0..h {
        for p in view.row(y as u32).as_chunks::<6>().0 {
            let mut px = [1.0f32; 4];
            for c in 0..3 {
                let v = u16::from_ne_bytes([p[2 * c], p[2 * c + 1]]) as f32 / 65535.0;
                px[c] = if dimmed { dim(v) } else { v };
            }
            data.push(px);
        }
    }
    Ok(CodeImage { w, h, data })
}

fn option(args: &[String], key: &str) -> Result<String, String> {
    let mut hits = args.iter().enumerate().filter(|(_, a)| *a == key);
    let (i, _) = hits.next().ok_or_else(|| format!("missing {key}"))?;
    if hits.next().is_some() {
        return Err(format!("duplicate {key}"));
    }
    args.get(i + 1)
        .filter(|v| !v.is_empty() && !v.starts_with("--"))
        .cloned()
        .ok_or_else(|| format!("missing value for {key}"))
}

fn run(args: &[String]) -> Result<(), String> {
    // The whole option contract is parsed before the admission file is read.
    let path = option(args, "--admission")?;
    let pin = option(args, "--admission-sha256")?;
    let out = option(args, "--out")?;
    if Path::new(&out).exists() || Path::new(&format!("{out}.manifest.json")).exists() {
        return Err("fresh extraction output required".into());
    }
    let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    if digest(&bytes) != pin {
        return Err("admission pin mismatch before frame reads".into());
    }
    let a: Admission = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    validate(&a)?; // complete metadata admission precedes every frame read
    if zensim::feature_v2::active_formula_revision() != zensim::feature_v2::FormulaRevision::Rev5 {
        return Err("HDRVID extraction requires process FormulaRevision::Rev5".into());
    }
    let req = zensim::research::Request::for_slots(
        zensim::feature_set_id::SlotSet::from_slots(a.requested_ids.iter().copied()),
        TRANSPORT_WIDTH,
    )
    .with_era_label("e31-hdrvid-native-hdr-rev5")
    .with_parallel(false);
    let jobs: Vec<(usize, usize)> = (0..a.rows.len())
        .flat_map(|r| (0..8).map(move |j| (r, j)))
        .collect();
    let manifest = Mutex::new(None::<String>);
    let done = std::sync::atomic::AtomicUsize::new(0);
    let lines: Vec<Result<String, String>> = jobs
        .par_iter()
        .map(|&(r, j)| {
            let row = &a.rows[r];
            let (w, h, peak_nits, dimmed) = config(&a.set, &row.config).expect("validated");
            let pair = &row.frames[j];
            let reference = load(&a.frame_root, &pair.reference, (w, h), dimmed)?;
            let distorted = load(&a.frame_root, &pair.distorted, (w, h), dimmed)?;
            let result = zensim::research::extract_hdr(
                &req,
                &reference,
                &distorted,
                zensim::feature_v2::HdrEncoding::Pq { peak_nits },
            )
            .map_err(|e| format!("{} {} f{j}: {e}", row.key, row.config))?;
            if result.values().len() != TRANSPORT_WIDTH {
                return Err("frozen transport width changed".into());
            }
            let metadata = result.manifest_json();
            {
                let mut m = manifest.lock().unwrap();
                match &*m {
                    Some(prev) if prev != &metadata => {
                        return Err("research metadata changed between pairs".into());
                    }
                    None => *m = Some(metadata),
                    _ => {}
                }
            }
            let mut line = format!("{}\t{}\t{j}", row.key, row.config);
            for (id, value) in result.values().iter().enumerate() {
                let requested = a.requested_ids.binary_search(&id).is_ok();
                if requested && !value.is_finite() {
                    return Err(format!("{} {} f{j}: nonfinite f{id}", row.key, row.config));
                }
                line.push_str(&format!(
                    "\t{:?}",
                    if requested { *value } else { f64::NAN }
                ));
            }
            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n.is_multiple_of(50) {
                eprintln!("HDRVID {} Rev5 extraction: {n}/{}", a.set, jobs.len());
            }
            Ok(line)
        })
        .collect();
    let mut output = String::from("key\tconfig\tj");
    for id in 0..TRANSPORT_WIDTH {
        output.push_str(&format!("\tf{id}"));
    }
    output.push('\n');
    for line in lines {
        output.push_str(&line?);
        output.push('\n');
    }
    let research = manifest.into_inner().unwrap().ok_or("empty extraction")?;
    let metadata = serde_json::json!({
        "schema": "hdrvid-extraction-result-v1", "set": a.set, "rows": a.rows.len(),
        "frame_pairs": jobs.len(), "formula_revision": 5, "input_contract": a.input_contract,
        "requested_ids": a.requested_ids, "admission_sha256": pin,
        "build_commit": zensim::research::BUILD_COMMIT,
        "research": serde_json::from_str::<serde_json::Value>(&research).map_err(|e| e.to_string())?,
        "display_configs": "A 4K Pq1000; B 4K Pq700; C 4K Pq700 dimmed/8; D 1080p Pq700; E 1080p Pq700 dimmed/8 (AVT: A only)",
        "dimming": "linear-srgb tf::pq: linear_to_pq(pq_to_linear(v) / 8), both images",
        "labels_read": false,
    });
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&out)
        .map_err(|e| e.to_string())?;
    file.write_all(output.as_bytes())
        .map_err(|e| e.to_string())?;
    let file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(format!("{out}.manifest.json"))
        .map_err(|e| e.to_string())?;
    serde_json::to_writer_pretty(file, &metadata).map_err(|e| e.to_string())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if let Err(e) = run(&args) {
        eprintln!("hdrvid_extract: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(set: &str, dir: &str, content: &str, stem: &str, j: usize) -> serde_json::Value {
        serde_json::json!({"rel": format!("{dir}/{set}/{content}/{stem}_f{j}.png"), "sha256": "1".repeat(64)})
    }

    /// An absolute root that must never be opened. Admission requires an
    /// absolute `frame_root`; on Windows an absolute path needs a drive (or
    /// UNC) prefix, so `/nonexistent-…` is relative there and every fixture
    /// would be refused for the wrong reason.
    #[cfg(windows)]
    const TRIPWIRE_ROOT: &str = r"C:\nonexistent-hdrvid-tripwire";
    #[cfg(not(windows))]
    const TRIPWIRE_ROOT: &str = "/nonexistent-hdrvid-tripwire";

    fn fixture(set: &str) -> serde_json::Value {
        let (videos, configs): (usize, &[&str]) = if set == "hdrvdc" {
            (116, &["A", "B", "C", "D", "E"])
        } else {
            (195, &["A"])
        };
        let mut rows = Vec::new();
        for v in 0..videos {
            let content = format!("c{}", v % 5);
            for c in configs {
                let dir = if matches!(*c, "D" | "E") {
                    "frames-1080"
                } else {
                    "frames"
                };
                let frames: Vec<_> = (0..8)
                    .map(|j| {
                        serde_json::json!({"j": j,
                        "reference": frame(set, dir, &content, &format!("{content}__ref"), j),
                        "distorted": frame(set, dir, &content, &format!("v{v}"), j)})
                    })
                    .collect();
                rows.push(
                    serde_json::json!({"key": format!("{set}/v{v}"), "content": content,
                    "config": c, "frames": frames}),
                );
            }
        }
        serde_json::json!({"schema": "hdrvid-extraction-admission-v1", "set": set,
            "role": "external-eval", "input_contract": CONTRACT, "formula_revision": 5,
            "frame_root": TRIPWIRE_ROOT, "requested_ids": canonical_ids(), "rows": rows})
    }

    fn checked(v: serde_json::Value) -> Result<(), String> {
        validate(&serde_json::from_value(v).unwrap())
    }

    #[test]
    fn complete_admissions_validate_without_frame_access() {
        assert!(std::path::Path::new(TRIPWIRE_ROOT).is_absolute());
        assert!(!std::path::Path::new(TRIPWIRE_ROOT).exists());
        assert!(checked(fixture("hdrvdc")).is_ok());
        assert!(checked(fixture("avt")).is_ok());
    }

    #[test]
    fn foreign_contract_role_revision_and_ids_refuse() {
        for (field, value) in [
            ("role", serde_json::json!("train")),
            ("formula_revision", serde_json::json!(4)),
            ("input_contract", serde_json::json!("hdrvdc-944")),
            ("set", serde_json::json!("chug")),
            (
                "requested_ids",
                serde_json::json!((0..420).collect::<Vec<_>>()),
            ),
        ] {
            let mut v = fixture("avt");
            v[field] = value;
            assert!(checked(v).is_err(), "{field}");
        }
    }

    #[test]
    fn late_rows_with_wrong_config_path_or_identity_refuse() {
        let mut v = fixture("avt");
        v["rows"][194]["config"] = "B".into();
        assert!(checked(v).is_err(), "AVT has only config A");
        let mut v = fixture("hdrvdc");
        v["rows"][579]["frames"][7]["distorted"]["rel"] = "frames/avt/c0/x_f7.png".into();
        assert!(checked(v).is_err(), "cross-set frame");
        let mut v = fixture("hdrvdc");
        v["rows"][579]["frames"][7]["distorted"] = v["rows"][579]["frames"][7]["reference"].clone();
        assert!(checked(v).is_err(), "identity pair");
        let mut v = fixture("hdrvdc");
        v["rows"][3]["frames"][0]["reference"]["rel"] = "frames/hdrvdc/c0/c0__ref_f0.png".into();
        assert!(checked(v).is_err(), "config D must read the 1080p frames");
        let mut v = fixture("hdrvdc");
        v["rows"].as_array_mut().unwrap().pop();
        assert!(checked(v).is_err(), "row census");
    }

    #[test]
    fn dimming_divides_spec_peak_luminance_by_eight() {
        use linear_srgb::default::pq_to_linear;
        for v in [0.1f32, 0.3, 0.5, 0.58, 0.75, 0.9] {
            let ratio = pq_to_linear(v) / pq_to_linear(dim(v));
            assert!((ratio - 8.0).abs() < 8.0 * 2e-3, "{v}: {ratio}");
        }
        assert_eq!(dim(0.0), 0.0);
    }
}
