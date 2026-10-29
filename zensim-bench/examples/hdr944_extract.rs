//! hdr944_extract — 944-regime HDR-PQ feature extraction for the hdr_v3mix
//! datagen pairs (SOTA-944 amendment; benchmarks/sota944_campaign_2026-08-03.md
//! "B-gap resolution").
//!
//! Input: the datagen pairs TSVs (`image_path codec q knob_tuple_json
//! ref_path dist_path` — fleet-container paths; only the BASENAMES are used),
//! with `--ref-root` (the imazen-26-hdr-grid PQ-PNG refs) and a paired
//! `--enc-root` per TSV (the datagen `enc/zenjxl` bitstore).
//!
//! Per pair: ref = 16-bit PNG (PQ code values), dist = zenjxl-decoded to
//! native RGB16 PQ with codestream-declared primaries; features = the CANONICAL
//! `Zensim::compute_folded720_append2_features_hdr` (944; `HdrEncoding::Pq
//! { peak_nits: 10_000 }`, default toggles — dst-activity OFF per the P1.5
//! adjudication), profile `codec_target`, per-pair single-threaded compute
//! with pair-level std threads. Output CSV: `dist_basename,q,f0..f943`.
//!
//! FRONT-END NOTE (documented in the leg's manifest): this is the CURRENT
//! HDR route (PU21 chunk-2 lineage) at 944 — a NEW-REGIME leg. The v3-era
//! `compute_pu_linear_extended_features` 372 front-end is superseded; the
//! legacy targets require separate admission and are never copied by this tool.
//! Repeated `--score-bake PATH` instead emits native `BakeScorer::compute_hdr` scores
//! for each final bake. Refusals remain keyed in a sidecar with NaN TSV cells and
//! a nonzero exit; extraction/audit mode retains its original output contract.
//! Optional `--audit-composition JSON` audits a hash-bound Rust ensemble against
//! canonical features, native scalar scoring and prepared HDR maps.

use std::io::{BufWriter, Write as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use zensim::feature_v2::{HdrEncoding, V2NewFeatureToggles, V2Scratch};
use zensim::{Zensim, ZensimProfile};

struct Pq16Image {
    data: Vec<[u16; 4]>,
    linear: Option<Vec<[f32; 4]>>,
    w: usize,
    h: usize,
    primaries: zensim::ColorPrimaries,
}

impl Pq16Image {
    fn from_rgb16(px: &[[u16; 3]], w: usize, h: usize, primaries: zensim::ColorPrimaries) -> Self {
        Self {
            data: px.iter().map(|&[r, g, b]| [r, g, b, 65535]).collect(),
            linear: None,
            w,
            h,
            primaries,
        }
    }
}

impl zensim::source::ImageSource for Pq16Image {
    fn width(&self) -> usize {
        self.w
    }
    fn height(&self) -> usize {
        self.h
    }
    fn pixel_format(&self) -> zensim::source::PixelFormat {
        if self.linear.is_some() { zensim::source::PixelFormat::LinearF32Rgba }
        else { zensim::source::PixelFormat::Srgb16Rgba }
    }
    fn row_bytes(&self, y: usize) -> &[u8] {
        match &self.linear {
            Some(px) => bytemuck::cast_slice(&px[y * self.w..(y + 1) * self.w]),
            None => bytemuck::cast_slice(&self.data[y * self.w..(y + 1) * self.w]),
        }
    }
    fn alpha_mode(&self) -> zensim::source::AlphaMode {
        zensim::source::AlphaMode::Opaque
    }
    fn is_hdr(&self) -> bool {
        true
    }
    fn color_primaries(&self) -> zensim::ColorPrimaries {
        self.primaries
    }
}

/// Registered July-29 display mapping; the as-is BT.2020 reinterpretation is
/// deliberate owner policy, not a general BT.2020-to-BT.709 conversion.
fn decode_rousselot(path: &Path, factor: f32) -> Result<Pq16Image, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let decoded = zenbitmaps_hdr::decode_hdr(&bytes, enough::Unstoppable).map_err(|e| e.to_string())?;
    if decoded.layout != zenbitmaps_hdr::PixelLayout::RgbF32 {
        return Err("native RGBE decoder must preserve f32 samples".into());
    }
    let mut pixels = Vec::with_capacity(decoded.width as usize * decoded.height as usize);
    for p in decoded.pixels().chunks_exact(12) {
        let mut rgba = [0., 0., 0., 1.];
        for c in 0..3 {
            let raw = f32::from_ne_bytes(p[c * 4..c * 4 + 4].try_into().unwrap());
            if !raw.is_finite() { return Err("nonfinite RGBE sample".into()); }
            rgba[c] = (raw * factor).clamp(0., 1000.);
        }
        pixels.push(rgba);
    }
    Ok(Pq16Image { data: Vec::new(), linear: Some(pixels), w: decoded.width as usize,
        h: decoded.height as usize, primaries: zensim::ColorPrimaries::Srgb })
}

#[derive(Clone)]
struct Cell {
    ref_file: PathBuf,
    dist_file: PathBuf,
    dist_base: String,
    q: String,
}

type DecodedPq16 = (Vec<[u16; 3]>, usize, usize, zensim::ColorPrimaries);

fn decode_ref_png16(path: &Path, declared_cicp: bool) -> Result<DecodedPq16, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let out = zenpng::decode(
        &bytes,
        &zenpng::PngDecodeConfig::default(),
        &enough::Unstoppable,
    )
    .map_err(|e| format!("ref {path:?}: {e}"))?;
    let info = &out.info;
    if info.bit_depth != 16
        || info.icc_profile.is_some()
        || info.srgb_intent.is_some()
        || info.source_gamma.is_some()
        || info.chromaticities.is_some()
    {
        return Err(format!(
            "ref {path:?}: requires native16 BT.2020 PQ or explicitly declared untagged datagen input; conflicting/ICC color unsupported"
        ));
    }
    let primaries = if declared_cicp {
        match info.cicp {
            Some(c)
                if c.transfer_characteristics == 16
                    && c.matrix_coefficients == 0
                    && c.full_range =>
            {
                match c.color_primaries {
                    1 => zensim::ColorPrimaries::Srgb,
                    9 => zensim::ColorPrimaries::Bt2020,
                    12 => zensim::ColorPrimaries::DisplayP3,
                    _ => return Err("unsupported HDR reference primaries".into()),
                }
            }
            _ => return Err("declared HDR reference requires full-range RGB PQ cICP".into()),
        }
    } else {
        if info
            .cicp
            .is_some_and(|c| c != zenpixels::Cicp::new(9, 16, 0, true))
        {
            return Err("reference conflicts with explicit BT.2020 PQ datagen contract".into());
        }
        zensim::ColorPrimaries::Bt2020
    };
    let buf = out.pixels;
    let (w, h) = (buf.width() as usize, buf.height() as usize);
    let bytes = buf.copy_to_contiguous_bytes();
    let channels = bytes.len() / (w * h * 2);
    if !matches!(channels, 3 | 4) {
        return Err("HDR reference requires RGB/RGBA16".into());
    }
    let mut pixels = Vec::with_capacity(w * h);
    for p in bytes.chunks_exact(channels * 2) {
        let c = |i: usize| u16::from_ne_bytes([p[i * 2], p[i * 2 + 1]]);
        if channels == 4 && c(3) != 65535 {
            return Err("HDR reference requires opaque alpha".into());
        }
        pixels.push([c(0), c(1), c(2)]);
    }
    Ok((pixels, w, h, primaries))
}

fn decode_dist_jxl16(path: &Path) -> Result<DecodedPq16, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {path:?}: {e}"))?;
    let out = zenjxl::decode(&bytes, None, &[zenpixels::PixelDescriptor::RGB16_BT2100_PQ])
        .map_err(|e| format!("jxl decode {path:?}: {e:?}"))?;
    // A preferred descriptor chooses sample storage, not a color conversion.
    // zenjxl keeps native code values and reports their color in JxlInfo.
    let primaries = match out.info.cicp {
        Some((p, 16, 0, true)) => match p {
            1 => zensim::ColorPrimaries::Srgb,
            9 => zensim::ColorPrimaries::Bt2020,
            12 => zensim::ColorPrimaries::DisplayP3,
            _ => return Err(format!("jxl {path:?}: unsupported PQ primaries {p}")),
        },
        _ => {
            return Err(format!(
                "jxl {path:?}: explicit full-range RGB PQ codestream required, got {:?}",
                out.info.cicp
            ));
        }
    };
    let buf = out.pixels;
    if buf.descriptor().channel_type() != zenpixels::ChannelType::U16 {
        return Err(format!(
            "jxl {path:?}: expected native16 storage, got {:?}",
            buf.descriptor()
        ));
    }
    let w = buf.width() as usize;
    let h = buf.height() as usize;
    let slice = buf.as_slice();
    let bytes = slice.contiguous_bytes();
    let u16s: &[u16] = bytemuck::try_cast_slice(bytes.as_ref())
        .map_err(|e| format!("jxl {path:?}: pixel cast: {e}"))?;
    let ch = u16s.len() / (w * h);
    if !matches!(ch, 3 | 4) {
        return Err(format!("jxl {path:?}: {ch} channels"));
    }
    if ch == 4 && u16s.as_chunks::<4>().0.iter().any(|p| p[3] != 65535) {
        return Err("HDR distorted image requires opaque alpha".into());
    }
    let px: Vec<[u16; 3]> = (0..w * h)
        .map(|i| [u16s[i * ch], u16s[i * ch + 1], u16s[i * ch + 2]])
        .collect();
    Ok((px, w, h, primaries))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // Source-identity adapter only. The existing D4 owner performs fingerprints
    // and match review on these native-decoder views; they never train a model.
    if args.get(1).map(String::as_str) == Some("--source-fingerprint-view") {
        assert_eq!(args.len(), 5, "source-fingerprint-view INPUT NEW_PPM MIDGRAY_NITS");
        let midgray: f32 = args[4].parse().expect("midgray nits");
        assert!([32., 100., 320.].contains(&midgray), "frozen source-screen exposure grid");
        let source = decode_rousselot(Path::new(&args[2]), 179.).expect("native RGBE reference");
        let rgb: Vec<u8> = source.linear.as_ref().unwrap().iter().flat_map(|p| {
            p[..3].iter().map(|&n| ((n / (n + midgray)).sqrt() * 255.).round() as u8)
        }).collect();
        let ppm = zenbitmaps_hdr::encode_ppm(&rgb, source.w as u32, source.h as u32,
            zenbitmaps_hdr::PixelLayout::Rgb8, enough::Unstoppable).expect("native source view encoder");
        std::fs::OpenOptions::new().write(true).create_new(true).open(&args[3]).unwrap().write_all(&ppm).unwrap();
        use sha2::Digest as _;
        let receipt = serde_json::json!({"schema":"hdr-source-identity-view-v1", "source":args[2],
            "source_sha256":sha2::Sha256::digest(std::fs::read(&args[2]).unwrap()).iter().map(|b| format!("{b:02x}")).collect::<String>(),
            "output_sha256":sha2::Sha256::digest(&ppm).iter().map(|b| format!("{b:02x}")).collect::<String>(),
            "midgray_nits":midgray,"mapping":"sqrt(nits/(nits+midgray))*255;nearest-u8",
            "native_ingress":"registered Rousselot K179;1000-nit ceiling;as-is primaries",
            "scope":"D4 source-identity screening only; never features, teacher targets or training images"});
        let mut receipt_file = std::fs::OpenOptions::new().write(true).create_new(true)
            .open(format!("{}.manifest.json",args[3])).unwrap();
        serde_json::to_writer_pretty(&mut receipt_file,&receipt).unwrap();
        return;
    }

    let mut pairs_tsvs: Vec<PathBuf> = Vec::new();
    let mut enc_roots: Vec<PathBuf> = Vec::new();
    let mut ref_root = PathBuf::new();
    let mut out_path = PathBuf::new();
    let mut n_threads = 8usize;
    let mut input_contract = None;
    let mut audit_composition: Option<PathBuf> = None;
    let mut score_bakes: Vec<PathBuf> = Vec::new();
    let mut requested_ids: Option<Vec<u16>> = None;
    let mut absolute_pairs = false;
    let mut era_label = String::from("e26-native-hdr-rev5");
    let mut pq_png_distortions = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--pairs" => {
                pairs_tsvs.push(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--enc-root" => {
                enc_roots.push(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--ref-root" => {
                ref_root = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--out" => {
                out_path = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--input-contract" => {
                assert!(input_contract.is_none(), "duplicate input contract");
                input_contract = Some(args[i + 1].clone());
                i += 2;
            }
            "--era-label" => {
                era_label = args[i + 1].clone();
                i += 2;
            }
            "--pq-png-distortions" => {
                pq_png_distortions = true;
                i += 1;
            }
            "--threads" => {
                n_threads = args[i + 1].parse().expect("threads");
                i += 2;
            }
            "--absolute-pairs" => {
                absolute_pairs = true;
                i += 1;
            }
            "--requested-ids" => {
                assert!(requested_ids.is_none(), "duplicate requested ids");
                let ids: Vec<u16> = args[i + 1]
                    .split(',')
                    .map(|x| x.parse().expect("feature id"))
                    .collect();
                assert!(
                    !ids.is_empty() && ids.windows(2).all(|w| w[0] < w[1]),
                    "sorted unique ids required"
                );
                requested_ids = Some(ids);
                i += 2;
            }
            "--score-bake" => {
                score_bakes.push(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--audit-composition" => {
                audit_composition = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            other => panic!("unknown arg {other}"),
        }
    }
    assert!(
        matches!(
            input_contract.as_deref(),
            Some("hdr-common-primaries-v2-bt2020-pq10000" | "hdr-common-primaries-v2-cicp-pq10000" | "rousselot-registered-k179-as-is" | "rousselot-registered-k100-as-is")
        ),
        "explicit current HDR input contract required; old caches are incompatible"
    );
    let rousselot_factor = match input_contract.as_deref() {
        Some("rousselot-registered-k179-as-is") => Some(179.),
        Some("rousselot-registered-k100-as-is") => Some(100.),
        _ => None,
    };
    let declared_cicp = input_contract.as_deref() == Some("hdr-common-primaries-v2-cicp-pq10000");
    assert!(
        n_threads > 0 && !out_path.exists(),
        "nonzero threads and fresh output required"
    );
    assert_eq!(
        pairs_tsvs.len(),
        enc_roots.len(),
        "--pairs and --enc-root must pair up"
    );
    assert!(!pairs_tsvs.is_empty() && ref_root.exists() && !out_path.as_os_str().is_empty());

    // Load cells from every TSV.
    let mut cells: Vec<Cell> = Vec::new();
    for (tsv, enc) in pairs_tsvs.iter().zip(&enc_roots) {
        let txt = std::fs::read_to_string(tsv).expect("pairs tsv");
        let mut lines = txt.lines();
        let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
        let col = |n: &str| header.iter().position(|h| *h == n).expect("column");
        let (c_q, c_ref, c_dist) = (col("q"), col("ref_path"), col("dist_path"));
        for line in lines {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() <= c_dist {
                continue;
            }
            let rb = Path::new(f[c_ref]).file_name().expect("ref base");
            let db = Path::new(f[c_dist]).file_name().expect("dist base");
            cells.push(Cell {
                ref_file: if absolute_pairs {
                    assert!(
                        Path::new(f[c_ref]).is_absolute(),
                        "absolute reference required"
                    );
                    PathBuf::from(f[c_ref])
                } else {
                    ref_root.join(rb)
                },
                dist_file: if absolute_pairs {
                    assert!(
                        Path::new(f[c_dist]).is_absolute(),
                        "absolute distortion required"
                    );
                    PathBuf::from(f[c_dist])
                } else {
                    enc.join(db)
                },
                dist_base: db.to_string_lossy().into_owned(),
                q: f[c_q].to_string(),
            });
        }
    }
    eprintln!(
        "hdr944_extract: {} cells, {} threads",
        cells.len(),
        n_threads
    );

    assert!(
        score_bakes.is_empty() || audit_composition.is_none(),
        "scoring and audit modes are separate"
    );
    assert!(
        requested_ids.is_none() || (score_bakes.is_empty() && audit_composition.is_none()),
        "research extraction is a separate mode"
    );
    if requested_ids.is_some() {
        match zensim::feature_v2::active_formula_revision() {
            zensim::feature_v2::FormulaRevision::Rev5 => {
                assert_eq!(era_label, "e26-native-hdr-rev5", "preserve E26 era");
            }
            zensim::feature_v2::FormulaRevision::Rev7 => {
                assert_eq!(era_label, "rev7_rowlanes", "Rev7 requires its explicit era");
            }
            _ => panic!("research HDR extraction admits only registered Rev5 or Rev7"),
        }
    }
    if pq_png_distortions {
        assert!(declared_cicp, "PQ PNG distortions require explicit per-image cICP");
    }
    // Match the frozen Rev7 bank layout, even as research adds unrelated families.
    let research_width = if zensim::feature_v2::active_formula_revision()
        == zensim::feature_v2::FormulaRevision::Rev7 { 1825 } else { zensim::research::full_width() };
    let research_req = requested_ids.as_ref().map(|ids| {
        assert!(
            ids.iter()
                .all(|&x| usize::from(x) < research_width),
            "feature id out of range"
        );
        zensim::research::Request::for_slots(
            zensim::feature_set_id::SlotSet::from_slots(ids.iter().map(|&x| usize::from(x))),
            research_width,
        )
        .with_era_label(era_label.clone())
        .with_parallel(false)
    });
    let research_manifest = std::sync::Mutex::new(None::<String>);
    let score_bytes: Vec<Vec<u8>> = score_bakes
        .iter()
        .map(|p| std::fs::read(p).expect("score bake"))
        .collect();
    let refusals = std::sync::Mutex::new(Vec::<serde_json::Value>::new());
    let identities = std::sync::Mutex::new(vec![false; cells.len()]);
    let composition: Option<serde_json::Value> = audit_composition.as_ref().map(|path| {
        serde_json::from_slice(&std::fs::read(path).expect("composition"))
            .expect("composition JSON")
    });
    let model_bytes: Vec<Vec<u8>> = composition.as_ref().map_or_else(Vec::new, |c| {
        use sha2::{Digest, Sha256};
        c["members"]
            .as_array()
            .expect("members")
            .iter()
            .map(|m| {
                let bytes =
                    std::fs::read(m["path"].as_str().expect("model path")).expect("model bytes");
                assert_eq!(
                    Sha256::digest(&bytes)
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect::<String>(),
                    m["sha256"].as_str().expect("model hash")
                );
                bytes
            })
            .collect()
    });
    assert!(
        composition.is_none() || !model_bytes.is_empty(),
        "empty composition"
    );
    let weights: Option<Vec<f64>> = composition.as_ref().map(|c| {
        c["weights"]
            .as_array()
            .expect("weights")
            .iter()
            .map(|x| x.as_f64().expect("weight"))
            .collect()
    });
    let audits = std::sync::Mutex::new(vec![None; cells.len()]);
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let mut rows: Vec<Option<String>> = vec![None; cells.len()];
    let rows_ptr = std::sync::Mutex::new(&mut rows);
    std::thread::scope(|s| {
        for _ in 0..n_threads {
            s.spawn(|| {
                let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
                let mut scratch = V2Scratch::new();
                let score_models: Vec<_> = score_bytes.iter().map(|b| zenpredict::Model::from_bytes(b).expect("score model")).collect();
                let mut score_scorers: Vec<_> = score_models.iter().map(|m| zensim::BakeScorer::new(m).map(|z| z.with_parallel(false))).collect();
                let models: Vec<_> = model_bytes.iter().map(|b| zenpredict::Model::from_bytes(b).expect("model")).collect();
                let mut scorer = (!models.is_empty()).then(|| zensim::BakeScorer::ensemble(&models, weights.as_deref()).expect("servable composition").with_parallel(false).with_finite_moment_refinement(true));
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= cells.len() {
                        break;
                    }
                    let c = &cells[i];
                    let row = (|| -> Result<String, String> {
                        let (source, distorted, encoding) = if let Some(factor) = rousselot_factor {
                            (decode_rousselot(&c.ref_file, factor)?, decode_rousselot(&c.dist_file, factor)?, HdrEncoding::Linear)
                        } else {
                        let (r16, rw, rh, primaries) =
                            decode_ref_png16(&c.ref_file, declared_cicp)?;
                        let (d16, dw, dh, dist_primaries) = if pq_png_distortions {
                            decode_ref_png16(&c.dist_file, true)?
                        } else {
                            decode_dist_jxl16(&c.dist_file)?
                        };
                        if (rw, rh) != (dw, dh) {
                            return Err(format!(
                                "dim mismatch {}: ref {rw}x{rh} vs dist {dw}x{dh}",
                                c.dist_base
                            ));
                        }
                        let source = Pq16Image::from_rgb16(&r16, rw, rh, primaries);
                        let distorted = Pq16Image::from_rgb16(&d16, dw, dh, dist_primaries);

                        (source, distorted, HdrEncoding::Pq { peak_nits: 10_000.0 })
                        };
                        let (rw, rh) = (source.w, source.h);
                        if (rw, rh) != (distorted.w, distorted.h) { return Err("native HDR dimension mismatch".into()); }
                        let identical = source.primaries == distorted.primaries && source.data == distorted.data && source.linear == distorted.linear;
                        identities.lock().unwrap()[i] = identical;
                        if !score_scorers.is_empty() {
                            let mut line = format!("{}\t{}", c.ref_file.display(), c.dist_file.display());
                            for (j, scorer) in score_scorers.iter_mut().enumerate() {
                                let score = match scorer {
                                    Ok(scorer) => scorer.compute_hdr(&source, &distorted, encoding, None).map_err(|e| e.to_string()),
                                    Err(e) => Err(e.to_string()),
                                };
                                match score {
                                    Ok(score) if score.is_finite() => line.push_str(&format!("\t{score:.17e}")),
                                    other => {
                                        let reason = match other { Ok(_) => "nonfinite score".to_owned(), Err(e) => e };
                                        refusals.lock().unwrap().push(serde_json::json!({"row_id":i,"bake":score_bakes[j],"ref_path":c.ref_file,"dist_path":c.dist_file,"reason":reason}));
                                        line.push_str("\tNaN");
                                    }
                                }
                            }
                            return Ok(line);
                        }
                        if let Some(req) = &research_req {
                            let result = zensim::research::extract_hdr(req, &source, &distorted, encoding)
                                .map_err(|e| format!("research {}: {e:?}", c.dist_base))?;
                            let manifest = result.manifest_json();
                            let mut recorded = research_manifest.lock().unwrap();
                            if let Some(previous) = recorded.as_ref() {
                                assert_eq!(previous, &manifest, "research provenance changed between rows");
                            } else { *recorded = Some(manifest); }
                            let mut line = format!("{}\t{}", c.dist_base, c.q);
                            for (id, value) in result.values().iter().enumerate() {
                                let measured = requested_ids.as_ref().unwrap().binary_search(&(id as u16)).is_ok();
                                if measured && !value.is_finite() { return Err(format!("nonfinite requested f{id}")); }
                                line.push_str(&format!("\t{:?}", if measured { *value } else { f64::NAN }));
                            }
                            return Ok(line);
                        }
                        let canonical = if let Some(scorer) = &scorer
                            && zensim::feature_v2::active_formula_revision() == zensim::feature_v2::FormulaRevision::Rev5
                        {
                            // Rev5 supports the registered planned basic/peaks/v2
                            // read set, not the legacy append2/full-pool walk.
                            let ids = scorer.consumed_feature_ids().map_err(|e| e.to_string())?;
                            let req = zensim::research::Request::for_slots(
                                zensim::feature_set_id::SlotSet::from_slots(ids.iter().map(|&x| usize::from(x))),
                                zensim::research::full_width(),
                            ).with_parallel(false);
                            zensim::research::extract_hdr(&req, &source, &distorted, encoding)
                                .map_err(|e| format!("audit research: {e:?}"))?.values().to_vec()
                        } else {
                            let r = z
                            .compute_folded720_append2_features_hdr(
                                &source,
                                &distorted,
                                encoding,
                                V2NewFeatureToggles {
                                    v1_pools: zensim::feature_v2::V1PoolsMode::Full,
                                    ..Default::default()
                                },
                                &mut scratch,
                            )
                            .map_err(|e| format!("compute {}: {e:?}", c.dist_base))?;
                            assert_eq!(r.features().len(), 944, "regime width");
                            r.features().to_vec()
                        };
                        let feats = &canonical;
                        if let Some(scorer) = &mut scorer {
                            let ids = scorer.consumed_feature_ids().map_err(|e| e.to_string())?;
                            let cached = scorer.score_features_with_identity(feats, rw as u32, rh as u32, None, identical).map_err(|e|e.to_string())?;
                            let scalar = scorer.compute_hdr(&source, &distorted, encoding, None).map_err(|e|e.to_string())?;
                            let mut worker = scorer.prepare_steering_hdr(&source, encoding, 8).map_err(|e|e.to_string())?;
                            let mapped = worker.compute(&distorted, None).map_err(|e|e.to_string())?;
                            let max_feature_error = ids.iter().map(|&i| (feats[i as usize]-mapped.result().features()[i as usize]).abs()).fold(0.0_f64,f64::max);
                            for &id in &ids {
                                let id=id as usize;
                                if (feats[id]-mapped.result().features()[id]).abs()>2e-9*feats[id].abs().max(1.) { return Err(format!("canonical feature mismatch f{id}")); }
                            }
                            if (cached-scalar).abs()>1e-5 || (scalar-mapped.result().score()).abs()>1e-5 { return Err("public HDR score parity failure".into()); }
                            if !mapped.unsupported_refinement_feature_ids().is_empty() || !mapped.refinement_gain(0,0,rw,rh).is_finite() { return Err("incomplete/nonfinite HDR refinement".into()); }
                            let audit=serde_json::json!({"row_id":i,"dist_basename":c.dist_base,"q":c.q,"width":rw,"height":rh,"cached_score":cached,"scalar_score":scalar,"map_score":mapped.result().score(),"consumed_ids":ids,"max_feature_error":max_feature_error,"unsupported_density_ids":mapped.unsupported_feature_ids(),"unsupported_refinement_ids":mapped.unsupported_refinement_feature_ids(),"full_image_gain":mapped.refinement_gain(0,0,rw,rh)});
                            let identity=worker.compute(&source,None).map_err(|e|e.to_string())?;
                            if identity.result().score()!=100. || identity.refinement_gain(0,0,rw,rh)!=0. { return Err("HDR identity failure".into()); }
                            audits.lock().unwrap()[i]=Some(audit);
                        }
                        let mut line =
                            String::with_capacity(16 + c.dist_base.len() + feats.len() * 20);
                        line.push_str(&c.dist_base);
                        line.push('\t');
                        line.push_str(&c.q);
                        for f in feats {
                            line.push('\t');
                            line.push_str(&format!("{f:?}"));
                        }
                        Ok(line)
                    })();
                    match row {
                        Ok(line) => {
                            rows_ptr.lock().unwrap()[i] = Some(line);
                        }
                        Err(e) => eprintln!("SKIP {e}"),
                    }
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if d.is_multiple_of(500) {
                        eprintln!("  {d}/{} cells", cells.len());
                    }
                }
            });
        }
    });

    assert!(
        rows.iter().all(Option::is_some),
        "failed HDR rows: refusing partial output"
    );
    let refused_count = refusals.lock().unwrap().len();
    let manifest = serde_json::json!({
        "input_contract":input_contract,
        "era_label":era_label,
        "distortion_storage":if pq_png_distortions { "native PQ16 PNG with cICP" } else { "native PQ JXL with cICP" },
        "formula_revision":format!("{:?}",zensim::feature_v2::active_formula_revision()),
        "rows":cells.len(), "feature_count":if score_bakes.is_empty() { Some(if requested_ids.is_some() { research_width } else if composition.is_some() && zensim::feature_v2::active_formula_revision() == zensim::feature_v2::FormulaRevision::Rev5 { zensim::research::full_width() } else { 944 }) } else { None },
        "score_bakes":score_bakes,
        "refused_scores":refused_count,
        "reference_primaries":if rousselot_factor.is_some() { "BT.2020 reinterpreted as-is under registered protocol" } else if declared_cicp { "per-image cICP" } else { "BT.2020" },
        "v1_pools":"full", "distorted_primaries":if rousselot_factor.is_some() { "same registered reinterpretation as reference" } else { "per-image cICP" },
        "transfer":if rousselot_factor.is_some() { "linear nits" } else { "PQ" },
        "display_peak_nits":if rousselot_factor.is_some() { 1000 } else { 10000 },
        "rousselot_raw_to_nits":rousselot_factor,
        "rousselot_color_policy":rousselot_factor.map(|_| "BT.2020 values reinterpreted as-is per registered July-29 protocol; no gamut conversion"),
        "untagged_reference_policy":"explicit datagen declaration; never inferred from bit depth",
        "source_manifests":pairs_tsvs, "decoder":if rousselot_factor.is_some() { "zenbitmaps HDR native f32" } else if pq_png_distortions { "zenpng native16 for both sides" } else { "zenpng native16 + zenjxl RGB16_BT2100_PQ" },
        "feature_input_era":"hdr-common-primaries-v2",
        "requested_ids":requested_ids,
        "pixels_identical":identities.into_inner().unwrap(),
        "research":research_manifest.into_inner().unwrap().map(|s| serde_json::from_str::<serde_json::Value>(&s).unwrap())
    });
    let manifest_path = out_path.with_extension("manifest.json");
    let mut manifest_file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(manifest_path)
        .expect("fresh manifest");
    serde_json::to_writer_pretty(&mut manifest_file, &manifest).unwrap();
    writeln!(manifest_file).unwrap();
    if composition.is_some() {
        let audits = audits.into_inner().unwrap();
        assert!(audits.iter().all(Option::is_some), "incomplete HDR audit");
        let f = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(out_path.with_extension("audit.json"))
            .expect("fresh audit");
        serde_json::to_writer_pretty(f,&serde_json::json!({"composition":composition,"rows":audits,"contract":input_contract,"scope":"native scalar/cached/prepared feature parity and identity; not HDR human or encoder RD qualification"})).unwrap();
    }
    let f = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&out_path)
        .expect("fresh out");
    let mut w = BufWriter::new(f);
    let mut header = if score_bakes.is_empty() {
        String::from("dist_basename\tq")
    } else {
        String::from("ref_path\tdist_path")
    };
    if score_bakes.is_empty() {
        for i in 0..if requested_ids.is_some() {
            research_width
        } else if composition.is_some()
                && zensim::feature_v2::active_formula_revision()
                    == zensim::feature_v2::FormulaRevision::Rev5
        {
            zensim::research::full_width()
        } else {
            944
        } {
            header.push_str(&format!("\tf{i}"));
        }
    } else {
        for p in &score_bakes {
            header.push('\t');
            header.push_str(&p.display().to_string());
        }
        let f = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(out_path.with_extension("refusals.json"))
            .expect("fresh refusals");
        serde_json::to_writer_pretty(f, &serde_json::json!({"bakes":score_bakes,"refusals":refusals.into_inner().unwrap(),"rows":cells.len(),"entry":"BakeScorer::compute_hdr"})).unwrap();
    }
    writeln!(w, "{header}").unwrap();
    let mut n_ok = 0usize;
    for r in rows.iter().flatten() {
        writeln!(w, "{r}").unwrap();
        n_ok += 1;
    }
    eprintln!(
        "hdr944_extract: wrote {n_ok}/{} rows -> {out_path:?}",
        cells.len()
    );
    w.flush().expect("flush complete output");
    if refused_count != 0 {
        eprintln!("REFUSED {refused_count} native HDR scores; retained keyed failure output");
        std::process::exit(1);
    }
    assert_eq!(
        n_ok,
        cells.len(),
        "SKIPped cells present — investigate before use"
    );
}

#[cfg(test)]
#[path = "hdr944_extract/tests.rs"]
mod tests;

#[cfg(test)]
mod registered_display_tests {
    use super::*;

    #[test]
    fn rgbe_display_mapping_and_robustness_keep_the_registered_policy() {
        let scratch = std::env::var_os("TMPDIR").expect("disk-backed TMPDIR required");
        let path = PathBuf::from(scratch).join(format!("hdr-display-contract-{}.hdr", std::process::id()));
        let mut fixture = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 2\n".to_vec();
        fixture.extend_from_slice(&[128, 64, 32, 129, 128, 128, 128, 136]);
        std::fs::OpenOptions::new().create_new(true).write(true).open(&path)
            .unwrap().write_all(&fixture).unwrap();
        let primary = decode_rousselot(&path, 179.).unwrap();
        let robustness = decode_rousselot(&path, 100.).unwrap();
        assert_eq!((primary.w, primary.h), (2, 1));
        let px = primary.linear.as_ref().unwrap();
        // Independent RGBE fixture: half-bin reconstruction, then the registered
        // efficacy and display ceiling. Above-SDR light must survive ingress.
        assert_eq!(px[0], [179.699_22, 90.199_22, 45.449_22, 1.]);
        assert_eq!(px[1], [1000., 1000., 1000., 1.]);
        assert_eq!(robustness.linear.unwrap()[0], [100.390625, 50.390625, 25.390625, 1.]);
        assert_eq!(primary.primaries, zensim::ColorPrimaries::Srgb);
        std::fs::remove_file(path).unwrap();
    }
}
