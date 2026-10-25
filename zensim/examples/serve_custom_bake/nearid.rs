//! Label-free near-identity measurements through the existing serving owners.
//! One process per model/revision. Input admission is pinned before pixel access.
#[path = "../support/zen_io.rs"]
mod zen_io;

use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path};
use zensim::{BakeScorer, RgbSlice, Zensim, ZensimProfile};

const FRAGILITY: [usize; 10] = [422, 480, 509, 538, 567, 596, 625, 654, 683, 712];

#[derive(Deserialize)]
struct Source {
    ref_group: String,
    ref_path: String,
    content_class: String,
    role: String,
    file_sha256: String,
    ref_pixels_sha256: Option<String>,
}
#[derive(Deserialize)]
struct Packet {
    sources: Vec<Source>,
}
fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}
fn step(value: u8, sign: i16) -> u8 {
    let next = i16::from(value) + sign;
    if !(0..=255).contains(&next) {
        (i16::from(value) - sign) as u8
    } else {
        next as u8
    }
}
fn permutation(n: usize) -> Vec<usize> {
    let mut order: Vec<_> = (0..n).collect();
    let mut state = 20261009;
    for i in (1..n).rev() {
        let j = (random(&mut state) % (i + 1) as u64) as usize;
        order.swap(i, j);
    }
    order
}
fn perturb(pixels: &[[u8; 3]], order: &[usize], count: usize) -> Vec<[u8; 3]> {
    let mut out = pixels.to_vec();
    for (i, &pos) in order[..count].iter().enumerate() {
        let channel = i % 3;
        out[pos][channel] = step(out[pos][channel], if i % 2 == 0 { 1 } else { -1 });
    }
    out
}
fn noise(pixels: &[[u8; 3]], amplitude: i16) -> Vec<[u8; 3]> {
    let mut state = 20261009;
    pixels
        .iter()
        .map(|p| {
            p.map(|v| {
                let u = random(&mut state) >> 1;
                let quantile = u as f64 / ((u64::MAX >> 1) as f64 + 1.0);
                let delta = (quantile * f64::from(2 * amplitude + 1)).floor() as i16 - amplitude;
                (i16::from(v) + delta).clamp(0, 255) as u8
            })
        })
        .collect()
}
fn gaussian(pixels: &[[u8; 3]], w: usize, h: usize, sigma: f32) -> Vec<[u8; 3]> {
    let cfg = zenresize::ResizeConfig::builder(w as u32, h as u32, w as u32, h as u32)
        .filter(zenresize::Filter::Box)
        .format(zenresize::PixelDescriptor::RGB8_SRGB)
        .srgb()
        .post_blur(sigma)
        .build();
    let input: Vec<_> = pixels.iter().flatten().copied().collect();
    zenresize::Resizer::new(&cfg)
        .resize(&input)
        .as_chunks::<3>()
        .0
        .to_vec()
}

pub(super) fn contact(args: &[String]) {
    assert_eq!(args.len(), 2);
    let packet: Packet = serde_json::from_slice(&fs::read(&args[0]).unwrap()).unwrap();
    let width = 6 * 160;
    let height = packet.sources.len().div_ceil(6) * 160;
    let mut canvas = vec![[255; 3]; width * height];
    for (i, source) in packet.sources.iter().enumerate() {
        assert_eq!(source.role, "TRAIN");
        assert_eq!(
            sha(&fs::read(&source.ref_path).unwrap()),
            source.file_sha256
        );
        let (p, w, h) = zen_io::decode_rgb8(Path::new(&source.ref_path));
        let thumb = zen_io::resize_rgb8(&p, w, h, 160, 160);
        for y in 0..160 {
            let start = (i / 6 * 160 + y) * width + i % 6 * 160;
            canvas[start..start + 160].copy_from_slice(&thumb[y * 160..(y + 1) * 160]);
        }
        eprintln!(
            "contact {i}: {} ({})",
            source.ref_group, source.content_class
        );
    }
    fs::write(&args[1], zen_io::encode_png_rgb8(&canvas, width, height)).unwrap();
}

pub(super) fn run(args: &[String]) {
    let name = &args[2];
    // A registered candidate (E33 section 9.2) serves like seed0 at Rev5; its SHA is pinned by the caller.
    let candidate = name.strip_prefix("candidate-").filter(|label| {
        !label.is_empty()
            && label
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
    });
    assert_eq!(
        args.len(),
        if candidate.is_some() { 5 } else { 4 },
        "--nearid PACKET.json OUTPUT_DIR seed0|B|A MODEL.bin, or candidate-<label> MODEL.bin SHA256"
    );
    let dynamic = name == "seed0" || candidate.is_some();
    let packet: Packet = serde_json::from_slice(&fs::read(&args[0]).unwrap()).unwrap();
    assert_eq!(packet.sources.len(), 24);
    for class in ["photo", "screen", "line_art"] {
        assert_eq!(
            packet
                .sources
                .iter()
                .filter(|s| s.content_class == class)
                .count(),
            8
        );
    }
    assert!(packet.sources.iter().all(|s| s.role == "TRAIN"));
    let root = Path::new(&args[1]);
    fs::create_dir_all(root).unwrap();
    assert!(matches!(name.as_str(), "seed0" | "B" | "A") || candidate.is_some());
    // A dynamic model serves at revision 5. A by-construction candidate (benchmarks/byconstruction_2026-10-22.md)
    // may serve at revision 7, which its bake must declare.
    let revision = std::env::var("ZENSIM_FORMULA_REV").unwrap();
    if candidate.is_some() {
        assert!(matches!(revision.as_str(), "5" | "7"));
    } else {
        assert_eq!(revision, if dynamic { "5" } else { "1" });
    }
    let bytes = fs::read(&args[3]).unwrap();
    let expected = match name.as_str() {
        "seed0" => "f803b74c4252952f337abdc0234c2930839d45dddc32ae9b8b5296d6c840f400",
        "B" => "a96a5a66cd16282ded84580c37207b403334f1af069c924d9883346d23941276",
        "A" => "de0ddb3dc78b8b0a4fc99645f949acb73e7e08bec9ed4d86793605d882e9176a",
        _ => args[4].as_str(),
    };
    assert_eq!(sha(&bytes), expected, "frozen model bytes");
    let model = zenpredict::Model::from_bytes(&bytes).unwrap();
    if revision == "7" {
        assert_eq!(
            model.metadata().get_utf8("zentrain.formula_revision").ok(),
            Some("7")
        );
    }
    let mut scorer = BakeScorer::new(&model).unwrap().with_parallel(false);
    let request = zensim::research::Request::for_bake_bytes(&bytes).unwrap();
    let metadata = json!({"model":name,"model_sha256":sha(&bytes),
        "consumed_feature_ids":request.want().iter_slots().collect::<Vec<_>>(),
        "formula_revision":std::env::var("ZENSIM_FORMULA_REV").unwrap(),
        "fragility_neutral_ids":FRAGILITY,"fragility_neutral_value":0.0,
        "packet_sha256":sha(&fs::read(&args[0]).unwrap()),
        "raw_model_score_semantics":"complete calibrated feature-only model score",
        "blur":"zenresize e3975fb9 same-size Box/sRGB post_blur, RGB8 rounding",
        "jpeg":"zen_io default YCbCr 4:4:4"});
    let mut metadata_file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join(format!("{name}.metadata.json")))
        .unwrap();
    writeln!(metadata_file, "{metadata}").unwrap();
    let mut raw = BakeScorer::new(&model)
        .unwrap()
        .with_parallel(false)
        .without_output_calibration();
    #[allow(deprecated)]
    let named = Zensim::new(if name == "A" {
        ZensimProfile::A
    } else {
        ZensimProfile::B
    })
    .with_parallel(false);
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join(format!("{name}.jsonl")))
        .unwrap();
    for source in &packet.sources {
        let original = fs::read(&source.ref_path).unwrap();
        assert_eq!(
            sha(&original),
            source.file_sha256,
            "source pin before decode"
        );
        if name == "seed0" {
            let archived = root.join(format!("source_{}.png", source.file_sha256));
            if archived.exists() {
                assert_eq!(fs::read(&archived).unwrap(), original);
            } else {
                fs::write(&archived, &original).unwrap();
            }
        }
        let (pixels, w, h) = zen_io::decode_rgb8(Path::new(&source.ref_path));
        if let Some(expected) = &source.ref_pixels_sha256 {
            assert_eq!(
                &sha(bytemuck::cast_slice(&pixels)),
                expected,
                "TRAIN bank pixel pin"
            );
        }
        let order = permutation(pixels.len());
        let mut rungs = vec![(
            "identity".to_string(),
            "0".to_string(),
            pixels.clone(),
            None,
        )];
        for sign in [1, -1] {
            let mut out = pixels.clone();
            out[order[0]][0] = step(out[order[0]][0], sign);
            rungs.push((format!("one_pixel_{sign}"), "1".into(), out, None));
        }
        for p in [1e-5_f64, 1e-4, 1e-3, 1e-2, 0.1, 1.0] {
            let n = (p * pixels.len() as f64).ceil() as usize;
            rungs.push((
                "fraction".into(),
                p.to_string(),
                perturb(&pixels, &order, n),
                None,
            ));
        }
        for k in [1, 2, 3, 4, 6, 8] {
            rungs.push(("noise".into(), k.to_string(), noise(&pixels, k), None));
        }
        for q in [100, 99, 98, 97, 95, 92, 90] {
            let encoded = zen_io::encode_jpeg_q_subsampled(
                &pixels,
                w,
                h,
                q,
                zenjpeg::encoder::ChromaSubsampling::None,
            );
            let path = root.join(format!("{}_q{q}.jpg", sha(source.ref_group.as_bytes())));
            // The fixture is shared across processes; identical bytes are mandatory.
            if path.exists() {
                assert_eq!(fs::read(&path).unwrap(), encoded);
            } else {
                fs::write(&path, &encoded).unwrap();
            }
            let (decoded, dw, dh) = zen_io::decode_rgb8(&path);
            assert_eq!((w, h), (dw, dh));
            rungs.push((
                "zenjpeg444".into(),
                q.to_string(),
                decoded,
                Some(sha(&encoded)),
            ));
        }
        for sigma in [0.05, 0.1, 0.2, 0.3, 0.5] {
            rungs.push((
                "gaussian".into(),
                sigma.to_string(),
                gaussian(&pixels, w, h, sigma),
                None,
            ));
        }
        let source_view = RgbSlice::new(&pixels, w, h);
        for (index, (ladder, rung, distorted, encoded_sha)) in rungs.into_iter().enumerate() {
            if name == "seed0" {
                let raw_pixels: &[u8] = bytemuck::cast_slice(&distorted);
                let path = root.join(format!("{}.rgb", sha(raw_pixels)));
                if path.exists() {
                    assert_eq!(fs::read(&path).unwrap(), raw_pixels);
                } else {
                    fs::write(&path, raw_pixels).unwrap();
                }
            }
            let dest = RgbSlice::new(&distorted, w, h);
            let changed = pixels
                .iter()
                .zip(&distorted)
                .filter(|(a, b)| a != b)
                .count();
            let mut native_features = None;
            let served = if dynamic {
                let result = scorer.compute(&source_view, &dest, None).unwrap();
                native_features = Some(result.features().to_vec());
                result.score()
            } else {
                named.compute(&source_view, &dest).unwrap().score()
            };
            let mut row = json!({"model":name,"model_sha256":sha(&bytes),"reference":source.ref_group,
                "class":source.content_class,"width":w,"height":h,"ladder":ladder,"rung":rung,
                "rung_index":index,"changed_pixels":changed,"identical":changed==0,
                "reference_pixels_sha256":sha(bytemuck::cast_slice(&pixels)),
                "dist_pixels_sha256":sha(bytemuck::cast_slice(&distorted)),
                "encoded_sha256":encoded_sha,"served_score":served,"served_bits":served.to_bits()});
            assert!(served.is_finite());
            if !dynamic && index == 1 {
                assert_eq!(
                    scorer
                        .compute(&source_view, &dest, None)
                        .unwrap()
                        .score()
                        .to_bits(),
                    served.to_bits(),
                    "named-profile / pinned dynamic bake agreement"
                );
            }
            if changed == 0 {
                assert_eq!(served, 100.0);
            }
            if dynamic {
                let canonical = zensim::research::extract(&request, &source_view, &dest).unwrap();
                let f = canonical.values();
                let native = native_features.as_ref().unwrap();
                for id in request.want().iter_slots() {
                    assert_eq!(
                        native[id].to_bits(),
                        f[id].to_bits(),
                        "canonical / served consumed feature f{id}: {} {ladder} {rung}",
                        source.ref_group
                    );
                }
                row["consumed_feature_mismatches"] = json!(0);
                let feature_score = scorer.score_features(f, w as u32, h as u32, None).unwrap();
                if changed != 0 {
                    assert_eq!(feature_score.to_bits(), served.to_bits());
                }
                let mut neutral = f.to_vec();
                for id in FRAGILITY {
                    neutral[id] = 0.0;
                }
                row["raw_model_score"] = json!(feature_score);
                row["precalibration_score"] =
                    json!(raw.score_features(f, w as u32, h as u32, None).unwrap());
                row["neutral_fragility_score"] = json!(
                    scorer
                        .score_features(&neutral, w as u32, h as u32, None)
                        .unwrap()
                );
                row["fragility_values"] = json!(FRAGILITY.map(|id| f[id]));
                row["consumed_feature_bits"] = json!(
                    request
                        .want()
                        .iter_slots()
                        .map(|id| f[id].to_bits())
                        .collect::<Vec<_>>()
                );
            }
            writeln!(output, "{row}").unwrap();
        }
        output.flush().unwrap();
        eprintln!("nearid {name}: {} {w}x{h} complete", source.ref_group);
    }
}

/// One E33 identity source: the file and decoded-pixel pins come from the
/// admitted instrument (the 38-row identity probe keys and NEARID's packet).
#[derive(Deserialize)]
struct E33Source {
    ref_group: String,
    ref_path: String,
    file_sha256: String,
    ref_pixels_sha256: Option<String>,
}

#[derive(Deserialize)]
struct Fx1 {
    schema: String,
    direct: Vec<usize>,
    products: Vec<[usize; 2]>,
}

/// A `--nonneg-distance`-shaped bake (zero hidden biases, ReLU, output
/// weights <= 0, output bias = pin) over E33 Arm A (`products` empty) or Arm
/// C, with deterministic positive first-layer weights. Its raw output is the
/// pin exactly when every model input is ±0 (registration §7 E7).
fn e33_nonneg_bake(fx1: &Fx1, products: bool, pin: f64, revision: &str) -> Vec<u8> {
    let mut read: Vec<usize> = fx1.direct.clone();
    let mut derived = String::from("zensim-derived-inputs v1\n");
    for id in &fx1.direct {
        derived.push_str(&format!("in {id}\n"));
    }
    if products {
        for [a, b] in &fx1.products {
            read.push(*b);
            derived.push_str(&format!("product {a} {b}\n"));
        }
    }
    read.sort_unstable();
    read.dedup();
    let width = fx1.direct.len() + if products { fx1.products.len() } else { 0 };
    let hidden = 8;
    let w1: Vec<f64> = (0..width * hidden)
        .map(|i| 0.01 + 0.001 * ((i * 7919) % 97) as f64)
        .collect();
    let mut metadata = vec![
        json!({"key":"zentrain.feature_ids","type":"utf8",
               "text":read.iter().map(usize::to_string).collect::<Vec<_>>().join(" ")}),
        json!({"key":"zentrain.formula_revision","type":"utf8","text":revision}),
    ];
    if products {
        metadata.push(json!({"key":"zensim.derived_inputs","type":"utf8","text":derived}));
    }
    let recipe = json!({
        "schema_hash": 33, "scaler_mean": vec![0.0; width],
        "scaler_scale": (0..width).map(|i| 0.5 + (i % 5) as f64).collect::<Vec<_>>(),
        "metadata": metadata,
        "layers": [
            {"in_dim":width,"out_dim":hidden,"activation":"relu","dtype":"f32",
             "weights":w1,"biases":vec![0.0; hidden]},
            {"in_dim":hidden,"out_dim":1,"activation":"identity","dtype":"f32",
             "weights":(0..hidden).map(|h| -0.5 - 0.1 * h as f64).collect::<Vec<_>>(),
             "biases":[pin]}]
    });
    zenpredict_bake::bake_from_json_str(&recipe.to_string()).unwrap()
}

/// E33 registration §7 E1/E3/E4/E7: on every admitted identity source and
/// every SIMD token permutation, the 410 direct (Difference) ids are exactly
/// ±0 after the declared f32 storage round trip, each fragility factor is
/// finite in (0, 1], each product is ±0, and nonneg A/C bakes over those
/// vectors score exactly the pin. E1 is zensim's own registry check of the
/// pairing (`BakeScorer::new` refuses a non-Difference or cross-cell pair).
pub(super) fn e33_identity(args: &[String]) {
    assert!(
        args.len() >= 4,
        "--e33-identity SOURCES.json FX1.json PRODUCTION_MODEL.bin OUTPUT.jsonl [CANDIDATE.bin ...]"
    );
    // Revision 5, or 7 for the by-construction revision-7 candidates: the production bake only names the read
    // set, the extraction runs at the process revision, and the nonneg bakes declare it.
    let revision = std::env::var("ZENSIM_FORMULA_REV").unwrap();
    assert!(matches!(revision.as_str(), "5" | "7"));
    let sources: Vec<E33Source> = serde_json::from_slice(&fs::read(&args[0]).unwrap()).unwrap();
    let fx1_bytes = fs::read(&args[1]).unwrap();
    let fx1: Fx1 = serde_json::from_slice(&fx1_bytes).unwrap();
    assert_eq!(fx1.schema, "zensim-fx1-v1");
    let bytes = fs::read(&args[2]).unwrap();
    assert_eq!(
        sha(&bytes),
        "f803b74c4252952f337abdc0234c2930839d45dddc32ae9b8b5296d6c840f400",
        "the production read set defines the extraction plan"
    );
    let request = zensim::research::Request::for_bake_bytes(&bytes).unwrap();
    let want: Vec<usize> = request.want().iter_slots().collect();
    let mut read: Vec<usize> = fx1.direct.clone();
    read.extend(fx1.products.iter().map(|p| p[1]));
    read.sort_unstable();
    read.dedup();
    assert_eq!(want, read, "production read set == fx1 read set (410 + 10)");
    let mut frag_ids: Vec<usize> = fx1.products.iter().map(|p| p[1]).collect();
    frag_ids.sort_unstable();
    frag_ids.dedup();
    assert_eq!(
        frag_ids, FRAGILITY,
        "fx1 reference factors are the ten fragility slots"
    );
    let pin = 100.0;
    let bake_a = e33_nonneg_bake(&fx1, false, pin, &revision);
    let bake_c = e33_nonneg_bake(&fx1, true, pin, &revision);
    let model_a = zenpredict::Model::from_bytes(&bake_a).unwrap();
    let model_c = zenpredict::Model::from_bytes(&bake_c).unwrap();
    // E1: zensim validates every product pair against its own registry.
    let mut scorer_a = BakeScorer::new(&model_a).unwrap().with_parallel(false);
    let mut scorer_c = BakeScorer::new(&model_c).unwrap().with_parallel(false);
    // E8: real candidate bakes (trainer outputs, packed bakes) must serve exactly 100 on every identity vector.
    let extra_bytes: Vec<(String, Vec<u8>)> = args[4..]
        .iter()
        .map(|p| (p.clone(), fs::read(p).unwrap()))
        .collect();
    let extra_models: Vec<zenpredict::Model> = extra_bytes
        .iter()
        .map(|(_, b)| zenpredict::Model::from_bytes(b).unwrap())
        .collect();
    let mut extra: Vec<BakeScorer> = extra_models
        .iter()
        .map(|m| BakeScorer::new(m).unwrap().with_parallel(false))
        .collect();
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[3])
        .unwrap();
    let mut decoded = Vec::new();
    for source in &sources {
        let original = fs::read(&source.ref_path).unwrap();
        assert_eq!(
            sha(&original),
            source.file_sha256,
            "source pin before decode"
        );
        let (pixels, w, h) = zen_io::decode_rgb8(Path::new(&source.ref_path));
        if let Some(expected) = &source.ref_pixels_sha256 {
            assert_eq!(&sha(bytemuck::cast_slice(&pixels)), expected, "pixel pin");
        }
        decoded.push((source.ref_group.clone(), pixels, w, h));
    }
    let mut reference: Vec<Option<Vec<u64>>> = vec![None; decoded.len()];
    let mut perm_names = Vec::new();
    let report = archmage::testing::for_each_token_permutation(
        archmage::testing::CompileTimePolicy::WarnStderr,
        |perm| {
            perm_names.push(perm.to_string());
            for (i, (group, pixels, w, h)) in decoded.iter().enumerate() {
                let view = RgbSlice::new(pixels, *w, *h);
                let canonical = zensim::research::extract(&request, &view, &view).unwrap();
                let f: Vec<f64> = canonical
                    .values()
                    .iter()
                    .map(|v| f64::from(*v as f32))
                    .collect();
                let mut nonzero_direct = Vec::new();
                for &id in &fx1.direct {
                    if f[id] != 0.0 {
                        nonzero_direct.push((id, f[id]));
                    }
                }
                let mut bad_fragility = Vec::new();
                let mut nonzero_products = 0usize;
                for &[a, b] in &fx1.products {
                    let frag = f[b];
                    if !(frag.is_finite() && frag > 0.0 && frag <= 1.0) {
                        bad_fragility.push((b, frag));
                    }
                    let product = (f[a] as f32) * (frag as f32);
                    if product != 0.0 {
                        nonzero_products += 1;
                    }
                }
                let score_a = scorer_a
                    .score_features(&f, *w as u32, *h as u32, None)
                    .unwrap();
                let score_c = scorer_c
                    .score_features(&f, *w as u32, *h as u32, None)
                    .unwrap();
                let extra_scores: Vec<f64> = extra
                    .iter_mut()
                    .map(|s| s.score_features(&f, *w as u32, *h as u32, None).unwrap())
                    .collect();
                let bits: Vec<u64> = want.iter().map(|&id| f[id].to_bits()).collect();
                let tier_identical = match &reference[i] {
                    None => {
                        reference[i] = Some(bits.clone());
                        true
                    }
                    Some(r) => *r == bits,
                };
                let row = json!({"permutation":perm.to_string(),"reference":group,"width":w,"height":h,
                    "direct_ids":fx1.direct.len(),"nonzero_direct":nonzero_direct,
                    "fragility_values":FRAGILITY.map(|id| f[id]),
                    "bad_fragility":bad_fragility,"nonzero_products":nonzero_products,
                    "nonneg_a_score_bits":score_a.to_bits(),"nonneg_c_score_bits":score_c.to_bits(),
                    "pin_bits":pin.to_bits(),"read_set_bits_equal_first_permutation":tier_identical,
                    "candidate_score_bits":extra_scores.iter().map(|v| v.to_bits()).collect::<Vec<_>>()});
                writeln!(output, "{row}").unwrap();
                assert!(
                    nonzero_direct.is_empty(),
                    "{group} {perm}: nonzero difference at identity"
                );
                assert!(
                    bad_fragility.is_empty(),
                    "{group} {perm}: fragility outside (0,1]"
                );
                assert_eq!(nonzero_products, 0, "{group} {perm}");
                assert_eq!(
                    score_a.to_bits(),
                    pin.to_bits(),
                    "{group} {perm}: A identity"
                );
                assert_eq!(
                    score_c.to_bits(),
                    pin.to_bits(),
                    "{group} {perm}: C identity"
                );
                assert!(
                    tier_identical,
                    "{group} {perm}: read set differs across tiers"
                );
                for ((path, _), score) in extra_bytes.iter().zip(&extra_scores) {
                    assert_eq!(
                        score.to_bits(),
                        pin.to_bits(),
                        "{group} {perm}: {path} identity"
                    );
                }
            }
            eprintln!("e33-identity: {} sources pass at {perm}", decoded.len());
        },
    );
    eprintln!("{report}");
    let summary = json!({"summary":true,"sources":decoded.len(),"permutations":perm_names,
        "fx1_sha256":sha(&fx1_bytes),"production_model_sha256":sha(&bytes),
        "sources_sha256":sha(&fs::read(&args[0]).unwrap()),"rows":decoded.len()*perm_names.len(),
        "candidates":extra_bytes.iter().map(|(p, b)| json!({"path":p,"sha256":sha(b)})).collect::<Vec<_>>()});
    writeln!(output, "{summary}").unwrap();
    output.flush().unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fraction_is_nested_and_always_changes_selected_pixels() {
        let pixels = vec![[0, 255, 128]; 101];
        let order = permutation(pixels.len());
        let a = perturb(&pixels, &order, 7);
        let b = perturb(&pixels, &order, 81);
        assert_eq!(a.iter().zip(&pixels).filter(|(a, b)| a != b).count(), 7);
        assert_eq!(b.iter().zip(&pixels).filter(|(a, b)| a != b).count(), 81);
        for &i in &order[..7] {
            assert_eq!(a[i], b[i]);
        }
    }
    #[test]
    fn shared_uniform_noise_is_bounded_and_nested_after_clamping() {
        let pixels: Vec<_> = (0..1024).map(|i| [0, (i % 256) as u8, 255]).collect();
        let mut previous = pixels.clone();
        for amplitude in [0, 1, 2, 3, 4, 6, 8] {
            let out = noise(&pixels, amplitude);
            assert_eq!(out, noise(&pixels, amplitude), "deterministic seed stream");
            for ((source, before), after) in pixels.iter().zip(&previous).zip(&out) {
                for c in 0..3 {
                    let delta = after[c].abs_diff(source[c]);
                    assert!(delta <= amplitude as u8);
                    assert!(delta >= before[c].abs_diff(source[c]));
                }
            }
            previous = out;
        }
        assert_ne!(previous, pixels);
    }

    #[test]
    fn gaussian_zero_preserves_pixels_and_positive_blur_changes_edge() {
        let mut p = vec![[0; 3]; 81];
        p[40] = [255; 3];
        assert_eq!(gaussian(&p, 9, 9, 0.0), p);
        let b = gaussian(&p, 9, 9, 0.5);
        assert_ne!(b, p);
        assert!(b[40][0] < 255);
        assert!(b[39][0] > 0);
    }
}
