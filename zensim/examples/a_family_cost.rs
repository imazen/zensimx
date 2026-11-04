// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.
//! Measurement-only owner for the 2026-11-04 A low-effect-family cost study.
//! No feature, bake, model, or serving arithmetic is modified.
#[path = "support/zen_io.rs"]
mod zen_io;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;
use zensim::feature_set_id::SlotSet;
use zensim::research::{self, Request};
use zensim::{BakeScorer, RgbSlice};

#[derive(Deserialize, Serialize)]
struct Source {
    id: String,
    path: PathBuf,
    size: String,
    side: Option<usize>,
    long_edge: Option<usize>,
}

#[derive(Deserialize, Serialize)]
struct Pair {
    id: String,
    size: String,
    width: usize,
    height: usize,
    source_sha256: String,
    reference_sha256: String,
    distorted_sha256: String,
    jpeg_sha256: String,
}

fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn prepare(manifest: &Path, out: &Path) {
    assert!(!out.exists(), "preserve existing generated data");
    std::fs::create_dir_all(out).unwrap();
    let sources: Vec<Source> = serde_json::from_slice(&std::fs::read(manifest).unwrap()).unwrap();
    let mut pairs = Vec::new();
    for src in sources {
        let source_sha256 = hash(&std::fs::read(&src.path).unwrap());
        let (mut px, mut w, mut h) = zen_io::decode_rgb8(&src.path);
        if let Some(side) = src.side {
            let n = w.min(h);
            assert!(side <= n, "never upscale {}", src.id);
            let (x0, y0) = ((w - n) / 2, (h - n) / 2);
            let crop: Vec<[u8; 3]> = (0..n)
                .flat_map(|y| px[(y0 + y) * w + x0..(y0 + y) * w + x0 + n].iter().copied())
                .collect();
            if side == n {
                px = crop;
            } else {
                use zenresize::{Filter, PixelDescriptor, ResizeConfig, Resizer};
                let cfg = ResizeConfig::builder(n as u32, n as u32, side as u32, side as u32)
                    .filter(Filter::Mitchell)
                    .format(PixelDescriptor::RGB8_SRGB)
                    .build();
                let resized = Resizer::new(&cfg).resize(bytemuck::cast_slice(&crop));
                px = resized.as_chunks::<3>().0.to_vec();
            }
            w = side;
            h = side;
        }
        if let Some(edge) = src.long_edge.filter(|&edge| w.max(h) > edge) {
            use zenresize::{Filter, PixelDescriptor, ResizeConfig, Resizer};
            let tw = (w * edge / w.max(h)).max(1);
            let th = (h * edge / w.max(h)).max(1);
            let cfg = ResizeConfig::builder(w as u32, h as u32, tw as u32, th as u32)
                .filter(Filter::Mitchell)
                .format(PixelDescriptor::RGB8_SRGB)
                .build();
            let resized = Resizer::new(&cfg).resize(bytemuck::cast_slice(&px));
            px = resized.as_chunks::<3>().0.to_vec();
            w = tw;
            h = th;
        }
        let jpg = zen_io::encode_jpeg_q(&px, w, h, 40);
        let jpg_path = out.join(format!("{}.jpg", src.id));
        std::fs::write(&jpg_path, &jpg).unwrap();
        let (dist, dw, dh) = zen_io::decode_rgb8(&jpg_path);
        assert_eq!((w, h), (dw, dh));
        assert_ne!(px, dist, "timing must not take the identity short circuit");
        let reference: &[u8] = bytemuck::cast_slice(&px);
        let distorted: &[u8] = bytemuck::cast_slice(&dist);
        std::fs::write(out.join(format!("{}.ref.rgb", src.id)), reference).unwrap();
        std::fs::write(out.join(format!("{}.dist.rgb", src.id)), distorted).unwrap();
        pairs.push(Pair {
            id: src.id,
            size: src.size,
            width: w,
            height: h,
            source_sha256,
            reference_sha256: hash(reference),
            distorted_sha256: hash(distorted),
            jpeg_sha256: hash(&jpg),
        });
        std::fs::write(
            out.join("pairs.json"),
            serde_json::to_vec_pretty(&pairs).unwrap(),
        )
        .unwrap();
        println!("prepared {} {}x{}", pairs.last().unwrap().id, w, h);
    }
}

fn omitted(audit: &Path) -> BTreeSet<usize> {
    let names = [
        "basic × ssim",
        "basic × mse",
        "basic × contrast_inc",
        "v2 × ssim",
        "v2 × mse",
        "v2 × ssim_soft_peak",
        "v2 × masked_ssim",
        "v2 × masked_art",
        "v2 × masked_mse",
        "v2 × iw_mse",
        "v2 × edge_width_change",
    ];
    let text = std::fs::read_to_string(audit).unwrap();
    let mut lines = text.lines();
    let head: Vec<_> = lines.next().unwrap().split('\t').collect();
    let col = |key| head.iter().position(|&s| s == key).unwrap();
    let mut ids = BTreeSet::new();
    let mut families = BTreeSet::new();
    for line in lines {
        let cells: Vec<_> = line.split('\t').collect();
        if names.contains(&cells[col("analysis_family")]) && cells[col("factor")].is_empty() {
            ids.insert(cells[col("feature")].parse().unwrap());
            families.insert(cells[col("analysis_family")].to_string());
        }
    }
    assert_eq!(ids.len(), 156);
    assert_eq!(families.len(), 11);
    ids
}

struct Loaded {
    pair: Pair,
    reference: Vec<[u8; 3]>,
    distorted: Vec<[u8; 3]>,
}

fn load(root: &Path, size: &str) -> Vec<Loaded> {
    let pairs: Vec<Pair> =
        serde_json::from_slice(&std::fs::read(root.join("pairs.json")).unwrap()).unwrap();
    let loaded: Vec<_> = pairs
        .into_iter()
        .filter(|p| p.size == size)
        .map(|p| {
            let r = std::fs::read(root.join(format!("{}.ref.rgb", p.id))).unwrap();
            let d = std::fs::read(root.join(format!("{}.dist.rgb", p.id))).unwrap();
            assert_eq!(hash(&r), p.reference_sha256);
            assert_eq!(hash(&d), p.distorted_sha256);
            assert_eq!(r.len(), p.width * p.height * 3);
            assert_eq!(d.len(), r.len());
            Loaded {
                pair: p,
                reference: r.as_chunks::<3>().0.to_vec(),
                distorted: d.as_chunks::<3>().0.to_vec(),
            }
        })
        .collect();
    assert!(loaded.len() >= 20, "size floor requires 20 images");
    loaded
}

fn perf_control(command: &[u8]) {
    if let Some(path) = std::env::var_os("A_COST_PERF_CONTROL") {
        std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .unwrap()
            .write_all(command)
            .unwrap();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.len() >= 4,
        "prepare SOURCES OUT | bench/profile PAIRS BAKE AUDIT SIZE OUT THREADS [REPEATS]"
    );
    if args[1] == "prepare" {
        return prepare(Path::new(&args[2]), Path::new(&args[3]));
    }
    assert_eq!(std::env::var("ZENSIM_FORMULA_REV").as_deref(), Ok("7"));
    let root = Path::new(&args[2]);
    let bytes = std::fs::read(&args[3]).unwrap();
    assert_eq!(
        hash(&bytes),
        "185f43f7e5e1327c34ad18f0b73b96b872663726a0fac6ff524430d0d8430088"
    );
    let omit = omitted(Path::new(&args[4]));
    let size = &args[5];
    let out = Path::new(&args[6]);
    std::fs::create_dir_all(out).unwrap();
    let threads: usize = args[7].parse().unwrap();
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    let parallel = threads > 1;
    let model: &'static zenpredict::Model =
        Box::leak(Box::new(zenpredict::Model::from_bytes(&bytes).unwrap()));
    let mut scorer = BakeScorer::new(model).unwrap().with_parallel(parallel);
    let full = Request::for_bake_bytes(&bytes)
        .unwrap()
        .with_parallel(parallel);
    assert!(omit.iter().all(|&id| full.want().contains(id)));
    let reduced = Request::for_slots(
        SlotSet::from_slots(full.want().iter_slots().filter(|id| !omit.contains(id))),
        full.layout_width(),
    )
    .with_parallel(parallel);
    assert_eq!(full.want().len() - reduced.want().len(), 156);
    let loaded = load(root, size);
    for (i, l) in loaded.iter().enumerate() {
        let (w, h) = (l.pair.width, l.pair.height);
        let r = RgbSlice::new(&l.reference, w, h);
        let d = RgbSlice::new(&l.distorted, w, h);
        let served = scorer.compute(&r, &d, None).unwrap();
        let a = research::extract(&full, &r, &d).unwrap();
        let b = research::extract(&reduced, &r, &d).unwrap();
        for id in reduced.want().iter_slots() {
            assert_eq!(
                a.values()[id].to_bits(),
                b.values()[id].to_bits(),
                "request parity {} f{id}",
                l.pair.id
            );
            assert_eq!(
                a.values()[id].to_bits(),
                served.features()[id].to_bits(),
                "serving parity {} f{id}",
                l.pair.id
            );
        }
        if i == 0 {
            std::fs::write(out.join("full-manifest.json"), a.manifest_json()).unwrap();
            std::fs::write(out.join("reduced-manifest.json"), b.manifest_json()).unwrap();
            std::fs::write(
                out.join("requests.json"),
                serde_json::to_vec_pretty(&serde_json::json!({
                    "full_ids": full.want().iter_slots().collect::<Vec<_>>(),
                    "reduced_ids": reduced.want().iter_slots().collect::<Vec<_>>(),
                    "omitted_damage_ids": omit, "threads": threads,
                    "bake_sha256": hash(&bytes), "layout_width": full.layout_width(),
                    "build_commit": option_env!("ZENSIM_BUILD_COMMIT"),
                    "warmup_ms_per_arm": 100, "rounds": 32,
                }))
                .unwrap(),
            )
            .unwrap();
        }
    }
    if args[1] == "profile" {
        let repeats: usize = args[8].parse().unwrap();
        perf_control(b"enable\n");
        for _ in 0..repeats {
            for l in &loaded {
                let r = RgbSlice::new(&l.reference, l.pair.width, l.pair.height);
                let d = RgbSlice::new(&l.distorted, l.pair.width, l.pair.height);
                std::hint::black_box(scorer.compute(&r, &d, None).unwrap());
            }
        }
        perf_control(b"disable\n");
        return;
    }
    assert_eq!(args[1], "bench");
    for l in loaded {
        let l = std::sync::Arc::new(l);
        let result = zenbench::run(|suite| {
            suite.compare(format!("{}_{}t", l.pair.id, threads), |g| {
                let c = g.config();
                c.min_rounds(32)
                    .max_rounds(32)
                    .auto_rounds(false)
                    .warmup_time(Duration::from_millis(100))
                    .max_time(Duration::from_secs(600))
                    .max_wall_time(Duration::from_secs(900));
                c.min_iterations = 1;
                c.max_iterations = 1;
                c.min_sample_ns = 0;
                c.stack_jitter = false;
                g.baseline("extract492");
                for (name, req) in [
                    ("extract492", full.clone()),
                    ("extract336", reduced.clone()),
                ] {
                    let l = std::sync::Arc::clone(&l);
                    g.bench(name, move |b| {
                        b.iter(|| {
                            let r = RgbSlice::new(&l.reference, l.pair.width, l.pair.height);
                            let d = RgbSlice::new(&l.distorted, l.pair.width, l.pair.height);
                            std::hint::black_box(research::extract(&req, &r, &d).unwrap())
                        })
                    });
                }
                let l = std::sync::Arc::clone(&l);
                let mut scorer = BakeScorer::new(model).unwrap().with_parallel(parallel);
                g.bench("compute_A", move |b| {
                    b.iter(|| {
                        let r = RgbSlice::new(&l.reference, l.pair.width, l.pair.height);
                        let d = RgbSlice::new(&l.distorted, l.pair.width, l.pair.height);
                        std::hint::black_box(scorer.compute(&r, &d, None).unwrap())
                    })
                });
            });
        });
        result
            .save(out.join(format!("{}.json", l.pair.id)))
            .unwrap();
        println!("completed {} {}t", l.pair.id, threads);
    }
}
