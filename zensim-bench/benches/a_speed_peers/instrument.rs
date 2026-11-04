//! Paired public-A / fast-ssim2 / selected margarine scoring, RGB8 sRGB.
//! The private margarine kernel is included unchanged from a pinned source
//! tree; no new margarine library API or subprocess timing is involved.
#![forbid(unsafe_code)]
#![allow(dead_code, deprecated)]
// harness=false still sets cfg(test), but does not run the included modules'
// unit tests; their test-only imports are otherwise reported as unused.
#![cfg_attr(test, allow(unused_imports))]
#![allow(clippy::manual_midpoint, clippy::chunks_exact_to_as_chunks)]
include!(env!("MARGARINE_KERNEL"));

use std::{path::Path, sync::Arc, time::Duration};

#[derive(serde::Deserialize)]
struct PairRecord {
    id: String,
    size: usize,
    width: usize,
    height: usize,
    reference: String,
    distorted: String,
}

struct Pair {
    record: PairRecord,
    reference: Vec<u8>,
    distorted: Vec<u8>,
}

impl Pair {
    fn load(root: &Path, record: PairRecord) -> Self {
        let reference = std::fs::read(root.join(&record.reference)).unwrap();
        let distorted = std::fs::read(root.join(&record.distorted)).unwrap();
        assert_eq!(reference.len(), record.width * record.height * 3);
        assert_eq!(reference.len(), distorted.len());
        assert_ne!(
            reference, distorted,
            "identity short-circuit is not a distortion benchmark"
        );
        Self {
            record,
            reference,
            distorted,
        }
    }

    fn score(&self, arm: &str, scorer: Option<&zensim::Zensim>) -> f64 {
        let (w, h) = (self.record.width, self.record.height);
        match arm {
            "zensim_a" | "serial" | "parallel" => {
                let a = zensim::RgbSlice::new(bytemuck::cast_slice(&self.reference), w, h);
                let b = zensim::RgbSlice::new(bytemuck::cast_slice(&self.distorted), w, h);
                scorer.unwrap().compute(&a, &b).unwrap().score()
            }
            "fast_ssim2" => {
                let a = fast_ssim2::PixelSlice::new(
                    &self.reference,
                    w as u32,
                    h as u32,
                    w * 3,
                    fast_ssim2::PixelDescriptor::RGB8_SRGB,
                )
                .unwrap();
                let b = fast_ssim2::PixelSlice::new(
                    &self.distorted,
                    w as u32,
                    h as u32,
                    w * 3,
                    fast_ssim2::PixelDescriptor::RGB8_SRGB,
                )
                .unwrap();
                fast_ssim2::compute_ssimulacra2(&a, &b).unwrap()
            }
            "margarine" => {
                let a = ingress::EncodedRows::new(
                    ingress::Samples::U8(&self.reference),
                    w,
                    h,
                    w * 3,
                    3,
                )
                .unwrap();
                let b = ingress::EncodedRows::new(
                    ingress::Samples::U8(&self.distorted),
                    w,
                    h,
                    w * 3,
                    3,
                )
                .unwrap();
                row_psycho::compute_geometry(&a, &b, 128, 512, &ButteraugliParams::default())
                    .unwrap()
                    .score
            }
            _ => panic!("unknown arm {arm}"),
        }
    }
}

fn main() {
    assert_eq!(CANDIDATE, "simd-row-malta");
    let args: Vec<_> = std::env::args().collect();
    let root = Path::new(&args[1]);
    let threads: usize = args[2].parse().unwrap();
    let parallel_gate = std::env::var_os("A_SPEED_PARALLEL_GATE").is_some();
    assert!(matches!(threads, 1 | 16) || (parallel_gate && threads == 4));
    assert_eq!(
        std::env::var("RAYON_NUM_THREADS").unwrap(),
        threads.to_string()
    );
    assert_eq!(rayon::current_num_threads(), threads);
    let size: usize = args[3].parse().unwrap();
    let records: Vec<PairRecord> =
        serde_json::from_slice(&std::fs::read(root.join("pairs.json")).unwrap()).unwrap();
    let selected: Vec<_> = records.into_iter().filter(|r| r.size == size).collect();
    assert!(
        selected.len() >= 20,
        "at least twenty images required for size {size}"
    );
    if args[4] == "scores" {
        assert!(parallel_gate);
        for record in selected {
            let pair = Pair::load(root, record);
            let serial = zensim::Zensim::new(zensim::ZensimProfile::A).with_parallel(false);
            let parallel = zensim::Zensim::new(zensim::ZensimProfile::A).with_parallel(true);
            let a = pair.score("serial", Some(&serial));
            let b = pair.score("parallel", Some(&parallel));
            println!(
                "{}\t{a:.17}\t{b:.17}\t{:016x}\t{:016x}",
                pair.record.id,
                a.to_bits(),
                b.to_bits()
            );
            assert_eq!(a.to_bits(), b.to_bits(), "{}", pair.record.id);
        }
        return;
    }
    if args[4] == "rss" {
        // One pair resident at a time, one arm per fresh process. Include all
        // twenty images so a content-dependent high-water mark is observable.
        for record in selected {
            let pair = Pair::load(root, record);
            let scorer = (args[5] == "zensim_a").then(|| {
                zensim::Zensim::new(zensim::ZensimProfile::A).with_parallel(threads == 16)
            });
            let score = pair.score(&args[5], scorer.as_ref());
            assert!(score.is_finite());
            println!("{}\t{}\t{score:.17}", pair.record.id, args[5]);
        }
        return;
    }
    assert_eq!(args[4], "latency");
    let output = &args[5];
    let pairs: Vec<_> = selected
        .into_iter()
        .map(|r| Arc::new(Pair::load(root, r)))
        .collect();
    let result = zenbench::run(|suite| {
        for pair in pairs {
            suite.compare(
                format!(
                    "{}/{}/{}",
                    size,
                    pair.record.width * pair.record.height,
                    pair.record.id
                ),
                |group| {
                    let cfg = group.config();
                    cfg.min_rounds = 32;
                    cfg.max_rounds = 64;
                    cfg.auto_rounds = false;
                    cfg.min_iterations = 1;
                    cfg.max_iterations = 1;
                    cfg.warmup_time = Duration::from_millis(100);
                    cfg.max_time = Duration::from_secs(600);
                    cfg.max_wall_time = Duration::from_secs(900);
                    let arms: &[&str] = if parallel_gate {
                        &["serial", "parallel"]
                    } else {
                        &["zensim_a", "fast_ssim2", "margarine"]
                    };
                    for &arm in arms {
                        let pair = pair.clone();
                        let scorer = matches!(arm, "zensim_a" | "serial" | "parallel").then(|| {
                            zensim::Zensim::new(zensim::ZensimProfile::A).with_parallel(
                                if parallel_gate {
                                    arm == "parallel"
                                } else {
                                    threads == 16
                                },
                            )
                        });
                        group.bench(arm, move |b| {
                            b.iter(|| zenbench::black_box(pair.score(arm, scorer.as_ref())))
                        });
                    }
                },
            );
        }
    });
    result.save(Path::new(output)).unwrap();
    assert!(
        !result.unreliable,
        "unreliable run retained but not admitted"
    );
    for group in &result.comparisons {
        let clean = group
            .samples
            .iter()
            .filter(|r| r.gate_clean == Some(true) && r.iterations == 1)
            .count();
        assert!(
            clean >= 32,
            "fewer than 32 clean single-call paired rounds: {}",
            group.group_name
        );
    }
}
