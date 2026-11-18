//! Interleaved resident RGB8 scoring and fresh-process high-water measurement.
#![forbid(unsafe_code)]

use std::{
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use archmage::{SimdToken, X64V3Token, X64V4Token, X64V4xToken};
use research::rev8::{self, RgbRows, Variant};

const ARMS: [(&str, Option<Variant>); 4] = [
    ("zensim-a-rev7", None),
    ("rev8-alg-s4", Some(Variant::AlgS4)),
    ("rev8-alg-s5", Some(Variant::AlgS5)),
    ("rev8-hybrid-s5", Some(Variant::HybridS5)),
];
const IMAGES: usize = 20;

struct Pair {
    a: Vec<[u8; 3]>,
    b: Vec<[u8; 3]>,
    width: usize,
    height: usize,
}

fn mix(x: u32) -> u32 {
    let x = (x ^ (x >> 16)).wrapping_mul(0x7feb_352d);
    let x = (x ^ (x >> 15)).wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

impl Pair {
    fn new(id: usize, width: usize, height: usize) -> Self {
        let seed = mix(id as u32 + 1);
        let mut a = Vec::with_capacity(width * height);
        let mut b = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let tile = mix(seed ^ ((x / 32) as u32 * 73) ^ ((y / 48) as u32 * 137));
                let noise = mix(seed ^ (x as u32 * 193) ^ (y as u32 * 283));
                let u = x * 256 / width;
                let v = y * 256 / height;
                let scene = ((u * (id + 6) + v * (id + 4)) >> 4) as i32 % 128;
                let region = ((tile >> 24) as i32 - 128) / (4 + (id % 4) as i32);
                let grain = ((noise >> 24) as i32 - 128) / (5 + (id % 5) as i32);
                let texture = if (u + v + id * 13) % 255 < 120 {
                    grain + (x % (3 + id % 7)) as i32 * 3
                } else {
                    grain / 5
                };
                let signal = 58 + scene + region + texture;
                let rgb: [i32; 3] = [
                    signal + u as i32 / 7,
                    signal - v as i32 / 9,
                    signal + (u as i32 - v as i32) / 8,
                ];
                let error = match id % 5 {
                    0 => 6 - grain,
                    1 => ((noise >> 16 & 63) as i32 - 32) / 2,
                    2 => ((x / 8 + y / 8) % 2) as i32 * 14 - 7,
                    3 => (x % 7) as i32 * 3 - 10,
                    _ => -(texture / 2) - 2,
                };
                a.push(rgb.map(|p| p.clamp(0, 255) as u8));
                b.push(core::array::from_fn(|c| {
                    let p = rgb[c] + error * (c as i32 + 2) / 3;
                    let q = [5, 7, 6, 3, 4][id % 5];
                    (p.div_euclid(q) * q).clamp(0, 255) as u8
                }));
            }
        }
        assert_ne!(a, b, "a timing pair must exercise the dense scorer");
        Self {
            a,
            b,
            width,
            height,
        }
    }

    fn score(&self, variant: Option<Variant>, baseline: &rev7::Zensim, parallel: bool) -> f64 {
        if let Some(variant) = variant {
            let a = RgbRows::new(&self.a, self.width, self.height, self.width).unwrap();
            let b = RgbRows::new(&self.b, self.width, self.height, self.width).unwrap();
            rev8::compute(a, b, variant, parallel).unwrap().score
        } else {
            let a = rev7::RgbSlice::new(&self.a, self.width, self.height);
            let b = rev7::RgbSlice::new(&self.b, self.width, self.height);
            baseline.compute(&a, &b).unwrap().score()
        }
    }
}

fn configure_tier(tier: &str) {
    match tier {
        "v3" => {
            X64V4xToken::dangerously_disable_token_process_wide(true).unwrap();
            X64V4Token::dangerously_disable_token_process_wide(true).unwrap();
            assert!(X64V3Token::summon().is_some());
            assert!(X64V4Token::summon().is_none() && X64V4xToken::summon().is_none());
            assert_eq!(rev8::tier(), "X64V3Token");
        }
        "v4x" => {
            assert!(X64V4xToken::summon().is_some());
            assert_eq!(rev8::tier(), "X64V4xToken");
        }
        _ => panic!("unknown tier {tier}"),
    }
}

fn latency(width: usize, height: usize, parallel: bool, output: &Path) {
    assert!(!output.exists());
    let rounds = std::env::var("REV8_MAX_ROUNDS").map_or(36, |v| v.parse::<usize>().unwrap());
    assert!((32..=64).contains(&rounds));
    let result = zenbench::run(|suite| {
        for id in 0..IMAGES {
            let pair = Arc::new(Pair::new(id, width, height));
            suite.compare(format!("{width}x{height}/{id:02}"), |group| {
                let cfg = group.config();
                cfg.min_rounds = 32;
                cfg.max_rounds = rounds;
                cfg.auto_rounds = false;
                cfg.min_iterations = 1;
                cfg.max_iterations = 1;
                cfg.warmup_time = Duration::from_millis(100);
                cfg.max_time = Duration::from_secs(900);
                cfg.max_wall_time = Duration::from_secs(1200);
                for (name, variant) in ARMS {
                    let pair = Arc::clone(&pair);
                    let scorer = rev7::Zensim::new(rev7::ZensimProfile::A).with_parallel(parallel);
                    group.bench(name, move |b| {
                        b.iter(|| zenbench::black_box(pair.score(variant, &scorer, parallel)));
                    });
                }
            });
        }
    });
    result.save(output).unwrap();
    assert!(!result.unreliable, "zenbench declined the capture");
    for group in &result.comparisons {
        let clean = group
            .samples
            .iter()
            .filter(|r| r.iterations == 1 && r.gate_clean == Some(true))
            .count();
        assert!(
            clean >= 32,
            "fewer than 32 paired clean rounds in {}",
            group.group_name
        );
    }
}

fn descriptive(width: usize, height: usize, parallel: bool, fixture: usize) {
    assert!(
        fixture < IMAGES,
        "fixture ID must be in the original packet"
    );
    let pair = Pair::new(fixture, width, height);
    let checksum = |rgb: &[[u8; 3]]| {
        rgb.iter()
            .flatten()
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, &byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
            })
    };
    println!(
        "fixture\t{fixture:02}\t{:016x}\t{:016x}",
        checksum(&pair.a),
        checksum(&pair.b)
    );
    let baseline = rev7::Zensim::new(rev7::ZensimProfile::A).with_parallel(parallel);
    let warmup =
        ARMS.map(|(_, variant)| zenbench::black_box(pair.score(variant, &baseline, parallel)));
    assert!(warmup.iter().all(|s| s.is_finite()));

    let mut results = [[(0u128, 0u64); ARMS.len()]; 3];
    for (round, samples) in results.iter_mut().enumerate() {
        for (slot, sample) in samples.iter_mut().enumerate() {
            let arm = if round % 2 == 0 {
                slot
            } else {
                ARMS.len() - 1 - slot
            };
            let start = Instant::now();
            let score = zenbench::black_box(pair.score(ARMS[arm].1, &baseline, parallel));
            let nanoseconds = start.elapsed().as_nanos();
            assert_eq!(score.to_bits(), warmup[arm].to_bits());
            *sample = (nanoseconds, score.to_bits());
        }
    }

    for (round, samples) in results.into_iter().enumerate() {
        for (slot, (elapsed, score)) in samples.into_iter().enumerate() {
            let arm = if round % 2 == 0 {
                slot
            } else {
                ARMS.len() - 1 - slot
            };
            let (name, _) = ARMS[arm];
            println!(
                "sample\t{}\t{}\t{name}\t{elapsed}\t{score:016x}",
                round + 1,
                slot + 1
            );
        }
    }
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [tier, threads, width, height, mode, destination] = args.as_slice() else {
        panic!(
            "usage: zensim-rev8-runtime TIER THREADS WIDTH HEIGHT latency PATH | descriptive FIXTURE | rss ARM"
        );
    };
    let threads: usize = threads.parse().unwrap();
    assert!(matches!(threads, 1 | 16));
    configure_tier(tier);
    assert_eq!(rayon::current_num_threads(), threads);
    let parallel = threads == 16;
    let width = width.parse().unwrap();
    let height = height.parse().unwrap();
    println!(
        "tier={} threads={threads} geometry={width}x{height}",
        rev8::tier()
    );
    match mode.as_str() {
        "latency" => latency(width, height, parallel, Path::new(destination)),
        "descriptive" => descriptive(width, height, parallel, destination.parse().unwrap()),
        "rss" => {
            let (_, variant) = ARMS.iter().find(|(name, _)| name == destination).unwrap();
            let baseline = rev7::Zensim::new(rev7::ZensimProfile::A).with_parallel(parallel);
            for id in 0..IMAGES {
                let pair = Pair::new(id, width, height);
                let score = zenbench::black_box(pair.score(*variant, &baseline, parallel));
                assert!(score.is_finite());
                println!("score {id:02} {:016x}", score.to_bits());
            }
        }
        _ => panic!("unknown mode {mode}"),
    }
}
