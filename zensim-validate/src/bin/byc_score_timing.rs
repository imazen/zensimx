//! Per-call scoring time of one bake on a fixed pair (by-construction perf prototype).
//!
//! Exploration record: `benchmarks/byconstruction_2026-10-22.md`. Builds a deterministic
//! textured reference of the requested size and a distorted copy (small noise), or reads a real
//! image and distorts it deterministically, warms up, then times `BakeScorer::compute` and prints
//! one JSON line: median and quartiles in milliseconds, plus the score (to confirm both arms score
//! the same pair). The caller alternates two arms in paired rounds (two builds, or one build under
//! two `ZENSIM_RING_FORM` values).
//!
//! `--dump PATH` also writes the score's and every emitted feature's bits (hex, one per line, line `k` is feature id `k − 1`) for
//! before/after bit-identity checks (`scripts/byconstruction/perf_bits.sh`); `--tier` forces a
//! dispatch tier by disabling archmage tokens process-wide, as `tier_audit_features` does.
//!
//! `--read-set` prints the bake's declared feature ids (comma-separated) and exits.
//!
//! Usage: byc_score_timing --bake PATH [--size N | --width W --height H | --image PNG]
//!        [--iters 30] [--warmup 5] [--dump PATH] [--tier v4x|v4|v3|scalar] [--read-set]
use std::time::Instant;

use zensim::{BakeScorer, RgbSlice};
use zensim_validate::bake_runtime::post_mode_params;

/// Disable the tokens above `tier` (x86-64 only; `v4x` disables nothing).
fn force_tier(tier: &str) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::{
            SimdToken, X64CryptoToken, X64V2Token, X64V3CryptoToken, X64V3Token, X64V4Token,
            X64V4xToken,
        };
        let off = |r: Result<(), _>, name: &str| {
            if let Err(e) = r {
                panic!("cannot disable {name}: {e:?}");
            }
        };
        let level = match tier {
            "v4x" => 0,
            "v4" => 1,
            "v3" => 2,
            "scalar" => 3,
            other => panic!("unknown tier {other}: expected v4x|v4|v3|scalar"),
        };
        if level >= 1 {
            off(
                X64V4xToken::dangerously_disable_token_process_wide(true),
                "v4x",
            );
        }
        if level >= 2 {
            off(
                X64V4Token::dangerously_disable_token_process_wide(true),
                "v4",
            );
        }
        if level >= 3 {
            off(
                X64V3CryptoToken::dangerously_disable_token_process_wide(true),
                "v3 crypto",
            );
            off(
                X64V3Token::dangerously_disable_token_process_wide(true),
                "v3",
            );
            off(
                X64CryptoToken::dangerously_disable_token_process_wide(true),
                "crypto",
            );
            off(
                X64V2Token::dangerously_disable_token_process_wide(true),
                "v2",
            );
        }
        let _ = X64V3Token::summon();
    }
    #[cfg(not(target_arch = "x86_64"))]
    assert!(
        tier == "v4x" || tier == "scalar",
        "tier {tier} is x86-64 only"
    );
}

/// A deterministic textured pair: the reference and a copy with uniform ±3 noise.
fn synthetic(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut rnd = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut src = Vec::with_capacity(w * h);
    let mut dst = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f64, y as f64);
            let base = [
                128.0
                    + 60.0 * (fx * 0.031).sin() * (fy * 0.017).cos()
                    + 30.0 * (fx * 0.23 + fy * 0.11).sin(),
                120.0 + 50.0 * (fy * 0.027).sin() + 25.0 * (fx * 0.19).cos(),
                110.0 + 40.0 * ((fx + fy) * 0.021).sin() + 20.0 * (fx * fy * 0.0007).cos(),
            ];
            let px = base.map(|v| v.clamp(0.0, 255.0) as u8);
            let noisy = base.map(|v| (v + 6.0 * (rnd() - 0.5)).clamp(0.0, 255.0) as u8);
            src.push(px);
            dst.push(noisy);
        }
    }
    (src, dst)
}

/// A real image and a deterministic distortion of it: each channel quantized to steps of 10
/// plus a hashed dither in −8..=7.
fn from_image(path: &str) -> (Vec<[u8; 3]>, Vec<[u8; 3]>, usize, usize) {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let src: Vec<[u8; 3]> = img.pixels().map(|p| p.0).collect();
    let dst = src
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let n = ((i as u32).wrapping_mul(2_654_435_761) >> 28) as i32 - 8;
            let f = |v: u8| ((i32::from(v / 10) * 10 + n).clamp(0, 255)) as u8;
            [f(p[0]), f(p[1]), f(p[2])]
        })
        .collect();
    (src, dst, w, h)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut bake, mut iters, mut warmup) = (None, 30usize, 5usize);
    let mut read_set = false;
    let (mut width, mut height, mut image, mut dump, mut tier) = (
        1024usize,
        None::<usize>,
        None::<String>,
        None::<String>,
        None::<String>,
    );
    let num = |v: Option<String>, flag: &str| -> usize {
        v.and_then(|v| v.parse().ok())
            .unwrap_or_else(|| panic!("{flag} N"))
    };
    while let Some(a) = args.next() {
        match a.as_str() {
            "--bake" => bake = args.next(),
            "--size" | "--width" => width = num(args.next(), &a),
            "--height" => height = Some(num(args.next(), "--height")),
            "--iters" => iters = num(args.next(), "--iters"),
            "--warmup" => warmup = num(args.next(), "--warmup"),
            "--image" => image = args.next(),
            "--dump" => dump = args.next(),
            "--tier" => tier = args.next(),
            "--read-set" => read_set = true,
            other => panic!("unknown argument {other}"),
        }
    }
    let bake = bake.expect("--bake PATH");
    if read_set {
        let model = zenpredict::Model::from_bytes(&std::fs::read(&bake).expect("read bake"))
            .expect("parse bake");
        let ids = zensim::declared_feature_ids(&model).expect("declared feature ids");
        let ids: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
        println!("{}", ids.join(","));
        return;
    }
    if let Some(t) = &tier {
        force_tier(t);
    }
    let (src, dst, w, h) = match &image {
        Some(path) => from_image(path),
        None => {
            let h = height.unwrap_or(width);
            let (s, d) = synthetic(width, h);
            (s, d, width, h)
        }
    };
    let model = zenpredict::Model::from_bytes(&std::fs::read(&bake).expect("read bake"))
        .expect("parse bake");
    let params = post_mode_params("raw").expect("raw post");
    let mut scorer = BakeScorer::new(&model)
        .expect("score metadata")
        .with_score_disposition(&params)
        .expect("disposition");
    let (s, d) = (RgbSlice::new(&src, w, h), RgbSlice::new(&dst, w, h));
    let mut score = 0.0;
    for _ in 0..warmup {
        score = scorer.compute(&s, &d, None).expect("score").score();
    }
    let mut ms = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let r = scorer.compute(&s, &d, None).expect("score");
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        score = r.score();
    }
    ms.sort_by(f64::total_cmp);
    let q = |p: f64| ms[((ms.len() - 1) as f64 * p).round() as usize];
    if let Some(path) = &dump {
        let r = scorer.compute(&s, &d, None).expect("score");
        let mut out = format!("{:016x}\n", r.score().to_bits());
        for f in r.features() {
            out.push_str(&format!("{:016x}\n", f.to_bits()));
        }
        std::fs::write(path, out).unwrap_or_else(|e| panic!("{path}: {e}"));
    }
    println!(
        "{{\"width\":{w},\"height\":{h},\"iters\":{iters},\"median_ms\":{:.4},\"p25_ms\":{:.4},\"p75_ms\":{:.4},\"score\":{score:.6},\"ring_form\":\"{}\",\"tier\":\"{}\"}}",
        q(0.5),
        q(0.25),
        q(0.75),
        std::env::var("ZENSIM_RING_FORM").unwrap_or_else(|_| "raw".into()),
        tier.as_deref().unwrap_or("default")
    );
}
