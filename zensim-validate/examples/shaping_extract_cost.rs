//! Revision-7 extraction cost, separate from vector scoring and image decoding.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, path::Path, sync::Arc, time::Duration};
use zensim::{
    PixelFormat, StridedBytes,
    feature_set_id::SlotSet,
    research::{self, Request},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Workload {
    schema: String,
    admission_sha256: String,
    raw_ids: Vec<usize>,
    size: usize,
    threads: usize,
    pixel_sha256: Vec<[String; 2]>,
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// Reuse the deterministic integer pattern of featacc_extract_ab, with three
// fixed salt pairs. Arithmetic is explicitly u64 for host-independent pixels.
fn pixels(size: usize, salt: u64) -> Vec<u8> {
    (0..size * size)
        .flat_map(|i| {
            let k = ((i as u64).wrapping_add(salt * 7919)).wrapping_mul(2654435761) % 65521;
            [
                (k % 256) as u8,
                ((k / 3) % 256) as u8,
                ((k / 7) % 256) as u8,
            ]
        })
        .collect()
}

fn padded(bytes: &[u8], size: usize) -> Vec<u8> {
    let stride = size * 3 + 17;
    let mut out = vec![0xa5; stride * size];
    for (src, dst) in bytes
        .chunks_exact(size * 3)
        .zip(out.chunks_exact_mut(stride))
    {
        dst[..src.len()].copy_from_slice(src);
    }
    out
}

fn extract(
    req: &Request,
    src: &[u8],
    dst: &[u8],
    size: usize,
    stride: usize,
) -> Result<research::Extraction> {
    let src = StridedBytes::try_new(src, size, size, stride, PixelFormat::Srgb8Rgb)?;
    let dst = StridedBytes::try_new(dst, size, size, stride, PixelFormat::Srgb8Rgb)?;
    Ok(research::extract(req, &src, &dst)?)
}

fn bits(values: &[f64]) -> Result<Vec<u64>> {
    ensure!(values.iter().all(|v| v.is_finite()), "nonfinite extraction");
    Ok(values.iter().map(|v| v.to_bits()).collect())
}

fn validate(w: &Workload) -> Result<()> {
    ensure!(w.schema == "shaping-extract-cost-v1", "unknown workload");
    ensure!(
        w.admission_sha256.len() == 64 && w.admission_sha256.bytes().all(|b| b.is_ascii_hexdigit()),
        "admission hash required"
    );
    ensure!([256, 1024, 4096].contains(&w.size), "unregistered size");
    ensure!([1, 4].contains(&w.threads), "unregistered threads");
    let ids: BTreeSet<_> = w.raw_ids.iter().copied().collect();
    ensure!(
        !ids.is_empty() && ids.len() == w.raw_ids.len() && *ids.last().unwrap() < 1853,
        "invalid revision-7 read set"
    );
    ensure!(
        w.pixel_sha256.len() == 3,
        "three frozen pixel pairs required"
    );
    Ok(())
}

#[cfg(target_arch = "x86_64")]
fn dispatch() -> &'static str {
    use archmage::{SimdToken, X64V3Token, X64V4Token, X64V4xToken};
    if X64V4xToken::summon().is_some() {
        "auto-v4x"
    } else if X64V4Token::summon().is_some() {
        "auto-v4"
    } else if X64V3Token::summon().is_some() {
        "auto-v3"
    } else {
        "auto-scalar"
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn dispatch() -> &'static str {
    "automatic-non-x86"
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 4 && matches!(args[1].as_str(), "qualify" | "bench"),
        "usage: shaping_extract_cost qualify|bench WORKLOAD.json NEW_OUTPUT_DIR"
    );
    ensure!(
        std::env::var("ZENSIM_FORMULA_REV").as_deref() == Ok("7"),
        "explicit revision 7 required"
    );
    let bytes = std::fs::read(&args[2])?;
    let w: Workload = serde_json::from_slice(&bytes)?;
    validate(&w)?;
    let output = Path::new(&args[3]);
    std::fs::create_dir(output).context("preserve existing output")?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(w.threads)
        .build_global()?;
    let req = Arc::new(
        Request::for_slots(SlotSet::from_slots(w.raw_ids.iter().copied()), 1853)
            .with_parallel(w.threads > 1),
    );
    req.validate()?;
    let mut pairs = Vec::new();
    let mut references = Vec::new();
    for pair in 0..3 {
        let src = pixels(w.size, pair * 2 + 1);
        let dst = pixels(w.size, pair * 2 + 2);
        ensure!(src != dst, "identical extraction pair");
        ensure!(
            [digest(&src), digest(&dst)] == w.pixel_sha256[pair as usize],
            "pixel hash differs"
        );
        let packed = extract(&req, &src, &dst, w.size, w.size * 3)?;
        let manifest: serde_json::Value = serde_json::from_str(&packed.manifest_json())?;
        ensure!(
            manifest["formula_revision"] == "Rev7",
            "actual revision differs"
        );
        std::fs::write(
            output.join(format!("pair-{pair}-extraction.json")),
            packed.manifest_json(),
        )?;
        let expected = bits(packed.values())?;
        let strided = extract(
            &req,
            &padded(&src, w.size),
            &padded(&dst, w.size),
            w.size,
            w.size * 3 + 17,
        )?;
        ensure!(
            expected == bits(strided.values())?,
            "strided extraction differs"
        );
        references.push(expected);
        pairs.push(Arc::new((src, dst)));
    }
    std::fs::write(
        output.join("IDENTITY.json"),
        serde_json::to_vec_pretty(&json!({
        "workload_sha256":digest(&bytes),"binary_sha256":digest(&std::fs::read(std::env::current_exe()?)?),
        "admission_sha256":w.admission_sha256,"size":w.size,"threads":w.threads,
        "raw_ids":w.raw_ids,"layout_width":1853,"pixel_sha256":w.pixel_sha256,
        "mode":args[1],"preflight_strided_exact":true,"reference_feature_bits":references,
        "dispatch_ceiling":dispatch(),"architecture":std::env::consts::ARCH,
        "scope":"native RGB8 extraction; excludes decode and scoring"}))?,
    )?;
    let mut flags = Vec::new();
    let mut batches = 0;
    let mut unreliable = false;
    if args[1] == "bench" {
        while flags.iter().filter(|v| **v == Some(true)).count() < 32
            && flags.len() < 64
            && !unreliable
        {
            let needed =
                (32 - flags.iter().filter(|v| **v == Some(true)).count()).min(64 - flags.len());
            let result = zenbench::run(|suite| {
                suite.compare("shaping_native_extraction", |group| {
                    group
                        .config()
                        .min_rounds(needed)
                        .max_rounds(needed)
                        .auto_rounds(false)
                        .warmup_time(if batches == 0 {
                            Duration::from_millis(100)
                        } else {
                            Duration::ZERO
                        })
                        .max_wall_time(Duration::from_secs(1800));
                    for (i, pair) in pairs.iter().enumerate() {
                        let pair = Arc::clone(pair);
                        let req = Arc::clone(&req);
                        let size = w.size;
                        group.bench(format!("pair-{i}"), move |b| {
                            b.iter(|| {
                                zenbench::black_box(
                                    extract(&req, &pair.0, &pair.1, size, size * 3)
                                        .expect("qualified extraction"),
                                )
                            });
                        });
                    }
                });
            });
            result.save(output.join(format!("zenbench-{batches:03}.json")))?;
            ensure!(result.comparisons.len() == 1, "comparison coverage differs");
            let c = &result.comparisons[0];
            let names: BTreeSet<_> = c.benchmarks.iter().map(|b| b.name.as_str()).collect();
            ensure!(
                c.benchmarks.len() == 3 && names == BTreeSet::from(["pair-0", "pair-1", "pair-2"]),
                "pair coverage differs"
            );
            for sample in &c.samples {
                let order: BTreeSet<_> = sample.execution_order.iter().copied().collect();
                ensure!(
                    sample.iterations > 0
                        && sample.elapsed_ns.len() == 3
                        && sample.compensated_ns.len() == 3
                        && sample.execution_order.len() == 3
                        && order.iter().copied().eq(0..3),
                    "incomplete paired round"
                );
            }
            ensure!(c.samples.len() <= needed, "unexpected round count");
            flags.extend(c.samples.iter().map(|s| s.gate_clean));
            unreliable |= result.unreliable;
            batches += 1;
            if c.samples.len() < needed {
                break;
            }
        }
    }
    for (pair, expected) in pairs.iter().zip(&references) {
        ensure!(
            bits(extract(&req, &pair.0, &pair.1, w.size, w.size * 3)?.values())? == *expected,
            "postflight extraction differs"
        );
    }
    ensure!(
        digest(&std::fs::read(&args[2])?) == digest(&bytes),
        "workload changed"
    );
    let retained: Vec<_> = flags
        .iter()
        .enumerate()
        .filter_map(|(i, v)| (*v == Some(true)).then_some(i))
        .take(32)
        .collect();
    let complete = args[1] == "qualify" || (retained.len() == 32 && !unreliable);
    std::fs::write(
        output.join("RESULT.json"),
        serde_json::to_vec_pretty(&json!({"complete":complete,
        "timing_measured":args[1]=="bench","postflight_exact":true,"gate_clean":flags,
        "retained_round_indices":retained,"unreliable":unreliable,"batches":batches}))?,
    )?;
    ensure!(
        complete,
        "incomplete timing retained; retry in a new directory"
    );
    Ok(())
}
