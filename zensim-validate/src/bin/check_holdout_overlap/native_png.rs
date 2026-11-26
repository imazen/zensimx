//! Shared strict native audit; no legacy image decoder/resizer executes here.
use super::Args;
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::Path;
use zenpixels::PixelFormat;
use zenpixels_convert::{ConvertPolicy, convert_to_buf};

use super::dhash_bits;

fn fingerprint(path: &Path, max_pixels: Option<u64>) -> Result<Value> {
    let bytes = fs::read(path).with_context(|| path.display().to_string())?;
    let mut config = zenpng::PngDecodeConfig::default();
    if let Some(max_pixels) = max_pixels {
        ensure!(max_pixels > 0, "pixel resource budget must be positive");
        config = config.with_max_pixels(max_pixels);
    }
    let decoded = zenpng::decode(&bytes, &config, &enough::Unstoppable)
        .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let (width, height) = (decoded.info.width, decoded.info.height);
    ensure!(
        width >= 9 && height >= 8,
        "native audit requires at least 9x8"
    );
    let rgb = convert_to_buf(
        &decoded.pixels.as_view(),
        &PixelFormat::RGB8_SRGB,
        &ConvertPolicy::permissive(),
    )
    .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let mut luma = Vec::with_capacity(width as usize * height as usize);
    let rgb_view = rgb.as_view();
    for y in 0..height {
        for &[r, g, b] in rgb_view.row(y).as_chunks::<3>().0 {
            luma.push(
                ((2126 * u32::from(r) + 7152 * u32::from(g) + 722 * u32::from(b)) / 10000) as u8,
            );
        }
    }
    ensure!(
        luma.len() == width as usize * height as usize,
        "native packing"
    );
    let config = zenresize::ResizeConfig::builder(width, height, 9, 8)
        .filter(zenresize::Filter::Lanczos)
        .format(zenresize::PixelDescriptor::GRAY8_SRGB)
        .build();
    let small = zenresize::Resizer::new(&config).resize(&luma);
    let hash = dhash_bits::from_luma9x8(small.as_slice().try_into().context("native 9x8")?);
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect();
    Ok(json!({"path":path,"sha256":digest,
        "width":width,"height":height,"dhash":hash}))
}

pub(super) fn run(args: &Args) -> Result<()> {
    let sidecar = args.out_tsv.with_extension("hashes.json");
    ensure!(
        !args.out_tsv.exists() && !sidecar.exists(),
        "native outputs must be fresh"
    );
    ensure!(
        args.threshold <= 16,
        "strict threshold must be <=16 review screen"
    );
    let (Some(nt), Some(nh)) = (args.expected_training, args.expected_holdout) else {
        anyhow::bail!("native mode requires expected training and holdout counts");
    };
    ensure!(nt > 0 && nh > 0, "expected counts must be nonzero");
    let holdout_paths = super::walk_image_dir(&args.cid22_refs)?;
    ensure!(
        holdout_paths.len() == nh,
        "holdout file coverage: {} != {nh}",
        holdout_paths.len()
    );
    let mut reader = csv::Reader::from_path(&args.training_csv)?;
    let mut sources = BTreeSet::new();
    for row in reader.records() {
        let row = row?;
        let path = row.get(0).context("source column")?;
        ensure!(!path.is_empty(), "empty source path");
        sources.insert(path.to_string());
    }
    ensure!(
        sources.len() == nt,
        "training source coverage: {} != {nt}",
        sources.len()
    );
    let fingerprint_fn = |path: &Path| {
        if args.native_linear {
            super::native_linear::fingerprint(path, args.assume_untagged_srgb)
        } else {
            fingerprint(path, args.max_decode_pixels)
        }
    };
    let training = sources
        .par_iter()
        .map(|p| fingerprint_fn(Path::new(p)))
        .collect::<Result<Vec<_>>>()?;
    let holdout = holdout_paths
        .par_iter()
        .map(|p| fingerprint_fn(p))
        .collect::<Result<Vec<_>>>()?;
    let h = |v: &Value| v["dhash"].as_u64().expect("fingerprint hash");
    let path = |v: &Value| v["path"].as_str().expect("fingerprint path").to_owned();
    let mut tsv =
        String::from("training_source\tdhash\tnearest_cid22_ref\tnearest_dhash\thamming\n");
    let mut close = Vec::new();
    for t in &training {
        let nearest = holdout
            .iter()
            .min_by_key(|r| (h(t) ^ h(r)).count_ones())
            .context("empty holdout")?;
        use std::fmt::Write;
        writeln!(
            tsv,
            "{}\t{:016x}\t{}\t{:016x}\t{}",
            path(t),
            h(t),
            path(nearest),
            h(nearest),
            (h(t) ^ h(nearest)).count_ones()
        )?;
        for r in &holdout {
            let d = (h(t) ^ h(r)).count_ones();
            if d <= 16 {
                close.push(json!({"training":path(t),"holdout":path(r),"hamming":d,"strict_flag":d<=args.threshold}));
            }
        }
    }
    let strict = close.iter().filter(|r| r["strict_flag"] == true).count();
    let (schema, era) = if args.native_linear {
        (
            "native-linear-dhash-audit-v1",
            "bt709-peak-normalized-log255-luma-zenresize-lanczos-f32-v1",
        )
    } else {
        (
            "native-png-dhash-audit-v1",
            "bt709-encoded-luma-zenresize-lanczos-gray8-v1",
        )
    };
    let report = json!({"schema":schema,"hash_era":era,
        "threshold":args.threshold,"review_threshold":16,"training":training,"holdout":holdout,
        "close_pairs":close,"strict_flags":strict,"admission_approved":false});
    // Only complete results are written; create_new protects existing evidence.
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.out_tsv)?
        .write_all(tsv.as_bytes())?;
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&sidecar)?
        .write_all(&serde_json::to_vec_pretty(&report)?)?;
    eprintln!(
        "native audit complete: {nt} training, {nh} holdout; {strict} strict flags; contextual admission review required"
    );
    Ok(())
}
