//! Corpus feature extractor for matched Rev7 + Rev8 Alg S4 MixF3 evaluation.
//!
//! Uses imazen's canonical magic-byte detection (`zencodec::ImageFormatRegistry::common`)
//! with `zenjpeg` for JPEG (including mislabeled JPEG bytes inside `.bmp` files) and
//! `zenpng` / `image` fallback for PNG/BMP/WebP/AVIF/JXL, matching the Rev7 bank pipeline.

use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::Deserialize;
use zencodec::{ImageFormat, ImageFormatRegistry};
use zenpixels::{ImageSpec, PixelBuf, SampleLayout, SampleType};
use zenpixels_convert::{ConvertPolicy, RowPlan};
use zensim::rev8::{IntensityTarget, RgbRows, Variant, extract_fx2};
use zensim::{RgbSlice, feature_set_id::FeatureSetId, research};

const REV7_FSID: &str = "basic+peaks+v2@w1825/rev7_rowlanes#36c3f3af";

#[derive(Deserialize)]
struct Fx2DeclJson {
    direct: Vec<usize>,
    products: Vec<[usize; 2]>,
}

struct ManifestRow {
    pair_idx: usize,
    ref_path: PathBuf,
    dist_path: PathBuf,
    need_rev7: bool,
}

struct DecodedImage {
    rgb: Vec<[u8; 3]>,
    w: usize,
    h: usize,
}

#[cfg(target_arch = "x86_64")]
fn apply_tier_cap() {
    let Some(value) = std::env::var_os("ZENSIM_MAX_TIER") else {
        return;
    };
    match value.to_str() {
        Some("v3") => {
            use archmage::{X64V4Token, X64V4xToken};
            X64V4xToken::dangerously_disable_token_process_wide(true)
                .unwrap_or_else(|e| panic!("cannot cap v4x: {e}"));
            X64V4Token::dangerously_disable_token_process_wide(true)
                .unwrap_or_else(|e| panic!("cannot cap v4: {e}"));
            eprintln!("tier cap: AVX-512 tokens disabled (ZENSIM_MAX_TIER=v3)");
        }
        other => panic!("ZENSIM_MAX_TIER={other:?}: only \"v3\" is supported"),
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn apply_tier_cap() {}

fn check_safe_path(path: &Path) -> Result<(), String> {
    let candidates = [
        path.to_path_buf(),
        path.canonicalize().unwrap_or_else(|_| path.to_path_buf()),
    ];
    for p in candidates {
        let lower_full = p.to_string_lossy().to_ascii_lowercase();
        for bad in ["cid22_validation_set.csv", "public_test", "pairs_test"] {
            if lower_full.contains(bad) {
                return Err(format!("refusing protected path {path:?} (matched {bad})"));
            }
        }
        for comp in p.components() {
            let s = comp.as_os_str().to_string_lossy().to_ascii_lowercase();
            if s.contains("_sealed")
                || s.contains("holdout")
                || s.contains("terminal")
                || s.starts_with("labels__")
                || s.contains("cid22val")
            {
                return Err(format!(
                    "refusing protected component {s:?} in path {path:?}"
                ));
            }
        }
    }
    Ok(())
}

fn pixelbuffer_to_rgb8(pb: &PixelBuf) -> Result<(Vec<[u8; 3]>, usize, usize), String> {
    let spec = pb.spec();
    let w = pb.width() as usize;
    let h = pb.height() as usize;
    let slice = pb.as_view();
    let stride = slice.stride();
    let data = slice.as_strided_bytes();
    let mut flat = Vec::new();
    if spec.layout == SampleLayout::RGB8 {
        flat.reserve(w * h * 3);
        let bpr = w * 3;
        for row in 0..h {
            let s = row * stride;
            flat.extend_from_slice(&data[s..s + bpr]);
        }
    } else if spec.layout == SampleLayout::RGBA8 {
        flat.reserve(w * h * 3);
        for row in 0..h {
            let s = row * stride;
            for px in data[s..s + w * 4].as_chunks::<4>().0 {
                flat.extend_from_slice(&px[..3]);
            }
        }
    } else {
        let dst_stride = w * 3;
        flat.resize(dst_stride * h, 0);
        let plan = RowPlan::build(
            spec,
            &ImageSpec::RGB8_SRGB,
            &ConvertPolicy::permissive(),
            None,
        )
        .map_err(|e| format!("cannot plan {spec:?} -> RGB8_SRGB: {e}"))?;
        for y in 0..h {
            plan.convert_row(
                slice.row(y as u32),
                &mut flat[y * dst_stride..(y + 1) * dst_stride],
                w as u32,
                y as u32,
            )
            .map_err(|e| format!("convert row {y}: {e}"))?;
        }
    }
    let rgb = flat.as_chunks::<3>().0.to_vec();
    Ok((rgb, w, h))
}

fn decode_png_rgb8(bytes: &[u8], path: &Path) -> Result<(Vec<[u8; 3]>, usize, usize), String> {
    let out = zenpng::decode(
        bytes,
        &zenpng::PngDecodeConfig::default(),
        &enough::Unstoppable,
    )
    .map_err(|e| format!("zenpng {path:?}: {e}"))?;
    let layout = out.pixels.spec().layout;
    if layout.sample == SampleType::U16 {
        // Match zen_io's rounded (v * 255 + 32767) / 65535 narrowing on 16-bit PNGs.
        let w = out.info.width as usize;
        let h = out.info.height as usize;
        let ch = layout.channels();
        let slice = out.pixels.as_view();
        let spr = w * ch;
        let mut samples = Vec::with_capacity(h * spr);
        for y in 0..h as u32 {
            let row = slice.row(y);
            for &pair in row.as_chunks::<2>().0.iter().take(spr) {
                let v = u16::from_ne_bytes(pair) as u32;
                samples.push(((v * 255 + 32767) / 65535) as u8);
            }
        }
        let rgb = match ch {
            3 => samples.as_chunks::<3>().0.to_vec(),
            4 => samples
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| [c[0], c[1], c[2]])
                .collect(),
            1 => samples.iter().map(|&g| [g, g, g]).collect(),
            2 => samples
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| [c[0], c[0], c[0]])
                .collect(),
            other => return Err(format!("unsupported U16 PNG channels {other} in {path:?}")),
        };
        return Ok((rgb, w, h));
    }
    pixelbuffer_to_rgb8(&out.pixels)
}

fn decode_rgb8(path: &Path) -> Result<(Vec<[u8; 3]>, usize, usize), String> {
    check_safe_path(path)?;
    let bytes = std::fs::read(path).map_err(|e| format!("read {path:?}: {e}"))?;
    let fmt = ImageFormatRegistry::common()
        .detect(&bytes)
        .ok_or_else(|| format!("undetectable image format in {path:?}"))?;
    match fmt {
        ImageFormat::Jpeg => {
            let result = zenjpeg::decoder::Decoder::new()
                .decode(&bytes, enough::Unstoppable)
                .map_err(|e| format!("zenjpeg decode {path:?}: {e}"))?;
            let (dw, dh) = result.dimensions();
            let (w, h) = (dw as usize, dh as usize);
            if let Some(px) = result.pixels_u8() {
                if px.len() == w * h * 3 {
                    return Ok((px.as_chunks::<3>().0.to_vec(), w, h));
                } else if px.len() == w * h {
                    let rgb = px.iter().map(|&g| [g, g, g]).collect();
                    return Ok((rgb, w, h));
                }
            }
            let img = image::ImageReader::new(std::io::Cursor::new(&bytes))
                .with_guessed_format()
                .map_err(|e| format!("guess format {path:?}: {e}"))?
                .decode()
                .map_err(|e| format!("image decode {path:?}: {e}"))?
                .into_rgb8();
            let (iw, ih) = img.dimensions();
            let rgb = img.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
            Ok((rgb, iw as usize, ih as usize))
        }
        ImageFormat::Png => decode_png_rgb8(&bytes, path),
        _ => {
            let img = image::ImageReader::new(std::io::Cursor::new(&bytes))
                .with_guessed_format()
                .map_err(|e| format!("guess format {path:?}: {e}"))?
                .decode()
                .map_err(|e| format!("image decode {path:?}: {e}"))?
                .into_rgb8();
            let (w, h) = img.dimensions();
            let rgb = img.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
            Ok((rgb, w as usize, h as usize))
        }
    }
}

fn write_f32_bin(path: &Path, data: &[f32]) -> Result<(), String> {
    let f = File::create(path).map_err(|e| format!("create {path:?}: {e}"))?;
    let mut w = BufWriter::new(f);
    for &v in data {
        w.write_all(&v.to_le_bytes())
            .map_err(|e| format!("write {path:?}: {e}"))?;
    }
    w.flush().map_err(|e| format!("flush {path:?}: {e}"))
}

fn worker_count() -> usize {
    let env_threads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    let n = env_threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(8)
    });
    n.clamp(1, 16)
}

fn parallel_map<T: Sync, R: Send, F: Fn(&T) -> Result<R, String> + Sync>(
    items: &[T],
    threads: usize,
    f: F,
) -> Result<Vec<R>, String> {
    if items.is_empty() {
        return Ok(Vec::new());
    }
    let next_idx = std::sync::atomic::AtomicUsize::new(0);
    let slots: Vec<std::sync::Mutex<Option<Result<R, String>>>> = (0..items.len())
        .map(|_| std::sync::Mutex::new(None))
        .collect();
    std::thread::scope(|scope| {
        for _ in 0..threads.min(items.len()) {
            scope.spawn(|| {
                loop {
                    let i = next_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= items.len() {
                        break;
                    }
                    let res = f(&items[i]);
                    *slots[i].lock().expect("slot lock") = Some(res);
                }
            });
        }
    });
    let mut out = Vec::with_capacity(items.len());
    for slot in slots {
        out.push(slot.into_inner().expect("slot").expect("populated")?);
    }
    Ok(out)
}

fn run_extract(manifest_tsv: &Path, fx2_decl_json: &Path, out_dir: &Path) -> Result<(), String> {
    check_safe_path(manifest_tsv)?;
    check_safe_path(fx2_decl_json)?;
    check_safe_path(out_dir)?;
    let decl_bytes =
        std::fs::read(fx2_decl_json).map_err(|e| format!("read {fx2_decl_json:?}: {e}"))?;
    let decl: Fx2DeclJson =
        serde_json::from_slice(&decl_bytes).map_err(|e| format!("parse decl JSON: {e}"))?;
    if decl.direct.len() != 492 || decl.products.len() != 492 {
        return Err(format!(
            "expected 492 direct + 492 products, got {} + {}",
            decl.direct.len(),
            decl.products.len()
        ));
    }

    let file = File::open(manifest_tsv).map_err(|e| format!("open {manifest_tsv:?}: {e}"))?;
    let mut rows = Vec::new();
    for (line_no, line_res) in BufReader::new(file).lines().enumerate() {
        let line = line_res.map_err(|e| format!("read {manifest_tsv:?}: {e}"))?;
        if line_no == 0 {
            if line.trim() != "pair_idx\tref_path\tdist_path\tneed_rev7" {
                return Err(format!("unexpected manifest header: {line:?}"));
            }
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() != 4 {
            return Err(format!(
                "manifest line {} has {} cols, expected 4",
                line_no + 1,
                cols.len()
            ));
        }
        let pair_idx: usize = cols[0]
            .parse()
            .map_err(|e| format!("bad pair_idx on line {}: {e}", line_no + 1))?;
        if pair_idx != rows.len() {
            return Err(format!(
                "manifest pair_idx must be dense 0..N, got {pair_idx} at row {}",
                rows.len()
            ));
        }
        let ref_path = PathBuf::from(cols[1]);
        let dist_path = PathBuf::from(cols[2]);
        check_safe_path(&ref_path)?;
        check_safe_path(&dist_path)?;
        let need_rev7 = match cols[3] {
            "0" => false,
            "1" => true,
            other => return Err(format!("bad need_rev7 {other:?} on line {}", line_no + 1)),
        };
        rows.push(ManifestRow {
            pair_idx,
            ref_path,
            dist_path,
            need_rev7,
        });
    }

    let any_rev7 = rows.iter().any(|r| r.need_rev7);
    if any_rev7 && std::env::var("ZENSIM_FORMULA_REV").as_deref() != Ok("7") {
        return Err("Rev7 extraction requested (need_rev7=1) but ZENSIM_FORMULA_REV!=7".into());
    }
    let rev7_req = if any_rev7 {
        let fsid = FeatureSetId::parse(REV7_FSID).ok_or("invalid REV7 FeatureSetId")?;
        let req = research::Request::for_set(&fsid)
            .map_err(|e| format!("research::Request::for_set: {e}"))?
            .with_parallel(false);
        req.validate()
            .map_err(|e| format!("Rev7 research validate failed: {e}"))?;
        Some(req)
    } else {
        None
    };

    let threads = worker_count();
    eprintln!(
        "[extract] {} pairs across {threads} threads (ref-sorted streaming batches of 48)",
        rows.len()
    );

    struct RowOut {
        rev8_255: [f32; 480],
        rev8_500: [f32; 480],
        rev7_984: Option<(usize, Vec<f32>, Vec<f32>)>,
    }

    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by(|&i, &j| rows[i].ref_path.cmp(&rows[j].ref_path).then(i.cmp(&j)));
    let mut per_row_opt: Vec<Option<RowOut>> = (0..rows.len()).map(|_| None).collect();

    for (batch_idx, chunk) in order.chunks(48).enumerate() {
        let mut unique_set = HashSet::new();
        let mut unique_paths = Vec::new();
        for &idx in chunk {
            let r = &rows[idx];
            if unique_set.insert(r.ref_path.clone()) {
                unique_paths.push(r.ref_path.clone());
            }
            if unique_set.insert(r.dist_path.clone()) {
                unique_paths.push(r.dist_path.clone());
            }
        }
        let decoded_vec = parallel_map(&unique_paths, threads, |p| {
            let (rgb, w, h) = decode_rgb8(p)?;
            Ok((p.clone(), DecodedImage { rgb, w, h }))
        })?;
        let decoded_map: BTreeMap<PathBuf, DecodedImage> = decoded_vec.into_iter().collect();

        let batch_out = parallel_map(chunk, threads, |&idx| {
            let r = &rows[idx];
            let a = decoded_map.get(&r.ref_path).expect("decoded ref");
            let b = decoded_map.get(&r.dist_path).expect("decoded dist");
            if (a.w, a.h) != (b.w, b.h) {
                return Err(format!(
                    "pair {} dimension mismatch: {}x{} vs {}x{} ({:?} vs {:?})",
                    r.pair_idx, a.w, a.h, b.w, b.h, r.ref_path, r.dist_path
                ));
            }
            let va = RgbRows::new(&a.rgb, a.w, a.h, a.w).ok_or("invalid RgbRows a")?;
            let vb = RgbRows::new(&b.rgb, b.w, b.h, b.w).ok_or("invalid RgbRows b")?;
            let e255 = extract_fx2(va, vb, Variant::AlgS4, IntensityTarget::Nits255, false)
                .ok_or("extract_fx2 Nits255 failed")?;
            let e500 = extract_fx2(va, vb, Variant::AlgS4, IntensityTarget::Nits500, false)
                .ok_or("extract_fx2 Nits500 failed")?;
            let mut rev8_255 = [0.0f32; 480];
            let mut rev8_500 = [0.0f32; 480];
            let raw255 = e255.alg_s4_fx2_inputs();
            let raw500 = e500.alg_s4_fx2_inputs();
            for i in 0..480 {
                if !raw255[i].is_finite() || !raw500[i].is_finite() {
                    return Err(format!(
                        "nonfinite rev8 feature at pair {} col {i}",
                        r.pair_idx
                    ));
                }
                rev8_255[i] = raw255[i] as f32;
                rev8_500[i] = raw500[i] as f32;
            }
            let rev7_984 = if let (true, Some(req)) = (r.need_rev7, rev7_req.as_ref()) {
                let sa = RgbSlice::new(&a.rgb, a.w, a.h);
                let sb = RgbSlice::new(&b.rgb, b.w, b.h);
                let ext = research::extract(req, &sa, &sb)
                    .map_err(|e| format!("rev7 extract pair {}: {e}", r.pair_idx))?;
                let vals = ext.values();
                if vals.len() != 1825 {
                    return Err(format!(
                        "expected 1825 raw Rev7 features, got {} at pair {}",
                        vals.len(),
                        r.pair_idx
                    ));
                }
                let raw1825: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
                let mut compact = Vec::with_capacity(984);
                for &i in &decl.direct {
                    compact.push(vals[i] as f32);
                }
                for &[a_i, b_i] in &decl.products {
                    compact.push((vals[a_i] * vals[b_i]) as f32);
                }
                Some((r.pair_idx, compact, raw1825))
            } else {
                None
            };
            Ok((
                idx,
                RowOut {
                    rev8_255,
                    rev8_500,
                    rev7_984,
                },
            ))
        })?;
        for (idx, item) in batch_out {
            per_row_opt[idx] = Some(item);
        }
        if (batch_idx + 1) % 20 == 0 || (batch_idx + 1) * 48 >= rows.len() {
            eprintln!(
                "[extract] progress: {}/{} pairs",
                ((batch_idx + 1) * 48).min(rows.len()),
                rows.len()
            );
        }
    }

    std::fs::create_dir_all(out_dir).map_err(|e| format!("mkdir {out_dir:?}: {e}"))?;
    let mut flat_255 = Vec::with_capacity(rows.len() * 480);
    let mut flat_500 = Vec::with_capacity(rows.len() * 480);
    let mut flat_rev7 = Vec::new();
    let mut flat_rev7_1825 = Vec::new();
    let mut rev7_idx = Vec::new();
    for item_opt in per_row_opt {
        let item = item_opt.expect("all rows extracted");
        flat_255.extend_from_slice(&item.rev8_255);
        flat_500.extend_from_slice(&item.rev8_500);
        if let Some((idx, v, r1825)) = item.rev7_984 {
            rev7_idx.push(idx);
            flat_rev7.extend_from_slice(&v);
            flat_rev7_1825.extend_from_slice(&r1825);
        }
    }
    write_f32_bin(&out_dir.join("rev8_nits255_f32.bin"), &flat_255)?;
    write_f32_bin(&out_dir.join("rev8_nits500_f32.bin"), &flat_500)?;
    write_f32_bin(&out_dir.join("rev7_extra_f32.bin"), &flat_rev7)?;
    write_f32_bin(&out_dir.join("rev7_raw1825_f32.bin"), &flat_rev7_1825)?;
    std::fs::write(
        out_dir.join("rev7_extra_idx.json"),
        serde_json::to_string(&rev7_idx).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("write rev7_extra_idx.json: {e}"))?;
    eprintln!(
        "[extract] wrote {} Rev8 rows + {} Rev7 extra rows to {}",
        rows.len(),
        rev7_idx.len(),
        out_dir.display()
    );
    Ok(())
}

fn main() {
    apply_tier_cap();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        eprintln!("Usage: rev8_pilot_extract <manifest.tsv> <fx2_decl.json> <out_dir>");
        std::process::exit(2);
    }
    if let Err(e) = run_extract(
        Path::new(&args[0]),
        Path::new(&args[1]),
        Path::new(&args[2]),
    ) {
        eprintln!("FATAL: {e}");
        std::process::exit(1);
    }
}
