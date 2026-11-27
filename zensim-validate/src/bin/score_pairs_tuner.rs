//! score_pairs_tuner — score (ref, dist) image pairs with named zensim
//! profiles and/or runtime ZNPR bakes, through the PUBLIC surface API.
//!
//! Reads a TSV of pairs and emits a parquet: every input column is carried
//! through verbatim, plus `ref_basename`, `ref_width`/`ref_height`, and one
//! `f64` column per requested model.
//!
//! The TSV header MUST contain `ref_path` and `dist_path`. Every other column
//! is a passthrough key written as Utf8, in input order — except a column
//! literally named `q`, which stays `Int64` so the original per-codec picker
//! data-prep shape (`image_path codec q knob_tuple_json ref_path dist_path`
//! → `… ref_basename achieved_zensim_tuner`) is unchanged.
//!
//! Models (repeatable, evaluated in the order given — profiles then bakes):
//!
//! * `--profile <name>` — a named `ZensimProfile` served through
//!   `Zensim::compute`. Accepted: `v0_2`, `a`, `b` / `codec-target`, `c`,
//!   `d`, `bhdr`, `chdr`, `tuner`, `latest`.
//! * `--ensemble <label>=<bake.bin>[,<bake.bin>…]` — one or more ZNPR v3
//!   bakes served through `BakeScorer::ensemble` at equal weights (a single
//!   path is exactly `BakeScorer::new`). Emits column `score_<label>`.
//!
//! With no `--profile`/`--ensemble` the legacy behaviour is kept: one
//! `PreviewV0_5Tuner` column named `achieved_zensim_tuner`.
//!
//! Decoding is zen-only: PNG through `zenpng` (the crate's own dependency —
//! no third-party image decoder is reachable from this binary). 16-bit PNG
//! samples narrow with the same rounded `v*255/65535` convention the phase-2
//! example tooling uses; alpha is dropped. Any pair whose two images disagree
//! on dimensions is `NaN`, counted, and reported.
//!
//! Usage:
//!   score_pairs_tuner --pairs <pairs.tsv> --output <out.parquet>
//!                     [--profile b --profile d]
//!                     [--ensemble r915y60=a.bin,b.bin,c.bin]
//!                     [--threads N]

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use clap::Parser;
use enough::Unstoppable;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;
use zenpredict::Model;

use zensim::{BakeScorer, RgbSlice, Zensim, ZensimProfile};

#[derive(Parser, Debug)]
#[command(name = "score_pairs_tuner")]
struct Args {
    /// TSV of pairs; header must include `ref_path` and `dist_path`.
    #[arg(long)]
    pairs: PathBuf,

    /// Output parquet path (one row per input pair).
    #[arg(long)]
    output: PathBuf,

    /// Named profile to score, repeatable. Emits `score_<name>`.
    #[arg(long)]
    profile: Vec<String>,

    /// `<label>=<bake.bin>[,<bake.bin>…]`, repeatable. Equal-weight
    /// `BakeScorer::ensemble`. Emits `score_<label>`.
    #[arg(long)]
    ensemble: Vec<String>,

    /// Number of rayon threads (default: auto).
    #[arg(long)]
    threads: Option<usize>,

    /// Cache reference images (saves I/O when many rows share a ref —
    /// always true for picker data prep where each ref maps to 19 q values).
    #[arg(long, default_value_t = true)]
    cache_refs: bool,
}

/// One requested model: either a named profile or an equal-weight bake set.
enum ModelSpec {
    Profile { column: String, zensim: Zensim },
    Bakes { column: String, models: Vec<Model> },
}

impl ModelSpec {
    fn column(&self) -> &str {
        match self {
            Self::Profile { column, .. } | Self::Bakes { column, .. } => column,
        }
    }
}

#[allow(deprecated)] // `a` stays selectable for rescoring against the deprecated v47/A bake
fn parse_profile(s: &str) -> Result<ZensimProfile> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "v0_2" | "v02" | "preview-v0.2" | "previewv0_2" => ZensimProfile::PreviewV0_2,
        "a" | "zensim-a" => ZensimProfile::A,
        "b" | "zensim-b" | "codec-target" | "codec_target" => ZensimProfile::codec_target(),
        "bhdr" | "b-hdr" | "zensim-b-hdr" => ZensimProfile::BHdr,
        "c" | "zensim-c" => ZensimProfile::C,
        "chdr" | "c-hdr" | "zensim-c-hdr" => ZensimProfile::CHdr,
        "d" | "zensim-d" => ZensimProfile::D,
        "latest" | "latest-preview" => ZensimProfile::latest_preview(),
        "tuner" | "v0_5_tuner" => zensim_experimental::preview_v0_5_tuner(),
        other => return Err(anyhow!("unknown --profile '{other}'")),
    })
}

/// Decode a PNG or JPEG to packed RGB8 + dimensions, zen decoders only.
/// 16-bit samples narrow as `(v*255 + 32767)/65535`; alpha is dropped.
fn load_rgb8(p: &Path) -> Result<(Vec<u8>, u32, u32)> {
    let bytes = std::fs::read(p).with_context(|| format!("reading {p:?}"))?;
    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "png" => decode_png_rgb8(&bytes).with_context(|| format!("zenpng decode {p:?}")),
        other => Err(anyhow!("unsupported extension {other:?} for {p:?}")),
    }
}

fn decode_png_rgb8(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    use zenpixels::SampleType;
    let cfg = zenpng::PngDecodeConfig::default();
    let out = zenpng::decode(bytes, &cfg, &Unstoppable).map_err(|e| anyhow!("{e:?}"))?;
    let (w, h) = (out.info.width as usize, out.info.height as usize);
    let layout = out.pixels.format().layout;
    let channels = layout.channels();
    let slice = out.pixels.as_view();
    let samples_per_row = w * channels;
    let mut samples: Vec<u8> = Vec::with_capacity(h * samples_per_row);
    match layout.sample {
        SampleType::U8 => {
            for y in 0..h as u32 {
                samples.extend_from_slice(&slice.row(y)[..samples_per_row]);
            }
        }
        SampleType::U16 => {
            for y in 0..h as u32 {
                for pair in slice.row(y).as_chunks::<2>().0.iter().take(samples_per_row) {
                    let v = u16::from_ne_bytes([pair[0], pair[1]]) as u32;
                    samples.push(((v * 255 + 32767) / 65535) as u8);
                }
            }
        }
        other => return Err(anyhow!("unsupported PNG channel type {other:?}")),
    }
    if channels == 3 {
        return Ok((samples, w as u32, h as u32));
    }
    let mut rgb = Vec::with_capacity(w * h * 3);
    match channels {
        4 => {
            for px in samples.as_chunks::<4>().0 {
                rgb.extend_from_slice(&px[..3]);
            }
        }
        2 => {
            for px in samples.as_chunks::<2>().0 {
                rgb.extend_from_slice(&[px[0], px[0], px[0]]);
            }
        }
        1 => {
            for &g in &samples {
                rgb.extend_from_slice(&[g, g, g]);
            }
        }
        other => return Err(anyhow!("unsupported PNG channel count {other}")),
    }
    Ok((rgb, w as u32, h as u32))
}

#[derive(Clone, Debug)]
struct PairRow {
    /// Passthrough key columns, in input-header order (minus ref/dist paths).
    keys: Vec<String>,
    ref_path: String,
    dist_path: String,
}

fn ref_basename(ref_path: &str) -> String {
    Path::new(ref_path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| ref_path.to_string())
}

/// Score one already-decoded pair with every requested model.
fn score_one(specs: &[ModelSpec], rw: u32, rh: u32, rpx: &[u8], dpx: &[u8]) -> Vec<f64> {
    let (w, h) = (rw as usize, rh as usize);
    let src = rpx.as_chunks::<3>().0;
    let dst = dpx.as_chunks::<3>().0;
    let rs = RgbSlice::new(src, w, h);
    let ds = RgbSlice::new(dst, w, h);
    specs
        .iter()
        .map(|m| match m {
            ModelSpec::Profile { zensim, .. } => zensim
                .compute(&rs, &ds)
                .map(|r| r.score())
                .unwrap_or(f64::NAN),
            ModelSpec::Bakes { models, .. } => BakeScorer::ensemble(models, None)
                .and_then(|mut s| s.compute(&rs, &ds, None))
                .map(|r| r.score())
                .unwrap_or(f64::NAN),
        })
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(t) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .map_err(|e| anyhow!("rayon init: {e}"))?;
    }

    // --- resolve the model roster
    let legacy = args.profile.is_empty() && args.ensemble.is_empty();
    let mut specs: Vec<ModelSpec> = Vec::new();
    if legacy {
        specs.push(ModelSpec::Profile {
            column: "achieved_zensim_tuner".to_string(),
            zensim: Zensim::new(zensim_experimental::preview_v0_5_tuner()).with_parallel(false),
        });
    }
    for name in &args.profile {
        specs.push(ModelSpec::Profile {
            column: format!("score_{}", name.replace(['.', '-'], "_")),
            zensim: Zensim::new(parse_profile(name)?).with_parallel(false),
        });
    }
    for spec in &args.ensemble {
        let (label, paths) = spec
            .split_once('=')
            .ok_or_else(|| anyhow!("--ensemble wants <label>=<path>[,<path>…], got {spec:?}"))?;
        let models = paths
            .split(',')
            .map(|p| {
                let bytes = std::fs::read(p).with_context(|| format!("reading bake {p:?}"))?;
                Model::from_bytes(&bytes).map_err(|e| anyhow!("parsing bake {p:?}: {e:?}"))
            })
            .collect::<Result<Vec<_>>>()?;
        // Fail at startup, not per row, if the roster is unservable.
        BakeScorer::ensemble(&models, None).map_err(|e| anyhow!("ensemble {label:?}: {e}"))?;
        specs.push(ModelSpec::Bakes {
            column: format!("score_{}", label.replace(['.', '-'], "_")),
            models,
        });
    }

    eprintln!("score_pairs_tuner");
    eprintln!("  pairs:  {:?}", args.pairs);
    eprintln!("  output: {:?}", args.output);
    for m in &specs {
        eprintln!("  model:  {}", m.column());
    }

    let t_start = Instant::now();

    // --- read pairs TSV
    let file = File::open(&args.pairs).with_context(|| format!("opening {:?}", args.pairs))?;
    let mut lines = BufReader::new(file).lines();
    let header = lines.next().ok_or_else(|| anyhow!("empty pairs TSV"))??;
    let cols: Vec<String> = header.split('\t').map(str::to_string).collect();
    let pos = |name: &str| -> Result<usize> {
        cols.iter()
            .position(|c| c == name)
            .ok_or_else(|| anyhow!("missing '{}' column in header (got {:?})", name, cols))
    };
    let ref_idx = pos("ref_path")?;
    let dist_idx = pos("dist_path")?;
    let key_idx: Vec<usize> = (0..cols.len())
        .filter(|i| *i != ref_idx && *i != dist_idx)
        .collect();
    let key_names: Vec<String> = key_idx.iter().map(|i| cols[*i].clone()).collect();

    let mut pairs: Vec<PairRow> = Vec::new();
    for (i, ln) in lines.enumerate() {
        let ln = ln?;
        let parts: Vec<&str> = ln.split('\t').collect();
        if parts.len() < cols.len() {
            return Err(anyhow!("malformed line {}: {:?}", i + 2, ln));
        }
        pairs.push(PairRow {
            keys: key_idx.iter().map(|k| parts[*k].to_string()).collect(),
            ref_path: parts[ref_idx].to_string(),
            dist_path: parts[dist_idx].to_string(),
        });
    }
    let total = pairs.len();
    eprintln!(
        "  read {} pairs ({} key columns) in {:.2}s",
        total,
        key_names.len(),
        t_start.elapsed().as_secs_f64()
    );

    // --- dedupe refs (essential — each ref is shared by a whole ladder)
    let mut seen: HashMap<String, u32> = HashMap::new();
    let mut unique_refs: Vec<String> = Vec::new();
    let mut row_ref_idx: Vec<u32> = Vec::with_capacity(total);
    for p in &pairs {
        if let Some(&idx) = seen.get(&p.ref_path) {
            row_ref_idx.push(idx);
        } else {
            let idx = unique_refs.len() as u32;
            seen.insert(p.ref_path.clone(), idx);
            unique_refs.push(p.ref_path.clone());
            row_ref_idx.push(idx);
        }
    }
    eprintln!(
        "  unique refs: {} (deduplicated from {} rows)",
        unique_refs.len(),
        total
    );

    // --- load each unique ref into RAM (parallel)
    let t_ref_load = Instant::now();
    let n_refs = unique_refs.len();
    let progress = AtomicUsize::new(0);
    let ref_cache: Vec<Option<(Vec<u8>, u32, u32)>> = unique_refs
        .par_iter()
        .map(|p| {
            let r = match load_rgb8(Path::new(p)) {
                Ok(v) => Some(v),
                Err(e) => {
                    eprintln!("  WARN: ref {p}: {e:#}");
                    None
                }
            };
            let n = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_multiple_of(100) || n == n_refs {
                eprintln!("  ref-load: {}/{}", n, n_refs);
            }
            r
        })
        .collect();
    eprintln!(
        "  ref-load: done in {:.2}s",
        t_ref_load.elapsed().as_secs_f64()
    );

    // --- score each pair with every model
    let t_score = Instant::now();
    let n_models = specs.len();
    let progress = AtomicUsize::new(0);
    let nan_row = vec![f64::NAN; n_models];
    let scored: Vec<(Vec<f64>, i64, i64)> = (0..total)
        .into_par_iter()
        .map(|i| {
            let row = &pairs[i];
            let ref_opt = ref_cache[row_ref_idx[i] as usize].as_ref();
            let dist_opt = match load_rgb8(Path::new(&row.dist_path)) {
                Ok(v) => Some(v),
                Err(e) => {
                    eprintln!("  WARN: dist {}: {e:#}", row.dist_path);
                    None
                }
            };
            let out = match (ref_opt, dist_opt) {
                (Some((rpx, rw, rh)), Some((dpx, dw, dh))) => {
                    if *rw != dw || *rh != dh || rpx.len() != dpx.len() {
                        eprintln!(
                            "  WARN: dim mismatch on row {} (ref={}x{} dist={}x{}) — skipping",
                            i, rw, rh, dw, dh
                        );
                        (nan_row.clone(), *rw as i64, *rh as i64)
                    } else {
                        (
                            score_one(&specs, *rw, *rh, rpx, &dpx),
                            *rw as i64,
                            *rh as i64,
                        )
                    }
                }
                _ => {
                    eprintln!(
                        "  WARN: load failure for row {} (ref={:?} dist={:?})",
                        i, row.ref_path, row.dist_path
                    );
                    (nan_row.clone(), 0, 0)
                }
            };
            let n = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_multiple_of(500) || n == total {
                let elapsed = t_score.elapsed().as_secs_f64();
                let rate = n as f64 / elapsed;
                let eta = (total - n) as f64 / rate;
                eprintln!(
                    "  score: {}/{} ({:.1}%) {:.1}s rate={:.1}/s eta={:.0}s",
                    n,
                    total,
                    100.0 * n as f64 / total as f64,
                    elapsed,
                    rate,
                    eta
                );
            }
            out
        })
        .collect();
    eprintln!(
        "  score: {} rows in {:.2}s",
        total,
        t_score.elapsed().as_secs_f64()
    );

    // --- write parquet: passthrough keys, ref_basename, geometry, scores
    let mut fields: Vec<Arc<Field>> = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();
    for (k, name) in key_names.iter().enumerate() {
        if name == "q" {
            let vals: Result<Vec<i64>> = pairs
                .iter()
                .map(|p| {
                    p.keys[k]
                        .parse::<i64>()
                        .with_context(|| format!("parsing q {:?}", p.keys[k]))
                })
                .collect();
            fields.push(Arc::new(Field::new("q", DataType::Int64, false)));
            arrays.push(Arc::new(Int64Array::from(vals?)));
        } else {
            fields.push(Arc::new(Field::new(name, DataType::Utf8, false)));
            arrays.push(Arc::new(StringArray::from(
                pairs.iter().map(|p| p.keys[k].clone()).collect::<Vec<_>>(),
            )));
        }
    }
    fields.push(Arc::new(Field::new("ref_basename", DataType::Utf8, false)));
    arrays.push(Arc::new(StringArray::from(
        pairs
            .iter()
            .map(|p| ref_basename(&p.ref_path))
            .collect::<Vec<_>>(),
    )));
    fields.push(Arc::new(Field::new("ref_width", DataType::Int64, false)));
    arrays.push(Arc::new(Int64Array::from(
        scored.iter().map(|(_, w, _)| *w).collect::<Vec<_>>(),
    )));
    fields.push(Arc::new(Field::new("ref_height", DataType::Int64, false)));
    arrays.push(Arc::new(Int64Array::from(
        scored.iter().map(|(_, _, h)| *h).collect::<Vec<_>>(),
    )));
    for (m, spec) in specs.iter().enumerate() {
        fields.push(Arc::new(Field::new(spec.column(), DataType::Float64, true)));
        arrays.push(Arc::new(Float64Array::from(
            scored.iter().map(|(s, ..)| s[m]).collect::<Vec<_>>(),
        )));
    }
    let out_schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(out_schema.clone(), arrays)?;

    let out_file =
        File::create(&args.output).with_context(|| format!("creating {:?}", args.output))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let mut writer = ArrowWriter::try_new(out_file, out_schema.clone(), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    for (m, spec) in specs.iter().enumerate() {
        let vals: Vec<f64> = scored.iter().map(|(s, ..)| s[m]).collect();
        let n_nan = vals.iter().filter(|v| v.is_nan()).count();
        let finite: Vec<f64> = vals.iter().copied().filter(|v| v.is_finite()).collect();
        let (mn, mx, mean) = if finite.is_empty() {
            (f64::NAN, f64::NAN, f64::NAN)
        } else {
            (
                finite.iter().copied().fold(f64::INFINITY, f64::min),
                finite.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                finite.iter().sum::<f64>() / finite.len() as f64,
            )
        };
        eprintln!(
            "  {}: min={:.4} mean={:.4} max={:.4} n_nan={}",
            spec.column(),
            mn,
            mean,
            mx,
            n_nan
        );
    }
    eprintln!("  wrote {} rows → {:?}", total, args.output);

    Ok(())
}
