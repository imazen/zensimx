// Legacy dataset extraction and diagnostic reports. Model training is owned by
// zensim_mlp_train; model evaluation is owned by bake_verdict through BakeScorer.
mod scale_invariance;

use calamine::{Reader, Xlsx};
use clap::Parser;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
// Canonical IQA stats, aliased onto this file's historical names (#41).
use zenstats::{pearson as pearson_correlation, spearman as spearman_correlation};

#[derive(Parser)]
#[command(
    name = "zensim-validate",
    about = "Validate zensim against human quality ratings"
)]
struct Args {
    /// Dataset directory (e.g., ./datasets/tid2013)
    /// Not required when --scale-invariance is set.
    #[arg(long)]
    dataset: Option<PathBuf>,

    /// Dataset type
    /// Not required when --scale-invariance is set.
    #[arg(long, value_enum)]
    format: Option<DatasetFormat>,

    /// Scale-invariance mode: ingest a pyramid CSV produced by
    /// `coefficient/examples/generate_scale_pyramid`, compute zensim per row,
    /// and emit per-(source × codec × quality × distortion) slope fits of
    /// `score = α + β · log2(pixel_count)` for zensim, ssim2, butteraugli, dssim.
    /// Bypasses dataset loading.
    #[arg(long)]
    scale_invariance: Option<PathBuf>,

    /// Output directory for scale-invariance reports (default: alongside the input CSV).
    #[arg(long)]
    scale_invariance_out: Option<PathBuf>,

    /// Additional datasets for diagnostic reports (format: type:path,type:path)
    #[arg(long)]
    also: Option<String>,

    /// Max images to process (0 = all)
    #[arg(long, default_value = "0")]
    max_images: usize,

    /// Output features CSV for external analysis
    #[arg(long)]
    features_csv: Option<PathBuf>,

    /// Box blur passes (1, 2, or 3, default: 1). 1 = rectangular, 2 = triangular, 3 ≈ Gaussian.
    #[arg(long, default_value = "1")]
    blur_passes: u8,

    /// Box blur radius at scale 0 (default: 5, giving 11-pixel kernel)
    #[arg(long, default_value = "5")]
    blur_radius: usize,

    /// Compute all features in the selected layout.
    /// Useful for exporting full feature vectors for offline analysis.
    #[arg(long, default_value = "false")]
    compute_all: bool,

    /// Extract features and save cache, then exit. No evaluation or training.
    /// Always extracts extended (300) features. Use with --dataset and --format.
    #[arg(long, default_value = "false")]
    extract_only: bool,

    /// Compute extended features (25 per channel instead of 13).
    /// Adds masked SSIM/edge/MSE + max/p95 percentile features.
    #[arg(long, default_value = "false")]
    extended_features: bool,

    /// Masking strength for extended features (default: 4.0).
    /// Only used when --extended-features is set.
    #[arg(long, default_value = "4.0")]
    extended_masking_strength: f32,

    /// Compute IW (information-content-weighted) features
    /// (Wang & Li 2011 IW-SSIM). 6 features per channel per scale —
    /// at 4 scales × 3 ch = 72 features. Texture-EMPHASISING
    /// counterpart to `--extended-features`. With this flag the
    /// emitted features.csv has 228 + 72 = 300 columns
    /// (basic + peaks + IW); with both `--extended-features` and
    /// `--iw-features` it has 372 columns
    /// (basic + peaks + masked + IW).
    #[arg(long, default_value = "false")]
    iw_features: bool,

    /// IW weighting strength: `iw_weight[i] = 1 + k * blur(|src - mu|)`.
    /// Only used when --iw-features is set. Default: 4.0
    /// (mirrors --extended-masking-strength).
    #[arg(long, default_value = "4.0")]
    iw_strength: f32,

    /// Downscale filter for pyramid construction: box, mitchell, lanczos
    #[arg(long, default_value = "box")]
    downscale_filter: String,

    /// Number of downscale levels (default: 4, max: 6)
    #[arg(long, default_value = "4")]
    num_scales: usize,

    /// Load custom weights from file (one weight per line).
    /// Evaluates these weights against the dataset(s) instead of the embedded weights.
    #[arg(long)]
    weights_file: Option<PathBuf>,

    /// Target metric for synthetic datasets (selects which column to use as ground truth)
    #[arg(long, value_enum)]
    target_metric: Option<TargetMetric>,

    /// Feature cache file path. Auto-derived from dataset path if omitted.
    #[arg(long)]
    feature_cache: Option<PathBuf>,

    /// Force recompute features, ignoring any existing cache.
    #[arg(long, default_value = "false")]
    recompute: bool,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum TargetMetric {
    /// GPU SSIMULACRA2 (ssimulacra2-cuda)
    GpuSsim2,
    /// GPU Butteraugli (butteraugli-cuda)
    GpuButteraugli,
    /// CPU SSIMULACRA2 (fast-ssim2)
    CpuSsim2,
    /// CPU Butteraugli max-norm (butteraugli crate `.score` field)
    CpuButteraugli,
    /// CPU Butteraugli 3-norm via libjxl-style averaged p-norm
    /// `((Σdᵖ/n)^(1/p) + (Σd^(2p)/n)^(1/(2p)) + (Σd^(4p)/n)^(1/(4p))) / 3` at p=3.
    /// Reads `butteraugli_3norm` column produced by zensim-bench's
    /// gen_butteraugli_3norm binary. Matches Cloudinary CID22 paper Table 4.
    CpuButteraugli3Norm,
    /// DSSIM (structural dissimilarity)
    Dssim,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum DatasetFormat {
    Tid2013,
    Kadid10k,
    Csiq,
    Pipal,
    Cid22,
    KonfigIqa,
    Synthetic,
}

/// A single reference-distorted pair with human score.
#[derive(Debug, Clone)]
struct ImagePair {
    reference: PathBuf,
    distorted: PathBuf,
    /// Human subjective score (higher = better quality, normalized to 0-1)
    human_score: f64,
}

/// A dataset with precomputed features and stable reference keys.
struct DatasetWithFeatures {
    name: String,
    human_scores: Vec<f64>,
    features: Vec<Vec<f64>>,
    ref_keys: Vec<String>,
}

struct CacheConfig {
    num_scales: u32,
    blur_passes: u8,
    blur_radius: u32,
    /// Feature-shape kind. Differentiates basic+peaks (228) /
    /// extended+masked (300) / IW (300) / both (372) caches so a
    /// recipe change doesn't accidentally read stale features.
    /// Added 2026-05-14 for V0_20a IW integration.
    feature_kind: u8,
}

impl CacheConfig {
    /// Pack the feature-shape into a single byte:
    ///   bit 0 = extended_features (masked block present)
    ///   bit 1 = compute_iw_features (IW block present)
    pub fn pack_kind(extended: bool, iw: bool) -> u8 {
        let mut k = 0u8;
        if extended {
            k |= 1;
        }
        if iw {
            k |= 2;
        }
        k
    }
}

/// Cached features without human scores (which are target-metric-dependent).
struct CachedFeatures {
    name: String,
    features: Vec<Vec<f64>>,
    ref_keys: Vec<String>,
    /// Original pair indices (which pairs from the dataset produced valid features).
    valid_indices: Vec<u32>,
}

fn save_feature_cache(
    path: &Path,
    ds: &DatasetWithFeatures,
    valid_indices: &[u32],
    config: &CacheConfig,
) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);

    // Magic + version (v3: f32 features for 2× smaller files)
    f.write_all(b"ZSFC")?;
    f.write_all(&3u32.to_le_bytes())?;

    // Validation fields
    f.write_all(&config.num_scales.to_le_bytes())?;
    f.write_all(&[config.blur_passes])?;
    f.write_all(&config.blur_radius.to_le_bytes())?;
    // Repurpose former masking_bits reserved field for feature_kind
    // (bit 0 = extended, bit 1 = iw). v3 caches were always written
    // with 0 here, which maps to "basic + peaks only" — that matches
    // existing pre-IW caches with 228 features. Caches written with
    // --iw-features set will have bit 1 = 1 and force re-extraction
    // on a non-IW load.
    f.write_all(&(config.feature_kind as u32).to_le_bytes())?;

    let n_pairs = ds.features.len() as u32;
    let n_features = if ds.features.is_empty() {
        0u16
    } else {
        ds.features[0].len() as u16
    };
    f.write_all(&n_pairs.to_le_bytes())?;
    f.write_all(&n_features.to_le_bytes())?;

    // Dataset name
    let name_bytes = ds.name.as_bytes();
    f.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
    f.write_all(name_bytes)?;

    // Valid pair indices (which original pairs produced non-NaN features)
    for &idx in valid_indices {
        f.write_all(&idx.to_le_bytes())?;
    }

    // Features as f32 (flat row-major) — halves storage vs f64
    for row in &ds.features {
        for &v in row {
            f.write_all(&(v as f32).to_le_bytes())?;
        }
    }

    // Ref keys
    for key in &ds.ref_keys {
        let kb = key.as_bytes();
        f.write_all(&(kb.len() as u16).to_le_bytes())?;
        f.write_all(kb)?;
    }

    f.flush()?;
    Ok(())
}

fn load_feature_cache(
    path: &Path,
    config: &CacheConfig,
) -> std::io::Result<Option<CachedFeatures>> {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };

    let mut pos = 0usize;

    let read_bytes = |pos: &mut usize, n: usize| -> std::io::Result<&[u8]> {
        if *pos + n > data.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "truncated cache file",
            ));
        }
        let slice = &data[*pos..*pos + n];
        *pos += n;
        Ok(slice)
    };

    // Magic
    let magic = read_bytes(&mut pos, 4)?;
    if magic != b"ZSFC" {
        eprintln!("Feature cache: invalid magic, recomputing");
        return Ok(None);
    }

    // Version
    let version = u32::from_le_bytes(read_bytes(&mut pos, 4)?.try_into().unwrap());
    if version != 2 && version != 3 {
        eprintln!(
            "Feature cache: version {} (expected 2 or 3), recomputing",
            version
        );
        return Ok(None);
    }

    // Validation fields (same layout for v2 and v3)
    let num_scales = u32::from_le_bytes(read_bytes(&mut pos, 4)?.try_into().unwrap());
    let blur_passes = read_bytes(&mut pos, 1)?[0];
    let blur_radius = u32::from_le_bytes(read_bytes(&mut pos, 4)?.try_into().unwrap());
    // Repurposed reserved field — feature_kind bit-flag
    // (bit 0 = extended, bit 1 = iw). Pre-IW caches wrote 0.
    let feature_kind = u32::from_le_bytes(read_bytes(&mut pos, 4)?.try_into().unwrap()) as u8;

    if num_scales != config.num_scales {
        eprintln!(
            "Feature cache: num_scales mismatch (cache={}, current={}), recomputing",
            num_scales, config.num_scales
        );
        return Ok(None);
    }
    if blur_passes != config.blur_passes {
        eprintln!(
            "Feature cache: blur_passes mismatch (cache={}, current={}), recomputing",
            blur_passes, config.blur_passes
        );
        return Ok(None);
    }
    if blur_radius != config.blur_radius {
        eprintln!(
            "Feature cache: blur_radius mismatch (cache={}, current={}), recomputing",
            blur_radius, config.blur_radius
        );
        return Ok(None);
    }
    if feature_kind != config.feature_kind {
        eprintln!(
            "Feature cache: feature_kind mismatch (cache=0b{:02b}, current=0b{:02b}; bit0=extended, bit1=iw), recomputing",
            feature_kind, config.feature_kind
        );
        return Ok(None);
    }

    // Pair/feature counts — v2 uses u64, v3 uses u32/u16
    let (n_pairs, n_features) = if version == 2 {
        let np = u64::from_le_bytes(read_bytes(&mut pos, 8)?.try_into().unwrap()) as usize;
        let nf = u64::from_le_bytes(read_bytes(&mut pos, 8)?.try_into().unwrap()) as usize;
        (np, nf)
    } else {
        let np = u32::from_le_bytes(read_bytes(&mut pos, 4)?.try_into().unwrap()) as usize;
        let nf = u16::from_le_bytes(read_bytes(&mut pos, 2)?.try_into().unwrap()) as usize;
        (np, nf)
    };

    // Name — v2 uses u32 len, v3 uses u16 len
    let name_len = if version == 2 {
        u32::from_le_bytes(read_bytes(&mut pos, 4)?.try_into().unwrap()) as usize
    } else {
        u16::from_le_bytes(read_bytes(&mut pos, 2)?.try_into().unwrap()) as usize
    };
    let name = String::from_utf8_lossy(read_bytes(&mut pos, name_len)?).to_string();

    // Valid pair indices
    let mut valid_indices = Vec::with_capacity(n_pairs);
    for _ in 0..n_pairs {
        valid_indices.push(u32::from_le_bytes(
            read_bytes(&mut pos, 4)?.try_into().unwrap(),
        ));
    }

    // Features — v2 stores f64, v3 stores f32 (promoted to f64 on load)
    let mut features = Vec::with_capacity(n_pairs);
    if version == 2 {
        for _ in 0..n_pairs {
            let mut row = Vec::with_capacity(n_features);
            for _ in 0..n_features {
                row.push(f64::from_le_bytes(
                    read_bytes(&mut pos, 8)?.try_into().unwrap(),
                ));
            }
            features.push(row);
        }
    } else {
        for _ in 0..n_pairs {
            let mut row = Vec::with_capacity(n_features);
            for _ in 0..n_features {
                row.push(f32::from_le_bytes(read_bytes(&mut pos, 4)?.try_into().unwrap()) as f64);
            }
            features.push(row);
        }
    }

    // Ref keys — v2 uses u32 len, v3 uses u16 len
    let mut ref_keys = Vec::with_capacity(n_pairs);
    for _ in 0..n_pairs {
        let klen = if version == 2 {
            u32::from_le_bytes(read_bytes(&mut pos, 4)?.try_into().unwrap()) as usize
        } else {
            u16::from_le_bytes(read_bytes(&mut pos, 2)?.try_into().unwrap()) as usize
        };
        let key = String::from_utf8_lossy(read_bytes(&mut pos, klen)?).to_string();
        ref_keys.push(key);
    }

    Ok(Some(CachedFeatures {
        name,
        features,
        ref_keys,
        valid_indices,
    }))
}

fn log_line(msg: &str, log: &mut Vec<String>) {
    println!("{}", msg);
    log.push(msg.to_string());
}

fn main() {
    let args = Args::parse();

    // Scale-invariance mode: ingest a pyramid CSV and emit slope analysis.
    // Bypasses dataset loading entirely.
    if let Some(csv_path) = args.scale_invariance.as_deref() {
        let weights: Vec<f64> = if let Some(wf) = args.weights_file.as_deref() {
            load_weights_file(wf)
        } else {
            zensim::WEIGHTS.to_vec()
        };
        if let Err(e) =
            scale_invariance::run(csv_path, args.scale_invariance_out.as_deref(), &weights)
        {
            eprintln!("scale-invariance: {}", e);
            std::process::exit(1);
        }
        return;
    }

    let args_dataset: PathBuf = args.dataset.clone().unwrap_or_else(|| {
        eprintln!("--dataset is required (or pass --scale-invariance for pyramid analysis)");
        std::process::exit(1);
    });
    let args_format: DatasetFormat = args.format.unwrap_or_else(|| {
        eprintln!("--format is required (or pass --scale-invariance for pyramid analysis)");
        std::process::exit(1);
    });

    let eval_only_weights = args.weights_file.is_some();
    let compute_all = args.compute_all || args.extract_only || eval_only_weights;
    let blur_passes = args.blur_passes;
    let blur_radius = args.blur_radius;
    let num_scales = args.num_scales;
    // Extraction uses the explicitly selected layout, without positional truncation.
    let extended_features = args.extended_features || args.extract_only;
    let extended_masking_strength = args.extended_masking_strength;
    // IW features are OFF by default and never auto-enabled — they
    // change the feature-vector shape, so callers must opt in explicitly.
    let iw_features = args.iw_features;
    let iw_strength = args.iw_strength;
    let downscale_filter = match args.downscale_filter.as_str() {
        "box" => zensim::DownscaleFilter::Box2x2,
        other => {
            eprintln!("Unknown downscale filter: {other}. Options: box");
            std::process::exit(1);
        }
    };

    let cache_config = CacheConfig {
        num_scales: num_scales as u32,
        blur_passes,
        blur_radius: blur_radius as u32,
        feature_kind: CacheConfig::pack_kind(extended_features, iw_features),
    };

    // Load and compute primary dataset (with optional caching)
    // Timestamped cache: saves produce `dataset.csv.features.YYYYMMDD_HHMMSS.bin`,
    // loads glob for `dataset.csv.features.*.bin` and pick the newest.
    let auto_cache_save_path = |dataset_path: &Path| -> PathBuf {
        let now = chrono::Local::now();
        let mut p = dataset_path.as_os_str().to_owned();
        p.push(format!(".features.{}.bin", now.format("%Y%m%d_%H%M%S")));
        PathBuf::from(p)
    };
    let find_latest_cache = |dataset_path: &Path| -> Option<PathBuf> {
        let mut pattern = dataset_path.as_os_str().to_owned();
        pattern.push(".features.*.bin");
        let pattern_str = pattern.to_string_lossy();
        let mut matches: Vec<PathBuf> = glob::glob(&pattern_str)
            .ok()?
            .filter_map(|r| r.ok())
            .collect();
        // Sort by filename descending — ISO timestamps sort lexicographically
        matches.sort();
        matches.pop()
    };
    // Also support legacy non-timestamped cache files
    let auto_cache_legacy_path = |dataset_path: &Path| -> PathBuf {
        let mut p = dataset_path.as_os_str().to_owned();
        p.push(".features.bin");
        PathBuf::from(p)
    };
    let find_cache_to_load = |dataset_path: &Path| -> Option<PathBuf> {
        // Prefer newest timestamped cache, fall back to legacy non-timestamped
        if let Some(p) = find_latest_cache(dataset_path) {
            return Some(p);
        }
        let legacy = auto_cache_legacy_path(dataset_path);
        if legacy.exists() { Some(legacy) } else { None }
    };

    let primary = if compute_all && !args.recompute {
        let explicit_cache = args.feature_cache.clone();
        let load_path = explicit_cache
            .clone()
            .or_else(|| find_cache_to_load(&args_dataset));
        let cache_start = std::time::Instant::now();
        let cached_result =
            load_path
                .as_ref()
                .and_then(|p| match load_feature_cache(p, &cache_config) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("Warning: failed to read cache {:?}: {}", p, e);
                        None
                    }
                });
        // Save path: explicit if given, otherwise timestamped
        let save_path = explicit_cache
            .clone()
            .unwrap_or_else(|| auto_cache_save_path(&args_dataset));
        match cached_result {
            Some(cached) => {
                // Reload pairs for fresh human_scores (target-metric-dependent)
                let pairs = load_pairs(
                    args_format,
                    &args_dataset,
                    args.max_images,
                    args.target_metric,
                );

                // Check if cache covers all pairs — if not, extract missing ones
                let max_cached_idx =
                    cached.valid_indices.iter().copied().max().unwrap_or(0) as usize;
                let cached_count = cached.valid_indices.len();

                if max_cached_idx < pairs.len().saturating_sub(1) && pairs.len() > cached_count {
                    // Incremental: cache is from a smaller dataset, extract new pairs
                    let cached_set: std::collections::HashSet<u32> =
                        cached.valid_indices.iter().copied().collect();
                    let new_pairs: Vec<(usize, ImagePair)> = pairs
                        .iter()
                        .enumerate()
                        .filter(|(idx, p)| {
                            !cached_set.contains(&(*idx as u32)) && !p.human_score.is_nan()
                        })
                        .map(|(idx, p)| (idx, p.clone()))
                        .collect();
                    let n_new = new_pairs.len();

                    println!(
                        "Cache has {} pairs, dataset has {} — extracting {} new pairs",
                        cached_count,
                        pairs.len(),
                        n_new
                    );

                    let ds = build_dataset_from_cache(cached, &pairs);
                    if n_new == 0 {
                        ds
                    } else {
                        // Extract features for new pairs using same logic as load_and_compute
                        let mut config = zensim::ZensimConfig::default();
                        config.compute_all_features = compute_all;
                        config.extended_features = extended_features;
                        config.extended_masking_strength = extended_masking_strength;
                        config.compute_iw_features = iw_features;
                        config.iw_strength = iw_strength;
                        config.blur_passes = blur_passes;
                        config.blur_radius = blur_radius;
                        config.num_scales = num_scales;
                        config.downscale_filter = downscale_filter;
                        let nan_result = zensim::ZensimResult::nan();

                        // Group new pairs by reference
                        let mut by_ref: std::collections::BTreeMap<
                            PathBuf,
                            Vec<(usize, ImagePair)>,
                        > = std::collections::BTreeMap::new();
                        for (idx, pair) in new_pairs {
                            by_ref
                                .entry(pair.reference.clone())
                                .or_default()
                                .push((idx, pair));
                        }

                        let progress_ctr = std::sync::atomic::AtomicU64::new(0);
                        let start_t = std::time::Instant::now();
                        let log_int = (n_new / 20).max(1000) as u64;

                        let ref_groups: Vec<(PathBuf, Vec<(usize, ImagePair)>)> =
                            by_ref.into_iter().collect();

                        let group_results: Vec<
                            Vec<(usize, String, f64, zensim::ZensimResult)>,
                        > = ref_groups
                            .par_iter()
                            .map(|(ref_path, group)| {
                                let fail = |grp: &[(usize, ImagePair)]| -> Vec<_> {
                                    progress_ctr.fetch_add(grp.len() as u64, std::sync::atomic::Ordering::Relaxed);
                                    grp.iter()
                                        .map(|(idx, pair)| {
                                            (
                                                *idx,
                                                reference_key(pair),
                                                pair.human_score,
                                                nan_result.clone(),
                                            )
                                        })
                                        .collect()
                                };

                                let src_img = match image::open(ref_path) {
                                    Ok(img) => img.to_rgb8(),
                                    Err(_) => return fail(group),
                                };
                                let (w, h) = src_img.dimensions();
                                let src_pixels: Vec<[u8; 3]> = src_img
                                    .pixels()
                                    .map(|p| [p.0[0], p.0[1], p.0[2]])
                                    .collect();

                                let precomputed = match zensim::precompute_reference_with_scales(
                                    &src_pixels,
                                    w as usize,
                                    h as usize,
                                    num_scales,
                                ) {
                                    Ok(p) => p,
                                    Err(_) => return fail(group),
                                };

                                group
                                    .par_iter()
                                    .map(|(idx, pair)| {
                                        let key = reference_key(pair);
                                        let result = match image::open(&pair.distorted) {
                                            Ok(img) => {
                                                let dst = img.to_rgb8();
                                                let (dw, dh) = dst.dimensions();
                                                if dw != w || dh != h {
                                                    nan_result.clone()
                                                } else {
                                                    let dst_pixels: Vec<[u8; 3]> = dst
                                                        .pixels()
                                                        .map(|p| [p.0[0], p.0[1], p.0[2]])
                                                        .collect();
                                                    zensim::compute_zensim_with_ref_and_config(
                                                        &precomputed,
                                                        &dst_pixels,
                                                        w as usize,
                                                        h as usize,
                                                        config,
                                                    )
                                                    .unwrap_or_else(|_| nan_result.clone())
                                                }
                                            }
                                            Err(_) => nan_result.clone(),
                                        };
                                        let prev = progress_ctr.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                        let cur = prev + 1;
                                        if prev / log_int != cur / log_int {
                                            let el = start_t.elapsed().as_secs_f64();
                                            let rate = cur as f64 / el;
                                            let eta = (n_new as f64 - cur as f64) / rate;
                                            eprintln!("  [{:.0}s] {}/{} new pairs ({:.1}%), {:.0}/s, ETA {:.0}s", el, cur, n_new, cur as f64 / n_new as f64 * 100.0, rate, eta);
                                        }
                                        (*idx, key, pair.human_score, result)
                                    })
                                    .collect()
                            })
                            .collect();

                        eprintln!(
                            "  New pairs extracted: {} in {:.1}s",
                            n_new,
                            start_t.elapsed().as_secs_f64()
                        );

                        // Merge cached + new features
                        let mut all_human_scores = ds.human_scores;
                        let mut all_features = ds.features;
                        let mut all_ref_keys = ds.ref_keys;
                        let mut all_valid_indices: Vec<u32> = cached_set.into_iter().collect();

                        let mut new_results: Vec<_> = group_results.into_iter().flatten().collect();
                        new_results.sort_by_key(|(idx, _, _, _)| *idx);

                        let mut n_new_valid = 0usize;
                        for (idx, key, hs, result) in new_results {
                            if result.score().is_finite() {
                                all_human_scores.push(hs);
                                all_features.push(result.into_features());
                                all_ref_keys.push(key);
                                all_valid_indices.push(idx as u32);
                                n_new_valid += 1;
                            }
                        }
                        all_valid_indices.sort();

                        println!(
                            "  Incremental: {} new valid pairs (total {})",
                            n_new_valid,
                            all_features.len()
                        );

                        let merged = DatasetWithFeatures {
                            name: ds.name,
                            human_scores: all_human_scores,
                            features: all_features,
                            ref_keys: all_ref_keys,
                        };

                        // Save updated cache (new timestamped file)
                        if let Err(e) = save_feature_cache(
                            &save_path,
                            &merged,
                            &all_valid_indices,
                            &cache_config,
                        ) {
                            eprintln!("Warning: failed to save updated cache: {}", e);
                        } else {
                            println!("Saved updated feature cache to {:?}", save_path);
                        }
                        merged
                    }
                } else {
                    let ds = build_dataset_from_cache(cached, &pairs);
                    println!(
                        "Loaded {} pairs ({} features) from cache {:?} ({:.1}s)",
                        ds.human_scores.len(),
                        if ds.features.is_empty() {
                            0
                        } else {
                            ds.features[0].len()
                        },
                        load_path.as_deref().unwrap_or(Path::new("?")),
                        cache_start.elapsed().as_secs_f64()
                    );
                    ds
                }
            }
            None => {
                let (ds, valid_indices) = load_and_compute(
                    &format!("{:?}", args_format),
                    args_format,
                    &args_dataset,
                    args.max_images,
                    compute_all,
                    blur_passes,
                    blur_radius,
                    num_scales,
                    args.target_metric,
                    extended_features,
                    extended_masking_strength,
                    iw_features,
                    iw_strength,
                    downscale_filter,
                );
                if let Err(e) = save_feature_cache(&save_path, &ds, &valid_indices, &cache_config) {
                    eprintln!("Warning: failed to save feature cache: {}", e);
                } else {
                    println!("Saved feature cache to {:?}", save_path);
                }
                ds
            }
        }
    } else {
        let (ds, valid_indices) = load_and_compute(
            &format!("{:?}", args_format),
            args_format,
            &args_dataset,
            args.max_images,
            compute_all,
            blur_passes,
            blur_radius,
            num_scales,
            args.target_metric,
            extended_features,
            extended_masking_strength,
            iw_features,
            iw_strength,
            downscale_filter,
        );
        if compute_all {
            let save_path = args
                .feature_cache
                .clone()
                .unwrap_or_else(|| auto_cache_save_path(&args_dataset));
            if let Err(e) = save_feature_cache(&save_path, &ds, &valid_indices, &cache_config) {
                eprintln!("Warning: failed to save feature cache: {}", e);
            } else {
                println!("Saved feature cache to {:?}", save_path);
            }
        }
        ds
    };

    let n_features_extracted = if primary.features.is_empty() {
        eprintln!("No valid results from primary dataset");
        return;
    } else {
        primary.features[0].len()
    };

    if args.extract_only {
        println!(
            "Extraction complete: {} pairs, {} features/pair",
            primary.features.len(),
            n_features_extracted
        );
        // Honor --features-csv even in extract-only mode so callers
        // can pipe features into external pipelines (e.g. the V0_6
        // mixed-supervision trainer).
        if let Some(ref csv_path) = args.features_csv {
            write_features_csv_with_refs(
                csv_path,
                &primary.human_scores,
                &primary.features,
                Some(&primary.ref_keys),
            );
            println!("Wrote features CSV: {}", csv_path.display());
        }
        return;
    }

    let n_features = n_features_extracted;

    // Load additional datasets if specified
    let mut all_datasets = vec![primary];
    if let Some(ref also_str) = args.also {
        for spec in also_str.split(',') {
            let parts: Vec<&str> = spec.splitn(2, ':').collect();
            if parts.len() != 2 {
                eprintln!("Invalid --also format: {}. Expected type:path", spec);
                continue;
            }
            let fmt = match parts[0] {
                "tid2013" => DatasetFormat::Tid2013,
                "kadid10k" => DatasetFormat::Kadid10k,
                "csiq" => DatasetFormat::Csiq,
                "pipal" => DatasetFormat::Pipal,
                "cid22" => DatasetFormat::Cid22,
                "konfig-iqa" | "konfig" => DatasetFormat::KonfigIqa,
                "synthetic" | "synth" => DatasetFormat::Synthetic,
                _ => {
                    eprintln!("Unknown format: {}", parts[0]);
                    continue;
                }
            };
            let also_path = Path::new(parts[1]);
            let also_load = find_cache_to_load(also_path);
            let also_save = auto_cache_save_path(also_path);
            let ds = if compute_all && !args.recompute {
                let t = std::time::Instant::now();
                let also_cached =
                    also_load
                        .as_ref()
                        .and_then(|p| match load_feature_cache(p, &cache_config) {
                            Ok(c) => c,
                            Err(e) => {
                                eprintln!("Warning: failed to read cache {:?}: {}", p, e);
                                None
                            }
                        });
                match also_cached {
                    Some(cached) => {
                        let pairs = load_pairs(fmt, also_path, 0, args.target_metric);
                        let ds = build_dataset_from_cache(cached, &pairs);
                        println!(
                            "Loaded {} pairs from cache {:?} ({:.1}s)",
                            ds.human_scores.len(),
                            also_load.as_deref().unwrap_or(Path::new("?")),
                            t.elapsed().as_secs_f64()
                        );
                        ds
                    }
                    None => {
                        let (ds, valid_indices) = load_and_compute(
                            parts[0],
                            fmt,
                            also_path,
                            0,
                            compute_all,
                            blur_passes,
                            blur_radius,
                            num_scales,
                            args.target_metric,
                            extended_features,
                            extended_masking_strength,
                            iw_features,
                            iw_strength,
                            downscale_filter,
                        );
                        if let Err(e) =
                            save_feature_cache(&also_save, &ds, &valid_indices, &cache_config)
                        {
                            eprintln!("Warning: failed to save feature cache: {}", e);
                        } else {
                            println!("Saved feature cache to {:?}", also_save);
                        }
                        ds
                    }
                }
            } else {
                let (ds, valid_indices) = load_and_compute(
                    parts[0],
                    fmt,
                    also_path,
                    0,
                    compute_all,
                    blur_passes,
                    blur_radius,
                    num_scales,
                    args.target_metric,
                    extended_features,
                    extended_masking_strength,
                    iw_features,
                    iw_strength,
                    downscale_filter,
                );
                if compute_all {
                    if let Err(e) =
                        save_feature_cache(&also_save, &ds, &valid_indices, &cache_config)
                    {
                        eprintln!("Warning: failed to save feature cache: {}", e);
                    } else {
                        println!("Saved feature cache to {:?}", also_save);
                    }
                }
                ds
            };
            all_datasets.push(ds);
        }
    }

    for ds in &all_datasets {
        if ds.features.iter().any(|row| row.len() != n_features) {
            eprintln!(
                "{}: inconsistent feature widths; regenerate an explicit common layout",
                ds.name
            );
            std::process::exit(1);
        }
    }

    // Normal mode: report correlations on primary dataset
    let ds = &all_datasets[0];
    let mut training_log: Vec<String> = Vec::new();
    report_embedded_correlations(ds, &mut training_log);

    // Evaluate custom weights if provided
    if let Some(ref weights_path) = args.weights_file {
        let custom_weights = load_weights_file(weights_path);
        if custom_weights.len() != n_features {
            eprintln!(
                "Weights file has {} values, expected {}",
                custom_weights.len(),
                n_features
            );
            std::process::exit(1);
        }

        let msg = format!("\n=== Custom weights from {:?} ===", weights_path);
        log_line(&msg, &mut training_log);
        for ds in &all_datasets {
            let feats: Vec<&[f64]> = ds.features.iter().map(|v| v.as_slice()).collect();
            let w = &custom_weights;
            let scores_and_dists: Vec<(f64, f64)> = feats
                .iter()
                .map(|f| {
                    zensim::try_score_from_features(f, w)
                        .expect("features and weights length mismatch")
                })
                .collect();
            let custom_scores: Vec<f64> = scores_and_dists.iter().map(|&(s, _)| s).collect();
            let raw_dists: Vec<f64> = scores_and_dists.iter().map(|&(_, d)| d).collect();
            let srocc = spearman_correlation(&ds.human_scores, &custom_scores);
            let plcc = pearson_correlation(&ds.human_scores, &custom_scores);
            // Raw distance correlations (negate distances since higher quality = lower distance)
            let neg_dists: Vec<f64> = raw_dists.iter().map(|d| -d).collect();
            let dist_srocc = spearman_correlation(&ds.human_scores, &neg_dists);
            // Count clamped-to-zero scores
            let n_clamped = custom_scores.iter().filter(|&&s| s <= 0.0).count();
            let pct_clamped = 100.0 * n_clamped as f64 / custom_scores.len() as f64;
            let krocc = fast_kendall(&ds.human_scores, &custom_scores);
            let dist_krocc = fast_kendall(&ds.human_scores, &neg_dists);
            log_line(
                &format!(
                    "  {}: SROCC={:.4}  KROCC={:.4}  PLCC={:.4}  | raw dist: SROCC={:.4} KROCC={:.4} | clamped: {}/{} ({:.1}%)",
                    ds.name,
                    srocc,
                    krocc,
                    plcc,
                    dist_srocc,
                    dist_krocc,
                    n_clamped,
                    custom_scores.len(),
                    pct_clamped
                ),
                &mut training_log,
            );
        }
    }

    // Output features CSV if requested
    if let Some(ref csv_path) = args.features_csv {
        write_features_csv(csv_path, &ds.human_scores, &ds.features);
    }
}

/// Load pairs from a dataset (for human_scores) without computing features.
/// Used when features are loaded from cache but human_scores need fresh loading.
fn load_pairs(
    format: DatasetFormat,
    path: &Path,
    max_images: usize,
    target_metric: Option<TargetMetric>,
) -> Vec<ImagePair> {
    let pairs = match format {
        DatasetFormat::Tid2013 => load_tid2013(path),
        DatasetFormat::Kadid10k => load_kadid10k(path),
        DatasetFormat::Csiq => load_csiq(path),
        DatasetFormat::Pipal => load_pipal(path),
        DatasetFormat::Cid22 => load_cid22(path),
        DatasetFormat::KonfigIqa => load_konfig_iqa(path),
        DatasetFormat::Synthetic => load_synthetic(path, target_metric),
    };
    if max_images > 0 && max_images < pairs.len() {
        pairs[..max_images].to_vec()
    } else {
        pairs
    }
}

/// Build a DatasetWithFeatures from cached features + freshly loaded pairs.
/// Uses valid_indices to look up human_scores from the original pairs list.
/// Filters out any cached entries whose indices exceed the pairs list
/// (can happen when the cache was built from a larger CSV than the current
/// target metric supports, e.g., DSSIM skips rows without dssim scores).
fn build_dataset_from_cache(cached: CachedFeatures, pairs: &[ImagePair]) -> DatasetWithFeatures {
    let n_pairs = pairs.len();
    let mut human_scores = Vec::with_capacity(cached.valid_indices.len());
    let mut features = Vec::with_capacity(cached.valid_indices.len());
    let mut ref_keys = Vec::with_capacity(cached.valid_indices.len());

    let mut nan_skipped = 0usize;
    for (i, &idx) in cached.valid_indices.iter().enumerate() {
        if (idx as usize) < n_pairs {
            let score = pairs[idx as usize].human_score;
            // Skip NaN placeholder pairs (rows without target metric, e.g. missing dssim)
            if score.is_nan() {
                nan_skipped += 1;
                continue;
            }
            human_scores.push(score);
            features.push(cached.features[i].clone());
            ref_keys.push(cached.ref_keys[i].clone());
        }
    }
    if nan_skipped > 0 {
        eprintln!(
            "  Skipped {nan_skipped} cached entries with NaN human_score (missing target metric)"
        );
    }

    DatasetWithFeatures {
        name: cached.name,
        human_scores,
        features,
        ref_keys,
    }
}

fn load_weights_file(path: &Path) -> Vec<f64> {
    let content = std::fs::read_to_string(path).expect("Failed to read weights file");
    content
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#') {
                return None;
            }
            // Handle Rust array format: "0.123456, " or plain "0.123456"
            let cleaned = trimmed.trim_end_matches(',').trim();
            cleaned.parse::<f64>().ok()
        })
        .collect()
}

/// Extract reference image key from a pair's reference path (file stem).
fn reference_key(pair: &ImagePair) -> String {
    pair.reference
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default()
}

/// Load a dataset and compute all features in parallel.
#[allow(clippy::too_many_arguments)]
fn load_and_compute(
    name: &str,
    format: DatasetFormat,
    path: &Path,
    max_images: usize,
    compute_all: bool,
    blur_passes: u8,
    blur_radius: usize,
    num_scales: usize,
    target_metric: Option<TargetMetric>,
    extended_features: bool,
    extended_masking_strength: f32,
    iw_features: bool,
    iw_strength: f32,
    downscale_filter: zensim::DownscaleFilter,
) -> (DatasetWithFeatures, Vec<u32>) {
    let pairs = match format {
        DatasetFormat::Tid2013 => load_tid2013(path),
        DatasetFormat::Kadid10k => load_kadid10k(path),
        DatasetFormat::Csiq => load_csiq(path),
        DatasetFormat::Pipal => load_pipal(path),
        DatasetFormat::Cid22 => load_cid22(path),
        DatasetFormat::KonfigIqa => load_konfig_iqa(path),
        DatasetFormat::Synthetic => load_synthetic(path, target_metric),
    };

    let pairs = if max_images > 0 && max_images < pairs.len() {
        pairs[..max_images].to_vec()
    } else {
        pairs
    };

    let total_pairs = pairs.len();
    println!("Loading {}: {} image pairs...", name, total_pairs);

    // Group pairs by reference image path for precomputed-reference reuse
    let mut by_ref: std::collections::BTreeMap<&Path, Vec<(usize, &ImagePair)>> =
        std::collections::BTreeMap::new();
    for (idx, pair) in pairs.iter().enumerate() {
        by_ref
            .entry(pair.reference.as_path())
            .or_default()
            .push((idx, pair));
    }
    let n_refs = by_ref.len();
    println!(
        "  {} unique references, {:.1} distorted/ref avg",
        n_refs,
        pairs.len() as f64 / n_refs as f64
    );

    let ref_groups: Vec<(&Path, Vec<(usize, &ImagePair)>)> = by_ref.into_iter().collect();

    let nan_result = zensim::ZensimResult::nan();

    let mut config = zensim::ZensimConfig::default();
    config.compute_all_features = compute_all;
    config.extended_features = extended_features;
    config.extended_masking_strength = extended_masking_strength;
    config.compute_iw_features = iw_features;
    config.iw_strength = iw_strength;
    config.blur_passes = blur_passes;
    config.blur_radius = blur_radius;
    config.num_scales = num_scales;
    config.downscale_filter = downscale_filter;

    // FPC is used for cache-key sizing. Extended adds 6/ch; IW adds 6/ch.
    let mut fpc = if config.extended_features {
        zensim::FEATURES_PER_CHANNEL_EXTENDED
    } else {
        zensim::FEATURES_PER_CHANNEL_WITH_PEAKS
    };
    if config.compute_iw_features {
        // FEATURES_PER_CHANNEL_IW = 6
        fpc += 6;
    }
    let total_features = num_scales * 3 * fpc;
    eprintln!(
        "  Config: scales={}, blur_passes={}, blur_radius={}, extended={}, downscale={:?}",
        num_scales, blur_passes, blur_radius, extended_features, downscale_filter,
    );
    eprintln!(
        "  Features: {} per channel × 3 channels × {} scales = {} total",
        fpc, num_scales, total_features,
    );
    if config.extended_features {
        eprintln!(
            "  Extended: masked_strength={:.1}, path=streaming",
            extended_masking_strength
        );
    }

    // Process reference groups in parallel
    let progress_counter = std::sync::atomic::AtomicU64::new(0);
    let start_time = std::time::Instant::now();
    let log_interval = (total_pairs / 20).max(1000) as u64; // ~5% increments

    let group_results: Vec<Vec<(usize, String, f64, zensim::ZensimResult)>> = ref_groups
        .par_iter()
        .map(|(ref_path, group)| {
            let fail = |grp: &[(usize, &ImagePair)]| -> Vec<_> {
                let prev = progress_counter
                    .fetch_add(grp.len() as u64, std::sync::atomic::Ordering::Relaxed);
                let new = prev + grp.len() as u64;
                if prev / log_interval != new / log_interval {
                    let elapsed = start_time.elapsed().as_secs_f64();
                    let rate = new as f64 / elapsed;
                    let eta = (total_pairs as f64 - new as f64) / rate;
                    eprintln!(
                        "  [{:.0}s] {}/{} pairs ({:.1}%), {:.0}/s, ETA {:.0}s",
                        elapsed,
                        new,
                        total_pairs,
                        new as f64 / total_pairs as f64 * 100.0,
                        rate,
                        eta,
                    );
                }
                grp.iter()
                    .map(|(idx, pair)| {
                        (
                            *idx,
                            reference_key(pair),
                            pair.human_score,
                            nan_result.clone(),
                        )
                    })
                    .collect()
            };

            // Load reference image once
            let src_img = match image::open(ref_path) {
                Ok(img) => img.to_rgb8(),
                Err(_) => return fail(group),
            };
            let (w, h) = src_img.dimensions();
            let src_pixels: Vec<[u8; 3]> =
                src_img.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();

            // Precompute reference XYB + downscale pyramid
            let precomputed = match zensim::precompute_reference_with_scales(
                &src_pixels,
                w as usize,
                h as usize,
                num_scales,
            ) {
                Ok(p) => p,
                Err(_) => return fail(group),
            };

            // Compare each distorted image against the reference (parallel)
            group
                .par_iter()
                .map(|(idx, pair)| {
                    let key = reference_key(pair);
                    let result = match image::open(&pair.distorted) {
                        Ok(img) => {
                            let dst = img.to_rgb8();
                            let (dw, dh) = dst.dimensions();
                            if dw != w || dh != h {
                                nan_result.clone()
                            } else {
                                let dst_pixels: Vec<[u8; 3]> =
                                    dst.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
                                zensim::compute_zensim_with_ref_and_config(
                                    &precomputed,
                                    &dst_pixels,
                                    w as usize,
                                    h as usize,
                                    config,
                                )
                                .unwrap_or_else(|_| nan_result.clone())
                            }
                        }
                        Err(_) => nan_result.clone(),
                    };
                    let prev = progress_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let new = prev + 1;
                    if prev / log_interval != new / log_interval {
                        let elapsed = start_time.elapsed().as_secs_f64();
                        let rate = new as f64 / elapsed;
                        let eta = (total_pairs as f64 - new as f64) / rate;
                        eprintln!(
                            "  [{:.0}s] {}/{} pairs ({:.1}%), {:.0}/s, ETA {:.0}s",
                            elapsed,
                            new,
                            total_pairs,
                            new as f64 / total_pairs as f64 * 100.0,
                            rate,
                            eta,
                        );
                    }
                    (*idx, key, pair.human_score, result)
                })
                .collect()
        })
        .collect();

    let total_elapsed = start_time.elapsed().as_secs_f64();
    eprintln!(
        "  Feature extraction: {} pairs in {:.1}s ({:.0}/s)",
        total_pairs,
        total_elapsed,
        total_pairs as f64 / total_elapsed,
    );

    // Flatten and sort back to original pair order
    let mut results: Vec<(usize, String, f64, zensim::ZensimResult)> =
        group_results.into_iter().flatten().collect();
    results.sort_by_key(|(idx, _, _, _)| *idx);

    let mut human_scores = Vec::new();
    let mut features = Vec::new();
    let mut ref_keys = Vec::new();
    let mut valid_indices = Vec::new();
    let mut n_valid = 0;

    for (idx, key, hs, result) in results {
        if result.score().is_finite() {
            human_scores.push(hs);
            features.push(result.into_features());
            ref_keys.push(key);
            valid_indices.push(idx as u32);
            n_valid += 1;
        }
    }

    println!("  {} valid pairs from {}", n_valid, name);

    (
        DatasetWithFeatures {
            name: name.to_string(),
            human_scores,
            features,
            ref_keys,
        },
        valid_indices,
    )
}

/// Expand embedded WEIGHTS (228 entries) to match a wider feature layout.
/// When extra scales are used, pads with zeros for extra scale features.
fn expand_embedded_weights(n_features: usize) -> Vec<f64> {
    let embedded = &zensim::WEIGHTS;
    if n_features == embedded.len() {
        return embedded.to_vec();
    }

    let mut expanded = vec![0.0; n_features];

    // Copy what fits from embedded weights
    let copy_len = n_features.min(embedded.len());
    expanded[..copy_len].copy_from_slice(&embedded[..copy_len]);

    expanded
}

/// Report correlations using embedded WEIGHTS.
fn report_embedded_correlations(ds: &DatasetWithFeatures, log: &mut Vec<String>) {
    let ew = expand_embedded_weights(ds.features[0].len());
    let metric_scores: Vec<f64> = ds
        .features
        .iter()
        .map(|f| {
            zensim::try_score_from_features(f, &ew)
                .expect("features and weights length mismatch")
                .0
        })
        .collect();

    let srocc = spearman_correlation(&ds.human_scores, &metric_scores);
    let plcc = pearson_correlation(&ds.human_scores, &metric_scores);

    log_line(
        &format!(
            "\n=== {} — Correlation with Human Ratings (embedded weights) ===",
            ds.name
        ),
        log,
    );
    let krocc = fast_kendall(&ds.human_scores, &metric_scores);
    log_line(&format!("SROCC (Spearman):  {:.4}", srocc), log);
    log_line(&format!("PLCC  (Pearson):   {:.4}", plcc), log);
    log_line(&format!("KROCC (Kendall):   {:.4}", krocc), log);

    let min_m = metric_scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_m = metric_scores
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let mean_m: f64 = metric_scores.iter().sum::<f64>() / metric_scores.len() as f64;
    log_line(
        &format!(
            "Metric score range: {:.2} to {:.2}, mean: {:.2}",
            min_m, max_m, mean_m
        ),
        log,
    );

    let raw_dists: Vec<f64> = ds
        .features
        .iter()
        .map(|f| {
            zensim::try_score_from_features(f, &ew)
                .expect("features and weights length mismatch")
                .1
        })
        .collect();
    let min_d = raw_dists.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_d = raw_dists.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean_d: f64 = raw_dists.iter().sum::<f64>() / raw_dists.len() as f64;
    let mut sorted_d = raw_dists.clone();
    sorted_d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p10 = sorted_d[sorted_d.len() / 10];
    let p50 = sorted_d[sorted_d.len() / 2];
    let p90 = sorted_d[sorted_d.len() * 9 / 10];
    log_line(
        &format!(
            "Raw distance: min={:.3}, p10={:.3}, p50={:.3}, p90={:.3}, max={:.3}, mean={:.3}",
            min_d, p10, p50, p90, max_d, mean_d
        ),
        log,
    );
    // Raw distance correlations (negate distances since higher quality = lower distance)
    let neg_dists: Vec<f64> = raw_dists.iter().map(|d| -d).collect();
    let dist_srocc = spearman_correlation(&ds.human_scores, &neg_dists);
    let n_clamped = metric_scores.iter().filter(|&&s| s <= 0.0).count();
    let pct_clamped = 100.0 * n_clamped as f64 / metric_scores.len() as f64;
    let dist_krocc = fast_kendall(&ds.human_scores, &neg_dists);
    log_line(
        &format!(
            "Raw dist corr: SROCC={:.4}  KROCC={:.4} | clamped scores: {}/{} ({:.1}%)\n",
            dist_srocc,
            dist_krocc,
            n_clamped,
            metric_scores.len(),
            pct_clamped
        ),
        log,
    );
}

fn write_features_csv(path: &Path, human_scores: &[f64], features: &[Vec<f64>]) {
    write_features_csv_with_refs(path, human_scores, features, None);
}

fn write_features_csv_with_refs(
    path: &Path,
    human_scores: &[f64],
    features: &[Vec<f64>],
    ref_basenames: Option<&[String]>,
) {
    use std::io::{BufWriter, Write};
    // BufWriter cuts the 10k-row CSV write from ~5 minutes to seconds
    // on /mnt/v (bytes-at-a-time `write!` over a CIFS-backed mount was
    // I/O-bound on syscalls).
    let raw = std::fs::File::create(path).expect("Failed to create features CSV");
    let mut f = BufWriter::with_capacity(1 << 20, raw);

    let n_features = features[0].len();
    // Optional `ref_basename` column lets the V0_6 trainer group rows
    // by source reference image for RankNet — without it, every pair
    // ends up in its own singleton group and the pairwise loss never
    // fires.
    //
    // When ref_basenames is provided we skip the metric_score /
    // raw_distance columns entirely. They're only useful for the
    // human-eval pipeline (analyzer plotting) and require running
    // zensim's V0_2 weights against every row, which costs ~20 ms on
    // a 300-feature vector × 10k rows = several minutes per dataset.
    let extract_mode = ref_basenames.is_some();
    if ref_basenames.is_some() {
        write!(f, "ref_basename,").unwrap();
    }
    write!(f, "human_score").unwrap();
    if !extract_mode {
        write!(f, ",metric_score,raw_distance").unwrap();
    }
    for i in 0..n_features {
        write!(f, ",f{}", i).unwrap();
    }
    writeln!(f).unwrap();

    let ew = if extract_mode {
        Vec::new()
    } else {
        expand_embedded_weights(n_features)
    };
    for (i, (human, feat)) in human_scores.iter().zip(features).enumerate() {
        if let Some(refs) = ref_basenames {
            write!(f, "{},", refs.get(i).map(|s| s.as_str()).unwrap_or("")).unwrap();
        }
        write!(f, "{}", human).unwrap();
        if !extract_mode {
            let (score, raw) = zensim::try_score_from_features(feat, &ew)
                .expect("features and weights length mismatch");
            write!(f, ",{},{}", score, raw).unwrap();
        }
        for v in feat {
            write!(f, ",{}", v).unwrap();
        }
        writeln!(f).unwrap();
    }
    f.flush().expect("Failed to flush features CSV");
    println!("Wrote features to {:?}", path);
}

/// O(n log n) Kendall tau-b using merge sort counting.
fn fast_kendall(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }

    // Sort by x, then count inversions in y using merge sort
    let mut pairs: Vec<(f64, f64)> = x.iter().copied().zip(y.iter().copied()).collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Count ties in x and y
    let mut x_ties = 0i64;
    let mut y_ties = 0i64;
    let mut xy_ties = 0i64;
    {
        let mut i = 0;
        while i < n {
            let mut j = i + 1;
            while j < n && pairs[j].0 == pairs[i].0 {
                j += 1;
            }
            let t = (j - i) as i64;
            x_ties += t * (t - 1) / 2;
            i = j;
        }
    }
    {
        let mut sorted_y: Vec<f64> = pairs.iter().map(|p| p.1).collect();
        sorted_y.sort_by(|a, b| a.total_cmp(b));
        let mut i = 0;
        while i < n {
            let mut j = i + 1;
            while j < n && sorted_y[j] == sorted_y[i] {
                j += 1;
            }
            let t = (j - i) as i64;
            y_ties += t * (t - 1) / 2;
            i = j;
        }
    }
    // Count joint ties
    {
        let mut i = 0;
        while i < n {
            let mut j = i + 1;
            while j < n && pairs[j].0 == pairs[i].0 && pairs[j].1 == pairs[i].1 {
                j += 1;
            }
            let t = (j - i) as i64;
            xy_ties += t * (t - 1) / 2;
            // Need to also check other groups with same x but different y
            i = j;
        }
    }

    // Merge sort on y values counts inversions (discordant pairs, excluding x-ties)
    let mut y_vals: Vec<f64> = pairs.iter().map(|p| p.1).collect();
    let swaps = merge_sort_count(&mut y_vals) as i64;

    let n_pairs = (n as i64) * (n as i64 - 1) / 2;
    // S = concordant - discordant = (non-tied pairs) - 2 * discordant
    let s = n_pairs - x_ties - y_ties + xy_ties - 2 * swaps;

    // Kendall tau-b: normalize by geometric mean of (pairs - x_ties) and (pairs - y_ties)
    let denom = ((n_pairs - x_ties) as f64 * (n_pairs - y_ties) as f64).sqrt();
    if denom == 0.0 {
        return 0.0;
    }

    s as f64 / denom
}

/// Merge sort that counts inversions (for Kendall tau computation).
fn merge_sort_count(arr: &mut [f64]) -> usize {
    let n = arr.len();
    if n <= 1 {
        return 0;
    }
    let mid = n / 2;
    let mut left = arr[..mid].to_vec();
    let mut right = arr[mid..].to_vec();
    let mut count = merge_sort_count(&mut left) + merge_sort_count(&mut right);

    let mut i = 0;
    let mut j = 0;
    let mut k = 0;
    while i < left.len() && j < right.len() {
        if left[i] <= right[j] {
            arr[k] = left[i];
            i += 1;
        } else {
            arr[k] = right[j];
            count += left.len() - i; // all remaining left elements are inversions
            j += 1;
        }
        k += 1;
    }
    while i < left.len() {
        arr[k] = left[i];
        i += 1;
        k += 1;
    }
    while j < right.len() {
        arr[k] = right[j];
        j += 1;
        k += 1;
    }
    count
}

fn load_tid2013(base: &Path) -> Vec<ImagePair> {
    let mos_path = base.join("mos_with_names.txt");
    if !mos_path.exists() {
        // Try alternative locations
        let alt = base.join("mos.txt");
        if alt.exists() {
            return load_tid2013_mos(&alt, base);
        }
        eprintln!("Cannot find mos_with_names.txt in {:?}", base);
        eprintln!("Expected: <base>/mos_with_names.txt with lines like: <mos> <filename>");
        return vec![];
    }
    load_tid2013_mos(&mos_path, base)
}

/// `lowercased file name -> real path` for one directory.
///
/// TID2013 ships its 25th reference LOWERCASE (`i25.bmp`, matching its source
/// `i25.bmp`) while the other 24 are uppercase (`I01.BMP`), and the distorted
/// set mixes cases as well. Forcing either case drops whole references.
fn case_insensitive_index(dir: &Path) -> std::collections::HashMap<String, PathBuf> {
    let mut idx = std::collections::HashMap::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for entry in rd.flatten() {
            idx.insert(
                entry.file_name().to_string_lossy().to_ascii_lowercase(),
                entry.path(),
            );
        }
    }
    idx
}

/// Look `name` up in a [`case_insensitive_index`], also trying the alternate
/// extensions a corpus is known to ship (`.bmp` vs `.png`).
fn lookup_ci(idx: &std::collections::HashMap<String, PathBuf>, name: &str) -> Option<PathBuf> {
    idx.get(&name.to_ascii_lowercase()).cloned()
}

fn load_tid2013_mos(mos_path: &Path, base: &Path) -> Vec<ImagePair> {
    let content = std::fs::read_to_string(mos_path).expect("Failed to read MOS file");
    let mut pairs = Vec::new();

    // Built ONCE per directory: the per-row lookup is a hash hit, not a readdir.
    let ref_idx = case_insensitive_index(&base.join("reference_images"));
    let dist_idx = case_insensitive_index(&base.join("distorted_images"));
    let mut missing: Vec<String> = Vec::new();
    let mut label_rows = 0usize;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Format: "mos_value filename" e.g., "5.16 I01_01_1.bmp"
        let parts: Vec<&str> = line.splitn(2, char::is_whitespace).collect();
        if parts.len() < 2 {
            continue;
        }
        let mos: f64 = match parts[0].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let filename = parts[1].trim();
        label_rows += 1;

        // Extract reference image number: I01_01_1.bmp -> reference_images/I01.BMP
        // BOTH sides resolve case-insensitively (see `case_insensitive_index`):
        // forcing the reference stem upper-case dropped all 120 rows of `i25`
        // for six weeks (2026-08-30; the same defect the Python pairs builder
        // carried, fixed in `build_fr_corpus_pairs.py` 657100db).
        let ref_stem = &filename[..3];
        let ref_path = match lookup_ci(&ref_idx, &format!("{ref_stem}.bmp"))
            .or_else(|| lookup_ci(&ref_idx, &format!("{ref_stem}.png")))
        {
            Some(p) => p,
            None => {
                missing.push(format!("reference_images/{ref_stem}.BMP (for {filename})"));
                continue;
            }
        };

        let dist_path = match lookup_ci(&dist_idx, filename) {
            Some(p) => p,
            None => {
                missing.push(format!("distorted_images/{filename}"));
                continue;
            }
        };

        // TID2013 MOS: 0-9 scale (higher = better)
        pairs.push(ImagePair {
            reference: ref_path,
            distorted: dist_path,
            human_score: mos / 9.0, // Normalize to 0-1
        });
    }

    // NO GRACEFUL SKIPS: a loader that silently emits fewer rows than the label
    // file lists produces a table that looks complete and is not. The skip is
    // the CALLER's decision and must be visible in the invocation.
    if !missing.is_empty() {
        let head = missing
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n  ");
        let more = if missing.len() > 10 {
            format!("\n  ... and {} more", missing.len() - 10)
        } else {
            String::new()
        };
        eprintln!(
            "load_tid2013: {} of {} label rows name a file that does not exist under {}:\n  {}{}",
            missing.len(),
            label_rows,
            base.display(),
            head,
            more
        );
        if std::env::var("ZENSIM_ALLOW_MISSING_PAIRS").as_deref() == Ok("1") {
            eprintln!(
                "load_tid2013: continuing with {} of {} pairs (ZENSIM_ALLOW_MISSING_PAIRS=1)",
                pairs.len(),
                label_rows
            );
        } else {
            eprintln!(
                "load_tid2013: refusing to emit a partial dataset. \
                 Set ZENSIM_ALLOW_MISSING_PAIRS=1 to opt in, visibly, at the call site."
            );
            std::process::exit(3);
        }
    }

    pairs
}

fn load_kadid10k(base: &Path) -> Vec<ImagePair> {
    let dmos_path = base.join("dmos.csv");
    if !dmos_path.exists() {
        eprintln!("Cannot find dmos.csv in {:?}", base);
        return vec![];
    }

    let mut rdr = csv::Reader::from_path(&dmos_path).expect("Failed to open dmos.csv");
    let mut pairs = Vec::new();

    for result in rdr.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => continue,
        };
        // KADID-10k CSV format: dist_img, ref_img, dmos, std
        if record.len() < 3 {
            continue;
        }

        let dist_name = &record[0];
        let ref_name = &record[1];
        let dmos: f64 = match record[2].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        let ref_path = base.join("images").join(ref_name);
        let dist_path = base.join("images").join(dist_name);

        // KADID DMOS: 1-5 scale (higher = better quality)
        pairs.push(ImagePair {
            reference: ref_path,
            distorted: dist_path,
            human_score: (dmos - 1.0) / 4.0, // Normalize to 0-1
        });
    }

    pairs
}

fn load_csiq(base: &Path) -> Vec<ImagePair> {
    // CSIQ uses an Excel file for DMOS
    let dmos_path = base.join("csiq.DMOS.xlsx");
    if !dmos_path.exists() {
        eprintln!("Cannot find csiq.DMOS.xlsx in {:?}", base);
        eprintln!("Looking for alternative formats...");

        // Try CSV version
        let csv_path = base.join("csiq_dmos.csv");
        if csv_path.exists() {
            return load_csiq_csv(&csv_path, base);
        }
        return vec![];
    }

    load_csiq_xlsx(&dmos_path, base)
}

fn load_csiq_xlsx(xlsx_path: &Path, base: &Path) -> Vec<ImagePair> {
    let mut workbook: Xlsx<_> =
        calamine::open_workbook(xlsx_path).expect("Failed to open CSIQ DMOS xlsx");

    let mut pairs = Vec::new();

    // Use 'all_by_image' sheet which has all pairs
    let range = workbook
        .worksheet_range("all_by_image")
        .expect("Failed to read 'all_by_image' sheet");

    // Mapping from xlsx distortion type to (directory name, filename label)
    let dist_map: HashMap<&str, (&str, &str)> = [
        ("noise", ("awgn", "AWGN")),
        ("blur", ("blur", "BLUR")),
        ("contrast", ("contrast", "contrast")),
        ("fnoise", ("fnoise", "fnoise")),
        ("jpeg", ("jpeg", "JPEG")),
        ("jpeg 2000", ("jpeg2000", "jpeg2000")),
    ]
    .into_iter()
    .collect();

    // Calamine strips leading empty columns. Actual layout:
    // [0]=image, [1]=dst_idx, [2]=dst_type, [3]=dst_lev, [4]=dmos_std, [5]=dmos
    for row in range.rows() {
        if row.len() < 6 {
            continue;
        }

        let img_name = match &row[0] {
            calamine::Data::String(s) => s.clone(),
            calamine::Data::Float(f) => format!("{}", *f as i64),
            calamine::Data::Int(i) => format!("{}", i),
            _ => continue,
        };

        // Skip header row
        if img_name == "image" {
            continue;
        }

        let dst_type = match &row[2] {
            calamine::Data::String(s) => s.clone(),
            _ => continue,
        };

        let dst_lev = match &row[3] {
            calamine::Data::Float(f) => *f as i64,
            calamine::Data::Int(i) => *i,
            _ => continue,
        };

        let dmos = match &row[5] {
            calamine::Data::Float(f) => *f,
            _ => continue,
        };

        let (dir_name, file_label) = match dist_map.get(dst_type.as_str()) {
            Some(v) => *v,
            None => {
                eprintln!("Unknown CSIQ distortion type: {}", dst_type);
                continue;
            }
        };

        let ref_path = base.join(format!("{}.png", img_name));
        let dist_path = base
            .join(dir_name)
            .join(format!("{}.{}.{}.png", img_name, file_label, dst_lev));

        // CSIQ DMOS: 0-1 scale (lower = better quality, higher = more distortion)
        pairs.push(ImagePair {
            reference: ref_path,
            distorted: dist_path,
            human_score: 1.0 - dmos, // Invert so higher = better
        });
    }

    pairs
}

fn load_csiq_csv(csv_path: &Path, base: &Path) -> Vec<ImagePair> {
    let mut rdr = csv::Reader::from_path(csv_path).expect("Failed to open CSIQ CSV");
    let mut pairs = Vec::new();

    for result in rdr.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => continue,
        };
        if record.len() < 3 {
            continue;
        }
        let dist_name = &record[0];
        let ref_name = &record[1];
        let dmos: f64 = match record[2].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        pairs.push(ImagePair {
            reference: base.join("src_imgs").join(ref_name),
            distorted: base.join("dst_imgs").join(dist_name),
            human_score: 1.0 - dmos,
        });
    }

    pairs
}

fn load_pipal(base: &Path) -> Vec<ImagePair> {
    let label_dir = base.join("Train_Label");
    let ref_dir = base.join("Train_Ref");

    if !label_dir.exists() {
        eprintln!("Cannot find Train_Label/ in {:?}", base);
        return vec![];
    }
    if !ref_dir.exists() {
        eprintln!("Cannot find Train_Ref/ in {:?}", base);
        return vec![];
    }

    // Distorted images are split across Distortion_1..4
    let dist_dirs: Vec<PathBuf> = (1..=4)
        .map(|i| base.join(format!("Distortion_{}", i)))
        .filter(|d| d.exists())
        .collect();

    if dist_dirs.is_empty() {
        eprintln!("Cannot find any Distortion_N/ directories in {:?}", base);
        return vec![];
    }

    // Read MOS range to normalize: PIPAL uses Elo-like scores (~900-1850)
    let mut all_scores: Vec<(PathBuf, PathBuf, f64, String)> = Vec::new();

    let mut label_files: Vec<_> = std::fs::read_dir(&label_dir)
        .expect("Failed to read Train_Label directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "txt"))
        .collect();
    label_files.sort_by_key(|e| e.file_name());

    for entry in &label_files {
        let label_path = entry.path();
        let ref_stem = label_path.file_stem().unwrap().to_string_lossy();
        let ref_path = ref_dir.join(format!("{}.bmp", ref_stem));

        if !ref_path.exists() {
            continue;
        }

        let content = match std::fs::read_to_string(&label_path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.splitn(2, ',').collect();
            if parts.len() != 2 {
                continue;
            }
            let dist_name = parts[0].trim();
            let mos: f64 = match parts[1].trim().parse() {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Find the distorted image across distortion directories
            let dist_path = dist_dirs
                .iter()
                .map(|d| d.join(dist_name))
                .find(|p| p.exists());

            if let Some(dist_path) = dist_path {
                all_scores.push((ref_path.clone(), dist_path, mos, ref_stem.to_string()));
            }
        }
    }

    if all_scores.is_empty() {
        eprintln!("No valid PIPAL pairs found");
        return vec![];
    }

    // Normalize MOS to 0-1: PIPAL uses Elo scores where higher = better
    let min_mos = all_scores
        .iter()
        .map(|(_, _, m, _)| *m)
        .fold(f64::INFINITY, f64::min);
    let max_mos = all_scores
        .iter()
        .map(|(_, _, m, _)| *m)
        .fold(f64::NEG_INFINITY, f64::max);
    let range = (max_mos - min_mos).max(1.0);

    let pairs: Vec<ImagePair> = all_scores
        .into_iter()
        .map(|(reference, distorted, mos, _)| ImagePair {
            reference,
            distorted,
            human_score: (mos - min_mos) / range,
        })
        .collect();

    println!(
        "  PIPAL: {} pairs, MOS range {:.1}..{:.1}",
        pairs.len(),
        min_mos,
        max_mos
    );

    pairs
}

fn load_cid22(base: &Path) -> Vec<ImagePair> {
    let csv_path = base.join("CID22_validation_set.csv");
    if !csv_path.exists() {
        eprintln!("Cannot find CID22_validation_set.csv in {:?}", base);
        return vec![];
    }

    let mut rdr =
        csv::Reader::from_path(&csv_path).expect("Failed to open CID22_validation_set.csv");
    let mut pairs = Vec::new();

    for result in rdr.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => continue,
        };
        // CID22 CSV: reference_img, distorted_img, encoder, setting, bpp, MCOS, RMOS, Elo, nb_pc_opinions
        if record.len() < 6 {
            continue;
        }

        let ref_name = &record[0];
        let dist_name = &record[1];
        let encoder = &record[2];

        // Skip self-reference rows
        if encoder == "Reference" {
            continue;
        }

        let mcos: f64 = match record[5].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        let ref_path = base.join(ref_name);
        let dist_path = base.join(dist_name);

        // MCOS: 0-100 scale (higher = better quality)
        pairs.push(ImagePair {
            reference: ref_path,
            distorted: dist_path,
            human_score: mcos / 100.0,
        });
    }

    println!("  CID22: {} pairs", pairs.len());
    pairs
}

fn load_konfig_iqa(base: &Path) -> Vec<ImagePair> {
    // KonFiG-IQA uses DCR (Degradation Category Rating) from EXP_III.
    // Raw data: individual worker ratings per (source, distortion, level).
    // We aggregate to mean DCR, then invert to quality (higher = better).
    let csv_path = base.join("DATA/EXP_III/data3.csv");
    if !csv_path.exists() {
        eprintln!("Cannot find DATA/EXP_III/data3.csv in {:?}", base);
        return vec![];
    }

    let mut rdr = csv::Reader::from_path(&csv_path).expect("Failed to open data3.csv");

    // Aggregate raw ratings: mean DCR per (source, distortion_type, level)
    let mut ratings: HashMap<(String, String, u32), Vec<u32>> = HashMap::new();
    for result in rdr.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => continue,
        };
        // Columns: Source, Distortion Type, BoostType, Distortion Level, HIT id,
        //          Assignment id, Worker id, Answer, Time
        if record.len() < 8 {
            continue;
        }
        let source = record[0].to_string();
        let dist_type = record[1].to_string();
        let level: u32 = match record[3].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let answer: u32 = match record[7].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        ratings
            .entry((source, dist_type, level))
            .or_default()
            .push(answer);
    }

    let mut pairs = Vec::new();
    for ((source, dist_type, level), vals) in &ratings {
        let mean_dcr: f64 = vals.iter().map(|&v| v as f64).sum::<f64>() / vals.len() as f64;
        // DCR: 0=imperceptible, 4=very annoying. Invert to quality 0-1.
        let quality = (4.0 - mean_dcr) / 4.0;

        // Image path: IMAGES/PartA/{source}/{distortion}/{source}_{distortion}_{level}.png
        let dist_path = base
            .join("IMAGES/PartA")
            .join(source)
            .join(dist_type)
            .join(format!("{source}_{dist_type}_{level}.png"));
        // Reference: IMAGES/reference_images/{source}_0.png
        let ref_path = base
            .join("IMAGES/reference_images")
            .join(format!("{source}_0.png"));

        if !dist_path.exists() {
            continue;
        }

        pairs.push(ImagePair {
            reference: ref_path,
            distorted: dist_path,
            human_score: quality,
        });
    }

    pairs.sort_by(|a, b| a.distorted.cmp(&b.distorted));
    println!(
        "  KonFiG-IQA: {} pairs from {} ratings",
        pairs.len(),
        ratings.len()
    );
    pairs
}

/// Load synthetic training CSV produced by generate_zensim_training.
///
/// Load synthetic training CSV with header-based column lookup.
///
/// Supports both old format (ssimulacra2, butteraugli columns) and new format
/// (gpu_ssimulacra2, gpu_butteraugli, cpu_ssimulacra2, cpu_butteraugli columns).
///
/// Scores are normalized to 0-1 (higher = better quality):
/// - SSIM2: `((ssim2 + 50) / 150).clamp(0, 1)` — monotonic, preserves SROCC
/// - Butteraugli: `1 / (1 + ba)` — monotonically decreasing, maps (0,inf) → (0,1]
fn load_synthetic(csv_path: &Path, target_metric: Option<TargetMetric>) -> Vec<ImagePair> {
    let metric = target_metric.unwrap_or_else(|| {
        eprintln!(
            "Warning: --target-metric not specified for synthetic dataset, defaulting to gpu-ssim2"
        );
        TargetMetric::GpuSsim2
    });

    let mut rdr = csv::Reader::from_path(csv_path).unwrap_or_else(|e| {
        eprintln!("Failed to open CSV {}: {}", csv_path.display(), e);
        std::process::exit(1);
    });

    // Header-based column lookup
    let headers = rdr.headers().unwrap().clone();
    let col = |name: &str| -> Option<usize> { headers.iter().position(|h| h == name) };
    let source_col = col("source_path").expect("CSV missing source_path column");
    let decoded_col = col("decoded_path").expect("CSV missing decoded_path column");

    // Resolve the metric column name based on target + available headers
    let metric_col = match metric {
        TargetMetric::GpuSsim2 => col("gpu_ssimulacra2").or_else(|| col("ssimulacra2")),
        TargetMetric::GpuButteraugli => col("gpu_butteraugli").or_else(|| col("butteraugli")),
        TargetMetric::CpuSsim2 => col("cpu_ssimulacra2"),
        TargetMetric::CpuButteraugli => col("cpu_butteraugli"),
        TargetMetric::CpuButteraugli3Norm => col("butteraugli_3norm"),
        TargetMetric::Dssim => col("dssim"),
    }
    .unwrap_or_else(|| {
        eprintln!("CSV missing column for {:?}", metric);
        eprintln!(
            "Available columns: {:?}",
            headers.iter().collect::<Vec<_>>()
        );
        std::process::exit(1);
    });

    let is_ssim2 = matches!(metric, TargetMetric::GpuSsim2 | TargetMetric::CpuSsim2);
    let is_dssim = matches!(metric, TargetMetric::Dssim);

    let mut pairs = Vec::new();
    let mut skipped = 0usize;

    for result in rdr.records() {
        let record = match result {
            Ok(r) => r,
            Err(e) => {
                eprintln!("  CSV parse error: {}", e);
                skipped += 1;
                continue;
            }
        };

        let source_path = PathBuf::from(&record[source_col]);
        let decoded_path = PathBuf::from(&record[decoded_col]);

        let raw_score: f64 = match record[metric_col].parse() {
            Ok(v) => v,
            Err(_) => {
                // Insert NaN placeholder to preserve row indices for cache alignment
                pairs.push(ImagePair {
                    reference: source_path,
                    distorted: decoded_path,
                    human_score: f64::NAN,
                });
                skipped += 1;
                continue;
            }
        };

        // Normalize to 0-1, higher = better
        let score = if is_ssim2 {
            ((raw_score + 50.0) / 150.0).clamp(0.0, 1.0)
        } else if is_dssim {
            // DSSIM: 0 = identical, ~0.1 = poor. Scale by 100 for better spread.
            1.0 / (1.0 + 100.0 * raw_score)
        } else {
            1.0 / (1.0 + raw_score)
        };

        if !score.is_finite() {
            // Insert NaN placeholder to preserve row indices for cache alignment
            pairs.push(ImagePair {
                reference: source_path,
                distorted: decoded_path,
                human_score: f64::NAN,
            });
            skipped += 1;
            continue;
        }

        pairs.push(ImagePair {
            reference: source_path,
            distorted: decoded_path,
            human_score: score,
        });
    }

    if skipped > 0 {
        eprintln!("  Synthetic: skipped {} invalid rows", skipped);
    }
    println!("  Synthetic: {} pairs, target={:?}", pairs.len(), metric,);
    pairs
}

// ===== Correlation statistics =====
//
// Dedup-K (2026-05-26): the local spearman/pearson/ranks impls (each
// ~25-50 LOC) were thin partial re-rolls of the canonical Mohammadi
// panel. Migrated to `zenstats::{spearman, pearson, ranks}`. The
// only API difference is panel.rs's `ranks` uses `(i+j-1)/2` while
// main.rs used `(i+j)/2 + 0.5` — both correctly emit mid-rank averages
// and produce identical Pearson-on-ranks (shift-invariant). The local
// `pearson_correlation` `var == 0.0` exact-zero guard becomes
// `zenstats::pearson`'s `< 1e-12` guard (both reject same values in
// practice). See `zensim/CHANGELOG.md` Unreleased / Changed.
//
// imazen/zensimx#41: the three delegating wrapper fns that lived here are now
// plain `use zenstats::{...}` aliases in the import block at the top of this
// file, so `zensim-validate/src` defines NO fn named spearman/pearson/ranks
// (gated by `tests/no_private_iqa_stats.rs`). `fast_kendall` above is NOT
// covered by that gate: it is an O(n log n) tau-b with an exact-tie
// predicate that `zenstats::kendall_tau`'s approximate-tie form cannot
// reproduce (CHANGELOG, "Knight's O(n log n) kendall_tau was rejected"),
// and it is a training objective (`TrainObjective::Krocc`), so replacing it
// is a trainer-number change that needs an owner decision.

#[cfg(test)]
mod tid_case_tests {
    use super::{case_insensitive_index, lookup_ci};
    use std::io::Write;

    /// TID2013 ships 24 uppercase references and ONE lowercase (`i25.bmp`).
    /// A loader that forces either case drops every row of the odd one out —
    /// 120 of 3,000 pairs, silently, which is exactly what happened until
    /// 2026-08-30.
    #[test]
    fn case_insensitive_index_resolves_both_casings() {
        let dir = std::env::temp_dir().join(format!(
            "zensim_validate_tid_case_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["I01.BMP", "i25.bmp"] {
            let mut f = std::fs::File::create(dir.join(name)).unwrap();
            f.write_all(b"x").unwrap();
        }

        let idx = case_insensitive_index(&dir);

        // Requested upper, stored upper.
        assert_eq!(lookup_ci(&idx, "I01.BMP").unwrap(), dir.join("I01.BMP"));
        // Requested lower, stored upper.
        assert_eq!(lookup_ci(&idx, "i01.bmp").unwrap(), dir.join("I01.BMP"));
        // Requested UPPER, stored lower -- the i25 case that was dropped.
        assert_eq!(lookup_ci(&idx, "I25.BMP").unwrap(), dir.join("i25.bmp"));
        // Absent stays absent (the loader must be able to fail loud).
        assert!(lookup_ci(&idx, "I26.BMP").is_none());

        std::fs::remove_dir_all(&dir).ok();
    }
}
