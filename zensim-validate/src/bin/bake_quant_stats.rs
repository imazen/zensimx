//! Cross-bake i8 quantization zero-out census for shipped/archived ZNPR v3
//! bakes in `zensim/weights/`.
//!
//! For each ZNPR v3 bake under `zensim/weights/` and `zensim/weights/archive/`
//! (recursive), inspect the L0 layer (the layer whose `in_dim == n_inputs`)
//! and report:
//!
//! - Per-feature `zero_fraction_at_l0`: fraction of L0 output columns where
//!   the *effective* quantized weight is zero. For I8 storage this is
//!   `weights[i, o] == 0` (the actual quantized bucket). For F32/F16 storage
//!   we simulate the same per-output `scale[o] = max_i |W[i, o]| / 127.0`
//!   quantization scheme so the f32 / f16 / i8 results are comparable.
//! - `mostly_zeroed` = `zero_fraction >= 0.5`, `fully_zeroed` =
//!   `zero_fraction == 1.0`, `survived` = `zero_fraction < 0.5`.
//! - `l1_share` = `Σ_o |W_f32[i, o]| / Σ_{i,o} |W_f32[i, o]|`.
//! - `importance` = `scaler_scale[i] * Σ_o |W_f32[i, o]|`.
//!
//! Outputs:
//! - `benchmarks/bake_quant_stats_2026-05-25/per_bake.tsv`
//! - `benchmarks/bake_quant_stats_2026-05-25/per_feature.tsv`
//! - `benchmarks/bake_quant_stats_2026-05-25/SUMMARY.md`
//!
//! Run:
//!   cargo run --release -p zensim-validate --bin bake_quant_stats

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use zenpredict::{Model, WeightStorage};

const REPO_ROOT: &str = "/home/lilith/work/zen/zensimx";
const OUT_DIR: &str = "/home/lilith/work/zen/zensimx/benchmarks/bake_quant_stats_2026-05-25";
const N_SCALES: usize = 4;
const N_CHANNELS: usize = 3;

// Per-channel block sizes from zensim/src/metric.rs constants.
const PER_CH_BASIC: usize = 13;
const PER_CH_WITH_PEAKS: usize = 19;
const PER_CH_EXTENDED: usize = 25;
const PER_CH_IW: usize = 6;

fn sha256_prefix(bytes: &[u8]) -> String {
    use std::io::Write as _;
    use std::process::{Command, Stdio};
    let mut child = Command::new("sha256sum")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("sha256sum");
    child.stdin.as_mut().unwrap().write_all(bytes).unwrap();
    let out = child.wait_with_output().unwrap();
    let s = String::from_utf8_lossy(&out.stdout);
    let full = s.split_whitespace().next().unwrap_or("").to_string();
    full.chars().take(8).collect()
}

fn discover_bakes(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for ent in entries.flatten() {
            let p = ent.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|e| e == "bin") {
                out.push(p);
            }
        }
    }
    walk(root, &mut out);
    out.sort();
    out
}

fn znpr_version(bytes: &[u8]) -> Option<u16> {
    if bytes.len() < 8 {
        return None;
    }
    if &bytes[..4] != b"ZNPR" {
        return None;
    }
    Some(u16::from_le_bytes([bytes[4], bytes[5]]))
}

/// Pick L0 — the first non-passthrough layer whose `in_dim == n_inputs`.
/// Falls back to first matching layer.
fn pick_l0_index(model: &Model) -> Option<usize> {
    let n_in = model.n_inputs();
    let layers: Vec<_> = model.layers().collect();
    for (idx, l) in layers.iter().enumerate() {
        if l.in_dim == n_in {
            let is_passthrough =
                l.in_dim == l.out_dim && matches!(l.activation, zenpredict::Activation::Identity);
            if !is_passthrough {
                return Some(idx);
            }
        }
    }
    for (idx, l) in layers.iter().enumerate() {
        if l.in_dim == n_in {
            return Some(idx);
        }
    }
    if layers.is_empty() { None } else { Some(0) }
}

#[derive(Debug)]
struct LayerStats {
    in_dim: usize,
    out_dim: usize,
    dtype: &'static str,
    /// Per-feature zero count across out_dim.
    zero_count: Vec<usize>,
    /// Per-feature sum |W_f32[i, o]|.
    sum_abs_w: Vec<f64>,
}

/// Compute LayerStats from a LayerView. For F32/F16 we simulate the
/// per-output i8 quantization to produce comparable zero counts.
fn compute_layer_stats(layer: &zenpredict::LayerView<'_>) -> LayerStats {
    let in_dim = layer.in_dim;
    let out_dim = layer.out_dim;
    let mut zero_count = vec![0usize; in_dim];
    let mut sum_abs_w = vec![0.0_f64; in_dim];

    let dtype = match &layer.weights {
        WeightStorage::F32(w) => {
            // Per-output column scale = max_i |W[i, o]| / 127.0
            // Pre-pass: compute scales.
            let mut col_max = vec![0.0_f32; out_dim];
            for i in 0..in_dim {
                let base = i * out_dim;
                for o in 0..out_dim {
                    let a = w[base + o].abs();
                    if a > col_max[o] {
                        col_max[o] = a;
                    }
                }
            }
            let mut scales = vec![1.0_f32; out_dim];
            for o in 0..out_dim {
                if col_max[o] > 0.0 {
                    scales[o] = col_max[o] / 127.0;
                }
            }
            for i in 0..in_dim {
                let base = i * out_dim;
                let mut s = 0.0_f64;
                for o in 0..out_dim {
                    let wv = w[base + o];
                    s += (wv as f64).abs();
                    let q = (wv / scales[o]).round() as i32;
                    let q_clamped = q.clamp(-128, 127);
                    if q_clamped == 0 {
                        zero_count[i] += 1;
                    }
                }
                sum_abs_w[i] = s;
            }
            "f32"
        }
        WeightStorage::F16(w) => {
            let mut col_max = vec![0.0_f32; out_dim];
            for i in 0..in_dim {
                let base = i * out_dim;
                for o in 0..out_dim {
                    let a = zenpredict::f16_bits_to_f32(w[base + o]).abs();
                    if a > col_max[o] {
                        col_max[o] = a;
                    }
                }
            }
            let mut scales = vec![1.0_f32; out_dim];
            for o in 0..out_dim {
                if col_max[o] > 0.0 {
                    scales[o] = col_max[o] / 127.0;
                }
            }
            for i in 0..in_dim {
                let base = i * out_dim;
                let mut s = 0.0_f64;
                for o in 0..out_dim {
                    let wv = zenpredict::f16_bits_to_f32(w[base + o]);
                    s += (wv as f64).abs();
                    let q = (wv / scales[o]).round() as i32;
                    let q_clamped = q.clamp(-128, 127);
                    if q_clamped == 0 {
                        zero_count[i] += 1;
                    }
                }
                sum_abs_w[i] = s;
            }
            "f16"
        }
        WeightStorage::I8 { weights, scales } => {
            for i in 0..in_dim {
                let base = i * out_dim;
                let mut s = 0.0_f64;
                for o in 0..out_dim {
                    let wq = weights[base + o];
                    if wq == 0 {
                        zero_count[i] += 1;
                    }
                    let wv = wq as f32 * scales[o];
                    s += (wv as f64).abs();
                }
                sum_abs_w[i] = s;
            }
            "i8"
        }
    };

    LayerStats {
        in_dim,
        out_dim,
        dtype,
        zero_count,
        sum_abs_w,
    }
}

/// Determine the per-channel feature stride from the bake's n_inputs and
/// the layout family. Returns (per_ch, family_name) where per_ch is the
/// features-per-channel count for this bake. Bakes that don't match a
/// known family return per_ch=0 and family="custom".
fn detect_layout(n_inputs: usize) -> (usize, &'static str) {
    let chs = N_CHANNELS * N_SCALES; // 12
    if n_inputs == chs * PER_CH_BASIC {
        (PER_CH_BASIC, "basic-156")
    } else if n_inputs == chs * PER_CH_WITH_PEAKS {
        (PER_CH_WITH_PEAKS, "with-peaks-228")
    } else if n_inputs == chs * (PER_CH_WITH_PEAKS + PER_CH_IW) {
        // 19+6=25-per-ch interpreted as peaks+iw: also 300 features.
        // Could collide with extended (also 25*12=300). Same total; treat
        // as extended for block labeling unless we know otherwise.
        (PER_CH_EXTENDED, "extended-300")
    } else if n_inputs == chs * (PER_CH_EXTENDED + PER_CH_IW) {
        // 25+6=31-per-ch = 372.
        (PER_CH_EXTENDED + PER_CH_IW, "extended-iw-372")
    } else {
        (0, "custom")
    }
}

/// Label feature `i` with `s<scale>.<channel>.<block>:<offset>` for known
/// layouts; falls back to `fNNN` for unknown layouts. Scale = scale index
/// (0..N_SCALES), channel ∈ {X,Y,B}, block = basic|peaks|masked|iw.
fn feature_label(i: usize, n_inputs: usize) -> String {
    let (per_ch, _family) = detect_layout(n_inputs);
    if per_ch == 0 {
        return format!("f{i}");
    }
    // Vector layout matches zensim::metric (see scale_indexing tests):
    //   feature = scale * (3 * per_ch) + channel * per_ch + offset
    let per_scale = 3 * per_ch;
    let scale = i / per_scale;
    let rem = i % per_scale;
    let channel = rem / per_ch;
    let offset = rem % per_ch;
    let ch_name = match channel {
        0 => "X",
        1 => "Y",
        2 => "B",
        _ => "?",
    };
    let block = block_label(offset, per_ch);
    format!("s{scale}.{ch_name}.{block}")
}

fn block_label(offset: usize, per_ch: usize) -> String {
    if offset < PER_CH_BASIC {
        format!("basic{offset}")
    } else if offset < PER_CH_WITH_PEAKS {
        format!("peaks{}", offset - PER_CH_BASIC)
    } else if offset < PER_CH_EXTENDED {
        format!("masked{}", offset - PER_CH_WITH_PEAKS)
    } else if offset < per_ch {
        // Layouts beyond the extended 25-per-ch slot are IW pool.
        format!("iw{}", offset - PER_CH_EXTENDED)
    } else {
        format!("off{offset}")
    }
}

/// Classify which top-level block the feature offset falls into.
fn block_name(offset: usize, per_ch: usize) -> &'static str {
    if offset < PER_CH_BASIC {
        "basic"
    } else if offset < PER_CH_WITH_PEAKS {
        "peaks"
    } else if offset < PER_CH_EXTENDED {
        "masked"
    } else if offset < per_ch {
        "iw"
    } else {
        "unknown"
    }
}

#[derive(Debug)]
struct BakeRecord {
    name: String,
    sha8: String,
    file_size: u64,
    n_inputs: usize,
    layer_in_dim: usize,
    layer_out_dim: usize,
    dtype: &'static str,
    family: &'static str,
    total_features: usize,
    zeroed_features: usize,
    mostly_zeroed: usize,
    l0_sum_abs: f64,
    l0_max_abs: f64,
}

#[derive(Debug, Clone)]
struct FeatureRow {
    bake_name: String,
    feature_idx: usize,
    label: String,
    block: &'static str,
    zero_count: usize,
    out_dim: usize,
    zero_fraction: f64,
    mostly_zeroed: bool,
    fully_zeroed: bool,
    sum_abs_w: f64,
    l1_share: f64,
    importance: f64,
}

fn process_bake(path: &Path) -> Result<(BakeRecord, Vec<FeatureRow>), Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    let file_size = bytes.len() as u64;
    let sha8 = sha256_prefix(&bytes);

    let ver = znpr_version(&bytes).ok_or("bad magic")?;
    if ver != 3 {
        return Err(format!("non-v3 bake: version={ver}").into());
    }

    let model = Model::from_bytes(&bytes)?;
    let n_inputs = model.n_inputs();
    let l_idx = pick_l0_index(&model).ok_or("no layers")?;
    let layer = model.layer(l_idx);
    let stats = compute_layer_stats(&layer);
    let scaler_scale = model.scaler_scale().to_vec();

    let total_l1: f64 = stats.sum_abs_w.iter().sum();
    let l0_max_abs: f64 = stats.sum_abs_w.iter().cloned().fold(0.0_f64, f64::max);

    let (per_ch, family) = detect_layout(n_inputs);

    let bake_name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    let mut feature_rows = Vec::with_capacity(stats.in_dim);
    let mut zeroed_features = 0usize;
    let mut mostly_zeroed = 0usize;

    for i in 0..stats.in_dim {
        let zero_fraction = stats.zero_count[i] as f64 / stats.out_dim as f64;
        let fully = stats.zero_count[i] == stats.out_dim;
        let mostly = zero_fraction >= 0.5;
        if fully {
            zeroed_features += 1;
        }
        if mostly {
            mostly_zeroed += 1;
        }
        let l1_share = if total_l1 > 0.0 {
            stats.sum_abs_w[i] / total_l1
        } else {
            0.0
        };
        let scaler = if i < scaler_scale.len() {
            scaler_scale[i] as f64
        } else {
            1.0
        };
        let importance = scaler * stats.sum_abs_w[i];

        let label = feature_label(i, n_inputs);
        let block = if per_ch > 0 {
            let per_scale = 3 * per_ch;
            let offset = (i % per_scale) % per_ch;
            block_name(offset, per_ch)
        } else {
            "custom"
        };

        feature_rows.push(FeatureRow {
            bake_name: bake_name.clone(),
            feature_idx: i,
            label,
            block,
            zero_count: stats.zero_count[i],
            out_dim: stats.out_dim,
            zero_fraction,
            mostly_zeroed: mostly,
            fully_zeroed: fully,
            sum_abs_w: stats.sum_abs_w[i],
            l1_share,
            importance,
        });
    }

    let record = BakeRecord {
        name: bake_name,
        sha8,
        file_size,
        n_inputs,
        layer_in_dim: stats.in_dim,
        layer_out_dim: stats.out_dim,
        dtype: stats.dtype,
        family,
        total_features: stats.in_dim,
        zeroed_features,
        mostly_zeroed,
        l0_sum_abs: total_l1,
        l0_max_abs,
    };
    Ok((record, feature_rows))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let weights_dir = PathBuf::from(format!("{REPO_ROOT}/zensim/weights"));
    fs::create_dir_all(OUT_DIR)?;

    let bakes = discover_bakes(&weights_dir);
    eprintln!(
        "found {} .bin files under {}",
        bakes.len(),
        weights_dir.display()
    );

    let mut records: Vec<BakeRecord> = Vec::new();
    let mut all_features: Vec<FeatureRow> = Vec::new();
    let mut skipped: Vec<(String, String)> = Vec::new();

    for path in &bakes {
        match process_bake(path) {
            Ok((r, feats)) => {
                eprintln!(
                    "  OK   {} ({} {}, {} feats, {} fully-zeroed, {} mostly)",
                    r.name, r.dtype, r.family, r.total_features, r.zeroed_features, r.mostly_zeroed
                );
                records.push(r);
                all_features.extend(feats);
            }
            Err(e) => {
                let name = path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                eprintln!("  SKIP {}: {}", name, e);
                skipped.push((name, e.to_string()));
            }
        }
    }

    // ---- Write per-bake TSV ----
    let per_bake_path = format!("{OUT_DIR}/per_bake.tsv");
    {
        let mut f = fs::File::create(&per_bake_path)?;
        writeln!(
            f,
            "bake_name\tn_inputs\tdtype\tfamily\ttotal_features\tzeroed_features\tmostly_zeroed\tl0_sum_abs\tl0_max_abs\tsha256_8\tfile_size"
        )?;
        for r in &records {
            writeln!(
                f,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{}\t{}",
                r.name,
                r.n_inputs,
                r.dtype,
                r.family,
                r.total_features,
                r.zeroed_features,
                r.mostly_zeroed,
                r.l0_sum_abs,
                r.l0_max_abs,
                r.sha8,
                r.file_size,
            )?;
        }
    }
    eprintln!("wrote {per_bake_path}");

    // ---- Write per-feature TSV ----
    let per_feat_path = format!("{OUT_DIR}/per_feature.tsv");
    {
        let mut f = fs::File::create(&per_feat_path)?;
        writeln!(
            f,
            "bake_name\tfeature_idx\tlabel\tblock\tzero_count\tout_dim\tzero_fraction\tmostly_zeroed\tfully_zeroed\tsum_abs_w\tl1_share\tl1_share_pct\timportance"
        )?;
        for row in &all_features {
            writeln!(
                f,
                "{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{}\t{}\t{:.6}\t{:.6}\t{:.4}\t{:.6}",
                row.bake_name,
                row.feature_idx,
                row.label,
                row.block,
                row.zero_count,
                row.out_dim,
                row.zero_fraction,
                row.mostly_zeroed as u8,
                row.fully_zeroed as u8,
                row.sum_abs_w,
                row.l1_share,
                row.l1_share * 100.0,
                row.importance,
            )?;
        }
    }
    eprintln!("wrote {per_feat_path}");

    // ---- Compute cross-bake aggregates ----

    // For "consistently zeroed / survived" we want only bakes with the same
    // n_inputs to be directly comparable. Group features by (n_inputs,
    // feature_idx) to keep buckets sane.
    #[derive(Default)]
    struct AggFeat {
        n_seen: usize,
        n_fully_zeroed: usize,
        n_mostly_zeroed: usize,
        sum_zero_fraction: f64,
        sum_importance: f64,
        label: String,
        block: &'static str,
    }

    let mut by_bake_n_inputs: HashMap<String, usize> = HashMap::new();
    for r in &records {
        by_bake_n_inputs.insert(r.name.clone(), r.n_inputs);
    }

    let mut agg: BTreeMap<(usize, usize), AggFeat> = BTreeMap::new();
    for row in &all_features {
        let n_in = *by_bake_n_inputs.get(&row.bake_name).unwrap_or(&0);
        let entry = agg.entry((n_in, row.feature_idx)).or_default();
        entry.n_seen += 1;
        if row.fully_zeroed {
            entry.n_fully_zeroed += 1;
        }
        if row.mostly_zeroed {
            entry.n_mostly_zeroed += 1;
        }
        entry.sum_zero_fraction += row.zero_fraction;
        entry.sum_importance += row.importance;
        entry.label = row.label.clone();
        entry.block = row.block;
    }

    // Cross-bake aggregated ranking
    #[derive(Debug, Clone)]
    struct AggRow {
        n_inputs: usize,
        feature_idx: usize,
        label: String,
        block: &'static str,
        n_seen: usize,
        n_fully_zeroed: usize,
        n_mostly_zeroed: usize,
        mean_zero_fraction: f64,
        mean_importance: f64,
    }
    let agg_rows: Vec<AggRow> = agg
        .iter()
        .map(|((n_in, idx), a)| AggRow {
            n_inputs: *n_in,
            feature_idx: *idx,
            label: a.label.clone(),
            block: a.block,
            n_seen: a.n_seen,
            n_fully_zeroed: a.n_fully_zeroed,
            n_mostly_zeroed: a.n_mostly_zeroed,
            mean_zero_fraction: a.sum_zero_fraction / a.n_seen as f64,
            mean_importance: a.sum_importance / a.n_seen as f64,
        })
        .collect();

    // ---- Per-block aggregate zero-fraction across bakes ----
    let mut block_agg: BTreeMap<&'static str, (f64, usize)> = BTreeMap::new();
    for row in &all_features {
        let entry = block_agg.entry(row.block).or_insert((0.0, 0));
        entry.0 += row.zero_fraction;
        entry.1 += 1;
    }

    // ---- Write SUMMARY.md ----
    let summary_path = format!("{OUT_DIR}/SUMMARY.md");
    let mut s = String::new();
    s.push_str("# Cross-bake L0 i8-quantization zero-out census\n\n");
    s.push_str("Date: 2026-05-25\n\n");
    s.push_str(&format!(
        "Scanned `{REPO_ROOT}/zensim/weights/` recursively. {} bakes loaded, {} skipped (non-v3 or parse error).\n\n",
        records.len(),
        skipped.len()
    ));
    if !skipped.is_empty() {
        s.push_str("## Skipped\n\n");
        for (n, e) in &skipped {
            s.push_str(&format!("- `{n}` — {e}\n"));
        }
        s.push('\n');
    }

    s.push_str("## Method\n\n");
    s.push_str(
        "For each ZNPR v3 bake, the L0 layer (`in_dim == n_inputs`) is loaded. \
For I8 bakes the per-feature zero count is the actual count of `weights[i, o] == 0` over \
the `out_dim` output columns. For F32 / F16 bakes the **same** scheme \
`scale[o] = max_i |W[i, o]| / 127.0; q = round(W[i, o] / scale[o]).clamp(-128, 127)` is \
simulated on the f32 / f16 weights so f32 / f16 / i8 columns are comparable. \
A feature is **fully_zeroed** when zero_fraction == 1.0, **mostly_zeroed** when ≥ 0.5.\n\n",
    );

    s.push_str("## Headline numbers\n\n");
    let total_features_seen: usize = records.iter().map(|r| r.total_features).sum();
    let total_full_zero: usize = records.iter().map(|r| r.zeroed_features).sum();
    let total_mostly: usize = records.iter().map(|r| r.mostly_zeroed).sum();
    s.push_str(&format!(
        "- {} bakes × variable n_inputs = {} feature×bake observations.\n",
        records.len(),
        total_features_seen
    ));
    s.push_str(&format!(
        "- {} feature×bake observations are **fully zeroed** at L0 ({:.2}%).\n",
        total_full_zero,
        100.0 * total_full_zero as f64 / total_features_seen.max(1) as f64
    ));
    s.push_str(&format!(
        "- {} feature×bake observations are **mostly zeroed** (≥ 50% of out_dim columns) ({:.2}%).\n\n",
        total_mostly,
        100.0 * total_mostly as f64 / total_features_seen.max(1) as f64
    ));

    s.push_str("## Per-bake totals\n\n");
    s.push_str(
        "| Bake | n_in | L0 dims | dtype | family | fully_zeroed | mostly_zeroed | L1_sum |\n",
    );
    s.push_str("|---|---:|---|---|---|---:|---:|---:|\n");
    for r in &records {
        s.push_str(&format!(
            "| `{}` | {} | {}×{} | {} | {} | {} | {} | {:.2} |\n",
            r.name,
            r.n_inputs,
            r.layer_in_dim,
            r.layer_out_dim,
            r.dtype,
            r.family,
            r.zeroed_features,
            r.mostly_zeroed,
            r.l0_sum_abs,
        ));
    }

    s.push_str("\n## Top-30 consistently zeroed features (across bakes of matching n_inputs)\n\n");
    s.push_str(
        "Each row is keyed by (n_inputs, feature_idx); a feature ranks higher when it's \
fully-zeroed in more of the bakes that share its n_inputs.\n\n",
    );
    s.push_str("| n_in | f | label | block | n_fully/n_seen | n_mostly/n_seen | mean_zero_frac | mean_imp |\n");
    s.push_str("|---:|---:|---|---|---|---|---:|---:|\n");
    let mut by_zeroed = agg_rows.clone();
    by_zeroed.sort_by(|a, b| {
        b.n_fully_zeroed
            .cmp(&a.n_fully_zeroed)
            .then(
                b.mean_zero_fraction
                    .partial_cmp(&a.mean_zero_fraction)
                    .unwrap(),
            )
            .then(a.mean_importance.partial_cmp(&b.mean_importance).unwrap())
    });
    for row in by_zeroed.iter().take(30) {
        s.push_str(&format!(
            "| {} | {} | `{}` | {} | {}/{} | {}/{} | {:.3} | {:.4} |\n",
            row.n_inputs,
            row.feature_idx,
            row.label,
            row.block,
            row.n_fully_zeroed,
            row.n_seen,
            row.n_mostly_zeroed,
            row.n_seen,
            row.mean_zero_fraction,
            row.mean_importance,
        ));
    }

    s.push_str("\n## Top-30 consistently survived features (zero in no bakes)\n\n");
    s.push_str(
        "Features that are NEVER fully-zeroed across every bake of their n_inputs cohort, \
ranked by mean importance (descending).\n\n",
    );
    s.push_str("| n_in | f | label | block | n_seen | mean_zero_frac | mean_imp |\n");
    s.push_str("|---:|---:|---|---|---:|---:|---:|\n");
    let mut survivors: Vec<&AggRow> = agg_rows.iter().filter(|r| r.n_fully_zeroed == 0).collect();
    survivors.sort_by(|a, b| b.mean_importance.partial_cmp(&a.mean_importance).unwrap());
    for row in survivors.iter().take(30) {
        s.push_str(&format!(
            "| {} | {} | `{}` | {} | {} | {:.3} | {:.4} |\n",
            row.n_inputs,
            row.feature_idx,
            row.label,
            row.block,
            row.n_seen,
            row.mean_zero_fraction,
            row.mean_importance,
        ));
    }

    s.push_str("\n## Per-block mean zero-fraction across all bakes\n\n");
    s.push_str("| block | n observations | mean zero_fraction |\n");
    s.push_str("|---|---:|---:|\n");
    for (block, (sum, n)) in &block_agg {
        s.push_str(&format!(
            "| {} | {} | {:.3} |\n",
            block,
            n,
            if *n > 0 { sum / *n as f64 } else { 0.0 }
        ));
    }

    // Sort agg_rows by total dropped features in their bake for callouts
    // Also identify per-bake interesting callouts.
    s.push_str("\n## Per-bake callouts\n\n");
    let mut by_drop = records.iter().collect::<Vec<_>>();
    by_drop.sort_by(|a, b| {
        let af = a.zeroed_features as f64 / a.total_features as f64;
        let bf = b.zeroed_features as f64 / b.total_features as f64;
        bf.partial_cmp(&af).unwrap()
    });
    s.push_str("### Bakes with the largest fully-zeroed fraction at L0\n\n");
    s.push_str("| Bake | dtype | fully_zeroed / total | frac |\n");
    s.push_str("|---|---|---|---:|\n");
    for r in by_drop.iter().take(5) {
        let frac = r.zeroed_features as f64 / r.total_features.max(1) as f64;
        s.push_str(&format!(
            "| `{}` | {} | {} / {} | {:.3} |\n",
            r.name, r.dtype, r.zeroed_features, r.total_features, frac
        ));
    }
    s.push_str("\n### Bakes with the smallest fully-zeroed fraction at L0\n\n");
    s.push_str("| Bake | dtype | fully_zeroed / total | frac |\n");
    s.push_str("|---|---|---|---:|\n");
    for r in by_drop.iter().rev().take(5) {
        let frac = r.zeroed_features as f64 / r.total_features.max(1) as f64;
        s.push_str(&format!(
            "| `{}` | {} | {} / {} | {:.3} |\n",
            r.name, r.dtype, r.zeroed_features, r.total_features, frac
        ));
    }

    // Add a specific note about v_tuner_v11 — the v0.3 ship — and its
    // "L0 dominator" feature f129.
    s.push_str("\n## Spot-check: v_tuner_v11_2026-05-24 (v0.3 ship)\n\n");
    if let Some(v11) = records
        .iter()
        .find(|r| r.name == "v_tuner_v11_2026-05-24.bin")
    {
        s.push_str(&format!(
            "- dtype: `{}`, n_inputs: {}, L0: {}×{}\n",
            v11.dtype, v11.n_inputs, v11.layer_in_dim, v11.layer_out_dim
        ));
        s.push_str(&format!(
            "- L0 fully-zeroed: **{}** / {} features ({:.2}%)\n",
            v11.zeroed_features,
            v11.total_features,
            100.0 * v11.zeroed_features as f64 / v11.total_features.max(1) as f64
        ));
        // pull out f129's row
        if let Some(f129) = all_features
            .iter()
            .find(|r| r.bake_name == v11.name && r.feature_idx == 129)
        {
            s.push_str(&format!(
                "- feature 129 (`{}`, block `{}`): zero_count={}/{} ({:.3}), L1 share {:.3}%, importance {:.4}\n",
                f129.label,
                f129.block,
                f129.zero_count,
                f129.out_dim,
                f129.zero_fraction,
                f129.l1_share * 100.0,
                f129.importance,
            ));
        }
        // Across all bakes of the same n_inputs cohort, how many fully-zero f129?
        let cohort_n = v11.n_inputs;
        let f129_obs: Vec<&FeatureRow> = all_features
            .iter()
            .filter(|r| {
                r.feature_idx == 129 && by_bake_n_inputs.get(&r.bake_name) == Some(&cohort_n)
            })
            .collect();
        let f129_zeroed = f129_obs.iter().filter(|r| r.fully_zeroed).count();
        s.push_str(&format!(
            "- across {} bakes with n_inputs={}, feature 129 is fully-zeroed in **{} of {}** ({:.1}%).\n",
            f129_obs.len(),
            cohort_n,
            f129_zeroed,
            f129_obs.len(),
            100.0 * f129_zeroed as f64 / f129_obs.len().max(1) as f64
        ));
    } else {
        s.push_str("v_tuner_v11_2026-05-24.bin not found.\n");
    }

    fs::write(&summary_path, s)?;
    eprintln!("wrote {summary_path}");

    Ok(())
}
