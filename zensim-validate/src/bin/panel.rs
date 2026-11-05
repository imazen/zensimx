//! `panel` — the canonical IQA correlation-statistics entry point.
//!
//! This is THE tool for computing the full Mohammadi 2025 statistical
//! panel (SROCC + PLCC + KROCC + OR + PWRC + Z-RMSE, plus the 4-parameter
//! logistic rescale that PLCC / Z-RMSE depend on) on an arbitrary table
//! of `(predicted, target)` pairs — i.e. the case that does NOT involve
//! a bake or a canonical corpus (use `bake_verdict` for that).
//!
//! ## Why this binary exists
//!
//! Across the zen workspace there were ~14 scattered Python
//! reimplementations of SROCC / PLCC / KROCC / OR / PWRC / Z-RMSE and
//! the 4-param logistic fit (see the dedup ledger
//! `benchmarks/dedup_VERIFIED_synthesis_2026-05-26.md` Tier-1 #2 and
//! `benchmarks/sweep_training_script_dedup_2026-05-26.md` Class 2). Each
//! hand-rolled `spearman` carried its own tie-handling and NaN-drop
//! policy; a metric verdict is only as trustworthy as its stat impl, and
//! CLAUDE.md mandates the *full* Mohammadi panel everywhere. This binary
//! collapses all of those onto the single canonical Rust home,
//! `zensim_validate::panel`, so every consumer computes bit-identical
//! stats.
//!
//! ## Zero new stat math
//!
//! Every number this tool prints comes from a direct call into
//! `panel.rs`. No statistic is reimplemented here. The wrapped functions
//! (with their definitions, all in `zensim-validate/src/panel.rs`):
//!
//! | Stat / op             | panel.rs fn                  | file:line |
//! |-----------------------|------------------------------|-----------|
//! | full 6-stat panel     | `compute_panel`              | panel.rs:656 |
//! | Spearman SROCC        | `spearman` (via compute_panel)| panel.rs:50 |
//! | Pearson PLCC          | `pearson` (via compute_panel) | panel.rs:72 |
//! | Kendall KROCC         | `kendall_tau` (via compute_panel) | panel.rs:93 |
//! | Outlier ratio OR      | `outlier_ratio` (via compute_panel) | panel.rs:129 |
//! | PWRC                  | `pwrc` (via compute_panel)    | panel.rs:163 |
//! | global Z-RMSE         | `z_rmse` (via compute_panel)  | panel.rs:193 |
//! | per-sample Z-RMSE     | `z_rmse_per_sample`          | panel.rs:234 |
//! | 4-param logistic      | `rescale_logistic`           | panel.rs:458 |
//! | rank vector           | `ranks`                      | panel.rs:30 |
//!
//! `compute_panel` (panel.rs:656) internally calls `spearman.abs()`,
//! `kendall_tau.abs()`, `pwrc.abs()`, `outlier_ratio`, then
//! `rescale_logistic` → `pearson.abs()` for PLCC and `z_rmse` on the
//! rescaled scores for the global Z-RMSE. The `--sigma` column, when
//! present, additionally feeds `z_rmse_per_sample` (panel.rs:234) on the
//! logistic-rescaled scores — matching the per-stimulus σ-normalized
//! Z-RMSE of Mohammadi 2025 Eq. 6.
//!
//! ## Input
//!
//! A TSV (`--input foo.tsv`) or Parquet (`--input foo.parquet`) with
//! named columns:
//! - `predicted` (required) — the metric / model output.
//! - `target`    (required) — the human MOS / reference quality.
//! - `sigma`     (optional) — per-stimulus observer σ (enables the
//!   per-sample Z-RMSE; the global Z-RMSE is always reported).
//! - `band`      (optional) — a grouping key. When present, the panel is
//!   reported per distinct band value (sorted) AND in aggregate.
//!
//! TSV: tab-separated, first row is the header, column order is free
//! (located by name). Rows with a non-finite `predicted` or `target`
//! are dropped (and counted in the report) — matching the NaN-drop
//! policy of the Python reference (`scipy.stats.spearmanr` drops NaN
//! pairwise; we drop the whole row, which is the conservative choice
//! and what every py reimpl in the ledger does via `np.isfinite` masks).
//!
//! ## Output
//!
//! Text panel by default; `--json` for a machine-readable object. The
//! JSON is emitted by hand (no serde dep in the workspace) with full
//! `{:.10}` float precision and explicit `null` for NaN.
//!
//! ## Examples
//!
//! ```sh
//! # Panel on a TSV of (predicted, target) pairs
//! panel --input scores.tsv
//!
//! # With per-stimulus sigma (enables per-sample Z-RMSE) + JSON
//! panel --input scores_with_sigma.tsv --json
//!
//! # Per-band breakdown from a parquet
//! panel --input eval.parquet --json > panel.json
//!
//! # Batch mode: N (x, y) vector pairs in, N panel rows out, ONE process
//! panel --batch jobs.tsv --stats srocc > sroccs.tsv
//! ```
//!
//! ## Batch mode (`--batch`, decision-surface audit 2026-07-31 gap 4)
//!
//! Per-call shelling is fine for aggregates but prohibitive inside
//! bootstrap loops (10k resamples = 10k process spawns). `--batch`
//! reads a manifest of many (x, y) vector pairs and emits one stats row
//! per pair **in one process**, so a 10k-resample bootstrap is a single
//! invocation. This is the canonical replacement for every
//! `scipy.stats.spearmanr`-in-a-loop call site (the banned pattern);
//! Python callers use `scripts/lib/zen_stats.panel_batch` /
//! `panel_batch_indexed`.
//!
//! Input (`--batch <FILE|->`, `-` = stdin) is line-oriented, tab-separated:
//!
//! ```text
//! #def NAME<TAB>v1,v2,v3,...            define a named base vector
//! LABEL<TAB>x1,x2,...<TAB>y1,y2,...     explicit (x, y) pair
//! LABEL<TAB>@X:@Y<TAB>i1,i2,...         indexed pair: (X[i], Y[i]) over
//!                                       previously #def'd bases (the SAME
//!                                       index set applies to both — the
//!                                       paired-bootstrap resample shape)
//! LABEL<TAB>@X:@Y<TAB>*                 indexed pair over ALL rows
//! ```
//!
//! Blank lines and `#`-comments (other than `#def`) are skipped. The
//! indexed form exists so a caller resampling the same base vectors 10k
//! times ships ~n integers per job instead of ~2n floats; the caller
//! keeps ownership of the resampling RNG (deterministic seeding stays at
//! the call site, e.g. upiq_panel.py's `np.random.default_rng(20260714)`),
//! and this binary stays RNG-free — batch output is a pure deterministic
//! function of the input bytes.
//!
//! Output: TSV with a header, one row per job in input order. Columns:
//!
//! * `--stats srocc` (the bootstrap fast path):
//!   `label n n_dropped srocc srocc_signed`
//! * `--stats full` (default — everything the aggregate panel emits):
//!   `label n n_dropped srocc srocc_signed plcc plcc_raw krocc or pwrc z_rmse`
//!
//! Zero new stat math, matching this binary's charter: `srocc_signed` is
//! a direct `panel::spearman` call (tie-correct midrank, pre-`.abs()`),
//! `plcc_raw` a direct `panel::pearson` on the raw pair (no logistic
//! rescale — some registered instruments, e.g. `scripts/hdr/upiq_panel.py`,
//! report raw-Pearson |PLCC|), and everything else comes from
//! `compute_panel` exactly as in aggregate mode. Non-finite (x, y) rows
//! are dropped per job and counted in `n_dropped` (same policy as
//! aggregate mode). Floats print as `{:.17e}` (round-trip exact).
//! `--json` is not supported with `--batch` (the TSV is the contract).
//! `--raw-errors` appends `mae_raw` without fitting a score mapping. The legacy
//! `mae` column in `--stats full` remains logistic-rescaled; it cannot measure
//! a target dial's calibration error. Raw errors also work with `--stats srocc`.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use arrow::array::{Array, Float32Array, Float64Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use zensim_validate::panel::{self, PanelStats};

// ----------------------------------------------------------------------
// CLI
// ----------------------------------------------------------------------

#[path = "../scatter_json.rs"]
mod scatter_json;

#[path = "../ladder_step.rs"]
mod ladder_step;
mod panel_e34;
mod panel_evaluate;
mod panel_rank_interval;

struct Args {
    /// Signed D2 correlations with a fixed higher-is-better quality direction.
    signed_quality: bool,
    scatter: bool,
    input: Option<PathBuf>,
    /// Batch mode: a manifest of many (x, y) vector pairs (`-` = stdin).
    /// Mutually exclusive with `--input`. See the module docs.
    batch: Option<PathBuf>,
    /// Batch column set: `full` (default) or `srocc` (bootstrap fast path).
    stats_srocc_only: bool,
    /// Append raw MAE alongside the legacy logistic-rescaled MAE in batch mode.
    raw_errors: bool,
    json: bool,
    /// Override default column names if a caller's table uses different
    /// headers. Defaults: predicted / target / sigma / band.
    col_predicted: String,
    col_target: String,
    col_sigma: String,
    col_band: String,
    /// Pairwise mode: a TSV of weighted forced-choice rows
    /// (`group`, `s_left`, `s_right`, `choice`, optional `weight`) — the
    /// triplet-comparison statistic owned by
    /// `zenstats::forced_choice::agreement` (zenmetrics-stats). `choice` may be
    /// `notsure`, counted half for each side. Mutually exclusive with
    /// `--input` / `--batch`.
    pairwise: Option<PathBuf>,
    /// Optional cluster-bootstrap resample manifest for `--pairwise`:
    /// lines `LABEL<TAB>g1,g2,...` of GROUP indices (`*` = all groups once).
    /// One output row per line. The caller owns the RNG; this binary stays
    /// deterministic, exactly as `--batch` does.
    resample: Option<PathBuf>,
    /// `--per-group`: additionally summarize SROCC computed WITHIN each
    /// `band` value — the canonical `zenstats::per_group_srocc`, i.e. the
    /// exact quantity `bake_verdict` publishes as `rank.<corpus>.per_ref_mean`
    /// / `per_ref_n` / `frac_negative`.
    ///
    /// Why this belongs here and not in a caller: a pooled SROCC conflates
    /// "does the metric order this image's own quality ladder correctly?"
    /// with "does it put different images on a common scale?", and the FIRST
    /// is what a codec target loop consumes. `bake_verdict` can only answer
    /// it for a BAKE against a feature parquet; a reference metric (ssim2,
    /// butteraugli, cvvdp) has only a stored per-pair table, so its
    /// within-image behaviour was structurally unmeasurable — which is why
    /// every peer row on the board carries an empty `per_ref_mean`.
    /// `--per-group` closes that with the same owner call, no new stat math.
    per_group: bool,
    /// Hidden debug flag (used by the parity cross-check
    /// `scripts/verify_panel_parity.py`): instead of the panel, print
    /// the 4-param-logistic-rescaled `predicted` column (one value per
    /// line, full precision) for the ALL group. This lets the Python
    /// reference feed panel.rs's EXACT rescaled scores into its PLCC /
    /// Z-RMSE computation so the logistic-optimizer difference is
    /// removed and the underlying stat math can be gated to 1e-9.
    emit_rescaled: bool,
}

fn print_usage() {
    eprintln!(
        "panel — canonical IQA statistical-panel entry point (Mohammadi 2025)\n\
         \n\
         USAGE:\n\
         \x20\x20panel evaluate --manifest <SUITE.json> --output <NEW_DIRECTORY>\n\
         \x20\x20panel --input <FILE.tsv|FILE.parquet> [--json] [column overrides]\n\
         \x20\x20panel --batch <FILE.tsv|-> [--stats full|srocc]\n\
         \n\
         INPUT COLUMNS (located by name; order free):\n\
         \x20\x20predicted   required   metric / model output\n\
         \x20\x20target      required   human MOS / reference quality\n\
         \x20\x20sigma       optional   per-stimulus observer σ (enables per-sample Z-RMSE)\n\
         \x20\x20band        optional   grouping key (per-band + aggregate panel)\n\
         \n\
         OPTIONS:\n\
         \x20\x20--input <PATH>          input TSV or Parquet (aggregate mode)\n\
         \x20\x20--batch <PATH|->        batch manifest: N (x,y) vector pairs in,\n\
         \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20N stat rows out, ONE process (see docs;\n\
         \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20lines: '#def N<TAB>csv', 'L<TAB>x-csv<TAB>y-csv',\n\
         \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20'L<TAB>@X:@Y<TAB>idx-csv|*')\n\
         \x20\x20--stats <full|srocc>    batch column set (default full)\n\
         \x20\x20--scatter              --input --json: full-population scatter diagnostics\n\
         \x20\x20--raw-errors            append raw MAE in batch mode (no score remapping)\n\
         \x20\x20--json                  emit JSON instead of text (aggregate mode only)\n\
         \x20\x20--col-predicted <NAME>  override the 'predicted' column name\n\
         \x20\x20--col-target <NAME>     override the 'target' column name\n\
         \x20\x20--col-sigma <NAME>      override the 'sigma' column name\n\
         \x20\x20--col-band <NAME>       override the 'band' column name\n\
         \x20\x20--signed-quality        signed quality SROCC/KROCC/fitted PLCC (--input --json)\n\
         \x20\x20--per-group             + within-band SROCC summary (per_group_srocc)\n\
         \x20\x20--pairwise <PATH|->     forced-choice (2AFC / triplet) agreement:\n\
         \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20TSV cols group,s_left,s_right,choice[,weight];\n\
         \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20choice left|right|notsure (not sure = half each)\n\
         \x20\x20--resample <PATH|->     cluster-bootstrap group-index manifest for\n\
         \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20--pairwise ('LABEL<TAB>g-csv' or 'LABEL<TAB>*')\n\
         \n\
         All statistics come from zensim_validate::panel (no stat math is\n\
         reimplemented here). See the module docs for the file:line map.\n\
         For bake-on-canonical-corpus evaluation use `bake_verdict` instead."
    );
}

fn parse_args() -> Result<Args, String> {
    let mut signed_quality = false;
    let mut input: Option<PathBuf> = None;
    let mut batch: Option<PathBuf> = None;
    let mut stats_srocc_only = false;
    let mut raw_errors = false;
    let mut scatter = false;
    let mut json = false;
    let mut col_predicted = "predicted".to_string();
    let mut col_target = "target".to_string();
    let mut col_sigma = "sigma".to_string();
    let mut col_band = "band".to_string();
    let mut per_group = false;
    let mut emit_rescaled = false;
    let mut pairwise: Option<PathBuf> = None;
    let mut resample: Option<PathBuf> = None;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--input" | "-i" => {
                input = Some(PathBuf::from(it.next().ok_or("--input requires a value")?));
            }
            "--batch" => {
                batch = Some(PathBuf::from(it.next().ok_or("--batch requires a value")?));
            }
            "--stats" => {
                let v = it.next().ok_or("--stats requires a value")?;
                stats_srocc_only = match v.as_str() {
                    "srocc" => true,
                    "full" => false,
                    other => return Err(format!("--stats must be 'full' or 'srocc', got {other}")),
                };
            }
            "--signed-quality" => signed_quality = true,
            "--json" => json = true,
            "--scatter" => scatter = true,
            "--raw-errors" => raw_errors = true,
            // Hidden — see Args::emit_rescaled.
            "--emit-rescaled" => emit_rescaled = true,
            "--col-predicted" => {
                col_predicted = it.next().ok_or("--col-predicted requires a value")?
            }
            "--col-target" => col_target = it.next().ok_or("--col-target requires a value")?,
            "--col-sigma" => col_sigma = it.next().ok_or("--col-sigma requires a value")?,
            "--col-band" => col_band = it.next().ok_or("--col-band requires a value")?,
            "--per-group" => per_group = true,
            "--pairwise" => {
                pairwise = Some(PathBuf::from(
                    it.next().ok_or("--pairwise requires a value")?,
                ));
            }
            "--resample" => {
                resample = Some(PathBuf::from(
                    it.next().ok_or("--resample requires a value")?,
                ));
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    if input.is_some() && batch.is_some() {
        return Err("--input and --batch are mutually exclusive".to_string());
    }
    if pairwise.is_some() && (input.is_some() || batch.is_some()) {
        return Err("--pairwise is mutually exclusive with --input / --batch".to_string());
    }
    if resample.is_some() && pairwise.is_none() {
        return Err("--resample requires --pairwise".to_string());
    }
    if pairwise.is_some() && json {
        return Err(
            "--json is not supported with --pairwise (the TSV is the contract)".to_string(),
        );
    }
    if batch.is_some() && json {
        return Err("--json is not supported with --batch (the TSV is the contract)".to_string());
    }
    if input.is_none() && batch.is_none() && pairwise.is_none() {
        return Err("--input, --batch or --pairwise is required".to_string());
    }

    if raw_errors && batch.is_none() {
        return Err("--raw-errors requires --batch".to_string());
    }
    if scatter && (input.is_none() || !json || per_group || emit_rescaled) {
        return Err("--scatter requires --input --json without other modes".into());
    }
    if signed_quality
        && (scatter
            || per_group
            || emit_rescaled
            || raw_errors
            || (batch.is_some() && stats_srocc_only)
            || (batch.is_none() && (input.is_none() || !json)))
    {
        return Err(
            "--signed-quality requires --input --json or --batch --stats full, without other modes"
                .into(),
        );
    }
    Ok(Args {
        signed_quality,
        scatter,
        input,
        batch,
        stats_srocc_only,
        raw_errors,
        json,
        col_predicted,
        col_target,
        col_sigma,
        col_band,
        per_group,
        emit_rescaled,
        pairwise,
        resample,
    })
}

// ----------------------------------------------------------------------
// Loaded columns
// ----------------------------------------------------------------------

/// Raw columns read from the input file. `sigma` / `band` are `None`
/// when the column is absent. All vectors are the same length (file row
/// count); row filtering happens later so band membership stays aligned.
struct Columns {
    predicted: Vec<f64>,
    target: Vec<f64>,
    sigma: Option<Vec<f64>>,
    band: Option<Vec<String>>,
}

fn load_columns(args: &Args) -> Result<Columns, String> {
    let path = args
        .input
        .as_ref()
        .expect("aggregate mode requires --input");
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "parquet" => load_parquet_columns(args),
        "tsv" | "csv" | "txt" | "" => load_tsv_columns(args),
        other => Err(format!(
            "unsupported input extension {other:?} (expected .tsv / .parquet)"
        )),
    }
}

fn load_tsv_columns(args: &Args) -> Result<Columns, String> {
    let path: &Path = args
        .input
        .as_ref()
        .expect("aggregate mode requires --input");
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path:?}: {e}"))?;
    let mut lines = text.lines();
    let header = lines
        .next()
        .ok_or_else(|| format!("{path:?}: empty file (no header)"))?;
    // Tab-separated by default; fall back to comma for .csv.
    let sep = if path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("csv"))
        .unwrap_or(false)
    {
        ','
    } else {
        '\t'
    };
    let cols: Vec<&str> = header.split(sep).map(|c| c.trim()).collect();
    let find = |name: &str| cols.iter().position(|c| *c == name);

    let pi = find(&args.col_predicted)
        .ok_or_else(|| format!("{path:?}: missing '{}' column", args.col_predicted))?;
    let ti = find(&args.col_target)
        .ok_or_else(|| format!("{path:?}: missing '{}' column", args.col_target))?;
    let si = find(&args.col_sigma);
    let bi = find(&args.col_band);

    let mut predicted = Vec::new();
    let mut target = Vec::new();
    let mut sigma = if si.is_some() { Some(Vec::new()) } else { None };
    let mut band = if bi.is_some() { Some(Vec::new()) } else { None };

    for (lineno, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(sep).collect();
        let get = |idx: usize| -> Result<&str, String> {
            fields
                .get(idx)
                .map(|s| s.trim())
                .ok_or_else(|| format!("{path:?}:{}: too few columns", lineno + 2))
        };
        let parse_f = |s: &str| -> f64 { s.parse::<f64>().unwrap_or(f64::NAN) };
        predicted.push(parse_f(get(pi)?));
        target.push(parse_f(get(ti)?));
        if let (Some(idx), Some(v)) = (si, sigma.as_mut()) {
            v.push(parse_f(get(idx)?));
        }
        if let (Some(idx), Some(v)) = (bi, band.as_mut()) {
            v.push(get(idx)?.to_string());
        }
    }

    Ok(Columns {
        predicted,
        target,
        sigma,
        band,
    })
}

fn load_parquet_columns(args: &Args) -> Result<Columns, String> {
    let path: &Path = args
        .input
        .as_ref()
        .expect("aggregate mode requires --input");
    let file = File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("{path:?}: parquet open: {e}"))?;
    let schema = builder.schema().clone();
    let names: Vec<String> = schema
        .fields()
        .iter()
        .map(|f| f.name().to_string())
        .collect();
    let find = |name: &str| names.iter().position(|c| c == name);

    let pi = find(&args.col_predicted)
        .ok_or_else(|| format!("{path:?}: missing '{}' column", args.col_predicted))?;
    let ti = find(&args.col_target)
        .ok_or_else(|| format!("{path:?}: missing '{}' column", args.col_target))?;
    let si = find(&args.col_sigma);
    let bi = find(&args.col_band);

    let reader = builder
        .with_batch_size(16384)
        .build()
        .map_err(|e| format!("{path:?}: parquet build reader: {e}"))?;

    let mut predicted = Vec::new();
    let mut target = Vec::new();
    let mut sigma = if si.is_some() { Some(Vec::new()) } else { None };
    let mut band = if bi.is_some() { Some(Vec::new()) } else { None };

    // Pull a numeric column as f64 (Float32/Float64/Int32/Int64), NaN for null.
    fn numeric_col(col: &dyn Array, name: &str) -> Result<Vec<f64>, String> {
        let n = col.len();
        match col.data_type() {
            DataType::Float64 => {
                let a = col.as_any().downcast_ref::<Float64Array>().unwrap();
                Ok((0..n)
                    .map(|i| if a.is_null(i) { f64::NAN } else { a.value(i) })
                    .collect())
            }
            DataType::Float32 => {
                let a = col.as_any().downcast_ref::<Float32Array>().unwrap();
                Ok((0..n)
                    .map(|i| {
                        if a.is_null(i) {
                            f64::NAN
                        } else {
                            a.value(i) as f64
                        }
                    })
                    .collect())
            }
            DataType::Int64 => {
                let a = col.as_any().downcast_ref::<Int64Array>().unwrap();
                Ok((0..n)
                    .map(|i| {
                        if a.is_null(i) {
                            f64::NAN
                        } else {
                            a.value(i) as f64
                        }
                    })
                    .collect())
            }
            DataType::Int32 => {
                let a = col.as_any().downcast_ref::<Int32Array>().unwrap();
                Ok((0..n)
                    .map(|i| {
                        if a.is_null(i) {
                            f64::NAN
                        } else {
                            a.value(i) as f64
                        }
                    })
                    .collect())
            }
            other => Err(format!(
                "column {name:?} has unsupported numeric dtype {other:?}"
            )),
        }
    }

    fn string_col(col: &dyn Array, name: &str) -> Result<Vec<String>, String> {
        let n = col.len();
        match col.data_type() {
            DataType::Utf8 => {
                let a = col.as_any().downcast_ref::<StringArray>().unwrap();
                Ok((0..n)
                    .map(|i| {
                        if a.is_null(i) {
                            String::new()
                        } else {
                            a.value(i).to_string()
                        }
                    })
                    .collect())
            }
            // A numeric band column is fine — stringify it.
            DataType::Float64 | DataType::Float32 | DataType::Int64 | DataType::Int32 => {
                Ok(numeric_col(col, name)?
                    .into_iter()
                    .map(|v| format!("{v}"))
                    .collect())
            }
            other => Err(format!(
                "band column {name:?} has unsupported dtype {other:?}"
            )),
        }
    }

    for batch_res in reader {
        let batch = batch_res.map_err(|e| format!("{path:?}: read batch: {e}"))?;
        predicted.extend(numeric_col(batch.column(pi), &args.col_predicted)?);
        target.extend(numeric_col(batch.column(ti), &args.col_target)?);
        if let (Some(idx), Some(v)) = (si, sigma.as_mut()) {
            v.extend(numeric_col(batch.column(idx), &args.col_sigma)?);
        }
        if let (Some(idx), Some(v)) = (bi, band.as_mut()) {
            v.extend(string_col(batch.column(idx), &args.col_band)?);
        }
    }

    Ok(Columns {
        predicted,
        target,
        sigma,
        band,
    })
}

// ----------------------------------------------------------------------
// Panel computation on a (filtered) slice
// ----------------------------------------------------------------------

/// One reported group: the aggregate or a single band.
struct GroupReport {
    label: String,
    panel: PanelStats,
    /// Per-sample σ-normalized Z-RMSE (Mohammadi Eq. 6) — only when a
    /// `sigma` column was supplied. Computed on the logistic-rescaled
    /// scores (same rescale `compute_panel` uses), so it lines up with
    /// the global Z-RMSE in `panel.z_rmse`.
    z_rmse_per_sample: Option<f64>,
    /// Rows kept after dropping non-finite (predicted, target) pairs.
    n_kept: usize,
    /// Rows dropped because predicted or target was non-finite.
    n_dropped: usize,
}

/// Filter to finite (predicted, target) rows, then run the panel. When
/// `sigma` is present, also computes the per-sample Z-RMSE on the
/// logistic-rescaled scores via `panel::rescale_logistic` +
/// `panel::z_rmse_per_sample`.
fn report_group(
    label: &str,
    predicted: &[f64],
    target: &[f64],
    sigma: Option<&[f64]>,
) -> GroupReport {
    let n = predicted.len().min(target.len());
    let mut p = Vec::with_capacity(n);
    let mut t = Vec::with_capacity(n);
    let mut s: Option<Vec<f64>> = sigma.map(|_| Vec::with_capacity(n));
    let mut n_dropped = 0usize;
    for i in 0..n {
        if predicted[i].is_finite() && target[i].is_finite() {
            p.push(predicted[i]);
            t.push(target[i]);
            if let (Some(sig), Some(out)) = (sigma, s.as_mut()) {
                out.push(sig[i]);
            }
        } else {
            n_dropped += 1;
        }
    }

    // compute_panel (panel.rs:656) — the full 6-stat Mohammadi panel.
    let panel = panel::compute_panel(&p, &t);

    // Per-sample Z-RMSE on the logistic-rescaled scores. We mirror what
    // compute_panel does internally for the global Z-RMSE: rescale the
    // predicted scores through the 4-param logistic (panel.rs:458), then
    // feed (rescaled, target, sigma) to z_rmse_per_sample (panel.rs:234).
    let z_rmse_per_sample = s.as_ref().map(|sig| {
        let rescaled = panel::rescale_logistic(&p, &t);
        panel::z_rmse_per_sample(&rescaled, &t, sig)
    });

    GroupReport {
        label: label.to_string(),
        panel,
        z_rmse_per_sample,
        n_kept: p.len(),
        n_dropped,
    }
}

fn build_reports(cols: &Columns) -> Vec<GroupReport> {
    let mut out = Vec::new();
    // Aggregate panel over all rows.
    out.push(report_group(
        "ALL",
        &cols.predicted,
        &cols.target,
        cols.sigma.as_deref(),
    ));

    // Per-band panels when a band column is present.
    if let Some(bands) = &cols.band {
        // Group row indices by band value, sorted lexicographically for
        // deterministic output.
        let mut by_band: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for (i, b) in bands.iter().enumerate() {
            by_band.entry(b.as_str()).or_default().push(i);
        }
        for (band_val, idxs) in by_band {
            let p: Vec<f64> = idxs.iter().map(|&i| cols.predicted[i]).collect();
            let t: Vec<f64> = idxs.iter().map(|&i| cols.target[i]).collect();
            let s: Option<Vec<f64>> = cols
                .sigma
                .as_ref()
                .map(|sig| idxs.iter().map(|&i| sig[i]).collect());
            out.push(report_group(
                &format!("band={band_val}"),
                &p,
                &t,
                s.as_deref(),
            ));
        }
    }
    out
}

// ----------------------------------------------------------------------
// Batch mode — N (x, y) pairs in, N stat rows out, one process.
// (decision_surface_audit_2026-07-31.md gap 4; zero new stat math —
// every number is a direct `zensim_validate::panel` owner call.)
// ----------------------------------------------------------------------

/// One parsed batch job. `Indexed` references `#def`'d base vectors by
/// slot so 10k resample jobs share storage instead of cloning floats.
enum BatchJob {
    Explicit {
        x: Vec<f64>,
        y: Vec<f64>,
    },
    Indexed {
        x_base: usize,
        y_base: usize,
        /// `None` = all rows (the `*` form).
        idx: Option<Vec<usize>>,
    },
}

struct BatchInput {
    bases: Vec<Vec<f64>>,
    jobs: Vec<(String, BatchJob)>,
}

fn parse_float_csv(s: &str, lineno: usize, what: &str) -> Result<Vec<f64>, String> {
    s.split(',')
        .map(|t| {
            let t = t.trim();
            t.parse::<f64>()
                .map_err(|e| format!("batch line {lineno}: bad {what} float {t:?}: {e}"))
        })
        .collect()
}

fn parse_batch(text: &str) -> Result<BatchInput, String> {
    let mut bases: Vec<Vec<f64>> = Vec::new();
    let mut base_names: BTreeMap<String, usize> = BTreeMap::new();
    let mut jobs: Vec<(String, BatchJob)> = Vec::new();

    for (i, line) in text.lines().enumerate() {
        let lineno = i + 1;
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("#def") {
            let mut parts = rest.trim_start().splitn(2, '\t');
            let name = parts
                .next()
                .filter(|n| !n.is_empty())
                .ok_or_else(|| format!("batch line {lineno}: #def needs NAME<TAB>csv"))?;
            let csv = parts
                .next()
                .ok_or_else(|| format!("batch line {lineno}: #def {name} missing vector"))?;
            let vec = parse_float_csv(csv, lineno, &format!("#def {name}"))?;
            if base_names.insert(name.to_string(), bases.len()).is_some() {
                return Err(format!("batch line {lineno}: duplicate #def {name}"));
            }
            bases.push(vec);
            continue;
        }
        if line.starts_with('#') {
            continue; // plain comment
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 3 {
            return Err(format!(
                "batch line {lineno}: expected LABEL<TAB>x<TAB>y (3 fields), got {}",
                fields.len()
            ));
        }
        let label = fields[0].to_string();
        let job = if let Some(refs) = fields[1].strip_prefix('@') {
            // Indexed form: @X:@Y <TAB> idx-csv | *
            let (xn, yn) = refs
                .split_once(":@")
                .ok_or_else(|| format!("batch line {lineno}: indexed form needs @X:@Y"))?;
            let x_base = *base_names
                .get(xn)
                .ok_or_else(|| format!("batch line {lineno}: undefined base @{xn}"))?;
            let y_base = *base_names
                .get(yn)
                .ok_or_else(|| format!("batch line {lineno}: undefined base @{yn}"))?;
            let (xl, yl) = (bases[x_base].len(), bases[y_base].len());
            if xl != yl {
                return Err(format!(
                    "batch line {lineno}: base @{xn} (len {xl}) and @{yn} (len {yl}) differ — \
                     the shared index set requires equal-length bases"
                ));
            }
            let idx = if fields[2].trim() == "*" {
                None
            } else {
                let idx: Vec<usize> = fields[2]
                    .split(',')
                    .map(|t| {
                        let t = t.trim();
                        t.parse::<usize>()
                            .map_err(|e| format!("batch line {lineno}: bad index {t:?}: {e}"))
                    })
                    .collect::<Result<_, _>>()?;
                if let Some(&bad) = idx.iter().find(|&&v| v >= xl) {
                    return Err(format!(
                        "batch line {lineno}: index {bad} out of bounds for base len {xl}"
                    ));
                }
                Some(idx)
            };
            BatchJob::Indexed {
                x_base,
                y_base,
                idx,
            }
        } else {
            let x = parse_float_csv(fields[1], lineno, "x")?;
            let y = parse_float_csv(fields[2], lineno, "y")?;
            if x.len() != y.len() {
                return Err(format!(
                    "batch line {lineno}: x (len {}) and y (len {}) differ",
                    x.len(),
                    y.len()
                ));
            }
            BatchJob::Explicit { x, y }
        };
        jobs.push((label, job));
    }
    Ok(BatchInput { bases, jobs })
}

/// One computed batch row. Field meanings match the aggregate panel;
/// `srocc_signed` / `plcc_raw` are the direct owner calls documented in
/// the module docs (pre-`.abs()` midrank Spearman; raw un-rescaled Pearson).
struct BatchRow {
    n: usize,
    n_dropped: usize,
    srocc: f64,
    srocc_signed: f64,
    mae_raw: f64,
    full: Option<(PanelStats, f64, f64)>, // (compute_panel stats, plcc_raw, mae)
}

// D2 fixes quality direction before any data is observed. The canonical
// logistic fit can be decreasing; restore its mapping to increasing score
// direction, rather than taking an absolute correlation or substituting raw
// Pearson. No correlation or logistic arithmetic is reimplemented here.
fn signed_quality_correlations(predicted: &[f64], target: &[f64]) -> serde_json::Value {
    let mapped = panel::rescale_logistic(predicted, target);
    let mapping_direction = panel::spearman(predicted, &mapped).signum();
    let rmse = (mapped
        .iter()
        .zip(target)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / target.len() as f64)
        .sqrt();
    serde_json::json!({
        "schema": "panel-signed-quality-v1",
        "orientation": "quality",
        "n": predicted.len(),
        "n_dropped": 0,
        "srocc_signed": panel::spearman(predicted, target),
        "krocc": panel::kendall_tau(predicted, target),
        "plcc": panel::pearson(&mapped, target) * mapping_direction,
        "plcc_raw": panel::pearson(predicted, target),
        "rmse": rmse,
        "plcc_form": "four-parameter logistic; increasing predicted-quality mapping",
    })
}

/// Signed point statistics and whole-reference resamples share the same owner.
fn run_signed_batch(input: &BatchInput) -> String {
    use rayon::prelude::*;
    input
        .jobs
        .par_iter()
        .map(|(label, job)| {
            let (x, y) = match job {
                BatchJob::Explicit { x, y } => (x.clone(), y.clone()),
                BatchJob::Indexed {
                    x_base,
                    y_base,
                    idx,
                } => {
                    let (x, y) = (&input.bases[*x_base], &input.bases[*y_base]);
                    match idx {
                        Some(indices) => (
                            indices.iter().map(|&i| x[i]).collect(),
                            indices.iter().map(|&i| y[i]).collect(),
                        ),
                        None => (x.clone(), y.clone()),
                    }
                }
            };
            assert_eq!(x.len(), y.len(), "aligned signed batch vectors");
            assert!(
                x.len() >= 3 && x.iter().chain(&y).all(|v| v.is_finite()),
                "finite signed batch vectors"
            );
            let mut result = signed_quality_correlations(&x, &y);
            result["label"] = serde_json::json!(label);
            format!("{result}\n")
        })
        .collect::<Vec<_>>()
        .join("")
}

// Shared by the existing mapped-MAE and explicit raw-error output. Keeping
// the arithmetic here makes the rescaling choice visible at the call site.
fn mean_absolute_error(predicted: &[f64], target: &[f64]) -> f64 {
    if predicted.is_empty() {
        f64::NAN
    } else {
        predicted
            .iter()
            .zip(target.iter())
            .map(|(r, t)| (r - t).abs())
            .sum::<f64>()
            / predicted.len() as f64
    }
}

fn compute_batch_row(x: &[f64], y: &[f64], srocc_only: bool) -> BatchRow {
    // Same finite-row drop policy as aggregate mode (report_group).
    let n_in = x.len().min(y.len());
    let mut xf = Vec::with_capacity(n_in);
    let mut yf = Vec::with_capacity(n_in);
    let mut n_dropped = 0usize;
    for i in 0..n_in {
        if x[i].is_finite() && y[i].is_finite() {
            xf.push(x[i]);
            yf.push(y[i]);
        } else {
            n_dropped += 1;
        }
    }
    // panel.rs:50 `spearman` — tie-correct midrank, signed.
    let srocc_signed = panel::spearman(&xf, &yf);
    let full = if srocc_only {
        None
    } else {
        // compute_panel (panel.rs:656) — the full 6-stat panel — plus the
        // raw (un-rescaled) Pearson via panel.rs:72 `pearson`.
        let stats = panel::compute_panel(&xf, &yf);
        let plcc_raw = panel::pearson(&xf, &yf);
        // MAE after the owner's 4-parameter logistic rescale (Mohammadi 2025
        // convention) — the same quantity `bake_verdict`'s per-band rows
        // publish, computed by the same owner call, so a band recomputed
        // through this binary carries every field the emitter does.
        let rescaled = panel::rescale_logistic(&xf, &yf);
        let mae = mean_absolute_error(&rescaled, &yf);
        Some((stats, plcc_raw, mae))
    };
    BatchRow {
        n: xf.len(),
        n_dropped,
        srocc: srocc_signed.abs(),
        srocc_signed,
        mae_raw: mean_absolute_error(&xf, &yf),
        full,
    }
}

/// Round-trip-exact float formatting for the batch TSV.
fn fmt_batch_f(v: f64) -> String {
    format!("{v:.17e}")
}

fn run_batch(input: &BatchInput, srocc_only: bool) -> String {
    run_batch_with_raw(input, srocc_only, false)
}

fn run_batch_with_raw(input: &BatchInput, srocc_only: bool, raw_errors: bool) -> String {
    use rayon::prelude::*;

    let rows: Vec<BatchRow> = input
        .jobs
        .par_iter()
        .map(|(_, job)| match job {
            BatchJob::Explicit { x, y } => compute_batch_row(x, y, srocc_only),
            BatchJob::Indexed {
                x_base,
                y_base,
                idx,
            } => {
                let (bx, by) = (&input.bases[*x_base], &input.bases[*y_base]);
                match idx {
                    None => compute_batch_row(bx, by, srocc_only),
                    Some(idx) => {
                        let x: Vec<f64> = idx.iter().map(|&i| bx[i]).collect();
                        let y: Vec<f64> = idx.iter().map(|&i| by[i]).collect();
                        compute_batch_row(&x, &y, srocc_only)
                    }
                }
            }
        })
        .collect();

    let mut out = String::new();
    if srocc_only {
        out.push_str("label\tn\tn_dropped\tsrocc\tsrocc_signed\n");
    } else {
        out.push_str(
            "label\tn\tn_dropped\tsrocc\tsrocc_signed\tplcc\tplcc_raw\tkrocc\tor\tpwrc\tz_rmse\tmae\n",
        );
    }
    if raw_errors {
        out.pop();
        out.push_str("\tmae_raw\n");
    }
    for ((label, _), r) in input.jobs.iter().zip(&rows) {
        out.push_str(label);
        out.push_str(&format!("\t{}\t{}", r.n, r.n_dropped));
        out.push_str(&format!(
            "\t{}\t{}",
            fmt_batch_f(r.srocc),
            fmt_batch_f(r.srocc_signed)
        ));
        if let Some((p, plcc_raw, mae)) = &r.full {
            out.push_str(&format!(
                "\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                fmt_batch_f(p.plcc),
                fmt_batch_f(*plcc_raw),
                fmt_batch_f(p.krocc),
                fmt_batch_f(p.or_ratio),
                fmt_batch_f(p.pwrc),
                fmt_batch_f(p.z_rmse),
                fmt_batch_f(*mae)
            ));
        }
        if raw_errors {
            out.push_str(&format!("\t{}", fmt_batch_f(r.mae_raw)));
        }
        out.push('\n');
    }
    out
}

// ----------------------------------------------------------------------
// Rendering
// ----------------------------------------------------------------------

fn fmt_stat(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.4}")
    } else {
        "  n/a ".to_string()
    }
}

fn render_text(reports: &[GroupReport], has_sigma: bool) -> String {
    let mut s = String::new();
    s.push_str("# panel — Mohammadi 2025 IQA statistical panel\n");
    s.push_str("# (all stats via zensim_validate::panel — see `panel --help`)\n\n");
    if has_sigma {
        s.push_str(&format!(
            "{:<24} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>12}\n",
            "group", "n", "SROCC", "PLCC", "KROCC", "OR", "PWRC", "Z-RMSE", "Z-RMSE/σ"
        ));
    } else {
        s.push_str(&format!(
            "{:<24} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}\n",
            "group", "n", "SROCC", "PLCC", "KROCC", "OR", "PWRC", "Z-RMSE"
        ));
    }
    for r in reports {
        let p = &r.panel;
        if has_sigma {
            s.push_str(&format!(
                "{:<24} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>12}\n",
                r.label,
                r.n_kept,
                fmt_stat(p.srocc),
                fmt_stat(p.plcc),
                fmt_stat(p.krocc),
                fmt_stat(p.or_ratio),
                fmt_stat(p.pwrc),
                fmt_stat(p.z_rmse),
                fmt_stat(r.z_rmse_per_sample.unwrap_or(f64::NAN)),
            ));
        } else {
            s.push_str(&format!(
                "{:<24} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}\n",
                r.label,
                r.n_kept,
                fmt_stat(p.srocc),
                fmt_stat(p.plcc),
                fmt_stat(p.krocc),
                fmt_stat(p.or_ratio),
                fmt_stat(p.pwrc),
                fmt_stat(p.z_rmse),
            ));
        }
    }
    let dropped: usize = reports.first().map(|r| r.n_dropped).unwrap_or(0);
    if dropped > 0 {
        s.push_str(&format!(
            "\n# {dropped} row(s) dropped (non-finite predicted/target) from the ALL group\n"
        ));
    }
    s
}

/// JSON helper: a finite f64 renders as `{:.10}`, non-finite as `null`.
fn json_f(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.10}")
    } else {
        "null".to_string()
    }
}

fn render_json(reports: &[GroupReport], has_sigma: bool) -> String {
    let mut s = String::from("{\n  \"groups\": [\n");
    for (gi, r) in reports.iter().enumerate() {
        let p = &r.panel;
        s.push_str("    {\n");
        s.push_str(&format!("      \"label\": {:?},\n", r.label));
        s.push_str(&format!("      \"n\": {},\n", r.n_kept));
        s.push_str(&format!("      \"n_dropped\": {},\n", r.n_dropped));
        s.push_str(&format!("      \"srocc\": {},\n", json_f(p.srocc)));
        s.push_str(&format!("      \"plcc\": {},\n", json_f(p.plcc)));
        s.push_str(&format!("      \"krocc\": {},\n", json_f(p.krocc)));
        s.push_str(&format!("      \"or\": {},\n", json_f(p.or_ratio)));
        s.push_str(&format!("      \"pwrc\": {},\n", json_f(p.pwrc)));
        if has_sigma {
            s.push_str(&format!("      \"z_rmse\": {},\n", json_f(p.z_rmse)));
            s.push_str(&format!(
                "      \"z_rmse_per_sample\": {}\n",
                json_f(r.z_rmse_per_sample.unwrap_or(f64::NAN))
            ));
        } else {
            s.push_str(&format!("      \"z_rmse\": {}\n", json_f(p.z_rmse)));
        }
        s.push_str("    }");
        if gi + 1 < reports.len() {
            s.push(',');
        }
        s.push('\n');
    }
    s.push_str("  ]\n}\n");
    s
}

// ----------------------------------------------------------------------
// pairwise (2AFC / triplet) mode
// ----------------------------------------------------------------------

fn read_text(p: &Path) -> Result<String, String> {
    if p.as_os_str() == "-" {
        use std::io::Read;
        let mut s = String::new();
        std::io::stdin()
            .read_to_string(&mut s)
            .map_err(|e| format!("read stdin: {e}"))?;
        Ok(s)
    } else {
        std::fs::read_to_string(p).map_err(|e| format!("read {p:?}: {e}"))
    }
}

/// The first nine columns are the pre-2026-10-11 contract, unchanged; the
/// cross-fitted ceilings that came with the move to `zenstats::forced_choice`
/// are appended so name- and position-based readers both keep working.
const PAIRWISE_HEADER: &str = "label\tn_groups\tn_responses\tacc_response\ttie_rate\tceiling_response\tacc_norm\tacc_group_majority\tn_groups_majority\tceiling_loo\tobserver_agreement\tacc_norm_loo\n";

fn pairwise_row(label: &str, s: &zenstats::forced_choice::PairwiseStats) -> String {
    format!(
        "{label}\t{}\t{:.0}\t{:.10}\t{:.10}\t{:.10}\t{:.10}\t{:.10}\t{}\t{:.10}\t{:.10}\t{:.10}\n",
        s.n_groups,
        s.n_responses,
        s.acc_response,
        s.tie_rate,
        s.ceiling_response,
        s.acc_norm,
        s.acc_group_majority,
        s.n_groups_majority,
        s.ceiling_loo,
        s.observer_agreement,
        s.acc_norm_loo
    )
}

/// `--pairwise`: read weighted forced-choice rows, emit one stat row (or one
/// per `--resample` line). No stat math here — every number comes from
/// `zenstats::forced_choice`.
fn run_pairwise(path: &Path, resample: Option<&Path>) -> Result<String, String> {
    use std::collections::HashMap;
    use zenstats::forced_choice::{
        Choice, PairwiseRow, agreement, agreement_by_group_index, index_rows_by_group,
    };

    let text = read_text(path)?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines.next().ok_or("--pairwise: empty input")?;
    let cols: Vec<&str> = header.split('\t').map(|c| c.trim()).collect();
    let col = |name: &str| cols.iter().position(|c| *c == name);
    let (ig, il, ir, ic) = (
        col("group").ok_or("--pairwise: missing column 'group'")?,
        col("s_left").ok_or("--pairwise: missing column 's_left'")?,
        col("s_right").ok_or("--pairwise: missing column 's_right'")?,
        col("choice").ok_or("--pairwise: missing column 'choice'")?,
    );
    let iw = col("weight");

    let mut key_to_idx: HashMap<String, usize> = HashMap::new();
    let mut group_names: Vec<String> = Vec::new();
    let mut rows: Vec<PairwiseRow> = Vec::new();
    for (n, line) in lines.enumerate() {
        let f: Vec<&str> = line.split('\t').collect();
        let need = ig.max(il).max(ir).max(ic).max(iw.unwrap_or(0));
        if f.len() <= need {
            return Err(format!("--pairwise: line {} has {} fields", n + 2, f.len()));
        }
        let key = f[ig].to_string();
        let g = *key_to_idx.entry(key.clone()).or_insert_with(|| {
            group_names.push(key);
            group_names.len() - 1
        });
        let parse = |v: &str, what: &str| -> Result<f64, String> {
            v.trim()
                .parse::<f64>()
                .map_err(|e| format!("--pairwise: line {}: {what}: {e}", n + 2))
        };
        let choice = Choice::parse(f[ic]).ok_or_else(|| {
            format!(
                "--pairwise: line {}: choice must be left|right|notsure, got {:?}",
                n + 2,
                f[ic]
            )
        })?;
        let weight = match iw {
            Some(i) => parse(f[i], "weight")?,
            None => 1.0,
        };
        if !(weight.is_finite() && weight >= 0.0) {
            return Err(format!(
                "--pairwise: line {}: weight must be finite >= 0",
                n + 2
            ));
        }
        rows.push(PairwiseRow {
            group: g,
            s_left: parse(f[il], "s_left")?,
            s_right: parse(f[ir], "s_right")?,
            choice,
            weight,
        });
    }
    if rows.is_empty() {
        return Err("--pairwise: no data rows".to_string());
    }

    let n_groups = group_names.len();
    let mut out = String::from(PAIRWISE_HEADER);
    match resample {
        None => out.push_str(&pairwise_row("ALL", &agreement(&rows, n_groups))),
        Some(rp) => {
            let by_group = index_rows_by_group(&rows, n_groups);
            let rtext = read_text(rp)?;
            for (n, line) in rtext.lines().enumerate() {
                let line = line.trim_end();
                if line.trim().is_empty() || line.starts_with('#') {
                    continue;
                }
                let (label, spec) = line
                    .split_once('\t')
                    .ok_or_else(|| format!("--resample: line {} needs LABEL<TAB>spec", n + 1))?;
                let picked: Vec<usize> = if spec.trim() == "*" {
                    (0..n_groups).collect()
                } else {
                    let mut v = Vec::new();
                    for t in spec.split(',') {
                        let i: usize = t
                            .trim()
                            .parse()
                            .map_err(|e| format!("--resample: line {}: index {t:?}: {e}", n + 1))?;
                        if i >= n_groups {
                            return Err(format!(
                                "--resample: line {}: group index {i} >= {n_groups}",
                                n + 1
                            ));
                        }
                        v.push(i);
                    }
                    v
                };
                out.push_str(&pairwise_row(
                    label,
                    &agreement_by_group_index(&by_group, &picked),
                ));
            }
        }
    }
    Ok(out)
}

// ----------------------------------------------------------------------
// main
// ----------------------------------------------------------------------

fn main() -> ExitCode {
    zensim_validate::tier_cap::apply_from_env();
    if std::env::args().nth(1).as_deref() == Some("e34") {
        return panel_e34::main();
    }
    if std::env::args().nth(1).as_deref() == Some("rank-interval") {
        return panel_rank_interval::main();
    }
    if std::env::args().nth(1).as_deref() == Some("evaluate") {
        return panel_evaluate::main();
    }
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("panel: {e}\n");
            print_usage();
            return ExitCode::from(2);
        }
    };

    // Pairwise mode: forced-choice agreement, optionally over a caller's
    // cluster-bootstrap resamples. See `zenstats::forced_choice`.
    if let Some(pw_path) = &args.pairwise {
        match run_pairwise(pw_path, args.resample.as_deref()) {
            Ok(out) => {
                print!("{out}");
                return ExitCode::SUCCESS;
            }
            Err(e) => {
                eprintln!("panel: {e}");
                return ExitCode::from(2);
            }
        }
    }

    // Batch mode: read the manifest (file or stdin), emit the TSV, done.
    if let Some(batch_path) = &args.batch {
        let text = if batch_path.as_os_str() == "-" {
            use std::io::Read;
            let mut s = String::new();
            if let Err(e) = std::io::stdin().read_to_string(&mut s) {
                eprintln!("panel: read stdin: {e}");
                return ExitCode::from(1);
            }
            s
        } else {
            match std::fs::read_to_string(batch_path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("panel: read {batch_path:?}: {e}");
                    return ExitCode::from(1);
                }
            }
        };
        let input = match parse_batch(&text) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("panel: {e}");
                return ExitCode::from(2);
            }
        };
        if args.signed_quality {
            print!("{}", run_signed_batch(&input));
        } else if args.raw_errors {
            print!(
                "{}",
                run_batch_with_raw(&input, args.stats_srocc_only, true)
            );
        } else {
            print!("{}", run_batch(&input, args.stats_srocc_only));
        }
        return ExitCode::SUCCESS;
    }

    let cols = match load_columns(&args) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("panel: {e}");
            return ExitCode::from(1);
        }
    };

    if args.signed_quality {
        if cols.predicted.len() != cols.target.len()
            || cols.predicted.len() < 3
            || cols
                .predicted
                .iter()
                .chain(&cols.target)
                .any(|v| !v.is_finite())
        {
            eprintln!("panel: signed quality requires aligned finite pairs");
            return ExitCode::from(2);
        }
        println!(
            "{}",
            signed_quality_correlations(&cols.predicted, &cols.target)
        );
        return ExitCode::SUCCESS;
    }

    if args.scatter {
        let value = scatter_json::assess(&cols.predicted, &cols.target);
        println!("{value}");
        return if value["status"] == "MEASURED" {
            ExitCode::SUCCESS
        } else {
            ExitCode::from(2)
        };
    }

    if cols.predicted.is_empty() {
        eprintln!("panel: no data rows in {:?}", args.input);
        return ExitCode::from(1);
    }

    // Hidden parity-gate path: emit the logistic-rescaled `predicted`
    // column for finite (predicted, target) rows of the ALL group, one
    // per line. Used by scripts/verify_panel_parity.py.
    if args.emit_rescaled {
        let n = cols.predicted.len().min(cols.target.len());
        let mut p = Vec::with_capacity(n);
        let mut t = Vec::with_capacity(n);
        for i in 0..n {
            if cols.predicted[i].is_finite() && cols.target[i].is_finite() {
                p.push(cols.predicted[i]);
                t.push(cols.target[i]);
            }
        }
        let rescaled = panel::rescale_logistic(&p, &t);
        let mut out = String::new();
        for v in rescaled {
            out.push_str(&format!("{v:.17e}\n"));
        }
        print!("{out}");
        return ExitCode::SUCCESS;
    }

    let has_sigma = cols.sigma.is_some();
    let reports = build_reports(&cols);

    // Within-band (per-reference) SROCC summary — the canonical owner call.
    let per_group = if args.per_group {
        let Some(bands) = &cols.band else {
            eprintln!(
                "panel: --per-group needs a grouping column; none found \
                 (expected {:?}, override with --col-band)",
                args.col_band
            );
            return ExitCode::from(2);
        };
        per_group_summary(&cols.predicted, &cols.target, bands)
    } else {
        None
    };

    if args.json {
        // ONE document. Before 2026-09-01 `--json --per-group` printed two
        // concatenated top-level objects, i.e. stdout was not valid JSON
        // (found by the ssim2-bar lane, A.7; nothing had tripped on it
        // because `zen_stats.panel` never passes `--per-group`). The
        // per-group block is now a key of the panel document.
        print!(
            "{}",
            render_json_with_per_group(&reports, has_sigma, per_group.as_ref())
        );
    } else {
        print!("{}", render_text(&reports, has_sigma));
        if args.per_group {
            print!("{}", render_per_group_text(per_group.as_ref()));
        }
    }
    ExitCode::SUCCESS
}

/// Within-group SROCC over string group labels — a thin adapter onto the
/// canonical [`panel::per_group_srocc`], which is generic over `Copy` keys.
///
/// `Orientation::Auto` matches what `compute_panel`'s `.abs()` does pooled, so
/// a group that disagrees with the pooled polarity still comes out NEGATIVE —
/// which is the detection this stat exists for and the reason it must not be
/// taken absolute per group.
fn per_group_summary(
    predicted: &[f64],
    target: &[f64],
    bands: &[String],
) -> Option<panel::PerGroupSrocc> {
    let n = predicted.len().min(target.len()).min(bands.len());
    let mut ids: BTreeMap<&str, usize> = BTreeMap::new();
    let mut keys: Vec<usize> = Vec::with_capacity(n);
    for b in bands.iter().take(n) {
        let next = ids.len();
        keys.push(*ids.entry(b.as_str()).or_insert(next));
    }
    panel::per_group_srocc(
        &predicted[..n],
        &target[..n],
        &keys,
        PER_GROUP_MIN_LEN,
        panel::Orientation::Auto,
    )
}

/// Minimum pairs in a group before its within-group SROCC means anything.
/// 3 is the smallest n for which a rank correlation is not forced to ±1,
/// and it is what `bake_verdict` uses for its `per_ref` rows.
const PER_GROUP_MIN_LEN: usize = 3;

fn render_per_group_text(g: Option<&panel::PerGroupSrocc>) -> String {
    match g {
        None => "\n# per-group SROCC: NOT MEASURED — no group cleared the \
                 min-length / spread filters\n"
            .to_string(),
        Some(g) => format!(
            "\n# within-group (per-reference) SROCC — zenstats::per_group_srocc, \
             the same quantity bake_verdict publishes as per_ref_mean\n\
             {:<24} {:>6} {:>8} {:>8} {:>10} {:>10}\n\
             {:<24} {:>6} {:>8.4} {:>8.4} {:>10.4} {:>10.4}\n",
            "stat",
            "groups",
            "mean",
            "median",
            "frac_neg",
            "frac_perf",
            "per_group",
            g.n_groups,
            g.mean,
            g.median,
            g.frac_negative,
            g.frac_perfect
        ),
    }
}

/// `render_json` + an optional `"per_group"` key, as ONE valid document.
fn render_json_with_per_group(
    reports: &[GroupReport],
    has_sigma: bool,
    g: Option<&panel::PerGroupSrocc>,
) -> String {
    let base = render_json(reports, has_sigma);
    let Some(g) = g else { return base };
    let trimmed = base.trim_end();
    let Some(body) = trimmed.strip_suffix('}') else {
        // render_json's shape changed; fail loud rather than emit bad JSON.
        panic!("render_json no longer ends with '}}' — cannot splice per_group");
    };
    format!(
        "{}, \"per_group\": {{\n\
         \x20\x20\x20\x20\"n_groups\": {},\n\
         \x20\x20\x20\x20\"mean\": {},\n\
         \x20\x20\x20\x20\"median\": {},\n\
         \x20\x20\x20\x20\"frac_negative\": {},\n\
         \x20\x20\x20\x20\"frac_perfect\": {}\n  }}\n}}\n",
        body.trim_end(),
        g.n_groups,
        json_f(g.mean),
        json_f(g.median),
        json_f(g.frac_negative),
        json_f(g.frac_perfect)
    )
}

#[allow(dead_code)]
fn render_per_group_json(g: &panel::PerGroupSrocc) -> String {
    format!(
        "{{\n  \"per_group\": {{\n\
         \x20\x20\x20\x20\"n_groups\": {},\n\
         \x20\x20\x20\x20\"mean\": {},\n\
         \x20\x20\x20\x20\"median\": {},\n\
         \x20\x20\x20\x20\"frac_negative\": {},\n\
         \x20\x20\x20\x20\"frac_perfect\": {}\n  }}\n}}\n",
        g.n_groups,
        json_f(g.mean),
        json_f(g.median),
        json_f(g.frac_negative),
        json_f(g.frac_perfect)
    )
}

// ----------------------------------------------------------------------
// Tests — the panel subcommand's own plumbing (NOT the stat math, which
// is tested in panel.rs). The cross-language parity gate lives in
// `tests/panel_parity.rs`.
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_quality_keeps_reversed_polarity_and_fitted_plcc() {
        let quality: Vec<f64> = (0..50).map(|i| (f64::from(i) - 25.0) / 3.0).collect();
        let y: Vec<f64> = quality.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
        for sign in [1.0, -1.0] {
            let x: Vec<f64> = quality.iter().map(|v| sign * v).collect();
            let r = signed_quality_correlations(&x, &y);
            assert_eq!(r["orientation"], "quality");
            assert!((r["srocc_signed"].as_f64().unwrap() - sign).abs() < 1e-12);
            assert!((r["krocc"].as_f64().unwrap() - sign).abs() < 1e-12);
            let mapped = panel::rescale_logistic(&x, &y);
            let fitted = panel::pearson(&mapped, &y);
            assert!((r["plcc"].as_f64().unwrap() - sign * fitted).abs() < 1e-12);
            assert!(r["plcc"].as_f64().unwrap() * sign > 0.99);
            assert!((r["plcc"].as_f64().unwrap() - r["plcc_raw"].as_f64().unwrap()).abs() > 0.01);
        }
    }

    /// The whole point of a per-GROUP stat: a group that ranks BACKWARDS must
    /// come out negative and be counted, where a pooled `.abs()` would hide it.
    #[test]
    fn per_group_counts_a_backwards_group_as_negative() {
        // two forward references, one exactly reversed.
        let mut pred = Vec::new();
        let mut targ = Vec::new();
        let mut band = Vec::new();
        for g in ["fwdA", "fwdB"] {
            for i in 0..6 {
                pred.push(i as f64);
                targ.push(i as f64);
                band.push(g.to_string());
            }
        }
        for i in 0..6 {
            pred.push(i as f64);
            targ.push(-(i as f64));
            band.push("backwards".to_string());
        }
        let g = per_group_summary(&pred, &targ, &band).expect("three groups");
        assert_eq!(g.n_groups, 3);
        assert!(
            (g.frac_negative - 1.0 / 3.0).abs() < 1e-9,
            "one of three groups is backwards, got frac_negative={}",
            g.frac_negative
        );
        // mean = (1 + 1 - 1)/3; a pooled abs would have reported ~1.0.
        assert!((g.mean - 1.0 / 3.0).abs() < 1e-9, "mean={}", g.mean);
        assert!((g.median - 1.0).abs() < 1e-9, "median={}", g.median);
    }

    /// Groups shorter than `PER_GROUP_MIN_LEN`, and groups with no spread,
    /// are dropped rather than contributing a forced ±1.
    #[test]
    fn per_group_drops_short_and_degenerate_groups() {
        let mut pred = Vec::new();
        let mut targ = Vec::new();
        let mut band = Vec::new();
        for i in 0..6 {
            pred.push(i as f64);
            targ.push(i as f64);
            band.push("ok".to_string());
        }
        // 2 rows: below PER_GROUP_MIN_LEN
        for i in 0..2 {
            pred.push(i as f64);
            targ.push(i as f64);
            band.push("tooshort".to_string());
        }
        // 5 rows, constant target: no spread
        for i in 0..5 {
            pred.push(i as f64);
            targ.push(7.0);
            band.push("flat".to_string());
        }
        let g = per_group_summary(&pred, &targ, &band).expect("one usable group");
        assert_eq!(g.n_groups, 1, "only the 6-row spread group is usable");
    }

    #[test]
    fn report_group_drops_nonfinite() {
        let predicted = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];
        let target = vec![1.0, 2.0, 3.0, f64::INFINITY, 5.0];
        let r = report_group("t", &predicted, &target, None);
        assert_eq!(r.n_kept, 3, "rows 0,1,4 kept");
        assert_eq!(r.n_dropped, 2, "rows 2,3 dropped");
    }

    #[test]
    fn report_group_perfect_rank() {
        let predicted: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let target: Vec<f64> = (0..50).map(|i| i as f64 * 2.0 + 3.0).collect();
        let r = report_group("t", &predicted, &target, None);
        assert!(
            (r.panel.srocc - 1.0).abs() < 1e-9,
            "srocc={}",
            r.panel.srocc
        );
        assert!(
            (r.panel.krocc - 1.0).abs() < 1e-9,
            "krocc={}",
            r.panel.krocc
        );
    }

    #[test]
    fn per_sample_z_rmse_present_only_with_sigma() {
        let predicted: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let target: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let without = report_group("t", &predicted, &target, None);
        assert!(without.z_rmse_per_sample.is_none());
        let sigma = vec![1.0; 40];
        let with = report_group("t", &predicted, &target, Some(&sigma));
        assert!(with.z_rmse_per_sample.is_some());
    }

    #[test]
    fn json_with_per_group_is_one_document() {
        // Regression for the two-concatenated-documents defect: stdout of
        // `--json --per-group` must parse as ONE object carrying per_group.
        let pred = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let targ = vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0];
        let band = vec![
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
            "b".to_string(),
        ];
        let reports = vec![GroupReport {
            label: "ALL".to_string(),
            panel: panel::compute_panel(&pred, &targ),
            z_rmse_per_sample: None,
            n_kept: pred.len(),
            n_dropped: 0,
        }];
        let g = per_group_summary(&pred, &targ, &band).expect("two groups");
        let out = render_json_with_per_group(&reports, false, Some(&g));
        // Exactly one top-level object: the text must open once and close once.
        assert_eq!(
            out.matches("\n}").count(),
            1,
            "more than one document:\n{out}"
        );
        assert!(out.contains("\"per_group\""), "{out}");
        assert!(out.contains("\"n_groups\": 2"), "{out}");
        // and the no-per-group form is unchanged
        assert_eq!(
            render_json_with_per_group(&reports, false, None),
            render_json(&reports, false)
        );
    }

    #[test]
    fn json_renders_nan_as_null() {
        assert_eq!(json_f(f64::NAN), "null");
        assert_eq!(json_f(f64::INFINITY), "null");
        assert_eq!(json_f(0.5), "0.5000000000");
    }

    // ------------------------------------------------------------------
    // Batch-mode plumbing (the stat math itself is tested in panel.rs /
    // tests/panel_parity.rs; scipy cross-check in
    // scripts/verify_panel_batch_parity.py).
    // ------------------------------------------------------------------

    #[test]
    fn pairwise_reads_weights_groups_and_resamples() {
        // Two triplets. Group A: humans 6-4 for LEFT, metric says LEFT worse
        // (60 < 90) -> agrees with the 6. Group B: humans 9-2 for RIGHT,
        // metric says LEFT worse (91 > 89 is false => s_left 89 < 91) ->
        // disagrees with the 9.
        let dir = std::env::temp_dir().join(format!("panel_pw_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let rows = dir.join("rows.tsv");
        std::fs::write(
            &rows,
            "group\ts_left\ts_right\tchoice\tweight\n\
             A\t60\t90\tleft\t6\n\
             A\t60\t90\tright\t4\n\
             B\t89\t91\tleft\t2\n\
             B\t89\t91\tright\t9\n",
        )
        .unwrap();
        let out = run_pairwise(&rows, None).unwrap();
        let line: Vec<&str> = out.lines().nth(1).unwrap().split('\t').collect();
        assert_eq!(line[0], "ALL");
        assert_eq!(line[1], "2", "n_groups");
        assert_eq!(line[2], "21", "n_responses = 6+4+2+9");
        // agreement = (6 + 2) / 21
        let acc: f64 = line[3].parse().unwrap();
        assert!((acc - 8.0 / 21.0).abs() < 1e-9, "{acc}");
        // ceiling = (max(6,4) + max(2,9)) / 21 = 15/21
        let ceil: f64 = line[5].parse().unwrap();
        assert!((ceil - 15.0 / 21.0).abs() < 1e-9, "{ceil}");
        // majority: A right, B wrong -> 0.5 over 2 groups
        let maj: f64 = line[7].parse().unwrap();
        assert!((maj - 0.5).abs() < 1e-9, "{maj}");

        // Resample: '*' must reproduce the point estimate; a duplicate draw
        // of group A alone must read a pure-A statistic.
        let man = dir.join("man.tsv");
        std::fs::write(&man, "POINT\t*\nAA\t0,0\n").unwrap();
        let out2 = run_pairwise(&rows, Some(&man)).unwrap();
        let l1: Vec<&str> = out2.lines().nth(1).unwrap().split('\t').collect();
        assert_eq!(l1[0], "POINT");
        assert_eq!(
            l1[3], line[3],
            "'*' must equal the no-resample point estimate"
        );
        let l2: Vec<&str> = out2.lines().nth(2).unwrap().split('\t').collect();
        assert_eq!(l2[1], "2", "group A drawn twice = two clusters");
        let acc_aa: f64 = l2[3].parse().unwrap();
        assert!((acc_aa - 0.6).abs() < 1e-9, "{acc_aa}");

        // A bad choice token must fail loud, not be silently dropped.
        let bad = dir.join("bad.tsv");
        std::fs::write(&bad, "group\ts_left\ts_right\tchoice\nA\t1\t2\tmaybe\n").unwrap();
        assert!(run_pairwise(&bad, None).is_err());
        // A missing required column must fail loud too.
        let nocol = dir.join("nocol.tsv");
        std::fs::write(&nocol, "group\ts_left\tchoice\nA\t1\tleft\n").unwrap();
        assert!(run_pairwise(&nocol, None).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pairwise_counts_not_sure_half_each_and_appends_cross_fitted_ceilings() {
        // One triplet: humans 6 LEFT, 4 RIGHT, 2 not sure; the metric says
        // LEFT is worse (60 < 90). Not sure adds 1 to each side: 7 vs 5.
        let dir = std::env::temp_dir().join(format!("panel_pw_ns_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let rows = dir.join("rows.tsv");
        std::fs::write(
            &rows,
            "group\ts_left\ts_right\tchoice\tweight\n\
             A\t60\t90\tleft\t6\n\
             A\t60\t90\tright\t4\n\
             A\t60\t90\tnotsure\t2\n",
        )
        .unwrap();
        let out = run_pairwise(&rows, None).unwrap();
        let header: Vec<&str> = out.lines().next().unwrap().split('\t').collect();
        assert_eq!(
            &header[9..],
            &["ceiling_loo", "observer_agreement", "acc_norm_loo"]
        );
        let line: Vec<&str> = out.lines().nth(1).unwrap().split('\t').collect();
        assert_eq!(line.len(), header.len());
        assert_eq!(line[2], "12", "n_responses = 6+4+2");
        let f = |i: usize| -> f64 { line[i].parse().unwrap() };
        assert!((f(3) - 7.0 / 12.0).abs() < 1e-9, "acc = (6 + 2/2) / 12");
        assert!((f(5) - 7.0 / 12.0).abs() < 1e-9, "ceiling = max(7, 5) / 12");
        assert!((f(6) - 1.0).abs() < 1e-9, "acc_norm");
        // leave one out: a left response sees 6 vs 5 (majority), a right one
        // 7 vs 4 (minority), so 7/12
        assert!((f(9) - 7.0 / 12.0).abs() < 1e-9, "ceiling_loo");
        assert!(
            (f(10) - (7.0 * 6.0 + 5.0 * 4.0) / (12.0 * 11.0)).abs() < 1e-9,
            "observer agreement"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn batch_parses_explicit_and_indexed_and_star() {
        let text = "#def X\t1,2,3,4\n#def Y\t10,20,30,40\n\
                    # a comment\n\
                    e1\t1,2,3\t3,2,1\n\
                    i1\t@X:@Y\t0,2,1\n\
                    all\t@X:@Y\t*\n";
        let b = parse_batch(text).unwrap();
        assert_eq!(b.bases.len(), 2);
        assert_eq!(b.jobs.len(), 3);
        match &b.jobs[0].1 {
            BatchJob::Explicit { x, y } => {
                assert_eq!(x, &[1.0, 2.0, 3.0]);
                assert_eq!(y, &[3.0, 2.0, 1.0]);
            }
            _ => panic!("job 0 should be explicit"),
        }
        match &b.jobs[1].1 {
            BatchJob::Indexed { idx: Some(idx), .. } => assert_eq!(idx, &[0, 2, 1]),
            _ => panic!("job 1 should be indexed"),
        }
        match &b.jobs[2].1 {
            BatchJob::Indexed { idx: None, .. } => {}
            _ => panic!("job 2 should be the * form"),
        }
    }

    #[test]
    fn batch_rejects_bad_input_loudly() {
        assert!(parse_batch("l\t1,2\t1,2,3\n").is_err(), "length mismatch");
        assert!(parse_batch("l\t@A:@B\t0\n").is_err(), "undefined base");
        assert!(
            parse_batch("#def A\t1,2\nl\t@A:@A\t5\n").is_err(),
            "index out of bounds"
        );
        assert!(parse_batch("l\t1,zz\t1,2\n").is_err(), "garbage float");
        assert!(
            parse_batch("#def A\t1,2\n#def A\t3,4\n").is_err(),
            "duplicate def"
        );
        assert!(
            parse_batch("#def A\t1,2\n#def B\t1,2,3\nl\t@A:@B\t*\n").is_err(),
            "unequal base lengths"
        );
    }

    #[test]
    fn batch_indexed_matches_explicit_and_is_deterministic() {
        // Same resample expressed both ways must produce identical rows,
        // and two runs must be byte-identical (no RNG anywhere).
        let text = "#def P\t12,9,30,25,5,40\n#def T\t80,85,40,55,92,20\n\
                    ex\t9,30,5,9\t85,40,92,85\n\
                    ix\t@P:@T\t1,2,4,1\n";
        let b = parse_batch(text).unwrap();
        let out1 = run_batch(&b, false);
        let out2 = run_batch(&b, false);
        assert_eq!(out1, out2, "batch output must be deterministic");
        let lines: Vec<&str> = out1.lines().collect();
        assert_eq!(lines.len(), 3);
        let ex_cols: Vec<&str> = lines[1].split('\t').collect();
        let ix_cols: Vec<&str> = lines[2].split('\t').collect();
        assert_eq!(ex_cols[1..], ix_cols[1..], "indexed == explicit");
    }

    #[test]
    fn batch_row_drops_nonfinite_and_counts() {
        let x = [1.0, f64::NAN, 3.0, 4.0];
        let y = [1.0, 2.0, f64::INFINITY, 4.0];
        let r = compute_batch_row(&x, &y, true);
        assert_eq!(r.n, 2);
        assert_eq!(r.n_dropped, 2);
    }

    #[test]
    fn batch_srocc_signed_carries_polarity() {
        // Anti-correlated pair: signed is negative, abs is 1.0.
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [10.0, 8.0, 6.0, 4.0, 2.0];
        let r = compute_batch_row(&x, &y, true);
        assert!((r.srocc - 1.0).abs() < 1e-12, "srocc={}", r.srocc);
        assert!(
            (r.srocc_signed + 1.0).abs() < 1e-12,
            "srocc_signed={}",
            r.srocc_signed
        );
    }

    #[test]
    fn batch_full_matches_aggregate_compute_panel() {
        // The full batch row must equal what aggregate mode's
        // report_group produces on the same pair.
        let pred: Vec<f64> = vec![12.0, 9.0, 30.0, 25.0, 5.0, 40.0, 22.0, 18.0];
        let tgt: Vec<f64> = vec![80.0, 85.0, 40.0, 55.0, 92.0, 20.0, 60.0, 70.0];
        let r = compute_batch_row(&pred, &tgt, false);
        let agg = report_group("t", &pred, &tgt, None);
        let (p, plcc_raw, _mae) = r.full.as_ref().unwrap();
        assert_eq!(p.srocc, agg.panel.srocc);
        assert_eq!(p.plcc, agg.panel.plcc);
        assert_eq!(p.krocc, agg.panel.krocc);
        assert_eq!(p.or_ratio, agg.panel.or_ratio);
        assert_eq!(p.pwrc, agg.panel.pwrc);
        assert_eq!(p.z_rmse, agg.panel.z_rmse);
        assert_eq!(r.srocc, agg.panel.srocc, "abs srocc column == panel srocc");
        // plcc_raw is signed raw Pearson (distance-shaped fixture → negative).
        assert!(*plcc_raw < 0.0, "raw pearson keeps polarity: {plcc_raw}");
    }

    #[test]
    fn batch_tie_heavy_midrank() {
        // Heavy exact ties; golden value from scipy.stats.spearmanr
        // (midrank) run on this exact fixture: 0.8846153846153847
        // (recomputed 2026-07-31; the full randomized sweep incl.
        // tie-heavy cases is scripts/verify_panel_batch_parity.py).
        let x = [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0];
        let y = [1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
        let r = compute_batch_row(&x, &y, true);
        assert!(
            (r.srocc_signed - 0.884_615_384_615_384_7).abs() < 1e-12,
            "tie-heavy midrank srocc={}",
            r.srocc_signed
        );
    }

    #[test]
    fn raw_error_keeps_calibration_offset_and_finite_row_policy() {
        let input = parse_batch("offset\t11,12,13,14,NaN\t1,2,3,4,5\n").unwrap();
        let legacy = run_batch(&input, false);
        let raw = run_batch_with_raw(&input, false, true);
        let header: Vec<_> = raw.lines().next().unwrap().split('\t').collect();
        let row: Vec<_> = raw.lines().nth(1).unwrap().split('\t').collect();
        assert_eq!(&row[1..3], &["4", "1"]);
        assert_eq!(header.last(), Some(&"mae_raw"));
        assert_eq!(row.last().unwrap().parse::<f64>().unwrap(), 10.0);
        assert_eq!(
            row[..row.len() - 1].join("\t"),
            legacy.lines().nth(1).unwrap(),
            "requesting raw error preserves every legacy statistic"
        );
    }

    #[test]
    fn signed_reference_resamples_preserve_negative_direction_and_ties() {
        let input = parse_batch(concat!(
            "#def\tx\t1,2,3,4,5,6\n",
            "#def\ty\t6,5,4,3,2,1\n",
            "full\t@x:@y\t*\n",
            "resampled\t@x:@y\t0,0,2,2,4,4\n",
        ))
        .unwrap();
        let rows: Vec<serde_json::Value> = run_signed_batch(&input)
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(rows.len(), 2);
        for row in rows {
            assert_eq!(row["n"], 6);
            for key in ["srocc_signed", "krocc", "plcc"] {
                assert!(row[key].as_f64().unwrap() < -0.99, "{key}: {row}");
            }
            assert!(row["rmse"].as_f64().unwrap().is_finite());
            assert_eq!(row["n_dropped"], 0);
        }
    }
}
