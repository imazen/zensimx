//! `--manifest <path.toml>` reproduce-this input mode for
//! `zensim_mlp_train`.
//!
//! A shipped bake's TOML manifest (`zensim/weights/manifests/*.toml`,
//! schema documented in `zensim/weights/manifests/README.md`) records
//! everything needed to reproduce its training run: the structured
//! `[training]` hyperparameters, the `groups` array, the
//! `auto_transforms` / `anchor_parquet` paths, the post-training
//! `steps`, and one `[inputs.<name>]` block per input file carrying its
//! `sha256` + `rows` + R2/Tower mirror URLs.
//!
//! This module flips the manifest from OUTPUT-only provenance into a
//! reproduce-this INPUT: parse the structured fields into a
//! [`ManifestConfig`], verify every referenced input file's sha256
//! before training, and let the binary map the config onto its `Args`
//! (manifest = defaults, explicit CLI flags override).
//!
//! ## Why structured fields, not the recorded command string
//!
//! The manifest carries BOTH a literal `command = """..."""` shell
//! string (with `$TRAIN` / `$CANON` env-var placeholders) AND the
//! equivalent structured fields (`seed`, `hidden`, `groups = [...]`,
//! etc.). We map the **structured fields**: the command string is
//! free-form shell with per-manifest env-var conventions that are not
//! machine-stable, whereas the structured fields are a fixed schema.
//! The manifest README's own schema makes the structured fields the
//! source of truth and the command a human-pasteable convenience. If
//! the two ever disagree, that is a manifest bug to catch in review —
//! not something this loader should silently reconcile.
//!
//! ## Path placeholder resolution
//!
//! `groups[].path`, `auto_transforms`, and `anchor_parquet` may use
//! `{canonical}` (resolved from `[inputs.canonical_root].local`) or
//! `{dial_dir}` (resolved from `[inputs.dial_dir].local`). Placeholders
//! are substituted against the manifest's own `[inputs.*]` root tables.
//! `auto_transforms` paths that are repo-relative (start with `../` or
//! `benchmarks/`) are resolved relative to the manifest file's parent
//! directory so a manifest under `zensim/weights/manifests/` can point
//! at `../../../benchmarks/...`.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

/// One training group resolved from the manifest's `groups` array.
#[derive(Debug, Clone, PartialEq)]
pub struct ManifestGroup {
    pub name: String,
    /// Fully-resolved filesystem path (placeholders substituted).
    pub path: PathBuf,
    pub train_w: f64,
    pub val_w: f64,
    /// `withinref = true` → the trainer draws RankNet pairs from WITHIN one
    /// reference image (the 5th `--group` field). For per-image distortion
    /// ladders (near-lossless HF, KonJND PJND) whose within-image signal
    /// cross-image pairing drowns out.
    pub within_ref: bool,
    /// `loss = "rank" | "mse" | "both"` → which loss terms this group feeds
    /// (the group's `GroupLossMode`). Absent = rank (the default).
    pub loss_mode: Option<String>,
}

impl ManifestGroup {
    /// Render as the `NAME:PATH:TRAIN_W:VAL_W[:flag,flag]` spec the trainer's
    /// `--group` flag accepts. Mirrors `parse_group_spec`'s inverse: the 5th
    /// field is a comma-joined set of `withinref` / `rank` / `mse` / `both`.
    pub fn to_group_spec(&self) -> String {
        let mut spec = format!(
            "{}:{}:{}:{}",
            self.name,
            self.path.display(),
            self.train_w,
            self.val_w
        );
        let mut flags: Vec<&str> = Vec::new();
        if self.within_ref {
            flags.push("withinref");
        }
        if let Some(m) = self.loss_mode.as_deref()
            && matches!(m, "rank" | "mse" | "both")
        {
            flags.push(m);
        }
        if !flags.is_empty() {
            spec.push(':');
            spec.push_str(&flags.join(","));
        }
        spec
    }
}

/// One input-file record from an `[inputs.<name>]` table that carries a
/// sha256 to verify.
#[derive(Debug, Clone, PartialEq)]
pub struct ManifestInput {
    /// The `[inputs.<name>]` table key (for error messages).
    pub key: String,
    /// Fully-resolved local filesystem path.
    pub path: PathBuf,
    /// Recorded sha256 (lowercase hex).
    pub sha256: String,
    /// Recorded row count, if present.
    pub rows: Option<u64>,
    /// R2 mirror URI for this input (from its root table), if known.
    pub r2: Option<String>,
    /// Tower mirror path for this input (from its root table), if known.
    pub tower: Option<String>,
}

/// The subset of a manifest this loader maps onto the trainer's `Args`,
/// plus the input records to sha-verify. Fields are `Option` where the
/// manifest may omit them; the binary applies present values as
/// defaults beneath explicit CLI flags.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ManifestConfig {
    pub groups: Vec<ManifestGroup>,
    pub inputs: Vec<ManifestInput>,

    // [training] structured fields → Args.
    pub hidden: Option<usize>,
    pub epochs: Option<usize>,
    pub pairs_per_epoch: Option<usize>,
    pub lr: Option<f64>,
    pub l2: Option<f64>,
    pub leaky_alpha: Option<f64>,
    pub seed: Option<u64>,
    pub val_policy: Option<String>,
    pub val_aggregate: Option<String>,
    pub max_features: Option<usize>,
    pub minibatch_size: Option<usize>,
    pub out_dtype: Option<String>,
    pub target_column: Option<String>,
    pub target_scale: Option<f64>,
    pub n_hidden_layers: Option<usize>,
    pub per_sample_alpha_head: Option<bool>,
    pub mse_weight: Option<f64>,
    pub ranknet_weight: Option<f64>,
    pub monotonicity_reg: Option<f64>,
    pub monotonicity_margin: Option<f64>,
    pub tanh_output_head_scale: Option<f64>,
    pub anchor_loss_weight: Option<f64>,
    pub anchor_step_p: Option<f64>,
    pub anchor_target_score: Option<f64>,

    /// Resolved `--auto-transforms` path.
    pub auto_transforms: Option<PathBuf>,
    /// Resolved `--anchor-parquet` path.
    pub anchor_parquet: Option<PathBuf>,

    /// Masked-monotone (`--monotone-cbc`): soft sign-penalty during
    /// training + hard sign projection at bake → bake monotone↓ in every
    /// sign-safe error feature (the V39 blur>identity fix).
    pub monotone_cbc: Option<bool>,
    /// `--monotone-feature-mask`: per-feature sign mask TSV (pin_geq0 /
    /// free). Required for the masked-monotone recipe.
    pub monotone_feature_mask: Option<PathBuf>,
    /// `--monotone-strict`: drop the non-pinned (sign-flip) features
    /// instead of leaving them free.
    pub monotone_strict: Option<bool>,
    /// `--monotone-pin-during-training`: soft-monotone-keep-72 mode
    /// (#39 followup #2). Hard-projects the 300 pinned-feature W1
    /// columns to ≥0 during training (matches the bake projection
    /// exactly); leaves the 72 unpinned features FREE. Orthogonal to
    /// `monotone_strict` — when set, the "drop unpinned" branch of
    /// `monotone_strict` is suppressed.
    pub monotone_pin_during_training: Option<bool>,
    /// `--qat-fine-tune-epochs`: train the last N epochs quantization-aware
    /// (f16+zerobias STE) so the packed bake == the validated net.
    pub qat_fine_tune_epochs: Option<usize>,
    /// `--qat-tau`: QAT zerobias threshold (relative to per-layer max).
    pub qat_tau: Option<f64>,
    /// `--group-eval-cap`: per-epoch group-eval row cap (0 = full).
    pub group_eval_cap: Option<usize>,

    /// STRATEGY-2026-07-02 toggles (all optional; 0/absent = off).
    pub konjnd_aggregation_weight: Option<f64>,
    pub konjnd_aggregation_parquet: Option<String>,
    pub konjnd_aggregation_step_p: Option<f64>,
    pub ema_decay: Option<f64>,
    pub hard_pair_frac: Option<f64>,
    pub hard_pair_max_delta: Option<f64>,
    pub stratified_bands: Option<usize>,
    pub dro_eta: Option<f64>,
    pub listwise_weight: Option<f64>,
    pub listwise_size: Option<usize>,
    pub listwise_frac: Option<f64>,
    pub triplet_weight: Option<f64>,
    pub triplet_frac: Option<f64>,
    pub triplet_tau: Option<f64>,
    pub triplet_sigma: Option<f64>,
    pub triplet_stimuli: Option<String>,
    pub triplet_responses: Option<String>,

    /// `[training].trainer_commit` — the git commit of the trainer that
    /// produced the recorded bake. Reproduce-exactly requires building the
    /// trainer AT this commit: the 2026-07-01 v47 reproduction proved
    /// training is deterministic (same code + data + seed → byte-identical
    /// bake) and that unrelated-looking trainer commits break it (current
    /// main produced a 57 KB collapsed bake from the same manifest). The
    /// binary compares this against its runtime `git rev-parse HEAD` and
    /// fails loud on mismatch (same override as input-sha drift).
    pub trainer_commit: Option<String>,

    /// Ordered post-training `steps` (spline injection etc.). Recorded
    /// here so the binary can surface them — the trainer cannot run them
    /// itself (they shell to external scripts), but a reproduce run must
    /// remind the operator they exist.
    pub post_training_steps: Vec<String>,

    /// `[bake].file` resolved to an output path, if the manifest carries
    /// one. Maps to `--out` (the manifest's bake filename).
    pub out: Option<PathBuf>,

    /// `[bake].sha256` — the sha256 the manifest CLAIMS its bake has.
    ///
    /// Parsed as of 2026-07-15. Before that the field was written by hand
    /// into every manifest and read by nothing — `RawBake` carried only
    /// `file`, so serde silently dropped it. The result: 128 of 142
    /// manifests recorded `d0ef7a30…`, the sha of the *shipped Profile A
    /// bake*, because each new experiment was forked from
    /// `v47_strict_qat.toml` and never updated the outcome fields. See
    /// [`verify_bake`].
    pub bake_sha256: Option<String>,

    /// `[bake].file_bytes` — the size the manifest CLAIMS its bake has.
    /// Same history as [`Self::bake_sha256`].
    pub bake_file_bytes: Option<u64>,
}

// ---- serde wire types (a subset of the full manifest schema) ----

#[derive(Debug, Deserialize)]
struct RawManifest {
    #[serde(default)]
    bake: Option<RawBake>,
    #[serde(default)]
    training: Option<RawTraining>,
    #[serde(default)]
    inputs: Option<toml::Table>,
}

#[derive(Debug, Deserialize)]
struct RawBake {
    /// `[bake].file` — relative to the manifest's parent dir.
    #[serde(default)]
    file: Option<String>,
    /// `[bake].sha256` — of the bake at `file`. See [`verify_bake`].
    #[serde(default)]
    sha256: Option<String>,
    /// `[bake].file_bytes` — size of the bake at `file`.
    #[serde(default)]
    file_bytes: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct RawTraining {
    #[serde(default)]
    hidden: Option<usize>,
    #[serde(default)]
    epochs: Option<usize>,
    #[serde(default)]
    pairs_per_epoch: Option<usize>,
    #[serde(default)]
    lr: Option<f64>,
    #[serde(default)]
    l2: Option<f64>,
    #[serde(default)]
    leaky_alpha: Option<f64>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    val_policy: Option<String>,
    #[serde(default)]
    val_aggregate: Option<String>,
    #[serde(default)]
    max_features: Option<usize>,
    #[serde(default)]
    minibatch_size: Option<usize>,
    #[serde(default)]
    out_dtype: Option<String>,
    #[serde(default)]
    target_column: Option<String>,
    #[serde(default)]
    target_scale: Option<f64>,
    #[serde(default)]
    n_hidden_layers: Option<usize>,
    #[serde(default)]
    per_sample_alpha_head: Option<bool>,
    #[serde(default)]
    mse_weight: Option<f64>,
    #[serde(default)]
    ranknet_weight: Option<f64>,
    #[serde(default)]
    monotonicity_reg: Option<f64>,
    #[serde(default)]
    monotonicity_margin: Option<f64>,
    #[serde(default)]
    tanh_output_head_scale: Option<f64>,
    #[serde(default)]
    anchor_loss_weight: Option<f64>,
    #[serde(default)]
    anchor_step_p: Option<f64>,
    /// `anchor_target_score` may be a numeric OR the literal string
    /// `"PER-ROW (multi-band)"` when the anchor is per-row. We only map
    /// the numeric form (the per-row form is driven by `anchor_parquet`).
    #[serde(default)]
    anchor_target_score: Option<toml::Value>,
    #[serde(default)]
    groups: Vec<RawGroup>,
    #[serde(default)]
    auto_transforms: Option<String>,
    #[serde(default)]
    anchor_parquet: Option<String>,
    monotone_cbc: Option<bool>,
    monotone_feature_mask: Option<String>,
    monotone_strict: Option<bool>,
    monotone_pin_during_training: Option<bool>,
    qat_fine_tune_epochs: Option<usize>,
    qat_tau: Option<f64>,
    group_eval_cap: Option<usize>,
    konjnd_aggregation_weight: Option<f64>,
    konjnd_aggregation_parquet: Option<String>,
    konjnd_aggregation_step_p: Option<f64>,
    ema_decay: Option<f64>,
    hard_pair_frac: Option<f64>,
    hard_pair_max_delta: Option<f64>,
    stratified_bands: Option<usize>,
    dro_eta: Option<f64>,
    listwise_weight: Option<f64>,
    listwise_size: Option<usize>,
    listwise_frac: Option<f64>,
    triplet_weight: Option<f64>,
    triplet_frac: Option<f64>,
    triplet_tau: Option<f64>,
    triplet_sigma: Option<f64>,
    triplet_stimuli: Option<String>,
    triplet_responses: Option<String>,
    trainer_commit: Option<String>,
    #[serde(default)]
    steps: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RawGroup {
    name: String,
    path: String,
    train_w: f64,
    val_w: f64,
    #[serde(default)]
    within_ref: bool,
    #[serde(default)]
    loss: Option<String>,
}

/// Error type for manifest loading + sha verification.
#[derive(Debug)]
pub enum ManifestError {
    Io(String),
    Parse(String),
    Schema(String),
    /// An input file's on-disk sha256 disagreed with the manifest.
    ShaMismatch {
        key: String,
        path: PathBuf,
        expected: String,
        actual: String,
    },
    /// An input file referenced by the manifest is not present locally;
    /// `mirror` points the operator at where to fetch it.
    MissingInput {
        key: String,
        path: PathBuf,
        mirror: Option<String>,
    },
    /// `[bake].sha256` disagreed with the bytes at `[bake].file`. Unlike
    /// [`Self::ShaMismatch`] (an *input* drifted, so the run won't
    /// reproduce), this means the manifest MISDESCRIBES ITS OWN OUTPUT.
    BakeShaMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    /// `[bake].file_bytes` disagreed with the size at `[bake].file`.
    BakeSizeMismatch {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ManifestError::Io(m) => write!(f, "manifest io error: {m}"),
            ManifestError::Parse(m) => write!(f, "manifest parse error: {m}"),
            ManifestError::Schema(m) => write!(f, "manifest schema error: {m}"),
            ManifestError::ShaMismatch {
                key,
                path,
                expected,
                actual,
            } => write!(
                f,
                "manifest input {key:?} sha256 MISMATCH for {}:\n  expected (manifest): {expected}\n  actual   (on disk):  {actual}\nThe input file drifted from the recorded bake — reproduce-exactly cannot proceed. \
                 Pass --manifest-allow-sha-drift to override (NOT recommended; the produced bake will not match the shipped one).",
                path.display()
            ),
            ManifestError::MissingInput { key, path, mirror } => {
                write!(
                    f,
                    "manifest input {key:?} not found locally at {}",
                    path.display()
                )?;
                if let Some(m) = mirror {
                    write!(f, "\n  fetch it from the recorded mirror: {m}")?;
                } else {
                    write!(
                        f,
                        "\n  no R2/Tower mirror recorded in the manifest for this input"
                    )?;
                }
                Ok(())
            }
            ManifestError::BakeShaMismatch {
                path,
                expected,
                actual,
            } => write!(
                f,
                "[bake].sha256 MISMATCH for {}:\n  expected (manifest): {expected}\n  actual   (on disk):  {actual}\n\
                 The manifest misdescribes its OWN OUTPUT. Usually this means the manifest was forked from another \
                 recipe and the outcome fields ([bake].sha256, [bake].file_bytes, [eval].*) were carried over \
                 unedited. Recompute them from the bake this manifest actually produces — do NOT copy them.",
                path.display()
            ),
            ManifestError::BakeSizeMismatch {
                path,
                expected,
                actual,
            } => write!(
                f,
                "[bake].file_bytes MISMATCH for {}: manifest says {expected}, on disk {actual}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for ManifestError {}

/// Substitute `{canonical}` / `{dial_dir}` placeholders in a manifest
/// path against the resolved root strings, then resolve repo-relative
/// paths against `manifest_dir`.
///
/// Resolution rules:
/// - `{canonical}/foo` → `<canonical_root>/foo` (root from
///   `[inputs.canonical_root].local`).
/// - `{dial_dir}/foo` → `<dial_dir>/foo` (root from `[inputs.dial_dir].local`).
/// - absolute paths (`/...`) pass through unchanged.
/// - a relative path whose first component is a known repo-top dir
///   (`benchmarks/`, `scripts/`, `docs/`, `weights/`) is resolved from
///   the repo root (`manifest_dir/../../..`).
/// - everything else is treated as relative to `manifest_dir`.
fn resolve_path(
    raw: &str,
    manifest_dir: &Path,
    canonical_root: Option<&str>,
    dial_dir: Option<&str>,
) -> Result<PathBuf, ManifestError> {
    if let Some(rest) = raw.strip_prefix("{canonical}") {
        let root = canonical_root.ok_or_else(|| {
            ManifestError::Schema(format!(
                "path {raw:?} uses {{canonical}} but [inputs.canonical_root].local is missing"
            ))
        })?;
        let rest = rest.trim_start_matches('/');
        return Ok(Path::new(root).join(rest));
    }
    if let Some(rest) = raw.strip_prefix("{dial_dir}") {
        let root = dial_dir.ok_or_else(|| {
            ManifestError::Schema(format!(
                "path {raw:?} uses {{dial_dir}} but [inputs.dial_dir].local is missing"
            ))
        })?;
        let rest = rest.trim_start_matches('/');
        return Ok(Path::new(root).join(rest));
    }
    let p = Path::new(raw);
    if p.is_absolute() {
        return Ok(p.to_path_buf());
    }
    // Repo-root-relative convention: the real manifests record some
    // paths from the repo root (e.g. `benchmarks/...`) and others
    // manifest-relative (e.g. `../../../benchmarks/...`). Manifests live
    // at `<repo>/zensim/weights/manifests/`, so the repo root is three
    // levels up. A relative path whose FIRST component is a known
    // repo-top directory is resolved from the repo root; everything else
    // is manifest-relative. This reconciles the two conventions the
    // shipped manifests use for the SAME files without a filesystem probe.
    const REPO_TOP_DIRS: [&str; 4] = ["benchmarks", "scripts", "docs", "weights"];
    if let Some(std::path::Component::Normal(first)) = p.components().next()
        && REPO_TOP_DIRS
            .iter()
            .any(|d| first == std::ffi::OsStr::new(d))
    {
        // repo root = manifest_dir/../../..
        return Ok(manifest_dir.join("../../..").join(p));
    }
    Ok(manifest_dir.join(p))
}

/// Pull a `local` string out of an `[inputs.<root>]` table, if present.
fn root_local<'a>(inputs: Option<&'a toml::Table>, root: &str) -> Option<&'a str> {
    inputs?.get(root)?.as_table()?.get("local")?.as_str()
}

/// Parse a manifest TOML file into a [`ManifestConfig`].
///
/// Maps the structured `[training]` fields, the `groups` array, and the
/// `[inputs.<name>]` blocks (those carrying a `sha256`) into resolved
/// filesystem paths. Does NOT touch the filesystem beyond reading the
/// manifest itself — sha verification is a separate, explicit step
/// ([`verify_inputs`]).
pub fn parse_manifest(path: &Path) -> Result<ManifestConfig, ManifestError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| ManifestError::Io(format!("read {}: {e}", path.display())))?;
    parse_manifest_str(&text, path)
}

/// Parse a manifest from an in-memory string + the manifest's path (used
/// for `manifest_dir` resolution + error messages). Split out from
/// [`parse_manifest`] so tests can drive it with synthetic content.
pub fn parse_manifest_str(text: &str, path: &Path) -> Result<ManifestConfig, ManifestError> {
    let raw: RawManifest = toml::from_str(text)
        .map_err(|e| ManifestError::Parse(format!("{}: {e}", path.display())))?;
    let manifest_dir = path.parent().unwrap_or_else(|| Path::new("."));

    let inputs_tbl = raw.inputs.as_ref();
    let canonical_root = root_local(inputs_tbl, "canonical_root");
    let dial_dir = root_local(inputs_tbl, "dial_dir");

    let mut cfg = ManifestConfig::default();

    if let Some(bake) = raw.bake.as_ref() {
        if let Some(file) = bake.file.as_ref() {
            cfg.out = Some(resolve_path(file, manifest_dir, canonical_root, dial_dir)?);
        }
        // Normalize to lowercase hex: `sha256_file` emits lowercase, so an
        // uppercase manifest entry would otherwise read as drift.
        //
        // `sha256 = ""` and `file_bytes = 0` are PLACEHOLDERS, not claims —
        // 8 manifests carry them, meaning "not recorded yet" (a bake is
        // never zero bytes). Normalizing them to `None` here keeps
        // "unrecorded" and "recorded wrong" as distinct states, so
        // `verify_bake` stays silent on the former and loud on the latter.
        cfg.bake_sha256 = bake
            .sha256
            .as_deref()
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty());
        cfg.bake_file_bytes = bake.file_bytes.filter(|&b| b != 0);
    }

    if let Some(t) = raw.training {
        cfg.hidden = t.hidden;
        cfg.epochs = t.epochs;
        cfg.pairs_per_epoch = t.pairs_per_epoch;
        cfg.lr = t.lr;
        cfg.l2 = t.l2;
        cfg.leaky_alpha = t.leaky_alpha;
        cfg.seed = t.seed;
        cfg.val_policy = t.val_policy;
        cfg.val_aggregate = t.val_aggregate;
        cfg.max_features = t.max_features;
        cfg.minibatch_size = t.minibatch_size;
        cfg.out_dtype = t.out_dtype;
        cfg.target_column = t.target_column;
        cfg.target_scale = t.target_scale;
        cfg.n_hidden_layers = t.n_hidden_layers;
        cfg.per_sample_alpha_head = t.per_sample_alpha_head;
        cfg.mse_weight = t.mse_weight;
        cfg.ranknet_weight = t.ranknet_weight;
        cfg.monotonicity_reg = t.monotonicity_reg;
        cfg.monotonicity_margin = t.monotonicity_margin;
        cfg.tanh_output_head_scale = t.tanh_output_head_scale;
        cfg.anchor_loss_weight = t.anchor_loss_weight;
        cfg.anchor_step_p = t.anchor_step_p;
        // anchor_target_score: only map the numeric form. The
        // "PER-ROW (multi-band)" string means the anchor is per-row
        // (driven by anchor_parquet), so there is no scalar to apply.
        cfg.anchor_target_score = t.anchor_target_score.and_then(|v| match v {
            toml::Value::Integer(i) => Some(i as f64),
            toml::Value::Float(fl) => Some(fl),
            _ => None,
        });
        cfg.post_training_steps = t.steps;

        for g in t.groups {
            let path = resolve_path(&g.path, manifest_dir, canonical_root, dial_dir)?;
            cfg.groups.push(ManifestGroup {
                name: g.name,
                path,
                train_w: g.train_w,
                val_w: g.val_w,
                within_ref: g.within_ref,
                loss_mode: g.loss,
            });
        }

        if let Some(at) = t.auto_transforms {
            cfg.auto_transforms = Some(resolve_path(&at, manifest_dir, canonical_root, dial_dir)?);
        }
        if let Some(ap) = t.anchor_parquet {
            cfg.anchor_parquet = Some(resolve_path(&ap, manifest_dir, canonical_root, dial_dir)?);
        }
        cfg.monotone_cbc = t.monotone_cbc;
        if let Some(mm) = t.monotone_feature_mask {
            cfg.monotone_feature_mask =
                Some(resolve_path(&mm, manifest_dir, canonical_root, dial_dir)?);
        }
        cfg.monotone_strict = t.monotone_strict;
        cfg.monotone_pin_during_training = t.monotone_pin_during_training;
        cfg.qat_fine_tune_epochs = t.qat_fine_tune_epochs;
        cfg.qat_tau = t.qat_tau;
        cfg.group_eval_cap = t.group_eval_cap;
        cfg.konjnd_aggregation_weight = t.konjnd_aggregation_weight;
        cfg.konjnd_aggregation_parquet = t.konjnd_aggregation_parquet.clone();
        cfg.konjnd_aggregation_step_p = t.konjnd_aggregation_step_p;
        cfg.ema_decay = t.ema_decay;
        cfg.hard_pair_frac = t.hard_pair_frac;
        cfg.hard_pair_max_delta = t.hard_pair_max_delta;
        cfg.stratified_bands = t.stratified_bands;
        cfg.dro_eta = t.dro_eta;
        cfg.listwise_weight = t.listwise_weight;
        cfg.listwise_size = t.listwise_size;
        cfg.listwise_frac = t.listwise_frac;
        cfg.triplet_weight = t.triplet_weight;
        cfg.triplet_frac = t.triplet_frac;
        cfg.triplet_tau = t.triplet_tau;
        cfg.triplet_sigma = t.triplet_sigma;
        cfg.triplet_stimuli = t.triplet_stimuli.clone();
        cfg.triplet_responses = t.triplet_responses.clone();
        cfg.trainer_commit = t.trainer_commit;
    }

    // Collect every [inputs.<name>] table that carries a sha256 — those
    // are the files reproduce-exactly must verify. Root tables
    // (canonical_root / dial_dir) carry no sha256 and are skipped.
    if let Some(tbl) = inputs_tbl {
        for (key, val) in tbl.iter() {
            let Some(t) = val.as_table() else { continue };
            let Some(sha) = t.get("sha256").and_then(|v| v.as_str()) else {
                continue;
            };
            let Some(raw_path) = t.get("path").and_then(|v| v.as_str()) else {
                return Err(ManifestError::Schema(format!(
                    "[inputs.{key}] has sha256 but no path"
                )));
            };
            let resolved = resolve_path(raw_path, manifest_dir, canonical_root, dial_dir)?;
            // Mirror hints: prefer this table's own r2/tower, else the
            // canonical_root's (since {canonical} paths live under it).
            let r2 = t
                .get("r2")
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .or_else(|| {
                    if raw_path.starts_with("{canonical}") {
                        inputs_tbl?
                            .get("canonical_root")?
                            .as_table()?
                            .get("r2")?
                            .as_str()
                            .map(str::to_string)
                    } else {
                        None
                    }
                });
            let tower = t
                .get("tower")
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .or_else(|| {
                    if raw_path.starts_with("{canonical}") {
                        inputs_tbl?
                            .get("canonical_root")?
                            .as_table()?
                            .get("tower")?
                            .as_str()
                            .map(str::to_string)
                    } else {
                        None
                    }
                });
            cfg.inputs.push(ManifestInput {
                key: key.clone(),
                path: resolved,
                sha256: sha.to_ascii_lowercase(),
                rows: t.get("rows").and_then(|v| v.as_integer()).map(|i| i as u64),
                r2,
                tower,
            });
        }
        // Deterministic order for stable error messages + tests.
        cfg.inputs.sort_by(|a, b| a.key.cmp(&b.key));
    }

    Ok(cfg)
}

/// Compute the lowercase-hex sha256 of a file's bytes.
///
/// Streams the file through a fixed 1 MiB buffer: the digest is the same
/// function of the same bytes as hashing a whole-file read, but a lane no
/// longer holds the largest training table (1 GB+ on the wide v2 recipe) in
/// memory while it hashes it.
pub fn sha256_file(path: &Path) -> Result<String, ManifestError> {
    use std::io::Read;
    let io_err = |e: std::io::Error| ManifestError::Io(format!("read {}: {e}", path.display()));
    let mut file = std::fs::File::open(path).map_err(io_err)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf).map_err(io_err)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

pub(crate) fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(char::from_digit((b >> 4) as u32, 16).unwrap());
        s.push(char::from_digit((b & 0x0f) as u32, 16).unwrap());
    }
    s
}

/// Verify that `[bake].sha256` / `[bake].file_bytes` actually describe the
/// bytes at `[bake].file`.
///
/// **The counterpart to [`verify_inputs`], and the one that was missing.**
/// `verify_inputs` has always checked that the manifest's *inputs* are the
/// bytes it claims; nothing ever checked its *output*. Until 2026-07-15 the
/// parser did not even read `[bake].sha256` — `RawBake` carried only `file`,
/// so serde dropped it and the field was inert prose in a TOML costume.
///
/// What that cost: of 142 manifests, **128 recorded the same sha256**
/// (`d0ef7a30…` — the shipped Profile A bake) and the same `[eval]` block
/// (`cid22_srocc = 0.8657` — Profile A's number). Each new experiment was
/// forked from `v47_strict_qat.toml`, had its seed/groups/hyperparams
/// edited, and kept the outcome fields verbatim. A manifest's whole job is
/// to identify a bake; 128 of them identified someone else's.
///
/// Returns `Ok(vec![])` when the manifest carries no `[bake].sha256`, or
/// when the bake is not on disk (probe bakes live on `/mnt/v`, which CI
/// does not mount — absence is not evidence of drift). With
/// `allow_sha_drift`, mismatches are downgraded to warnings.
pub fn verify_bake(
    cfg: &ManifestConfig,
    allow_sha_drift: bool,
) -> Result<Vec<String>, ManifestError> {
    let mut warnings = Vec::new();
    let Some(path) = cfg.out.as_ref() else {
        return Ok(warnings);
    };
    // Not-on-disk is silence, not failure: a manifest may describe a bake
    // that lives on /mnt/v or was never kept. Only *present* bytes are
    // checked, so this stays CI-safe.
    if !path.exists() {
        return Ok(warnings);
    }
    if let Some(expected) = cfg.bake_sha256.as_ref() {
        let actual = sha256_file(path)?;
        if &actual != expected {
            if allow_sha_drift {
                warnings.push(format!(
                    "[bake].sha256 drift ALLOWED via --manifest-allow-sha-drift: \
                     expected {expected} got {actual} for {}",
                    path.display()
                ));
            } else {
                return Err(ManifestError::BakeShaMismatch {
                    path: path.clone(),
                    expected: expected.clone(),
                    actual,
                });
            }
        }
    }
    if let Some(expected) = cfg.bake_file_bytes {
        let actual = std::fs::metadata(path)
            .map_err(|e| ManifestError::Io(format!("stat {}: {e}", path.display())))?
            .len();
        if actual != expected {
            if allow_sha_drift {
                warnings.push(format!(
                    "[bake].file_bytes drift ALLOWED: expected {expected} got {actual} for {}",
                    path.display()
                ));
            } else {
                return Err(ManifestError::BakeSizeMismatch {
                    path: path.clone(),
                    expected,
                    actual,
                });
            }
        }
    }
    Ok(warnings)
}

/// Verify every manifest input file's sha256 against its on-disk bytes.
///
/// This is the load-bearing reproduce-exactly check: if a referenced
/// file drifted, the produced bake will NOT match the shipped one, so we
/// FAIL LOUD with the mismatch. Missing local files produce a
/// [`ManifestError::MissingInput`] pointing at the recorded mirror.
///
/// When `allow_sha_drift` is `true` (the `--manifest-allow-sha-drift`
/// escape hatch — OFF by default), mismatches are downgraded to the
/// returned `warnings` vec instead of erroring; missing files still
/// error (we can't train on a file that isn't there).
pub fn verify_inputs(
    inputs: &[ManifestInput],
    allow_sha_drift: bool,
) -> Result<Vec<String>, ManifestError> {
    let mut warnings = Vec::new();
    for inp in inputs {
        if !inp.path.exists() {
            return Err(ManifestError::MissingInput {
                key: inp.key.clone(),
                path: inp.path.clone(),
                mirror: inp.r2.clone().or_else(|| inp.tower.clone()),
            });
        }
        let actual = sha256_file(&inp.path)?;
        if actual != inp.sha256 {
            if allow_sha_drift {
                warnings.push(format!(
                    "input {:?} sha256 drift ALLOWED via --manifest-allow-sha-drift: \
                     expected {} got {} for {}",
                    inp.key,
                    inp.sha256,
                    actual,
                    inp.path.display()
                ));
            } else {
                return Err(ManifestError::ShaMismatch {
                    key: inp.key.clone(),
                    path: inp.path.clone(),
                    expected: inp.sha256.clone(),
                    actual,
                });
            }
        }
    }
    Ok(warnings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tmp(name: &str, contents: &[u8]) -> PathBuf {
        // Per project rules /tmp is fine for one-shot test scratch.
        let dir = std::env::temp_dir().join(format!(
            "zensim_manifest_test_{}_{}",
            std::process::id(),
            name
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(contents).unwrap();
        p
    }

    #[test]
    fn resolve_canonical_placeholder() {
        let dir = Path::new("/repo/zensim/weights/manifests");
        let got = resolve_path(
            "{canonical}/safesyn.parquet",
            dir,
            Some("/mnt/v/train"),
            None,
        )
        .unwrap();
        assert_eq!(got, PathBuf::from("/mnt/v/train/safesyn.parquet"));
    }

    #[test]
    fn resolve_repo_relative_against_manifest_dir() {
        let dir = Path::new("/repo/zensim/weights/manifests");
        // `../...` is a ParentDir-led path → stays manifest-relative.
        let got = resolve_path("../../../benchmarks/x.tsv", dir, None, None).unwrap();
        assert_eq!(
            got,
            PathBuf::from("/repo/zensim/weights/manifests/../../../benchmarks/x.tsv")
        );
    }

    #[test]
    fn resolve_repo_top_dir_from_repo_root() {
        // A bare `benchmarks/...` (the convention the [inputs.*] blocks
        // use) resolves from the repo root (manifest_dir/../../..) so it
        // points at the SAME file as the [training] field's
        // `../../../benchmarks/...`.
        let dir = Path::new("/repo/zensim/weights/manifests");
        let got = resolve_path("benchmarks/screen.tsv", dir, None, None).unwrap();
        assert_eq!(
            got,
            PathBuf::from("/repo/zensim/weights/manifests/../../../benchmarks/screen.tsv")
        );
        // A non-repo-top relative path stays manifest-relative.
        let local = resolve_path("sibling.bin", dir, None, None).unwrap();
        assert_eq!(
            local,
            PathBuf::from("/repo/zensim/weights/manifests/sibling.bin")
        );
    }

    #[test]
    fn missing_canonical_root_for_placeholder_errors() {
        let dir = Path::new("/repo");
        let err = resolve_path("{canonical}/x.parquet", dir, None, None).unwrap_err();
        assert!(matches!(err, ManifestError::Schema(_)));
    }

    #[test]
    fn sha256_streaming_equals_one_shot_across_buffer_boundaries() {
        // 2.5 MiB + 7 bytes: crosses two full 1 MiB reads plus a short tail.
        let mut bytes = Vec::with_capacity((5 << 19) + 7);
        let mut x = 0x9E37_79B9u32;
        for _ in 0..(5 << 19) + 7 {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            bytes.push((x >> 24) as u8);
        }
        let p = write_tmp("multi_chunk.bin", &bytes);
        let mut one_shot = Sha256::new();
        one_shot.update(&bytes);
        assert_eq!(sha256_file(&p).unwrap(), hex_lower(&one_shot.finalize()));
        assert!(matches!(
            sha256_file(Path::new("/nonexistent/zensim-sha-test")),
            Err(ManifestError::Io(_))
        ));
    }

    #[test]
    fn sha256_matches_known_vector() {
        // sha256("") is the well-known empty-input digest.
        let p = write_tmp("empty.bin", b"");
        let got = sha256_file(&p).unwrap();
        assert_eq!(
            got,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    /// The check that was missing until 2026-07-15: does `[bake].sha256`
    /// describe the bytes at `[bake].file`?
    #[test]
    fn verify_bake_passes_on_match_fails_on_drift() {
        let p = write_tmp("bake.bin", b"pretend this is a znpr v3 bake");
        let real_sha = sha256_file(&p).unwrap();
        let real_len = std::fs::metadata(&p).unwrap().len();

        let good = ManifestConfig {
            out: Some(p.clone()),
            bake_sha256: Some(real_sha.clone()),
            bake_file_bytes: Some(real_len),
            ..Default::default()
        };
        assert!(verify_bake(&good, false).unwrap().is_empty());

        // The fork signature: outcome fields carried over from the parent
        // manifest, so the sha names a DIFFERENT bake.
        let forked = ManifestConfig {
            out: Some(p.clone()),
            bake_sha256: Some("d0ef7a30".to_string() + &"0".repeat(56)),
            bake_file_bytes: Some(real_len),
            ..Default::default()
        };
        let err = verify_bake(&forked, false).unwrap_err();
        assert!(
            matches!(err, ManifestError::BakeShaMismatch { .. }),
            "{err}"
        );
        // The message must name the fork, since that is the actual cause.
        assert!(err.to_string().contains("misdescribes its OWN OUTPUT"));
        let warns = verify_bake(&forked, true).unwrap();
        assert_eq!(warns.len(), 1);
        assert!(warns[0].contains("drift ALLOWED"));

        let wrong_size = ManifestConfig {
            out: Some(p.clone()),
            bake_sha256: Some(real_sha),
            bake_file_bytes: Some(real_len + 1),
            ..Default::default()
        };
        assert!(matches!(
            verify_bake(&wrong_size, false).unwrap_err(),
            ManifestError::BakeSizeMismatch { .. }
        ));
    }

    /// Absence must be silence, not failure — probe manifests point at
    /// `/mnt/v`, which CI does not mount. A manifest with no recorded sha
    /// is also fine (nothing was claimed, so nothing can be wrong).
    #[test]
    fn verify_bake_is_silent_when_nothing_is_claimed_or_present() {
        // No [bake] at all.
        assert!(
            verify_bake(&ManifestConfig::default(), false)
                .unwrap()
                .is_empty()
        );
        // Claimed, but the bake is not on disk (the /mnt/v case).
        let absent = ManifestConfig {
            out: Some(PathBuf::from("/nonexistent/probe/bake.bin")),
            bake_sha256: Some("0".repeat(64)),
            bake_file_bytes: Some(123),
            ..Default::default()
        };
        assert!(verify_bake(&absent, false).unwrap().is_empty());
        // Present, but the manifest claims nothing about it.
        let p = write_tmp("unclaimed.bin", b"bytes");
        let unclaimed = ManifestConfig {
            out: Some(p),
            ..Default::default()
        };
        assert!(verify_bake(&unclaimed, false).unwrap().is_empty());
    }

    /// `[bake].sha256` must actually reach `ManifestConfig` — the exact bug
    /// that made 128 wrong hashes invisible was that `RawBake` carried only
    /// `file`, so serde dropped this field on the floor.
    #[test]
    fn bake_sha256_and_file_bytes_are_parsed_not_dropped() {
        let text = r#"
[bake]
name = "zensim-a"
file = "../some_bake.bin"
sha256 = "D0EF7A3054D1ED9E70086D306CDA69B71FC95072C6EF3351F362F27DA096D4FC"
file_bytes = 27316
arch = "372 -> 128 -> 64"
"#;
        let cfg = parse_manifest_str(text, Path::new("/tmp/m/x.toml")).unwrap();
        assert_eq!(
            cfg.bake_sha256.as_deref(),
            Some("d0ef7a3054d1ed9e70086d306cda69b71fc95072c6ef3351f362f27da096d4fc"),
            "sha256 must be parsed AND normalized to lowercase hex (sha256_file emits lowercase, \
             so an uppercase manifest entry would otherwise false-positive as drift)"
        );
        assert_eq!(cfg.bake_file_bytes, Some(27316));
    }

    /// `sha256 = ""` / `file_bytes = 0` are placeholders for "not recorded
    /// yet" (8 manifests carry them). They must normalize to `None`, so
    /// "unrecorded" stays distinct from "recorded wrong" — otherwise
    /// `verify_bake` reports every unfilled manifest as drift and the gate
    /// becomes noise nobody reads.
    #[test]
    fn empty_bake_sha_and_zero_bytes_are_placeholders_not_claims() {
        let text = r#"
[bake]
file = "../probe.bin"
sha256 = ""
file_bytes = 0
"#;
        let cfg = parse_manifest_str(text, Path::new("/tmp/m/x.toml")).unwrap();
        assert_eq!(
            cfg.bake_sha256, None,
            "empty sha256 must not read as a claim"
        );
        assert_eq!(
            cfg.bake_file_bytes, None,
            "zero file_bytes must not read as a claim"
        );
    }

    #[test]
    fn verify_inputs_passes_on_match_fails_on_drift() {
        let p = write_tmp("data.parquet", b"hello world");
        let real_sha = sha256_file(&p).unwrap();
        let good = vec![ManifestInput {
            key: "data".into(),
            path: p.clone(),
            sha256: real_sha.clone(),
            rows: Some(3),
            r2: None,
            tower: None,
        }];
        assert!(verify_inputs(&good, false).unwrap().is_empty());

        let bad = vec![ManifestInput {
            key: "data".into(),
            path: p.clone(),
            sha256: "0".repeat(64),
            rows: Some(3),
            r2: Some("s3://bucket/data.parquet".into()),
            tower: None,
        }];
        // Hard check rejects drift...
        let err = verify_inputs(&bad, false).unwrap_err();
        assert!(matches!(err, ManifestError::ShaMismatch { .. }));
        // ...escape hatch downgrades to a warning.
        let warns = verify_inputs(&bad, true).unwrap();
        assert_eq!(warns.len(), 1);
        assert!(warns[0].contains("drift ALLOWED"));
    }

    #[test]
    fn verify_inputs_missing_file_points_at_mirror() {
        let missing = vec![ManifestInput {
            key: "safesyn".into(),
            path: PathBuf::from("/nonexistent/zensim/safesyn.parquet"),
            sha256: "0".repeat(64),
            rows: None,
            r2: Some("s3://zentrain/canonical/safesyn.parquet".into()),
            tower: None,
        }];
        // Missing files error even with drift allowed.
        let err = verify_inputs(&missing, true).unwrap_err();
        match err {
            ManifestError::MissingInput { mirror, .. } => {
                assert_eq!(
                    mirror.as_deref(),
                    Some("s3://zentrain/canonical/safesyn.parquet")
                );
            }
            other => panic!("expected MissingInput, got {other:?}"),
        }
    }

    /// The 5th group-spec field (withinref / loss mode) must round-trip from
    /// the manifest through `to_group_spec` to the form `parse_group_spec`
    /// consumes — the path iteration 2's HF/KonJND :withinref fixes ride on.
    #[test]
    fn group_spec_renders_withinref_and_loss_mode() {
        let plain = ManifestGroup {
            name: "safesyn".into(),
            path: PathBuf::from("/x/s.parquet"),
            train_w: 1.0,
            val_w: 0.5,
            within_ref: false,
            loss_mode: None,
        };
        assert_eq!(plain.to_group_spec(), "safesyn:/x/s.parquet:1:0.5");
        let wr = ManifestGroup {
            name: "hf".into(),
            path: PathBuf::from("/x/hf.parquet"),
            train_w: 0.5,
            val_w: 0.0,
            within_ref: true,
            loss_mode: None,
        };
        assert_eq!(wr.to_group_spec(), "hf:/x/hf.parquet:0.5:0:withinref");
        let both = ManifestGroup {
            name: "s".into(),
            path: PathBuf::from("/x/s.parquet"),
            train_w: 1.0,
            val_w: 0.0,
            within_ref: true,
            loss_mode: Some("mse".into()),
        };
        assert_eq!(both.to_group_spec(), "s:/x/s.parquet:1:0:withinref,mse");
        // an unknown loss string is dropped, not emitted as a bogus flag
        let bad = ManifestGroup {
            name: "s".into(),
            path: PathBuf::from("/x/s.parquet"),
            train_w: 1.0,
            val_w: 0.0,
            within_ref: false,
            loss_mode: Some("garbage".into()),
        };
        assert_eq!(bad.to_group_spec(), "s:/x/s.parquet:1:0");
    }

    /// A manifest group parses `within_ref` + `loss` from TOML.
    #[test]
    fn manifest_group_parses_withinref_and_loss() {
        let text = r#"
[[training.groups]]
name = "hf"
path = "/abs/hf.parquet"
train_w = 0.5
val_w = 0.0
within_ref = true
loss = "mse"
"#;
        let cfg = parse_manifest_str(text, Path::new("/tmp/m/x.toml")).unwrap();
        assert_eq!(cfg.groups.len(), 1);
        assert!(cfg.groups[0].within_ref);
        assert_eq!(cfg.groups[0].loss_mode.as_deref(), Some("mse"));
        assert_eq!(
            cfg.groups[0].to_group_spec(),
            "hf:/abs/hf.parquet:0.5:0:withinref,mse"
        );
    }
}
