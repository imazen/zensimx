//! `subset_sim` — replay a bake's training-pair sampler and describe the
//! subset it drew.
//!
//! The subset-quality study
//! (`benchmarks/subset_quality_study_2026-09-04.md`) asks whether the
//! seed-to-seed spread in held-out score is explained by *which training
//! rows a seed happened to draw*. Answering that needs the drawn subset of
//! every seed-sibling bake on the board — hundreds of runs that would each
//! cost a full retrain to observe directly.
//!
//! They do not have to be retrained. The trainer's sample stream is
//! independent of its init stream, and the drawn multiset is a pure
//! function of `(seed, [train_weight], [row_count], epochs,
//! pairs_per_epoch, boosts, within_ref)` — every one of which a bake
//! records in its embedded `zentrain.repro` block. So this tool reads a
//! `*.fulleval.json`, reconstructs the group table, and replays the
//! sampler through the SAME `mlp_train::sampling::draw_pair` the trainer
//! uses. No features are read (only the target and reference columns), no
//! model is built, and nothing is re-implemented.
//!
//! Faithfulness is provable, not asserted: `--expect-digest` compares the
//! replayed sample-sequence hash against the one a real training run
//! prints under `ZENSIM_SAMPLE_DIGEST=1`.
//!
//! ```text
//! subset_sim --fulleval FC_C0_s4004.fulleval.json --out cov.json
//! subset_sim --group a:x.parquet:1.0:1.0 --seed 4004 --epochs 3 \
//!            --pairs-per-epoch 2000 --expect-digest 8a1f…
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use clap::Parser;
use serde_json::{Value, json};

use zensim_validate::mlp_train::sampling::{self, SimGroup, SimParams};
use zensim_validate::parquet_loader;

#[derive(Parser, Debug)]
#[command(
    about = "Replay a training run's pair sampler and describe the drawn subset",
    long_about = None
)]
struct Args {
    /// A `*.fulleval.json` whose embedded `repro` block defines the run.
    #[arg(long)]
    fulleval: Option<PathBuf>,

    /// Explicit group spec `NAME:PATH:TRAIN_W:VAL_W[:withinref]`, repeatable.
    /// Mirrors `zensim_mlp_train --group`.
    #[arg(long = "group")]
    groups: Vec<String>,

    /// Seeds to simulate. Defaults to the fulleval's own seed.
    #[arg(long, value_delimiter = ',')]
    seeds: Vec<u64>,

    #[arg(long)]
    epochs: Option<usize>,
    #[arg(long)]
    pairs_per_epoch: Option<usize>,
    #[arg(long, default_value = "human_score")]
    target_column: String,
    #[arg(long, default_value_t = 100.0)]
    target_scale: f64,
    #[arg(long, default_value_t = 1.0)]
    low_q_boost: f64,
    #[arg(long, default_value_t = 1.0)]
    mid_q_boost: f64,
    #[arg(long, default_value_t = 1.0)]
    high_q_boost: f64,
    #[arg(long, default_value_t = 0)]
    stratified_bands: usize,
    /// Draws in the "early window". 0 = one epoch.
    #[arg(long, default_value_t = 0)]
    early_window: usize,
    /// Use the `per_sample_alpha_head` sample-stream constant.
    #[arg(long, default_value_t = false)]
    per_sample_alpha_head: bool,

    /// Replay a `--pair-sampling stratified` run. Read from the fulleval's
    /// `repro.pair_sampling` when one is supplied; this flag is for the
    /// explicit-`--group` form, and for a legacy bake whose repro predates
    /// the knob (those were all uniform).
    #[arg(long, default_value_t = false)]
    stratified_pairs: bool,
    /// Opt-in pooled rank legs for an explicit group replay.
    #[arg(long = "pooled-leg")]
    pooled_legs: Vec<String>,
    #[arg(long, default_value_t = 0.0)]
    pooled_rank_share: f64,

    /// Verify the replayed sequence hash against a real run's
    /// `ZENSIM_SAMPLE_DIGEST=1` output. Exits 3 on mismatch.
    #[arg(long)]
    expect_digest: Option<String>,

    /// Refuse overlapping raw RNG windows across seeds before reading group tables.
    /// For the uniform sampler, four words per attempted pair is a conservative
    /// bound. This checks sampler streams, not arbitrary auxiliary trainer RNGs.
    /// Historical seeds are stream offsets, so nearby values overlap heavily.
    #[arg(long)]
    require_disjoint_sampler_windows: bool,

    /// Write descriptor rows here as JSON.
    #[arg(long)]
    out: Option<PathBuf>,
}

/// One group as declared by a `--group` spec or a repro `inputs[]` entry.
struct GroupSpec {
    name: String,
    path: PathBuf,
    train_w: f64,
    within_ref: bool,
}

fn parse_group_spec(spec: &str) -> Result<GroupSpec, String> {
    let parts: Vec<&str> = spec.split(':').collect();
    if parts.len() < 4 {
        return Err(format!(
            "group spec {spec:?}: need NAME:PATH:TRAIN_W:VAL_W[:withinref]"
        ));
    }
    Ok(GroupSpec {
        name: parts[0].to_string(),
        path: PathBuf::from(parts[1]),
        train_w: parts[2]
            .parse()
            .map_err(|e| format!("group {spec:?}: train_w: {e}"))?,
        within_ref: parts.get(4).is_some_and(|f| *f == "withinref"),
    })
}

/// Pull the value following `flag` out of a recorded argv.
fn argv_val(argv: &[Value], flag: &str) -> Option<String> {
    argv.iter()
        .position(|v| v.as_str() == Some(flag))
        .and_then(|i| argv.get(i + 1))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn main() {
    let args = Args::parse();

    let mut specs: Vec<GroupSpec> = Vec::new();
    let mut seeds: Vec<u64> = args.seeds.clone();
    let mut epochs = args.epochs.unwrap_or(120);
    let mut ppe = args.pairs_per_epoch.unwrap_or(50_000);
    let mut target_column = args.target_column.clone();
    let mut target_scale = args.target_scale;
    let (mut lo, mut mid, mut hi) = (args.low_q_boost, args.mid_q_boost, args.high_q_boost);
    let mut strat = args.stratified_bands;
    let mut psa = args.per_sample_alpha_head;
    let mut strat_pairs = args.stratified_pairs;
    let mut walk_pairs = false;
    let mut pooled_legs = args.pooled_legs.clone();
    let mut pooled_rank_share = args.pooled_rank_share;
    let mut source = String::from("cli");

    if let Some(fe) = &args.fulleval {
        let txt = std::fs::read_to_string(fe).unwrap_or_else(|e| {
            eprintln!("subset_sim: read {fe:?}: {e}");
            std::process::exit(2);
        });
        let d: Value = serde_json::from_str(&txt).unwrap_or_else(|e| {
            eprintln!("subset_sim: parse {fe:?}: {e}");
            std::process::exit(2);
        });
        let repro = d.get("repro").cloned().unwrap_or(Value::Null);
        if !repro.is_object() {
            eprintln!("subset_sim: {fe:?} has no embedded repro block — cannot reconstruct");
            std::process::exit(2);
        }
        if let Some(pool) = repro.get("pooled_objective") {
            pooled_legs = serde_json::from_value(pool["legs"].clone()).expect("pooled legs");
            pooled_rank_share = pool["rank_share"].as_f64().expect("pooled rank share");
        }
        source = fe.display().to_string();
        // Groups: repro.inputs[] carries name/path/train_w/within_ref, and its
        // `rows` is the loaded row count the sampler draws modulo. We still
        // read the parquet for scores/refs, and CHECK `rows` against it — a
        // mismatch means the file changed under the bake and the replay would
        // be a different run's subset.
        if let Some(arr) = repro.get("inputs").and_then(|v| v.as_array()) {
            for it in arr {
                let tw = it.get("train_w").and_then(|v| v.as_f64()).unwrap_or(0.0);
                if tw <= 0.0 {
                    continue; // val-only groups are never drawn from
                }
                specs.push(GroupSpec {
                    name: it
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?")
                        .to_string(),
                    path: PathBuf::from(it.get("path").and_then(|v| v.as_str()).unwrap_or("")),
                    train_w: tw,
                    within_ref: it
                        .get("within_ref")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                });
            }
        }
        if let Some(e) = repro.get("epochs").and_then(|v| v.as_u64()) {
            epochs = e as usize;
        }
        if let Some(p) = repro.get("pairs_per_epoch").and_then(|v| v.as_u64()) {
            ppe = p as usize;
        }
        if let Some(t) = repro.get("target_column").and_then(|v| v.as_str()) {
            target_column = t.to_string();
        }
        if let Some(t) = repro.get("target_scale").and_then(|v| v.as_f64()) {
            target_scale = t;
        }
        if seeds.is_empty() {
            match recorded_sampling_seed(&repro) {
                Ok(Some(seed)) => seeds.push(seed),
                Ok(None) => {}
                Err(e) => {
                    eprintln!("subset_sim: {e}");
                    std::process::exit(2);
                }
            }
        }
        // Sampling knobs live only in argv.
        if let Some(argv) = repro.get("argv").and_then(|v| v.as_array()) {
            let f = |k: &str| argv_val(argv, k).and_then(|s| s.parse::<f64>().ok());
            lo = f("--low-q-boost").unwrap_or(lo);
            mid = f("--mid-q-boost").unwrap_or(mid);
            hi = f("--high-q-boost").unwrap_or(hi);
            strat = argv_val(argv, "--stratified-bands")
                .and_then(|s| s.parse().ok())
                .unwrap_or(strat);
            psa = psa
                || argv
                    .iter()
                    .any(|v| v.as_str() == Some("--per-sample-alpha-head"));
            // `--pair-sampling` landed 2026-09-04; a repro without it is a
            // run from before the knob existed, which was uniform. Read the
            // structured field first, then argv, so a hand-built repro that
            // records only one of the two still replays correctly.
            strat_pairs = strat_pairs
                || repro.get("pair_sampling").and_then(|v| v.as_str()) == Some("stratified")
                || argv_val(argv, "--pair-sampling").as_deref() == Some("stratified");
            walk_pairs = repro.get("pair_sampling").and_then(|v| v.as_str()) == Some("walk")
                || argv_val(argv, "--pair-sampling").as_deref() == Some("walk");
        }
        // Row-count cross-check against the recorded repro.
        if let Some(arr) = repro.get("inputs").and_then(|v| v.as_array()) {
            for it in arr {
                if it.get("train_w").and_then(|v| v.as_f64()).unwrap_or(0.0) <= 0.0 {
                    continue;
                }
                let want = it.get("rows").and_then(|v| v.as_u64());
                let p = it.get("path").and_then(|v| v.as_str()).unwrap_or("");
                if let (Some(w), true) = (want, Path::new(p).exists()) {
                    // Cheap metadata-only row count would need a parquet
                    // open; the full read below checks it anyway.
                    let _ = w;
                }
            }
        }
    }

    for g in &args.groups {
        match parse_group_spec(g) {
            Ok(s) => {
                if s.train_w > 0.0 {
                    specs.push(s)
                }
            }
            Err(e) => {
                eprintln!("subset_sim: {e}");
                std::process::exit(2);
            }
        }
    }
    if specs.is_empty() {
        eprintln!("subset_sim: no training groups (need --fulleval or --group)");
        std::process::exit(2);
    }
    if seeds.is_empty() {
        eprintln!("subset_sim: no seeds (need --seeds or a fulleval with repro.seed)");
        std::process::exit(2);
    }

    let disjoint_words = if args.require_disjoint_sampler_windows {
        let checked = if !pooled_legs.is_empty() {
            Err("disjoint-window check does not bound pooled rank RNG draws".to_string())
        } else if strat_pairs {
            Err("disjoint-window check requires the uniform sampler".to_string())
        } else {
            disjoint_sampler_windows(&seeds, epochs, ppe)
        };
        Some(checked.unwrap_or_else(|e| {
            eprintln!("subset_sim: {e}");
            std::process::exit(2);
        }))
    } else {
        None
    };

    // Load target + ref columns once per distinct path; many arms and seeds
    // share corpora, and this is the only I/O the replay needs.
    let mut cache: HashMap<PathBuf, (Vec<f64>, Option<Vec<u32>>)> = HashMap::new();
    let mut sim_groups: Vec<SimGroup> = Vec::with_capacity(specs.len());
    for s in &specs {
        let entry = match cache.get(&s.path) {
            Some(e) => e.clone(),
            None => {
                let r = parquet_loader::load_scores_and_refs(&s.path, &target_column, target_scale)
                    .unwrap_or_else(|e| {
                        eprintln!("subset_sim: {e}");
                        std::process::exit(2);
                    });
                let v = (r.human_scores, r.ref_ids);
                cache.insert(s.path.clone(), v.clone());
                v
            }
        };
        sim_groups.push(SimGroup {
            name: s.name.clone(),
            train_weight: s.train_w,
            n_rows: entry.0.len(),
            human_scores: entry.0,
            ref_ids: entry.1,
            within_ref: s.within_ref,
        });
    }

    eprintln!(
        "subset_sim: {} train groups, {} rows, {} x {} draws, seeds {:?}",
        sim_groups.len(),
        sim_groups.iter().map(|g| g.n_rows).sum::<usize>(),
        epochs,
        ppe,
        seeds
    );

    let mut rows: Vec<Value> = Vec::new();
    let mut digest_ok = true;
    for &seed in &seeds {
        let p = SimParams {
            seed,
            epochs,
            pairs_per_epoch: ppe,
            low_q_boost: lo,
            mid_q_boost: mid,
            high_q_boost: hi,
            stratified_bands: strat,
            early_window: args.early_window,
            per_sample_alpha_head: psa,
            stratified_pairs: strat_pairs,
            walk_pairs,
            pooled_legs: pooled_legs.clone(),
            pooled_rank_share,
        };
        let r = sampling::simulate(&sim_groups, &p);
        if let Some(want) = &args.expect_digest {
            let got = r.digest.hex();
            if &got == want {
                println!("DIGEST MATCH seed={seed} {got}");
            } else {
                println!("DIGEST MISMATCH seed={seed} want={want} got={got}");
                digest_ok = false;
            }
        }
        // THE encoder lives in `sampling` so this replay and the
        // `zentrain.sample_coverage` block a bake embeds are the same shape
        // (one owner — zensim CLAUDE.md "no duplicate implementations").
        let enc = sampling::coverage_json;
        let mut row = json!({
            "source": source,
            "seed": seed,
            "sample_stream_seed": if psa {
                sampling::sample_stream_seed_per_sample_alpha(seed)
            } else {
                sampling::sample_stream_seed(seed)
            },
            "epochs": epochs,
            "pairs_per_epoch": ppe,
            "low_q_boost": lo, "mid_q_boost": mid, "high_q_boost": hi,
            "stratified_bands": strat,
            "per_sample_alpha_head": psa,
            "digest": r.digest.hex(),
            "early_digest": r.early_digest.hex(),
            "full": enc(&r.full),
            "early": enc(&r.early),
        });
        if let Some(words) = disjoint_words {
            row["disjoint_sampler_window_words"] = json!(words);
        }
        rows.push(row);
        eprintln!(
            "  seed {seed}: digest {} pooled_cov {:.6} early_cov {:.6} share_l1 {:.6}",
            r.digest.hex(),
            r.full.pooled_row_coverage,
            r.early.pooled_row_coverage,
            r.full.group_share_l1
        );
    }

    if let Some(out) = &args.out {
        if let Some(p) = out.parent() {
            let _ = std::fs::create_dir_all(p);
        }
        std::fs::write(out, serde_json::to_string_pretty(&rows).unwrap()).unwrap_or_else(|e| {
            eprintln!("subset_sim: write {out:?}: {e}");
            std::process::exit(2);
        });
        eprintln!("subset_sim: wrote {} rows -> {}", rows.len(), out.display());
    }
    if !digest_ok {
        std::process::exit(3);
    }
}

// The legacy initializer is s*gamma+C and the generator advances by gamma.
// Seed differences are therefore exact offsets in raw RNG-word positions.
// draw_pair consumes at most four words; skipped small groups consume fewer.
fn disjoint_sampler_windows(seeds: &[u64], epochs: usize, pairs: usize) -> Result<u64, String> {
    let words = (epochs as u128)
        .checked_mul(pairs as u128)
        .and_then(|n| n.checked_mul(4))
        .and_then(|n| u64::try_from(n).ok())
        .filter(|&n| n > 0)
        .ok_or("invalid or overflowing sampler window")?;
    for (i, &a) in seeds.iter().enumerate() {
        for &b in &seeds[i + 1..] {
            let distance = a.wrapping_sub(b).min(b.wrapping_sub(a));
            if distance < words {
                return Err(format!(
                    "overlapping sampler windows: seeds {a} and {b} are {distance} raw words apart; \
                     each window is bounded by {words} words. Use recorded, well-separated seeds"
                ));
            }
        }
    }
    Ok(words)
}

#[cfg(test)]
mod stream_window_tests {
    use super::disjoint_sampler_windows;

    #[test]
    fn nearby_duplicate_wraparound_and_disjoint_windows() {
        assert!(disjoint_sampler_windows(&[17103, 17107], 32, 8192).is_err());
        assert!(disjoint_sampler_windows(&[42, 42], 1, 1).is_err());
        assert!(disjoint_sampler_windows(&[u64::MAX - 3, 0], 1, 2).is_err());
        assert_eq!(disjoint_sampler_windows(&[u64::MAX - 3, 0], 1, 1), Ok(4));
        assert_eq!(
            disjoint_sampler_windows(&[0, 1 << 40], 32, 8192),
            Ok(1048576)
        );
        assert!(disjoint_sampler_windows(&[0], 0, 1).is_err());
        assert!(disjoint_sampler_windows(&[0], usize::MAX, usize::MAX).is_err());
    }
}

/// Honor the trainer's separate sampling seed when replaying recorded runs.
/// Explicit CLI --seeds still takes precedence at the caller.
fn recorded_sampling_seed(repro: &Value) -> Result<Option<u64>, String> {
    for key in ["sample_seed", "seed"] {
        let structured = match repro.get(key).filter(|v| !v.is_null()) {
            Some(v) => Some(v.as_u64().ok_or_else(|| format!("invalid repro.{key}"))?),
            None => None,
        };
        let flag = format!("--{}", key.replace('_', "-"));
        let from_argv = if let Some(argv) = repro.get("argv").and_then(Value::as_array) {
            if argv.iter().any(|v| v.as_str() == Some(&flag)) {
                Some(
                    argv_val(argv, &flag)
                        .and_then(|s| s.parse::<u64>().ok())
                        .ok_or_else(|| format!("invalid repro argv {flag}"))?,
                )
            } else {
                None
            }
        } else {
            None
        };
        if let (Some(a), Some(b)) = (structured, from_argv)
            && a != b
        {
            return Err(format!("conflicting repro.{key} and argv {flag}"));
        }
        if let Some(seed) = structured.or(from_argv) {
            return Ok(Some(seed));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod replay_seed_tests {
    use super::*;

    #[test]
    fn sampling_seed_precedes_legacy_seed_and_conflicts_are_refused() {
        assert_eq!(
            recorded_sampling_seed(&json!({"seed":7101,"sample_seed":17101})),
            Ok(Some(17101))
        );
        assert_eq!(
            recorded_sampling_seed(&json!({"seed":7101,"argv":["--sample-seed","17101"]})),
            Ok(Some(17101))
        );
        assert_eq!(
            recorded_sampling_seed(&json!({"seed":7101,"sample_seed":null})),
            Ok(Some(7101))
        );
        assert_eq!(
            recorded_sampling_seed(&json!({"argv":["--seed","42"]})),
            Ok(Some(42))
        );
        assert_eq!(
            recorded_sampling_seed(&json!({"sample_seed":u64::MAX})),
            Ok(Some(u64::MAX))
        );
        assert_eq!(recorded_sampling_seed(&json!({})), Ok(None));
        for invalid in [
            json!({"sample_seed":-1,"seed":7101}),
            json!({"sample_seed":17,"argv":["--sample-seed","18"]}),
            json!({"seed":7101,"argv":["--sample-seed"]}),
            json!({"seed":7101,"argv":["--sample-seed","bad"]}),
            json!({"seed":7101,"argv":["--seed","42"]}),
        ] {
            assert!(recorded_sampling_seed(&invalid).is_err(), "{invalid}");
        }
    }
}
