//! Score pre-extracted feature rows against an arbitrary ZNPR v3 bake.
//!
//! This is the feature-cache fast path for `cross_codec_consistency.py`
//! (EVAL-ACCEL 2026-05-19). Instead of decoding images and recomputing
//! features per `measure(q)` call, the script reads the pre-extracted
//! 372-feature parquet sidecars at
//! `/mnt/v/zen/picker-training/2026-05-19/butter/<codec>.parquet`, packs
//! the relevant rows into a tiny binary blob, and shells to this binary
//! to get the scores. Skips the ~5-15 s per call that
//! `score_pair_with_bake` spent on image decode + feature extract.
//!
//! Wire format (input file via `--features-file <path>`):
//!     u32 LE n_features
//!     u32 LE n_rows
//!     f32 LE feature_matrix[n_rows][n_features]  (row-major)
//!
//! `--f64-wire` uses the same header and f64 LE payload, preserving native
//! research precision and printing 17 decimal places. `--production` keeps
//! the bake's default score disposition rather than applying --bake-post.
//!
//! Smaller fast path (`--features <space-sep floats>`): a single row of
//! features as a CLI arg, identical semantics to a 1-row input file.
//!
//! Output: one score per row, one `%.6f` per line on stdout.
//! `--score-bits` emits the exact native served f64 bit pattern as u64 instead.
//!
//! Honors the same `--bake-post {raw|clamp|mapped[:a,b]}` semantics as
//! `score_pair_with_bake`, plus the full V_24 dispatch path
//! (per-sample-α head, hybrid head, tanh output pin) so the produced
//! score is bit-exact with the slow path on the same feature row.

#[path = "research_cached/mod.rs"]
mod research_cached;

use std::path::PathBuf;
use std::process::ExitCode;

use zenpredict::Model;
use zensim::BakeScorer;
use zensim_validate::bake_runtime::post_mode_params;

fn parse_features_arg(s: &str) -> Result<(usize, usize, Vec<f32>), String> {
    let vals: Result<Vec<f32>, _> = s.split_whitespace().map(|t| t.parse::<f32>()).collect();
    let vals = vals.map_err(|e| format!("--features parse: {e}"))?;
    if vals.is_empty() {
        return Err("--features is empty".into());
    }
    Ok((vals.len(), 1, vals))
}

fn read_features_file(path: &PathBuf, wide: bool) -> Result<(usize, usize, Vec<f64>), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {path:?}: {e}"))?;
    if bytes.len() < 8 {
        return Err(format!(
            "{path:?}: header too short ({} bytes)",
            bytes.len()
        ));
    }
    let n_features = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let n_rows = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let expected_floats = n_rows
        .checked_mul(n_features)
        .ok_or_else(|| format!("{path:?}: n_rows*n_features overflow ({n_rows} * {n_features})"))?;
    let bytes_per_value = if wide { 8 } else { 4 };
    let expected_bytes = expected_floats
        .checked_mul(bytes_per_value)
        .and_then(|n| n.checked_add(8))
        .ok_or_else(|| "feature payload byte count overflow".to_string())?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "{path:?}: payload size mismatch: header says {n_rows} rows × {n_features} features = {expected_bytes} bytes, got {}",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(expected_floats);
    for i in 0..expected_floats {
        let off = 8 + i * bytes_per_value;
        out.push(if wide {
            f64::from_le_bytes(bytes[off..off + 8].try_into().unwrap())
        } else {
            f64::from(f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()))
        });
    }
    Ok((n_features, n_rows, out))
}

fn print_usage() {
    eprintln!(
        "predict_features_with_bake — bake forward pass over pre-extracted features\n\
\n\
USAGE:\n\
    predict_features_with_bake --bake <path> [--bake-post raw|clamp|mapped[:a,b]] \\\n\
        (--features 'f0 f1 f2 ...' | --features-file <path>)\n\
\n\
The --features-file format is u32 LE n_features, u32 LE n_rows, then\n\
n_rows*n_features f32 LE features (row-major). Output is one\n\
'%.6f'-formatted score per row, one per line, on stdout.\n"
    );
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut bake: Option<PathBuf> = None;
    let mut bake_post: String = "clamp".to_string();
    let mut features_arg: Option<String> = None;
    let mut features_file: Option<PathBuf> = None;
    let mut codec_hint: Option<String> = None;
    let mut f64_wire = false;
    let mut production = false;
    let mut score_bits = false;
    let mut research_palette_cached = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--research-palette-cached" => research_palette_cached = true,
            "--f64-wire" => f64_wire = true,
            "--production" => production = true,
            "--score-bits" => score_bits = true,
            "--codec" => {
                codec_hint = match args.next() {
                    Some(v) => Some(v),
                    None => {
                        eprintln!("--codec requires a value");
                        return ExitCode::FAILURE;
                    }
                };
            }
            "--bake" => {
                let v = match args.next() {
                    Some(v) => v,
                    None => {
                        eprintln!("--bake requires a value");
                        return ExitCode::FAILURE;
                    }
                };
                bake = Some(v.into());
            }
            "--bake-post" => {
                bake_post = match args.next() {
                    Some(v) => v,
                    None => {
                        eprintln!("--bake-post requires a value");
                        return ExitCode::FAILURE;
                    }
                };
            }
            "--features" => {
                features_arg = args.next();
                if features_arg.is_none() {
                    eprintln!("--features requires a value");
                    return ExitCode::FAILURE;
                }
            }
            "--features-file" => {
                let v = match args.next() {
                    Some(v) => v,
                    None => {
                        eprintln!("--features-file requires a value");
                        return ExitCode::FAILURE;
                    }
                };
                features_file = Some(v.into());
            }
            "-h" | "--help" => {
                print_usage();
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown arg: {other}");
                print_usage();
                return ExitCode::FAILURE;
            }
        }
    }
    let bake = match bake {
        Some(b) => b,
        None => {
            eprintln!("--bake is REQUIRED");
            print_usage();
            return ExitCode::FAILURE;
        }
    };
    let (n_features_in, n_rows, feature_buf) = match (features_arg, features_file) {
        (Some(s), None) => match parse_features_arg(&s)
            .map(|(w, n, v)| (w, n, v.into_iter().map(f64::from).collect::<Vec<_>>()))
        {
            Ok(t) => t,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        },
        (None, Some(p)) => match read_features_file(&p, f64_wire) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        },
        (Some(_), Some(_)) => {
            eprintln!("specify --features OR --features-file, not both");
            return ExitCode::FAILURE;
        }
        (None, None) => {
            eprintln!("one of --features or --features-file is REQUIRED");
            print_usage();
            return ExitCode::FAILURE;
        }
    };

    let bake_bytes = match std::fs::read(&bake) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read bake {bake:?}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let model = match Model::from_bytes(&bake_bytes) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("parse ZNPR bake: {e:?}");
            return ExitCode::FAILURE;
        }
    };
    if research_palette_cached {
        if !production || !f64_wire || codec_hint.is_some() {
            eprintln!("cached palette research requires --production --f64-wire and no codec hint");
            return ExitCode::FAILURE;
        }
        let mut scorer = match research_cached::PaletteScorer::new(&model) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        };
        for row in feature_buf.chunks_exact(n_features_in) {
            match scorer.score(row) {
                Ok(score) => {
                    if score_bits {
                        println!("{}", score.to_bits());
                    } else {
                        println!("{score:.17}");
                    }
                }
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::FAILURE;
                }
            }
        }
        return ExitCode::SUCCESS;
    }
    let mut scorer = BakeScorer::new(&model).expect("invalid score metadata");
    let params = post_mode_params(&bake_post).expect("invalid bake-post");
    if !production {
        scorer = scorer
            .with_score_disposition(&params)
            .expect("invalid score disposition");
    }
    let mut row_f64 = Vec::with_capacity(n_features_in);

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    use std::io::Write as _;

    for row_idx in 0..n_rows {
        let start = row_idx * n_features_in;
        let end = start + n_features_in;
        let row = &feature_buf[start..end];
        row_f64.clear();
        row_f64.extend_from_slice(row);
        let score = scorer
            .score_features(&row_f64, 0, 0, codec_hint.as_deref())
            .expect("invalid feature row");
        let written = if score_bits {
            writeln!(out, "{}", score.to_bits())
        } else if f64_wire {
            writeln!(out, "{score:.17}")
        } else {
            writeln!(out, "{score:.6}")
        };
        if written.is_err() {
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}
