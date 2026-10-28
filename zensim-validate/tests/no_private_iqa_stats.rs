//! imazen/zensimx#41 Tier-1 #2 gate: `zensim-validate` must not define its
//! own IQA correlation primitives — `zenstats` (reached as
//! `zensim_validate::panel::*` or `zenstats::*`) is the single owner.
//!
//! The 2026-05-26 consolidation shipped the owner but left the callers: on
//! 2026-08-27 this crate still carried two ORDINAL-rank `fn spearman` copies
//! (`bin/train_minmax.rs`, the `mlp_train::minmax_monotone` test module)
//! plus five delegating wrapper fns. "Landing the owner is half the job;
//! deleting the callers is the other half, and it is the half that gets
//! skipped" (CLAUDE.md). This test is the enforcement: it walks every `.rs`
//! under `zensim-validate/src` and fails on any `fn` whose name is one of the
//! canonical stat primitives, wrapper or not — a wrapper is a `use`, never a
//! `fn`.
//!
//! Mutation-verified: adding `fn spearman(a: &[f64], b: &[f64]) -> f64` to
//! any file under `src/` fails this test.
//!
//! 2026-10-11: the statistics zenmetrics-stats absorbed from zensim (DS-AUC,
//! forced-choice agreement) and its interval and multiple-comparison owners are
//! banned too, so the deleted copies cannot come back under their names.
//!
//! Deliberately NOT in the banned list: `fast_kendall` (`src/main.rs`) — an
//! O(n log n) tau-b with an exact-tie predicate that `zenstats::kendall_tau`
//! cannot reproduce, used as a training objective; its fate is an open
//! owner decision on #41, not something a grep gate should silently settle.

use std::path::{Path, PathBuf};

/// Names whose definition as a `fn` anywhere in `src/` is a duplicate of a
/// `zenstats` primitive.
const BANNED_FN_NAMES: &[&str] = &[
    "spearman",
    "pearson",
    "ranks",
    "midrank",
    "midranks",
    "spearman_correlation",
    "pearson_correlation",
    "srocc",
    "plcc",
    "krocc",
    "kendall_tau",
    // Moved to zenmetrics-stats on 2026-10-11 (`pairs::ds_auc`,
    // `forced_choice`, `resample`, `multiple`); the zensim copies were deleted.
    "ds_auc",
    "agreement_by_group_index",
    "index_rows_by_group",
    "fit_observer_slope",
    "spearman_interval",
    "cluster_bootstrap",
    "holm",
    "holm_reject",
];

fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {dir:?}: {e}")) {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.is_dir() {
            rs_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Returns the fn name if `line` is a fn definition (`fn name(` / `fn name<`
/// after optional `pub`, `pub(crate)`, `const`, `unsafe`, `async`), else None.
fn fn_def_name(line: &str) -> Option<&str> {
    let mut s = line.trim_start();
    loop {
        let before = s;
        for qual in [
            "pub(crate) ",
            "pub(super) ",
            "pub ",
            "const ",
            "unsafe ",
            "async ",
        ] {
            if let Some(rest) = s.strip_prefix(qual) {
                s = rest;
            }
        }
        if s == before {
            break;
        }
    }
    let rest = s.strip_prefix("fn ")?;
    let end = rest.find(|c: char| !(c.is_alphanumeric() || c == '_'))?;
    let name = &rest[..end];
    let after = &rest[end..];
    if after.starts_with('(') || after.starts_with('<') {
        Some(name)
    } else {
        None
    }
}

#[test]
fn zensim_validate_defines_no_private_iqa_stat_fns() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rs_files(&src, &mut files);
    files.sort();
    assert!(
        files.len() > 10,
        "expected to scan the whole crate, found only {} files",
        files.len()
    );

    let mut offenders = Vec::new();
    for path in &files {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        for (i, line) in text.lines().enumerate() {
            if fn_def_name(line).is_some_and(|name| BANNED_FN_NAMES.contains(&name)) {
                offenders.push(format!(
                    "{}:{}: `{}`",
                    path.strip_prefix(&src).unwrap_or(path).display(),
                    i + 1,
                    line.trim()
                ));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "private IQA-stat fn definitions found in zensim-validate/src — use \
         `zenstats::panel::*` (via `zensim_validate::panel`) instead of \
         redefining:\n  {}",
        offenders.join("\n  ")
    );
}

#[test]
fn fn_def_name_parser_recognises_definitions() {
    assert_eq!(
        fn_def_name("fn spearman(a: &[f64]) -> f64 {"),
        Some("spearman")
    );
    assert_eq!(
        fn_def_name("    pub fn ranks(v: &[f64]) -> Vec<f64> {"),
        Some("ranks")
    );
    assert_eq!(
        fn_def_name("pub(crate) fn pearson<T>(a: T) {"),
        Some("pearson")
    );
    assert_eq!(
        fn_def_name("        fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {"),
        Some("spearman_correlation")
    );
    assert_eq!(fn_def_name("let srocc = spearman(&pred, &scores);"), None);
    assert_eq!(fn_def_name("// fn spearman is gone"), None);
    assert_eq!(
        fn_def_name("fn spearman_like_thing(a: f64) {"),
        Some("spearman_like_thing")
    );
    assert!(!BANNED_FN_NAMES.contains(&"spearman_like_thing"));
}
