//! Whole-vector tier-parity gate for `FormulaRevision::Rev4` (the
//! `tiercanon` era — `featcanon` lane).
//!
//! Under Rev4 every feature-producing leaf runs the canonical
//! `featcanon` arithmetic (fixed 8-virtual-lane f32 pools + fixed pairwise
//! reduce + inherent fused `mul_add`), so `research::extract(everything())`
//! must emit a BIT-IDENTICAL 1825-slot vector on every dispatch tier —
//! not just the score, every feature.
//!
//! `ZENSIM_FORMULA_REV` is a `OnceLock`, so each revision's assertions own a
//! child process (`reexec_at_rev`, the same protocol as
//! `ssim_form::run_at_revision`): the parent proves the child ran by
//! requiring a sentinel on stdout.

#![allow(deprecated)]

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use zensim::RgbSlice;
use zensim::research::{self, Request};

const N_SLOTS: usize = 1825;

/// Whether this target has runtime-disableable token slots (else the walk
/// yields exactly one permutation and cross-tier claims are vacuous).
const HAS_RUNTIME_TOKENS: bool = cfg!(any(target_arch = "x86_64", target_arch = "aarch64"));

/// Deterministic textured image pair — strong enough structure that real
/// SSIM activity exists (a flat pair pools zeros and a parity test would be
/// vacuous).
fn test_images(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let n = w * h;
    let mut src = vec![[0u8; 3]; n];
    let mut dst = vec![[0u8; 3]; n];
    for y in 0..h {
        for x in 0..w {
            // Smooth gradient + high-frequency texture + a plateau, so every
            // pooled family sees non-degenerate input.
            let g = ((x * 7 + y * 13) % 251) as u8;
            let t = (((x * 31) ^ (y * 17)) % 67) as u8;
            let s = g.wrapping_add(t);
            src[y * w + x] = [s, g.wrapping_add(t / 2), 255u8 - s];
            // Nonlinear distortion: asymmetric clip + additive wobble.
            let dv = (s as i32 + ((x * y) % 29) as i32 - 14).clamp(0, 255) as u8;
            dst[y * w + x] = [dv, g.wrapping_add(t / 3), 255u8 - dv];
        }
    }
    (src, dst)
}

fn extract_bits(src: &RgbSlice<'_>, dst: &RgbSlice<'_>) -> Vec<u64> {
    research::extract(
        &Request::for_slots(Request::everything().want().clipped_to(N_SLOTS), N_SLOTS),
        src,
        dst,
    )
    .expect("extract")
    .values()
    .iter()
    .map(|v| v.to_bits())
    .collect()
}

/// Re-exec THIS test binary with `ZENSIM_FORMULA_REV=<rev>` when the current
/// process is not already pinned there; the child runs exactly `test_path`
/// and must print `sentinel` (proves the filter matched and the body ran).
fn at_revision(rev: &str, test_path: &str, sentinel: &str) -> bool {
    if std::env::var("ZENSIM_FORMULA_REV").as_deref() == Ok(rev) {
        return true;
    }
    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args([test_path, "--exact", "--nocapture", "--test-threads=1"])
        .env("ZENSIM_FORMULA_REV", rev)
        .output()
        .expect("re-exec the test binary");
    assert!(
        out.status.success(),
        "child at rev {rev} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        String::from_utf8_lossy(&out.stdout).contains(sentinel),
        "child at rev {rev} did not run {test_path} (sentinel missing)"
    );
    false
}

/// The Rev4 gate: every 1825-slot vector, bit-identical under every token
/// permutation, including nonzero-remainder geometries (97/63, 131/65 — the
/// SIMD tail paths the tier-dispatched kernels used to diverge on).
#[test]
fn rev4_vector_bit_identical_under_token_permutations() {
    const SENTINEL: &str = "REV4_CANON_OK";
    if !at_revision(
        "4",
        "rev4_vector_bit_identical_under_token_permutations",
        SENTINEL,
    ) {
        return;
    }
    for &(w, h) in &[(64usize, 64usize), (97, 63), (131, 65), (255, 129)] {
        let (src, dst) = test_images(w, h);
        let s = RgbSlice::new(&src, w, h);
        let d = RgbSlice::new(&dst, w, h);

        let mut vectors: Vec<(String, Vec<u64>)> = Vec::new();
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            vectors.push((perm.label.clone(), extract_bits(&s, &d)));
        });
        if HAS_RUNTIME_TOKENS {
            assert!(report.permutations_run >= 2);
        }
        assert_eq!(vectors[0].1.len(), N_SLOTS);
        let base = &vectors[0].1;
        for (label, v) in &vectors[1..] {
            let diffs: Vec<usize> = base
                .iter()
                .zip(v.iter())
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .map(|(i, _)| i)
                .collect();
            assert!(
                diffs.is_empty(),
                "rev4 {w}x{h}: permutation {label} vs {} differs in {} slots \
                 (first: f{})",
                vectors[0].0,
                diffs.len(),
                diffs.first().copied().unwrap_or(usize::MAX),
            );
        }
        eprintln!("rev4 {w}x{h}: {} permutations bit-identical", vectors.len());
    }
    println!("{SENTINEL}");
}

/// Negative control: the same extraction at Rev3 (production, tier-dispatched
/// arithmetic) must NOT be tier-identical on this host — otherwise the Rev4
/// gate above would be vacuous. Requires x86_64 token machinery.
#[test]
fn rev3_negative_control_can_diverge() {
    const SENTINEL: &str = "REV3_DIV_OK";
    if !at_revision("3", "rev3_negative_control_can_diverge", SENTINEL) {
        return;
    }
    let (w, h) = (255usize, 129usize);
    let (src, dst) = test_images(w, h);
    let s = RgbSlice::new(&src, w, h);
    let d = RgbSlice::new(&dst, w, h);

    let mut vectors: Vec<(String, Vec<u64>)> = Vec::new();
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        vectors.push((perm.label.clone(), extract_bits(&s, &d)));
    });
    let distinct: std::collections::BTreeSet<&Vec<u64>> = vectors.iter().map(|(_, v)| v).collect();
    eprintln!(
        "rev3 255x129: {} permutations -> {} distinct vectors",
        report.permutations_run,
        distinct.len()
    );
    if HAS_RUNTIME_TOKENS && report.permutations_run >= 2 {
        assert!(
            distinct.len() > 1,
            "negative control: rev3 tiers produced identical vectors — \
             the production path is unexpectedly deterministic here"
        );
    }
    println!("{SENTINEL}");
}

/// Rev3 output must differ from Rev4 output on textured input — the revision
/// actually selects different arithmetic (guards a no-op wiring).
///
/// Each revision's vector comes from its own child process (the switch is a
/// `OnceLock`) and is read back from the child's stdout in memory — no file,
/// so concurrent runs cannot race on a shared path.
#[test]
fn rev4_moves_slots_vs_rev3() {
    const TAG: &str = "FEATCANON_VEC";
    if std::env::var("FEATCANON_VEC_CHILD").is_err() {
        let vector_at = |rev: &str| -> Vec<u64> {
            let exe = std::env::current_exe().expect("test binary path");
            let out = std::process::Command::new(&exe)
                .args(["rev4_moves_slots_vs_rev3", "--exact", "--nocapture"])
                .env("ZENSIM_FORMULA_REV", rev)
                .env("FEATCANON_VEC_CHILD", "1")
                .output()
                .expect("re-exec");
            let stdout = String::from_utf8_lossy(&out.stdout).to_string();
            assert!(
                out.status.success(),
                "rev{rev} child failed:\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
            // Serial libtest can prefix the first uncaptured receipt with its banner.
            let marker = format!("{TAG} ");
            let mut vectors = stdout
                .lines()
                .filter_map(|line| line.split_once(&marker).map(|(_, vector)| vector));
            let line = vectors
                .next()
                .unwrap_or_else(|| panic!("rev{rev} child did not emit its vector:\n{stdout}"));
            assert!(
                vectors.next().is_none(),
                "rev{rev} child emitted multiple vectors:\n{stdout}"
            );
            let v: Vec<u64> = line
                .split_whitespace()
                .map(|t| u64::from_str_radix(t, 16).expect("hex bits"))
                .collect();
            assert_eq!(v.len(), N_SLOTS, "rev{rev} child vector width");
            v
        };
        let (a, b) = (vector_at("3"), vector_at("4"));
        let moved = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        eprintln!("rev4 vs rev3 at 131x65: {moved}/{N_SLOTS} slots differ");
        assert!(moved > 0, "rev3 and rev4 vectors must differ");
        return;
    }
    // Child body: print the vector bits on one line.
    let (w, h) = (131usize, 65usize);
    let (src, dst) = test_images(w, h);
    let v = extract_bits(&RgbSlice::new(&src, w, h), &RgbSlice::new(&dst, w, h));
    let hex: Vec<String> = v.iter().map(|b| format!("{b:x}")).collect();
    println!("{TAG} {}", hex.join(" "));
}
