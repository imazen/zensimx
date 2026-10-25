//! Whole-vector tier-parity gate for `FormulaRevision::Rev7` (the `rowlanes`
//! era): `research::extract` over its whole scope (basic + peaks + v2 +
//! globaff) must be BIT-IDENTICAL under every dispatch-tier permutation.
//! Rev7 adds each pool column to sixteen f32 row lanes in a fixed order and
//! folds them into the f64 lanes at fixed points (each row's end of complete
//! 8-column chunks, and after every 4,096th column), so the geometries
//! include remainder columns and a scale-0 width past 4,096.
//!
//! `ZENSIM_FORMULA_REV` is a process-wide `OnceLock`, so the assertions run in
//! a re-exec'd child (same protocol as `featcanon_tier_parity.rs`).

#![cfg(feature = "feature-regime-v2")]
#![allow(deprecated)]

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use zensim::RgbSlice;
use zensim::feature_set_id::ComputeToken;
use zensim::research::{self, Request};

const HAS_RUNTIME_TOKENS: bool = cfg!(any(target_arch = "x86_64", target_arch = "aarch64"));

fn test_images(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let n = w * h;
    let mut src = vec![[0u8; 3]; n];
    let mut dst = vec![[0u8; 3]; n];
    for y in 0..h {
        for x in 0..w {
            let g = ((x * 7 + y * 13) % 251) as u8;
            let t = (((x * 31) ^ (y * 17)) % 67) as u8;
            let s = g.wrapping_add(t);
            src[y * w + x] = [s, g.wrapping_add(t / 2), 255u8 - s];
            let dv = (s as i32 + ((x * y) % 29) as i32 - 14).clamp(0, 255) as u8;
            dst[y * w + x] = [dv, g.wrapping_add(t / 3), 255u8 - dv];
        }
    }
    (src, dst)
}

/// Rev7's whole scope: basic + peaks + v2 (revision 5's measurements, raw ringing) and Rev6's globaff family.
fn rev7_request() -> Request {
    let want = research::family_slots(ComputeToken::Basic)
        .union(&research::family_slots(ComputeToken::Peaks))
        .union(&research::family_slots(ComputeToken::V2))
        .union(&research::family_slots(ComputeToken::GlobAff));
    Request::for_slots(want, research::full_width())
}

/// The same scope without globaff, which revisions below 6 refuse.
fn rev5_request() -> Request {
    let want = research::family_slots(ComputeToken::Basic)
        .union(&research::family_slots(ComputeToken::Peaks))
        .union(&research::family_slots(ComputeToken::V2));
    Request::for_slots(want, research::full_width())
}

fn extract(src: &RgbSlice<'_>, dst: &RgbSlice<'_>) -> Vec<f64> {
    research::extract(&rev7_request(), src, dst)
        .expect("extract")
        .values()
        .to_vec()
}

fn extract_bits(src: &RgbSlice<'_>, dst: &RgbSlice<'_>) -> Vec<u64> {
    extract(src, dst).iter().map(|v| v.to_bits()).collect()
}

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

#[test]
fn rev7_vector_bit_identical_under_token_permutations() {
    const SENTINEL: &str = "REV7_CANON_OK";
    if !at_revision(
        "7",
        "rev7_vector_bit_identical_under_token_permutations",
        SENTINEL,
    ) {
        return;
    }
    let _guard = archmage::testing::lock_token_testing();
    let mut cross_target = Vec::new();
    for &(w, h) in &[
        (64usize, 64usize),
        (97, 63),
        (131, 65),
        (255, 129),
        (4133, 33),
    ] {
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
        let base = &vectors[0].1;
        cross_target.extend(base.iter().flat_map(|v| v.to_le_bytes()));
        assert!(base.iter().any(|&b| b != 0), "vacuous: all-zero vector");
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
                "rev7 {w}x{h}: permutation {label} vs {} differs in {} slots (first: f{})",
                vectors[0].0,
                diffs.len(),
                diffs.first().copied().unwrap_or(usize::MAX),
            );
        }
        eprintln!(
            "rev7 {w}x{h}: {} permutations bit-identical over {} slots",
            vectors.len(),
            base.len()
        );
    }
    // Optional cross-target gate, exercised by the native/scalar/WASI runs.
    // Capture is explicit and refuses to replace existing evidence.
    if let Some(path) = std::env::var_os("REV7_PARITY_CAPTURE") {
        use std::io::Write;
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap()
            .write_all(&cross_target)
            .unwrap();
    }
    if let Some(path) = std::env::var_os("REV7_PARITY_EXPECT") {
        assert_eq!(
            cross_target,
            std::fs::read(path).unwrap(),
            "cross-target full-vector bytes"
        );
    }
    println!("{SENTINEL}");
}

/// Rev7 must move values (the era is not a relabel), and only by rounding: the forms are exact rewrites and the lanes
/// regroup the same additions. Its basic, peaks and v2 slots, raw ringing included, are compared with Rev5's and its
/// globaff slots with Rev6's; the 12 ringing slots must also differ from Rev6's high-pass form. Each revision's vector
/// comes from its own child process.
#[test]
fn rev7_moves_slots_by_rounding_only() {
    const TAG: &str = "REV7_VEC";
    if std::env::var("REV7_VEC_CHILD").is_err() {
        let vector_at = |rev: &str| -> Vec<f64> {
            let exe = std::env::current_exe().expect("test binary path");
            let out = std::process::Command::new(&exe)
                .args([
                    "rev7_moves_slots_by_rounding_only",
                    "--exact",
                    "--nocapture",
                ])
                .env("ZENSIM_FORMULA_REV", rev)
                .env("REV7_VEC_CHILD", "1")
                .output()
                .expect("re-exec");
            let stdout = String::from_utf8_lossy(&out.stdout).to_string();
            assert!(
                out.status.success(),
                "rev{rev} child failed:\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
            let line = stdout
                .lines()
                .find_map(|l| l.strip_prefix(TAG))
                .unwrap_or_else(|| panic!("rev{rev} child did not emit its vector:\n{stdout}"));
            line.split_whitespace()
                .map(|t| f64::from_bits(u64::from_str_radix(t, 16).expect("hex bits")))
                .collect()
        };
        let (r5, r6, r7) = (vector_at("5"), vector_at("6"), vector_at("7"));
        assert_eq!(r5.len(), r7.len(), "rev5/rev7 vector width");
        assert_eq!(r6.len(), r7.len(), "rev6/rev7 vector width");
        let bpv = rev5_request().want().clone();
        let globaff = research::family_slots(ComputeToken::GlobAff);
        let ringing: Vec<usize> = (0..12).map(|cell| 372 + 29 * cell + 26).collect();
        // Relative change, with the slot's magnitude floored at 1e-3 so that
        // features near zero (where the cancellation-free forms correct the
        // old forms' rounding by more than 100%) are judged absolutely.
        let compare = |base: &[f64], ids: &mut dyn Iterator<Item = usize>, label: &str| {
            let (mut moved, mut n, mut worst, mut worst_id) = (0usize, 0usize, 0.0f64, usize::MAX);
            for id in ids {
                let (a, b) = (base[id], r7[id]);
                assert_eq!(
                    a.is_finite(),
                    b.is_finite(),
                    "f{id}: finiteness changed against {label}"
                );
                n += 1;
                if a.to_bits() != b.to_bits() {
                    moved += 1;
                }
                if !a.is_finite() {
                    continue;
                }
                let rel = (a - b).abs() / a.abs().max(1e-3);
                if rel > worst {
                    (worst, worst_id) = (rel, id);
                }
            }
            eprintln!(
                "rev7 vs {label} at 131x65: {moved}/{n} slots differ; largest floored relative \
                 change {worst:.3e} at f{worst_id}"
            );
            assert!(moved > 0, "{label} and rev7 vectors must differ");
            // 1e-2 is my bound, not a measured one: a gross-error guard (a lost
            // fold or a wrong lane order moves a slot by its whole size).
            assert!(
                worst < 1e-2,
                "f{worst_id}: rev7 moves a slot by {worst:.3e} (floored relative) against {label}"
            );
        };
        compare(&r5, &mut bpv.iter_slots(), "rev5 (basic + peaks + v2)");
        // The research vector carries the globaff family (f1867..) only with `training`: without it the vector ends at
        // the palette block (`research::full_width`), so there is no globaff slot to compare.
        if cfg!(feature = "training") {
            compare(&r6, &mut globaff.iter_slots(), "rev6 (globaff)");
        } else {
            assert!(
                globaff.iter_slots().all(|id| id >= r7.len()),
                "globaff slots inside a vector built without `training`"
            );
        }
        // Rev7 reads raw ringing, so its ringing slots must leave Rev6's high-pass values behind.
        let ring_moved = ringing
            .iter()
            .filter(|&&id| (r6[id] - r7[id]).abs() > 1e-3 * r6[id].abs().max(1e-3))
            .count();
        assert!(
            ring_moved > 0,
            "rev7's ringing slots equal rev6's high-pass form"
        );
        return;
    }
    let (w, h) = (131usize, 65usize);
    let (src, dst) = test_images(w, h);
    let request = if std::env::var("ZENSIM_FORMULA_REV").as_deref() == Ok("5") {
        rev5_request()
    } else {
        rev7_request()
    };
    let (rs, ds) = (RgbSlice::new(&src, w, h), RgbSlice::new(&dst, w, h));
    let v: Vec<u64> = research::extract(&request, &rs, &ds)
        .expect("extract")
        .values()
        .iter()
        .map(|x| x.to_bits())
        .collect();
    let hex: Vec<String> = v.iter().map(|b| format!("{b:x}")).collect();
    println!("{TAG} {}", hex.join(" "));
}
