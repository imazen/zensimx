//! Tier-parity audit for `zensim::research::extract` — MEASUREMENT ONLY.
//!
//! Runs the full-width research extraction (all families, current revision)
//! on real TRAIN image pairs under each forcible SIMD tier by disabling
//! archmage tokens process-wide:
//!
//!   v4x    — nothing disabled (best token wins; V4x where consulted)
//!   v4     — X64V4xToken disabled
//!   v3     — X64V4xToken + X64V4Token disabled (canonical AVX2 arithmetic)
//!   scalar — every x86 vector token disabled (incant falls to scalar)
//!
//! Compares each tier's `Extraction::values()` against the v3 baseline
//! bit-for-bit and by max |relative difference|, then rolls up per feature
//! block (`provenance.n`) so the report can name the kernel responsible.
//!
//! Usage: `tier_audit_features <pairs.tsv>` where the TSV has a header
//! `ref_path<TAB>dist_path...` (the Rev4-featbank pair-list shape), or
//! `tier_audit_features --pair <label> <ref> <dist>` repeated.
//!
//! Prints a per-(pair, tier, block) table plus a per-(tier, block) rollup
//! across pairs to stdout; write `TIER_AUDIT_OUT=<path>` to also emit TSV.
//!
//! Also the featcanon measurement driver (absorbed from the retired
//! `featcanon_audit` bin, featcanon-fix 2026-09-26 — one owner, not two):
//!
//! * `ZENSIM_FEATCANON_DUMP=<dir>` writes every tier's raw vector as
//!   little-endian f64 to `<dir>/<mode>_<tier>_<pair>.f64bin`, plus
//!   `<dir>/names_<pair>.tsv` (`id\tfamily\tname`), where `<mode>` is
//!   `ZENSIM_FEATCANON` or `prod`.
//! * `TIER_AUDIT_ONLY=<tier>` runs one tier and skips the comparison (timing).
//! * `ZENSIM_FEATCANON=exact|c32|c64|neum|off` selects a measurement
//!   arithmetic. It exists only in a build with zensim's `oracle` feature:
//!   `cargo build --release -p zensim-validate --bin tier_audit_features
//!   --features featcanon-oracle`. A build without it refuses the variable
//!   rather than dumping production vectors under a candidate's name.

use std::fmt::Write as _;
use std::process::exit;

use zensim::RgbSlice;
use zensim::research::{self, Request};

fn decode_rgb(path: &str) -> (Vec<[u8; 3]>, usize, usize) {
    let img = image::open(path).unwrap_or_else(|e| panic!("decode {path}: {e}"));
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let raw = rgb.into_raw();
    let (chunks, rem) = raw.as_chunks::<3>();
    assert!(rem.is_empty(), "{path}: RGB pixel count not divisible by 3");
    (chunks.to_vec(), w, h)
}

/// A decoded pair ready for extraction.
struct LoadedPair {
    label: String,
    src: Vec<[u8; 3]>,
    dst: Vec<[u8; 3]>,
    w: usize,
    h: usize,
}

/// Reviewer probe format (featcanon review, `/var/tmp/review-featcanon/probe`):
/// `label \t grid \t crop(x,y,w,h|-) \t ref1 \t dst1 [\t ref2 \t dst2 ...]`
/// where grid `g` tiles `g*g` equal-size pairs row-major into one mosaic and
/// `crop` then applies to the mosaic. Detected when column 1 parses as `usize`
/// and the line has >= 5 columns.
fn is_probe_line(c: &[&str]) -> bool {
    c.len() >= 5 && c[1].parse::<usize>().is_ok() && (c[2] == "-" || c[2].contains(','))
}

fn load_probe_line(line: &str) -> LoadedPair {
    let c: Vec<&str> = line.split('\t').collect();
    let label = c[0].to_string();
    let g: usize = c[1].parse().unwrap();
    let crop = c[2];
    let paths = &c[3..];
    assert_eq!(paths.len(), 2 * g * g, "{label}: need {} paths", 2 * g * g);
    let mut tiles = Vec::new();
    for k in 0..g * g {
        let (a, aw, ah) = decode_rgb(paths[2 * k]);
        let (b, bw, bh) = decode_rgb(paths[2 * k + 1]);
        assert_eq!((aw, ah), (bw, bh), "{label}: tile {k} size mismatch");
        tiles.push((a, b, aw, ah));
    }
    let (tw, th) = (tiles[0].2, tiles[0].3);
    for t in &tiles {
        assert_eq!((t.2, t.3), (tw, th), "{label}: mosaic tiles differ in size");
    }
    let (mut w, mut h) = (tw * g, th * g);
    let mut src = vec![[0u8; 3]; w * h];
    let mut dst = vec![[0u8; 3]; w * h];
    for (k, t) in tiles.iter().enumerate() {
        let (ox, oy) = ((k % g) * tw, (k / g) * th);
        for y in 0..th {
            let row = (oy + y) * w + ox;
            src[row..row + tw].copy_from_slice(&t.0[y * tw..y * tw + tw]);
            dst[row..row + tw].copy_from_slice(&t.1[y * tw..y * tw + tw]);
        }
    }
    if crop != "-" {
        let v: Vec<usize> = crop.split(',').map(|s| s.parse().unwrap()).collect();
        let (x0, y0, cw, ch) = (v[0], v[1], v[2], v[3]);
        assert!(x0 + cw <= w && y0 + ch <= h, "{label}: crop out of bounds");
        let mut s2 = Vec::with_capacity(cw * ch);
        let mut d2 = Vec::with_capacity(cw * ch);
        for y in y0..y0 + ch {
            s2.extend_from_slice(&src[y * w + x0..y * w + x0 + cw]);
            d2.extend_from_slice(&dst[y * w + x0..y * w + x0 + cw]);
        }
        src = s2;
        dst = d2;
        w = cw;
        h = ch;
    }
    assert!(src != dst, "{label}: identity pair");
    LoadedPair {
        label,
        src,
        dst,
        w,
        h,
    }
}

struct Tier {
    label: &'static str,
    /// Tokens to force OFF for this measurement.
    disable: &'static [&'static str],
}

const TIERS: &[Tier] = &[
    Tier {
        label: "v4x",
        disable: &[],
    },
    Tier {
        label: "v4",
        disable: &["x86-64-v4x"],
    },
    Tier {
        label: "v3",
        disable: &["x86-64-v4x", "x86-64-v4"],
    },
    Tier {
        label: "scalar",
        disable: &[
            "x86-64-v4x",
            "x86-64-v4",
            "x86-64-v3 Crypto",
            "x86-64-v3",
            "x86-64 Crypto",
            "x86-64-v2",
        ],
    },
];

/// Toggle one named token-disable flag. Names match
/// `archmage::testing::for_each_token_permutation`'s labels.
fn set_token_disabled(name: &str, off: bool) {
    use archmage::{
        X64CryptoToken, X64V2Token, X64V3CryptoToken, X64V3Token, X64V4Token, X64V4xToken,
    };
    let r = match name {
        "x86-64-v4x" => X64V4xToken::dangerously_disable_token_process_wide(off),
        "x86-64-v4" => X64V4Token::dangerously_disable_token_process_wide(off),
        "x86-64-v3 Crypto" => X64V3CryptoToken::dangerously_disable_token_process_wide(off),
        "x86-64-v3" => X64V3Token::dangerously_disable_token_process_wide(off),
        "x86-64 Crypto" => X64CryptoToken::dangerously_disable_token_process_wide(off),
        "x86-64-v2" => X64V2Token::dangerously_disable_token_process_wide(off),
        _ => panic!("unknown token group {name}"),
    };
    if let Err(e) = r {
        // A compile-time-guaranteed token cannot be disabled — on this build
        // (no target-cpu pinning) that never happens; surface it loudly anyway.
        panic!("cannot set {name} disabled={off}: {e}");
    }
}

fn apply_tier(tier: &Tier) {
    // Re-enable everything first, then apply this tier's disable set —
    // keeps each measurement independent of ordering.
    for name in [
        "x86-64-v4x",
        "x86-64-v4",
        "x86-64-v3 Crypto",
        "x86-64-v3",
        "x86-64 Crypto",
        "x86-64-v2",
    ] {
        set_token_disabled(name, false);
    }
    for &name in tier.disable {
        set_token_disabled(name, true);
    }
}

fn main() {
    if let Ok(mode) = std::env::var("ZENSIM_FEATCANON")
        && !cfg!(feature = "featcanon-oracle")
    {
        eprintln!(
            "ZENSIM_FEATCANON={mode} needs zensim's measurement arithmetic, which this \
             build does not contain; rebuild with --features featcanon-oracle"
        );
        exit(2);
    }
    let mut pairs: Vec<LoadedPair> = Vec::new();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--pair" => {
                let (src, sw, sh) = decode_rgb(&args[i + 2]);
                let (dst, dw, dh) = decode_rgb(&args[i + 3]);
                assert_eq!((sw, sh), (dw, dh), "size mismatch");
                pairs.push(LoadedPair {
                    label: args[i + 1].clone(),
                    src,
                    dst,
                    w: sw,
                    h: sh,
                });
                i += 4;
            }
            tsv => {
                let text =
                    std::fs::read_to_string(tsv).unwrap_or_else(|e| panic!("read {tsv}: {e}"));
                for (n, line) in text.lines().enumerate() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    let c: Vec<&str> = line.split('\t').collect();
                    if n == 0 && c.first().is_some_and(|h| h.contains("path")) {
                        continue; // legacy header
                    }
                    if is_probe_line(&c) {
                        pairs.push(load_probe_line(line));
                    } else if c.len() >= 2 && !c[0].is_empty() && n > 0 {
                        // Legacy: ref_path \t dist_path [\t label]
                        let set = tsv
                            .rsplit('/')
                            .next()
                            .unwrap_or(tsv)
                            .trim_end_matches(".tsv");
                        let label = c
                            .get(2)
                            .filter(|s| !s.is_empty())
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| format!("{set}#{n}"));
                        let (src, sw, sh) = decode_rgb(c[0]);
                        let (dst, dw, dh) = decode_rgb(c[1]);
                        assert_eq!((sw, sh), (dw, dh), "{label}: size mismatch");
                        pairs.push(LoadedPair {
                            label,
                            src,
                            dst,
                            w: sw,
                            h: sh,
                        });
                    }
                }
                i += 1;
            }
        }
    }
    if pairs.is_empty() {
        eprintln!("no pairs given");
        exit(2);
    }

    // Sanity: confirm the host can actually summon v4/v3 — else the audit
    // silently measures degenerate tiers.
    {
        use archmage::SimdToken;
        eprintln!(
            "tokens: v4x={} v4={} v3={} formula_revision={:?} featcanon={}",
            archmage::X64V4xToken::summon().is_some(),
            archmage::X64V4Token::summon().is_some(),
            archmage::X64V3Token::summon().is_some(),
            zensim::feature_v2::active_formula_revision(),
            std::env::var("ZENSIM_FEATCANON").unwrap_or_else(|_| "prod".into()),
        );
    }

    let mut out = String::new();
    writeln!(
        out,
        "pair\ttier\tn_diff\tmax_abs_diff\tmax_rel_diff\tblock\tslot\tslot_name"
    )
    .unwrap();

    for p in &pairs {
        let label = &p.label;
        let (rs, rd) = (
            RgbSlice::new(&p.src, p.w, p.h),
            RgbSlice::new(&p.dst, p.w, p.h),
        );
        eprintln!("== {label} {}x{} ==", p.w, p.h);

        // v3 baseline first — canonical arithmetic.
        let dump_dir = std::env::var("ZENSIM_FEATCANON_DUMP").ok();
        let mode_tag = std::env::var("ZENSIM_FEATCANON").unwrap_or_else(|_| "prod".into());
        // featacc: the blur-recurrence axis is a second measurement knob —
        // fold it into the dump tag when explicitly set so `exact` and
        // `exact+fresh-blur` cannot collide on the filename.
        // `TIER_AUDIT_TAG` overrides the whole tag for axes this binary does
        // not know about (e.g. ZENSIM_ERA2_DENSE=0).
        let mode_tag = match (
            std::env::var("TIER_AUDIT_TAG"),
            std::env::var("ZENSIM_FEATCANON_BLUR"),
        ) {
            (Ok(t), _) => t,
            (Err(_), Ok(b)) => format!("{mode_tag}.blur-{b}"),
            (Err(_), Err(_)) => mode_tag,
        };
        let mut per_tier: Vec<(
            &'static str,
            Vec<f64>,
            Vec<zensim::research::FeatureProvenance>,
        )> = Vec::new();
        // `TIER_AUDIT_ONLY=<label>` restricts to one tier — for timing runs.
        let only = std::env::var("TIER_AUDIT_ONLY").ok();
        for tier in TIERS {
            if only.as_deref().is_some_and(|o| o != tier.label) {
                continue;
            }
            apply_tier(tier);
            let t0 = std::time::Instant::now();
            let rev = std::env::var("ZENSIM_FORMULA_REV")
                .ok()
                .and_then(|r| r.parse::<u32>().ok())
                .unwrap_or(0);
            let request = if rev >= 5 {
                use zensim::feature_set_id::ComputeToken;
                // Revisions 5+ compute basic, peaks and v2; revision 6 adds globaff.
                let mut slots = research::family_slots(ComputeToken::Basic)
                    .union(&research::family_slots(ComputeToken::Peaks))
                    .union(&research::family_slots(ComputeToken::V2));
                if rev >= 6 {
                    slots = slots.union(&research::family_slots(ComputeToken::GlobAff));
                }
                Request::for_slots(slots, research::full_width())
            } else {
                Request::everything()
            };
            let e = research::extract(&request, &rs, &rd)
                .unwrap_or_else(|e| panic!("{label} tier {}: {e}", tier.label));
            let secs = t0.elapsed().as_secs_f64();
            if let Some(dir) = &dump_dir {
                // Raw little-endian f64 values, one file per (mode, tier, pair).
                let path = format!("{dir}/{mode_tag}_{}_{label}.f64bin", tier.label);
                let bytes: Vec<u8> = e.values().iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(&path, &bytes).unwrap_or_else(|e| panic!("write {path}: {e}"));
            }
            eprintln!("   {} {secs:.3}s", tier.label);
            per_tier.push((tier.label, e.values().to_vec(), e.provenance().to_vec()));
        }
        if let Some(dir) = &dump_dir {
            // The slot-name table, once per pair (mode-independent).
            let mut names = String::new();
            for p in &per_tier[0].2 {
                writeln!(names, "{}\t{}\t{}", p.id, p.family, p.name).unwrap();
            }
            let path = format!("{dir}/names_{label}.tsv");
            std::fs::write(&path, names).unwrap_or_else(|e| panic!("write {path}: {e}"));
        }
        if only.is_some() {
            continue; // timing run: no comparison
        }
        let base = &per_tier.iter().find(|t| t.0 == "v3").unwrap().1;
        let prov = &per_tier[0].2;

        for (tlabel, vals, _) in &per_tier {
            if *tlabel == "v3" {
                continue;
            }
            let mut n_diff = 0usize;
            let mut worst_rel = 0.0f64;
            let mut worst_slot = usize::MAX;
            for (idx, (a, b)) in vals.iter().zip(base.iter()).enumerate() {
                if a.to_bits() == b.to_bits() {
                    continue;
                }
                let abs = (a - b).abs();
                let rel = abs / b.abs().max(1e-300);
                if rel > worst_rel {
                    worst_rel = rel;
                    worst_slot = idx;
                }
                n_diff += 1;
                writeln!(
                    out,
                    "{label}\t{tlabel}\t1\t{abs:.6e}\t{rel:.6e}\t{}\t{}\t{}",
                    prov[idx].family, prov[idx].id, prov[idx].name
                )
                .unwrap();
            }
            eprintln!(
                "   {tlabel}: {n_diff}/{} slots differ vs v3; worst rel {worst_rel:.3e} {}",
                vals.len(),
                if worst_slot == usize::MAX {
                    String::new()
                } else {
                    format!("at f{} {}", prov[worst_slot].id, prov[worst_slot].name)
                }
            );
        }
    }

    print!("{out}");
}
