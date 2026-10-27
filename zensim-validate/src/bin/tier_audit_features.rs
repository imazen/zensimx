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
//! Native HDR: `--hdr-pq <label> <ref.png> <dist.png|perturb:KIND>
//! <srgb|p3|bt2020> <peak_nits>` reads 16-bit RGB PNG PQ code values as
//! opaque `Srgb16Rgba` and extracts through `research::extract_hdr` with
//! `HdrEncoding::Pq { peak_nits }`, requesting basic + peaks + v2 at
//! revisions 5 and later. `perturb:KIND` derives the distorted image from the
//! reference in code space instead of reading a file (see [`perturb`]).
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

use zensim::feature_v2::HdrEncoding;
use zensim::research::{self, Request};
use zensim::{AlphaMode, ColorPrimaries, PixelFormat, RgbSlice, StridedBytes};

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

/// A native PQ pair (`--hdr-pq`): opaque RGBA u16 code values.
struct HdrPair {
    label: String,
    src: Vec<[u16; 4]>,
    dst: Vec<[u16; 4]>,
    w: usize,
    h: usize,
    primaries: ColorPrimaries,
    peak_nits: f32,
}

/// One measured pair, SDR RGB8 or native HDR.
enum Input {
    Sdr(LoadedPair),
    Hdr(HdrPair),
}

impl Input {
    fn label(&self) -> &str {
        match self {
            Input::Sdr(p) => &p.label,
            Input::Hdr(p) => &p.label,
        }
    }

    fn dims(&self) -> (usize, usize) {
        match self {
            Input::Sdr(p) => (p.w, p.h),
            Input::Hdr(p) => (p.w, p.h),
        }
    }

    fn extract(&self, request: &Request) -> Result<research::Extraction, research::ResearchError> {
        match self {
            Input::Sdr(p) => research::extract(
                request,
                &RgbSlice::new(&p.src, p.w, p.h),
                &RgbSlice::new(&p.dst, p.w, p.h),
            ),
            Input::Hdr(p) => {
                fn bytes(px: &[[u16; 4]]) -> Vec<u8> {
                    px.iter().flatten().flat_map(|v| v.to_ne_bytes()).collect()
                }
                fn view<'a>(bytes: &'a [u8], p: &HdrPair) -> StridedBytes<'a> {
                    StridedBytes::with_alpha_mode(
                        bytes,
                        p.w,
                        p.h,
                        p.w * 8,
                        PixelFormat::Srgb16Rgba,
                        AlphaMode::Opaque,
                    )
                    .with_color_primaries(p.primaries)
                }
                let (src, dst) = (bytes(&p.src), bytes(&p.dst));
                research::extract_hdr(
                    request,
                    &view(&src, p),
                    &view(&dst, p),
                    HdrEncoding::Pq {
                        peak_nits: p.peak_nits,
                    },
                )
            }
        }
    }
}

fn decode_rgb16(path: &str) -> (Vec<[u16; 4]>, usize, usize) {
    let img = image::open(path).unwrap_or_else(|e| panic!("decode {path}: {e}"));
    let rgb = img.to_rgb16();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let raw = rgb.into_raw();
    let (chunks, rem) = raw.as_chunks::<3>();
    assert!(
        rem.is_empty(),
        "{path}: RGB16 sample count not divisible by 3"
    );
    (
        chunks
            .iter()
            .map(|p| [p[0], p[1], p[2], u16::MAX])
            .collect(),
        w,
        h,
    )
}

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic code-space distortions for precision probes, so a pair needs
/// no second file: `dither1` moves about one channel value in ten by one code
/// (sign and position from a fixed hash), `shift4`/`shift64` add 4/64 codes to
/// every channel, `blend5` moves 5% toward an edge-clamped 3×3 box mean.
/// Results round to the nearest code and clamp to [0, 65535].
fn perturb(src: &[[u16; 4]], w: usize, h: usize, kind: &str) -> Vec<[u16; 4]> {
    let code = |v: f64| v.round().clamp(0.0, 65535.0) as u16;
    let mut out = src.to_vec();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            for c in 0..3 {
                let v = f64::from(src[i][c]);
                out[i][c] = match kind {
                    "dither1" => {
                        let r = splitmix64((i * 3 + c) as u64);
                        if !r.is_multiple_of(10) {
                            continue;
                        }
                        code(v + if r >> 63 == 1 { 1.0 } else { -1.0 })
                    }
                    "shift4" => code(v + 4.0),
                    "shift64" => code(v + 64.0),
                    "blend5" => {
                        let mut sum = 0.0;
                        for dy in [-1isize, 0, 1] {
                            for dx in [-1isize, 0, 1] {
                                let yy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                                let xx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                                sum += f64::from(src[yy * w + xx][c]);
                            }
                        }
                        code(0.95 * v + 0.05 * sum / 9.0)
                    }
                    other => panic!("unknown perturbation {other}"),
                };
            }
        }
    }
    out
}

fn load_hdr_pq(args: &[String]) -> HdrPair {
    let [label, reference, distorted, primaries, peak] = args else {
        panic!("--hdr-pq needs <label> <ref.png> <dist.png|perturb:KIND> <primaries> <peak_nits>");
    };
    let (src, w, h) = decode_rgb16(reference);
    let dst = match distorted.strip_prefix("perturb:") {
        Some(kind) => perturb(&src, w, h, kind),
        None => {
            let (dst, dw, dh) = decode_rgb16(distorted);
            assert_eq!((w, h), (dw, dh), "{label}: size mismatch");
            dst
        }
    };
    assert!(src != dst, "{label}: identity pair");
    let primaries = match primaries.as_str() {
        "srgb" => ColorPrimaries::Srgb,
        "p3" => ColorPrimaries::DisplayP3,
        "bt2020" => ColorPrimaries::Bt2020,
        other => panic!("{label}: unknown primaries {other}"),
    };
    HdrPair {
        label: label.clone(),
        src,
        dst,
        w,
        h,
        primaries,
        peak_nits: peak
            .parse()
            .unwrap_or_else(|e| panic!("{label}: peak {peak}: {e}")),
    }
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
    let mut pairs: Vec<Input> = Vec::new();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--pair" => {
                let (src, sw, sh) = decode_rgb(&args[i + 2]);
                let (dst, dw, dh) = decode_rgb(&args[i + 3]);
                assert_eq!((sw, sh), (dw, dh), "size mismatch");
                pairs.push(Input::Sdr(LoadedPair {
                    label: args[i + 1].clone(),
                    src,
                    dst,
                    w: sw,
                    h: sh,
                }));
                i += 4;
            }
            "--hdr-pq" => {
                pairs.push(Input::Hdr(load_hdr_pq(&args[i + 1..i + 6])));
                i += 6;
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
                        pairs.push(Input::Sdr(load_probe_line(line)));
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
                        pairs.push(Input::Sdr(LoadedPair {
                            label,
                            src,
                            dst,
                            w: sw,
                            h: sh,
                        }));
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
        let label = p.label();
        let (w, h) = p.dims();
        eprintln!("== {label} {w}x{h} ==");

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
                // Revisions 5+ compute basic, peaks and v2; revision 6 adds
                // globaff, which the native HDR walk is not measured with.
                let mut slots = research::family_slots(ComputeToken::Basic)
                    .union(&research::family_slots(ComputeToken::Peaks))
                    .union(&research::family_slots(ComputeToken::V2));
                if rev >= 6 && matches!(p, Input::Sdr(_)) {
                    slots = slots.union(&research::family_slots(ComputeToken::GlobAff));
                }
                Request::for_slots(slots, research::full_width())
            } else {
                Request::everything()
            };
            let e = p
                .extract(&request)
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
