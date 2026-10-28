//! Reference values for the repository-split parity gate.
//!
//! The slim published crate (imazen/zensim, cut from this one on 2026-10-28) must score every
//! pair bit-identically to this research crate for the profiles it keeps: `PreviewV0_1`,
//! `PreviewV0_2`, `A`, and the two curated revision-7 models. This example prints one TSV line
//! per (surface, profile, pair) with the f64/f32 bits of the result; the slim crate's
//! `split_parity_check` example prints the same lines from its own API, and the two outputs are
//! compared with `diff`.
//!
//! The revision-7 models are loaded from bake files and served through
//! `ZensimProfile::Custom` (the production `Zensim::compute` path) and through `BakeScorer`
//! (the dynamic surface the runtime gate measured); both rows are printed so the profile path
//! is proven equal to the measured surface. Optional third and fourth bakes are the
//! metadata-stripped copies the slim crate embeds; their rows prove stripping changed no score.
//!
//! ```text
//! cargo run --release -p zensim --features custom-profiles,training,classification --example split_parity_golden -- \
//!     <general.bin> <codec.bin> [<general-stripped.bin> <codec-stripped.bin>] > golden.tsv
//! ```
#![allow(deprecated)]

include!("split_parity/pairs.rs");
include!("split_parity/surfaces.rs");

use std::sync::OnceLock;
use zensim::profile::ProfileParams;
use zensim::{DiffmapOptions, RgbSlice, Zensim, ZensimProfile, ZensimScratch};

static BAKES: OnceLock<Vec<&'static [u8]>> = OnceLock::new();

fn bake0() -> &'static [u8] {
    BAKES.get().unwrap()[0]
}
fn bake1() -> &'static [u8] {
    BAKES.get().unwrap()[1]
}
fn bake2() -> &'static [u8] {
    BAKES.get().unwrap()[2]
}
fn bake3() -> &'static [u8] {
    BAKES.get().unwrap()[3]
}

fn rev7_profile(bytes: fn() -> &'static [u8], name: &'static str) -> ZensimProfile {
    // The same runtime shape as the shipped MLP profiles (`PROFILE_B` in profile.rs).
    let params: &'static ProfileParams = Box::leak(Box::new(
        ProfileParams::builder()
            .skip_score_mapping(true)
            .mlp(bytes)
            .extended_features(true)
            .compute_iw_features(true)
            .extrapolate_score(true)
            .build(),
    ));
    ZensimProfile::Custom { params, name }
}

fn fnv64(words: impl Iterator<Item = u64>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for w in words {
        for b in w.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn f64s(v: &[f64]) -> u64 {
    fnv64(v.iter().map(|x| x.to_bits()))
}

fn f32s(v: &[f32]) -> u64 {
    fnv64(v.iter().map(|x| x.to_bits() as u64))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(args.len() == 2 || args.len() == 4, "usage: <general.bin> <codec.bin> [<general-stripped.bin> <codec-stripped.bin>]");
    let bytes: Vec<&'static [u8]> = args
        .iter()
        .map(|p| &*Box::leak(std::fs::read(p).expect("read bake").into_boxed_slice()))
        .collect();
    BAKES.set(bytes).unwrap();

    let mut profiles: Vec<(&str, ZensimProfile)> = vec![
        ("PreviewV0_1", ZensimProfile::PreviewV0_1),
        ("PreviewV0_2", ZensimProfile::PreviewV0_2),
        ("A", ZensimProfile::A),
        ("E", rev7_profile(bake0, "rev7-general")),
        ("ECodec", rev7_profile(bake1, "rev7-codec")),
    ];
    if args.len() == 4 {
        profiles.push(("E-stripped", rev7_profile(bake2, "rev7-general-stripped")));
        profiles.push(("ECodec-stripped", rev7_profile(bake3, "rev7-codec-stripped")));
    }
    let models: Vec<(&str, zenpredict::Model)> = [("E", 0usize), ("ECodec", 1)]
        .into_iter()
        .map(|(n, i)| (n, zenpredict::Model::from_bytes(BAKES.get().unwrap()[i]).expect("parse bake")))
        .collect();

    let pairs = split_parity_pairs();
    let hex = |x: f64| format!("{:016x}", x.to_bits());
    let line = |surface: &str, profile: &str, pair: &str, value: String| println!("{surface}\t{profile}\t{pair}\t{value}");
    for pair in &pairs {
        let src = RgbSlice::new(bytemuck::cast_slice(&pair.src), pair.width, pair.height);
        let dst = RgbSlice::new(bytemuck::cast_slice(&pair.dst), pair.width, pair.height);
        for (pname, profile) in &profiles {
            let z = Zensim::new(*profile);
            let v = match z.compute(&src, &dst) {
                Ok(r) => format!("{} {} {:016x}", hex(r.score()), hex(r.raw_distance()), f64s(r.features())),
                Err(e) => format!("ERR {e:?}"),
            };
            line("compute", pname, &pair.id, v);
            let pre = z.precompute_reference(&src);
            let v = match &pre {
                Ok(pre) => match z.compute_with_ref(pre, &dst) {
                    Ok(r) => format!("{} {}", hex(r.score()), hex(r.raw_distance())),
                    Err(e) => format!("ERR {e:?}"),
                },
                Err(e) => format!("ERR {e:?}"),
            };
            line("compute_with_ref", pname, &pair.id, v);
            let v = match &pre {
                Ok(pre) => {
                    let mut scratch = ZensimScratch::new();
                    match z.compute_with_ref_into(pre, &dst, &mut scratch) {
                        Ok(r) => format!("{} {}", hex(r.score()), hex(r.raw_distance())),
                        Err(e) => format!("ERR {e:?}"),
                    }
                }
                Err(e) => format!("ERR {e:?}"),
            };
            line("compute_with_ref_into", pname, &pair.id, v);
            let v = match z.compute_with_diffmap(&src, &dst, DiffmapOptions::default()) {
                Ok(d) => format!("{} {:016x}", hex(d.score()), f32s(d.diffmap())),
                Err(e) => format!("ERR {e:?}"),
            };
            line("compute_with_diffmap", pname, &pair.id, v);
            let v = match &pre {
                Ok(pre) => match z.compute_with_ref_and_diffmap(pre, &dst, DiffmapOptions::default()) {
                    Ok(d) => format!("{} {:016x}", hex(d.score()), f32s(d.diffmap())),
                    Err(e) => format!("ERR {e:?}"),
                },
                Err(e) => format!("ERR {e:?}"),
            };
            line("compute_with_ref_and_diffmap", pname, &pair.id, v);
            let v = match z.compute_all_features(&src, &dst) {
                Ok(r) => format!("{} {:016x}", hex(r.score()), f64s(r.features())),
                Err(e) => format!("ERR {e:?}"),
            };
            line("compute_all_features", pname, &pair.id, v);
        }
        for (pname, model) in &models {
            let v = match zensim::BakeScorer::new(model) {
                Ok(s) => match s.with_parallel(true).compute(&src, &dst, None) {
                    Ok(r) => format!("{} {} {:016x}", hex(r.score()), hex(r.raw_distance()), f64s(r.features())),
                    Err(e) => format!("ERR {e:?}"),
                },
                Err(e) => format!("ERR {e:?}"),
            };
            line("bake_scorer", pname, &pair.id, v);
        }
    }
    extra_surfaces(&profiles[..5], &pairs, &line);
}
