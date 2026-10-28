//! Per-call wall time of one profile on one generated pair, single-threaded.
//!
//! The research half of the slim-versus-research paired timing (the slim crate
//! imazen/zensim has a twin with the same arguments and output). The revision-7
//! profiles are served from bake files through `ZensimProfile::Custom`, the same
//! production path the slim crate's named profiles take.
//!
//! ```text
//! cargo run --release -p zensim --features custom-profiles --example split_timing -- \
//!     <E|ECodec|A|PreviewV0_2> <pair-id> <warmup> <calls> [<general.bin> <codec.bin>]
//! ```
//! Prints one line per timed call: nanoseconds.
#![allow(deprecated)]

include!("split_parity/pairs.rs");

use std::hint::black_box;
use std::sync::OnceLock;
use std::time::Instant;
use zensim::profile::ProfileParams;
use zensim::{RgbSlice, Zensim, ZensimProfile};

static BAKE: OnceLock<&'static [u8]> = OnceLock::new();

fn bake() -> &'static [u8] {
    BAKE.get().unwrap()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let rev7 = |path: &str| {
        let bytes: &'static [u8] = Box::leak(std::fs::read(path).expect("read bake").into_boxed_slice());
        BAKE.set(bytes).unwrap();
        let params: &'static ProfileParams = Box::leak(Box::new(
            ProfileParams::builder()
                .skip_score_mapping(true)
                .mlp(bake)
                .extended_features(true)
                .compute_iw_features(true)
                .extrapolate_score(true)
                .build(),
        ));
        ZensimProfile::Custom { params, name: "rev7" }
    };
    let profile = match a[0].as_str() {
        "E" => rev7(&a[4]),
        "ECodec" => rev7(&a[5]),
        "A" => ZensimProfile::A,
        "PreviewV0_2" => ZensimProfile::PreviewV0_2,
        other => panic!("unknown profile {other}"),
    };
    let pair = split_parity_pairs().into_iter().find(|p| p.id == a[1]).expect("pair id");
    let (warmup, calls): (usize, usize) = (a[2].parse().unwrap(), a[3].parse().unwrap());
    let src = RgbSlice::new(bytemuck::cast_slice(&pair.src), pair.width, pair.height);
    let dst = RgbSlice::new(bytemuck::cast_slice(&pair.dst), pair.width, pair.height);
    let z = Zensim::new(profile).with_parallel(false);
    for _ in 0..warmup {
        black_box(z.compute(&src, &dst).unwrap());
    }
    for _ in 0..calls {
        let t = Instant::now();
        let r = z.compute(&src, &dst).unwrap();
        let dt = t.elapsed();
        black_box(r);
        println!("{}", dt.as_nanos());
    }
}
