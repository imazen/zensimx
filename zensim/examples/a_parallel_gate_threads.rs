//! Bitwise pool invariance for an explicit revision-7 bake.
//!
//! `ZENSIM_FORMULA_REV=7 cargo run --release -p zensim --features
//! custom-profiles,training,classification --example a_parallel_gate_threads -- <bake>`
//! Uses the split-parity pairs, shared pool-sweep geometries and bake read set.

include!("split_parity/pairs.rs");

#[path = "../tests/common/generators.rs"]
mod generators;
#[path = "../tests/common/parity_cells.rs"]
mod parity_cells;

use zensim::research::{self, Request};
use zensim::{BakeScorer, RgbSlice};

type Snapshot = (u64, u64, Vec<u64>, Vec<u64>);

fn collect(bytes: &[u8], pairs: &[SplitPair], parallel: bool) -> Vec<Snapshot> {
    let model = zenpredict::Model::from_bytes(bytes).expect("parse revision-7 bake");
    let mut scorer = BakeScorer::new(&model)
        .expect("servable bake")
        .with_parallel(parallel);
    let request = Request::for_bake_bytes(bytes)
        .expect("bake read set")
        .with_parallel(parallel);
    request
        .validate()
        .expect("revision-7 read set is supported");
    pairs
        .iter()
        .map(|pair| {
            let src = RgbSlice::new(bytemuck::cast_slice(&pair.src), pair.width, pair.height);
            let dst = RgbSlice::new(bytemuck::cast_slice(&pair.dst), pair.width, pair.height);
            let result = scorer.compute(&src, &dst, None).expect("score pair");
            let extracted = research::extract(&request, &src, &dst).expect("extract read set");
            assert!(result.score().is_finite() && result.raw_distance().is_finite());
            (
                result.score().to_bits(),
                result.raw_distance().to_bits(),
                result.features().iter().map(|v| v.to_bits()).collect(),
                extracted.values().iter().map(|v| v.to_bits()).collect(),
            )
        })
        .collect()
}

fn main() {
    assert_eq!(std::env::var("ZENSIM_FORMULA_REV").as_deref(), Ok("7"));
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert_eq!(args.len(), 1, "usage: a_parallel_gate_threads <bake>");
    let bytes = std::fs::read(&args[0]).expect("read required bake");
    let mut pairs = split_parity_pairs();
    for (width, height) in parity_cells::pool_sweep_cells() {
        let src = generators::gen_value_noise(width, height, 0xC0FFEE);
        let dst = generators::distort_block_artifacts(&src, width, height);
        pairs.push(SplitPair {
            id: format!("pool-sweep-{width}x{height}"),
            width,
            height,
            src: bytemuck::cast_slice(&src).to_vec(),
            dst: bytemuck::cast_slice(&dst).to_vec(),
        });
    }
    let serial = collect(&bytes, &pairs, false);
    for threads in [1usize, 2, 3, 4, 8, 16] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("build pool");
        let got = pool.install(|| collect(&bytes, &pairs, true));
        assert_eq!(got.len(), serial.len());
        for ((expected, actual), pair) in serial.iter().zip(&got).zip(&pairs) {
            assert_eq!(expected.0, actual.0, "{threads}T {} score", pair.id);
            assert_eq!(expected.1, actual.1, "{threads}T {} raw distance", pair.id);
            for (surface, a, b) in [
                ("scorer features", &expected.2, &actual.2),
                ("research features", &expected.3, &actual.3),
            ] {
                assert_eq!(a.len(), b.len(), "{threads}T {} {surface} width", pair.id);
                for (index, (a, b)) in a.iter().zip(b).enumerate() {
                    assert_eq!(a, b, "{threads}T {} {surface} f{index}", pair.id);
                }
            }
        }
        println!(
            "{threads}T: {}/{} split pairs match serial score, distance and feature bits",
            got.len(),
            pairs.len()
        );
    }
}
