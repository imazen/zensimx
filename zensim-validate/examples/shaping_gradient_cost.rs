//! Test the profiled shaper gradient's two dot products without changing training.
#![forbid(unsafe_code)]

use anyhow::{Result, ensure};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

const WIDTH: usize = 128;
const FEATURES: usize = 984;

fn separate(weights: &[f64], a: &[f64], b: &[f64]) -> (f64, f64) {
    let ga = weights.iter().zip(a).map(|(w, d)| w * d).sum();
    let gb = weights.iter().zip(b).map(|(w, d)| w * d).sum();
    (ga, gb)
}

fn paired(weights: &[f64], a: &[f64], b: &[f64]) -> (f64, f64) {
    assert_eq!(weights.len(), a.len());
    assert_eq!(a.len(), b.len());
    // Use the same additive identity as Iterator::sum, including signed zero.
    let zero: f64 = std::iter::empty::<f64>().sum();
    let (mut ga, mut gb) = (zero, zero);
    for ((w, da), db) in weights.iter().zip(a).zip(b) {
        ga += w * da;
        gb += w * db;
    }
    (ga, gb)
}

#[inline(never)]
fn separate_bank(weights: &[f64], a: &[f64], b: &[f64]) -> (f64, f64) {
    let mut result = (0., 0.);
    for row in weights.as_chunks::<WIDTH>().0 {
        let (ga, gb) = separate(row, a, b);
        result.0 += ga;
        result.1 += gb;
    }
    result
}

#[inline(never)]
fn paired_bank(weights: &[f64], a: &[f64], b: &[f64]) -> (f64, f64) {
    let mut result = (0., 0.);
    for row in weights.as_chunks::<WIDTH>().0 {
        let (ga, gb) = paired(row, a, b);
        result.0 += ga;
        result.1 += gb;
    }
    result
}

fn bits(pair: (f64, f64)) -> (u64, u64) {
    (pair.0.to_bits(), pair.1.to_bits())
}

fn next(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5) * 2.
}

fn qualify() {
    let mut state = 20261109;
    for width in [0, 1, 2, 3, 7, 31, 32, 63, 64, 127, 128, 129] {
        for _ in 0..128 {
            let w: Vec<_> = (0..width).map(|_| next(&mut state)).collect();
            let a: Vec<_> = (0..width).map(|_| next(&mut state) * 1e-8).collect();
            let b: Vec<_> = (0..width).map(|_| next(&mut state) * 1e8).collect();
            assert_eq!(bits(separate(&w, &a, &b)), bits(paired(&w, &a, &b)));
        }
        for zero in [0., -0.] {
            let w = vec![1.; width];
            let a = vec![zero; width];
            let b = vec![-zero; width];
            assert_eq!(bits(separate(&w, &a, &b)), bits(paired(&w, &a, &b)));
        }
    }
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 2,
        "usage: shaping_gradient_cost NEW_OUTPUT_DIR"
    );
    let output = Path::new(&args[1]);
    std::fs::create_dir(output)?;
    qualify();
    let mut state = 20261109;
    let weights: Arc<Vec<_>> = Arc::new((0..FEATURES * WIDTH).map(|_| next(&mut state)).collect());
    let a: Arc<Vec<_>> = Arc::new((0..WIDTH).map(|_| next(&mut state)).collect());
    let b: Arc<Vec<_>> = Arc::new((0..WIDTH).map(|_| next(&mut state)).collect());
    ensure!(
        bits(separate_bank(&weights, &a, &b)) == bits(paired_bank(&weights, &a, &b)),
        "full-bank gradient arithmetic differs"
    );
    let result = zenbench::run(|suite| {
        suite.compare("984_features_128_hidden_two_gradients", |group| {
            group
                .config()
                .min_rounds(32)
                .max_rounds(32)
                .auto_rounds(false)
                .warmup_time(Duration::from_millis(100))
                .max_wall_time(Duration::from_secs(120));
            for (name, kernel) in [
                (
                    "two_passes",
                    separate_bank as fn(&[f64], &[f64], &[f64]) -> (f64, f64),
                ),
                ("paired_pass", paired_bank),
            ] {
                let (weights, a, b) = (Arc::clone(&weights), Arc::clone(&a), Arc::clone(&b));
                group.bench(name, move |bench| {
                    bench.iter(|| {
                        zenbench::black_box(kernel(
                            zenbench::black_box(&weights),
                            zenbench::black_box(&a),
                            zenbench::black_box(&b),
                        ));
                    });
                });
            }
        });
    });
    result.save(output.join("zenbench.json"))?;
    ensure!(
        result.comparisons.len() == 1,
        "one paired comparison required"
    );
    let samples = &result.comparisons[0].samples;
    let complete = !result.unreliable
        && samples.len() == 32
        && samples
            .iter()
            .all(|s| s.gate_clean == Some(true) && s.elapsed_ns.len() == 2);
    std::fs::write(
        output.join("RESULT.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "complete":complete,"preflight_exact":true,"features":FEATURES,"hidden":WIDTH,
            "rounds":samples.len(),"unreliable":result.unreliable,"training_changed":false,
            "scope":"synthetic dot-product bank only; excludes the remainder of the shaper step and training"
        }))?,
    )?;
    ensure!(
        complete,
        "timing gate refused; preserved evidence is not a speed claim"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn same_order_and_signed_zero() {
        super::qualify();
    }
}
