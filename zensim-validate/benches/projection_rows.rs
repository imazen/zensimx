//! zenbench comparison of the old capture/clamp/gate walk and the planned walk.
//! Restoring the same input weights is included in BOTH timings. These are row
//! kernel costs, not end-to-end trainer throughput or scientific fits.
#[allow(dead_code)]
#[path = "../src/mlp_train/gate_projection.rs"]
mod gate_projection;
use gate_projection::{MaxPairRows, project_gate};
use zenbench::prelude::*;

#[inline(never)]
fn original_rows(w: &mut [f64], hidden: usize, positions: &[[usize; 2]]) {
    let mut cap = vec![0.0; positions.len()];
    for row in w.chunks_exact_mut(hidden) {
        for (c, &[_, v]) in cap.iter_mut().zip(positions) {
            *c = row[v];
        }
        for x in row.iter_mut() {
            *x = x.max(0.0);
        }
        for (&b, &[u, v]) in cap.iter().zip(positions) {
            let (a, b) = project_gate(row[u], b, 1.0, 1.0);
            row[u] = a;
            row[v] = b;
        }
    }
}

// From the registered ReLU rescue's native keep mask: 492 kept base inputs
// plus 492 derived products, 2345 total rows. The previous 1853-row prototype
// was a proxy; the benchmark below uses the observed native dimensions.
const ACTIVE: &[(u32, u32)] = &[
    (0, 156),
    (372, 393),
    (394, 422),
    (423, 451),
    (452, 480),
    (481, 509),
    (510, 538),
    (539, 567),
    (568, 596),
    (597, 625),
    (626, 654),
    (655, 683),
    (684, 712),
    (713, 720),
    (1853, 2345),
];

fn benches(suite: &mut Suite) {
    let hidden = 128;
    for indexed in [false, true] {
        for feasible in [false, true] {
            let positions: Vec<_> = (0..32)
                .map(|i| {
                    if indexed {
                        [hidden - 1 - 2 * i, hidden - 2 - 2 * i]
                    } else {
                        [2 * i, 2 * i + 1]
                    }
                })
                .collect();
            let mut r = 0x51515eed_u64;
            let mut input: Vec<_> = (0..2345 * hidden)
                .map(|_| {
                    r ^= r << 13;
                    r ^= r >> 7;
                    r ^= r << 17;
                    (r >> 11) as f64 / (1u64 << 53) as f64 - 0.5
                })
                .collect();
            // Include the exact zero rows in the current keep-features shape.
            for (i, row) in input.chunks_exact_mut(hidden).enumerate() {
                if !ACTIVE
                    .iter()
                    .any(|&(a, b)| a as usize <= i && i < b as usize)
                {
                    row.fill(0.0);
                }
            }
            if feasible {
                original_rows(&mut input, hidden, &positions);
                for x in &mut input {
                    if *x != 0.0 {
                        *x -= 0.00001;
                    }
                }
            }
            let name = format!(
                "project_2345x128_32_{}_{}",
                if indexed { "permuted" } else { "contiguous" },
                if feasible { "near_feasible" } else { "initial" }
            );
            suite.group(name, |g| {
                let source = input.clone();
                let pos = positions.clone();
                let mut w = vec![0.0; source.len()];
                g.bench("capture_clamp_gate", move |b| {
                    b.iter(|| {
                        w.copy_from_slice(black_box(&source));
                        original_rows(black_box(&mut w), black_box(hidden), black_box(&pos));
                        black_box(&w);
                    })
                });
                let source = input.clone();
                let rows = MaxPairRows::new(hidden, positions.clone());
                let mut w = vec![0.0; source.len()];
                g.bench("planned_active_pairs", move |b| {
                    b.iter(|| {
                        w.copy_from_slice(black_box(&source));
                        rows.project_active_rows(black_box(&mut w), Some(black_box(ACTIVE)));
                        black_box(&w);
                    })
                });
            });
        }
    }
}
zenbench::main!(benches);
