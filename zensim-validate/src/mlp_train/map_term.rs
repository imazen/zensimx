//! `--map-pool`: fit the model's attribution map to a spatial target (E36 experiment 3,
//! `benchmarks/e36_subset_test_2026-11-01.md`).
//!
//! A view's map is `m_b = sign * sum_k (dy/df_k) I_k(b)`: the raw output's sensitivity to each read-set feature
//! times that feature's block integrand (`BakeScorer::feature_block_integrands`). The term is
//! `weight * mean over drawn views of (1 - pearson(m, t))`, with `t` the view's block target (a within-view rank,
//! scaled to zero mean and unit variance). The sensitivities run back through the derived inputs (direct columns
//! and difference x reference products) and the one-hidden-layer network. With `g = dL/dm` and `v` the input
//! direction `sum_b g_b I(b)` pushed through the derived-input Jacobian, the loss moves with
//! `Phi = sum_j w2_j slope_j (W1 v)_j`, so `dPhi/dw2_j = slope_j (W1 v)_j` and `dPhi/dW1[c][j] = w2_j slope_j v_c`.
//! The leaky slope is piecewise constant, so no other term survives.

use super::{AdamState, SplitMix64};

/// Where a kept network input comes from in the raw read set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MapColumn {
    /// The raw feature at this read-set position.
    Direct(usize),
    /// The product of the raw features at these read-set positions.
    Product(usize, usize),
}

/// One view of a map pool.
#[derive(Clone, Debug)]
pub struct MapView {
    /// Kept-column derived row, before standardization (the order of [`MapPool::columns`]).
    pub derived: Vec<f64>,
    /// Raw read-set row.
    pub raw: Vec<f64>,
    /// Read-set position of each integrand row.
    pub positions: Vec<usize>,
    pub blocks: usize,
    /// Integrand rows (`positions.len() x blocks`, row-major).
    pub integrands: Vec<f32>,
    /// Block target, zero mean and unit variance.
    pub target: Vec<f64>,
}

/// `--map-pool`: views, the kept columns' sources and the term's settings.
#[derive(Clone, Debug)]
pub struct MapPool {
    pub views: Vec<MapView>,
    /// Per kept network input, its source in the raw read set.
    pub columns: Vec<MapColumn>,
    pub read_width: usize,
    pub weight: f64,
    pub batch: usize,
    /// `+1` or `-1`: the served map is `dscore/df`, and a distance-shaped raw output runs opposite to the score.
    pub sign: f64,
}

/// The term's loss on one view and, when `grads` is given, its gradient scaled by `scale`.
#[allow(clippy::too_many_arguments)]
fn view_loss(
    pool: &MapPool,
    view: &MapView,
    net_columns: &[usize],
    mean: &[f64],
    std: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    nf: usize,
    nh: usize,
    alpha: f64,
    grads: Option<(f64, &mut [f64], &mut [f64])>,
) -> f64 {
    let mut x = vec![0.0; nf];
    for (k, &c) in net_columns.iter().enumerate() {
        x[c] = (view.derived[k] - mean[k]) / std[k];
    }
    let (_, pre, ..) = super::forward_pair(&x, &x, w1, b1, w2, b2, nf, nh, alpha);
    let slope: Vec<f64> = pre
        .iter()
        .map(|&p| if p >= 0.0 { 1.0 } else { alpha })
        .collect();
    let a: Vec<f64> = (0..nh).map(|j| w2[j] * slope[j]).collect();
    // dy/df over the raw read set.
    let mut s = vec![0.0; pool.read_width];
    for (k, (&c, col)) in net_columns.iter().zip(&pool.columns).enumerate() {
        let row = &w1[c * nh..(c + 1) * nh];
        let dydx: f64 = row.iter().zip(&a).map(|(w, a)| w * a).sum::<f64>() / std[k];
        match *col {
            MapColumn::Direct(p) => s[p] += dydx,
            MapColumn::Product(p, q) => {
                s[p] += dydx * view.raw[q];
                s[q] += dydx * view.raw[p];
            }
        }
    }
    let nb = view.blocks;
    let mut m = vec![0.0; nb];
    for (r, &p) in view.positions.iter().enumerate() {
        let coef = pool.sign * s[p];
        if coef != 0.0 {
            for (mb, &i) in m.iter_mut().zip(&view.integrands[r * nb..(r + 1) * nb]) {
                *mb += coef * f64::from(i);
            }
        }
    }
    let n = nb as f64;
    let mbar = m.iter().sum::<f64>() / n;
    let sd = (m.iter().map(|v| (v - mbar) * (v - mbar)).sum::<f64>() / n).sqrt();
    if nb < 2 || sd < 1e-12 {
        return 0.0;
    }
    let corr = m
        .iter()
        .zip(&view.target)
        .map(|(v, t)| (v - mbar) * t)
        .sum::<f64>()
        / (n * sd);
    let Some((scale, gw1, gw2)) = grads else {
        return 1.0 - corr;
    };
    // dL/dm_b for L = 1 - corr, with t zero mean and unit variance.
    let gm: Vec<f64> = m
        .iter()
        .zip(&view.target)
        .map(|(v, t)| -(t - corr * (v - mbar) / sd) / (n * sd))
        .collect();
    let mut g = vec![0.0; pool.read_width];
    for (r, &p) in view.positions.iter().enumerate() {
        g[p] += pool.sign
            * gm.iter()
                .zip(&view.integrands[r * nb..(r + 1) * nb])
                .map(|(gm, &i)| gm * f64::from(i))
                .sum::<f64>();
    }
    let mut u = vec![0.0; nh];
    let mut v = vec![0.0; net_columns.len()];
    for (k, (&c, col)) in net_columns.iter().zip(&pool.columns).enumerate() {
        v[k] = match *col {
            MapColumn::Direct(p) => g[p],
            MapColumn::Product(p, q) => view.raw[q] * g[p] + view.raw[p] * g[q],
        } / std[k];
        if v[k] != 0.0 {
            for (uj, w) in u.iter_mut().zip(&w1[c * nh..(c + 1) * nh]) {
                *uj += w * v[k];
            }
        }
    }
    for j in 0..nh {
        gw2[j] += scale * slope[j] * u[j];
    }
    for (k, &c) in net_columns.iter().enumerate() {
        if v[k] != 0.0 {
            for j in 0..nh {
                gw1[c * nh + j] += scale * a[j] * v[k];
            }
        }
    }
    1.0 - corr
}

/// Whether the first scheduled batch's loss has been printed.
static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Draw `pool.batch` views on `rng` and accumulate the term's gradient into `adam`; returns the term's loss.
#[allow(clippy::too_many_arguments)]
pub(super) fn accumulate(
    pool: &MapPool,
    rng: &mut SplitMix64,
    net_columns: &[usize],
    mean: &[f64],
    std: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    adam: &mut AdamState,
    nf: usize,
    nh: usize,
    alpha: f64,
) -> f64 {
    let n = pool.views.len() as u64;
    let limit = u64::MAX - (u64::MAX % n);
    let scale = pool.weight / pool.batch as f64;
    let mut loss = 0.0;
    let first = !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed);
    for _ in 0..pool.batch {
        let draw = loop {
            let v = rng.next_u64();
            if v < limit {
                break v;
            }
        };
        let view = &pool.views[(draw % n) as usize];
        loss += scale
            * view_loss(
                pool,
                view,
                net_columns,
                mean,
                std,
                w1,
                b1,
                w2,
                b2,
                nf,
                nh,
                alpha,
                Some((scale, &mut adam.gw1, &mut adam.gw2)),
            );
    }
    if first {
        // The sign check: a map that agrees with its target starts well below 1 - corr = 1.
        println!(
            "  map term: first batch mean 1-pearson {:.4} over {} views",
            loss / pool.weight,
            pool.batch
        );
    }
    loss
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (MapPool, Vec<usize>, Vec<f64>, Vec<f64>) {
        let mut r = SplitMix64::new(7);
        let mut u = || r.next_f64_unit() * 2.0 - 1.0;
        // Raw read set of 4; kept inputs: direct 0, direct 2, product (0, 3), placed at network columns 0, 2, 5.
        let columns = vec![
            MapColumn::Direct(0),
            MapColumn::Direct(2),
            MapColumn::Product(0, 3),
        ];
        let net_columns = vec![0, 2, 5];
        let views = (0..3)
            .map(|_| {
                let raw: Vec<f64> = (0..4).map(|_| u()).collect();
                let derived = vec![raw[0], raw[2], raw[0] * raw[3]];
                let blocks = 6;
                let positions = vec![0, 2, 3];
                let integrands = (0..positions.len() * blocks).map(|_| u() as f32).collect();
                let t: Vec<f64> = (0..blocks).map(|_| u()).collect();
                let mean = t.iter().sum::<f64>() / blocks as f64;
                let sd =
                    (t.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / blocks as f64).sqrt();
                MapView {
                    derived,
                    raw,
                    positions,
                    blocks,
                    integrands,
                    target: t.iter().map(|v| (v - mean) / sd).collect(),
                }
            })
            .collect();
        let pool = MapPool {
            views,
            columns,
            read_width: 4,
            weight: 1.0,
            batch: 3,
            sign: -1.0,
        };
        (
            pool,
            net_columns,
            vec![0.1, -0.2, 0.05],
            vec![1.3, 0.7, 0.9],
        )
    }

    #[test]
    fn map_term_gradient_matches_finite_difference() {
        let (pool, cols, mean, std) = fixture();
        let (nf, nh, alpha) = (6, 5, 0.1);
        let mut r = SplitMix64::new(11);
        let mut u = || r.next_f64_unit() * 2.0 - 1.0;
        let w1: Vec<f64> = (0..nf * nh).map(|_| u()).collect();
        let b1: Vec<f64> = (0..nh).map(|_| 0.3 * u()).collect();
        let w2: Vec<f64> = (0..nh).map(|_| u()).collect();
        let b2 = vec![0.2];
        let total = |w1: &[f64], w2: &[f64]| -> f64 {
            pool.views
                .iter()
                .map(|v| {
                    view_loss(
                        &pool, v, &cols, &mean, &std, w1, &b1, w2, &b2, nf, nh, alpha, None,
                    )
                })
                .sum()
        };
        let (mut gw1, mut gw2) = (vec![0.0; nf * nh], vec![0.0; nh]);
        let base = pool
            .views
            .iter()
            .map(|v| {
                view_loss(
                    &pool,
                    v,
                    &cols,
                    &mean,
                    &std,
                    &w1,
                    &b1,
                    &w2,
                    &b2,
                    nf,
                    nh,
                    alpha,
                    Some((1.0, &mut gw1, &mut gw2)),
                )
            })
            .sum::<f64>();
        assert!((base - total(&w1, &w2)).abs() < 1e-12);
        assert!(base > 0.0);
        let h = 1e-6;
        let mut checked = 0;
        for i in 0..w1.len() {
            let (mut p, mut m) = (w1.clone(), w1.clone());
            p[i] += h;
            m[i] -= h;
            let fd = (total(&p, &w2) - total(&m, &w2)) / (2.0 * h);
            assert!(
                (fd - gw1[i]).abs() < 1e-6 * (1.0 + fd.abs()),
                "w1[{i}]: {fd} vs {}",
                gw1[i]
            );
            checked += usize::from(gw1[i] != 0.0);
        }
        for j in 0..nh {
            let (mut p, mut m) = (w2.clone(), w2.clone());
            p[j] += h;
            m[j] -= h;
            let fd = (total(&w1, &p) - total(&w1, &m)) / (2.0 * h);
            assert!(
                (fd - gw2[j]).abs() < 1e-6 * (1.0 + fd.abs()),
                "w2[{j}]: {fd} vs {}",
                gw2[j]
            );
        }
        // Only the kept columns' rows of W1 carry gradient.
        assert_eq!(checked, cols.len() * nh);
    }
}
