//! Exact max-pair projection kernels. No optimizer or sampling state.

/// Smallest direct weight that may carry a negative product weight: the f16
/// normal range, so packing cannot flush `a` to zero while `b < 0` survives.
pub(super) const GATE_MIN_DIRECT: f64 = 1.0 / 8192.0;
/// Relative margin kept inside the gate cone so f16 packing (relative error
/// <= 2^-11 per weight) cannot move a boundary pair outside it.
pub(super) const GATE_MARGIN: f64 = 1.0 / 256.0;

/// Projects one unit's `(a, b)` (direct, product weights) onto the gate cone
/// `{a >= 0, s_p·a + s_d·b >= GATE_MARGIN·s_p·a}`: the closer of the two edge
/// rays when outside, then the f16 safety rules above.
pub(super) fn project_gate(a: f64, b: f64, sd: f64, sp: f64) -> (f64, f64) {
    let (mut a, mut b) = if a >= 0.0 && sp * a + sd * b >= 0.0 {
        (a, b)
    } else {
        let n = (sd * sd + sp * sp).sqrt();
        let (ux, uy) = (sd / n, -sp / n);
        let t1 = b.max(0.0);
        let t2 = (a * ux + b * uy).max(0.0);
        let d1 = a * a + (b - t1) * (b - t1);
        let d2 = (a - t2 * ux).powi(2) + (b - t2 * uy).powi(2);
        if d1 <= d2 {
            (0.0, t1)
        } else {
            (t2 * ux, t2 * uy)
        }
    };
    if b < 0.0 {
        if a < GATE_MIN_DIRECT {
            b = 0.0;
        } else {
            b = b.max(-a * (sp / sd) * (1.0 - GATE_MARGIN));
        }
    }
    a = a.max(0.0);
    (a, b)
}

/// A disjoint pair layout, prepared once before the training loop.
/// Capturing every v into scratch is unnecessary: a disjoint pair can be read
/// and projected before either of its entries is overwritten.
pub(super) struct MaxPairRows {
    hidden: usize,
    pairs: Vec<[usize; 2]>,
    contiguous: bool,
}

impl MaxPairRows {
    pub(super) fn new(hidden: usize, pairs: Vec<[usize; 2]>) -> Self {
        assert!(hidden > 0);
        let mut used = vec![false; hidden];
        for &[u, v] in &pairs {
            assert!(u < hidden && v < hidden && u != v);
            assert!(
                !used[u] && !used[v],
                "max-pair projection needs disjoint units"
            );
            used[u] = true;
            used[v] = true;
        }
        Self {
            hidden,
            contiguous: pairs
                .iter()
                .enumerate()
                .all(|(i, p)| *p == [2 * i, 2 * i + 1]),
            pairs,
        }
    }

    pub(super) fn project_rows(&self, weights: &mut [f64]) {
        assert!(weights.len().is_multiple_of(self.hidden));
        let mut cap = if self.contiguous {
            Vec::new()
        } else {
            vec![0.0; self.pairs.len()]
        };
        self.project_rows_scratch(weights, &mut cap);
    }

    fn project_rows_scratch(&self, weights: &mut [f64], cap: &mut [f64]) {
        for row in weights.chunks_exact_mut(self.hidden) {
            if self.contiguous {
                let (paired, tail) = row.split_at_mut(2 * self.pairs.len());
                for pair in paired.as_chunks_mut::<2>().0 {
                    let (a, b) = project_gate(pair[0].max(0.0), pair[1], 1.0, 1.0);
                    pair[0] = a;
                    pair[1] = b;
                }
                for x in tail {
                    *x = x.max(0.0);
                }
            } else {
                // An indexed clamp over only the unpaired units regressed in
                // the measured permuted layout. Keep the original dense clamp.
                for (c, &[_, v]) in cap.iter_mut().zip(&self.pairs) {
                    *c = row[v];
                }
                for x in row.iter_mut() {
                    *x = x.max(0.0);
                }
                for (&b, &[u, v]) in cap.iter().zip(&self.pairs) {
                    let (a, b) = project_gate(row[u], b, 1.0, 1.0);
                    row[u] = a;
                    row[v] = b;
                }
            }
        }
    }

    /// Only for the post-fused-Adam path: omitted rows are the exact +0
    /// fixed points already omitted by Adam itself. Initialization, auxiliary
    /// updates, and other optimizer paths continue to use the full row walk.
    pub(super) fn project_active_rows(&self, weights: &mut [f64], ranges: Option<&[(u32, u32)]>) {
        match ranges {
            None => self.project_rows(weights),
            Some(ranges) => {
                let mut end = 0;
                let mut cap = if self.contiguous {
                    Vec::new()
                } else {
                    vec![0.0; self.pairs.len()]
                };
                for &(a, b) in ranges {
                    let (a, b) = (a as usize, b as usize);
                    assert!(a >= end && a <= b && b <= weights.len() / self.hidden);
                    self.project_rows_scratch(
                        &mut weights[a * self.hidden..b * self.hidden],
                        &mut cap,
                    );
                    end = b;
                }
            }
        }
    }

    pub(super) fn pairs(&self) -> &[[usize; 2]] {
        &self.pairs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn original_rows(w: &mut [f64], hidden: usize, positions: &[[usize; 2]]) {
        // Original capture -> clamp -> indexed projection order; retain this
        // oracle when changing the production row walk.
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

    fn same(original: &[f64], actual: &[f64]) {
        assert_eq!(original.len(), actual.len());
        for (i, (&a, &b)) in original.iter().zip(actual).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "weight {i}: {a:?} vs {b:?}");
        }
    }

    #[test]
    fn row_walk_matches_original_at_gate_boundaries() {
        let values = [
            0.0,
            -0.0,
            f64::from_bits(1),
            -f64::from_bits(1),
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            GATE_MIN_DIRECT.next_down(),
            GATE_MIN_DIRECT,
            GATE_MIN_DIRECT.next_up(),
            -GATE_MIN_DIRECT,
            1.0 - GATE_MARGIN,
            -(1.0 - GATE_MARGIN),
            1.0,
            -1.0,
            f64::MAX,
            -f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(0x7ff8000000000023),
        ];
        for positions in [vec![[0, 1]], vec![[2, 0]], vec![[3, 1], [0, 2]]] {
            let plan = MaxPairRows::new(4, positions.clone());
            assert_eq!(plan.pairs(), positions);
            for &a in &values {
                for &b in &values {
                    let mut w = vec![a, b, b, a];
                    let mut expected = w.clone();
                    original_rows(&mut expected, 4, &positions);
                    plan.project_rows(&mut w);
                    same(&expected, &w);
                }
            }
        }
    }

    #[test]
    fn row_walk_matches_original_for_training_shapes_and_repeated_updates() {
        let mut rng = 0x51515eed_u64;
        for hidden in [4, 64, 128] {
            let n = hidden / 4;
            for indexed in [false, true] {
                let positions: Vec<_> = (0..n)
                    .map(|i| {
                        if indexed {
                            [hidden - 1 - 2 * i, hidden - 2 - 2 * i]
                        } else {
                            [2 * i, 2 * i + 1]
                        }
                    })
                    .collect();
                let plan = MaxPairRows::new(hidden, positions.clone());
                let mut w = vec![0.0; 1853 * hidden];
                for step in 0..6 {
                    for x in &mut w {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        *x += ((rng >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
                            * if step == 0 { 2.0 } else { 0.0001 };
                    }
                    let mut expected = w.clone();
                    original_rows(&mut expected, hidden, &positions);
                    plan.project_rows(&mut w);
                    same(&expected, &w);
                }
            }
        }
    }

    #[test]
    fn active_zero_fixed_point_rows_match_full_projection() {
        let ranges = [(0, 3), (7, 9), (12, 13)];
        for positions in [vec![[0, 1], [2, 3]], vec![[7, 5], [0, 6]]] {
            let plan = MaxPairRows::new(8, positions);
            let mut full = vec![0.0; 17 * 8];
            for &(a, b) in &ranges {
                for (i, x) in full[a as usize * 8..b as usize * 8].iter_mut().enumerate() {
                    *x = (i as f64 - 8.0) * 0.01;
                }
            }
            let mut active = full.clone();
            plan.project_rows(&mut full);
            plan.project_active_rows(&mut active, Some(&ranges));
            same(&full, &active);
            let mut all = active.clone();
            plan.project_active_rows(&mut all, None);
            plan.project_rows(&mut active);
            same(&all, &active);
        }
    }

    #[test]
    #[should_panic(expected = "disjoint")]
    fn refuses_overlapping_pair_layout() {
        MaxPairRows::new(4, vec![[0, 1], [1, 2]]);
    }
}
