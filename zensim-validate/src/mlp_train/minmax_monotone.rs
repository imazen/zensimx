//! Min-max monotone network (Sill 1998) — an EXPRESSIVE monotone-by-construction
//! head, the architecture chosen (2026-07-16) to break the consistent-dial ⟂
//! real-content tension: the positive-weight LeakyReLU MLP (`monotone_cbc`)
//! craters imazen-26 to ~0.03 when constrained, but a min-max of linear pieces is
//! a *universal approximator for monotone functions* (Sill), so it can (in
//! principle) fit the real-codec ssim2 surface while staying monotone in codec
//! quality — the thing a dial needs to be target-searchable.
//!
//! # The function
//!
//! `score(x) = min_g  max_h  (w[g][h] · x + b[g][h])`
//!
//! over `K` groups (the outer min) of `J` linear pieces each (the inner max),
//! with each weight vector `w[g][h]` SIGN-CONSTRAINED per the feature-sign mask:
//! `sign[f] · w[g][h][f] ≥ 0`. Because every linear piece is then monotone
//! non-decreasing in codec quality (a quality-increasing feature has `w ≥ 0`, a
//! quality-decreasing feature `w ≤ 0`), and max/min of monotone-non-decreasing
//! functions are monotone-non-decreasing, `score` is monotone in quality BY
//! CONSTRUCTION — regardless of the weight values. That is what
//! [`MinMaxMonotone::monotone_by_construction`] proves.
//!
//! Features with `sign[f] == 0` (the 72 "free" / sign-ambiguous features) are
//! dropped (`w == 0`) so monotonicity is exact; the min-max's inner/outer
//! structure is where the expressiveness that `monotone_strict` lost is meant to
//! come back.

/// A trained (or in-training) min-max monotone head.
#[derive(Clone, Debug)]
pub struct MinMaxMonotone {
    /// Outer min groups.
    pub k: usize,
    /// Inner max pieces per group.
    pub j: usize,
    pub n_features: usize,
    /// Row-major `[g][h][f]`, length `k*j*n_features`.
    pub w: Vec<f64>,
    /// Row-major `[g][h]`, length `k*j`.
    pub b: Vec<f64>,
    /// Per-feature monotone direction: `+1` increases with quality (`w ≥ 0`),
    /// `-1` decreases (`w ≤ 0`), `0` dropped (`w ≡ 0`). Length `n_features`.
    pub sign: Vec<f64>,
}

impl MinMaxMonotone {
    /// Index into `w` for group `g`, piece `h`, feature `f`.
    #[inline]
    fn wi(&self, g: usize, h: usize, f: usize) -> usize {
        (g * self.j + h) * self.n_features + f
    }

    /// One linear piece: `w[g][h] · x + b[g][h]`.
    #[inline]
    fn piece(&self, g: usize, h: usize, x: &[f64]) -> f64 {
        let base = (g * self.j + h) * self.n_features;
        let mut acc = self.b[g * self.j + h];
        for f in 0..self.n_features {
            acc += self.w[base + f] * x[f];
        }
        acc
    }

    /// Forward pass. Returns `(score, active_g, active_h)` — the argmin group and
    /// the argmax piece within it, which is where the sub-gradient flows.
    pub fn forward(&self, x: &[f64]) -> (f64, usize, usize) {
        debug_assert_eq!(x.len(), self.n_features);
        let mut best_min = f64::INFINITY;
        let (mut arg_g, mut arg_h) = (0usize, 0usize);
        for g in 0..self.k {
            // inner max over J pieces
            let mut best_max = f64::NEG_INFINITY;
            let mut hstar = 0usize;
            for h in 0..self.j {
                let v = self.piece(g, h, x);
                if v > best_max {
                    best_max = v;
                    hstar = h;
                }
            }
            // outer min over K groups
            if best_max < best_min {
                best_min = best_max;
                arg_g = g;
                arg_h = hstar;
            }
        }
        (best_min, arg_g, arg_h)
    }

    /// Project weights onto the sign constraint `sign[f] · w[f] ≥ 0` (and `w == 0`
    /// where `sign[f] == 0`). Applied after every optimizer step and at bake so
    /// the shipped bytes are monotone-by-construction, exactly like
    /// `monotone_cbc`'s per-step projection.
    pub fn project(&mut self) {
        for g in 0..self.k {
            for h in 0..self.j {
                for f in 0..self.n_features {
                    let i = self.wi(g, h, f);
                    let s = self.sign[f];
                    if s > 0.0 {
                        if self.w[i] < 0.0 {
                            self.w[i] = 0.0;
                        }
                    } else if s < 0.0 {
                        if self.w[i] > 0.0 {
                            self.w[i] = 0.0;
                        }
                    } else {
                        self.w[i] = 0.0;
                    }
                }
            }
        }
    }
}

/// Minibatch RankNet training for a min-max monotone head. `groups[i]` is
/// `(features_row_major, scores)` for one corpus: `features` is `n_rows *
/// n_features` and `scores[r]` is that row's quality label (higher = better).
/// Pairs are drawn WITHIN a group (a codec/quality ladder), the natural unit for
/// a dial. The sub-gradient flows only to the active (argmin-group, argmax-piece)
/// linear of each side; weights are sign-projected after every step so the
/// result is monotone-by-construction throughout.
///
/// Returns the trained head. The caller fits the output→[0,100] dial spline.
#[allow(clippy::too_many_arguments)]
pub fn train_ranknet(
    groups: &[(Vec<f64>, Vec<f64>)],
    sign: &[f64],
    k: usize,
    j: usize,
    n_features: usize,
    epochs: usize,
    pairs_per_epoch: usize,
    lr: f64,
    seed: u64,
) -> MinMaxMonotone {
    let mut state = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0xDEAD_BEEF_CAFE_BABE)
        | 1;
    let mut u = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    // He-ish init scaled small so the initial min-max is well-conditioned.
    let scale = (1.0 / n_features as f64).sqrt() * 0.5;
    let w: Vec<f64> = (0..k * j * n_features)
        .map(|_| (u() * 2.0 - 1.0) * scale)
        .collect();
    let b: Vec<f64> = (0..k * j).map(|_| (u() * 2.0 - 1.0) * scale).collect();
    let mut m = MinMaxMonotone {
        k,
        j,
        n_features,
        w,
        b,
        sign: sign.to_vec(),
    };
    m.project();

    // Adam state over the FULL parameter vector (w then b). Sparse updates —
    // only active pieces move each step — but Adam moments are kept dense for
    // simplicity; unused slots just decay.
    let np = k * j * n_features + k * j;
    let (mut mt, mut vt) = (vec![0.0f64; np], vec![0.0f64; np]);
    let (b1, b2, eps) = (0.9f64, 0.999f64, 1e-8f64);
    let mut t = 0i32;

    // group sampling weighted by row count (bigger ladders contribute more pairs)
    let group_rows: Vec<usize> = groups.iter().map(|(f, _)| f.len() / n_features).collect();
    let total: usize = group_rows.iter().sum();

    for _ in 0..epochs {
        for _ in 0..pairs_per_epoch {
            // pick a group ∝ rows, then two distinct rows in it
            let mut pick = (u() * total as f64) as usize;
            let mut gi = 0;
            for (i, &r) in group_rows.iter().enumerate() {
                if pick < r {
                    gi = i;
                    break;
                }
                pick -= r;
            }
            let nr = group_rows[gi];
            if nr < 2 {
                continue;
            }
            let (feats, scores) = &groups[gi];
            let ra = (u() * nr as f64) as usize;
            let mut rb = (u() * nr as f64) as usize;
            if rb == ra {
                rb = (rb + 1) % nr;
            }
            let xa = &feats[ra * n_features..(ra + 1) * n_features];
            let xb = &feats[rb * n_features..(rb + 1) * n_features];
            // target: +1 if a is higher quality than b
            let tgt = if scores[ra] > scores[rb] { 1.0 } else { -1.0 };
            let (ya, ga, ha) = m.forward(xa);
            let (yb, gb, hb) = m.forward(xb);
            // RankNet: want tgt*(ya-yb) large. loss = softplus(-tgt*(ya-yb)).
            // d loss / d(ya-yb) = -tgt * sigmoid(-tgt*(ya-yb)).
            let d = ya - yb;
            let sig = 1.0 / (1.0 + (tgt * d).exp());
            let dl_dd = -tgt * sig;
            t += 1;
            let bc1 = 1.0 - b1.powi(t);
            let bc2 = 1.0 - b2.powi(t);
            // apply to the two active pieces: +dl_dd to a's piece, -dl_dd to b's.
            let mut step = |g: usize, h: usize, x: &[f64], grad_y: f64| {
                let base = (g * j + h) * n_features;
                for f in 0..n_features {
                    let gi = base + f;
                    let grad = grad_y * x[f];
                    mt[gi] = b1 * mt[gi] + (1.0 - b1) * grad;
                    vt[gi] = b2 * vt[gi] + (1.0 - b2) * grad * grad;
                    m.w[gi] -= lr * (mt[gi] / bc1) / ((vt[gi] / bc2).sqrt() + eps);
                }
                let bi = k * j * n_features + g * j + h;
                mt[bi] = b1 * mt[bi] + (1.0 - b1) * grad_y;
                vt[bi] = b2 * vt[bi] + (1.0 - b2) * grad_y * grad_y;
                m.b[g * j + h] -= lr * (mt[bi] / bc1) / ((vt[bi] / bc2).sqrt() + eps);
            };
            step(ga, ha, xa, dl_dd);
            step(gb, hb, xb, -dl_dd);
            m.project();
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    // Canonical midrank Spearman (imazen/zensimx#41): the test used to carry
    // its own ordinal-rank copy.
    use crate::panel::spearman;

    /// Deterministic pseudo-random weights in [-scale, scale] (no rng dep).
    fn seeded(k: usize, j: usize, nf: usize, sign: Vec<f64>) -> MinMaxMonotone {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut nxt = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let w: Vec<f64> = (0..k * j * nf).map(|_| nxt()).collect();
        let b: Vec<f64> = (0..k * j).map(|_| nxt()).collect();
        let mut m = MinMaxMonotone {
            k,
            j,
            n_features: nf,
            w,
            b,
            sign,
        };
        m.project();
        m
    }

    /// The load-bearing property: after `project`, increasing any quality-
    /// increasing feature (or decreasing any quality-decreasing one) NEVER
    /// decreases the score — i.e. the score is monotone in codec quality by
    /// construction, for ARBITRARY weights. This is the whole reason to use
    /// min-max over the MLP.
    #[test]
    fn monotone_by_construction() {
        let nf = 12;
        // 8 quality-increasing (+1), 2 decreasing (-1), 2 dropped (0).
        let sign: Vec<f64> = (0..nf)
            .map(|f| match f % 6 {
                5 => 0.0,
                4 => -1.0,
                _ => 1.0,
            })
            .collect();
        let m = seeded(4, 3, nf, sign.clone());

        let mut state = 0xDEAD_BEEF_CAFE_BABEu64;
        let mut nxt = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        // 200 random base points × every feature × a positive quality step.
        for _ in 0..200 {
            let x: Vec<f64> = (0..nf).map(|_| nxt() * 4.0 - 2.0).collect();
            let (base, _, _) = m.forward(&x);
            for f in 0..nf {
                if sign[f] == 0.0 {
                    continue;
                }
                // move feature f in the QUALITY-increasing direction
                let mut xq = x.clone();
                xq[f] += sign[f] * (0.1 + nxt()); // +step for sign+1, −step for sign−1
                let (up, _, _) = m.forward(&xq);
                assert!(
                    up >= base - 1e-9,
                    "monotonicity violated: feat {f} sign {} moved quality-up but score {base} -> {up}",
                    sign[f]
                );
            }
        }
    }

    /// Training learns to RANK a non-linear monotone target — the point of
    /// min-max over a single linear. Target = `max(2·x0 + x1, x2 − x3)`, a
    /// monotone piecewise-linear (increasing in x0/x1/x2, decreasing in x3).
    /// A single linear can't fit a max-of-two; the min-max (K≥1, J≥2) can.
    #[test]
    fn training_learns_to_rank_monotone_target() {
        let nf = 4;
        let sign = vec![1.0, 1.0, 1.0, -1.0];
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let mut u = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let target = |x: &[f64]| (2.0 * x[0] + x[1]).max(x[2] - x[3]);
        // one "group" (ladder) of 400 rows
        let n = 400;
        let mut feats = Vec::with_capacity(n * nf);
        let mut scores = Vec::with_capacity(n);
        for _ in 0..n {
            let x: Vec<f64> = (0..nf).map(|_| u() * 2.0 - 1.0).collect();
            scores.push(target(&x));
            feats.extend_from_slice(&x);
        }
        let m = train_ranknet(
            &[(feats.clone(), scores.clone())],
            &sign,
            2, // K groups
            2, // J pieces (needs ≥2 to fit the max)
            nf,
            60,
            2000,
            5e-3,
            7,
        );
        // Spearman of predictions vs target on the training ladder.
        let pred: Vec<f64> = (0..n)
            .map(|r| m.forward(&feats[r * nf..(r + 1) * nf]).0)
            .collect();
        let srocc = spearman(&pred, &scores);
        assert!(
            srocc > 0.90,
            "min-max should rank a monotone piecewise-linear target well; got SROCC {srocc:.3}"
        );
    }

    /// `project` zeroes dropped features and clamps signs.
    #[test]
    fn project_enforces_signs() {
        let sign = vec![1.0, -1.0, 0.0];
        let m = seeded(2, 2, 3, sign);
        for g in 0..m.k {
            for h in 0..m.j {
                assert!(m.w[m.wi(g, h, 0)] >= 0.0, "sign+1 feature must be >=0");
                assert!(m.w[m.wi(g, h, 1)] <= 0.0, "sign-1 feature must be <=0");
                assert_eq!(m.w[m.wi(g, h, 2)], 0.0, "sign0 feature must be dropped");
            }
        }
    }
}
