//! Probe: axioms-only "residual-from-identity" metric.
//!
//!   score(x) = 100 − λ · Σ_j w_j · (h_j(x) − h_j(x_id))²
//!
//! where `h = LeakyReLU(W1·x_std + b1)` is an UNCONSTRAINED encoder,
//! `x_id` is the standardized identity feature vector (raw identity
//! features are all 0 → standardized = −mean/std), and `w_j, λ ≥ 0`
//! (via softplus). This guarantees by construction:
//!   - A1 bounded ≤ 100 (subtracting a non-negative term),
//!   - A2 self-identity is the unique max (the squared distance is 0
//!     exactly at x_id, > 0 elsewhere when w > 0),
//!
//! WITHOUT forcing A3 (degradation monotonicity) — the encoder is free,
//! so g = ‖h(x) − h_id‖²_w can be non-monotone in the input features.
//! That is the falsification target: does an expressive axioms-only
//! metric (a) keep the human-MOS panel and (b) still invert on the OOD
//! blur ladder?
//!
//! `h_id` is recomputed each step from the current encoder but its
//! gradient is stopped (g(x_id) = 0 holds structurally regardless, and
//! the MSE on non-identity targets prevents the collapse-to-100
//! degenerate). Self-contained — no production-runtime surgery.
//!
//! Run: cargo run --release -p zensim-validate --bin residual_identity_probe

use std::path::PathBuf;
use zensim_validate::parquet_loader::load_parquet;
// Spearman is zenstats' — the single owner of stat math (imazen/zensimx#41).
use zenstats::panel::spearman;

const NF: usize = 372;
const NH: usize = 64;
const CANON: &str = "/mnt/v/zen/zensim-training/canonical-2026-05-21/train";
const VAL: &str = zensim_validate::eval_roots::STORED_FEATURES_ROOT_2026_05_15;

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self {
        Self(s | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[inline]
fn softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { (x.exp() + 1.0).ln() }
}
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

struct Model {
    w1: Vec<f64>,      // [NF*NH] row-major [feature][hidden]
    b1: Vec<f64>,      // [NH]
    theta_w: Vec<f64>, // [NH] → w_j = softplus
    theta_lam: f64,    // λ = softplus
    leaky: f64,
}

impl Model {
    fn new(rng: &mut Rng) -> Self {
        let scale = (2.0 / NF as f64).sqrt();
        Self {
            w1: (0..NF * NH).map(|_| rng.gauss() * scale).collect(),
            b1: vec![0.0; NH],
            theta_w: vec![0.0; NH], // softplus(0)=0.693 ≈ 0.7 initial weight
            theta_lam: 0.0,         // λ ≈ 0.7 initial
            leaky: 0.01,
        }
    }
    /// h = LeakyReLU(W1·x + b1). Returns (h_pre, h).
    fn encode(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut hp = self.b1.clone();
        for (i, &xi) in x[..NF].iter().enumerate() {
            if xi == 0.0 {
                continue;
            }
            let row = &self.w1[i * NH..(i + 1) * NH];
            for j in 0..NH {
                hp[j] += xi * row[j];
            }
        }
        let h: Vec<f64> = hp
            .iter()
            .map(|&v| if v >= 0.0 { v } else { self.leaky * v })
            .collect();
        (hp, h)
    }
}

/// Standardize raw features in place against (mean, std).
fn standardize(rows: &mut [Vec<f64>], mean: &[f64], std: &[f64]) {
    for r in rows.iter_mut() {
        for d in 0..NF {
            r[d] = (r[d] - mean[d]) / std[d].max(1e-8);
        }
    }
}

fn load(path: &str, name: &str) -> Option<(Vec<Vec<f64>>, Vec<f64>)> {
    match load_parquet(&PathBuf::from(path), name, "human_score", 1.0) {
        Ok(g) => Some((g.feature_rows, g.human_scores)),
        Err(e) => {
            eprintln!("  load {name} skipped: {e}");
            None
        }
    }
}

fn main() {
    let mut rng = Rng::new(0x5151_2727_3939_4242);

    // --- Load training groups (skip the 196k safesyn for probe speed) ---
    eprintln!("loading train groups…");
    let train_specs = [
        (
            format!("{CANON}/cid22_train_norm.parquet"),
            "cid22_train",
            1.5,
        ),
        (format!("{CANON}/kadid.parquet"), "kadid", 1.0),
        (format!("{CANON}/tid.parquet"), "tid", 1.0),
        (format!("{CANON}/konjnd-dense-norm.parquet"), "konjnd", 1.2),
    ];
    let mut groups: Vec<(Vec<Vec<f64>>, Vec<f64>, f64)> = Vec::new();
    for (p, n, w) in &train_specs {
        if let Some((rows, hs)) = load(p, n) {
            eprintln!("  {n}: {} rows", rows.len());
            groups.push((rows, hs, *w));
        }
    }

    // --- Scaler from all train rows ---
    let mut mean = vec![0.0; NF];
    let mut cnt = 0u64;
    for (rows, _, _) in &groups {
        for r in rows {
            for d in 0..NF {
                mean[d] += r[d];
            }
            cnt += 1;
        }
    }
    let n = cnt.max(1) as f64;
    for m in &mut mean {
        *m /= n;
    }
    let mut var = vec![0.0; NF];
    for (rows, _, _) in &groups {
        for r in rows {
            for d in 0..NF {
                let dx = r[d] - mean[d];
                var[d] += dx * dx;
            }
        }
    }
    let std: Vec<f64> = var.iter().map(|&v| (v / n).sqrt().max(1e-8)).collect();
    for (rows, _, _) in &mut groups {
        standardize(rows, &mean, &std);
    }
    // standardized identity (raw features 0 → −mean/std)
    let x_id: Vec<f64> = (0..NF).map(|d| -mean[d] / std[d].max(1e-8)).collect();

    // --- Train ---
    let mut m = Model::new(&mut rng);
    // Adam state
    let np_w1 = NF * NH;
    let mut adam = |_: ()| {};
    let _ = &mut adam;
    let mut mw1 = vec![0.0; np_w1];
    let mut vw1 = vec![0.0; np_w1];
    let mut mb1 = vec![0.0; NH];
    let mut vb1 = vec![0.0; NH];
    let mut mtw = vec![0.0; NH];
    let mut vtw = vec![0.0; NH];
    let (mut mtl, mut vtl) = (0.0, 0.0);
    let (b1c, b2c, eps): (f64, f64, f64) = (0.9, 0.999, 1e-8);
    let mut t = 0i32;

    let n_epochs = 120;
    let pairs_per_epoch = 30000;
    let total_w: f64 = groups.iter().map(|g| g.2).sum();
    let cdf: Vec<f64> = {
        let mut c = 0.0;
        groups
            .iter()
            .map(|g| {
                c += g.2;
                c / total_w
            })
            .collect()
    };

    // RBF (Gaussian-kernel) form: score = 100·exp(−λ·g), g = Σ w_j(h-h_id)².
    // Identity (g=0) → 100 (unique max, A2). Bounded (0,100] (A1).
    // Well-conditioned (no quadratic blow-up). Returns (score, lam, g).
    let score = |m: &Model, h: &[f64], h_id: &[f64]| -> (f64, f64, f64) {
        let lam = softplus(m.theta_lam);
        let mut g = 0.0;
        for j in 0..NH {
            let wj = softplus(m.theta_w[j]);
            let d = h[j] - h_id[j];
            g += wj * d * d;
        }
        let s = 100.0 * (-lam * g).exp();
        (s, lam, g)
    };

    for epoch in 0..n_epochs {
        let lr = 1e-3 * 0.5 * (1.0 + (std::f64::consts::PI * (epoch % 40) as f64 / 40.0).cos());
        // h_id (stop-grad) for this epoch's current encoder
        let (_, h_id) = m.encode(&x_id);
        let mut gw1 = vec![0.0; np_w1];
        let mut gb1 = vec![0.0; NH];
        let mut gtw = vec![0.0; NH];
        let mut gtl = 0.0;
        let mut total_loss = 0.0;
        let mut steps = 0u64;
        for _ in 0..pairs_per_epoch {
            let u = rng.unit();
            let gi = cdf.partition_point(|&c| c < u).min(groups.len() - 1);
            let (rows, hs, _) = &groups[gi];
            let nrow = rows.len();
            if nrow < 2 {
                continue;
            }
            let ia = (rng.next_u64() as usize) % nrow;
            let ib = (rng.next_u64() as usize) % nrow;
            if ia == ib {
                continue;
            }
            let ta = hs[ia] * 100.0;
            let tb = hs[ib] * 100.0;
            let target = (ta - tb).signum();
            if target == 0.0 {
                continue;
            }
            let (hpa, ha) = m.encode(&rows[ia]);
            let (hpb, hb) = m.encode(&rows[ib]);
            let (sa, lam, _ga) = score(&m, &ha, &h_id);
            let (sb, _, _gb) = score(&m, &hb, &h_id);

            // RankNet (target sign) + MSE(score, target).
            let z = -target * (sb - sa);
            let lrk = if z > 40.0 {
                z
            } else if z < -40.0 {
                0.0
            } else {
                (z.exp() + 1.0).ln()
            };
            let sig = sigmoid(-z);
            // dL_rank/d(sb-sa) = -target*sig ; dsa = +target*sig, dsb = -target*sig
            let rn_w = 0.6;
            let mut dsa = rn_w * target * sig;
            let mut dsb = -rn_w * target * sig;
            // MSE
            let mse_w = 0.6;
            let nrm = (2.0 * pairs_per_epoch as f64).max(1.0);
            dsa += mse_w * 2.0 * (sa - ta) / nrm;
            dsb += mse_w * 2.0 * (sb - tb) / nrm;
            total_loss += lrk * rn_w + mse_w * ((sa - ta).powi(2) + (sb - tb).powi(2)) / nrm;
            steps += 1;

            // Backprop each side: score = 100 - lam*Σ wj (h-hid)²
            // dscore/dh_j = -lam * wj * 2 (h_j - h_id_j)
            // dscore/dtheta_w_j = -lam * (h-hid)² * softplus'(theta_w_j)
            // dscore/dtheta_lam = -g * softplus'(theta_lam)
            // RBF: score = 100·exp(−λg). dscore/dg = −λ·score.
            //   dscore/dh_j     = −λ·score·2·w_j·(h_j−h_id_j)
            //   dscore/dθ_w_j   = −λ·score·(h_j−h_id_j)²·σ(θ_w_j)
            //   dscore/dθ_lam   = −g·score·σ(θ_lam)
            let backprop_side = |ds: f64,
                                 s: f64,
                                 h: &[f64],
                                 hp: &[f64],
                                 x: &[f64],
                                 gw1: &mut [f64],
                                 gb1: &mut [f64],
                                 gtw: &mut [f64],
                                 gtl: &mut f64| {
                let mut g_side = 0.0;
                for j in 0..NH {
                    let wj = softplus(m.theta_w[j]);
                    let d = h[j] - h_id[j];
                    g_side += wj * d * d;
                    let dl_dh = ds * (-lam * s * 2.0 * wj * d);
                    let dl_dhp = if hp[j] >= 0.0 { dl_dh } else { dl_dh * m.leaky };
                    gb1[j] += dl_dhp;
                    let sp_w = sigmoid(m.theta_w[j]);
                    gtw[j] += ds * (-lam * s * d * d) * sp_w;
                    for i in 0..NF {
                        let xi = x[i];
                        if xi != 0.0 {
                            gw1[i * NH + j] += dl_dhp * xi;
                        }
                    }
                }
                let sp_l = sigmoid(m.theta_lam);
                *gtl += ds * (-g_side * s) * sp_l;
            };
            backprop_side(
                dsa, sa, &ha, &hpa, &rows[ia], &mut gw1, &mut gb1, &mut gtw, &mut gtl,
            );
            backprop_side(
                dsb, sb, &hb, &hpb, &rows[ib], &mut gw1, &mut gb1, &mut gtw, &mut gtl,
            );
        }

        // Adam update
        t += 1;
        let bc1 = 1.0 - b1c.powi(t);
        let bc2 = 1.0 - b2c.powi(t);
        let upd = |w: &mut [f64], g: &[f64], mm: &mut [f64], vv: &mut [f64]| {
            for i in 0..w.len() {
                mm[i] = b1c * mm[i] + (1.0 - b1c) * g[i];
                vv[i] = b2c * vv[i] + (1.0 - b2c) * g[i] * g[i];
                w[i] -= lr * (mm[i] / bc1) / ((vv[i] / bc2).sqrt() + eps);
            }
        };
        upd(&mut m.w1, &gw1, &mut mw1, &mut vw1);
        upd(&mut m.b1, &gb1, &mut mb1, &mut vb1);
        upd(&mut m.theta_w, &gtw, &mut mtw, &mut vtw);
        {
            let mut tl = [m.theta_lam];
            let g = [gtl];
            let mut mm = [mtl];
            let mut vv = [vtl];
            upd(&mut tl, &g, &mut mm, &mut vv);
            m.theta_lam = tl[0];
            mtl = mm[0];
            vtl = vv[0];
        }

        if epoch % 20 == 0 || epoch == n_epochs - 1 {
            eprintln!(
                "epoch {epoch:3} lr={lr:.4} loss={:.4} λ={:.3} steps={steps}",
                total_loss / steps.max(1) as f64,
                softplus(m.theta_lam)
            );
        }
    }

    // --- Eval: panel SROCC on val corpora ---
    eprintln!("\n=== PANEL (SROCC, residual-identity probe) ===");
    let (_, h_id) = m.encode(&x_id);
    let val_specs = [
        ("cid22", "cid22_features_372col_2026-05-15.parquet"),
        ("kadid", "kadid_features_372col_2026-05-15.parquet"),
        ("tid", "tid_features_372col_2026-05-15.parquet"),
        ("konjnd", "konjnd_features_372col_2026-05-15.parquet"),
        ("aic3", "aic3_features_372col_2026-05-15.parquet"),
    ];
    for (name, fname) in &val_specs {
        if let Some((mut rows, hs)) = load(&format!("{VAL}/{fname}"), name) {
            standardize(&mut rows, &mean, &std);
            let preds: Vec<f64> = rows
                .iter()
                .map(|r| {
                    let (_, h) = m.encode(r);
                    score(&m, &h, &h_id).0
                })
                .collect();
            let sr = spearman(&preds, &hs);
            let pmin = preds.iter().cloned().fold(f64::INFINITY, f64::min);
            let pmax = preds.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            eprintln!(
                "  {name:8} SROCC={sr:.4}  score_range=[{pmin:.1}, {pmax:.1}]  n={}",
                hs.len()
            );
        }
    }

    // --- Falsify: blur-ladder monotonicity (the A3 test) ---
    eprintln!("\n=== BLUR-LADDER monotonicity (A3 falsification) ===");
    for c in ["color_blocks", "checker", "mandelbrot", "value_noise"] {
        let path = format!("/tmp/blur_ladder_{c}.featmat");
        let Ok(bytes) = std::fs::read(&path) else {
            eprintln!("  {c}: missing featmat");
            continue;
        };
        let nf = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let nr = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let mut scores = Vec::new();
        for r in 0..nr {
            let mut x = vec![0.0; NF];
            for d in 0..nf.min(NF) {
                let off = 8 + (r * nf + d) * 4;
                let raw = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as f64;
                x[d] = (raw - mean[d]) / std[d].max(1e-8);
            }
            let (_, h) = m.encode(&x);
            scores.push(score(&m, &h, &h_id).0);
        }
        let id = scores[0];
        let mut inv = 0;
        let mut above = 0;
        for w in 1..scores.len() {
            if scores[w] > scores[w - 1] + 0.01 {
                inv += 1;
            }
            if scores[w] > id + 0.01 {
                above += 1;
            }
        }
        let s: Vec<String> = scores.iter().map(|v| format!("{v:.1}")).collect();
        eprintln!(
            "  {c:13} [{}]  inversions={inv} above_identity={above}",
            s.join(" ")
        );
    }
    eprintln!(
        "\n(correct-by-axioms → 0 above_identity by construction; inversions>0 = A3 NOT guaranteed)"
    );
}
