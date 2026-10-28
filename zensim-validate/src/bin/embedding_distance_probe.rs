//! V3 probe: IDENTITY-ANCHORED EMBEDDING-DISTANCE metric.
//!
//!   score(x) = 100 − λ · D(x),   D(x) = ‖φ(x) − φ(id)‖_RMS
//!
//! where φ: R^372 → R^K is a small UNCONSTRAINED MLP (no sign / monotone
//! constraint), and D is the RMS distance of x's embedding from the
//! identity embedding φ(id). λ = softplus(λθ) > 0.
//!
//! Guarantees by construction:
//!   A1 (bounded ≤ 100):  D ≥ 0 ⇒ score ≤ 100.
//!   A2 (defect half):    score(id) = 100 is the GLOBAL MAX — no distortion
//!                        can score above the reference (D(id)=0, D(x)≥0).
//!                        This is exactly the V39 blur>identity defect,
//!                        fixed structurally. (Strict uniqueness needs φ
//!                        injective; training pressure delivers it in
//!                        practice. The defect-relevant half — nothing
//!                        exceeds 100 — is unconditional.)
//!   Resolution:          D unbounded above ⇒ score unbounded below; a
//!                        terrible distortion gets a genuinely low score.
//!   A3 (monotonicity):   DROPPED as an absolute (φ is unconstrained). Per
//!                        the user's reframe A3 is human-authority, not
//!                        internal math; the structural-corruption corpus
//!                        (codec-corpus#7) is the held-out gate for the
//!                        property that matters: broken < honest-LQ.
//!
//! Falsify V2's "architecture ceiling": V2 (partial-monotone) capped CID22
//! at 0.625 because the unbounded monotone term drowned the bounded free
//! head. Dropping the monotone constraint (keeping only A1+A2) should free
//! the panel toward the expressive ~0.88 ceiling. Does it?
//!
//! Self-contained probe; no production-runtime surgery until this passes.

use std::path::PathBuf;
use zensim_validate::parquet_loader::load_parquet;
// Spearman is zenstats' — the single owner of stat math (imazen/zensimx#41).
use zenstats::panel::spearman;

const NF: usize = 372;
const H: usize = 64; // φ hidden width
const K: usize = 32; // embedding dim
const CANON: &str = "/mnt/v/zen/zensim-training/canonical-2026-05-21/train";
const VAL: &str = zensim_validate::eval_roots::STORED_FEATURES_ROOT_2026_05_15;
const L2: f64 = 1e-5; // weight decay on φ

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self {
        Self(s | 1)
    }
    fn n(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn u(&mut self) -> f64 {
        (self.n() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn g(&mut self) -> f64 {
        let u1 = self.u().max(1e-12);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * self.u()).cos()
    }
    fn idx(&mut self, n: usize) -> usize {
        (self.n() as usize) % n
    }
}

#[inline]
fn sp(x: f64) -> f64 {
    if x > 20.0 { x } else { (x.exp() + 1.0).ln() }
}
#[inline]
fn sig(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn load(path: &str, name: &str) -> Option<(Vec<Vec<f64>>, Vec<f64>)> {
    match load_parquet(&PathBuf::from(path), name, "human_score", 1.0) {
        Ok(g) => Some((g.feature_rows, g.human_scores)),
        Err(e) => {
            eprintln!("  {name} skip: {e}");
            None
        }
    }
}

struct Model {
    w1: Vec<f64>, // [NF*H]
    b1: Vec<f64>, // [H]
    w2: Vec<f64>, // [H*K]
    b2: Vec<f64>, // [K]
    lam: f64,     // θ → softplus
    leaky: f64,
}
impl Model {
    fn new(rng: &mut Rng) -> Self {
        let s1 = (2.0 / NF as f64).sqrt();
        let s2 = (2.0 / H as f64).sqrt();
        Self {
            w1: (0..NF * H).map(|_| rng.g() * s1).collect(),
            b1: vec![0.0; H],
            w2: (0..H * K).map(|_| rng.g() * s2).collect(),
            b2: vec![0.0; K],
            lam: 0.0, // softplus(0)=0.69
            leaky: 0.01,
        }
    }
    /// embedding φ(x); returns (e[K], h_pre[H], h[H]) for backprop.
    fn embed(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut hp = self.b1.clone();
        for (di, &xi) in x.iter().enumerate().take(NF) {
            if xi == 0.0 {
                continue;
            }
            let row = &self.w1[di * H..(di + 1) * H];
            for j in 0..H {
                hp[j] += xi * row[j];
            }
        }
        let h: Vec<f64> = hp
            .iter()
            .map(|&v| if v >= 0.0 { v } else { self.leaky * v })
            .collect();
        let mut e = self.b2.clone();
        for (j, &hj) in h.iter().enumerate() {
            if hj == 0.0 {
                continue;
            }
            let row = &self.w2[j * K..(j + 1) * K];
            for d in 0..K {
                e[d] += hj * row[d];
            }
        }
        (e, hp, h)
    }
}

fn main() {
    let mut rng = Rng::new(0x0273_9999_4242_1357u64.wrapping_add(17));

    eprintln!("loading train…");
    let specs = [
        (format!("{CANON}/safesyn.parquet"), "safesyn", 1.0),
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
    for (p, n, w) in &specs {
        if let Some((r, h)) = load(p, n) {
            eprintln!("  {n}: {} rows", r.len());
            groups.push((r, h, *w));
        }
    }

    // scaler
    let mut mean = vec![0.0; NF];
    let mut cnt = 0u64;
    for (r, _, _) in &groups {
        for row in r {
            for d in 0..NF {
                mean[d] += row[d];
            }
            cnt += 1;
        }
    }
    let n = cnt.max(1) as f64;
    for m in &mut mean {
        *m /= n;
    }
    let mut var = vec![0.0; NF];
    for (r, _, _) in &groups {
        for row in r {
            for d in 0..NF {
                let dx = row[d] - mean[d];
                var[d] += dx * dx;
            }
        }
    }
    let std: Vec<f64> = var.iter().map(|&v| (v / n).sqrt().max(1e-8)).collect();
    let standardize = |row: &mut Vec<f64>| {
        for d in 0..NF {
            row[d] = (row[d] - mean[d]) / std[d];
        }
    };
    for (r, _, _) in &mut groups {
        for row in r.iter_mut() {
            standardize(row);
        }
    }
    let x_id: Vec<f64> = (0..NF).map(|d| -mean[d] / std[d]).collect();

    let mut m = Model::new(&mut rng);

    // Adam buffers
    macro_rules! av {
        ($n:expr) => {
            (vec![0.0; $n], vec![0.0; $n])
        };
    }
    let (mut m_w1, mut v_w1) = av!(NF * H);
    let (mut m_b1, mut v_b1) = av!(H);
    let (mut m_w2, mut v_w2) = av!(H * K);
    let (mut m_b2, mut v_b2) = av!(K);
    let (mut m_lm, mut v_lm) = (0.0f64, 0.0f64);
    let (b1c, b2c, eps): (f64, f64, f64) = (0.9, 0.999, 1e-8);
    let mut t = 0i32;

    // score(x) given the (stop-grad) identity embedding e_id; returns
    // (score, D, e, hp, h) so the caller can backprop.
    let score = |m: &Model, x: &[f64], e_id: &[f64]| -> (f64, f64, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (e, hp, h) = m.embed(x);
        let mut ss = 0.0;
        for d in 0..K {
            let de = e[d] - e_id[d];
            ss += de * de;
        }
        let dd = (ss / K as f64).sqrt().max(1e-9); // RMS distance
        let lam = sp(m.lam);
        (100.0 - lam * dd, dd, e, hp, h)
    };

    let epochs = 160;
    let ppe = 60000;
    let totw: f64 = groups.iter().map(|g| g.2).sum();
    let cdf: Vec<f64> = {
        let mut c = 0.0;
        groups
            .iter()
            .map(|g| {
                c += g.2;
                c / totw
            })
            .collect()
    };

    for epoch in 0..epochs {
        let lr = 1e-3 * 0.5 * (1.0 + (std::f64::consts::PI * (epoch % 50) as f64 / 50.0).cos());
        // identity embedding (stop-grad)
        let (e_id, _, _) = m.embed(&x_id);
        let lam = sp(m.lam);

        let mut g_w1 = vec![0.0; NF * H];
        let mut g_b1 = vec![0.0; H];
        let mut g_w2 = vec![0.0; H * K];
        let mut g_b2 = vec![0.0; K];
        let mut g_lm = 0.0;
        let mut tot = 0.0;
        let mut steps = 0u64;
        let nrm = (2.0 * ppe as f64).max(1.0);

        for _ in 0..ppe {
            let u = rng.u();
            let gi = cdf.partition_point(|&c| c < u).min(groups.len() - 1);
            let (rows, hs, _) = &groups[gi];
            let nr = rows.len();
            let (ia, ib) = (rng.idx(nr), rng.idx(nr));
            if ia == ib {
                continue;
            }
            let ta = hs[ia] * 100.0;
            let tb = hs[ib] * 100.0;
            let target = (ta - tb).signum();
            if target == 0.0 {
                continue;
            }
            let (sa, dda, ea, hpa, ha) = score(&m, &rows[ia], &e_id);
            let (sb, ddb, eb, hpb, hb) = score(&m, &rows[ib], &e_id);

            // RankNet + Huber-clamped MSE (margin bounded so the unbounded
            // distance can't explode the gradient).
            let z = -target * (sb - sa);
            let lrk = if z > 40.0 {
                z
            } else if z < -40.0 {
                0.0
            } else {
                (z.exp() + 1.0).ln()
            };
            let s = sig(-z);
            let rn = 0.7;
            let mse = 0.5;
            let dh = 50.0;
            let ra = (sa - ta).clamp(-dh, dh);
            let rb = (sb - tb).clamp(-dh, dh);
            let dsa = rn * target * s + mse * 2.0 * ra / nrm;
            let dsb = -rn * target * s + mse * 2.0 * rb / nrm;
            tot += lrk * rn + mse * (ra * ra + rb * rb) / nrm;
            steps += 1;

            // backprop score = 100 - lam*D, D = RMS(e - e_id)
            let mut bp = |ds: f64, x: &[f64], dd: f64, e: &[f64], hp: &[f64], h: &[f64]| {
                // dscore/dlam = -D
                g_lm += ds * (-dd) * sig(m.lam);
                // dscore/de_d = -lam * (e_d - e_id_d)/(K*D)
                let inv = 1.0 / (K as f64 * dd);
                let mut de = vec![0.0; K];
                for d in 0..K {
                    de[d] = ds * (-lam) * (e[d] - e_id[d]) * inv;
                }
                // e = W2·h + b2
                let mut dh = vec![0.0; H];
                for d in 0..K {
                    g_b2[d] += de[d];
                }
                for j in 0..H {
                    let hj = h[j];
                    let row = &m.w2[j * K..(j + 1) * K];
                    let mut acc = 0.0;
                    for d in 0..K {
                        g_w2[j * K + d] += de[d] * hj;
                        acc += de[d] * row[d];
                    }
                    dh[j] = acc;
                }
                // h = leaky(hp)
                for j in 0..H {
                    let dhp = if hp[j] >= 0.0 { dh[j] } else { dh[j] * m.leaky };
                    g_b1[j] += dhp;
                    if dhp != 0.0 {
                        for (di, &xi) in x.iter().enumerate().take(NF) {
                            if xi != 0.0 {
                                g_w1[di * H + j] += dhp * xi;
                            }
                        }
                    }
                }
            };
            bp(dsa, &rows[ia], dda, &ea, &hpa, &ha);
            bp(dsb, &rows[ib], ddb, &eb, &hpb, &hb);
        }

        t += 1;
        let bc1 = 1.0 - b1c.powi(t);
        let bc2 = 1.0 - b2c.powi(t);
        let upd = |w: &mut [f64], g: &[f64], mm: &mut [f64], vv: &mut [f64], decay: bool| {
            for i in 0..w.len() {
                let gi = if decay { g[i] + L2 * w[i] } else { g[i] };
                mm[i] = b1c * mm[i] + (1.0 - b1c) * gi;
                vv[i] = b2c * vv[i] + (1.0 - b2c) * gi * gi;
                w[i] -= lr * (mm[i] / bc1) / ((vv[i] / bc2).sqrt() + eps);
            }
        };
        upd(&mut m.w1, &g_w1, &mut m_w1, &mut v_w1, true);
        upd(&mut m.b1, &g_b1, &mut m_b1, &mut v_b1, false);
        upd(&mut m.w2, &g_w2, &mut m_w2, &mut v_w2, true);
        upd(&mut m.b2, &g_b2, &mut m_b2, &mut v_b2, false);
        {
            let mut a = [m.lam];
            let gg = [g_lm];
            let mut mm = [m_lm];
            let mut vv = [v_lm];
            upd(&mut a, &gg, &mut mm, &mut vv, false);
            m.lam = a[0];
            m_lm = mm[0];
            v_lm = vv[0];
        }
        if epoch % 20 == 0 || epoch == epochs - 1 {
            eprintln!(
                "epoch {epoch:3} lr={lr:.4} loss={:.4} λ={:.3}",
                tot / steps.max(1) as f64,
                sp(m.lam)
            );
        }
    }

    // eval
    let (e_id, _, _) = m.embed(&x_id);
    eprintln!("\n=== PANEL (SROCC, embedding-distance) ===");
    for (name, fname) in [
        ("cid22", "cid22_features_372col_2026-05-15.parquet"),
        ("kadid", "kadid_features_372col_2026-05-15.parquet"),
        ("tid", "tid_features_372col_2026-05-15.parquet"),
        ("konjnd", "konjnd_features_372col_2026-05-15.parquet"),
        ("aic3", "aic3_features_372col_2026-05-15.parquet"),
    ] {
        if let Some((mut rows, hs)) = load(&format!("{VAL}/{fname}"), name) {
            for r in rows.iter_mut() {
                standardize(r);
            }
            let preds: Vec<f64> = rows.iter().map(|r| score(&m, r, &e_id).0).collect();
            let sr = spearman(&preds, &hs);
            let pmin = preds.iter().cloned().fold(f64::INFINITY, f64::min);
            let pmax = preds.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            eprintln!(
                "  {name:8} SROCC={sr:.4}  range=[{pmin:.1},{pmax:.1}]  n={}",
                hs.len()
            );
        }
    }

    eprintln!("\n=== BLUR-LADDER (A2: above_id MUST be 0 by construction) ===");
    for c in ["color_blocks", "checker", "mandelbrot", "value_noise"] {
        let Ok(bytes) = std::fs::read(format!("/tmp/blur_ladder_{c}.featmat")) else {
            continue;
        };
        let nf = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let nr = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let mut sc = Vec::new();
        for r in 0..nr {
            let mut x = vec![0.0; NF];
            for d in 0..nf.min(NF) {
                let off = 8 + (r * nf + d) * 4;
                let raw = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as f64;
                x[d] = (raw - mean[d]) / std[d];
            }
            sc.push(score(&m, &x, &e_id).0);
        }
        let id = sc[0];
        let mut inv = 0;
        let mut ab = 0;
        for w in 1..sc.len() {
            if sc[w] > sc[w - 1] + 0.01 {
                inv += 1;
            }
            if sc[w] > id + 0.01 {
                ab += 1;
            }
        }
        let s: Vec<String> = sc.iter().map(|v| format!("{v:.1}")).collect();
        eprintln!("  {c:13} [{}]  inv={inv} above_id={ab}", s.join(" "));
    }
    eprintln!("\n(A1+A2-defect-half hold by construction: above_id=0 always.");
    eprintln!(" inv may be >0 — A3 is intentionally dropped. Resolution → wide negative range.)");
}
