//! Option-3 probe: PARTIAL-MONOTONE RESIDUAL metric.
//!
//!   score(x) = 100 − λ_m·(D_m(x_mono) − D_m(id)) − F_b(x_free)
//!
//! where
//!   - x_mono = the 300 sign-safe features (feature_sign_mask: pin_geq0),
//!     x_free  = the 72 sign-flip features.
//!   - D_m(x_mono) = Σ_j c_j · LeakyReLU(W_m·x_mono + b_m)_j  with W_m, c ≥ 0
//!     (softplus) → a monotone-↑ scalar "dissimilarity" in x_mono. Since
//!     identity has the minimal x_mono (error features = their min at
//!     identity), D_m(x) ≥ D_m(id) ⇒ the subtracted term is ≥ 0.
//!   - F_b(x_free) = δ·tanh( relu(M(x_free) − M(id)) / δ )  ∈ [0, δ): a
//!     BOUNDED downward refinement from the free features (unconstrained
//!     MLP M), 0 at identity.
//!
//! Guarantees by construction:
//!   A1 bounded ≤ 100 (both subtracted terms ≥ 0),
//!   A2 identity is the UNIQUE max (both terms = 0 only at identity),
//!   A3 monotone-↓ in every sign-safe feature (D_m monotone-↑, F_b can
//!      only lower the score by ≤ δ ⇒ score is monotone-↓ up to a bounded
//!      δ slack — "bounded non-monotonicity").
//! Crucially the mono term is UNBOUNDED below ⇒ a terrible distortion
//! gets a genuinely low score (not capped at 99) → resolution preserved.
//!
//! Falsify: does it hold the human-MOS panel (esp. KADID/TID, which
//! v47-strict lost by dropping x_free) AND give a usable dial AND stay
//! monotone? Self-contained probe; no production-runtime surgery.

use std::path::PathBuf;
use zensim_validate::parquet_loader::load_parquet;
// Spearman is zenstats' — the single owner of stat math (imazen/zensimx#41).
use zenstats::panel::spearman;

const NF: usize = 372;
const HM: usize = 96; // mono backbone hidden
const HF: usize = 48; // free MLP hidden
const DELTA: f64 = 25.0; // bounded free-modulation magnitude
const CANON: &str = "/mnt/v/zen/zensim-training/canonical-2026-05-21/train";
const VAL: &str = zensim_validate::eval_roots::STORED_FEATURES_ROOT_2026_05_15;
const MASK: &str = "/home/lilith/work/zen/zensimx/benchmarks/feature_sign_mask_2026-05-26.tsv";

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
}

#[inline]
fn sp(x: f64) -> f64 {
    if x > 20.0 { x } else { (x.exp() + 1.0).ln() }
}
#[inline]
fn sig(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Read the sign mask → (mono_idx, free_idx).
fn load_mask() -> (Vec<usize>, Vec<usize>) {
    let txt = std::fs::read_to_string(MASK).expect("mask");
    let (mut mono, mut free) = (Vec::new(), Vec::new());
    for (i, line) in txt.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let mut c = line.split('\t');
        let idx: usize = c.next().and_then(|s| s.trim().parse().ok()).unwrap();
        let m = c.next().unwrap_or("").trim();
        if m == "pin_geq0" {
            mono.push(idx);
        } else {
            free.push(idx);
        }
    }
    (mono, free)
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
    mono: Vec<usize>,
    free: Vec<usize>,
    // mono backbone (W_m≥0 via softplus on raw θ)
    wm: Vec<f64>, // [NM*HM]
    bm: Vec<f64>, // [HM]
    cm: Vec<f64>, // [HM]  (≥0 via softplus) reducer
    lam_m: f64,   // θ → softplus
    // free MLP
    wf: Vec<f64>, // [NFREE*HF]
    bf: Vec<f64>, // [HF]
    vf: Vec<f64>, // [HF] reducer → scalar M
    leaky: f64,
}
impl Model {
    fn new(rng: &mut Rng, mono: Vec<usize>, free: Vec<usize>) -> Self {
        let nm = mono.len();
        let nfree = free.len();
        let sm = (2.0 / nm as f64).sqrt();
        let sf = (2.0 / nfree as f64).sqrt();
        Self {
            // W_m ≥ 0 (monotone backbone): init non-negative; projected ≥0
            // after every Adam step in the loop.
            wm: (0..nm * HM).map(|_| rng.g().abs() * sm).collect(),
            bm: vec![0.0; HM],
            cm: vec![0.0; HM],
            lam_m: -1.0, // softplus(-1)=0.31 initial λ_m
            wf: (0..nfree * HF).map(|_| rng.g() * sf).collect(),
            bf: vec![0.0; HF],
            vf: vec![0.0; HF],
            leaky: 0.01,
            mono,
            free,
        }
    }
    /// mono dissimilarity D_m(x) + cache h_pre/h for backprop.
    fn dm(&self, x: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
        let nm = self.mono.len();
        let mut hp = self.bm.clone();
        for (mi, &fi) in self.mono.iter().enumerate() {
            let xi = x[fi];
            if xi == 0.0 {
                continue;
            }
            let row = &self.wm[mi * HM..(mi + 1) * HM];
            for j in 0..HM {
                hp[j] += xi * row[j];
            }
        }
        let _ = nm;
        let h: Vec<f64> = hp
            .iter()
            .map(|&v| if v >= 0.0 { v } else { self.leaky * v })
            .collect();
        let mut d = 0.0;
        for (&c, &hv) in self.cm.iter().zip(h.iter()) {
            d += sp(c) * hv;
        }
        (d, hp, h)
    }
    /// free scalar M(x).
    fn mf(&self, x: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
        let mut hp = self.bf.clone();
        for (fi2, &fi) in self.free.iter().enumerate() {
            let xi = x[fi];
            if xi == 0.0 {
                continue;
            }
            let row = &self.wf[fi2 * HF..(fi2 + 1) * HF];
            for j in 0..HF {
                hp[j] += xi * row[j];
            }
        }
        let h: Vec<f64> = hp
            .iter()
            .map(|&v| if v >= 0.0 { v } else { self.leaky * v })
            .collect();
        let mut m = 0.0;
        for (&v, &hv) in self.vf.iter().zip(h.iter()) {
            m += v * hv;
        }
        (m, hp, h)
    }
}

fn main() {
    let mut rng = Rng::new(0x0273_9999_4242_1357u64.wrapping_add(7));
    let (mono, free) = load_mask();
    eprintln!("mask: {} mono / {} free", mono.len(), free.len());

    eprintln!("loading train…");
    // V2 probe: ADD safesyn (196k rows — the bulk of the corpus the V1 probe
    // omitted). Hypothesis under test: the V1 0.61 CID22 is data-starvation,
    // not an architecture ceiling. safesyn target is ssim2-derived (synthetic,
    // CID22-leak-purged) → weight 1.0 so it shapes rank without swamping the
    // human sets' calibration.
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

    let mut m = Model::new(&mut rng, mono, free);
    let nm = m.mono.len();
    let nfree = m.free.len();

    // Adam buffers
    macro_rules! adamvec {
        ($n:expr) => {
            (vec![0.0; $n], vec![0.0; $n])
        };
    }
    let (mut m_wm, mut v_wm) = adamvec!(nm * HM);
    let (mut m_bm, mut v_bm) = adamvec!(HM);
    let (mut m_cm, mut v_cm) = adamvec!(HM);
    let (mut m_lm, mut v_lm) = (0.0f64, 0.0f64);
    let (mut m_wf, mut v_wf) = adamvec!(nfree * HF);
    let (mut m_bf, mut v_bf) = adamvec!(HF);
    let (mut m_vf, mut v_vf) = adamvec!(HF);
    let (b1, b2, eps): (f64, f64, f64) = (0.9, 0.999, 1e-8);
    let mut t = 0i32;

    let score = |m: &Model, x: &[f64], dm_id: f64, m_id: f64| -> (f64, f64, f64, f64) {
        // returns (score, lam_m, dm, free_arg) where free_arg = relu(M - M_id)
        let lam = sp(m.lam_m);
        let (dm, _, _) = m.dm(x);
        let (mfv, _, _) = m.mf(x);
        let free_arg = (mfv - m_id).max(0.0);
        let fb = DELTA * (free_arg / DELTA).tanh();
        (100.0 - lam * (dm - dm_id) - fb, lam, dm, free_arg)
    };

    let epochs = 140;
    let ppe = 60000; // doubled: safesyn adds 196k rows to the sampling pool
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
        let lr = 1e-3 * 0.5 * (1.0 + (std::f64::consts::PI * (epoch % 45) as f64 / 45.0).cos());
        // identity anchors (stop-grad on id terms)
        let (dm_id, _, _) = m.dm(&x_id);
        let (m_id, _, _) = m.mf(&x_id);

        let mut g_wm = vec![0.0; nm * HM];
        let mut g_bm = vec![0.0; HM];
        let mut g_cm = vec![0.0; HM];
        let mut g_lm = 0.0;
        let mut g_wf = vec![0.0; nfree * HF];
        let mut g_bf = vec![0.0; HF];
        let mut g_vf = vec![0.0; HF];
        let mut tot = 0.0;
        let mut steps = 0u64;

        for _ in 0..ppe {
            let u = rng.u();
            let gi = cdf.partition_point(|&c| c < u).min(groups.len() - 1);
            let (rows, hs, _) = &groups[gi];
            let nr = rows.len();
            let ia = (m_lr_idx(&mut rng, nr), m_lr_idx(&mut rng, nr));
            let (ia, ib) = ia;
            if ia == ib {
                continue;
            }
            let ta = hs[ia] * 100.0;
            let tb = hs[ib] * 100.0;
            let target = (ta - tb).signum();
            if target == 0.0 {
                continue;
            }
            let (sa, lam, _da, fa_arg) = score(&m, &rows[ia], dm_id, m_id);
            let (sb, _, _db, fb_arg) = score(&m, &rows[ib], dm_id, m_id);

            // RankNet + MSE
            let z = -target * (sb - sa);
            let lrk = if z > 40.0 {
                z
            } else if z < -40.0 {
                0.0
            } else {
                (z.exp() + 1.0).ln()
            };
            let s = sig(-z);
            let rn = 0.6;
            let mse = 0.6;
            let nrm = (2.0 * ppe as f64).max(1.0);
            // Huber-clamp the MSE residual: the unbounded mono term sends
            // heavy-distortion scores very negative; an unclamped MSE
            // gradient (∝ residual) then explodes. Clamping the residual
            // to ±δ_h bounds the gradient while keeping the score unbounded
            // (full resolution preserved).
            let dh = 50.0;
            let ra = (sa - ta).clamp(-dh, dh);
            let rb = (sb - tb).clamp(-dh, dh);
            let dsa = rn * target * s + mse * 2.0 * ra / nrm;
            let dsb = -rn * target * s + mse * 2.0 * rb / nrm;
            tot += lrk * rn + mse * (ra * ra + rb * rb) / nrm;
            steps += 1;

            // backprop one side: score = 100 - lam*(dm - dm_id) - DELTA*tanh(relu(M-M_id)/DELTA)
            let mut bp = |ds: f64, x: &[f64], free_arg: f64| {
                // mono path: dscore/ddm = -lam ; dm = Σ sp(c_j) h_j
                let (_dm, hp_m, h_m) = m.dm(x);
                let dscore_ddm = -lam;
                for j in 0..HM {
                    let cj = sp(m.cm[j]);
                    // dL/dh_m_j = ds * dscore_ddm * cj
                    let dl_dh = ds * dscore_ddm * cj;
                    let dl_dhp = if hp_m[j] >= 0.0 {
                        dl_dh
                    } else {
                        dl_dh * m.leaky
                    };
                    g_bm[j] += dl_dhp;
                    // dL/dc_j (sp' = sigmoid)
                    g_cm[j] += ds * dscore_ddm * h_m[j] * sig(m.cm[j]);
                    for (mi, &fi) in m.mono.iter().enumerate() {
                        let xi = x[fi];
                        if xi != 0.0 {
                            g_wm[mi * HM + j] += dl_dhp * xi;
                        }
                    }
                }
                // dL/dlam = ds * -(dm - dm_id) * sp'(lam_θ)
                let (dm_now, _, _) = m.dm(x);
                g_lm += ds * (-(dm_now - dm_id)) * sig(m.lam_m);
                // free path: fb = DELTA*tanh(free_arg/DELTA); dscore/dfree_arg = -sech²(free_arg/DELTA)
                if free_arg > 0.0 {
                    let th = (free_arg / DELTA).tanh();
                    let dfb_darg = 1.0 - th * th; // d(DELTA*tanh(a/DELTA))/da
                    let dscore_dm = -dfb_darg; // since free_arg=relu(M-M_id), d/dM=1 when >0
                    let (_m, hp_f, h_f) = m.mf(x);
                    for j in 0..HF {
                        let dl_dh = ds * dscore_dm * m.vf[j];
                        let dl_dhp = if hp_f[j] >= 0.0 {
                            dl_dh
                        } else {
                            dl_dh * m.leaky
                        };
                        g_bf[j] += dl_dhp;
                        g_vf[j] += ds * dscore_dm * h_f[j];
                        for (fi2, &fi) in m.free.iter().enumerate() {
                            let xi = x[fi];
                            if xi != 0.0 {
                                g_wf[fi2 * HF + j] += dl_dhp * xi;
                            }
                        }
                    }
                }
            };
            bp(dsa, &rows[ia], fa_arg);
            bp(dsb, &rows[ib], fb_arg);
        }

        t += 1;
        let bc1 = 1.0 - b1.powi(t);
        let bc2 = 1.0 - b2.powi(t);
        let upd = |w: &mut [f64], g: &[f64], mm: &mut [f64], vv: &mut [f64]| {
            for i in 0..w.len() {
                mm[i] = b1 * mm[i] + (1.0 - b1) * g[i];
                vv[i] = b2 * vv[i] + (1.0 - b2) * g[i] * g[i];
                w[i] -= lr * (mm[i] / bc1) / ((vv[i] / bc2).sqrt() + eps);
            }
        };
        upd(&mut m.wm, &g_wm, &mut m_wm, &mut v_wm);
        // Project the monotone backbone weights ≥0 (makes D_m monotone-↑
        // in x_mono → A1/A2/A3 by construction).
        for w in m.wm.iter_mut() {
            if *w < 0.0 {
                *w = 0.0;
            }
        }
        upd(&mut m.bm, &g_bm, &mut m_bm, &mut v_bm);
        upd(&mut m.cm, &g_cm, &mut m_cm, &mut v_cm);
        upd(&mut m.wf, &g_wf, &mut m_wf, &mut v_wf);
        upd(&mut m.bf, &g_bf, &mut m_bf, &mut v_bf);
        upd(&mut m.vf, &g_vf, &mut m_vf, &mut v_vf);
        {
            let mut a = [m.lam_m];
            let gg = [g_lm];
            let mut mm = [m_lm];
            let mut vv = [v_lm];
            upd(&mut a, &gg, &mut mm, &mut vv);
            m.lam_m = a[0];
            m_lm = mm[0];
            v_lm = vv[0];
        }
        if epoch % 20 == 0 || epoch == epochs - 1 {
            eprintln!(
                "epoch {epoch:3} lr={lr:.4} loss={:.4} λm={:.3}",
                tot / steps.max(1) as f64,
                sp(m.lam_m)
            );
        }
    }

    // eval
    let (dm_id, _, _) = m.dm(&x_id);
    let (m_id, _, _) = m.mf(&x_id);
    eprintln!("\n=== PANEL (SROCC, partial-monotone residual) ===");
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
            let preds: Vec<f64> = rows.iter().map(|r| score(&m, r, dm_id, m_id).0).collect();
            let sr = spearman(&preds, &hs);
            let pmin = preds.iter().cloned().fold(f64::INFINITY, f64::min);
            let pmax = preds.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            eprintln!(
                "  {name:8} SROCC={sr:.4}  range=[{pmin:.1},{pmax:.1}]  n={}",
                hs.len()
            );
        }
    }

    eprintln!("\n=== BLUR-LADDER monotonicity ===");
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
            sc.push(score(&m, &x, dm_id, m_id).0);
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
    eprintln!(
        "\n(A2 → above_id=0 by construction; A3 → inv ≤ bounded-δ slack; resolution → wide negative range)"
    );
}

fn m_lr_idx(rng: &mut Rng, n: usize) -> usize {
    (rng.n() as usize) % n
}
