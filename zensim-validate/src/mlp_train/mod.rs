//! Two-layer MLP trainer for V0_4.
//!
//! Architecture: `n_features → n_hidden (LeakyReLU) → 1 (Identity)`.
//! Loss: RankNet pairwise (sigmoid cross-entropy on signed distance
//! deltas). Optimizer: Adam with cosine annealing.
//!
//! Output: a ZNPR v3 byte stream that loads via
//! `zensim::mlp::Model::from_bytes` (the `zensim::mlp` module re-exports
//! `zenpredict::Model`).
//!
//! This is the runtime-side counterpart to zenpicker's Python
//! distillation pipeline (`tools/train_hybrid.py`). Pure Rust, no
//! external numerics deps — the network is small enough (228×32 +
//! 32 = ~7.3K weights) that hand-rolled Adam is plenty fast.
//!
//! ## Multi-dataset training (V0_4 lessons from V0_2 audit)
//!
//! V0_2 was trained on a single concordant synthetic dataset; CMA-ES
//! hit higher synthetic SROCC than the shipped NM weights but lost
//! on KADIK / TID human holdouts. The takeaway for V0_4: ruthlessly
//! gate on human-dataset SROCC, not synthetic.
//!
//! This trainer takes [`TrainingGroup`]s with explicit `train_weight`.
//! Groups with `train_weight > 0` contribute to RankNet pair sampling
//! in proportion to their weight; groups with `train_weight == 0` are
//! validation-only — their per-epoch SROCC is logged and the best
//! model is the one with the highest validation mean.

use rayon::prelude::*;
use std::time::Instant;
use zenpredict::FeatureTransform;
use zenpredict::{Activation, WeightDtype};
use zenpredict_bake::{BakeLayer, BakeMetadataEntry, BakeRequest, bake};

// Adam SIMD kernel — replaces the per-element scalar loop in
// `AdamState::step` with an AVX-512 (f64x8) / AVX2 (f64x4) / NEON / WASM
// / scalar dispatched kernel. Math is bit-identical to the scalar
// reference (hardware `sqrt`, FMA fusion only where LLVM would already
// fuse the scalar version).
pub mod sampling;
pub mod strategy;
use crate::adam_simd;

// Norm-in-Norm + RankNet hybrid loss (Li, Jiang, Jiang 2020,
// arXiv:2008.03889). Opt-in via `MlpHyperparams::norm_in_norm_weight >
// 0`. Computes a batch-correlated auxiliary loss on the 2K predictions
// produced by K mini-batch pair forwards, with closed-form gradient
// added to the RankNet per-prediction backprop direction. See module
// docs for the exact formula and gradient derivation.
mod loss_norm_in_norm;

mod goals;

pub mod minmax_monotone;
pub use goals::{ValidationPolicy, compute_goal_scores};

/// Knobs for [`train_mlp`]. Defaults match the V0_4 placeholder
/// architecture (228 → 32 → 1) with `Min` validation gating.
#[derive(Clone, Debug)]
pub struct MlpHyperparams {
    pub n_hidden: usize,
    pub n_epochs: usize,
    pub pairs_per_epoch: usize,
    pub initial_lr: f64,
    pub leaky_alpha: f64,
    /// The single seed. Drives BOTH the init stream and the sample stream
    /// unless one is overridden below — which is how every bake trained
    /// before 2026-09-04 was produced, and why those bakes cannot separate
    /// an init effect from a sampling effect.
    pub seed: u64,
    /// Override the weight-init stream's seed. `None` = use `seed`.
    ///
    /// The two streams were ALREADY independent internally (the separation
    /// predates this field: it exists so a 228-vs-372 A/B sees the same pair
    /// draws even though init consumes a different number of normals). These
    /// fields only expose that split to the CLI, so a run can hold the drawn
    /// subset fixed while varying the init, or vice versa.
    pub init_seed: Option<u64>,
    /// Override the pair-sampling stream's seed. `None` = use `seed`.
    pub sample_seed: Option<u64>,
    pub log_every: usize,
    /// H-TRAJ (balance campaign 2026-08-28): dump a spline-less bake of the
    /// CURRENT projected net every N epochs (0 = off). Fires at validation
    /// points only, so the effective cadence is epochs that are multiples of
    /// BOTH `log_every` and N. Wired for the per-sample-α 0-hidden lane;
    /// other architectures refuse loudly (assert) rather than skip silently.
    pub dump_checkpoints_every: usize,
    /// Directory for `dump_checkpoints_every` output (`ckpt_epochNNN.bin`).
    /// None = current directory.
    pub dump_checkpoints_dir: Option<std::path::PathBuf>,
    /// L2 regularization on layer weights (not biases). 0 disables.
    pub l2_lambda: f64,
    /// Stop after this many epochs of no validation improvement.
    /// 0 disables early stopping.
    pub early_stop_patience: usize,
    pub validation_policy: ValidationPolicy,
    /// Per-group stat aggregation for checkpoint selection. See
    /// `panel::ValAggregate` for options. Default `GeomeanSPP` per
    /// Mohammadi 2025's finding that multi-stat evaluation is required.
    pub val_aggregate: crate::panel::ValAggregate,
    /// Cycle-9 row-weight boost for B0 + B1 (low-quality) rows.
    /// When > 1.0, biases per-step pair sampling within each training
    /// group toward rows whose `human_score` (0-100 scale) is below 50
    /// (full boost) or in [50, 65) (sqrt(boost)). Default 1.0 = no-op,
    /// uniform within-group sampling. Composes multiplicatively with
    /// `mid_q_boost`. The original Python-side experiments (cycle-9,
    /// zensim 4b998258) tested 1.5x and 3x; 1.5x was the sweet spot
    /// for B0 SROCC gains without B2 regression.
    pub low_q_boost: f64,
    /// Cycle-12 row-weight boost for B1 + B2 (medium-quality) rows.
    /// When > 1.0, biases per-step pair sampling toward rows whose
    /// `human_score` is in [50, 90). Default 1.0 = no-op. The Python-
    /// side cycle-12 finding (zensim 4da7d1fa) at boost=1.5 was a
    /// σ-tightener (4x tighter seed-to-seed CID22 variance) with small
    /// +0.003 mean lift — useful for downstream codec orchestrators
    /// that need stable per-image ranking.
    pub mid_q_boost: f64,
    /// V0_20a-era row-weight boost for B3 (visually-lossless, human_score
    /// ≥ 90) rows. When > 1.0, biases per-step pair sampling toward
    /// rows in the visually-lossless tail. Default 1.0 = no-op.
    ///
    /// **Motivation**: with val_policy=min selecting on KADID/TID/KonJND
    /// (none of which span B3 well), the trainer structurally underfits
    /// the visually-lossless tail. The B3 band is exactly where IW-SSIM-
    /// style features (Wang & Li 2011, Mohammadi 2025) carry signal —
    /// `--high-q-boost` ensures the trainer gets enough B3 supervision
    /// for the MLP to learn the relevant weights. Recommended values:
    /// 2.0-4.0 depending on B3 sample density. Composes multiplicatively
    /// with `mid_q_boost` (boundary at 90 means it's mutually exclusive
    /// with `mid_q_boost`'s [50, 90) range).
    pub high_q_boost: f64,
    /// Output weight dtype for baked ZNPR v3. Default F32 matches every
    /// shipped bake through V0_17. I8 saves ~74 % bin size with no
    /// measurable SROCC change (verified on V0_18 across KADID, TID,
    /// CID22, AIC-3, AIC-4, KonJND). F16 saves ~50 % with bit-identical
    /// SROCC. Dequantization is inline in `zenpredict::saxpy_matmul_*`.
    pub out_dtype: WeightDtype,

    /// V0_20 input-shaping research (2026-05-14): optional per-feature
    /// transforms applied BEFORE the scaler in the runtime. When set,
    /// the bake's `zentrain.feature_transforms` metadata records the
    /// list; the runtime applies them via
    /// [`zenpredict::Predictor::predict_transformed`].
    ///
    /// Length MUST equal `n_features` when present. Caller is
    /// responsible for applying the same transforms to feature_rows
    /// before constructing `TrainingGroup` (since the trainer never
    /// sees the raw rows separately — the scaler is fit to whatever
    /// the trainer receives). An all-`Identity` vector emits no
    /// metadata (consumers treat absence as all-identity).
    pub feature_transforms: Option<Vec<FeatureTransform>>,

    /// V0_20 parameterized feature-transform params (added 2026-05-15).
    /// Parallel to `feature_transforms`; each inner `Vec<f32>` is the
    /// param list for that feature (empty for non-parameterized
    /// variants). When `Some(_)` AND any inner vec is non-empty,
    /// `zentrain.feature_transform_params` metadata is emitted in the
    /// bake. The runtime reads it via `Model::feature_transform_params()`
    /// and applies via `FeatureTransform::apply_with_params`.
    pub feature_transform_params: Option<Vec<Vec<f32>>>,

    /// T8.1 (2026-05-16): mini-batch SGD. When > 1, the trainer
    /// accumulates K RankNet pair-gradient contributions between each
    /// Adam step instead of stepping per-pair. Default 1 = per-pair
    /// SGD (bit-identical to legacy behavior).
    ///
    /// **Behavior change**: Adam's internal `t` counter increments K×
    /// less often, so bias correction `(1 - β^t)` decays K× slower —
    /// mathematically still correct, just on a different schedule. A
    /// final-flush Adam step at epoch end handles leftover accumulated
    /// gradients when `pairs_per_epoch % K != 0`, so gradients don't
    /// carry between epochs.
    ///
    /// **Convergence**: K > 1 produces less noisy gradients than
    /// per-pair SGD. Usually helps generalization; can hurt
    /// regularization on small datasets. Recommended sweep: K ∈ {1, 8,
    /// 64, 256} on a representative recipe.
    ///
    /// **Determinism**: the sample sequence (which (group, ia, ib)
    /// tuple is drawn at each step) is bit-identical regardless of K
    /// — only the Adam update cadence changes. Bake bytes for
    /// `train_mlp(K=1)` are identical to the legacy trainer; bake
    /// bytes for `train_mlp(K=K)` are deterministic given seed alone.
    pub minibatch_size: usize,

    /// T8.2 (2026-05-16): enable rayon parallel-batch within each
    /// mini-batch. When `true` AND `minibatch_size > 1`, the K
    /// per-pair gradient contributions are computed concurrently via
    /// `rayon::par_iter` with per-thread accumulators that reduce
    /// into a single LocalGrads before the Adam step.
    ///
    /// **Default `true`** (2026-05-17 CLAUDE.md "fast mode on by
    /// default" directive). Parallelism does NOT change behavior:
    /// the sample-drawing sequence runs sequentially on the main
    /// RNG to produce a `Vec` of K (group_idx, ia, ib) tuples, the
    /// per-pair LocalGrads reduce runs in source order — same
    /// `seed + minibatch_size` produces bit-identical bake bytes
    /// regardless of thread count (T8.2 determinism gate).
    ///
    /// When `minibatch_size == 1` this flag is ignored (the
    /// sequential path is always taken — no per-batch overhead).
    pub parallel_batch: bool,

    /// PWRC-aligned per-pair weighting on the RankNet loss (Wu et al.
    /// 2018, "A Perceptually Weighted Rank Correlation Indicator for
    /// Objective Image Quality Assessment", IEEE TIP, DOI
    /// 10.1109/TIP.2018.2799331; reference MATLAB at
    /// <https://github.com/wqb-uestc/PWRC>). When `false` (default),
    /// every drawn (ia, ib) pair contributes loss & gradient at
    /// weight 1.0 — bit-identical to the pre-PWRC trainer.
    ///
    /// When `true`, each pair's contribution to the loss & its gradient
    /// is multiplied by `pwrc_pair_weight(MOS_a, MOS_b)`. Reference
    /// `PWRC.m`:
    ///
    /// ```text
    /// level    = max(label_r_left, label_r_right) - 1     // rank-quality
    /// diff     = |label_r - pred_r|  (left) + |label_r - pred_r|  (right)
    /// w_pre    = exp( level/(len-1) + diff/(2*(len-1)) )  // unnormalized
    /// w_norm   = w_pre / sum(w_pre)                       // normalized
    /// ```
    ///
    /// The `diff` term penalizes ranking mismatch between predictions
    /// and labels, but it depends on the CURRENT model's full-corpus
    /// ranking — recomputing it per Adam step would dominate runtime.
    /// We drop it and keep the label-only term, which is the
    /// "perceptually weighted" piece: pairs whose maximum quality is
    /// high get more weight. In rank-normalized form on dense MOS that
    /// reduces to `w(a, b) = exp(max(MOS_a, MOS_b) / 100)`, the
    /// trainer's closed-form default when `pwrc_band_weights` is None.
    ///
    /// **Band-weight override**: if `pwrc_band_weights` is `Some(v)`,
    /// the per-pair weight is `v[band_idx(max(MOS_a, MOS_b))]` where
    /// `band_idx(s) = clamp(floor(s/10), 0, len-1)`. A 10-element
    /// vector matches the 10-band B0..B9 grid (CLAUDE.md "Per-band
    /// reporting rule"). Use this to invert the Wu 2018 direction —
    /// e.g., `[5,4,3,2,1.5,1,1,1,1,1]` upweights B0..B5 (zensim
    /// "low-q priority").
    ///
    /// No per-pair normalization is applied: the published formula
    /// divides by `omega = sum_pairs w_unnorm` to make PWRC a unit-
    /// range indicator, but for SGD any constant scale is absorbed by
    /// the learning rate. Skipping the normalization keeps the
    /// gradient magnitude in the same ballpark as the unweighted
    /// trainer (mean weight ≈ exp(0.5) ≈ 1.65 under the closed-form
    /// default), so `--initial-lr` doesn't need retuning.
    pub pwrc_pair_weight: bool,

    /// PWRC sensory threshold `T` (Wu et al. 2018) on the same scale
    /// as `human_scores` (= `score_zensim` in 0..100 units). Pairs with
    /// `|MOS_a - MOS_b| < pwrc_sensory_threshold` are dropped (they're
    /// perceptually tied and not informative for ranking learning).
    /// Default `0.0` = no drop, matching the legacy `target.signum() ==
    /// 0.0` "skip exact ties only" behavior.
    ///
    /// The published recommendation is `T = 5` MOS units on a 100-unit
    /// scale — pairs within ~5 % of MOS range are dropped. Set to 0.0
    /// to disable. Active only when `pwrc_pair_weight` is true.
    ///
    /// **Determinism note**: dropped pairs decrement the per-epoch
    /// step counter but the RNG sequence (which `(ia, ib)` is drawn
    /// at each step) is unchanged — same `seed` produces same draws
    /// regardless of `pwrc_sensory_threshold`.
    pub pwrc_sensory_threshold: f64,

    /// Optional per-band weight vector for PWRC pair weighting.
    /// `None` = use closed-form Wu 2018 default `exp(max_MOS / 100)`.
    /// `Some(v)` = look up `v[clamp(floor(max_MOS / band_width), 0,
    /// v.len()-1)]` where `band_width = 100.0 / v.len()`. Typical:
    /// 10-element vector aligned with the B0..B9 grid. Active only
    /// when `pwrc_pair_weight` is true.
    pub pwrc_band_weights: Option<Vec<f64>>,

    /// Norm-in-Norm + RankNet hybrid loss weight `β` (Li, Jiang, Jiang
    /// 2020, "Norm-in-Norm Loss with Faster Convergence and Better
    /// Performance for Image Quality Assessment", ACM MM, arXiv:
    /// 2008.03889; reference impl
    /// <https://github.com/lidq92/LinearityIQA>).
    ///
    /// **Default `0.0` = pure RankNet** — bit-identical bake bytes to
    /// the pre-NiN trainer at any `minibatch_size`. When `> 0.0`, an
    /// auxiliary loss term is added on top of every mini-batch's
    /// RankNet gradients:
    ///
    /// ```text
    ///   total_loss = ranknet_loss + norm_in_norm_weight · norm_in_norm_loss
    /// ```
    ///
    /// where `norm_in_norm_loss` is computed over the `2K` predictions
    /// generated by the `K` pair forwards in the mini-batch (each pair
    /// contributes both `y_a` and `y_b`, paired with the matching
    /// human scores `MOS_a, MOS_b`). The Norm-in-Norm loss itself,
    /// with `ε = 1e-8`:
    ///
    /// ```text
    ///   ŝ_n = (Ŝ - mean(Ŝ)) / (||Ŝ - mean(Ŝ)||_q + ε)
    ///   s_n = (S - mean(S)) / (||S - mean(S)||_q + ε)
    ///   loss_NiN = (||ŝ_n - s_n||_p / scale)^p
    ///   scale    = 2^max(1, 1/q) · N^max(0, 1/p - 1/q)
    /// ```
    ///
    /// Per Li 2020 Table 2 last row (KonIQ-10k headline result), the
    /// recommended hybrid is `β = 0.1, p = 1, q = 2` — that
    /// configuration lifted SROCC 0.928 → 0.937 (RankNet alone →
    /// hybrid) and PLCC 0.928 → 0.947 vs MSE 0.851 / 0.825.
    ///
    /// **Sign convention**: the zensim trainer's MLP outputs
    /// `raw_distance` (LOWER = more similar; opposite of MOS). The NiN
    /// loss therefore pairs `Ŝ` with `−MOS` rather than `+MOS` so the
    /// gradient pulls the prediction surface in the
    /// `distance ↔ −MOS` direction — same orientation as the RankNet
    /// `(MOS_a − MOS_b).signum() · (y_a − y_b)` convention already
    /// established in the trainer.
    ///
    /// **Requires `minibatch_size ≥ 16`** for stable batch statistics.
    /// The trainer errors out at `K < 16` when `norm_in_norm_weight >
    /// 0` with a clear message ("requires --minibatch-size >= 16 for
    /// stable batch statistics; bump K or disable").
    pub norm_in_norm_weight: f64,

    /// Inner-norm exponent `p` for the Norm-in-Norm loss (Li 2020).
    /// Default `1.0` matches Table 1's best single-loss setting and
    /// the headline `(β=0.1, p=1, q=2)` hybrid recommendation. Set to
    /// `2.0` alongside `norm_in_norm_q = 2.0` to recover the
    /// PLCC-induced special case (paper Section 2.2, Eqn 14):
    /// `loss ∝ (1 − PLCC)`.
    ///
    /// Only meaningful when `norm_in_norm_weight > 0`.
    pub norm_in_norm_p: f64,

    /// Outer (denominator) q-norm exponent for the Norm-in-Norm loss
    /// (Li 2020). Default `2.0` recovers the z-score-equivalent
    /// normalization of the reference impl
    /// (<https://github.com/lidq92/LinearityIQA/blob/master/IQAloss.py>
    /// — `torch.norm(y_pred, p=q)`).
    ///
    /// Only meaningful when `norm_in_norm_weight > 0`.
    pub norm_in_norm_q: f64,

    /// EX-2 std-pool head (`PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md §3`).
    /// When `true`, the trainer replaces the standard `n_hidden → 1`
    /// linear output with `pool[μ, σ, max, p_6] → 4 → 1` reducer
    /// (GMSD + Butteraugli p-norm + IW-style pooling). The bake emits
    /// a passthrough second layer plus a `zentrain.pool_head_reducer`
    /// metadata key carrying `[w_μ, w_σ, w_max, w_p6, b, p_norm]` —
    /// the runtime detects the metadata and routes through the pool
    /// head. Default `false` keeps the legacy V_22-style head.
    ///
    /// **SIMD note**: the current pool-head backprop is scalar (the
    /// SIMD parity work is queued). When `pool_head` is `true`, the
    /// trainer uses a scalar forward/backward path for the pool stats
    /// and reducer; layer-1 gradient accumulation is unchanged scalar
    /// math. NiN, parallel-batch (>1 K), and TV regularizer DO compose
    /// with pool_head (each backward routes ∂L/∂y through the pool
    /// chain rule in `zensim_train_core::pool_head::backprop_step_pool_head`).
    pub pool_head: bool,

    /// EX-2 follow-up: hybrid pool + rank head. When `true`, the trainer
    /// runs BOTH the rank-net scalar head AND the pool-head reducer on
    /// the same encoder, then blends via a sigmoid-bounded learned `α`
    /// (`y = α · y_rank + (1−α) · y_pool`).
    ///
    /// Both binary pool/no-pool experiments produced Pareto-tight
    /// tradeoffs (pool wins KonJND, rank wins CID22 / KADID / TID). The
    /// learned `α` lets the loss balance the two paths per-bake instead
    /// of forcing all-or-nothing.
    ///
    /// **What composes (v0 wiring, 2026-05-18):**
    /// - RankNet pair loss + Adam + cosine LR + early-stop
    /// - `--minibatch-size K` (sequential gradient accumulation)
    /// - `--pwrc-pair-weight` + `--pwrc-sensory-threshold` +
    ///   `--pwrc-band-weights`
    /// - TV regularizer (`--tv-pairs-file` + `--tv-weight`)
    /// - L2 (`--l2`) on layer-1 + rank_w + reducer_w (alpha_logit
    ///   unregularized)
    /// - Low/Mid/High-q row boosts
    ///
    /// **What is omitted (v0):**
    /// - NiN hybrid loss (per-prediction grad scatter) — falls back to
    ///   plain RankNet. NiN integration is queued for v1.
    /// - Parallel-batch flag (sequential mini-batch path; per-pair work
    ///   is small enough at h=128 to run V_22-LARGE recipe in ~25 min
    ///   wall on the 7950X).
    ///
    /// Mutually exclusive with `pool_head` — the trainer panics if both
    /// are set. When `hybrid_head` is true, the bake metadata key
    /// changes from `zentrain.pool_head_reducer` to
    /// `zentrain.hybrid_head` (with the full payload including
    /// `rank_w[n_hidden]`, `rank_b`, `alpha_logit`, `reducer_w[4]`,
    /// `reducer_b`, `p_norm`).
    pub hybrid_head: bool,

    /// EX-2 follow-up²: per-sample α head. Replaces the bake-level
    /// scalar α_logit with a learned per-pair α via `α(x) =
    /// sigmoid(W_α · h + b_α)`. Lets photo-like inputs pull α toward
    /// the rank head while JND-step-grid inputs pull α toward the
    /// pool head.
    ///
    /// Mutually exclusive with `pool_head` AND `hybrid_head`. The
    /// bake metadata key becomes `zentrain.per_sample_alpha_head`
    /// with payload `[W_α[n_hidden] | b_α | rank_w[n_hidden] |
    /// rank_b | reducer_w[4] | reducer_b | p_norm]`. Architectural
    /// cost: +(n_hidden + 1) weights vs scalar-α hybrid.
    ///
    /// **What composes (v0 wiring, 2026-05-18):**
    /// - RankNet pair loss + Adam + cosine LR + early-stop
    /// - `--minibatch-size K` (sequential gradient accumulation)
    /// - NiN composition (per V_22-LARGE recipe)
    /// - `--pwrc-pair-weight` + `--pwrc-sensory-threshold` +
    ///   `--pwrc-band-weights`
    /// - L2 (`--l2`) on layer-1 + rank_w + reducer_w + W_α
    ///   (b_α unregularized)
    /// - Low/Mid/High-q row boosts
    ///
    /// **What is omitted (v0):**
    /// - TV regularizer (skipped on per-sample α path; the V_22
    ///   recipe doesn't use TV; reactivating requires extending
    ///   the per-sample backprop to the TV gradient path).
    /// - Parallel-batch flag (sequential mini-batch only).
    pub per_sample_alpha_head: bool,

    /// Add a learned 372→1 linear skip connection alongside the MLP.
    /// The final output is `y_mlp + w_skip · x + b_skip`. This lets
    /// features with a direct linear relationship to quality bypass
    /// the hidden-layer bottleneck. ~375 extra params. Composes with
    /// any head type. Default `false`.
    pub skip_connection: bool,

    /// Number of hidden layers. Default 1 (372→128→heads). Setting
    /// to 2 adds a second hidden layer: 372→128→64→heads. The
    /// second layer uses the same LeakyReLU activation. The
    /// n_hidden field controls the FIRST hidden layer width; the
    /// second is always n_hidden/2 (clamped to ≥8).
    pub n_hidden_layers: usize,

    /// MSE-target weight (`PreviewV0_5Tuner` experiment, 2026-05-18).
    ///
    /// When `> 0`, an auxiliary per-prediction MSE loss
    /// `(y_i - target_i)^2` is added on top of the RankNet pair loss.
    /// Composes with the per-sample-α head path. The trainer treats
    /// `human_score` as the regression target (not as a rank label
    /// only). Setting `--ranknet-weight 0 --mse-weight 1` runs as
    /// pure MSE; mixing both (e.g. `--ranknet-weight 0.5
    /// --mse-weight 0.5`) hybridizes rank-honesty with calibration
    /// honesty.
    ///
    /// Gradient: `dL/dy_i = 2 * mse_weight * (y_i - target_i) / N`
    /// where `N = 2 * pairs_per_step` (one MSE term per ya AND yb).
    /// Flows through the per-sample-α head's `(rank_w, reducer_w,
    /// W_α, b_α, w1, b1)` backprop in the same scatter as the
    /// RankNet ∂L/∂y. Active ONLY on the
    /// `per_sample_alpha_head = true` path; trainer panics if set
    /// on other heads.
    pub mse_weight: f64,

    /// When true AND groups have `metric_sigmas`, per-pair MSE loss
    /// is weighted by `1 / max(σ, 0.05)²` — directly optimizing Z-RMSE
    /// (Mohammadi 2025 Eq. 6). Errors on high-consensus stimuli (where
    /// cvvdp, iwssim, and ssim2 agree) are penalized more.
    pub sigma_weighted_mse: bool,

    /// STRATEGY-2026-07-02: EMA of weights. `0.0` = off. Typical 0.999.
    /// Maintains exponential moving averages of every weight tensor,
    /// updated after each Adam step; the best-epoch bake snapshots the
    /// EMA copies instead of the live weights (variance reduction —
    /// attacks the measured seed instability: v52 CID22 0.715..0.851).
    pub ema_decay: f64,

    /// STRATEGY-2026-07-02: hard-pair mining for RankNet. With
    /// probability `hard_pair_frac`, the second row of a pair is
    /// re-drawn (≤16 tries) until |Δtarget| ≤ `hard_pair_max_delta` —
    /// concentrating gradient on near-threshold pairs (the measured
    /// KonJND / HQ-zone weakness). `0.0` = off.
    pub hard_pair_frac: f64,
    /// |Δ human_score| ceiling for a "hard" pair (native target units).
    pub hard_pair_max_delta: f64,

    /// STRATEGY-2026-07-02: stratified band sampling. When > 0, each
    /// group's rows are pre-bucketed into this many target-quantile
    /// bands; row A of each pair samples band-uniform then row-uniform
    /// within band (kills band starvation under pooled sampling).
    /// Ignored for groups with per-row CDFs. `0` = off.
    pub stratified_bands: usize,

    /// STRATEGY-2026-07-02: GroupDRO-style worst-group emphasis.
    /// Per-epoch, group sampling weights become
    /// `train_w · exp(dro_eta · normalized_group_loss)` (multiplicative
    /// weights on the observed per-group mean loss). `0.0` = off.
    pub dro_eta: f64,

    /// STRATEGY-2026-07-02: ListMLE (Plackett–Luce NLL) listwise loss
    /// over within-`ref_basename` lists. `0.0` = off.
    pub listwise_weight: f64,
    /// Rows per sampled list (default 8).
    pub listwise_size: usize,
    /// Fraction of steps that run a listwise step instead of a pair
    /// step (default 0.15 when listwise_weight > 0).
    pub listwise_frac: f64,

    /// STRATEGY-2026-07-02: ordered-probit triplet NLL on raw human
    /// triplet responses (KonFiG/AIC-3 lineage: pivot = pristine,
    /// response = which side is MORE DISTORTED, or notsure). `0.0` = off.
    pub triplet_weight: f64,
    /// Fraction of steps that run a triplet step (default 0.2 when active).
    pub triplet_frac: f64,
    /// Indecision threshold τ in model-score units (default 0.6).
    pub triplet_tau: f64,
    /// Observer noise σ in model-score units (default 1.0; the model
    /// output is tanh-pinned ≈[0,100]-scaled, so τ/σ are in that space).
    pub triplet_sigma: f64,

    /// RankNet pair-loss weight (`PreviewV0_5Tuner` experiment,
    /// 2026-05-18). Default `1.0` matches legacy behavior. Setting
    /// to `0.0` disables RankNet entirely — use with
    /// `--mse-weight > 0` for pure-MSE training.
    pub ranknet_weight: f64,

    /// Monotonicity penalty weight (`PreviewV0_5Tuner` experiment,
    /// 2026-05-18). When `> 0`, adds a per-pair quadratic-hinge
    /// loss `(max(0, target_violation))^2` that penalizes the model
    /// when its predicted ordering disagrees with the target's
    /// ordering. For a pair `(xa, xb)` drawn from the same
    /// `ref_basename` group with `target_a > target_b`:
    ///
    /// ```text
    ///   violation = y_b - y_a + margin
    ///   loss_mono = monotonicity_reg * max(0, violation)^2
    ///   ∂loss/∂y_a = -2 * monotonicity_reg * max(0, violation)
    ///   ∂loss/∂y_b = +2 * monotonicity_reg * max(0, violation)
    /// ```
    ///
    /// `margin = 0.0` by default (active iff strict inversion).
    /// Because pairs are drawn from per-image curves, "target_a >
    /// target_b" includes the q↑→quality↑ structure that codec
    /// auto-targeting depends on. Effective only on the
    /// `per_sample_alpha_head = true` path.
    pub monotonicity_reg: f64,

    /// **Correct-by-construction monotone mode.** When `true` (only on
    /// the `per_sample_alpha_head` path), after every Adam step the
    /// trainer PROJECTS weights to a sign pattern that makes the bake
    /// monotone-by-construction in every (non-negative dissimilarity)
    /// feature:
    ///   - encoder `w1`, `w2_enc` → clamped `≥ 0` (LeakyReLU monotone ⟹
    ///     hidden activations monotone↑ in each feature),
    ///   - `rank_w`, `w_skip` → clamped `≤ 0` (so `y_rank` is monotone↓
    ///     in distortion),
    ///   - the per-sample-α gate is forced to ≈1 (`w_alpha = 0`,
    ///     `b_alpha = 30`) so `y = y_rank` (the α-gated mix of two
    ///     functions is not monotone, so a single head is required).
    ///
    /// Combined with the increasing tanh pin `100·σ(y/scale)`, the score
    /// is bounded `[0,100]` AND monotone non-increasing in distortion on
    /// the ENTIRE input domain — the G1+G3 codec goals by construction.
    /// Requires `tanh_output_head_scale > 0`; pairs best with
    /// `skip_connection = false`. Default `false`.
    pub monotone_cbc: bool,

    /// Per-feature sign mask for `monotone_cbc` (length `n_features`).
    /// `Some(pin)` where `pin[j] == true` means W1's column for input
    /// feature `j` is constrained `≥ 0`; `false` means feature `j` is
    /// NOT sign-safe (it flips correlation with distortion across
    /// distortion types — see `benchmarks/feature_sign_mask_2026-05-26.tsv`)
    /// and is left free (partial) or dropped (strict). `None` → all
    /// features pinned `≥ 0` (the original whole-encoder behavior, which
    /// collapses the dial because it mis-constrains the ~72 sign-flip
    /// features). Only consulted when `monotone_cbc = true`.
    pub monotone_feature_pin: Option<Vec<bool>>,

    /// When `true` AND a `monotone_feature_pin` is set, the non-pinned
    /// (sign-flip) features are DROPPED — their W1 columns are driven to
    /// 0 (soft penalty toward 0 during training + zeroed at bake) so the
    /// shipped bake is STRICTLY monotone (depends only on sign-safe
    /// features). When `false` (partial), the non-pinned features stay
    /// free — the bake keeps their signal but is monotone only in the
    /// sign-safe subset (no strict guarantee). Only consulted when
    /// `monotone_cbc = true` and `monotone_feature_pin` is `Some`.
    pub monotone_strict: bool,

    /// "Soft-monotone-keep-72" mode (#39 followup #2, 2026-05-28). When
    /// `true` AND `monotone_cbc = true` AND `monotone_feature_pin = Some`:
    ///   - The PINNED features' W1 columns are HARD-projected to `≥ 0`
    ///     after every Adam step (matching the final bake projection
    ///     exactly — no train/bake sign drift).
    ///   - The UNPINNED ("sign-flip") features stay FREE throughout
    ///     training and bake (no soft penalty, no projection).
    ///   - Rank head still projected `≤ 0` at bake; encoder ≥0 + α≡1
    ///     unchanged.
    ///
    /// Replicates the MVP-Python `scipy.optimize.lsq_linear` BVLS
    /// behavior: bounds enforced THROUGHOUT optimization (not via soft
    /// penalty), unpinned features kept free. MVP measured CID22 0.824
    /// SROCC; the Rust soft-only `monotone_strict=true` mode dropped
    /// the same 72 features and got 0.658, soft-only partial got 0.614.
    ///
    /// **Orthogonal to `monotone_strict`.** When this flag is `true`
    /// the value of `monotone_strict` is IGNORED (the 72 unpinned stay
    /// free regardless). The flag is intentionally orthogonal so callers
    /// can layer it onto existing recipes without changing
    /// `monotone_strict`'s docs / defaults.
    ///
    /// Why this isn't "per-step hard clamp" (which v45 falsified): the
    /// v45 collapse came from clamping ALL weights — including sign-flip
    /// features whose information lives in the negative half-plane.
    /// Here we only hard-project the 300 sign-safe features (which the
    /// soft penalty is already trying to keep ≥0), and leave the 72
    /// sign-flip features untouched. The model retains full expressivity
    /// on the 72; only the 300 are constrained.
    ///
    /// Only consulted when `monotone_cbc = true` and
    /// `monotone_feature_pin = Some`. Default `false`.
    pub monotone_pin_during_training: bool,

    /// **Quantization-aware fine-tune (QAT).** When `> 0`, the LAST
    /// `qat_fine_tune_epochs` epochs train with a straight-through
    /// estimator: the forward uses f16-rounded + zerobiased weights while
    /// Adam updates the f32 master. The network learns weights robust to
    /// the f16+zerobias packing, so the shipped packed bake == the
    /// validated network (no post-hoc identity/dial surprise). 0 = off
    /// (default; production-safe — pure f32 as before).
    pub qat_fine_tune_epochs: usize,
    /// Zerobias threshold during QAT (relative to per-layer max, matching
    /// the bake-time `apply_zero_bias_per_layer`). Default 0.005.
    pub qat_tau: f64,

    /// Per-epoch group-eval row cap (0 = full, historical behavior). When
    /// set greater than 0, oversized groups' per-epoch diagnostics/selection
    /// forwards run on a deterministic stride sample of at most this many
    /// rows — pre-gathered once, no RNG, training byte-stream untouched. The
    /// big iteration-speed lever for multi-million-row groups (v51: 3.4M
    /// forwards/epoch → ~50k). Selection SROCC on 50k rows is within
    /// ±0.005 of full. Recorded in manifests as `group_eval_cap`.
    pub group_eval_cap: usize,

    /// Margin for the monotonicity-reg hinge (default 0.0).
    /// `target_a - target_b > monotonicity_margin` activates the
    /// penalty; otherwise the pair contributes 0. Useful when you
    /// want only "near-tied" pairs to participate.
    pub monotonicity_margin: f64,

    /// `PreviewV0_5TunerV2` cross-codec JND anchor loss weight
    /// (2026-05-19). When `> 0`, an additional anchor MSE loss is
    /// applied each pair-step with probability ~`anchor_step_p`:
    /// a single row is drawn from the anchor pool, forwarded through
    /// the per-sample-α head, and `w · (y - anchor_target_score)²` is
    /// added to the loss. Gradients flow through the same per-sample-α
    /// backprop as the MSE pair loss.
    ///
    /// The anchor pool is supplied at the call boundary via the
    /// `AnchorRows` argument to `train_mlp_with_tv_anchored`. Rows
    /// carry per-row weights so KonJND PJND anchors (real human data)
    /// can be weighted higher than synthetic ssim2-derived anchors.
    /// Default `0.0` = no anchor step. Only wired on the
    /// `per_sample_alpha_head = true` path.
    pub anchor_loss_weight: f64,

    /// `PreviewV0_5TunerV2` target score each anchor row regresses to
    /// (2026-05-19). Default `63.0` matches CID22 paper Table 4 PJND
    /// calibration. Per-row anchor weights multiply this target's MSE
    /// contribution; the target itself is shared across rows.
    pub anchor_target_score: f64,

    /// Fraction of pair-steps that get accompanied by an extra anchor
    /// step (default `0.10` = 10% of steps). The anchor step samples
    /// one row, forwards, and applies anchor MSE backprop. Higher
    /// values give the anchor more gradient bandwidth at the cost of
    /// rank-loss signal. Only effective when `anchor_loss_weight > 0`.
    pub anchor_step_p: f64,

    /// Cross-codec equivalence pair loss weight (EXP-CROSS-CODEC-METRIC,
    /// 2026-05-19). When `> 0`, an additional shape-free equivalence
    /// loss is applied each pair-step with probability `equiv_step_p`:
    /// a single pair `(features_a, features_b)` is drawn from the
    /// `EquivPairs` pool, both vectors are forwarded through the
    /// per-sample-α head, and `w · (y_a − y_b)²` is added to the loss.
    /// Gradients flow through both forward passes and accumulate into
    /// a single Adam step.
    ///
    /// The pair pool is supplied via the `EquivPairs` argument to
    /// `train_mlp_with_tv_anchored_equiv`. Default `0.0` = no equiv
    /// step. Only wired on the `per_sample_alpha_head = true` path.
    pub cross_codec_eq_weight: f64,

    /// Fraction of pair-steps that get an extra equivalence step
    /// (default `0.10` = 10% of steps). Same semantics as
    /// `anchor_step_p` but for the equivalence pool. Only effective
    /// when `cross_codec_eq_weight > 0`.
    pub cross_codec_eq_step_p: f64,

    /// EXP-CROSS-CODEC-V3 (2026-05-19) rank-preserve regularizer
    /// applied on equivalence pairs that have a non-zero
    /// `butter_diff`. The cross-codec-eq MSE term `(y_a − y_b)²`
    /// can collapse both outputs to a point mass when its weight is
    /// loud enough to bind; rank-preserve pushes back proportionally
    /// to `|butter_diff|` (the pivot-metric quality gap) so the
    /// network can't satisfy the equivalence loss by predicting the
    /// same value for both sides.
    ///
    /// Loss formulation (RankNet-style sigmoid):
    /// `L_rp = w · |Δb| · −log(sigmoid(sign(Δb) · (y_a − y_b)))` where
    /// `Δb = butter_a − butter_b`. Because butter is LOWER for higher
    /// quality, `Δb > 0` means A is WORSE than B → we want `y_a > y_b`
    /// (in distance-shape) OR `y_a < y_b` (in score-shape) depending
    /// on training target. The orientation convention here uses the
    /// per-sample-α head's own output sign-convention: the trainer
    /// uses MSE against `mix_cv40_iw60` (score-shape, HIGHER = better
    /// quality), so for `Δb > 0` (A worse) we want `y_a < y_b`. Hence
    /// `sign(butter_diff) × (y_b − y_a)` is the score-shape rank-
    /// preserve target; the implementation uses
    /// `sign(Δb) × (y_b − y_a)` as the RankNet logit.
    ///
    /// Default `0.0` = off. Only wired on the per-sample-α head and
    /// only effective when `cross_codec_eq_weight > 0` AND the equiv
    /// pool has `butter_diff` populated.
    pub cross_codec_rank_preserve_weight: f64,

    /// EXP-CROSS-CODEC-V3 (2026-05-19) dynamic-range floor regularizer.
    /// When `> 0`, every pair-step with probability
    /// `dynamic_range_step_p` triggers a "q-sweep probe": sample
    /// `dynamic_range_probe_n` random feature vectors from the equiv
    /// pool's A-side (a proxy for the per-image q-sweep span), forward
    /// each through the per-sample-α head, compute the observed σ
    /// across the outputs, and penalize the network when
    /// `σ_obs < dynamic_range_sigma_threshold`:
    ///
    /// `L_dr = w · max(0, σ_threshold − σ_obs)²`
    ///
    /// The gradient propagates through each forward as
    /// `dL/dy_i = −2 · w · max(0, σ_threshold − σ_obs) · (y_i − μ) / (σ · N)`
    /// where `μ` and `σ` are the batch mean and standard deviation.
    /// All `N` per-row gradients accumulate into a single Adam step.
    ///
    /// This directly addresses the cc4v2 collapse failure mode where
    /// the network minimized the cross-codec-eq + anchor losses by
    /// predicting a constant ~63 score for every input. With a σ-floor
    /// of e.g. 15, the network must spread its outputs by at least
    /// that much across the probe to escape the penalty — which forces
    /// it to retain dynamic range across quality levels.
    ///
    /// Default `0.0` = off. Only wired on the per-sample-α head and
    /// requires equiv pool data (uses equiv-pool A-side as the probe).
    pub dynamic_range_floor_weight: f64,

    /// Target σ across the q-sweep probe outputs (default `15.0`
    /// score units). Probes whose σ ≥ this contribute no penalty;
    /// probes below incur quadratic penalty per
    /// `dynamic_range_floor_weight`. Only effective when
    /// `dynamic_range_floor_weight > 0`.
    pub dynamic_range_sigma_threshold: f64,

    /// Fraction of pair-steps that trigger a q-sweep probe
    /// (default `0.05` = 5%, ~2500 probes per epoch at
    /// `--pairs-per-epoch 50000`). Higher values give the
    /// range-floor more gradient bandwidth at the cost of pair-loss
    /// signal. Only effective when `dynamic_range_floor_weight > 0`.
    pub dynamic_range_step_p: f64,

    /// Number of feature vectors sampled per q-sweep probe
    /// (default `40` = 8 refs × 5 q values conceptually). Affects
    /// the σ estimator's variance — too few rows and σ becomes
    /// noisy; too many and probe cost dominates per-step time.
    pub dynamic_range_probe_n: usize,

    /// EXP-CROSS-CODEC-V4 tanh-pinned [0, 100] output head scale
    /// (2026-05-19). When `> 0`, wraps the per-sample-α head's raw
    /// output `y_pre = α·y_rank + (1−α)·y_pool` in a sigmoid pin:
    ///
    /// `y_score = 100 · σ(y_pre / scale)`
    ///
    /// where `scale` is this value (recommended `10.0` so the active
    /// linear region spans `y_pre ∈ [−30, 30]` mapping to roughly
    /// `[5, 95]` score units, with saturation past that).
    ///
    /// Backprop: `dL/dy_pre = dL/dy_score · (100/scale) · σ' = (100/scale)
    /// · σ(y_pre/scale) · (1 − σ(y_pre/scale))`. The chain factor (100 ·
    /// σ' / scale) is multiplied into every per-pair upstream gradient
    /// computed for the per-sample-α head (RankNet, MSE, monotonicity,
    /// anchor, cross-codec-eq, rank-preserve, range-floor probes).
    ///
    /// Bake-side: a `zentrain.tanh_output_head` metadata entry is
    /// emitted with payload `[scale: f32]` (`u32 LE`); the runtime
    /// recognizes the key and applies the matching sigmoid pin.
    ///
    /// Default `0.0` = off (legacy linear output, requires affine
    /// post-hoc calibration to reach [0, 100]). Only wired on the
    /// per-sample-α head. V3 falsification documented this as the
    /// dominant mono-violation cause (post-affine β amplifies per-pair
    /// jitter); pinning the output at training time eliminates the
    /// β-amplification path entirely.
    pub tanh_output_head_scale: f64,

    /// EXP-V11-D-PJND-DOMINANT (2026-05-20, task #198) KonJND-PJND
    /// passthrough anchor weight. Wires a SECOND anchor pool alongside
    /// the existing `anchor_loss_weight` cross-codec-eq anchor. The pool
    /// is drawn from `train/konjnd-dense.parquet`; every sampled row
    /// regresses to `pjnd_passthrough_target_score` via
    /// `w · (y_score − target)²`. Per-row weight is constant 1.0
    /// (no per-row weighting on the KonJND PJND pool).
    ///
    /// Tested hypothesis: a PJND anchor passthrough at weight >>
    /// `cross_codec_eq_weight` can break the cross-codec-eq + KonJND
    /// bistability so that some w-band lets cc_eq fire without
    /// catastrophic KonJND collapse. Either it breaks the basin
    /// (KonJND survives) or it's overrun by cc_eq (KonJND still
    /// collapses), and the result closes V11 with structural finality.
    ///
    /// Default `0.0` = off. Only wired on the per-sample-α head.
    pub pjnd_passthrough_weight: f64,

    /// EXP-V11-D-PJND-DOMINANT probability per pair-step that the
    /// PJND-passthrough anchor fires. Default `0.30`. Higher values
    /// give the PJND anchor more gradient bandwidth at the cost of
    /// pair-loss + cross-codec-eq bandwidth. Only effective when
    /// `pjnd_passthrough_weight > 0`.
    pub pjnd_passthrough_step_p: f64,

    /// EXP-V11-D-PJND-DOMINANT constant target score that every
    /// PJND-passthrough row regresses to. Default `80.0` matches the
    /// V10 PJND-anchored calibration (V10 maps PJND ssim2≈63 to
    /// score=80). Only effective when `pjnd_passthrough_weight > 0`.
    pub pjnd_passthrough_target_score: f64,

    /// KONJND-AGGREGATION-HEAD (2026-05-24, task #4) — per-source
    /// aggregation training-time loss for konjnd-dense.
    ///
    /// Structurally different from `pjnd_passthrough_weight`: rather
    /// than regressing each row's prediction against a (per-row or
    /// constant) target, pools predictions across a ref's distortion
    /// levels BEFORE computing the loss:
    ///
    /// ```text
    /// For each gradient step:
    ///   Sample K refs from konjnd-dense pool.
    ///   For each ref r:
    ///     Sample S rows from r's distortion-level pool.
    ///     Forward S × MLP passes → y_{r,1}..y_{r,S}.
    ///     agg_r = (1/S) · Σ y_{r,i}
    ///   Loss = w · Σ_r (agg_r − pjnd_target_r)²
    ///   Backprop: ∂L/∂y_{r,i} = (2w/S) · (agg_r − pjnd_target_r)
    /// ```
    ///
    /// Critical property: the within-ref residual `(agg − t)` is
    /// non-zero in general, so gradient flows uniformly to all S rows
    /// — fixing the zero-gradient pathology that pjnd_passthrough hit
    /// at V11-D (per-pair MSE against per-source-constant target
    /// produces no useful learning signal; recovery_phase3b root
    /// cause).
    ///
    /// Runtime unchanged — this is purely a training-time augmentation.
    /// The bake's network is a normal per-sample MLP; aggregation only
    /// constrains the *mean* across the 20 distortion levels per ref.
    /// No new bake metadata key.
    ///
    /// Default `0.0` = off. Only wired on the per-sample-α head.
    /// Implementation gated on
    /// [`Self::konjnd_aggregation_parquet`] being supplied.
    /// See `docs/KONJND_AGGREGATION_HEAD_DESIGN_2026-05-24.md`.
    pub konjnd_aggregation_weight: f64,

    /// KONJND-AGGREGATION-HEAD step probability — every pair-step,
    /// fire an aggregation step with this probability (alongside the
    /// primary RankNet/MSE step). Default `0.30`. Only effective when
    /// `konjnd_aggregation_weight > 0`.
    pub konjnd_aggregation_step_p: f64,

    /// KONJND-AGGREGATION-HEAD: number of rows to sample per ref per
    /// aggregation step (the `S` parameter). The aggregate is the mean
    /// over these S predictions; variance ≈ σ²(y_{r,*})/S. Default `5`.
    /// Higher = better aggregate estimate but more compute per step.
    /// Only effective when `konjnd_aggregation_weight > 0`.
    pub konjnd_aggregation_samples_per_ref: usize,

    /// KONJND-AGGREGATION-HEAD: number of refs to sample per
    /// aggregation step (the `K` parameter). Total forwards per step =
    /// K × S. Default `8`. Only effective when
    /// `konjnd_aggregation_weight > 0`.
    pub konjnd_aggregation_refs_per_step: usize,
}

impl Default for MlpHyperparams {
    fn default() -> Self {
        Self {
            n_hidden: 32,
            n_epochs: 200,
            pairs_per_epoch: 50_000,
            initial_lr: 0.001,
            leaky_alpha: 0.01,
            seed: 42,
            init_seed: None,
            sample_seed: None,
            log_every: 10,
            l2_lambda: 1e-5,
            early_stop_patience: 50,
            validation_policy: ValidationPolicy::Min,
            val_aggregate: crate::panel::ValAggregate::GeomeanSPP,
            low_q_boost: 1.0,
            mid_q_boost: 1.0,
            high_q_boost: 1.0,
            out_dtype: WeightDtype::F32,
            feature_transforms: None,
            feature_transform_params: None,
            minibatch_size: 1,
            parallel_batch: true,
            pwrc_pair_weight: false,
            pwrc_sensory_threshold: 0.0,
            pwrc_band_weights: None,
            norm_in_norm_weight: 0.0,
            norm_in_norm_p: 1.0,
            norm_in_norm_q: 2.0,
            pool_head: false,
            hybrid_head: false,
            per_sample_alpha_head: false,
            skip_connection: false,
            n_hidden_layers: 1,
            dump_checkpoints_every: 0,
            dump_checkpoints_dir: None,
            mse_weight: 0.0,
            sigma_weighted_mse: false,
            ema_decay: 0.0,
            hard_pair_frac: 0.0,
            hard_pair_max_delta: 0.05,
            stratified_bands: 0,
            dro_eta: 0.0,
            listwise_weight: 0.0,
            listwise_size: 8,
            listwise_frac: 0.15,
            triplet_weight: 0.0,
            triplet_frac: 0.2,
            triplet_tau: 0.6,
            triplet_sigma: 1.0,
            ranknet_weight: 1.0,
            monotonicity_reg: 0.0,
            monotone_cbc: false,
            monotone_feature_pin: None,
            monotone_pin_during_training: false,
            monotone_strict: false,
            qat_fine_tune_epochs: 0,
            qat_tau: 0.005,
            group_eval_cap: 0,
            monotonicity_margin: 0.0,
            anchor_loss_weight: 0.0,
            anchor_target_score: 63.0,
            anchor_step_p: 0.10,
            cross_codec_eq_weight: 0.0,
            cross_codec_eq_step_p: 0.10,
            cross_codec_rank_preserve_weight: 0.0,
            dynamic_range_floor_weight: 0.0,
            dynamic_range_sigma_threshold: 15.0,
            dynamic_range_step_p: 0.05,
            dynamic_range_probe_n: 40,
            tanh_output_head_scale: 0.0,
            pjnd_passthrough_weight: 0.0,
            pjnd_passthrough_step_p: 0.30,
            pjnd_passthrough_target_score: 80.0,
            konjnd_aggregation_weight: 0.0,
            konjnd_aggregation_step_p: 0.30,
            konjnd_aggregation_samples_per_ref: 5,
            konjnd_aggregation_refs_per_step: 8,
        }
    }
}

/// Compute the per-pair PWRC loss weight given the two human scores
/// (in `score_zensim` 0..100 units) and an optional band-weight
/// vector.
///
/// **Closed-form default (`band_weights == None`)**: `exp(max(a, b) /
/// 100)` — the label-only term from Wu et al. 2018 `PWRC.m` after
/// dropping the prediction-error `diff` piece (see `MlpHyperparams::
/// pwrc_pair_weight` docs for the full derivation). Range:
/// `[exp(0), exp(1)] = [1.0, 2.718]`.
///
/// **Band-weight override**: `band_weights[clamp(floor(max(a,b) /
/// band_width), 0, n-1)]` where `band_width = 100.0 / n`. For a
/// 10-element vector each entry covers 10 score units: index 0 =
/// [0, 10), index 9 = [90, 100].
///
/// Always returns a finite, positive weight. NaN-safe: clamps inputs
/// into `[0, 100]` first so the band index can't index OOB.
/// Best validation score of the LAST completed training run in this process —
/// read by the trainer bin for spec.json + the embedded `zentrain.repro` (the
/// bimodal-seed campaigns select seeds by internal val; a Mutex global instead
/// of return-type churn across the five train variants). `None` until a
/// variant reports (the GPU path currently does not).
pub static LAST_BEST_VAL: std::sync::Mutex<Option<f64>> = std::sync::Mutex::new(None);

/// Per-INPUT-FEATURE L2 multiplier on layer-1 rows (len = n_features), set by
/// the trainer bin before training. The scale-mass regularizer: coarse-scale
/// inputs get mult > 1 so the optimizer prefers fine-scale reliance — the
/// ModelSensitivity fold blends per-pixel maps by gradient mass, and coarse
/// concentration collapses the map to 1/8-resolution (E-M6, 2026-07-29:
/// data-mix-driven basic-scale mass {1,8,30,59}% -> M3 0.11-0.25 while
/// M2=0.99). `None` = uniform L2 (bit-identical legacy behavior).
pub static L2_FEATURE_MULT: std::sync::Mutex<Option<std::sync::Arc<Vec<f64>>>> =
    std::sync::Mutex::new(None);

fn l2_feature_mult() -> Option<std::sync::Arc<Vec<f64>>> {
    L2_FEATURE_MULT.lock().ok().and_then(|g| g.clone())
}

/// Add the layer-1 L2 gradient `scale * mult[feature] * w` into `g`,
/// row by row.
///
/// **Why this is not just the obvious loop.** The obvious loop —
/// `for (idx, (g, &w)) in g.iter_mut().zip(w.iter()).enumerate() { *g +=
/// scale * mult[idx / n_hidden] * w }` — emits a 32-bit integer DIVIDE per
/// weight. At 944×128 weights, applied once per pair, once per pair-draw,
/// that is ~6 × 10⁹ divides per 50 k-pair epoch, and a 2026-08-04 `perf`
/// profile of a live SOTA-944 training run measured **43 % of total trainer
/// cycles inside that one loop** — more than Adam (26 %) and more than the
/// AVX-512 forward + backprop combined (21 %). The divide is pure
/// index arithmetic: walking `w1` as `n_hidden`-wide ROWS (which is exactly
/// its row-major layout) removes it and hoists the per-feature multiplier
/// out of the inner loop.
///
/// **Bit-identical by construction.** Rust evaluates `scale * m * w` as
/// `(scale * m) * w`, so hoisting `sm = scale * m` per row preserves the
/// exact association and rounding; the inner `*g += sm * w` is elementwise
/// (no cross-lane reduction), so even if LLVM vectorizes it the result is
/// unchanged. Gated by `l2_row_form_matches_divided_index_form_bitwise`.
///
/// The trap this loop is in: `--coarse-decay` alone (no `--coarse-l2-mult`)
/// sets `L2_FEATURE_MULT` to `Some(2.0 on the coarse rows)` purely so the
/// DECOUPLED decay gate `m > 1` engages — and thereby switches this loop
/// onto the divide path, even though the coupled L2 multiplier is documented
/// two functions up as "neutralized by Adam". Every campaign recipe that
/// passes `--coarse-decay` paid for it.
fn add_l2_grad_layer1(g: &mut [f64], w: &[f64], scale: f64, n_hidden: usize, mult: Option<&[f64]>) {
    let n = g.len().min(w.len());
    let (g, w) = (&mut g[..n], &w[..n]);
    let Some(mult) = mult else {
        for (g, &w) in g.iter_mut().zip(w.iter()) {
            *g += scale * w;
        }
        return;
    };
    if n_hidden == 0 {
        // Degenerate; the divided-index form would panic on `idx / 0`, so
        // preserve that rather than inventing behaviour.
        for (idx, (g, &w)) in g.iter_mut().zip(w.iter()).enumerate() {
            *g += scale * mult[idx / n_hidden] * w;
        }
        return;
    }
    for (feat, (grow, wrow)) in g.chunks_mut(n_hidden).zip(w.chunks(n_hidden)).enumerate() {
        // Indexing `mult[feat]` panics on the same feature index the
        // divided form would have.
        let sm = scale * mult[feat];
        for (g, &w) in grow.iter_mut().zip(wrow.iter()) {
            *g += sm * w;
        }
    }
}

/// Decoupled (AdamW-style) per-feature weight decay on layer-1 rows, applied
/// AFTER the Adam step. Coupled L2-via-gradient is neutralized by Adam's
/// per-parameter rescaling (measured 2026-07-29: mult 8-32 on lambda 1e-5
/// produced byte-different but functionally IDENTICAL bakes); decoupling
/// bypasses the rescale so the coarse-scale pressure actually binds.
/// `mult` rows (the coarse-scale features) decay by `lr * rate * mult[i]`;
/// rows at mult 1.0 are left untouched (rate applies only where mult > 1).
fn apply_coarse_decay(w1: &mut [f64], n_hidden: usize, lr: f64, rate: f64) {
    if rate <= 0.0 {
        return;
    }
    // debug telemetry (ZENSIM_DECAY_DEBUG=1): prove the decay executes + bites.
    // The env lookup is cached: this runs once per Adam step (50 000× per
    // epoch at K=1), and `std::env::var` allocates a String and takes the
    // environment lock every time.
    fn decay_debug_on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("ZENSIM_DECAY_DEBUG").as_deref() == Ok("1"))
    }
    if decay_debug_on() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static CALLS: AtomicU64 = AtomicU64::new(0);
        let c = CALLS.fetch_add(1, Ordering::Relaxed);
        if c.is_multiple_of(500) {
            eprintln!("[decay-debug] call #{c} lr={lr} rate={rate}");
        }
    }
    if let Some(mult) = l2_feature_mult() {
        for (i, &m) in mult.iter().enumerate() {
            if m > 1.0 {
                let f = 1.0 - (lr * rate * m).min(0.5);
                for w in &mut w1[i * n_hidden..(i + 1) * n_hidden] {
                    *w *= f;
                }
            }
        }
    }
}

/// Decay rate for [`apply_coarse_decay`] — set by the bin (0.0 = off).
pub static COARSE_DECAY_RATE: std::sync::Mutex<f64> = std::sync::Mutex::new(0.0);

fn coarse_decay_rate() -> f64 {
    COARSE_DECAY_RATE.lock().map(|g| *g).unwrap_or(0.0)
}

/// Group-lasso (group-L1) strength over layer-1 **input columns** — set by
/// the bin (0.0 = off). See [`apply_group_l1`].
pub static GROUP_L1_LAMBDA: std::sync::Mutex<f64> = std::sync::Mutex::new(0.0);

fn group_l1_lambda() -> f64 {
    GROUP_L1_LAMBDA.lock().map(|g| *g).unwrap_or(0.0)
}

/// Decoupled **group-lasso** (a.k.a. group-L1 / block-ℓ2,1) proximal step on
/// layer-1 input rows, applied AFTER the Adam step.
///
/// The penalty is `λ · Σ_k ‖W1[k, :]‖₂` — the ℓ2 norm of each input's whole
/// outgoing weight row, summed over inputs. Unlike elementwise L1 this drives
/// a *whole input column* to exactly zero, so the fit **learns which inputs to
/// keep** instead of ranking them post-hoc.
///
/// **Why proximal and not a gradient penalty.** A coupled penalty (adding
/// `λ ∂‖W‖/∂W` to the gradient) is neutralized by Adam's per-parameter
/// rescaling — measured on this trainer for L2 (2026-07-29: mult 8-32 on
/// λ=1e-5 produced byte-different but functionally identical bakes), which is
/// why `--coarse-decay` is decoupled. Worse, a coupled subgradient can never
/// reach *exact* zero: it shrinks asymptotically and every column stays alive
/// at 1e-12, so nothing is prunable. The proximal operator has a hard
/// threshold and lands on exact 0.0.
///
/// Block soft-threshold, per input row `w = W1[k, :]` with `τ = lr · λ`:
/// - `‖w‖₂ ≤ τ` ⇒ `w := 0` (every element exactly `0.0`)
/// - else `w := w · (1 − τ/‖w‖₂)`
///
/// Rows already exactly zero stay zero (`‖w‖ = 0 ≤ τ`), so this composes with
/// the [`INPUT_KEEP_MASK`] pinning without fighting it.
fn apply_group_l1(w1: &mut [f64], n_hidden: usize, lr: f64, lambda: f64) {
    if lambda <= 0.0 || n_hidden == 0 {
        return;
    }
    let tau = lr * lambda;
    if tau <= 0.0 {
        return;
    }
    for row in w1.chunks_mut(n_hidden) {
        let norm = row.iter().map(|&w| w * w).sum::<f64>().sqrt();
        if norm <= tau {
            for w in row.iter_mut() {
                *w = 0.0;
            }
        } else {
            let f = 1.0 - tau / norm;
            for w in row.iter_mut() {
                *w *= f;
            }
        }
    }
}

/// Post-Adam decoupled penalties on layer-1 weights, in a fixed order:
/// coarse-scale decay (per-feature, `--coarse-decay`) then the group-lasso
/// proximal step (`--group-l1`). Both are no-ops at their default 0.0, so
/// this is bit-identical to the historical `apply_coarse_decay` call when
/// neither flag is passed.
fn apply_post_adam_penalties(w1: &mut [f64], n_hidden: usize, lr: f64) {
    apply_coarse_decay(w1, n_hidden, lr, coarse_decay_rate());
    apply_group_l1(w1, n_hidden, lr, group_l1_lambda());
}

/// Per-input keep mask (`true` = the input participates) — set by the bin
/// from `--keep-features` (`None` = every input participates).
///
/// The bin zeroes the **raw** feature values of dropped columns before
/// training, which makes their standardized value exactly `0.0` (the scaler
/// sees a constant-zero column: `mean = 0`, `std` floored, `(0−0)/s = 0`).
/// A zero standardized input contributes exactly `0.0` to every layer-1
/// gradient, so once the corresponding `W1` rows are zeroed at init they can
/// never move again (L2 adds `λ·0`, coarse decay scales `0`, the group-lasso
/// prox keeps `0`). Training is therefore *exactly* a K-wide fit, and the
/// baked layer-1 rows are exactly zero ⇒ prunable by `bake_dial_refit pack`.
pub static INPUT_KEEP_MASK: std::sync::Mutex<Option<Vec<bool>>> = std::sync::Mutex::new(None);

fn input_keep_mask() -> Option<Vec<bool>> {
    INPUT_KEEP_MASK.lock().ok().and_then(|g| g.clone())
}

/// Zero the layer-1 rows of every masked-out input, once, right after init.
/// Returns the number of rows zeroed (0 when no mask is set).
///
/// Safe to call exactly once: dropped inputs are standardized-zero, so their
/// gradients are exactly `0.0` for the rest of training and the rows stay at
/// `0.0` without any per-step cost.
fn zero_masked_w1_rows(w1: &mut [f64], n_hidden: usize, mask: Option<&[bool]>) -> usize {
    let Some(mask) = mask else {
        return 0;
    };
    if n_hidden == 0 {
        return 0;
    }
    let mut zeroed = 0usize;
    for (k, row) in w1.chunks_mut(n_hidden).enumerate() {
        if !mask.get(k).copied().unwrap_or(true) {
            for w in row.iter_mut() {
                *w = 0.0;
            }
            zeroed += 1;
        }
    }
    zeroed
}

fn record_best_val(v: f64) {
    if let Ok(mut g) = LAST_BEST_VAL.lock() {
        *g = Some(v);
    }
}

#[inline]
pub fn pwrc_pair_weight(a: f64, b: f64, band_weights: Option<&[f64]>) -> f64 {
    let max_mos = a.max(b).clamp(0.0, 100.0);
    match band_weights {
        Some(v) if !v.is_empty() => {
            let band_width = 100.0 / v.len() as f64;
            // floor; cap at len-1 (so MOS == 100.0 lands in the last bin).
            let raw_idx = (max_mos / band_width).floor() as isize;
            let idx = raw_idx.clamp(0, v.len() as isize - 1) as usize;
            v[idx]
        }
        // Closed-form Wu 2018 (label-only, see fn docstring).
        _ => (max_mos / 100.0).exp(),
    }
}

/// The RAW (unstandardized) feature rows of a [`TrainingGroup`], in one of
/// two ownership shapes.
///
/// **Why two.** Raw features are read exactly twice per run — by
/// [`compute_scaler_from_groups`], then by
/// [`standardize_groups_releasing_raw`]. After that they are dead:
/// everything downstream reads the standardized buffer, and of the raw
/// table only [`len`] (the row count the pair sampler draws against) is
/// ever consulted again.
///
/// At production scale that dead copy is half the trainer's memory. Wave 10
/// loads 779,290 rows × 944 features: the raw rows are 5.91 GB, the
/// standardized copy another 5.89 GB, together the whole of a lane's
/// ~11.9 GB RSS — which capped this 28-core box at 3 concurrent trainers
/// while it sat CPU-idle. [`FeatureRows::Releasable`] hands the trainer the
/// raw buffer ITSELF: standardization takes it and transforms it in place,
/// so the run never materializes a second copy at all.
///
/// The shape is one flat row-major buffer per group, not per-row `Vec`s,
/// and that is load-bearing: a first version of this optimization freed
/// each row `Vec` after copying it out, which emptied the Vecs but moved
/// peak RSS by only ~0.4 GB on the full wave-10 recipe — the ~7.5 KB row
/// chunks were interleaved across glibc arenas by the loaders and became
/// interior free-list holes that never return to the OS, while the
/// standardized buffers were fresh mmaps that could not reuse them. One
/// flat buffer per group is a single large mapping, and taking it in place
/// means there is nothing to give back. Method + full-data bit-identity
/// gate: `benchmarks/trainer_mem_release_2026-08-04.md`
/// (accounting: `benchmarks/trainer_perf_2026-08-04.md` §2).
///
/// [`len`]: FeatureRows::len
#[derive(Debug)]
pub enum FeatureRows<'a> {
    /// A read-only row table. The trainer copies out of it and leaves it
    /// intact, so the caller may reuse the rows afterwards (train twice off
    /// one dataset, assert on the inputs, …). Nothing is consumed — use this
    /// when the rows are small or still needed.
    Borrowed(&'a [&'a [f64]]),
    /// One flat row-major buffer (`n_rows × n_features`) the trainer MAY
    /// TAKE during standardization.
    ///
    /// [`standardize_groups_releasing_raw`] moves the buffer out
    /// (`std::mem::take`) and standardizes it in place — same expression,
    /// same element order, so the bake is bit-identical to the
    /// [`FeatureRows::Borrowed`] copy path. **Treat the buffer as
    /// moved-from once a `train_mlp*` call returns.** [`FeatureRows::len`]
    /// reports `n_rows` (cached here, not derived from the buffer), so the
    /// hot loop's `g.features.len()` stays correct after the take;
    /// [`FeatureRows::row`] is only valid BEFORE standardization.
    Releasable {
        /// Flat row-major values; `data.len() == n_rows * n_features`
        /// until taken, `0` after.
        data: &'a mut Vec<f64>,
        n_rows: usize,
        n_features: usize,
    },
}

impl<'a> FeatureRows<'a> {
    /// Number of rows. Cached at construction for [`Self::Releasable`], so
    /// it stays correct after standardization takes the buffer.
    pub fn len(&self) -> usize {
        match self {
            Self::Borrowed(r) => r.len(),
            Self::Releasable { n_rows, .. } => *n_rows,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Row `i`'s raw values. Only valid BEFORE standardization — a taken
    /// [`Self::Releasable`] buffer panics here (nothing calls this after
    /// standardization; the panic is the guard that keeps it that way).
    pub fn row(&self, i: usize) -> &[f64] {
        match self {
            Self::Borrowed(r) => r[i],
            Self::Releasable {
                data, n_features, ..
            } => &data[i * n_features..(i + 1) * n_features],
        }
    }

    /// Iterate the raw rows. Same validity window as [`Self::row`].
    pub fn iter(&self) -> impl Iterator<Item = &[f64]> {
        (0..self.len()).map(move |i| self.row(i))
    }
}

/// One named slice of training/validation data.
///
/// Multi-group training resolves V0_2's "synthetic dominates by 15×"
/// imbalance: per-step sampling picks a group in proportion to
/// `train_weight`, then samples a pair within.
///
/// `train_weight` and `validation_weight` are independent: a group
/// can be in both pools (trained on AND gated against), in only one,
/// or in neither (per-epoch SROCC still logged for transparency).
#[derive(Debug)]
pub struct TrainingGroup<'a> {
    pub name: String,
    pub human_scores: &'a [f64],
    /// RAW (unstandardized) feature rows. See [`FeatureRows`] for the two
    /// ownership shapes and which one releases memory.
    pub features: FeatureRows<'a>,
    /// Per-row metric-disagreement σ (from parquet_loader's
    /// auto-computation). When present and σ-weighted MSE is enabled,
    /// the MSE loss divides by max(σ, ε) per pair — errors on
    /// high-consensus stimuli (low σ) are penalized more.
    pub metric_sigmas: Option<&'a [f64]>,
    /// Weight in the per-step group selection distribution. The
    /// per-pair sampling probability is `train_weight / total_weight`,
    /// so doubling `train_weight` doubles the sampling rate.
    /// Set to `0.0` to exclude this group from training.
    pub train_weight: f64,
    /// Weight in the per-epoch validation aggregation. `0.0` excludes
    /// the group from best-checkpoint scoring (it's still reported in
    /// the log). For `ValidationPolicy::Min`, weights act as a soft
    /// inclusion mask — any group with `validation_weight > 0`
    /// participates in the min.
    pub validation_weight: f64,

    /// Per-row reference-image identity. When `Some`, RankNet pairs for
    /// this group are drawn WITHIN a single reference image: pick a ref
    /// uniformly, then two distinct rows inside it. When `None` (the
    /// default, and every group's behavior before 2026-07-15), pairs are
    /// drawn uniformly across the whole group, which mixes images.
    ///
    /// **Why this exists.** A cross-image pair teaches "image A outranks
    /// image B" — a statement about between-image *scale*. A within-image
    /// pair teaches "this distortion is worse than that one" — the actual
    /// ranking task. When a corpus's within-image ladder is small next to
    /// its between-image spread, uniform draws bury the ladder under
    /// scale noise. MEASURED on the post-jxl-fix near-lossless corpus:
    /// the ssim2 ladder moves ~0.92 pts within an image against ~6 pts
    /// between images, so its pooled SROCC reads +0.204 while its per-ref
    /// SROCC reads +0.916 — the same confound as the documented AIC-3
    /// "0.79 pooled / 0.93 per-ref". Training that corpus with uniform
    /// pairs fits the scale, not the ladder.
    ///
    /// Groups whose target is already cross-image-comparable (safesyn,
    /// cid22_train, kadid, tid — all MOS/ssim2-anchored on a shared
    /// scale) should leave this `None`: for them a cross-image pair is a
    /// *true* and useful statement, and restricting to within-ref would
    /// discard most of the available signal.
    pub ref_ids: Option<&'a [u32]>,

    /// Which loss terms this group contributes. See [`GroupLossMode`].
    /// Defaults to [`GroupLossMode::Rank`] — the behavior of every group
    /// before 2026-07-15, so recipes that don't set it are unchanged.
    pub loss_mode: GroupLossMode,
}

/// Which loss terms a training group contributes.
///
/// **Why this is per-group.** A corpus's target column determines what it
/// can teach. `safesyn`/`bigcodec`/`kadis` carry an ssim2-derived score on
/// a shared cross-image scale — an absolute regression target is
/// meaningful for them. The near-lossless HF corpus does not: its ssim2
/// ladder moves ~0.92 pts within an image against ~6 pts between images
/// (see [`TrainingGroup::ref_ids`]), so an absolute term there fits
/// between-image noise. It can only be consumed as rank.
///
/// Before this existed, the plain (non-α-head) path was RankNet-only for
/// *every* group and `mse_weight` was rejected outright, so "MSE on the
/// main groups, rank-only on HF" — the round-7 recipe — was not
/// expressible in Rust at all. Measured consequence:
/// `benchmarks/r7_hf_rust_reproduction_2026-07-15.md` reproduced round-7's
/// CID22 (−0.0047 vs −0.0041) and non-photo (−0.0016 vs −0.0017) deltas but
/// inverted KonJND (−0.035 vs +0.033), because a rank-only objective has no
/// absolute dial for KonJND — the most calibration-sensitive corpus — to
/// track.
///
/// The RankNet term is scale-free, so a `Rank` group cannot drag the
/// absolute dial that the `Mse` groups establish.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GroupLossMode {
    /// RankNet pairs only. The default, and the behavior of every group
    /// before 2026-07-15. Pairs with a tied target are skipped (they
    /// carry no ranking signal).
    #[default]
    Rank,
    /// Absolute regression only (`mse_weight · (y − target)²`), no rank
    /// term. Tied targets are NOT skipped — an absolute target is still
    /// valid when two rows happen to score alike.
    Mse,
    /// Both terms. Tied-target pairs contribute the regression term only.
    Both,
}

impl GroupLossMode {
    /// Does this group contribute the RankNet term?
    pub(crate) fn has_rank(self) -> bool {
        matches!(self, Self::Rank | Self::Both)
    }
    /// Does this group contribute the absolute-regression term?
    pub(crate) fn has_mse(self) -> bool {
        matches!(self, Self::Mse | Self::Both)
    }
}

/// Row indices bucketed by reference image, for within-ref pair draws.
///
/// Built once per group at train start (refs with a single row are
/// dropped — they can't yield a pair). `flat` holds the row indices of
/// every usable ref back-to-back; `spans` gives each ref's `(start,
/// len)` window into it. This layout keeps the draw to two RNG calls and
/// two index reads, matching the cost of the uniform path it replaces.
#[derive(Debug, Default)]
pub(crate) struct RefBuckets {
    flat: Vec<usize>,
    spans: Vec<(usize, usize)>,
    /// row index -> its span slot, `u32::MAX` for rows whose ref was
    /// dropped as unusable. Lets a partner be re-drawn from the same ref
    /// (see `redraw_partner`).
    slot_of_row: Vec<u32>,
}

impl RefBuckets {
    /// Bucket `ref_ids` (dense, `0..n_refs`) into per-ref row-index runs,
    /// dropping refs with fewer than 2 rows. Returns `None` if no ref has
    /// a usable pair — the caller must then refuse to train the group
    /// within-ref rather than silently fall back to cross-image draws.
    pub(crate) fn build(ref_ids: &[u32]) -> Option<Self> {
        let n_refs = ref_ids.iter().copied().max().map(|m| m as usize + 1)?;
        let mut by_ref: Vec<Vec<usize>> = vec![Vec::new(); n_refs];
        for (row, &r) in ref_ids.iter().enumerate() {
            by_ref[r as usize].push(row);
        }
        let mut flat = Vec::with_capacity(ref_ids.len());
        let mut spans = Vec::new();
        let mut slot_of_row = vec![u32::MAX; ref_ids.len()];
        for rows in by_ref {
            if rows.len() < 2 {
                continue;
            }
            let slot = spans.len() as u32;
            spans.push((flat.len(), rows.len()));
            for &r in &rows {
                slot_of_row[r] = slot;
            }
            flat.extend(rows);
        }
        if spans.is_empty() {
            return None;
        }
        Some(Self {
            flat,
            spans,
            slot_of_row,
        })
    }

    /// Number of refs with at least one drawable pair.
    pub(crate) fn n_refs(&self) -> usize {
        self.spans.len()
    }

    /// Rows reachable by a within-ref draw. Less than the group's row
    /// count when some refs were dropped for having a single row.
    pub(crate) fn n_rows(&self) -> usize {
        self.flat.len()
    }

    /// Draw a within-ref pair: a ref uniformly, then two rows in it.
    /// May return `ia == ib`; the caller already skips that case, which
    /// keeps the RNG draw count identical to the uniform path.
    pub(crate) fn draw(&self, u_ref: u64, u_a: u64, u_b: u64) -> (usize, usize) {
        let (start, len) = self.spans[(u_ref as usize) % self.spans.len()];
        (
            self.flat[start + (u_a as usize) % len],
            self.flat[start + (u_b as usize) % len],
        )
    }

    /// Re-draw a partner for `row` from `row`'s OWN ref.
    ///
    /// The STRATEGY hard-pair miner re-draws row B uniformly over the
    /// group (`% n`) when hunting a near-threshold pair. On a within-ref
    /// group that would silently reintroduce cross-image pairs — the
    /// exact confound within-ref exists to prevent — so the miner must
    /// route through this instead. Returns `row` unchanged if its ref was
    /// dropped as unusable; the caller's `ia == ib` skip absorbs that.
    pub(crate) fn redraw_partner(&self, row: usize, u: u64) -> usize {
        match self.slot_of_row.get(row).copied() {
            Some(s) if s != u32::MAX => {
                let (start, len) = self.spans[s as usize];
                self.flat[start + (u as usize) % len]
            }
            _ => row,
        }
    }
}

/// STRATEGY-2026-07-02: raw human triplet responses for the ordered-probit
/// NLL loss (KonFiG / AIC-3 lineage). `features` are RAW (unstandardized —
/// the trainer standardizes with the same scaler as the groups). `responses`
/// index into `features`: (left, right, resp) with resp 0 = left judged MORE
/// DISTORTED, 1 = right, 2 = not sure (the trap-verified convention).
#[derive(Debug, Default)]
pub struct TripletPool {
    pub features: Vec<Vec<f64>>,
    pub responses: Vec<(u32, u32, u8)>,
}

/// Cross-codec JND anchor rows for `PreviewV0_5TunerV2` (2026-05-19).
///
/// Each row carries a feature vector + a row weight; all rows share the
/// global anchor target score (typically 63, the CID22-paper PJND
/// calibration point). During training, anchor steps sample a row,
/// forward through the head, and apply MSE against `target_score`.
///
/// **Why a separate anchor type instead of a regular training group**:
/// regular groups participate in RankNet *pair* sampling (need ≥2 rows
/// per source). Anchors are *single-row* MSE supervision against a
/// constant target — they want different sampling semantics. Keeping
/// them as a sibling struct keeps the loop unambiguous.
#[derive(Debug)]
pub struct AnchorRows<'a> {
    pub name: String,
    pub features: &'a [&'a [f64]],
    pub row_weights: &'a [f64],
    /// EXP-CROSS-CODEC-V5 (2026-05-19): per-row anchor score target.
    /// When `Some` and the slice has the same length as `features`, the
    /// anchor MSE step uses this row's target instead of the global
    /// `hyperparams.anchor_target_score`. Enables piecewise multi-band
    /// anchors (e.g. row 0 targets score=90 at butter=0.3, row 1 targets
    /// score=63 at butter=1.5, etc.). When `None` or empty, the trainer
    /// falls back to `hyperparams.anchor_target_score` as in V4.
    pub target_scores: Option<&'a [f64]>,
}

/// KONJND-AGGREGATION-HEAD (2026-05-24, task #4) — per-source-grouped
/// feature pool for the konjnd-dense aggregation training step.
///
/// Each ref carries N (typically 20) standardized feature rows + ONE
/// per-ref scalar `pjnd_target`. The aggregation step samples K refs
/// per fire, S rows per ref, forwards K·S times, computes K aggregate
/// scalars (the mean per ref), and applies MSE against `pjnd_target`
/// per ref. Backprop scales the residual `(agg − t)` by `(2w/S)`
/// uniformly to all S rows of that ref before flowing through the
/// existing per-sample-α head gradient plumbing.
///
/// Unlike [`AnchorRows`] (per-row MSE against a constant or per-row
/// target), this struct's loss is computed on the *aggregate of S
/// predictions per ref*. The within-ref gradient is non-zero in
/// general — this is what fixes the per-pair-MSE zero-gradient
/// pathology that pjnd_passthrough hit at V11-D.
///
/// Memory: `Vec<Vec<f64>>` of all standardized rows + per-ref slice
/// indices. We keep the row-features layout to share the same feature
/// scaler and SIMD-friendly contiguous representation as the primary
/// stream; the per-ref grouping is expressed as `(start_row, n_rows)`
/// in a sidecar table.
#[derive(Debug)]
pub struct KonjndAggregationPool<'a> {
    pub name: String,
    /// Flat row storage: one Vec<f64> per row, n_features long.
    /// Equivalent to AnchorRows.features layout for SIMD-friendly access.
    pub features: &'a [&'a [f64]],
    /// Per-ref ranges into `features`: `(start_row, n_rows)` per ref.
    /// `n_refs = ref_ranges.len()`. Sum of `n_rows` across all refs
    /// equals `features.len()`.
    pub ref_ranges: &'a [(usize, usize)],
    /// Per-ref pjnd_target (the constant value shared across all rows
    /// of that ref). `ref_pjnd_target.len() == ref_ranges.len()`.
    pub ref_pjnd_target: &'a [f64],
    /// Per-ref training weight (defaults to 1.0). Allows future
    /// preferential ref sampling without changing the API.
    pub ref_weight: &'a [f64],
}

/// Cross-codec equivalence pairs (2026-05-19, EXP-CROSS-CODEC-METRIC).
///
/// Each pair carries two feature vectors `(features_a[i], features_b[i])`
/// that are derived from distortions produced by DIFFERENT codecs at q
/// values empirically aligned to the same butteraugli level (the "pivot
/// metric" for cross-codec equivalence). During training, an equivalence
/// step samples a pair, forwards BOTH feature vectors, and applies a
/// squared-difference loss `w · (y_a - y_b)²` so the metric learns to
/// score perceptually-equivalent cross-codec outputs at the same value.
///
/// Unlike AnchorRows (single-row MSE to fixed target), EquivPairs supplies
/// SHAPE-FREE supervision — the metric is free to choose ANY score for
/// the equivalence class, as long as both members map to the same number.
/// This is the core mechanism for intrinsic cross-codec consistency.
///
/// **EXP-CROSS-CODEC-V3 (2026-05-19): `butter_diff` field.** Optional per-
/// pair `butter_a − butter_b` (in butteraugli-pnorm3 units). When the
/// `cross_codec_rank_preserve_weight` hyperparameter is `> 0` and this
/// slice has the same length as `features_a`, the training loop adds an
/// auxiliary RankNet-style rank-preservation term whose magnitude scales
/// with `|butter_diff|` — this prevents the equivalence-MSE term from
/// collapsing the network's outputs to a constant. An empty slice
/// disables rank-preserve regardless of the hyperparameter setting.
#[derive(Debug)]
pub struct EquivPairs<'a> {
    pub name: String,
    pub features_a: &'a [&'a [f64]],
    pub features_b: &'a [&'a [f64]],
    pub row_weights: &'a [f64],
    /// Per-pair butter_a − butter_b (butteraugli-pnorm3 score units).
    /// LOWER butter = HIGHER quality, so `butter_diff > 0` means A is
    /// quality-WORSE than B. Empty slice = rank-preserve disabled.
    pub butter_diff: &'a [f64],
}

/// Train a 2-layer MLP across multiple datasets via RankNet pairwise
/// loss + Adam, with per-dataset SROCC tracking and best-checkpoint
/// selection on the validation mean.
///
/// `human_scores[i]` is the human-rated quality for pair `i`; HIGHER
/// means MORE similar to the source. The MLP must produce LOWER
/// raw_distance for higher-scored pairs.
///
/// Returns the bytes of the best-validation checkpoint (ZNPK v1).
pub fn train_mlp(
    groups: &mut [TrainingGroup<'_>],
    n_features: usize,
    hyperparams: &MlpHyperparams,
    log: &mut Vec<String>,
) -> Vec<u8> {
    train_mlp_with_tv(groups, n_features, hyperparams, log, None)
}

/// TV (total-variation) regularizer for adjacent-q monotonicity.
///
/// `pairs[k] = (lo_idx, hi_idx)` references rows in the concatenated
/// trainer-feature space (group 0 rows first, then group 1, etc.).
/// Penalty per pair = `max(0, pred[hi_idx] - pred[lo_idx])` — Rust
/// trainer outputs are distance-like (lower = better quality), so a
/// monotone curve has `pred[lo_q] > pred[hi_q]`. Violations have
/// `pred[hi_q] > pred[lo_q]`.
pub struct TvRegularizer {
    pub pairs: Vec<(usize, usize)>,
    pub features: Vec<Vec<f64>>,
    pub weight: f64,
    /// Apply TV update every N RankNet pair updates. 50 is a good
    /// default — 50,000 / 50 = 1000 TV steps per epoch.
    pub apply_every: usize,
    /// Mini-batch size of TV pairs per update. 32 is fine.
    pub batch: usize,
    /// Per-pair band id (0..3 → B0/B1/B2/B3 per CID22 paper Table 5).
    /// When `Some`, must have the same length as `pairs`. The TV
    /// gradient for pair `k` is scaled by `band_weights[band_id[k]]`
    /// (if `band_weights` is also set) instead of the flat `weight`.
    pub band_id: Option<Vec<u8>>,
    /// Per-band TV weights `[B0, B1, B2, B3]`. When `Some`, must be
    /// paired with `band_id`. Used in place of `weight` for per-pair
    /// scaling. Set to e.g. `[10.0, 20.0, 10.0, 10.0]` to push B1
    /// harder than other bands.
    pub band_weights: Option<[f64; 4]>,
    /// Anti-collapse margin for the within-ladder hinge. The penalty
    /// becomes `max(0, y_harsher - y_milder + margin)`, forcing a
    /// minimum per-step gap of `margin` between adjacent severity
    /// levels instead of merely non-increasing. A pure hinge
    /// (margin=0) is minimized by collapsing every ladder flat, which
    /// destroys dynamic range AND (empirically) craters cross-image
    /// rank on analytic corpora; a positive margin spreads the ladder
    /// instead. Units are the network's raw output scale. Default 0.0
    /// (pure hinge).
    pub margin: f64,
}

impl TvRegularizer {
    fn n_features_check(&self) -> usize {
        self.features.first().map(|v| v.len()).unwrap_or(0)
    }
}

/// Internal entry point that accepts an optional TV regularizer.
pub fn train_mlp_with_tv(
    groups: &mut [TrainingGroup<'_>],
    n_features: usize,
    hyperparams: &MlpHyperparams,
    log: &mut Vec<String>,
    tv: Option<&TvRegularizer>,
) -> Vec<u8> {
    train_mlp_with_tv_anchored(groups, n_features, hyperparams, log, tv, None)
}

/// `PreviewV0_5TunerV2` (2026-05-19) entry point with optional JND
/// anchor data. When `anchor` is provided AND
/// `hyperparams.anchor_loss_weight > 0` AND
/// `hyperparams.per_sample_alpha_head` is enabled, the per-sample-α
/// training loop interleaves anchor MSE steps with regular pair-loss
/// steps. The anchor rows regress to the constant
/// `hyperparams.anchor_target_score`.
///
/// For every other head (pool_head / hybrid_head / plain MLP) the
/// anchor argument is ignored (anchor wiring is currently
/// tuner-specific). Trainer prints a warning if anchor data is supplied
/// but the per-sample-α path isn't active.
pub fn train_mlp_with_tv_anchored(
    groups: &mut [TrainingGroup<'_>],
    n_features: usize,
    hyperparams: &MlpHyperparams,
    log: &mut Vec<String>,
    tv: Option<&TvRegularizer>,
    anchor: Option<&AnchorRows<'_>>,
) -> Vec<u8> {
    train_mlp_with_tv_anchored_equiv(groups, n_features, hyperparams, log, tv, anchor, None)
}

/// Entry point with optional cross-codec equivalence pair pool
/// (EXP-CROSS-CODEC-METRIC, 2026-05-19). When `equiv` is provided AND
/// `hyperparams.cross_codec_eq_weight > 0` AND
/// `hyperparams.per_sample_alpha_head` is enabled, the per-sample-α
/// training loop interleaves equivalence pair steps with regular
/// pair-loss + anchor steps. Each equivalence step samples one pair
/// `(features_a, features_b)`, forwards both, and applies
/// `w · (y_a − y_b)²` MSE on their difference. Gradients flow through
/// both forward passes into the same Adam state.
///
/// For every other head this argument is currently ignored (equiv
/// wiring is per-sample-α-specific, like anchor).
pub fn train_mlp_with_tv_anchored_equiv(
    groups: &mut [TrainingGroup<'_>],
    n_features: usize,
    hyperparams: &MlpHyperparams,
    log: &mut Vec<String>,
    tv: Option<&TvRegularizer>,
    anchor: Option<&AnchorRows<'_>>,
    equiv: Option<&EquivPairs<'_>>,
) -> Vec<u8> {
    train_mlp_with_tv_anchored_equiv_pjnd(
        groups,
        n_features,
        hyperparams,
        log,
        tv,
        anchor,
        equiv,
        None,
        None,
    )
}

/// EXP-V11-D-PJND-DOMINANT entry point (2026-05-20, task #198) with
/// optional SECOND anchor pool (`pjnd_anchor`) for the KonJND-PJND
/// passthrough loss. Functions identically to the cross-codec-eq
/// anchor pool, but fires with `hyperparams.pjnd_passthrough_step_p`
/// and applies `pjnd_passthrough_weight · row_w · (y − target)²`
/// where `target` is per-row from `pjnd_anchor.target_scores` if
/// supplied, else `hyperparams.pjnd_passthrough_target_score`.
///
/// The pjnd pool is independent of the V11 cross-codec-eq anchor;
/// both may be active simultaneously (this is the V11-D experimental
/// setup — see CLAUDE.md task #198).
///
/// Only wired on the `per_sample_alpha_head = true` path; ignored
/// elsewhere (with a warning).
#[allow(clippy::too_many_arguments)] // one optional research-pool input per aux loss
pub fn train_mlp_with_tv_anchored_equiv_pjnd(
    groups: &mut [TrainingGroup<'_>],
    n_features: usize,
    hyperparams: &MlpHyperparams,
    log: &mut Vec<String>,
    tv: Option<&TvRegularizer>,
    anchor: Option<&AnchorRows<'_>>,
    equiv: Option<&EquivPairs<'_>>,
    pjnd_anchor: Option<&AnchorRows<'_>>,
    konjnd_agg: Option<&KonjndAggregationPool<'_>>,
) -> Vec<u8> {
    train_mlp_strategy(
        groups,
        n_features,
        hyperparams,
        log,
        tv,
        anchor,
        equiv,
        pjnd_anchor,
        konjnd_agg,
        None,
    )
}

/// STRATEGY-2026-07-02 entry point: everything the pjnd entry does, plus an
/// optional raw-human-triplet pool for the ordered-probit NLL loss.
#[allow(clippy::too_many_arguments)]
pub fn train_mlp_strategy(
    groups: &mut [TrainingGroup<'_>],
    n_features: usize,
    hyperparams: &MlpHyperparams,
    log: &mut Vec<String>,
    tv: Option<&TvRegularizer>,
    anchor: Option<&AnchorRows<'_>>,
    equiv: Option<&EquivPairs<'_>>,
    pjnd_anchor: Option<&AnchorRows<'_>>,
    konjnd_agg: Option<&KonjndAggregationPool<'_>>,
    triplets: Option<&TripletPool>,
) -> Vec<u8> {
    // EX-2 std-pool head dispatch (scalar fallback path). Pool-head
    // backprop has not been SIMD-fused yet; we trade ~1.7× per-pair
    // time for the architectural lift (GMSD's std-pooling +
    // Butteraugli's p-norm + IW-style pooling) per
    // `PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md §3`. NiN composes with
    // pool-head via the same per-prediction grad-scatter pattern as
    // the standard head (the per-prediction grad is added to the
    // RankNet `dl_dy` before routing through `backprop_step_pool_head`);
    // parallel-batch over chunks is out of scope for the v0 prod
    // wire-in (the sequential mini-batch path covers V_22-mix-LARGE's
    // K=256 recipe at h=128 in ~25 min wall on the 7950X — fast enough
    // that a SIMD/par-chunks port is queued, not load-bearing).
    let head_flags = (hyperparams.pool_head as u8)
        + (hyperparams.hybrid_head as u8)
        + (hyperparams.per_sample_alpha_head as u8);
    assert!(
        head_flags <= 1,
        "pool_head / hybrid_head / per_sample_alpha_head are mutually exclusive"
    );

    // Loss-term reachability gates. These MUST live here, in the
    // dispatcher, ahead of every head branch — the versions they replace
    // sat INSIDE `train_mlp_per_sample_alpha_head` and tested
    // `!per_sample_alpha_head`, which is unreachable there by
    // construction. So they were dead code that could never fire, and
    // their doc ("trainer panics if set on other heads") was false:
    // `--mse-weight` on a non-α head silently trained pure rank and threw
    // the flag away. Found 2026-07-15 by a `should_panic` test that did
    // not panic.
    if hyperparams.monotonicity_reg > 0.0 && !hyperparams.per_sample_alpha_head {
        panic!(
            "--monotonicity-reg is only wired on the per_sample_alpha_head \
             path (set --per-sample-alpha-head)."
        );
    }
    if hyperparams.mse_weight > 0.0
        && !hyperparams.per_sample_alpha_head
        && !groups.iter().any(|g| g.loss_mode.has_mse())
    {
        // On the plain path the absolute term is opt-in PER GROUP, so a
        // run that sets the weight but flags no group would train pure
        // rank and ignore the flag — exactly the silent failure above.
        panic!(
            "--mse-weight is set but no group opted into an absolute term. \
             On the plain path the regression term is per-group: append \
             `:mse` or `:both` to a --group spec (or use \
             --per-sample-alpha-head)."
        );
    }
    // The rest of the STRATEGY family is in the SAME silent-no-op class, and
    // was unguarded until 2026-09-04: `ema`, `hard_pair`, `dro` and
    // `listwise` are all read ONLY inside `train_mlp_per_sample_alpha_head`
    // (verified by line position — every `hyperparams.<knob>` read sits
    // after that function's start), so setting one on any other head threw
    // it away without a word.
    //
    // Found while scoping the subset-quality study: `--ema-decay` looked
    // like the natural lever for reducing seed variance — its own CLI doc
    // says "seed-variance reduction" — and it would have silently done
    // nothing on the path every board bake trains through. That is a
    // measurement that reads as "EMA does not help" when the truth is "EMA
    // never ran".
    //
    // Fixing them properly means threading a weight-EMA / DRO / listwise
    // step through three more training loops, which is a real change to the
    // optimizer and the bake path, not a drive-by. Failing loud is the
    // repo's established response to this class (see the guards above), and
    // it is what turns a silent wrong answer into a stopped run.
    for (name, on, flag) in [
        ("--ema-decay", hyperparams.ema_decay > 0.0, "ema_decay"),
        (
            "--hard-pair-frac",
            hyperparams.hard_pair_frac > 0.0,
            "hard_pair_frac",
        ),
        ("--dro-eta", hyperparams.dro_eta > 0.0, "dro_eta"),
        (
            "--listwise-weight",
            hyperparams.listwise_weight > 0.0,
            "listwise_weight",
        ),
    ] {
        if on && !hyperparams.per_sample_alpha_head {
            panic!(
                "{name} ({flag}) is only wired on the per_sample_alpha_head path; this run \
                 has per_sample_alpha_head=false, so the flag would be silently ignored and \
                 the bake would be byte-identical to a run that never set it. Add \
                 --per-sample-alpha-head, or drop the flag."
            );
        }
    }
    // Same silent-no-op class as the two guards above: the triplet step lives
    // ONLY inside `train_mlp_per_sample_alpha_head`, so a run that loads a
    // triplet pool but does NOT set --per-sample-alpha-head silently ignores it
    // and produces a bake byte-identical to a no-triplet run. Measured
    // 2026-07-16: depth_v6 (--triplet-weight 0.5, plain 2-layer) == depth_v2
    // byte-for-byte. Fail loud instead of throwing the flag away.
    let triplet_requested = hyperparams.triplet_weight > 0.0
        && triplets.map(|t| !t.responses.is_empty()).unwrap_or(false);
    if triplet_requested && !hyperparams.per_sample_alpha_head {
        panic!(
            "--triplet-weight/-stimuli/-responses are only wired on the \
             per_sample_alpha_head path; this run has per_sample_alpha_head=false so the \
             loaded triplet pool ({} responses) would be silently ignored (the bake would \
             be byte-identical to a no-triplet run). Set --per-sample-alpha-head, or drop \
             the triplet flags.",
            triplets.map(|t| t.responses.len()).unwrap_or(0),
        );
    }

    if hyperparams.per_sample_alpha_head {
        return train_mlp_per_sample_alpha_head(
            groups,
            n_features,
            hyperparams,
            log,
            anchor,
            equiv,
            pjnd_anchor,
            konjnd_agg,
            tv,
            triplets,
        );
    }
    if anchor.is_some() && hyperparams.anchor_loss_weight > 0.0 {
        eprintln!(
            "WARNING: --anchor-loss-weight is only wired on the per-sample-α head; \
             anchor data ignored on this head."
        );
    }
    if pjnd_anchor.is_some() && hyperparams.pjnd_passthrough_weight > 0.0 {
        eprintln!(
            "WARNING: --pjnd-passthrough-weight is only wired on the per-sample-α head; \
             pjnd anchor data ignored on this head."
        );
    }
    if konjnd_agg.is_some() && hyperparams.konjnd_aggregation_weight > 0.0 {
        eprintln!(
            "WARNING: --konjnd-aggregation-weight is only wired on the per-sample-α head; \
             konjnd-aggregation pool ignored on this head."
        );
    }
    if hyperparams.hybrid_head {
        return train_mlp_hybrid_head_with_tv(groups, n_features, hyperparams, log, tv);
    }
    if hyperparams.pool_head {
        return train_mlp_pool_head_with_tv(groups, n_features, hyperparams, log, tv);
    }

    let n_outputs = 1usize;
    let n_hidden = hyperparams.n_hidden;

    assert!(!groups.is_empty(), "need at least one training group");
    for g in groups.iter() {
        assert_eq!(
            g.human_scores.len(),
            g.features.len(),
            "{}: scores/features length mismatch",
            g.name
        );
        assert!(
            g.features.iter().all(|f| f.len() == n_features),
            "{}: feature length mismatch",
            g.name
        );
    }

    let train_total: f64 = groups.iter().map(|g| g.train_weight).sum();
    assert!(
        train_total > 0.0,
        "no training groups (all train_weight == 0)"
    );

    // OUTPUT POLARITY. The model has one output, so the two loss terms must
    // agree on what it means — and by default they do NOT:
    //
    //   RankNet (legacy): target = sign(mos_a − mos_b), z = −target·(y_b − y_a).
    //     For mos_a > mos_b this minimizes softplus(y_a − y_b), pushing
    //     y_a < y_b — higher quality → LOWER y. DISTANCE-shaped.
    //   Absolute term: y is regressed onto `human_score` directly —
    //     higher quality → HIGHER y. SCORE-shaped.
    //
    // Mixed naively, the two terms pull in opposite directions and the rank
    // group's own corpus inverts. MEASURED 2026-07-15 on the round-7 recipe
    // (main groups `mse`, HF `withinref,rank`): HF held-out per-ref SROCC
    // went +0.6393 / 6% backwards WITHOUT the HF group to −0.3454 / 75%
    // backwards WITH it — adding rank supervision to a corpus made that
    // corpus rank backwards, which is only possible if the term is fighting
    // the absolute one.
    //
    // Python's reference recipe has no such conflict: its RankNet is
    // `BCE(s_i − s_j, 1 if quality_i > quality_j)`, which pushes higher
    // quality → HIGHER s, agreeing with its `smooth_l1`. So we reconcile the
    // same way: when ANY group carries the absolute term the model is
    // score-shaped and the rank term flips to match. With no absolute term
    // the sign is +1 and the legacy distance convention is preserved
    // bit-for-bit (every existing recipe is `Rank`-only).
    let rank_target_sign = if groups.iter().any(|g| g.loss_mode.has_mse()) {
        -1.0
    } else {
        1.0
    };

    let train_indices: Vec<usize> = groups
        .iter()
        .enumerate()
        .filter_map(|(i, g)| if g.train_weight > 0.0 { Some(i) } else { None })
        .collect();
    let val_indices: Vec<usize> = groups
        .iter()
        .enumerate()
        .filter_map(|(i, g)| {
            if g.validation_weight > 0.0 {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    let log_line = |msg: &str, log: &mut Vec<String>| {
        eprintln!("{msg}");
        log.push(msg.to_string());
    };

    log_line(
        &format!(
            "MLP train: arch=[{n_features} → {n_hidden} (LeakyReLU α={alpha}) → 1], val_policy={:?}",
            hyperparams.validation_policy,
            alpha = hyperparams.leaky_alpha,
        ),
        log,
    );
    for (i, g) in groups.iter().enumerate() {
        let role = match (g.train_weight > 0.0, g.validation_weight > 0.0) {
            (true, true) => "train+val",
            (true, false) => "train",
            (false, true) => "val-only",
            (false, false) => "report",
        };
        log_line(
            &format!(
                "  {role:>9} group {i}: '{}' n={} train_w={:.3} val_w={:.3} withinref={} loss={:?}",
                g.name,
                g.features.len(),
                g.train_weight,
                g.validation_weight,
                g.ref_ids.is_some(),
                g.loss_mode,
            ),
            log,
        );
    }

    // 1. Compute per-feature scaler (mean / std) using ALL training-group
    //    samples. Validation-only groups are excluded from the scaler so
    //    we never look at validation data during fit.
    let (scaler_mean, scaler_scale) =
        compute_scaler_from_groups(groups, &train_indices, n_features);

    // 2. Standardize features per group up-front. Standardizing now
    //    avoids redoing it inside the per-step inner loop and lets the
    //    inner loop just slice into a flat f64 buffer per group.
    //    Group g's standardized features live in std_features[g], shape
    //    (n_pairs[g] × n_features).
    let std_features =
        standardize_groups_releasing_raw(groups, n_features, &scaler_mean, &scaler_scale);

    // Standardize TV-regularizer features using the same scaler, in
    // their flat (n_rows × n_features) form. The TV pairs reference
    // row indices in this flat buffer.
    let tv_std: Option<Vec<f64>> = tv.map(|t| {
        assert_eq!(
            t.n_features_check(),
            n_features,
            "TV features dimensionality must match training features"
        );
        let n_rows = t.features.len();
        let mut buf = vec![0.0f64; n_rows * n_features];
        for (i, f) in t.features.iter().enumerate() {
            for d in 0..n_features {
                buf[i * n_features + d] = (f[d] - scaler_mean[d]) / scaler_scale[d].max(1e-12);
            }
        }
        buf
    });

    // 3. Initialize weights (Xavier-Glorot for tanh/leaky-relu).
    //
    // METHODOLOGY: the init RNG and the sampler RNG are SEPARATE so
    // that comparing trainers at different `n_features` (e.g.
    // 228 baseline vs 372 IW A/B) sees the SAME sequence of training
    // pair draws. Using one RNG for both would cause the sampler
    // state at epoch 0 to differ between the two arms (init consumes
    // more normals when n_features is larger), making the A/B
    // unfair. Both seeds derive deterministically from
    // `hyperparams.seed` so reproducibility is preserved.
    let mut init_rng = SplitMix64::new(hyperparams.init_seed.unwrap_or(hyperparams.seed));
    let mut rng = SplitMix64::new(sampling::sample_stream_seed(
        hyperparams.sample_seed.unwrap_or(hyperparams.seed),
    ));
    let std1 = (2.0 / (n_features + n_hidden) as f64).sqrt();
    let std2 = (2.0 / (n_hidden + n_outputs) as f64).sqrt();
    let mut w1 = (0..n_features * n_hidden)
        .map(|_| init_rng.next_normal() * std1)
        .collect::<Vec<_>>();
    let mut b1 = vec![0.0f64; n_hidden];
    let mut w2 = (0..n_hidden * n_outputs)
        .map(|_| init_rng.next_normal() * std2)
        .collect::<Vec<_>>();
    let mut b2 = vec![0.0f64; n_outputs];

    // Feature-subset pinning (`--keep-features`): zero the layer-1 rows of
    // dropped inputs once. Their standardized values are exactly 0.0 (the bin
    // zeroed the raw column), so the rows can never move again — this is an
    // exact K-wide fit with the SAME init draws and the SAME sampled pairs as
    // the full-width run at this seed (init and sampler RNGs are separate by
    // design, see above), which is what makes the K-sweep a controlled
    // ablation rather than a re-roll.
    let masked_rows = zero_masked_w1_rows(&mut w1, n_hidden, input_keep_mask().as_deref());
    if masked_rows > 0 {
        log_line(
            &format!(
                "[keep-features] pinned {masked_rows} of {n_features} layer-1 input rows to 0 \
                 (effective width {})",
                n_features - masked_rows
            ),
            log,
        );
    }
    if group_l1_lambda() > 0.0 {
        log_line(
            &format!(
                "[group-l1] decoupled group-lasso prox ON: lambda {} (threshold lr*lambda per step)",
                group_l1_lambda()
            ),
            log,
        );
    }

    let mut adam = AdamState::new(w1.len(), b1.len(), w2.len(), b2.len());

    // 4. Training loop.
    let start = Instant::now();
    // The "validation score" is the mean SROCC across val_indices.
    // If there are no validation-only groups, fall back to the mean
    // across all groups (= training SROCC mean).
    let mut best_val_score = f64::NEG_INFINITY;
    let mut best_bake: Option<Vec<u8>> = None;
    let mut stale_epochs = 0usize;

    // Pre-compute the cumulative-distribution table for group sampling.
    let cdf: Vec<f64> = {
        let mut cum = 0.0;
        train_indices
            .iter()
            .map(|&gi| {
                cum += groups[gi].train_weight;
                cum / train_total
            })
            .collect()
    };

    // Pre-compute per-row sampling CDFs for each training group when
    // any of low_q_boost / mid_q_boost / high_q_boost is non-trivial.
    // Indexed by position-in-train_indices so the hot loop can look up
    // by the same index used for the group CDF.
    //
    // Band cuts are anchored to CID22 Table 5 (human_score 0-100 scale,
    // see zensim/CLAUDE.md "Per-band reporting rule"):
    //   B0 < 50               full low_q_boost
    //   B1 [50, 65)           sqrt(low_q_boost), AND full mid_q_boost
    //   B2 [65, 90)           full mid_q_boost
    //   B3 ≥ 90               full high_q_boost (V0_20a addition)
    // When all boosts are 1.0 the per-row CDF is None and within-group
    // sampling stays uniform (identical to V0_5/V0_15/V0_16 trainer
    // behavior).
    let needs_row_boost = hyperparams.low_q_boost != 1.0
        || hyperparams.mid_q_boost != 1.0
        || hyperparams.high_q_boost != 1.0;
    let per_row_cdfs: Vec<Option<Vec<f64>>> = train_indices
        .iter()
        .map(|&gi| {
            if !needs_row_boost {
                return None;
            }
            let g = &groups[gi];
            let mut cum = 0.0;
            let raw: Vec<f64> = g
                .human_scores
                .iter()
                .map(|&s| {
                    let mut w = 1.0;
                    if hyperparams.low_q_boost != 1.0 {
                        if s < 50.0 {
                            w *= hyperparams.low_q_boost;
                        } else if s < 65.0 {
                            w *= hyperparams.low_q_boost.sqrt();
                        }
                    }
                    if hyperparams.mid_q_boost != 1.0 && (50.0..90.0).contains(&s) {
                        w *= hyperparams.mid_q_boost;
                    }
                    if hyperparams.high_q_boost != 1.0 && s >= 90.0 {
                        w *= hyperparams.high_q_boost;
                    }
                    cum += w;
                    cum
                })
                .collect();
            let total = *raw.last().unwrap_or(&1.0);
            Some(raw.into_iter().map(|c| c / total).collect())
        })
        .collect();
    // Within-ref pair buckets, parallel to `per_row_cdfs`. `Some` only for
    // groups the caller opted in via `TrainingGroup::ref_ids` (the binary
    // sets that only when the group spec asked for within-ref). Groups
    // without it keep the uniform cross-image draw, byte-for-byte.
    let ref_buckets: Vec<Option<RefBuckets>> = train_indices
        .iter()
        .map(|&gi| groups[gi].ref_ids.and_then(RefBuckets::build))
        .collect();
    // Row counts by position-in-train_indices, for the owner draw step.
    // `FeatureRows::len` is cached at construction (it stays correct after
    // standardization takes the buffer), so hoisting it out of the hot
    // loop is exact, not an approximation.
    let row_counts: Vec<usize> = train_indices
        .iter()
        .map(|&gi| groups[gi].features.len())
        .collect();
    // Opt-in sample-sequence digest: the faithfulness proof that
    // `sampling::simulate` replays THIS run's draws. Off by default, so a
    // normal run is byte- and cost-identical.
    let mut sample_digest = std::env::var("ZENSIM_SAMPLE_DIGEST")
        .ok()
        .filter(|v| v == "1")
        .map(|_| sampling::SampleSequenceDigest::new());
    // STRATEGY stratified row-A bands. This used to be built in ONE of the
    // four training loops, so `--stratified-bands` was a silent no-op on
    // every other path — including the standard path every board bake
    // trained through. Empty when the flag is 0, which keeps the default
    // byte-identical.
    let strat_bands: Vec<Vec<Vec<usize>>> = if hyperparams.stratified_bands > 0 {
        train_indices
            .iter()
            .map(|&gi| strategy::build_bands(groups[gi].human_scores, hyperparams.stratified_bands))
            .collect()
    } else {
        Vec::new()
    };
    // Say so out loud: which pairing mode a group trained under changes what
    // the bake learned, and it must never be a silent default.
    for (pos, rb) in ref_buckets.iter().enumerate() {
        if let Some(rb) = rb {
            println!(
                "  {}: WITHIN-REF pairs over {} refs ({} rows usable)",
                groups[train_indices[pos]].name,
                rb.n_refs(),
                rb.n_rows(),
            );
        }
    }

    // T8.1 (2026-05-16): mini-batch SGD. K=1 keeps per-pair Adam
    // (bit-identical to the legacy trainer). K>1 accumulates K
    // RankNet pair gradients between Adam updates, with a final-flush
    // step at epoch end if `pairs_per_epoch % K != 0`. The TV branch
    // mirrors the same modulo gate.
    //
    // T8.2: when `parallel_batch` AND K>1, the K forward+backward
    // computations within a mini-batch are dispatched to rayon. The
    // sample-drawing sequence is run sequentially on the main RNG —
    // only the per-pair compute is parallelized — so same seed +
    // same K produces bit-identical bake bytes regardless of
    // thread count.
    let k = hyperparams.minibatch_size.max(1);
    let parallel = hyperparams.parallel_batch && k > 1;
    // Buffer holding sequentially-drawn (group_idx, ia, ib) samples
    // for the parallel-batch path. Always pre-allocated to capacity K
    // so push/clear in the hot loop don't realloc.
    let mut parallel_batch_buffer: Vec<(usize, usize, usize)> = Vec::with_capacity(k);

    // Norm-in-Norm hybrid loss gate (Li et al. 2020). When opted in via
    // `norm_in_norm_weight > 0.0`, the auxiliary loss is computed on
    // each mini-batch's 2K predictions. Batch statistics are unstable
    // at N < ~16 so we enforce `K >= 16` up front rather than
    // silently producing nonsense gradients.
    let nin_on = hyperparams.norm_in_norm_weight > 0.0;
    if nin_on {
        assert!(
            k >= 16,
            "--norm-in-norm-weight={} requires --minibatch-size >= 16 \
             for stable batch statistics; bump K (currently {}) or set \
             --norm-in-norm-weight 0 to disable",
            hyperparams.norm_in_norm_weight,
            k
        );
        log_line(
            &format!(
                "MLP train: Norm-in-Norm hybrid loss active (Li 2020): \
                 β={:.4} p={:.2} q={:.2} on 2K={} predictions per batch",
                hyperparams.norm_in_norm_weight,
                hyperparams.norm_in_norm_p,
                hyperparams.norm_in_norm_q,
                2 * k,
            ),
            log,
        );
    }

    for epoch in 0..hyperparams.n_epochs {
        let lr = hyperparams.initial_lr
            * 0.5
            * (1.0 + (std::f64::consts::PI * (epoch % 50) as f64 / 50.0).cos());

        let mut total_loss = 0.0f64;
        let mut n_steps = 0u64;
        // Counts gradient-contributing steps since the last Adam call;
        // controls the final-flush at epoch end when not aligned with K.
        let mut steps_since_adam = 0u64;

        for _ in 0..hyperparams.pairs_per_epoch {
            // Pick a training group via inverse-CDF sampling, then a
            // pair (ia, ib) within that group. If per-row CDFs are
            // populated (boost != 1.0), use weighted sampling for the
            // pair; otherwise uniform.
            // Draw via THE owner (`sampling::draw_pair`) — see that module
            // for why the RNG consumption pattern is a wire contract.
            let drawn = sampling::draw_pair(
                &sampling::PairDrawCtx {
                    cdf: &cdf,
                    row_counts: &row_counts,
                    per_row_cdfs: &per_row_cdfs,
                    ref_buckets: &ref_buckets,
                    strat_bands: &strat_bands,
                },
                &mut rng,
            );
            if let Some(d) = sample_digest.as_mut() {
                d.push(drawn);
            }
            let sampling::Draw::Pair { train_pos, ia, ib } = drawn else {
                continue;
            };
            let g_idx = train_indices[train_pos];
            let g = &groups[g_idx];

            // Norm-in-Norm (Li 2020) path: ALWAYS buffer K samples and
            // route through `run_minibatch_with_nin` (sequential within
            // the mini-batch — the NiN loss is a batch-correlated
            // function and cannot be cleanly chunked the way pure
            // RankNet can). Forward, NiN-loss + grad, backward, single
            // Adam step. Takes precedence over the `parallel` path
            // because the two are mutually exclusive routings.
            if nin_on {
                parallel_batch_buffer.push((g_idx, ia, ib));
                if parallel_batch_buffer.len() >= k {
                    let (steps_added, loss_added) = run_minibatch_with_nin(
                        &parallel_batch_buffer,
                        groups,
                        &std_features,
                        &w1,
                        &b1,
                        &w2,
                        &b2,
                        &mut adam,
                        n_features,
                        n_hidden,
                        hyperparams.leaky_alpha,
                        hyperparams.l2_lambda,
                        hyperparams.pwrc_pair_weight,
                        hyperparams.pwrc_sensory_threshold,
                        hyperparams.pwrc_band_weights.as_deref(),
                        hyperparams.norm_in_norm_weight,
                        hyperparams.norm_in_norm_p,
                        hyperparams.norm_in_norm_q,
                    );
                    parallel_batch_buffer.clear();
                    total_loss += loss_added;
                    n_steps += steps_added;
                    if steps_added > 0 {
                        adam.step(&mut w1, &mut b1, &mut w2, &mut b2, lr);
                        apply_post_adam_penalties(&mut w1, n_hidden, lr);
                    }
                }
                continue;
            }

            // T8.2 parallel-batch path: buffer up to K (g_idx, ia, ib)
            // samples on the main RNG (sequential), then process the
            // entire batch concurrently into a Vec<LocalGrads> whose
            // source-order is preserved by `par_iter().collect()`. The
            // per-pair grads are summed sequentially (deterministic FP
            // reduce) into the AdamState, optional L2 reg is added,
            // and a single Adam step closes the batch.
            if parallel {
                parallel_batch_buffer.push((g_idx, ia, ib));
                if parallel_batch_buffer.len() >= k {
                    let (steps_added, loss_added) = run_parallel_minibatch(
                        &parallel_batch_buffer,
                        groups,
                        &std_features,
                        &w1,
                        &b1,
                        &w2,
                        &b2,
                        &mut adam,
                        n_features,
                        n_hidden,
                        hyperparams.leaky_alpha,
                        hyperparams.l2_lambda,
                        hyperparams.pwrc_pair_weight,
                        hyperparams.pwrc_sensory_threshold,
                        hyperparams.pwrc_band_weights.as_deref(),
                    );
                    parallel_batch_buffer.clear();
                    total_loss += loss_added;
                    n_steps += steps_added;
                    if steps_added > 0 {
                        adam.step(&mut w1, &mut b1, &mut w2, &mut b2, lr);
                        apply_post_adam_penalties(&mut w1, n_hidden, lr);
                    }
                }
                continue;
            }

            let g_feats = &std_features[g_idx];
            let xa = &g_feats[ia * n_features..(ia + 1) * n_features];
            let xb = &g_feats[ib * n_features..(ib + 1) * n_features];
            let (ya, ha_pre, ha) = forward(
                xa,
                &w1,
                &b1,
                &w2,
                &b2,
                n_features,
                n_hidden,
                hyperparams.leaky_alpha,
            );
            let (yb, hb_pre, hb) = forward(
                xb,
                &w1,
                &b1,
                &w2,
                &b2,
                n_features,
                n_hidden,
                hyperparams.leaky_alpha,
            );

            let mos_a = g.human_scores[ia];
            let mos_b = g.human_scores[ib];
            // `rank_target_sign` reconciles the rank term with the absolute
            // term's polarity; it is +1 (a no-op) unless a group carries MSE.
            let target = rank_target_sign * (mos_a - mos_b).signum();
            // Per-group loss mode (2026-07-15). `Rank` is the default and
            // reproduces the pre-existing control flow exactly: the two
            // `continue`s below fire unconditionally, as they always did.
            // A group carrying an absolute term must NOT be dropped by
            // either — a tied or sub-threshold pair still has a valid
            // regression target. Both drops happen AFTER the (ia,ib) draw,
            // so the RNG advances identically either way and an all-`Rank`
            // run stays bit-identical to the legacy trainer.
            let mode = g.loss_mode;
            let rank_tied = target == 0.0;
            let rank_sub_threshold = hyperparams.pwrc_pair_weight
                && hyperparams.pwrc_sensory_threshold > 0.0
                && (mos_a - mos_b).abs() < hyperparams.pwrc_sensory_threshold;
            // PWRC sensory-threshold drop (Wu et al. 2018): pairs with
            // |ΔMOS| < T are perceptually tied and uninformative for
            // ranking learning.
            let want_rank = mode.has_rank() && !rank_tied && !rank_sub_threshold;
            let want_mse = mode.has_mse() && hyperparams.mse_weight > 0.0;
            if !want_rank && !want_mse {
                continue;
            }
            // PWRC per-pair weight (default 1.0 when disabled). Scales
            // both loss & dl/dy linearly — the gradient is `pair_weight
            // * dl/dy` because loss is linear in pair_weight.
            let pair_weight = if hyperparams.pwrc_pair_weight {
                pwrc_pair_weight(mos_a, mos_b, hyperparams.pwrc_band_weights.as_deref())
            } else {
                1.0
            };
            // RankNet term. Zeroed for `GroupLossMode::Mse` groups and for
            // tied / sub-threshold pairs of a `Both` group.
            let (dl_dya_rn, dl_dyb_rn) = if want_rank {
                let pred_diff = yb - ya;
                let z = -target * pred_diff;
                let loss_raw = if z > 50.0 {
                    z
                } else if z < -50.0 {
                    0.0
                } else {
                    (z.exp() + 1.0).ln()
                };
                total_loss += loss_raw * pair_weight;

                let sig_z = 1.0 / (1.0 + (-z).exp());
                let dl_d_pred_diff = -target * sig_z * pair_weight;
                (-dl_d_pred_diff, dl_d_pred_diff)
            } else {
                (0.0, 0.0)
            };

            // Absolute-regression term, per prediction:
            //   loss  = mse_weight · ((y_a−t_a)² + (y_b−t_b)²) / 2
            //   dL/dy = mse_weight · (y − t)
            // (the 2 from d/dy of the square cancels the /2 over the pair's
            // two predictions).
            //
            // NOTE the normalization deliberately DIFFERS from the
            // per-sample-α path's `2·w/(2K)`. There, MSE is an auxiliary
            // regularizer explicitly scaled to "one RankNet pair's gradient"
            // — i.e. ~1/K of the rank term. Here it is the PRIMARY term for
            // an `Mse` group, so a 1/K factor would make `mse_weight`'s
            // meaning depend on `pairs_per_epoch` (measured: at w=1, K=400
            // the term is ~1/400 of the rank gradient and a `Both` group
            // lands 24.9 score units off its target). K-independent keeps
            // the knob readable: at `mse_weight = 1` the absolute gradient
            // is ~|y − t|, directly comparable to the rank term's O(1), so
            // ~0.1 is the balanced setting for `Both`.
            let (dl_dya_mse, dl_dyb_mse) = if want_mse {
                let (ra, rb) = (ya - mos_a, yb - mos_b);
                total_loss += hyperparams.mse_weight * (ra * ra + rb * rb) / 2.0;
                (hyperparams.mse_weight * ra, hyperparams.mse_weight * rb)
            } else {
                (0.0, 0.0)
            };

            n_steps += 1;
            steps_since_adam += 1;

            let dl_dya = dl_dya_rn + dl_dya_mse;
            let dl_dyb = dl_dyb_rn + dl_dyb_mse;

            backprop_step(
                xa,
                &ha_pre,
                &ha,
                dl_dya,
                &w1,
                &mut adam.gw1,
                &mut adam.gb1,
                &w2,
                &mut adam.gw2,
                &mut adam.gb2,
                n_features,
                n_hidden,
                hyperparams.leaky_alpha,
            );
            backprop_step(
                xb,
                &hb_pre,
                &hb,
                dl_dyb,
                &w1,
                &mut adam.gw1,
                &mut adam.gb1,
                &w2,
                &mut adam.gw2,
                &mut adam.gb2,
                n_features,
                n_hidden,
                hyperparams.leaky_alpha,
            );

            if hyperparams.l2_lambda > 0.0 {
                let fmult = l2_feature_mult();
                add_l2_grad_layer1(
                    &mut adam.gw1,
                    &w1,
                    hyperparams.l2_lambda,
                    n_hidden,
                    fmult.as_ref().map(|v| v.as_slice()),
                );
                for (g, &w) in adam.gw2.iter_mut().zip(w2.iter()) {
                    *g += hyperparams.l2_lambda * w;
                }
            }

            // T8.1: K=1 → step every pair (bit-identical to legacy).
            // K>1 sequential → step once per K accumulated pairs.
            if k == 1 || steps_since_adam >= k as u64 {
                adam.step(&mut w1, &mut b1, &mut w2, &mut b2, lr);
                apply_post_adam_penalties(&mut w1, n_hidden, lr);
                steps_since_adam = 0;
            }

            // TV regularizer step (per-curve adjacent-q monotonicity).
            // Apply every `apply_every` pair updates. Penalty per TV pair:
            // max(0, pred[hi_q] - pred[lo_q]) — Rust trainer output is
            // distance-like (lower = better), so a monotone curve has
            // pred[lo_q] > pred[hi_q]; the ReLU penalizes the opposite.
            if let (Some(tv_cfg), Some(tv_buf)) = (tv, tv_std.as_ref())
                && tv_cfg.weight > 0.0
                && tv_cfg.apply_every > 0
                && n_steps.is_multiple_of(tv_cfg.apply_every as u64)
                && !tv_cfg.pairs.is_empty()
            {
                let flat_scale = tv_cfg.weight / tv_cfg.batch.max(1) as f64;
                let per_band_active = tv_cfg.band_id.is_some() && tv_cfg.band_weights.is_some();
                let mut tv_steps_since_adam = 0u64;
                for tv_iter in 0..tv_cfg.batch {
                    let pair_idx = (rng.next_u64() as usize) % tv_cfg.pairs.len();
                    let (lo, hi) = tv_cfg.pairs[pair_idx];
                    let scale = if per_band_active {
                        let band = tv_cfg.band_id.as_ref().unwrap()[pair_idx] as usize;
                        let bw = tv_cfg.band_weights.as_ref().unwrap()[band];
                        bw / tv_cfg.batch.max(1) as f64
                    } else {
                        flat_scale
                    };
                    let xlo = &tv_buf[lo * n_features..(lo + 1) * n_features];
                    let xhi = &tv_buf[hi * n_features..(hi + 1) * n_features];
                    let (y_lo, h_lo_pre, h_lo) = forward(
                        xlo,
                        &w1,
                        &b1,
                        &w2,
                        &b2,
                        n_features,
                        n_hidden,
                        hyperparams.leaky_alpha,
                    );
                    let (y_hi, h_hi_pre, h_hi) = forward(
                        xhi,
                        &w1,
                        &b1,
                        &w2,
                        &b2,
                        n_features,
                        n_hidden,
                        hyperparams.leaky_alpha,
                    );
                    // Violation = (y_hi - y_lo) > 0 (worse direction).
                    let viol = y_hi - y_lo;
                    if viol <= 0.0 {
                        continue;
                    }
                    // d_loss/d_y_hi = +scale ; d_loss/d_y_lo = -scale
                    backprop_step(
                        xhi,
                        &h_hi_pre,
                        &h_hi,
                        scale,
                        &w1,
                        &mut adam.gw1,
                        &mut adam.gb1,
                        &w2,
                        &mut adam.gw2,
                        &mut adam.gb2,
                        n_features,
                        n_hidden,
                        hyperparams.leaky_alpha,
                    );
                    backprop_step(
                        xlo,
                        &h_lo_pre,
                        &h_lo,
                        -scale,
                        &w1,
                        &mut adam.gw1,
                        &mut adam.gb1,
                        &w2,
                        &mut adam.gw2,
                        &mut adam.gb2,
                        n_features,
                        n_hidden,
                        hyperparams.leaky_alpha,
                    );
                    tv_steps_since_adam += 1;
                    // T8.1 mirror: in K=1 mode, step per TV pair (legacy
                    // bit-identical). In K>1 mode, step once per K TV
                    // gradient contributions and flush leftover at the
                    // end of the TV batch.
                    let is_last_tv = tv_iter + 1 == tv_cfg.batch;
                    if k == 1 || tv_steps_since_adam >= k as u64 || is_last_tv {
                        if tv_steps_since_adam > 0 {
                            adam.step(&mut w1, &mut b1, &mut w2, &mut b2, lr);
                            apply_post_adam_penalties(&mut w1, n_hidden, lr);
                        }
                        tv_steps_since_adam = 0;
                    }
                }
            }
        }

        // T8.1 final-flush: leftover gradients accumulated since the
        // last Adam step (for K>1 sequential only — K=1 always flushes
        // every pair). Skip when the sequential loop already stepped
        // (steps_since_adam resets to 0 after each Adam call).
        if k > 1 && !parallel && steps_since_adam > 0 {
            adam.step(&mut w1, &mut b1, &mut w2, &mut b2, lr);
            apply_post_adam_penalties(&mut w1, n_hidden, lr);
        }
        // T8.2 final-flush for parallel buffer: handle the partial
        // batch at epoch end if pairs_per_epoch % K != 0. Buffer is
        // empty in K=1 / non-parallel modes (never populated). When
        // NiN is on, route the leftover through `run_minibatch_with_nin`
        // — same routing as the main loop — but only when the leftover
        // is large enough for stable batch statistics (≥ 16 samples).
        // Partial batches smaller than 16 are dropped (RNG sequence is
        // unchanged because we drew them; only the gradient update is
        // skipped — corresponds to "we observed the pairs but the NiN
        // batch statistics would be too noisy to use").
        if !parallel_batch_buffer.is_empty() {
            if nin_on {
                if parallel_batch_buffer.len() >= 16 {
                    let (steps_added, loss_added) = run_minibatch_with_nin(
                        &parallel_batch_buffer,
                        groups,
                        &std_features,
                        &w1,
                        &b1,
                        &w2,
                        &b2,
                        &mut adam,
                        n_features,
                        n_hidden,
                        hyperparams.leaky_alpha,
                        hyperparams.l2_lambda,
                        hyperparams.pwrc_pair_weight,
                        hyperparams.pwrc_sensory_threshold,
                        hyperparams.pwrc_band_weights.as_deref(),
                        hyperparams.norm_in_norm_weight,
                        hyperparams.norm_in_norm_p,
                        hyperparams.norm_in_norm_q,
                    );
                    total_loss += loss_added;
                    n_steps += steps_added;
                    if steps_added > 0 {
                        adam.step(&mut w1, &mut b1, &mut w2, &mut b2, lr);
                        apply_post_adam_penalties(&mut w1, n_hidden, lr);
                    }
                }
                parallel_batch_buffer.clear();
            } else {
                let (steps_added, loss_added) = run_parallel_minibatch(
                    &parallel_batch_buffer,
                    groups,
                    &std_features,
                    &w1,
                    &b1,
                    &w2,
                    &b2,
                    &mut adam,
                    n_features,
                    n_hidden,
                    hyperparams.leaky_alpha,
                    hyperparams.l2_lambda,
                    hyperparams.pwrc_pair_weight,
                    hyperparams.pwrc_sensory_threshold,
                    hyperparams.pwrc_band_weights.as_deref(),
                );
                parallel_batch_buffer.clear();
                total_loss += loss_added;
                n_steps += steps_added;
                if steps_added > 0 {
                    adam.step(&mut w1, &mut b1, &mut w2, &mut b2, lr);
                    apply_post_adam_penalties(&mut w1, n_hidden, lr);
                }
            }
        }

        let avg_loss = if n_steps > 0 {
            total_loss / n_steps as f64
        } else {
            0.0
        };

        if epoch % hyperparams.log_every == 0 || epoch == hyperparams.n_epochs - 1 {
            // Per-group SROCC. The MLP outputs raw_distance (lower =
            // more similar); human_scores are quality (higher = more
            // similar). They're anti-correlated by design, so we
            // compute SROCC against `-predictions` to surface positive
            // numbers that match V0_2's reporting convention.
            let agg_mode = hyperparams.val_aggregate;
            // Per-group panels are independent (each reads only its own rows +
            // the shared immutable weights) and the panel's PWRC is O(n²) on the
            // subsample cap, so the groups run on rayon and are collected in group
            // order — same values, same order, bit-identical.
            let group_panels: Vec<crate::panel::LightPanel> = groups
                .par_iter()
                .enumerate()
                .map(|(gi, g)| {
                    let preds = predict_group(
                        &std_features[gi],
                        g.features.len(),
                        n_features,
                        &w1,
                        &b1,
                        &w2,
                        &b2,
                        n_hidden,
                        hyperparams.leaky_alpha,
                    );
                    let neg_preds: Vec<f64> = preds.iter().map(|&p| -p).collect();
                    crate::panel::compute_light_panel_subsampled(&neg_preds, g.human_scores)
                })
                .collect();

            let group_scores: Vec<f64> =
                group_panels.iter().map(|p| p.aggregate(agg_mode)).collect();

            let val_score = if val_indices.is_empty() {
                group_scores.iter().sum::<f64>() / group_scores.len() as f64
            } else {
                match hyperparams.validation_policy {
                    ValidationPolicy::Mean => {
                        let total: f64 = val_indices
                            .iter()
                            .map(|&i| groups[i].validation_weight)
                            .sum();
                        val_indices
                            .iter()
                            .map(|&i| group_scores[i] * groups[i].validation_weight)
                            .sum::<f64>()
                            / total
                    }
                    ValidationPolicy::Min => val_indices
                        .iter()
                        .map(|&i| group_scores[i])
                        .fold(f64::INFINITY, f64::min),
                    ValidationPolicy::Goals => {
                        let gs = compute_goal_scores(&group_panels, groups, None, None);
                        gs.aggregate()
                    }
                }
            };

            let elapsed = start.elapsed().as_secs_f64();
            let per_group = group_panels
                .iter()
                .zip(groups.iter())
                .map(|(p, g)| {
                    format!(
                        "{}: srocc={:.4} plcc={:.4} pwrc={:.4}",
                        g.name, p.srocc, p.plcc, p.pwrc
                    )
                })
                .collect::<Vec<_>>()
                .join(" | ");
            log_line(
                &format!(
                    "  epoch {epoch:>3} | lr={lr:.5} | loss={avg_loss:.4} | val({agg_mode})={val_score:.4} (best={best_val_score:.4}) | {per_group} | t={elapsed:.1}s"
                ),
                log,
            );

            // H-TRAJ checkpoint dump (balance campaign 2026-08-28), plain-MLP
            // lane. TWIN OF the best-val snapshot below — same serialization,
            // current weights, spline-less (the pack step fits the spline).
            if hyperparams.dump_checkpoints_every > 0
                && epoch % hyperparams.dump_checkpoints_every == 0
            {
                let ckpt_bytes = bake_two_layer_znpr_v3(
                    &scaler_mean,
                    &scaler_scale,
                    &w1,
                    &b1,
                    &w2,
                    &b2,
                    n_features,
                    n_hidden,
                    n_outputs,
                    hyperparams.out_dtype,
                    hyperparams.feature_transforms.as_deref(),
                    hyperparams.feature_transform_params.as_deref(),
                );
                let dir = hyperparams
                    .dump_checkpoints_dir
                    .clone()
                    .unwrap_or_else(|| std::path::PathBuf::from("."));
                let ckpt_path = dir.join(format!("ckpt_epoch{epoch:03}.bin"));
                std::fs::write(&ckpt_path, &ckpt_bytes)
                    .expect("H-TRAJ checkpoint dump write failed");
                log_line(
                    &format!(
                        "  checkpoint dump: {} ({} B)",
                        ckpt_path.display(),
                        ckpt_bytes.len()
                    ),
                    log,
                );
            }
            if val_score > best_val_score {
                best_val_score = val_score;
                stale_epochs = 0;
                best_bake = Some(bake_two_layer_znpr_v3(
                    &scaler_mean,
                    &scaler_scale,
                    &w1,
                    &b1,
                    &w2,
                    &b2,
                    n_features,
                    n_hidden,
                    n_outputs,
                    hyperparams.out_dtype,
                    hyperparams.feature_transforms.as_deref(),
                    hyperparams.feature_transform_params.as_deref(),
                ));
            } else {
                stale_epochs += hyperparams.log_every;
                if hyperparams.early_stop_patience > 0
                    && stale_epochs >= hyperparams.early_stop_patience
                {
                    log_line(
                        &format!(
                            "  early stop at epoch {epoch} (no validation improvement for {stale_epochs} epochs)"
                        ),
                        log,
                    );
                    break;
                }
            }
        }
    }
    record_best_val(best_val_score);

    log_line(
        &format!("MLP train: best validation mean SROCC = {best_val_score:.4}"),
        log,
    );
    // Faithfulness hook: `sampling::simulate` must reproduce this hash.
    if let Some(d) = sample_digest.as_ref() {
        println!("ZENSIM_SAMPLE_DIGEST {}", d.hex());
    }
    best_bake.unwrap_or_else(|| {
        bake_two_layer_znpr_v3(
            &scaler_mean,
            &scaler_scale,
            &w1,
            &b1,
            &w2,
            &b2,
            n_features,
            n_hidden,
            n_outputs,
            hyperparams.out_dtype,
            hyperparams.feature_transforms.as_deref(),
            hyperparams.feature_transform_params.as_deref(),
        )
    })
}

/// EX-2 std-pool head trainer (scalar grad). Same recipe machinery as
/// [`train_mlp_with_tv`] — group sampling, per-row low/mid/high-q
/// boosts, cosine LR, mini-batch SGD with sequential gradient
/// accumulation, PWRC pair weighting + sensory threshold, TV
/// regularizer, L2 on layer weights, early stop on val SROCC, AND
/// NiN hybrid loss (per-prediction grad scattered through the pool-
/// head chain rule) — but substitutes the final scalar output with
/// the GMSD-style `pool[μ, σ, max, p_6]` reducer per
/// `PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md §3` and bakes via the
/// passthrough-layer + `zentrain.pool_head_reducer` metadata wire
/// format (see `zensim_train_core::pool_head::bake_pool_head_v3`).
///
/// **What's omitted vs `train_mlp_with_tv`:**
/// - **Parallel-batch (rayon par_chunks)** — pool-head per-pair work
///   is small enough that the K=256 mini-batch sequential loop runs
///   the full V_22-mix-LARGE recipe in ~25 min wall on the 7950X.
///   SIMD pool-head backprop + par_chunks are queued, not load-
///   bearing for the seed=3 first verdict.
/// - **TV regularizer with NiN active** — when both are on the
///   sequential code skips TV. V_22-mix-LARGE doesn't use TV so this
///   is a no-op for the head-to-head comparison.
fn train_mlp_pool_head_with_tv(
    groups: &mut [TrainingGroup<'_>],
    n_features: usize,
    hyperparams: &MlpHyperparams,
    log: &mut Vec<String>,
    tv: Option<&TvRegularizer>,
) -> Vec<u8> {
    use zensim_train_core::pool_head as ph;

    let n_hidden = hyperparams.n_hidden;
    let alpha = hyperparams.leaky_alpha;

    assert!(!groups.is_empty(), "need at least one training group");
    for g in groups.iter() {
        assert_eq!(
            g.human_scores.len(),
            g.features.len(),
            "{}: scores/features length mismatch",
            g.name
        );
        assert!(
            g.features.iter().all(|f| f.len() == n_features),
            "{}: feature length mismatch",
            g.name
        );
    }

    let train_total: f64 = groups.iter().map(|g| g.train_weight).sum();
    assert!(
        train_total > 0.0,
        "no training groups (all train_weight == 0)"
    );

    let train_indices: Vec<usize> = groups
        .iter()
        .enumerate()
        .filter_map(|(i, g)| if g.train_weight > 0.0 { Some(i) } else { None })
        .collect();
    let val_indices: Vec<usize> = groups
        .iter()
        .enumerate()
        .filter_map(|(i, g)| {
            if g.validation_weight > 0.0 {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    let log_line = |msg: &str, log: &mut Vec<String>| {
        eprintln!("{msg}");
        log.push(msg.to_string());
    };

    log_line(
        &format!(
            "MLP train (POOL-HEAD): arch=[{n_features} → {n_hidden} (LeakyReLU α={alpha}) \
             → pool[μ,σ,max,p_{:.0}] → 4→1 reducer], val_policy={:?}",
            ph::POOL_P_NORM,
            hyperparams.validation_policy,
        ),
        log,
    );
    for (i, g) in groups.iter().enumerate() {
        let role = match (g.train_weight > 0.0, g.validation_weight > 0.0) {
            (true, true) => "train+val",
            (true, false) => "train",
            (false, true) => "val-only",
            (false, false) => "report",
        };
        log_line(
            &format!(
                "  {role:>9} group {i}: '{}' n={} train_w={:.3} val_w={:.3} withinref={} loss={:?}",
                g.name,
                g.features.len(),
                g.train_weight,
                g.validation_weight,
                g.ref_ids.is_some(),
                g.loss_mode,
            ),
            log,
        );
    }

    let (scaler_mean, scaler_scale) =
        compute_scaler_from_groups(groups, &train_indices, n_features);

    let std_features =
        standardize_groups_releasing_raw(groups, n_features, &scaler_mean, &scaler_scale);

    let tv_std: Option<Vec<f64>> = tv.map(|t| {
        assert_eq!(
            t.n_features_check(),
            n_features,
            "TV features dimensionality must match training features"
        );
        let n_rows = t.features.len();
        let mut buf = vec![0.0f64; n_rows * n_features];
        for (i, f) in t.features.iter().enumerate() {
            for d in 0..n_features {
                buf[i * n_features + d] = (f[d] - scaler_mean[d]) / scaler_scale[d].max(1e-12);
            }
        }
        buf
    });

    // Init layer-1 weights via Xavier-Glorot (same RNG split scheme as
    // the standard head). The pool-head reducer is initialized via
    // `PoolHeadModel::new` and then we copy out its reducer init.
    let mut init_rng = SplitMix64::new(hyperparams.init_seed.unwrap_or(hyperparams.seed));
    let mut rng = SplitMix64::new(sampling::sample_stream_seed(
        hyperparams.sample_seed.unwrap_or(hyperparams.seed),
    ));
    let std1 = (2.0 / (n_features + n_hidden) as f64).sqrt();
    let mut w1 = (0..n_features * n_hidden)
        .map(|_| init_rng.next_normal() * std1)
        .collect::<Vec<_>>();
    let mut b1 = vec![0.0f64; n_hidden];
    // Reducer init: std-pool dominant, others small (matches
    // PoolHeadModel::new).
    let mut reducer_w: [f64; 4] = [0.05, 1.0, 0.05, 0.05];
    let mut reducer_b: f64 = 0.0;

    // Adam: gw1/gb1 sized as before; "w2 slot" holds reducer_w (4
    // entries) and "b2 slot" holds [reducer_b] (1 entry).
    let mut adam = AdamState::new(w1.len(), b1.len(), 4, 1);

    let start = Instant::now();
    let mut best_val_score = f64::NEG_INFINITY;
    let mut best_bake: Option<Vec<u8>> = None;
    let mut stale_epochs = 0usize;

    let cdf: Vec<f64> = {
        let mut cum = 0.0;
        train_indices
            .iter()
            .map(|&gi| {
                cum += groups[gi].train_weight;
                cum / train_total
            })
            .collect()
    };

    // Per-row boost CDFs (same band cuts as standard head).
    let needs_row_boost = hyperparams.low_q_boost != 1.0
        || hyperparams.mid_q_boost != 1.0
        || hyperparams.high_q_boost != 1.0;
    let per_row_cdfs: Vec<Option<Vec<f64>>> = train_indices
        .iter()
        .map(|&gi| {
            if !needs_row_boost {
                return None;
            }
            let g = &groups[gi];
            let mut cum = 0.0;
            let raw: Vec<f64> = g
                .human_scores
                .iter()
                .map(|&s| {
                    let mut w = 1.0;
                    if hyperparams.low_q_boost != 1.0 {
                        if s < 50.0 {
                            w *= hyperparams.low_q_boost;
                        } else if s < 65.0 {
                            w *= hyperparams.low_q_boost.sqrt();
                        }
                    }
                    if hyperparams.mid_q_boost != 1.0 && (50.0..90.0).contains(&s) {
                        w *= hyperparams.mid_q_boost;
                    }
                    if hyperparams.high_q_boost != 1.0 && s >= 90.0 {
                        w *= hyperparams.high_q_boost;
                    }
                    cum += w;
                    cum
                })
                .collect();
            let total = *raw.last().unwrap_or(&1.0);
            Some(raw.into_iter().map(|c| c / total).collect())
        })
        .collect();
    // Within-ref pair buckets, parallel to `per_row_cdfs`. `Some` only for
    // groups the caller opted in via `TrainingGroup::ref_ids` (the binary
    // sets that only when the group spec asked for within-ref). Groups
    // without it keep the uniform cross-image draw, byte-for-byte.
    let ref_buckets: Vec<Option<RefBuckets>> = train_indices
        .iter()
        .map(|&gi| groups[gi].ref_ids.and_then(RefBuckets::build))
        .collect();
    // Row counts by position-in-train_indices, for the owner draw step.
    // `FeatureRows::len` is cached at construction (it stays correct after
    // standardization takes the buffer), so hoisting it out of the hot
    // loop is exact, not an approximation.
    let row_counts: Vec<usize> = train_indices
        .iter()
        .map(|&gi| groups[gi].features.len())
        .collect();
    // Opt-in sample-sequence digest: the faithfulness proof that
    // `sampling::simulate` replays THIS run's draws. Off by default, so a
    // normal run is byte- and cost-identical.
    let mut sample_digest = std::env::var("ZENSIM_SAMPLE_DIGEST")
        .ok()
        .filter(|v| v == "1")
        .map(|_| sampling::SampleSequenceDigest::new());
    // STRATEGY stratified row-A bands. This used to be built in ONE of the
    // four training loops, so `--stratified-bands` was a silent no-op on
    // every other path — including the standard path every board bake
    // trained through. Empty when the flag is 0, which keeps the default
    // byte-identical.
    let strat_bands: Vec<Vec<Vec<usize>>> = if hyperparams.stratified_bands > 0 {
        train_indices
            .iter()
            .map(|&gi| strategy::build_bands(groups[gi].human_scores, hyperparams.stratified_bands))
            .collect()
    } else {
        Vec::new()
    };
    // Say so out loud: which pairing mode a group trained under changes what
    // the bake learned, and it must never be a silent default.
    for (pos, rb) in ref_buckets.iter().enumerate() {
        if let Some(rb) = rb {
            println!(
                "  {}: WITHIN-REF pairs over {} refs ({} rows usable)",
                groups[train_indices[pos]].name,
                rb.n_refs(),
                rb.n_rows(),
            );
        }
    }

    let k = hyperparams.minibatch_size.max(1);
    if hyperparams.parallel_batch && k > 1 {
        log_line(
            "pool_head: parallel-batch flag ignored (sequential mini-batch path).",
            log,
        );
    }
    let nin_on = hyperparams.norm_in_norm_weight > 0.0;
    if nin_on {
        assert!(
            k >= 16,
            "pool_head + --norm-in-norm-weight={} requires --minibatch-size >= 16 \
             for stable batch statistics (currently {})",
            hyperparams.norm_in_norm_weight,
            k
        );
        log_line(
            &format!(
                "pool_head: NiN hybrid ACTIVE (β={:.3}, p={:.2}, q={:.2}); per-prediction grad scattered through pool-head chain rule",
                hyperparams.norm_in_norm_weight,
                hyperparams.norm_in_norm_p,
                hyperparams.norm_in_norm_q,
            ),
            log,
        );
    }
    log_line(
        &format!(
            "pool_head: ENABLED — std-pool-head (Butteraugli p_6, GMSD σ, IW pooling, K={k} sequential)"
        ),
        log,
    );

    // Buffer for NiN-aware mini-batch path: stores per-pair forward
    // state across the K-pair batch so the second pass can route
    // both RankNet and NiN per-prediction grads through the pool-head
    // chain rule. Only used when `nin_on`. Pre-allocated to capacity K.
    let mut nin_buffer: Vec<Option<PoolPairForward<'_>>> = if nin_on {
        Vec::with_capacity(k)
    } else {
        Vec::new()
    };

    for epoch in 0..hyperparams.n_epochs {
        let lr = hyperparams.initial_lr
            * 0.5
            * (1.0 + (std::f64::consts::PI * (epoch % 50) as f64 / 50.0).cos());

        let mut total_loss = 0.0f64;
        let mut n_steps = 0u64;
        let mut steps_since_adam = 0u64;

        for _ in 0..hyperparams.pairs_per_epoch {
            // Draw via THE owner (`sampling::draw_pair`) — see that module
            // for why the RNG consumption pattern is a wire contract.
            let drawn = sampling::draw_pair(
                &sampling::PairDrawCtx {
                    cdf: &cdf,
                    row_counts: &row_counts,
                    per_row_cdfs: &per_row_cdfs,
                    ref_buckets: &ref_buckets,
                    strat_bands: &strat_bands,
                },
                &mut rng,
            );
            if let Some(d) = sample_digest.as_mut() {
                d.push(drawn);
            }
            let sampling::Draw::Pair { train_pos, ia, ib } = drawn else {
                continue;
            };
            let g_idx = train_indices[train_pos];
            let g = &groups[g_idx];

            let g_feats = &std_features[g_idx];
            let xa = &g_feats[ia * n_features..(ia + 1) * n_features];
            let xb = &g_feats[ib * n_features..(ib + 1) * n_features];

            let (ya, ha_pre, ha, sa, max_a) = ph::forward_pool_head(
                xa, &w1, &b1, &reducer_w, reducer_b, n_features, n_hidden, alpha,
            );
            let (yb, hb_pre, hb, sb, max_b) = ph::forward_pool_head(
                xb, &w1, &b1, &reducer_w, reducer_b, n_features, n_hidden, alpha,
            );

            let mos_a = g.human_scores[ia];
            let mos_b = g.human_scores[ib];
            let target = (mos_a - mos_b).signum();
            if target == 0.0 {
                if nin_on {
                    nin_buffer.push(None);
                    if nin_buffer.len() >= k {
                        flush_pool_head_nin_batch(
                            &mut nin_buffer,
                            &mut w1,
                            &mut b1,
                            &mut reducer_w,
                            &mut reducer_b,
                            &mut adam,
                            n_features,
                            n_hidden,
                            alpha,
                            hyperparams.l2_lambda,
                            hyperparams.norm_in_norm_weight,
                            hyperparams.norm_in_norm_p,
                            hyperparams.norm_in_norm_q,
                            lr,
                            &mut total_loss,
                            &mut n_steps,
                        );
                    }
                }
                continue;
            }
            if hyperparams.pwrc_pair_weight
                && hyperparams.pwrc_sensory_threshold > 0.0
                && (mos_a - mos_b).abs() < hyperparams.pwrc_sensory_threshold
            {
                if nin_on {
                    nin_buffer.push(None);
                    if nin_buffer.len() >= k {
                        flush_pool_head_nin_batch(
                            &mut nin_buffer,
                            &mut w1,
                            &mut b1,
                            &mut reducer_w,
                            &mut reducer_b,
                            &mut adam,
                            n_features,
                            n_hidden,
                            alpha,
                            hyperparams.l2_lambda,
                            hyperparams.norm_in_norm_weight,
                            hyperparams.norm_in_norm_p,
                            hyperparams.norm_in_norm_q,
                            lr,
                            &mut total_loss,
                            &mut n_steps,
                        );
                    }
                }
                continue;
            }
            let pair_weight = if hyperparams.pwrc_pair_weight {
                pwrc_pair_weight(mos_a, mos_b, hyperparams.pwrc_band_weights.as_deref())
            } else {
                1.0
            };
            let pred_diff = yb - ya;
            let z = -target * pred_diff;
            let loss_raw = if z > 50.0 {
                z
            } else if z < -50.0 {
                0.0
            } else {
                (z.exp() + 1.0).ln()
            };
            total_loss += loss_raw * pair_weight;
            n_steps += 1;

            let sig_z = 1.0 / (1.0 + (-z).exp());
            let dl_d_pred_diff = -target * sig_z * pair_weight;
            let dl_dya_rn = -dl_d_pred_diff;
            let dl_dyb_rn = dl_d_pred_diff;

            // NiN-aware path: buffer the per-pair forward; flush the
            // K-pair batch through `flush_pool_head_nin_batch` which
            // computes NiN over all 2K predictions and routes per-
            // prediction grad through the pool-head chain rule.
            if nin_on {
                nin_buffer.push(Some(PoolPairForward {
                    xa,
                    xb,
                    ya,
                    yb,
                    ha_pre,
                    ha,
                    hb_pre,
                    hb,
                    sa,
                    sb,
                    max_a,
                    max_b,
                    dl_dya_rn,
                    dl_dyb_rn,
                    mos_a,
                    mos_b,
                }));
                if nin_buffer.len() >= k {
                    flush_pool_head_nin_batch(
                        &mut nin_buffer,
                        &mut w1,
                        &mut b1,
                        &mut reducer_w,
                        &mut reducer_b,
                        &mut adam,
                        n_features,
                        n_hidden,
                        alpha,
                        hyperparams.l2_lambda,
                        hyperparams.norm_in_norm_weight,
                        hyperparams.norm_in_norm_p,
                        hyperparams.norm_in_norm_q,
                        lr,
                        &mut total_loss,
                        &mut n_steps,
                    );
                }
                // TV regularizer skipped on NiN-active path for
                // simplicity (the V_22-mix-LARGE recipe doesn't use
                // TV; the NiN-aware composition with TV is queued).
                continue;
            }

            steps_since_adam += 1;
            let dl_dya = dl_dya_rn;
            let dl_dyb = dl_dyb_rn;

            // Reducer grad sized [4] + scalar; accumulate locally per
            // pair and fold into adam.gw2/gb2 after both forwards.
            let mut g_red_w: [f64; 4] = [0.0; 4];
            let mut g_red_b: f64 = 0.0;
            ph::backprop_step_pool_head(
                xa,
                &ha_pre,
                &ha,
                &sa,
                max_a,
                dl_dya,
                &reducer_w,
                &mut adam.gw1,
                &mut adam.gb1,
                &mut g_red_w,
                &mut g_red_b,
                n_features,
                n_hidden,
                alpha,
            );
            ph::backprop_step_pool_head(
                xb,
                &hb_pre,
                &hb,
                &sb,
                max_b,
                dl_dyb,
                &reducer_w,
                &mut adam.gw1,
                &mut adam.gb1,
                &mut g_red_w,
                &mut g_red_b,
                n_features,
                n_hidden,
                alpha,
            );
            // L2 on w1 + reducer_w (excludes biases) — mirrors the
            // standard head's per-pair scaling. The cumulative effect
            // over a K-step mini-batch is K · λ · w, identical to the
            // standard head's L2-application schedule.
            if hyperparams.l2_lambda > 0.0 {
                let l2 = hyperparams.l2_lambda;
                for (g, &w) in adam.gw1.iter_mut().zip(w1.iter()) {
                    *g += l2 * w;
                }
                for k in 0..4 {
                    g_red_w[k] += l2 * reducer_w[k];
                }
            }
            // Fold reducer grads into Adam's w2/b2 slots.
            for kk in 0..4 {
                adam.gw2[kk] += g_red_w[kk];
            }
            adam.gb2[0] += g_red_b;

            if k == 1 || steps_since_adam >= k as u64 {
                // Adam step: reducer_w lives in the "w2 slot", reducer_b
                // in the "b2 slot". `adam.step` is shape-agnostic so we
                // pass aliased vectors and copy back.
                let mut r_w_vec: Vec<f64> = reducer_w.to_vec();
                let mut r_b_vec: Vec<f64> = vec![reducer_b];
                adam.step(&mut w1, &mut b1, &mut r_w_vec, &mut r_b_vec, lr);
                reducer_w = [r_w_vec[0], r_w_vec[1], r_w_vec[2], r_w_vec[3]];
                reducer_b = r_b_vec[0];
                steps_since_adam = 0;
            }

            // TV regularizer (per-curve adjacent-q monotonicity).
            if let (Some(tv_cfg), Some(tv_buf)) = (tv, tv_std.as_ref())
                && tv_cfg.weight > 0.0
                && tv_cfg.apply_every > 0
                && n_steps.is_multiple_of(tv_cfg.apply_every as u64)
                && !tv_cfg.pairs.is_empty()
            {
                let flat_scale = tv_cfg.weight / tv_cfg.batch.max(1) as f64;
                let per_band_active = tv_cfg.band_id.is_some() && tv_cfg.band_weights.is_some();
                let mut tv_steps_since_adam = 0u64;
                for tv_iter in 0..tv_cfg.batch {
                    let pair_idx = (rng.next_u64() as usize) % tv_cfg.pairs.len();
                    let (lo, hi) = tv_cfg.pairs[pair_idx];
                    let scale = if per_band_active {
                        let band = tv_cfg.band_id.as_ref().unwrap()[pair_idx] as usize;
                        let bw = tv_cfg.band_weights.as_ref().unwrap()[band];
                        bw / tv_cfg.batch.max(1) as f64
                    } else {
                        flat_scale
                    };
                    let xlo = &tv_buf[lo * n_features..(lo + 1) * n_features];
                    let xhi = &tv_buf[hi * n_features..(hi + 1) * n_features];
                    let (y_lo, h_lo_pre, h_lo, s_lo, max_lo) = ph::forward_pool_head(
                        xlo, &w1, &b1, &reducer_w, reducer_b, n_features, n_hidden, alpha,
                    );
                    let (y_hi, h_hi_pre, h_hi, s_hi, max_hi) = ph::forward_pool_head(
                        xhi, &w1, &b1, &reducer_w, reducer_b, n_features, n_hidden, alpha,
                    );
                    let viol = y_hi - y_lo;
                    if viol <= 0.0 {
                        continue;
                    }
                    let mut tv_red_w: [f64; 4] = [0.0; 4];
                    let mut tv_red_b: f64 = 0.0;
                    ph::backprop_step_pool_head(
                        xhi,
                        &h_hi_pre,
                        &h_hi,
                        &s_hi,
                        max_hi,
                        scale,
                        &reducer_w,
                        &mut adam.gw1,
                        &mut adam.gb1,
                        &mut tv_red_w,
                        &mut tv_red_b,
                        n_features,
                        n_hidden,
                        alpha,
                    );
                    ph::backprop_step_pool_head(
                        xlo,
                        &h_lo_pre,
                        &h_lo,
                        &s_lo,
                        max_lo,
                        -scale,
                        &reducer_w,
                        &mut adam.gw1,
                        &mut adam.gb1,
                        &mut tv_red_w,
                        &mut tv_red_b,
                        n_features,
                        n_hidden,
                        alpha,
                    );
                    for kk in 0..4 {
                        adam.gw2[kk] += tv_red_w[kk];
                    }
                    adam.gb2[0] += tv_red_b;
                    tv_steps_since_adam += 1;
                    let is_last_tv = tv_iter + 1 == tv_cfg.batch;
                    if k == 1 || tv_steps_since_adam >= k as u64 || is_last_tv {
                        if tv_steps_since_adam > 0 {
                            let mut r_w_vec: Vec<f64> = reducer_w.to_vec();
                            let mut r_b_vec: Vec<f64> = vec![reducer_b];
                            adam.step(&mut w1, &mut b1, &mut r_w_vec, &mut r_b_vec, lr);
                            reducer_w = [r_w_vec[0], r_w_vec[1], r_w_vec[2], r_w_vec[3]];
                            reducer_b = r_b_vec[0];
                        }
                        tv_steps_since_adam = 0;
                    }
                }
            }
        }

        // Final-flush leftover K>1 accumulated gradient (RankNet path).
        if k > 1 && !nin_on && steps_since_adam > 0 {
            let mut r_w_vec: Vec<f64> = reducer_w.to_vec();
            let mut r_b_vec: Vec<f64> = vec![reducer_b];
            adam.step(&mut w1, &mut b1, &mut r_w_vec, &mut r_b_vec, lr);
            reducer_w = [r_w_vec[0], r_w_vec[1], r_w_vec[2], r_w_vec[3]];
            reducer_b = r_b_vec[0];
        }

        // Final-flush leftover NiN buffer if any surviving pairs ≥ 16.
        // Mirrors the standard head's `run_minibatch_with_nin` final-
        // flush at the same threshold.
        if nin_on && !nin_buffer.is_empty() {
            let surviving = nin_buffer.iter().filter(|p| p.is_some()).count();
            if surviving >= 16 {
                flush_pool_head_nin_batch(
                    &mut nin_buffer,
                    &mut w1,
                    &mut b1,
                    &mut reducer_w,
                    &mut reducer_b,
                    &mut adam,
                    n_features,
                    n_hidden,
                    alpha,
                    hyperparams.l2_lambda,
                    hyperparams.norm_in_norm_weight,
                    hyperparams.norm_in_norm_p,
                    hyperparams.norm_in_norm_q,
                    lr,
                    &mut total_loss,
                    &mut n_steps,
                );
            } else {
                nin_buffer.clear();
            }
        }

        let avg_loss = if n_steps > 0 {
            total_loss / n_steps as f64
        } else {
            0.0
        };

        if epoch % hyperparams.log_every == 0 || epoch == hyperparams.n_epochs - 1 {
            // Per-group SROCC. Pool-head output `y` is score-shaped
            // (higher = more similar) when the reducer is initialized
            // with `w_σ > 0` (the std-pool dominant init) — but the
            // overall direction depends on what the trainer converges
            // to. We use the same `-predictions` convention as the
            // standard head so the per-group SROCC numbers stay sign-
            // consistent across runs. (If pool_head converges to a
            // sign-flipped surface, the rank ordering is identical, so
            // SROCC magnitude is unchanged — only the sign is reported
            // negative; the bake at inference is consumed via runtime
            // pool-head dispatch which doesn't apply this negation.)
            let agg_mode = hyperparams.val_aggregate;
            let group_panels: Vec<crate::panel::LightPanel> = groups
                .iter()
                .enumerate()
                .map(|(gi, g)| {
                    let preds = predict_group_pool_head(
                        &std_features[gi],
                        g.features.len(),
                        n_features,
                        &w1,
                        &b1,
                        &reducer_w,
                        reducer_b,
                        n_hidden,
                        alpha,
                    );
                    let neg_preds: Vec<f64> = preds.iter().map(|&p| -p).collect();
                    crate::panel::compute_light_panel_subsampled(&neg_preds, g.human_scores)
                })
                .collect();

            let group_scores: Vec<f64> =
                group_panels.iter().map(|p| p.aggregate(agg_mode)).collect();

            let val_score = if val_indices.is_empty() {
                group_scores.iter().sum::<f64>() / group_scores.len() as f64
            } else {
                match hyperparams.validation_policy {
                    ValidationPolicy::Mean => {
                        let total: f64 = val_indices
                            .iter()
                            .map(|&i| groups[i].validation_weight)
                            .sum();
                        val_indices
                            .iter()
                            .map(|&i| group_scores[i] * groups[i].validation_weight)
                            .sum::<f64>()
                            / total
                    }
                    ValidationPolicy::Min => val_indices
                        .iter()
                        .map(|&i| group_scores[i])
                        .fold(f64::INFINITY, f64::min),
                    ValidationPolicy::Goals => {
                        let gs = compute_goal_scores(&group_panels, groups, None, None);
                        gs.aggregate()
                    }
                }
            };

            let elapsed = start.elapsed().as_secs_f64();
            let per_group = group_panels
                .iter()
                .zip(groups.iter())
                .map(|(p, g)| {
                    format!(
                        "{}: srocc={:.4} plcc={:.4} pwrc={:.4}",
                        g.name, p.srocc, p.plcc, p.pwrc
                    )
                })
                .collect::<Vec<_>>()
                .join(" | ");
            log_line(
                &format!(
                    "  epoch {epoch:>3} | lr={lr:.5} | loss={avg_loss:.4} | val({agg_mode})={val_score:.4} (best={best_val_score:.4}) | reducer_w=[μ={:.3},σ={:.3},max={:.3},p6={:.3}] reducer_b={:.3} | {per_group} | t={elapsed:.1}s",
                    reducer_w[0], reducer_w[1], reducer_w[2], reducer_w[3], reducer_b,
                ),
                log,
            );

            if val_score > best_val_score {
                best_val_score = val_score;
                stale_epochs = 0;
                let model = ph::PoolHeadModel {
                    scaler_mean: scaler_mean.clone(),
                    scaler_scale: scaler_scale.clone(),
                    w1: w1.clone(),
                    b1: b1.clone(),
                    reducer_w,
                    reducer_b,
                    n_hidden,
                    n_features,
                };
                best_bake = Some(ph::bake_pool_head_v3(&model));
            } else {
                stale_epochs += hyperparams.log_every;
                if hyperparams.early_stop_patience > 0
                    && stale_epochs >= hyperparams.early_stop_patience
                {
                    log_line(
                        &format!(
                            "  early stop at epoch {epoch} (no validation improvement for {stale_epochs} epochs)"
                        ),
                        log,
                    );
                    break;
                }
            }
        }
    }
    record_best_val(best_val_score);

    log_line(
        &format!(
            "MLP train (POOL-HEAD): best validation mean SROCC = {best_val_score:.4} | final reducer_w=[μ={:.3},σ={:.3},max={:.3},p6={:.3}] reducer_b={:.3}",
            reducer_w[0], reducer_w[1], reducer_w[2], reducer_w[3], reducer_b,
        ),
        log,
    );
    // Faithfulness hook: `sampling::simulate` must reproduce this hash.
    if let Some(d) = sample_digest.as_ref() {
        println!("ZENSIM_SAMPLE_DIGEST {}", d.hex());
    }
    best_bake.unwrap_or_else(|| {
        let model = ph::PoolHeadModel {
            scaler_mean: scaler_mean.clone(),
            scaler_scale: scaler_scale.clone(),
            w1: w1.clone(),
            b1: b1.clone(),
            reducer_w,
            reducer_b,
            n_hidden,
            n_features,
        };
        ph::bake_pool_head_v3(&model)
    })
}

/// Predict pool-head outputs for every row in a flat (n_pairs ×
/// n_features) standardized feature buffer. Mirrors [`predict_group`]
/// for the standard head.
#[allow(clippy::too_many_arguments)]
fn predict_group_pool_head(
    std_x: &[f64],
    n_pairs: usize,
    n_features: usize,
    w1: &[f64],
    b1: &[f64],
    reducer_w: &[f64; 4],
    reducer_b: f64,
    n_hidden: usize,
    alpha: f64,
) -> Vec<f64> {
    use zensim_train_core::pool_head as ph;
    (0..n_pairs)
        .map(|i| {
            let xi = &std_x[i * n_features..(i + 1) * n_features];
            let (y, _, _, _, _) = ph::forward_pool_head(
                xi, w1, b1, reducer_w, reducer_b, n_features, n_hidden, alpha,
            );
            y
        })
        .collect()
}

/// EX-2 follow-up: hybrid pool + rank head trainer.
///
/// Same structural shape as `train_mlp_pool_head_with_tv` but the
/// per-pair forward + backprop go through
/// `zensim_train_core::hybrid_head::{forward_hybrid_head,
/// backprop_step_hybrid_head}` and the bake metadata key changes to
/// `zentrain.hybrid_head`. Supports RankNet + minibatch + PWRC + L2 +
/// TV + **NiN (Li 2020 norm-in-norm hybrid loss)**.
///
/// NiN composition: when `norm_in_norm_weight > 0`, the per-pair
/// forward state is buffered until K pairs accumulate, then
/// `flush_hybrid_head_nin_batch` computes NiN over the 2K predictions,
/// scatters per-prediction grad back through `backprop_step_hybrid_head`
/// (which routes through BOTH rank-head + pool-head + sigmoid α),
/// and Adam-steps. The TV regularizer is skipped on the NiN-active
/// path (mirrors pool-head trainer).
///
/// **Adam slot layout** (different from pool_head trainer):
/// - gw1/gb1: layer-1 (n_features × n_hidden) + (n_hidden) — unchanged
/// - gw2: concatenation `[rank_w (n_hidden) | reducer_w (4) | alpha_logit (1)]`
///   sized `n_hidden + 5`
/// - gb2: `[rank_b, reducer_b]` sized 2
///
/// Final flush + early-stop semantics match pool-head trainer.
fn train_mlp_hybrid_head_with_tv(
    groups: &mut [TrainingGroup<'_>],
    n_features: usize,
    hyperparams: &MlpHyperparams,
    log: &mut Vec<String>,
    tv: Option<&TvRegularizer>,
) -> Vec<u8> {
    use zensim_train_core::hybrid_head as hh;

    let n_hidden = hyperparams.n_hidden;
    let leaky = hyperparams.leaky_alpha;

    assert!(!groups.is_empty(), "need at least one training group");
    for g in groups.iter() {
        assert_eq!(
            g.human_scores.len(),
            g.features.len(),
            "{}: scores/features length mismatch",
            g.name
        );
        assert!(
            g.features.iter().all(|f| f.len() == n_features),
            "{}: feature length mismatch",
            g.name
        );
    }
    let nin_on = hyperparams.norm_in_norm_weight > 0.0;
    if nin_on {
        assert!(
            hyperparams.minibatch_size >= 16,
            "hybrid_head + NiN composition: K (minibatch_size) must be ≥16 for stable batch statistics; got {}",
            hyperparams.minibatch_size
        );
    }

    let train_total: f64 = groups.iter().map(|g| g.train_weight).sum();
    assert!(
        train_total > 0.0,
        "no training groups (all train_weight == 0)"
    );

    let train_indices: Vec<usize> = groups
        .iter()
        .enumerate()
        .filter_map(|(i, g)| if g.train_weight > 0.0 { Some(i) } else { None })
        .collect();
    let val_indices: Vec<usize> = groups
        .iter()
        .enumerate()
        .filter_map(|(i, g)| {
            if g.validation_weight > 0.0 {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    let log_line = |msg: &str, log: &mut Vec<String>| {
        eprintln!("{msg}");
        log.push(msg.to_string());
    };

    log_line(
        &format!(
            "MLP train (HYBRID-HEAD): arch=[{n_features} → {n_hidden} (LeakyReLU α={leaky}) \
             → (h · rank_w + rank_b) ⊕α (pool[μ,σ,max,p_6] · reducer + b)], val_policy={:?}",
            hyperparams.validation_policy,
        ),
        log,
    );
    for (i, g) in groups.iter().enumerate() {
        let role = match (g.train_weight > 0.0, g.validation_weight > 0.0) {
            (true, true) => "train+val",
            (true, false) => "train",
            (false, true) => "val-only",
            (false, false) => "report",
        };
        log_line(
            &format!(
                "  {role:>9} group {i}: '{}' n={} train_w={:.3} val_w={:.3} withinref={} loss={:?}",
                g.name,
                g.features.len(),
                g.train_weight,
                g.validation_weight,
                g.ref_ids.is_some(),
                g.loss_mode,
            ),
            log,
        );
    }

    let (scaler_mean, scaler_scale) =
        compute_scaler_from_groups(groups, &train_indices, n_features);

    let std_features =
        standardize_groups_releasing_raw(groups, n_features, &scaler_mean, &scaler_scale);

    let tv_std: Option<Vec<f64>> = tv.map(|t| {
        assert_eq!(
            t.n_features_check(),
            n_features,
            "TV features dimensionality must match training features"
        );
        let n_rows = t.features.len();
        let mut buf = vec![0.0f64; n_rows * n_features];
        for (i, f) in t.features.iter().enumerate() {
            for d in 0..n_features {
                buf[i * n_features + d] = (f[d] - scaler_mean[d]) / scaler_scale[d].max(1e-12);
            }
        }
        buf
    });

    // Init from HybridHeadModel::new — both heads + α_logit init.
    let init_model = hh::HybridHeadModel::new(n_features, n_hidden, hyperparams.seed);
    let mut w1 = init_model.w1.clone();
    let mut b1 = init_model.b1.clone();
    let mut rank_w = init_model.rank_w.clone();
    let mut rank_b = init_model.rank_b;
    let mut reducer_w = init_model.reducer_w;
    let mut reducer_b = init_model.reducer_b;
    let mut alpha_logit = init_model.alpha_logit;

    let mut rng = SplitMix64::new(sampling::sample_stream_seed(
        hyperparams.sample_seed.unwrap_or(hyperparams.seed),
    ));

    // Adam slot sizes: w2 = n_hidden (rank_w) + 4 (reducer_w) + 1 (α_logit).
    // b2 = 2 (rank_b, reducer_b).
    let n_w2 = n_hidden + 4 + 1;
    let n_b2 = 2;
    let mut adam = AdamState::new(w1.len(), b1.len(), n_w2, n_b2);

    let start = Instant::now();
    let mut best_val_score = f64::NEG_INFINITY;
    let mut best_bake: Option<Vec<u8>> = None;
    let mut stale_epochs = 0usize;

    let cdf: Vec<f64> = {
        let mut cum = 0.0;
        train_indices
            .iter()
            .map(|&gi| {
                cum += groups[gi].train_weight;
                cum / train_total
            })
            .collect()
    };

    let needs_row_boost = hyperparams.low_q_boost != 1.0
        || hyperparams.mid_q_boost != 1.0
        || hyperparams.high_q_boost != 1.0;
    let per_row_cdfs: Vec<Option<Vec<f64>>> = train_indices
        .iter()
        .map(|&gi| {
            if !needs_row_boost {
                return None;
            }
            let g = &groups[gi];
            let mut cum = 0.0;
            let raw: Vec<f64> = g
                .human_scores
                .iter()
                .map(|&s| {
                    let mut w = 1.0;
                    if hyperparams.low_q_boost != 1.0 {
                        if s < 50.0 {
                            w *= hyperparams.low_q_boost;
                        } else if s < 65.0 {
                            w *= hyperparams.low_q_boost.sqrt();
                        }
                    }
                    if hyperparams.mid_q_boost != 1.0 && (50.0..90.0).contains(&s) {
                        w *= hyperparams.mid_q_boost;
                    }
                    if hyperparams.high_q_boost != 1.0 && s >= 90.0 {
                        w *= hyperparams.high_q_boost;
                    }
                    cum += w;
                    cum
                })
                .collect();
            let total = *raw.last().unwrap_or(&1.0);
            Some(raw.into_iter().map(|c| c / total).collect())
        })
        .collect();
    // Within-ref pair buckets, parallel to `per_row_cdfs`. `Some` only for
    // groups the caller opted in via `TrainingGroup::ref_ids` (the binary
    // sets that only when the group spec asked for within-ref). Groups
    // without it keep the uniform cross-image draw, byte-for-byte.
    let ref_buckets: Vec<Option<RefBuckets>> = train_indices
        .iter()
        .map(|&gi| groups[gi].ref_ids.and_then(RefBuckets::build))
        .collect();
    // Row counts by position-in-train_indices, for the owner draw step.
    // `FeatureRows::len` is cached at construction (it stays correct after
    // standardization takes the buffer), so hoisting it out of the hot
    // loop is exact, not an approximation.
    let row_counts: Vec<usize> = train_indices
        .iter()
        .map(|&gi| groups[gi].features.len())
        .collect();
    // Opt-in sample-sequence digest: the faithfulness proof that
    // `sampling::simulate` replays THIS run's draws. Off by default, so a
    // normal run is byte- and cost-identical.
    let mut sample_digest = std::env::var("ZENSIM_SAMPLE_DIGEST")
        .ok()
        .filter(|v| v == "1")
        .map(|_| sampling::SampleSequenceDigest::new());
    // STRATEGY stratified row-A bands. This used to be built in ONE of the
    // four training loops, so `--stratified-bands` was a silent no-op on
    // every other path — including the standard path every board bake
    // trained through. Empty when the flag is 0, which keeps the default
    // byte-identical.
    let strat_bands: Vec<Vec<Vec<usize>>> = if hyperparams.stratified_bands > 0 {
        train_indices
            .iter()
            .map(|&gi| strategy::build_bands(groups[gi].human_scores, hyperparams.stratified_bands))
            .collect()
    } else {
        Vec::new()
    };
    // Say so out loud: which pairing mode a group trained under changes what
    // the bake learned, and it must never be a silent default.
    for (pos, rb) in ref_buckets.iter().enumerate() {
        if let Some(rb) = rb {
            println!(
                "  {}: WITHIN-REF pairs over {} refs ({} rows usable)",
                groups[train_indices[pos]].name,
                rb.n_refs(),
                rb.n_rows(),
            );
        }
    }

    let k = hyperparams.minibatch_size.max(1);
    if hyperparams.parallel_batch && k > 1 {
        log_line(
            "hybrid_head: parallel-batch flag ignored (sequential mini-batch path).",
            log,
        );
    }
    log_line(
        &format!(
            "hybrid_head: ENABLED — sigmoid-bounded α-mix init α_logit=0 (α=0.5), K={k} sequential, NiN={}",
            if nin_on {
                format!(
                    "w={:.3} p={:.2} q={:.2}",
                    hyperparams.norm_in_norm_weight,
                    hyperparams.norm_in_norm_p,
                    hyperparams.norm_in_norm_q
                )
            } else {
                "off".to_string()
            }
        ),
        log,
    );

    // Adam-step helper: pack/unpack our (rank_w, reducer_w, alpha_logit,
    // rank_b, reducer_b) into the Adam w2/b2 slots and step.
    let do_adam_step = |adam: &mut AdamState,
                        w1: &mut Vec<f64>,
                        b1: &mut Vec<f64>,
                        rank_w: &mut Vec<f64>,
                        rank_b: &mut f64,
                        reducer_w: &mut [f64; 4],
                        reducer_b: &mut f64,
                        alpha_logit: &mut f64,
                        lr: f64,
                        n_hidden: usize| {
        let mut w2_vec = vec![0.0f64; n_hidden + 4 + 1];
        w2_vec[..n_hidden].copy_from_slice(&rank_w[..n_hidden]);
        w2_vec[n_hidden..n_hidden + 4].copy_from_slice(&reducer_w[..]);
        w2_vec[n_hidden + 4] = *alpha_logit;
        let mut b2_vec = vec![*rank_b, *reducer_b];
        adam.step(w1, b1, &mut w2_vec, &mut b2_vec, lr);
        rank_w[..n_hidden].copy_from_slice(&w2_vec[..n_hidden]);
        reducer_w.copy_from_slice(&w2_vec[n_hidden..n_hidden + 4]);
        *alpha_logit = w2_vec[n_hidden + 4];
        *rank_b = b2_vec[0];
        *reducer_b = b2_vec[1];
    };

    // NiN buffer — accumulates K hybrid-head pair forwards (or None for
    // dropped pairs) when nin_on. Reset each epoch (final-flush at end).
    let mut nin_buffer: Vec<Option<HybridPairForward<'_>>> = if nin_on {
        Vec::with_capacity(k)
    } else {
        Vec::new()
    };

    for epoch in 0..hyperparams.n_epochs {
        let lr = hyperparams.initial_lr
            * 0.5
            * (1.0 + (std::f64::consts::PI * (epoch % 50) as f64 / 50.0).cos());

        let mut total_loss = 0.0f64;
        let mut n_steps = 0u64;
        let mut steps_since_adam = 0u64;

        for _ in 0..hyperparams.pairs_per_epoch {
            // Draw via THE owner (`sampling::draw_pair`) — see that module
            // for why the RNG consumption pattern is a wire contract.
            let drawn = sampling::draw_pair(
                &sampling::PairDrawCtx {
                    cdf: &cdf,
                    row_counts: &row_counts,
                    per_row_cdfs: &per_row_cdfs,
                    ref_buckets: &ref_buckets,
                    strat_bands: &strat_bands,
                },
                &mut rng,
            );
            if let Some(d) = sample_digest.as_mut() {
                d.push(drawn);
            }
            let sampling::Draw::Pair { train_pos, ia, ib } = drawn else {
                continue;
            };
            let g_idx = train_indices[train_pos];
            let g = &groups[g_idx];

            let g_feats = &std_features[g_idx];
            let xa = &g_feats[ia * n_features..(ia + 1) * n_features];
            let xb = &g_feats[ib * n_features..(ib + 1) * n_features];

            let (ya, ya_rank, ya_pool, alpha_a, ha_pre, ha, sa, max_a) = hh::forward_hybrid_head(
                xa,
                &w1,
                &b1,
                &rank_w,
                rank_b,
                &reducer_w,
                reducer_b,
                alpha_logit,
                n_features,
                n_hidden,
                leaky,
            );
            let (yb, yb_rank, yb_pool, alpha_b, hb_pre, hb, sb, max_b) = hh::forward_hybrid_head(
                xb,
                &w1,
                &b1,
                &rank_w,
                rank_b,
                &reducer_w,
                reducer_b,
                alpha_logit,
                n_features,
                n_hidden,
                leaky,
            );

            let mos_a = g.human_scores[ia];
            let mos_b = g.human_scores[ib];
            let target = (mos_a - mos_b).signum();
            if target == 0.0 {
                if nin_on {
                    nin_buffer.push(None);
                    if nin_buffer.len() >= k {
                        flush_hybrid_head_nin_batch(
                            &mut nin_buffer,
                            &mut w1,
                            &mut b1,
                            &mut rank_w,
                            &mut rank_b,
                            &mut reducer_w,
                            &mut reducer_b,
                            &mut alpha_logit,
                            &mut adam,
                            n_features,
                            n_hidden,
                            leaky,
                            hyperparams.l2_lambda,
                            hyperparams.norm_in_norm_weight,
                            hyperparams.norm_in_norm_p,
                            hyperparams.norm_in_norm_q,
                            lr,
                            &mut total_loss,
                            &mut n_steps,
                            &do_adam_step,
                        );
                    }
                }
                continue;
            }
            if hyperparams.pwrc_pair_weight
                && hyperparams.pwrc_sensory_threshold > 0.0
                && (mos_a - mos_b).abs() < hyperparams.pwrc_sensory_threshold
            {
                if nin_on {
                    nin_buffer.push(None);
                    if nin_buffer.len() >= k {
                        flush_hybrid_head_nin_batch(
                            &mut nin_buffer,
                            &mut w1,
                            &mut b1,
                            &mut rank_w,
                            &mut rank_b,
                            &mut reducer_w,
                            &mut reducer_b,
                            &mut alpha_logit,
                            &mut adam,
                            n_features,
                            n_hidden,
                            leaky,
                            hyperparams.l2_lambda,
                            hyperparams.norm_in_norm_weight,
                            hyperparams.norm_in_norm_p,
                            hyperparams.norm_in_norm_q,
                            lr,
                            &mut total_loss,
                            &mut n_steps,
                            &do_adam_step,
                        );
                    }
                }
                continue;
            }
            let pair_weight = if hyperparams.pwrc_pair_weight {
                pwrc_pair_weight(mos_a, mos_b, hyperparams.pwrc_band_weights.as_deref())
            } else {
                1.0
            };
            let pred_diff = yb - ya;
            let z = -target * pred_diff;
            let loss_raw = if z > 50.0 {
                z
            } else if z < -50.0 {
                0.0
            } else {
                (z.exp() + 1.0).ln()
            };
            total_loss += loss_raw * pair_weight;
            n_steps += 1;

            let sig_z = 1.0 / (1.0 + (-z).exp());
            let dl_d_pred_diff = -target * sig_z * pair_weight;
            let dl_dya_rn = -dl_d_pred_diff;
            let dl_dyb_rn = dl_d_pred_diff;

            // NiN-aware path: buffer the per-pair forward; flush the
            // K-pair batch through `flush_hybrid_head_nin_batch` which
            // computes NiN over all 2K predictions and routes per-
            // prediction grad through the hybrid-head chain rule
            // (rank + pool + sigmoid α).
            if nin_on {
                nin_buffer.push(Some(HybridPairForward {
                    xa,
                    xb,
                    ya,
                    yb,
                    ya_rank,
                    yb_rank,
                    ya_pool,
                    yb_pool,
                    alpha_a,
                    alpha_b,
                    ha_pre,
                    ha,
                    hb_pre,
                    hb,
                    sa,
                    sb,
                    max_a,
                    max_b,
                    dl_dya_rn,
                    dl_dyb_rn,
                    mos_a,
                    mos_b,
                }));
                if nin_buffer.len() >= k {
                    flush_hybrid_head_nin_batch(
                        &mut nin_buffer,
                        &mut w1,
                        &mut b1,
                        &mut rank_w,
                        &mut rank_b,
                        &mut reducer_w,
                        &mut reducer_b,
                        &mut alpha_logit,
                        &mut adam,
                        n_features,
                        n_hidden,
                        leaky,
                        hyperparams.l2_lambda,
                        hyperparams.norm_in_norm_weight,
                        hyperparams.norm_in_norm_p,
                        hyperparams.norm_in_norm_q,
                        lr,
                        &mut total_loss,
                        &mut n_steps,
                        &do_adam_step,
                    );
                }
                // TV regularizer is skipped on the NiN-active path
                // (mirrors pool-head trainer; the V_22/V_24 recipe
                // doesn't use TV).
                continue;
            }

            let dl_dya = dl_dya_rn;
            let dl_dyb = dl_dyb_rn;
            steps_since_adam += 1;

            // Per-pair gradient accumulators (rank_w / rank_b / α_logit
            // need their own grad slots before folding into Adam).
            let mut g_rank_w_buf = vec![0.0f64; n_hidden];
            let mut g_rank_b_buf = 0.0f64;
            let mut g_red_w: [f64; 4] = [0.0; 4];
            let mut g_red_b: f64 = 0.0;
            let mut g_alpha_logit: f64 = 0.0;

            hh::backprop_step_hybrid_head(
                xa,
                &ha_pre,
                &ha,
                &sa,
                max_a,
                ya_rank,
                ya_pool,
                alpha_a,
                dl_dya,
                &rank_w,
                &reducer_w,
                &mut adam.gw1,
                &mut adam.gb1,
                &mut g_rank_w_buf,
                &mut g_rank_b_buf,
                &mut g_red_w,
                &mut g_red_b,
                &mut g_alpha_logit,
                n_features,
                n_hidden,
                leaky,
            );
            hh::backprop_step_hybrid_head(
                xb,
                &hb_pre,
                &hb,
                &sb,
                max_b,
                yb_rank,
                yb_pool,
                alpha_b,
                dl_dyb,
                &rank_w,
                &reducer_w,
                &mut adam.gw1,
                &mut adam.gb1,
                &mut g_rank_w_buf,
                &mut g_rank_b_buf,
                &mut g_red_w,
                &mut g_red_b,
                &mut g_alpha_logit,
                n_features,
                n_hidden,
                leaky,
            );

            if hyperparams.l2_lambda > 0.0 {
                let l2 = hyperparams.l2_lambda;
                for (g, &w) in adam.gw1.iter_mut().zip(w1.iter()) {
                    *g += l2 * w;
                }
                for j in 0..n_hidden {
                    g_rank_w_buf[j] += l2 * rank_w[j];
                }
                for kk in 0..4 {
                    g_red_w[kk] += l2 * reducer_w[kk];
                }
                // α_logit unregularized.
            }

            // Fold per-pair grads into Adam w2/b2 slots.
            for j in 0..n_hidden {
                adam.gw2[j] += g_rank_w_buf[j];
            }
            for kk in 0..4 {
                adam.gw2[n_hidden + kk] += g_red_w[kk];
            }
            adam.gw2[n_hidden + 4] += g_alpha_logit;
            adam.gb2[0] += g_rank_b_buf;
            adam.gb2[1] += g_red_b;

            if k == 1 || steps_since_adam >= k as u64 {
                do_adam_step(
                    &mut adam,
                    &mut w1,
                    &mut b1,
                    &mut rank_w,
                    &mut rank_b,
                    &mut reducer_w,
                    &mut reducer_b,
                    &mut alpha_logit,
                    lr,
                    n_hidden,
                );
                steps_since_adam = 0;
            }

            // TV regularizer (per-curve adjacent-q monotonicity).
            // Skipped on the NiN-active path (mirrors pool-head trainer).
            if !nin_on
                && let (Some(tv_cfg), Some(tv_buf)) = (tv, tv_std.as_ref())
                && tv_cfg.weight > 0.0
                && tv_cfg.apply_every > 0
                && n_steps.is_multiple_of(tv_cfg.apply_every as u64)
                && !tv_cfg.pairs.is_empty()
            {
                let flat_scale = tv_cfg.weight / tv_cfg.batch.max(1) as f64;
                let per_band_active = tv_cfg.band_id.is_some() && tv_cfg.band_weights.is_some();
                let mut tv_steps_since_adam = 0u64;
                for tv_iter in 0..tv_cfg.batch {
                    let pair_idx = (rng.next_u64() as usize) % tv_cfg.pairs.len();
                    let (lo, hi) = tv_cfg.pairs[pair_idx];
                    let scale = if per_band_active {
                        let band = tv_cfg.band_id.as_ref().unwrap()[pair_idx] as usize;
                        let bw = tv_cfg.band_weights.as_ref().unwrap()[band];
                        bw / tv_cfg.batch.max(1) as f64
                    } else {
                        flat_scale
                    };
                    let xlo = &tv_buf[lo * n_features..(lo + 1) * n_features];
                    let xhi = &tv_buf[hi * n_features..(hi + 1) * n_features];
                    let (y_lo, _, _, _, h_lo_pre, h_lo, s_lo, max_lo) = hh::forward_hybrid_head(
                        xlo,
                        &w1,
                        &b1,
                        &rank_w,
                        rank_b,
                        &reducer_w,
                        reducer_b,
                        alpha_logit,
                        n_features,
                        n_hidden,
                        leaky,
                    );
                    let (y_hi, _, _, _, h_hi_pre, h_hi, s_hi, max_hi) = hh::forward_hybrid_head(
                        xhi,
                        &w1,
                        &b1,
                        &rank_w,
                        rank_b,
                        &reducer_w,
                        reducer_b,
                        alpha_logit,
                        n_features,
                        n_hidden,
                        leaky,
                    );
                    let viol = y_hi - y_lo;
                    if viol <= 0.0 {
                        continue;
                    }
                    let mut tv_rank_w_buf = vec![0.0f64; n_hidden];
                    let mut tv_rank_b_buf = 0.0f64;
                    let mut tv_red_w: [f64; 4] = [0.0; 4];
                    let mut tv_red_b: f64 = 0.0;
                    let mut tv_alpha_logit: f64 = 0.0;
                    // hi: gradient +scale on y. lo: -scale.
                    let alpha_hi_eff = {
                        let xc = alpha_logit.clamp(-20.0, 20.0);
                        1.0 / (1.0 + (-xc).exp())
                    };
                    let y_hi_rank = h_hi
                        .iter()
                        .zip(rank_w.iter())
                        .map(|(&h, &w)| h * w)
                        .sum::<f64>()
                        + rank_b;
                    let y_hi_pool = s_hi[0] * reducer_w[0]
                        + s_hi[1] * reducer_w[1]
                        + s_hi[2] * reducer_w[2]
                        + s_hi[3] * reducer_w[3]
                        + reducer_b;
                    hh::backprop_step_hybrid_head(
                        xhi,
                        &h_hi_pre,
                        &h_hi,
                        &s_hi,
                        max_hi,
                        y_hi_rank,
                        y_hi_pool,
                        alpha_hi_eff,
                        scale,
                        &rank_w,
                        &reducer_w,
                        &mut adam.gw1,
                        &mut adam.gb1,
                        &mut tv_rank_w_buf,
                        &mut tv_rank_b_buf,
                        &mut tv_red_w,
                        &mut tv_red_b,
                        &mut tv_alpha_logit,
                        n_features,
                        n_hidden,
                        leaky,
                    );
                    let alpha_lo_eff = alpha_hi_eff;
                    let y_lo_rank = h_lo
                        .iter()
                        .zip(rank_w.iter())
                        .map(|(&h, &w)| h * w)
                        .sum::<f64>()
                        + rank_b;
                    let y_lo_pool = s_lo[0] * reducer_w[0]
                        + s_lo[1] * reducer_w[1]
                        + s_lo[2] * reducer_w[2]
                        + s_lo[3] * reducer_w[3]
                        + reducer_b;
                    hh::backprop_step_hybrid_head(
                        xlo,
                        &h_lo_pre,
                        &h_lo,
                        &s_lo,
                        max_lo,
                        y_lo_rank,
                        y_lo_pool,
                        alpha_lo_eff,
                        -scale,
                        &rank_w,
                        &reducer_w,
                        &mut adam.gw1,
                        &mut adam.gb1,
                        &mut tv_rank_w_buf,
                        &mut tv_rank_b_buf,
                        &mut tv_red_w,
                        &mut tv_red_b,
                        &mut tv_alpha_logit,
                        n_features,
                        n_hidden,
                        leaky,
                    );
                    for j in 0..n_hidden {
                        adam.gw2[j] += tv_rank_w_buf[j];
                    }
                    for kk in 0..4 {
                        adam.gw2[n_hidden + kk] += tv_red_w[kk];
                    }
                    adam.gw2[n_hidden + 4] += tv_alpha_logit;
                    adam.gb2[0] += tv_rank_b_buf;
                    adam.gb2[1] += tv_red_b;
                    tv_steps_since_adam += 1;
                    let is_last_tv = tv_iter + 1 == tv_cfg.batch;
                    if k == 1 || tv_steps_since_adam >= k as u64 || is_last_tv {
                        if tv_steps_since_adam > 0 {
                            do_adam_step(
                                &mut adam,
                                &mut w1,
                                &mut b1,
                                &mut rank_w,
                                &mut rank_b,
                                &mut reducer_w,
                                &mut reducer_b,
                                &mut alpha_logit,
                                lr,
                                n_hidden,
                            );
                        }
                        tv_steps_since_adam = 0;
                    }
                    let _ = y_hi;
                    let _ = y_lo;
                }
            }
        }

        // Final-flush leftover K>1 accumulated gradient (RankNet path).
        if k > 1 && !nin_on && steps_since_adam > 0 {
            do_adam_step(
                &mut adam,
                &mut w1,
                &mut b1,
                &mut rank_w,
                &mut rank_b,
                &mut reducer_w,
                &mut reducer_b,
                &mut alpha_logit,
                lr,
                n_hidden,
            );
        }

        // Final-flush leftover NiN buffer if any surviving pairs ≥ 16.
        // Mirrors the pool-head trainer's final-flush at the same
        // threshold (NiN needs ≥16 predictions for stable batch
        // mean/std; smaller batches are dropped).
        if nin_on && !nin_buffer.is_empty() {
            let surviving = nin_buffer.iter().filter(|p| p.is_some()).count();
            if surviving >= 16 {
                flush_hybrid_head_nin_batch(
                    &mut nin_buffer,
                    &mut w1,
                    &mut b1,
                    &mut rank_w,
                    &mut rank_b,
                    &mut reducer_w,
                    &mut reducer_b,
                    &mut alpha_logit,
                    &mut adam,
                    n_features,
                    n_hidden,
                    leaky,
                    hyperparams.l2_lambda,
                    hyperparams.norm_in_norm_weight,
                    hyperparams.norm_in_norm_p,
                    hyperparams.norm_in_norm_q,
                    lr,
                    &mut total_loss,
                    &mut n_steps,
                    &do_adam_step,
                );
            } else {
                nin_buffer.clear();
            }
        }

        let avg_loss = if n_steps > 0 {
            total_loss / n_steps as f64
        } else {
            0.0
        };

        if epoch % hyperparams.log_every == 0 || epoch == hyperparams.n_epochs - 1 {
            let alpha_eff = {
                let xc = alpha_logit.clamp(-20.0, 20.0);
                1.0 / (1.0 + (-xc).exp())
            };
            let agg_mode = hyperparams.val_aggregate;
            let group_panels: Vec<crate::panel::LightPanel> = groups
                .iter()
                .enumerate()
                .map(|(gi, g)| {
                    let preds = predict_group_hybrid_head(
                        &std_features[gi],
                        g.features.len(),
                        n_features,
                        &w1,
                        &b1,
                        &rank_w,
                        rank_b,
                        &reducer_w,
                        reducer_b,
                        alpha_logit,
                        n_hidden,
                        leaky,
                    );
                    let neg_preds: Vec<f64> = preds.iter().map(|&p| -p).collect();
                    crate::panel::compute_light_panel_subsampled(&neg_preds, g.human_scores)
                })
                .collect();

            let group_scores: Vec<f64> =
                group_panels.iter().map(|p| p.aggregate(agg_mode)).collect();

            let val_score = if val_indices.is_empty() {
                group_scores.iter().sum::<f64>() / group_scores.len() as f64
            } else {
                match hyperparams.validation_policy {
                    ValidationPolicy::Mean => {
                        let total: f64 = val_indices
                            .iter()
                            .map(|&i| groups[i].validation_weight)
                            .sum();
                        val_indices
                            .iter()
                            .map(|&i| group_scores[i] * groups[i].validation_weight)
                            .sum::<f64>()
                            / total
                    }
                    ValidationPolicy::Min => val_indices
                        .iter()
                        .map(|&i| group_scores[i])
                        .fold(f64::INFINITY, f64::min),
                    ValidationPolicy::Goals => {
                        let gs = compute_goal_scores(&group_panels, groups, None, None);
                        gs.aggregate()
                    }
                }
            };

            let elapsed = start.elapsed().as_secs_f64();
            let per_group = group_panels
                .iter()
                .zip(groups.iter())
                .map(|(p, g)| {
                    format!(
                        "{}: srocc={:.4} plcc={:.4} pwrc={:.4}",
                        g.name, p.srocc, p.plcc, p.pwrc
                    )
                })
                .collect::<Vec<_>>()
                .join(" | ");
            log_line(
                &format!(
                    "  epoch {epoch:>3} | lr={lr:.5} | loss={avg_loss:.4} | val({agg_mode})={val_score:.4} (best={best_val_score:.4}) | α={alpha_eff:.3} (logit={alpha_logit:+.3}) reducer_w=[μ={:.3},σ={:.3},max={:.3},p6={:.3}] | {per_group} | t={elapsed:.1}s",
                    reducer_w[0], reducer_w[1], reducer_w[2], reducer_w[3],
                ),
                log,
            );

            if val_score > best_val_score {
                best_val_score = val_score;
                stale_epochs = 0;
                let model = hh::HybridHeadModel {
                    scaler_mean: scaler_mean.clone(),
                    scaler_scale: scaler_scale.clone(),
                    w1: w1.clone(),
                    b1: b1.clone(),
                    rank_w: rank_w.clone(),
                    rank_b,
                    reducer_w,
                    reducer_b,
                    alpha_logit,
                    n_hidden,
                    n_features,
                };
                best_bake = Some(hh::bake_hybrid_head_v3(&model));
            } else {
                stale_epochs += hyperparams.log_every;
                if hyperparams.early_stop_patience > 0
                    && stale_epochs >= hyperparams.early_stop_patience
                {
                    log_line(
                        &format!(
                            "  early stop at epoch {epoch} (no validation improvement for {stale_epochs} epochs)"
                        ),
                        log,
                    );
                    break;
                }
            }
        }
    }

    let alpha_final = {
        let xc = alpha_logit.clamp(-20.0, 20.0);
        1.0 / (1.0 + (-xc).exp())
    };
    record_best_val(best_val_score);
    log_line(
        &format!(
            "MLP train (HYBRID-HEAD): best validation SROCC = {best_val_score:.4} | final α={alpha_final:.4} (logit={alpha_logit:+.4}) reducer_w=[μ={:.3},σ={:.3},max={:.3},p6={:.3}]",
            reducer_w[0], reducer_w[1], reducer_w[2], reducer_w[3],
        ),
        log,
    );
    // Faithfulness hook: `sampling::simulate` must reproduce this hash.
    if let Some(d) = sample_digest.as_ref() {
        println!("ZENSIM_SAMPLE_DIGEST {}", d.hex());
    }
    best_bake.unwrap_or_else(|| {
        let model = hh::HybridHeadModel {
            scaler_mean: scaler_mean.clone(),
            scaler_scale: scaler_scale.clone(),
            w1: w1.clone(),
            b1: b1.clone(),
            rank_w: rank_w.clone(),
            rank_b,
            reducer_w,
            reducer_b,
            alpha_logit,
            n_hidden,
            n_features,
        };
        hh::bake_hybrid_head_v3(&model)
    })
}

/// Predict hybrid-head outputs for every row in a flat (n_pairs ×
/// n_features) standardized feature buffer.
#[allow(clippy::too_many_arguments)]
fn predict_group_hybrid_head(
    std_x: &[f64],
    n_pairs: usize,
    n_features: usize,
    w1: &[f64],
    b1: &[f64],
    rank_w: &[f64],
    rank_b: f64,
    reducer_w: &[f64; 4],
    reducer_b: f64,
    alpha_logit: f64,
    n_hidden: usize,
    leaky: f64,
) -> Vec<f64> {
    use zensim_train_core::hybrid_head as hh;
    (0..n_pairs)
        .map(|i| {
            let xi = &std_x[i * n_features..(i + 1) * n_features];
            let (y, _, _, _, _, _, _, _) = hh::forward_hybrid_head(
                xi,
                w1,
                b1,
                rank_w,
                rank_b,
                reducer_w,
                reducer_b,
                alpha_logit,
                n_features,
                n_hidden,
                leaky,
            );
            y
        })
        .collect()
}

/// Per-pair forward state retained across an NiN-aware pool-head
/// mini-batch (Li 2020 hybrid loss). The flush pass walks every
/// `Some(_)` entry, computes NiN over their 2N predictions, then
/// routes per-prediction grads through the pool-head chain rule.
/// `None` entries are dropped pairs (target=0 or PWRC sensory-
/// threshold violations) — kept in the buffer to preserve the
/// per-pair RNG draw schedule, but they contribute neither to NiN
/// statistics nor backprop.
pub(crate) struct PoolPairForward<'a> {
    pub(crate) xa: &'a [f64],
    pub(crate) xb: &'a [f64],
    pub(crate) ya: f64,
    pub(crate) yb: f64,
    pub(crate) ha_pre: Vec<f64>,
    pub(crate) ha: Vec<f64>,
    pub(crate) hb_pre: Vec<f64>,
    pub(crate) hb: Vec<f64>,
    pub(crate) sa: [f64; 4],
    pub(crate) sb: [f64; 4],
    pub(crate) max_a: usize,
    pub(crate) max_b: usize,
    pub(crate) dl_dya_rn: f64,
    pub(crate) dl_dyb_rn: f64,
    pub(crate) mos_a: f64,
    pub(crate) mos_b: f64,
}

/// Flush a NiN-aware pool-head mini-batch. Computes NiN over the
/// 2N surviving predictions, scatters per-prediction grad back to
/// each pair's `dl_dya/dl_dyb` (combined with the cached RankNet
/// contribution), and routes through `backprop_step_pool_head` —
/// accumulating into `adam.gw1/gb1/gw2/gb2` (the latter two repurposed
/// for the 4-wide reducer + scalar bias). Performs one Adam step
/// after the batch. L2 is applied K· λ ·w mirroring the standard
/// head's per-pair scaling (consistent with `run_minibatch_with_nin`).
///
/// The buffer is cleared on exit.
#[allow(clippy::too_many_arguments, private_interfaces)]
pub(crate) fn flush_pool_head_nin_batch(
    nin_buffer: &mut Vec<Option<PoolPairForward<'_>>>,
    w1: &mut [f64],
    b1: &mut [f64],
    reducer_w: &mut [f64; 4],
    reducer_b: &mut f64,
    adam: &mut AdamState,
    n_features: usize,
    n_hidden: usize,
    alpha: f64,
    l2_lambda: f64,
    nin_weight: f64,
    nin_p: f64,
    nin_q: f64,
    lr: f64,
    total_loss: &mut f64,
    n_steps: &mut u64,
) {
    use zensim_train_core::pool_head as ph;
    // Gather all surviving predictions into the NiN input vectors.
    let mut nin_preds: Vec<f64> = Vec::with_capacity(2 * nin_buffer.len());
    let mut nin_labels: Vec<f64> = Vec::with_capacity(2 * nin_buffer.len());
    let mut nin_idx_map: Vec<(usize, bool)> = Vec::with_capacity(2 * nin_buffer.len());
    for (pi, slot) in nin_buffer.iter().enumerate() {
        if let Some(p) = slot {
            nin_preds.push(p.ya);
            nin_labels.push(-p.mos_a);
            nin_idx_map.push((pi, false));
            nin_preds.push(p.yb);
            nin_labels.push(-p.mos_b);
            nin_idx_map.push((pi, true));
        }
    }
    let (nin_loss, nin_grad) = if nin_preds.len() >= 2 {
        loss_norm_in_norm::compute_norm_in_norm_loss_and_grad(&nin_preds, &nin_labels, nin_p, nin_q)
    } else {
        (0.0, vec![0.0; nin_preds.len()])
    };
    *total_loss += nin_weight * nin_loss;

    let mut steps_added: u64 = 0;
    let mut g_red_w: [f64; 4] = [0.0; 4];
    let mut g_red_b: f64 = 0.0;

    // Backward pass: per-prediction NiN grad + per-pair RankNet grad
    // routed through pool-head chain rule. Each pair contributes
    // 2 predictions to NiN; steps_added counts each PAIR once (so the
    // mini-batch's Adam step cadence matches the standard head's
    // `run_minibatch_with_nin`).
    for (nin_pos, &(pi, is_b)) in nin_idx_map.iter().enumerate() {
        let p = match &nin_buffer[pi] {
            Some(p) => p,
            None => continue, // unreachable per nin_idx_map construction
        };
        let nin_g = nin_grad[nin_pos] * nin_weight;
        if is_b {
            let dl_dy = p.dl_dyb_rn + nin_g;
            ph::backprop_step_pool_head(
                p.xb,
                &p.hb_pre,
                &p.hb,
                &p.sb,
                p.max_b,
                dl_dy,
                reducer_w,
                &mut adam.gw1,
                &mut adam.gb1,
                &mut g_red_w,
                &mut g_red_b,
                n_features,
                n_hidden,
                alpha,
            );
            // Count one step per pair, on the `b` half (so we don't
            // double-count). The first iteration over a pair handles
            // `a` (is_b=false); the second handles `b` (is_b=true).
            steps_added += 1;
        } else {
            let dl_dy = p.dl_dya_rn + nin_g;
            ph::backprop_step_pool_head(
                p.xa,
                &p.ha_pre,
                &p.ha,
                &p.sa,
                p.max_a,
                dl_dy,
                reducer_w,
                &mut adam.gw1,
                &mut adam.gb1,
                &mut g_red_w,
                &mut g_red_b,
                n_features,
                n_hidden,
                alpha,
            );
        }
    }

    // L2 on w1 + reducer_w, scaled by steps_added (matches
    // run_minibatch_with_nin's `scale = l2_lambda * steps_added` and
    // the standard pool-head K=1 path's per-pair L2 accumulation).
    if l2_lambda > 0.0 && steps_added > 0 {
        let scale = l2_lambda * steps_added as f64;
        for (g, &w) in adam.gw1.iter_mut().zip(w1.iter()) {
            *g += scale * w;
        }
        for kk in 0..4 {
            g_red_w[kk] += scale * reducer_w[kk];
        }
    }
    // Fold reducer grads into Adam's w2/b2 slots.
    for kk in 0..4 {
        adam.gw2[kk] += g_red_w[kk];
    }
    adam.gb2[0] += g_red_b;

    if steps_added > 0 {
        let mut r_w_vec: Vec<f64> = reducer_w.to_vec();
        let mut r_b_vec: Vec<f64> = vec![*reducer_b];
        adam.step(w1, b1, &mut r_w_vec, &mut r_b_vec, lr);
        *reducer_w = [r_w_vec[0], r_w_vec[1], r_w_vec[2], r_w_vec[3]];
        *reducer_b = r_b_vec[0];
        *n_steps += steps_added;
    }
    nin_buffer.clear();
}

/// Per-pair forward state retained across an NiN-aware hybrid-head
/// mini-batch (Li 2020 hybrid loss composed with the rank ⊕α pool
/// head). The flush pass walks every `Some(_)` entry, computes NiN
/// over their 2N predictions, then routes per-prediction grads
/// through `backprop_step_hybrid_head` (which back-routes through
/// both heads + the sigmoid-bounded learned α). `None` entries are
/// dropped pairs (target=0 or PWRC sensory-threshold violations) —
/// kept in the buffer to preserve the per-pair RNG draw schedule,
/// but they contribute neither to NiN statistics nor backprop.
pub(crate) struct HybridPairForward<'a> {
    pub(crate) xa: &'a [f64],
    pub(crate) xb: &'a [f64],
    pub(crate) ya: f64,
    pub(crate) yb: f64,
    pub(crate) ya_rank: f64,
    pub(crate) yb_rank: f64,
    pub(crate) ya_pool: f64,
    pub(crate) yb_pool: f64,
    pub(crate) alpha_a: f64,
    pub(crate) alpha_b: f64,
    pub(crate) ha_pre: Vec<f64>,
    pub(crate) ha: Vec<f64>,
    pub(crate) hb_pre: Vec<f64>,
    pub(crate) hb: Vec<f64>,
    pub(crate) sa: [f64; 4],
    pub(crate) sb: [f64; 4],
    pub(crate) max_a: usize,
    pub(crate) max_b: usize,
    pub(crate) dl_dya_rn: f64,
    pub(crate) dl_dyb_rn: f64,
    pub(crate) mos_a: f64,
    pub(crate) mos_b: f64,
}

/// Flush an NiN-aware hybrid-head mini-batch. Computes NiN over the
/// 2N surviving predictions (composite outputs `y = α·y_rank +
/// (1−α)·y_pool`), scatters per-prediction grad back to each pair's
/// `dl_dya/dl_dyb` (combined with the cached RankNet contribution),
/// and routes through `backprop_step_hybrid_head` — accumulating
/// into the hybrid Adam slots:
/// - `gw1/gb1` (layer-1)
/// - `gw2[0..n_hidden]` (rank_w), `gw2[n_hidden..n_hidden+4]` (reducer_w),
///   `gw2[n_hidden+4]` (α_logit)
/// - `gb2[0]` (rank_b), `gb2[1]` (reducer_b)
///
/// After accumulation, L2 (scaled by `steps_added`) is applied to
/// `w1 + rank_w + reducer_w` (α_logit unregularized) and the Adam
/// step is performed via `do_adam_step`. This matches the pool-head
/// trainer's flush cadence: K pairs → 1 Adam step.
#[allow(clippy::too_many_arguments)]
fn flush_hybrid_head_nin_batch<F>(
    nin_buffer: &mut Vec<Option<HybridPairForward<'_>>>,
    w1: &mut Vec<f64>,
    b1: &mut Vec<f64>,
    rank_w: &mut Vec<f64>,
    rank_b: &mut f64,
    reducer_w: &mut [f64; 4],
    reducer_b: &mut f64,
    alpha_logit: &mut f64,
    adam: &mut AdamState,
    n_features: usize,
    n_hidden: usize,
    leaky_alpha: f64,
    l2_lambda: f64,
    nin_weight: f64,
    nin_p: f64,
    nin_q: f64,
    lr: f64,
    total_loss: &mut f64,
    n_steps: &mut u64,
    do_adam_step: &F,
) where
    F: Fn(
        &mut AdamState,
        &mut Vec<f64>,
        &mut Vec<f64>,
        &mut Vec<f64>,
        &mut f64,
        &mut [f64; 4],
        &mut f64,
        &mut f64,
        f64,
        usize,
    ),
{
    use zensim_train_core::hybrid_head as hh;

    // Gather all surviving predictions into the NiN input vectors.
    // Sign convention mirrors the pool-head flush: NiN labels are
    // `-mos` (raw_distance, LOWER = more similar), so the NiN loss
    // computes against distance-space rather than score-space.
    let mut nin_preds: Vec<f64> = Vec::with_capacity(2 * nin_buffer.len());
    let mut nin_labels: Vec<f64> = Vec::with_capacity(2 * nin_buffer.len());
    let mut nin_idx_map: Vec<(usize, bool)> = Vec::with_capacity(2 * nin_buffer.len());
    for (pi, slot) in nin_buffer.iter().enumerate() {
        if let Some(p) = slot {
            nin_preds.push(p.ya);
            nin_labels.push(-p.mos_a);
            nin_idx_map.push((pi, false));
            nin_preds.push(p.yb);
            nin_labels.push(-p.mos_b);
            nin_idx_map.push((pi, true));
        }
    }
    let (nin_loss, nin_grad) = if nin_preds.len() >= 2 {
        loss_norm_in_norm::compute_norm_in_norm_loss_and_grad(&nin_preds, &nin_labels, nin_p, nin_q)
    } else {
        (0.0, vec![0.0; nin_preds.len()])
    };
    *total_loss += nin_weight * nin_loss;

    let mut steps_added: u64 = 0;
    let mut g_rank_w_buf = vec![0.0f64; n_hidden];
    let mut g_rank_b_buf = 0.0f64;
    let mut g_red_w: [f64; 4] = [0.0; 4];
    let mut g_red_b: f64 = 0.0;
    let mut g_alpha_logit: f64 = 0.0;

    // Backward pass: per-prediction NiN grad + per-pair RankNet grad
    // routed through hybrid-head chain rule. Each pair contributes
    // 2 predictions to NiN; steps_added counts each PAIR once (so
    // the mini-batch's Adam step cadence matches the pool-head
    // trainer's `flush_pool_head_nin_batch`).
    for (nin_pos, &(pi, is_b)) in nin_idx_map.iter().enumerate() {
        let p = match &nin_buffer[pi] {
            Some(p) => p,
            None => continue, // unreachable per nin_idx_map construction
        };
        let nin_g = nin_grad[nin_pos] * nin_weight;
        if is_b {
            let dl_dy = p.dl_dyb_rn + nin_g;
            hh::backprop_step_hybrid_head(
                p.xb,
                &p.hb_pre,
                &p.hb,
                &p.sb,
                p.max_b,
                p.yb_rank,
                p.yb_pool,
                p.alpha_b,
                dl_dy,
                rank_w,
                reducer_w,
                &mut adam.gw1,
                &mut adam.gb1,
                &mut g_rank_w_buf,
                &mut g_rank_b_buf,
                &mut g_red_w,
                &mut g_red_b,
                &mut g_alpha_logit,
                n_features,
                n_hidden,
                leaky_alpha,
            );
            // Count one step per pair, on the `b` half (so we don't
            // double-count). The `a` iteration handles is_b=false,
            // the `b` iteration handles is_b=true.
            steps_added += 1;
        } else {
            let dl_dy = p.dl_dya_rn + nin_g;
            hh::backprop_step_hybrid_head(
                p.xa,
                &p.ha_pre,
                &p.ha,
                &p.sa,
                p.max_a,
                p.ya_rank,
                p.ya_pool,
                p.alpha_a,
                dl_dy,
                rank_w,
                reducer_w,
                &mut adam.gw1,
                &mut adam.gb1,
                &mut g_rank_w_buf,
                &mut g_rank_b_buf,
                &mut g_red_w,
                &mut g_red_b,
                &mut g_alpha_logit,
                n_features,
                n_hidden,
                leaky_alpha,
            );
        }
    }

    // L2 on w1 + rank_w + reducer_w, scaled by steps_added (mirrors
    // the pool-head trainer's per-step L2 schedule). α_logit
    // unregularized.
    if l2_lambda > 0.0 && steps_added > 0 {
        let scale = l2_lambda * steps_added as f64;
        for (g, &w) in adam.gw1.iter_mut().zip(w1.iter()) {
            *g += scale * w;
        }
        for j in 0..n_hidden {
            g_rank_w_buf[j] += scale * rank_w[j];
        }
        for kk in 0..4 {
            g_red_w[kk] += scale * reducer_w[kk];
        }
    }

    // Fold per-pair grads into Adam w2/b2 slots (same layout as the
    // RankNet path: gw2 = [rank_w | reducer_w | α_logit], gb2 =
    // [rank_b, reducer_b]).
    for j in 0..n_hidden {
        adam.gw2[j] += g_rank_w_buf[j];
    }
    for kk in 0..4 {
        adam.gw2[n_hidden + kk] += g_red_w[kk];
    }
    adam.gw2[n_hidden + 4] += g_alpha_logit;
    adam.gb2[0] += g_rank_b_buf;
    adam.gb2[1] += g_red_b;

    if steps_added > 0 {
        do_adam_step(
            adam,
            w1,
            b1,
            rank_w,
            rank_b,
            reducer_w,
            reducer_b,
            alpha_logit,
            lr,
            n_hidden,
        );
        *n_steps += steps_added;
    }
    nin_buffer.clear();
}

/// Bake a 2-layer MLP (LeakyReLU → Identity) into ZNPR v3 bytes.
/// Converts f64 weights to f32 once and feeds them to [`bake`].
/// ZNPR v2 production is prohibited per CLAUDE.md (2026-05-15).
///
/// `dtype` controls the weight encoding for BOTH layers. F32 produces
/// the historical 355 KB-ish bake; F16 halves it; I8 (per-output f32
/// scales) cuts it to ~26 %. See V0_18 (2026-05-13) for cross-corpus
/// quality validation of I8 quant on the V0_17 weights.
#[allow(clippy::too_many_arguments)]
pub fn bake_two_layer_znpr_v3(
    scaler_mean: &[f64],
    scaler_scale: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    n_inputs: usize,
    n_hidden: usize,
    n_outputs: usize,
    dtype: WeightDtype,
    feature_transforms: Option<&[FeatureTransform]>,
    feature_transform_params: Option<&[Vec<f32>]>,
) -> Vec<u8> {
    let scaler_mean_f32: Vec<f32> = scaler_mean.iter().map(|&v| v as f32).collect();
    let scaler_scale_f32: Vec<f32> = scaler_scale.iter().map(|&v| v as f32).collect();
    let w1_f32: Vec<f32> = w1.iter().map(|&v| v as f32).collect();
    let b1_f32: Vec<f32> = b1.iter().map(|&v| v as f32).collect();
    let w2_f32: Vec<f32> = w2.iter().map(|&v| v as f32).collect();
    let b2_f32: Vec<f32> = b2.iter().map(|&v| v as f32).collect();
    let layers = [
        BakeLayer {
            in_dim: n_inputs,
            out_dim: n_hidden,
            activation: Activation::LeakyRelu,
            dtype,
            weights: &w1_f32,
            biases: &b1_f32,
        },
        BakeLayer {
            in_dim: n_hidden,
            out_dim: n_outputs,
            activation: Activation::Identity,
            dtype,
            weights: &w2_f32,
            biases: &b2_f32,
        },
    ];
    // Feature-transforms metadata (V0_20 input-shaping research). Bakers
    // omit the key entirely when every feature is `Identity` per the
    // zenpredict convention — consumers treat absence as all-identity.
    let transforms_blob: Option<String> = feature_transforms.and_then(|ts| {
        if ts.iter().all(|t| *t == FeatureTransform::Identity) {
            None
        } else {
            assert_eq!(
                ts.len(),
                n_inputs,
                "feature_transforms.len()={} must equal n_inputs={}",
                ts.len(),
                n_inputs
            );
            Some(
                ts.iter()
                    .map(|t| t.as_token())
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
        }
    });
    // Feature-transform-params metadata (V0_20 parameterized variants).
    // Newline-separated per feature; each line is comma-separated f32
    // (empty for non-parameterized features). Omitted when no feature
    // carries params.
    let params_blob: Option<String> = feature_transform_params.and_then(|params| {
        if params.iter().all(|p| p.is_empty()) {
            None
        } else {
            assert_eq!(
                params.len(),
                n_inputs,
                "feature_transform_params.len()={} must equal n_inputs={}",
                params.len(),
                n_inputs
            );
            Some(
                params
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|v| format!("{v}"))
                            .collect::<Vec<_>>()
                            .join(",")
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
        }
    });
    let mut metadata: Vec<BakeMetadataEntry<'_>> = Vec::new();
    if let Some(blob) = &transforms_blob {
        metadata.push(BakeMetadataEntry {
            key: zenpredict::keys::FEATURE_TRANSFORMS,
            kind: zenpredict::MetadataType::Utf8,
            value: blob.as_bytes(),
        });
    }
    if let Some(blob) = &params_blob {
        metadata.push(BakeMetadataEntry {
            key: zenpredict::keys::FEATURE_TRANSFORM_PARAMS,
            kind: zenpredict::MetadataType::Utf8,
            value: blob.as_bytes(),
        });
    }
    bake(&BakeRequest {
        schema_hash: 0,
        flags: 0,
        scaler_mean: &scaler_mean_f32,
        scaler_scale: &scaler_scale_f32,
        layers: &layers,
        feature_bounds: &[],
        metadata: &metadata,
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
        feature_order: None,
        output_order: None,
        compressed: false,
        hu_permutations: None,
    })
    .expect("v3 bake of 2-layer MLP")
}

/// Standardize every group's features up-front into one flat
/// `(n_rows × n_features)` f64 buffer per group, **reusing the raw buffer
/// itself when the caller allows it**.
///
/// Standardizing here rather than inside the per-step inner loop lets that
/// loop just slice into a flat buffer. Group `g`'s standardized features
/// live in `out[g]` with row `i` at `[i * n_features .. (i+1) * n_features]`.
///
/// The four trainer heads (plain / pool / hybrid / per-sample-α) each ran a
/// byte-identical copy of this loop; they now share this one. For
/// [`FeatureRows::Releasable`] the raw flat buffer is TAKEN and standardized
/// in place — same expression, same element order, only the destination
/// changes (each element is read before it is overwritten), so bakes are
/// bit-identical to the [`FeatureRows::Borrowed`] copy path and to the
/// pre-refactor trainer.
///
/// MEMORY (the reason `groups` is `&mut`): the raw rows and the standardized
/// copy are the same size, and the raw ones are dead as soon as they are
/// standardized. Standardizing in place means the run never holds two copies
/// of the feature matrix — which is what caps how many trainers fit on a
/// box. See the [`FeatureRows`] note and
/// `benchmarks/trainer_mem_release_2026-08-04.md`.
fn standardize_groups_releasing_raw(
    groups: &mut [TrainingGroup<'_>],
    n_features: usize,
    scaler_mean: &[f64],
    scaler_scale: &[f64],
) -> Vec<Vec<f64>> {
    groups
        .iter_mut()
        .map(|g| match &mut g.features {
            FeatureRows::Borrowed(rows) => {
                let mut buf = vec![0.0f64; rows.len() * n_features];
                for (i, f) in rows.iter().enumerate() {
                    for d in 0..n_features {
                        buf[i * n_features + d] =
                            (f[d] - scaler_mean[d]) / scaler_scale[d].max(1e-12);
                    }
                }
                buf
            }
            FeatureRows::Releasable {
                data,
                n_rows,
                n_features: nf,
            } => {
                assert_eq!(*nf, n_features, "Releasable width / trainer width");
                assert_eq!(data.len(), *n_rows * n_features, "Releasable buffer size");
                // Take the raw buffer and standardize IN PLACE: the raw
                // value is read, transformed, and written back to the same
                // element, so no second copy of the feature matrix ever
                // exists. `g.features.len()` keeps reporting n_rows (cached
                // in the enum), which is all the hot loop reads.
                let mut buf = std::mem::take(*data);
                for i in 0..*n_rows {
                    for d in 0..n_features {
                        let idx = i * n_features + d;
                        buf[idx] = (buf[idx] - scaler_mean[d]) / scaler_scale[d].max(1e-12);
                    }
                }
                buf
            }
        })
        .collect()
}

fn compute_scaler_from_groups(
    groups: &[TrainingGroup<'_>],
    train_indices: &[usize],
    n_features: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut count = 0u64;
    let mut mean = vec![0.0f64; n_features];
    for &gi in train_indices {
        for f in groups[gi].features.iter() {
            for d in 0..n_features {
                mean[d] += f[d];
            }
            count += 1;
        }
    }
    let n = count.max(1) as f64;
    for m in &mut mean {
        *m /= n;
    }
    let mut var = vec![0.0f64; n_features];
    for &gi in train_indices {
        for f in groups[gi].features.iter() {
            for d in 0..n_features {
                let dx = f[d] - mean[d];
                var[d] += dx * dx;
            }
        }
    }
    let std = var.iter().map(|&v| (v / n).sqrt().max(1e-8)).collect();
    (mean, std)
}

/// MLP forward pass: `x → (linear w1+b1) → LeakyReLU → (linear w2+b2) → y`.
///
/// Returns `(y, h_pre, h)` where `h_pre` is the pre-activation hidden
/// vector (needed for the LeakyReLU derivative in backprop) and `h` is
/// the post-activation hidden vector (needed for the gw2 outer-product).
///
/// Delegates to the SIMD-dispatched implementation in [`crate::simd_mlp`];
/// the scalar fallback path there is bit-identical to the historical
/// implementation. On Zen 4 / Sapphire Rapids / Ice Lake the AVX-512
/// f64x8 path runs ~1.75-1.88× faster than scalar on the production
/// (372 × 128) shape.
#[allow(clippy::too_many_arguments)]
fn forward(
    x: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    n_features: usize,
    n_hidden: usize,
    alpha: f64,
) -> (f64, Vec<f64>, Vec<f64>) {
    crate::simd_mlp::forward(x, w1, b1, w2, b2, n_features, n_hidden, alpha)
}

/// RankNet-style backprop step: accumulates `∂L/∂w1`, `∂L/∂b1`,
/// `∂L/∂w2`, `∂L/∂b2` from a single `(y, h_pre, h, dl_dy)` quadruple.
///
/// Gradient buffers are accumulated INTO (not overwritten); the caller
/// is responsible for zeroing them at the start of an Adam step. The
/// `_w1` parameter is unused (kept for caller signature compatibility).
///
/// Delegates to [`crate::simd_mlp`]; see that module's docs for the
/// dispatch tree and bit-identity guarantees.
#[allow(clippy::too_many_arguments)]
fn backprop_step(
    x: &[f64],
    h_pre: &[f64],
    h: &[f64],
    dl_dy: f64,
    _w1: &[f64],
    gw1: &mut [f64],
    gb1: &mut [f64],
    w2: &[f64],
    gw2: &mut [f64],
    gb2: &mut [f64],
    n_features: usize,
    n_hidden: usize,
    alpha: f64,
) {
    crate::simd_mlp::backprop_step(
        x, h_pre, h, dl_dy, gw1, gb1, w2, gw2, gb2, n_features, n_hidden, alpha,
    );
}

/// T8.2: per-thread gradient accumulator for rayon parallel-batch.
///
/// Each thread in the rayon par_iter computes its forward+backward
/// into a fresh `LocalGrads`, and the reduce step sums them into a
/// single accumulator that gets transferred into the Adam state's
/// gw1/gb1/gw2/gb2 before the Adam step.
#[derive(Clone)]
struct LocalGrads {
    gw1: Vec<f64>,
    gb1: Vec<f64>,
    gw2: Vec<f64>,
    gb2: Vec<f64>,
}

impl LocalGrads {
    fn zero(n_features: usize, n_hidden: usize) -> Self {
        Self {
            gw1: vec![0.0; n_features * n_hidden],
            gb1: vec![0.0; n_hidden],
            gw2: vec![0.0; n_hidden],
            gb2: vec![0.0; 1],
        }
    }

    fn add(mut self, other: Self) -> Self {
        for (a, b) in self.gw1.iter_mut().zip(other.gw1.iter()) {
            *a += b;
        }
        for (a, b) in self.gb1.iter_mut().zip(other.gb1.iter()) {
            *a += b;
        }
        for (a, b) in self.gw2.iter_mut().zip(other.gw2.iter()) {
            *a += b;
        }
        for (a, b) in self.gb2.iter_mut().zip(other.gb2.iter()) {
            *a += b;
        }
        self
    }
}

/// T8.2: process a sequentially-drawn mini-batch of `(g_idx, ia, ib)`
/// tuples through rayon::par_chunks, accumulating per-chunk
/// `LocalGrads` into a deterministic-ordered Vec, then summing
/// sequentially in chunk-index order.
///
/// **Chunked design**: per-pair allocation of one `LocalGrads` would
/// dominate runtime (each LocalGrads is `n_features * n_hidden + …`
/// f64 = ~48 KB at 372×128, and K=64 batches would allocate 3 MB per
/// batch). Instead, we split the K samples into `NUM_CHUNKS` chunks
/// (one per rayon thread), each chunk processes its samples
/// sequentially into a single thread-local LocalGrads, and the
/// `NUM_CHUNKS` chunk results are summed sequentially in source
/// order. Allocation drops from O(K) per batch to O(num_threads).
///
/// **Determinism**: chunk-source-order is preserved by `par_chunks`
/// → `collect::<Vec<_>>` (rayon docs guarantee this for `par_chunks` +
/// `map` + `collect`). Within each chunk, the sequential
/// `backprop_into` ordering is identical between runs. Across
/// chunks, the final sequential `fold` runs in chunk-index order. So
/// the FP reduce sequence is fully determined by `K + samples`,
/// independent of thread count.
///
/// Returns `(gradient_contributing_steps, accumulated_loss)`. The
/// caller is responsible for calling `adam.step(...)` after this
/// (when `steps_added > 0`) to consume the gradients and advance `t`.
///
/// Skipped pairs (target=0) contribute 0 to grads and 0 to
/// steps_added — they're functionally no-ops, just like the
/// sequential path.
#[allow(clippy::too_many_arguments)]
fn run_parallel_minibatch(
    samples: &[(usize, usize, usize)],
    groups: &[TrainingGroup<'_>],
    std_features: &[Vec<f64>],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    adam: &mut AdamState,
    n_features: usize,
    n_hidden: usize,
    alpha: f64,
    l2_lambda: f64,
    pwrc_enabled: bool,
    pwrc_sensory_threshold: f64,
    pwrc_band_weights: Option<&[f64]>,
) -> (u64, f64) {
    // Chunk size is a **fixed function of K** (not thread count) so
    // the chunk partition — and therefore the FP reduce order — is
    // identical across thread-pool sizes. Determinism requires this:
    // changing `RAYON_NUM_THREADS` must not move sample boundaries
    // between chunks (each chunk is summed sequentially within one
    // thread; cross-chunk reduce is sequential in chunk-index order).
    //
    // Chunk-size policy (alloc-amortization on 7950X, 372→128→1):
    // - LocalGrads holds gw1 = n_features × n_hidden f64 = ~380 KB at
    //   372×128 (was ~234 KB at 228×128). Each chunk's
    //   `LocalGrads::zero` allocates and zeros that block, costing
    //   roughly 50–100 µs.
    // - Forward+backward per pair is ~30–50 µs SIMD. 16 pairs ≈
    //   500–800 µs of work — comfortably amortizes the alloc.
    // - Going below 16 hurts: 4-pair chunks would be ~150 µs work vs
    //   100 µs alloc → 40% overhead. Stick with 16 as the floor.
    // - Target 16-way parallelism on the 16-core 7950X via
    //   `samples / 16`.
    // - Cap at samples.len() so tiny batches produce one chunk.
    //
    // For K=8:    1 chunk  (floor=16 > 8)         → sequential.
    // For K=32:   2 chunks (sz=16)                → 2-way parallel.
    // For K=64:   4 chunks (sz=16)                → 4-way parallel.
    // For K=256: 16 chunks (sz=16)                → fully cored.
    // For K=1024:16 chunks (sz=64)                → fully cored.
    //
    // Push K to 256+ to fully feed the 16-core pool. Smaller K trades
    // parallelism for tighter convergence on the sequential pair-grad
    // path (the empirical sweet spot for V_X bakes is K=32–64).
    let target_threads = 16usize;
    let min_chunk = 16usize;
    let chunk_size = samples
        .len()
        .div_ceil(target_threads)
        .max(min_chunk)
        .min(samples.len().max(1));

    // Each chunk produces (chunk_loss, chunk_steps, Option<LocalGrads>).
    // collect preserves source order — the K=1 boundary is not crossed
    // because samples.len() always >= K >= 2 in this branch.
    let chunk_results: Vec<(f64, u64, Option<LocalGrads>)> = samples
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut chunk_loss = 0.0f64;
            let mut chunk_steps = 0u64;
            let mut local = LocalGrads::zero(n_features, n_hidden);
            for &(g_idx, ia, ib) in chunk {
                let g_feats = &std_features[g_idx];
                let xa = &g_feats[ia * n_features..(ia + 1) * n_features];
                let xb = &g_feats[ib * n_features..(ib + 1) * n_features];
                let (ya, ha_pre, ha) = forward(xa, w1, b1, w2, b2, n_features, n_hidden, alpha);
                let (yb, hb_pre, hb) = forward(xb, w1, b1, w2, b2, n_features, n_hidden, alpha);

                let mos_a = groups[g_idx].human_scores[ia];
                let mos_b = groups[g_idx].human_scores[ib];
                let target = (mos_a - mos_b).signum();
                if target == 0.0 {
                    continue;
                }
                // PWRC sensory-threshold drop (Wu et al. 2018) — mirrors
                // the sequential path. Drops happen before grad accum
                // so chunk_steps doesn't tick (the Adam step counter
                // tracks gradient-contributing draws only).
                if pwrc_enabled
                    && pwrc_sensory_threshold > 0.0
                    && (mos_a - mos_b).abs() < pwrc_sensory_threshold
                {
                    continue;
                }
                let pair_weight = if pwrc_enabled {
                    pwrc_pair_weight(mos_a, mos_b, pwrc_band_weights)
                } else {
                    1.0
                };
                let pred_diff = yb - ya;
                let z = -target * pred_diff;
                let loss_raw = if z > 50.0 {
                    z
                } else if z < -50.0 {
                    0.0
                } else {
                    (z.exp() + 1.0).ln()
                };
                chunk_loss += loss_raw * pair_weight;
                chunk_steps += 1;

                let sig_z = 1.0 / (1.0 + (-z).exp());
                let dl_d_pred_diff = -target * sig_z * pair_weight;
                let dl_dya = -dl_d_pred_diff;
                let dl_dyb = dl_d_pred_diff;

                backprop_into(
                    &mut local, xa, &ha_pre, &ha, dl_dya, w2, n_features, n_hidden, alpha,
                );
                backprop_into(
                    &mut local, xb, &hb_pre, &hb, dl_dyb, w2, n_features, n_hidden, alpha,
                );
            }
            (
                chunk_loss,
                chunk_steps,
                if chunk_steps > 0 { Some(local) } else { None },
            )
        })
        .collect();

    // Sequential reduce in chunk-source order — deterministic FP
    // regardless of thread count.
    let mut total_loss = 0.0f64;
    let mut steps_added: u64 = 0;
    let mut acc: Option<LocalGrads> = None;
    for (chunk_loss, chunk_steps, chunk_grads) in chunk_results.into_iter() {
        if chunk_steps == 0 {
            continue;
        }
        total_loss += chunk_loss;
        steps_added += chunk_steps;
        let chunk_grads = chunk_grads.expect("steps > 0 ⇒ Some(grads)");
        acc = Some(match acc {
            None => chunk_grads,
            Some(prev) => prev.add(chunk_grads),
        });
    }

    if let Some(acc) = acc {
        // Transfer accumulated grads into AdamState's per-param
        // buffers (the sequential path adds directly into
        // adam.gw1/etc, so we do the same).
        for (a, b) in adam.gw1.iter_mut().zip(acc.gw1.iter()) {
            *a += b;
        }
        for (a, b) in adam.gb1.iter_mut().zip(acc.gb1.iter()) {
            *a += b;
        }
        for (a, b) in adam.gw2.iter_mut().zip(acc.gw2.iter()) {
            *a += b;
        }
        adam.gb2[0] += acc.gb2[0];

        // L2 regularization: in the sequential path L2 is applied
        // once per pair (so K pair updates add K*λ*w to grads). We
        // mirror that scaling here: apply L2 `steps_added` times.
        // This makes the K>1 parallel path equivalent to the K>1
        // sequential path on the same drawn samples (up to FP
        // reduce order, which we control above).
        if l2_lambda > 0.0 && steps_added > 0 {
            let scale = l2_lambda * steps_added as f64;
            for (g, &w) in adam.gw1.iter_mut().zip(w1.iter()) {
                *g += scale * w;
            }
            for (g, &w) in adam.gw2.iter_mut().zip(w2.iter()) {
                *g += scale * w;
            }
        }
    }

    (steps_added, total_loss)
}

/// Norm-in-Norm + RankNet hybrid mini-batch (Li et al. 2020,
/// arXiv:2008.03889). Runs the entire mini-batch sequentially because
/// the Norm-in-Norm loss is batch-correlated: it depends on the mean
/// and q-norm of the full set of 2K predictions, and chunking the
/// batch across rayon threads would either change the FP reduce order
/// per-thread-count (breaks determinism) or require an extra sync
/// barrier between forward and backward. Sequential within the mini-
/// batch is fast enough — K ≤ ~2048 is the practical range and at
/// 372 × 128 the per-pair forward+backward is ~50 µs / pair → ≤ 200
/// ms per batch — and produces bit-identical bake bytes regardless of
/// thread pool.
///
/// Flow:
/// 1. Forward all 2K predictions (sequential).
/// 2. Compute per-pair RankNet loss + per-pair `dl_dya, dl_dyb`
///    contributions (sequential, identical math to the legacy path).
/// 3. Compute Norm-in-Norm loss and per-prediction gradient over the
///    2K (prediction, `-MOS`) pairs in one call to
///    [`loss_norm_in_norm::compute_norm_in_norm_loss_and_grad`].
/// 4. Combine: `dl_dy_total = dl_dy_ranknet + β · nin_grad`.
/// 5. Backward all 2K predictions with the combined `dl_dy_total`
///    into a single `LocalGrads`, transferred into the AdamState.
///
/// **Sign convention**: zensim MLP outputs `raw_distance` (lower =
/// more similar), opposite of MOS. So the labels passed to NiN are
/// `-MOS`, not `+MOS`, to keep the Δ direction consistent with
/// RankNet's `(MOS_a - MOS_b).signum() · (y_a - y_b)` target.
///
/// **PWRC composition**: when `pwrc_enabled`, the RankNet term is
/// scaled per-pair by the PWRC weight. The Norm-in-Norm term is
/// not — it's a global loss over the whole batch and the per-pair
/// weight concept doesn't apply (the paper's loss is unweighted).
///
/// Returns `(gradient_contributing_steps, accumulated_loss)`. The
/// caller is responsible for calling `adam.step(...)` after this
/// (when `steps_added > 0`) to consume the gradients and advance `t`.
///
/// Skipped pairs (target=0 OR PWRC sensory drop) contribute 0 to grads
/// and 0 to steps_added — they're functionally no-ops. Their NiN
/// contribution is also dropped: the prediction is forward'd but
/// excluded from the NiN batch (otherwise the NiN batch would have
/// "ghost" predictions with no matching label gradient pull).
#[allow(clippy::too_many_arguments)]
fn run_minibatch_with_nin(
    samples: &[(usize, usize, usize)],
    groups: &[TrainingGroup<'_>],
    std_features: &[Vec<f64>],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    adam: &mut AdamState,
    n_features: usize,
    n_hidden: usize,
    alpha: f64,
    l2_lambda: f64,
    pwrc_enabled: bool,
    pwrc_sensory_threshold: f64,
    pwrc_band_weights: Option<&[f64]>,
    nin_weight: f64,
    nin_p: f64,
    nin_q: f64,
) -> (u64, f64) {
    // Per-pair forward outputs kept around for the second pass (the
    // NiN-augmented backward). `Option` because skipped pairs (target
    // = 0 or PWRC drop) contribute nothing.
    struct PairForward<'a> {
        xa: &'a [f64],
        xb: &'a [f64],
        ya: f64,
        yb: f64,
        ha_pre: Vec<f64>,
        ha: Vec<f64>,
        hb_pre: Vec<f64>,
        hb: Vec<f64>,
        // RankNet contributions (already PWRC-scaled when enabled).
        dl_dya_rn: f64,
        dl_dyb_rn: f64,
        ranknet_loss: f64,
        mos_a: f64,
        mos_b: f64,
    }
    let mut forwards: Vec<Option<PairForward<'_>>> = Vec::with_capacity(samples.len());
    let mut total_loss = 0.0f64;
    let mut steps_added: u64 = 0;

    for &(g_idx, ia, ib) in samples {
        let g_feats = &std_features[g_idx];
        let xa = &g_feats[ia * n_features..(ia + 1) * n_features];
        let xb = &g_feats[ib * n_features..(ib + 1) * n_features];
        let mos_a = groups[g_idx].human_scores[ia];
        let mos_b = groups[g_idx].human_scores[ib];
        let target = (mos_a - mos_b).signum();
        if target == 0.0 {
            forwards.push(None);
            continue;
        }
        if pwrc_enabled
            && pwrc_sensory_threshold > 0.0
            && (mos_a - mos_b).abs() < pwrc_sensory_threshold
        {
            forwards.push(None);
            continue;
        }
        let pair_weight = if pwrc_enabled {
            pwrc_pair_weight(mos_a, mos_b, pwrc_band_weights)
        } else {
            1.0
        };
        let (ya, ha_pre, ha) = forward(xa, w1, b1, w2, b2, n_features, n_hidden, alpha);
        let (yb, hb_pre, hb) = forward(xb, w1, b1, w2, b2, n_features, n_hidden, alpha);
        let pred_diff = yb - ya;
        let z = -target * pred_diff;
        let loss_raw = if z > 50.0 {
            z
        } else if z < -50.0 {
            0.0
        } else {
            (z.exp() + 1.0).ln()
        };
        let loss = loss_raw * pair_weight;
        total_loss += loss;
        steps_added += 1;

        let sig_z = 1.0 / (1.0 + (-z).exp());
        let dl_d_pred_diff = -target * sig_z * pair_weight;
        let dl_dya_rn = -dl_d_pred_diff;
        let dl_dyb_rn = dl_d_pred_diff;

        forwards.push(Some(PairForward {
            xa,
            xb,
            ya,
            yb,
            ha_pre,
            ha,
            hb_pre,
            hb,
            dl_dya_rn,
            dl_dyb_rn,
            ranknet_loss: loss,
            mos_a,
            mos_b,
        }));
    }
    // Sanity: total_loss already accumulated above. Pull the ranknet
    // loss field back out per-pair only for diagnostics; the running
    // `total_loss` is canonical.
    let _ = |p: &PairForward<'_>| p.ranknet_loss;

    // Compute Norm-in-Norm loss + per-prediction gradient over the 2N
    // surviving predictions. Labels = -MOS so the NiN gradient pulls
    // in the same `distance ↔ -MOS` direction as the RankNet term
    // (see function docstring for sign convention).
    let mut nin_preds: Vec<f64> = Vec::with_capacity(2 * forwards.len());
    let mut nin_labels: Vec<f64> = Vec::with_capacity(2 * forwards.len());
    // Map: nin_preds index → (pair_idx, is_b). We need this to scatter
    // the per-prediction gradient back to the right pair forward.
    let mut nin_idx_map: Vec<(usize, bool)> = Vec::with_capacity(2 * forwards.len());
    for (pi, p) in forwards.iter().enumerate() {
        if let Some(p) = p {
            nin_preds.push(p.ya);
            nin_labels.push(-p.mos_a);
            nin_idx_map.push((pi, false));
            nin_preds.push(p.yb);
            nin_labels.push(-p.mos_b);
            nin_idx_map.push((pi, true));
        }
    }
    let (nin_loss, nin_grad) = if nin_preds.len() >= 2 {
        loss_norm_in_norm::compute_norm_in_norm_loss_and_grad(&nin_preds, &nin_labels, nin_p, nin_q)
    } else {
        (0.0, vec![0.0; nin_preds.len()])
    };
    total_loss += nin_weight * nin_loss;

    // Backward pass: combine the per-prediction NiN gradient with the
    // per-pair RankNet `dl_dy` contributions and run backprop into a
    // single LocalGrads. Sequential — same FP reduce order regardless
    // of thread count, identical for any pool size.
    let mut local = LocalGrads::zero(n_features, n_hidden);
    for (nin_pos, &(pi, is_b)) in nin_idx_map.iter().enumerate() {
        let p = forwards[pi].as_ref().expect("None pair excluded above");
        let nin_g = nin_grad[nin_pos] * nin_weight;
        if is_b {
            let dl_dy_b = p.dl_dyb_rn + nin_g;
            backprop_into(
                &mut local, p.xb, &p.hb_pre, &p.hb, dl_dy_b, w2, n_features, n_hidden, alpha,
            );
        } else {
            let dl_dy_a = p.dl_dya_rn + nin_g;
            backprop_into(
                &mut local, p.xa, &p.ha_pre, &p.ha, dl_dy_a, w2, n_features, n_hidden, alpha,
            );
        }
    }

    // Transfer accumulated grads into AdamState's per-param buffers.
    for (a, b) in adam.gw1.iter_mut().zip(local.gw1.iter()) {
        *a += b;
    }
    for (a, b) in adam.gb1.iter_mut().zip(local.gb1.iter()) {
        *a += b;
    }
    for (a, b) in adam.gw2.iter_mut().zip(local.gw2.iter()) {
        *a += b;
    }
    adam.gb2[0] += local.gb2[0];

    // L2 regularization: in the sequential path L2 is applied once per
    // pair (so K pair updates add K*λ*w to grads). Mirror that scaling
    // here: apply L2 `steps_added` times (counts only gradient-
    // contributing pairs, matching the existing run_parallel_minibatch
    // convention).
    if l2_lambda > 0.0 && steps_added > 0 {
        let scale = l2_lambda * steps_added as f64;
        let fmult = l2_feature_mult();
        add_l2_grad_layer1(
            &mut adam.gw1,
            w1,
            scale,
            n_hidden,
            fmult.as_ref().map(|v| v.as_slice()),
        );
        for (g, &w) in adam.gw2.iter_mut().zip(w2.iter()) {
            *g += scale * w;
        }
    }

    (steps_added, total_loss)
}

/// T8.2: variant of [`backprop_step`] that accumulates into a
/// caller-supplied `LocalGrads` instead of the AdamState's per-param
/// buffers. Mathematically identical — only the destination differs.
///
/// T8.5: delegates to [`crate::simd_mlp::backprop_step`] for AVX-512 /
/// AVX2 / scalar dispatch. LocalGrads' four Vec<f64> buffers map 1:1
/// onto backprop_step's `&mut [f64]` grad parameters.
#[allow(clippy::too_many_arguments)]
fn backprop_into(
    local: &mut LocalGrads,
    x: &[f64],
    h_pre: &[f64],
    h: &[f64],
    dl_dy: f64,
    w2: &[f64],
    n_features: usize,
    n_hidden: usize,
    alpha: f64,
) {
    crate::simd_mlp::backprop_step(
        x,
        h_pre,
        h,
        dl_dy,
        &mut local.gw1,
        &mut local.gb1,
        w2,
        &mut local.gw2,
        &mut local.gb2,
        n_features,
        n_hidden,
        alpha,
    );
}

#[allow(clippy::too_many_arguments)]
fn predict_group(
    std_x: &[f64],
    n_pairs: usize,
    n_features: usize,
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    n_hidden: usize,
    alpha: f64,
) -> Vec<f64> {
    // Rows are INDEPENDENT — `forward` reads `std_x[i]` and the (immutable)
    // weights and returns a value; nothing accumulates across `i`. Writing
    // each result to its own index under rayon is therefore bit-identical to
    // the sequential map, while the validation pass (up to ~730 k rows across
    // 11 groups at 944 features) was the trainer's one remaining fully
    // serial phase.
    //
    // Below `PREDICT_PARALLEL_MIN_ROWS` the task overhead exceeds the
    // forward, and the small validation groups (SDR25 is 50 rows) hit that
    // constantly.
    const PREDICT_PARALLEL_MIN_ROWS: usize = 2048;
    const PREDICT_CHUNK_ROWS: usize = 512;
    let mut out = vec![0.0f64; n_pairs];
    if n_pairs < PREDICT_PARALLEL_MIN_ROWS {
        for (i, o) in out.iter_mut().enumerate() {
            let xi = &std_x[i * n_features..(i + 1) * n_features];
            let (y, _, _) = forward(xi, w1, b1, w2, b2, n_features, n_hidden, alpha);
            *o = y;
        }
        return out;
    }
    out.par_chunks_mut(PREDICT_CHUNK_ROWS)
        .enumerate()
        .for_each(|(c, dst)| {
            let base = c * PREDICT_CHUNK_ROWS;
            for (k, o) in dst.iter_mut().enumerate() {
                let i = base + k;
                let xi = &std_x[i * n_features..(i + 1) * n_features];
                let (y, _, _) = forward(xi, w1, b1, w2, b2, n_features, n_hidden, alpha);
                *o = y;
            }
        });
    out
}

mod utils;
#[allow(unused_imports)] // used by mod-internal (partly cfg(test)) callers and
// re-exported to the zensim_mlp_train bin; some target/feature combinations
// under --all-targets see the re-export as unused.
pub use utils::{spearman_correlation, sweep_nan_inf};

struct AdamState {
    gw1: Vec<f64>,
    gb1: Vec<f64>,
    gw2: Vec<f64>,
    gb2: Vec<f64>,
    mw1: Vec<f64>,
    mb1: Vec<f64>,
    mw2: Vec<f64>,
    mb2: Vec<f64>,
    vw1: Vec<f64>,
    vb1: Vec<f64>,
    vw2: Vec<f64>,
    vb2: Vec<f64>,
    t: u64,
}

impl AdamState {
    fn new(nw1: usize, nb1: usize, nw2: usize, nb2: usize) -> Self {
        Self {
            gw1: vec![0.0; nw1],
            gb1: vec![0.0; nb1],
            gw2: vec![0.0; nw2],
            gb2: vec![0.0; nb2],
            mw1: vec![0.0; nw1],
            mb1: vec![0.0; nb1],
            mw2: vec![0.0; nw2],
            mb2: vec![0.0; nb2],
            vw1: vec![0.0; nw1],
            vb1: vec![0.0; nb1],
            vw2: vec![0.0; nw2],
            vb2: vec![0.0; nb2],
            t: 0,
        }
    }

    fn step(&mut self, w1: &mut [f64], b1: &mut [f64], w2: &mut [f64], b2: &mut [f64], lr: f64) {
        self.t += 1;
        let beta1: f64 = 0.9;
        let beta2: f64 = 0.999;
        let eps: f64 = 1e-8;
        // For sufficiently large `t`, `beta^t` underflows to 0.0 exactly
        // and `bc = 1.0`. The scalar path divides by `bc` either way; we
        // pass it through unchanged so the SIMD result is bit-identical.
        let bc1 = 1.0 - beta1.powi(self.t as i32);
        let bc2 = 1.0 - beta2.powi(self.t as i32);

        let step_one = |w: &mut [f64], g: &mut [f64], m: &mut [f64], v: &mut [f64]| {
            let mut args = adam_simd::AdamUpdateArgs {
                w,
                g,
                m,
                v,
                beta1,
                beta2,
                eps,
                bc1,
                bc2,
                lr,
            };
            adam_simd::adam_update(&mut args);
        };
        step_one(w1, &mut self.gw1, &mut self.mw1, &mut self.vw1);
        step_one(b1, &mut self.gb1, &mut self.mb1, &mut self.vb1);
        step_one(w2, &mut self.gw2, &mut self.mw2, &mut self.vw2);
        step_one(b2, &mut self.gb2, &mut self.mb2, &mut self.vb2);
    }
}

pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub(crate) fn next_f64_unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
    }

    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64_unit().max(1e-12);
        let u2 = self.next_f64_unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Per-pair forward state retained across an NiN-aware per-sample
/// α-head mini-batch. The flush walks every `Some(_)` entry, computes
/// NiN over the 2N predictions, and routes per-prediction grad through
/// `backprop_step_per_sample_alpha_head` (which extends hybrid_head
/// with the W_α path's contribution to ∂L/∂h_j).
pub(crate) struct PerSampleAlphaPairForward<'a> {
    pub(crate) xa: &'a [f64],
    pub(crate) xb: &'a [f64],
    pub(crate) ya: f64,
    pub(crate) yb: f64,
    pub(crate) ya_rank: f64,
    pub(crate) yb_rank: f64,
    pub(crate) ya_pool: f64,
    pub(crate) yb_pool: f64,
    pub(crate) alpha_a: f64,
    pub(crate) alpha_b: f64,
    pub(crate) ha_pre: Vec<f64>,
    pub(crate) ha: Vec<f64>,
    pub(crate) hb_pre: Vec<f64>,
    pub(crate) hb: Vec<f64>,
    pub(crate) sa: [f64; 4],
    pub(crate) sb: [f64; 4],
    pub(crate) max_a: usize,
    pub(crate) max_b: usize,
    pub(crate) dl_dya_rn: f64,
    pub(crate) dl_dyb_rn: f64,
    pub(crate) mos_a: f64,
    pub(crate) mos_b: f64,
}

/// QAT straight-through quantize: per-layer zerobias (relative to the
/// layer's max, matching `apply_zero_bias_per_layer_in_place`) + f16
/// round-trip, on the trainer's f64 weight buffer. Matches the bake-time
/// f16+zerobias packing so the QAT forward sees exactly what the shipped
/// bake will store. Returns a quantized COPY (master is left untouched).
fn qat_quantize_copy(w: &[f64], tau: f64) -> Vec<f64> {
    if w.is_empty() {
        return Vec::new();
    }
    let mut f: Vec<f32> = w.iter().map(|&x| x as f32).collect();
    zenpredict_bake::apply_zero_bias_per_layer_in_place(&mut f, tau as f32);
    f.iter()
        .map(|&x| zenpredict::f16_bits_to_f32(zenpredict_bake::composer::f32_to_f16_bits(x)) as f64)
        .collect()
}

/// Production trainer for the per-sample α head. Mirrors
/// `train_mlp_hybrid_head_with_tv` but routes through
/// `backprop_step_per_sample_alpha_head` and packs the per-sample
/// `(W_α, b_α)` parameters into Adam.
///
/// **Adam w2/b2 layout**:
/// - `gw2[0..n_hidden]`           = rank_w
/// - `gw2[n_hidden..n_hidden+4]`  = reducer_w
/// - `gw2[n_hidden+4..2·n_hidden+4]` = W_α
/// - `gw2[2·n_hidden+4]`          = b_α
/// - `gb2[0]`                     = rank_b
/// - `gb2[1]`                     = reducer_b
///
/// TV regularizer is OMITTED on this path (the V_22-LARGE recipe
/// doesn't use TV). NiN composes via `flush_per_sample_alpha_nin_batch`
/// in the same per-pair grad-scatter pattern as the hybrid path.
/// Standardized TV-endpoint feature buffers + regularizer knobs, unpacked
/// once at the top of [`train_mlp_per_sample_alpha_head`] from the
/// optional `tv` config. `(features, pairs, weight, batch_size, margin)`.
type StdTvUnpacked = (Vec<Vec<f64>>, Vec<(usize, usize)>, f64, usize, f64);

#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_arguments)] // one optional research-pool input per aux loss
fn train_mlp_per_sample_alpha_head(
    groups: &mut [TrainingGroup<'_>],
    n_features: usize,
    hyperparams: &MlpHyperparams,
    log: &mut Vec<String>,
    anchor: Option<&AnchorRows<'_>>,
    equiv: Option<&EquivPairs<'_>>,
    pjnd_anchor: Option<&AnchorRows<'_>>,
    konjnd_agg: Option<&KonjndAggregationPool<'_>>,
    tv: Option<&TvRegularizer>,
    triplets: Option<&TripletPool>,
) -> Vec<u8> {
    use zensim_train_core::per_sample_alpha_head as psah;

    let n_hidden = hyperparams.n_hidden;
    let leaky = hyperparams.leaky_alpha;

    assert!(!groups.is_empty(), "need at least one training group");
    for g in groups.iter() {
        assert_eq!(
            g.human_scores.len(),
            g.features.len(),
            "{}: scores/features length mismatch",
            g.name
        );
        assert!(
            g.features.iter().all(|f| f.len() == n_features),
            "{}: feature length mismatch",
            g.name
        );
    }

    // PreviewV0_5TunerV2: validate anchor rows match n_features and
    // log presence. The anchor is consulted ONLY if both anchor data
    // is provided AND anchor_loss_weight > 0; otherwise it's a no-op.
    let anchor_active = anchor.is_some() && hyperparams.anchor_loss_weight > 0.0;
    let equiv_active = equiv.is_some() && hyperparams.cross_codec_eq_weight > 0.0;
    // EXP-V11-D-PJND-DOMINANT (task #198) — second anchor pool.
    let pjnd_active = pjnd_anchor.is_some() && hyperparams.pjnd_passthrough_weight > 0.0;
    if let Some(pa) = pjnd_anchor {
        assert_eq!(
            pa.features.len(),
            pa.row_weights.len(),
            "pjnd_anchor '{}': features/row_weights length mismatch",
            pa.name
        );
        for (i, f) in pa.features.iter().enumerate() {
            assert_eq!(
                f.len(),
                n_features,
                "pjnd_anchor '{}' row {}: feature length {} != n_features {}",
                pa.name,
                i,
                f.len(),
                n_features
            );
        }
    }
    // KONJND-AGGREGATION-HEAD (task #4) — per-source pool with per-ref
    // grouping. Validate shape invariants up-front so the training step
    // can index without bounds checks.
    let konjnd_agg_active = konjnd_agg.is_some() && hyperparams.konjnd_aggregation_weight > 0.0;
    if let Some(ka) = konjnd_agg {
        assert_eq!(
            ka.ref_ranges.len(),
            ka.ref_pjnd_target.len(),
            "konjnd_agg '{}': ref_ranges/ref_pjnd_target length mismatch",
            ka.name
        );
        assert_eq!(
            ka.ref_ranges.len(),
            ka.ref_weight.len(),
            "konjnd_agg '{}': ref_ranges/ref_weight length mismatch",
            ka.name
        );
        let total_rows: usize = ka.ref_ranges.iter().map(|&(_, n)| n).sum();
        assert_eq!(
            total_rows,
            ka.features.len(),
            "konjnd_agg '{}': Σ ref_ranges.n_rows={} != features.len()={}",
            ka.name,
            total_rows,
            ka.features.len()
        );
        for (i, f) in ka.features.iter().enumerate() {
            assert_eq!(
                f.len(),
                n_features,
                "konjnd_agg '{}' row {}: feature length {} != n_features {}",
                ka.name,
                i,
                f.len(),
                n_features
            );
        }
    }
    if konjnd_agg_active {
        assert!(
            hyperparams.norm_in_norm_weight == 0.0,
            "per_sample_alpha_head + NiN: konjnd-aggregation is not yet composed with NiN. \
             Disable NiN or set --konjnd-aggregation-weight 0."
        );
        assert!(
            (0.0..=1.0).contains(&hyperparams.konjnd_aggregation_step_p),
            "--konjnd-aggregation-step-p must be in [0, 1]; got {}",
            hyperparams.konjnd_aggregation_step_p
        );
        assert!(
            hyperparams.konjnd_aggregation_samples_per_ref >= 1,
            "--konjnd-aggregation-samples-per-ref must be ≥ 1; got {}",
            hyperparams.konjnd_aggregation_samples_per_ref
        );
        assert!(
            hyperparams.konjnd_aggregation_refs_per_step >= 1,
            "--konjnd-aggregation-refs-per-step must be ≥ 1; got {}",
            hyperparams.konjnd_aggregation_refs_per_step
        );
    }
    if pjnd_active {
        assert!(
            hyperparams.norm_in_norm_weight == 0.0,
            "per_sample_alpha_head + NiN: pjnd-passthrough is not yet composed with NiN. \
             Disable NiN (--norm-in-norm-weight 0) or set --pjnd-passthrough-weight 0."
        );
        assert!(
            (0.0..=1.0).contains(&hyperparams.pjnd_passthrough_step_p),
            "--pjnd-passthrough-step-p must be in [0, 1]; got {}",
            hyperparams.pjnd_passthrough_step_p
        );
    }
    if let Some(e) = equiv {
        assert_eq!(
            e.features_a.len(),
            e.features_b.len(),
            "equiv '{}': features_a/features_b length mismatch",
            e.name
        );
        assert_eq!(
            e.features_a.len(),
            e.row_weights.len(),
            "equiv '{}': features_a/row_weights length mismatch",
            e.name
        );
        for (i, f) in e.features_a.iter().enumerate() {
            assert_eq!(
                f.len(),
                n_features,
                "equiv '{}' row {} (A): feature length {} != n_features {}",
                e.name,
                i,
                f.len(),
                n_features
            );
        }
        for (i, f) in e.features_b.iter().enumerate() {
            assert_eq!(
                f.len(),
                n_features,
                "equiv '{}' row {} (B): feature length {} != n_features {}",
                e.name,
                i,
                f.len(),
                n_features
            );
        }
    }
    if let Some(a) = anchor {
        assert_eq!(
            a.features.len(),
            a.row_weights.len(),
            "anchor '{}': features/row_weights length mismatch",
            a.name
        );
        for (i, f) in a.features.iter().enumerate() {
            assert_eq!(
                f.len(),
                n_features,
                "anchor '{}' row {}: feature length {} != n_features {}",
                a.name,
                i,
                f.len(),
                n_features
            );
        }
    }
    let nin_on = hyperparams.norm_in_norm_weight > 0.0;
    if nin_on {
        assert!(
            hyperparams.minibatch_size >= 16,
            "per_sample_alpha_head + NiN: K (minibatch_size) must be ≥16; got {}",
            hyperparams.minibatch_size
        );
        // PreviewV0_5Tuner auxiliary losses (mse_weight / monotonicity_reg)
        // are wired only on the plain-RankNet path (the NiN flush helper
        // packages its own forwards). Composing both requires extending
        // flush_per_sample_alpha_nin_batch — queued for v1. Trainer
        // panics here so a misconfigured run fails loud.
        assert!(
            hyperparams.mse_weight == 0.0 && hyperparams.monotonicity_reg == 0.0,
            "per_sample_alpha_head + NiN: --mse-weight and --monotonicity-reg are not yet \
             composed with NiN. Disable NiN (--norm-in-norm-weight 0) or these aux losses."
        );
    }
    // PreviewV0_5TunerV2: anchor + cross-codec-eq + dynamic-range +
    // rank-preserve aux loss steps are NiN-incompatible (the NiN flush
    // helper packages its own forwards). K-batching is supported via
    // SPEED-B (2026-05-19): aux steps fire on Adam-step boundaries
    // (steps_since_adam == 0) and each fire processes K samples into
    // the shared adam.g* buffer before one do_adam_step. K=1 callers
    // get bit-identical semantics (every iteration is an Adam boundary,
    // K samples = 1 sample); K=32+ callers get the rayon parallel-batch
    // speedup the T8.1-T8.11 optimizations were designed for.
    if hyperparams.anchor_loss_weight > 0.0 && anchor.is_some() {
        assert!(
            !nin_on,
            "per_sample_alpha_head + NiN: anchor loss is not yet composed with NiN. \
             Disable NiN (--norm-in-norm-weight 0) or set --anchor-loss-weight 0."
        );
        assert!(
            (0.0..=1.0).contains(&hyperparams.anchor_step_p),
            "--anchor-step-p must be in [0, 1]; got {}",
            hyperparams.anchor_step_p
        );
    }
    if equiv_active {
        assert!(
            !nin_on,
            "per_sample_alpha_head + NiN: cross-codec-eq loss is not yet composed with NiN. \
             Disable NiN (--norm-in-norm-weight 0) or set --cross-codec-eq-weight 0."
        );
        assert!(
            (0.0..=1.0).contains(&hyperparams.cross_codec_eq_step_p),
            "--cross-codec-eq-step-p must be in [0, 1]; got {}",
            hyperparams.cross_codec_eq_step_p
        );
    }

    // EXP-CROSS-CODEC-V4 (2026-05-19): tanh-output-head sanity gates.
    // SPEED-B (2026-05-19): tanh-pin is per-prediction (pin_forward
    // closure), works with any K. Removed K=1 assert.
    if hyperparams.tanh_output_head_scale > 0.0 {
        assert!(
            hyperparams.per_sample_alpha_head,
            "--tanh-output-head-scale > 0 is only wired on the \
             per_sample_alpha_head path (set --per-sample-alpha-head)."
        );
        assert!(
            !nin_on,
            "per_sample_alpha_head + NiN: --tanh-output-head-scale is not yet \
             composed with NiN. Disable NiN (--norm-in-norm-weight 0) or set \
             --tanh-output-head-scale 0."
        );
    }

    // Architecture flags — gate aux losses that haven't been wired
    // for multi-layer / skip yet.
    let use_2layer = hyperparams.n_hidden_layers >= 2;
    let use_skip = hyperparams.skip_connection;
    if use_2layer || use_skip {
        assert!(
            !nin_on,
            "multi-layer / skip + NiN: not yet composed. Disable NiN."
        );
        if hyperparams.cross_codec_eq_weight > 0.0 {
            panic!(
                "multi-layer / skip + cross_codec_eq: aux loss not yet wired. \
                 Set --cross-codec-eq-weight 0."
            );
        }
        // Anchor loss is now wired through arch_backward for 2-layer.
        if hyperparams.pjnd_passthrough_weight > 0.0 {
            panic!(
                "multi-layer / skip + pjnd_passthrough: not yet wired. \
                 Set --pjnd-passthrough-weight 0."
            );
        }
        // konjnd_aggregation is now wired through arch_forward /
        // arch_backward for 2-layer / skip (task #4 follow-on,
        // G5 lever). The aggregation step below dispatches on
        // use_2layer / use_skip just like the anchor step.
    }

    let n_hidden_final = if use_2layer {
        (n_hidden / 2).max(8)
    } else {
        n_hidden
    };

    let train_total: f64 = groups.iter().map(|g| g.train_weight).sum();
    assert!(train_total > 0.0, "no training groups");

    let train_indices: Vec<usize> = groups
        .iter()
        .enumerate()
        .filter_map(|(i, g)| if g.train_weight > 0.0 { Some(i) } else { None })
        .collect();
    let val_indices: Vec<usize> = groups
        .iter()
        .enumerate()
        .filter_map(|(i, g)| {
            if g.validation_weight > 0.0 {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    let log_line = |msg: &str, log: &mut Vec<String>| {
        eprintln!("{msg}");
        log.push(msg.to_string());
    };

    log_line(
        &format!(
            "MLP train (PER-SAMPLE-α): arch=[{n_features} → {n_hidden} (LeakyReLU α={leaky}) \
             → (h · rank_w + rank_b) ⊕α(x) (pool[μ,σ,max,p_6] · reducer + b)], val_policy={:?}",
            hyperparams.validation_policy,
        ),
        log,
    );
    for (i, g) in groups.iter().enumerate() {
        let role = match (g.train_weight > 0.0, g.validation_weight > 0.0) {
            (true, true) => "train+val",
            (true, false) => "train",
            (false, true) => "val-only",
            (false, false) => "report",
        };
        log_line(
            &format!(
                "  {role:>9} group {i}: '{}' n={} train_w={:.3} val_w={:.3} withinref={} loss={:?}",
                g.name,
                g.features.len(),
                g.train_weight,
                g.validation_weight,
                g.ref_ids.is_some(),
                g.loss_mode,
            ),
            log,
        );
    }

    let (scaler_mean, scaler_scale) =
        compute_scaler_from_groups(groups, &train_indices, n_features);

    let std_features =
        standardize_groups_releasing_raw(groups, n_features, &scaler_mean, &scaler_scale);

    // Per-epoch group-eval SAMPLING (2026-07-02, iteration-speed fix).
    // The per-epoch diagnostics/selection forward EVERY row of EVERY group
    // (v51: 3.4M forwards/epoch ≈ 3× the training compute) even though the
    // panel stats then stride-decimate to 4096 rows anyway. When
    // `group_eval_cap > 0`, pre-gather a deterministic stride sample of each
    // oversized group ONCE and forward only that per epoch. Sampling is
    // stride-based (no RNG) so the training byte-stream is untouched;
    // default 0 = full (exact historical behavior — old manifests stay
    // byte-reproducible). New recipes opt in via `[training].group_eval_cap`.
    let eval_cap = hyperparams.group_eval_cap;
    let (eval_features, eval_humans): (Vec<Vec<f64>>, Vec<Vec<f64>>) = if eval_cap == 0 {
        (Vec::new(), Vec::new())
    } else {
        let mut efs = Vec::with_capacity(groups.len());
        let mut ehs = Vec::with_capacity(groups.len());
        for (gi, g) in groups.iter().enumerate() {
            let n = g.features.len();
            if n <= eval_cap {
                efs.push(Vec::new()); // sentinel: use full std_features[gi]
                ehs.push(Vec::new());
                continue;
            }
            let stride = n.div_ceil(eval_cap);
            let mut fb = Vec::with_capacity((n / stride + 1) * n_features);
            let mut hb = Vec::with_capacity(n / stride + 1);
            let mut i = 0usize;
            while i < n {
                fb.extend_from_slice(&std_features[gi][i * n_features..(i + 1) * n_features]);
                hb.push(g.human_scores[i]);
                i += stride;
            }
            log_line(
                &format!(
                    "group-eval sampling: '{}' {} rows → {} (stride {})",
                    g.name,
                    n,
                    hb.len(),
                    stride
                ),
                log,
            );
            efs.push(fb);
            ehs.push(hb);
        }
        (efs, ehs)
    };

    // Within-ladder monotonicity TV pairs (KADIS severity ladders), wired into
    // the per-sample-α path 2026-07-01. Previously the TvRegularizer was dropped
    // at the dispatch (train_mlp_with_tv_anchored_equiv_pjnd → this fn), so TV was
    // a silent no-op on this head. Standardize the TV endpoint features against
    // the SAME training scaler (mirrors the anchor pool below) so the hinge
    // penalty operates in the network's input space.
    let tv_active = tv
        .map(|t| t.weight > 0.0 && !t.pairs.is_empty())
        .unwrap_or(false);
    let (std_tv_features, tv_pairs_vec, tv_weight_val, tv_batch_val, tv_margin_val): StdTvUnpacked =
        if let (Some(t), true) = (tv, tv_active) {
            let mut bufs: Vec<Vec<f64>> = Vec::with_capacity(t.features.len());
            for f in t.features.iter() {
                let mut buf = vec![0.0f64; n_features];
                for d in 0..n_features {
                    buf[d] = (f[d] - scaler_mean[d]) / scaler_scale[d].max(1e-12);
                }
                bufs.push(buf);
            }
            log_line(
                &format!(
                    "tv(α-head): ENABLED — {} within-ladder pairs, weight={:.3}, batch={}, margin={:.3} (hinge max(0, y_harsher − y_milder + margin))",
                    t.pairs.len(),
                    t.weight,
                    t.batch.max(1),
                    t.margin,
                ),
                log,
            );
            (bufs, t.pairs.clone(), t.weight, t.batch.max(1), t.margin)
        } else {
            if tv.is_some() && !tv_active {
                log_line("tv: data supplied but weight=0 or no pairs — ignored", log);
            }
            (Vec::new(), Vec::new(), 0.0, 0, 0.0)
        };

    // PreviewV0_5TunerV2 anchor data: standardize against the SAME
    // training-group scaler (consistent with the bake's runtime scaler;
    // the bake metadata captures `scaler_mean` / `scaler_scale` once).
    // Also build a row-weight CDF over the anchor pool so we can sample
    // KonJND rows preferentially (per the per-row weights in
    // AnchorRows.row_weights).
    let (std_anchor_features, anchor_row_cdf, anchor_total_weight): (Vec<Vec<f64>>, Vec<f64>, f64) =
        if let (Some(a), true) = (anchor, anchor_active) {
            let mut bufs: Vec<Vec<f64>> = Vec::with_capacity(a.features.len());
            for &f in a.features.iter() {
                let mut buf = vec![0.0f64; n_features];
                for d in 0..n_features {
                    buf[d] = (f[d] - scaler_mean[d]) / scaler_scale[d].max(1e-12);
                }
                bufs.push(buf);
            }
            let total: f64 = a.row_weights.iter().sum();
            let mut cum = 0.0f64;
            let cdf: Vec<f64> = a
                .row_weights
                .iter()
                .map(|&w| {
                    cum += w.max(0.0);
                    if total > 0.0 { cum / total } else { 0.0 }
                })
                .collect();
            let target_mode = if a
                .target_scores
                .map(|ts| ts.len() == a.features.len())
                .unwrap_or(false)
            {
                "PER-ROW (V5 multi-band)".to_string()
            } else {
                format!("{:.2} (global fallback)", hyperparams.anchor_target_score)
            };
            log_line(
                &format!(
                    "anchor: ENABLED — '{}' n={} target_score={} step_p={:.3} weight={:.3}",
                    a.name,
                    a.features.len(),
                    target_mode,
                    hyperparams.anchor_step_p,
                    hyperparams.anchor_loss_weight
                ),
                log,
            );
            (bufs, cdf, total)
        } else {
            if anchor.is_some() && !anchor_active {
                log_line(
                    "anchor: data supplied but anchor_loss_weight = 0 — ignored",
                    log,
                );
            }
            (Vec::new(), Vec::new(), 0.0)
        };

    // EXP-CROSS-CODEC-METRIC equiv data: standardize each pair's A/B
    // feature vectors against the SAME training-group scaler. Row CDF
    // mirrors AnchorRows for per-row weight sampling.
    type EquivPool = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>, f64);
    let (std_equiv_a, std_equiv_b, equiv_row_cdf, equiv_total_weight): EquivPool = if let (
        Some(e),
        true,
    ) =
        (equiv, equiv_active)
    {
        let mut bufs_a: Vec<Vec<f64>> = Vec::with_capacity(e.features_a.len());
        let mut bufs_b: Vec<Vec<f64>> = Vec::with_capacity(e.features_b.len());
        for (&fa, &fb) in e.features_a.iter().zip(e.features_b.iter()) {
            let mut buf_a = vec![0.0f64; n_features];
            let mut buf_b = vec![0.0f64; n_features];
            for d in 0..n_features {
                buf_a[d] = (fa[d] - scaler_mean[d]) / scaler_scale[d].max(1e-12);
                buf_b[d] = (fb[d] - scaler_mean[d]) / scaler_scale[d].max(1e-12);
            }
            bufs_a.push(buf_a);
            bufs_b.push(buf_b);
        }
        let total: f64 = e.row_weights.iter().sum();
        let mut cum = 0.0f64;
        let cdf: Vec<f64> = e
            .row_weights
            .iter()
            .map(|&w| {
                cum += w.max(0.0);
                if total > 0.0 { cum / total } else { 0.0 }
            })
            .collect();
        log_line(
            &format!(
                "cross-codec-eq: ENABLED — '{}' n={} step_p={:.3} weight={:.3}",
                e.name,
                e.features_a.len(),
                hyperparams.cross_codec_eq_step_p,
                hyperparams.cross_codec_eq_weight
            ),
            log,
        );
        if hyperparams.cross_codec_rank_preserve_weight > 0.0 {
            let butter_n = e.butter_diff.len();
            log_line(
                &format!(
                    "cross-codec-rank-preserve: ENABLED — w={:.3} (butter_diff present for {}/{} pairs)",
                    hyperparams.cross_codec_rank_preserve_weight,
                    butter_n,
                    e.features_a.len(),
                ),
                log,
            );
        }
        if hyperparams.dynamic_range_floor_weight > 0.0 {
            log_line(
                &format!(
                    "dynamic-range-floor: ENABLED — w={:.3} σ_thresh={:.2} step_p={:.3} probe_n={}",
                    hyperparams.dynamic_range_floor_weight,
                    hyperparams.dynamic_range_sigma_threshold,
                    hyperparams.dynamic_range_step_p,
                    hyperparams.dynamic_range_probe_n,
                ),
                log,
            );
        }
        (bufs_a, bufs_b, cdf, total)
    } else {
        if equiv.is_some() && !equiv_active {
            log_line(
                "cross-codec-eq: data supplied but cross_codec_eq_weight = 0 — ignored",
                log,
            );
        }
        (Vec::new(), Vec::new(), Vec::new(), 0.0)
    };

    // EXP-V11-D-PJND-DOMINANT (2026-05-20, task #198) — second anchor
    // pool (KonJND-PJND passthrough). Standardize against the SAME
    // training-group scaler. Build a row-weight CDF mirroring the
    // primary anchor pool's machinery.
    let (std_pjnd_features, pjnd_row_cdf, pjnd_total_weight): (Vec<Vec<f64>>, Vec<f64>, f64) =
        if let (Some(pa), true) = (pjnd_anchor, pjnd_active) {
            let mut bufs: Vec<Vec<f64>> = Vec::with_capacity(pa.features.len());
            for &f in pa.features.iter() {
                let mut buf = vec![0.0f64; n_features];
                for d in 0..n_features {
                    buf[d] = (f[d] - scaler_mean[d]) / scaler_scale[d].max(1e-12);
                }
                bufs.push(buf);
            }
            let total: f64 = pa.row_weights.iter().sum();
            let mut cum = 0.0f64;
            let cdf: Vec<f64> = pa
                .row_weights
                .iter()
                .map(|&w| {
                    cum += w.max(0.0);
                    if total > 0.0 { cum / total } else { 0.0 }
                })
                .collect();
            let target_mode = if pa
                .target_scores
                .map(|ts| ts.len() == pa.features.len())
                .unwrap_or(false)
            {
                "PER-ROW".to_string()
            } else {
                format!("{:.2} (global)", hyperparams.pjnd_passthrough_target_score)
            };
            log_line(
                &format!(
                    "pjnd-passthrough: ENABLED — '{}' n={} target_score={} step_p={:.3} weight={:.3}",
                    pa.name,
                    pa.features.len(),
                    target_mode,
                    hyperparams.pjnd_passthrough_step_p,
                    hyperparams.pjnd_passthrough_weight
                ),
                log,
            );
            (bufs, cdf, total)
        } else {
            if pjnd_anchor.is_some() && !pjnd_active {
                log_line(
                    "pjnd-passthrough: data supplied but pjnd_passthrough_weight = 0 — ignored",
                    log,
                );
            }
            (Vec::new(), Vec::new(), 0.0)
        };

    // KONJND-AGGREGATION-HEAD (task #4, 2026-05-24) — standardize the
    // per-ref-grouped pool with the SAME training-group scaler.
    // Output `std_konjnd_agg_features` is flat (preserves the input
    // ref_ranges layout); the training step indexes via ref_ranges.
    let std_konjnd_agg_features: Vec<Vec<f64>> = if let (Some(ka), true) =
        (konjnd_agg, konjnd_agg_active)
    {
        let mut bufs: Vec<Vec<f64>> = Vec::with_capacity(ka.features.len());
        for &f in ka.features.iter() {
            let mut buf = vec![0.0f64; n_features];
            for d in 0..n_features {
                buf[d] = (f[d] - scaler_mean[d]) / scaler_scale[d].max(1e-12);
            }
            bufs.push(buf);
        }
        log_line(
            &format!(
                "konjnd-aggregation: ENABLED — '{}' rows={} refs={} S={} K={} step_p={:.3} weight={:.3}",
                ka.name,
                ka.features.len(),
                ka.ref_ranges.len(),
                hyperparams.konjnd_aggregation_samples_per_ref,
                hyperparams.konjnd_aggregation_refs_per_step,
                hyperparams.konjnd_aggregation_step_p,
                hyperparams.konjnd_aggregation_weight,
            ),
            log,
        );
        bufs
    } else {
        if konjnd_agg.is_some() && !konjnd_agg_active {
            log_line(
                "konjnd-aggregation: data supplied but konjnd_aggregation_weight = 0 — ignored",
                log,
            );
        }
        Vec::new()
    };

    // For 2-layer: heads use n_hidden_final, so init model with that.
    // For 1-layer: n_hidden_final == n_hidden, so unchanged.
    let init_model = psah::PerSampleAlphaHeadModel::new(n_features, n_hidden, hyperparams.seed);
    let mut w1 = init_model.w1.clone();
    let mut b1 = init_model.b1.clone();

    // Feature-subset pinning (`--keep-features`): the SAME one-time row
    // zeroing the plain path applies (see `train_mlp_strategy`'s call) —
    // this head's layer-1 `w1` is `n_features × n_hidden` in EITHER
    // architecture (`use_2layer` only adds a second encoder layer `w2_enc`
    // that reads the HIDDEN layer's output, never the raw features), so
    // `zero_masked_w1_rows` applies to the identical row layout regardless
    // of `--n-hidden-layers`. That is what makes a 2-layer student of a
    // 265-of-944 slice buildable at all.
    //
    // Dropped inputs are standardized to exactly 0.0 upstream (the same
    // CLI-level ablation the plain path relies on), so the gradient into a
    // zeroed row is exactly 0.0 for the rest of training and the row can
    // never move again — zero per-step cost, and the shipped bake's
    // layer-1 truly does not read the dropped columns, as opposed to
    // merely never having been exercised by training data that happened to
    // zero them.
    let masked_rows = zero_masked_w1_rows(&mut w1, n_hidden, input_keep_mask().as_deref());
    if masked_rows > 0 {
        log_line(
            &format!(
                "[keep-features] pinned {masked_rows} of {n_features} layer-1 input rows to 0 \
                 (effective width {})",
                n_features - masked_rows
            ),
            log,
        );
    }

    // 2nd encoder layer (n_hidden → n_hidden_final), initialized if 2-layer.
    let mut w2_enc: Vec<f64> = if use_2layer {
        let scale = (2.0 / n_hidden as f64).sqrt();
        let mut rng2 = SplitMix64::new(hyperparams.seed ^ 0xDEAD_2222_0000_0000);
        (0..n_hidden * n_hidden_final)
            .map(|_| {
                let u1 = ((rng2.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)).max(1e-12);
                let u2 = (rng2.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
                (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos() * scale
            })
            .collect()
    } else {
        Vec::new()
    };
    let mut b2_enc: Vec<f64> = if use_2layer {
        vec![0.0; n_hidden_final]
    } else {
        Vec::new()
    };

    // Skip connection (n_features → 1), initialized to zero so the
    // network starts as the pure MLP.
    let mut w_skip: Vec<f64> = if use_skip {
        vec![0.0; n_features]
    } else {
        Vec::new()
    };
    let mut b_skip: f64 = 0.0;

    // Head weights use n_hidden_final for 2-layer, n_hidden for 1-layer.
    let mut rank_w: Vec<f64> = if use_2layer {
        let scale = 1.0 / (n_hidden_final as f64).sqrt();
        let mut rng3 = SplitMix64::new(hyperparams.seed ^ 0xBEEF_0000_DEAD_0000);
        (0..n_hidden_final)
            .map(|_| {
                let u1 = ((rng3.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)).max(1e-12);
                let u2 = (rng3.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
                (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos() * scale
            })
            .collect()
    } else {
        init_model.rank_w.clone()
    };
    // monotone_cbc with strict mode requires rank_w ≤ 0 at bake time
    // (proj_leq0 clips positives to 0). With h=1, the SINGLE rank_w
    // weight has a 50/50 chance of starting positive under random
    // init; the soft penalty (`λ·max(0, rank_w)²`, λ=0.5) is too weak
    // to flip its sign reliably in 200 epochs, so the projection
    // zeros it → constant-output bake. Issue #40 root cause for h=1.
    //
    // Fix: at monotone_strict, init rank_w to -|N(0, scale)| so the
    // soft penalty's job is "stay negative" (one-sided gentle pull)
    // rather than "flip sign" (cross-zero).
    //
    // GATED TO h=1 (2026-07-01). The original #40 change applied this
    // flip at EVERY width; its "larger h is unaffected" claim was
    // empirically false — flipping the initial signs of a 64-wide rank
    // head changes the optimization trajectory from step 0 and, on the
    // v47 recipe at seed 17, lands in a collapse basin (AIC-4 0.885 →
    // 0.546). The un-flipped pinned tree (e9442678) reproduces shipped
    // Profile A byte-identically; with the flip it does not. h=1 keeps
    // the fix for its actual root cause; wider heads keep the original
    // symmetric init that every shipped bake trained with.
    if hyperparams.monotone_cbc && hyperparams.monotone_strict && rank_w.len() == 1 {
        for v in rank_w.iter_mut() {
            *v = -v.abs();
        }
    }
    let mut rank_b = init_model.rank_b;
    let mut reducer_w = init_model.reducer_w;
    let mut reducer_b = init_model.reducer_b;
    let mut w_alpha: Vec<f64> = if use_2layer {
        vec![0.0; n_hidden_final]
    } else {
        init_model.w_alpha.clone()
    };
    let mut b_alpha = init_model.b_alpha;

    let mut rng = SplitMix64::new(sampling::sample_stream_seed_per_sample_alpha(
        hyperparams.sample_seed.unwrap_or(hyperparams.seed),
    ));

    // Adam slot sizes:
    //   w1 slot: n_features × n_hidden (+ optional n_hidden × n_hidden_final for 2-layer
    //            + optional n_features for skip)
    //   b1 slot: n_hidden (+ optional n_hidden_final for 2-layer + optional 1 for skip)
    //   w2 slot: n_hidden_final (rank_w) + 4 (reducer_w) + n_hidden_final (W_α) + 1 (b_α)
    //   b2 slot: 2 (rank_b, reducer_b)
    let n_w1_total = w1.len() + w2_enc.len() + w_skip.len();
    let n_b1_total = b1.len() + b2_enc.len() + if use_skip { 1 } else { 0 };
    let n_w2 = n_hidden_final + 4 + n_hidden_final + 1;
    let n_b2 = 2;
    let mut adam = AdamState::new(n_w1_total, n_b1_total, n_w2, n_b2);

    let start = Instant::now();
    let mut best_val_score = f64::NEG_INFINITY;
    let mut best_bake: Option<Vec<u8>> = None;
    let mut stale_epochs = 0usize;

    let cdf: Vec<f64> = {
        let mut cum = 0.0;
        train_indices
            .iter()
            .map(|&gi| {
                cum += groups[gi].train_weight;
                cum / train_total
            })
            .collect()
    };

    let needs_row_boost = hyperparams.low_q_boost != 1.0
        || hyperparams.mid_q_boost != 1.0
        || hyperparams.high_q_boost != 1.0;
    let per_row_cdfs: Vec<Option<Vec<f64>>> = train_indices
        .iter()
        .map(|&gi| {
            if !needs_row_boost {
                return None;
            }
            let g = &groups[gi];
            let mut cum = 0.0;
            let raw: Vec<f64> = g
                .human_scores
                .iter()
                .map(|&s| {
                    let mut w = 1.0;
                    if hyperparams.low_q_boost != 1.0 {
                        if s < 50.0 {
                            w *= hyperparams.low_q_boost;
                        } else if s < 65.0 {
                            w *= hyperparams.low_q_boost.sqrt();
                        }
                    }
                    if hyperparams.mid_q_boost != 1.0 && (50.0..90.0).contains(&s) {
                        w *= hyperparams.mid_q_boost;
                    }
                    if hyperparams.high_q_boost != 1.0 && s >= 90.0 {
                        w *= hyperparams.high_q_boost;
                    }
                    cum += w;
                    cum
                })
                .collect();
            let total = *raw.last().unwrap_or(&1.0);
            Some(raw.into_iter().map(|c| c / total).collect())
        })
        .collect();
    // Within-ref pair buckets, parallel to `per_row_cdfs`. `Some` only for
    // groups the caller opted in via `TrainingGroup::ref_ids` (the binary
    // sets that only when the group spec asked for within-ref). Groups
    // without it keep the uniform cross-image draw, byte-for-byte.
    let ref_buckets: Vec<Option<RefBuckets>> = train_indices
        .iter()
        .map(|&gi| groups[gi].ref_ids.and_then(RefBuckets::build))
        .collect();
    // Row counts by position-in-train_indices, for the owner draw step.
    // `FeatureRows::len` is cached at construction (it stays correct after
    // standardization takes the buffer), so hoisting it out of the hot
    // loop is exact, not an approximation.
    let row_counts: Vec<usize> = train_indices
        .iter()
        .map(|&gi| groups[gi].features.len())
        .collect();
    // Opt-in sample-sequence digest: the faithfulness proof that
    // `sampling::simulate` replays THIS run's draws. Off by default, so a
    // normal run is byte- and cost-identical.
    let mut sample_digest = std::env::var("ZENSIM_SAMPLE_DIGEST")
        .ok()
        .filter(|v| v == "1")
        .map(|_| sampling::SampleSequenceDigest::new());
    // STRATEGY stratified row-A bands. This used to be built in ONE of the
    // four training loops, so `--stratified-bands` was a silent no-op on
    // every other path — including the standard path every board bake
    // trained through. Empty when the flag is 0, which keeps the default
    // byte-identical.
    let strat_bands: Vec<Vec<Vec<usize>>> = if hyperparams.stratified_bands > 0 {
        train_indices
            .iter()
            .map(|&gi| strategy::build_bands(groups[gi].human_scores, hyperparams.stratified_bands))
            .collect()
    } else {
        Vec::new()
    };
    // Say so out loud: which pairing mode a group trained under changes what
    // the bake learned, and it must never be a silent default.
    for (pos, rb) in ref_buckets.iter().enumerate() {
        if let Some(rb) = rb {
            println!(
                "  {}: WITHIN-REF pairs over {} refs ({} rows usable)",
                groups[train_indices[pos]].name,
                rb.n_refs(),
                rb.n_rows(),
            );
        }
    }

    let k = hyperparams.minibatch_size.max(1);
    log_line(
        &format!(
            "per_sample_alpha_head: ENABLED — α(x)=sigmoid(W_α·h + b_α), init W_α=0 b_α=0 → α=0.5 ∀x, K={k} sequential, NiN={}",
            if nin_on {
                format!(
                    "w={:.3} p={:.2} q={:.2}",
                    hyperparams.norm_in_norm_weight,
                    hyperparams.norm_in_norm_p,
                    hyperparams.norm_in_norm_q
                )
            } else {
                "off".to_string()
            }
        ),
        log,
    );

    if hyperparams.sigma_weighted_mse && hyperparams.mse_weight > 0.0 {
        let n_with = groups.iter().filter(|g| g.metric_sigmas.is_some()).count();
        log_line(
            &format!(
                "σ-weighted MSE ACTIVE — {n_with}/{} groups have metric_sigmas, ε=0.05",
                groups.len()
            ),
            log,
        );
    }

    // EXP-CROSS-CODEC-V4 (2026-05-19): tanh-pinned output head.
    // When `tanh_output_head_scale > 0`, every raw `y_pre = α·y_rank +
    // (1−α)·y_pool` from the per-sample-α forward is wrapped as
    // `y_score = 100·σ(y_pre/scale)`, and every upstream gradient `dl_dy`
    // (before being passed to `backprop_step_per_sample_alpha_head`)
    // is multiplied by `(100/scale)·σ·(1−σ)`. This is the chain-rule
    // pin: train losses see score-shaped outputs in [0, 100], so
    // anchor MSE, monotonicity, range-floor, and rank-preserve losses
    // all operate in the SAME units as the final score (eliminating
    // V3's β-amplification mono-violation path).
    let tanh_pin_active = hyperparams.tanh_output_head_scale > 0.0;
    let tanh_scale = hyperparams.tanh_output_head_scale.max(1e-9);
    if tanh_pin_active {
        log_line(
            &format!(
                "tanh_output_head: ENABLED — y_score = 100·σ(y_pre/{tanh_scale:.3}) (active linear region y_pre∈[−3·scale, 3·scale])"
            ),
            log,
        );
    }
    // Inline closures (capture tanh_scale + tanh_pin_active).
    // `pin_forward(y_pre) → (y_score, dy_dy_pre)` so loss sites use
    // `y_score` as the prediction and gradients are scaled by `dy_dy_pre`.
    // When the pin is off, returns the identity (y_pre, 1.0) so the
    // pair-loss path is unchanged.
    let pin_forward = |y_pre: f64| -> (f64, f64) {
        if !tanh_pin_active {
            (y_pre, 1.0)
        } else {
            let xc = (y_pre / tanh_scale).clamp(-30.0, 30.0);
            let s = 1.0 / (1.0 + (-xc).exp());
            let y_score = 100.0 * s;
            // dy_score/dy_pre = (100/scale) · σ · (1−σ)
            let dy = (100.0 / tanh_scale) * s * (1.0 - s);
            (y_score, dy)
        }
    };

    // Correct-by-construction monotone projection (captured by the Adam
    // closure below). When set, after every step the weights are
    // projected onto the sign pattern that makes the bake monotone↓ in
    // every non-negative dissimilarity feature: encoder ≥0, head ≤0,
    // α-gate forced to ≈1. See `MlpHyperparams::monotone_cbc`.
    let monotone_cbc = hyperparams.monotone_cbc;
    // Soft sign-penalty strength (λ). The per-step penalty `±2λw` on
    // wrong-sign weights smoothly biases encoder weights ≥0 / rank_w ≤0
    // WITHOUT a per-step hard clamp (the clamp kills weights and
    // collapses training — v45/v45b/v45c/v45d all cratered with one).
    // The soft penalty keeps weights NEAR the correct sign during
    // training (smooth, no dead-weight collapse); the final HARD
    // projection at every bake-emission site (below) makes the SHIPPED
    // bake's signs EXACT — encoder ≥0, rank_w + w_skip ≤0, α≡1 — so the
    // bake is monotone non-increasing in every non-negative dissimilarity
    // feature, by construction. `--monotone-cbc` is OFF by default;
    // production unaffected.
    const MONOTONE_CBC_PENALTY: f64 = 1.0;
    // Per-feature sign mask for the W1 (feature→hidden) layer. `pin[j]`
    // ⇒ feature j's W1 column is constrained ≥0 (sign-safe). Defaults to
    // all-pinned when no mask is supplied (original behavior). Features
    // NOT pinned are dropped (strict) or left free (partial).
    let feature_pin: Vec<bool> = hyperparams
        .monotone_feature_pin
        .clone()
        .unwrap_or_else(|| vec![true; n_features]);
    assert!(
        !monotone_cbc || feature_pin.len() == n_features,
        "monotone_feature_pin length {} != n_features {}",
        feature_pin.len(),
        n_features
    );
    let monotone_strict = hyperparams.monotone_strict;
    if monotone_cbc {
        let n_pinned = feature_pin.iter().filter(|&&p| p).count();
        let mode = if hyperparams.monotone_pin_during_training {
            "free (KEEP-72: pinned hard-projected during training + at bake; unpinned untouched throughout)"
        } else if monotone_strict {
            "DROPPED (strict)"
        } else {
            "free (partial)"
        };
        log_line(
            &format!(
                "monotone_cbc: ENABLED — W1 per-feature mask: {n_pinned}/{n_features} pinned ≥0, \
                 {} {} ; w2_enc≥0, rank_w≤0, α≡1. Soft penalty (λ=1) + final projection at bake.",
                n_features - n_pinned,
                mode,
            ),
            log,
        );
    }
    // Final hard projection closures applied at every bake-emission site
    // (best-epoch checkpoint save + post-training spline re-bake).
    let proj_geq0 = |w: &[f64]| -> Vec<f64> {
        if monotone_cbc {
            w.iter().map(|&v| v.max(0.0)).collect()
        } else {
            w.to_vec()
        }
    };
    // Per-feature W1 projection: pinned features clamped ≥0, sign-flip
    // features zeroed (strict) or left as-is (partial). W1 is row-major
    // [feature][hidden] so feature j owns block [j*n_hidden, (j+1)*n_hidden).
    // When monotone_pin_during_training is set, monotone_strict's
    // "zero the 72 unpinned at bake" branch is SUPPRESSED — the 72
    // stay free at bake too, matching the train-time treatment
    // (#39 followup #2).
    let strict_drop_unpinned_at_bake = monotone_strict && !hyperparams.monotone_pin_during_training;
    let proj_w1_masked = |w: &[f64]| -> Vec<f64> {
        if !monotone_cbc {
            return w.to_vec();
        }
        let mut out = w.to_vec();
        for j in 0..n_features {
            let block = &mut out[j * n_hidden..(j + 1) * n_hidden];
            if feature_pin[j] {
                for v in block.iter_mut() {
                    *v = v.max(0.0);
                }
            } else if strict_drop_unpinned_at_bake {
                for v in block.iter_mut() {
                    *v = 0.0;
                }
            }
        }
        out
    };
    let proj_leq0 = |w: &[f64]| -> Vec<f64> {
        if monotone_cbc {
            w.iter().map(|&v| v.min(0.0)).collect()
        } else {
            w.to_vec()
        }
    };
    let proj_w_alpha_zero = |w: &[f64]| -> Vec<f64> {
        if monotone_cbc {
            vec![0.0; w.len()]
        } else {
            w.to_vec()
        }
    };
    let proj_b_alpha_one = |b: f64| -> f64 { if monotone_cbc { 30.0 } else { b } };

    // F32 weight scratch for arch_forward_f32 hot path. Refreshed at
    // the END of every do_adam_step (the only point where weights
    // change). Held in a RefCell so the closure can borrow_mut()
    // it without changing the closure's parameter list.
    let scratch_f32 = std::cell::RefCell::new(arch_f32::WeightScratchF32::new(
        n_features,
        n_hidden,
        n_hidden_final,
        use_2layer,
        use_skip,
    ));

    // Adam-step closure: pack/unpack our parameters into the Adam slots.
    // Adam step closure — packs all weight vectors into the concatenated
    // adam slots, runs one step, then unpacks. For multi-layer / skip,
    // the w1 slot holds [w1 | w2_enc | w_skip] concatenated.
    //
    // QAT fine-tune state. `qat_active` flips true for the last
    // `qat_fine_tune_epochs` epochs (set in the epoch loop). When active,
    // do_adam_step refreshes the f32 forward scratch from f16+zerobiased
    // COPIES of the (f32 master) weights — straight-through estimator:
    // forward/loss see the quantized weights, Adam keeps updating the f32
    // master, so the net learns weights robust to the bake-time packing.
    let qat_fine_tune_epochs = hyperparams.qat_fine_tune_epochs;
    let qat_tau = hyperparams.qat_tau;
    let qat_active = std::cell::Cell::new(false);

    let do_adam_step = |adam: &mut AdamState,
                        w1: &mut Vec<f64>,
                        b1: &mut Vec<f64>,
                        w2_enc: &mut Vec<f64>,
                        b2_enc: &mut Vec<f64>,
                        w_skip: &mut Vec<f64>,
                        b_skip: &mut f64,
                        rank_w: &mut Vec<f64>,
                        rank_b: &mut f64,
                        reducer_w: &mut [f64; 4],
                        reducer_b: &mut f64,
                        w_alpha: &mut Vec<f64>,
                        b_alpha: &mut f64,
                        lr: f64,
                        nh_final: usize| {
        // Concatenate encoder weights into a single buffer matching adam.gw1.
        let mut w1_concat: Vec<f64> = Vec::with_capacity(w1.len() + w2_enc.len() + w_skip.len());
        w1_concat.extend_from_slice(w1);
        w1_concat.extend_from_slice(w2_enc);
        w1_concat.extend_from_slice(w_skip);

        let mut b1_concat: Vec<f64> =
            Vec::with_capacity(b1.len() + b2_enc.len() + if !w_skip.is_empty() { 1 } else { 0 });
        b1_concat.extend_from_slice(b1);
        b1_concat.extend_from_slice(b2_enc);
        if !w_skip.is_empty() {
            b1_concat.push(*b_skip);
        }

        // Pack head weights into w2 slot.
        let mut w2_vec = vec![0.0f64; nh_final + 4 + nh_final + 1];
        w2_vec[..nh_final].copy_from_slice(&rank_w[..nh_final]);
        w2_vec[nh_final..nh_final + 4].copy_from_slice(&reducer_w[..]);
        for j in 0..nh_final {
            w2_vec[nh_final + 4 + j] = w_alpha[j];
        }
        w2_vec[nh_final + 4 + nh_final] = *b_alpha;
        let mut b2_vec = vec![*rank_b, *reducer_b];

        // monotone_cbc: SOFT sign-penalty added to the gradient (NO
        // per-step hard clamp — that collapses training via dead weights:
        // v45/v45b/v45c all cratered SROCC 0.94→0.30 by epoch 25). A
        // one-sided quadratic penalty `λ·max(0,∓w)²` nudges encoder
        // weights toward ≥0 and rank_w toward ≤0; its gradient `±2λw` is
        // added to the wrong-sign entries. The final hard projection
        // (after training, before the bake) makes the SHIPPED bake
        // monotone-by-construction; the penalty keeps weights near the
        // correct sign so that projection is gentle and preserves the fit.
        if monotone_cbc {
            let two_lam = 2.0 * MONOTONE_CBC_PENALTY;
            let w1_len = w1.len();
            let w2_len = w2_enc.len();
            // W1 (feature→hidden), PER-FEATURE: pinned features → ≥0
            // (penalize negatives); sign-flip features → 0 in strict mode
            // (penalize BOTH signs, L2-toward-0 so they drop out cleanly),
            // or untouched in partial mode. W1 is row-major [feature][hidden]
            // so index i belongs to feature i / n_hidden.
            //
            // Soft-monotone-keep-72 (#39 followup #2): the new
            // `monotone_pin_during_training` flag SUPPRESSES the
            // "drive unpinned to 0" branch — the 72 sign-flips stay
            // free regardless of monotone_strict. The 300 pinned still
            // get the negative-side soft penalty (cheap nudge toward ≥0)
            // AND the hard-projection-after-step that the flag triggers
            // below.
            let strict_drop_unpinned = monotone_strict && !hyperparams.monotone_pin_during_training;
            for i in 0..w1_len {
                let feat = i / n_hidden;
                if feature_pin[feat] {
                    if w1_concat[i] < 0.0 {
                        adam.gw1[i] += two_lam * w1_concat[i];
                    }
                } else if strict_drop_unpinned {
                    adam.gw1[i] += two_lam * w1_concat[i];
                }
            }
            // w2_enc (hidden→hidden) → ≥0: hidden is already monotone↑ in
            // distortion, so the 2nd layer must preserve the direction.
            for i in w1_len..(w1_len + w2_len) {
                if w1_concat[i] < 0.0 {
                    adam.gw1[i] += two_lam * w1_concat[i];
                }
            }
            // w_skip → ≤0: penalize positive entries.
            for i in (w1_len + w2_len)..w1_concat.len() {
                if w1_concat[i] > 0.0 {
                    adam.gw1[i] += two_lam * w1_concat[i];
                }
            }
            // rank_w (w2 slot [0, nh_final)) → ≤0: penalize positives.
            for j in 0..nh_final {
                if w2_vec[j] > 0.0 {
                    adam.gw2[j] += two_lam * w2_vec[j];
                }
            }
        }

        adam.step(&mut w1_concat, &mut b1_concat, &mut w2_vec, &mut b2_vec, lr);

        // Soft-monotone-keep-72 mode (#39 followup #2): after the Adam
        // step, HARD-project the pinned features' W1 columns to ≥0 in
        // place (matching the final bake projection exactly). The
        // unpinned 72 features stay free — no projection, no penalty.
        // This eliminates the train/bake sign-drift that the soft-only
        // monotone_strict=false (partial) path suffered from. Rank head
        // + w2_enc + w_skip + b_alpha projections are deferred to the
        // bake step as before; only the W1 pinned columns are
        // hard-projected during training (the 300 features the soft
        // penalty is already trying to keep ≥0; cf. v45 collapse incident
        // where clamping ALL weights — including the 72 sign-flips —
        // killed expressivity).
        if hyperparams.monotone_pin_during_training && monotone_cbc {
            let w1_len_pre = w1.len();
            for i in 0..w1_len_pre {
                let feat = i / n_hidden;
                if feature_pin[feat] && w1_concat[i] < 0.0 {
                    w1_concat[i] = 0.0;
                }
            }
        }

        // Unpack encoder weights back. Pre-compute lengths to avoid
        // overlapping borrows.
        let w1_len = w1.len();
        let w2_len = w2_enc.len();
        let ws_len = w_skip.len();
        let b1_len = b1.len();
        let b2_len = b2_enc.len();

        let mut off = 0;
        w1.copy_from_slice(&w1_concat[off..off + w1_len]);
        off += w1_len;
        if w2_len > 0 {
            w2_enc.copy_from_slice(&w1_concat[off..off + w2_len]);
            off += w2_len;
        }
        if ws_len > 0 {
            w_skip.copy_from_slice(&w1_concat[off..off + ws_len]);
        }

        let mut boff = 0;
        b1.copy_from_slice(&b1_concat[boff..boff + b1_len]);
        boff += b1_len;
        if b2_len > 0 {
            b2_enc.copy_from_slice(&b1_concat[boff..boff + b2_len]);
            boff += b2_enc.len();
        }
        if !w_skip.is_empty() {
            *b_skip = b1_concat[boff];
        }

        // Unpack head weights.
        rank_w[..nh_final].copy_from_slice(&w2_vec[..nh_final]);
        reducer_w.copy_from_slice(&w2_vec[nh_final..nh_final + 4]);
        for j in 0..nh_final {
            w_alpha[j] = w2_vec[nh_final + 4 + j];
        }
        *b_alpha = w2_vec[nh_final + 4 + nh_final];
        *rank_b = b2_vec[0];
        *reducer_b = b2_vec[1];

        // monotone_cbc: force the per-sample-α gate to ≈1 every step so
        // y = y_rank (a single monotone head — the α-gated mix of two
        // functions is not monotone). σ(30) ≈ 1 − 9.4e-14. Reset their
        // Adam state so the optimizer doesn't fight the forced values.
        // (The encoder/rank SIGN constraint is enforced softly via the
        // pre-step penalty above + a final hard projection before baking,
        // NOT by a per-step hard clamp here — that collapses training.)
        if monotone_cbc {
            for j in 0..nh_final {
                w_alpha[j] = 0.0;
                let a = nh_final + 4 + j;
                adam.mw2[a] = 0.0;
                adam.vw2[a] = 0.0;
            }
            *b_alpha = 30.0;
            let ba = nh_final + 4 + nh_final;
            adam.mw2[ba] = 0.0;
            adam.vw2[ba] = 0.0;
        }

        // Refresh the f32 weight scratch — weights just changed.
        // RefCell mutable borrow is safe here because arch_forward_f32
        // callers only hold immutable borrows during their own scope,
        // which is disjoint from this closure's invocation.
        if qat_active.get() {
            // STE: forward scratch holds f16+zerobias COPIES; the master
            // weight buffers (w1 … w_alpha) stay f32 for the next Adam step.
            let qw1 = qat_quantize_copy(w1, qat_tau);
            let qw2 = qat_quantize_copy(w2_enc, qat_tau);
            let qws = qat_quantize_copy(w_skip, qat_tau);
            let qrw = qat_quantize_copy(rank_w, qat_tau);
            let qwa = qat_quantize_copy(w_alpha, qat_tau);
            scratch_f32.borrow_mut().refresh(
                &qw1, b1, &qw2, b2_enc, &qws, *b_skip, &qrw, *rank_b, reducer_w, *reducer_b, &qwa,
                *b_alpha,
            );
        } else {
            scratch_f32.borrow_mut().refresh(
                w1, b1, w2_enc, b2_enc, w_skip, *b_skip, rank_w, *rank_b, reducer_w, *reducer_b,
                w_alpha, *b_alpha,
            );
        }
    };

    // Initial cast (before any Adam step has fired).
    scratch_f32.borrow_mut().refresh(
        &w1, &b1, &w2_enc, &b2_enc, &w_skip, b_skip, &rank_w, rank_b, &reducer_w, reducer_b,
        &w_alpha, b_alpha,
    );

    let mut nin_buffer: Vec<Option<PerSampleAlphaPairForward<'_>>> = if nin_on {
        Vec::with_capacity(k)
    } else {
        Vec::new()
    };

    // Pre-compute per-group σ medians (only when σ-weighted MSE is active).
    let sigma_medians: Vec<f64> = if hyperparams.sigma_weighted_mse {
        groups
            .iter()
            .map(|g| {
                g.metric_sigmas
                    .map(|sigmas| {
                        let mut sorted: Vec<f64> = sigmas.to_vec();
                        sorted
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        if sorted.is_empty() {
                            1.0
                        } else {
                            sorted[sorted.len() / 2]
                        }
                    })
                    .unwrap_or(1.0)
            })
            .collect()
    } else {
        vec![1.0; groups.len()]
    };

    // ---------------- STRATEGY-2026-07-02 state ----------------
    let ema_active = hyperparams.ema_decay > 0.0;
    let mut ema = strategy::EmaState::new(hyperparams.ema_decay);
    let dro_active = hyperparams.dro_eta > 0.0;
    let mut cdf = cdf; // rebindable: DRO rebuilds the group CDF per epoch
    let mut dro_loss_sum: Vec<f64> = vec![0.0; train_indices.len()];
    let mut dro_loss_n: Vec<u64> = vec![0; train_indices.len()];
    let listwise_active = hyperparams.listwise_weight > 0.0;
    let triplet_active = hyperparams.triplet_weight > 0.0
        && triplets.map(|t| !t.responses.is_empty()).unwrap_or(false);
    assert!(
        !(nin_on && (listwise_active || triplet_active)),
        "STRATEGY: listwise/triplet steps are not composed with NiN batching yet"
    );
    // standardize triplet stimuli with the SAME scaler as the groups
    let triplet_std: Vec<f64> = if let (true, Some(tp)) = (triplet_active, triplets) {
        let mut buf = vec![0.0f64; tp.features.len() * n_features];
        for (ri, f) in tp.features.iter().enumerate() {
            assert_eq!(
                f.len(),
                n_features,
                "triplet stimulus feature length mismatch"
            );
            for j in 0..n_features {
                buf[ri * n_features + j] = (f[j] - scaler_mean[j]) / scaler_scale[j];
            }
        }
        buf
    } else {
        Vec::new()
    };
    {
        let mut active: Vec<String> = Vec::new();
        if ema_active {
            active.push(format!("ema={}", hyperparams.ema_decay));
        }
        if hyperparams.hard_pair_frac > 0.0 {
            active.push(format!(
                "hardpair={}@{}",
                hyperparams.hard_pair_frac, hyperparams.hard_pair_max_delta
            ));
        }
        if !strat_bands.is_empty() {
            active.push(format!("strat={}", hyperparams.stratified_bands));
        }
        if dro_active {
            active.push(format!("dro_eta={}", hyperparams.dro_eta));
        }
        if listwise_active {
            active.push(format!(
                "listmle w={} K={} frac={}",
                hyperparams.listwise_weight, hyperparams.listwise_size, hyperparams.listwise_frac
            ));
        }
        if triplet_active {
            active.push(format!(
                "triplet w={} frac={} tau={} sigma={} ({} responses, {} stimuli)",
                hyperparams.triplet_weight,
                hyperparams.triplet_frac,
                hyperparams.triplet_tau,
                hyperparams.triplet_sigma,
                triplets.map(|t| t.responses.len()).unwrap_or(0),
                triplets.map(|t| t.features.len()).unwrap_or(0)
            ));
        }
        if !active.is_empty() {
            log_line(&format!("STRATEGY active: {}", active.join(" | ")), log);
        }
    }
    macro_rules! ema_swap_all {
        () => {{
            std::mem::swap(&mut w1, &mut ema.tensors[0]);
            std::mem::swap(&mut b1, &mut ema.tensors[1]);
            std::mem::swap(&mut w2_enc, &mut ema.tensors[2]);
            std::mem::swap(&mut b2_enc, &mut ema.tensors[3]);
            std::mem::swap(&mut w_skip, &mut ema.tensors[4]);
            std::mem::swap(&mut rank_w, &mut ema.tensors[5]);
            std::mem::swap(&mut w_alpha, &mut ema.tensors[6]);
            for i in 0..4 {
                std::mem::swap(&mut reducer_w[i], &mut ema.tensors[7][i]);
            }
            std::mem::swap(&mut b_skip, &mut ema.scalars[0]);
            std::mem::swap(&mut rank_b, &mut ema.scalars[1]);
            std::mem::swap(&mut reducer_b, &mut ema.scalars[2]);
            std::mem::swap(&mut b_alpha, &mut ema.scalars[3]);
        }};
    }

    for epoch in 0..hyperparams.n_epochs {
        let lr = hyperparams.initial_lr
            * 0.5
            * (1.0 + (std::f64::consts::PI * (epoch % 50) as f64 / 50.0).cos());

        // QAT fine-tune: activate the straight-through quantized forward for
        // the last `qat_fine_tune_epochs` epochs. Refresh the scratch from
        // quantized copies NOW so even this epoch's first minibatch (before
        // the first do_adam_step) sees the f16+zerobias weights.
        qat_active
            .set(qat_fine_tune_epochs > 0 && epoch + qat_fine_tune_epochs >= hyperparams.n_epochs);
        if qat_active.get() {
            let qw1 = qat_quantize_copy(&w1, qat_tau);
            let qw2 = qat_quantize_copy(&w2_enc, qat_tau);
            let qws = qat_quantize_copy(&w_skip, qat_tau);
            let qrw = qat_quantize_copy(&rank_w, qat_tau);
            let qwa = qat_quantize_copy(&w_alpha, qat_tau);
            scratch_f32.borrow_mut().refresh(
                &qw1, &b1, &qw2, &b2_enc, &qws, b_skip, &qrw, rank_b, &reducer_w, reducer_b, &qwa,
                b_alpha,
            );
        }

        let mut total_loss = 0.0f64;
        let mut n_steps = 0u64;
        let mut steps_since_adam = 0u64;

        for _ in 0..hyperparams.pairs_per_epoch {
            // STRATEGY: listwise / triplet steps steal a fraction of the
            // pair budget (Adam cadence identical: one logical step each).
            if listwise_active || triplet_active {
                let su = rng.next_f64_unit();
                let t_frac = if triplet_active {
                    hyperparams.triplet_frac
                } else {
                    0.0
                };
                if triplet_active && su < t_frac {
                    let tp = triplets.unwrap();
                    let (li, ri, resp) =
                        tp.responses[(rng.next_u64() as usize) % tp.responses.len()];
                    let xa = &triplet_std[li as usize * n_features..(li as usize + 1) * n_features];
                    let xb = &triplet_std[ri as usize * n_features..(ri as usize + 1) * n_features];
                    let fwd_a = {
                        let sc = scratch_f32.borrow();
                        arch_f32::arch_forward_f32(
                            xa,
                            &sc,
                            n_features,
                            n_hidden,
                            n_hidden_final,
                            leaky as f32,
                            use_2layer,
                            use_skip,
                        )
                        .to_archforward()
                    };
                    let fwd_b = {
                        let sc = scratch_f32.borrow();
                        arch_f32::arch_forward_f32(
                            xb,
                            &sc,
                            n_features,
                            n_hidden,
                            n_hidden_final,
                            leaky as f32,
                            use_2layer,
                            use_skip,
                        )
                        .to_archforward()
                    };
                    let (ya_p, dya_dpre) = pin_forward(fwd_a.y);
                    let (yb_p, dyb_dpre) = pin_forward(fwd_b.y);
                    let (t_loss, dl_dd) = strategy::triplet_probit_loss_dgrad(
                        ya_p,
                        yb_p,
                        hyperparams.triplet_tau,
                        hyperparams.triplet_sigma,
                        resp,
                    );
                    total_loss += t_loss * hyperparams.triplet_weight;
                    n_steps += 1;
                    let dl_dya = -dl_dd * hyperparams.triplet_weight * dya_dpre;
                    let dl_dyb = dl_dd * hyperparams.triplet_weight * dyb_dpre;
                    strategy_backward_rows(
                        &[xa, xb],
                        &[fwd_a, fwd_b],
                        &[dl_dya, dl_dyb],
                        &mut w1,
                        &mut b1,
                        &mut w2_enc,
                        &mut b2_enc,
                        &mut w_skip,
                        &mut b_skip,
                        &mut rank_w,
                        &mut rank_b,
                        &mut reducer_w,
                        &mut reducer_b,
                        &mut w_alpha,
                        &mut b_alpha,
                        &mut adam,
                        n_features,
                        n_hidden,
                        n_hidden_final,
                        leaky,
                        use_2layer,
                        use_skip,
                        hyperparams.l2_lambda,
                        lr,
                        k,
                        &mut steps_since_adam,
                        &do_adam_step,
                    );
                    continue;
                }
                if listwise_active && su < t_frac + hyperparams.listwise_frac {
                    let u2 = rng.next_f64_unit();
                    let lpos = cdf.partition_point(|&c| c < u2).min(cdf.len() - 1);
                    let lg = &groups[train_indices[lpos]];
                    let ln = lg.features.len();
                    let kk = hyperparams.listwise_size.max(2).min(ln);
                    if ln >= 2 {
                        let mut rows: Vec<usize> = Vec::with_capacity(kk);
                        while rows.len() < kk {
                            let r = if !strat_bands.is_empty() {
                                let bands = &strat_bands[lpos];
                                let b = &bands[(rng.next_u64() as usize) % bands.len()];
                                b[(rng.next_u64() as usize) % b.len()]
                            } else {
                                (rng.next_u64() as usize) % ln
                            };
                            if !rows.contains(&r) {
                                rows.push(r);
                            }
                        }
                        let lg_feats = &std_features[train_indices[lpos]];
                        let mut xs: Vec<&[f64]> = Vec::with_capacity(kk);
                        let mut fwds = Vec::with_capacity(kk);
                        let mut ys = Vec::with_capacity(kk);
                        let mut dpres = Vec::with_capacity(kk);
                        let mut tgts = Vec::with_capacity(kk);
                        for &r in &rows {
                            let x = &lg_feats[r * n_features..(r + 1) * n_features];
                            let fwd = {
                                let sc = scratch_f32.borrow();
                                arch_f32::arch_forward_f32(
                                    x,
                                    &sc,
                                    n_features,
                                    n_hidden,
                                    n_hidden_final,
                                    leaky as f32,
                                    use_2layer,
                                    use_skip,
                                )
                                .to_archforward()
                            };
                            let (yp, dp) = pin_forward(fwd.y);
                            xs.push(x);
                            fwds.push(fwd);
                            ys.push(yp);
                            dpres.push(dp);
                            tgts.push(lg.human_scores[r]);
                        }
                        let (l_loss, l_grads) = strategy::listmle_loss_grad(&ys, &tgts);
                        total_loss += l_loss * hyperparams.listwise_weight;
                        n_steps += 1;
                        if dro_active {
                            dro_loss_sum[lpos] += l_loss;
                            dro_loss_n[lpos] += 1;
                        }
                        let dls: Vec<f64> = l_grads
                            .iter()
                            .zip(&dpres)
                            .map(|(&g, &dp)| g * hyperparams.listwise_weight * dp)
                            .collect();
                        strategy_backward_rows(
                            &xs,
                            &fwds,
                            &dls,
                            &mut w1,
                            &mut b1,
                            &mut w2_enc,
                            &mut b2_enc,
                            &mut w_skip,
                            &mut b_skip,
                            &mut rank_w,
                            &mut rank_b,
                            &mut reducer_w,
                            &mut reducer_b,
                            &mut w_alpha,
                            &mut b_alpha,
                            &mut adam,
                            n_features,
                            n_hidden,
                            n_hidden_final,
                            leaky,
                            use_2layer,
                            use_skip,
                            hyperparams.l2_lambda,
                            lr,
                            k,
                            &mut steps_since_adam,
                            &do_adam_step,
                        );
                    }
                    continue;
                }
            }
            // Draw via THE owner (`sampling::draw_pair`). This site keeps
            // `mut ib` because the hard-pair miner below re-draws it.
            let drawn = sampling::draw_pair(
                &sampling::PairDrawCtx {
                    cdf: &cdf,
                    row_counts: &row_counts,
                    per_row_cdfs: &per_row_cdfs,
                    ref_buckets: &ref_buckets,
                    strat_bands: &strat_bands,
                },
                &mut rng,
            );
            if let Some(d) = sample_digest.as_mut() {
                d.push(drawn);
            }
            // `SameRow` still reaches the miner below, exactly as the
            // pre-extraction code did: it dropped through to `if ia == ib`
            // only AFTER the miner had a chance to re-draw ib.
            let (train_pos, ia, mut ib) = match drawn {
                sampling::Draw::GroupTooSmall => continue,
                sampling::Draw::SameRow { train_pos, row } => (train_pos, row, row),
                sampling::Draw::Pair { train_pos, ia, ib } => (train_pos, ia, ib),
            };
            let g_idx = train_indices[train_pos];
            let g = &groups[g_idx];
            let n = g.features.len();
            // STRATEGY: hard-pair mining — with prob hard_pair_frac, re-draw
            // row B (≤16 tries) until the pair is near-threshold.
            if hyperparams.hard_pair_frac > 0.0 && rng.next_f64_unit() < hyperparams.hard_pair_frac
            {
                for _ in 0..16 {
                    if ib != ia
                        && strategy::hard_pair_ok(
                            g.human_scores[ia],
                            g.human_scores[ib],
                            hyperparams.hard_pair_max_delta,
                        )
                    {
                        break;
                    }
                    // Re-draw within ia's ref when the group is within-ref;
                    // a plain `% n` here would smuggle cross-image pairs
                    // back in through the miner.
                    ib = match &ref_buckets[train_pos] {
                        Some(rb) => rb.redraw_partner(ia, rng.next_u64()),
                        None => (rng.next_u64() as usize) % n,
                    };
                }
            }
            if ia == ib {
                continue;
            }

            let g_feats = &std_features[g_idx];
            let xa = &g_feats[ia * n_features..(ia + 1) * n_features];
            let xb = &g_feats[ib * n_features..(ib + 1) * n_features];

            // F32-native forward: weights pre-cast in scratch_f32 (refreshed
            // by do_adam_step), x cast once into thread-local buffer.
            // Eliminates ~190 KB of per-pair weight cast + 5 Vec allocs.
            // (Backward stays on the f64 wrapper: arch_backward_f32 added a
            // redundant grad-buffer zero + cast-add pass that measured net
            // slower than the f64 wrapper's in-place accumulate.)
            let fwd_a = {
                let s = scratch_f32.borrow();
                arch_f32::arch_forward_f32(
                    xa,
                    &s,
                    n_features,
                    n_hidden,
                    n_hidden_final,
                    leaky as f32,
                    use_2layer,
                    use_skip,
                )
                .to_archforward()
            };
            let fwd_b = {
                let s = scratch_f32.borrow();
                arch_f32::arch_forward_f32(
                    xb,
                    &s,
                    n_features,
                    n_hidden,
                    n_hidden_final,
                    leaky as f32,
                    use_2layer,
                    use_skip,
                )
                .to_archforward()
            };
            let (ya, dya_dpre) = pin_forward(fwd_a.y);
            let (yb, dyb_dpre) = pin_forward(fwd_b.y);
            let (ya_rank, ya_pool, alpha_a) = (fwd_a.y_rank, fwd_a.y_pool, fwd_a.alpha);
            let (yb_rank, yb_pool, alpha_b) = (fwd_b.y_rank, fwd_b.y_pool, fwd_b.alpha);
            // Aliases for NiN buffer + aux losses that reference old tuple names.
            // NiN buffer stores owned values; clone from the forward result.
            let (ha_pre, ha) = (fwd_a.h_pre.clone(), fwd_a.h.clone());
            let (hb_pre, hb) = (fwd_b.h_pre.clone(), fwd_b.h.clone());
            let (sa, max_a) = (fwd_a.stats, fwd_a.max_idx);
            let (sb, max_b) = (fwd_b.stats, fwd_b.max_idx);

            let mos_a = g.human_scores[ia];
            let mos_b = g.human_scores[ib];
            let target = (mos_a - mos_b).signum();
            if target == 0.0 {
                if nin_on {
                    nin_buffer.push(None);
                    if nin_buffer.len() >= k {
                        flush_per_sample_alpha_nin_batch(
                            &mut nin_buffer,
                            &mut w1,
                            &mut b1,
                            &mut w2_enc,
                            &mut b2_enc,
                            &mut w_skip,
                            &mut b_skip,
                            &mut rank_w,
                            &mut rank_b,
                            &mut reducer_w,
                            &mut reducer_b,
                            &mut w_alpha,
                            &mut b_alpha,
                            &mut adam,
                            n_features,
                            n_hidden,
                            leaky,
                            hyperparams.l2_lambda,
                            hyperparams.norm_in_norm_weight,
                            hyperparams.norm_in_norm_p,
                            hyperparams.norm_in_norm_q,
                            lr,
                            &mut total_loss,
                            &mut n_steps,
                            &do_adam_step,
                        );
                    }
                }
                continue;
            }
            if hyperparams.pwrc_pair_weight
                && hyperparams.pwrc_sensory_threshold > 0.0
                && (mos_a - mos_b).abs() < hyperparams.pwrc_sensory_threshold
            {
                if nin_on {
                    nin_buffer.push(None);
                    if nin_buffer.len() >= k {
                        flush_per_sample_alpha_nin_batch(
                            &mut nin_buffer,
                            &mut w1,
                            &mut b1,
                            &mut w2_enc,
                            &mut b2_enc,
                            &mut w_skip,
                            &mut b_skip,
                            &mut rank_w,
                            &mut rank_b,
                            &mut reducer_w,
                            &mut reducer_b,
                            &mut w_alpha,
                            &mut b_alpha,
                            &mut adam,
                            n_features,
                            n_hidden,
                            leaky,
                            hyperparams.l2_lambda,
                            hyperparams.norm_in_norm_weight,
                            hyperparams.norm_in_norm_p,
                            hyperparams.norm_in_norm_q,
                            lr,
                            &mut total_loss,
                            &mut n_steps,
                            &do_adam_step,
                        );
                    }
                }
                continue;
            }
            let pair_weight = if hyperparams.pwrc_pair_weight {
                pwrc_pair_weight(mos_a, mos_b, hyperparams.pwrc_band_weights.as_deref())
            } else {
                1.0
            };
            let pred_diff = yb - ya;
            let z = -target * pred_diff;
            let loss_raw = if z > 50.0 {
                z
            } else if z < -50.0 {
                0.0
            } else {
                (z.exp() + 1.0).ln()
            };
            total_loss += loss_raw * pair_weight;
            n_steps += 1;
            if dro_active {
                dro_loss_sum[train_pos] += loss_raw * pair_weight;
                dro_loss_n[train_pos] += 1;
            }

            let sig_z = 1.0 / (1.0 + (-z).exp());
            let dl_d_pred_diff = -target * sig_z * pair_weight * hyperparams.ranknet_weight;
            let dl_dya_rn = -dl_d_pred_diff;
            let dl_dyb_rn = dl_d_pred_diff;

            // PreviewV0_5Tuner auxiliary losses (2026-05-18):
            //   1. MSE per-prediction: 2·mse_weight·(y - target)/(2K). The
            //      /(2K) factor (where K=pairs_per_epoch) normalizes the
            //      total per-epoch MSE contribution to be comparable to a
            //      single RankNet pair's gradient magnitude (~O(1)).
            //   2. Monotonicity hinge: penalize y_low - y_high > -margin
            //      among pairs drawn from the same ref_basename group.
            //
            // Both apply only on the plain-RankNet path (NiN composition
            // is queued for a follow-up; the trainer asserts above when
            // NiN + tuner auxes are both requested).
            let (dl_dya_mse, dl_dyb_mse, mse_loss_pair) = if hyperparams.mse_weight > 0.0 {
                let n_norm = (2.0 * hyperparams.pairs_per_epoch.max(1) as f64).max(1.0);
                let base_scale = 2.0 * hyperparams.mse_weight / n_norm;
                // σ-weighted MSE: weight = median(σ_group)/max(σ_i, ε).
                // Within each group, rows where metrics agree (low σ) get
                // up-weighted; rows where metrics disagree (high σ) get
                // down-weighted. Normalizing by median keeps the total
                // gradient magnitude stable across groups with different
                // σ distributions (safesyn σ~0.01 vs KADID σ~0.2).
                let (wa, wb) = if hyperparams.sigma_weighted_mse {
                    if let Some(sigmas) = g.metric_sigmas {
                        let eps = 1e-4;
                        let sigma_med = sigma_medians[g_idx];
                        let sa = sigma_med / sigmas[ia].max(eps);
                        let sb = sigma_med / sigmas[ib].max(eps);
                        // Clamp to [0.2, 5.0] to avoid extreme amplification
                        (sa.clamp(0.2, 5.0), sb.clamp(0.2, 5.0))
                    } else {
                        (1.0, 1.0)
                    }
                } else {
                    (1.0, 1.0)
                };
                let da = base_scale * (ya - mos_a) * wa;
                let db = base_scale * (yb - mos_b) * wb;
                let l = hyperparams.mse_weight
                    * ((ya - mos_a).powi(2) * wa + (yb - mos_b).powi(2) * wb)
                    / n_norm;
                (da, db, l)
            } else {
                (0.0, 0.0, 0.0)
            };
            total_loss += mse_loss_pair;

            let (dl_dya_mono, dl_dyb_mono, mono_loss_pair) =
                if hyperparams.monotonicity_reg > 0.0 && target != 0.0 {
                    // target = signum(mos_a - mos_b); target > 0 means a is hi.
                    let target_gap = (mos_a - mos_b).abs();
                    if target_gap > hyperparams.monotonicity_margin {
                        let (y_hi, y_lo, sign_hi_is_a) = if target > 0.0 {
                            (ya, yb, true)
                        } else {
                            (yb, ya, false)
                        };
                        let violation = (y_lo - y_hi) + hyperparams.monotonicity_margin;
                        if violation > 0.0 {
                            let l = hyperparams.monotonicity_reg * violation * violation;
                            let g_hi = -2.0 * hyperparams.monotonicity_reg * violation;
                            let g_lo = 2.0 * hyperparams.monotonicity_reg * violation;
                            if sign_hi_is_a {
                                (g_hi, g_lo, l)
                            } else {
                                (g_lo, g_hi, l)
                            }
                        } else {
                            (0.0, 0.0, 0.0)
                        }
                    } else {
                        (0.0, 0.0, 0.0)
                    }
                } else {
                    (0.0, 0.0, 0.0)
                };
            total_loss += mono_loss_pair;

            if nin_on {
                nin_buffer.push(Some(PerSampleAlphaPairForward {
                    xa,
                    xb,
                    ya,
                    yb,
                    ya_rank,
                    yb_rank,
                    ya_pool,
                    yb_pool,
                    alpha_a,
                    alpha_b,
                    ha_pre,
                    ha,
                    hb_pre,
                    hb,
                    sa,
                    sb,
                    max_a,
                    max_b,
                    dl_dya_rn,
                    dl_dyb_rn,
                    mos_a,
                    mos_b,
                }));
                if nin_buffer.len() >= k {
                    flush_per_sample_alpha_nin_batch(
                        &mut nin_buffer,
                        &mut w1,
                        &mut b1,
                        &mut w2_enc,
                        &mut b2_enc,
                        &mut w_skip,
                        &mut b_skip,
                        &mut rank_w,
                        &mut rank_b,
                        &mut reducer_w,
                        &mut reducer_b,
                        &mut w_alpha,
                        &mut b_alpha,
                        &mut adam,
                        n_features,
                        n_hidden,
                        leaky,
                        hyperparams.l2_lambda,
                        hyperparams.norm_in_norm_weight,
                        hyperparams.norm_in_norm_p,
                        hyperparams.norm_in_norm_q,
                        lr,
                        &mut total_loss,
                        &mut n_steps,
                        &do_adam_step,
                    );
                }
                continue;
            }

            // Plain RankNet path (no NiN).
            // PreviewV0_5Tuner: fold MSE + monotonicity gradients into
            // dl_dy* before the backprop. RankNet contribution is already
            // weighted by hyperparams.ranknet_weight in dl_d_pred_diff.
            // V4 tanh pin: multiply dL/dy_score by dy_score/dy_pre to
            // get dL/dy_pre for the per-sample-α backprop.
            let dl_dya = (dl_dya_rn + dl_dya_mse + dl_dya_mono) * dya_dpre;
            let dl_dyb = (dl_dyb_rn + dl_dyb_mse + dl_dyb_mono) * dyb_dpre;
            steps_since_adam += 1;

            let mut g_rank_w_buf = vec![0.0f64; n_hidden_final];
            let mut g_rank_b_buf = 0.0f64;
            let mut g_red_w: [f64; 4] = [0.0; 4];
            let mut g_red_b: f64 = 0.0;
            let mut g_w_alpha_buf = vec![0.0f64; n_hidden_final];
            let mut g_b_alpha: f64 = 0.0;

            arch_backward(
                xa,
                &fwd_a,
                dl_dya,
                &w1,
                &w2_enc,
                &rank_w,
                &reducer_w,
                &w_alpha,
                &mut adam.gw1,
                &mut adam.gb1,
                &mut g_rank_w_buf,
                &mut g_rank_b_buf,
                &mut g_red_w,
                &mut g_red_b,
                &mut g_w_alpha_buf,
                &mut g_b_alpha,
                n_features,
                n_hidden,
                n_hidden_final,
                leaky,
                use_2layer,
                use_skip,
            );
            arch_backward(
                xb,
                &fwd_b,
                dl_dyb,
                &w1,
                &w2_enc,
                &rank_w,
                &reducer_w,
                &w_alpha,
                &mut adam.gw1,
                &mut adam.gb1,
                &mut g_rank_w_buf,
                &mut g_rank_b_buf,
                &mut g_red_w,
                &mut g_red_b,
                &mut g_w_alpha_buf,
                &mut g_b_alpha,
                n_features,
                n_hidden,
                n_hidden_final,
                leaky,
                use_2layer,
                use_skip,
            );

            if hyperparams.l2_lambda > 0.0 {
                let l2 = hyperparams.l2_lambda;
                for (g, &w) in adam.gw1.iter_mut().zip(w1.iter()) {
                    *g += l2 * w;
                }
                for j in 0..n_hidden_final {
                    g_rank_w_buf[j] += l2 * rank_w[j];
                    g_w_alpha_buf[j] += l2 * w_alpha[j];
                }
                for kk in 0..4 {
                    g_red_w[kk] += l2 * reducer_w[kk];
                }
            }

            // Fold per-pair grads into Adam w2/b2 slots.
            for j in 0..n_hidden_final {
                adam.gw2[j] += g_rank_w_buf[j];
            }
            for kk in 0..4 {
                adam.gw2[n_hidden_final + kk] += g_red_w[kk];
            }
            for j in 0..n_hidden_final {
                adam.gw2[n_hidden_final + 4 + j] += g_w_alpha_buf[j];
            }
            adam.gw2[n_hidden_final + 4 + n_hidden_final] += g_b_alpha;
            adam.gb2[0] += g_rank_b_buf;
            adam.gb2[1] += g_red_b;

            if k == 1 || steps_since_adam >= k as u64 {
                do_adam_step(
                    &mut adam,
                    &mut w1,
                    &mut b1,
                    &mut w2_enc,
                    &mut b2_enc,
                    &mut w_skip,
                    &mut b_skip,
                    &mut rank_w,
                    &mut rank_b,
                    &mut reducer_w,
                    &mut reducer_b,
                    &mut w_alpha,
                    &mut b_alpha,
                    lr,
                    n_hidden_final,
                );
                steps_since_adam = 0;
            }

            // PreviewV0_5TunerV2 anchor step (2026-05-19).
            // After each pair Adam step, with probability anchor_step_p,
            // sample K anchor rows, forward each, and apply
            // w · (y - target)² MSE gradients via K backprop steps
            // sharing the same Adam state (single Adam step at end).
            // The anchor row's weight scales BOTH the loss and the
            // gradient (so KonJND PJND rows pull harder than synthetic
            // rows). The implicit "scale across all anchors" is the
            // probability `anchor_step_p` itself.
            //
            // SPEED-B (2026-05-19): K-batched. At K=1 fires every
            // iteration with 1 sample (bit-identical to pre-SPEED-B);
            // at K=32 fires on Adam-step boundaries (post-flush) with
            // 32 samples per fire. Per-pair aux work rate is preserved
            // because the main pair-loop fires ~pairs_per_epoch/K
            // boundaries while each aux fire processes K samples.
            if anchor_active && steps_since_adam == 0 {
                let u_take = rng.next_f64_unit();
                if u_take < hyperparams.anchor_step_p
                    && let Some(a) = anchor
                    && !std_anchor_features.is_empty()
                    && anchor_total_weight > 0.0
                {
                    let n_anchor = std_anchor_features.len();
                    let mut g_rank_w_buf = vec![0.0f64; n_hidden_final];
                    let mut g_rank_b_buf = 0.0f64;
                    let mut g_red_w: [f64; 4] = [0.0; 4];
                    let mut g_red_b: f64 = 0.0;
                    let mut g_w_alpha_buf = vec![0.0f64; n_hidden_final];
                    let mut g_b_alpha: f64 = 0.0;
                    let mut any_step = false;

                    for _ in 0..k {
                        let u_row = rng.next_f64_unit();
                        let ai = anchor_row_cdf
                            .partition_point(|&c| c < u_row)
                            .min(n_anchor - 1);
                        let xa = std_anchor_features[ai].as_slice();
                        let fwd_anc = arch_forward(
                            xa,
                            &w1,
                            &b1,
                            &w2_enc,
                            &b2_enc,
                            &w_skip,
                            b_skip,
                            &rank_w,
                            rank_b,
                            &reducer_w,
                            reducer_b,
                            &w_alpha,
                            b_alpha,
                            n_features,
                            n_hidden,
                            n_hidden_final,
                            leaky,
                            use_2layer,
                            use_skip,
                        );
                        let (ya, dya_dpre) = pin_forward(fwd_anc.y);
                        let target = a
                            .target_scores
                            .and_then(|ts| ts.get(ai).copied())
                            .unwrap_or(hyperparams.anchor_target_score);
                        let row_w = a.row_weights[ai];
                        let err = ya - target;
                        let scale = 2.0 * hyperparams.anchor_loss_weight * row_w;
                        let dl_dy = scale * err * dya_dpre;
                        let loss = hyperparams.anchor_loss_weight * row_w * err * err;
                        total_loss += loss;
                        n_steps += 1;

                        arch_backward(
                            xa,
                            &fwd_anc,
                            dl_dy,
                            &w1,
                            &w2_enc,
                            &rank_w,
                            &reducer_w,
                            &w_alpha,
                            &mut adam.gw1,
                            &mut adam.gb1,
                            &mut g_rank_w_buf,
                            &mut g_rank_b_buf,
                            &mut g_red_w,
                            &mut g_red_b,
                            &mut g_w_alpha_buf,
                            &mut g_b_alpha,
                            n_features,
                            n_hidden,
                            n_hidden_final,
                            leaky,
                            use_2layer,
                            use_skip,
                        );
                        any_step = true;
                    }

                    if any_step {
                        if hyperparams.l2_lambda > 0.0 {
                            let l2 = hyperparams.l2_lambda;
                            for (g, &w) in adam.gw1.iter_mut().zip(w1.iter()) {
                                *g += l2 * w;
                            }
                            for j in 0..n_hidden_final {
                                g_rank_w_buf[j] += l2 * rank_w[j];
                                g_w_alpha_buf[j] += l2 * w_alpha[j];
                            }
                            for kk in 0..4 {
                                g_red_w[kk] += l2 * reducer_w[kk];
                            }
                        }

                        for j in 0..n_hidden_final {
                            adam.gw2[j] += g_rank_w_buf[j];
                        }
                        for kk in 0..4 {
                            adam.gw2[n_hidden_final + kk] += g_red_w[kk];
                        }
                        for j in 0..n_hidden_final {
                            adam.gw2[n_hidden_final + 4 + j] += g_w_alpha_buf[j];
                        }
                        adam.gw2[n_hidden_final + 4 + n_hidden_final] += g_b_alpha;
                        adam.gb2[0] += g_rank_b_buf;
                        adam.gb2[1] += g_red_b;

                        do_adam_step(
                            &mut adam,
                            &mut w1,
                            &mut b1,
                            &mut w2_enc,
                            &mut b2_enc,
                            &mut w_skip,
                            &mut b_skip,
                            &mut rank_w,
                            &mut rank_b,
                            &mut reducer_w,
                            &mut reducer_b,
                            &mut w_alpha,
                            &mut b_alpha,
                            lr,
                            n_hidden_final,
                        );
                    }
                }
            }

            // Within-ladder monotonicity TV step (KADIS ladders, 2026-07-01).
            // For each sampled pair (lo = milder/higher-quality, hi = harsher/
            // lower-quality) apply a hinge penalty max(0, y_hi − y_lo): push the
            // harsher sample DOWN and the milder UP only when the metric has them
            // out of order. Supervises monotonicity ONLY on real distortion
            // ladders, leaving codec-rank feature combinations FREE (no global
            // sign-mask) — the targeted middle the cbc-vs-multicodec tension
            // needs. Fires at Adam boundaries like the anchor/pjnd/equiv aux
            // steps; consumes RNG only when tv_active, so non-TV bakes stay
            // bit-identical.
            if tv_active && steps_since_adam == 0 && !tv_pairs_vec.is_empty() {
                let n_tv = tv_pairs_vec.len();
                let mut g_rank_w_buf = vec![0.0f64; n_hidden_final];
                let mut g_rank_b_buf = 0.0f64;
                let mut g_red_w: [f64; 4] = [0.0; 4];
                let mut g_red_b: f64 = 0.0;
                let mut g_w_alpha_buf = vec![0.0f64; n_hidden_final];
                let mut g_b_alpha: f64 = 0.0;
                let mut any_step = false;

                for _ in 0..tv_batch_val {
                    let pick = (rng.next_f64_unit() * n_tv as f64) as usize;
                    let (lo, hi) = tv_pairs_vec[pick.min(n_tv - 1)];
                    let xlo = std_tv_features[lo].as_slice();
                    let xhi = std_tv_features[hi].as_slice();
                    let fwd_lo = arch_forward(
                        xlo,
                        &w1,
                        &b1,
                        &w2_enc,
                        &b2_enc,
                        &w_skip,
                        b_skip,
                        &rank_w,
                        rank_b,
                        &reducer_w,
                        reducer_b,
                        &w_alpha,
                        b_alpha,
                        n_features,
                        n_hidden,
                        n_hidden_final,
                        leaky,
                        use_2layer,
                        use_skip,
                    );
                    let (ylo, dylo_dpre) = pin_forward(fwd_lo.y);
                    let fwd_hi = arch_forward(
                        xhi,
                        &w1,
                        &b1,
                        &w2_enc,
                        &b2_enc,
                        &w_skip,
                        b_skip,
                        &rank_w,
                        rank_b,
                        &reducer_w,
                        reducer_b,
                        &w_alpha,
                        b_alpha,
                        n_features,
                        n_hidden,
                        n_hidden_final,
                        leaky,
                        use_2layer,
                        use_skip,
                    );
                    let (yhi, dyhi_dpre) = pin_forward(fwd_hi.y);
                    // Anti-collapse margin: enforce y_milder - y_harsher >= margin
                    // (a minimum per-step gap) rather than merely non-increasing.
                    // margin=0 is the pure hinge. The +margin is a constant offset,
                    // so d(viol)/dy is unchanged — the gradient direction is identical;
                    // it just fires on correctly-ordered pairs whose gap is too small,
                    // spreading the ladder instead of letting it collapse flat.
                    let viol = yhi - ylo + tv_margin_val;
                    if viol > 0.0 {
                        total_loss += tv_weight_val * viol;
                        n_steps += 1;
                        let dl_dy_hi = tv_weight_val * dyhi_dpre; // push y_hi down
                        let dl_dy_lo = -tv_weight_val * dylo_dpre; // push y_lo up
                        arch_backward(
                            xhi,
                            &fwd_hi,
                            dl_dy_hi,
                            &w1,
                            &w2_enc,
                            &rank_w,
                            &reducer_w,
                            &w_alpha,
                            &mut adam.gw1,
                            &mut adam.gb1,
                            &mut g_rank_w_buf,
                            &mut g_rank_b_buf,
                            &mut g_red_w,
                            &mut g_red_b,
                            &mut g_w_alpha_buf,
                            &mut g_b_alpha,
                            n_features,
                            n_hidden,
                            n_hidden_final,
                            leaky,
                            use_2layer,
                            use_skip,
                        );
                        arch_backward(
                            xlo,
                            &fwd_lo,
                            dl_dy_lo,
                            &w1,
                            &w2_enc,
                            &rank_w,
                            &reducer_w,
                            &w_alpha,
                            &mut adam.gw1,
                            &mut adam.gb1,
                            &mut g_rank_w_buf,
                            &mut g_rank_b_buf,
                            &mut g_red_w,
                            &mut g_red_b,
                            &mut g_w_alpha_buf,
                            &mut g_b_alpha,
                            n_features,
                            n_hidden,
                            n_hidden_final,
                            leaky,
                            use_2layer,
                            use_skip,
                        );
                        any_step = true;
                    }
                }

                if any_step {
                    for j in 0..n_hidden_final {
                        adam.gw2[j] += g_rank_w_buf[j];
                    }
                    for kk in 0..4 {
                        adam.gw2[n_hidden_final + kk] += g_red_w[kk];
                    }
                    for j in 0..n_hidden_final {
                        adam.gw2[n_hidden_final + 4 + j] += g_w_alpha_buf[j];
                    }
                    adam.gw2[n_hidden_final + 4 + n_hidden_final] += g_b_alpha;
                    adam.gb2[0] += g_rank_b_buf;
                    adam.gb2[1] += g_red_b;

                    do_adam_step(
                        &mut adam,
                        &mut w1,
                        &mut b1,
                        &mut w2_enc,
                        &mut b2_enc,
                        &mut w_skip,
                        &mut b_skip,
                        &mut rank_w,
                        &mut rank_b,
                        &mut reducer_w,
                        &mut reducer_b,
                        &mut w_alpha,
                        &mut b_alpha,
                        lr,
                        n_hidden_final,
                    );
                }
            }

            // EXP-V11-D-PJND-DOMINANT PJND-passthrough step
            // (2026-05-20, task #198). Structurally identical to the
            // V11 cross-codec-eq anchor step above — second anchor
            // pool with its own weight + step_p + target. Fires
            // independently of the primary anchor; both may fire on
            // the same Adam boundary so each contributes to the
            // gradient accumulation pre-Adam-step.
            if pjnd_active && steps_since_adam == 0 {
                let u_take = rng.next_f64_unit();
                if u_take < hyperparams.pjnd_passthrough_step_p
                    && let Some(pa) = pjnd_anchor
                    && !std_pjnd_features.is_empty()
                    && pjnd_total_weight > 0.0
                {
                    let n_pjnd = std_pjnd_features.len();
                    let mut g_rank_w_buf = vec![0.0f64; n_hidden_final];
                    let mut g_rank_b_buf = 0.0f64;
                    let mut g_red_w: [f64; 4] = [0.0; 4];
                    let mut g_red_b: f64 = 0.0;
                    let mut g_w_alpha_buf = vec![0.0f64; n_hidden_final];
                    let mut g_b_alpha: f64 = 0.0;
                    let mut any_step = false;

                    for _ in 0..k {
                        let u_row = rng.next_f64_unit();
                        let pi = pjnd_row_cdf.partition_point(|&c| c < u_row).min(n_pjnd - 1);
                        let xa = std_pjnd_features[pi].as_slice();
                        let (
                            ya_pre,
                            ya_rank_a,
                            ya_pool_a,
                            alpha_a_a,
                            _,
                            ha_pre_a,
                            ha_a,
                            sa_a,
                            max_a_a,
                        ) = psah::forward_per_sample_alpha_head(
                            xa, &w1, &b1, &rank_w, rank_b, &reducer_w, reducer_b, &w_alpha,
                            b_alpha, n_features, n_hidden, leaky,
                        );
                        let (ya, dya_dpre) = pin_forward(ya_pre);
                        let target = pa
                            .target_scores
                            .and_then(|ts| ts.get(pi).copied())
                            .unwrap_or(hyperparams.pjnd_passthrough_target_score);
                        let row_w = pa.row_weights[pi];
                        let err = ya - target;
                        let scale = 2.0 * hyperparams.pjnd_passthrough_weight * row_w;
                        let dl_dy = scale * err * dya_dpre;
                        let loss = hyperparams.pjnd_passthrough_weight * row_w * err * err;
                        total_loss += loss;
                        n_steps += 1;

                        psah::backprop_step_per_sample_alpha_head(
                            xa,
                            &ha_pre_a,
                            &ha_a,
                            &sa_a,
                            max_a_a,
                            ya_rank_a,
                            ya_pool_a,
                            alpha_a_a,
                            dl_dy,
                            &rank_w,
                            &reducer_w,
                            &w_alpha,
                            &mut adam.gw1,
                            &mut adam.gb1,
                            &mut g_rank_w_buf,
                            &mut g_rank_b_buf,
                            &mut g_red_w,
                            &mut g_red_b,
                            &mut g_w_alpha_buf,
                            &mut g_b_alpha,
                            n_features,
                            n_hidden,
                            leaky,
                        );
                        any_step = true;
                    }

                    if any_step {
                        if hyperparams.l2_lambda > 0.0 {
                            let l2 = hyperparams.l2_lambda;
                            for (g, &w) in adam.gw1.iter_mut().zip(w1.iter()) {
                                *g += l2 * w;
                            }
                            for j in 0..n_hidden_final {
                                g_rank_w_buf[j] += l2 * rank_w[j];
                                g_w_alpha_buf[j] += l2 * w_alpha[j];
                            }
                            for kk in 0..4 {
                                g_red_w[kk] += l2 * reducer_w[kk];
                            }
                        }

                        for j in 0..n_hidden_final {
                            adam.gw2[j] += g_rank_w_buf[j];
                        }
                        for kk in 0..4 {
                            adam.gw2[n_hidden_final + kk] += g_red_w[kk];
                        }
                        for j in 0..n_hidden_final {
                            adam.gw2[n_hidden_final + 4 + j] += g_w_alpha_buf[j];
                        }
                        adam.gw2[n_hidden_final + 4 + n_hidden_final] += g_b_alpha;
                        adam.gb2[0] += g_rank_b_buf;
                        adam.gb2[1] += g_red_b;

                        do_adam_step(
                            &mut adam,
                            &mut w1,
                            &mut b1,
                            &mut w2_enc,
                            &mut b2_enc,
                            &mut w_skip,
                            &mut b_skip,
                            &mut rank_w,
                            &mut rank_b,
                            &mut reducer_w,
                            &mut reducer_b,
                            &mut w_alpha,
                            &mut b_alpha,
                            lr,
                            n_hidden,
                        );
                    }
                }
            }

            // KONJND-AGGREGATION-HEAD step (task #4, 2026-05-24).
            // For each fire: sample K refs uniformly from
            // konjnd-dense's ref_ranges; for each ref, sample S of its
            // distortion-level rows; forward all K·S; compute K
            // per-ref aggregates (mean over S); apply MSE against the
            // per-ref pjnd_target; backprop each row's contribution
            // with gradient (2w/S)·residual scaled by tanh-pin Jacobian.
            //
            // Mechanism rationale: the existing pjnd_passthrough fires
            // a per-row regression, which V11-D proved is structurally
            // incompatible with the per-source-constant pjnd_target
            // (zero-gradient pathology + per-source-mean collapse).
            // Aggregating predictions before computing the loss
            // restores a non-zero within-ref gradient and decouples
            // the per-pair feature → distortion-level mapping from
            // the per-ref target constraint. See
            // `docs/KONJND_AGGREGATION_HEAD_DESIGN_2026-05-24.md`.
            if konjnd_agg_active && steps_since_adam == 0 {
                let u_take = rng.next_f64_unit();
                if u_take < hyperparams.konjnd_aggregation_step_p
                    && let Some(ka) = konjnd_agg
                    && !std_konjnd_agg_features.is_empty()
                    && !ka.ref_ranges.is_empty()
                {
                    let n_refs = ka.ref_ranges.len();
                    let s_per_ref = hyperparams.konjnd_aggregation_samples_per_ref.max(1);
                    let k_refs = hyperparams.konjnd_aggregation_refs_per_step.max(1);
                    let inv_s = 1.0 / (s_per_ref as f64);
                    let mut g_rank_w_buf = vec![0.0f64; n_hidden_final];
                    let mut g_rank_b_buf = 0.0f64;
                    let mut g_red_w: [f64; 4] = [0.0; 4];
                    let mut g_red_b: f64 = 0.0;
                    let mut g_w_alpha_buf = vec![0.0f64; n_hidden_final];
                    let mut g_b_alpha: f64 = 0.0;
                    let mut any_step = false;

                    // Stash forward-pass intermediates per (ref, s).
                    // arch_backward consumes the full ArchForward
                    // struct (which carries h_pre / h / stats /
                    // max_idx + the 2-layer h1_pre / h1 + skip
                    // state), so we cache (ri_global, ya, dya_dpre,
                    // ArchForward) per row and replay it in the
                    // backprop pass. This is the 2-layer / skip
                    // generalization of the prior 1-layer-only
                    // psah cache.
                    let mut fw_cache: Vec<(
                        usize,       // row index into std_konjnd_agg_features
                        f64,         // ya (post-pin score)
                        f64,         // dya_dpre (pin Jacobian)
                        ArchForward, // full forward intermediates
                    )> = Vec::with_capacity(k_refs * s_per_ref);

                    // For each picked ref: sample S rows + forward all S.
                    // (fw_cache_start_idx, S_actual, sum_y, target)
                    let mut ref_sums: Vec<(usize, usize, f64, f64)> = Vec::with_capacity(k_refs);

                    for _ in 0..k_refs {
                        let u_ref = rng.next_f64_unit();
                        let ri = ((u_ref * (n_refs as f64)) as usize).min(n_refs - 1);
                        let (row_start, n_rows_in_ref) = ka.ref_ranges[ri];
                        if n_rows_in_ref == 0 {
                            continue;
                        }
                        let target = ka.ref_pjnd_target[ri];
                        let cache_start = fw_cache.len();
                        let mut sum_y = 0.0f64;
                        let s_actual = s_per_ref.min(n_rows_in_ref);
                        for _ in 0..s_actual {
                            let u_row = rng.next_f64_unit();
                            let off =
                                ((u_row * (n_rows_in_ref as f64)) as usize).min(n_rows_in_ref - 1);
                            let ri_global = row_start + off;
                            let xa = std_konjnd_agg_features[ri_global].as_slice();
                            let fwd_kah = arch_forward(
                                xa,
                                &w1,
                                &b1,
                                &w2_enc,
                                &b2_enc,
                                &w_skip,
                                b_skip,
                                &rank_w,
                                rank_b,
                                &reducer_w,
                                reducer_b,
                                &w_alpha,
                                b_alpha,
                                n_features,
                                n_hidden,
                                n_hidden_final,
                                leaky,
                                use_2layer,
                                use_skip,
                            );
                            let (ya, dya_dpre) = pin_forward(fwd_kah.y);
                            sum_y += ya;
                            fw_cache.push((ri_global, ya, dya_dpre, fwd_kah));
                        }
                        ref_sums.push((cache_start, s_actual, sum_y, target));
                    }

                    // Backprop: each ref's residual contributes
                    // (2w/S)·residual to every cached row in
                    // that ref. Loss = w · Σ_r (agg_r − t_r)².
                    for &(cache_start, s_actual, sum_y, target) in &ref_sums {
                        if s_actual == 0 {
                            continue;
                        }
                        let inv_s_actual = 1.0 / (s_actual as f64);
                        let agg = sum_y * inv_s_actual;
                        let err = agg - target;
                        let scale = 2.0 * hyperparams.konjnd_aggregation_weight * inv_s_actual;
                        let loss = hyperparams.konjnd_aggregation_weight * err * err;
                        total_loss += loss;
                        n_steps += 1;
                        for j in 0..s_actual {
                            let entry = &fw_cache[cache_start + j];
                            let ri_global = entry.0;
                            let dya_dpre = entry.2;
                            let fwd_kah = &entry.3;
                            let xa = std_konjnd_agg_features[ri_global].as_slice();
                            let dl_dy = scale * err * dya_dpre;
                            arch_backward(
                                xa,
                                fwd_kah,
                                dl_dy,
                                &w1,
                                &w2_enc,
                                &rank_w,
                                &reducer_w,
                                &w_alpha,
                                &mut adam.gw1,
                                &mut adam.gb1,
                                &mut g_rank_w_buf,
                                &mut g_rank_b_buf,
                                &mut g_red_w,
                                &mut g_red_b,
                                &mut g_w_alpha_buf,
                                &mut g_b_alpha,
                                n_features,
                                n_hidden,
                                n_hidden_final,
                                leaky,
                                use_2layer,
                                use_skip,
                            );
                            any_step = true;
                        }
                        let _ = inv_s; // reserved for variants where S is fixed across refs
                    }

                    if any_step {
                        if hyperparams.l2_lambda > 0.0 {
                            let l2 = hyperparams.l2_lambda;
                            for (g, &w) in adam.gw1.iter_mut().zip(w1.iter()) {
                                *g += l2 * w;
                            }
                            for j in 0..n_hidden_final {
                                g_rank_w_buf[j] += l2 * rank_w[j];
                                g_w_alpha_buf[j] += l2 * w_alpha[j];
                            }
                            for kk in 0..4 {
                                g_red_w[kk] += l2 * reducer_w[kk];
                            }
                        }

                        for j in 0..n_hidden_final {
                            adam.gw2[j] += g_rank_w_buf[j];
                        }
                        for kk in 0..4 {
                            adam.gw2[n_hidden_final + kk] += g_red_w[kk];
                        }
                        for j in 0..n_hidden_final {
                            adam.gw2[n_hidden_final + 4 + j] += g_w_alpha_buf[j];
                        }
                        adam.gw2[n_hidden_final + 4 + n_hidden_final] += g_b_alpha;
                        adam.gb2[0] += g_rank_b_buf;
                        adam.gb2[1] += g_red_b;

                        do_adam_step(
                            &mut adam,
                            &mut w1,
                            &mut b1,
                            &mut w2_enc,
                            &mut b2_enc,
                            &mut w_skip,
                            &mut b_skip,
                            &mut rank_w,
                            &mut rank_b,
                            &mut reducer_w,
                            &mut reducer_b,
                            &mut w_alpha,
                            &mut b_alpha,
                            lr,
                            n_hidden_final,
                        );
                    }
                }
            }

            // EXP-CROSS-CODEC-METRIC equivalence step (2026-05-19).
            // After each pair Adam step, with probability eq_step_p,
            // sample K (A, B) equivalence pairs, forward both halves
            // of each, and apply w · (y_a - y_b)² MSE gradient via
            // 2K backprops sharing the same Adam state (single Adam
            // step at end). Includes the optional rank-preserve
            // regularizer driven by per-pair butter_diff.
            //
            // SPEED-B (2026-05-19): K-batched. At K=1 fires every
            // iteration with 1 pair (bit-identical to pre-SPEED-B);
            // at K=32 fires on Adam-step boundaries (post-flush) with
            // 32 pairs per fire. Per-pair aux work rate is preserved.
            if equiv_active && steps_since_adam == 0 {
                let u_take = rng.next_f64_unit();
                if u_take < hyperparams.cross_codec_eq_step_p
                    && let Some(e) = equiv
                    && !std_equiv_a.is_empty()
                    && equiv_total_weight > 0.0
                {
                    let n_equiv = std_equiv_a.len();
                    let mut g_rank_w_buf = vec![0.0f64; n_hidden_final];
                    let mut g_rank_b_buf = 0.0f64;
                    let mut g_red_w: [f64; 4] = [0.0; 4];
                    let mut g_red_b: f64 = 0.0;
                    let mut g_w_alpha_buf = vec![0.0f64; n_hidden_final];
                    let mut g_b_alpha: f64 = 0.0;
                    let mut any_step = false;

                    for _ in 0..k {
                        let u_row = rng.next_f64_unit();
                        let ei = equiv_row_cdf
                            .partition_point(|&c| c < u_row)
                            .min(n_equiv - 1);
                        let xa = std_equiv_a[ei].as_slice();
                        let xb = std_equiv_b[ei].as_slice();

                        // Forward both halves.
                        let (ya_pre, ya_rank, ya_pool, alpha_a, _, ha_pre, ha, sa, max_a) =
                            psah::forward_per_sample_alpha_head(
                                xa, &w1, &b1, &rank_w, rank_b, &reducer_w, reducer_b, &w_alpha,
                                b_alpha, n_features, n_hidden, leaky,
                            );
                        let (yb_pre, yb_rank, yb_pool, alpha_b, _, hb_pre, hb, sb, max_b) =
                            psah::forward_per_sample_alpha_head(
                                xb, &w1, &b1, &rank_w, rank_b, &reducer_w, reducer_b, &w_alpha,
                                b_alpha, n_features, n_hidden, leaky,
                            );
                        // V4 tanh pin (identity when off).
                        let (ya, dya_dpre) = pin_forward(ya_pre);
                        let (yb, dyb_dpre) = pin_forward(yb_pre);
                        let row_w = e.row_weights[ei];
                        let diff = ya - yb;
                        // L_eq = w · row_w · diff²
                        // dL_eq/dy_a (score) = 2 · w · row_w · diff
                        // dL_eq/dy_b (score) = −2 · w · row_w · diff
                        // Accumulate in score-space; chain-rule to y_pre
                        // before backprop.
                        let scale = 2.0 * hyperparams.cross_codec_eq_weight * row_w;
                        let mut dl_dya_score = scale * diff;
                        let mut dl_dyb_score = -scale * diff;
                        let mut loss = hyperparams.cross_codec_eq_weight * row_w * diff * diff;

                        // EXP-CROSS-CODEC-V3 (2026-05-19): rank-preserve
                        // regularizer on equiv pairs that have butter_diff.
                        // Loss = w_rp · |Δb| · −log(sigmoid(s · (y_b − y_a)))
                        // where s = sign(Δb) and Δb = butter_a − butter_b.
                        // Δb > 0 → A is butter-worse than B → we want
                        // y_a < y_b (score-shape, HIGHER quality = higher
                        // output) → use logit s·(y_b − y_a) in the
                        // sigmoid. This is the same RankNet derivation
                        // used for the main pair loss above.
                        //
                        // Gradient (let u = s·(y_b − y_a), σ = sigmoid(u),
                        // w = w_rp · |Δb|):
                        //   dL/dy_b = -w · s · (1 − σ)
                        //   dL/dy_a = +w · s · (1 − σ)
                        if hyperparams.cross_codec_rank_preserve_weight > 0.0
                            && !e.butter_diff.is_empty()
                            && ei < e.butter_diff.len()
                        {
                            let db = e.butter_diff[ei];
                            if db.is_finite() && db != 0.0 {
                                let s = if db > 0.0 { 1.0 } else { -1.0 };
                                let abs_db = db.abs();
                                let w_rp = hyperparams.cross_codec_rank_preserve_weight * abs_db;
                                let u = s * (yb - ya);
                                // softplus(-u) = log(1 + exp(-u)) =
                                //   -log(sigmoid(u)). Stable form:
                                let softplus = if u >= 0.0 {
                                    (-u).exp().ln_1p()
                                } else {
                                    -u + u.exp().ln_1p()
                                };
                                let l_rp = w_rp * softplus;
                                // sigmoid(u) numerically stable:
                                let sig = if u >= 0.0 {
                                    1.0 / (1.0 + (-u).exp())
                                } else {
                                    let eu = u.exp();
                                    eu / (1.0 + eu)
                                };
                                // dL/dy_b = -w · s · (1 − σ)
                                // dL/dy_a = +w · s · (1 − σ)
                                let g = w_rp * s * (1.0 - sig);
                                dl_dya_score += g;
                                dl_dyb_score -= g;
                                loss += l_rp;
                            }
                        }
                        total_loss += loss;
                        n_steps += 1;

                        // V4 tanh pin: chain-rule dL/dy_score → dL/dy_pre.
                        let dl_dya = dl_dya_score * dya_dpre;
                        let dl_dyb = dl_dyb_score * dyb_dpre;

                        // Backprop through A (accumulates into shared buffers).
                        psah::backprop_step_per_sample_alpha_head(
                            xa,
                            &ha_pre,
                            &ha,
                            &sa,
                            max_a,
                            ya_rank,
                            ya_pool,
                            alpha_a,
                            dl_dya,
                            &rank_w,
                            &reducer_w,
                            &w_alpha,
                            &mut adam.gw1,
                            &mut adam.gb1,
                            &mut g_rank_w_buf,
                            &mut g_rank_b_buf,
                            &mut g_red_w,
                            &mut g_red_b,
                            &mut g_w_alpha_buf,
                            &mut g_b_alpha,
                            n_features,
                            n_hidden,
                            leaky,
                        );

                        // Backprop through B (accumulates into same buffers).
                        psah::backprop_step_per_sample_alpha_head(
                            xb,
                            &hb_pre,
                            &hb,
                            &sb,
                            max_b,
                            yb_rank,
                            yb_pool,
                            alpha_b,
                            dl_dyb,
                            &rank_w,
                            &reducer_w,
                            &w_alpha,
                            &mut adam.gw1,
                            &mut adam.gb1,
                            &mut g_rank_w_buf,
                            &mut g_rank_b_buf,
                            &mut g_red_w,
                            &mut g_red_b,
                            &mut g_w_alpha_buf,
                            &mut g_b_alpha,
                            n_features,
                            n_hidden,
                            leaky,
                        );
                        any_step = true;
                    }

                    if any_step {
                        if hyperparams.l2_lambda > 0.0 {
                            let l2 = hyperparams.l2_lambda;
                            for (g, &w) in adam.gw1.iter_mut().zip(w1.iter()) {
                                *g += l2 * w;
                            }
                            for j in 0..n_hidden_final {
                                g_rank_w_buf[j] += l2 * rank_w[j];
                                g_w_alpha_buf[j] += l2 * w_alpha[j];
                            }
                            for kk in 0..4 {
                                g_red_w[kk] += l2 * reducer_w[kk];
                            }
                        }

                        for j in 0..n_hidden_final {
                            adam.gw2[j] += g_rank_w_buf[j];
                        }
                        for kk in 0..4 {
                            adam.gw2[n_hidden_final + kk] += g_red_w[kk];
                        }
                        for j in 0..n_hidden_final {
                            adam.gw2[n_hidden_final + 4 + j] += g_w_alpha_buf[j];
                        }
                        adam.gw2[n_hidden_final + 4 + n_hidden_final] += g_b_alpha;
                        adam.gb2[0] += g_rank_b_buf;
                        adam.gb2[1] += g_red_b;

                        do_adam_step(
                            &mut adam,
                            &mut w1,
                            &mut b1,
                            &mut w2_enc,
                            &mut b2_enc,
                            &mut w_skip,
                            &mut b_skip,
                            &mut rank_w,
                            &mut rank_b,
                            &mut reducer_w,
                            &mut reducer_b,
                            &mut w_alpha,
                            &mut b_alpha,
                            lr,
                            n_hidden,
                        );
                    }
                }
            }

            // EXP-CROSS-CODEC-V3 (2026-05-19) dynamic-range floor probe.
            // With probability `dynamic_range_step_p`, sample `probe_n`
            // random A-side rows from the equiv pool, forward each, and
            // penalize the network if the std deviation of the outputs
            // falls below `sigma_threshold`. Penalty is quadratic:
            //   L_dr = w · max(0, σ_threshold − σ_obs)²
            // The gradient w.r.t. each per-row output `y_i` is:
            //   dL_dr/dy_i = -2 · w · max(0, σ_threshold − σ_obs)
            //                  · (y_i − μ) / (σ · N)
            // All N per-row gradients accumulate into one Adam step.
            //
            // Requires equiv pool (uses A-side as the probe substrate);
            // skipped silently if equiv pool empty.
            // SPEED-B (2026-05-19): dyn-range step already uses
            // probe_n samples in a batch internally, so we only
            // need to gate on Adam-step boundaries (post-flush)
            // to avoid polluting mid-batch accumulation. K=1 path
            // is preserved (boundary fires every iteration).
            if hyperparams.dynamic_range_floor_weight > 0.0
                && steps_since_adam == 0
                && !std_equiv_a.is_empty()
            {
                let u_take = rng.next_f64_unit();
                if u_take < hyperparams.dynamic_range_step_p {
                    let probe_n = hyperparams
                        .dynamic_range_probe_n
                        .max(2)
                        .min(std_equiv_a.len());
                    let sigma_thresh = hyperparams.dynamic_range_sigma_threshold;
                    let w_dr = hyperparams.dynamic_range_floor_weight;

                    // Sample probe_n distinct indices (with replacement is
                    // fine — the σ estimator is approximately the same).
                    let mut probe_idx: Vec<usize> = Vec::with_capacity(probe_n);
                    let mut probe_y: Vec<f64> = Vec::with_capacity(probe_n);
                    // Cache forward residuals so we can backprop after
                    // computing the σ across all forwards. V4 adds a
                    // `dy_dpre` slot to chain-rule through the tanh pin.
                    type ProbeFwd = (
                        Vec<f64>, // ha_pre
                        Vec<f64>, // ha
                        [f64; 4], // sa (reducer std-pool state)
                        usize,    // max_a (argmax index for max-pool)
                        f64,      // ya_rank
                        f64,      // ya_pool
                        f64,      // alpha
                        f64,      // dy_score/dy_pre (V4 tanh pin chain rule)
                    );
                    let mut probe_fwd: Vec<ProbeFwd> = Vec::with_capacity(probe_n);
                    for _ in 0..probe_n {
                        let u = rng.next_f64_unit();
                        let pi =
                            ((u * std_equiv_a.len() as f64) as usize).min(std_equiv_a.len() - 1);
                        probe_idx.push(pi);
                        let xa = std_equiv_a[pi].as_slice();
                        let (y_pre, y_rank, y_pool, alpha, _, ha_pre, ha, sa, max_a) =
                            psah::forward_per_sample_alpha_head(
                                xa, &w1, &b1, &rank_w, rank_b, &reducer_w, reducer_b, &w_alpha,
                                b_alpha, n_features, n_hidden, leaky,
                            );
                        let (y, dy_dpre) = pin_forward(y_pre);
                        probe_y.push(y);
                        probe_fwd.push((ha_pre, ha, sa, max_a, y_rank, y_pool, alpha, dy_dpre));
                    }

                    let n_p = probe_n as f64;
                    let mu: f64 = probe_y.iter().sum::<f64>() / n_p;
                    let var: f64 = probe_y.iter().map(|y| (y - mu).powi(2)).sum::<f64>() / n_p;
                    let sigma_obs = var.sqrt();
                    let viol = sigma_thresh - sigma_obs;
                    if viol > 0.0 && sigma_obs > 1e-9 {
                        let loss = w_dr * viol * viol;
                        total_loss += loss;
                        n_steps += 1;

                        // dL/dσ_obs = -2 · w · viol
                        // dσ/dy_i = (y_i − μ) / (σ · N)
                        // → dL/dy_i = -2 · w · viol · (y_i − μ) / (σ · N)
                        let grad_scale = -2.0 * w_dr * viol / (sigma_obs * n_p);

                        let mut g_rank_w_buf = vec![0.0f64; n_hidden_final];
                        let mut g_rank_b_buf = 0.0f64;
                        let mut g_red_w: [f64; 4] = [0.0; 4];
                        let mut g_red_b: f64 = 0.0;
                        let mut g_w_alpha_buf = vec![0.0f64; n_hidden_final];
                        let mut g_b_alpha: f64 = 0.0;

                        for i in 0..probe_n {
                            let pi = probe_idx[i];
                            let xa = std_equiv_a[pi].as_slice();
                            let y_i = probe_y[i];
                            let (ha_pre, ha, sa, max_a, y_rank, y_pool, alpha, dy_dpre) = (
                                &probe_fwd[i].0,
                                &probe_fwd[i].1,
                                &probe_fwd[i].2,
                                probe_fwd[i].3,
                                probe_fwd[i].4,
                                probe_fwd[i].5,
                                probe_fwd[i].6,
                                probe_fwd[i].7,
                            );
                            // dL/dy_score = grad_scale · (y_i_score − μ_score)
                            // dL/dy_pre = dL/dy_score · dy_score/dy_pre
                            let dl_dy = grad_scale * (y_i - mu) * dy_dpre;

                            psah::backprop_step_per_sample_alpha_head(
                                xa,
                                ha_pre,
                                ha,
                                sa,
                                max_a,
                                y_rank,
                                y_pool,
                                alpha,
                                dl_dy,
                                &rank_w,
                                &reducer_w,
                                &w_alpha,
                                &mut adam.gw1,
                                &mut adam.gb1,
                                &mut g_rank_w_buf,
                                &mut g_rank_b_buf,
                                &mut g_red_w,
                                &mut g_red_b,
                                &mut g_w_alpha_buf,
                                &mut g_b_alpha,
                                n_features,
                                n_hidden,
                                leaky,
                            );
                        }

                        if hyperparams.l2_lambda > 0.0 {
                            let l2 = hyperparams.l2_lambda;
                            for (g, &w) in adam.gw1.iter_mut().zip(w1.iter()) {
                                *g += l2 * w;
                            }
                            for j in 0..n_hidden {
                                g_rank_w_buf[j] += l2 * rank_w[j];
                                g_w_alpha_buf[j] += l2 * w_alpha[j];
                            }
                            for kk in 0..4 {
                                g_red_w[kk] += l2 * reducer_w[kk];
                            }
                        }

                        for j in 0..n_hidden {
                            adam.gw2[j] += g_rank_w_buf[j];
                        }
                        for kk in 0..4 {
                            adam.gw2[n_hidden + kk] += g_red_w[kk];
                        }
                        for j in 0..n_hidden {
                            adam.gw2[n_hidden + 4 + j] += g_w_alpha_buf[j];
                        }
                        adam.gw2[n_hidden + 4 + n_hidden] += g_b_alpha;
                        adam.gb2[0] += g_rank_b_buf;
                        adam.gb2[1] += g_red_b;

                        do_adam_step(
                            &mut adam,
                            &mut w1,
                            &mut b1,
                            &mut w2_enc,
                            &mut b2_enc,
                            &mut w_skip,
                            &mut b_skip,
                            &mut rank_w,
                            &mut rank_b,
                            &mut reducer_w,
                            &mut reducer_b,
                            &mut w_alpha,
                            &mut b_alpha,
                            lr,
                            n_hidden_final,
                        );
                    }
                }
            }
        }

        // Final-flush leftover K>1 (RankNet path).
        if k > 1 && !nin_on && steps_since_adam > 0 {
            do_adam_step(
                &mut adam,
                &mut w1,
                &mut b1,
                &mut w2_enc,
                &mut b2_enc,
                &mut w_skip,
                &mut b_skip,
                &mut rank_w,
                &mut rank_b,
                &mut reducer_w,
                &mut reducer_b,
                &mut w_alpha,
                &mut b_alpha,
                lr,
                n_hidden_final,
            );
        }
        if nin_on && !nin_buffer.is_empty() {
            let surviving = nin_buffer.iter().filter(|p| p.is_some()).count();
            if surviving >= 16 {
                flush_per_sample_alpha_nin_batch(
                    &mut nin_buffer,
                    &mut w1,
                    &mut b1,
                    &mut w2_enc,
                    &mut b2_enc,
                    &mut w_skip,
                    &mut b_skip,
                    &mut rank_w,
                    &mut rank_b,
                    &mut reducer_w,
                    &mut reducer_b,
                    &mut w_alpha,
                    &mut b_alpha,
                    &mut adam,
                    n_features,
                    n_hidden_final,
                    leaky,
                    hyperparams.l2_lambda,
                    hyperparams.norm_in_norm_weight,
                    hyperparams.norm_in_norm_p,
                    hyperparams.norm_in_norm_q,
                    lr,
                    &mut total_loss,
                    &mut n_steps,
                    &do_adam_step,
                );
            } else {
                nin_buffer.clear();
            }
        }

        let avg_loss = if n_steps > 0 {
            total_loss / n_steps as f64
        } else {
            0.0
        };

        // STRATEGY: fold live weights into the per-epoch EMA (before the
        // validation gate so the gate always sees an initialized EMA).
        if ema_active {
            let reducer_slice: Vec<f64> = reducer_w.to_vec();
            ema.update(
                &[
                    &w1,
                    &b1,
                    &w2_enc,
                    &b2_enc,
                    &w_skip,
                    &rank_w,
                    &w_alpha,
                    &reducer_slice,
                ],
                &[b_skip, rank_b, reducer_b, b_alpha],
            );
        }
        // STRATEGY: GroupDRO — rebuild the group-sampling CDF from decayed
        // per-group mean losses (multiplicative-weights emphasis on the
        // currently-worst group). η=0 keeps this branch off entirely.
        if dro_active {
            let means: Vec<f64> = dro_loss_sum
                .iter()
                .zip(&dro_loss_n)
                .map(|(&sm, &nn)| if nn > 0 { sm / nn as f64 } else { 0.0 })
                .collect();
            let base: Vec<f64> = train_indices
                .iter()
                .map(|&gi| groups[gi].train_weight)
                .collect();
            let w = strategy::dro_reweight(&base, &means, hyperparams.dro_eta);
            let mut cum = 0.0;
            cdf = w
                .iter()
                .map(|&x| {
                    cum += x;
                    cum
                })
                .collect();
            for v in &mut dro_loss_sum {
                *v *= 0.5;
            }
            for v in &mut dro_loss_n {
                *v /= 2;
            }
        }
        if epoch % hyperparams.log_every == 0 || epoch == hyperparams.n_epochs - 1 {
            // STRATEGY EMA: validate AND snapshot on the EMA weights so the
            // shipped bake equals the validated net; swapped back at every
            // exit of this block.
            if ema_active {
                ema_swap_all!();
            }
            // Per-sample α diagnostic: compute α at a sample of inputs
            // to log mean/min/max α at this epoch.
            let mut alpha_samples: Vec<f64> = Vec::new();
            for (gi, gfeats) in std_features.iter().enumerate() {
                let n = groups[gi].features.len();
                if n == 0 {
                    continue;
                }
                let step = (n / 64).max(1);
                let mut i = 0;
                while i < n && alpha_samples.len() < 512 {
                    let xi = &gfeats[i * n_features..(i + 1) * n_features];
                    let af_tmp = arch_forward(
                        xi,
                        &w1,
                        &b1,
                        &w2_enc,
                        &b2_enc,
                        &w_skip,
                        b_skip,
                        &rank_w,
                        rank_b,
                        &reducer_w,
                        reducer_b,
                        &w_alpha,
                        b_alpha,
                        n_features,
                        n_hidden,
                        n_hidden_final,
                        leaky,
                        use_2layer,
                        use_skip,
                    );
                    alpha_samples.push(af_tmp.alpha);
                    i += step;
                }
            }
            let alpha_mean = alpha_samples.iter().sum::<f64>() / alpha_samples.len().max(1) as f64;
            let alpha_min = alpha_samples.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let alpha_max = alpha_samples
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            let agg_mode = hyperparams.val_aggregate;
            let mse_only = hyperparams.mse_weight > 0.0 && hyperparams.ranknet_weight <= 0.0;
            let group_panels: Vec<crate::panel::LightPanel> = groups
                .iter()
                .enumerate()
                .map(|(gi, g)| {
                    // Sampled eval buffers when group_eval_cap is active and
                    // this group is oversized (empty vec = use the full group).
                    let (feat_buf, n_rows, humans): (&[f64], usize, &[f64]) =
                        if eval_cap > 0 && !eval_features[gi].is_empty() {
                            (
                                eval_features[gi].as_slice(),
                                eval_humans[gi].len(),
                                eval_humans[gi].as_slice(),
                            )
                        } else {
                            (
                                std_features[gi].as_slice(),
                                g.features.len(),
                                g.human_scores,
                            )
                        };
                    let preds = predict_group_per_sample_alpha_head(
                        feat_buf,
                        n_rows,
                        n_features,
                        &w1,
                        &b1,
                        &w2_enc,
                        &b2_enc,
                        &w_skip,
                        b_skip,
                        &rank_w,
                        rank_b,
                        &reducer_w,
                        reducer_b,
                        &w_alpha,
                        b_alpha,
                        n_hidden,
                        n_hidden_final,
                        leaky,
                        use_2layer,
                        use_skip,
                    );
                    let signed_preds: Vec<f64> = if mse_only {
                        preds.clone()
                    } else {
                        preds.iter().map(|&p| -p).collect()
                    };
                    crate::panel::compute_light_panel_subsampled(&signed_preds, humans)
                })
                .collect();

            let group_scores: Vec<f64> =
                group_panels.iter().map(|p| p.aggregate(agg_mode)).collect();

            // Anchor-based goal inputs: compute mean prediction on anchor
            // rows (for G2 JND check) and anchor Z-RMSE (for G8).
            let (anchor_mean_pred, anchor_zrmse) =
                if anchor_active && !std_anchor_features.is_empty() {
                    let anchor_feats = &std_anchor_features;
                    let mut preds = Vec::with_capacity(anchor_feats.len());
                    for af in anchor_feats {
                        let af_tmp = arch_forward(
                            af,
                            &w1,
                            &b1,
                            &w2_enc,
                            &b2_enc,
                            &w_skip,
                            b_skip,
                            &rank_w,
                            rank_b,
                            &reducer_w,
                            reducer_b,
                            &w_alpha,
                            b_alpha,
                            n_features,
                            n_hidden,
                            n_hidden_final,
                            leaky,
                            use_2layer,
                            use_skip,
                        );
                        let (pinned, _) = pin_forward(af_tmp.y);
                        preds.push(pinned);
                    }
                    let mean_p = preds.iter().sum::<f64>() / preds.len().max(1) as f64;
                    let zrmse = if let Some(a) = anchor {
                        if a.row_weights.len() == preds.len() {
                            crate::panel::z_rmse(&preds, a.row_weights)
                        } else {
                            f64::NAN
                        }
                    } else {
                        f64::NAN
                    };
                    (
                        Some(mean_p),
                        if zrmse.is_finite() { Some(zrmse) } else { None },
                    )
                } else {
                    (None, None)
                };

            let val_score = match hyperparams.validation_policy {
                ValidationPolicy::Goals => {
                    let gs =
                        compute_goal_scores(&group_panels, groups, anchor_mean_pred, anchor_zrmse);
                    gs.aggregate()
                }
                _ => {
                    if val_indices.is_empty() {
                        group_scores.iter().sum::<f64>() / group_scores.len() as f64
                    } else {
                        match hyperparams.validation_policy {
                            ValidationPolicy::Mean => {
                                let total: f64 = val_indices
                                    .iter()
                                    .map(|&i| groups[i].validation_weight)
                                    .sum();
                                val_indices
                                    .iter()
                                    .map(|&i| group_scores[i] * groups[i].validation_weight)
                                    .sum::<f64>()
                                    / total
                            }
                            ValidationPolicy::Min => val_indices
                                .iter()
                                .map(|&i| group_scores[i])
                                .fold(f64::INFINITY, f64::min),
                            ValidationPolicy::Goals => unreachable!(),
                        }
                    }
                }
            };

            let elapsed = start.elapsed().as_secs_f64();
            let per_group = group_panels
                .iter()
                .zip(groups.iter())
                .map(|(p, g)| {
                    format!(
                        "{}: srocc={:.4} plcc={:.4} pwrc={:.4}",
                        g.name, p.srocc, p.plcc, p.pwrc
                    )
                })
                .collect::<Vec<_>>()
                .join(" | ");
            let goal_info = if matches!(hyperparams.validation_policy, ValidationPolicy::Goals) {
                let gs = compute_goal_scores(&group_panels, groups, anchor_mean_pred, anchor_zrmse);
                format!(
                    " | goals: g2={:.2} g5={:.2} g6={:.2} g7={:.2} g8={:.2}",
                    gs.g2_jnd_anchor,
                    gs.g5_hf_rank,
                    gs.g6_mf_band_coverage,
                    gs.g7_cid22_rank,
                    gs.g8_zrmse
                )
            } else {
                String::new()
            };
            log_line(
                &format!(
                    "  epoch {epoch:>3} | lr={lr:.5} | loss={avg_loss:.4} | val({agg_mode})={val_score:.4} (best={best_val_score:.4}){goal_info} | α(x): μ={alpha_mean:.3} [min={alpha_min:.3}, max={alpha_max:.3}] | reducer_w=[μ={:.3},σ={:.3},max={:.3},p6={:.3}] | {per_group} | t={elapsed:.1}s",
                    reducer_w[0], reducer_w[1], reducer_w[2], reducer_w[3],
                ),
                log,
            );

            if val_score > best_val_score {
                best_val_score = val_score;
                stale_epochs = 0;
                if use_2layer {
                    // monotone_cbc: hard-project the weights to exact signs
                    // before baking so the saved best-epoch bake is monotone-
                    // by-construction (no-op when monotone_cbc=false).
                    let mut bake_w1 = proj_w1_masked(&w1);
                    let mut bake_w2_enc = proj_geq0(&w2_enc);
                    let mut bake_rank_w = proj_leq0(&rank_w);
                    let mut bake_w_alpha = proj_w_alpha_zero(&w_alpha);
                    let bake_b_alpha = proj_b_alpha_one(b_alpha);
                    if qat_fine_tune_epochs > 0 {
                        // QAT: ship the f16+zerobias weights the forward was
                        // trained against (quantize AFTER the hard sign
                        // projection). With --out-dtype f16 + compression the
                        // bake is small AND == the validated net.
                        bake_w1 = qat_quantize_copy(&bake_w1, qat_tau);
                        bake_w2_enc = qat_quantize_copy(&bake_w2_enc, qat_tau);
                        bake_rank_w = qat_quantize_copy(&bake_rank_w, qat_tau);
                        bake_w_alpha = qat_quantize_copy(&bake_w_alpha, qat_tau);
                    }
                    let model = psah::PerSampleAlphaHeadModel {
                        scaler_mean: scaler_mean.clone(),
                        scaler_scale: scaler_scale.clone(),
                        w1: bake_w1,
                        b1: b1.clone(),
                        rank_w: bake_rank_w,
                        rank_b,
                        reducer_w,
                        reducer_b,
                        w_alpha: bake_w_alpha,
                        b_alpha: bake_b_alpha,
                        n_hidden: n_hidden_final,
                        n_features,
                        leaky_alpha: hyperparams.leaky_alpha,
                    };
                    best_bake = Some(psah::bake_per_sample_alpha_head_v3_2layer(
                        &model,
                        &bake_w2_enc,
                        &b2_enc,
                        n_hidden,
                        n_hidden_final,
                        if tanh_pin_active {
                            Some(tanh_scale)
                        } else {
                            None
                        },
                        hyperparams.feature_transforms.as_deref(),
                        hyperparams.feature_transform_params.as_deref(),
                        None, // spline added post-training
                        hyperparams.out_dtype,
                    ));
                } else if use_skip {
                    // Skip-only: the bake format is standard 1-layer; skip
                    // weights could be stored as metadata but runtime doesn't
                    // consume them yet. Use standard bake for now.
                    best_bake = Some(vec![0u8; 4]); // TODO: skip bake metadata
                } else {
                    // monotone_cbc: hard-project the weights (single-layer path).
                    let bake_w1 = proj_w1_masked(&w1);
                    let bake_rank_w = proj_leq0(&rank_w);
                    let bake_w_alpha = proj_w_alpha_zero(&w_alpha);
                    let bake_b_alpha = proj_b_alpha_one(b_alpha);
                    let model = psah::PerSampleAlphaHeadModel {
                        scaler_mean: scaler_mean.clone(),
                        scaler_scale: scaler_scale.clone(),
                        w1: bake_w1,
                        b1: b1.clone(),
                        rank_w: bake_rank_w,
                        rank_b,
                        reducer_w,
                        reducer_b,
                        w_alpha: bake_w_alpha,
                        b_alpha: bake_b_alpha,
                        n_hidden: n_hidden_final,
                        n_features,
                        leaky_alpha: hyperparams.leaky_alpha,
                    };
                    best_bake = Some(if tanh_pin_active {
                        psah::bake_per_sample_alpha_head_v3_with_tanh_and_transforms(
                            &model,
                            tanh_scale,
                            hyperparams.feature_transforms.as_deref(),
                            hyperparams.feature_transform_params.as_deref(),
                            None, // spline added post-training
                        )
                    } else {
                        psah::bake_per_sample_alpha_head_v3(&model)
                    });
                } // end else !use_2layer && !use_skip
            } else {
                stale_epochs += hyperparams.log_every;
                if hyperparams.early_stop_patience > 0
                    && stale_epochs >= hyperparams.early_stop_patience
                {
                    log_line(
                        &format!(
                            "  early stop at epoch {epoch} (no val improvement for {stale_epochs} epochs)"
                        ),
                        log,
                    );
                    if ema_active {
                        ema_swap_all!();
                    }
                    break;
                }
            }
            // H-TRAJ checkpoint dump (balance campaign 2026-08-28). TWIN OF
            // the L0 best-val snapshot above — keep the model build in
            // lockstep; the dump is DEFINED as "what the best-val snapshot
            // would save at this epoch". Runs while EMA weights are swapped
            // in (before the swap-back below), spline-less like the
            // snapshot: the pack step fits the spline.
            if hyperparams.dump_checkpoints_every > 0
                && epoch % hyperparams.dump_checkpoints_every == 0
            {
                assert!(
                    !use_2layer && !use_skip,
                    "--dump-checkpoints-every: 2-layer/skip not wired (plain-MLP and 0/1-hidden per-sample-alpha lanes are)"
                );
                let bake_w1 = proj_w1_masked(&w1);
                let bake_rank_w = proj_leq0(&rank_w);
                let bake_w_alpha = proj_w_alpha_zero(&w_alpha);
                let bake_b_alpha = proj_b_alpha_one(b_alpha);
                let model = psah::PerSampleAlphaHeadModel {
                    scaler_mean: scaler_mean.clone(),
                    scaler_scale: scaler_scale.clone(),
                    w1: bake_w1,
                    b1: b1.clone(),
                    rank_w: bake_rank_w,
                    rank_b,
                    reducer_w,
                    reducer_b,
                    w_alpha: bake_w_alpha,
                    b_alpha: bake_b_alpha,
                    n_hidden: n_hidden_final,
                    n_features,
                    leaky_alpha: hyperparams.leaky_alpha,
                };
                let ckpt_bytes = if tanh_pin_active {
                    psah::bake_per_sample_alpha_head_v3_with_tanh_and_transforms(
                        &model,
                        tanh_scale,
                        hyperparams.feature_transforms.as_deref(),
                        hyperparams.feature_transform_params.as_deref(),
                        None, // spline added at pack time
                    )
                } else {
                    psah::bake_per_sample_alpha_head_v3(&model)
                };
                let dir = hyperparams
                    .dump_checkpoints_dir
                    .clone()
                    .unwrap_or_else(|| std::path::PathBuf::from("."));
                let ckpt_path = dir.join(format!("ckpt_epoch{epoch:03}.bin"));
                std::fs::write(&ckpt_path, &ckpt_bytes)
                    .expect("H-TRAJ checkpoint dump write failed");
                log_line(
                    &format!(
                        "  checkpoint dump: {} ({} B)",
                        ckpt_path.display(),
                        ckpt_bytes.len()
                    ),
                    log,
                );
            }
            if ema_active {
                ema_swap_all!();
            }
        }
    }
    record_best_val(best_val_score);

    log_line(
        &format!(
            "MLP train (PER-SAMPLE-α): best val SROCC = {best_val_score:.4} | final reducer_w=[μ={:.3},σ={:.3},max={:.3},p6={:.3}] b_α={:.3}",
            reducer_w[0], reducer_w[1], reducer_w[2], reducer_w[3], b_alpha,
        ),
        log,
    );
    // Faithfulness hook: `sampling::simulate` must reproduce this hash.
    if let Some(d) = sample_digest.as_ref() {
        println!("ZENSIM_SAMPLE_DIGEST {}", d.hex());
    }
    let bake_bytes = best_bake.unwrap_or_else(|| {
        let model = psah::PerSampleAlphaHeadModel {
            scaler_mean: scaler_mean.clone(),
            scaler_scale: scaler_scale.clone(),
            w1: w1.clone(),
            b1: b1.clone(),
            rank_w: rank_w.clone(),
            rank_b,
            reducer_w,
            reducer_b,
            w_alpha: w_alpha.clone(),
            b_alpha,
            n_hidden,
            n_features,
            leaky_alpha: hyperparams.leaky_alpha,
        };
        if tanh_pin_active {
            psah::bake_per_sample_alpha_head_v3_with_tanh_and_transforms(
                &model,
                tanh_scale,
                hyperparams.feature_transforms.as_deref(),
                hyperparams.feature_transform_params.as_deref(),
                None, // spline added post-training
            )
        } else {
            psah::bake_per_sample_alpha_head_v3(&model)
        }
    });

    // Post-training output calibration spline (V9-style, matching
    // scripts/v_next/calibrate_v9_spline.py). When anchor data is
    // available, forward the anchor rows through the best weights,
    // group by target_score, take median prediction per band, build
    // strictly-increasing PCHIP knots. The spline payload is saved
    // as a sidecar file via ZENSIM_SPLINE_SIDECAR env var.
    //
    // The actual injection into the bake requires zenpredict-bake's
    // JSON pipeline (the existing Python script calls `zenpredict bake`
    // on a modified JSON). The sidecar approach lets the caller script
    // handle the injection.
    if anchor_active
        && !std_anchor_features.is_empty()
        && let Some(a) = anchor
    {
        // Forward the anchor through the net the bake will SHIP — when
        // QAT is on, that's the f16+zerobias-quantized net, so the dial
        // spline calibrates exactly what ships (quantize-then-calibrate
        // in ONE pass; no post-hoc identity/dial surprise). Then fit a
        // quantile-binned monotone PCHIP. This REPLACES the old
        // band-by-target + 0.01-pred-filter, which collapsed to 2
        // degenerate knots on the narrow (~0.4-wide) tanh-pin band and
        // shipped a broken (all-negative / identity=0) dial.
        // Forward the SHIPPED net EXACTLY: the bake applies the hard sign
        // projection (encoder ≥0, rank_w ≤0, α≡1) THEN — when QAT —
        // f16+zerobias quantize. Forwarding the UN-projected net here
        // inverts the pred↔target correlation (the projection flips
        // signs), so the spline gets fit on a different net than ships
        // and its direction comes out wrong. Project (+ quantize) so the
        // dial calibrates the actual shipped net.
        let pw1 = proj_w1_masked(&w1);
        let pw2 = proj_geq0(&w2_enc);
        let prw = proj_leq0(&rank_w);
        let pwa = proj_w_alpha_zero(&w_alpha);
        let sba = proj_b_alpha_one(b_alpha);
        let (sw1, sw2, srw, swa) = if qat_fine_tune_epochs > 0 {
            (
                qat_quantize_copy(&pw1, qat_tau),
                qat_quantize_copy(&pw2, qat_tau),
                qat_quantize_copy(&prw, qat_tau),
                qat_quantize_copy(&pwa, qat_tau),
            )
        } else {
            (pw1, pw2, prw, pwa)
        };
        let mut sp_preds: Vec<f64> = Vec::with_capacity(std_anchor_features.len());
        let mut sp_targets: Vec<f64> = Vec::with_capacity(std_anchor_features.len());
        for (i, af) in std_anchor_features.iter().enumerate() {
            let fwd = arch_forward(
                af,
                &sw1,
                &b1,
                &sw2,
                &b2_enc,
                &w_skip,
                b_skip,
                &srw,
                rank_b,
                &reducer_w,
                reducer_b,
                &swa,
                sba,
                n_features,
                n_hidden,
                n_hidden_final,
                leaky,
                use_2layer,
                use_skip,
            );
            let (pinned, _) = pin_forward(fwd.y);
            let target = a
                .target_scores
                .and_then(|ts| ts.get(i).copied())
                .unwrap_or(hyperparams.anchor_target_score);
            sp_preds.push(pinned);
            sp_targets.push(target);
        }

        // Diagnostic: log the SHIPPED net's anchor-output range BEFORE
        // attempting spline fit. When the spline fit fails (too few
        // distinct knots, non-monotone bin medians, etc.) without this
        // log we silently ship a bake with no dial calibration — issue
        // #40 (hidden=1 case) revealed this gap.
        let (sp_min, sp_max) = sp_preds
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &v| {
                (mn.min(v), mx.max(v))
            });
        let (st_min, st_max) = sp_targets
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &v| {
                (mn.min(v), mx.max(v))
            });
        log_line(
            &format!(
                "output calibration spline: anchor n={} pred [{:.4}, {:.4}] target [{:.4}, {:.4}]",
                sp_preds.len(),
                sp_min,
                sp_max,
                st_min,
                st_max
            ),
            log,
        );

        let spline_attempt =
            crate::output_calibration_spline::fit_monotone_spline(&sp_preds, &sp_targets, 18);
        if spline_attempt.is_none() {
            log_line(
                "output calibration spline: fit_monotone_spline RETURNED None — bake will ship WITHOUT a dial spline (pred→score on raw net output). Check anchor pred range + monotonicity vs target.",
                log,
            );
        }
        if let Some(payload) = spline_attempt {
            log_line(
                &format!(
                    "output calibration spline: fit_monotone_spline on {} anchor rows (qat={})",
                    sp_preds.len(),
                    qat_fine_tune_epochs > 0
                ),
                log,
            );

            // Re-bake the model with the spline included.
            // monotone_cbc: project here too — the final spline-augmented
            // bake must be monotone-by-construction (spline is monotone,
            // so the network must be).
            let mut bake_w1 = proj_w1_masked(&w1);
            let mut bake_w2_enc = proj_geq0(&w2_enc);
            let mut bake_rank_w = proj_leq0(&rank_w);
            let mut bake_w_alpha = proj_w_alpha_zero(&w_alpha);
            let bake_b_alpha = proj_b_alpha_one(b_alpha);
            if qat_fine_tune_epochs > 0 {
                // QAT: ship the f16+zerobias weights the forward + the
                // spline-fit above were calibrated against (quantize
                // AFTER the hard sign projection).
                bake_w1 = qat_quantize_copy(&bake_w1, qat_tau);
                bake_w2_enc = qat_quantize_copy(&bake_w2_enc, qat_tau);
                bake_rank_w = qat_quantize_copy(&bake_rank_w, qat_tau);
                bake_w_alpha = qat_quantize_copy(&bake_w_alpha, qat_tau);
            }
            let model = psah::PerSampleAlphaHeadModel {
                scaler_mean: scaler_mean.clone(),
                scaler_scale: scaler_scale.clone(),
                w1: bake_w1,
                b1: b1.clone(),
                rank_w: bake_rank_w,
                rank_b,
                reducer_w,
                reducer_b,
                w_alpha: bake_w_alpha,
                b_alpha: bake_b_alpha,
                n_hidden: n_hidden_final,
                n_features,
                leaky_alpha: hyperparams.leaky_alpha,
            };
            let rebaked = if use_2layer {
                psah::bake_per_sample_alpha_head_v3_2layer(
                    &model,
                    &bake_w2_enc,
                    &b2_enc,
                    n_hidden,
                    n_hidden_final,
                    if tanh_pin_active {
                        Some(tanh_scale)
                    } else {
                        None
                    },
                    hyperparams.feature_transforms.as_deref(),
                    hyperparams.feature_transform_params.as_deref(),
                    Some(&payload),
                    hyperparams.out_dtype,
                )
            } else if tanh_pin_active {
                psah::bake_per_sample_alpha_head_v3_with_tanh_and_transforms(
                    &model,
                    tanh_scale,
                    hyperparams.feature_transforms.as_deref(),
                    hyperparams.feature_transform_params.as_deref(),
                    Some(&payload),
                )
            } else {
                // No spline for non-tanh bakes (V0_1/V0_2 path)
                return bake_bytes;
            };
            log_line(
                &format!(
                    "re-baked with output calibration spline ({} bytes)",
                    rebaked.len()
                ),
                log,
            );
            return rebaked;
        }
    }

    bake_bytes
}

/// Predict per-sample α head outputs for every row in a flat
/// (n_pairs × n_features) standardized feature buffer.
///
/// Each forward is independent of every other (no batch state), so the
/// `n_pairs` rows are dispatched through `into_par_iter`. On the 16-core
/// 7950X this drops per-epoch validation-prediction time from ~2.3 s to
/// ~0.2 s when scoring the full canonical training corpus (247 k rows).
///
/// **Determinism**: row outputs are independent and order-preserving
/// (`par_iter`/`collect` preserves source order per rayon docs), so the
/// returned `Vec<f64>` is bit-identical to the sequential `.map(...)`
/// reference. The downstream `spearman_correlation` consumes the vector
/// in order, so the validation SROCC is unaffected by thread count.
#[allow(clippy::too_many_arguments)]
fn predict_group_per_sample_alpha_head(
    std_x: &[f64],
    n_pairs: usize,
    n_features: usize,
    w1: &[f64],
    b1: &[f64],
    w2_enc: &[f64],
    b2_enc: &[f64],
    w_skip: &[f64],
    b_skip: f64,
    rank_w: &[f64],
    rank_b: f64,
    reducer_w: &[f64; 4],
    reducer_b: f64,
    w_alpha: &[f64],
    b_alpha: f64,
    n_hidden1: usize,
    n_hidden_final: usize,
    leaky: f64,
    use_2layer: bool,
    use_skip: bool,
) -> Vec<f64> {
    (0..n_pairs)
        .into_par_iter()
        .map(|i| {
            let xi = &std_x[i * n_features..(i + 1) * n_features];
            let af_tmp = arch_forward(
                xi,
                w1,
                b1,
                w2_enc,
                b2_enc,
                w_skip,
                b_skip,
                rank_w,
                rank_b,
                reducer_w,
                reducer_b,
                w_alpha,
                b_alpha,
                n_features,
                n_hidden1,
                n_hidden_final,
                leaky,
                use_2layer,
                use_skip,
            );
            af_tmp.y
        })
        .collect()
}

/// Forward result from `arch_forward`. Carries all intermediates
/// needed for exact backward — including optional layer-1 intermediates
/// for the 2-layer encoder.
mod arch;
mod arch_f32;
pub use arch::{ArchForward, arch_backward, arch_forward};

/// STRATEGY-2026-07-02: shared backward tail for the listwise / triplet step
/// types — mirrors the plain-RankNet pair tail verbatim (grad buffers → one
/// `arch_backward` per row → L2 → fold into Adam slots → cadence step). One
/// call = one logical step toward the Adam cadence regardless of row count.
#[allow(clippy::too_many_arguments)]
fn strategy_backward_rows<F>(
    xs: &[&[f64]],
    fwds: &[ArchForward],
    dl_dys: &[f64],
    w1: &mut Vec<f64>,
    b1: &mut Vec<f64>,
    w2_enc: &mut Vec<f64>,
    b2_enc: &mut Vec<f64>,
    w_skip: &mut Vec<f64>,
    b_skip: &mut f64,
    rank_w: &mut Vec<f64>,
    rank_b: &mut f64,
    reducer_w: &mut [f64; 4],
    reducer_b: &mut f64,
    w_alpha: &mut Vec<f64>,
    b_alpha: &mut f64,
    adam: &mut AdamState,
    n_features: usize,
    n_hidden: usize,
    n_hidden_final: usize,
    leaky: f64,
    use_2layer: bool,
    use_skip: bool,
    l2_lambda: f64,
    lr: f64,
    k: usize,
    steps_since_adam: &mut u64,
    do_adam_step: &F,
) where
    F: Fn(
        &mut AdamState,
        &mut Vec<f64>,
        &mut Vec<f64>,
        &mut Vec<f64>,
        &mut Vec<f64>,
        &mut Vec<f64>,
        &mut f64,
        &mut Vec<f64>,
        &mut f64,
        &mut [f64; 4],
        &mut f64,
        &mut Vec<f64>,
        &mut f64,
        f64,
        usize,
    ),
{
    let mut g_rank_w_buf = vec![0.0f64; n_hidden_final];
    let mut g_rank_b_buf = 0.0f64;
    let mut g_red_w: [f64; 4] = [0.0; 4];
    let mut g_red_b: f64 = 0.0;
    let mut g_w_alpha_buf = vec![0.0f64; n_hidden_final];
    let mut g_b_alpha: f64 = 0.0;
    for ((x, fwd), &dl) in xs.iter().zip(fwds).zip(dl_dys) {
        arch_backward(
            x,
            fwd,
            dl,
            w1,
            w2_enc,
            rank_w,
            reducer_w,
            w_alpha,
            &mut adam.gw1,
            &mut adam.gb1,
            &mut g_rank_w_buf,
            &mut g_rank_b_buf,
            &mut g_red_w,
            &mut g_red_b,
            &mut g_w_alpha_buf,
            &mut g_b_alpha,
            n_features,
            n_hidden,
            n_hidden_final,
            leaky,
            use_2layer,
            use_skip,
        );
    }
    if l2_lambda > 0.0 {
        for (g, &w) in adam.gw1.iter_mut().zip(w1.iter()) {
            *g += l2_lambda * w;
        }
        for j in 0..n_hidden_final {
            g_rank_w_buf[j] += l2_lambda * rank_w[j];
            g_w_alpha_buf[j] += l2_lambda * w_alpha[j];
        }
        for kk in 0..4 {
            g_red_w[kk] += l2_lambda * reducer_w[kk];
        }
    }
    for j in 0..n_hidden_final {
        adam.gw2[j] += g_rank_w_buf[j];
    }
    for kk in 0..4 {
        adam.gw2[n_hidden_final + kk] += g_red_w[kk];
    }
    for j in 0..n_hidden_final {
        adam.gw2[n_hidden_final + 4 + j] += g_w_alpha_buf[j];
    }
    adam.gw2[n_hidden_final + 4 + n_hidden_final] += g_b_alpha;
    adam.gb2[0] += g_rank_b_buf;
    adam.gb2[1] += g_red_b;
    *steps_since_adam += 1;
    if k == 1 || *steps_since_adam >= k as u64 {
        do_adam_step(
            adam,
            w1,
            b1,
            w2_enc,
            b2_enc,
            w_skip,
            b_skip,
            rank_w,
            rank_b,
            reducer_w,
            reducer_b,
            w_alpha,
            b_alpha,
            lr,
            n_hidden_final,
        );
        *steps_since_adam = 0;
    }
}

/// NiN-aware flush for the per-sample α head. Computes NiN over the
/// 2N surviving predictions and routes per-prediction grad through
/// `backprop_step_per_sample_alpha_head`. Accumulates into the Adam
/// slots laid out in `train_mlp_per_sample_alpha_head`, then performs
/// one Adam step. L2 is applied K·λ·w scaled by `steps_added`
/// (matches hybrid_head flush).
#[allow(clippy::too_many_arguments)]
fn flush_per_sample_alpha_nin_batch<F>(
    nin_buffer: &mut Vec<Option<PerSampleAlphaPairForward<'_>>>,
    w1: &mut Vec<f64>,
    b1: &mut Vec<f64>,
    w2_enc: &mut Vec<f64>,
    b2_enc: &mut Vec<f64>,
    w_skip: &mut Vec<f64>,
    b_skip: &mut f64,
    rank_w: &mut Vec<f64>,
    rank_b: &mut f64,
    reducer_w: &mut [f64; 4],
    reducer_b: &mut f64,
    w_alpha: &mut Vec<f64>,
    b_alpha: &mut f64,
    adam: &mut AdamState,
    n_features: usize,
    n_hidden: usize,
    leaky_alpha: f64,
    l2_lambda: f64,
    nin_weight: f64,
    nin_p: f64,
    nin_q: f64,
    lr: f64,
    total_loss: &mut f64,
    n_steps: &mut u64,
    do_adam_step: &F,
) where
    F: Fn(
        &mut AdamState,
        &mut Vec<f64>,
        &mut Vec<f64>,
        &mut Vec<f64>,
        &mut Vec<f64>,
        &mut Vec<f64>,
        &mut f64,
        &mut Vec<f64>,
        &mut f64,
        &mut [f64; 4],
        &mut f64,
        &mut Vec<f64>,
        &mut f64,
        f64,
        usize,
    ),
{
    use zensim_train_core::per_sample_alpha_head as psah;

    let mut nin_preds: Vec<f64> = Vec::with_capacity(2 * nin_buffer.len());
    let mut nin_labels: Vec<f64> = Vec::with_capacity(2 * nin_buffer.len());
    let mut nin_idx_map: Vec<(usize, bool)> = Vec::with_capacity(2 * nin_buffer.len());
    for (pi, slot) in nin_buffer.iter().enumerate() {
        if let Some(p) = slot {
            nin_preds.push(p.ya);
            nin_labels.push(-p.mos_a);
            nin_idx_map.push((pi, false));
            nin_preds.push(p.yb);
            nin_labels.push(-p.mos_b);
            nin_idx_map.push((pi, true));
        }
    }
    let (nin_loss, nin_grad) = if nin_preds.len() >= 2 {
        loss_norm_in_norm::compute_norm_in_norm_loss_and_grad(&nin_preds, &nin_labels, nin_p, nin_q)
    } else {
        (0.0, vec![0.0; nin_preds.len()])
    };
    *total_loss += nin_weight * nin_loss;

    let mut steps_added: u64 = 0;
    let mut g_rank_w_buf = vec![0.0f64; n_hidden];
    let mut g_rank_b_buf = 0.0f64;
    let mut g_red_w: [f64; 4] = [0.0; 4];
    let mut g_red_b: f64 = 0.0;
    let mut g_w_alpha_buf = vec![0.0f64; n_hidden];
    let mut g_b_alpha: f64 = 0.0;

    for (nin_pos, &(pi, is_b)) in nin_idx_map.iter().enumerate() {
        let p = match &nin_buffer[pi] {
            Some(p) => p,
            None => continue,
        };
        let nin_g = nin_grad[nin_pos] * nin_weight;
        if is_b {
            let dl_dy = p.dl_dyb_rn + nin_g;
            psah::backprop_step_per_sample_alpha_head(
                p.xb,
                &p.hb_pre,
                &p.hb,
                &p.sb,
                p.max_b,
                p.yb_rank,
                p.yb_pool,
                p.alpha_b,
                dl_dy,
                rank_w,
                reducer_w,
                w_alpha,
                &mut adam.gw1,
                &mut adam.gb1,
                &mut g_rank_w_buf,
                &mut g_rank_b_buf,
                &mut g_red_w,
                &mut g_red_b,
                &mut g_w_alpha_buf,
                &mut g_b_alpha,
                n_features,
                n_hidden,
                leaky_alpha,
            );
            steps_added += 1;
        } else {
            let dl_dy = p.dl_dya_rn + nin_g;
            psah::backprop_step_per_sample_alpha_head(
                p.xa,
                &p.ha_pre,
                &p.ha,
                &p.sa,
                p.max_a,
                p.ya_rank,
                p.ya_pool,
                p.alpha_a,
                dl_dy,
                rank_w,
                reducer_w,
                w_alpha,
                &mut adam.gw1,
                &mut adam.gb1,
                &mut g_rank_w_buf,
                &mut g_rank_b_buf,
                &mut g_red_w,
                &mut g_red_b,
                &mut g_w_alpha_buf,
                &mut g_b_alpha,
                n_features,
                n_hidden,
                leaky_alpha,
            );
        }
    }

    if l2_lambda > 0.0 && steps_added > 0 {
        let scale = l2_lambda * steps_added as f64;
        for (g, &w) in adam.gw1.iter_mut().zip(w1.iter()) {
            *g += scale * w;
        }
        for j in 0..n_hidden {
            g_rank_w_buf[j] += scale * rank_w[j];
            g_w_alpha_buf[j] += scale * w_alpha[j];
        }
        for kk in 0..4 {
            g_red_w[kk] += scale * reducer_w[kk];
        }
    }

    for j in 0..n_hidden {
        adam.gw2[j] += g_rank_w_buf[j];
    }
    for kk in 0..4 {
        adam.gw2[n_hidden + kk] += g_red_w[kk];
    }
    for j in 0..n_hidden {
        adam.gw2[n_hidden + 4 + j] += g_w_alpha_buf[j];
    }
    adam.gw2[n_hidden + 4 + n_hidden] += g_b_alpha;
    adam.gb2[0] += g_rank_b_buf;
    adam.gb2[1] += g_red_b;

    if steps_added > 0 {
        do_adam_step(
            adam, w1, b1, w2_enc, b2_enc, w_skip, b_skip, rank_w, rank_b, reducer_w, reducer_b,
            w_alpha, b_alpha, lr, n_hidden,
        );
        *n_steps += steps_added;
    }
    nin_buffer.clear();
}

#[cfg(test)]
mod ref_bucket_tests {
    use super::RefBuckets;

    /// The invariant the whole feature rests on: a within-ref draw must
    /// never pair rows from two different reference images. If it does,
    /// the pair teaches between-image scale instead of the within-image
    /// distortion ladder — the confound that makes the near-lossless
    /// corpus read pooled +0.204 vs per-ref +0.916.
    #[test]
    fn draws_never_cross_a_ref() {
        // 3 refs, uneven row counts, interleaved rather than contiguous —
        // real parquets are not sorted by ref.
        let ref_ids = [0u32, 1, 0, 2, 1, 2, 0, 1, 2, 2];
        let rb = RefBuckets::build(&ref_ids).expect("all refs have >= 2 rows");
        assert_eq!(rb.n_refs(), 3);
        assert_eq!(rb.n_rows(), 10);

        // Sweep the draw space exhaustively enough to catch any leak.
        for u_ref in 0..64u64 {
            for u_a in 0..16u64 {
                for u_b in 0..16u64 {
                    let (ia, ib) = rb.draw(u_ref, u_a, u_b);
                    assert_eq!(
                        ref_ids[ia], ref_ids[ib],
                        "draw({u_ref},{u_a},{u_b}) crossed refs: row {ia} (ref {}) vs row {ib} (ref {})",
                        ref_ids[ia], ref_ids[ib]
                    );
                }
            }
        }
    }

    /// Refs with a single row can't yield a pair and must be dropped —
    /// otherwise a draw landing there returns `ia == ib` forever and
    /// silently wastes that share of the sampling budget.
    #[test]
    fn singleton_refs_are_dropped() {
        let ref_ids = [0u32, 0, 1, 2, 2]; // ref 1 is a singleton
        let rb = RefBuckets::build(&ref_ids).expect("refs 0 and 2 are usable");
        assert_eq!(rb.n_refs(), 2, "the singleton ref must not be drawable");
        assert_eq!(rb.n_rows(), 4, "row 2 is unreachable");
        for u in 0..64u64 {
            let (ia, ib) = rb.draw(u, u * 7 + 1, u * 13 + 3);
            assert_ne!(ref_ids[ia], 1, "singleton ref must never be drawn");
            assert_eq!(ref_ids[ia], ref_ids[ib]);
        }
    }

    /// A corpus with no repeated ref has no drawable pair at all. Build
    /// must report that (None) so the caller refuses to train rather than
    /// silently reverting to cross-image pairs.
    #[test]
    fn all_singletons_yields_none() {
        assert!(RefBuckets::build(&[0u32, 1, 2, 3]).is_none());
        assert!(RefBuckets::build(&[]).is_none());
    }

    /// The hard-pair miner re-draws row B; on a within-ref group that
    /// re-draw must stay inside row A's ref.
    #[test]
    fn redraw_partner_stays_in_ref() {
        let ref_ids = [0u32, 1, 0, 2, 1, 2, 0, 1, 2, 2];
        let rb = RefBuckets::build(&ref_ids).expect("usable");
        for (row, &r) in ref_ids.iter().enumerate() {
            for u in 0..64u64 {
                let p = rb.redraw_partner(row, u);
                assert_eq!(ref_ids[p], r, "redraw for row {row} left ref {r}");
            }
        }
    }

    /// A row whose ref was dropped has no partner; returning the row
    /// itself lets the caller's `ia == ib` skip absorb it.
    #[test]
    fn redraw_partner_of_dropped_row_is_identity() {
        let ref_ids = [0u32, 0, 1]; // row 2 is a singleton -> dropped
        let rb = RefBuckets::build(&ref_ids).expect("ref 0 is usable");
        assert_eq!(rb.redraw_partner(2, 12345), 2);
    }
}

#[cfg(test)]
mod l2_row_form_tests {
    use super::add_l2_grad_layer1;

    /// The reference: the divided-index form this helper replaced, verbatim.
    fn divided_index_form(
        g: &mut [f64],
        w: &[f64],
        scale: f64,
        n_hidden: usize,
        mult: Option<&[f64]>,
    ) {
        for (idx, (g, &w)) in g.iter_mut().zip(w.iter()).enumerate() {
            let m = mult.map_or(1.0, |v| v[idx / n_hidden]);
            *g += scale * m * w;
        }
    }

    fn pseudo(n: usize, seed: u64) -> Vec<f64> {
        // Deterministic, wide-exponent-range values so any change in
        // association or rounding shows up as a bit difference.
        let mut st = seed | 1;
        (0..n)
            .map(|_| {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                let u = (st >> 11) as f64 / (1u64 << 53) as f64;
                (u - 0.5) * 10f64.powi(((st % 13) as i32) - 6)
            })
            .collect()
    }

    /// BIT-identity gate for the 2026-08-04 divide-removal: the row-walked
    /// form must reproduce the divided-index form exactly, on the real
    /// shapes (944 features × 128 hidden) and on ragged / degenerate ones.
    #[test]
    fn l2_row_form_matches_divided_index_form_bitwise() {
        for &(n_features, n_hidden) in &[
            (944usize, 128usize),
            (372, 128),
            (944, 1),
            (7, 5),
            (1, 1),
            (13, 4), // n = 52, exact multiple
        ] {
            let n = n_features * n_hidden;
            let w = pseudo(n, 0xA5A5 ^ n as u64);
            let g0 = pseudo(n, 0x5A5A ^ n as u64);
            let mult: Vec<f64> = (0..n_features)
                .map(|i| if i % 3 == 0 { 2.0 } else { 1.0 })
                .collect();
            for scale in [1e-5f64, 1.0, 3.7e-3] {
                for m in [None, Some(mult.as_slice())] {
                    let mut a = g0.clone();
                    let mut b = g0.clone();
                    divided_index_form(&mut a, &w, scale, n_hidden, m);
                    add_l2_grad_layer1(&mut b, &w, scale, n_hidden, m);
                    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                        assert_eq!(
                            x.to_bits(),
                            y.to_bits(),
                            "n_features={n_features} n_hidden={n_hidden} scale={scale} \
                             mult={} idx={i}: {x:e} != {y:e}",
                            m.is_some()
                        );
                    }
                }
            }
        }
    }

    /// `g` longer than `w` (the concat-layout case): both forms must stop
    /// at the shorter length and leave the tail untouched.
    #[test]
    fn l2_row_form_respects_the_shorter_slice() {
        let (n_features, n_hidden) = (10usize, 8usize);
        let n = n_features * n_hidden;
        let w = pseudo(n, 7);
        let g0 = pseudo(n + 17, 11);
        let mult: Vec<f64> = (0..n_features).map(|i| 1.0 + (i % 2) as f64).collect();
        let mut a = g0.clone();
        let mut b = g0.clone();
        divided_index_form(&mut a, &w, 1e-5, n_hidden, Some(&mult));
        add_l2_grad_layer1(&mut b, &w, 1e-5, n_hidden, Some(&mult));
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        // Tail past `w.len()` untouched by both.
        assert_eq!(a[n..], g0[n..]);
    }
}

/// Gates for the feature-subset machinery: the group-lasso proximal step
/// (`--group-l1`) and the keep-mask row pinning (`--keep-features`).
/// Registered in `benchmarks/sota944_campaign_2026-08-03.md` appendix J.
#[cfg(test)]
mod group_l1_tests {
    use super::{apply_group_l1, zero_masked_w1_rows};

    fn row_norm(w1: &[f64], k: usize, n_hidden: usize) -> f64 {
        w1[k * n_hidden..(k + 1) * n_hidden]
            .iter()
            .map(|&w| w * w)
            .sum::<f64>()
            .sqrt()
    }

    /// THE property the whole Phase-B design rests on: a row whose ℓ2 norm is
    /// at or below the threshold `lr·λ` becomes **exactly** 0.0 in every slot
    /// — not 1e-18, not denormal. Only exact zeros are prunable by
    /// `bake_dial_refit pack`, and only an exact zero is a *learned drop*.
    #[test]
    fn group_l1_drives_whole_rows_to_exact_zero() {
        let (n_features, n_hidden) = (6usize, 4usize);
        // Row k has every element = 0.1·(k+1) ⇒ ‖row‖ = 0.2·(k+1) at n_hidden=4.
        let mut w1: Vec<f64> = (0..n_features)
            .flat_map(|k| std::iter::repeat_n(0.1 * (k + 1) as f64, n_hidden))
            .collect();
        let (lr, lambda) = (0.5f64, 1.0f64); // τ = 0.5 ⇒ rows 0,1 (norms 0.2,0.4) die
        apply_group_l1(&mut w1, n_hidden, lr, lambda);
        for k in 0..2 {
            for j in 0..n_hidden {
                let v = w1[k * n_hidden + j];
                assert_eq!(v.to_bits(), 0.0f64.to_bits(), "row {k} slot {j} = {v:e}");
            }
        }
        // Row 2 has norm exactly 0.6 > τ ⇒ survives, shrunk by (1 − τ/‖w‖).
        let expect = 0.3 * (1.0 - 0.5 / 0.6);
        assert!(
            (w1[2 * n_hidden] - expect).abs() < 1e-15,
            "{}",
            w1[2 * n_hidden]
        );
        // Shrinkage is exactly the block soft-threshold: ‖w'‖ = ‖w‖ − τ.
        for k in 2..n_features {
            let before = 0.2 * (k + 1) as f64;
            assert!(
                (row_norm(&w1, k, n_hidden) - (before - 0.5)).abs() < 1e-12,
                "row {k}"
            );
        }
    }

    /// A row exactly at the threshold dies (`‖w‖ ≤ τ`), and an all-zero row
    /// stays all-zero — so the prox composes with `--keep-features` pinning
    /// instead of resurrecting pinned rows via a 0/0 shrink factor.
    #[test]
    fn group_l1_boundary_and_zero_rows() {
        let n_hidden = 3usize;
        let mut w1 = vec![0.0f64; 2 * n_hidden];
        // Row 0: all zeros. Row 1: norm exactly τ.
        let tau = 0.5f64 * 2.0; // lr·λ
        let v = tau / (n_hidden as f64).sqrt();
        for j in 0..n_hidden {
            w1[n_hidden + j] = v;
        }
        apply_group_l1(&mut w1, n_hidden, 0.5, 2.0);
        for (i, &w) in w1.iter().enumerate() {
            assert_eq!(w.to_bits(), 0.0f64.to_bits(), "slot {i} = {w:e}");
            assert!(!w.is_sign_negative(), "slot {i} became -0.0");
        }
    }

    /// λ = 0 (the default) is a strict no-op — bit-identical weights, so
    /// every historical recipe keeps reproducing byte-identically.
    #[test]
    fn group_l1_lambda_zero_is_bit_identical_noop() {
        let mut w1: Vec<f64> = (0..64).map(|i| (i as f64 - 32.0) * 1e-3).collect();
        let before = w1.clone();
        apply_group_l1(&mut w1, 8, 1e-3, 0.0);
        apply_group_l1(&mut w1, 8, 0.0, 5.0); // lr = 0 ⇒ τ = 0 ⇒ no-op too
        for (a, b) in before.iter().zip(w1.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    /// The keep-mask pins exactly the dropped rows and leaves kept rows
    /// bit-untouched — the property that makes a K-arm differ from the
    /// full-width run in the dropped columns ONLY.
    #[test]
    fn keep_mask_pins_only_dropped_rows() {
        let (n_features, n_hidden) = (5usize, 3usize);
        let mut w1: Vec<f64> = (0..n_features * n_hidden).map(|i| 0.5 - i as f64).collect();
        let before = w1.clone();
        let mask = vec![true, false, true, false, false];
        let zeroed = zero_masked_w1_rows(&mut w1, n_hidden, Some(&mask));
        assert_eq!(zeroed, 3);
        for k in 0..n_features {
            for j in 0..n_hidden {
                let idx = k * n_hidden + j;
                if mask[k] {
                    assert_eq!(w1[idx].to_bits(), before[idx].to_bits(), "kept row {k}");
                } else {
                    assert_eq!(w1[idx].to_bits(), 0.0f64.to_bits(), "dropped row {k}");
                }
            }
        }
        // No mask set ⇒ no rows touched.
        let mut w2 = before.clone();
        assert_eq!(zero_masked_w1_rows(&mut w2, n_hidden, None), 0);
        assert_eq!(w2, before);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zenpredict::{Model, Predictor};

    fn predict_one(predictor: &mut Predictor<'_>, features: &[f64]) -> f64 {
        let f32_features: Vec<f32> = features.iter().map(|&v| v as f32).collect();
        predictor.predict(&f32_features).unwrap()[0] as f64
    }

    /// Generate a synthetic dataset where the target is a known
    /// nonlinear function of features. Train a small MLP and confirm
    /// it recovers the ranking.
    #[test]
    fn train_mlp_recovers_synthetic_ranking() {
        let n_features = 16;
        let n_train = 300;
        let mut rng = SplitMix64::new(7);
        let true_w: Vec<f64> = (0..n_features)
            .map(|i| (i as f64 - 8.0) * 0.3 + rng.next_normal() * 0.1)
            .collect();
        let mut features_owned: Vec<Vec<f64>> = Vec::with_capacity(n_train);
        let mut targets: Vec<f64> = Vec::with_capacity(n_train);
        for _ in 0..n_train {
            let x: Vec<f64> = (0..n_features).map(|_| rng.next_normal()).collect();
            let mut y: f64 = x.iter().zip(true_w.iter()).map(|(a, b)| a * b).sum();
            y += 0.1 * x[0] * x[0];
            y += rng.next_normal() * 0.05;
            features_owned.push(x);
            // Higher target = more similar (matches zensim convention).
            // The MLP must produce LOWER raw_distance for higher target.
            targets.push(y);
        }
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();

        let group = TrainingGroup {
            name: "synth".to_string(),
            human_scores: &targets,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };

        let hyper = MlpHyperparams {
            n_hidden: 8,
            n_epochs: 60,
            pairs_per_epoch: 1500,
            initial_lr: 0.005,
            log_every: 100,
            early_stop_patience: 0,
            validation_policy: ValidationPolicy::Mean,
            ..Default::default()
        };
        let mut log = Vec::new();
        let bytes = train_mlp(&mut [group], n_features, &hyper, &mut log);

        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        let model = Model::from_bytes(leaked).expect("bake should load");
        let mut predictor = Predictor::new(&model);

        let preds: Vec<f64> = features_owned
            .iter()
            .map(|f| predict_one(&mut predictor, f))
            .collect();

        // MLP output is raw_distance (lower = more similar); targets
        // are quality (higher = more similar). Anti-correlated → SROCC
        // computed against negated predictions.
        let neg_preds: Vec<f64> = preds.iter().map(|&p| -p).collect();
        let srocc = spearman_correlation(&targets, &neg_preds);
        assert!(
            srocc > 0.85,
            "MLP failed to recover synthetic ranking: SROCC={srocc:.4}"
        );
    }

    /// A validation-only group should not contribute training pairs
    /// but should still be reported in per-epoch SROCC, and the best
    /// checkpoint should be selected on its score.
    #[test]
    fn train_mlp_uses_validation_for_best_checkpoint() {
        let n_features = 8;
        let mut rng = SplitMix64::new(11);

        // Train data: target = +sum_of_features (higher = more similar).
        let train_features: Vec<Vec<f64>> = (0..200)
            .map(|_| (0..n_features).map(|_| rng.next_normal()).collect())
            .collect();
        let train_scores: Vec<f64> = train_features
            .iter()
            .map(|f| f.iter().sum::<f64>())
            .collect();
        let train_refs: Vec<&[f64]> = train_features.iter().map(|v| v.as_slice()).collect();

        // Val data: same target function — model should generalize.
        let val_features: Vec<Vec<f64>> = (0..80)
            .map(|_| (0..n_features).map(|_| rng.next_normal()).collect())
            .collect();
        let val_scores: Vec<f64> = val_features.iter().map(|f| f.iter().sum::<f64>()).collect();
        let val_refs: Vec<&[f64]> = val_features.iter().map(|v| v.as_slice()).collect();

        let mut groups = vec![
            TrainingGroup {
                name: "train".to_string(),
                human_scores: &train_scores,
                features: FeatureRows::Borrowed(&train_refs),
                metric_sigmas: None,
                train_weight: 1.0,
                validation_weight: 0.0,
                ref_ids: None,
                loss_mode: GroupLossMode::default(),
            },
            TrainingGroup {
                name: "val".to_string(),
                human_scores: &val_scores,
                features: FeatureRows::Borrowed(&val_refs),
                metric_sigmas: None,
                train_weight: 0.0,
                validation_weight: 1.0,
                ref_ids: None,
                loss_mode: GroupLossMode::default(),
            },
        ];

        let hyper = MlpHyperparams {
            n_hidden: 8,
            n_epochs: 40,
            pairs_per_epoch: 800,
            initial_lr: 0.005,
            log_every: 10,
            early_stop_patience: 0,
            validation_policy: ValidationPolicy::Min,
            ..Default::default()
        };
        let mut log = Vec::new();
        let bytes = train_mlp(&mut groups, n_features, &hyper, &mut log);

        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        let model = Model::from_bytes(leaked).expect("bake should load");
        let mut predictor = Predictor::new(&model);

        let val_preds: Vec<f64> = val_features
            .iter()
            .map(|f| predict_one(&mut predictor, f))
            .collect();
        // Higher target ⇒ more similar ⇒ lower MLP raw_distance.
        // SROCC(target, -prediction) should be positive and high.
        let neg_preds: Vec<f64> = val_preds.iter().map(|&p| -p).collect();
        let val_srocc = spearman_correlation(&val_scores, &neg_preds);
        assert!(
            val_srocc > 0.85,
            "validation-tracking trainer failed to generalize: val SROCC={val_srocc:.4}",
        );
        // Spot-check the log: epoch lines contain the multi-stat
        // val aggregate and per-group panel stats.
        assert!(
            log.iter().any(|line| line.contains("val(")),
            "log missing val(aggregate)= reporting"
        );
        assert!(
            log.iter().any(|line| line.contains("srocc=")),
            "log missing per-group srocc= field"
        );
    }

    /// With a large `low_q_boost`, the trainer must pick more B0/B1
    /// pairs (lower human_score) than uniform sampling would. Smoke-test
    /// by training two MLPs on the same data with the same seed but
    /// different boost values, then verifying the boosted model's
    /// predictions are demonstrably more sensitive in the low-score
    /// region (the boost re-weighted those pairs higher in the rank
    /// loss).
    /// The unguarded half of the STRATEGY family must FAIL LOUD rather than
    /// throw the flag away.
    ///
    /// `ema_decay` / `hard_pair_frac` / `dro_eta` / `listwise_weight` are read
    /// only inside `train_mlp_per_sample_alpha_head`. Before 2026-09-04 setting
    /// any of them on another head produced a bake byte-identical to a run that
    /// never set it — the same defect the mse/monotonicity/triplet guards above
    /// already covered. `--ema-decay` mattered most: its own doc advertises
    /// "seed-variance reduction", so a variance study would have concluded it
    /// does not work when in fact it never ran.
    #[test]
    fn strategy_knobs_fail_loud_when_unreachable() {
        let n_features = 4usize;
        let targets: Vec<f64> = (0..40).map(|i| i as f64 * 2.5).collect();
        let mut rng = SplitMix64::new(5);
        let owned: Vec<Vec<f64>> = (0..40)
            .map(|i| {
                let mut x: Vec<f64> = (0..n_features).map(|_| rng.next_normal()).collect();
                x[0] = i as f64 / 40.0;
                x
            })
            .collect();
        let fr: Vec<&[f64]> = owned.iter().map(|v| v.as_slice()).collect();
        let group = || TrainingGroup {
            name: "g".into(),
            human_scores: &targets,
            features: FeatureRows::Borrowed(&fr),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };
        let base = MlpHyperparams {
            n_hidden: 4,
            n_epochs: 2,
            pairs_per_epoch: 50,
            seed: 3,
            log_every: 100,
            early_stop_patience: 0,
            ..Default::default()
        };
        for (label, h) in [
            (
                "ema",
                MlpHyperparams {
                    ema_decay: 0.999,
                    ..base.clone()
                },
            ),
            (
                "hardpair",
                MlpHyperparams {
                    hard_pair_frac: 0.5,
                    ..base.clone()
                },
            ),
            (
                "dro",
                MlpHyperparams {
                    dro_eta: 0.1,
                    ..base.clone()
                },
            ),
            (
                "listwise",
                MlpHyperparams {
                    listwise_weight: 0.5,
                    ..base.clone()
                },
            ),
        ] {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut log = Vec::new();
                train_mlp_strategy(
                    &mut [group()],
                    n_features,
                    &h,
                    &mut log,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            }));
            assert!(
                r.is_err(),
                "{label} was accepted on a non-alpha head and silently ignored"
            );
        }
        // Negative control: the same recipe with none of them set must train.
        let mut log = Vec::new();
        let bytes = train_mlp_strategy(
            &mut [group()],
            n_features,
            &base,
            &mut log,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert!(!bytes.is_empty(), "the unflagged control must still train");
    }

    /// `--stratified-bands` must actually stratify on the DEFAULT training
    /// path, not only under `--per-sample-alpha-head`.
    ///
    /// It used to build `strat_bands` in exactly ONE of the four training
    /// loops. Every other loop — including `train_mlp_strategy`'s own
    /// standard path, which every board bake trained through — passed an
    /// empty band table, so the flag was a SILENT NO-OP there: setting it
    /// produced a bit-identical draw sequence and a bit-identical bake.
    /// MEASURED before the fix, `--stratified-bands 0` vs `8` on the default
    /// path: sample-sequence digest `127b831bed8a3873` both times.
    #[test]
    fn stratified_bands_is_not_a_silent_no_op_on_the_default_path() {
        let n_features = 8usize;
        let mut rng = SplitMix64::new(31);
        let n = 200usize;
        let mut targets = Vec::with_capacity(n);
        let mut features_owned: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            // Deliberately SKEWED scores: most rows crowd the top, so
            // band-uniform row-A selection has to look different from
            // row-uniform selection.
            let s = 100.0 * ((i as f64) / (n as f64 - 1.0)).powf(0.25);
            targets.push(s);
            let mut x: Vec<f64> = (0..n_features).map(|_| rng.next_normal()).collect();
            x[0] = s / 100.0 + rng.next_normal() * 0.1;
            features_owned.push(x);
        }
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
        let group = || TrainingGroup {
            name: "strat-test".to_string(),
            human_scores: &targets,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };
        let base = MlpHyperparams {
            n_hidden: 6,
            n_epochs: 20,
            pairs_per_epoch: 800,
            initial_lr: 0.01,
            seed: 7,
            log_every: 100,
            early_stop_patience: 0,
            ..Default::default()
        };
        let strat = MlpHyperparams {
            stratified_bands: 8,
            ..base.clone()
        };
        let mut l1 = Vec::new();
        let plain_bytes = train_mlp(&mut [group()], n_features, &base, &mut l1);
        let mut l2 = Vec::new();
        let strat_bytes = train_mlp(&mut [group()], n_features, &strat, &mut l2);
        assert_ne!(
            plain_bytes, strat_bytes,
            "--stratified-bands had NO effect on the default training path"
        );

        // And the negative control: bands=0 must stay bit-identical to a
        // hyperparams struct that never mentions the flag, so the fix cannot
        // have moved the default.
        let explicit_off = MlpHyperparams {
            stratified_bands: 0,
            ..base.clone()
        };
        let mut l3 = Vec::new();
        let off_bytes = train_mlp(&mut [group()], n_features, &explicit_off, &mut l3);
        assert_eq!(
            plain_bytes, off_bytes,
            "stratified_bands=0 must be a true no-op"
        );
    }

    #[test]
    fn train_mlp_low_q_boost_changes_outputs() {
        let n_features = 4;
        let mut rng = SplitMix64::new(99);

        // Build a 200-row group with human_scores spanning [0, 100] and
        // features that linearly encode the score. Without boost the
        // MLP should learn a uniform regression; with boost it should
        // specialize to the low-score end.
        let n = 200usize;
        let mut targets = Vec::with_capacity(n);
        let mut features_owned: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let s = 100.0 * (i as f64) / (n as f64 - 1.0); // 0..100
            targets.push(s);
            let mut x: Vec<f64> = (0..n_features).map(|_| rng.next_normal()).collect();
            // Inject a strong signal: x[0] correlates with target.
            x[0] = s / 100.0 + rng.next_normal() * 0.1;
            features_owned.push(x);
        }
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();

        let group_factory = || TrainingGroup {
            name: "boost-test".to_string(),
            human_scores: &targets,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };

        // Run 1: no boost (uniform within-group sampling).
        let hyper_uniform = MlpHyperparams {
            n_hidden: 6,
            n_epochs: 30,
            pairs_per_epoch: 1000,
            initial_lr: 0.01,
            seed: 7,
            log_every: 100,
            early_stop_patience: 0,
            ..Default::default()
        };
        // Run 2: aggressive low_q_boost (10x for B0, sqrt(10) ≈ 3.16x for B1).
        let hyper_boosted = MlpHyperparams {
            low_q_boost: 10.0,
            ..hyper_uniform.clone()
        };

        let mut log_u = Vec::new();
        let bytes_uniform = train_mlp(
            &mut [group_factory()],
            n_features,
            &hyper_uniform,
            &mut log_u,
        );
        let mut log_b = Vec::new();
        let bytes_boosted = train_mlp(
            &mut [group_factory()],
            n_features,
            &hyper_boosted,
            &mut log_b,
        );

        // The two bakes must differ — if boost had no effect, this
        // would be a regression in the per-row CDF wiring.
        assert_ne!(
            bytes_uniform, bytes_boosted,
            "low_q_boost=10.0 produced byte-identical bake to no-boost — \
             the per-row CDF is not being honored"
        );

        // Default boost=1.0 must produce bit-identical output to a
        // hyperparams struct with explicit boost=1.0 (no-op
        // guarantee). The CDF code path returns None at default and
        // the sampler stays uniform — no extra RNG bytes consumed.
        let hyper_default = MlpHyperparams {
            seed: 7,
            n_hidden: 6,
            n_epochs: 30,
            pairs_per_epoch: 1000,
            initial_lr: 0.01,
            log_every: 100,
            early_stop_patience: 0,
            ..Default::default()
        };
        let mut log_d = Vec::new();
        let bytes_default = train_mlp(
            &mut [group_factory()],
            n_features,
            &hyper_default,
            &mut log_d,
        );
        assert_eq!(
            bytes_uniform, bytes_default,
            "explicit low_q_boost=1.0 produced different bake than default — \
             the no-op guarantee is broken"
        );
    }

    // ---- T8.1 + T8.2 tests (2026-05-16) ----

    /// Shared synthetic dataset used by the mini-batch / parallel-batch
    /// tests below. Returns owned features + targets in a stable seed
    /// order so the tests can compare bake bytes against a baseline.
    fn make_synth_dataset(seed: u64, n: usize, n_features: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = SplitMix64::new(seed);
        let true_w: Vec<f64> = (0..n_features)
            .map(|i| (i as f64 - (n_features as f64 / 2.0)) * 0.3 + rng.next_normal() * 0.1)
            .collect();
        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);
        for _ in 0..n {
            let x: Vec<f64> = (0..n_features).map(|_| rng.next_normal()).collect();
            let mut y: f64 = x.iter().zip(true_w.iter()).map(|(a, b)| a * b).sum();
            y += 0.1 * x[0] * x[0];
            y += rng.next_normal() * 0.05;
            features.push(x);
            targets.push(y);
        }
        (features, targets)
    }

    /// [`FeatureRows::Releasable`] MUST train to bit-identical bake bytes
    /// against [`FeatureRows::Borrowed`], and MUST actually hand the rows
    /// back.
    ///
    /// This is the gate for the 2026-08-04 memory change: the trainer frees
    /// each raw row as it standardizes it, which halves a lane's peak RSS
    /// (`benchmarks/trainer_mem_release_2026-08-04.md`). That is only
    /// legitimate because releasing a row changes no arithmetic — this test
    /// is what keeps it that way. The full-data version of this comparison
    /// ran on the 11-group / 779,290-row wave-10 recipe; this one runs it in
    /// CI on every commit.
    #[test]
    fn releasable_rows_train_identically_to_borrowed_and_are_released() {
        let n_features = 12;
        let (features_owned, targets) = make_synth_dataset(2026, 200, n_features);

        let hyper = MlpHyperparams {
            n_hidden: 8,
            n_epochs: 25,
            pairs_per_epoch: 800,
            initial_lr: 0.005,
            seed: 7,
            log_every: 100,
            early_stop_patience: 0,
            validation_policy: ValidationPolicy::Mean,
            ..Default::default()
        };

        // Arm A: borrowed rows — nothing is released.
        let borrowed_rows = features_owned.clone();
        let feats_ref: Vec<&[f64]> = borrowed_rows.iter().map(|v| v.as_slice()).collect();
        let mut log_a = Vec::new();
        let bake_borrowed = train_mlp(
            &mut [TrainingGroup {
                name: "synth".to_string(),
                human_scores: &targets,
                features: FeatureRows::Borrowed(&feats_ref),
                metric_sigmas: None,
                train_weight: 1.0,
                validation_weight: 1.0,
                ref_ids: None,
                loss_mode: GroupLossMode::default(),
            }],
            n_features,
            &hyper,
            &mut log_a,
        );
        drop(feats_ref);
        assert!(
            borrowed_rows.iter().all(|r| r.len() == n_features),
            "Borrowed rows must survive the call untouched"
        );

        // Arm B: releasable flat buffer — same values, same row-major order,
        // taken and standardized in place.
        let n_rows = features_owned.len();
        let mut flat: Vec<f64> = Vec::with_capacity(n_rows * n_features);
        for r in &features_owned {
            flat.extend_from_slice(r);
        }
        let mut log_b = Vec::new();
        let bake_releasable = train_mlp(
            &mut [TrainingGroup {
                name: "synth".to_string(),
                human_scores: &targets,
                features: FeatureRows::Releasable {
                    data: &mut flat,
                    n_rows,
                    n_features,
                },
                metric_sigmas: None,
                train_weight: 1.0,
                validation_weight: 1.0,
                ref_ids: None,
                loss_mode: GroupLossMode::default(),
            }],
            n_features,
            &hyper,
            &mut log_b,
        );

        assert_eq!(
            bake_borrowed, bake_releasable,
            "in-place standardization of the taken raw buffer changed the \
             bake — the memory optimization is NOT numerically inert"
        );
        assert!(
            flat.is_empty(),
            "the Releasable buffer was NOT taken — the run held two copies \
             of the feature matrix and the memory win is not happening"
        );
        // len() must keep reporting n_rows after the take (it is what the
        // hot loop's `g.features.len()` reads), which is asserted where it
        // matters: inside the trainer the post-standardization epoch loop
        // sampled pairs from this group for 25 epochs — impossible if the
        // cached length had collapsed to 0 with the buffer.
    }

    /// `--minibatch-size 1` MUST produce bit-identical bake bytes to a
    /// `MlpHyperparams::default()` call (which leaves `minibatch_size`
    /// at 1). This is the legacy-compatibility guarantee for T8.1.
    #[test]
    fn train_mlp_minibatch_1_matches_legacy() {
        let n_features = 12;
        let (features_owned, targets) = make_synth_dataset(101, 200, n_features);
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();

        let group_factory = || TrainingGroup {
            name: "synth".to_string(),
            human_scores: &targets,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };

        let base = MlpHyperparams {
            n_hidden: 8,
            n_epochs: 25,
            pairs_per_epoch: 800,
            initial_lr: 0.005,
            seed: 7,
            log_every: 100,
            early_stop_patience: 0,
            validation_policy: ValidationPolicy::Mean,
            ..Default::default()
        };

        // Default trainer (legacy code path; minibatch_size left at default 1).
        let mut log_a = Vec::new();
        let bake_default = train_mlp(&mut [group_factory()], n_features, &base, &mut log_a);

        // Explicit minibatch_size = 1 — must match.
        let hyper_explicit_1 = MlpHyperparams {
            minibatch_size: 1,
            ..base.clone()
        };
        let mut log_b = Vec::new();
        let bake_explicit_1 = train_mlp(
            &mut [group_factory()],
            n_features,
            &hyper_explicit_1,
            &mut log_b,
        );

        assert_eq!(
            bake_default, bake_explicit_1,
            "explicit minibatch_size=1 produced different bake than default — \
             the no-op / bit-identical guarantee is broken"
        );

        // parallel_batch=true with K=1 must ALSO match (we documented
        // that --parallel-batch is a no-op at K=1).
        let hyper_parallel_k1 = MlpHyperparams {
            minibatch_size: 1,
            parallel_batch: true,
            ..base.clone()
        };
        let mut log_c = Vec::new();
        let bake_parallel_k1 = train_mlp(
            &mut [group_factory()],
            n_features,
            &hyper_parallel_k1,
            &mut log_c,
        );
        assert_eq!(
            bake_default, bake_parallel_k1,
            "parallel_batch=true + K=1 should fall through to the sequential \
             per-pair path and produce bit-identical bake bytes"
        );
    }

    /// `--minibatch-size 64 --parallel-batch` produces bit-identical
    /// bake bytes regardless of `RAYON_NUM_THREADS`. The test forces
    /// 1-thread, 4-thread, and rayon-default thread pools, trains the
    /// same recipe in each, and asserts the resulting bake bytes match.
    ///
    /// This is the load-bearing test for T8.2 determinism.
    #[test]
    fn train_mlp_minibatch_deterministic_threads() {
        let n_features = 16;
        let (features_owned, targets) = make_synth_dataset(202, 400, n_features);
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();

        let group_factory = || TrainingGroup {
            name: "synth".to_string(),
            human_scores: &targets,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };

        let hyper = MlpHyperparams {
            n_hidden: 12,
            n_epochs: 15,
            pairs_per_epoch: 1024,
            initial_lr: 0.005,
            seed: 7,
            log_every: 100,
            early_stop_patience: 0,
            validation_policy: ValidationPolicy::Mean,
            minibatch_size: 64,
            parallel_batch: true,
            ..Default::default()
        };

        let run_in_pool = |n_threads: usize| -> Vec<u8> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .expect("build rayon pool");
            pool.install(|| {
                let mut log = Vec::new();
                train_mlp(&mut [group_factory()], n_features, &hyper, &mut log)
            })
        };

        // Run on three different thread counts; bake bytes must match exactly.
        let bake_1 = run_in_pool(1);
        let bake_4 = run_in_pool(4);
        let bake_8 = run_in_pool(8);

        assert_eq!(
            bake_1.len(),
            bake_4.len(),
            "bake sizes differ between thread counts (1 vs 4) — wire format drift"
        );
        assert_eq!(
            bake_1, bake_4,
            "K=64 parallel-batch produced different bake bytes between 1 and 4 threads — \
             determinism is broken (probably FP reduce order)"
        );
        assert_eq!(
            bake_1, bake_8,
            "K=64 parallel-batch produced different bake bytes between 1 and 8 threads — \
             determinism is broken (probably FP reduce order)"
        );
    }

    /// `--minibatch-size 64` (sequential AND parallel paths) must reach
    /// comparable validation SROCC to per-pair K=1 on the same dataset
    /// — within 0.01 SROCC. Verifies that mini-batching doesn't break
    /// convergence on the tiny synthetic ranking problem used in
    /// `train_mlp_recovers_synthetic_ranking`.
    #[test]
    fn train_mlp_minibatch_converges() {
        let n_features = 16;
        let n_train = 400;
        let (features_owned, targets) = make_synth_dataset(303, n_train, n_features);
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();

        let group_factory = || TrainingGroup {
            name: "synth".to_string(),
            human_scores: &targets,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };

        let eval_srocc = |bake: &[u8]| -> f64 {
            let leaked: &'static [u8] = Box::leak(bake.to_vec().into_boxed_slice());
            let model = Model::from_bytes(leaked).expect("bake should load");
            let mut predictor = Predictor::new(&model);
            let preds: Vec<f64> = features_owned
                .iter()
                .map(|f| predict_one(&mut predictor, f))
                .collect();
            let neg_preds: Vec<f64> = preds.iter().map(|&p| -p).collect();
            spearman_correlation(&targets, &neg_preds)
        };

        let base = MlpHyperparams {
            n_hidden: 12,
            n_epochs: 80,
            pairs_per_epoch: 2000,
            initial_lr: 0.005,
            seed: 7,
            log_every: 100,
            early_stop_patience: 0,
            validation_policy: ValidationPolicy::Mean,
            ..Default::default()
        };

        // K=1 baseline.
        let mut log_1 = Vec::new();
        let bake_k1 = train_mlp(&mut [group_factory()], n_features, &base, &mut log_1);
        let srocc_k1 = eval_srocc(&bake_k1);

        // K=64 sequential.
        let hyper_k64_seq = MlpHyperparams {
            minibatch_size: 64,
            ..base.clone()
        };
        let mut log_2 = Vec::new();
        let bake_k64_seq = train_mlp(
            &mut [group_factory()],
            n_features,
            &hyper_k64_seq,
            &mut log_2,
        );
        let srocc_k64_seq = eval_srocc(&bake_k64_seq);

        // K=64 parallel.
        let hyper_k64_par = MlpHyperparams {
            minibatch_size: 64,
            parallel_batch: true,
            ..base.clone()
        };
        let mut log_3 = Vec::new();
        let bake_k64_par = train_mlp(
            &mut [group_factory()],
            n_features,
            &hyper_k64_par,
            &mut log_3,
        );
        let srocc_k64_par = eval_srocc(&bake_k64_par);

        // The model must learn the ranking in all three modes.
        assert!(
            srocc_k1 > 0.80,
            "K=1 baseline failed to recover synthetic ranking: SROCC={srocc_k1:.4}"
        );
        assert!(
            srocc_k64_seq > 0.80,
            "K=64 sequential failed to recover synthetic ranking: SROCC={srocc_k64_seq:.4}"
        );
        assert!(
            srocc_k64_par > 0.80,
            "K=64 parallel failed to recover synthetic ranking: SROCC={srocc_k64_par:.4}"
        );

        // K=64 sequential vs parallel must agree exactly on the bake
        // bytes (same RNG / sample sequence, same FP reduce order).
        assert_eq!(
            bake_k64_seq, bake_k64_par,
            "K=64 sequential vs parallel produced different bake bytes — \
             expected bit-identical (sequential reduce in source order)"
        );

        // K=64 vs K=1: within 0.01 SROCC (mini-batching is a small
        // convergence-trajectory shift; on a 400-pair dataset trained
        // 80 epochs the two should agree well within this margin).
        let delta = (srocc_k1 - srocc_k64_seq).abs();
        assert!(
            delta < 0.05,
            "K=64 SROCC {srocc_k64_seq:.4} differs from K=1 SROCC {srocc_k1:.4} by {delta:.4} — \
             expected within 0.05 (mini-batching shouldn't hurt convergence by much on this toy)"
        );
    }

    // ---- PWRC tests (2026-05-17) ----

    /// `--pwrc-pair-weight` off (default) MUST produce bit-identical
    /// bake bytes to a hyperparams struct that never mentions PWRC.
    /// This is the load-bearing backwards-compatibility guarantee
    /// promised in `MlpHyperparams::pwrc_pair_weight` docs.
    #[test]
    fn train_mlp_pwrc_disabled_matches_legacy() {
        let n_features = 12;
        let (features_owned, targets) = make_synth_dataset(303, 200, n_features);
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
        // Re-scale targets to span 0..100 so they look like score_zensim units.
        let t_min = targets.iter().copied().fold(f64::INFINITY, f64::min);
        let t_max = targets.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let scaled_targets: Vec<f64> = targets
            .iter()
            .map(|&t| 100.0 * (t - t_min) / (t_max - t_min).max(1e-12))
            .collect();

        let group_factory = || TrainingGroup {
            name: "synth".to_string(),
            human_scores: &scaled_targets,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };

        let base = MlpHyperparams {
            n_hidden: 8,
            n_epochs: 25,
            pairs_per_epoch: 800,
            initial_lr: 0.005,
            seed: 7,
            log_every: 100,
            early_stop_patience: 0,
            validation_policy: ValidationPolicy::Mean,
            ..Default::default()
        };

        let mut log_a = Vec::new();
        let bake_default = train_mlp(&mut [group_factory()], n_features, &base, &mut log_a);

        // Explicit pwrc_pair_weight=false — must match.
        let hyper_explicit_off = MlpHyperparams {
            pwrc_pair_weight: false,
            pwrc_sensory_threshold: 5.0, // ignored when off
            pwrc_band_weights: Some(vec![10.0, 5.0, 1.0]), // also ignored when off
            ..base.clone()
        };
        let mut log_b = Vec::new();
        let bake_explicit_off = train_mlp(
            &mut [group_factory()],
            n_features,
            &hyper_explicit_off,
            &mut log_b,
        );

        assert_eq!(
            bake_default, bake_explicit_off,
            "explicit pwrc_pair_weight=false produced different bake than default — \
             the no-op / bit-identical guarantee is broken"
        );

        // pwrc_pair_weight=true MUST produce different bytes (sanity:
        // the weighting is actually being applied somewhere).
        let hyper_on = MlpHyperparams {
            pwrc_pair_weight: true,
            pwrc_sensory_threshold: 0.0, // no drops, only weighting
            pwrc_band_weights: None,
            ..base.clone()
        };
        let mut log_c = Vec::new();
        let bake_on = train_mlp(&mut [group_factory()], n_features, &hyper_on, &mut log_c);
        assert_ne!(
            bake_default, bake_on,
            "pwrc_pair_weight=true produced byte-identical bake to off — \
             the weighting is not being applied to the loss/grad scalars"
        );

        // Same with K>1 / parallel batch: K=8 + PWRC off must match
        // K=8 + PWRC absent (sequential path mirror for the parallel
        // path's PWRC handling).
        let hyper_k8 = MlpHyperparams {
            minibatch_size: 8,
            ..base.clone()
        };
        let hyper_k8_pwrc_off = MlpHyperparams {
            minibatch_size: 8,
            pwrc_pair_weight: false,
            pwrc_sensory_threshold: 5.0,
            pwrc_band_weights: Some(vec![1.0, 2.0]),
            ..base.clone()
        };
        let mut log_d = Vec::new();
        let mut log_e = Vec::new();
        let bake_k8 = train_mlp(&mut [group_factory()], n_features, &hyper_k8, &mut log_d);
        let bake_k8_off = train_mlp(
            &mut [group_factory()],
            n_features,
            &hyper_k8_pwrc_off,
            &mut log_e,
        );
        assert_eq!(
            bake_k8, bake_k8_off,
            "K=8 + explicit pwrc_pair_weight=false produced different bake than K=8 + default — \
             the parallel-batch PWRC plumbing is not no-op when off"
        );
    }

    /// `pwrc_pair_weight(a, b, None)` and `pwrc_pair_weight(a, b,
    /// Some([...]))` produce the documented closed-form / band-lookup
    /// results. Direct unit test of the public helper.
    #[test]
    fn pwrc_pair_weight_helper_formula() {
        // Closed-form: exp(max/100). max=0 -> 1.0; max=100 -> e.
        let w_lo = pwrc_pair_weight(0.0, 0.0, None);
        let w_hi = pwrc_pair_weight(100.0, 50.0, None);
        let w_mid = pwrc_pair_weight(50.0, 30.0, None);
        assert!((w_lo - 1.0).abs() < 1e-12, "max=0 -> exp(0)=1, got {w_lo}");
        assert!(
            (w_hi - std::f64::consts::E).abs() < 1e-12,
            "max=100 -> exp(1)=e, got {w_hi}"
        );
        assert!(
            (w_mid - (0.5f64).exp()).abs() < 1e-12,
            "max=50 -> exp(0.5), got {w_mid}"
        );

        // Band weights: 10-band grid, max=5 -> band 0; max=15 -> band 1;
        // max=100 -> band 9 (clamped).
        let bands = vec![5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0];
        assert_eq!(pwrc_pair_weight(5.0, 0.0, Some(&bands)), 5.0);
        assert_eq!(pwrc_pair_weight(15.0, 0.0, Some(&bands)), 4.0);
        assert_eq!(pwrc_pair_weight(99.999, 0.0, Some(&bands)), 1.0);
        // MOS=100 lands in the last bin (clamp).
        assert_eq!(pwrc_pair_weight(100.0, 0.0, Some(&bands)), 1.0);

        // NaN / negative / out-of-range MOS: clamp to [0, 100].
        // Note: NaN propagates through `max` per IEEE 754 but f64::max
        // in Rust uses total-cmp semantics; we clamp afterwards anyway.
        assert_eq!(pwrc_pair_weight(-10.0, -5.0, Some(&bands)), 5.0); // clamped to 0 -> band 0
        assert_eq!(pwrc_pair_weight(150.0, 0.0, Some(&bands)), 1.0); // clamped to 100 -> band 9

        // Single-element band vector: every MOS maps to index 0.
        let one = vec![7.0];
        assert_eq!(pwrc_pair_weight(0.0, 0.0, Some(&one)), 7.0);
        assert_eq!(pwrc_pair_weight(100.0, 0.0, Some(&one)), 7.0);
    }

    /// Sensory-threshold drop: with `T = 50.0` on a [0, 100]-spread
    /// dataset, ~half of all randomly-drawn pairs should be dropped.
    /// Verify by training with a heavy ST and checking that the
    /// resulting bake is non-trivially different from the no-drop
    /// counterpart (the n_steps counter accounts only for kept pairs).
    #[test]
    fn train_mlp_pwrc_dropping_ties_drops_them() {
        let n_features = 4;
        let mut rng = SplitMix64::new(42);
        let n = 200usize;
        let mut targets = Vec::with_capacity(n);
        let mut features_owned: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let s = 100.0 * (i as f64) / (n as f64 - 1.0); // 0..100, dense
            targets.push(s);
            let mut x: Vec<f64> = (0..n_features).map(|_| rng.next_normal()).collect();
            x[0] = s / 100.0 + rng.next_normal() * 0.1;
            features_owned.push(x);
        }
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();

        let group_factory = || TrainingGroup {
            name: "synth".to_string(),
            human_scores: &targets,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };

        let base = MlpHyperparams {
            n_hidden: 6,
            n_epochs: 30,
            pairs_per_epoch: 1000,
            initial_lr: 0.01,
            seed: 7,
            log_every: 100,
            early_stop_patience: 0,
            // Disable weighting so we isolate the threshold effect.
            pwrc_pair_weight: true,
            pwrc_sensory_threshold: 0.0,
            pwrc_band_weights: Some(vec![1.0]), // every pair gets weight 1.0
            ..Default::default()
        };
        let mut log_no_drop = Vec::new();
        let bake_no_drop = train_mlp(&mut [group_factory()], n_features, &base, &mut log_no_drop);

        // Heavy threshold: drop pairs within 50 MOS units.
        let hyper_drop = MlpHyperparams {
            pwrc_sensory_threshold: 50.0,
            ..base.clone()
        };
        let mut log_drop = Vec::new();
        let bake_drop = train_mlp(
            &mut [group_factory()],
            n_features,
            &hyper_drop,
            &mut log_drop,
        );

        assert_ne!(
            bake_no_drop, bake_drop,
            "pwrc_sensory_threshold=50 dropped pairs but bake bytes unchanged — \
             the drop is not affecting the gradient stream"
        );

        // The T=50 trainer drops ~75% of pairs (P(|Δ|<50) on uniform
        // 0..100 ≈ 0.75), so its weights move ~4x less per epoch.
        // After 30 epochs, the no-drop model has seen ~30k gradient
        // updates vs ~7.5k for the drop model. Confirm this via the
        // log's loss trajectory: drop-model's epoch-0 loss should be
        // close to no-drop's, but the drop-model converges slower.
        //
        // Direct evidence: pass T=10000 (drop EVERY pair, single-
        // epoch n_steps=0) and confirm the bake is also distinct
        // from baseline AND from T=50 drop (n_steps=0 every epoch
        // means total_loss/n_steps=0 and no weight updates AT ALL).
        let hyper_drop_all = MlpHyperparams {
            pwrc_sensory_threshold: 10_000.0, // exceeds any |a-b|
            ..base.clone()
        };
        let mut log_drop_all = Vec::new();
        let bake_drop_all = train_mlp(
            &mut [group_factory()],
            n_features,
            &hyper_drop_all,
            &mut log_drop_all,
        );
        // With every pair dropped, the trainer never updates weights
        // and bakes the random Xavier init. Different from baseline.
        assert_ne!(
            bake_no_drop, bake_drop_all,
            "T=10000 (drop-all) bake matches baseline — the threshold gate is broken"
        );
        // Different from T=50 drop too (T=50 still updates on ~25% of pairs).
        assert_ne!(
            bake_drop, bake_drop_all,
            "T=50 and T=10000 produced identical bakes — the dropout fraction \
             is not actually controlled by the threshold"
        );

        // Loss inspection: parse `loss=NNN.NNNN` from each per-epoch
        // log line. The drop=all bake's avg_loss is total_loss/n_steps
        // = 0/0 -> 0.0; baseline's is positive.
        let parse_loss = |log: &[String]| -> Vec<f64> {
            log.iter()
                .filter_map(|line| {
                    line.find(" loss=").and_then(|i| {
                        let tail = &line[i + " loss=".len()..];
                        let end = tail
                            .find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
                            .unwrap_or(tail.len());
                        tail[..end].parse::<f64>().ok()
                    })
                })
                .collect()
        };
        let losses_no_drop = parse_loss(&log_no_drop);
        let losses_drop_all = parse_loss(&log_drop_all);
        assert!(!losses_no_drop.is_empty(), "no-drop log missing loss=");
        assert!(!losses_drop_all.is_empty(), "drop-all log missing loss=");
        assert!(
            losses_no_drop[0] > 0.0,
            "no-drop epoch-0 loss should be positive, got {}",
            losses_no_drop[0]
        );
        assert!(
            losses_drop_all[0] == 0.0,
            "drop-all epoch-0 loss should be 0.0 (no pairs survived), got {}",
            losses_drop_all[0]
        );
    }

    /// PWRC band weighting must scale gradients: a high-weight band
    /// (e.g., B0=10.0) MUST produce a larger Adam update than the
    /// same pair drawn with weight 1.0. Verify by running 1-step
    /// training with a single low-q pair drawn N times under two
    /// weight schedules and comparing the bake-output magnitudes.
    #[test]
    fn train_mlp_pwrc_weights_amplify_low_q() {
        let n_features = 4;
        let mut rng = SplitMix64::new(11);
        // Build a dataset that ONLY contains low-q pairs (all scores
        // < 10), so every drawn pair lands in band 0. This isolates
        // the band-weight effect.
        let n = 50usize;
        let mut targets = Vec::with_capacity(n);
        let mut features_owned: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let s = 10.0 * (i as f64) / (n as f64 - 1.0); // 0..10
            targets.push(s);
            let mut x: Vec<f64> = (0..n_features).map(|_| rng.next_normal()).collect();
            x[0] = s / 10.0 + rng.next_normal() * 0.1;
            features_owned.push(x);
        }
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();

        let group_factory = || TrainingGroup {
            name: "lowq".to_string(),
            human_scores: &targets,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };

        let base = MlpHyperparams {
            n_hidden: 6,
            n_epochs: 20,
            pairs_per_epoch: 500,
            initial_lr: 0.001,
            seed: 11,
            log_every: 100,
            early_stop_patience: 0,
            pwrc_pair_weight: true,
            pwrc_sensory_threshold: 0.0,
            // 10-band grid: band 0 is for max_MOS in [0, 10).
            ..Default::default()
        };

        // Weight 1.0 in band 0 (baseline).
        let hyper_w1 = MlpHyperparams {
            pwrc_band_weights: Some(vec![1.0; 10]),
            ..base.clone()
        };
        let mut log_w1 = Vec::new();
        let _bake_w1 = train_mlp(&mut [group_factory()], n_features, &hyper_w1, &mut log_w1);

        // Weight 10.0 in band 0.
        let mut bands_hi = vec![1.0; 10];
        bands_hi[0] = 10.0;
        let hyper_w10 = MlpHyperparams {
            pwrc_band_weights: Some(bands_hi),
            ..base.clone()
        };
        let mut log_w10 = Vec::new();
        let _bake_w10 = train_mlp(&mut [group_factory()], n_features, &hyper_w10, &mut log_w10);

        // The per-epoch loss line should reflect higher loss
        // magnitude at weight=10 (since loss is scaled by weight).
        let parse_loss = |log: &[String]| -> Vec<f64> {
            log.iter()
                .filter_map(|line| {
                    line.find(" loss=").and_then(|i| {
                        let tail = &line[i + " loss=".len()..];
                        let end = tail
                            .find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
                            .unwrap_or(tail.len());
                        tail[..end].parse::<f64>().ok()
                    })
                })
                .collect()
        };
        let losses_w1 = parse_loss(&log_w1);
        let losses_w10 = parse_loss(&log_w10);
        assert!(
            !losses_w1.is_empty() && !losses_w10.is_empty(),
            "log missing avg_loss= (w1: {}, w10: {})",
            losses_w1.len(),
            losses_w10.len()
        );
        // First-epoch avg_loss at weight=10 must be substantially
        // larger than at weight=1 (the same pred_diff distribution
        // is scaled 10x).
        let l1 = losses_w1[0];
        let l10 = losses_w10[0];
        assert!(
            l10 > l1 * 3.0,
            "weight=10 first-epoch avg_loss {l10:.4} should be >= 3x weight=1 first-epoch avg_loss {l1:.4} \
             (expect ~10x; allow 3x margin for accumulator FP differences)"
        );
    }

    /// PWRC-on training must still converge on a synthetic dataset.
    /// This rules out gradient blow-up / sign-flip bugs from the
    /// pair_weight scaling.
    #[test]
    fn train_mlp_pwrc_converges() {
        let n_features = 16;
        let n_train = 300;
        let mut rng = SplitMix64::new(7);
        let true_w: Vec<f64> = (0..n_features)
            .map(|i| (i as f64 - 8.0) * 0.3 + rng.next_normal() * 0.1)
            .collect();
        let mut features_owned: Vec<Vec<f64>> = Vec::with_capacity(n_train);
        let mut targets: Vec<f64> = Vec::with_capacity(n_train);
        for _ in 0..n_train {
            let x: Vec<f64> = (0..n_features).map(|_| rng.next_normal()).collect();
            let mut y: f64 = x.iter().zip(true_w.iter()).map(|(a, b)| a * b).sum();
            y += 0.1 * x[0] * x[0];
            y += rng.next_normal() * 0.05;
            features_owned.push(x);
            targets.push(y);
        }
        // Rescale targets to [0, 100] so PWRC band weights interpret
        // them as score_zensim units.
        let t_min = targets.iter().copied().fold(f64::INFINITY, f64::min);
        let t_max = targets.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let scaled: Vec<f64> = targets
            .iter()
            .map(|&t| 100.0 * (t - t_min) / (t_max - t_min).max(1e-12))
            .collect();
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();

        let group = TrainingGroup {
            name: "synth".to_string(),
            human_scores: &scaled,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };

        let hyper = MlpHyperparams {
            n_hidden: 8,
            n_epochs: 80,
            pairs_per_epoch: 1500,
            initial_lr: 0.005,
            log_every: 100,
            early_stop_patience: 0,
            validation_policy: ValidationPolicy::Mean,
            pwrc_pair_weight: true,
            pwrc_sensory_threshold: 5.0,
            pwrc_band_weights: None, // closed-form Wu 2018 default
            ..Default::default()
        };
        let mut log = Vec::new();
        let bytes = train_mlp(&mut [group], n_features, &hyper, &mut log);

        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        let model = Model::from_bytes(leaked).expect("bake should load");
        let mut predictor = Predictor::new(&model);

        let preds: Vec<f64> = features_owned
            .iter()
            .map(|f| predict_one(&mut predictor, f))
            .collect();
        let neg_preds: Vec<f64> = preds.iter().map(|&p| -p).collect();
        let srocc = spearman_correlation(&scaled, &neg_preds);
        assert!(
            srocc > 0.80,
            "PWRC-on trainer failed to converge on synthetic ranking: SROCC={srocc:.4}"
        );
    }

    // ---- Norm-in-Norm + RankNet hybrid tests (Li 2020) ----

    /// `--norm-in-norm-weight 0.0` (default) MUST produce bit-identical
    /// bake bytes to a `MlpHyperparams::default()` call — the legacy
    /// compatibility guarantee. Same RNG sequence, same Adam cadence,
    /// no NiN code path entered (verified by absence of NiN log line).
    #[test]
    fn train_mlp_nin_disabled_matches_legacy() {
        let n_features = 12;
        let (features_owned, targets) = make_synth_dataset(101, 200, n_features);
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();

        let group_factory = || TrainingGroup {
            name: "synth".to_string(),
            human_scores: &targets,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };

        let base = MlpHyperparams {
            n_hidden: 8,
            n_epochs: 25,
            pairs_per_epoch: 800,
            initial_lr: 0.005,
            seed: 7,
            log_every: 100,
            early_stop_patience: 0,
            validation_policy: ValidationPolicy::Mean,
            ..Default::default()
        };

        // K=1 default (legacy code path; both NiN and minibatch off).
        let mut log_a = Vec::new();
        let bake_default = train_mlp(&mut [group_factory()], n_features, &base, &mut log_a);

        // Explicit norm_in_norm_weight=0.0 — must match.
        let hyper_explicit_off = MlpHyperparams {
            norm_in_norm_weight: 0.0,
            norm_in_norm_p: 1.0,
            norm_in_norm_q: 2.0,
            ..base.clone()
        };
        let mut log_b = Vec::new();
        let bake_explicit_off = train_mlp(
            &mut [group_factory()],
            n_features,
            &hyper_explicit_off,
            &mut log_b,
        );
        assert_eq!(
            bake_default, bake_explicit_off,
            "norm_in_norm_weight=0.0 (default) produced different bake than \
             explicit 0.0 — the no-op / bit-identical guarantee is broken"
        );
        assert!(
            !log_a.iter().any(|l| l.contains("Norm-in-Norm")),
            "NiN log line should not appear when weight=0.0"
        );

        // K=64 minibatch + parallel_batch + NiN-off must match the
        // pre-NiN K=64 trainer (i.e., the existing T8.2 path).
        let hyper_k64 = MlpHyperparams {
            minibatch_size: 64,
            parallel_batch: true,
            ..base.clone()
        };
        let mut log_c = Vec::new();
        let bake_k64_legacy = train_mlp(&mut [group_factory()], n_features, &hyper_k64, &mut log_c);

        let hyper_k64_nin_off = MlpHyperparams {
            norm_in_norm_weight: 0.0,
            ..hyper_k64.clone()
        };
        let mut log_d = Vec::new();
        let bake_k64_nin_off = train_mlp(
            &mut [group_factory()],
            n_features,
            &hyper_k64_nin_off,
            &mut log_d,
        );
        assert_eq!(
            bake_k64_legacy, bake_k64_nin_off,
            "K=64 with norm_in_norm_weight=0.0 produced different bake than \
             K=64 alone — NiN-off path must route through the legacy parallel \
             code path, not the NiN path"
        );
    }

    /// With `β=0.1, p=1, q=2`, the hybrid trainer MUST reach comparable
    /// SROCC to RankNet-alone on a synthetic ranking problem. Per Li
    /// 2020 Table 2 the hybrid wins by ~0.01 SROCC on real corpora; on
    /// synthetic toys the two should be within noise (we accept ≥ 0.80
    /// SROCC and within 0.10 of pure-RankNet — generous because the
    /// NiN path uses a different forward order / Adam cadence).
    #[test]
    fn train_mlp_norm_in_norm_converges() {
        let n_features = 16;
        let n_train = 400;
        let (features_owned, targets) = make_synth_dataset(303, n_train, n_features);
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();

        let group_factory = || TrainingGroup {
            name: "synth".to_string(),
            human_scores: &targets,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };

        let eval_srocc = |bake: &[u8]| -> f64 {
            let leaked: &'static [u8] = Box::leak(bake.to_vec().into_boxed_slice());
            let model = Model::from_bytes(leaked).expect("bake should load");
            let mut predictor = Predictor::new(&model);
            let preds: Vec<f64> = features_owned
                .iter()
                .map(|f| predict_one(&mut predictor, f))
                .collect();
            let neg_preds: Vec<f64> = preds.iter().map(|&p| -p).collect();
            spearman_correlation(&targets, &neg_preds)
        };

        let base = MlpHyperparams {
            n_hidden: 12,
            n_epochs: 80,
            pairs_per_epoch: 2000,
            initial_lr: 0.005,
            seed: 7,
            log_every: 100,
            early_stop_patience: 0,
            validation_policy: ValidationPolicy::Mean,
            minibatch_size: 32,
            parallel_batch: true,
            ..Default::default()
        };

        // RankNet-only baseline at the same K.
        let mut log_rn = Vec::new();
        let bake_rn = train_mlp(&mut [group_factory()], n_features, &base, &mut log_rn);
        let srocc_rn = eval_srocc(&bake_rn);

        // Hybrid β=0.1, p=1, q=2 (paper-recommended).
        let hybrid = MlpHyperparams {
            norm_in_norm_weight: 0.1,
            norm_in_norm_p: 1.0,
            norm_in_norm_q: 2.0,
            ..base.clone()
        };
        let mut log_h = Vec::new();
        let bake_h = train_mlp(&mut [group_factory()], n_features, &hybrid, &mut log_h);
        let srocc_h = eval_srocc(&bake_h);

        // Both must learn the ranking; hybrid is within 0.1 of RN.
        assert!(
            srocc_rn > 0.80,
            "RankNet baseline failed to recover synthetic ranking: SROCC={srocc_rn:.4}"
        );
        assert!(
            srocc_h > 0.80,
            "NiN+RankNet hybrid failed to recover synthetic ranking: SROCC={srocc_h:.4}"
        );
        let delta = (srocc_rn - srocc_h).abs();
        assert!(
            delta < 0.10,
            "Hybrid SROCC {srocc_h:.4} differs from RN SROCC {srocc_rn:.4} by \
             {delta:.4} — expected within 0.10 on this synthetic"
        );
        // Hybrid log MUST mention NiN activation (sanity check that
        // the path was actually taken).
        assert!(
            log_h.iter().any(|l| l.contains("Norm-in-Norm")),
            "Hybrid log missing NiN activation line — path may not have been entered"
        );
    }

    /// With β=0.1, the trained MLP's output range should be closer to
    /// the (negated) MOS range than pure RankNet's. RankNet is rank-
    /// only and produces unbounded outputs; NiN normalizes to the
    /// label scale via batch statistics. This is the calibration win
    /// the paper claims at the trainer level (not just at eval-time
    /// post-hoc affine).
    ///
    /// **Test design**: train both at K=32 on a 200-pair synthetic
    /// where MOS ∈ [0, 100]. RankNet should produce wider-range
    /// predictions (output range many σs wide); hybrid should produce
    /// a tighter range that — after sign-flip — lands closer to the
    /// MOS distribution's [0, 100] envelope.
    #[test]
    fn train_mlp_norm_in_norm_helps_calibration() {
        let n_features = 8;
        let mut rng = SplitMix64::new(151);
        let n = 200usize;
        let mut targets: Vec<f64> = Vec::with_capacity(n);
        let mut features_owned: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let s = 100.0 * (i as f64) / (n as f64 - 1.0); // 0..100
            targets.push(s);
            let mut x: Vec<f64> = (0..n_features).map(|_| rng.next_normal()).collect();
            x[0] = s / 100.0 + rng.next_normal() * 0.1;
            features_owned.push(x);
        }
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
        let group_factory = || TrainingGroup {
            name: "calib".to_string(),
            human_scores: &targets,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };

        let base = MlpHyperparams {
            n_hidden: 8,
            n_epochs: 60,
            pairs_per_epoch: 1500,
            initial_lr: 0.005,
            seed: 13,
            log_every: 100,
            early_stop_patience: 0,
            validation_policy: ValidationPolicy::Mean,
            minibatch_size: 32,
            parallel_batch: true,
            ..Default::default()
        };
        let hybrid = MlpHyperparams {
            norm_in_norm_weight: 0.1,
            norm_in_norm_p: 1.0,
            norm_in_norm_q: 2.0,
            ..base.clone()
        };

        let mut log_rn = Vec::new();
        let bake_rn = train_mlp(&mut [group_factory()], n_features, &base, &mut log_rn);
        let mut log_h = Vec::new();
        let bake_h = train_mlp(&mut [group_factory()], n_features, &hybrid, &mut log_h);

        let preds = |bake: &[u8]| -> Vec<f64> {
            let leaked: &'static [u8] = Box::leak(bake.to_vec().into_boxed_slice());
            let model = Model::from_bytes(leaked).expect("bake should load");
            let mut predictor = Predictor::new(&model);
            // Trainer output is distance-like (LOWER = higher MOS), so
            // negate to put predictions on the MOS axis for range
            // comparison. The MOS range here is [0, 100].
            features_owned
                .iter()
                .map(|f| -predict_one(&mut predictor, f))
                .collect()
        };
        let p_rn = preds(&bake_rn);
        let p_h = preds(&bake_h);
        // Compute range as (max - min) of each prediction set.
        let range = |p: &[f64]| -> f64 {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for &v in p {
                lo = lo.min(v);
                hi = hi.max(v);
            }
            hi - lo
        };
        let r_rn = range(&p_rn);
        let r_h = range(&p_h);
        // The hybrid's range may be smaller (calibration to label scale)
        // or larger (the NiN gradient pulls toward the MOS range scale,
        // which is ~100 units, while pure RankNet typically produces
        // smaller-range outputs since it's optimizing pairwise margins).
        // What we ACTUALLY want to assert is that the hybrid output
        // range is closer to the MOS range (= 100) than RankNet's.
        let mos_range = 100.0f64;
        let dist_rn = (r_rn - mos_range).abs();
        let dist_h = (r_h - mos_range).abs();
        assert!(
            dist_h < dist_rn,
            "Hybrid range {r_h:.2} should be closer to MOS range {mos_range} than \
             RankNet's {r_rn:.2}; got dist_h={dist_h:.2}, dist_rn={dist_rn:.2}"
        );
        // Also assert the hybrid output doesn't blow up to insane
        // ranges (calibration sanity bound from task spec).
        assert!(
            r_h < 500.0,
            "Hybrid output range {r_h:.2} exceeds [0, 500] — gradient may be \
             miscalibrated or NiN sign convention wrong"
        );
    }

    /// `--norm-in-norm-weight > 0` with K < 16 MUST panic with a clear
    /// message at trainer entry — the batch statistics required for
    /// NiN are unstable below ~16 samples.
    #[test]
    #[should_panic(expected = "requires --minibatch-size >= 16")]
    fn train_mlp_norm_in_norm_errors_on_small_k() {
        let n_features = 4;
        let (features_owned, targets) = make_synth_dataset(401, 50, n_features);
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
        let group = TrainingGroup {
            name: "tiny".to_string(),
            human_scores: &targets,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };
        let hyper = MlpHyperparams {
            n_hidden: 4,
            n_epochs: 5,
            pairs_per_epoch: 100,
            minibatch_size: 8, // < 16 → must error
            parallel_batch: true,
            norm_in_norm_weight: 0.1,
            norm_in_norm_p: 1.0,
            norm_in_norm_q: 2.0,
            log_every: 100,
            early_stop_patience: 0,
            ..Default::default()
        };
        let mut log = Vec::new();
        let _ = train_mlp(&mut [group], n_features, &hyper, &mut log);
    }

    /// EX-2 prod wire-in smoke test: pool_head=true on a synthetic
    /// ranking task converges to a usable Spearman, and the bake
    /// carries the `zentrain.pool_head_reducer` metadata key (runtime
    /// dispatch identifier).
    #[test]
    fn train_mlp_pool_head_recovers_synthetic_ranking() {
        let n_features = 8;
        let (features_owned, targets) = make_synth_dataset(11, 200, n_features);
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
        let group = TrainingGroup {
            name: "synth".to_string(),
            human_scores: &targets,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };
        let hyper = MlpHyperparams {
            n_hidden: 16,
            n_epochs: 20,
            pairs_per_epoch: 1000,
            initial_lr: 5e-3,
            seed: 11,
            log_every: 100,
            early_stop_patience: 0,
            l2_lambda: 0.0,
            minibatch_size: 1,
            pool_head: true,
            ..Default::default()
        };
        let mut log = Vec::new();
        let bytes = train_mlp(&mut [group], n_features, &hyper, &mut log);
        // Smoke: pool_head metadata present.
        let needle = b"zentrain.pool_head_reducer";
        assert!(
            bytes.windows(needle.len()).any(|w| w == needle),
            "expected pool_head metadata in bake bytes"
        );
        // Version must be v3.
        assert_eq!(&bytes[..4], b"ZNPR");
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        assert_eq!(version, 3, "pool_head bake must be ZNPR v3");
    }

    /// EX-2 prod wire-in: pool_head + minibatch_size=8 still bakes
    /// correctly (mini-batch sequential path exercised).
    #[test]
    fn train_mlp_pool_head_with_minibatch_8() {
        let n_features = 8;
        let (features_owned, targets) = make_synth_dataset(12, 200, n_features);
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
        let group = TrainingGroup {
            name: "synth".to_string(),
            human_scores: &targets,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };
        let hyper = MlpHyperparams {
            n_hidden: 16,
            n_epochs: 5,
            pairs_per_epoch: 400,
            initial_lr: 5e-3,
            seed: 12,
            log_every: 100,
            early_stop_patience: 0,
            l2_lambda: 1e-5,
            minibatch_size: 8,
            pool_head: true,
            ..Default::default()
        };
        let mut log = Vec::new();
        let bytes = train_mlp(&mut [group], n_features, &hyper, &mut log);
        assert_eq!(&bytes[..4], b"ZNPR");
    }

    /// EX-2 prod wire-in: pool_head + NiN composes (Li 2020 hybrid loss
    /// scattered through pool-head chain rule).
    #[test]
    fn train_mlp_pool_head_with_nin_composes() {
        let n_features = 8;
        let (features_owned, targets) = make_synth_dataset(13, 200, n_features);
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
        let group = TrainingGroup {
            name: "synth".to_string(),
            human_scores: &targets,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };
        let hyper = MlpHyperparams {
            n_hidden: 16,
            n_epochs: 5,
            pairs_per_epoch: 320,
            initial_lr: 5e-3,
            seed: 13,
            minibatch_size: 16,
            norm_in_norm_weight: 0.1,
            norm_in_norm_p: 1.0,
            norm_in_norm_q: 2.0,
            pool_head: true,
            log_every: 100,
            early_stop_patience: 0,
            ..Default::default()
        };
        let mut log = Vec::new();
        let bytes = train_mlp(&mut [group], n_features, &hyper, &mut log);
        // Smoke: NiN-composed pool-head bake still has the pool_head metadata.
        let needle = b"zentrain.pool_head_reducer";
        assert!(
            bytes.windows(needle.len()).any(|w| w == needle),
            "expected pool_head metadata in NiN-composed bake bytes"
        );
        // Version must be v3.
        assert_eq!(&bytes[..4], b"ZNPR");
    }

    /// EX-2 prod wire-in: pool_head + NiN with K < 16 must error
    /// out (NiN batch statistics unstable below 16 predictions).
    #[test]
    #[should_panic(expected = "requires --minibatch-size >= 16")]
    fn train_mlp_pool_head_with_nin_small_k_panics() {
        let n_features = 4;
        let (features_owned, targets) = make_synth_dataset(14, 50, n_features);
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
        let group = TrainingGroup {
            name: "tiny".to_string(),
            human_scores: &targets,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };
        let hyper = MlpHyperparams {
            n_hidden: 16,
            n_epochs: 5,
            pairs_per_epoch: 100,
            minibatch_size: 8, // < 16 — must error
            norm_in_norm_weight: 0.1,
            pool_head: true,
            log_every: 100,
            early_stop_patience: 0,
            ..Default::default()
        };
        let mut log = Vec::new();
        let _ = train_mlp(&mut [group], n_features, &hyper, &mut log);
    }

    #[test]
    #[should_panic(expected = "silently ignored")]
    fn train_mlp_strategy_triplet_without_alpha_head_panics() {
        // Regression for the 2026-07-16 silent no-op: a triplet pool loaded on
        // the plain (non-per-sample-alpha) path was thrown away, producing a
        // bake byte-identical to a no-triplet run. The guard must fail loud.
        let n_features = 4;
        let (features_owned, targets) = make_synth_dataset(14, 50, n_features);
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
        let group = TrainingGroup {
            name: "tiny".to_string(),
            human_scores: &targets,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };
        let mut pool = TripletPool::default();
        for f in &features_owned {
            pool.features.push(f.clone());
        }
        pool.responses.push((0, 1, 0));
        pool.responses.push((1, 0, 1));
        let hyper = MlpHyperparams {
            n_hidden: 16,
            n_epochs: 5,
            pairs_per_epoch: 100,
            triplet_weight: 0.5,          // triplet requested...
            per_sample_alpha_head: false, // ...on the plain path -> must panic
            log_every: 100,
            early_stop_patience: 0,
            ..Default::default()
        };
        let mut log = Vec::new();
        let _ = train_mlp_strategy(
            &mut [group],
            n_features,
            &hyper,
            &mut log,
            None,
            None,
            None,
            None,
            None,
            Some(&pool),
        );
    }

    // ---- KONJND-AGGREGATION-HEAD smoke tests (task #4, 2026-05-24) ----

    const KAH_N_REFS: usize = 30;
    const KAH_ROWS_PER_REF: usize = 8;
    const KAH_N_FEATURES: usize = 4;
    const KAH_N_PRIMARY_PAIRS: usize = 50;

    /// `(features, ref_ranges, ref_pjnd_target, primary_scores)` fixture pool.
    type KahPool = (Vec<Vec<f64>>, Vec<(usize, usize)>, Vec<f64>, Vec<f64>);

    fn kah_build_pool() -> KahPool {
        let mut features: Vec<Vec<f64>> = Vec::with_capacity(KAH_N_REFS * KAH_ROWS_PER_REF);
        let mut ref_ranges: Vec<(usize, usize)> = Vec::with_capacity(KAH_N_REFS);
        let mut ref_pjnd_target: Vec<f64> = Vec::with_capacity(KAH_N_REFS);
        let mut cursor = 0usize;
        for r in 0..KAH_N_REFS {
            let f1 = -1.0 + 2.0 * (r as f64) / ((KAH_N_REFS - 1) as f64);
            let target = 40.0 + 20.0 * f1; // [20, 60]
            ref_pjnd_target.push(target);
            let start = cursor;
            for k in 0..KAH_ROWS_PER_REF {
                let f0 = (k as f64) / ((KAH_ROWS_PER_REF - 1) as f64);
                let f2 = 0.1 * ((r * 7 + k) as f64).sin();
                let f3 = 0.1 * ((r * 11 + k) as f64).cos();
                features.push(vec![f0, f1, f2, f3]);
                cursor += 1;
            }
            ref_ranges.push((start, KAH_ROWS_PER_REF));
        }
        let ref_weight = vec![1.0_f64; KAH_N_REFS];
        (features, ref_ranges, ref_pjnd_target, ref_weight)
    }

    /// Score the bake's raw output, then apply the same tanh
    /// output-head pin the trainer used (`y_score = 100 / (1 +
    /// exp(-y_pre / scale))`). `predictor.predict()` returns the raw
    /// network output — the per-sample-α head's `y_pre` scalar
    /// (zenpredict doesn't dispatch on `tanh_output_head` metadata;
    /// that lives in `zensim::metric::forward_one_bake`). For the
    /// test, we replicate the pin here to compare apples-to-apples
    /// with the trainer's loss.
    /// H-TRAJ dump feature, PLAIN-MLP lane (the W10L9 recipe's lane —
    /// caught missing on first deployment 2026-08-28: the recipe logs the
    /// plain-loop format, and the alpha-lane-only hook emitted nothing).
    #[test]
    fn htraj_checkpoint_dumps_plain_lane() {
        let dir = std::env::temp_dir().join(format!("htraj_plain_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut primary_features: Vec<Vec<f64>> = Vec::with_capacity(KAH_N_PRIMARY_PAIRS);
        let mut primary_scores: Vec<f64> = Vec::with_capacity(KAH_N_PRIMARY_PAIRS);
        for i in 0..KAH_N_PRIMARY_PAIRS {
            let frac = (i as f64) / ((KAH_N_PRIMARY_PAIRS - 1).max(1) as f64);
            primary_features.push(vec![frac, -1.0 + 2.0 * frac, 0.0, 0.0]);
            primary_scores.push(0.5 + 0.5 * (-1.0 + 2.0 * frac));
        }
        let primary_refs: Vec<&[f64]> = primary_features.iter().map(|r| r.as_slice()).collect();
        let mut groups = [TrainingGroup {
            name: "synthetic_primary".to_string(),
            human_scores: &primary_scores,
            features: FeatureRows::Borrowed(&primary_refs),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        }];
        let hp = MlpHyperparams {
            n_hidden: 8,
            n_epochs: 30,
            pairs_per_epoch: 200,
            minibatch_size: 1,
            initial_lr: 1e-2,
            l2_lambda: 1e-6,
            leaky_alpha: 0.01,
            early_stop_patience: 0,
            ranknet_weight: 1.0,
            seed: 42,
            dump_checkpoints_every: 10,
            dump_checkpoints_dir: Some(dir.clone()),
            ..Default::default()
        };
        let mut log: Vec<String> = Vec::new();
        let _bake = train_mlp_with_tv_anchored_equiv_pjnd(
            &mut groups,
            KAH_N_FEATURES,
            &hp,
            &mut log,
            None,
            None,
            None,
            None,
            None,
        );
        let dumps: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("ckpt_epoch")
            })
            .collect();
        assert!(
            dumps.len() >= 3,
            "expected >=3 plain-lane checkpoint dumps, got {dumps:?}"
        );
        for d in &dumps {
            let bytes = std::fs::read(d).unwrap();
            assert_eq!(bytes[4], 3, "dump not ZNPR v3: {}", d.display());
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// H-TRAJ dump feature (balance campaign 2026-08-28): a tiny
    /// per-sample-α train with --dump-checkpoints-every must emit v3 bakes
    /// at the val-aligned epochs, byte-parseable (header byte 4 == 3).
    #[test]
    fn htraj_checkpoint_dumps_emit_v3_bakes() {
        let dir = std::env::temp_dir().join(format!("htraj_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (pool_rows, ref_ranges, ref_pjnd_target, ref_weight) = kah_build_pool();
        let pool_row_refs: Vec<&[f64]> = pool_rows.iter().map(|r| r.as_slice()).collect();
        let konjnd_agg = KonjndAggregationPool {
            name: "synthetic_konjnd".to_string(),
            features: pool_row_refs.as_slice(),
            ref_ranges: ref_ranges.as_slice(),
            ref_pjnd_target: ref_pjnd_target.as_slice(),
            ref_weight: ref_weight.as_slice(),
        };
        let mut primary_features: Vec<Vec<f64>> = Vec::with_capacity(KAH_N_PRIMARY_PAIRS);
        let mut primary_scores: Vec<f64> = Vec::with_capacity(KAH_N_PRIMARY_PAIRS);
        for i in 0..KAH_N_PRIMARY_PAIRS {
            let frac = (i as f64) / ((KAH_N_PRIMARY_PAIRS - 1).max(1) as f64);
            primary_features.push(vec![frac, -1.0 + 2.0 * frac, 0.0, 0.0]);
            primary_scores.push(0.5 + 0.5 * (-1.0 + 2.0 * frac));
        }
        let primary_refs: Vec<&[f64]> = primary_features.iter().map(|r| r.as_slice()).collect();
        let mut groups = [TrainingGroup {
            name: "synthetic_primary".to_string(),
            human_scores: &primary_scores,
            features: FeatureRows::Borrowed(&primary_refs),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 0.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        }];
        let mut hp = MlpHyperparams {
            n_hidden: 16,
            n_hidden_layers: 1,
            n_epochs: 30,
            pairs_per_epoch: 200,
            minibatch_size: 1,
            initial_lr: 1e-2,
            l2_lambda: 1e-6,
            leaky_alpha: 0.01,
            early_stop_patience: 0,
            per_sample_alpha_head: true,
            tanh_output_head_scale: 20.0,
            ranknet_weight: 1.0,
            konjnd_aggregation_weight: 1.0,
            konjnd_aggregation_step_p: 1.0,
            konjnd_aggregation_samples_per_ref: 8,
            konjnd_aggregation_refs_per_step: 8,
            seed: 42,
            dump_checkpoints_every: 10,
            dump_checkpoints_dir: Some(dir.clone()),
            ..Default::default()
        };
        hp.validation_policy = ValidationPolicy::Min;
        let mut log: Vec<String> = Vec::new();
        let _bake = train_mlp_with_tv_anchored_equiv_pjnd(
            &mut groups,
            KAH_N_FEATURES,
            &hp,
            &mut log,
            None,
            None,
            None,
            None,
            Some(&konjnd_agg),
        );
        let mut dumps: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("ckpt_epoch")
            })
            .collect();
        dumps.sort();
        assert!(
            dumps.len() >= 3,
            "expected >=3 checkpoint dumps (epochs 0,10,20), got {dumps:?}"
        );
        for d in &dumps {
            let bytes = std::fs::read(d).unwrap();
            assert!(
                bytes.len() > 64,
                "dump too small: {} ({} B)",
                d.display(),
                bytes.len()
            );
            assert_eq!(bytes[4], 3, "dump not ZNPR v3: {}", d.display());
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    fn kah_aggregation_mse_under_bake(
        bake_bytes: &[u8],
        features: &[Vec<f64>],
        ref_ranges: &[(usize, usize)],
        ref_pjnd_target: &[f64],
        tanh_scale: f64,
    ) -> f64 {
        let model = zenpredict::Model::from_bytes(bake_bytes).expect("bake parse");
        let mut predictor = zenpredict::Predictor::new(&model);
        let mut sum_sq_err = 0.0f64;
        let mut n_ok = 0usize;
        for (&(start, n_rows), &target) in ref_ranges.iter().zip(ref_pjnd_target.iter()) {
            let mut sum_y = 0.0f64;
            let mut s_actual = 0usize;
            for i in start..(start + n_rows) {
                let row: Vec<f32> = features[i].iter().map(|&v| v as f32).collect();
                match predictor.predict(&row) {
                    Ok(y) => {
                        let y_pre = y[0] as f64;
                        // Apply tanh pin to match the trainer's
                        // forward path: y_score = 100 / (1 + exp(-y_pre / scale)).
                        let y_score = if tanh_scale > 0.0 {
                            let z = -y_pre / tanh_scale;
                            // Clamp z for numerical stability.
                            let z = z.clamp(-50.0, 50.0);
                            100.0 / (1.0 + z.exp())
                        } else {
                            y_pre
                        };
                        sum_y += y_score;
                        s_actual += 1;
                    }
                    Err(_) => continue,
                }
            }
            if s_actual > 0 {
                let agg = sum_y / (s_actual as f64);
                let err = agg - target;
                sum_sq_err += err * err;
                n_ok += 1;
            }
        }
        sum_sq_err / (n_ok.max(1) as f64)
    }

    /// Aggregation step actually backprops a non-trivial gradient:
    /// 30 epochs of training on a synthetic per-ref-target problem
    /// should yield a bake whose per-ref aggregation MSE is finite +
    /// inside the loose [0, 2500] sanity ceiling. (Perfect-init worst
    /// case is ~2500 on a 0..100 dial vs [20, 60] target.)
    /// Body shared by the 1-layer and 2-layer aggregation-step tests.
    /// `n_hidden_layers` selects the architecture; both must train
    /// without NaN and produce a non-empty bake whose per-ref
    /// aggregation MSE stays inside the loose sanity ceiling.
    fn kah_run_train_and_check(n_hidden_layers: usize) {
        let (pool_rows, ref_ranges, ref_pjnd_target, ref_weight) = kah_build_pool();
        let pool_row_refs: Vec<&[f64]> = pool_rows.iter().map(|r| r.as_slice()).collect();
        let konjnd_agg = KonjndAggregationPool {
            name: "synthetic_konjnd".to_string(),
            features: pool_row_refs.as_slice(),
            ref_ranges: ref_ranges.as_slice(),
            ref_pjnd_target: ref_pjnd_target.as_slice(),
            ref_weight: ref_weight.as_slice(),
        };
        let mut primary_features: Vec<Vec<f64>> = Vec::with_capacity(KAH_N_PRIMARY_PAIRS);
        let mut primary_scores: Vec<f64> = Vec::with_capacity(KAH_N_PRIMARY_PAIRS);
        for i in 0..KAH_N_PRIMARY_PAIRS {
            let frac = (i as f64) / ((KAH_N_PRIMARY_PAIRS - 1).max(1) as f64);
            primary_features.push(vec![frac, -1.0 + 2.0 * frac, 0.0, 0.0]);
            primary_scores.push(0.5 + 0.5 * (-1.0 + 2.0 * frac));
        }
        let primary_refs: Vec<&[f64]> = primary_features.iter().map(|r| r.as_slice()).collect();
        let mut groups = [TrainingGroup {
            name: "synthetic_primary".to_string(),
            human_scores: &primary_scores,
            features: FeatureRows::Borrowed(&primary_refs),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 0.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        }];
        let mut hp = MlpHyperparams {
            n_hidden: 16,
            n_hidden_layers,
            n_epochs: 30,
            pairs_per_epoch: 200,
            minibatch_size: 1,
            initial_lr: 1e-2,
            l2_lambda: 1e-6,
            leaky_alpha: 0.01,
            early_stop_patience: 0,
            per_sample_alpha_head: true,
            tanh_output_head_scale: 20.0,
            ranknet_weight: 1.0,
            mse_weight: 0.0,
            monotonicity_reg: 0.0,
            norm_in_norm_weight: 0.0,
            konjnd_aggregation_weight: 1.0,
            konjnd_aggregation_step_p: 1.0,
            konjnd_aggregation_samples_per_ref: 8,
            konjnd_aggregation_refs_per_step: 8,
            seed: 42,
            ..Default::default()
        };
        hp.validation_policy = ValidationPolicy::Min;
        let mut log: Vec<String> = Vec::new();
        let bake_bytes = train_mlp_with_tv_anchored_equiv_pjnd(
            &mut groups,
            KAH_N_FEATURES,
            &hp,
            &mut log,
            None,
            None,
            None,
            None,
            Some(&konjnd_agg),
        );
        assert!(
            !bake_bytes.is_empty(),
            "bake bytes empty after training ({n_hidden_layers}-layer)"
        );
        let mse = kah_aggregation_mse_under_bake(
            &bake_bytes,
            &pool_rows,
            &ref_ranges,
            &ref_pjnd_target,
            hp.tanh_output_head_scale,
        );
        assert!(
            mse.is_finite(),
            "aggregation MSE not finite after training ({n_hidden_layers}-layer): {mse}"
        );
        // With perfect failure (every ref pinned at score=50, all
        // targets ∈ [20, 60]): worst per-ref err ≈ 30 → MSE ≈ 400.
        // Random init: ~1000-3000. Successful training brings the
        // mean error well below the worst case. Gate is loose
        // because the synthetic problem is small + the test runs
        // briefly; any working gradient flow easily passes.
        assert!(
            mse < 1500.0,
            "aggregation MSE {mse:.2} suspiciously high ({n_hidden_layers}-layer) — gradient may be broken"
        );
        let any_enabled_log = log
            .iter()
            .any(|l| l.contains("konjnd-aggregation: ENABLED"));
        assert!(
            any_enabled_log,
            "expected 'konjnd-aggregation: ENABLED' log ({n_hidden_layers}-layer); got logs: {log:?}"
        );
    }

    #[test]
    fn konjnd_aggregation_step_runs_and_backprops() {
        // 1-layer (legacy path through arch_forward/arch_backward
        // with use_2layer=false).
        kah_run_train_and_check(1);
    }

    /// 2-layer is the production architecture (shipped V39 is
    /// 2-layer). This guards the G5 lever (konjnd-aggregation head)
    /// against the previously-gated 2-layer panic — it must now
    /// train through arch_forward / arch_backward without NaN.
    #[test]
    fn konjnd_aggregation_step_runs_and_backprops_2layer() {
        kah_run_train_and_check(2);
    }

    /// With weight=0 the pool is supplied but the step never fires.
    /// Training should run + bake should be valid + log should say so.
    #[test]
    fn konjnd_aggregation_disabled_means_pool_ignored() {
        let (pool_rows, ref_ranges, ref_pjnd_target, ref_weight) = kah_build_pool();
        let pool_row_refs: Vec<&[f64]> = pool_rows.iter().map(|r| r.as_slice()).collect();
        let konjnd_agg = KonjndAggregationPool {
            name: "synthetic_konjnd_disabled".to_string(),
            features: pool_row_refs.as_slice(),
            ref_ranges: ref_ranges.as_slice(),
            ref_pjnd_target: ref_pjnd_target.as_slice(),
            ref_weight: ref_weight.as_slice(),
        };
        let primary_features: Vec<Vec<f64>> = (0..KAH_N_PRIMARY_PAIRS)
            .map(|i| {
                let f = (i as f64) / ((KAH_N_PRIMARY_PAIRS - 1).max(1) as f64);
                vec![f, -1.0 + 2.0 * f, 0.0, 0.0]
            })
            .collect();
        let primary_scores: Vec<f64> = primary_features.iter().map(|r| 0.5 + 0.5 * r[1]).collect();
        let primary_refs: Vec<&[f64]> = primary_features.iter().map(|r| r.as_slice()).collect();
        let mut groups = [TrainingGroup {
            name: "primary_disabled".to_string(),
            human_scores: &primary_scores,
            features: FeatureRows::Borrowed(&primary_refs),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 0.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        }];
        let hp = MlpHyperparams {
            n_hidden: 8,
            n_epochs: 3,
            pairs_per_epoch: 50,
            minibatch_size: 1,
            per_sample_alpha_head: true,
            tanh_output_head_scale: 20.0,
            ranknet_weight: 1.0,
            norm_in_norm_weight: 0.0,
            konjnd_aggregation_weight: 0.0,
            early_stop_patience: 0,
            seed: 7,
            ..Default::default()
        };
        let mut log: Vec<String> = Vec::new();
        let bake = train_mlp_with_tv_anchored_equiv_pjnd(
            &mut groups,
            KAH_N_FEATURES,
            &hp,
            &mut log,
            None,
            None,
            None,
            None,
            Some(&konjnd_agg),
        );
        assert!(!bake.is_empty(), "bake empty when konjnd-agg disabled");
        let any_ignored_log = log
            .iter()
            .any(|line| line.contains("konjnd-aggregation") && line.contains("ignored"));
        assert!(
            any_ignored_log,
            "expected 'konjnd-aggregation ... ignored' log; got logs: {log:?}"
        );
    }

    /// Numerical-gradient check for the 2-layer konjnd-aggregation
    /// path. Builds a single-ref pool of S rows, forwards every row
    /// through `arch_forward` (use_2layer=true), aggregates the pinned
    /// scores into `agg = mean(y)`, and checks the analytic w1
    /// gradient produced by `arch_backward` — accumulated exactly the
    /// way the trainer's aggregation step accumulates it
    /// (`dl_dy = (2w/S)·(agg − t)·dya_dpre` per row) — against a
    /// centered finite difference of the per-ref loss
    /// `L = w·(agg − t)²`. This is the delicate chain: the 2-layer
    /// encoder backward (`w1` through layer-1 → leaky → layer-2 →
    /// heads → pin → aggregate). Matching to ~1e-3 relative confirms
    /// `arch_backward` is correct on the aggregation loss in 2-layer
    /// mode.
    #[test]
    fn konjnd_aggregation_2layer_w1_gradient_matches_finite_difference() {
        let n_features = 4usize;
        let n_hidden1 = 6usize;
        let n_hidden_final = (n_hidden1 / 2).max(8); // matches trainer
        let leaky = 0.01f64;
        let use_2layer = true;
        let use_skip = false;
        let tanh_scale = 20.0f64;
        let weight = 0.7f64; // konjnd_aggregation_weight
        let target = 42.0f64; // per-ref pjnd_target
        let s_rows = 5usize;

        // Deterministic LCG so the test is reproducible without
        // pulling a test-only RNG into the validate crate.
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut nxt = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // map to [-0.5, 0.5)
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };

        let mut w1: Vec<f64> = (0..n_features * n_hidden1).map(|_| nxt()).collect();
        let b1: Vec<f64> = (0..n_hidden1).map(|_| nxt()).collect();
        let w2_enc: Vec<f64> = (0..n_hidden1 * n_hidden_final).map(|_| nxt()).collect();
        let b2_enc: Vec<f64> = (0..n_hidden_final).map(|_| nxt()).collect();
        let w_skip: Vec<f64> = Vec::new();
        let b_skip = 0.0f64;
        let rank_w: Vec<f64> = (0..n_hidden_final).map(|_| nxt()).collect();
        let rank_b = nxt();
        let reducer_w: [f64; 4] = [nxt(), nxt(), nxt(), nxt()];
        let reducer_b = nxt();
        let w_alpha: Vec<f64> = (0..n_hidden_final).map(|_| nxt()).collect();
        let b_alpha = nxt();

        let rows: Vec<Vec<f64>> = (0..s_rows)
            .map(|_| (0..n_features).map(|_| nxt()).collect())
            .collect();

        // tanh pin used by the trainer (tanh_output_head_scale > 0).
        let pin = |y_pre: f64| -> (f64, f64) {
            let xc = (y_pre / tanh_scale).clamp(-30.0, 30.0);
            let s = 1.0 / (1.0 + (-xc).exp());
            let y_score = 100.0 * s;
            let dy = (100.0 / tanh_scale) * s * (1.0 - s);
            (y_score, dy)
        };

        // Per-ref loss as a pure function of w1 (everything else fixed).
        let loss_fn = |w1v: &[f64]| -> f64 {
            let mut sum_y = 0.0f64;
            for row in &rows {
                let fwd = arch_forward(
                    row,
                    w1v,
                    &b1,
                    &w2_enc,
                    &b2_enc,
                    &w_skip,
                    b_skip,
                    &rank_w,
                    rank_b,
                    &reducer_w,
                    reducer_b,
                    &w_alpha,
                    b_alpha,
                    n_features,
                    n_hidden1,
                    n_hidden_final,
                    leaky,
                    use_2layer,
                    use_skip,
                );
                let (y, _) = pin(fwd.y);
                sum_y += y;
            }
            let agg = sum_y / (s_rows as f64);
            let err = agg - target;
            weight * err * err
        };

        // Analytic gradient via arch_backward — mirror the trainer's
        // two-pass aggregation accumulation exactly.
        let n_w1_concat = n_features * n_hidden1 + n_hidden1 * n_hidden_final;
        let mut gw1 = vec![0.0f64; n_w1_concat];
        let mut gb1 = vec![0.0f64; n_hidden1 + n_hidden_final];
        let mut g_rank_w = vec![0.0f64; n_hidden_final];
        let mut g_rank_b = 0.0f64;
        let mut g_red_w: [f64; 4] = [0.0; 4];
        let mut g_red_b = 0.0f64;
        let mut g_w_alpha = vec![0.0f64; n_hidden_final];
        let mut g_b_alpha = 0.0f64;

        // Pass 1: forward all rows, cache (dya_dpre, ArchForward), sum y.
        let mut cache: Vec<(f64, ArchForward)> = Vec::with_capacity(s_rows);
        let mut sum_y = 0.0f64;
        for row in &rows {
            let fwd = arch_forward(
                row,
                &w1,
                &b1,
                &w2_enc,
                &b2_enc,
                &w_skip,
                b_skip,
                &rank_w,
                rank_b,
                &reducer_w,
                reducer_b,
                &w_alpha,
                b_alpha,
                n_features,
                n_hidden1,
                n_hidden_final,
                leaky,
                use_2layer,
                use_skip,
            );
            let (y, dya_dpre) = pin(fwd.y);
            sum_y += y;
            cache.push((dya_dpre, fwd));
        }
        let agg = sum_y / (s_rows as f64);
        let err = agg - target;
        let scale = 2.0 * weight / (s_rows as f64);
        // Pass 2: backprop each row with dl_dy = scale·err·dya_dpre.
        for (i, (dya_dpre, fwd)) in cache.iter().enumerate() {
            let dl_dy = scale * err * dya_dpre;
            arch_backward(
                &rows[i],
                fwd,
                dl_dy,
                &w1,
                &w2_enc,
                &rank_w,
                &reducer_w,
                &w_alpha,
                &mut gw1,
                &mut gb1,
                &mut g_rank_w,
                &mut g_rank_b,
                &mut g_red_w,
                &mut g_red_b,
                &mut g_w_alpha,
                &mut g_b_alpha,
                n_features,
                n_hidden1,
                n_hidden_final,
                leaky,
                use_2layer,
                use_skip,
            );
        }

        // The forward computes in f32 (arch_forward → dot_bias casts f64→f32),
        // so a central difference is floor-limited: rounding noise in (f₊−f₋)
        // is ~ε_f32·|L| while the signal is 2·ε·|∂L|, so the relative error of
        // the numeric derivative is ~ε_f32/(2·ε·|∂L|/|L|). At ε=1e-6 that floor
        // is O(1) — the original "~2× gradient bug" was entirely this f32 floor,
        // NOT a gradient bug (confirmed by the train-core backprop_heads FD test
        // which isolates the encoder gradient and passes cleanly). ε=1e-2 pushes
        // the floor to ~1e-4 while O(ε²) truncation stays small; 2e-3 gates both.
        let eps = 1e-2;
        let n_w1_first = n_features * n_hidden1;

        // FD-check the SECOND-layer w2_enc gradient too (analytical =
        // gw1[n_w1_first + idx], concat = [w1|w2_enc|w_skip]) — strengthens the
        // test to both layers of the 2-layer chain.
        {
            let loss_fn_w2 = |w2v: &[f64]| -> f64 {
                let mut sum_y = 0.0f64;
                for row in &rows {
                    let fwd = arch_forward(
                        row,
                        &w1,
                        &b1,
                        w2v,
                        &b2_enc,
                        &w_skip,
                        b_skip,
                        &rank_w,
                        rank_b,
                        &reducer_w,
                        reducer_b,
                        &w_alpha,
                        b_alpha,
                        n_features,
                        n_hidden1,
                        n_hidden_final,
                        leaky,
                        use_2layer,
                        use_skip,
                    );
                    let (y, _) = pin(fwd.y);
                    sum_y += y;
                }
                let e = sum_y / (s_rows as f64) - target;
                weight * e * e
            };
            let mut w2m = w2_enc.clone();
            for idx2 in [0usize, (n_hidden1 * n_hidden_final) / 2] {
                let orig = w2m[idx2];
                w2m[idx2] = orig + eps;
                let fp = loss_fn_w2(&w2m);
                w2m[idx2] = orig - eps;
                let fm = loss_fn_w2(&w2m);
                w2m[idx2] = orig;
                let numerical = (fp - fm) / (2.0 * eps);
                let analytical = gw1[n_w1_first + idx2];
                // Combined atol+rtol (numpy/jax/pytorch gradcheck form): a pure
                // relative gate is unbounded as the true gradient → 0, and the
                // f32-forward FD floor (~1e-5 abs at ε=1e-2) dominates there.
                let abs_err = (numerical - analytical).abs();
                let tol = 2e-5 + 2e-3 * numerical.abs().max(analytical.abs());
                assert!(
                    abs_err < tol,
                    "2-layer konjnd-agg gw2_enc[{idx2}] numerical={numerical:.8} \
                     analytical={analytical:.8} abs_err={abs_err:.2e} tol={tol:.2e}"
                );
            }
        }

        // FD-check the first-layer w1 entries (the deepest part of the
        // 2-layer chain). Check several spread across the matrix.
        for idx in (0..n_w1_first).step_by((n_w1_first / 6).max(1)) {
            let orig = w1[idx];
            w1[idx] = orig + eps;
            let fp = loss_fn(&w1);
            w1[idx] = orig - eps;
            let fm = loss_fn(&w1);
            w1[idx] = orig;
            let numerical = (fp - fm) / (2.0 * eps);
            let analytical = gw1[idx];
            // Combined atol+rtol — many w1 entries have near-zero true gradient
            // (e.g. gw1[4]≈-9e-4) where a pure relative gate is meaningless and
            // the f32-forward FD floor (~1e-5 abs at ε=1e-2) sets the scale.
            let abs_err = (numerical - analytical).abs();
            let tol = 2e-5 + 2e-3 * numerical.abs().max(analytical.abs());
            assert!(
                abs_err < tol,
                "2-layer konjnd-agg gw1[{idx}] numerical={numerical:.8} \
                 analytical={analytical:.8} abs_err={abs_err:.2e} tol={tol:.2e}"
            );
        }
    }

    // ===================================================================
    // Per-sample-alpha-head dedicated tests (previously zero coverage).
    // ===================================================================

    #[test]
    fn psah_recovers_synthetic_ranking() {
        let n_features = 16;
        let (features_owned, targets) = make_synth_dataset(500, 400, n_features);
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
        let group = TrainingGroup {
            name: "synth".to_string(),
            human_scores: &targets,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };
        let mut log = Vec::new();
        let bake = train_mlp_per_sample_alpha_head(
            &mut [group],
            n_features,
            &MlpHyperparams {
                n_hidden: 16,
                n_epochs: 80,
                pairs_per_epoch: 2000,
                initial_lr: 0.005,
                per_sample_alpha_head: true,
                minibatch_size: 1,
                ..Default::default()
            },
            &mut log,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert!(!bake.is_empty(), "per-sample-α head produced empty bake");
        assert!(
            log.iter().any(|l| l.contains("srocc=")),
            "log should contain per-group panel stats"
        );
    }

    #[test]
    fn psah_tanh_pin_bounds_output() {
        let n_features = 12;
        let (features_owned, targets) = make_synth_dataset(600, 200, n_features);
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
        let group = TrainingGroup {
            name: "synth".to_string(),
            human_scores: &targets,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };
        let mut log = Vec::new();
        let bake = train_mlp_per_sample_alpha_head(
            &mut [group],
            n_features,
            &MlpHyperparams {
                n_hidden: 8,
                n_epochs: 30,
                pairs_per_epoch: 500,
                initial_lr: 0.003,
                per_sample_alpha_head: true,
                tanh_output_head_scale: 30.0,
                mse_weight: 1.0,
                ranknet_weight: 0.0,
                minibatch_size: 1,
                ..Default::default()
            },
            &mut log,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert!(!bake.is_empty());
        // With tanh pin scale=30, raw output y_pre maps through
        // 100·σ(y_pre/30) → predictions should be in [0, 100].
        // The log should show val(...) which is computed from pinned
        // predictions.
        let has_val = log.iter().any(|l| l.contains("val("));
        assert!(has_val, "expected val() in log");
    }

    #[test]
    fn psah_nan_transform_sweep_catches_poison() {
        use zenpredict::FeatureTransform;
        let rows = vec![vec![0.0, 1.0, 2.0], vec![0.0, 0.5, 1.5]];
        let transforms = vec![
            FeatureTransform::Identity,
            FeatureTransform::Identity,
            FeatureTransform::Identity,
        ];
        // No NaN expected with identity transforms.
        assert!(
            sweep_nan_inf(rows.iter().map(|r| r.as_slice()), &transforms, "test-clean").is_ok()
        );

        // Manually poison a feature to simulate log(0) = -inf.
        let mut poisoned_rows = rows.clone();
        poisoned_rows[0][1] = f64::NAN;
        poisoned_rows[1][2] = f64::INFINITY;
        let transforms_with_log = vec![
            FeatureTransform::Identity,
            FeatureTransform::Log,
            FeatureTransform::Log,
        ];
        let result = sweep_nan_inf(
            poisoned_rows.iter().map(|r| r.as_slice()),
            &transforms_with_log,
            "test-poison",
        );
        assert!(result.is_err(), "sweep should catch NaN/inf");
        let msg = result.unwrap_err();
        assert!(msg.contains("f1"), "should identify feature f1: {msg}");
        assert!(msg.contains("f2"), "should identify feature f2: {msg}");
    }

    #[test]
    fn psah_goals_policy_produces_score() {
        let n_features = 12;
        let (features_owned, targets) = make_synth_dataset(700, 200, n_features);
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
        let group = TrainingGroup {
            name: "konjnd_test".to_string(),
            human_scores: &targets,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };
        let mut log = Vec::new();
        let bake = train_mlp_per_sample_alpha_head(
            &mut [group],
            n_features,
            &MlpHyperparams {
                n_hidden: 8,
                n_epochs: 20,
                pairs_per_epoch: 500,
                initial_lr: 0.003,
                per_sample_alpha_head: true,
                validation_policy: ValidationPolicy::Goals,
                minibatch_size: 1,
                ..Default::default()
            },
            &mut log,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert!(!bake.is_empty());
        // Goals policy should produce a goals breakdown in the log.
        let has_goals = log.iter().any(|l| l.contains("goals:"));
        assert!(
            has_goals,
            "expected 'goals:' in log with --val-policy goals"
        );
    }

    #[test]
    fn light_panel_geomean_used_by_default() {
        let n_features = 12;
        let (features_owned, targets) = make_synth_dataset(800, 200, n_features);
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
        let group = TrainingGroup {
            name: "synth".to_string(),
            human_scores: &targets,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };
        let mut log = Vec::new();
        let _bake = train_mlp_per_sample_alpha_head(
            &mut [group],
            n_features,
            &MlpHyperparams {
                n_hidden: 8,
                n_epochs: 10,
                pairs_per_epoch: 300,
                initial_lr: 0.003,
                per_sample_alpha_head: true,
                minibatch_size: 1,
                ..Default::default()
            },
            &mut log,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        // Default val_aggregate is GeomeanSPP; log should show val(geomean3).
        let has_geomean = log.iter().any(|l| l.contains("val(geomean3)"));
        assert!(
            has_geomean,
            "default val_aggregate should be geomean3; log: {:?}",
            log.last()
        );
    }

    #[test]
    fn psah_2layer_trains_without_crash() {
        let n_features = 16;
        let (features_owned, targets) = make_synth_dataset(900, 200, n_features);
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
        let group = TrainingGroup {
            name: "synth".to_string(),
            human_scores: &targets,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };
        let mut log = Vec::new();
        let bake = train_mlp_per_sample_alpha_head(
            &mut [group],
            n_features,
            &MlpHyperparams {
                n_hidden: 16,
                n_epochs: 20,
                pairs_per_epoch: 500,
                initial_lr: 0.005,
                per_sample_alpha_head: true,
                n_hidden_layers: 2,
                minibatch_size: 1,
                ..Default::default()
            },
            &mut log,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert!(!bake.is_empty(), "2-layer variant produced empty bake");
        assert!(
            log.iter().any(|l| l.contains("srocc=")),
            "2-layer log should contain panel stats"
        );
    }

    #[test]
    fn psah_skip_connection_trains_without_crash() {
        let n_features = 16;
        let (features_owned, targets) = make_synth_dataset(1000, 200, n_features);
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
        let group = TrainingGroup {
            name: "synth".to_string(),
            human_scores: &targets,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };
        let mut log = Vec::new();
        let bake = train_mlp_per_sample_alpha_head(
            &mut [group],
            n_features,
            &MlpHyperparams {
                n_hidden: 16,
                n_epochs: 20,
                pairs_per_epoch: 500,
                initial_lr: 0.005,
                per_sample_alpha_head: true,
                skip_connection: true,
                minibatch_size: 1,
                ..Default::default()
            },
            &mut log,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert!(!bake.is_empty(), "skip variant produced empty bake");
    }

    #[test]
    fn psah_2layer_skip_combined_trains() {
        let n_features = 16;
        let (features_owned, targets) = make_synth_dataset(1100, 200, n_features);
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
        let group = TrainingGroup {
            name: "synth".to_string(),
            human_scores: &targets,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };
        let mut log = Vec::new();
        let bake = train_mlp_per_sample_alpha_head(
            &mut [group],
            n_features,
            &MlpHyperparams {
                n_hidden: 16,
                n_epochs: 20,
                pairs_per_epoch: 500,
                initial_lr: 0.005,
                per_sample_alpha_head: true,
                n_hidden_layers: 2,
                skip_connection: true,
                minibatch_size: 1,
                ..Default::default()
            },
            &mut log,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert!(!bake.is_empty(), "2-layer+skip produced empty bake");
    }

    #[test]
    fn psah_sigma_weighted_mse_trains() {
        let n_features = 12;
        let (features_owned, targets) = make_synth_dataset(600, 200, n_features);
        let feats_ref: Vec<&[f64]> = features_owned.iter().map(|v| v.as_slice()).collect();
        let n = targets.len();
        // Synthetic σ: low for first half (high confidence), high for second half.
        let sigmas: Vec<f64> = (0..n).map(|i| if i < n / 2 { 0.1 } else { 0.8 }).collect();
        let group = TrainingGroup {
            name: "sigma_test".to_string(),
            human_scores: &targets,
            features: FeatureRows::Borrowed(&feats_ref),
            metric_sigmas: Some(&sigmas),
            train_weight: 1.0,
            validation_weight: 1.0,
            ref_ids: None,
            loss_mode: GroupLossMode::default(),
        };
        let mut log = Vec::new();
        let bake = train_mlp_per_sample_alpha_head(
            &mut [group],
            n_features,
            &MlpHyperparams {
                n_hidden: 8,
                n_epochs: 30,
                pairs_per_epoch: 500,
                initial_lr: 0.003,
                per_sample_alpha_head: true,
                mse_weight: 1.0,
                sigma_weighted_mse: true,
                ranknet_weight: 0.5,
                minibatch_size: 1,
                ..Default::default()
            },
            &mut log,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert!(!bake.is_empty(), "σ-weighted MSE should produce a bake");
    }

    /// STRATEGY-2026-07-02 end-to-end smoke: ALL strategies active at once
    /// on synthetic data — catches runtime panics (indexing, borrow, NaN
    /// cascades) that the per-algorithm reference tests cannot. A failure
    /// here is an IMPL BUG (not strategy).
    #[test]
    fn strategy_smoke_all_active_end_to_end() {
        const NF: usize = 6;
        let n = 80usize;
        let mut feats: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut scores: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 / (n - 1) as f64;
            feats.push(vec![t, 1.0 - t, (t * 7.0).sin() * 0.1, t * t, 0.5, -t]);
            scores.push(t); // monotone target in [0,1]
        }
        let refs: Vec<&[f64]> = feats.iter().map(|r| r.as_slice()).collect();
        let sigmas: Vec<f64> = (0..n)
            .map(|i| 0.05 + 0.1 * ((i % 7) as f64) / 7.0)
            .collect();
        let mut groups = [
            TrainingGroup {
                name: "syn_a".to_string(),
                human_scores: &scores,
                features: FeatureRows::Borrowed(&refs),
                metric_sigmas: Some(&sigmas),
                train_weight: 1.0,
                validation_weight: 1.0,
                ref_ids: None,
                loss_mode: GroupLossMode::default(),
            },
            TrainingGroup {
                name: "syn_b".to_string(),
                human_scores: &scores,
                features: FeatureRows::Borrowed(&refs),
                metric_sigmas: None,
                train_weight: 0.5,
                validation_weight: 0.0,
                ref_ids: None,
                loss_mode: GroupLossMode::default(),
            },
        ];
        // triplet pool: stimuli = the same rows; responses consistent with
        // "left more distorted" when its quality target is lower.
        let mut pool = TripletPool::default();
        for f in &feats {
            pool.features.push(f.clone());
        }
        for i in (0..60).step_by(3) {
            let (l, r) = (i as u32, (i + 7) as u32);
            // scores[i] < scores[i+7] -> left lower quality -> resp 0
            pool.responses.push((l, r, 0));
            pool.responses.push((r, l, 1));
        }
        let hp = MlpHyperparams {
            n_hidden: 12,
            n_hidden_layers: 2,
            n_epochs: 12,
            pairs_per_epoch: 400,
            minibatch_size: 4,
            initial_lr: 5e-3,
            l2_lambda: 1e-6,
            leaky_alpha: 0.01,
            early_stop_patience: 0,
            per_sample_alpha_head: true,
            tanh_output_head_scale: 20.0,
            ranknet_weight: 1.0,
            mse_weight: 0.3,
            sigma_weighted_mse: true,
            // ALL strategies on:
            ema_decay: 0.9,
            hard_pair_frac: 0.5,
            hard_pair_max_delta: 0.1,
            stratified_bands: 5,
            dro_eta: 1.0,
            listwise_weight: 0.5,
            listwise_size: 6,
            listwise_frac: 0.15,
            triplet_weight: 0.5,
            triplet_frac: 0.15,
            triplet_tau: 0.6,
            triplet_sigma: 5.0,
            seed: 17,
            ..Default::default()
        };
        let mut log: Vec<String> = Vec::new();
        let bake = train_mlp_strategy(
            &mut groups,
            NF,
            &hp,
            &mut log,
            None,
            None,
            None,
            None,
            None,
            Some(&pool),
        );
        assert!(
            !bake.is_empty(),
            "IMPL BUG (not strategy): all-active strategy training produced an empty bake"
        );
        let joined = log.join("\n");
        assert!(
            joined.contains("STRATEGY active:"),
            "IMPL BUG (not strategy): strategy activation line missing from log"
        );
        assert!(
            !joined.contains("NaN") || joined.contains("val"),
            "IMPL BUG (not strategy): NaN observed in training log:\n{joined}"
        );
        // determinism: same seed -> byte-identical bake
        let mut log2: Vec<String> = Vec::new();
        let bake2 = train_mlp_strategy(
            &mut groups,
            NF,
            &hp,
            &mut log2,
            None,
            None,
            None,
            None,
            None,
            Some(&pool),
        );
        assert_eq!(
            bake, bake2,
            "IMPL BUG (not strategy): strategy training is not deterministic under a fixed seed"
        );
    }
}
