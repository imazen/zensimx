# Identity, activations and distortion-order study — 2026-11-06

The [paper-derived architecture queue](MONOTONE_ARCHITECTURE_EXPERIMENTS.md)
continues this study with mixed caps, activation switches, splines and
Lipschitz residuals, followed by separately controlled training experiments.

Zensimx is the experiment platform. This study does not modify serving profiles,
feature extraction, the running shared-depth comparison, or its frozen inputs.
The owner requested exploration of identity-preserving monotone activations,
regularization, distortion weights and training order. Experiments below distinguish
mathematical feasibility, short optimization diagnostics, full-budget development
comparisons and eventual product qualification. None implies the next.

## Construction and its limits

After the existing certified first-layer max-pair sums, let z >= 0 be a
nonnegative weighted sum. Define h(z;b) = f(b+z)-f(b). For any nondecreasing f,
h(0;b)=0, h>=0 and dh/dz=f'(b+z)>=0. Nonnegative intermediate weights and
score = 100 - sum(a_j h_j), a_j>=0, preserve identity and coordinate order.
The first-layer paired coefficients remain tied. This guarantees order in the
specified distortion features, not perceptual ordering of every possible image.

Train the anchor, not just the shifted activation: dh/db=f'(b+z)-f'(b).
For learned temperature/shape parameters, differentiate BOTH terms as well.
An anchor detached from autodiff gives the wrong bias gradient. Nonnegative
weights need projection after every update, including momentum/EMA finalization.

Floating-point identity is a separate contract. Evaluate both branches using the
same rounded weights, biases, activation and operation order. At z=0 their
subtraction must produce zero and the final score exactly 100. Baking a
compensating scalar bias can lose that property through rounding. Native
runtime support and bake parity are mandatory before calling any analytic
candidate a usable metric. The existing ZNPR route serves ReLU and SmoothHinge;
it does not yet implement these general anchored operators.

| Family | f(x), before anchoring | Derivative / experiment |
|---|---|---|
| ReLU control | max(x,0) | Existing native baseline; nonpositive bias makes anchor zero |
| SmoothHinge control | Existing tau=1/8 owner | Existing native smooth control |
| Leaky ReLU | max(x, alpha*x), 0<alpha<=1 | alpha below zero, 1 above; alpha={0.01,0.1} |
| Softplus | tau*log(1+exp(x/tau)) | sigmoid(x/tau); stable log1p formulation |
| Gaussian rectifier | x*Phi(x/tau)+tau*phi(x/tau) | Phi(x/tau); unlike ordinary GELU, never negative |
| Rational hinge | 0 if x<=0 else x^2/(x+tau) | x*(x+2*tau)/(x+tau)^2 above zero |
| Linear+tanh | x+a*tau*tanh(x/tau), a>=0 | 1+a*sech(x/tau)^2; positive with changing curvature |
| Monotone spline | positive segment slopes; shared ordered knots | Nonconvex curvature allowed; explicit extrapolation slope |

All parameters must be finite; tau>0. Fix tau initially; derive its scale only
from TRAIN preactivations, never development targets. Compare normalized slopes
at a TRAIN operating point to avoid confusing activation shape with effective
learning rate. Audit zero, signed zero, thresholds, tiny increments, extreme
biases, large inputs, f32 rounding and derivative underflow. No claim of novel
activation functions is made.

The first five candidate/control shapes are convex (for leaky alpha<=1), so a
positive composition can impose convex distance beyond monotonicity. The tanh
and spline arms test whether this restriction, rather than insufficient depth,
is limiting fit. Ordinary SiLU/GELU are negative controls: their derivatives can
be negative. Anchoring fixes identity, not their monotonicity.

## Dropout and regularization

Start with second-layer inverted dropout p={0,0.05,0.15}; freeze a mask for both
members of each rank pair AND for their identity-anchor evaluation. Independent
pair masks add ranking noise unrelated to image differences. A fixed nonnegative
mask preserves the monotone function. Masks in the first layer must be shared
within each certified max pair; independent masks can destroy its proof.
Inference disables dropout. Verify p=0 matches the baseline exactly and that
training with any mask retains identity and monotonicity. Treat mask RNG as a
third stream, independent of initialization and pair order.

The baseline uses coupled L2=1e-5 on weights, not biases, with Adam. First test
L2={0,1e-5,1e-4} without dropout; then dropout on a fixed activation. AdamW is a
separate optimizer arm, not an interchangeable spelling of L2. Record zero-bound
weights, dead units, gradient norms, saturation, and curvature; never assume a
plateau is a bad local minimum. Loss minimization can also stall at saddles,
projection boundaries, noise floors, or an overly restrictive function class.

## Distortion taxonomy and weights

Keep dataset, original source, distortion family, severity, codec and human-vs-
teacher provenance as separate keys. Suggested broad families: codec artifacts;
blur/detail loss; additive or signal-dependent noise; ringing/sharpening;
color/chroma; contrast/exposure; spatial/structured defects. Map native distortion
IDs explicitly, version the mapping, allow multi-label membership with mass split
across labels, and refuse unknown IDs. Do not infer distortion category from the
dataset name or silently re-admit previously excluded human distortions.

Balance hierarchically: source of supervision -> dataset -> distortion family ->
reference -> severity/pair type. Start from the currently admitted mixture.
Compare family-balanced and codec-emphasized weights (codec factor 2, other
families factor 1), renormalizing within each human dataset so dataset mass stays
fixed. Keep teacher mass separate; test relative mass {0.5,1,2}. Log actual draws,
accepted/rejected pairs, unique rows/references and cumulative family weight:
nominal sampling weights are not measured exposure. Importance-weighted
oversampling and a genuinely changed objective are distinct experiments.

Human ordinal labels supervise ranking within their permitted comparisons.
Do not compare raw MOS/JND numerically across datasets or let teacher scales
override human ordering. Severity is not pair difficulty: strong distortions
can be easy to rank; near-ties with uncertain human labels can be hard/noisy.
Use TRAIN label gaps, uncertainty when available, and codec settings separately.

## Order experiments with matched exposure

Freeze a complete pair multiset per seed before an order-only trial; reorder it
without changing multiplicities, weights, augmentations, masks or update count.
All stages retain at least 20% replay of the fixed full-mixture distribution.
Use the last 25% of updates for the same full-mixture consolidation in every arm.

1. Uniform interleaving (control).
2. Coverage walk: existing native `walk` preserves group draws and weights,
   changes reference/row coverage. This is NOT pure order-only and is labeled so.
3. Easy-to-hard versus hard-to-easy: reorder TRAIN pairs by confidence-adjusted
   target gap within dataset, with exact overall exposure and replay fixed.
4. Family-blocked versus shuffled blocks versus short round-robin interleaving.
   Rotate the family permutation across seeds; no permanent codec-last advantage.
5. Teacher-first/human-first versus interleaved, with equal total exposures and
   the identical consolidation tail. Track forgetting on every human population.
6. Bounded TRAIN-only adaptive sampling after a fixed warmup: cap family mass
   to [0.5,2] times its baseline, retain replay, and compare against a static
   matched-exposure replay of the realized schedule. Development scores never
   drive weights. Gradient-conflict surgery is a separate later optimizer arm.

Do not implement curricula as repeated `--init-from` calls: current bakes do not
carry the complete optimizer/RNG state, and constrained shared depth refuses
continuation. Restarting Adam/scaler/LR would confound order. Add schedules to
the existing sampler owner with global-step indexing and exact optimizer resume.
The current `stratified` mode ignores relative group weights and must not be
represented as a mass-matched alternative to uniform. `walk` is the initial
native coverage diagnostic; custom family-order curricula still need implementation.

## Registered staged experiments

A. Analytic audit: all listed scalar families, gradients in z and b, positive and
negative controls, dropout mask counterexample and matched-exposure schedules.
Independent Python references are feasibility tools, not candidate scorers.

B. Short native pilot: 12 epochs x 5,000 draws, final epoch 11, shared 128->64
ReLU, the admitted human+SSIM2 tables. Six arms: control; walk; initial LR 3e-4;
L2 1e-4; teacher mass 0.5; teacher mass 2. Three (init,sample) combinations:
(1101,101), (1107,101), (1101,200000101), separating an initialization change
from a sampling-stream change. No claim that these are three independent
replicates. Full four-population native signed/per-reference/scatter panels.
These short fits do not encounter the 50-epoch cosine restart and cannot settle
restart efficacy, long-run regularization or the earlier 120-epoch comparison.

C. Full-budget activation comparison: existing two controls plus anchored leaky,
softplus, Gaussian, rational and linear+tanh; three paired seeds, 120x50,000 draws,
fixed final epoch. Requires Rust forward/backward, native baking/prediction and
f32 constraint gates first. No silently approximated or Python-served models.

D. Focused follow-ups: L2 and dropout separately; then six order controls above
with matched exposures. Compare current cosine restarts, monotone decay and a
matched late low-LR tail, keeping optimizer-step budgets fixed. Do not choose
activation, dropout, sampler and LR simultaneously in an unpaired grid.
Select at most two development candidates using the full population trade-off
and constraint gates, then run the preregistered interactions, including a
nonconvex monotone activation x ordering comparison. No new terminal reads.

Record every failed arm. Report per-population/within-reference rank, raw scatter,
score spread/tails, severity and family coverage, forgetting after each stage,
seed variation and final-epoch performance. Source-clustered uncertainty uses
zenstats; KonFiG's limited development-source coverage prevents a broad
new-source claim. Never turn missing panels into a pass or average away a weak
population. Run independent fits concurrently within aggregate CPU/RAM budgets.

## Implementation ownership and evidence

- Scalar references / schedule conservation: `scripts/rev4_featpot/activation_curriculum.py`.
- Native pilot orchestration: same module, reuses `v2_lodo_mlp.train_command`,
  strict admission, native trainer, `bake_dial_refit` prediction and `panel`.
- Native candidate implementation: `mlp_train/shared_depth.rs`, existing
  `smooth_hinge`/SIMD owners, and the existing runtime/bake owner. Unsupported
  activations remain explicitly unsupported until those paths are complete.
- Results and unresolved items: [study evidence](../benchmarks/activation_curriculum_2026-11-06.md).

## Primary research context

[Curriculum Learning (Bengio et al., 2009)](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf)
provides motivation to test order, not a promise that easy-first wins here.
[Catastrophic forgetting study (Goodfellow et al.)](https://arxiv.org/abs/1312.6211)
motivates measuring performance on earlier categories throughout a sequence.
[Deep Lattice Networks](https://arxiv.org/abs/1709.06680) provides an alternative
monotone function class; monotonicity need not imply convexity.
[Activation definitions](https://docs.pytorch.org/docs/stable/nn.functional.html)
anchor the SiLU/GELU/leaky controls; the integrated-Gaussian derivative follows
by differentiating the formula above. These are hypotheses to test on admitted
zensim data, not external evidence that an arm improves zensim.

## Verified category-join limitation

The pilot's five fit tables contain features, `ref_basename` and `human_score`
only. Their sidecar keys contain `pair_key`, `source_row_id`, `ref_basename`
and `member_set`. No explicit distortion-family, codec or severity column is
present. Family curricula therefore require a versioned join to canonical
metadata using those keys, with complete coverage and no row movement across
source splits. The executed pilot tests population mass and coverage walking;
it does not pretend dataset names are distortion labels.

The subsequent [tree study](TREE_REGRESSION_STUDY.md) implements this join for
the same admitted rows: canonical bank pair keys supply distortion codes,
severity knobs and decoded reference hashes. Its metadata artifact retains
complete row order and checks reference separation across the inner folds and
development. This supplies metadata for a future neural curriculum; it does
not turn the earlier coverage-walk pilot into a family-order experiment.
