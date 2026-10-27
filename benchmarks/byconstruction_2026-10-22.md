# Monotone by construction — exploration (2026-10-22)

## Dataset naming for the current October 27 comparison

**AIC-2026 public_train is not AIC-3.** The current Mlic/Neff/Vote/Order
continuations use D6-admitted AIC-2026 public_train: 3,920 PTC crops from
28 sources. Fold results split those TRAIN sources into 24 fitted and four
held-out sources. “Held-out” there means out-of-fold within public_train,
not AIC-3, AIC-2026 public_test, or an independent final evaluation.
Full-data results fit all 28 sources and are explicitly in sample, as the
owner requested. Do not present their correlation or error as test accuracy.

AIC-3 CTC is a separate 600-pair, 10-reference corpus and retains its holdout
role. This continuation study does not read it. Historical exposure recorded
in DATA_SPLITS remains recorded; “holdout role” is not a claim that AIC-3 has
never been inspected. AIC-2026 public_test (14 sources), AIC-4 and SDR25 are
also outside this new training/comparison read scope. Public_test's earlier
one-read exposure remains spent; these results do not reopen it.

Use **AIC-2026 public_train** in current result prose and tables. Frozen raw
artifacts retain their internal `aic` population key and historical filenames;
for this registered study that key maps to AIC-2026 public_train, with table
SHA256 `a2969c70f2e1c96f18348b0b8e66a28e41ee87ad839636e8329e63408b138b71`.
Do not rename or rewrite frozen artifacts to fix display terminology.

Question from the owner: can a metric that is correct **by construction** be competitive with
B (`diff+frag-820`, the E33/E34 `base-c` recipe)?

"By construction" here means the network guarantees, for every input and not only on data:

1. `raw(0⃗) = pin` bit-exactly and `raw ≤ pin` (B already has these through `--nonneg-distance`);
2. `raw` is **non-increasing in every input**, so more of any measured damage never raises the score.

B does not have (2). Its first layer has 67,779 negative weights against 22,582 positive. The
LIVE color-noise inversion (`benchmarks/gauntlet_live_color_trace_2026-10-18.md`) is this
mechanism: positive damage passes through negative first-layer weights and switches distance
units off.

This is an exploration record, not a registered experiment. It selects on E21, which is a
TRAIN-population cross-dataset statistic (LODO over KADID TRAIN+SELECT, TID2013, KonFiG
TRAIN+VAL and CID22-A(25)), already exposed by E33/E34/V40. Any candidate it produces needs a
separately registered assessment before it can claim more than "competitive on E21".

## Facts established before any arm ran

- **Domain.** All 420 read features are ≥ 0 with no NaNs on every fit, dev and coverage table and
  on all four E21 assessment tables (266,375 training/dev rows; the 42,021-row coverage pool).
  The 410 fragility products are therefore ≥ 0 as well (fragility ∈ (0, 1]). Monotonicity in
  each input is monotonicity on the whole data domain.
- **Zero-bias monotone networks are linear.** With `b1 = 0`, `W1 ≥ 0` and non-negative inputs,
  every ReLU unit is active on every row, so `raw = pin − c·x`. `--nonneg-distance
  --monotone-cbc` at any width is E34 arm L's class. E34 measured L (A family, 10 seeds) at
  **−0.0226** against `base-a`, with KADID −0.017, TID2013 −0.035, KonFiG −0.037.
- **Capacity inside the monotone class needs thresholds.** `--nonneg-thresholds` (this branch)
  projects `b1 ≤ 0` instead of freezing it at 0. `ReLU(b1) = 0` keeps (1) exact; with
  `W1 ≥ 0` the model is a non-negative sum of hinges `max(0, a·x − t)`, `a ≥ 0`, `t ≥ 0`:
  monotone, convex, piecewise-linear. Within-pair rank loss gives `b1` an exactly zero gradient
  while every unit is active, so thresholds start at per-unit pre-activation quantiles
  `U[0, 0.9)` of 4,096 training rows (own RNG stream).
- **Harness parity.** This workspace's trainer (flag off) and the `v2_lodo_mlp.py` strict route
  reproduce E34's `base-c` KADID seed-0 cell: training curve identical at every logged epoch and
  all 7,869 held-out predictions bit-identical to E34's stored predictions. The scorer
  (`scripts/rev4_featpot/byc_cells.py score`) reproduces E34's published point estimates for
  w-c (+0.00213532) and k-c (−0.00069957) against `base-c`.

## Arms (C family; every other recipe element is B's)

| Arm | Tokens | Class | Guarantee |
|---|---|---|---|
| control | — | E34 `base-c` stored predictions (seeds 0–9) | pin only |
| Lc | `H1:mono` | linear, `W1 ≥ 0` | monotone + pin |
| T | `mono:thr` | sum of hinges, `W1 ≥ 0`, `b1 ≤ 0`, H128 | monotone + pin |
| Ts | `thr` | B plus thresholds, signed `W1` | pin only (capacity control) |
| R | `mono:thr:rp32` | 64 hinges + 32 tied ramp pairs (H128), `W1 ≥ 0` | monotone + pin, non-convex |

**Amendment before any arm result (2026-10-22 ~07:55Z), from a methodology review.** T tests
*convex* monotonicity, so a T loss falsifies convexity, not "monotone by construction". R is the
non-convex monotone arm: each tied pair `w2·(ReLU(z − t₁) − ReLU(z − t₂))`, `t₂ ≥ t₁ ≥ 0`, is a
bounded saturating ramp in the existing wire format; `certify_monotone_bake` verifies the
structure from the bake bytes. B's fragility products are already a narrow content gain under
`W1 ≥ 0` (effective weight `a_k + b_k·f_k ≥ 0`, `f_k ∈ (0, 1]`); a richer content-gated
partial-monotone arm (free content network × monotone damage terms) is a later step.
Because E21 is leave-one-dataset-out, it cannot see damage-mixture shift (the LIVE color-noise
failure came from families the coverage mask excluded). Two checks are added beside it for any
competitive arm: pair classes by distortion type (E34 §4.6 owner), and a label-free
damage-mixture test (adding a second distortion to a distorted image must not raise the score).

## Decision rules (fixed before any arm result)

- Screen: seeds 0–2 × four folds, paired with `base-c`'s same seed indices. Statistic: E21 point
  estimate (mean over the four sources of signed SROCC difference, averaged over seeds) with
  the seed-unit SE.
- **Competitive on E21** = E33's point form: mean Δ ≥ −0.002 and every dataset Δ ≥ −0.005.
- An arm that is competitive on the 3-seed screen goes to seeds 3–9, then E34's full statistic
  through the `panel e34` owner and the full-data label-free gates (identity, dial C1–C6,
  G-STEER). An arm that misses by more than 2 seed-unit SEs on the screen is reported as
  measured and not extended.
- Every result is reported, including failures. No threshold above changes after a result.

## Results

Cells: `~/tmp/byc/cells/<arm>/without_<fold>_s<seed>/` on dev (train.log, `byc_record.json` with
bake and prediction hashes, held-out `pred_<fold>.tsv`). Arm T used the `8ba33c7a` trainer
(`/var/tmp/byconstruction-target/release` at the time), its predictions the `4024c31b`
`bake_dial_refit`, which was re-checked bit-identical on E34's `base-c` KADID seed-0 cell.
Each T cell: 984–1013 s single-threaded on a contended box, peak RSS 0.99 GiB.

### Arm T (`mono:thr`, convex monotone): not competitive on E21

| | E21 Δ vs `base-c`, seeds 0–2 |
|---|---|
| mean (seed-unit SE) | **−0.0205 (0.0029)**; per seed −0.0154, −0.0255, −0.0207 |
| KADID / TID2013 / KonFiG / CID22-A | −0.0192 / −0.0423 / −0.0267 / **+0.0060** |
| absolute SROCC | 0.8714 / 0.8590 / 0.7904 / 0.8860 |

That is close to E34's linear L (−0.0226, A family). **Correction (same session, after
inspecting the bakes):** T did not use its convex capacity. Every T cell, and the full-data T
fit, ends with **one** live hidden unit out of 128 (two in one cell), sometimes with a small
threshold: the model is at most a single hinge `pin − c·max(0, a·x − t)`, essentially E34 L's
linear class with a dead zone. The other units' output weights sit at the `w2 ≤ 0` clamp.
Hinges are convex (they accelerate distance with damage); the training loss, through its
MSE leg on the teacher groups, wants a *saturating* response, so it pushes hinge output weights
positive and the clamp kills them. So T measures the linear monotone class on this data, not
convex monotone capacity, and the ramp arm R (saturating, still monotone) is the real capacity
test. By the rule above T is not extended. The loss has a specific shape.

**Pair classes** (`byc_cells.py pairclass`, an independent numpy implementation of E34 §4.6's
classes; concordance on target-ordered pairs, exact target ties excluded, cross-source pairs
for KADID a fixed 2M sample; mean over seeds 0–2; Δ = T − `base-c`):

| Dataset | same ladder | cross-type, same reference | cross-source |
|---|---|---|---|
| KADID | **+0.0067** | −0.0133 | −0.0120 |
| TID2013 | **+0.0053** | −0.0205 | −0.0234 |
| KonFiG | **+0.0015** | +0.0041 | −0.0132 |
| CID22-A (codecs) | −0.0006 | −0.0019 | **+0.0032** |

Within a ladder (same reference, same distortion) the monotone model orders as well or better,
up to +0.22 within-reference SROCC on KADID's contrast-change type (`byc_cells.py bytype`).
What it loses is calibration of severity *between* distortion types and contents on the
synthetic-distortion sets. On codec data it is neutral to slightly better.

**Development curves** (TRAIN-internal reference-disjoint dev groups, 12 paired cells, epoch 0 →
119): B's recipe loses human-dev SROCC while it keeps fitting the metric teachers; T does not.

| Dev group | T | `base-c` | T − `base-c` at 119 |
|---|---|---|---|
| human | 0.8122 → 0.8204 (+0.0082) | 0.8201 → 0.8092 (−0.0108) | **+0.0112** |
| CID22 teacher | +0.0028 | +0.0054 | −0.0041 |
| SafeSyn teacher | +0.0027 | +0.0035 | −0.0007 |

**Why convex monotone cannot calibrate across types.** A non-negative sum of hinges
`Σ c·ReLU(a·x − t)`, `a ≥ 0`, cannot express `max(x₁, x₂)`, which is monotone and convex:
`max(x₁, x₂) = x₁ + ReLU(x₂ − x₁)` needs a signed first-layer row. Max-like pooling is how a
metric avoids counting one artifact several times through redundant features; with only
non-negative rows, a distortion type that lights many correlated features is over-weighted
relative to one that lights few. That is the measured cross-type loss. Monotone classes that
do allow max-pooling (min-max networks; certifiable signed constructions) are the next arms.

### Damage-mixture test (label-free): B inverts, monotone does not

`scripts/byconstruction/mixture_test.py` (generate / score / report). 48 TRAIN-eligible 256-px
references from E34's inventory (`p4-inventory-r2/references`; E34's 128 augmentation and 8
development origins excluded; sha256 order within category), 4 base distortions D1 (Gaussian
blur σ 1.2, luma noise σ 4, chroma noise σ 8, posterize 12 levels) × 6 added distortions D2
(luma noise σ 4/8/16, chroma noise σ 8/16/32, BT.601 YCbCr, fixed seeds): 1,152 D1 → D2∘D1
steps, every one of which raises RGB error against the reference. Scored with
`score_pair_with_bake --bake-post raw` (`526261c3` build). An inversion is a score that rises
when D2 is added. Images and scores: `/mnt/v/output/zensim/byconstruction-2026-10-22/mixture/`.

| Model | Inversions / 1,152 | > 1 point | Max rise |
|---|---|---|---|
| B (production `89cb4861`) | **148** | 131 | **+44.2** |
| E34 `base-c` cell (KADID held out, seed 0) | 110 | 107 | +42.9 |
| T cell (same fold and seed) | **0** | 0 | −0.10 (smallest drop) |

B's worst cell, blur σ 1.2 then chroma noise σ 8, inverts on **48 of 48** references: mean B
score 15.0 for the blurred image, 33.8 after adding chroma noise (median rise +17.1, max +44.2).
Example 9940: B 26.5 → 67.3 while RGB MSE rises 6.0 → 80.3; T 66.8 → 33.1. This is the LIVE
color-noise failure mode reproduced systematically on TRAIN images, without labels. The
monotone model's guarantee is in feature space; on these pixel mixtures it also held end to
end.

Across all 12 matched LODO cells (4 folds × seeds 0–2; E34's stored `base-c` dense bakes vs T's):

| | Inversions / 1,152 per cell | Max rise per cell |
|---|---|---|
| `base-c` (B's recipe) | 86–171 | +24.7 to +61.5 |
| T (convex monotone) | 0–4 | ≤ +4.03 |

T's only inversions are posterize-12 followed by *light* noise (luma σ 4 or chroma σ 8), at most
3 references per cell: dithering. Light noise breaks up posterization contours and the
banding-type features fall, and dithered banding often looks better, so these are plausibly
correct. B's inversions are blur followed by chroma noise, with no such rationale. T's
guarantee is in feature space; a feature that falls when damage is added (here, banding under
dither) is the only route to an end-to-end rise.

### Arm R (`mono:thr:rp32`, saturating monotone): uses its capacity, no E21 gain

`4024c31b` trainer. 12 cells, 2006 s for the batch on a contended box, peak RSS 1.04 GiB.

| | E21 Δ vs `base-c`, seeds 0–2 |
|---|---|
| mean (seed-unit SE) | **−0.0225 (0.0024)**; per seed −0.0181, −0.0265, −0.0230 |
| KADID / TID2013 / KonFiG / CID22-A | −0.0168 / −0.0391 / −0.0299 / −0.0043 |

Unlike T, R keeps its structure: 13–29 live ramp pairs per cell (`live_units.py`), plus exactly
one live plain hinge (the other 63 hinges die, as in T). Saturation is used and does not move
E21: the missing ingredient is not the response shape.

Pair classes (Δ vs `base-c`): same ladder KADID +0.0070, TID +0.0085, KonFiG +0.0016, CID22-A
−0.0001; cross-type KADID −0.0109, TID −0.0172, KonFiG −0.0004, CID22-A −0.0019; cross-source
KADID −0.0098, TID −0.0226, KonFiG −0.0170, CID22-A −0.0031.

**Development curves revise the overfitting reading.** R's human-dev SROCC falls over training,
0.8080 → 0.7903 (−0.0177; `base-c` −0.0108), while teacher dev rises (+0.0016 SafeSyn, +0.0027
CID22). T's flat human-dev curve came from collapsing to a linear model, not from monotonicity:
with capacity, a monotone model also trades human ranking for teacher fit. The drift is a
capacity × teacher/human-recipe effect, not a sign-freedom effect.

Mixture test: 4–9 inversions per cell, max +5.22, all posterize followed by light noise
(dithering), the same cases as T.

### Where the monotone deficit lives (post hoc)

The type set below was chosen after reading the per-type table, so this is descriptive, not a
test. Removing the "benign global change" distortion types (KADID 7/8 saturation, 16 brighten,
17 darken, 18 mean shift, 20 non-eccentricity patch, 25 contrast change; TID2013 16 mean
shift, 17 contrast change, 18 saturation change; `scripts/byconstruction/benign_types.py`):

| Arm | KADID: all types → without | TID2013: all types → without |
|---|---|---|
| T | −0.0192 → −0.0090 | −0.0423 → −0.0129 |
| R | −0.0168 → −0.0062 | −0.0391 → −0.0114 |

Roughly 60–70% of the deficit on these two sets comes from distortions that move many damage
features while humans object little; a signed model can subtract the explained part, a
monotone one cannot. KonFiG (−0.027 / −0.030) has none of these types and keeps its deficit.

### The exposed LIVE color-noise case (report-only)

`manfishing`, `wn/img41` (severe, human 1 − DMOS/100 = −0.0024) vs `wn/img73` (milder, 0.3097),
`score_pair_with_bake --bake-post raw`. These examples are exposed and stay out of fitting and
selection (CLAUDE.md, 2026-10-18 Known Bug); this is a diagnostic read only.

| Model | severe img41 | milder img73 | ordered correctly |
|---|---|---|---|
| B (`89cb4861`) | 49.71 | −13.91 | no (matches the trace's 49.7143 / −13.9146) |
| `base-c` KADID-held-out cell s0 | 1.72 | −21.65 | no |
| T same cell | −207.42 | −68.62 | yes |
| R same cell | −171.78 | −46.56 | yes |
| T full-data, packed f16 + spline | −163.72 | −60.27 | yes |

### Label-free product gates: the full-data monotone model passes all of them

Full-data T, seed 0, packed by the E33 output stage (`v2_confirm_fit.py --pack-production
--e33-output-stage`, 776 s, peak RSS 0.99 GiB; `production-f16.bin` 21,215 B because only one
hidden unit is live). `bake_dial_refit certify-monotone`: certified, monotone. Gates through
`scripts/byconstruction/gates.py` (E33/E34 owners and binaries; outputs
`/mnt/v/output/zensim/byconstruction-2026-10-22/gates/`), against E34's `base-c` seed 0:

| Gate | Monotone T (full) | `base-c` s0 |
|---|---|---|
| C1 dial monotonicity, standard / ladder | **0.9970 / 0.9914** | 0.9954 / 0.9882 |
| C2–C6 (both grids) | pass | pass |
| G-DIAL (p5 ≤ 25, p95 ≥ 85, mono ≥ .93) | pass (p5 13.8 / −9.9, p95 97.5 / 95.2) | pass |
| NEARID N1–N3 | pass, **140/144** monotone ladders | pass, 130/144 |
| G-STEER (D12 exact gradients, 135 cases) | **135/135** | 130/135 |
| Identity at every SIMD tier (`e33_gates.identity`, C5 as registered) | 62 sources pass at all 10 permutations (620 rows) | (E34 gate record) |


The model is effectively linear, which is E34 L's class (E34's L also cleared G-STEER), so the
steering pass is expected, but it is now attached to a certified-monotone bake with a measured
mixture-test record.

### Arms added after T/R (definitions recorded before their results)

| Arm | Tokens | Class | Certified guarantee |
|---|---|---|---|
| E5 / E20 | `mono:thr:fr32:fb5` / `…:fb20` | 64 monotone hinges + 32 sign-free ramp pairs, summed amplitude ≤ 5 / 20 raw units | pin; raising any input raises the score by ≤ 5 / 20 |
| G | `mono:thr:gate` | monotone hinges; each fx1 (direct, product) pair projected onto `{a ≥ 0, a/s_d + b/s_p ≥ 0}` | pin; non-increasing in every damage feature for every fragility in (0, 1] |
| Lc | `H1:mono` | linear, `W1 ≥ 0`, C family | monotone (E34 L in B's family) |
| Ts | `thr` | B + trainable thresholds, signed `W1` | pin only (capacity control) |

G is the content-gated partial-monotone class: T forced both weights ≥ 0, so a unit's gain on
damage could only grow with fragility; G lets it rise or fall with content while staying
non-negative. The gate cone keeps a 2⁻⁸ margin and a direct-weight floor so f16 packing
cannot cross it; `certify_monotone_bake` checks it from the bake's `zensim.derived_inputs`
declaration (product factors must be `pjnd_fragility` slots). Smoke: a 2-epoch real-data G
cell's dense bake certifies monotone with 5,535 negative product weights.

The same decision rule applies to each: E33 point form on the 3-seed screen, then seeds 3–9
and the full-data gates for any arm that meets it.

**Per-type calibration bias** (`scripts/byconstruction/type_bias.py`; mean pooled prediction rank
minus pooled human rank, seeds 0–2; negative = rated worse than humans). Largest T − `base-c`
gaps:

| Set | Type | T | `base-c` |
|---|---|---|---|
| KADID | contrast change | −0.132 | −0.023 |
| KADID | color shift | −0.006 | +0.098 |
| KADID | color quantization | −0.059 | +0.003 |
| KADID | non-eccentricity patch | −0.086 | −0.036 |
| TID2013 | non-eccentricity pattern noise | −0.217 | −0.095 |
| TID2013 | local block-wise | +0.040 | +0.150 |
| TID2013 | denoising | −0.087 | −0.003 |

The deficit is two-sided: the monotone model over-penalizes types that light up many damage
features (contrast change, pattern noise, denoising), while B's recipe over-forgives some
others (color shift, block-wise) where T is closer to the human rank. "Benign global change"
is part of the story, not all of it.

### Arm E5 (`mono:thr:fr32:fb5`, certified budget 5): budget fully used, no E21 gain

`526261c3` trainer, 1,910 s for 12 cells, peak RSS 1.03 GiB.

| | E21 Δ vs `base-c`, seeds 0–2 |
|---|---|
| mean (seed-unit SE) | **−0.0194 (0.0027)** |
| KADID / TID2013 / KonFiG / CID22-A | −0.0158 / −0.0403 / −0.0231 / +0.0014 |

Every cell certifies (`certify-monotone` on its dense bake) with 5–21 live free ramps and a
certified free amplitude of 3.47–4.98 of the 5-unit budget: the optimizer wants more
non-monotone room than 5 points, and 5 points buys nothing measurable on E21. Pair classes
match T (same ladder KADID +0.0067, TID +0.0057; cross-type −0.0110 / −0.0189; cross-source
−0.0099 / −0.0226). Mixture test: 3–8 inversions per cell, max rise +5.01 (dithering cases; the
budget bound plus the feature-space-to-pixel gap).

### Information vs calibration: an oracle per-type calibration (diagnostic upper bound)

`scripts/byconstruction/oracle_type_calibration.py` gives each distortion type its own
least-squares affine map from prediction to human score, using the labels themselves (an
oracle, not a model), applied identically to the arm and to `base-c`; then pooled SROCC Δ:

| Arm | KADID raw → calibrated | TID2013 | KonFiG | CID22-A |
|---|---|---|---|---|
| T | −0.0192 → **+0.0015** | −0.0423 → −0.0054 | −0.0267 → −0.0153 | +0.0060 → +0.0078 |
| R | −0.0168 → **+0.0025** | −0.0391 → −0.0081 | −0.0299 → −0.0159 | −0.0043 → −0.0025 |
| E5 | −0.0158 → **+0.0019** | −0.0403 → −0.0058 | −0.0231 → −0.0144 | +0.0014 → +0.0055 |

Within each distortion type, across content and severity, the monotone models carry as much
human ranking information as B's recipe (slightly more on KADID). The E21 gap is almost
entirely how *different distortion types* are placed on one severity scale, which B obtains
from sign contrasts between correlated damage features. KonFiG keeps about −0.015. CID22-A,
whose "types" are codecs, is positive in raw form for T and E5.

So, on this feature set: coordinate-wise monotonicity and cross-type severity calibration
conflict; ranking within a distortion type does not suffer. For codec targeting, where the
types are codecs (CID22-A), the monotone models are at parity or better.

**Arm M (`mono:thr:mp32`), recorded before its result.** The oracle above says the monotone
deficit is cross-type severity calibration. The standard monotone way to stop one artifact
being counted several times through redundant features is to pool with a maximum instead of a
sum, and `W1 ≥ 0` hinge sums cannot express `max(x₁, x₂)`. M's first 64 hidden units form 32
certified max pairs `c·max(a·x, b·x − t)` (u: row `a ≥ 0`, bias 0; v: row `b − a`, `b ≥ 0`,
bias `−t ≤ 0`; tied `c ≤ 0`), the other 64 stay monotone hinges. A 2-epoch real-data smoke keeps
31 live max pairs and certifies monotone. Queue order (serial, one heavy job at a time): E20, G,
M, Lc, Ts.

### Prior evidence: data and inputs alone do not remove the inversion (registered, 2026-10-18)

[color_noise_repair_2026-10-18.md](color_noise_repair_2026-10-18.md) added the color/noise
coverage families that `cf98` excludes, restored fine-scale X/B inputs, and did both, on B's
signed class, as single models and as 3-seed ensembles. Every composition still scores the
severe LIVE `wn/img41` above the milder `wn/img73` (severe 0.26–62.6 vs milder −31.3 to −5.95).
The monotone bakes here order that pair correctly by construction. Coverage is a complement to
the guarantee, not a substitute for it.

### Arm E20 (`mono:thr:fr32:fb20`): the certified budget trades E21 for bounded inversions

`526261c3` trainer, 1,915 s for 12 cells, peak RSS 1.02 GiB. Every dense bake certifies; the
free amplitude is 19.67–20.00 in 9 cells and 5.3–6.6 in 3.

| Non-monotone budget ε (certified) | E21 Δ vs `base-c` (seed-unit SE) | Mixture inversions / cell | Max rise |
|---|---|---|---|
| 0 (T) | −0.0205 (0.0029) | 0–4 | +4.0 (dithering) |
| 5 (E5) | −0.0194 (0.0027) | 3–8 | +5.0 (dithering) |
| 20 (E20) | **−0.0154 (0.0026)** | 4–37 | +7.3 |
| unbounded (`base-c`) | 0 | 86–171 | +24.7 to +61.5 |

E20 per dataset: KADID −0.0110, TID2013 −0.0364, KonFiG −0.0167, CID22-A +0.0026. Pair
classes vs `base-c`: cross-type KADID −0.0078 (T −0.0133), TID −0.0160 (T −0.0205).

In one E20 cell (TID held out, seed 0) the free room reproduces B's failure mode: blur σ 1.2
then chroma noise σ 8 inverts on 26 of 48 references (max +7.3, inside the certificate's 20).
The non-monotone room that buys E21 is spent partly on the same inversions B makes; the
certificate bounds their size.

### Correction: the logged human-dev decline is not a clean memorization signal

The trainer's per-epoch dev SROCC (`compute_light_panel_subsampled` over all rows of a group)
is pooled across the human dev group's sources, each with its own label scale; E21 scores each
held-out dataset separately. Testing it directly: every cell also saved `refit/best.bin`, the
trainer's best dev-geomean3 epoch (the harness's `best_dev` rule).
`byc_cells.py predict-best` + `score --rule best` and `scripts/byconstruction/epoch_rules.py`:

| Model | mean E21 pooled SROCC, last epoch → best-dev epoch |
|---|---|
| `base-c` | 0.8722 → 0.8642 (**−0.0080**) |
| T | 0.8517 → 0.8518 |
| E5 | 0.8528 → 0.8533 |
| E20 | 0.8569 → 0.8529 |
| R | 0.8497 → 0.8461 |

Selecting by dev hurts B's recipe on held-out datasets; later epochs are what make it better
there. Under the `best_dev` rule the arm deltas shrink (T −0.0124, E5 −0.0109, E20 −0.0113, R
−0.0181) only because the control gets worse. The registered last-epoch comparison above
stands, and the earlier statements that R's or B's dev decline shows memorization are
withdrawn: what the logged metric shows is a shift in pooled cross-source ranking on the
in-population dev references, not an E21 loss.

## Where this stands (09:40Z; G, M, Lc, Ts still running)

**Can a metric that is correct by construction be competitive with B?**

- *Product gates:* yes. The certified-monotone full-data model passes every label-free gate
  run here (C1–C6, G-DIAL, N1–N3, identity at every tier, G-STEER 135/135 where `base-c` gets
  130/135), with higher dial monotonicity.
- *Robustness:* strictly better. 0–9 mixture inversions per cell (all dithering) against
  86–171 with rises to +61 for B's recipe; the exposed LIVE pair is ordered correctly. A
  registered repair with more coverage and inputs did not achieve this on B's signed class.
- *Codec calibration:* at parity. CID22-A (codec types) is +0.006 for T in raw form.
- *E21:* not yet. Best certified-monotone arm −0.0194 (E5); with a certified non-monotone
  budget of 20 raw units, −0.0154. The deficit is cross-type severity calibration on synthetic
  distortion sets: with an oracle per-type affine calibration the monotone arms are at parity on
  KADID (+0.002) and within −0.005 to −0.008 on TID.

**Mechanism.** B places distortion types on one severity scale with sign contrasts between
correlated damage features (half of its damage features have a positive derivative on a
typical row). The same contrasts produce its inversions. Coordinate-wise monotone combiners
built from these features cannot do that calibration; response shape (saturating ramps) does
not help, and a certified non-monotone budget buys it back slowly while spending part of the
room on the same inversions.

**Not supported by the data:** that sign freedom causes memorization (the logged dev decline
is pooled across sources; selecting by it lowers B's E21 by 0.008).

### A feature-level lead for the cross-type deficit (descriptive)

T's KADID-held-out seed-0 cell has one live unit (bias 0), so its distance splits exactly into
per-input contributions (`scripts/byconstruction/feature_contributions.py`). Share of total
distance by feature family, global-change types vs all KADID rows:

| Types | v2 ringing | other over-represented families | mean distance |
|---|---|---|---|
| contrast change | 15.0% (all types 7.2%) | hf_loss 9.0% (6.7%), transducers ~1.5% (0.9%) | 68.9 |
| saturation 1+2 | 12.6% | iw_ssim 8.1% (2.6%), iw_mse 3.3% (0.9%) | 129.1 |
| mean shift / brighten / darken | 16.4% | det family slightly up | 67.9 |
| all types | 7.2% | — | 86.1 |

The ringing feature (dilated-edge form) responds strongly to global tone and saturation
changes, which a ringing detector should not; on saturation the IW-weighted SSIM/MSE features
are about 3× over-represented. Tone-normalized versions of these features are the concrete
feature-side experiment for letting a monotone combiner calibrate across types. Not tested
here.

**Arm FR (`mono:thr:freering`), recorded before its result; post hoc.** Same as T, but the 10
v2 ringing direct inputs and their 10 fragility products (20 of 820 inputs;
`scripts/byconstruction/masks/free_ringing.tsv`) are sign-free via the existing
`--monotone-feature-mask`. Guarantee: monotone in the other 800 inputs. The feature was chosen
after reading contributions on KADID assessment rows, so FR is exploratory and cannot by itself
support a claim. It tests whether the cross-type deficit sits in a few mis-specified features.
Queued after Ts.

### Arm G (`mono:thr:gate`, content-gated monotone): no change

`4112601a` trainer. E21 Δ **−0.0202 (0.0030)**; KADID −0.0178, TID −0.0429, KonFiG −0.0261,
CID22-A +0.0060 (T: −0.0205). Dense bakes certify monotone through the gate rule. Like T, each
cell keeps one live unit, now a linear model whose damage gains move with fragility in both
directions (product weights down to −1.99, inside the cone). Pair classes match T (cross-type
KADID −0.0123, TID −0.0205). Content-dependent gain is not what the monotone class lacks; the
deficit is cross-type. Mixture test: 0–4 inversions per cell, max +4.49, dithering cases only.

### Arm M (`mono:thr:mp32`, certified monotone max pairs): best exactly-monotone arm

`d2b7ce30` trainer. Every dense bake certifies monotone (1–2 live signed max pairs each).

| | E21 Δ vs `base-c`, seeds 0–2 |
|---|---|
| mean (seed-unit SE) | **−0.0144 (0.0016)**; per seed −0.0133, −0.0123, −0.0175 |
| KADID / TID2013 / KonFiG / CID22-A | −0.0147 / −0.0252 / −0.0183 / +0.0008 |

Paired with T by seed: +0.0021, +0.0132, +0.0032. In 10 of 12 cells exactly two hidden units are
live, i.e. the whole model is `pin − c·max(a·x, b·x − t)`: one learned maximum between two
non-negative damage detectors. That single pooled pair recovers a large part of the cross-type
deficit:

| Pair class, Δ vs `base-c` | T | M |
|---|---|---|
| TID2013 cross-type | −0.0205 | **−0.0120** |
| TID2013 cross-source | −0.0234 | **−0.0152** |
| KADID cross-type | −0.0133 | **−0.0101** |
| KonFiG cross-source | −0.0132 | **−0.0088** |
| same ladder (all sets) | +0.0015 to +0.0067 | +0.0016 to +0.0064 |

Max-pooling is the monotone mechanism for cross-type calibration, as predicted from the
`max(x₁, x₂)` argument, and training keeps only one pair alive.

**Mixture test: feature-space monotonicity is not pixel-space monotonicity.** M has 1–32
inversions per cell (max +6.54), now including blur followed by noise. Some features are not
monotone in pixel damage: `hf_loss` (detail lost) falls when noise adds high-frequency energy
back to a blurred image, and a max lets the score follow whichever detector dominates. The
rises are small next to B's (+61), but a pixel-level guarantee needs damage-monotone features
as well as a monotone combiner; the mixture test is the practical check.

### Arm Lc (`H1:mono`, linear monotone in B's family)

E21 Δ **−0.0204 (0.0029)**; KADID −0.0186, TID −0.0423, KonFiG −0.0266, CID22-A +0.0060 — the
same as T, confirming T was the linear class in effect. (E34's L in the A family: −0.0226.)
12 cells in 68 s: width 1 is cheap.

### Seed stability (`scripts/byconstruction/seed_spread.py`)

Mean E21 pooled SROCC over the four sources, seeds 0–2:

| Model | mean | per seed | sd |
|---|---|---|---|
| `base-c` (B's recipe) | 0.8722 | 0.8665 / 0.8769 / 0.8733 | 0.0053 |
| E34 `w-c` (H32) | 0.8775 | 0.8758 / 0.8785 / 0.8782 | 0.0015 |
| T / Lc / G / E5 / E20 / R | 0.8497–0.8569 | — | 0.0008–0.0015 |
| M | 0.8579 | 0.8532 / 0.8646 / 0.8558 | 0.0060 |

The sign-constrained arms are 4–6× more seed-stable than B's recipe; M's single max pair brings
the variance back (which pair is found depends on the seed). E34's narrow `w-c` is both more
stable and higher than B's own width: plain capacity reduction is a cheap stabilizer.

**Seed ensembles** (`seed_ensemble.py`; mean raw prediction over seeds 0–2; averaging
certified-monotone models stays monotone): `base-c` 0.8722 → 0.8760, M 0.8579 → 0.8599, E20
0.8569 → 0.8578, T 0.8517 → 0.8519. Ensemble gaps vs `base-c`: M −0.0161, E20 −0.0182, T
−0.0241. B's recipe gains more from ensembling, so this does not close the gap.

### Arm Ts (`thr`, B plus thresholds, signed W1): parity

E21 Δ **−0.0008 (0.0012)**; KADID −0.0004, TID −0.0036, KonFiG +0.0016, CID22-A −0.0006 (per seed
+0.0004, −0.0032, +0.0005). 12–15 live units per cell (B's own bake has 87). Thresholds and unit
count do not matter for the signed class; the 0.02 between the monotone arms and B is the sign
structure.

### Arm FR (`mono:thr:freering`, monotone in 800 of 820 inputs): one feature carries both the gap and the inversion

`d2b7ce30` trainer, 12 cells in 757 s. Exploratory (ringing was chosen after reading KADID
assessment rows).

| | E21 Δ vs `base-c`, seeds 0–2 |
|---|---|
| mean (seed-unit SE) | **−0.0039 (0.0029)**; per seed −0.0011, −0.0097, −0.0009 |
| KADID / TID2013 / KonFiG / CID22-A | −0.0046 / −0.0153 / −0.0043 / **+0.0085** |

Pair classes vs `base-c`: cross-type KADID −0.0028, TID −0.0056 (T: −0.0133, −0.0205);
cross-source KADID −0.0022, TID −0.0086; same ladder still ≥ 0 except CID22-A −0.0002. Freeing
the sign of one feature family (20 inputs) recovers about 80% of the E21 gap. The learned
ringing weights are large and negative (to −22): the model subtracts ringing.

**And the inversion comes back through the same feature.** Mixture test: 2–41 inversions per
cell, max rise +26.2, dominated by blur σ 1.2 then chroma noise σ 8 (up to 27 of 48 references)
— B's signature failure. Feature dump (`score_pair_with_bake --dump-features-to`, B's read set):
adding chroma noise to the blurred image raises the ringing slots sharply, e.g. reference 9940
slot values 0.0012 → 0.0582 and 0.0019 → 0.0349 (references 3318 and 5050 behave alike).

**Mechanism.** The v2 ringing feature responds to benign global tone and saturation changes
*and* to chroma noise. Calibrating the benign changes requires subtracting it; the same
subtraction forgives chroma noise. That one conflation explains most of the cross-type
calibration gap and B's blur + chroma-noise inversion (and plausibly the LIVE color-noise case).
*Correction (arm Bnr, below): this holds for FR, whose only free features are ringing. It does not
hold for B: B retrained without ringing keeps the same inversion, so B's comes from its sign
freedom over correlated features, not from ringing alone.*
The fix is at the feature level: a ringing measure that does not respond to global tone or
saturation changes (or the two components as separate features, with the global-change part
treated as a non-damage descriptor). Then a monotone combiner can use the damage part without
subtracting anything.

**Arm DR (`mono:thr:dropring`), recorded before its result.** T with the same 20 ringing
inputs *dropped* (zeroed every step via `--monotone-feature-mask … --monotone-strict`): exactly
monotone over the remaining 800, certificate passes, no subtraction route. Tests whether
ringing forced to count as damage is what over-penalizes the global-change types. Smoke:
certified, 20 zero input rows, no negative weights. Queued after `mp64`.

**Why ringing conflates the two** (`zensim/src/feature_v2.rs`, `idx::RINGING`): per pixel,
`saturate(|src − dst|, C_RING_ERR=0.05) · saturate(activity, C_ACTIVITY) ·
(1 − saturate(|∇src|, C_RING_EDGE=0.02))`, mean-pooled. The error term is the raw per-pixel
difference with no local mean removed, so a global tone, contrast or saturation change makes it
uniformly large off edges, and chroma noise raises it in X/B. **Proposed feature fix (not
done; a feature-system change for the owner):** compute the ringing error on a local-mean-removed
(high-pass) difference, as a new append-only slot, so low-frequency tone shifts stop
registering while noise and true ringing still do. The IW-weighted SSIM/MSE slots, 3×
over-represented on saturation changes, are the next candidates for the same check. Any such
change means new feature ids or a formula revision and re-extraction of every training table.

### Feature probe: ringing confirmed flawed; IW-SSIM and hf_loss not confirmed

`scripts/byconstruction/feature_probe.py` (24 of the mixture references; B's computed Rev5
vector via `score_pair_with_bake --dump-features-to`; median over references of each family
summed over its 12 cells; outputs `/mnt/v/output/zensim/byconstruction-2026-10-22/feature-probe/`):

| Family | mean shift +8 | contrast ×0.85 | saturation ×1.25 | saturation ×0.8 | blur σ1.2 | luma noise σ4 | chroma noise σ8 | blur + luma noise |
|---|---|---|---|---|---|---|---|---|
| ringing | **0.3336** | 0.2308 | 0.1869 | 0.1596 | 0.1389 | 0.0829 | 0.2031 | 0.1710 |
| iw_ssim | 0.0168 | 0.0533 | 0.0481 | 0.0408 | 0.2856 | 0.0344 | 0.2892 | 0.3256 |
| iw_mse | 0.2579 | 0.1837 | 0.0422 | 0.0304 | 0.1989 | 0.0218 | 0.0512 | 0.2137 |
| hf_loss | 0.0653 | 0.5329 | 0.0440 | 0.4889 | 0.8970 | 0.2048 | 0.4639 | 0.9086 |
| mse | 0.2580 | 0.1787 | 0.0369 | 0.0254 | 0.1536 | 0.0218 | 0.0508 | 0.1702 |

- **Ringing is flawed**: a barely visible +8 luma shift is its largest response in the probe,
  2.4× a visible blur and 4× luma noise; contrast and saturation edits also exceed blur.
- **IW-SSIM** responds weakly to global edits; the earlier share-of-distance excess on
  saturation was not a feature defect.
- **hf_loss** falls when luma noise is added to the blurred image on only 6 of 24 references;
  the hypothesis that it drives M's blur + noise inversions is not supported.
- `mse` / `iw_mse` respond to a mean shift by definition (they measure it); not a defect.

### Full-data M: every product gate passes, best dial numbers in this study

Full-data M seed 0 (packed by the E33 output stage; `certify-monotone`: monotone, 1 live max
pair). Gates in `/mnt/v/output/zensim/byconstruction-2026-10-22/gates-M/`:

| Gate | M (full) | `base-c` s0 |
|---|---|---|
| C1 dial monotonicity, standard / ladder | **0.9984 / 0.9935** | 0.9954 / 0.9882 |
| C2–C6, G-DIAL (both grids) | pass | pass |
| NEARID N1–N3 | pass, **143/144** monotone ladders | pass, 130/144 |
| G-STEER (D12 exact gradients) | **135/135** | 130/135 |
| Identity at every SIMD tier | 62 sources × 10 permutations pass | — |

GitHub: imazen/zensim#72 records the ringing feature defect found here.

### Feature probe, all families (`feature_probe.py --all-families`)

Median over the 24 references, each v2 family summed over its 12 cells (outputs
`feature-probe/probe_all.json`). Clipping caveat: `mean_shift+8` and `saturation_x1.25` can clip
at 255, which creates real plateaus and contours; `contrast_x0.85` and `saturation_x0.8` do not.

| Family | +8 shift | contrast ×0.85 | sat ×1.25 | sat ×0.8 | blur σ1.2 | luma noise σ4 | chroma noise σ8 |
|---|---|---|---|---|---|---|---|
| ringing | 0.334 | **0.231** | 0.187 | **0.160** | 0.139 | 0.083 | 0.203 |
| banding | 0.011 | 0.011 | 0.198 | 0.011 | 0.132 | 0.196 | 1.269 |
| blockiness | 0.008 | 0.010 | 0.109 | 0.018 | 0.135 | 0.148 | 0.619 |
| hf_gain | 0.038 | 0.022 | 0.543 | 0.035 | 0.275 | 0.288 | 1.194 |
| hf_loss | 0.065 | 0.533 | 0.044 | 0.489 | 0.897 | 0.205 | 0.464 |
| pjnd_transducer | 0.705 | 0.494 | 0.376 | 0.314 | 0.357 | 0.151 | 0.407 |
| transducer_lowk | 0.801 | 0.575 | 0.398 | 0.330 | 0.399 | 0.169 | 0.425 |
| mse | 0.258 | 0.179 | 0.037 | 0.025 | 0.154 | 0.022 | 0.051 |
| edge_width | 0.000 | 0.000 | 0.036 | 0.038 | 0.181 | 0.003 | 0.913 |

Only **ringing** responds to clipping-free global edits more than to a visible blur, against
its purpose (filed as imazen/zensim#72, with this check added as a comment). banding,
blockiness and hf_gain spike only on the clipping-prone saturation-up edit; the transducers'
response to contrast reduction tracks a visible change; hf_loss under contrast/saturation
reduction and mse under a mean shift are what those measures define. Not filed.

### Arm DR (`mono:thr:dropring`): dropping ringing does not help

E21 Δ **−0.0216 (0.0031)**; KADID −0.0192, TID −0.0447, KonFiG −0.0280, CID22-A +0.0053 (T −0.0205;
FR −0.0039). Ringing forced to count as damage is not what over-penalizes the global-change
types; the mse-type, transducer and hf_loss families do that. What ringing gives a signed model
is an accidental **global-change detector**, which it subtracts to discount those other
features (FR's large negative ringing weights). Fixing ringing alone would remove that signal. DR's mixture
test: 0–3 inversions per cell, max +4.26, dithering only (like T).

**Revised design.** (1) Make ringing measure ringing (high-pass error), closing the chroma-noise
loophole. (2) Add an explicit global-change descriptor, zero at identity and blind to noise —
e.g. per channel, the fraction of the error explained by a global affine map `dst ≈ a·src + b`
(≈1 for mean shift, contrast or saturation, ≈0 for noise or blur) — and let it *reduce*
damage gains through the existing gated-product structure (gain ≥ 0). The model stays monotone
in every damage feature, forgives global changes explicitly, and noise cannot exploit the
descriptor because noise is not affine-explainable. Posted to imazen/zensim#72 as a correction.

**Prior art for the revised design (zenpapers corpus).** The decomposition is the classical
one: SSIM factors local similarity into luminance, contrast and structure, the structure term
being invariant to local affine luminance/contrast changes (Wang et al. 2004,
doi:10.1109/TIP.2003.819861; multiscale 2003, doi:10.1109/ACSSC.2003.1292216); CW-SSIM is
designed to be insensitive to luminance and contrast changes (2009, doi:10.1109/TIP.2009.2025923);
MAD separates detection-based from appearance-based judgment (Larson & Chandler 2010,
doi:10.1117/1.3267105). Applied here: compute the damage bank on the distorted image after
removing the best global (or low-frequency) affine tone map per channel — the residual damage —
plus a small set of global-change magnitude features, and combine all of them monotonically.
Global changes are then discounted by a small learned weight on their own terms, with nothing
subtracted.

**Arm DG (`mono:thr:dropglobal`), rule fixed before its result.** Exactly monotone (strict
mask) with every damage family dropped whose probe response to a mild or clipping-free global
edit exceeds its response to a visible σ1.2 blur: mse, masked_mse, iw_mse, pjnd_transducer,
transducer_lowk, transducer_highk, ringing (`masks/global_sensitive.tsv`). The selection uses
only the label-free probe, not assessment labels. A cheap proxy for "structure-only residual
damage"; if it beats T clearly, the affine-residual feature design is worth building.

### Arm DG (`mono:thr:dropglobal`): removing the global-sensitive families does not help

E21 Δ **−0.0229 (0.0032)**; KADID −0.0198, TID −0.0478, KonFiG −0.0304, CID22-A +0.0064 (T −0.0205,
DR −0.0216). The mse-type, transducer and ringing families still carry information a monotone
model needs (for noise, for example); removing them costs more than their over-penalization of
global changes. The calibration cannot be fixed by subtracting families from the bank: it needs
the same measures recomputed on the affine-residual difference, or an explicit discount signal.

### Arm M64 (`mono:thr:mp64`, all units max pairs)

E21 Δ **−0.0131 (0.0023)**; KADID −0.0166, TID −0.0215, KonFiG −0.0150, CID22-A +0.0005 (per seed
−0.0111, −0.0177, −0.0107). Certified monotone; 1–2 live max pairs in 11 of 12 cells (one cell
keeps all 128 units). Mixture test 1–6 inversions per cell, max +7.56. Within noise of M (−0.0144).

### Arm Bnr (`fx1nr`): B without ringing, signed

Owner question (13:00Z): is ringing essential, and what does doing it right cost? Bnr is B's signed
recipe (`sel:c6a94f8337b0@h32:H128:cv16:cf98:fx1nr`) with the 10 ringing ids removed from both legs:
400 direct inputs plus 400 fragility products (`scripts/rev4_featpot/byc_fx1nr_declaration.json`,
sha256 `923da70c…`). Everything else, including the signed first layer, is B's.

E21 Δ **−0.0003 (0.0009)**; KADID −0.0010, TID −0.0023, KonFiG +0.0041, CID22-A −0.0019 (per seed
+0.0008, +0.0002, −0.0020); E33 point form passes. B does not need ringing for its rank
performance: the signed model finds the same global-change discount through other features.

Mixture test: 72–178 inversions per cell (`base-c` 86–171), max rise +53.1 (`base-c` +61.5). The
worst combination is still blur σ1.2 + chroma noise σ8, which inverts on 28–48 of 48 references per
cell, followed by blur + chroma σ16, blur + luma σ4 and posterize + chroma noise. Removing ringing
does not remove B's inversion; sign freedom does. In the monotone class, ringing's only value was
as a subtractable global-change signal (FR −0.0039 vs DR −0.0216).

### Cost of the ringing fix, and of dropping ringing

Bench: `byc_score_timing` (new, `zensim-validate/src/bin/`) times `BakeScorer::compute` (score only,
no diffmap) on one deterministic textured pair with uniform ±3 noise. Settings: single thread
(`RAYON_NUM_THREADS=1`), pinned to core 2, default dispatch on dev (Zen 5). Each round is the median
of 40 calls (512²) or 8 calls (2048²) after 3 warmups; 12 rounds alternate the order of the two arms
(`scripts/byconstruction/ring_timing.sh`). Paired Δ is the median of the per-round ratios, with a
95% bootstrap CI (5,000 resamples). Raw rounds and hashes are in
`benchmarks/byconstruction_timing_2026-10-22/`.

**High-pass ringing prototype** (`RUSTFLAGS=--cfg byc_hp_ring`, all five ringing sites in
`zensim/src/feature_v2_kernels.rs`). The ringing error becomes `(src − mean4(src)) − (dst − mean4(dst))`
over the 4-neighbourhood the kernel already loads, instead of `src − dst`, so a global offset or
a smooth tone map no longer counts as ringing. Both builds score B's served bake (`89cb4861`):

| Size | Baseline median | High-pass median | Paired Δ (95% CI) |
|---|---|---|---|
| 512² | 5.979 ms | 6.021 ms | +0.78% (+0.62..+0.92) |
| 2048² | 87.016 ms | 87.263 ms | +0.24% (−0.39..+0.71) |

**Dropping ringing from the model** (baseline binary only; `base-c` kadid_s0 bake `7bd14f56` vs
Bnr kadid_s0 bake `3b7af590`):

| Size | `base-c` median | Bnr median | Paired Δ (95% CI) |
|---|---|---|---|
| 512² | 6.017 ms | 6.017 ms | −0.03% (−0.08..+0.09) |
| 2048² | 86.658 ms | 86.739 ms | +0.07% (−0.11..+0.41) |

Ringing is computed in the shared gradient kernel whether or not a bake reads it, so dropping it
saves nothing measurable, and fixing it costs under 1% single-threaded.

Limits:
- Load average was 3–8 during the runs, decaying from my own finished jobs; no foreign build or
  training ran.
- Tested on one synthetic content type, one host and one dispatch tier. Not measured
  multi-threaded.
- The prototype changes only the extraction kernel, not the spatial/diffmap sites in
  `feature_v2.rs`.
- It changes ringing values (512² timing pair: 88.2805 → 88.2424), so it is a new arithmetic
  revision. No stored table or bake is valid with it.
- Whether it improves accuracy is not measured; that needs re-extracted training tables and refits.
- The cost of the affine-residual design is not measured; whether each design streams is in the
  next section. (An earlier version of this note said a coarse-scale fit would avoid a pre-pass.
  It would not: the streaming walk builds coarse rows from fine rows as they arrive.)

### Can the fixes stream?

The Rev5 scorer is a single streaming pass. `StripPlaneProducer` (`zensim/src/feature_v2_stream.rs`)
keeps rolling row windows per scale and builds each coarser scale from finer rows as they arrive.
It emits 128-row kernel strips top to bottom, and per-strip accumulators merge in a fixed reduction
tree, so strip geometry does not change the result. The timings above ran through this walk: the
raw 2048² profile (`perf.data`, see the timing README) has samples in `StripPlaneProducer`,
`RollingPlane` and `stream_phase_b`.

| Design | Streams? |
|---|---|
| High-pass ringing | Yes, unchanged: the same 4-neighbourhood inside the same strip halo. |
| Global-change descriptor used as a non-negative gate | Yes, in the same pass. The affine fit `dst ≈ a·src + b` per channel, and the share of the error it explains, need only five sums (Σs, Σd, Σs², Σd², Σs·d) that add across strips in the existing reduction tree. The descriptor is final when the walk ends, which is when the model runs anyway. |
| Damage bank recomputed on the global affine residual | Not in one pass. The fit must exist before the damage kernels run, and every scale's rows arrive top to bottom. It needs a moment pre-pass. |
| Damage bank on a local affine residual (window means and variances, as in SSIM's structure term) | Yes, within the existing halo. |

Notes:
- The steering map would need the gate descriptor's gradient as an extra term; whether the map's
  retained per-strip data covers it is not checked.
- A pre-pass keeps memory bounded: `ImageSource` gives random access to borrowed rows, so it can
  accumulate moments without materializing planes. Its cost is not measured. It would repeat the
  sRGB→XYB conversion, which took 13.2% of all samples in a single-thread 2048² profile of this
  scorer (`benchmarks/byconstruction_timing_2026-10-22/perf_2048_symbols.txt`; about 8% of the
  samples are the benchmark's own image synthesis). A subsampled pre-pass converts proportionally
  fewer pixels. In reference-fed encoder loops the reference pyramid is already cached, so only the
  distorted side would need it.
- A fit taken from the strips seen so far is not an option: the top of the image would use a
  different fit than the bottom.
- Whether a local residual can share the SSIM pass's local moments is not checked.

### Ringing corrected (`ZENSIM_RING_FORM=hpm`): implementation, cost, retraining

Owner direction (2026-10-23): correct ringing with performance parity, train and test, then look for the next
feature improvement.

**What changed** (`08930023`, `980fde3f`). `RingForm` in `zensim/src/feature_v2.rs` is a per-process
measurement arm, selected with `ZENSIM_RING_FORM` and modelled on `ZENSIM_HF_GAIN`. `raw` (the default)
keeps every stored value. `hpm` measures the ringing error as `(src − mu1) − (dst − mu2)`, where `mu1`/`mu2`
are the SSIM local means the dense kernel already holds for the strip. The three saturations are unchanged.
The form is a const generic on the canonical gradient kernels, so the raw instantiations compile to the
previous code. The diffmap and both attribution passes follow the form. Local refinement refuses `hpm`,
because a local change moves the local means over the blur radius rather than the ±1 dilation it patches.
The first version (`hp4`, `08930023`) used the 4-neighbour mean instead; it cost about eight vector
operations per 8 pixels and is superseded.

Checks:
- B's 512² timing pair scores 88.280549 under `raw` with the old and new binaries, and 88.281888 under `hpm`.
- Vector against scalar canonical body is bit-identical on scalar, v3, v4 and v4x for both forms
  (extended `gradient_canon64_vec_matches_scalar_on_every_compiled_tier`). New tests cover offset
  invariance, noise response and an independent f64 per-pixel sum. `cargo test --release -p zensim
  --all-features`: 907 passed, 0 failed, 27 ignored. CI-exact clippy clean.
- Rev5 extraction under `raw` with the new extractor reproduces the stored bank on all 576 requested
  slots (600 safesyn rows across jpg/webp/avif/jxl, 600 cid22_a25 rows). Under `hpm`, every one of the
  eight training/assessment sets differs from the stored bank in exactly the 12 ringing slots.

**Cost, and a 9% speedup found on the way.** All timings: `byc_score_timing` (B's served bake, deterministic
textured pair), paired rounds with alternating order, pinned core(s), dev (Zen 5); raw rounds in
`benchmarks/byconstruction_timing_2026-10-23/`.

1. The first form (`hp4`, 4-neighbour mean, `08930023`) added about eight vector operations per 8 pixels: +0.97%
   at 512² and +0.55% at 2048² single-thread against the old binary (folding three divisions into one saved nothing).
2. The `hpm` form measured +3.0% (512²) and +3.5% (2048²) against the old code, and that was robust: five lld
   link-order shuffles (`--shuffle-sections`, SD 0.1–0.2 points), uniform function alignment 2^5–2^7, and
   environment padding all left it in place. The phase timer (`ZENSIM_FOLD_TIMING`) put all of it in the dense
   kernel (+10–14%), whose machine code is byte-identical; exact counts showed the same instructions (−0.3%), +4.5%
   cycles and +40% op-cache misses. Bisection on the source found the trigger: passing the `mu` planes to the
   gradient call at all (a literal `None` there was at parity across shuffles), even though the dense kernel runs
   first and reads the same data.
3. Mechanism: `Lanes16F64` (`[f64; 16]`, 8-byte aligned) holds the dense kernel's accumulators, which it adds in
   64-byte chunks every few pixels; where the array falls relative to a cache line depends on the caller's frame.
   `#[repr(align(64))]` on `Lanes16F64` and `LanesF64` (`ec764cd7`; values unchanged, scores bit-identical, full
   suite passes) made the **old** code 9.32% faster at 2048² (CI −9.63..−8.98) and removed the sensitivity.

Final measurement at `ec764cd7`, five lld link-order shuffles per comparison (`scripts/byconstruction/final_shuffle.sh`,
mean paired Δ, SD over seeds), single thread unless noted:

| Comparison | 2048² | 512² | 64² (seed 1) | 2048², 8 threads (seed 1) |
|---|---|---|---|---|
| shipped code → corrected, `hpm` | **−8.45% (0.38)** | **−7.45% (0.10)** | −1.40% (CI −1.49..−1.28) | −6.49% (CI −6.85..−4.90) |
| shipped code → corrected, `raw` | −8.85% (0.21) | | | |
| aligned old code → corrected, `hpm` (the ring form's own cost) | +0.49% (0.21) | +0.57% (0.19) | +1.05% (CI +0.75..+1.23) | +0.07% (CI −0.34..+1.69) |

So the corrected scorer is faster than the shipped one at every measured size and thread count, because of the
alignment fix; the ring form by itself costs about 0.5% (1% at 64²), positive on every shuffle seed. Single host,
one synthetic content type, x86 v4x dispatch only.

Lesson for timing this code: whole-walk comparisons between two builds can move 3–10% with nothing changed in the
kernel that moves. Use several `--shuffle-sections` seeds, exact `perf stat` counts and the phase timer before
crediting or blaming a change. Uniform function alignment does not randomize placement (relative order is kept).

**Training root.** `scripts/byconstruction/ring_bank.sh` re-extracted every set of the v40r4 `v2e29` root
(kadid train/select, tid2013, konfig train/val, cid22_a25, cid22_train, safesyn; 1,037 s on 24 threads,
peak RSS 7.05 GiB) and the E15 coverage-pool selection (44,000 pairs) under `hpm` with the pinned
decoders. `scripts/byconstruction/ring_root.py build` wrote a derived root
(`/mnt/v/output/zensim/byconstruction-2026-10-22/ringhpm/root`). In every table, all 564 non-ringing
measured columns are bit-identical to the stored root, and only the 12 ringing columns change. The
derived feature-set id `basic+peaks+v2@w1825/rev5_localwin_ringhpm#36c3f3af` is appended to the registry.
Strict admission (Python and the Rust trainer) accepts it only on declarations that record
`byc_derived_from`.

**Feature probe under `hpm`** (`feature_probe.py --all-families`, same 24 references; ringing summed over its
12 cells, median over references):

| | +8 shift | contrast ×0.85 | sat ×1.25 | sat ×0.8 | blur σ1.2 | luma noise σ4 | chroma noise σ8 |
|---|---|---|---|---|---|---|---|
| ringing, raw | 0.334 | 0.231 | 0.187 | 0.160 | 0.139 | 0.083 | 0.203 |
| ringing, `hpm` | **0.017** | 0.130 | 0.075 | 0.063 | 0.143 | 0.082 | 0.185 |

Every other family reproduces the earlier probe values exactly.

**Retrained on the derived root** (`scripts/byconstruction/ring_arms.sh`, trainer `35e90254`, 12 E21 cells per arm,
seeds 0–2; scored against E34 `base-c` by `byc_cells.py score`; mixture test on the same 1,152 steps, scored under
`ZENSIM_RING_FORM=hpm` with the `35e90254` scorer, which reproduces all 1,344 stored raw Bnr mixture scores exactly
under `raw`):

| Arm | E21 Δ (SE) | KADID | TID2013 | KonFiG | CID22-A | Mixture inversions / cell (max rise) |
|---|---|---|---|---|---|---|
| Bhpm: B's signed recipe | **−0.0002 (0.0023)** | −0.0037 | −0.0026 | +0.0045 | +0.0009 | 78–184 (+53.8) |
| FRhpm: monotone, ringing sign-free | −0.0042 (0.0029) | −0.0067 | −0.0027 | −0.0131 | +0.0058 | 6–70 (+35.8) |
| Mhpm: certified monotone max pairs | −0.0158 (0.0017) | −0.0159 | −0.0302 | −0.0183 | +0.0014 | 2–35 (+8.0) |
| for reference: Bnr | −0.0003 (0.0009) | −0.0010 | −0.0023 | +0.0041 | −0.0019 | 72–178 (+53.1) |
| for reference: FR (raw ringing free) | −0.0039 (0.0029) | −0.0046 | −0.0153 | −0.0043 | +0.0085 | 2–41 (+26.2) |
| for reference: M (raw ringing) | −0.0144 (0.0016) | | | | | 1–32 (+6.5) |

- B's recipe is at parity on the corrected feature (E33 point form passes) and keeps its inversion: blur σ1.2
  then chroma noise σ8 inverts on 48 of 48 references in every cell, as in `base-c` and Bnr.
- In the certified-monotone class the correction alone does not move E21: Mhpm −0.0158 against M −0.0144, under one
  seed-unit SE apart. Its mixture behaviour is unchanged too (worst: blur σ1.2 then luma σ4, small rises).
- Freeing the corrected ringing still recovers most of the monotone gap (T −0.0205) and still opens the
  chroma-noise route (blur + chroma σ8 inverts on up to 38 references). The high-pass error still responds to
  noise, so a sign-free ringing input can still be used to forgive it. Inside the monotone class ringing has to
  count as damage; the sign-free channel needs a signal noise cannot drive.


### Next feature: a global affine-change descriptor (`byc_globaff`)

**Why this one.** Under `hpm` the probe flags one remaining family by the same criterion (response to a
clipping-free global edit above a visible blur): the JND transducers, which also use the raw per-pixel error
(`pjnd_transducer` contrast ×0.85 0.494 vs blur 0.357; +8 shift 0.705). Error-magnitude families measure global
changes by definition, and DG showed dropping them costs more than it saves. FR and FRhpm show the monotone class
recovers most of its gap when it gets a sign-free signal it can subtract, and that ringing is the wrong such signal
because noise drives it. So the next feature is an explicit descriptor that a global change drives and noise does
not: per scale and XYB channel, the share of the mean squared error that a least-squares global fit
`dst ≈ a·src + b` explains, and that explained error (24 values). It streams in one pass (five additive sums per
channel; see "Can the fixes stream?").

**Implementation (research only).** `zensim-bench/examples/byc_globaff.rs` decodes with the shared decode owner,
converts with zensim's own XYB conversion and builds the 4-scale pyramid with zensim's own 2× downscale (two-pass f64
moments). It is a sidecar at research ids f1867–f1890, registered as
`basic+peaks+v2@w1891/rev5_localwin_ringhpm_globaff#d56ad477` (hash from the new `byc_fsid` bin, which reproduces the
registered `#36c3f3af`); zensim cannot compute these ids, so such bakes cannot be served. Training reads them from
a widened derived root (`ring_root.py globaff`); predictions use `byc_predict` (zenpredict, the trainer's row
layout), which reproduces all 7,869 stored `bake_dial_refit --score-units` predictions of a Bhpm cell bit for bit;
the mixture test uses a table of dumped feature vectors plus the descriptor, which reproduces the pixel-route
scores of a Bhpm cell exactly on all 1,344 images. 271,590 training pairs plus the pool: 482 s on 20 threads.

**Label-free check on the probe images** (median share at scale 0, X / Y / B, 24 references):

| Edit | Share |
|---|---|
| +8 luma shift | 0.37 / **0.997** / 0.22 |
| contrast ×0.85 | 0.23 / **0.997** / 0.16 |
| saturation ×1.25 | **0.97** / 0.35 / **0.94** |
| saturation ×0.8 | **0.99** / 0.41 / **0.97** |
| blur σ1.2 | 0.17 / 0.16 / 0.15 |
| luma noise σ4 | 0.003 / 0.000 / 0.004 |
| chroma noise σ8 | 0.003 / 0.14 / 0.002 |

(At the coarsest scale chroma noise leaves a tiny Y error that is mostly "explained", share 0.85 but explained
energy 4e-6; that is why both numbers are inputs.)

**Arms** (`globaff_arms.sh`, `globaff_arms2.sh`; trainer and scorers `58a4d100`; 12 E21 cells each):

| Arm | Class | E21 Δ (SE) | KADID | TID2013 | KonFiG | CID22-A | Mixture inversions / cell (max rise) |
|---|---|---|---|---|---|---|---|
| **Tghpm** | monotone in all 820 damage inputs, descriptor sign-free | **−0.0101 (0.0024)** | −0.0066 | −0.0236 | −0.0175 | +0.0071 | **2–7 (+7.1)** |
| Thpm (control) | monotone, no descriptor | −0.0221 (0.0032) | −0.0192 | −0.0452 | −0.0296 | +0.0055 | 0–4 (+4.2) |
| Bghpm | B's signed recipe plus the descriptor | −0.0031 (0.0017) | −0.0010 | −0.0033 | −0.0044 | −0.0037 | 70–190 (+61.4) |

- The descriptor recovers about half of the monotone class's E21 gap (Thpm −0.0221 → Tghpm −0.0101) **without** the
  chroma-noise route: Tghpm's only inversions are posterize then light noise (dithering, at most 3 references per
  cell), the same benign pattern as T. Sign-free ringing recovered more (FRhpm −0.0042) but inverts blur + chroma
  noise on up to 38 of 48 references. Tghpm has the best E21 of the arms that keep inversions bounded, ahead of the
  certified-monotone M64 (−0.0131, 1–6 inversions) and M (−0.0144, blur + luma σ4 on up to 24 references); its
  0.003 margin over M64 is about one seed-unit SE.
- Tghpm's guarantee is weaker than M's: it is monotone in every damage input, but the descriptor is free, so the
  monotonicity is not certified end to end. The descriptor is built to fall when unexplained error is added, and
  on the 1,152 mixture steps that held.
- Adding the descriptor to B's signed recipe does not help (−0.0031, E33 point form fails) and leaves B's
  inversions (blur + chroma σ8 on 48 of 48 references).
- Remaining gap: TID2013 −0.0236 and KonFiG −0.0175. Not tested: combining the descriptor with max pairs (the
  free-mask route refuses `mp`), gating products instead of sign-free inputs, and a served (streaming) version.

### AIC2026 TRAIN: how far image features go toward SROCC 0.99 (2026-10-23)

**Goal (owner, 2026-10-23):** convex, monotone scoring across every distortion type and dataset, and SROCC above
0.99 on AIC2026 TRAIN, with less compute and without memorization or overfitting.

**Owner decisions later the same day.** (1) The SROCC 0.99 target is dropped; the rest of the goal stands. (2) Scale 0
is mandatory: no candidate drops the finest scale, so Ts1a and the `fx1gs1` read set are out, and the 41–46% instruction
saving measured below applies to no candidate. (3) Smoothness counts as much as inversions: scores should follow human
ratings on a smooth curve, judged by error statistics as well as rank ("Smoothness" below).

**Population.** AIC2026 TRAIN is WG1's `public_train` split (document m112107, preliminary public-split JND
reconstruction, CSV SHA-256 `bd5ea3c5…`, tower IQA library `jpeg-aic2026`): **3,920 PTC crops of 28 sources**,
label `JND_score` (lower is better; quality SROCC uses −JND). `public_test` is the other 1,848 crops of 14 sources;
28 further sources have no labels. The first half of this study used the owner's separately supplied overlay
`jnd_reconstruction_2026-10-19.csv` (`JND_curve`) over all 42 labelled sources before the split was found; those
results are kept below and labelled "42-source pool". `JND_curve` and `JND_score` are two reconstructions of the
same study and agree at SROCC **0.9835** on `public_train` (0.986 on `public_test`); their difference grows with the
distortion level (σ 0.07 JND at level 1, 0.50 at level 20; `JND_score` is higher at the top).

**Role and exposure.** The owner's goal names the split TRAIN, so this branch fits on it, and **every AIC2026
number reported is either a frozen model (no fit on AIC2026) or out of fold** (sources split into folds; a source's
rows are predicted by a model fitted without any row of that source); in-sample numbers are never reported. The
42-source pool phase fitted the `JND_curve` labels of the 14 `public_test` sources inside its folds, so `public_test`
informed this study's design choices and is not a clean test for them; it is read only to assess frozen candidates.
The 28 unlabelled sources and every other protected population stay untouched. Ledger: `docs/DATA_SPLITS.md`,
2026-10-23. All 5,810 crop files on dev were SHA-256-equal to the 2026-10-19 admission before any use.

Tooling: `mixture_table.py build-pairs` (Rev5 vector + descriptor table; the table route reproduces all 5,768
stored B scores bit for bit), `aic_train.py` (zero-shot, out-of-fold fits, `analyze`), `jnd_band_proto.py` (band
feature prototypes), `aic_rust_cv.py` (the same folds through the Rust trainer), `byc_cells.py --aic-weight` (E21
cells that also fit the split). Data: `/mnt/v/output/zensim/byconstruction-2026-10-22/aic2026train/` on dev
(`folds_train/`: the split, its 7 source folds, pairs lists and the WG1 CSV).

**`public_train`, WG1 `JND_score`:**

| Model | Fit on AIC2026 | Global SROCC | Within source | Within ladder |
|---|---|---|---|---|
| B (`diff+frag-820`) | no | 0.9091 | 0.9558 | 0.9995 |
| full-data M (certified monotone) | no | 0.9108 | 0.9508 | 0.9998 |
| full-data T (convex monotone) | no | **0.9115** | 0.9494 | 0.9997 |
| NNLS linear on fx1g (844 inputs) | out of fold, 7 folds × 3 | 0.934 | 0.959 | 0.999 |
| additive hinges, **convex** monotone | out of fold | 0.939 | 0.963 | 0.997 |
| additive hinges + ramps, monotone | out of fold | 0.939 | 0.967 | 0.999 |
| gradient boosting, monotone | out of fold | 0.953 | 0.964 | 0.996 |
| gradient boosting, unconstrained | out of fold | 0.957 | 0.967 | 0.997 |
| fx1g + all 75 supplied peer metrics, monotone boosting | out of fold | 0.964 | — | — |
| fx1g + all 75 supplied peer metrics, unconstrained boosting | out of fold | **0.968** | — | — |

**Through the Rust trainer** (`aic_rust_cv.py`, AIC2026 alone, global rank loss on the split, 60 epochs × 20k pairs,
same 7 folds; `58a4d100` trainer): certified convex monotone **T 0.9527**, convex monotone max pairs M 0.9512, signed
S 0.9566 (within source 0.968 / 0.968 / 0.969). The servable convex monotone class matches monotone boosting.
**Memorization gap:** T fitted on all of `public_train` scores 0.9743 on it in sample, 0.022 above its out-of-fold
0.9527 (source-level offsets it can learn but not transfer). Even in sample this class does not reach 0.99.

The last two rows are the measured information ceiling of every full-reference measurement available here (our 844
inputs plus 75 published metrics, DVIFM excluded because its constants were trained on AIC2026): **0.968, 0.022
below the target**, and the two label reconstructions agree with each other only at 0.9835.

**E21 cells that also fit `public_train`** (`aic_arms.sh`; `byc_cells.py --aic-weight W`): every E21 cell adds an
AIC2026 group (global rank loss over the split, research replay) without one of the 7 source folds and predicts that
fold; seeds 0–2 cover every fold (`aicscore` pools the held-out predictions; a fold held out by two cells takes their
mean). E21 Δ is against `base-c` as everywhere in this record; mixture test as above (1,152 steps per cell).

| Arm | Class | AIC weight | E21 Δ (SE) | KADID / TID / KonFiG / CID22-A | AIC `public_train` OOF | Mixture inversions / cell (max rise) |
|---|---|---|---|---|---|---|
| Tghpm (no AIC leg) | monotone in damage inputs, descriptor sign-free | — | −0.0101 (0.0024) | −0.007 / −0.024 / −0.018 / +0.007 | — | 2–7 (+7.1) |
| **Tga** | same | 32 | **−0.0089 (0.0038)** | −0.012 / −0.023 / −0.011 / +0.010 | **0.932** (within source 0.958) | **1–4 (+4.1)**, dithering only |

| Ma | certified convex monotone max pairs (descriptor constrained) | 32 | −0.0172 (0.0053) | −0.017 / −0.039 / −0.019 / +0.007 | 0.934 (within source 0.958) | 2–5 (+6.3) |
| Bga | B's signed recipe | 32 | −0.0031 (0.0049) | −0.006 / −0.005 / +0.002 / −0.003 | 0.949 (within source 0.968) | 52–211 (+62.6) |
| Ta | certified convex monotone hinges (descriptor constrained) | 32 | −0.0221 (0.0045) | −0.026 / −0.043 / −0.029 / +0.009 | 0.932 | 1–3 (+4.4) |
| **Ts1a** | as Tga, **no scale-0 input** (`fx1gs1`) | 32 | −0.0099 (0.0039) | −0.010 / −0.026 / −0.013 / +0.010 | 0.928 | 0–4 (+3.7) |
| Mgfa | max pairs with the descriptor sign-free in the v rows (trainer `805242d9`) | 32 | −0.0166 (0.0046) | −0.017 / −0.027 / −0.029 / +0.007 | 0.932 | 1–6 (+5.9) |
| Tga128 | as Tga, AIC weight 128 | 128 | −0.0107 (0.0046) | −0.015 / −0.029 / −0.013 / +0.015 | **0.935** | **0–2 (+1.4)** |

**Candidate by the committed rule: Tga.** Eligible (at most 10 inversions per cell, E21 Δ ≥ −0.010): Tga (AIC 0.932) and
Ts1a (0.928). Tga128 has the best monotone AIC (0.935) and the fewest inversions, but its E21 Δ −0.0107 misses the bar by
0.0007, so it is not picked; it stays the closest alternative. Max pairs (Ma, Mgfa) cost E21 here without an AIC gain.

**Correction (2026-10-23): who set this rule.** Its thresholds (at most 10 mixture inversions per cell, E21 Δ ≥ −0.010)
were written in this session and committed (04743622) before the last four arms reported; the owner did not approve
them. The registered E21 bar is E33's point form ("Decision rules" above: mean Δ ≥ −0.002 and every dataset Δ ≥ −0.005).
No arm in this record meets it, signed or monotone (Bga: mean −0.0031, KADID −0.006). Tga is the session's best balance
of rank, inversions and AIC, not a qualified candidate; Ts1a is out (scale 0 is mandatory).

Dropping every scale-0 input costs Tga 0.001 on E21 (inside one SE) and 0.004 on AIC public_train, for 21–25% fewer
instructions per score before any engine change (see Compute). On these development numbers the finest scale costs
little; that is not yet a clean result (see "Clean evaluation status").

The signed class takes the AIC leg best (Bga: E21 at parity with `base-c`, AIC 0.949) and keeps B's inversions
(blur + chroma noise); the monotone arms give up about 0.015 on AIC and 0.006–0.014 on E21 to stay inversion-free.

**Candidate rule, committed (04743622) before the results of Ta, Ts1a, Mgfa and Tga128.** One combined arm becomes
the frozen candidate: among arms with at most 10 mixture inversions per cell and E21 Δ ≥ −0.010 against `base-c`, the
highest AIC public_train out-of-fold SROCC; ties (within 0.002) go to the better E21 Δ. Its full-data fit
(`aic_full.sh`, seed 0) is then read once on public_test and once in sample on public_train. No other model is read on
public_test.

**Candidate reads (development numbers, not clean).** Tga fitted once on every E21 TRAIN leg plus all of
public_train (`aic_full.sh`, seed 0, trainer `58a4d100`, bake SHA-256 `2664b4c7…`): public_test 0.9365 (PLCC 0.921,
KROCC 0.788, OR 0, PWRC 0.991, Z-RMSE 0.389; B on the same rows 0.9255), public_train in sample 0.9411 against 0.932 out
of fold; mixture test 3 of 1,152 steps raise the score, at most +1.95 (posterize then light noise). The public_test
number is **not** evidence of generalization: its 14 sources were inside the first phase's fits (overlay labels).

**Clean evaluation status (2026-10-23).**
- *Clean:* the 28 unlabelled AIC2026 sources (no labels exist yet); every protected holdout not named in DATA_SPLITS'
  2026-10-23 ledger (AIC-3, the AIC-4 sample, CID22-B, KonJND JPEG, CSIQ, KonFiG test, secret holdouts).
- *Exposed, development only:* AIC2026 public_test (fitted under the overlay labels in the first phase; read once for
  frozen B and once for this candidate) and public_train (all selection in this study used it).
- *Not evidence for a claim:* any AIC2026 number in this section beyond "how a model ranks the labels it was developed
  against". Out-of-fold numbers are clean of fitting but not of selection: the arms, inputs and model classes were chosen
  by looking at the same sources, so the chosen arm's score is optimistic by an unknown amount. E21 is likewise an
  exposed TRAIN-population statistic (see the top of this record).
- A clean claim needs one frozen candidate read once on data no step has touched, under a rule written first.

**Contamination audit (AIC2026 / AIC-4), 2026-10-23.** The owner states no further datasets will exist, so AIC2026's
public_test cannot be replaced; the standard is containment.
- *No zensim feature was tuned on AIC2026, and no AIC-4 data was read this session.* Feature definitions, kernels and
  constants are unchanged; the per-scale skip is bit-identical on every read slot, and the max-pairs change is a model
  class. The candidate's inputs (fx1g: B's fx1 plus the byc_globaff descriptor) were fixed in earlier sessions.
- *Informed by AIC2026 including the 14 public_test sources (overlay labels, first phase):* the Python feature
  prototypes (band errors, tile pooling, gain/loss, saliency; all rejected, none in any model), the scale-0-free input
  set `fx1gs1` (only in Ts1a; to adopt it, rely on its E21 result alone), the information-ceiling conclusion, and the
  general picture of which contents and codecs fail.
- *Trained on AIC2026 labels, as the goal directed:* the combined arms and the candidate (public_train only); the
  first-phase research cells also fitted the public_test sources' overlay labels and are quarantined.
- Quarantine list: `/mnt/v/output/zensim/byconstruction-2026-10-22/aic2026train/QUARANTINE.md` on dev.

Tghpm's 12 cells (no AIC leg) score 0.907 on `public_train` zero-shot (range 0.904–0.909, as B's 0.909): the AIC leg
is worth +0.025 out of fold. Per codec, Tga beats B on 13 of 14 codecs out of fold (JPEG XL modular 0.801 → 0.833,
JPEG 2000 0.890 → 0.932, WebP 0.908 → 0.950; VVC 0.978 → 0.967), and it is more even than the AIC-only T, which
loses VVC (0.908), PNG and HEVC: the E21 legs act against fitting the five common codecs.

**Mohammadi 2025 panel per dataset and split** (`aic_panel.py`, native `panel --batch`; JSON
`aic2026train/panel_mohammadi.json`): for each E21 set, out of sample = the LODO cells trained without it (seeds 0–2,
mean), in sample = a seed-0 cell trained with it. In-sample minus out-of-sample SROCC, B's recipe (`base-c`): TID2013
0.940 vs 0.901, KonFiG 0.875 vs 0.817, CID22-A 0.881 vs 0.880; Tga: TID2013 0.890 vs 0.879, KonFiG 0.819 vs 0.807, CID22-A
0.892 vs 0.890. The monotone models gain less from seeing a dataset than B's recipe does. AIC2026 public_train, T: in
sample 0.974 / out of fold 0.953 (PLCC 0.964 / 0.933, Z-RMSE 0.265 / 0.359).

The same panel for every set, in-sample minus out-of-sample SROCC (KADID / TID2013 / KonFiG / CID22-A): `base-c`
+0.050 / +0.039 / +0.058 / +0.001; M64 +0.038 / +0.019 / +0.027 / −0.002; M64lin +0.024 / +0.032 / +0.037 / −0.004;
Thpm +0.023 / +0.028 / +0.010 / −0.002; Tga +0.022 / +0.011 / +0.012 / +0.002. Out of sample, `base-c` still leads M64
on the synthetic sets (0.891 / 0.901 / 0.817 against 0.874 / 0.880 / 0.802) and ties it on CID22-A (0.880 both). Over
the three synthetic sets the constraint cuts what a model gains from seeing a dataset by about 40% (mean +0.049 for
`base-c`, +0.028 for M64), while `base-c`'s mean out-of-sample lead of 0.018 shows that part of what its signs buy
does transfer (cross-type calibration).

**AIC2026 test data used (counted from the commands run, 2026-10-23).** Before the split was found, the 14 public_test
sources were inside the 42-source pool with the overlay labels: 1 frozen-model read, 10 descriptive label reads, 32 Python
source-fold runs and 4 Rust trainer runs that fitted and scored them. After the split was found, the official public_test
labels were read for frozen B once (SROCC 0.9255, with the reconstruction-agreement check), once more for the panel on the
same predictions, and once for per-source label statistics that were then left off the dashboard. No fit, model choice
or candidate read used them.

**42-source pool (`JND_curve` overlay; the first phase).**

**Zero-shot (no AIC2026 fit), global quality SROCC / mean within reference / mean within ladder:**

| Model | Global | Within reference | Within ladder |
|---|---|---|---|
| B (`diff+frag-820`) | 0.9113 | 0.9672 | 0.9993 |
| full-data T (convex monotone) | **0.9153** | 0.9625 | 0.9994 |
| full-data M (certified monotone max pairs) | 0.9104 | 0.9605 | 0.9995 |

The monotone models are at parity with B on these codec ladders, as on CID22-A. Every model orders each ladder
almost perfectly; the whole problem is placing different codecs and contents on one JND scale.

**Out-of-fold fits on B's fx1 inputs plus the descriptor (844 inputs), mean of 3 fold draws:**

| Model | Guarantee | Global | Within reference |
|---|---|---|---|
| NNLS linear | monotone | 0.950 | 0.971 |
| additive hinges (NNLS) | **convex** monotone | 0.962 | 0.973 |
| additive hinges + ramps | monotone | 0.963 | 0.975 |
| gradient boosting, monotone constraints | monotone | 0.965 | 0.975 |
| gradient boosting | none | 0.969 | 0.977 |

**The ceiling is information, not data or learner.**

- Learning curve (monotone boosting): 21 training references 0.961, 28: 0.964, 35: 0.966, 41: 0.966. More references
  like these do not help.
- Target transforms (log, square root, rank) change nothing (0.962–0.967).
- New feature families measured in Python (`jnd_band_proto.py`: masking-normalised Laplacian band errors at 7 scales,
  tile-max/top-10% pooling, band-energy gain vs loss): the best single feature is luma scale-2 band-energy loss,
  top-tile pooled, 0.932 (B's own best single input, `det_4th` luma scale 2, is 0.929; B itself 0.911). Alone the
  families reach 0.955–0.960; added to fx1 they reach 0.966 (monotone) and 0.971 (unconstrained).
- Each of 70+ supplied peer columns added to fx1 gains at most +0.005 (AIMOS, FSIM, PieAPP, butteraugli). **All 75
  supplied peers together with fx1 reach 0.974 (monotone boosting) / 0.977 (unconstrained)**; the peers alone 0.967 /
  0.974. (DVIFM columns excluded: their author constants were trained on AIC2026.)
- No reference-content descriptor (mean luminance, chroma, flat-area fractions, band energies) correlates with the
  per-source residual beyond |ρ| = 0.23 over 42 sources.

**The labels support 0.99.** The supplied pointwise confidence intervals are honest: (pointwise − curve)/pointwise
σ has SD 1.13. The curve's own σ is 0.073 JND, about the pointwise σ over √7, as a 7-point fit should be.
Resampling the curve labels from their intervals, independently or shared along each ladder, gives SROCC 0.998. So
the label noise is about 0.07 JND, while the best out-of-fold error is about 0.33 JND (RMSE).

**Where the error is** (monotone boosting, `aic_train.py analyze`):

| Oracle (applied to out-of-fold predictions) | Global SROCC |
|---|---|
| none | 0.9655 |
| remove the mean residual of each codec | 0.9672 |
| remove the mean residual of each source | 0.9723 |
| remove the mean residual of each (source, codec) ladder | 0.9834 |
| perfect shape within each ladder, ladder offsets kept | 0.9678 |

98% of discordant pairs are cross-source, and 62% are pairs whose labels differ by less than 0.25 JND (discordant
34% of the time). SROCC 0.99 needs discordance to fall from 8.2% to about 3.5%: the prediction error has to approach
the label noise. The largest single errors are ladders on a few contents: JPEG / JPEG 2000 / JPEG XL on a white
product shot (S55, up to 1.3 JND too lenient: block quantisation of faint shading in a flat background), most codecs
on a saturated fish on bokeh (S45), and JPEG / JPEG AI on S70 (too harsh).

**Compute.** Dropping every scale-0 slot (three quarters of the pixels) leaves the monotone linear and convex
additive models unchanged on the 42-source pool (0.950 → 0.950, 0.962 → 0.962); monotone boosting loses 0.006. On
`public_train` (7 folds × 3): NNLS 0.9344 → 0.9350, convex additive 0.9389 → 0.9373, monotone boosting 0.9529 → 0.9505. B's
own selection reads only 41 scale-0 slots of its 410 direct inputs (`fx1gs1` = fx1g without them: 387 direct, 369
products). Exact per-call user-space instructions of `BakeScorer::compute` (`perf stat`, slope between two iteration
counts so setup cancels, one thread, the `byc_score_timing` synthetic pair, two 1-epoch timing-only bakes over B's fx1
read set and over fx1 without scale 0):

| Size | fx1 read set | fx1 without scale 0 | Change |
|---|---|---|---|
| 512² | 95,488,070 | 75,092,745 | −21.4% |
| 2048² | 1,459,946,055 | 1,100,277,015 | −24.6% |

**Correction:** both timing bakes were trained in replay mode without a `zentrain.formula_revision` entry, so the table
above measures the shipped-revision (Rev1) walk, which keeps the scale-0 mean-offset side channel. Stamped with Rev5
(`bake_dial_refit append-meta`, weights byte-untouched), per-call instructions on today's tree (`a508d68f` sources;
`byc_score_timing`, slope method, one thread), without and with the per-scale skip attempt below:

| Read set (Rev5) | 512², no skip | 512², skip | 2048², no skip | 2048², skip |
|---|---|---|---|---|
| fx1 (B's) | 94,954,148 | 94,950,443 | 1,362,670,609 | 1,362,901,153 |
| fx1 without scale 0 | 99,645,449 | **55,617,427** | 1,426,163,942 | **735,160,388** |

With the skip, the scale-0-free read set costs 41% (512²) and 46% (2048²) fewer instructions than B's read set; without
it, it costs slightly more. The Rev1 rows above stay as measured (the skip gains only 2–3% there, since the Rev1 side
channel keeps scale 0).

Less than the pixel share because the planner gates v2 per scale (`v2_scales`) but the v1 basic family as one switch:
a profile of the scale-1+ bake at 2048² still spends 35% of samples in the basic SSIM blurs (`fused_vblur_ssim`,
`fused_blur_h_ssim`, `box_blur_v_copy`) and 10% in `memmove`, at full resolution. A per-scale basic gate is the next
engine change if the E21 arms show scale 0 can go. Wall time was not measured (the box was running training).

**Landed (this branch):** `ComputeSet::v1_scales` from the declared ids (v1-era reads plus free-tranche reads, which the
v1 fold produces), and one rule, `ComputeSet::scale_dead` (no v1-era read and no v2-era work at that scale), used by
`at_scale`, `channel_active` and the streaming walk's strip skip, so every walk (scalar, cached-reference attribution,
basic attribution, refinement) computes nothing there and they agree. Profile plans keep every v1 scale (a profile's
linear tail reads the v1 prefix undeclared). `cargo test -p zensim --all-features`: 908 passed, 0 failed, 27 ignored;
the new `declared_plan_skips_unread_scales_bit_identically` checks the skip and bit-identical read slots at the default
revision and at Rev5. Final per-call instructions (Rev5-stamped timing bakes, one thread): fx1 94,977,354 / 1,362,595,463
(512² / 2048², unchanged), fx1 without scale 0 55,651,205 / 734,904,178 (−41.4% / −46.1%); scores bit-identical to the
build without the skip. Wall time not measured (training was running).

The first attempt (`ComputeSet::v1_scales` from the declared ids; the streaming walk skips a scale with no v1-era read
and no v2-era work) is bit-identical on every requested slot in its own test, at the default revision and at Rev5,
and passes all 454 library tests. With all features on it fails 6 tests, all for one reason: the scalar walk skips
the dead scale while the attribution, session and refinement walks still compute it, so their full feature vectors
(unread slots included) disagree; one refinement golden also moves on a read slot (f488, local 1.0e-2 vs full 2.1e-5).
Profile plans needed every v1 scale kept (a profile's linear tail reads the v1 prefix undeclared). Not landed: every
walk must honour the skip first. Patch (261 lines, SHA-256 `0e3195eb…`):
`/mnt/v/output/zensim/byconstruction-2026-10-22/aic2026train/v1_scale_skip_attempt_2026-10-23.patch` on dev.

### Smoothness: error around a smooth curve (2026-10-23)

Owner, 2026-10-23: "I also really care about smoothness, ie error stats rather than just inversions; I want a smooth
curve against human ratings." Statistics on every row, nothing dropped or sampled (`aic_panel.py --geo`):

- **Z-RMSE** (native panel): RMS error around the panel's own 4-parameter logistic fit, in SDs of the labels.
- **Error p99 / max**: absolute error around the same fitted curve (Rust `--emit-rescaled`, now `zen_stats.logistic_fit`),
  as a share of the label span (p1–p99), and the share farther than 4 robust SDs.
- **Rank geometry** (zenstats `scatter::diagnose`): on the rank map, `out4` (share outside the ±4 robust-SD envelope),
  `p99d` and `maxd` (p99 and max residual over the span). It counts ordering errors only.

E21, out of sample (cells trained without the dataset, seeds 0–2 mean), Z-RMSE / error p99:

| Model | KADID | TID2013 | KonFiG | CID22-A | Mean Z-RMSE | Mean error p99 | Mean rank out4 |
|---|---|---|---|---|---|---|---|
| `base-c` (B's recipe) | **0.454** / 0.372 | **0.423** / 0.270 | 0.567 / 0.555 | 0.488 / 0.377 | **0.483** | 0.394 | 1.05% |
| Tga | 0.480 / 0.384 | 0.442 / 0.278 | 0.581 / 0.559 | 0.475 / 0.348 | 0.494 | **0.392** | 1.00% |
| Tga128 | 0.485 / 0.391 | 0.453 / 0.283 | 0.582 / 0.554 | **0.467** / **0.343** | 0.497 | 0.393 | 0.97% |
| Tghpm (no AIC leg) | 0.469 / 0.383 | 0.441 / 0.274 | 0.589 / 0.564 | 0.479 / 0.358 | 0.495 | 0.395 | 1.14% |
| Bga (signed + AIC) | 0.465 / 0.395 | 0.433 / 0.281 | **0.566** / **0.535** | 0.488 / 0.362 | 0.488 | 0.393 | **0.88%** |
| Ma (certified max pairs) | 0.488 / 0.389 | 0.468 / 0.282 | 0.599 / 0.608 | 0.477 / 0.359 | 0.508 | 0.410 | 1.15% |
| Ta (certified hinges) | 0.504 / 0.395 | 0.465 / 0.289 | 0.610 / 0.621 | 0.474 / 0.360 | 0.513 | 0.416 | 1.28% |
| Mgfa | 0.488 / 0.392 | 0.449 / 0.275 | 0.610 / 0.627 | 0.480 / 0.357 | 0.507 | 0.413 | 1.12% |
| DVIFM 0.3 (zero-shot) | 0.623 / 0.494 | 0.571 / 0.378 | 0.666 / 0.613 | 0.554 / 0.500 | 0.604 | 0.496 | 2.09% |

- B's recipe is the smoothest on KADID and TID2013. The monotone arms are 0.011–0.030 worse in mean Z-RMSE (Tga 0.494
  against 0.483) with the same p99 tails (0.392 against 0.394). On CID22-A every monotone arm is smoother than B's
  recipe (Tga128 0.467 against 0.488).
- The max-pair and certified-hinge arms (Ma, Mgfa, Ta) are the least smooth of our models: 0.03–0.05 above B's
  recipe in Z-RMSE on KADID and KonFiG, and their p99 tails grow most on KonFiG (Ta 0.621 against 0.555).
- No training term asks for a smooth curve against people. The human groups (`human`, `coverage`, `aic`) train with
  `loss=Rank`; only the SafeSyn and CID22 teacher legs use `Both` (rank plus regression to their SSIMULACRA2-derived
  targets). Smoothness against human ratings is whatever the rank fit leaves.

AIC2026 public_train (T, Tga, Tga128 and Bga out of fold; the rest fixed models):

| Model | Z-RMSE | Error p99 | Error max | Rank out4 | Rank p99 | Rank max |
|---|---|---|---|---|---|---|
| B (zero-shot) | 0.459 | 0.378 | 0.640 | 1.94% | 0.388 | 0.630 |
| T (AIC only) | **0.359** | **0.292** | 0.716 | 3.49% | **0.297** | 0.771 |
| Tga | 0.410 | 0.379 | 0.562 | 3.57% | 0.379 | 0.592 |
| Tga128 | 0.392 | 0.327 | 0.523 | 2.55% | 0.322 | 0.592 |
| Bga | 0.371 | 0.356 | 0.614 | 2.70% | 0.369 | 0.605 |
| butteraugli 3-norm | 0.453 | 0.379 | 0.909 | 4.03% | 0.380 | 0.771 |
| DVIFM 0.3 (constants fitted on AIC2026) | 0.364 | 0.327 | **0.479** | 2.73% | 0.333 | 0.771 |

**Seed ensembles do not close the gap.** Averaging the z-scored out-of-sample predictions of seeds 0–2 lifts B's recipe
from 0.8722 to 0.8763 mean SROCC (Z-RMSE 0.483 → 0.477) but the monotone arms by at most 0.0014 (Tga 0.8634 → 0.8638,
M64 0.8591 → 0.8605, Tgg and Thpm +0.0002): their deficit is systematic cross-type calibration, not seed variance.

**Which inputs the monotone models use** (`l0_usage.py`, layer-0 input norms ≥ 1% of the cell's largest, a difference
and its products counted together): the hinge arms leave about two thirds of fx1's 410 differences unused in every
cell (Thpm 274, Tgg 273, Tga 213); M64 uses nearly all (16 unused in every cell). Only 12 are unused by every cell of
all four arms (instances of banding, blockiness, gms, masked_art, masked_ssim, ssim_mean). The engine computes features
per family and scale, so this alone does not reduce extraction.

The dashboard's scatter tab plots every model on one JND scale three ways (the logistic fit, the zenstats rank map, and
raw output with mean and SD matched), with these statistics beside each plot.

### Global edits in the mixture test: the sign-free descriptor inverts (2026-10-23)

The mixture test's base distortions (blur, luma noise, chroma noise, posterize) never fire the global-change
descriptor, so they could not test the arms that use it. `mixture_test.py generate --d1 global` uses six global edits
instead (luma offset ±8, luma contrast ×0.85 / ×1.15 about the mean, chroma ×0.8 / ×1.25) with the same six noise
steps on top: 48 references, 1,728 steps per cell, `mixture_global/` on dev. RGB MSE rises in 1,727 of them.

| Model | Inversions per cell (all cells) | Over 1 point | Largest rise | Main steps |
|---|---|---|---|---|
| Thpm (monotone, no descriptor) | **0** (12 cells) | 0 | −3.8 | — |
| DVIFM 0.3 | **0** | 0 | −0.09 JND | — |
| Tga128 | 0–27 (60 in all) | 0–12 | +3.7 | saturation ×0.8 / ×1.25, then chroma noise σ8 |
| Tga | 4–47 (291) | 0–33 | +9.8 | the same; brighten/darken then chroma noise σ8 |
| Tghpm | 5–72 (385) | 2–56 | +11.9 | the same |
| B | 88 | 74 | +18.3 | saturation/contrast, then luma or chroma noise |
| Bga | 41–368 (2,052) | 36–341 | +48.4 | contrast ×0.85, then chroma noise |

**Owner check of the assumption (2026-10-24).** An owner-only picker page (`picker_global.py`; side-by-side and flicker
views at exact pixel zoom, models revealed only after each pick) showed 60 rounds: 49 steps where B or a Tga cell
scored edit + noise above the edit alone (at most four per edit × noise combination) and 11 controls where no model
did. The owner judged the noise-free version less bad in **60 of 60**, with no "can't tell", across all six edits and
all six noise levels. So adding noise after a global edit does make these images worse, and every inversion here is a
model error. Agreement with the owner's picks: M64, Tgg, Tga128 (cell mean) and DVIFM 0.3 60/60, Tga (cell mean) 55/60,
Bga 35/60, B 24/60. Picks: `/mnt/v/output/zensim/byconstruction-2026-10-22/picker/picks_2026-10-24.json` on dev (copied from the page's private store).

**The sign-free descriptor reopens an inversion route.** Without it (Thpm) no step inverts; with it, adding chroma
noise to a saturation-changed image raises the score by up to 9.8 points (Tga). The "1–4 inversions" reported for Tga
above hold only for the original base distortions. The candidate's monotonicity claim is therefore weaker than stated
there: it is monotone in its 820 damage inputs, and the free descriptor moves in the wrong direction under exactly these
steps.

### Gated damage (`fx1gg`): a global discount with every weight non-negative (2026-10-23)

**Design (fixed before any result).** Instead of feeding the descriptor sign-free, gate the damage with it. For each
of fx1's 410 differences `d` (cell `k` = its scale and channel, E33 §5.1), add the input `d·(1 − share_k)`, where
`share_k` is the byc_globaff share of cell `k`; the descriptor itself is not an input. With `W1 ≥ 0` on all 820 direct
inputs (and fx1's fragility products), a hinge sees `a·d + b·d·(1 − share)` with `a, b ≥ 0`. Adding unexplained error
raises `d` and lowers `share`, so it can only lower the score; a global edit lowers the gate, so the model can learn a
discount `a/(a + b)`. Identity keeps the pin (`d = 0`). The gated columns are exact f32 products
(`ring_root.py gated`, root `ringhpm/root_globaff_gated`, era `rev5_localwin_ringhpm_globaffgate` `#45e0e84c`;
declaration `byc_fx1gg`, 820 direct inputs). On KADID the median gated/direct damage ratio is 0.06 for color
saturation 1, 0.15 mean shift, 0.21 brighten, 0.29 contrast change, 0.85–0.90 blur and 0.96–0.99 noise; jitter is 0.14
(a small shift is mostly an affine match), the one surprise.

Arm **Tgg** (`mono:thr` on fx1gg, no AIC leg, trainer `70eba638`): same 12 E21 cells, compared with Thpm (same class
without the descriptor) and Tghpm (descriptor sign-free). Reported as measured: E21 Δ, smoothness, both mixture tests,
AIC2026 public_train zero-shot.

**Result.** Out of sample (seeds 0–2); Z-RMSE is the smoothness statistic of the previous section:

| Arm | E21 Δ (SE) | KADID / TID2013 / KonFiG / CID22-A Δ | Mean Z-RMSE | Standard mixture / cell (max rise) | Global-edit mixture / cell (max rise) |
|---|---|---|---|---|---|
| Thpm (no descriptor) | −0.0221 (0.0032) | −0.019 / −0.045 / −0.030 / +0.006 | 0.512 | 0–4 (+4.2) | **0** (−3.8) |
| **Tgg** (gated, every weight ≥ 0) | −0.0194 (0.0029) | −0.010 / −0.044 / −0.029 / +0.005 | 0.511 | 1–9 (+7.9) | **0** (−4.0) |
| Tghpm (descriptor sign-free) | −0.0101 (0.0024) | −0.007 / −0.024 / −0.018 / +0.007 | 0.495 | 2–7 (+7.1) | 5–72 (+11.9) |
| M (max pairs, mp32) | −0.0144 | — | 0.502 | up to 24 references (blur + luma σ4) | **0** (−0.6) |
| M64 (max pairs, mp64) | −0.0131 | — | 0.502 | 1–6 | **0** (−0.05) |
| Mhpm (mp32, hpm ringing) | −0.0158 | — | 0.504 | 2–35 (+8.0) | **0** (−0.06) |

- The gate does what it was built for: no global-edit step inverts, and KADID's gap halves (−0.019 → −0.010). It
  recovers nothing on TID2013 or KonFiG. KonFiG has no global edits at all, so the sign-free descriptor's gain there
  is not a global-edit discount: the free share acts as a distortion-kind signal for cross-type calibration, the same
  freedom that lets added chroma noise raise the score. Tgg's standard-mixture steps (posterize or blur, then noise)
  are damage features falling when noise refills lost detail; Thpm shows the same pattern, less often.
- AIC2026 public_train zero-shot: Tgg 0.908 (12 cells, 0.903–0.912), Tghpm 0.907.
- **Strictly monotone frontier.** The certified max-pair arms show no global-edit inversion either, so among arms with
  0 global-edit inversions M64 leads on E21 (−0.0131) and on smoothness (0.502, against 0.483 for B's recipe). Next:
  max pairs on the gated inputs (Mgg64), which combines the two strictly monotone gains.

**Mgg64** (`mono:thr:mp64` on fx1gg, no AIC leg, trainer `70eba638`): E21 Δ **−0.0161** (SE 0.0019): KADID **−0.003**,
TID2013 −0.028, KonFiG −0.032, CID22-A −0.001. Mean Z-RMSE 0.507 (KADID 0.462, TID2013 0.456, KonFiG 0.611, CID22-A
0.497). Global-edit mixture 0 inversions in all 12 cells; standard mixture 2–36 per cell (88 in all, max +7.6; blur or
posterize, then light noise). AIC2026 public_train zero-shot 0.903 (0.899–0.910).

The gate gives the monotone class its best KADID result (Δ −0.003 and Z-RMSE 0.462, against M64's −0.017 and 0.487),
where global edits are, and costs KonFiG (−0.032 against −0.015) and TID2013. Net, it is worse than M64 on E21
(−0.0161 against −0.0131) and on smoothness (0.507 against 0.502). M64, which reads no descriptor at all, stays the
best arm with no global-edit inversion.

### Map todo: guided redistribution and coarse-upsample kernels for JXL map utility (2026-10-24)

Owner, 2026-10-24: "On the todo list for maps: Can we get the new features into a fully-optimized scalar + map path?
guided_coarse_redistribution < helpful on no on map utility for jxl? Also can we investigate better interpolation
methods, for diffmaps as well as what might help for the laplacian."

**What the JXL loop's map is.** jxl-encoder's zensim loop (`JXL_ZENSIM_RD_PROFILE=b`) steers with zensim's fused
`DiffmapWeighting::Trained` map. Under profile B that map is the legacy `WEIGHTS_PREVIEW_V0_2` basic-plane map (B's
network supplies the score, not the map). The loop pools it per 8×8 block as an L2 norm (`compute_tile_dist`). Guided
redistribution (E-JBU) keeps every coarse cell's mass inside its own footprint (at most 8 px), so it changes a block's L2
only by concentrating that mass.

**Setup.** `ZENSIM_DIFFMAP_UPSAMPLE` (custom-profiles research knob, default unchanged, `1a8f3ebc`) selects bilinear,
Catmull-Rom, cubic B-spline or Lanczos-3 sampling of each coarse plane at pixel centres, optionally `-mass` (each
footprint keeps its nearest-neighbour mass, with the interpolation as guide). jxl-encoder gained
`JXL_ZENSIM_GUIDED_COARSE` (`32e42f9b`, default off). The harness binary was built from a symlink mirror so that every
path dependency resolves to this branch's zensim (`/var/tmp/byconstruction-bin/jxl-maprd-e37ed3ee/`, PROVENANCE
there). Corpus: 24 TRAIN images (`jxl_map_corpus.py`: 10 CID22 training photos, 2 CID22 training graphics, 6 gb82-sc
screen crops, 6 imazen-26 TRAIN illustration crops; 512 px, pixel-exact crops). Seven distances 0.6–10, 4 loop
iterations, 9 arms (no-loop baseline, nearest, guided, four kernels, two `-mass` forms). Judges: SSIMULACRA2 and
butteraugli 3-norm from an independent zenmetrics build (`/var/tmp/byconstruction-bin/zenmetrics-judge-7a0f8394/`), and
that build's zensim (home turf). Run: `jxl_map_rd.sh`, outputs `/mnt/v/output/zensim/byconstruction-2026-10-22/jxl-maprd/`.

**The map against no loop** (bytes saved at equal judged score, median over matched points): nearest saves +2.3% /
+0.1% / +0.1% (SSIMULACRA2 / butteraugli / zensim) on photos, +4.4 / +2.8 / +1.9 on screens, +1.0 / −2.2 / −0.8 on
illustrations. Comparing modes through these unpaired medians is noisy; the paired comparison below is the measurement.

**Against today's nearest map** (`rd_probe_analyze_2026-07-18.py RD --paired-vs nearest`: bytes saved at equal judged
score, mean over images of each image's median, 95% bootstrap interval over images; positive is better, %):

| Mode | SSIMULACRA2 | butteraugli | zensim |
|---|---|---|---|
| guided redistribution | −0.29 [−0.78, +0.23] | −0.51 [−1.71, +0.26] | −0.15 [−0.91, +0.49] |
| bilinear | +0.01 [−0.66, +0.78] | +0.33 [−1.21, +1.97] | −0.15 [−0.64, +0.36] |
| bilinear-mass | −0.12 [−0.42, +0.13] | −0.66 [−1.89, +0.14] | −0.21 [−0.57, +0.12] |
| Catmull-Rom | +0.06 [−0.27, +0.40] | +0.61 [−0.39, +1.88] | −0.09 [−0.65, +0.38] |
| Catmull-Rom-mass | −0.24 [−0.50, −0.03] | +0.34 [−0.68, +1.60] | −0.04 [−0.30, +0.20] |
| cubic B-spline | +0.03 [−0.39, +0.47] | −0.61 [−1.72, +0.43] | −0.01 [−0.56, +0.48] |
| Lanczos-3 | +0.12 [−0.28, +0.53] | +1.22 [−0.31, +3.00] | +0.12 [−0.36, +0.63] |

**Larger corpus for the two best kernels** (78 TRAIN images: 38 photos, 22 graphics and illustrations, 18 screen crops,
three per screenshot; `jxl_map_corpus.py --photos 40 --illustrations 20 --screen-crops 3`; `jxl-maprd-large/`):

| Mode | SSIMULACRA2 | butteraugli | zensim |
|---|---|---|---|
| Catmull-Rom | **+0.29 [+0.03, +0.61]** | +0.14 [−0.92, +1.20] | +0.09 [−0.23, +0.43] |
| Lanczos-3 | −0.07 [−0.56, +0.34] | +0.20 [−0.98, +1.33] | −0.28 [−0.70, +0.12] |

Lanczos-3's butteraugli edge does not hold overall (+1.22 on 24 images, +0.20 on 78); it holds on illustrations (+2.22
[+0.57, +4.14], 22 images) and turns negative on photos (−0.71 [−2.38, +0.20]). Catmull-Rom is the only kernel whose
overall interval clears zero, on SSIMULACRA2 only, by 0.3% of bytes.

- **Guided redistribution does not help JXL.** It is slightly worse overall, clearly worse on screens (SSIMULACRA2
  −0.66% [−0.91, −0.42], butteraugli −2.41% [−6.46, +0.35]), and gains only on photos under the home-turf zensim judge
  (+0.57% [+0.13, +1.01]). Leave it off.
- **Interpolation is not a meaningful lever for JXL.** On 78 images Catmull-Rom saves 0.3% of bytes on SSIMULACRA2
  (interval just above zero) and nothing measurable on the other judges; Lanczos-3 helps illustrations on butteraugli
  only. Sub-block shape matters little to a loop that pools each 8×8 block with an L2 norm.
- **Not covered:** the attribution-map arms (`JXL_ZENSIM_MODEL_MAP`, a map that follows the model's own gradient) do
  not use this fusion path, so these kernels do not reach them.

**New features in the scalar and map paths.** Not built. The global-change descriptor helps only the sign-free arms,
which invert on global edits; gated into a monotone model it helps KADID and costs TID2013 and KonFiG (Tgg, Mgg64), and
the strictly monotone frontier (M64) reads neither it nor the hpm ringing. Serving it (a research family at f1867–f1890
like palette's, five running sums per channel and scale fused into the walk, closed-form map derivatives) waits for a
model that needs it.

### Serving the top candidate: formula revision 6 (2026-10-24)

Owner, 2026-10-24: "I really want to get the current top candidate code optimized and formalized to test in the new
zenmetrics workbench." The candidate (Tga) read two research-only inputs: the `hpm` ringing form, a per-process
`ZENSIM_RING_FORM` arm, and the `byc_globaff` descriptor, a sidecar zensim could not compute. Both are now served.

- **Formula revision 6** = revision 5 with the high-pass ringing form, selected by the bake's declared revision
  (`zentrain.formula_revision = 6`). B (revision 5) and a revision-6 bake serve in one process and in wasm, where no
  environment exists. The registry carries the change as the per-signal era `ringhpm` on the 12 ringing slots
  (`FormulaRevision::Rev6`, `#[non_exhaustive]`, so the enum change is additive).
- **`globaff` family**, f1867..1890 (after the research palette block), computed at revision 6 only. The streaming walk
  adds each emitted strip's rows of every channel and scale into five f64 sums per cell (16 fixed lanes, column mod 16,
  one fixed pairwise reduce; `zensim/src/globaff.rs`), so the result does not depend on SIMD tier, strip tiling or
  thread count. The scale-zero chroma rows stay in the rolling planes for it; no kernel runs on them. A revision-6
  bake's plan derives at its own revision, so the read set reaching f1890 does not switch on the layout-only blocks or
  the scale-zero chroma kernels.
- Checks: the walk's descriptor equals a two-pass f64 reference over the walk's own planes at 64², 97×71 and 203×300;
  revision 6 against revision 5 on the same read set moves only the 12 ringing slots; identity gives exact zeros;
  plans below revision 6 refuse the family. `cargo clippy --workspace --all-targets --all-features` is clean.
- **Served against training (pixel route vs table route).** The candidate bake, stamped revision 6
  (`bake_stamp_revision`), scored through `score_pair_with_bake --bake-post raw` with no ring-form environment,
  reproduces its training-route predictions (`byc_predict` on the research tables) on all 3,920 AIC2026 public_train
  pairs: max |Δ| 7.5e-6, p99 5.0e-7 (scores on a 0–100 scale). A Tga E21 cell reproduces its stored table-route
  mixture scores on all 1,344 images: max |Δ| 7.0e-6 (those scores are rounded to 1e-6). The served descriptor
  therefore matches the sidecar to well below anything a score can resolve.
- Stamped bakes (private: trained on AIC2026 public_train labels): `~/tmp/byc/parity/tga_full_rev6.bin` on dev, SHA-256
  `5ece9e45…`; the source bake is `aic2026train/full/Tga/refit/last.bin` (`2664b4c7…`).

**M64a** (`mono:thr:mp64` on fx1 with the AIC2026 TRAIN leg at weight 32): E21 Δ **−0.0222** (SE 0.0032; KADID
−0.013, TID2013 −0.039, KonFiG −0.042, CID22-A +0.006), AIC2026 public_train out of fold 0.929 (within source 0.952),
standard mixture 2–12 inversions per cell (max +4.5), global-edit mixture **0** in all 12 cells. The AIC leg costs the
max-pair class 0.009 on E21 (M64 −0.0131) for no AIC gain over Tga (0.932). Tga stays the candidate by the committed
rule; M64a keeps M64's strict global-edit monotonicity.

### Attribution-map correctness (2026-10-24)

Owner, 2026-10-24: "make sure we test EVERYTHING for catastrophic cancellation". The candidate map
(`BakeScorer::compute_with_ref_and_attribution`) was checked by giving one feature a unit sensitivity, summing its map
over the image and comparing with the feature (`metric::bake::map_split_tests`; shipped B bake, 410 probed features,
a 192×160 blur pair). Four defects, all fixed on this branch:

1. **Pass B read the process revision** (`ec9289df`). The retained planes follow the bake's revision, but the map
   kernels chose the SSIM form (direct error moment, Rev3+) and the ringing form (high-pass, revision 6) from
   `ZENSIM_FORMULA_REV`. B (revision 5) in a default process read its error-moment plane as a product moment and the
   SSIM deviation pools amplified the result: the map summed to −1.8e19 (one probe, f635, to −2.8e20). After the fix the
   total is 356, every probe is bounded, and B and a revision-6 bake give the same map bits in processes at revisions
   1, 5 and 6 (`attribution_maps_serve_the_declared_revision_in_every_process`; with the bug restored it fails). The
   same code is on main (`feature_v2.rs`, three `active_revision()` reads in pass B): main's B maps are affected in a
   default process (not measured on main). The JXL loop's default B map is the legacy basic-plane map, not this one;
   its `JXL_ZENSIM_MODEL_MAP` arms follow the model's gradient and would be (not checked in jxl-encoder).
2. **The deviation-pool polynomial cancelled in f32.** Evaluated in raw powers of `d`, the fourth-moment pools' maps
   miss their sum identity by up to 2.2% on the blur pair (6 of 410 probes outside 1e-4; negative control of
   `exact_split_maps_sum_to_their_features`). The kernels now evaluate one quartic in `e = d − μ` (Horner), and the
   gradient/edge deviation pools one quadratic in `e = v − m`, in the f64 and both f32 kernels.
3. **A revision-5/6 map leaves out the v2 features of scales 1–3** (f459–f719). For steering, the neighbour-exact
   snapshot zeroes them in the density and serves them as exact finite deltas through `refinement_gain`; a map for
   viewing loses most of its coarse-scale signal. A scorer with research map modes (`with_map_paint`) now keeps them
   and takes no snapshot. The earlier map-study dumps and the workbench module's maps were missing them and are rerun.
4. **The removal split is not additive.** The map gives each pixel its first-order removal effect: exact for mean
   pools, `f/p` in total for an Lp pool, and for the SSIM deviation pools a near-uniform positive field (removing a
   pixel moves it away from the mean) that dominates the map: one probe (f577, a feature of 0.00074) integrates to 32.
   The new **exact split** (`ZENSIM_MAP_SPLIT=exact`, or `with_map_paint(.., .., "exact")`) gives each pixel its share
   of the feature, as DVIFM splits its block terms: Euler shares for the Lp, L8 and deviation pools, `w·v/W` for soft
   peaks, the secant share for the global mean, global contrast and HF-gain ratios; mean and linear pools are
   unchanged. Under it, 400 of the 410 probes sum to minus their feature within 1e-4 on the blur pair, and all of them
   on a near-constant pair (a uniform +4 brightening, where every deviation pool is tiny against its mean). The 10
   others are the cross-scale edge-width slot (v2 slot 28), which stays linearized in both splits. Removal stays the
   default; steering relies on it.

Also: the research map painter's box sums now sum each window afresh instead of a running add-and-subtract, which left
a large value's rounding error in every later window (`map_paint::box_sum`).

**globaff one-pass moments cancelled** (fixed). The served descriptor formed the error as
`(ms − md)² + vss + vdd − 2·vsd` from raw one-pass sums: quantities the size of the texture subtracted to get one the
size of the error. On a 2048×1024 plane with mean 0.6 and a 2e-5 offset, `share` came out 0.99971 instead of 1
(`large_mean_tiny_error_matches_compensated_two_pass`, against compensated two-pass moments). The walk now sums the
error `e = s − d` (exact in f64) and its moments, so `mse = mean(e)² + var(e)` and
`explained = mean(e)² + min(var(e), cov(s, e)²/var(s))` add non-negative terms; all three tiny-edit cases agree to
1e-6. Served Tga against its training route after the fix: AIC2026 public_train max |Δ| **5.0e-7** over 3,920 pairs
(7.5e-6 before), the standard mixture cell 5.0e-6 over 1,344 images, and the global-edit mixture cell 7.0e-6 over
2,016 images (tables rounded to 1e-6; `serve_parity.py`, `just byc-serve-parity-pairs`).

**v2 SSIM luminance term cancels in the features themselves** (open: a fix changes feature bits). The direct form
computes `nb − na = (m1² + m2² + C1) − (2·m1·m2 + C1)` in f32, whose true value is `(m1 − m2)²`; the walk
(`feature_v2_kernels.rs`) and the map kernels share it. `map_split_tests::v2_ssim_luminance_cancellation` recomputes
each cell's mean `d` from the walk's own f32 moments in f64: on near-identical pairs the walk's v2 SSIM means of the
chroma channels at scales 1–3 are off by up to +336% (+1 LSB on every 7th pixel, B channel, scale 3) and −87% (a uniform
+4 on a flat image, X channel); a structured blur is within 1e-7. Computing `(m1 − m2)²` directly in f32 agrees with
f64 to 1e-5 in every cell. These features are small (1e-8 to 1e-9) but this is the near-lossless regime. B (revision
5) carries it; changing it needs a new formula revision and a retrain.

**Feature audit, all served families** (revision 5, see "Formula revision 7" below): against the exact f64 oracle
on 93 near-identical TRAIN pairs, every basic, peaks and v2 feature is within 5.1e-5 median relative error, and
switching every cancelling form to its exact equivalent moves B's score by at most 7.9e-5 points.

**Read, no subtraction to cancel** (code reading, not a test): the workbench loop map (a positive power of the
density, then a 16-norm) and the attribution's Lp and L8 coefficients (products and powers). **Measured elsewhere**: the
DENSEPOOL precision record puts revision 5's v2 pools at a maximum relative error of 1.09e-3 against the exact
summation oracle (`524a04c28`). **Known, frozen eras**: revisions 1–4 finalize the SSIM deviation pools from raw
moments (fixed at revision 5); the append block's deviation pools and global variances use `E[v²] − E[v]²` over f64
sums.

### Map study after the fixes (2026-10-24)

Owner, 2026-10-24: "I want excellent quality diffmaps, whether for jxl or not, down to per-pixel refinement ... I saw
much larger streaking on zensim b. Try what dvifm 0.3 uses." Every candidate dump made before the fixes above was
invalid (moved to `maps/study/invalid-2026-10-24-pre-fix/` with a note). Fresh dumps (`scripts/byconstruction/map_study.sh`,
`map_dump`, no `ZENSIM_*` map or revision variables): the stamped Tga bake (`5ece9e45…`) and B, over the 150 study pairs
(72 KADID E21 TRAIN pairs at level 3, 78 JPEG XL d=4 corpus pairs: CID22 training photos, imazen-26 TRAIN
illustrations, gb82 screen crops) and 60 repaired images (`map_quality.py leak`: a 64-px square of the distorted image
replaced by the reference, at butteraugli's hottest square and at a fixed one). Judges: butteraugli, SSIMULACRA2's score
mass and DVIFM 0.3's painted map. Statistics: median over pairs of the SROCC of 16-px block sums; the median share of a
map's mass on pixels where the two images are equal; the share of a map's change that falls outside a repaired square
(leak; 0 is perfectly local). Outputs: `/mnt/v/output/zensim/byconstruction-2026-10-22/maps/study/` (`metrics.json`).

| Map | butteraugli | SSIMULACRA2 | DVIFM 0.3 | Mass where equal | Leak median / p90 |
|---|---|---|---|---|---|
| B legacy (basic-plane diffmap) | 0.566 | 0.683 | 0.377 | 0.66% | — |
| B, removal split, box | 0.649 | 0.666 | 0.222 | 0.48% | 0.52 / 0.88 |
| B, exact split, guided window + paint | **0.703** | 0.706 | 0.261 | 0.04% | 0.48 / 0.89 |
| B, exact split, centred window + guided paint | 0.697 | 0.710 | 0.260 | 0.05% | 0.48 / 0.74 |
| Tga, removal split, box | 0.653 | 0.695 | 0.247 | 0.36% | 0.44 / 0.80 |
| Tga, removal split, guided + guided | 0.659 | 0.694 | 0.245 | 0.04% | 0.44 / 0.81 |
| Tga, exact split, box | 0.698 | **0.731** | 0.265 | 0.32% | 0.44 / 0.69 |
| Tga, exact split, centred window | 0.693 | 0.726 | 0.268 | 0.32% | 0.43 / 0.67 |
| Tga, exact split, guided window | 0.694 | 0.724 | 0.261 | 0.30% | 0.43 / 0.86 |
| Tga, exact split, guided window + paint | 0.695 | 0.724 | 0.261 | **0.03%** | 0.43 / 0.86 |
| Tga, exact split, centred window + guided paint | 0.693 | 0.726 | 0.267 | 0.04% | 0.43 / **0.67** |

The judges agree with each other at 0.579 (butteraugli–SSIMULACRA2), 0.168 (butteraugli–DVIFM) and 0.204
(SSIMULACRA2–DVIFM).

- **The exact split is the lever**: +0.045 to +0.054 against butteraugli and +0.036 to +0.040 against SSIMULACRA2 over
  the removal split, for both bakes. Window and paint modes move block agreement by less than 0.01.
- **Guided paint puts a tenth of the mass on unchanged pixels** (0.3% → 0.03–0.05%) at no cost in block agreement:
  it is the per-pixel refinement. **The centred window leaks least** (p90 0.67 against 0.86 guided).
- **Against B's legacy map**, the exact maps agree better with butteraugli (+0.13) and SSIMULACRA2 (+0.03 to +0.04)
  and less with DVIFM (which the other judges barely agree with). By group (butteraugli / SSIMULACRA2), B exact
  centred+guided against legacy: illustrations 0.56 / 0.60 against 0.45 / 0.50, KADID 0.72 / 0.74 against 0.61 / 0.75,
  photos 0.79 / 0.72 against 0.60 / 0.65. On screens no map agrees with butteraugli (−0.01 to 0.13); against
  SSIMULACRA2 they are 0.67–0.74.
- The workbench module's `diffmap` is now the centred window with guided paint and the exact split; `exact_box`,
  `removal` (today's map) and `blocks8` (8×8, centred, exact) stay alongside.
- **Tga's globaff slots now have a map term** (`globaff::map_planes`; they were `unsupported: 24` in every record, one
  of them Tga's largest sensitivity, about 6000). `explained` (degree 2 in `e = s − d`) takes the exact pixel share
  `(mean(e)·e_i + (cov(s,e)/var(s))·(s_i − mean(s))·e_i)/N`; `share` (degree 0) takes `(E_i − share·e_i²/N)/mse`,
  which moves mass between pixels and sums to zero; the removal split is twice each. Test:
  `exact_split_globaff_maps_sum_to_their_features` (a global edit and a blur; negative control: the removal factor in
  the exact split fails it). Rerun with the term (Tga, exact split): centred + guided paint 0.696 / 0.714 / 0.257
  (butteraugli / SSIMULACRA2 / DVIFM), box 0.697 / 0.716 / 0.260; illustrations gain against butteraugli (0.577 from
  0.544), KADID and screens lose about 0.012 against SSIMULACRA2. The weakest KADID types against SSIMULACRA2 are the
  global edits (brighten 0.20, darken 0.12, contrast 0.20), where the descriptor spreads its mass over the image by
  design. Kept: the map now carries every feature Tga reads.

Gallery (reference, distorted, B legacy, B removal, B exact centred+guided, Tga removal, Tga exact centred+guided,
butteraugli, SSIMULACRA2, DVIFM 0.3): `http://dev.lan:3300/zensim/byconstruction-2026-10-22/maps/study/gallery/`.

**Workbench module.** `zensim-workbench` (metrics `zensim-byc-tga` and `zensim-b`; maps `diffmap` = centred window +
guided paint + exact split, `exact_box`, `removal`, `blocks8`) is served privately on the LAN (it embeds Tga, trained
on WG1 labels): site `/mnt/v/zenmetrics-private/zensimx-workbench/site-2026-10-24/`. Checks: `just
byc-workbench-smoke` (native against the wasm build in node: scores within 1e-6, map sums within 1e-4) and `just
byc-workbench-check` (the site in headless Chromium through the workbench's own worker; scores and every map equal
native, the metric menu lists the module). The smoke check found the wasm clock panic above. Loop defaults, by
zenmetrics' method on its 2026-10-21 calibration encodes (fit: CID22-512 training; held out: gb82 screenshots; log r
of block distances against butteraugli's):

| Map | Tga scale / exponent | Tga fit r / held-out r | B scale / exponent | B fit r / held-out r |
|---|---|---|---|---|
| diffmap | 0.1412 / 0.4 | 0.672 / 0.495 | 0.1748 / 0.35 | 0.593 / 0.281 |
| exact_box | 0.1793 / 0.4 | 0.645 / 0.535 | 0.1751 / 0.4 | 0.590 / 0.371 |
| removal | 0.1500 / 0.5 | 0.601 / 0.512 | 0.1741 / 0.45 | 0.584 / 0.331 |
| blocks8 | 0.1553 / 0.6 | 0.771 / 0.611 | 0.2028 / 0.5 | 0.707 / 0.438 |

Tga's numbers include the globaff term; without it they were 0.695 / 0.518 (diffmap) and 0.802 / 0.664 (blocks8):
the descriptor adds image-wide mass that butteraugli's local block distances do not share. For comparison, our AIC-4
port reports held-out r 0.51–0.69 for the Haar diffmaps and 0.83 for DVIFM-min on the same encodes; on screenshots
butteraugli's map agrees with no other map in the study.

### Streaking: dithered gradients, the guide and perceptual integration (2026-10-24)

Owner, 2026-10-24: "i'm still seeing streaking". The workbench's default pair (`maps/demo/`, captured from the page:
its 768×512 synthetic scene and the default zenjpeg encode at quality 40) shows horizontal lines across the sky in
every per-pixel map. The error has that shape. The reference sky is a vertical gradient with ±1 dithering; the JPEG
flattens each 8-row block row to one level (green, column 300, rows 0–23: reference `167 167 167 168 167 168 168 168
168 169 168 169 168 169 169 170 169 170 170 170 170 170 170 171`, distorted `167`×8, `169`×8, `170`×8). Over the top
32 rows, the row means of |error| vary with a coefficient of variation of 0.396 and the column means with 0.035. The
block steps are visible banding; the dither is not. Butteraugli's map shows the same horizontal banding.

The per-pixel map draws the rows at twice the error's own contrast (row-mean CV over the same band; Tga, exact split,
centred window, guided paint):

| Map | Row-mean CV | × the error's |
|---|---|---|
| σ 0, squared guide (`diffmap_px`) | 0.832 | 2.10 |
| σ 0, absolute guide | 0.682 | 1.72 |
| σ 1.5, squared guide | 0.497 | 1.26 |
| σ 3, squared guide (`diffmap`) | 0.277 | 0.70 |
| σ 3, absolute guide | 0.264 | 0.67 |

Two changes, both research map modes (nothing in the score path reads them):

- **Perceptual integration** (`ZENSIM_MAP_SMOOTH=<σ>`, `BakeScorer::with_map_smooth`): after painting, a separable
  Gaussian scatter of the per-pixel map, each pixel's weights renormalized over its in-bounds neighbours (radius
  ⌈3σ⌉), so the map's total is unchanged (`map_paint::tests::smoothing_conserves_mass_and_zero_is_identity`).
- **Guide norm** (`ZENSIM_MAP_GUIDE=sq|abs`, `BakeScorer::with_map_guide`): the guided window and paint weigh a pixel
  by its squared difference (the default) or its absolute difference. The squared guide accounts for part of the
  exaggeration; with the absolute guide the per-pixel map still shows 1.72×, so the rest is not in the paint.

Map study (the 150 pairs, 60 repaired images and judges of "Map study after the fixes"; same statistics):

| Map | butteraugli | SSIMULACRA2 | DVIFM 0.3 | Mass where equal | Leak median / p90 |
|---|---|---|---|---|---|
| Tga, σ 0 | 0.696 | 0.714 | 0.257 | 0.04% | 0.44 / 0.67 |
| Tga, σ 0, absolute guide | 0.694 | 0.716 | 0.263 | 0.03% | 0.43 / 0.67 |
| Tga, σ 0.75 | 0.700 | 0.717 | 0.263 | 0.27% | 0.44 / 0.67 |
| Tga, σ 1.5 | 0.707 | 0.721 | 0.263 | 0.40% | 0.44 / 0.67 |
| Tga, σ 1.5, absolute guide | 0.706 | 0.721 | 0.263 | 0.41% | 0.44 / 0.67 |
| Tga, σ 3 | **0.716** | 0.728 | 0.268 | 0.50% | 0.44 / **0.66** |
| Tga, σ 3, absolute guide | 0.715 | **0.729** | 0.268 | 0.51% | 0.44 / **0.66** |
| B, σ 0 | 0.697 | 0.710 | 0.260 | 0.05% | 0.48 / 0.74 |
| B, σ 0, absolute guide | 0.697 | 0.710 | 0.255 | 0.05% | 0.48 / 0.73 |
| B, σ 1.5 | 0.711 | 0.713 | 0.268 | 0.38% | 0.48 / 0.73 |
| B, σ 3 | **0.718** | 0.723 | 0.268 | 0.48% | 0.48 / 0.72 |
| B, σ 3, absolute guide | 0.717 | **0.724** | **0.269** | 0.49% | 0.48 / 0.72 |

- **σ 3 raises agreement with all three judges** (Tga +0.020 / +0.014 / +0.011; B +0.021 / +0.014 / +0.008) and
  keeps locality (leak p90 0.66 against 0.67). Butteraugli's and SSIMULACRA2's maps are smooth themselves, so part
  of the gain is shared smoothness rather than better placement. Mass on unchanged pixels rises tenfold (0.04% to
  0.50%), as an integration that reaches 9 px from a change must.
- **The guide norm changes no judged statistic** by more than 0.006. It stays a research knob; the squared guide
  stays the default.
- The integrated map also agrees better with butteraugli's block distances in the JPEG XL loop calibration (below):
  held-out r 0.653 against 0.495 (Tga), 0.681 against 0.457 (M64a), 0.491 against 0.281 (B).

**Workbench (second build).** The module's metrics are named by arm and served revision: `zensim-Tga@rev6`,
`zensim-M64a-noKADID@rev6` (M64a's KADID-held-out fold cell, the only stamped M64a bake; `b8555871…`) and
`zensim-B@rev5`. Its menu group names the zensim version and the commit the module was built from (`just
byc-workbench-module` sets `ZENSIM_WORKBENCH_LABEL`), and comes first among the extra modules
(`workbench_site.py update` swaps a built module into the served site in place). Maps: `diffmap` (σ 3, the default),
`diffmap_px` (the same map at σ 0), `removal` and `blocks8`. The page's own "zensim" entry is zenmetrics' built-in
metric (zensim 0.3.0 from its main checkout; its B map is the legacy basic-plane diffmap), not this module. Loop
defaults by the method above (`loop_calibration_2026-10-24c.tsv`; fit r / held-out r):

| Map | Tga scale / exponent | Tga r | M64a scale / exponent | M64a r | B scale / exponent | B r |
|---|---|---|---|---|---|---|
| diffmap (σ 3) | 0.1330 / 0.6 | 0.809 / 0.653 | 0.1372 / 0.6 | 0.806 / 0.681 | 0.1303 / 0.6 | 0.748 / 0.491 |
| diffmap_px | 0.1412 / 0.4 | 0.672 / 0.495 | 0.1450 / 0.4 | 0.651 / 0.457 | 0.1748 / 0.35 | 0.593 / 0.281 |
| removal | 0.1500 / 0.5 | 0.601 / 0.512 | 0.1506 / 0.5 | 0.533 / 0.396 | 0.1741 / 0.45 | 0.584 / 0.331 |
| blocks8 | 0.1553 / 0.6 | 0.771 / 0.611 | 0.1591 / 0.6 | 0.782 / 0.642 | 0.2028 / 0.5 | 0.707 / 0.438 |

`diffmap_px` reproduces the first build's `diffmap` fits exactly (the same map), a check on the calibration.

### Formula revision 7: the batch (2026-10-24)

Owner, 2026-10-24: "let's do a revision 7 - but also figure out what else we can batch with it." Every item changes
feature bits, so one revision and one retrain carry them all. Inventory:

1. **v2 SSIM luminance term**: `nb − na = (m1² + m2² + C1) − (2·m1·m2 + C1)` becomes `(m1 − m2)²`. Measured above
   (+336% / −87% on near-identical chroma at scales 1–3); with the stable form the walk agrees with f64 on its own
   moments to 4.3e-6 or better in every cell of all four diagnostic pairs.
2. **Bounded dissimilarities**: `1 − (2ab + c)/(a² + b² + c)` becomes `(a − b)²/(a² + b² + c)`, its exact value,
   for v2 edge artifact and detail loss, GMS, texture and the edge-width decay terms (walk, map and steering replay).
3. **v1 edge ratio**: `(1 + x)/(1 + y) − 1` becomes `(x − y)/(1 + y)` (v1 basic edge artifact and detail loss and the
   pools built on them, every kernel tier).
4. **DENSEPOOL f32 lanes**: 11.3–13.9% faster end to end on i265 (AVX2, one thread), features move by at most
   6.05e-8 relative (`524a04c28` on main). The blocked interchange (8.4–8.6%, bit-identical) needs no revision.

Not needed, already fixed: the XYB chunk-versus-remainder divergence (revision 4's canonical opsin conversion), the
running-sum drift and the raw-moment deviation pools (revision 5). Every era the registry marks "proposed" is already
in revisions 2–6.

Candidates that need an experiment before they join (from the HaarPSI / HaarXYB comparison): coarse scales averaged
in linear light, a joint-channel distance, distorted-side unmasking, orientation-selective bands (new families,
append-only).

Items 1–3 were first implemented as helpers (`bounded_dissim`, `bounded_dissim_v`, `bounded_dissim32`) and paired
forms behind `--cfg zensim_stable_forms` (a custom cfg rather than a cargo feature, so `--all-features` in CI never
turned a measurement form on). The cfg is gone: revision 7 selects the forms (below, "Formula revision 7
implemented"). This first measurement compared two builds of one revision. Measurement: `tier_audit_features` dumps of the full research
vector at revision 6 on 93 near-identical pairs (KADID level 1 of all 25 types on three TRAIN references; 18 JPEG XL
d=0.6 corpus encodes), default and stable-forms builds against the exact f64 oracle (`ZENSIM_FEATCANON=exact`);
`scripts/byconstruction/rev7_forms.py`; scores: `scripts/byconstruction/rev7_score_compare.py`. Outputs:
`/mnt/v/output/zensim/byconstruction-2026-10-22/rev7/`.

**Results (revision 5, AVX-512 tier).** Against the exact oracle, today's arithmetic is already within 5.1e-5 (median
relative error over the 93 pairs) for every basic, peaks and v2 feature; the stable forms give 4.7e-5. They help where
they apply, by 2–4× at the 90th percentile on the coarse SSIM pools (Y at scale 3: deviation 1.5–1.7e-4 → 4.1–7.1e-5,
soft peak 1.8e-4 → 4.9e-5). The largest real-content errors come from two other shapes, both tiny in absolute terms:
the HF magnitude loss turns blur rounding into positive mass through its `max(0, ·)` (3.3e-6 where the exact value
is 0, screenshot crops) and v1's `var_loss`/`tex_loss` take a difference of sums of squares (1e-2 relative, about 2e-6
absolute).

**Score impact** (default against stable-forms scorer, the 93 pairs plus the three synthetic extremes of the SSIM
diagnostic): B moves by at most **7.9e-5** points (median 3.0e-6), Tga by at most **2.3e-5**. The extreme feature
errors (+336% / −87%) sit in features of 1e-8 to 1e-9 that both models map to zero either way.

**So the cancellation fixes are hygiene, not a reason for a revision.** They cost nothing to batch into a revision
that happens for other reasons. The value changes worth a retrain are DENSEPOOL's f32 lanes (11–14% faster) and any
feature change that proves out in an experiment, starting with linear-light coarse scales.

### Speed pass before revision 7 (2026-10-24, in progress)

Owner, 2026-10-24: "I'd also like a performance pass before rev7 is finalized so we can absolutely sqeeze it for
everything possible." Changes that keep every bit land on revisions 5 and 6 as they are measured; changes that move
feature bits are collected for the revision-7 cut.

**Harness.** `byc_score_timing` (`zensim-validate`) times `BakeScorer::compute` and now also dumps the score's and
every emitted feature's bits (`--dump`), reads real images (`--image`, a fixed quantize-and-dither distortion), takes
any geometry (`--width/--height`) and forces a tier (`--tier`, as `tier_audit_features` does). Bit identity:
`perf_bits.sh` dumps B (revision 5) and Tga (revision 6) on six synthetic geometries (97×63 to 2050×1030) and five
gb82 photos at v4x, v4, v3 and scalar with one thread and v4x with eight, 110 dumps; `perf_bits_compare.py` compares
two dump sets. Speed: `perf_grid.sh`, paired rounds alternating the arm order (`ring_timing.sh`,
`timing_summary.py`), one thread pinned to core 2. Profiles: `perf record` / `perf annotate`.

**Where the time went** (B, revision 5, 2048², v4x, one thread, 78.97 ms; share of zensim's samples): the v2 dense
pool kernel 27%, sRGB → XYB 16%, the vertical box blur 10%, the fused SSIM pools 9%, `OnlineMoments::block` 9%, the
horizontal fused blur 9%, the gradient kernel 8%. Thread scaling at 2048² on the large-cache CCD: 1 / 2 / 4 / 8 / 16
threads 70.6 / 51.1 / 38.7 / 20.2 / 25.7 ms (after the changes below), 3.5× at eight threads.

**Bit-identical changes** (110 of 110 dumps identical; zensim's full suite 929 passed, 0 failed, including
`featcanon_rev5_parity` over every token permutation, `featcanon_tier_parity`, `featcanon_rev4_contract`,
`fold_engine_parity`, `research_engine_parity` and `cross_tier` on dev, Zen 5):

1. **DENSEPOOL's blocked interchange** in the dense kernel for the sixteen-lane pools (`DENSE_ILV = 16` chunks
   buffered; 8 ties, 32 is slower): each pool loads its two lane halves once per block instead of once per chunk.
   −4.5% at v3; within 0.6% at v4 and v4x, where the f64 conversions, not the pool round trips, bound the kernel.
2. **The sRGB table in registers** on v4x (`srgb_xyb_canon_vec_v4x`): one `vpermb` deinterleaves sixteen pixels and
   eight `vpermt2ps` plus a blend tree look each channel up, instead of three 16-lane gathers per chunk. −4.2%.
3. **Eight central-moment blocks per SIMD call** in the dense kernel: one block per `f64x8` lane, each lane running
   `OnlineMoments::block`'s scalar sequence op for op, merged in raster order. −6.8%.

Paired, 10 rounds, against the session's base binary (`benchmarks/byconstruction_perf_2026-10-24/step1/grid.tsv`;
raw rounds on dev under `/mnt/v/output/zensim/byconstruction-2026-10-22/perf/2026-10-24/step1/`; load average 2–4
from other sessions during the run):

| Bake | Tier | 512² | 2048² |
|---|---|---|---|
| B (rev 5) | v4x | 5.479 → 4.983 ms, −9.07% (−9.21..−8.95) | 78.77 → 70.50 ms, −10.53% (−10.65..−10.40) |
| B (rev 5) | v3 | 7.284 → 6.861 ms, −5.77% (−5.88..−5.66) | 108.86 → 102.27 ms, −6.12% (−6.22..−5.98) |
| Tga (rev 6) | v4x | 5.750 → 5.239 ms, −8.87% (−8.95..−8.73) | 82.35 → 74.50 ms, −9.64% (−9.86..−9.44) |
| Tga (rev 6) | v3 | 7.731 → 7.317 ms, −5.32% (−5.41..−5.26) | 116.06 → 109.21 ms, −5.82% (−6.07..−5.76) |

Not measured yet: v4 (only as a spot check), multithreaded paired rounds, aarch64 and wasm (their kernels did not
change), an Intel host.

**Where the time goes now** (B, 2048², v4x): the dense kernel 34% of zensim's samples, about a quarter of it the f64
pool conversions and adds (`vcvtps2pd`, `vaddpd`), which only revision 7's f32 lanes remove; sRGB → XYB 12%, now
register copies around the destructive permutes; the four blur kernels and the fused SSIM pools 34% together; the
gradient kernel 8%.

4. **Horizontal blurs without per-tap range checks** (`box_blur_h_local11_wide`, `fused_blur_h_local11_wide`, v4x
   only): each 16-output block reads one checked 26-leaf window and loads its eleven taps at constant offsets. Their
   integer and branch share fell from 87% / 62% of their samples to 26% / 9%. Paired, 8 rounds, against change 3:
   B −3.24% (512²) and −3.39% (2048²) at v4x, Tga −2.81% / −2.98%; v3, which uses other blur kernels, ±0.1%
   (`step3/grid.tsv`). 110 of 110 dumps identical; full suite 929 passed.

Hoisting the fused SSIM kernel's eleven mirrored window rows out of its chunk loop changed nothing measurable (68.13
against 68.14 ms); LLVM already hoists them, so it was not kept.

**Revision-7 candidates, measured** (`--cfg zensim_rev7_pools`, measurement only; the default build is bit-identical
to the base, 110 of 110; the cfg is gone, and revision 7 is now the implementation, "Formula revision 7 implemented"
below):

- **f32 row lanes.** Each pool's chunk adds go to sixteen f32 lanes for the row, folded into the f64 lanes once per
  row, in the dense kernel (29 pools), the fused SSIM kernel and the gradient kernel.
- **Central moments two-pass per row.** A sixteen-lane f64 row mean, then the centred sums (d−μ)², (d−μ)³, (d−μ)⁴ in
  sixteen f64 lanes, one Chan merge per row instead of one per 16 columns.
- **Not every pool tolerates f32.** Pools whose features take a difference or ratio of two large sums lose up to
  2.4e-4 (relative) of those features to the cancellation: revision 1's variance and texture loss and contrast
  increase (Σ(s − μ₁)² against Σ(d − μ₂)²; 2.4e-4 on `basic_contrast_inc_s3_b`) and the edge-width change (Σ gradient
  of the source against the distorted; 1.6e-5). Those pools, and every band-lifetime pool (the raw moments, some
  signed, and the bounded-error bins), keep f64 lanes. With them in f64 the largest change of any feature over the 110 dumps is **6.3e-8** (relative); the score
  moves in 4 of the 110 dumps.

Paired, 8 rounds, against the current default build (`rev7c/grid.tsv`):

| Bake | Tier | 512² | 2048² |
|---|---|---|---|
| B (rev 5) | v4x | 4.819 → 4.171 ms, −13.41% (−13.76..−13.16) | 68.17 → 56.02 ms, −17.80% (−17.89..−17.71) |
| B (rev 5) | v3 | 6.859 → 5.403 ms, −21.24% (−21.37..−21.16) | 102.17 → 76.91 ms, −24.74% (−24.86..−24.63) |
| Tga (rev 6) | v4x | 5.098 → 4.444 ms, −12.84% (−13.18..−12.78) | 72.26 → 60.25 ms, −16.71% (−16.77..−16.62) |
| Tga (rev 6) | v3 | 7.298 → 5.850 ms, −19.78% (−19.95..−19.60) | 109.21 → 84.20 ms, −22.88% (−23.01..−22.83) |

The dense kernel alone (`rev7a/grid.tsv`, an earlier build): −11.4% / −13.1% (B, v4x), −14.3% / −16.3% (B, v3).
Since the session's base, B at 2048² on v4x has gone 78.77 → 68.17 ms (bit-identical) → 56.02 ms with these
candidates.

What a real revision 7 needs beyond the prototype: one canonical definition independent of SIMD width (every column,
the `width % 8` tail included, adds to f32 lane `x & 15`; the fold to f64 at the row's end, or every 4,096 columns so
an f32 lane never takes more than 256 adds), the same arithmetic in the scalar, NEON and wasm bodies (the prototype
changed only the v4x/v4/v3 kernels, so its scalar-tier dumps still equal revision 6), and the tier-parity suite. A
cleaner fix than keeping the cancellation-prone pools in f64 is to accumulate the per-pixel difference those
features need, which f32 lanes carry without loss; that changes those features' definitions slightly and belongs to
the same revision.

**Remaining time** (prototype, B, 2048², v4x, one thread, fold-timing phases of a 58.8 ms walk): sRGB → XYB 10.5 ms
(18%), dense 14.0, vertical planes 11.2, fold 6.4, gradient 6.7, horizontal blur 5.8, blockiness 2.3, downscale 1.3.
Not yet pursued:

- The dense, gradient and fused SSIM kernels evaluate their per-pixel terms eight columns at a time on every tier
  (`f32x8`), on AVX-512 too. Sixteen columns per step is bit-identical (element-wise terms, the same per-lane pool
  order) and would halve the term arithmetic on v4/v4x.
- The XYB conversion is now bound by the register table lookup (permutes, blends and the copies around the
  destructive permute, about half its samples); its divisions (the cube root's Halley steps) are 4%.
- The vertical blur keeps its frozen-tree pair and quad sums in a stack ring indexed per row; 66% of its samples are
  integer work.
- **Threads.** At eight threads the walk takes 20.2 ms (3.5×); 6.7 ms per walk is outside every timed phase, the
  consume phase runs at 0.60 occupancy, and scratch zeroing grows 2.4×. The fold-MT lane
  (`fold_mt_scaling_2026-08-31.md`) measured the same ceiling as a property of memory behaviour, not scheduling:
  independent single-threaded processes saturate at 3.5× too. The working set explains it: a strip is
  `STRIP_ROWS` = 128 rows plus a 10-row halo each side, so at 2048 wide one f32 plane of a strip is 1.21 MB, more
  than a core's 1 MB L2, and a channel's strip keeps about fourteen planes; every pass streams from the shared L3.
  The structural lever is a line-buffer pipeline (horizontal blur, the vertical window over an 11-row ring, then
  the pixel kernels, row by row), whose working set fits in L2. Revision 5's definitions (per-row pools, 32-row
  bands merged in order, mirroring at image edges only) already allow it, so it needs no revision change.
- **NEON runs scalar.** The dense, gradient, fused SSIM and XYB kernels' NEON variants call the scalar canonical
  bodies, because magetypes' NEON `max`/`min` are `FMAX`/`FMIN` (NaN-propagating) while the canon's `f32::max`
  returns the non-NaN operand. AArch64 `FMAXNM`/`FMINNM` have exactly `f32::max`'s semantics (Rust's scalar
  `f32::max` compiles to them there), so NEON kernels using them would be bit-identical to the canon.

Tried and not kept: the vertical blur with its window in registers, one or two column vectors at a time down the
whole plane instead of a stack ring over eight-vector tiles. Bit-identical, but **slower**: +11% (v4x) and +5% (v3)
at 2048², because the tiles' row-major locality matters more than the ring's loads and stores.

### Formula revision 7 implemented (2026-10-24)

`FormulaRevision::Rev7` (era `rowlanes`, registered as proposed; `ZENSIM_FORMULA_REV=7`, or a bake declaring revision
7) is revision 6 with the three changes below, in one canonical definition that every body computes: the scalar
canonical bodies (which NEON and wasm128 also run) and the v3, v4 and v4x vector kernels.

- **f32 row lanes** in the dense, fused-SSIM and gradient pools. The columns of each row's complete 8-column chunks
  (x < 8·⌊w/8⌋; the gradient kernel's interior chunks start at x = 8) add to sixteen f32 lanes, lane `x & 15`, which
  fold into the f64 lanes at the end of the chunks and after every column x with (x + 1) mod 4,096 = 0, so no f32 lane
  takes more than 256 adds. The remaining columns add in f64, as before. The cancellation-prone pools stay f64: the HF
  energy and magnitude sums of source and distorted (whose difference forms v1's variance and texture loss and
  contrast increase), the gradient sums of source and distorted (whose ratio forms the edge-width change), and every
  band-lifetime pool.
- **Per-row central moments.** The SSIM deviation moments are two-pass per row (a sixteen-lane f64 mean, then the
  centred sums of (d − μ)², (d − μ)³ and (d − μ)⁴), with one Chan/Pébay merge per row.
- **Cancellation-free forms**, items 1–3 of the batch.

Revisions 5 and 6 keep their bits: the default build's 110 dumps (B at revision 5, Tga at revision 6) are identical to
the session's base.

The stamps copy a bake with a new declared revision and byte-identical weights:
`bake_stamp_revision zensim/weights/b_sdr_diff_frag_820_rev5_2026-10-19.bin 7 b_rev7.bin` (SHA-256 `5fe0a962…`;
revision 6, `a2a31635…`). Tga's stamps are private and stay on dev.

**Tier identity.** `perf_bits.sh` with revision-7 stamps of B and Tga (`bake_stamp_revision`): 22 bake/pair groups,
110 dumps, 0 tier mismatches (v4x, v4, v3 and scalar with one thread, v4x with eight). The new
`zensim/tests/featcanon_rev7_parity.rs` extracts the whole revision-7 scope (basic + peaks + v2 + globaff) under every
token permutation on 64², 97×63, 131×65, 255×129 and 4133×33 (scale 0 crosses the 4,096-column fold): bit-identical.

**What moves** (revision-6 against revision-7 stamps, same binary, the 110 dumps): 19,905 of B's 39,600 emitted
feature values. The relative change has median 1.6e-8, 90th percentile 2.4e-6, 99th percentile 6.2e-5 and maximum
4.1e-3, the last on `v2_ssim_dev4_s3_y` of the 97×63 pair (5.733e-6 → 5.709e-6, an absolute 2.4e-8). B's score moves
by at most **7.1e-6** points (median 0); Tga's score does not move in any of its 55 dumps. On the parity test's 131×65
pair 446 of 1,891 slots move, by at most 9.0e-6 relative with the magnitude floored at 1e-3 (the test's 1e-2 guard
against a lost fold is my bound, not a measured one).

**Against the exact f64 oracle** (`ZENSIM_FEATCANON=exact`, the 93 near-identical pairs of the batch measurement, v4x,
each revision against its own exact reference; `rev7_forms.py --exact-stable`). The oracle's gradient body now
follows the revision's ringing form and forms, so revisions 6 and 7 have an f64 reference; the two references differ
by at most 6.4e-7. Per-feature median error:

| | p90 | p99 | worst |
|---|---|---|---|
| Revision 6 | 2.80e-6 | 3.80e-5 | 5.10e-5 (`v2_masked_ssim_s1_b`) |
| Revision 7 | 2.68e-6 | 3.71e-5 | 4.93e-5 (`peaks_ssim_max_s2_b`) |

Revision 7 is closer to the exact value on 307 features (by a median 3.6e-8; `v2_ssim_dev4_s3_y` 1.02e-5 → 6.3e-6,
`peaks_edge_det_max_s3_b` 7.2e-6 → 2.5e-6) and further on 196 (by a median 1.8e-9, at most 1.0e-6 on
`v2_ssim_soft_peak_s0_b`); 1,388 tie, most of them slots outside revision 7's scope. Outputs:
`/mnt/v/output/zensim/byconstruction-2026-10-22/rev7/rev7impl_accuracy/` (dumps, the oracle binary, the JSON).

**Speed** (paired, 8 rounds, one thread pinned to core 2, the same binary with revision-6 against revision-7 stamps;
`benchmarks/byconstruction_perf_2026-10-24/rev7impl/grid.tsv`; raw rounds on dev under
`/mnt/v/output/zensim/byconstruction-2026-10-22/perf/2026-10-24/rev7impl/`; load average 0.7–1.2):

| Bake | Tier | 512² | 2048² |
|---|---|---|---|
| B | v4x | 5.109 → 4.246 ms, −16.88% (−17.24..−16.79) | 72.30 → 56.71 ms, −21.58% (−21.65..−20.96) |
| B | v3 | 7.066 → 5.612 ms, −20.60% (−20.65..−20.58) | 104.77 → 79.76 ms, −23.90% (−23.96..−23.72) |
| Tga | v4x | 5.352 → 4.502 ms, −16.01% (−16.22..−15.75) | 76.19 → 60.16 ms, −20.99% (−21.13..−20.86) |
| Tga | v3 | 7.470 → 6.022 ms, −19.42% (−19.51..−19.29) | 111.66 → 86.61 ms, −22.43% (−22.53..−22.26) |

Not measured: multithreaded paired rounds, aarch64, wasm and an Intel host at revision 7. Revision 7 is not
finalized: the speed pass continues, and the models need re-extraction and a retrain at revision 7 before any of it
serves.

### Revision 7 speed after the implementation (2026-10-24)

Owner, 2026-10-24: "i actually just care about a fully optimized rev7", then "please only benchmark dimensions that
matter". After one 18-case size × content matrix (synthetic 64² to 4096², photos, web and mobile screenshots, a GUI,
a plot, clipart, a text scan and a 4096² brochure crop) showed content type does not change the results while size
and tier do, timings cover revision 7 only: B (Tga runs the same kernels), v4x and v3, a 256² photo crop, a 1024² plot
and a 1507×2048 photo (`scripts/byconstruction/perf_cases.py`, `cases_focus.tsv`; case images and raw rounds with
SHA256SUMS on dev under `/mnt/v/output/zensim/byconstruction-2026-10-22/perf/`). Every change was checked against the
110 dumps at revisions 5/6 (identical unless the change is revision 7's) and at revision 7 (identical across tiers).

**Cumulative** (paired, 8 rounds, one thread; the revision-7 commit `bad544a3` against `b6a7e510`;
`benchmarks/byconstruction_perf_2026-10-24/rev7_bad544a3_vs_b6a7e510/grid.tsv`):

| Tier | 256² photo | 1024² plot | 1507×2048 photo |
|---|---|---|---|
| v4x | 1.356 → 1.071 ms, −20.97% (−21.61..−20.64) | 15.61 → 12.29 ms, −21.30% (−21.56..−20.86) | 43.19 → 34.09 ms, −21.01% (−21.18..−20.48) |
| v3 | 1.659 → 1.282 ms, −22.70% (−22.95..−19.88) | 21.12 → 16.94 ms, −20.03% (−20.10..−19.60) | 59.71 → 46.46 ms, −22.17% (−22.45..−22.00) |

The changes, in order:

1. **Sixteen columns per step on AVX-512** in the dense, gradient and fused-SSIM kernels (`01b2677e`, `4e17ba37`):
   every term is element-wise, so a lane computes the eight-wide step's bits; revision 7 adds a step's sixteen values to
   the sixteen f32 row lanes in one op. Measured slower at revision 5 (its f64 lanes split each step back into two), so
   since `db964918` the wide steps run at revision 7 only. The gradient chunks also load from windows cut to their
   exact span; the term helpers are generic over a `Vf32` trait (magetypes' 8- and 16-lane types have the same
   per-lane ops).
2. **sRGB8 → linear through byte permutes, 64 pixels per step** on v4x (`98fc621b`, every revision): the table as
   four byte planes, two 128-entry `vpermt2b` per plane blended on index bit 7, a deinterleave whose permute tables
   are transposed so two unpack levels yield pixels in order. 50.8 → 48.5 ms on the synthetic 2048² pair.
3. Matrix after 1–2 against `bad544a3` (`matrix_rev7_bad544a3_vs_db964918/grid.tsv`, B and Tga): v4x −13% to −18% for
   images of 576² and up, −8% to −9% at 256², about −1% at 64², for every content class alike; v3 +0.3% to +1.2%
   (recovered by 5–6).
4. **Difference forms and the cube root** (`819e2c49`, revision 7): the v1 HF sums and the edge-width gradient sums
   move to the f32 row lanes as source sums plus differences, `(vd − vs)(vd + vs)`, `|vd| − |vs|`, `gdst − gsrc`
   (exact subtractions when the two are within a factor of 2), folded as source → both sums, difference → the
   distorted sum; the opsin cube root is `det_math::cbrt_newton_f32` (a reciprocal-root seed, one Newton step, one
   residual Halley step: one division; within 3 ulp below 2^126 and correctly rounded for 72.7% of [1, 8), against
   `cbrt_midp`'s 3 ulp and 56%). Paired against `db964918`: v4x −0.78% / −2.05% / −3.03%, v3 −1.37% / −2.62% /
   −2.63% (256² / 1024² / 1507×2048).
5. **v3/v4 horizontal blurs read one checked window per block** (`ea78465e`, every revision), as the v4x versions did:
   v3 −7.69% / −10.52% / −13.20%.
6. **Vertical blurs cut each row's tile span once** (`cb4dba9b`, every revision): v3 −1.66% / −2.30% / −2.73%; v4x
   −0.58% / −0.26% / −0.44% (the last two inside their intervals).
7. **A one-thread rayon pool runs the sequential paths** (`77ac5c2a`): every parallel section had still injected its
   work into the pool and blocked the caller on a futex. 64² 0.2535 → 0.2203 ms, 256² 1.232 → 1.140 ms (one thread,
   unpaired medians). With eight threads parallel work still wins at 64² (0.163 against 0.219 ms).
8. **BakeScorer caches its composed plan** (`d7282076`): it was rebuilt several times per score. 64² 0.2198 →
   0.1867 ms, 256² 1.137 → 1.096 ms.
9. **Folds only written pools; a revision-7 dense kernel without the eight-lane buffers** (`5e706fc9`): 64² 0.1875 →
   0.1813 ms, 256² 1.096 → 1.060 ms, 1024² 12.16 → 11.97 ms.
10. **Gradient head and tail as one eight-wide step each** (`bbeb1f9b`, revision 7): without the gmsbank and rev4-bank
    paths and in rows of at least ten columns, the head (x = 1..8) and the columns after the chunks are one
    `gradient_terms32_v` each, whose kept lanes add to the f64 pools in ascending x. v4x, one thread, unpaired medians of
    two runs: 64² 0.1777 → 0.164 ms, 256² 1.067 → 0.996 ms, 1024² 11.97–12.46 → 11.68–11.70 ms.
11. **Two redundant divisions in the dense terms** (`b9c9e5d6`, every revision): the HF gain/loss pair (its operands are
    squares, so one numerator is always zero and `b − a` is exactly `−(a − b)`) and the art/det soft-peak saturations
    (never both nonzero; `saturate(+0) = +0`) take one division each, and the copy of `d` into the row-moment buffer
    compiles to vector converts. Paired, 10 rounds (`rev7_bbeb1f9b_vs_densediv`): v4x −1.02% / −0.65% / +0.23%
    (−1.13..+0.63), v3 −0.35% / −0.48% / −0.40%. The dense kernel is not bound by its divider.
12. **Vertical blurs walk 2 KiB column panels** (`59595d48`, every revision): retirement waited on a load
    (`ex_no_retire.load_not_complete`) for 16% of all cycles at 1507×2048 on v4x, and 39% of those samples were in the
    vertical blur, whose 512-byte tiles step one 6 KB row stride per few cache lines, which the stream prefetchers do
    not follow. The panels walk the full height in contiguous 2 KiB row segments; rings of four pair and eight
    quadruple segments hold every live tree node, and every output keeps its operands and tree. Paired
    (`rev7_densediv_vs_vpanel`): v4x −1.22% / −4.33% / −3.26%, v3 −1.92% / −2.91% / −2.47%. 4 KiB panels were 1.7–2.1%
    slower than 2 KiB (`rev7_vpanel2k_vs_4k`), 1 KiB within ±0.9% (`rev7_vpanel2k_vs_1k`). The blur's load-stall samples
    fell 35%; it still holds 29% of them.
13. **Band- and call-lifetime f64 lanes** (revision 7): equal-pixel synthetic images expose a fixed cost of about
    0.2 µs per channel-scale row (v4x, one thread: 1536×2048 32.7 ms, 768×4096 35.2 ms, 384×8192 39.5 ms, 192×16384
    51.1 ms, 96×32768 83.1 ms). Per row, the dense, fused-SSIM and gradient kernels zeroed 67 sixteen-lane f64 pools,
    folded the row lanes into them and ran 67 sixteen-lane tree finishes. Revision 7's f64 lanes now live for a 32-row
    band (dense) or for the call (fused SSIM and gradient: the 128-row strip) and each pool finishes once there. Values
    move by at most 1.8e-11 relative (110 dumps, still identical across tiers and thread counts); B's and Tga's scores
    on the 96 scoring pairs are equal at the scorer's six decimals; against the exact oracle the per-feature median
    error keeps p90 2.39e-6, p99 3.78e-5 and worst 5.91e-5 (26 features closer than the previous revision-7 build, 24
    further; `rev7/rev7band_accuracy/`). Paired (`rev7_vpanel_vs_band`): v4x −11.56% / −4.66% / −4.04%, v3 −6.98% /
    −2.32% / −1.46%.

14. **Blockiness eight steps at a time** (revision 7's lanes; the vector steps at every revision): one shared row
    kernel (`blockiness_row`) serves the strip and whole-range walks. With the eight-lane f64 accumulators it computes
    eight steps per vector, each lane the scalar step's IEEE operations, the vertical lattice columns read through one
    checked 58-value window per eight. Horizontal steps add to lane `x & 7` and keep their bits at every revision; at
    revisions 5 and 6 the vertical steps still add to lane 0 in column order (bit-identical, 110 of 110 dumps), and at
    revision 7 the step at column `8j` adds to lane `(j − 1) & 7`, which removes the serial chain (values move by at
    most 1.3e-14 relative; tiers identical). Blockiness falls from 5.4% to 2.0% of v4x samples. Paired
    (`rev7_band_vs_blockiness`): v4x −2.08% / −3.01% / −3.26%, v3 −0.67% / −1.96% / −2.00%.

15. **No raw `d²`, `d³`, `d⁴` pools at revision 7**: from revision 5 on the SSIM deviation features read the central
    moments, and nothing reads `sum_d2`, `sum_d3` or `sum_d4` (their only other use, the attribution context, is
    overridden by the central moments). Revision 7's dense kernel no longer accumulates them (26 of 29 pools; the
    three sums stay zero). Every emitted value keeps its bits at every revision (110 of 110 dumps at revisions 5/6
    and at revision 7). Paired (`rev7_blockiness_vs_raw3`): v4x −0.32% / −2.78% / −1.61%, v3 −1.35% / −2.96% /
    −1.67%. Skipping the three pools with `continue` inside the 29-pool loop instead was 12% slower (it changed the
    loop's code generation); the kernel takes a constant index list.

16. **Blockiness in the parallel strip jobs** (revision 7): with eight or more threads the walk runs strip jobs in
    parallel batches, but the merge after each batch computed every strip's blockiness on the main thread, because
    revisions 4–6 keep one running sum across strips. A latency profile at eight threads (`perf record --latency`)
    put 30.9% of the wall time at one running thread, blockiness 7.8% and the window copies 8.0% of it. Revision 7
    sums each 128-row kernel strip into its own eight lanes, in the job, and the merge adds them in strip order; the
    single-thread strip walk and the whole-range form (`blockiness_sparse_rows`, per-128-row partials) do the same,
    so every path and thread count gives the same bits (`rev7_blockiness_strip_partials_match_the_whole_range_form`
    compares the two forms at five geometries; a whole-range running sum fails it by one ulp). Revisions 5/6: 110 of
    110 dumps identical; revision 7: values move by at most 2.0e-15, tiers and 1/8 threads identical. Paired, B,
    v4x: eight threads (cpus 0–7) −9.38% at 1507×2048 and −10.48% at 1024² (`rev7_raw3_vs_blockpar_t8`); one thread
    +0.31% / +0.20%, intervals spanning zero (`rev7_raw3_vs_blockpar_t1`).

17. **Strip batches from two threads** (every revision): the walk ran strip jobs in parallel batches only when the
    rayon pool had at least eight threads, so two to six threads got only the kernels' own parallelism (1507×2048, B
    at revision 7, v4x: 1.40× at two threads, 1.85× at four). Batching now starts at two threads, one job per thread
    and at least four per batch; the merge order is unchanged, so every value keeps its bits (110 of 110 dumps at
    revisions 5/6 and 7; revision-7 dumps at 2, 3, 4 and 6 threads identical to one thread). Unpaired medians:
    2 threads 21.2 → 18.7 ms, 3 → 17.9 → 15.1, 4 → 16.2 → 11.4, 6 → 14.9 → 9.4, 1 and 8 unchanged. Paired, 8 rounds
    (`rev7_blockpar_vs_batch2_t2`, `_t4`; the commit message of `8b3c06ed` says ten): two threads −13.01% / −9.89%
    (1507×2048 / 1024²), four −30.15% / −28.05%. Batches of twice the thread count were slower at eight threads
    (7.8 → 10.1 ms) and at sixteen.

18. **Strip jobs flow through a pipeline** (every revision): the batch runner gathered a batch of strip jobs on the
    walk's thread, ran the batch in parallel and merged it, so the workers idled while the walk's thread gathered the
    next batch's windows and converted its rows, and each batch waited for its slowest job. Walks that retain nothing
    and read no cached reference now spawn each job as soon as it is gathered (`Rev5Pipeline`); finished jobs come
    back over a channel and merge in the producer's order, and the producer waits only when the in-flight limit (the
    batch size: one job per thread, 4 to 16) is reached. The merge order is the batch runner's, so every value keeps
    its bits (110 of 110 dumps at revisions 5/6 and 7; revision-7 dumps at 2, 4, 8 and 16 threads identical to one
    thread). Paired, 8 rounds, B at revision 7, v4x (`rev7_batch2_vs_pipeline_t8`, `_t4`): eight threads (cpus 0–7)
    7.960 → 6.387 ms, −20.31% (−21.95..−18.66) at 1507×2048 and 2.548 → 2.188 ms, −14.22% at 1024²; four threads
    (cpus 0–3) −8.06% / −7.36%. Unpaired medians of 20 iterations at 1507×2048: sixteen threads on CCD0's cores and
    their SMT siblings (cpus 0–7, 16–23) 11.36 → 9.69 ms, sixteen over both CCDs (cpus 0–15) 18.17 → 11.86 ms,
    32 threads 15.67 → 11.03 ms. Lower in-flight limits were slower at eight threads (4: 9.10 ms, 6: 7.53, 8: 6.33,
    unpaired). Eight threads on one CCD remain the fastest configuration: each in-flight job's planes take about
    10 MB, so sixteen jobs no longer fit CCD0's 96 MiB L3.

19. **Rows produced in parallel tasks; jobs copy their own windows and release their scratch on finishing** (every
    revision; plain SDR walks whose live scales all run the v2 blocks, at two or more threads). With the pipeline the
    walk's thread was the critical path at eight threads: the fold timers put 2.9 ms of a 6.2 ms walk inside row
    production (conversion and downscale fork-joins queued behind strip jobs in the pool), and a profile put 1.3 ms
    of its samples in copying strip windows (about 57 MB per walk at 1507×2048). Now each 64-row chunk of each image
    is a task (`feature_v2_blocks`) that converts its rows, on the producer's 64-row lattice with its row offset, and
    computes every deeper scale's rows from them (each child row reads two parent rows of the same chunk); each
    plane's rows are immutable blocks, strip jobs copy their windows out of the blocks, and the walk's thread only
    schedules, adds `globaff` rows and merges (`rev5_block_walk`). Moving work off that thread alone gained little at
    eight threads (paired against the pipeline: −0.60%, interval −3.88..+1.34, at 1507×2048; −8.67% at 1024²;
    `rev7_pipeline_vs_blockwalk_noearly_t8`), because a finished job kept its 9.8 MB scratch until every earlier job
    had merged, so one long scale-0 job stalled submissions. The merge reads only a job's small results except for
    revision-5/6 blockiness, so every other job now returns its scratch the moment it finishes and the in-flight limit
    counts scratch in use. A limit of sixteen at sixteen threads on eight cores then overflowed CCD0's 96 MiB L3:
    against sixteen, twelve gave −13.18% / −5.30% there and −2.46% / −2.55% at 32 threads (intervals spanning zero),
    eight −20.81% / −0.41% and +8.79% / +13.55% (`rev7_blockwalk_cap16_vs_cap12_*`, `_cap8_*`), so at most twelve
    jobs hold scratch; production runs at most eight chunks ahead (−3.60% / −3.68% at sixteen threads, +0.94% /
    −7.63% at 32; `rev7_blockwalk_ahead_*`). Bits: 110 of 110 dumps at revisions 5/6 and at revision 7; revision-7
    dumps of B and Tga on three photos at 1, 2, 3, 4, 6, 8, 12, 16, 24 and 32 threads identical to one thread; a unit
    test (`block_producer_windows_match_the_streaming_producer`) compares every strip window of the block producer
    with the streaming producer's at revisions 5 and 7, with and without scale-0 chroma, over seven geometries, and
    fails with the omitted-mode X and B outputs swapped. Paired, 10 rounds, against the pipeline (18), B at revision
    7 (`rev7_pipeline_vs_blockwalk_*`):

    | Threads (cpus) | Tier | 1507×2048 photo | 1024² plot |
    |---|---|---|---|
    | 2 (0–1) | v4x | 16.66 → 15.97 ms, −3.59% (−5.26..−1.08) | 5.966 → 5.628 ms, −5.50% |
    | 4 (0–3) | v4x | 10.59 → 9.039 ms, −15.20% (−17.34..−11.92) | 3.679 → 3.615 ms, −2.06% (−12.22..+0.98) |
    | 8 (0–7) | v4x | 6.386 → 5.261 ms, −17.60% (−20.73..−17.09) | 2.175 → 1.713 ms, −21.22% |
    | 16 (0–7, 16–23) | v4x | 9.788 → 8.814 ms, −9.99% (−11.23..−8.46) | 2.325 → 2.183 ms, −6.83% |
    | 32 (0–31) | v4x | 16.39 → 9.197 ms, −43.89% (−46.45..−29.61) | 4.994 → 2.965 ms, −39.94% |
    | 8 (0–7) | v3 | 8.631 → 6.933 ms, −20.41% (−21.19..−18.92) | 3.022 → 2.416 ms, −20.04% |

    One thread does not use this path. Sixteen threads on eight cores remain slower than eight: each running job
    still writes and reads eleven planes of 148 rows (9.8 MB at 1507 columns), which is the next target.

20. **The SSIM moments in one pass per column panel** (every revision from 5; strip jobs in pools of eight or more
    threads): each job's four horizontal moment planes (148 rows of the full width) were written by one pass and read
    back by four vertical blurs. `blur::fused_blur_hv_ssim_at_revision` computes a 256-column panel's horizontal rows
    into the first rows of those planes (about 600 KB, held in L2) and runs the vertical blur on them at once, with
    both kernels' trees, operands and roundings at every tier (16 lanes on v4x, eight on v4 and v3, the two scalar
    kernels in turn on the scalar tier). Bits: 110 of 110 dumps at revisions 5/6 and at revision 7; 72 revision-7
    dumps (B and Tga, three photos, v4x and v3, 1 to 32 threads) identical to v4x one thread;
    `fused_moment_pass_matches_the_two_passes` compares the fused pass with the two passes at every tier
    permutation over twelve geometries (remainder columns, the panel edge, heights 1 to 148) and fails when the
    eight-lane kernel reads a wrong vertical tap. Paired, 10 rounds, against item 19 (B at revision 7,
    `rev7_blockwalk_vs_fusedhv_jobs_*`): eight threads −3.62% (−4.61..−1.73) at 1507×2048 and +0.11% (−1.72..+3.71)
    at 1024², sixteen on eight cores −21.99% / −16.36%, 32 threads −10.79% / −3.06%, v3 at eight threads −1.88% /
    −0.51%. Four threads measured +2.41% (+0.89..+8.00) / −2.99%, and the same pass in the one-thread walk +1.02% /
    +1.01% / +1.95% on v4x (1507×2048 / 256² / 1024², `rev7_blockwalk_vs_fusedhv_serial_t1`): a panel's
    horizontal pass reads each row in 1 KiB runs, shorter than the stream prefetchers like, and an uncontended L3
    absorbs the two-pass traffic. So pools below eight threads and the serial walk keep the two passes.

21. **The activity plane in one pass per column panel** (strip jobs in pools of sixteen or more threads):
    `|src − mu1|` went to one image-wide plane, its horizontal blur to another, and the vertical blur read that.
    `blur::activity_at_revision` computes each row's panel span of `|src − mu1|` into a row buffer, its horizontal
    sums into the panel buffer and the vertical blur from there, with the three kernels' operations at every tier.
    Bits: 110 of 110 dumps at revisions 5/6 and at revision 7; 72 revision-7 dumps (v4x and v3, 1 to 32 threads)
    unchanged; `fused_activity_pass_matches_the_two_passes` compares it with the two passes at every tier
    permutation over thirteen geometries and fails when the eight-lane kernel reads a wrong vertical tap. Paired,
    10 rounds, against item 20 (`rev7_fusedhv_vs_fusedact_*`): sixteen threads on eight cores −13.49% (−15.00..−12.52)
    at 1507×2048 and −4.91% at 1024², 32 threads −8.09% (−13.14..+3.94) / −7.93% (−10.07..+0.77), but eight threads
    +1.43% (−1.42..+2.30) / +1.02% (+0.32..+1.96) and v3 at eight +0.59% / +1.28%, so only pools of sixteen or more
    threads take it.

22. **A lean sixteen-column loop for the v1 fold** (revision 7, AVX-512 tiers): the v1 SSIM/edge/HF fold kernel
    (`fused_vblur_ssim_canon64_vec`, 10.4% of one thread) tested about a dozen work flags per 16-column step, re-read
    from the stack (compare and branch instructions held 21.7% of its samples), and loaded, added and stored each pool's
    row lanes per step (stack moves 20.8%, memory-operand adds 12.3%). B and Tga run one configuration on every
    channel and scale (radius 0, direct SSIM, edges and HF on, no peaks, extensions, free extras or stores), so that
    configuration now has its own loop: its fourteen row pools stay in registers between folds and no flag is tested
    per step; each lane adds the same values in the same column order. Bits: 110 of 110 dumps at revisions 5/6 and at
    revision 7, 72 revision-7 thread and tier dumps unchanged; `vblur_ssim_rev7_lean_loop_matches_every_tier` compares
    the kernel at that configuration on every compiled tier with the scalar body (widths 16 to 4133, so the
    4,096-column fold falls inside the loop) and fails with the high-pass term in its revision-6 form. Paired (one
    thread 8 rounds, eight threads 10; `rev7_fusedact_vs_leanv1_t1`, `_t8`): v4x −3.86% (1507×2048), −2.42% (256²),
    −4.72% (1024²); eight threads −4.11% / −4.66%; v3, which keeps the eight-column loop, −0.31% / −0.91% / −0.10%.

23. **The same configuration in the v1 fold's eight-column loop** (revision 7; v3, and the odd last chunk on AVX-512):
    AVX2 has sixteen vector registers, too few for the fourteen pools' two half-row accumulators, so this loop keeps
    the pool updates in memory and drops only the per-chunk flag tests. Bits: 110 of 110 dumps at revisions 5/6 and 7;
    the lean-loop tier test now also fails on v3 when the high-pass term takes its revision-6 form. Paired, 8 rounds,
    one thread (`rev7_leanv1_vs_lean8_t1`): v3 −1.20% (1507×2048), −0.55% (256²), −1.61% (1024²); v4x −0.09% /
    +0.10% / +0.11%, inside noise. Holding the dense kernel's twenty-six row pools in registers the same way was
    slower on v4x (28.7 → 33–35 ms at 1507×2048, unpaired): with the twenty-nine term vectors live as well, the
    register allocator spilled more than the explicit updates move.

24. **The opsin cube root without sign handling** (revision 7, v4x/v4/v3): the opsin mix ends in `max(0)`, so its
    three cube roots never see a negative input or `−0`, and `cbrt_newton_x16`'s absolute value and sign restore are
    no-ops there. `cbrt_newton_nonneg_x16` drops them (it keeps the `+0` case); every lane has the full form's bits.
    Bits: 110 of 110 dumps at revisions 5/6 and 7, 72 revision-7 thread and tier dumps unchanged;
    `cbrt_newton_nonneg_matches_scalar` compares it with the scalar root on every compiled tier over `+0`,
    subnormals, all of `[1, 8)` and every 97th positive bit pattern to `+inf`, and fails without the `+0` case.
    Paired, 8 rounds, one thread (`rev7_lean8_vs_nonnegcbrt_t1`): v4x −0.74% (1507×2048), −0.76% (256²), −1.51%
    (1024²); v3 −2.31% / −2.64% / −2.46%.

**Tried and not kept** (after 23, all bit-identical): the gradient kernel's six 16-lane row-lane arrays held in
registers across its steps: no change (28.42–28.55 against 28.43–28.53 ms, unpaired; LLVM already kept them there).
`|src − mu1|` formed a row at a time inside the activity's horizontal blur instead of as a plane (the difference pass
is 1.8% of one thread): no change at one thread (28.5–28.6 → 28.6–28.8 ms, unpaired), as the per-row form tried after
13 found. A software prefetch four rows
ahead in the vertical blurs, where 36% of the load-stall cycles fall (13.5% of all cycles wait on a load): paired,
v4x −1.05% at 1507×2048 but +0.60% at 1024², v3 +0.08% / −0.32% / +0.58%, eight threads +0.67% / +0.38%
(`rev7_lean8_vs_vblurprefetch_t1`, `_t8`); eight rows ahead was no better, sixteen into L2 worse. The phase-B kernels
hold 14% of the load stalls (dense 9.5%, gradient 2.4%, v1 fold 2.1%), so fusing them into one pass per row band
would not remove much waiting: the dense kernel is bound by its twenty-six pools' per-step updates.
After 24, the AVX-512 vertical blur as a column walk (each 16-column vector down the whole height, its four pair and
eight quadruple tree nodes in registers through an eight-row unroll, one load per row, no node stores; bit-identical,
the fused-pass tier tests pass against it): 28.2–28.4 → 27.8–28.2 ms at 1507×2048 but 9.7 → 13.3 ms at 1024², one
thread, unpaired. With a power-of-two row stride every input load shares its low twelve address bits with the output
stores still in flight from the rows above (4K aliasing); the panel order loads a row segment before it stores into
the same columns.

**Tried and not kept** (after 21): the horizontal moment blur rewritten to share each output's pair and quad tree
nodes between adjacent outputs (the same trees, half the adds), once through row buffers and once through `valignd`
on adjacent vectors, both bit-identical: 29.7 → 32.1 ms and 29.6 → 31.1 ms at 1507×2048, one thread, v4x (unpaired).
The buffered form reloads windows straddling just-stored vectors, which store forwarding cannot serve; the shifted form
costs more in shuffles than the eleven L1 tap loads it replaces.

**Tried and not kept** (after 13): the three opsin cube roots interleaved step by step (bit-identical): −0.06% /
−0.90% on v4x (`rev7_band_vs_cbrt3`, the 1507×2048 and 1024² cases), noise on v3; the conversion's inner loop runs at
about two instructions per cycle, but interleaving did not raise it. The activity chain's `|src − mu1|` computed per
row into L1 and blurred from there instead of a strip plane (bit-identical): v4x −0.02% to −0.22%, v3 +0.30% to
+1.15% (`rev7_band_vs_actfuse`): the per-row blur calls cost what the saved L3 round trip gave.

**Cumulative after 13** (paired, 8 rounds, one thread; `rev7_bad544a3_vs_f96d7218/grid.tsv` and, one `f96d7218`
binary with B's revision-6 against its revision-7 stamp, `rev6_vs_rev7_f96d7218/grid.tsv`):

| Tier | Comparison | 256² photo | 1024² plot | 1507×2048 photo |
|---|---|---|---|---|
| v4x | revision 7, `bad544a3` → `f96d7218` | 1.356 → 0.872 ms, −35.79% | 15.57 → 10.86 ms, −30.36% | 43.10 → 31.21 ms, −27.43% |
| v3 | revision 7, `bad544a3` → `f96d7218` | 1.660 → 1.110 ms, −33.16% | 21.12 → 15.60 ms, −26.16% | 59.91 → 43.98 ms, −26.58% |
| v4x | revision 6 → 7, `f96d7218` | 1.225 → 0.874 ms, −28.73% | 16.55 → 10.87 ms, −34.44% | 47.59 → 31.17 ms, −34.34% |
| v3 | revision 6 → 7, `f96d7218` | 1.550 → 1.111 ms, −28.30% | 22.64 → 15.65 ms, −30.80% | 64.62 → 43.94 ms, −31.97% |

The bit-identical steps (2, 5–8, 10–12) also speed up revisions 5 and 6, so the second pair of rows is revision 7's own
margin today. 95% intervals are in the grids (within 1.5 points of each median; within 0.6 for the revision-6 rows).

**Revision 7 against revision 6 now** (B's 55 dumps, one binary, the revision-6 and revision-7 stamps): 22,190 of
39,600 feature values move, by a median 1.1e-6 relative (90th percentile 1.7e-5, 99th 2.3e-4, largest 4.0e-3 on the
97×63 pair's slot 664, 5.733e-6 → 5.710e-6, an absolute 2.3e-8), and the score by at most 4.0e-5 points (median
7.0e-6). The 1.6e-8 median and 7.1e-6 score bound above predate 4 (the difference forms and the cube root); 13 adds at
most 1.8e-11.

**Accuracy of 4 against the exact f64 oracle** (`rev7_forms.py --exact-stable`, the 93 near-identical pairs; outputs
under `/mnt/v/output/zensim/byconstruction-2026-10-22/rev7/rev7c1_accuracy/`): per-feature median error p90 2.80e-6
(revision 6) → 2.39e-6, p99 3.80e-5 → 3.78e-5; HF magnitude loss 9.2e-6 → 4.4e-6, edge-width change 1.6e-5 → 9.5e-6;
the worst feature, `peaks_ssim_max_s3_b`, 4.65e-5 → 5.91e-5 (the B-channel SSIM peaks move with the cube root's error
pattern).

**Tried and not kept**: a division-free cube root (two Newton steps and a scaled correction; within 1 ulp, 91.4%
correctly rounded) made the opsin step slower, because it is bound by FP op count rather than the divider (13.1% →
15.1% of samples); spreading revision 7's vertical blockiness lattice over eight lanes removed the lane-0 add chain but
not the time (1.09 against 1.11 ms; the scalar divisions bound it); a 16-wide loop at revision 5 (above).

**Where the time goes now** (B, revision 7, the 1507×2048 photo, one thread, after 23; harness PNG decoding
excluded): v4x — the dense kernel 20.5%, sRGB → XYB 18.0% (opsin math and the cube root), the vertical blurs 14.5%, the
horizontal moment blur 13.3%, the gradient kernel 7.3%, the v1 fold 6.1% (10.4% before 22), the activity plane's
horizontal blur 4.9%, downscaling 2.6%, blockiness 2.3%, copies 1.9%, the activity difference 1.9%; v3 — dense 22.1%,
XYB 20.1%, horizontal moment blur 14.6%, vertical blurs 10.8%, gradient 9.1%, v1 fold 8.7%. Retirement waits on a
load for 13.5% of cycles (v4x, `perf stat`, 30 iterations), 36% of those in the vertical blurs and 14% in the phase-B
kernels. The dense kernel's samples are mostly its twenty-six pools' per-step load/add/store (stack moves 40%,
memory-operand adds 16%); the conversion is bound by its three cube roots' operations. One thread is now 28.4 ms (v4x)
and 41.7 ms (v3) at 1507×2048, against 43.2 / 59.7 ms at revision 7's implementation commit (separate runs); eight
threads 4.84 ms (v4x).

Open: NEON still runs the scalar canonical bodies (magetypes' NEON `max`/`min` propagate NaN where the canon's
`f32::max` does not; `FMAXNM`/`FMINNM` would match it); multithreaded scaling is bound by the strips' memory traffic
(about 430 bytes per pixel move from L3 to L2 at 2048²; a line-buffer walk would keep them in L2).

### The JPEG XL loop at low distances: why it misses the requested distance (2026-10-24)

Owner report: "at .45d, the jxlencoder doesnt converge on the reqhested distance". The workbench's loop player runs
jxl-encoder's perceptual loop (`butteraugli-loop` with an external metric) at the requested butteraugli distance,
steered by butteraugli itself or by a module map placed on butteraugli's scale (`scale · e^exponent`, score = the
map's largest value). Reproduced natively by `zensim-workbench/loop-converge` (a standalone crate: `loop-converge`
mirrors the wasm codec's adapter, `map_probe` reads one pair's signed map under every map mode, `loop_pairs` writes
calibration pairs), on a 512² center crop of the speed pass's CLIC photo, effort 7, the player's 4 iterations.
Final score ÷ requested distance:

| Steering | d 0.3 | d 0.45 | d 0.6 | d 1.0 | d 2.0 |
|---|---|---|---|---|---|
| butteraugli (built in) | 1.27 | 1.28 | 1.36 | 1.30 | 1.19 |
| B `diffmap` (shipped placement) | 4.86 | 1.84 | 1.65 | 1.62 | 1.15 |
| Tga `diffmap` | 1.85 | 2.13 | 1.46 | 1.46 | 1.49 |
| B `blocks8` | 8.09 | 5.42 | 3.01 | 1.37 | 1.13 |

1. **Butteraugli meets the loop's own rule but not the max.** The loop (libjxl's FindBestQuantization) bumps every
   8×8 block whose tile distance, 1.2 × the block's 16-norm, exceeds the target. With 8 iterations at d 0.45 it
   reaches a fixpoint at the seventh frame with no block above 0.45 and the quant field far below its cap (4.49 of
   13.3), while the pixel maximum the player shows is 0.548. The final maximum is 1.06–1.22× the largest tile
   distance at the five distances above.
2. **A zensim map's level falls too slowly near lossless.** Plain encodes of the crop (no loop): B's deficit falls
   like butteraugli's maximum to the power 0.81 between d 1 and d 0.3 but only 0.48 between d 0.2 and d 0.05
   (B 85.16 at d 1, 93.79 at d 0.3, 95.32 at d 0.2, 97.66 at d 0.05). On calibration pairs at 12 distances
   (`loop_pairs`: the 17 CID22-512 training images of the earlier calibration, the 10 gb82-sc screenshots held
   out, 0.1 to 6), the shipped placement's mean log ratio to butteraugli's block distances rises from −0.50 at
   d 6 to +1.39 at d 0.1 (B `diffmap`, fit images), the same ramp for every map and bake. The shipped placement was
   fitted at 0.5–6 only; refitting one power law over 0.1–6 drives the exponent to 0.15–0.2 and keeps the ramp,
   and per-distance fits pick exponents of 0.1–0.25 (within one quality level the zensim block map barely predicts
   butteraugli's). Correcting only the scale per distance (shipped exponent, scale × e^−bias) brings d 0.3 to
   1.35× at the butteraugli loop's bytes but leaves d 0.45 at 1.96×.
3. **Fourth-power pools make the map spike at the worst pixel.** From iteration 2 at d 0.3 the B map's maximum sits
   at pixel (1, 417) of the left edge (density × N 3150–4980 under every window and paint mode with the exact
   split, 750–1170 with the repair split; the map total is about 4); repairing that one pixel raises B from 95.9002 to 95.9171. Its contributions come from
   `v2_ssim_dev4_s0_y` (−14% from the one pixel), `basic_edge_det_4th_s1_y`, `basic_ssim_4th_s0_y` and
   `basic_edge_det_4th_s0_y`. The reconstruction there is a real +4/−3 level error in textured content (local std
   6.2 levels), not a window artifact; a fourth-root moment's exact share grows as 1/f³ while the rest of the
   image improves, so the largest value follows whichever pixel is currently worst and jumps between iterations.
4. **jxl-encoder's own zensim loop clamps below d 0.5.** `zensim_targets.rs`'s distance table starts at 0.5 and
   returns its first value for anything lower, so `--zensim-iters` at d 0.45 aims at d 0.5's target; the table is
   also seeded against profile A.

Outputs: calibration pairs `/mnt/v/zenmetrics-private/aic4/loop-pairs/2026-10-24-hq/` (dev; 324 pairs), fits
`/mnt/v/zenmetrics-private/zensimx-workbench/loop_calibration_2026-10-24hq.tsv`.

### Coarse scales in linear light: a label-free probe (2026-10-24)

The first revision-7 candidate from the HaarPSI / HaarXYB comparison: zensim converts each image to positive XYB at full
resolution and builds scales 1–3 from 2×2 averages of XYB values; HaarXYB averages linear RGB and converts each level.
`scripts/byconstruction/linpyr_probe.py` builds both pyramids for every TRAIN pair (KADID train split, TID2013, KonFiG
train split; keys only, no labels; the 8,165 pairs whose pixels differ) and reports, per distortion type, scale and channel, the median log ratio
of two damage measures, linear-light over XYB-averaged: the mean squared XYB difference, and the mean squared difference
of 7×7 box detail (pixel minus local mean). Outputs and SHA256SUMS: `/mnt/v/output/zensim/byconstruction-2026-10-22/linpyr/`.

The ratio is not uniform across types. Detail damage, scale 1 / 3, X, Y, B channels:

| Type | Scale 1 | Scale 3 |
|---|---|---|
| KADID impulse noise / TID2013 impulse noise | ×2.1–2.5 / ×1.9–2.1 | ×3.9–5.3 / ×2.7–3.8 |
| KADID Gaussian, lens, motion blur | ×1.0 | ×1.5–2.0 |
| TID2013 Gaussian blur | ×1.0 | ×1.4–1.8 |
| KADID sharpen / KonFiG high sharpen | ×1.0–1.1 | ×1.2–1.6 / ×1.1–2.3 |
| TID2013 JPEG transmission errors | ×1.0 | Y ×7.2 |
| TID2013 masked noise | ×1.2 | ×1.2–1.6 |
| global edits (KADID 16–18, 25; TID2013 16–18) | ×0.95–1.05 | ×0.94–1.31 (TID mean shift Y) |
| non-eccentricity patch / pattern noise | ×1.0 | ×1.0–1.06 |
| denoising (KADID 15 / TID2013 9) | ×1.0 | ×1.0–1.06 |

Linear-light averaging keeps the energy of outliers and sharp edges that the cube root compresses before XYB averaging, so
outlier- and edge-heavy damage weighs more at coarse scales, while global edits, displacement and denoising keep their
coarse damage. The types the monotone arms over-penalise (contrast change, non-eccentricity, saturation, denoising) are
not among those that move; a gain for monotone models would come indirectly, if lower coarse-feature weights then fit
the outlier-heavy types. The probe shows the change is type-dependent, not that it helps.

**Experiment (recorded before any result).** A research toggle builds scales 1–3 from 2×2 averages of linear RGB,
converted to positive XYB per level, in both pyramid producers; scale 0 and every default path keep their bits. The eight
training and assessment sets are re-extracted with it into a derived root (same feature IDs, its own era label) and the
strictly monotone frontier recipe (`mono:thr:mp64`, M64) and the best-balance recipe (Tga) are trained on it: seeds 0–2,
the twelve E21 cells each. Reported against the same recipes on the XYB-averaged root: E21 Δ and per dataset, per-type
bias, smoothness, both mixture tests. It joins revision 7 only if E21 improves by more than the seed-unit SE with no new
mixture inversions.

*Amendment, also before any result:* M64 only. Tga's inputs include the `byc_globaff` descriptor, whose bank columns
came from the separate `byc_globaff` example with its own per-scale planes; re-deriving those under the toggle is a
second producer change, so Tga waits for an M64 result worth following.

**Implementation** (`ec21497b`). `ZENSIM_PYRAMID=linear` (unset or `xyb` keeps the default; other values panic):
alongside each scale-0 row's XYB conversion, the walk keeps the row in linear RGB (the sRGB8 table), averages 2×2 blocks
of linear RGB with the same `downscale_2x_into` and row targets, and converts each coarse level to positive XYB at the
walk's revision. Both pyramid producers carry it: the streaming producer and the block walk's chunk tasks. It refuses
scale-0 chroma omission, cached references, sampling and HDR, and the materialized pyramid builders refuse it, so no
other path can mix pyramids silently. Research manifests record `"pyramid"`.

- Unset: 110 of 110 score-and-feature dumps identical at revisions 5/6 and at revision 7 (`perf_bits.sh`).
- `linear_pyramid_producers_match_the_whole_image_form`: every strip window of both producers, both sides, every
  channel and scale, bit for bit against the whole-image form (one conversion call per level), revisions 5 and 7, six
  sizes down to odd and short ones; the coarse levels differ from the XYB-averaged ones (negative control).
- Set, on 14 KADID pairs: the 141 measured columns that read only scale 0 keep their bits; the other 435 move: every
  column of scales 1–3, plus scale 0's edge-width change, a two-scale ratio (scale 0 against scale 1). Served features
  (`score_pair_with_bake`, an M64 bake) equal the extracted ones on all 410 inputs of that bake, both arms.
- The sibling codecs now use zencodec's incremental-decode API, so `zensim-bench` pins zencodec `d327434b` (their
  rev) instead of `4fda4391`. With it, the 14 pairs reproduce the stored Rev5 bank bit for bit; `ring_root.py build`
  checks the same identity on every row of every table.
- `ring_root.py` and `ring_bank.sh` take the arm (`--arm linpyr`): era `rev5_localwin_linpyr`, feature-set id
  `basic+peaks+v2@w1825/rev5_localwin_linpyr#36c3f3af` (registered, admitted like the `hpm` root).

**AVIF decode drift, and a second amendment before any result.** The derived-root build refused safesyn: 33,384 of its
34,001 AVIF rows (`zenavif-s5-e6`) differ from the stored root in columns the toggle cannot touch, while every JPEG,
JXL and WebP row and every row of the seven other sets is bit-identical. With the toggle off, the extractor built
today (`678e280c`, zencodec `d327434b`) decodes different pixels from the 2026-10-23 extractor (`1152f194`, which
reproduced the stored root): 10 of 10 sampled AVIF files differ, their references and 3 of 3 JXL files do not. Neither
decode equals libavif's (Pillow 12.3; ours differs from it by at most 2 levels, on 2–55% of values per image). Against
an exact float64 conversion of the same 10-bit YUV (ffmpeg decodes the YUV; the files are CICP 1/13/6, full range,
4:4:4), today's decode is off by one on 4.54% of values (705,672 of 15,544,662 over the 10 files, always ±1, at values
a few hundredths of a level from a rounding boundary), and the 2026-10-23 decode equals the exact conversion bit for bit
on 2 of the 10 (where today's is off on 0.015% and 0.057%); libavif is further off (45–76% exact, up to 3 levels). The
AVIF route is zenavif Rgb16 → `zenpixels_convert::RowConverter` → RGB8_SRGB. The sibling codecs have not moved since
2026-10-19 and zensim-bench pinned zencodec `4fda4391` throughout; what made the 2026-10-23 build compile and decode
differently is not yet known.

The pyramid comparison must hold pixels fixed, so: the safesyn and coverage-pool sets are extracted again with the
toggle off and the same binary, into a control root (era `rev5_localwin_avifdec`) that differs from the stored root
only in AVIF rows; the linear-pyramid root takes its AVIF rows from the same decode. The baseline becomes **M64ctl**
(the M64 recipe on the control root); the stored M64 stays as a reference, and one M64 cell (KADID, seed 0) retrained
on the stored root with today's trainer checks that the trainer reproduces it. The adoption rule is unchanged and
applies to M64lin against M64ctl.

*Cause found, and the control withdrawn (still before any result).* `byc_avif_route` (new, `zensim-bench/examples/`)
decodes through the owner and projects zenavif's native buffer (Rgb16, sRGB-tagged, BT.709, full range) three ways. On
10 of 10 files the owner's projection, with or without the transfer tag, gives today's pixels, and rounding the same
16-bit samples by hand (`round(v · 255 / 65535)`) gives the 2026-10-23 pixels: zenavif's samples did not change, the
16→8-bit narrowing did. The lockfile resolves zenpixels-convert from the registry (0.2.16, the newest published), whose
narrowing is not correctly rounded; the local, unpublished 0.2.17 narrows as `round(v / 257)` (zenpixels `d019ae9`,
2026-09-23). With 0.2.17 patched in (`zensim-bench/Cargo.toml`, dropped once it is published), the extractor reproduces
the 2026-10-23 extractor bit for bit on 13 of 13 pairs (10 AVIF, 3 JXL: pixel hashes and all 7,488 feature cells). The
stored root's AVIF pixels are the correctly rounded ones. The linear-pyramid bank is extracted again with that build,
the comparison returns to the registered baseline (the stored M64), and the control arm is dropped; its partial bank is
kept as `linpyr_ctl.superseded`. The reproduction cell stays.

**Derived root** (`/mnt/v/output/zensim/byconstruction-2026-10-22/linpyr/root`, extractor `67c82b65` at `d13211dd`,
1,177 s on 20 threads, peak RSS 7.06 GiB): in every table of the stored root, including all 196,086 safesyn rows and
the 42,021-row coverage pool, the 141 columns that read only scale 0 are bit-identical, and 93.4–97.9% of the 435
replaced cells change (per table; the lowest are safesyn's). The drifted first extraction is kept under `linpyr/superseded-narrowing-0.2.16/`. M64lin (the
M64 recipe, 12 cells) trains on it with trainer `ede5b94c`, the build the reproduction cell uses.

*Exploratory, label-free, not a selection criterion:* on the mixture steps (D1 → D2∘D1, each adding error energy),
the 370 M64 inputs that the pyramid changes fall less often under linear light. Standard set (1,152 steps): mean share
of steps on which an input falls 0.184 → 0.171, inputs falling on more than 25% of steps 85 → 53. Global-edit set
(1,728 steps): 0.099 → 0.092, inputs falling on more than 10% of steps 162 → 146 (`mixture_test.py feature-steps`;
tables and SHA256SUMS in `linpyr/mixture_features/`). The monotone recipe treats every input as damage, so a falling
input on an added-damage step is one way a monotone model can invert.

*Trainer check:* the M64 KADID seed-0 cell retrained on the stored root with trainer `ede5b94c` (`M64repro`, 3,557 s)
predicts all 7,869 KADID rows bit-identically to the stored M64 cell (prediction SHA-256 `3026b06d…` both), with the
same validation curve; the bake files' bytes differ, their outputs do not. So stored M64 is the right baseline for
M64lin.

**Result: not adopted.** M64lin (12 cells, trainer `ede5b94c`, 3,224–3,663 s each) against stored M64, out of sample:

| | M64 | M64lin | M64lin − M64 |
|---|---|---|---|
| E21 Δ against `base-c` (seed-unit SE) | −0.0131 (0.0023) | −0.0134 (0.0034) | **−0.0003** (paired SE 0.0012) |
| KADID / TID2013 / KonFiG / CID22-A Δ | −0.0166 / −0.0215 / −0.0150 / +0.0005 | −0.0042 / −0.0282 / −0.0250 / +0.0036 | **+0.0124** / −0.0067 / −0.0101 / +0.0031 |
| Z-RMSE KADID / TID2013 / KonFiG / CID22-A (mean) | 0.487 / 0.441 / 0.591 / 0.489 (0.502) | 0.463 / 0.447 / 0.606 / 0.482 (0.499) | |
| Standard mixture: inversions per cell, total, max rise | 1–6, 38, +7.56 | 2–5, 36, +3.91 | 10 steps newly invert, 12 stop |
| Global-edit mixture | 0, max −0.05 | 0, max −0.25 | |
| AIC2026 public_train zero-shot, mean over cells (mean prediction) | 0.9080 (0.9093) | 0.9048 (0.9059) | −0.0032; 2 of 12 cells better |

Per seed the paired difference is −0.0002, −0.0025, +0.0018. E21 does not improve, so the registered rule keeps linear
light out of revision 7. What it does: coarse damage measured in linear light helps KADID a lot (+0.012, Z-RMSE 0.487 →
0.463) and CID22-A a little, and costs TID2013 and KonFiG about as much. Per type (`type_bias.py`), the over-penalties M64
puts on global edits shrink slightly (KADID contrast −0.121 → −0.108, non-eccentricity patch −0.105 → −0.099, TID2013
pattern noise −0.198 → −0.186) while TID2013 masked noise (+0.026 → +0.006) and sparse sampling (−0.018 → −0.033)
move the wrong way. The halved worst mixture rise and the fewer falling inputs on added-damage steps are real but do not
buy E21. The toggle stays as a research arm; the AIC zero-shot rows (an unregistered view: the cells never fit AIC) use
`linpyr/aic_public_train/` and `byc_cells.py aiczero`.

*Post hoc, chosen after seeing the result above:* the two arms err on different datasets, so their plain mean,
cell by cell (**M64ens**: the average of the M64 and M64lin outputs; no weights fitted; monotone in every input because
both are), scores E21 −0.0121 (SE 0.0029): +0.0010 over M64 with paired SE 0.0006 (per dataset −0.0088 / −0.0234 /
−0.0191 / +0.0028; mean Z-RMSE 0.499). Its standard-mixture scores, the per-image mean of the two arms' stored scores,
give 2–5 inversions per cell (38 in all, max rise +4.74) and the global-edit set none. It is the best strictly monotone
E21 so far, but it was picked after the fact and needs both pyramids, so it is a lead, not a selection. Tga (descriptor
sign-free, AIC leg) still has the best E21 of all arms: −0.0089 (SE 0.0038). The dashboard's scatters and arms table
include M64lin and M64ens (`aic_dashboard_data.py`, `POSTHOC`).

### Orientation-selective bands: a label-free probe (2026-10-24)

The second missing idea from the HaarPSI / HaarXYB comparison: every zensim edge and texture feature is isotropic.
`scripts/byconstruction/haar_orient_probe.py` takes the three Haar detail bands (horizontal, vertical, diagonal) of
each scale's 2×2 blocks of zensim's own pyramid (Y, scales 0–3) for the same 8,165 TRAIN pairs as the linear-light probe
(keys only, no labels), and reports per distortion type the diagonal share of the band-difference energy and the
horizontal/vertical asymmetry `|E_H − E_V| / (E_H + E_V)`. Outputs and SHA256SUMS:
`/mnt/v/output/zensim/byconstruction-2026-10-22/haar_orient/`.

- Natural content puts 0.09–0.12 of its band energy in the diagonal band (per scale and set).
- Additive noise puts a third there with no asymmetry (KADID 11–14, TID2013 1/2/6/19, KonFiG multiplicative noise:
  0.31–0.33, asymmetry ≤ 0.06); spectrally shaped noise more at fine scales (TID2013 masked noise 0.48, high-frequency
  noise 0.45, scale 0).
- Directional damage is strongly asymmetric: KADID motion blur 0.37 → 0.61 (scale 0 → 2), TID2013 chromatic aberration
  0.45–0.57, TID2013 JPEG2000 transmission errors 0.26–0.40; colour block (KADID 23) and colour saturation (KADID 7)
  have the lowest diagonal shares (0.05–0.11).
- KADID's Gaussian and lens blur move damage toward the diagonal band with scale (0.14–0.15 at scale 0, 0.31–0.32 at
  scale 3); TID2013's Gaussian blur stays at 0.14–0.22.

So orientation separates noise-like from structural damage, and directional from isotropic, where our features
cannot. The visual system is less sensitive to oblique orientations at high frequencies, so a damage feature split by
orientation could let a monotone model weight noise below structure. This says orientation is missing information, not
that it helps a model.

**Experiment (recorded before any result).** `byc_orient` (`zensim-bench/examples/`) writes, per pair, the RMS of the
horizontal, vertical and diagonal Haar bands of the error `dst − src` at every scale (0–3) and XYB channel: 36
non-negative damage values from zensim's own conversion and 2× downscale through the shared decode owner, as the
globaff sidecar was built. `ring_root.py orient` widens the stored root with them at research ids f2301–f2336
(`basic+peaks+v2@w2337/rev5_localwin_orient#24a93918`, admitted like the other derived roots); declaration `fx1o` is
fx1 plus the 36 as unpaired direct inputs. The strictly monotone recipe trains on it (`mono:thr:mp64` on
`sel:46516c473ae7@h32:H128:cv16:cf98:fx1o`, **M64o**, 12 E21 cells, seeds 0–2) and is compared with M64: E21 Δ and per
dataset, per-type bias, smoothness and both mixture tests (mixture images through the same sidecar and `byc_predict`).
It is a candidate for a served family only if E21 improves by more than the seed-unit SE with no new mixture
inversions; its compute cost is measured only then.

**Result (M64o, 12 cells, seeds 0–2; pinned binaries `bdc6b7662175`).** E21 Δ **−0.0168** (seed-unit SE 0.0020;
per seed −0.0152, −0.0208, −0.0145), against M64's −0.0131. Paired with M64: **−0.0037 (SE 0.0003)**, worse in every
seed (−0.0041, −0.0031, −0.0038). Per dataset against M64: KADID +0.0020, TID2013 −0.0147, KonFiG −0.0035, CID22-A
+0.0015. Mixture tests through the table route (`mixture_table.py build --orient`: B's computed vector plus the
`byc_orient` sidecar; on an M64 cell it reproduces the stored pixel-route scores on every image of both sets): standard
set 2–18 inversions per cell, 57 in all (M64 1–6, 38), largest rise +7.56 (M64 +7.56); global-edit set none in any
cell. **Not adopted:** E21 is worse by 13 seed-unit SE and the standard mixture gains inversions. The bands do carry
type information (the probe above), but the monotone recipe spends it on KADID and CID22-A and loses TID2013. A
reproduction cell (M64's KADID-held-out seed-0 cell, retrained in the same run) gives predictions bit-identical to the
stored M64 cell (`3026b06d…`). Smoothness (`aic_panel.py --no-public-test`, out of sample): mean Z-RMSE over the
four E21 sets 0.505 against M64's 0.502 (TID2013 0.457 against 0.441; KADID 0.483 against 0.487, CID22-A 0.486
against 0.489, KonFiG 0.594 against 0.591).

### DVIFM 0.3 on the E21 sets and the mixture test (2026-10-23)

Owner request: "eval dvifm 0.3 on kadid and other distortion//corrputions". Scorer: the frozen runtime of the
2026-10-20 AIC2026 run (dvifmish `31c4e256`, preset `port-lap-v0.3`, float path, binary SHA-256 `4827663c…`), which
`dvifm_e21.py` checks before scoring. The author trained its constants on AIC2026 (compression only), so the E21 sets
and the mixture set are unseen by it; nothing is fitted or calibrated. Pairs are the E21 fold tables' keys joined to
the promoted feature bank's paths: 13,817 rows, 13,814 unique pairs, in E34 panel-input order (reference names and
labels equal row for row). 13,814 pairs took 4.5 s on 16 threads and the 1,344 mixture images 0.1 s (run-heavy peak RSS
0.16 GiB). Outputs on dev: `/mnt/v/output/zensim/byconstruction-2026-10-22/dvifm03/` (`rows.parquet` `0603444e…`,
`pairs.tsv` `b043e1bd…`, `dvifm.tsv` `dee17427…`) and `mixture/scores_DVIFM03.json` (quality = −JND).

| Dataset | DVIFM 0.3 (zero-shot) | `base-c` (out of sample) | Tga (out of sample) |
|---|---|---|---|
| KADID TRAIN+SELECT | 0.787 (Z-RMSE 0.623) | 0.891 (0.454) | 0.879 (0.480) |
| TID2013 | 0.788 (0.571) | 0.901 (0.423) | 0.879 (0.442) |
| KonFiG TRAIN+VAL | 0.733 (0.666) | 0.817 (0.567) | 0.807 (0.581) |
| CID22-A | 0.831 (0.554) | 0.880 (0.488) | 0.890 (0.475) |

- **Within a type it is close; across types its scale is off** (`aic_panel.py --per-type`, SROCC within each type).
  KADID: at least 0.90 on 11 of 25 types, within 0.02 of `base-c` or better on 19, above both `base-c` and Tga on 9.
  TID2013: at least 0.90 on 11 of 24, within 0.02 or better on 12, above both on 4. KonFiG: above both on 3 of 7.
  CID22-A: above both on 1 of 8 codecs.
- Weakest types: KADID color saturation 1 0.476, non-eccentricity patch 0.495, color block 0.553, pixelate 0.598, mean
  shift 0.628; TID2013 contrast change 0.444, local block-wise distortion 0.519, mean shift 0.683, color saturation
  0.712; KonFiG color diffusion 0.519. Best relative to ours: KADID contrast change 0.751 (`base-c` 0.462, Tga 0.636) and
  color diffusion 0.900 (`base-c` 0.845).
- **Mixture test:** 31 of 1,152 steps raise its score (26 blur then luma noise, at most 0.53 JND; 4 posterize then chroma
  noise σ8; 1 blur then chroma noise σ8). B: 148; Tga: 1–4 per cell.
- **Smoothness:** the largest Z-RMSE of any model on every E21 set; on AIC2026 public_train, which it was fitted on, the
  smoothest fixed model (Z-RMSE 0.364).

### JPEG XL reachability with Tga: placement fits or a different map? (2026-10-24)

Owner, 2026-10-24: "lead with figuring out even reachability for jxl target distance with tga, and if we just need
better fits or integral changes". The workbench loop places a map on butteraugli's scale (`scale · v^exponent`, `v` the
map's density times the pixel count), compares each 8×8 tile's 16-norm × 1.2 with the requested distance, multiplies
an above-target block's quant field by tile / target (from iteration 2 on, only those blocks), and reports the placed
map's largest value as its score.

**One image first** (`loop-converge` as built at 08:23, single thread; CID22 training photo `1129482`, 512²,
effort 7):

- *The loop misses at every distance, not only near lossless.* Final score ÷ requested, 4 iterations, d 0.3 / 0.45 /
  0.6 / 1 / 2 / 4: butteraugli 1.07 / 1.07 / 1.11 / 1.06 / 1.04 / 0.99; Tga `diffmap` at its shipped placement
  (0.1330 / 0.6) 6.13 / 5.29 / 4.99 / 4.98 / 3.58 / 2.44, at more bytes than butteraugli (d 0.45: 29,712 against
  24,400). B's `diffmap` 7.15 / 4.90 / 4.56 / 5.61 / 4.88 / 2.36; Tga `blocks8` and `diffmap_px` alike. At d 0.45 Tga's
  loop stops at a fixed point after two adjustments (iterations 3 and 4 identical, 816 blocks above target).
- *Static reach.* Without the loop (frame 0, the encode before any adjustment) over encoder distances 0.05–10, Tga's
  largest placed tile never falls below 1.47 (d 0.05, where butteraugli's is 0.080): at the shipped placement no target
  below about 1.5 is reachable on this image at any uniform quality. The map itself does fall with quality: at
  exponent 1 its largest tile spans 50.6 → 3,810 over d 0.05 → 10 (butteraugli 0.080 → 7.25), a log–log slope of 0.96
  against butteraugli's; exponent 0.6 flattens that slope to about 0.5, which is the floor.
- *A placement that tracks the level converges.* At exponent 1 and scale 1.2e-3 (chosen on this image's static curve,
  so not a fit) the Tga-steered loop ends with no block above target at every distance, final 0.77 / 0.64 / 0.54 /
  0.88 / 0.89 / 0.51 of the request, at fewer bytes than butteraugli at the same request (d 0.45: 20,368 against
  24,400; d 1: 13,586 against 15,767); 16 iterations change nothing (a fixed point by iteration 3). Its first
  adjustment still makes the map 3–40× worse (d 0.3: 0.232 → 9.38) before iterations 2–3 repair it: the reclaim step
  coarsens blocks by (tile / target)^0.2, and an attribution map reads most blocks as nearly free (an L8 pool's pixel
  share goes as x⁸, a fourth-moment pool's as e⁴, so the map concentrates on the worst pixels).

**Fitting the placement to the tail** (`loop_calibrate` `LOOP_CAL_TAIL`, new; the calibration pairs of "The JPEG XL
loop at low distances": 17 CID22-512 training images fitted, 10 gb82 screenshots held out, 12 distances 0.1–6). The
shipped method fits every block with butteraugli ≥ 0.05 equally; over 0.1–6 it picks exponent 0.15 (Tga `diffmap`:
fit r 0.488, held out 0.447), which compresses the map until the loop cannot move it. Fitting each pair's largest tile
instead gives scale 2.2571e-2, exponent 0.7, fit r 0.900, held out 0.758 (99th percentile: 2.5142e-2 / 0.8, r 0.919 /
0.829). On the fitted photos the tail fit's bias (log placed − log butteraugli, largest tile) stays within ±0.13 from
d 0.2 to 4 (+0.41 at d 0.1, −0.25 at d 6). On the held-out screenshots Tga reads +0.97 to +1.61 above butteraugli for
d ≤ 1, a content-dependent level no single placement can serve.

**The tail placement on two images** (one CID22 training photo, one held-out gb82 screenshot; scale 2.2571e-2, exponent
0.7, 4 iterations). Final ÷ requested at d 0.3 / 0.45 / 0.6 / 1 / 2 / 4: photo 1.39 / 1.02 / 1.28 / 1.13 / 1.08 / 0.82
at 1–7% more bytes than butteraugli's loop (shipped placement: 6.13 / 5.29 / 4.99 / 4.98 / 3.58 / 2.44); screenshot 1.56
/ 0.94 / 0.92 / 1.40 / 0.93 / 0.96 (butteraugli 0.76–0.91). Blocks still above target after the last iteration were
still falling over the last steps (photo d 0.3: 147 → 74 → 42 after the first step sent the score to 13.5, 45× the
target), so the loop is converging, slowly. Static frame-0 tiles on the screenshot read about 3× butteraugli's at d
0.3–1, and the curve is not monotone at the finest encoder distances (0.46 at d 0.05, 0.29 at d 0.1).

**What the map tells the loop about the rest of the image** (the photo at d 0.45, frame dumps; tile / target per
8×8 block): butteraugli's median block sits at 0.20 of the target, 1% of blocks below 0.1, and the first step coarsens
the below-target blocks by a median factor of 0.73 ((tile / target)^0.2). Tga at the tail placement puts its median
block at 0.003 of the target, 78% of blocks below 0.1 and 62% below 0.01, and coarsens them by a median factor of 0.30.
The tail fit makes the worst blocks reachable; it does not change the map's spread across blocks, which an attribution
map gets from splitting pooled norms (shares ∝ e⁴, x⁸) and which no power-law placement can remove.

**Corpus test (recorded before its results).** `jxl_reach.sh` on the 78 TRAIN images of the steering study:
`static` (no loop, 20 encoder distances 0.05–10), `butteraugli`, `Tga` (shipped placement), `Tgatail` (tail-max fit)
and `Tgatail8` (8 iterations), distances 0.3–10, effort 7, every output probed by Tga's map at both placements, frames
dumped for the per-block response (`jxl_reach.py elastic`: ln(tile_next / tile) / ln(q_next / q) for adjusted blocks;
the bump rule assumes −1). Reading (my bars): placement fits suffice if, at the tail placement, every target from 0.3
is statically reachable on at least 95% of images and the 4-iteration loop ends with no block above target on at least
90% of photos, its largest tile within 0.8–1.25 of the request; otherwise, or if bumped blocks respond with a median
above −0.3, the map needs a different construction (for example a split that gives each pixel its local error
amplitude instead of its share). Screens are reported separately.

**Corpus results** (`jxl_reach.sh`, steer_rd `3ba4f4ba1cb8`, 153 s on dev, run-heavy peak RSS 2.31 GiB; tables and
SHA256SUMS in `/mnt/v/output/zensim/byconstruction-2026-10-22/jxl-reach/`, decoded images, bitstreams and frames in
`/var/tmp/byconstruction-jxl-steer/` on dev). Six CID22 images the tail fit used are left out: 72 images (34 photos, 20
illustrations, 18 screen crops of the held-out screenshots).

*Static reach* (no loop, 20 encoder distances 0.05–10; share of images whose Tga tile, on the decoded output, falls to
the target at some uniform quality):

| Placement | 0.3 | 0.45 | 0.6 | 1.0 | 1.6 | 2.5 | Floor median / p90 |
|---|---|---|---|---|---|---|---|
| Shipped (0.1330 / 0.6), all 78 | 1% | 14% | 47% | 86% | 95% | 100% | 0.668 / 1.376 |
| Tail (2.2571e-2 / 0.7), all 78 | 88% | 96% | 97% | 100% | 100% | 100% | 0.145 / 0.337 |
| Tail, photos | 94% | 97% | 100% | 100% | 100% | 100% | 0.133 / 0.202 |
| Tail, illustrations | 95% | 100% | 100% | 100% | 100% | 100% | 0.150 / 0.301 |
| Tail, screens | 72% | 89% | 89% | 100% | 100% | 100% | 0.243 / 0.502 |

Every image's curve rises at least once as the encoder distance falls (the largest value follows the worst pixel).

*The loop* (4 iterations unless noted; distances 0.3 / 0.45 / 0.6 / 1 / 1.6 / 2.5 / 4 / 6.4 / 10; "converged" = no
block above target at the end):

| Arm | Largest tile ÷ target, median | Converged | Bytes against butteraugli's loop |
|---|---|---|---|
| butteraugli | 1.00 / 1.00 / 1.00 / 0.99 / 0.98 / 0.95 / 0.89 / 0.83 / 0.77 | 54 / 63 / 50 / 72 / 85 / 100 / 100 / 100 / 100% | — |
| Tga, shipped placement | 2.91 / 2.21 / 3.23 / 2.22 / 1.55 / 1.28 / 1.12 / 1.01 / 0.95 | 0 / 0 / 0 / 0 / 1 / 11 / 18 / 49 / 79% | +72% at d 0.3, +34% at d 0.45 |
| Tga, tail placement | 1.04 / 0.98 / 0.99 / 0.96 / 0.89 / 0.81 / 0.71 / 0.56 / 0.44 | 47 / 58 / 57 / 68 / 83 / 93 / 96 / 99 / 99% | +2% at d 0.3, −11% to −16% above |
| Tga, tail, 8 iterations | 0.98 / 0.96 / 0.98 / 0.95 / 0.88 / 0.81 / 0.71 / 0.56 / 0.44 | 65 / 82 / 76 / 90 / 94 / 97 / 99 / 99 / 100% | the same bytes |

By class at the tail placement (converged at d ≤ 1, 4 / 8 iterations; butteraugli at 4): photos 71–85% / 91–100%
(butteraugli 18–65%), 14–19% fewer bytes than butteraugli at every distance; illustrations 45–80% / 70–95%
(45–75%); screens 6–22% / 11–67% (78–100%), 3–11% more bytes at d ≤ 1.

*The blocks' response* (`jxl_reach.py elastic`, median of ln(tile_next / tile) / ln(q_next / q) over bumped
above-target blocks; the bump rule assumes −1): butteraugli −0.97 to −1.11 at d ≤ 1.6. Tga at the shipped placement
−0.39 (d 0.3), −0.60, −0.79, −1.21 (d 1), with a quarter of its bumped blocks getting worse at d 0.3 and unadjusted
blocks changing by a median 16% there (their tiles depend on neighbours and on the pools' global normalisation). Tga at
the tail placement −2.6 to −4.1 at every distance: the blocks the loop bumps behave like shares of fourth-power pools
(∝ e⁴, so e^(4 × 0.7) ≈ e^2.8 of the error amplitude), so each bump overshoots, which is why it lands at 0.44–0.72 of
the target from d 2.5 up.

**Reading** (the rule recorded above, my bars): not met. The tail fit is what makes the target reachable at all,
which settles the d 0.45 report for photos (0.95 of the request, 82% converged at 4 iterations, 97% at 8), but static
reach of d 0.3 is 88% (bar 95%), photo convergence at 4 iterations is 71–85% at d ≤ 1 (bar 90%; butteraugli itself
reaches 18–65%), the landing ratio leaves 0.8–1.25 from d 2.5 up, and screens do not converge at low distances. So
better fits are necessary, and they are not enough: the remaining failures come from what the map is, a per-pixel
split of pooled power norms whose response to a block's quantization is −2.6 to −4 for the worst blocks, near zero and
coupled for most others, where the loop assumes −1. A map whose value per pixel is a local error amplitude fixes that
response but not the steering (tested below: it loses bytes at equal quality); a loop whose bump uses the map's own
response is untested. Whether the 14–19% fewer bytes on photos are the same quality is the steering study's question
(below: not on CVVDP).

**The integral change, tried: an amplitude split** (`SplitMode::Amplitude`, `ZENSIM_MAP_SPLIT=amplitude`, workbench
map `diffmap_amp`; `bb484919`). Every nonlinear pool (the basic L4/L2 pools and the peak cells' L8 slots, the v2 soft
peaks and the SSIM, GMS, artifact and detail deviation pools) is distributed as a mean pool of its per-pixel quantity,
`−s·v_i/N`, folded into the existing linear coefficients; linear, weighted, ratio and global terms keep their exact
shares, and the repair and exact maps keep their bits (592 library tests). A pixel's value then moves with its own
error as the pool moves with a uniform change.

- *Calibration* (tail max, the same pairs): Tga 4.7213e-2 / 0.9, fit r 0.831, held out 0.483 (exact split: 0.900 /
  0.758); M64a 4.6644e-2 / 1.0, **0.951 / 0.873** (0.877 / 0.786); B 4.5157e-2 / 0.9, 0.811 / 0.690 (0.807 / 0.778). The
  bulk fits rise from exponent 0.15–0.2 and r 0.48–0.52 to 0.35–0.45 and 0.62–0.67: block by block the amplitude map is
  closer to butteraugli's. Tga's held-out (screen) tail falls; its sign-free descriptor terms fold their sign into the
  map, which M64a (every weight non-negative) cannot do.
- *The response is fixed.* Median response of bumped blocks at d ≤ 1: Tga −1.07, −1.22, −1.06, −1.26; M64a −0.68,
  −1.21, −1.06, −1.07 (exact split at the tail placement: −2.6); unadjusted blocks no longer move (median |Δ log tile|
  0.000–0.016). B's response falls toward 0 from d 2.5 up (−0.15 to −0.02), unexplained.
- *Reach is worse.* Static reach of the Tga amplitude map (no loop, held-out images): photos 76 / 91 / 94 / 97% at
  0.3 / 0.45 / 0.6 / 1 (exact tail 94 / 97 / 100 / 100%), illustrations 85 / 90 / 100 / 100%, screens 56 / 78 / 78 / 83%,
  floors median 0.223 (photos) and 0.524 (screens; the dithered palette `windows95` crops 3.3–3.4 at d 0.05). A share
  suppresses small per-pixel values (v^p / Σv^p); the amplitude counts them linearly, so low-level per-pixel terms set
  a floor. The four-iteration loop now ends near the target (largest tile 0.94–1.27 of it at d ≤ 1.6) instead of below
  it, converging on photos 41 / 50 / 50 / 77% (Tga) at d ≤ 1 against 71 / 82 / 82 / 85% for the exact tail fit, and on
  screens 6–11%.

- *Bytes at equal quality* (the steering study's judges, the same 78 images; against butteraugli's loop,
  SSIMULACRA2 / CVVDP): Tga amplitude −3.98 [−5.83, −2.39] / −4.47 [−6.63, −2.57] over all images (Tga at the tail
  placement −1.54 / −2.01), photos −0.79 / −1.45 (+0.74 / −0.38), screens −12.67 / −13.96 (−6.77 / −6.58); M64a amplitude
  −2.48 / −2.74 (tail −1.04 / −2.09), photos +0.43 [−0.54, +1.13] / +0.55 [−0.35, +1.20], screens −9.65 / −11.41; B
  amplitude −9.08 / −8.44, screens −25.4 / −25.8.

**Reading.** Fixing the per-block response does not make the map a better steering signal: with every pool counted
linearly the loop no longer overshoots, but it spends bytes where small per-pixel terms sit, raises the reachable
floor and loses bytes against butteraugli everywhere. So neither the fit nor this change to the map's construction
makes our maps steer JPEG XL better than butteraugli; the fit is the one to keep, because it is what makes the
requested distance reachable. Screens are the open problem for every zensim map (5–26% more bytes at equal quality).
The split stays a research mode; the defaults are unchanged.

### JPEG XL steering: our maps against butteraugli and DVIFM 0.3 (2026-10-24)

Owner, 2026-10-24: "also remember to eval our maps against butter and dvifm 03 as jxl spatial steering candidates".
jxl-encoder's perceptual loop bumps the quantization of every 8×8 block whose tile distance (1.2 × the block's
16-norm of the steering map) exceeds the target; the map decides where bytes go. This compares maps as that steering
signal, by rate–distortion against judges.

**Design (recorded before any result).**

- *Runner.* `zensim-workbench/loop-converge` `steer_rd`: every arm in one binary, so one jxl-encoder build; effort 7
  and 4 iterations (the workbench loop player's defaults), default strategy (Zenjxl), distances 0.3, 0.45, 0.6, 1, 1.6,
  2.5, 4, 6.4 and 10; each bitstream decoded by zenjxl-decoder to dithered sRGB8 as the earlier harness decodes, kept
  by SHA-256. A failed map comparison stops the run (the loop itself would keep the previous iteration silently).
- *Arms.* `none` (no loop), `butteraugli` (the built-in loop), the workbench `diffmap` (σ 3) of B (`89cb4861`), Tga
  (`5ece9e45`) and M64a (`b8555871`) at their fitted placements (table above, "Workbench (second build)"), and
  `dvifm03`: DVIFM 0.3's `diffmap` (`aic4-dvifm-lap-v0.3`, author constants) at its fitted placement (scale 0.8846,
  exponent 0.2, held-out r 0.139), served by the private aic4 module through `loop_map_server` (zenmetrics-private,
  `aic4-wasm` example; the protocol is `MapServer`'s). Tga and M64a were trained with AIC2026 public_train labels and
  DVIFM's constants were fitted on AIC2026 by its author; none of the judges below was.
- *Corpus.* The 78 TRAIN images of the map-todo study (`jxl-maprd-large/corpus`: 38 CID22 training photos, 22 graphics
  and illustrations, 18 gb82-sc screen crops). Overlap: the loop placements were fitted on CID22-512 training images.
- *Judges.* SSIMULACRA2, butteraugli 3-norm, zensim and ColorVideoVDP (CPU, `cvvdp_cpu_imazen_v0_1_0`) from the
  pinned judge build (`zenmetrics-judge-7a0f8394`; its zensim is an August 2026 build), and DVIFM 0.3 from its frozen
  runtime (`dvifm_e21.py judge`). SSIMULACRA2 and CVVDP are independent of every arm. Home turf: butteraugli for the
  butteraugli arm, DVIFM for `dvifm03`, zensim (partly: a different model on shared features) for B, Tga and M64a.
- *Statistic.* `rd_probe_analyze_2026-07-18.py --paired-vs`: bytes saved at equal judged score against a base arm,
  per image the median over its matched points, then the mean over images with a 95% bootstrap interval over images;
  against `none` and against `butteraugli`, overall and per content class.
- *Reading (my bar).* An arm steers better than another when its saving is positive with the interval above zero
  under both independent judges; an interval that straddles zero under either is a tie. Home-turf cells are reported
  and not counted. Encode time per arm is reported.

**Amendment (before the encodes).** The reachability result added tail-fit placements (`loop_calibrate`
`LOOP_CAL_TAIL`, largest tile): Tga 2.2571e-2 / 0.7, M64a 2.4053e-2 / 0.7, B 1.8152e-2 / 0.7, and DVIFM 0.3
2.960654e-1 / 0.35 (the private aic4-wasm `loop_calibrate`, same method: fit r 0.328, held out 0.462; its bulk fit is
exponent 0.3, r 0.164). Each map runs at both placements. The no-loop baseline is `jxl_reach.sh`'s 20-distance sweep
(the same bitstreams as a 9-distance run, more points for the equal-quality interpolation).

**Results** (`jxl_steer_rd.sh`, steer_rd `3ba4f4ba1cb8`, DVIFM server `aic4-eb492421acd3`, judge build
`zenmetrics-judge-7a0f8394`, DVIFM runtime `4827663c…`; 8,580 judged images, run-heavy 901 s, peak RSS 1.88 GiB;
tables and SHA256SUMS in `/mnt/v/output/zensim/byconstruction-2026-10-22/jxl-steer/`). Bytes saved against the
butteraugli loop at equal judged score (%, mean over images of each image's median, 95% bootstrap interval; positive
is fewer bytes):

| Arm | Photos SSIMULACRA2 | Photos CVVDP | Illustrations SSIMULACRA2 / CVVDP | Screens SSIMULACRA2 / CVVDP | All SSIMULACRA2 | All CVVDP |
|---|---|---|---|---|---|---|
| Tga | **+2.61** [+1.72, +3.48] | +0.69 [−0.57, +1.84] | −1.37 / −2.45 | −7.05 / −12.68 | −0.74 [−2.05, +0.50] | −3.28 [−5.61, −1.23] |
| Tga, tail | +0.74 [+0.18, +1.23] | −0.38 [−0.89, +0.16] | −1.21 / −1.11 | −6.77 / −6.58 | −1.54 [−2.77, −0.54] | −2.01 [−3.14, −1.00] |
| M64a | **+2.56** [+1.48, +3.55] | +0.70 [−0.62, +1.95] | −0.26 / −1.55 | −7.07 / −11.76 | −0.46 [−2.08, +0.93] | −2.81 [−5.36, −0.72] |
| M64a, tail | +0.81 [+0.21, +1.33] | −0.26 [−0.89, +0.32] | −1.04 / −0.71 | −4.96 / −7.63 | −1.04 [−2.19, −0.08] | −2.09 [−3.57, −0.88] |
| B | +1.95 [+0.82, +3.02] | −0.82 [−2.54, +0.76] | −1.15 / −5.24 | −9.52 / −14.38 | −1.57 [−3.32, −0.04] | −5.19 [−8.04, −2.53] |
| B, tail | −0.15 [−1.08, +0.63] | −1.54 [−2.40, −0.70] | −0.96 / −1.53 | −10.92 / −11.33 | −2.86 [−4.78, −1.28] | −3.80 [−5.72, −2.14] |
| DVIFM 0.3 | −0.32 [−0.75, +0.07] | −1.35 [−1.77, −0.88] | +0.17 / −1.27 | −1.72 / −2.34 | −0.51 [−1.13, −0.00] | −1.56 [−2.33, −0.81] |
| DVIFM 0.3, tail | −1.40 [−1.90, −0.91] | −1.37 [−1.97, −0.84] | −0.68 / −1.46 | −1.06 / −0.89 | −1.12 [−1.61, −0.65] | −1.28 [−1.85, −0.77] |

Against no loop the butteraugli loop saves +1.20% [+0.78, +1.60] (SSIMULACRA2) and +1.12% [+0.54, +1.72] (CVVDP)
over all 78 images; Tga −0.87 / −4.04, Tga tail −0.84 / −2.64, DVIFM 0.3 −2.79 / −5.24, DVIFM tail −0.47 / −1.70.
Eight-iteration Tga tail is within 0.2 point of the four-iteration one on SSIMULACRA2 and CVVDP in every class. Home-turf cells (butteraugli on
its own loop, DVIFM on its own map, zensim on ours) are in the full tables and not read.

**Reading** (my bar: positive with the interval above zero under both SSIMULACRA2 and CVVDP): no map steers JPEG XL
better than butteraugli on this corpus. Our maps are ahead on photos by SSIMULACRA2 only (+2.6% at the shipped
placement; CVVDP +0.7% with an interval across zero) and behind on screens by 5–14% under both judges; DVIFM 0.3 stays
within 2.4% of butteraugli everywhere and is not ahead with an interval above zero anywhere. The tail placement, which makes the target
reachable, costs our maps about 1.9 points of SSIMULACRA2 savings on photos against the shipped placement (+0.74
against +2.61 for Tga) and gains on screens (CVVDP −6.6 against −12.7): once the loop believes it has met its target
it coarsens the blocks the map reads as free (the reclaim step, "What the map tells the loop" above), and the
independent judges charge for it. That is the same construction problem the reachability test found, seen in bytes.

### Gate review: what E21 parity selected for, against the product's goals (2026-10-25)

Owner, 2026-10-25: "the E2/E33 gates might be damaging our ability to get a good all around model? consider what has
changed because of older gates, vs our current smooth-dial, codec-uniform robust and spatial steering goals." (A
session goal still carried the dropped AIC2026 0.99 target; the last steps it drove, ensembles and capacity for E21,
are recorded above and below.)

**What the screen measures.** E21 is each held-out dataset's SROCC pooled over all its distortion types, against
`base-c`; E33's point form asks for parity (mean ≥ −0.002, every set ≥ −0.005), and only an arm at parity is extended
to the product gates (dial C1–C6, NEARID, identity, G-STEER; "Decision rules" above). Three of the four sets are
synthetic (KADID 25 types, TID2013 24, KonFiG 7), so the statistic is dominated by placing unlike distortions (contrast
change, saturation, mean shift, colour blocks, noise, codecs) on one human scale.

**Where the monotone models stand once that is separated** (seeds 0–2, out of sample, no label read that was not read
before; `byc_cells.py bytype`, `type_bias.py` extended to every set, `aiczero` with per-codec bias):

| | `base-c` (B's recipe) | M64 | M (32 pairs) | T (convex) |
|---|---|---|---|---|
| E21 pooled Δ (the screen) | 0 | −0.0131 | −0.0144 | −0.0205 |
| Within-reference ordering per type, Δ: KADID / TID2013 / KonFiG / CID22-A | 0 | **+0.013 / +0.005 / +0.004** / 0.000 | **+0.014 / +0.006 / +0.004** / 0.000 | **+0.015 / +0.008 / +0.004** / 0.000 |
| Per-type bias spread across types (SD of rank bias): KADID / TID2013 / KonFiG | 0.060 / 0.054 / 0.079 | 0.083 / 0.078 / 0.088 | 0.082 / 0.082 / 0.085 | 0.084 / 0.099 / 0.090 |
| CID22-A codec bias spread | 0.0340 | 0.038 | **0.0345** | 0.038 |
| Standard mixture inversions per cell (largest rise) | 86–171 (+61.5) | 1–6 (+7.6) | 1–32 (+6.5) | **0–4 (+4.0)** |
| Global-edit inversions per cell | (frozen B: 88 in all, +18.3) | 0 | 0 | 0 (Thpm) |
| Full-data product gates (C1 standard / ladder, NEARID, G-STEER) | 0.9954 / 0.9882, 130/144, 130/135 | **0.9984 / 0.9937, 142/144, 135/135** (2026-10-25) | **0.9984 / 0.9935, 143/144, 135/135** | 0.9970 / 0.9914, 140/144, 135/135 |

Reported only, not for selection (AIC2026 labels must not tune models): AIC2026 public_train zero-shot global SROCC
0.9072 / 0.9080 / 0.9091 / 0.9108, within source 0.9530 / 0.9492 / 0.9498 / 0.9496, per-codec bias spread (mean over
cells) 0.061 / 0.074 / 0.071 / 0.075; every model orders each codec ladder at 0.9996–0.9998; the spread is mostly PNG (palette
quantization), which every model overrates and the monotone ones more (+0.18 to +0.20 against +0.13).

**Reading.** The monotone models order severity within every distortion type better than B's recipe, match it on
codec data and are far more robust; their whole E21 deficit is cross-type calibration, mainly global and colour edits
(rated against people: saturation +0.21 to +0.34 and mean shift +0.12 in TID2013, contrast −0.12 and Gaussian blur
+0.10 in KADID). E21 parity therefore selected for exactly the capacity that makes B invert (sign contrasts between
correlated damage features), and it stopped every monotone arm before the product gates: M64 has never been measured
on them; T and M were, and pass every one with better dial and steering numbers than `base-c`. The full-data monotone
fits keep one or two live units, so capacity, ensembles and new features barely move them, and none of that work
could reach parity.

What the old gates changed in this record:
1. *Screen and extension*: no arm extended since T and M; the product gates were never run on M64 or its relatives.
2. *Feature adoption* ("E21 improves by more than one seed-unit SE"): linear-light coarse scales and orientation bands
   were judged by E21 alone (linear light was neutral there, better on KADID and CID22-A, half the largest mixture rise).
3. *Recipe choice*: M64 over M and T by E21 (−0.0131 against −0.0144 / −0.0205), though T has the fewest mixture
   inversions and M the best dial, NEARID and codec spread.
4. *The candidate rule and the AIC leg* (mine, 04743622; the dropped 0.99 goal): selection by AIC2026 public_train out
   of fold, which tuned models on AIC2026 labels, and Tga, a partly monotone model with 4–47 global-edit inversions
   per cell, as the session's best. Revision 6's `globaff` family exists to serve it.

**Proposed gates for the current goals** (for the owner; nothing changes until approved): (1) certified monotone with
the identity pin; (2) robustness: no global-edit inversion, at most a few standard-mixture inversions per cell; (3)
dial: within-reference ordering on every set no worse than `base-c`, the full-data dial gates C1–C6, G-DIAL and NEARID;
(4) codec uniformity and smoothness on CID22-A (pooled SROCC, per-codec bias spread, Z-RMSE), no AIC2026 label used;
(5) steering: G-STEER, map agreement with independent judges, JPEG XL steering against butteraugli. E21's pooled
statistic stays as a reported diagnostic of cross-type calibration.

**M64's full-data run** (2026-10-25, after the gates were adopted; `scripts/byconstruction/full_fit_gates.sh
mono-thr-mp64_s0 mono:thr:mp64 m64full`, trainer `774955ad` at one thread as M's and T's ran, run-heavy 3,013 s, peak
RSS 0.96 GiB): certified monotone (one live max pair), packed f16 bake `0ff53fdc…` (22,560 B). Dial C1 0.9984 standard
and 0.9937 ladder with C2–C6 passing on both grids, G-DIAL (p5 13.3 / −9.2, p95 97.5 / 95.1), NEARID N1–N3 with 142 of 144 monotone ladders, G-STEER 135/135,
identity at every SIMD tier (10 permutations, 62 sources each). It is level with M on the product gates; M keeps the
better CID22-A codec spread (0.0345 against 0.038), M64 the fewer standard-mixture inversions per cell (1–6 against
1–32).

**The two features E21 alone judged, on this scorecard** (seeds 0–2, out of sample, the same cells; `byc_cells.py
bytype`, `type_bias.py`):

| | M64 | M64lin (coarse scales from linear RGB) | M64o (orientation bands) |
|---|---|---|---|
| Within-reference ordering Δ vs `base-c` (KADID / TID2013 / KonFiG / CID22-A) | +0.0132 / +0.0054 / +0.0037 / 0.0000 | +0.0134 / +0.0052 / +0.0037 / −0.0002 | +0.0142 / +0.0057 / +0.0037 / 0.0000 |
| Per-type bias spread (KADID / TID2013 / KonFiG) | 0.083 / 0.078 / 0.088 | **0.069** / 0.085 / 0.092 | 0.082 / 0.093 / 0.093 |
| CID22-A codec bias spread | 0.0382 | 0.0387 | 0.0370 |
| Mixture inversions per cell (largest rise) | 1–6 (+7.6) | 2–5 (**+3.9**) | 2–18 (+7.6) |

Neither feature changes within-reference ordering by more than 0.001 (no seed-level uncertainty is computed for these
deltas, so read them as unchanged). Linear light halves the largest mixture rise and calibrates KADID's types better,
at a cost on TID2013's; orientation bands add mixture inversions. Under the proposed gates neither is a clear
adoption either; linear light stays the better-behaved of the two.

**Smoothness of the candidates** (the "Smoothness" section's statistics, out of sample, seeds 0–2; `aic_panel.py --geo
--no-aic`, a new switch that reads no AIC2026 label; `base-c` reproduces that section's row exactly; JSON
`aic2026train/panel_smooth_monotone_e21only.json`). Z-RMSE / error p99:

| Model | KADID | TID2013 | KonFiG | CID22-A | Mean Z-RMSE | Mean error p99 | Mean rank out4 |
|---|---|---|---|---|---|---|---|
| `base-c` | **0.454** / **0.372** | **0.423** / **0.270** | **0.567** / **0.555** | 0.488 / 0.377 | **0.483** | **0.394** | **1.05%** |
| M64 | 0.487 / 0.392 | 0.441 / 0.272 | 0.591 / 0.593 | 0.489 / 0.376 | 0.502 | 0.408 | 1.12% |
| M | 0.483 / 0.387 | 0.442 / 0.276 | 0.594 / 0.591 | 0.489 / 0.377 | 0.502 | 0.408 | 1.17% |
| T | 0.492 / 0.391 | 0.460 / 0.291 | 0.607 / 0.604 | **0.481** / **0.369** | 0.510 | 0.414 | 1.30% |

On codec data (CID22-A) the monotone models are as smooth as B's recipe (T slightly smoother). On the synthetic sets
they are 0.02–0.04 worse in Z-RMSE, with KonFiG's error tail the largest gap (0.59–0.60 against 0.555): one logistic
curve fitted across every distortion type absorbs the per-type offsets of the table above, so this is the same
cross-type calibration seen as error rather than rank. Nothing in training asks for it (the human legs train on
rank alone, "Smoothness" above).

**Compute of the candidates.** Neither full-data bake has an exactly dead input (`zenpredict-inspect --weights` on
the dense bakes): M keeps two live units (both output weights −0.5174) and T one, every one of the 820 derived inputs
has a nonzero weight into a live unit, and all 420 raw feature ids (`basic+peaks+v2@w1825/rev5_localwin#36c3f3af`,
revision 5) are read. So M and T cost what B costs to extract; nothing can be dropped without a refit. The weight is
concentrated (with each input's weight divided by its scaler scale, 8 / 114 / 474 of M's inputs carry 50 / 90 / 99% of
the total, T's 6 / 89 / 420), which is where a reduced input set would start.

**Bake sizes.** Every bake here is ZNPR v3 with its payload LZ4-compressed (flags 0x0003). The production packs used
`--dtype f16 --zerobias-bulk 0 --protect-last`: zerobias zeroed nothing (L0 0 of 104,960, L1 0 of 128; at the dense
default τ 0.005 it would have killed 17 of M's and 44 of T's 820 live input lines outright) and prune found no dead
input column. The 126 / 127 dead hidden units are stored as zeros, so the payloads are as large as B's, and LZ4
removes them:

| Bake | Payload, uncompressed | On disk | Without the `zentrain.repro` record | Also without `zentrain.sample_coverage` |
|---|---|---|---|---|
| B (`89cb4861`, f16) | 254,512 B | 214,904 B | 210,617 B | — |
| M dense (f32) | 464,174 B | 26,259 B | — | — |
| M packed (f16, `6291eeb5`) | 254,466 B | **22,797 B** | 18,615 B | 15,620 B |
| T dense (f32) | 464,140 B | 23,057 B | — | — |
| T packed (f16, `8c40f24c`) | 254,432 B | **21,215 B** | 17,045 B | 14,050 B |

So M and T ship at a tenth of B's size; about a third of what is left is provenance metadata (36.6 KB of text before
compression: the repro record 19.9 KB, the derived-input declaration 9.3 KB, the coverage record 5.5 KB), which
serving needs only for the declaration. The stripped files (`bake_dial_refit strip`) were measured only; the
production bakes keep every record.

### JPEG XL steering with the full-data monotone models (2026-10-25)

The steering study's zensim maps were B and two models trained with AIC2026 labels (Tga, M64a). The models the
proposed gates favour, the full-data certified-monotone fits M (`mono-thr-mp32_s0`, packed `6291eeb5…`) and T
(`mono-thr_s0`, packed `8c40f24c…`), have never steered the loop. This measures them; it reads no human label.

**Design (recorded before any result).**
- *Build.* One new steer_rd and loop_calibrate (`just byc-loop-bins`) embedding `zensim-Mfull@rev5`,
  `zensim-Tfull@rev5` and `zensim-B@rev5` (`89cb4861…`). Controls: the new build's B arm (`Bcheck`) must reproduce
  every one of the study's 702 B bitstreams byte for byte, and loop_calibrate's B fits must reproduce B's placements
  (bulk 1.302887e-1 / 0.6 on the 2026-10-21 pairs, tail 1.8152e-2 / 0.7 on the 2026-10-24-hq pairs). A failed control
  stops the run before any M or T result is read.
- *Placements.* Each map's `diffmap` (σ 3) at a bulk placement (loop_calibrate's shipped method on the 2026-10-21
  calibration pairs, as B's, Tga's and M64a's shipped placements were fitted) and at a tail placement
  (`LOOP_CAL_TAIL`, each pair's largest tile, on the 2026-10-24-hq pairs, as the tail arms). Arms `Mfull`, `Tfull`,
  `Mfulltail`, `Tfulltail`.
- *Everything else as the steering study:* the 78 TRAIN images, effort 7, 4 iterations, distances 0.3–10, the same
  judge build and DVIFM runtime, `--paired-vs` against `none`, `butteraugli` and, new, `B`. The earlier judge tables
  move to `judged-<time>/` and the panel re-judges every arm, so every earlier arm's numbers must come out again.
- *Reading (my bar, unchanged):* an arm steers better than another when its saving is positive with the interval
  above zero under both SSIMULACRA2 and CVVDP; an interval across zero under either is a tie. Against B this answers
  whether a monotone model's map is a better steering signal than B's at the same kind of placement.

**Placements** (`5d15916f107f-loop-mt` build, `loop_calibration_2026-10-25mt.tsv` and `…mttail.tail.tsv` in
`/mnt/v/zenmetrics-private/zensimx-workbench/`; 187 s, run-heavy peak RSS 0.37 GiB). Control passed: B's bulk fit
(1.302887e-1 / 0.6, fit r 0.7482 over 476,229 blocks), its tail fit (1.815210e-2 / 0.7, r 0.8074 / 0.7779) and the
tail run's bulk row reproduce the recorded rows digit for digit.

| Map | Bulk scale / exponent | Fit r / held-out r | Tail (largest) scale / exponent | Fit r / held-out r |
|---|---|---|---|---|
| M (`6291eeb5`) | 1.319342e-1 / 0.6 | 0.811 / **0.638** | 2.629540e-2 / 0.7 | 0.848 / 0.683 |
| T (`8c40f24c`) | 1.358903e-1 / 0.6 | 0.814 / 0.622 | 2.510790e-2 / 0.7 | 0.842 / 0.675 |
| B (`89cb4861`) | 1.302887e-1 / 0.6 | 0.748 / 0.491 | 1.815210e-2 / 0.7 | 0.807 / **0.778** |

Across all blocks the monotone maps agree with butteraugli's block distances better than B's (held-out r 0.638 /
0.622 against 0.491, as Tga's 0.653 and M64a's 0.681 did); on each pair's largest tile, which decides reach, B's
agrees better on the held-out screenshots (0.778 against 0.683 / 0.675).

**Results** (`jxl_steer_rd.sh` with `ARMS="Bcheck Mfull Tfull Mfulltail Tfulltail"`, run-heavy 1,462 s, peak RSS
1.02 GiB; tables, placements, logs and SHA256SUMS in `/mnt/v/output/zensim/byconstruction-2026-10-22/jxl-steer-mt/`).
Both controls passed: `Bcheck` reproduced all 702 of B's bitstreams, and the re-judged panel reproduced every earlier
value (10,686 rows under each of the six judge tables). Bytes saved at equal judged score (%, mean over images of each
image's median, 95% bootstrap interval; positive is fewer bytes):

| Arm | vs | All SSIMULACRA2 | All CVVDP | Photos SSIMULACRA2 / CVVDP | Illustrations SSIMULACRA2 / CVVDP | Screens SSIMULACRA2 / CVVDP |
|---|---|---|---|---|---|---|
| M | B | **+1.78** [+1.24, +2.36] | **+2.25** [+1.59, +2.89] | **+1.64 / +2.27** | **+1.34 / +1.73** | **+2.61 / +2.83** |
| T | B | **+1.60** [+0.95, +2.37] | **+1.80** [+1.00, +2.58] | **+1.59 / +2.11** | +0.61 [−0.22, +1.36] / +1.81 | +2.84 / +1.12 [−1.67, +3.61] |
| M, tail | B, tail | **+1.93** [+1.29, +2.82] | **+2.58** [+2.01, +3.33] | **+1.44 / +2.77** | **+0.70 / +1.49** | **+4.48 / +3.51** |
| T, tail | B, tail | **+1.77** [+1.16, +2.53] | **+2.27** [+1.70, +2.94] | **+1.36 / +2.43** | **+0.67 / +1.38** | **+3.95 / +3.06** |
| M | butteraugli | −0.54 [−2.01, +0.69] | −3.34 [−5.97, −1.11] | **+2.57** / +0.76 [−0.27, +1.79] | −0.65 / −2.62 | −6.98 / −12.88 |
| T | butteraugli | −0.39 [−1.95, +0.91] | −2.91 [−5.16, −0.99] | **+2.76** / +0.51 [−0.68, +1.65] | −0.60 / −2.11 | −6.77 / −11.10 |
| M, tail | butteraugli | −1.41 [−2.64, −0.37] | −2.05 [−3.44, −0.84] | +0.88 / +0.15 [−0.41, +0.72] | −1.34 / −1.16 | −6.32 / −7.77 |
| T, tail | butteraugli | −1.51 [−2.79, −0.40] | −2.14 [−3.59, −0.90] | +0.71 / −0.09 [−0.64, +0.49] | −1.26 / −1.20 | −6.49 / −7.63 |

Bold: positive with the interval above zero. Against butteraugli the earlier arms read, over all images, Tga −0.74 /
−3.28 and M64a −0.46 / −2.81 (SSIMULACRA2 / CVVDP), B −1.57 / −5.19. Within this run the maps cost B's encode time
(median 403–410 ms against `Bcheck`'s 402; times from different runs are not comparable). Encodes that end with no
block above target: M 165 and T 152 of 702 at the bulk placement (B 116), 548 / 549 at the tail placement (B 531,
butteraugli 566).

**Reading** (my bar). The full-data monotone models' maps steer JPEG XL better than B's: at either kind of
placement, in every content class and under both independent judges, except T's bulk placement on illustrations
(SSIMULACRA2) and screens (CVVDP), which tie. They steer as well as the AIC2026-trained Tga and M64a, so dropping
those models loses nothing here. None of them steers better than butteraugli: M and T save 2.6–2.8% on photos by
SSIMULACRA2 with CVVDP tied, and screens still cost 6–13%.

### A luminance-only gate (2026-10-25)

Owner, 2026-10-25: "I thought we added some global features, why still 820?" The full-data M and T read B's 820
inputs because they were fitted before the `byc_globaff` descriptor existed. The descriptor was tried three ways in E21
cells only: as 24 sign-free inputs (Tghpm, 844 inputs; E21 −0.0101, but 5–72 global-edit inversions per cell), and as
410 gated copies of the damage inputs with every weight non-negative (Tgg and Mgg64, 1,230 inputs; no global-edit
inversion; E21 −0.0194 and −0.0161). The only full-data fit with it is Tga, trained on AIC2026 labels.

**The all-channel gate on the goal-aligned scorecard** (`byc_cells.py bytype`, `type_bias.py`; seeds 0–2, out of
sample). Rank bias per type (prediction rank − human rank; negative: the model rates the edit worse than people do):

| Type | `base-c` | M64 | Mgg64 | Thpm | Tgg | Tghpm |
|---|---|---|---|---|---|---|
| KADID contrast change | −0.023 | −0.121 | −0.076 | −0.127 | −0.097 | −0.073 |
| KADID brighten | −0.075 | −0.091 | −0.055 | −0.090 | −0.066 | −0.039 |
| KADID darken | −0.075 | −0.092 | −0.063 | −0.093 | −0.077 | −0.063 |
| TID2013 contrast change | −0.072 | −0.064 | −0.001 | −0.073 | +0.010 | −0.109 |
| TID2013 mean shift | −0.025 | +0.117 | +0.089 | +0.118 | +0.109 | +0.100 |
| TID2013 saturation | +0.025 | +0.212 | **+0.277** | +0.352 | +0.345 | +0.222 |

| | M64 | Mgg64 | Thpm | Tgg | Tghpm |
|---|---|---|---|---|---|
| Within-reference ordering Δ vs `base-c` (KADID / TID2013 / KonFiG / CID22-A) | +0.0132 / +0.0054 / +0.0037 / 0.0000 | +0.0123 / +0.0103 / +0.0036 / 0.0000 | +0.0143 / +0.0089 / +0.0037 / 0.0000 | +0.0130 / +0.0096 / +0.0036 / 0.0000 | +0.0109 / +0.0084 / +0.0036 / +0.0001 |
| Per-type bias spread (KADID / TID2013 / KonFiG) | 0.083 / 0.078 / 0.088 | 0.077 / 0.083 / 0.105 | 0.083 / 0.101 / 0.093 | 0.079 / 0.099 / 0.097 | 0.078 / 0.081 / 0.105 |
| CID22-A codec bias spread | 0.0382 | 0.0404 | 0.0380 | 0.0363 | 0.0409 |

The gate removes much of the luminance edits' over-penalty (KADID contrast −0.121 → −0.076, TID2013 contrast −0.064 →
−0.001), but it discounts every affine change, and the monotone models already under-penalize saturation (TID2013
+0.21 to +0.35 against `base-c`'s +0.025); the M-class gate pushes it further (+0.277). The descriptor's shares say
why: on KADID the median share (mean over the four scales) is X / Y / B 1.000 / 0.687 / 1.000 for color saturation 1,
0.326 / 0.942 / 0.502 for brighten, 0.484 / 0.862 / 0.599 for contrast change and 0.629 / 0.981 / 0.804 for mean shift,
so the saturation change is explained in every channel and the all-channel gate forgives it most.

**Design (recorded before any result).**
- *Inputs* `fx1gy` (`ring_root.py gated --channels Y`, declaration `byc_fx1gy_declaration.json`, SHA-256
  `d850c8a8…`, written by the new `gated-decl`, which reproduces `byc_fx1gg_declaration.json` byte for byte with
  `--channels XYB`): fx1's 410 differences and 410 fragility products plus 164 gated copies, `d · (1 − share)`, of the
  differences in the four luminance cells only (f1891–f2054; root `ringhpm/root_globaff_gatedy`, era
  `rev5_localwin_ringhpm_globaffgatey`, `basic+peaks+v2@w2055/rev5_localwin_ringhpm_globaffgatey#7b645e91`); 984
  inputs, every weight non-negative, the descriptor itself not read. Chroma damage is never discounted. Rebuilding the
  all-channel root with the changed builder reproduces all 63 of its files byte for byte; the all-channel mixture table
  reproduces bit for bit (its hand-written `table.json` differed only by a trailing newline).
- *Arms* (no AIC leg; the same 12 E21 cells, seeds 0–2; trainer `70eba638`, as Tgg and Mgg64): **Tgy**
  (`mono:thr`) against Tgg and Thpm, and **Mgy64** (`mono:thr:mp64`) against Mgg64. Mgy64 against M64 mixes the gate
  with the ringing form (M64 reads raw ringing; no mp64 arm reads `hpm` ringing without the descriptor).
- *Statistics:* the tables above (per-type rank bias, per-type and codec bias spread, within-reference ordering),
  smoothness (`aic_panel.py --geo --no-aic`), the standard and global-edit mixture tests (`aic_mixture.sh` on the
  luminance-gated mixture tables), and E21's pooled Δ as a diagnostic. No AIC2026 label is read.
- *Reading (my bars).* The luminance gate does what it is built for when, in each class, TID2013 saturation's bias is
  no larger than the ungated arm's (Thpm +0.352, M64 +0.212) while KADID contrast change keeps at least half of the
  all-channel gate's improvement (its bias at least −0.112 in the T class, −0.099 in the M class). It becomes a candidate under the proposed
  gates if, in addition, no global-edit step inverts, standard-mixture inversions stay within the ungated arm's range,
  within-reference ordering is no worse than `base-c` on every set, and CID22-A's codec spread is no worse than the
  ungated arm's (T 0.0380, M 0.0382).

*Amendment (before any result).* The first launch was refused by strict admission before any cell trained: the new
root's feature set was not in the admitted list. It is now registered in the trainer's compiled list
(`group_admission.rs`), the Python admission, the feature-set registry and zensim's layout widths (the
`every_registered_layout_width_is_a_candidate` gate failed until w2055 joined; `feature_set_match` 10/10 after). So
the cells run on trainer `774955ad` (`just byc-train-bins`), not `70eba638`; no trainer source changed in between. A
reproduction cell (Tgg's KADID-held-out seed-0 cell retrained in the same run) must give predictions bit-identical to
the stored Tgg cell, or the comparison with Tgg and Mgg64 is not made.

**Result, T class** (`run_gy.sh`; 12 cells 01:30–01:46 UTC, trainer `774955ad`). Control passed: the reproduction
cell's predictions are bit-identical to the stored Tgg cell's (`3e156ccd…` both). Seeds 0–2, out of sample:

| | Thpm (no gate) | Tgg (all channels) | **Tgy (luminance)** |
|---|---|---|---|
| KADID contrast change / brighten / darken bias | −0.127 / −0.090 / −0.093 | −0.097 / −0.066 / −0.077 | −0.099 / −0.066 / −0.078 |
| TID2013 saturation / mean shift / contrast bias | +0.352 / +0.118 / −0.073 | +0.345 / +0.109 / +0.010 | +0.344 / +0.108 / +0.011 |
| Within-reference ordering Δ vs `base-c` (KADID / TID2013 / KonFiG / CID22-A) | +0.0143 / +0.0089 / +0.0037 / 0.0000 | +0.0130 / +0.0096 / +0.0036 / 0.0000 | +0.0131 / +0.0104 / +0.0036 / 0.0000 |
| Per-type bias spread (KADID / TID2013 / KonFiG) | 0.083 / 0.101 / 0.093 | 0.079 / 0.099 / 0.097 | 0.080 / 0.099 / 0.096 |
| CID22-A codec bias spread | 0.0380 | 0.0363 | 0.0365 |
| Standard mixture inversions per cell (all, largest rise) | 0–4 (29, +4.19) | 1–9 (73, +7.85) | 1–11 (75, +8.00) |
| Global-edit mixture inversions | 0 | 0 | 0 |
| Mean Z-RMSE (CID22-A Z-RMSE / error p99) | 0.512 (0.482 / 0.368) | 0.511 (0.486 / 0.366) | 0.510 (0.485 / 0.370) |
| E21 Δ, diagnostic (KADID / TID2013 / KonFiG / CID22-A) | −0.0221 | −0.0194 | −0.0188 (−0.011 / −0.043 / −0.027 / +0.006) |

Against the bars: TID2013 saturation +0.344 is no larger than Thpm's +0.352 and KADID contrast change −0.099 clears
−0.112, so the luminance gate does what it is built for. It is not a candidate: standard-mixture inversions rise to
1–11 per cell, outside Thpm's 0–4 (every other candidate bar passes). In this class the luminance gate and the
all-channel gate agree to within 0.002 on every statistic; the T-class all-channel gate never worsened saturation
(+0.345 against +0.352), so the class where the two gates should differ is M (Mgg64 +0.277 against M64's +0.212),
running now.

**Result, M class** (Mgy64, 12 cells 01:49–02:57 UTC; run-heavy for both arms 5,220 s, peak RSS 1.32 GiB). Seeds 0–2,
out of sample. Mhpm (max pairs mp32 on the `hpm` ringing root, no descriptor) is added because the gated roots read the
`hpm` ringing form and M64 does not:

| | M64 (raw ringing) | Mhpm (mp32, `hpm`, no gate) | Mgg64 (all channels) | **Mgy64 (luminance)** |
|---|---|---|---|---|
| KADID contrast change / brighten / darken bias | −0.121 / −0.091 / −0.092 | −0.119 / — / — | −0.076 / −0.055 / −0.063 | −0.084 / −0.053 / −0.062 |
| TID2013 saturation / mean shift / contrast bias | +0.212 / +0.117 / −0.064 | +0.282 / — / −0.068 | +0.277 / +0.089 / −0.001 | **+0.309** / +0.110 / −0.023 |
| Within-reference ordering Δ vs `base-c` (KADID / TID2013 / KonFiG / CID22-A) | +0.0132 / +0.0054 / +0.0037 / 0.0000 | — | +0.0123 / +0.0103 / +0.0036 / 0.0000 | +0.0130 / +0.0123 / +0.0036 / +0.0001 |
| Per-type bias spread (KADID / TID2013 / KonFiG) | 0.083 / 0.078 / 0.088 | 0.084 / 0.086 / 0.087 | 0.077 / 0.083 / 0.105 | 0.080 / 0.089 / 0.101 |
| CID22-A codec bias spread | 0.0382 | 0.0383 | 0.0404 | 0.0413 |
| Knob-concurring inversions (KADID / TID2013), ladder error (KADID / CID22-A) | 0.0141 / 0.0095, 0.0894 / 0.0701 | 0.0140 / 0.0084, 0.0896 / 0.0700 | 0.0132 / 0.0089, 0.0857 / 0.0707 | 0.0134 / **0.0080**, 0.0864 / 0.0706 |
| Standard mixture inversions per cell (all, largest rise) | 1–6 (38, +7.56) | 2–35 (+8.0) | 2–36 (88, +7.55) | 2–33 (79, +7.83) |
| Global-edit mixture inversions | 0 | 0 | 0 | 0 |
| Mean Z-RMSE (CID22-A Z-RMSE / error p99) | 0.502 (0.489 / 0.376) | 0.504 (0.488 / 0.375) | 0.507 (0.497 / 0.376) | 0.511 (0.497 / 0.378) |
| E21 Δ, diagnostic | −0.0131 | −0.0158 | −0.0161 | −0.0181 |

Against the bars: the luminance gate keeps the contrast correction (KADID contrast change −0.084 clears −0.099) but
TID2013 saturation rises to +0.309, above M64's +0.212, so it fails the bar it was built for. The ringing form
explains part of the gap to M64 (Mhpm alone +0.282 against M's +0.246 on the raw root, both mp32), and Mgy64 is above
that too: gating only luminance does not spare saturation. It also fails as a candidate on standard-mixture
inversions (2–33 per cell against 1–6) and codec spread (0.0413 against 0.0382). It has the fewest knob-concurring
TID2013 inversions of any arm (0.0080).

**Reading.** The global descriptor, as sign-free inputs or as either gate, does not give a candidate under the adopted
gates: every form trades robustness (standard-mixture inversions) and, in the M class, codec uniformity for its
correction of luminance edits, and none fixes the saturation under-penalty. That under-penalty is a property of the
monotone combination of these damage features (B's recipe gets saturation right through sign contrasts), not of a
missing global term. Full-data M, which reads no descriptor, stays the leading candidate.

### Margarine as the steering map (2026-10-25)

Owner, 2026-10-25: "Also add margaraine to the spatial steer and see if it can approach butter better." Margarine is
zenmetrics' unpublished Butteraugli approximation (`crates/margarine`, candidate `simd-row-malta`; its README notes that
spatial encoder steering had not been evaluated end to end). It emits a native-size diffmap on Butteraugli's scale, so
it steers at its own placement (scale 1, exponent 1, nothing fitted): the loop treats its map as it treats Butteraugli's.
`margarine_map_server.py` serves it to steer_rd (one server per loop worker, one Margarine thread per call); its
`--check` mode passed before the run: on a 512×512 CID22 training pair the served map is bit-identical to the CLI's and
the score equal (1.0024 for a no-loop d = 1.0 encode). Binary `22a3ff33…`, built from zenmetrics `86b48522`.

**Results** (same 78 TRAIN images, judges and statistic as the steering study; run-heavy 1,436 s, peak RSS 0.60 GiB;
re-judging reproduced all 14,196 earlier values in every judge table; tables and SHA256SUMS in
`/mnt/v/output/zensim/byconstruction-2026-10-22/jxl-steer-margarine/`). Bytes saved against the butteraugli loop at
equal judged score (%, 95% interval):

| Map | All SSIMULACRA2 | All CVVDP | Photos SSIMULACRA2 / CVVDP | Illustrations SSIMULACRA2 / CVVDP | Screens SSIMULACRA2 / CVVDP |
|---|---|---|---|---|---|
| **Margarine** | −1.33 [−2.26, −0.57] | **−1.21** [−2.13, −0.45] | **−0.18 [−0.65, +0.14] / −0.15 [−0.59, +0.11]** | −0.99 / −0.80 | −4.20 / −3.95 |
| DVIFM 0.3 | **−0.51** [−1.10, −0.00] | −1.56 [−2.37, −0.82] | −0.32 [−0.76, +0.05] / −1.35 [−1.77, −0.90] | +0.17 / −1.27 | **−1.72 / −2.34** |
| full-data M | −0.54 [−2.01, +0.69] | −3.34 [−5.97, −1.11] | **+2.57** / +0.76 [−0.28, +1.84] | −0.65 / −2.62 | −6.98 / −12.88 |
| B | −1.57 [−3.32, −0.04] | −5.19 [−8.04, −2.53] | +1.95 / −0.82 | −1.15 / −5.24 | −9.52 / −14.38 |

Against no loop, Margarine's loop saves −0.12% [−1.27, +0.85] (SSIMULACRA2) and −0.36% [−1.38, +0.57] (CVVDP) over all
images (butteraugli's +1.20 / +1.12), +1.19 / +0.74 on photos and −2.98 / −3.08 on screens. 533 of its 702 encodes end
with no block above target (butteraugli's loop 566).

**Reading** (my bar). Margarine does approach Butteraugli: it is the only map whose photos are a tie with the
butteraugli loop under both independent judges, and the closest overall by CVVDP (−1.2%). It is not better than
Butteraugli anywhere, and it is not ahead of the other maps everywhere: DVIFM 0.3 is closer overall by SSIMULACRA2 and on
screens, and full-data M saves more on photos by SSIMULACRA2 (+2.6%, CVVDP tied). On screens it loses 4% against our maps'
7–15%.

**Where its remaining gap comes from** (owner: "Are there any obvious cheap improvements margarine could make?").
Margarine's map against Butteraugli's block distances on the loop calibration encodes, by `loop_calibrate`'s method
(`scripts/byconstruction/margarine_calibrate.py`; 2026-10-24-hq pairs: 17 CID22-512 training photos fitted, 10 gb82 screenshots held out):

| Map (native placement: scale 1, exponent 1) | Worst block, fit / held-out r | Worst-block log bias (held out) | All blocks, fit / held-out r | All-block log bias, photos / screenshots |
|---|---|---|---|---|
| Margarine, selected (`simd-row-malta`) | 0.997 / **0.990** | +0.017 | 0.675 / 0.561 | −0.16 / **−0.85** |
| Margarine, `native-uhf` control | 0.997 / 0.991 | +0.041 | 0.674 / 0.562 | −0.14 / −0.85 |
| Margarine's teacher (frozen Butteraugli `13c49cbc`, `margarine-score teacher`) | 0.998 / 0.991 | +0.038 | 0.669 / 0.557 | −0.16 / −0.93 |

- No placement is needed: on each pair's worst block, which decides whether the loop reaches its target, the native map
  agrees with Butteraugli at r 0.99 with almost no bias; a fitted placement (1.1 / 0.97) does no better held out. Our maps
  reach 0.68–0.79 there.
- On the bulk of blocks Margarine reads lower than the loop's Butteraugli on screenshots (log −0.85, about 0.43×), so the
  loop treats those blocks as nearly free and coarsens them: the screens' 4%.
- Computing the ultra-high-frequency Malta banks natively changes nothing (built from a copy of the crate outside
  zenmetrics with the CLI's candidate check widened; maps differ in 116k–197k of 262k pixels, means within 0.3%).
- Margarine's own teacher shows the same bulk bias (−0.93): Margarine is faithful to its teacher, and the gap lies
  between that frozen Butteraugli (`13c49cbc`) and the one jxl-encoder's loop and the calibration use (0.9.4 at
  `d2466a4`), with the same default parameters. Which change moved low-error blocks is not resolved here: the local
  Butteraugli checkout does not hold `13c49cbc`'s line.

**Correction (2026-10-25, later): the gap is the 8-bit rounding, not the teacher's revision.** The Margarine lane
(zenmetrics `77fc41d4`, `a3237b6a`, `31856c6c`; `crates/margarine/STEERING_REFERENCE.md`) pinned the three Butteraugli
sources (the encoder loop's `d2466a4`, the teacher `13c49cbc`, current `554a42f8`) and found their maps bit-identical
on identical RGBA8 pixels of all 324 calibration pairs, so the revision explains nothing. The cause is the inputs: the
loop's block targets (`butter.f32`) come from Butteraugli on the encoder's floating-point reconstruction, while the
saved distorted image, and every external map in this study (Margarine through its server, our zensim maps through
the workbench adapter), sees that reconstruction rounded and clamped to sRGB8. Checked here on the saved pairs
(numpy, the lane's replayed float reconstructions, which reproduce the saved RGBA8 byte for byte): among blocks whose
target is at least 0.05, **44.8% of the screenshots' (150,189 of 335,301) and 6.7% of the photos' (53,664 of 805,248)
are pixel-identical to the reference after rounding**, with a median largest float error of 2.1e-4 (screenshots) and
7.7e-4 (photos) in linear light, under half an 8-bit step. Any 8-bit map reads those blocks as undamaged, so the loop
coarsens them: the screens' loss. Fed the unrounded floats (`margarine-steer`, the same kernels), Margarine's
screenshot bulk log bias goes from −0.85 to +0.02 and its worst-block held-out r to 0.9992, with no fitted placement.
The earlier statement that "the gap lies between that frozen Butteraugli and the one jxl-encoder's loop uses" is
wrong. Whether the floats also steer better, for Margarine and for our maps, is measured next (`steer_rd --linear`).

### JPEG XL steering on the loop's float reconstruction (2026-10-25)

**Design (recorded before any result).** The correction above says every external map steered from an 8-bit view the
loop's own Butteraugli never sees. This measures what each map does when it sees the float reconstruction instead
(build `59e0c1d40640-loop-lin`, `scripts/byconstruction/jxl_steer_rd.sh` with `LINEAR_ARMS`/`LINEAR_SERVER_ARMS`, the same
78 TRAIN images, effort 7, 4 iterations, distances 0.3–10, judges and statistic as the steering study).
- *Placements.* Our maps' tail placements refitted on the replayed float reconstructions of the 2026-10-24-hq pairs
  (`loop_calibrate` with `LOOP_CAL_TAIL` and `LOOP_CAL_LINEAR`); Margarine's native placement (scale 1, exponent 1), as its 8-bit arm.
- *Arms:* `Mlintail`, `Tlintail`, `Blintail`, `M64lintail` (full-data M, T, M64 and B, `steer_rd --linear`) and
  `margarinelin` (zenmetrics' `margarine-steer --server`, binary `953fe4ae…`, one thread per worker).
- *Controls* (a failure stops the run before any linear result is read): `Bcheck2` must reproduce all 702 of B's
  bitstreams, and the new build's 8-bit tail refit must reproduce the recorded M, T and B rows (scale, exponent, both r).
- *Reading (my bar, as before):* an arm steers better than another when its saving is positive with the interval above
  zero under both SSIMULACRA2 and CVVDP. Each linear arm is compared with its own 8-bit arm (`--paired-vs`) and with the
  butteraugli loop, per class; loop convergence at d ≤ 2.5 (each map's last largest tile over the request, and the
  share of encodes ending with no block above target) is reported beside it.

**Controls** (both passed before any result was read): `Bcheck2` reproduced all 702 of B's bitstreams, and the new
build's 8-bit tail refit reproduced the six recorded M, T and B rows. Re-judging reproduced every earlier value of all
14,898 rows under SSIMULACRA2, butteraugli, zensim, CVVDP and DVIFM (DVIFM's table differs only in its row numbering).
Linear tail placements: M 2.590255e-2 / 0.7, T 2.479297e-2 / 0.7, B 1.779112e-2 / 0.7, M64 2.589969e-2 / 0.7 (the 8-bit
fits were 2.629540e-2, 2.510790e-2 and 1.815210e-2, all at 0.7). The first margarinelin attempt failed on the server's
protocol (a zenmetrics build from an uncommitted state wanted lowercase tags); it ran on the lane's v2 build `baf0500c`
instead, whose calibration matches. Run-heavy 1,820 s for the resume, peak RSS 0.60 GiB; tables and logs in
`/var/tmp/byconstruction-jxl-steer/` on dev.

**Results.** Bytes saved at equal judged score (%, mean over images of each image's median, 95% bootstrap interval;
SSIMULACRA2 / CVVDP). Each map's float arm against its own 8-bit arm:

| Map | All | Photos | Illustrations | Screens |
|---|---|---|---|---|
| Margarine | **+1.11 / +1.31** | +0.37 [−0.00, +0.91] / **+0.55** | +0.68 [−0.27, +1.98] / **+1.08** | **+3.18 / +3.16** |
| M (full data) | **+0.76 / +0.39** | +0.17 [+0.00, +0.40] / **+0.20** | **+0.68** / +0.21 [−0.50, +0.83] | **+2.09** / +1.01 [−0.18, +2.16] |
| T (full data) | **+0.69 / +0.56** | **+0.29 / +0.20** | **+0.45** / +0.35 [−0.17, +0.88] | **+1.84 / +1.57** |
| B | **+1.17 / +0.93** | **+0.43 / +0.30** | **+0.85 / +0.91** | **+3.12 / +2.28** |

Against the butteraugli loop (the encoder's own Butteraugli on its floats):

| Arm | All | Photos | Illustrations | Screens |
|---|---|---|---|---|
| **Margarine, floats** | +0.00 [−0.13, +0.13] / **+0.16** [+0.04, +0.28] | −0.02 / **+0.15** | −0.16 / +0.11 | **+0.26** [+0.03, +0.48] / +0.24 [−0.19, +0.68] |
| Margarine, 8-bit | −1.33 / −1.21 | −0.18 / −0.15 | −0.99 / −0.80 | −4.20 / −3.95 |
| M, floats (tail) | −0.71 [−1.68, +0.11] / −1.31 | **+1.06** / +0.27 [−0.21, +0.76] | −0.42 / −0.61 | −4.80 / −5.49 |
| M, 8-bit (tail) | −1.41 / −2.05 | **+0.88** / +0.15 | −1.34 / −1.16 | −6.32 / −7.77 |
| T, floats (tail) | −0.61 / −1.73 | **+1.09** / +0.07 | −0.22 / −0.46 | −4.69 / −7.05 |
| M64, floats (tail) | −0.71 / −1.47 | **+1.21** / +0.33 | −0.44 / −0.39 | −5.06 / −6.59 |
| B, floats (tail) | −1.38 / −2.28 | +0.39 / −1.30 | −0.23 / −0.60 | −6.53 / −6.40 |

Bold: positive with the interval above zero. Loop convergence (each map in its own units, median over images of the
last largest tile ÷ the request, `~/tmp/byc/converge_lin.py`): on screens M's loop ends at 1.13× the request at d 0.3
(3.03× from 8 bits), T 1.24 (3.74), B 1.41 (4.17), and half of M's d 1.0 encodes end with no block above target (28%);
illustrations improve alike, photos are unchanged; Margarine's photo loop ends at 1.06 at d 0.3 (1.16).

**Reading** (my bar). Every map steers better from the floats the loop holds than from their 8-bit rounding, under both
judges. Margarine fed the floats **is the butteraugli loop's equal**: a tie by SSIMULACRA2 in every class, slightly
better by CVVDP overall and on photos, with the screen gap gone. That was entirely the rounding. Our maps keep a real
screen deficit even from floats (4.7–6.5% against butteraugli by SSIMULACRA2, 5.5–7.1% by CVVDP), while M, T and M64
still beat it on photos by SSIMULACRA2 (+1.06 to +1.21%, CVVDP tied). An encoder that steers by an external metric should pass it the
reconstruction it holds; the zenmetrics web adapter (`web/codecs/jxl`) and the workbench's `loop_map` take 8-bit
pixels today.

### Owner decisions (2026-10-25)

Owner, 2026-10-25: "I take your reccomendation for euler split, yes. … For rev 7, we can restructure to our heart's
content for speed; we will eventually be shifting zensim stuff into zenmetrics and pruning zensim back to JUST rev 7 +
legacy A support. For datasets, you can add more datasets, just located an honor their train/test splits. I approve
all of them that have useful stats. And you can switch the workbench placement. Also add margaraine to the spatial
steer and see if it can approach butter better. For e21-parity screen, yes, I take your reccomendations." And: "but
also mse might matter more than ladder inversions - or another error amount distributions, if the ladder inversions
are subjective-only rather than subjective+ecnoder-knob concurring".

- **Selection gates:** the five goal-aligned gates of the gate review replace the E21-parity screen; E21's pooled
  statistic stays a reported diagnostic. Gate 3 (dial) is refined by the second message: inversions count only where
  people and the encoder knob agree, and the error amount inside ladders is weighed with them (next section).
- **Euler (exact) split** becomes the default map, as recommended (its condition: better block agreement with the
  three judges and no worse leak). *Done* (`3f779c8c`): `BakeScorer`'s served map defaults to the exact split. The
  fused and folded attribution entries keep the removal split: their standalone reference densities are removal-split,
  and the stale single-pass route always derives removal-split coefficients, so a shared exact default made two
  identical stale calls differ (caught by `stale_binned_reuse_and_no_canvas`). `attribution_cross_tier` now checks the
  served map's exact identity on every SIMD tier and keeps the removal identity through an explicit `repair` scorer.
  The full suite (`cargo test -p zensim --all-features`) passed 941 of 942 before that one expectation update, which
  then passes on all 12 tier permutations; workspace clippy is clean.
- **Workbench JPEG XL loop placements** switch to the tail fits. *Done* (`3a311ce5`): `LOOP_DEFAULTS` holds the
  largest-tile fits for Tga, M64a, B and the full-data M and T across all five maps; the served LAN workbench
  (`site-2026-10-24`, previous module kept beside it) carries the five metrics, and native/wasm smoke values agree for
  all of them. The steering scripts name the bulk placements their recorded arms used.
- **Margarine** (zenmetrics' Butteraugli approximation) joins the JPEG XL steering comparison.
- **Datasets:** every collection with useful statistics may be added, honouring its published train/test split.
- **Revision 7** may be restructured freely for speed; zensim will later shrink to revision 7 plus legacy profile A
  support, with the rest moving to zenmetrics.
- The "own versions of the missing ideas" were already built and measured: coarse scales from linear RGB (M64lin)
  and orientation bands (M64o), neither adopted ("Coarse scales in linear light", "Orientation bands" above, and the
  goal-aligned scorecard in the gate review).

**Later the same day** (after the steering gate's M3f fell under the exact split): "the jxl loop is the only thing
that matters, you should have run that first". Map, split, placement and steering defaults are decided by the JPEG XL
loop: bytes saved at equal SSIMULACRA2 and CVVDP, one variable changed per arm, with reproduction controls. Judge
block agreement, G-STEER, leak and mass on unchanged pixels are diagnostics. The exact split became the served default
(`3f779c8c`) on judge agreement alone; its loop comparison follows ("Which split steers the JPEG XL loop").

**Data, evening of 2026-10-25.** zenpapers' training-confidence manifests (zenpapers `docs/training-confidence.md`,
made that day at the owner's request) apply the owner's licence policy: research-only or non-commercial data validates
but never trains a commercial bake. Three of Mdata's legs are `validate-only` (VCL@FER, PieAPP, FG-IQA), so neither
Mdata nor Mdata64 can ship as trained; the training-mix study adds a licence-clean arm (`Mlic`). Asked about
KonFiG-IQA, which the manifests put in tier X because its references are crops of MCL-JCI sources, the owner ruled:
"mcl jci and konfig iqa can be sources. and the 3. see if it goes well." KonFiG-IQA stays a training set and the overlap
is accepted (`docs/DATA_SPLITS.md`, D3); "the 3" read as the three validate-only datasets: the licence-clean arm goes
ahead.

### Ladder inversions that people and the knob agree on (2026-10-25)

`byc_cells.py knobinv` (new): within each ladder (one reference and distortion type with levels for KADID, TID2013
and KonFiG; one reference, codec and fixed other setting with quality `q` for CID22-A), every pair with different knob
values whose human order agrees with the knob's is **concurring**; the model's inversions are counted on those pairs
only, each with its size on the human scale (the panel's logistic map, as a share of the label span p1–p99). Pairs
where people disagree with the knob are counted apart. `ladder_rms` is the error amount inside ladders: the logistic-
map residual with each ladder's mean removed, RMS over rows, as a share of the span. Knobs come from the bank keys
(`ref_group`, `codec`, `knob`), joined on `pair_key`; directions were checked without labels (KonFiG's level raises
pixel MSE in all 56 ladders, τ = +1.000; CID22-A's `q` raises encoded size for every codec except aom, whose `q` is
a quantizer, τ −0.857). Seeds 0–2, out of sample (JSON `~/tmp/byc/knobinv.json` on dev).

| | KADID (13,794 concurring / 1,285 people-only) | TID2013 (5,522 / 468) | KonFiG (5,166 / 0) | CID22-A (9,988 / 0) |
|---|---|---|---|---|
| Concurring inversion rate: `base-c` / M64 / M / T | 0.0166 / 0.0141 / 0.0139 / **0.0130** | 0.0113 / **0.0095** / 0.0097 / 0.0106 | 0.0021 / 0.0001 / **0.0000** / 0.0001 | 0.0002 / 0.0002 / 0.0001 / 0.0001 |
| Inversion size p90 (share of span) | 0.065 / 0.071 / 0.072 / 0.067 | 0.092 / 0.062 / 0.066 / **0.058** | 0.007 / 0 / 0 / 0 | 0.007 / 0.002 / 0.003 / 0.001 |
| Error inside ladders (`ladder_rms`) | 0.0926 / 0.0894 / 0.0895 / 0.0909 | 0.0643 / 0.0593 / 0.0587 / **0.0584** | **0.1258** / 0.1314 / 0.1324 / 0.1361 | **0.0694** / 0.0701 / 0.0700 / 0.0703 |

The monotone models invert less than B's recipe on every set where people and the knob agree, and their inversions
are no larger. The error amount inside ladders is smaller for them on KADID and TID2013 and larger on KonFiG (by 4–8%)
and on CID22-A (by 1%): on codec ladders B's recipe spaces the levels slightly closer to people's spacing. KonFiG and
CID22-A have no pairs where people disagree with the knob; where KADID and TID2013 have them, B's recipe follows the
knob more often (0.70 and 0.59 against 0.65–0.67 and 0.51–0.56), which says nothing about who is right. The gated arms
(Tgg, Tgy, Mgg64) sit within 0.001 of their classes on inversions; Mgg64 has the smallest KADID ladder error (0.0857).

Refined gate 3 (dial): no more concurring inversions than `base-c` on any set, inversions no larger, and the error
inside ladders compared on every set (reported with Z-RMSE and its p99), with the full-data C1–C6, G-DIAL and NEARID.

### Which cells can revision 7 skip? Scale 0 (2026-10-25)

**Withdrawn (my error).** The owner ruled on 2026-10-23 that "scale 0 is mandatory: no candidate drops the finest
scale" ("AIC2026 TRAIN" above, owner decisions; its 41–46% instruction saving was measured then). I missed that
decision and pre-registered this experiment anyway. The T class had already run when I found it; its result below
agrees with the ruling (without scale 0, JPEG is rated too well). The M class was stopped at 05:08 UTC with no cell
finished; its partial cells are kept aside (`~/tmp/byc/cells/Mns0.stopped-scale0-mandatory-0525`). `cell_share.py`
and `byc_stub_bake` stay as tools; no read set without scale 0 is a candidate.

The owner's goals include less compute, and revision 7 may be restructured freely for speed. Dropping single
features saves nothing when their kernel still runs for other features (ringing, "Cost of the ringing fix" above); time
goes only when a whole kernel, scale or channel goes. The walk already skips scale 0's chroma for fx1 models: B, M, T
and M64 read none of its 82 X and B inputs (`full_res_xb` off, so the producer omits those planes). What remains at
scale 0 is its luma cell, which runs every kernel at full resolution: about as many pixels as the nine coarser cells
together.

**How much of the score scale 0 carries** (`scripts/byconstruction/cell_share.py`, label-free: feature columns of the
training root's `cid22_fit`, `safesyn_fit` and `human_all_fit` tables, no label read). Each row's score drop from the
identity input is split over inputs by the live units' slopes; a group's ablation sets its damage inputs and their
fragility products to their identity value. The models are close to linear, so both agree:

| Share of the score drop | M (cid22 / safesyn / human) | T | M64 |
|---|---|---|---|
| Scale 0 (luma only) | 0.242 / 0.247 / 0.227 | 0.234 / 0.222 / 0.215 | 0.241 / 0.245 / 0.226 |
| Scale 1 | 0.343 / 0.328 / 0.319 | 0.403 / 0.363 / 0.346 | 0.344 / 0.328 / 0.320 |
| Scale 2 | 0.282 / 0.316 / 0.300 | 0.242 / 0.283 / 0.273 | 0.283 / 0.317 / 0.301 |
| Scale 3 | 0.126 / 0.106 / 0.141 | 0.121 / 0.132 / 0.166 | 0.125 / 0.105 / 0.140 |

By family (M, the same three tables): edge detail loss 4th power 0.19, artifact soft peak 0.15–0.17, HF loss
0.06–0.08, edge detail loss 2nd power 0.04–0.07, blockiness 0.03–0.07; the 20 smallest of 39 families carry 7% together,
but they share kernels with the large ones. So scale 0 is the one large unit of work a retrained model could give up.

**Design (recorded before any result).**
- *Speed:* `byc_stub_bake` (new, `zensim-validate`) writes a linear stub over a list of feature ids, so the scorer
  plans exactly that read set. Paired timing (`perf_grid.sh`, 8 rounds, one thread on core 2, v4x and v3, the focus
  cases 256² photo, 1024² plot, 1507×2048 photo) of a stub over M's 420 ids against the same ids without scale 0 (378:
  the cell's 41 damage ids and its fragility id), both at revision 7.
- *Arms:* `fx1ns0` (new declaration `byc_fx1ns0_declaration.json`, SHA-256 `e74f8cc1…`): fx1 without its scale-0 cell,
  369 differences and their 369 fragility products, selection `sel:04207431eeb5`. **Mns0** (`mono:thr:mp32`, M's recipe)
  and **Tns0** (`mono:thr`), the 12 E21 cells, seeds 0–2, on the raw root (as M and T), trainer `774955ad`. Control:
  M's KADID seed-0 cell retrained on `774955ad` (**Mrepro**) must give predictions bit-identical to the stored M cell
  (trainer `d2b7ce30`), or no comparison with M is made.
- *Statistics:* the goal-aligned scorecard against M and T: within-reference ordering and per-type and codec bias
  spread (`byc_cells.py bytype`, `type_bias.py`), knob-concurring inversions and ladder error (`knobinv`), smoothness
  (`aic_panel.py --geo --no-aic`), the standard and global-edit mixture tests (image route, `mixture_test.py`), and E21's
  pooled Δ as a diagnostic. No AIC2026 label is read.
- *Reading (my bars).* The scale-0 drop goes to a full-data fit (NEARID, the dial gates, G-STEER, identity, JPEG XL
  steering) if, in its class: no global-edit step inverts; standard-mixture inversions stay within the full arm's range
  (M 1–32, T 0–4 per cell); knob-concurring inversions are no more than `base-c`'s on every set; within-reference
  ordering is no worse than `base-c` on every set; and CID22-A's codec spread is within 0.005 of the full arm's (M 0.0345,
  T 0.038). Losses smaller than these against M or T are the price of the measured speed, reported, not hidden.

**Result, T class** (Tns0, 12 cells 04:51–05:03 UTC, trainer `774955ad`; seeds 0–2, out of sample):

| | `base-c` | T | **Tns0 (no scale 0)** |
|---|---|---|---|
| Within-reference ordering Δ vs `base-c` (KADID / TID2013 / KonFiG / CID22-A) | 0 | +0.0146 / +0.0080 / +0.0036 / 0.0000 | +0.0131 / +0.0109 / +0.0037 / +0.0001 |
| Knob-concurring inversions (KADID / TID2013) | 0.0166 / 0.0113 | 0.0130 / 0.0106 | 0.0135 / **0.0116** |
| Inversion size p90 (KADID / TID2013 / CID22-A) | 0.065 / 0.092 / 0.007 | 0.067 / 0.058 / 0.001 | 0.077 / 0.066 / 0.004 |
| Ladder error (KADID / TID2013 / KonFiG / CID22-A) | 0.0926 / 0.0643 / 0.1258 / 0.0694 | 0.0909 / 0.0584 / 0.1361 / 0.0703 | 0.0910 / 0.0591 / 0.1363 / 0.0698 |
| Per-type bias spread (KADID / TID2013 / KonFiG) | 0.060 / 0.054 / 0.079 | 0.084 / 0.099 / 0.090 | 0.083 / 0.102 / 0.089 |
| CID22-A codec bias spread | 0.0340 | 0.0376 | **0.0450** |
| E21 Δ, diagnostic (KADID / TID2013 / KonFiG / CID22-A) | 0 | −0.0205 | −0.0227 (−0.018 / −0.047 / −0.029 / +0.003) |

Against the bars it fails twice: CID22-A's codec spread rises by 0.0074 (bar 0.005), and on TID2013 it inverts more
knob-agreed pairs than `base-c` (0.0116 against 0.0113). The codec spread says where scale 0 matters. Without it the model
rates mozjpeg better than people do by +0.069 (T +0.049) and JPEG 2000 by +0.056 (T +0.045), while AVIF (aom −0.031,
vis_avif −0.048) and WebP (−0.048) move the other way: JPEG's fine-scale artifacts live at full resolution, and the coarser
scales do not see them. Ordering within references barely moves (KADID −0.0015, TID2013 +0.0029 against T). The mixture
tests were not run, and the M class was stopped (see the note at the top of this section).

### Before the revision-7 cut: which ringing form (2026-10-25, for the owner)

Every bit-moving speed item listed for revision 7 has landed ("Revision 7 speed after the implementation"); NEON vector
kernels (set aside by the owner on 2026-10-25) and the multithreaded line-buffer walk keep every bit, so neither blocks a
cut. One
definition does: revision 7 is revision 6 plus the row lanes, and revision 6 computes ringing in its high-pass form
(`revision.ring_high_pass()`, chosen to serve Tga, a candidate the gate review retired). So the leading model's recipe
retrained at revision 7 would be **Mhpm**, not M. On the goal-aligned scorecard (E21 cells, seeds 0–2; tables above):

| | M (raw ringing, revision 5) | Mhpm (high-pass ringing) |
|---|---|---|
| CID22-A codec bias spread | **0.0345** | 0.0383 |
| Standard mixture inversions per cell (largest rise) | **1–32 (+6.5)** | 2–35 (+8.0) |
| TID2013 saturation bias | **+0.246** | +0.282 |
| Knob-concurring inversions, TID2013 | 0.0097 | **0.0084** |
| Mean Z-RMSE (CID22-A Z-RMSE / error p99) | 0.502 (0.489 / 0.377) | 0.504 (0.488 / 0.375) |
| E21 Δ, diagnostic | −0.0144 | −0.0158 |

For a monotone model the raw form scores better on codec uniformity and robustness. Its defect (imazen/zensim#72:
ringing responds to global edits) matters when a signed model subtracts it as a global-change detector. A monotone model
cannot subtract, and the raw response to saturation pushes saturation's under-penalty the right way. Options for revision 7: (a) keep
revision 6's high-pass form; (b) restore revision 5's raw ringing in revision 7, keeping the high-pass form as a research
toggle. My recommendation is (b); the re-extraction and retrain wait for this choice.

### Revision 7 with raw ringing (2026-10-25)

Owner, 2026-10-25: "ok, restore and train/test raw ringing then." Revision 7 now computes revision 5's measurements, raw
ringing included, with revision 6's `globaff` family and the row-lane arithmetic; `ZENSIM_RING_FORM=hpm` still selects
the high-pass form as a research arm (`FormulaRevision::ring_high_pass` is revision 6 only; the gradient kernel gains the
revision-7 raw-ringing instantiation in its canonical and exact-oracle dispatch; local refinement, which refuses the
high-pass form, works at revision 7 again).

Checks: `cargo test --release -p zensim --all-features` 942 passed, 0 failed, 29 ignored; CI-exact clippy clean.
`featcanon_rev7_parity` now compares revision 7 with revision 5 on basic + peaks + v2 and with revision 6 on `globaff`
(both by rounding only, the existing 1e-2 floored-relative guard), and requires the ringing slots to leave revision 6's
high-pass values. Bit dumps of B stamped at revisions 5 and 7 (one binary, the 11 inputs of `perf_bits.sh`): revision 7
is identical across v4x, v4, v3 and scalar at one thread and v4x at eight (55 runs, 0 mismatches); against revision 5,
4,438 of 7,920 finite feature values move, by a median 1.1e-6 relative (p90 1.7e-5, p99 2.3e-4, largest 4.0e-3 on the
97×63 pair's f664, an absolute 2.3e-8), and B's score by at most 5.4e-5 points (median 6.7e-6).

**Plan (recorded before any result).** (1) Re-extract the eight training and assessment sets of the `v2e29` root and
its E15 coverage pool at revision 7 (`ring_bank.sh … rev7`, era `rev7_rowlanes`, feature set
`basic+peaks+v2@w1825/rev7_rowlanes#36c3f3af`); control first: the same extractor at revision 5 must reproduce the stored
Rev5 bank bit for bit on `konfig_train` and `cid22_a25`, and every revision-7 value must stay within 1e-2 relative
(+1e-6) of its Rev5 value (`ring_root.py` refuses more than 0.1% of cells beyond that). (2) Admit the root (registry era
and set, Python and trainer admission accept formula revision 7 for that set only). (3) Train M (`mono:thr:mp32`) and T
on the 12 E21 cells at revision 7 and compare with their revision-5 cells on the goal-aligned scorecard. Reading (my
bar): revision 7 is a faithful re-extraction when every scorecard statistic stays within its seed spread of the
revision-5 cell; any larger move is reported as a finding. (4) Full-data M at revision 7 through the product gates,
with gate tools rebuilt at the head after they reproduce M's revision-5 gate results exactly.

**Root built** (`/mnt/v/output/zensim/byconstruction-2026-10-22/rev7raw/root`, extractor `ed495741…` at `b77883be`,
the `zensim-bench` lockfile of the earlier branch builds). Control passed first: at revision 5 the new extractor
reproduces the stored Rev5 bank on all 576 requested slots of `konfig_train` (327 rows) and `cid22_a25` (2,192 rows).
At revision 7, values that ring_root.py's check counts as moved (beyond 1e-2 relative + 1e-6), per table:

| Tables | Cells beyond the bound | Most in one row (of 576) | Columns that moved most |
|---|---|---|---|
| Human legs (KADID, TID2013, KonFiG, CID22-A and the `human_*` tables) | 0–7 per table | 2 | none systematic |
| `cid22_fit` / `cid22_dev` (codec encodes of training references) | 2,192 / 1,314 (0.03% / 0.06%) | 24 / 28 | v2 HF magnitude loss |
| `safesyn_fit` / `safesyn_dev` (synthetic distortions) | 505,949 / 146,733 (0.62% / 0.66%) | 49 / 47 | v2 HF magnitude loss (every cell), v1 texture loss on flat chroma |
| E15 coverage pool | 11,925 (0.05%) | 46 | v2 HF magnitude loss |

So revision 7 moves natural-image features by rounding (median 4–6e-7 relative), but on near-degenerate content it
corrects values revision 5 computed from noise. On safesyn, `hf_mag_loss` drops from about 1e-4 toward its exact
value (revision 5's `max(0, ·)` turned blur rounding into positive mass). The v1 texture loss on flat chroma goes from
about 0.79, a ratio of two vanishing sums, to 0. Edge-width change and blockiness move at their noise floor. The
dump-based measurement above ("moves only by rounding") holds for the natural images it measured, not for this
content. My first closeness bound (refuse more than 0.1% of cells beyond 1e-2) mistook these corrections for a join
error and refused safesyn_dev. A wrong join moves most of a row: with safesyn's rows shuffled, a row disagrees in a mean of
546 of its 576 columns, against at most 49 for the true join, so the check now refuses more than 0.1% of rows with
over a quarter of their columns moved.

**Cells** (`Tr7` = T's recipe `mono:thr`, `Mr7` = M's `mono:thr:mp32`, B's fx1 spec and columns C, the 12 E21 cells,
seeds 0–2, the revision-7 root; trainer binaries `be72a3dfb039`; scorecard `~/tmp/byc/score_r7.sh`, outputs under
`~/tmp/byc/r7_scorecard/` on dev):

| | T (rev 5) | Tr7 | M (rev 5) | Mr7 |
|---|---|---|---|---|
| E21 Δ (diagnostic), seed-unit SE | −0.0205 ± 0.0029 | −0.0203 ± 0.0029 | −0.0144 ± 0.0016 | −0.0127 ± 0.0016 |
| Paired Δ against the revision-5 cell, per seed | | +0.0001 / +0.0002 / +0.0005 | | +0.0038 / −0.0021 / +0.0034 |
| SROCC KADID / TID2013 / KonFiG / CID22-A | .8714 / .8590 / .7904 / .8860 | .8720 / .8593 / .7905 / .8860 | .8759 / .8761 / .7988 / .8808 | .8793 / .8736 / .8007 / .8846 |
| Codec/type bias spread KADID / TID / KonFiG / CID22-A | .0840 / .0985 / .0899 / .0376 | .0832 / .0983 / .0899 / .0376 | .0817 / .0824 / .0852 / .0345 | .0796 / .0854 / .0903 / .0356 |
| Knob-concurring inversions KADID / TID2013 | .0130 / .0106 | .0131 / .0106 | .0139 / .0097 | .0136 / .0091 |
| Standard mixture: inversions per cell (all cells), largest rise | 0–4 (28), +4.03 | 0–4 (28), +4.04 | 1–32 (82), +6.54 | 1–14 (57), +6.28 |
| Global mixture: inversions | — | 0 | 0 | 0 |

Reading against the bar set above: **T at revision 7 reproduces T** (every statistic within its seed spread, most to
the last printed digit). **M at revision 7 stays within seed spread on E21** (per-seed differences of mixed sign);
M's training amplifies small input changes more than T's, so its per-dataset statistics move by a few thousandths in
both directions (bias spreads: KADID better, TID2013/KonFiG/CID22-A worse, no seed interval computed for the pooled
spreads). One finding: Mr7's standard-mixture inversions fall from 82 to 57 (the worst cell 32 → 14). The bias spreads
are pooled over seeds, so they carry no seed interval here; their moves are the size of the M/Mhpm differences above.
Next: the full-data Mr7 fit and its gates at revision 7 (rebuilt tools, after the revision-5 controls reproduce M's
stored gates).

### Which split steers the JPEG XL loop (2026-10-25)

The served map moved to the exact split (`3f779c8c`) on judge block agreement alone; on G-STEER (E34's 135
block-repair cases, at revision 5 with the rebuilt instrument) the exact split lowers M3f on 130 of 135 cases (mean
−0.045, worst 0.930 → 0.800; all still pass), because G-STEER's truth is block removal, which the removal split
predicts by construction. Owner: "the jxl loop is the only thing that matters, you should have run that first".

**Design.** The loop of the float-reconstruction study ("JPEG XL steering on the loop's float reconstruction": the
same 78 TRAIN images, distances 0.3–10, effort 7, 4 iterations, judges and statistic), build `ba73151f5e1e-loop-split`,
full-data M (`6291eeb5`) and B (`89cb4861`), float input, each map at its own tail placement refitted on the same float
replay. Maps (workbench `MAPS`): `diffmap` (exact split, centred window, guided paint, σ 3: the recorded `Mlintail` and
`Blintail` arms), `diffmap_rem` (the same with the removal split, new), `removal` (the old served map: removal split,
box window, nearest paint, no smoothing) and `diffmap_amp` (amplitude split, σ 3). Placements (scale, exponent): M
2.590255e-2/0.7, 4.547074e-2/0.7, 2.193165e-2/0.6, 4.220431e-2/1; B 1.779112e-2/0.7, 4.183675e-2/0.7, 3.565719e-2/0.5,
4.641337e-2/0.9 (`loop_calibration_2026-10-25split.tail.tsv`).

**Controls** (all passed): the new build's `diffmap` float tail refit reproduces the four recorded M and B rows; `Bcheck3`
reproduces all 702 of B's bitstreams and `Mlinchk` all 702 of `Mlintail`'s; re-judging reproduces every earlier value
of the 19,110 earlier rows under SSIMULACRA2, butteraugli, zensim and CVVDP, and DVIFM's values (its table renumbers
rows). Run-heavy 3,014 s, peak RSS 1.05 GiB.

**Results.** Bytes saved at equal judged score (%, mean over images of each image's median, 95% bootstrap interval;
SSIMULACRA2 / CVVDP), each arm against its model's exact-split `diffmap` arm:

| Arm | All | Photos | Illustrations | Screens |
|---|---|---|---|---|
| M, removal split (`diffmap_rem`) | **−1.19 / −1.03** | −0.62 / −0.09 | **−2.57 / −2.91** | −0.72 / −0.73 |
| M, old served map (`removal`) | −0.30 / +0.14 | **−0.76 / −0.34** | −0.79 / −0.36 | **+1.27** / +1.77 [−0.09, +3.68] |
| M, amplitude split | −0.48 / +1.01 | **+0.99 / +3.37** | −0.14 / +0.75 | **−4.00** / −3.64 [−9.21, +0.31] |
| B, removal split | −0.80 / −0.21 | +0.34 / +1.12 | −0.99 / −0.23 | **−3.00 / −2.98** |
| B, old served map | +0.13 / **+0.56** | **−0.46 / +0.49** | +0.08 / +0.25 | +1.44 [−0.18, +3.40] / +1.10 |
| B, amplitude split | **−3.19** / −0.87 | +0.35 / **+3.57** | **−3.28** / −2.10 | **−10.57 / −8.74** |

Bold: the interval excludes zero (negative: worse than the exact split). Against the butteraugli loop, M's exact-split
arm stays the best of M's maps overall (−0.71 / −1.31; photos +1.06 / +0.27; screens −4.80 / −5.49); the old map is
−1.72 / −2.50 overall and −4.11 / −4.42 on screens.

**Reading.** In the loop, **the exact split is the better split**: changing only the split to removal costs M 1.2% /
1.0% of bytes (illustrations 2.6% / 2.9%) and costs B 3.0% on screens, with no class where removal wins. G-STEER's
preference for the removal split does not carry to the encoder. The old served map's screen advantage for M (+1.27 /
+1.77) therefore comes from its box window, nearest paint or missing smoothing, not from its split; it pays for it on
photos. The amplitude split helps photos (+1.0% / +3.4% for M, +3.6% CVVDP for B) and fails on screens. Next
(running): the exact split with the old map's box window and nearest paint, unsmoothed and at σ 3, and `diffmap`
unsmoothed, to place the screen gain; and full-data M retrained at revision 7 in the same loop.

### The loop statistic favours the challenger; the loop results by BD-rate (2026-10-25)

**The finding.** `rd_probe_analyze_2026-07-18.py --paired-vs` puts each of an arm's points on the baseline's per-image
curve (log bytes, straight lines between the baseline's nine points) and takes each image's median saving. On a
convex rate curve the straight lines overstate the baseline's bytes, so the arm under test gains: `exact_box` against
`diffmap` "saves" +0.25% and `diffmap` against `exact_box` "saves" +1.55% (SSIMULACRA2, M, all images), and the
direct comparison disagreed with the one through the butteraugli loop (`exact_box` −0.69 points behind `diffmap` there).
The new `--bdrate-vs LABEL` (`4b58f139`) takes, per image and judge, the mean log-bytes difference of both arms'
piecewise-linear curves over the score range both reach, then the mean over images with a 95% bootstrap interval.
Checks: identical arms (`Mlinchk`, `Mlinchk2`) give exactly 0.00; swapping the arms flips the sign (−1.21 / +1.16);
the direct difference agrees with the one through butteraugli (−1.21 against −1.04 SSIMULACRA2, −1.35 against −1.22
CVVDP). Every loop number in this section is BD-rate: bytes saved at equal judged score (%, SSIMULACRA2 / CVVDP, 95%
interval, 78 TRAIN images; bold where the interval excludes zero).

**Superseded by it** (sections above, left as written): in "JPEG XL steering on the loop's float reconstruction",
our maps beating the butteraugli loop on photos (M +1.06, T +1.09, M64 +1.21, SSIMULACRA2) and the size of the float
gains; in "Which split steers the JPEG XL loop", the old served map's overall tie. The split conclusion and the
float-over-8-bit conclusion stand.

**Revision 7 in the loop** (run `00cc14ed2af0-loop-r7win`; controls: `Bcheck4` and `Mlinchk2` reproduce all 702
bitstreams of B and `Mlintail`, the `diffmap` float placement refits reproduce, re-judging reproduces all 24,726
earlier rows under all five judges; run-heavy 3,014 s for the split run, 3,812 s for this one, peak RSS 1.06 GiB). Full-data M retrained at revision 7 (`bc471d21`,
certified monotone, float placement 2.586523e-2 / 0.7) against M at revision 5, both `diffmap` on floats: all +0.02 /
−0.07, photos −0.02 / −0.01, illustrations +0.03 / +0.08, screens +0.10 / −0.37, every interval across zero.
**Revision 7 steers the JPEG XL loop as revision 5 does.**

**Maps** (each against its model's exact-split `diffmap`, floats, own tail placement):

| M (full data) | All | Photos | Illustrations | Screens |
|---|---|---|---|---|
| removal split (`diffmap_rem`) | **−1.46 / −1.07** | −0.79 / −0.13 | **−2.23 / −1.88** | −1.94 / −2.04 |
| old served map (removal, box, no smoothing) | **−1.22 / −1.03** | **−1.68 / −1.95** | **−1.80 / −1.77** | +0.48 / **+1.83** |
| exact, box, no smoothing (`exact_box`) | **−1.21 / −1.35** | **−2.05 / −2.57** | **−1.16 / −1.47** | +0.50 / +1.37 |
| exact, box, σ 3 (`exact_box3`) | +0.01 / +0.08 | **−0.12** / −0.05 | −0.23 / −0.07 | +0.59 / +0.55 |
| `diffmap` unsmoothed (`diffmap_px`) | **−2.11 / −2.44** | **−2.35 / −2.89** | **−2.23 / −2.53** | **−1.47 / −1.37** |
| amplitude split (`diffmap_amp`) | −0.24 / **+1.10** | **+0.90 / +2.63** | −0.50 / +0.71 | **−2.33** / −1.66 |

B gives the same picture (removal −0.77 / −0.18; old map −0.08 / +0.18 with photos **−1.04 / −0.79** and screens
**+2.44 / +2.72**; `exact_box` **−1.84 / −2.40**; `exact_box3` +0.16 / +0.09; `diffmap_px` **−1.99 / −2.58**; amplitude
**−3.60** / −0.77 with photos +0.38 / **+3.20** and screens **−11.71 / −8.34**).

**Against the butteraugli loop:** M `diffmap` **−2.34 / −4.26** (photos **−0.79 / −3.24**, illustrations
**−1.74 / −2.79**, screens **−6.34 / −8.21**); revision-7 M **−2.32 / −4.32**; B **−3.37 / −5.63**; Margarine on
floats −0.03 / −0.04 (a tie in every class). **Floats against 8 bits:** Margarine **+0.93 / +0.97**, B
**+0.92 / +0.85**, M **+0.48 / +0.38**, T **+0.44** / +0.34.

**Reading.** The exact split stays the right default. Smoothing is the lever among the factorial's parts: removing
σ 3 costs 1.2–2.4% with either window, while the box window and nearest paint at σ 3 tie `diffmap`. The old map's
screen gain is a CVVDP effect bought with photos and illustrations. The amplitude split is the one map that gains on
photos (+0.9% / +2.6% for M), at a cost on screens. Our maps still trail the encoder's own Butteraugli loop
everywhere, by about 2.3% / 4.3% for M; Margarine on floats is its equal.

**Revision-7 speed** (paired, 8 rounds, one thread pinned, one binary `9b1f6dca`, B's revision-5 bake against its
byte-identical revision-7 stamp; load 4.5–6.8 during the run; grids in `benchmarks/byconstruction_perf_2026-10-24/`):
v4x 1507×2048 photo 47.09 → 28.43 ms (−39.60%, −39.99..−39.08), 256² photo 1.219 → 0.824 ms (−32.68%), 1024² plot
16.35 → 9.79 ms (−40.22%); v3 −36.09%, −31.83%, −35.68%. Revision 7's raw ringing against its high-pass research
arm: the high-pass form costs +0.3% to +1.4%.

### The new datasets' EVAL splits (2026-10-25)

The provided-library collections admitted on 2026-10-25 (`docs/DATA_SPLITS.md`), scored on their frozen EVAL rows,
which no model has trained on: 11 collections, 9,658 rows, 4,678 distinct image pairs
(`scripts/byconstruction/dataset_eval.py`; outputs in `/var/tmp/byc-dataset-eval-2026-10-25/` on dev). Models from the
revision-7 feature banks through the production predictor in score units: **M7** (full-data M retrained at revision 7,
`bc471d21`) and **B7** (B's weights stamped revision 7, `5fe0a962`). Peers on the same pixels: SSIMULACRA2, Butteraugli
(p-norm 3) and CVVDP (`cvvdp_cpu_imazen_v0_1_0`) from the pinned judge build. MOS tables are oriented to higher =
better and scored by Spearman correlation over all rows (and the mean within references); PieAPP and FG-IQA by the
two-alternative agreement `p_a·[A ahead] + (1 − p_a)·[B ahead] + ½·[tie]`. Intervals: 95% bootstrap over references
(2,000 resamples), per model, not paired.

Controls: the judge build cannot read BMP, so the 1,555 distinct BMP images were decoded, their RGB8 SHA-256 checked
against the pixel hash the feature extractor recorded for each (all equal), and written as PNG for the judges; JPEG
inputs (888) the judges decode themselves. No EVAL row has a source in our existing training data under the admission
screen (`existing_training_source_exposure` false for all 9,658). Label reads: EVAL rows only, listed in
`LABEL_READS.json` beside the outputs. Left out: LocVisVC (spatial markings), Colourlab chroma masking (a
masking-threshold design), DiffIQA and SPCD (no pristine reference), CID:PPS (TIFF), PIPAL and IE-Bench (no EVAL rows).

| Collection | Rows | Statistic | B7 | M7 | butteraugli | cvvdp | ssim2 |
|---|---|---|---|---|---|---|---|
| cid-iq | 210 | SROCC | 0.871 [0.841, 0.899] | 0.861 [0.841, 0.883] | 0.590 [0.519, 0.659] | 0.748 [0.715, 0.795] | 0.756 [0.735, 0.785] |
| | | within-reference SROCC | 0.882 | 0.865 | 0.594 | 0.745 | 0.763 |
| cid-oa | 91 | SROCC | 0.880 [0.802, 0.976] | 0.913 [0.859, 0.972] | 0.582 [0.472, 0.908] | 0.912 [0.874, 0.961] | 0.847 [0.772, 0.949] |
| | | within-reference SROCC | 0.988 | 0.989 | 0.988 | 0.988 | 0.988 |
| colourlab-rating-variability-2022 | 192 | SROCC | 0.703 [0.696, 0.732] | 0.725 [0.695, 0.764] | 0.375 [0.247, 0.590] | 0.614 [0.543, 0.731] | 0.606 [0.570, 0.696] |
| | | within-reference SROCC | 0.724 | 0.735 | 0.381 | 0.621 | 0.634 |
| fg-iqa | 540 | 2AFC agreement | 0.758 [0.731, 0.784] | 0.757 [0.731, 0.782] | 0.749 [0.726, 0.772] | 0.755 [0.728, 0.782] | 0.758 [0.729, 0.784] |
| iqa-odi | 48 | SROCC | 0.849 [0.716, 0.909] | 0.843 [0.703, 0.904] | 0.806 [0.639, 0.896] | 0.833 [0.701, 0.898] | 0.854 [0.726, 0.906] |
| | | within-reference SROCC | nan | nan | nan | nan | nan |
| irccyn-ivc | 60 | SROCC | 0.959 [0.911, 0.977] | 0.933 [0.826, 0.957] | 0.909 [0.898, 0.945] | 0.926 [0.892, 0.937] | 0.949 [0.924, 0.979] |
| | | within-reference SROCC | 0.982 | 0.964 | 0.948 | 0.926 | 0.970 |
| live-md | 180 | SROCC | 0.879 [0.843, 0.920] | 0.891 [0.867, 0.934] | 0.837 [0.772, 0.917] | 0.899 [0.861, 0.939] | 0.916 [0.889, 0.947] |
| | | within-reference SROCC | 0.932 | 0.955 | 0.953 | 0.959 | 0.958 |
| mdid | 480 | SROCC | 0.892 [0.858, 0.929] | 0.894 [0.860, 0.930] | 0.572 [0.444, 0.718] | 0.829 [0.781, 0.884] | 0.787 [0.713, 0.858] |
| | | within-reference SROCC | 0.908 | 0.907 | 0.592 | 0.860 | 0.803 |
| nits-iqa | 405 | SROCC | 0.703 [0.671, 0.745] | 0.716 [0.678, 0.764] | 0.705 [0.681, 0.732] | 0.721 [0.674, 0.774] | 0.776 [0.732, 0.829] |
| | | within-reference SROCC | 0.711 | 0.728 | 0.712 | 0.741 | 0.795 |
| pieapp | 7260 | 2AFC agreement | 0.579 [0.572, 0.587] | 0.582 [0.576, 0.588] | 0.535 [0.524, 0.546] | 0.570 [0.562, 0.578] | 0.560 [0.551, 0.568] |
| vcl-fer | 192 | SROCC | 0.884 [0.846, 0.921] | 0.938 [0.923, 0.959] | 0.877 [0.848, 0.906] | 0.933 [0.912, 0.956] | 0.945 [0.930, 0.965] |
| | | within-reference SROCC | 0.910 | 0.957 | 0.917 | 0.953 | 0.967 |

Reading. Against SSIMULACRA2, revision-7 M is ahead on CID:IQ (0.861 against 0.756), MDID (0.894 / 0.787), Colourlab
(0.725 / 0.606), CID:OA (0.913 / 0.847) and PieAPP (0.582 / 0.560, intervals apart); behind on NITS (0.716 / 0.776) and
LIVE-MD (0.891 / 0.916), with overlapping intervals; tied on IRCCyN/IVC, IQA-ODI, VCL@FER and FG-IQA. Butteraugli is
last or near last on nine of eleven. B7 and M7 differ little except VCL@FER (M7 0.938, B7 0.884) and IRCCyN/IVC (B7
0.959, M7 0.933). The per-model intervals are not a paired test of the differences.

## Summary (all arms, E21 screen, seeds 0–2, Δ vs `base-c`)

| Arm | Guarantee | E21 Δ (SE) | Mixture inversions / cell (max rise) |
|---|---|---|---|
| Ts: B + thresholds | pin only | −0.0008 (0.0012) | — |
| Bhpm: B, ringing `hpm` | pin only | −0.0002 (0.0023) | 78–184 (+53.8) |
| Bghpm: B + descriptor | pin only | −0.0031 (0.0017) | 70–190 (+61.4) |
| Bnr: B without ringing | pin only | −0.0003 (0.0009) | 72–178 (+53.1) |
| FR: ringing sign-free | monotone in 800 of 820 | −0.0039 (0.0029) | 2–41 (+26.2) |
| FRhpm: `hpm` ringing sign-free | monotone in 800 of 820 | −0.0042 (0.0029) | 6–70 (+35.8) |
| **Tghpm: descriptor sign-free** | **monotone in all 820 damage inputs** | **−0.0101 (0.0024)** | **2–7 (+7.1)** |
| **M64: max pairs** | **certified monotone** | **−0.0131 (0.0023)** | 1–6 (+7.6) |
| M64lin: M64, coarse scales from linear RGB | certified monotone | −0.0134 (0.0034) | 2–5 (+3.9) |
| M64o: M64 + orientation bands | certified monotone | −0.0168 (0.0020) | 2–18 (+7.6) |
| **M: max pairs + hinges** | **certified monotone** | **−0.0144 (0.0016)** | 1–32 (+6.5) |
| Mhpm: M, ringing `hpm` | certified monotone | −0.0158 (0.0017) | 2–35 (+8.0) |
| E20: budget 20 | ≤ +20 certified | −0.0154 (0.0026) | 4–37 (+7.3) |
| E5: budget 5 | ≤ +5 certified | −0.0194 (0.0027) | 3–8 (+5.0) |
| G: content-gated | certified monotone | −0.0202 (0.0030) | 0–4 (+4.5) |
| Lc: linear | certified monotone | −0.0204 (0.0029) | — |
| T: thresholds | certified monotone | −0.0205 (0.0029) | 0–4 (+4.0) |
| Thpm: T, ringing `hpm` | certified monotone | −0.0221 (0.0032) | 0–4 (+4.2) |
| DR: ringing dropped | certified monotone | −0.0216 (0.0031) | 0–3 (+4.3) |
| R: ramps | certified monotone | −0.0225 (0.0024) | 4–9 (+5.2) |
| DG: global-sensitive dropped | certified monotone | −0.0229 (0.0032) | — |
| `base-c` (B's recipe) | pin only | 0 | 86–171 (+61.5) |

Full-data certified-monotone models (T, M, M64) pass every label-free product gate; M: C1
0.9984 / 0.9935, NEARID 143/144, G-STEER 135/135, identity at all tiers; M64: C1 0.9984 / 0.9937,
NEARID 142/144, G-STEER 135/135, identity at all tiers.

**Answer to the question.** A metric that is monotone by construction is competitive with B
on the product contract (dial, identity, steering) and far more robust (inversions bounded to a
few points, no blur + chroma-noise failure), and at parity on codec data (CID22-A). It is not yet
competitive on the E21 cross-dataset rank statistic: the best certified-monotone arm is −0.013.
That gap is cross-type severity calibration on synthetic distortion sets; with oracle per-type
calibration the monotone models match B. Max-pooling recovers about a third of it inside the
monotone class. Inside the monotone class most of the rest runs through one conflated feature
(v2 ringing, imazen/zensim#72): freeing its sign alone recovers it (FR −0.0039), because ringing
doubles as a global-change detector. B itself does not depend on ringing (Bnr −0.0003, the same
inversions): its sign freedom finds that discount in other features and forgives chroma noise
with it. The high-pass ringing fix costs under 1% single-threaded. Closing the remaining gap by construction needs a feature
change: damage measured on the affine-residual difference plus explicit global-change terms
(the SSIM luminance/contrast/structure decomposition applied to this bank).

**Update 2026-10-23.** Ringing is corrected (`ZENSIM_RING_FORM=hpm`, research arm; raw stays the default and every
stored value is unchanged) and retrained. The correction does not change any model's E21 beyond noise (Bhpm −0.0002,
Mhpm −0.0158, Thpm −0.0221), and it does not remove B's inversion: that comes from sign freedom. The next feature, a
global affine-change descriptor that noise cannot drive, gives the monotone class a safe way to discount global
changes: Tghpm reaches −0.0101 with 2–7 mixture inversions per cell (dithering only), the best E21 of any arm that
keeps inversions bounded. It is a research sidecar today; serving it needs the descriptor in the streaming walk (five
additive sums per channel, see "Can the fixes stream?"). Timing the correction found a 9% speedup for the shipped
scorer (64-byte-aligned lane accumulators), so the corrected scorer is 7–8.5% faster than the shipped one; the ring
form itself costs about 0.5%.

### Revision-7 M with provided-library TRAIN datasets (2026-10-25)

The full-data candidate passes the label-free checks, while the three-seed
E21 mean falls by 0.002701 SROCC against Mr7 (seed-unit SE 0.002163).
The new EVAL results are mixed; BAPPS soft agreement rises from 0.661796
to 0.673924. No serving profile changes. All 12 cells and both mixture
tests are complete.

Registered before reading the new TRAIN labels: `Mdata` keeps M's
`sel:3b7bd5ebe929@h32:H128:cv16:cf98:fx1:mono:thr:mp32`, columns C, seeds,
120 × 50,000 budget and fixed final epoch 119. Eight MOS collections enter as
within-reference ranks with native `higher_is_better` orientation; BAPPS,
PieAPP and FG-IQA enter as pair-local hard preferences: `p_a > .5` prefers A,
`p_a < .5` prefers B, and exact `.5` comparisons drop. Probability magnitudes
and vote counts do not enter the loss. Each added collection has nominal
weight 0.5, corrected for uniform within-reference self draws by the existing
sampling owner. This is a fixed recipe, with no held-out weight selection.

The fresh revision-7 authority extends the existing subjective/group admission
owners. Metadata, ordered keys and source-family reservations pass before
payload reads. Every EVAL/TEST/CALIBRATION family, including excluded
memberships and source-exposed TEST diagnostics, stays outside fits. The
label-free audit found seven inherited TID families (840 full-human fit rows)
in excluded TEST memberships; the derived root masks these too. Thus the
comparison includes both new datasets and that required family mask. Frozen
revision-7 source tables stay intact. The new root, label-read ledger and full fit live under `rev7data/`;
assessments and cells live in sibling directories of the study.
No production profile changes.

The full-data candidate is now frozen at
`/var/tmp/datalegs-20261025/rev7data/full/refit/production-f16.bin`, SHA-256
`9fe1aa47a82730550ed1b9dd73addeacf31304485a12213c4e3e9f151bff5117`.
The selected dense final-epoch bake is `62c844a22b49585320ef8b8b74c2956816738511f4b9f6d6ff7781230d288b69`;
the packed output uses the inherited TRAIN calibration and identity-knot tail.
`certify-monotone` reports `certified=true`, `monotone=true`, zero free ramps
and zero free amplitude. The full fit ran with one trainer thread:
`run-heavy: done rc=0 1735s | peak-RSS 2.06GiB | min-avail 43154MiB | peak-load 5.77`.
`CANDIDATE_FREEZE.json` binds this candidate, M5, M7 and the new root's
`wide/frozen.json` (`e127eef91efce57c6bb980c3c6e047e0a49079d57f337fc122be74e70cbef1ad`)
before this lane's held-out scoring. The prior coordinator EVAL results and
published documentation examples were already exposed; no result selected
the new weights or epoch. The documentation-example disclosure in
[`DATA_SPLITS.md`](../docs/DATA_SPLITS.md) still applies.

Label reads are retained at
`/var/tmp/datalegs-20261025/rev7data/LABEL_READS.json`: original TRAIN row
indices, selected fields, every tokenized field, and repeated reads from
preserved failed assembly attempts. It also records derived TRAIN pair-count
reads. EVAL reads use separate ledgers in `eval/` (a checksum-verified copy
of the coordinator's input) and `eval-bapps/`. The source root remains
`/mnt/v/output/zensim/byconstruction-2026-10-22/rev7raw/root`; the new dev-local
path `/mnt/v/output/zensim/byconstruction-2026-10-22/rev7data` resolves to
`/var/tmp/datalegs-20261025/rev7data`. Exact native pixels and confirmed D4
aliases define the family audit's bounded coverage; it does not prove perfect
crop recall. The inherited development mask removes another 120 rows.

The existing dataset evaluator now delegates MOS SROCC, midranks and
reference-bootstrap intervals to the canonical Rust statistics owner.
Its earlier `argsort(argsort(...))` tie handling assigned distinct ranks to
ties, so this run recomputes M7, B7 and all peers with the same corrected
method. MOS intervals use 2,000 reference resamples, seed 1, type 7 quantiles;
pairwise evaluation retains the original soft-probability agreement and
reference bootstrap, including `.5` comparisons. BAPPS's 36,344 frozen EVAL
comparisons extend the coordinator's 11-collection assessment using its
existing feature bank for models only. The owner explicitly prohibited
unpacking BAPPS again; its peer scoring waits for the IMI archive route.

The 12 E21 cells use four one-thread fits inside each bounded seed job, with
the shared heavy lock released between seeds. The rebuilt trainer reproduces the unchanged-root Mr7 KADID seed 0
predictions byte for byte.
The owner superseded the JPEG XL part of this run before calibration:
`lslocks` showed the queued calibration as `WRITE*` (not acquired), and its
PID was terminated. No calibration, encode, judge or analysis step ran.
The preparation build and candidate freeze remain preserved. The already
running control fit was stopped at epoch 85 to put EVAL first; its partial
outputs and scratch were renamed with `.interrupted-for-eval.bak` suffixes.
Its fresh full-budget restart completed after feature-only BAPPS scoring.

The copied eleven-collection assessment added only Mdata model predictions;
M7, B7 and all three peer TSVs retain their copied hashes. MOS uses signed
SROCC; FG-IQA and PieAPP use expected2AFC agreement under native `p_a`.
Reference-bootstrap95% intervals and within-reference MOS statistics are in
`/var/tmp/datalegs-20261025/eval/report.{json,md}`. The point estimates are:

| EVAL collection | Mdata | M7 | B7 | SSIMULACRA2 | Butteraugli | CVVDP |
|---|---:|---:|---:|---:|---:|---:|
| cid-iq | 0.870 | 0.861 | 0.871 | 0.757 | 0.592 | 0.749 |
| cid-oa | 0.893 | 0.914 | 0.882 | 0.847 | 0.577 | 0.912 |
| colourlab-rating-variability-2022 | 0.722 | 0.723 | 0.701 | 0.604 | 0.373 | 0.612 |
| fg-iqa | 0.758 | 0.757 | 0.758 | 0.758 | 0.749 | 0.755 |
| iqa-odi | 0.841 | 0.843 | 0.849 | 0.854 | 0.806 | 0.833 |
| irccyn-ivc | 0.934 | 0.935 | 0.960 | 0.949 | 0.912 | 0.928 |
| live-md | 0.892 | 0.891 | 0.879 | 0.913 | 0.827 | 0.896 |
| mdid | 0.902 | 0.894 | 0.892 | 0.787 | 0.572 | 0.829 |
| nits-iqa | 0.727 | 0.717 | 0.704 | 0.777 | 0.706 | 0.721 |
| pieapp | 0.586 | 0.582 | 0.579 | 0.560 | 0.535 | 0.570 |
| vcl-fer | 0.945 | 0.938 | 0.884 | 0.945 | 0.877 | 0.933 |

LIVE-MD peers cover all 180 rows after the coordinator’s `7bd403d2`
`pair_key` correction; distorted paths alone had retained only 162 rows.
All nine input hashes match the corrected coordinator report; the prior
report is preserved in `eval/prior-report-before-pair-key-fix/`. Model
values are unchanged. Corrected within-reference SROCC for SSIMULACRA2,
Butteraugli and CVVDP is 0.961, 0.954 and 0.960.

The changes are mixed: MDID, NITS, VCL-FER and PieAPP improve in point
estimate over M7, while CID:OA falls. The coordinator’s paired MOS report
(`fb25f4a2`, copied to `eval/paired-Mdata/`) resamples whole references for
both arms. Except IQA-ODI (42), these cohorts have 3–12 references, so the
intervals are descriptive. MDID’s paired difference is +0.0082
[+0.0011,+0.0179], NITS +0.0106 [+0.0049,+0.0192], and CID:OA −0.0209
[−0.0351,+0.0000]. No comparison selects model weights or epoch.

BAPPS model-only EVAL completes all 36,344 comparisons from the existing
feature bank, without image extraction or peer rescoring:

| Model | Soft2AFC agreement | Reference-bootstrap95% interval |
|---|---:|---:|
| Mdata | 0.673924 | [0.671068, 0.677020] |
| M7 | 0.661796 | [0.658769, 0.664932] |
| B7 | 0.639217 | [0.636058, 0.642483] |

The BAPPS peer columns are unmeasured in this run and wait for the owner’s
IMI archive route. This result does not change the registered training mix.

The unchanged-root KADID seed 0 control reproduces all 7,869 Mr7 predictions
byte for byte (`e03397f6b7e769708d9a299366f90c19f708fedcb37b3ec38c515c4e2244c9f6`).
The native bake hashes differ because their `zentrain.repro` invocation
provenance differs. Removing only that entry through `bake_dial_refit strip`
produces identical 22,230-byte files; weights, scaler and all remaining
metadata reproduce. `CONTROL_REPRODUCTION.json` binds both original hashes
and the stripped comparison. All 12 Mdata cells use the same fixed budget.
`NATIVE_CELL_CONTROLS.json` verifies all 24 Mdata/Mr7 cell identities,
epochs, model hashes, prediction hashes and singular dense bakes.
The label ledger also records the 13 completed native fits against all
11 derived TRAIN legs, reconstructed after fitting from their bound
commands and authorities without reinterpreting target values.

All eleven legs deliver 350,396 derived TRAIN rows from 223,275 native
memberships. Informative pairs below count the possible within-group rank
comparisons after exact target ties, rather than independent human votes.
Native reference groups stay separate across LIVE-MD study parts and IVC
reference renditions; source families can therefore be fewer than groups.
For pairwise legs each non-tied comparison is its own two-row group.

| TRAIN leg | Memberships | Derived rows | Groups | Families | Informative pairs | Dropped p_a=.5 |
|---|---:|---:|---:|---:|---:|---:|
| pipal | 23,200 | 23,200 | 200 | 200 | 1,334,000 | 0 |
| mdid | 800 | 800 | 10 | 10 | 31,599 | 0 |
| live-md | 210 | 210 | 14 | 7 | 1,469 | 0 |
| cid-iq | 330 | 330 | 11 | 11 | 4,737 | 0 |
| cid-oa | 143 | 143 | 11 | 11 | 847 | 0 |
| vcl-fer | 264 | 264 | 11 | 11 | 3,036 | 0 |
| irccyn-ivc | 105 | 105 | 7 | 5 | 905 | 0 |
| colourlab-rating-variability-2022 | 320 | 320 | 5 | 5 | 9,931 | 0 |
| bapps | 146,003 | 223,068 | 111,534 | 111,363 | 111,534 | 34,469 |
| pieapp | 51,000 | 100,202 | 50,101 | 140 | 50,101 | 899 |
| fg-iqa | 900 | 1,754 | 877 | 50 | 877 | 23 |

Exact MOS target ties remove 1 MDID, 1 LIVE-MD, 48 CID:IQ, 11 CID:OA, 5 IVC
and 149 Colourlab candidate pairs; no singleton rows drop. Counts and
acceptance-corrected weights are in `DATASET_LEG_COUNTS.json`; its derived
TRAIN target reads are appended to the same label-read ledger.

#### E21: all 12 cells, final epoch 119

| Held-out dataset | Mdata mean SROCC | Mr7 mean SROCC | Mdata − Mr7 |
|---|---:|---:|---:|
| kadid | 0.875736 | 0.879282 | -0.003547 |
| tid2013 | 0.872804 | 0.873635 | -0.000832 |
| konfig | 0.800243 | 0.800749 | -0.000506 |
| cid22_a25 | 0.878678 | 0.884600 | -0.005922 |

The equal-cohort paired mean is **−0.002701497**, seed-unit SE
**0.002163089**, over three seeds. Per-seed differences are −0.001699541,
−0.006847188 and +0.000442237. Against the inherited `base-c` control,
Mdata is −0.015375027 (SE 0.003000262), Mr7 −0.012673530
(SE 0.001565142); both registered E33 point-form flags are false.
This comparison includes the added data and the required inherited family
mask. Seed-unit uncertainty does not establish uncertainty over new cohorts.

The per-type owner covers 25 KADID, 24 TID2013, seven KonFiG and eight
CID22-A codec types. The table averages within-reference per-type SROCC
changes against Mr7; bias SD is the SD of per-type mean pooled rank bias.
Knob inversion rates use only pairs where human ordering and the knob
agree. Ladder RMS is a fraction of the human p1–p99 span. All are
three-seed summaries.

| Dataset | Per-type within-ref Δ | Bias SD Mdata / Mr7 | Knob inversion % Mdata / Mr7 | Ladder RMS Mdata / Mr7 |
|---|---:|---:|---:|---:|
| kadid | -0.000376 | 0.080899 / 0.079641 | 1.4910 / 1.3629 | 0.089059 / 0.088447 |
| tid2013 | -0.002389 | 0.084836 / 0.085366 | 1.0141 / 0.9115 | 0.059200 / 0.058431 |
| konfig | -0.000095 | 0.087935 / 0.090295 | 0.0194 / 0.0065 | 0.131721 / 0.131885 |
| cid22_a25 | -0.000030 | 0.036431 / 0.035638 | 0.0200 / 0.0100 | 0.070738 / 0.070090 |

The full panel, including PLCC, KROCC, outliers, PWRC, Z-RMSE and scatter
geometry, is preserved in `scorecard/panel.{json,txt}`. No AIC population
is read by this scorecard (`--no-aic`).

#### Existing label-free mixture tests

| Test | Compositions per cell | Model | Inversions per cell | Total across 12 cells | Largest score rise |
|---|---:|---|---:|---:|---:|
| mixture | 1,152 | Mdata | 2–5 | 38 | +7.08 |
| mixture | 1,152 | Mr7 | 1–14 | 57 | +6.28 |
| mixture_global | 1,728 | Mdata | 0–0 | 0 | -0.89 |
| mixture_global | 1,728 | Mr7 | 0–0 | 0 | -1.12 |

The same 48-reference manifests and copied Mr7 score files are used; all
24 model/seed score files per test bind the correct dense bake hashes, raw
ring form and XYB pyramid. Standard-mixture inversion counts fall, while
the largest rise grows. Global mixtures have zero inversions for both arms.

RGB MSE rises in all 1,152 standard compositions and 1,727 of 1,728 global
compositions. For origin 7076, `contrast1.15` followed by `luma4`, MSE
changes from 84.20919799804688 to 83.93456013997395. The registered tests
retain every composition, including this exception; no count or expectation
is relaxed. `MIXTURE_MANIFEST_CHECK.json` pins both manifests and this row.

#### Full-data label-free gates

The coordinator ran the frozen `9fe1aa47…` candidate through the same
revision-7 instruments and empty human-corpus root as full-data Mr7, using
`688746b78bee-gates` (`bake_verdict` at 688746b7; serving/steering/timing
tools from 9b1f6dca). This lane copied the text outputs with SHA-256 checks,
verified the model hash and checked all 744 identity rows.

| Check | Mdata | Full-data Mr7 |
|---|---|---|
| NEARID N1–N3 | pass; 143 monotone ladders | pass; 143 |
| Standard dial grid, C1–C6 | PASS; C1 0.9972, C3 0.819 | PASS; C1 0.9984, C3 0.793 |
| Ladder dial grid, C1–C6 | PASS; C1 0.9914 | PASS; C1 0.9935 |
| G-STEER, exact-split map | 134/135 | 133/135 |
| Identity, 12 SIMD permutations | 744/744 equal read-set bits; no bad fragility | 744/744 equal |

These are label-free checks; the human A-series verdicts are not measured
by this gate run. The E21 point-form result above remains false. The source
is the canonical `gates-Mdata/` output and `gates-Mdata.summary.json` under
`/mnt/v/output/zensim/byconstruction-2026-10-22/`; this study preserves the
text copy, command and source tool hashes in `gates-Mdata/`. JPEG XL loop
work remains canceled, and BAPPS peer scores wait for the IMI route.

Large private artifacts and hashes: [Mdata artifact pointer](mdata_2026-10-25.pointer.md).

The closed artifact snapshot is SHA-256 verified at
`/mnt/tower/output/zensim-datalegs-archive-2026-10-25/` (checked through
the mounted NAS view on dev). It preserves roots, models, all cell records
and predictions, ledgers, copied EVAL/gate outputs, configuration and
source-cut hashes. All inventory checks and three random readbacks pass;
receipt `d91ab3c54aae6b06f011e43b0b9210d86d0971f51d66dde5a93b948349273117`.
The [artifact pointer](mdata_2026-10-25.pointer.md) binds the inventory hash
and local/NAS paths. No R2 mirror was created. The preservation wrapper
reported `run-heavy: done rc=0 229s | peak-RSS 0.02GiB | min-avail 43986MiB | peak-load 2.11`.


## New dataset training mixes — preregistered 2026-10-25

The original five-arm registration below is extended by the owner addendum.

### Owner licence addendum — 2026-10-25

The owner's later addendum adds **Mlic**: the same M shape, nominal 0.5,
seeds 0–2, with PIPAL, MDID, LIVE-MD, CID:IQ, CID:OA, IRCCyN/IVC,
Colourlab rating variability and BAPPS. VCL@FER, PieAPP and FG-IQA are
validate-only under the owner's October 25 policy. They remain usable as
out-of-fit validation populations, including their frozen TRAIN memberships.

The original five arms remain registered research comparisons. All five
include at least one validate-only leg and are ineligible as commercial
defaults under this new-leg policy; the separately owned Mdata64 also uses
those legs. Mlic excludes all three and is the eligible new-leg arm.
BAPPS is retained and reported as tier C with two judgments per TRAIN pair.
No per-item confidence weights or binomial vote loss are introduced here;
those follow-ups are owned separately.

The unchanged rule below now ranks six arms for at most two research
finalists. Eligibility is reported separately and does not replace the
preregistered scores or tie-breaks. Mlic receives the same twelve E21 cells
and nine group holdouts; its pairwise holdout scores all three pair datasets,
and its seven-MOS holdout also scores VCL@FER, which it never fits. This
raises the requested inventory to 123 cells: 111 new fits and 12 reused
Mdata E21 cells. Both mixtures cover every cell, and the native hard-pair
point control covers 54 model/collection comparisons.

This extension is registered before Mlic assembly or measurements. The
Mdata baseline and twelve Mw025 cells had completed when the addendum was
implemented; no Mlic result informed it. The frozen confidence manifest is
copied with its SHA256SUMS entry, with the licence classifications recorded
in `LICENCE_POLICY.json`. Root source-label and per-cell derived-label
ledgers continue to precede TRAIN target reads.


This follow-up compares M-shape Mw025 (all eleven legs, nominal 0.25),
Mdata (all eleven, 0.5), Mw100 (all eleven, 1.0), Mmos (eight MOS legs,
0.5), and Mpair (three pairwise legs, 0.5), seeds 0–2. Existing Mdata E21
cells are reused. Full-data seeds and shape experiments owned elsewhere
are not duplicated. All roots retain the same frozen inputs, source-family
reservations and fixed final epoch 119, with separate label-read ledgers.

The following selection rule is written before any new mix results are read.
For each seed, the old-data score is the equal mean of the four E21 SROCCs,
rescaled to [0,1] as (rho+1)/2. The new-data score is the equal mean over
eleven collections: within-reference SROCC, rescaled the same way, for each
of the eight MOS sets; hard-pair agreement with half credit for predicted
ties for each of BAPPS, PieAPP and FG-IQA. Holdouts omit PIPAL, all three
pairwise legs, or the other seven MOS legs. A subset arm already excludes
some of these groups: its full-data fit is an out-of-group model for those
groups, reused rather than fitting identical no-op holdouts. All scoring
labels are TRAIN. Collection means give each set equal mass regardless of
its number of rows or references. References with fewer than two rows or
constant targets/predictions are reported and excluded from the SROCC mean;
a collection with no defined correlations blocks selection.

The balanced score is the equal mean of the old-data and new-data scores.
Select at most two distinct finalists: the largest three-seed mean balanced
score, and the largest three-seed mean old-data score. Ties break by larger
new-data score, then lexicographic arm name. Report paired seed-unit SEs,
the complete old/new tradeoff, type-bias SD, knob-concurring inversions,
Z-RMSE, error p99, rank geometry and both mixture tests for every arm.
These diagnostics remain visible; they do not introduce post-hoc selection
thresholds. This exploration cannot establish product qualification.

After freezing finalist names, inputs and model hashes, evaluate seed 0 of
each finalist once against the coordinator's frozen EVAL copy. If Mdata is
a finalist, reuse its already-frozen seed-0 predictions, with no fresh
assessment or candidate tuning. EVAL cannot change the finalists, weights,
shape, seed or checkpoint. No TEST/calibration labels are read. New-data
group holdouts assess omitted dataset labels; they are not claimed to prove
source independence between different TRAIN collections.


### Mix implementation and assembly controls

Implementation 348407450b0d is verified on origin/main. Native admission and
ranking controls, strict workspace/all-target/all-feature Clippy, script lint
and the release trainer build pass. Source build cut 0572d08b8d0e precedes a
rebase containing only the coordinator's dashboard update; all captured
training-source hashes still match the landed files.

The pinned pre-parameterization builder (337382e4) and the new builder's
0.5/all-collections defaults rebuilt at the same fresh output path. Every
root file is byte-identical (`diff -qr` is empty); both frozen-file hashes are
`743d20148bbd265c829490621ed45a74bcde0842fdc47cc2b3f1f67b2e0d0115`.
The original Mdata root remains unchanged at `e127eef91efc…`.
The new source-label ledgers have 11, 11, 8 and 3 entries for Mw025, Mw100,
Mmos and Mpair respectively; per-cell ledgers separately authorize derived
TRAIN reads. Mdata reuses its earlier ledger and cells.

| Arm | Collections | Nominal per collection | Total nominal new mass | Derived TRAIN rows | Frozen root SHA-256 |
|---|---:|---:|---:|---:|---|
| Mw025 | 11 | 0.25 | 2.75 | 350,396 | `d02b01d290feb8a9b5b107aed0941ffa3eb864d2e3b91ba36333886976db382e` |
| Mdata | 11 | 0.5 | 5.5 | 350,396 | `e127eef91efce57c6bb980c3c6e047e0a49079d57f337fc122be74e70cbef1ad` |
| Mw100 | 11 | 1 | 11 | 350,396 | `28ab159f4be7b5ab614055d1f6a2070e310edecc571a63cd800f0b8430b772e8` |
| Mmos | 8 | 0.5 | 4 | 25,372 | `322ea0f8518ad004d6314557cbc7fcab7339269a08b01821db3cf2aec487abcd` |
| Mpair | 3 | 0.5 | 1.5 | 325,024 | `c538fd925c2d84808998f737a0e020151c71029455b5b91f44a352634af61f03` |

All five roots reserve the same 40,904 known families and report zero
remaining inherited reservations within the existing exact-pixel/D4 guard.
The bounded family screen does not prove perfect crop recall.
Preparation resource record: `run-heavy: done rc=0 272s | peak-RSS 10.84GiB |
min-avail 36184MiB | peak-load 11.19` (16 GiB scope).
The fit queue uses up to twelve one-thread cells under a 36 GiB scope,
matching the completed shape-control schedule; large new-data predictions
are serialized. Rescheduling the initial six-cell queue happened while
`lslocks` still marked it `WRITE*`; no fit was canceled.

Artifacts checked on dev:
`/mnt/v/output/zensim/byconstruction-2026-10-22/data-mixes` points to
`/var/tmp/byc-mixes-20261025`. The source cut, patch, binary hashes, root
inventory, default control and ledgers are retained there. At this assembly
record, mix fits and finalist EVAL assessment are not yet complete.


A pre-fit receipt check also confirms that each variant retains all 18
inherited payload digests unchanged, and that every included dataset table
has Mdata's payload digest (11, 11, 8, 3 tables for Mw025, Mw100, Mmos, Mpair).
Native admission verifies the actual payload hashes before fitting; the
end-of-run audit repeats the byte checks. The local control receipt is
`ROOT_ROW_EQUIVALENCE.json`. Columns C are now copied into the study before
fitting and bound with `BYC_COLS`; their SHA-256 is
`e662c7225c7f6aca27bf208161b0d1b99f6d2ba7be427dfa37c4ac4ec1c49c75`.


### Mdata held-out-new-TRAIN baseline

All nine cells completed at epoch 119; predictions and group statistics are
saved in each `byc_record.json`. Every MOS reference correlation is defined.
Each seed trains one model without PIPAL, one without the three pairwise
sets, and one without the other seven MOS sets. All retain the older human
TRAIN groups. Pair scores
below use hard preferences; EVAL retains the original probabilities.

| TRAIN collection | Statistic | Seed 0 | Seed 1 | Seed 2 |
|---|---|---:|---:|---:|
| bapps | hard-pair-agreement | 0.773495 | 0.752766 | 0.771227 |
| cid-iq | within-reference-srocc | 0.902953 | 0.902042 | 0.912140 |
| cid-oa | within-reference-srocc | 0.986498 | 0.986498 | 0.986498 |
| colourlab-rating-variability-2022 | within-reference-srocc | 0.764149 | 0.802475 | 0.761858 |
| fg-iqa | hard-pair-agreement | 0.898518 | 0.897377 | 0.899658 |
| irccyn-ivc | within-reference-srocc | 0.946764 | 0.942434 | 0.948892 |
| live-md | within-reference-srocc | 0.967918 | 0.955419 | 0.971745 |
| mdid | within-reference-srocc | 0.950206 | 0.923020 | 0.950881 |
| pieapp | hard-pair-agreement | 0.675685 | 0.669498 | 0.670456 |
| pipal | within-reference-srocc | 0.648894 | 0.698702 | 0.645379 |
| vcl-fer | within-reference-srocc | 0.942767 | 0.922134 | 0.947431 |

Resource record: `run-heavy: done rc=0 2040s | peak-RSS 3.43GiB |
min-avail 29536MiB | peak-load 11.16` (36 GiB batch scope; RSS is the
wrapper's process maximum, not a sum across the concurrent fits).
At this baseline record the four other arms remain queued; no finalist or
new EVAL assessment exists. The preregistered rule is unchanged.

### Mlic assembly and first mix-arm completion

Mlic's eight-leg root is frozen at
`a7fd4af0c2bb5a8981be636e5c1cb1a51db57496b7c60d7bf6da01d71dd16952`:
248,176 derived TRAIN rows, total nominal new mass 4, 40,904 known reserved
families and zero remaining inherited reservations. Every included dataset
payload and all 18 inherited tables have Mdata's actual byte digest. Eight
source-label ledger entries precede the reads. The copied confidence
manifest has SHA-256
`48ab25bde4fc5fc573b8de8886b34e4318d86de1695065bef3cac9e468bf4218`.
BAPPS contributes 111,534 hard TRAIN pairs (223,068 rows) after excluding
34,469 exact probability ties from 146,003 native memberships; it remains
tier C, with two judgments per original TRAIN pair. Its nominal collection
weight stays 0.5, without confidence weighting or binomial vote loss.

The Mlic extension, policy-copy and finalist-input controls are on main in
c2a1fcdb8f93, f3567339e4a1 and 51ff84985962. The dataset fixture controls,
legacy authority controls, undefined-name check and script lint pass. The
native trainer binary is reused unchanged. Existing family guards remain
bounded to their recorded exact-pixel/D4 coverage.

Mw025 completed all 21 requested cells at epoch 119, without retries.
Resource record: `run-heavy: done rc=0 4081s | peak-RSS 3.43GiB |
min-avail 19312MiB | peak-load 14.98` (36 GiB scope, up to twelve one-thread
fits). Mlic then acquired the shared lock for assembly and its 21 cells.
At this record, 42 of the expanded 123 cells are complete; no six-arm
finalist selection or new EVAL assessment has run.

### Fleet ownership for the remaining mix arms

The owner assigned Mw100 (21 cells), Mmos (21) and Mpair (18) exclusively
to the fleet-cells lane: 60 cells, seeds 0–2, with the registered folds.
All 21 submitted Mlic cells remain local (twelve fitting and nine queued
inside the active executor at the ownership agreement). The completed
Mdata and Mw025 cells stay unchanged. No fleet-owned cell had started
locally: the waiting Mw100 flock was verified WRITE* and canceled before
lock acquisition, which exited the local queue before Mmos/Mpair launch.

Fleet scripts are pinned to b7cb1a224dc7a4835e7259fe41c447ffa4c811e4,
with the six exact study binaries and frozen columns C. All binary hashes
were rechecked at the agreement. The Mw025/without_kadid_s0 control's
prediction SHA-256 is
`9333a13dff53fa4200166e723a9c33f0a48cf32e6beacd40856163571d536eff`.
Fleet arms remain gated on prediction parity and normalized-model parity;
the prediction digest above is the local reference, not a claimed fleet
pass. `FLEET_OWNERSHIP.json` records every assigned cell and the source pin.
Verified complete cell directories are harvested into the same study tree;
the consolidated 123-cell audit and TRAIN-only scorecard remain this study's
acceptance path. Local and fleet jobsets do not overlap.

The scorecard also binds each cell's recorded trainer hash to the six frozen
study binaries before reading scoring targets. A partial metadata audit
verified 54 completed cells: 42 use the current pinned trainer, and the
twelve reused Mdata E21 cells retain their recorded earlier trainer. Two
negative controls refuse a changed trainer hash and incomplete coverage;
neither writes a successful receipt. The complete audit requires all 123
records and exact agreement with the native cell inventory. Fitting scripts
remain pinned to b7cb1a224dc7; this check only extends the scorecard recipe.

### Mlic cell completion

Mlic completed all 21 cells at fixed epoch 119 without retries, using the
same pinned trainer as the new Mdata holdouts and Mw025. Their recorded
budgets are all 120 epochs and 50,000 pairs per epoch. This brings local
coverage to 63 of the 123 requested cells; the remaining 60 are exclusively
fleet-owned. No six-arm selection or new finalist EVAL assessment has run.

Resource record: `run-heavy: done rc=0 4086s | peak-RSS 10.86GiB |
min-avail 23279MiB | peak-load 15.30` (36 GiB scope). This wrapper covers
Mlic root assembly and the 21-cell batch; its RSS is the maximum single
process, not an aggregate or a fit-only bound. The completion receipt is
`MLIC_COMPLETION.json`. After the wrapper exited, the queued decoder
acquired the shared heavy lock and fleet image/data preparation remained
queued behind it. No new local heavy job was submitted at release.

The completed local arms provide the following held-out-new-TRAIN means.
Each seed averages the same eleven collection statistics under the
registered transformation; differences pair the three seeds against Mdata.
These are record aggregates, with no new target table read or finalist
selection. The complete scorecard still awaits the fleet arms.

| Arm | New-data mean | Seed-unit SE | Difference from Mdata | Paired seed-unit SE |
|---|---:|---:|---:|---:|
| Mdata | 0.899773554 | 0.000523623 | 0 | 0 |
| Mw025 | 0.899266963 | 0.000625827 | -0.000506591 | 0.000109575 |
| Mlic | 0.899509341 | 0.000905100 | -0.000264213 | 0.000484277 |

`LOCAL_TRAIN_GROUP_RESULTS.json` retains all per-seed collection values and
their source record hashes. The queued decoder subsequently completed
successfully in one second; fleet image/data preparation acquired the lock.

The existing E21 owner also scored these three local arms while the fleet
control replay ran remotely. The Mdata baseline is unchanged, including its
per-seed/per-dataset values. All target reads have the existing TRAIN ledger.

| Arm | E21 mean signed SROCC | Paired difference from Mdata | Seed-unit SE of difference |
|---|---:|---:|---:|
| Mdata | 0.856865163 | 0 | 0 |
| Mw025 | 0.861762900 | +0.004897736 | 0.001756875 |
| Mlic | 0.861284769 | +0.004419605 | 0.000249599 |

| Arm | KADID | TID2013 | Konfig | CID22 A25 |
|---|---:|---:|---:|---:|
| Mdata | 0.875735715 | 0.872803620 | 0.800242948 | 0.878678370 |
| Mw025 | 0.879206151 | 0.875520167 | 0.809619830 | 0.882705450 |
| Mlic | 0.882458850 | 0.874057075 | 0.807786333 | 0.880836816 |

The local E21 receipt retains each per-set paired seed-unit SE and all
three seed differences. Resource record: `run-heavy: done rc=0 1s |
peak-RSS 0.24GiB | min-avail 45817MiB | peak-load 1.68` (4 GiB scope).
This interim comparison does not select finalists or assess new EVAL rows.
R2's image/control-input preparation receipt is copied into the study;
its complete scientific-arm archives and numeric control remain separately
owned by the fleet lane.

The mixture score owner now verifies a cache hit against the actual bake
digest, postprocessing/ring/pyramid settings, complete manifest image keys
and finite scores before reuse. An original Mdata cache passes; changed
bake, settings, image coverage and nonfinite-score controls all refuse and
preserve their input. Evidence is in `mixture-cache-controls/`. The mixture
manifest and every comparison remain unchanged, including the previously
recorded global MSE exception. Completed local models can be scored while
fleet fits run; the complete scorecard reuses only verified cache entries
and still requires all 123 models in both mixture sets.

The first local mixture pass found wrapper-log files matching the cell
glob. The owner now selects directories explicitly; all 21 directories in
each completed local arm remain selected. Its successful cached score is
preserved, along with the failed-pass log. Reporting moved into a Just
recipe so the resource wrapper cannot expand a shell report-name variable.

### Result archive transport

The shared mounted-filesystem mirror from dc79d8dbacd5 remains the owner.
This study adds optional cold readback: fsync copied files and request
Linux page-cache eviction before client SHA-256 verification. The existing
mounted-transport, corruption and artifact-kind tests pass unchanged.
The final archive will use this mode, without starting remote processes.

An initial experimental transport control and its source patch are retained
under `mounted-mirror-control*` and `MOUNTED_MIRROR_INITIAL_CONTROL_SOURCE.patch`.
The consolidated CLI then mirrored the same three actual result files
(20,660 bytes), verified all artifact digests plus its receipt and performed
three additional readbacks. An independent coreutils inventory check passes.
Its trace contains five fsync calls, sixteen cache-advice calls, and no
Docker or SSH invocation. A cold-readback corruption control refuses before
creating a verification receipt. Evidence is in
`mounted-mirror-canonical-control*` and `MOUNTED_MIRROR_CANONICAL_CONTROLS.json`.
The final study archive still awaits the fleet cells and completed scorecard.


### Completed local-arm mixtures and diagnostics

Both mixture sets now cover all 63 completed local models, with each
score cache bound to its actual bake and complete comparison inventory.
The primary comparison below uses the twelve older-data E21 models per
arm; each model retains 1,152 standard and 1,728 global comparisons.

| Arm | Standard inversions, total | Range per E21 model | Largest standard rise | Global inversions, total | Largest global rise |
|---|---:|---:|---:|---:|---:|
| Mdata | 38 | 2–5 | +7.082146 | 0 | -0.889359 |
| Mw025 | 43 | 1–6 | +7.173730 | 0 | -1.097992 |
| Mlic | 41 | 3–5 | +6.172589 | 0 | -0.790390 |

The global MSE exception remains included; no comparisons or thresholds
changed. Raw results for all 63 models are retained separately from the
E21 summaries. Resource record: `run-heavy: done rc=0 78s | peak-RSS
0.04GiB | min-avail 44326MiB | peak-load 10.86` (12 GiB scope, jobs 12;
RSS is the maximum single process, not an aggregate).

The following diagnostic rows are three-seed means from dataset-held-out
TRAIN predictions. They exclude the panel's in-sample rows. The complete
per-type and rank-geometry outputs, including tails, coverage, clumping and
clamping, remain in `local-diagnostics/`; these diagnostics do not gate the
registered selection.

| Arm | Dataset | Type-bias SD | Knob inversion rate | Ladder RMS | Z-RMSE | Rank error p99 | Fitted error p99 |
|---|---|---:|---:|---:|---:|---:|---:|
| Mdata | kadid | 0.080899 | 0.014910 | 0.089059 | 0.484136 | 0.408663 | 0.379704 |
| Mw025 | kadid | 0.079287 | 0.013581 | 0.088365 | 0.477914 | 0.407721 | 0.381542 |
| Mlic | kadid | 0.077532 | 0.013097 | 0.087282 | 0.472303 | 0.401431 | 0.377887 |
| Mdata | tid2013 | 0.084836 | 0.010141 | 0.059200 | 0.444892 | 0.296727 | 0.277738 |
| Mw025 | tid2013 | 0.082506 | 0.010081 | 0.059223 | 0.441296 | 0.298891 | 0.276827 |
| Mlic | tid2013 | 0.084821 | 0.010141 | 0.058626 | 0.442287 | 0.298518 | 0.276314 |
| Mdata | konfig | 0.087935 | 0.000194 | 0.131721 | 0.591952 | 0.661607 | 0.574378 |
| Mw025 | konfig | 0.084933 | 0.000194 | 0.128165 | 0.577721 | 0.638393 | 0.569735 |
| Mlic | konfig | 0.084633 | 0.000129 | 0.128728 | 0.582080 | 0.654167 | 0.565668 |
| Mdata | cid22_a25 | 0.036431 | 0.000200 | 0.070738 | 0.493620 | 0.402445 | 0.377916 |
| Mw025 | cid22_a25 | 0.032990 | 0.000100 | 0.069846 | 0.486263 | 0.398364 | 0.372084 |
| Mlic | cid22_a25 | 0.038681 | 0.000167 | 0.070297 | 0.489250 | 0.395900 | 0.378415 |

Mlic lowers Z-RMSE on all four sets relative to Mdata, with mixed tails
and type bias: CID22 A25 type-bias SD increases from 0.036431 to 0.038681,
and TID2013 rank error p99 increases from 0.296727 to 0.298518. No finalist
is named from this partial study. Resource record: `run-heavy: done rc=0
47s | peak-RSS 0.32GiB | min-avail 44849MiB | peak-load 1.91` (16 GiB
scope, jobs 1). TRAIN label-read ledgers precede the diagnostic reads.


### Exact fleet control before scientific cells

The AMD replay of frozen Mw025/without_kadid_s0 completed at epoch 119.
All 23 registered control checks pass. The prediction file is byte-identical
to the completed local cell, SHA-256
`9333a13dff53fa4200166e723a9c33f0a48cf32e6beacd40856163571d536eff`.
The study independently checked the returned prediction bytes and retained
refit/dense file digests, then stripped only `zentrain.repro` with its pinned
native binary. Both fresh stripped checkpoints are byte-identical, SHA-256
`854f75f3a6ec08581faf6b54e88afdffa8a47ebe4a6574778baa565c49fa3e6b`.
Original local and returned bakes remain preserved. The extra Intel control
was still running at this observation; it is not claimed to have passed.

The fleet lane can now execute its exclusively assigned 60 scientific
cells. The independent verification and copied source receipts are under
`fleet-coordination/`; no scientific result has yet been harvested into the
123-cell scorecard at this control record. The registered TRAIN selection
and EVAL freeze order remain unchanged. Every candidate cell/full-data bake
will be retained for later AIC2026 evaluation; new full-data candidates will
be added to `candidate_registry.py` and its verified private mirror.

### The licence-clean finalist's shape: rule registered before any Mlic64 result (coordinator, 2026-10-26)

Under the owner's licence policy, Mlic is the only eligible arm of the six; the six-arm selection above stays a
research comparison and is not changed by this section. The shippable candidate is Mlic in one of two shapes: M's
(`mono:thr:mp32`, as trained in the study) or M64's (`mono:thr:mp64`), which gained +0.0046 (seed-unit SE 0.0017) on
the Mdata data. Runs, all on the Mlic root (`MLIC_ROOT.json`, frozen sha256 `a7fd4af0…`) with the study's binaries
(`BIN_SHA256SUMS`): Mlic full-data seeds 0–2, packed and certified monotone (dev); Mlic64's 12 E21 cells and three
full-data seeds (dev); Mlic64's nine held-out-new-data cells (fleet, after its program accepts `mp64`).

This rule is written at 02:20Z on 2026-10-26, before any Mlic64 cell has finished; no Mlic64 number has been read.

1. **Mlic64 replaces Mlic** if its old-data score (the study's equal mean of the four E21 SROCCs, per seed, paired by
   seed against Mlic) is higher by at least two seed-unit SEs, and its new-data score (the study's eleven-collection
   mean) is not lower by more than two seed-unit SEs. If the nine new-data cells are not in yet, the choice is made on
   old data and reversed if the new-data score later falls by more than two SEs. Otherwise Mlic stays.
2. The chosen model's full-data seed-0 bake must pass every label-free gate run for revision-7 M (NEARID N1–N3, dial
   C1–C6 on both grids, G-STEER, identity at 12 SIMD permutations). A failure removes it and the other shape is tried.
3. After the choice, its three full-data seeds and their mean are read once on the EVAL copies already holding M7,
   Mdata and Mdata64, with paired comparisons against those three. EVAL cannot change the shape, seed or weights.
4. Reported, not deciding: knob-concurring inversions, type-bias SD, Z-RMSE, error p99 and both mixture tests.

An arm adding the six admitted library subsets (`Mlic+sub`) registers its own weights and rule when its root is built.

### Mlic+sub root preparation — preread registration (2026-10-26)

Build only: preserve Mlic’s eight legs and their nominal 0.5 weights. The five additional collections with admitted TRAIN memberships share nominal total mass 0.5: 0.1 each for Cognitive Science Experiment 2, Broken Arrows, DIBR, DWT/DT-CWT and Enrico. DIBR retains three separate protocol legs at 0.1/3 each: ACR, scaled pairwise scores and pair-local hard preferences. Exact probability ties are removed; no comparison crosses protocols or native references. The existing within-reference self-pair correction remains. These repeated protocols are not independent votes. No item-confidence weighting or binomial loss is introduced.

This limits the entire small-data addition to one existing large leg’s nominal mass. Toyama has no admitted TRAIN rows and receives no leg. Replay the hash-bound Cognitive Experiment 2 parent-identifier join and require exact native reference binding. Verify each collection’s A/B tier and train licence policy; reserve all existing and additive held-out families, including excluded memberships. Read only admitted TRAIN targets, with a ledger entry persisted before each read.

This registration precedes all new subset target reads and measurements. Mlic+sub stays outside the original six-arm/123-cell selection. No fit is launched; a later TRAIN-only comparison rule, finalist shape and cell assignment require the original selection first. Root results and exclusions will be appended after assembly.

### Six-arm TRAIN selection completed (2026-10-26)

All 123 registered cells are installed and independently audited: 111 new native fits and 12 reused Mdata E21 cells. Binary controls cover 123/123; both mixture inventories cover every cell, with 1,152 standard and 1,728 global comparisons per model. Diagnostics use TRAIN labels only.

The registered point rule selects Mlic for balanced mean and Mw025 for old-data mean. Their balanced difference is only 0.0000016562213902; this does not establish a clear improvement. Only Mlic is licence-eligible; Mw025 remains research-only.

| Arm | E21 SROCC | New TRAIN mean | Balanced mean | Balanced seed-unit SE |
|---|---:|---:|---:|---:|
| Mdata | 0.856865163 | 0.899773554 | 0.914103068 | 0.000526650 |
| Mw025 | 0.861762900 | 0.899266963 | 0.915074207 | 0.000193175 |
| Mw100 | 0.860211355 | 0.899369940 | 0.914737809 | 0.000558124 |
| Mmos | 0.857982613 | 0.900323762 | 0.914657534 | 0.000542511 |
| Mpair | 0.861435854 | 0.899025803 | 0.914871865 | 0.000544080 |
| Mlic | 0.861284769 | 0.899509341 | 0.915075863 | 0.000742086 |

Balanced mean averages normalized old-data SROCC `(rho+1)/2` with the eleven-collection new-TRAIN mean. Three seeds are the uncertainty units. No EVAL result chose weights, epochs, arms or the rule.

| Arm | Standard inversions, 12 E21 models | Global inversions, 12 E21 models |
|---|---:|---:|
| Mdata | 38 | 0 |
| Mw025 | 43 | 0 |
| Mw100 | 39 | 0 |
| Mmos | 53 | 0 |
| Mpair | 45 | 0 |
| Mlic | 41 | 0 |

Raw reports retain all 123 models separately from these primary twelve-model summaries. Type/bias, knob-order, Z-RMSE, rank geometry and tail diagnostics are complete for all six arms and remain non-gating. The original global-mixture MSE exception remains counted; no expectation changed. Scorecard wrapper: `run-heavy: done rc=0 195s | peak-RSS 0.53GiB | min-avail 43666MiB | peak-load 16.50` (16 GiB/jobs 12).

Mlic seed 0 reuses the coordinator’s certified packed bake `7612c2e46154fe9c0c359bfd02e521013f5717e3bf8e11967f5434150e220bcc`. Its three full candidates are registered and mirrored. Mw025’s required packed full seed 0 is active before the finalists’ one frozen EVAL assessment; no new EVAL values are claimed here.

The prepared [Mlic+sub root](mlic_subsets_2026-10-26.pointer.md) stays outside the registered comparison, with no subset fit.

### Mlic+sub against Mlic: rule registered before any Mlic+sub cell (coordinator, 2026-10-26)

Mlic+sub (root sha256 `89b5f88d…`, subset weights registered in `34a32339`) runs Mlic's 21 cells (the four E21 folds
and the three held-out-new-data groups × seeds 0–2) in M's shape (`mono:thr:mp32`) with the study's binaries, so it
pairs with Mlic cell for cell. This rule is written at 03:06Z on 2026-10-26, before any Mlic+sub cell has started.

1. The subsets' own gain has no TRAIN-only out-of-sample measure in this design (Mlic is the fit without them), so the
   TRAIN comparison is non-inferiority. **Mlic+sub replaces Mlic** as the shippable data recipe if its balanced score
   (the study's formula, paired by seed against Mlic) is not lower by more than one seed-unit SE and its old-data score
   is not lower by more than two seed-unit SEs. Otherwise Mlic stays.
2. Shape is decided on the chosen data: if Mlic+sub is chosen, the registered shape rule above applies to Mlic+sub64
   against Mlic+sub (its 12 E21 cells, then three packed full-data seeds).
3. The chosen model passes every label-free gate on its full-data seed-0 bake, then gets one EVAL read of its three
   seeds and their mean, which includes the subsets' own EVAL rows. If Mlic+sub is chosen, Mlic's EVAL read is
   reported beside it as the reference. EVAL cannot change either choice.
4. Reported, not deciding: per-collection new-data statistics, knob-concurring inversions, type-bias SD, Z-RMSE,
   error p99 and both mixture tests.

**Amendment, 03:09Z 2026-10-26, before any Mlic+sub cell has started.** The data-legs lane noted that Cognitive
Science Experiment 2 reuses the exact Colourlab parent stimuli. In Mlic+sub's mos7 holdout, Colourlab's own leg is
omitted while Cognitive Experiment 2 would stay in the fit, so Mlic+sub's Colourlab score there would share images with
a trained protocol and favour it in the non-inferiority test (Colourlab is one twenty-second of the balanced score; a
shift of about 0.02 in its SROCC moves the balanced score by about one seed-unit SE). Therefore **each held-out-new-data
fold of Mlic+sub also omits every added subset collection that shares a source family with a held-out collection**,
by the study's source-family registry. The linkage list is computed label-free and recorded before those cells run.
Folds with no linked subset (the E21 folds, pipal and pairs, unless the registry says otherwise) are unchanged. Every
other part of the rule stands.
The same omission applies to the four E21 folds: a subset collection that shares a source family with KADID, TID2013,
KonFiG-IQA or CID22-A is left out of that fold's Mlic+sub fit (clarified at 03:10Z, still before any Mlic+sub cell).
Linkage, computed label-free by the data-legs lane (source `1076a19e`, 19 focused tests), recorded at 03:26Z
before any Mlic+sub cell: KADID, TID2013, KonFiG-IQA, CID22-A, PIPAL and pairs have no linked added subset; **mos7 also
omits `colourlab-cognitive-science-2022`** (all ten of its registered parent families overlap Colourlab). The cells
use the derived root `/var/tmp/byc-subsets-20261026/Mlic+sub-linked/root` (frozen sha256 `6abfac53…`), whose frozen
packet binds `datasets/linked-subset-omissions.json` (sha256 `e7772dbc…`); the original root `89b5f88d…` is unchanged.

### Complete mix scorecard and frozen finalists (2026-10-26)

All numbers below come from the completed stored TRAIN diagnostics. Pooled E21 is the equal four-dataset mean, paired by seed against Mdata; SE uses the three seeds. Per-dataset entries are paired differences (seed-unit SE).

| Arm | E21 delta vs Mdata | SE | KADID | TID2013 | KonFiG | CID22 A25 |
|---|---:|---:|---:|---:|---:|---:|
| Mw025 | +0.004898 | 0.001757 | +0.003470 (0.004837) | +0.002717 (0.006501) | +0.009377 (0.009935) | +0.004027 (0.001335) |
| Mw100 | +0.003346 | 0.002193 | +0.003700 (0.001622) | +0.000332 (0.003064) | +0.005121 (0.009291) | +0.004233 (0.000588) |
| Mmos | +0.001117 | 0.001652 | +0.006370 (0.004231) | -0.003025 (0.003262) | -0.003340 (0.009980) | +0.004465 (0.001062) |
| Mpair | +0.004571 | 0.000478 | +0.004019 (0.003522) | +0.001996 (0.005482) | +0.009696 (0.003070) | +0.002572 (0.001821) |
| Mlic | +0.004420 | 0.000250 | +0.006723 (0.004402) | +0.001253 (0.000359) | +0.007543 (0.005197) | +0.002158 (0.000290) |

New-data fold entries are mean (seed-unit SE): within-reference SROCC normalized as `(rho+1)/2` for MOS, hard-direction agreement for pairs. Each arm is evaluated on every collection; an arm omitting a collection reuses its registered full recipe. These are TRAIN assessments, distinct from the later soft-preference EVAL agreement.

| Collection | Mdata | Mw025 | Mw100 | Mmos | Mpair | Mlic |
|---|---:|---:|---:|---:|---:|---:|
| bapps | 0.7658 (0.0066) | 0.7622 (0.0073) | 0.7595 (0.0047) | 0.7658 (0.0066) | 0.7597 (0.0092) | 0.7617 (0.0065) |
| cid-iq | 0.9057 (0.0032) | 0.9050 (0.0037) | 0.9085 (0.0036) | 0.9025 (0.0068) | 0.9035 (0.0034) | 0.9038 (0.0039) |
| cid-oa | 0.9865 (0.0000) | 0.9865 (0.0000) | 0.9865 (0.0000) | 0.9868 (0.0002) | 0.9865 (0.0000) | 0.9865 (0.0000) |
| colourlab-rating-variability-2022 | 0.7762 (0.0132) | 0.7865 (0.0086) | 0.7873 (0.0064) | 0.7784 (0.0112) | 0.7730 (0.0145) | 0.7778 (0.0080) |
| fg-iqa | 0.8985 (0.0007) | 0.8970 (0.0008) | 0.8997 (0.0007) | 0.8985 (0.0007) | 0.8978 (0.0004) | 0.8989 (0.0008) |
| irccyn-ivc | 0.9460 (0.0019) | 0.9453 (0.0011) | 0.9457 (0.0025) | 0.9486 (0.0006) | 0.9477 (0.0024) | 0.9471 (0.0004) |
| live-md | 0.9650 (0.0049) | 0.9638 (0.0039) | 0.9597 (0.0059) | 0.9686 (0.0048) | 0.9664 (0.0048) | 0.9652 (0.0033) |
| mdid | 0.9414 (0.0092) | 0.9430 (0.0079) | 0.9350 (0.0099) | 0.9437 (0.0024) | 0.9392 (0.0088) | 0.9439 (0.0078) |
| pieapp | 0.6719 (0.0019) | 0.6704 (0.0009) | 0.6714 (0.0013) | 0.6719 (0.0019) | 0.6706 (0.0015) | 0.6703 (0.0003) |
| pipal | 0.6643 (0.0172) | 0.6588 (0.0146) | 0.6722 (0.0167) | 0.6678 (0.0137) | 0.6674 (0.0190) | 0.6684 (0.0119) |
| vcl-fer | 0.9374 (0.0078) | 0.9359 (0.0067) | 0.9301 (0.0062) | 0.9382 (0.0067) | 0.9388 (0.0082) | 0.9348 (0.0081) |

Frozen finalists: `FINALISTS.json` SHA256 `8191f8ced4488629c9c8d37b7439df203b0f1a9b58804bd6e9e312c8692c24c8`. Mlic reuses the coordinator’s seed-0 packed SHA256 `7612c2e46154fe9c0c359bfd02e521013f5717e3bf8e11967f5434150e220bcc`; Mw025 packed SHA256 `a6d43ef2ec717cb8f9b07ecb85f7f4c76aae7995cbce9b42e4d39ef7123ea4ac`. Both certify monotone with zero ramps. Mw025 used the registered 120 epochs × 50,000 pairs, fixed epoch119, seed0; its `run-heavy` completion was rc0, 2,271s, peak-RSS2.06GiB, min-avail15,297MiB, peak-load20.03. The RSS is the maximum single process; coordinator-owned Mlic64 work shared the explicitly approved42GiB/no-swap scope and remains separate from this study.

Mw025-s0 is retained in the candidate registry (source a9a7d34997cc). All19 registry bakes passed `sha256sum -c` through the mounted tower mirror, with checksum compute on dev and no remote process. Initial rsync ownership requests failed; the content-only retry passed. Prior registry metadata and both logs are preserved. `CANDIDATE_REGISTRY_VERIFIED.json` SHA256 `20255999ae094ae034fbebed23a5ae59c70c23e42b0820a0c2cddc26ca36a42f`. No AIC2026 target was opened.

### Six-arm diagnostic panel and first frozen EVAL report (2026-10-26)

The table retains only dataset-held-out rows, averaged over three seeds. Full per-type results remain in `scorecard/bytype_<arm>.json`; signed biases, clumping/clamping and raw/fit tails remain in `scorecard/{type_bias,knobinv,panel}.json`. Diagnostics do not choose finalists.

| Arm | Dataset | Type-bias SD | Knob inversion rate | Ladder RMS | Z-RMSE | Rank error p99 | Fitted error p99 |
|---|---|---:|---:|---:|---:|---:|---:|
| Mdata | kadid | 0.080899 | 0.014910 | 0.089059 | 0.484136 | 0.408663 | 0.379704 |
| Mw025 | kadid | 0.079287 | 0.013581 | 0.088365 | 0.477914 | 0.407721 | 0.381542 |
| Mw100 | kadid | 0.078583 | 0.016891 | 0.088331 | 0.477121 | 0.404256 | 0.375138 |
| Mmos | kadid | 0.077046 | 0.013532 | 0.087567 | 0.472411 | 0.399247 | 0.376680 |
| Mpair | kadid | 0.079390 | 0.014209 | 0.088273 | 0.477231 | 0.402072 | 0.379320 |
| Mlic | kadid | 0.077532 | 0.013097 | 0.087282 | 0.472303 | 0.401431 | 0.377887 |
| Mdata | tid2013 | 0.084836 | 0.010141 | 0.059200 | 0.444892 | 0.296727 | 0.277738 |
| Mw025 | tid2013 | 0.082506 | 0.010081 | 0.059223 | 0.441296 | 0.298891 | 0.276827 |
| Mw100 | tid2013 | 0.083095 | 0.010322 | 0.059587 | 0.444894 | 0.293984 | 0.275805 |
| Mmos | tid2013 | 0.089178 | 0.010503 | 0.059354 | 0.449385 | 0.304244 | 0.278204 |
| Mpair | tid2013 | 0.083482 | 0.009658 | 0.059466 | 0.445374 | 0.296650 | 0.273758 |
| Mlic | tid2013 | 0.084821 | 0.010141 | 0.058626 | 0.442287 | 0.298518 | 0.276314 |
| Mdata | konfig | 0.087935 | 0.000194 | 0.131721 | 0.591952 | 0.661607 | 0.574378 |
| Mw025 | konfig | 0.084933 | 0.000194 | 0.128165 | 0.577721 | 0.638393 | 0.569735 |
| Mw100 | konfig | 0.085640 | 0.000194 | 0.128845 | 0.584439 | 0.638393 | 0.569416 |
| Mmos | konfig | 0.089329 | 0.000065 | 0.133154 | 0.597035 | 0.691369 | 0.593593 |
| Mpair | konfig | 0.089025 | 0.000194 | 0.127953 | 0.575647 | 0.630357 | 0.561813 |
| Mlic | konfig | 0.084633 | 0.000129 | 0.128728 | 0.582080 | 0.654167 | 0.565668 |
| Mdata | cid22_a25 | 0.036431 | 0.000200 | 0.070738 | 0.493620 | 0.402445 | 0.377916 |
| Mw025 | cid22_a25 | 0.032990 | 0.000100 | 0.069846 | 0.486263 | 0.398364 | 0.372084 |
| Mw100 | cid22_a25 | 0.029139 | 0.000234 | 0.070553 | 0.486535 | 0.395027 | 0.374898 |
| Mmos | cid22_a25 | 0.035268 | 0.000100 | 0.069992 | 0.484925 | 0.397081 | 0.374690 |
| Mpair | cid22_a25 | 0.033190 | 0.000100 | 0.070161 | 0.488375 | 0.396106 | 0.375014 |
| Mlic | cid22_a25 | 0.038681 | 0.000167 | 0.070297 | 0.489250 | 0.395900 | 0.378415 |

The frozen single-seed EVAL point table follows. MOS uses signed midrank SROCC; FG-IQA/PieAPP use soft-probability 2AFC agreement. LIVE-MD peers cover all180 rows. Mdata and peers reuse the corrected coordinator copy; only Mlic and Mw025 are newly scored. BAPPS and paired reference-bootstrap reports are still pending at this record.

| Collection | Mlic | Mw025 | Mdata | M7 | B7 | SSIMULACRA2 | Butteraugli | CVVDP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cid-iq | 0.870 | 0.865 | 0.870 | 0.861 | 0.871 | 0.757 | 0.592 | 0.749 |
| cid-oa | 0.893 | 0.911 | 0.893 | 0.914 | 0.882 | 0.847 | 0.577 | 0.912 |
| colourlab-rating-variability-2022 | 0.722 | 0.723 | 0.722 | 0.723 | 0.701 | 0.604 | 0.373 | 0.612 |
| fg-iqa | 0.758 | 0.759 | 0.758 | 0.757 | 0.758 | 0.758 | 0.749 | 0.755 |
| iqa-odi | 0.842 | 0.848 | 0.841 | 0.843 | 0.849 | 0.854 | 0.806 | 0.833 |
| irccyn-ivc | 0.927 | 0.937 | 0.934 | 0.935 | 0.960 | 0.949 | 0.912 | 0.928 |
| live-md | 0.890 | 0.888 | 0.892 | 0.891 | 0.879 | 0.913 | 0.827 | 0.896 |
| mdid | 0.899 | 0.888 | 0.902 | 0.894 | 0.892 | 0.787 | 0.572 | 0.829 |
| nits-iqa | 0.724 | 0.715 | 0.727 | 0.717 | 0.704 | 0.777 | 0.706 | 0.721 |
| pieapp | 0.585 | 0.584 | 0.586 | 0.582 | 0.579 | 0.560 | 0.535 | 0.570 |
| vcl-fer | 0.940 | 0.937 | 0.945 | 0.938 | 0.884 | 0.945 | 0.877 | 0.933 |

The balanced TRAIN Mlic-minus-Mw025 difference is 0.0000016562213901, paired seed-unit SE 0.0006342087471001. The registered rule uses the point estimate. EVAL differences are mixed; no EVAL result changes the selected weights or candidates. Mlic’s licence eligibility does not establish product qualification.

Across all123 models, the standard set retains442 inversions (122 models have at least one), largest rise8.103523. The global set retains0 inversions, largest rise−0.730728. These are raw all-cell totals, separate from the primary twelve-E21-model table. Script lint passed on the final recipe/source tree; no Rust arithmetic or native binary changed for the linked-fold transport.

### AIC2026 public_train zero-shot of the revision-7 candidates (2026-10-26)

None of these models trains on any AIC2026 data. public_train is the development split: reported, never used to choose a
model; public_test was not opened. Revision-7 features of the 3,920 public_train pairs were dumped through B's
revision-7 stamp (B7, the same 420-id read set) with the mix study's binaries (`mixture_table.py build-pairs`, the route
that built the revision-5 `table_raw`; table `/var/tmp/byc-finalists-20261026/aic-rev7/table/table.parquet`, sha256 `a2969c70f2e1c96f…`). The table
route agrees with the pixel route on 8 spread pairs (max |difference| 4.9e-7), and B7 reproduces B's revision-5 global
SROCC (0.9091). Scored by `aic_train.panel` (SROCC against −JND); results `/var/tmp/byc-finalists-20261026/aic-rev7/zeroshot.json` (sha256 `7da5eded1d3bb03b…`).

| Model | Global, seeds 0 / 1 / 2 | Global, mean prediction of 3 seeds | Within source | Within ladder |
|---|---|---:|---:|---:|
| B (B7) | 0.9091 | — | 0.9558 | 0.9995 |
| M7 | 0.9121 / 0.9082 / 0.9047 | 0.9095 | 0.9498 | 0.9998 |
| Mlic | 0.9094 / 0.9089 / 0.9023 | 0.9076 | 0.9485 | 0.9998 |
| Mdata64 | 0.9077 / 0.9048 / 0.9019 | 0.9056 | 0.9497 | 0.9998 |
| Mdata | 0.9024 / 0.9048 / 0.9037 | 0.9050 | 0.9496 | 0.9998 |
| Mw025 | 0.9085 (seed 0) | — | 0.9503 | 0.9998 |

Globally every candidate ties B within the seed spread (0.902–0.912); the new datasets do not help here. B leads within
source (0.956 against about 0.949), the same price of monotonicity seen on the E21 sets.

### Mlic frozen EVAL paired comparisons (2026-10-26)

Whole-reference paired bootstrap,2,000 resamples/seed1, using stored predictions. The table shows signed SROCC differences. Except IQA-ODI42 references, cohorts contain3–12 references; the percentile intervals are descriptive and can under-cover. These do not select or retune models.

| Collection | Mlic minus | Delta SROCC [95%] | References |
|---|---|---:|---:|
| cid-iq | Mdata | +0.000189 [-0.003421, +0.004004] | 7 |
| cid-iq | M7 | +0.009458 [-0.003026, +0.021095] | 7 |
| cid-oa | Mdata | -0.000175 [-0.006703, +0.008004] | 7 |
| cid-oa | M7 | -0.021111 [-0.033628, -0.001387] | 7 |
| colourlab-rating-variability-2022 | Mdata | -0.000220 [-0.003001, +0.002630] | 3 |
| colourlab-rating-variability-2022 | M7 | -0.001348 [-0.008545, +0.009167] | 3 |
| iqa-odi | Mdata | +0.001086 [-0.007912, +0.012161] | 42 |
| iqa-odi | M7 | -0.000977 [-0.021805, +0.012834] | 42 |
| irccyn-ivc | Mdata | -0.007199 [-0.016933, -0.001400] | 4 |
| irccyn-ivc | M7 | -0.008756 [-0.016282, +0.000890] | 4 |
| live-md | Mdata | -0.001761 [-0.007023, +0.003704] | 12 |
| live-md | M7 | -0.001056 [-0.006278, +0.004983] | 12 |
| mdid | Mdata | -0.003719 [-0.012213, +0.002518] | 6 |
| mdid | M7 | +0.004469 [+0.000457, +0.008413] | 6 |
| nits-iqa | Mdata | -0.003481 [-0.006789, -0.001044] | 9 |
| nits-iqa | M7 | +0.007162 [+0.002204, +0.013487] | 9 |
| vcl-fer | Mdata | -0.004430 [-0.007531, -0.000464] | 8 |
| vcl-fer | M7 | +0.002058 [-0.001847, +0.008863] | 8 |

Relative to M7, Mlic has positive point differences on MDID and NITS and a negative difference on CID:OA with descriptive intervals excluding zero. Relative to Mdata, its IRCCyN/IVC, NITS and VCL differences have negative descriptive intervals. This preserves the mixed EVAL outcome alongside the TRAIN-only licence-eligible choice. The full report also retains B7 and all three peer comparisons and the labelled legacy row-bootstrap auxiliary field.

### Mlic64 against Mlic: the registered shape rule keeps M's shape (2026-10-26)

Mlic64 (the Mlic root with M64's shape, `mono:thr:mp64`) ran its 12 E21 cells and three full-data seeds on dev
(16 trainers in one 42 GiB scope, 02:58–04:03Z; peak 33.8 GB, no OOM kill). Its full-data seeds are certified monotone
(zero free ramps, selected epoch 119; packed sha256 seed 0 `7a05deec…`, seed 1 `2f4fbabc…`, seed 2 `85532fb1…`).

| Set (E21, out of sample, three seeds) | Mlic | Mlic64 | Paired difference (seed-unit SE) |
|---|---:|---:|---:|
| KADID | 0.8825 | 0.8829 | +0.0005 (0.0011) |
| TID2013 | 0.8741 | 0.8731 | −0.0010 (0.0010) |
| KonFiG-IQA | 0.8078 | 0.7993 | −0.0085 (0.0065) |
| CID22-A | 0.8808 | 0.8786 | −0.0022 (0.0037) |
| Mean of the four (per-seed differences +0.0000, −0.0043, −0.0042) | | | **−0.0028 (0.0014)** |

Rule 1 required a gain of at least two seed-unit SEs; Mlic64 is about two SEs worse, so **Mlic keeps M's shape**. M64's
gain on the Mdata data (+0.0046, SE 0.0017) does not carry over to the licence-clean data. Its nine held-out-new-data
cells could not change this outcome and were withdrawn from the fleet queue. Reported, not deciding: standard-mixture
inversions over the 12 cells, Mlic 41 (largest rise +6.17) and Mlic64 36 (+5.83); global edits 0 for both. Mlic's
full-data seed 0 already passes every label-free gate (NEARID 143 monotone ladders; dial C1 0.9977 / 0.9932, C3 0.7885;
G-STEER 132/135; identity at 12 SIMD permutations), so its one EVAL read follows (seed 0 reuses the data-legs lane's
frozen read). Mlic64's bakes stay registered for later evaluation.

### Mlic's one EVAL read: three full-data seeds (2026-10-26)

After the shape rule chose Mlic, its three full-data seeds were read once on the EVAL copy that already holds M7, Mdata
and Mdata64 (`/var/tmp/byc-dataset-eval-2026-10-25-seeds`; `eval_rows.json` and `features.parquet` byte-identical to the data-legs lane's copy). Seed 0
reuses the lane's frozen read (`pred_Mlic.tsv`, sha256 `88953d92…`); only seeds 1 and 2 were predicted here.
Report sha256 `c2a6ccdceef2adcb…`; paired report `/var/tmp/byc-dataset-eval-2026-10-25-seeds/paired-Mlicens/paired.md` (`391efe9f170a0bdb…`). SROCC of the three-seed
mean prediction (pairwise agreement for FG-IQA and PieAPP):

| EVAL set | Mlic | M7 | Mdata | Mdata64 | B (B7) | SSIMULACRA2 |
|---|---:|---:|---:|---:|---:|---:|
| CID:IQ | 0.870 | 0.856 | 0.869 | 0.868 | 0.871 | 0.757 |
| CID:OA | 0.890 | 0.904 | 0.901 | 0.901 | 0.882 | 0.847 |
| Colourlab | 0.726 | 0.728 | 0.720 | 0.721 | 0.701 | 0.604 |
| IQA-ODI | 0.842 | 0.845 | 0.843 | 0.843 | 0.849 | 0.854 |
| IRCCyN/IVC | 0.934 | 0.940 | 0.935 | 0.935 | 0.960 | 0.949 |
| LIVE-MD | 0.890 | 0.893 | 0.890 | 0.892 | 0.879 | 0.913 |
| MDID | 0.895 | 0.893 | 0.891 | 0.893 | 0.892 | 0.787 |
| NITS | 0.716 | 0.720 | 0.720 | 0.723 | 0.704 | 0.777 |
| VCL@FER | 0.937 | 0.939 | 0.941 | 0.942 | 0.884 | 0.945 |
| FG-IQA (pairs) | 0.758 | 0.757 | 0.758 | 0.758 | 0.758 | 0.758 |
| PieAPP (pairs) | 0.584 | 0.582 | 0.585 | 0.585 | 0.579 | 0.560 |

Paired (whole references resampled; 3–42 references per set, so descriptive): against M7, CID:IQ +0.014 (p 0.001),
CID:OA −0.014 (p 0.012), NITS −0.004 (p 0.040), the rest level; against Mdata and Mdata64, CID:OA −0.011, NITS
−0.004 / −0.007 and VCL@FER −0.004 / −0.005 (the cost of leaving out the validate-only data), Colourlab +0.006 against
Mdata, the rest level. Seed spread is large on NITS (seed 1 0.692, seed 2 0.732). BAPPS follows (reusing the lane's
seed 0). EVAL changes neither the shape nor the data choice.

### Final mix EVAL completion (2026-10-26)

Both frozen seed0 candidates were scored once on the copied eleven-collection feature table and once on the separate BAPPS feature table. Corrected peers and Mdata predictions were reused; no image was unpacked or peer recomputed. Both paired reports contain54 measured MOS comparisons (nine collections × six baselines). No EVAL result changes the TRAIN selection.

| BAPPS model | Soft agreement [95%] | Comparisons |
|---|---:|---:|
| Mlic | 0.673258310 [0.670479241, 0.676144688] | 36344 |
| Mw025 | 0.668503743 [0.665523746, 0.671545098] | 36344 |
| Mdata | 0.673924170 [0.671029194, 0.676816050] | 36344 |
| M7 | 0.661795620 [0.658752271, 0.664907551] | 36344 |
| B7 | 0.639216928 [0.636115728, 0.642518780] | 36344 |

BAPPS peer metrics remain deferred to the owner’s IMI archive route. Mlic’s seed0 predictions are reused by the coordinator’s separate three-seed EVAL read: main TSV SHA256 `88953d92fd88c410802faa4281e6f629e9bf5cf43de413aa4d4a1d2c9689c07b`, BAPPS TSV SHA256 `b2264f9944df460c5c947f3d408de146eb2d51e1ca48cd08b8a8d7787d7ee8e0`. No second seed0 model evaluation is needed.

| Collection | Mw025 minus | Delta SROCC [95%] | References |
|---|---|---:|---:|
| cid-iq | Mdata | -0.004487 [-0.010211, +0.000730] | 7 |
| cid-iq | M7 | +0.004781 [-0.002786, +0.012065] | 7 |
| cid-oa | Mdata | +0.017987 [-0.002640, +0.034435] | 7 |
| cid-oa | M7 | -0.002949 [-0.008893, +0.002312] | 7 |
| colourlab-rating-variability-2022 | Mdata | +0.001030 [-0.005308, +0.005740] | 3 |
| colourlab-rating-variability-2022 | M7 | -0.000097 [-0.001420, +0.002865] | 3 |
| iqa-odi | Mdata | +0.006839 [-0.003477, +0.024908] | 42 |
| iqa-odi | M7 | +0.004776 [-0.013952, +0.023866] | 42 |
| irccyn-ivc | Mdata | +0.003030 [-0.004452, +0.006542] | 4 |
| irccyn-ivc | M7 | +0.001473 [-0.001902, +0.010006] | 4 |
| live-md | Mdata | -0.003932 [-0.009445, +0.001779] | 12 |
| live-md | M7 | -0.003226 [-0.007217, +0.001002] | 12 |
| mdid | Mdata | -0.014768 [-0.028696, -0.005127] | 6 |
| mdid | M7 | -0.006579 [-0.011627, -0.002086] | 6 |
| nits-iqa | Mdata | -0.011931 [-0.019184, -0.006314] | 9 |
| nits-iqa | M7 | -0.001288 [-0.002621, +0.001194] | 9 |
| vcl-fer | Mdata | -0.007387 [-0.015144, -0.000064] | 8 |
| vcl-fer | M7 | -0.000899 [-0.002579, +0.000433] | 8 |

Intervals remain descriptive for the small reference cohorts. Mw025 has negative intervals against Mdata on MDID, NITS and VCL and against M7 on MDID. Its recovered old-data TRAIN mean is not a uniform new-data EVAL gain.

The complete owner command exited0 in40:00.00; `/usr/bin/time -v` maximum RSS1,738,980KiB. It ran inside the existing42GiB/no-swap scope with two native statistics threads. The measured RSS is for the owner process tree, not all separately joined coordinator work. `EVAL_COMPLETE.json` SHA256 `c5db7b5929d172732110176f7f75f66b3a3db9862f9aed5a7faeab237ba1947e`. Coordinator AIC public_train aggregate results were exposed during a post-freeze document merge; `POST_FREEZE_AGGREGATE_EXPOSURE.json` records that no raw AIC target table was opened by this lane and no selection or tuning followed.
BAPPS (36,344 held-out EVAL pairs, hard-pair agreement; seed 0 reuses the data-legs lane's frozen read, sha256
`b2264f99…`; seeds 1–2 predicted on the byte-identical copy `/var/tmp/byc-dataset-eval-bapps-seeds`): Mlic seeds
0.673 / 0.663 / 0.668, three-seed mean prediction 0.669 [0.666, 0.672]; M7 0.663 [0.660, 0.666]; Mdata 0.672;
Mdata64 0.672; B 0.639 [0.636, 0.643]. BAPPS TRAIN is one of Mlic's legs, so this is in distribution.


## Dataset mix archive closure — 2026-10-26

The six-arm study is complete: 123 native cell records (111 new fits and 12
reused), both all-cell mixture sets, 54 native hard-choice comparisons, the
TRAIN-selected Mlic/Mw025 full-data candidates, and one seed-0 EVAL assessment
per finalist on each feature bank. Closed `NUMERIC_SUMMARY.json` SHA-256 is
`842fdde7cbe67bf2a864667368f6dc22131d682e3a53ff6abdbabad9ab5bf1ea`.
Mlic's point-rule balanced advantage remains 0.0000016562213902; the mixed
EVAL results do not establish a uniform gain. Only Mlic meets the new-leg
licence policy. There is no serving-profile change.

The fresh snapshot `/var/tmp/byc-mixes-20261025/archive-ready/` and verified
NAS mirror `/mnt/tower/output/zensim-datamixes-archive-2026-10-25/` preserve all
study cell/full bakes, partial attempts and subset-root preparation artifacts.
All 9,682 artifact files, totalling 13,961,632,597 bytes, passed SHA-256 readback;
three additional random readbacks passed. The final checksum inventory has
9,683 entries including the receipt. Receipt SHA-256:
`041c5ff14bc7612038a64ed3d35ce6849a4258b12e2ab6cbe5e47072ddd4d675`;
inventory SHA-256:
`f93e5e7c07f375d3ce4bbebcf7f9be6a0be95a2a1dbf45aa5e7b563ba1dc34ca`.
Snapshot source: c2727d2cb135a06f8bbffceceddad9d557748674. Readback used client
fsync/POSIX_FADV_DONTNEED and hashing on dev; physical NAS cache temperature
is unmeasured. No remote process was started and no R2 mirror is claimed.

The coordinator-prioritized fleet native batch was admitted before this
single postprocessor. Its 4 GiB/no-swap child stayed under the unchanged
42 GiB/no-swap ancestor, preserving six science slots. It exited 0 in
344.11 s with `/usr/bin/time -v` maximum RSS 119,448 KiB; both child and parent
recorded zero OOM. Closed admission SHA-256:
`7254fc625190e216b3766ebd0a8b19d5e16132fadec447c813591178963a457a`.
The admission and full execution log are verified beside the NAS archive as
`.postprocess.json` and `.mirror.log`. The canonical dev `DATA_PROVENANCE.md`
records the closure and immutable original fleet archive/object pointers;
a verified final copy is beside the archive as `.data-provenance.md`.

The linked Mlic+sub root and all seven fold-admission controls are included.
The subsequent 21 subset cells and coordinator's subset selection remain a
separate study. The fleet reported its first two verified installs after the
snapshot; this lane launched no subset fits. Full artifact hashes and the
historical execution records are in the
[mix pointer](dataset_mixes_2026-10-25.pointer.md) and
[subset pointer](mlic_subsets_2026-10-26.pointer.md).

### Mlic+sub against Mlic: the registered rule keeps Mlic (2026-10-26)

All 21 Mlic+sub cells (linked root `6abfac53…`; 14 on the fleet, 7 on dev) were installed at 06:11Z and scored with
the study's own formula (`~/tmp/byc/subs_rule.py`, which reproduces Mlic's frozen balanced score 0.915076). Paired by
seed against Mlic: balanced −0.00072 (seed-unit SE 0.00036), old data −0.00255 (0.00038), new data +0.00112 (0.00081).
The rule needed the balanced difference within one SE and the old-data difference within two; both fail, so **Mlic
stays**. The five small subsets cost accuracy on the four older sets and gained little on the new data.

Owner ruling D6 (2026-10-26, `docs/DATA_SPLITS.md`): AIC2026 public_train may be a training leg. Next: a short
warm-started fine-tune of Mlic with that leg, assessed out of fold on rotating AIC2026 source folds, on E21 and on both
mixture tests; its rule is registered before any cell runs.

### Mlic AIC2026 warm start: rule registered before results (2026-10-26)

Owner ruling D6 admits exactly WG1 public_train. This follow-up starts from
Mlic's unpacked final checkpoints; it does not repeat its original training.
Only public_train labels may be opened. public_test and all EVAL rows remain
unopened by this lane. These development comparisons cannot establish a clean
unseen-population claim; prior AIC development exposure remains disclosed above.

Two fixed strengths retain Mlic's inputs, M shape (`mono:thr:mp32`), scaling,
old training legs and weights, pair budget (50,000 per epoch), and seed streams.
Adam state starts fresh. Both use constant learning rate 0.00001 and select the
final epoch, without development-based checkpoint selection:

| Arm | AIC pooled-rank weight | Extra epochs |
|---|---:|---:|
| MlicFT8x4 | 8 | 4 |
| MlicFT16x8 | 16 | 8 |

Each arm continues all twelve E21 cells (four omitted datasets, seeds 0–2).
The AIC held-out source fold is `(4*seed + E21_fold_index) mod 7`, preserving
the existing assignment of 28 public_train sources. Revision-7 fold tables
must be derived from the complete revision-7 public_train table, with row keys
and source exclusions checked. No revision-5/6 feature table is a training input.
Each comparison uses the same initial Mlic checkpoint and held-out rows.

For each seed, old-data delta is the equal mean of four signed E21 SROCC
differences against Mlic. Its standard error is the sample standard deviation
of those three seed means divided by sqrt(3). An arm qualifies only when every
cell has **zero global-edit mixture inversions**, and its mean old-data delta
is at least minus twice that paired seed-unit standard error. The existing
mixture comparison inventory and tolerances remain unchanged. Among qualifying
arms, select the largest mean of twelve per-cell AIC held-out global SROCC
gains against the corresponding Mlic zero-shot prediction. Exact ties prefer
the smaller epoch count, then the smaller AIC weight. No qualifying arm means
no finalist; do not loosen this rule after results.

Report both arms' complete E21 deltas and seed-unit errors; AIC global,
within-source, within-ladder and per-codec results, including source-fold
coverage and Mlic's same-fold zero-shot reference; both mixture tests;
knob-concurring inversions, type bias, Z-RMSE and error p99. Also report the
complete OOF row-level aggregate, averaging only predictions from cells that
omit that row's source fold. It is descriptive, not a replacement selector.

Before these arms, require byte-identical held-out predictions after loading
a Mlic final checkpoint and taking zero optimizer steps, refusal controls for
mismatched initialization contracts, existing trainer tests, and a default-run
arithmetic control against the frozen trainer. Freeze the new binary and its
SHA256 separately; never replace finished studies' binaries.

Only a qualifying winner receives three full-data continuations from Mlic's
three unpacked full-data seeds, with all public_train rows in the AIC leg.
Pack and certify those final bakes, register their hashes, and mirror them.
The coordinator owns subsequent product gates and the single EVAL assessment.


### Effective-sample collection masses: prospective TRAIN registration (2026-10-26)

Registered before any MlicNeff fit or candidate statistic. The hypothesis comes
from zenpapers `docs/mlic-recipe-review-2026-10-26.md` at `4e81c03a`, using the
design effect in `docs/composite-jnd-dataset.md` at the same revision. It does
not come from the completed AIC continuation's EVAL residuals. Those EVAL
populations are exposed and cannot become fresh selectors for this experiment.

**One changed factor:** collection mass. Keep the Mlic root `a7fd4af0…`, all
row payloads, family protections, target orientation, memberships, seven
holdout assignments, Rev7 features, M shape (`mono:thr:mp32`), 120 epochs of
50,000 pairs, fixed final epoch 119 and seeds 0–2. Use the original frozen
trainer SHA256 `acfb2a9f24fd7a294f027b552b19eef1b9b4d707cbdf5e0bdf22ef328db3fe28`
and companion binaries from the completed mix study. This is a derived mass
view of admitted Mlic data, not a new admission or use of its old confidence
manifest to admit another collection. No AIC leg is added in this experiment.

For each of the seven MOS collections, count rows N and distinct source-image
families G from the frozen label-free keys. Fix rho=0.33 and compute
`N_eff = N / (1 + (N/G - 1)*rho)`, exactly the review's mean-cluster design-effect
approximation. Rho is a hypothesis transferred from threshold data, not a
measured correlation for these seven datasets. IRCCyN has unequal family
sizes; this registration does not substitute another effective-size formula.
Use nominal mass `3.5*sqrt(N_eff)/sum_MOS(sqrt(N_eff))`. BAPPS keeps 0.5:
its independent parent-photo count is unresolved, so pair groups and patch
identities must not be substituted for photos. Total new-data mass remains
4.0. Preserve the existing within-reference acceptance correction separately.
LIVE-MD has seven image families but fourteen ranking groups; IRCCyN has five
families but seven groups. Neither grouping or pair ordering changes.

**Paired control and decision:** first reproduce Mlic's KADID seed-0 cell with
the frozen binary and require byte-identical held-out predictions. Then fit
MlicNeff's 21 cells: the four E21 omissions plus pipal, pairs and mos7, each
with seeds 0–2. Compare against the completed corresponding 21 Mlic cells.
Use the existing balanced score: half the equal mean of the four E21 SROCCs
mapped by `(rho+1)/2`, half the equal mean of eleven held-out TRAIN collection
statistics (MOS mapped similarly, pair accuracy unchanged). The three
validate-only new collections remain omitted from every fit.

Advance MlicNeff to full-data qualification only if its paired balanced mean
strictly exceeds Mlic's, its paired old-data mean delta is at least minus two
seed-unit standard errors, and every candidate cell has zero global-edit
mixture inversions under the existing inventory and tolerance. Compute seed
SE as sample SD of the three paired seed means divided by sqrt(3). Exact
balanced ties retain Mlic. Preserve every existing production gate; this
screen grants no release/default change. Report per-collection results,
standard-mixture inversions, knob-concurring inversions, type-bias SD, Z-RMSE
and error p99 through their existing owners. Do not choose checkpoints,
weights or variants from these diagnostics or exposed EVAL.

A qualifying candidate gets three full-data fits and the existing label-free,
steering, numerical and runtime qualification against exact model identities.
Runtime comparisons use permitted hardware, with no benchmarks on dev and no
use of r7900x. If the screen fails, retain Mlic and proceed to the separately
registered vote-loss or ordering proposal; a negative result is not task
completion. The seamless HDR frontend remains queued behind adoption of a
stable, qualified candidate. All artifacts live in the separate
`/mnt/v/output/zensim/mlic-neff-20261026/` study; original studies stay frozen.

The prospective selector shares the original balanced-statistic owner. Before
new candidate cells, its extracted helper reproduced every numeric field of
all six frozen original arm records exactly. Decision controls cover ties,
old-data regression, a single global inversion, missing cells and nonfinite
statistics. This changes no old study result or selection threshold.


### Native BAPPS vote likelihood: prospective TRAIN registration (2026-10-26)

Registered before constructing a vote-preserving root or fitting its candidate.
MlicVote tests the zenpapers vote-count proposal independently of MlicNeff:
its control is the original Mlic recipe and its eight nominal masses remain
0.5 each. No result of the mass trial changes this arm. The inference model,
Rev7 feature set, M shape, original seven holdouts, seed indices 0–2, 120 epochs
of 50,000 pair draws and final epoch 119 remain fixed. No AIC leg is added.

The [BAPPS paper](https://arxiv.org/html/1801.03924v2#S2.SS2) and
[dataset owner](https://github.com/richzhang/PerceptualSimilarity#c-about-the-dataset)
state two judgments per training comparison. Use only the 146,003 originally
frozen TRAIN memberships and their existing revision-7 feature mapping.
Excluded memberships and EVAL stay excluded. Require native integer counts
`n_a+n_b=2`, `n_tie=0`, `n=2`, and `p_a=n_a/2`; preserve the source count values
and pre-read ledger. A disagreement `(1,1)` is not an observer choosing an
explicit third tie response. It must remain a two-sided comparison with equal
probability, rather than being discarded or assigned a hard winner. Missing
or inconsistent counts block admission; do not infer new votes from a model.

Use the binomial negative log likelihood per vote, with the existing RankNet
logistic link: `p=n_a/2`, `d=distance_b-distance_a`,
`L=-p*log(sigmoid(d))-(1-p)*log(sigmoid(-d))`. Its derivative with respect to d
is `sigmoid(d)-p`. The factor of two is normalized out because every admitted
comparison has exactly two votes; total BAPPS sampling mass stays 0.5, with
the existing two-row acceptance correction. Keep the legacy arithmetic path
for old hard-rank groups and verify endpoint agreement, swap symmetry,
finite large-logit behavior and finite-difference gradients. A split vote's
minimum is equal predicted distances; it is not a skipped pair.

This isolates retaining vote information with the current logistic rank scale.
It does not implement zenpapers' broader probit/JND model, dataset unit or
viewing-condition parameters, or claim that a native logit is a calibrated
JND. Those require separate anchor admission and prospective comparisons.
Use the private subjective-ingress/loss owners with explicit new declarations;
no public trainer signature or inference API change is part of this trial.

Before fitting, require native/Python refusal controls for the vote schema,
counts, source pairing, non-TRAIN rows and producer/feature mismatches; an
actual unchanged-Mlic KADID seed-0 fit through the new frozen trainer must
reproduce the original 7,869 held-out predictions byte for byte. Freeze its
binary separately. The candidate runs all 21 registered cells. Pair-omission
cells must also reproduce their corresponding Mlic predictions exactly,
because their fitting inputs omit BAPPS entirely.

Apply the same prospective screen as the effective-sample trial: strict gain
in the existing balanced TRAIN score against Mlic, old-data paired delta at
least minus two seed-unit SEs, and zero global-edit mixture inversions in every
candidate cell. Exact ties retain Mlic. Preserve all production numerical,
identity, steering and speed gates. Report per-collection held-out TRAIN
results, both mixture tests and the existing smoothness/bias diagnostics.
The fixed held-out BAPPS accuracy population remains the original hard-pair
TRAIN assessment; tie restoration changes fitting only, not the selector's
population. No EVAL reselection or post-result weight changes.

A qualifying arm receives three full-data fits and full qualification; a
failed screen retains Mlic and leads to the separately registered ordering
comparison. Combining successful recipe factors requires another prospective
TRAIN comparison. The completed AIC and mass studies remain immutable.

Native vote-loss implementation checks (2026-10-26): the private sequential
CPU path accepts the reserved `subjective_vote_bapps` protocol and retains
the original arithmetic for hard-rank groups. The existing polarity owner
maps raw scores to distance differences. Invented controls verify the
two-Bernoulli likelihood, finite-difference gradients, side-swap symmetry,
finite logits at ±1000, actual training toward equal predictions for a 1–1
disagreement, and both unanimous preference directions. Unsupported parallel
and pooled routes and non-native probabilities refuse before optimization.
The four existing subjective hard-rank controls pass unchanged.

On i265, the native loss/control test command completed with
`run-heavy: done rc=0 166s | peak-RSS 1.96GiB | min-avail 19965MiB | peak-load 8.54`.
Strict workspace clippy passed afterward. Logs are in
`~/tmp/zensim-vote-preflight-20261026/{vote-loss-tests,vote-admission-checks}.log`
on that host. These are implementation controls, not runtime benchmarks or
evidence of candidate quality; the registered full-budget Mlic prediction
control and the 21 candidate cells remain required.

The private v3 declaration binds the native vote sidecar, two ordered sides,
fit-only role, original nominal mass, producer and feature contract. Python
and Rust metadata controls accept an absent response file without reading
it; the later payload phase verifies integer counts, original probabilities,
ordered keys and loaded targets. Native pairing also rejects references
that combine comparisons. The v1/v2 admission controls remain unchanged.
The admission suite and strict workspace clippy passed on i265:
`run-heavy: done rc=0 15s | peak-RSS 1.31GiB | min-avail 23032MiB | peak-load 4.68`.
Python `just dataset-leg-tests` passes, including response-before-admission
and count/target/role corruption controls. Real BAPPS counts have not yet
been admitted by this implementation; that remains a separate frozen-root
preparation and full-budget execution gate.

`just dataset-vote-root` prepares that fresh root from the original Mlic
root and BAPPS membership/feature pins. It preserves the seven MOS payloads
and nominal masses, keeps every one of the 146,003 admitted comparisons,
and writes a count sidecar bound to the ordered probability targets. The
corrected confidence policy is pinned to
`40407d74cf998da64400b36a7885b798333c886227c9c6aef5927cb1777e3e02`;
the older Mlic studies retain their original policy bytes. The new builder
refuses a combined subset/vote trial or missing policy before label reads.
Synthetic observation, admission and builder-refusal controls pass; the
fresh real-data root has not yet run, so counts remain unverified until its
admitted preparation succeeds.

The independent vote trial reuses the mass coordinator's fit, diagnostics,
mixture and paired-selection owners. It admits the fresh root, executes the
new trainer's unchanged-Mlic control, then requires all 21 candidate cells.
Before selection, the three pair-omission cells must reproduce all nine
original Mlic held-out prediction files exactly. Synthetic controls verify
the same numeric screen for either registered candidate and rejection of
changed omission predictions. Completed mass artifacts keep their archived
driver; extending the coordinator does not alter that running study.

Frozen native-vote trainer: source `e3151ba3642312be7599cd4c5584d15449fef491`,
binary SHA256 `dafda1f44e0dc2b18e0bbe8b878061cd9441e8ad58f656a6c8da6763ab390972`.
Rust 1.99.0, default portable release flags, thin LTO; build on i265:
`run-heavy: done rc=0 61s | peak-RSS 1.76GiB | min-avail 20271MiB | peak-load 5.03`.
The private study's `trainer/BUILD.json` records the compiler, lockfile and
binary identities. Its bytes were verified after copying; inactive Cargo
output was removed only after preserving this reference binary and build log.

The existing observation mapper now has an explicit vote-preserving mode.
Synthetic controls retain a `(1,1)` disagreement as two 0.5 targets and reject
missing, fractional, inconsistent or three-vote counts. The original hard-pair
mode and its tie-dropping test remain unchanged. Native loss/admission and
actual vote-root construction are still required before this arm can fit.


The unchanged-Mlic KADID seed-0 control completed: all 7,869 held-out
predictions are byte-identical, SHA256
`0cae0f6fcc1ca3f83108df4c3fd400751a6321842a56a0e942ecba09ab4bd8fa`.
Resource wrapper: `rc=0 1563s | peak-RSS 1.91GiB | min-avail 42743MiB | peak-load 3.99`.
This is training execution evidence, not a runtime performance comparison.
The derived MlicNeff root is frozen at SHA256
`772faa767a4399887ade260607b8c5fa11953eac9464c61a71f15caf2652187f`;
its tables and keys passed the byte-preservation audit. The trial driver is
frozen from source `cd2438b9e29e416ee33cec8a42cc7d29461946d0` under the
study's `source/`. `BASELINE.json` binds original models, predictions, binary
inventory and mixture manifests before candidate results. Live progress now
refuses a control-cell population when the recorded active command names a
candidate population. No candidate outcome is claimed here.

### Matched-codec sampling: prospective TRAIN registration (2026-10-27)

MlicOrder tests a bounded version of zenpapers' between-codec sampling
proposal, independently against original Mlic. It retains original collection
masses, features, M architecture, seven holdouts, seeds 0–2, 120 × 50,000
draws and final epoch 119. No mass, vote-loss or AIC continuation change is
combined with it. The initial admitted codec slice is KADID JPEG/JPEG 2000;
MIDD-GAR and JND-MID-MA need separate admission and are not part of this arm.

Match distances using the existing Mlic model for the same omission and seed.
Its fitting population excludes that cell's held-out collection. This is a
frozen-model matching proxy, not a replication of CVVDP/JND unit calibration.
For every admitted KADID reference, each JPEG row selects the JPEG 2000 row
with closest frozen prediction, and conversely; take the unique union of
these directed nearest-neighbor pairs. Break equal-distance ties by stable
pair key. Preserve every match and report its prediction-distance gap; do
not add a post-result distance cutoff. Native human scores remain the loss
targets. Pair construction uses no held-out response and no human target
comparison to choose a pair.

The base uniform sampler runs unchanged. For an accepted human-group draw
whose reference has matched pairs, replace visits 0, 4, 8, … with a matched
pair from that same reference. The matched pairs cycle in a seeded affine
permutation; alternate their orientation. This private schedule consumes no
base-sampler RNG values. Same-row collisions, group choice, reference choice
and every ineligible draw remain unchanged. This changes pair exposure and
order, not just order. Group/reference draw counts and collision counts must
match the original stream exactly in mechanical controls. Keep the legacy
E29 explicit-pair behavior unchanged; its collision replacement is a different
registered protocol.

Bind the per-cell list to the original Mlic root, exact fitting row keys,
promoted-bank codec identity, frozen matching model and prediction hashes.
Verify every endpoint is a distinct, same-source, cross-codec TRAIN row.
The KADID-omission cells have no eligible stratum and must reproduce the
three corresponding original Mlic prediction files exactly. Require a new
binary's unchanged-Mlic full-budget KADID seed-0 reproduction before candidate
fits. No public trainer signature or inference feature change is included.

Use the already registered paired screen: balanced TRAIN mean strictly above
Mlic, old-data paired mean delta at least minus two seed-unit SEs, and zero
global-edit mixture inversions in every candidate cell. Ties retain Mlic.
Report all per-collection statistics and the existing mixture/bias/smoothness
diagnostics. A passing arm receives three full-data fits and unchanged
production qualification. EVAL cannot select the match rule, share, seed or
model; combining successful factors requires a separate prospective trial.

### MlicNeff TRAIN screen completed: retain Mlic (2026-10-27)

All 21 registered cells finished epoch 119. The unchanged-Mlic driver control
had already reproduced all 7,869 predictions exactly. The fixed selection
rejects MlicNeff: balanced gain and old-data non-inferiority both fail;
the zero-global-inversion gate passes. No EVAL labels were used.

| TRAIN aggregate | Mlic | MlicNeff | Paired delta | Seed-unit SE of delta |
|---|---:|---:|---:|---:|
| Old four datasets, mapped rho | 0.930642384256 | 0.929448387557 | -0.001193996699 | 0.000499447448 |
| Eleven new dataset statistics | 0.899509341379 | 0.900327695895 | 0.000818354515 | 0.000333598169 |
| Equal old/new balance | 0.915075862817 | 0.914888041726 | -0.000187821092 | 0.000324928923 |

The old-data paired delta is below the registered minus-two-SE floor,
`-0.0009988948954513645`. The new-dataset mean gain does not override that
failure or the balanced regression. This rejects this fixed mass allocation;
it does not test per-item confidence weighting. No serving model was changed
and no full-data candidate fits were warranted by this screen.

Artifacts remain under `/mnt/v/output/zensim/mlic-neff-20261026`, including
`TRAIN_SELECTION.json` (SHA256
`3bc5c83846244a395e7862a0da5ff8ac70d066d2a7ff858225e7eea75d1ba59c`),
all per-collection results, diagnostics, mixtures, models and the frozen source. The supervised chain reported:
`run-heavy: done rc=0 4121s | peak-RSS 3.43GiB | min-avail 21560MiB | peak-load 15.64`.
This is an execution record, not a runtime performance benchmark.

The next registered independent trial is MlicVote. Its coordinator is frozen
from `def6372e067a02b13b398b2a43327874c7854a8e`, including the checked v3
wrapper dispatch, and first requires native-count admission and a full-budget
unchanged-Mlic reproduction with its separately built trainer. The production
candidate/HDR objective remains active after this negative screen.

### BAPPS released-aggregate source audit, before vote fits (2026-10-27)

The first MlicVote preparation refused blank `n_a`, `n_b` and `n_tie`
fields in the normalized library. No control or candidate fit started.
The failed preparation, source archive, ledger and logs remain preserved.

The original [BAPPS paper, Appendix B](https://arxiv.org/html/1801.03924#Sx4)
specifies two training judgments grouped into a target, with 0.5 for a split
vote. The [released dataset documentation](https://github.com/richzhang/PerceptualSimilarity#c-about-the-dataset)
also specifies two judgments per training triplet and the `judge` direction.
The library's pinned importer stores `p_a=1-judge` and `n=2`, leaving the
separate count columns blank. This admits a lossless source-format recovery,
not an inferred preference or an additional observation.

Before any vote fit, independently verify the released TRAIN archive SHA256
`b4f5c772c7cd88ac44bcb4ff0741745d434c05f0094f2bfb262ddcd0055bf393`.
Interpret only the 146,003 frozen TRAIN judge members, binding their triplet
paths to the existing membership and their original array bytes to hashes.
Require a single native aggregate exactly in `{0, 0.5, 1}`, normalized
`p_a=1-judge`, and stored `n=2`. Then the sufficient statistics are uniquely
`n_b=2*judge`, `n_a=2-n_b`, `n_tie=0`. No rounding, tolerance, model output,
new vote or changed membership is permitted. Do not parse excluded TRAIN
judge arrays or open the validation archive. Log the opaque archive hash
separately from the selected response reads.

A complete audit may supply an explicitly pinned count-recovery sidecar to
a fresh preparation. The direct missing/inconsistent-count gate remains
unchanged, as do the loss, masses, budgets and selection/production gates.
Absent a verified complete audit, the vote trial remains blocked while the
independent ordering trial proceeds. Synthetic controls check exact recovery,
wrong aggregate/direction/count refusal and validation-path refusal.

### Matched-codec sampler implementation controls (2026-10-27)

The private schedule passes a draw-by-draw comparison against the base
sampler: identical RNG continuation, group/reference choice, collisions and
ineligible draws; only every fourth eligible accepted-reference visit can
change endpoints. An affine cycle covers every admitted match, reversing
each pair on the following cycle, including even-length lists. The legacy
E29 pair-list control and the pinned legacy sample digest also pass.

Actual trainer controls confirm that admitted matches affect training,
reversing a pair preserves its updates, and unsupported heads/mining or an
empty schedule refuse. Existing hard-rank and native-vote controls pass.
Checks ran on i265, not as performance benchmarks:
`run-heavy: done rc=0 108s | peak-RSS 1.43GiB | min-avail 20919MiB | peak-load 7.66`.
Strict clippy passed for the changed library, trainer and integration targets:
`run-heavy: done rc=0 9s | peak-RSS 0.67GiB | min-avail 22915MiB | peak-load 1.55`.
The full-budget unchanged-Mlic control, source-bound matching preparation,
21 candidate fits and all quality/production gates remain required.

### Released TRAIN vote audit passed (2026-10-27)

The complete source audit recovered all 146,003 admitted comparisons exactly:
56,056 have `n_a=0`, 34,469 have `n_a=1`, and 55,478 have `n_a=2`.
Every recovered probability equals the existing normalized response. Excluded
TRAIN judge arrays parsed: zero; EVAL judge arrays parsed: zero. The validation
archive was not opened. The released source and interpretation are linked above.

The immutable audit is under
`/mnt/v/output/zensim/mlic-vote-source-audit-20261027/audit`; its recovered-count
payload SHA256 is `d8217592296df5e5819a5b9d0a9a5f74b3290d7bb04e91816f5983134b92b20c`.
The exact auditor script SHA256 is
`c18ed6b81dba2f76ab8638c58c1d8dfce35f08d4ea716f25f9d76fd44e80243d`.
Resource record: `run-heavy: done rc=0 122s | peak-RSS 2.37GiB | min-avail 43188MiB | peak-load 1.22`.

Fresh vote roots may bind this complete audit through independently frozen
metadata. Recovery logs the selected count read, verifies its digest and ordered
membership, checks triplet/source identity and every integer count against the
native response, then supplies the counts to the unchanged observation gate.
It does not modify the normalized library. Synthetic controls cover metadata-only
admission, source/role/population corruption, protected paths, payload hash drift,
count corruption and contradictions with existing native fields. The complete
Python dataset/subjective/vote admission suite passes. The failed first root
remains preserved; a fresh attempt still requires the unchanged-Mlic control
before any candidate fit.

### Matched-codec admission and execution wiring (2026-10-27)

The producer now freezes each original matching model, its fitting-table
predictions, row keys, promoted-bank codec keys and exact omission/seed. Native
admission constructs the bidirectional nearest-prediction matches, checks
same-reference identity and retains the pair/gap report in the trainer's
reproduction metadata. Matching never compares human targets. The candidate
registry contains 18 active cells and three inactive KADID omissions.

Metadata tests refuse changed policies, cells, seeds, roots and row keys before
response reads; attaching the schedule preserves group masses and development
legs. Native tests cover the nearest-prediction union and missing-manifest
refusal before table access. Final native checks and strict clippy on i265:
`run-heavy: done rc=0 4s | peak-RSS 0.54GiB | min-avail 24033MiB | peak-load 0.06`.

Portable release trainer source: `cbda4f0c734a38fa4c362f435f75f6d649c7a242`;
binary SHA256 `a13d014d65ff8c3e925bee21f0421c4c9dd18a9ba821290cd1ac4c4256342b54`.
Build record: `run-heavy: done rc=0 60s | peak-RSS 1.85GiB | min-avail 20783MiB | peak-load 3.26`.
Binary, compiler/configuration identity and logs are preserved outside Cargo
output. These are correctness/build records, not inference speed measurements.

The shared coordinator executes MlicOrder independently against original Mlic:
unchanged-Mlic control, matching preparation, 21 cells, existing diagnostics and
mixtures, then the unchanged registered screen. It binds each cell's schedule
and requires all three KADID-omission predictions to reproduce the originals
exactly. Synthetic decision controls retain Mlic on a tie, old-data regression
or any global inversion. No candidate quality or production qualification has
yet been measured for this arm.

### Stage-aware supervision (2026-10-27)

`just dataset-study-progress <study>...` resolves the active control/candidate
population from the recorded command and checks the actual process identity
and compute-lock ancestry. It also checks a queued launch's PID/start identity
before reporting it queued. Each study gets a durable `CURRENT_RUN.json` with
source/binary identity, active population, completed/required cells, current
processes, lock owner, last check, failure and next action. A dead queue cannot
report running; completed cells without a live worker or the final screen
marker report an interrupted chain. Synthetic dead-queue and stage-transition
controls pass. A TRAIN screen never marks the overall production/HDR goal
complete.

### MlicOrder stimulus-identity repair before launch (2026-10-27)

The initial queued study `mlic-order-20261027` was canceled before it entered
the compute slot. Its `QUEUE_CANCELLED.json` preserves the exact queue process
identity. The original fitting loader retains multiple observations that
share decoded pixels; the new ordering admission had incorrectly required
unique pixel keys and used the bank's representative codec filename.

The repaired private admission identifies observations by pixel key plus
source-row ID. It obtains each native codec from the original pinned source
TSV's filename, without parsing human scores. Equal matching-distance ties
use `(pair_key, source_row_id)` in that order; the source ID resolves only
otherwise identical pixel keys. The fixed matching rule, quarter-share
schedule, budget, reference masses and qualification gates remain unchanged.
No candidate predictions or outcomes informed this repair. The next launch
uses a fresh study and rebuilt trainer, with the unchanged full-budget control
still required before candidate fits.

Python metadata/recipe controls pass. Native sampler, rank/vote and private
admission controls pass on i265, including repeated pixel keys, distinct
source observations and stable bidirectional matching.
`run-heavy: done rc=0 137s | peak-RSS 2.11GiB | min-avail 18705MiB | peak-load 11.39`.
This is a correctness-check resource record, not an inference benchmark.

The replacement queue uses `just dataset-order-queue`, an attached coordinator
that records its PID/start identity before waiting for the shared compute
slot. It verifies committed source and trainer pins, preserves the complete
launch log, and writes a durable failure if launching the chain fails. Status
distinguishes an intentionally canceled queue from an unexplained dead one;
neither is reported running. A synthetic failed-launch control checks the
actual command construction and durable failure path without running a fit.

### Native-vote full-fit follow-through prepared (2026-10-27)

`just dataset-vote-full <screen-study> <fresh-output>` requires a completed,
passing native-vote screen. It recomputes the registered decision, verifies
the screen's input hashes, and uses that study's frozen training owners and
exact screened trainer. It fits seeds 0–2 at the unchanged full budget and
packs/certifies each final epoch through the existing production-pack owner.
Assessment seed 0 is fixed; the other seeds measure stability, not a pool
from which to choose a favorable result. Full fits do not establish production
qualification or authorize a serving change.

The runner writes process identity, completed models and failures as each cell
finishes. `dataset-study-progress` checks those models against their recorded
hashes and resolves the live process/lock before reporting progress. Synthetic
controls reject a changed or failing screen and a missing claimed-complete
model. No full vote fit is launched by this preparation; the independent
screen remains in progress.

### Confidence linking calibration: prospective admission and read scope (2026-10-27)

This registration defines the calibration prerequisite for the requested
per-item confidence experiment. It does not authorize a candidate fit before
its complete weight/sampler registration. It preserves original Mlic inputs,
all existing studies and every held-out source restriction. No new calibration
response has been interpreted to choose this scope or method.

Reuse zenpapers `jnd_utility_analyze.py` at `4e81c03a46fe2a9d86a73c3da75edaaec7fd9b9c`.
The old aggregate linking dispersion is inadmissible: validate-only FG-IQA
influenced its fit before final item-policy filtering. In addition, zensim's
owner-approved D4 roles in `docs/DATA_SPLITS.md` reserve MIDD-CAP and MIDD-GAR
for validation/source-exposed diagnostics. Their labels cannot fit a linking
parameter even though zenpapers' policy calls them trainable. Exclude all
three collections before reading fitting responses. VLIC is outside this
reconstruction; its visually-lossless criterion is not an existing donor
under the estimator's criterion rule.

The candidate donors are CEED2016 and PC-IQA pairwise slopes, plus the JPEG
and BPG first-JND units from JND-MID-MA's colour JND-TV protocol: at most four
aggregate units across three collections. CEED/PC require newly frozen,
source-cleared TRAIN membership. JND-TV retains its calibration-anchor
purpose and requires exact source-family admission for this linking use.
Its raw thresholds never become Mlic training targets. Inventory all three
JND modalities for shared-scene reservations, but fit only the colour protocol.
The tracked `confidence_calibration_targets.json` names the identifier tables.

Before responses, run the canonical native source-identity screen, unchanged
26-positive/3,181-unrelated calibration and inherited family/role checks.
Keep every unresolved match excluded. Inherit prior admissions, completed
fit exposures and protected estates; do not use a broad source allowlist to
clear them. Every scene, crop, modality and codec ladder shares its original
family. Newly assigned roles follow the existing canonical origin rule;
empty release split strings do not imply TRAIN. Freeze exact permitted
tables, rows, families, native source hashes, teacher identity intersection
and read-purpose receipts. The CEED img12 teacher omission (native size
mismatch, all six methods) is fixed before calibration; do not resize it.

After admission, the requested confidence experiment authorizes reading only
these frozen TRAIN/calibration observations, with a preread ledger. Teacher
distance remains `max(0, 10 - CVVDP_JOD)` from the pinned existing extraction.
Pairwise slopes retain the existing probit-binomial likelihood, half-vote tie
rule, at least ten usable pairs, 200 reference-bootstrap resamples, positive
lower bootstrap slope bound and teacher-majority agreement at least 0.7.
Process CEED then PC with the existing RNG seed 20261025 freshly initialized
for each independently admitted omission root. Preserve native vote counts;
missing values refuse rather than becoming hard preferences.

JND-TV uses the existing within-ladder linear interpolation and separate
JPEG/BPG axes. Never extrapolate a criterion beyond scored ladder support.
Keep full unit provenance (study, condition, codec, tables and family set).
Preserve the legacy three-decimal unit serialization before synthesis and
the population-SD convention: the linking floor is `std(log(u))` over at
least three positive, valid admitted unit entries. Retain the full linking
variance term; an insufficient or nonfinite calibration is INCOMPLETE,
never zero variance, a borrowed validation estimate or a chosen constant.
The two JND-TV codecs are not independent studies. Report study/family
dependence and retained support; this is an effective heterogeneity model,
not a demonstrated universal JND variance.

Each omission removes its held-out source families and aliases from donor
fits as well as native rating/observer fits, then recomputes affected units
and the floor. Reuse a calibration only when its entire permitted input set
is hash-identical. An omitted rating collection contributes no scale-map or
observer fit. Fit each retained rating collection's existing four-parameter
teacher-scale map using only its exact Mlic fitting rows, native uncertainty
semantics, fixed bounds and slope floor. Native targets remain unchanged.
Store the fitted coefficients, complete read ledger and all rejected units.
The inverse-variance weight construction, normalization and rank-pair mapping
still require prospective registration before candidate fitting.

The label-blind inventory at zenpapers `cf1efd5c3547` contains CEED2016's
450 pairs and 30 reference paths, PC-IQA's 1,800 pairs and 15 reference paths,
and JND-MID-MA's 210 threshold identities and 105 reference paths. The JND
views retain 35 shared scene tokens. Path preflight found all 210, 240 and
105 distinct projected paths, respectively; it read no pixels or responses.

`dataset-confidence-admission` supervises the next native source screen. Its
plan pins the committed producer, identifiers and original screening inputs.
The attached queue waits for the registered ordering study's verified screen
completion, then acquires the same resource lock. Native decoding, the fixed
D4 calibration and feature replay, and conservative match retention remain
owned by zenpapers. Completion requires the actual screen artifacts and is
explicitly **not** membership freeze or calibration-label read permission.
The queue refuses changed inputs, dead predecessor processes and completion
markers that disagree with their selection artifact. Failed attempts remain
on disk with their process identity, stage, error and complete log.

### MlicConfidence: prospective contrast-precision trial (2026-10-27)

This independent arm starts from original Mlic, with its original eight
dataset-leg masses, features, targets, group/reference sampling and private
TRAIN controls. It does not combine the Neff, vote-loss, matched-order or AIC
arms. No linking-calibration responses or confidence candidate outcomes have
been read to choose the following construction.

Weight only the registered KADID, TID2013, LIVE-MD and Colourlab rating-
variability fitting populations. Every other reference retains its baseline
loss explicitly; this is not an assertion that its uncertainty is known.
Within a covered reference, require complete item coverage, finite native
uncertainties, its admitted TRAIN-only teacher-scale map, and the legacy
teacher-placement gate (serialized training Spearman at least 0.600).
Missing coverage, a failed map or insufficient linking-unit support makes
the arm INCOMPLETE. Do not substitute band residuals, weight one, zero
linking variance or a validation-derived estimate for missing prerequisites.

Use the shared multiplicative scale-error model for a ranking contrast.
For item i, let x_i be its pinned teacher distance and
s_i = max(0.05, SE_teacher_i), using the existing 0.05-JOD item floor.
For a legal pair (i,j), freeze

`V_ij = s_i^2 + s_j^2 + sigma_L^2 * (x_i - x_j)^2`,

`a_ij = 1 / V_ij`, and `w_ij = a_ij / Z_gr`.

Here Z_gr is the deterministic expectation of a_ij under the original
conditional pair law for group g and reference r, after exactly the
existing same-row and rank-neutral exclusions. Enumerate its complete legal
support before fitting, including the existing modulo-draw probabilities;
do not estimate it from the realized minibatch. This keeps the conditional
expected scalar multiplier one and leaves original group/reference draws,
RNG consumption and skipped-pair behavior unchanged. A reference with no
active rank pairs keeps its original no-update behavior and is reported;
it supplies no invented normalizer. A single legal unordered pair has
multiplier one. Apply the multiplier to both the existing rank loss and its
gradient, without adding a new target or changing any absolute-loss term.

This model assumes independent native item errors and a shared first-order
dataset scale error. The native marginal SEs do not identify observer/item
covariances. The complete linking term remains present and is never divided
by an item, pair or observer count. Its cancellation at equal teacher
distances follows the declared shared-error model. This is a registered
adaptation of item confidence to rank contrasts; it does not import the
composite export's equal-band redistribution or global dataset-share cap.
Do not compare diagonal and shared-error variants and select one afterwards.
Any weight concentration statistic is descriptive, not a measured human
effective sample size and not a reason to shrink uncertainty.

Each of the seven original omission roots removes its omitted source
families and aliases before donor calibration, native observer/scale-map
fits and normalization. Reuse requires identical permitted-input hashes.
Use the same three fixed seeds, 120 epochs, 50,000 draws per epoch and
checkpoint 119 as the other registered Mlic comparisons. Before candidate
cells, the new binary must reproduce the full-budget unweighted Mlic
control byte-for-byte. Verify equal-precision loss parity, symmetric pair
exchange, conditional mean-one weights under unequal draw probabilities,
unchanged unweighted groups, source-wise masses and refusal on invalid
inputs. Preserve the original private/public API boundary.

The TRAIN screen remains the previously registered paired balanced-mean
gain, old-leg change at least minus two standard errors, and zero global
mixture inversions for all 21 cells. A passing screen proceeds to three
fixed full fits, fixed assessment seed zero and the unchanged numerical,
runtime and production qualification. No EVAL-based weighting, reselection
or serving promotion is authorized by this registration.


### Native-vote TRAIN outcome and owner-directed follow-through (2026-10-27)

All 21 MlicVote cells completed epoch 119 in
`/mnt/v/output/zensim/mlic-vote-20261027-run02`. The full-budget Mlic control
reproduced all 7,869 predictions byte-for-byte (SHA256
`0cae0f6fcc1ca3f83108df4c3fd400751a6321842a56a0e942ecba09ab4bd8fa`).
The frozen trainer SHA256 is
`dafda1f44e0dc2b18e0bbe8b878061cd9441e8ad58f656a6c8da6763ab390972`;
coordinator source is `39419ed821205201216d10e9369c4cf5305b6176`.

| TRAIN statistic | Mlic | MlicVote | Paired change | Seed-unit SE |
|---|---:|---:|---:|---:|
| Old collections | 0.9306423842555338 | 0.9292780130858365 | -0.0013643711696972316 | 0.00010319529093388322 |
| New collections | 0.899509341379377 | 0.9000448749395029 | 0.0005355335601258485 | 0.0002495605749905272 |
| Balanced | 0.9150758628174555 | 0.9146614440126698 | -0.00041441880478569154 | 0.00015546942979057443 |

All 21 global-mixture inversion counts were zero. The original registered
screen rejects this arm: balanced change is negative and the old-leg change
is below its -0.00020639058186776643 bound. Independent reconstruction of
that decision matched every stored decision field. `TRAIN_SELECTION.json`
SHA256 is `c8f07927432949f2892eb42af459cc431f79a7bba07722b4b4f42ea250a89de2`.
No EVAL read or serving adoption followed. The resource wrapper recorded
`rc=0 6854s | peak-RSS 12.05 GiB | min-avail 16734 MiB | peak-load 14.32`.

The owner's subsequent instruction requests an independent AIC adaptation
comparison and full-panel interpretation instead of strict SROCC gates.
This does not rewrite the completed screen or its numerical result. It
explicitly keeps this arm eligible for the prospective comparison below.

### AIC adaptation across training recipes: prospective panel study (2026-10-27)

Registered after observing the earlier Mlic, Neff and vote TRAIN studies and
Mlic AIC continuation, and before any new cross-recipe AIC continuation.
The owner requests in-sample AIC2026 public_train adaptation and complete
panels rather than strict SROCC cutoffs. Prior development exposure is known;
these are development comparisons, not a clean unseen-population claim.
No public_test, EVAL, TEST or TERMINAL target is admitted by this study.

Compare the original Mlic, MlicNeff, MlicVote and MlicOrder starting models.
The confidence arm joins only after its separately registered native-input
admission, calibration and TRAIN fitting. A failed historical SROCC screen
does not remove an arm. Keep its result visible in the panel. Do not merge
base-recipe factors or retune their finished fits.

First use each arm's twelve existing E21-omission checkpoints (four omitted
collections, seeds 0–2), preserving the same omission/seed pairing. Use the
original Mlic continuation inputs and losses for every arm: only initial
weights differ. In particular, Neff mass, vote likelihood and matched-order
sampling are properties of the starting checkpoint, not additional changes
to the common continuation. Use both previously registered strengths,
weight 8 for four epochs and weight 16 for eight, constant LR 0.00001,
50,000 draws per epoch, fresh Adam state, unchanged seed streams and final
epoch. Preserve revision-7 features, D6 admission, existing native source
folds and the rotation `(4*seed + E21_fold_index) mod 7`.

For each cell report predictions on both the 24 fitted AIC sources and the
four held-out AIC sources. Keep those panels separate. Report each starting
checkpoint on the identical rows, the post-continuation absolute statistics
and paired changes. Aggregate a row only over models with its declared
fitted/held-out status; expose the unequal model count per source. A fitted
panel is explicitly in-sample. Reuse completed Mlic continuations only when
the checkpoint, root, fold, binary, budget and prediction identities match.

Then compare all 28 public_train sources in three fixed full-data seeds for
each recipe and both strengths, using the original recipe to produce any
missing pre-AIC full fit. This follow-through is not gated on SROCC. Order
full fits require matches from the paired full Mlic model, not an omission
model. Fixed seed zero remains the assessment model; seeds one and two show
stability. No best-seed or best-epoch selection. Controls and immutable input
receipts precede each new stage; no partial stage is called complete.

For both fitted and held-out AIC panels report global signed SROCC and
Kendall correlation; within-source and within-codec-ladder rankings; every
source and codec's statistics; rank residual codec bias; canonical full
panel calibration statistics, Z-RMSE and error p99; knob-concurring
inversions; and paired seed summaries. Retain the source-level results and
use source-cluster intervals as descriptive uncertainty, never treating
repeated predictions of one source as independent observations. Report the
complete old-data panels, both mixture tests and new-data TRAIN results
alongside AIC. Record undefined metrics and denominators explicitly.

There is no scalar SROCC pass/fail or automatic winner. Compare the complete
tradeoffs and retain nondominated alternatives; explain any recommendation
using the panel. Existing pixel, arithmetic, source-separation, monotonicity
and production correctness requirements still apply. Serving adoption also
requires full runtime and steering qualification on permitted hardware;
identical model shape alone is not a speed measurement. This study neither
changes old EVAL outcomes nor authorizes EVAL-based fitting or reselection.

Reuse the frozen warm-start implementation and its existing default-run
control without merging its public-API PR. For every new starting recipe,
verify zero-step prediction parity before positive-step fitting. Pin all
starting checkpoints and their fit/omission receipts before launch, plus the
common root, continuation binary, source owners and public_train manifests.
Keep complete logs and process/lock-backed progress, and serialize compute
behind the active TRAIN study and its queued source-admission evaluation.


The adaptation runner is frozen at `91cf6403db0d`, with the original
warm-start owners at `87864f94` and trainer SHA256
`eae55210ab1a0b7efd1cb9bd07b2a745bb5b77dd0f139dc30b31f05d7cca1ded`.
Its plan lives in `/mnt/v/output/zensim/mlic-aic-adaptation-20261027-run01`;
PLAN.json SHA256 is
`d0a53135a567f190303e032d19e8003fc84d4a06c94b429802497dcb95b5a284`.
It is queued behind the active ordering study and the native source screen.
The fold stage requires 72 new continuations plus 24 verified Mlic reuses.
A reader check against actual Mlic checkpoint/continuation receipts passed.
Five refusal/contract tests passed on i265 under the resource wrapper:
`rc=0 0s | peak-RSS 0.03 GiB | min-avail 24494 MiB | peak-load 0.03`.

Before new panel results, fix the uncertainty display to the canonical
native full statistics, three paired seed units, and 2,000 whole-source
paired bootstrap draws with NumPy default_rng seed 1, percentile 2.5/97.5
bounds using linear interpolation. Reuse identical source draws across
contrasts; never resample rows independently. Show both each continuation
minus its own initial recipe and each alternative minus Mlic at the same
continuation strength. Source intervals describe the role-specific mean
prediction, not independent replication across overlapping fitted models.
Undefined values remain explicit with their denominators. Native full-panel
Kendall/PWRC magnitudes retain their owner's polarity convention; signed
SROCC is reported alongside them. Logistic calibration/tail statistics use
the canonical diagnostic fit on the reported cohort, not a new deployed
calibrator. No diagnostic fit or source interval is a selection gate.


The panel partition/refusal checks pass on i265 (nine tests total), including
unequal source-fold coverage, exclusion of fitted predictions from held-out
means, changed initialization/predictions and stale process/completion refusal.
The native wrapper recorded
`rc=0 0s | peak-RSS 0.04 GiB | min-avail 24489 MiB | peak-load 0.12`;
Python undefined-name/import checks also passed. These checks verify the
orchestration and role partition; they are not measured model-quality results.


Source-only review found that the new caller assumed `refit/last.bin` for its
zero-step control, whereas the frozen owner selects `refit/best.bin` at zero
epochs and records that path/hash in result.json. Run01 was cancelled while
waiting, before any control or continuation fit; its plan and cancellation
receipt remain intact. The repaired caller follows and verifies that receipt
(selected epoch -1), retains byte-exact prediction parity, and applies the
same linked omission exclusions as the fit owner. Positive-step fits still
require the registered final checkpoint. No historical owner was changed.

Initial recipe identity now joins each model to its completed study's exact
recipe/fold/seed inventory, model and receipt hashes, frozen root, dataset
partition and feature list. This prevents another same-shaped recipe from
acquiring the requested label. All 36 currently completed Mlic/Neff/vote
checkpoint bindings passed; Order binds after its own study completes.
Continuation receipts also bind their native authority and complete train log.
Eleven refusal/partition/owner-interface tests pass on i265, including the
zero-step directory containing only best.bin and a wrong-recipe model:
`rc=0 0s | peak-RSS 0.04 GiB | min-avail 24479 MiB | peak-load 0.39`.
The review also identified panel provenance and diagnostic coverage work;
those corrections precede acceptance of the new panel evidence.


### AIC panel integrity controls (2026-10-27)

The new comparison owner retains complete native statistics, including raw
PLCC, MAE and dropped-row counts. Undefined folds or seeds remain explicit;
the declared mean is undefined when a required component is missing. Source
bootstrap intervals accompany their own mean-prediction SROCC difference,
not the distinct mean-cell delta. Initialization rows are labeled zero-shot
on their paired continuation cohorts. Historical scalar decisions are
retained as evidence but excluded from this study's selection.

AIC ladder diagnostics bind the `quality` field to the published filename's
distortion-level index, lower being better; it is not a codec-native quality
knob. The provenance is the [dataset description](https://doi.org/10.18419/DARUS-6156),
whose local copy is pinned in the panel. Counts separate human-concurring
pairs, human disagreement, ties and undefined comparisons. Inversion sizes
use the canonical cohort-fitted diagnostic logistic map and p99-p1 span.
Fitted and held-out sources remain separate throughout.

Mixture evaluation derives each dense model in a fresh cache through the
existing bit-identity gate. Copied scores pass the canonical cache validator,
which checks model hash, settings, exact image coverage and finite scores.
The image inventories are compared by bytes and recorded with the scorer
identity. A completion inventory closes over generated predictions, scores,
metadata, logs and label-read ledgers.

Fifteen runner and panel controls pass on i265, including the best-only
zero-step export, wrong-recipe refusal, cohort isolation, undefined-denominator
handling and changed mixture model/settings/coverage refusal. The resource
wrapper recorded `rc=0 0s | peak-RSS 0.12 GiB | min-avail 24490 MiB | peak-load 0.03`.
These are implementation checks, not measured candidate-quality results.


### Full-data Order admission for the AIC panel study (2026-10-27)

The full-data continuation registration now has a separate matching admission
path. It uses the three original full Mlic models, bound to their completed
result hashes and fixed seeds, to predict the unchanged human fitting table.
The native owner requires the exact full-data schema, original model, full
target cell and matching seed. Omission admission keeps its previous schema
and refusal behavior. No matching pair is selected using human responses.

Five native admission/metadata controls and six Python matching controls pass
on i265. Targeted clippy passes with warnings denied. The resource wrapper
recorded `rc=0 157s | peak-RSS 1.95 GiB | min-avail 19227 MiB | peak-load 5.30`.
These checks establish admission behavior; the full-budget binary control
and the actual full-data Order fits have not run yet.


### Full-data adaptation coordinator checks (2026-10-27)

The separate full-data coordinator implements the registered follow-through
for every recipe and both strengths, without consulting historical scalar
selectors. It waits for the complete fold panel, replays a full Mlic fit with
the new Order binary, requires identical final and packed models, prepares
full-model matches, and fits the nine missing original-recipe seeds. Each
recipe then receives a zero-step initialization control before common AIC
continuation. The three existing Mlic 16×8 continuations are reused only
after checkpoint, seed, authority, actual AIC mass and packing checks.

Metadata validation passed for all three original Mlic full fits and all
three existing 16×8 full continuations. Four new orchestration controls pass
on i265, including wrong seed, changed initialization, omission, optimizer
state and AIC mass refusals. The resource wrapper recorded
`rc=0 0s | peak-RSS 0.02 GiB | min-avail 23316 MiB | peak-load 3.26`.
This coordinator produces fit artifacts; full-data panels and production
qualification remain required after it completes. No new full-data fit
quality result is claimed by these checks.


### Full-data panel representations (2026-10-27, before new full fits)

The full-data report compares the final f32 model and the packed, TRAIN-
calibrated serving model for every recipe, strength and fixed seed. All AIC
continuation sources are explicitly in sample; each initialization is
zero-shot on AIC. The four original E21 populations are also in sample for
these full fits, including original development rows. Their separate
omission panels remain linked evidence. No EVAL or public_test is opened.

Retain the complete native statistics, source/codec/ladder panels, codec
rank bias, scatter/error tails, AIC level-concurring inversions, paired
three-seed changes and both mixture tests. The same registered 2,000 whole-
source bootstrap draws (seed 1, linear quantiles) describe each AIC
mean-prediction contrast, conditional on the fitted models. Fixed seed zero
remains the assessment model; the three-seed mean is only a stability
diagnostic. Undefined statistics retain their denominators.

The deployment contrast includes f16 quantization and the existing output
calibration together. It must not be described as isolated precision loss.
Runtime cost and native HDR precision remain separate qualification work.
Three new controls pass on i265 for exact seed coverage, explicit undefined
seeds and refusal of a stale or incomplete preceding fit. The resource
wrapper recorded `rc=0 0s | peak-RSS 0.04 GiB | min-avail 24500 MiB | peak-load 0.71`.
No full-panel quality or speed result has been measured by these controls.

### Active execution index and recovery, 2026-10-27

The owner requested durable documentation independent of agent memory.
[SESSION-RESUME](../SESSION-RESUME.md#current-work--october-27-2026) now owns
the current goal, exact run roots, plan hashes, process-backed status commands,
source owners, pending confidence implementation and HDR dependency. Run
`just mlic-priority-progress` before any status claim or restart. This additive
index does not alter any registration, frozen source snapshot or prior verdict.

At 04:19:40 UTC, Order was running with 12/21 cells complete, no failure and
the remaining nine advancing. The verified compute lock belonged to its
ancestor PID 1017472; coordinator 1341080 had start ticks 229427382. The source
screen, AIC fold fits, fold panels, full fits and full panels were all queued
with no failure. These are observations to re-resolve, not permanent liveness
claims. Full qualification and adoption remain outstanding.

The operative AIC fold plan is **run02**, SHA256
`1cfdfdf601efefc79a8e8292cf077776f52c071ddc780965051cb503bc3adca7`,
source `86748f41`. The earlier run01 paragraph above describes the preserved,
cancelled attempt; its frozen warm-start source is still an input, not a live
queue. Fold panels use `3ab3f980`. The full-data plan is
`/mnt/v/output/zensim/mlic-aic-full-20261027-run01/PLAN.json`, SHA256
`633362ce633d5f172041ac88728c61552c0b16e4368480933c8d27c092144bf7`,
with full fit owner `7ec5f55f` and full panel owner `aa300e1d`.
Its separate full-Order trainer SHA256 is
`a34ab388f205a9641090ab9838be9d91ee0ae6630b0e1a616351bddc02b50dad`;
the active omission study retains `441bda583ed1ff728a4cc50565a0a114d00b12e0b22a282d36ea2ddf15614571`.

Confidence source screening uses PLAN-v2, SHA256
`8e33269e3ec70b71f48a1c74b508fe3ed493a156afea9202d1c2eee82024576a`.
Its metadata-only `FREEZE_REQUEST.json` is prepared, not queued or executed.
Screen completion still requires retained-match review before membership;
membership then requires independent validation and an exact preread ledger.
The purpose-bound response adapter and native contrast-weighted training
implementation remain work to do. No historical composite floor is reused.

### Confidence omission identities before calibration (2026-10-27)

The metadata adapter now freezes all eight omission/full scopes from the
original Mlic receipts. For an E21 omission it uses the complete assessment
identifier table, not only the smaller fitting subset: the original receipts
declare 65 KADID, 25 TID, 14 KonFiG and 25 CID22-A assessment references.
It verifies the retained fitting keys against the exact original filter,
preserves distinct source-row observations, and binds original source paths
through the pinned stimulus metadata. Dataset omissions retain every original
TRAIN family ID from the receipt-bound identifier keys. No feature, target or
uncertainty value is interpreted by this preparation.

This is identity preparation, not completed native family admission. The next
step must join every source reference and old-family alias against the
independently validated native source registry before selecting donor or rating
responses. Unknown/ambiguous joins cannot be silently omitted. Three metadata
controls pass on i265, including an assessment reference absent from fitting,
pixel-identical distinct observations, wrong omission/source refusal and role
or family mutations. The resource wrapper recorded
`rc=0 0s | peak-RSS 0.07GiB | min-avail 24494MiB | peak-load 0.03`.
The owner is `scripts/byconstruction/confidence_omissions.py`, exposed through
`just dataset-confidence-omissions`; the real metadata run follows this check.

The first real metadata run refused before output: KonFiG/CID22 source-row
numbers are local to their collections and do not index the KADID/TID source
file. Its reference guard caught this mismatch; no source was admitted and no
response interpreted. The corrected adapter uses each collection's pinned
bank pair key/reference group and retains reference pixel hashes/dimensions.
Distinct observation IDs remain distinct. Original manifest constants were
moved unchanged into a passive shared owner, avoiding filesystem reads merely
to import those identities. The canonical bank resolver selected the original
preserved snapshot, since mutable promoted manifests have changed.

Four controls now pass, including colliding row numbers across collections:
`rc=0 1s | peak-RSS 0.07GiB | min-avail 24508MiB | peak-load 0.07`.
Run01 and its failed join are retained at
`/mnt/tower/output/zensim/mlic-confidence-omissions-20261027-run01`.
Run02 has a readback-verified identifier-only snapshot of all six member
manifests and key tables. No feature/label-bearing bank table was transported.

Run02 then refused the native dataset identifier schema: its family IDs are
canonical decimal strings, while the new adapter expected JSON integers.
The adapter now accepts the exact canonical numeric identity and rejects
booleans, signs, padding and non-integer notation. Four controls pass on i265
with native-style string IDs (`rc=0 0s | peak-RSS 0.07GiB | min-avail 24522MiB |
peak-load 0.01`). Run02 and its refusal remain preserved; no output or response
read resulted. Run03 reuses the verified bank snapshot and a new source identity.

Run03 completed the real metadata preparation with source `8369c385`:
`rc=0 2s | peak-RSS 0.42GiB | min-avail 24521MiB | peak-load 0.00`.
Its `identities/OMISSION_IDENTITIES.json` SHA256 is
`b9760b0be03fc33997d5bc054285269b77b170c36e16681c520519a47ec71194`,
under `/mnt/tower/output/zensim/mlic-confidence-omissions-20261027-run03`.
It retains 65/25/14/25 complete KADID/TID/KonFiG/CID22-A assessment references
and 200/111,363/38 original PIPAL/pairs/mos7 dataset family IDs. Full has no
omissions. This is not the native family join or permission to read responses.
The run receipt explicitly records both as false. Every reference pixel identity
and inherited prior-family alias must resolve against the independently
validated native registry before omission-specific calibration can start.

### Admission dependency path repair, 2026-10-27

Order finished 21/21 cells at final epoch119. The next source screen failed
at 04:52:35 UTC while writing its decode receipt because its generic packed
reader path did not exist. The original verified binary is preserved under
its SHA-suffixed name; no source threshold or numeric implementation changed.
The failed attempt recorded `rc=1 2s | peak-RSS 0.05GiB | min-avail 43632MiB |
peak-load 7.49`. No screen completion or response read resulted. Each of the
four AIC downstream stages refused its failed dependency before fitting.

The coordinator now requires the packed reader to be an existing input with
an explicit SHA binding before queueing. Five controls pass on i265, including
missing/unbound reader refusal before native work (`rc=0 0s | peak-RSS 0.02GiB |
min-avail 24506MiB | peak-load 0.04`). Fresh admission/fold/full run directories
preserve failed outputs. Their scientific populations, recipe/seed budgets,
source-screen controls and full-panel policy remain unchanged.

Order's immutable TRAIN result SHA256 is
`e82ab44b6f9d6a62ed1a50e754af0d1e5979cc206ded92ef09440b663262e9db`.
Paired old/new/balanced mean deltas were -0.002100909334729025,
+0.0005991971695652835 and -0.0007508560825818522, with seed-unit SEs
0.00042630698299435675, 0.000633077031635272 and 0.0005203539720612126.
All 21 Order cells have zero global-mixture inversions. The frozen screen
retains Mlic; no EVAL read or adoption follows. Order remains eligible in
the owner's new full-panel AIC study. Resource receipt:
`rc=0 7084s | peak-RSS 3.43GiB | min-avail 19911MiB | peak-load 15.35`.

At 04:55:32 UTC replacement admission run02 was running its source screen,
native decode complete, with lock-holder PID1832007. The four replacement
AIC stages were alive/queued, without failure. Current plans are:

- `/mnt/tower/output/zensim/mlic-confidence-admission-20261027-run02/PLAN.json`,
  SHA `5514892579153da37ae638aa30e4194917fdf3bb6282aeb6ca793f9430268fc1`.
- `/mnt/v/output/zensim/mlic-aic-adaptation-20261027-run03/PLAN.json`,
  SHA `98f72bda706481f885ddd1c295e0ce3ab9c1f38f29d8afd0c59115022a8afa54`.
- `/mnt/v/output/zensim/mlic-aic-full-20261027-run02/PLAN.json`,
  SHA `220bc378937354fc5cc7ca394eda580a343765ba83a39a3c94f471a840c1147a`.

Only the admission preflight owner changes (`fc8836f1`). AIC source owners,
all existing input pins, full Order source/binary, recipes, thresholds and
budgets remain unchanged. Old source snapshots remain explicit frozen inputs.
`just mlic-priority-progress` now resolves the replacement chain. Do not
restart the preserved failed plans or treat old state files as live progress.

The omission native-pixel audit on i265 matches all 65/65 KADID, 25/25 TID,
14/14 KonFiG and 25/25 CID22-A assessment references against the original
486,013-image source inventory. Exact RGB hashes and dimensions all match;
no source is dropped. The run03 `NATIVE_PIXEL_COVERAGE.json` binds the
inventory SHA `5c16fb53d31f2407e373bf83eb74b89653c7721499b4c7b6e807c3f25cb51968`.
Resource receipt: `rc=0 5s | peak-RSS 0.80GiB | min-avail 23477MiB | peak-load 0.10`.
This is metadata compatibility only. The private family resolver now has six
passing alias/omission controls; its real graph replay still requires completed
source screening, frozen membership and independent validation before any
purpose-bound response ledger. No confidence weights or maps follow yet.

### Source admission completed; AIC continuation running (2026-10-27)

The repaired screen completed all 150 native paths (147 unique query pixels)
against 486,039 images, retaining 45 possible matches. All 26 positive controls
and the original one-of-3,181 unrelated survivor remain unchanged; 3,603 native
calibration feature arrays are bit-identical. Screen summary SHA256:
`3575b1ced6dce17be0ff0d67ea438e93bb46e36604c2b015c4a33b011920148c`.
`rc=0 937s | peak-RSS 8.31GiB | min-avail 35731MiB | peak-load 1.98`.
AIC fold run03 acquired the lock, passed zero-step controls, verified all 24
Mlic reused cells and began new fits; observed completion reached 30/96.

Forty source matches were already within declared/confirmed families. The
remaining five were visually verified color/grayscale versions of identical
scenes. The reviewed freeze and independent validation preserve 225 CEED
TRAIN pairs, 600 PC-IQA TRAIN pairs, and exactly 70 JND-TV calibration rows;
140 GR/IR rows remain reservation-only. No current fit/held-out family or
prior reserved-family intersection remains. These are metadata admissions,
not new label reads or model-quality results.

The complete omission-family replay matches the registry and inherited aliases.
Its output SHA256 is `ca0629ca172daee0196c1f4f383b3c8bc5e4e6ad3b694c52f517bc2b25d6a845`;
TID and mos7 each intersect five current registry families. All 14 KonFiG
reference identities remain represented through seven confirmed components.
Unknown/ambiguous identities were not dropped. The next step is the exact
omission-bound preread ledger and purpose-bound linking reader, followed by
native maps/weights and controls. Fresh confidence fits remain outstanding.

### Exact confidence preread scope completed (2026-10-27 05:29 UTC)

The metadata-only run02 ledger is frozen at SHA256
`4fef7c7b97099f4591f15f39446369f245acd73ec50323bfc2e2b551c42a2bb8`,
private owner `55d3bebb`. Its row scopes and input hashes equal run01; it adds
the explicit teacher column `cvvdp_cpu_imazen_v0_1_0_standard_fhd`.
Permitted counts are 225 CEED/600 PC-IQA/70 JND-TV except mos7's 195/0/70.
No quality responses or teacher values were extracted. Resource receipt:
`rc=0 1s | peak-RSS 0.24GiB | min-avail 24354MiB | peak-load 0.00`.
The purpose-bound reader, omission linking fits, native maps/weights and all
21 confidence fits remain outstanding. Missing positive linking units cannot
be replaced by a zero floor. AIC independently reached 57/96 fold cells with
no failures; all four zero-step controls precede the fit loop and passed.

### Native omission linking completed (2026-10-27)

The eight scoped linking fits are complete, with index SHA256
`486aeae1fcc68dcce5ceb56440ca714179700ac55e32c0086d631bbb34924a2d`.
Seven scopes have four positive units across three studies and sigma_L
0.4442630190242332; mos7 has three units across two studies and sigma_L
0.5263977834858703. JPEG/BPG units share observers. No JND threshold was
dropped. Exact registered responses/teacher values have now been read.
The first run's empty PC-IQA caller failure is retained; the caller repair
reports absence without changing native fitter validation. Four initial and
four repaired results are independently hash-checked, with exact source
snapshots. Repaired run resource: `rc=0 5s | peak-RSS 0.21GiB |
min-avail 24210MiB | peak-load 0.23`. Native rating maps, observer fits,
contrast weights, parity controls and 21 TRAIN fits remain outstanding.

### Native rating maps completed (2026-10-27)

All eight scopes / 28 retained maps pass the unchanged input-validity gate.
Completion index SHA256:
`aefbc62b2aceaed2b648e1d8bd346d2bd5c10dc0fff974fa4287896b6f0b14fb`.
Serialized teacher correlations: KADID 0.850, TID 0.861, LIVE-MD 0.863,
Colourlab 0.692. These are calibration-input checks, not candidate selection.
Full-precision coefficients and native item SEs are preserved. Two KADID and
two TID SEs are zero; the registered 0.05-JOD contrast floor remains binding.
Legacy unused per-band information diagnostics warned on those zeros; no
infinite item SE was exported. `rc=0 7s | peak-RSS 0.26GiB |
min-avail 24127MiB | peak-load 0.26`. Exact contrast normalizers, private native
loss integration, full-budget parity and 21 TRAIN fits are next.

### Native conditional contrast normalizer (2026-10-27)

`zensim-validate/src/mlp_train/contrast_precision.rs` now implements the
registered inverse variance and deterministic conditional mean precision.
It counts modulo residue multiplicities with u128 integers, enumerates all
ordered legal pairs in original bucket order, and consumes no RNG. The
caller must supply the existing tie predicate: legacy human distinct rows
with equal finite scores remain legal; subjective-rank groups exclude exact
ties. No sampling or tie rule changes are authorized. Equal precision returns
exact weight one; one unordered legal pair does too. Empty support produces
no normalizer. Equal SE alone does not imply equal precision.

Four native tests pass on i265, including exhaustive 3/5/8-bit RNG spaces
with unequal modulo probabilities, both legal-support policies, 64-bit mass
counts, symmetry, the unchanged item floor/linking term and invalid inputs.
`rc=0 0s | peak-RSS 0.12GiB | min-avail 24368MiB | peak-load 0.08`.
The initial exhaustive oracle used uncompensated repeated sums; replacing
only that oracle accumulation with compensated summation passes the same
2e-12 tolerance. The arithmetic owner is not yet integrated into training.
Next bind exact original group rows and reference order to the admitted maps,
then multiply only native rank loss/gradient and run the full-budget parity
control before confidence candidate cells. No public API change is required
by this arithmetic owner.

### Exact confidence row binding (2026-10-27)

The prospective binder reads only original TRAIN `human_score` and
`ref_basename` columns after verifying complete calibration results, original
cell authorities and archived table hashes. It records an exposure event
before target interpretation. Its digest follows the native loader: f64 target
bytes and u32 dense first-seen reference IDs, row by row. Native item identities
bind calibrated x/SE to original table positions; duplicate, missing,
nonfinite, wrong-reference and partially covered reference inputs refuse.
Uncovered references remain explicit baseline rows. All original group
weights, modes, seeds and full training budgets remain in the cell declaration.

Two binding controls pass on i265: `rc=0 1s | peak-RSS 0.07GiB |
min-avail 24365MiB | peak-load 0.03`; lint passes. These tests establish the
row-binding contract, not native training integration or candidate fitting.
The private native integration will consume a hash-pinned research manifest
without extending the public training API, validate the loaded row digest
and exact cell configuration, and apply the multiplier only to rank loss
and gradient. No sampler RNG, pair replacement or absolute-loss changes.

The first real row-binding pass stopped after 12 cell manifests because dataset
omission authorities retain their original `full-sN` namespace. The caller
now resolves that exact historical namespace for pipal/pairs/mos7, checks seed
identity, and preserves the source authority unchanged. The partial run and
exposure receipt remain; fresh outputs are required. Three binding/namespace
controls pass: `rc=0 0s | peak-RSS 0.07GiB | min-avail 24366MiB |
peak-load 0.13`. Failed real pass: `rc=1 4s | peak-RSS 0.37GiB |
min-avail 24051MiB | peak-load 0.28`. No confidence candidate fit started.


### Native confidence integration and logical-width repair (2026-10-27)

The private trainer integration binds original group order/masses, loaded-row
hashes, seeds, epoch budget and logical feature width before applying contrast
precision to native rank loss and gradient. With its research environment
absent, the original path remains selected. It preserves native ties, pair
sampling, absolute loss and uncovered references. Two loaded-row integration
controls and four arithmetic controls pass: `rc=0 94s | peak-RSS 0.95GiB |
min-avail 23063MiB | peak-load 3.57`. These are implementation tests, not
full-budget reproduction or model qualification.

The second binding run produced all 21 manifests, index SHA256
`8e037987ed658efb4c42013e2fc79d1ebdbd2731a3c4cfa4018f69c2deb6c9ff`,
but remains unexecuted and superseded: `n_features=410` confused selected raw
IDs with the trainer's logical width. Original raw width is 1,853, selected
raw IDs 410, registered derived products 410 and logical width 2,263.
The corrected producer separately records these dimensions and binds the
registered derived declaration, SHA256
`a97a5fa63937ed8fade3c56fd7470765a031d1d309afc36e631f371a1796e648`.
Four Python binding controls pass, including mismatched declaration refusal:
`rc=0 1s | peak-RSS 0.07GiB | min-avail 24325MiB | peak-load 0.18`.
No confidence fit used the wrong-width manifests. Generate fresh run03 outputs;
keep the old exposure/result files. Full-budget unweighted parity and 21
confidence fits remain outstanding.

At 06:03:37 UTC, AIC fold run03 completed all 96 cells; its panel worker had
144/144 model predictions and was running mixture checks. Full fits and full
panels remain queued. No candidate adoption or HDR qualification is implied.

Targeted validation-library Clippy with `-D warnings` passes: `rc=0 21s |
peak-RSS 0.67GiB | min-avail 23150MiB | peak-load 1.16`. Full-workspace
Clippy has not been rerun for this private integration.

`just lint-scripts` on i265 reports one unchanged pre-existing dependency:
`scripts/byconstruction/candidate_registry.py` references the absent historical
`zensim--rev7main/` checkout. No new confidence-owner lint finding was reported.
Do not fabricate that checkout or alter frozen candidate provenance to hide it.


Corrected binding run03 is complete for all 21 cells, each logical width 2,263;
index SHA256 `9dd58d78711449b9c699165ae8651ae632010acfd8875bed4cc4affdec73ad16`,
source `482485e8`. Source snapshot and provenance are beside the manifests.
`rc=0 3s | peak-RSS 0.37GiB | min-avail 24083MiB | peak-load 0.43`.
At 06:06:03 UTC, AIC fold panels and both mixture reports are complete;
full-data followthrough has started its unweighted control. No new confidence
fit or production adoption has occurred.


### Confidence execution binding (2026-10-27)

The existing dataset coordinator and cell wrapper now admit MlicConfidence
as its independent registered arm (`d37a01340fa9`). The unweighted full-budget
Mlic control runs first with no confidence environment. Each subsequent child
receives its own omission/seed manifest and fresh native normalizer receipt;
parent process environment remains unchanged. Cached cells and the final
historical TRAIN screen verify both hashes. Original numerical screen rules,
root, group masses, seeds and final epoch are unchanged. AIC eligibility is
not determined by that historical screen.

The attached confidence queue waits for the pinned AIC full-panel completion
before competing for the compute lock. A failed/exited dependency is reported;
a stale state file cannot authorize progress. Seven binding/wrapper/dependency
controls pass, including cross-cell substitution, mutated inputs, parent-env
isolation and stale PID refusal. Existing six ordering, 23 dataset-leg and nine
vote controls also pass. Binding-test resource: `rc=0 0s | peak-RSS 0.11GiB |
min-avail 24321MiB | peak-load 0.08`.

Native trainer source `482485e8` built on i265 with the existing release
recipe, no native-CPU flags, SHA256
`e2ffd0c0279ed31f69c2b6ba4f32656372b9de4359051765a56819a9d3ba3c82`.
Build: `rc=0 31s | peak-RSS 1.30GiB | min-avail 22883MiB | peak-load 1.01`.
A successful build does not establish unweighted parity or fit completion.


### Completed AIC fold panels: observed tradeoffs (2026-10-27)

All 96 fold fits and panels for 144 models are complete in AIC fold run03.
The following are native aggregate metrics of role-specific mean predictions,
rounded to six decimals. Held-out sources are within AIC public_train; they are
not public_test/EVAL. Fitted and held-out predictions are never pooled.

| 16×8 initialization | Fitted raw PLCC | Fitted MAE | Held-out raw PLCC | Held-out MAE | Held-out p99d |
|---|---:|---:|---:|---:|---:|
| Mlic | 0.873138 | 0.424413 | 0.864285 | 0.437603 | 0.381582 |
| Neff | 0.875803 | 0.421144 | 0.870400 | 0.428946 | 0.355456 |
| Vote | 0.877119 | 0.420159 | 0.874990 | 0.427192 | 0.344673 |
| Order | 0.874942 | 0.422257 | 0.867990 | 0.432921 | 0.368828 |

Vote leads these aggregate AIC measures, but the paired held-out MAE deltas
versus Mlic are -0.00014676607301669642, -0.0003682783077534335 and
-0.03486738937983683 across seeds 0/1/2. The advantage is dominated by seed 2.
The E21 KonFiG omission SROCC delta versus equally continued Mlic is
-0.00754409939712571 for Vote; its four-dataset mean delta is
-0.0013182671225938636. Neither is a new rejection cutoff.
All 12 arms have zero global-mixture inversions across their 12 fold/seed
cells. Local-mixture inversions remain: at 16×8 Mlic 53, Neff 50, Vote 50,
Order 46; original Mlic has 41. These are totals over cells, not unique images.
No production adoption follows from aggregate AIC gains. Full-data fixed-seed
and packed-model panels remain necessary, followed by production qualification.

Evidence owner `panels/PANEL_COMPLETE.json` binds all native outputs, including
`AIC_AGGREGATE_PANELS.json` SHA256
`d7009c02fa44f12ee33a1999acac6dfeb79ab9a150262babd981ecd9d7a35244`,
paired seeds, E21 panels and both mixture reports. No scalar selector ran.

Confidence study run01 is queued after the pinned AIC full panels, with source
`d77c968934d08d34363ada295e04a6920ef33289` and the native binary above.
Its unweighted control and 21 cells are not yet complete. The first launch
command supplied a malformed source revision and refused before queue creation;
the actual queue resolves the full committed revision and verifies ancestry.


### Native confidence equal-precision control (2026-10-27)

A process-isolated test now runs the actual trainer in baseline, equal-precision
and unequal-precision modes, each for 120 epochs × 50,000 draws. The invented
five-row fixture includes a covered rank reference, an uncovered reference,
and a separate MSE group with unchanged source mass. All 120 exported f32
checkpoint byte sequences and all 120 logged loss/validation lines (excluding
elapsed time) match baseline under equal precision. Unequal precision changes
the trajectory and final checkpoint, proving the branch is engaged.
This is exported-model/log parity, not a claim to inspect every f64 optimizer
bit, and does not replace the queued full original-Mlic reproduction.

All three native binding/training tests and four arithmetic tests pass:
`rc=0 14s | peak-RSS 0.80GiB | min-avail 23511MiB | peak-load 1.54`.
Validation-library/test Clippy with `-D warnings` passes:
`rc=0 12s | peak-RSS 0.67GiB | min-avail 23026MiB | peak-load 1.31`.
The preserved fixture checkpoints, manifests, logs and hash inventory are in
shared admission run02 `native-parity-run01/VERIFIED.json`; production trainer
bytes and the already queued experiment source remain unchanged.


### Frozen Mlic packed-control certificates (2026-10-27)

All three original full-data Mlic packed seeds pass the existing native
`bake_dial_refit certify-monotone` owner: certified/monotone true, zero free
ramps; live max-pair counts 3/2/2 for seeds 0/1/2. Seed zero was previously
certified; the same check now explicitly covers all three controls. No fit,
quality-response read, model edit or promotion occurred. This feature-coordinate
certificate does not imply pixel-space mixture monotonicity or a release pass.

The pinned models, native binary, input inventory and certificates are retained
at `/mnt/tower/output/zensim/mlic-qualification-20261027-run01/`.
`CERTIFICATES.json` binds each original result/model hash and certificate log.
Run on i265: `rc=0 0s | peak-RSS 0.03GiB | min-avail 24171MiB | peak-load 0.12`.
`just dataset-certify-bake <binary> <bake>` exposes the existing native command.
The runtime hardware constraint is explicit in the release scorecard; no
cross-host measurement can silently pass the original 9950X3D performance row.


### AIC fold source and codec tradeoffs (2026-10-27)

The native aggregate panel's 16×8 held-out role-specific mean predictions
were compared within each source/codec group against equally continued Mlic.
Neff improves the reported MAE on 16/28 sources and 7/14 codec groups; Vote on
19/28 and 8/14; Order on 17/28 and 9/14. These are descriptive sign counts,
not independent tests, votes for a winner, or substitutes for full panels.

Vote's held-out codec SROCC differences include JXL +0.007664975243634631,
JXLM -0.014924647544968361 and PNG -0.01095099604207217. Its corresponding
MAE differences are -0.007990610424413624, +0.008886379629542918 and
+0.039916098695024105. Original released codec labels are retained verbatim;
no codec implementation equivalence is inferred from those names.
Order's largest source MAE regression is PTC_S51_Ref_00 (+0.03815172114033716),
with SROCC change -0.021329076556384052. These findings reinforce source,
codec, tail and fixed-seed review without changing the registered fit grid.

The derived review and every stratum delta are stored at
`/mnt/tower/output/zensim/mlic-priority-20261027/AIC_FOLD_TRADEOFF_REVIEW.json`,
bound to the unchanged aggregate output hash. No new responses were read,
no model was selected, and no cutoff was added.


### Live epoch reporting verification — October 27, 2026, 06:38 UTC

The moving-main full-data status reader now binds logged epochs to trainers in
the verified worker process tree (`active_training` in `CURRENT_FULL.json`).
It records PID/start ticks, output directory and epoch budget; old logs cannot
create a live trainer or imply completion. Missing logs and malformed argument
bindings remain explicit. All 24 AIC adaptation/orchestration/panel controls
passed on i265: `just aic-recipe-adaptation-tests`; run-heavy rc=0, 0s,
peak RSS 0.08 GiB, minimum available 24,332 MiB, peak load 0. The actual AIC
full control was live at logged epoch 110 with no reported failure. Its
completion still requires native artifact/control verification. Frozen worker
source, recipe, inputs and gates are unchanged. Confidence remains queued at
0/21 behind the full-panel dependency.


### Full control provenance comparison repair — October 27, 2026

**06:41 UTC failure and diagnosis:** full run02 finished its 120-epoch control,
but the newly written raw-file hash check rejected different embedded run
provenance. Both final-f32 and production-f16 models are byte-identical after
the existing native owner removes only `zentrain.repro` from diagnostic copies;
all other result fields match except output paths/model hashes. Original files
are preserved. Full panels and confidence run01 refused their failed dependency
before fitting. No worker from those failed queues should be called running.
The repaired comparison follows the established E33 control method: exact
non-provenance model bytes plus matching full recipe/admission/result fields.
It retains all numerical/serving metadata, including calibration. Run02 remains
failed historical evidence; use a fresh pinned recovery plan, binding the
completed control artifacts, instead of rerunning or overwriting its outputs.

Diagnostic root: `/mnt/v/output/zensim/mlic-aic-full-20261027-control-diagnostic-run01`.
Native owner SHA256 `ce9ff24b62b940877ddc94c1a794437c0a1b07075cbe6ddcf21ad404f116b942`.
Final-f32 non-repro SHA256 on both sides:
`6a0f762eae417bae2dc9f94066c54cccc97f7d7e191e2b43219a9b314643adb7`;
packed non-repro SHA256 on both sides:
`4fcc35b13dca0f8059770cb467400c9069df1a1a9d35ded839754dd2ca541a24`.
Diagnostic run-heavy: rc=0, 0s, peak RSS 0.02 GiB, minimum available
41,037 MiB, peak load 0.78. Repaired orchestration tests on i265 pass all
25 cases, including changed recipe and serving-byte rejection: rc=0, 0s,
peak RSS 0.08 GiB, minimum available 24,319 MiB, peak load 0.15.
This is a correction to the new session's comparator, not a tolerance or
relaxation of an established numerical requirement. No model was adopted.


### Latest recovery — 2026-10-27 06:45 UTC

Active full-data root is now `/mnt/v/output/zensim/mlic-aic-full-20261027-run03`;
plan SHA256 `4684cbb8cd653f5409e292a4c6233ac74a4e797aa696f55be6a64a273a0fa4da`,
coordinator `d84cfdb235aee5d4e2f507235a9e2fcc324e461a`. Its
`CONTROL_VERIFIED.json` verifies the preserved full-budget run02 control using
exact native model bytes excluding only run-specific `zentrain.repro`, plus
matching recipe/admission/result fields. No numerical tolerance was introduced.
Matched schedules are complete. Six Neff/Vote seed trainers were verified live
in the single 24-GiB/6-job scope; Order's three seeds follow. Full coordinator
PID 2634178/start ticks 230811185; lock holder 2634157. These are dated
observations: re-resolve with `just mlic-priority-progress` before acting.

The existing full-panel source now waits on run03. Confidence's active queue is
`/mnt/v/output/zensim/mlic-confidence-20261027-run02`, with unchanged source
`d77c968934d08d34363ada295e04a6920ef33289`, trainer and corrected manifest index.
Its `QUEUED.json` binds the new full-panel dependency. It remains 0/21 and
requires its own full-budget unweighted control. Old full run02 and confidence
run01 are preserved failures; do not restart them. Completed fold evidence,
scientific inputs, strengths, seeds and all historical verdicts are unchanged.



### Full-data confidence binding extension — registered October 27, 2026

Before any confidence full-data candidate fit, bind all three original Mlic
full-data authorities to the already-admitted `full` linking/rating scope.
Use original init/sample seed pairs (1101,101), (1103,100000101),
(1107,200000101), 120 epochs × 50,000 draws, final119, original groups/masses,
logical width 2263, and the same native contrast loss. No new map fit, source
selection, loss tuning, factor combination or AIC response read is authorized
by this metadata/manifest step. Full native maps were already purpose-bound
and admitted. Preserve the 21-cell omission registry byte-for-byte; a separate
`mlic-full-contrast-manifests-v1` registry contains only full-s0/s1/s2.

The generator requires the exact full-authority index hash, verifies each
original authority hash and rejects missing/duplicate seeds, omission scopes,
changed seed streams and non-full routes. It uses the existing native-row,
complete-reference and map binding checks. `READ_EVENT.json` records any TRAIN
target/row reads. Existing confidence omission fitting remains first; the full
extension is preparation for the subsequent common AIC comparison, not evidence
that confidence has fit, qualified or been adopted.


### Full confidence manifest preparation — October 27, 2026

Three full-data manifests are verified under the shared admission run02
`full-contrast-bindings-run01/`. Source `2f9268f2`; manifest index SHA256
`d73a3422de00f6150f50e934dc6821ab7894f6eb85fe4a3e732fefbacc088bfc`;
original full-authority index SHA256
`7dbf9cb8deabdfc4b42b72637dd17112a41281e01094a5fe65ae987525f58923`.
`SOURCE.py`, `READ_EVENT.json` and `VERIFIED.json` preserve source/exposure and
all three seed bindings. Each full seed covers 8,339/10,590 human rows,
210/210 LIVE-MD and 320/320 Colourlab rows; uncovered references retain the
original loss. Complete-reference coverage checks pass. The 21 omission
manifests are byte-unchanged. Full native normalizers and fits have **not** run.

Eight binding/wrapper/dependency tests pass on i265 using the existing
confidence-admission Python environment (system Python lacks pyarrow).
Generator run-heavy: rc=0, 3s, peak RSS 0.36 GiB, minimum available 24,006 MiB,
peak load 0.06. This prepares the original full-data confidence fits; their
launcher must validate the separate full registry, use child-only native
confidence environment variables, verify normalizer receipts and final119,
then clear the confidence factors for the common AIC continuation. Do not feed
the full registry to the frozen 21-cell omission queue or modify that queue.
All full-fit/AIC steps retain the original seeds, strengths and panel policy.



### Confidence full-data launch owner — October 27, 2026

`scripts/byconstruction/mlic_confidence_full.py` reuses the existing strict
full-fit command and final-checkpoint/packing verifier. Before any fit it
requires a pinned plan/runner/imported owners, the completed independent
confidence screen, the entire 42-cell paired omission inventory, and every
screen input hash. A historical rejection is retained and does not exclude
the arm from the new AIC study. All three original full seeds are required.

Only each native fit subprocess gets its own manifest/hash/fresh-normalizer
receipt environment; inherited confidence variables are rejected. Certification
and later callers retain the clean parent environment. Completion requires
final119, exact recipe/seed/root/packed identity, native normalizer receipt and
native monotonic certificate for all three models. It does not launch AIC or
promote a model. Plan creation and execution await actual omission completion;
no missing completion hash is invented. Common AIC continuation must use the
existing warm-start owner with original Mlic losses and no confidence factors.


### Confidence full-fit launcher — October 27, 2026

Committed owner `88155360` adds `just confidence-full-fit <plan>`. It requires
completed paired omission evidence and frozen source/binary/registry hashes;
it does not exclude confidence based on the historical SROCC verdict. It
launches all three original full seeds through the existing strict owner,
checks final119/packing and native normalizer receipts, then certifies each
packed model. Confidence environment variables are child-only, and inherited
ones are rejected. It does not perform common AIC continuation or promotion.

Four boundary tests pass on i265: complete evidence despite historical
rejection, missing plan pins, full-versus-omission/swapped-seed rejection,
and child environment/receipt isolation. Run-heavy rc=0, 1s, peak RSS
0.07 GiB, minimum available 24,285 MiB, peak load 0.21. The actual three full
manifests also pass the launch consumer's checks: rc=0, 0s, peak RSS 0.07 GiB,
minimum available 24,291 MiB, peak load 0.23. This is preparation, not an
executed full-confidence fit. A plan with real omission completion hashes
cannot be frozen until that study completes. The next implementation step is
the separate confidence AIC continuation/panel extension, preserving the
running four-recipe study and its frozen sources.



### Separate confidence AIC fold extension — registered October 27, 2026

The new `aic-confidence-adaptation-plan-v1` contract contains exactly Mlic and
MlicConfidence. It requires 48 continuation cells: 24 hash-verified original
Mlic reuses and 24 confidence fits (four original omissions × three seeds ×
two strengths). Its panel includes the 24 initial models plus 48 continuations,
72 total. The original four-arm contract still requires its exact four recipes,
96 continuations and 144 panel models. Neither contract can substitute the
other's smaller/larger inventory. Running source snapshots are unchanged.

The confidence extension reuses the same zero-step prediction parity,
strict native warm-start owner, original Mlic root/losses, fresh Adam at 1e-5,
50,000 draws/epoch, 8×4 and 16×8 strengths, source-fold rotation and final-epoch
checks. Only the starting model differs. It verifies each confidence manifest
and native normalizer receipt against the completed omission record and pinned
baseline registry. Inherited native or wrapper confidence environment variables
are rejected before any continuation or panel read; the common recipe cannot
accidentally retain its starting model's confidence factor.

The same panel owner preserves fitted/held-out separation, native statistics,
source/codec/ladder diagnostics, seed contrasts, whole-source bootstrap and
both mixture tests. It compares confidence against its own initialization and
matched Mlic continuation, with no SROCC selector. No extension plan is queued
until real confidence omission completion and artifact hashes exist. The
separate full-confidence AIC continuation/panel extension remains required.

All 28 AIC orchestration/panel tests pass on i265, including rejection of mixed
protocol inventories, an incomplete original study masquerading as an extension,
and inherited confidence factors. Run-heavy rc=0, 0s, peak RSS 0.12 GiB,
minimum available 24,299 MiB, peak load 0.12. This validates orchestration
boundaries; confidence adaptation has not run yet.



### Separate confidence AIC full-data extension — registered October 27, 2026

`aic-confidence-full-plan-v1` is exactly Mlic plus MlicConfidence. It requires
six initial models and twelve continuations: the three original Mlic full
models, three completed independent confidence full fits, and both common AIC
strengths for all seeds. The six Mlic continuations are reused from the complete
four-arm full-data study, with their completion-plan identity, three-seed
inventory, recipe/strength, result/input hashes and final model checks. Only
six new confidence continuations are needed. No best-seed selection or SROCC
eligibility cutoff is introduced.

Confidence initial reuse requires its completed full-fit plan, every pinned
prerequisite, exact recipe/final119/packed identity and per-seed manifest/native
normalizer receipt. It does not refit the base model inside the continuation
owner. Common continuation rejects inherited confidence factors and keeps the
same original Mlic root, fresh Adam, fixed budgets and zero-step prediction
identity controls. Order matching is performed only in the original four-arm
protocol. The original protocol remains 12 initial models plus 24 continuations;
its smaller confidence counterpart cannot satisfy those counts.

The same full panel owner covers 18 models × two representations × five
populations = 180 prediction outputs, plus both mixture checks and the existing
statistics/source/codec/seed/whole-source uncertainty panels. The original
four-arm panel remains 36 models / 360 predictions. Serving-f16 versus final-f32
still includes calibration and is not isolated packing error. Running plans,
source snapshots and results remain immutable. No confidence extension is yet
queued: exact completed omission/full-fit hashes must exist first.

All 32 AIC orchestration/panel controls pass on i265, including refusal of
missing/duplicate seeds, mismatched protocol inventories, wrong reuse strength,
changed reuse evidence and partial full-panel completion. Run-heavy rc=0, 0s,
peak RSS 0.12 GiB, minimum available 24,296 MiB, peak load 0.16. These are
orchestration controls, not evidence of completed confidence fits or adoption.



### Attached confidence followthrough supervisor — October 27, 2026

`confidence_followthrough.py` freezes its own source/config before waiting for
the independent confidence omission study. It checks the live PID/start time
while waiting, refuses failed/exited prerequisites, and verifies actual complete
paired evidence and source/trainer identities before constructing plans. A
historical rejection does not stop the new AIC comparison. Missing future
completion hashes are never invented: they are bound only when the corresponding
stage completes, under the already registered recipes and budgets.

Execution is serial: 48 matched fold continuation cells (24 Mlic reuses),
72-model fitted/held-out panels, three original full-confidence fits, twelve
full continuations (six Mlic reuses), then 180 full-panel prediction outputs
and both mixture tests. Each existing heavy owner acquires the shared lock and
run-heavy scope; the waiting supervisor holds neither. All commands/logs and
terminal failures are durable in its run directory. No detached agent or
background daemon is created. No automatic model adoption occurs.

Four supervisor tests pass on i265: live-wait versus PID reuse/death,
historical rejection with complete evidence, unchanged common fold recipe,
and correct full-control reuse/budgets. Run-heavy rc=0, 0s, peak RSS 0.07 GiB,
minimum available 24,297 MiB, peak load 0.03. Full native execution remains
pending the original AIC and confidence omission dependencies.



### Live serial confidence followthrough — October 27, 2026, 07:09 UTC

The attached supervisor is running at
`/mnt/v/output/zensim/mlic-confidence-followthrough-20261027-run01`.
Frozen source `5474e6b4d99ad432830e2d89fca900a6eec5f97c`; CONFIG.json SHA256
`60cb9ed63f18ef6e8f0af6f27adbac3fb1e8f78f63c77a5c6ec101279eb557a3`.
Verified worker PID 2703508/start ticks 230955857 waits for confidence run02's
complete omission evidence, without taking the compute lock. It then runs the
registered fold fits/panels, full confidence fits and full AIC fits/panels in
sequence. `just mlic-priority-progress` includes its live status. Read its
STATE.json/CURRENT.json and actual process tree before any intervention; never
launch duplicate stages while this supervisor is live. Its source snapshots,
config, stage plans and logs remain preserved on failure.

Five supervisor tests pass, including real process status and stale PID reuse:
run-heavy rc=0, 0s, peak RSS 0.07 GiB, minimum available 24,298 MiB, peak load
0.02 on i265. No confidence fit has started; the supervisor waits behind the
original AIC full pipeline and confidence omission queue. As of this observation,
all six original full trainers were advancing at logged epochs 72–88/120,
with no reported failure. This timestamped observation does not replace a live
check. No candidate is adopted and the production/HDR objective remains active.


### October 27: isolate original-Mlic parameter packing on TRAIN

The registered diagnostic `mlic-parameter-packing-train-v1` compares all three
original full-data seeds on the frozen TRAIN calibration table (SHA256
`07e2f4b71fe0ccc67848f2896344e6d0b6a255b0e91a7da9062d471ea3e11100`).
It does not select a model, change a gate, read EVAL or claim runtime speed.
Owner: `scripts/byconstruction/mlic_precision_audit.py`; entry point:
`just mlic-precision-audit <plan> <fresh-output>`, under the shared compute lock
and `run-heavy` on an eligible idle host.

Inputs, source and native binary are SHA256-bound before prediction. The
existing native pack owner creates an f32 control with zero zerobias, protected
last layer and no pruning. The existing strip owner removes only output spline
metadata from that control and the original production-f16 bake. Both use the
production score predictor, which builds derived inputs correctly; the raw
predictor explicitly refuses derived-input models and is not substituted.
Original pack logs must confirm zero removed weights and no pruned columns.
The f32 control must exactly reproduce every original dense-model prediction
before any f16 difference is accepted. Report all rows, mean signed/absolute
delta, RMSE, nearest-rank p99 and maximum absolute delta in pre-spline score
units. This isolates this parameter-packing path on this TRAIN population;
it does not measure feature extraction precision or f64-reference error.

Preregistration SHA256:
`e9f31fafcab9575e49d2afed6826ae74f328f0e033f0cd216ed3ba6d7df14538`.
Artifacts: shared output `zensim/mlic-precision-20261027-run01/`, preserving
`PLAN.json`, `SOURCE.py`, original model/log copies and all native command logs.
Execution waits for the existing build to leave i265 before acquiring its
compute lock. Results are pending; no precision bound is claimed yet.

### October 27, 07:21 UTC: isolated parameter-packing result

The registered three-seed TRAIN audit completed. All three stripped f32 controls
are byte-identical to their original dense models; every one of the 12,163
predictions per seed is exactly identical. All artifact hashes and paired TSV
coverage were independently reread after completion. Before output calibration:

| Seed | Mean absolute delta | RMSE | p99 absolute (nearest rank) | Maximum absolute |
|---|---:|---:|---:|---:|
| 0 | 0.000924370962 | 0.001041982565 | 0.002151489258 | 0.003734588623 |
| 1 | 0.000397940701 | 0.000531791949 | 0.001613616943 | 0.003211975098 |
| 2 | 0.000917169125 | 0.001031623618 | 0.002243041992 | 0.003612518311 |

These are native f32-inference score-unit changes caused by the original f16
parameter pack with the spline removed, on the frozen TRAIN calibration
population. They are not native HDR precision, feature extraction error,
post-calibration bounds or speed qualification. No candidate is adopted.
Evidence: shared `zensim/mlic-precision-20261027-run01/results/COMPLETE.json`,
with sibling `VERIFIED.json` at the run root.
Completion SHA256: `47e3ab97651a81b5a9caa4825eeee291b459f6c1d150954510f211f6ae09dd7b`.

Run-heavy on i265: rc=0, 7s, peak RSS 0.47 GiB, minimum available 23,044 MiB, peak load 6.13. The duration is an execution receipt, not a runtime benchmark.

### October 27: native input baseline at Rev7

`just mlic-native-input-tests` preserves the five legacy native-HDR tests and
adds a separate Rev7 child-process check of the supported 720-feature HDR
entry. Running the legacy append-family tests wholesale with
`ZENSIM_FORMULA_REV=7` fails before color arithmetic because that feature family
is deliberately unsupported. The rejection and old test expectations remain
unchanged; this failed invocation is retained in `native-hdr-rev7.log`.

The new check uses a 97×65 image with native u16 codes and a one-code red-channel
distortion. For PQ and HLG, each in sRGB, Display-P3 and BT.2020, the native-u16
features match the equivalent normalized-f32 code features exactly, remain
finite, and differ from identity. All six cases pass. This proves the tested
input-container and supported-entry behavior; it does not prove every possible
code step survives display clamps, establish a new f64 error bound, qualify
Mlic's full model/input composition, or show HDR perceptual quality or speed.
The original five tests also pass, retaining their independent f64 color/display
reference and prepared-map checks under their original arithmetic.

Evidence: shared output `zensim/mlic-precision-20261027-run01/`, including
`native-hdr-build-initial.log`, `native-hdr-rev7.log` and
`native-hdr-supported-rev7-run02.log`. The first build/default-five-test run:
run-heavy rc=0, 147s, peak RSS 1.92 GiB, minimum available 21,921 MiB, peak load
3.80. The final six-test run including explicit Rev7 child: rc=0, 9s, peak RSS
0.59 GiB, minimum available 23,443 MiB, peak load 1.41. These are validation
resource receipts, not performance qualification. No frontend arithmetic,
feature identity, trained artifact or running experiment was changed.

Targeted clippy (`-p zensim --test native_hdr_color --features custom-profiles -- -D warnings`) also passes: run-heavy rc=0, 10s, peak RSS 0.64 GiB, minimum available 23,212 MiB, peak load 1.42. No source warnings were suppressed.

### October 27: certificates for completed Neff/Vote full initial models

All six completed full-data initial models (Neff and Vote, seeds 0/1/2) pass
the existing native `bake_dial_refit certify-monotone --pin 100` owner. Inputs
were bound to their final119/120-epoch fit receipts and packed-model hashes
before checking. Every certificate reports zero free amplitude and free ramps.
Max-pair counts are Neff 2/1/2 and Vote 3/1/2. This is coordinate monotonicity
of those initial models, not pixel-mixture behavior, AIC continuation
qualification or a product-adoption decision. Order and continued models are
not covered by this receipt.

Artifacts: shared output `zensim/mlic-full-initial-certificates-20261027-run01/`
(`PLAN.json`, six frozen model copies/native logs, `COMPLETE.json`). Completion
SHA256 `3574cc7ede2010b83ae90393d291bc1b772b2613461f2a08cef5011dc64e82d6`.
The coordinator reread every copied model hash after completion. Run-heavy
on i265: rc=0, 1s, peak RSS 0.02 GiB, minimum available 23,964 MiB, peak load
1.25. This is a validation resource receipt, not a runtime benchmark.

### October 27: obsolete research entry points now fail before work

Owner request: block outdated paths that waste compute, beginning with silent
Rev6 restamping. `historical_guard.py` now denies new default invocations of
`serve_parity.py`, `candidate_registry.py`, `perf_grid.sh`, `perf_bits.sh`,
`map_study.sh`, `jxl_steer_rd.sh`, `jxl_reach.sh` and `dataset_loop.sh`. The guard
runs before argument expansion requiring data paths, scientific dependencies,
output creation or subprocess work. Exit 64 explains the obsolete assumption
and points at the current session/priority owner. The dated registry includes
a deleted checkout; the old parity owner always restamps to Rev6; the shell
studies carry fixed historical model, placement, judge or output defaults.

Intentional reproduction requires `--historical-replay` as the first argument
to the script. A trailing flag or environment setting does not authorize it.
That opt-in does not waive science, input freezing or resource rules. Historical
benchmark/map/RD launchers still refuse dev, and all guarded paths refuse
r7900x. Historical regression tests and valid old-model serving remain enabled.
Frozen as-run source snapshots, current AIC/confidence queues, artifacts and
existing numerical gates are unchanged. The parent Mdata loop propagates its
explicit replay authorization to its historical JXL child.

`just byc-historical-guard-tests` checks all eight entry points without model
files, optional Python dependencies or positional arguments; each must fail
with the guard's diagnostic instead of loading data or launching work. It also
checks explicit first-argument handling, host restrictions and unknown-entry
refusal. This is a targeted guard inventory, not a claim that every historical
script in the repository has been audited. New research launchers should use
the current frozen-plan owners instead of copying these dated defaults.

Guard verification on i265: all four tests pass, including actual refusal of all eight entrypoints under dependency-free invocations. Run-heavy rc=0, 0s, peak RSS 0.02 GiB, minimum available 23,824 MiB, peak load 0.08. Six shell launchers pass `bash -n`; no experiment was launched by these tests. Logs: shared output `zensim/obsolete-entrypoint-guards-20261027/tests.log`.

### October 27: full AIC fits and remaining coordinate certificates complete

Full run03 finished all 12 initial models and 24 continuations. The coordinator
reread both model hashes and every recorded input hash for all 36 records in
`FULL_FITS_COMPLETE.json`. Full-data panels subsequently started and reached
350/360 population predictions at the recorded check; this is not panel or
mixture completion. No failures were recorded. The independent confidence
omission queue still waits for full-panel completion.

The existing native `certify-monotone --pin 100` owner also passed for all
27 remaining packed models: three Order initial models and all 24 continued
models. Every certificate reports monotone/certified true and zero free
amplitude/ramps. Earlier receipts cover the nine Mlic/Neff/Vote initial models.
This completes coordinate-certificate coverage of the 36 models, not pixel
monotonicity, mixture behavior, runtime qualification or candidate adoption.

Evidence: shared output `zensim/mlic-full-certificates-20261027-run01/`, with
frozen PLAN, model copies, native logs and COMPLETE. Completion SHA256:
`d54070c71ee3628121803b771e7011132b51fb119b4891fbe9ab1a66d96bbc05`.
All copied model and log hashes were reread after completion. Certificate
run-heavy receipt on i265: rc=0, 3s, peak RSS 0.03 GiB, minimum available
23,793 MiB, peak load 0.16. The dev full-fit receipt is rc=0, 4,822s,
peak RSS 2.55 GiB, minimum available 26,792 MiB, peak load 9.44.
These are execution resource receipts, not runtime benchmarks.

### October 27: full-data cell statistics expose an ordering tradeoff

The 360 native cell panels are complete; source/codec/interval and mixture
work remains in progress. These are partial study results, not a selection.
For the registered serving seed0, production-f16-calibrated 16×8 arms have
AIC MAE / p99 distance: Mlic 0.428860 / 0.383393; Neff 0.434599 / 0.391138;
Vote 0.429270 / 0.374389; Order 0.421418 / 0.363907. Order's AIC p99 distances
are lower than Mlic's for all three seeds. AIC is in sample here.

Order also has worse seed0 MAE than Mlic on every other TRAIN panel:
KADID 9.039121 vs 8.763874; TID2013 6.322748 vs 6.132106;
KonFiG 13.380590 vs 12.363111; CID22-A25 8.075303 vs 8.020150.
This prevents interpreting its AIC improvement as a universal improvement.
Full native statistics, geometry and per-seed records are retained in shared
`mlic-priority-20261027/FULL_CELL_EARLY_REVIEW.json`, bound to the current
`panels/CELL_PANELS.json` hash. No candidate is adopted or excluded by a new
SROCC cutoff; wait for the remaining registered panels and confidence arm.

The existing `featcanon_rev7_parity` tests also passed on i265 at source
`32ea661e768a`: two tests, zero failures/ignores. They check the full Rev7
feature vector across available dispatch permutations at five geometries,
including remainder columns and width4133, plus unchanged revision-rounding
bounds. This does not establish candidate pixel/cache score parity, another
architecture's behavior, or performance. Receipt:
`mlic-precision-20261027-run01/REV7_FEATURE_PARITY_VERIFIED.json` and its log.
Run-heavy rc=0, 47s, peak RSS 2.16 GiB, minimum available 21,557 MiB,
peak load 1.02; execution resource receipt, not a runtime benchmark.

### October 27: full-data source/codec and seed review

All 69,552 aggregate native panels and registered source intervals are now
written; pixel-mixture checks remain live. The descriptive review compares
16×8 packed/calibrated arms with Mlic16×8 using the mean predictions of three
fixed seeds. This stability diagnostic is distinct from serving seed0 and
from averaging three seed-level statistics. All AIC sources are in sample.

| Arm | AIC sources with lower MAE | AIC codecs with lower MAE | Aggregate AIC MAE delta |
|---|---:|---:|---:|
| Neff16×8 | 10/28 | 3/14 | +0.007089 |
| Vote16×8 | 16/28 | 3/14 | +0.002324 |
| Order16×8 | 13/28 | 6/14 | −0.002510 |

Every count has complete defined coverage. Order's aggregate MAE deltas on
KADID/TID2013/KonFiG/CID22-A25 are +0.204870/+0.115273/+0.435511/−0.053202.
Thus its seed0 regression on all four other populations is not identical to
the three-seed aggregate: CID22-A25 improves in the latter. Vote's corresponding
deltas are −0.043590/+0.016107/−0.082979/−0.051431. These mixed changes do not
establish a stable replacement, and no arm is rejected by a new SROCC gate.

Paired individual-seed AIC MAE deltas for Order are
−0.007442/+0.002467/−0.002893; Vote +0.000410/+0.016214/−0.007568;
Neff +0.005739/+0.016832/−0.000588. The registered 2,000 whole-source bootstrap
intervals are descriptive, conditional on these fitted models, and in-sample;
they do not measure independent test uncertainty. Preserve them alongside the
full panels rather than treating interval crossing as a selection threshold.

Evidence: shared `zensim/mlic-full-panel-review-20261027-run01/`, including
five source-artifact copies and hashes, full per-source/codec deltas in
`REVIEW.json` (SHA256 `823461f1d650307d8df519e5bde82227f8ca0d6a6909a89fde804802a7022b63`), and the exact review command/log. The i265
run-heavy receipt is rc=0, 2s, peak RSS 0.17 GiB, minimum available
23,819 MiB, peak load 0.15. No new training or calibration was performed.
Next remains the registered mixture completion and independent confidence
comparison, followed by outstanding production qualification; no adoption.

### October 27: AIC-2026 public_train full panels complete; confidence control live

Full run03 completed all 360 population predictions, 69,552 aggregate panels,
paired-seed contrasts, source intervals and both pixel-mixture sets. Each
mixture set covers 36 models × two representations = 72 cells. Every global
mixture cell has zero inversions. Local mixtures still have reversals: for
packed/calibrated seed0 at 16×8, inversion counts are Mlic 3, Neff 3, Vote 3,
Order 4 (1,152 comparisons per cell). Their maximum score rises are
5.644995, 5.004366, 6.430682 and 3.900804 respectively. Coordinate-monotone
certificates do not eliminate these pixel-mixture failures. No adoption.

The coordinator reread all 1,476 inventory-file hashes, source hashes and native
tool hashes, plus completion/plan/fold dependencies. Shared verification:
`mlic-priority-20261027/FULL_PANEL_VERIFIED.json`; full completion SHA256
`fd787030c17f916c06dc8207b30dd353873e983befdea305754cbbb6ce0f805a`.
Panel run-heavy: rc=0, 745s, peak RSS 0.65 GiB, minimum available 43,435 MiB,
peak load 5.37. This is an evaluation resource receipt, not runtime qualification.

The confidence run02 dependency completed and its full-budget unweighted
control started on dev; at 08:19 UTC its live worker was 3388770/start 231371214,
lock holder 3388749. Re-resolve with `just mlic-priority-progress`. The control
must match 7,869 original predictions before any of the 21 confidence fits.

Separately, the existing near-identity owner is running on i265 for all 12
seed0 initial/continued models, using 24 frozen TRAIN images, unchanged
N1/N2 ≥99 and N3 ≥122/144 contracts, revision 7/v3. Shared root:
`mlic-nearid-20261027-run03`; PLAN SHA256
`b43474f3c414c92a12efc279eda8b781e73bbc5cb14967c08b94ae611fc3e0e7`.
Its STATE records real PID/start ticks and per-model results. First two models
Mlic/Neff pass all three gates; remaining models were still running at entry.
The frozen source packet is unchanged apart from explicit image-path remapping;
all model and image hashes were verified. No quality labels are read.

Run01 refused uppercase CLI labels before measurements. Run02 failed parsing
its command before execution. Both are preserved, with no scientific results.
Run03 uses lowercase labels and the identical native binary/model/image bytes.
The reusable `just dataset-nearid-candidate` now validates lowercase labels
before invoking native work, then uses the same native and reporting owners.
Invalid-label refusal and command expansion were checked. Native build receipt:
rc=0, 15s, peak RSS 0.84 GiB, minimum available 22,950 MiB, peak load 0.41.

### October 27: all twelve near-identity checks complete

All twelve registered seed0 initial/continued models pass the unchanged N1,
N2 and N3 near-identity gates on the 24-reference TRAIN packet. Each model
has 648 measurement rows. N1 requires both one-pixel rungs ≥99 on every
reference; N2 requires a nonidentical score ≥99 on every reference; N3 requires
at least 122 of 144 nonincreasing ladders. Observed N3 counts are:

| Recipe | Initial | 8×4 | 16×8 |
|---|---:|---:|---:|
| Mlic | 143 | 143 | 143 |
| Neff | 140 | 139 | 139 |
| Vote | 143 | 143 | 143 |
| Order | 138 | 138 | 138 |

The coordinator checked all twelve metadata/model/packet/revision bindings,
648-row counts and gate results, then hashed every result/log file. Shared
receipt: `mlic-nearid-20261027-run03/VERIFIED.json`; completion SHA256
`c70c483ff5e30ac2e1a2b0eb59e94e8d915906ff4ce5b2b6fa7ab4ee687b6b1a`.
Run-heavy on i265: rc=0, 136s, peak RSS 0.25 GiB, minimum available
23,730 MiB, peak load 0.96. This is validation resource evidence, not timing
qualification. N1–N3 passing does not clear the local-mixture findings,
other production gates, runtime qualification or stable adoption.

### October 27: explicit Rev7 synthetic serving parity

The existing `serve_custom_bake` probe now has a separate `--prodqual-rev7`
entry, exposed by `just dataset-serving-parity-rev7`. The original `--prodqual`
still requires revision 5. Both entries require their process and bake revisions
to match exactly; no model is restamped. Three actual refusal controls cover
Rev7 process through the old entry, Rev5 process through the new entry, and
Rev5 model through the new entry. All refuse before synthetic measurements.
The arithmetic, geometries, identity bands and exact-bit comparisons are unchanged.

All twelve frozen initial/continued seed0 models completed the new probe on
i265: four geometries × seven bit-truncation levels × eight available dispatch
permutations per model, 2,688 model/pair/permutation rows. Direct pixel scores,
prepared-cache scores and identity-aware feature scores agree exactly; consumed
and cached features agree exactly; cross-permutation row mismatches are zero.
All scores are finite, pixel identity is exactly 100, feature identity remains
in the unchanged 97.5–100 band, and no tested distortion exceeds 100. The original
synthetic ladder observations remain report-only. The raw artifact's `seed`
field is the argument index, not the fitting seed: all inputs here are seed0.
No WASM, ARM, AVX-512, real-corpus or speed claim follows from this i265 run.

Shared evidence: `zensim/mlic-serving-parity-20261027-run01/` with preregistered
PLAN, frozen source/binary binding, NATIVE, SUMMARY, refusal logs and VERIFIED.
The coordinator checked source/model/binary hashes and every summary comparison.
Build run-heavy: rc=0, 14s, peak RSS 0.84 GiB, minimum available 23,132 MiB, peak load 0.48.
Measurement run-heavy: rc=0, 17s, peak RSS 0.24 GiB, minimum available 23,693 MiB,
peak load 0.96. Resource receipts are not runtime benchmarks. Rust formatting
checks pass. Confidence control training continues; no candidate is adopted.

## October 27 — AIC-only historical evidence and matched continuation diagnostic

The saved `aic2026train/panel_mohammadi.json` and
`rust/folds_train/T_full/train.log` were inspected again. The historical T fit
used one AIC-2026 public_train group, 3,920 crops from 28 sources, 60 epochs
of 20,000 pairs, initial learning rate 0.001, and 844 fx1g inputs. Its saved
in-sample panel is SROCC 0.9742999206247418, PLCC 0.9641750402938799,
Kendall 0.8587809369174624 and Z-RMSE 0.2652668310857874. The seven-source-fold
panel is SROCC 0.9526610561260485, PLCC 0.9332146636245136, Kendall
0.8113205128561868 and Z-RMSE 0.3593193448677381. These are the saved panel's
statistics and conventions, not newly computed raw calibration errors.
The training receipt explicitly has `qualified_provenance=false` and unknown
feature-set identity. This is historical research evidence, not a qualified
Rev7 Mlic control. AIC-3 and AIC-2026 public_test are not part of this read.

This evidence changes the next mechanism question. The current mixed-data
continuations do not establish an AIC-specific fit ceiling. Their `dev_curve`
tracks the older development groups; AIC is explicitly not evaluated during
training. `train_and_select` retains the final checkpoint but deletes its
checkpoint directory, so those outputs do not provide an AIC convergence
trajectory. Do not infer convergence from the rounded mixed training loss.

### Prospective bounded screen: population versus continuation budget

Before any new fit, register the following paired TRAIN-only diagnostic in a
fresh directory with exact source, binary, table, authority and checkpoint
hashes. This section fixes the design; it does not claim a launch.

- Start both arms from original Mlic full seed 0 `refit/last.bin`, keeping its
  Rev7 feature identities, Mlic architecture and parameter constraints.
- Use the admitted D6 all-public_train table and unchanged target orientation.
  One arm retains the existing full-data group mixture plus AIC weight 16;
  the other updates on AIC alone. The latter is **AIC-only continuation of a
  mixed-data pretrained model**, not training from scratch on AIC alone.
- Both arms use fresh optimizer state, constant learning rate 0.00001, the
  existing seed-0 streams and 50,000 draws per epoch. Run 60 epochs per arm;
  preserve checkpoints after epochs 7 and 59. No adaptive early stopping or
  best-checkpoint selection. Compare matched draw budgets; actual AIC
  exposure differs by design and must be reported from native sampling receipts.
- Require the mixed arm's epoch-7 checkpoint to reproduce the existing
  eight-epoch Mlic continuation exactly after removing only run-specific
  reproduction metadata with the existing native owner. Failure blocks
  interpretation of the matched contrast, not preservation of its artifacts.
- Evaluate both fixed checkpoints and the shared initial checkpoint using
  existing native prediction and panel owners on the five already admitted
  TRAIN reporting populations. Report raw-f32 panels first: pooled, source,
  codec and ladder ranks; calibration/scatter/tails; paired-source differences;
  global and local mixtures. Do not silently include older-data calibration
  in an artifact called an AIC-only fit. Packing/calibration is a separate
  explicitly named stage if the diagnostic advances.
- This seed-0 screen diagnoses a mechanism; it cannot establish stability or
  adoption. Review full panels without a new SROCC cutoff. Any advancing
  recipe receives a separate preregistered three-seed/source-fold comparison
  and all unchanged production gates. No EVAL/public_test read is authorized.

The existing frozen continuation authority accepts only 0/4/8 epochs and
retains its original groups. It must not be edited or passed a relabeled
60-epoch/AIC-only command. Extend the canonical admission/continuation owners
with a separate explicit diagnostic authority and refusal tests; preserve
legacy/default arithmetic and the frozen source trees. Native and Python
admission must agree on excluded groups, checkpoint, budget and D6 inputs.
Retain intermediate artifacts for the new diagnostic. The active confidence
control/omission/follow-through sequence keeps its compute priority.

### Matched-continuation admission implemented and queued (2026-10-27, 09:21 UTC)

Ownership returned to the original Claude session at 08:51 UTC (tag
`zensim-claude-handoff-20261027`). The confidence run02 full-budget
unweighted control then matched all 7,869 original predictions byte for byte
(`CONTROL_VERIFIED.json`); its 21 omission cells started at 08:53 UTC.

The registered diagnostic above now has its admission, in `fb3b0a79`:

- **Continuation on main.** The owner-approved warm-start trainer delta from
  PR 73 (`87864f94`) was applied to main from the branch's own diffs; four
  overlaps with confidence, matched-codec and vote code keep both sides. The
  frozen 0/4/8-epoch contract is unchanged. A checkpoint without its
  continuation binding is now refused: before, a version-2 authority plus
  `--init-from` at 120 epochs passed admission.
- **Separate version-4 authority**, mirrored natively and in Python, bound to
  its unmodified version-2 base by pin. Mixed+AIC16 keeps every base group.
  AIC-only keeps exactly the base's validation-only groups and lists every
  training group (positive train weight, base order) as excluded, so the
  logged development curve stays and no older-data leg updates. Both arms:
  60 epochs, LR 1e-5 constant, fresh Adam, AIC weight 16, every-epoch dumps;
  epochs 7 and 59 are the fixed assessment checkpoints.
- **Coordinator** `scripts/byconstruction/aic_matched_continuation.py`
  (plan/queue/run/status). It reuses the full-panel owner's five-population
  and pixel-mixture blocks, extracted as functions with unchanged behavior.
  Group weights are sampling probabilities only, so the AIC-only arm's
  single leg draws all 50,000 pairs per epoch; actual exposure is read from
  each checkpoint's `zentrain.sample_coverage` through the new
  `bake_dial_refit read-meta`.
- **Controls:** mixed epoch 7 versus `MlicFT16x8` seed 0 `refit/last.bin`
  (`dee9b5de…`) after the native strip of `zentrain.repro`, plus 14 shared
  result fields; Initial and mixed-epoch-7 predictions versus the stored full
  study predictions, byte for byte.

Checks on i265, with source identical to `fb3b0a79`: rustfmt; 348 library
tests (2 ignored), 54 trainer CLI tests, the read-meta test and strict clippy
pass (run-heavy rc=0, 36 s, peak RSS 1.59 GiB). Python: 6 authority, 5
coordinator, 32 AIC owner, 6 subjective, 23 dataset-leg, 9 vote and 2 vote
full-route tests pass. A dry run of both arms through the frozen source's
Python admission, stopped before training, passes; the mixed arm's trainer
argv equals the reference's except epochs, dumps, output and authority paths,
and its keep list and derived-input declaration hash to the reference's.

Plan `/mnt/v/output/zensim/mlic-matched-continuation-20261027-run01/PLAN.json`,
SHA256 `bf4cf8e39e92a965be01e1895009d0d7d11325875045853435f7c206ec2c32f2`;
trainer `d5275ca5…` (built on i265, receipt `TRAINER_BUILD.json`); panel and
prediction tools are the fold study's frozen run-05 binaries. The queue waits,
holding no lock, until the confidence followthrough completes, then takes
`heavy.lock` with run-heavy (24 GiB, 4 jobs). Queue PID 3572059, start ticks
231746803, in herdr tab `matched-continuation-queue`. No result exists yet.

### Local pixel mixtures in context (2026-10-27, existing scores, no new fit)

The full-data candidates' remaining local mixture reversals are small next to
the current default's. On the same 1,152-row local mixture set (48 TRAIN
references × 24 distortion combinations, `mixture/manifest.json`), the shipped
B bake reverses 148 rows (131 by more than one point), up to +44.21 points.
Its worst combination is blur1.2+chroma8, reversed on all 48 references, then
blur1.2+chroma16 (41) and blur1.2+luma4 (19): adding colour noise to a blurred
image raises B's score, the known colour-noise inversion. B's recipe out of
sample (base-c, 12 E21 cells) reverses 86–171 rows per cell.

Every seed-0 full-data candidate (Mlic, Neff, Vote, Order; initial, 8×4 and
16×8; final f32 and packed) reverses 2–4 rows, up to +3.00 to +6.43 points.
They concentrate in poster12+luma4 (2–3 references) and poster12+chroma8 (one
reference); none is a blur+colour-noise row. Their E21 cells show the same
pattern (Mlic 2–5 per cell, maximum rise 6.17). The global mixture set has no
reversal for any candidate. Scores: `mixture/scores_B.json`,
`mixture/scores_basec_*`, and the full study's `panels/mixture.json`; counts
from `mixture_test.report`. The residual posterization rows remain a real
limitation to report with any adoption decision, not a pass.

### Prospective source-fold matched continuation (registered 2026-10-27, before any matched-continuation result)

Registered at 09:55 UTC, before either seed-0 arm above has trained. It runs
whatever the seed-0 screen shows, so no outcome chooses an arm.

- **Design.** Both diagnostic populations (mixed+AIC16, AIC-only) on the D6
  seven-fold source assignment: original Mlic full seeds 0, 1 and 2, each
  continued on `aic_k{k}_fit.parquet` (24 public_train sources, fold k held
  out, k = 0–6) under the same diagnostic authority and budget as the seed-0
  screen (60 epochs × 50,000 draws, LR 1e-5 constant, fresh Adam, AIC weight
  16, every-epoch dumps, fixed epochs 7 and 59). 2 × 3 × 7 = 42 continuations.
- **Out-of-fold AIC.** For each arm, seed and fixed epoch, the seven fold
  models predict only their held-out sources, giving one complete out-of-fold
  prediction of all 3,920 rows. Report pooled, per-source, per-codec and
  ladder panels, per seed and for the three-seed mean prediction, against the
  zero-shot initial models. The historical AIC-only T's seven-fold panel
  (SROCC 0.9527; `qualified_provenance=false`) is a reference point, not a gate.
- **Drift.** Fitted-source in-sample AIC panels and the four E21 populations
  (in sample) for every fold model; both pixel-mixture sets at both epochs.
- **Uncertainty.** Paired seed contrasts; whole-source bootstrap (2,000 draws,
  seed 1) of out-of-fold SROCC differences, descriptive only.
- **Controls.** Fold tables have no prior model to reproduce; the seed-0
  screen's exact epoch-7 control binds the shared machinery, and every fold
  table, authority and checkpoint hash is pinned in a fresh plan.
- No selector, packing, calibration, EVAL or public_test read. Any adoption
  still needs the unchanged production gates. Queued after the candidate gates.

Source-fold study implemented and queued (09:50 UTC): coordinator `9da7d041`
(`aic_matched_continuation.py fold-plan/fold-queue/fold-run/fold-status`);
plan `/mnt/v/output/zensim/mlic-matched-continuation-folds-20261027-run01/PLAN.json`,
SHA256 `ca11bdee2da8b64128cdae3d69d364f41207cc49d637b76cc0368982f56bb454`; same
trainer `d5275ca5…` and frozen panel tools as the seed-0 screen; seed-1/seed-2
initial models `a1283179…`/`346a5e0f…`. Python admission dry runs pass for
AICOnly s1/k3 and MixedAIC16 s2/k0 with the original seeds and fold tables.
The dev queue is now serial: confidence omissions → confidence follow-through
→ seed-0 screen → candidate gates (12 models) → this study (run-heavy 40 GiB,
12 concurrent single-thread fits). No result exists yet.

### Where the AIC-2026 gap sits: per-codec ranking versus cross-codec scale (2026-10-27, existing outputs)

Why does Mlic (0.9075, zero-shot) trail the historical AIC-only T (0.9527 out
of fold)? Per-codec SROCC (rows of one codec pooled over sources; final f32;
the full study's three-seed mean predictions for every Mlic column, against
T's saved seven-fold out-of-fold panel `rust/folds_train/T/oof_panel.json`):

| Codec | T out of fold | Mlic zero-shot | MlicFT16x8 | MlicVoteFT16x8 | MlicOrderFT16x8 |
|---|---:|---:|---:|---:|---:|
| AVIF | 0.9550 | 0.9569 | 0.9607 | 0.9609 | 0.9629 |
| AVIFQ | 0.9806 | 0.9549 | 0.9560 | 0.9547 | 0.9556 |
| CCW | 0.9697 | 0.9450 | 0.9485 | 0.9458 | 0.9482 |
| HEVC | 0.9533 | 0.9638 | 0.9661 | 0.9663 | 0.9663 |
| J2K | 0.9712 | 0.9131 | 0.9215 | 0.9174 | 0.9226 |
| JAI | 0.9542 | 0.9081 | 0.9173 | 0.9157 | 0.9147 |
| JPG | 0.9569 | 0.9346 | 0.9434 | 0.9444 | 0.9411 |
| JPGL | 0.9700 | 0.9796 | 0.9797 | 0.9784 | 0.9781 |
| JPGR | 0.9615 | 0.9568 | 0.9595 | 0.9570 | 0.9607 |
| JXL | 0.9544 | 0.9093 | 0.9199 | 0.9150 | 0.9187 |
| JXLM | 0.9026 | 0.8336 | 0.8620 | 0.8378 | 0.8504 |
| PNG | 0.9205 | 0.9363 | 0.9427 | 0.9413 | 0.9400 |
| VVC | 0.9081 | 0.9737 | 0.9721 | 0.9732 | 0.9729 |
| WEBP | 0.9525 | 0.9037 | 0.9179 | 0.9144 | 0.9111 |
| All rows | 0.9527 | 0.9075 | 0.9146 | 0.9139 | 0.9158 |

Mlic already ranks five codecs as well as or better than T (VVC +0.066, PNG
+0.016, HEVC +0.011, JPGL +0.010, AVIF +0.002), and trails on J2K, JAI, JXL,
JXLM, WebP, CCW, AVIFQ and JPG. The pooled gap (0.045) is larger than most
per-codec gaps because the codecs sit on different scales: Mlic's codec rank
bias spreads 0.075 (PNG +0.211, VVC +0.142, HEVC +0.105; JPG −0.059, WebP
−0.041), and eight epochs of continuation leave it there (MlicFT16x8 0.076,
Order 0.0715; `bias-*|final-f32|aic.json`). The mixed-data model has learned a
cross-codec scale that AIC's observers do not share; the short continuations
fix little of it. The matched continuation (60 epochs, AIC-only arm) and the
source-fold study measure how much more AIC exposure moves that scale. T's
cells are seven-fold out of fold; the continuation columns are in sample;
T's provenance remains unqualified. Read only, no new fit.

### MlicConfidence TRAIN screen complete (2026-10-27, 10:18 UTC)

Confidence run02 completed all 21 omission cells after its exact control
(7,869 predictions byte-identical) and wrote `TRAIN_SCREEN_COMPLETE.json`.
Against Mlic, paired by seed (seed-unit SE): balanced −0.00097 (0.00137),
old-data E21 −0.00249 (0.00181), new data +0.00055 (0.00102). The
registered checks: balanced gain false, old-data non-inferiority true, zero
global-edit mixture inversions true; finalist none, Mlic retained. Against
B's recipe, E21 SROCC per set (three-seed mean): MlicConfidence KADID 0.8841,
TID2013 0.8799, KonFiG 0.7819, CID22-A 0.8793 (Mlic 0.8825 / 0.8741 / 0.8078 /
0.8808): better on TID2013, 0.026 worse on KonFiG. Local mixtures are worse
than Mlic's: 0–16 reversals per cell (Mlic 2–5), largest rise +6.66 (Mlic
+6.17); the worst cells add blur-plus-luma-noise reversals (pipal s2: 16,
mostly blur1.2+luma4) that Mlic's cells do not show. Global edits: none.
As registered, the historical screen does not exclude the arm from the AIC
comparison; the follow-through started its fold continuations at 10:19 UTC.
Evidence: `mlic-confidence-20261027-run02/TRAIN_SELECTION.json`, `scorecard/`.

Mlic's twelve out-of-sample E21 cells (models that never saw AIC-2026 data),
zero-shot on all 3,920 public_train rows (2026-10-26, `byc_cells.py aiczero`;
`/var/tmp/byc-finalists-20261026/aic-rev7/aiczero_Mlic_cells.json`, sha256
`f2037e57…`): mean per-cell global SROCC 0.9075 (range 0.8991–0.9163), mean
within-source 0.9485, per-cell codec-bias spread 0.0743; the mean prediction of
the twelve cells reaches 0.9095 (within source 0.9501). The cells agree with
the full-data three-seed result above.

The full-data Mlic seed-0 model's only two local reversals are posterization
plus luma noise on smooth backgrounds (references 9870 and 5050): poster12
scores 27.37 / 29.12 and poster12+luma4 31.78 / 30.51. The added noise breaks
up the posterization contours, as dither does, so "more error energy is worse"
is a questionable label for these two rows; B's reversals (blur plus colour
noise on every reference, up to +44 points) are not of this kind. Image strips
(reference | poster12 | poster12+luma4): `/mnt/v/output/zensim/mixture-dither-20261027/`.

### Native HDR precision at revision 7 (2026-10-27, label-free)

The precision run of October 27 left one gap open: no f64 error bound for HDR features. This entry measures it.
`tier_audit_features --hdr-pq` (new) reads 16-bit PQ PNG codes as opaque `Srgb16Rgba` and extracts basic, peaks
and v2 through `research::extract_hdr`. A pair's distorted side can be a deterministic code-space perturbation of
the reference: `dither1` moves about one value in ten by ±1 code, `shift4` and `shift64` add 4 or 64 codes, and
`blend5` moves 5% toward a 3×3 box mean. Inputs are six imazen-26-hdr TRAIN references, chosen as the first six by
sha256 of the stem (1064, 1234, 1547, 1534, 1533, 1544). They are 384×512, BT.709 primaries, PQ (cICP 1/16/0/1),
peak 10000. That gives 24 pairs at revision 7, tier v3 (i265, one thread); Zen 5 / v4x was not measured.

The two arms are the production canonical arithmetic and `ZENSIM_FEATCANON=exact`. The exact arm uses f64 PU-XYB
elements, a fresh f64 blur and compensated accumulation. Both arms share the f32 PQ decode table and f32 plane
storage; the display-light parity tests in `native_hdr_color.rs` cover the decode. `rev7_forms.py` (now with a
single-arm mode and a score step) gives the errors. `predict_features_with_bake --f64-wire --production` scores
both arms' rows with the packed seed-0 Mlic and MlicFT16x8 bakes.

- **Score:** Mlic moves by at most 1.83e-4 points (median 3.7e-5) and MlicFT16x8 by 1.65e-4 (median 4.5e-5), on
  scores of 94.4–99.99.
- **Moderate distortion** (`blend5`, feature magnitudes around 1e-4): the median per-feature relative error is
  1.1e-5 (basic), 2.5e-5 (peaks) and 1.8e-5 (v2), with a worst of 1.3e-3. That is the same class as the SDR
  revision-7 measurement above (5.1e-5).
- **Near-identity** (`dither1`, `shift4`, magnitudes 1e-8 to 1e-7): relative errors run from 1e-3 to 1. Absolute
  errors stay at most 8.3e-5 over all pairs, with a p99 of 3e-6 or less.
- **Finding: the L8 peak pools underflow.** They form `d⁸` as `d4 * d4` in f32 lanes before the f64 reduction. That
  term is subnormal below a per-pixel error of about 1.6e-5 and zero below about 4e-6. Production emits exactly 0 in
  191 cells where the oracle has up to 9e-7 (for example `peaks_ssim_l8` at near-identity). The score bound above
  includes it. A fix changes feature bits, so it belongs to a future revision, for instance the HDR frontend's.

Artifacts: tower `/mnt/tower/output/zensim/hdr-precision-20261027/` holds `REFS.json`, the dumps, `PRECISION.*`,
`SCORE_IMPACT.json`, the bakes and `bin/SHA256SUMS`: `tier_audit_features` 1d7a817f…, built with
`--features featcanon-oracle`, and `predict_features_with_bake` a53c55ad…. `replay-final/` reruns the same probe
through the recipe below, with the committed source, and reproduces every dump and score bit for bit:

`just hdr-precision <refs> <out> <tier_audit_features> <predict_features_with_bake> 7 srgb 10000 v3 NAME=BAKE...`

### MlicConfidence fold and full-data results (2026-10-27, follow-through complete 11:25 UTC)

The serial follow-through (`/mnt/v/output/zensim/mlic-confidence-followthrough-20261027-run01`, CONFIG `60cb9ed6…`)
ran 48 fold continuations (24 new), their panels, three 120-epoch full confidence fits, six new full continuations
and the full panels, without failures. Pooled SROCC on AIC-2026 public_train (fold: seven folds of 4 held-out
sources, three seeds) and E21 out of sample, paired by seed:

| Model | Fold fitted | Fold held-out | E21 vs Mlic at the same budget |
|---|---:|---:|---|
| MlicFT8x4 | 0.9117 | 0.9080 | reference |
| MlicConfidenceFT8x4 | 0.9078 | 0.9078 | −0.0040 (SE 0.0039) |
| MlicFT16x8 | 0.9143 | 0.9097 | reference |
| MlicConfidenceFT16x8 | 0.9114 | 0.9099 | −0.0037 (SE 0.0038); KonFiG −0.019 |

Initial models: MlicConfidence 0.9060 fitted and 0.8976 held out, against Mlic's 0.9103 and 0.8996.

Full data, in sample, packed serving models (three-seed mean): ConfidenceFT16x8 against MlicFT16x8 scores AIC-2026
0.9108 against 0.9146, KADID 0.9132 against 0.9124, TID2013 0.8948 against 0.8893, KonFiG 0.8163 against 0.8210
and CID22-A 0.8828 against 0.8837. MlicConfidence against Mlic scores AIC-2026 0.9038 against 0.9076.

Local mixture reversals are 4, 7 and 6 rows (initial, 8×4, 16×8; up to +8.0 points), against 2–4 for Mlic, Neff,
Vote and Order. Global mixtures are 0. Near-identity passes for all three seed-0 models (N1, N2; N3 143, 143, 142
of 144), in `/mnt/tower/output/zensim/mlic-confidence-nearid-20261027-run01/COMPLETE.json`, with the binary,
packet and reporter of run03.

Confidence weighting does not improve on Mlic in any setting measured here. It stays an experiment. The
matched-continuation diagnostic took the compute lock at 11:26 UTC.

### Matched-continuation diagnostic: result (2026-10-27, complete 11:44 UTC)

Run `/mnt/v/output/zensim/mlic-matched-continuation-20261027-run01`. **Controls hold.** Mixed epoch 7 equals the
registered 8-epoch MlicFT16x8 seed 0 byte for byte once only `zentrain.repro` is removed, and every result field
matches (`CONTROL.json`). The initial model's and mixed epoch 7's predictions equal the full study's on all five
populations (`panels/PREDICTION_CONTROLS.json`). **Exposure** (native replay of the sample coverage): over 60
epochs the mixed arm drew 626,417 AIC pairs of 2,738,532 (22.9%); the AIC-only arm drew 2,999,262, all AIC. Both
touched every AIC row.

Seed 0, raw f32 models, in sample on every population:

| Model | AIC-2026 public_train | KADID | TID2013 | KonFiG | CID22-A |
|---|---:|---:|---:|---:|---:|
| Initial (Mlic) | 0.9094 | 0.9037 | 0.8937 | 0.8270 | 0.8739 |
| Mixed + AIC×16, epoch 7 (= MlicFT16x8) | 0.9139 | 0.9134 | 0.8930 | 0.8303 | 0.8870 |
| Mixed + AIC×16, epoch 59 | 0.9219 | 0.9132 | 0.8953 | 0.8330 | 0.8889 |
| AIC-only, epoch 7 | 0.9527 | 0.8432 | 0.8564 | 0.8173 | 0.8859 |
| AIC-only, epoch 59 | 0.9698 | 0.6604 | 0.7505 | 0.7944 | 0.8929 |

Whole-source bootstrap, 95% interval of the AIC SROCC difference (28 sources, 2,000 resamples, in sample,
descriptive):

| Difference | 95% interval |
|---|---|
| mixed epoch 59 − epoch 7 | [+0.0042, +0.0117] |
| mixed epoch 59 − initial | [+0.0061, +0.0197] |
| AIC-only epoch 59 − mixed epoch 59 | [+0.0315, +0.0622] |
| AIC-only epoch 7 − initial | [+0.0241, +0.0612] |

Local mixtures are 2–3 rows for every model; the largest rise is +6.5 (mixed epoch 59) and the AIC-only models'
rises are the smallest (+0.75 at epoch 59). Global mixtures are 0.

**Reading.** Trained on AIC alone, Mlic's recipe fits AIC-2026 public_train almost as tightly as the historical
AIC-only model (0.9698 in sample at epoch 59; T 0.974). Neither its inputs nor its capacity cap AIC; the mixed
population does. Fifty-two more mixed epochs gain +0.008 on AIC in sample and hold the other four sets. AIC-only
continuation gains about +0.06 on AIC by giving up KADID (0.904 to 0.660) and TID2013 (0.894 to 0.751). All of this
is in sample and seed 0. The source-fold study (42 fits, three seeds, out-of-fold AIC and E21) tests whether the
mixed epoch-59 gain survives on held-out sources; it is running.

**Gates queue repair.** The gates' run01 queue exited at 11:44 UTC before any gate. Its source snapshot carried the
justfile but not the `benchmarks/speedq.just` it imports, so `just` refused to parse. The source-fold study, queued
behind the gates, then took the lock as designed. Run02 (`mlic-candidate-gates-20261027-run02`) uses a complete
snapshot of main `49a6eba6`. `gates.py` and `scripts/rev4_featpot` are unchanged since run01's `39d6420a`. It runs
the same 15 models after the fold study.

### Source-fold matched continuation: result (2026-10-27, complete 13:11 UTC)

All 42 fits finished; 87 models were predicted on the five populations. Out of fold on AIC-2026 public_train
(three-seed mean of the assembled held-out predictions; Initial is zero-shot):

| Model | Three-seed mean | Seeds 0 / 1 / 2 |
|---|---:|---|
| Initial (Mlic) | 0.9075 | 0.9094 / 0.9089 / 0.9023 |
| Mixed + AIC×16, epoch 7 | 0.9117 | 0.9119 / 0.9131 / 0.9092 |
| Mixed + AIC×16, epoch 59 | 0.9170 | 0.9168 / 0.9159 / 0.9170 |
| AIC-only, epoch 7 | 0.9410 | 0.9393 / 0.9375 / 0.9381 |
| AIC-only, epoch 59 | 0.9524 | 0.9498 / 0.9525 / 0.9528 |

Paired by seed (seed-unit SE), out of fold:

| Difference | Mean (SE) | Whole-source 95% interval |
|---|---|---|
| mixed epoch 59 − epoch 7 | +0.0051 (0.0014) | [+0.003, +0.007] |
| mixed epoch 59 − initial | +0.0096 (0.0025) | [+0.005, +0.014] |
| AIC-only epoch 59 − mixed epoch 59 | +0.0352 (0.0011) | [+0.016, +0.054] |

The older sets, in sample, as the mean over fold models against the initial model:
- Mixed epoch 59 gains on every set: KADID +0.007, TID2013 +0.004, KonFiG +0.010, CID22-A +0.014.
- AIC-only epoch 59 loses KADID (−0.227, SE 0.008), TID2013 (−0.130) and KonFiG (−0.018), and gains CID22-A
  (+0.021).

Local mixtures stay at 4–5 rows per cell (largest rise +6.6, mixed epoch 59). Global mixtures are 0.

The mixed epoch-59 gain holds on held-out AIC sources, and AIC-only reaches the historical AIC-only model's
out-of-fold 0.9527. These E21 numbers are in sample: every continuation starts from full-data Mlic, so they measure
forgetting, not generalization.

### AIC-share frontier: can rehearsal reach 0.95 out of fold and keep the older sets? (registered 2026-10-27, before any frontier fit)

Owner, 2026-10-27, about 13:05 UTC, verbatim: "using newer weighting techinques try to get 0.95 aic while recovering
older dataset perf". The source-fold result above bounds the problem. Mixed + AIC×16 (about 23% of draws) reaches
0.917 out of fold and keeps the older sets in sample. AIC-only reaches 0.952 and loses KADID and TID2013. The older
human legs train only within-reference ranks, while the AIC leg ranks pooled across sources, so AIC-heavy training
reorders what the older legs no longer rehearse.

**Stage A (static mixtures).**
- Arms: Mlic full seed 0, continued from its final checkpoint for 60 epochs (constant LR 1e-5, fresh Adam) on its
  original mixture plus AIC-2026 public_train at weight W = 64, 256 or 1024. The mixture's total weight is 60.7, so
  AIC is about 51%, 81% and 94% of draws. Each arm runs on each of the seven D6 four-source folds.
- Checkpoints 7, 15, 29 and 59 are retained. The source-fold study's seed-0 fits of mixed ×16 and AIC-only are reused
  at the same checkpoints.
- Panels: out-of-fold AIC-2026 public_train (seed 0); the older sets (KADID, TID2013, KonFiG, CID22-A) in sample,
  as the mean over fold models, which measures forgetting; source, codec and ladder aggregates; both mixture tests;
  whole-source intervals. No EVAL, public_test or AIC-4 read.

**Advancement (my rule, declared before results; not a winner claim).** A point (W, epoch) keeps the older sets
when its in-sample SROCC on each of KADID, TID2013, KonFiG and CID22-A is at least the initial model's minus 0.005.
Among those points, the highest out-of-fold AIC advances, and ties within 0.002 go to the smaller W. If no point
keeps the older sets, the highest out-of-fold AIC with a worst older-set loss under 0.02 advances, flagged.

**Stage B (registered here, run after A).** At the advanced budget, continue the MlicNeff, MlicVote, MlicOrder and
MlicConfidence full seed-0 models the same way, on the same folds, with the same panels and rule. Their own
weighting stays active in their older legs.

**Stage C.** The chosen arm runs seeds 1 and 2 on the folds. Then come full-data continuations of all three full
seeds, packing, E21 out of sample through the existing omission cells continued the same way, the full panels and
mixtures, near-identity and the label-free gates. No adoption follows from Stage A or B alone.

The native and Python admissions add a second registered protocol, `aic-share-frontier-v1`, beside the matched
diagnostic's. It admits only these three populations, each at its own public_train weight, with the four
checkpoints. The matched diagnostic's own rules are unchanged.

**Stage A2, recovery continuation (registered 2026-10-27, about 13:50 UTC, before any stage A result).**
- Schedule: AIC first, then recovery. Each fold's seed-0 AIC-only epoch-59 model from the source-fold study
  continues 60 more epochs (constant LR 1e-5, fresh Adam). It trains on the original mixture plus the same fold's AIC
  table, at W = 16, 64 or 256.
- Checkpoints 7, 15, 29 and 59.
- Out-of-fold validity holds: in neither phase does a fold model see its held-out sources.
- Its points join stage A's frontier under the same panels and the same advancement rule.
- Comparisons: the starting AIC-only model (how much AIC the recovery gives back) and the initial Mlic (what the older
  sets regain).
- Admission is unchanged: the populations are the registered ones, and only the continuation checkpoint differs.
- The sampler stays a declared function of seed, groups and weights. Loss-driven reweighting would break that
  replay contract and is not used.

### Label-free gates for all 15 seed-0 candidates (2026-10-27, complete 14:06 UTC)

Gates run02 (`mlic-candidate-gates-20261027-run02`; run01 failed before any gate, see above) ran `gates.py` through
the E33/E34 owners on each packed seed-0 serving model. It used the revision-7 tools
(`688746b78bee-gates`), the rev7raw instruments and the empty human corpus.

All 15 pass the dial contract C1–C6 on both grids. All 15 pass identity on 62 sources at all 12 SIMD
permutations. G-STEER (135 cases):

| Model | C1 standard / ladder | C3 | G-STEER |
|---|---|---:|---:|
| Mlic | 0.9977 / 0.9932 | 0.7885 | 132 |
| MlicFT8x4 | 0.9977 / 0.9928 | 0.7575 | 133 |
| MlicFT16x8 | 0.9977 / 0.9926 | 0.7685 | 129 |
| MlicNeff | 0.9975 / 0.9931 | 0.7685 | 134 |
| MlicNeffFT8x4 | 0.9977 / 0.9926 | 0.7655 | 131 |
| MlicNeffFT16x8 | 0.9977 / 0.9925 | 0.7765 | 134 |
| MlicVote | 0.9975 / 0.9930 | 0.7880 | 134 |
| MlicVoteFT8x4 | 0.9975 / 0.9925 | 0.7490 | 134 |
| MlicVoteFT16x8 | 0.9977 / 0.9920 | 0.7755 | 135 |
| MlicOrder | 0.9975 / 0.9931 | 0.8000 | 135 |
| MlicOrderFT8x4 | 0.9977 / 0.9922 | 0.7795 | 135 |
| MlicOrderFT16x8 | 0.9975 / 0.9921 | 0.7890 | 135 |
| MlicConfidence | 0.9986 / 0.9931 | 0.7215 | 135 |
| MlicConfidenceFT8x4 | 0.9984 / 0.9930 | 0.7230 | 134 |
| MlicConfidenceFT16x8 | 0.9986 / 0.9930 | 0.7250 | 134 |

For comparison, B (revision 5) has G-STEER 128. The gate summaries are in the run's `out/`; the compact table is
`GATES_SUMMARY.json`. These gates are label-free. Open for every candidate, B included: the registered runtime
gate (hardware decision), G-TARGET (every dial, peers included, failed its last run), G-RD (loop parked by the
owner) and an integrity head.

### AIC-share frontier stage A: result (2026-10-27, complete 15:05 UTC)

Run `/mnt/v/output/zensim/mlic-aic-share-frontier-20261027-run01` (PLAN `966ca45d…`, trainer `5558ea6d…` from main
`e6f6829c`). All 21 fits finished. **The control holds.** This trainer's refit of mixed ×16, seed 0, fold 0 equals the
source-fold study's fit at epochs 7, 15, 29 and 59, byte for byte once only `zentrain.repro` is removed
(`CONTROL.json`). **Exposure** (native sample-coverage replay, mean over the seven fold models): AIC is 22.9% of
draws at ×16, 54.2% at ×64, 82.6% at ×256 and 95.0% at ×1024.

Out-of-fold AIC-2026 public_train (seed 0; the initial model is zero-shot at 0.9094). The older sets are in sample,
as the mean over fold models minus the initial model's SROCC. "Keeps" is the registered rule (every older set within
0.005 of the initial model).

| Point | Out-of-fold AIC | KADID | TID2013 | KonFiG | CID22-A | Keeps |
|---|---:|---:|---:|---:|---:|---|
| ×16 e59 (reused) | 0.9168 | +0.0096 | +0.0016 | +0.0060 | +0.0149 | yes |
| ×64 e7 | 0.9203 | +0.0094 | −0.0002 | +0.0017 | +0.0151 | yes |
| ×64 e59 | 0.9288 | +0.0088 | −0.0005 | +0.0024 | +0.0182 | yes |
| **×256 e7** | **0.9312** | +0.0063 | −0.0028 | −0.0033 | +0.0130 | **yes** |
| ×256 e15 | 0.9359 | +0.0060 | −0.0049 | −0.0070 | +0.0179 | no |
| ×256 e59 | 0.9418 | +0.0034 | −0.0086 | −0.0108 | +0.0222 | no |
| ×1024 e7 | 0.9376 | −0.0020 | −0.0132 | −0.0111 | +0.0172 | no |
| ×1024 e59 | 0.9465 | −0.0194 | −0.0259 | −0.0208 | +0.0252 | no |
| AIC-only e29 (reused) | 0.9515 | −0.1968 | −0.1141 | −0.0223 | +0.0172 | no |
| AIC-only e59 (reused) | 0.9498 | −0.2436 | −0.1421 | −0.0324 | +0.0182 | no |

All twenty points are in the run's `panels/OUT_OF_FOLD_PANELS.json` and `panels/PAIRED_SEEDS.json`. Under the
registered rule, **×256 at epoch 7 advances** (0.9312; the next keeper, ×64 e59, is 0.0024 lower). Stage A2's
points join before stage B uses the budget. Whole-source 95% intervals of the out-of-fold AIC difference (28 sources,
2,000 resamples, descriptive): ×256 e7 − initial [+0.010, +0.033]; ×256 e7 − ×16 e7 [+0.008, +0.030]; ×1024 e59 −
×16 e59 [+0.012, +0.048]; AIC-only e59 − ×1024 e59 [−0.003, +0.011]. Local mixtures stay at 1–3 rows per cell
(largest rise +6.6, ×16 e59; the AIC-heavy models rise least); global mixtures are 0.

**Reading.** Static mixing trades AIC against TID2013 and KonFiG; KADID holds up to ×256 and CID22-A gains at every
weight, AIC-only included. No static mixture reaches 0.95 out of fold while keeping the older sets: ×1024 e59 gets
0.9465 and loses about 0.02 on three sets, and AIC-only gets 0.9515 and loses KADID by 0.20. Seed 0 only.

**Diagnostic: where the older sets lose** (frozen predictions, the canonical `panel --per-group`; TRAIN labels read
under a ledger; `/mnt/v/output/zensim/frontier-withinref-diagnostic-20261027/`). Mean over fold models, change
against the initial model in global SROCC / mean within-reference SROCC:

| Point | TID2013 | KonFiG | KADID |
|---|---|---|---|
| ×256 e7 | −0.003 / +0.001 | −0.003 / −0.001 | +0.006 / +0.007 |
| ×256 e59 | −0.009 / −0.003 | −0.011 / −0.004 | +0.003 / +0.005 |
| ×1024 e59 | −0.026 / −0.018 | −0.021 / −0.007 | −0.019 / −0.017 |
| AIC-only e29 | −0.114 / −0.108 | −0.022 / −0.008 | −0.197 / −0.197 |

At ×256 most of the loss is cross-reference scale: the older human legs train only within-reference ranks, so their
cross-reference order is not rehearsed. With more AIC, the within-reference order goes too.

**Where the AIC gain sits** (stage A's own out-of-fold source and codec panels; no new label read). From the initial
model to AIC-only e29, pooled out-of-fold SROCC rises +0.042 (0.9094 to 0.9515), but the mean within-source SROCC
over the 28 sources rises only +0.016 (0.9493 to 0.9655); within one source and codec the ladders are ordered for every
model (0.9997–0.9998). Most of the AIC gain is cross-source calibration, the same kind of order the older sets lose.
An AIC leg ranked only within sources could not reach 0.95 pooled, so that variant is not pursued.

### Stage B details (registered 2026-10-27, 15:10 UTC, before any stage A2 result or stage B fit)

The stage B registration above stands; this fixes its mechanics.
- **Budget.** The point that advances after stage A2 joins, under the registered rule. A static point (W, epoch)
  runs each recipe 60 epochs at W on the seven folds with checkpoints 7, 15, 29 and 59. A recovery point runs each
  recipe's AIC-only continuation first (60 epochs; no older training leg, so no recipe weighting), then the recovery
  at W with the weighting active.
- **One run per recipe**, each with its own initial model (the recipe's full seed-0 fit), root and recipe inputs:
  Neff and Vote on their own roots; Order on the Mlic root with its full-model matching schedule
  (`mlic-aic-full-20261027-run03/matching/full_s0`, matching model still the frozen full Mlic seed 0); Confidence
  on its root with its full-fit confidence manifest.
- **Native admission.** Order's matched-codec admission now accepts a registered 60-epoch continuation (version-4
  authority, a population with the human group) and keeps every other check. Confidence accepts a continuation
  manifest (`mlic-contrast-confidence-continuation-v1`) that wraps the unchanged full-fit manifest, binds the starting
  checkpoint's bytes and the 60-epoch budget, and appends AIC public_train unweighted; its receipt records the
  continuation. Original full-fit behaviour and receipts are unchanged.
- **Control, per recipe:** this trainer replays the recipe's 120-epoch seed-0 full fit through its original wrapper
  route and frozen tools. The existing E33 control owner must find the final raw and packed models byte-identical
  without `zentrain.repro`, and every result field equal apart from paths and model hashes; for Confidence the native
  normalizer receipt must match too. A failed control stops that recipe's contrasts.
- **Panels and rule:** as stage A, with each recipe's own initial model as "Initial". Contrasts: each point against
  its own initial and against Mlic's point at the same budget from stage A or A2. Order's matched-codec route has no
  sample-coverage replay, so its exposure is not recorded. Seed 0; no adoption from stage B alone.

### Stage C mechanics and the stage B → C choice (registered 2026-10-27, 15:40 UTC, before any stage A2 or B result)

**Which arm stage C runs.** The point that advances after stage A2 joins is Mlic's. A stage B recipe point replaces
it only if that point keeps the older sets against the recipe's own initial model and its out-of-fold AIC exceeds
the advanced Mlic point's by more than 0.002. Ties and smaller gains keep Mlic, the simpler recipe.

**Steps** (coordinator `aic_matched_continuation.py`, all built and smoke-tested on real inputs before this entry):
1. `seeds-plan`: the chosen point on the seven folds for seeds 0, 1 and 2. Seed 0 must reproduce the advancing
   run's 28 checkpoints (7 folds × 4 epochs) byte for byte without `zentrain.repro`; a mismatch stops the step.
   Out-of-fold AIC and the in-sample older sets, paired by seed with seed-unit SE, against the initial model,
   ×16 and AIC-only.
2. `omission-plan`: E21 out of sample. Each Mlic omission cell (four E21 datasets × three seeds) continues 60 epochs
   at the chosen population on the LODO diagnostic route (`v2_lodo_mlp.py --diagnostic-population`), with the
   common adaptation study's rotating AIC fold table. The existing E21 owner scores the held-out dataset at every
   studied epoch against Mlic's and MlicFT16x8's cells, paired by seed.
3. `full-plan`: full Mlic seeds 0–2 continue on all of public_train; the chosen epoch's checkpoint is packed in the
   canonical deployment order (densify, f16, E33 output stage, TRAIN CID22 calibration). Every packed model is scored
   in sample with the packed Mlic and MlicFT16x8 models, paired by seed, and pixel mixtures run.
4. Near-identity and the label-free gates (dial C1–C6 on both grids, G-STEER, identity at every SIMD permutation)
   through their existing owners, for the packed seed-0 model.

Stage C reports; it does not adopt. Adoption stays the owner's D1 decision. A recovery point runs steps 2 and 3 in
two phases, as stage A2 did: AIC-only for 60 epochs, then the recovery from that phase's registered start epoch
(built and pushed in `df36955c`, 15:37 UTC, still before any stage A2 result). A recipe point needs recipe variants
of steps 2 and 3, which are not built yet; they would be built and registered before they run. Smokes (scratch, not evidence): Order and Confidence continuations passed native admission and trained
epoch 0; a LODO omission continuation did the same; the packing snippet packed a stage A checkpoint.

### Stage A2 result and stage B launch (2026-10-27, A2 fits and panels complete 15:51 UTC)

Run `mlic-aic-recovery-20261027-run01`: each fold's seed-0 AIC-only epoch-59 model (out-of-fold AIC 0.9498) recovered 60
epochs on the mixture. Exposure equals the static arms at the same weight (×16 22.7–22.9% AIC, ×64 54.1–54.2%, ×256
82.6%), because the sampler depends only on seed, groups and weights. Out-of-fold AIC; the older sets in sample against
the initial Mlic model:

| Point | Out-of-fold AIC | KADID | TID2013 | KonFiG | CID22-A | Keeps |
|---|---:|---:|---:|---:|---:|---|
| recover ×16 e7 | 0.9214 | −0.0653 | −0.0589 | −0.0632 | +0.0121 | no |
| recover ×16 e59 | 0.9147 | −0.0064 | −0.0197 | −0.0167 | +0.0129 | no |
| recover ×64 e59 | 0.9295 | −0.0122 | −0.0235 | −0.0238 | +0.0196 | no |
| recover ×256 e7 | 0.9323 | −0.1043 | −0.0828 | −0.0695 | +0.0188 | no |
| recover ×256 e59 | 0.9428 | −0.0303 | −0.0338 | −0.0380 | +0.0246 | no |

No recovery point keeps the older sets, so **×256 at epoch 7 (stage A, 0.9312) stays the advanced point**. Recovery
converges to about the static arm's AIC at the same weight (e59: 0.9147 / 0.9295 / 0.9428 against 0.9168 / 0.9288 /
0.9418) while the older sets are still recovering: the AIC-only detour buys no lasting AIC and costs the older sets.

**Stage B launched** (queues started 15:51 UTC; registered mechanics above): four chained runs at ×256,
`mlic-aic-recipe-{order,confidence,neff,vote}-aic256-20261027-run01`, trainer `587e4764…` (main `216481a8`, built on
i265), source `1f334203`. Each fits its recipe's full seed-0 model on the seven folds and replays the recipe's
120-epoch full fit as its control.

### Stage D: within-dataset rehearsal of the older sets (registered 2026-10-27, 16:00 UTC, before any stage B result)

**Why.** Stages A and A2 show that the AIC gain is mostly cross-source calibration and that the older sets lose
cross-reference order. Mlic's human leg (KADID, TID2013, KonFiG, CID22-A) draws its ranking pairs within one reference,
so nothing rehearses that order. UNIQUE (Zhang et al., "Uncertainty-Aware Blind Image Quality Assessment in the
Laboratory and Wild", TIP 2021, in the zenpapers corpus) trains several datasets together on pairs drawn within each
dataset. Stage D asks whether rehearsing each older dataset's own cross-reference order lets AIC rise further without
the loss, or whether the two orders genuinely conflict.

**Arms.** Mlic seed 0 continued as in stage A at AIC ×256 and ×1024, plus one added leg: the same human table and train
weight (32.37), its pairs drawn within each dataset (the keys' member sets: kadid_train and kadid_select as KADID,
konfig_train and konfig_val as KonFiG, tid2013, cid22_a25), rank loss. Seven folds, 60 epochs, LR 1e-5, fresh Adam,
checkpoints 7, 15, 29 and 59, the same panels.

**Decision.** The added leg trains the in-sample panel order directly, so the in-sample rule is reported but does not
decide. D's candidate is its highest out-of-fold AIC point whose in-sample older sets stay within 0.005. It is judged on
E21 out of sample through the omission cells continued the same way (three seeds), against stage A's advanced point
(stage C step 2 runs for both). D's point replaces the advanced point for stage C only if its out-of-fold AIC is higher
by more than 0.002 and its E21 out of sample (three-seed mean) is no more than 0.005 lower on every dataset.

**Order.** After stage B, before stage C. The native admission, Python mirror and coordinator support are built and
tested before anything runs; the trainer gets a separate registered protocol, and the existing protocols are unchanged.

**Queue, 16:10 UTC.** Stage D's admission is pushed (`4424837b`, protocol `aic-share-xref-v1`; trainer `b3b86f18…`
from main `8b66cca2`, built on i265). A real-input smoke loaded the rehearsal leg as 10,590 rows paired within 4
datasets and trained epoch 0. Plans frozen and queued behind stage B: stage D (`mlic-aic-xref-20261027-run01`,
x256 and x1024 with the rehearsal leg; control: stage A's x256 seed 0 fold 0 refit byte for byte), then stage C step 2
for stage A's advanced arm (`mlic-aic-omission-aic256-20261027-run01`, the twelve Mlic omission cells at x256, E21 out
of sample at every studied epoch), which is the comparator stage D's decision needs.

### Stage B, Order: result (2026-10-27, complete 16:48 UTC)

Run `mlic-aic-recipe-order-aic256-20261027-run01`. **The control holds.** This trainer (`587e4764…`) replays Order's
120-epoch full seed-0 fit byte for byte without `zentrain.repro`, raw and packed models, with every result field equal.
Order's matched-codec route has no sample-coverage replay, so its exposure is not recorded. The initial Order model
scores 0.9099 zero-shot. Out-of-fold AIC; the older sets in sample against Order's own initial model:

| Point | Out-of-fold AIC | vs Mlic ×256, same epoch (source 95%) | KADID | TID2013 | KonFiG | CID22-A | Keeps |
|---|---:|---|---:|---:|---:|---:|---|
| Order ×256 e7 | 0.9222 | −0.0090 [−0.014, −0.004] | +0.0004 | +0.0004 | −0.0020 | +0.0127 | yes |
| Order ×256 e15 | 0.9279 | −0.0080 [−0.014, −0.003] | −0.0015 | −0.0012 | −0.0038 | +0.0172 | yes |
| Order ×256 e29 | 0.9321 | −0.0063 [−0.011, −0.002] | −0.0034 | −0.0013 | −0.0045 | +0.0194 | yes |
| Order ×256 e59 | 0.9367 | −0.0052 [−0.009, −0.002] | −0.0061 | −0.0017 | −0.0043 | +0.0209 | no |

Order learns AIC more slowly than Mlic but forgets much less: it keeps the older sets through epoch 29 (TID2013 −0.001
against Mlic ×256's −0.007 there). Its best keeper (0.9321) exceeds the advanced Mlic point (0.9312) by 0.0009, under the
registered 0.002 margin. Local pixel mixtures fall to 1–2 rows (largest rise +2.3, against +4.7 to +5.2 for Mlic ×256);
global mixtures are 0. Seed 0.

### Stage B, Confidence: result (2026-10-27, complete 17:34 UTC)

Run `mlic-aic-recipe-confidence-aic256-20261027-run01`. **The control holds**: the full confidence fit replays byte for
byte (raw and packed, every result field) and its native normalizer receipt is byte-identical (`14e24ce6…`), so the
continuation support leaves original confidence fits unchanged. Each fold continuation's native receipt records the
wrapped full-fit manifest and an unweighted AIC group. Exposure equals Mlic ×256 (82.6% AIC): confidence weights
pairs, not draws. Initial zero-shot 0.9088.

| Point | Out-of-fold AIC | vs Mlic ×256, same epoch (source 95%) | KADID | TID2013 | KonFiG | CID22-A | Keeps |
|---|---:|---|---:|---:|---:|---:|---|
| Confidence ×256 e7 | 0.9179 | −0.0134 [−0.021, −0.006] | +0.0019 | +0.0020 | +0.0002 | +0.0088 | yes |
| Confidence ×256 e15 | 0.9223 | −0.0136 [−0.022, −0.006] | +0.0007 | +0.0023 | −0.0018 | +0.0119 | yes |
| Confidence ×256 e29 | 0.9268 | −0.0117 [−0.018, −0.005] | −0.0004 | +0.0023 | −0.0035 | +0.0143 | yes |
| Confidence ×256 e59 | 0.9320 | −0.0098 [−0.015, −0.005] | −0.0021 | +0.0026 | −0.0032 | +0.0165 | yes |

Confidence learns AIC most slowly and keeps the older sets for all 60 epochs. Its best keeper (0.9320) exceeds the
advanced Mlic point by 0.0008, under the margin. Local mixtures 2–6 rows (largest rise +5.3 at e7, +2.2 at e59);
global 0. **Reading so far:** Mlic ×256 e7 (0.9312 at worst −0.0033), Order e29 (0.9321 at −0.0045) and Confidence e59
(0.9320 at −0.0032) reach about the same AIC for the same older-set cost. The weightings change how fast a model moves
along the trade-off, not the trade-off itself. Seed 0.
