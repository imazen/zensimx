# Monotone by construction — exploration (2026-10-22)

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
