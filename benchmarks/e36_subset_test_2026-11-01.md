# E36 ladder subset as a training term: registration (2026-11-01)

Registered before any E36 feature, score or inversion count was read. Owner, 2026-11-01: "do the 1 to 3 subset test,
once rev7 features are extracted, the decodes can be dropped".

## Question

E36 (`e36_registration_draft_2026-10-16.md`, generated 2026-10-17, never fitted) holds 14,289 native quality ladders
on 200 imazen-26 TRAIN sources (160 fit, 40 selection), four imazen codecs (zenjpeg, the SVT port, zenwebp, JXL), 61
quality steps and 20 sizes, and 2.87 million within-ladder pairs that butteraugli (3-norm) and CVVDP (standard_fhd) both
order by more than 0.01. Does a within-ladder hinge over a subset of those pairs, added to round-3 F's general
continuation, make the model order held-out ladders better without costing anything the current gates measure?

## Data

- **Features.** Revision-7 features (`basic+peaks+v2@w1825/rev7_rowlanes#36c3f3af`, frozen extractor `ed495741`) of
  every E36 decoded view whose reference has area below 1 MP (the owner's training-view rule), extracted from E36's
  stored decoded pixels by zenfleet (`e36_pack_index.py`, `evif_features.py` PNG members; zenmetrics `d46ec235`). Each
  view's reference and distorted RGB8 pixel hashes must equal E36's recorded hashes.
- **Truth.** E36's registered consensus pairs, unchanged: both references agree by more than 0.01, ladder lags 1, 2, 4,
  8 and 16. No SSIMULACRA2 value enters the truth, so views of 64 px or less stay in (the same two-reference rule
  G-TRADE uses at that size).
- **Splits.** Training pairs come only from E36's 160 fit sources. Every assessment below uses only the 40 selection
  sources, which no arm trains on.

## Step 0: how often the current models disagree

Score E, ECodec, F3 and F3codec (seed-0 packed bakes) on every E36 view; report, per split and per stratum (codec;
content photo / line art / mixed / screen; reference size <=64, 80-256, 320-1024; quality band of the lower endpoint
0-20, 22-60, 62-100), the share of consensus pairs a model inverts: `score(better) <= score(worse)`.

If F3 inverts fewer than 2,000 fit-split pairs, the training arms are not run: the pool would be too small to matter,
and that is the result.

## Arms

All arms are round-3 F's general continuation (`recipes/F3.json` in zensim-training: initialized from the Mlicfx2 base
fit's `last.bin`, 60 epochs, x256 AIC public_train plus rehearsal), seeds 0, 1 and 2, packed at epoch 59, with one
trainer build (zensimx, `zensim_mlp_train --tv-pool`).

| Arm | Within-ladder term |
|---|---|
| control | none |
| pool-1 | pool, lambda 1 |
| pool-10 | pool, lambda 10 |

Before any pool arm, the control's seed-0 run with the new build must reproduce F3's continuation: all 60 checkpoints
identical to the recorded ones apart from provenance. If it does not, the trainer change is wrong and nothing proceeds.

**The pool** (subset rule 1): every fit-split consensus pair that F3 (seed 0) inverts, plus an equal number of the pairs
it orders correctly, drawn at random (seed 20261101). **Weights** (rule 2): class weight 1, doubled for screen and line
art content, doubled again when the lower endpoint's quality is 40 or below (classes 1, 2, 2, 4). **Hinge**: E36's
registered form, `max(0, ladder_sign * (raw(better) - raw(worse)) + 0.05)` in raw output units, 32 pairs every 50
primary pairs, on its own RNG stream (the primary sampler's draws are unchanged), scaled by lambda times the class
weight over the batch's mean class weight.

## Outcome and guards

**Primary:** the inversion share on the 40 selection sources, all strata, mean of the three seeds, pool arm minus
control, with a 95% bootstrap interval over selection references (10,000 draws, seed 20261101).

**Guards**, each against the control's three-seed mean:

1. Label-free product gates as F3 passed them: dial C1-C6, NEARID, G-STEER, identity at every SIMD tier.
2. Held-out human collections (`dataset_eval.py mix-eval`, the 11 EVAL collections): no collection's SROCC or 2AFC
   agreement lower by more than 0.01, and the mean over collections not lower by more than 0.003.
3. G-TRADE (a) on the gauntlet (views above 64 px): inversion share no higher than the control's plus 0.5 points.
4. G-CHROMA-L: mean absolute error against the teacher no more than 5% above the control's in any family.

**Decision:** an arm is recommended to the owner only if its primary difference is negative with an interval that
excludes zero and it holds every guard. Both arms are reported either way, with every stratum.

After the features are extracted and checked, E36's decoded PNGs may be dropped (owner); the encoded files, metadata
and maps stay until the owner decides on the maps.

## Results (2026-11-02)

Everything below follows the registration above; the deviations are listed at the end.

**Data.** Run `e36-features-20261101` (zenfleet, i270 + r3800x) extracted all 8,657 jobs (820,466 decoded views of 3,074
references below 1 MP; 20,228 views refused at or above 1 MP), every view's reference and distorted pixel hashes equal
to E36's. Bank: `/mnt/v/zensim/e36-features/2026-11-01/bank` (f32, `features_sha256` `001480f4…`, keys `bd5e8a51…`).

**Control check.** The control's three seeds with the new trainer (`zensim_mlp_train` `a1f7620f`, build of `cc2add0b`)
reproduce F3's continuation: 60 of 60 checkpoints per seed identical apart from provenance, packed models identical after
stripping provenance (seed 0 `e17e6175…`). Control seed 0 scores every E36 view, every gauntlet row and every gate
exactly as F3.

**Step 0.** Of 2,866,213 consensus pairs, 2,787,241 have both views scored (78,972 have a reference at or above 1 MP).
Inversion share, `score(better) <= score(worse)`:

| split | pairs | E | ECodec | F3 | F3codec |
|---|---:|---:|---:|---:|---:|
| fit | 2,140,564 | 4.61% | 6.43% | 2.89% | 3.84% |
| selection | 646,677 | 5.25% | 7.06% | 3.35% | 4.47% |

F3 inverts 61,769 fit pairs, so the arms ran. Per-stratum tables:
`/mnt/v/zensim/e36-features/2026-11-01/inversions/INVERSIONS.json`.

**Pool.** 61,769 inverted + 61,769 anchor pairs; 411 of them repeat another pair's two decoded views (distinct
encodes, such as JPEG q5 and q10 of one 64 px view, decode to identical pixels) and the trainer refuses duplicates, so
the pool keeps the first copy: 61,376 inverted + 61,751 anchor = 123,127 pairs over 186,425 views (classes 0-3: 28,307
/ 32,054 / 28,779 / 33,987). Pool manifest: `/mnt/v/zensim/e36-features/2026-11-01/pool-F3/manifest.json`.

**Primary** (40 selection sources, 646,677 pairs, mean of seeds 0-2, 10,000 source draws, seed 20261101):

| arm | inversion share | seeds 0 / 1 / 2 | minus control | 95% interval |
|---|---:|---|---:|---|
| control | 3.358% | 3.351 / 3.367 / 3.355 | | |
| pool-1 | 3.170% | 3.165 / 3.186 / 3.159 | **-0.188 points** | [-0.269, -0.124] |
| pool-10 | 2.714% | 2.707 / 2.716 / 2.720 | **-0.643 points** | [-0.841, -0.484] |

Every selection stratum is lower for both arms (three-seed means):

| stratum | pairs | control | pool-1 | Δ | pool-10 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| all | 646,677 | 3.36% | 3.17% | -0.19 | 2.71% | -0.64 |
| codec=jpeg | 191,381 | 3.76% | 3.50% | -0.26 | 2.83% | -0.93 |
| codec=jxl | 135,980 | 3.14% | 3.00% | -0.14 | 2.59% | -0.55 |
| codec=svt | 147,586 | 2.01% | 1.83% | -0.18 | 1.53% | -0.47 |
| codec=webp | 171,730 | 4.25% | 4.09% | -0.16 | 3.70% | -0.54 |
| content=lineart | 146,029 | 4.34% | 3.95% | -0.40 | 3.21% | -1.14 |
| content=mixed | 162,131 | 3.61% | 3.47% | -0.14 | 3.03% | -0.58 |
| content=photo | 176,203 | 2.19% | 2.08% | -0.12 | 1.79% | -0.41 |
| content=screen | 162,314 | 3.48% | 3.36% | -0.12 | 2.96% | -0.52 |
| size_band=320-1024 | 173,210 | 1.06% | 0.99% | -0.07 | 0.79% | -0.27 |
| size_band=80-256 | 284,008 | 2.42% | 2.26% | -0.16 | 1.91% | -0.51 |
| size_band=<=64 | 189,459 | 6.86% | 6.52% | -0.34 | 5.68% | -1.18 |
| q_band=0-20 | 188,339 | 4.11% | 3.95% | -0.15 | 3.48% | -0.63 |
| q_band=22-60 | 285,388 | 3.52% | 3.30% | -0.22 | 2.84% | -0.68 |
| q_band=62-100 | 172,950 | 2.28% | 2.11% | -0.17 | 1.67% | -0.60 |

**Guards** (each against the control's three-seed mean):

1. Product gates. C1-C6 pass on both grids for all nine models; identity passes at all 12 SIMD permutations for all
   nine; NEARID N1-N3 pass for all nine (monotone ladders: control 140/140/140, pool-1 140/140/140, pool-10
   140/139/139; bar 122 of 144); G-STEER: control 135/135/133, pool-1 135/135/134, pool-10 135/135/134 (seed 0 of
   both arms 135/135, F3's bar). **Pass, both arms.**
2. Held-out human collections (11 EVAL collections, three-seed mean SROCC or 2AFC): pool-1 worst collection -0.0011
   (irccyn-ivc), mean +0.0002; pool-10 worst -0.0032 (live-md), mean +0.0010. Bars -0.01 / -0.003. **Pass, both arms.**
3. G-TRADE (a), gauntlet views above 64 px (18,267 trade pairs, 173 sources): control 2.36%, pool-1 2.30%, pool-10
   2.36%. Bar: control + 0.5 points. **Pass, both arms.**
4. G-CHROMA-L (dev rows, 12,694; views of 64 px or less dropped): pool-1 MAE ratio to control 0.992-0.997 per family;
   pool-10 0.962 (band-ac) to 1.018 (ycc420). Bar 1.05. Per-ladder Spearman 1.0 in every family for all nine models.
   **Pass, both arms.**

**Decision rule.** Both arms meet it: negative primary difference with an interval excluding zero, and every guard held.
pool-10 lowers the selection inversion share by 19% of the control's and holds every guard with margin.

**Deviations and incidents.**
- The pool drops 411 duplicate view pairs (above); the registered rule did not anticipate distinct encodes decoding to
  identical pixels.
- Guard 2 ran `dataset_eval.py score` + `report` on a copy of the EVAL folder's non-image files, which is what
  `mix-eval` does for the 11 collections (mix-eval also writes a BAPPS copy, which is not part of the guard).
- The first G-CHROMA-L scoring of the controls used the tool's default roles (dev and calibration) and so read the
  calibration rows once for F3 and the three controls; that output was set aside unused, and every number above is
  from the dev rows only.
- Training ran the six arms concurrently on one host, each pinned to one thread as the recipe requires.

Artifacts: bakes (production-f16 and provenance-stripped) with SHA256SUMS at
`/mnt/v/output/zensim/e36-subset-test-2026-11-01/bakes/` and `/mnt/tower/output/zensim/e36-subset-test-2026-11-01/bakes/`;
E36 scores of all nine models `/mnt/v/zensim/e36-features/2026-11-01/scores-arms`; inversion tables `inversions-arms/`.

## Amendment (2026-11-02, before any pool-30 or pool-100 run)

Owner, 2026-11-02: "what about the lambda 30 exploration to find better candidates to lower median error vs human".
pool-10 beat pool-1 and kept margin on every guard, so the lambda axis is extended upward.

- **New arms:** pool-30 (lambda 30) and pool-100 (lambda 100, a bracket), seeds 0, 1 and 2, otherwise identical to
  pool-1 and pool-10: the same pool (`pool-F3/manifest.json`, 123,127 pairs), hinge, class weights, trainer build and
  packing.
- **Primary, guards and decision rule:** unchanged, against the same control. Among arms that meet the decision rule,
  the recommendation is the one with the lowest primary.
- **Error against human ratings (reported, not used to select):** for each MOS collection among the 11 EVAL
  collections, the median absolute residual around the panel's 4-parameter logistic fit of the model's score to the
  human score, divided by the collection's target span (`aic_panel.geometry` arrays), plus its 90th percentile and
  zenstats `z_rmse`; three-seed means for every arm including control, pool-1 and pool-10. The 2AFC collections
  (fg-iqa, pieapp) have no MOS and are left out of this table. EVAL is the held-out guard set: choosing an arm by these
  numbers would make it selection data, so they are reported alongside and do not change the recommendation.

## Amendment results (2026-11-02): pool-30 and pool-100

Same measurements as above, all five arms side by side (three-seed means; the control is unchanged).

| | control | pool-1 | pool-10 | pool-30 | pool-100 |
|---|---:|---:|---:|---:|---:|
| **Primary**: selection inversion share | 3.358% | 3.170% | 2.714% | 2.349% | 1.896% |
| minus control [95% interval, 40 sources] | | -0.188 [-0.269, -0.124] | -0.643 [-0.841, -0.484] | -1.009 [-1.211, -0.827] | -1.462 [-1.680, -1.252] |
| G1 C1-C6 (both grids), identity, NEARID | pass | pass | pass | pass | pass |
| G1 NEARID monotone ladders (s0/s1/s2) | 140/140/140 | 140/140/140 | 140/139/139 | 139/139/139 | 140/139/139 |
| G1 G-STEER of 135 (s0/s1/s2) | 135/135/133 | 135/135/134 | 135/135/134 | **133/133/134** | **134/134/134** |
| G2 EVAL worst collection vs control | | -0.0011 | -0.0032 | -0.0059 | -0.0070 |
| G2 EVAL mean vs control | | +0.0002 | +0.0010 | +0.0030 | +0.0064 |
| G3 G-TRADE (a) >64 px | 2.36% | 2.30% | 2.36% | 2.24% | 1.93% |
| G4 G-CHROMA-L worst MAE ratio (family) | | 0.997 | 1.018 (ycc420) | **1.0508 (ycc420)** | **1.0514 (ycc420)** |
| Error vs human, median / p90 (mean of 9 MOS collections) | 8.55% / 21.87% | 8.52% / 21.82% | 8.41% / 21.70% | 8.35% / 21.45% | 8.16% / 21.15% |

G-STEER's failing cases differ by seed (pool-30: broad-38-b64 and broad-143-b64 / broad-38-b32 and broad-38-b64 /
broad-97-b64; pool-100: I21_10_05 / I21_10_05 / broad-17-b64; control seed 2: broad-97-b64 and dog-256-50).

**Decision rule.** pool-30 and pool-100 do not meet it: G-CHROMA-L's ycc420 MAE is 5.08% and 5.14% above the
control's (bar 5%), and G-STEER is below F3's 135/135 on seed 0 and below the control's three-seed mean (134.33)
for both. pool-10 remains the arm that meets every registered condition. Error against human ratings falls
monotonically with lambda (reported, not used to select).

## Amendment 2 (2026-11-02, before any experiment 1, 2 or 4 run): level anchor, pool refresh, pool ablations

Owner, 2026-11-02: "work to optimize this so it can run 4x faster, then proceed through all of these". Every arm below
is F3's general continuation (same recipe, seeds 0-2, packed at epoch 59) with the trainer build named in its results,
whose control reproduces F3's 60 checkpoints; primary, guards and decision rule are unchanged; error against human
ratings is reported as in amendment 1.

**Experiment 1, level anchor.** The hinge constrains order only; pool-30 and pool-100 drifted G-CHROMA-L's ycc420
level 5.1% while keeping per-ladder rank. A level term `--level-pool` holds the raw output on TRAIN chroma ladder
rows near the control's: 32,768 rows drawn (seed 20261102) from the merged chroma bank's `fit` and `new-train` roles
(`/mnt/v/zensim/e36-features/2026-11-01/level-chroma-32k`, scripts/byconstruction/level_pool.py; ycc420 7,340,
ycc444 8,088, xyb-extended 6,749, band-ac 6,739, xyb-capped 3,852 rows)
(never `dev` or `calibration`, which G-CHROMA-L reads), features in the same derived-input read set and scaler as the
training rows; target = the same seed's control checkpoint (epoch 59) raw output on each row, computed by the trainer
at start; every 50 primary pairs, 32 rows on their own RNG stream, loss `mu * mean((raw - target)^2)` added to that
step's gradient. Arms: pool-30+level and pool-100+level, mu = 1. The term's finite-difference gradient test passes
(zensimx 2857ca0c). The control (no pool, no level) runs with the same build in the same set; no arm result is read
unless it reproduces F3's 60 checkpoints.

**Experiment 2, pool refresh.** Pool v2 = every scored fit pair inverted by F3 seed 0 or by pool-100 seed 0, plus an
equal random sample (seed 20261101) of pairs neither inverts; duplicate view pairs dropped as before; same classes and
weights. Built: 66,694 inverted + 67,084 anchor pairs after 444 duplicates (`pool-v2-F3-p100s0`). Arms: pool2-30 and pool2-100.

**Experiment 4, pool ablations at lambda 30** (pool v1 unless named): flat class weights (1,1,1,1); margin 0.02;
margin 0.1; three anchors per inverted pair (`pool-F3-ar3`: 61,376 inverted + 185,145 anchors). The flat-weight
pool's pairs are byte-identical to pool v1; only its manifest's class weights differ (`pool-F3-flat`).

**Selection among arms.** Among arms that meet the decision rule, the recommendation is the lowest primary; ties
within 0.1 points go to the lower lambda. Runs go up to 16 at a time on dev (each trainer peaks at about 2 GiB) and, after a per-host control, on i270 and r3800x.

## Amendment 2 results (2026-11-02)

Trainer build `d2e89524` (zensimx 2857ca0c); its control (no pool, no level, seed 0) reproduces F3's 60 checkpoints
after stripping provenance, so the arms below are read. Three-seed means against the same control as above.

| arm | primary (selection inversions) | minus control [95%] | G-CHROMA-L worst ratio | EVAL worst / mean | G-TRADE (a) | G-STEER s0/s1/s2 | NEARID, C1-C6, identity | rule |
|---|---:|---|---|---|---:|---|---|---|
| p30L (lambda 30 + level) | 2.382% | -0.975 [-1.181, -0.799] | 0.998 (ycc420) | -0.0053 / +0.0032 | 2.20% | 135/134/135 | pass | **meets** |
| p100L (lambda 100 + level) | 1.911% | -1.447 [-1.663, -1.235] | 1.028 (ycc444) | -0.0053 / +0.0073 | 1.91% | 134/133/134 | pass | G-STEER |
| p2-30 (pool v2, lambda 30) | 2.375% | -0.983 [-1.183, -0.804] | 1.025 (ycc420) | -0.0086 / +0.0022 | 2.14% | 135/135/135 | pass | **meets** |
| p2-100 (pool v2, lambda 100) | 1.888% | -1.470 [-1.674, -1.264] | **1.078** (band-ac) | **-0.0206** / +0.0038 | 1.84% | 134/133/135 | pass | 3 guards |
| p30-flat | 2.383% | -0.975 [-1.177, -0.795] | **1.054** (ycc420) | -0.0055 / +0.0026 | 2.32% | 132/134/134 | pass | 2 guards |
| p30-m002 (margin 0.02) | 2.358% | -1.000 [-1.203, -0.818] | **1.051** (ycc420) | -0.0059 / +0.0030 | 2.23% | 132/134/134 | pass | 2 guards |
| p30-m01 (margin 0.1) | 2.339% | -1.019 [-1.221, -0.836] | **1.051** (ycc420) | -0.0049 / +0.0031 | 2.24% | 131/135/132 | pass | 2 guards |
| p30-ar3 (3 anchors per inversion) | 2.588% | -0.770 [-0.977, -0.599] | 1.027 (ycc420) | -0.0049 / +0.0015 | 2.35% | 135/134/135 | pass | **meets** |

Control: 3.358% primary, G-TRADE 2.36%, G-STEER 135/135/133. EVAL worst is the most negative collection difference
(irccyn-ivc for every arm); bars -0.01 / -0.003. G-STEER failures: the lambda-100 arms fail I21_10_05 on m3f
(0.674-0.698, bar 0.70) with or without the level term; the lambda-30 ablations fail m2 (local linearity) on
broad-38/97/143 at 32 and 64 px blocks.

Error against human ratings (median / p90 absolute logistic-fit residual over the target span, mean of the 9 MOS
collections, three-seed means; reported, not used to select): control 8.55% / 21.87%; p30L 8.39% / 21.48%;
p100L 8.13% / 21.14%; p2-30 8.44% / 21.55%; p2-100 8.17% / 21.29%; p30-flat 8.36% / 21.47%; p30-m002 8.36% /
21.45%; p30-m01 8.36% / 21.43%; p30-ar3 8.41% / 21.66%.

**Readings.** The level term does what it was registered to do: at lambda 30 it brings G-CHROMA-L's ycc420 ratio from
1.051 (pool-30) to 0.998 at a primary cost of 0.034 points, and it holds every guard. It does not rescue lambda 100's
G-STEER case. Refreshing the pool (p2-30) gives the same primary as pool-30 with a clean G-STEER on all seeds; at
lambda 100 it costs held-out human ratings (irccyn-ivc -0.021). Class weights and margin are not what limits lambda
30: flat weights and both margins land within 0.05 points of pool-30 and keep its ycc420 failure. Three anchors per
inversion trade primary (-0.77) for chroma margin.

**Selection under the registered rule.** p30L, p2-30 and p30-ar3 meet the rule (with pool-1 and pool-10). The lowest
primary is p2-30 (2.375%) by 0.007 points over p30L (2.382%); the tie rule (lower lambda within 0.1 points) does not
separate them. The registered recommendation is therefore p2-30; p30L is within its interval and better on chroma
level, EVAL worst and error against human ratings.

Process: 25 runs in three waves of 7-9 on dev (about 15 min each); measurement chain 68 min, of which G-STEER (24
models one after another, 27 shards each) took 44 min.

## Amendment 3 (2026-11-02, before any of these arms runs): combination and the lambda boundary

Amendment 2 left two questions: whether pool v2 and the level term combine, and where between lambda 30 and 100
G-STEER's I21_10_05 case breaks with the level term. Same build (`d2e89524`, its control reproduced F3), recipe,
control, primary, guards, decision rule and selection rule as amendment 2.

| arm | pool | lambda | level term |
|---|---|---:|---|
| p2-30L | pool v2 (`pool-v2-F3-p100s0`) | 30 | as p30L |
| p2-50L | pool v2 | 50 | as p30L |
| p50L | pool v1 (`pool-F3`) | 50 | as p30L |

## Amendment 3 results (2026-11-02)

Same build and control (60/60 again). Three-seed means against the same control.

| arm | primary | minus control [95%] | G-CHROMA-L worst | EVAL worst / mean | G-TRADE (a) | G-STEER s0/s1/s2 | error vs human (median / p90) | rule |
|---|---:|---|---|---|---:|---|---|---|
| p2-30L | 2.404% | -0.954 [-1.155, -0.778] | 0.984 (ycc444) | -0.0088 / +0.0023 | 2.05% | 135/135/134 | 8.45% / 21.54% | **meets** |
| p2-50L | 2.175% | -1.183 [-1.380, -0.995] | 0.991 (band-ac) | **-0.0117** / +0.0033 | 1.90% | **131**/134/135 | 8.34% / 21.31% | 2 guards |
| p50L | 2.171% | -1.186 [-1.386, -0.996] | 1.006 (ycc444) | -0.0075 / +0.0046 | 2.05% | **132**/133/135 | 8.34% / 21.30% | G-STEER |

Both lambda-50 arms fail G-STEER's I21_10_05 on m3f (0.685-0.697) on two seeds, as lambda 100 did; p2-50L adds
broad-97 linearity misses and irccyn-ivc -0.0117. So the boundary for the level-anchored hinge lies between lambda 30
and 50, and it is the I21_10_05 steering case that sets it. Combining pool v2 with the level term (p2-30L) gives the
best chroma level of any arm (every family below the control's MAE) at a primary between p30L and p2-30.

Arms meeting the registered rule, by primary: p2-30 (-0.983), p30L (-0.975), p2-30L (-0.954), p30-ar3 (-0.770),
pool-10 (-0.643), pool-1 (-0.188). The three lambda-30 arms are within 0.03 points of each other.

Process note: G-STEER with 30 concurrent shard processes ran at ~10 s per case against ~1.5-2 s alone (the
instrument is memory-bound on dev); about 9 concurrent shards, one model at a time, is the efficient setting.

## Experiment 3a (2026-11-02, before any loop run): do the candidates' maps steer JPEG XL better?

The owner's ruling: map choices are decided in the JPEG XL loop, by BD-rate. F3 and the hinge arms have never
steered it. Their attribution maps already agree more with butteraugli's on 240 E36 TRAIN views (served exact-split
map, median per-view Spearman of 8 px block sums: F3 0.700, p30L 0.740, pool-100 0.764), which is a reason to look,
not a result.

- **Arms** (seed-0 packed models, served map, linear float input, tail placement fitted by `loop_calibrate`
  `LOOP_CAL_TAIL=1` on the 2026-10-24-hq calibration pairs, as the Mdata study's `*lintail` arms): F3, p30L, p2-30.
- **Controls** (stop before any arm's result is read if either fails): the new build's `zensim-B@rev5` and
  `zensim-Mfull@rev7` placement rows equal the 2026-10-25r7win calibration's exactly, and their check arms reproduce
  the stored B and M7lintail bitstreams (702 each) byte for byte.
- **Everything else as the steering study:** the 78 TRAIN images, effort 7, 4 iterations, distances 0.3-10, the same
  judge build (`zenmetrics-judge-7a0f8394`).
- **Statistic:** BD-rate (`rd_probe_analyze --bdrate-vs`) of each arm against the butteraugli loop and against F3,
  under SSIMULACRA2 and CVVDP. Reading: an arm steers better when its BD-rate saving is positive with the interval
  above zero under both judges; an interval across zero under either is a tie.

### Experiment 3a amendment (2026-11-02, before any arm was judged)

The repo's host guard refuses `jxl_steer_rd.sh` on dev (benchmarks never run there), so the loop runs on i270. There
the registered bitstream control fails for a host reason, not a build one: the new build's B arm differs from the
stored (dev-encoded) B bitstreams in 74 of 702 encodes, and so does the original study build (`3ba4f4ba1cb8`) run on
i270; the new build reproduces the original build on i270 in 702 of 702 (zensim's map extraction is tier-dependent,
dev has AVX-512 and i270 does not). Stored dev arms are therefore not comparable with i270 encodes.

Amended: every compared arm is encoded on i270 in a fresh RD directory (`none`, `butteraugli`, `B`, `M7lintail`,
`F3lintail`, `p30Llintail`, `p230lintail`) and judged there with the same judge build; the control is "the new build
reproduces the original build's B arm on the same host" (702 of 702, measured above). Statistic, placements and
reading unchanged. No arm had been judged when this was written.

### Experiment 3a results (2026-11-02, i270)

Control passed: the new build's B arm equals the original study build's B arm on i270 in 702 of 702 encodes. All
seven arms encoded and judged on i270 (78 TRAIN images, effort 7, 4 iterations, 9 distances; judge build
`zenmetrics-judge-7a0f8394`). BD-rate, % fewer bytes (positive = the arm saves bytes), 95% bootstrap over images.

| arm vs F3 | all: SSIMULACRA2 | all: CVVDP | photo: SSIMULACRA2 | photo: CVVDP | screen: SSIMULACRA2 / CVVDP |
|---|---|---|---|---|---|
| p30L | +0.03 [-0.17, +0.22] | +0.12 [-0.16, +0.38] | +0.10 [-0.02, +0.23] | **+0.17 [+0.05, +0.30]** | +0.37 / +0.17 (both across 0) |
| p2-30 | +0.11 [-0.11, +0.33] | +0.37 [-0.07, +0.81] | **+0.19 [+0.03, +0.37]** | **+0.26 [+0.15, +0.40]** | +0.48 / +0.95 (both across 0) |
| M7 (reference) | +0.15 [-0.16, +0.47] | +0.12 [-0.45, +0.67] | +0.58 (CI above 0) | +0.56 (CI above 0) | -0.30 / -0.77 |
| butteraugli loop | +2.22 [+1.48, +3.05] | +4.08 [+3.31, +5.00] | +1.34 | +3.68 | +5.12 / +6.77 |

Against the butteraugli loop every zensim map costs bytes over all images: F3 -2.42% (SSIMULACRA2) / -4.44% (CVVDP),
p30L -2.39 / -4.34, p2-30 -2.34 / -4.10, M7 -2.32 / -4.32 (B, bulk placement: -3.32 / -5.53).

**Reading (the registered rule).** Neither p30L nor p2-30 steers JPEG XL better than F3 over all images: both are
ties. On photos p2-30 is better under both judges (+0.19% / +0.26%), p30L under CVVDP only. The hinge arms' higher
agreement with butteraugli's map (0.70 -> 0.74 at 8 px) buys at most a few tenths of a percent, and every zensim map
remains 2-4% behind butteraugli's own loop. The candidates cost nothing in steering, which removes the loop as an
obstacle to adopting one. Encodes, judge tables and BD-rate tables: `/mnt/tower/output/zensim/e36-subset-test-2026-11-01/exp3a-jxl-i270/`
(SHA256SUMS).

## Experiment 5 (2026-11-02, before any run): which human MOS sources transfer to the other MOS sets?

Owner, 2026-11-02: "experiment with which human mos, or pair of mos, produce the best score medians in the other
human mos sets". Base fits from scratch (F3's `base-fit` step: `sel:45acffa603e8@h32:H128:cv16:cf98:fx2:mono:thr:mp32`,
strict admission, packed), trainer build `d2e89524`, with the non-human legs unchanged (SafeSyn and CID22 oracle
teachers, coverage, BAPPS, chroma teacher) and the human MOS training legs varied.

- **MOS sources (11, TRAIN parts only):** the design-released human table's four sources (KADID, TID2013, KonFiG,
  CID22-A25; split by each row's `member_set` into strict-admissible subset tables with their own pins) and the
  dataset legs PIPAL, MDID, LIVE-MD, CID-IQ, CID-OA, IRCCyN-IVC, Colourlab.
- **Mechanism:** a recorded copy of the frozen scripts (`~/tmp/e36test/src-exp5`, `v2_confirm_fit.py` +
  `human_subset.py`) adds `--human-sources`, `--drop-groups` and `--mos-share`; dataset legs use the existing
  `--exclude-datasets`. Every strict admission check still runs.
- **Weight:** in every arm the kept MOS groups share the full recipe's total MOS train weight (36.035) equally.
- **Stage 1 (seed 0):** control (no new option), `allsub` (all four human sources through the subset path; must equal
  the control), `none` (no MOS group), `all-equal` (all 11 kept, equal split), and each of the 11 sources alone.
- **Stage 2:** pairs among the five best singles (10 arms, seed 0); then the three best configurations of stages 1-2
  at seeds 0-2.
- **Outcome (owner's measure):** per arm, the mean over the EVAL MOS collections the arm did not train on of the
  median absolute residual around each collection's logistic fit, over its target span (the human-error readout of
  amendment 1); also SROCC, and the TRAIN-side human_development and cid22_development values the trainer logs.
- **Caveat stated before results:** choosing a training mix by EVAL makes EVAL selection data for this question; a
  final claim about a chosen mix needs a confirmation on collections no arm used (other tower IQA library sets).

## Experiment 6 (2026-11-02, before any run): pruning human rows, and coverage that does not depend on the seed

Owner, 2026-11-02: "how we can prune rows from training or ensure a wider set of source images are involved in
training regardless of seeds, in order to raise scores". Base fits as in experiment 5 (F3's `base-fit` step, strict
admission, packed); the control is experiment 5's `control-s0`.

**Where the seed decides coverage (from the F3 recipe's weights, 120 epochs x 50,000 draws = 6.0 M draws per run).**
The human leg (10,590 rows over 100 references) gets 52% of draws, about 600 per row; the coverage leg, CID22 and the
small MOS legs are similarly saturated, so every one of their rows is used whatever the seed. Three legs are not:
SafeSyn (141,054 rows over 2,312 references, about 99 k draws), the chroma teacher (231,367 rows over 5,645
references, about 101 k draws) and BAPPS (223,068 rows in 111,534 two-row references, about 97 k draws, half of them
same-row skips). Which of their rows, and which BAPPS references, a run sees is a property of the sample seed. The
measured per-leg coverage is in each run's `[coverage]` line and embedded `zentrain.sample_coverage`.

**Arms (seed 0):**

- `walk`: new `--pair-sampling walk` (this commit). It keeps the uniform draw's group sequence and random-number
  consumption, so group weights and per-group draw counts are unchanged. Within a within-reference group, the
  reference is the next one on a seeded round-robin and row A is the reference's next row on a seeded bijection.
  Row B stays uniform over the reference's rows, so same-row skip rates and effective group weights are also
  unchanged. Every reference and row of a group is visited before any repeats. Run with `ZENSIM_SAMPLE_DIGEST=1`:
  the trainer's digest must equal the coverage replay's.
- `control-wb`: the walk build in uniform mode. It must reproduce `control-s0`'s 120 checkpoints bit-exactly, or the
  walk arm has no control.
- `strat`: the existing `--pair-sampling stratified` (2026-09-04). It ignores group weights: each (reference,
  quality band) stratum gets equal draws, and pairs come from one band. It is a different mixture, run because it is
  the existing seed-independent mode.
- `prune-g5`, `prune-g10`: drop the 5% / 10% of human fit rows with the highest within-reference discordance under
  the F3 base-fit seed-0 bake (`full_s0/refit/production-f16.bin`, sha256 `6afaf106…`). A row's discordance is
  `sum_j |y_i - y_j| [sign(p_i - p_j) != sign(y_i - y_j)] / sum_j |y_i - y_j|` over the rows of its reference
  (y = human score, p = bake score). Over all rows the median is 0.036 and the 90th percentile 0.110.
- `prune-s10`: drop the top 10% within each source, so the source mix is kept.
- `prune-r10`: drop a random 10% (numpy `default_rng(20261102)`). This separates "fewer rows" from "these rows".
- The human group's weight and its 100 references are unchanged in every prune arm.

| source | rows | median discordance | share in the global top 10% |
|---|---:|---:|---:|
| CID22-A25 | 1,579 | 0.011 | 0.4% |
| KADID | 6,659 | 0.040 | 10.3% |
| KonFiG | 672 | 0.026 | 17.7% |
| TID2013 | 1,680 | 0.047 | 14.6% |

**Outcome:** experiment 5's readout over all nine EVAL MOS collections (every arm trains the same six dataset legs):
the mean over collections of the median absolute residual around each collection's logistic fit (over the target
span), and SROCC; also the trainer-logged development values. **Stage 2:** arms whose seed-0 result beats the
control by more than the spread of the control's seeds 0-2 run at seeds 0-2. The prune lists were chosen with a seed-0
model of the same recipe, so seeds 1-2 are the cleaner test.

**Caveats stated before results:** as in experiment 5, choosing by EVAL makes EVAL selection data for this question.
A pruning arm that wins removes rows a model of this recipe disagrees with; it is not evidence that those ratings are
wrong.

### Experiment 6 amendment (2026-11-02, before the walk and strat arms trained)

- The trainer admits subjective (MOS and BAPPS) legs only on the uniform pair path. `walk` keeps that path's
  within-reference pairs, random stream and same-row skips; BAPPS's two-row comparisons are drawn exactly as uniform
  draws them. The gate now also accepts `walk` (CLI and the native-vote check). The walk arm trains on that build;
  it differs from the `control-wb` build only in those two admission checks.
- `strat` cannot run under this recipe. `--pair-sampling stratified` pairs only inside (reference, quality band)
  strata, so each BAPPS comparison whose two sides fall in different bands becomes a singleton stratum and is
  dropped, and the gate refuses it. The arm is recorded as not run, for that reason.

## Experiment 3, steps 2-4 (2026-11-02, before any map-term run): fit the attribution map to butteraugli

Step 1 (above, experiment 3a) measured headroom: on 240 E36 fit views, the per-view Spearman between a model's
attribution map and butteraugli's diffmap at 16 px blocks is 0.713 for F3, 0.752 for p30L and 0.760 for pool-100.
The owner trusts butteraugli spatially, and the JPEG XL loop with butteraugli's map leads every zensim map by 2-4%.

**Step 2, data.** `BakeScorer::feature_block_integrands` (PR #75, under review) gives each read feature's block
integrand `I_k(b)`; the served density is `sum_k s_k I_k`. The `integrand_dump` example writes them for a bake's
read set, with butteraugli's block sums on the same grid. The sample is 1,967 E36 fit views (128-1024 px), stratified
by codec x content x quality band (seed 20261103). `scripts/byconstruction/map_pool.py` joins the dump to the views'
bank feature rows as a `map-pool-v1`. Integrand rows that are all zero (reference-only ids) are dropped, and the
targets are each view's butteraugli block sums and their within-view rank z-scores.

**Step 3, the term.** `zensim_mlp_train --map-pool` (beside `--tv-pool`; `mlp_train/map_term.rs`). On each scheduled
pool step it draws `--map-batch` views on the pool's RNG stream and adds
`--map-weight * mean(1 - pearson(m, t))`. Here `m_b = sign * sum_k dy/df_k I_k(b)` and `t` is the rank z-score
target. The sensitivities run back through the derived inputs and the network analytically; a finite-difference
test checks the gradient.

**Arms (continuations from F3's base fit, as p30L; seed 0 first).** p30L plus the map term at `--map-weight` 1, 10
and 30 (batch 8, 16 px blocks). The control is p30L's seed-0 run from amendment 2. Before any arm runs, a fidelity
check must pass: on dumped views, `sum_k s_k I_k` from the served sensitivities must reproduce the same build's served
density (max block error under 1e-4 of the largest block).

**Outcome.**
1. Map agreement on 240 held-out E36 selection views (seed 20261104): median per-view Spearman against butteraugli
   at 8 and 16 px blocks.
2. The amendment-2 measurements: primary, error against human ratings, G-CHROMA-L and G-STEER.
3. For the arm with the best map agreement that keeps the product gates: the JPEG XL loop (experiment 3a protocol,
   i270), BD-rate against p30L.

**Caveats stated before results.** The map term supervises the frozen density. The served revision-7 map replaces
the coarse v2 pools (f459-719) with exact neighbour deltas, so the loop can differ from what the term fits.
Butteraugli is a model of human vision, not a human rating.

### Experiment 3 amendment (2026-11-02, at the first map-term batch)

The map arms first started with `--map-sign -1`. Their first batch logged `1 - pearson` 1.684, a map running
opposite to butteraugli, and they were stopped at epoch 0 before any checkpoint. They rerun with `--map-sign 1`,
which is now the default. `p30L-mapbuild` (no map term) runs on the same build and must reproduce amendment 2's
p30L seed 0.

## Experiment 5 stage 1 and experiment 6 seed 0: results (2026-11-02)

Base fits at seed 0, scored on EVAL's nine MOS collections (error = median absolute residual around each
collection's logistic fit, over the target span, averaged over collections). "Held-out" averages over the
collections the arm did not train on, so its set differs by arm; the three collections no arm trains on (IQA-ODI,
NITS-IQA, VCL-FER) are the common comparison. Readout `scripts/byconstruction/e36_exp56/readout.py`.

| arm (seed 0) | held-out sets | held-out median | 3 shared sets: median / SROCC | all 9 median |
|---|---:|---:|---:|---:|
| tid2013 | 9 | 8.33% | 8.27% / 0.853 | 8.33% |
| pipal | 9 | 8.33% | 8.97% / 0.827 | 8.33% |
| cid-iq | 8 | 8.45% | 8.21% / 0.856 | 8.42% |
| mdid | 8 | 8.46% | 8.45% / 0.847 | 8.21% |
| kadid | 9 | 8.51% | 9.08% / 0.829 | 8.51% |
| konfig | 9 | 8.65% | 8.71% / 0.836 | 8.65% |
| cid-oa | 8 | 8.70% | 8.29% / 0.855 | 8.69% |
| live-md | 8 | 8.72% | 8.17% / 0.864 | 8.57% |
| x6-prune-g5 | 3 | 8.77% | 8.77% / 0.836 | 8.33% |
| cid22_a25 | 9 | 8.78% | 8.33% / 0.859 | 8.78% |
| all-equal | 3 | 8.80% | 8.80% / 0.836 | 8.35% |
| control | 3 | 8.81% | 8.81% / 0.834 | 8.38% |
| allsub | 3 | 8.81% | 8.81% / 0.834 | 8.38% |
| x6-control-wb | 3 | 8.81% | 8.81% / 0.834 | 8.38% |
| none | 9 | 8.85% | 8.42% / 0.856 | 8.85% |
| x6-prune-r10 | 3 | 8.86% | 8.86% / 0.835 | 8.35% |
| x6-prune-g10 | 3 | 8.87% | 8.87% / 0.837 | 8.38% |
| x6-prune-s10 | 3 | 8.93% | 8.93% / 0.833 | 8.43% |
| colourlab | 8 | 9.27% | 9.90% / 0.803 | 9.38% |
| irccyn-ivc | 8 | 9.53% | 8.63% / 0.857 | 9.17% |

- `allsub` (all four human sources through the subset path) reproduces the control's scores exactly, and `control-wb`
  (the walk build in uniform mode) reproduces the control's checkpoints bit-exactly (2/2).
- **Every single MOS source beats the full mix on the collections it did not train on** except IRCCyN-IVC and
  Colourlab. TID2013 alone is best by the registered measure (8.33% vs 8.81%) and among the best on the three shared
  collections (8.27%, SROCC 0.853 vs 0.834). With no MOS leg at all, error on the shared collections (8.42%) is lower
  than with the full mix (8.81%); the MOS legs mainly fit the collections they come from (MDID 7.71% vs 9.92%,
  Colourlab 12.10% vs 15.06%).
- Stage 2 runs the 10 pairs among the five best singles (TID2013, PIPAL, CID-IQ, MDID, KADID).
- **Experiment 6 at seed 0:** no pruning arm moves the all-nine error beyond what the control's own seeds may spread
  (control 8.38%; prune 5% 8.33%, random 10% 8.35%, discordance 10% 8.38%, per-source 10% 8.43%). Control seeds 1-2
  are training. The walk arm reruns on the build that admits it.
- One seed each: these rank candidates; the registered stage 2 confirms at seeds 0-2.

### Experiment 5 stage 2 and experiment 6, seed 0 (2026-11-02), and an amendment before any further seed

Control base fits at seeds 0-2 give all-nine error 8.38%, 8.71% and 8.38%, and shared-three error 8.81%, 9.41%
and 8.90%: the seed alone moves these by up to 0.6 points. Pairs (seed 0, held-out / shared-three / all nine):
TID2013+MDID 8.31% / 8.38% / 8.11%; PIPAL+MDID 8.39% / 8.83% / 8.18%; MDID+KADID 8.43% / 8.84% / 8.24%;
CID-IQ+KADID 8.54% / 8.98% / 8.41%; CID-IQ+MDID 8.56% / 8.30% / 8.12%; PIPAL+CID-IQ 8.58% / 8.86% / 8.46%;
TID2013+CID-IQ 8.62% / 8.37% / 8.61%; TID2013+PIPAL 8.63% / 9.37% / 8.63%; TID2013+KADID 8.48% / 9.03% / 8.48%;
PIPAL+KADID 9.08% / 9.67% / 9.08%. The three best configurations by the registered measure, TID2013+MDID, TID2013 and
PIPAL, now run at seeds 1-2.

Experiment 6: no arm beats the control by more than the control's seed spread (0.33 points all-nine), so by the
registered rule no pruning arm goes to stage 2. The walk (all-nine 8.61% at seed 0; pooled row coverage 0.630 vs
0.563; its trainer digest equals the coverage replay's) is for seed independence, which one seed cannot show.
**Amendment, before running:** the walk also runs at seeds 1-2. Its outcome is the spread of all-nine error across
seeds 0-2 against the control's (0.33 points), with the mean reported beside it.

### Experiment 5 three-seed confirmation (2026-11-02)

| configuration | held-out error, seeds 0 / 1 / 2 (mean) | shared three: error / SROCC (means) | all nine (mean) |
|---|---|---:|---:|
| full MOS mix (control) | 8.81 / 9.41 / 8.90 (9.04%) | 9.04% / 0.832 | 8.49% |
| TID2013 + MDID | 8.31 / 8.53 / 8.30 (8.38%) | 8.45% / 0.846 | 8.17% |
| TID2013 alone | 8.33 / 8.55 / 8.47 (8.45%) | 8.46% / 0.849 | 8.45% |
| PIPAL alone | 8.33 / 8.71 / 8.33 (8.46%) | 9.04% / 0.828 | 8.46% |

TID2013 + MDID (the human leg restricted to TID2013's rows and the MDID dataset leg, sharing the full MOS weight
equally) is better than the full mix at every seed: its worst held-out seed (8.53%) beats the control's best (8.81%),
and on the three collections no arm trains on it lowers error by 0.59 points and raises SROCC by 0.014. These are
base fits; EVAL chose the configuration, so EVAL is now selection data for it.

### Experiment 5 stage 3 (registered 2026-11-02, before any run): the TID2013 + MDID mix through the full recipe

- **Arms (seeds 0-2):** F3's continuation step (60 epochs, AIC x256) from the TID2013 + MDID base fits, with the same
  MOS mix in the continuation, plus p30L's hinge and level terms (pool-F3 lambda 30, level weight 1, the level
  reference unchanged: F3's control at epoch 59). Control: p30L seeds 0-2 (amendment 2).
- **Measures:** amendment 2's set (primary, error against human ratings over EVAL, G-CHROMA-L dev, G-STEER, NEARID) and
  the held-out map agreement of experiment 3.
- **Confirmation on data no arm selected with:** the tower IQA library's MOS datasets that are in no training leg and
  not in EVAL, scored with each dataset's official test split. Which datasets qualify is resolved from the library
  catalogue before any is scored, and listed here with the result.

### Experiment 3 step 3, seed 0: held-out map agreement, and an amendment (2026-11-02)

The map build without the map term reproduces p30L's seed-0 checkpoints (60/60 identical). Median per-view Spearman
against butteraugli on the 240 held-out selection views (10th percentile in brackets):

| arm | 8 px, full density | 8 px, served | 16 px, full density | 16 px, served |
|---|---:|---:|---:|---:|
| F3 | 0.706 (0.413) | 0.702 (0.366) | 0.784 (0.436) | 0.764 (0.478) |
| p30L (map build, no term) | 0.738 (0.440) | 0.754 (0.422) | 0.801 (0.462) | 0.794 (0.476) |
| p30L + map 1 | 0.738 (0.444) | 0.754 (0.421) | 0.801 (0.462) | 0.793 (0.476) |
| p30L + map 10 | 0.740 (0.452) | 0.756 (0.424) | 0.802 (0.467) | 0.794 (0.480) |
| p30L + map 30 | 0.745 (0.462) | 0.764 (0.431) | 0.805 (0.479) | 0.797 (0.483) |

The effect grows with the weight but is small at 30. The continuation trains at a constant 1e-5 for 60 epochs.
**Amendment, before running:** p30L + map at weights 100, 300 and 1000 (seed 0, same pool and batch), measured the
same way; the product gates follow for any arm whose agreement gain over p30L exceeds 0.02 at 8 px.

### Experiment 3 amendment 2 (2026-11-02, before any run): the map term in the base fit

The continuation trains at a constant 1e-5, so a term added there has little room. The base fit (120 epochs from
scratch, cosine restarts from 1e-3) is where the network forms. `--map-pool` without `--tv-pool` now runs the map
term alone on the pool schedule (every 50 primary steps, batch 8). **Arms:** F3's base fit plus the map term at
weights 10, 30 and 100, seed 0. Control: experiment 5's control-s0, which reproduces F3's base fit. **Measures:**
held-out map agreement, primary, and error against human ratings on the base fits. If agreement gains more than 0.02
at 8 px over the control, the arm continues through p30L's continuation and the product gates.
Early-run results so far: G-CHROMA-L passes for p30L + map 1/10/30. Against p30L, map 30 moves the primary from
-0.964 to -1.013 (CI -1.22 to -0.83) and all-nine error against human ratings from 8.40% to 8.32%.
G-STEER for the continuation map arms: map 1 passes 135/135, map 10 and map 30 each fail one case (broad-206-b64,
m2 0.958; dog-256-20, m2 0.941; the bar is m2 >= 0.99 and m3f >= 0.70). The no-map reproduction passes 135/135. On
dog-256-20 map 30 raises m3f (the map's rank agreement with the true repair gains) from 0.851 to 0.902, but m2 (the
first-order prediction from the exact feature changes against the true score change) falls from 1.000 to 0.941: the
term makes the score less linear around that case. That is the same failure as the hinge at lambda >= 50.

### Experiment 6 result: the walk makes training less seed-dependent (2026-11-02)

| sampler | all-nine error, seeds 0 / 1 / 2 | mean | spread | shared-three spread | pooled row coverage |
|---|---|---:|---:|---:|---:|
| uniform (control) | 8.38 / 8.71 / 8.38 | 8.49% | 0.33 | 0.61 | 0.563 |
| `--pair-sampling walk` | 8.61 / 8.51 / 8.60 | 8.57% | 0.10 | 0.21 | 0.630 |

The walk cuts the seed-to-seed spread of error against human ratings about threefold at a similar mean (8.57% vs
8.49%, well inside the control's own spread). Three seeds per arm make the spread estimate coarse. Pruning (seed 0)
gave nothing beyond the control's spread, and stratified sampling cannot run under this recipe. Recipe consequence:
the walk is a candidate default for E36-family base fits, because it makes a single-seed comparison more
trustworthy; it does not by itself lower error.

### Experiment 3 amendment 2 result and amendment 3 (2026-11-02, before the amendment-3 runs)

Base fits with the map term (seed 0), held-out agreement at 8 px (median per-view Spearman against butteraugli,
10th percentile in brackets):

| base fit | full density | served map |
|---|---:|---:|
| control (F3's base fit) | 0.720 (0.441) | 0.751 (0.466) |
| + map 10 | 0.728 (0.449) | 0.756 (0.469) |
| + map 30 | 0.744 (0.474) | 0.762 (0.478) |
| + map 100 | 0.792 (0.527) | 0.779 (0.454) |

Inside the base fit the term works: at weight 100 the full density gains 0.072 and the served map 0.027, past the
registered 0.02. **Amendment 3, before running:** (a) p30L's continuation from the map-30 and map-100 base fits
(seed 0), measured with the amendment-2 set including G-STEER; (b) the map term at weight 300 in the base fit (seed 0),
since the gain still rises with weight; (c) map-100 base fits at seeds 1-2.

### Experiment 3, continuation map weights 100-1000 (seed 0) and amendment 4 (2026-11-02)

| arm | agreement 8 px full / served (p10 full) | agreement 16 px full / served | shared-three error / SROCC | all-nine error | G-STEER | G-CHROMA-L | human_development SROCC |
|---|---|---|---:|---:|---|---|---:|
| p30L | 0.738 / 0.754 (0.440) | 0.801 / 0.794 | 8.96% / 0.843 | 8.40% | 135/135 | pass | 0.835 |
| + map 100 | 0.760 / 0.778 (0.478) | 0.820 / 0.809 | 8.80% / 0.846 | 8.31% | 135/135 | pass | |
| + map 300 | 0.789 / 0.810 (0.539) | 0.846 / 0.836 | 8.64% / 0.852 | 8.03% | 135/135 | pass | 0.824 |
| + map 1000 | 0.836 / 0.841 (0.602) | 0.877 / 0.879 | 8.30% / 0.864 | 8.24% | 135/135 | pass | 0.799 |

p30L's seeds 1-2 give shared-three error 8.85% and 8.82%. At weights 100 and above G-STEER passes every case and its
median m3f rises (0.962, 0.968, 0.979); weights 10 and 30 had each missed one case. The cost is in-distribution: at
weight 1000 the trainer's development SROCC falls (human 0.835 to 0.799, CID22 0.984 to 0.971, SafeSyn 0.991 to
0.984); at 300 it falls less (human 0.824). The primary for these arms is being rescored (a duplicated control column
broke the first scoring pass).

**Amendment 4, before running:** p30L + map 300 and p30L + map 1000 at seeds 1-2; NEARID for both; and step 4, the
JPEG XL loop (experiment 3a protocol on i270, BD-rate) for p30L + map 300 and p30L + map 1000 against p30L and the
butteraugli loop.

**Primary for the continuation map arms (seed 0; registered primary, 95% bootstrap over the 40 selection sources):**
p30L -0.964 (CI -1.173 to -0.783, inversion share 2.387%); + map 100 -1.110 (-1.321 to -0.918, 2.240%); + map 300
-1.225 (-1.424 to -1.036, 2.125%); + map 1000 -1.322 (-1.486 to -1.168, 2.029%). The map term lowers within-ladder
misordering as well as raising map agreement and lowering error against human ratings. Its only measured cost so
far is the trainer's in-distribution development SROCC. Seeds 1-2, NEARID and the JPEG XL loop are running.

### Experiment 5 stage 3 result and amendment 5 (2026-11-02)

The TID2013 + MDID mix through the full recipe (continuation + p30L), seeds 0 / 1 / 2, against p30L's seeds:

| measure | p30L | TID2013 + MDID + p30L |
|---|---|---|
| shared-three error | 8.96 / 8.85 / 8.82% | 8.38 / 8.37 / 8.28% |
| shared-three SROCC | 0.843 / 0.843 / 0.841 | 0.857 / 0.858 / 0.858 |
| all-nine error | 8.40 / 8.37 / 8.41% | 7.86 / 7.90 / 7.85% |
| primary (mean of seeds) | -0.975 | -1.009 (CI -1.215 to -0.811) |
| G-CHROMA-L | pass | pass |
| G-STEER | 135/135 | 132 / 132 / 131 of 135 |
| held-out map agreement, 8 px full | 0.738 (seed 0) | 0.723 / 0.732 / 0.735 |

The human-agreement gain survives the full recipe at every seed, but the mix fails G-STEER: broad-220-b8 at every
seed (m3f 0.675-0.688 against 0.70), plus m2 near-misses (0.947-0.987) that change with the seed. **Amendment 5,
before running:** the mix with p30L and the map term at weight 300 (seeds 0-2), since the map term raised m3f and
kept G-STEER clean on p30L. Measured with the same set; its JPEG XL loop follows if it passes the gates.

### Experiment 3 amendment 4 result: p30L + map at three seeds (2026-11-02)

Means over seeds 0-2 (per-seed values where they matter):

| arm | primary (CI) | shared-three error / SROCC | all-nine error | agreement 8 px full / served | G-STEER | G-CHROMA-L | NEARID (seed 0) |
|---|---|---|---:|---|---|---|---|
| control (F3) | 0 | 8.80% / 0.835 | 8.55% | | | | |
| p30L | -0.975 (-1.181, -0.799) | 8.88% / 0.843 | 8.39% | 0.738 / 0.754 (seed 0) | 135, 135, 135 | pass | pass |
| p30L + map 300 | -1.241 (-1.436, -1.057) | 8.68% / 0.852 | 8.07% | 0.789-0.796 / 0.800-0.810 | 135, 135, 135 | pass x3 | pass (N3 137 ladders) |
| p30L + map 1000 | -1.339 (-1.502, -1.183) | 8.32% / 0.865 | 8.23% | 0.836-0.840 / 0.839-0.841 | 135, 134, 134 | pass x3 | pass (136) |

p30L + map 300 improves on p30L in every measure at every seed and keeps every product gate. Map 1000 is stronger on
the primary, the shared collections and agreement, but misses one G-STEER case at seeds 1 and 2 and costs more
in-distribution development SROCC. The JPEG XL loop (step 4) for both is running on i270.

### Experiment 3 step 4 result: the map term in the JPEG XL loop (2026-11-02, i270)

Every arm encoded fresh on i270 by one loop build (zensimx c7556f66, `e36_loop_map.sh`); B/M7 placement rows reproduce
the earlier calibration exactly; placements p30L 2.277e-2, map 300 3.527e-2, map 1000 4.826e-2 (exponent 0.7).
BD-rate against p30L, 78 images (positive = fewer bytes for equal judged quality; 95% CI):

| arm | ssim2 | butteraugli | zensim | DVIFM | CVVDP |
|---|---|---|---|---|---|
| p30L + map 300 | +0.51 [+0.27, +0.72] | +0.86 [+0.51, +1.18] | +0.29 [-0.08, +0.64] | +0.49 [-0.64, +1.67] | +0.86 [+0.53, +1.22] |
| p30L + map 1000 | +0.94 [+0.67, +1.19] | +1.51 [+1.24, +1.78] | +0.39 [+0.05, +0.71] | +0.36 [-0.64, +1.38] | +1.72 [+1.36, +2.06] |
| F3 | -0.04 [-0.22, +0.16] | -0.30 [-0.51, -0.12] | -0.21 [-0.49, +0.04] | +0.09 [-0.70, +0.94] | -0.13 [-0.40, +0.14] |
| butteraugli loop | +2.21 [+1.51, +3.03] | +4.55 [+3.88, +5.28] | +1.52 [+0.74, +2.32] | +2.60 [+1.36, +3.85] | +4.00 [+3.18, +4.84] |

Against the butteraugli loop (CVVDP judge): p30L -4.34%, map 300 -3.49%, map 1000 -2.53%. The map term closes about a
fifth (300) to two fifths (1000) of the gap to butteraugli's own loop, and the gain shows on the judges the term did
not train toward (ssim2, CVVDP), not only butteraugli. Full tables: `benchmarks/e36_loop_map_2026-11-02/`.

### Experiment 3 amendment 3 result: map term in the base fit, then p30L's continuation (seed 0)

| arm | agreement 8 px full / served | primary (CI) | shared-three / all-nine error | G-STEER | G-CHROMA-L |
|---|---|---|---|---|---|
| p30L | 0.738 / 0.754 | -0.964 (-1.173, -0.783) | 8.96% / 8.40% | 135/135 | pass |
| base map 30, then p30L | 0.754 / 0.758 | -1.075 (-1.289, -0.883) | 8.96% / 8.42% | 135/135 | pass |
| base map 100, then p30L | 0.782 / 0.768 | -1.265 (-1.467, -1.072) | 8.87% / 8.40% | 134/135 | pass |

The base-fit route keeps the trainer's in-distribution development score (0.958, as p30L) and improves agreement and
the primary, but not error against human ratings, and map 100 misses one G-STEER case. Base fits alone: map 100 at
seeds 0-2 gives agreement 0.792 / 0.789 / 0.791 (control 0.720) with error against human ratings inside the control's
seed spread; map 300 reaches 0.827. The continuation route (p30L + map 300) stays the candidate.

### Experiment 5 amendment 5 result: the TID2013 + MDID mix with p30L and map 300 (seeds 0-2)

| arm (means over seeds 0-2) | held-out error | shared three: error / SROCC | all nine | primary (CI) | agreement 8 px full / served | G-STEER | G-CHROMA-L | NEARID (s0) |
|---|---:|---|---:|---|---|---|---|---|
| p30L | 8.88% | 8.88% / 0.843 | 8.39% | -0.975 (-1.181, -0.799) | 0.738 / 0.754 (s0) | 135, 135, 135 | pass | pass |
| p30L + map 300 | 8.68% | 8.68% / 0.852 | 8.07% | -1.241 (-1.436, -1.057) | 0.789-0.796 / 0.800-0.810 | 135, 135, 135 | pass | pass |
| mix + p30L + map 300 | 7.82% | 8.03% / 0.865 | 7.63% | -1.296 (-1.499, -1.094) | 0.783-0.788 / 0.806-0.816 | 135, 134, 134 | pass | pass |

The mix with the map term has the lowest error against human ratings of any arm at every seed (shared three
8.02 / 8.07 / 8.00%) and the best primary among gate-tested 3-seed arms, and the map term cut the mix's G-STEER misses
from 3-4 per seed to 0-1: seed 1 misses broad-220-b8 (m3f 0.686, bar 0.70), seed 2 broad-164-b64 (m2 0.986, bar 0.99).
By the registered rule it does not pass G-STEER, so its loop run is not a selection step.

**Amendment 6, before running:** the JPEG XL loop (step 4 protocol, i270) for mix + p30L + map 300 at seed 0 (its
G-STEER-clean seed), recorded as information for the owner's decision on the single-case G-STEER misses, against p30L,
p30L + map 300 and the butteraugli loop.

### Experiment 5 amendment 6 result: the mix with p30L and map 300 in the JPEG XL loop (seed 0, i270)

Fresh loop build (zensimx f9e05d82, `e36_loop_mix.sh`); B/M7 placements reproduce, and the arms shared with step 4
(p30L, p30L + map 300, butteraugli, F3, M7, B) give the same BD-rates. 78 images; positive = fewer bytes:

| arm | vs | ssim2 | butteraugli | zensim | DVIFM | CVVDP |
|---|---|---|---|---|---|---|
| mix + p30L + map 300 | p30L | +0.88 [+0.62, +1.15] | +0.98 [+0.65, +1.31] | +0.61 [+0.21, +1.02] | +1.04 [+0.39, +1.74] | +1.02 [+0.60, +1.37] |
| mix + p30L + map 300 | p30L + map 300 | +0.36 [+0.06, +0.70] | +0.10 [-0.23, +0.47] | +0.28 [-0.14, +0.76] | +0.51 [-0.30, +1.30] | +0.16 [-0.22, +0.50] |
| mix + p30L + map 300 | butteraugli loop | -1.52 | -3.91 | -1.01 | -1.72 | -3.31 |

The mix arm improves on p30L for every judge, with every interval above zero, and is the best zensim arm in the loop.
Its only failed check is G-STEER at seeds 1-2 (one case each). Tables: `benchmarks/e36_loop_mix_2026-11-02/`.

### Experiment 5 amendment 7 (2026-11-03, before any run): the mix with a light map term

An out-of-fold check on AIC2026 public_train (members-only numbers, kept in the private training repo) found the map
term's cost there growing with its weight, and the TID2013 + MDID mix without the map term costing nothing; the mix
alone fails G-STEER (3-4 cases per seed), and map 300 brought that to 0-1. **Arms:** mix + p30L + map 30 and
mix + p30L + map 100, seeds 0-2, full recipe, measured with the amendment-2 set (G-STEER, G-CHROMA-L, primary,
error against human ratings, held-out map agreement; NEARID at seed 0), plus the AIC2026 out-of-fold check. A
configuration is a candidate only if G-STEER passes 135/135 at every seed.

## Revised plan (method review, `docs/E36_METHOD_REVIEW_2026-11-03.md`)

The review's experiments are numbered R1-R6 here, so they don't collide with experiments 1-6 above.

### R2 registration (2026-11-03, before any run): human rows by distortion type (method review M3)

Owner, 2026-11-03: "m3, yes, reweight away from synthetic rows if it seems to help". `scripts/byconstruction/e36_exp56/m3_rows.py`
tags each of the 10,590 human fit rows with its distortion type (the E36 bank's `codec`, joined on
`member_set`/`pair_key`) and writes `--human-rows` keep lists:

| tier | types | rows |
|---|---|---:|
| C, compression | KADID 09 JPEG 2000, 10 JPEG; TID2013 10 JPEG, 11 JPEG 2000, 21 lossy compression of noisy images; KonFiG JPEG 2000; all eight CID22-A25 codecs | 2,411 |
| N, encoder-like artifacts | KADID 01 Gaussian blur, 06 colour quantization, 15 denoise, 21 pixelate, 22 quantization; TID2013 07 quantization noise, 08 Gaussian blur, 09 image denoising, 22 colour quantization with dither | 1,604 |
| S, the rest | noise, colour and brightness shifts, contrast, jitter, transmission errors, sharpening, ... | 6,575 |

**Arms:** `m3c` keeps tier C (2,411 rows), `m3cn` tiers C and N (4,015 rows). Both keep 93 of the 100 references:
KonFiG's seven part-B references hold only motion blur and leave the leg. **Control:** experiment 6's walk arm (all
rows). **Recipe:** F3's base-fit step through the experiment copy, walk sampler (trainer bdf7a835), seeds 0-2. The
human group's weight and its uniform draw over references are unchanged, so a kept reference keeps an equal share and
an arm changes only which rows its pairs compare.

**Measures:** error against human ratings (the experiment 5 readout: shared three and all nine), the M1 panel per
collection (`dataset_eval.py paired`, arm minus walk control at each seed, whole-reference bootstrap), and the E36
primary against the walk control. **Rule:** the reweighting "helps" if the seed-mean all-nine error falls by more than
the walk control's seed spread (0.10 points) and the shared-three error does not rise; then it moves into the full
recipe (continuation with p30L) and the gate set. Otherwise it is recorded as not adopted.

### M2 registration (2026-11-03, before any read): dataset trust screen

Owner, 2026-11-03: "on m2, it would be good to know which datasets are trustworthy and seem to reasonably correlate
with ssim2 and other leading models". Exposure registered in `docs/DATA_SPLITS.md` (2026-11-03 entry): NNCD-IQA, MMSPG
JPEG XR, MIDD-CAP, MIDD-GAR, CLIC 2021's validation pairs, mucped23 and Fidelity-LIC, scored with SSIMULACRA2,
butteraugli, CVVDP (pinned judge build) and DVIFM 0.3 (frozen runtime). No zensim model is scored on them. Per set:
each metric's rank agreement (SROCC overall and within reference for MOS sets, two-alternative agreement for pair
sets, with whole-reference intervals), the metrics' agreement with each other, and label-quality signals (rater
counts, standard errors, whether ratings fall as the codec setting gets worse). The EVAL collections' existing
SSIMULACRA2, butteraugli and CVVDP scores join the table without a new read.
