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
| p30L | -0.975 (-1.181, -0.799) | 8.88% / 0.843 | 8.39% | 0.738 / 0.754 (seed 0) | 135, 134, 135 | pass | pass |
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
| p30L | 8.88% | 8.88% / 0.843 | 8.39% | -0.975 (-1.181, -0.799) | 0.738 / 0.754 (s0) | 135, 134, 135 | pass | pass |
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

### R2 amendment 1 (2026-11-03, before any R2 run): decide on the EVAL compression rows

The registered measure (error on all nine EVAL collections) is mostly non-codec distortions: 417 of the 1,858 EVAL
MOS rows are compression (`scripts/byconstruction/e36_exp56/ceval.py`: CID:IQ JPEG and JPEG 2000 70, Colourlab 18,
IQA-ODI 48, IRCCyN-IVC 50, NITS-IQA D7-D9 135, VCL-FER 96). R2 tests compression relevance, so it is judged on
those rows. On existing arms (seed 0 unless noted), row-weighted SROCC on the compression rows: control 0.893-0.895
(seeds 0-2), walk control 0.886 / 0.889 / 0.895, p30L 0.894-0.896, mix + p30L + map 300 0.898, SSIMULACRA2 0.896,
CVVDP 0.876, butteraugli 0.854.

**Amended rule.** An R2 arm helps if, against the walk control at the same seed, (1) its row-weighted SROCC on the
compression rows rises at all three seeds and by at least 0.005 on the seed mean, (2) the E36 primary's interval
(arm minus walk control) does not lie wholly above zero, and (3) its all-nine error rises by no more than 0.30
points on the seed mean. The all-nine readout and the M1 per-collection panel are reported either way.

### R5 (method review M4): linear-probe transfer screen (2026-11-03)

`scripts/byconstruction/e36_exp56/probe_transfer.py` fits a linear head on F3's 492 input features per training set
and scores the nine EVAL MOS collections (mean SROCC over the collections a set does not train on). Three objectives:
ridge on z-scored targets (global), ridge on within-reference deviations (within), and a pairwise-logistic ranker on
pairs inside each reference (rank, the trainer's human-leg shape). The F3 walk control scores 0.851-0.854 on the same
mean. Rank objective, held-out mean SROCC:

| training set | rows | held-out | | training set | rows | held-out |
|---|---:|---:|---|---|---:|---:|
| KADID all | 6,659 | 0.822 | | TID2013 all | 1,680 | 0.832 |
| KADID C | 550 | 0.827 | | TID2013 C | 210 | 0.831 |
| KADID C+N | 1,874 | 0.792 | | TID2013 C+N | 490 | 0.849 |
| KADID S | 4,785 | 0.806 | | TID2013 S | 1,190 | 0.808 |
| human, four sources | 10,590 | 0.823 | | CID22-A25 | 1,579 | 0.803 |
| human C (R2 m3c) | 2,411 | 0.786 | | KonFiG | 672 | 0.765 |
| human C+N (R2 m3cn) | 4,015 | 0.803 | | leg CID:IQ | 330 | 0.849 |
| leg MDID | 800 | 0.832 | | leg CID:OA | 143 | 0.675 |

Within one source the compression rows transfer about as well as all rows and better than the synthetic rows; pooled
across sources, compression-only is worse than all rows. All three objectives agree on the single-source ordering;
the global ridge's pooled sets are distorted by each source's label scale (pooled compression rows 0.524), which the
rank objective removes. R2's retrains are the check on whether the probe predicts a trained network.

### Method review readouts on the measured arms (2026-11-03)

**EVAL compression rows** (`ceval.py`, 417 rows in six collections; row-weighted SROCC / mean logistic-fit error):

| model | seed 0 | seed 1 | seed 2 | error (s0) |
|---|---:|---:|---:|---:|
| F3 control | 0.8949 | 0.8926 | 0.8935 | 7.03% |
| F3 walk control | 0.8861 | 0.8892 | 0.8951 | 7.43% |
| p30L | 0.8956 | 0.8957 | 0.8935 | 6.82% |
| p30L + map 300 | 0.8954 | 0.8942 | 0.8945 | 6.88% |
| mix + p30L | 0.8978 | 0.8987 | 0.8982 | 6.61% |
| mix + p30L + map 300 | 0.8982 | 0.8975 | 0.8976 | 6.60% |
| experiment 5, no human MOS (s0) | 0.8948 | | | 7.03% |
| experiment 5, KADID only (s0) | 0.8892 | | | 7.13% |
| B7 / M7 | 0.8934 / 0.8904 | | | 7.01% / 7.33% |
| SSIMULACRA2 | 0.8963 | | | 6.92% |
| CVVDP | 0.8763 | | | 7.07% |
| butteraugli | 0.8544 | | | 8.61% |

Within each reference, SSIMULACRA2 leads every zensim arm on CID:IQ (0.948 vs 0.942-0.943), IRCCyN-IVC (0.986 vs
0.967-0.974) and Colourlab (0.876 vs 0.837; 18 rows, 2 references), and ties on NITS-IQA and VCL-FER. On the
product's distortions the program's arms sit at SSIMULACRA2's level; the gains reported on all nine collections come
mostly from their non-compression rows.

**M1 paired panel** (`dataset_eval.py paired` per seed, arm minus comparator, 1,000 whole-reference resamples;
`m1_summary.py`). Seed-mean change in SROCC per collection, and how many of the three seeds' intervals lie wholly
above zero:

| collection (rows / refs) | p30L − control | map 300 − p30L | mix + map 300 − p30L | mix − p30L |
|---|---|---|---|---|
| CID:IQ (210 / 7) | +0.0005 (0) | +0.0177 (3) | +0.0279 (3) | +0.0103 (3) |
| CID:OA (91 / 7) | +0.0046 (0) | +0.0103 (0) | +0.0194 (0) | +0.0142 (0) |
| Colourlab (192 / 3) | +0.0126 (2) | −0.0020 (0) | −0.0066 (0) | +0.0248 (0) |
| IQA-ODI (48 / 42) | +0.0085 (0) | +0.0073 (0) | +0.0141 (1) | +0.0094 (0) |
| IRCCyN-IVC (60 / 4) | −0.0053 (0) | −0.0043 (0) | +0.0002 (0) | +0.0055 (0) |
| LIVE-MD (180 / 12) | −0.0041 (0) | +0.0067 (0) | +0.0014 (0) | +0.0041 (0) |
| MDID (480 / 6) | +0.0070 (3) | +0.0071 (3) | +0.0259 (3) | +0.0302 (3) |
| NITS-IQA (405 / 9) | +0.0120 (3) | +0.0161 (3) | +0.0415 (3) | +0.0255 (3) |
| VCL-FER (192 / 8) | +0.0014 (0) | +0.0050 (3) | +0.0112 (3) | +0.0102 (3) |
| row-weighted mean | +0.0058 | +0.0089 | +0.0208 | +0.0197 |

No seed of any comparison has an interval wholly below zero on any collection. The mix arms train on MDID's TRAIN
split and not on CID:IQ's, and still gain on CID:IQ's EVAL rows over p30L, which trains on both. Collections with
fewer than about 30 references give descriptive intervals only.

**R5 on the compression rows** (rank objective): all four human sources 0.872, compression rows only (m3c) 0.828,
compression + encoder-like rows (m3cn) 0.845; single sources: KADID 0.854 vs its compression rows 0.847, TID2013
0.864 vs 0.850 (C) and 0.896 (C+N). The probe predicts that R2's arms will not beat the walk control.

### Experiment 5 amendment 7 result: the mix with a light map term (seeds 0-2)

| arm | primary (vs control) | shared three: error / SROCC | all nine | compression rows (s0/s1/s2) | map agreement 8 px full | G-STEER | G-CHROMA-L | NEARID (s0) |
|---|---|---|---|---|---|---|---|---|
| p30L | -0.98 | 8.88% / 0.843 | 8.39% | 0.896 / 0.896 / 0.894 | 0.738 | 135, 134, 135 | pass | pass |
| mix + p30L + map 30 | -1.08 | 8.29% / 0.858 | 7.85% | 0.898 / 0.899 / 0.898 | 0.729-0.743 | 134, 133, 133 | pass | pass |
| mix + p30L + map 100 | -1.18 | 8.32% / 0.859 | 7.83% | 0.899 / 0.899 / 0.898 | 0.753-0.756 | 134, 130, 132 | pass | pass |
| mix + p30L + map 300 | -1.30 | 8.03% / 0.865 | 7.63% | 0.898 / 0.898 / 0.898 | 0.783-0.788 | 135, 134, 134 | pass | pass |

Neither light map term is a candidate under the registered rule: every seed misses broad-220-b8 on map rank (m3f
0.62-0.67, bar 0.70), and seeds 1-2 add one to four m2 near-misses (0.956-0.98, bar 0.99) on broad-38, broad-76 and
broad-143 cases. The private out-of-fold check (numbers in the training repository) found no AIC2026 cost for either,
like the mix without a map term. M1 panel against p30L: map 30 +0.0197 and map 100 +0.0205 row-weighted SROCC, every
collection's seed mean at or above zero, positive at all three seeds on CID:IQ, MDID, NITS-IQA and VCL-FER.

**Reading.** Map 300 remains the only mix configuration within one case of G-STEER at every seed; lighter map terms
lose map rank on broad-220-b8 without buying anything on the other measures, and on the EVAL compression rows every
mix arm sits at 0.898, level with SSIMULACRA2 (0.896).

### M2 result: dataset trust screen (2026-11-03)

`scripts/byconstruction/dataset_trust.py` on i270 (`trust_decode` decodes BMP/PNM with zenbitmaps and, for DVIFM only,
WebP with zenwebp). MOS sets: SROCC with a whole-reference interval (within-reference SROCC after "w"). Pair sets:
two-alternative agreement (soft labels, whole-reference interval). Observer agreement: each vote of a pair with at
least two votes predicted by the majority of the pair's other votes.

| set | rows / refs | label quality | SSIMULACRA2 | butteraugli 3-norm | CVVDP | DVIFM 0.3 | exposure |
|---|---|---|---|---|---|---|---|
| NNCD-IQA (learned codecs, JPEG 2000) | 320 / 16 | 0.4% of ladder steps against the ladder | 0.930 [0.908, 0.953] w 0.961 | 0.928 w 0.950 | 0.922 w 0.957 | 0.925 w 0.956 | TID2013 / Kodak sources |
| MMSPG JPEG XR (JPEG XR, JPEG, JPEG 2000) | 186 / 6 | reliability 0.978 (13 raters); 10.7% ladder steps against | 0.928 [0.891, 0.953] w 0.938 | 0.892 | 0.911 | 0.928 | none known; 6 sources |
| MIDD-CAP (near-lossless JPEG, identification) | 192 / 24 | 6.0% against | 0.877 [0.839, 0.920] w 0.977 | 0.892 w 0.977 | 0.848 w 0.977 | 0.910 w 0.977 | TID2013 sources |
| MIDD-GAR (grayscale, JPEG/WebP, identification) | 200 / 10 | 5.0% against | 0.782 [0.712, 0.852] w 0.852 | 0.749 | 0.756 | 0.780 | TID2013 sources |
| CLIC 2021 validation (learned codecs, 2AFC) | 4,812 / 315 | observer agreement 0.769 (262 pairs); 1 vote median | 0.626 [0.612, 0.641] | 0.606 | 0.639 | 0.629 | source-independent |
| mucped23 (jpegli, MozJPEG, libjpeg-turbo q55-95) | 9,314 / 49 | observer agreement 0.637 (3,091 pairs); 1 vote median | 0.739 [0.721, 0.756] | 0.740 | 0.725 | 0.740 | 17 of its 49 sources are CID22-A25 training sources |
| Fidelity-LIC (GAN vs non-GAN learned codec) | 132 / 42 | observer agreement 0.675; 20 votes, 34% ties | 0.361 [0.328, 0.396] | 0.412 | 0.322 | 0.598 | some sources protected (D4) |

- **mucped23's library table is inverted.** The authors' `compute_elo.ipynb` counts a win for a method when the
  other method is named in `answerValue`, so `answerValue` names the distortion judged farther from the original;
  `answerValue == A` agrees with the authors' published Elo order in 3,609 of 13,488 answers (26.8%). The library
  (zenpapers `scripts/iqa_library/organize.py`, mucped23 adapter) counts it as the closer one. The screen corrects the
  orientation (`INVERTED_ANSWERS`); uncorrected, every metric scored 0.26. **Fixed in the library on 2026-11-03:** zenpapers 5bbd3ed0
  corrects the adapter (with a check against the authors' Elo: 9,879 of 13,488 answers agree), the rebuilt table
  swaps n_a and n_b in all 9,314 rows with every image unchanged, and the screen's correction is removed; the old
  build is kept as `datasets/mucped23.bak-20261103T134815Z`.
- **Fidelity-LIC measures something fidelity metrics miss.** Observers chose the GAN crop as closer to the reference
  at low rates while the non-GAN crop has the higher PSNR in 128 of 132 triplets; only DVIFM agrees above chance.
  Ten references embed a non-sRGB ICC profile (Adobe RGB (1998), "c2") while their encodes are tagged sRGB; without
  the 31 triplets that touch them the agreements barely move (SSIMULACRA2 0.356, DVIFM 0.588).
- **CLIC 2021's validation pairs are hard for every metric:** 0.61-0.64 against an observer agreement of 0.77.
- **Trust.** NNCD, MMSPG JPEG XR and MIDD-CAP have consistent labels and leading metrics agree with them (0.85-0.93);
  all but MMSPG reuse TID2013's sources. CLIC 2021's validation crops are the only large, source-independent
  compression set; mucped23 is product-relevant (JPEG encoders) but noisy and shares 17 sources with our training.
- **Recommended confirmation set:** CLIC 2021's validation pairs (statistic: two-alternative agreement, whole-reference
  interval, 315 references), with NNCD-IQA and MIDD-CAP as source-exposed diagnostics. CIQA (JPEG at paired
  qualities, source-independent) would add the product's codec but needs its TensorFlow-adjusted reference pixels,
  which were not stored; regenerating them needs the owner's sign-off.

### R3 registration (2026-11-03, before any run): the map term on low-quality views only

**Hypothesis** (method review, experiment 3 of the revised plan): butteraugli's block map misleads near the
visibility threshold, where AIC2026 public_train lives; the private out-of-fold check found the map term's AIC2026 cost
growing with its weight. **Arms:** p30L + map 300 and mix + p30L + map 300 with the map pool restricted to views whose
encoder quality setting is at most 60: 1,310 of the pool's 1,967 views (`scripts/byconstruction/map_pool_subset.py`,
`map-pool-16-q0-60`; the dropped band is q62-100, 657 views). Each is its ungated twin with only the pool path changed
(commands differ in the pool path and destination). Full data at seeds 0-2, and the AIC2026 public_train folds at
seed 0. **Measures:** the amendment-2 set (G-STEER, G-CHROMA-L, primary, error against human ratings, EVAL compression
rows, held-out map agreement; NEARID at seed 0), the M1 panel against each ungated twin, and the private out-of-fold
check. **Rule:** a gated arm is a candidate only if G-STEER passes 135/135 at every seed; it removes the near-threshold
cost if its out-of-fold difference against p30L has an interval containing zero. A gated arm that does both goes to
the JPEG XL loop.

### R2 result (2026-11-03): human rows by distortion type

Six walk base fits (`m3c` and `m3cn` at seeds 0-2) against experiment 6's walk arm at the same seeds.

| measure | walk (s0 / s1 / s2) | m3c (compression rows only) | m3cn (compression and encoder-like rows) |
|---|---|---|---|
| compression-row SROCC | 0.8861 / 0.8892 / 0.8951 | 0.8872 / 0.8895 / 0.8939 (mean +0.0001) | 0.8851 / 0.8887 / 0.8903 (mean -0.0021) |
| all-nine error | 8.61 / 8.51 / 8.60% | 8.86 / 8.70 / 8.49% (mean +0.11) | 8.50 / 8.50 / 8.63% (mean -0.03) |
| shared-three error | 9.17 / 9.09 / 8.96% | 9.00 / 8.92 / 8.69% (mean -0.20) | 8.91 / 8.75 / 9.01% (mean -0.18) |
| shared-three SROCC | 0.830 / 0.834 / 0.835 | 0.845 / 0.847 / 0.850 | 0.839 / 0.848 / 0.838 |
| primary minus walk [95%] | | -0.124 [-0.228, -0.014] | +0.043 [-0.050, +0.126] |
| M1 panel, row-weighted ΔSROCC | | +0.0049 | +0.0044 |

**Rule (amendment 1): neither arm helps.** Compression-row SROCC does not rise at all three seeds, and its seed mean
moves +0.0001 and -0.0021 against a bar of +0.005, as R5's probe screen predicted. The original rule fails too
(all-nine error +0.11 and -0.03 against a required fall of 0.10). Not adopted. Reported either way: keeping only
compression rows lowers the shared-three error by about 0.2 points at every seed, and the M1 panel moves NITS-IQA up
(m3c +0.036, interval above zero at all three seeds) and CID:IQ (+0.015, two seeds) while Colourlab (-0.016) and MDID
(-0.017) move down. The shared-three error is not the registered statistic, and experiment 5's TID2013 + MDID mix
lowers it by 0.5-0.85 points, so no follow-up arm is registered. Readouts: `~/tmp/e36test/postR2/` on dev.

### R3 result (2026-11-03): the map term on views at quality <= 60

| arm (seeds 0-2) | G-STEER | primary (CI) | shared-three / all-nine error | compression rows | agreement 8 px full | M1 vs ungated twin |
|---|---|---|---|---|---|---|
| p30L + map 300 | 135, 135, 135 | -1.241 (-1.436, -1.057) | 8.64-8.71 / 8.03-8.10% | 0.894-0.895 | 0.789-0.796 | |
| p30L + map 300, q <= 60 | **131, 133, 133** | -1.234 (-1.423, -1.061) | 8.65-8.70 / 8.13-8.15% | 0.892-0.894 | 0.781-0.786 | -0.0007 |
| mix + p30L + map 300 | 135, 134, 134 | -1.296 (-1.499, -1.094) | 8.00-8.07 / 7.58-7.67% | 0.8975-0.8982 | 0.783-0.788 | |
| mix + p30L + map 300, q <= 60 | **131, 134, 134** | -1.305 (-1.508, -1.105) | 8.06-8.15 / 7.67-7.73% | 0.8974-0.8978 | 0.784-0.786 | -0.0016 |

G-CHROMA-L passes for all six runs; NEARID passes at seed 0 (138 and 136 of 144 ladders monotone). G-STEER misses:
broad-220-b8 at every seed of both gated arms (m3f 0.671-0.695, bar 0.70), I21_10_05 at seeds 0-1 of the p30L arm
(m3f 0.692-0.694), broad-206 at b8/b16/b64 at seed 0 of both (m2 0.951-0.981, bar 0.99) and broad-73-b64 at seed 2
of the p30L arm (m2 0.988). The M1 panel against the ungated twins is at or below zero in every collection but
NITS-IQA; the mix arm's CID:IQ and MDID intervals lie below zero at all three seeds.

**Rule: neither gated arm is a candidate** (G-STEER fails at every seed). The private out-of-fold check (numbers in
the training repository) finds the gate leaves the map term's AIC2026 cost in place: both gated arms' intervals
against p30L still exclude zero. **Reading:** the cost does not come from the high-quality views, and those views
carry the map rank that G-STEER's broad-220-b8 and I21_10_05 cases check. The hypothesis is not supported; the ungated
map 300 arms stand as before. Readouts: `~/tmp/e36test/postR3/` on dev.

### Correction (2026-11-03): p30L's G-STEER is 135, 134, 135

p30L's seed 1 misses broad-206-b64 (m2 0.979, bar 0.99). Amendment 2's table had this right; the experiment 3,
experiment 5 and amendment 7 tables listed p30L as 135, 135, 135 and are corrected above. Consequence: amendments 5
and 7 and R3 made "135/135 at every seed" the candidate rule, which p30L itself does not meet. Under amendment 2's
guard (G-STEER as F3 passed it: 135 at seed 0 and a three-seed mean not below the control's 134.33), p30L (mean
134.67) and mix + p30L + map 300 (135, 134, 134; mean 134.33) both pass. Which rule governs the choice between them is
the owner's decision; the registered rules are not changed after the fact.

### Confirmation read 1 registration (2026-11-03, before the read): CLIC 2021 validation

Owner, 2026-11-03: "use CLIC 2021 validation". **Set:** `clic-2021-perceptual`, the native validation pairs only
(4,812 questions over 315 reference crops); exposure entry in `docs/DATA_SPLITS.md` (2026-11-03). **Disclosures**
(D4): CLIC 2021 shares CLIC's test protocol with CLIC 2024, R7's proposed training leg; the M2 screen already scored
these pairs with third-party metrics (SSIMULACRA2 0.626, butteraugli 0.606, CVVDP 0.639, DVIFM 0.629; observer
agreement 0.769). **Features:** the EVAL collections' extractor and bank format (revision 7, basic+peaks+v2@w1825), keys
built from paths only. **Statistic:** two-alternative agreement in `dataset_eval.py`'s soft-label form per model, and
each challenger's paired difference against p30L at the same seed through a whole-reference cluster bootstrap (315
references, 10,000 draws, seed 20261103), summarized as the seed-mean difference with the interval of the seed-mean
difference. **Models:** F3 control, p30L, p30L + map 300 and mix + p30L + map 300, seeds 0-2 each. **Reading rule:** a
challenger is "supported" if its interval lies wholly above zero, "contradicted" if wholly below, otherwise
"neutral". A contradicted challenger is not recommended; otherwise the existing measures and the owner decide.
Nothing is fitted, tuned or selected on this read beyond that rule.

### R7 registration (2026-11-03, before the leg builder reads any label): CLIC 2024 as a training leg

Owner, 2026-11-03: "go ahead with R7". **Data:** `clic-2024-perceptual-image`, all 24,805 questions (native split test).
Per D4 the 24,786 reference crops are exact subimages of 32 originals that are SafeSyn training sources, so the set
can only train. **Leg:** each question becomes a two-row comparison in the BAPPS leg's form: (reference crop, a) and
(reference crop, b) with targets p_a = (n_a + n_tie/2)/n and 1 - p_a, mode `withinref,rank`, nominal weight 0.5 (train
weight 1.0 after the two-sided acceptance correction); questions with p_a = 1/2 carry no order and are dropped. Family:
the question's original, from D4's crop-to-original table (`followup/clic-2024-perceptual-image-source-disposition.tsv`).
**Features:** the same extractor and bank format; every crop is 768x768. **Arms** (the leg added to the base fit,
everything else unchanged): (a) p30L's recipe, (b) mix + p30L + map 300's recipe; seeds 0-2, plus the AIC2026 folds
at seed 0 (private). **Twins:** p30L and mix + p30L + map 300 at the same seeds. **Measures:** the amendment-2 set
(primary, G-STEER, G-CHROMA-L, NEARID at seed 0, error against human ratings, EVAL compression rows, map agreement),
the M1 panel against each twin, the private out-of-fold check, and confirmation read 2 on CLIC 2021 validation (read
1's statistic and reading rule, each arm against its twin). **Rule:** an arm helps if its compression-row SROCC seed
mean falls by no more than 0.003 against its twin, its all-nine error rises by no more than 0.30 points, G-STEER
passes amendment 2's guard (the stricter every-seed rule reported alongside), G-CHROMA-L and NEARID pass, and read 2
supports it. **Disclosure:** for R7 the CLIC 2021 read is the decision statistic, so an adopted R7 arm has no
independent confirmation on CLIC 2021; its read there is a selection read.

### Confirmation read 1 result (2026-11-03): CLIC 2021 validation

4,812 questions over 315 references; features from the EVAL extractor (2,415 distinct pairs,
`/mnt/v/zensim/dataset-features-rev7-clic-20261103/`, `scripts/byconstruction/e36_exp56/clic.py`). Two-alternative
agreement (soft labels), per seed; for reference, the M2 screen's metrics on the same questions: SSIMULACRA2 0.626,
butteraugli 0.606, CVVDP 0.639, DVIFM 0.629; observer agreement 0.769.

| model | seed 0 | seed 1 | seed 2 | seed-mean difference vs p30L [95%] | reading |
|---|---:|---:|---:|---|---|
| F3 control | 0.6445 | 0.6443 | 0.6398 | +0.0029 [+0.0002, +0.0056] | above p30L |
| p30L | 0.6411 | 0.6417 | 0.6373 | | incumbent |
| p30L + map 300 | 0.6348 | 0.6357 | 0.6340 | -0.0052 [-0.0085, -0.0019] | contradicted |
| mix + p30L + map 300 | 0.6353 | 0.6357 | 0.6351 | -0.0047 [-0.0084, -0.0010] | contradicted |

**Reading (registered rule):** both map-300 challengers are contradicted, so neither is recommended. p30L stays the
incumbent. Reported beside the rule: F3's control scores above p30L at every seed (+0.0034, +0.0026, +0.0025), so
the hinge and level terms cost about 0.3 points of agreement here, and the map term about another 0.5. AIC2026
public_train (private out-of-fold check) showed the map term's cost too, so two independent pairwise human sets now
disagree with the map term's direction while E36's primary, EVAL's error and the JPEG XL loop favour it. Every
zensim model here is above SSIMULACRA2 (0.626); F3's control is also above CVVDP (0.639). Readout:
`~/tmp/e36test/clic21-confirm-1/CONFIRM.json` on dev.

### R7 amendment 1 (2026-11-03, after confirmation read 1, before any R7 run)

Read 1 contradicts mix + p30L + map 300, so arm (b) changes to F3's control recipe with the leg added (twin: F3's
control at the same seeds); arm (a) is unchanged. Both arms continue from the same three base fits (F3's base-fit
step with the CLIC 2024 leg), so the mix base fits are dropped. The AIC2026 folds at seed 0 run for both arms and,
since none exist yet, for F3's control recipe too (seven twin folds). Rule, measures and confirmation read 2 are as
registered, each arm against its twin.

### R7 result (2026-11-03): CLIC 2024 as a training leg

**As run:** the R7 root (the original root with the leg appended, `clic.py leg`) replaced the original root in every
R7 command, base fit and continuation alike (`exp9/make_cmds.py`), so the continuation trains on the leg too; the
registration's "the leg added to the base fit" is narrower than what ran. Each step loaded 49,608 rows (24,804
questions; one with p_a = 1/2 dropped). Read 2's `LABEL_READS.json` names "confirmation read 1" as its registration,
a label fixed in the script; read 2 was registered with R7 above, and `clic.py confirm` now takes `--registration`.

Each arm against its twin, seeds 0-2 (means unless stated):

| R7 rule | p30L + CLIC 2024 vs p30L | F3 control + CLIC 2024 vs F3 control |
|---|---|---|
| EVAL compression-row SROCC (fall at most 0.003) | -0.0045 (0.8904 vs 0.8949; seeds -0.0145, -0.0004, +0.0013): fails | -0.0009 (0.8927 vs 0.8937): passes |
| all-nine error (rise at most 0.30 points) | +0.03 (8.42% vs 8.39%): passes | +0.05 (8.60% vs 8.55%): passes |
| G-STEER, amendment 2's guard (135 at seed 0, mean at least 134.33) | 134, 134, 135: fails (seed 0 misses broad-34-b64, m2 0.944; seed 1 broad-206-b64, m2 0.965) | 135, 134, 135 (mean 134.67): passes; the every-seed rule fails (seed 1 misses dog-256-50, m2 0.909, m3f 0.777) |
| G-CHROMA-L (dev) | passes at every seed | passes at every seed |
| NEARID (seed 0) | passes (N3 139 of 144 ladders) | passes (N3 139 of 144) |
| confirmation read 2, CLIC 2021 validation | -0.0004 [-0.0016, +0.0008], neutral: fails | +0.0026 [+0.0012, +0.0040], supported: passes |
| **reading** | **does not help** | **helps** |

Beside the rule:

| measure | p30L | p30L + CLIC 2024 | F3 control | F3 control + CLIC 2024 |
|---|---|---|---|---|
| E36 primary vs F3 control, points [95%] | -0.975 [-1.181, -0.799] | -1.047 [-1.251, -0.869] | 0 (3.36%) | +0.053 [+0.028, +0.076] |
| shared-three error / SROCC | 8.88% / 0.843 | 8.93% / 0.840 | 8.80% / 0.835 | 8.80% / 0.834 |
| CLIC 2021 agreement, seeds 0 / 1 / 2 | 0.6411 / 0.6417 / 0.6373 | 0.6346 / 0.6425 / 0.6417 | 0.6445 / 0.6443 / 0.6398 | 0.6431 / 0.6470 / 0.6462 |
| map agreement, 8 px full / served | 0.738 / 0.754 (seed 0) | 0.762 / 0.762, 0.740 / 0.752, 0.741 / 0.755 | 0.706 / 0.702 (seed 0) | 0.718 / 0.703, 0.707 / 0.702, 0.707 / 0.701 |
| M1 row-weighted SROCC change vs twin (descriptive) | | -0.0026 | | -0.0017 |
| G-STEER | 135, 134, 135 | 134, 134, 135 | 135, 135, 133 | 135, 134, 135 |

**Reading (registered rule):** the leg helps F3's control recipe and does not help p30L's. For F3's control it raises
CLIC 2021 agreement by 0.0026 (seeds -0.0014, +0.0027, +0.0064) with no measured change in error against human
ratings or on the compression rows, and adds 0.05 points of E36 misordering (interval above zero). With p30L the
leg's seed 0 is worse on every human measure (compression rows 0.8811, CLIC 2021 0.6346, all-nine error 8.53%) while
having the lowest misordering of any seed (2.11%); seeds 1 and 2 sit near the twin. The private AIC2026 out-of-fold
check (numbers in the training repository) finds no change from the leg for either arm and, now that F3's control
folds exist, ranks F3's control above p30L, as read 1 did. Arm (b)'s CLIC 2021 read is a selection read (disclosed
at registration). Which candidate ships stays the owner's decision. Readouts: `~/tmp/e36test/postR7/` and
`~/tmp/e36test/clic21-confirm-2/` on dev; bakes under `/mnt/v/output/zensim/e36-subset-test-2026-11-01/bakes/r7-*`
(tower mirror hash-checked, 176 files).

### R8 registration (2026-11-03, before any run): the hinge's weight on the pairwise human sets

After R7, two held-out pairwise human sets rank F3's control above p30L: CLIC 2021 (read 1, +0.0029 [+0.0002,
+0.0056]) and AIC2026 out of fold (private record, same direction), while E36's primary favours p30L by 0.98 points
and the JPEG XL loop ties them. **Question:** does a lighter hinge keep part of the E36 gain without the pairwise
cost? **Arms:** pool-10 and pool-1 (the subset test's arms: pool v1, lambda 10 and 1, no level term; primary -0.64
and -0.19; both met the registered rule) at seed 0, through the private AIC2026 out-of-fold protocol: D6's seven
source folds, each arm's continuation command with only the fold table and destination changed (14 runs).
**Comparison:** each arm against F3's control folds and against p30L's, with the private record's whole-source
bootstrap. CLIC 2021 is not read. **Reading:** an arm whose interval against F3's control contains zero is reported
to the owner as keeping F3's pairwise level, beside its E36 numbers; nothing is adopted on this read alone.

### R8 result (2026-11-03)

Neither lighter hinge keeps F3's pairwise level on AIC2026 out of fold: both arms' intervals against F3's control
lie below zero (numbers in the training repository). The cost grows with lambda, pool-1 < pool-10 < p30L, at a nearly
constant rate per point of E36 misordering removed. **Reading (registered):** no arm is reported as keeping F3's
pairwise level. At every weight tried the hinge trades agreement with these human labels for agreement with the
two reference metrics it is built from. The first attempt (one wave of all 14 runs) was stopped by run-heavy at
3.8 GiB available; the results are from the rerun in two waves of 7.

### Dial comparison (2026-11-03, after R8; descriptive, nothing selected or adopted)

Owner, 2026-11-03: "what is best as a dial model? smoothest and most consistently aligned across codecs with human
perception?" and "I'm unsure cvvdp is a decent metric, it has high volatility across images and codecs compared to
butteraugli." No new label read: the smoothness readout is label-free and the codec readout reuses the private
out-of-fold check's labels.

**Where p30L's E36 gain sits** (`e36_exp56/inversion_by_gap.py` on `inversions-R7`, seed means, selection split), by
the gap between a pair's two quality settings:

| gap | pairs | F3 control | p30L | share of p30L's reduction |
|---|---:|---:|---:|---:|
| 1-2 | 125,702 | 9.80% | 7.42% | 47.5% |
| 3-4 | 131,898 | 4.57% | 3.07% | 31.4% |
| 5-8 | 135,853 | 1.74% | 1.03% | 15.3% |
| 9-16 | 130,591 | 0.62% | 0.39% | 4.6% |
| 17-32 | 122,633 | 0.17% | 0.10% | 1.3% |

The fit split is the same within a point or two (51.2% / 31.1% / 13.5% / 3.7% / 0.5%).

**Smoothness along held-out ladders** (`e36_exp56/ladder_smoothness.py`): 2,833 ladders (one reference view x codec x
mode, baseline panel, the 40 selection sources, views below 1 MP), 144,956 steps after removing steps whose two encodes
decode to identical pixels; every metric on the same steps, oriented higher = better. A reversal is a pair of settings
where the higher one scores strictly worse; the wrong-way share is a ladder's downward movement over its total movement
(0 = monotone; scale-free); seed means for the zensim arms.

| model | reversals, steps 1 / 4 / 8 / 16 apart | wrong-way share | ladders with no reversal |
|---|---|---:|---:|
| butteraugli 3-norm | 15.90 / 3.51 / 1.07 / 0.69% | 10.15% | 6.8% |
| CVVDP | 13.13 / 3.00 / 1.19 / 0.83% | 7.27% | 15.7% |
| E (shipped) | 18.43 / 5.29 / 1.99 / 0.95% | 11.93% | 7.3% |
| ECodec (shipped) | 22.08 / 7.50 / 2.86 / 1.10% | 15.75% | 5.4% |
| F3 control | 15.22 / 3.12 / 0.85 / 0.65% | 9.47% | 9.4% |
| F3 control + CLIC 2024 | 15.34 / 3.14 / 0.86 / 0.66% | 9.51% | 9.3% |
| pool-1 | 14.79 / 2.95 / 0.80 / 0.64% | 9.09% | 10.0% |
| pool-10 | 13.72 / 2.59 / 0.73 / 0.66% | 8.32% | 11.4% |
| p30L | 12.75 / 2.29 / 0.70 / 0.68% | 7.66% | 12.8% |
| mix + p30L | 12.74 / 2.41 / 0.72 / 0.69% | 8.05% | 12.3% |
| p30L + map 300 | 11.77 / 2.12 / 0.66 / 0.67% | 6.99% | 14.4% |
| mix + p30L + map 300 | 11.60 / 2.12 / 0.67 / 0.68% | 7.23% | 14.5% |

- Neighbouring settings often differ by less than any metric resolves: every model reverses 11.6-22.1% of adjacent
  steps, and 0.7-2.9% of steps 8 apart.
- JPEG XL ladders reverse in about 4% of pairs 16 steps apart under every metric (butteraugli 4.07%, CVVDP 4.52%, the
  zensim models 3.92-4.26%), against at most 1.02% for any other codec under any metric. Both reference metrics agree, so this is the
  encoder's setting order going backwards over wide spans, not metric noise.
- CVVDP against butteraugli along a ladder: fewer reversals between close settings (4 apart: 3.00% against 3.51%;
  wrong-way share 7.27% against 10.15%), slightly more between distant ones (16 apart: 0.83% against 0.69%), and
  0.16% exact ties (it saturates at 10 JOD near lossless).
- The hinge and map arms are the smoothest zensim models; the shipped E and ECodec are the roughest.

**Consistency across codecs with people** (AIC2026 public_train, 14 codecs, out of fold; private, numbers in the
training repository, `evaluations/scripts/aic2026_codec_uniformity.py`): the per-codec rank-bias spread (byc_cells.py's
codec-uniformity statistic) is lowest, and level within whole-source intervals, for F3's control, mix + p30L, pool-1
and SSIMULACRA2. p30L, pool-10 and p30L + map 300 are less uniform than F3's control, with intervals that exclude zero;
butteraugli's 3-norm is higher, with an interval against F3 that includes zero. CVVDP is the least uniform of every
metric scored, PSNR included, and nearly all of that comes from one codec, PNG (palette), where people see far more
damage than CVVDP predicts. Without PNG's rows CVVDP is level with F3's control and SSIMULACRA2, and butteraugli's
3-norm is the least uniform of the three reference metrics (SSIMULACRA2, butteraugli 3-norm, CVVDP). CVVDP orders
images across sources (global SROCC) worst of those three, with or without PNG.

**Reading.** No candidate leads on both properties. The hinge buys smoothness along ladders, mostly between settings
1-4 apart, and costs consistency across codecs with people, as it cost agreement on CLIC 2021 and AIC2026 (read 1,
R8). F3's control is among the most codec-uniform, leads both pairwise human sets and sits in the middle on
smoothness. mix + p30L comes within 0.12 points of p30L's smoothness at 4 steps apart and ties F3's control on codec
uniformity, but it fails G-STEER at every seed (broad-220-b8 map rank) and has not been read on CLIC 2021. On the
owner's CVVDP question: the AIC2026 data support it for ordering across images and for palette damage, while along
a single ladder CVVDP is about as smooth as butteraugli. Readouts: `~/tmp/e36test/dial/` on dev.

### R10 registration (2026-11-03, before any run): the TID2013 + MDID mix on F3's recipe, without the hinge

**Question.** The dial comparison and read 1 favour F3's control over p30L on every independent human measure of
agreement across images and codecs, and the TID2013 + MDID mix gave the program's lowest error against human ratings,
but only ever with p30L's terms (experiment 5 stage 3; its AIC2026 out-of-fold check was neutral). Does the mix keep its
error gain on F3's recipe, without the hinge's pairwise cost? **Arm** `mixF3`: each command is experiment 5 stage 3's
mix + p30L at the same seed (`exp5c` `tid2013_mdid-p30L-s{seed}`: continuation from the mix's base fit, mix flags,
F3's continuation otherwise) with the trainer wrapper's hinge and level arguments removed, and only the destination and
scratch directory changed (`e36_exp56/r10_make_cmds.py`). The trainer is the map build, whose run without the map term
reproduced p30L's seed 0 (60 of 60 checkpoints), and the hinge and level terms draw on their own RNG streams, so the
primary sampler's path is F3's. Seeds 0-2, plus the AIC2026 public_train folds at seed 0 (the fold twins of `mix-k*`,
private). **Measures:** the amendment-2 set (primary, G-STEER, G-CHROMA-L, NEARID at seed 0, error against human
ratings on the shared three and all nine, EVAL compression rows, map agreement), the M1 panel against F3's control,
the dial comparison's ladder smoothness, the private out-of-fold check against F3's control and against mix + p30L,
and the codec-uniformity readout. CLIC 2021 is not read. **Reading:** `mixF3` is reported as a candidate to replace
F3's control if its out-of-fold interval against F3's control does not lie wholly below zero, its compression-row SROCC
seed mean is no more than 0.003 below F3's, G-STEER passes amendment 2's guard (135 at seed 0, mean at least 134.33),
G-CHROMA-L and NEARID pass, and its shared-three error seed mean is below F3's control's (8.80%). Nothing is adopted on
this read; the owner decides.

### R10 result (2026-11-03): the TID2013 + MDID mix on F3's recipe, without the hinge

Ten runs on dev in two waves (800 s for the first seven), then `postR10.sh`; the G-CHROMA-L level check (MAE against
the teacher, against F3's control seed 0) ran after it, as `chroma_ladder_gate.py level`. Seeds 0 / 1 / 2:

| measure | F3 control | mix on F3's recipe (`mixF3`) | R10 rule |
|---|---|---|---|
| error vs human, shared three | 8.79 / 8.74 / 8.89% | 8.27 / 8.39 / 8.40% | below F3's mean (8.81%): **passes** |
| SROCC, shared three | 0.835 / 0.835 / 0.836 | 0.850 / 0.848 / 0.849 | |
| error vs human, all nine | 8.54 / 8.58 / 8.52% | 7.94 / 8.09 / 8.05% | |
| EVAL compression rows, SROCC | 0.8949 / 0.8926 / 0.8935 | 0.8954 / 0.8941 / 0.8940 | mean at most 0.003 lower: **passes** (+0.0008) |
| AIC2026 out of fold vs F3's control | | interval contains 0 (private) | not wholly below 0: **passes** |
| G-STEER | 135, 135, 133 | 133, 134, 134 | 135 at seed 0, mean >= 134.33: **fails** |
| G-CHROMA-L | | rank gate passes; worst MAE ratio 1.008 (ycc420, seed 0) | **passes** |
| NEARID (seed 0) | | N1-N3 pass, 132 of 144 ladders monotone | **passes** |
| E36 primary vs F3's control | 3.36% | 3.40%: +0.04 [-0.06, +0.14] points | |
| ladder smoothness, reversals 4 steps apart / wrong-way share | 3.12% / 9.47% | 3.23% / 9.88% | |
| map agreement, 8 px full | 0.706 (seed 0) | 0.701 / 0.690 / 0.698 | |
| M1 panel vs F3's control, row-weighted SROCC | | +0.0215; above 0 at all three seeds on MDID, NITS-IQA and VCL-FER, below 0 nowhere | |

G-STEER misses broad-220-b8 at every seed (m3f 0.694 / 0.685 / 0.686, bar 0.70), the case the mix also missed with
p30L's terms, and broad-13-b64 at seed 0 (m2 0.959, bar 0.99). On AIC2026's 14 codecs (private numbers in the
training repository) `mixF3` is more uniform than F3's control, with an interval that excludes zero, and the most
uniform of every model and reference metric scored, SSIMULACRA2 included; most of the gain is on palette PNG, and
without PNG it is level with F3's control.

**Reading (registered rule):** `mixF3` meets every condition but G-STEER, so it is not reported as a candidate to
replace F3's control. Beside the rule: the mix's error gain does not depend on the hinge (shared-three error 0.45
points below F3's control's mean and lower at every seed; mix + p30L sits 0.53 below p30L), and without the hinge the mix costs no pairwise
agreement on AIC2026 and improves consistency across codecs, while E36 misordering and ladder smoothness stay at
F3's level. Its only failure is the steering map's rank on one G-STEER case at every seed, which the map term repaired
for mix + p30L at a CLIC 2021 cost (read 1). Readouts: `~/tmp/e36test/postR10/` on dev; bakes
`/mnt/v/output/zensim/e36-subset-test-2026-11-01/bakes/r10-mixF3-s*` (tower mirror hash-checked, 182 files).

### R10 amendment 1 (2026-11-03, before any loop run): the mix on F3's recipe in the JPEG XL loop

`mixF3`'s only failed condition is G-STEER (broad-220-b8's map rank at every seed), and the owner's rule is that
steering is decided in the JPEG XL loop. Run, recorded as information for the owner's decision: experiment 5
amendment 6's protocol (`scripts/byconstruction/e36_loop_r10.sh`, built from `e36_loop_mix.sh` with the arm list
changed; i270, 78 TRAIN images, effort 7, 4 iterations, 9 distances, judge build `zenmetrics-judge-7a0f8394`,
`e36_exp56/r10_loop_i270_run.sh`). Arms, all encoded fresh in one directory: none, butteraugli, B, M7, F3, p30L and
`mixF3` at seed 0, the zensim arms with loop_calibrate tail placements. **Controls:** the new build's B and M7
placement rows equal the 2026-10-25r7win calibration's, and the shared arms (none, butteraugli, B, M7, F3, p30L)
reproduce amendment 6's i270 bitstreams. **Statistic:** BD-rate (`rd_probe_analyze --bdrate-vs`) of `mixF3` against
F3 and against the butteraugli loop. **Reading:** `mixF3` shows no steering cost if its BD-rate against F3 has an
interval containing or above zero under both SSIMULACRA2 and butteraugli; CVVDP, zensim and DVIFM are reported
beside them (CVVDP does not decide, per the dial comparison).

### R10 amendment 1 result (2026-11-03, i270): the mix on F3's recipe in the JPEG XL loop

Fresh loop build (`e36_loop_r10.sh`); the B and M7 placement rows reproduce the 2026-10-25r7win calibration (4 of 4),
and all six shared arms (none, butteraugli, B, M7, F3, p30L) reproduce experiment 5 amendment 6's i270 bitstreams in
702 of 702 encodes each. Placements: F3 2.292e-2, p30L 2.277e-2, `mixF3` 2.248e-2 (exponent 0.7). BD-rate, 78
images, positive = fewer bytes at equal judged quality, 95% bootstrap over images:

| `mixF3` against | ssim2 | butteraugli | zensim | DVIFM | CVVDP |
|---|---|---|---|---|---|
| F3, all images | +0.19 [-0.09, +0.46] | -0.21 [-0.56, +0.12] | +0.26 [-0.02, +0.56] | -0.25 [-0.99, +0.40] | +0.17 [-0.22, +0.59] |
| F3, photo (38) | +0.37 [+0.22, +0.51] | -0.22 [-0.40, -0.03] | +0.27 [-0.01, +0.53] | +0.12 [-0.48, +0.95] | -0.03 [-0.18, +0.12] |
| F3, illustration (22) / screen (18) | -0.43 / +0.56 (both across 0) | -0.64 / +0.35 (both across 0) | | | |
| butteraugli loop, all images | -2.18 | -5.31 | -1.66 | -3.04 | -4.22 |

**Reading (registered):** `mixF3` shows no steering cost: against F3 both deciding intervals (SSIMULACRA2 and
butteraugli) contain zero over all images. On photos it gains under SSIMULACRA2 (+0.37%) and loses slightly under
butteraugli (-0.22%). Like every zensim arm it stays 2-5% behind butteraugli's own loop (F3: -2.42 / -5.17 / -4.44).
G-STEER's broad-220-b8 miss does not show in the loop. Whether a single-case G-STEER miss with a clean loop result is
acceptable for adoption is the owner's decision; nothing is adopted on this read. Tables:
`benchmarks/e36_loop_r10_2026-11-03/`; encodes and judge tables on tower
`/mnt/tower/output/zensim/e36-subset-test-2026-11-01/r10-amend1-jxl-i270/` (SHA256SUMS), beside the experiment 3
step 4 and experiment 5 amendment 6 runs (`exp3-step4-jxl-i270/`, `exp5-amend6-jxl-i270/`), mirrored from i270 the
same day.

### R4 registration (2026-11-03, before any run): DVIFM's painted map as the map-term target

Owner, 2026-11-03, on decision 5: "I will take your reccomendation as I trust your experiment ideas to be worth
testing" (option b1). **Hypothesis** (method review, experiment 4): butteraugli's block map is one metric's account of
where distortion sits; a second spatial target may keep the map term's JPEG XL loop gains with less cost on the
pairwise human sets (read 1: p30L + map 300 -0.0052 against p30L on CLIC 2021; the private out-of-fold check found a
cost of the same sign). **Target:** dvifmish preset `ycbcr3-curve-ours-safesyn` (fitted on SafeSyn TRAIN with
SSIMULACRA2 as the teacher; no human labels), its painted map summed on the map pool's 16 px grid by
`examples/block_sums.rs` (dvifmish 263636ef, binary sha256 db64937e1eecff4d559c896ade04e4ddc151a745973e6730dbf80b06d4447e55)
on the map pool's views minus the one view whose DVIFM map is all zero (`rav1e-160-100-df626527`), 1,966 views;
`map_pool_subset.py --exclude-keys`, then `map_pool_retarget.py` (zensimx bf9e227b). On those views the preset's map
has median per-view Spearman 0.54 with the absolute pixel difference (butteraugli's map 0.80) and 0.61 with
butteraugli's map. **Arm** `p30L-map300dv`: each command is experiment 3's p30L + map 300 (amendment 4) at the same
seed with only the map pool, destination and scratch directory changed (`e36_exp56/r4_make_cmds.py`); seeds 0-2, and
the AIC2026 public_train folds at seed 0 (private). The twin's pool keeps the dropped view. The consensus arm of the
draft is not run (option b1): its loop would have had CVVDP as its only independent judge. **Measures:** the
amendment-2 set (primary, G-STEER, G-CHROMA-L, NEARID at seed 0, error against human ratings, EVAL compression rows),
held-out map agreement against butteraugli's and DVIFM's block sums (`map_agreement.py`), the M1 panel against the
twin, the private out-of-fold check against the twin, p30L and F3's control, and, if the arm passes the guards, the
JPEG XL loop at seed 0 judged by butteraugli (SSIMULACRA2 taught the preset), with CVVDP, SSIMULACRA2, zensim and
DVIFM reported. CLIC 2021 is not read. **Guards** (against the twin): compression-row SROCC seed mean falls by no more
than 0.003, all-nine error rises by no more than 0.30 points, G-STEER passes amendment 2's guard, G-CHROMA-L and
NEARID pass. **Reading:** an arm that passes the guards and whose out-of-fold interval against the twin lies above
zero is reported as reducing the map term's pairwise cost; if its interval against p30L also contains zero, it is
reported as a map-term arm without a measured pairwise cost, beside its loop result. Nothing is adopted on this read.

### R10 amendment 2 (2026-11-03, before any of these run): remaining product gates and confirmation read 3

Owner, 2026-11-03: "The mix looks promising, but we need to do more evals". R10's post (like every post since
experiment 3) ran G-STEER, G-CHROMA-L and NEARID at seed 0 only; the subset test's guards and amendment 2's set also
ran G-TRADE, the dial contract and identity at every seed. For `mixF3` seeds 0-2, with the same tools
(`e36_exp56/postR10b.sh`):

- **G-TRADE (a):** the gauntlet trade pairs above 64 px both references call worse (`gauntlet_features.py`, evif
  `tools/trade_gate.py`), share the model scores better (`e36_exp56/gtrade_guard.py`, which reproduces the subset
  test's recorded values: E 0.0761, F3 0.0250 over 18,267 pairs). Guard: the three-seed mean no more than 0.5 points
  above F3's control's (2.36%).
- **Dial contract C1-C6 on both grids and identity at 12 SIMD permutations** (`fchroma-gates.sh` steps verdict and
  identity): every seed must pass, as F3 does.
- **NEARID at seeds 1 and 2:** N1-N3 must pass.
- **Confirmation read 3, CLIC 2021 validation** (registered here, before the read): `mixF3` and mix + p30L, seeds
  0-2, each against F3's control at the same seed; read 1's features, statistic (two-alternative agreement, soft
  labels), whole-reference bootstrap (315 references, 10,000 draws) and reading rule (supported if the seed-mean
  difference's interval lies above zero, contradicted if below, otherwise neutral). Disclosure: this is CLIC 2021's
  third read, and F3's control has been read twice.

Reading: `mixF3` passes this amendment if every gate above passes and read 3 does not contradict it against F3's
control. Its R10 G-STEER miss stands as recorded; nothing is adopted on this read.

### R10 amendment 3 (2026-11-03, before the read): CIQA as a confirmation set for the dial candidates

Owner, 2026-11-03: "Yes, zenjpeg has fast integer idct zenjpeg that can be bitexact with libjpeg-turbo; you can verify
a few images match tensorflow." CIQA's stimuli are the 2026-10-16 private recreation (`docs/DATA_SPLITS.md` D4:
TensorFlow 2.21.0's encoder, 13,713 JPEGs of 6,596 AVA references; evaluation only, never training, tuning,
distillation or an admission gate). Its labels have never been read (D4; the M2 trust screen's "not read"): a
full-reference score needs the pipeline's adjusted pixels, which the recreation recorded only as SHA-256s of
TensorFlow's `decode_jpeg(channels=3, fancy_upscaling=True, dct_method="INTEGER_FAST")` output
(`reference_rgb_sha256`, `adjusted_rgb_sha256`, H×W×3 uint8).

- **Pixels (done before this entry, label-free):** zenjpeg main (`762fa712`) decodes with its jpegli-derived IDCT or
  libjpeg's slow integer IDCT; neither reproduces TensorFlow's fast-integer decode (0 of 20 sampled references;
  libjpeg-turbo's JDCT_IFAST through mozjpeg-sys, the oracle: 20 of 20). The decode was rebuilt on zenjpeg's
  coefficient decoder (`Decoder::decode_coefficients`) with libjpeg-turbo's fast-integer reconstruction: the IFAST
  multiplier table (jddctmgr.c), the x86-64 SSE2 IDCT kernel lane for lane (jidctfst-sse2.asm: 16-bit wrapping
  arithmetic, `pmulhw` on pre-shifted operands, saturating output), fancy upsampling (jdsample.c h2v1, h2v2, h1v2)
  and the 16-bit YCbCr tables (jdcolor.c). All 6,596 references and 13,713 stimuli reproduce their recorded hashes
  (`ciqa-check`, zensim-training `tools/ciqa-check/`; mozjpeg-sys is its oracle, test code only). A first kernel
  written from jidctfst.c's C path matched 20,279 of 20,309; the 30 misses are blocks where a 16-bit intermediate
  overflows (the SIMD kernel saturates where the C path wraps through its range-limit table), and the SSE2-faithful
  kernel matches all 30 with 0 bytes differing from the oracle. The PNGs are private:
  `/mnt/v/zensim/ciqa-private/{references,stimuli}-zenjpeg-ifast/` on dev, mode 0700, never published or mirrored.
- **Features (label-free):** `e36_exp56/ciqa.py keys` reads only pairs.tsv's `ref`, `a`, `b` and `split` and writes
  the 13,713 distinct (reference, stimulus) pairs as assessment keys (token `ciqa`, `instrument.json` with
  `labels_read: false`); `rev5_bank.py extract` with the frozen Rev7 extractor every E36 bank uses (build
  `b77883be`, binary `ed495741…`, era `rev7_rowlanes`), into `/mnt/v/zensim/ciqa-private/e36-read/bank/ciqa/`.
- **The read (registered here, before it happens; `e36_exp56/ciqa_confirm.sh`):** pairs.tsv's 7,902 questions
  (6,329 native train, 1,573 native test; `p_a` = share of raters preferring `a`; the library already dropped 16
  exact repeats), every row its own question, soft-label two-alternative agreement (clic.py's `pair_agreement`, a
  tie counts one half). Models: F3's control, p30L, `mixF3` and mix + p30L at seeds 0-2 (the dial comparison's
  candidates), each arm against F3's control at the same seed; whole-reference bootstrap (6,596 references, 10,000
  draws, seed 20261103) of the seed-mean paired difference; all rows primary, the native test split reported beside
  it. Reading rule as read 1: supported if the interval lies above zero, contradicted if below, otherwise neutral.
  `ciqa.py confirm` ledgers the read in `LABEL_READS.json` before parsing `p_a`; outputs stay under the private root.
- **Disclosures:** the stimuli are a recreation, not the raters' files, so parity with the originals is unknowable
  (D4); raters compared two compressed versions of one photo, so each question is a within-reference JPEG ordering
  at one codec; the raters' display decode is unknown and the recreation's decode is the one scored; every
  reference is itself a JPEG; no zensim model, third-party metric or statistic has touched these labels before this
  read; CIQA stays evaluation-only under the owner's 2026-10-16 condition, so nothing is fitted, tuned or selected
  on it except by this reading rule.

Reading: a further evaluation of the dial candidates, as the owner asked ("we need to do more evals"); nothing is
adopted on it.

### R10 amendment 2 result (2026-11-03): the mix passes the remaining gates; read 3 is neutral

`mixF3` seeds 0-2 (`e36_exp56/postR10b.sh`; outputs under `~/tmp/e36test/postR10b/`, `gates/mixF3_s*-vi/`,
`nearid/out-mixf3s{1,2}/` and `clic21-confirm-3/` on dev):

- **G-TRADE (a)**, 18,267 trade pairs over 64 px, 173 sources: `mixF3` 2.47% / 2.55% / 2.38% (mean 2.47%) against
  F3's control 2.50% / 2.35% / 2.24% (mean 2.36%, the subset test's recorded value). +0.11 points against a bar of
  +0.5: **pass**.
- **Dial contract C1-C6:** PASS on both grids at every seed (standard C1 0.997 / 0.998 / 0.998, ladder C1 0.992 /
  0.994 / 0.992; C3 0.768 / 0.741 / 0.724). **Identity:** 12 SIMD permutations, 62 sources identical at every tier,
  every seed. **pass**.
- **NEARID at seeds 1 and 2:** N1-N3 pass, 132 of 144 monotone ladders at both (seed 0: 132). **pass**.
- **Confirmation read 3, CLIC 2021 validation** (4,812 questions, 315 references; `clic21-confirm-3/CONFIRM.json`):
  F3's control 0.6445 / 0.6443 / 0.6398; `mixF3` 0.6417 / 0.6456 / 0.6444, seed-mean difference **+0.0010 [-0.0024,
  +0.0044], neutral**; mix + p30L 0.6386 / 0.6382 / 0.6378, **-0.0047 [-0.0091, -0.0003], contradicted**.

Reading: `mixF3` passes the amendment (every gate passes and read 3 does not contradict it). Its R10 G-STEER miss
(133 / 134 / 134, broad-220-b8 at every seed) stands as recorded. Mix + p30L is contradicted on CLIC 2021, as the
hinge arms were in read 1. CLIC 2021 validation has now been read three times; F3's control four.

### R10 amendment 3 result (2026-11-03): CIQA cannot separate the candidates

`e36_exp56/ciqa_confirm.sh`. Features: 13,713 pairs in 47 s wall (peak RSS 1.1 GiB) on dev. The read is ledgered in
`confirm-1/LABEL_READS.json` under the private CIQA root; its outputs stay there.

| model (seeds 0 / 1 / 2) | all 7,902 questions, 6,596 references | native test split, 1,573 questions, 267 references |
|---|---|---|
| F3 control | 0.6752 / 0.6753 / 0.6753 | 0.6821 / 0.6821 / 0.6821 |
| p30L | 0.6752 / 0.6752 / 0.6752 | 0.6822 / 0.6823 / 0.6823 |
| `mixF3` | 0.6752 / 0.6752 / 0.6753 | 0.6819 / 0.6818 / 0.6821 |
| mix + p30L | 0.6753 / 0.6753 / 0.6753 | 0.6820 / 0.6822 / 0.6822 |

Every model's whole-reference interval is about [0.671, 0.680] on all rows. Arms against F3's control (seed means):
p30L -0.0000 [-0.0001, +0.0001], `mixF3` -0.0000 [-0.0002, +0.0002], mix + p30L +0.0001 [-0.0001, +0.0003]: all
**neutral**, and the same on the test split.

Why the read is flat (diagnostics on the same read; the quality settings and the ceiling are label-free or follow
from the labels already read):

- Every question compares two JPEG quality settings of one photograph (median gap 27 settings; 9.9% of questions
  within 5). The rule "the higher setting wins" scores 0.6750. The ceiling, siding with every majority, is 0.6975.
  Every candidate orders 99.4-99.5% of the questions as the setting does (40-51 exceptions of 7,902), and the
  candidates differ from one another on 0.2-0.3% of the questions (F3's control against `mixF3` 24, against p30L 14,
  against mix + p30L 24; the control's own seeds differ on 4-7), at gaps averaging 4 settings.
- The majority prefers the lower setting on 1,211 questions (15.3%; median gap 14). On those no candidate agrees
  better than 0.49-0.53. By the lower setting of the pair: at q 10-30 (3,195 questions) the setting rule scores 0.803
  against a ceiling of 0.808; at q 71-90 (751) 0.508 against 0.579; at q 91-100 (73) 0.480 against 0.579. The room
  a metric could use is in the near-transparent pairs, and no candidate uses it.

Consequence: CIQA as published (one codec, within-reference pairs, soft labels) measures whether a metric orders JPEG
settings, which every candidate does; it is not an instrument for choosing among them. It has now been read once;
no further read is planned unless a candidate disagrees with the setting order. The owner's evaluation-only
condition stands.

### R4 result (2026-11-03): the map term's pairwise cost does not come from butteraugli's map

Arm `p30L-map300dv` (p30L + map 300 trained toward DVIFM's `ycbcr3-curve-ours-safesyn` block sums), seeds 0-2 and
the seven AIC2026 folds at seed 0; twin `p30L-map300` (butteraugli's map); `~/tmp/e36test/postR4/` on dev.

| measure (seeds 0 / 1 / 2) | p30L | twin: p30L + map 300 (butteraugli) | p30L + map 300 (DVIFM) |
|---|---|---|---|
| E36 misordering, 40 held-out sources (seed mean, vs control 3.36%) | 2.38% (-0.975) | 2.12% (-1.241 [-1.436, -1.057]) | 2.31% (-1.047 [-1.243, -0.874]) |
| EVAL compression rows, row-weighted SROCC | 0.8956 / 0.8957 / 0.8935 | 0.8954 / 0.8942 / 0.8945 | 0.8982 / 0.8982 / 0.8976 |
| EVAL compression rows, mean error | 6.82 / 6.85 / 7.27% | 6.88 / 6.90 / 6.87% | 6.58 / 6.63 / 6.67% |
| error vs human, common three | 8.96 / 8.85 / 8.82% | 8.64 / 8.71 / 8.69% | 8.69 / 8.72 / 8.73% |
| error vs human, all nine | 8.40 / 8.37 / 8.41% | 8.03 / 8.10 / 8.08% | 8.06 / 8.10 / 8.07% |
| M1 panel against the twin (row-weighted SROCC) | | 0 | +0.0027 (descriptive) |
| G-STEER, of 135 | 135 / 134 / 135 | | 135 / 134 / 135 |
| G-CHROMA-L; NEARID (seed 0) | pass | pass | pass; N1-N3 pass |
| held-out map agreement with DVIFM's block sums, 8 px, served median (239 views) | 0.536 | 0.570 / 0.568 / 0.571 | 0.575 / 0.573 / 0.574 |
| the same with butteraugli's block sums | | (experiment 3) | 0.781 / 0.781 / 0.778 |

(F3's control agrees with DVIFM's block sums at 0.512; at 16 px the pattern is the same: F3 0.614, p30L 0.618, twin
0.654 / 0.654 / 0.650, DVIFM arm 0.648 / 0.649 / 0.650.)

Guards against the twin: compression-row SROCC rises (+0.003 seed mean; guard: fall no more than 0.003); all-nine
error +0.01 points (guard +0.30); G-STEER within amendment 2's guard; G-CHROMA-L and NEARID pass. **All pass.**

**The registered question** (private out-of-fold check on AIC2026 public_train, zensim-training
`evaluations/aic2026_oof_e36_candidates_2026-11-02.md`, section R4): the DVIFM arm's interval against the twin
contains zero, and its intervals against p30L and against F3's control lie below zero, as the twin's do. So the arm
is **not** reported as reducing the map term's pairwise cost: training the map toward DVIFM's account of where the
damage sits costs the same pairwise agreement across images as training it toward butteraugli's. The two targets
also pull the map the same way: the DVIFM arm agrees with DVIFM's block sums only 0.004 better than the twin does,
and both gain about 0.04 over p30L.

Reading: the hypothesis of the method review's experiment 4 (that the cost came from butteraugli's map in
particular) is not supported; the cost belongs to steering the map at all, at this weight. The DVIFM arm keeps most
of the twin's E36 gain (2.31% against 2.12%, p30L 2.38%) and improves the EVAL compression rows, so it is a map-term
arm with the same pairwise cost as the twin. As registered, the arm passed its guards, so its JPEG XL loop at seed 0
runs next on i270, judged by butteraugli with CVVDP, SSIMULACRA2, zensim and DVIFM reported (`e36_loop_r4.sh`);
the loop result is appended below when it lands. Nothing is adopted on this read.

### R10 amendment 4 (2026-11-03, before the read): the library's other MOS sets, resolved to one

Experiment 5 stage 3 registered "other library MOS sets in no training leg and not in EVAL" as the mix's confirmation.
Resolved from the IQA catalogue (`zenpapers/datasets/iqa_catalog.tsv`, 681 rows, 103 full-reference image sets in
the library) against F3's legs (safesyn, cid22, coverage, the human sources KADID, TID2013, KonFiG, CID22-A25, AIC-3,
the dataset legs PIPAL, MDID, LIVE-MD, CID:IQ, CID:OA, IRCCyN-IVC, Colourlab, BAPPS, chroma teacher), EVAL's nine
collections (CID:IQ, CID:OA, Colourlab, IQA-ODI, IRCCyN-IVC, LIVE-MD, MDID, NITS-IQA, VCL-FER) and `docs/DATA_SPLITS.md`:

- Reserved or sealed, not touched: CSIQ, LIVE release 2, KonJND JPEG, MCL-JCI (Rev4 confirmation); CLIC 2022, CID22-B,
  KADID TERMINAL, AIC2026 public_test; the threshold sets (JND-MID-MA, VLIC, Konstanz PJND).
- Share a training source: NNCD-IQA, MIDD-CAP, MIDD-GAR, TID2008 (TID2013's photographs); mucped23 (17 CID22-A25
  sources); IRCCyN/IVC Toyama and MICT Toyama (Kodak scenes that contain TID2013 and SafeSyn sources; DATA_SPLITS,
  October 25); IDSQS-AIC and Konstanz feedback (AIC-3 sources); PC-IQA (9 of 15 references are IRCCyN-IVC sources, 6
  are LIVE's); mabyduck's CLIC 2024 reproduction (CLIC 2024, R7's leg, and CLIC 2021's protocol family).
- Already read: CLIC 2021 validation (three times); CIQA (amendment 3). Fidelity-LIC measures GAN-vs-non-GAN
  preference that fidelity metrics miss (M2) and has protected sources.
- Not in the library's content (catalogued only): A57, ESPL synthetic, IRCCyN/IVC LAR, JPEG AI subjective (EPFL),
  MD-IVL, MDID2013, MP2020, SD-IVL, VDID2014, CAIP HEVC patches, EPFL nearly visually lossless. The owner's rule
  (2026-10-25) forbids downloading datasets, so none is read.
- Remaining compression-centric set with content, in no leg, not in EVAL, not reserved, no known source overlap:
  **MMSPG JPEG XR** (186 rated conditions, 6 photographs, 13 raters; JPEG 4:2:0, JPEG 2000 4:2:0 and 4:4:4, JPEG XR
  with and without overlap, 6 rates each). Prior exposure: the M2 trust screen scored it with third-party metrics
  only (SSIMULACRA2 0.928, butteraugli 3-norm 0.892, CVVDP 0.911, DVIFM 0.3 0.928; reliability 0.978).
  Non-compression sets with content (watermarking, DIBR, denoising, enhancement, super-resolution, tone mapping) are
  outside this question and are not read here.

**The read** (`e36_exp56/mos_confirm.py`, token `mmspg`): pixels are the M2 screen's zenbitmaps decodes of the
library's BMPs (`decode_map.tsv`, content-addressed PNGs copied from i270 to `/mnt/v/zensim/e36-read4-mmspg/`);
`keys` reads `item` and `ref` only; features with the frozen Rev7 extractor; `confirm` ledgers the read before parsing
`score`. All 186 rows (the 6 reference rows included, as the screen scored them). Models: F3's control, p30L, `mixF3`,
mix + p30L at seeds 0-2. Per model: SROCC with the canonical whole-reference interval, within-reference SROCC, error
(median absolute residual around the 4PL fit over the 1-99% span, the EVAL readout's statistic). Arms against F3's
control at the same seed: seed-mean SROCC and error differences, whole-reference bootstrap (10,000 draws, seed
20261103). **Disclosure stated before the read:** six references cannot support a bootstrap interval; the read is
descriptive, with the within-reference SROCC (which does not depend on cross-photograph level) the more informative
number. Nothing is adopted on it.

### R10 amendment 4 result (2026-11-03): MMSPG JPEG XR, neutral for every arm

`mos_confirm.py confirm` (`/mnt/v/zensim/e36-read4-mmspg/confirm-1/`; features 186 pairs in 2.8 s). 186 rated conditions,
6 photographs; intervals are the canonical whole-reference bootstrap over those 6 photographs (descriptive, as
disclosed before the read).

| model (seeds 0 / 1 / 2) | SROCC | within-reference SROCC | error (median |residual| around the 4PL fit, of the 1-99% span) |
|---|---|---|---|
| F3 control | 0.9203 / 0.9217 / 0.9212 | 0.925 / 0.928 / 0.928 | 5.44 / 5.47 / 5.43% |
| p30L | 0.9172 / 0.9175 / 0.9171 | 0.925 / 0.925 / 0.927 | 5.19 / 5.30 / 5.27% |
| `mixF3` | 0.9201 / 0.9219 / 0.9215 | 0.931 / 0.928 / 0.928 | 5.63 / 5.53 / 5.47% |
| mix + p30L | 0.9191 / 0.9192 / 0.9200 | 0.924 / 0.929 / 0.926 | 5.48 / 5.76 / 5.65% |

Each model's SROCC interval is about [0.88, 0.95]. Arms against F3's control (seed-mean SROCC difference, bootstrap
over the 6 photographs; error difference in points): p30L -0.0038 [-0.0082, +0.0001], -0.20 points; `mixF3` +0.0001
[-0.0052, +0.0055], +0.09 points; mix + p30L -0.0016 [-0.0110, +0.0099], +0.18 points: all **neutral**. For scale,
the M2 screen's third-party metrics on the same rows: SSIMULACRA2 0.928, DVIFM 0.3 0.928, CVVDP 0.911, butteraugli
3-norm 0.892; every zensim candidate sits between butteraugli and SSIMULACRA2.

Reading: the set does not separate the candidates either. With this read, every evaluation registered for the mix is
done except its JPEG XL loop, which already ran (amendment 1): the mix passes every product gate but one G-STEER case,
no held-out human read contradicts it (CLIC 2021, CIQA and MMSPG neutral; AIC2026 out of fold level, codec
consistency better), and its gains are the EVAL human sets it was chosen on. The choice between F3 and the mix is the
owner's.

### R4 loop result (2026-11-03, i270): the DVIFM-target map saves more bytes than its butteraugli-target twin

`e36_loop_r4.sh` (zensimx 6ce4faa3; loop build `/var/tmp/byconstruction-bin/6ce4faa33c5e-loop-e36r4`): every arm
encoded fresh on i270 by one build; the B/M7 placement rows reproduce the earlier calibration exactly; the seven
shared arms (none, butteraugli, B, M7, F3, p30L, map 300) reproduce experiment 3 step 4's 702 bitstreams each, 0
differ (`REPRODUCTION.txt`). 78 images; BD-rate (positive = fewer bytes at equal judged quality; 95% bootstrap over
images); butteraugli judges, the others are reported. Tables: `benchmarks/e36_loop_r4_2026-11-03/`; encodes, judge
tables and BD-rate tables: `/mnt/tower/output/zensim/e36-subset-test-2026-11-01/r4-jxl-i270/` (SHA256SUMS).

| arm, against p30L | ssim2 | butteraugli | zensim | DVIFM | CVVDP |
|---|---|---|---|---|---|
| p30L + map 300, DVIFM target (R4) | +1.03 [+0.77, +1.30] | +1.18 [+0.90, +1.45] | +0.31 [+0.02, +0.62] | +1.21 [+0.48, +1.99] | +1.34 [+1.03, +1.66] |
| p30L + map 300, butteraugli target (the twin) | +0.51 [+0.27, +0.72] | +0.86 [+0.51, +1.18] | +0.29 [-0.08, +0.64] | +0.49 [-0.64, +1.67] | +0.86 [+0.53, +1.22] |
| F3 | -0.04 [-0.22, +0.16] | -0.30 [-0.51, -0.12] | -0.21 [-0.49, +0.04] | +0.09 [-0.70, +0.94] | -0.13 [-0.40, +0.14] |
| butteraugli loop | +2.21 [+1.51, +3.03] | +4.55 [+3.88, +5.28] | +1.52 [+0.74, +2.32] | +2.60 [+1.36, +3.85] | +4.00 [+3.18, +4.84] |

Against the twin, the DVIFM arm: ssim2 +0.49 [+0.22, +0.79], butteraugli +0.27 [-0.15, +0.68], zensim -0.04 [-0.48,
+0.43], DVIFM +0.34 [-0.34, +1.07], CVVDP +0.47 [+0.11, +0.86]. Against the butteraugli loop (ssim2 / butteraugli /
CVVDP): the DVIFM arm -1.35 / -3.68 / -2.91, the twin -1.89 / -4.00 / -3.49, p30L -2.39 / -4.88 / -4.34, so the DVIFM
arm closes a quarter of p30L's butteraugli gap and a third of its CVVDP gap (the twin a fifth of each). By class
against p30L (ssim2 / butteraugli / CVVDP): photo (38) +0.81 / +0.99 / +1.01, illustration (22) +0.99 / +0.91 / +1.09;
the twin +0.75 / +1.27 / +1.13 and +0.43 / +0.80 / +0.63.

Reading (the registered rule): the arm passed its guards, and its out-of-fold interval against the twin contains
zero, so it is reported as a map-term arm with the twin's pairwise cost, beside this loop result: its byte savings
exceed the twin's on the two judges the map term never trained toward (SSIMULACRA2 and CVVDP, intervals above zero)
and tie on butteraugli. The map term's question for the owner is sharper, not different: about 1% of JPEG XL bytes
at equal judged quality (1.2% on the butteraugli judge) against a small, consistent loss of pairwise agreement
across images (AIC2026 out of fold -0.0155 [-0.0245, -0.0071] global SROCC against p30L; the twin's CLIC 2021 read 1
-0.0052). Nothing is adopted on this read.

### R10 amendment 5 (2026-11-03, before the runs): reference metrics on the confirmation sets

Owner, 2026-11-03: "Do we have comparative evals vs ssim2/butteraugli-3norm/fsimc/haarxyb-ssim-mask-deltaL1248 (we
should consider the latter as a teacher, if it does as well as f3mix or better on other dataset evals)". The
candidates' confirmation reads so far compared them with each other; SSIMULACRA2, butteraugli and CVVDP sit beside
them on EVAL's compression rows (R2 amendment 1), CLIC 2021 and MMSPG (the M2 screen) and AIC2026 (private), FSIMc only
on AIC2026 (zenmetrics' port, scores of 2026-10-20), and the AIC-4 Haar metrics only on AIC2026 and MCL-JCI
(zenmetrics-private's `aic4` port of the AIC-4 framework at `ecec4e0`, the published `HaarXYB-SSIM-deltaL1248` and
`HaarXYB-SSIM-mask-deltaL1248` configs; zenpapers `docs/jpeg-aic/metric-accuracy-and-jnd-mapping.md` §14).

**Registered here:** score SSIMULACRA2 (fast-ssim2 0.9.0), butteraugli 3-norm (0.9.3), FSIMc and FSIM-Y (zenmetrics
`fsim`), HaarPSI (zenmetrics `haarpsi`) and the two published AIC-4 Haar configs (JND output; `jnd-ladder-score`,
weights `/mnt/v/zenmetrics-private/aic4/models/ecec4e0`) on exactly the rows the candidates were read on, and report
them with the candidates' statistics: EVAL's nine collections (readout.py error and SROCC, ceval.py compression rows;
selection data for the mix, not for the metrics), CLIC 2021 validation (4,812 questions; agreement), MMSPG JPEG XR
(186 rows; SROCC, within-reference, error), CIQA (7,902 questions; agreement), and AIC2026 public_train (private:
global and within-source SROCC, codec rank-bias and JND-unit spreads, `aic2026_codec_uniformity.py`).
`e36_exp56/refmetric_judges.py` writes the scores as `judge_*.tsv` files beside the candidates' predictions, so the
existing `confirm` scripts report them. Disclosures: these are further reads of CLIC 2021's, MMSPG's and CIQA's labels
(by metrics, no zensim model); the AIC-4 Haar configs were fitted by their author on AIC data, so their AIC2026
numbers are in-sample and a Haar-taught zensim leg would carry that fit (which may include the sealed public_test)
into training. Nothing is adopted on these reads; the teacher question is decided afterwards, on the held-out sets.

### R10 amendment 5 result (2026-11-03): reference metrics on the confirmation sets

Scored as registered: `jnd-ladder-score` (zenmetrics-private `experiments/jnd-ladder` at `040c780f`, built on i265)
on the pair lists of EVAL (4,678 decoded pairs), CLIC 2021 validation (4,812 questions), MMSPG JPEG XR (186) and CIQA
(7,902), judge files written by `refmetric_judges.py`, statistics by the same `confirm` scripts, `readout.py`,
`ceval.py` and the new `refmetric_eval_readout.py` (every MOS collection, with coverage). Driver:
`e36_exp56/refmetrics_run.sh`. Memory: the Haar pyramids take ~14 GB per pair per thread on IQA-ODI's 8K JPEGs; a
16-thread run died at its 28G cap and a 6-thread run under 44G pushed dev out of memory machine-wide (other sessions'
jobs were killed) before the cap fired, so EVAL's IQA-ODI pairs were rescored one pair and one config at a time under
20G (peak RSS 11.2 GiB). AIC2026 is in the private evaluation
(`zensim-training/evaluations/aic2026_codec_uniformity_refmetrics_2026-11-03.md`).

**Held-out sets** (the candidates' numbers are the confirmation reads above; metrics on the same rows):

| score | CLIC 2021 validation, agreement [95%] | MMSPG JPEG XR: SROCC / within-reference / error | CIQA, agreement |
|---|---|---|---|
| mix on F3's recipe, seeds 0 / 1 / 2 | 0.6417 / 0.6456 / 0.6444 [about ±0.013] | 0.920-0.922 / 0.928-0.931 / 5.47-5.63% | 0.6752-0.6753 |
| F3 control, seeds 0 / 1 / 2 | 0.6445 / 0.6443 / 0.6398 | 0.920-0.922 / 0.925-0.928 / 5.43-5.47% | 0.6752-0.6753 |
| SSIMULACRA2 | 0.6261 [0.6122, 0.6404] | 0.934 / 0.942 / 5.19% | 0.6753 |
| butteraugli 3-norm | 0.6056 [0.5915, 0.6204] | 0.902 / 0.896 / 6.58% | 0.6750 |
| FSIMc | 0.6374 [0.6241, 0.6515] | 0.929 / 0.912 / 4.69% | 0.6748 |
| FSIM (luma) | 0.6381 [0.6245, 0.6522] | 0.931 / 0.922 / 4.26% | 0.6748 |
| HaarPSI | 0.6031 [0.5894, 0.6178] | 0.921 / 0.923 / 5.64% | 0.6757 |
| HaarXYB-SSIM-deltaL1248 (AIC-4) | 0.6100 [0.5966, 0.6247] | 0.923 / 0.922 / 4.89% | 0.6752 |
| HaarXYB-SSIM-mask-deltaL1248 (AIC-4) | 0.6303 [0.6163, 0.6449] | 0.926 / 0.926 / 5.10% | 0.6753 |

**EVAL** (`refmetric_eval_readout.py` over the nine MOS collections: error = median absolute residual around the 4PL
fit over the target's 1-99% span, mean over collections; `ceval.py` compression rows, 417 rows in six collections).
"Eight" = the collections the mix holds out (all but MDID, which the mix trains on; F3 trains on six of the nine, so
its eight-collection number is in-sample on five); the three shared sets are IQA-ODI, NITS-IQA and VCL-FER.

| score | eight held-out: error / SROCC | three shared: error / SROCC | compression rows: SROCC / error | MDID SROCC |
|---|---|---|---|---|
| mix, seeds 0 / 1 / 2 | 8.18 / 8.32 / 8.28% · 0.861 / 0.858 / 0.858 | 8.27 / 8.39 / 8.40% · 0.850 / 0.848 / 0.849 | 0.8954 / 0.8941 / 0.8940 · 6.95 / 6.99 / 7.13% | 0.927 (trained) |
| F3, seeds 0 / 1 / 2 | 8.67 / 8.72 / 8.66% · 0.848 / 0.848 / 0.848 | 8.79 / 8.74 / 8.89% · 0.835 / 0.835 / 0.836 | 0.8949 / 0.8926 / 0.8935 · 7.03 / 7.08 / 7.01% | 0.886 (trained) |
| SSIMULACRA2 | 9.13% · 0.831 | 8.52% · 0.859 | 0.8963 · 6.92% | 0.787 |
| butteraugli 3-norm | 11.57% · 0.709 | 10.47% · 0.796 | 0.8544 · 8.61% | 0.572 |
| CVVDP | 9.82% · 0.823 | 9.52% · 0.829 | 0.8763 · 7.07% | 0.829 |
| FSIMc | 9.97% · 0.830 | 8.84% · 0.864 | 0.9055 · 6.73% | 0.893 |
| FSIM (luma) | 10.06% · 0.822 | 8.86% · 0.857 | 0.9021 · 6.67% | 0.886 |
| HaarPSI | 8.92% · 0.843 | 8.99% · 0.828 | 0.8878 · 6.90% | 0.920 |
| HaarXYB-SSIM-deltaL1248 | 9.66% · 0.805 | 9.63% · 0.822 | 0.8848 · 7.32% | 0.770 |
| HaarXYB-SSIM-mask-deltaL1248 | 9.38% · 0.810 | 9.19% · 0.829 | 0.8927 · 7.18% | 0.777 |

Per collection (SROCC, seed 0 for the models): NITS-IQA SSIMULACRA2 0.777, FSIMc 0.746, mix 0.734, F3 0.708;
IQA-ODI FSIMc 0.900, FSIM 0.898, mix 0.867, F3 0.858, SSIMULACRA2 0.854; Colourlab mix 0.735, HaarPSI 0.727, F3
0.689, FSIMc 0.624, SSIMULACRA2 0.604, HaarXYB-mask 0.558; CID:IQ mix 0.868, F3 0.863, HaarPSI 0.816, SSIMULACRA2
0.757, FSIMc 0.710; IRCCyN-IVC HaarXYB 0.952, SSIMULACRA2 0.949, F3 0.926, mix 0.919. Full tables:
`~/tmp/e36test/refmetrics/{eval-refmetrics,eval-ceval,heldout_refmetrics_table}.md` on dev.

**Reading.** No reference metric matches the mix across the sets. CLIC 2021, the registered confirmation set, puts
the mix (0.644) above every metric: FSIM/FSIMc 0.638, HaarXYB-mask 0.630, SSIMULACRA2 0.626, HaarXYB 0.610,
butteraugli 0.606, HaarPSI 0.603. On EVAL's eight held-out collections the mix leads by 0.7 points of error and
0.018 of SROCC over the best metric (HaarPSI); SSIMULACRA2 and FSIMc rank better than the mix on the three shared
sets (0.859 and 0.864 against 0.850, from NITS-IQA and IQA-ODI) while the mix's residual error there is the lowest.
On the compression rows FSIMc (0.906) and FSIM (0.902) edge SSIMULACRA2 (0.896), the mix (0.895) and F3 (0.895),
all within 0.011; on MMSPG every score lies within one interval (SSIMULACRA2 0.934 and FSIM 0.931 against the mix's
0.921, the mix ahead within reference); CIQA is flat for everything (the quality-setting ceiling, amendment 3).
**Teacher question (the owner's):** HaarXYB-SSIM-mask-deltaL1248 trails the mix on CLIC (0.630 vs 0.644), on
EVAL's eight (SROCC 0.810 vs 0.861) and on the compression rows (0.893 vs 0.895), is level on MMSPG, and leads only
on AIC2026, where its author fitted it (private evaluation; in-sample, and a Haar-taught leg could carry the sealed
public_test into training). It is not a teacher candidate on this evidence; nothing is adopted.

**Why the CLIC 2021 agreement curve zigzags** (owner question on the methodology doc's chart; corrected after the
owner pointed out that F3's line does not zigzag; `e36_exp56/clic21_pair_agreement.py --paired`, gallery at
`/mnt/v/output/zensim/clic21-dip-2026-11-03/`, http://dev.lan:3300/zensim/clic21-dip-2026-11-03/ on the LAN). The
zigzag is the mix's own and sits at the tie: in its two bins nearest zero (median differences -1.2 and +1.7) the share
preferring crop a is 0.521 then 0.436 at seed 0, while F3's bins there read 0.478 then 0.504 and stay monotone. It is
not an a/b offset (every model's mean difference is +0.5 with 50.7% positive; the raters' share for a is 0.501). Where
the mix's own difference is within 2.4 points, its ordering agrees with the raters 0.438 / 0.463 / 0.456 at seeds 0-2
(n about 640, s.e. 0.020) against F3's 0.483 / 0.481 / 0.473 in its own band (n about 600): the band effect is
seed-consistent, the up-then-down shape partly sampling noise (seeds 1 and 2 read 0.50 then 0.46). On the 680
questions either model calls within 2.4 points, the mix's sign follows SSIMULACRA2's on 76% and PSNR's on 68% (F3:
66% and 64%); those are the questions where the raters go against every reference metric (against PSNR in 75% of the
answers, against the four metrics' consensus on 254 of the 357 questions where they agree). The questions the two
models order differently are about 100 of 4,812 (2%), and which model the raters side with on them changes with the
seed (F3 at seed 0, the mix at seeds 1 and 2), so sign flips are not the cause. Cost: 2-4.5 points of agreement on 13%
of the questions, 0.3-0.6 points overall, returned at moderate differences (the bins at +5.5 and +9.4: 0.621 / 0.714
against F3's 0.588 / 0.704), which is why the totals tie. A natural reading is that TID2013's noise and blur families
make the mix treat added high-frequency texture as damage where CLIC's raters prefer the sharper crop; the crops are
anonymised, so this was not verified. Separately, both lines drop at the largest positive differences (bins 10-11:
0.66-0.68 after 0.74-0.76); that dip is in the questions, not the models: every reference metric and the organisers'
PSNR order those pairs as zensim does, and among the questions answered by two or more raters agreement is 0.80 at
12-20 points, 0.54 at 20-27 and 0.65 above 27. The rated reference images in EVAL (CID:OA 7, Colourlab 3; both models
score an image identical to its reference 100 by construction, the observers rated them 3.2-4.9 of 5) were drawn as
coincident dots in the doc's scatter chart and looked like a ceiling only the mix hit; they are rings now.

**Follow-up (2026-11-04; owner: "verify the sharper-crop reading on a few dip pairs, and eval the iqa metrics against
the non-codec distortions"). Both are further analyses of rows already read; nothing new was exposed.**

*Sharper-crop reading, verified* (`clic21_pair_agreement.py --sharpness`; sharpness = a crop's high-pass energy, luma
minus a sigma-1.5 Gaussian, over its original's, so 1 = the original's fine detail and below 1 = smoothed; the 686
questions whose crops differ by under 0.02 are excluded; luma from the library PNGs through PIL, no colour management
involved). The raters choose the crop with more fine detail on 75% of the 4,126 decidable questions and on 71-78% in
every band. What changes by band is whether a score's preferred crop is the sharper one: F3's is on 58% of all
questions, 68% in the band before the top-end dip, 57% in the dip and 46% in its own near-tie band; the mix's on 57%
overall and **34% in its own near-tie band** (SSIMULACRA2 16%, PSNR 17% there). Where the raters went against the mix
in that band (314 questions) their crop is the sharper one on 85%; where they agreed, on 64%. Over all questions the
mix's score difference minus F3's correlates with the sharpness difference at Spearman -0.20 (and +0.01 with the two
crops' total high-pass energy): relative to F3 the mix moves against the sharper crop of a pair. In the top-end dip
the lower-scored crop is the sharper one on 43% of the questions, and on 70% of those where the raters chose it. Four
zoomed near-tie questions (`zoom/` under the gallery: full-resolution 256-pixel crops at the region of largest
high-pass difference, original | A | B): in three the crop the raters chose visibly keeps dock icons, branch texture or
a knit pattern that the other crop smooths, and in one of them (`neartie_4`) F3 preferred that crop while the mix
preferred the smoothed one; the fourth pair differs by 0.02 and cannot be called by eye. Median sharpness ratio 0.89
over all questions, 0.82 in the near-tie band.

*Non-codec distortions* (`ceval.py --rows noncodec --by-distortion`; 871 EVAL rows in seven collections: CID:IQ blur,
Poisson noise and the two gamut mappings; Colourlab's nineteen other KADID-style distortions and its rated references;
CID:OA's optical aberrations and references; IRCCyN-IVC blur; LIVE-MD blur+noise; NITS-IQA D1-D6; VCL-FER white noise
and blur. MDID, LIVE-MD blur+JPEG and IQA-ODI have a codec in every row and are left out). Row-weighted SROCC / mean
error, seed 0 for the models (seeds 1-2 within 0.005 and 0.2 points):

| score | non-codec rows: SROCC / error | CID:IQ | Colourlab | CID:OA | IRCCyN blur (20) | LIVE-MD blur+noise | NITS D1-D6 | VCL-FER noise, blur |
|---|---|---|---|---|---|---|---|---|
| mix | 0.790 / 8.66% | 0.837 | 0.714 | 0.911 | 0.915 | 0.903 | 0.672 | 0.955 |
| F3 | 0.767 / 9.28% | 0.832 | 0.664 | 0.905 | 0.915 | 0.885 | 0.643 | 0.947 |
| HaarPSI | 0.774 / 9.76% | 0.768 | 0.708 | 0.882 | 0.976 | 0.890 | 0.672 | 0.956 |
| SSIMULACRA2 | 0.759 / 10.59% | 0.698 | 0.575 | 0.847 | 0.915 | 0.921 | 0.748 | 0.960 |
| CVVDP | 0.755 / 10.76% | 0.744 | 0.579 | 0.912 | 0.915 | 0.891 | 0.699 | 0.950 |
| FSIMc | 0.745 / 10.79% | 0.657 | 0.596 | 0.889 | 0.976 | 0.896 | 0.712 | 0.938 |
| HaarXYB-SSIM-mask-deltaL1248 | 0.735 / 10.63% | 0.723 | 0.524 | 0.895 | 0.915 | 0.876 | 0.693 | 0.954 |
| FSIM (luma) | 0.734 / 11.02% | 0.645 | 0.577 | 0.887 | 0.976 | 0.891 | 0.706 | 0.907 |
| HaarXYB-SSIM-deltaL1248 | 0.723 / 11.03% | 0.661 | 0.526 | 0.900 | 0.915 | 0.863 | 0.687 | 0.951 |
| butteraugli 3-norm | 0.661 / 13.07% | 0.652 | 0.316 | 0.577 | 0.794 | 0.840 | 0.749 | 0.950 |

Per distortion type (20 rows or more; SROCC): CID:IQ Gaussian blur CVVDP 0.82, mix 0.81, HaarXYB-mask 0.81, F3 0.79,
HaarPSI 0.71, FSIMc 0.66, SSIMULACRA2 0.64; SGCK gamut mapping CVVDP 0.94, SSIMULACRA2 0.93, F3 0.91, mix 0.87, FSIMc
0.86; DeltaE gamut mapping and Poisson noise 0.94-0.97 for everything but butteraugli. CID:OA defocus / spherical
aberration / vertical astigmatism: mix 0.88 / 0.85 / 0.87, F3 0.87 / 0.83 / 0.86, CVVDP 0.88 / 0.86 / 0.86, FSIMc 0.85
/ 0.85 / 0.78, SSIMULACRA2 0.79 / 0.82 / 0.67, butteraugli 0.42 / 0.53 / 0.37. NITS-IQA D1-D3 (Gaussian blur, Gaussian
and uniform noise) 0.89-0.94 for everything; D4 contrast change fails for every score (zensim 0.22, SSIMULACRA2 0.46,
butteraugli 0.43, the rest 0.39-0.41), which is where zensim loses its pooled NITS number (0.64-0.67 against
SSIMULACRA2's 0.75); D5 pixelate mosaic butteraugli 0.85, HaarPSI 0.85, FSIMc 0.83, zensim 0.76-0.80; D6 motion blur
0.61-0.64 for everything. VCL-FER white noise 0.95-0.98 (SSIMULACRA2, CVVDP, HaarPSI 0.97; zensim 0.95-0.96), blur
0.94-0.96. Colourlab pooled by KADID family: blur (27 rows) CVVDP 0.96, HaarPSI 0.95, SSIMULACRA2 0.94, zensim 0.89,
FSIMc 0.78; noise (27) zensim 0.92-0.93, the metrics 0.88-0.90; colour (36: colour shift, colour quantization,
saturation up and down) SSIMULACRA2 0.60, FSIMc 0.59, mix 0.54, F3 0.48, CVVDP 0.38, HaarPSI 0.37;
brightness/contrast (45) HaarPSI 0.77, FSIMc 0.76, mix 0.74, F3 0.73, SSIMULACRA2 0.72, butteraugli 0.58;
sharpen/quantize (18) HaarXYB-mask 0.82, CVVDP 0.79, mix 0.73, F3 0.70; geometric (18) 0.47-0.55 for everything.

Reading: on distortions with no codec in them the mix leads every reference metric by 0.016-0.13 of row-weighted SROCC
and by at least 1.1 points of error (HaarPSI nearest), and its gain over F3 (0.790 against 0.767) comes from
Colourlab's colour, brightness/contrast and sharpen/quantize families, LIVE-MD blur+noise and NITS, while it gives back
0.04 on CID:IQ's SGCK gamut mapping. The reference metrics' strengths are specific: CVVDP and HaarPSI on blur,
SSIMULACRA2 and FSIMc on Colourlab's colour family, SSIMULACRA2 on LIVE-MD blur+noise, butteraugli and HaarPSI on
pixelation; butteraugli is weak on optical aberrations and colour. Contrast and brightness changes, geometric
distortions and NITS D4 defeat every score here, zensim included. With the compression rows (FSIMc 0.906, mix 0.895)
this is the shape of the comparison: the metrics sit at or just above zensim on codec distortions and below it on
everything else.

### R10 amendment 6 (2026-11-04, before the run): HDR evidence panel for F3 and the mix

Owner, 2026-11-04: "how does the mix do on hdr vs natural butter-3 and ssim2?" Neither candidate has an HDR read; E's
is `benchmarks/hdr_e_vs_field_2026-10-30.md`. **Registered here:** F3 (seeds 0-2) and `mixF3` (seeds 0-2) through the
HDR evidence panel exactly as E was read (`scripts/byconstruction/hdr_panel_upiq_all.py`, native revision-7 banks under
`/var/tmp/byc-hdr-20261029/banks` on dev, PU21 front end, `byc_predict` with the fx2 declaration; no re-extraction, no
fit, no selection): UPIQ-380 (380 HDR pairs, 30 references, JOD; T2 TRAIN rows that no E36 candidate ever trained on,
so held out for them, and already read for E on 2026-10-30), its 50 development rows, HDRTEACH VAL (3,900 rows,
HDR-VDP-3 Q_JOD: agreement with a model, not with people), the 4Kdtb EVAL half (48 rows, MOS) and the consumer-EVAL
HDR chroma ladders (3,245 rungs, rank agreement with natural SSIMULACRA2, label-free). Reference metrics: the stored
per-pair UPIQ panels (butteraugli 3-norm and max, CVVDP default and 10k nits, HDR-VDP-2, PU-PieAPP) as
`hdr_e_vs_field.py` reads them, plus SSIMULACRA2, butteraugli, FSIMc and HaarPSI computed now with `jnd-ladder-score`
on the stored PU-encoded 8-bit renderings of the same 380 pairs (`/mnt/v/output/zenmetrics/upiq-pu/png`, the June
2026 inputs of the stored butteraugli panel; the metrics in their standard form on PU-encoded pixels, not on linear
nits). Statistics: pooled and per-study SROCC (Narwaria, Korshunov), within-reference SROCC, paired bootstrap over
the 30 references, candidate minus metric and mix minus F3. Report-only; nothing is adopted on this read.

### R10 amendment 6 result (2026-11-04): HDR evidence panel for F3 and the mix

Run as registered: `hdr_panel_upiq_all.py` and `hdr_panel.py` with the fx2 declaration and `byc_predict` (`10d087cd`)
on the 2026-10-29 banks, compared by `e36_exp56/hdr_candidates_vs_field.py`; SSIMULACRA2, butteraugli, FSIMc and
HaarPSI scored with `jnd-ladder-score` on the 380 pairs of 8-bit PU renderings stored with the 2026-06-09
PU-integrated UPIQ study (`/mnt/v/output/zenmetrics/upiq-pu/png`, file names the md5 of each EXR path; 16 s under a
24G cap, peak RSS 8.2 GiB). Tables: `~/tmp/e36test/hdr-amend6/candidates_vs_field.md` on dev. E's numbers are the
2026-10-30 read.

| score | UPIQ pooled | Narwaria | Korshunov | within-reference |
|---|---:|---:|---:|---:|
| mix, seeds 0 / 1 / 2 | 0.783 / 0.786 / 0.785 | 0.884 / 0.879 / 0.876 | 0.945 / 0.944 / 0.944 | 0.954 / 0.956 / 0.956 |
| F3, seeds 0 / 1 / 2 | 0.775 / 0.775 / 0.773 | 0.883 / 0.882 / 0.883 | 0.945 / 0.945 / 0.945 | 0.955 / 0.954 / 0.955 |
| E (2026-10-30) | 0.792 | 0.900 | 0.947 | 0.958 |
| CVVDP 10k nits | 0.831 | 0.781 | 0.969 | 0.958 |
| CVVDP default | 0.758 | 0.664 | 0.965 | 0.954 |
| HDR-VDP-2 | 0.812 | 0.886 | 0.948 | 0.957 |
| PU-PieAPP | 0.875 | 0.778 | 0.925 | 0.922 |
| butteraugli 3-norm, stored HDR path (default / 10k nits) | 0.628 / 0.640 | 0.354 / 0.349 | 0.940 / 0.943 | 0.908 / 0.912 |
| butteraugli max, stored | 0.630 | 0.400 | 0.917 | 0.887 |
| PU-SSIM / PU-FSIM (stored) | 0.740 / 0.718 | 0.676 / 0.872 | 0.919 / 0.936 | 0.933 / 0.951 |
| SSIMULACRA2 on the PU renderings | 0.610 | 0.707 | 0.820 | 0.931 |
| butteraugli 3-norm / max on the PU renderings | 0.519 / 0.529 | 0.399 / 0.428 | 0.713 / 0.695 | 0.805 / 0.765 |
| FSIMc / HaarPSI on the PU renderings | 0.599 / 0.628 | 0.752 / 0.797 | 0.853 / 0.879 | 0.938 / 0.945 |

Paired bootstrap over the 30 references (2,000 draws). Mix minus F3 at the same seed: pooled +0.007 [-0.003, +0.023],
+0.011 [+0.002, +0.025], +0.011 [+0.002, +0.026]; Narwaria +0.001, -0.002, -0.006 (intervals about ±0.03);
Korshunov -0.001 [-0.004, +0.002] at every seed. Mix (seed 0) minus metric, pooled / Narwaria / Korshunov: CVVDP 10k
-0.047 [-0.098, +0.014] / +0.099 [-0.017, +0.265] / -0.024 [-0.044, -0.008]; HDR-VDP-2 -0.029 [-0.069, +0.006] /
0.000 / -0.004; butteraugli 3-norm (stored) +0.151 [+0.001, +0.309] / +0.492 [+0.169, +0.735] / +0.004 [-0.014,
+0.028]; SSIMULACRA2 (PU renderings) +0.165 [+0.031, +0.287] / +0.158 [+0.001, +0.304] / +0.120 [+0.047, +0.194];
FSIMc +0.176 / +0.118 / +0.088; HaarPSI +0.148 / +0.078 / +0.063 (F3 within 0.01 of the mix on every line).

Other populations: HDRTEACH VAL (3,900 rows, target HDR-VDP-3 Q_JOD, agreement with a model): pooled F3 0.898 / 0.908
/ 0.911, mix 0.907 / 0.917 / 0.910; within-reference 0.9999 against 1.0000; inversions 5-6 against 1 of 23,400; CVVDP
0.821 / 0.9821 / 602. 4Kdtb EVAL half (48 rows, MOS): pooled F3 0.795-0.798, mix 0.804-0.807, within-source 0.81-0.82
for both. Consumer-EVAL HDR chroma ladders (3,245 rungs, rank agreement with natural SSIMULACRA2): per-ladder mean F3
0.972-0.977, mix 0.975-0.980, paired deltas within ±0.005 with intervals through zero. UPIQ development rows (50): F3
0.824-0.831, mix 0.835-0.842.

Caveats. (1) The reference metrics on the 8-bit PU renderings read well below the stored HDR-path butteraugli on
Korshunov (0.713 against 0.940 for the 3-norm), so the PU-rendering numbers for SSIMULACRA2, FSIMc and HaarPSI are
lower bounds of what an HDR-aware input would give them; "natural" SSIMULACRA2 on linear nits/100 (the HDR-A3 probe's
convention) is not measured on UPIQ. (2) UPIQ's pooled SROCC mostly measures how the two studies' JOD scales line up;
the per-study columns are the fairer read. (3) HDRTEACH is agreement with HDR-VDP-3, not with people.

**Reading.** On HDR the mix is level with F3: UPIQ pooled +0.011 at two of three seeds with intervals just above zero,
per-study differences within ±0.006, HDRTEACH one inversion against five or six, 4Kdtb +0.009, ladders level. Both
candidates sit a little below E on UPIQ (pooled 0.775-0.786 against 0.792, Narwaria 0.88 against 0.90), level with
HDR-VDP-2, below CVVDP at 10k nits on Korshunov by 0.024 (interval excludes zero) and above it on Narwaria by 0.10
(interval touches zero), clearly above butteraugli's 3-norm on the stored HDR path (+0.15 pooled, +0.49 on Narwaria,
level on Korshunov) and above SSIMULACRA2 on the PU renderings (+0.17 pooled, +0.12 on Korshunov, +0.16 on
Narwaria). Nothing is adopted on this read.

**AIC2026 leg share (owner question, 2026-11-04).** The trainer samples a pair with probability proportional to its
group's `train_weight` (`zensim-train-core/src/lib.rs`), so a group's share of the draws is rows × weight over the
total. In the continuation step (60 epochs × 50,000 draws, `train.log` of `mixF3-s0`) the AIC2026 public_train leg is
3,920 pairs at ×256 = 48.3% of the draws (coverage 18.3%, chroma teacher 11.6%, BAPPS 10.7%, SafeSyn 6.9%, the human
MOS legs 1.5% each, MDID 0.7%, CID22 0.6%), about 370 passes over each AIC pair; the 120-epoch base fit has no AIC
leg, so over base fit plus continuation AIC2026 is 16.1% of all draws. At ×64 it would be 18.9% of the continuation's
draws, at ×16 5.5%. The weight was never ablated (method review, experiment 1).

*Addendum, SSIMULACRA2 on HDR done properly.* zenmetrics' `benchmarks/pu_integrated_upiq_2026-06-09.md` scored the
same 380 UPIQ pairs with PU21 integrated into SSIMULACRA2's perceptual encoding (fast-ssim2 `hdr-pu`, absolute
luminance in): pooled SROCC 0.704 (GPU port 0.704; the published PU-SSIM bar is 0.740), and it measured the "PU-encode
the input, leave the pipeline alone" shell at about 0.61, which is exactly what the PU-rendering column above
reproduces (0.610). So the fair SSIMULACRA2 number on UPIQ is 0.704 pooled, and the mix (0.783-0.786) is 0.08 above it
there; per-study figures for the integrated path were not recorded.

*Addendum (owner, 2026-11-04: "NO to pu21. Use actual hdr path for butter/ssim2 as currently implemented").* The
PU-rendering rows above are withdrawn as the field comparison. The metrics' own HDR paths, as implemented today:
butteraugli's linear entry on display-relative planes (`nits / peak` clamped to [0, 1], `intensity_target = peak`; the
feeding zenmetrics-api's `HdrFeeding::LinearPlanes` uses) and SSIMULACRA2's native HDR path (fast-ssim2 0.9.1, which
deprecates PU21 upstream: linear f32 BT.709 with a diffuse-white anchor of 1.0, so the samples are nits).
`jnd-ladder-score --hdr-nits` (zenmetrics-private `1832b895`) scored the 380 UPIQ EXR pairs through both (28 s, peak
RSS 8.4 GiB; `~/tmp/e36test/refmetrics/scores/upiq-exr.tsv`, comparison `hdr-amend6/candidates_vs_field_native.md`).
The butteraugli columns reproduce the stored June panels exactly at both peaks, which verifies the feeding.

| score | UPIQ pooled | Narwaria | Korshunov | within-reference |
|---|---:|---:|---:|---:|
| mix, seed 0 (seeds 1-2 within 0.008) | 0.783 | 0.884 | 0.945 | 0.954 |
| F3, seed 0 | 0.775 | 0.883 | 0.945 | 0.955 |
| SSIMULACRA2, native HDR path | 0.765 | 0.662 | 0.962 | 0.955 |
| butteraugli 3-norm, linear path, peak 1,000 cd/m² (= stored panel) | 0.628 | 0.354 | 0.940 | 0.908 |
| butteraugli 3-norm, linear path, peak 10,000 (= stored 10k panel) | 0.640 | 0.349 | 0.943 | 0.912 |
| butteraugli max, linear path, peak 1,000 / 10,000 | 0.630 / 0.638 | 0.400 / 0.380 | 0.917 / 0.920 | 0.887 / 0.878 |

Paired over the 30 references, mix (seed 0) minus the metric: SSIMULACRA2 native +0.016 pooled [-0.080, +0.163],
+0.199 Narwaria [-0.021, +0.548], -0.017 Korshunov [-0.038, +0.004]; butteraugli 3-norm (peak 1,000) +0.151 [+0.001,
+0.309], +0.492 [+0.169, +0.735], +0.004 [-0.014, +0.028]; F3 within 0.01 of the mix on every line. Reading: against
SSIMULACRA2 on its own HDR path the candidates are level pooled, ahead on Narwaria (ten references; the interval
reaches zero) and 0.017 behind on Korshunov (interval touches zero); against butteraugli they are clearly ahead pooled
and on Narwaria and level on Korshunov, where every score ties near 0.94-0.96. Note for zenmetrics: its `hdr_feeding`
table still routes SSIMULACRA2 through the PU21 path that fast-ssim2 0.9.1 deprecates; the native path should replace
it there.

### Owner decision (2026-11-04): the mix on F3's recipe is adopted

Owner, 2026-11-04: "let's adopt the mix and make sure it is fully documented so another person could recreate it
from scratch" (after R10 and its amendments 1-6, R4, and the reference-metric and HDR comparisons above). `mixF3`,
the TID2013 + MDID mix on F3's recipe without the hinge (seeds 0-2; bakes `pack_e059/refit/production-f16.bin`),
replaces F3's control as the program's model. Marks accepted with it: G-STEER's broad-220-b8 case at every seed (8-px
map rank 0.69 against the 0.70 bar on one text screenshot; the JPEG XL loop shows no byte cost), the near-tie texture
aversion on CLIC 2021 (0.3-0.6 points overall, returned at moderate differences) and the 0.04 it gives back to F3 on
CID:IQ's SGCK gamut mapping. Not adopted with it: p30L, the map term, the CLIC 2024 leg, any Haar-taught leg. AIC2026
public_test stays sealed and is never folded into training (owner, 2026-11-04). Follow-ups: the from-scratch recipe
in zensim-training (`recipes/MixF3.json`, `docs/RECIPE_MIXF3.md`, replayed in the release sandbox) and a zensim
profile for it (pull request).

*Shipping (2026-11-04):* the zensim profile pull request is https://github.com/imazen/zensim/pull/2 (`ZensimProfile::F`, `zensim-f`, returned by
`latest()`; stacked on `slim-0.3.0`; 40 of 40 tests and 3,927 of 3,927 production parity rows on i265).

### HDR front-end experiment (2026-11-04, registered before any extraction): anchored cube-root against PU21 for F, F3 and E

Owner, 2026-11-04: "let's figure out if we can do better than pu21. ssim2 did, and we do similar approaches, so let's
give f a fair shake." fast-ssim2 0.9.1's native HDR path keeps the cube-root XYB opsin and scales it by an
absolute-light anchor (UPIQ 0.765 pooled, against 0.704 for its deprecated PU21 path). zensim's HDR front end is PU21
per channel divided by PU21(100 cd/m²), then the opponent axes (`color::linear_to_pu_xyb_planar_into`).

**Arms (front ends), on the same decoded nits:** PU21 (the current path; the stored `upiq-r2-bank`), and the anchored
cube-root front end, nits divided by a unit luminance and then the unclamped cube-root XYB
(`color::linear_to_positive_xyb_planar_into_unclamped`, values above 1.0 retained), at units of 100 cd/m² (zensim's
PU normalisation point), 203 (BT.2408 diffuse white), 255 (the libjxl and fast-ssim2 default for linear input) and
1,000 (the HDR reference peak). Implementation: `HdrEncoding::LinearCubeRoot { unit_nits }` in the research crate,
chosen in `upiq_pu_score`'s allowlisted extraction by `--front-end cube-root:<nits>`; the extraction manifest records
the front end. **Models:** F (`mixF3`, seeds 0-2), F3 (seeds 0-2) and E (the 2026-10-30 read's bake), forwarded by
`byc_predict` with their declarations on each bank; no fit, no selection. **Populations, in order:** UPIQ-380 (380
pairs, 30 references, JOD; T2 TRAIN rows no candidate trained on, read again report-only as in amendment 6); then, only
if the UPIQ read favours an anchored arm under the rule below, the 4Kdtb EVAL half, HDRTEACH VAL and the chroma ladders
through the fleet extractor with the same front end (a later registration names that build). **Statistics:**
amendment 6's: pooled and per-study SROCC, within-reference SROCC, and the paired bootstrap over the 30 references of
each arm minus PU21 for the same model. **Reading rule:** an anchored front end counts as better only if, for F at
every seed, its pooled and both per-study differences against PU21 are non-negative and the pooled interval lies above
zero; otherwise PU21 stands. Nothing ships from this read; a front-end change is a revision-8 decision for the owner.

### HDR front-end experiment result (2026-11-04): PU21 stands; the anchored cube root trades Narwaria for Korshunov

Run as registered. **Extraction:** `upiq_pu_score --training-allowlist` (zensimx `60223363`, binary sha256
`aa23606d…`, built on i270, run on dev, `ZENSIM_FORMULA_REV=7`, five arms in parallel at one thread each, 50-53 s and
1.2 GB peak per arm). The PU21 control arm's `features.tsv` is **byte-identical** to the stored 2026-10-29 fleet
extraction (sha256 `668f726d…`), so the refactor left the shipped front end untouched and dev reproduces the fleet
build. Banks assembled with `hdr_upiq_bank.py --tsv` (schema `hdr-native-rev7-bank-v1`, `hdr_front_end` recorded),
scored with `hdr_panel_upiq_all.py --upiq-bank` and `byc_predict` (`10d087cd`), compared with `hdr_frontend_compare.py`
(paired bootstrap over the 30 references, 2,000 draws). The PU21 rows reproduce amendment 6 exactly. Tables and
per-arm manifests: `~/tmp/e36test/hdr-fe/` on dev (`compare/frontend_compare.md`, `arm-*/`). Models: F = `mixF3`
seeds 0-2, F3 seeds 0-2, E (the 2026-10-30 read's bake); all trained through the PU21 front end on SDR data, so every
arm is a zero-shot front-end swap.

| front end | F (seeds 0/1/2): pooled | Narwaria | Korshunov | within-ref | F3 (s0): pooled / Narw. / Korsh. | E: pooled / Narw. / Korsh. |
|---|---:|---:|---:|---:|---:|---:|
| PU21 (shipped) | 0.783 / 0.786 / 0.785 | 0.884 / 0.879 / 0.876 | 0.945 / 0.944 / 0.944 | 0.954-0.956 | 0.775 / 0.883 / 0.945 | 0.792 / 0.900 / 0.947 |
| cube root, 100 cd/m² ↦ 1 | 0.809 / 0.810 / 0.809 | 0.838 / 0.841 / 0.842 | 0.956 / 0.956 / 0.955 | 0.961 | 0.814 / 0.829 / 0.949 | 0.830 / 0.875 / 0.952 |
| cube root, 203 | 0.811 / 0.812 / 0.811 | 0.823 / 0.828 / 0.828 | 0.958 / 0.957 / 0.956 | 0.962-0.963 | 0.813 / 0.808 / 0.952 | 0.830 / 0.852 / 0.954 |
| cube root, 255 | 0.811 / 0.812 / 0.813 | 0.814 / 0.822 / 0.820 | 0.958 / 0.957 / 0.957 | 0.962-0.963 | 0.812 / 0.796 / 0.953 | 0.830 / 0.841 / 0.955 |
| cube root, 1000 | 0.809 / 0.811 / 0.811 | 0.731 / 0.740 / 0.738 | 0.959 / 0.958 / 0.958 | 0.962-0.964 | 0.807 / 0.717 / 0.954 | 0.826 / 0.758 / 0.956 |

Paired bootstrap, cube root minus PU21 for F (seed 0; seeds 1-2 within 0.003): at 100 cd/m² pooled +0.025 [-0.024,
+0.074], Narwaria -0.040 [-0.119, +0.052], Korshunov +0.012 [-0.001, +0.025], within-reference +0.006 [-0.004,
+0.016]; at 203: +0.028 / -0.053 / +0.013 [+0.000, +0.028] / +0.008; at 255: +0.028 / -0.061 / +0.013 [+0.000,
+0.028] / +0.008; at 1,000: +0.026 [-0.058, +0.101] / -0.136 [-0.294, +0.024] / +0.014 [+0.000, +0.030] / +0.010.
F3 and E move the same way (pooled +0.03 to +0.04, Narwaria -0.02 to -0.15, Korshunov +0.004 to +0.010).

**Reading (registered rule).** No anchored arm has non-negative deltas on both studies for F: Korshunov and the
within-reference order improve a little at every unit (+0.011 to +0.014, intervals touching zero), Narwaria worsens
at every unit and progressively with the unit (-0.04 at 100 cd/m², -0.14 at 1,000; intervals through zero below
1,000). The pooled gain (+0.025) is mostly the two studies' JOD scales lining up differently, the caveat amendment 6
already noted. **PU21 stands**; nothing ships from this read. What the sweep says: the unit that keeps the shipped
model's dark-end behaviour closest (100 cd/m², PU21's normalisation point) loses least on Narwaria, and a large unit
pushes HDR content into the SDR model's dark regime. The anchored cube root with the same 255 cd/m² convention as
SSIMULACRA2's native path puts F at 0.811 / 0.814 / 0.958 against SSIMULACRA2's own 0.765 / 0.662 / 0.962 (amendment 6
addendum), so under one convention zensim is far ahead on Narwaria and level on Korshunov. Options for the owner (a
revision-8 decision): keep PU21 (the default), or register a fit through the cube-root front end at 100 cd/m² (this read
only swapped the front end under PU21-trained weights; a fit would say whether the Narwaria loss is the front end or the
training), or a front end that is PU21-like below about 100 cd/m² and cube-root above (a design, not a measurement).

### R11 (2026-11-04, registered before the run): a codec-only continuation of the mix (`mixF3codec`)

Owner, 2026-11-04: "What about a codec-optimized mix variant? Did we try that already?" Not yet: R10 ran only the
mix's general continuation (`mixF3`). **Arm:** the mix's base fit (`tid2013_mdid-s{0,1,2}`, the R3 step of the mix
recipe) continued exactly as F3codec and ECodec were: the AIC-2026 public_train table alone, `--aic-weight 16.0
--diagnostic-population aic-only --train-only`, 60 epochs at lr 1e-5, F3codec's trainer build and source
(`fx2-full-codec-e29`), the checkpoint after epoch 29 packed with the codec pack program (no input pruning), seeds 0-2;
runs under `~/tmp/e36test/r11/` on dev. **Reads (report-only; nothing ships from them):** the frozen held-out panel
(`dataset_eval.py mix-eval`, the 11 EVAL collections and the 2AFC sets) beside ECodec, F3codec and the mix; the E36
selection-source inversion share (`e36_exp56/primary.py`, the 40 selection sources, per stratum) beside the same three.
The JPEG XL loop is not read here. **What would make it a candidate:** a codec-stratum inversion share no worse than
F3codec's at every seed and held-out MOS no worse than ECodec's on the codec-heavy collections (LIVE-MD, VCL@FER,
IRCCyN/IVC, NITS-IQA) within the intervals. Shipping would additionally need the R6 label-free gates (dial contract
C1-C6, G-STEER, identity at the SIMD permutations) and an owner decision.

**R11 amendment (owner, 2026-11-04: "to be clear this is a continuation of the adopted new F, yes?").** Arm A above is
F3codec's recipe on F's *base fit* (the relationship ECodec has to E, F3codec to F3). The literal reading is **arm B**:
the adopted F itself (the mix's general continuation, `refit/last.bin` = the checkpoint after epoch 59, seeds 0-2)
continued on the AIC-2026 public_train table alone with the same flags (`--aic-weight 16.0 --diagnostic-population
aic-only --train-only`, 60 epochs at lr 1e-5, F3codec's build and source), epoch 29 packed the same way; runs under
`~/tmp/e36test/r11/runs/mixF3codecB-s{0,1,2}`. Both arms get the reads registered above, side by side.

### R11 result (2026-11-04): the codec-only continuations of the mix

Run as registered (`~/tmp/e36test/r11/run_mixF3codec{,B}.sh`, F3codec's build and source, 60 epochs at lr 1e-5, AIC
public_train ×16, epoch 29 packed with the codec pack program; seeds 0-2 each, 1,146 s and 1,132 s wall for the three
seeds in parallel). Packed bakes: arm A `66492cab…`, `9649c36b…`, `805a2c10…`; arm B `b82f1a47…`, `8d776f0c…`,
`514b2cba…` (kept under `/mnt/v/output/zensim/e36-subset-test-2026-11-01/bakes/r11-*`). Reads by `~/tmp/e36test/postR11.sh`:
E36 scored into `scores-R11`/`inversions-R11`, `primary.py` with F3codec as the control; the EVAL study gained ECodec,
F3codec and both arms (seeds 0-2) and `readout.py` / `ceval.py` ran over all five models. Tables: `~/tmp/e36test/postR11/`.

**E36 selection-source inversion share** (646,677 consensus pairs on the 40 selection sources, mean of seeds; lower is
better; paired bootstrap over sources, arm minus F3codec):

| stratum | F3codec | ECodec | F (mix, general) | arm A `mixF3codec` | arm B `mixF3codecB` |
|---|---:|---:|---:|---:|---:|
| all | 4.50% | 6.65% | **3.40%** | 4.38% (-0.12 pp [-0.20, -0.04]) | 4.60% (+0.11 pp [+0.04, +0.17]) |
| codec = jpeg / jxl / svt / webp | 5.20 / 4.39 / 2.69 / 5.35 | 7.08 / 6.29 / 4.59 / 8.21 | 3.73 / 3.00 / 2.14 / 4.42 | 5.01 / 4.13 / 2.70 / 5.32 | 5.26 / 4.41 / 2.86 / 5.51 |
| content = lineart / mixed / photo / screen | 6.80 / 4.75 / 2.68 / 4.14 | 9.97 / 7.31 / 3.89 / 5.99 | 4.22 / 3.64 / 2.32 / 3.59 | 6.59 / 4.50 / 2.64 / 4.15 | 6.85 / 4.85 / 2.75 / 4.34 |
| size ≤64 / 80-256 / 320-1024 | 8.74 / 3.40 / 1.66 | 11.90 / 5.36 / 3.02 | 6.78 / 2.55 / 1.09 | 8.58 / 3.31 / 1.54 | 8.86 / 3.53 / 1.70 |
| quality 0-20 / 22-60 / 62-100 | 5.15 / 4.73 / 3.41 | 6.62 / 6.87 / 6.32 | 4.17 / 3.59 / 2.24 | 5.15 / 4.62 / 3.14 | 5.32 / 4.87 / 3.38 |

Seeds: F 3.39 / 3.36 / 3.44%; arm A 4.31 / 4.41 / 4.42%; arm B 4.57 / 4.60 / 4.64%; F3codec 4.47 / 4.47 / 4.56%.

**Held-out panel** (`readout.py`: median absolute residual around each collection's logistic fit; `ceval.py`: row-weighted
SROCC and mean error on the 417 compression rows of six collections and the 871 non-codec rows of seven):

| model (seeds 0-2) | held-out 3 median | all-9 median | common-3 SROCC | compression SROCC / error | non-codec SROCC / error |
|---|---:|---:|---:|---:|---:|
| F (mix, general) | 8.27-8.40% | 7.94-8.09% | 0.848-0.850 | 0.894-0.895 / 6.95-7.13% | 0.786-0.790 / 8.45-8.66% |
| arm A `mixF3codec` | 8.63-8.68% | 8.73-8.83% | 0.853-0.854 | **0.897-0.898 / 6.76-6.83%** | 0.787-0.788 / 9.34-9.45% |
| arm B `mixF3codecB` | 8.70-8.78% | 8.67-8.76% | 0.851-0.852 | 0.895-0.897 / 6.90-6.95% | 0.785-0.787 / 9.24-9.35% |
| F3codec | 9.10-9.20% | 9.13-9.18% | 0.838-0.843 | 0.893-0.895 / 6.90-7.03% | 0.761-0.768 / 9.86-9.99% |
| ECodec | 9.24-10.02% | 9.91-10.32% | 0.812-0.819 | 0.888-0.890 / 7.04-7.30% | 0.726-0.738 / 9.98-10.91% |

Per collection on the compression rows (seed 0, arm A against F): IRCCyN/IVC 0.932 against 0.912, NITS-IQA 0.862 against
0.853, Colourlab 0.810 against 0.801, VCL@FER 0.937 against 0.950, CID:IQ and IQA-ODI level. On the non-codec rows arm A
loses Colourlab (0.603 against 0.714) and gains CID:OA (0.925 against 0.911), LIVE-MD (0.922 against 0.903) and NITS-IQA
(0.723 against 0.672).

**Reading.** Against the registered bar, arm A qualifies as a candidate (every seed below F3codec's inversion share;
compression rows ahead of ECodec on every codec-heavy collection) and arm B does not (above F3codec at every seed). But
the bar was set against the codec models, and the adopted general F beats both arms and both codec models on the E36
ladders in every stratum (3.40% against 4.38% and 4.60%; every codec, content, size and quality band) and on held-out
error (8.3% against 8.7%). What the AIC-only continuation buys is +0.003 row-weighted SROCC and -0.2 points of error on
the 417 compression rows, mostly IRCCyN/IVC and NITS-IQA, at the cost of a point of inversion share on the encoder
ladders, a 0.4-point error rise on the held-out collections and Colourlab's non-codec rows. Continuing the adopted F
itself (arm B) is worse than continuing its base (arm A) on every read. **Recommendation:** no codec variant of the mix
ships now; F covers codec output better than any codec-only continuation on the ladders the product optimises. If the
owner wants a codec profile regardless, arm A is the one, and it would still need the R6 label-free gates.

### Owner decision (2026-11-04): PU21 is not the HDR front end; the HDR front-end rule is set aside

After the Ultra HDR probe's addendum (two thirds of zensim's SDR-to-HDR gap on the camera files is the PU21 path's own
severity on identical pixels, growing with the assumed luminance), the owner: "That's why I said to not use PU21. Ignore the
gate on that until we sort this out - let's use a proper model that makes continuity better." Recorded consequences:

- The reading rule of the "HDR front-end experiment" (PU21 stands unless an anchored arm is non-negative on both UPIQ
  studies) is **set aside**. It compared zero-shot front-end swaps under PU21-trained weights, which is not the question.
- **Continuity is the requirement**: an SDR image fed through the HDR path at its reference white must score as it does
  through the SDR path, and HDR light above white must extend that scale, not a different one. The anchored cube root
  (`HdrEncoding::LinearCubeRoot`, nits over the SDR white luminance, unclamped) has this by construction: at a unit of
  1 cd/m² on light in [0, 1] it equals the SDR gamut-preserving path bit for bit (`linear_cube_root_front_end_is_the_sdr_cube_root_on_the_same_light`),
  and the unit only sets which luminance is "white".
- **Next: a model fit through that front end** rather than another zero-shot read; the registration follows below once
  the HDR training path's inputs are re-extracted through it. The shipped profiles' PU21 HDR entry (`compute_hdr`) stays
  as is until that model exists; its scores are a separate regime from the SDR scores and must be labelled so.

### HC1 (2026-11-04, registered before any re-extraction or fit): the continuity front end and a model through it

Follows the owner decision above. **Front end:** `HdrFrontEnd::CubeRoot { unit_nits }`, the SDR cube-root opsin on
nits over the SDR white luminance, unclamped, now a parameter orthogonal to the transfer (`HdrEncoding::Pq`/`Hlg`/`Linear`
decode to cd/m² as before, then the front end applies), so PQ-coded ladders and teacher frames take it too. **Unit:**
203 cd/m² (BT.2408 reference white; the Ultra HDR decode convention, where an SDR base displayed in HDR is exactly 203
cd/m² white, so the continuity property lands where the product meets HDR content) is the fitting arm; 100 cd/m² (the
PU21 normalisation point, the zero-shot read's least-bad anchor) is extracted beside it as a reserve, not fit unless the
owner asks. Step 1, extraction (label-free): the five 2026-10-29 plans (`fhdr-plans/{ladders-r1, teacher-train-r1,
teacher-val-r1, 4kdtb-r1}` and UPIQ-380 through `upiq_pu_score`) re-extracted on i270 with
`scripts/byconstruction/hdr_frontend_reextract.py` driving `hdr944_extract --front-end cube-root:<nits>` (one process
per plan view, the fleet's shard layout, `hdr_native_bank.py` unchanged for assembly; the new plan directories carry
the new producer pins and the front end in every cell). **Controls before any bank is used:** a `--front-end pu21`
re-run of three views per plan must reproduce the stored 2026-10-29 bank rows bit for bit (the refactor is inert for
PU21; the UPIQ PU21 control already reproduced its stored extraction byte for byte), and every cube-root bank's row
count, keys and membership map must equal the PU21 bank's. Step 2 (registered separately once the banks exist): the
mix's recipe with the four HDR legs through the cube-root banks (P3 on the mix: the fchroma fx2 root plus
`hdr_upiq380`, `hdr_4kdtb_train`, `hdr_ladder_chroma`, `hdr_teacher_hdrvdp3`, masses and dev-fold scale rule as in
`HDR_P2P3_EXECUTION_2026-10-29.md`), three seeds, the AIC ×256 continuation with rehearsal, epoch 59; reads: the HDR
development rows (UPIQ's 50, HDRTEACH VAL, 4Kdtb EVAL, the ladders' EVAL rungs), the SDR no-regression set (held-out
panel, E36 inversion share, G-STEER, dial contract), the continuity check (SDR images through `compute_hdr` at 203
cd/m² must equal their SDR scores), and the Ultra HDR probe's SDR-as-HDR arm, whose gap must be zero at the unit.

**Amendment (2026-11-04, owner, before any extraction ran):** the extraction stays on dev (the owner's call: dev's
remit is zensim training and evaluation, and this extraction feeds the fit; the justfile's `hdr-native-build` guard
against dev dates from the fleet era and is bypassed by running cargo directly), only the 203 cd/m² unit is extracted
(the 100 cd/m² reserve arm is dropped), and a profiling and optimisation pass on the extractor precedes the run so that
the wall time is spent once, on a measured binary (14,814 pairs: ladders 10,422 at 256², teacher train+val 10,304 at
1200×1600 PQ16 PNG reference with JXL distortions, 4Kdtb 88 at 4K Radiance, all read from tower over NFS). The pass may
change only how the work is scheduled, cached and read (the reference is decoded once per candidate today; process and
thread counts; I/O staging), never a numeric operation or its order: its gate is the PU21 control above (three views
per plan bit-identical to the stored banks) plus byte-identical cube-root outputs before and after on the same views;
its record is `benchmarks/hc1_extractor_profile_2026-11-04.md`. Build note: the standalone bench root could not resolve
the live zenjpeg sibling (crates.io version edges on the zencodec 0.2 family since zenjpeg d099267c), so zensim-bench
pins zenjpeg at 4933461f, the revision its lockfile already resolved; `hdr944_extract` does not use zenjpeg.

**Step 1 result (2026-11-04, dev; root `/mnt/v/output/zensim/byc-hdr-20261104/`, mirrored to
`/mnt/tower/output/zensim/byc-hdr-20261104/`).** Build: zensimx main `94095e06`; `hdr944_extract` sha256
`b61ae3f50e903f52…`, `upiq_pu_score` `ea4298a3ee3fca22…`. The profiling pass was dropped before it began: three baseline
views (one process, one thread) took 0.8 s (ladders, 14 pairs at 256²), 4.0 s (teacher, 13 pairs at 1200×1600) and
9.5 s (4Kdtb, 11 pairs at 4K), which projects the whole extraction at about five minutes, and the run took 437 s
including its rebuild (4Kdtb 25 s, ladders 109 s, teacher-train 134 s, teacher-val 95 s at 28 processes, peak RSS
12.55 GiB; UPIQ 2:02 for the PU21 arm on cold NFS, 0:28 for the cube-root arm). **Controls, all passed:** `--front-end
pu21` on views 0,1,2 of each plan against the stored 2026-10-29 banks (`hdr_control_compare.py`: 4Kdtb 33 rows, ladders
48, teacher-train 38, teacher-val 39, 0 mismatches; `control-<plan>.json`); the UPIQ PU21 arm byte-identical to the
stored 2026-10-29 `features.tsv` (`668f726d…`); the UPIQ cube-root:203 arm byte-identical to the front-end experiment's
arm built from `60223363` (`73209d64…`), so the `HdrFrontEnd` refactor changed no value on either front end. **Banks
(`cube-root:203`, `build_commit 94095e06`, `hdr_front_end` in every manifest, `keys_sha256` equal to the PU21 banks'):**
4Kdtb-r1 96 rows (features `9c9bd0d1…`), ladders-r1 10,422 (`992e442b…`), teacher-train-r1 7,390 (`742fd8f3…`),
teacher-val-r1 3,900 (`ce52c268…`), UPIQ-380 (`upiq/arm-cube-root-203/bank`). The 100 cd/m² reserve was not extracted.

**Owner decisions (2026-11-04, after step 1):** the public release names the adopted model **A** (research F → public
A; public profile letters restart with 0.3.0, so public names and research names are separate namespaces and the
mapping is written in the public crate's `docs/PROFILES.md`); **no codec variant** ships (R11 stands); work on this
record stops when the session's token budget reaches 36 percent used, with mechanical lanes on cheaper models.

**Step 2 registration (2026-11-04, before any fit, dev-fold score or checkpoint selection).** *Root:* the adopted mix's
base root `/var/tmp/byc-fchroma-20261029/r3/w100/root` (frozen `14f53dac…`; the MixF3 recipe: `--human-sources tid2013
--exclude-datasets pipal,live-md,cid-iq,cid-oa,irccyn-ivc,colourlab-rating-variability-2022 --mos-share 36.035`, spec
`sel:45acffa603e8@h32:H128:cv16:cf98:fx2:mono:thr:mp32`, the 492 columns of the P3 study) plus the four HDR legs from the
cube-root:203 banks through `hdr_root.py --front-end cube-root:203`, base masses as P3 (UPIQ 0.5, 4Kdtb 0.5, ladders 1.0,
teacher 0.5) times the scale. The producer set (`subjective_training_leg.HDR_PRODUCERS`, mirrored in `hdr_admission.rs`)
is keyed by front end and both admissions refuse a declaration and a bank of different front ends, so PU21 and cube-root
legs cannot mix; the 2026-10-29 artefacts, which predate the field, mean `pu21`. *Scripts:* zensimx main, with the
experiment-5 options ported byte for byte from the released `src-exp5` snapshot. *Trainer:* `zensim_mlp_train` built
from main at the commit recorded in the study (NEWBIN). *Control first:* NEWBIN retrains the mix's seed-0 base fit on the
parent root (no HDR legs); its `refit/last.bin` stripped of `zentrain.repro` must equal the recipe's `6ce7a42e37b72904…`;
a failure is recorded and nothing is compared. *Scale:* TRAIN-side dev folds exactly as P3 (s ∈ {0.5, 1.0}, folds 0 and
1, seed 0, `hdr_devfold_score.py`, larger statistic wins, tie to the smaller scale; EVAL, VAL and the 50 UPIQ development
rows never select). *Then:* three full seeds at the chosen scale (120 epochs, `--train-only --pack-production
--e33-output-stage`), the AIC ×256 continuation with rehearsal from each seed's `refit/last.bin` (60 epochs, lr 1e-5,
`--aic-weight 256.0 --diagnostic-population mixed-aic256-xref`), pack `ckpt_epoch059` with the CID22 fit table anchor,
strip. *Reads* as registered above; their label exposure is registered in `docs/DATA_SPLITS.md` before any read.
*Wall:* about 30 min per fit (measured on the mix: base 120 epochs 30 min, continuation 60 epochs 25 min, one thread);
control and the four dev fits run together, then the three seeds, then the three continuations: about 1.75 h on dev. No
trainer profiling pass: the trainer's numerics are pinned by the control, and a faster build would have to reproduce the
same bytes.

**Step 2, phase A result (2026-11-04, 10:10–10:40Z on dev; `/var/tmp/byc-hdr-20261104/`).** Trainer NEWBIN
`zensim_mlp_train` sha256 `57a8d54b…` built from zensimx `b4dde3b9` (the producer-set commit); support binaries the
r10 `bin-map` set (`bin/BIN_SHA256SUMS`). **Control passed:** the mix's seed-0 base fit retrained on the parent root gave
stripped `refit/last.bin` `6ce7a42e37b72904…`, the recipe's value, so main's trainer and scripts (with the experiment-5
options ported) reproduce the adopted model byte for byte. Derived roots `roots/mix-cr203-{s05,s10}{,-f0,-f1}`
(`hdr_root.py --front-end cube-root:203`, legs admitted by the Python and Rust admissions under the new producer set).
**Dev-fold rule (`hdr_select.py`, `SELECTION_hc1.json`):** s05 statistic 0.9719 (ladders 0.9551, teacher 1.0000, UPIQ
0.9606; 4Kdtb report-only, 0.9137 / 0.9006 at s05 / s10 fold 0), s10 0.9713 (0.9530 / 1.0000 / 0.9608): **scale 0.5
chosen**, as P3 chose on the PU21 banks. Five single-thread fits took 28–31 min each in parallel (whole phase 1,878 s,
peak RSS 1.97 GiB). Phase B (three seeds at s05, then the continuations, pack, strip) started 10:40:45Z.

**Step 2, phase B result (2026-11-04, 10:40–11:21Z on dev).** Three full seeds on `roots/mix-cr203-s05` (28–29 min
each, in parallel), the AIC ×256 continuation with rehearsal from each `refit/last.bin` (60 epochs, 12 min each; the
HDR legs load as 330 / 48 / 7,177 / 7,390 rows at 1,853 columns and rehearse at their s05 weights), pack of
`ckpt_epoch059` with the CID22 fit anchor (e33 output stage), strip. **Bakes (`/var/tmp/byc-hdr-20261104/cont-s05/
cont_s{0,1,2}/pack_e059/refit/`):** `production-f16.bin` `18f37bea…` / `a53446e8…` / `eee01349…`, `shipped.bin`
`f065d81e…` (24,444 B) / `4f915e08…` (23,212 B) / `3174aaee…` (24,398 B). Whole phase 2,448 s, peak RSS 1.99 GiB. No label
of any read set has been read yet; the registered reads follow (research name of these bakes: `mixHC1` seeds 0–2).

**Continuity read (2026-11-04; `zensim-bench/examples/hdr_continuity.rs`, result `benchmarks/hc1_continuity_2026-11-04.tsv`).**
`BakeScorer` gained `with_hdr_front_end` (default `Pu21`; HDR steering and attribution refuse any other front end). Six
CLIC 2025 1024² PNGs from codec-corpus, four pairs each (identity, box blur r=1 and r=2, ±8 noise), scored by the
`mixHC1` seed-0 and the mix F seed-0 bakes three ways: the SDR path, `compute_hdr` on linear nits = sRGB EOTF × 203 cd/m²
through the cube-root front end at 203, and through PU21. **Cube root: |HDR − SDR| mean 0.0001, max 0.0003 for both bakes
(24 pairs each)**; the residual is the production SDR path's clamped sRGB kernel against the unclamped HDR opsin. **PU21:
mean |gap| 4.74 (mixHC1) and 4.51 (F), max 33.4 and 31.1** (the ±8-noise pair of one image), the HDR reading always lower.
Identity pairs score 100 on every path. The property is the front end's, not the model's: F through the cube root is as
continuous as mixHC1; what the HDR legs buy is calibration on HDR content, which the HDR panel reads measure.

**HDR evidence panel read (2026-11-04, 11:28–12:00Z; `hdr_panel.py --banks`, `hdr_panel_upiq_all.py`; outputs
`~/tmp/hc1-reads/hdr-panel-{cr203,pu21}/`, `hdr-upiq-cr203/`).** Every model read on both bank sets, the 2026-10-29
PU21 banks and the cube-root:203 banks (pooled SROCC; ladder inversions in brackets):

| population | F s0/s1/s2 on PU21 | F s0/s1/s2 on cube root | E PU21 → cube root | mixHC1 s0/s1/s2 on cube root |
|---|---|---|---|---|
| UPIQ 50 development rows | 0.835 / 0.842 / 0.841 | 0.943 / 0.941 / 0.939 | 0.839 → 0.943 | 0.924 / 0.923 / 0.926 |
| HDRTEACH VAL (3,900; HDR-VDP-3 teacher) | 0.908 / 0.917 / 0.910 | 0.890 / 0.896 / 0.889 | 0.941 → 0.914 | 0.878 / 0.857 / 0.898 |
| 4Kdtb EVAL (48, MOS) | 0.804 / 0.807 / 0.804 | 0.805 / 0.806 / 0.805 | 0.808 → 0.813 | 0.817 / 0.811 / 0.805 |
| Ladders EVAL (3,245 rungs, natural SSIMULACRA2) | 0.774 / 0.771 / 0.773 [60/51/59] | 0.786 / 0.785 / 0.787 [50/51/49] | 0.757 → 0.776 [114 → 103] | 0.780 / 0.783 / 0.781 [108/105/108] |
| UPIQ all 380 (mixHC1 trained on 330) | 0.783 / 0.786 / 0.785 (R10 am. 6) | 0.811 / 0.812 / 0.811 | 0.830 (cube root) | 0.820 / 0.820 / 0.823 |

**Reading.** (1) The front end alone moves the mix: through the cube root, zero-shot, F gains +0.10 on UPIQ's human
development rows and +0.013 on the ladders with fewer inversions, is level on 4Kdtb MOS, and loses 0.02 on HDRTEACH VAL,
whose labels are the HDR-VDP-3 teacher; E moves the same way. (2) The HC1 retrain buys nothing held out: on the
cube-root banks mixHC1 is below F on UPIQ's development rows (0.925 vs 0.942), level or below on HDRTEACH VAL with a
wide seed spread, +0.01 on 4Kdtb's 48 rows, and doubles the ladder inversions (108 vs 50); its only lead is on the 330
UPIQ rows it trained on. (3) Continuity is the front end's property (above). **Recommendation:** ship the cube-root
front end at 203 cd/m² as the public HDR path for A (the mix); do not ship the HC1 retrain; PU21 stays research-only.
The SDR no-regression reads of mixHC1 continue for the record.

**SDR no-regression reads of mixHC1 (2026-11-04, 12:00–12:07Z; `~/tmp/hc1-reads/`, the mix's R10 commands).** EVAL
held-out MOS sets: median error 8.16 / 8.28 / 8.31 % (seeds 2/0/1) against F's 8.27 / 8.39 / 8.40 %, common-3 SROCC
0.853–0.855 against 0.848–0.850; compression rows (417): row-weighted SROCC 0.898 / 0.898 / 0.898 against 0.895 /
0.894 / 0.894. **E36 selection-source inversion share: 5.01 % (4.90 / 5.00 / 5.13) against F's 3.40 % (3.39 / 3.36 /
3.44), +1.61 pp [95 % CI +1.23, +2.02] — a regression on the program's primary statistic.** G-CHROMA-L: rank passes,
**level fails** at every seed (ycc444 offset 4.0–4.3 vs F's 2.76, ycc420 6.0–6.3 vs 3.32, xyb-extended 4.9–5.3 vs 3.58).
G-STEER 132 / 134 / 131 of 135 (F 133 / 134 / 134). Dial contract C1–C6 PASS on both grids; identity: 62 sources pass at
every SIMD tier; NEARID N1–N3 pass (137 / 137 / 138 of 144 monotone ladders). **Verdict: the HC1 retrain is not
shippable (E36 share +1.6 pp, G-CHROMA-L level fail, no held-out HDR gain); the cube-root front end is adopted on its
own merits (continuity by construction; the mix gains +0.10 UPIQ dev / +0.013 ladders zero-shot). Nothing from these
reads selects a model; the mix F stays the adopted model.** Bakes archived at `/var/tmp/byc-hdr-20261104/cont-s05/`
(mirror to tower with the banks).

**Owner decisions (2026-11-04, 12:30Z, on the HC1 result):** (1) the public 0.3.0 HDR path is the cube-root front end at
203 cd/m² for A (one queued breaking entry; PU21 stays research-only); (2) `E` and `ECodec` leave the public crate (public
line: PreviewV0_1, PreviewV0_2, LegacyA, A; `codec_target()` returns A); (3) the obsolescence inventory resumes on a
Sonnet lane. Both public-crate changes go to imazen/zensim PR #2 through the `zensim-profile-a` Codex lane (phase 2 brief
`~/tmp/zensim-profile-a-brief/PHASE2.md`): the HDR port follows zensimx `94095e06` and `41ea3ecf`, with A's HDR goldens
derived from the research `BakeScorer::with_hdr_front_end` and a public continuity test.
**Owner decisions (2026-11-04, ~14:30Z):** `LegacyA` (the v47 development model, never published) is dropped from
the public crate together with the revision-1 372-feature MLP path only it used; the previews `PreviewV0_1`/`PreviewV0_2`
(the 0.2.7 API) go behind a cargo feature `legacy-models`, off by default, scores unchanged. Public 0.3.0 default line:
`A` only. Phase 3 brief `~/tmp/zensim-profile-a-brief/PHASE3.md`.


### R12 registration (2026-11-04, before any fit or new read): linear formulas over A's inputs

Owner asks what linear weights buy and give up against the adopted mix (research F, public A).
Training root: `/var/tmp/byc-fchroma-20261029/r3/w100/root`, the mix's admitted TRAIN legs,
`--human-sources tid2013 --exclude-datasets pipal,live-md,cid-iq,cid-oa,irccyn-ivc,colourlab-rating-variability-2022
--mos-share 36.035`. Same 492 damage IDs as `~/tmp/hc1-profile/cols_fx2.txt` and their 492 fx2
fragility products. Reuse `v2_confirm_fit.py`, strict admission, the mix's loss/targets, masses,
samplers, 120 epochs and final-epoch rule. The supported linear route is plain N, H1, mono,
without thresholds/max pairs: scale-only inputs, W1 >= 0, b1 = 0, W2 <= 0, b2 = 100.
Thus raw = 100 - sum(nonnegative coefficient * input); both direct and product damage
coefficients are nonnegative. Three registered seed indices 0–2, because Adam initialization
and pair sampling are random. No fit or selection uses EVAL.

- **L1 dense:** all 984 inputs available.
- **L2 sparse family formula:** retain only basic edge_det, v2 art_soft_peak, hf_mag_loss,
  blockiness, hf_loss, ssim_dev4, basic edge_art and v2 iw_art, and their products. Keep the
  same fx2 declaration; fix all other input weights to zero through the existing strict
  feature mask. All nonzero coefficients with canonical feature names will be reported.
- **L3 conditional:** L1 only, AIC-2026 public_train ×256 continuation with original rehearsal,
  60 epochs, lr 1e-5, epoch 59, three seeds, only if L1's seed-mean absolute score error on
  the root's teacher development groups is at most one score point above A's seed mean.
  Score these frozen root development rows with final packed bytes and compare with
  100 * human_score; pool teacher rows, disclose group-wise errors too. This is a
  training-side feasibility check, not an EVAL decision. If it fails, omit L3 and report why.

Packing: canonical `v2_production_pack.pack_production`, f16 before CID22 TRAIN dial
calibration, E33 identity knot and tail factor 2, no automatic input pruning (preserve fx2
positions); native monotone certificate. The fitted head is linear before the common
output stage/dial spline. Record every departure from A's objective, if required.
Freeze fits/hashes in a second commit before report-only reads.

Reads: exactly the applicable HC1/mix owners in `~/tmp/hc1-profile/hc1-reads.sh`: held-out
EVAL collections (`dataset_eval.py score`, `readout.py`, compression/non-codec `ceval.py`),
E36 selection-source inversions and `primary.py` with CONTROL=F, G-CHROMA-L dev >64px,
G-STEER, C1–C6, identity, NEARID, and HDR panel plus UPIQ-all on cube-root:203 banks
`/mnt/v/output/zensim/byc-hdr-20261104/banks-cube-root-203-aliases`. Baselines F seeds 0–2
at `~/tmp/e36test/r10/runs/mixF3-s*/pack_e059/refit/production-f16.bin`.
No confirmation-set, public_test or other sealed read. Confirmation sets' prior exposure
still binds; DATA_SPLITS ledger records it. No AIC-2026 statistics in committed files.
Measure head-only forward time per pair against A, with extraction unchanged; L2's
family-removal opportunity is separate and refers to the extraction-cost report if available.
All fits/reads on dev, serialized through heavy.lock and run-heavy with 24G cap; scratch
`~/tmp/zensim-linear`, large artifacts `/var/tmp/byc-linear-20261104`.

Result: report every registered read beside A, full L2 coefficient formula, hashes, head-only
timings and product-gate verdict in `benchmarks/r12_linear_head_2026-11-04.md`.
Negative/incomplete outcomes are results, never adoption or permission to weaken a gate.


**R12 development-unit correction (2026-11-04, before any L3 fit or EVAL read).**
The registration's `100 * human_score` is an erroneous conversion: the teacher tables
already store score points, and the canonical trainer uses `--target-scale 1`.
The first R12 development check multiplied them again and is invalid (retained in
`/var/tmp/byc-linear-20261104/dev-unit-error/`). Compare final packed scores directly
with the stored `human_score` instead. The population, row weighting, three-seed mean
and one-score-point allowance remain exactly as registered. A fresh corrected TRAIN
development check must qualify L3; the invalid result does not authorize continuation.
This corrects a unit mistake introduced by R12, with no EVAL-guided selection.

**R12 coordinator addition (2026-11-04, before any L0/L4 construction or read).**
The coordinator requests two more arms, with all earlier R12 fits/results and
registered reads retained unchanged:

- **L0, structurally pruned A:** for each baseline seed, remove only hidden
  units whose entire encoder column is exactly zero and whose bias is <= 0.
  Preserve the remaining units' order, all input declarations/scalers, live
  weights, output bias, output stage and dial spline. Seed 0 must remove the
  audit's 95 dead units. This is data-free pruning, not selection or a fit.
  Prove strict served-score bit identity against its corresponding A seed
  across the registered feature fixtures and pixel gates, including identity
  and NEARID; any differing bit fails the exact-pruning claim and is reported.
  Report L0 beside A and L1–L3 on every registered read, plus head-only timing.
  Optional **L0r** diagnostic: retain only A0's audited live max pair, removing
  the 31 residual units too. Measure score differences on the same registered
  fixtures; do not assume identity, select on the result, or silently promote
  this approximate variant to L0. L0r is a fixture-difference diagnostic only.
- **L4, fitted two-piece max-linear:** three seeds, same admitted mix root,
  all 984 inputs, loss/targets/group weights/sampler and 120-epoch final-epoch
  rule as L1/L2; no AIC continuation or EVAL selection. Raw score is
  `100 - c * max(a dot x, b dot x)`, with `c >= 0`, `a >= 0`, `b >= 0`,
  both affine offsets fixed at zero, and no residual units. Use the canonical
  max-pair projection and trainer, with a narrowly bound R12 authorization
  for H2/one pair/zero thresholds; any necessary admission/orchestration
  extension preserves historical recipe contracts. The existing coupled
  representation is `a dot x + ReLU((b-a) dot x)`; a signed stored correction
  is allowed only when both complete pieces remain non-negative.
  Pack/calibrate exactly as L1/L2. Every registered read is report-only and
  reported beside A, L0 and L1–L3, including head-only forward time.

Keep the original L1/L2/L3 choices frozen. L0 and L4 reuse already-exposed
assessments, not fresh confirmation. Existing qualification thresholds remain.

**R12 L4 offset amendment (2026-11-04, before L4 construction).**
Owner answer: L4 learns **one non-positive offset on the second piece**, so
`raw = 100 - c * max(a dot x, b dot x - t)`, with `t >= 0`, `a >= 0`,
`b >= 0`, `c >= 0`. This replaces the addition's "offsets fixed at zero";
**one learned non-positive offset on the second piece; amended before L4
construction**. A0's audited pair biases are 0 and −5.868288993835449.
Use the existing threshold-enabled max-pair owner, keep the first bias at
zero and learn the second at <= 0. No additional arm, fit budget, selection
read or evaluation population is introduced. No L4 model has been constructed,
fitted or assessed before this amendment.

**R12 original-arm read result and L0/L4 freeze (2026-11-04).**
The [R12 report](r12_linear_head_2026-11-04.md) and
[read tables](r12_linear_reads_2026-11-04.md) record completed original-arm reads.
E36 inversion share: A 3.396647%, L1 3.148249%, L2 3.897773%, L3 3.595252%.
Dense L1's paired change is −0.248398 percentage points (95% origin bootstrap
−0.427255 to −0.060774); this is reused instrument evidence, not fresh confirmation.
All original fitted seeds pass G-STEER, C1–C6, identity and NEARID; all fail
G-CHROMA-L level qualification. No candidate is adopted. The six L0/L4 bakes
were frozen in `007c4a00` before additional-arm scoring, with all three L4
certificates showing one max pair and no free-ramp amplitude. A0 structural
pruning removes exactly 95 units; proof is measured separately, never assumed.

**R12 completed result (2026-11-04).** All fifteen registered fits/derivations
and all original/additional-arm reads are complete. The
[final six-arm report](r12_linear_head_2026-11-04.md) and
[full read tables](r12_linear_reads_2026-11-04.md) retain the comparisons.
L0 matches all registered score bits and all reported HDR metrics; adopted
A0's pruned head takes 2.839829 µs versus its matched A0's 5.766177 µs.
L4 E36 inversion share is 3.161444%, paired change −0.235202 percentage points
(95% origin bootstrap −0.412274 to −0.049230). Its all-nine MOS normalized median
error is 8.33836%, versus A's 8.02794%. Every fitted L1/L2/L3/L4 seed fails
G-CHROMA-L level qualification, while passing G-STEER 135/135, C1–C6, identity
and NEARID. No model is adopted. The sparse formula has 144 damage rows and
288 available coefficients, with all weights and feature names committed.
Extraction remains unchanged; reported speed is the head alone.
