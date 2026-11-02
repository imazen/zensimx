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
| M7 (reference) | +0.15 [-0.16, +0.47] | +0.12 [-0.45, +0.67] | +0.58 / +0.56 (both above 0) | | -0.30 / -0.77 |
| butteraugli loop | +2.22 [+1.48, +3.05] | +4.08 [+3.31, +5.00] | +1.34 / +3.68 | | +5.12 / +6.77 |

Against the butteraugli loop every zensim map costs bytes over all images: F3 -2.42% (SSIMULACRA2) / -4.44% (CVVDP),
p30L -2.39 / -4.34, p2-30 -2.34 / -4.10, M7 -2.32 / -4.32 (B, bulk placement: -3.32 / -5.53).

**Reading (the registered rule).** Neither p30L nor p2-30 steers JPEG XL better than F3 over all images: both are
ties. On photos p2-30 is better under both judges (+0.19% / +0.26%), p30L under CVVDP only. The hinge arms' higher
agreement with butteraugli's map (0.70 -> 0.74 at 8 px) buys at most a few tenths of a percent, and every zensim map
remains 2-4% behind butteraugli's own loop. The candidates cost nothing in steering, which removes the loop as an
obstacle to adopting one. Encodes, judge tables and BD-rate tables: `/mnt/tower/output/zensim/e36-subset-test-2026-11-01/exp3a-jxl-i270/`
(SHA256SUMS).
