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
