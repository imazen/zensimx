# Chroma-trade gauntlet: rev7 feature sidecar, and F3 / F3codec on G-TRADE (2026-10-31)

The chroma-trade gauntlet (evif, run 2026-10-30, archive
`/mnt/tower/output/evif-chroma-trade-archive-2026-10-30/`) scored every encode with a fixed judge
set: E, ECodec, F, Fcodec and the references. F3 and F3codec (round 3, `benchmarks/byconstruction_2026-10-22.md`)
came later. The ledgers keep scores only, so this run extracted revision-7 features once for the
gauntlet's samples. Any bake now scores the gauntlet's ladders and size-matched pairs from features.

## What was extracted

- Every `.evif` sample of the 2,755 gauntlet trade jobs: 1,359,371 pack members, 1,341,311 unique
  containers (split-family templates of one view share some). These hold the decoded pixels the
  judges saw.
- Not covered: the 452,335 DE non-finds, which were stored as bitstreams only. They matter only for
  the search measure (b), which needs new searches for any new target anyway.
- Extractor: the frozen stage-C one (`evif-rev7-features-v1`, binary ed495741, feature set
  basic+peaks+v2 w1825 rev7_rowlanes, hash 36c3f3af, build b77883be), unchanged. zenmetrics
  `evif_features.py` reads the containers out of the run's CAS tar packs (efc5dbeb), indexed by
  `evif_pack_index.py` (c3040926), declared by `evif_features_spec.py` (113ea42e).
- Run `gauntlet-features-20261031`: 11,906 zenfleet jobs on i265 + i270, all done, no duplicate
  executions, 55 min. Image `zenfleet-worker:evif-features-packs-20261031` (local on both hosts, id
  5fb60764) = the stage-C feature image 8baba59e plus that executor file.
- Harvest: `scripts/byconstruction/gauntlet_features.py harvest`, with the stage-C audit checks
  (reference and distorted pixel hashes, canonical f32 feature digest) on every row.

## Check against the judges

`gauntlet_features.py augment` puts the bank's scores beside the ledger's own. Over all 1,387,305
gauntlet rows with features, the E, ECodec, F and Fcodec values from features equal the inline judge
values: 5,549,123 bit for bit and 97 at the predictor's printed precision (17 decimal places, below
0.1). There are no mismatches. Rerunning evif `tools/trade_gate.py` (same `--dims`) on the augmented
ledgers reproduces the original gauntlet `TALLIES.json` exactly for the four original models.

Join key: (view, encoded_sha256), not the bitstream hash alone. At the lowest qualities, two tiny views
(49x64) can encode to the same bytes, and against different references they have different features.

## F3 / F3codec on G-TRADE (registered metrics, unchanged tool and bootstrap)

Primary stratum (gauntlet partition, views above 64 px, 173 sources). (a) is the share of decided-worse
trade pairs a model scores higher; (a2) is the same for byte-matched pairs. Paired source bootstrap
against E (evif `trade-bootstrap`, binary b7d6d89c, 10,000 resamples).

| Model | (a) | vs E [95%] | (a2) | vs E [95%] |
|---|---|---|---|---|
| E | 7.61% (1,390/18,267) | | 1.19% (609/51,361) | |
| ECodec | 9.02% | +1.41 [+0.43, +2.48] worse | 1.38% | +0.19 [-0.01, +0.43] same |
| F | 2.90% | -4.71 [-5.78, -3.73] better | 0.68% | -0.51 [-0.67, -0.35] better |
| Fcodec | 4.44% | -3.17 [-4.13, -2.27] better | 0.83% | -0.36 [-0.54, -0.17] better |
| **F3** | **2.50%** (456) | **-5.11 [-6.23, -4.09] better** | 0.80% | -0.39 [-0.58, -0.19] better |
| **F3codec** | **2.90%** (529) | **-4.71 [-5.74, -3.78] better** | 0.69% | -0.50 [-0.67, -0.34] better |

Small views (64 px or less, butteraugli + CVVDP rule, 858 decided-worse pairs): (a) E 8.74%, ECodec
12.70%, F 6.41%, Fcodec 7.93%, F3 4.43% (-4.31 [-7.95, -0.77], better), F3codec 3.50% (-5.24
[-8.82, -1.87], better). (a2) on small views: no model is distinguishable from E.

### Held-out caveat: most gauntlet sources are in F's chroma training data

148 of the 173 gauntlet sources (by reference group) are in the stage-C chroma bank that F and F3
trained on. Split by that, descriptive only (ratio of sums, simple source bootstrap, seed 20261031;
the registered bootstrap covers all 173):

| Model | (a) 25 clean sources, 702 pairs | vs E | (a) 148 overlap sources | vs E |
|---|---|---|---|---|
| E | 15.10% | | 7.31% | |
| ECodec | 18.38% | +3.28 [-8.85, +16.93] | 8.65% | +1.34 [+0.46, +2.32] |
| F | 0.28% | -14.81 [-21.82, -8.48] | 3.00% | -4.31 [-5.36, -3.37] |
| Fcodec | 7.26% | -7.83 [-17.14, +1.59] | 4.33% | -2.98 [-3.89, -2.14] |
| F3 | 0.85% | -14.25 [-21.13, -8.04] | 2.56% | -4.75 [-5.83, -3.75] |
| F3codec | 2.85% | -12.25 [-18.69, -6.39] | 2.90% | -4.41 [-5.42, -3.51] |

(a2) on the 25 clean sources (1,800 pairs): E 2.67%, ECodec 10.17% (+7.50 [+3.73, +11.74]), F 1.56%
(-1.11 [-2.36, -0.19]), Fcodec 5.67% (+3.00 [+1.60, +4.76]), F3 2.50% (-0.17 [-0.72, +0.32]),
F3codec 3.11% (+0.44 [-0.19, +1.13]).

Reading: the inversion gain of F3 and F3codec over E also appears on the 25 sources they never
trained on, so it is not only memorised content. Those 25 sources are few, and their intervals are wide.
F3codec removes most of the size-matched regression that ECodec and Fcodec show on the clean
sources.

## Where the data is (dev, checked 2026-10-31; tower mirror verified by sha256)

- Bank: `/mnt/v/zensim/gauntlet-features/2026-10-31/bank/` (`features.parquet` 0eaf5c32, 5.5 GB,
  1,341,311 x 1825 float64, NaN for unrequested slots; `keys.parquet` 52f4a85a; `_MANIFEST.json`
  bcf92779; `SHA256SUMS`). Tower: `/mnt/tower/output/zensim/gauntlet-features-20261031/bank/`.
- Scores of six bakes: `.../scores/scores.parquet` 66544c46 (bake hashes in `SCORED.json`).
- Gate outputs: `.../gate-f3/` (`TALLIES.json` 5dbe6aae, `BOOTSTRAP.json` ad99559b,
  `source_labels.json`, `view-dims.json` from the original gate), checksums in `SHA256SUMS-results`.
- Feature-run CAS (11,906 job outputs): `/mnt/tower/output/zensim/gauntlet-features-20261031/cas/`;
  pack index, selection, spec and manifest beside it.

Reproduce: `gauntlet_features.py score BANK OUT --predict predict_features_with_bake --bake NAME=PATH ...`,
then `augment SCORES ASSEMBLED OUT --key NAME=LEDGER_KEY ...`, then evif `tools/trade_gate.py OUT/*/
--dims view-dims.json --models ...` and `evif trade-bootstrap`.
