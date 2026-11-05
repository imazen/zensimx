# Final confirmatory read of the shipped model A (zensim 0.3.0) on the sealed sets — registration, 2026-11-05

Owner, 2026-11-05 (verbatim): "score on the final sets". Nothing in this read selects, tunes or
changes any model; failure or success is final and is reported as measured. Written before any label is opened by this assessment; historical exposures are disclosed below. To be committed to the zensimx record (`benchmarks/final_read_a_2026-11-05.md`)
and appended to `docs/DATA_SPLITS.md` exposure ledger before the lane reads a label.

## Model under read

* Public `zensim` 0.3.0 line, profile `A` (`zensim-a`), bake
  `weights/rev7_mix_h128_2026-11-03.bin`, SHA-256
  `d5e2a56db35c32620d9f9d9c19ba3bc78bd8fea2b8cd933cde8447428d85e62a` (research `F`, recipe
  `mixF3`, seed 0, continuation epoch 59). Frozen at imazen/zensim PR #2 head `bebc6ea4`
  (profile-f `70469822` adds only the parallel cutoff, scores unchanged).
* Scoring surface: the research crate at a pinned commit whose `split_parity_golden` is byte-identical
  to the public crate's `split_parity_check` (1,386 rows, 2026-11-04). The lane records the commit,
  the binary SHA-256 and the parity receipt. Serial extraction, `ZENSIM_FORMULA_REV=7`.
* Comparators, reported beside A, never selecting: SSIMULACRA2 and butteraugli 3-norm (the R10
  amendment 5 reference-metric tooling, `scripts/byconstruction/e36_exp56/refmetrics_run.sh`);
  for KADID TERMINAL additionally research profile `B` as its 2026-10-07 registration requires.

## Sets (approved frozen-A assessment; prior exposure disclosed per population)

| Population | Rows / references | Label | Pixels | Statistic |
|---|---|---|---|---|
| CID22-B | 24 references, their validation pairs | MCOS (quality) | library `cid22` PNG/JPEG decodes | signed SROCC/KROCC/PLCC(4PL), RMSE on the collection scale, within-reference SROCC Prior zensim read: October 4 R7/R7a and October 18 candidate assessment (2,100 rows); `docs/DATA_SPLITS.md:1613–1617,2761–2791`. |
| CSIQ | 866 distorted / 30 references | DMOS (distortion; sign disclosed, reported as quality = 1−DMOS) | library `csiq` | same panel Prior zensim read: October 4 R7/R7a and October 18 candidate assessment (865 rows); `docs/DATA_SPLITS.md:1613–1617,2761–2791`. This registration retains the approved 866-row population. |
| LIVE release 2 (open external since 2026-10-03) | 779 / 29 references; also the 10 content-disjoint references (267 pairs) fixed by the 2026-10-03 audit | `1 − dmos_new/100` | library `live-iqa` | same panel, both populations Prior external zensim reads since October 3; October 18 clean10 assessment (267 rows); `docs/DATA_SPLITS.md:2761–2791` and `scripts/rev4_featpot/external_sets.py`. |
| KonJND JPEG TERMINAL | the registered TERMINAL split only | PJND (threshold) | library `konjnd-1k` JPEG, decoded by zenjpeg main `762fa712` | signed SROCC vs PJND; within-reference SROCC Prior zensim read: October 4 R7/R7a, SELECT and TERMINAL labels; `docs/DATA_SPLITS.md:1613–1617`. |
| KADID TERMINAL | 2,000 pairs | MOS | local `kadid10k` | the 2026-10-07 registration exactly: signed SROCC, KROCC, PLCC, within-reference SROCC, per-distortion-type SROCC, `zenstats::scatter` geometry; comparators the shipped research profile B (`codec_target()` of the research crate at the pinned commit, bake SHA recorded) and the Rev4 research by_v2fy bake (resolved from `benchmarks/release_gate_map_2026-10-07.md` / `releasegate5_admission_2026-10-07.md`, SHA recorded); confirmation rule: A's signed SROCC not worse than B's by more than 2 SE (paired bootstrap over references, 10,000 draws, seed fixed in the pin) and not worse than by_v2fy's by more than 0.005; failing either is a failed confirmation, recorded as such. If the by_v2fy bake cannot be located, the read still happens, the B rule is applied, and the missing comparator is recorded as a deviation, never silently dropped No prior zensim model scored this population; third-party-metric/reliability reads October 24–26 are disclosed at `docs/DATA_SPLITS.md:2892–2924`. |
| AIC-4 sample | 300 pairs / 5 references; SDR25 = its 50-row JPEG-AI subset, reported as a subset, not independently | `q_jnd` (distortion-oriented; reported signed, sign disclosed) | library `jpeg-aic4-sample` | signed SROCC, within-reference SROCC Prior zensim read: October 4 R7/R7a; `docs/DATA_SPLITS.md:1613–1617`. Prior JND-unit study: `docs/DATA_SPLITS.md:2886–2889`. SDR25 inherits this exposure as its subset. |
| MCL-JCI | all rated conditions | JND-based quality | library `mcl-jci` | signed SROCC, within-reference SROCC. Prior exposure, disclosed: zenpapers read every subject's JND chain and 243 library score rows on 2026-10-24/25 for JND-unit and observer studies with third-party metrics only, without a preregistration (ledger, "Exposure ledger — 2026-10-24/25"); no zensim model was scored, fitted or selected on them. D3 made its images training sources (KonFiG-IQA crops ten of them); its ratings never trained any zensim model Prior zensim read: October 4 R7/R7a, including the 40-source subset excluding KonFiG crops; `docs/DATA_SPLITS.md:1601–1617`. Additional October 24/25 exposure: `docs/DATA_SPLITS.md:2867–2899`. |
| CLIC 2022 perceptual (TERMINAL) | 57,243 2AFC questions | human preference | library `clic-2022-perceptual` | agreement with the majority choice, as the CLIC 2021 read |

Excluded and why: AIC-2026 `public_test` (its one read is spent, 2026-10-23; AIC-2026 numbers never
leave private files); CID22-A25, AIC-3, KADID TRAIN/SELECT, TID2013, KonFiG (training sources);
EVAL's nine collections (the mix was chosen on them; already reported); the threshold sets
JND-MID-MA, VLIC and Konstanz PJND (different instrument, no registered statistic).

## Reading rule

* One read per population, ledgered in `LABEL_READS.json` before any label column is parsed
  (`mos_confirm.py confirm` pattern); the DATA_SPLITS exposure ledger is appended under `flock`
  before the first label open. A failed or partial read is still spent and is reported as such.
* Resampling unit: references (whole-reference bootstrap, 10,000 draws, fixed seed recorded).
  Every statistic is reported with its interval; no threshold decides anything except KADID
  TERMINAL's registered rule, which is reported pass/fail.
* Orientation: `--signed-quality`; distortion-oriented targets (CSIQ DMOS, AIC-4 `q_jnd`) are
  reported with the sign and the orientation declared, never flipped silently.
* Disclosures carried into every report of these numbers: LIVE's 19 Kodak references contain a
  TID2013 reference (training source); MCL-JCI shares sources with KonFiG-IQA (training); SDR25 is
  inside AIC-4; CID22-B is the second half of a reference set whose first half trained A; CLIC 2022
  shares CLIC's protocol with the CLIC 2021 confirmation read.
* Prior label exposure, disclosed rather than denied: the exposure ledger records that MCL-JCI and
  Konstanz PJND labels (2026-10-24/25, zenpapers JND-unit and observer studies) and KADID TERMINAL
  labels (2026-10-24 to 26, reliability and metric-fit analyses) were read by third-party-metric
  studies before this registration, with no zensim candidate scored, fitted or selected on them, and
  that no preregistration preceded those reads. Earlier zensim reads are also on the ledger and
  are disclosed here: the R7/R7a set comparison of 2026-10-04 scored four frozen models on CID22-B,
  the AIC-4 sample, CSIQ, MCL-JCI and KonJND JPEG (SELECT and TERMINAL), and the 2026-10-18
  candidate reads covered CID22-B (2,100 rows), CSIQ (865) and LIVE's 10 content-disjoint references
  (267 rows). This read is therefore a frozen-A assessment on populations that earlier frozen
  models were already assessed on, with the one exception of KADID TERMINAL, which no zensim model
  has been scored on; the lane records the exact ledger lines beside each population.
* Publication: results go into the zensimx record and the public `zensim` write-up
  (`docs/`), with the disclosures above. Nothing about AIC-2026 appears in public files.

## Pixels and provenance

Library content read in place from `/mnt/tower/input/iqa/datasets/<id>/` (dev; local copies under
`/mnt/v/dataset/` only where DATA_SPLITS already names them). PNG/BMP decodes through the research
crate's existing decode path (zenbitmaps/zenpng); JPEG stimuli through zenjpeg main `762fa712`
(the pin that reproduced the frozen EVAL table bit for bit, 2026-11-05). Every decode map records
`rgb8_sha256`; feature banks record the extractor SHA; no third-party decoder.

## Execution pin and pre-read status

- Research source pinned to `bd384ffb408833231a3a91158b879c401340836a` (the parent of the registration workspace).
- Public parity source: imazen/zensim `70469822`, example `zensim/examples/split_parity_check.rs`.
- Bootstrap: 10,000 whole-reference draws, seed `20261105`; percentile 95% intervals. KADID comparisons use paired draws.
- Registration is committed before any label column is parsed. Scorer/extractor binary hashes, label-file hashes and parity receipts will be appended in a committed pre-read pin before any label parse.
- At registration: no labels parsed or sets scored by this assessment. Each actual read requires its own pre-open LABEL_READS.json and exposure append.
- Coordinator amendments: preserve both KADID comparators/limits; if by_v2fy cannot be located, proceed with B and disclose the missing comparator. October 4/18 prior zensim exposures replace the earlier unread/first-zensim claims; approved populations unchanged.
