# Owner-approved frozen-A AIC-2026 assessment — 2026-11-05

The owner approved “read it”; execute once after CLIC 2022 finishes. This is a
separate assessment, outside the eight-population final-read record.

- Population: all 1,848 `public_test` pairs over 14 source groups.
- Label authority: `/mnt/v/output/zensim/byconstruction-2026-10-22/aic2026train/folds_train/pairs_test.tsv`,
  SHA256 `b0fb0c0fbb9fd05a6a28d4b485f8a793d0ad384c4ca7d61172ed1f49a917f63b`.
  Use column `jnd` and quality target `−jnd`, exactly as the October 23 owner
  `scripts/byconstruction/aic_panel.py:247–251`; labels are preliminary.
- Frozen A: public source `704698224dd75b55289d2f570bc20fed317c13b7`, bake SHA256
  `d5e2a56db35c32620d9f9d9c19ba3bc78bd8fea2b8cd933cde8447428d85e62a`.
  Reuse the frozen Rev7 serial extractor and Rust scorers from the completed
  final-read instrument; no fitting, tuning or model selection.
- Compute A, SSIMULACRA2 and butteraugli 3-norm on identical native RGB pixels,
  hash-verified with independent readback. Butteraugli quality is explicitly `−p3`.
- Report signed SROCC, KROCC, PLCC after the registered statistical 4PL fit, and
  mean within-source SROCC, with 95% whole-source percentile intervals:
  10,000 draws, seed `20261105`, type7 endpoints. No acceptance threshold.
- Exposure disclosures: frozen B and Tga were read on October 23; public_test
  sources informed that October study's design. A's continuation trained on
  `public_train`, with no `public_test` source in that continuation. This is a
  frozen-A assessment of an exposed population, not a clean confirmation.
- Quote B's stored `panel_public_test_frozen.json` row without recomputing B;
  current stored-file SHA256 `3b39463cf6bc6c52d2c6a764bac608619934a4807994a02d0732e8add07dcb54`.
- Before target parsing: commit/push/verify this registration and the DATA_SPLITS
  authorization entry; ledger the private read before its single label parse.
  A failed or partial parse is spent and requires coordinator direction to reopen.
- **All evaluation numbers, including the stored B row, belong only in private
  `imazen/zensim-training`, under `evaluations/`, and chat.** No numerical result
  in this repository, public zensim, or another public file. Add nothing about
  this population to `docs/FINAL_READ_A_2026-11-05.md`.

Status at registration: label-file hash, header and key/source columns inspected;
no `jnd` target parsed and no frozen-A assessment performed on this population.

Completion: the single authorized assessment finished all1,848 pairs/14 sources
after CLIC2022. Completed journal SHA256
`2bb36a5e33610cb6fc4d378d5b445a19604881eb7fddfc1663b32c1a8236d67e`.
The private evaluation record is `imazen/zensim-training/evaluations/frozen_a_aic2026_public_test_2026-11-05.md`;
no numerical result is recorded here. No raw-label reopen or B recomputation.
