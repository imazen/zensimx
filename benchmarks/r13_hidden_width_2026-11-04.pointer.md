# R13 completed width experiment — artifact pointer

The six candidate seed-0 bakes were fixed before R13 assessments; all registered reads are complete. A is the frozen mix F seeds 0–2, without refitting. The owner cut ten seed repetitions on cost and added W4 s0 with the full continuation.

- Host checked: dev, 2026-11-05 UTC.
- Artifact root: `/var/tmp/byc-width-20261104/`.
- Frozen inventory: `FROZEN_FITS.json`, 26,103 bytes, SHA-256 `7cbcb2e7e5726e4b31738da1de6ac480a7c58786cd3c6b12970969a4ea12e9b1`.
- Packed identities, capacity partitions and fit/resource receipts: [retained fit records](r13_fit_progress_2026-11-04.json).
- Every packed bake has a native monotone certificate. Metadata stripping preserves the native inspection of layers, scalers, derived declaration and dial exactly. All 984 inputs remain.
- Original numerical trainer source: `ccdb5c7aa0d4b67d52f0e2874f325fbc84ddb11c`; original trainer SHA-256 `267a3492e14bd2c2988e49a056b9a1840b188aada796804974f635b733c670cf`.
- W4 uses that numerical source with the sole registered native admission overlay. Its trainer SHA-256 is `d14d9c8b69067fa0761608f12248c6801a9e4ace0a678ee11602beee6dd0d2c9`; native admission/build source `509c4f09`, scoped recipe-parser repair `e39e7302`.
- Full file and binary pins: `SOURCE_PINS.json`, `W4_SOURCE_PINS.json`, `W4_FIT_SOURCE_PINS.json`.
- The W4 setup failure occurred before optimization and is retained under `failed-attempts/W4-s0-recipe-parser-v1/` with its 1.24 s receipt. The full successful fit starts from fresh seed 0.
- Tower mirror: `/mnt/tower/output/zensim/r13-width-20261104/`, verified 2026-11-05T02:13:38.022014+00:00. Inventory: 14,425 files, 2,818,361,257 bytes; `SHA256SUMS` SHA-256 `6cb73cd092869d1a6dc838aefe93c639043bec632e767924b9f654fc25076c6d`. Full rsync checksum comparison is empty; three random SHA-256 readbacks match. [Committed archive receipt](r13_archive_2026-11-04.json).
- R2 mirror: none; this run retains its evidence locally and at the owner-specified Tower destination.

Registration: `25e77292`; cost cut and W4 amendment: `397d67be`. No AIC-2026 assessment statistics are published. Public_test and the closed confirmation collections remain sealed.

Results commit `85231eb9`: [width report](r13_hidden_width_2026-11-04.md), [complete read tables](r13_hidden_width_reads_2026-11-04.md). Completed exposure entry `4621a33d`.

`FINAL_VALIDATION.json` verifies six full 119/59 fits, 12 capacity cells, six native head pairs and all 36 read receipts. `read-source/`, `read-tools/` and `READ_TOOL_ARCHIVE.json` preserve the pinned read implementation and executable bytes. Rebuildable Cargo output and live bookkeeping are excluded from the mirror inventory; final archive log/time/receipt are separately copied and hash-verified after completion. No scientific read output is excluded.

Archive resource receipt: run-heavy: done rc=0 147s | peak-RSS 0.11GiB | min-avail 34941MiB | peak-load 2.67.
