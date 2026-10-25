# Frozen-membership confidence metadata — 2026-10-25

Missing: the published confidence manifests do not supply per-item metadata for many collections. All original membership rows are retained, with null fields and explicit missing reasons. No count, spread, weight or keep flag is inferred from a dataset-wide median. The new subset source-screen admission is still pending; this pointer currently covers the 21 historical membership collections only.

Checked on dev: `/mnt/v/zensim/dataset-subsets-20261025/confidence-existing/`.
The 21 Parquet side tables contain **498,216 rows**, in exactly the original `(dataset_id,input_table,row_index)` order, with identical role, state and family ID. **2,444 rows** match published item/pair metadata. The `eligible_for_training` metadata flag is true for **1,280 rows** after keep, FROZEN TRAIN and licence-policy checks; it grants no new fitting permission and a missing flag does not alter an existing membership. No model was trained and no source quality target or MCL-JCI rating payload was read.

Original membership root: `/mnt/v/zensim/dataset-admission-20261025/`. Confidence source: `/mnt/tower/output/zenpapers/training-confidence-2026-10-25/`, inventory SHA-256 `11d594a1f243b184158bf377ebb9207f0008579e66337b6a95f5a984c39c7b7c`. Source owner: zenpapers `scripts/iqa_library/training_confidence.py`. Join producer: zenpapers `8985c1dd4e686af7175b8b755439929cb5bb4c0f`, `confidence_memberships.py`. Current matching logic preserves published rounding and binds secondary protocols separately. A MOS weight is never transferred to a separate pairwise table for the same image.

Rated items carry keep, published weight, rating count and normalized SE where supplied. The producer computes `min(1,(0.03/SE)^2)` before rounding, and uses weight 1 when SE is unavailable; a null `se_norm` distinguishes that fallback from measured precision. Pairs carry total votes and keep flags, with no preference probability or winner counts. Validate-only populations retain their membership roles but their training-permitted flag is false. LS2N's 900 historical metadata rows remain explicitly out of scope; its source files were untouched. No TEST/EVAL/CALIBRATION role became TRAIN.

| Collection | Membership rows | Matched confidence rows |
|---|---:|---:|
| bapps | 187,744 | 0 |
| cid-iq | 690 | 0 |
| cid-oa | 299 | 143 |
| cid-pps | 30 | 12 |
| colourlab-chroma-masking | 960 | 0 |
| colourlab-rating-variability-2022 | 640 | 320 |
| diffiqa | 179,208 | 0 |
| fg-iqa | 1,800 | 900 |
| ie-bench-db | 3,984 | 523 |
| iqa-odi | 960 | 72 |
| irccyn-ivc | 210 | 0 |
| live-md | 450 | 210 |
| locvis | 557 | 0 |
| locvisvc | 967 | 0 |
| ls2n-ipi-sr | 900 | 0 |
| mdid | 1,600 | 0 |
| nits-iqa | 405 | 0 |
| pieapp | 63,060 | 0 |
| pipal | 23,200 | 0 |
| spcd | 30,000 | 0 |
| vcl-fer | 552 | 264 |

The verification checks every key, role, membership state, family ID and row order against the original admission, all side-table hashes, and every true training flag. No human response table is opened by that check.

| Receipt | SHA-256 |
|---|---|
| `_MANIFEST.json` | `56196f7410c8a10eefeb6fc5d6e43e3ca740e4c7297b60290cd4046d78bf43cd` |
| `SHA256SUMS` | `3a2693fb3632e3b1dac7ba07c3c2cf7c6295a1a1bd290d1954fc680a8fd8ef18` |
| Adjacent `confidence-existing-verified.json` | `8dbb8097ff478b5160307ff671a8f8663c9bc57c93e698488dc8a8ccf0f365e0` |

These private data are local-only until the subset task's final verified mirror. IMI remains deferred.
