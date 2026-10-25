# Frozen-membership confidence metadata — 2026-10-25

Missing: the published confidence manifests do not supply per-item metadata for many collections. All original membership rows are retained, with null fields and explicit missing reasons. No count, spread, weight or keep flag is inferred from a dataset-wide median. The additive admission is complete; the historical and new side tables are described separately below.

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

The complete task root is mirrored at `/mnt/tower/output/zensim/dataset-subsets-20261025/`,
with all hashes checked on dev over NFS and three random readbacks. No process runs
on tower. Final root inventory SHA-256
`4d337b49392d5e7db15f2a98fda5f7e0b3db360c4ea7f83a9f66885d2c7319fc`;
mirror receipt `1777840efd5fd695d11a6f27d508dcb508c76781b38bf0720b91ae5d26b573d1`.
IMI remains deferred.

The new `confidence-new/` folder preserves all 1,215 additional membership rows,
including 112 excluded Toyama rows. **800 rows** match source metadata and **412**
have the combined FROZEN TRAIN / keep / policy flag. Missing secondary protocols
stay null: Cognitive Experiment 2 (219), DIBR scaled-score table (84), Toyama
JPEG2000 table (112). No Experiment 1 weight transfers to Experiment 2.

| Collection | Rows | Matched metadata |
|---|---:|---:|
| colourlab-cognitive-science-2022 | 219 | 0 |
| irccyn-ivc-broken-arrows | 120 | 120 |
| irccyn-ivc-dibr-images | 432 | 348 |
| irccyn-ivc-dwt-dtcwt | 120 | 120 |
| irccyn-ivc-enrico | 100 | 100 |
| irccyn-ivc-toyama-scores | 224 | 112 |

New manifest SHA-256 `941be14cbaf8de0b404d015814cdd40305e7887598dd026cc213d4ace3747583`;
new inventory `dc1a71e3af32dbcc58aa5bab7050fec2187aa2e34e17a957724fb6e730910a9c`;
identity-verification receipt `9d236b556c7412c0ad5257c70f4ed6064d60747fe9d029d4cc67856203945030`.
Producer zenpapers `71f8920cac4ed6679bc7b92eebe5d42c7319e4eb`.
[Subset feature pointer](dataset_subsets_rev7_2026-10-25.pointer.md) records the
frozen role counts and all bank hashes. Across both scopes, 499,431 identities
are preserved and 3,244 have matched metadata; missing fields grant no fitting
permission and never alter the accepted role.
