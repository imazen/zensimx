# Provided-library subset Rev7 features — 2026-10-25

Missing: no TRAIN rows were admitted for Toyama because 112 prospective TRAIN
rows retain unresolved source matches. These are not confirmed overlaps. HDR-JPEG
and PairComp TMO require a native HDR/viewing recipe; Cognitive Experiment 1 and
demosaicing have no pristine reference; Fourier is tier C. None is silently adapted
to the pinned SDR extractor. No quality targets were read and no model was trained.

The six collections contain 1,215 projected identifier rows: **577 FROZEN TRAIN,
390 FROZEN EVAL, 136 FROZEN TEST, 112 excluded TRAIN**. Every required image side
of all **1,103 FROZEN rows** is covered by **753 distinct native file pairs**.
There are zero no-reference skips, decode failures or unresolved feature rows.
All old memberships stay unchanged. [The membership registration](../docs/DATA_SPLITS.md#october-25--additional-provided-library-subset-memberships)
binds native table keys, per-role membership hashes and source-screen limits.

Checked on dev:

- `/mnt/v/zensim/dataset-subsets-20261025/admission/`: immutable membership and family receipts.
- `/mnt/v/zensim/dataset-subsets-20261025/features/bank/<token>/`: `features.parquet`, `keys.parquet`, `_MANIFEST.json`.
- `/mnt/v/zensim/dataset-subsets-20261025/features/inputs/<collection>/`: every membership-to-pair mapping and preparation receipt.
- `/mnt/v/zensim/dataset-subsets-20261025/features/FEATURE_COVERAGE.json`: role-preserving coverage, including both A/B sides.
- `/mnt/tower/output/zensim/dataset-subsets-20261025/`: verified mirror of the complete private artifact root.

| Collection/token | TRAIN | EVAL | TEST | Pairs | Bank manifest SHA-256 |
|---|---:|---:|---:|---:|---|
| colourlab_cognitive_science_2022 | 109 | 66 | 44 | 219 | `e0b8363619a643afc36144c31b3775c3f93b6661c0cb50cefaf2334c54e6cd7d` |
| irccyn_ivc_broken_arrows | 60 | 36 | 24 | 120 | `a92a41767f21a09bbd96920e9fbde240e1b85232913557912349cfe9b7a21dc4` |
| irccyn_ivc_dibr_images | 288 | 144 | 0 | 96 | `2054f11daa8d1bb18e7c50a79dffecf8f721acc7c27b9397f039e666ff816729` |
| irccyn_ivc_dwt_dtcwt | 60 | 40 | 20 | 120 | `73cb179dbfb20b596b39fe8658f7fe453e10790e0a255c5df30c6c0b651a64d6` |
| irccyn_ivc_enrico | 60 | 40 | 0 | 100 | `1543253ecf1d5f17042122777e6bf7a16f093663e5520a8f6a7cbd35c3a26f4e` |
| irccyn_ivc_toyama_scores | 0 | 64 | 48 | 98 | `d37908d5fc15fe92b0f255c7b19cc92df6dc93421d2de4c56759f53657386705` |

DIBR's 432 membership observations use 96 distinct pairs; all four viewpoints
of each sequence share a source family. Its ACR, scaled-score and comparison
protocols remain separate. Cognitive Science Experiment 2 uses the exact recorded
parent identifier join into Colourlab rating variability. A future label reader
must replay that binding; it may not bypass the native reference check.

Extractor build `b77883be18db5933a9db4b9e2670a2ba88a9729b`, binary SHA-256
`ed4957417339779dd8ed25fe59a6166326ba9db86d5c1530b1e1c788c9d614d8`;
feature set `basic+peaks+v2@w1825/rev7_rowlanes#36c3f3af`, float64 width 1825,
576 populated slots `[0,228)` + `[372,720)`, all absent slots NaN. Assembly uses
`scripts/rev4_featpot/rev5_bank.py`, SHA-256
`2a0835487a5ff6fb559bab7e00a159cdb5286c8ee7adde6e42f09f40f86cefdd`.
Admission/confidence producer: zenpapers `71f8920cac4ed6679bc7b92eebe5d42c7319e4eb`;
key preparation: zensim `97c10a044501`; mounted mirror owner `dc79d8dbacd5`.
Native decoder audits, pair order, file/pixel hashes and source manifests remain
in the private bank work directories; no source image is copied locally.

All CPU work runs on dev under the shared heavy lock. Extraction acquired and
released it once per collection, smallest first. Each collection returned rc=0;
recorded run-heavy peak RSS lines were 0.94, 0.65, 0.47, 0.46, 0.46 and 0.73 GiB
in queue order DIBR, Toyama, Enrico, Broken Arrows, DWT, Cognitive Experiment 2.
These are wrapper observations, not a comparative performance claim. The calibrated
source screen reported `rc=0 992s | peak-RSS 8.28GiB`; its recall is bounded.

The mounted-filesystem mirror copies and hashes on dev, starts no remote process,
and verifies every artifact plus three random readbacks. It contains 195 artifact
files (244,870,970 bytes) before its receipt, 196 final inventory entries. Tower
supplies library reads and modest NFS writes; no compute or container runs there.

| Receipt | SHA-256 |
|---|---|
| `admission/validation-receipt.json` | `ba44794cdf87818ae112b0198cb34965c6f7a926683dae5580895ba87d2271b9` |
| `admission/source-family-registry.tsv` | `16af89ed47b1f66ad32196e3696146a28bdd579045723e0ae84bf7ee69260916` |
| `features/FEATURE_COVERAGE.json` | `ba4b8df5ed00cd7b273909d3f8335d53a9564885ed32ace57a0f795a0cb637eb` |
| `confidence-new/_MANIFEST.json` | `941be14cbaf8de0b404d015814cdd40305e7887598dd026cc213d4ace3747583` |
| `confidence-new-verified.json` | `9d236b556c7412c0ad5257c70f4ed6064d60747fe9d029d4cc67856203945030` |
| `MIRROR_VERIFIED.json` | `1777840efd5fd695d11a6f27d508dcb508c76781b38bf0720b91ae5d26b573d1` |
| `SHA256SUMS` | `4d337b49392d5e7db15f2a98fda5f7e0b3db360c4ea7f83a9f66885d2c7319fc` |

[Confidence side tables](dataset_confidence_2026-10-25.pointer.md) preserve all 27
collection identities and explicit missing fields. D3 accepts MCL-JCI/KonFiG source
overlap and preserves KonFiG TRAIN; MCL-JCI ratings remain unopened. No training,
assessment, serving-profile change, dataset fetch or web request occurred. Existing
IMI work stays deferred. Original protected and historical artifacts remain intact.
