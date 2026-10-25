# Provided-library revision-7 dataset features — 2026-10-25

MISSING: DiffIQA has no pristine reference (61,032 frozen memberships); the owner directs the same skip for SPCD (29,947). The pinned extractor cannot decode CID:PPS TIFFs (30 memberships: TRAIN 12, EVAL 6, TEST 12). Exact failing paths/native errors are in `DECODE_FAILURES.json`. No reference substitution, format conversion or extractor change. No unresolved memberships remain.

Sixteen banks contain **417,060 distinct file pairs**, covering **278,053 of 369,062** in-scope FROZEN memberships. These pair counts sum each collection's deduplicated reference/distorted path pairs; they are not global pixel deduplication or counts of independent ratings. Pairwise sides have separate feature rows. All 20 membership mappings preserve native admitted roles, including explicit skips and LocVis's zero FROZEN rows.

No response table, workbook, participant response payload or label column was opened; nothing trained. Protocol-document reading did expose published illustrative score/vote/preference examples and historical aggregate score checks. This is an exception to the literal no-label-values-read condition, recorded in `README_EXPOSURE.json`. Identifier-only matching found two FROZEN TRAIN, one FROZEN EVAL and four FROZEN TEST rows, plus six already-EXCLUDED TRAIN rows; FG-IQA/PieAPP illustrations remain unresolved to a unique membership. Roles did not change. No examples entered features, fitting or selection. Future assessments must disclose this documentation exposure.

## Locations and pins

Checked on **dev**, 2026-10-25 (`/mnt/v` is host-specific):

- Logical feature root: `/mnt/v/zensim/dataset-features-rev7-20261025/`.
- Actual disk-backed target: `/var/tmp/dataset-features-rev7-20261025/` (logical root is a symlink).
- Verified tower mirror: `/mnt/tower/output/zensim/dataset-features-rev7-20261025/`.
- Originally provided image library: `/mnt/tower/input/iqa/`; original pixels/archives remain there.
- No phase-2 R2 mirror was created. These artifacts remain private.

Pinned extractor: `/var/tmp/byconstruction-bin/b77883be18db-rev7/extract_features_372col`, SHA-256 `ed4957417339779dd8ed25fe59a6166326ba9db86d5c1530b1e1c788c9d614d8`; build `b77883be18db5933a9db4b9e2670a2ba88a9729b`.

Existing owner: `scripts/rev4_featpot/rev5_bank.py extract --instrument-manifest`, SHA-256 `2a0835487a5ff6fb559bab7e00a159cdb5286c8ee7adde6e42f09f40f86cefdd`. No replacement extractor, arithmetic or assembler. Feature ID `basic+peaks+v2@w1825/rev7_rowlanes#36c3f3af`, formula Rev7, era `rev7_rowlanes`; float64 canonical width 1,825 with **576 populated slots**, `[0,228)` and `[372,720)`. Other slots are NaN. Native per-row f32 digests, exact paths, raw/pixel hashes and empty model inputs were checked by the owner. The native audit's `human_score` is an ordinal row key only, not a subjective target.

`inputs/<id>/keys.parquet` carries only assessment key columns; `membership-to-pairs.parquet` preserves membership-to-side joins. Each runnable assessment manifest binds its keys, binary and immutable source roots. `bank/<token>/{features.parquet,keys.parquet,_MANIFEST.json}` is the existing owner's delivery. `TOOL_PROVENANCE.json` binds helper sources at `3b4fd51f9ee3105941d075c1e9e7bc67fc57094c`; the earlier dated environment snapshot is retained separately.

## Measured coverage

T=TRAIN, C=CALIBRATION, E=EVAL, S=TEST, D=SOURCE_EXPOSED_TEST_DIAGNOSTIC. Counts in the feature-pair column are distinct within each role/collection. A covered membership has every required side present.

| Collection | FROZEN memberships | Covered memberships | Feature pairs by role (total) | Missing memberships |
| --- | --- | --- | --- | --- |
| bapps | T 146,003; E 36,344 | T 146,003; E 36,344 | T 292,006; E 72,688 (364,694) | 0 |
| cid-iq | T 330; E 210; S 150 | T 330; E 210; S 150 | T 330; E 210; S 150 (690) | 0 |
| cid-oa | T 143; E 91; S 65 | T 143; E 91; S 65 | T 143; E 91; S 65 (299) | 0 |
| cid-pps | T 12; E 6; S 12 | 0 | 0 (0) | TIFF: T 12; E 6; S 12 |
| colourlab-chroma-masking | E 288; S 192 | E 288; S 192 | E 288; S 192 (480) | 0 |
| colourlab-rating-variability-2022 | T 320; E 192; S 128 | T 320; E 192; S 128 | T 320; E 192; S 128 (640) | 0 |
| diffiqa | T 7,272; E 17,916; S 35,844 | 0 | 0 (0) | no pristine reference: T 7,272; E 17,916; S 35,844 |
| fg-iqa | T 900; E 540; S 360 | T 900; E 540; S 360 | T 600; E 360; S 240 (1,200) | 0 |
| ie-bench-db | T 523; S 1,000 | T 523; S 1,000 | T 523; S 1,000 (1,523) | 0 |
| iqa-odi | T 72; E 48; S 191 | T 72; E 48; S 191 | T 72; E 48; S 191 (311) | 0 |
| irccyn-ivc | T 105; E 60; S 45 | T 105; E 60; S 45 | T 105; E 60; S 45 (210) | 0 |
| live-md | T 210; E 180; S 60 | T 210; E 180; S 60 | T 210; E 180; S 60 (450) | 0 |
| locvis | 0 | 0 | 0 (0) | 0 |
| locvisvc | T 337; E 30; S 119 | T 337; E 30; S 119 | T 337; E 30; S 119 (486) | 0 |
| mdid | T 800; E 480; S 320 | T 800; E 480; S 320 | T 800; E 480; S 320 (1,600) | 0 |
| nits-iqa | E 405 | E 405 | E 405 (405) | 0 |
| pieapp | T 51,000; E 7,260; S 4,800 | T 51,000; E 7,260; S 4,800 | T 17,220; E 2,460; S 640 (20,320) | 0 |
| pipal | T 23,200 | T 23,200 | T 23,200 (23,200) | 0 |
| spcd | C 9,947; D 20,000 | 0 | 0 (0) | no pristine reference: C 9,947; D 20,000 |
| vcl-fer | T 264; E 192; S 96 | T 264; E 192; S 96 | T 264; E 192; S 96 (552) | 0 |

Totals: covered memberships TRAIN 224,207; EVAL 46,320; TEST 7,526. Delivered pairs TRAIN 336,130; EVAL 77,684; TEST 3,246. No-reference skips total 90,979; decode failures 30; unresolved 0. `FEATURE_COVERAGE.json` binds the final per-collection bank manifests, key joins and counts.

## Checksums, mirror and temporary PNGs

The final inventory has 375 entries, all verified on tower. The pre-receipt artifact set was 374 files / 2,688,928,711 bytes; the final inventory also includes the mirror receipt. The inventory itself is bound by its SHA-256 below. Copying and remote hashing used a bounded resident container without network access or image pulls. Three random readbacks also matched:

| Random readback | SHA-256 |
| --- | --- |
| `bank/_work/cid_iq/chunk_000.csv.research_manifest.json` | `3dd6dc615e6d4f05d491633f0105deee63928d8df5a1a7d49b9fa7c2a226e23b` |
| `bank/_work/iqa_odi/chunk_000.audit.jsonl` | `958aed9c3e66ec3f9ca274ab948935c27c21f7038c1e4061fe7ac681cd51016f` |
| `DECODE_FAILURES.initial-batch.bak.json` | `a1c67386003be038796e69d4cf06b2f55bb8460b55bcb3c6d9424c13add76474` |

| Receipt / inventory | SHA-256 |
| --- | --- |
| `SHA256SUMS` | `ba20d1b8f283edd00f404e67b3c516f23db0706501a6028b2de63cf4971a6463` |
| `MIRROR_VERIFIED.json` | `aae8b9c098038d4031f7ef7cd64c72589def954499b0fb0ac07009c465d8797a` |
| `FEATURE_COVERAGE.json` | `0e140887799c9468adf4374eb5c2778af07345212833a8aca657414d3d4abb41` |
| `BAPPS_SOURCE_BINDING.json` | `c8e7212cdbd17c84baa4f740edb2effe550c24a3402e49e12ad0dcaabc01172c` |
| `DECODE_FAILURES.json` | `b3113944254f7dbda809728ffdd750926ee023db7d8a693f2e5526b2082d88fe` |
| `README_EXPOSURE.json` | `0ecb131b05091bc23ba46a4df491032b6a6968519d1811ff3130b068807a57e9` |
| `EXAMPLE_EXPOSURE_IDENTIFIERS.json` | `193bc380577e1718df09026607efa93f80e7c9bb27ff2e637601bb8fcd2bb65a` |

BAPPS required 547,041 original PNGs. Each matched the provided library index; both retained archive hashes matched, and 729,388 native file-hash bindings matched the unpack receipt. No judge arrays were opened and no PNG was re-encoded. Original archives, packed library content and all hash/native audit receipts remain intact. Owned unpack scratch was removed only after the complete feature root was mirrored and verified. The adjacent removal receipt is `/mnt/tower/output/zensim/dataset-features-rev7-20261025-scratch-removal.json`, SHA-256 `1302320fd59f8e8742f27a231fd15652bc20793a5e1c8ca5eaba1051ff013042`; it is outside the immutable feature root.

## Scope, roles and proposal

Only the originally provided library is in scope. LS2N IPI SR and ESPL were the only successfully fetched post-library collection payloads identified by the acquisition-log/holdings audit; both are excluded and their files remain untouched. Pre-correction LS2N key metadata is retained unused, with no feature bank. SCID, SIQAD, QACS and JPEG-AI EPFL are closed as not in the library, with no retry. No dataset downloads, author/mirror/FTP probes or web searches followed the owner's correction; final protocol evidence uses retained local papers/READMEs only.

Protected CID22 gold, CIQA, MCL-JCI/D3, AIC, KonJND JPEG, CSIQ, LIVE and CLIC 2022 retain their roles. NITS is T0 EVAL-only; CLIC 2024 remains source-exposed diagnostic. SPCD remains CALIBRATION / source-exposed diagnostic despite feature skips. LocVis duplicate rows, automatic/teacher maps, temporal flicker, ambiguous IQA-ODI ID233, inherited held-out families and Colourlab chroma's unresolved CSIQ-parent calibration rows remain excluded. No accepted membership is rewritten here.

Membership registration landed on main as `c3dc1a1ad64109d0cf5120fa531ecdc4065174c9`, with documentation-exposure additions `65ab5067567806489c93e6a0dd052226bfbfd52b` and `d5a2949b9505a16a09592da900eabc351a401bc2`. See [DATA_SPLITS.md](https://github.com/imazen/zensim/blob/main/docs/DATA_SPLITS.md) and [DATASET_HISTORY.md](https://github.com/imazen/zensim/blob/main/docs/DATASET_HISTORY.md) on main; the proposal is separate and awaits coordinator approval. It covers all 17 in-scope frozen fit populations, distinguishes native protocols, and states where existing trainer semantics do not fit. This feature delivery grants no label access, fitting, calibration or assessment permission.

Resource serialization used one shared-lock acquisition per collection, smallest first, with separate acquisitions for final checks, mirror and cleanup. BAPPS wrapper measurement: `run-heavy: done rc=0 2660s | peak-RSS 3.48GiB | min-avail 42560MiB | peak-load 11.95`. Owner-produced bank hashes and complete role coverage passed; all dataset helper scripts pass Ruff. No Rust implementation or public API changed in this lane.
