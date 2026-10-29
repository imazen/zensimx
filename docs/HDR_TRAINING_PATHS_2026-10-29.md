# Revision-7 HDR training paths: admission and inventory

Registered 2026-10-29 before any new subjective-label or pixel read, feature
extraction, fit, checkpoint selection or evaluation in this study.

**MISSING:** new source memberships and official-split verification, native
revision-7 feature banks, exact reproduction control, P1's completed root,
the full HDR chroma leg, and P2/P3 results. This is an admission record,
not a completed split or training result. E and ECodec remain unchanged.

## Three paths and two roles

- P1: the finalized recipe plus full-resolution chroma inputs (`fx2`) and
  the SDR chroma-ladder leg. The existing chroma lane owns this path;
  consume its frozen root/results instead of duplicating its work.
- P2: the finalized recipe plus admitted human HDR TRAIN rows, the HDR
  chroma-ladder leg and permitted existing HDR teacher TRAIN rows. Scale-0
  chroma inputs are allowed. Preserve every existing development/VAL role.
- P3: combine P1/P2 legs; choose weights using TRAIN-side validation only.

Each path has a general-model and codec-model arm. Starting artifacts are
the finalized E (`7642eff4…`, continued with rehearsal through epoch 59)
and ECodec (`63a1c5d3…`, codec continuation epoch 29), as recorded in
[the by-construction record](../benchmarks/byconstruction_2026-10-22.md).
The frozen base root has SHA-256
`a7fd4af0c2bb5a8981be636e5c1cb1a51db57496b7c60d7bf6da01d71dd16952`.
No frozen root or prior run may be edited.

## Metadata-only inventory

The committed catalogue/adapters at zenpapers
`5a6be41bfccc4b79832d32f76f6c2b673d2893a4` and the existing library
README/MANIFEST metadata were read. All nine entries have catalogue status
`organized`. No missing/pointer-only collection is admitted. Counts below
are library metadata counts, not new payload measurements.

| Collection | Available population | Admission constraint |
|---|---|---|
| [UPIQ](https://doi.org/10.17863/CAM.68372) | 4,159 conditions; HDR subset 380 | Only UPIQ-380 is T2 TRAIN. Retain its existing 330 fit / 50 development observations; all other UPIQ remains T0. |
| [SI-HDR](https://doi.org/10.17863/CAM.87333) | 27 rated scenes; six reconstruction methods at two exposures | Released raw reconstructions lack the camera-response correction used for the rated stimuli. Historical raw-output diagnostics do not establish training-target fidelity. |
| [IRCCyN HDR-JPEG](https://doi.org/10.5281/zenodo.21408688) | 10 references; 140 compressed images | Narwaria's UPIQ subset, not an additional independent collection. Preserve its UPIQ source roles and disclose inherited exposure. |
| [Rousselot 4Kdtb](https://doi.org/10.3390/jimaging5010018) | 8 references; 96 distorted images | Previously evaluated; archive/page disagree on CC BY version. October 27 evaluation scope must be reconciled before any TRAIN assignment. |
| [Rousselot HDdtb](https://doi.org/10.23919/EUSIPCO.2018.8553212) | 8 references; 96 distorted images, naive/expert protocols | Licence unstated; existing policy permits validation, not training. Keep protocols distinct. |
| [Ghent colour-graded ITM](https://doi.org/10.5281/zenodo.7669815) | 12 rated scenes; 108 HDR outputs; 432 comparison pairs | JOD is centered within scene, not a shared absolute-quality scale. Source metadata names SI-HDR scenes 066 and 195: reserve cross-dataset families together before splitting. |
| [TMQI-DB](https://doi.org/10.1109/TIP.2012.2221725) | 15 HDR sources; 120 tone-mapped images | SDR-display, cross-domain evaluation. Mean rank is lower-is-better and comparable within source only. |
| [TMQID](https://doi.org/10.1109/JSTSP.2016.2637168) | 20 HDR sources; 180 outputs per protocol | Research-use-only licence: no product-training admission under existing policy. Bradley-Terry offsets are source/protocol-specific. |
| [IRCCyN PairComp TMO](https://doi.org/10.5281/zenodo.21411296) | 10 HDR sources; 90 outputs per protocol | Separate with-reference/without-reference SDR-display evaluations; source-specific Bradley-Terry scales. |

None of these nine committed adapters assigns an official `split` field.
That establishes only what the adapters implement; it does **not** prove
that the original publications prescribe no split. UPIQ's and SI-HDR's
shipped README protocols and Ghent's shipped README/source list were read
without opening scores; they state no official train/test partition.
Complete publication/protocol verification remains required for any new
TRAIN admission. Do not read labels to discover a split.

## Split and exposure contract

[DATA_SPLITS](DATA_SPLITS.md), including its October 27 HDR decision, is
the authority over the catalogue's older `not assigned` descriptions.
Existing T0/eval-only memberships never move to TRAIN. UPIQ-380's existing
source-hash fit/development rule is retained, not replaced by a new 70/30
partition. HDR-VDC and AVT-VQDB-UHD-1-HDR remain frozen evaluation-only.

For a newly permitted collection: establish its official split from stored
release metadata/publication first. Honor it if present. Otherwise freeze
approximately 70% TRAIN / 30% EVAL by canonical source-family identity,
deterministically and before target access. All distortions, crops,
exposures, scene variants and cross-collection duplicates inherit their
family's role. Use the existing source admission/screening owners; a
dataset-local filename hash cannot establish family independence. Freeze
the exact member list, rule, producer commit, metadata hashes and excluded
families, then append an exposure entry before any TRAIN label read.

**Unresolved role question:** October 27 designated all scoreable human HDR
sets as evaluation for the frontend and explicitly left training legs
unchanged. The new study requests additional human training splits while
preserving prior eval-only assignments. Whether CC-BY 4Kdtb/Ghent may
receive new TRAIN subsets needs an explicit scope resolution. Until then
they receive no new fit permission. This record grants no such permission.

**Exposure in this lane:** catalogue, adapter source, README/MANIFEST
metadata and Ghent's source-only `images.csv`; no subjective response table,
label payload, objective-score table or image pixels. The historical SI-HDR
and Rousselot exposures stay disclosed; a new partition cannot make them
untouched tests. Narwaria MOS and UPIQ JOD are alternative labels on shared
content, not independent evidence.

HDR-VDP-3 was calibrated on UPIQ. Every student trained on its teacher labels
must identify UPIQ results as non-independent; UPIQ non-380 remains T0 even
when reported with that caveat.

## Native revision-7 route and existing owners

The route exists: `zensim/src/research.rs::extract_hdr` plans the declared
read set, then calls `feature_v2::compute_folded720_hdr_streaming_extras`.
That uses `FrontEnd::Hdr(encoding)` and the shared streaming walk with the
plan's formula revision. Revision-7 basic/peaks/v2 requests are supported;
raw legacy append layouts are not a substitute. Preserve opaque native
PQ/HLG code values or absolute-nits linear float input, declared primaries,
dimensions and row stride. Do not convert to RGB8 or relabel Rev5 columns.

`zensim-validate/src/bin/tier_audit_features.rs` already supports native
PQ `--hdr-pq` and basic/peaks/v2 at revisions 5 and later. This source audit
establishes the available route, not a newly executed extraction/parity gate.

The existing E26 `scripts/rev4_featpot/rev5_bank.py::cmd_hdr_extract` and
UPIQ `scripts/rev4_featpot/upiq380.py::admit_metadata` deliberately require
revision 5. Extend their owners with a separate, explicit revision-7
admission contract; preserve the old refusal tests and frozen runs. Their
old tables cannot feed this study. UPIQ input must still be the isolated
authorized HDR population, never its mixed subjective CSV.

Existing continuation owner:
`scripts/byconstruction/aic_matched_continuation.py`; existing library-bank
driver: `scripts/byconstruction/dataset_features.py`. These are under
`byconstruction`, not `rev4_featpot` as some briefs suggest. Reuse root
composition/selection-rebind in `scripts/byconstruction/ring_root.py`.
