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
| [Rousselot 4Kdtb](https://doi.org/10.3390/jimaging5010018) | 8 references; 96 distorted images | Previously evaluated; archive/page disagree on CC BY version. October 29 owner decision permits source-disjoint TRAIN/EVAL admission with prior exposure disclosed. |
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
Existing T0/eval-only memberships remain fixed except for the two explicitly
authorized October 29 collections below. UPIQ-380's existing
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

**Owner decision (October 29):** "for hdr, we probs need human training data, as well as eval sets".
This supersedes the October 27 eval-only role for `rousselot-4kdtb` and
`itm-color-graded-hdr-images` only. Both require source-disjoint approximately
70/30 partitions and historical-exposure disclosure. Every other human HDR
collection retains its existing role; non-commercial/unstated licences,
including CHUG and Rousselot HDdtb, permit validation only. This resolves
permission, not native-input fidelity, source screening or completed admission.

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

## Prospective gate contract, before fits

This contract applies separately to all six path/role combinations. A
missing measurement is INCOMPLETE. No candidate is selected on EVAL, TEST,
TERMINAL, the frozen video sets or the T0 portion of UPIQ. Exact admitted memberships and native-input contracts must be frozen before
any training plan executes.

### Reproduction, features and serving

1. Reproduce a stored Mlic control cell with the existing fx2-study control
   owner before interpreting any comparison. Predictions must match as
   bytes. Model bytes must match after the existing native tool strips
   only `zentrain.repro`; preserve both raw artifacts and their hashes.
   A reused claim of a prior control passing is insufficient for this run.
2. Re-extract stored revision-7 source rows using the declared request and
   compare every consumed feature bit. Record extractor source/binary,
   feature-set identity, native input hashes, row order, display/transfer
   model and strided dimensions. New HDR rows need the same explicit
   revision/read-set contract; no column-count compatibility inference.
3. Final assessments execute the packed Rust serving surface, including
   derived-input declarations and TRAIN-only calibration. Expose fx2's
   complete consumed inputs in assessment views. Keep uncomputed slots
   absent/NaN; do not borrow the fx1 assessment read set.

### Inherited SDR and correctness gates

Retain the five goal-aligned families in the by-construction record:
certified monotonicity/identity; robustness and local/global mixtures;
within-reference ordering/dial/near-identity; codec bias/smoothness; and
steering. Preserve the existing native owners and numerical expectations.
Assess against E for the general role and ECodec for the codec role.

- Certified monotone in the packed artifact, with the existing identity pin.
- C1–C6 on both standard and ladder grids, plus G-DIAL, using the existing
  contract fields and thresholds. A passed subset is not an overall pass.
- G-STEER: retain every one of the baseline's 135 passing cases.
- N1–N3: retain the existing contracts, including N1/N2's 99-point bars and
  N3's 122-of-144 monotone-ladder bar. Report the baseline's actual count
  separately; do not relabel the baseline count as the historical threshold.
- Identity: all 62 sources at all 12 registered SIMD permutations, exact
  identity score and consumed-feature consistency.
- Zero global-mixture inversions; report every local inversion, largest
  rise, source and seed. Keep fold and full-data mixtures separate.
- Complete TRAIN-source-fold scorecards: per-reference/per-distortion
  ordering, codec bias, Z-RMSE, knob-concurring inversions and geometry.
  Report pooled E21 as a diagnostic alongside the goal-aligned evidence.
  The codec role's older-set evidence is compression-only, while reporting
  all-type changes separately, as in its original registration.

For the added SDR no-regression check, use the existing E26 as-good rule
against the finalized role-matched baseline: paired signed SROCC mean
change >= -0.002, each source-population change >= -0.005, and W2 > -2 SE.
Run the corresponding omission/source-fold cells with seeds 0, 1 and 2;
do not substitute fitted-source results. Retain the fx2 continuation
CID22-A bias-spread allowance of +0.005. Record all comparisons and failed
checks. These checks add protection; they do not weaken inherited gates.

### HDR reporting and independence

Freeze each assessment's exact source families, original role, score
orientation, protocol, viewing model, calibration and population hashes
before scoring a candidate. Report paired deltas against the role-matched
baseline, per-seed scores and whole-source uncertainty. Human labels,
metric-teacher labels and chroma-ladder teacher ranks are separate evidence.

| Population | Required readout and restriction |
|---|---|
| New permitted human source-held-out EVAL subsets | Signed SROCC and raw PLCC per dataset; source/protocol-specific JOD or BT additionally needs within-source ranks. Preserve official TEST labels as TEST. No EVAL-fitted logistic mapping. |
| UPIQ-380 | Existing fit/development split and TRAIN-side source folds only; Narwaria/Korshunov separate. No independent-test claim. |
| UPIQ outside 380 | T0, report-only; preserve SDR corpus overlaps and explicitly disclose HDR-VDP-3 teacher calibration on UPIQ. It is not an additional independent HDR image test. |
| Registered HDR teacher VAL | Within-reference rank and pooled score agreement per teacher. Never choose leg weight/checkpoint on VAL. Preserve HDRTEACH agreement membership and existing target transforms. |
| HDR-VDC / AVT | Frozen stored-artifact route only, separate datasets and display conditions, source/video-level grouping. No re-decoding and no Rev5-to-Rev7 feature relabeling. Missing matching stored Rev7 features means INCOMPLETE, not permission to reuse old columns. |
| Held-out HDR chroma ladders | Per-source/per-view rank agreement with natural SSIMULACRA2, inversion/tie counts and paired baseline changes; no pooling unrelated source score offsets. Keep the lane's existing source-role and admission rules. |
| Tone-mapping and raw SI-HDR diagnostics | Report their cross-domain/rated-stimulus limitations separately; they cannot stand in for exact native-HDR compression targets. |

The owner scope decision is resolved. Freeze the following additional decision
rule before fitting: no single human collection is an automatic winner or veto.
Report every dataset's signed SROCC/raw PLCC, with source-family paired
resampling (all frames of a sequence together). Undefined correlations and
insufficient family counts remain explicit missing uncertainty, never zero.
Do not tune on these EVAL readouts. Qualification requires complete inherited
correctness/SDR gates and the full HDR evidence panel; adoption remains a
separate owner review. Historical exposure and HDR-VDP-3/UPIQ dependence are
mandatory on that panel.

UPIQ-380's reserved 50-row development population remains final-checkpoint
report-only under its original admission; it never selects weights, calibration
or epochs. Any new UPIQ weight selection uses source folds inside the 330 fit
rows. HDRTEACH VAL and hdr_v3mix VAL likewise stay report-only.

TRAIN-side selection is lexicographic: reject any cell failing an inherited
correctness or SDR no-regression gate; then maximize the unweighted mean of
within-reference HDR rank correlations across admitted human, ladder and
teacher TRAIN-development legs. A leg with fewer than two development families
cannot select a weight and is report-only for that fold. Break exact ties by
lower added HDR weight, then lower seed-independent configuration identifier.
Report all eligible and rejected configurations. Seed 0/1/2 means and whole
source-family resampling assess stability; EVAL never resolves a tie.

For P3, each corresponding TRAIN-development SDR and HDR population must retain
its selected P1/P2 score within the already registered no-regression allowances
(mean -0.002, population -0.005, W2 > -2 SE). Report any infeasible combination;
never weaken a gate or change the frozen recipe after reading EVAL. The external
human panel remains descriptive evidence, as required by the October 27 rule.

### Plan and execution admission

Before fitting, freeze each exact leg/weight grid, TRAIN-side source folds,
seeds, initialization, epoch budget, fixed checkpoints, sampler policy,
target transforms, input/model/tool hashes and gate configuration in a
manifest carrying `build_commit`. TRAIN-side validation alone selects
weights for P2/P3. P3 must retain P1's SDR gain and P2's HDR gain on those
development folds; frozen external assessment follows selection.

Features run through zenfleet. Training uses the shared heavy-work lock
and resource wrapper. Resolve live workers and queued jobs first. Preserve
run directories, incomplete/failure states and source/binary identities;
never report a queued stage as an active fit. No job has been started by
this admission record, and neither P1 nor the full HDR ladder root was
available in the producing lanes' reports at this registration.
