# Activation and shaping completion: execution evidence

Scientific comparisons are in progress. No quality winner or completed
source-generalization result is claimed here. The
[registered protocol](../docs/SHAPING_COMPLETION_PROTOCOL.md) and
[implementation qualification](../docs/ANALYTIC_HEAD_IMPLEMENTATION.md)
separate this work from the recovered November 6 studies.

Registration at `1e1df57a` verified the original inputs and source memberships.
Its PLAN SHA-256 is
`bad4e75b15cbcceedd7906096ded2e60eaf2c7ff706c51383607464201d67976`.
The matrix contains 1,879 required fit cells, 720 conditional equal-opportunity
neural rescue cells and one resource pilot. Training-only pretraining is charged
against exposure. The prior exposed development partition is excluded from
training commands and selection. All scientific completion counts remain zero
until native qualification and the required evaluations are inspected.

The frozen release binaries were built from `1e1df57a`, with native shaping code
at `2c68977e`. The build used thin LTO and four build jobs on i270.
Wrapper: `run-heavy: done rc=0 66s | peak-RSS 1.38GiB | min-avail 24715MiB | peak-load 2.76`.
The new binary set reproduced all three frozen native control bakes exactly
(after the established removal of repro metadata): single, shared ReLU and
shared SmoothHinge. Wrapper:
`run-heavy: done rc=0 11s | peak-RSS 0.16GiB | min-avail 44453MiB | peak-load 10.54`.

The registered one-epoch resource pilot used the first inner ReLU partition,
50,000 pair draws and setting 0. It is excluded from scientific selection.
Native wrapper:
`run-heavy: done rc=0 18s | peak-RSS 0.09GiB | min-avail 45789MiB | peak-load 0.46`.
The reported training artifact inventory before adding the result receipt was
606,443 bytes. These are measurements of this pilot, not estimates for other
families or whole runs. The initial orchestration attempt selected a nonexistent
default debug binary; it exited before training. Its failed receipt/log is
preserved, and execution now verifies the selected command against the admitted
binary directory before launching.

At the November 7 execution check, 12 ReLU cells were training concurrently
(four settings and three paired seeds on inner-0-0), together with four
interleaved analytic cells. Every native process has one thread and a 1 GiB
memory cap. Analytic admission alternates Softplus, monotone SiLU and unrestricted
SiLU across paired seeds. The scientific budget and selection rules are unchanged.
This is a dated observation; inspect current processes and checkpoints with:

```
just shaping-completion-status /mnt/v/output/zensim/shaping-completion-20261107/study
```

The status command checks each recorded PID's command line as well as terminal
receipts and epoch logs; a progress file alone is not evidence of a live fit.
The frozen registration, binary set, successful/failed pilot attempts and new
baseline replay have checksum-verified archives. Exact manifests and receipts
are under the corresponding local/NAS `shaping-completion-20261107` roots.
The complete study, shaped-family execution, convergence investigations,
source-fold reports, learning curves, shuffles, human-only trials and finalist
inference measurements remain required.

## Completed shaped-additive inner tuning

All 24 registered quadratic inner cells completed: four regularization choices
across six source-disjoint inner partitions. Every cell passed the unchanged
native KKT/gap certificate, independent basis and f64 prediction checks, and
f32 identity/damage probes. No outer results were read.

| Outer fold | Selected lambda | Mean inner human composite |
|---|---:|---:|
| 0 | 0.01 | 0.870036250 |
| 1 | 0.01 | 0.850801756 |
| 2 | 1e-06 | 0.869986132 |

Each partition ran its four solves sequentially under a four-GiB controller
and separate four-GiB native solve cap, with one thread per process. The six
controller run-heavy receipts report 15, 13, 13, 15, 14 and 13 seconds,
respectively; measured peak RSS was 1.02 or 1.03 GiB per controller. These are
training measurements, not runtime benchmark claims. Full logs and native
certificates are retained with the artifacts.

The completed quadratic wave archive contains 245 artifact files totaling
610,412,250 bytes. All destination artifact hashes passed client readback
after fsync and POSIX_FADV_DONTNEED; this does not claim physical NAS cache
eviction. Manifest SHA-256:
`01f1ce9634b8164f6d1821061ca1b7990ade5fbb33585e987a64153f01e0df5f`.
Archive suffix: `shaping-completion-20261107/linear-inner-archive`.

## Convergence evaluation implementation

The evaluator accepts only completed registered fits and final-epoch selection.
Registered intermediate epochs produce separately marked diagnostic reports;
they cannot become selection candidates. Native inspection records output
coefficient counts. The convergence checker requires all scheduled loss epochs
and all registered checkpoint panels before producing a receipt. It compares
the latest available pair with matching epoch modulo 50; for continuous fits
that is epochs 119 and 219, not mismatched learning-rate phases 199 and 239.
The native four-decimal loss log has a 0.00005 rounding bound; a one-percent
threshold interval crossing zero triggers investigation. The guard tests pass;
actual completed-fit curve evaluation remains to be exercised.

## Remaining optimizer-diagnostic qualification

Source inspection found that `monotone_research::report_diagnostics` and its
accumulators require `config.training`. The initial analytic scientific
manifests omit that object, so these runs do not log the implemented gradient,
hidden-activity and projection diagnostics. Saved native checkpoints support
coefficient inspection, but do not reconstruct historical projection counts.
Preserve these fits and qualify a diagnostic replay before using those traces.
Plain shared-ReLU initialization and consolidation rescue controls also require
native implementation; the existing training-research configuration is attached
to a research head and must not substitute a different head for that control.

## Local Gram reclamation, November 7 UTC

Under the completion goal's storage authorization, all 264 historical
`fits/*/gram.npz` files were reread locally and from the durable mirror against
manifest `642428868aada79b69ee4358b14e01a279b7c068e6dd1f0bdea2e478de7b8882`.
Open-file, memory-map and active-study-pin checks found no users. Only those
local copies were removed, reclaiming 7,177,361,457 bytes. Reports, models,
predictions, fit specifications and active inputs remain local. The original
manifest now describes the complete durable archive; the local tree is partial.
`LOCAL_ARCHIVE_STATUS.json` in the historical root gives the restoration root,
and `shaping-completion-20261107/RECLAIMED_OLD_GRAMS.json` lists every removed
relative path, hash and byte count. The durable archive was not changed.

## Orchestration-source receipt precision

Early Python controllers computed `driver_sha256` by reading their source file
when writing each receipt. For a process that stayed alive across a source edit,
that field identifies the file at receipt time, not necessarily the code already
loaded in memory. Preserve those original receipts with this qualification.
Their exact native command lines, input pins and immutable binary identities
remain the reproduction authority. New controllers capture the driver hash at
module startup. This change does not alter scientific commands or checkpoints.

## Complete inner control selection

The selection owner recomputes the equal-human composite from four populations,
requires all four settings, both inner folds and all three seeds (one seedless
solve for quadratic controls), and refuses outer-exposed or incomplete reports.
The selected tree setting for outer folds 0 and 1 is min_samples_leaf=8 and
max_features=0.33. Its mean inner human composites are 0.917060008 and
0.889150989, respectively. These are inner tuning measurements, not outer
quality claims. Quadratic choices reproduce the table above. Selection receipts
retain report hashes and keep outer evaluation closed pending neural convergence
and rescue resolution.

The first fresh-head control wave uses the registered 120 epochs and 50,000 draws
per epoch. Inspected command lines for seeds 0 and 1 use initialization/sampling
pairs (1001101,1000101) and (1001107,201000101). These controls have six million
scheduled draws and remain separately identified from primary twelve-million-draw
fits. Two one-thread, one-GiB native jobs run concurrently under a two-GiB
controller cap, after the verified Gram reclamation restored storage headroom.

The storage reclamation receipt itself has a verified durable mirror:
`ed94eb9b0ff4748a0d90675ac16c7de21b37d32a25a1017cda6796746a8b2a9f`.

## All tree inner cells evaluated

All 72 registered tree inner cells passed the exact native export gate and
human/teacher panels. Settings selected from complete inner evidence are
(leaf=8, features=0.33) for outer folds 0 and 1, and (leaf=8, features=1.0)
for outer fold 2. Their mean inner human composites are 0.917060008,
0.889150989 and 0.905468636. Outer results remain closed.
The six controller run-heavy receipts report 324, 291, 264, 323, 253 and
365 seconds, each with measured peak RSS 0.39 GiB. These are training/evaluation
measurements, not inference benchmarks.

## First complete neural fit: ordinary SiLU diagnostic

Cell `184c4027e954a4c3eafc`, inner-0-0, setting 0, seed 0, completed all 240
scheduled epochs. Native final export and finite-output probes passed.
Its final-epoch human composite is 0.912114887; this single tuning cell is not
a family comparison or outer generalization result. The final checkpoint remains
the selection candidate even though earlier diagnostic checkpoints score higher.

| Population | Inner pooled signed SROCC | Inner within-source signed SROCC |
|---|---:|---:|
| kadid | 0.880458712 | 0.904324169 |
| tid2013 | 0.880485119 | 0.909956034 |
| konfig | 0.930216490 | 0.903411967 |
| cid22_a25 | 0.924434432 | 0.963632174 |

Teacher agreement, excluded from that human composite: pooled 0.985546444,
within-source 0.990438656. The ordinary diagnostic makes no identity/order
claim; its probes recorded identity 103.413970947 and 59 damage-direction
increases. All seven inspected checkpoints retained 64 nonzero output
coefficients. The final twenty mean logged losses were 0.160995, compared
with 0.223560 in the preceding twenty. That exceeds the registered loss
improvement trigger even accounting for log rounding. Same-cycle inner
composite change (119 to 219) was -0.000426859.

The outer-0 rescue condition is therefore active: all 240 registered rescue
cells for that fold are required after complete base selection. No family is
rejected from this fit. A separate full diagnostic replay is running; its
traces require canonical model equality before attribution to the original.
The completed original and all checkpoint panels are archived with manifest
`db7c1b9fbdcd09b81a8876e31e48bf5ec331147d8f5522fc9eb130b7589a93bc`.
Native fit wrapper: `run-heavy: done rc=0 2840s | peak-RSS 0.09GiB |
min-avail 41595MiB | peak-load 22.27`.

The complete tree inner wave and its three selection receipts are mirrored
and reread: 584 artifact files, 3,251,255,749 bytes, all hashes matched.
Archive manifest: `2e7685d7746262d40a42a242358a1084f39c4d839e9944c3b5a72c71acb0213f`.

## Registered subset admission

The supplementary membership policy was frozen before variant fitting. Its
registry covers 672 cells using 33 distinct row memberships. Two quarter-source
samples lack one human training population; preserve those hash-selected
samples and report unavailable calibration errors explicitly. Rank evaluation
on the fixed test populations remains available. The default evaluator still
refuses undeclared missing training maps.

Actual preparation and strict table admission passed for an outer-0 human-only
view and the outer-1 quarter-source view lacking TID2013. The former contains
all four human populations and no teacher group; its rank-only recipe uses
MSE weight zero. The latter contains three human groups and the teacher.
All generated table, key, manifest and admission hashes were rechecked.
`study/VARIANT_ADMISSION_QUALIFICATION.json` preserves the per-group evidence.
Two population-admission tests and two absent-map tests pass. These are input
qualification results, not completed variant fits or generalization results.

The variant registry and admission evidence archive verified all five artifact
files (7,580,046 bytes). Manifest SHA-256:
`bfd4ad5273825f11406cc0e0b8427a2a0c651d088e7417bc0892882feb109753`.
Suffix: `shaping-completion-20261107/variant-registration-archive`.

New neural controllers reserve pending checkpoint space before every fit, in
a short admission transaction. The allowance is one GiB per unresolved attempt
plus three GiB for auxiliary output above the frozen 30-GiB reserve; these are
conservative admission budgets, not measured output forecasts. Below the
budget, admission waits for archival/reclamation without holding a compute
lock. Reservations deduplicate running attempts and are released on normal
exit or exception. Unexpected controller death retains its reservation for
inspection. Three tests cover unresolved attempts, overlapping reservations,
exception release, and waiting until sufficient space exists. Previously
started controllers retain their loaded implementation and require continued
manual headroom supervision; they have not silently adopted this change.

## Rescue admission implementation

Rescue setting resolution now rechecks the frozen matrix, triggering convergence
receipt, complete paired inner reports, report hashes, and each selected family's
convergence evidence before resolving `base_selected`. It recomputes the human
selection rather than trusting the saved winning index. Two tests exercise
complete coverage, missing reports, modified evidence and optimizer settings.
No rescue fit has started.

Analytic rescue controls use the existing training-research owner, which requires
exact admitted row-family groups and one teacher. Empty groups are rejected by
the Python builder; native code independently checks names and row counts. This
owner materializes the event stream, unlike independent diagnostics; its actual
registered-exposure resource use and neutral replay must be qualified before
scientific rescue launch. Pretrained rescue execution and nonmatching-schedule
pretraining remain implementation work.

The actual analytic neutral-control qualification reproduced epoch zero of the
completed ordinary-SiLU fit exactly after removing only repro and declared
research-configuration metadata. Native weights, scaler and retained metadata
hash: `74edb955ecaca2022eb93d0b75df7455011351988416ae5d98f576d82881491b`.
It used the same admitted rows and 50,000 draws with random order, unit teacher
mass, no dropout and no balancing. Uniform family codes are a neutral-control
fixture only. Native wrapper: `run-heavy: done rc=0 20s | peak-RSS 0.09GiB`.
The enclosing wrapper measured 21 seconds and 0.09 GiB. This one-epoch result
does not establish memory use at 240 epochs or full-run equivalence.

## First ReLU checkpoint feature-gate failure

Cell `6df732ab294270460360` (inner-0-0, setting 0, paired seed index 2)
completed training. Its final human inner composite is 0.9197590984467099 and
its final feature probes pass. Epoch 119 fails the unchanged damage-order
gate: one increase of 0.0000152587890625 exceeds 0.00001, while identity is
100 and predictions are finite. This cell is not scientifically complete;
the failed checkpoint, predictions and gate receipt remain intact.

Supervisor v2 stopped on that measured failure. V3 records failed checkpoints
as investigation requirements, emits no convergence/selection completion for
the affected cell, and continues other checkpoints and independent fits. A
control test verifies that failure remains visible and cannot count as
completion. No feature threshold or scientific expectation was relaxed.

For the seed-2 failing pair (probe 29), native predictions move from
-70.53814697265625 to -70.53813171386719. An independent f64 evaluation of
the stored f32 layers gives -70.53811054625083 and -70.53811059123643, a
decrease of 4.498559746934916e-8. Derived inputs 152 and 644 change. The
first layer has opposing activation changes; the f64 second layer has no
decreasing activations. This supports numerical cancellation as the current
investigation direction, not a proven native root cause or a qualified repair.
The native gate remains failed. The trace uses the existing native model
inspector and a separately labelled f64 algebra reference; it is not a
replacement scoring implementation.

The follow-up scalar reference uses libc `fmaf`, the stored f32 scaler and
the predictor's bias-first SAXPY order. It reproduces both failing native
scores bit for bit. The second layer contains an activation decrease of
0.00048828125, while its f64 counterpart has no decrease. Tied signed
first-layer groups have nondecreasing summed f32 activations in this pair;
accumulating their separate terms in the middle layer nevertheless reverses
the ordering. This localizes the observed reversal to rounded dense evaluation
of the tied-pair representation. It is not a repaired or passing model.
The scalar diagnostic wrapper measured 1 second and peak RSS 0.15 GiB.

## Full unchanged-fit SiLU diagnostic replay

The ordinary SiLU inner-0-0 setting-0 seed-0 replay completed all 240 epochs.
After removing only `zentrain.repro`, both final models have canonical SHA256
`499e335ce89b363d879b2404833db985847ac36f368d9058cb88ef5d9dd279fd`.
The diagnostic traces therefore correspond to the original fit; this replay is
not an additional selection candidate.

At epoch 239 all 64 output coefficients remain nonzero, with zero output
projection hits. All 64 final hidden units are active on at least one of the
386 sampled training responses. The sampled first-layer values include
47,683 exact zeros among 49,408 values. Their squared gradient norm averaged
per sampled response is 0.05944560457827638, and the regularized output-gradient
statistic is 1458.6049832207204. These observations do not establish globally
dead units or convergence, and do not explain the separate constrained
Softplus collapse; that unchanged-fit replay remains in progress.

Native wrapper: `run-heavy: done rc=0 2986s | peak-RSS 0.09GiB | min-avail 37941MiB | peak-load 23.43`.
All trace epochs, checkpoints, commands and original/replay model identities are
preserved in archive suffix `diagnostic-184c4027e954a4c3eafc`.
Destination readback passed manifest SHA256
`684b38f5997461f9c76dbf3b052167b8c6340a5c712ceb235a27c0ff30e0c2ca`.

## First completed jointly learned shaping pair

On inner-0-0, setting 0 and paired seed 0, the completed product-shaping fit
has human composite 0.9245500073515289 with 44 nonzero output coefficients;
the separate damage/fragility fit has 0.9201898276009219 with 40. Both pass
all scheduled native panels and unchanged feature gates. Both trigger the
registered loss-improvement investigation. These single-fold, single-setting,
single-seed observations are not an architecture comparison or evidence of
convergence. Paired tuning and rescue resolution remain required.

The complete parent fits and their epoch-119 prefixes are immutable before
pretrained children start. Frozen and joint product-shaping children started
in attempt 2 after exact-f32 knot validation repaired the serialization check;
the attempt-1 failures ran no optimizer steps and are preserved. The separate
shape children use the same completed-parent requirement.

The first uniform repaired-ReLU wave has 104 passing final/checkpoint panels
across 12 base fits and two fresh controls. Original failed gates remain intact.
Its archive includes the separately pinned model representations, reports,
convergence records and the refusal to select on incomplete inner-fold coverage.

Storage admission held a child while pending-write reservations would have
threatened the 30 GiB reserve. All 60 historical tree parity-input NPZ files
were rehashed locally and at the existing durable mirror, checked against
active input pins and live file descriptors/maps, then removed only locally.
Measured reclaimed bytes: 2,713,454,008. Models, reports, predictions,
specifications and active input arrays remain local. The historical local tree
archive is now explicitly marked partial; its durable archive is intact.
Wrapper: `run-heavy: done rc=0 28s | peak-RSS 0.02GiB | min-avail 40134MiB | peak-load 22.19`.

Verified product shaping archive manifest SHA256: `20b9c31bdb5cf03765f16857e6b6fae31bed698e4c22897a0bb6fd4a191d7e30`.

Verified separate shaping archive manifest SHA256: `4565ad00600c15a0d836d8887816e2dae9c311b2da4cd958adeea7bbe0224252`.

Verified uniform ReLU re-export archive manifest SHA256: `89a47c3bb2acf8ae09b7a0526a5b419ffa6d742244fab4f9e5b621dfccbd2290`.

## Full analytic neutral-control qualification

The neutral analytic training-control route replayed all 240 epochs and
12,000,000 registered pair draws of the completed ordinary-SiLU fit. Original
and replay canonical models match SHA256 `1b8b5a46a19501c142ab5941a40de485add3425a926e55ca67e959567ec59672` after
removing only the declared repro/config metadata. The full precomputed event sequence
therefore qualifies beyond the earlier one-epoch prefix; the scientific fit
itself is unchanged and this replay does not become a selection candidate.

Native wrapper: `run-heavy: done rc=0 3186s | peak-RSS 1.27GiB | min-avail 37988MiB | peak-load 25.58`.
This measured peak is why rescue analytic phases receive a 2 GiB native cap.
The completed replay and checkpoints passed destination readback against manifest
SHA256 `3ccbcf768644d28c6c70ba5e778a6edc246695553b80ecba9b0b1421be89791d`.

## Additional qualified shaping worker

A frozen source snapshot at `39beed49e651` and the existing study binaries
were copied to i270. Every copied input was SHA256-verified. The first native
replay refused missing recipe provenance before training; the preserved retry
included the original receipt and human-role decision. All three native
compatibility models then matched their canonical hashes exactly. Only the
declaration's filesystem path changed, with identical declaration bytes.

Replay wrapper: `run-heavy: done rc=0 9s | peak-RSS 0.16GiB | min-avail 26872MiB | peak-load 0.27`.
This is compatibility/resource evidence, not an isolated runtime benchmark.

The worker runs the 24 registered inner-0-1 jointly learned product/separate
shaping cells, eight native single-thread fits concurrently, with a 12 GiB
aggregate cap and 1 GiB per-trainer cap. Explicit assignment prevents another
host from fitting those same IDs. Source memberships, training-only knots,
seeds, settings, exposure and final-checkpoint rules are unchanged. At the first
live check all eight trainers had advanced to epoch 2 or 3, with no unexpected
stops. Scientific results await completed training and the same native panels.

Worker qualification archive suffix: `worker-i270-qualification`; manifest
SHA256 `ef92f329e39f0830d2ca5c600425953ee9eb86c3be6ba895255524788bb52e5e`.
It preserves copied input bindings, source snapshot, failed/successful replay
commands and results. Active training outputs are not represented as archived.

Completed-worker collection now has a dedicated transfer owner. It packages only
successful terminal fits from the assigned host, binds the complete checkpoint
inventory and source/build/plan identities, and rehashes every received file before
publishing an attempt to the evaluator. Existing cell directories are refused;
failed imports stay in a separate staging directory. The original host and command
remain in RUN.json. Three local controls passed: byte-exact readback and corruption
refusal, traversal/symlink refusal, and terminal/checkpoint coverage admission.
This validates transfer mechanics, not a scientific fit or a completed remote wave.

A supplemental evaluation watcher now applies the same separately pinned native
max-pair representation to newly completed ReLU inner fits and fresh controls.
It uses all scheduled checkpoints, preserves original failed reports, and refuses
selection completion for any supplemental gate failure. Two orchestration controls
passed, including isolation of a failed supplemental checkpoint while the next cell
finishes. The watcher does not select settings or consume outer scores.

Additional registered analytic execution (inner-1-0) now runs on the qualified
worker with eight independent one-thread trainers, alongside its eight shaping
trainers. The assignment binds all 36 existing analytic cells; the immutable
training snapshot and binary set are unchanged. Twenty partition files totaling
23,335,757 bytes passed full remote hash verification before fitting. Native
epoch-zero checkpoints were observed. No new score or selection conclusion follows.

The partition bundle and admission evidence were archived with full destination
readback under `worker-i270-analytic-inner-1-0-inputs`; manifest SHA-256:
`432c3b60b03ded57c26e0984d862db437ace467ee61a545cf2ac8bdeccc47440`. Preparation used 3 s / peak RSS 0.42 GiB; packaging used
3 s / peak RSS 0.20 GiB, as reported by run-heavy. These are resource receipts,
not isolated speed benchmarks. A separate archive supervisor now handles fully
evaluated single-attempt inner cells; it preserves local artifacts and refuses
active, failed, or unreconciled multiple-attempt cells.

The doubled-exposure registration policy now fixes the trigger and pairing before
any affected fits: selected continuing-improvement families plus selected ReLU,
both inner folds and all three seeds. Continuous models receive 480 epochs;
pretrained models receive 240+240 with independent pretraining and fresh heads.
Matched 240-epoch fresh unshaped controls are deduplicated only by identical
settings and optimizer choice. Collapse alone remains a separate unresolved
diagnostic. Three registration controls passed, covering pairing, exposure,
deduplication and incomplete selection refusal. No extension is registered or
fitted until the complete base/rescue decisions exist; the native extension
execution/evaluation integration remains unfinished.

Native extension execution is implemented through the same qualified phase owner.
It reconstructs the registered initialization and optimizer choice, applies exact
fresh-seed offsets, independently retrains extended pretraining, and validates
embedded shape layout. Eight combined extension/rescue controls passed; issued
commands preserve groups, objective, learning rate and paired seeds. The 480-epoch
phase has an explicit 4 GiB native cap because analytic events are materialized;
actual extended-fit resource use is not yet measured. Existing rescue phase
budgets and commands are unchanged. Evaluation/supervision integration and the
first scientifically admitted extension run remain required before qualification.
The new extension diagnostic schedule includes epochs 429 and 479 to compare
its final checkpoint at the same restart-cycle position; no affected fit exists.

Extension evaluation now re-resolves the frozen registration and selected evidence,
checks every completed phase and manifest, and uses the unchanged native feature
gates. The diagnostic schedule compares epochs 429/479 at the same restart-cycle
position for a 480-epoch phase. Original 120/240-epoch schedules are unchanged.
Targeted evaluation, convergence, supervision, extension and rescue checks passed.
Original and supplemental supervisors recognize registered extension cells; the
currently live supervisors retain their previously loaded code and must be
restarted deliberately before extensions enter their queues. No extension fit
or extension native resource measurement has yet been made.

The unchanged-fit Softplus diagnostic replay completed all 240 epochs and matches
the original final model after stripping only `zentrain.repro`: canonical SHA-256
`eb3b1260fb65d4a29b86f4f66f52ee6322ebd25cf4ecc45fc99b5eb1d3aece40`.
The trace reaches one nonzero output coefficient at epoch 69 and retains exactly
one through epoch 239. It records 13,557,450 negative-output-weight projection
hits over the run, with the last hit in epoch 69. These are counts of weights
clipped after optimizer steps, not counts of distinct units or examples.

At epoch 239, 63 of 64 final hidden units are inactive in the 386 sampled training
responses; the remaining unit is active in all 386. The sampled first-activation
gradient squared-norm mean is 0.09522677921903464; the regularized output-gradient
squared-norm mean per optimizer step is 12363.682204193905. Neither statistic
establishes global deadness or causation. This is reproducible optimization
collapse evidence, not proof of an architectural/feature capacity ceiling or a
linear whole network. The registered initialization/schedule rescue comparisons
remain required. Native wrapper: 4751 s, peak RSS 0.09 GiB, minimum available
37225 MiB, peak load 25.58; this concurrent run is not an isolated benchmark.

The older standalone inspector rejected Softplus activation byte 0x4. Its failed
empty output is preserved with an error record. The pinned native bake inspector
successfully read the final model; no training or gate changed to accommodate
the older reader.

A second qualified worker now runs eight analytic inner-1-1 fits. Its 87 shared
files (409,044,047 bytes) and 20 partition files (24,040,656 bytes) passed hash
verification. All three canonical baseline replays matched; wrapper receipt:
9 s, peak RSS 0.16 GiB, minimum available 24684 MiB, peak load 1.32.
The first bootstrap attempt rejected a mistyped expected hash before fitting;
its log remains preserved. The corrected hash and original baseline directory
were verified before admission. Source snapshot and native binaries remain the
same frozen versions used by the other worker.

First completed product-pretrained-frozen cell (`1605f99e85bbbe3cd6e0`,
inner-0-0 / setting 0 / seed 0) passes final and scheduled native gates.
Its human composite is 0.8959433619066952 versus 0.9202173469027812
for the matched fresh unshaped head; both head fits use 120 epochs and the
same offset seeds. The frozen model additionally uses the registered 120-epoch
supervised shape prefix. Its final head has 13 nonzero output coefficients,
and the registered convergence triggers are absent. This is one paired
observation, not a family ranking or proof of global convergence. The continuous
240-epoch joint-product parent scores 0.9245500073515289 and still triggers
loss-improvement investigation; it is not the matched fresh-head control.

The failed pre-training attempt was reconciled against its actual inventory:
keep file, empty checkpoint directory, failure receipt, no training RUN/log.
Both it and the completed retry are retained in the verified archive
`inner-1605f99e85bbbe3cd6e0`, manifest SHA-256
`b92a4cfd00b0dfd07efbf545ec4f3d9351277acd821e77d84165b3b9472d4638`. The fresh head resource receipt is
2956 s / peak RSS 0.09 GiB / minimum available 36977 MiB / peak load 25.58;
this excludes the shared pretraining cost and is not an isolated benchmark.

The progress auditor now distinguishes successful processes, final evaluation
reports, diagnostic receipts, open investigation triggers, original failed gates,
and cell-level verified archives. It checks receipt/report hashes and labels
worker observations stale after 120 seconds rather than treating an old PID
report as live evidence. Controls confirm that a successful process without
reports is not counted as evaluated and that report hash drift refuses the audit.
The first actual audit found 126/816 inner fits with final reports and 4/72 fresh
inner controls; these are not 130 scientifically completed comparisons. The
required matrix currently contains 2,119 cells including 240 triggered rescues.
Outer, curve, shuffle, human-only and full-FIT stages remain unexecuted. Central
control-wave archives are already verified but are deliberately separate from
the auditor's cell-level archive-receipt count.

The matching product-pretrained-joint cell (`070101f07a58f717d498`,
inner-0-0 / setting 0 / seed 0) now passes all scheduled native evaluations.
Its human composite is 0.9022832465164315, against 0.8959433619066952 frozen
and 0.9202173469027812 fresh unshaped. Both pretrained variants retain 13
nonzero output coefficients and have no registered convergence trigger in
this cell. These are one-fold, one-setting, one-seed observations; no family
is selected or rejected from them. Joint fresh-stage wrapper: 3519 s, peak RSS
0.09 GiB, minimum available 37118 MiB, peak load 25.58, excluding its shared
pretraining cost. Both the successful retry and retained zero-step admission
failure are archived under `inner-070101f07a58f717d498`; manifest SHA-256
`bfbc50e3bd93d39eb4cf4be08d64cf4da56170de7e4912f171639a21102d84f5`.

The outer-0 rescue dispatcher now waits for all 24 paired base reports and
diagnostics per family, reproduces the inner selection, and rechecks the
registered trigger before admitting a child. Two native slots have a 6 GiB
parent cap; per-phase native caps remain explicit. No rescue fit is admitted
yet. The original evaluator was retired while sleeping between completed
sweeps with no child processes. Its replacement explicitly distinguishes
primary and optimizer-enabled binary sets and preserves all existing reports.
Targeted dispatch/evaluator checks passed.

Four registered fresh-control inner-0-1 fits now run on the second qualified
worker, sharing its already verified partition. Its twelve-cell queue uses
four workers, a 6 GiB parent cap, and 1 GiB/one-thread per native fit. The
admission archive is verified at `worker-i265-fresh-inner-0-1-admission-r2`,
manifest SHA-256 `2551487803fd285de940c0c3037733550cbe8a0ab210c1a90a6bb8a8ecfcdae2`. The first destination remains
preserved after the two-file supplemental-sampling bug; full artifact hashing
was unchanged by the repair.

## Collection readiness and observed progress at 06:17 UTC

Collection tooling at `581c43d5` reserves worker packaging space and preserves
an exact copy of the frozen derived-input declaration. A separate source capsule
was installed on i270 and i265; both copies matched SHA-256
`ffd9c67f61b650958df61299d6106bdce6227879f28f8b2b5d5adf77d7f6c793`.
The running training capsules were not changed. Its two-file, 4,209,048-byte
archive at `collection-tools-581c43d5` under the completion-study mirror passed
full SHA-256 readback. No remote scientific fit had been imported at this check.

The collection integration fixture now passes the real terminal-receipt and
240-checkpoint coverage validator, packaging, reserved import and atomic
publication. It also proves refusal to overwrite an existing cell and detects
declaration drift after relocation. This is transport qualification, not native
model or scientific qualification.

The 06:17:16 UTC progress audit records 132/816 primary inner cells with final
reports and 6/72 fresh inner controls with final reports. Of the primary cells,
36 have neural convergence receipts, 34 trigger investigation and four retain
original diagnostic-gate failures. These counts do not imply scientific closure.
The cell-level archive receipts cover 32 primary and six fresh cells; control-wave
archives are separately recorded above. Source/row curves, shuffles, human-only,
outer and full-FIT stages remain unexecuted.

Live process inspection confirms 19 local, 16 i270 and 12 i265 fits, with no
unexpectedly stopped attempts. Free SSD bytes were 57,131,855,872,
61,988,958,208 and 219,861,602,304 respectively. The original evaluator,
supplemental evaluator, archive supervisor and rescue dispatcher were verified
against their live process command lines. Rescue admission still awaits complete
paired base-family evidence; no rescue fit was reported as running.

## Generalization input preparation and transport-copy reclamation

The registered input-preparation driver now shares views across model roles and
seeds while retaining permutation type and seed in each view identity. Four
membership/reuse tests pass, and both pre-existing prepared views passed actual
membership and input-hash verification. Scientific fitting and finalist selection
remain separate gates.

Preparation reached its SSD admission boundary after four views. To preserve
pending trainer writes and the 30 GiB reserve, 72 duplicate `model.parity.npz`
files were removed only from local `tree-inner-archive/runs` after verifying each
against the archive manifest, a fresh NAS readback, and the retained original in
`study/runs`. No process held those transport copies. The removed files totaled 2,637,786,497 logical bytes; this did not produce
corresponding free SSD space. Shared filesystem blocks are a possible cause,
not verified. Original study arrays, all models and reports remain on SSD.
The local archive is explicitly marked partial with restoration paths and hashes.
Receipt: `RECLAIMED_TREE_TRANSPORT.json` under the completion-study root.

Reclamation wrapper: `run-heavy: done rc=0 33s | peak-RSS 0.02GiB | min-avail 40440MiB | peak-load 20.10`.
This is a resource observation, not a storage performance benchmark.

A second, explicitly scoped pass verified all 72 completed tree fits and their
final-report pins, then rehashed their parity-input arrays against the durable
archive. No process held the inputs. These local parity inputs were removed;
models, prediction panels, reports and parity logs remain locally available.
Free filesystem space measured immediately before and after this removal was
56,575,799,296 and 59,213,643,776 bytes. Concurrent writers remain active, so this
is an observed filesystem change, not a general file-size-to-space estimate.
Restoration is from `tree-inner-archive` on the durable mirror, with individual
hashes in `RECLAIMED_COMPLETED_TREE_PARITY.json`. The earlier transport removal
and this second pass are separately preserved.

Second-pass wrapper: `run-heavy: done rc=0 28s | peak-RSS 0.02GiB | min-avail 40351MiB | peak-load 20.68`.

## All registered generalization input views qualified

All 42 distinct prepared views supporting the 672 registered source-curve,
row-curve, shuffled-label and human-only cells are complete. Independent audit
checks every prepared TRAIN row, unchanged numeric feature bits and keys,
registered population/conditional label permutations, and exact matrix coverage.
All 42 pass. The three strict human-only outer-training views contain 7,060,
6,837 and 6,947 rows, each with zero teacher rows. This is input qualification;
none of the 672 model cells is claimed fitted or evaluated by this audit.

Preparation wrapper: `run-heavy: done rc=0 358s | peak-RSS 0.52GiB | min-avail 40346MiB | peak-load 21.31`.
Audit wrapper: `run-heavy: done rc=0 35s | peak-RSS 0.72GiB | min-avail 40308MiB | peak-load 20.35`.
These are resource observations during concurrent training, not speed benchmarks.

The `variant-input-archive` mirror under the completion-study archive contains
790 artifact files totaling 1,261,291,003 logical bytes; all artifact checksums
passed destination readback at 06:28:20 UTC. Its local staging snapshot uses
hardlinks to the immutable admitted inputs; its logical size is not additional
SSD allocation. `VARIANT_INPUT_AUDIT.json`, `VARIANT_INPUT_PROGRESS.json`, frozen
memberships, preparation receipts and per-file hashes are retained. Two tests
cover corruption of feature bits/keys/null masks/labels and independent replay
of both registered permutation schemes and seeds.

## Additional registered analytic waves at 06:34 UTC

The untouched `inner-2-0` analytic wave is assigned to i265 and `inner-2-1` to
i270. Each contains 36 registered Softplus/monotone-SiLU/ordinary-SiLU cells,
with four concurrent trainers per wave. Both use the existing qualified frozen
training capsule and binary set, unchanged 240 x 50,000 exposure, settings,
seeds and training-complement rows. Each trainer has a one-thread, 1 GiB cap;
each new controller has a 6 GiB, four-worker cap. These are admission caps,
not measured peak memory claims.

All 20 transferred inputs per partition passed exact checksum verification:
23,190,369 bytes for inner-2-0 and 24,109,088 for inner-2-1. The second transfer
uses the committed installer, which rejects unexpected paths, symlinks,
existing files and checksum mismatches before scientific execution. Its test
verifies exact installation, overwrite refusal and wrong-hash refusal.

At 06:34:48 UTC, live process checks found 18 dev, 20 i270 and 16 i265 fits,
with no unexpectedly stopped attempts. All eight newly admitted native wrappers
were verified live; the four i265 fits had logged epochs 2, 5, 1 and 2. The four
i270 fits were initializing. Free SSD bytes were 57,352,433,664,
60,921,602,048 and 219,094,581,248 respectively. No global compute lock was added.

Both input/admission archives passed all-file destination readback:
`worker-i265-analytic-inner-2-0-inputs` (nine files, 23,245,851 bytes) and
`worker-i270-analytic-inner-2-1-inputs` (ten files, 24,171,364 bytes).
These archives qualify inputs and placement; completed scientific fit outputs
will be collected and archived separately.

The refreshed progress audit records 133/816 primary inner final reports and
7/72 fresh-control final reports. Thirty-seven primary cells have neural
convergence receipts, 34 retain investigation triggers, and four retain original
diagnostic-gate failures. No outer-selection or convergence-completion claim
follows from these process/report counts.

## Generalization shaping initializations

All 33 unique registered feature-training memberships now have separate product
and damage/fragility initializations: 66 manifests bound to the 672 generalization
cells. Each uses the existing raw-domain q50/q90 implementation on that view's
TRAIN rows. Label permutations share only these label-independent initializations;
no supervised pretraining or learned shape is reused across labels. Native scale
fitting and supervised shape learning remain requirements of the later refits.

The initializer now rejects cached manifests whose plan, raw-feature pin, kind
or frozen declaration differs. Eight shape/membership tests pass, including
held-out-value exclusion, product/separate knot domains, invalid domains,
source/row membership and cache identity drift. Actual preparation verified all
66 manifest and receipt hashes before archival.

Preparation wrapper: `run-heavy: done rc=0 17s | peak-RSS 0.26GiB | min-avail 40919MiB | peak-load 19.48`.
This is a resource observation during concurrent fitting, not a speed benchmark.
The `variant-shape-archive` destination passed all-file readback at 06:38:22 UTC:
139 artifact files, 28,243,529 logical bytes. Local staging uses hardlinks to
immutable initializations; its logical size is not additional SSD allocation.
This completes initialization preparation, not the registered model fits.

## Per-feature reproducibility in the recovered additive fits

The owner asked whether individual feature shapers recur across trainings. A new
read-only analysis compares nine historical positive-shaped-products fits at fixed
lambda 1e-4: each of three source folds has its full-source fit and two 75%-source
refits. These are deterministic additive solutions, not neural initialization
replicates. No labels, outer predictions or selection metrics enter this analysis.

Reconstruct each term's continuous piecewise-linear contribution in raw feature
units, undoing input and basis normalization. Compare at the union of all fitted
knots, from zero through the largest second knot. Normalize each nonzero curve to
unit endpoint; preserve its unnormalized endpoint strength separately. Differences
are linear between knots, so the maximum normalized disagreement occurs on this
grid. This does not establish behavior beyond the stated domain.

Eight of 984 terms are nonzero in every fit; 781 are zero in every fit. The two
most similar normalized curves among those eight are:

| Term | Registry interpretation | Maximum normalized curve spread | Maximum/minimum endpoint strength |
|---|---|---:|---:|
| f585 | v2 art_soft_peak, scale 2, Y | 0.07191324435473345 | 4.4936803309626905 |
| f14 × f422 | basic ssim_4th × v2 pjnd_fragility, scale 0, Y | 0.09116305067813268 | 21.311593278910188 |

Across all nine fits, f585 has exactly zero first-segment slope and a positive
second-segment slope: a dead zone followed by a rising response. Its first raw
knot ranges from 0.0706766824687084 to 0.07400064202942325. Its far-tail slope
is not stable. The f14 × f422 term has a positive first-segment slope and exactly
zero second/third slopes in every fit: a rising response capped at the first
knot, which ranges from 0.016456244122225128 to 0.018106926808493018.

Feature IDs use the original 720-slot layout, not positions in the reduced raw
array. Names follow `zensim/src/feature_defs.rs`. The next two always-active terms
have normalized spreads 0.15555768971263073 and 0.24901071965308388; the remaining
four have spreads at least 0.9731927214486901. Stability is therefore mixed.
These are descriptive candidates, not a selected transferable shaper set. Strength
variation can reflect correlated inputs and fitted target scales; it does not
alone invalidate similarity of normalized shapes. Transfer into frozen-shape
neural trials and consistency across neural seeds remain unanswered.

The analysis stores all 984 per-term results, nine fit identities, and SHA-256
pins for every specification, task and coefficient archive. A synthetic check
separates scalar gain changes from shape changes, exercises nonunit input scales,
excludes zero curves from useful stability, and refuses overwriting output.
`just shaped-linear-feature-curves` reproduces the analysis without modifying the
historical study. Results are in the completion root's
`feature-curve-stability-v2/FEATURE_CURVE_STABILITY.json`, mirrored with complete
destination checksum readback under the same NAS-relative path.

Wrapper: `run-heavy: done rc=0 1s | peak-RSS 0.20GiB | min-avail 40617MiB | peak-load 19.76`
for the first calculation; the v2 receipt adds original feature-ID mapping. This
is resource accounting during concurrent training, not an inference benchmark.

## Explicit human-only native controls and replay qualification

The analytic optimizer-control configuration previously required exactly one
teacher population. Human-only supervision now has an explicit opt-in flag that
requires zero teachers, neutral teacher mass and a non-teacher-warm curriculum.
The default retains the original one-teacher requirement and omits the new flag
from legacy serialization. Human-only commands require exactly the four human
TRAIN ranking groups and set MSE weight to zero; feature and optimizer settings
remain unchanged. This implements a necessary capability, not a selected model.

Native source `709392a65a28b1565d299569c430c9bf3218a435` passed all four training
research tests, all twelve monotone research tests, and release-profile Clippy
with warnings denied. The release trainer and packer were built and their remote
hashes verified after transfer. `human-only-bin/BUILD.json` has SHA-256
`ce8199538b28165f69f6f9b2ea23a1e5609a27a31e8d427b65a67b7de642dd3d`;
unchanged panel/parity tools retain their separately recorded source provenance.

All three analytic families passed two independent one-epoch checks, each using
50,000 pair draws. First, neutral mixed-data controls reproduced the original
frozen prefix's canonical model bytes exactly. Second, human-only neutral controls
reproduced native streaming bytes exactly on the registered outer-0 training
complement: 7,060 human rows, zero teacher rows. Canonical comparison strips only
reproduction and research-configuration metadata, retaining weights and scales.
No outer labels or selection metrics were read. These short checks establish
replay behavior; they are not full-budget scientific fits or new convergence data.

| Family | Human-only canonical SHA-256 |
|---|---|
| softplus | `223ca465a6553e124f820b4cca37ef286b4d0c35e65a902bbf0996890fe45794` |
| silu_monotone | `49f04cdfd02667c68256b456626f88f7b7f003e118e6f7164a7c2673e1ae45e2` |
| silu_diagnostic | `68dce17d885b20965dba4e739500cb5fca4496f7384aebdadfca1afe5bf87fb2` |

Resource-wrapper observations, during concurrent training:

- native-tests-r3: `run-heavy: done rc=0 555s | peak-RSS 1.69GiB | min-avail 22848MiB | peak-load 21.85`.
- monotone-tests: `run-heavy: done rc=0 0s | peak-RSS 0.09GiB | min-avail 24424MiB | peak-load 20.96`.
- clippy: `run-heavy: done rc=0 107s | peak-RSS 0.65GiB | min-avail 23834MiB | peak-load 21.39`.
- build: `run-heavy: done rc=0 303s | peak-RSS 1.69GiB | min-avail 22824MiB | peak-load 22.61`.
- softplus: `run-heavy: done rc=0 54s | peak-RSS 0.10GiB | min-avail 39174MiB | peak-load 20.88`.
- silu_monotone: `run-heavy: done rc=0 54s | peak-RSS 0.10GiB | min-avail 39104MiB | peak-load 20.88`.
- silu_diagnostic: `run-heavy: done rc=0 37s | peak-RSS 0.10GiB | min-avail 39352MiB | peak-load 20.30`.
- legacy-softplus: `run-heavy: done rc=0 41s | peak-RSS 0.09GiB | min-avail 37587MiB | peak-load 20.89`.
- legacy-silu_monotone: `run-heavy: done rc=0 26s | peak-RSS 0.09GiB | min-avail 39371MiB | peak-load 20.87`.
- legacy-silu_diagnostic: `run-heavy: done rc=0 20s | peak-RSS 0.09GiB | min-avail 39352MiB | peak-load 20.59`.

The source capsule, failed preparation logs, native test/build logs, all six
qualification arms and results remain under `human-only-native-qualification/`.
No failed attempt was overwritten. Native scientific human-only refits remain
gated on inner-only finalist selection.

The qualification archive passed destination readback for all 102 artifacts
(135,606,307 bytes) at 07:11:51 UTC. The binary archive independently passed all
five artifact hashes (32,854,017 bytes) at 07:11:07 UTC. Both include their source
and binary identities; neither receipt claims completion of the 48 registered
scientific human-only fits.

Seven further worker fits were collected with complete checkpoint inventories:
three 240-epoch analytic fits and four 120-epoch fresh ReLU controls. Each durable
bundle passed full checksum readback and assignment checks before per-member
verification and atomic local import. Failed transport/preparation logs remain
preserved. At 07:11 UTC the progress audit counted 147/816 primary inner final
reports and 12/72 fresh-control reports; 51 primary and 12 fresh-control neural
convergence receipts were present. Of those, 48 primary and four fresh controls
retained investigation triggers. These counts are execution progress, not
convergence qualification or a family ranking.

### Archived checkpoint retention

The checkpoint-retention command keeps epoch zero, epoch 119, the scheduled
final epoch, and every registered diagnostic epoch locally. Other checkpoints
are eligible only in completed, convergence-recorded, fully mirrored cells.
Before removal it rereads each local and durable SHA-256, checks open descriptors
and mappings twice, and verifies local file identities again. A durable receipt
records each removed path and its archive counterpart; per-cell notes identify
the partial local copy. Failed runs and registered input pins are excluded.
Tests verify retained epochs, successful duplicate removal, late archive
corruption with no deletion, busy files, and source changes during verification.
This changes local storage placement, not the save-every-epoch record or
scientific checkpoint-selection policy.

The first retention batch removed 17,864 local duplicate checkpoints from 88
completed cells after rereading all 4,444,420,965 logical bytes from their durable
copies. The local receipt and post-removal check confirm every removed path is
absent, every durable counterpart exists, and every required retained epoch is
present. Free bytes measured immediately before/after were 52,574,863,360 and
57,033,805,824; concurrent workloads mean that delta is not an isolated allocation
measurement. `run-heavy`: 229 seconds, peak RSS 0.11 GiB, minimum available RAM
30,277 MiB, peak load 18.71. Evidence lives in the completion artifact root's
`checkpoint-retention-qualification/`, including the full per-file receipt.

### Native full-FIT setting-search runner

The native search runner resolves each of the registered selection cells through
the complete three-fold inner-evidence admission. It retains the fixed family,
optimizer and exposure, uses that cell's original setting and paired seed, and
refits preprocessing on the registered training complement. Pretrained models
learn new shapes there before starting a fresh head. It shares the existing
native phase and objective owners with outer refits. Inputs, command templates,
qualification, binaries and final checkpoints are pinned; reports are scoped
`full_fit_only`. Source membership drift, admission drift, duplicate attempts and
using final-refit cells as selection cells are rejected in orchestration tests.
These tests are not scientific fits. Full-FIT evaluation, nonneural execution,
selection receipts and final refit execution remain to be connected; no full-FIT
scientific result is claimed by this implementation.

Full-FIT native search evaluation now uses the same native prediction, TRAIN-only
calibration, feature probes and final-checkpoint checks as the existing evaluator.
Its globally admitted family controls the identity/monotonicity claim and its
uniform representation must match inner admission. The report field is
`full_fit_validation`, scoped `full_fit_only`; it records exposure of these
previously outer FIT rows and cannot pass the inner selector's exposure gate.
Validation checkpoint diagnostics are refused, preserving final-epoch-only
selection. Twelve evaluation tests pass, including full-FIT admission and scope
isolation; this is implementation evidence, not completed full-FIT fitting.

The existing tree and shaped-additive control owner now also accepts the admitted
full-FIT search. It retains all four settings, all three tree seeds and the one
deterministic quadratic solve per setting. Dedicated full-FIT cache paths avoid
reusing or overwriting outer-control Gram matrices. The existing TRAIN-only
basis/map fitting, native export checks and objectives remain unchanged. Reports
use the same full-FIT-only scope as the neural search. Five control-admission
checks pass; no control fits are claimed by those orchestration tests.

### Full-FIT training-only convergence evidence

Full-FIT search diagnostics reuse the registered 20-versus-20-epoch loss rule,
including the native log's rounding interval, and record final output activity.
Every scheduled epoch in each phase and its final model must be present. The
result is explicitly `inconclusive_training_only`: validation checkpoint trends
are not measured, and validation/development values cannot decide extra fitting.
Tests change validation scores without changing the diagnosis and reject duplicate
loss epochs. Existing inner convergence tests retain their original outcomes.
This supplies a limited, honest classification; it does not establish convergence
or count unexecuted scientific cells as complete.

### Verified full-FIT setting decisions

The setting-selection receipt owner now requires complete original fold/seed
coverage and reproduces the equal-human composite decision. It binds admission,
run/result/report identities, final scheduled neural checkpoint, repaired runtime
when required, and each control's native model/parity evidence. Neural training-only
classifications must reproduce from the phase logs and final model evidence.
Changes to a model, report scope or convergence conclusion invalidate admission;
incomplete evidence creates no selection receipt. Receipts are exclusive and
recomputed before downstream use. Three provenance tests pass, alongside the
existing full-FIT human-only scoring/tie tests. No real setting decision exists
until the registered scientific search has completed.

### Final all-FIT neural refits

The shared full-FIT native runner now resolves final cells only after every role's
setting decision reproduces. It retains the selected setting, fixed optimizer and
exposure, starts new weights and supervised shapes, and pins all setting receipts.
All-FIT preparation records no outer fold and passes only registered FIT rows to
the trainer; exposed development remains outside its groups. Five preparation,
search and final-admission tests pass, including refusal when another role used
development or when a development row enters training. Final scientific refits
remain pending their selection gates; this implementation is not a measured fit.

Final neural evaluation now admits exactly the registered exposed-development
membership after all full-FIT settings are fixed. Native scoring still fits maps
on FIT rows only. Reports use `exposed_development` and `descriptive_only`, record
that development was read, and cannot enter either earlier selector. The cached
view is bound to the full-FIT admission and all setting receipts. Development
checkpoint diagnostics are refused. Fifteen evaluation tests pass, including
wrong membership/population refusal and final schedule checks. No untouched final
holdout is referenced by this path; no real development score is claimed yet.

Final tree/additive refits now resolve all role decisions through the same final
admission gate, use their selected settings and paired seeds, and retain separate
final-refit caches. Their FIT-only objectives and calibration are unchanged;
development appears only in descriptive evaluation. Six control-admission tests
pass, including unchanged TRAIN membership and preserved empty registered test
membership. Final neural training-only diagnostics accept the descriptive report
but never use development scores; the four convergence tests cover that scope
alongside the unchanged inner loss rule. Scientific final fits remain gated.

### Worker transport retention

A completed worker import can leave both original fit files and a redundant
transport tar on the worker SSD. The transport-retention tool first requires the
matching durable transfer receipt, producer sidecar and successful local import,
then cold-rereads the durable tar. On the worker it verifies the tar and sidecar
again, checks open files twice and rechecks file identity before removal. It
removes only selected transport tar files; original fit directories and sidecars
remain. Corrupt durable/worker bytes and open transports refuse removal in tests.
This addresses storage admission without reducing the registered reserve or
trainer write allowances.

Worker transport retention on i270 verified and removed 16 duplicate tar files,
1,004,451,840 logical bytes, with every original fit directory retained. Free
bytes before/after were 56,372,252,672 and 57,376,747,520. The first attempt refused
before deletion because lsof was absent; after installing it, the second attempt
repeated durable readback and worker checks. `run-heavy` reported 14 seconds and
0.02 GiB peak RSS for the coordinator; worker removal reported 1 second and
0.02 GiB peak RSS. The pending packaging job could then retain its original
reserve policy.

A separate later-stage mode in the existing evaluation supervisor now handles
admitted outer, variant and full-FIT native fits. It evaluates only final epochs,
resolves the actual family and binary identity, and uses the admitted repaired
ReLU representation when required. Inner evaluation remains in its existing
supervisor. Full-FIT evaluations receive reproducible training-only diagnostics;
failed gates remain explicit. Supervisor tests cover stage isolation, final-only
scoring and continuing independent cells after a native gate failure. Other
later-stage convergence analysis and aggregate comparisons remain required.

### Full-length human-capable trainer compatibility

The human-capable native build manifest `ce8199538b28165f69f6f9b2ea23a1e5609a27a31e8d427b65a67b7de642dd3d`
completed the registered 240-epoch neutral replay against primary cell
`8889ddf0ed4c3fd80ba0`. Both canonical model files independently hash to
`8c770a06ccd258a4d4eccd3473f395587c34c73908187bf424afeb27eda67973`;
all 240 checkpoints are present. `run-heavy`: 3,777 seconds, peak RSS 1.27 GiB,
minimum available RAM 30,312 MiB, peak load 20.90. This is compatibility evidence
for neutral mixed-supervision execution, not a scientific candidate or evidence
that human-only training improves quality. The earlier human-only one-epoch
qualification remains separately identified. Artifacts live in the completion
root's `human-build-neutral-full/` directory.

Inner original and supplemental evaluation watchers now filter their owned stages
before checking base-matrix membership. Conditional matched outer controls belong
to the later evaluator and must not stop an inner watcher merely because they
are registered by an outer-admission receipt instead of the base matrix. A
regression test verifies both inner watchers leave that result to its owner;
existing inner failure and later final-only behavior remain tested.

## Third-fold rescue admission and continued execution (08:59 UTC)

Registered inner cell `bf9567983d426a188f00` (ordinary SiLU diagnostic,
inner-2-1, setting 0, seed 0) completed 240 epochs. Its final 20-epoch
mean logged loss was 0.16346, versus 0.22818 in the preceding window.
The registered rounding-aware loss test requires investigation; all seven
diagnostic checkpoints retain 64 active output coefficients. This is evidence
of continuing optimization, not output-coefficient collapse or a quality win.

The trigger registrar reproduced the diagnostics from pinned reports and the
complete training log, verified the terminal bake, and wrote
`RESCUE_TRIGGER_OUTER_2.json`. Its convergence receipt SHA-256 is
`43a5c7fd1caf027521b1dbd01f75a8434f5472dc0e6282e119afc023a0fb4183`.
All three outer folds now require their pre-registered 240-cell rescue waves:
720 rescue fits in total. The third dispatcher waits for complete paired base
selection; registration is not execution or convergence. No outer results were
read to make this decision.

The inner-0-1 pretrained dispatcher now has two verified live native fits,
released from completed matching parent fits. A second checkpoint-retention
pass reread and removed 5,220 local duplicates from 25 completed archived cells,
preserving diagnostic, final and epoch-119 prefix checkpoints. Removed logical
bytes: 1,399,134,243. Observed filesystem free bytes before/after were
56,591,933,440 / 57,996,812,288; concurrent jobs were still writing. Wrapper:
`run-heavy: done rc=0 39s | peak-RSS 0.10GiB | min-avail 33559MiB | peak-load 20.63`.
The four-file retention evidence archive (6,239,399 bytes) passed full destination
SHA-256 readback under `checkpoint-retention-0858-qualification` in the study's
local and durable roots. The full scientific matrix remains incomplete.
