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

### Fresh durable verification of both inner control archives

On November 7 at 12:35 UTC, all 24 shaped-additive and 72 tree inner fits
were confirmed covered by their central batch archives. Current RUN, RESULT,
REPORT and COMPLETE hashes match the batch manifests for every control.
The per-cell archive count in the progress audit excludes these batch archives;
that omission is not evidence that the 96 controls lack durable copies.

Fresh cold destination readback passed all 246 linear-archive manifest entries
(610413374 bytes) and all 585 tree-archive entries (3251256867 bytes), including
the original mirror receipts. The tree manifest is
`2e7685d7746262d40a42a242358a1084f39c4d839e9944c3b5a72c71acb0213f`;
the linear manifest remains the hash above. Local copies are explicitly partial;
this verification establishes complete durable copies, not complete local ones.
Verification receipts, resource logs and coverage are retained in
`inner-control-batch-reverification` under the completion-study roots.

## Additional joint-shaping training capacity

The 24 registered inner-2-0 joint-shaping cells are now assigned to i270, with
four live fit processes verified at launch. The worker retains the existing
frozen training source and binary; the 24-file input capsule (23812705 bytes)
passed installation checksums. The controller has a 4 GiB cap and each fit a
1 GiB cap with one thread. The other worker controllers could occupy at most
3, 8 and 4 threads at admission, giving 19 fit threads on 24 CPUs.

Disk admission followed removal of 37 imported transport duplicates
(2212474880 bytes) after durable readback and import checks, plus 1390587639
bytes of inactive compiler intermediates with executables preserved. The first
installation command referenced a file absent from the old source snapshot;
the first launch then stopped in input preparation before creating fit attempts.
The standalone verified installer and subsequent launch succeeded. Both failure
logs remain preserved. Input and launch/resource evidence passed archive
readback in `i270-joint20-inputs` and `i270-joint20-qualification`. This records
execution admission, not completed quality results.

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

## Immutable terminal failure receipts

Repeated evaluator sweeps previously rewrote identical failure receipts, leaving
an unnecessary truncation window during archival. The original, supplemental
and later-stage supervisors now preserve an existing byte-identical receipt
without opening it for writing; differing evidence stops the sweep and preserves
the original. The native gates and failure classifications are unchanged.
The regression check verifies unchanged inode/mtime on repeated observation and
preservation of original bytes on conflicting evidence.

## Bounded outer neural dispatch

`shaping_outer_dispatch.py` waits for `OUTER_ADMISSION.json`, then reproduces
that admission with the existing complete inner/rescue/extension evidence gate.
It prepares TRAIN inputs through the existing owner and queues all 30 paired
outer neural cells plus the exact admitted fresh controls. Each native child
retains the selected schedule, optimizer, seed, qualification and disk admission.
The command uses two concurrent children; a per-fold queue ownership lock prevents
duplicate dispatchers without serializing other folds or training workloads.

Existing fits are resumed only with matching plan/build/admission, cell and
schedule identities; completed fits also require the final model hash. Failed,
ambiguous or stale attempts require explicit reconciliation and are not overwritten.
Five dispatcher tests cover exact pairing, changed model/admission refusal,
failed/ambiguous attempt preservation, no launch before admission, and a complete
synthetic queue that launches each registered cell once with at most two children.
The queue test verifies orchestration, not native training quality. Tree/additive
outer controls retain their separate existing execution owner. Scientific outer
fits remain unexecuted until the full evidence admission becomes available.

## Completed ReLU consolidation-rescue optimizer traces

Both seed-0 `consolidate_lr` ReLU rescues on outer fold 0 have complete
240-epoch native diagnostic traces. The summary command verifies their final
model/report/convergence provenance through the supplemental ReLU representation
before reading the training log. It rejects missing/duplicate epochs, inconsistent
activity counts and nonfinite evidence rather than reporting a complete trace.

| Inner partition | Epochs with at most one nonzero output | First such epoch | Total output projection hits | Final mean squared first-activation gradient |
|---|---:|---:|---:|---:|
| inner-0-0 | 226 / 240 | 14 | 855,406 | 1.443929485482317 |
| inner-0-1 | 225 / 240 | 14 | 836,323 | 1.520907514992422 |

At epoch 239 both fits have one nonzero output coefficient and one active
last-layer unit among 386 sampled training responses. First-layer activation
zeros are respectively 43,100 and 44,931 of 49,408 sampled values. These
first-activation gradients are not first-weight gradients. Sampled inactivity
is not proof of globally dead units, and a single surviving output does not
establish a linear function or a feature-information ceiling. Consolidation
has not prevented collapse in these two fits; the other paired seeds and rescue
choices remain required before comparing remedies. The original native-gate
failure for `ddc7932564aefab339d1` remains preserved independently of its
supplemental evaluation. Summaries and exact input hashes are under
`relu-rescue-optimizer-diagnosis` in the completion-study artifact roots.

## Automatic terminal worker collection

`shaping_worker_watch.py` refreshes each worker's process-backed status and
passes freshly observed, completed and unimported fits to the existing terminal
wave collector. That owner preserves producer checksums, rereads durable transfer
bytes and validates the local import. Each completed wave's records also pass the
full archive readback owner. The watcher uses one ownership lock per worker;
manual collection must not overlap that worker's watcher.

A failed collection stops for explicit reconciliation, preserving partial output.
An SSH timeout or command failure records unknown remote training state and retries
observation; it neither restarts nor marks remote training failed. Tests cover
terminal-only admission, stale observations, already imported runs, stopped-job
reconciliation, no transfer on an empty cycle, no archival/retry after collection
failure and the unknown-state timeout receipt. Scientific gates and model
selection are unchanged.

## Paired rescue-selection and extension admission supervision

`shaping_selection_watch.py` advances triggered folds only through the existing
selection and admission owners. It waits for all 24 rescue candidates per family,
then invokes the full paired base/rescue provenance gate. All ten neural decisions
must verify before extension registration. Existing registrations are reproduced,
including empty matrices and pinned decision hashes. Required extension reports
and diagnostics must exist before the complete outer-admission gate runs.

Presence checks schedule validation; they never stand in for provenance or
scientific completion. Errors preserve existing artifacts and stop the watcher.
Tests verify that missing diagnostics and ambiguous candidates are refused,
incomplete rescue waves cannot select or admit, and apparent completion markers
still lead to the full provenance gate. This watcher does not execute extension
fits; their native execution owner remains separate. The three triggered folds
still require their unfinished paired experiments.

## Historical comparator parity retention (09:20 UTC)

A completed worker import waited at storage admission while pending writes were
reserved. Eighteen historical shaped-linear Extra Trees comparator parity inputs
were eligible archived duplicates: exactly three folds × three label modes ×
two seeds, each containing only `test_X.npy`, `test_raw.npy` and `test_p.npy`.
The fixed historical manifest hash matched both copies; all 18 local and durable
files passed SHA-256 reread, with POSIX_FADV_DONTNEED on the durable copy. Two
lsof inspections and a final inode/size/mtime check preceded removal. None was
pinned by the active study. Models, standalone predictions, specifications and
active input arrays were retained; the prior Gram-retention receipt was preserved.

Removed logical bytes: 1,088,053,114. Observed free bytes before/after were
56,265,535,488 / 57,353,674,752 while other jobs continued writing. Wrapper:
`run-heavy: done rc=0 20s | peak-RSS 0.10GiB | min-avail 33315MiB | peak-load 21.45`.
The waiting worker import subsequently completed and its automatic wave archive
verified. Exact removal inventory and restoration root are recorded in
`RECLAIMED_LINEAR_PARITY.json` under the completion-study artifact root. This
retention did not lower the 30 GiB reserve or any pending-write allowance.

## Doubled-exposure native dispatch

`shaping_extension_dispatch.py` waits for the extension registration, reproduces
all paired rescue decisions and the exact registered extension matrix, and queues
native fits through the existing extension owner. It preserves continuous 480-epoch
fits, pretrained 240+240 phases and the registered fresh controls. Each fold uses
two slots; concurrent copies of that fold's dispatcher are refused. Existing fits
require matching plan, build, registration, cell, exposure and final model hashes.
Failed or ambiguous attempts are preserved for explicit reconciliation.

Four tests exercise refusal before registration, terminal identity/exposure/model
checks, preservation of failures, an empty verified matrix, and a nonempty queue
with continuous/pretrained/fresh-control exposures and a measured two-child
maximum. These are orchestration checks, not scientific fits. Extension evaluation
and outer admission still require the existing complete diagnostic gates.

## Native input-response reproducibility, outer-0 inner partitions, setting 0

This descriptive comparison uses all three paired seeds on both inner partitions
for each identity-initialized joint-shaping family. It reads final native shaping
metadata only after completed candidate provenance verifies. Neither outer
responses nor quality rankings select these fits. Exact f32 parameter values are
recovered from the metadata decimals before evaluating real-valued curves.

Among 984 terms, product shaping has 14 active in all six fits and 438 zero in
all six. Separate shaping has 12 active throughout and 613 zero throughout.
The five smallest normalized response spreads among always-active product terms
are f19 (0.034083065198891704), f597 (0.04340139520968023), f59×f509
(0.05733979197287997), f586 (0.06337348810652772), and f403
(0.0972438711225124). For separate shaping, f403 has the smallest such spread,
0.0859823010854921. This is descriptive ordering, not a registered quality selection.

Across both architectures' twelve fits, five of 492 direct terms remain active.
Only f403 has a normalized spread below 0.8: 0.16523522064482415, with response
strength varying by 3.98478155991701. The other four spreads range from
0.8517161053827695 to 1.0. This supports a limited repeatability observation;
it does not establish a reusable shaper bank or improved generalization.
Downstream weights are excluded, so a nonzero input response is not proof of
usefulness to the fitted network. Zero terms are never counted as useful stability.

The comparison domain ends at the largest TRAIN-fitted second knot. Separate
terms compare the full damage/reference response rectangle at the union of shape
knots and gate vertices (reference 0, 0.5, 1), rather than ignoring gate variation.
No extrapolation or bit-exact runtime-score claim is made. Per-term slope/gate
ranges, source hashes and full manifests are in the three
`native-*-stability-o0-s0-v2` artifact directories. Initial decimal-f64 summaries
are retained separately; v2 uses exact native f32 parameters. Four tests cover
curve/strength normalization, wire-equivalent decimal spellings, separate gates,
and exclusion of zero/invalid terms.

Diagnostic coverage remains incomplete: existing optimizer traces record
first-activation gradients, not first-weight gradients. The latter must not be
reported as measured from these traces. Source inspection confirms this distinction
in `shared_depth.rs` and `monotone_research.rs`; native first-weight diagnostics
remain required by the frozen protocol.

## Historical tree model retention (09:41 UTC)

The historical tree archive manifest
`c98912ff5e927496899a5f6d4bd5aaa8a60f4b331d5b1fa483966ce1ac35ecad`
identifies 189 Python model pickles: 129 search, 54 peer-search and six constraint
controls. All 189 completed result records match that manifest. Each model passed
local SHA-256 and cold destination SHA-256 verification, two process/open-file
checks and final inode/size/mtime checks before its local duplicate was removed.
No active completion-study input pins one of these models. Durable models, local
native exports, reports, predictions and input arrays remain available.

The receipt records 2,667,085,403 logical bytes removed; filesystem free space
changed from 56,618,467,328 to 59,285,913,600 bytes. The resource wrapper reported
`done rc=0 38s | peak-RSS 0.09GiB | min-avail 33034MiB | peak-load 21.26`.
The operation did not change the 30 GiB reserve or pending-write allowances.
The exact receipt, source, log and completed-result checks are preserved under
`tree-model-retention-qualification` in the completion-study artifact root.

## Paired source-bootstrap draw registration

Before opening completion-study outer predictions, `shaping_bootstrap_plan.py`
registered 2,000 population-stratified source-component draws per outer fold,
using one NumPy PCG64 stream seeded 20261107. Every candidate/control and model
seed will use the same draw arrays. The stream advances across strata rather
than restarting, so equally sized populations/folds do not receive artificially
identical samples. Each draw preserves all rows and duplicate multiplicity of a
sampled component. This is conditional uncertainty over observed sources and
fitted models, not training-seed or population-distribution uncertainty.

The metadata audit found no component spanning populations and no source overlap
between outer folds. Component counts by fold are: CID22-human 6/6/6,
CID22-SSIM2 47/46/46, KADID 18/19/18, KonFiG 2/2/2, and TID2013 4/5/5.
KonFiG's two components per fold sharply limit the source variation represented
by its bootstrap. No labels, predictions, or quality-selection results were read.

Four tests cover complete-component multiplicity, cross-population component
rejection, exclusion of unselected rows, and reproducible sampling with an
advancing stream. All 15 generated arrays passed shape/range checks; the
407,965-byte `draws.npz` has SHA-256
`6f83b42a08e946b4acd5d324402ca044f565913a056d3e0ba186dfab9fb4f1e0`.
Artifacts are in `source-bootstrap-plan` under the completion-study root.
These are registered sampling inputs, not completed confidence intervals;
consuming admitted outer predictions and computing the native rank panels
remain required work.

### Native bootstrap rank calculation

`shaping_bootstrap_stats.py` computes pooled and mean-within-component signed
Spearman samples using the existing Rust panel, with every model evaluated on
identical whole-component draws. It retains model/seed samples separately and
does not average predictions. Source rows must partition the input exactly;
nonfinite data, invalid draws, native response identity changes and dropped rows
are errors. Index jobs are processed in batches of 32 replicates.

Four checks passed against the frozen study panel. The numerical check covers
36 replicates across the batch boundary, tied ranks, repeated sources, identical
candidate pairing and reversed-score polarity, compared with SciPy Spearman to
nine decimal places. The resource wrapper reported `done rc=0 1s | peak-RSS
0.09GiB | min-avail 31882MiB | peak-load 22.53`. This is evaluator qualification
on an artificial fixture, not scientific uncertainty or an inference benchmark.
The admitted outer-evidence loader and final interval reduction remain to be
connected before scientific use. Evidence: `bootstrap-stats-qualification`.

### Prediction-table binding for outer uncertainty

New neural and control evaluations bind `predictions.npz` into `REPORT.json`
with `predictions_sha256` before the completion marker is written. This adds
provenance without changing predictions, ranks, gates or selection. Earlier
reports remain preserved; an unbound table cannot enter the new uncertainty
reader merely because its report has a completion marker.

`shaping_prediction_evidence.py` verifies report/marker identity, exact admitted
outer role, plan/admission hashes, prediction digest, ordered row membership,
shape and finite coverage. Its caller must revalidate the full outer admission
before reading responses; this helper alone is not the complete study loader.
Five new evidence checks, 15 existing evaluation checks and six outer-control
checks pass. Qualification is preserved in `prediction-evidence-qualification`.

### Complete primary outer-evidence loader

`shaping_outer_evidence.collect` revalidates all three complete inner-admission
records, then requires exactly the registered 102 primary outer cells (all 12
families, three folds, and their declared seeds). Every cell must have one
successful terminal attempt and a completed evaluation before any prediction
table opens. It verifies plan/admission identity, selected schedule or control
setting, original model hashes, repaired-ReLU model/build provenance, bound
prediction hashes and complete ordered row membership. Convergence classifications
remain attached; collection does not reselect models using outer results.

Four tests cover matrix completeness, admission-before-read ordering, missing
fit refusal, and a complete synthetic 102-cell artifact matrix including repaired
ReLU representations. Mutating a completed prediction table is rejected. On the
live study, the missing outer-0 admission produced the expected refusal with zero
prediction-reader calls. This validates admission behavior, not scientific outer
completion. The final bootstrap runner and interval reduction remain pending.
Evidence is in `outer-evidence-qualification` under the completion-study root.

### Primary paired-interval reduction

`shaping_bootstrap_reduce.py` requires all 12 families, three outer folds, the
registered one or three model seeds, five populations, pooled/within rank panels,
and all 2,000 replicates. It reports every fold and the equal-fold mean, each
population separately, and the equal four-human composite. Teacher agreement
never enters that composite. Means are over seed performance, not predictions;
95% percentile intervals and differences from ReLU use the same replicate index.
No model selection or pooled-OOF headline occurs in this reduction.

Four checks pass: teacher isolation, exact population/fold/seed weighting,
cancellation of common source variation in paired differences, and rejection of
missing families/axes or nonfinite samples. The wrapper reported `done rc=0 1s |
peak-RSS 0.07GiB | min-avail 32384MiB | peak-load 23.96`. This is synthetic-array
qualification, not completed scientific intervals. Evidence is in
`bootstrap-reduce-qualification`; the end-to-end runner remains to be connected.

The bootstrap rank helper now explicitly rejects constant predictions, constant
targets and singleton source groups before calling the native panel. This matches
the existing study metric's undefined-correlation classification and prevents
numeric zero conventions from becoming measured source ranks. All five helper
checks pass, including the independent tied-rank oracle and 36-replicate batch
boundary. No scientific bootstrap used the earlier helper. Updated qualification
is archived separately in `bootstrap-undefined-qualification`; earlier evidence
is preserved. A refusal requires reporting undefined uncertainty, not dropping
sources or silently weakening the comparison.

### Complete primary bootstrap runner

`just shaping-completion-bootstrap STUDY DRAWS BINARIES OUTPUT` now connects the
102-cell outer-evidence loader, registered component draws, native rank panels
and paired interval reduction. It reproduces every draw from the registered seed,
verifies source membership and input hashes, checks recomputed point ranks against
completed reports, retains all outer population/distortion/source panels, and
carries convergence classifications into the summary. It writes per-stratum
samples and a final hash-bound summary only after all inputs pass re-verification.
Failures preserve partial artifacts without a completion marker. Active output
must share the study SSD and pass the existing reserve/admission owner.

Four tests pass. The synthetic numerical path covers the complete 102-model
matrix, all 15 strata and 2,000 replicates through the actual native panel. An
integration check also writes and verifies final summary/sample receipts with
synthetic admitted inputs. Other checks reject altered draws/source membership
and response reads before admission. The resource wrapper reported `done rc=0
21s | peak-RSS 0.21GiB | min-avail 32229MiB | peak-load 24.15`; these are fixture
qualification resources, not measured scientific-run or inference costs.

The live runner refused the missing outer-0 admission with zero array-load calls.
No scientific intervals have been computed. This runner covers primary outer
comparisons; fresh-head controls and generalization variants still need their
separate matched comparisons. Evidence: `bootstrap-runner-qualification`.

## Third completed ReLU consolidation rescue

Cell `1039887855a540dae1ee` is `inner-0-1`, seed index 1, under
`consolidate_lr`. Its complete 240-epoch native trace has at most one output
coefficient at 206 epochs, first at epoch 31. The final checkpoint has one
nonzero output coefficient and one sampled active last-hidden unit among 64,
using 386 sampled responses. It records 45,945 zero first activations among
49,408 sampled values and 929,999 output-projection hits over training.
The final squared-gradient summaries are 0.7688965147802748 for the first
activation gradient and 1445658.1106648236 for the regularized output gradient.
These are not first-weight gradients or proof of globally dead units.

The final evaluation still triggers both continuing same-cycle improvement and
one-output-coefficient investigation. Native original evaluation passed; the
uniform repaired-ReLU representation is also preserved. The fit, repaired
reports and re-export artifacts passed destination SHA-256 verification:
714 files, 54,234,876 bytes across three archives. The diagnostic summary and
pinned final/convergence reports are separately archived under
`relu-rescue-seed1-diagnosis`.

A log-hash-verified comparison of all three completed consolidation rescues
(`ddc7932564aefab339d1`, `e07b843ce1d5b313c0a9`, `1039887855a540dae1ee`) finds
exactly one output coefficient at every recorded epoch from 180 through 239,
all 60 checkpoints in each fit. Collapse first appeared before the schedule's
epoch-180 learning-rate change in all three fits. Thus this late consolidation
has not revived the collapsed outputs in these observations. This is evidence
across two seeds and two inner partitions, not a complete paired rescue result
or a rejection of the architecture. Other initialization/schedule trials remain
required. The late-phase measurements and input log hashes are preserved in
`relu-consolidation-late-phase` in both artifact roots. A single surviving output
coefficient still combines a nonlinear hidden network; it is not a feature ceiling.

## Fourth completed ReLU consolidation rescue

Cell `88b347ffcffe8506a7ad` completes the second seed on `inner-0-0`
with the registered 240-epoch consolidation schedule. The native optimizer
trace records at most one output coefficient at 205 of 240 checkpoints, first
at epoch 25 and continuously from epoch 37 through 239. All 60 checkpoints
after the consolidation change at epoch 180 therefore retain one coefficient.
Together with the preceding three cells, both paired seeds on both inner
partitions show persistence of collapse through the late consolidation phase.
This does not settle the other registered rescue schedules or reject ReLU.

At epoch 239, one of 64 last-layer units responds in the 386 sampled responses;
43,070 of 49,408 sampled first-layer activations are zero. Mean squared first
activation gradient is 1.282870966060248 and mean squared regularized output
gradient is 3837056.5065637473. The full trace records 1,065,276 output
projection hits. Activation gradients are not first-weight gradients, and
sampled inactivity does not establish global deadness. Convergence still
requires investigation for both the one-output condition and same-cycle
inner-composite improvement above 0.002. No outer results were read.

The original run, uniform max-pair evaluation and reexport records passed
complete destination SHA-256 readback: 714 artifact files, 67,416,352 bytes.
The independently stored summary and archive receipts are in
`relu-rescue-seed1-second-partition-diagnosis`. At 10:16:42 UTC the matrix
had 225/816 completed inner fits, 24/72 fresh inner controls and 4/720
rescues; outer, learning-curve, shuffled-label and human-only studies remained
unexecuted. These counts are progress evidence, not completion.

## Automatic admission for outer additive and tree controls

The existing control trainer now supports waiting for complete outer admission.
Nine control tests pass, including deferred admission without response reads,
changed-input and invalid-admission refusal, preservation of existing attempts,
original paired seed coverage and the existing variant/full-FIT routing checks.
The queue keeps the current fit owner, selected settings, SSD reservation and
native evaluation gates. Watcher tests use mocked fits and do not count as
scientific control fits. Qualification wrapper: `run-heavy: done rc=0 1s |
peak-RSS 0.07GiB | min-avail 32251MiB | peak-load 25.42`.

## Checkpoint retention at 10:22 UTC

Seventeen newly archived primary fits supplied 3,944 eligible intermediate
checkpoint duplicates totaling 768,849,168 logical bytes. Each local file and
its durable copy passed SHA-256 verification, including destination reread
after cache eviction. Two open-file checks and final inode/size/mtime checks
preceded removal. Required diagnostic epochs, final models and failed attempts
remain local; durable checkpoints are preserved. Free bytes measured directly
before/after removal were 59,260,928,000 and 60,018,589,696 while other jobs wrote
outputs. This remains below the then-current 60,129,542,144-byte admission
requirement for an additional pending fit; the reserve was not lowered.

Wrapper: `run-heavy: done rc=0 30s | peak-RSS 0.10GiB | min-avail 32132MiB |
peak-load 24.59`. The plan, removal receipt and log passed complete destination
readback in `checkpoint-retention-1022-qualification`: three artifact files,
4,977,414 bytes. These storage operations do not add scientific fit coverage.

## Native generalization queue admission

The existing bounded outer dispatcher now has an explicit neural-variant mode.
It requires the exact registered 168 neural cells per outer fold: source curves,
row curves, shuffled labels and human-only trials for ReLU and the three
inner-selected neural finalist roles. This covers 504 registered cells across
three folds; the 168 nonneural variant cells require their control launch path.
No outer scores enter finalist choice. The dispatcher verifies prepared variant
views through the existing variant resolver and calls the existing native
variant trainer, preserving TRAIN-only preprocessing, absence declarations,
neutral-replay qualifications and SSD reservation.

The qualification exercises all 168 cells in a simulated two-slot queue,
requires each cell exactly once through the variant owner, checks admission
refusal before variant resolution/launch, checks exact matrix coverage, and
rejects altered checkpoints or failed attempts on resume. Existing outer queue
checks also pass. These tests use simulated child fits and are not scientific
experiments. Wrapper: `run-heavy: done rc=0 1s | peak-RSS 0.04GiB |
min-avail 30167MiB | peak-load 28.08`.

`just shaping-completion-dispatch-variants STUDY BINARY_DIR NEUTRAL HUMAN OUTER`
waits for complete outer admission with one worker and a 3 GiB scope per fold.
The resource bounds are independent of other queues; no global training lock
is introduced. Native child commands retain their own one-thread resource cap.

## Nonneural generalization queue admission

The same dispatcher now routes the remaining 168 registered control variants
through the existing additive/tree fit owners: 56 per outer fold, including
14 additive and 42 tree cells. The actual variant registration has 42 additive
cells with 42 distinct preprocessing cache keys, so their independent basis
preparations do not collide. The three tree seeds share only immutable view
inputs and retain independent fit directories.

A complete simulated 56-cell queue exercises both trainer modes, exact coverage,
two-worker admission and completed evidence verification. Changing a completed
prediction table without its bound report is rejected. Neural and original
outer queue regressions also pass. The tests do not perform scientific fits.
Wrapper: `run-heavy: done rc=0 1s | peak-RSS 0.04GiB | min-avail 32278MiB |
peak-load 25.29`.

`just shaping-completion-dispatch-variant-controls STUDY BINARY_DIR` processes
all three folds with one control fit at a time in an 8 GiB, one-thread scope.
Each fold waits for verified outer admission and each trainer retains its SSD
reservation. This bounded control queue can run alongside independent neural
queues; it is not a global training lock. Together, both dispatch modes cover
all 672 registered generalization cells. Execution and scientific evaluation
remain required; launch coverage is not experiment completion.

## Full-FIT search launch automation

The existing full-FIT execution owner now offers `--watch-search`. It waits for
all three complete outer admissions, registers or reproduces the global search
plan, and requires exactly the registered 192 setting-search cells. Native
fits remain independent jobs; additive and tree settings are grouped by role
and partition so the additive input cache is prepared only once per search.
There are 150 child jobs: 144 native fits and six control groups covering 48
fits. At most two jobs run concurrently in the queue. Existing attempts or a
previous queue require explicit reconciliation; failed children stop new
launches while already admitted children finish and retain their evidence.

Outer dispatch owns creation of shared partition inputs. This queue waits for
its preparation receipts before revalidating and reusing those inputs, avoiding
concurrent creation by two owners. Neither admission nor preparation reads
exposed development responses. Completion of the queue means search fit receipts
are available; final evaluation, setting selection and 16 final refits remain
separate requirements.

The first synthetic queue execution found a child-record tuple mismatch before
any scientific run; its failing log is preserved. After correction, the full
192-cell simulated queue executes each cell once with two slots. Checks cover
missing admissions, missing shared preparation, exact coverage and draining
children after failure. All 18 full-FIT tests pass. Wrapper: `run-heavy: done
rc=0 3s | peak-RSS 0.19GiB | min-avail 32108MiB | peak-load 24.22`.

## Full-FIT setting-selection supervision

The existing full-FIT selection owner now waits for the admitted search plan,
then for every registered candidate's completed evaluation and every neural
candidate's training-only convergence receipt. Readiness checks inspect metadata
only; the existing candidate verifier checks actual report/model/diagnostic
hashes and scope before any setting choice is written. It validates all six
roles before registering new choices, then reproduces all six decisions and
records their hashes. Failed fits, failed native gates and ambiguous completed
candidates stop the watcher with preserved evidence. An existing watcher record
requires explicit reconciliation.

The new checks exercise missing admission before selection reads, delayed neural
diagnostics, repaired-evaluation routing, failed-gate preservation and refusal to
write partial choices when the last role fails qualification. All 21 full-FIT
tests pass. Wrapper: `run-heavy: done rc=0 2s | peak-RSS 0.19GiB |
min-avail 32150MiB | peak-load 22.87`. These are orchestration qualifications,
not completed search experiments or newly selected settings.

## Final full-FIT refit queue

The same execution queue now supports the registered 16 final refits after
all six setting decisions exist and reproduce. It pins those decisions and
rechecks cross-role scope through the existing final-fit resolver before any
final input preparation. The exact matrix contains 12 native fits and two
control groups covering four fits. Two jobs may run concurrently; each final
fit still passes its own admission and SSD reservation. Final control routing
uses `--full-fit-refit`, preserving descriptive development evaluation and
excluding those results from setting selection.

The simulated final queue covers all 16 cells exactly once, verifies two-slot
admission and checks that a missing sixth role prevents both preparation and
launch. Earlier search, failure-draining, selection and native execution checks
remain green: 23 full-FIT tests pass. Wrapper: `run-heavy: done rc=0 2s |
peak-RSS 0.19GiB | min-avail 32204MiB | peak-load 23.48`. These qualification
fits are simulated, not scientific results. The queue's terminal stage explicitly
requires reports; it does not certify inference costs, archives or study completion.

## Registered descriptive regularization contrast for input responses

Before inspecting new curves, extend the existing metadata comparison to the
complete paired setting-0/setting-1 fits on inner partition `inner-0-1`.
Both settings use learning rate 0.001; L2 is respectively zero and 1e-5.
Compare all three registered seeds for each identity-initialized product and
separate family, plus their direct-term intersection across both families.
This is a fixed twelve-fit metadata comparison, not a quality-based selection.
It tests sensitivity to regularization on one training partition; it cannot
establish source generalization or downstream utility. Existing exact-f32
curve normalization, domain boundaries and treatment of zero terms are retained.
Five tests pass, including exact paired coverage and refusal of missing seeds
or duplicate settings. The original single-setting/two-partition mode remains.

## Measured response stability across regularization settings

The registered inner-0-1 contrast completed for both architectures, all three
seeds and both L2 settings. Product shaping has 35/984 terms active in all six
fits and 702 zero throughout; separate shaping has 39/984 active throughout
and 721 zero throughout. Across both families, 14/492 direct terms remain
active in all twelve fits; 289 are zero throughout.

The five smallest normalized spreads among those fourteen direct terms are:

| Direct feature | Maximum normalized response spread | Maximum/minimum response strength |
|---|---:|---:|
| f506 | 0.0009952661275045216 | 4.648842792356843 |
| f587 | 0.05075509235122355 | 1.7646469859154306 |
| f98 | 0.0638835321321295 | 3.0533278791481813 |
| f500 | 0.06871856240492602 | 4.545224377112383 |
| f428 | 0.1040924590395571 | 3.1234335160221436 |

f403 remains active in all twelve, with spread 0.25057572542573014 and
strength ratio 3.6570070057787825. Within product shaping alone, f597, f506,
f23, f589 and f19 have spreads from 0.00021089314417922704 through
0.0012399178272332753 across the six matched regularization fits.

This is stronger evidence of same-partition repeatability than of transfer
between source partitions. In the earlier setting-0/two-partition comparison,
f506, f587 and f98 were active in only six of twelve fits, and f500 in seven.
f428 was active in all twelve but its spread was 0.9311656651542348, versus
0.1040924590395571 in the new same-partition contrast. f403 remains active
throughout both comparisons. The comparisons share six fits and therefore
must not be described as 24 independent trainings. No downstream-weight or
quality-utility claim follows from normalized input-curve stability.

All three `regularization-curves-o0-i1-*` directories passed complete destination
checksum readback: 30 artifact files, 6,665,737 bytes. They retain every compared
native shaping configuration, fit/input hashes and analysis logs. Outer
predictions and exposed development results were not used.

## Checkpoint retention and fresh-control admission at 10:44 UTC

Sixteen newly archived fits supplied 3,480 intermediate checkpoint duplicates,
789,651,408 logical bytes. The existing retention owner reread and SHA-256
verified every durable copy, checked local hashes and file identities, and
performed two open-file checks before removal. Required diagnostic epochs,
final models and failed attempts remain local. Free bytes measured immediately
before/after removal were 59,193,442,304 and 59,987,755,008 with concurrent writes.
Wrapper: `run-heavy: done rc=0 29s | peak-RSS 0.10GiB | min-avail 32088MiB |
peak-load 25.52`. The three-file plan/receipt/log archive passed full checksum
readback: 4,368,988 bytes in `checkpoint-retention-1042-qualification`.

Fresh control `de1b752f8826a840b5e7` was independently admitted at 10:40 UTC,
before this retention pass. Its live process and epoch-8 checkpoint of the
registered 120-epoch fit were verified at 10:44 UTC. The other queued seed
still awaited pending-write headroom. This run must not be attributed to the
later reclamation, and its partial checkpoints do not complete the fit.

### Undefined frozen-shape fit and evaluator recovery

At 10:49 UTC the original inner evaluator process was absent. Its retained v6
log ends in a null-score subtraction in `shaping_convergence.diagnostics`.
Cell `7acaa2df5782f9d05f08` is product-pretrained-frozen, inner-0-0, seed 0,
setting 1. Its checkpoint panels report undefined human composites and zero
active output coefficients. This is an optimization failure requiring
investigation, not evidence that the shaping architecture is ineffective.

The reporter now explicitly refuses null/nonfinite same-cycle scores. The
supervisor records an immutable hashed convergence failure and evaluates peers;
it does not emit a convergence receipt for the failed fit. All existing
thresholds and scientific admission rules remain unchanged. Five convergence
and eight supervision tests pass, including undefined endpoints, repeated
failure preservation, and continuation to an independent cell.

The restarted supervisor completed a sweep at 10:51:27 UTC and verified its
live PID. The failed fit's final epoch 119 and diagnostic epochs 49, 99, 119 all
have zero active output coefficients and undefined human composites. Its complete
retained attempt was archived with cold destination SHA-256 readback:
233 artifact files, 11,619,827 bytes, manifest
`f4b9950e8ad2f3f7cbe178834f2542cb1533a130b3368757b566298177a280fb`.
The archive is `failed-frozen-product-7acaa2df5782f9d05f08` under the study's
existing durable root. No local artifact was removed.

The progress audit now separately counts convergence failures and verifies their
input hashes. Four audit tests pass, including tampered failure evidence refusal.
The observed audit has 248/816 primary fits, 247 final reports, 150 diagnostic
receipts, one convergence failure, and five original native-gate failures.
A trained fit and an archived failure remain distinct from scientific completion.
Undefined final scores also prevent this family's existing paired base selection;
its registered rescue admission still needs resolution without imputing ranks or
dropping paired cells. The independent families continue.

### Native checkpoint trace of the zero-output frozen-shape fit

The native reader inspected all 120 scheduled checkpoints of
`7acaa2df5782f9d05f08`, with the original build and every input hash pinned.
At epoch 0, the three layers retain 64, 0, and 4 nonzero stored weights.
At epoch 1, all three counts become zero and remain zero through epoch 119.
The second hidden layer therefore already blocks the learned signal at the first
saved checkpoint; the subsequent disappearance of output weights is not the
first observed failure. These are stored-weight counts, not activation or
independent-parameter counts. The native training log independently reports zero
rank correlations from its first epoch; the later evaluator did not create this
collapse.

The frozen input manifest retains positive slopes in 101 of 984 terms, so an
entirely zero shaper bank does not explain it. The parent pretraining checkpoint
at epoch 119 retains 600, 16, and 2 nonzero stored weights in its three exported
layers. Its exported shaping representation includes routing coefficients;
these counts do not prove equivalence of hidden capacity to the fresh frozen head.
No causal claim about learning rate, regularization or gradients follows from
these structural observations alone. An optimizer trace remains necessary.

`just shaping-completion-structure-trace` adds this analysis to the existing
convergence owner. It requires the complete scheduled checkpoint sequence and
pinned native reader, rechecks input hashes after inspection, preserves existing
outputs, and never computes scores or selects checkpoints. The real 120-epoch
trace and six convergence/structure checks passed.

The structure diagnosis bundle has a cold-readback verified durable copy under
`zero-output-structure-diagnosis`: five artifact files, 183,630 bytes, manifest
`adf6b54c16635374e51d683def060ca88e02170658471ae4f59e9319e989714a`.
It contains the complete checkpoint trace, parent structure observation,
pretraining provenance, resource log and measured summary. The original failed
fit archive remains separate and unchanged.

### Qualification requirement for the zero-output optimizer replay

Before replaying the frozen product-shape failure, require canonical equality at
all 120 saved epochs, not just the final all-zero network. Equality of a collapsed
final network cannot validate the transient optimizer trace. The existing replay
owner now offers `--all-checkpoints`, retaining every stripped original/replay
comparison and refusing any mismatch. A fixture deliberately matches the final
checkpoint while changing an earlier one and verifies the earlier mismatch is
retained. A separate test verifies storage reservation precedes launch.

Diagnostic replays now use the same pending-write reservation owner as scientific
fits, preserving the 30 GiB reserve, auxiliary allowance and one-job allowance.
This does not change scientific fits, selection, training parameters or inputs.
The proposed replay uses the already qualified optimizer binary and the failed
fit's exact original command, seeds and frozen shape manifest, with diagnostics
activated. It is excluded from model selection. Its result is not yet available.

The all-checkpoint replay controller was verified live while waiting for storage
admission; no training claim is made before its RUN receipt and native process
exist. Four newly archived cells permitted removal of 812 redundant local
checkpoints (171,464,073 logical bytes) after cold destination readback and two
open-file checks. Required checkpoints and failed fits were preserved. The
retention plan, removal receipt, resource log, replay request and qualification
test log have a separately verified durable copy in
`replay-and-retention-1058-qualification`.

A proposed policy amendment is awaiting owner approval: mark a whole setting
ineligible when any paired human score is undefined, retaining all failed fits,
and choose the rescue base among fully defined settings. The existing finite
four-setting requirement remains in force. No individual seed has been dropped,
undefined score imputed, or rescue selection changed.

### Paired pretrained-product collapse and a separate analytic order failure

Fresh unshaped control `d3b2f4deeaa0ab19d572`, frozen product
`7acaa2df5782f9d05f08`, and jointly updated pretrained product
`e2818d1db67981205488` share inner-0-0, setting 1, seed 0, native build,
feature-list hash, fresh-head seeds 1001101/1000101 and 120 fresh-head epochs.
After replacing only output/checkpoint/feature-list paths, their command arrays
are identical. The two shaped fits start from identical pretrained shape terms;
only their frozen/trainable flag differs. The unshaped control retains one output
coefficient with inner human composite 0.9211705296939692. Both pretrained
variants end with zero output coefficients and undefined composites.
This isolates a failure associated with introducing this pretrained bank in this
matched setting, but does not establish the optimizer mechanism or an architecture
ranking. The comparison and input pins have a verified durable archive under
`zero-output-paired-control-diagnosis`.

The second undefined fit was archived separately: 233 artifact files,
16,880,273 bytes. Softplus `6e988eec6f23b804ff31` independently fails the unchanged
final order probe with one increase of 0.00001239776611328125 (limit 0.00001),
finite outputs and exact identity 100. Its failed final and epoch-239 gates remain
failed; 454 artifact files, 59,319,043 bytes have a cold-readback verified archive
under `failed-softplus-6e988eec6f23b804ff31`. No tolerance was changed.

An early prefix comparison against the running optimizer replay was refused:
intermediate bakes lack final reproducibility metadata until the native trainer
finishes. The incomplete comparison is preserved in
`zero-output-prefix-qualification`; it does not qualify the provisional trace.
The registered full all-checkpoint comparison still runs after trainer completion.

### Softplus order failure: scalar reproduction and direct-max diagnostic

The native inspector rebuilt from zenanalyze `8bf61bc9` reads the analytic
activation; the older inspector refused activation byte 4. The independent
scalar float32/FMA trace exactly reproduces both failing native scores for pair
35: -11.888920783996582 → -11.888908386230469. Float64 dense algebra from the
stored float32 coefficients instead decreases by 0.000004769070841348366.
For tied first-layer units 125/127, the float64 activation sum increases by
0.000010011641109031189 while the float32 sum decreases by 0.00000762939453125.
The trace extends the existing owner with the native stable Softplus construction;
its native-bit equality guard remains mandatory. Two boundary/refusal tests pass.

The exported nonzero signed pairs are 123/124, 122/126 and 125/127. Assuming the
training-time adjacent-pair layout was correctly refused by the native tied-weight
validator; export ordering must be read from the model. With the actual pairs,
the existing native research direct-max operator yields zero violations across
all 128 registered order probes, maximum increase zero, finite outputs and exact
identity 100. Pair 35 becomes a tie at -11.889023780822754. No refitting occurred.
This establishes a working diagnostic repair for this failure, not complete
qualification or permission to mix representations in model selection.

The first diagnostic wrapper also counted identity as an extra probe, although
it is already the first of the nine ray rows. Correcting that new wrapper to the
existing 265-row layout leaves the original study gates unchanged. All failed
attempts are retained separately. The successful trace and probe-only repair
are archived as `softplus-order-probe-diagnosis-r2` and
`softplus-maxpair-diagnostic-r3`, with cold destination checksum readback.

### First-layer weight-gradient observer: native implementation checks

The optional `ZENSIM_OPTIMIZER_DIAGNOSTICS` path now samples first-layer weight
pair gradients every 128 optimizer pairs, before input-shape and weight updates.
It uses the existing canonical backward and coupled-L2 functions in separate,
lazily allocated scratch state. It never shares the training optimizer's gradients
or moments. `FIRST_WEIGHT_DIAGNOSTICS` reports mean squared pair-gradient norms
before and after coupled L2, not means per coefficient; it excludes Adam moments,
projection and decoupled decay. Existing activation-gradient records retain their
schema and meaning.

The finite-difference test checks the sampled data gradient and the existing L2
owner checks the regularized result. The complete native library suite passed
402 tests (two existing ignored), the CLI suite passed 65, and strict release
clippy passed. The initial cached-dependency failure and subsequent type-error
attempt remain preserved. The corrected native source is snapshot `1eb2f09d`
with the recorded `shared_depth.rs` correction hash. This is implementation
evidence; a full canonical-neutral replay is still required before attributing
new weight-gradient traces to an original scientific fit.

## First-weight replay coverage gate

The diagnostic replay owner now has an explicit first-weight trace requirement.
It refuses missing, duplicated or reordered epochs before qualifying a replay,
and preserves the sampled weight traces separately from activation diagnostics.
The dedicated recipe also requires canonical parity at every scheduled checkpoint.
Three replay tests pass, including transient model drift despite matching final
models and exact weight-trace epoch coverage. Native replay qualification remains
pending; these tests do not establish observer neutrality on scientific fits.

## Collapsed-fit optimizer replay: complete checkpoint parity

The frozen-product cell `7acaa2df5782f9d05f08` completed its full 120-epoch
diagnostic replay. All 120 canonical checkpoints match the original after
removing run metadata. Final canonical SHA-256 is
`511f6d4e47bc5fc828e97ae9c65e68b0e15dae925afcaab318294f69c318c2fa`.
This qualifies attribution of the existing activation/output optimizer traces
to that original fit; it does not qualify the new first-weight observer.

Epoch 0 records 60 zero output weights and 353954 output projection hits.
Epoch 1 records 64 zero output weights and 23623 projection hits; its sampled
first-activation gradient squared norm is zero, as it is at every later epoch.
At epoch 2 all 49408 sampled first activations are zero, and the recorded
regularized output-gradient squared norm is zero. These observations locate
the collapse during optimization; they do not isolate the causal contribution
of first-weight data gradients, L2, Adam history or projection. The new
weight-gradient replay remains required for that investigation.

The replay was archived with cold destination readback: 368 artifact files,
24966111 bytes, SHA manifest
`291b783df9dbfd1eca594214a7675e4bc6d2c4bc5a1df999e834309352d62add`.
Artifact suffix: `zero-output-optimizer-replay` beneath the completion-study
local and durable roots. The replay remains excluded from model selection.

## Collapsed-fit first-weight replay: complete checkpoint parity

The additional first-layer weight observer completed the same frozen-product
cell `7acaa2df5782f9d05f08` for all 120 epochs. Every canonical checkpoint
matches the original, including transient states. The final canonical hash is
`511f6d4e47bc5fc828e97ae9c65e68b0e15dae925afcaab318294f69c318c2fa`.
This qualifies the new weight traces for this fit; the observer does not change
its optimizer trajectory.

At epoch 0, the sampled mean squared data-pair weight-gradient norm is
60.93524851312713 and the norm including coupled L2 is 60.93488391083481.
At epoch 1, they are 0 and 1.824576873014883e-51. Both are zero at every
epoch from 2 through 119. Each of these first three epochs samples 386 pairs
with stride 128. These are squared pair-gradient norms, not per-parameter means
or full-population measurements. They exclude Adam moments, projection and
decoupled decay. Similar epoch-zero aggregate norms do not prove that L2 had
no effect on individual coordinates, and the traces do not isolate the causal
contributions of Adam history or projection to collapse. No architecture-wide
rejection follows from this single failed fit.

The complete replay passed cold archive readback: 369 artifact files,
25060478 bytes, under `zero-output-first-weight-replay` in the completion-study
local and durable roots. SHA manifest:
`6c1ee1651006d739cb910be96f5ef0c9d4628f453f05f99ee00b9fe1c80c66b4`.
The launch wrapper reports rc=0, 2337 seconds,
peak RSS 0.09 GiB, minimum available memory 34940 MiB, peak load 26.68.
These are replay resource observations, not inference benchmark measurements.
The replay remains excluded from scientific model selection.

The next diagnostic comparison is the exactly matched unshaped fresh-head
control `d3b2f4deeaa0ab19d572`: same partition, setting, seed pair and 120-epoch
head exposure as the collapsed frozen-product fit. The replay owner admits
registered fresh ReLU controls with that unchanged exposure, still excludes
outer/full-data runs, and requires all-checkpoint parity before attributing new
weight traces. Its archived intermediate checkpoints are restored with hashes
before replay. This is a diagnostic replay, not another tuning opportunity;
no comparison result is claimed until replay qualification completes.

The matched replay was initially held by the unchanged storage gate, then
admitted after inactive compiler-intermediate reclamation. Its first weight
trace has been emitted; complete checkpoint neutrality remains unverified.
Restoration preserved and verified all 120 original checkpoints, including
116 restored files totaling 9149742 bytes. Keep that sequence until replay ends.

The subsequent storage wave removed 5916 local checkpoint duplicates from 27
completed cells, totaling 1706658047 logical bytes, after cold SHA-256 readback
of every durable counterpart and two open-file checks. It excludes the matched
replay originals. Available SSD bytes measured immediately before/after removal
were 62171017216 and 63885348864. The wrapper reports rc=0, 87 seconds,
peak RSS 0.10 GiB, minimum available memory 35811 MiB and peak load 24.48.
The plan, removal receipt and log are preserved in
`checkpoint-retention-1226-evidence`; separate compiler-cache receipts are in
`compiler-cache-reclamation-1227`. No scientific thresholds or storage
allowances changed.

## Supplemental analytic re-export qualification scope

Before evaluating repaired Softplus quality panels, extend the existing native
re-export owner with explicit inner-only diagnostic admission for Softplus.
The native operator still admits only valid first-layer ReLU pairs with tied
following coefficients; the later anchored Softplus layer remains unchanged.
Evaluate the failed cell's final checkpoint and every registered convergence
checkpoint. Retain original failures and use the existing feature-order, identity,
parity and panel gates. The selection owner still refuses non-ReLU re-exports;
these supplemental panels cannot select a Softplus setting or admit outer data.
The default exporter and running ReLU supervisor retain their previous scope.

## Softplus supplemental panels and convergence

Cell `6e988eec6f23b804ff31` now has full supplemental re-export panels for
the scheduled final checkpoint and epochs 49, 99, 119, 149, 199, 219 and 239.
All eight evaluations pass the unchanged finite, exact-identity and damage-order
gates with zero probe violations. Final inner human composite is
0.9052339145577766, with two active output coefficients. This is one cell's
inner diagnostic, not a family comparison or a source-generalization result.

The same-cycle composite change from epoch 119 to 219 is
0.001105667166776958. Late mean training loss changes from 1.466835 to
1.36391; the existing loss-improvement trigger requires investigation. No
convergence threshold changed, no original failure was removed, and the
selection owner still refuses supplemental Softplus representations.

Cold destination SHA-256 verification passed for both bundles:
- `softplus-reexport-panels-6e988eec6f23b804ff31`: 209 files, 8283740 bytes;
  manifest `1874573b94c221bd2da9946f8c427a027dcfa12925b72300126ebff8c5d9f93d`.
- `softplus-reexport-bakes-6e988eec6f23b804ff31`: 48 files, 24980391 bytes;
  manifest `80e7e5c12c3bacc9426ca4d93c164c948450dd720230f44f67c30c44a76f6537`.

The explicit analytic admission test, fifteen evaluation tests and six
convergence tests pass. Full paired uniform-representation qualification remains
unexecuted; these supplemental results do not admit outer evaluation.

The supplemental supervisor now accepts an explicit Softplus family scope,
covering every completed registered inner fit and its scheduled diagnostic
checkpoints uniformly, including future completions. Its default remains ReLU,
and separate status records prevent the two supervisors from overwriting each
other. Family isolation is tested. This starts uniform evidence collection,
not selection admission; original reports and native failures remain unchanged.

The first-weight observer build source is durably preserved in
`first-weight-build-source`: source snapshot `1eb2f09d74dd60f6e480bfdd6e9d032085389b9f`
plus the corrected `shared_depth.rs` and reconstruction instructions. Complete
cold destination checksum readback verified 3 files and 111338813 bytes. The
remote training-module, shared-depth, root manifest and lockfile hashes match
the landed implementation. Binary-build and neutral-replay qualification remain
separate pending gates.

## Supplemental archive supervision

The existing archive owner now has a separate supplemental mode. Before
mirroring panels and native re-export evidence, it requires complete registered
diagnostic checkpoint coverage, final and diagnostic completion markers, source
checkpoint hashes, repaired-bake and artifact hashes, and a consistent evaluator
identity. It preserves incomplete or failed evaluations without archiving them
as complete and does not reclaim local data. A negative-control test refuses
missing checkpoint coverage and corrupted repair artifacts before any copy;
complete inputs mirror both bundles and are not recopied on the next sweep.

## First-weight binary set and native qualification archive

The observer binary set built successfully with release thin LTO and one
code-generation unit. The resource wrapper measured 591 seconds, 1.70 GiB
peak RSS, 22944 MiB minimum available RAM and 20.56 peak load; this contended
build duration is not an inference benchmark. Copied binary hashes match the
remote build outputs. Complete cold destination verification passed for
`first-weight-bin` (5 files, 32876354 bytes) and
`first-weight-native-qualification` (10 files, 677200 bytes).

The full 120-epoch replay controller was launched with required per-epoch weight
traces and all-checkpoint canonical matching. Storage admission initially held
it because pending-job allowances exceeded available SSD space. No training
started through that hold, and no reserve or allowance was weakened.

The 11:42 checkpoint-retention pass removed 2204 local duplicate checkpoints
from 11 completed fits, totaling 435056860 logical bytes. Every removed file
passed local hashing, cold durable readback, unchanged-file checks and two
open-file checks. Registered diagnostic/final checkpoints and failed runs remain
local. The plan, receipt and execution log passed durable archive verification
in `checkpoint-retention-1142-evidence`. Replay admission remained pending after
this pass; pending-write allowances and the 30 GiB reserve are unchanged.

## Replay storage admission restored

The first incremental-cache cleanup was on the separate root filesystem and
did not release study-SSD space. A second pass targeted inactive Cargo compiler
intermediates on the verified study filesystem: 2786 files, 3051351244 logical
bytes. Before/after filesystem free space measured 56891715584 and 59452190720
bytes. Both process checks found no target-directory users; only `.rlib`,
`.rmeta`, `.o` and `.bc` files were removed. Executable binaries and generated
experiment data were retained. The cleanup plan and receipts passed durable
readback in `compiler-cache-reclamation-1145`.

The first-weight replay then passed the unchanged admission gate and started
its registered 120-epoch training. Full checkpoint parity and complete new
weight diagnostics are still pending; launch is not qualification.

## Analytic runtime comparison qualification

Before admitting supplemental Softplus results to selection, the existing
re-export owner now supports a separate full TRAIN/inner prediction comparison.
It pins the original training runtime and repaired runtime, reproduces stored
original predictions, requires bit-identical float32 unmarked-model outputs
across runtimes, and measures repaired-model score differences on every
population. It refuses missing, reordered or nonfinite prediction records and
checks all input hashes again after scoring. This diagnostic writes separate
artifacts and never changes study reports or chooses settings.

Softplus runtime comparison completed on ten TRAIN/inner population tables.
Original-runtime replay and unmarked cross-runtime float32 predictions match
exactly. Maximum repaired-model score change is 0.0002899169921875. The separate
`softplus-runtime-comparison` archive passed full destination readback:
81 files, 4183261 bytes. No outer responses
were consumed, and selection admission remains unchanged.

## Uniform Softplus representation admission implementation

The existing repaired-representation selection and rescue gates now admit
Softplus alongside ReLU. Every paired report still requires one representation
and evaluation binary set; final/diagnostic agreement, full seed/fold/setting
coverage and all finite human scores remain mandatory. Later-stage re-export
resolves the actual fitted family and requires its verified inner representation
admission. Explicit inner diagnostic scope is unchanged. The three re-export,
seven selection, eight rescue and nine supervision tests pass, including mixed
runtime/representation refusal and later-stage admission rejection. Actual
paired coverage and rescue completion are still required before outer fitting.

## Matched uncertainty computation validation

The source-bootstrap owner now computes paired candidate/control arrays with
all three outer folds and the exact registered seed set. It reuses the existing
native rank-panel and whole-source draw implementation, verifies each point
estimate against its completed report, and retains separate model-seed axes.
The matched reducer keeps teacher agreement separate and reports paired
candidate-minus-control intervals. Six reducer tests pass; the native runner
suite, including the matched comparison and missing-seed refusal, also passes.
This validates computation only. Variant/fresh-control provenance collection
and actual registered scientific bootstrap executions remain unfinished.

## Matched uncertainty provenance and runner

The existing outer-evidence owner now collects variant/full-data and
pretrained/fresh pairs. All three outer admissions, every terminal fit, exact
phase exposure, native model hashes, report scope, repaired representation and
prediction hashes are validated before any prediction array is opened. The
bootstrap runner accepts an explicit comparison JSON and pins its analysis
policy alongside the original source draws. Nine evidence tests pass, including
complete deterministic and neural-pair fixtures and pre-read refusal of missing
fits, model corruption, phase-exposure drift and absent outer admission.
Six native runner tests pass, including complete matched summary/artifact
production; wrapper peak RSS was 0.21 GiB. These are implementation tests, not
completed scientific comparisons. Actual outer/variant fitting remains required.

## Complete matched-comparison registration

The bootstrap plan owner derives 88 fixed comparisons from the registered
matrix: 84 variant/full-data comparisons (six finalist roles, six source
fraction/order conditions, three row fractions, four shuffle conditions and
one human-only condition) plus four pretrained/fresh comparisons. Every run
requires the complete comparison registry, unchanged analysis policy, source
draw settings and all specification hashes. Five registration tests and six
native runner tests pass, including refusal when another registered comparison
specification changes. No scientific outer bootstrap has executed yet.


## Qualified matched unshaped first-weight replay

The matched fresh unshaped ReLU control `d3b2f4deeaa0ab19d572` completed the
120-epoch first-weight observer replay. All 120 canonical checkpoints match the
original training run; the final canonical hash is
`a3ee84cc6140485e9b339c413ef6a4d01c6f5fd74055ac51ea73da2bc7586cc8`.
Its restored original checkpoints remained available throughout qualification.

The sampled first-layer data pair-gradient squared norm is positive in all
120 epochs: minimum 259.6048862165868, maximum 3335.051155762056. Values at
epochs 0, 1, 2 and 119 are respectively 1072.1407198492502,
798.7777045375468, 376.92650152863183 and 529.2501790020109. At epoch 119,
the data-plus-coupled-L2 norm is 529.2501693345246; the model retains one
nonzero output coefficient (63 of 64 are zero).

The separately qualified frozen-product replay has a positive sampled data
pair-gradient norm only at epoch 0 (60.93524851312713), followed by 119 zero
values, and ends with all 64 output coefficients zero. This contrast establishes
that the matched unshaped control retains sampled first-layer gradient signal
under the same registered fresh-training seed and exposure. It does not isolate
L2, Adam or projection as the cause of collapse, prove that one surviving output
uses the nominal hidden capacity, or demonstrate generalization. The different
input transforms also prevent interpreting the gradient-norm magnitudes as a
common quality scale. These are sampled pair-gradient squared norms, not means
per parameter; they exclude Adam moments, projection and decoupled decay.

The training wrapper reports `rc=0 3233s | peak-RSS 0.09GiB | min-avail 35530MiB |
peak-load 31.04`. This supervised diagnostic run is not a runtime benchmark.
The complete `unshaped-first-weight-replay` archive passed durable checksum
readback: 369 files, 28,370,021 bytes, retaining both checkpoint sequences,
canonical comparisons, traces, source/binary identity and full logs.


## Verified checkpoint retention at 14:54 UTC

The existing retention owner removed 15,196 local intermediate checkpoint
copies from 70 completed runs, totaling 3,601,657,508 logical bytes. Every
removed file matched its local SHA-256 and a fresh durable readback; open-file
checks ran before verification and again before removal. Final checkpoints,
registered diagnostic epochs, epoch 119 and pinned inputs remain local.
Complete checkpoint histories remain in their durable archives. No training
was replayed or restarted.

Filesystem free space was 75,917,742,080 bytes before removal and
79,537,983,488 bytes afterward, while other jobs continued writing. These
filesystem observations are not an isolated allocation measurement.
The wrapper reported `rc=0 126s | peak-RSS 0.12GiB | min-avail 30462MiB |
peak-load 32.07`.

The plan, per-file removal receipt and full log are preserved in
`/mnt/v/output/zensim/shaping-completion-20261107/checkpoint-retention-1455-evidence`
and the corresponding durable archive. All three artifacts (19,075,705 bytes)
passed destination checksum readback; the archive receipt SHA-256 is
`673bb0d0762cea75a027e0840fd5471e6778868b449521f12473ea0b6faadc64`.


## Worker checkpoint retention implementation

The worker retention command verifies imported RUN/RESULT identities against
complete durable transport bundles, rereads each bundle SHA-256, and verifies
worker checkpoint hashes before removal. It preserves final and diagnostic
checkpoints, rejects pinned inputs and multiple attempts, and checks open files
twice. Failed verification removes nothing; successful removal keeps an exact
per-file receipt. This applies only to completed, imported worker fits and does
not restart training. Three unit tests cover retained epochs, corruption,
metadata drift, busy files, pinned inputs, duplicate paths and extra attempts.
The completed worker qualification is recorded below.


## Worker reclamation and fresh-control admission at 15:02 UTC

The first worker cleanup refused before deletion because the global open-file
scan encountered unrelated stale NFS mounts and nonexistent portal exclusions.
The corrected implementation checks exact checkpoint filenames in bounded
argument batches, still refusing warnings or busy files. All four regression
tests pass. Both failed and successful operation logs are preserved.

On i270, complete durable bundles for 74 imported fits were reread and verified.
The worker then verified metadata, local checkpoint hashes and open-file state,
and removed 17,168 intermediate copies totaling 4,029,621,240 bytes. Final and
diagnostic epochs remain; complete histories remain in the archived bundles.
Free space changed from 52,775,612,416 to 56,827,318,272 bytes during concurrent
training. The worker wrapper reports `rc=0 21s | peak-RSS 0.06GiB |
min-avail 24352MiB | peak-load 16.89`; the coordinator reports `rc=0 87s |
peak-RSS 0.09GiB | min-avail 30800MiB | peak-load 32.27`.

The six evidence artifacts (9,862,780 bytes) passed durable readback in
`i270-checkpoint-retention-1500-evidence`, receipt SHA-256
`92080712f065b258bc0d1ba036c171379286c00540b5c30df3fb85a3a59590da`.
The newly assigned inner-1-1 fresh ReLU controls contain 12 previously unstarted
registered cells. Three workers were admitted with 56,729,882,624 free bytes
against 55,834,574,848 reserved bytes (30 GiB reserve, 3 GiB auxiliary allowance,
19 one-GiB pending-fit allowances). Their controller and first three children
were verified live. No completed fit was replayed. The 24,090,132-byte input
capsule and install receipt passed durable readback in `i270-fresh11-inputs`,
receipt SHA-256 `465b5debbaad42daea9b73bbd4d45aa80adc810fb795be01cad47ea79aeca80a`.


## Completed failure evidence archived at 15:07 UTC

Thirteen original fit directories with recorded numerical-gate or undefined
convergence failures passed durable artifact readback: 4,575 files totaling
431,258,975 bytes. Their scientific failure classifications remain unchanged;
archival does not qualify them for selection. Complete original checkpoints and
reports remain preserved. The first pass archived five cells before refusing a
rescue checkpoint path that omitted its declared `fit/` directory. The corrected
owner respects that declaration; the second pass archived eight further cells.
Four targeted failure-archive tests pass, including nested rescue checkpoints;
the preceding combined archive suite passed eleven tests before that addition.
The second-pass wrapper reports `rc=0 17s | peak-RSS 0.07GiB |
min-avail 29772MiB | peak-load 31.87`.

The inventory and both operation logs passed separate durable readback in
`failed-original-archive-1507-evidence` (three artifacts, 8,753 bytes), receipt
SHA-256 `2ed19635812ae2f22ae5f60ff4f43d4b3fecbfcb817a21c51df07f66511eb653`.
A dedicated failure archiver now runs from pinned source `6683c197c3cc`, avoiding
live source-edit drift. It skips existing verified archives and preserves new
completed failure records without rerunning training or changing scientific gates.


## Pretrained inner-1-1 workers admitted at 15:18 UTC

Twenty-four previously unstarted pretrained cells had completed matching parents.
Placement used sorted registered IDs, alternating between two workers, without
reading quality scores. Each host received twelve cells and one concurrent fit.
Every parent-prefix extraction passed the native metadata/shape checks on its
assigned worker before training. The original 120-epoch prefixes were reused;
no pretraining or completed fit was replayed. Scientific inputs and total
registered exposure remain unchanged.

At admission, i265 had 15 existing trainers and 284,392,497,152 free bytes,
against a 52,613,349,376-byte reserve including pending fits. i270 had 18
existing trainers and 56,198,414,336 free bytes, against 55,834,574,848 reserved.
Each new controller has a 2 GiB cap and one thread; child fits retain 1 GiB caps.
Both controllers and their first children were verified live with no reported
training failure. These are admission observations, not throughput measurements.

The source snapshot, placement, installation and all 24 extraction records
passed durable readback in `pretrained11-admission-evidence` (64 artifacts,
11,370,515 bytes), receipt SHA-256
`67eba32af8073f29466d04c54f830ce1c6481cd8a523fe559a764ff9e0cead5c`.
Both input capsules and install receipts also passed durable readback:
`i265-pretrained11-inputs` (31,262,233 bytes), receipt
`df616616e8a1878df8d673ab454b0682292074a9e2e1835a05cf4d0a472e29f0`;
`i270-pretrained11-inputs` (30,453,269 bytes), receipt
`6115b8627fda894179251dbdfdee9d2e9fdc0e3ac305741cc803afbe125d4ca2`.


## Saved training resource reader and additional ready pretrained fits

Source `37efb4dade1c` adds saved-run resource accounting. Eleven saved runs
qualified across ten neural families and a completed rescue. Three parser
regressions pass. No scientific fit was replayed. The qualification wrapper
reported `rc=0 2s | peak-RSS 0.04GiB | min-avail 32439MiB | peak-load 29.80`.
Full-parent wrapper cost and the observed native epoch-119 timestamp remain
separate from child cost; prefix-only wrapper time/RSS were not measured.
The reader evidence archive verified 25 artifacts, 181,928 bytes, receipt
`eae7921f5888cbb7ef52e306d21ab5aa5a71552416585f80c3bbe81c4a540b47`.

Eight previously unstarted inner-2-0 pretrained cells had four completed matching
parents. Their original prefixes passed native extraction qualification on i265
and were admitted with six workers using source `ce9229d0362d`. At admission,
nine existing trainers were present, available RAM was 25,439,903,744 bytes and
free disk was 285,267,173,376 bytes against a 52,613,349,376-byte requirement.
The controller retains a 2 GiB cap and each native child a 1 GiB cap. All six
native child PIDs were verified live; these observations do not measure speedup.
The assigned eight fits preserve the frozen matrix, seeds, exposure and gates.

Cold readback verified `pretrained20-admission-evidence`: 26 artifacts,
2,461,951 bytes, receipt
`0149b120fcad376b455c3822ea5fd8eb93addb1b8455c843d07b98c0d850d74f`.
The input capsule archive verified three artifacts, 26,992,412 bytes, receipt
`b016ffc9fedbc145db4b43e5e360bb60f7e3509e157943cc7c29226eb13eb949`.


## Saved resource coverage at 15:44 UTC

The batch reader inspected completed neural fits and wrote 332 per-fit resource
reports with zero refused completed fits; 1,445 concrete neural cells were still
pending. It neither loaded final holdout responses nor replayed training.
The resource wrapper reported `rc=0 11s | peak-RSS 0.04GiB | min-avail 20552MiB |
peak-load 32.67`. The three parser regressions passed. Evidence lives in
`training-resource-batch-1545` under the study output root. This is resource
coverage for available completed fits; the scientific experiment matrix remains
incomplete. Nonneural control costs are outside this reader's scope.


## Saved inner control resource accounting

The control reader qualified all twelve completed inner partition/family
batches, covering 72 tree fits and 24 shaped-additive fits. It verified exact
registered coverage, RUN/RESULT identity and PID agreement with the saved logs.
Two parser tests pass. The qualification wrapper reported
`rc=0 1s | peak-RSS 0.03GiB | min-avail 22137MiB | peak-load 31.75`.
The reports preserve whole-batch time/RSS including shared preparation, export
and evaluation, plus labeled per-fit civil-clock intervals. Per-fit peak RSS
and training-only timing are unmeasured. No training was replayed. Evidence:
`control-resources-1547.json` under the completion study output root.


## Newly admitted ReLU inner-2-1 wave

After i270 completed its analytic inner-1-0 wave, the twelve previously unstarted
registered ReLU inner-2-1 fits were assigned there with two workers. Frozen
input installation verified checksums and explicitly reused identical inputs.
The first admission check refused before training because 17 active trainers
were observed against an anticipated 16; that refusal and original command are
preserved. The corrected admission accounts for 19 aggregate single-thread
trainers on the observed 24-thread host, within its 20-thread compute budget.

Admission measured 56,143,802,368 free disk bytes against 55,834,574,848 required
(including the 30 GiB reserve, shared allowance and pending writes), and
25,424,977,920 available memory bytes. The controller has a 2 GiB cap; each child
retains its 1 GiB cap. Both native children were verified live. This starts no
completed fit again and changes no registered scientific input or schedule.
Additional admissions there are held until headroom improves. Worker source:
`ce9229d0362d`; commands and receipts: `relu21-admission-evidence`.

Cold readback verified `relu21-admission-evidence`: 11 artifacts, 16922 bytes, receipt
`167c165462d35362b00a8a6f411271ae471c22212f3d8c5a790bf8a3b13b6460`.

Cold readback verified `i270-relu21-inputs`: 3 artifacts, 24151161 bytes, receipt
`eb7aaedd2b94c77b63d6aa8459c3bc680b0b3e9678635f3f6c3dcce9f41e4a0f`.


## Incremental worker checkpoint retention at 15:53 UTC

Source `19c404ec6c3b` permits explicit completed prior retention receipts while
keeping all checks for newly imported fits. Five regression tests passed. The
i270 pass excluded 74 previously handled cells and verified eight new completed
fits against durable bundles and current worker checkpoint hashes. Both exact
open-file inspections passed before removal. It removed 1,624 intermediate
copies totaling 258,571,249 bytes; final/diagnostic checkpoints remain. Free disk
was 56,107,479,040 bytes before and 56,369,238,016 after. No training was replayed.
The coordinator wrapper reported `rc=0 10s | peak-RSS 0.07GiB | min-avail 21732MiB |
peak-load 32.19`; the worker reported `rc=0 2s | peak-RSS 0.02GiB |
min-avail 23829MiB | peak-load 19.05`.

Cold readback verified `i270-checkpoint-retention-1553-evidence`: 7
artifacts, 7207168 bytes, receipt
`0b8080b5d108a22a79a40651957a5caa523e8f5528feead9650b14c0d4f25fa5`.


## Full-setting descriptive shaping check within inner-0-1

Existing final checkpoints supplied 24 fits (four settings, three seeds, two
architectures). The unchanged native curve reader verified candidate provenance
and TRAIN shape layout. Eleven of 492 direct terms are nonzero in all 24 fits;
seventeen are zero in all. f403, f603 and f690 also survive the earlier
twelve-fit two-partition scope. Six fits overlap, giving thirty unique fits;
the union is not a balanced partition/setting comparison. f403's new normalized
spread is 0.2514907727773016, strength ratio 4.755047675175603. f506's new spread
is 0.0009952661275045216, but its zero response in six earlier fits contradicts
source-independent activity. f603/f690 strength ratios exceed 4,452, so activity
alone does not establish stable magnitude. No quality benefit is inferred.

Evidence: `native-direct-allsettings-inner01-1556`. The wrapper reported
`rc=0 1s | peak-RSS 0.04GiB | min-avail 22664MiB | peak-load 29.35`.
This reads stored parameters, not new training or predictions.


## Newly ready pretrained inner-2-1 pair

One newly completed matching joint-shaping parent released two previously
unstarted registered inner-2-1 pretrained fits. They were assigned to i265
with one worker, preserving its aggregate 16-thread budget. Both native prefix
extractions and checksummed input installation passed before admission. The
controller and first native trainer were verified live. Worker source remains
`ce9229d0362d`; no completed fit or pretraining prefix was replayed.

Admission observed 15 existing trainers, 24,569,634,816 available memory bytes,
and 284,626,165,760 free disk bytes against 52,613,349,376 required. The controller
has a 2 GiB cap; the child retains a 1 GiB cap. Placement, commands and parent
provenance are retained in `pretrained21-admission-evidence`.

Cold readback verified `pretrained21-admission-evidence`: 15 artifacts,
609304 bytes, receipt
`e5dbdb38794c71896f6b036f7116dae467d67143b790642200d94f9331cad3fb`.

Cold readback verified `i265-pretrained21-inputs`: 3 artifacts,
25094457 bytes, receipt
`1b0a321d3e959cf5ab14dd152256a252d697edd3fff71fb9f876ff622823980a`.


## Completed local checkpoint retention at 16:20 UTC

The existing retention owner verified archive identity, local and durable
checkpoint hashes, required retained epochs and both open-file inspections.
It removed 8,236 intermediate copies from 42 completed runs, totaling
1,610,531,022 logical bytes. Final and diagnostic epochs remain. Free space
was 76,345,839,616 bytes before and 77,966,934,016 afterward. No fit or
pretraining was replayed. Wrapper: `rc=0 66s | peak-RSS 0.10GiB |
min-avail 22983MiB | peak-load 30.77`.

Cold readback verified `checkpoint-retention-1619-evidence`: 4 artifacts,
5127468 bytes, receipt
`e974aa29d06516f8385500cd3fdadc826def675111c7288a5b2d42a22adc8368`.


## Second newly ready inner-2-1 pretrained pair

A second completed matching parent released two previously unstarted registered
pretrained fits. Both native prefix extractions passed before assigning the
pair locally with two workers. Admission observed 26 existing trainers,
25,652,330,496 available memory bytes and 77,756,665,856 free disk bytes against
65,498,251,264 required, including reserve and pending writes. The two new
trainers bring aggregate training demand to the host's 28-thread budget.
Both native PIDs were verified live. The controller retains a 2 GiB cap and
children retain 1 GiB caps. Frozen scientific inputs and training exposure
remain unchanged; no pretraining or completed fit was replayed.

Cold readback verified `local-pretrained21-admission-evidence`: 9 artifacts,
619384 bytes, receipt
`eb17adf8211b14febbc4e3479fdf692a8429343fdfe56bfd8ca73d75524f3228`.


## i270 retention at 16:28 UTC

Durable bundle cold readback and exact open-file checks qualified removal of
17 imported transport duplicates totaling 766,894,080 bytes. Original fit
directories remain. Free disk was 55,865,823,232 bytes before and
56,632,745,984 after. Coordinator wrapper: `rc=0 13s | peak-RSS 0.02GiB |
min-avail 24853MiB | peak-load 31.01`. Worker inventory and removal receipts
reported peak RSS 0.02 GiB, elapsed 0/1 seconds respectively.

The incremental checkpoint pass excluded prior retained cells and verified
seven newly imported fits. It removed 1,160 intermediate copies totaling
331,391,733 bytes, preserving required final/diagnostic epochs. Free disk was
56,607,264,768 bytes before and 56,930,967,552 after. Coordinator wrapper:
`rc=0 10s | peak-RSS 0.07GiB | min-avail 25647MiB | peak-load 31.09`.
No training or completed fit was replayed.

Cold readback verified `i270-retention-1628-evidence`: 6 artifacts,
505001 bytes, receipt
`35a0369593c37e9f977e0d95b4dc597a224c84afd0cb55b537d5f307c97538e1`.


## Reference preservation, compiler-cache reclamation and new i270 pair

Seven cost/inventory reference executables (20,757,904 bytes) were copied with
source paths and hashes, then cold-read verified in durable storage before
removing inactive Cargo output. No cargo/rustc process was present; both exact
open-file inspections passed and file identities were rechecked. The first
check refused before deletion on an additional Cargo future-incompatibility
report; that refusal remains. The corrected check admits this compiler report.
The removed output contained 3,827 files totaling 1,171,748,382 logical bytes.
Free disk was 56,606,826,496 bytes before and 57,760,239,616 after. Wrapper:
`rc=0 5s | peak-RSS 0.02GiB | min-avail 23891MiB | peak-load 19.32`.

This enabled one additional i270 worker for two previously unstarted registered
inner-2-1 pretrained fits. Both native prefix extractions passed and immutable
input installation verified checksums. Admission observed 19 existing trainers,
25,033,318,400 available RAM bytes and 57,673,064,448 free disk bytes against
56,908,316,672 required. The new single-thread trainer fills the host's
20-thread training budget. The controller has a 2 GiB cap, the child a 1 GiB cap.
Both controller and first native trainer were verified live. Worker source
remains `ce9229d0362d`; no training or completed pretraining was replayed.

Cold readback verified `i270-preserved-cost-binaries-1633`: 8 artifacts,
20760609 bytes, receipt
`d063cc8c87560ade70d07f6055f8a2475eecd85e3a3f44900bc1603dab05abe8`.

Cold readback verified `i270-pretrained21b-admission-evidence`: 19 artifacts,
646339 bytes, receipt
`fe8096327641094144ed34b90a04d0a3420bb44eb38900e13135fd2fc4ef5fa1`.

Cold readback verified `i270-pretrained21b-inputs`: 3 artifacts,
25104697 bytes, receipt
`df1f337c0869fa3c307c7efa7ae9f71b5bd8f35f153d8885a75d50ea5ceddf64`.


## Rescue resource-control qualification and idle queue upgrade

Source `f7e0ea10f51d` adds worker-budget and drain controls while retaining the
existing one-to-four bound. Five tests pass. An isolated child-process
simulation proves a drained queue retains two completed children and leaves
238 registered cells unstarted; no native training occurs in that test.
Admission counts observed native trainers plus this dispatcher's starting
children, and requires CPU headroom and 6 GiB available RAM before new launches.
Budget files cannot change scientific fields.

Outer fold 1 was already running two rescue fits when checked, so its idle-only
replacement guard refused before any signal or mutation. Active fold-0 and
fold-1 controllers remain unchanged. Only the idle fold-2 dispatcher was
replaced, with a verified empty process-child list and no fit interruption.
The new dispatcher is live, supervising its original 240 required cells at a
two-worker limit, with resource-file hash and drain state recorded. The mutable
control directory remains separate from the immutable evidence snapshot.
No native training or completed fit was replayed.

Cold readback verified `rescue-resource-controls-evidence-1650`: 10 artifacts,
38035 bytes, receipt
`0bb570925450025080b196b47ae330f0d079daa3a65dfaf839353dfc2209e524`.


## Four additional ready inner-2-0 pretrained fits

Four previously unstarted registered descendants from two newly completed
parents were assigned to i265 with four workers. The existing analytic
inner-1-1 and ReLU inner-2-0 waves had each completed their full 36/12 fits,
respectively, with no reported native training failure. All four native
prefix extractions and checksummed input installation passed before admission.
Both controller and all four native PIDs were verified live.

Admission observed nine existing trainers, 25,545,216,000 available memory bytes,
and 284,350,447,616 free disk bytes against 52,613,349,376 required. The
controller has a 2 GiB cap and each child a 1 GiB cap. The admission guard
allows at most twelve existing trainers for this four-worker placement and
requires 12 GiB available RAM. Worker source remains `ce9229d0362d`.
No completed fit or pretraining was replayed.

Cold readback verified `i265-pretrained20b-admission-evidence`: 19 artifacts,
1217036 bytes, receipt
`1c6c71d85672f9af8e1f7c630517b8e20aa52207675ed297f086e6a20c8df727`.

Cold readback verified `i265-pretrained20b-inputs`: 3 artifacts,
24767960 bytes, receipt
`5e44c1e035ffde8128fe7c5897a0125d384ec7494795ec78869ba01429a38cd0`.


## Shared admission for optional expanded rescue pools

Source `db8f1a97260f` retains the default four-worker capacity and adds an
explicit larger-capacity option bounded by the host CPU budget. Upgraded
queues reserve starting children through a shared short admission transaction;
the transaction lock is never held during native training. CPU admission
counts observed native processes plus shared claims not yet visible as native
trainers. RAM admission retains 6 GiB headroom plus 2 GiB for each such pending
child. Normal completion releases claims; unexpected owner loss requires
explicit reconciliation rather than silently freeing a slot.

Eight tests pass, including eight separate processes contending for a
four-slot fixture budget: exactly four were admitted. Tests also retain
unknown-owner claims, enforce default capacity and reject scientific fields
in resource controls. These tests execute no native training.

Only the idle outer-2 watcher was replaced, after verifying no active fits or
process children. It is live with declared capacity 16, an initial two-worker
limit and an 8 GiB controller cap. This prepares later scaling; it does not
claim sixteen scientific fits are active. Fold-0 and fold-1 active fits were
not interrupted, and no completed fit or pretraining was replayed.

Cold readback verified `rescue-shared-pool-evidence`: 15 artifacts,
63623 bytes, receipt
`3f802c4f8e51240ce33b6ec7c510b54ae5b49c7ca0b4059be368ca5260adf6dd`.


## Two further paired pretrained waves on i265

Two newly completed parents released four unstarted registered fits: a product
pair in inner-1-1 and a separate-shaping pair in inner-2-0. Both two-worker
waves passed native prefix extraction and checksummed immutable input
installation. All four native PIDs and both controllers were verified live.
Raw admission observations are retained in the evidence snapshot.

The first admission observed twelve existing trainers, 24,995,106,816 available
RAM bytes and 283,612,766,208 free disk bytes. The second observed fourteen
existing trainers, 24,720,584,704 available RAM bytes and 283,610,284,032 free
disk bytes. Each required 52,613,349,376 free bytes, including reserve and
pending writes. Together the new waves fill the 16-thread training budget.
Each controller has a 2 GiB cap and each child a 1 GiB cap; the admission
guard requires 12 GiB available RAM. Worker source is `ce9229d0362d`.
No completed fit or pretraining was replayed.

Cold readback verified `pretrained-ready-1715-evidence`: 28 artifacts,
1275647 bytes, receipt
`7419cf3ab50ec576ffb2196cd263cb2b6ccff68422bf33e6de73b64ef6da9680`.

Cold readback verified `i265-pretrained11b-inputs`: 3 artifacts,
25176377 bytes, receipt
`cdc3548758e01006f760eb77f6a2a4f60af676286c055136db88e9dbc9c5efc2`.

Cold readback verified `i265-pretrained20c-inputs`: 3 artifacts,
23753015 bytes, receipt
`d1f99bc39af0f34569852b8e1030135cfb290ee622dfdb2c64dad9e878cc0568`.


## Interrupted joint-shaping wave and unstarted-cell recovery

The inner-0-0 joint-shaping controller and two native fits were absent on
2026-11-07 after their scopes ended at 17:18:37 UTC. The product fit retained
19 of 240 scheduled checkpoints, and the separate fit retained 219 of 240.
The cause is unconfirmed. Neither has a final checkpoint; both are explicitly
recorded as interrupted with an unknown exit code, rather than successful.
Their partial files and original controller status remain preserved. Cold
readback verified 252 artifacts, 334457522 bytes, under
`interrupted-joint-00-evidence`, receipt
`d6fa13e6a927ce474d7a5a247af0d508e511337470f40ab98f9ec7a5c8a87271`.

The wave contains 13 successful fits, two interrupted fits and nine cells that
never started. Restarting the two partial fits awaits the owner's clarification
of the no-replay instruction. Existing native warm start restores model weights,
not exact optimizer and sampling state, so it cannot be claimed as a faithful
continuation. No partial or completed fit was restarted.

`shaping_inner_dispatch.py` accepts explicit registered inner cells only and
refuses every existing attempt. It uses the same shared CPU/RAM admission as
rescue dispatch, with the existing per-fit SSD reservation and native execution
owner. Successful children must preserve registered cell, plan, binary, epoch
and pair-budget identity and the final-checkpoint checksum. Drain requests and
invalid resource controls let live children finish before the controller exits.
Four queue tests and nine shared-admission tests pass without native training.
