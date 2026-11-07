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
