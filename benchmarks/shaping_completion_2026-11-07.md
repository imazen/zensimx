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
