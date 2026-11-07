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
