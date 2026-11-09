# Existing-worker efficiency observations — November 9, 2026

Eleven native trainers on r5600g were consuming CPU while advancing fewer epochs
than the observed r3800x trainers. These are different CPUs, seeds, concurrency
levels and training positions; the observations do not establish a speed ratio.
No training was replayed, stopped or numerically changed for this investigation.

A five-second r5600g sample of one product-shaping trainer recorded 4,261.35 ms
of task time, 18,777,039,707 cycles and 7,611,308,149 instructions. Its sampled
cycle profile attributed 51.96% to fused Adam, 12.87% to shaping backward and
10.34% to projection. A subsequent five-second aggregate observation of all
eleven trainers recorded 388 migrations and 21,071 context switches. CPU
frequency observations were 4.441 GHz; the governor label alone did not establish
underclocking. No affinity changes have been made.

Read-only native-state witnesses included 2,168,675 numeric values each, including
first-layer weights. At epoch 73 the selected r5600g state had 32 subnormal values;
at epoch 341 the selected r3800x state had 90. All occurred in depth.mid, with
none in either state's Adam moments or first-layer weights. This does not
establish subnormal Adam as the cause of the observed worker difference.
Changing arithmetic or flush-to-zero behavior is not justified by these data.
The probes used peak RSS 0.24 GiB under run-heavy; this is probe memory, not a
native trainer memory measurement.

Private artifact root: the study's `worker-efficiency-evidence-20261109` directory.
It preserves raw counters, the sampled perf file, readable per-array summaries,
individual native-state hashes and a compressed pair of bounded original state
witnesses. These snapshots are diagnostic witnesses, not qualified recovery
points. The study retains its scientific gates and registered exposures.

The initial resource evidence passed durable cold checksum readback: 26 payload
files, 4,022,872 bytes, plus the receipt. Both compressed native state witnesses
were independently decompressed and matched their original per-file SHA-256.

`just shaping-affinity-probe STUDY OUTPUT --pid PID --cpu CPU ...` performs a
bounded scheduling observation on existing study trainers. It records original
thread affinities, samples baseline/pinned/restored perf counters, and restores
original affinities in a finally block. Sampling failure restoration and
withdrawn-host refusal are covered. This tool does not change trainer arguments,
weights, snapshots, seeds, exposure or native arithmetic. Its implementation
checks do not establish a throughput improvement; an actual observation is
required before any scheduling recommendation.

The actual eleven-fit baseline/pinned/restored observation recorded respectively
836 / 2 / 715 CPU migrations and 172,131,043,830 / 210,217,888,489 /
161,789,228,369 retired instructions in ten-second perf windows. Aggregate task
times were 95,551.66 / 93,948.12 / 96,717.51 ms. All original affinities were
restored successfully. This suggests a scheduling benefit but is not an isolated
training speedup measurement. `--apply-observation` can reuse exactly these
assignments after verifying every process identity, original thread affinity,
worker eligibility and the completed observation's input hashes; `--restore`
reverts them. Native scientific inputs and eleven-fit concurrency are retained.

The exact observed CPU assignments were subsequently applied to the same eleven
verified trainers on r5600g, retaining their original affinities for rollback.
The application affects these processes only and ends when they exit. Its
observation, application receipt, original affinities and source passed durable
cold readback under `r5600g-affinity-evidence-20261109`: 14 payload files,
184,787 bytes, plus receipt. Eleven-fit concurrency remains; an epoch-throughput
speedup has not yet been established.

The fold-2 product fit was subsequently given its physical core's training
capacity: it remains on CPU 7, while the other trainer previously on sibling
CPU 1 moved to CPU 0. All eleven fits continued with the same code and inputs;
logical CPU 1 supplies headroom instead of CPU 0. Its ten-second samples recorded
14,735,546,216 versus 39,633,349,823 retired instructions. These counters alone
are not a training-speed measurement.

Existing epoch logs give an observed wall-time comparison. Five complete old
intervals, epochs 99–104, had median 155.8 s; five recent intervals after priority
pinning, epochs 119–124, had median 56.0 s. Transition-spanning intervals are
excluded. Epochs, learned sparsity and competing work differ, so this does not
isolate causality or establish a general concurrency optimum. No gradient or
completed epoch was replayed. The full points, resource times and affinity
journals passed durable cold readback under `critical-fold-affinity-evidence-20261109`:
21 payload files, 267,443 bytes, plus receipt. `shaping_epoch_windows.py` preserves
the interval selection and refuses missing or unordered epochs.

The supplemental archive observer previously reopened and hashed every native
bake, conversion artifact and report before reaching its already-archived check.
That repeated work on every polling pass. Closed panel/bake pairs now reuse
their original cold-readback receipts after checking both local/durable manifest
identities, the archive destination and the convergence record's archived hash.
Initial and partially completed archives still execute every original validation
gate before mirroring; no scientific eligibility or retention rule changed.

A read-only pass verified all 931 closed supplemental pairs through this path
without reading native bakes: `run-heavy: done rc=0 6s | peak-RSS 0.02GiB |
min-avail 44754MiB | peak-load 1.81`. This is metadata verification cost. The old
full pass was not replayed to manufacture a timing comparison. Tests retain
the first-archive refusal cases and verify that closed pairs read no native bake,
while changed archive manifests or convergence hashes refuse.
