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
