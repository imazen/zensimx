# Corrected-runtime qualification observations

These are engineering controls, outside the 1,461 scientific-fit budget.
The native BUILD SHA-256 is
`56a05880e05dcfa78d28207a88df6f605888814e47707e216e5941d743ac0cff`.
Source for the human worker qualification was `148df35c6d53`.

All three human-only analytic controls passed exact canonical-bake equality
between streaming and explicit human-only-control arms. Each arm used 7,060
admitted TRAIN rows, four human populations, zero teacher rows and one epoch
of 50,000 pair draws. No outer responses were read.

| Family | Streaming wrapper seconds | Controlled wrapper seconds | Canonical SHA-256 prefix |
|---|---:|---:|---|
| Anchored Softplus | 138 | 206 | `223ca465a6553e12` |
| Monotone SiLU | 140 | 205 | `49f04cdfd02667c6` |
| Diagnostic SiLU | 129 | 127 | `68dce17d885b2096` |

Every native arm's `run-heavy` receipt reported peak RSS 0.09 GiB. Minimum
available memory ranged from 39,717 to 39,725 MiB and peak load from 14.46
to 15.33. Full receipt lines and hashes are in `SUMMARY.json`. These concurrent
run observations are not isolated speed benchmarks or evidence about human
quality. Finalist ablations remain unexecuted.

The result archive contains 44 artifacts, 6,008,029 bytes, with full cold
readback. Its receipt SHA-256 is
`20fae6c46d9291b24de3df1d8b2e3fc0e5bf411e9d04b56b6092f0a535597fd3`.
Local root on dev:
`/mnt/v/output/zensim/shaping-completion-20261107/human-only-v3-qualification-20261108/evidence-v1`.
Durable root:
`/mnt/tower/output/zensim/shaping-completion-20261107/human-only-v3-qualification-20261108-evidence-v1`.

The separate input supplement stores 22 worker-missing files in standard
tar+zstd. The 29,736,960-byte tar became 15,852,985 bytes; complete decompressed
bytes matched the original SHA-256. The 41-file full input manifest also
references shared native tools and study inputs rather than duplicating them.
Intermediate canonical files were not copied into the result archive; their
hashes and actual finalized native outputs remain preserved. Nothing was
deleted or uploaded to R2.

## Active-training CPU observations

Read-only ten-second `perf stat` samples observed 9,684.86 ms task-clock for
i134 PID 3204350 and 10,001.58 ms for r3800x PID 3156680. i134 allowed CPUs
0–15 and had no finite quota on the existing ancestor `cpu.max` files. Its
CPU resolved to an Intel Core i5-13400F. These windows do not establish a
cross-host speed ratio; workloads, concurrency and hardware differ.

A separate i134 call-stack sample recorded 995 samples with zero lost, split
between hybrid CPU events. Projection accounted for 42.32% of atom samples
and 38.14% of core samples; fused pair Adam updates accounted for 41.91% and
41.84%, respectively. The observed trainer was the already-running frozen
optimizer binary. No fit was interrupted, moved or replayed. The later
optimized runtime's performance is not measured by this sample.

Both stat wrappers reported profiler peak RSS 0.02 GiB; the record wrapper
reported the same. These are profiler footprints, not trainer peak memory.
All raw counters, call stacks, wrapper lines and observation limits are in
`/mnt/v/output/zensim/shaping-completion-20261107/cpu-efficiency-evidence-20261108`,
mirrored under the same relative study path on `/mnt/tower/output/zensim`.
All six artifacts, 140,827 bytes, passed cold readback; receipt SHA-256
`9129d43b4c4f53788f6cdf942a2f59e1610efc9390d9ef08813022a2b04eb08f`.
