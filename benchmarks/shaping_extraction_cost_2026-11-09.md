# Frozen-read-set extraction cost

All six prescribed extraction workloads completed on i270, an Intel Core Ultra
7 270K Plus. Each uses three frozen, nonidentical RGB8 pairs and the study's
576 raw feature IDs. The process had an enforced cpuset on performance CPUs
0–7, verified by the bundled zenbench CPU tool. The native instrument reports
automatic v3 dispatch. Formula revision 7 and strided/packed exact parity were
checked before timing; exact feature bits were checked again afterward.

Each entry below is the range of **three separately measured per-pair medians**,
in milliseconds. It is not a confidence interval or a pooled timing sample.

| Image dimensions | One thread | Four threads |
|---|---:|---:|
| 256 × 256 | 2.843–3.022 | 4.183–4.517 |
| 1024 × 1024 | 37.545–37.792 | 31.307–31.732 |
| 4096 × 4096 | 595.284–599.198 | 385.446–387.930 |

All six measurements retained the first 32 clean paired rounds within their
64-round budget. Each of the three image pairs has its own 95% bootstrap
interval in the retained report. Thread counts ran in separate processes;
these are not interleaved thread-scaling comparisons. Timings exclude decode
and vector scoring. No end-to-end latency follows from adding these medians to
another benchmark. Finalist vector scoring remains outstanding.

Run-heavy reported these complete-process resource receipts:

| Case | Wrapper seconds | Peak RSS, GiB | Minimum available, MiB | Peak load |
|---|---:|---:|---:|---:|
| 256, one thread | 8 | 0.03 | 26320 | 0.48 |
| 256, four threads | 8 | 0.04 | 26290 | 0.41 |
| 1024, one thread | 9 | 0.07 | 26251 | 0.58 |
| 1024, four threads | 8 | 0.08 | 26263 | 0.77 |
| 4096, one thread | 72 | 0.46 | 25842 | 1.03 |
| 4096, four threads | 48 | 0.52 | 25737 | 1.96 |

The native binary is unchanged from the qualified extraction instrument:
SHA-256 `2bdb45c8f947ecc0f4c87a63b6baf0cba70f7d0a522fc5e299d3ff30fc33fde7`.
The first 256/one-thread benchmark completed before its reporting interpreter
failed to import NumPy; its original timing was recovered without replay.
The first four-thread attempt retained only 25 clean rounds out of 64 because
benchmark ownership was omitted. That incomplete result remains preserved.
Subsequent timings register the real launcher PID through zenbench's existing
ownership interface; foreign-process, memory, temperature and round gates keep
their original settings.

The owner runs the native command through `run-heavy`, then the published
metric repository's unchanged `scripts/profile_test_cpuset.sh` to enforce the
CPU boundary with the same finite memory cap. The launcher verifies the CPU
class before executing the instrument. Registration and source:
`shaping_extract_bench.py` at `aa6b91833609`.

Original timings and collection hashes are on dev under
`/mnt/v/output/zensim/shaping-completion-20261107/extraction-timing-i270-20261109`,
`extraction-timing-i270-ready-20261109` (incomplete),
`extraction-timing-i270-owned-20261109`, and
`extraction-timing-collection-20261109`.

The consolidated archive passed complete durable readback at
`/mnt/tower/output/zensim/shaping-completion-20261107/extraction-cost-results-ready-20261109`.
Its 111 logical payload files occupy 59,363,477 bytes before hard-link
deduplication and compression. The standalone tar.zst is 4,816,479 bytes,
SHA-256 `59b04c62b99b5eaf3cec4e698308f6731f47e026d6862c90d286b206b48aa36f`.
Every logical tar member's hash, length and mode was checked; hard-link targets
were resolved and verified. Identical native tools are stored once and restore
with ordinary `tar -I zstd -xf`, including their executable bits. No compiler
is required for native extraction replay. The archive includes both incomplete
attempts, all six completed reports, raw rounds, source, CPU-boundary launcher
and prebuilt native tools. PACK_VERIFIED.json, MEMBERS.json and the durable
MIRROR_VERIFIED.json retain the exact inventory and readback evidence.

## CPU profile of the frozen native implementation

The owner asked which code accounts for the 4096-square latency. The release
instrument calls `research::extract`, then
`compute_folded720_streaming_extras`, then `foldapp_streaming_walk_impl`.
It requests slots 0–227 and 372–719 and creates a fresh `V2Scratch` per call.
The full request uses the all-channel specialization. The measured region
includes feature computation and result allocation; pixel generation, hashes
and parity checks are outside that region.

A separate 199 Hz DWARF perf capture on the same i270 CPU boundary and identical
native binary recorded 1,159 samples with zero lost samples. It profiles native
qualification, including nine extraction calls and fixture setup. These sampled
cycle percentages describe that diagnostic process, not a new latency run:

| Leaf kernel | Sampled cycles |
|---|---:|
| Fused vertical blur/SSIM, canonical f64, v3 | 22.78% |
| Vertical local-11 box blur, v3 | 16.63% |
| Fused horizontal local-11 blur, v3 | 12.90% |
| Horizontal local-11 box blur, v3 | 2.94% |
| Dense feature block, canonical f64, v3 | 12.56% |
| sRGB-to-XYB conversion, v3 | 5.98% |
| Gradient feature block, canonical f64, v3 | 4.94% |

The sampled hot symbols confirm AVX2/v3 execution. Blur and fused SSIM account
for 55.25% of sampled cycles in total. Memory copying and clearing account for
3.21% and 1.21%, respectively; allocation reuse is not established as the main
bottleneck. The reported process RSS also includes the instrument's three
resident RGB8 pairs and preflight setup, rather than isolating extraction heap.
Parent call chains include unresolved frames, so these are leaf self-cycle
percentages; no complete caller attribution is claimed.

The profile's run-heavy receipt is `done rc=0 6s | peak-RSS 0.46GiB |
min-avail 25909MiB | peak-load 0.40`. This duration is diagnostic process time
and does not replace the unprofiled benchmark results above. Raw perf data,
full reports, launch identity and native qualification outputs are under
`/mnt/v/output/zensim/shaping-completion-20261107/extraction-profile-i270-20261109`.
