# A parallel extraction gate — 2026-11-04

The cutoff is 36,864 pixels (192²): the higher of the measured 16-thread
160² crossover and four-thread 192² crossover. This follows the requested
first-p50-win rule. The four-thread 192² p95 is worse than serial; this
single-host measurement does not establish a universal optimum.

The final admitted scope is **49/50 cells: 25 baseline and 24 gated square cells**.
All **980/980** printed and bitwise serial/parallel score pairs match, including
before/after and pool-size identity. The gated 12 MP cell is **not measured**.
i270 auto-rebooted at 04:00 UTC on November 5 for a kernel update. Its scoring
containers restarted and kept load1 above the unchanged 2.0 admission gate.
The coordinator directed finalization at 49 cells because that evaluation work
outranks this benchmark. The interrupted timing and subsequent zero-sample
preflight attempt are retained and excluded.

## Before measurements

| Size | 16T serial p50/p95 | 16T parallel p50/p95 | Serial/parallel | 4T serial p50/p95 | 4T parallel p50/p95 | Serial/parallel |
|---|---:|---:|---:|---:|---:|---:|
| 64² | 0.8557/2.3167 | 1.8539/3.2500 | 0.462× | 0.8320/1.3340 | 1.9478/3.4166 | 0.427× |
| 96² | 1.2052/3.2460 | 2.1173/3.9598 | 0.569× | 1.1505/1.8595 | 2.3405/4.3923 | 0.492× |
| 128² | 1.6517/3.9895 | 2.4762/4.6111 | 0.667× | 1.5668/2.6164 | 2.5073/4.8919 | 0.625× |
| 160² | 2.2436/3.8718 | 2.0572/3.5337 | 1.091× | 2.0775/3.2753 | 2.1982/3.5091 | 0.945× |
| 192² | 2.7681/4.6096 | 2.2065/3.7183 | 1.254× | 2.5022/4.1735 | 2.4563/4.6534 | 1.019× |
| 256² | 3.7594/6.2500 | 3.2760/5.5246 | 1.148× | 3.8813/6.0495 | 3.3901/5.3817 | 1.145× |
| 320² | 4.8037/8.0167 | 4.3556/6.9398 | 1.103× | 4.7080/7.4568 | 4.8147/8.2816 | 0.978× |
| 384² | 5.8448/10.4137 | 5.9112/7.8958 | 0.989× | 6.4504/10.3288 | 5.7808/7.8488 | 1.116× |
| 512² | 9.1680/17.1411 | 7.0153/9.2289 | 1.307× | 9.5644/17.7277 | 6.8959/10.3159 | 1.387× |
| 640² | 12.8102/24.4834 | 9.5784/11.7520 | 1.337× | 13.6186/24.7416 | 9.6019/13.1801 | 1.418× |
| 768² | 17.7190/34.2597 | 10.7617/13.5585 | 1.646× | 17.0443/33.3215 | 12.9097/17.2089 | 1.320× |
| 1024² | 30.6286/56.8208 | 14.1228/16.8510 | 2.169× | 30.3912/32.8359 | 19.6642/25.0426 | 1.546× |

Crossover: 16T 160²; 4T 192². Gate: **36,864 pixels**, the higher crossover.

Printed and bitwise serial/parallel scores: **0 differences in 980 comparisons across 49 admitted cells**. Before/after and pool-size score bits match. Full 50-cell admission remains unmet because the gated 12 MP cell is not measured.


The 12 MP 16T control has serial p50/p95 **345.6629/377.2978 ms** and
parallel **117.5212/127.2016 ms**, a p50 ratio of **2.941×**.

The grid is non-monotone: 384²/16T and 320²/4T favor serial at p50.
The floor follows the requested first-win rule; it does not establish a
parallel win at every larger size.

## Before and after measurements

All values are p50/p95 milliseconds. “Not measured” excludes the interrupted
12 MP attempt; it is not an estimated or retained partial timing.

| Threads | Size | Serial before p50/p95 | Serial after p50/p95 | Parallel before p50/p95 | Parallel after p50/p95 |
|---:|---|---:|---:|---:|---:|
| 16 | 64² | 0.8557/2.3167 | 0.6994/1.1352 | 1.8539/3.2500 | 0.6952/1.1632 |
| 16 | 96² | 1.2052/3.2460 | 1.0569/1.6686 | 2.1173/3.9598 | 1.0374/1.6888 |
| 16 | 128² | 1.6517/3.9895 | 1.4224/2.4099 | 2.4762/4.6111 | 1.4026/2.3571 |
| 16 | 160² | 2.2436/3.8718 | 2.0243/3.2377 | 2.0572/3.5337 | 1.9848/3.2994 |
| 16 | 192² | 2.7681/4.6096 | 2.8424/4.8720 | 2.2065/3.7183 | 2.5222/4.1158 |
| 16 | 256² | 3.7594/6.2500 | 3.9837/6.4142 | 3.2760/5.5246 | 3.4488/5.5073 |
| 16 | 320² | 4.8037/8.0167 | 5.1060/8.4200 | 4.3556/6.9398 | 4.5173/6.8046 |
| 16 | 384² | 5.8448/10.4137 | 5.8369/10.8461 | 5.9112/7.8958 | 5.0290/7.5533 |
| 16 | 512² | 9.1680/17.1411 | 9.0803/16.7956 | 7.0153/9.2289 | 7.0908/9.5125 |
| 16 | 640² | 12.8102/24.4834 | 12.8759/24.6735 | 9.5784/11.7520 | 8.4830/11.4869 |
| 16 | 768² | 17.7190/34.2597 | 17.7083/34.6424 | 10.7617/13.5585 | 10.3978/13.4176 |
| 16 | 1024² | 30.6286/56.8208 | 30.6894/58.1914 | 14.1228/16.8510 | 15.3225/18.2562 |
| 16 | 12 MP class | 345.6629/377.2978 | not measured | 117.5212/127.2016 | not measured |
| 4 | 64² | 0.8320/1.3340 | 0.7350/1.2775 | 1.9478/3.4166 | 0.7403/1.2116 |
| 4 | 96² | 1.1505/1.8595 | 1.0010/1.5843 | 2.3405/4.3923 | 0.9902/1.5999 |
| 4 | 128² | 1.5668/2.6164 | 1.4483/2.4873 | 2.5073/4.8919 | 1.4384/2.4403 |
| 4 | 160² | 2.0775/3.2753 | 1.8802/2.9094 | 2.1982/3.5091 | 1.8714/2.7569 |
| 4 | 192² | 2.5022/4.1735 | 2.3906/3.9968 | 2.4563/4.6534 | 2.2864/3.9369 |
| 4 | 256² | 3.8813/6.0495 | 3.1701/5.4045 | 3.3901/5.3817 | 3.6688/6.4396 |
| 4 | 320² | 4.7080/7.4568 | 5.4240/8.1702 | 4.8147/8.2816 | 4.2833/6.9735 |
| 4 | 384² | 6.4504/10.3288 | 5.9831/10.1935 | 5.7808/7.8488 | 5.2697/8.9040 |
| 4 | 512² | 9.5644/17.7277 | 9.1918/16.6293 | 6.8959/10.3159 | 6.8557/9.8216 |
| 4 | 640² | 13.6186/24.7416 | 12.0455/23.4537 | 9.6019/13.1801 | 9.6718/13.2484 |
| 4 | 768² | 17.0443/33.3215 | 17.6336/33.9739 | 12.9097/17.2089 | 11.9060/17.6283 |
| 4 | 1024² | 30.3912/32.8359 | 30.3065/57.1494 | 19.6642/25.0426 | 18.8150/24.2259 |

The 1024²/16T parallel p50 is **8.5% higher** in the gated run.
The serial control and parallel timings vary across runs; the measurements do
not establish that every large-image latency is unchanged. The gate retains
the parallel branch above its cutoff, and all admitted scores remain identical.
No additional 1024² repeat was run: recovery was limited to the interrupted cell.
The 12 MP four-thread control was outside the declared study and was not measured.

## Method and identities

Host i270, Intel Core Ultra 7 270K Plus, v3/AVX2, affinity CPUs 0–15;
Rust 1.99.1, LLVM 23.1.1. Release builds use thin LTO and one codegen unit.
Neither build sets target-cpu=native or additional rustflags.

The baseline public library is `85bd08d1587161cccd64bda88c00567a0d12d215`,
including removal of the unused zenresize dependency. The gated public source
starts at `3431470c29b77d6aff1357ac058bd06397576720`; its intervening change is
reference-validation documentation and a just recipe. The paired instrument is
`f3e19a475729ff842f0064166df54f90b25dfeb4` in imazen/zensimx.

The A bake is unchanged, SHA-256
`d5e2a56db35c32620d9f9d9c19ba3bc78bd8fea2b8cd933cde8447428d85e62a`.
The research gate is `bfabf0d50e312e866a719a36c214c4bdba7adbd1`;
the identical public insertion is `398578fc5d513db7ced45062e152c68acdd3edef`.
The frozen baseline executable is SHA-256
`e78d47c13428618da4ee4cf7e5656860bcf349b3564eb40530bf825d3dabc791`;
the gated executable is
`d03774877de2e095768d5ddf98673246588dfe5fdecbf782611ddb9ff7fa3dc6`.
The binary receipts, source snapshots and raw files retain the compiler,
fixtures, rounds and resource logs. The [artifact pointer](a_parallel_gate_2026-11-04.pointer.md)
records the frozen inventory and verified copies.

Each process compares `A.with_parallel(false)` with `A.with_parallel(true)`
using interleaved paired zenbench rounds. Warmup is 100 ms per arm and image;
64 rounds are collected, and the first 32 clean paired rounds per image produce
640 individual calls per
arm per cell. Tables use pooled type-7 p50/p95, in milliseconds. Timing cells
require the same quiet-host and retained-round admission before and after;
missing, contended or incomplete cells cannot qualify the matrix.

Through 512², each size uses ten CID22 TRAIN and ten CLIC TRAIN sources.
Larger square cells use twenty CLIC TRAIN sources because the CID22 inputs
are 512². Sources are Mitchell-downscaled, never enlarged; distortions are
zenjpeg q40 4:2:0 reconstructions. The 12 MP control reuses the original
speed instrument’s twenty PNG-v3 fixture pairs, with geometries 3000×4000,
4000×3000, 4032×3024 and 4096×3072. This is a targeted A crossover
study on these fixtures, with a fixed codec setting and host.

The private gate is derived once at the top of
`foldapp_streaming_walk_impl`, from the walk dimensions after existing small-image
padding and before every parallel choice. Both crates use the same insertion.
Legacy diffmap paths are unchanged.

## Validation

Independent ImageMagick readback decodes all 260 retained q40 JPEGs to full
8-bit RGB output, with every retained JPEG hash unchanged. The instrument
continues to time the original imazen-decoded RGB fixtures. `FIXTURE_ORACLE.json`
records the decoder version and all input hashes.

Research full-crate and native HDR tests pass. The strided batch-route test
covers 191×193 (one pixel below the cutoff), 192×192 and 192×193, retaining
bitwise feature and memory-budget assertions. Below the cutoff it also asserts
that no parallel queue is retained.

The actual revision-7 A bake matches serial score, distance, scorer feature
and requested research feature bits for all 255 split/shared-geometry pairs
at 1, 2, 3, 4, 8 and 16 threads. The existing broad research identity test also
passes at its supported revisions 1 and 4, with its assertions unchanged.

The research reference uses `--public-a` with the actual public A bake,
revision 7 and `ZENSIM_POW_FORM=libm`, matching the public library’s frozen
parameters. Revision 1 supplies the historical generic/config/preview rows.
Normalization selects the shipped public surfaces, maps research E to public A,
and requires identical complete key sets for that contract. It preserves every
value difference. All 135,625 normalized contract rows
match both the ungated public baseline and the gated public checker. Both sides have SHA-256
`39a12c5f89d8f020540e5fc786ce6919b28a07beddd234ba018a522e7901b8c1`.
The gated and ungated revision-7 research reference TSVs also compare byte-for-byte.

Validation setup exposed two pre-existing fixture/build issues: native HDR
tests needed their API’s feature guard (`3caa771a`), and auto-discovered split
examples needed explicit required features (`76bd8aee`). The baseline registry
helper was synchronized with the existing main fix `e517c776`. Workspace clippy’s
unused training import was synchronized with main `bfa87dd7`. Original failure
logs and source copies are retained. No score expectation or quality threshold
was relaxed, and no test was newly ignored.

An archival correction was required: the initial cache helper scanned only
`target` while deleting both target directories. It missed the public checker
in `target-tests`; the original timing executables, checker TSV and complete
source snapshot survived. The rebuilt checker reproduces the original raw TSV
byte-for-byte (SHA-256
`a068018b369a3fb553eb92a81a46e7941fa92efcb717312fd52e4d892949926f`).
The rebuilt executable is frozen outside Cargo output, SHA-256
`2609d89b6b07305e15f9997d561cb285ef00867de62e3227f8175feed3078c9b`.
The original checker executable's hash was not recorded, so binary identity
with that discarded executable is not claimed. The corrected helper preserves
release roots/examples in each existing named cache before removing it; both
new regression controls and the actual rebuild/preservation run pass.

Before the final inventory, two further controls found that the manifest tool
excluded preserved directories named `target` and silently omitted directory
symlinks. It now includes preserved nested controls, excludes only root Cargo
caches and VCS metadata, and refuses directory links. All five fixture aliases
were materialized as verified hard links, preserving the original bytes. All
23 admission, preservation and inventory controls pass. No raw measurement
was rerun for this archival correction.

## Hardware-counter observation

For the same 20-pair 64²/16T score-check command, `perf stat` recorded
1,644 context switches before the gate and 49 after, with 108 and 13 CPU
migrations respectively. These are whole-process observations, including
startup and both scoring arms. Core/atom PMU counters were multiplexed, so
no cycle comparison or kernel-time percentage is inferred. Both raw captures
are retained. Resource receipts:

```text
before: run-heavy: done rc=0 0s | peak-RSS 0.02GiB | min-avail 25555MiB | peak-load 1.58
after: run-heavy: done rc=0 0s | peak-RSS 0.02GiB | min-avail 25727MiB | peak-load 1.55
```

## Commands

All heavy payloads run on i270 under the shared `flock` and `run-heavy`
16 GiB cap. Builds and checks use eight jobs; timing cells use sixteen,
with the scoring pool explicitly set to sixteen or four. The collector
acquires the lock separately for each cell and records its live owner.
It waits for load1 below 2.0 and no compiler/trainer before latency work.
The instrument then admits clean paired rounds independently.

The build payload is identical for the two frozen libraries, with the
standalone manifest dependency selecting `public-before/zensim` or
`public-after/zensim`. The archive retains both manifest versions and Cargo
artifact JSON. From the artifact root:

```sh
root="$HOME/tmp/zensim-pargate"
export TMPDIR="$root/scratch"
export CARGO_TARGET_DIR="$root/target"
export MARGARINE_SOURCE="$HOME/tmp/zensim-bench/margarine"
flock "$HOME/tmp/heavy.lock" \
  "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 8 -- \
  just --justfile "$root/research-after/justfile" a-parallel-gate-build "$root"
```

The declared matrix uses these collector payloads for each frozen phase;
`run.sh` supplies the lock, resource wrapper, affinity, pool environment,
quiet-state checks, per-process RSS capture and immutable output names:

```sh
export A_SPEED_PARALLEL_GATE=1 BENCH_CPUS=0-15 BENCH_MODES='latency scores'
BENCH_THREADS='16 4' BENCH_SIZES='64 96 128 160 192 256 320 384 512 640 768 1024' \
  bash source/a-speed-runtime/run.sh "$root/after" "$root/after/bin/a-speed-peers"
BENCH_THREADS=16 BENCH_SIZES=4096 \
  bash source/a-speed-runtime/run.sh "$root/after" "$root/after/bin/a-speed-peers"
```

For the baseline, initial 16T cells through 320² were followed by the full
4T grid and the remaining 16T cells. A queued 384² attempt yielded before
any timing pair started; its empty attempt and receipt are preserved.
The after controller validates both complete, contiguous crossover prefixes
before starting gated cells. Larger sizes cannot change the first win in those
prefixes. The two collectors serialize all heavy work through the same lock;
the original full-matrix admission still requires every before and after cell.
That gate was not relaxed: this report explicitly finalizes 49 admitted cells
and records the missing cell. Only the shared first 32 clean rounds enter the tables.

Research check payloads, from `research-after/`, run under the same locked
16 GiB/eight-job wrapper and `taskset -c 0-15`:

```sh
just a-parallel-gate-test
just a-parallel-gate-hdr-test
just a-parallel-gate-thread-test
just a-parallel-gate-bake-thread-test "$root/public-before/zensim/weights/rev7_mix_h128_2026-11-03.bin"
just clippy
just a-parallel-gate-reference "$root/public-before/zensim/weights/rev7_mix_h128_2026-11-03.bin" 7
just a-parallel-gate-reference "$root/public-before/zensim/weights/rev7_mix_h128_2026-11-03.bin" 1
```

The public check driver records each complete command and its log hash in
`PUBLIC_CHECKS.json`. Its all-target test commands use the existing
`profile_test_cpuset.sh 0-7` helper to enforce one CPU class and preserve the
finite memory cap; it runs default and `legacy-models` configurations, all-feature
profile goldens, default/legacy/all-feature strict clippy, and the release split
checker. It preserves the original ignored PNG fixtures by hash-verified links.

```sh
cd "$root/public-after"
python3 -B ../source/a-speed-runtime/check_gate.py .. public
```

Raw timing, score, reference, validation and hardware-counter logs are retained
with their source/binary identities. The artifact pointer records archive verification.

## Re-run only the missing 12 MP cell

Run this single command only when i270 is idle. It uses the frozen binary and
fixtures, checks the recorded binary hash, takes the existing shared lock and
keeps the original load and clean-round gates. It writes to a fresh sibling
directory and does not overwrite the finalized 49-cell capture. This recipe
was syntax-checked; it has not been executed because evaluation owns the host.

```sh
ssh i270 'set -eu
  r="$HOME/tmp/zensim-pargate"
  out="$HOME/tmp/zensim-pargate-idle-12mp-$(date -u +%Y%m%dT%H%M%SZ)"
  sha256sum -c "$r/after/benchmark.sha256"
  mkdir "$out"
  ln -s "$r/fixtures" "$out/fixtures"
  cp "$r/after/BINARY.json" "$out/BINARY.json"
  A_SPEED_PARALLEL_GATE=1 BENCH_CPUS=0-15 BENCH_THREADS=16 \
    BENCH_SIZES=4096 BENCH_MODES="latency scores" \
    bash "$r/source/a-speed-runtime/run.sh" "$out" "$r/after/bin/a-speed-peers"
'
```

The instrument’s legacy size tag `4096` selects the existing 12 MP fixtures;
it does not create a 4096² corpus. A single-cell retry is intentionally not a
complete matrix. Its latency JSON, score log and resource log remain available
for independent admission without rerunning the measurement.

## Resource observations

Runner resource observations (whole-process peaks, not a separate memory comparison):

```text
before 16T 64: run-heavy: done rc=0 289s | peak-RSS 0.04GiB | min-avail 25295MiB | peak-load 2.84
before 16T 96: run-heavy: done rc=0 289s | peak-RSS 0.04GiB | min-avail 25129MiB | peak-load 2.40
before 16T 128: run-heavy: done rc=0 290s | peak-RSS 0.04GiB | min-avail 25361MiB | peak-load 2.35
before 16T 160: run-heavy: done rc=0 290s | peak-RSS 0.05GiB | min-avail 25310MiB | peak-load 1.83
before 16T 192: run-heavy: done rc=0 290s | peak-RSS 0.06GiB | min-avail 24852MiB | peak-load 2.32
before 16T 256: run-heavy: done rc=0 290s | peak-RSS 0.07GiB | min-avail 25207MiB | peak-load 2.31
before 16T 320: run-heavy: done rc=0 289s | peak-RSS 0.08GiB | min-avail 25237MiB | peak-load 3.26
before 16T 384: run-heavy: done rc=0 289s | peak-RSS 0.10GiB | min-avail 25240MiB | peak-load 1.95
before 16T 512: run-heavy: done rc=0 290s | peak-RSS 0.13GiB | min-avail 25191MiB | peak-load 2.57
before 16T 640: run-heavy: done rc=0 287s | peak-RSS 0.18GiB | min-avail 25136MiB | peak-load 1.68
before 16T 768: run-heavy: done rc=0 286s | peak-RSS 0.22GiB | min-avail 25105MiB | peak-load 3.23
before 16T 1024: run-heavy: done rc=0 285s | peak-RSS 0.32GiB | min-avail 24967MiB | peak-load 1.80
before 16T 4096: run-heavy: done rc=0 794s | peak-RSS 1.95GiB | min-avail 23187MiB | peak-load 5.71
before 4T 64: run-heavy: done rc=0 291s | peak-RSS 0.04GiB | min-avail 25194MiB | peak-load 1.79
before 4T 96: run-heavy: done rc=0 289s | peak-RSS 0.04GiB | min-avail 25256MiB | peak-load 1.79
before 4T 128: run-heavy: done rc=0 290s | peak-RSS 0.04GiB | min-avail 24963MiB | peak-load 1.75
before 4T 160: run-heavy: done rc=0 290s | peak-RSS 0.05GiB | min-avail 25270MiB | peak-load 1.74
before 4T 192: run-heavy: done rc=0 289s | peak-RSS 0.05GiB | min-avail 25266MiB | peak-load 2.16
before 4T 256: run-heavy: done rc=0 290s | peak-RSS 0.06GiB | min-avail 25258MiB | peak-load 3.32
before 4T 320: run-heavy: done rc=0 288s | peak-RSS 0.06GiB | min-avail 25254MiB | peak-load 1.80
before 4T 384: run-heavy: done rc=0 289s | peak-RSS 0.07GiB | min-avail 25271MiB | peak-load 1.86
before 4T 512: run-heavy: done rc=0 289s | peak-RSS 0.10GiB | min-avail 25214MiB | peak-load 2.37
before 4T 640: run-heavy: done rc=0 287s | peak-RSS 0.13GiB | min-avail 25190MiB | peak-load 1.93
before 4T 768: run-heavy: done rc=0 287s | peak-RSS 0.16GiB | min-avail 25141MiB | peak-load 1.62
before 4T 1024: run-heavy: done rc=0 284s | peak-RSS 0.23GiB | min-avail 25110MiB | peak-load 1.94
after 16T 64: run-heavy: done rc=0 289s | peak-RSS 0.03GiB | min-avail 25264MiB | peak-load 1.84
after 16T 96: run-heavy: done rc=0 288s | peak-RSS 0.03GiB | min-avail 25303MiB | peak-load 1.92
after 16T 128: run-heavy: done rc=0 289s | peak-RSS 0.04GiB | min-avail 25258MiB | peak-load 2.04
after 16T 160: run-heavy: done rc=0 290s | peak-RSS 0.04GiB | min-avail 25228MiB | peak-load 1.73
after 16T 192: run-heavy: done rc=0 291s | peak-RSS 0.06GiB | min-avail 25190MiB | peak-load 1.72
after 16T 256: run-heavy: done rc=0 289s | peak-RSS 0.07GiB | min-avail 25225MiB | peak-load 2.08
after 16T 320: run-heavy: done rc=0 289s | peak-RSS 0.08GiB | min-avail 25230MiB | peak-load 2.48
after 16T 384: run-heavy: done rc=0 289s | peak-RSS 0.10GiB | min-avail 25152MiB | peak-load 1.84
after 16T 512: run-heavy: done rc=0 289s | peak-RSS 0.13GiB | min-avail 25133MiB | peak-load 2.09
after 16T 640: run-heavy: done rc=0 288s | peak-RSS 0.18GiB | min-avail 25099MiB | peak-load 3.16
after 16T 768: run-heavy: done rc=0 287s | peak-RSS 0.23GiB | min-avail 25037MiB | peak-load 3.24
after 16T 1024: run-heavy: done rc=0 285s | peak-RSS 0.32GiB | min-avail 24958MiB | peak-load 3.38
after 4T 64: run-heavy: done rc=0 289s | peak-RSS 0.03GiB | min-avail 25239MiB | peak-load 1.84
after 4T 96: run-heavy: done rc=0 288s | peak-RSS 0.04GiB | min-avail 25134MiB | peak-load 1.89
after 4T 128: run-heavy: done rc=0 290s | peak-RSS 0.04GiB | min-avail 24867MiB | peak-load 1.96
after 4T 160: run-heavy: done rc=0 288s | peak-RSS 0.04GiB | min-avail 25199MiB | peak-load 1.47
after 4T 192: run-heavy: done rc=0 290s | peak-RSS 0.05GiB | min-avail 25211MiB | peak-load 1.81
after 4T 256: run-heavy: done rc=0 288s | peak-RSS 0.06GiB | min-avail 25222MiB | peak-load 2.02
after 4T 320: run-heavy: done rc=0 290s | peak-RSS 0.06GiB | min-avail 25186MiB | peak-load 2.34
after 4T 384: run-heavy: done rc=0 289s | peak-RSS 0.07GiB | min-avail 25213MiB | peak-load 1.68
after 4T 512: run-heavy: done rc=0 291s | peak-RSS 0.10GiB | min-avail 25168MiB | peak-load 1.65
after 4T 640: run-heavy: done rc=0 287s | peak-RSS 0.13GiB | min-avail 25127MiB | peak-load 1.72
after 4T 768: run-heavy: done rc=0 286s | peak-RSS 0.16GiB | min-avail 25091MiB | peak-load 2.18
after 4T 1024: run-heavy: done rc=0 285s | peak-RSS 0.24GiB | min-avail 25041MiB | peak-load 2.35
```
