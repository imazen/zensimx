# Public A extraction fusion — 2026-11-04

Status: investigation in progress. No fusion from this study is qualified or landed.
The experiments use the frozen F seed-0 bake served through `BakeScorer`, SHA-256
`185f43f7e5e1327c34ad18f0b73b96b872663726a0fac6ff524430d0d8430088`.
This is the brief's public-A control, distinct from the deprecated research
`ZensimProfile::A` model.

## Inputs and execution

Phase-1 profile parent: `ccdb5c7a`. All builds, tests and measurements run on i270,
Core Ultra 7 270K Plus, native v3. One-thread measurements use P core 2;
16-thread measurements use CPUs 0–15. AVX-512 speed is **not measured on Zen 5**.
Every native job acquires the host's shared heavy lock and runs through
`run-heavy --mem 16G`; other lanes use the same lock.

Reuse the extraction-cost study's twenty zenjpeg-q40 pairs per size, with
unchanged file/pixel hashes: 256², 1024² and the large photographic class.
Large images retain that study's actual geometry, including its 4096-long-edge
cap; this is not a claim that every image has exactly twelve million pixels.
Profiles use the cost study's preserved executable, SHA-256
`cdf7e8a2bfefe6a7030253581cbd171928c14f156e9b4d0fd2e101be9b488072`.
Its five extraction source files match the parent byte-for-byte.

The native fixture packet contains 135 steering cases, 62 identity sources,
24 NEARID references and all 4,678 EVAL feature-pair keys from the R12 proof.
No fit, target selection or retraining is authorized by this study.

## Phase 1: observed cost

Ten complete walks over twenty pairs per size, `perf record -g`, 997 Hz
user cycles, frame-pointer call chains; self samples are weighted by period.
The denominator selects samples under
`zensim::fold_engine::compute_fold_backed_stoppable`, exactly as the cost
study's existing attribution owner does. Separate controlled `perf stat`
windows exclude input loading and preflight. Only the pinned P-core PMU is
used; the inactive atom PMU's uncounted rows are excluded.

| Extraction self bucket | 256² | 1024² | Large |
|---|---:|---:|---:|
| Blur / pyramid | 37.796% | 43.915% | 46.024% |
| Dense v2 | 24.123% | 20.799% | 19.070% |
| Basic fused | 11.496% | 9.467% | 8.855% |
| Color | 9.991% | 8.310% | 8.053% |
| Gradients | 8.422% | 8.284% | 7.018% |
| Memory routines | 3.549% | 3.843% | 3.904% |
| Blockiness | 2.447% | 1.776% | 2.367% |
| Other extraction | 2.175% | 3.607% | 4.709% |

| Counter per image | 256² | 1024² | Large |
|---|---:|---:|---:|
| Cycles | 10,155,962.635 | 151,511,784.265 | 1,783,832,779.540 |
| Instructions | 37,059,592.910 | 496,023,138.920 | 5,473,410,965.735 |
| L1d load misses | 171,924.835 | 2,589,976.580 | 28,390,243.255 |
| LLC load misses | 6.055 | 13,147.975 | 3,278,281.390 |
| Instructions / cycle | 3.649 | 3.274 | 3.068 |
| LLC load miss fraction | 0.054% | 1.897% | 40.221% |

| Topdown metric | 256² | 1024² | Large |
|---|---:|---:|---:|
| Memory bound (slots) | 18.4% | 25.5% | 29.4% |
| Core bound (slots) | 28.2% | 25.9% | 27.1% |
| Retiring (slots) | 48.2% | 43.9% | 40.8% |
| DRAM bound (cycles) | 0.2% | 2.0% | 7.0% |

The large case has increased cache pressure and mixed memory/core limits.
These observations do not support calling the entire walk purely DRAM-bound.
Topdown's slot and cycle percentages have different denominators.
Topdown capture receipt: `run-heavy: done rc=0 97s | peak-RSS 1.61GiB |
min-avail 23694MiB | peak-load 1.77`. This is whole-driver RSS, including
all twenty input pairs, not isolated scorer memory.

## Pass table: one channel at one scale

This table inventories materializations, not measured memory-controller bytes.
Let W be plane width, S the core strip height and N = W(S + 20). A full
128-row core has a 148-row wide window. Every listed float is f32 (4 bytes).
An intermediate's minimum write-plus-later-read round trip is 8N bytes,
or **9.25 MB per core MP** for full strips. Stencil tap counts, cache-line
write allocation, reflection gathers, partial strips and compaction add traffic;
registers and cache hits must not be counted as DRAM transfers.

| Plane | Extent | Writer / phase | Later readers | Materialization |
|---|---|---|---|---|
| Source / distorted XYB | Two rolling W-wide row windows | Conversion or preceding-scale downsample / producer | Moment H, activity, dense/basic, gradient stencil, sparse blockiness, next-scale downsample | One write per produced sample; reused windows and possible compaction |
| Mean source H / mean distorted H | Two N-element planes | Fused moment H / A | Moment V | One write; V tree reads taps and caches pair/quadruple nodes |
| Summed second moment H | N elements; r²+d², **not two separate planes** | Same fused H / A | Moment V | Same |
| Error moment H | N elements; (r−d)² at revision 7, **not rd** | Same fused H / A | Moment V | Same |
| Four vertical moments | Four N-element planes | Moment V / A | Dense / B, basic bands / B; source mean also feeds activity / A | Each written once; core reread by two reduction kernels |
| Absolute source deviation | N elements | abs(src−mean_source) / A | Activity H / A | One write and later stencil reads |
| Activity H | N elements | Activity H / A | Activity V / A | One write and later V reads |
| Activity V | N elements | Activity V / A | Dense and gradient / B | One write, two core consumers |
| Gradients | No image plane | Gradient kernel / B | Lane accumulators | Register values and bounded accumulators only |
| Next pyramid level | Two planes, floor(W/2) × floor(H/2) | Downsample / producer | Next-scale walk | One write per child sample; no whole-image pyramid |
| bs2 / distorted activity | Unread by this bake | Not produced | None | Serial scratch can still reserve bs2; queued jobs do not |
| Scratch / queue copies | Route-dependent wide windows | Gather / queue preparation | A and B | Interior serial windows borrowed; queued uncached windows copied |

At one thread the four horizontal moments expose **37 MB per core MP** of
minimum full-plane round-trip traffic; the two activity intermediates expose
**18.5 MB**. Dense and basic separately consume six common f32 values per
core pixel: **24 MB/MP** of duplicate reads. These are fusion opportunities,
not measured savings. A ring can replace a full-plane lifetime without reducing
the number of logical stores, and can increase tap reads while reducing the
cache footprint.

The 16-thread queued path already uses 256-column fused H/V panels, and fuses
activity too. It reuses horizontal scratch for activity. The serial path uses
separate moment and activity passes. `stream_fold_v1` already consumes vertical
moments at revision 7: there is no second basic V-blur to remove. Color already
uses an exact sRGB8 lookup table. The producer already streams and cascades
its pyramid, including early downsampling of unread scale-zero chroma.

## Candidate inventory (reported in chat before editing code)

| Rank | Candidate | Exactness | Exposure / mechanism |
|---|---|---|---|
| 1 | Horizontal moment row ring feeding vertical moments | Exact | Replace four full-strip intermediate lifetimes; preserve every H and V tree and intermediate f32 rounding. Compare with the existing queued panels too. |
| 2 | Bounded absolute-deviation / activity blur | Exact | Replace two full-strip intermediate lifetimes; existing 16-thread panel fusion is the control. |
| 3 | Dense + basic reduction fusion | Conditional exact | Up to 24 MB/MP duplicate reads; must preserve each kernel's lane, row, band and strip reductions independently. |
| 4 | Earlier next-level production and dead scratch removal | Exact | Downsample parent reads total 8 MB per parent MP for both images per channel; much is already fused for unread chroma. Allocation effects need native measurement. |
| 5 | Panel geometry | Exact | No inherent logical-byte saving; compare cache behavior without moving canonical strip/band boundaries. |
| 6 | Sliding blur recurrences | Era break | Reassociates horizontal and/or vertical sums. |
| 7 | Remove intermediate horizontal rounding | Era break | Changes precision and the separable operation tree. |
| 8 | Lower-precision moment storage | Era break | Changes intermediate precision. |

A candidate qualifies only if p50 improves at both 1024² and large size at both
one and sixteen threads, with all required bit gates unchanged. The remaining
exact candidates need measurement before the 2% stopping rule can be invoked.
No such stopping claim has been made.

## Correctness and comparison receipts

The paired harness extends `a_family_cost` and follows SPEEDQ:
persistent before/after workers, inner `BakeScorer::compute` timers, paired
zenbench admission, first 32 clean rounds of at most 64, 100 ms initial warmup
per arm (each call traverses all twenty images). Input decode/loading, IPC and result destruction are outside
the measured call. Rejected attempts and their raw samples are preserved.
Each worker records `/usr/bin/time -v` and its own `run-heavy` receipt.

Two brief assumptions require explicit handling: this repository commits the
`split_parity_golden` **generator**, not a split-output TSV; testable dispatch
enumerates executable host tokens and does not by itself emulate AVX-512.
A new before-source output can establish before/after parity but cannot be
represented as the nonexistent committed TSV. Intel SDE 10.13.1, build
2026-07-28, passed its i270 launch probe. Its Sapphire Rapids model enables
additional tier correctness checks; emulator timings cannot qualify native speed.
The downloaded archive is pinned by SHA-256
`94e97d623fec54385686e1e7ba65ebc9941748c05ee451423948334892bf2b50`.
[Intel's SDE download and release documentation](https://www.intel.com/content/www/us/en/download/684897/intel-software-development-emulator.html)
describes the emulator.

The original baseline's two registry tests failed because the existing
`fx2_consumed_504` entry had no test-helper decoder. The test-only correction
derives the selection from the existing declaration's 492 direct IDs and its
product factors, yielding the registered 504 slots. Existing coverage and
registered hash/equality checks remain unchanged. The complete release crate
test command then passed: library 638 passed, 16 pre-existing ignored tests;
integration suites and doctests also passed. Full driver receipt (including
baseline golden generation and eight native identity permutations):
`run-heavy: done rc=0 636s | peak-RSS 2.19GiB | min-avail 20316MiB |
peak-load 9.41`. This is a prerequisite test fix, not a serving optimization.

The paired harness's same-binary controls admitted 32 clean rounds from
45 total at one thread and 32 from 32 at sixteen threads, with no unreliable
status. These are harness controls, not optimization speed claims.

### Baseline finding: explicitly ignored legacy pool identity

The unchanged source reproduces the explicitly ignored
`ssim_form::tests::stable_moments_are_exact_across_simd_tiers` failure at both
revision 1 and revision 7: **41 differing pooled values natively, 40 under
SDE**. The stable per-pixel output component agrees. The differing component
is `stable_pool_cases`, the legacy weighted reducers; this is separate from
the served Rev4/5/7 feature-vector gates, which pass all sixteen SDE
permutations for the ring candidate.

The owner explicitly directed that the legacy test, its assertions, its
ignore marker and legacy arithmetic remain unchanged. The passing baseline
gates define fusion qualification. Additionally, **every fusion must rerun
the ignored test on both arms and compare its complete differing-value lists**:
same counts, indices, component names and both bit values. A new difference
blocks qualification. `scripts/bench/a_fusion_stable.py` owns that comparison;
it saves all assertion output and every differing value, without truncating
the lists or representing a failed test as a pass.

Exact baseline commands, on i270, under the shared heavy lock:

```sh
STUDY="$HOME/tmp/zensim-fusion"
for revision in 1 7; do
  env ZENSIM_FORMULA_REV="$revision" "$STUDY/baseline-stable/zensim-test" \
    --exact ssim_form::tests::stable_moments_are_exact_across_simd_tiers \
    --ignored --nocapture --test-threads=1
  env ZENSIM_FORMULA_REV="$revision" \
    "$STUDY/tools/sde-external-10.13.1-2026-07-28-lin/sde64" -spr -- \
    "$STUDY/baseline-stable/zensim-test" \
    --exact ssim_form::tests::stable_moments_are_exact_across_simd_tiers \
    --ignored --nocapture --test-threads=1
done
```

Original logs: `$STUDY/baseline-stable/rev{1,7}-{native,sde}.log`;
source hashes: `$STUDY/baseline-stable/SOURCE_SHA256SUMS`.
The probe records each test's exit 101 separately; its enclosing diagnostic
driver exits zero only after completing all four captures and parsing the
failures. Its whole-driver receipt is `rc=0 129s | peak-RSS 2.19GiB |
min-avail 22876MiB | peak-load 4.74`.
Per-fusion complete lists, commands, binary/log hashes and equality results
belong in each candidate's `stable-diff/receipt.json`.

The differing pools and element counts below are identical at Rev1 and Rev7.
The ring's before/after lists match exactly, including both stored values:
`ring/stable-diff/receipt.json` (`equal: true`). Its driver receipt is
`rc=0 7s | peak-RSS 0.17GiB | min-avail 25127MiB | peak-load 1.68`.

| Differing component | Native element counts | SDE element counts |
|---|---|---|
| iw_sum | 16,17,31,33,64,129,512 | 64,129,512 |
| iw_fourth | 16,17,31,33,64,129,512 | 64,129,512 |
| iw_square | 16,17,31,64,129,512 | 16,17,31,33,64,129,512 |
| iw_only_sum | 16,17,31,33,64,129,512 | 64,129,512 |
| masked_sum | 33,64,129,512 | 33,64,129,512 |
| mask_only_sum | 33,64,129,512 | 33,64,129,512 |
| masked_fourth | 129,512 | 16,17,31,33,129,512 |
| masked_square | 129,512 | 64,129,512 |
| mask_only_fourth_xor_square_bits | 129,512 | 16,17,31,33,64,129,512 |

### EVAL decoder-era provenance

An unchanged-baseline replay of all 4,678 EVAL feature keys differs from the
frozen feature predictions on 645 pairs, after joining by the frozen table's
`ref_basename` rather than assuming row order. This is upstream of extraction:
the first affected ColourLab file has matching encoded bytes and different
decoded RGB bytes. The frozen extractor binary is
`ed4957417339779dd8ed25fe59a6166326ba9db86d5c1530b1e1c788c9d614d8`,
built from metric commit `b77883be18db5933a9db4b9e2670a2ba88a9729b`.
Its preserved Cargo.lock names zenjpeg 0.9.0 as a local path dependency and
records no codec commit. The replay uses published zenjpeg 0.8.4. Both that
version's direct decoder and its zencodec wrapper produce the same replay RGB.

First affected distorted file:
`colourlab-rating-variability-2022/content/images/060_01_01.jpg`, 800×800.
Encoded SHA-256:
`98b0ca11d5ad5a84f11c274f05f2449e541d3068ec402b11c33c4d9410ebd2f0`.
Frozen distorted RGB:
`c5cfa8511171491822c9b7c3fc457439ac1dda89ef9b507a01c908c2b38064c6`;
replay distorted RGB:
`5f3bf9a36bfbd623295c059048e4416f8191387db1d2840088cc7c340ab1fec8`.
The corresponding reference `060.jpg` also differs: frozen RGB
`6816d8b4fc46c604b8f2d2d516a53988ce6ffc2c692fa176779f22ff0c989ffe`,
replay RGB `a94b169b2e43ebfdb1cf757ac02c2fff1ff4d1bafdb7b5f56176588e803fbeba`.
Receipts: `before3/EVAL_FEATURE_REPLAY-keyed.json`, `jpeg-probe/RESULT.json`,
and the original bank's pinned `chunk_000.audit.jsonl`.

The coordinator explicitly decided that frozen-prediction replay is not a
fusion gate. The fusion gate compares old and new extraction on **identical
pixels decoded once**, using the published decoder above: all 504 consumed
feature-slot bits and served f64 score bits on all 4,678 pairs. Original
feature tables, source images and predictions remain unchanged. No labels
are read and no decoder is selected using score error. The era experiments
use those same pixels to isolate extraction's score changes.

The one-decode ingress cache is complete: 4,678 pairs, 4,792 distinct encoded
source paths. It is stored at
`/mnt/tower/output/zensim/fusion-eval-pixels-2026-11-04/`, checked from i270.
Its manifest records both encoded and decoded hashes and geometry; raw RGB
files are addressed by decoded SHA-256. Manifest SHA-256:
`c95c89eb0ecd6f020882ba42f7e0f8aafaf5914b645e848e06b264313bc8ca32`;
pairs-table SHA-256:
`584e9a2099170579e40ba449176689e9296e88715ea283ad8ccbdef71ef47572`.
Preparation receipt: `rc=0 211s | peak-RSS 0.22GiB | min-avail 25175MiB |
peak-load 2.53`. Baseline capture of all 4,678 rows completed with
`rc=0 238s | peak-RSS 0.41GiB | min-avail 25116MiB | peak-load 3.17`.
Those are gate-driver receipts, not extraction benchmark timings.

The focused identity scorer reproduces all 744 original E33/Frozen-A scores
exactly (`tools-check-v2/EQUIVALENCE.json`). The full original E33 assertions
remain unchanged. Future complete captures use four disjoint permutation
shards within one heavy scope; the merge requires exactly 744 distinct cells,
12 permutations and 62 sources. Features are compared between old and new at
the same permutation, without assuming additional cross-tier f64 contracts.

Workspace CI-exact clippy and all 16 script-linter regression tests passed on
the native source export: `rc=0 60s | peak-RSS 1.05GiB | min-avail 22938MiB |
peak-load 5.86`. The source-export tracked-path guard passed 5,216 paths,
1,539 scripts and five malformed-manifest rejection controls:
`rc=0 5s | peak-RSS 0.03GiB | min-avail 25580MiB | peak-load 1.39`.
Its standalone prerequisite landed as `4a4033083ac4`; remote ancestry was
verified. The serving source is unchanged between the registry prerequisite
`e517c776` and that main tip; intervening training-only changes do not define
an extraction speed claim.

The ring's complete EVAL comparison passes: all 504 consumed slots and score
bits equal on all 4,678 pairs (`ring-final/EVAL_EARLY_BITS.json`). Comparator
receipt: `rc=0 1s | peak-RSS 0.52GiB | min-avail 25546MiB | peak-load 3.07`.
The after-arm gate driver completed with `rc=0 276s | peak-RSS 0.39GiB |
min-avail 24975MiB | peak-load 1.82`; those wall times include all input I/O,
feature serialization and hashing and are not performance comparisons.

Frozen A's steering ranking reports 133/135 qualitative passes. The ring has
the same two ranking outcomes: case 19 `broad-13-b64` has M2
0.9588235294117619, and case 72 `broad-220-b8` has M3f
0.6942114754688108. The fusion gate compares all 135 bases and every repair
bit-for-bit, including those cases; it does not reinterpret that old ranking
report as a 135/135 pass. Ranking thresholds and assertions remain unchanged.

### Moment-ring qualification

The final mandatory pixel capture passed: 744 identity cells include all 504
consumed feature slots and scores at each matching permutation; G-STEER
includes all 135 bases and 55,380 repairs, with full feature hashes and score
bits; NEARID includes 648 complete feature rows and 768 ladder scores.
Receipts: `ring-final/IDENTITY_BITS.json`, `ring-final/PIXEL_FEATURES.json`,
`ring/PIXEL_BITS.json`, and `ring-final/EVAL_EARLY_BITS.json`.
The final pixel/stability driver completed with `rc=0 699s | peak-RSS 2.21GiB |
min-avail 23432MiB | peak-load 7.93`.

The timing batch rebuilt the baseline instrument under one uninterrupted
parent heavy lock and required its executable bytes to match the previously
captured control. It then reran all eight ignored-test captures. The complete
41-value native and 40-value SDE lists match at both Rev1 and Rev7, including
both values of every differing component:
`ring-final/stable-diff-verified/receipt.json` (`equal: true`).
The test, assertions, ignore marker and legacy weighted arithmetic are unchanged.

The ring is **rejected**: it loses three of the four required 1024²/large
comparisons. All six cells admitted the first 32 clean paired rounds. Values
below are medians of twenty per-image percentiles, in milliseconds.

| Size | Threads | Before p50 | After p50 | Before p95 | After p95 | p50 change |
|---|---:|---:|---:|---:|---:|---:|
| 256² | 1 | 1.85892825 | 2.1282415 | 1.873429625 | 2.142187975 | +14.488% |
| 1024² | 1 | 27.57252625 | 32.22658 | 27.860940875 | 32.472816025 | +16.879% |
| Large | 1 | 330.7046135 | 367.94362375 | 345.605104825 | 377.4711496 | +11.261% |
| 256² | 16 | 0.824015 | 0.82001725 | 1.214681825 | 1.152679775 | −0.485% |
| 1024² | 16 | 7.37269875 | 7.13245075 | 8.17033595 | 8.224881075 | −3.259% |
| Large | 16 | 98.2321585 | 105.64631775 | 102.774094575 | 109.4544248 | +7.548% |

Each row below quotes the two workers' `run-heavy` lines as
`seconds / peak-RSS GiB / min-avail MiB / peak-load`; every exit is zero.
Worker RSS includes all twenty loaded input pairs and scratch, rather than
isolating scorer-owned allocations. `/usr/bin/time -v` receipts are retained
alongside them.

| Size / threads | Before receipt | After receipt | Before / after time-v max RSS, KiB |
|---|---|---|---|
| 256 / 1 | 7 / 0.02 / 25496 / 1.44 | 7 / 0.02 / 25492 / 1.44 | 23632 / 24208 |
| 1024 / 1 | 47 / 0.14 / 25342 / 1.40 | 46 / 0.15 / 25346 / 1.40 | 151876 / 152668 |
| large / 1 | 495 / 1.44 / 22270 / 1.97 | 486 / 1.43 / 22285 / 1.97 | 1510924 / 1498128 |
| 256 / 16 | 7 / 0.05 / 25295 / 2.12 | 7 / 0.05 / 25312 / 2.12 | 48536 / 50024 |
| 1024 / 16 | 16 / 0.27 / 24983 / 3.63 | 15 / 0.27 / 24983 / 3.63 | 285176 / 284720 |
| large / 16 | 148 / 1.89 / 21645 / 10.63 | 145 / 1.89 / 21605 / 10.63 | 1977284 / 1981184 |

Batch receipt, including the fresh baseline instrument and ignored-test
captures: `rc=0 856s | peak-RSS 2.21GiB | min-avail 21619MiB |
peak-load 10.63`. Cell directories:
`$STUDY/ring-final/timing/{256,1024,large}-{1,16}t/`; each preserves
`summary.json`, `inner.json`, `admission.json`, both worker-ready records,
zenbench batches, worker stderr and time-v receipts.

The rejected implementation remains an isolated jj trial. Its code is not
on main's served path. The activity, dense/basic, scratch, panel and era
prototypes are isolated from it; an unqualified change cannot become their
control. The capture tool landed as `d6a537eef423`, with remote ancestry
verified. A later main change permits transformed derived inputs; it does
not modify the extraction kernels measured in this trial. Qualification of
any landing also checks the rebased source.

### Detector-map export

The optional standalone visualizer regeneration is not yet run. Its
`atlas-pass` command can export maps without a bank audit or certifier.
The six original decoded PNG pairs and 2,088 detector PNGs are prepared for
this proof, including seven pixels of row padding. The source exporter
retains a separate Arrow/Parquet build graph. This optional build follows
the mandatory gates; historical 2,088-PNG identities and 8,298,195 formula
replays are not receipts for these fusions.

Porting bit-identical fusions to imazen/zensim is a 0.3.1 follow-up after 0.3.0
ships; the public crate's `examples/split_parity_check.rs` against this
repository's `split_parity_golden` is the porting gate. This study does not
modify the public crate.

The expanded probes are now built and their baseline captures are complete.
Each record includes the returned vector length (720) and a SHA-256 over every
little-endian f64 bit pattern, alongside all 504 consumed slot values. The
ring receipts above retain their original 504-slot EVAL/identity coverage;
full-vector G-STEER hashes and tier gates also passed. This stronger format
landed as `32f6e69f848a`, with remote ancestry verified.

Expanded baseline receipts (`baseline-complete-v2/`): 744 identity cells,
12 permutations, 62 sources, `rc=0 321s | peak-RSS 0.29GiB | min-avail
24125MiB | peak-load 4.53`; all 4,678 EVAL pairs, `rc=0 248s | peak-RSS
0.41GiB | min-avail 25144MiB | peak-load 3.58`; NEARID's 648 feature rows
and 768 ladder scores, `rc=0 14s | peak-RSS 0.06GiB | min-avail 25522MiB |
peak-load 1.53`. These are gate-driver receipts, not speed comparisons.

The native export's two missing tracked Python fixture files were supplied;
the strict full-script lint then passed 1,550 scripts:
`rc=0 7s | peak-RSS 0.03GiB | min-avail 25574MiB | peak-load 1.41`.
The stage owners, full-vector comparators, paired aggregation and unchanged
legacy-failure-list comparator landed as `2190da00a19a`; remote ancestry was
verified. The source-export linter still rejects missing files.

### Activity-ring code gates

The activity-ring trial (`activity-qualified/`) passes the code gates. All
913 baseline test cases retain their exact statuses, and four R14 main tests
are added, yielding 917 cases (`TEST_COVERAGE.json`). The command runs all
library/integration test targets and doctests explicitly, with a case/status
comparison against the original default `cargo test` log. Its three serving
examples are compiled separately by the normal binary build.

Rev4/5/7 vector gates pass all sixteen emulated permutations; thread identity,
odd-size/strided-row tests and the split-output comparison pass. The ignored
legacy test still fails with exactly the original 41-value native and
40-value SDE lists at both Rev1 and Rev7; all differing components and both
values match (`activity-qualified/stable-diff/receipt.json`).
Combined normal build/code-gate receipt:
`rc=0 687s | peak-RSS 2.22GiB | min-avail 20781MiB | peak-load 8.77`.
The full-vector pixel replay passes all 720 returned values and score bits:
4,678 EVAL pairs, 744 identity cells, 648 NEARID rows, 135 steering bases and
55,380 repairs, plus all 768 ladder scores. Receipts: `pixel-bits.json`,
`pixel-features.json`, `eval-bits.json`; driver:
`rc=0 679s | peak-RSS 0.52GiB | min-avail 24145MiB | peak-load 6.30`.

Main subsequently acquired the parallel-size gate (`bfabf0d50e31`): below
192² the walk stays serial. A fresh baseline/activity timing pair was built
with that same parent on both arms. The earlier moment-ring table
retains its own unchanged before/after binaries and source hashes.

### Other trial state

The scratch-removal prototype's code gates also pass: normal crate/doctests,
Rev4/5/7 emulated vectors, split-output equality and complete ignored-list
invariance. Receipt: `rc=0 767s | peak-RSS 2.21GiB | min-avail 19922MiB |
peak-load 9.19`. Its pixel and timing cells remain outstanding.
Band interleaving and the first shared-leaf prototype pass strict native
release clippy/build. A vector-accumulator refinement of the shared-leaf
prototype still needs compilation and full qualification. No speed claim is
made for those builds. Panel and era qualification is outstanding.

### Full-width activity-ring rejection

This implementation is **rejected**: three of four required p50 comparisons
lose. The common-parent timing workers differ only in `blur.rs` and
`feature_v2_kernels.rs` (the activity module is identical on disk, included
only by the candidate). Their native source-compatibility receipt is
`activity-current/GATE_SOURCE_COMPATIBILITY.json`. All six cells admitted
32 clean paired rounds. Times are medians of twenty per-image percentiles.

| Size | Threads | Before p50 ms | After p50 ms | Before p95 ms | After p95 ms | p50 change |
|---|---:|---:|---:|---:|---:|---:|
| 256² | 1 | 1.861314 | 1.89195225 | 1.8761556 | 1.91038685 | +1.646% |
| 1024² | 1 | 27.27656625 | 27.59267175 | 28.21516105 | 28.7371612 | +1.159% |
| Large | 1 | 327.72765975 | 330.0050795 | 337.87387945 | 341.870055775 | +0.695% |
| 256² | 16 | 0.894088 | 0.8674865 | 1.31848935 | 1.2035196 | −2.975% |
| 1024² | 16 | 7.323532 | 7.18946225 | 8.2222606 | 8.11347905 | −1.831% |
| Large | 16 | 99.44721675 | 99.52491775 | 107.0192824 | 106.613366425 | +0.078% |

Per-worker `run-heavy` receipts below use
`seconds / peak-RSS GiB / min-avail MiB / peak-load`; all exits are zero.

| Size / threads | Before receipt | After receipt | Before / after time-v max RSS, KiB |
|---|---|---|---|
| 256 / 1 | 8 / 0.02 / 25238 / 1.01 | 8 / 0.02 / 25243 / 1.01 | 24080 / 24708 |
| 1024 / 1 | 46 / 0.15 / 25151 / 1.13 | 45 / 0.15 / 25118 / 1.13 | 152664 / 152144 |
| large / 1 | 468 / 1.44 / 22312 / 2.04 | 459 / 1.43 / 22294 / 2.04 | 1511908 / 1503296 |
| 256 / 16 | 8 / 0.05 / 25590 / 1.74 | 8 / 0.05 / 25590 / 1.74 | 48840 / 49392 |
| 1024 / 16 | 15 / 0.27 / 25098 / 2.35 | 15 / 0.27 / 25098 / 2.35 | 285808 / 286232 |
| large / 16 | 152 / 1.90 / 21360 / 11.24 | 149 / 1.88 / 21361 / 11.24 | 1987252 / 1970308 |

Batch: `rc=0 698s | peak-RSS 1.90GiB | min-avail 21370MiB |
peak-load 11.24`. Directory:
`$STUDY/activity-current/timing/{256,1024,large}-{1,16}t/`.
Before worker SHA-256:
`213659ebc0d7368540fd3f0ed2a31799dfbbbb758ffc2ecdd26b398f311fb35e`;
after worker:
`e5dc899b970686ebebf03c9259b4a9d400a125bbd2a3d2bea058af250df7e268`.

The same ranked activity-fusion candidate now has a 128-column row-ring
prototype in qualification. It preserves the H/V trees, intermediate f32
rounding and final scalar tails, with the absolute-deviation row's mirrored
halo prepared before the H tree. Its native code gates pass: all 913 baseline
case/status pairs are preserved among 917 cases, Rev4/5/7 vector gates pass
all sixteen emulated permutations, and the split golden is byte-identical.
The ignored test's four before/after comparisons retain exactly the original
41 native / 40 SDE differing values, including both bit patterns and component
names at Rev1 and Rev7. Receipts: `activity-panel/TEST_COVERAGE.json`,
`activity-panel/stable-diff/receipt.json`, `activity-panel/logs/sde-rev{4,5,7}.log`
and `activity-panel/trial-split.log`. Native build plus code gates:
`rc=0 689s | peak-RSS 2.22GiB | min-avail 20419MiB | peak-load 8.00`.
The commands were `a_fusion_trial_v5.sh build` followed by `gates`, with
study root, label `activity-panel` and `activity-panel-stage` as arguments,
inside one shared-lock/run-heavy scope. The immutable export is
`ac7cafa5c16ac85ddbd569668ecf4c1d9d53ce9e`; its source and binary hash
manifests are in `activity-panel/`. Full-vector pixel replay is queued, and
speed is not measured yet. Both full-width rings remain outside main's
served path.
