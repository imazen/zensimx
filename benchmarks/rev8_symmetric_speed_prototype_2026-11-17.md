# Rev8 symmetric streaming prototype

## Missing and limitations

**Capture in progress.** 1 of 30 paired latency cells and 80 of 120 fresh-process RSS cells have admissible evidence. Em dashes mark missing cells.

The max-pair + PCHIP head is a timing fixture with synthetic coefficients.
The benchmark uses generated RGB8 SDR scenes. It does not establish perceptual
quality, corpus rank, HDR behavior, or a production adoption decision. Rev8
defines a separate set of features and cannot serve Rev7 bakes. The current
scope covers three variants, with four or five scales.

The x86 native reciprocal estimates differ by tier. Exact swap and thread
reproducibility are within each tier. Cross-tier absolute, relative and
representational differences are measured on the same witnesses. The
AArch64 and Wasm SIMD builds were checked for compilation.

## Recipe

The RGB8 LUT and public opsin matrix feed a divide-free cube-root refinement.
A fixed black offset is subtracted on all three responses. Each strip owns one
rolling pyramid for both images. Odd axes repeat the last sample for 2×2
downsampling. An 11-row ring streams six f32 moment planes per channel and
uses periodic reflect-101 on both axes. Each output coordinate starts a fresh
pairwise box sum, so a small tap reappears after a large tap leaves the window.
Coarse-scale halos are recomputed at strip boundaries. Working row storage
scales with width and the number of active strips.

The error is formed before either input is centered. Its mean and second
moment supply the structure loss with nonnegative variances and a 0.0009 floor.
L1 and D21 retain the luminance and structure damages. L2/L4 and D42 pool
powers of those signals. SD2/SD4/SD8 pool powers of the combined dissimilarity.
Powers are formed in f64, and the other dense features accumulate in f32 and
widen once per row. The ordered strip merge is shared by all thread counts.

Visibility thresholds in standard-deviation units are 0.03, 0.05 and 0.10.
Full visibility selects the safe operand before reciprocal and power evaluation.
Appearance uses a 0.045 smooth-side decay and attenuation uses a 0.08 texture
gate. The exponential has an exact zero at and below −32, where the discarded
absolute tail is below 1.27e-14. Reciprocal floors are positive normals and
guard zero and subnormals before seeding. Positive infinity yields zero.
Within the normal upper shoulder below 2^126 the native path refines a hardware
estimate once. Scalar and Wasm refine the portable seed an additional time.

The hybrid uses base-scale luma for orientation. Interior gradient activity
is saturated as g/(g+0.05), then discounted by 0.50 times the excess jump on
the 8-pixel grid and 0.35 times the local slope reversal. The blend uses a
0.05 temperature. Exact ties average the two directed summaries. Each other
pair evaluates the logistic once from the positive gap and blends in descending
orientation order. The grid and reversal terms are local heuristic signals.

## Benchmark method

All four arms consume the same twenty deterministic textured RGB8 pairs per
geometry, with procedural flat and grain regions, gradients, clipping, chroma
changes, grid artifacts, ringing-like periodic patterns and quantization.
There is a 100 ms warmup per arm and image. Zenbench interleaves one scorer call
per arm in randomized order. The first 32 clean paired rounds per image
give 640 individual calls per arm and cell. P50 and p95 are pooled type-7
quantiles in milliseconds. Throughput is pixels divided by median latency.
Speedup is the Rev7 median divided by the arm median.

The published Rev7 control is [the pinned public A scorer](https://github.com/imazen/zensim/tree/bebc6ea45d2b6adb6ab70affc3e289707ef74fb0)
with the ordinary RGB8 `Zensim::compute` entry point, build bake SHA-256
`d5e2a56db35c32620d9f9d9c19ba3bc78bd8fea2b8cd933cde8447428d85e62a`.
The three Rev8 arms include feature extraction, pooling and the timing head.
No decode, I/O, prepared reference, maps, or process-start cost enters latency.

dev pins CPU 0 for 1T and CPUs 0–7,16–23 for 16T. The i265 pins are CPU 0 and
CPUs 0–15. The native dev tier is v4x, and the forced v3 run disables both
AVX-512 tokens before constructing either scorer. The Intel process uses the
same stripped dev-built x86-64 binary and runtime v3 dispatch. Each process
verifies the tier and Rayon pool size. The 64² walk is serial in the prototype
under either pool configuration. Available strips bound worker use at 256².

Each arm, thread count and size gets a separate GNU time process for RSS.
The process scores the twenty pairs with one pair resident at a time. RSS is
peak KiB for scorer, allocator, runtime and input buffers together. The latency
process holds the complete packet resident outside the measured calls. The
host-lock collector retains Zenbench round traces and run-heavy high-water
receipts for each invocation.

Table cells read **p50/p95 ms · Mpx/s · speedup vs Rev7 · peak RSS KiB**.
| Host / tier | Threads | Geometry | Rev7 | Alg S4 | Alg S5 | Hybrid S5 |
|---|---:|---|---|---|---|---|
| dev / v4x | 1 | 64x64 | — · 7,660 | — · 4,376 | — · 4,292 | — · 4,316 |
| dev / v4x | 1 | 256x256 | — · 14,668 | — · 4,840 | — · 5,044 | — · 5,052 |
| dev / v4x | 1 | 840x944 | — · 28,100 | — · 10,260 | — · 10,332 | — · 10,660 |
| dev / v4x | 1 | 1024x1024 | — · 32,772 | — · 12,244 | — · 12,284 | — · 12,296 |
| dev / v4x | 1 | 4000x3000 | — · 178,180 | — · 82,712 | — · 82,992 | — · 83,172 |
| dev / v4x | 16 | 64x64 | — · 7,348 | — · 4,304 | — · 3,968 | — · 4,540 |
| dev / v4x | 16 | 256x256 | — · 26,948 | — · 11,576 | — · 11,924 | — · 11,760 |
| dev / v4x | 16 | 840x944 | — · 127,716 | — · 37,432 | — · 38,820 | — · 38,720 |
| dev / v4x | 16 | 1024x1024 | — · 154,112 | — · 44,384 | — · 46,064 | — · 46,256 |
| dev / v4x | 16 | 4000x3000 | — · 598,820 | — · 208,608 | — · 215,080 | — · 215,460 |
| dev / v3 | 1 | 64x64 | — · 7,884 | — · 4,252 | — · 4,040 | — · 4,152 |
| dev / v3 | 1 | 256x256 | — · 14,680 | — · 4,920 | — · 4,872 | — · 4,888 |
| dev / v3 | 1 | 840x944 | — · 27,912 | — · 10,368 | — · 10,444 | — · 10,664 |
| dev / v3 | 1 | 1024x1024 | — · 32,748 | — · 12,232 | — · 12,300 | — · 12,300 |
| dev / v3 | 1 | 4000x3000 | — · 178,148 | — · 82,696 | — · 83,048 | — · 82,980 |
| dev / v3 | 16 | 64x64 | — · 7,428 | — · 4,036 | — · 4,036 | — · 4,076 |
| dev / v3 | 16 | 256x256 | — · 27,052 | — · 9,828 | — · 10,828 | — · 10,272 |
| dev / v3 | 16 | 840x944 | — · 126,152 | — · 37,244 | — · 39,324 | — · 38,992 |
| dev / v3 | 16 | 1024x1024 | — · 158,124 | — · 44,536 | — · 46,332 | — · 45,920 |
| dev / v3 | 16 | 4000x3000 | — · 614,196 | — · 209,576 | — · 215,448 | — · 215,524 |
| i265 / v3 | 1 | 64x64 | — · — | — · — | — · — | — · — |
| i265 / v3 | 1 | 256x256 | — · — | — · — | — · — | — · — |
| i265 / v3 | 1 | 840x944 | — · — | — · — | — · — | — · — |
| i265 / v3 | 1 | 1024x1024 | — · — | — · — | — · — | — · — |
| i265 / v3 | 1 | 4000x3000 | — · — | — · — | — · — | — · — |
| i265 / v3 | 16 | 64x64 | 2.187/3.106 · 1.87 · 1.000× · — | 0.348/1.861 · 11.76 · 6.281× · — | 0.355/1.403 · 11.53 · 6.158× · — | 0.425/1.207 · 9.64 · 5.149× · — |
| i265 / v3 | 16 | 256x256 | — · — | — · — | — · — | — · — |
| i265 / v3 | 16 | 840x944 | — · — | — · — | — · — | — · — |
| i265 / v3 | 16 | 1024x1024 | — · — | — · — | — · — | — · — |
| i265 / v3 | 16 | 4000x3000 | — · — | — · — | — · — | — · — |

## Numerical verification

The nine prototype tests pass, including 58 image pairs at eight geometries
for every variant. All score and feature bits match after swapping inputs
within each of v4x, v4, v3 and scalar on dev. On the native tier,
thread pools of 1, 2, 4, 8 and 16 reproduce both input orders on the odd
and large geometry witnesses. Identity scores 100 with positive-zero damage.
The tests also cover mismatched geometry and backing lengths, independent
periodic reflect-101 witnesses, 1-LSB and uniform shifts, reciprocal rails,
the black offset, exp monotonicity and cutoff, visibility crossovers, exact
ties and large orientation gaps.

The independent f64 walk computes means and centered variances from the
121 samples at every compared coordinate. The largest pooled absolute deltas
across five odd/tiny fixtures and all active scales were 9.4610e-8 for L1,
1.1358e-6 for D21 and 2.7653e-6 for MSE. It caught a blue-channel jump at the
black endpoint during development. Subtracting the same black offset across
the entire third response removed that jump.

| Tier vs dev v4x | Values checked | Different f64 bits | Max absolute | Max relative | Max f32 ULP | Max f64 ULP |
|---|---:|---:|---:|---:|---:|---:|
| v4 | 62,814 | 0 | 0 | 0 | 0 | 0 |
| v3 | 62,814 | 13,286 | 7.249418e-6 | 4.000000e-2 | 524,288 | 281,474,976,710,656 |
| Scalar | 62,814 | 11,449 | 9.986735e-7 | 4.761905e-3 | 66,576 | 35,742,854,185,481 |

The value counts include the score and every reported feature slot. ULP
distances are measured after f32 cast or on the original f64 outputs as
labeled. The large ULP distances arise on small near-zero features and show
that the original fractional-ULP target is unfulfilled on v3 and scalar.
The largest absolute differences in both comparisons are 0–100 dial scores.

The x86 assembly audit covers 94 dense symbols including all native
color, horizontal, vertical and feature row kernels, and their outlined
helpers. It finds zero packed divides and checks YMM on v3 and ZMM on v4/v4x.
The narrow scalar postprocessing divisions are outside the dense row kernels.

The default library tests passed 483 cases, and the frozen Rev4 integration
test passed eight. The full `cargo test -p zensim --features rev8-proto` suite passed on the
final capture source. Workspace Clippy and the script lint also passed. The
frozen served-source checkers compare all 135,625 expected split rows
bit-for-bit against the sibling checker, including their feature hashes.
The shipped bake-inclusion gate passed. `feature_v2.rs` is 556,514 bytes with
SHA-256 `3214c4459e7611befba8d19fb21f0c6304997684c76a6db5b8806475c8824f13`.
The snapshot helper refused discovery because the research workspace
contains zero publishable library crates. The prototype API is hidden and
requires the non-default feature.

## Diagnostic witnesses

The fragility columns compare max and mean across both image sides after
multiplying by damage. Phi pairs show the unpenalized saturated activity and
the discounted orientation. Measurements use base-scale luma.

| Witness | Fragility max / mean | Phi A raw → discounted | Phi B raw → discounted |
|---|---:|---:|---:|
| Flat / texture transition | 0.26900 / 0.13450 | 0 → 0 | 0.43537 → 0.36022 |
| Blur / grain | 0.03032 / 0.02019 | 0.47131 → 0.46807 | 0.54533 → 0.53539 |
| Brightness shift | 0.0008241 / 0.0008239 | 0.50059 → 0.49712 | 0.49635 → 0.49294 |
| Exposure shift | 0.02282 / 0.01776 | 0.50059 → 0.49712 | 0.51150 → 0.50723 |
| Clipping | 0.02098 / 0.01590 | 0.50059 → 0.49712 | 0.29596 → 0.29362 |
| Luminance-preserving translation | 0.04445 / 0.02465 | 0.50059 → 0.49712 | 0.50556 → 0.50163 |
| Blocking shifted off the grid | 0 / 0 | 0 → −0.00585 | 0.08517 → 0.08517 |
| 7-pixel block spacing | 0 / 0 | 0 → −0.00585 | 0.08517 → 0.08433 |
| Ringing pulse train | 0.07719 / 0.03859 | 0 → 0 | 0.20386 → 0.19254 |
| Noise / periodic texture | 0.32585 / 0.19790 | 0.67440 → 0.64266 | 0.50059 → 0.49712 |
| Identical flat fields | 0 / 0 | 0 → 0 | 0 → 0 |

Across the 11 witnesses the minimum, median and maximum absolute orientation
gaps were 0, 0.090179 and 0.360215, with one exact tie. The ringing and
blocking discounts affect these deliberate witnesses. The noise and
periodic texture pair shows substantial residual orientation after applying
both discounts.

## Learnings

- Rebuilding the box tree at each output coordinate retains small
  post-transition values that a subtract-and-add f32 recurrence erases.
  Centering and the directly pooled error plane control cancellation in
  the normalized structure signal.
- A proof on the scoring entry point does not inline primitive SIMD
  operations into a Rayon worker. Explicit target-feature boundaries on
  the dense rows let the compiler use the expected registers directly.
- Native recognition of reciprocal estimates and exact swap reproducibility
  are separate properties. The same arithmetic tree can be symmetric at
  every tier while individual values drift across tiers.
- Brightness and texture changes move the orientation proxy even after
  discounts. These witnesses establish what enters the timing fixture.

The fresh-process RSS passes supply 800 paired 1T/16T score comparisons, with 0 differing score bit patterns.

## Build and resource receipts

Release uses thin LTO and one codegen unit with runtime dispatch. The measured scorer and fixture
implementation are in local revision `6a48974a`. The stripped runtime SHA-256 is
`65235d83b3a392648c5280b438d274ad55bfd51a9645df7ad9f40ce85c4df68e`.

All accepted worker receipts accompany the raw captures. The following high-water receipts
are from the fresh-process 12 MP RSS runs. The timing process retains the 20-pair
fixture packet, so its own resource high-water measurements have a different scope.

| Host / tier | Threads | Arm | Run-heavy receipt |
|---|---:|---|---|
| dev / v4x | 1 | zensim-a-rev7 | `run-heavy: done rc=0 7s · peak-RSS 0.17GiB · min-avail 39352MiB · peak-load 3.92` |
| dev / v4x | 1 | rev8-alg-s4 | `run-heavy: done rc=0 11s · peak-RSS 0.08GiB · min-avail 39355MiB · peak-load 4.26` |
| dev / v4x | 1 | rev8-alg-s5 | `run-heavy: done rc=0 11s · peak-RSS 0.08GiB · min-avail 39523MiB · peak-load 4.48` |
| dev / v4x | 1 | rev8-hybrid-s5 | `run-heavy: done rc=0 12s · peak-RSS 0.08GiB · min-avail 38712MiB · peak-load 4.72` |
| dev / v4x | 16 | zensim-a-rev7 | `run-heavy: done rc=0 5s · peak-RSS 0.57GiB · min-avail 37504MiB · peak-load 9.46` |
| dev / v4x | 16 | rev8-alg-s4 | `run-heavy: done rc=0 4s · peak-RSS 0.20GiB · min-avail 39351MiB · peak-load 8.73` |
| dev / v4x | 16 | rev8-alg-s5 | `run-heavy: done rc=0 5s · peak-RSS 0.21GiB · min-avail 39143MiB · peak-load 8.45` |
| dev / v4x | 16 | rev8-hybrid-s5 | `run-heavy: done rc=0 5s · peak-RSS 0.21GiB · min-avail 38218MiB · peak-load 9.55` |
| dev / v3 | 1 | zensim-a-rev7 | `run-heavy: done rc=0 10s · peak-RSS 0.17GiB · min-avail 38336MiB · peak-load 3.07` |
| dev / v3 | 1 | rev8-alg-s4 | `run-heavy: done rc=0 15s · peak-RSS 0.08GiB · min-avail 38456MiB · peak-load 3.53` |
| dev / v3 | 1 | rev8-alg-s5 | `run-heavy: done rc=0 18s · peak-RSS 0.08GiB · min-avail 38356MiB · peak-load 3.87` |
| dev / v3 | 1 | rev8-hybrid-s5 | `run-heavy: done rc=0 20s · peak-RSS 0.08GiB · min-avail 38438MiB · peak-load 4.52` |
| dev / v3 | 16 | zensim-a-rev7 | `run-heavy: done rc=0 5s · peak-RSS 0.59GiB · min-avail 37885MiB · peak-load 2.76` |
| dev / v3 | 16 | rev8-alg-s4 | `run-heavy: done rc=0 5s · peak-RSS 0.20GiB · min-avail 38464MiB · peak-load 2.37` |
| dev / v3 | 16 | rev8-alg-s5 | `run-heavy: done rc=0 5s · peak-RSS 0.21GiB · min-avail 38405MiB · peak-load 3.45` |
| dev / v3 | 16 | rev8-hybrid-s5 | `run-heavy: done rc=0 6s · peak-RSS 0.21GiB · min-avail 38566MiB · peak-load 4.09` |
