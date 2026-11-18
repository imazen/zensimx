# Rev8 symmetric streaming prototype

## Missing and limitations

The owner narrowed the timing matrix to 1024×1024 and 4000×3000. Accepted evidence covers 12 of 12 descriptive latency cells and 96 of 96 scoped RSS measurements. The original full-sweep capture contributes 1 accepted Zenbench cell.

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

## Descriptive end-to-end timings

The timed call includes the RGB8 scorer, feature extraction, pooling and
the complete scoring head. The same resident source and distorted RGB8
buffers are used by all arms. Fixture 00 was held fixed from the existing
deterministic packet, at each geometry. The four arms each get one
untimed warm-up call. The three matched, single-call rounds run in
forward, reverse and forward arm order. The measured range is the
smallest and largest of those three wall-clock observations.

The public control is [the pinned public A scorer](https://github.com/imazen/zensim/tree/bebc6ea45d2b6adb6ab70affc3e289707ef74fb0)
with the ordinary RGB8 `Zensim::compute` entry point and bake SHA-256
`d5e2a56db35c32620d9f9d9c19ba3bc78bd8fea2b8cd933cde8447428d85e62a`.

Dev pins CPU 0 for 1T and CPUs 0–7,16–23 for 16T. The i265 pins are CPU 0
and CPUs 0–15. The native dev tier is v4x, and forced v3 disables both
AVX-512 tokens before constructing either scorer. Intel uses the same
stripped dev-built binary with runtime v3 dispatch. The host locks
serialize the requested captures. The descriptive replay uses the normal
host placement and resource caps.

Each cell reads **median [min, max] ms · Mpx/s at median · Rev7 median / arm median**.

| Host / tier | Threads | Geometry | Fixture | Rev7 | Alg S4 | Alg S5 | Hybrid S5 |
|---|---:|---|---:|---|---|---|---|
| dev / v4x | 1 | 1024x1024 | 00 | 19.143 [18.648, 19.973] · 54.78 · 1.000× | 38.445 [37.965, 38.581] · 27.28 · 0.498× | 45.579 [45.462, 45.597] · 23.01 · 0.420× | 50.127 [50.103, 50.257] · 20.92 · 0.382× |
| dev / v4x | 1 | 4000x3000 | 00 | 185.615 [185.473, 185.988] · 64.65 · 1.000× | 373.520 [373.394, 374.476] · 32.13 · 0.497× | 408.740 [408.317, 409.752] · 29.36 · 0.454× | 459.625 [458.943, 464.005] · 26.11 · 0.404× |
| dev / v4x | 16 | 1024x1024 | 00 | 7.336 [4.474, 13.081] · 142.94 · 1.000× | 5.105 [4.527, 5.244] · 205.41 · 1.437× | 5.254 [5.063, 5.277] · 199.58 · 1.396× | 5.640 [5.578, 5.658] · 185.92 · 1.301× |
| dev / v4x | 16 | 4000x3000 | 00 | 101.406 [94.872, 111.741] · 118.34 · 1.000× | 86.273 [84.648, 88.852] · 139.09 · 1.175× | 87.485 [87.464, 88.501] · 137.17 · 1.159× | 89.105 [86.383, 92.027] · 134.67 · 1.138× |
| dev / v3 | 1 | 1024x1024 | 00 | 30.667 [30.408, 31.350] · 34.19 · 1.000× | 69.718 [69.195, 70.261] · 15.04 · 0.440× | 89.962 [89.140, 90.051] · 11.66 · 0.341× | 99.619 [98.517, 99.734] · 10.53 · 0.308× |
| dev / v3 | 1 | 4000x3000 | 00 | 284.115 [282.363, 284.386] · 42.24 · 1.000× | 637.313 [636.454, 638.821] · 18.83 · 0.446× | 732.648 [732.194, 734.986] · 16.38 · 0.388× | 841.602 [840.880, 842.856] · 14.26 · 0.338× |
| dev / v3 | 16 | 1024x1024 | 00 | 9.374 [6.975, 13.592] · 111.86 · 1.000× | 7.470 [7.306, 10.045] · 140.36 · 1.255× | 9.039 [9.011, 12.497] · 116.01 · 1.037× | 10.204 [10.047, 10.246] · 102.77 · 0.919× |
| dev / v3 | 16 | 4000x3000 | 00 | 115.847 [110.487, 122.790] · 103.58 · 1.000× | 108.275 [107.156, 109.394] · 110.83 · 1.070× | 117.882 [115.354, 118.502] · 101.80 · 0.983× | 129.690 [128.811, 129.721] · 92.53 · 0.893× |
| i265 / v3 | 1 | 1024x1024 | 00 | 35.259 [33.325, 35.693] · 29.74 · 1.000× | 81.953 [81.211, 82.998] · 12.79 · 0.430× | 108.044 [106.387, 108.515] · 9.71 · 0.326× | 118.295 [117.697, 118.873] · 8.86 · 0.298× |
| i265 / v3 | 1 | 4000x3000 | 00 | 367.063 [366.505, 373.640] · 32.69 · 1.000× | 883.314 [881.459, 884.115] · 13.59 · 0.416× | 1004.441 [1004.302, 1004.950] · 11.95 · 0.365× | 1128.383 [1127.636, 1130.898] · 10.63 · 0.325× |
| i265 / v3 | 16 | 1024x1024 | 00 | 22.266 [16.618, 37.817] · 47.09 · 1.000× | 9.628 [8.271, 10.091] · 108.91 · 2.313× | 10.318 [9.471, 11.086] · 101.63 · 2.158× | 11.269 [11.254, 12.064] · 93.05 · 1.976× |
| i265 / v3 | 16 | 4000x3000 | 00 | 188.802 [176.360, 206.132] · 63.56 · 1.000× | 202.667 [201.920, 203.235] · 59.21 · 0.932× | 211.478 [209.952, 212.659] · 56.74 · 0.893× | 216.065 [215.834, 240.844] · 55.54 · 0.874× |

All three Rev8 variants had higher medians than Rev7 in the six 1T cells. Alg S4 had the lowest 16T median in all four dev cells and at 1024² on i265. At 12 MP and 16T on i265, Rev7 recorded 188.802 ms and Alg S4 recorded 202.667 ms. Each Rev8 variant used less peak RSS than Rev7 at 12 MP in every placement and thread count. The latency findings describe fixture 00 and three rounds per cell.

| Geometry | Fixture ID | RGB8 source FNV-1a 64 | RGB8 distorted FNV-1a 64 |
|---|---:|---|---|
| 1024x1024 | 00 | `64ff9309176504db` | `fa2ec446e996b737` |
| 4000x3000 | 00 | `65379aaed2fd6068` | `1f77de5b662649ec` |

## Original admitted Zenbench evidence

The full clean-round sweep was withdrawn. The earlier accepted cell
retains its original method of 20 fixtures, 100 ms warm-up per arm and
fixture, and the first 32 clean interleaved single-call rounds per fixture.
P50 and p95 are pooled type-7 quantiles. Rejected and interrupted captures
remain in the handoff alongside the admitted receipts. One dev attempt
started after the scratch admission check changed and was stopped. It
remains in the rejected receipts.

| Host / tier | Threads | Geometry | Calls per arm | Rev7 p50 / p95 ms | Alg S4 | Alg S5 | Hybrid S5 |
|---|---:|---|---:|---:|---:|---:|---:|
| i265 / v3 | 16 | 64x64 | 640 | 2.1869 / 3.1063 | 0.3482 / 1.8613 | 0.3552 / 1.4035 | 0.4247 / 1.2068 |

## Peak RSS

Each arm, thread count and geometry uses a fresh GNU time process. It
scores the original twenty pairs with one pair resident at a time. The
peak is KiB for scorer, allocator, runtime and buffers together. The dev
RSS reads were captured with the original instrument. The scoped Intel
RSS reads use the rebuilt instrument with identical scorer source.

| Host / tier | Threads | Geometry | Rev7 KiB | Alg S4 KiB | Alg S5 KiB | Hybrid S5 KiB |
|---|---:|---|---:|---:|---:|---:|
| dev / v4x | 1 | 64x64 | 7,660 | 4,376 | 4,292 | 4,316 |
| dev / v4x | 1 | 256x256 | 14,668 | 4,840 | 5,044 | 5,052 |
| dev / v4x | 1 | 840x944 | 28,100 | 10,260 | 10,332 | 10,660 |
| dev / v4x | 1 | 1024x1024 | 32,772 | 12,244 | 12,284 | 12,296 |
| dev / v4x | 1 | 4000x3000 | 178,180 | 82,712 | 82,992 | 83,172 |
| dev / v4x | 16 | 64x64 | 7,348 | 4,304 | 3,968 | 4,540 |
| dev / v4x | 16 | 256x256 | 26,948 | 11,576 | 11,924 | 11,760 |
| dev / v4x | 16 | 840x944 | 127,716 | 37,432 | 38,820 | 38,720 |
| dev / v4x | 16 | 1024x1024 | 154,112 | 44,384 | 46,064 | 46,256 |
| dev / v4x | 16 | 4000x3000 | 598,820 | 208,608 | 215,080 | 215,460 |
| dev / v3 | 1 | 64x64 | 7,884 | 4,252 | 4,040 | 4,152 |
| dev / v3 | 1 | 256x256 | 14,680 | 4,920 | 4,872 | 4,888 |
| dev / v3 | 1 | 840x944 | 27,912 | 10,368 | 10,444 | 10,664 |
| dev / v3 | 1 | 1024x1024 | 32,748 | 12,232 | 12,300 | 12,300 |
| dev / v3 | 1 | 4000x3000 | 178,148 | 82,696 | 83,048 | 82,980 |
| dev / v3 | 16 | 64x64 | 7,428 | 4,036 | 4,036 | 4,076 |
| dev / v3 | 16 | 256x256 | 27,052 | 9,828 | 10,828 | 10,272 |
| dev / v3 | 16 | 840x944 | 126,152 | 37,244 | 39,324 | 38,992 |
| dev / v3 | 16 | 1024x1024 | 158,124 | 44,536 | 46,332 | 45,920 |
| dev / v3 | 16 | 4000x3000 | 614,196 | 209,576 | 215,448 | 215,524 |
| i265 / v3 | 1 | 1024x1024 | 33,000 | 12,484 | 12,680 | 12,572 |
| i265 / v3 | 1 | 4000x3000 | 178,572 | 82,900 | 83,256 | 83,292 |
| i265 / v3 | 16 | 1024x1024 | 154,356 | 45,960 | 47,096 | 47,244 |
| i265 / v3 | 16 | 4000x3000 | 586,664 | 209,976 | 215,768 | 215,184 |

The fresh-process RSS passes supplied 960 paired 1T/16T score-bit comparisons, with 0 differences. The descriptive capture reader also checks the score bits across thread counts wherever both cells exist.

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

## Build and resource receipts

The scored code is from local revision `6a48974a`. The release build
uses thin LTO and one codegen unit with runtime CPU dispatch. The three-round
mode adds timing and fixture identity outside the shared scoring calls.

| Instrument | Stripped-binary SHA-256 |
|---|---|
| Original Zenbench and dev RSS | `65235d83b3a392648c5280b438d274ad55bfd51a9645df7ad9f40ce85c4df68e` |
| Descriptive timings and scoped i265 RSS | `63be36b3aebfceb4e5620c20cf01efc23ff3a618bdbe09dcd2d43070b09e0647` |

All admitted worker logs retain their run-heavy receipts. The table gives
the receipts for the 12 MP fresh-process RSS workers.

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
| i265 / v3 | 1 | zensim-a-rev7 | `run-heavy: done rc=0 11s · peak-RSS 0.17GiB · min-avail 23530MiB · peak-load 6.87` |
| i265 / v3 | 1 | rev8-alg-s4 | `run-heavy: done rc=0 22s · peak-RSS 0.08GiB · min-avail 23561MiB · peak-load 5.01` |
| i265 / v3 | 1 | rev8-alg-s5 | `run-heavy: done rc=0 23s · peak-RSS 0.08GiB · min-avail 23642MiB · peak-load 3.87` |
| i265 / v3 | 1 | rev8-hybrid-s5 | `run-heavy: done rc=0 26s · peak-RSS 0.08GiB · min-avail 23197MiB · peak-load 3.11` |
| i265 / v3 | 16 | zensim-a-rev7 | `run-heavy: done rc=0 8s · peak-RSS 0.56GiB · min-avail 23054MiB · peak-load 2.67` |
| i265 / v3 | 16 | rev8-alg-s4 | `run-heavy: done rc=0 7s · peak-RSS 0.20GiB · min-avail 23259MiB · peak-load 2.67` |
| i265 / v3 | 16 | rev8-alg-s5 | `run-heavy: done rc=0 7s · peak-RSS 0.21GiB · min-avail 23291MiB · peak-load 3.61` |
| i265 / v3 | 16 | rev8-hybrid-s5 | `run-heavy: done rc=0 8s · peak-RSS 0.21GiB · min-avail 23422MiB · peak-load 4.71` |

The public diff was checked for household identifiers, LAN addresses,
MAC-address patterns and private source names. The frozen
`feature_v2.rs` hash, all revised scorer tests and the split comparison
are recorded above.
