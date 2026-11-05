# zensim-viz report

## Public A explorer — 2026-11-05

The A-only replacement is running on port 3310. Final source pin, one-object
R2 publication, final captures and the exact Pages workflow diff are being
completed. Pages has not been enabled, its workflow has not been installed in
imazen/zensim, and no workflow has been dispatched.

The browser computes detector maps and all PDP/ICE through the Rust owners.
Shipped A is exactly SHA-256
`d5e2a56db35c32620d9f9d9c19ba3bc78bd8fea2b8cd933cde8447428d85e62a`,
24,504 bytes. It has 984 declared inputs and 126,080 weights. The compact
samples contain 200 observed rows per bank: SDR population 12,163 and HDR
population 7,390. Each bank's sample is 403,200 bytes, raw quantiles 42,336
bytes and declared-input quantiles 82,656 bytes. The measured maximum sampled
served-score change from f32 storage is 0.0 in both banks.

The complete public data is one range-readable IMIJ-set with 14 internal
entries; the current verified archive is 1,608,550 bytes. There are zero sample
image assets. The browser reader verifies footer identity, bounds and every
payload SHA-256; the existing native Writer/Reader checks full byte roundtrip
and native checksums. Public JSON contains registry equations and public
bibliographic links, with no paper prose, private paper links or local paths.

The single-thread SIMD Wasm is 1,645,879 bytes; generated JavaScript glue is
23,303 bytes. The module contains the exact A bake and no built-in B bake.
The native and Wasm detector paths retain the old file exporter and share its
map owner. Five dimension/stride/identity cases compare all 720 pooled f64
positions with an independent control: zero changed bits. Four cases cover
all 504 consumed raw slots (444 spatial, 60 global), and the 1024×768 case
covers eight representative slots. The browser checks 2,024 raw entries,
every spatial value and map sum, and 180 basic cursor witnesses.

Dev Chromium: score maximum delta 0; cell maximum scaled delta
1.1102230246251565e-16; map-sum maximum scaled delta
1.5406422314976243e-15. The unchanged limits are 1e-6 for scores and 1e-4
for values/sums, relative to max(1, native magnitude). Both banks' curves,
centered bands, rise witnesses, family curves and diagnostics pass 200,598
numeric comparisons; maximum scaled delta 2.22006761361609e-16.
There are also 162 independent direct BakeScorer spot checks.

| Dev browser probe | Pair extraction, ms | Spatial maps | Median map, ms | Maximum map, ms |
| --- | ---: | ---: | ---: | ---: |
| 16×19, padded rows | 19.8 | 444 | 0.0 | 1.3 |
| 65×67, padded rows | 4.9 | 444 | 0.1 | 0.6 |
| 96×80 | 5.0 | 444 | 0.1 | 2.2 |
| 97×81, identity, padded rows | 4.6 | 444 | 0.1 | 0.5 |
| 1024×768, padded rows | 256.2 | 7 | 33.3 | 39.0 |

These are one-pass browser API timings, including extraction/retention or map
serialization and parsing, without image-file decoding. The first case
includes first-use costs; values of 0.0 are below the browser timer's 0.1 ms
resolution, not zero work. Empty global entries are excluded from map timing
summaries. The guarded dev batch recorded peak RSS 1.39 GiB, minimum available
43,614 MiB and peak load 2.19. Its scientific gates passed; its page smoke then
rejected an older bank-source fingerprint. Fresh matching exports passed the
complete browser checks; the strict fingerprint check was retained.

All 34,927 native audit flags, thresholds and counts exactly match the frozen
census. In particular, 32,344 zero-MAD row findings retain null statistics,
which the UI displays as undefined. The 10 native tests and separate file-map
parity test pass without ignored tests. Exporter and workbench strict clippy
pass. The normal workbench mode also passes its score and eight-loop-map
regression after the Pages-only Wasm export split. The current-main guarded
qualification on r5900xt took 314 seconds with peak RSS 1.53 GiB, minimum
available 45,250 MiB and peak load 5.33. The six existing Wasm-only unused-item
warnings remain; no warning or numerical expectation was disabled.

A's canonical certificate has one max pair, no free ramps and no free
amplitude. The function retains all 31 nonzero affine-background units and
both pair units. On the certified nonnegative standardized-input domain,
there are at most two affine head forms before the 21-knot output
calibration. This does not bound the pixel-to-score function to two affine
forms. Memorization presence or absence is not measured or established.

All 984 inputs join to 35 registry signal equations/descriptions, plus the
fragility-product equation and its native reference-gradient factor. The
reference export preserves all 36 formulas and 36 bibliographic links from
the previous bundle. Output-layer indexes link to hidden units, not input
features. Matrix ordering comes from the native exporter; stored identities
and 14 px cells are preserved. The supplied design directory is unchanged.

Compared with the review build, the page has one bake, code-generated/local
pairs, on-demand maps and on-demand 200-row curves. It omits cross-bake
comparisons, bundled photographs, private paper prose, the difference-image
exhibit, image zoom/pan, resizable gutters, matrix minimap and complete-bank
ablation rankings. Ablation/separation now use the labelled 200-row sample;
full-population quantile grids and full SDR audit findings are retained.
Tree chips show native population p50 values. The ordinary workbench keeps
its attribution/loop APIs; those unused exports are absent from the public
Pages module. No native HDR image decoding or empirical memorization test is
claimed.

The source census includes native preparation, all historical visualizer
tests, the workbench API/glue, both observer modules and shared integration
additions. It excludes prose, lockfiles, generated artifacts and the separate
58,856-byte JPEG XL loop-convergence CLI, which is not in the Wasm dependency
graph. All native preparation helpers and old numeric gates were preserved.
The retired eight-file viewer runtime is preserved byte for byte with the
review artifacts. Final exact code bytes, source commit, object receipt and
capture directory follow after publication qualification.

---

## Explained MLP function — owner follow-up completed

MLP feature references now open the rendered formula, detector description,
background and citations. Concerns, ablations, selected weights, incoming-weight
strips, hidden-unit input tables and coefficient tables use named links.
Reading-unit links open its max pair or individual term. Copied feature/unit URLs
restore bake and bank, preserving the raw/fragility join and stored neuron IDs.
The copied E h126 URL is checked against its actual pair `[124,126]`.

The new Function exhibit exposes the algebra behind the stored head:

`p(z) = 100 + L(z) + Σ c_k max(A_k(z), B_k(z))`.

Here `d_i` is a declared raw feature or damage × fragility product, and
`z_i = d_i / scale_i` for these zero-mean, positive-scale bakes. Each pair uses
`A = w_u·z`, `B = (w_u+w_v)·z+b_v` and the equal stored output coefficient.
The coefficient, bias, full planes, input scales and the complete background
`L(z)` have native exports and linked tables. Display equations round their
coefficients and use an approximation sign; the exact stored values remain
available. This is real-arithmetic interpretation, not a replacement scorer
or a promise that reassociated arithmetic preserves f32 bits.

The native domain check proves F0's 31 small residual units and F2's one are
always active on nonnegative standardized inputs: all their encoder weights
and biases are nonnegative. They form an affine background without extra
switches. F1's 31 otherwise nondead residual units and E's 29 have exactly zero
output weights. No nonzero term was removed because it was small.

| Bake | Max pairs | Nonzero linear background units | Other nonzero hinges | Other nondead zero-output units | Affine forms ≤ | Pieces per line ≤ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F seed 0 | 1 | 31 | 0 | 0 | 2 | 2 |
| F seed 1 | 1 | 0 | 0 | 31 | 2 | 2 |
| F seed 2 | 1 | 1 | 0 | 0 | 2 | 2 |
| E | 3 | 0 | 0 | 29 | 8 | 4 |

These bounds apply **before output calibration, on the certified nonnegative
model-input domain**. Each max pair chooses between two affine branches;
a linear background adds no branch. Three pairs therefore yield at most eight
combinations and at most four pieces along a straight line. Choices can be
unreachable or identical; this is not a measured region count. Raw feature
extraction and fragility products are nonlinear, and all four bakes have a
21-knot output calibration spline. The bounds do not describe the complete
pixel-to-score function as a piecewise affine map.

**Memorization presence or absence is not established.** Learned coefficients
can overfit despite a small number of pieces. No random-label fit, training /
held-out gap, membership or duplicate-source experiment was run. This follow-up
reads model bytes and existing admitted inspection bundles, without a training
fit or an AIC table read. The UI states that distinction beside the capacity
bounds and retains every original per-weight finding.

Each pair and remaining nonzero unit has a 65-point native basis slice. The
hidden-layer/output-layer replay must match the complete Predictor's f32 bits
at every sample. Complete served-score curves call BakeScorer at 512×512 with
no codec hint. These slices set other raw slots and fragility to zero and are
labelled as mathematical examples, not observed images or PDP. Network and
served-score modes keep the output calibration visible.

All retained Design 1/2/review-2 browser checks pass. New checks visit every
pair's full coefficient table and native curve values, the complete affine
background, raw/product links, and copied feature/unit URLs. All 14 Function
captures have document dimensions equal to the viewport. The entire chart and
x-axis fit the pane; at 1600×1000 its plot area measures 544px in a 900px stage.
There are no uncaught browser errors. The gallery has 25 PNGs: 14 Function views
and 11 retained atlas/lens/Δ/weights views. F0/E have both themes and sizes;
F1/F2 have both themes at 1600×1000; two additional dark views inspect the
linear residual terms.
Earlier plot/bound captures remain in `function-screenshots-before-affine-fit/`.

The preservation recheck matches all 2,088 detector PNGs, original feature fields,
pooled/layout bits and original weight fields. Existing audits, flags and curves
are reused unchanged. No established expectation, threshold or numeric guard
was relaxed. Native validation passes the frozen F audit, direct serving-curve
comparisons, layer replay, affine-background proof and odd/strided Atlas checks.
Strict clippy passes for the tool and metric library, and script hygiene lint
passes. Resource receipts:

- Native tests: `rc=0 30s | peak-RSS 0.82GiB | min-avail 36672MiB | peak-load 3.13`.
- Tool clippy: `rc=0 0s | peak-RSS 0.23GiB | min-avail 37743MiB | peak-load 2.47`.
- Library clippy: `rc=0 5s | peak-RSS 0.66GiB | min-avail 35201MiB | peak-load 2.77`.
- Browser/captures: `rc=0 24s | peak-RSS 0.34GiB | min-avail 36318MiB | peak-load 1.70`.
- Script lint: `rc=0 17s | peak-RSS 0.04GiB | min-avail 37995MiB | peak-load 0.77`.

CI was inspected without dispatching a workflow under repository policy.
Native milestone `3782f794b638`, navigation/UI `998762b89783`, affine proof
`594565a0aba9`; final capture source `3e875e25ff78`. Hashes, model identities,
structural bounds and receipts are in [function-evidence.json](function-evidence.json).
The final capture directory is
`/mnt/v/output/zensim/viz/2026-11-04/function-screenshots/`.
Existing unsupported HDR maps, missing paper prose and teacher/human limits below
remain explicit. The function exporter currently refuses transformed or auxiliary
heads outside its declared supported shape.

http://dev.lan:3310/#bake=F0&bank=cid22-fit&unit=127

http://dev.lan:3300/zensim/viz/2026-11-04/function-screenshots/function-index.html

## Design review 2 — 8/8 done and measured

All eight decisions are implemented, verified and captured. Native metadata
landed at `43fe65e9952e`; the UI at `ceeef101f79a`; guarded recipes at
`581365b6bff8`; the final capture source is `ccafbc48abaa`.

1. **Done — visible Atlas.** The photographic base is grey with a selected
   map, except in Δ mode. Exact zeros stay transparent; positive pixels use
   `slider × (0.25 + 0.75t)`. Native cached p2–p98 map-pixel bounds are the
   default, with own and bank p1–p99 controls. Map-only is opaque. For the
   pinned f20 building map, 129,423 pixels have alpha 67–255 before the 0.65
   slider; 79,576 remain exactly transparent. Own upper bound is
   0.35488855838775635; robust upper bound is 0.0904617893929575.
   All four Atlas captures visibly show the map on the building.
2. **Done — stage-filling Lens.** Dependence and range plot areas measure
   568px in the 900px stage at 1600×1000, including after gutter resize.
   The concern table is 631px tall and the weight matrix region 682px.
   The per-weight audit remains expandable; ranked strips and full tables
   remain accessible through the exhibit's internal scroll.
3. **Done — model strip.** Native total |weight|, ablation rank, reading
   units and flag count/severity appear in two lines directly below the
   pooled value. The full model section remains. Cursor terms appear only
   in Atlas; Lens shows input/unit facts. Missing bank evidence clears
   stale model and capacity facts.
4. **Done — sidebar footer.** The permanent sentence moved to the app-name
   title and the Concerns footnote.
5. **Done — compact bank selector.** Below 1400px the selected label reads
   `CID22 TRAIN`; native option labels and titles retain the full bank name.
6. **Done — carried details.** The tree has the five-step p0–p100 chip legend,
   duplicate-input rows show signed native `r`, and the global-input glyph
   has a visible label and title in the weight legend.
7. **Done — capacity strip.** Every bake has 128 native-classified cells,
   coefficient/bias tooltips and stored-ID click/keyboard selection. Dead
   means an all-zero encoder column and bias ≤ 0. Of the remaining units,
   residual means |output coefficient| < 1e-3 of the largest; the rest are live.
8. **Done — preservation and acceptance.** The original Design 1/2 browser
   gates pass alongside the new checks. All eight size/theme/view combinations,
   Δ and the weight exhibit have document dimensions equal to the viewport.
   There are no uncaught browser errors. Design assets remain verbatim.

| Bake | Live | Residual | Dead |
| --- | ---: | ---: | ---: |
| F seed 0 | 2 | 31 | 95 |
| F seed 1 | 2 | 31 | 95 |
| F seed 2 | 2 | 1 | 125 |
| E | 6 | 29 | 93 |

The six final pair bundles retain **2,088 byte-identical detector PNGs** and
all original feature fields. Each pair preserves 504 populated-slot and 720
layout-bit identities; **8,298,195 native sampled formula replays** still match.
All four weight documents preserve every original field. Six bank audits,
existing curves and all measured flags are reused unchanged. No threshold or
zero-tolerance check was relaxed. The new bank display bounds share the
original dependence-grid expression, preserving even its floating-point
endpoint representation.

Native validation passed: 8 unit tests and the odd-size/strided Atlas test;
standalone and metric-library strict clippy. Test resource receipt:
`rc=0 30s | peak-RSS 0.82GiB | min-avail 39726MiB | peak-load 4.37`.
Standalone clippy: `rc=0 1s | peak-RSS 0.23GiB | min-avail 40906MiB |
peak-load 1.55`; metric-library clippy: `rc=0 1s | peak-RSS 0.07GiB |
min-avail 42594MiB | peak-load 1.88`. Final guarded browser/capture receipt:
`rc=0 13s | peak-RSS 0.31GiB | min-avail 42280MiB | peak-load 1.33`.
Syntax, numeric formatting, network HTTP and design-copy checks pass.
CI was inspected; no Actions run was dispatched under repository policy.
These are local verification results, not a CI-green claim.

The final capture directory is
`/mnt/v/output/zensim/viz/2026-11-04/review2-screenshots/`: eight overview
images, one Δ image, two capacity/weight images, a gallery and both acceptance
receipts. Inspection caught an early Δ screenshot taken before image decode;
the capture now awaits decode and verifies its native image URL. The earlier
set remains in `review2-screenshots-before-delta-decode/`.
Hashes and measurements are preserved in [review2-evidence.json](review2-evidence.json).
Existing HDR-map, teacher/human and missing-paper-prose limits below remain.
No review-2 item is open.

http://dev.lan:3310/

http://dev.lan:3300/zensim/viz/2026-11-04/review2-screenshots/index.html

http://dev.lan:3300/zensim/viz/2026-11-04/review2-screenshots/browser-review.json

## Design 2 — completed and measured

Both views now use the fixed 44px header, virtual family/signal/scale tree,
central stage, reference pane and 56px filmstrip. Exactly one of Concerns,
Weights, Dependence, Ranges and Attribution occupies the lens stage. The
resizable gutters support drag and arrows; search, channel cells, pins, image
modes, zoom, pan and reset have keyboard controls. All 984 stored inputs are
reachable through the tree and its fragility toggle. Hover previews until a pin
holds the selection; filmstrip clicks swap it, and Release pin resumes previews.

Limits first: cursor terms cover the 4 local basic signal groups (SSIM,
added edges, lost detail and MSE): 120 raw slots and their 120 fragility twins.
The 3 global basic HF ratios have no local result. Other families say terms
are not exported. Native HDR pixel maps and real human-response panels remain
unmeasured/unsupported as described below.

The reference join has **36/36 rendered equations**: 35 family/signal entries
plus the fragility product. **22 entries have paper descriptions; 14 say
“no paper text yet”** (13 individual v2 signals plus the product). Both broad
families have paper descriptions. Bibliography author/year/title/URLs and
section source anchors come from read-only zenpapers commit
`8ac8a2b6d020282373977092627b34041693106b`. The current draft uses Typst and four
split bibliography files, without the suggested fenced LaTeX blocks. Equations
are explicitly code-derived, with source hashes; missing inspirations are not
invented. KaTeX 0.19.1 loads once from cdnjs. Every reference entry reaches a
stored row, and the browser visits all 984 virtual inputs (72 cells initially
in the DOM).

Native terms use plane-major little-endian f32, on exactly the map sampling
grid. Revision 7 exports the direct error moment rather than reconstructed
covariance. All **8,298,195 sampled SSIM formula replays** across the six pairs
match the retained detector bits. Every pair retains 504 populated-slot and
720 full-layout bit identities. All **2,088 detector PNGs** remain byte-identical
to review 1; every prior feature field also matches. All four weight documents
retain every original field, adding native per-input total |weight|. Six bank
audits and all existing curves/strict flags are reused unchanged. No tolerance,
severity threshold or frozen result was relaxed.

The additional native test compares term results with decoded detector PNGs
on all three odd-size fixtures and both tight/strided rows. Product ranges are
checked separately: a fragility input shares the raw normalized PNG and scales
its display range. An initial test applied that scaled range to the raw term;
the corrected join keeps both checks exact.

Browser acceptance passes at **1600×1000 and 1280×800, both themes and views**:
8/8 captures report document dimensions equal to viewport dimensions. The
1600×1000 stage is **1008×900**, above the required 880×700. Keyboard gutter
resizing reaches the 388px left-panel cap with the default right panel while
preserving the 55% stage minimum. Measured preloaded preview-to-canvas times
are **4.6, 4.0, 4.6, 4.8, 4.4, 4.2 ms** when rounded to one decimal; the exact
six measurements are in the receipt. Four cursor checks match native f32
bytes, including zoom/pan, scale 1 at the lower-right boundary, and scale 3.
Reference flags join stored layer/input IDs; the f20 check exposes all 33
applicable native audit flags. Review-1 paging, zero/MAD distinction,
minimap/zoom IDs, centered ICE, alpha and bank-scope checks also pass.

Screenshot inspection caught and fixed selector text inheriting a zero font
size and stale atlas image sizing after a hidden-pane resize. Both have browser
regressions. Original capture attempts remain under sibling archive directories;
the current eight images are in `design2-screenshots/`.

Validation receipts: native tests `rc=0 14s | peak-RSS 0.44GiB | min-avail
39882MiB | peak-load 2.10`; standalone strict clippy `rc=0 4s | peak-RSS
0.64GiB | min-avail 39968MiB | peak-load 1.07`; metric-library strict clippy
`rc=0 1s | peak-RSS 0.50GiB | min-avail 40168MiB | peak-load 1.07`.
Web syntax, numeric formatting, network HTTP and verbatim design-asset checks
pass. CI was inspected; no new Actions run was dispatched under repository
policy, so this is local verification rather than a CI-green claim.

Landed milestones: frame `bbb3d090519b`; all equations `950df2e647b6`;
native terms `53f83a0a850c`; studio interactions `bda091b81227`; viewport fixes
and capture source `eca9e4f88d48`. Full measurements, hashes and source identity
are in [design2-evidence.json](design2-evidence.json).

Open questions for the coordinator: **none block this implementation**. Paper
follow-up remains the 13 missing v2 detector descriptions and the fragility
product prose; the UI explicitly identifies each gap. The previous RGB8/HDR
and teacher-versus-human limits remain unchanged.

http://dev.lan:3310/

http://dev.lan:3300/zensim/viz/2026-11-04/design2-screenshots/index.html

http://dev.lan:3300/zensim/viz/2026-11-04/design2-screenshots/browser-review.json


## Coverage and limits

Both requested visualizers are implemented and exercised with native data (2/2).
Unmeasured or unsupported: native HDR pixel maps, alpha/16-bit PNG ingress,
real human-response range panels, and HDR curves for F seeds 1/2. The supplied
banks used here contain teacher labels; those are not presented as human scores.
These inspections do not constitute model-quality or production qualification.

Delivered coverage:

- Atlas: 6/6 content-varied training pairs, each with 984 declared model inputs.
  888/984 entries have spatial detector maps; 96/984 are explicitly labeled
  global statistics. All 504 populated pooled slots match the uninstrumented
  control bit for bit on every pair; all 720 layout positions, including unused
  positions, also match. The 492 damage inputs and 492 fragility
  products retain separate values, percentiles and model-input identities.
- Lens: 4/4 supplied bakes on all 12,163 CID22 TRAIN teacher rows. Each has
  529/529 curve bundles: 492 raw damage interventions, 35 signal groups derived
  from canonical registry names, and the 2 original broad families (`basic`, `v2`). E's additional F-only
  probe curves correctly show the response to unconsumed slots.
- HDR teacher TRAIN: F seed 0 and E on all 7,390 cube-root/203 rows, including
  drift audits, full ablations and 529/529 curves each. Both banks are available
  for atlas percentiles. Missing F1/F2 HDR evidence is explicitly unavailable.
- Every curve uses 200 recorded observed rows, 21 points and the native
  `BakeScorer` at fixed 512×512 scoring geometry. Raw-input axes use bank
  p1–p99. Family axes multiply each observed damage value from 0–2; they do
  not claim to be feature percentiles. Raw interventions jointly recompute
  the associated fragility products.
- Weight audits cover every stored weight, bias, scaler, declared input and
  hidden unit. Row/column modified z scores and zero-MAD cases are available
  per weight. Native layer recomposition must match the full Predictor bits
  on every audited row.

The source design directory remains byte-identical to `web/design/`.
Images, full JSON/TSV, original source paths/hashes, binary/source fingerprints,
commands and resource logs stay in the local output directory. The small
[evidence.json](evidence.json) preserves the measured census and receipts in git.

## The numerical finding

All four bakes pass the existing native structural monotonicity certificate.
F seeds 0/1/2 each have one certified max pair; E has three; all have zero free
ramps/amplitude. Negative entries in the paired encoder can be legal:
the audit checks the coupled constraints rather than imposing a blanket sign rule.
There are **zero sign/scaler violations** in every measured audit.

Structural certification does not guarantee strict floating-point monotonicity
of the served score. `curve-audit` independently rechecks every adjacent pair
in the persisted native ICE points and verifies every recorded flag. Each row
below covers 2,116,000 adjacent comparisons across 529 curves. The original
494 curves and their 2,095,548 saved values (axes, means and ICE points) remain
bit-identical in every one of the six curve banks.

| Bank | Bake | Strict rises | Largest observed score rise |
| --- | --- | ---: | ---: |
| CID22 TRAIN teacher | F seed 0 | 1,663 | 0.000012293133387686339 |
| CID22 TRAIN teacher | F seed 1 | 20,805 | 0.00001792646713738577 |
| CID22 TRAIN teacher | F seed 2 | 1,023 | 0.00001119673164851065 |
| CID22 TRAIN teacher | E | 1,985 | 0.00002825151364049816 |
| HDR teacher TRAIN | F seed 0 | 1,808 | 0.000008526020465637885 |
| HDR teacher TRAIN | E | 7,259 | 0.00001760564769881512 |

F seed 0's largest CID22 example is f594, physical bank row 6294, grid step 7:
x increases from 0.0007679860121570527 to 0.0008795049698092043;
served score increases from 57.29511144966739 to 57.295123742800776.
All rises remain visible as grouped notes, with zero tolerance. No metric arithmetic
or pre-existing expectation was changed. The numerical cause has not been
isolated; attributing it to a particular stage would require further work.
The finding is recorded in the project's Known Bugs.

## Weight-audit census

These counts use the **whole CID22 TRAIN teacher bank**, SHA-256
`07e2f4b71fe0ccc67848f2896344e6d0b6a255b0e91a7da9062d471ea3e11100`.
F has 126,080 weights (984→128→1); E has 105,088 (820→128→1).

| Bake | Row-MAD notes | Column-MAD notes | Bias notes | Near-zero outgoing rows | Inactive / flat hidden | Correlated input pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F seed 0 | 32,344 | 2,275 | 0 | 64 | 95 / 95 | 54 |
| F seed 1 | 18,913 | 11,458 | 1 | 152 | 95 / 95 | 54 |
| F seed 2 | 2,511 | 929 | 1 | 126 | 125 / 125 | 54 |
| E | 20,564 | 3,375 | 0 | 122 | 93 / 93 | 46 |

There are zero constant inputs in this bank for every bake. The new exact-zero
`dead-input` finding identifies 2 incoming rows in F1 and none in F0, F2 or E.
It is separate from the retained 1e-6 near-zero findings. F0 input 371 has
33 nonzero weights out of 128; its zero MAD does not mean the row is dead. F0's 64 and E's 122
near-zero outgoing rows are all hidden→output rows, not dead incoming feature
rows. F1 has 26 incoming rows plus 126 hidden→output rows; F2 has 2 plus 124.
Inactive/flat are two labels on the same observed ReLU behavior, not disjoint
populations. They do not establish deadness on the complete training mixture.

Modified z uses `0.6744897501960817 × (weight − median) / MAD`, threshold 3.5.
A zero MAD has no finite z score: nonzero deviations above 1e-6 are notes with
null statistics. Near-zero uses 1e-6; correlation flags use |Pearson| > 0.999.
These are diagnostics rather than evidence of incorrect coefficients.
The original F/E census files remain pinned, and the full F regression passes.

The HDR teacher TRAIN bank has SHA-256
`742fd8f314dcc5d1f812faa9924ac8b3fb048ba07dbc5f53a0b7f2b984857101`.
Against the full CID22 training p1–p99 ranges, F0 has 842 derived-input drift
flags and 8 constant inputs; E has 700 drift flags and 6 constant inputs.
A drift flag means at least one new-bank value lies outside that training
interval. It is descriptive tail exposure, not a significance or quality test.

## Atlas data and implementation

The six pairs cover architecture (JPEG q20), printed music (q5), forest texture
(q10), a weathered facade (q40), foliage (q60), and flowers/leaves (q20).
They use the recorded training codec `mozjpeg-rs-420-e4-v0.5.4`.
Each bundle records original file paths and SHA-256s; no source image is in git.

The private default-off `viz-export` feature taps the existing retained fold
planes. Shared detector helpers produce the display maps; the original fold
owns every pooled number. Weighted maps identify their numerator role; global
moment/scale ratios have no fabricated spatial map. Fragility maps multiply the
damage detector by the actual pooled reference factor. Display maps crop metric
padding, retain their native grid when small, and nearest-subsample to at most
512 pixels. Metadata distinguishes the display grid from the pooling grid.
8-bit maps retain their numeric ranges; they are visual summaries, not lossless
reconstruction data. Imazen PNG/JPEG codecs own all ingress and PNG output. All six references are
8-bit RGB PNG (IHDR depth 8/type 2); the current ingress refuses higher
precision rather than downconverting it. The retained earlier binaries and
metadata describe their own ingress behavior; no downconversion occurred in
these six 8-bit examples.

The browser provides hover preview, shared-range pins, family small multiples,
stable matrix positions under filtering, scales/channels/pooling/fragility,
percentile-bank selection, signed color, image flip, synchronized image/overlay
zoom and pan, map-only mode, keyboard navigation and numeric tables.
The lens provides sortable concerns, individual weight navigation, architectural
path, hidden-unit measurements, PDP/ICE, quantile separation, complete family and
raw-input ablation rankings, E/F overlays and weight/audit drift comparisons.
Neuron indices are not aligned across independently trained bakes; row norms and
stored-index drift are labeled accordingly. Ablations overlap and are not additive.

## Verification and retained evidence

- Real-bake release tests pass: pinned F census, native raw and family ICE oracle,
  legal/adversarial paired-weight constraints, zero-MAD/quantile behavior, and
  admission refusal before prohibited table I/O. Atlas parity covers 65×67,
  79×53 and 129×97 images, with tight rows and five pixels of stride padding.
- Strict standalone all-target clippy and library clippy with `viz-export` pass.
  Formatting, number formatting, JavaScript parsing and exact design-copy checks pass.
- Real Chromium checks pass for populated audits/maps/curves, themes, pinning,
  keyboard movement, family lanes, comparison, both banks, missing-bank states,
  and contained scrolling of expanded numeric tables. No uncaught page errors.
- The latest captured preloaded hover measurements are
  `[3.6000003814697266,3.299999713897705,3.4000000953674316,3.5999999046325684,3,3.200000286102295,3.4000000953674316,5.199999809265137]` ms.
  Each of these eight actual browser measurements is below 50 ms; this does not
  measure a cold network fetch or all devices.
- Dark/light atlas and lens screenshots are under `review1-screenshots/` in the demo
  directory. Earlier captures and failed attempts are preserved.

All native heavy work ran serially behind the shared lock with a 16G cap.
Representative receipts (full logs and every job's receipt are retained):

```text
F audit: run-heavy: done rc=0 11s | peak-RSS 0.29GiB | min-avail 38622MiB | peak-load 7.61
E audit: run-heavy: done rc=0 7s | peak-RSS 0.25GiB | min-avail 38835MiB | peak-load 7.92
F curve bank: run-heavy: done rc=0 68s | peak-RSS 0.14GiB | min-avail 38716MiB | peak-load 7.81
Real-bake tests: run-heavy: done rc=0 32s | peak-RSS 0.70GiB | min-avail 38389MiB | peak-load 8.52
Library lint: run-heavy: done rc=0 3s | peak-RSS 0.66GiB | min-avail 39289MiB | peak-load 1.96
```

The measured bundles use frozen binary
`99a7f3c03e16711273a0613a00924647e83933329b471c9699e2128cd49d10dc`,
preserved locally under `review1-logs/native/`. Each export records its compiled source and
lock hashes; later guard, formatting and inspection changes do not relabel those
artifacts as newly generated. Curve audits have their own binary fingerprints.
No AIC table was read or published. Manual CI was not dispatched; the latest
listed automatic runs are unrelated dependency/site jobs.

## Design review 1 — 10/10 implemented

The reviewed interface and acceptance checks landed in `25e8a80e1b0c`
(`feat: apply the ten visualizer design review decisions`), verified on main.

1. Rises use note <0.001, warning <0.05, serious <0.5, critical ≥0.5.
   Structural findings stay critical. All currently measured rises are notes.
   Concerns group identical input/kind/reason with count and worst witness;
   severity/magnitude ordering and 50-row paging retain the full findings.
2. Exact all-zero incoming rows emit critical `dead-input`; their readout says
   all 128 weights are zero. Zero MAD alone is not classified as deadness.
3. Native interquartile bands use all 200 curves at 15% fill; at most 40 ICE
   traces are drawn. Centered ICE subtracts each curve's first native value.
   The y extent uses the band/mean, with the comparison mean; the legend sits
   above the plot, and every original value remains in the table/export.
4. Rows follow native family/signal/ID order; columns use native total |weight|
   or selected-input magnitude order. A 1:1 minimap accompanies a 24×16 zoom
   window. Named signed strips show 32 inputs for a unit and 16 units for an
   input. Every selection maps back to the stored input/unit ID.
5. Human responses use a separate aligned panel in native units (confirmed Q2),
   series-2 points/whiskers and a half-bin offset. No real human bank was
   admitted here; that panel remains unmeasured on real responses.
6. Family attribution ranks 35 registry-name signal groups; top-input labels
   include short names and IDs. Original broad-family curves remain exported.
7. Overlay alpha is zero at the low end and rises with map value, multiplied
   by the selected overlay opacity. Map-only remains opaque. Legends include
   pooled values and pooling grids.
8. Percentile color is the default. Diverging value encoding is applied only
   to features whose selected native bank range crosses zero; one-sided
   features retain sequential percentile encoding.
9. The 14px marks/2px gaps remain, with 18px hover padding and nearest-mark
   pointer resolution, keyboard and table access. Global scalars occupy one
   hollow group affordance per signal/scale row in the global column; channel
   filters, tooltips and the table retain their individual values.
10. The empty pin strip stays visible with “pin a feature: Enter”.

All six audit baselines preserve every original flag's numeric fields. F1 adds
only its 2 exact-zero findings; other audit flag counts are unchanged. Each of
six curve baselines preserves all 494 original curves and 2,095,548 numeric
values, including every original strict rise. Added signal-family curves explain
additional rises in the expanded 529-curve census; the model did not change.

The populated review browser checks pass for grouped paging, note severity,
all-zero versus zero-MAD behavior, native IQR/centered data, minimap/zoom identity,
rank labels, overlay/raw alpha, signed-range gating, global placement, empty pins,
both themes and missing-bank states. Visual screenshot review also caught and
corrected an opaque SVG hit target over the signed strips. Source design assets
remain byte-identical. Q1–Q3 are resolved by the supplied review.

## Local preview

http://dev.lan:3310/

http://dev.lan:3300/zensim/viz/2026-11-04/review1-screenshots/atlas-dark.png

http://dev.lan:3300/zensim/viz/2026-11-04/review1-screenshots/mlp-lens-dark.png

http://dev.lan:3300/zensim/viz/2026-11-04/review1-screenshots/atlas-light.png

http://dev.lan:3300/zensim/viz/2026-11-04/review1-screenshots/mlp-lens-light.png
