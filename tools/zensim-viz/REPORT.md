# zensim-viz report — 2026-11-04

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
