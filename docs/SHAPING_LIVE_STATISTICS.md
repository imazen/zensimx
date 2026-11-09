# Live shaping-study statistics

The gallery study's `stats/index.html` compares registered approaches, paired
seeds, learning curves and pipeline coverage. Clicking an approach opens scalar
scatterplots and source/distortion panels for individual completed fits. Human
populations and the SSIM2 teacher remain separate.

`shaping_scatter_prepare.py` publishes only completed inner-study responses. It
verifies the frozen labels and membership, report completion, and prediction
identity. Historical reports without an inline NPZ hash use their existing
archive manifest, binding both the original report and prediction table; their
original reports remain unchanged. Newer reports use the inline prediction pin.
Outputs have a 2-GiB budget. The first real publication produced 819 reports and
851,583,391 bytes, with no pending provenance records. Wrapper accounting:

```text
run-heavy: done rc=0 27s | peak-RSS 0.25GiB | min-avail 38282MiB | peak-load 0.86
```

These are display-export resource observations, not a model-speed comparison.

`shaping_scatter_native.py` adds the canonical native Zenstats panel from those
stored scalar pairs on approved workers, through shared CPU/RAM admission. Each
process uses one native thread. It computes the full statistical panel, signed
quality correlations and scatter diagnostics through separate supported native
CLI modes. The signed-quality and scatter flags cannot be combined in this
frozen CLI. Missing native results remain pending; undefined saved correlations
remain undefined. This display does not change model selection or training.

The viewer shows unclipped scores beside the native normalized geometry. Its
human coordinate uses the existing TRAIN-only population range mapping; no map
is fitted to held-apart responses. Source and distortion filters affect plotted
rows only; full-population statistical denominators remain explicit. The CSV
download retains raw labels, TRAIN-scaled labels, predictions and row identities.
Pixel diffmaps and image extraction are not part of this workflow.

The native publication completed 639 current-comparison fits: 320 on one worker
and 319 on another, with zero failed fits. Imported panels bind their prediction
input, report and output hashes. The earlier 180 raw-coordinate pretrained
responses remain preserved as historical evidence outside the corrected primary
comparison. They are not silently treated as corrected fits.

The first real neural and tree panel jobs completed with wrapper observations:

```text
run-heavy: done rc=0 4s | peak-RSS 0.05GiB | min-avail 12859MiB | peak-load 1.90
run-heavy: done rc=0 3s | peak-RSS 0.05GiB | min-avail 40067MiB | peak-load 4.17
```

These measurements describe scalar statistics jobs, not model inference speed.
The committed Playwright browser check exercised all five populations, source
filtering, training and validation views, native panel visibility and mobile
layout. It passed with no page errors or mobile overflow. Runtime evidence lives
beside the generated page under `zenstats-dashboard-20261108/`.

## Conditional exposure curves

The longer-exposure selector displays only registered extension cohorts, using
their selected settings and optimizer choices. All registered seeds remain
required for each aggregate. Training-phase lengths and total pair draws are
shown in fit details; fresh-head controls, phase budgets and optimizer choices
are distinct cohorts. Curves show only saved evaluated checkpoints from the
final phase. They do not authorize retuning or checkpoint selection.

The snapshot retains original-budget `records` and `comparisons` unchanged for
existing scatter consumers. Extensions use separate `extension_records` and
`extension_comparisons`. Their cell identities come from the audit's already
validated whole-fold registrations, without another per-cell resolution pass.
No additional training, inference or pixel maps are needed. Extension scalar
scatters remain outside the original-budget scatter publisher's scope.

Qualification compared all 1,608 original records' saved panels, curves, failure
flags and reasons unchanged, while adding all 60 registered extensions in 20
three-seed cohorts. The actual metadata snapshot took 27 seconds with
`peak-RSS 0.12GiB | min-avail 43450MiB | peak-load 3.02`. Twenty-two focused
checks pass on i270. The browser witness renders extension curves and then
checks the original five-population scatters, source filtering, detailed native
panels and mobile layout: three seconds, `peak-RSS 0.12GiB | min-avail 43452MiB |
peak-load 3.14`. These are metadata/UI checks, not new scientific evaluations.

The deployed publisher runs the committed `d91a8abe` source. A separate worker
verified HTTP 200 for its page and 5,251,276-byte JSON snapshot containing all 60
extensions and 20 cohorts. The closed qualification/activation evidence passed
durable cold checksum readback as 44 payload files, 6,623,564 bytes, plus its
receipt, under `extension-curves-dashboard-evidence-20261109` beside the study.
The archive's evaluation-artifact mode permits the browser witness PNGs; the
initial feature-artifact mode refused before any destination payload was created.
Its refusal log is retained. Existing native point assets are reused in place.

