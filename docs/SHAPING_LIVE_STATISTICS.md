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

