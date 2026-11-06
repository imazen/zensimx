# Monotone architecture comparison, 2026-11-06

## Completed coverage

24/24 short fits; eight registered architectures and three independent seed pairs.
All train for 12 epochs x 5,000 draws and use fixed final epoch 11. These are
short development results, not full-budget selection or a shipping qualification.
The registered 120 x 50,000 comparison remains separate. T1--T7 remain open.

## Short-screen results

Three-seed mean pooled signed SROCC; higher is better. KonFiG development
contains one source, so its changes do not establish broad source generalization.

| Architecture | KADID | TID2013 | KonFiG F | CID22 human |
|---|---:|---:|---:|---:|
| shared_smooth | 0.895592 | 0.901077 | 0.938645 | 0.903337 |
| shared_relu | 0.895497 | 0.901726 | 0.938537 | 0.903020 |
| shared_caps | 0.887953 | 0.895004 | 0.920948 | 0.902739 |
| shared_switch | 0.891662 | 0.898529 | 0.931356 | 0.902816 |
| shared_switch_deep | 0.890882 | 0.898168 | 0.929756 | 0.902539 |
| shared_spline | 0.889983 | 0.896720 | 0.923418 | 0.902422 |
| shared_lipschitz | 0.890988 | 0.895636 | 0.923087 | 0.902464 |
| single_ramps | 0.888379 | 0.894212 | 0.918720 | 0.903966 |

At this horizon, none of the new shared heads beats shared ReLU on all four
populations. The single-layer ramp control has the highest mean CID22-human
rank, with lower means on the other three populations. This does not settle
long-run convergence, and no arm is removed from the registered long comparison.
Per-seed ranges and within-source means are in the adjacent compact JSON.

## Correctness and provenance

Native finite-difference gradients, nonconvex monotone target fitting, native
f32 zero identity, coordinate order and training/bake agreement pass. Existing
trainer regression expectations remain unchanged. All 24 final bakes return
exactly 100 on the zero feature vector; each has 128 coordinate-order pairs.
Across those probes, 1 positive score deltas were observed, and none exceeded 1e-5. No score exceeded 100.
These are feature-space diagnostics, not a pixel-domain near-lossless gate.

All three frozen compatibility replays (single, shared ReLU, shared SmoothHinge)
match byte-for-byte after removing only run provenance metadata. Source audit:
24,852 rows, 232 fitting sources, 21 development sources, zero exact decoded
reference overlap. The F-only prepared KonFiG data already excludes Part B.

Native source content: `105ef30b44e8d6015f7b3bc61c419f2e56534b9a`.
Trainer SHA-256: `48f539f80a675ee653a168daa4efe8f3fb3169c2fea3db5ff330f9f91fc8cf99`.
The artifact directory contains input pins, per-cell commands, manifests,
checkpoints, native predictions/panels/scatter, probes and full logs.

## Reproduction and resources

Use `just monotone-architecture-screen`, `just monotone-architecture-long` and
`just monotone-architecture-replay` with the admitted plan and frozen binary
directory. The driver rejects input drift and existing output directories.
The screen ran six concurrent single-threaded children, each under a 2 GiB cap.
For the long comparison, the concurrency-only amendment permits twelve children
under a 26 GiB outer cap; 44 GiB available RAM and no active saturated workload
were observed before launch. No scientific settings change with concurrency.

Screen wrapper: `run-heavy: done rc=0 168s | peak-RSS 0.46GiB | min-avail 44192MiB | peak-load 6.50`.
Replay wrapper: `run-heavy: done rc=0 10s | peak-RSS 0.16GiB | min-avail 45171MiB | peak-load 6.30`.
These are resource observations, not aggregate process-tree memory or speed benchmarks.

Artifacts observed on dev: `/mnt/v/output/zensim/monotone-architectures-20261106/`.
The completed stage-1 snapshot has a verified mirror, detailed below. The
ongoing full-budget/training runs require a separate final closure.

## Native capacity census

Native inspection of seed-0 final screen bakes reports stored coefficients
(including frozen routing/expanded weights) and actual compressed bake bytes:

| Arm | Stored coefficients | Bake bytes | Expanded dense layers |
|---|---:|---:|---:|
| shared_relu | 134401 | 148431 | 3 |
| shared_smooth | 134401 | 126398 | 3 |
| shared_caps | 159201 | 216299 | 5 |
| shared_switch | 159201 | 216087 | 5 |
| shared_switch_deep | 165409 | 238627 | 7 |
| shared_spline | 200417 | 194100 | 5 |
| shared_lipschitz | 196355 | 226017 | 6 |
| single_ramps | 126209 | 247118 | 2 |

These are not independent trainable-parameter counts or runtime benchmarks.
Frozen routing makes the expanded representations larger; compression and
zero coefficients make byte size differ from dense coefficient count. Exact
layer dimensions and coefficient sign/zero counts are in `CAPACITY.json`.

Input closure now contains 71 indexed files / 69 unique SHA-256 blobs,
including the admitted prepared tables, sidecars, source-key metadata, baseline
plan and its dependencies. Index SHA-256:
`30a907d2c514d3dc57b2cdad5c3131c878eb3763be128f593f4c69201dfbd981`.

Nominal independent trainable coefficient counts, computed from all three
seeds' actual native layer dimensions and enforced parameter ties:
shared ReLU / SmoothHinge / switch 132,320; caps 132,352; spline 132,640;
Lipschitz residual 132,321; two-block switch 134,368; single ramps 110,384.
Scaler constants, fixed routing, frozen max-pair biases and unused cap
thresholds are excluded. Zero weights at a constraint boundary remain trainable
parameters in this count; it is not a measure of effective learned capacity.
`PARAMETERS.json` preserves the calculation inputs and all model dimensions.

Completed stage-1 closure is mirrored at
`/mnt/tower/output/zensim/monotone-architectures-20261106-stage1/`:
2,330 artifact files, 426,717,327 bytes, every SHA-256 verified by cold client
readback. Receipt SHA-256:
`fcd150cbe25627073b8a3153498956c1aa8359005b75edcb22b5e777d7832397`.
This snapshot excludes the ongoing full-budget/training runs.

## Signed source and distortion breakdowns

The 24 short fits now have canonical native signed ranks for every source,
distortion family and source/family combination. The older `--per-group`
panel returns absolute correlation; it is retained as historical output but
is not the authority for signed subgroup claims. No model or checkpoint was
selected from these breakdowns. All subgroup correlations are defined in this
screen; subsequent studies must report undefined cells explicitly.

All 24 short models reverse the pooled ordering of TID2013 `tid_15` in
development (15 rows, three sources). The three-seed mean is -0.347619 for
shared ReLU, -0.328571 for one switch block and -0.292857 for two blocks.
The existing taxonomy names this local block-wise distortion. The small
panel and already-exposed development set do not support selecting a fix;
they show why a pooled population rank near 0.90 is insufficient evidence.
Within-source results are mixed: ReLU seed 0 gives 0.7, -0.7 and 0.7 on the
three five-row sources. Thus the negative pooled rank does not mean every
source reverses, and cross-source calibration is a separate concern.
Family weighting was registered before this breakdown was computed.

`screen-breakdowns/INDEX.json` SHA-256:
`e3d7d9876243e3f77765c2515a89419bd0bd22f033e3da6efbb6d9815e4730a5`.
`screen-report/SUMMARY.json` SHA-256:
`0be671b541b4b0221248dcdee4a058fe527dbd4243ef46bf5bf21e2ded128417`.
The report retains all nine fitting/development populations, seed ranges,
signed subgroup means/minima, calibrated/raw correlations and native scatter
tails. Seed ranges are descriptive, not confidence intervals. Reproduce with
`just monotone-family-panels` followed by `just monotone-study-summary`.

The leaky-capable trainer's neutral configuration reproduces all three earlier
switch fits exactly after removing run metadata and the old coverage entry.
The nonzero .05/.1 slope trials use the same 120 x 5,000 budget as T1--T7;
the six runs are separate from the ten-times-larger architecture comparison.

The completed 2,000-draw source-cluster bootstrap conditions on all three
fixed short fits. Against shared ReLU, one switch block has KADID mean signed
rank delta -0.003835, percentile interval [-0.005505, -0.001949]; CID22 human
delta -0.000204, interval [-0.004031, 0.003818]. SmoothHinge intervals overlap
zero on all three populations with multiple sources. ReLU self-comparisons
are exactly zero in every replicate. These are exposed-development diagnostics
without multiplicity correction, not a reason to drop long-budget arms.
There are 10 KADID, three TID and seven CID22 development sources; no
new-source interval is estimable for the single KonFiG development source.
The bootstrap input report adds exact prediction-table pins while replaying
every earlier measured summary value unchanged (`screen-report-v2`).

`source/DRIVERS.json` resolves all eight registered stages to preserved driver
bytes matching their original SHA-256 pins. Future launches copy those bytes
into their own run directory before writing the plan. The signed/diagnostic
summary also passes a real three-cell neutral replay with complete epoch and
event receipts. The final script scan covers 1,582 runnable scripts.
