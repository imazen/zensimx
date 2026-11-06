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
Mirror verification is pending; the existing source data closure has its own
verified mirror in the earlier tree and activation studies.
