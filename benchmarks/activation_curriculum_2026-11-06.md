# Activation and training-order study evidence — 2026-11-06

## Scope and missing work

New analytic activations have **not** been fitted or baked in Rust. Family curricula and dropout training are not implemented by this study. General anchoring needs the native forward/backward and bake/runtime work specified in the [protocol](../docs/MONOTONE_ACTIVATION_CURRICULUM_STUDY.md); family schedules additionally need a complete distortion metadata join.

Completed: 18/18 short native optimization fits; all four development populations per fit have signed-quality, within-reference and scatter panels, zero dropped rows, and measured scatter. Eight scalar positive controls and two negative controls have independent numerical probes. This is development evidence, not qualification or independent final-test accuracy.

## Native pilot

12 epochs x 5,000 draws; fixed final epoch 11; shared 128->64 ReLU; unchanged admitted four-human plus CID22-only SSIM2 inputs. Three diagnostic seed pairs separate one init change from one sampling-stream change. They are not three independent replicates. No comparison to a 120-epoch result is valid.

Pooled SROCC, arithmetic mean across the three diagnostic cells (no significance claim):

| Arm | KADID | TID2013 | KonFiG F | CID22-A |
|---|---:|---:|---:|---:|
| control | 0.894991 | 0.900588 | 0.934589 | 0.901678 |
| walk | 0.891235 | 0.900823 | 0.940056 | 0.901375 |
| lr_low | 0.894687 | 0.900823 | 0.936890 | 0.902513 |
| l2_high | 0.895272 | 0.900449 | 0.933738 | 0.901918 |
| teacher_half | 0.892398 | 0.899618 | 0.933637 | 0.899302 |
| teacher_double | 0.893033 | 0.899079 | 0.931329 | 0.902416 |

Per-cell metrics, including within-reference SROCC, are in [the native summary](activation_curriculum_2026-11-06.native.json).

Matched observations at this short horizon:

- Lower LR (3e-4 versus 1e-3) increases CID22-A pooled SROCC in all three pairs and decreases KADID in all three.
- Halving teacher mass decreases CID22-A pooled SROCC in all three pairs. Doubling teacher mass increases CID22-A and decreases TID2013 in all three pairs.
- Coverage walk and higher L2 have mixed population/seed effects. None is established as a universal improvement.
- These fits do not cross the baseline 50-epoch cosine restart. They cannot answer whether restarts, late consolidation or dropout help long-run generalization.
- KonFiG development has limited source diversity. Its apparent changes are not broad new-source evidence.

## Numerical construction evidence

The [scalar audit](activation_curriculum_2026-11-06.analytic.json) checks ReLU, SmoothHinge, anchored leaky/softplus/Gaussian/rational/linear+tanh and a positive-slope nonconvex spline. Each has 8,193 grid points, no decreasing f32-rounded values, identity checks and finite-difference gradients in input and bias. Ordinary SiLU and GELU produce the expected decreases despite having an identity anchor. This uses f64 scalar formulas rounded to f32; it is **not** native f32 implementation qualification.

Five focused construction tests cover the missing-anchor-gradient failure, an exhaustive last-hidden dropout mask expectation/identity/order check, nonconvex monotonicity, extreme scalar inputs, and the SiLU/GELU negative controls. The first-layer independent-mask counterexample decreases from 1 to 0 as one feature increases; tied masks retain the certified response. Four synthetic pair schedules retain identical event multisets and a common 25% tail. No real distortion curriculum was executed.

## Reproduction and resources

Source of native pilot orchestration: `cb502a81f02b`; pinned Rust training source: `698db9bd43c3`. Analytic audit also records its exact source hash. Scripts invoke the existing strict group admission, trainer, bake/prediction and panel owners; no new metric scorer or statistics implementation.

Commands: `just activation-curriculum-tests`; `just activation-curriculum-audit <fresh-json>`; `just activation-curriculum-pilot <admitted-baseline-plan> <fresh-output>`. The pilot refuses changed input pins and existing output directories. Raw inputs/labels stay outside git.

Native wrapper: `run-heavy: done rc=0 179s | peak-RSS 0.26GiB | min-avail 42515MiB | peak-load 3.85`. Three fit workers ran concurrently in an 8 GiB hard scope with one thread per child. This is a resource observation, not a benchmark or aggregate process-tree memory claim.

Local artifacts checked on dev: `/mnt/v/output/zensim/activation-curriculum-20261106/`. Mirror verification is recorded below after preservation.

Mirror: `/mnt/tower/output/zensim/activation-curriculum-20261106/`; 641 files, 128,106,604 bytes, all remotely read and SHA-256 verified. Includes the admitted prepared tables, key/manifest sidecars, role decision, three pinned binaries, source snapshots, native model/prediction/panel outputs and logs. Manifest SHA-256: `e938ae8e6c7326ac416b13a8f251d17776f9a13541f9b44cf5933509af9105bc`.

The first rsync attempt copied correct bytes but returned code 23 because network storage refused ownership metadata. The attempt log is preserved. Retrying content/timestamps without ownership changes succeeded; every file passed verification.
