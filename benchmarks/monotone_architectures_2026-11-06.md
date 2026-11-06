# Monotone architecture comparison, 2026-11-06

## Completed coverage

129/129 registered native fitting runs reached their fixed final epoch:
24 short architecture fits, 24 long architecture fits, 66 training controls,
six leaky-switch fits and nine isolated-order fits. **123/129 original final
bakes pass the feature gate; six long spline/ramp exports fail.** All failures
are preserved. Six separate spline re-exports pass, without additional fitting.

All 15 checkpoint-retention replays reproduce final model bytes and logged
trajectories, retaining 1,800 checkpoints. Neutral compatibility replays also
pass. This is a completed development experiment matrix, not a shipping
qualification. Exact smooth-activation operators and ramp-export repair remain
explicitly unimplemented; see the experiment queue's verified runtime limits.

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

## T8 completed: leaky activation switches

Six of six fits completed at 120 x 5,000 draws, fixed final epoch 119. Their
exact native event streams, per-population counts and consolidation tails
match the corresponding neutral control seeds. All six feature probes pass;
all six final event-receipt copies preserve model sections byte-for-byte.

Three-seed mean signed development SROCC:

| Switch slope | KADID | TID2013 | KonFiG F | CID22 human |
|---|---:|---:|---:|---:|
| 0 (control) | 0.893218 | 0.898381 | 0.951834 | 0.902551 |
| 0.05 | 0.894719 | 0.899908 | 0.947940 | 0.902880 |
| 0.10 | 0.894807 | 0.901067 | 0.948642 | 0.902302 |

Every leaky fit has all 64 last-hidden units nonzero in the final response
sample, versus 53--57 inactive units in the neutral controls. That is not
64 effective learned directions: slope .05 has zero output amplitudes
[0,0,34] across seeds, and slope .10 has [0,0,35]. The slope floor addresses
the sampled activation inactivity, but neither slope wins all populations.
KADID/TID improve in the mean; the one-source KonFiG result declines, and
CID22 changes are small and mixed. Source-bootstrap intervals will accompany
the complete control-matrix report; no slope is selected from development.

Native summary SHA-256:
`53db2b7907dac4c6932614eaed19289102e79fea530b0576856b3534ede9d497`.
Artifacts: `leaky-report`, `leaky-breakdowns`, `leaky-qualified`.
Wrapper: `run-heavy: done rc=0 1415s | peak-RSS 0.39GiB | min-avail 39102MiB | peak-load 22.39`.
This is a resource observation under concurrent training, not a speed benchmark
or an aggregate process-tree memory measurement.

## Completed T1–T7 training comparison

66/66 fits completed: 22 arms, three fixed seed pairs, 120 x 5,000 draws,
fixed epoch 119. These use one tenth of the architecture-long draw budget.
Every final feature gate passes; all 66 event-receipt corrections preserve
the model bytes after stripping only the appended event metadata.

| Arm | KADID | TID2013 | KonFiG F | CID22 human |
|---|---:|---:|---:|---:|
| control | 0.893218 | 0.898381 | 0.951834 | 0.902551 |
| init_half | 0.891374 | 0.894913 | 0.938639 | 0.902657 |
| init_double | 0.894931 | 0.900592 | 0.950424 | 0.902199 |
| bias_zero | 0.894584 | 0.899273 | 0.949438 | 0.902552 |
| bias_double | 0.893532 | 0.898764 | 0.951915 | 0.902681 |
| l2_none | 0.893993 | 0.899227 | 0.952880 | 0.902344 |
| l2_high | 0.892877 | 0.898875 | 0.947771 | 0.902500 |
| adamw_low | 0.894291 | 0.899028 | 0.953535 | 0.902309 |
| adamw_high | 0.894089 | 0.899069 | 0.952651 | 0.902175 |
| dropout_05 | 0.893661 | 0.899281 | 0.948041 | 0.902535 |
| dropout_15 | 0.893891 | 0.898605 | 0.948054 | 0.901080 |
| easy_first | 0.886864 | 0.895136 | 0.920691 | 0.896742 |
| hard_first | 0.889352 | 0.897248 | 0.947312 | 0.896746 |
| family_blocks | 0.888910 | 0.895415 | 0.925882 | 0.903807 |
| teacher_warm | 0.888465 | 0.893521 | 0.956093 | 0.895672 |
| family_balance | 0.894652 | 0.898715 | 0.951193 | 0.902579 |
| source_family_balance | 0.893734 | 0.899362 | 0.947582 | 0.902848 |
| teacher_zero | 0.901648 | 0.916268 | 0.964497 | 0.889162 |
| teacher_half | 0.897289 | 0.898045 | 0.954183 | 0.902284 |
| teacher_double | 0.892959 | 0.902252 | 0.949114 | 0.901923 |
| lr_constant | 0.892937 | 0.898388 | 0.953690 | 0.902534 |
| lr_consolidate | 0.895487 | 0.897665 | 0.955796 | 0.901477 |

Doubling initial bias spread raises all four means by 0.000081–0.000383;
three of its four panels do not establish a source-level improvement (KonFiG
has no estimable interval). This is a small descriptive gain, not a selected
setting. Removing teacher labels gives the largest
TID2013 change, with a CID22-human trade-off. This arm keeps the same scaler
membership, including teacher-feature rows; it isolates supervision removal
and is not a strict human-data-only preprocessing recipe. Halving teacher mass
raises KADID mean while leaving a small CID22-human decrease.

Dropout .05 and .15 give mixed changes and no four-population advantage.
Decoupled AdamW improves KADID/TID/KonFiG means at its low setting, with a small
CID22-human decrease. Greater initialization scale and zero initial bias
also merit separate TRAIN-fold follow-up; these development results do not
authorize selecting or combining settings.

Global easy/hard, family-block and teacher-warm schedules mostly lose quality
at this fixed budget. Their matched event multisets and final-quarter tails
are verified; T9 separately isolates within-population order from timing.

Paired source-bootstrap examples (2,000 draws, same three fits; no multiplicity
correction):
- teacher_zero, human_tid2013_development: delta 0.017887, 95% interval [0.009890, 0.026557].
- teacher_half, human_kadid_development: delta 0.004071, 95% interval [0.001916, 0.006314].
- family_balance, human_kadid_development: delta 0.001434, 95% interval [0.000577, 0.002372].
- family_blocks, human_kadid_development: delta -0.004307, 95% interval [-0.006914, -0.001990].

No training-control CID22-human interval excludes zero. TID2013 has only
three development sources and KonFiG has one; no KonFiG source interval is
estimable. Leaky-switch intervals also cross zero on all three multi-source
populations. Seed ranges are descriptive, not confidence intervals.

Native event audits confirm zero teacher pair events in all three zero-mass
fits, and identical original, ordered and tail digests for loss-only controls.
Source/family balancing changes the number of weighting cells, not proposed
pair counts; the report compares group/pair counts separately from that
taxonomy count. The first report attempt refused the mismatched taxonomy
comparison before producing a summary, and its numerical inputs were preserved.

Artifacts: `training-report/SUMMARY.json`, `training-bootstrap/SOURCE_BOOTSTRAP.json`,
`training-breakdowns/INDEX.json`, and `training-qualified/` under the study root.
Training summary SHA-256: `67d4c551b6b46afb6de093cdc4c207f78213e09550e21c0e45a0000a2d5a5b13`.
Bootstrap SHA-256: `9f899e339e3d4d2f168c9b1a89e9fc95c8c3f60ee95bc56d942aea55c810adfa`.

Training wrapper: `run-heavy: done rc=0 4485s | peak-RSS 0.55GiB | min-avail 36820MiB | peak-load 23.86`.
This is a resource observation, not aggregate process-tree memory or a speed benchmark.

## Exact checkpoint replay and TRAIN trajectories

15/15 replays match their original final model bytes, every logged epoch
trajectory and the native event receipt. All 1,800 epoch checkpoints are
retained. Additional panels use TRAIN rows only at epochs 0, 29, 59, 89 and
119; no checkpoint is selected. The five schedules all complete.

| Schedule | KADID 89 → 119 | TID2013 89 → 119 | KonFiG F 89 → 119 | CID22 human 89 → 119 |
|---|---:|---:|---:|---:|
| control | 0.893265 → 0.890360 | 0.876921 → 0.879592 | 0.896252 → 0.890867 | 0.863766 → 0.862510 |
| easy_first | 0.891260 → 0.886080 | 0.856675 → 0.863597 | 0.847755 → 0.878312 | 0.861396 → 0.865853 |
| hard_first | 0.891964 → 0.888793 | 0.874288 → 0.871542 | 0.897689 → 0.890107 | 0.866735 → 0.867554 |
| family_blocks | 0.865683 → 0.886789 | 0.859949 → 0.870757 | 0.865680 → 0.871572 | 0.863605 → 0.863889 |
| teacher_warm | 0.896362 → 0.887819 | 0.870231 → 0.871092 | 0.903440 → 0.889171 | 0.868345 → 0.868516 |

The shared random tail can recover fit lost earlier in a schedule, but does
not erase every difference. For example, easy-first KonFiG rises during the
last quarter, while remaining below the random control. Family-block KADID
recovers strongly over the same tail. These observed training changes are
consistent with interference and recovery, not proof of a local minimum.

All TRAIN family changes and per-seed population values are retained in the
plot receipt; native panels are in `stages/*/STAGES.json` and `PANELS.parquet`.
Stage summary SHA-256: `ef155a4b5406cf14888c7d46112d0dbaac5a31ee012ff6a74e7b15fdec354027`.

## T9: ordering with population and teacher positions fixed

9/9 fits and corrected receipts complete. Every native 600,000-event stream
preserves population slots, teacher events, skips, replay slots and the final
quarter. The prefix assertion checks all 450,000 positions; eligible human
pair order changes. All nine final feature gates pass.

| Order | KADID | TID2013 | KonFiG F | CID22 human |
|---|---:|---:|---:|---:|
| within_population_easy | 0.893390 | 0.897728 | 0.951875 | 0.902868 |
| within_population_hard | 0.892008 | 0.900268 | 0.954642 | 0.902801 |
| within_population_family | 0.892191 | 0.899777 | 0.948972 | 0.902784 |

The large CID22-human losses from global easy/hard sorting disappear when
supervision positions stay fixed: isolated easy/hard reach 0.902868/0.902801,
versus global 0.896742/0.896746 and random control 0.902551. This supports
population timing as a contributor to the global-schedule loss; it does not
establish that isolated curricula generally beat random mixing.

Within-population hard-first improves TID2013 by 0.001888 with a conditional
source interval [0.001579, 0.002434], but reduces KADID by 0.001209 with interval
[-0.002285, -0.000375]. Easy-first reduces TID2013 by 0.000653 with interval
[-0.000888, -0.000359]. All three CID22-human intervals cross zero. No arm
improves all four means. Preserve interleaved population exposure when testing
distortion order, and keep random mixing as the control.

Artifacts: `isolated-order-report/`, `isolated-order-bootstrap/`,
`isolated-order-breakdowns/`, `isolated-order-qualified/`.

## Full-budget architecture outcomes and numerical repair

24/24 original fitting runs reached epoch 119 at 120 x 50,000 draws.
18/24 original final bakes pass the unchanged feature gate; all three spline
and all three single-ramp bakes fail it. The native driver exits nonzero and
preserves those failures. Separate export evaluation retains every failed
candidate, adds three spline re-exports, and performs zero optimizer steps.

| Architecture | Feature gate | KADID | TID2013 | KonFiG F | CID22 human |
|---|---|---:|---:|---:|---:|
| shared_relu | PASS 3/3 | 0.897470 | 0.905377 | 0.948689 | 0.904456 |
| shared_smooth | PASS 3/3 | 0.896479 | 0.902115 | 0.948176 | 0.904561 |
| shared_caps | PASS 3/3 | 0.897168 | 0.899309 | 0.954372 | 0.903166 |
| shared_switch | PASS 3/3 | 0.896213 | 0.899685 | 0.956249 | 0.903207 |
| shared_switch_deep | PASS 3/3 | 0.897202 | 0.900033 | 0.960238 | 0.901846 |
| shared_spline | FAIL 3/3 | 0.898087 | 0.898855 | 0.953535 | 0.902947 |
| shared_lipschitz | PASS 3/3 | 0.897299 | 0.898477 | 0.956249 | 0.902463 |
| single_ramps | FAIL 3/3 | 0.893920 | 0.885626 | 0.958422 | 0.902418 |
| shared_spline_repaired | PASS 3/3 | 0.898087 | 0.898855 | 0.953535 | 0.902947 |

No new architecture improves all four means over shared ReLU. The deeper
switch reaches the highest KonFiG mean, but that panel contains one source
and its TID2013/CID22-human means are lower. The repaired spline raises KADID
mean slightly, with lower TID2013/CID22-human means. Source intervals retain
the uncertainty: the repaired-spline KADID difference is 0.000617 with interval
[-0.003577, 0.004785]. The one-block switch TID2013 difference is -0.005692
with interval [-0.009924, -0.002598]. No CID22-human architecture interval
excludes zero. These are conditional development diagnostics, not selection.

Capacity limitation: every long shared-ReLU, SmoothHinge and spline fit ends
with exactly one nonzero final output amplitude out of 64. Switch has
2/3/2 nonzero amplitudes; deep switch has 2/8/2 out of 32. These counts are
stored model coefficients, not sampled activity or proof that only one input
feature matters. They show that nominal width was not retained as distinct
final paths. The comparison cannot conclude that nonconvex capacity is useless;
future optimization work must explicitly track and preserve useful paths.

Original spline coordinate increases reach 0.00002288818359375; original ramp
increases reach the same maximum. All retain exact identity score 100. The
spline repair replaces signed unbounded-hinge cancellation with positive
bounded segments and keeps the first two layers/final head/scaler. Hidden-unit
connectivity is recovered from the serialized graph. All six short/long
re-exports pass the unchanged gate, with no negative reconstructed slope
requiring clipping. Their four development pooled SROCC values are unchanged.

Largest prepared-row score changes per long seed: 0.00022125244139914457,
0.000370025634765625 and 0.000152587890625. These are numerical export changes,
not bit-identical models. The short maxima are 0.00005340576171875,
0.00006866455078125 and 0.0000457763671875. TRAIN rank differences can be
nonzero because rounding changes ties. Historical checkpoints remain unchanged.

The repaired spline has 496097 stored coefficients in seven dense layers, versus
200417 in five for the old export. Both correspond to 132640 nominal
trainable coefficients in the original fitting parameterization. Dense expansion
is not a compute match, and no runtime speed claim is made.

The ramp export remains a measured native-gate failure; it is not silently
repaired or accepted. Strict floating-point monotonicity is not established
by the real-arithmetic construction or these finite feature probes. Pixel-domain,
near-lossless and cross-target shipping qualification remain outside this trial.

Artifacts: `long/SUMMARY.json` retains the six failures. `long-export-audit/`
contains original/repaired bakes, native panels and per-row change receipts;
`long-export-report/`, `long-export-bootstrap/` and `long-export-breakdowns/`
contain every candidate, including failed ones. `LONG_CAPACITY.json` pins the
native inspection and original fitting parameterization.
Long report SHA-256: `c639f38231bb58559e5bf0a8baf26eafeaf4414aacc8b874ba5a3ee445b5ea41`.
Bootstrap SHA-256: `588b9823f293fb357c2c90bea50d7a5cf62768288e03b056891fa4cab283c627`.

Long wrapper: `run-heavy: done rc=1 6367s | peak-RSS 0.67GiB | min-avail 36928MiB | peak-load 23.86`.
The nonzero exit records feature-gate failures after completed fits; it is not
an interrupted optimizer or an all-pass result. The resource line does not
measure aggregate process-tree memory or benchmark speed.

Final native checks: 169 trainer tests passed with one pre-existing ignore;
65 trainer-CLI tests and 38 bake-tool tests passed. The analytic construction
checks and script lint pass. CI is manual and was not dispatched.
