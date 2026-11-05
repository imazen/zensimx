# R14: smooth per-input transforms for A

**All 3 required T1 fits, 6 capacity cells, 24 read stages and 3 i270 head pairs
are complete. Final archive verification is pending.** Frozen A remains unchanged;
no transformed model is adopted. T2 and seeds 1–2 are not run under the registered
training-side conditions.

Yeo-Johnson (T1a) buys selective improvements in chroma, exposed E36 ordering and
compression median error, but worsens teacher development and the overall EVAL
means while increasing the measured head cost. The stronger soft-clip and mixed
arms do not improve the overall tradeoff. These are seed-0 results, without
candidate across-seed stability evidence.

[Full R14 read tables](r14_input_transform_reads_2026-11-04.md) include every EVAL
collection, compression/noncodec split, product gate and HDR population beside
all three A seeds. [Completed R13 comparisons](r14_r13_comparison_2026-11-04.md)
include all six width arms' available EVAL/HDR/capacity reads and their completed
product cells. [Artifact pointer](r14_input_transforms_2026-11-04.pointer.md).

Registration `5b5a0134` in [the experiment record](e36_subset_test_2026-11-01.md)
incorporates the coordinator's finite-gradient correction before screening.
[Exposure ledger](../docs/DATA_SPLITS.md): `bd790002`.

The allowed input kinds are identity, log1p, signed_log1p, Yeo-Johnson,
soft_clip and soft_sign, with validated finite parameters. They preserve zero
and have finite positive analytic one-sided slopes at zero (1 for all except
soft_sign's 1/scale). Root, dead-zone, step and arity-changing kinds are excluded.
Products precede transforms; scale-only statistics follow transforms.

T1a fits marginal TRAIN Yeo-Johnson lambdas and records TRAIN medians beside them.
T1b uses median-knee soft_clip with soft=knee. T1c applies the registered
TRAIN-only Pearson lift screen. T2 learns log knees only if the registered
teacher-development feasibility rule passes. Seeds1–2 require the separate
owner cost decision after seed0 passes its product/development conditions.
The full registration owns all constants, populations, objectives and reads.

## Frozen-model accuracy

| Model | Teacher dev MAE | E36 inversions (%) | EVAL9 error (%) | EVAL8 excluding MDID (%) |
|---|---:|---:|---:|---:|
| A0 | 3.225105 | 3.386235 | 7.94419 | 8.17742 |
| A1 | 3.237627 | 3.362575 | 8.09039 | 8.31690 |
| A2 | 3.393835 | 3.441131 | 8.04925 | 8.27506 |
| T1a seed 0 | 3.510737 | 3.227577 | 8.20518 | 8.45104 |
| T1b seed 0 | 4.404213 | 3.869474 | 8.46816 | 8.76595 |
| T1c seed 0 | 4.941278 | 5.353987 | 8.40385 | 8.71204 |

Lower is better in every numeric column. EVAL error is the equal-collection mean
of median residuals after the assessment-only logistic fit, normalized by each
collection's target 1–99% span. MDID is a training source, so EVAL8 also removes
that collection. The models and transform tables were frozen before these reads.

E36 compares 646,677 already-exposed selection pairs from 40 origins; it is not
held-out confirmation. Its source-bootstrap changes against the three-seed A
mean, with 10,000 draws, are:

| Arm | Inversion change (percentage points) | 95% source-bootstrap interval |
|---|---:|---|
| T1a | -0.169069 | [-0.324519, -0.025491] |
| T1b | +0.472827 | [+0.272937, +0.678314] |
| T1c | +1.957340 | [+1.718568, +2.225726] |

Yeo-Johnson improves the E36 selection-source ordering but worsens the teacher
development error and both EVAL means beside every A seed. Median-knee soft-clip
and the mixed screen worsen all three summaries. Some individual collection/HDR
statistics improve; the full read tables retain those results. EVAL does not
select a candidate, release T2 or authorize repeats.

## Product and HDR outcomes

| Read | A seeds 0 / 1 / 2 | T1a | T1b | T1c |
|---|---|---:|---:|---:|
| Chroma level / rank | A0 reference | pass / pass | fail / pass | fail / pass |
| Steering successes /135 (bar 128) | 133 / 134 / 134 | 134 | 127 | 128 |
| C1–C6 on both grids | all pass | pass | pass | pass |
| G-DIAL on both grids | all pass | pass | pass | pass |
| Identity: all 744 exactly 100, read sets invariant | all pass | pass | pass | pass |
| NEARID N1 / N2 | all pass | pass | pass | pass |
| NEARID N3 monotone ladders /144 (bar 122) | 132 / 132 / 132 | 131 | 135 | 137 |
| G-ADDR grading | not measured | not measured | not measured | not measured |

The inherited standard/ladder grids are absent from the G-ADDR floor registry,
and the resolvable-floor owner has no mentor per-cell truth for A7r. This affects
A as well as the candidates. Numeric dial measurements are retained, but no
G-ADDR pass is asserted and no gate/input is changed. The separate C1–C6 and
G-DIAL outcomes above are measured passes. All three candidates miss strict
teacher-development improvement; T1b/c also fail chroma, and T1b fails steering.
No arm is eligible for seed repeats. `SEED_ELIGIBILITY.json` records each condition.

T1a's compression equal-collection median error is 6.858%, against the three-seed
A mean of 7.021%; its compression row-weighted SROCC is 0.89341 versus 0.89449.
Its noncodec median error rises from 8.584% to 8.870%. These are descriptive
assessment splits, not candidate-selection rules. T1b/c's extended-XYB chroma
MAEs rise to 4.204848/4.566479 points versus A0's 3.576321; T1a is 3.462384.

HDR is mixed and report-only. T1c raises UPIQ-all pooled SROCC to 0.825999
against A's three-seed mean 0.811412, while within-reference Spearman falls
to 0.956577 from 0.962233 and inversions rise to 131 versus A's 121/121/124.
Its ladder inversions rise to 95 versus 50/51/49. T1a's UPIQ-all within-reference
Spearman is 0.963536 with 120 inversions, but its ladder inversions are 54.
The full tables preserve every panel result and the native JSON preserves
whole-family intervals. The single-family 4Kdtb interval remains undefined.

The completed R13 comparison uses identical A EVAL dictionaries and identical
capacity-bank hashes/row counts. T1a's EVAL9 mean error is above W32/W64/W256/W512
and below W4/W16. R13 W512 retains two live pairs, qualifying the earlier
one-pair premise; T1a also retains two. Neither retained pair count nor a
selection-source improvement establishes a held-out accuracy gain.

## Implementation and gates

- `726278fc`: extend the existing screen with an explicit variant filter and
  positive-percentile soft_clip/soft_sign candidates; historical default
  variants are preserved. The R14 pool excludes validation-only groups and
  checks regenerated coverage against A's receipt.
- `a3a40842`: trainer admission binds the exact registered A head and screen
  table through the native/Python authority; warm starts must preserve the
  transform table, and compaction retains it in caller order. ARM smooth-kind,
  negative-control and existing warm-start checks pass; trainer CLI compiles.
- `51d9e36c`: native certification accepts the smooth allow-list and emits
  tokens, params and analytic zero slopes. The linear signed-product exemption
  remains only for untransformed inputs; transformed inputs require ordinary
  nonnegative columns or the certified max-pair construction. ARM certificate
  negative controls and transformed warm-start equality checks pass.

ARM trainer-check receipt: 334s, run-heavy peak-RSS1.54GiB, minimum available
23539MiB. Certificate-check receipt: 42s, peak-RSS1.43GiB, minimum available
26022MiB. The additional warm-start/CLI check passes in3s. These are check-run
receipts, not inference-performance measurements.

The pre-serving-change `split_parity_golden --public-a` run over frozen A0/A1
has145327 records, zero error records, SHA256
`15641f4bbd3ebd26ab1f5a06fb08e7dca306b4dafb28df575cd1b4d6b83b1a90`.
It retains scalar, feature, HDR, pixel-format and diffmap outputs. The post-change
byte comparison passes: all145327 records are identical, with the same SHA256.
The ARM serving/gradient/compaction check sequence passes in693s, run-heavy
peak-RSS4.79GiB, minimum available22690MiB. New tests cover512 fixture rows
with exact caller/output bits, transformed product chain-rule derivatives,
finite zero-damage gradients and runtime attribution through that same gradient. Full artifacts and logs are under dev
`/var/tmp/byc-transforms-20261104/arm/`; generated data is not committed.
The default `cargo test -p zensim -- --test-threads=1` command passes on ARM,
including integration tests and doctests, without changing any existing
expectation or ignore marker. The authority/exporter/default-suite sequence
passes in1001s (run-heavy peak-RSS2.49GiB, minimum available25474MiB).
The exporter checks transformed split-layer execution against complete native
forward bits on256 rows. Native authority tests reject a missing registration,
changed screen table and changed A head.

A native toy fit verifies post-transform scaler statistics exclude validation-only
rows, preserves transform metadata and certifies the exact zero-input100 pin.
The final transformed-training/native-CLI lint sequence passes in54s,
run-heavy peak-RSS1.31GiB, minimum available26745MiB, with no warning in the
new transform module. R14 identity slots narrow to f32 alongside nonidentity
slots, matching the compact baseline loader and inference caller arithmetic.

Before the first fit, a continuation-loader audit found that checkpoint
transforms are restored after loading, while lazy compact rows carry raw
products. R14 continuations now select dense rows so the normal transform pass
runs before scaling. Untransformed A keeps its original loader selection.
The initial native CLI regression and preflight tests, formatting check and targeted
clippy pass on ARM in 35 seconds (peak RSS 1.17 GiB, minimum available memory
26,108 MiB). Parameter-count checks for the newly admitted CLI kinds are scoped
to R14; historical non-R14 argument admission is preserved.

The first T1a base fit completed all 120 epochs, but its continuation failed
before any optimizer step because direct dense loading encountered an unused
nullable column. The follow-up loads A's declared raw subset through the
existing compact reader, then materializes the model-visible callers before
the transform pass. Required-column nulls still fail; no data or expectation
was relaxed. A 256-row products/transform fixture is exact-bit, and a nullable
parquet fixture checks both the allowed and refused cases. The ARM sequence
passes in 69 seconds (peak RSS 1.18 GiB, minimum available 26,622 MiB).
The completed base checkpoint and failed continuation remain immutable; the
retry uses a separate directory and records native identities per stage.
The repaired trainer is pinned to `fc41da56`, SHA256
`002740271cb808b0709b1f9ff6da3b89ca1d25bd399a41b109fb59c9715a8d35`;
its native build passed in 47 seconds (peak RSS 1.80 GiB, minimum available
39,106 MiB). All serving/read binaries retain the original pin below.
The shared recipe parser subsequently gained an unrelated registered width
argument. The R14 controller verifies that the rest of that module's AST is
identical and that both committed parser versions produce the identical
registered H128 recipe. It preserves both hashes and the comparison receipt.

The conditional T2 controller requires all three frozen T1 fits before applying
the registered development rule. It uses the successor trainer above, preserves
the initial winner's transform kinds, and checks final learned knees against
the frozen TRAIN bounds. It records changed knees and bound hits. The release
path is implemented; the measured release decision below does not authorize a
T2 fit.

The untouched ARM strict-library-clippy baseline reports eight unused-item
warnings as errors; this baseline receipt is retained without suppressing them.
The Python execution-owner syntax, shell syntax and CLI setup checks also pass
on ARM (`just r14-python-checks`, source `3e0c561c`, peak RSS 0.02 GiB).
These checks load no scientific data; runtime fit/read receipts remain required.

The final native binaries are pinned to `912609aa` (Rust 1.99.1), with all ten
binary hashes in `SOURCE_PINS.json`. Trainer SHA256:
`3c9b88b82ab9f1688a4ec9dc4ddd2b9321d9ca94275829973d8d77d20d483949`.
The first native build compiled every binary but failed its final provenance
assertion: this jj version prints a zero-file summary for an empty diff.
The correction uses jj's `empty` predicate. Both attempts are preserved:
332 seconds/peak RSS 2.24 GiB for the compile attempt (exit 1), then
1 second/peak RSS 0.09 GiB for the successful cached rerun. Both used the
shared resource lock and 24 GiB cap. No fit started before finalization.

The final native no-op proof also passes: all three frozen A seeds have zero
served f64-bit differences between the old and rebuilt scorers on all 12,163
CID22 TRAIN rows and 7,390 HDR teacher TRAIN rows. There are 58,659 paired
comparisons in total, with identical score-stream SHA256s for each bank/seed.
No targets were read. The run took 3 seconds (peak RSS 0.42 GiB, minimum
available memory 40,365 MiB). [Native proof receipt](r14_noop_2026-11-04.json)
retains both instrument identities, model/bank hashes and score-stream hashes.

## TRAIN screen

Pre-read amendment `447fcc16` restricts the absolute variance <1e-10 guard
to T1a's marginal MLE. T1b keeps its positive-median rule, while T1c keeps the
existing Pearson std <1e-12 guard and omits Yeo-Johnson where T1a has no fit.
This correction landed before the first TRAIN payload read. In the actual
pooled bank no input meets either near-constant guard.

The registered screen completed on 627,732 rows from A's seven positive-weight
TRAIN groups. It read no development or EVAL rows. All 984 caller inputs were
nonnegative and nonconstant; none met the Yeo-Johnson near-constant guard.
The target was the trainer's stored `human_score`, with scale 1.

| Arm | Yeo-Johnson | Soft-clip | Soft-sign | Identity |
|---|---:|---:|---:|---:|
| T1a | 984 | 0 | 0 | 0 |
| T1b | 0 | 956 | 0 | 28 |
| T1c | 5 | 327 | 588 | 64 |

T1b's 28 identity inputs have zero TRAIN medians, which cannot define a positive
knee. T1c keeps identity when no admitted candidate reaches lift 0.005.
Its log1p and signed_log1p candidates were available but never selected.
In particular, f12/f25/f51/f103 have zero pooled TRAIN medians but positive
variances (respectively 0.07987541485675369, 0.02307203676416467,
0.06759058838090709, 0.010680416719629264). They are not constant in this
seven-group population, so T1a retains Yeo-Johnson and T1b uses identity.

| Selected positive parameter | Count | Minimum | Median | Maximum |
|---|---:|---:|---:|---:|
| T1b soft-clip knee | 956 | 1.459379973312025e-6 | 0.011053490452468395 | 0.26553836464881897 |
| T1c soft-clip knee | 327 | 5.6206893361832044e-8 | 0.018190236762166023 | 0.28836750984191895 |
| T1c soft-sign scale | 588 | 1.459379973312025e-6 | 0.00710236607119441 | 0.5779024362564087 |

The T1a lambda histogram, in bins with edges
`[-2,-1.5,-1,-0.5,0,0.5,1,1.5,2]`, is
`[976,3,3,2,0,0,0,0]`. This is the registered bounded MLE sweep, rather than
an unconstrained search over all finite lambdas. There are 971 lambdas below
−1.9999. The median lambda is −1.999995470046997; its maximum is
−0.20050404965877533. The median of the inputs' TRAIN p50 values is
0.01059551676735282 (direct inputs: 0.01506041269749403; products:
0.006778999697417021). Every input's TRAIN p50 and
lambda, and T1b/T1c's exact per-input parameters, are retained in the screen
artifacts. No performance or accuracy result follows from these choices alone.

The screen receipt is 183 seconds, run-heavy peak RSS 3.58 GiB, minimum
available memory 39,930 MiB, exit 0. Producer commit `bfa87dd7` verifies the
screen-owner source bytes before loading data. Artifact paths and SHA256s are
recorded in [the pointer](r14_input_transforms_2026-11-04.pointer.md).

## Conditional T2 implementation

T2 starts a fresh A seed-0 network with the best T1 transform table. Only its
eligible soft-clip knees learn; softness remains tied to the knee. The native
K=1 loop propagates both RankNet and MSE through the same pre-update encoder
weights. Above the knee, the parameter derivative is
`dt/dlog(k) = k*ln(x/k)`; inside the identity core it is zero.
The native Adam kernel uses the network's learning-rate schedule and its
existing 0.9/0.999/1e-8 constants and moment policy. Log knees project to the registered TRAIN bounds; nonfinite states
and nonpositive/nonrepresentable stored proposals are refused without an epsilon.

The optimizer uses the initial post-transform scale throughout each fit.
Each serialized checkpoint recomputes the learned columns' current
post-transform TRAIN scales and compensates their encoder rows by the positive
scale ratio. This preserves the real-valued function and monotone constraints;
serialization has the ordinary f64-to-f32 parameter rounding. The independent
four-hidden-unit export fixture checks function preservation to 1e-12 before casting and 3e-5
after native f32 serialization/inference. This tolerance is for T2 checkpoint
quantization; the serving parity and no-transform A gates remain exact-bit gates.
Validation-only rows are excluded from both initial and serialized scale fits.

The native CLI binds `--r14-knee-bounds` to the frozen TRAIN statistics and
best initial table, and hashes the teacher-development decision. It recomputes
the best-arm/zero-allowance release rule before loading training payloads.
T1 arms refuse the learning flag; T2 refuses an absent flag or failed release.
The existing T1 wrapper is unchanged. A separate T2 wrapper composes the same
canonical fit owners and carries the binding through continuation.

ARM checks pass for native learning, finite-difference parameter gradients,
projection, zero-lower-bound refusal, transform/warm-start regressions and the
release binding. The final combined sequence takes 74 seconds (peak RSS
1.39 GiB, minimum available 26,157 MiB). The all-targets compile check also
passes in 26 seconds (peak RSS 0.56 GiB, minimum available 26,460 MiB).
The subsequent percentile-boundary regression verifies inward f32 rounding
after log projection: a stored knee remains inside the exact TRAIN interval
even when the percentile itself is not representable. Zero underflow still
fails without an epsilon. All five knee tests and targeted clippy pass on ARM
in 51 seconds (peak RSS 1.55 GiB, minimum available 25,925 MiB). A conditional
native T2 trainer build has its own selection receipt; T1 trainer pins stay fixed.
T2's scientific fit was not released: the registered T1 development comparison
below fails feasibility, and its winning Yeo-Johnson table has no soft-clip slots.
All serving/read binaries remain pinned to `912609aa`. Trainer successors
are pinned separately, including each fit stage's native identity.

## Results

T1a seed 0 completed its 120 base epochs and 60 continuation epochs. Its final
40,076-byte packed bake is certified, SHA256
`6a28893f896696fde6832782631e0af0582e9a317478fc5dc621198716935487`.
Measured payload wall is 2,291.47 seconds: base 1,615.09; continuation 675.52;
packing/inspection 0.86. Shared-lock waits and the failed pre-optimizer loader
attempt are excluded. Structural capacity is 94 dead / 30 residual / 4 live
hidden units; the final certificate retains two active max pairs.

Base resource receipt: `run-heavy: done rc=0 1615s | peak-RSS 15.52GiB | min-avail 27156MiB | peak-load 3.05`.
Continuation: `run-heavy: done rc=0 676s | peak-RSS 12.50GiB | min-avail 27634MiB | peak-load 2.32`.
The first post-pack inspection failed because the frozen R12 executable's
inspection branch called its older serving metadata validator. The canonical
pack and certificate had already succeeded. Rebuilding the same inspector
against transformed-derived serving support took 39 seconds (peak RSS
1.64 GiB, minimum available 36,291 MiB). Recovery verifies the original command
and recorded SHA256 and inspects those same packed bytes; no calibration or
fit repeats. The R12 timing executable remains unchanged. Both failed attempts
and stage-specific native identities are retained in the artifact root.
[Fit receipt](r14_fit_progress_2026-11-04.json) contains the exact successful
times, hashes, certified transform inventory and resource receipts.

T1b seed 0 is also packed and certified, with its 956 soft-clip and 28 identity
inputs exactly matching the frozen table. Its 58,269-byte packed bake has SHA256
`7a02cd54ba692f5995477657d9ec5ec125602678e1f397dcde9af46a53c75065`.
Payload wall totals 1,941.22 seconds: base 1,326.00; continuation 614.39;
pack 0.83. Structural capacity is 116 dead / 5 residual / 7 live units.
Both scientific stages use the selected-caller trainer `fc41da56`.
Base resource receipt: `run-heavy: done rc=0 1326s | peak-RSS 12.50GiB | min-avail 27598MiB | peak-load 4.26`.
Continuation: `run-heavy: done rc=0 615s | peak-RSS 12.50GiB | min-avail 27612MiB | peak-load 2.39`.
These payload sums exclude shared-lock waits; training-wall comparisons also
carry T1a's distinct base-loader version, so they are not inference-speed claims.

T1c seed 0 is packed and certified with the frozen mixed transform table.
Its 47,108-byte packed bake has SHA256
`de187fbabbdb1a1e999fed21c19f4589281e3e6184893c4ba559921922b71d98`.
Payload wall totals 2,045.30 seconds: base 1,446.13; continuation 598.38;
pack 0.79. Structural capacity is 123 dead / 0 residual / 5 live units,
with one certified max pair. Both scientific stages use `fc41da56`.
Base resource receipt: `run-heavy: done rc=0 1447s | peak-RSS 12.50GiB | min-avail 27525MiB | peak-load 2.16`.
Continuation: `run-heavy: done rc=0 599s | peak-RSS 12.50GiB | min-avail 27275MiB | peak-load 2.83`.
All three final T1 model hashes are frozen before development feasibility or
any EVAL read. T1a has two certified max pairs; T1b and T1c each have one.

### Registered development comparison

| Final bake, seed 0 | SafeSyn MAE (38,757 rows) | CID22 MAE (3,785 rows) | Pooled MAE |
|---|---:|---:|---:|
| A0 | 3.393523 | 1.500572 | 3.225105 |
| T1a | 3.705736 | 1.514020 | 3.510737 |
| T1b | 4.679956 | 1.580704 | 4.404213 |
| T1c | 5.267466 | 1.601232 | 4.941278 |

Errors use the stored teacher `human_score` points, with target scale 1.
Lower is better. T1a wins among T1 arms but misses the registered <=A0
zero-allowance feasibility rule; it also has no eligible soft-clip parameters.
**T2 is not run.** All T1 arms miss the strict development improvement required
for seeds 1–2, so no repeat is eligible regardless of the outstanding product
reads. No EVAL value informed either decision.
[Development and release receipt](r14_development_2026-11-04.json).
The comparison job passes in 5 seconds (run-heavy peak RSS 0.97 GiB,
minimum available 39,810 MiB).

### Capacity on the complete CID22 TRAIN bank

All 12,163 rows and all 492 joint damage-slot interventions are included.

| Arm | Dead / residual / live | Live max pairs | Largest pair share (%) | Inputs at 50/90/99% ablation mass |
|---|---|---:|---:|---|
| T1a | 94 / 30 / 4 | 2 | 68.947980 | 12 / 88 / 176 |
| T1b | 116 / 5 / 7 | 1 | 98.990304 | 19 / 134 / 266 |
| T1c | 123 / 0 / 5 | 1 | 97.482152 | 27 / 133 / 246 |

The native audit and ablation owners use the same R13 definitions: exact-zero
encoder columns with nonpositive biases are dead; residual excludes dead and
has absolute output weight below 1e-3 of the largest. Pair share sums measured
absolute raw hidden contributions, `abs(w2)*mean(abs(activation))`, against
all units. Each ablation jointly zeros a damage slot's direct and product
callers; effects overlap and are not an additive score decomposition.
[Capacity receipt](r14_capacity_progress_2026-11-04.json).

On all 7,390 rows of the frozen HDR TRAIN feature bank, the structural counts
are unchanged and the measured contributions are:

| Arm | Live max pairs | Largest pair share (%) | Inputs at 50/90/99% ablation mass |
|---|---:|---:|---|
| T1a | 2 | 69.680195 | 12 / 78 / 163 |
| T1b | 1 | 99.586897 | 19 / 111 / 243 |
| T1c | 1 | 98.078254 | 20 / 123 / 242 |

All six cells passed the final bank/model identity, full-row coverage,
packed structural-count and canonical native-certificate checks. These are
label-free TRAIN diagnostics, separate from the HDR quality panel.

Frozen A is the control and is not refitted. The linked full read tables retain all completed pixel/dial jobs. No missing evidence counts as a pass.

The public crate (imazen/zensim) is untouched: 0.3.0 ships A without transforms;
porting transform support to its derived path is a later follow-up.

## Head timing on i270

The exact frozen R12 head executable, SHA256
`aa409c8e689323f7ed0f8ca4f07fa34565d673efa18e26806f98572677af3a33`,
ran each final packed candidate against A0 on i270, observed as Intel Core
Ultra 7 270K Plus. It uses one thread, `ZENSIM_MAX_TIER=v3`, the default
process affinity, and the same 256 deterministic positive caller rows per pair.
This measures transforms, scaling and network forward; it excludes extraction,
derived-product formation and the served dial. It is not full-image throughput.

| Candidate | Paired A0 mean (µs/image pair) | Candidate mean (µs/image pair) | Gate-clean / retained rounds |
|---|---:|---:|---:|
| T1a | 11.486382 | 19.557497 | 198 / 200 |
| T1b | 11.809412 | 18.689147 | 180 / 200 |
| T1c | 9.905098 | 15.115407 | 170 / 200 |

Each cell uses the same gate-clean round indices for its A/candidate pair.
All retained rounds, paired indices, all-round means and hashes are in the
[head receipt](r14_head_timing_2026-11-04.json). No value-based trimming,
round truncation or timing-based candidate selection was performed. A0 differs
between pairs, so the table retains each contemporaneous control.

The native zenbench sampler checks convergence from 30 rounds, then every 10,
with a 200-round cap. All three runs reached the cap, with native
`unreliable=false`; precision convergence is not established by these
cap-ending runs. These are descriptive means with that limit. R12's historical
summary requires exactly 30 rounds because its historical runs stopped there.
That consumer remains unchanged; R14's separate summary validates the frozen
instrument's actual adaptive population and preserves all 200 samples. The
first R14 controller exited after successful native timing when it borrowed
R12's 30-round assumption; the retained native execution was exit 0 and was
not rerun. An earlier staging attempt failed before any measurement because
remote rsync was absent; its receipt is retained and SCP was used.

Resource receipt: `run-heavy: done rc=0 139s | peak-RSS 0.03GiB | min-avail 25179MiB | peak-load 1.61`.

## Packed and stripped artifacts

| Arm | Packed bytes | Stripped bytes | Stripped SHA256 |
|---|---:|---:|---|
| T1a | 40,076 | 28,905 | `ea208e75ff725b3f820ad9eb01dd3c37a56e8056ab3caf598115953c62e3288c` |
| T1b | 58,269 | 47,120 | `046f3893cb17d2f6f74723d8a99d9b8eba4786e2f38d8c0a071753b24f77e1d4` |
| T1c | 47,108 | 36,160 | `b7d19edb5c72f39a6d3dc6d5c211259615f1342eeb6f315fb8b402734a179623` |

Stripping preserves the native layer/scaler/derived/dial inspection exactly.
Both `zentrain.feature_transforms` and `zentrain.feature_transform_params` are
byte-identical before and after stripping; they remain serving metadata.
`FROZEN_FITS.json` retains both paths, sizes, hashes and equality receipts.
Quality and timing reads use the frozen packed bytes, without recalibration.

The final ARM Python-owner check parses all 13 R14 modules and passes the
existing shell/CLI checks in 1 second (run-heavy peak RSS 0.03 GiB, minimum
available 27,178 MiB). Its retained parity executable has SHA256
`2756292de94009ef3683add19f1884959f47d15b88946b83054db1b11e080492`.
The copy was verified against the ARM source binary before removing this
experiment's inactive Cargo target; the golden streams and full logs remain.

## Final native trainer verification

The final native trainer, including inward f32 knee-bound projection, builds
on dev under the shared lock. SHA256:
`49a0a692863bb89b2fb6d3275711543ba15ffef33ebca7589bfda02005d5796e`.
It has a separate selection receipt and leaves all fitted-model, serving-binary
and T1 trainer pins unchanged. T2 release remains false; this is a build check,
not a scientific T2 fit.

Resource receipt: `run-heavy: done rc=0 34s | peak-RSS 1.77GiB | min-avail 40354MiB | peak-load 1.66`.
The lock-inclusive wall time was 1,081.96 seconds; it is not compile time.
The exact pre-rebase build tree `58451fce12cb` is preserved as a git archive
with its embedded commit ID. Its native compile inputs equal the published
`ea0e8ea1` tree; only the report/receipt and two standalone analysis modules
differ. [Final trainer receipt](r14_final_trainer_2026-11-04.json) owns both
identities, source hashes, capsule hash and the resource receipt.

## Interpretation

These fixed transforms do not buy an overall accuracy/cost improvement over A
in this registered seed-0 test. Yeo-Johnson is the strongest tested family: it
passes chroma, steering, identity, NEARID and the measured dial contracts, and
improves exposed E36 ordering plus compression median error. Its teacher-dev
MAE and overall EVAL error are worse beside every A seed, its G-ADDR evidence is
ungraded, and its descriptive head mean is 19.56 µs versus paired A0's 11.49 µs.
The stronger soft-clip and mixed screen worsen the aggregate errors and fail
chroma; T1b also misses the steering bar. Some individual EVAL/HDR improvements
remain visible in the full tables. T2 would add supervised adaptation of
soft-clip knees jointly with the network, within frozen TRAIN percentile bounds;
no T2 accuracy is measured because its preregistered development rule fails
and the winning T1a table has no eligible soft-clip slots. No repeat or adoption
is selected from EVAL.
