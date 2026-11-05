# R14: smooth per-input transforms for A

**Incomplete: all 3 T1 fits and the T2 release decision are complete; product/EVAL/HDR reads, capacity
panels, head timing and archive verification are not yet measured.** No candidate
is selected or adopted. No EVAL data has been read by R14.

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

## Implementation and gates so far

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

Full-bank capacity contributions and all
quality/product reads remain outstanding. No missing evidence counts as a pass.
Frozen A is the control and is not refitted.

The public crate (imazen/zensim) is untouched: 0.3.0 ships A without transforms;
porting transform support to its derived path is a later follow-up.
