# R14: smooth per-input transforms for A

**Incomplete: fits, development feasibility, product/EVAL/HDR reads, capacity
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
The native CLI regression and preflight tests, formatting check and targeted
clippy pass on ARM in 35 seconds (peak RSS 1.17 GiB, minimum available memory
26,108 MiB). Parameter-count checks for the newly admitted CLI kinds are scoped
to R14; historical non-R14 argument admission is preserved.

The untouched ARM strict-library-clippy baseline reports eight unused-item
warnings as errors; this baseline receipt is retained without suppressing them.

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

## Results

Not measured yet. Complete every registered arm/read or report a verified blocker;
no missing evidence counts as a pass. Frozen A is the control and is not refitted.

The public crate (imazen/zensim) is untouched: 0.3.0 ships A without transforms;
porting transform support to its derived path is a later follow-up.
