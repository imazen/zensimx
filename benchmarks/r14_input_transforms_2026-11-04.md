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

The untouched ARM strict-library-clippy baseline reports eight unused-item
warnings as errors; this baseline receipt is retained without suppressing them.

## Results

Not measured yet. Complete every registered arm/read or report a verified blocker;
no missing evidence counts as a pass. Frozen A is the control and is not refitted.

The public crate (imazen/zensim) is untouched: 0.3.0 ships A without transforms;
porting transform support to its derived path is a later follow-up.
