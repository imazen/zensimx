# Native analytic heads for the shaping completion study

Implementation work for the [completion evidence record](SHAPING_COMPLETION_EVIDENCE.md),
not a registration of scientific fits or a quality result. The exact runtime
operators are under review in [zenanalyze PR 93](https://github.com/imazen/zenanalyze/pull/93).
The research dependency pins that implementation; no serving model is promoted.

The private, independently hash-bound `monotone-head-v1` manifest admits three
new families. Each requires an `analytic` object with `temperature`,
`learn_temperature` and `bias`. Existing manifests omit this object and retain
existing initialization draws and numerical paths.

- `softplus`: nonnegative hidden weights, signed bias, anchored unit Softplus.
- `silu_monotone`: nonnegative hidden weights and bias, anchored unit SiLU.
  Its certified first-layer pair sums are nonnegative, so all analytic
  preactivations stay in SiLU's nonnegative, increasing domain.
- `silu_diagnostic`: ordinary unanchored SiLU, signed intermediate weights and
  unrestricted output amplitudes. It has no identity or monotonicity claim.
  By default the first layer still uses the constrained shared-depth owner.
  `unrestricted_first: true`, admitted only with this family and `max_pairs: 0`,
  disables first-layer weight/bias projection and pair ties. That explicit mode
  is the fully unrestricted diagnostic. Both diagnostic modes learn the output
  bias and retain the same scale-only feature transformation as the controls.
  The manifest and training log distinguish them; neither claims monotonicity.

For the anchored families the response is
`tau * (f((b + W*x)/tau) - f(b/tau))`. Bias derivatives subtract both branch
slopes. Log-temperature derivatives subtract both `f(u)-u*f'(u)` terms and
multiply by tau. Temperature is positive by log parameterization, with log tau
projected to [-6,6]. A fixed temperature receives no optimizer gradient.
Projection preserves the claimed weight/bias domain after each update.

Native export emits activation branches, a separate unit-coefficient subtraction,
then temperature scaling. Combining arbitrary temperature scaling into the
subtraction can leave a fused-multiply-add residual at identity. Scaling follows
exact cancellation; the score head follows scaling. Tests include non-power-of-two
temperatures, signed Softplus anchors, finite extremes, parameter finite
differences, post-update identity/order and native-versus-training predictions.
These feature-space gates do not imply pixel-domain qualification.

Training diagnostics retain output amplitudes, counts of negative proposals
clipped at the output boundary, mean squared regularized output-gradient norm,
and analytic temperatures, alongside existing sampled hidden activity and
first-layer gradients. Counts distinguish repeated projection from merely
observing zero weights at the end. They do not by themselves prove a cause of
capacity collapse. Ordinary diagnostic outputs are not projected nonnegative.

No scientific fitting has started. Frozen protocol, fair tuning, baseline replay,
input/fragility shaping, convergence experiments and generalization evaluations
remain required by the completion goal.

## Measured implementation qualification

At source `6bb5bd47`, `just analytic-head-check` passed on i270: 386 library
tests passed with two existing ignores unchanged, all 65 trainer CLI tests
passed, and strict library/trainer clippy passed. The library run includes the
new full-network analytic parameter gradients, post-update export/order gates,
dropout-anchor checks and finite-extreme probes. This does not replace the
required actual-data baseline replay or fair scientific fits.

Wrapper: `run-heavy: done rc=0 77s | peak-RSS 1.63GiB | min-avail 25459MiB | peak-load 3.53`.
This is qualification resource evidence, not an inference timing benchmark.
Logs, including the preserved dependency-resolution and incomplete-cache
failures, live in `/mnt/v/output/zensim/shaping-completion-20261107/trainer-qualification/`.
The isolated Cargo source cache repaired the missing dependency source files
without changing the dependency version or any test expectation.

The qualification directory is mirrored under the matching NAS output path.
Cold client readback verified every artifact; inventory SHA-256 is
`282d47327d9f8730b825401e2896d428b1504e9c0a672fa65f39947f87045800`.

## Compatibility replay and unrestricted mode

The release binaries from `6bb5bd47` reproduce all three frozen native control
bakes byte-for-byte after removing only `zentrain.repro`: single-layer, shared
ReLU and shared SmoothHinge. The commands, original expected hashes and replay
outputs are retained under `shaping-completion-20261107/baseline-replay/`, with
a verified NAS mirror. This replay predates the unrestricted-mode extension.

The unrestricted-mode extension is covered by an isolated-process test invoking
the actual CPU trainer with its hash-bound manifest and synthetic data. It
checks retained negative first-layer weights, unchanged scale-only preprocessing
and non-pinned identity output. Separate tests cover learned output bias, signed
head paths and export parity. Its initial fixed 1e-5 export tolerance measured
a 1.4603684121539118e-05 residual near score 100; this new test now uses the same
f32 pin/damage rounding bound as the existing network export gate. No established
expectation changed. These checks remain implementation evidence, not a
human-data fitting or generalization result.

At compiled source `d93fe729`, the full qualification passes: 388 library tests
(two existing ignores unchanged), 65 CLI tests and strict clippy. The tested
Cargo manifests and implementation files match the coordinating checkout by
SHA-256. Wrapper: `run-heavy: done rc=0 38s | peak-RSS 1.63GiB | min-avail 25068MiB | peak-load 3.33`.
The complete success/failure logs and isolated-process fixture are mirrored
under `shaping-completion-20261107/unrestricted-qualification/`; verified
inventory SHA-256: `5fc593ba04273859f4f6877429bfe1e3084e639af3b2650e2fd8d3cb1e4d1c81`.
