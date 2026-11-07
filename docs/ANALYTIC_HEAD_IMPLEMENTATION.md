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

Scientific fitting began after the frozen completion protocol and baseline replay.
Current execution evidence is recorded in
[the completion results](../benchmarks/shaping_completion_2026-11-07.md).
Implementation qualification alone does not establish scientific completion;
paired tuning, convergence diagnostics and generalization comparisons remain
separate requirements.

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

## Native input-shaping runtime

The private `zensim-derived-inputs research-shaping-v1` declaration distinguishes
`shape d k0 k1 s0 s1 s2`, `shape-product d f k0 k1 s0 s1 s2`, and
`separate d f k0 k1 s0 s1 s2 h0 hhalf h1`. Older runtimes reject its header.
Damage shapes integrate three nonnegative slopes over intervals delimited by
positive increasing knots. They preserve zero damage and are monotone on the
nonnegative damage domain. Unit slopes use an exact identity fast path.
Separate gates interpolate nonnegative heights at 0, 0.5 and 1, with constant
extrapolation outside this interval. Nonnegative gates do not imply monotonicity
in fragility; no such claim is made here. Negative damage is clamped to zero.
The exact mathematical sensitivity uses the right-hand slope at internal knots
and zero slope at the clamped domain boundary.

At `29895a8b`, the targeted runtime gate passed all 14 derived-input tests,
including a baked-model round trip, the identity score, the distinct product
operations and their full chain rule against served central differences.
A separate analytic-activation sensitivity test passed for Softplus and SiLU
on both sides of zero. Strict runtime library clippy passed. These gates do
not qualify the still-pending learned-shaper training/export integration.

Wrapper for the shaped round trip: `run-heavy: done rc=0 29s | peak-RSS 2.60GiB | min-avail 24610MiB | peak-load 1.61`.
Wrapper for analytic sensitivity: `run-heavy: done rc=0 24s | peak-RSS 2.60GiB | min-avail 24522MiB | peak-load 1.95`.
Wrapper for strict runtime clippy: `run-heavy: done rc=0 7s | peak-RSS 0.60GiB | min-avail 26441MiB | peak-load 1.92`.

## Learned input-shaping training owner

`ZENSIM_INPUT_SHAPING_MANIFEST` plus the independently supplied
`ZENSIM_INPUT_SHAPING_SHA256` enables the private `zensim-input-shaping-v1`
manifest. It declares logical columns, raw damage/reference ids, fixed knots,
initial slopes/gate heights, and `trainable`. The admitted owner is the
shared-depth CPU trainer with compact raw fields and scale-only standardization.
Separate gates require reference values in [0,1]. No product/damage division is
used to recover a reference field. Frozen and joint modes use the same runtime
f32 shaping arithmetic before the original feature scale.

Joint updates reuse native Adam, differentiating both factors before changing
the first-layer weights. Shape slopes and gate heights are projected nonnegative;
no monotonicity in the gate is claimed. Shape parameters currently have no
regularization penalty. That differs from weight regularization and must be
explicit in the scientific protocol; the factorization also has scale freedom.
Validation consumes current shapes. Every selected/final/epoch checkpoint
contains current parameters and the initial manifest hash. Native fx1/fx2
compaction emits the research declaration and runs its bit-exact served-forward
gate. `BakeScorer` refuses an intermediate shaping checkpoint lacking native
compaction; the trainer CLI requires `--derived-inputs` when shaping is enabled.

At `2c68977e` (identical tested tree before a description-only history rewrite), `just input-shaping-check` passed: 14 runtime derived-input tests,
one runtime analytic sensitivity test, 396 trainer library tests with two existing
ignores unchanged, 65 trainer CLI tests, and strict clippy for both owners.
Tests exercise independent parameter central differences, frozen versus joint
updates, current-shape validation predictions, and isolated real CPU training
whose final and every epoch bake passes native compaction. The initial fixture
omitted the production keep mask and correctly failed the unchanged dropped-row
export gate; the corrected fixture uses that mask. A preceding fixture compile
failure and all test logs are preserved. These are synthetic implementation
fixtures, not the scientific study.

Wrapper: `run-heavy: done rc=0 48s | peak-RSS 1.63GiB | min-avail 25486MiB | peak-load 3.88`.
Training-only pretraining, protocol registration, fair scientific fits and final
quality/generalization conclusions remain required. Supplied initial parameters
do not by themselves establish that a shaper was pretrained.
