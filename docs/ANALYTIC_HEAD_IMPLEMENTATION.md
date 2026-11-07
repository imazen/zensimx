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
  **The first layer still uses the constrained shared-depth owner.** This is a
  diagnostic head, not a fully unconstrained MLP. A fully unrestricted control
  remains separate implementation work; do not relabel this one.

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
