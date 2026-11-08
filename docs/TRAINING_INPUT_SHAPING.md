# Native input shaping for constrained CPU models

The shared two-hidden-layer research owner and the constrained one-hidden-layer
owner support the same hash-bound input shaping manifest. The latter preserves
the original 128-unit architecture; adding shaping does not add a hidden layer.

Plain-owner admission requires K=1, a positive hidden width divisible by four,
nonnegative-distance, monotone-CBC and nonnegative-threshold constraints. Auxiliary
pools, transforms, smooth hinges, EMA, ramps and alternative heads are refused.
The shaper receives both sides' first-preactivation derivatives before the fused
weight update. Validation and every checkpoint use the current shape parameters.
The unshaped branch retains its existing readers, arithmetic and checkpoint path.

For continuation, supply a new independently hash-bound manifest containing the
previous checkpoint's exact shaping config. The native reader checks the compact
shaping declaration against that config and the original derived-input layout.
Shape parameters and network parameters continue together; Adam starts fresh,
as it does in ordinary checkpoint continuation. This is different from exact
full-state crash recovery. Full-state recovery admission remains shared-depth only.

The isolated-process native test runs frozen and jointly learned shaping through
both architectures, checking all epoch exports and native compaction gates.
Additional checks exercise plain validation against current shape values and
reject unsupported owners. These engineering checks do not establish scientific
quality gains or justify reusing controls trained under a different recipe.
