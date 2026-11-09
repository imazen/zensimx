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

## Reusable defaults and predictable training

The owner clarified on November 9 that shaping should reduce outlier ratios and
scatter sprawl, and make good training runs easier to understand and achieve.
Average rank correlation alone does not establish that benefit. Compare tails,
raw calibration, normalized scatter geometry and paired-seed reliability while
retaining the registered human headline and separate teacher diagnostics.

Curve agreement is descriptive: downstream weights can compensate for a shaper's
scale and form. A transferable default needs explicit anchor/scale conventions,
training-complement-only derivation and an actual matched frozen-bank or
joint-from-bank trial. A bank assembled across folds that saw an evaluation
source cannot establish source-disjoint transfer on that source. Register any
additional trials before fitting; the current primary selection rule is unchanged.

`just shaping-tail-summary STUDY DASHBOARD OUTPUT` reads existing native scalar
panels for every one of the 72 registered setting/partition/seed conditions in
each neural primary family. It preserves missing/undefined metrics without a
partial aggregate, compares identical conditions against ReLU, and separates
raw residual tails from quantile-normalized geometry. Human summaries give the
four rated populations equal weight; CID22 SSIM2 remains separate. It performs
no fitting, inference or architecture selection. Corpus-standard-deviation
outlier ratio is a broad threshold, so zero OR does not establish tight scatter.
