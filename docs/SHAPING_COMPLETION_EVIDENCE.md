# Evidence carried into the activation and shaping follow-up

This reconstruction distinguishes existing evidence from the new study's open
questions. It is not an experiment registration or authorization to select on
outer folds. Numerical claims below belong to the linked result records;
fresh archive verification receipts are separate evidence of artifact retention.

## What the completed work establishes

The [shaped-linear study](SHAPED_LINEAR_GENERALIZATION_STUDY.md) uses three
source-component outer folds within FIT and two inner folds. Reference aliases
and exact distorted duplicates remain grouped. Feature scales, segment knots,
label maps and selection belong to training complements. Exposed development
is descriptive; final holdouts remain untouched.

Its certified quadratic optima concern a fixed basis, weighting, regularization
and squared-error objective. They do not bound rank correlation or the information
in the measured features. Positive shaping changes the additive basis; it does
not train a shaper jointly with a nonlinear network. The aggregate outer shaping
gain has an interval crossing zero, while CID22-human benefits more strongly.
See the [results](../benchmarks/shaped_linear_generalization_2026-11-06.md).

Extra Trees outperform the fixed shared-ReLU comparator on those source folds.
These are matched-data diagnostics, with different objectives and optimization
budgets, not equal-search evidence that trees dominate every neural construction.
Trees' shuffled-label fitting demonstrates memorization capacity, not that their
real-label generalization is merely memorization. Failure of a constrained fit
to memorize also does not prove a general monotonicity limitation.

The [architecture results](../benchmarks/monotone_architectures_2026-11-06.md)
are development comparisons, not the later source-fold composites. Long ReLU,
SmoothHinge and spline fits each retain one nonzero output amplitude. Upstream
nonlinearity survives; this is neither a linear model nor evidence that nominal
width was effectively used. Leaky switches remove sampled inactive units but
do not establish a uniform quality gain. Original spline/ramp numerical gate
failures remain failures; spline re-exports are separately measured repairs.

## Questions the follow-up must actually answer

1. Can exact anchored Softplus and separately constrained SiLU train and export
   correctly? Anchoring alone does not make ordinary SiLU monotone. Both anchor
   terms require differentiation, including learned bias and temperature.
2. Does shaping help a shared MLP when frozen after training-only pretraining,
   initialized to identity and jointly learned, or pretrained and jointly learned?
   Each needs the matched unshaped control and identical source boundaries.
3. Does separate damage/reference-gate shaping improve on shaping their product?
   Damage identity and monotonicity, nonnegative reference gates, and any claimed
   fragility monotonicity are distinct contracts.
4. Do initialization, projection, regularization or schedule explain collapsed
   output paths? Track gradients, boundary coefficients and useful paths through
   training; one failed fit is inconclusive about representational capacity.
5. Do gains survive paired seeds, inner-only selection, source learning curves,
   distortion breakdowns, shuffles and human-only supervision? Human outcomes
   stay separate from teacher agreement, with source-bootstrap uncertainty.

Before scientific fitting, register tuning opportunities, paired seeds, exposure,
checkpoint selection and convergence criteria. Existing controls can substitute
only when data, objective, budget and selection conditions match. Native runtime
parity and numerical gates precede quality interpretation. Inference cost is a
separate measured comparison; expanded coefficient counts are not timing results.
