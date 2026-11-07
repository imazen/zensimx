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

## Archive re-verification, November 7 UTC

All three immutable archive manifests were reread against both local and NAS
copies with client page-cache eviction requested. Every listed file matched;
original manifests and receipts were left unchanged. Counts include each
original mirror receipt. The source of this verifier is `c8e87356`.

| Study | Files per copy | Bytes per copy | Manifest SHA-256 |
|---|---:|---:|---|
| shaped-linear | 4404 | 10391488123 | `642428868aada79b69ee4358b14e01a279b7c068e6dd1f0bdea2e478de7b8882` |
| architectures | 19562 | 1862519579 | `bd18be6b0bef866c781b5a79231713b89ca54ab32e56a990738b85c7532924e7` |
| trees | 6197 | 6450819470 | `c98912ff5e927496899a5f6d4bd5aaa8a60f4b331d5b1fa483966ce1ac35ecad` |

Fresh receipts and complete wrapper logs live under
`/mnt/v/output/zensim/shaping-completion-20261107/archive-verification/`.
This verifies retention of the earlier experiments, not completion of the new
activation/shaping comparisons. Their scientific fitting started subsequently;
see [the execution record](../benchmarks/shaping_completion_2026-11-07.md) for
completed controls and current qualification limits.

### Dependency-aware pretrained dispatch

The execution owner accepts explicit registered cell IDs within its selected
stage, partition and family. This changes dispatch order only, not the matrix,
training budget or completion requirement. Duplicate and out-of-scope IDs are
refused. Pretrained children require a terminal successful matching parent before
any child attempt is created; the existing extraction owner then validates its
complete identity, exposure, binary and prefix hashes. Subset controllers have
separate status filenames. This allows frozen and joint fresh-head trials to
start while other independent parent settings are still training.

The pretrained dispatcher supervises two child fits at a time and records its
PID, required IDs, child PIDs and terminal status under `study/dispatch/pretrained/`.
It checks matching-parent terminal artifacts before release, preserves existing
child attempts, and waits for already started children if a child or parent
fails. Per-job storage reservations still govern admission. This is dependency
scheduling, not an exclusive compute lock; independent controllers keep running.

Actual first-parent prefix extraction exposed equivalent f32 knot decimals with
different Python-f64 values. Exact f32-bit comparison now matches the native
contract; the regression rejects an adjacent f32 knot. Both frozen and jointly
fine-tuned manifests extract all 984 terms from the completed product parent.
The failed pre-training attempt is preserved; no optimizer steps were run.

Resumed pretrained dispatch uses a new audit directory. It reconciles completed
children by their plan, build, cell, exposure and final checkpoint hash, and
counts already running native children against the same worker limit after
checking their live command lines. It never silently relaunches a missing or
failed process. Pre-training failure receipts remain alongside repaired attempts.
