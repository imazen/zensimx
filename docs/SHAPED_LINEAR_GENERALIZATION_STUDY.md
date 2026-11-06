# Shaped-linear fitting bounds and memorization study

Registered November 6, 2026, before any scientific fit. Owner instruction:
write a comprehensive plan and execute it in zensimx. This extends the
[monotone architecture study](MONOTONE_ARCHITECTURE_EXPERIMENTS.md) and reuses
the [forest study](TREE_REGRESSION_STUDY.md)'s admitted preparation.

## Questions and prior evidence

Separate optimization failure, restrictions imposed by monotonicity, missing
nonlinear shaping, source generalization, distortion transfer and memorization
capacity. A low training loss is not generalization; failed fitting is not a
feature-information ceiling. One surviving deep-network output coefficient
does not make the upstream representation linear.

[R12](../benchmarks/r12_linear_head_2026-11-04.md) fitted revision-7 dense and
sparse nonnegative projections using Adam and the older mixture. Its output
stage/dial remained nonlinear, and its runs did not certify a global optimum.
The recent 129-run architecture/training study did not include a directly
solved linear/basis baseline for the new four-human plus SSIM2 preparation.

For fixed basis B, fit damage d=1-quality by minimizing
`0.5 * sum_i a_i (B_i w - d_i)^2 / sum_i a_i + lambda/2 * ||w||^2`.
Compare free coefficients with w>=0. There is no intercept: zero damage maps
to quality 1. A learned scalar multiplier is folded into w, not factorized
into another optimization variable. This is a convex quadratic. Its optimum
is a bound for this fixed basis, weighting and objective only; it is not an
upper bound on Spearman correlation or all possible predictors of these inputs.
Regularization paths establish regularized optima, not an unregularized ceiling.

## Frozen data and roles

Reuse `HumanSSIM2-depth-v1.trial.json`, SHA-256
`80ec04c1cba1ef851192e63c0a03687a9da019ae483713c978c1dc0daa7ec915`,
and PREPARED SHA-256
`04963be1bf38e792def29d05fa382a49687614c829265d29561e02d4d0cc85f8`.
There are 24,852 prepared rows, 232 fitting reference-pixel hashes and 21
development hashes. All approved human distortions remain, KonFiG is F only,
and CID22 SSIMULACRA2 is the sole synthetic teacher. No pixel extraction,
additional label population, secret/final holdout or serving change is included.

Verify every table/key/manifest hash and the existing keyed metadata. Group
folds by decoded reference identity globally across human and teacher rows;
also union known Part A/B/source aliases and exact distorted-image duplicates.
Keep each connected component on one side of a split. Report any additional
overlap discovered before proceeding; do not silently change frozen roles.
Near-duplicate candidates are diagnostics requiring inspection, not automatic
deletion. Preserve original arrays and all failed attempts.

Use three outer folds within FIT and two inner folds within each outer training
partition. Balance component assignment across populations deterministically,
using labels only for population membership, never quality values. Record every
membership. All scales, knots, label maps and model selection use the training
complement only. Report each outer fold independently; pooled OOF rankings are
supplementary because different folds can have different score calibration.
Development is already exposed and is a final descriptive comparison, not a
selection surface. KonFiG development has one source.

Each human population has equal fitting mass, references share that mass
equally, and the teacher has the mass of one human population. Use the existing
training-only per-population range target map as the primary assumption; retain
raw MOS/F/JND labels for native per-population rank reports. Do not interpret
these mapped scales as a perceptually calibrated common dial.

## S1: projection and shaping matrix

| Family | Basis | Coefficients |
|---|---|---|
| free_raw | 576 measured raw inputs, including context | unrestricted diagnostic |
| free_products | 492 damage inputs plus 492 fragility products | unrestricted diagnostic |
| positive_products | same 984 inputs | nonnegative |
| positive_shaped_direct | three monotone segments per damage input | nonnegative |
| positive_shaped_products | three monotone segments per direct/product input | nonnegative |

Use training-weighted RMS scales without centering. Segment knots are the
training positive-value median and 90th percentile, with explicit deterministic
handling of zeros/equal knots. Basis segments are bounded first/second intervals
and an unbounded final tail; they are continuous, nonnegative, increasing and
zero at zero. Record exact f32 basis parameters. Learned segment coefficients
permit increasing or decreasing sensitivity. No knot search uses outer labels.

For every family test lambda={1e-6,1e-4,1e-2}. Select lambda within that family
using equal-human-population mean of pooled and within-source signed SROCC in
the inner folds; retain both components and all candidates. Ties select larger
lambda. The registered matrix is 90 inner fits, 15 outer refits, 45 full-FIT
selection-fold fits and five full-FIT refits: 155 quadratic fits. Deterministic
solves have no invented initialization-seed replication.

Extend the existing Rust Gram/ridge/box-solver owner. Reuse its Cholesky solve
for free and active-set systems. Emit independently recomputed KKT residuals,
primal objective and a strong-convexity objective-gap bound for lambda>0. A
maximum iteration count is not convergence. Require finite inputs, symmetric
Gram, feasible coefficients, KKT infinity residual <=1e-7 and reported objective
gap <=1e-6. Preserve and diagnose failures without weakening these tolerances.
Use independent SciPy/NumPy small-system references and planted correlated,
singular, active-bound and zero-column fixtures. Those are numeric oracles,
not replacement fit owners. Document Gram assembly and floating-point scope.

The canonical ZNPR baker/ Predictor own f32 projection arithmetic. The basis
adapter is private research preprocessing and must have an independent reference
parity test, zero/coordinate/extreme probes and explicit input-schema pins.
These are opaque-vector experiments until pixel-domain integration is qualified.
Keep f64 optimum predictions and baked f32 predictions separate; quantify drift.
Do not silently quantize a certified optimum and retain its original certificate.

## S2: calibration and objective diagnostics

For the 15 outer and five final models, separately fit an anchored monotone
output calibration on training predictions using the existing native spline
owner. Keep the raw projection as the ranking result; report calibration/ties
and native parity independently. Calibration cannot reverse an inverted pair.
No held-out label enters knots, ranges or calibration coefficients.

Add fixed within-reference pair-difference least-squares controls for the two
product-projection families on all three outer folds and full FIT (eight fits).
Use the same registered pair list and group masses for both. This is a convex
ranking-oriented surrogate, not the neural RankNet objective. Compare native
pair loss and rankings, and explicitly retain that objective mismatch. Diagnose
unregularized feasibility with training pair-order constraints only where a
verified solver can certify it; never label ridge fit error an unregularized bound.

## S3: source learning curves

Use positive_products and positive_shaped_products with fixed lambda=1e-4.
For each outer fold, retain 25%, 50%, 75% and 100% of its training components,
stratified across populations, retaining all rows of selected components.
Use two declared hash-order seeds for the fractions; 100% is fitted once.
Recompute weights, maps, RMS and knots on the retained set. The fixed outer
test sources never move. This is 42 fits. Compare with a matched row-budget
control that subsamples variants while retaining sources (18 additional fits,
three fractions x two families x three folds, one fixed row seed).
Report distinct sources and rows, train/outer error and rank, and coefficient
stability. Additional variants of the same source are not new-source evidence.

## S4: memorization and model-family controls

For all five linear families at lambda=1e-4, run two label permutations with
two independent seeds in each outer fold: within population, and within
population/distortion family. This is 60 fits. Permute only training labels;
held-out labels remain real. Report how much ranking structure the conditional
permutation preserves, native fit error, actual-target outer performance and
permuted-target training performance. Derive weights/maps after permutation.
Keep teacher rows in their own population; no cross-scale label permutation.

Compare the fixed prior Extra Trees configuration and shared [128,64] ReLU
control on the same three outer folds, true labels plus both permutation modes,
with two training seeds: 36 fits. Freeze their settings before outer reads.
The neural budget is 120 x 5,000 draws per fit, fixed final epoch, with unchanged
constraints. Label permutations are separately admitted research data; original
prepared tables remain immutable. These are matched-data diagnostics, not
matched-compute comparisons with a direct convex solve.

Random-label fitting demonstrates capacity to memorize, not actual memorization
on genuine labels. Failure can reflect monotonicity conflicts or optimization
failure. Compare constrained/unconstrained models and retain these distinctions.
Monotonicity does not constrain the ordering of incomparable feature vectors.

## S5: distortion transfer, shortcuts and source influence

Use the two positive families at lambda=1e-4 on each outer fold. Exclude each
of eight predeclared families from its training complement: AVIF (aom/cld_avif/
vis_avif together), JPEG (mozjpeg), JXL (libjxl), WebP (cld_webp), kadid_01,
kadid_20, tid_01 and KonFiG jitter. Match exclusions across human/teacher
rows. This is 48 fits. Report excluded-family performance on unseen sources
and performance on retained families; the unchanged full-family fit is control.
Do not infer that other dataset codes represent the same distortion mechanism.

Fit a reference-context-only linear baseline on each outer fold (three fits,
lambda=1e-4), using only the reference operands identified by the pinned fx2
declaration. A useful score here flags a dataset/source-distribution shortcut;
it does not establish that another model uses that shortcut.

For each positive full-FIT family, choose the two highest training residual-mass
components per population and refit without each selected component (at most
20 fits; shared selections are deduplicated and counts frozen before refitting).
Report coefficient/prediction changes and per-source influence. Selection uses
training residuals only, and these are sensitivity diagnostics, not independent
generalization tests. Audit exact feature duplicates, nearest cross-role/source
neighbors and label disagreement with no automatic deletion or relabeling.

## S6: reporting, retention and acceptance

Freeze each executable matrix and every input/source/binary hash before launch.
Report completed/required fits by stage, failures, optimality certificates,
native precision differences, signed pooled/within-source/family panels and
unmapped target errors. Use the Rust panel/scatter owner. Undefined small-family
correlations stay explicit. Report paired whole-source bootstrap intervals
(2,000 draws, seed 20261106), conditional on exposed sources, without claiming
multiple-testing correction. Compare fold metrics rather than using a pooled
OOF correlation as the sole generalization estimate.

Keep a live PID/artifact progress record. Independent cells run concurrently
under run-heavy with explicit per-job RAM/thread budgets; start with four
workers and observe aggregate demand before increasing it. NumPy/BLAS Gram
assembly has an explicit thread limit; optimization and baked prediction are
Rust. Benchmarks are not part of this study. Check disk before retaining large
matrices; reuse pinned bases/Grams and archive raw outputs. Generated data/model
binaries go under `/mnt/v/output/zensim/`, with verified storage mirrors.

Completion means all registered stages are measured or have a specific verified
blocker, original failures remain visible, tests/independent numeric gates pass,
and results plus limitations are written in the result record and chat. No
serving promotion, API expansion, relaxed scientific gate or final-test access
is authorized by a favorable research result.

Method references: [convex least squares](https://www.cvxpy.org/examples/basic/least_squares.html),
[grouped validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data),
[random-label capacity](https://arxiv.org/abs/1611.03530),
[selection overfitting](https://www.jmlr.org/papers/v11/cawley10a.html).

## Implementation status

The private `bake_dial_refit linear-study` adapter reuses the Rust Gram owner's
Cholesky solver, with active-set nonnegative fitting and independent KKT/gap
recomputation. The canonical baker and Predictor evaluate the f32 projection;
the existing native identity-pinned spline owner performs optional calibration.
Small correlated/singular/zero-column and active-bound fixtures pass. All 41
bake-tool tests and scoped clippy pass. No scientific fit has run at this point;
the independent Python numeric reference and input/fold admission remain gates.

The eight independent NumPy/SciPy numeric systems pass, as do three controller
leakage/grouping tests and signed/raw native panel checks. Preparation retains
232 FIT and 21 development source components with no cross-role overlap. The
155 S1 fits completed and passed their numerical certificates. Their results
remain preliminary while the registered negative controls execute.

Comparator preparation reuses strict D1/revision/key admission. Each derived
training table records its original parent, ordered subset and label permutation;
the original tables remain immutable. An actual population-permutation admission
passed for all nine groups, changing 15,066 training labels and zero outer labels.
The neural comparator retains human rank-only and teacher rank/MSE objectives;
its data match the linear study but its objective differs. It selects epoch 119
regardless of development logs. Seeds are the registered two seed values for
initialization, with sampling seed equal to initialization plus 100,000,000.
