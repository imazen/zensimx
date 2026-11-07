# Shaped-linear fitting bounds and memorization

Protocol: [registered study](../docs/SHAPED_LINEAR_GENERALIZATION_STUDY.md).
Source implementation starts at `29e350a6` (native solve), `21bea651` (matrix),
`5e037122` (comparators) and `4fe1e464` (evidence reduction). Resource-startup
correction and row-space audits: `dd117206`.

All 390 registered fits are complete: 354 quadratic fits, 18 Extra Trees
controls and 18 shared-ReLU controls. All numerical and native feature gates
passed for the applicable constrained models. Twenty calibration attempts gave
18 measured curves and two explicit anchor refusals. The 2,000-draw paired
source bootstrap, exact-feature/nearest-source audits, all family exclusions
and all source-influence refits are complete. Artifacts are retained under
`/mnt/v/output/zensim/shaped-linear-generalization-20261106/`; every artifact has a verified cold-readback storage mirror.

## Data and comparison

The frozen four-human plus CID22-SSIMULACRA2 recipe contains 24,852 observations.
Its 232 FIT source components comprise 55 KADID, 14 TID2013, six KonFiG,
18 CID22-A25 and 139 teacher sources. The 21 development components remain
separate. Reference pixel hashes, known aliases and exact distorted duplicates
are kept together. No component crosses FIT/development roles.

Three outer FIT folds and two inner folds per outer training partition select
lambda within each linear family. Each population has equal training mass;
each source shares its population's mass equally. Target ranges, RMS scales,
segment knots and calibration use training rows only. The common reported
composite is the equal-human-population mean of pooled and within-source
**signed** SROCC, averaged across outer folds. These are exposed research data,
not a new final test. The comparator settings were frozen from the prior study;
no new comparator hyperparameter search uses these outer scores.

## Projection results

| Family | Training composite | Unseen-source composite |
|---|---:|---:|
| Free 576 raw inputs | 0.920275 | 0.881225 |
| Free 984 direct/product inputs | 0.926689 | 0.881300 |
| Positive 984 direct/product inputs | 0.878418 | 0.859236 |
| Positive shaped direct inputs | 0.889582 | 0.863715 |
| Positive shaped direct/products | 0.896138 | 0.865817 |
| Extra Trees, mean of two seeds | 0.987419 | 0.911052 |
| Shared [128,64] ReLU, mean of two seeds | 0.899916 | 0.885679 |

Paired source-resampling differences (conditional 95% intervals):

- Extra Trees minus shared ReLU: +0.025373 [+0.014977, +0.034836].
- Extra Trees minus free products: +0.029753 [+0.019871, +0.036296].
- Shared ReLU minus free products: +0.004379 [-0.007818, +0.014321].
- Shaped positive products minus positive products: +0.006581 [-0.007387, +0.018944].

These intervals resample whole sources within each population and outer fold,
then average fold metrics. They are conditional on these fits and exposed
sources, without a multiple-testing correction. The free projection is
competitive with this shared-ReLU control; the aggregate shaping gain is
uncertain. CID22-A25's shaping difference is +0.036831 [+0.007275, +0.057536].

The fixed-lambda pair-difference control raises positive-products outer ranking
from 0.859198 to 0.867900. Its training composite is 0.883353. Free products under
this objective reach training 0.938588 and outer 0.870948. These are separately
registered least-squares pair objectives, not RankNet, and their lambda remains
1e-4. Objective choice matters alongside model capacity.

Shaping is not a uniform improvement. Positive shaped products raise the CID22-A25
composite from 0.845914 to 0.882745, while KADID, TID2013 and KonFiG decline
slightly. The free-product baseline improves KonFiG relative to free raw inputs
but loses CID22-A25 ranking. The final report retains every population and fold.

Free coefficients and unconstrained trees are diagnostic models. They are not
accepted identity/monotonicity-preserving serving candidates. Positive direct
and product bases vanish when damage inputs vanish, even when reference context
is nonzero. Input segment slopes remain nonnegative while their sensitivity
can increase or decrease across intervals. Their native feature probes are
separate from pixel-domain qualification.

## What the bounds establish

Every quadratic solve passed the registered KKT and strong-convexity gap gates.
The maximum KKT residual is `1.4679842723234403e-14`; the maximum regularized
objective-gap bound is `8.539334766667133e-21`. These are numerical certificates
for the supplied PSD Gram and double-precision coefficients, not interval-
arithmetic proofs or certificates for rounded model weights.

Independent row-space losses agree with the Gram results within
`7.66424684134348e-14` across all 354 fits. The largest observed change in
weighted half-MSE from native f32 prediction is `4.515659207671163e-09`;
the largest individual scientific-row score change is `0.005335942363512203`.
No performance or memory extrapolation is implied by these arithmetic checks.

All 27 matched free-versus-positive product systems use identical Grams. The
positive restriction raises the regularized objective in every case, by
0.002466231424598786–0.006930616563996794. Thus these fits identify restrictions
of this basis/coefficient class, rather than a local optimizer failure. They
do not bound SROCC, all monotone functions, or all information in the features.
The stronger tree result is already evidence against calling the linear result
a general feature ceiling.

The exact-feature audit finds one four-row KADID cluster with identical image
pair keys and feature vectors but human scores 91.22340425531915,
85.90425531914893, 85.90425531914893 and 89.8936170212766. All four stay in one
source/fold. This alone gives a genuine but very weak unregularized weighted
half-MSE floor of `3.279455944689098e-08` on the frozen mapped FIT labels for any
deterministic function of these features. Nothing was deleted or relabeled.
There are no exact feature duplicates crossing source components or FIT/
development roles; nearest-feature candidates remain diagnostics, not proof
about perceptual image duplicates.

## Overfitting and memorization controls

Population permutations preserve essentially no original training ranking:
mean label-to-label composite `-0.0017810980284897443`. Permuting within
population and distortion family preserves `0.1920081322821666`; this second
negative control deliberately retains family-level structure.

| Model/control | Shuffled-training composite | Real unseen-source composite |
|---|---:|---:|
| Free products, population shuffle | 0.242040 | 0.180387 |
| Positive products, population shuffle | 0.011494 | 0.704914 |
| Positive shaped products, population shuffle | 0.023175 | 0.698357 |
| Extra Trees, population shuffle | 0.943560 | 0.205364 |
| Extra Trees, conditional shuffle | 0.924708 | 0.497598 |
| Shared ReLU, population shuffle | 0.006077 | 0.599914 |
| Shared ReLU, conditional shuffle | 0.100525 | 0.659089 |

Trees can fit much of the shuffled-label structure; this demonstrates capacity,
not proof that their real-label result is memorization. Positive models fit
population-shuffled targets poorly while retaining a quality ordering on real
outer labels. Their damage-feature prior and restrictions matter; rank
correlation alone should not be mistaken for accurate score calibration.
Native raw errors and scatter diagnostics are retained alongside rankings.
Failure to fit shuffled labels in one constrained model class is not a theorem
that monotonicity prevents memorization.

Source learning curves improve with added sources. Positive-products outer
composites are 0.840412, 0.849916, 0.853779 and 0.859198 at 25%, 50%, 75% and
100% of training sources (nominal fractions, rounded up within population); shaped products give 0.851041, 0.855571, 0.860677 and
0.866127. The 100% rows use fixed lambda=1e-4, rather than the separately
selected S1 lambda. Matched row-budget controls retain all sources, and all
60 curve fits include basis-slope stability diagnostics on common training-only
raw-feature quantiles.

The 12-input fx2 reference-context-only models are constant within each source, so their
within-source correlations are explicitly undefined. Their pooled outer
correlations range from -0.167101 to 0.152083 across human populations/folds.
This does not prove that another model ignores context shortcuts.

All 48 family exclusions and 20 source-influence refits finished. Exclusions
match human and teacher codec families. The influence refits hold the selected
full-FIT lambda fixed; family exclusions use their registered lambda=1e-4.
Held-family and retained-family panels remain separate. Removing AVIF lowers
unseen-source AVIF SROCC on CID22-A25 by 0.023577 for positive products and
0.046050 for shaped products; teacher changes are -0.039779 and -0.056378.
Removing KADID-20 lowers its held-family correlation by 0.010563 and 0.019259.
These are descriptive three-fold means, not uncertainty-adjusted claims.

The selected-source removals move individual development predictions by up to
15.619113 score points for shaped products; one removal moves its development
predictions by 1.310266 points on average. These are sensitivity diagnostics,
with target maps/basis parameters refitted on the retained training rows.
Full source hashes,
selected residual-mass sources and coefficient/prediction changes are retained.

## Calibration and verification

Twenty native identity-pinned output calibrations were attempted: 18 passed,
two free-raw models were refused because their last fitted input knot exceeded
the identity score 100. The refusals remain visible; no anchor was relaxed.
Measured calibration outputs never reverse the raw ordering. Monotone output
calibration can introduce ties but cannot repair an inverted pair.

The independent numeric gate passed eight NumPy/SciPy systems, including
correlated, singular, active-bound and zero-column cases. All 41 bake-tool tests,
scoped Rust clippy, three controller grouping/leakage tests, signed/raw native
panel checks and actual nine-group comparator admission passed. Shuffling that
admission fixture changed 15,066 training labels and zero outer labels.

The first entry point initialized NumPy before its one-thread setting. Replaying
that exact import order under its four-job wrapper measured four BLAS threads;
four Gram workers could therefore use sixteen threads. Native fitting children
were limited to one thread. The original driver and actual Grams remain intact.
Subsequent entry points set limits before NumPy import; the corrected audit
reports one thread. This is recorded rather than retroactively relabeling runs.

## Verified retention

The archive contains 4,403 artifacts before its mirror receipt, totaling
10,391,487,000 bytes. Every file passed fsync plus client cold-readback SHA-256
verification in the mirror at
`/mnt/tower/output/zensim/shaped-linear-generalization-20261106/`.
Receipt SHA-256:
`4a797ece6ebe382be7460e6c8cabe16554547377cca743a597d12db17f17f846`.
The checksum inventory SHA-256 is
`642428868aada79b69ee4358b14e01a279b7c068e6dd1f0bdea2e478de7b8882`.

The retained closure pins report SHA-256
`5288a8f09262168f7f9df6506a1c81b62fdb57a7c7e9a20153233e87486d3397`
and source-bootstrap SHA-256
`82901d7992c2be4cc0a1f923bbb73d4eb404a58492a69a7d2cb3ffaf9647157d`.
Reproduction sources were captured at `c48eecca08c1`; this later pointer update
does not modify the closed experiment tree. All original input and model hashes
were rechecked before archive creation. Scientific fits did not access a final
holdout or promote a serving profile. CI is manual; the recorded checks are local.
