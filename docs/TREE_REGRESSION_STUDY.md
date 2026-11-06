# Forest and boosting study — 2026-11-06

The owner requested actual random-forest trials and new techniques in zensimx.
This study compares six tree configurations plus two human-only ablations.
It uses the admitted four-human-population/CID22-SSIMULACRA2 preparation from
the [activation study](MONOTONE_ACTIVATION_CURRICULUM_STUDY.md). It does not
read additional populations, alter frozen runs or select a production model.

## Registered comparison

All arms read the same 492 direct distortion features in the pinned fx2
declaration. Products are excluded from every arm. Inputs retain f32 values;
there is no fitted feature scaler, clipping or imputation. Missing, nonfinite
or negative declared values fail admission. This is a common-input comparison
between tree methods, not an architecture-only comparison with the 984-input
neural models. Tree thresholds do not need StandardScaler.

| Arm | Trees/iterations | Split regularization | Feature fraction | Monotone |
|---|---:|---|---|---|
| rf | 128 | minimum leaf 8; at most 1024 leaves | sqrt | no |
| rf_wide | 128 | minimum leaf 8; at most 1024 leaves | 0.33 | no |
| extra | 128 | minimum leaf 8; at most 1024 leaves | 0.33 | no |
| hgb | 160 | 31 leaves; minimum leaf 20; L2=1; learning rate=.05 | 1 | no |
| rf_mono | 128 | same as rf | sqrt | decreasing in all inputs |
| hgb_mono | 160 | same as hgb | 1 | decreasing in all inputs |
| rf_human | 128 | same as rf | sqrt | no |
| hgb_human | 160 | same as hgb | 1 | no |

Forests bootstrap rows; Extra Trees uses randomized thresholds without
bootstrap. OOB is not used as independent validation: variants of one source
would otherwise appear on both sides. HGB automatic early stopping is disabled.
These are fixed configurations, not a claim of exhaustive hyperparameter tuning.

Each human population receives equal objective mass; within a population each
reference receives equal mass and its rows share that mass. The teacher has the
mass of one human population, except in the two human-only arms. Weights are
normalized to mean one for the included fitting rows. scikit-learn 1.9 uses
weights as bootstrap sampling probabilities for RF; use max_samples=None so
the draw count is the number of fitting rows, independent of weight magnitude.

Raw MOS/F/JND/teacher numbers are not pooled as one interval scale. In each
fitting partition, a separate reference-balanced weighted empirical mid-CDF
maps each population's quality labels to [0,1]. Held-out labels never enter
these maps. This imposes equal population quantiles, an explicit assumption
that can be wrong; report per-population results, not a calibrated common dial.
This squared-error objective differs from the neural ranking objective.

Two deterministic source-grouped folds inside FIT assess sensitivity. Source
names are sorted by a pinned SHA-256 key within population and alternated;
KonFiG Part A/B aliases share a source. Record every membership. Fit each arm
on each complement with seed 20261106. Then fit all eight arms on all FIT rows
with seeds 20261106, 20261107 and 20261108, without selecting configurations
from development results: 40 fits total. Internal development is already
exposed design data, not an independent final test. KonFiG development has
only one source. Seed spread is not a confidence interval over new images.

## Native evidence and identity

Reuse the canonical ZCTH exporter and `CorruptionHead::decision_function`
through `corrhead_parity`; do not introduce another production tree walker.
An adapter exposes fitted RF/Extra Trees nodes to the existing HGB exporter.
RF leaf values are divided by tree count before ordered accumulation. Compare
that reference with sklearn's sum-then-divide prediction and record the actual
rounding difference. The exported arithmetic must agree with Rust at **0 ULP**
on every evaluated row, plus independent synthetic threshold/anchor probes.
The existing probability/fire-set checks remain enabled. Statistics use the
canonical Rust `panel` with signed, within-reference and full scatter outputs.

These are **opaque-vector numerical experiments**. ZCTH v2 supplies the needed
f32 tree arithmetic, but its legacy header does not declare revision 7.
Provenance records the actual revision-7 table identity and this incompatibility.
Never attach these artifacts to a pixel profile or present them as serving
bakes. Native tree arithmetic can be checked now; revision-7 pixel admission,
`BakeScorer` integration and runtime qualification remain missing. No existing
revision-admission expectation is relaxed to obtain a positive result.

For every final fit also test an anchored variant: subtract each tree's value
at the zero vector from every leaf and set the baseline to 1. Zero inputs then
score exactly 1. This changes only a global offset mathematically, so it does
not establish better ranking. With decreasing constrained trees and
nonnegative inputs it also bounds quality above by 1. It does not make an
unconstrained forest monotone. Test coordinate increases against randomized
nonnegative vectors, report unconstrained violations, and fail constrained arms
on a violation. Test near-zero ray plateaus: exact identity alone does not
guarantee useful near-lossless sensitivity. These are feature-space probes,
not image/pixel correctness or perceptual proofs.

## Evidence and execution

Persist input hashes, source identities, fold memberships, label transforms,
package versions, estimator parameters, fitted models, export artifacts,
all-row native parity logs, signed/within-source/scatter panels and failures.
Save full-fit in-sample panels separately from development and out-of-fold
panels. No cross-population pooled correlation is a selection metric.
Run independent cells concurrently with explicit thread and memory budgets.
Archive the complete closure, including the reused native evaluator and data
pins; do not publish images, human rows or model binaries into git.

Forest fitting greedily chooses splits rather than using neural gradients.
Dropout/activation/epoch-order interventions therefore do not transfer directly.
Source balance, leaf size, randomized features, tree averaging and stage count
are the relevant controls. Boosting fits residual stages in sequence; shuffled
rows are not a distortion curriculum. Family-specific weighting still requires
the canonical metadata join identified by the activation study.

Primary references: [Breiman's Random Forests paper](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf),
[RF API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html),
[Extra Trees API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html),
and [histogram boosting API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).
Installed implementation at registration: scikit-learn 1.9.0; version-specific
bootstrap behavior was checked against its local source.
