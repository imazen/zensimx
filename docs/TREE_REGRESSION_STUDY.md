# Forest and boosting study — 2026-11-06

The owner requested actual random-forest trials and new techniques in zensimx.
**Pre-fit amendment:** the owner clarified that predictive quality from cached
features, peer scores and human judgments is the priority, with a fair tuning
effort. No scientific fit had started when this amendment was made. The search
below supersedes the initial 40-fit screening table later in this document.

The executed search has 38 configurations and three source folds (114 fits),
followed by three full-FIT seeds for the best configuration within each of five
families (15 fits): 129 fits. RF searches leaf size 2/16, sqrt/.33 feature
subsampling and three input sets; Extra Trees searches leaf size 2/16 across
those input sets; HGB searches 15/63 leaves across them. The sets are fx2 direct
492, all 576 explicitly measured raw columns (f0..227 and f372..719), and fx2's
984 direct/product inputs. Products use f32 multiplication of cached values.
Only the direct set is assigned all-decreasing constraints. Other inputs may
contain reference context; an all-zero vector is not a pixel-identity claim.

For each unconstrained family, four further raw-feature configurations compare
CDF/range label mapping and teacher mass 0/.25/1, with leaf size 8 and .33 feature
sampling (RF/Extra) or 31 leaves (HGB). These are targeted ablations, not a full
factorial. Range mapping uses only fitting-partition minimum and maximum.
Forests have 128 trees and at most 2048 leaves; HGB has 200 stages, learning
rate .05 and L2=1. Constrained RF and HGB controls use direct inputs, CDF targets
and teacher mass 1. All settings are in `tree_study.candidates()` before fits.

The within-family selection objective equally averages the four human
populations and three folds, with equal pooled and within-reference signed
SROCC. Teacher accuracy is reported but does not select the model. Missing
panels fail selection. All finalist settings are frozen before development is
scored. No single cross-population pooled correlation is optimized. Native
packing is a finalist arithmetic check, not a prerequisite for a search fit.
The original registration below explains the control choices; its 40-fit grid,
two-fold schedule and fixed-finalist rule are superseded by this amendment.

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

## Matched peer and source audit

`tree_study_report.py` joins only the admitted preparation's pair keys. The
frozen revision-7 bank keys provide decoded reference/distorted pixel hashes,
distortion labels and severity knobs. The decoded reference audit checks both
FIT/development and all inner folds across population boundaries, not merely
matching names within a dataset. Historical GMSD/GMSM rows must match both pixel
hashes and their successful digest-check flag. SSIMULACRA2 and butteraugli
3-norm use the October 28 E21 cache, exact pair keys and exact image paths.
Duplicate keys with different scores fail the join. Missing admitted human
rows fail the comparison. Peer tables' human labels are never imported.

The four peer baselines receive the same signed/within-reference/scatter panels
as the trees, for fitting and development separately. Their scores are not
inputs to the initial 129-fit search. SSIMULACRA2 on the separate CID22 training
images is the only synthetic supervision in that search. Other peers are
zero-shot comparators. Detailed distortion/severity metadata is saved for
subgroup diagnosis; dataset identity is not substituted for distortion type.

## Peer-assisted extension, registered before its fits

The owner's cached-feature/peer-score clarification also warrants a separate
human-only stacking comparison. Twelve settings compare RF, Extra Trees and
HGB using either the four peer scores alone or 576 raw features plus those
scores. RF/Extra search minimum leaf 2/16, 128 trees, 2048-leaf cap; feature
fraction is 1 for four peers and .33 for 580 combined inputs. HGB searches
15/63 leaves, 200 stages, minimum leaf 20, learning rate .05, L2=1. The same
within-FIT CDF mapping, equal reference/population weighting and source folds
apply. No teacher or human development label fits these models.

Three folds per setting give 36 fits; the best setting per family/input set
then receives three full-FIT seeds (18 fits): 54 total. Selection uses the same
composite and never development. The metadata join and initial study inputs
are immutable. This experiment was registered after peer baseline panels and
48 initial tree fold cells were available, before initial tree development
results. It does not claim a newly untouched final test.

These stacks require the peer metrics at inference; their quality cannot be
credited to a feature-only zensim model. Compare peer-only versus combined
inputs to measure the added predictive value of the cached features. Native
opaque-vector parity covers all final fitting and development rows. No
pixel-serving, identity, runtime-cost or monotonicity qualification is implied.
