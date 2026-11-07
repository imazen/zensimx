# Activation and learned-shaping source-fold protocol

Registered before new scientific fitting, November 7, 2026 UTC. This follows
[the recovered evidence](SHAPING_COMPLETION_EVIDENCE.md) and uses the native
[qualified implementations](ANALYTIC_HEAD_IMPLEMENTATION.md). It does not promote
a serving model. Existing study results are historical comparisons, not cells
in this matrix.

## Data and decision boundaries

Reuse the immutable shaped-linear preparation and its exact source-component
outer/inner memberships: three outer FIT folds and two inner folds per outer
training complement. Verify original PLAN, PREPARED, table/key/manifest and
membership hashes before preparing views. Keep aliases and exact-image duplicate
components intact. Preserve all approved human distortion types, KonFiG F,
CID22-human and CID22 SSIM2 as the sole synthetic teacher. No final holdout is
opened. Exposed development is descriptive only and is excluded from training
commands, checkpoint selection, tuning and shortlisting.

Every transform, knot, target range, pretraining fit and hyperparameter choice
belongs to the current training complement. Use the existing revision-7 fx2
492 direct damage inputs and 492 same-cell fragility products. Neural input
scales are the native training standard deviations without centering. For each
shape, choose raw-domain positive-value q50/q90 knots on that complement;
if equal, double the lower knot; an all-zero column uses [1,2]. Store finite
positive increasing f32 knots and reject unrepresentable widening. These are
raw-domain knots, not the older linear study's RMS-normalized basis parameters.

Use the existing reference-balanced population masses and native within-reference
human RankNet plus teacher RankNet/MSE recipe. Keep teacher agreement separate
from the human headline. Quadratic and tree controls retain their explicitly
different reference-balanced mapped-target MSE objective; they are not called
objective-matched neural controls. Refit range target maps on training rows.
All primary comparisons use the same rows and population masses.

## Families and tuning opportunities

Native shared networks have first width 128 and second semantic width 64.
Keep the established max-pair constraints for constrained models. The ordinary
SiLU diagnostic uses signed first-layer weights and no max-pair ties; report
its independent parameter count and do not claim identity or monotonicity.

| Family | Input and head |
|---|---|
| relu | Unshaped shared ReLU control |
| softplus | Anchored Softplus, positive hidden weights, signed anchors |
| silu_monotone | Anchored SiLU restricted to nonnegative preactivations |
| silu_diagnostic | Ordinary unanchored SiLU and unrestricted first layer/output |
| product_joint_identity | Identity-initialized g(d) and g(d*f), jointly learned, ReLU head |
| product_pretrained_frozen | Same shapes learned on training rows, then frozen before a fresh ReLU head |
| product_pretrained_joint | Same pretrained shapes, jointly refined with a fresh ReLU head |
| separate_joint_identity | Identity g and h, g(d) plus g(d)*h(f), jointly learned, ReLU head |
| separate_pretrained_frozen | Supervised pretrained g/h, frozen before a fresh ReLU head |
| separate_pretrained_joint | Supervised pretrained g/h, jointly refined with a fresh ReLU head |
| shaped_additive | Native positive three-segment quadratic solve over direct/products |
| extra_trees | Existing Extra Trees owner, raw measured inputs, native export gate |

Each family receives four registered settings. Neural tuples are
`(initial_lr, weight_L2) = (1e-3,0), (1e-3,1e-5), (3e-4,0), (3e-4,1e-5)`.
The same 50-epoch cosine-restart schedule and initialization owner apply to
all four. Shape parameters have no penalty; report their norms, ranges and
zero counts separately, since their scale freedom can change the effect of
weight L2. The zero-L2 cells are essential to interpreting that confound.

For analytic heads, the four tuples additionally use temperature/anchor settings
`(.25,fixed,0), (1,fixed,0), (1,learned,.25), (4,learned,1)` in that order.
Softplus's final anchor is -1 instead of 1; SiLU's monotone anchor stays
nonnegative. This is a four-point joint search, not a factorial causal estimate
of temperature, initialization and regularization. Report every setting.

Shaped-additive lambda is `{1e-7,1e-6,1e-4,1e-2}`. Keep the existing KKT and
objective-gap certificate gates unchanged. Extra Trees uses 128 trees,
max_leaf_nodes=2048, and `(min_samples_leaf,max_features)` in
`{(2,.33),(8,.33),(16,.33),(8,1.0)}`. Reuse the prior estimator and native
prediction gate; do not silently replace its fit owner with another algorithm.

Use paired `(initialization,sampling)` seeds `(1101,101)`,
`(1107,200000101)`, `(1129,400000101)`. Tree seeds are the initialization
seeds. Deterministic quadratic solves run once per setting/partition, without
invented seed replication. Neural and tree selection averages both inner folds
and all three seeds; quadratic selection averages the two folds. The selection
metric is the existing equal-human-population mean of signed pooled and
within-source SROCC. Keep both components. Ties within 1e-12 choose the lower
setting index. Teacher agreement never breaks a tie.

## Pretraining, exposure and checkpoints

The primary neural budget is 12 million pair draws: 240 epochs x 50,000.
Every primary cell has that exposure, counting supervised pretraining. Plain
and identity-joint models train continuously for 240 epochs. Pretrained models
use 120 epochs to learn shapes jointly with a ReLU network on the same training
complement, discard that network, then train a fresh network for 120 epochs
with frozen or trainable shapes. This pretraining is a supervised nonlinear
fit, not the earlier certified quadratic solution. No old outer-fitted shape
is reused on an inner partition.

Reuse the matching identity-joint model's epoch-119 shapes as pretraining
artifacts only after verifying prefix equivalence: same raw rows, scales,
knots, settings, initialization and sampling streams, and no schedule depending
on total epochs. Both frozen and joint descendants share that artifact. Their
fresh-head seed is initialization+1,000,000 and their sampling seed is
sampling+1,000,000. Record the prefix hash and both stages' exposure and cost;
shared pretraining costs are not silently counted as free. If a rescue schedule
breaks prefix equivalence, fit its 120-epoch pretraining stage independently;
never reuse a nonmatching prefix.

Also fit a fresh unshaped 120-epoch ReLU control with those second-stage seeds
and settings. This isolates the fresh-head comparison. It has six million draws
and is reported separately from the 12-million-draw primary control; the discarded
unshaped first stage carries no information and need not be recomputed to pretend
that exposure was useful.

Use the final scheduled epoch, never the trainer's best-validation file, for
all selections and comparisons. Save every epoch checkpoint and curve so
prefix reuse and convergence are auditable. The frozen/continued shape state
must be embedded in every checkpoint and pass native compaction. Outer results
are read only after that outer fold's settings are selected using inner data.
Fit all registered families on outer training complements with their selected
settings and all three seeds (one quadratic solve).

## Convergence and optimization diagnosis

Compare training objective and human/teacher panels at epochs 49,99,119,149,
199,219,239 for continuous fits, and the corresponding available phase-B
checkpoints for pretrained fits. Measure active output coefficients, projection
hits and gradients for analytic heads; inspect native layer weights for ReLU.
Record hidden activation activity, first-layer gradients, and shape/gate
parameter ranges. One surviving coefficient is a diagnostic, not a conclusion
that the network is linear or at a feature-information ceiling.

An inner candidate needs investigation if its final 20-epoch mean training loss
improves by more than 1% over the preceding 20 epochs, its last two available
same-cycle human composites differ by more than .002 in the improving direction,
or gradients/weights become nonfinite. A one-coefficient collapse additionally
triggers investigation even if those curves plateau. Diagnose using training and
inner validation only. If any neural family triggers investigation, give every neural family the same
four registered rescue choices, starting from that family’s inner-selected base
setting: half initialization scale, double initialization
scale, constant learning rate, and final-quarter learning-rate consolidation.
Keep the selected base setting's other choices fixed; use all three paired seeds
and both inner folds. Rescue fitting gets the same total exposure, including any
pretraining. Implement missing control support before its fit; do not substitute
another head. Retain original failures. Select among base and rescue cells on inner data. This makes the rescue wave
equal-opportunity across neural families; deterministic and tree controls retain
their separately reported four-setting search.

If the selected model still improves at its exposure limit, rerun it at doubled
exposure with paired controls and retain the original. Do not call an inadequately
trained family a negative result. Report inconclusive convergence where evidence
remains insufficient; such cells are not marked scientifically completed merely
because the process exited.

## Generalization, memorization and final comparisons

Report every human population separately, teacher agreement separately, pooled
and within-source signed ranks, native errors, distortion-family panels, and
per-source results. Use 2,000 paired source-component bootstrap replicates with
seed 20261107 for 95% percentile intervals; keep rows of a component together
and pair candidate/control replicates. Report all outer folds; pooled OOF is
supplementary because calibration differs between folds.

Choose finalists using inner results only: ReLU, Extra Trees, shaped-additive,
the best analytic family, the best product-shaping family and the best
separate-shaping family. Break family ties by the listed order. For these six
roles, on each outer fold use training-source fractions 25%,50%,75%,100% with
hash-order seeds 20261107 and 20261108; reuse the full fraction. Keep the outer
test fixed and refit preprocessing/pretraining. Use all three paired model seeds.
Add row-budget controls at 25%,50%,75% with sources retained and row seed
20261109. Settings stay those selected by the full inner protocol; label these
as fixed-setting learning curves, not freshly tuned small-data optima.

For the same finalists and outer folds, permute only training labels within
population and within population/distortion family, using permutation seeds
20261107 and 20261108 and all paired model seeds. Refit target maps after
permutation. Report permuted-target training fit and actual-target outer fit,
including residual structure in conditional permutations. This tests capacity
and generalization, not a claim that genuine-label success is memorization.

Refit finalists without the teacher on each outer fold with all paired seeds,
using human-only scales/maps/knots and pretraining. Keep the selected full-recipe
settings fixed and state that choice. No teacher examples enter these fits.
Finally select full-FIT settings with three source-component selection folds,
then fit finalists on full FIT and report the already-exposed development
partition descriptively. No selection uses development.

Benchmark finalist native exports through zenbench on an eligible non-training
benchmark host, with source/binary/model hashes and fixed image dimensions,
thread settings and CPU dispatch. Report model bytes, trainable versus expanded
stored coefficients, qualification limits and actual measured cost; coefficient
counts are not latency measurements.

## Execution and completion evidence

The machine-readable matrix expands the rules above before any scientific fit.
Hash it, the source/binaries, data membership and this protocol in PLAN.json.
Any semantic amendment is dated before affected fits and preserves old cells.
A one-epoch resource pilot uses the first inner ReLU cell and is excluded from
scientific selection; it determines RAM, time and checkpoint-disk admission,
not hyperparameters. Share immutable partition views between seeds/settings.
Run independent seeds concurrently within measured memory and aggregate CPU
headroom. Keep at least 30 GiB SSD free; archive completed waves and rehash their
destination before reclaiming local outputs. Preserve failures and provenance.

Completion requires every registered cell or triggered diagnostic to have an
inspected result, native qualification, honest convergence classification and
verified archive. Keep completed/required counts and actual process identities
in the durable progress record. Implementation tests and existing archives do
not substitute for these scientific fits.
