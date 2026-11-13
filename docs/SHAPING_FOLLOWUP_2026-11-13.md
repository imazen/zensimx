# Bounded shaping follow-up

The owner approved all three directions on November 13, prioritizing paired
tree/product error analysis. The previous 1,461-fit study stays closed. This
phase has **15 new fitting conditions**, three paired seeds per condition:

| Direction | New conditions | Fits |
|---|---|---:|
| Human hard cases | Joint product shaping, bounded cross-fitted human rank-loss weights | 3 |
| Optimization | Negative-magnitude output initialization; same initialization with negative-Softplus output parameterization | 6 |
| A recipe stability | Joint shapes at 0.1 times head LR with identity penalty; one common frozen learned bank | 6 |

Human-recipe trials retain the completed product finalist's outer-0 recipe,
240 epochs, paired initialization/sampling seeds, selected setting and constant
learning rate. This is a **single development-fold screen**, not a new full
three-fold comparison. Its test sources were exposed in the previous study.
The matched controls already exist; no completed fit is replayed. Unchanged
default behavior requires native neutral-control qualification before reuse.

Hard-case identification uses only outer-0 TRAIN sources. For each of its two
inner folds, read all three seed-matched product/tree validation predictions at
their previously selected settings. Every row's predictors exclude its source
component. Average prediction percentile ranks across seeds separately for each
inner fold and population. Flag a human row if its product absolute rank error
is in the top decile and the tree has smaller error. Assign weight two to
flagged rows and one otherwise, then normalize to mean one within each source.
The arithmetic mean of the two row weights multiplies the human RankNet term.
Keep sampling, targets, group selection masses, teacher rank/MSE terms and all
other losses unchanged. Weights must remain in [0.5, 2]. No tree prediction
becomes a training target. The only synthetic supervision remains CID22 SSIM2.

The optimization arms compare the same all-active negative-magnitude output
initialization under projected updates versus `weight = -softplus(latent)`.
The latter applies the exact chain rule to the existing effective-weight
gradient, including L2. Mid-layer and max-pair constraints stay intact. Export
ordinary effective weights through the existing consumer format. Interpret
active-coefficient counts alongside activation, gradient and prediction
diagnostics; one coefficient alone does not prove loss of functional capacity.

The A arms retain the original one-hidden-layer architecture, full frozen A
recipe, 120 base plus 60 continuation epochs and original three seed indices.
The stability arm combines shape LR scaling 0.1 with a mean squared departure
penalty from identity of 0.001. It tests that combined intervention; it cannot
separately attribute its effects. The common frozen bank is the coordinatewise
median of the three **already trained base-phase** A banks, with identical
feature identities, knots and anchors required. It is shared across new seeds,
never taken from the other recipe's feature columns. Reused bank-training
exposure is reported separately rather than counted as newly executed fits.
Freeze shapes in both phases. These hypotheses are registered without a
hyperparameter search or checkpoint selection.

Compare fixed final checkpoints once on the existing development views. Report
each human dataset and seed, pooled/within-source correlation, and error tails.
Keep synthetic agreement separate. A's exposed CSIQ/LIVE views remain exposed;
LIVE clean10 is mandatory because the previous joint A trial hurt that subset.
No final holdout, model promotion, release or additional teacher is authorized
by this phase. Fresh confirmation remains closed until a candidate is frozen.

Completion requires the 15 terminal fit records, fixed-checkpoint development
panels, a concise comparison, bounded crash snapshots and checksum-readback
archives. Failed attempts remain visible and preserved. No extra tuning phase
starts automatically. Native tests, clippy, an unchanged-default binary check
and consumer identity/monotonicity/finite-output probes form the finite gate.

## Paired residual discovery

Cached outer predictions from the completed study establish complementary
errors. Across the three folds and three paired seeds, trees have smaller
absolute rank error on 85.68% of the product model's worst-decile KADID rows
and 82.08% of its worst-decile TID2013 rows. KonFiG's descriptive fixed equal-rank
blend scores 0.89283 pooled SROCC, versus product 0.88175 and tree 0.88191.
CID22 human favors product: 0.87812 versus tree 0.85867 and blend 0.87260.
These are means of population/fold/seed rank comparisons, not the previous
study's pooled-plus-within primary composite. A rank blend needs population
context and is **not a standalone serving score**. This analysis is hypothesis
discovery, not independent confirmation or proof that extra interactions help.

The source owner is `scripts/rev4_featpot/shaping_residual_followup.py`;
generated predictions and row-level evidence remain outside git.

## Execution checks

Native trainer tests and clippy pass. A short real-input comparison reproduces
both checkpoints from the old trainer byte for byte after removing only
`zentrain.repro`, for both one-layer and shared-depth owners. The post-fit
evaluator reproduces the completed product control's development composite and
all three exposed A control points exactly. A short, fully finalized CLI run
also passes the consumer max-pair conversion, identity, finite-output and
damage-order probes with the new negative-Softplus parameterization.

Partial library checkpoints carry binary provenance until the trainer CLI
finalizes them. The private max-pair converter requires the finalized UTF-8
provenance. Retain refused partial conversions and use completed CLI outputs
for consumer qualification; do not change a running checkpoint or weaken that
admission gate. Final scientific evaluations still require the registered last
epoch, independently of the short format qualification.

The finite finish owner, `shaping_followup_finish.py`, checks actual native PIDs
and terminal records, collects completed fits, evaluates each final checkpoint
once, packs quiescent artifacts and performs durable checksum readback. It
starts no fitting or selection work and exits after the 15-result closure.
Errors become an explicit `needs_repair` state. Reused input archives are named
by manifest identity; checkpoint bundles are compressed without changing their
contents. No final holdout is opened by this owner.
