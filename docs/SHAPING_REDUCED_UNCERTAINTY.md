# Uncertainty for the approved reduced shaping study

This implements the owner-approved 1,281-fit amendment. It preserves the original
registration and its 88 matched comparison specifications. Register a separate
reduced comparison directory before reading any outer predictions.

The primary analysis requires all 48 outer fits: six inner-selected roles across
three source-disjoint outer folds, three paired seeds per stochastic role, and
one deterministic additive fit per fold. Role identities can resolve to different
candidate families in different folds; record those identities rather than
presenting the role as a single fixed architecture. Reproduce all three outer
admissions before opening predictions. Missing or undefined fits prevent the
complete comparison; never omit individual seeds or invent scores.

There are 24 matched analyses: each of six roles compares its full-training outer
fit against 25% and 50% source-fraction fits, the population-shuffled fit, and the
human-only fit. Use source order/permutation seed 20261107. Removed row curves,
other orders/permutations and unselected-family comparisons are outside the
reduced uncertainty queue. Preserve their historical artifacts. Fresh controls
retain their matched reports and admission checks; the original four
family-specific fresh-control bootstrap comparisons are not part of this reduced
queue, because those families need not be shortlisted across all three folds.

Reuse the original 2,000 paired, population-stratified whole-source draws, seed
20261107, including repeated-source multiplicity. Average rank performance over
paired model seeds and then equally over outer folds. Report each human dataset,
the equal-human composite and CID22-SSIM2 agreement separately. Keep the teacher
out of the human headline. Preserve signed correlations, paired differences,
percentile intervals and source membership checks. The bootstrap is conditional
on fitted models and observed sources; it does not estimate training-seed or
source-order uncertainty. Do not read the final holdout or select on outer scores.
