# Full-FIT selection and descriptive development evaluation

This resolves the frozen protocol's full-FIT placeholders before any full-FIT
selection or refit. The original 192 selection cells and 16 final fits remain.
No development or final-holdout score enters the decisions below.

Require all three outer admissions to reproduce their complete inner evidence,
including triggered rescues and exposure extensions. Choose the three learned
finalist families by averaging each family's inner-selected primary-exposure
human composite across the three outer training complements. Keep the fixed
ReLU, tree and shaped-additive roles. Ties within 1e-12 use protocol family order.
These are globally fixed families for full-FIT selection; per-fold finalists in
the generalization study remain unchanged.

For each neural finalist, fix an optimizer choice before the new four-setting
search. Compare only choices measured in every outer fold's inner search; average
their inner human composites equally across folds. Ties use base, then the
registered rescue order. If any fold did not trigger rescues, only choices with
complete three-fold evidence are eligible. These scores were measured at each
fold's selected base setting; they do not establish a causal optimizer effect.

Use the longest registered exposure required for that family across the three
outer admissions, preserving supervised pretraining and fresh-head accounting.
This conservative fixed schedule does not establish convergence for the new
full-FIT runs. Keep unresolved convergence limitations explicit. Nonneural
controls retain their original objectives and four-setting searches.

Run all four original settings on each of the three registered source-component
selection folds, with all three paired model seeds (one deterministic quadratic
solve). Refit scales, maps, knots and supervised pretraining on each training
complement. Select by the equal-human signed pooled/within-source composite,
averaging folds and seeds equally; ties within 1e-12 choose the lower setting.
Use final scheduled checkpoints only. Do not select families, optimizer choices
or schedules again using these selection-fold results.

These selection folds reuse the existing outer FIT memberships, as required by
the frozen matrix. Their reports are scoped `full_fit_only`; they cannot enter
the earlier nested comparison or its family/settings decisions. Their results
are tuning evidence for the final refit, not another unbiased evaluation.

Refit the selected settings from scratch on all FIT rows, with TRAIN-only
preprocessing and all paired seeds. Evaluate the already-exposed development
partition descriptively, identifying it as exposed. Never use development for
checkpoint selection, tuning, shortlisting or convergence-driven retraining.
No untouched final holdout is opened. Preserve training diagnostics and report
insufficient optimization as inconclusive, including in descriptive results.
