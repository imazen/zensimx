# Paired rescue selection

This implementation policy is recorded before scientific rescue fitting. It
implements the frozen completion protocol without changing its tuning budget.

For each neural family and outer fold, compare the inner-selected base setting
against all four registered rescue choices. Each choice requires both inner
folds and all three paired seeds: 30 final-checkpoint reports total. Use the same
equal-human pooled/within composite; teacher agreement never selects a choice.
Missing, failed, nonfinite or diagnostically incomplete cells prevent selection.

For scores within 1e-12 of the largest choice mean, prefer the base, then the
registered rescue order: half initialization, double initialization, constant
learning rate, final-quarter consolidation. The choice order is fixed before
rescue results exist. Use one qualified evaluation representation across the
comparison; repaired ReLU reports must share their evaluator identity.

Selection alone does not admit outer evaluation. Preserve the selected seeds'
convergence evidence. Continuing improvement requires the protocol's registered
doubled-exposure comparison with paired controls; collapse or other diagnostic
failures still require investigation. No failed original run is overwritten,
and no final holdout is opened by this procedure.
