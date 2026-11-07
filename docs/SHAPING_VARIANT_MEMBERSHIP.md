# Completion-study variant membership details

This implementation specification is frozen before any learning-curve, shuffle
or human-only fit. It expands the existing registered matrix without changing
primary fits, source components, seeds, fractions, objectives or selection.
No target scores or outer results are read during membership registration.

Source curves order the outer-training component identifiers by SHA-256 of
UTF-8 `seed|component`, with the identifier as a collision tie-break. Keep the
first floor(fraction × component count), including every row of each selected
component. The two registered orders are nested across fractions. Record actual
source and row counts rather than claiming an exact percentage after rounding.

Row-budget controls order outer-training row indices by SHA-256 of UTF-8
`seed|index`, breaking collisions by numeric index. Retain the first ordered
row of every source component, then fill to floor(fraction × row count) from
the same ordering. Refuse a budget smaller than the source count. This keeps
all training sources, exact row budgets and nested subsets.

Do not redraw a subset when a training population is absent. Record it and pass
its explicit absence to the training-view owner. The fixed outer evaluation
still reports every test population. For an absent population, pooled and
within-source rank metrics remain defined when the data support them; mapped
and inverse-mapped errors are unavailable because no training-only map exists.
Never fit a substitute map on validation or outer rows. Primary fits keep their
existing strict requirement for every population.

Human-only cells remove CID22-SSIM2 training rows before scales, knots, target
maps or supervised pretraining. Their outer test stays fixed, including teacher
agreement as a separate descriptive panel. Shuffle cells retain the full outer
training complement; the existing permutation owner permutes training labels
within population or population/distortion family with the registered seeds.
Actual outer labels are never permuted. Settings and finalist families remain
subject to the inner-selection and convergence gates before execution.
