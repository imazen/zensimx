# Owner-approved reduced shaping study

The November 7, 2026 approval reduces the core matrix from 2,599 to 1,281 fits.
The original PLAN, MATRIX, partition memberships, failed attempts and completed
results remain immutable. This amendment changes coverage, not quality thresholds
or the human-only selection statistic. Interrupted fits are not replayed and
comparisons they prevent remain incomplete.

| Stage | Required fits |
| --- | ---: |
| Primary inner screen | 816 |
| Fresh inner controls | 72 |
| Fresh outer controls | 9 |
| Targeted optimizer rescues | 144 |
| Finalist outer fits | 48 |
| Source fractions | 96 |
| Population shuffle | 48 |
| Human-only ablations | 48 |
| Row fractions, full-data selection and full-data refits | 0 |

Before rescue outcomes or outer responses are read, select four neural roles
from the complete original inner screen in each outer fold: ReLU, best analytic,
best product shaping and best separate shaping. Retain the existing family order
for ties. Each role receives half initialization and constant learning rate,
both inner folds and three paired seeds. Frozen shortlists prevent a later
rescue outcome from changing which architectures receive additional tuning.
An incomplete candidate pool cannot produce a best-family claim.

Outer fits retain these four neural roles plus Extra Trees and shaped additive.
Use all three paired seeds for stochastic models and one deterministic additive
fit. Keep the existing fresh outer controls; any additional matched controls
required by different pretrained schedules must be reported separately from the
1,281-cell core. Preserve existing convergence classifications and numerical
qualification gates.

Source curves use fractions 0.25 and 0.5 with source-order seed 20261107, reusing
the full-fraction outer models. Shuffled-label controls use population shuffling
and permutation seed 20261107. Human-only ablations retain their original scope.
Prioritize outer quality and human-only comparisons, followed by source curves
and shuffling. Measure cost using actual outer models.

Keep completed and active cells outside the reduced scope as additional evidence.
They do not create replacement obligations or count toward reduced core completion.
Report reduced rescue coverage, the absent row/source separation, one source
order and permutation, and the absence of a full-data optimized final bake.
No poorly optimized model is thereby established to be architecturally inferior.

`shaping_reduced_scope.py` registers the approval, proposal, original registration
and this policy by checksum. It enumerates the fixed subset and resolves the
remaining original cell IDs from a complete inner-only shortlist. Registration
does not itself launch fits or relax an admission gate.
