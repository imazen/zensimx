# Outer refit admission and matched controls

Implementation policy fixed before any outer refit or outer response inspection.
The frozen completion protocol, original matrix and all earlier cells remain
unchanged. This resolves their selected-setting placeholders and expands the
matched fresh controls required by the pretraining comparison.

Require complete paired inner base selection for all twelve families. If the
fold triggers rescues, require all four rescue choices for every neural family
and reproduce the base/rescue choices from their pinned human-only selection
panels. Teacher agreement never chooses settings or families. Require every
triggered doubled-exposure cell, including its paired ReLU and fresh controls,
to finish native evaluation and convergence diagnosis before outer admission.
No best-checkpoint files or outer responses enter these decisions.

Use the selected base/rescue human composite at the common primary exposure to
choose the analytic, product-shaping and separate-shaping finalist roles. Keep
the fixed ReLU, quadratic and tree roles. Family ties within 1e-12 use the original
protocol's family order. Extension panels diagnose training sufficiency; they do
not provide a second family search with unequal exposure budgets. Preserve both
primary and extended results in the comparison.

Outer refits inherit the selected setting and optimizer choice. A family required
to extend uses its fixed doubled schedule, including the paired ReLU control.
Pretrained families independently relearn shapes on the outer training complement
before fitting the fresh head: 120+120 epochs normally, 240+240 when extended.
Continuous families use 240 or 480 epochs. No inner-fitted scales, knots or learned
shapes transfer into the outer fit. The final scheduled checkpoint is used.

The original nine `fresh_control_outer` cells remain: a 120-epoch fresh ReLU at
the selected ReLU setting/optimizer for each fold and paired seed. They cannot
stand in for a differently configured pretrained model. Add conditional fresh
controls for each distinct `(setting, optimizer choice, phase-B epochs)` required
by the four pretrained families. Share one control only when all three fields
match; seed pairs always retain their registered million offset. If a tuple equals
the original fresh-control tuple, reuse that exact cell. Additional control IDs
are deterministic and registered with their paired families before fitting.
This adds necessary comparisons; it does not remove any original cell or award
extra tuning opportunities.

After the complete registered optimization/extension attempts, remaining collapse
or improving curves must remain explicitly inconclusive. Outer measurement may
report such candidates with that limitation; it cannot turn them into converged
negative results. Unexecuted diagnostics, missing numerical gates, incomplete
pairing or altered provenance block outer admission. No final holdout is opened.

The nonneural controls can wait on admission with
`just shaping-completion-watch-outer-controls STUDY BINARY_DIR tree` and the
corresponding `linear` command. Each family processes the three outer folds
sequentially; the two families can run concurrently within the host RAM budget.
Each fitting process has a 6 GiB memory cap and one compute thread. Waiting does
not open response tables. The watcher verifies the full existing outer admission,
selected settings and seed coverage, then calls the existing control trainer,
which rechecks admission after SSD reservation. Queue ownership is per family and
fold; it does not serialize unrelated training. Existing queue records or cell
attempts require explicit reconciliation rather than automatic retries. Plan,
build manifest or driver changes during the wait also stop the queue.
