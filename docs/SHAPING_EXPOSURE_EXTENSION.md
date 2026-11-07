# Doubled-exposure convergence comparison

Implementation policy fixed before affected fitting. The original completion
protocol and its matrix remain immutable. This adds the explicitly required
comparison when an inner-selected neural model continues to improve.

Resolve all ten neural families' complete paired base/rescue selections for an
outer fold before registering its extension. Reproduce each selection from its
hashed reports. Any selected seed with the registered loss-improvement trigger
(including the log-rounding boundary) or same-cycle human improvement trigger
requires that family's extension across both inner folds and all three seeds.
One-coefficient collapse alone does not imply continuing improvement; it remains
an unresolved diagnostic even when no extension is triggered.

Include the selected ReLU control at doubled exposure whenever any family needs
an extension. Keep every family's selected setting, optimizer choice, population
masses, seed pair, training rows, preprocessing and numerical gates unchanged.
Do not retune or pick a checkpoint using extension curves. Preserve original
12-million-draw models and reports for the paired comparison.

Continuous fits restart from their registered initialization and run 480 epochs
of 50,000 draws. Pretrained fits independently learn shapes for 240 epochs, then
discard the head and train the fresh head for 240 epochs, with the existing
million-offset seeds. Frozen and joint variants retain their respective shape
update rules. Total exposure is 24 million draws. Cosine restarts remain every
50 epochs; constant learning rate remains constant; consolidation starts at the
final quarter of each phase. Do not extend a saved optimizer state or borrow a
prefix whose schedule differs.

For every extended pretrained setting/optimizer choice, include a fresh unshaped
240-epoch ReLU control with the same fresh-head seed pair and optimizer choice.
Deduplicate controls only for exactly identical settings and choices. This
12-million-draw fresh control is reported separately from the continuous
24-million-draw control; it does not pretend discarded training conveys learning.

Evaluate the final scheduled checkpoint. Retain all checkpoints and report
training loss, human/teacher panels and capacity diagnostics at the original
scheduled epochs plus 299,349,399,449,459,479 when the phase is long enough.
The existing final-forty-loss and last same-cycle criteria apply unchanged.
Continued improvement or unresolved collapse is reported as inconclusive, never
as proof that an architecture has reached its limit. Registration and fitting
alone do not admit outer evaluation or consume a final holdout.
