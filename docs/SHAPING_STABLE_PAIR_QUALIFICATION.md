# Qualified stable first-layer exports

Four original diagnostic checkpoints have a damage-order reversal just above the
unchanged 1e-5 native gate. Native direct-max export removes each reversal while
preserving every stored parameter, scaler and unit-order bit. Original failed
reports and models remain preserved. This extends the existing ReLU/Softplus
numerical repair; it introduces no training, tuning, checkpoint selection or new
human/teacher labels.

Seven further constrained families require an explicit certificate before using
this representation: monotone SiLU and all product/separate shaping variants.
Without it, the existing unsupported-family checks still refuse those families.
Ordinary unconstrained SiLU, trees and additive models remain outside this rule.

`shaping_stable_pairs.py` verifies the four original failing checkpoints and four
fixed setting-zero, seed-one, inner-0-0 pretrained models. Every case must pass
the same 128 paired probes, exact identity and finite extreme-ray checks. Its
complete TRAIN and inner-validation population comparison must reproduce the
stored original predictions and show bit-identical unmarked-runtime predictions.
The native baker independently validates the tied-pair structure and parameter
bit preservation. Population score deltas are measured, not tuned or treated as
quality improvements. Qualification reads no outer or untouched final holdout.

The sealed certificate pins the original plan, corrected runtime, native probe
and population receipts, original failed gates and qualification source. Drift
refuses admission. Qualifying a family applies the same export rule to every
paired setting, seed, fold and scheduled diagnostic, including future rescues
and refits. A per-cell repair choice is prohibited. The seven families contribute
504 primary conditions and 2,880 final/diagnostic report positions; these are
scalar reevaluations, not additional scientific fits.

Versioned stable-export base-selection paths preserve the earlier original
decisions. Selection still requires complete paired reports and diagnostic
evidence in one representation and one qualified evaluator per family. All
numerical gates, source boundaries, tuning budgets and failure preservation
remain unchanged. Implementation and qualification do not prove that those
supplemental reports or downstream experiments have completed.
