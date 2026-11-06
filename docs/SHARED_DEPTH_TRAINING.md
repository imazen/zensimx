# Constrained shared-depth experiment

The requested experiment adds a shared second hidden layer to the constrained
plain trainer: inputs → 128 → 64 → 1. The first width remains configurable;
the second follows the existing depth convention, max(first/2, 8). The
single-hidden-layer path and its arithmetic remain unchanged.

Keep the existing first-layer ReLU and max-pair projection. In every column of
the new intermediate matrix, tie the coefficients of each max pair. Project
intermediate weights nonnegative, second-layer biases nonpositive, final
weights nonpositive and final bias to the identity pin. Each paired first-layer
sum retains its monotone max construction; nonnegative mixtures and monotone
activations preserve that order. Both hidden layers output zero at identity.

Compare ReLU with fixed tau=1/8 SmoothHinge only in the new second layer.
First-layer ReLU is part of the max-pair proof. The second-layer activation
must be nonnegative, nondecreasing and zero for nonpositive input. The existing
SmoothHinge owner has those properties; no new activation formula is needed.

Reuse the plain sampler, rank/MSE losses, scale-only normalization, Adam,
regularization, validation and checkpoint selection. Admit only the explicitly
supported constrained K=1 CPU path; refuse unimplemented auxiliaries instead
of silently dropping them. Existing unsupported-mode tests remain unchanged.
The native bake uses three ordinary dense layers and the existing activation
encoding, without a new serving API or model format.

Required evidence: finite-difference gradients through all three matrices,
first-pair ties after optimization, identity and monotonicity after f32 baking,
training/runtime agreement for both activations, unchanged single-layer
regressions, and native feature-subset/densification parity. Match the registered
human-plus-single-teacher trial's rows, seeds and final-epoch schedule. These
are development comparisons, not new final-test reads or model adoption.

## Validation scope

The first focused checks pass for both activations: all weight/bias gradients
against central finite differences, fused versus unfused first-layer Adam,
projection after optimizer updates, exact f32 identity, sampled nonnegative-input
monotonicity and f64-training/f32-bake agreement. The latter permits f32 rounding
(8 epsilon times max(1, magnitude)); it is not a bit-exact f64/f32 claim.
Only f32 research bakes are admitted on the new path.

The existing trainer regression pass has 160 passing library tests, with its
one existing ignored test unchanged, and 65 passing CLI tests including the new
CLI admission fixture. Full workspace all-target/all-feature Clippy passed.
The script suite has 70 passing tests and two errors in historical part-B arm
pin loading; both errors reproduce on the unchanged `083480f2` test source.
No arm pin or expectation was altered. The new depth argv fixture passes and
verifies that stripping just the four depth tokens restores the old command.
Native executable smoke, unchanged-baseline replay, and the scientific fits
remain required before reporting a depth result.

The follow-up [activation and curriculum study](MONOTONE_ACTIVATION_CURRICULUM_STUDY.md)
separates analytic identity/monotonicity checks, native optimization pilots and
full-budget candidate trials. It includes anchored activations, dropout, curvature,
matched-exposure order controls, replay and dataset/family mass accounting.

The [tree regression study](TREE_REGRESSION_STUDY.md) trials random forests,
Extra Trees, gradient boosting, monotonic controls and peer-assisted stacks on
the same admitted human rows. It separates predictive quality from pixel-model
qualification and compares completed neural assessments on identical rows.
