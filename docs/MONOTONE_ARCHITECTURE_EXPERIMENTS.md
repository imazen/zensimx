# Monotone architecture and training experiments

Owner authorization, 2026-11-06: write the durable experiment list and pursue it
in zensimx. This is research; no serving profile or public API changes are
authorized by a quality result. The existing
[activation/curriculum study](MONOTONE_ACTIVATION_CURRICULUM_STUDY.md) and
[tree study](TREE_REGRESSION_STUDY.md) remain the owners of their frozen results.

## Question and literature

The completed shared ReLU and SmoothHinge comparison kept convex distance in
the expanded input coordinates. It did not test whether removing convexity,
while retaining monotonicity and identity, improves perceptual prediction.
Nonlinear feature products mean this is not a convexity statement about pixels.

[Amos, Xu and Kolter (2017)](https://proceedings.mlr.press/v70/amos17b/amos17b.pdf)
establish input convexity from convex nondecreasing activations and nonnegative
hidden connections. Input convexity does not make weight training convex, and
their unconstrained direct input connections do not guarantee monotonicity.
[Sartor et al. (2025), requested v1](https://arxiv.org/html/2505.02537v1)
instead alternate saturation sides and introduce activation switches. The
three-hidden-layer construction is not a finite-width fit guarantee. Their
tabular experiments do not establish image-quality performance.

Additional original sources: [mixed constrained activations](https://proceedings.mlr.press/v202/runje23a.html)
and [Lipschitz monotonic networks](https://arxiv.org/html/2307.07512).
The [requested overview](https://www.emergentmind.com/topics/input-monotonic-neural-networks)
is a bibliography, not the authority for its malformed displayed equations.

## Frozen comparison contract

- Reuse the admitted four-human plus CID22-SSIM2 preparation, revision 7,
  fx2 direct/products. No additional population, pixel extraction, secret
  holdout or final-test read. Keep all approved human distortions, KonFiG F
  only, and SSIM2 as the only synthetic teacher.
- The older mixed parent has a documented KonFiG Part A/B alias. The prepared
  F-only fit contains 504 rows on six Part A sources; development has 84 on
  src50_parta. Verify decoded-source hashes before execution. Do not alter
  frozen parent tables or describe the one-source development set as broad
  new-source evidence.
- Freeze table, key, manifest, role, script, source, binary and configuration
  hashes. Keep initialization, sampling and dropout RNG streams distinct.
- Architecture comparisons keep group masses, losses, examples, scaler and
  pair budget fixed. Compare at matched width and report actual trainable
  parameters and expanded bake dimensions; width is not a compute match.
- Diagnostic seeds: independent pairs (1101,101), (1107,200000101),
  (1129,400000101). First screen: 12 epochs x 5,000 draws, final epoch 11.
  Long comparison: 120 x 50,000, fixed final epoch 119. Short fits cannot be
  compared as equal-budget alternatives to the existing long fits.
- Tune only with source-grouped TRAIN folds. Development is already exposed;
  report it as development, never use it to choose settings or checkpoints.
  Preserve failed arms and every trial, including negative results.
- Use native training, ZNPR v3 baking, canonical prediction and Rust panel
  statistics. Report signed pooled and within-source ranks, distortion-family
  panels, scatter and tails. No composite invented after seeing results.

## Architecture queue

| ID | Experiment | What it separates | Initial status |
|---|---|---|---|
| A0 | Shared 128->64 ReLU and SmoothHinge controls | Reproduce the current constrained owner | Existing implementation; new matched fits required |
| A1 | Mixed uncapped hinges and bounded ramps/caps after the certified first-layer sums | Increasing versus decreasing sensitivity | Implement native shared head; reuse ramp algebra |
| A2 | One pre-activation-switch block after the first layer | Nonconvex monotonicity at matched semantic depth | Implement native forward/backward and expanded ReLU bake |
| A3 | Two switch blocks after the first layer | Three hidden layers versus two | Same gates, separate capacity comparison |
| A4 | Positive-slope piecewise-linear spline head | Learned curvature without weight-sign switching | Implement native piecewise gradients and bake |
| A5 | Lipschitz-bounded residual head | Signed expressive subnetwork plus certified monotone residual | Verify norm bound and anchor before fitting |
| A6 | Existing single-layer monotone-ramp owner | Reuse control for the cost of an extra layer | Existing implementation; matched fitting required |

Do not substitute ordinary SiLU/GELU into a sign-constrained monotone path:
their derivatives can be negative. Leaky ReLU with slope in [0,1] remains
convex and is an optimization control, not a nonconvexity experiment.

## Native construction gates

Keep the first-layer max-pair algebra intact. Individual members need not be
monotone; any new head must consume their sum or preserve exactly tied paths.
Nonnegative inputs and zero identity are part of the contract.

For a switch, h(x)=ReLU(W+ x+b)-ReLU(W- x+b), where W+=max(W,0) and
W-=min(W,0). Both branches share b, so h(0)=0; both input derivatives are
nonnegative. Allow both bias signs: b<=0 with nonnegative inputs would erase
the second branch and recover the convex restriction. A cap is
min(z,t)=z-ReLU(z-t), z,t>=0; a bounded ramp is
ReLU(z-a)-ReLU(z-a-t), a,t>=0.

Compile supported constructions to existing dense/ReLU ZNPR operators rather
than add an unqualified scoring path. Explicit subtraction must retain the
same rounded operands at identity. Folding an anchor into a scalar output
bias is not sufficient evidence of bit-exact identity.

Required gates: independent finite-difference gradients (including anchors),
known nonconvex monotone fitting tasks, rejected nonmonotone controls, exact
native f32 zero identity, coordinate-order probes including near identity,
finite extreme inputs, and training/bake/runtime agreement. Check structural
constraints after optimizer updates and in baked f32 weights. Sampled tests
supplement an algebraic proof; they do not replace it. Preserve unsupported
mode refusals and unchanged baseline replay. No public serving API additions.

## Training queue

| ID | Experiment | Fair comparison |
|---|---|---|
| T1 | Initialization scale, signed switch bias spread, dead/saturated-unit and gradient diagnostics | Separate init seeds from sample order; TRAIN-only scale estimates |
| T2 | Coupled L2 {0,1e-5,1e-4}, then AdamW | Do not call coupled Adam L2 weight decay; keep LR schedule fixed |
| T3 | Last-hidden dropout {0,.05,.15} | Same mask for rank pair and anchor; tied max-pair masks; p=0 replay |
| T4 | Random, easy-first, hard-first and family-block order | Identical precomputed event multiset; same 25% random consolidation tail |
| T5 | Source/family-balanced mass and replay | Separate objective-weight changes from order; record realized counts |
| T6 | Teacher mass {0,.5,1,2}, joint versus teacher warm-up then human consolidation | Matched cumulative teacher/human exposure, with explicit replay |
| T7 | Fixed LR versus cosine/restarts and late lower-LR consolidation | Long enough to cross existing 50-epoch restarts; matched total budget |

Difficulty comes from TRAIN-only human-pair margins within each population,
not incomparable pooled MOS/F/JND scales. Distortion severity is a separate
axis from pair difficulty. Family curricula require complete keyed metadata,
source grouping, a nonzero replay floor, and realized mass accounting. A
plateau may be approximation error, saturation, a projection boundary, noisy
supervision or optimization; do not label it a local minimum without evidence.

## Execution and reporting

Extend the existing private shared-depth implementation and experiment driver.
Use an opt-in, hash-bound research manifest rather than enlarge public APIs.
Independent fits may run concurrently, each single-threaded with explicit RAM
limits. Benchmark isolation remains zenbench's responsibility; no global
exclusive training lock. Preserve all outputs and verified storage mirrors.

The durable result record must identify completed/required cells, failed
gates, exact source/binary/model identities and remaining experiments. A plan,
scalar reference, short pilot and full-budget fit are distinct deliverables.
Update this queue and its linked evidence as each stage actually completes.
