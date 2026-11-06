# Random forests, boosting and peer stacks — 2026-11-06

Trees can predict these human judgments well. The source-fold-tuned random
forest and Extra Trees improve KADID and TID2013 development correlations over
the shared-ReLU human+SSIM2 neural comparator. Their CID22-A pooled scores are
lower. No candidate wins every population, and no serving model is adopted.

All **189 registered scientific fits completed**: 150 source-fold search fits
and 39 full-fitting-population fits. This includes the feature-only search,
peer-only/feature-plus-peer search, and matched monotonicity controls. There
were no omitted failed cells. [Protocol](../docs/TREE_REGRESSION_STUDY.md),
[compact measurements](tree_study_2026-11-06.json),
[paired source intervals](tree_study_2026-11-06.bootstrap.json).

## Predictive quality

Signed development SROCC, arithmetic mean across three declared seeds. The
neural rows are completed 120-epoch models, not the earlier short pilots.
These are attained-quality comparisons with different objectives/search
budgets, not architecture-only or equal-compute ablations. HGB is deterministic
across these seeds under the chosen settings; its repeated fits are not three
independent stochastic replications.

| Model | KADID | TID2013 | KonFiG F | CID22-A |
|---|---:|---:|---:|---:|
| Random forest, features only | 0.930113 | 0.925117 | 0.945353 | 0.888417 |
| Extra Trees, features only | 0.930996 | 0.928569 | 0.944514 | 0.892787 |
| Histogram boosting, features only | 0.944073 | 0.914086 | 0.948041 | 0.882372 |
| Shared-ReLU MLP, human + SSIM2 | 0.893798 | 0.904876 | 0.953191 | 0.904434 |
| One-hidden-layer MLP, human only | 0.905255 | 0.907207 | 0.958523 | 0.874998 |
| One-hidden-layer MLP, human + SSIM2 | 0.893609 | 0.897943 | 0.939887 | 0.906024 |
| SSIMULACRA2, zero-shot | 0.798920 | 0.869443 | 0.947838 | 0.894113 |

The three unconstrained tree winners all use the 576 raw cached features,
per-population fitting-only range target maps and teacher mass 1. Forests use
128 trees, minimum leaf 8, feature fraction .33 and a 2048-leaf cap. Boosting
uses 200 stages, 31 leaves, minimum leaf 20, learning rate .05 and L2=1.
This is what the source-fold comparison selected; development did not choose
the settings. The other 492-direct/984-product configurations remain in the
full search record.

CID22-A within-reference ranking is closer than its pooled result:

| Model | KADID within | TID within | CID22-A within |
|---|---:|---:|---:|
| Random forest | 0.941915 | 0.926774 | 0.963372 |
| Extra Trees | 0.944410 | 0.929120 | 0.963965 |
| Histogram boosting | 0.948650 | 0.930456 | 0.955198 |
| Shared-ReLU MLP | 0.900954 | 0.908708 | 0.966010 |

That pattern suggests cross-reference calibration is an important remaining
question for the forests. It does not establish that a calibration change will
recover the pooled deficit.

## What the tuning established

The search used three decoded-source-separated folds inside fitting data.
Every transform was fitted within its training complement. Equal population
mass and equal reference mass prevented the larger teacher table from winning
by row count. Tree inputs did not use StandardScaler, clipping or imputation.

Matched range-map teacher ablations increased mean inner-fold CID22-A pooled
SROCC when teacher mass went from 0 to 1: RF 0.866173 → 0.885886, Extra Trees
0.860703 → 0.877670, HGB 0.855659 → 0.874868. Effects on other populations were
mixed. Removing the teacher is not a general improvement.

Matched human-only target-map ablations also matter: replacing the weighted
CDF with a fitting-only range map increased inner-fold CID22-A from 0.850128
to 0.866173 for RF, 0.831396 to 0.860703 for Extra Trees, and 0.778830 to
0.855659 for HGB. These labels span different scales and quality distributions;
the initial percentile mapping was not a harmless preprocessing choice.

The separate human-only stacks tested four peer scores alone and those scores
plus the cached features. Extra Trees with features plus peers reached
0.941421 KADID / 0.938574 TID2013, but 0.819275 CID22-A. Its peer-only version
reached 0.847139 / 0.839597 / 0.853704 on those populations. Features therefore
add useful prediction capacity, while the stack still has a cross-population
trade-off. Those models require the peer metrics at inference; their gains
cannot be credited to a feature-only metric. All stack and peer rows are in
the compact measurements.

Distortion panels retain actual bank distortion codes. For the first RF seed
versus shared ReLU, observed subgroup SROCC differences include +0.219089 for
`kadid_20`, +0.163872 for `kadid_21`, -0.079569 for `kadid_06`, and -0.052106
for CID22-A `cld_avif`. These exploratory subgroup results have small samples
and no multiplicity correction. They identify weaknesses that a pooled score
would conceal, not established distortion-family wins.

## Source uncertainty

The development populations contain 1210/360/84/613 rows from **10/3/1/7
sources**, respectively. They are already-exposed design data, not independent
final tests. The exact decoded-reference audit passed across both development
and inner folds, including reference aliases and population boundaries.

Paired 2000-draw source-cluster bootstrap, first predeclared training seed:

| RF minus shared-ReLU MLP | Point difference | Conditional 95% interval |
|---|---:|---:|
| KADID | +0.030813 | [+0.016188, +0.045100] |
| TID2013 | +0.017870 | [-0.005612, +0.045168] |
| CID22-A | -0.016195 | [-0.050220, +0.016980] |

The KADID gain is supported by this conditional source-resampling check. TID
has only three sources; KonFiG's one source cannot estimate new-source
uncertainty. These intervals are not intervals for the three-seed means above,
do not correct multiple comparisons, and do not erase prior development
exposure. Per-row resampling was not substituted for source resampling.

## Identity and monotonicity

Matched controls change only the monotonic constraint, retaining direct
features, CDF targets, weights, parameters and seeds:

| Model | KADID | TID2013 | KonFiG F | CID22-A |
|---|---:|---:|---:|---:|
| RF unconstrained twin | 0.910853 | 0.905370 | 0.937246 | 0.878262 |
| RF monotone | 0.734416 | 0.809554 | 0.771528 | 0.845837 |
| HGB unconstrained twin | 0.925179 | 0.925008 | 0.960919 | 0.845711 |
| HGB monotone | 0.855797 | 0.888609 | 0.889900 | 0.855567 |

The constraint costs substantial quality in these matched settings. This is
not an exhaustive search over constrained models or proof that monotone trees
cannot work. Unconstrained winners are not monotone: the three RF seeds show
17/19/25 violations among 512 coordinate-increase probes each. Both constrained
families show zero sampled violations.

Subtracting each tree's zero-vector leaf and setting baseline 1 gives exact
zero-vector identity. It does not ensure useful behavior near identity. The
monotone HGB first-seed anchor drops from 1 to 0.182521 at 1e-8 times the
training-median feature ray; the RF winner stays exactly 1 through 1e-4 on its
probe. Those jumps/plateaus matter independently of ranking. The raw-context
models' zero vector is not an actual pixel-identity input contract.

## Numeric evidence, implementation and limits

Training uses sklearn's compiled implementations. Final tree arithmetic runs
through the existing Rust `CorruptionHead::decision_function` API and canonical
ZCTH exporter. All **60 final export checks**, totaling **1,315,488 row
evaluations** across models/probes, agree at **0 ULP**. The measured unanchored
RF/Extra export difference from sklearn is also zero for these 128-tree fits.
Signed, within-reference, scatter, distortion and bootstrap statistics use the
canonical Rust panel owner. Synthetic threshold/anchor controls pass, and
deliberately wrong reference scores fail the unchanged native parity gate.

These are opaque cached-vector experiments. ZCTH's legacy header cannot admit
revision-7 pixel models. Pixel integration, runtime benchmarks, perceptual
identity/near-lossless qualification and dial calibration remain missing. The
study does not change the published zensim crate or weaken existing gates.
No model is promoted. Ranking boosting, full tree-count/depth searches and
constraint-specific tuning were not performed.

Source pins: feature search `7b57f9cbc259`; peer search `8c39499e5916`; matched
controls `2de60dca6c2b`. The archive includes exact source bytes, dependency
versions, native binaries, 177 input-file records, fold memberships, target
maps, fitted models, all predictions/panels and complete logs. Reproduction
commands are the `tree-study*` just recipes; they refuse existing run paths.

Resource observations from the wrappers (not runtime benchmarks or aggregate
process-tree RSS claims):

```text
feature search, 4 concurrent fits, 12G hard cap:
run-heavy: done rc=0 1354s | peak-RSS 0.68GiB | min-avail 40560MiB | peak-load 10.12
peer search, 8 concurrent fits, 8G hard cap:
run-heavy: done rc=0 127s | peak-RSS 0.19GiB | min-avail 40563MiB | peak-load 10.12
matched controls, 6 concurrent fits, 6G hard cap:
run-heavy: done rc=0 121s | peak-RSS 0.19GiB | min-avail 43865MiB | peak-load 5.69
```

Artifacts: `/mnt/v/output/zensim/tree-study-20261106/`. The verified mirror
receipt below records the archive after copying and reading every file back.

Mirror: `/mnt/tower/output/zensim/tree-study-20261106/`. All 6,196 scientific
artifact files (6,450,818,333 bytes) were copied and SHA-256 checked with client
cache-eviction requests; the receipt and checksum manifest were subsequently
copied and checked too. Receipt SHA-256:
`d3bef1d968718b932bab6dae86f29872db4016eb02179e30d60273e9e1554b21`.
Checksum-manifest SHA-256:
`c98912ff5e927496899a5f6d4bd5aaa8a60f4b331d5b1fa483966ce1ac35ecad`.
