# Completed activation and shaping study — November 13, 2026

Joint product shaping improves the observed held-out human composite over matched ReLU. Extra Trees has the highest point estimate. These results support further use of joint product shaping, but do not establish one reusable per-feature default bank or global optimization convergence.

All 1,461 approved fitting conditions have successful terminal receipts: 1,281 effective conditions plus 180 retained, superseded threshold conditions. The effective primary screen contains 816 conditions; the 180 old threshold outcomes are auxiliary evidence and never replace corrected outcomes. The original-A comparison also has all six models and their existing archives. The [complete twelve-family primary screen and rescue results](shaping_primary_inner_2026-11-08.md) remain separate from these outer comparisons.

All 25 registered source-bootstrap analyses are complete and cold-readback archived. All 48 finalist timing and fresh-process memory cases are complete and archived. The 252 later-stage deliveries and four retained extension deliveries have 256 closed archive records. No completed fit was retrained during the November 13 recovery; its only new numerical training was the already registered 120-epoch fit after recovered pretraining.

## Primary source-disjoint comparison

The human composite equally averages KADID, TID2013, KonFiG F and CID22 human ratings, pooled and within-source Spearman ranks, the three outer folds, and each role’s paired seeds. SSIM2 teacher agreement is separate. Hyperparameters and finalist families were selected on inner partitions. No untouched final holdout was read.

Intervals are 95% percentile intervals from 2,000 population-stratified whole-source bootstrap replicates. They are conditional on the fitted models and observed sources; they do not estimate retraining-seed uncertainty. These are means of fold statistics, not pooled out-of-fold correlations.

| Approach | Human composite | Difference from ReLU | Paired difference interval | SSIM2 teacher composite |
| --- | ---: | ---: | --- | ---: |
| ReLU | 0.889729 | +0.000000 | [+0.000000, +0.000000] | 0.990369 |
| Softplus | 0.891277 | +0.001547 | [-0.000236, +0.003353] | 0.990541 |
| Joint product shaping | 0.901866 | +0.012136 | [+0.008196, +0.017353] | 0.991333 |
| Selected separate shaping | 0.896845 | +0.007116 | [+0.002132, +0.012010] | 0.990667 |
| Extra Trees | 0.910558 | +0.020829 | [+0.010441, +0.030322] | 0.984080 |
| Shaped additive | 0.865817 | -0.023912 | [-0.035614, -0.010967] | 0.940369 |

Product shaping selects the identity-initialized joint family in all three outer folds. Separate shaping selects joint identity initialization in folds 0 and 2 and pretrained frozen shaping in fold 1. Softplus supplies the analytic finalist in all three folds. Ordinary and monotone SiLU received the registered screen but were not selected as the analytic finalists.

Ten of the twelve shortlisted neural family/fold groups retain registered optimization triggers. All three product groups are marked inconclusive after registered optimization. Positive observed-model differences are therefore not a proof that an architecture is superior or fully converged. In particular, a surviving output coefficient is not evidence of using the nominal hidden capacity.

| Approach | KADID | TID2013 | KonFiG F | CID22 human |
| --- | ---: | ---: | ---: | ---: |
| ReLU | 0.898939 | 0.878663 | 0.864485 | 0.916831 |
| Softplus | 0.899402 | 0.880481 | 0.867055 | 0.918168 |
| Joint product shaping | 0.905981 | 0.890357 | 0.889919 | 0.921206 |
| Selected separate shaping | 0.902136 | 0.889517 | 0.874571 | 0.921158 |
| Extra Trees | 0.933147 | 0.913398 | 0.883919 | 0.911771 |
| Shaped additive | 0.875772 | 0.847310 | 0.857442 | 0.882745 |

## Human-only ablation, source fractions and shuffled labels

Each condition below is compared with its own role’s full-source, mixed human-plus-SSIM2 model. All original approved human distortion types remain admitted.

| Approach | 25% sources | 50% sources | Population shuffle | Human only | Human-only difference interval |
| --- | ---: | ---: | ---: | ---: | --- |
| ReLU | 0.881364 | 0.883078 | 0.580033 | 0.885193 | [-0.012271, +0.006127] |
| Softplus | 0.879311 | 0.885489 | 0.565219 | 0.890630 | [-0.008042, +0.008889] |
| Joint product shaping | 0.886417 | 0.889002 | 0.509906 | 0.904944 | [-0.003465, +0.011310] |
| Selected separate shaping | 0.876898 | 0.878750 | 0.498620 | 0.887118 | [-0.020533, +0.002080] |
| Extra Trees | 0.869495 | 0.888083 | 0.263794 | 0.904031 | [-0.009959, -0.002002] |
| Shaped additive | 0.842381 | 0.845886 | 0.712310 | 0.859637 | [-0.010663, -0.000920] |

Human-only product shaping reaches 0.904944; its +0.003078 difference from mixed supervision has an interval crossing zero. Extra Trees and shaped additive have lower human-only scores with negative paired intervals. The neural human-only intervals all cross zero. This does not establish a universal benefit or harm from adding SSIM2 supervision.

Every role scores lower at both reduced source fractions, with negative paired difference intervals. Extra Trees has the largest point decline at 25% sources, from 0.910558 to 0.869495. The reduced source-order coverage and fixed optimization budgets limit causal interpretation of learning-curve differences.

Shuffled-label models retain substantial quality correlation: 0.580033 for ReLU and 0.712310 for the constrained additive model. Structural quality priors and population effects can retain correlation under this control. It is not a zero-correlation null or proof that all shortcuts and memorization have been excluded.

## Scoring cost and storage

Timings were measured on r3800x through zenbench with the registered cpuset and paired-round gates. The table shows the median across a role’s trained models of their median 40-row batch mean per row. It excludes feature extraction, model loading and end-to-end image latency. Pruning differs between trained models, so these are costs of the actual finalists, not equal-width architecture microbenchmarks.

Fresh-process RSS uses GNU time, including loading, the native runtime and qualification inputs. It is not model heap alone.

| Approach | Models | Median serialized bytes | Batch mean per row, µs | Median fresh-process peak RSS, KiB |
| --- | ---: | ---: | ---: | ---: |
| ReLU | 9 | 184,900 | 26.32 | 7,788 |
| Softplus | 9 | 167,631 | 24.18 | 8,196 |
| Joint product shaping | 9 | 125,970 | 28.39 | 8,004 |
| Selected separate shaping | 9 | 94,506 | 18.07 | 8,028 |
| Extra Trees | 9 | 12,054,962 | 23.31 | 28,528 |
| Shaped additive | 3 | 104,583 | 11.58 | 7,244 |

Extra Trees has a 12,054,962-byte median serialized model and comparable measured batch scoring time to the neural finalists, with a larger fresh-process footprint. Storage size alone does not describe its measured latency.

The first timing preparation incorrectly used a dataframe-parsed cached decimal as a bit oracle. The original native TSV text proves the exact score. Corrected preparation reads that text directly and retains bit-exact checks. The refused timing and its original workload remain preserved; no quality evaluation or model fit was replayed for this correction.

## Original A architecture and full A recipe

The six earlier A-data models were recovered from the existing study record, not retrained. All three unshaped controls match their original base-fit and continuation f32 checkpoints after the originally specified repro-metadata removal. Comparison uses fixed continuation epoch 59 and forbids selection.

| Dataset / population | Unshaped mean | Joint product mean | Paired mean difference |
| --- | ---: | ---: | ---: |
| csiq / all | 0.945941 | 0.945895 | -0.000046 |
| live / all | 0.967995 | 0.970132 | +0.002137 |
| live / clean10 | 0.967127 | 0.964314 | -0.002813 |

All three paired LIVE deltas are positive; all three clean10 deltas are negative; CSIQ is mixed. No paired-delta confidence interval was computed in that summary. These exposed development comparisons do not establish a general improvement over the serving A model. Their six original model archives, shared inputs, recipe parity, replay guide and summary remain available.

## Reusable shaping and limits

Joint product shaping is the clearest neural candidate in this study. A fixed per-feature default bank is still unproven: the frozen-bank trials use each seed’s own learned prefix, rather than one shared bank transferred across seeds. The earlier matched inner-tail analysis also found mixed tail gains and higher product-shaping seed spread. See [tail diagnostics](shaping_tail_diagnostics_2026-11-09.md).

Identity and monotonicity claims retain their original native gates; ordinary SiLU remains an unconstrained diagnostic. Observed tree performance does not confer neural monotonicity guarantees. Nothing here promotes a serving model or changes the frozen metric profiles.

The owner-approved reduction retains one source order and one population shuffle, omits row-fraction curves and full-data retuning/refits, and restricts optimizer rescue coverage. Completion of this registered scope is not completion of those omitted experiments.

## Reproduction and retained evidence

- Core ledger: `shaping-completion-20261107/study-resume-20261111/final-recovery-20261113/CORE_COMPLETION_AUDIT.json`.
- Source uncertainty: `study-resume-20261111/bootstrap-final/`, mirrored under `bootstrap-autonomous-20261111/`.
- Cost and memory: `study-resume-20261111/cost-final-corrected-20261113/`, mirrored under `cost-autonomous-corrected-20261113/{prepared,vector}`.
- Delivery closure: `study-resume-20261111/archive-once/STATUS.json`, with 256 delivery closures spanning the durable `later-*` run, panel, bake and control roots.
- A-data evidence: `zensim-training/a-shaping-20261108/{shared,models,comparison-summary-v1,replay-guide-v1}` in the durable archive tree.
- Coordinator recovery and a fresh four-worker snapshot-retirement audit are preserved separately from the historical failed attempts. Original worker trees were retained when coordination moved to r3500.

The scalar report at `shaping-study-final-20261113/` includes primary and matched tables, all five populations, three selectable paired seeds, 80 cached scatter series and saved native population/source panels. Scatter axes use within-dataset, within-fold percentile ranks and never refit a calibrator. Its JSON remains usable without retraining or rebuilding native code.

`just shaping-final-report STUDY DISPATCH NEW_OUTPUT A_DATA_SUMMARY` reproduces the view from admitted cached outputs. The real-data build reported `run-heavy: done rc=0 44s | peak-RSS 0.39GiB | min-avail 13015MiB | peak-load 0.53`. That footprint belongs to report generation, not inference. JavaScript syntax and network delivery of the generated HTML and JSON were checked.
