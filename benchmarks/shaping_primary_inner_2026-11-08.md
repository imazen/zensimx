# Primary inner-screen results, November 8, 2026

These are completed paired **inner selection** screens, not outer generalization
results. Each neural entry selects among the same four settings using both inner
source folds and three paired seeds (24 fits per family and outer group). The
headline is the equal mean of pooled and within-source signed SROCC for four
human datasets. SSIM2 teacher agreement is excluded. Additive controls have one
deterministic fit per setting and inner fold.

All primary entries are complete as of November 9; no seeds or settings were dropped. The corrected
pretrained families use their newly registered threshold coordinates. Every
constrained family uses its uniformly qualified evaluation representation.

| Approach | Outer group 0 inner screen | Outer group 1 inner screen | Outer group 2 inner screen |
|---|---:|---:|---:|
| ReLU | 0.910938 | 0.886364 | 0.897003 |
| Softplus | 0.910232 | 0.886076 | 0.899551 |
| Monotone SiLU | 0.909820 | 0.885259 | 0.899243 |
| Ordinary SiLU (diagnostic) | 0.901238 | 0.870856 | 0.890654 |
| Product joint, identity start | 0.915949 | 0.892362 | 0.909471 |
| Product pretrained, frozen | 0.913847 | 0.892133 | 0.907606 |
| Product pretrained, joint | 0.913668 | 0.888432 | 0.908108 |
| Separate joint, identity start | 0.915177 | 0.884217 | 0.901057 |
| Separate pretrained, frozen | 0.913044 | 0.884333 | 0.897197 |
| Separate pretrained, joint | 0.912063 | 0.882653 | 0.893694 |
| Shaped additive | 0.870036 | 0.850802 | 0.869986 |
| Extra Trees | 0.917060 | 0.889151 | 0.905469 |

Joint product shaping from identity exceeds the matched ReLU screen in all three
outer groups. Separate identity shaping is mixed. Softplus and monotone SiLU
remain close to ReLU; ordinary SiLU is lower in all three groups. These are
descriptive selection results. Completed targeted rescues are below; outer scores,
human-only ablations and uncertainty remain necessary before claiming a
generalization improvement.

This supports further work on product shaping. It does not establish a universal
fixed shaper for each feature: curve agreement, feature redundancy, compensation
by downstream weights and source-disjoint transfer are separate questions.

Selection receipts and exact hashes are preserved in the study artifact
`primary-inner-results-20261109.json` (the earlier November 8 snapshot remains preserved). Historical threshold trials and failed
numerical exports remain preserved and are excluded from these corrected entries.

## Completed targeted rescues, November 9

All 144 registered rescue fits and their evaluations are complete. Each shortlisted
role compared its selected base with half initialization and constant learning
rate, across both inner folds and all three paired seeds. These additional
choices apply to the four shortlisted neural roles; the full primary table above
retains its equal four-setting search.

| Shortlisted role | Group 0 | Group 1 | Group 2 |
|---|---:|---:|---:|
| ReLU | 0.910938 | 0.886364 | 0.897003 |
| Softplus | 0.910232 | 0.886076 | 0.899551 |
| Joint product shaping | 0.915987 | 0.893995 | 0.909471 |
| Selected separate shaping | 0.916268 | 0.886375 | 0.904915 |

The separate role uses identity-start joint shaping in groups 0 and 2, and frozen
pretrained shaping in group 1. Constant learning rate was selected for product
in groups 0 and 1 and separate shaping in group 0. Half initialization was
selected for separate shaping in groups 1 and 2. ReLU and Softplus retained
the base in every group; product retained it in group 2.

Product shaping still exceeds ReLU in all three groups. Rescue tuning helps
separate shaping, but its group-1 composite remains close to ReLU. These are
selection outcomes, not independently tested improvements. Eleven of the 12 selected
role/group combinations retain an investigation trigger in at least one paired
fit. The exposure rule specifically responds to continued loss or human-score
improvement; a surviving-coefficient warning alone does not add another fit.
The registered conditional exposure comparisons must finish before outer
admission; no global-convergence claim follows from a selected score.
