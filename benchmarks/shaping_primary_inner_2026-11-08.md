# Primary inner-screen results, November 8, 2026

These are completed paired **inner selection** screens, not outer generalization
results. Each neural entry selects among the same four settings using both inner
source folds and three paired seeds (24 fits per family and outer group). The
headline is the equal mean of pooled and within-source signed SROCC for four
human datasets. SSIM2 teacher agreement is excluded. Additive controls have one
deterministic fit per setting and inner fold.

Pending entries remain pending; no seeds or settings were dropped. The corrected
pretrained families use their newly registered threshold coordinates. Every
constrained family uses its uniformly qualified evaluation representation.

| Approach | Outer group 0 inner screen | Outer group 1 inner screen | Outer group 2 inner screen |
|---|---:|---:|---:|
| ReLU | 0.910938 | 0.886364 | 0.897003 |
| Softplus | 0.910232 | 0.886076 | 0.899551 |
| Monotone SiLU | 0.909820 | 0.885259 | 0.899243 |
| Ordinary SiLU (diagnostic) | 0.901238 | 0.870856 | 0.890654 |
| Product joint, identity start | 0.915949 | 0.892362 | 0.909471 |
| Product pretrained, frozen | 0.913847 | Pending | Pending |
| Product pretrained, joint | Pending | Pending | Pending |
| Separate joint, identity start | 0.915177 | 0.884217 | 0.901057 |
| Separate pretrained, frozen | Pending | Pending | Pending |
| Separate pretrained, joint | Pending | Pending | Pending |
| Shaped additive | 0.870036 | 0.850802 | 0.869986 |
| Extra Trees | 0.917060 | 0.889151 | 0.905469 |

Joint product shaping from identity exceeds the matched ReLU screen in all three
outer groups. Separate identity shaping is mixed. Softplus and monotone SiLU
remain close to ReLU; ordinary SiLU is lower in all three groups. These are
descriptive selection results: rescue trials, outer scores, human-only ablations
and uncertainty remain necessary before claiming a generalization improvement.

This supports further work on product shaping. It does not establish a universal
fixed shaper for each feature: curve agreement, feature redundancy, compensation
by downstream weights and source-disjoint transfer are separate questions.

Selection receipts and exact hashes are preserved in the study artifact
`primary-inner-results-20261108.json`. Historical threshold trials and failed
numerical exports remain preserved and are excluded from these corrected entries.
