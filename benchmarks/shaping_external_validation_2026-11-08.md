# Fixed external shaping comparison — November 8, 2026

Incomplete: the three tree models, additive model and twelve corrected pretrained
models are not yet reported here. The first 18 neural models have both external
datasets reported, with every registered paired seed retained. These are fixed
setting-0 / inner-0-0 comparisons, not tuned finalist results.

Mean signed SROCC across the three paired seeds, computed from the preserved
native per-fit panels:

| Approach | CSIQ866 | LIVE clean10/267 | LIVE full779, overlap |
|---|---:|---:|---:|
| product_joint_identity | 0.953635 | 0.962849 | 0.965796 |
| relu | 0.942569 | 0.962992 | 0.964200 |
| separate_joint_identity | 0.955053 | 0.956459 | 0.965584 |
| silu_diagnostic | 0.893703 | 0.797161 | 0.838427 |
| silu_monotone | 0.939651 | 0.963340 | 0.963649 |
| softplus | 0.940742 | 0.963019 | 0.963212 |

Product and separate identity-initialized shaping improve these fixed-condition
CSIQ point means relative to ReLU. The clean LIVE subset does not show a consistent
gain: product is close to ReLU and separate shaping is lower. This does not
establish universally better per-feature shapes. Ordinary SiLU's unconstrained
diagnostic is weaker in these conditions; this is not its selected best-setting
comparison. Differences between the anchored activations and ReLU are small on
LIVE clean10. No significance claim is made from these means.

The interactive external page retains each fit's signed SROCC, Kendall,
increasing-logistic PLCC, raw PLCC, 95% whole-source SROCC interval and raw
scatter. Intervals use 1,000 draws and seed20261108, with the same source order
for paired seeds. Their descriptive interpretation and small-source limitation
remain visible in the native report.

Registration and exact model bindings precede target parsing; see
[the protocol](../docs/SHAPING_EXTERNAL_VALIDATION.md). Features use the exact
Rev7 extraction build and feature-set identity admitted by training. No external
outcome influences tuning, rescues, selection, calibration or serving promotion.

Artifacts: `/mnt/v/output/zensim/shaping-completion-20261107/external-validation-20261108/`.
The initial neural batch preserved 35 successful dataset jobs, two failed additive
adapter invocations and the earlier successful ReLU/CSIQ pilot. The additive bake
requires its frozen 2,952-term basis; the ordinary dense-feature predictor refused
it. No training was repeated. The dedicated native basis adapter is pending
qualification. Neural batch wrapper accounting, including an SSD admission wait:

```text
run-heavy: done rc=1 244s | peak-RSS 0.26GiB | min-avail 23679MiB | peak-load 6.43
```

The pilot independently completed with:

```text
run-heavy: done rc=0 1s | peak-RSS 0.25GiB | min-avail 24310MiB | peak-load 6.24
```

Browser verification passed for CSIQ866, LIVE clean267 and LIVE full779 scatter
points, with no page errors or mobile overflow. Build and adapter qualification
are separate from training-fit counts. Archive verification remains pending for
this new external batch.
